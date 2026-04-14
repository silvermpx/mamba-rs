//! LLM token sampling: temperature, top-k, top-p, min-p, repetition penalty.

/// Sampling parameters for text generation.
#[derive(Clone, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 256,
            eos_token_ids: vec![],
            seed: 42,
        }
    }
}

/// xoshiro256++ PRNG — passes BigCrush, supports jump() for parallel streams.
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        for slot in &mut s {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = z ^ (z >> 31);
        }
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Apply repetition penalty (CTRL paper, Keskar et al. 2019).
///
/// `penalty` must be strictly positive and not 1.0. Values ≤ 0 or NaN are
/// ignored (treated as "disabled") to avoid producing `+Inf` logits via
/// division by zero.
pub fn apply_repetition_penalty(logits: &mut [f32], seen_tokens: &[u32], penalty: f32) {
    if !(penalty > 0.0) || penalty == 1.0 {
        return;
    }
    for &tok in seen_tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Apply temperature scaling.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature != 1.0 && temperature > 0.0 {
        let inv_t = 1.0 / temperature;
        for l in logits.iter_mut() {
            *l *= inv_t;
        }
    }
}

/// Apply top-k filtering: set all logits outside the top-k to -inf.
/// Returns the number of surviving tokens (min(k, vocab_size)).
pub fn apply_top_k(logits: &mut [f32], k: usize) -> usize {
    if k == 0 || k >= logits.len() {
        return logits.len();
    }
    let n = logits.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.select_nth_unstable_by(k, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    for &idx in &indices[k..] {
        logits[idx] = f32::NEG_INFINITY;
    }
    k
}

/// Softmax in-place (numerically stable).
pub fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - max).exp();
        sum += *l;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for l in logits.iter_mut() {
            *l *= inv_sum;
        }
    }
}

/// Apply top-p (nucleus) filtering on probabilities (must be after softmax).
///
/// `p ≥ 1.0` or `p ≤ 0.0` (or NaN) disables filtering — the ≤ 0 case would
/// otherwise silently collapse to a single-token distribution (whichever
/// token sorted first by probability) which surprises callers expecting
/// "disabled" behavior matching `top_k = 0`.
pub fn apply_top_p(probs: &mut [f32], p: f32) {
    if !(p > 0.0) || p >= 1.0 {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cumsum = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff = i + 1;
            break;
        }
    }
    for &(idx, _) in &indexed[cutoff..] {
        probs[idx] = 0.0;
    }
    renormalize(probs);
}

/// Apply min-p filtering on probabilities (must be after softmax).
pub fn apply_min_p(probs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
    renormalize(probs);
}

fn renormalize(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 && sum != 1.0 {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
}

/// Weighted random sample from a probability distribution.
pub fn weighted_sample(probs: &[f32], rng: &mut Xoshiro256PlusPlus) -> u32 {
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Full sampling pipeline: repetition penalty → temperature → top-k → softmax → top-p → min-p → sample.
pub fn sample_token(
    logits: &mut [f32],
    params: &SampleParams,
    seen_tokens: &[u32],
    rng: &mut Xoshiro256PlusPlus,
) -> u32 {
    apply_repetition_penalty(logits, seen_tokens, params.repetition_penalty);

    if params.temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    apply_temperature(logits, params.temperature);
    apply_top_k(logits, params.top_k);
    softmax_inplace(logits);
    apply_top_p(logits, params.top_p);
    apply_min_p(logits, params.min_p);
    weighted_sample(logits, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_argmax() {
        let mut logits = vec![1.0, 5.0, 2.0, 0.5];
        let params = SampleParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = Xoshiro256PlusPlus::new(42);
        assert_eq!(sample_token(&mut logits, &params, &[], &mut rng), 1);
    }

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![2.0, 1.0, 0.0, -1.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_topk_k1_is_argmax() {
        let mut rng = Xoshiro256PlusPlus::new(42);
        for _ in 0..100 {
            let mut logits = vec![1.0, 5.0, 2.0, 0.5];
            let params = SampleParams {
                temperature: 1.0,
                top_k: 1,
                ..Default::default()
            };
            assert_eq!(sample_token(&mut logits, &params, &[], &mut rng), 1);
        }
    }

    #[test]
    fn test_topk_k2_excludes_rest() {
        let mut rng = Xoshiro256PlusPlus::new(42);
        for _ in 0..1000 {
            let mut logits = vec![1.0, 5.0, 3.0, 0.1];
            let params = SampleParams {
                temperature: 1.0,
                top_k: 2,
                ..Default::default()
            };
            let tok = sample_token(&mut logits, &params, &[], &mut rng);
            assert!(tok == 1 || tok == 2, "got token {tok}, expected 1 or 2");
        }
    }

    #[test]
    fn test_topp_filters_tail() {
        let mut probs = vec![0.6, 0.3, 0.05, 0.05];
        apply_top_p(&mut probs, 0.85);
        assert!(probs[2] == 0.0 || probs[3] == 0.0);
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
    }

    #[test]
    fn test_minp_filters_by_fraction() {
        let mut probs = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        apply_min_p(&mut probs, 0.2);
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        assert_eq!(probs[3], 0.0);
        assert_eq!(probs[4], 0.0);
    }

    #[test]
    fn test_repetition_penalty_positive_divides() {
        let mut logits = vec![2.0, -1.0, 0.0];
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_multiplies() {
        let mut logits = vec![2.0, -1.0, 0.0];
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        assert!((logits[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let params = SampleParams {
            temperature: 0.8,
            ..Default::default()
        };
        let mut tokens1 = vec![];
        let mut tokens2 = vec![];
        let mut rng1 = Xoshiro256PlusPlus::new(42);
        let mut rng2 = Xoshiro256PlusPlus::new(42);
        for _ in 0..20 {
            let mut l1 = vec![1.0, 2.0, 1.5, 0.8, 3.0, 0.1, 0.5, 2.5];
            let mut l2 = l1.clone();
            tokens1.push(sample_token(&mut l1, &params, &[], &mut rng1));
            tokens2.push(sample_token(&mut l2, &params, &[], &mut rng2));
        }
        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_different_seed_different_output() {
        let params = SampleParams {
            temperature: 0.8,
            ..Default::default()
        };
        let mut tokens1 = vec![];
        let mut tokens2 = vec![];
        let mut rng1 = Xoshiro256PlusPlus::new(42);
        let mut rng2 = Xoshiro256PlusPlus::new(99);
        for _ in 0..50 {
            let mut l1 = vec![1.0, 2.0, 1.5, 0.8, 3.0, 0.1, 0.5, 2.5];
            let mut l2 = l1.clone();
            tokens1.push(sample_token(&mut l1, &params, &[], &mut rng1));
            tokens2.push(sample_token(&mut l2, &params, &[], &mut rng2));
        }
        assert_ne!(tokens1, tokens2);
    }
}
