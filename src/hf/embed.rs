//! Embedding lookup and lm_head projection for LM inference.

use crate::ops::blas::{matvec_forward, sgemm_forward};

/// Look up one token's embedding vector.
///
/// Returns a slice `[d_model]` into the embedding table.
/// Panics with a descriptive message if `token_id >= vocab_size`.
#[inline]
pub fn embed_lookup(embed: &[f32], token_id: u32, d_model: usize, vocab_size: usize) -> &[f32] {
    let id = token_id as usize;
    assert!(
        id < vocab_size,
        "token_id {id} out of range (vocab_size={vocab_size})"
    );
    &embed[id * d_model..(id + 1) * d_model]
}

/// Compute logits = embed @ hidden (tied lm_head).
///
/// `embed`: `[vocab_size_padded * d_model]` row-major — each row is one token's embedding.
/// `hidden`: `[d_model]`.
/// `logits`: `[vocab_size]` output (clamped to real vocab, not padded).
///
/// Mathematically: `logits[v] = dot(embed[v, :], hidden)` = `embed[V,D] @ hidden[D,1]`.
/// This is `sgemm_forward(batch=V, n_in=D, n_out=1)` with embed as "X" and hidden as "W".
/// Compute logits with tied weights. `scratch` must be `[vocab_size_padded]` (pre-allocated, reused).
pub fn lm_head_logits(
    logits: &mut [f32],
    hidden: &[f32],
    embed: &[f32],
    vocab_size: usize,
    vocab_size_padded: usize,
    d_model: usize,
    scratch: &mut [f32],
) {
    use rayon::prelude::*;
    debug_assert!(scratch.len() >= vocab_size_padded);
    // Every logit is one row's dot product, its terms added in ascending k
    // whatever rows share a call, so blocks of rows on the pool give the
    // single call's bits; the block is a fixed constant, never derived from
    // the thread count. One token's head reads the whole embedding, the
    // largest single read of a decode step.
    const ROWS_PER_TASK: usize = 2048;
    scratch[..vocab_size_padded]
        .par_chunks_mut(ROWS_PER_TASK)
        .enumerate()
        .for_each(|(i, out)| {
            let r0 = i * ROWS_PER_TASK;
            let rows = out.len();
            sgemm_forward(
                out,
                &embed[r0 * d_model..(r0 + rows) * d_model],
                hidden,
                None,
                rows,
                d_model,
                1,
            );
        });
    logits[..vocab_size].copy_from_slice(&scratch[..vocab_size]);
}

/// Compute logits using a separate lm_head weight matrix (untied weights).
///
/// `lm_head_w`: `[vocab_size * d_model]` row-major (PyTorch layout: [out, in]).
/// After transpose during HF loading, this becomes `[d_model, vocab_size]`.
/// So we use matvec: `logits[V] = hidden[D] @ W[D,V]`.
pub fn lm_head_logits_untied(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head_w: &[f32],
    vocab_size: usize,
    d_model: usize,
) {
    matvec_forward(logits, hidden, lm_head_w, None, d_model, vocab_size);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocked head gives one serial GEMM's bits.
    #[test]
    fn blocked_head_matches_one_gemm_bitwise() {
        let (vocab, padded, d) = (5003usize, 5008usize, 96usize);
        let mut s = 0x9E37_79B9u32;
        let mut draw = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 32768.0 - 1.0
        };
        let embed: Vec<f32> = (0..padded * d).map(|_| draw()).collect();
        let hidden: Vec<f32> = (0..d).map(|_| draw()).collect();
        let mut want = vec![0f32; padded];
        sgemm_forward(&mut want, &embed, &hidden, None, padded, d, 1);
        let mut logits = vec![0f32; vocab];
        let mut scratch = vec![0f32; padded];
        lm_head_logits(&mut logits, &hidden, &embed, vocab, padded, d, &mut scratch);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&logits), bits(&want[..vocab]));
    }
}
