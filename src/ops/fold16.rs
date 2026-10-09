//! Sums with one result on every machine.
//!
//! A reduction over a SIMD axis would otherwise take the instruction set's
//! lane count and horizontal-add order into its bits. Here every sum runs
//! over sixteen logical lanes, element i into lane i mod 16 with a fused
//! multiply-add in ascending i, and the lanes then fold in a fixed tree of
//! same-lane adds: AVX-512 holds the lanes in one register, AVX2 in two,
//! NEON in four and the scalar loop in sixteen, and all of them agree.

/// The number of logical lanes.
pub const FOLD16: usize = 16;

/// Fold sixteen lane sums: halves, then quarters, then pairs.
#[inline(always)]
pub fn fold16_tree(l: &[f32; FOLD16]) -> f32 {
    let mut t8 = [0f32; 8];
    for (j, t) in t8.iter_mut().enumerate() {
        *t = l[j] + l[j + 8];
    }
    let mut t4 = [0f32; 4];
    for (j, t) in t4.iter_mut().enumerate() {
        *t = t8[j] + t8[j + 4];
    }
    let t2 = [t4[0] + t4[2], t4[1] + t4[3]];
    t2[0] + t2[1]
}

/// sum_i a[i] * b[i] in the sixteen-lane order.
#[inline]
pub fn dot_fold16(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; FOLD16];
    let (a_full, a_tail) = a.as_chunks::<FOLD16>();
    let (b_full, b_tail) = b.as_chunks::<FOLD16>();
    for (ca, cb) in a_full.iter().zip(b_full) {
        for j in 0..FOLD16 {
            acc[j] = ca[j].mul_add(cb[j], acc[j]);
        }
    }
    for (j, (&x, &y)) in a_tail.iter().zip(b_tail).enumerate() {
        acc[j] = x.mul_add(y, acc[j]);
    }
    fold16_tree(&acc)
}

/// sum_i x[i]^2 in the sixteen-lane order.
#[inline]
pub fn sum_sq_fold16(x: &[f32]) -> f32 {
    dot_fold16(x, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition, spelled out element by element.
    fn reference(a: &[f32], b: &[f32]) -> f32 {
        let mut lanes = [0f32; FOLD16];
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            lanes[i % FOLD16] = x.mul_add(y, lanes[i % FOLD16]);
        }
        fold16_tree(&lanes)
    }

    #[test]
    fn matches_the_lane_definition_at_every_length() {
        let a: Vec<f32> = (0..200)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let b: Vec<f32> = (0..200)
            .map(|i| ((i * 53 % 89) as f32 - 44.0) / 3.0)
            .collect();
        for n in 0..=200 {
            assert_eq!(
                dot_fold16(&a[..n], &b[..n]).to_bits(),
                reference(&a[..n], &b[..n]).to_bits(),
                "n={n}"
            );
        }
    }
}
