//! The gated MLP on the CPU: one token at a time for decode, all rows of a
//! window at once for prefill.

use super::GatedMlpLayerWeights;
use crate::ops::blas::matvec_forward;
use crate::ops::fast_math::fast_exp_scalar;
use crate::ops::norms::rms_norm_weighted;

/// `v * silu(g)`, the gate the reference applies (`F.silu`), with the same
/// exponential every other CPU gate here uses.
#[inline(always)]
fn gated(v: f32, g: f32) -> f32 {
    v * (g / (1.0 + fast_exp_scalar(-g)))
}

/// Per-token buffers of one MLP, sized once.
pub struct GatedMlpStepScratch {
    norm: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    down: Vec<f32>,
}

impl GatedMlpStepScratch {
    /// Buffers for a `d_model` by `hidden` MLP; zero-sized when the model
    /// has no MLP.
    pub fn new(d_model: usize, hidden: usize) -> Self {
        let d = if hidden == 0 { 0 } else { d_model };
        Self {
            norm: vec![0.0; d],
            up: vec![0.0; 2 * hidden],
            act: vec![0.0; hidden],
            down: vec![0.0; d],
        }
    }
}

/// One token through one MLP branch: `temporal += mlp(norm2(temporal))`.
pub fn gated_mlp_step(
    temporal: &mut [f32],
    w: &GatedMlpLayerWeights,
    s: &mut GatedMlpStepScratch,
    eps: f32,
) {
    let d = w.norm_weight.len();
    let h = w.hidden();
    rms_norm_weighted(&mut s.norm[..d], &temporal[..d], &w.norm_weight, eps);
    matvec_forward(&mut s.up[..2 * h], &s.norm[..d], &w.fc1_w, None, d, 2 * h);
    let (val, gate) = s.up[..2 * h].split_at(h);
    for ((a, &v), &g) in s.act[..h].iter_mut().zip(val).zip(gate) {
        *a = gated(v, g);
    }
    matvec_forward(&mut s.down[..d], &s.act[..h], &w.fc2_w, None, h, d);
    for (t, &r) in temporal[..d].iter_mut().zip(&s.down[..d]) {
        *t += r;
    }
}

/// Window buffers of one MLP for `rows` tokens, grown on demand.
#[derive(Default)]
pub struct GatedMlpRowsScratch {
    norm: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    down: Vec<f32>,
}

/// A row GEMM `out[rows, n] = x[rows, k] @ w[k, n]` with no bias, as the
/// prefill's Single and Parallel modes supply it.
pub type RowsGemm = fn(&mut [f32], &[f32], &[f32], Option<&[f32]>, usize, usize, usize);

/// `rows` tokens through one MLP branch: each row of `temporal` gets
/// `mlp(norm2(row))` added. `norm_rows` writes the normalized rows.
pub fn gated_mlp_rows(
    temporal: &mut [f32],
    rows: usize,
    w: &GatedMlpLayerWeights,
    s: &mut GatedMlpRowsScratch,
    gemm: RowsGemm,
    norm_rows: impl Fn(&mut [f32], &[f32]),
) {
    let d = w.norm_weight.len();
    let h = w.hidden();
    s.norm.resize(rows * d, 0.0);
    s.up.resize(rows * 2 * h, 0.0);
    s.act.resize(rows * h, 0.0);
    s.down.resize(rows * d, 0.0);
    norm_rows(&mut s.norm[..rows * d], &temporal[..rows * d]);
    gemm(&mut s.up, &s.norm, &w.fc1_w, None, rows, d, 2 * h);
    for (act_row, up_row) in s.act.chunks_exact_mut(h).zip(s.up.chunks_exact(2 * h)) {
        let (val, gate) = up_row.split_at(h);
        for ((a, &v), &g) in act_row.iter_mut().zip(val).zip(gate) {
            *a = gated(v, g);
        }
    }
    gemm(&mut s.down, &s.act, &w.fc2_w, None, rows, h, d);
    for (t, &r) in temporal[..rows * d].iter_mut().zip(&s.down[..rows * d]) {
        *t += r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::blas::sgemm_forward;

    fn draw(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s & 0xFFFF) as f32 / 65536.0 - 0.5
            })
            .collect()
    }

    /// The step and the rows form agree with a plain f64 transcription of
    /// the reference block branch.
    #[test]
    fn gated_mlp_matches_the_reference_branch() {
        let (d, h, rows) = (24usize, 128usize, 5usize);
        let w = GatedMlpLayerWeights {
            norm_weight: draw(d, 1).iter().map(|v| 1.0 + v).collect(),
            fc1_w: draw(d * 2 * h, 2),
            fc2_w: draw(h * d, 3),
        };
        let x = draw(rows * d, 4);
        let eps = 1e-5f32;
        let mut want = vec![0f64; rows * d];
        for r in 0..rows {
            let row = &x[r * d..(r + 1) * d];
            let ms = row
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                / d as f64;
            let inv = 1.0 / (ms + f64::from(eps)).sqrt();
            let n: Vec<f64> = (0..d)
                .map(|i| f64::from(row[i]) * inv * f64::from(w.norm_weight[i]))
                .collect();
            let up: Vec<f64> = (0..2 * h)
                .map(|j| {
                    (0..d)
                        .map(|i| n[i] * f64::from(w.fc1_w[i * 2 * h + j]))
                        .sum()
                })
                .collect();
            let act: Vec<f64> = (0..h)
                .map(|j| up[j] * (up[h + j] / (1.0 + (-up[h + j]).exp())))
                .collect();
            for o in 0..d {
                let down: f64 = (0..h).map(|j| act[j] * f64::from(w.fc2_w[j * d + o])).sum();
                want[r * d + o] = f64::from(row[o]) + down;
            }
        }
        let mut step_out = x.clone();
        let mut ss = GatedMlpStepScratch::new(d, h);
        for r in 0..rows {
            gated_mlp_step(&mut step_out[r * d..(r + 1) * d], &w, &mut ss, eps);
        }
        let mut rows_out = x.clone();
        let mut rs = GatedMlpRowsScratch::default();
        gated_mlp_rows(
            &mut rows_out,
            rows,
            &w,
            &mut rs,
            sgemm_forward,
            |dst, src| {
                for (o, i) in dst.chunks_exact_mut(d).zip(src.chunks_exact(d)) {
                    rms_norm_weighted(o, i, &w.norm_weight, eps);
                }
            },
        );
        for (label, got) in [("step", &step_out), ("rows", &rows_out)] {
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (f64::from(g) - w).abs() <= 1e-5 * w.abs().max(1.0),
                    "{label}[{i}]: {g} vs {w}"
                );
            }
        }
    }
}
