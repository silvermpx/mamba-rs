//! The gated MLP on the CPU: one token at a time for decode, all rows of a
//! window at once for prefill.
//!
//! In a reduced precision the branch rounds where the GPU's stores it: the
//! normalized rows, both projections' outputs and the gated activation in
//! the forward; the residual gradient's copy for the GEMMs, both input
//! gradients and the gate's gradient in the backward. The norm's
//! statistics, the residual and the weight gradients stay f32.

use super::GatedMlpLayerWeights;
use crate::ops::fast_math::fast_exp_scalar;
use crate::ops::lane::LaneGemm;
use crate::ops::norms::{layer_norm_row, layer_norm_row_backward, rms_norm_weighted};
use crate::ops::packed::PackedLinear;
use crate::ops::rounding::store;
use crate::precision::WeightDtype;

/// `v * silu(g)`, the gate the reference applies (`F.silu`), with the same
/// exponential every other CPU gate here uses.
#[inline(always)]
fn gated(v: f32, g: f32) -> f32 {
    v * (g / (1.0 + fast_exp_scalar(-g)))
}

/// The pre-MLP norm of one row: RMSNorm ([`rms_norm_weighted`]), or
/// LayerNorm for a model whose norms carry a bias.
#[inline]
pub fn mlp_norm_row(out: &mut [f32], x: &[f32], w: &GatedMlpLayerWeights, eps: f32) {
    if w.norm_bias.is_empty() {
        rms_norm_weighted(out, x, &w.norm_weight, eps);
    } else {
        layer_norm_row(out, x, (&w.norm_weight, &w.norm_bias), eps);
    }
}

/// Per-token buffers of one MLP, sized once.
pub struct GatedMlpStepScratch {
    norm: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    down: Vec<f32>,
    /// The precision the branch computes in.
    pub(crate) dtype: WeightDtype,
    gemm: LaneGemm,
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
            dtype: WeightDtype::F32,
            gemm: LaneGemm::default(),
        }
    }
}

/// One MLP's projections packed for decode (see [`PackedLinear`]).
#[derive(Clone, Debug)]
pub struct GatedMlpPacked {
    pub fc1: PackedLinear,
    pub fc2: PackedLinear,
}

impl GatedMlpPacked {
    /// Pack both projections of `w`.
    pub fn pack(w: &GatedMlpLayerWeights) -> Self {
        let (d, h) = (w.norm_weight.len(), w.hidden());
        Self {
            fc1: PackedLinear::pack(&w.fc1_w, d, 2 * h),
            fc2: PackedLinear::pack(&w.fc2_w, h, d),
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
    mlp_step(temporal, (w, None), s, eps);
}

/// [`gated_mlp_step`] with the projections taken from `p` (packed from `w`).
pub fn gated_mlp_step_packed(
    temporal: &mut [f32],
    (w, p): (&GatedMlpLayerWeights, &GatedMlpPacked),
    s: &mut GatedMlpStepScratch,
    eps: f32,
) {
    mlp_step(temporal, (w, Some(p)), s, eps);
}

fn mlp_step(
    temporal: &mut [f32],
    (w, p): (&GatedMlpLayerWeights, Option<&GatedMlpPacked>),
    s: &mut GatedMlpStepScratch,
    eps: f32,
) {
    let d = w.norm_weight.len();
    let h = w.hidden();
    let dtype = s.dtype;
    mlp_norm_row(&mut s.norm[..d], &temporal[..d], w, eps);
    store(dtype, &mut s.norm[..d]);
    s.gemm.row(
        dtype,
        (&mut s.up[..2 * h], &s.norm[..d]),
        (&w.fc1_w, p.map(|p| &p.fc1)),
        None,
    );
    let (val, gate) = s.up[..2 * h].split_at(h);
    for ((a, &v), &g) in s.act[..h].iter_mut().zip(val).zip(gate) {
        *a = gated(v, g);
    }
    store(dtype, &mut s.act[..h]);
    s.gemm.row(
        dtype,
        (&mut s.down[..d], &s.act[..h]),
        (&w.fc2_w, p.map(|p| &p.fc2)),
        None,
    );
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
    /// The precision the branch computes in.
    pub(crate) dtype: WeightDtype,
    gemm: LaneGemm,
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
    let lane = (gemm, s.dtype);
    norm_rows(&mut s.norm[..rows * d], &temporal[..rows * d]);
    store(s.dtype, &mut s.norm[..rows * d]);
    s.gemm.forward(
        lane,
        (&mut s.up, &s.norm),
        (&w.fc1_w, None),
        (rows, d, 2 * h),
    );
    for (act_row, up_row) in s.act.chunks_exact_mut(h).zip(s.up.chunks_exact(2 * h)) {
        let (val, gate) = up_row.split_at(h);
        for ((a, &v), &g) in act_row.iter_mut().zip(val).zip(gate) {
            *a = gated(v, g);
        }
    }
    store(s.dtype, &mut s.act[..rows * h]);
    s.gemm
        .forward(lane, (&mut s.down, &s.act), (&w.fc2_w, None), (rows, h, d));
    for (t, &r) in temporal[..rows * d].iter_mut().zip(&s.down[..rows * d]) {
        *t += r;
    }
}

/// Buffers for [`gated_mlp_rows_backward`], grown on demand.
#[derive(Default)]
pub struct GatedMlpBackwardScratch {
    inv: Vec<f32>,
    /// A LayerNorm's std and mean per row.
    ln_stats: Vec<[f32; 2]>,
    norm: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    d_act: Vec<f32>,
    d_up: Vec<f32>,
    d_norm: Vec<f32>,
    /// The precision the branch computes in.
    pub(crate) dtype: WeightDtype,
    gemm: LaneGemm,
}

/// The backward of one MLP branch over `rows` tokens. `x_in` are the rows
/// before the branch; `d_temporal` holds d loss / d (the rows after it) and
/// leaves holding d loss / d (the rows before it), the residual included.
/// The weight gradients accumulate into `g`. The norm is
/// [`rms_norm_weighted`], `(x * inv_rms) * w`, as both forward paths apply,
/// or [`layer_norm_row`] for a model whose norms carry a bias.
pub fn gated_mlp_rows_backward(
    d_temporal: &mut [f32],
    x_in: &[f32],
    rows: usize,
    (w, g): (&GatedMlpLayerWeights, &mut GatedMlpLayerWeights),
    s: &mut GatedMlpBackwardScratch,
    eps: f32,
) {
    use crate::ops::blas::sgemm_forward;
    let d = w.norm_weight.len();
    let h = w.hidden();
    let dtype = s.dtype;
    s.inv.resize(rows, 0.0);
    s.norm.resize(rows * d, 0.0);
    s.up.resize(rows * 2 * h, 0.0);
    s.act.resize(rows * h, 0.0);
    s.d_act.resize(rows * h, 0.0);
    s.d_up.resize(rows * 2 * h, 0.0);
    s.d_norm.resize(rows * d, 0.0);

    // The forward again: norm, up projection, gate.
    let layer_norm = !w.norm_bias.is_empty();
    if layer_norm {
        s.ln_stats.resize(rows, [0.0; 2]);
        for ((st, n), x) in s
            .ln_stats
            .iter_mut()
            .zip(s.norm.chunks_exact_mut(d))
            .zip(x_in.chunks_exact(d))
        {
            *st = layer_norm_row(n, x, (&w.norm_weight, &w.norm_bias), eps);
        }
    } else {
        for ((inv, n), x) in s
            .inv
            .iter_mut()
            .zip(s.norm.chunks_exact_mut(d))
            .zip(x_in.chunks_exact(d))
        {
            *inv = rms_norm_weighted(n, x, &w.norm_weight, eps);
        }
    }
    store(dtype, &mut s.norm[..rows * d]);
    s.gemm.forward(
        (sgemm_forward as RowsGemm, dtype),
        (&mut s.up, &s.norm),
        (&w.fc1_w, None),
        (rows, d, 2 * h),
    );
    for (act_row, up_row) in s.act.chunks_exact_mut(h).zip(s.up.chunks_exact(2 * h)) {
        let (val, gate) = up_row.split_at(h);
        for ((a, &v), &gt) in act_row.iter_mut().zip(val).zip(gate) {
            *a = gated(v, gt);
        }
    }
    store(dtype, &mut s.act[..rows * h]);

    // Down projection, gate, up projection.
    s.gemm.backward(
        dtype,
        (&mut s.d_act, &mut g.fc2_w, None),
        (&d_temporal[..rows * d], &s.act, &w.fc2_w),
        (rows, h, d),
    );
    for ((d_up_row, up_row), d_act_row) in s
        .d_up
        .chunks_exact_mut(2 * h)
        .zip(s.up.chunks_exact(2 * h))
        .zip(s.d_act.chunks_exact(h))
    {
        let (val, gate) = up_row.split_at(h);
        let (d_val, d_gate) = d_up_row.split_at_mut(h);
        for j in 0..h {
            let sg = 1.0 / (1.0 + fast_exp_scalar(-gate[j]));
            let silu = gate[j] * sg;
            d_val[j] = d_act_row[j] * silu;
            d_gate[j] = d_act_row[j] * val[j] * (sg * (1.0 + gate[j] * (1.0 - sg)));
        }
    }
    store(dtype, &mut s.d_up[..rows * 2 * h]);
    s.gemm.backward(
        dtype,
        (&mut s.d_norm, &mut g.fc1_w, None),
        (&s.d_up, &s.norm, &w.fc1_w),
        (rows, d, 2 * h),
    );

    // The norm. LayerNorm: its own backward, added onto the residual's.
    if layer_norm {
        for (((dt_row, x), dn), &st) in d_temporal[..rows * d]
            .chunks_exact_mut(d)
            .zip(x_in.chunks_exact(d))
            .zip(s.d_norm.chunks_exact(d))
            .zip(&s.ln_stats)
        {
            layer_norm_row_backward(
                dt_row,
                (&mut g.norm_weight, &mut g.norm_bias),
                (dn, x),
                &w.norm_weight,
                st,
                true,
            );
        }
        return;
    }
    // RMSNorm: xh = x * inv, n = xh * w.
    for (((dt_row, x), dn), &inv) in d_temporal[..rows * d]
        .chunks_exact_mut(d)
        .zip(x_in.chunks_exact(d))
        .zip(s.d_norm.chunks_exact(d))
        .zip(&s.inv)
    {
        let mut c = 0.0f32;
        for ((gw, &xv), (&dnv, &wv)) in g
            .norm_weight
            .iter_mut()
            .zip(x)
            .zip(dn.iter().zip(&w.norm_weight))
        {
            let xh = xv * inv;
            *gw += dnv * xh;
            c += (dnv * wv) * xh;
        }
        let c = c / d as f32;
        for ((dtv, &xv), (&dnv, &wv)) in dt_row.iter_mut().zip(x).zip(dn.iter().zip(&w.norm_weight))
        {
            *dtv += inv * (dnv * wv - (xv * inv) * c);
        }
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

    /// The reference block branch `x + mlp(norm2(x))` of one row in f64:
    /// RMSNorm, or LayerNorm when the bias is not empty.
    fn reference_branch(row: &[f64], w: &[Vec<f64>; 4], h: usize, eps: f64) -> Vec<f64> {
        let [nw, nb, fc1, fc2] = w;
        let d = row.len();
        let n: Vec<f64> = if nb.is_empty() {
            let ms = row.iter().map(|v| v * v).sum::<f64>() / d as f64;
            let inv = 1.0 / (ms + eps).sqrt();
            (0..d).map(|i| row[i] * inv * nw[i]).collect()
        } else {
            let m = row.iter().sum::<f64>() / d as f64;
            let var = row.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / d as f64;
            let sd = (var + eps).sqrt();
            (0..d).map(|i| (row[i] - m) / sd * nw[i] + nb[i]).collect()
        };
        let up: Vec<f64> = (0..2 * h)
            .map(|j| (0..d).map(|i| n[i] * fc1[i * 2 * h + j]).sum())
            .collect();
        let act: Vec<f64> = (0..h)
            .map(|j| up[j] * (up[h + j] / (1.0 + (-up[h + j]).exp())))
            .collect();
        (0..d)
            .map(|o| row[o] + (0..h).map(|j| act[j] * fc2[j * d + o]).sum::<f64>())
            .collect()
    }

    fn mlp_weights(d: usize, h: usize, layer_norm: bool) -> GatedMlpLayerWeights {
        GatedMlpLayerWeights {
            norm_weight: draw(d, 1).iter().map(|v| 1.0 + v).collect(),
            norm_bias: if layer_norm { draw(d, 5) } else { Vec::new() },
            fc1_w: draw(d * 2 * h, 2),
            fc2_w: draw(h * d, 3),
        }
    }

    fn wide(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&a| f64::from(a)).collect()
    }

    /// The step and the rows form agree with a plain f64 transcription of
    /// the reference block branch, under either norm.
    #[test]
    fn gated_mlp_matches_the_reference_branch() {
        let (d, h, rows) = (24usize, 128usize, 5usize);
        let eps = 1e-5f32;
        for layer_norm in [false, true] {
            let w = mlp_weights(d, h, layer_norm);
            let wd = [
                wide(&w.norm_weight),
                wide(&w.norm_bias),
                wide(&w.fc1_w),
                wide(&w.fc2_w),
            ];
            let x = draw(rows * d, 4);
            let want: Vec<f64> = x
                .chunks_exact(d)
                .flat_map(|row| reference_branch(&wide(row), &wd, h, f64::from(eps)))
                .collect();
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
                        mlp_norm_row(o, i, &w, eps);
                    }
                },
            );
            for (label, got) in [("step", &step_out), ("rows", &rows_out)] {
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (f64::from(g) - w).abs() <= 1e-5 * w.abs().max(1.0),
                        "{label} (layer norm {layer_norm})[{i}]: {g} vs {w}"
                    );
                }
            }
        }
    }

    /// The backward against central differences of `sum(g * out)` taken
    /// through the f64 reference, under either norm: the input gradient
    /// (the residual's included), every norm parameter and a sweep of each
    /// projection's entries.
    #[test]
    fn gated_mlp_backward_is_the_reference_derivative() {
        let (d, h, rows) = (12usize, 8usize, 3usize);
        let eps = 1e-5f64;
        for layer_norm in [false, true] {
            let w = mlp_weights(d, h, layer_norm);
            let x = draw(rows * d, 6);
            let g_out = draw(rows * d, 7);
            let mut d_t = g_out.clone();
            let mut grads = GatedMlpLayerWeights {
                norm_weight: vec![0.0; d],
                norm_bias: vec![0.0; w.norm_bias.len()],
                fc1_w: vec![0.0; d * 2 * h],
                fc2_w: vec![0.0; h * d],
            };
            let mut s = GatedMlpBackwardScratch::default();
            gated_mlp_rows_backward(&mut d_t, &x, rows, (&w, &mut grads), &mut s, eps as f32);

            let g64 = wide(&g_out);
            let loss = |x: &[f64], wd: &[Vec<f64>; 4]| -> f64 {
                x.chunks_exact(d)
                    .zip(g64.chunks_exact(d))
                    .map(|(row, g)| {
                        reference_branch(row, wd, h, eps)
                            .iter()
                            .zip(g)
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                    })
                    .sum()
            };
            let x64 = wide(&x);
            let wd = [
                wide(&w.norm_weight),
                wide(&w.norm_bias),
                wide(&w.fc1_w),
                wide(&w.fc2_w),
            ];
            let step = 1e-6f64;
            let close = |got: f32, want: f64, what: &str| {
                assert!(
                    (f64::from(got) - want).abs() <= 2e-4 * want.abs().max(1.0),
                    "layer norm {layer_norm} {what}: {got} vs {want}"
                );
            };
            for i in 0..rows * d {
                let (mut p, mut m) = (x64.clone(), x64.clone());
                p[i] += step;
                m[i] -= step;
                let want = (loss(&p, &wd) - loss(&m, &wd)) / (2.0 * step);
                close(d_t[i], want, &format!("dx[{i}]"));
            }
            let analytic = [
                &grads.norm_weight,
                &grads.norm_bias,
                &grads.fc1_w,
                &grads.fc2_w,
            ];
            for (t, got) in analytic.iter().enumerate() {
                for i in (0..got.len()).step_by(got.len().div_ceil(16).max(1)) {
                    let (mut p, mut m) = (wd.clone(), wd.clone());
                    p[t][i] += step;
                    m[t][i] -= step;
                    let want = (loss(&x64, &p) - loss(&x64, &m)) / (2.0 * step);
                    close(got[i], want, &format!("tensor {t}[{i}]"));
                }
            }
        }
    }
}
