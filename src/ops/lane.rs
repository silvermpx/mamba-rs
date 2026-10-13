//! The GEMMs of a precision lane on the CPU, as the GPU lane of the same
//! precision runs them: f32 accumulation in every lane; TF32 operands
//! rounded to nearest with ties away from zero; bf16 and f16 products of
//! values already in their storage, the result rounded once at its store.
//! The f32 lane calls the f32 GEMMs as they are, so its bits do not move.

use crate::mlp::cpu::RowsGemm;
use crate::ops::blas::sgemm_backward;
use crate::ops::packed::{PackedLinear, linear_row};
use crate::ops::rounding::{operand, store};
use crate::precision::WeightDtype;

/// The operand copies a lane's GEMMs round into, grown on first use.
#[derive(Debug, Default)]
pub(crate) struct LaneGemm {
    x: Vec<f32>,
    dy: Vec<f32>,
}

/// `src` copied into `buf` and rounded to a TF32 operand.
fn tf32_copy<'a>(buf: &'a mut Vec<f32>, src: &[f32]) -> &'a [f32] {
    buf.clear();
    buf.extend_from_slice(src);
    operand(WeightDtype::Tf32, buf);
    buf
}

impl LaneGemm {
    /// `y = x w + b` over `batch` rows of `n_in` into `n_out` in `dtype`,
    /// `gemm` the f32 GEMM underneath. The weights are taken as given: a
    /// lane keeps its bulk matrices rounded once (the backbones at
    /// construction, the trainers after every step). TF32 rounds the rows
    /// here; a half lane's rows are already stored values, and its result
    /// is rounded at the store.
    pub(crate) fn forward(
        &mut self,
        (gemm, dtype): (RowsGemm, WeightDtype),
        (y, x): (&mut [f32], &[f32]),
        (w, b): (&[f32], Option<&[f32]>),
        (batch, n_in, n_out): (usize, usize, usize),
    ) {
        if dtype == WeightDtype::Tf32 {
            let x = tf32_copy(&mut self.x, &x[..batch * n_in]);
            gemm(y, x, w, b, batch, n_in, n_out);
        } else {
            gemm(y, x, w, b, batch, n_in, n_out);
            store(dtype, &mut y[..batch * n_out]);
        }
    }

    /// One row through a linear layer in `dtype`, from the packed panels
    /// when there are some: [`LaneGemm::forward`] for a decode step.
    pub(crate) fn row(
        &mut self,
        dtype: WeightDtype,
        (y, x): (&mut [f32], &[f32]),
        (w, packed): (&[f32], Option<&PackedLinear>),
        b: Option<&[f32]>,
    ) {
        let dims = (x.len(), y.len());
        if dtype == WeightDtype::Tf32 {
            let x = tf32_copy(&mut self.x, x);
            linear_row(y, x, (w, packed), b, dims);
        } else {
            linear_row(y, x, (w, packed), b, dims);
            store(dtype, y);
        }
    }

    /// The backward of [`LaneGemm::forward`]: `dx = dy wᵀ` (overwritten),
    /// `dw += xᵀ dy`, and `db += colsum(dy)` of `dy` as given. The products
    /// take the lane's `dy`: rounded to the half storage (the GPU keeps a
    /// rounded copy of an f32 gradient for its GEMMs, the bias summing the
    /// f32 one) or to a TF32 operand, with `x` too in TF32; a half `dx` is
    /// rounded at its store.
    pub(crate) fn backward(
        &mut self,
        dtype: WeightDtype,
        (dx, dw, db): (&mut [f32], &mut [f32], Option<&mut [f32]>),
        (dy, x, w): (&[f32], &[f32], &[f32]),
        dims: (usize, usize, usize),
    ) {
        if dtype == WeightDtype::F32 {
            sgemm_backward(dx, dw, db, dy, x, w, dims);
            return;
        }
        let (batch, n_in, n_out) = dims;
        let dy = &dy[..batch * n_out];
        self.dy.clear();
        self.dy.extend_from_slice(dy);
        let x = if dtype == WeightDtype::Tf32 {
            operand(dtype, &mut self.dy);
            tf32_copy(&mut self.x, &x[..batch * n_in])
        } else {
            store(dtype, &mut self.dy);
            &x[..batch * n_in]
        };
        sgemm_backward(dx, dw, None, &self.dy, x, w, dims);
        store(dtype, &mut dx[..batch * n_in]);
        if let Some(db) = db {
            for row in dy.chunks_exact(n_out) {
                for (b, &g) in db.iter_mut().zip(row) {
                    *b += g;
                }
            }
        }
    }
}

/// Round in place the bulk matrices a lane keeps: to the half storage, or
/// to TF32 operands, which the GPU rounds at every load and which round
/// the same once.
pub(crate) fn round_matrix(dtype: WeightDtype, w: &mut [f32]) {
    match dtype {
        WeightDtype::Tf32 => operand(dtype, w),
        other => store(other, w),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::blas::sgemm_forward;
    use crate::ops::rounding::{round_bf16, round_tf32};

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// The f32 lane is the f32 GEMM; a half lane rounds the f32 GEMM's
    /// result; TF32 multiplies rounded operands.
    #[test]
    fn forward_is_the_f32_gemm_on_the_lanes_operands() {
        let (b, k, n) = (5, 37, 9);
        let (x, w, bias) = (values(b * k, 1), values(k * n, 2), values(n, 3));
        let f32_y = {
            let mut y = vec![0.0; b * n];
            sgemm_forward(&mut y, &x, &w, Some(&bias), b, k, n);
            y
        };
        let mut g = LaneGemm::default();
        let mut y = vec![0.0; b * n];
        let f32_lane = (sgemm_forward as RowsGemm, WeightDtype::F32);
        g.forward(f32_lane, (&mut y, &x), (&w, Some(&bias)), (b, k, n));
        assert_eq!(y, f32_y);
        let bf16_lane = (sgemm_forward as RowsGemm, WeightDtype::Bf16);
        g.forward(bf16_lane, (&mut y, &x), (&w, Some(&bias)), (b, k, n));
        assert_eq!(y, f32_y.iter().map(|&v| round_bf16(v)).collect::<Vec<_>>());
        let wt: Vec<f32> = w.iter().map(|&v| round_tf32(v)).collect();
        let xt: Vec<f32> = x.iter().map(|&v| round_tf32(v)).collect();
        let mut want = vec![0.0; b * n];
        sgemm_forward(&mut want, &xt, &wt, Some(&bias), b, k, n);
        let tf32_lane = (sgemm_forward as RowsGemm, WeightDtype::Tf32);
        g.forward(tf32_lane, (&mut y, &x), (&wt, Some(&bias)), (b, k, n));
        assert_eq!(y, want);
    }

    /// The bias sums the gradient as given; the products take the lane's.
    #[test]
    fn backward_takes_the_lanes_gradient_and_the_given_bias_sum() {
        let (b, k, n) = (4, 11, 6);
        let (x, w, dy) = (values(b * k, 4), values(k * n, 5), values(b * n, 6));
        let xh: Vec<f32> = x.iter().map(|&v| round_bf16(v)).collect();
        let dyh: Vec<f32> = dy.iter().map(|&v| round_bf16(v)).collect();
        let (mut dx, mut dw, mut db) = (vec![0.0; b * k], vec![0.0; k * n], vec![0.0; n]);
        let mut g = LaneGemm::default();
        g.backward(
            WeightDtype::Bf16,
            (&mut dx, &mut dw, Some(&mut db)),
            (&dy, &xh, &w),
            (b, k, n),
        );
        let (mut want_dx, mut want_dw, mut want_db) =
            (vec![0.0; b * k], vec![0.0; k * n], vec![0.0; n]);
        sgemm_backward(&mut want_dx, &mut want_dw, None, &dyh, &xh, &w, (b, k, n));
        sgemm_backward(
            &mut vec![0.0; b * k],
            &mut vec![0.0; k * n],
            Some(&mut want_db),
            &dy,
            &xh,
            &w,
            (b, k, n),
        );
        assert_eq!(
            dx,
            want_dx.iter().map(|&v| round_bf16(v)).collect::<Vec<_>>()
        );
        assert_eq!(dw, want_dw);
        assert_eq!(db, want_db);
    }
}
