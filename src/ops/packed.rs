//! A linear layer's weight packed for decode: `W[K, N]` stored as panels of
//! [`PANEL`] columns, each panel `[K][PANEL]` contiguous.
//!
//! A decode step multiplies one row (or a few) by every weight once, so it
//! is bound by how fast the weights stream from memory. Row-major `W` read
//! by column ranges strides a whole row per step of `k`, a page apart for a
//! large layer, which the prefetchers do not follow; a panel is one
//! contiguous block, so the panels split across the cores read the weights
//! at the bandwidth of all of them.
//!
//! Every output sums `x[k] * W[k][j]` over `k` ascending onto the bias, a
//! rounded multiply then a rounded add: the bits of
//! [`sgemm_forward`](super::blas::sgemm_forward) without a BLAS.

use pulp::{Arch, Simd, WithSimd};

use super::blas::matvec_forward;

/// Columns per panel.
pub const PANEL: usize = 64;

/// Above this many weights a lone row splits its panels across the rayon
/// pool: below it one core's share of the bandwidth finishes before the
/// pool would wake.
const PARALLEL_FLOATS: usize = 1 << 21;

/// A weight `W[n_in, n_out]` in panels of [`PANEL`] columns.
#[derive(Clone, Debug)]
pub struct PackedLinear {
    n_in: usize,
    n_out: usize,
    /// `[n_out.div_ceil(PANEL)][n_in][PANEL]`, the last panel's columns
    /// past `n_out` zero.
    panels: Vec<f32>,
}

impl PackedLinear {
    /// Pack a row-major `w` (`n_in * n_out` values, more are ignored).
    ///
    /// # Panics
    ///
    /// When `w` holds fewer than `n_in * n_out` values.
    pub fn pack(w: &[f32], n_in: usize, n_out: usize) -> Self {
        assert!(
            w.len() >= n_in * n_out,
            "PackedLinear::pack: w holds {} values, {n_in} x {n_out} need more",
            w.len()
        );
        let np = n_out.div_ceil(PANEL);
        let mut panels = vec![0f32; np * n_in * PANEL];
        if n_in > 0 {
            for (p, panel) in panels.chunks_exact_mut(n_in * PANEL).enumerate() {
                let j0 = p * PANEL;
                let width = PANEL.min(n_out - j0);
                for (k, row) in panel.as_chunks_mut::<PANEL>().0.iter_mut().enumerate() {
                    row[..width].copy_from_slice(&w[k * n_out + j0..k * n_out + j0 + width]);
                }
            }
        }
        Self {
            n_in,
            n_out,
            panels,
        }
    }

    /// Inputs per row.
    pub fn n_in(&self) -> usize {
        self.n_in
    }

    /// Outputs per row.
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// `y[B, N] = x[B, K] W + bias` for `batch` rows.
    ///
    /// # Panics
    ///
    /// When `x`, `y` or `bias` hold fewer values than the shapes need.
    pub fn forward(&self, y: &mut [f32], x: &[f32], bias: Option<&[f32]>, batch: usize) {
        let (n_in, n_out) = (self.n_in, self.n_out);
        assert!(x.len() >= batch * n_in, "PackedLinear: x too short");
        assert!(y.len() >= batch * n_out, "PackedLinear: y too short");
        let y = &mut y[..batch * n_out];
        match bias {
            Some(b) => {
                assert!(b.len() >= n_out, "PackedLinear: bias too short");
                for row in y.chunks_exact_mut(n_out) {
                    row.copy_from_slice(&b[..n_out]);
                }
            }
            None => y.fill(0.0),
        }
        if n_in == 0 || n_out == 0 || batch == 0 {
            return;
        }
        let x = &x[..batch * n_in];
        let stride = n_in * PANEL;
        if batch == 1
            && n_in * n_out > PARALLEL_FLOATS
            && rayon::current_num_threads() > 1
            && rayon::current_thread_index().is_none()
        {
            use rayon::prelude::*;
            y.par_chunks_mut(PANEL)
                .zip(self.panels.par_chunks_exact(stride))
                .for_each(|(yc, panel)| {
                    let width = yc.len();
                    Arch::new().dispatch(Panels {
                        y: yc,
                        x,
                        panels: panel,
                        n_in,
                        n_out: width,
                        rows: 1,
                    });
                });
            return;
        }
        Arch::new().dispatch(Panels {
            y,
            x,
            panels: &self.panels,
            n_in,
            n_out,
            rows: batch,
        });
    }
}

/// One row `out = x W + bias` from the panels when there are some, else from
/// the row-major `w`: the same bits without a BLAS.
#[inline]
pub fn linear_row(
    out: &mut [f32],
    x: &[f32],
    (w, packed): (&[f32], Option<&PackedLinear>),
    bias: Option<&[f32]>,
    (n_in, n_out): (usize, usize),
) {
    match packed {
        Some(p) => p.forward(out, x, bias, 1),
        None => matvec_forward(out, x, w, bias, n_in, n_out),
    }
}

/// Every panel of `panels` against `rows` rows of `x`, onto the rows of
/// `y` (`n_out` wide; the panels cover them in order).
struct Panels<'a> {
    y: &'a mut [f32],
    x: &'a [f32],
    panels: &'a [f32],
    n_in: usize,
    n_out: usize,
    rows: usize,
}

impl WithSimd for Panels<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        match S::F32_LANES {
            4 => self.run::<S, 16>(simd),
            8 => self.run::<S, 8>(simd),
            16 => self.run::<S, 4>(simd),
            _ => self.run_scalar(),
        }
    }
}

impl Panels<'_> {
    /// Rows four at a time (then the rest one by one), each panel's sums
    /// held in `V` vectors per row across the whole of `k`.
    #[inline(always)]
    fn run<S: Simd, const V: usize>(self, simd: S) {
        let Self {
            y,
            x,
            panels,
            n_in,
            n_out,
            rows,
        } = self;
        for (p, panel) in panels.chunks_exact(n_in * PANEL).enumerate() {
            let j0 = p * PANEL;
            let width = PANEL.min(n_out - j0);
            let mut r = 0;
            while r + 4 <= rows {
                panel_rows::<S, 4, V>(simd, (&mut *y, x, panel), (n_in, n_out, j0, width), r);
                r += 4;
            }
            while r < rows {
                panel_rows::<S, 1, V>(simd, (&mut *y, x, panel), (n_in, n_out, j0, width), r);
                r += 1;
            }
        }
    }

    /// The plain loop, for a SIMD width the panel does not divide into.
    fn run_scalar(self) {
        let Self {
            y,
            x,
            panels,
            n_in,
            n_out,
            rows,
        } = self;
        for (p, panel) in panels.chunks_exact(n_in * PANEL).enumerate() {
            let j0 = p * PANEL;
            let width = PANEL.min(n_out - j0);
            for r in 0..rows {
                for j in 0..width {
                    let mut sum = y[r * n_out + j0 + j];
                    for k in 0..n_in {
                        sum += x[r * n_in + k] * panel[k * PANEL + j];
                    }
                    y[r * n_out + j0 + j] = sum;
                }
            }
        }
    }
}

/// `R` rows from `r0` of one panel: the sums start at `y` (the bias), take
/// `x[k] * W[k][j]` for `k` ascending, and the panel's `width` real columns
/// go back to `y`.
#[inline(always)]
fn panel_rows<S: Simd, const R: usize, const V: usize>(
    simd: S,
    (y, x, panel): (&mut [f32], &[f32], &[f32]),
    (n_in, n_out, j0, width): (usize, usize, usize, usize),
    r0: usize,
) {
    let mut start = [[0f32; PANEL]; R];
    for (r, s) in start.iter_mut().enumerate() {
        let row = (r0 + r) * n_out + j0;
        s[..width].copy_from_slice(&y[row..row + width]);
    }
    let mut acc = [[simd.splat_f32s(0.0); V]; R];
    for (a, s) in acc.iter_mut().zip(&start) {
        let (v, _) = S::as_simd_f32s(&s[..]);
        a.copy_from_slice(&v[..V]);
    }
    for (k, wk) in panel.as_chunks::<PANEL>().0.iter().enumerate() {
        let (wv, _) = S::as_simd_f32s(&wk[..]);
        for (r, a) in acc.iter_mut().enumerate() {
            let xk = simd.splat_f32s(x[(r0 + r) * n_in + k]);
            for (sum, &w) in a.iter_mut().zip(&wv[..V]) {
                *sum = simd.add_f32s(*sum, simd.mul_f32s(xk, w));
            }
        }
    }
    for (r, a) in acc.iter().enumerate() {
        let mut out = [0f32; PANEL];
        let (v, _) = S::as_mut_simd_f32s(&mut out[..]);
        v[..V].copy_from_slice(a);
        let row = (r0 + r) * n_out + j0;
        y[row..row + width].copy_from_slice(&out[..width]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::blas::sgemm_forward;

    fn det(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// The packed product gives the row-major product's bits for every
    /// row count and a ragged last panel, on one thread and on the pool.
    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        feature = "gemm-blas"
    )))]
    #[test]
    fn packed_linear_matches_the_row_major_bits() {
        for (k, n) in [(3, 5), (64, 64), (100, 130), (512, 1030), (700, 900)] {
            let w = det(k * n, 1);
            let b = det(n, 2);
            let packed = PackedLinear::pack(&w, k, n);
            for batch in [1usize, 2, 4, 7] {
                let x = det(batch * k, 3 + batch as u64);
                let mut want = vec![0f32; batch * n];
                sgemm_forward(&mut want, &x, &w, Some(&b), batch, k, n);
                let mut got = vec![f32::NAN; batch * n];
                packed.forward(&mut got, &x, Some(&b), batch);
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{k} x {n}, {batch} rows"
                );
                let mut no_bias = vec![f32::NAN; batch * n];
                packed.forward(&mut no_bias, &x, None, batch);
                sgemm_forward(&mut want, &x, &w, None, batch, k, n);
                assert_eq!(no_bias, want, "{k} x {n}, {batch} rows, no bias");
            }
        }
    }

    /// A lone row inside a pool task runs its panels on that thread and
    /// gives the same bits as on the pool.
    #[test]
    fn packed_linear_lone_row_is_the_same_on_one_thread() {
        let (k, n) = (700, 900);
        let w = det(k * n, 4);
        let x = det(k, 5);
        let packed = PackedLinear::pack(&w, k, n);
        let mut pooled = vec![0f32; n];
        packed.forward(&mut pooled, &x, None, 1);
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut serial = vec![0f32; n];
        one.install(|| packed.forward(&mut serial, &x, None, 1));
        assert_eq!(pooled, serial);
    }
}
