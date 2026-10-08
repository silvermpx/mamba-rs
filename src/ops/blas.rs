//! CPU BLAS routines for Mamba training and inference.
//!
//! Platform dispatch:
//! - `accelerate` feature (macOS): Apple Accelerate `cblas_sgemm` via AMX coprocessor
//! - `gemm-blas` feature (any): `gemm` crate with AVX2/AVX-512/NEON microkernels
//! - fallback: pure Rust register tiles under the runtime-chosen SIMD level
//!   (pulp), each output summing in ascending k like the plain loop

// ---------------------------------------------------------------------------
// Apple Accelerate FFI (macOS only, behind `accelerate` feature)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

// ---------------------------------------------------------------------------
// SGEMM forward: Y[B,N] = X[B,K] @ W[K,N] + bias[N]
// ---------------------------------------------------------------------------

/// Batched linear forward: `Y[B,N] = X[B,K] @ W[K,N] + bias[N]`.
///
/// All matrices are flat row-major `[rows * cols]`.
/// Dispatches to platform BLAS when feature flags are enabled.
pub fn sgemm_forward(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    n_in: usize,
    n_out: usize,
) {
    // Guard the FFI paths: cblas_sgemm/gemm read raw pointers with sizes
    // derived from the dims, so undersized slices (e.g. weights from an
    // unvalidated checkpoint) would read out of bounds. Negligible cost
    // relative to the GEMM itself.
    assert!(
        x.len() >= batch * n_in,
        "sgemm_forward: x.len() {} < batch*n_in {}",
        x.len(),
        batch * n_in
    );
    assert!(
        w.len() >= n_in * n_out,
        "sgemm_forward: w.len() {} < n_in*n_out {}",
        w.len(),
        n_in * n_out
    );
    assert!(
        y.len() >= batch * n_out,
        "sgemm_forward: y.len() {} < batch*n_out {}",
        y.len(),
        batch * n_out
    );

    // Pre-fill with bias
    if let Some(b) = bias {
        for row in 0..batch {
            let off = row * n_out;
            y[off..off + n_out].copy_from_slice(&b[..n_out]);
        }
    } else {
        y[..batch * n_out].fill(0.0);
    }

    // Dispatch SGEMM: Y += X @ W
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    unsafe {
        cblas_sgemm(
            101,            // CblasRowMajor
            111,            // CblasNoTrans
            111,            // CblasNoTrans
            batch as i32,   // M
            n_out as i32,   // N
            n_in as i32,    // K
            1.0,            // alpha
            x.as_ptr(),     // A
            n_in as i32,    // lda
            w.as_ptr(),     // B
            n_out as i32,   // ldb
            1.0,            // beta (accumulate into bias)
            y.as_mut_ptr(), // C
            n_out as i32,   // ldc
        );
    }

    #[cfg(all(
        feature = "gemm-blas",
        not(all(feature = "accelerate", target_os = "macos"))
    ))]
    unsafe {
        gemm::gemm(
            batch,
            n_out,
            n_in,
            y.as_mut_ptr(),
            1,              // dst col stride
            n_out as isize, // dst row stride
            true,           // read dst (alpha=1, accumulate into bias)
            x.as_ptr(),
            1,             // lhs col stride
            n_in as isize, // lhs row stride
            w.as_ptr(),
            1,              // rhs col stride
            n_out as isize, // rhs row stride
            1.0,            // alpha (scale existing dst)
            1.0,            // beta (scale product)
            false,
            false,
            false,
            gemm::Parallelism::None, // caller controls threading via rayon
        );
    }

    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        feature = "gemm-blas"
    )))]
    gemm_accumulate(
        &mut y[..batch * n_out],
        &x[..batch * n_in],
        &w[..n_in * n_out],
        (n_in, n_out),
    );
}

/// `Y[B,N] += X[B,K] @ W[K,N]` without a BLAS. Every output adds its
/// products in ascending k onto the value already in `y`, each one a
/// rounded multiply and then a rounded add, exactly as the plain triple
/// loop does; only the running sums of a tile of rows and columns stay in
/// vector registers across a block of k instead of going back to memory
/// with every product, and a tile of rows shares each loaded vector of
/// `W`. A pass of a tile reads four rows of `W` at a time: four
/// sequential streams the prefetchers keep up with, where a pass down
/// every row of a large `W` strides a page a row and left a large model's
/// decode slower than the plain row-by-row loop.
#[cfg(not(any(
    all(feature = "accelerate", target_os = "macos"),
    feature = "gemm-blas"
)))]
fn gemm_accumulate(y: &mut [f32], x: &[f32], w: &[f32], (n_in, n_out): (usize, usize)) {
    use pulp::{Arch, Simd, WithSimd};

    /// Above this many elements of W (2 MiB, the order of a core's L2)
    /// a lone row reads W in storage order.
    const STREAM_FLOATS: usize = 1 << 19;

    /// One tile: `R` rows by `V` vectors of columns starting at `j0`, over
    /// the rows `ks` of `W`.
    #[inline(always)]
    fn tile<S: Simd, const R: usize, const V: usize>(
        simd: S,
        (y, x, w): (&mut [f32], &[f32], &[f32]),
        (n_in, n_out, j0): (usize, usize, usize),
        ks: core::ops::Range<usize>,
    ) {
        let width = V * S::F32_LANES;
        let mut acc = [[simd.splat_f32s(0.0); V]; R];
        for (r, row) in acc.iter_mut().enumerate() {
            let (sums, _) = S::as_simd_f32s(&y[r * n_out + j0..r * n_out + j0 + width]);
            row.copy_from_slice(sums);
        }
        for k in ks {
            let (wk, _) = S::as_simd_f32s(&w[k * n_out + j0..k * n_out + j0 + width]);
            for (r, row) in acc.iter_mut().enumerate() {
                let xk = simd.splat_f32s(x[r * n_in + k]);
                for (sum, &wv) in row.iter_mut().zip(wk) {
                    *sum = simd.add_f32s(*sum, simd.mul_f32s(xk, wv));
                }
            }
        }
        for (r, row) in acc.iter().enumerate() {
            let (sums, _) = S::as_mut_simd_f32s(&mut y[r * n_out + j0..r * n_out + j0 + width]);
            sums.copy_from_slice(row);
        }
    }

    /// Every column of `R` rows: wide tiles, then narrower ones, then the
    /// columns short of a vector one by one.
    #[inline(always)]
    fn rows<S: Simd, const R: usize>(
        simd: S,
        (y, x, w): (&mut [f32], &[f32], &[f32]),
        (n_in, n_out): (usize, usize),
    ) {
        let lanes = S::F32_LANES;
        let block = 4;
        let mut j0 = 0;
        for k0 in (0..n_in).step_by(block) {
            let ks = k0..(k0 + block).min(n_in);
            j0 = 0;
            // A lone row keeps eight independent sums in flight per k; a
            // tile of four rows fits its sums in the register file, four
            // vectors a row with four-lane vectors and two with wider ones.
            if R == 1 {
                while j0 + 8 * lanes <= n_out {
                    tile::<S, R, 8>(simd, (&mut *y, x, w), (n_in, n_out, j0), ks.clone());
                    j0 += 8 * lanes;
                }
            } else if lanes <= 4 {
                while j0 + 4 * lanes <= n_out {
                    tile::<S, R, 4>(simd, (&mut *y, x, w), (n_in, n_out, j0), ks.clone());
                    j0 += 4 * lanes;
                }
            }
            while j0 + 2 * lanes <= n_out {
                tile::<S, R, 2>(simd, (&mut *y, x, w), (n_in, n_out, j0), ks.clone());
                j0 += 2 * lanes;
            }
            while j0 + lanes <= n_out {
                tile::<S, R, 1>(simd, (&mut *y, x, w), (n_in, n_out, j0), ks.clone());
                j0 += lanes;
            }
        }
        for r in 0..R {
            for j in j0..n_out {
                let mut sum = y[r * n_out + j];
                for k in 0..n_in {
                    sum += x[r * n_in + k] * w[k * n_out + j];
                }
                y[r * n_out + j] = sum;
            }
        }
    }

    /// One row in the plain loop's order: for each k, every column's sum
    /// in memory takes x[k] * W[k, j].
    #[inline(always)]
    fn stream_row<S: Simd>(simd: S, (y, x, w): (&mut [f32], &[f32], &[f32]), n_out: usize) {
        for (&xk, wk) in x.iter().zip(w.chunks_exact(n_out)) {
            let xv = simd.splat_f32s(xk);
            let (sums, sums_tail) = S::as_mut_simd_f32s(&mut *y);
            let (wv, w_tail) = S::as_simd_f32s(wk);
            for (sum, &wj) in sums.iter_mut().zip(wv) {
                *sum = simd.add_f32s(*sum, simd.mul_f32s(xv, wj));
            }
            for (sum, &wj) in sums_tail.iter_mut().zip(w_tail) {
                *sum += xk * wj;
            }
        }
    }

    struct Gemm<'a> {
        y: &'a mut [f32],
        x: &'a [f32],
        w: &'a [f32],
        n_in: usize,
        n_out: usize,
    }
    impl WithSimd for Gemm<'_> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let Self {
                y,
                x,
                w,
                n_in,
                n_out,
            } = self;
            let mut ys = y.chunks_exact_mut(4 * n_out);
            let mut xs = x.chunks_exact(4 * n_in);
            for (yr, xr) in (&mut ys).zip(&mut xs) {
                rows::<S, 4>(simd, (yr, xr, w), (n_in, n_out));
            }
            let rest = ys.into_remainder();
            // A lone row reads each element of W once, so a W too large
            // to stay in cache streams from memory whatever the tiling;
            // its rows then go in storage order, the one sequential stream
            // the prefetchers serve best, with the sums through memory.
            let streams = n_in * n_out > STREAM_FLOATS;
            for (yr, xr) in rest
                .chunks_exact_mut(n_out)
                .zip(xs.remainder().chunks_exact(n_in))
            {
                if streams {
                    stream_row(simd, (yr, xr, w), n_out);
                } else {
                    rows::<S, 1>(simd, (yr, xr, w), (n_in, n_out));
                }
            }
        }
    }
    if n_in == 0 || n_out == 0 {
        return;
    }
    Arch::new().dispatch(Gemm {
        y,
        x,
        w,
        n_in,
        n_out,
    });
}

/// [`sgemm_forward`] with the gemm crate's internal rayon parallelism —
/// the prefill serving path's single-page latency lever (a lone sequence
/// cannot amortize across a batch, so the GEMM itself must parallelize).
/// Tile-local accumulation keeps results bit-identical to the serial path
/// (verified by the prefill parallel-invariance test). The Accelerate
/// branch already threads internally and the scalar fallback is not a
/// serving configuration — both delegate to [`sgemm_forward`].
pub fn sgemm_forward_par(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    n_in: usize,
    n_out: usize,
) {
    #[cfg(all(
        feature = "gemm-blas",
        not(all(feature = "accelerate", target_os = "macos"))
    ))]
    {
        assert!(
            x.len() >= batch * n_in,
            "sgemm_forward_par: x.len() {} < batch*n_in {}",
            x.len(),
            batch * n_in
        );
        assert!(
            w.len() >= n_in * n_out,
            "sgemm_forward_par: w.len() {} < n_in*n_out {}",
            w.len(),
            n_in * n_out
        );
        assert!(
            y.len() >= batch * n_out,
            "sgemm_forward_par: y.len() {} < batch*n_out {}",
            y.len(),
            batch * n_out
        );
        if let Some(b) = bias {
            for row in 0..batch {
                let off = row * n_out;
                y[off..off + n_out].copy_from_slice(&b[..n_out]);
            }
        } else {
            y[..batch * n_out].fill(0.0);
        }
        unsafe {
            gemm::gemm(
                batch,
                n_out,
                n_in,
                y.as_mut_ptr(),
                1,
                n_out as isize,
                true,
                x.as_ptr(),
                1,
                n_in as isize,
                w.as_ptr(),
                1,
                n_out as isize,
                1.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(rayon::current_num_threads()),
            );
        }
    }
    #[cfg(not(all(
        feature = "gemm-blas",
        not(all(feature = "accelerate", target_os = "macos"))
    )))]
    sgemm_forward(y, x, w, bias, batch, n_in, n_out);
}

/// Single-sample matrix-vector forward: `y[N] = x[K] @ W[K,N] + bias[N]`.
pub fn matvec_forward(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    n_in: usize,
    n_out: usize,
) {
    sgemm_forward(y, x, w, bias, 1, n_in, n_out);
}

// ---------------------------------------------------------------------------
// SGEMM backward: dX = dY @ W^T, dW += X^T @ dY, dBias += colsum(dY)
// ---------------------------------------------------------------------------

/// Batched linear backward: computes dX, dW, and optionally dBias.
///
/// - `dx = dY @ W^T` (overwritten)
/// - `dw += X^T @ dY` (accumulated)
/// - `db += colsum(dY)` (accumulated, if present)
pub fn sgemm_backward(
    dx: &mut [f32],
    dw: &mut [f32],
    db: Option<&mut [f32]>,
    dy: &[f32],
    x_saved: &[f32],
    w: &[f32],
    dims: (usize, usize, usize), // (batch, n_in, n_out)
) {
    let (batch, n_in, n_out) = dims;

    // Guard the FFI paths (see sgemm_forward).
    assert!(
        dy.len() >= batch * n_out,
        "sgemm_backward: dy.len() {} < batch*n_out {}",
        dy.len(),
        batch * n_out
    );
    assert!(
        x_saved.len() >= batch * n_in,
        "sgemm_backward: x_saved.len() {} < batch*n_in {}",
        x_saved.len(),
        batch * n_in
    );
    assert!(
        w.len() >= n_in * n_out,
        "sgemm_backward: w.len() {} < n_in*n_out {}",
        w.len(),
        n_in * n_out
    );
    assert!(
        dx.len() >= batch * n_in,
        "sgemm_backward: dx.len() {} < batch*n_in {}",
        dx.len(),
        batch * n_in
    );
    assert!(
        dw.len() >= n_in * n_out,
        "sgemm_backward: dw.len() {} < n_in*n_out {}",
        dw.len(),
        n_in * n_out
    );

    // dX[B,K] = dY[B,N] @ W^T[N,K]
    dx[..batch * n_in].fill(0.0);

    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    unsafe {
        // dX = dY @ W^T => CblasTrans on B
        cblas_sgemm(
            101,          // CblasRowMajor
            111,          // CblasNoTrans (A = dY)
            112,          // CblasTrans (B = W^T)
            batch as i32, // M
            n_in as i32,  // N (output cols = n_in)
            n_out as i32, // K (shared dim = n_out)
            1.0,
            dy.as_ptr(),
            n_out as i32,
            w.as_ptr(),
            n_out as i32, // ldb = n_out (before transpose)
            0.0,
            dx.as_mut_ptr(),
            n_in as i32,
        );
    }

    #[cfg(all(
        feature = "gemm-blas",
        not(all(feature = "accelerate", target_os = "macos"))
    ))]
    unsafe {
        // dX = dY @ W^T
        gemm::gemm(
            batch,
            n_in,
            n_out,
            dx.as_mut_ptr(),
            1,             // dst col stride
            n_in as isize, // dst row stride
            false,
            dy.as_ptr(),
            1,              // lhs col stride
            n_out as isize, // lhs row stride
            w.as_ptr(),
            n_out as isize, // rhs col stride (transposed: was row)
            1,              // rhs row stride (transposed: was col)
            0.0,            // alpha (don't read dst)
            1.0,            // beta (scale product)
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }

    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        feature = "gemm-blas"
    )))]
    {
        for row in 0..batch {
            let dy_off = row * n_out;
            let dx_off = row * n_in;
            for j in 0..n_out {
                let dv = dy[dy_off + j];
                for k in 0..n_in {
                    dx[dx_off + k] += dv * w[k * n_out + j];
                }
            }
        }
    }

    // dW[K,N] += X^T[K,B] @ dY[B,N]
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    unsafe {
        cblas_sgemm(
            101,
            112,          // CblasTrans (A = X^T)
            111,          // CblasNoTrans (B = dY)
            n_in as i32,  // M
            n_out as i32, // N
            batch as i32, // K
            1.0,
            x_saved.as_ptr(),
            n_in as i32, // lda = n_in (before transpose)
            dy.as_ptr(),
            n_out as i32,
            1.0, // beta = 1.0 (accumulate)
            dw.as_mut_ptr(),
            n_out as i32,
        );
    }

    #[cfg(all(
        feature = "gemm-blas",
        not(all(feature = "accelerate", target_os = "macos"))
    ))]
    unsafe {
        // dW += X^T @ dY
        gemm::gemm(
            n_in,
            n_out,
            batch,
            dw.as_mut_ptr(),
            1,              // dst col stride
            n_out as isize, // dst row stride
            true,           // accumulate
            x_saved.as_ptr(),
            n_in as isize, // lhs col stride (transposed X: rs and cs swapped)
            1,             // lhs row stride
            dy.as_ptr(),
            1,              // rhs col stride
            n_out as isize, // rhs row stride
            1.0,            // alpha (accumulate)
            1.0,            // beta
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }

    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        feature = "gemm-blas"
    )))]
    {
        for row in 0..batch {
            let x_off = row * n_in;
            let dy_off = row * n_out;
            for k in 0..n_in {
                let xv = x_saved[x_off + k];
                let w_off = k * n_out;
                for j in 0..n_out {
                    dw[w_off + j] += xv * dy[dy_off + j];
                }
            }
        }
    }

    // dBias[N] += colsum(dY) — always scalar (tiny)
    if let Some(db) = db {
        for row in 0..batch {
            let dy_off = row * n_out;
            for j in 0..n_out {
                db[j] += dy[dy_off + j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemm_forward_identity() {
        let n = 4;
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let mut y = vec![0.0; n];
        sgemm_forward(&mut y, &x, &w, None, 1, n, n);
        for (i, yi) in y.iter().enumerate().take(n) {
            assert!((*yi - (i + 1) as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sgemm_forward_with_bias() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, 4.0];
        let bias = vec![10.0, 20.0];
        let mut y = vec![0.0; 2];
        sgemm_forward(&mut y, &x, &w, Some(&bias), 1, 2, 2);
        assert!((y[0] - 13.0).abs() < 1e-6);
        assert!((y[1] - 24.0).abs() < 1e-6);
    }

    /// The fallback's register tiles against the plain triple loop: every
    /// output bit for bit (any NaN matching any NaN, whose payload the
    /// compiler may pick from either operand), across row counts that leave
    /// remainders of the four-row tiles, column counts that leave every
    /// narrower tile and a scalar tail, lone rows of a W large enough to go
    /// in storage order, and values that reach rounding, signed zeros,
    /// infinities and NaN.
    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        feature = "gemm-blas"
    )))]
    #[test]
    fn test_sgemm_forward_matches_the_plain_loop_bitwise() {
        let synth = |n: usize, seed: u32| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|i| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let v = ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 3.0;
                    match i % 97 {
                        13 => -0.0,
                        41 => f32::INFINITY,
                        77 => f32::NAN,
                        _ => v,
                    }
                })
                .collect()
        };
        for (batch, n_in, n_out) in [
            (1, 128, 596),
            (1, 7, 3),
            (3, 33, 70),
            (4, 16, 64),
            (5, 9, 131),
            (9, 64, 37),
            (1, 1, 1),
            (1, 600, 1003),
            (5, 600, 1003),
        ] {
            let x = synth(batch * n_in, 1 + n_in as u32);
            let w = synth(n_in * n_out, 2 + n_out as u32);
            let bias = synth(n_out, 3);
            let mut want = vec![0.0f32; batch * n_out];
            for row in 0..batch {
                want[row * n_out..(row + 1) * n_out].copy_from_slice(&bias);
                for k in 0..n_in {
                    let xv = x[row * n_in + k];
                    for j in 0..n_out {
                        want[row * n_out + j] += xv * w[k * n_out + j];
                    }
                }
            }
            let mut got = vec![0.0f32; batch * n_out];
            sgemm_forward(&mut got, &x, &w, Some(&bias), batch, n_in, n_out);
            let differ = want
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
                .count();
            assert_eq!(differ, 0, "{batch}x{n_in}x{n_out}: {differ} outputs differ");
        }
    }

    #[test]
    fn test_sgemm_backward_gradient() {
        let batch = 2;
        let n_in = 3;
        let n_out = 2;
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let dy = vec![1.0, 1.0, 1.0, 1.0];
        let mut dx = vec![0.0; batch * n_in];
        let mut dw = vec![0.0; n_in * n_out];
        sgemm_backward(&mut dx, &mut dw, None, &dy, &x, &w, (batch, n_in, n_out));
        // dx[0] = dy[0] @ W^T row 0 = [1,1] @ [[1,3,5],[2,4,6]]^T col 0 = 1*1+1*2 = 3
        assert!((dx[0] - 3.0).abs() < 1e-5);
        assert!((dx[1] - 7.0).abs() < 1e-5);
        assert!((dx[2] - 11.0).abs() < 1e-5);
    }
}
