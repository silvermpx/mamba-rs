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

/// One tile of the BLAS-free GEMM: `R` rows by `V` vectors of columns
/// starting at `j0`, over the rows `ks` of `W`; the running sums start from
/// `y` and go back to it.
#[cfg(not(any(
    all(feature = "accelerate", target_os = "macos"),
    feature = "gemm-blas"
)))]
#[inline(always)]
fn tile<S: pulp::Simd, const R: usize, const V: usize>(
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

/// [`gemm_accumulate`] for one block of rows, ordered for a `W` larger
/// than the caches: the columns go in panels of 64 and, for each, `k` in
/// slabs of 256 rows of `W`, so a slab of the panel (64 KB) stays in L1
/// while every 4-row tile of the block passes over it; the block then
/// reads `W` once instead of once per 4 rows. Every output still adds its
/// products onto its value in ascending `k` with a rounded multiply and a
/// rounded add (the sums go back to `y` between slabs exactly), so the
/// bits are the serial loop's.
#[cfg(not(any(
    all(feature = "accelerate", target_os = "macos"),
    feature = "gemm-blas"
)))]
fn gemm_accumulate_panels(y: &mut [f32], x: &[f32], w: &[f32], (n_in, n_out): (usize, usize)) {
    use pulp::{Arch, Simd, WithSimd};
    const PANEL: usize = 64;
    const SLAB: usize = 256;

    /// The tiles of `R` rows across the columns `c0..c1`, then the
    /// columns short of a vector one by one.
    #[inline(always)]
    fn strips<S: Simd, const R: usize>(
        simd: S,
        (y, x, w): (&mut [f32], &[f32], &[f32]),
        (n_in, n_out, c0, c1): (usize, usize, usize, usize),
        ks: core::ops::Range<usize>,
    ) {
        let lanes = S::F32_LANES;
        let mut j = c0;
        if lanes <= 4 {
            while j + 4 * lanes <= c1 {
                tile::<S, R, 4>(simd, (&mut *y, x, w), (n_in, n_out, j), ks.clone());
                j += 4 * lanes;
            }
        }
        while j + 2 * lanes <= c1 {
            tile::<S, R, 2>(simd, (&mut *y, x, w), (n_in, n_out, j), ks.clone());
            j += 2 * lanes;
        }
        while j + lanes <= c1 {
            tile::<S, R, 1>(simd, (&mut *y, x, w), (n_in, n_out, j), ks.clone());
            j += lanes;
        }
        for r in 0..R {
            for jj in j..c1 {
                let mut sum = y[r * n_out + jj];
                for k in ks.clone() {
                    sum += x[r * n_in + k] * w[k * n_out + jj];
                }
                y[r * n_out + jj] = sum;
            }
        }
    }

    struct Panels<'a> {
        y: &'a mut [f32],
        x: &'a [f32],
        w: &'a [f32],
        n_in: usize,
        n_out: usize,
    }
    impl WithSimd for Panels<'_> {
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
            let rows = y.len() / n_out;
            for c0 in (0..n_out).step_by(PANEL) {
                let c1 = (c0 + PANEL).min(n_out);
                for k0 in (0..n_in).step_by(SLAB) {
                    let ks = k0..(k0 + SLAB).min(n_in);
                    let mut r = 0;
                    while r + 4 <= rows {
                        let (yr, xr) = (&mut y[r * n_out..(r + 4) * n_out], &x[r * n_in..]);
                        strips::<S, 4>(simd, (yr, xr, w), (n_in, n_out, c0, c1), ks.clone());
                        r += 4;
                    }
                    for r in r..rows {
                        let (yr, xr) = (&mut y[r * n_out..(r + 1) * n_out], &x[r * n_in..]);
                        strips::<S, 1>(simd, (yr, xr, w), (n_in, n_out, c0, c1), ks.clone());
                    }
                }
            }
        }
    }
    if n_in == 0 || n_out == 0 {
        return;
    }
    Arch::new().dispatch(Panels {
        y,
        x,
        w,
        n_in,
        n_out,
    });
}

/// Whether a BLAS-free GEMM of `macs` products goes across the rayon
/// pool: enough work, more than one thread, and a caller not already
/// inside a pool task (a per-sample trainer keeps its GEMMs serial).
#[cfg(not(any(
    all(feature = "accelerate", target_os = "macos"),
    feature = "gemm-blas"
)))]
fn pool_gemm(macs: usize) -> bool {
    macs >= 1 << 22 && rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none()
}

/// `dX[B,K] = dY[B,N] W^T` without a BLAS: each block of 64 rows of `W`
/// goes transposed into an `[N][64]` panel and the forward GEMM runs `dY`
/// against it into a `[B][64]` block of `dX`. Every element is 0 plus its
/// products over ascending `j`, each a rounded multiply and a rounded add,
/// the chain of the plain loop over `j`; only the walk over memory
/// changes (the plain loop read `W` a column at a time).
#[cfg(not(any(
    all(feature = "accelerate", target_os = "macos"),
    feature = "gemm-blas"
)))]
fn backward_dx(dx: &mut [f32], dy: &[f32], w: &[f32], (batch, n_in, n_out): (usize, usize, usize)) {
    const PANEL: usize = 64;
    let block = |k0: usize| -> Vec<f32> {
        let kn = PANEL.min(n_in - k0);
        let mut wt = vec![0.0f32; n_out * kn];
        for kk in 0..kn {
            let row = &w[(k0 + kk) * n_out..(k0 + kk + 1) * n_out];
            for (j, &v) in row.iter().enumerate() {
                wt[j * kn + kk] = v;
            }
        }
        let mut out = vec![0.0f32; batch * kn];
        gemm_accumulate(&mut out, dy, &wt, (n_out, kn));
        out
    };
    let place = |dx: &mut [f32], k0: usize, out: &[f32]| {
        let kn = out.len() / batch.max(1);
        for (r, src) in out.chunks_exact(kn.max(1)).enumerate() {
            dx[r * n_in + k0..r * n_in + k0 + kn].copy_from_slice(src);
        }
    };
    if batch == 0 || n_in == 0 {
        return;
    }
    if pool_gemm(batch * n_in * n_out) {
        use rayon::prelude::*;
        let blocks: Vec<(usize, Vec<f32>)> = (0..n_in)
            .into_par_iter()
            .step_by(PANEL)
            .map(|k0| (k0, block(k0)))
            .collect();
        for (k0, out) in &blocks {
            place(dx, *k0, out);
        }
    } else {
        for k0 in (0..n_in).step_by(PANEL) {
            let out = block(k0);
            place(dx, k0, &out);
        }
    }
}

/// `dW[K,N] += X^T[K,B] dY[B,N]` without a BLAS: `X` transposed once, then
/// the forward GEMM over blocks of rows of `dW` with `dY` as the weights.
/// Every element adds its products onto its value over ascending batch
/// rows with a rounded multiply and a rounded add, the plain loop's chain
/// (which rewrote all of `dW` once per batch row).
#[cfg(not(any(
    all(feature = "accelerate", target_os = "macos"),
    feature = "gemm-blas"
)))]
fn backward_dw(dw: &mut [f32], x: &[f32], dy: &[f32], (batch, n_in, n_out): (usize, usize, usize)) {
    if batch == 0 || n_in == 0 || n_out == 0 {
        return;
    }
    let mut xt = vec![0.0f32; n_in * batch];
    for (r, row) in x.chunks_exact(n_in).enumerate() {
        for (k, &v) in row.iter().enumerate() {
            xt[k * batch + r] = v;
        }
    }
    if pool_gemm(batch * n_in * n_out) {
        use rayon::prelude::*;
        let threads = rayon::current_num_threads();
        let block = n_in.div_ceil(2 * threads).next_multiple_of(4).clamp(4, 64);
        dw.par_chunks_mut(block * n_out)
            .zip(xt.par_chunks(block * batch))
            .for_each(|(d, xb)| gemm_accumulate(d, xb, dy, (batch, n_out)));
    } else {
        gemm_accumulate(dw, &xt, dy, (batch, n_out));
    }
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

/// [`sgemm_forward`] spread over the rayon pool — the prefill and training
/// forward's latency lever (a lone sequence cannot amortize across a
/// batch, so the GEMM itself must parallelize). With the gemm crate its
/// internal parallelism runs; without a BLAS the rows split into blocks,
/// one task each, walked panel by panel; the
/// Accelerate branch threads internally. Every output keeps the serial
/// chain, so the bits do not depend on the pool (the prefill
/// parallel-invariance test and the plain-loop bit test hold).
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
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    sgemm_forward(y, x, w, bias, batch, n_in, n_out);
    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        feature = "gemm-blas"
    )))]
    {
        // Rows split into blocks across the pool, each block one task over
        // its own rows of Y: every output's chain stays where the serial
        // loop has it, so the bits do not depend on the pool. Inside a
        // rayon task (a per-sample trainer, a batch of decodes) the pool is
        // already busy and the GEMM stays serial; small GEMMs too.
        if batch >= 8 && pool_gemm(batch * n_in * n_out) {
            let threads = rayon::current_num_threads();
            use rayon::prelude::*;
            assert!(
                x.len() >= batch * n_in && w.len() >= n_in * n_out && y.len() >= batch * n_out,
                "sgemm_forward_par: undersized operands for {batch}x{n_in}x{n_out}"
            );
            let y = &mut y[..batch * n_out];
            match bias {
                Some(b) => y
                    .chunks_exact_mut(n_out)
                    .for_each(|r| r.copy_from_slice(&b[..n_out])),
                None => y.fill(0.0),
            }
            let block = batch.div_ceil(2 * threads).next_multiple_of(4).clamp(4, 64);
            let w = &w[..n_in * n_out];
            y.par_chunks_mut(block * n_out)
                .zip(x[..batch * n_in].par_chunks(block * n_in))
                .for_each(|(yb, xb)| gemm_accumulate_panels(yb, xb, w, (n_in, n_out)));
        } else {
            sgemm_forward(y, x, w, bias, batch, n_in, n_out);
        }
    }
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
    backward_dx(
        &mut dx[..batch * n_in],
        &dy[..batch * n_out],
        &w[..n_in * n_out],
        (batch, n_in, n_out),
    );

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
    backward_dw(
        &mut dw[..n_in * n_out],
        &x_saved[..batch * n_in],
        &dy[..batch * n_out],
        (batch, n_in, n_out),
    );

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

    /// The pool-parallel forward (row blocks walked panel by panel) gives
    /// the plain loop's bits at shapes large enough to take it, ragged rows
    /// and columns included.
    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        feature = "gemm-blas"
    )))]
    #[test]
    fn test_sgemm_forward_par_matches_the_plain_loop_bitwise() {
        let synth = |n: usize, seed: u32| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|i| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let v = ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 3.0;
                    if i % 211 == 7 { -0.0 } else { v }
                })
                .collect()
        };
        for (batch, n_in, n_out) in [
            (64, 256, 300),
            (37, 600, 1003),
            (9, 700, 700),
            (130, 97, 515),
        ] {
            let x = synth(batch * n_in, 11 + n_in as u32);
            let w = synth(n_in * n_out, 12 + n_out as u32);
            let bias = synth(n_out, 13);
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
            sgemm_forward_par(&mut got, &x, &w, Some(&bias), batch, n_in, n_out);
            let differ = want
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(differ, 0, "{batch}x{n_in}x{n_out}: {differ} outputs differ");
        }
    }

    /// The panel-based backward GEMMs give the plain loops' bits: dX as 0
    /// plus the products over ascending j, dW as its value plus the
    /// products over ascending batch rows; serial and pool-sized shapes,
    /// ragged edges, signed zeros and non-finite values included.
    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        feature = "gemm-blas"
    )))]
    #[test]
    fn test_sgemm_backward_matches_the_plain_loops_bitwise() {
        let synth = |n: usize, seed: u32| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|i| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let v = ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 3.0;
                    match i % 101 {
                        13 => -0.0,
                        41 => f32::INFINITY,
                        _ => v,
                    }
                })
                .collect()
        };
        let same = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .filter(|(x, y)| x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()))
                .count()
        };
        for (batch, n_in, n_out) in [
            (1, 7, 3),
            (5, 9, 131),
            (3, 130, 70),
            (64, 300, 260),
            (37, 129, 1003),
        ] {
            let dy = synth(batch * n_out, 21 + n_out as u32);
            let x = synth(batch * n_in, 22 + n_in as u32);
            let w = synth(n_in * n_out, 23);
            let dw0 = synth(n_in * n_out, 24);
            let mut want_dx = vec![0.0f32; batch * n_in];
            let mut want_dw = dw0.clone();
            for row in 0..batch {
                for j in 0..n_out {
                    let dv = dy[row * n_out + j];
                    for k in 0..n_in {
                        want_dx[row * n_in + k] += dv * w[k * n_out + j];
                    }
                }
                for k in 0..n_in {
                    let xv = x[row * n_in + k];
                    for j in 0..n_out {
                        want_dw[k * n_out + j] += xv * dy[row * n_out + j];
                    }
                }
            }
            let mut dx = vec![7.0f32; batch * n_in];
            let mut dw = dw0.clone();
            sgemm_backward(&mut dx, &mut dw, None, &dy, &x, &w, (batch, n_in, n_out));
            assert_eq!(same(&want_dx, &dx), 0, "dX {batch}x{n_in}x{n_out}");
            assert_eq!(same(&want_dw, &dw), 0, "dW {batch}x{n_in}x{n_out}");
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
