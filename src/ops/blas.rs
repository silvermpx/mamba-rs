//! CPU BLAS routines for Mamba training and inference.
//!
//! Platform dispatch:
//! - `accelerate` feature (macOS): Apple Accelerate `cblas_sgemm` via AMX coprocessor
//! - `gemm-blas` feature (any): `gemm` crate with AVX2/AVX-512/NEON microkernels
//! - fallback: pure Rust scalar loops (LLVM auto-vectorizes with target-cpu=native)

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
    {
        for row in 0..batch {
            let x_off = row * n_in;
            let y_off = row * n_out;
            for k in 0..n_in {
                let xv = x[x_off + k];
                let w_off = k * n_out;
                for j in 0..n_out {
                    y[y_off + j] += xv * w[w_off + j];
                }
            }
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
