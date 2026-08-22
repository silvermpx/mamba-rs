//! P0.5 — CUDA-13 cuBLAS compute-mode probe (the pedantic-pin decision
//! instrument).
//!
//! History: 61325b3 (RTX 4090, CUDA 12.8) observed K-scaling accumulation
//! degradation on bf16 GEMMs under default cuBLAS math (~2% KL on
//! mamba-1.4b, greedy divergence after 4 tokens) and pinned the typed
//! compute type to `CUBLAS_COMPUTE_32F_PEDANTIC`. PEDANTIC forbids ALL
//! algorithmic shortcuts; the suspected culprit was only the
//! reduced-precision split-K reduction, which the dedicated
//! `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` handle bit forbids
//! while keeping tensor-core BMMA speed. This probe measures each handle
//! cell against an fp64 reference (same bf16 bits, widened) so the pin
//! can be replaced by the narrow flag IF AND ONLY IF the data says so.
//!
//! Cells (bf16 A/B, f32 C):
//!   V0 = COMPUTE_32F_PEDANTIC + CUBLAS_DEFAULT_MATH   (today's pin)
//!   V1 = COMPUTE_32F_PEDANTIC + TF32                  (math-mode inertness check)
//!   V2 = COMPUTE_32F          + TF32                  (today's FAST_GEMM)
//!   V3 = COMPUTE_32F          + DEFAULT
//!   V4 = COMPUTE_32F          + (TF32 | DISALLOW_REDUCED_PRECISION_REDUCTION)
//!   V5 = COMPUTE_32F          + DISALLOW_REDUCED_PRECISION_REDUCTION
//! f32 lane (f32 A/B/C):
//!   V6d = COMPUTE_32F + DEFAULT, V6t = COMPUTE_32F + TF32,
//!   V6e = COMPUTE_32F_EMULATED_16BFX9 + DEFAULT (skipped as UNSUPPORTED
//!         where the toolkit refuses it).
//!
//! Input families: `normal` = N(0, 1/sqrt(K)) via Box-Muller (cancellation
//! realism — uniform inputs hide it); `silu` = the same normals passed
//! through x*sigmoid(x) on the activation side (post-SiLU skew, the shape
//! mamba out_proj actually consumes). Real-checkpoint weights are a
//! follow-up lane (needs the HF pull wired into this harness); the
//! decision gate for ANY default flip additionally requires the
//! end-to-end mamba-1.4b greedy-decode check via the P2.6 compute-type
//! plumb, per the plan — this TSV alone never flips a default.
//!
//! Run (GPU box):
//!   cargo test --release --features cuda,gemm-blas --test cublas_compute_probe \
//!     -- --ignored --nocapture
//! Artifact: TSV at $MAMBA_RS_PROBE_TSV (default /tmp/cublas_probe.tsv).
//! Falsification guard: if V2 is clean everywhere on CUDA 13 the run is
//! INCONCLUSIVE about the 12.8 breakage — the version window gets pinned
//! in docs/determinism-benchmarks.md and nothing is flipped.

#![cfg(feature = "cuda")]

use std::ffi::{c_int, c_void};
use std::io::Write as _;
use std::sync::Arc;

use cudarc::cublas::sys as cb;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};

const DISALLOW_RPR: u32 = 16; // CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
const TF32: u32 = 3; // CUBLAS_TF32_TENSOR_OP_MATH

fn math_mode(bits: u32) -> cb::cublasMath_t {
    // repr(u32) in cudarc sys; the OR-combo (TF32|DISALLOW = 19) has no
    // named variant, which is exactly why the transmute exists.
    unsafe { std::mem::transmute::<u32, cb::cublasMath_t>(bits) }
}

fn compute_type(raw: u32) -> cb::cublasComputeType_t {
    unsafe { std::mem::transmute::<u32, cb::cublasComputeType_t>(raw) }
}

const COMPUTE_32F: u32 = 68;
const COMPUTE_32F_PEDANTIC: u32 = 69;
const COMPUTE_64F: u32 = 70;
const COMPUTE_32F_EMULATED_16BFX9: u32 = 78;

struct Cell {
    name: &'static str,
    bf16_inputs: bool,
    compute: u32,
    math: u32,
}

const CELLS: &[Cell] = &[
    Cell { name: "V0_pedantic_default", bf16_inputs: true, compute: COMPUTE_32F_PEDANTIC, math: 0 },
    Cell { name: "V1_pedantic_tf32", bf16_inputs: true, compute: COMPUTE_32F_PEDANTIC, math: TF32 },
    Cell { name: "V2_32f_tf32", bf16_inputs: true, compute: COMPUTE_32F, math: TF32 },
    Cell { name: "V3_32f_default", bf16_inputs: true, compute: COMPUTE_32F, math: 0 },
    Cell { name: "V4_32f_tf32_disallow", bf16_inputs: true, compute: COMPUTE_32F, math: TF32 | DISALLOW_RPR },
    Cell { name: "V5_32f_disallow", bf16_inputs: true, compute: COMPUTE_32F, math: DISALLOW_RPR },
    Cell { name: "V6d_f32_default", bf16_inputs: false, compute: COMPUTE_32F, math: 0 },
    Cell { name: "V6t_f32_tf32", bf16_inputs: false, compute: COMPUTE_32F, math: TF32 },
    Cell { name: "V6e_f32_emulated_bf16x9", bf16_inputs: false, compute: COMPUTE_32F_EMULATED_16BFX9, math: 0 },
];

/// xorshift32 → Box-Muller standard normal, scaled.
struct Rng(u32);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn uniform01(&mut self) -> f32 {
        // (0,1] to keep ln() finite.
        ((self.next_u32() >> 8) as f32 + 1.0) / 16_777_217.0
    }
    fn normal(&mut self, sigma: f32) -> f32 {
        let u1 = self.uniform01();
        let u2 = self.uniform01();
        sigma * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn bf16_bits(x: f32) -> u16 {
    // Round-to-nearest-even truncation to bf16, matching
    // __float2bfloat16_rn (the widen side is exact by construction).
    let b = x.to_bits();
    let rounding = 0x7FFF + ((b >> 16) & 1);
    ((b + rounding) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[allow(clippy::too_many_arguments, reason = "raw probe plumbing: one call site, mirrors the cublasGemmEx parameter list it wraps")]
unsafe fn gemm_row_major(
    handle: cb::cublasHandle_t,
    m: usize,
    n: usize,
    k: usize,
    a_ptr: u64,
    a_type: cb::cudaDataType,
    b_ptr: u64,
    b_type: cb::cudaDataType,
    c_ptr: u64,
    c_type: cb::cudaDataType,
    compute: cb::cublasComputeType_t,
    f64_scalars: bool,
) -> Result<(), String> {
    // Row-major C[M,N] = A[M,K] @ B[K,N] via the column-major identity
    // C^T = B^T A^T: gemm(N, N, n, m, k, B(ld=n), A(ld=k), C(ld=n)) —
    // the same trick blas.rs uses throughout.
    let alpha32: f32 = 1.0;
    let beta32: f32 = 0.0;
    let alpha64: f64 = 1.0;
    let beta64: f64 = 0.0;
    let (alpha, beta): (*const c_void, *const c_void) = if f64_scalars {
        (&alpha64 as *const f64 as *const c_void, &beta64 as *const f64 as *const c_void)
    } else {
        (&alpha32 as *const f32 as *const c_void, &beta32 as *const f32 as *const c_void)
    };
    let st = unsafe {
        cb::cublasGemmEx(
            handle,
            cb::cublasOperation_t::CUBLAS_OP_N,
            cb::cublasOperation_t::CUBLAS_OP_N,
            n as c_int,
            m as c_int,
            k as c_int,
            alpha,
            b_ptr as *const c_void,
            b_type,
            n as c_int,
            a_ptr as *const c_void,
            a_type,
            k as c_int,
            beta,
            c_ptr as *mut c_void,
            c_type,
            n as c_int,
            compute,
            cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    if st != cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("{st:?}"));
    }
    Ok(())
}

struct ShapeBufs {
    a_bf16: CudaSlice<u16>,
    b_bf16: CudaSlice<u16>,
    a_f32: CudaSlice<f32>,
    b_f32: CudaSlice<f32>,
    a_f64: CudaSlice<f64>,
    b_f64: CudaSlice<f64>,
    c_f32: CudaSlice<f32>,
    c_f64: CudaSlice<f64>,
}

#[allow(clippy::too_many_arguments, reason = "probe-local helper: shape + handle + stream + bufs is the irreducible cell context")]
fn run_cell(
    stream: &Arc<CudaStream>,
    handle: cb::cublasHandle_t,
    cell: &Cell,
    m: usize,
    n: usize,
    k: usize,
    bufs: &mut ShapeBufs,
    c_ref: &[f64],
    bit_repeats: usize,
) -> Result<(f64, f64, f64, bool, f64), String> {
    unsafe {
        cb::cublasSetMathMode(handle, math_mode(cell.math));
    }
    let (a_ptr, a_ty, b_ptr, b_ty) = if cell.bf16_inputs {
        (
            bufs.a_bf16.device_ptr(stream).0,
            cb::cudaDataType::CUDA_R_16BF,
            bufs.b_bf16.device_ptr(stream).0,
            cb::cudaDataType::CUDA_R_16BF,
        )
    } else {
        (
            bufs.a_f32.device_ptr(stream).0,
            cb::cudaDataType::CUDA_R_32F,
            bufs.b_f32.device_ptr(stream).0,
            cb::cudaDataType::CUDA_R_32F,
        )
    };
    let c_ptr = bufs.c_f32.device_ptr(stream).0;

    let t0 = std::time::Instant::now();
    unsafe {
        gemm_row_major(
            handle,
            m,
            n,
            k,
            a_ptr,
            a_ty,
            b_ptr,
            b_ty,
            c_ptr,
            cb::cudaDataType::CUDA_R_32F,
            compute_type(cell.compute),
            false,
        )?;
    }
    stream.synchronize().map_err(|e| format!("{e:?}"))?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    let out: Vec<f32> = stream
        .memcpy_dtov(&bufs.c_f32.slice(0..m * n))
        .map_err(|e| format!("{e:?}"))?;
    let mut max_rel = 0f64;
    let mut sum_rel = 0f64;
    let mut max_abs = 0f64;
    for (got, want) in out.iter().zip(c_ref) {
        let g = f64::from(*got);
        let abs = (g - want).abs();
        let rel = abs / want.abs().max(1e-30);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        sum_rel += rel;
    }
    let mean_rel = sum_rel / out.len() as f64;

    let mut bits_stable = true;
    if bit_repeats > 0 {
        let first: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
        for _ in 0..bit_repeats {
            unsafe {
                gemm_row_major(
                    handle,
                    m,
                    n,
                    k,
                    a_ptr,
                    a_ty,
                    b_ptr,
                    b_ty,
                    c_ptr,
                    cb::cudaDataType::CUDA_R_32F,
                    compute_type(cell.compute),
                    false,
                )?;
            }
            stream.synchronize().map_err(|e| format!("{e:?}"))?;
            let rep: Vec<f32> = stream
                .memcpy_dtov(&bufs.c_f32.slice(0..m * n))
                .map_err(|e| format!("{e:?}"))?;
            if rep.iter().map(|v| v.to_bits()).ne(first.iter().copied()) {
                bits_stable = false;
                break;
            }
        }
    }
    Ok((max_rel, mean_rel, max_abs, bits_stable, ms))
}

#[test]
#[ignore = "GPU probe with a TSV artifact; run explicitly on the perf box"]
fn cublas_compute_probe() {
    let tsv_path = std::env::var("MAMBA_RS_PROBE_TSV")
        .unwrap_or_else(|_| "/tmp/cublas_probe.tsv".to_string());
    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();
    let mut raw: cb::cublasHandle_t = std::ptr::null_mut();
    unsafe {
        assert_eq!(
            cb::cublasCreate_v2(&mut raw),
            cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS
        );
        assert_eq!(
            cb::cublasSetStream_v2(raw, stream.cu_stream() as *mut _),
            cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS
        );
        let mut ver: c_int = 0;
        cb::cublasGetVersion_v2(raw, &mut ver);
        eprintln!("cublas version: {ver}");
    }

    // Shape set: the 61325b3 breaker + its K-scaling series, the tied
    // head, and the M/K/N sweep. (M, K, N) row-major.
    let mut shapes: Vec<(usize, usize, usize, &'static str)> = vec![
        (8, 1024, 2048, "breaker_k1024"),
        (8, 4096, 2048, "breaker_k4096"),
        (8, 16384, 2048, "breaker_k16384"),
        (8, 2048, 50304, "tied_head_m8"),
        (512, 2048, 50304, "tied_head_m512"),
    ];
    for &m in &[1usize, 8, 128, 512, 4096] {
        for &k in &[768usize, 1536, 4096, 8192] {
            for &n in &[768usize, 2048, 4096] {
                shapes.push((m, k, n, "sweep"));
            }
        }
    }

    let max_m = shapes.iter().map(|s| s.0).max().unwrap();
    let max_k = shapes.iter().map(|s| s.1).max().unwrap();
    let max_n = shapes.iter().map(|s| s.2).max().unwrap();
    let max_a = max_m * max_k;
    let max_b = max_k * max_n;
    let max_c = max_m * max_n;

    let mut bufs = ShapeBufs {
        a_bf16: stream.alloc_zeros::<u16>(max_a).expect("a16"),
        b_bf16: stream.alloc_zeros::<u16>(max_b).expect("b16"),
        a_f32: stream.alloc_zeros::<f32>(max_a).expect("a32"),
        b_f32: stream.alloc_zeros::<f32>(max_b).expect("b32"),
        a_f64: stream.alloc_zeros::<f64>(max_a).expect("a64"),
        b_f64: stream.alloc_zeros::<f64>(max_b).expect("b64"),
        c_f32: stream.alloc_zeros::<f32>(max_c).expect("c32"),
        c_f64: stream.alloc_zeros::<f64>(max_c).expect("c64"),
    };

    let mut tsv = std::fs::File::create(&tsv_path).expect("tsv create");
    writeln!(tsv, "family\tshape\tM\tK\tN\tcell\tmax_rel\tmean_rel\tmax_abs\tbits_stable\tms").unwrap();

    for family in ["normal", "silu"] {
        for &(m, k, n, tag) in &shapes {
            // Deterministic per-(family, shape) inputs. The PROBE VALUES
            // ARE bf16 (generated, rounded to bf16, widened back) so the
            // f64 reference sees the SAME bits every lane consumes and
            // the f32 lane measures compute error only, never input
            // quantization.
            let sigma = 1.0 / (k as f32).sqrt();
            let mut rng = Rng(0x9E37_79B9 ^ (m as u32) ^ ((k as u32) << 8) ^ ((n as u32) << 16) ^ if family == "silu" { 0x5115 } else { 0 });
            let a_host: Vec<u16> = (0..m * k)
                .map(|_| {
                    let x = rng.normal(sigma);
                    bf16_bits(if family == "silu" { silu(x * (k as f32).sqrt()) * sigma } else { x })
                })
                .collect();
            let b_host: Vec<u16> = (0..k * n).map(|_| bf16_bits(rng.normal(sigma))).collect();
            let a32: Vec<f32> = a_host.iter().map(|&b| bf16_to_f32(b)).collect();
            let b32: Vec<f32> = b_host.iter().map(|&b| bf16_to_f32(b)).collect();
            let a64: Vec<f64> = a32.iter().map(|&x| f64::from(x)).collect();
            let b64: Vec<f64> = b32.iter().map(|&x| f64::from(x)).collect();

            stream.memcpy_htod(&a_host, &mut bufs.a_bf16.slice_mut(0..m * k)).unwrap();
            stream.memcpy_htod(&b_host, &mut bufs.b_bf16.slice_mut(0..k * n)).unwrap();
            stream.memcpy_htod(&a32, &mut bufs.a_f32.slice_mut(0..m * k)).unwrap();
            stream.memcpy_htod(&b32, &mut bufs.b_f32.slice_mut(0..k * n)).unwrap();
            stream.memcpy_htod(&a64, &mut bufs.a_f64.slice_mut(0..m * k)).unwrap();
            stream.memcpy_htod(&b64, &mut bufs.b_f64.slice_mut(0..k * n)).unwrap();

            // fp64 reference on-device (same bits, widened).
            unsafe {
                cb::cublasSetMathMode(raw, math_mode(0));
                gemm_row_major(
                    raw,
                    m,
                    n,
                    k,
                    bufs.a_f64.device_ptr(&stream).0,
                    cb::cudaDataType::CUDA_R_64F,
                    bufs.b_f64.device_ptr(&stream).0,
                    cb::cudaDataType::CUDA_R_64F,
                    bufs.c_f64.device_ptr(&stream).0,
                    cb::cudaDataType::CUDA_R_64F,
                    compute_type(COMPUTE_64F),
                    true,
                )
                .expect("f64 reference gemm");
            }
            stream.synchronize().unwrap();
            let c_ref: Vec<f64> = stream.memcpy_dtov(&bufs.c_f64.slice(0..m * n)).unwrap();

            let bit_repeats = if tag == "sweep" { 0 } else { 20 };
            for cell in CELLS {
                match run_cell(&stream, raw, cell, m, n, k, &mut bufs, &c_ref, bit_repeats) {
                    Ok((max_rel, mean_rel, max_abs, stable, ms)) => {
                        writeln!(
                            tsv,
                            "{family}\t{tag}\t{m}\t{k}\t{n}\t{}\t{max_rel:.3e}\t{mean_rel:.3e}\t{max_abs:.3e}\t{stable}\t{ms:.3}",
                            cell.name
                        )
                        .unwrap();
                    }
                    Err(e) => {
                        writeln!(
                            tsv,
                            "{family}\t{tag}\t{m}\t{k}\t{n}\t{}\tUNSUPPORTED\t{e}\t-\t-\t-",
                            cell.name
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
    unsafe {
        cb::cublasDestroy_v2(raw);
    }
    eprintln!("probe TSV written: {tsv_path}");
}
