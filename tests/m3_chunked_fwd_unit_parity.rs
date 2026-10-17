//! per-kernel parity tests for the typed M3 chunked parallel forward
//! kernels: `m3_preprocess_chunks`, `m3_chunk_state_fwd` and
//! `m3_chunk_scan_fwd`. Each typed variant is compared against its f32
//! oracle on identical random inputs. The fused preprocess/state path also
//! has exact same-dtype comparisons against the unfused pair, including its
//! intermediate outputs.
//!
//! The fused state pass is held bit for bit, in every dtype, to the three
//! launches it replaced (the entering fold, the inter-chunk pass and the
//! window-final writeback), kept frozen in
//! `fixtures/m3_state_pass_reference.cu`.
//!
//! The scan-state arithmetic (`m3_dA_cumsum`, the inter-chunk pass) stays
//! pure-f32 per the Tri Dao invariant (compounding O(T) prefix arithmetic
//! must not be stored in half precision); only the K and V inputs of the
//! writeback are typed.

#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use cudarc::driver::{CudaModule, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::kernels::{cuda_include_paths, state_capacity};
use mamba_rs::gpu::launch::{IndexWidth, y_runs, yz_runs};
use mamba_rs::mamba3_siso::gpu::kernels::{
    ChunkLimits, Mamba3Kernels, chunk_fused_cfg, chunk_scan_cfg, chunk_state_cfg,
    state_pass_fused_cfg,
};
use std::sync::Arc;

// Small config — T=10, CS=4 forces 3 chunks with the LAST chunk PARTIAL (len=2).
// Previously T=8 CS=4 left the partial-last-chunk path of the typed kernels
// (zero-pad of smem tiles, chunk_len = min(CS, T-chunk_start)) untested at
// bf16/f16 rounding boundary.
const B: usize = 2;
const T: usize = 10;
const NH: usize = 2;
const HD: usize = 4;
const DS: usize = 4;
const CHUNK_SIZE: usize = 4;
const N_CHUNKS: usize = T.div_ceil(CHUNK_SIZE);
const D_INNER: usize = NH * HD;

fn det_rand(n: usize, seed: u32) -> Vec<f32> {
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

fn cos_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        1.0
    };
    let ratio = if na > 0.0 {
        ((nb.sqrt()) / (na.sqrt())) as f32
    } else {
        1.0
    };
    (cos, ratio)
}

fn assert_close(label: &str, a: &[f32], b: &[f32], cos_min: f32, norm_tol: f32) {
    let (cos, ratio) = cos_norm(a, b);
    eprintln!("  {label}: cos={cos:.6} norm={ratio:.4} n={}", a.len());
    assert!(cos >= cos_min, "{label}: cos {cos} < {cos_min}");
    assert!(
        (ratio - 1.0).abs() <= norm_tol,
        "{label}: norm {ratio} outside [1 ± {norm_tol}]"
    );
}

fn upload_typed(ctx: &GpuCtx, data: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let buf = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    buf.upload_f32(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    buf
}

fn download_typed(ctx: &GpuCtx, buf: &DtypedBuf) -> Vec<f32> {
    let mut out = vec![0f32; buf.len_elems()];
    buf.download_f32(&ctx.stream, &mut out).unwrap();
    ctx.stream.synchronize().unwrap();
    out
}

fn upload_f32(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    ctx.stream.synchronize().unwrap();
    b.upload(&ctx.stream, data).unwrap();
    ctx.stream.synchronize().unwrap();
    b
}

fn download_f32(ctx: &GpuCtx, buf: &GpuBuffer, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    buf.download(&ctx.stream, &mut v).unwrap();
    ctx.stream.synchronize().unwrap();
    v
}

fn make_m3k(ctx: &GpuCtx) -> Mamba3Kernels {
    Mamba3Kernels::compile(ctx.stream.context(), arch::arch0()).unwrap()
}

/// The frozen three-launch state pass, compiled for the production
/// module's target with its numeric options.
fn state_pass_reference(ctx: &GpuCtx) -> Arc<CudaModule> {
    let source = [
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("fixtures/m3_state_pass_reference.cu"),
    ]
    .join("\n");
    let (mut major, mut minor) = (0, 0);
    let status = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    assert_eq!(status, cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS);
    let mut options = vec![
        "--fmad=true".to_owned(),
        "--extra-device-vectorization".to_owned(),
        "-DNDEBUG".to_owned(),
    ];
    if (major, minor) >= (12, 9) {
        options.push("--frandom-seed=1295203121".to_owned());
    }
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        &source,
        cudarc::nvrtc::CompileOptions {
            arch: Some(arch::arch0()),
            options,
            include_paths: cuda_include_paths(),
            ..Default::default()
        },
    )
    .expect("state pass reference compilation");
    ctx.stream
        .context()
        .load_module(ptx)
        .expect("load state pass reference")
}

fn tolerances(dtype: WeightDtype) -> (f32, f32) {
    match dtype {
        WeightDtype::Bf16 => (0.99, 0.05),
        WeightDtype::F16 => (0.999, 0.02),
        WeightDtype::F32 | WeightDtype::Tf32 => unreachable!(),
    }
}

// ─── m3_preprocess_chunks ──────────────────────────────────────────────

fn check_preprocess_chunks(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    // Shared random inputs (same values for both f32 and typed runs).
    let k_vals = det_rand(B * T * NH * DS, 0x8C01);
    let q_vals = det_rand(B * T * NH * DS, 0x8C02);
    // DT must be positive (post-softplus), trap_sig in (0,1) (post-sigmoid).
    let dt_vals: Vec<f32> = (0..B * T * NH)
        .map(|i| 0.1 + 0.01 * (i as f32 % 5.0))
        .collect();
    let trap_vals: Vec<f32> = (0..B * T * NH)
        .map(|i| 0.3 + 0.02 * (i as f32 % 7.0))
        .collect();

    let k_f32 = upload_f32(&ctx, &k_vals);
    let q_f32 = upload_f32(&ctx, &q_vals);
    let dt_buf = upload_f32(&ctx, &dt_vals);
    let trap_buf = upload_f32(&ctx, &trap_vals);

    let ks_n = B * T * NH * DS;
    let th_n = B * T * NH;

    // f32 oracle.
    let ks_f32 = GpuBuffer::zeros(&ctx.stream, ks_n).unwrap();
    let qk_f32 = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    let scale_f32 = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    let gamma_f32 = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    ctx.stream.synchronize().unwrap();

    let cfg = LaunchConfig {
        grid_dim: ((B * N_CHUNKS) as u32, NH as u32, 1),
        block_dim: (CHUNK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bi, ti, nhi, dsi, csi) = (B as i32, T as i32, NH as i32, DS as i32, CHUNK_SIZE as i32);
    for (y_first, cfg) in y_runs(cfg) {
        let mut bld = ctx.stream.launch_builder(&m3k.m3_preprocess_chunks.i32);
        let (a, b, c, d, e, f, g, h) = (
            ks_f32.cached_ptr(),
            qk_f32.cached_ptr(),
            scale_f32.cached_ptr(),
            gamma_f32.cached_ptr(),
            k_f32.cached_ptr(),
            q_f32.cached_ptr(),
            dt_buf.cached_ptr(),
            trap_buf.cached_ptr(),
        );
        bld.arg(&a);
        bld.arg(&b);
        bld.arg(&c);
        bld.arg(&d);
        bld.arg(&e);
        bld.arg(&f);
        bld.arg(&g);
        bld.arg(&h);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&y_first);
        bld.arg(&0u64);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    let ks_ref = download_f32(&ctx, &ks_f32, ks_n);
    let qk_ref = download_f32(&ctx, &qk_f32, th_n);
    let scale_ref = download_f32(&ctx, &scale_f32, th_n);
    let gamma_ref = download_f32(&ctx, &gamma_f32, th_n);

    // Typed variant.
    let k_t = upload_typed(&ctx, &k_vals, dtype);
    let q_t = upload_typed(&ctx, &q_vals, dtype);
    let ks_t = DtypedBuf::zeros(&ctx.stream, ks_n, dtype).unwrap();
    let qk_t = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    let scale_t = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    let gamma_t = GpuBuffer::zeros(&ctx.stream, th_n).unwrap();
    ctx.stream.synchronize().unwrap();

    for (y_first, cfg) in y_runs(cfg) {
        let mut bld = ctx
            .stream
            .launch_builder(m3k.m3_preprocess_chunks_typed.i32.get(dtype));
        let (a, b, c, d, e, f, g, h) = (
            ks_t.cached_ptr(),
            qk_t.cached_ptr(),
            scale_t.cached_ptr(),
            gamma_t.cached_ptr(),
            k_t.cached_ptr(),
            q_t.cached_ptr(),
            dt_buf.cached_ptr(),
            trap_buf.cached_ptr(),
        );
        bld.arg(&a);
        bld.arg(&b);
        bld.arg(&c);
        bld.arg(&d);
        bld.arg(&e);
        bld.arg(&f);
        bld.arg(&g);
        bld.arg(&h);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&y_first);
        bld.arg(&0u64);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();

    let ks_got = download_typed(&ctx, &ks_t);
    let qk_got = download_f32(&ctx, &qk_t, th_n);
    let scale_got = download_f32(&ctx, &scale_t, th_n);
    let gamma_got = download_f32(&ctx, &gamma_t, th_n);

    eprintln!("m3_preprocess_chunks {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    // scale/gamma derive from f32 DT/trap_sig only → bit-exact.
    assert_close("scale (f32)", &scale_ref, &scale_got, 1.0 - 1e-6, 1e-4);
    assert_close("gamma (f32)", &gamma_ref, &gamma_got, 1.0 - 1e-6, 1e-4);
    // qk_dot = Σ(Q*K) * gamma, reduced from typed Q/K → dtype-bounded drift.
    assert_close("qk_dot", &qk_ref, &qk_got, cos_min, norm_tol);
    // K_scaled goes through bf16/f16 round-trip.
    assert_close("K_scaled", &ks_ref, &ks_got, cos_min, norm_tol);
}

#[test]
fn m3_preprocess_chunks_bf16() {
    check_preprocess_chunks(WeightDtype::Bf16);
}
#[test]
fn m3_preprocess_chunks_f16() {
    check_preprocess_chunks(WeightDtype::F16);
}

// ─── m3_chunk_state_fwd ────────────────────────────────────────────────

fn check_chunk_state_fwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let x_vals = det_rand(B * T * D_INNER, 0x8C11);
    let ks_vals = det_rand(B * T * NH * DS, 0x8C12);
    // dA_cumsum values — negative (decay)
    let da_vals: Vec<f32> = (0..B * N_CHUNKS * NH * CHUNK_SIZE)
        .map(|i| -0.02 * (i as f32 % 4.0 + 1.0))
        .collect();

    let x_f32 = upload_f32(&ctx, &x_vals);
    let ks_f32_buf = upload_f32(&ctx, &ks_vals);
    let da_buf = upload_f32(&ctx, &da_vals);

    let states_n = B * N_CHUNKS * NH * HD * DS;

    // f32 oracle.
    let states_f32 = GpuBuffer::zeros(&ctx.stream, states_n).unwrap();
    ctx.stream.synchronize().unwrap();
    let cfg = LaunchConfig {
        grid_dim: ((B * N_CHUNKS) as u32, NH as u32, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bi, ti, nhi, hdi, dsi, csi) = (
        B as i32,
        T as i32,
        NH as i32,
        HD as i32,
        DS as i32,
        CHUNK_SIZE as i32,
    );
    for (y_first, cfg) in y_runs(cfg) {
        let mut bld = ctx.stream.launch_builder(&m3k.m3_chunk_state_fwd.i32);
        let (a, b, c, d) = (
            states_f32.cached_ptr(),
            x_f32.cached_ptr(),
            ks_f32_buf.cached_ptr(),
            da_buf.cached_ptr(),
        );
        bld.arg(&a);
        bld.arg(&b);
        bld.arg(&c);
        bld.arg(&d);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&y_first);
        bld.arg(&0u64);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();
    let ref_states = download_f32(&ctx, &states_f32, states_n);

    // Typed variant.
    let x_t = upload_typed(&ctx, &x_vals, dtype);
    let ks_t = upload_typed(&ctx, &ks_vals, dtype);
    let states_t = GpuBuffer::zeros(&ctx.stream, states_n).unwrap();
    ctx.stream.synchronize().unwrap();
    for (y_first, cfg) in y_runs(cfg) {
        let mut bld = ctx
            .stream
            .launch_builder(m3k.m3_chunk_state_fwd_typed.i32.get(dtype));
        let (a, b, c, d) = (
            states_t.cached_ptr(),
            x_t.cached_ptr(),
            ks_t.cached_ptr(),
            da_buf.cached_ptr(),
        );
        bld.arg(&a);
        bld.arg(&b);
        bld.arg(&c);
        bld.arg(&d);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&y_first);
        bld.arg(&0u64);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();
    let got_states = download_f32(&ctx, &states_t, states_n);

    eprintln!("m3_chunk_state_fwd {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    assert_close("states (f32)", &ref_states, &got_states, cos_min, norm_tol);
}

#[test]
fn m3_chunk_state_fwd_bf16() {
    check_chunk_state_fwd(WeightDtype::Bf16);
}
#[test]
fn m3_chunk_state_fwd_f16() {
    check_chunk_state_fwd(WeightDtype::F16);
}

/// The fused pre-state kernel against the unfused trio it replaces, bit
/// for bit in every dtype and with either save omitted, the trio's chunk
/// states in each of its layouts: the scalar one it runs wherever the quad
/// tile does not fit, and the quad one where it does.
#[test]
fn m3_chunk_pre_state_fused_matches_unfused_bits() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    // The hd=40 shape gives each fused thread multiple state quads to own.
    // The last three are past the 48 KB default, the official head among
    // them: the budget moved them from the unfused trio's scalar layout to
    // the fused kernel.
    let shapes: [(usize, usize, usize, usize, usize, usize); 9] = [
        (2, 1, 3, 5, 4, 32),
        (1, 13, 2, 7, 8, 64),
        (2, 64, 3, 16, 16, 64),
        (1, 65, 3, 5, 32, 32),
        (2, 129, 2, 40, 4, 32),
        (1, 13, 2, 3, 64, 32),
        (2, 70, 2, 96, 96, 64),
        (2, 129, 2, 64, 128, 64),
        (1, 100, 3, 100, 128, 64),
    ];
    for cap in [16, 32, 64, 96, 128] {
        let m3k = Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), cap)
            .unwrap();
        for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            for (batch, t, nh, hd, ds, cs) in shapes {
                let cap = m3k.state_cap;
                // Match production's tight capacity, and retain the wider-module
                // oracle coverage for every shape supported by the default cap.
                if ds > cap || (cap != 64 && state_capacity(ds).unwrap() != cap) {
                    continue;
                }
                let label =
                    format!("{dtype:?} cap={cap} B={batch} T={t} nh={nh} hd={hd} ds={ds} cs={cs}");
                let nc = t.div_ceil(cs);
                let fused_cfg = chunk_fused_cfg(batch, nc, nh, hd, ds, cs, m3k.smem_optin)
                    .unwrap_or_else(|| panic!("{label}: fused launch must be supported"));
                let limits = m3k.chunk_limits(IndexWidth::I32);
                let state_cfg = chunk_state_cfg(batch, nc, nh, hd, ds, cs, limits);
                // The scalar layout, which the unfused trio runs wherever the
                // quad tile does not fit.
                let scalar_cfg =
                    chunk_state_cfg(batch, nc, nh, hd, ds, cs, ChunkLimits { smem: 0, ..limits });
                assert!(ds <= m3k.state_cap, "{label}: state capacity");
                for cfg in [&fused_cfg, &state_cfg, &scalar_cfg] {
                    let (bx, by, bz) = cfg.block_dim;
                    assert!((1..=1024).contains(&(bx * by * bz)), "{label}: block");
                    assert!(
                        cfg.shared_mem_bytes as usize <= m3k.smem_optin,
                        "{label}: shared memory"
                    );
                }
                let th_n = batch * t * nh;
                let ks_n = th_n * ds;
                let states_n = batch * nc * nh * hd * ds;
                // Both arms consume the same typed inputs. The unfused arm
                // materializes K_scaled in its activation dtype before the fold.
                let k = upload_typed(&ctx, &det_rand(ks_n, 0x8C41), dtype);
                let q = upload_typed(&ctx, &det_rand(ks_n, 0x8C42), dtype);
                let x = upload_typed(&ctx, &det_rand(th_n * hd, 0x8C43), dtype);
                let dt_vals: Vec<f32> = det_rand(th_n, 0x8C44)
                    .into_iter()
                    .map(|v| 0.01 + v.abs() * 0.4)
                    .collect();
                let trap_vals: Vec<f32> = det_rand(th_n, 0x8C45)
                    .into_iter()
                    .map(|v| 0.5 + v * 0.9)
                    .collect();
                let dt = upload_f32(&ctx, &dt_vals);
                let trap = upload_f32(&ctx, &trap_vals);
                // The a*dt rows both arms fold into the decay cumsum.
                let mut adt_vals = vec![0.0f32; th_n];
                for b in 0..batch {
                    for chunk in 0..nc {
                        for h in 0..nh {
                            let chunk_len = cs.min(t - chunk * cs);
                            for tl in 0..chunk_len {
                                let mut a =
                                    -(0.01 + 0.003 * ((tl + 3 * h + b + 5 * chunk) % 7) as f32);
                                // Earlier timesteps underflow after the midpoint
                                // drop; later timesteps still have ordinary decay.
                                if chunk_len >= 4 && tl == chunk_len / 2 {
                                    a -= 160.0;
                                }
                                // A rising last step makes the endpoint
                                // difference positive, exercising fminf.
                                if chunk_len >= 3 && tl == chunk_len - 1 {
                                    a = 0.125;
                                }
                                adt_vals[(b * t + chunk * cs + tl) * nh + h] = a;
                            }
                        }
                    }
                }
                let adt = upload_f32(&ctx, &adt_vals);
                let da_n = batch * nc * nh * cs;
                let dims = [batch, t, nh, hd, ds, cs].map(|v| v as i32);
                let pre_dims = [dims[0], dims[1], dims[2], dims[4], dims[5]];
                let pre_cfg = LaunchConfig {
                    grid_dim: ((batch * nc) as u32, nh as u32, 1),
                    block_dim: (cs as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                // The fused kernel for `None`, the unfused trio with its chunk
                // states in the layout of `Some(cfg)` otherwise.
                let run = |unfused: Option<LaunchConfig>, save_scale: bool, save_gamma: bool| {
                    // Independent, differently poisoned allocations expose missing
                    // writes even if both kernels skip the same output element.
                    let poison = if unfused.is_none() { -8192.0 } else { 8192.0 };
                    let ks = upload_typed(&ctx, &vec![poison; ks_n], dtype);
                    let [qk, scale, gamma, states] = [th_n, th_n, th_n, states_n]
                        .map(|len| upload_f32(&ctx, &vec![poison; len]));
                    // Padding past a chunk's length stays NaN in both arms, so
                    // the endpoint must use the actual chunk length.
                    let da = upload_f32(&ctx, &vec![f32::NAN; da_n]);
                    let scale_ptr = if save_scale { scale.cached_ptr() } else { 0 };
                    let gamma_ptr = if save_gamma { gamma.cached_ptr() } else { 0 };
                    if let Some(state_cfg) = unfused {
                        let block_x = nh.min(256) as u32;
                        let da_cfg = LaunchConfig {
                            grid_dim: (
                                batch as u32,
                                nc as u32,
                                nh.div_ceil(block_x as usize) as u32,
                            ),
                            block_dim: (block_x, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        for (y_first, z_first, cfg) in yz_runs(da_cfg) {
                            let mut bld = ctx.stream.launch_builder(&m3k.m3_da_cumsum.i32);
                            let (da_ptr, adt_ptr) = (da.cached_ptr(), adt.cached_ptr());
                            bld.arg(&da_ptr);
                            bld.arg(&adt_ptr);
                            let da_dims = [dims[0], dims[1], dims[2], dims[5]];
                            for dim in &da_dims {
                                bld.arg(dim);
                            }
                            bld.arg(&y_first);
                            bld.arg(&z_first);
                            bld.arg(&0u64);
                            unsafe { bld.launch(cfg) }.unwrap();
                        }
                        let ptrs = [
                            ks.cached_ptr(),
                            qk.cached_ptr(),
                            scale_ptr,
                            gamma_ptr,
                            k.cached_ptr(),
                            q.cached_ptr(),
                            dt.cached_ptr(),
                            trap.cached_ptr(),
                        ];
                        for (y_first, cfg) in y_runs(pre_cfg) {
                            let mut bld = ctx
                                .stream
                                .launch_builder(m3k.m3_preprocess_chunks_typed.i32.get(dtype));
                            for ptr in &ptrs {
                                bld.arg(ptr);
                            }
                            for dim in &pre_dims {
                                bld.arg(dim);
                            }
                            bld.arg(&y_first);
                            bld.arg(&0u64);
                            unsafe { bld.launch(cfg) }.unwrap();
                        }
                        let ptrs = [
                            states.cached_ptr(),
                            x.cached_ptr(),
                            ks.cached_ptr(),
                            da.cached_ptr(),
                        ];
                        for (y_first, cfg) in y_runs(state_cfg) {
                            let mut bld = ctx
                                .stream
                                .launch_builder(m3k.m3_chunk_state_fwd_typed.i32.get(dtype));
                            for ptr in &ptrs {
                                bld.arg(ptr);
                            }
                            for dim in &dims {
                                bld.arg(dim);
                            }
                            bld.arg(&y_first);
                            bld.arg(&0u64);
                            unsafe { bld.launch(cfg) }.unwrap();
                        }
                    } else {
                        let ptrs = [
                            ks.cached_ptr(),
                            qk.cached_ptr(),
                            scale_ptr,
                            gamma_ptr,
                            states.cached_ptr(),
                            k.cached_ptr(),
                            q.cached_ptr(),
                            dt.cached_ptr(),
                            trap.cached_ptr(),
                            x.cached_ptr(),
                            adt.cached_ptr(),
                            da.cached_ptr(),
                        ];
                        for (y_first, cfg) in y_runs(fused_cfg) {
                            let mut bld = ctx
                                .stream
                                .launch_builder(m3k.m3_chunk_pre_state_fused_typed.i32.get(dtype));
                            for ptr in &ptrs {
                                bld.arg(ptr);
                            }
                            for dim in &dims {
                                bld.arg(dim);
                            }
                            bld.arg(&y_first);
                            bld.arg(&0u64);
                            unsafe { bld.launch(cfg) }.unwrap();
                        }
                    }
                    ctx.stream.synchronize().unwrap();
                    [
                        download_typed(&ctx, &ks),
                        download_f32(&ctx, &qk, th_n),
                        download_f32(&ctx, &scale, th_n),
                        download_f32(&ctx, &gamma, th_n),
                        download_f32(&ctx, &states, states_n),
                        download_f32(&ctx, &da, da_n),
                    ]
                };
                let names = [
                    "K_scaled",
                    "qk_dot",
                    "scale",
                    "gamma",
                    "chunk_states",
                    "dA_cumsum",
                ];
                let mut layouts = vec![("scalar layout", scalar_cfg)];
                if state_cfg.shared_mem_bytes > 0 {
                    layouts.push(("quad layout", state_cfg));
                }
                for (layout, layout_cfg) in layouts {
                    let reference = run(Some(layout_cfg), true, true);
                    for (save_scale, save_gamma) in
                        [(true, true), (false, true), (true, false), (false, false)]
                    {
                        let actual = run(None, save_scale, save_gamma);
                        for (out, ((got, want), name)) in
                            actual.iter().zip(&reference).zip(names).enumerate()
                        {
                            assert_eq!(got.len(), want.len(), "{label}: {name} length");
                            for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
                                let want = if (out == 2 && !save_scale) || (out == 3 && !save_gamma)
                                {
                                    -8192.0f32
                                } else {
                                    // The cumsum's padding is NaN in both arms.
                                    assert!(
                                        want.is_finite() || out == 5,
                                        "{label}: {name}[{i}] oracle nonfinite"
                                    );
                                    want
                                };
                                // Widening finite half values is exact, including signed zero.
                                assert_eq!(
                                    got.to_bits(),
                                    want.to_bits(),
                                    "{label} {layout}: {name}[{i}] save_scale={save_scale} \
                                     save_gamma={save_gamma}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// ─── m3_state_pass_fused ───────────────────────────────────────────────

/// The fused state pass against the three launches it replaced, the
/// entering fold, the inter-chunk pass and the window-final writeback,
/// each launched whole from the frozen reference. Both arms take the same
/// dtype, so every output must agree bit for bit:
/// the entering state of every chunk, the exit SSM state and the K and V
/// boundaries. The seeded arm passes the persistent states as their own
/// seed, as a continued prefill does, so a write racing a read shows up.
#[test]
fn m3_state_pass_fused_matches_separate_bits() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);
    let reference = state_pass_reference(&ctx);
    let load = |name: &str| reference.load_function(name).expect(name);
    let (entering, passing) = (
        load("m3_reference_entering_state"),
        load("m3_reference_state_passing"),
    );
    // (batch, t, nh, hd, ds, cs): a partial last chunk, a single partial
    // chunk, the classifier head, a state wider than one 1024-thread pass,
    // and a state that is not a whole number of warps.
    let shapes: [(usize, usize, usize, usize, usize, usize); 5] = [
        (2, 10, 2, 4, 4, 4),
        (1, 3, 3, 16, 16, 64),
        (2, 64, 2, 16, 16, 64),
        (2, 129, 2, 32, 64, 32),
        (1, 70, 3, 5, 7, 16),
    ];
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        let writeback = load(match dtype {
            WeightDtype::Bf16 => "m3_reference_writeback_bf16",
            WeightDtype::F16 => "m3_reference_writeback_f16",
            _ => "m3_reference_writeback_f32",
        });
        for (batch, t, nh, hd, ds, cs) in shapes {
            let nc = t.div_ceil(cs);
            let dim = hd * ds;
            let th_n = batch * t * nh;
            let states_n = batch * nc * nh * dim;
            let ssm_n = batch * nh * dim;
            let contrib = det_rand(states_n, 0x5A01);
            // Negative decay increments, so the chunk decays stay in (0, 1].
            let da_vals: Vec<f32> = det_rand(batch * nc * nh * cs, 0x5A02)
                .into_iter()
                .map(|v| -v.abs() * 2.0)
                .collect();
            let dt_vals: Vec<f32> = det_rand(th_n, 0x5A03)
                .into_iter()
                .map(|v| 0.01 + v.abs() * 0.4)
                .collect();
            let trap_vals: Vec<f32> = det_rand(th_n, 0x5A04)
                .into_iter()
                .map(|v| 0.5 + v * 0.9)
                .collect();
            let ssm0 = det_rand(ssm_n, 0x5A05);
            let k0 = det_rand(batch * nh * ds, 0x5A06);
            let v0 = det_rand(batch * nh * hd, 0x5A07);
            let da = upload_f32(&ctx, &da_vals);
            let dt = upload_f32(&ctx, &dt_vals);
            let trap = upload_f32(&ctx, &trap_vals);
            let k_flat = upload_typed(&ctx, &det_rand(th_n * ds, 0x5A08), dtype);
            let x_flat = upload_typed(&ctx, &det_rand(th_n * hd, 0x5A09), dtype);
            let (bi, nci, nhi, hdi, dsi, csi, ti) = (
                batch as i32,
                nc as i32,
                nh as i32,
                hd as i32,
                ds as i32,
                cs as i32,
                t as i32,
            );
            for seeded in [false, true] {
                let label = format!(
                    "{dtype:?} B={batch} T={t} nh={nh} hd={hd} ds={ds} cs={cs} seeded={seeded}"
                );
                // The three launches: the fold into a copy of the SSM state,
                // the pass, then the writeback from the pass's final states.
                let states_a = upload_f32(&ctx, &contrib);
                let final_a = upload_f32(&ctx, &vec![f32::NAN; ssm_n]);
                let ssm_a = upload_f32(&ctx, &ssm0);
                let k_a = upload_f32(&ctx, &k0);
                let v_a = upload_f32(&ctx, &v0);
                let init = upload_f32(&ctx, &ssm0);
                let init_ptr = if seeded {
                    let block_x = dim.min(256) as u32;
                    let cfg = LaunchConfig {
                        grid_dim: (
                            batch as u32,
                            nh as u32,
                            dim.div_ceil(block_x as usize) as u32,
                        ),
                        block_dim: (block_x, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let ptrs = [
                        init.cached_ptr(),
                        k_a.cached_ptr(),
                        v_a.cached_ptr(),
                        dt.cached_ptr(),
                        trap.cached_ptr(),
                    ];
                    let mut bld = ctx.stream.launch_builder(&entering);
                    for ptr in &ptrs {
                        bld.arg(ptr);
                    }
                    for arg in [&bi, &nhi, &hdi, &dsi, &ti] {
                        bld.arg(arg);
                    }
                    unsafe { bld.launch(cfg) }.unwrap();
                    init.cached_ptr()
                } else {
                    0
                };
                {
                    let block_x = dim.min(256) as u32;
                    let cfg = LaunchConfig {
                        grid_dim: (
                            batch as u32,
                            nh as u32,
                            dim.div_ceil(block_x as usize) as u32,
                        ),
                        block_dim: (block_x, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let ptrs = [
                        states_a.cached_ptr(),
                        final_a.cached_ptr(),
                        da.cached_ptr(),
                        init_ptr,
                    ];
                    let mut bld = ctx.stream.launch_builder(&passing);
                    for ptr in &ptrs {
                        bld.arg(ptr);
                    }
                    for arg in [&bi, &nci, &nhi, &hdi, &dsi, &csi, &ti] {
                        bld.arg(arg);
                    }
                    unsafe { bld.launch(cfg) }.unwrap();
                }
                {
                    let cfg = LaunchConfig {
                        grid_dim: (batch as u32, nh as u32, 1),
                        block_dim: (hd.max(ds) as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let ptrs = [
                        ssm_a.cached_ptr(),
                        k_a.cached_ptr(),
                        v_a.cached_ptr(),
                        final_a.cached_ptr(),
                        k_flat.cached_ptr(),
                        x_flat.cached_ptr(),
                    ];
                    let mut bld = ctx.stream.launch_builder(&writeback);
                    for ptr in &ptrs {
                        bld.arg(ptr);
                    }
                    for arg in [&bi, &ti, &nhi, &hdi, &dsi] {
                        bld.arg(arg);
                    }
                    unsafe { bld.launch(cfg) }.unwrap();
                }
                // The fused launch, its seed the persistent states themselves.
                let states_b = upload_f32(&ctx, &contrib);
                let ssm_b = upload_f32(&ctx, &ssm0);
                let k_b = upload_f32(&ctx, &k0);
                let v_b = upload_f32(&ctx, &v0);
                {
                    let seed = if seeded { ssm_b.cached_ptr() } else { 0 };
                    let ptrs = [
                        states_b.cached_ptr(),
                        ssm_b.cached_ptr(),
                        k_b.cached_ptr(),
                        v_b.cached_ptr(),
                        seed,
                        k_b.cached_ptr(),
                        v_b.cached_ptr(),
                        dt.cached_ptr(),
                        trap.cached_ptr(),
                        da.cached_ptr(),
                        k_flat.cached_ptr(),
                        x_flat.cached_ptr(),
                    ];
                    let cfg = state_pass_fused_cfg(batch, nh, hd, ds);
                    let (bx, by, bz) = cfg.block_dim;
                    assert!((1..=1024).contains(&(bx * by * bz)), "{label}: block");
                    for (y_first, cfg) in y_runs(cfg) {
                        let mut bld = ctx
                            .stream
                            .launch_builder(m3k.m3_state_pass_fused_typed.i32.get(dtype));
                        for ptr in &ptrs {
                            bld.arg(ptr);
                        }
                        for arg in [&bi, &nci, &nhi, &hdi, &dsi, &csi, &ti] {
                            bld.arg(arg);
                        }
                        bld.arg(&y_first);
                        bld.arg(&0u64);
                        unsafe { bld.launch(cfg) }.unwrap();
                    }
                }
                ctx.stream.synchronize().unwrap();
                let pairs = [
                    (
                        "entering states",
                        download_f32(&ctx, &states_a, states_n),
                        download_f32(&ctx, &states_b, states_n),
                    ),
                    (
                        "ssm_state",
                        download_f32(&ctx, &ssm_a, ssm_n),
                        download_f32(&ctx, &ssm_b, ssm_n),
                    ),
                    (
                        "k_state",
                        download_f32(&ctx, &k_a, batch * nh * ds),
                        download_f32(&ctx, &k_b, batch * nh * ds),
                    ),
                    (
                        "v_state",
                        download_f32(&ctx, &v_a, batch * nh * hd),
                        download_f32(&ctx, &v_b, batch * nh * hd),
                    ),
                ];
                for (name, want, got) in &pairs {
                    for (i, (w, g)) in want.iter().zip(got).enumerate() {
                        assert!(w.is_finite(), "{label}: {name}[{i}] oracle nonfinite");
                        assert_eq!(g.to_bits(), w.to_bits(), "{label}: {name}[{i}]");
                    }
                }
            }
        }
    }
}

// ─── m3_chunk_scan_fwd ─────────────────────────────────────────────────

fn check_chunk_scan_fwd(dtype: WeightDtype) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = make_m3k(&ctx);

    let x_vals = det_rand(B * T * D_INNER, 0x8C31);
    let q_vals = det_rand(B * T * NH * DS, 0x8C32);
    let ks_vals = det_rand(B * T * NH * DS, 0x8C33);
    let qk_vals = det_rand(B * T * NH, 0x8C34);
    let da_vals: Vec<f32> = (0..B * N_CHUNKS * NH * CHUNK_SIZE)
        .map(|i| -0.02 * (i as f32 % 4.0 + 1.0))
        .collect();
    let prev_vals = det_rand(B * N_CHUNKS * NH * HD * DS, 0x8C35);
    let d_vals: Vec<f32> = (0..NH).map(|i| 0.5 + 0.1 * (i as f32)).collect();

    let x_f32 = upload_f32(&ctx, &x_vals);
    let q_f32 = upload_f32(&ctx, &q_vals);
    let ks_f32_buf = upload_f32(&ctx, &ks_vals);
    let qk_buf = upload_f32(&ctx, &qk_vals);
    let da_buf = upload_f32(&ctx, &da_vals);
    let prev_buf = upload_f32(&ctx, &prev_vals);
    let d_buf = upload_f32(&ctx, &d_vals);

    let y_n = B * T * D_INNER;

    // f32 oracle.
    let y_f32 = GpuBuffer::zeros(&ctx.stream, y_n).unwrap();
    ctx.stream.synchronize().unwrap();
    let cfg = LaunchConfig {
        grid_dim: ((B * N_CHUNKS) as u32, NH as u32, 1),
        block_dim: (HD as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bi, ti, nhi, hdi, dsi, csi) = (
        B as i32,
        T as i32,
        NH as i32,
        HD as i32,
        DS as i32,
        CHUNK_SIZE as i32,
    );
    for (y_first, cfg) in y_runs(cfg) {
        let mut bld = ctx.stream.launch_builder(&m3k.m3_chunk_scan_fwd.i32);
        let (a, b, c, d, e, f, g, h) = (
            y_f32.cached_ptr(),
            x_f32.cached_ptr(),
            q_f32.cached_ptr(),
            ks_f32_buf.cached_ptr(),
            qk_buf.cached_ptr(),
            da_buf.cached_ptr(),
            prev_buf.cached_ptr(),
            d_buf.cached_ptr(),
        );
        bld.arg(&a);
        bld.arg(&b);
        bld.arg(&c);
        bld.arg(&d);
        bld.arg(&e);
        bld.arg(&f);
        bld.arg(&g);
        bld.arg(&h);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&y_first);
        bld.arg(&0u64);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();
    let y_ref = download_f32(&ctx, &y_f32, y_n);

    // Typed variant.
    let x_t = upload_typed(&ctx, &x_vals, dtype);
    let q_t = upload_typed(&ctx, &q_vals, dtype);
    let ks_t = upload_typed(&ctx, &ks_vals, dtype);
    let y_t = DtypedBuf::zeros(&ctx.stream, y_n, dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    for (y_first, cfg) in y_runs(cfg) {
        let mut bld = ctx
            .stream
            .launch_builder(m3k.m3_chunk_scan_fwd_typed.i32.get(dtype));
        let (a, b, c, d, e, f, g, h) = (
            y_t.cached_ptr(),
            x_t.cached_ptr(),
            q_t.cached_ptr(),
            ks_t.cached_ptr(),
            qk_buf.cached_ptr(),
            da_buf.cached_ptr(),
            prev_buf.cached_ptr(),
            d_buf.cached_ptr(),
        );
        bld.arg(&a);
        bld.arg(&b);
        bld.arg(&c);
        bld.arg(&d);
        bld.arg(&e);
        bld.arg(&f);
        bld.arg(&g);
        bld.arg(&h);
        bld.arg(&bi);
        bld.arg(&ti);
        bld.arg(&nhi);
        bld.arg(&hdi);
        bld.arg(&dsi);
        bld.arg(&csi);
        bld.arg(&y_first);
        bld.arg(&0u64);
        unsafe { bld.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();
    let y_got = download_typed(&ctx, &y_t);

    eprintln!("m3_chunk_scan_fwd {dtype:?}:");
    let (cos_min, norm_tol) = tolerances(dtype);
    assert_close("y_out", &y_ref, &y_got, cos_min, norm_tol);
}

#[test]
fn m3_chunk_scan_fwd_bf16() {
    check_chunk_scan_fwd(WeightDtype::Bf16);
}
#[test]
fn m3_chunk_scan_fwd_f16() {
    check_chunk_scan_fwd(WeightDtype::F16);
}

// ─── the cooperative scan's class ──────────────────────────────────────

/// The cooperative scan keeps the shapes whose tile fits 48 KB, whatever
/// the block's budget: it rounds the output's last terms differently from
/// the static-tile kernel, so a shape between 48 KB and the budget (a head
/// of 100 over 24 states, 58 KB) stays on the static tiles and keeps its
/// bits, and the serve shape stays cooperative.
#[test]
fn the_cooperative_scan_keeps_its_48_kb_shapes_under_any_budget() {
    for smem in [64 * 1024, 99 * 1024, 227 * 1024] {
        let limits = ChunkLimits {
            threads: 1024,
            smem,
        };
        let (coop, _) = chunk_scan_cfg(4, 8, 24, 100, 24, 64, limits);
        assert!(
            !coop,
            "budget {smem}: hd 100 ds 24 stays on the static tiles"
        );
        let (coop, _) = chunk_scan_cfg(1, 73, 48, 16, 16, 64, limits);
        assert!(coop, "budget {smem}: the serve shape stays cooperative");
    }
}
