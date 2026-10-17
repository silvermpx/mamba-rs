//! Bit gates for the kernels of the SISO chunked forward, in every dtype.
//!
//! The pre-state (each row's scale, gamma and Q.K dot, the prescaled K
//! rows, the chunk's decay cumsum and its states) runs on the fused kernel
//! where its tile of whole rows fits a block and on the sliced kernel
//! everywhere; the intra-chunk scan likewise on the cooperative kernel and
//! the sliced one. Every arm is held bit for bit to the kernels it
//! replaced (the preprocess, the decay cumsum, the chunk state and the
//! two-head static-tile scan), kept frozen in
//! `fixtures/m3_chunked_forward_reference.cu`.
//!
//! The fused state pass is held bit for bit, in every dtype, to the three
//! launches it replaced (the entering fold, the inter-chunk pass and the
//! window-final writeback), kept frozen in
//! `fixtures/m3_state_pass_reference.cu`.
//!
//! The scan-state arithmetic (the decay cumsum, the inter-chunk pass) stays
//! pure-f32 per the Tri Dao invariant (compounding O(T) prefix arithmetic
//! must not be stored in half precision); only the K and V inputs of the
//! writeback are typed.

#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;
#[path = "common/frozen_kernels.rs"]
mod frozen_kernels;

use cudarc::driver::sys::CUfunction_attribute_enum as CUfunction_attribute;
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use frozen_kernels::frozen_module;
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer, GpuByteBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::kernels::cuda_include_paths;
use mamba_rs::gpu::launch::{y_runs, yz_runs};
use mamba_rs::mamba3_siso::gpu::kernels::{
    ChunkArm, Mamba3Kernels, chunk_fused_cfg, chunk_pre_state_sliced_cfg, chunk_scan_cfg,
    chunk_scan_coop_cfg, chunk_scan_sliced_cfg, state_pass_fused_cfg,
};
use std::sync::Arc;

/// The chunked forward's kernels as they were released before the sliced
/// kernels replaced them.
const FROZEN_FORWARD: &str = include_str!("fixtures/m3_chunked_forward_reference.cu");

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

// ─── the pre-state arms ────────────────────────────────────────────────

/// A chunked forward stage's shape: (batch, t, nh, hd, ds, chunk).
type StageShape = (usize, usize, usize, usize, usize, usize);

/// One pre-state arm's outputs on the device: K_scaled; qk_dot, scale,
/// gamma and the chunk states; the decay cumsum.
type StageOutputs = (DtypedBuf, [GpuBuffer; 4], GpuBuffer);

/// The shapes of each state capacity's module the stage tests run: one
/// partial chunk, rows that end mid-chunk, heads narrower than a warp and
/// wider than a block of the sliced kernels, state widths that are not a
/// whole number of quads, and shapes past every whole-row tile, the
/// official head among them.
const STAGE_SHAPES: [(usize, &[StageShape]); 6] = [
    (
        16,
        &[
            (2, 1, 3, 5, 4, 32),
            (1, 13, 2, 7, 8, 64),
            (2, 64, 3, 16, 16, 64),
            (2, 70, 2, 9, 13, 64),
        ],
    ),
    (32, &[(1, 65, 3, 5, 32, 32), (1, 70, 2, 24, 30, 64)]),
    (
        64,
        &[
            (2, 129, 2, 40, 4, 32),
            (1, 13, 2, 3, 64, 32),
            (2, 130, 3, 32, 64, 64),
            (1, 100, 2, 70, 50, 64),
            (1, 70, 2, 130, 64, 64),
        ],
    ),
    (96, &[(2, 70, 2, 96, 96, 64)]),
    (
        128,
        &[
            (2, 129, 2, 64, 128, 64),
            (1, 100, 3, 100, 128, 64),
            (1, 77, 3, 16, 128, 64),
        ],
    ),
    (256, &[(1, 70, 2, 64, 256, 64), (1, 64, 1, 256, 256, 64)]),
];

/// The sequence ids of `batch` rows of `t` positions: the carried sequence,
/// then a new one at a different position in every row and another 50
/// positions later, so chunks begin inside, at and across a start.
fn packed_ids(batch: usize, t: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(batch * t);
    for b in 0..batch {
        let first = 37 + 11 * b;
        ids.extend((0..t).map(|i| i32::from(i >= first) + i32::from(i >= first + 50)));
    }
    ids
}

/// A packing of `batch` rows of `t` positions on the device.
fn packing(ctx: &GpuCtx, batch: usize, t: usize) -> GpuByteBuffer {
    let bytes: Vec<u8> = packed_ids(batch, t)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut buf = GpuByteBuffer::zeros(&ctx.stream, bytes.len()).unwrap();
    buf.upload_bytes(&ctx.stream, &bytes).unwrap();
    buf
}

/// The block a frozen two-head scalar kernel launches for a head: every
/// lane a thread of the head's elements, at most half the threads the
/// kernel's registers allow, whole warps.
fn frozen_lanes(f: &CudaFunction, hd: usize) -> usize {
    let threads = f
        .get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        .unwrap() as usize;
    hd.min((threads / 2 / 32 * 32).max(32))
}

fn reference_name(base: &str, dtype: WeightDtype) -> String {
    match dtype {
        WeightDtype::Bf16 => format!("{base}_bf16"),
        WeightDtype::F16 => format!("{base}_f16"),
        WeightDtype::F32 | WeightDtype::Tf32 => base.to_owned(),
    }
}

/// The fused and the sliced pre-state against the frozen trio they replace
/// (the preprocess, the decay cumsum and the chunk state), bit for bit in
/// every dtype, with and without a packing and with either save omitted,
/// the trio's chunk states in each of its layouts: the scalar one and,
/// where its tile fits, the quad one. The sliced kernel runs at every
/// shape; the fused one wherever its tile of whole rows fits the block's
/// budget.
#[test]
fn the_pre_state_arms_write_the_frozen_trio_bits() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    for (cap, shapes) in STAGE_SHAPES {
        let m3k = Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), cap)
            .unwrap();
        let frozen = frozen_module(
            ctx.stream.context(),
            arch::arch0(),
            Some(cap),
            FROZEN_FORWARD,
        );
        let cumsum = frozen.load_function("m3_reference_da_cumsum").unwrap();
        for &(batch, t, nh, hd, ds, cs) in shapes {
            let nc = t.div_ceil(cs);
            let th_n = batch * t * nh;
            let ks_n = th_n * ds;
            let states_n = batch * nc * nh * hd * ds;
            let da_n = batch * nc * nh * cs;
            let fused_cfg = chunk_fused_cfg(batch, nc, nh, hd, ds, cs, m3k.smem_optin);
            let sliced_cfg = chunk_pre_state_sliced_cfg(batch, nc, nh, hd);
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
            // The a*dt rows every arm folds into the decay cumsum.
            let mut adt_vals = vec![0.0f32; th_n];
            for b in 0..batch {
                for chunk in 0..nc {
                    for h in 0..nh {
                        let chunk_len = cs.min(t - chunk * cs);
                        for tl in 0..chunk_len {
                            let mut a = -(0.01 + 0.003 * ((tl + 3 * h + b + 5 * chunk) % 7) as f32);
                            // Earlier steps underflow after the midpoint drop;
                            // later steps still have ordinary decay.
                            if chunk_len >= 4 && tl == chunk_len / 2 {
                                a -= 160.0;
                            }
                            // A rising last step makes the endpoint difference
                            // positive, exercising the cap at zero.
                            if chunk_len >= 3 && tl == chunk_len - 1 {
                                a = 0.125;
                            }
                            adt_vals[(b * t + chunk * cs + tl) * nh + h] = a;
                        }
                    }
                }
            }
            let adt = upload_f32(&ctx, &adt_vals);
            let ids = packing(&ctx, batch, t);
            let dims = [batch, t, nh, hd, ds, cs].map(|v| v as i32);
            let pre_dims = [dims[0], dims[1], dims[2], dims[4], dims[5]];
            let cumsum_dims = [dims[0], dims[1], dims[2], dims[5]];
            for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
                let k = upload_typed(&ctx, &det_rand(ks_n, 0x8C41), dtype);
                let q = upload_typed(&ctx, &det_rand(ks_n, 0x8C42), dtype);
                let x = upload_typed(&ctx, &det_rand(th_n * hd, 0x8C43), dtype);
                let preprocess = frozen
                    .load_function(&reference_name("m3_reference_preprocess_chunks", dtype))
                    .unwrap();
                let state = frozen
                    .load_function(&reference_name("m3_reference_chunk_state", dtype))
                    .unwrap();
                // The trio's chunk states in the scalar layout, and in the
                // quad one where its tile fits a block without an opt-in.
                let lanes = frozen_lanes(&state, hd);
                let mut layouts = vec![(
                    "scalar layout",
                    LaunchConfig {
                        grid_dim: ((batch * nc) as u32, nh.div_ceil(2) as u32, 1),
                        block_dim: (lanes as u32, 2, 1),
                        shared_mem_bytes: 0,
                    },
                )];
                let quad_smem = 2 * (cs * hd + cs * ds + cs) * 4;
                let quad_threads = state
                    .get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                    .unwrap() as usize;
                if ds.is_multiple_of(4)
                    && hd * (ds / 4) * 2 <= quad_threads
                    && quad_smem <= 48 * 1024
                {
                    layouts.push((
                        "quad layout",
                        LaunchConfig {
                            grid_dim: ((batch * nc) as u32, nh.div_ceil(2) as u32, 1),
                            block_dim: (hd as u32, (ds / 4) as u32, 2),
                            shared_mem_bytes: quad_smem as u32,
                        },
                    ));
                }
                for packed in [false, true] {
                    let seq_idx = if packed { ids.cached_ptr() } else { 0 };
                    // The outputs of one arm: K_scaled, qk_dot, scale, gamma,
                    // the chunk states and the decay cumsum. Each arm writes
                    // into its own poisoned buffers, so an element no arm
                    // writes cannot agree; the cumsum's padding past a chunk's
                    // end stays NaN in every arm.
                    let outputs = |poison: f32| {
                        let ks = upload_typed(&ctx, &vec![poison; ks_n], dtype);
                        let rows = [th_n, th_n, th_n, states_n]
                            .map(|len| upload_f32(&ctx, &vec![poison; len]));
                        let da = upload_f32(&ctx, &vec![f32::NAN; da_n]);
                        (ks, rows, da)
                    };
                    let download = |out: &StageOutputs| {
                        let (ks, rows, da) = out;
                        [
                            download_typed(&ctx, ks),
                            download_f32(&ctx, &rows[0], th_n),
                            download_f32(&ctx, &rows[1], th_n),
                            download_f32(&ctx, &rows[2], th_n),
                            download_f32(&ctx, &rows[3], states_n),
                            download_f32(&ctx, da, da_n),
                        ]
                    };
                    let run_frozen = |layout: LaunchConfig| {
                        let out = outputs(8192.0);
                        let (ks, rows, da) = &out;
                        let block_x = nh.min(256) as u32;
                        let cumsum_cfg = LaunchConfig {
                            grid_dim: (
                                batch as u32,
                                nc as u32,
                                nh.div_ceil(block_x as usize) as u32,
                            ),
                            block_dim: (block_x, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        for (y_first, z_first, cfg) in yz_runs(cumsum_cfg) {
                            let mut bld = ctx.stream.launch_builder(&cumsum);
                            let (dap, adtp) = (da.cached_ptr(), adt.cached_ptr());
                            bld.arg(&dap);
                            bld.arg(&adtp);
                            for dim in &cumsum_dims {
                                bld.arg(dim);
                            }
                            bld.arg(&y_first);
                            bld.arg(&z_first);
                            bld.arg(&seq_idx);
                            unsafe { bld.launch(cfg) }.unwrap();
                        }
                        let ptrs = [
                            ks.cached_ptr(),
                            rows[0].cached_ptr(),
                            rows[1].cached_ptr(),
                            rows[2].cached_ptr(),
                            k.cached_ptr(),
                            q.cached_ptr(),
                            dt.cached_ptr(),
                            trap.cached_ptr(),
                        ];
                        let pre_cfg = LaunchConfig {
                            grid_dim: ((batch * nc) as u32, nh as u32, 1),
                            block_dim: (cs as u32, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        for (y_first, cfg) in y_runs(pre_cfg) {
                            let mut bld = ctx.stream.launch_builder(&preprocess);
                            for ptr in &ptrs {
                                bld.arg(ptr);
                            }
                            for dim in &pre_dims {
                                bld.arg(dim);
                            }
                            bld.arg(&y_first);
                            bld.arg(&seq_idx);
                            unsafe { bld.launch(cfg) }.unwrap();
                        }
                        let ptrs = [
                            rows[3].cached_ptr(),
                            x.cached_ptr(),
                            ks.cached_ptr(),
                            da.cached_ptr(),
                        ];
                        for (y_first, cfg) in y_runs(layout) {
                            let mut bld = ctx.stream.launch_builder(&state);
                            for ptr in &ptrs {
                                bld.arg(ptr);
                            }
                            for dim in &dims {
                                bld.arg(dim);
                            }
                            bld.arg(&y_first);
                            bld.arg(&seq_idx);
                            unsafe { bld.launch(cfg) }.unwrap();
                        }
                        ctx.stream.synchronize().unwrap();
                        download(&out)
                    };
                    let run_arm = |kernel: &CudaFunction,
                                   cfg: LaunchConfig,
                                   save_scale: bool,
                                   save_gamma: bool| {
                        let out = outputs(-8192.0);
                        let (ks, rows, da) = &out;
                        let ptrs = [
                            ks.cached_ptr(),
                            rows[0].cached_ptr(),
                            if save_scale { rows[1].cached_ptr() } else { 0 },
                            if save_gamma { rows[2].cached_ptr() } else { 0 },
                            rows[3].cached_ptr(),
                            k.cached_ptr(),
                            q.cached_ptr(),
                            dt.cached_ptr(),
                            trap.cached_ptr(),
                            x.cached_ptr(),
                            adt.cached_ptr(),
                            da.cached_ptr(),
                        ];
                        for (y_first, cfg) in y_runs(cfg) {
                            let mut bld = ctx.stream.launch_builder(kernel);
                            for ptr in &ptrs {
                                bld.arg(ptr);
                            }
                            for dim in &dims {
                                bld.arg(dim);
                            }
                            bld.arg(&y_first);
                            bld.arg(&seq_idx);
                            unsafe { bld.launch(cfg) }.unwrap();
                        }
                        ctx.stream.synchronize().unwrap();
                        download(&out)
                    };
                    let mut arms = vec![(
                        "sliced",
                        m3k.m3_chunk_pre_state_sliced_typed.i32.get(dtype),
                        sliced_cfg,
                    )];
                    if let Some(cfg) = fused_cfg {
                        arms.push((
                            "fused",
                            m3k.m3_chunk_pre_state_fused_typed.i32.get(dtype),
                            cfg,
                        ));
                    }
                    let names = [
                        "K_scaled",
                        "qk_dot",
                        "scale",
                        "gamma",
                        "chunk_states",
                        "dA_cumsum",
                    ];
                    for &(layout, layout_cfg) in &layouts {
                        let reference = run_frozen(layout_cfg);
                        for (arm, kernel, cfg) in &arms {
                            for (save_scale, save_gamma) in
                                [(true, true), (false, true), (true, false), (false, false)]
                            {
                                let actual = run_arm(kernel, *cfg, save_scale, save_gamma);
                                let label = format!(
                                    "{dtype:?} cap {cap} B {batch} T {t} nh {nh} hd {hd} ds {ds} \
                                     cs {cs} packed {packed}: {arm} against the trio's {layout}, \
                                     save_scale {save_scale} save_gamma {save_gamma}"
                                );
                                for (out, ((got, want), name)) in
                                    actual.iter().zip(&reference).zip(names).enumerate()
                                {
                                    let skipped =
                                        (out == 2 && !save_scale) || (out == 3 && !save_gamma);
                                    for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
                                        let want = if skipped { -8192.0f32 } else { want };
                                        assert_eq!(
                                            got.to_bits(),
                                            want.to_bits(),
                                            "{label}: {name}[{i}]"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ─── the intra-chunk scan arms ─────────────────────────────────────────

/// The sliced scan against the frozen two-head static-tile scan it
/// replaces, bit for bit in every dtype, with and without a packing, at
/// every shape: the static kernel's arithmetic, its last sums in the form
/// the card's compiler gives them, at shapes on both sides of the
/// cooperative kernel's tile.
#[test]
fn the_sliced_scan_writes_the_frozen_static_tile_bits() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    for (cap, shapes) in STAGE_SHAPES {
        let m3k = Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), cap)
            .unwrap();
        let frozen = frozen_module(
            ctx.stream.context(),
            arch::arch0(),
            Some(cap),
            FROZEN_FORWARD,
        );
        for &(batch, t, nh, hd, ds, cs) in shapes {
            let nc = t.div_ceil(cs);
            let th_n = batch * t * nh;
            let sliced_cfg = chunk_scan_sliced_cfg(batch, nc, nh, hd);
            let x_vals = det_rand(th_n * hd, 0x5C01);
            let q_vals = det_rand(th_n * ds, 0x5C02);
            let k_vals = det_rand(th_n * ds, 0x5C03);
            let qk = upload_f32(&ctx, &det_rand(th_n, 0x5C04));
            // Decay cumsums falling from zero inside every chunk.
            let mut da_vals = Vec::with_capacity(batch * nc * nh * cs);
            let steps = det_rand(batch * nc * nh * cs, 0x5C05);
            for row in steps.chunks(cs) {
                let mut acc = 0.0f32;
                da_vals.extend(row.iter().map(|v| {
                    acc -= 0.02 + 0.1 * v.abs();
                    acc
                }));
            }
            let da = upload_f32(&ctx, &da_vals);
            let prev = upload_f32(&ctx, &det_rand(batch * nc * nh * hd * ds, 0x5C06));
            let d_vals: Vec<f32> = (0..nh).map(|i| 0.5 + 0.1 * i as f32).collect();
            let d = upload_f32(&ctx, &d_vals);
            let ids = packing(&ctx, batch, t);
            for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
                let x = upload_typed(&ctx, &x_vals, dtype);
                let q = upload_typed(&ctx, &q_vals, dtype);
                let k = upload_typed(&ctx, &k_vals, dtype);
                let frozen_scan = frozen
                    .load_function(&reference_name("m3_reference_chunk_scan", dtype))
                    .unwrap();
                let lanes = frozen_lanes(&frozen_scan, hd);
                let frozen_cfg = LaunchConfig {
                    grid_dim: ((batch * nc) as u32, nh.div_ceil(2) as u32, 1),
                    block_dim: (lanes as u32, 2, 1),
                    shared_mem_bytes: 0,
                };
                for packed in [false, true] {
                    let seq_idx = if packed { ids.cached_ptr() } else { 0 };
                    // Each arm writes into its own poisoned rows, so an
                    // element no arm writes cannot agree.
                    let run = |kernel: &CudaFunction, cfg: LaunchConfig, poison: f32| {
                        let y = upload_typed(&ctx, &vec![poison; th_n * hd], dtype);
                        let ptrs = [
                            y.cached_ptr(),
                            x.cached_ptr(),
                            q.cached_ptr(),
                            k.cached_ptr(),
                            qk.cached_ptr(),
                            da.cached_ptr(),
                            prev.cached_ptr(),
                            d.cached_ptr(),
                        ];
                        let dims = [batch, t, nh, hd, ds, cs].map(|v| v as i32);
                        for (y_first, cfg) in y_runs(cfg) {
                            let mut bld = ctx.stream.launch_builder(kernel);
                            for ptr in &ptrs {
                                bld.arg(ptr);
                            }
                            for dim in &dims {
                                bld.arg(dim);
                            }
                            bld.arg(&y_first);
                            bld.arg(&seq_idx);
                            unsafe { bld.launch(cfg) }.unwrap();
                        }
                        ctx.stream.synchronize().unwrap();
                        download_typed(&ctx, &y)
                    };
                    let want = run(&frozen_scan, frozen_cfg, 8192.0);
                    let got = run(
                        m3k.m3_chunk_scan_fwd_sliced_typed.i32.get(dtype),
                        sliced_cfg,
                        -8192.0,
                    );
                    let label = format!(
                        "{dtype:?} cap {cap} B {batch} T {t} nh {nh} hd {hd} ds {ds} packed {packed}"
                    );
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(g.to_bits(), w.to_bits(), "{label}: y[{i}]");
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

// ─── the cooperative scan's class ──────────────────────────────────────

/// The cooperative scan keeps the shapes whose tile fits 48 KB and no more:
/// it rounds the output's last terms differently from the static-tile
/// kernel whose arithmetic the sliced scan keeps, so a shape past that tile
/// (a head of 100 over 24 states, 58 KB) runs sliced and keeps its bits,
/// and the serve shape stays cooperative.
#[test]
fn the_cooperative_scan_keeps_its_48_kb_shapes() {
    assert!(chunk_scan_coop_cfg(4, 8, 24, 100, 24, 64).is_none());
    assert!(chunk_scan_coop_cfg(1, 73, 48, 16, 16, 64).is_some());
    assert_eq!(chunk_scan_cfg(4, 8, 24, 100, 24, 64).0, ChunkArm::Sliced);
    assert_eq!(chunk_scan_cfg(1, 73, 48, 16, 16, 64).0, ChunkArm::Whole);
}
