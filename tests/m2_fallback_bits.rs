//! The Mamba-2 kernels that take another route past a size give the same
//! bits on either route:
//!
//! - the gated norm streamed for a group too wide to stage, forward and
//!   backward, against the staged kernel,
//! - the chunk state in narrow channel tiles against the whole head,
//! - the fused dx and token sums in register tiles against the dx kernel
//!   and the token sums it replaces,
//! - the conv weight gradient with its taps spread over blocks against one
//!   block,
//! - the sequential backward that reads its states from the replay tape
//!   against the one that holds them in shared memory.
//!
//! Each pair runs on shapes both routes serve, in f32, bf16 and f16.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{y_runs, yz_runs};
use mamba_rs::mamba2::gpu::kernels::{
    CHUNK, Mamba2Kernels, chunk_state_geom, dx_tiles_geom, gated_norm_bwd_smem_bytes,
    gated_norm_cfg, seq_bwd_smem_bytes,
};

const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];

fn det(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s & 0xFF_FFFF) as f32 / 16_777_216.0) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn f32_buf(ctx: &GpuCtx, v: &[f32]) -> GpuBuffer {
    GpuBuffer::from_cpu(&ctx.stream, v).unwrap()
}

fn typed_buf(ctx: &GpuCtx, v: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let b = DtypedBuf::zeros(&ctx.stream, v.len(), dtype).unwrap();
    ctx.stream.synchronize().unwrap();
    b.upload_f32(&ctx.stream, v).unwrap();
    ctx.stream.synchronize().unwrap();
    b
}

fn bits_f32(ctx: &GpuCtx, b: &GpuBuffer) -> Vec<u32> {
    ctx.stream.synchronize().unwrap();
    b.to_cpu(&ctx.stream)
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn bits_typed(ctx: &GpuCtx, b: &DtypedBuf) -> Vec<u32> {
    ctx.stream.synchronize().unwrap();
    let mut v = vec![0f32; b.len_elems()];
    b.download_f32(&ctx.stream, &mut v).unwrap();
    v.iter().map(|x| x.to_bits()).collect()
}

fn setup() -> (GpuDevice, GpuCtx, Mamba2Kernels) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let k = Mamba2Kernels::compile(dev.context(), arch::arch0()).unwrap();
    (dev, ctx, k)
}

/// Group shapes (d_ssm, groups, headdim): more heads than a pass of threads,
/// a head no power of two, two groups.
const NORM_SHAPES: [(usize, usize, usize); 3] = [(2048, 1, 4), (640, 1, 80), (1536, 2, 48)];

#[test]
fn m2_gated_norm_streamed_is_the_staged_kernel() {
    let (_dev, ctx, k) = setup();
    let rows = 3usize;
    for (d_ssm, ng, hd) in NORM_SHAPES {
        for dtype in DTYPES {
            for nbg in [0i32, 1] {
                let (ld_out, out_col, ld_proj, z_off) = (d_ssm + 5, 2i32, d_ssm + 7, 3i32);
                let y = f32_buf(&ctx, &det(rows * d_ssm, 1, 1.0));
                let proj = typed_buf(&ctx, &det(rows * ld_proj, 2, 1.0), dtype);
                let w = f32_buf(&ctx, &det(d_ssm, 3, 1.0));
                let mut outs = Vec::new();
                for cap in [k.smem_optin, 0] {
                    let (staged, cfg) = gated_norm_cfg(rows, (d_ssm, ng, hd), cap);
                    assert_eq!(staged, cap != 0, "the route the cap asks for");
                    let kernel = if staged {
                        &k.gated_norm_fwd
                    } else {
                        &k.gated_norm_fwd_stream
                    };
                    let out = DtypedBuf::zeros(&ctx.stream, rows * ld_out, dtype).unwrap();
                    let (o, yp, pp, wp) = (
                        out.cached_ptr(),
                        y.cached_ptr(),
                        proj.cached_ptr(),
                        w.cached_ptr(),
                    );
                    let (ld_out_l, ld_proj_l, rows_l) =
                        (ld_out as i64, ld_proj as i64, rows as i64);
                    let (d_ssm_i, ng_i, hd_i, eps) = (d_ssm as i32, ng as i32, hd as i32, 1e-5f32);
                    for (y_first, cfg) in y_runs(cfg) {
                        let mut b = ctx.stream.launch_builder(kernel.get(dtype));
                        b.arg(&o);
                        b.arg(&ld_out_l);
                        b.arg(&out_col);
                        b.arg(&yp);
                        b.arg(&pp);
                        b.arg(&ld_proj_l);
                        b.arg(&z_off);
                        b.arg(&wp);
                        b.arg(&rows_l);
                        b.arg(&d_ssm_i);
                        b.arg(&ng_i);
                        b.arg(&hd_i);
                        b.arg(&eps);
                        b.arg(&nbg);
                        b.arg(&y_first);
                        unsafe { b.launch(cfg) }.unwrap();
                    }
                    outs.push(bits_typed(&ctx, &out));
                }
                assert_eq!(
                    outs[0], outs[1],
                    "{dtype:?} d_ssm {d_ssm} groups {ng} head {hd} norm first {nbg}"
                );
            }
        }
    }
}

#[test]
fn m2_gated_norm_backward_streamed_is_the_staged_kernel() {
    let (_dev, ctx, k) = setup();
    let rows = 3usize;
    for (d_ssm, ng, hd) in NORM_SHAPES {
        for dtype in DTYPES {
            for nbg in [0i32, 1] {
                let (ld_out, out_col, ld_proj, z_off) = (d_ssm + 5, 2i32, d_ssm + 7, 3i32);
                let y = f32_buf(&ctx, &det(rows * d_ssm, 4, 1.0));
                let proj = typed_buf(&ctx, &det(rows * ld_proj, 5, 1.0), dtype);
                let dout = typed_buf(&ctx, &det(rows * ld_out, 6, 1.0), dtype);
                let w = f32_buf(&ctx, &det(d_ssm, 7, 1.0));
                let mut runs = Vec::new();
                for staged in [true, false] {
                    let smem = if staged {
                        gated_norm_bwd_smem_bytes(d_ssm, ng, hd)
                    } else {
                        128 * 4
                    };
                    let kernel = if staged {
                        &k.bwd.gated_norm
                    } else {
                        &k.bwd.gated_norm_stream
                    };
                    let dy = GpuBuffer::zeros(&ctx.stream, rows * d_ssm).unwrap();
                    let d_proj = DtypedBuf::zeros(&ctx.stream, rows * ld_proj, dtype).unwrap();
                    let dw = GpuBuffer::zeros(&ctx.stream, rows * d_ssm).unwrap();
                    let ptrs = [
                        dy.cached_ptr(),
                        d_proj.cached_ptr(),
                        dw.cached_ptr(),
                        dout.cached_ptr(),
                        y.cached_ptr(),
                        proj.cached_ptr(),
                        w.cached_ptr(),
                    ];
                    let (ld_out_l, ld_proj_l, rows_l) =
                        (ld_out as i64, ld_proj as i64, rows as i64);
                    let (d_ssm_i, ng_i, hd_i, eps) = (d_ssm as i32, ng as i32, hd as i32, 1e-5f32);
                    let cfg = LaunchConfig {
                        grid_dim: (rows as u32, ng as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: smem as u32,
                    };
                    for (y_first, cfg) in y_runs(cfg) {
                        let mut b = ctx.stream.launch_builder(kernel.get(dtype));
                        b.arg(&ptrs[0]);
                        b.arg(&ptrs[1]);
                        b.arg(&ld_proj_l);
                        b.arg(&z_off);
                        b.arg(&ptrs[2]);
                        b.arg(&ptrs[3]);
                        b.arg(&ld_out_l);
                        b.arg(&out_col);
                        b.arg(&ptrs[4]);
                        b.arg(&ptrs[5]);
                        b.arg(&ptrs[6]);
                        b.arg(&rows_l);
                        b.arg(&d_ssm_i);
                        b.arg(&ng_i);
                        b.arg(&hd_i);
                        b.arg(&eps);
                        b.arg(&nbg);
                        b.arg(&y_first);
                        unsafe { b.launch(cfg) }.unwrap();
                    }
                    runs.push((
                        bits_f32(&ctx, &dy),
                        bits_typed(&ctx, &d_proj),
                        bits_f32(&ctx, &dw),
                    ));
                }
                let label =
                    format!("{dtype:?} d_ssm {d_ssm} groups {ng} head {hd} norm first {nbg}");
                assert_eq!(runs[0].0, runs[1].0, "{label}: dy");
                assert_eq!(runs[0].1, runs[1].1, "{label}: dz");
                assert_eq!(runs[0].2, runs[1].2, "{label}: weight parts");
            }
        }
    }
}

#[test]
fn m2_chunk_state_tiles_are_the_whole_head() {
    let (_dev, ctx, k) = setup();
    let (batch, seq) = (2usize, 100usize);
    for (nh, ng, hd, ds) in [(2usize, 1usize, 40usize, 20usize), (2, 2, 64, 128)] {
        let nc = seq.div_ceil(CHUNK);
        let b_col = nh * hd;
        let ld_xbc = b_col + 2 * ng * ds;
        for dtype in DTYPES {
            let xbc = typed_buf(&ctx, &det(batch * seq * ld_xbc, 8, 1.0), dtype);
            let per = batch * nc * nh * CHUNK;
            let dt = f32_buf(
                &ctx,
                &det(per, 9, 0.5).iter().map(|v| v.abs()).collect::<Vec<_>>(),
            );
            let acs = f32_buf(&ctx, &det(per, 10, 2.0));
            let (cols, whole, _) = chunk_state_geom(hd, ds, k.smem_optin);
            assert_eq!(whole, hd.next_multiple_of(4), "the whole head fits");
            let mut runs = Vec::new();
            for pt in [whole, 8] {
                let states = GpuBuffer::zeros(&ctx.stream, batch * nc * nh * hd * ds).unwrap();
                let (sp, xp, dtp, ap) = (
                    states.cached_ptr(),
                    xbc.cached_ptr(),
                    dt.cached_ptr(),
                    acs.cached_ptr(),
                );
                let ld = ld_xbc as i64;
                let ints = [
                    b_col as i32,
                    batch as i32,
                    seq as i32,
                    nh as i32,
                    ng as i32,
                    hd as i32,
                    ds as i32,
                    pt as i32,
                ];
                let cfg = LaunchConfig {
                    grid_dim: ((batch * nc) as u32, nh as u32, ds.div_ceil(cols) as u32),
                    block_dim: ((32 * pt / 4).min(256) as u32, 1, 1),
                    shared_mem_bytes: (CHUNK * (pt + cols + 1) * 4) as u32,
                };
                for (y_first, cfg) in y_runs(cfg) {
                    let mut b = ctx.stream.launch_builder(k.chunk_state.get(dtype));
                    b.arg(&sp);
                    b.arg(&xp);
                    b.arg(&ld);
                    b.arg(&ints[0]);
                    b.arg(&dtp);
                    b.arg(&ap);
                    for v in &ints[1..] {
                        b.arg(v);
                    }
                    b.arg(&y_first);
                    unsafe { b.launch(cfg) }.unwrap();
                }
                runs.push(bits_f32(&ctx, &states));
            }
            assert_eq!(runs[0], runs[1], "{dtype:?} head {hd} state {ds}");
        }
    }
}

/// The register-tile dx and token sums against `m2b_dx` and `m2b_rowsums`:
/// a head of 4 channels, one no multiple of 4 over two groups, and one of
/// 40 over a state of 24, with D per head and per channel. The tiled kernel
/// serves only heads its block holds, so past that the backward takes the
/// pair, and both give every output the same bits.
#[test]
fn m2_dx_tiles_are_the_dx_and_token_sums() {
    let (_dev, ctx, k) = setup();
    let (batch, seq) = (2usize, 100usize);
    let nc = seq.div_ceil(CHUNK);
    let pairs = CHUNK * (CHUNK + 1) / 2;
    for (nh, ng, hd, ds, per_channel) in [
        (2usize, 1usize, 4usize, 16usize, 0i32),
        (4, 2, 18, 8, 1),
        (2, 1, 40, 24, 1),
    ] {
        let d_ssm = nh * hd;
        let (b_col, c_col) = (d_ssm, d_ssm + ng * ds);
        let ld_xbc = d_ssm + 2 * ng * ds;
        let rows = batch * seq;
        let per = batch * nc * nh * CHUNK;
        let (nk, smem) = dx_tiles_geom(hd, ds, k.smem_optin, k.bwd.dx_tiles_threads)
            .expect("the tiled dx serves this head");
        let dy = f32_buf(&ctx, &det(rows * d_ssm, 21, 1.0));
        let states = f32_buf(&ctx, &det(batch * nc * nh * hd * ds, 22, 1.0));
        let ds_in = f32_buf(&ctx, &det(batch * nc * nh * hd * ds, 23, 1.0));
        let cb = f32_buf(&ctx, &det(batch * nc * ng * pairs, 24, 0.5));
        let dt = f32_buf(
            &ctx,
            &det(per, 25, 0.5)
                .iter()
                .map(|v| v.abs())
                .collect::<Vec<_>>(),
        );
        let acs = f32_buf(&ctx, &det(per, 26, 2.0));
        let d = f32_buf(&ctx, &det(nh * hd, 27, 1.0));
        let ints = |v: &[usize]| -> Vec<i32> { v.iter().map(|&x| x as i32).collect() };
        for dtype in DTYPES {
            let xbc = typed_buf(&ctx, &det(rows * ld_xbc, 28, 1.0), dtype);
            let ld = ld_xbc as i64;
            let (b_col_i, c_col_i) = (b_col as i32, c_col as i32);
            let ptrs = [
                dy.cached_ptr(),
                xbc.cached_ptr(),
                states.cached_ptr(),
                ds_in.cached_ptr(),
                cb.cached_ptr(),
                dt.cached_ptr(),
                acs.cached_ptr(),
                d.cached_ptr(),
            ];
            let mut runs = Vec::new();
            for tiled in [false, true] {
                let dxbc = GpuBuffer::zeros(&ctx.stream, rows * ld_xbc).unwrap();
                let sums: Vec<GpuBuffer> = (0..3)
                    .map(|_| GpuBuffer::zeros(&ctx.stream, per).unwrap())
                    .collect();
                let (dxp, ddtx, qv, inter) = (
                    dxbc.cached_ptr(),
                    sums[0].cached_ptr(),
                    sums[1].cached_ptr(),
                    sums[2].cached_ptr(),
                );
                if tiled {
                    let head = [dxp, ddtx, qv, inter, ptrs[0], ptrs[1]];
                    let cfg = LaunchConfig {
                        grid_dim: ((batch * nc) as u32, nh as u32, 1),
                        block_dim: ((16 * hd.div_ceil(4)) as u32, 1, 1),
                        shared_mem_bytes: smem as u32,
                    };
                    for (y_first, cfg) in y_runs(cfg) {
                        let mut b = ctx.stream.launch_builder(k.bwd.dx_tiles.get(dtype));
                        for p in &head {
                            b.arg(p);
                        }
                        b.arg(&ld);
                        b.arg(&b_col_i);
                        b.arg(&c_col_i);
                        let rest = [ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7]];
                        for p in &rest {
                            b.arg(p);
                        }
                        b.arg(&per_channel);
                        let tail = ints(&[seq, nh, ng, hd, ds, nk]);
                        for v in &tail {
                            b.arg(v);
                        }
                        b.arg(&y_first);
                        unsafe { b.launch(cfg) }.unwrap();
                    }
                } else {
                    let scratch: Vec<GpuBuffer> = (0..3)
                        .map(|_| GpuBuffer::zeros(&ctx.stream, rows * d_ssm).unwrap())
                        .collect();
                    let (acc, ste, yi) = (
                        scratch[0].cached_ptr(),
                        scratch[1].cached_ptr(),
                        scratch[2].cached_ptr(),
                    );
                    let head = [dxp, acc, ste, yi, ptrs[0], ptrs[1]];
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (batch * nc) as u32,
                            nh as u32,
                            (CHUNK * hd).div_ceil(128) as u32,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    for (y_first, z_first, cfg) in yz_runs(cfg) {
                        let mut b = ctx.stream.launch_builder(k.bwd.dx.get(dtype));
                        for p in &head {
                            b.arg(p);
                        }
                        b.arg(&ld);
                        b.arg(&b_col_i);
                        b.arg(&c_col_i);
                        let rest = [ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7]];
                        for p in &rest {
                            b.arg(p);
                        }
                        b.arg(&per_channel);
                        let tail = ints(&[batch, seq, nh, ng, hd, ds]);
                        for v in &tail {
                            b.arg(v);
                        }
                        b.arg(&y_first);
                        b.arg(&z_first);
                        unsafe { b.launch(cfg) }.unwrap();
                    }
                    let sums_head = [ddtx, qv, inter, acc, ste, yi, ptrs[0], ptrs[1]];
                    let cfg = LaunchConfig {
                        grid_dim: ((batch * nc) as u32, nh as u32, 1),
                        block_dim: (CHUNK as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    for (y_first, cfg) in y_runs(cfg) {
                        let mut b = ctx.stream.launch_builder(k.bwd.rowsums.get(dtype));
                        for p in &sums_head {
                            b.arg(p);
                        }
                        b.arg(&ld);
                        b.arg(&ptrs[5]);
                        let tail = ints(&[batch, seq, nh, hd]);
                        for v in &tail {
                            b.arg(v);
                        }
                        b.arg(&y_first);
                        unsafe { b.launch(cfg) }.unwrap();
                    }
                }
                runs.push([
                    bits_f32(&ctx, &dxbc),
                    bits_f32(&ctx, &sums[0]),
                    bits_f32(&ctx, &sums[1]),
                    bits_f32(&ctx, &sums[2]),
                ]);
            }
            for (i, name) in ["dx", "dt token sums", "decay sums", "inter-chunk sums"]
                .iter()
                .enumerate()
            {
                assert!(
                    runs[1][i] == runs[0][i],
                    "{dtype:?} head {hd} state {ds} groups {ng}: {name}"
                );
            }
        }
    }
}

#[test]
fn m2_conv_weight_taps_over_blocks_are_one_block() {
    let (_dev, ctx, k) = setup();
    let (batch, seq, cd, d_conv) = (2usize, 150usize, 72usize, 12usize);
    let (ld_proj, xbc_col) = (cd + 9, 4i32);
    for dtype in DTYPES {
        let dv = f32_buf(&ctx, &det(batch * seq * cd, 11, 1.0));
        let state0 = f32_buf(&ctx, &det(batch * cd * d_conv, 12, 1.0));
        let proj = typed_buf(&ctx, &det(batch * seq * ld_proj, 13, 1.0), dtype);
        let mut runs = Vec::new();
        for warps in [d_conv + 1, 4] {
            let g_w = GpuBuffer::zeros(&ctx.stream, cd * d_conv).unwrap();
            let g_b = GpuBuffer::zeros(&ctx.stream, cd).unwrap();
            let ptrs = [
                g_w.cached_ptr(),
                g_b.cached_ptr(),
                dv.cached_ptr(),
                state0.cached_ptr(),
                proj.cached_ptr(),
            ];
            let ld = ld_proj as i64;
            let ints = [batch as i32, seq as i32, cd as i32, d_conv as i32];
            let mut b = ctx.stream.launch_builder(k.bwd.conv_dw.get(dtype));
            for p in &ptrs {
                b.arg(p);
            }
            b.arg(&ld);
            b.arg(&xbc_col);
            for v in &ints {
                b.arg(v);
            }
            let cfg = LaunchConfig {
                grid_dim: (
                    cd.div_ceil(32) as u32,
                    (d_conv + 1).div_ceil(warps) as u32,
                    1,
                ),
                block_dim: (32, warps as u32, 1),
                shared_mem_bytes: ((128 + 128 + d_conv - 1) * 32 * 4) as u32,
            };
            unsafe { b.launch(cfg) }.unwrap();
            runs.push((bits_f32(&ctx, &g_w), bits_f32(&ctx, &g_b)));
        }
        assert_eq!(
            runs[0], runs[1],
            "{dtype:?}: conv weight and bias gradients"
        );
    }
}

/// Sequential backward shapes (sequence, heads, groups, headdim, states):
/// three segments of 16 tokens over rows of four-float steps, and two
/// segments over rows of 18 states. Both heads leave lanes past the last
/// channel.
const SEQ_SHAPES: [(usize, usize, usize, usize, usize); 2] =
    [(37, 4, 2, 72, 16), (21, 2, 1, 40, 18)];

/// The global sequential backward against m2s_bwd: dS in shared memory and
/// in the global scratch, a block wider than the head and blocks the head is
/// wider than (whose tree runs its upper levels on each thread's own
/// leaves), with and without a per-channel D and a dt limit. The replay
/// tape is compared too.
#[test]
fn m2_sequential_backward_global_is_the_shared_kernel() {
    let (_dev, ctx, k) = setup();
    let s = &ctx.stream;
    let (batch, stride) = (2usize, 16usize);
    for (seq, nh, ng, hd, ds) in SEQ_SHAPES {
        let (rows, nseg, d_ssm) = (batch * seq, seq.div_ceil(stride), nh * hd);
        let conv_dim = d_ssm + 2 * ng * ds;
        let (ld_proj, dt_off) = (conv_dim + nh + 5, conv_dim as i32 + 2);
        let lanes = hd.next_multiple_of(32);
        let shifted = |v: Vec<f32>, by: f32| v.into_iter().map(|x| x + by).collect::<Vec<_>>();
        let dy = f32_buf(&ctx, &det(rows * d_ssm, 21, 1.0));
        let ckpt = f32_buf(&ctx, &det(batch * nseg * nh * hd * ds, 22, 1.0));
        let dt_bias = f32_buf(&ctx, &shifted(det(nh, 23, 0.5), -0.5));
        let a_neg = f32_buf(&ctx, &shifted(det(nh, 24, 0.6), -0.9));
        let d = f32_buf(&ctx, &det(nh * hd, 25, 1.0));
        for dtype in DTYPES {
            let xbc = typed_buf(&ctx, &det(rows * conv_dim, 26, 1.0), dtype);
            let proj = typed_buf(&ctx, &det(rows * ld_proj, 27, 2.0), dtype);
            for (flags, lo, hi) in [(0i32, 0.0f32, f32::INFINITY), (1 | 4, 0.05, 0.4)] {
                // None runs m2s_bwd, Some((threads, dS in shared memory))
                // the global kernel.
                let run = |global: Option<(usize, bool)>| -> Vec<Vec<u32>> {
                    let z = |n: usize| GpuBuffer::zeros(s, n).unwrap();
                    let dxbc = z(rows * conv_dim);
                    let d_proj = DtypedBuf::zeros(s, rows * ld_proj, dtype).unwrap();
                    let (db_h, dc_h) = (z(rows * nh * ds), z(rows * nh * ds));
                    let (da, dbias) = (z(batch * nh), z(batch * nh));
                    let rec = z(batch * nh * stride * hd * ds);
                    let scal = z(rows * nh * 3);
                    let ds_glob = global
                        .filter(|&(_, shared)| !shared)
                        .map(|_| z(batch * nh * hd * ds));
                    let wide = global.is_some_and(|(threads, _)| lanes > threads);
                    let leaves = wide.then(|| z(batch * nh * 2 * lanes));
                    let ptr = |b: &Option<GpuBuffer>| b.as_ref().map_or(0u64, |b| b.cached_ptr());
                    let outs = [
                        dxbc.cached_ptr(),
                        d_proj.cached_ptr(),
                        db_h.cached_ptr(),
                        dc_h.cached_ptr(),
                        da.cached_ptr(),
                        dbias.cached_ptr(),
                        rec.cached_ptr(),
                    ];
                    let (ds_ptr, leaves_ptr) = (ptr(&ds_glob), ptr(&leaves));
                    let ins = [
                        dy.cached_ptr(),
                        xbc.cached_ptr(),
                        proj.cached_ptr(),
                        ckpt.cached_ptr(),
                        scal.cached_ptr(),
                        dt_bias.cached_ptr(),
                        a_neg.cached_ptr(),
                        d.cached_ptr(),
                    ];
                    let ld = ld_proj as i64;
                    let ints = [
                        seq as i32,
                        stride as i32,
                        nh as i32,
                        ng as i32,
                        hd as i32,
                        ds as i32,
                        conv_dim as i32,
                        flags,
                    ];
                    let (block, smem) = match global {
                        None => (lanes, seq_bwd_smem_bytes(hd, ds, lanes)),
                        Some((threads, shared)) => {
                            let kl = if wide { threads } else { lanes };
                            let tile = if shared { hd * (ds + 1) } else { 0 };
                            (threads, (tile + 2 * ds + 2 * kl) * 4)
                        }
                    };
                    if global.is_some() {
                        let rows_l = rows as i64;
                        let mut b = s.launch_builder(k.bwd.seq_scalars.get(dtype));
                        b.arg(&ins[4]);
                        b.arg(&ins[1]);
                        b.arg(&ins[2]);
                        b.arg(&ld);
                        b.arg(&dt_off);
                        b.arg(&ins[5]);
                        b.arg(&ins[6]);
                        b.arg(&rows_l);
                        for v in &ints[2..] {
                            b.arg(v);
                        }
                        b.arg(&lo);
                        b.arg(&hi);
                        let cfg = LaunchConfig {
                            grid_dim: ((rows * nh).div_ceil(256) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        unsafe { b.launch(cfg) }.unwrap();
                    }
                    let kernel = if global.is_some() {
                        &k.bwd.seq_bwd_global
                    } else {
                        &k.bwd.seq_bwd
                    };
                    let cfg = LaunchConfig {
                        grid_dim: (batch as u32, nh as u32, 1),
                        block_dim: (block as u32, 1, 1),
                        shared_mem_bytes: smem as u32,
                    };
                    for (y_first, cfg) in y_runs(cfg) {
                        let mut b = s.launch_builder(kernel.get(dtype));
                        b.arg(&outs[0]);
                        b.arg(&outs[1]);
                        b.arg(&ld);
                        b.arg(&dt_off);
                        for o in &outs[2..] {
                            b.arg(o);
                        }
                        if global.is_some() {
                            b.arg(&ds_ptr);
                            b.arg(&leaves_ptr);
                        }
                        for (i, p) in ins.iter().enumerate() {
                            if i != 4 || global.is_some() {
                                b.arg(p);
                            }
                        }
                        for v in &ints {
                            b.arg(v);
                        }
                        b.arg(&lo);
                        b.arg(&hi);
                        b.arg(&y_first);
                        unsafe { b.launch(cfg) }.unwrap();
                    }
                    vec![
                        bits_f32(&ctx, &dxbc),
                        bits_typed(&ctx, &d_proj),
                        bits_f32(&ctx, &db_h),
                        bits_f32(&ctx, &dc_h),
                        bits_f32(&ctx, &da),
                        bits_f32(&ctx, &dbias),
                        bits_f32(&ctx, &rec),
                    ]
                };
                let want = run(None);
                let names = [
                    "dx",
                    "dt column",
                    "dB",
                    "dC",
                    "dA parts",
                    "dt bias parts",
                    "tape",
                ];
                for threads in [256usize, 64, 32] {
                    for shared in [true, false] {
                        let got = run(Some((threads, shared)));
                        for (i, name) in names.iter().enumerate() {
                            assert!(
                                got[i] == want[i],
                                "{dtype:?} head {hd} states {ds} flags {flags}, block {threads}, \
                                 dS in shared memory {shared}: {name}"
                            );
                        }
                    }
                }
            }
        }
    }
}
