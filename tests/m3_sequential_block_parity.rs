//! The sequential scan's block-tier training kernels against the warp
//! kernels they extend.
//!
//! On every head of one warp the block-tier kernels must give the warp
//! kernels' bits: the f32 forward, backward and target forward, and the
//! bf16 and f16 forward. The half kernels compute in f32 after widening
//! their inputs, so on inputs the half type holds exactly they give the
//! f32 kernel's bits at any width, and the target network's forward the
//! training forward's. That holds at state widths the f32 warp kernels have
//! no constant-width version of: those versions fold the skip term into
//! the first output step, which the half kernels do not. Wider heads, which
//! only the block tier serves, are checked against the CPU by the training
//! parity suite.

#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::{IndexWidth, y_runs};
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::route::{
    block_tier_threads, sequential_backward_block_smem_bytes, sequential_forward_block_smem_bytes,
};

const B: usize = 2;
const NH: usize = 2;

fn det(n: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            lo + (hi - lo) * ((s & 0xFFFF) as f32 / 65536.0)
        })
        .collect()
}

fn upload(ctx: &GpuCtx, data: &[f32]) -> GpuBuffer {
    let mut b = GpuBuffer::zeros(&ctx.stream, data.len()).unwrap();
    b.upload(&ctx.stream, data).unwrap();
    b
}

fn upload_typed(ctx: &GpuCtx, data: &[f32], dtype: WeightDtype) -> DtypedBuf {
    let b = DtypedBuf::zeros(&ctx.stream, data.len(), dtype).unwrap();
    b.upload_f32(&ctx.stream, data).unwrap();
    b
}

fn download(ctx: &GpuCtx, buf: &GpuBuffer) -> Vec<f32> {
    ctx.stream.synchronize().unwrap();
    buf.to_cpu(&ctx.stream).unwrap()
}

/// The value the half type stores for `v`, widened back.
fn round_to(dtype: WeightDtype, v: &[f32]) -> Vec<f32> {
    match dtype {
        WeightDtype::Bf16 => v
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_f32())
            .collect(),
        WeightDtype::F16 => v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect(),
        WeightDtype::F32 | WeightDtype::Tf32 => v.to_vec(),
    }
}

fn assert_bits(label: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{label}: lengths");
    let differ: Vec<usize> = want
        .iter()
        .zip(got)
        .enumerate()
        .filter(|(_, (w, g))| w.to_bits() != g.to_bits())
        .map(|(i, _)| i)
        .collect();
    if let Some(&i) = differ.first() {
        panic!(
            "{label}: {} of {} elements differ, first at {i}: want {:e}, got {:e}",
            differ.len(),
            want.len(),
            want[i],
            got[i]
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    hd: usize,
    ds: usize,
    t: usize,
}

impl Shape {
    fn dims(&self) -> [i32; 5] {
        [B, self.t, NH, self.hd, self.ds].map(|v| v as i32)
    }
    fn warp_cfg(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (B as u32, NH as u32, 1),
            block_dim: (self.hd as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
    fn block_cfg(&self, smem: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (B as u32, NH as u32, 1),
            block_dim: (block_tier_threads(self.hd) as u32, 1, 1),
            shared_mem_bytes: smem as u32,
        }
    }
}

/// The host values of one layer's sequential scan. Activations hold the
/// values the half type stores when `dtype` is one, so the f32 kernels see
/// exactly what the half kernels widen.
struct Values {
    x: Vec<f32>,
    k: Vec<f32>,
    q: Vec<f32>,
    d_y: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    gamma: Vec<f32>,
    d_param: Vec<f32>,
    ssm: Vec<f32>,
    k_state: Vec<f32>,
    v_state: Vec<f32>,
}

impl Values {
    fn new(s: Shape, dtype: WeightDtype) -> Self {
        let seed = (s.hd * 131 + s.ds * 17 + s.t) as u32;
        let (bt, di) = (B * s.t, NH * s.hd);
        Self {
            x: round_to(dtype, &det(bt * di, seed ^ 1, -1.0, 1.0)),
            k: round_to(dtype, &det(bt * NH * s.ds, seed ^ 2, -1.0, 1.0)),
            q: round_to(dtype, &det(bt * NH * s.ds, seed ^ 3, -1.0, 1.0)),
            d_y: round_to(dtype, &det(bt * di, seed ^ 4, -1.0, 1.0)),
            alpha: det(bt * NH, seed ^ 5, 0.5, 0.99),
            beta: det(bt * NH, seed ^ 6, 0.0, 0.2),
            gamma: det(bt * NH, seed ^ 7, 0.0, 0.2),
            d_param: det(NH, seed ^ 8, -0.5, 0.5),
            ssm: det(B * NH * s.hd * s.ds, seed ^ 9, -0.5, 0.5),
            k_state: det(B * NH * s.ds, seed ^ 10, -0.5, 0.5),
            v_state: det(B * NH * s.hd, seed ^ 11, -0.5, 0.5),
        }
    }
}

/// What the forward leaves behind.
struct Forward {
    y: Vec<f32>,
    h_saved: Vec<f32>,
    k_prev: Vec<f32>,
    v_prev: Vec<f32>,
    ssm: Vec<f32>,
    k_state: Vec<f32>,
    v_state: Vec<f32>,
}

/// One forward launch: the f32 kernels when `dtype` is f32, the half ones
/// otherwise, with the saves or without.
fn forward(
    ctx: &GpuCtx,
    (kernel, cfg): (&CudaFunction, LaunchConfig),
    (s, dtype, save): (Shape, WeightDtype, bool),
    v: &Values,
) -> Forward {
    let (bt, di) = (B * s.t, NH * s.hd);
    let ssm = upload(ctx, &v.ssm);
    let k_state = upload(ctx, &v.k_state);
    let v_state = upload(ctx, &v.v_state);
    let y = DtypedBuf::zeros(&ctx.stream, bt * di, dtype).unwrap();
    let h_saved = GpuBuffer::zeros(&ctx.stream, B * (s.t + 1) * di * s.ds).unwrap();
    let k_prev = GpuBuffer::zeros(&ctx.stream, bt * NH * s.ds).unwrap();
    let v_prev = GpuBuffer::zeros(&ctx.stream, bt * di).unwrap();
    let x = upload_typed(ctx, &v.x, dtype);
    let k = upload_typed(ctx, &v.k, dtype);
    let q = upload_typed(ctx, &v.q, dtype);
    let (alpha, beta, gamma) = (
        upload(ctx, &v.alpha),
        upload(ctx, &v.beta),
        upload(ctx, &v.gamma),
    );
    let d_param = upload(ctx, &v.d_param);
    let mut ptrs = vec![
        ssm.cached_ptr(),
        k_state.cached_ptr(),
        v_state.cached_ptr(),
        y.cached_ptr(),
    ];
    if save {
        ptrs.extend([
            h_saved.cached_ptr(),
            k_prev.cached_ptr(),
            v_prev.cached_ptr(),
        ]);
    }
    ptrs.extend([
        x.cached_ptr(),
        k.cached_ptr(),
        q.cached_ptr(),
        alpha.cached_ptr(),
        beta.cached_ptr(),
        gamma.cached_ptr(),
        d_param.cached_ptr(),
    ]);
    let dims = s.dims();
    for (y_first, cfg) in y_runs(cfg) {
        let mut b = ctx.stream.launch_builder(kernel);
        for p in &ptrs {
            b.arg(p);
        }
        for d in &dims {
            b.arg(d);
        }
        b.arg(&y_first);
        // One sequence a row: no packing.
        b.arg(&0u64);
        unsafe { b.launch(cfg) }.unwrap();
    }
    ctx.stream.synchronize().unwrap();
    let y_host = {
        let mut out = vec![0f32; bt * di];
        y.download_f32(&ctx.stream, &mut out).unwrap();
        out
    };
    Forward {
        y: y_host,
        h_saved: download(ctx, &h_saved),
        k_prev: download(ctx, &k_prev),
        v_prev: download(ctx, &v_prev),
        ssm: download(ctx, &ssm),
        k_state: download(ctx, &k_state),
        v_state: download(ctx, &v_state),
    }
}

/// One backward launch over a forward's saves; returns its seven outputs.
fn backward(
    ctx: &GpuCtx,
    (kernel, cfg): (&CudaFunction, LaunchConfig),
    (s, dtype): (Shape, WeightDtype),
    v: &Values,
    fwd: &Forward,
) -> Vec<(&'static str, Vec<f32>)> {
    let (bt, di) = (B * s.t, NH * s.hd);
    let h_saved = upload(ctx, &fwd.h_saved);
    let k_prev = upload(ctx, &fwd.k_prev);
    let v_prev = upload(ctx, &fwd.v_prev);
    let x = upload_typed(ctx, &v.x, dtype);
    let k = upload_typed(ctx, &v.k, dtype);
    let q = upload_typed(ctx, &v.q, dtype);
    let d_y = upload_typed(ctx, &v.d_y, dtype);
    let (alpha, beta, gamma) = (
        upload(ctx, &v.alpha),
        upload(ctx, &v.beta),
        upload(ctx, &v.gamma),
    );
    let d_param = upload(ctx, &v.d_param);
    let z = |n: usize| GpuBuffer::zeros(&ctx.stream, n).unwrap();
    let outs = [
        ("dX", z(bt * di)),
        ("dK", z(bt * NH * s.ds)),
        ("dQ", z(bt * NH * s.ds)),
        ("dalpha", z(bt * NH)),
        ("dbeta", z(bt * NH)),
        ("dgamma", z(bt * NH)),
        ("dD local", z(B * di)),
    ];
    for (y_first, cfg) in y_runs(cfg) {
        let mut b = ctx.stream.launch_builder(kernel);
        let ins = [
            h_saved.cached_ptr(),
            k_prev.cached_ptr(),
            v_prev.cached_ptr(),
            x.cached_ptr(),
            k.cached_ptr(),
            q.cached_ptr(),
            alpha.cached_ptr(),
            beta.cached_ptr(),
            gamma.cached_ptr(),
            d_param.cached_ptr(),
            d_y.cached_ptr(),
        ];
        let out_ptrs: Vec<_> = outs.iter().map(|(_, o)| o.cached_ptr()).collect();
        for p in ins.iter().chain(&out_ptrs) {
            b.arg(p);
        }
        let dims = s.dims();
        for d in &dims {
            b.arg(d);
        }
        b.arg(&y_first);
        // One sequence a row: no packing.
        b.arg(&0u64);
        unsafe { b.launch(cfg) }.unwrap();
    }
    outs.iter()
        .map(|(name, o)| (*name, download(ctx, o)))
        .collect()
}

fn setup() -> (GpuCtx, Mamba3Kernels) {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(ctx.stream.context(), arch::arch0()).unwrap();
    (ctx, m3k)
}

fn assert_forward(label: &str, want: &Forward, got: &Forward) {
    for (name, w, g) in [
        ("y", &want.y, &got.y),
        ("h tape", &want.h_saved, &got.h_saved),
        ("k_prev", &want.k_prev, &got.k_prev),
        ("v_prev", &want.v_prev, &got.v_prev),
        ("ssm state", &want.ssm, &got.ssm),
        ("k state", &want.k_state, &got.k_state),
        ("v state", &want.v_state, &got.v_state),
    ] {
        assert_bits(&format!("{label} {name}"), w, g);
    }
}

const WARP_SHAPES: [(usize, usize); 8] = [
    (1, 8),
    (4, 16),
    (8, 17),
    (16, 16),
    (16, 32),
    (32, 8),
    (32, 64),
    (32, 5),
];

/// Every warp head: the block tier gives the warp kernels' bits, at a step
/// and over windows.
#[test]
fn m3_sequential_block_tier_matches_the_warp_kernels() {
    let (ctx, m3k) = setup();
    let f32 = WeightDtype::F32;
    for (hd, ds) in WARP_SHAPES {
        for t in [1usize, 7, 33] {
            let s = Shape { hd, ds, t };
            let label = format!("hd {hd} ds {ds} T {t}");
            let v = Values::new(s, f32);
            let fwd_smem = sequential_forward_block_smem_bytes(ds);
            let fwd_cfg = s.block_cfg(fwd_smem);
            let warp_target = forward(
                &ctx,
                (&m3k.m3_burnin_fwd_nosave.i32, s.warp_cfg()),
                (s, f32, false),
                &v,
            );
            let block_target = forward(
                &ctx,
                (&m3k.m3_burnin_fwd_nosave_block, fwd_cfg),
                (s, f32, false),
                &v,
            );
            for (name, w, g) in [
                ("y", &warp_target.y, &block_target.y),
                ("ssm state", &warp_target.ssm, &block_target.ssm),
                ("k state", &warp_target.k_state, &block_target.k_state),
                ("v state", &warp_target.v_state, &block_target.v_state),
            ] {
                assert_bits(&format!("{label} target {name}"), w, g);
            }
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                let hv = Values::new(s, dtype);
                let warp = forward(
                    &ctx,
                    (
                        m3k.burnin_fwd_typed_for_state(dtype, ds, IndexWidth::I32),
                        s.warp_cfg(),
                    ),
                    (s, dtype, true),
                    &hv,
                );
                let block = forward(
                    &ctx,
                    (m3k.m3_burnin_fwd_block.get(dtype), fwd_cfg),
                    (s, dtype, true),
                    &hv,
                );
                assert_forward(&format!("{label} {dtype:?} forward"), &warp, &block);
            }
            let warp = forward(
                &ctx,
                (m3k.burnin_fwd_for_state(ds, IndexWidth::I32), s.warp_cfg()),
                (s, f32, true),
                &v,
            );
            let block = forward(
                &ctx,
                (m3k.m3_burnin_fwd_block.get(f32), s.block_cfg(fwd_smem)),
                (s, f32, true),
                &v,
            );
            assert_forward(&format!("{label} forward"), &warp, &block);
            let bwd_smem = sequential_backward_block_smem_bytes(hd, ds);
            let warp_bwd = backward(
                &ctx,
                (
                    m3k.backward_seq_for_state(ds, IndexWidth::I32),
                    s.warp_cfg(),
                ),
                (s, f32),
                &v,
                &warp,
            );
            let block_bwd = backward(
                &ctx,
                (m3k.m3_backward_seq_block.get(f32), s.block_cfg(bwd_smem)),
                (s, f32),
                &v,
                &warp,
            );
            for ((name, w), (_, g)) in warp_bwd.iter().zip(&block_bwd) {
                assert_bits(&format!("{label} backward {name}"), w, g);
            }
        }
    }
}

/// Heads on both tiers, warp and wider, at state widths with no
/// constant-width f32 warp kernel: the half kernels give the f32 kernel's
/// bits on inputs the half type holds exactly, and the target network's
/// forward gives the training forward's output and state.
#[test]
fn m3_sequential_block_half_kernels_widen_exactly() {
    let (ctx, m3k) = setup();
    let f32 = WeightDtype::F32;
    for (hd, ds) in [(16usize, 17usize), (32, 5), (48, 24), (64, 40), (100, 17)] {
        let s = Shape { hd, ds, t: 33 };
        let fwd_cfg = s.block_cfg(sequential_forward_block_smem_bytes(ds));
        let bwd_cfg = s.block_cfg(sequential_backward_block_smem_bytes(hd, ds));
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let label = format!("{dtype:?} hd {hd} ds {ds}");
            let v = Values::new(s, dtype);
            let wide = forward(
                &ctx,
                (m3k.m3_burnin_fwd_block.get(f32), fwd_cfg),
                (s, f32, true),
                &v,
            );
            let half = forward(
                &ctx,
                (m3k.m3_burnin_fwd_block.get(dtype), fwd_cfg),
                (s, dtype, true),
                &v,
            );
            let wide_y = round_to(dtype, &wide.y);
            assert_bits(&format!("{label} y"), &wide_y, &half.y);
            assert_bits(&format!("{label} h tape"), &wide.h_saved, &half.h_saved);
            assert_bits(&format!("{label} ssm state"), &wide.ssm, &half.ssm);
            let wide_bwd = backward(
                &ctx,
                (m3k.m3_backward_seq_block.get(f32), bwd_cfg),
                (s, f32),
                &v,
                &wide,
            );
            let half_bwd = backward(
                &ctx,
                (m3k.m3_backward_seq_block.get(dtype), bwd_cfg),
                (s, dtype),
                &v,
                &wide,
            );
            for ((name, w), (_, g)) in wide_bwd.iter().zip(&half_bwd) {
                assert_bits(&format!("{label} backward {name}"), w, g);
            }
        }
        let v = Values::new(s, f32);
        let train = forward(
            &ctx,
            (m3k.m3_burnin_fwd_block.get(f32), fwd_cfg),
            (s, f32, true),
            &v,
        );
        let target = forward(
            &ctx,
            (&m3k.m3_burnin_fwd_nosave_block, fwd_cfg),
            (s, f32, false),
            &v,
        );
        let label = format!("target hd {hd} ds {ds}");
        assert_bits(&format!("{label} y"), &train.y, &target.y);
        assert_bits(&format!("{label} ssm state"), &train.ssm, &target.ssm);
        assert_bits(&format!("{label} k state"), &train.k_state, &target.k_state);
        assert_bits(&format!("{label} v state"), &train.v_state, &target.v_state);
    }
}
