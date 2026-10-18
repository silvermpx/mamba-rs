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
use mamba_rs::mamba3_siso::gpu::kernels::{Mamba3Kernels, SequentialShape};
use mamba_rs::mamba3_siso::route::{
    BLOCK_TIER_BACKWARD_STATES_MAX, block_tier_threads, sequential_backward_block_smem_bytes,
    sequential_backward_sliced_smem_bytes, sequential_forward_block_smem_bytes,
    sequential_forward_sliced_smem_bytes, sequential_sliced_block_threads,
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
    fn sliced_cfg(&self, smem: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (B as u32, NH as u32, 1),
            block_dim: (
                sequential_sliced_block_threads(self.hd, self.ds) as u32,
                1,
                1,
            ),
            shared_mem_bytes: smem as u32,
        }
    }
    fn sliced_fwd_cfg(&self) -> LaunchConfig {
        self.sliced_cfg(sequential_forward_sliced_smem_bytes(self.hd, self.ds))
    }
    fn sliced_bwd_cfg(&self) -> LaunchConfig {
        self.sliced_cfg(sequential_backward_sliced_smem_bytes(self.hd, self.ds))
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
                    (m3k.burnin_block_for(ds).get(dtype), fwd_cfg),
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
                (m3k.burnin_block_for(ds).get(f32), s.block_cfg(fwd_smem)),
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
                (
                    m3k.backward_seq_block_for(ds).get(f32),
                    s.block_cfg(bwd_smem),
                ),
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
                (m3k.burnin_block_for(ds).get(f32), fwd_cfg),
                (s, f32, true),
                &v,
            );
            let half = forward(
                &ctx,
                (m3k.burnin_block_for(ds).get(dtype), fwd_cfg),
                (s, dtype, true),
                &v,
            );
            let wide_y = round_to(dtype, &wide.y);
            assert_bits(&format!("{label} y"), &wide_y, &half.y);
            assert_bits(&format!("{label} h tape"), &wide.h_saved, &half.h_saved);
            assert_bits(&format!("{label} ssm state"), &wide.ssm, &half.ssm);
            let wide_bwd = backward(
                &ctx,
                (m3k.backward_seq_block_for(ds).get(f32), bwd_cfg),
                (s, f32),
                &v,
                &wide,
            );
            let half_bwd = backward(
                &ctx,
                (m3k.backward_seq_block_for(ds).get(dtype), bwd_cfg),
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
            (m3k.burnin_block_for(ds).get(f32), fwd_cfg),
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

/// A state of one slice: the sliced forwards give the block tier's general
/// entry's bits, the training forward in f32 and the half types and the
/// target network's, heads of one warp and of many and one wider than a
/// block among them.
#[test]
fn m3_sliced_forwards_give_the_block_tiers_bits_at_one_slice() {
    let (ctx, m3k) = setup();
    let f32 = WeightDtype::F32;
    for (hd, ds) in [
        (1, 5),
        (16, 12),
        (32, 7),
        (48, 13),
        (64, 9),
        (100, 15),
        (600, 11),
    ] {
        for t in [1usize, 7, 33] {
            let s = Shape { hd, ds, t };
            let fwd_cfg = s.block_cfg(sequential_forward_block_smem_bytes(ds));
            for dtype in [f32, WeightDtype::Bf16, WeightDtype::F16] {
                let label = format!("{dtype:?} hd {hd} ds {ds} T {t}");
                let v = Values::new(s, dtype);
                let block = forward(
                    &ctx,
                    (m3k.burnin_block_for(ds).get(dtype), fwd_cfg),
                    (s, dtype, true),
                    &v,
                );
                let sliced = forward(
                    &ctx,
                    (m3k.m3_burnin_fwd_sliced.get(dtype), s.sliced_fwd_cfg()),
                    (s, dtype, true),
                    &v,
                );
                assert_forward(&format!("{label} sliced forward"), &block, &sliced);
            }
            let v = Values::new(s, f32);
            let block = forward(
                &ctx,
                (&m3k.m3_burnin_fwd_nosave_block, fwd_cfg),
                (s, f32, false),
                &v,
            );
            let sliced = forward(
                &ctx,
                (&m3k.m3_burnin_fwd_nosave_sliced, s.sliced_fwd_cfg()),
                (s, f32, false),
                &v,
            );
            let label = format!("target hd {hd} ds {ds} T {t}");
            assert_bits(&format!("{label} y"), &block.y, &sliced.y);
            assert_bits(&format!("{label} ssm state"), &block.ssm, &sliced.ssm);
            assert_bits(&format!("{label} k state"), &block.k_state, &sliced.k_state);
            assert_bits(&format!("{label} v state"), &block.v_state, &sliced.v_state);
        }
    }
}

/// The forward in f64 from the same inputs: each step's output, without
/// the kernels' rounding.
fn forward_y_f64(s: Shape, v: &Values) -> Vec<f64> {
    let (t_len, hd, ds) = (s.t, s.hd, s.ds);
    let di = NH * hd;
    let mut y = vec![0f64; B * t_len * di];
    for b in 0..B {
        for h in 0..NH {
            let base = (b * NH + h) * hd;
            let mut state: Vec<f64> = v.ssm[base * ds..(base + hd) * ds]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let mut x_prev: Vec<f64> = v.v_state[base..base + hd]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            for t in 0..t_len {
                let bt = b * t_len + t;
                let co = bt * NH + h;
                let (alpha, beta, gamma) = (
                    f64::from(v.alpha[co]),
                    f64::from(v.beta[co]),
                    f64::from(v.gamma[co]),
                );
                let row = bt * NH * ds + h * ds;
                let k_prev = |n: usize| {
                    f64::from(if t == 0 {
                        v.k_state[(b * NH + h) * ds + n]
                    } else {
                        v.k[row - NH * ds + n]
                    })
                };
                for (p, (st, xp)) in state.chunks_exact_mut(ds).zip(&mut x_prev).enumerate() {
                    let xi = bt * di + h * hd + p;
                    let x = f64::from(v.x[xi]);
                    let (bv, gx) = (beta * *xp, gamma * x);
                    let mut out = f64::from(v.d_param[h]) * x;
                    for (n, hn) in st.iter_mut().enumerate() {
                        *hn = gx * f64::from(v.k[row + n]) + alpha * *hn + bv * k_prev(n);
                        out += f64::from(v.q[row + n]) * *hn;
                    }
                    y[xi] = out;
                    *xp = x;
                }
            }
        }
    }
    y
}

/// The sliced forward at state widths no entry specialises: the state, its
/// tape and the carried rows are the block tier's bits (each state is one
/// thread's, stepped as the block tier steps it), and the output is the
/// same on every run and as close to the f64 forward as the block tier's.
#[test]
fn m3_sliced_forward_keeps_the_state_bits_and_is_as_close_to_f64() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k =
        Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), 256).unwrap();
    let f32 = WeightDtype::F32;
    for (hd, ds) in [(16, 128), (64, 96), (100, 256), (32, 48), (8, 200)] {
        for t in [1usize, 33] {
            let s = Shape { hd, ds, t };
            let label = format!("hd {hd} ds {ds} T {t}");
            let v = Values::new(s, f32);
            let block = forward(
                &ctx,
                (
                    m3k.burnin_block_for(ds).get(f32),
                    s.block_cfg(sequential_forward_block_smem_bytes(ds)),
                ),
                (s, f32, true),
                &v,
            );
            let sliced = forward(
                &ctx,
                (m3k.m3_burnin_fwd_sliced.get(f32), s.sliced_fwd_cfg()),
                (s, f32, true),
                &v,
            );
            let again = forward(
                &ctx,
                (m3k.m3_burnin_fwd_sliced.get(f32), s.sliced_fwd_cfg()),
                (s, f32, true),
                &v,
            );
            for (name, w, g) in [
                ("h tape", &block.h_saved, &sliced.h_saved),
                ("k_prev", &block.k_prev, &sliced.k_prev),
                ("v_prev", &block.v_prev, &sliced.v_prev),
                ("ssm state", &block.ssm, &sliced.ssm),
                ("k state", &block.k_state, &sliced.k_state),
                ("v state", &block.v_state, &sliced.v_state),
            ] {
                assert_bits(&format!("{label} {name}"), w, g);
            }
            assert_bits(&format!("{label} y again"), &sliced.y, &again.y);
            let exact = forward_y_f64(s, &v);
            let (eb, es) = (
                relative_error(&block.y, &exact),
                relative_error(&sliced.y, &exact),
            );
            println!("{label} y: block {eb:.2e}, sliced {es:.2e}");
            assert!(
                es <= 2.0 * eb + 1e-6 && es < 1e-4,
                "{label} y: sliced {es:e} against block {eb:e}"
            );
        }
    }
}

/// A state of one slice: the sliced backward gives the block tier's bits,
/// heads of one warp and of many and one wider than a block among them, in
/// f32 and from the half types' inputs.
#[test]
fn m3_sliced_backward_gives_the_block_tiers_bits_at_one_slice() {
    let (ctx, m3k) = setup();
    let f32 = WeightDtype::F32;
    for (hd, ds) in [
        (1, 5),
        (16, 12),
        (32, 7),
        (48, 13),
        (64, 9),
        (100, 15),
        (600, 11),
    ] {
        for t in [1usize, 7, 33] {
            let s = Shape { hd, ds, t };
            let fwd_cfg = s.block_cfg(sequential_forward_block_smem_bytes(ds));
            let bwd_cfg = s.block_cfg(sequential_backward_block_smem_bytes(hd, ds));
            for dtype in [f32, WeightDtype::Bf16, WeightDtype::F16] {
                let label = format!("{dtype:?} hd {hd} ds {ds} T {t}");
                let v = Values::new(s, dtype);
                let fwd = forward(
                    &ctx,
                    (m3k.burnin_block_for(ds).get(f32), fwd_cfg),
                    (s, f32, true),
                    &v,
                );
                let block = backward(
                    &ctx,
                    (m3k.backward_seq_block_for(ds).get(dtype), bwd_cfg),
                    (s, dtype),
                    &v,
                    &fwd,
                );
                let sliced = backward(
                    &ctx,
                    (m3k.m3_backward_seq_sliced.get(dtype), s.sliced_bwd_cfg()),
                    (s, dtype),
                    &v,
                    &fwd,
                );
                for ((name, w), (_, g)) in block.iter().zip(&sliced) {
                    assert_bits(&format!("{label} sliced {name}"), w, g);
                }
            }
        }
    }
}

/// The backward through time in f64 from the same inputs and saves: what
/// the kernels compute, without their rounding.
fn backward_f64(s: Shape, v: &Values, fwd: &Forward) -> Vec<(&'static str, Vec<f64>)> {
    let (t_len, hd, ds) = (s.t, s.hd, s.ds);
    let di = NH * hd;
    let nhd_ds = di * ds;
    let mut d_x = vec![0f64; B * t_len * di];
    let mut d_k = vec![0f64; B * t_len * NH * ds];
    let mut d_q = vec![0f64; B * t_len * NH * ds];
    let mut d_ab = [
        vec![0f64; B * t_len * NH],
        vec![0f64; B * t_len * NH],
        vec![0f64; B * t_len * NH],
    ];
    let mut d_d = vec![0f64; B * di];
    for b in 0..B {
        for h in 0..NH {
            let mut dh = vec![0f64; hd * ds];
            let mut dv_carry = vec![0f64; hd];
            let mut carry = vec![0f64; ds];
            let mut sum_d = vec![0f64; hd];
            for t in (0..t_len).rev() {
                let bt = b * t_len + t;
                let co = bt * NH + h;
                let (alpha, beta, gamma) = (
                    f64::from(v.alpha[co]),
                    f64::from(v.beta[co]),
                    f64::from(v.gamma[co]),
                );
                let row = bt * NH * ds + h * ds;
                let mut sums = vec![(0f64, 0f64, 0f64); ds];
                let mut scal = [0f64; 3];
                let rows = dh.chunks_exact_mut(ds);
                for (p, ((dv, sd), dh_row)) in dv_carry
                    .iter_mut()
                    .zip(sum_d.iter_mut())
                    .zip(rows)
                    .enumerate()
                {
                    let xi = bt * di + h * hd + p;
                    let (dy, x, vp) = (
                        f64::from(v.d_y[xi]),
                        f64::from(v.x[xi]),
                        f64::from(fwd.v_prev[xi]),
                    );
                    *sd += dy * x;
                    let mut dx = f64::from(v.d_param[h]) * dy + *dv;
                    let mut vprev_acc = 0f64;
                    for (n, (g, (q_sum, kp_sum, kc_sum))) in
                        dh_row.iter_mut().zip(sums.iter_mut()).enumerate()
                    {
                        let (kc, kp, qc) = (
                            f64::from(v.k[row + n]),
                            f64::from(fwd.k_prev[row + n]),
                            f64::from(v.q[row + n]),
                        );
                        let tape = b * (t_len + 1) * nhd_ds + n * di + h * hd + p;
                        let h_prev = f64::from(fwd.h_saved[tape + t * nhd_ds]);
                        let h_curr = f64::from(fwd.h_saved[tape + (t + 1) * nhd_ds]);
                        *g += dy * qc;
                        *q_sum += dy * h_curr;
                        let dh_n = *g;
                        scal[0] += dh_n * h_prev;
                        scal[1] += dh_n * vp * kp;
                        scal[2] += dh_n * x * kc;
                        vprev_acc += dh_n * beta * kp;
                        *kp_sum += dh_n * beta * vp;
                        dx += dh_n * gamma * kc;
                        *kc_sum += dh_n * gamma * x;
                        *g = alpha * dh_n;
                    }
                    d_x[xi] = dx;
                    *dv = vprev_acc;
                }
                let last = t + 1 == t_len;
                for (n, (c, &(q_sum, kp_sum, kc_sum))) in carry.iter_mut().zip(&sums).enumerate() {
                    d_k[row + n] = if last { 0.0 } else { *c } + kc_sum;
                    *c = kp_sum;
                    d_q[row + n] = q_sum;
                }
                for (out, sum) in d_ab.iter_mut().zip(scal) {
                    out[co] = sum;
                }
            }
            d_d[b * di + h * hd..b * di + (h + 1) * hd].copy_from_slice(&sum_d);
        }
    }
    let [d_alpha, d_beta, d_gamma] = d_ab;
    vec![
        ("dX", d_x),
        ("dK", d_k),
        ("dQ", d_q),
        ("dalpha", d_alpha),
        ("dbeta", d_beta),
        ("dgamma", d_gamma),
        ("dD local", d_d),
    ]
}

/// The largest distance of `got` from the f64 `want`, over `want`'s largest
/// magnitude.
fn relative_error(got: &[f32], want: &[f64]) -> f64 {
    let scale = want
        .iter()
        .fold(0f64, |m, w| m.max(w.abs()))
        .max(f64::MIN_POSITIVE);
    got.iter()
        .zip(want)
        .fold(0f64, |m, (&g, &w)| m.max((f64::from(g) - w).abs()))
        / scale
}

/// The sliced backward at state widths no entry specialises: the same bits
/// on every run, and every output as close to the f64 backward as the block
/// tier's whole-state walk it replaces, at heads of a warp and wider.
#[test]
fn m3_sliced_backward_is_deterministic_and_as_close_to_f64_as_the_block_tier() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k =
        Mamba3Kernels::compile_with_state_cap(ctx.stream.context(), arch::arch0(), 256).unwrap();
    let f32 = WeightDtype::F32;
    // 64 states included: past the block tier's register states the
    // backward of every head it would serve takes the sliced kernel.
    for (hd, ds) in [
        (16, 128),
        (64, 96),
        (100, 256),
        (32, 48),
        (8, 200),
        (64, 64),
        (40, 64),
        (128, 64),
    ] {
        for t in [1usize, 33] {
            let s = Shape { hd, ds, t };
            let label = format!("hd {hd} ds {ds} T {t}");
            let v = Values::new(s, f32);
            let fwd = forward(
                &ctx,
                (
                    m3k.burnin_block_for(ds).get(f32),
                    s.block_cfg(sequential_forward_block_smem_bytes(ds)),
                ),
                (s, f32, true),
                &v,
            );
            let block = backward(
                &ctx,
                (
                    m3k.backward_seq_block_for(ds).get(f32),
                    s.block_cfg(sequential_backward_block_smem_bytes(hd, ds)),
                ),
                (s, f32),
                &v,
                &fwd,
            );
            let sliced = backward(
                &ctx,
                (m3k.m3_backward_seq_sliced.get(f32), s.sliced_bwd_cfg()),
                (s, f32),
                &v,
                &fwd,
            );
            let again = backward(
                &ctx,
                (m3k.m3_backward_seq_sliced.get(f32), s.sliced_bwd_cfg()),
                (s, f32),
                &v,
                &fwd,
            );
            let exact = backward_f64(s, &v, &fwd);
            for (((name, b), (_, g)), ((_, a), (_, w))) in
                block.iter().zip(&sliced).zip(again.iter().zip(&exact))
            {
                assert_bits(&format!("{label} {name} again"), g, a);
                let (eb, es) = (relative_error(b, w), relative_error(g, w));
                println!("{label} {name}: block {eb:.2e}, sliced {es:.2e}");
                assert!(
                    es <= 2.0 * eb + 1e-6 && es < 1e-4,
                    "{label} {name}: sliced {es:e} against block {eb:e}"
                );
            }
        }
    }
}

/// The sequential backward's launch at the widths with an entry of their
/// own: the f32 warp kernels keep the heads of one warp, the block tier
/// serves every other head up to the states its thread holds, and past them
/// the sliced kernel takes every head the block tier would have served, the
/// bf16 and f16 backward included.
#[test]
fn m3_sequential_backward_takes_the_sliced_kernel_past_the_block_tier_states() {
    let (_ctx, m3k) = setup();
    for ds in [8usize, 16, 32, 64] {
        for hd in [4usize, 16, 32, 40, 48, 64, 128, 512] {
            for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
                let shape = SequentialShape {
                    batch: B,
                    nheads: NH,
                    headdim: hd,
                    d_state: ds,
                };
                let (_, cfg) = m3k.sequential_backward(dtype, shape, IndexWidth::I32);
                let warp = hd <= 32 && hd.is_power_of_two() && dtype == WeightDtype::F32;
                let want = if warp {
                    (hd, 0)
                } else if ds > BLOCK_TIER_BACKWARD_STATES_MAX {
                    (
                        sequential_sliced_block_threads(hd, ds),
                        sequential_backward_sliced_smem_bytes(hd, ds),
                    )
                } else {
                    (
                        block_tier_threads(hd),
                        sequential_backward_block_smem_bytes(hd, ds),
                    )
                };
                assert_eq!(
                    (cfg.block_dim.0 as usize, cfg.shared_mem_bytes as usize),
                    want,
                    "{dtype:?} hd {hd} ds {ds}"
                );
            }
        }
    }
}
