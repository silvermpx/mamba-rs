//! Large state dimensions on the fused decode path. The step goes a slice
//! of states at a time whatever d_state, and the sequential kernels keep
//! the SSM state in per-thread register arrays whose size is a JIT-time
//! capacity knob; these tests prove a d_state past the historical 64
//! ceiling runs the same code path and matches the CPU reference, which
//! has no ceiling and serves as the independent oracle, and that the step
//! keeps the bits of the released one that held every state at once.

#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;
#[path = "common/frozen_kernels.rs"]
mod frozen_kernels;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use mamba_rs::MambaBackbone;
use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::launch::IndexWidth;
use mamba_rs::mamba1::MambaConfig;
use mamba_rs::mamba1::MambaWeights;
use mamba_rs::mamba1::gpu::inference::MambaGpuBackbone;

fn det(n: usize, seed: u32) -> Vec<f32> {
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

fn check_step_parity(d_state: usize) {
    let cfg = MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        block_norm: mamba_rs::BlockNorm::Rms,
        final_norm: true,
        bidirectional: false,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
        d_intermediate: 0,
    };
    let input_dim = cfg.d_model;
    let dm = cfg.d_model;

    let cpu_bb = MambaBackbone::init(cfg, input_dim, 42);
    let weights: MambaWeights = cpu_bb.weights().clone();
    let mut state = cpu_bb.alloc_state();
    let mut scratch = cpu_bb.alloc_scratch();

    let mut gpu_bb =
        MambaGpuBackbone::new(&weights, cfg, input_dim, 1, GpuOptions::default()).unwrap();

    let steps = 12usize;
    let mut cpu_out = vec![0.0f32; dm];
    let mut gpu_out = vec![0.0f32; dm];
    for t in 0..steps {
        let input = det(input_dim, 0x51 + t as u32);
        cpu_bb.forward_step(&input, &mut cpu_out, &mut state, &mut scratch);
        gpu_bb.step(&input, &mut gpu_out).unwrap();
        let (mut dot, mut na, mut nb, mut num, mut den) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&x, &y) in cpu_out.iter().zip(gpu_out.iter()) {
            dot += (x as f64) * (y as f64);
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
            num += ((x - y) as f64) * ((x - y) as f64);
            den += (x as f64) * (x as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
        let rel = (num / den.max(1e-30)).sqrt();
        // Loose enough for the engine's TF32 GEMM tier accumulating over
        // the recurrent steps; tight enough to catch the capacity-guard
        // failure mode (a kernel returning without writing leaves zeros,
        // which collapses the cosine).
        assert!(
            cos > 0.999 && rel < 2e-2,
            "d_state={d_state} step {t}: cos={cos:.6} rel_l2={rel:.3e} \
             (GPU fused step diverged from the CPU reference)"
        );
    }
}

/// The decode step as released held every state of a channel in register
/// arrays of the compiled capacity; it now goes 16 states at a time with
/// the output's chain unbroken. Against that kernel, frozen and compiled at
/// the largest capacity: states inside one slice, on a slice's edge, past
/// it and up to 256, in f32, bf16 and f16 and both index widths, every
/// state and output bit for bit.
#[test]
fn m1_step_keeps_the_bits_of_its_released_form() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new_with_state_cap(&dev, 256).unwrap();
    let frozen = frozen_kernels::frozen_module(
        dev.context(),
        arch::arch0(),
        Some(256),
        include_str!("fixtures/m1_step_reference.cu"),
    );
    let (batch, di, dt_rank) = (3usize, 48usize, 5usize);
    let typed = |v: &[f32], dt: WeightDtype| {
        let b = DtypedBuf::zeros(&ctx.stream, v.len(), dt).unwrap();
        ctx.stream.synchronize().unwrap();
        b.upload_f32(&ctx.stream, v).unwrap();
        ctx.stream.synchronize().unwrap();
        b
    };
    let f32_buf = |v: &[f32]| GpuBuffer::from_cpu(&ctx.stream, v).unwrap();
    let mut cases = 0;
    for ds in [1usize, 15, 16, 17, 33, 64, 100, 128, 200, 256] {
        let xdbl_stride = dt_rank + 2 * ds;
        let gate_stride = 2 * di;
        let h0 = det(batch * di * ds, 0x61);
        let a_neg: Vec<f32> = det(di * ds, 0x62)
            .iter()
            .map(|v| -(v + 0.6) * 2.0)
            .collect();
        let d = f32_buf(&det(di, 0x63));
        let a = f32_buf(&a_neg);
        for dt in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            let delta_raw = typed(
                &det(batch * di, 0x64)
                    .iter()
                    .map(|v| v * 4.0)
                    .collect::<Vec<_>>(),
                dt,
            );
            let u = typed(&det(batch * di, 0x65), dt);
            let xdbl = typed(&det(batch * xdbl_stride, 0x66), dt);
            let gate = typed(&det(batch * gate_stride, 0x67), dt);
            let suffix = match dt {
                WeightDtype::Bf16 => "bf16",
                WeightDtype::F16 => "f16",
                _ => "f32",
            };
            for width in [IndexWidth::I32, IndexWidth::I64] {
                let wide = if width == IndexWidth::I64 { "_i64" } else { "" };
                let released = frozen
                    .load_function(&format!("m1_reference_step_{suffix}{wide}"))
                    .unwrap();
                let run = |f: &CudaFunction| {
                    let h = f32_buf(&h0);
                    let y = typed(&vec![0.0; batch * di], dt);
                    let ints = [gate_stride as i32];
                    let tail = [
                        batch as i32,
                        di as i32,
                        ds as i32,
                        xdbl_stride as i32,
                        dt_rank as i32,
                        (dt_rank + ds) as i32,
                    ];
                    let ptrs = [
                        h.cached_ptr(),
                        y.cached_ptr(),
                        delta_raw.cached_ptr(),
                        u.cached_ptr(),
                        xdbl.cached_ptr(),
                        gate.cached_ptr(),
                    ];
                    let mut b = ctx.stream.launch_builder(f);
                    for p in &ptrs {
                        b.arg(p);
                    }
                    b.arg(&ints[0]);
                    let (ap, dp) = (a.cached_ptr(), d.cached_ptr());
                    b.arg(&ap);
                    b.arg(&dp);
                    for v in &tail {
                        b.arg(v);
                    }
                    let cfg = LaunchConfig {
                        grid_dim: ((batch * di).div_ceil(128) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { b.launch(cfg) }.unwrap();
                    ctx.stream.synchronize().unwrap();
                    let hs: Vec<u32> = h
                        .to_cpu(&ctx.stream)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    let mut ys = vec![0f32; batch * di];
                    y.download_f32(&ctx.stream, &mut ys).unwrap();
                    (hs, ys.iter().map(|v| v.to_bits()).collect::<Vec<u32>>())
                };
                let want = run(&released);
                let got = run(ctx.kernels.ssm_step_fwd_fused_typed.get(width).get(dt));
                assert!(want.0 == got.0, "{dt:?} {width:?} d_state {ds}: states");
                assert!(want.1 == got.1, "{dt:?} {width:?} d_state {ds}: outputs");
                cases += 1;
            }
        }
    }
    eprintln!("m1 decode step: the released kernel's bits on {cases} cases");
}

/// The Mamba-1 sequential kernels give a channel a warp segment of lanes,
/// each holding a slice of at most 16 states, and hand the output chains
/// from lane to lane. Against the released kernels, which held a channel's
/// whole state in one thread and are frozen and compiled at the largest
/// capacity: states inside one lane, on a lane's edge, with an idle last
/// lane and up to 256; a step, a few and more than a lane's rounds; f32,
/// bf16 and f16 and both index widths; every output of the forward with
/// and without saves (the latter gated and not) and of both backwards, bit
/// for bit.
#[test]
fn m1_sequential_kernels_keep_the_bits_of_their_released_form() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new_with_state_cap(&dev, 256).unwrap();
    let frozen = frozen_kernels::frozen_module(
        dev.context(),
        arch::arch0(),
        Some(256),
        include_str!("fixtures/m1_seq_reference.cu"),
    );
    let (batch, di) = (2usize, 24usize);
    let typed = |v: &[f32], dt: WeightDtype| {
        let b = DtypedBuf::zeros(&ctx.stream, v.len(), dt).unwrap();
        ctx.stream.synchronize().unwrap();
        b.upload_f32(&ctx.stream, v).unwrap();
        ctx.stream.synchronize().unwrap();
        b
    };
    let f32_buf = |v: &[f32]| GpuBuffer::from_cpu(&ctx.stream, v).unwrap();
    let floats = |b: &GpuBuffer| -> Vec<u32> {
        b.to_cpu(&ctx.stream)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let typed_bits = |b: &DtypedBuf, n: usize| -> Vec<u32> {
        let mut v = vec![0f32; n];
        b.download_f32(&ctx.stream, &mut v).unwrap();
        v.iter().map(|x| x.to_bits()).collect()
    };
    let launch = |f: &CudaFunction, ptrs: &[u64], ints: &[i32], threads: usize| {
        let mut b = ctx.stream.launch_builder(f);
        for p in ptrs {
            b.arg(p);
        }
        for v in ints {
            b.arg(v);
        }
        let cfg = LaunchConfig {
            grid_dim: (threads.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.unwrap();
        ctx.stream.synchronize().unwrap();
    };
    let mut cases = 0;
    for ds in [1usize, 15, 16, 17, 33, 48, 64, 65, 100, 129, 200, 256] {
        let lanes = mamba_rs::gpu::launch::seq_lanes(ds);
        let a_neg: Vec<f32> = det(di * ds, 0x71)
            .iter()
            .map(|v| -(v + 0.6) * 2.0)
            .collect();
        let a = f32_buf(&a_neg);
        let d = f32_buf(&det(di, 0x72));
        for t in [1usize, 5, 37] {
            let (bt, st) = (batch * t, batch * di * ds);
            let tape_len = batch * (t + 1) * di * ds;
            let h0 = det(st, 0x73);
            let tape0 = det(tape_len, 0x74);
            for dt in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
                let delta_raw = typed(
                    &det(bt * di, 0x75)
                        .iter()
                        .map(|v| v * 4.0)
                        .collect::<Vec<_>>(),
                    dt,
                );
                let delta = typed(
                    &det(bt * di, 0x76)
                        .iter()
                        .map(|v| v.abs() + 0.05)
                        .collect::<Vec<_>>(),
                    dt,
                );
                let u = typed(&det(bt * di, 0x77), dt);
                let bm = typed(&det(bt * ds, 0x78), dt);
                let cm = typed(&det(bt * ds, 0x79), dt);
                let gate = typed(&det(bt * 2 * di, 0x7a), dt);
                let dy = typed(&det(bt * di, 0x7b), dt);
                let suffix = match dt {
                    WeightDtype::Bf16 => "bf16",
                    WeightDtype::F16 => "f16",
                    _ => "f32",
                };
                for width in [IndexWidth::I32, IndexWidth::I64] {
                    let wide = if width == IndexWidth::I64 { "_i64" } else { "" };
                    let name = |kernel: &str| format!("m1_reference_{kernel}_{suffix}{wide}");
                    let dims = [batch as i32, t as i32, di as i32, ds as i32];
                    let new_dims = [batch as i32, t as i32, di as i32, ds as i32, lanes as i32];
                    // Each kernel run on fresh outputs: (released, new).
                    let k = &ctx.kernels;
                    for gate_stride in [0usize, 2 * di] {
                        let nosave = |f: &CudaFunction, ints: &[i32], threads: usize| {
                            let h = f32_buf(&h0);
                            let y = typed(&vec![0.0; bt * di], dt);
                            let ptrs = [
                                h.cached_ptr(),
                                y.cached_ptr(),
                                delta.cached_ptr(),
                                u.cached_ptr(),
                                bm.cached_ptr(),
                                cm.cached_ptr(),
                                a.cached_ptr(),
                                d.cached_ptr(),
                                gate.cached_ptr(),
                            ];
                            let mut all = vec![gate_stride as i32];
                            all.extend_from_slice(ints);
                            launch(f, &ptrs, &all, threads);
                            (floats(&h), typed_bits(&y, bt * di))
                        };
                        let released = frozen.load_function(&name("burnin_nosave")).unwrap();
                        let want = nosave(&released, &dims, batch * di);
                        let got = nosave(
                            k.ssm_burnin_nosave_typed.get(width).get(dt),
                            &new_dims,
                            batch * di * lanes,
                        );
                        let what =
                            format!("{dt:?} {width:?} d_state {ds} T {t} gate {gate_stride}");
                        assert!(want.0 == got.0, "forward without saves, states: {what}");
                        assert!(want.1 == got.1, "forward without saves, outputs: {what}");
                    }
                    let saving = |f: &CudaFunction, ints: &[i32], threads: usize| {
                        let h = f32_buf(&h0);
                        let y = typed(&vec![0.0; bt * di], dt);
                        let tape = f32_buf(&vec![0.0; tape_len]);
                        let saved = typed(&vec![0.0; bt * di], dt);
                        let ptrs = [
                            h.cached_ptr(),
                            y.cached_ptr(),
                            tape.cached_ptr(),
                            delta_raw.cached_ptr(),
                            saved.cached_ptr(),
                            u.cached_ptr(),
                            bm.cached_ptr(),
                            cm.cached_ptr(),
                            a.cached_ptr(),
                            d.cached_ptr(),
                        ];
                        launch(f, &ptrs, ints, threads);
                        (
                            floats(&h),
                            typed_bits(&y, bt * di),
                            floats(&tape),
                            typed_bits(&saved, bt * di),
                        )
                    };
                    let released = frozen.load_function(&name("burnin")).unwrap();
                    let want = saving(&released, &dims, batch * di);
                    let got = saving(
                        k.ssm_burnin_fwd_typed.get(width).get(dt),
                        &new_dims,
                        batch * di * lanes,
                    );
                    let what = format!("{dt:?} {width:?} d_state {ds} T {t}");
                    assert!(want == got, "saving forward: {what}");
                    let tape = f32_buf(&tape0);
                    let backward = |f: &CudaFunction, ints: &[i32], threads: usize| {
                        let dd = typed(&vec![0.0; bt * di], dt);
                        let du = typed(&vec![0.0; bt * di], dt);
                        let db = typed(&vec![0.0; bt * di * ds], dt);
                        let dc = typed(&vec![0.0; bt * di * ds], dt);
                        let d_d = f32_buf(&vec![0.0; batch * di]);
                        let d_a = f32_buf(&vec![0.0; st]);
                        let ptrs = [
                            tape.cached_ptr(),
                            delta.cached_ptr(),
                            u.cached_ptr(),
                            bm.cached_ptr(),
                            cm.cached_ptr(),
                            a.cached_ptr(),
                            d.cached_ptr(),
                            dy.cached_ptr(),
                            dd.cached_ptr(),
                            du.cached_ptr(),
                            db.cached_ptr(),
                            dc.cached_ptr(),
                            d_d.cached_ptr(),
                            d_a.cached_ptr(),
                        ];
                        launch(f, &ptrs, ints, threads);
                        (
                            typed_bits(&dd, bt * di),
                            typed_bits(&du, bt * di),
                            typed_bits(&db, bt * di * ds),
                            typed_bits(&dc, bt * di * ds),
                            floats(&d_d),
                            floats(&d_a),
                        )
                    };
                    let released = frozen.load_function(&name("backward_local")).unwrap();
                    let want = backward(&released, &dims, batch * di);
                    let got = backward(
                        k.ssm_backward_local_typed.get(width).get(dt),
                        &new_dims,
                        batch * di * lanes,
                    );
                    assert!(want == got, "typed backward: {what}");
                    if dt == WeightDtype::F32 {
                        let released = frozen
                            .load_function(&format!("m1_reference_backward_local{wide}"))
                            .unwrap();
                        let want = backward(&released, &dims, batch * di);
                        let got = backward(
                            k.ssm_backward_local.get(width),
                            &new_dims,
                            batch * di * lanes,
                        );
                        assert!(want == got, "f32 backward: {what}");
                    }
                    cases += 1;
                }
            }
        }
    }
    eprintln!("m1 sequential kernels: the released kernels' bits on {cases} cases");
}

#[test]
fn m1_gpu_step_matches_cpu_at_d_state_128() {
    check_step_parity(128);
}

#[test]
fn m1_gpu_step_matches_cpu_at_d_state_256() {
    check_step_parity(256);
}

/// An explicit Sequential at 128 states runs the sequential scan and trains
/// like the parallel scan: the forward rows and every gradient agree to
/// rounding, the same recurrence summed in another order. Before, the mode
/// was silently replaced by Parallel above 64 states.
#[test]
fn m1_explicit_sequential_trains_at_d_state_128() {
    use mamba_rs::config::ScanMode;
    use mamba_rs::{BackwardOpts, MambaGpuTrainer, TrainSessionCfg};
    let (batch, seq_len, dm) = (2, 40, 64);
    let run = |scan_mode: ScanMode| -> (Vec<f32>, Vec<f32>) {
        let cfg = MambaConfig {
            d_model: dm,
            n_layers: 2,
            d_state: 128,
            d_conv: 4,
            expand: 2,
            scan_mode,
            block_norm: mamba_rs::BlockNorm::Rms,
            final_norm: true,
            bidirectional: false,
            rms_norm_eps: 1e-5,
            bcdt_rms_eps: None,
            dt_rank: None,
            proj_bias: false,
            conv_bias: true,
            d_intermediate: 0,
        };
        let w = MambaWeights::init(&cfg, dm, 7);
        let session = TrainSessionCfg {
            input_dim: dm,
            batch,
            seq_len,
            lr: 1e-3,
            weight_decay: 0.0,
        };
        let mut t = MambaGpuTrainer::new(&w, cfg, session, GpuOptions::default()).unwrap();
        let mut rows = vec![0.0f32; batch * seq_len * dm];
        t.forward(&det(batch * seq_len * dm, 0x71), &mut rows)
            .unwrap();
        let d = det(batch * seq_len * dm, 0x72);
        t.backward_step(&d, BackwardOpts::default()).unwrap();
        let stream = t.ctx().stream.clone();
        (rows, t.grad_arena().to_cpu(&stream).unwrap())
    };
    let (seq_rows, seq_grads) = run(ScanMode::Sequential);
    let (par_rows, par_grads) = run(ScanMode::Parallel);
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
        (num / den.max(1e-30)).sqrt()
    };
    let (r, g) = (rel(&seq_rows, &par_rows), rel(&seq_grads, &par_grads));
    eprintln!("sequential vs parallel at 128 states: rows {r:.2e}, grads {g:.2e}");
    assert!(
        seq_rows != par_rows || seq_grads != par_grads,
        "the explicit Sequential gave the parallel scan's bits: it was not honoured"
    );
    assert!(r < 1e-4, "forward rows differ by {r}");
    assert!(g < 1e-3, "gradients differ by {g}");
}

#[test]
fn state_capacity_range() {
    use mamba_rs::gpu::kernels::state_capacity;
    // 16-granular: the cap sizes per-thread register
    // arrays; a 64-floor at small d_state was 4x local-memory waste.
    assert_eq!(state_capacity(8).unwrap(), 16);
    assert_eq!(state_capacity(16).unwrap(), 16);
    assert_eq!(state_capacity(17).unwrap(), 32);
    assert_eq!(state_capacity(64).unwrap(), 64);
    assert_eq!(state_capacity(65).unwrap(), 80);
    assert_eq!(state_capacity(128).unwrap(), 128);
    assert_eq!(state_capacity(200).unwrap(), 208);
    assert_eq!(state_capacity(256).unwrap(), 256);
    assert!(state_capacity(0).is_err());
    assert!(state_capacity(257).is_err());
}
