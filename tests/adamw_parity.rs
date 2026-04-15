//! Step 12 — GPU AdamW parity vs PyTorch-equivalent CPU reference.
//!
//! Asserts the fused `adamw_step_f32` kernel produces bit-close updates
//! to the canonical `torch.optim.AdamW` formulation:
//!
//!   m_t = β1·m_{t-1} + (1 - β1)·g
//!   v_t = β2·v_{t-1} + (1 - β2)·g²
//!   p_t = p_{t-1} · (1 - lr·wd) - lr · (m_t/(1-β1ᵗ)) / (√(v_t/(1-β2ᵗ)) + ε)
//!
//! Tolerance ~1e-5 (single-precision GEMM-style noise).

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::adamw::GpuAdamW;
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::kernels::MambaKernels;

#[allow(clippy::too_many_arguments)]
fn cpu_adamw_step(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step: i32,
) {
    let bc1 = 1.0 / (1.0 - (beta1 as f64).powi(step));
    let bc2 = 1.0 / (1.0 - (beta2 as f64).powi(step));
    let bc1 = bc1 as f32;
    let bc2 = bc2 as f32;
    let one_minus_b1 = 1.0 - beta1;
    let one_minus_b2 = 1.0 - beta2;
    let decay = 1.0 - lr * wd;
    for i in 0..p.len() {
        let gi = g[i];
        let mi = m[i] * beta1 + one_minus_b1 * gi;
        let vi = v[i] * beta2 + one_minus_b2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        let m_hat = mi * bc1;
        let v_hat = vi * bc2;
        p[i] = decay * p[i] - lr * m_hat / (v_hat.sqrt() + eps);
    }
}

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

fn assert_close(label: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{label} len mismatch");
    let mut max_err = 0.0_f32;
    let mut sum_sq_err = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        let e = (x - y).abs();
        max_err = max_err.max(e);
        sum_sq_err += (x - y).powi(2);
        sum_sq += y.powi(2);
    }
    let rms = (sum_sq_err / a.len() as f32).sqrt();
    let rel = (sum_sq_err / sum_sq.max(1e-30)).sqrt();
    eprintln!("{label}: max={max_err:.3e} rms={rms:.3e} rel={rel:.3e}");
    assert!(max_err < tol, "{label} max_err={max_err} > tol={tol}");
}

#[test]
fn adamw_single_tensor_step1_matches_cpu() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let kern = MambaKernels::compile(dev.context(), "sm_89").unwrap();

    let n = 2048;
    let mut p = det_rand(n, 0xA1);
    let g = det_rand(n, 0xA2);
    let mut m = vec![0.0_f32; n];
    let mut v = vec![0.0_f32; n];

    // GPU
    let p_gpu = GpuBuffer::from_cpu(&ctx.stream, &p).unwrap();
    let g_gpu = GpuBuffer::from_cpu(&ctx.stream, &g).unwrap();
    let mut adam = GpuAdamW::new(&ctx.stream, n)
        .unwrap()
        .with_lr(3e-4)
        .with_betas(0.9, 0.999)
        .with_eps(1e-8)
        .with_weight_decay(1e-2);

    let (_, bc1, bc2) = adam.advance();
    adam.step_one(
        &ctx,
        &kern.adamw_step_f32,
        p_gpu.cached_ptr(),
        g_gpu.cached_ptr(),
        adam.m.cached_ptr(),
        adam.v.cached_ptr(),
        n,
        bc1,
        bc2,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    let p_after = p_gpu.to_cpu(&ctx.stream).unwrap();
    let m_after = adam.m.to_cpu(&ctx.stream).unwrap();
    let v_after = adam.v.to_cpu(&ctx.stream).unwrap();

    // CPU reference
    cpu_adamw_step(&mut p, &g, &mut m, &mut v, 3e-4, 0.9, 0.999, 1e-8, 1e-2, 1);

    assert_close("param", &p_after, &p, 1e-6);
    assert_close("m", &m_after, &m, 1e-6);
    assert_close("v", &v_after, &v, 1e-6);
}

#[test]
fn adamw_multi_step_accumulates_correctly() {
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let kern = MambaKernels::compile(dev.context(), "sm_89").unwrap();

    let n = 1024;
    let mut p = det_rand(n, 0xB1);
    let mut m = vec![0.0_f32; n];
    let mut v = vec![0.0_f32; n];

    let p_gpu = GpuBuffer::from_cpu(&ctx.stream, &p).unwrap();
    let mut adam = GpuAdamW::new(&ctx.stream, n)
        .unwrap()
        .with_lr(1e-3)
        .with_weight_decay(1e-2);

    for step in 1..=10 {
        let g = det_rand(n, 0xC0 + step as u32);
        let g_gpu = GpuBuffer::from_cpu(&ctx.stream, &g).unwrap();
        let (_, bc1, bc2) = adam.advance();
        adam.step_one(
            &ctx,
            &kern.adamw_step_f32,
            p_gpu.cached_ptr(),
            g_gpu.cached_ptr(),
            adam.m.cached_ptr(),
            adam.v.cached_ptr(),
            n,
            bc1,
            bc2,
        )
        .unwrap();
        cpu_adamw_step(
            &mut p, &g, &mut m, &mut v, 1e-3, 0.9, 0.999, 1e-8, 1e-2, step,
        );
    }
    ctx.stream.synchronize().unwrap();

    let p_after = p_gpu.to_cpu(&ctx.stream).unwrap();
    assert_close("param after 10 steps", &p_after, &p, 5e-6);
}

#[test]
fn adamw_zero_weight_decay_matches_adam() {
    // With wd=0 AdamW collapses to Adam — verify that path still matches.
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let kern = MambaKernels::compile(dev.context(), "sm_89").unwrap();

    let n = 256;
    let mut p = det_rand(n, 0xD1);
    let g = det_rand(n, 0xD2);
    let mut m = vec![0.0_f32; n];
    let mut v = vec![0.0_f32; n];

    let p_gpu = GpuBuffer::from_cpu(&ctx.stream, &p).unwrap();
    let g_gpu = GpuBuffer::from_cpu(&ctx.stream, &g).unwrap();
    let mut adam = GpuAdamW::new(&ctx.stream, n)
        .unwrap()
        .with_lr(1e-3)
        .with_weight_decay(0.0);

    let (_, bc1, bc2) = adam.advance();
    adam.step_one(
        &ctx,
        &kern.adamw_step_f32,
        p_gpu.cached_ptr(),
        g_gpu.cached_ptr(),
        adam.m.cached_ptr(),
        adam.v.cached_ptr(),
        n,
        bc1,
        bc2,
    )
    .unwrap();
    cpu_adamw_step(&mut p, &g, &mut m, &mut v, 1e-3, 0.9, 0.999, 1e-8, 0.0, 1);
    ctx.stream.synchronize().unwrap();

    let p_after = p_gpu.to_cpu(&ctx.stream).unwrap();
    assert_close("param wd=0", &p_after, &p, 1e-6);
}

/// Backbone-level: walks every M1 master tensor through `step_m1`, then
/// asserts the in-place-updated flat grad arena image (rebuilt by gathering
/// per-tensor weights) matches a CPU reference walk in the same arena order.
#[test]
fn adamw_m1_backbone_walks_all_tensors() {
    use mamba_rs::config::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::adamw::step_m1;
    use mamba_rs::mamba_ssm::gpu::weights::{GpuMambaGrads, GpuMambaTrainWeights};
    use mamba_rs::weights::MambaWeights;

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let kern = MambaKernels::compile(dev.context(), "sm_89").unwrap();

    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        ..Default::default()
    };
    let input_dim = cfg.d_model;
    let cpu_w = MambaWeights::init(&cfg, input_dim, 0xCAFE);

    let mut weights = GpuMambaTrainWeights::from_cpu(&ctx.stream, &cpu_w).unwrap();
    let mut grads = GpuMambaGrads::new(&ctx.stream, &cfg, input_dim).unwrap();

    // Seed the flat grad arena with deterministic non-zero values so the
    // adamw step has something to do.
    let n = grads.flat.len();
    let g_seed = det_rand(n, 0xF7);
    grads.flat.upload(&ctx.stream, &g_seed).unwrap();
    ctx.stream.synchronize().unwrap();

    // Snapshot ALL master tensors as (label, before-image) in arena order.
    let snap_w = |w: &GpuMambaTrainWeights| -> Vec<(String, Vec<f32>)> {
        let mut v = vec![
            (
                "input_proj_w".into(),
                w.input_proj_w.to_cpu(&ctx.stream).unwrap(),
            ),
            (
                "input_proj_b".into(),
                w.input_proj_b.to_cpu(&ctx.stream).unwrap(),
            ),
        ];
        for (li, lw) in w.layers.iter().enumerate() {
            v.push((
                format!("L{li}.norm"),
                lw.norm_weight.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.in_proj"),
                lw.in_proj_w.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.conv1d_w"),
                lw.conv1d_weight.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.conv1d_b"),
                lw.conv1d_bias.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.x_proj"),
                lw.x_proj_w.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.dt_proj_w"),
                lw.dt_proj_w.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.dt_proj_b"),
                lw.dt_proj_b.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.a_log"),
                lw.a_log.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.d_param"),
                lw.d_param.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.out_proj"),
                lw.out_proj_w.to_cpu(&ctx.stream).unwrap(),
            ));
        }
        v.push((
            "norm_f".into(),
            w.norm_f_weight.to_cpu(&ctx.stream).unwrap(),
        ));
        v
    };
    let before = snap_w(&weights);
    // Sanity: snap order length matches flat arena length.
    let total_before: usize = before.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(total_before, n);

    // GPU step_m1
    let mut adam = GpuAdamW::new(&ctx.stream, n)
        .unwrap()
        .with_lr(3e-4)
        .with_weight_decay(1e-2);
    step_m1(&ctx, &kern.adamw_step_f32, &mut adam, &mut weights, &grads).unwrap();
    ctx.stream.synchronize().unwrap();
    let after = snap_w(&weights);

    // CPU reference: walk arena in order, run cpu_adamw_step on each slice.
    let mut m_cpu = vec![0.0_f32; n];
    let mut v_cpu = vec![0.0_f32; n];
    let mut off = 0usize;
    for ((label, p_before), (_, p_after)) in before.iter().zip(after.iter()) {
        let len = p_before.len();
        let mut p = p_before.clone();
        let g = &g_seed[off..off + len];
        cpu_adamw_step(
            &mut p,
            g,
            &mut m_cpu[off..off + len],
            &mut v_cpu[off..off + len],
            3e-4,
            0.9,
            0.999,
            1e-8,
            1e-2,
            1,
        );
        assert_close(label, p_after, &p, 1e-5);
        off += len;
    }
    assert_eq!(off, n);
}

/// M3 backbone equivalent of `adamw_m1_backbone_walks_all_tensors`.
#[test]
fn adamw_m3_backbone_walks_all_tensors() {
    use mamba_rs::mamba_ssm::gpu::adamw::step_m3;
    use mamba_rs::mamba3_siso::config::Mamba3Config;
    use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
    use mamba_rs::mamba3_siso::gpu::weights::{GpuMamba3Grads, GpuMamba3Weights};
    use mamba_rs::mamba3_siso::weights::Mamba3Weights;

    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let m3k = Mamba3Kernels::compile(dev.context(), "sm_89").unwrap();

    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
    };
    let input_dim = cfg.d_model;
    let cpu_w = Mamba3Weights::init(&cfg, input_dim, 0xC0DE);

    let mut weights = GpuMamba3Weights::from_cpu(&ctx.stream, &cpu_w, &cfg, input_dim).unwrap();
    let mut grads = GpuMamba3Grads::new(&ctx.stream, &cfg, input_dim).unwrap();

    let n = grads.flat.len();
    let g_seed = det_rand(n, 0xF8);
    grads.flat.upload(&ctx.stream, &g_seed).unwrap();
    ctx.stream.synchronize().unwrap();

    let snap_w = |w: &GpuMamba3Weights| -> Vec<(String, Vec<f32>)> {
        let mut v = vec![
            (
                "input_proj_w".into(),
                w.input_proj_w.to_cpu(&ctx.stream).unwrap(),
            ),
            (
                "input_proj_b".into(),
                w.input_proj_b.to_cpu(&ctx.stream).unwrap(),
            ),
        ];
        for (li, lw) in w.layers.iter().enumerate() {
            v.push((
                format!("L{li}.norm"),
                lw.norm_weight.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.in_proj"),
                lw.in_proj_w.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.dt_bias"),
                lw.dt_bias.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.b_norm"),
                lw.b_norm_weight.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.c_norm"),
                lw.c_norm_weight.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.b_bias"),
                lw.b_bias.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.c_bias"),
                lw.c_bias.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.d_param"),
                lw.d_param.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.norm_gate"),
                lw.norm_gate_weight.to_cpu(&ctx.stream).unwrap(),
            ));
            v.push((
                format!("L{li}.out_proj"),
                lw.out_proj_w.to_cpu(&ctx.stream).unwrap(),
            ));
        }
        v.push((
            "norm_f".into(),
            w.norm_f_weight.to_cpu(&ctx.stream).unwrap(),
        ));
        v
    };
    let before = snap_w(&weights);
    let total_before: usize = before.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(total_before, n, "snapshot len must match flat grad arena");

    let mut adam = GpuAdamW::new(&ctx.stream, n)
        .unwrap()
        .with_lr(3e-4)
        .with_weight_decay(1e-2);
    step_m3(&ctx, &m3k.adamw_step_f32, &mut adam, &mut weights, &grads).unwrap();
    ctx.stream.synchronize().unwrap();
    let after = snap_w(&weights);

    let mut m_cpu = vec![0.0_f32; n];
    let mut v_cpu = vec![0.0_f32; n];
    let mut off = 0usize;
    for ((label, p_before), (_, p_after)) in before.iter().zip(after.iter()) {
        let len = p_before.len();
        let mut p = p_before.clone();
        let g = &g_seed[off..off + len];
        cpu_adamw_step(
            &mut p,
            g,
            &mut m_cpu[off..off + len],
            &mut v_cpu[off..off + len],
            3e-4,
            0.9,
            0.999,
            1e-8,
            1e-2,
            1,
        );
        assert_close(label, p_after, &p, 1e-5);
        off += len;
    }
    assert_eq!(off, n);
}

#[test]
fn adamw_zero_grad_is_pure_decay() {
    // g=0 → m,v stay 0 → m_hat=0, v_hat=0 → update is just `p *= (1-lr·wd)`.
    let dev = GpuDevice::new(0).unwrap();
    let ctx = GpuCtx::new(&dev).unwrap();
    let kern = MambaKernels::compile(dev.context(), "sm_89").unwrap();

    let n = 64;
    let p = det_rand(n, 0xE1);
    let g = vec![0.0_f32; n];

    let p_gpu = GpuBuffer::from_cpu(&ctx.stream, &p).unwrap();
    let g_gpu = GpuBuffer::from_cpu(&ctx.stream, &g).unwrap();
    let lr = 1e-2;
    let wd = 1e-1;
    let mut adam = GpuAdamW::new(&ctx.stream, n)
        .unwrap()
        .with_lr(lr)
        .with_weight_decay(wd);

    let (_, bc1, bc2) = adam.advance();
    adam.step_one(
        &ctx,
        &kern.adamw_step_f32,
        p_gpu.cached_ptr(),
        g_gpu.cached_ptr(),
        adam.m.cached_ptr(),
        adam.v.cached_ptr(),
        n,
        bc1,
        bc2,
    )
    .unwrap();
    ctx.stream.synchronize().unwrap();

    let p_after = p_gpu.to_cpu(&ctx.stream).unwrap();
    let expected: Vec<f32> = p.iter().map(|x| x * (1.0 - lr * wd)).collect();
    assert_close("decay-only update", &p_after, &expected, 1e-6);
    let m = adam.m.to_cpu(&ctx.stream).unwrap();
    let v = adam.v.to_cpu(&ctx.stream).unwrap();
    assert!(m.iter().all(|&x| x == 0.0), "m must stay zero");
    assert!(v.iter().all(|&x| x == 0.0), "v must stay zero");
}
