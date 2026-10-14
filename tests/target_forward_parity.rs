//! Parity of the GPU target-network forwards with their references.
//!
//! `gpu_forward_mamba_target_burnin` and `gpu_forward_mamba3_target_burnin`
//! are the burn-in forwards that RL consumers run for a target network:
//! the whole sequence in one batched pass, no activation saves, the last
//! timestep out. Nothing else in the tree exercised them. The Mamba-1
//! forward is compared, sample by sample and on both scan routes, against
//! the CPU `forward_mamba_target_sequence` on the same pre-batched input
//! projection. The Mamba-3 forward, which has no CPU twin, is compared
//! against the last timestep of the training backbone forward on the same
//! weights and input, on both scan routes. The tolerances absorb the
//! different summation orders of the routes; a wrong kernel argument or a
//! stale buffer would miss them by orders of magnitude.
#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;
#[path = "common/digest.rs"]
mod digest;

use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::mamba1::MambaWeights;
use mamba_rs::mamba1::cpu::target::{MambaTargetSeqScratch, forward_mamba_target_sequence};
use mamba_rs::mamba1::gpu::backward::GpuMambaTargetScratch;
use mamba_rs::mamba1::gpu::forward::{GpuMambaDims, gpu_forward_mamba_target_burnin};
use mamba_rs::mamba1::gpu::weights::GpuMambaTrainWeights;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::forward::{
    gpu_forward_mamba3_backbone, gpu_forward_mamba3_target_burnin,
};
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::state::{
    GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3Scratch, GpuMamba3StateBufs,
    GpuMamba3TargetScratch, M3Exec,
};
use mamba_rs::mamba3_siso::gpu::weights::GpuMamba3Weights;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{config::ScanMode, mamba1::MambaConfig};

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

fn train_weights_from(w: &MambaWeights) -> TrainMambaWeights {
    TrainMambaWeights {
        input_proj_w: w.input_proj_w.clone(),
        input_proj_b: w.input_proj_b.clone(),
        layers: w
            .layers
            .iter()
            .map(|lw| TrainMambaLayerWeights {
                norm_weight: lw.norm_weight.clone(),
                norm_bias: lw.norm_bias.clone(),
                in_proj_w: lw.in_proj_w.clone(),
                in_proj_b: lw.in_proj_b.clone(),
                conv1d_weight: lw.conv1d_weight.clone(),
                conv1d_bias: lw.conv1d_bias.clone(),
                x_proj_w: lw.x_proj_w.clone(),
                dt_proj_w: lw.dt_proj_w.clone(),
                dt_proj_b: lw.dt_proj_b.clone(),
                a_log: lw.a_log.clone(),
                d_param: lw.d_param.clone(),
                out_proj_w: lw.out_proj_w.clone(),
                out_proj_b: lw.out_proj_b.clone(),
            })
            .collect(),
        norm_f_weight: w.norm_f_weight.clone(),
        norm_f_bias: w.norm_f_bias.clone(),
        mlp: w.mlp.clone(),
    }
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na < 1e-20 || nb < 1e-20 {
        return 1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_rel_err_masked(a: &[f32], b: &[f32], atol: f32) -> f32 {
    let mut worst = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.abs().max(y.abs()) < atol {
            continue;
        }
        let d = (x - y).abs();
        worst = worst.max(d / x.abs().max(y.abs()).max(atol));
    }
    worst
}

fn run_case(scan_mode: ScanMode, seq_len: usize, rms_norm_eps: f32) {
    let dev = GpuDevice::new(0).expect("GpuDevice");
    let ctx = GpuCtx::new(&dev).expect("GpuCtx");
    let cfg = MambaConfig {
        d_model: 32,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode,
        block_norm: mamba_rs::BlockNorm::Rms,
        final_norm: true,
        rms_norm_eps,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
        d_intermediate: 0,
    };
    let input_dim = cfg.d_model;
    let batch = 2;
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let dc = cfg.d_conv;
    let dr = cfg.dt_rank();
    let nl = cfg.n_layers;
    let cpu_weights = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    // The target forward consumes the input projection output; feed both
    // sides the same pre-batched values.
    let ip_out = det(batch * seq_len * dm, 0xAA, 0.05);

    // CPU reference, one sample at a time with fresh states.
    let cpu_tw = train_weights_from(&cpu_weights);
    let mut cpu_scratch = MambaTargetSeqScratch::new(dm, di, ds, dc, dr, nl, seq_len);
    let mut cpu_out = vec![0.0f32; batch * dm];
    for b in 0..batch {
        cpu_scratch.reset_states();
        forward_mamba_target_sequence(
            &mut cpu_out[b * dm..(b + 1) * dm],
            &ip_out[b * seq_len * dm..(b + 1) * seq_len * dm],
            &cpu_tw,
            &mut cpu_scratch,
            (dm, di, ds, dc, dr, seq_len),
            (cfg.rms_norm_eps, cfg.bcdt_rms_eps),
        );
    }

    // GPU, the whole batch in one pass.
    let gpu_w = GpuMambaTrainWeights::from_cpu(&ctx.stream, &cpu_weights)
        .expect("GpuMambaTrainWeights::from_cpu");
    let gpu_dims = GpuMambaDims {
        batch,
        d_model: dm,
        d_inner: di,
        d_state: ds,
        d_conv: dc,
        dt_rank: dr,
        xdbl_dim: cfg.xdbl_dim(),
        seq_len,
        mamba_input_dim: input_dim,
        n_layers: nl,
        scan_mode,
        rms_norm_eps,
        block_norm: cfg.block_norm,
        bcdt_rms_eps: None,
    };
    let mut scratch = GpuMambaTargetScratch::new(&ctx.stream, &gpu_dims).expect("target scratch");
    let mut a_neg_flat = vec![0.0f32; nl * di * ds];
    for (l, lw) in cpu_weights.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg_flat[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    let mut a_neg = GpuBuffer::zeros(&ctx.stream, nl * di * ds).expect("a_neg");
    a_neg
        .upload(&ctx.stream, &a_neg_flat)
        .expect("a_neg upload");
    let mut ip_gpu = GpuBuffer::zeros(&ctx.stream, batch * seq_len * dm).expect("ip_out");
    ip_gpu.upload(&ctx.stream, &ip_out).expect("ip_out upload");
    let mut out_gpu = GpuBuffer::zeros(&ctx.stream, batch * dm).expect("target out");
    gpu_forward_mamba_target_burnin(&ctx, &mut out_gpu, &ip_gpu, &gpu_w, &a_neg, &mut scratch)
        .expect("gpu target forward");
    ctx.stream.synchronize().expect("sync");
    let mut gpu_out = vec![0.0f32; batch * dm];
    out_gpu
        .download(&ctx.stream, &mut gpu_out)
        .expect("download");
    if rms_norm_eps == 1e-5 {
        eprintln!(
            "HASH target:m1:{scan_mode:?}:T{seq_len} {:016x}",
            digest::fnv1a_f32(&gpu_out)
        );
    }

    assert!(
        gpu_out.iter().any(|v| v.abs() > 1e-3),
        "{scan_mode:?} T={seq_len}: the GPU target output is all zeros"
    );
    let cs = cos_sim(&cpu_out, &gpu_out);
    let rel = max_rel_err_masked(&cpu_out, &gpu_out, 1e-4);
    assert!(
        cs > 0.9999 && rel < 1e-3,
        "{scan_mode:?} T={seq_len}: cos={cs} max_rel={rel}\ncpu={cpu_out:?}\ngpu={gpu_out:?}"
    );
}

#[test]
fn m1_target_forward_matches_cpu_on_the_sequential_route() {
    run_case(ScanMode::Sequential, 8, 1e-5);
}

#[test]
fn m1_target_forward_matches_cpu_on_the_parallel_route_across_chunks() {
    // Longer than one scan chunk, so the inter-chunk prefix path runs too.
    run_case(ScanMode::Parallel, 1100, 1e-5);
}

/// The CPU target takes the model's norm eps as the GPU target does: at an
/// eps far from the default the two still agree.
#[test]
fn m1_target_forward_matches_cpu_at_the_model_eps() {
    run_case(ScanMode::Sequential, 8, 1e-2);
}

fn m3_dims(cfg: &Mamba3Config, batch: usize, seq_len: usize, parallel: bool) -> GpuMamba3Dims {
    GpuMamba3Dims {
        batch,
        d_model: cfg.d_model,
        d_inner: cfg.d_inner(),
        d_state: cfg.d_state,
        nheads: cfg.nheads(),
        headdim: cfg.headdim,
        ngroups: cfg.ngroups,
        in_proj_dim: cfg.in_proj_out_dim(),
        seq_len,
        mamba_input_dim: cfg.d_model,
        n_layers: cfg.n_layers,
        n_angles: cfg.num_rope_angles(),
        a_floor: cfg.a_floor,
        rms_norm_eps: cfg.rms_norm_eps,
        ssm_norm_eps: cfg.ssm_norm_eps,
        block_norm: cfg.block_norm,
        is_outproj_norm: cfg.is_outproj_norm,
        use_parallel_scan: parallel,
    }
}

fn m3_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    }
}

fn run_m3_case(cfg: Mamba3Config, parallel: bool, seq_len: usize, max_rel: f32) {
    let dev = GpuDevice::new(0).expect("GpuDevice");
    let ctx = GpuCtx::new(&dev).expect("GpuCtx");
    let batch = 2;
    let input_dim = cfg.d_model;
    let m3k = Mamba3Kernels::compile_with_state_cap(
        dev.context(),
        arch::arch0(),
        mamba_rs::gpu::kernels::state_capacity(cfg.d_state).unwrap(),
    )
    .expect("Mamba3Kernels");
    let dims = m3_dims(&cfg, batch, seq_len, parallel);
    let bt = batch * seq_len;
    let dm = dims.d_model;
    let (nh, hd, ds, nl) = (dims.nheads, dims.headdim, dims.d_state, dims.n_layers);
    let na = dims.n_angles.max(1);
    let mut cpu_w = Mamba3Weights::init(&cfg, input_dim, 0x9060_0003);
    // A LayerNorm's biases start at zero; drawn ones put the bias add in
    // the rows too. An RMSNorm model has none.
    let biases = cpu_w
        .layers
        .iter_mut()
        .map(|lw| &mut lw.norm_bias)
        .chain(std::iter::once(&mut cpu_w.norm_f_bias));
    for (k, b) in biases.enumerate() {
        let d = det(b.len(), 0x1B + k as u32, 0.5);
        b.copy_from_slice(&d);
    }
    let label = if cfg.block_norm.rms_norm() {
        ""
    } else {
        ":layernorm"
    };
    let gpu_w = GpuMamba3Weights::from_cpu(&ctx.stream, &cpu_w, &cfg, input_dim)
        .expect("GpuMamba3Weights::from_cpu");
    let input = det(bt * input_dim, 0xCC, 0.05);
    let mut input_buf = GpuBuffer::zeros(&ctx.stream, bt * input_dim).expect("input");
    input_buf.upload(&ctx.stream, &input).expect("input upload");
    let exec = M3Exec {
        ctx: &ctx,
        kernels: &m3k,
        dims: &dims,
    };

    // Training backbone forward: the reference for the last timestep.
    let mut acts = GpuMamba3BackboneActs::new(&ctx.stream, &dims).expect("acts");
    let mut scratch = GpuMamba3Scratch::new(&ctx.stream, &dims).expect("scratch");
    let mut ssm = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd * ds).expect("ssm");
    let mut k_st = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * ds).expect("k");
    let mut v_st = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd).expect("v");
    let mut angle_st = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * na).expect("angle");
    let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * dm).expect("temporal");
    gpu_forward_mamba3_backbone(
        &exec,
        &mut temporal,
        &mut acts,
        &gpu_w,
        &input_buf,
        GpuMamba3StateBufs {
            ssm: &mut ssm,
            k: &mut k_st,
            v: &mut v_st,
            angle: &mut angle_st,
        },
        &mut scratch,
    )
    .expect("backbone forward");
    ctx.stream.synchronize().expect("sync");
    let full = temporal.to_cpu(&ctx.stream).expect("download");
    let mut reference = vec![0.0f32; batch * dm];
    for b in 0..batch {
        let last = (b * seq_len + seq_len - 1) * dm;
        reference[b * dm..(b + 1) * dm].copy_from_slice(&full[last..last + dm]);
    }

    // Target forward: the last timestep straight out.
    let mut tgt = GpuMamba3TargetScratch::new(&ctx.stream, &dims).expect("target scratch");
    let mut out = GpuBuffer::zeros(&ctx.stream, batch * dm).expect("target out");
    gpu_forward_mamba3_target_burnin(&exec, &mut out, &gpu_w, &input_buf, &mut tgt)
        .expect("target forward");
    ctx.stream.synchronize().expect("sync");
    let got = out.to_cpu(&ctx.stream).expect("download");
    eprintln!(
        "HASH target:m3:parallel={parallel}:T{seq_len}{label} {:016x}",
        digest::fnv1a_f32(&got)
    );

    assert!(
        got.iter().any(|v| v.abs() > 1e-3),
        "M3 parallel={parallel} T={seq_len}: the target output is all zeros"
    );
    let cs = cos_sim(&reference, &got);
    let rel = max_rel_err_masked(&reference, &got, 1e-4);
    assert!(
        cs > 0.9999 && rel < max_rel,
        "M3 parallel={parallel} T={seq_len}: cos={cs} max_rel={rel}\nbackbone={reference:?}\ntarget={got:?}"
    );
}

#[test]
fn m3_target_forward_matches_the_backbone_on_the_sequential_route() {
    run_m3_case(m3_cfg(), false, 8, 1e-4);
}

#[test]
fn m3_target_forward_matches_the_backbone_on_the_chunked_route() {
    // Longer than one chunk, so the chunk boundary carries state.
    run_m3_case(m3_cfg(), true, 300, 1e-3);
}

/// LayerNorm block norms with their biases, on a single head: the target
/// scratch keeps two statistics a row for them, more than the one head's
/// gated-norm statistics it is otherwise sized by.
#[test]
fn m3_target_forward_matches_the_backbone_with_layer_norm() {
    let cfg = Mamba3Config {
        headdim: 64,
        block_norm: mamba_rs::BlockNorm::Layer,
        ..m3_cfg()
    };
    assert_eq!(cfg.nheads(), 1);
    run_m3_case(cfg, false, 8, 1e-4);
    run_m3_case(cfg, true, 300, 1e-3);
}
