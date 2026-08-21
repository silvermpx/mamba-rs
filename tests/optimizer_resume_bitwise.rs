//! Bit-continuous training resume: a run that checkpoints weights AND
//! optimizer state at step k, reloads both into a fresh trainer, and
//! trains k more steps must land bit-for-bit where the unbroken 2k-step
//! run lands. A weights-only "resume" re-warms Adam from zero and
//! provably diverges — the control case asserts that divergence, so this
//! suite fails loudly if the comparison ever stops biting.

#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    }
}

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

/// Flatten every weight tensor to raw f32 bits for exact comparison.
fn weight_bits(w: &Mamba3Weights) -> Vec<u32> {
    let mut out = Vec::new();
    let mut push = |v: &[f32]| out.extend(v.iter().map(|x| x.to_bits()));
    push(&w.input_proj_w);
    push(&w.input_proj_b);
    for lw in &w.layers {
        push(&lw.norm_weight);
        push(&lw.in_proj_w);
        push(&lw.dt_bias);
        push(&lw.b_norm_weight);
        push(&lw.c_norm_weight);
        push(&lw.b_bias);
        push(&lw.c_bias);
        push(&lw.d_param);
        push(&lw.norm_gate_weight);
        push(&lw.out_proj_w);
    }
    push(&w.norm_f_weight);
    out
}

fn build(cpu: &Mamba3Weights, dtype: WeightDtype) -> Mamba3Trainer {
    let c = cfg();
    Mamba3Trainer::new_with_dtype(0, cpu, c.clone(), c.d_model, 1, 64, dtype).unwrap()
}

/// The f32 forward has no identity-proj branch (it always runs the GEMM),
/// while the mixed pipeline requires the identity branch — same identity
/// semantics, expressed per arm as eye() vs cleared.
fn base_weights(c: &Mamba3Config, dtype: WeightDtype) -> Mamba3Weights {
    let mut cpu = Mamba3Weights::init(c, c.d_model, 0xC0FFEE);
    if matches!(dtype, WeightDtype::F32) {
        cpu.input_proj_w = (0..c.d_model * c.d_model)
            .map(|i| {
                if i / c.d_model == i % c.d_model {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        cpu.input_proj_b = vec![0.0; c.d_model];
    } else {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }
    cpu
}

fn run_steps(trainer: &mut Mamba3Trainer, from: usize, to: usize, n: usize, dm_n: usize) {
    for s in from..to {
        let input = det(n, 0x1000 + s as u32);
        let d_temporal = det(dm_n, 0x2000 + s as u32);
        trainer.step(&input, &d_temporal).unwrap();
    }
}

fn check_resume(dtype: WeightDtype) {
    let c = cfg();
    let n = 64 * c.d_model;
    let dm_n = 64 * c.d_model;
    let cpu = base_weights(&c, dtype);
    let k = 3usize;

    // Unbroken run: 2k steps.
    let mut a = build(&cpu, dtype);
    run_steps(&mut a, 0, 2 * k, n, dm_n);
    let w_a = weight_bits(&a.snapshot_master().unwrap());
    let blob_a = a.optimizer_state().unwrap();

    // Checkpointed run: k steps, export weights + optimizer, rebuild,
    // import, k more steps.
    let mut b1 = build(&cpu, dtype);
    run_steps(&mut b1, 0, k, n, dm_n);
    let w_mid = b1.snapshot_master().unwrap();
    let blob_mid = b1.optimizer_state().unwrap();
    assert_eq!(blob_mid.step, k as u64, "step counter rides the blob");
    drop(b1);

    let mut b2 = build(&w_mid, dtype);
    b2.load_optimizer_state(&blob_mid).unwrap();
    run_steps(&mut b2, k, 2 * k, n, dm_n);
    let w_b = weight_bits(&b2.snapshot_master().unwrap());
    let blob_b = b2.optimizer_state().unwrap();

    assert_eq!(
        w_a, w_b,
        "resumed weights must equal the unbroken run bit-for-bit"
    );
    assert_eq!(blob_a.step, blob_b.step);
    assert_eq!(
        blob_a.m.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        blob_b.m.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "Adam m must be bit-continuous"
    );
    assert_eq!(
        blob_a.v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        blob_b.v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "Adam v must be bit-continuous"
    );

    // Control: weights-only resume (optimizer re-warmed from zero) must
    // NOT reproduce the unbroken run — otherwise this suite tests nothing.
    let mut c2 = build(&w_mid, dtype);
    run_steps(&mut c2, k, 2 * k, n, dm_n);
    let w_c = weight_bits(&c2.snapshot_master().unwrap());
    assert_ne!(
        w_a, w_c,
        "weights-only resume unexpectedly matched the unbroken run — the comparison lost its teeth"
    );
}

#[test]
fn optimizer_resume_is_bit_continuous_f32() {
    check_resume(WeightDtype::F32);
}

#[test]
fn optimizer_resume_is_bit_continuous_bf16() {
    check_resume(WeightDtype::Bf16);
}

#[test]
fn optimizer_state_len_mismatch_is_refused() {
    let c = cfg();
    let cpu = base_weights(&c, WeightDtype::F32);
    let mut t = build(&cpu, WeightDtype::F32);
    let mut blob = t.optimizer_state().unwrap();
    blob.m.pop();
    let err = t.load_optimizer_state(&blob).unwrap_err();
    assert!(err.contains("different parameterization"), "{err}");
}

/// The split pair — window-closing `backward_step(accumulate_only=true)`
/// followed by `apply_step` — must be bit-identical to a single applying
/// `backward_step`. This is the seam a distributed gradient reducer
/// slots into; if the pair ever drifts from the fused call, distributed
/// and single-process training silently diverge.
#[test]
fn apply_step_pair_matches_fused_applying_backward() {
    use mamba_rs::mamba_ssm::gpu::trainer::BackwardOpts;

    let c = cfg();
    let n = 64 * c.d_model;
    let cpu = base_weights(&c, WeightDtype::Bf16);
    let input = det(n, 0x51);
    let d_temporal = det(n, 0x52);
    let mut out = vec![0.0f32; n];

    let mut fused = build(&cpu, WeightDtype::Bf16);
    fused.forward(&input, &mut out).unwrap();
    fused
        .backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1.0))
        .unwrap();
    let w_fused = weight_bits(&fused.snapshot_master().unwrap());

    let mut split = build(&cpu, WeightDtype::Bf16);
    split.forward(&input, &mut out).unwrap();
    let m1 = split
        .backward_step(
            &d_temporal,
            BackwardOpts::default().with_accumulate_only(true),
        )
        .unwrap();
    assert!(!m1.optimizer_stepped);
    // (a reducer would sum grad_arena() across ranks right here)
    let _ = split.grad_arena().len();
    let m2 = split.apply_step(Some(1.0)).unwrap();
    assert!(m2.optimizer_stepped);
    let w_split = weight_bits(&split.snapshot_master().unwrap());

    assert_eq!(
        w_fused, w_split,
        "split accumulate+apply_step must equal the fused applying backward bit-for-bit"
    );

    // A second apply_step on the closed window is refused.
    let err = split.apply_step(None).unwrap_err();
    assert!(err.contains("open accumulation window"), "{err}");
}

/// The M1 (sequential-scan) family carries conv + SSM state across
/// steps, so its bit-continuous resume needs THREE blobs: weights,
/// optimizer, and the carried recurrence. This is the TBPTT window
/// handoff contract.
#[test]
fn m1_resume_with_recurrent_state_is_bit_continuous() {
    use mamba_rs::config::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::trainer::MambaTrainer;
    use mamba_rs::weights::MambaWeights;

    fn m1_cfg() -> MambaConfig {
        MambaConfig {
            d_model: 32,
            n_layers: 1,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            scan_mode: mamba_rs::config::ScanMode::Sequential,
            rms_norm_eps: 1e-5,
        }
    }
    fn m1_weights(c: &MambaConfig) -> MambaWeights {
        let mut cpu = MambaWeights::init(c, c.d_model, 0xF32C0FF);
        for lw in cpu.layers.iter_mut() {
            lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
        }
        cpu
    }
    fn m1_bits(w: &MambaWeights) -> Vec<u32> {
        let mut out = Vec::new();
        let mut push = |v: &[f32]| out.extend(v.iter().map(|x| x.to_bits()));
        push(&w.input_proj_w);
        push(&w.input_proj_b);
        for lw in &w.layers {
            push(&lw.norm_weight);
            push(&lw.in_proj_w);
            push(&lw.conv1d_weight);
            push(&lw.conv1d_bias);
            push(&lw.x_proj_w);
            push(&lw.dt_proj_w);
            push(&lw.dt_proj_b);
            push(&lw.a_log);
            push(&lw.d_param);
            push(&lw.out_proj_w);
        }
        push(&w.norm_f_weight);
        out
    }
    fn m1_build(cpu: &MambaWeights, c: &MambaConfig) -> MambaTrainer {
        MambaTrainer::new_with_dtype(0, cpu, c.clone(), c.d_model, 1, 4, WeightDtype::F32).unwrap()
    }

    let c = m1_cfg();
    let n = 4 * c.d_model;
    let cpu = m1_weights(&c);

    let mut a = m1_build(&cpu, &c);
    for s in 0..2u32 {
        a.step(&det(n, 0x31 + s), &det(n, 0x41 + s)).unwrap();
    }
    let w_a = m1_bits(&a.snapshot_master().unwrap());
    let rec_a = a.recurrent_state().unwrap();

    let mut b1 = m1_build(&cpu, &c);
    b1.step(&det(n, 0x31), &det(n, 0x41)).unwrap();
    let w_mid = b1.snapshot_master().unwrap();
    let blob_opt = b1.optimizer_state().unwrap();
    let blob_rec = b1.recurrent_state().unwrap();
    drop(b1);

    let mut b2 = m1_build(&w_mid, &c);
    b2.load_optimizer_state(&blob_opt).unwrap();
    b2.load_recurrent_state(&blob_rec).unwrap();
    b2.step(&det(n, 0x32), &det(n, 0x42)).unwrap();
    let w_b = m1_bits(&b2.snapshot_master().unwrap());
    let rec_b = b2.recurrent_state().unwrap();

    assert_eq!(w_a, w_b, "M1 resumed weights must be bit-continuous");
    assert_eq!(
        rec_a
            .ssm_states
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        rec_b
            .ssm_states
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        "carried SSM state must be bit-continuous"
    );
    assert_eq!(
        rec_a
            .conv_states
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        rec_b
            .conv_states
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        "carried conv state must be bit-continuous"
    );
}
