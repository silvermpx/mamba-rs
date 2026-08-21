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
