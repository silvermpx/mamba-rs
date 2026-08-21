//! End-to-end data-parallel oracle on ONE GPU, no communicator: W
//! trainer replicas with identical weights each backward their own
//! micro-batch, the emulated world folds the gradient arenas with the
//! contract reduction, every replica applies the same reduced gradient
//! — and the replicas must stay bit-identical to each other, replay
//! bit-identically, and be immune to transport delivery order.

#![cfg(feature = "cuda")]

use mamba_rs::dist::EmulatedWorld;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::BackwardOpts;
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

/// Run `rounds` data-parallel optimizer steps across `world` emulated
/// ranks on one GPU and return the final master-weight bits (identical
/// on every rank — asserted inside). `permute_delivery` scrambles the
/// order the fold receives peer contributions in. `bi_tier` flips every
/// rank onto the batch-invariant house GEMM kernels — the reduction
/// must compose with any per-rank compute tier unchanged.
fn run_emulated_ddp_tier(
    world: usize,
    rounds: usize,
    permute_delivery: bool,
    bi_tier: bool,
    dtype: WeightDtype,
) -> Vec<u32> {
    let c = cfg();
    let seq_len = 64usize;
    let n = seq_len * c.d_model;
    let mut cpu = Mamba3Weights::init(&c, c.d_model, 0xC0FFEE);
    if dtype == WeightDtype::F32 {
        // The f32 backbone has no identity-proj branch — it always runs
        // the input-projection GEMM; eye(d_model) + zero bias is the
        // pass-through. Empty-means-identity is the mixed convention.
        let dm = c.d_model;
        let mut eye = vec![0.0f32; dm * dm];
        for i in 0..dm {
            eye[i * dm + i] = 1.0;
        }
        cpu.input_proj_w = eye;
        cpu.input_proj_b = vec![0.0f32; dm];
    } else {
        cpu.input_proj_w.clear();
        cpu.input_proj_b.clear();
    }

    let ew = EmulatedWorld::new(world).unwrap();
    let mut ranks: Vec<Mamba3Trainer> = (0..world)
        .map(|_| Mamba3Trainer::new_with_dtype(0, &cpu, c, c.d_model, 1, seq_len, dtype).unwrap())
        .collect();
    if bi_tier {
        for t in &ranks {
            t.ctx().set_batch_invariant(true);
        }
    }

    let mut out = vec![0.0f32; n];
    for round in 0..rounds {
        // Each rank backwards its own micro-batch and leaves the window
        // open; the emulated world then folds the arenas and every rank
        // applies the identical reduced gradient.
        let mut arenas: Vec<Vec<f32>> = Vec::with_capacity(world);
        for (r, t) in ranks.iter_mut().enumerate() {
            let input = det(n, 0x1000 + (round * world + r) as u32);
            let d_temporal = det(n, 0x2000 + (round * world + r) as u32);
            t.forward(&input, &mut out).unwrap();
            let m = t
                .backward_step(
                    &d_temporal,
                    BackwardOpts::default().with_accumulate_only(true),
                )
                .unwrap();
            assert!(!m.optimizer_stepped);
            let stream = t.ctx().stream.clone();
            arenas.push(t.grad_arena().to_cpu(&stream).unwrap());
        }
        let delivery: Option<Vec<usize>> = if permute_delivery {
            Some((0..world).rev().collect())
        } else {
            None
        };
        ew.all_reduce_mean(&mut arenas, delivery.as_deref())
            .unwrap();
        for (t, reduced) in ranks.iter_mut().zip(&arenas) {
            let stream = t.ctx().stream.clone();
            t.grad_arena().upload(&stream, reduced).unwrap();
            let m = t.apply_step(None).unwrap();
            assert!(m.optimizer_stepped);
        }
        // Replica law: every rank holds identical master weights after
        // every applied step.
        let w0 = weight_bits(&ranks[0].snapshot_master().unwrap());
        for (r, t) in ranks.iter().enumerate().skip(1) {
            let wr = weight_bits(&t.snapshot_master().unwrap());
            assert_eq!(
                w0, wr,
                "round {round}: rank {r} master weights diverged from rank 0"
            );
        }
    }
    weight_bits(&ranks[0].snapshot_master().unwrap())
}

#[test]
fn emulated_ddp_replays_bitwise_and_ignores_delivery_order() {
    let a = run_emulated_ddp_tier(2, 3, false, false, WeightDtype::Bf16);
    let b = run_emulated_ddp_tier(2, 3, false, false, WeightDtype::Bf16);
    assert_eq!(a, b, "same logical run must replay bit-identically");
    let c = run_emulated_ddp_tier(2, 3, true, false, WeightDtype::Bf16);
    assert_eq!(
        a, c,
        "transport delivery order leaked into the reduced gradient"
    );
}

#[test]
fn emulated_ddp_composes_with_the_deterministic_gemm_tier() {
    // The reducer consumes finished gradient arenas and never
    // participates in how they were computed, so every per-rank GEMM
    // tier must ride DDP unchanged. Pin the batch-invariant house tier
    // on both trainer dtypes (on f32 the flag routes every projection
    // GEMM to the sgemm_bi kernels by program text): replay stays
    // bitwise and the fold stays delivery-order immune. No cross-tier
    // bit-difference assertion — at these small shapes two correct
    // GEMMs may legitimately agree bitwise, so a difference is an
    // implementation coincidence, not a contract.
    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        let a = run_emulated_ddp_tier(2, 2, false, true, dtype);
        let b = run_emulated_ddp_tier(2, 2, false, true, dtype);
        assert_eq!(
            a, b,
            "deterministic-tier DDP must replay bit-identically ({dtype:?})"
        );
        let c = run_emulated_ddp_tier(2, 2, true, true, dtype);
        assert_eq!(
            a, c,
            "transport delivery order leaked into the deterministic-tier reduction ({dtype:?})"
        );
    }
}

#[test]
fn emulated_ddp_world_size_is_a_numeric_route() {
    // Different logical world sizes fold the same global data with a
    // different tree — the bits are ALLOWED to differ, and do. This
    // test pins the honest scope: W is part of the numeric identity,
    // not a free deployment knob.
    let w2 = run_emulated_ddp_tier(2, 2, false, false, WeightDtype::Bf16);
    let w4 = run_emulated_ddp_tier(4, 2, false, false, WeightDtype::Bf16);
    assert_ne!(
        w2, w4,
        "different logical world sizes unexpectedly produced identical bits — \
         if this ever holds, the honest-scope statement can be strengthened"
    );
}
