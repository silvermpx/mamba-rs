//! Run-to-run bit determinism of TRAINING in the parallel-scan regime.
//!
//! The existing determinism pins run small-T shapes that dispatch the
//! sequential kernels; this suite pins the same law for the parallel scan
//! (T > PARALLEL_SCAN_THRESHOLD, ScanMode::Auto): two identical training
//! runs from identical init must produce bit-identical master weights.
//!
//! BOTH GEMM tiers are covered, because both ship: the cuBLAS tier (crate
//! default) and the batch-invariant custom-kernel tier with its tensor-core
//! variant (`set_batch_invariant` + `set_bi_tensor_cores` — what the
//! production classifier trainer enables). A tier is only pinned by the
//! test that actually selects it.

#![cfg(feature = "cuda")]

mod common;

use mamba_rs::config::{MambaConfig, ScanMode};
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::forward::PARALLEL_SCAN_THRESHOLD;
use mamba_rs::mamba_ssm::gpu::trainer::{BackwardOpts, MambaTrainer};
use mamba_rs::weights::MambaWeights;

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

fn flatten(w: &MambaWeights) -> Vec<f32> {
    let mut out = Vec::new();
    out.extend_from_slice(&w.input_proj_w);
    out.extend_from_slice(&w.input_proj_b);
    for l in &w.layers {
        out.extend_from_slice(&l.norm_weight);
        out.extend_from_slice(&l.in_proj_w);
        out.extend_from_slice(&l.conv1d_weight);
        out.extend_from_slice(&l.conv1d_bias);
        out.extend_from_slice(&l.x_proj_w);
        out.extend_from_slice(&l.dt_proj_w);
        out.extend_from_slice(&l.dt_proj_b);
        out.extend_from_slice(&l.a_log);
    }
    out.extend_from_slice(&w.norm_f_weight);
    out
}

/// Which GEMM tier the run selects. Both ship; both must be deterministic.
#[derive(Clone, Copy, Debug)]
enum GemmTier {
    /// Crate default: cuBLAS GemmEx.
    Cublas,
    /// Batch-invariant custom kernels, scalar tier.
    BatchInvariant,
    /// Batch-invariant custom kernels, tensor-core tier (production).
    BatchInvariantTc,
}

fn run_once(steps: usize, tier: GemmTier) -> Vec<f32> {
    run_once_at(steps, tier, PARALLEL_SCAN_THRESHOLD + 44)
}

/// Shape-parameterized twin: the historical digest shape (T=300) runs a
/// SINGLE parallel-scan chunk, so the inter-chunk carry — exactly what
/// the S2 tape work rewires — was outside the instrument. Multi-chunk
/// arms pin it.
fn run_once_at(steps: usize, tier: GemmTier, seq_len: usize) -> Vec<f32> {
    let (batch, input_dim) = (1usize, 48usize);
    let cfg = MambaConfig {
        d_model: 64,
        n_layers: 2,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    };
    let mut cpu = MambaWeights::init(&cfg, input_dim, 0xC0FFEE);
    for lw in cpu.layers.iter_mut() {
        lw.a_neg = lw.a_log.iter().map(|&v| -v.exp()).collect();
    }
    let mut trainer =
        MambaTrainer::new_with_dtype(0, &cpu, cfg, input_dim, batch, seq_len, WeightDtype::Bf16)
            .expect("trainer");
    match tier {
        GemmTier::Cublas => {}
        GemmTier::BatchInvariant => trainer.ctx().set_batch_invariant(true),
        GemmTier::BatchInvariantTc => {
            trainer.ctx().set_batch_invariant(true);
            trainer.ctx().set_bi_tensor_cores(true);
        }
    }
    let mut temporal = vec![0.0f32; batch * seq_len * cfg.d_model];
    for s in 0..steps {
        let input = det(batch * seq_len * input_dim, 0xA5 + s as u32, 0.05);
        let d_temporal = det(batch * seq_len * cfg.d_model, 0xB6 + s as u32, 0.01);
        trainer.forward(&input, &mut temporal).expect("fwd");
        trainer
            .backward_step(&d_temporal, BackwardOpts::default())
            .expect("bwd");
    }
    flatten(&trainer.snapshot_master().expect("snapshot"))
}

fn assert_reproducible(tier: GemmTier) {
    let a = run_once(4, tier);
    let b = run_once(4, tier);
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{tier:?}: master weight [{i}] diverged across identical parallel-mode runs: \
             {x} vs {y}"
        );
    }
}

#[test]
fn parallel_training_is_bit_reproducible_cublas() {
    assert_reproducible(GemmTier::Cublas);
}

#[test]
fn parallel_training_is_bit_reproducible_batch_invariant() {
    assert_reproducible(GemmTier::BatchInvariant);
}

/// The production classifier tier: batch-invariant custom kernels with the
/// tensor-core variant enabled (classify_trainer sets exactly this pair).
#[test]
fn parallel_training_is_bit_reproducible_batch_invariant_tc() {
    assert_reproducible(GemmTier::BatchInvariantTc);
}

/// The two BI tiers have SEPARATE numeric contracts (the tensor-core
/// reduction tree is not the scalar FMA chain), so they must NOT be
/// expected to agree bit-for-bit — pin that they differ deliberately
/// rather than by accident, and that neither matches cuBLAS.
#[test]
fn gemm_tiers_are_distinct_numeric_routes() {
    let cublas = run_once(2, GemmTier::Cublas);
    let bi = run_once(2, GemmTier::BatchInvariant);
    let bi_tc = run_once(2, GemmTier::BatchInvariantTc);
    let differs = |a: &[f32], b: &[f32]| a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(
        differs(&cublas, &bi),
        "cuBLAS vs BI-scalar unexpectedly bit-equal"
    );
    assert!(
        differs(&bi, &bi_tc),
        "BI-scalar vs BI-tensor-core unexpectedly bit-equal"
    );
}

/// Manual (--ignored): stable digest of a fixed parallel-mode training run
/// per GEMM tier. Used to A/B two BUILDS (e.g. before/after a kernel edit
/// that must not move bits) — run on both, diff the printed lines.
#[test]
#[ignore]
fn print_run_digests() {
    use common::bench::fnv1a_f32;
    for tier in [
        GemmTier::Cublas,
        GemmTier::BatchInvariant,
        GemmTier::BatchInvariantTc,
    ] {
        let w = run_once(4, tier);
        let h = fnv1a_f32(&w);
        println!("DIGEST {tier:?}: {h:016x} ({} weights)", w.len());
        common::evidence::record_digest(
            "parallel_run_determinism",
            "run_digests",
            &format!("{tier:?}"),
            h,
        );
    }
}

/// Multi-chunk (T=1300 -> 2 chunks) run-to-run bit identity on the
/// batch-invariant tier — the production shape's chunk count.
#[test]
fn multichunk_run_to_run_bit_identical_bi() {
    let a = run_once_at(3, GemmTier::BatchInvariant, 1300);
    let b = run_once_at(3, GemmTier::BatchInvariant, 1300);
    let diverged = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(diverged, 0, "multichunk BI run-to-run diverged");
}

/// Digest printer for the multi-chunk shapes (2 and 3 chunks) — the A/B
/// instrument for tape-layout work. Compare across BUILDS, not runs.
#[test]
#[ignore = "digest printer for cross-build A/B"]
fn print_run_digests_multichunk() {
    use common::bench::fnv1a_f32;
    for (label, t) in [("T1300(2ch)", 1300usize), ("T2100(3ch)", 2100)] {
        for tier in [
            GemmTier::Cublas,
            GemmTier::BatchInvariant,
            GemmTier::BatchInvariantTc,
        ] {
            let w = run_once_at(3, tier, t);
            let h = fnv1a_f32(&w);
            println!("DIGEST-MC {label} {tier:?}: {h:016x} ({} weights)", w.len());
            common::evidence::record_digest(
                "parallel_run_determinism",
                "run_digests_multichunk",
                &format!("{label}.{tier:?}"),
                h,
            );
        }
    }
}
