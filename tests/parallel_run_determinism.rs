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
    let (batch, input_dim) = (1usize, 48usize);
    let seq_len = PARALLEL_SCAN_THRESHOLD + 44;
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
    for tier in [
        GemmTier::Cublas,
        GemmTier::BatchInvariant,
        GemmTier::BatchInvariantTc,
    ] {
        let w = run_once(4, tier);
        // FNV-1a over the raw bits: order-sensitive, collision-safe enough
        // for an A/B, and dependency-free.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in &w {
            for b in v.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        println!("DIGEST {tier:?}: {h:016x} ({} weights)", w.len());
    }
}
