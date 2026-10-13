//! End-to-end data-parallel oracle on ONE GPU, no communicator: W
//! trainer replicas with identical weights each backward their own
//! micro-batch, the emulated world folds the gradient arenas with the
//! contract reduction, every replica applies the same reduced gradient
//! — and the replicas must stay bit-identical to each other, replay
//! bit-identically, and be immune to transport delivery order.

#![cfg(feature = "cuda")]

use mamba_rs::dist::EmulatedWorld;
use mamba_rs::gpu::GemmMode;
use mamba_rs::gpu::GpuOptions;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::train::BackwardMetrics;
use mamba_rs::gpu::train::BackwardOpts;
use mamba_rs::gpu::train::TrainSessionCfg;
use mamba_rs::mamba1::gpu::trainer::MambaGpuTrainer;
use mamba_rs::mamba1::{MambaConfig, MambaWeights};
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::trainer::Mamba3GpuTrainer;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::{
    Mamba2Config, Mamba2GpuTrainer, Mamba2Weights, Mamba3MimoConfig, Mamba3MimoGpuTrainer,
    Mamba3MimoWeights,
};

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
    let mut ranks: Vec<Mamba3GpuTrainer> = (0..world)
        .map(|_| {
            Mamba3GpuTrainer::new(
                &cpu,
                c,
                TrainSessionCfg::new(c.d_model, 1, seq_len),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap()
        })
        .collect();
    if bi_tier {
        for t in &ranks {
            t.ctx().set_gemm_mode(GemmMode::Deterministic).unwrap();
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
            t.upload_grad_arena(reduced).unwrap();
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
    // GEMM to the gemm_bi kernels by program text): replay stays
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

/// One rank of an emulated world, of any family.
trait Rank {
    fn forward(&mut self, input: &[f32], out: &mut [f32]);
    fn accumulate(&mut self, d: &[f32]) -> BackwardMetrics;
    fn arena(&self) -> Vec<f32>;
    fn apply(&mut self, reduced: &[f32]) -> BackwardMetrics;
    fn bits(&self) -> Vec<u32>;
    fn scaler(&self) -> Option<(f32, u32)>;
}

fn to_bits(values: impl IntoIterator<Item = f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

impl Rank for MambaGpuTrainer {
    fn forward(&mut self, input: &[f32], out: &mut [f32]) {
        MambaGpuTrainer::forward(self, input, out).unwrap();
    }
    fn accumulate(&mut self, d: &[f32]) -> BackwardMetrics {
        self.backward_step(d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap()
    }
    fn arena(&self) -> Vec<f32> {
        self.grad_arena().to_cpu(&self.ctx().stream).unwrap()
    }
    fn apply(&mut self, reduced: &[f32]) -> BackwardMetrics {
        self.upload_grad_arena(reduced).unwrap();
        self.apply_step(None).unwrap()
    }
    fn bits(&self) -> Vec<u32> {
        let w = self.snapshot_master().unwrap();
        let mut out = to_bits(w.input_proj_w.iter().chain(&w.input_proj_b).copied());
        for l in &w.layers {
            for t in [
                &l.norm_weight,
                &l.in_proj_w,
                &l.conv1d_weight,
                &l.conv1d_bias,
                &l.x_proj_w,
                &l.dt_proj_w,
                &l.dt_proj_b,
                &l.a_log,
                &l.d_param,
                &l.out_proj_w,
            ] {
                out.extend(to_bits(t.iter().copied()));
            }
        }
        out.extend(to_bits(w.norm_f_weight.iter().copied()));
        out
    }
    fn scaler(&self) -> Option<(f32, u32)> {
        self.scaler_state()
    }
}

impl Rank for Mamba3GpuTrainer {
    fn forward(&mut self, input: &[f32], out: &mut [f32]) {
        Mamba3GpuTrainer::forward(self, input, out).unwrap();
    }
    fn accumulate(&mut self, d: &[f32]) -> BackwardMetrics {
        self.backward_step(d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap()
    }
    fn arena(&self) -> Vec<f32> {
        self.grad_arena().to_cpu(&self.ctx().stream).unwrap()
    }
    fn apply(&mut self, reduced: &[f32]) -> BackwardMetrics {
        self.upload_grad_arena(reduced).unwrap();
        self.apply_step(None).unwrap()
    }
    fn bits(&self) -> Vec<u32> {
        weight_bits(&self.snapshot_master().unwrap())
    }
    fn scaler(&self) -> Option<(f32, u32)> {
        self.scaler_state()
    }
}

impl Rank for Mamba2GpuTrainer {
    fn forward(&mut self, input: &[f32], out: &mut [f32]) {
        Mamba2GpuTrainer::forward(self, input, out).unwrap();
    }
    fn accumulate(&mut self, d: &[f32]) -> BackwardMetrics {
        self.backward_step(d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap()
    }
    fn arena(&self) -> Vec<f32> {
        self.grad_arena()
            .unwrap()
            .to_cpu(&self.ctx().stream)
            .unwrap()
    }
    fn apply(&mut self, reduced: &[f32]) -> BackwardMetrics {
        self.upload_grad_arena(reduced).unwrap();
        self.apply_step(None).unwrap()
    }
    fn bits(&self) -> Vec<u32> {
        let w = self.snapshot_master().unwrap();
        let mut out = to_bits(w.input_proj_w.iter().chain(&w.input_proj_b).copied());
        for l in &w.layers {
            for t in [
                &l.norm_weight,
                &l.in_proj_w,
                &l.in_proj_b,
                &l.conv_w,
                &l.conv_b,
                &l.dt_bias,
                &l.a_log,
                &l.d_param,
                &l.gate_norm_weight,
                &l.out_proj_w,
                &l.out_proj_b,
            ] {
                out.extend(to_bits(t.iter().copied()));
            }
        }
        out.extend(to_bits(w.norm_f_weight.iter().copied()));
        out
    }
    fn scaler(&self) -> Option<(f32, u32)> {
        self.scaler_state()
    }
}

impl Rank for Mamba3MimoGpuTrainer {
    fn forward(&mut self, input: &[f32], out: &mut [f32]) {
        Mamba3MimoGpuTrainer::forward(self, input, out).unwrap();
    }
    fn accumulate(&mut self, d: &[f32]) -> BackwardMetrics {
        self.backward_step(d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap()
    }
    fn arena(&self) -> Vec<f32> {
        self.grad_arena()
            .unwrap()
            .to_cpu(&self.ctx().stream)
            .unwrap()
    }
    fn apply(&mut self, reduced: &[f32]) -> BackwardMetrics {
        self.upload_grad_arena(reduced).unwrap();
        self.apply_step(None).unwrap()
    }
    fn bits(&self) -> Vec<u32> {
        let w = self.snapshot_master().unwrap();
        let mut out = to_bits(w.input_proj_w.iter().chain(&w.input_proj_b).copied());
        for l in &w.layers {
            for t in [
                &l.norm_weight,
                &l.in_proj_w,
                &l.dt_bias,
                &l.b_norm_weight,
                &l.c_norm_weight,
                &l.b_bias,
                &l.c_bias,
                &l.d_param,
                &l.mimo_x,
                &l.mimo_z,
                &l.mimo_o,
                &l.norm_gate_weight,
                &l.out_proj_w,
            ] {
                out.extend(to_bits(t.iter().copied()));
            }
        }
        out.extend(to_bits(w.norm_f_weight.iter().copied()));
        out
    }
    fn scaler(&self) -> Option<(f32, u32)> {
        self.scaler_state()
    }
}

/// What one round of a world closed with on every rank: whether the
/// window stepped and the loss scale the round left behind.
type Round = (bool, Option<(f32, u32)>);

/// `rounds` data-parallel windows over the ranks, each rank on its own
/// micro-batch of `n` values, rank `blow.1` overflowing in round `blow.0`
/// (an f16 world). Every rank must close every round alike: the same skip
/// decision, the same scale and the same master bits. Returns rank 0's
/// final bits and the rounds.
fn run_world<R: Rank>(
    mut ranks: Vec<R>,
    n: usize,
    rounds: usize,
    blow: Option<(usize, usize)>,
) -> (Vec<u32>, Vec<Round>) {
    let world = ranks.len();
    let ew = EmulatedWorld::new(world).unwrap();
    let mut out = vec![0.0f32; n];
    let mut closed = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let mut arenas = Vec::with_capacity(world);
        for (r, t) in ranks.iter_mut().enumerate() {
            let input = det(n, 0x3000 + (round * world + r) as u32);
            // Small upstream gradients, so a window stays clear of the f16
            // range at the scaler's starting scale of 2^16.
            let mut d: Vec<f32> = det(n, 0x4000 + (round * world + r) as u32)
                .into_iter()
                .map(|v| v * 0.01)
                .collect();
            if blow == Some((round, r)) {
                d[0] = f32::MAX;
            }
            t.forward(&input, &mut out);
            assert!(!t.accumulate(&d).optimizer_stepped);
            arenas.push(t.arena());
        }
        ew.all_reduce_mean(&mut arenas, None).unwrap();
        let metrics: Vec<BackwardMetrics> = ranks
            .iter_mut()
            .zip(&arenas)
            .map(|(t, reduced)| t.apply(reduced))
            .collect();
        let m0 = &metrics[0];
        if m0.loss_scale.is_some() {
            assert_eq!(m0.overflow_skipped, Some(!m0.optimizer_stepped));
        }
        let bits0 = ranks[0].bits();
        let scale0 = ranks[0].scaler();
        for (r, (t, m)) in ranks.iter().zip(&metrics).enumerate().skip(1) {
            assert_eq!(
                (m.optimizer_stepped, m.loss_scale, m.overflow_skipped),
                (m0.optimizer_stepped, m0.loss_scale, m0.overflow_skipped),
                "round {round}: rank {r} took another decision than rank 0"
            );
            assert_eq!(t.scaler(), scale0, "round {round}: rank {r}'s scale");
            assert_eq!(t.bits(), bits0, "round {round}: rank {r}'s weights");
        }
        closed.push((m0.optimizer_stepped, scale0));
    }
    (ranks[0].bits(), closed)
}

const D_MODEL: usize = 32;

/// The f16 ranks' starting loss scale: 2^10 keeps a clean window's
/// gradients in f16's range at these shapes, so an overflow comes only from
/// the round that is made to overflow. No effect in the other precisions.
const F16_START_SCALE: f32 = 1024.0;

fn m1_ranks(world: usize, seq_len: usize, dtype: WeightDtype) -> Vec<MambaGpuTrainer> {
    let c = MambaConfig {
        d_model: D_MODEL,
        n_layers: 2,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        scan_mode: mamba_rs::config::ScanMode::Sequential,
        rms_norm_eps: 1e-5,
    };
    let w = MambaWeights::init(&c, D_MODEL, 0xC0FFEE);
    (0..world)
        .map(|_| {
            let mut t = MambaGpuTrainer::new(
                &w,
                c,
                TrainSessionCfg::new(D_MODEL, 1, seq_len),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap();
            t.load_scaler_state(F16_START_SCALE, 0);
            t
        })
        .collect()
}

fn siso_ranks(world: usize, seq_len: usize, dtype: WeightDtype) -> Vec<Mamba3GpuTrainer> {
    let c = cfg();
    let w = Mamba3Weights::init(&c, D_MODEL, 0xC0FFEE);
    (0..world)
        .map(|_| {
            let mut t = Mamba3GpuTrainer::new(
                &w,
                c,
                TrainSessionCfg::new(D_MODEL, 1, seq_len),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap();
            t.load_scaler_state(F16_START_SCALE, 0);
            t
        })
        .collect()
}

fn m2_ranks(world: usize, seq_len: usize, dtype: WeightDtype) -> Vec<Mamba2GpuTrainer> {
    let c = Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba2Config::default()
    };
    let w = Mamba2Weights::init(&c, D_MODEL, 0xC0FFEE);
    (0..world)
        .map(|_| {
            let mut t = Mamba2GpuTrainer::new(
                &w,
                c,
                TrainSessionCfg::new(D_MODEL, 1, seq_len),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap();
            t.load_scaler_state(F16_START_SCALE, 0);
            t
        })
        .collect()
}

fn mimo_ranks(world: usize, seq_len: usize, dtype: WeightDtype) -> Vec<Mamba3MimoGpuTrainer> {
    let c = Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    };
    let w = Mamba3MimoWeights::init(&c, D_MODEL, 0xC0FFEE);
    (0..world)
        .map(|_| {
            let mut t = Mamba3MimoGpuTrainer::new(
                &w,
                c,
                TrainSessionCfg::new(D_MODEL, 1, seq_len),
                GpuOptions::default().dtype(dtype),
            )
            .unwrap();
            t.load_scaler_state(F16_START_SCALE, 0);
            t
        })
        .collect()
}

type World = Box<dyn Fn(WeightDtype, Option<(usize, usize)>) -> (Vec<u32>, Vec<Round>)>;

/// A world of two ranks on 16-step windows for three rounds, per family.
fn worlds() -> [(&'static str, World); 4] {
    let (world, seq_len, rounds) = (2usize, 16usize, 3usize);
    let n = seq_len * D_MODEL;
    [
        (
            "Mamba-1",
            Box::new(move |dtype, blow| {
                run_world(m1_ranks(world, seq_len, dtype), n, rounds, blow)
            }),
        ),
        (
            "SISO",
            Box::new(move |dtype, blow| {
                run_world(siso_ranks(world, seq_len, dtype), n, rounds, blow)
            }),
        ),
        (
            "Mamba-2",
            Box::new(move |dtype, blow| {
                run_world(m2_ranks(world, seq_len, dtype), n, rounds, blow)
            }),
        ),
        (
            "MIMO",
            Box::new(move |dtype, blow| {
                run_world(mimo_ranks(world, seq_len, dtype), n, rounds, blow)
            }),
        ),
    ]
}

/// Every family keeps its ranks on one replica through the split seam
/// (accumulate, reduce, apply) and replays bit for bit.
#[test]
fn emulated_ddp_keeps_every_family_on_one_replica() {
    for (family, run) in &worlds() {
        for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
            let (bits, rounds) = run(dtype, None);
            assert!(
                rounds.iter().all(|r| r.0),
                "{family} {dtype:?}: a window skipped"
            );
            assert_eq!(run(dtype, None).0, bits, "{family} {dtype:?}: replay");
        }
    }
}

/// f16 data parallelism reduces the loss-scaled gradients, so an
/// overflow on one rank reaches every rank's arena as inf or NaN: every
/// rank skips that window, backs off to the same scale and keeps the same
/// weights, and the next window steps on every rank at the lower scale.
/// A clean world replays bit for bit.
#[test]
fn emulated_ddp_in_f16_takes_one_overflow_decision_on_every_rank() {
    for (family, run) in &worlds() {
        let (clean, clean_rounds) = run(WeightDtype::F16, None);
        assert!(
            clean_rounds.iter().all(|r| r.0),
            "{family}: a clean window skipped"
        );
        assert_eq!(
            run(WeightDtype::F16, None).0,
            clean,
            "{family}: a clean f16 world replays"
        );
        let (_, blown) = run(WeightDtype::F16, Some((1, 1)));
        assert!(
            blown[0].0 && !blown[1].0 && blown[2].0,
            "{family}: {blown:?}"
        );
        let scale = |r: &Round| r.1.expect("an f16 scaler").0;
        assert!(
            scale(&blown[1]) < scale(&blown[0]),
            "{family}: the overflow backed the scale off: {blown:?}"
        );
    }
}
