//! The matrix on the device: every family over configurations that differ
//! in each shape feature its kernels route on (one feature wide, head
//! widths of one lane, of a warp and of a block, odd widths, the largest
//! state, conv widths past the fast kernels, groups, ranks, the gated MLP,
//! the output norm, a zero A floor), on every path (the decode step, a prompt, a training window),
//! in a covering set of scan modes, storage precisions and GEMM modes:
//! every pair of a scan mode and a precision, of a precision and a GEMM
//! mode, and of a scan mode and a GEMM mode runs on every configuration
//! and path.
//!
//! Each cell asks `admit` first, which takes every one of these small
//! shapes, then runs on the GPU and is checked against the host, which
//! runs every configuration `validate` accepts: the decode and prompt rows
//! within the cell's precision, and a training window's global gradient
//! norm within it, the window applied. Where the host itself moves further
//! than that when its inputs move by one rounding of the cell's precision
//! (a model one feature wide normalizes a single value, whose gradient
//! peaks where the stream crosses zero), the f32 host cannot judge the
//! cell more closely than its own spread, and the cell is held to a few
//! times that spread instead; the run lists those cells.
//!
//! A bf16 or f16 cell is also held to the host computing in its own
//! precision, which rounds where the device stores: within a few roundings
//! of that precision, the distance a rounding flipped by the two sides'
//! summation orders spreads to through the later stores. That host stays
//! close where the f32 host's spread is wide, so the ill-conditioned cells
//! keep a tight check.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::mamba1::cpu::prefill::PrefillMode;
use mamba_rs::matrix::{Path, Request};
use mamba_rs::{
    AnyCpuTrainer, AnyGpuBackbone, AnyGpuTrainer, BackwardOpts, CpuOptions, CpuTrainer, GemmMode,
    GpuBackbone, GpuOptions, GpuTrainer, Mamba2Backbone, Mamba2Config, Mamba2CpuTrainer,
    Mamba2GpuBackbone, Mamba2GpuTrainer, Mamba2Weights, Mamba3Backbone, Mamba3Config,
    Mamba3CpuTrainer, Mamba3GpuBackbone, Mamba3GpuTrainer, Mamba3MimoBackbone, Mamba3MimoConfig,
    Mamba3MimoCpuTrainer, Mamba3MimoGpuBackbone, Mamba3MimoGpuTrainer, Mamba3MimoWeights,
    Mamba3Weights, MambaBackbone, MambaConfig, MambaCpuTrainer, MambaGpuBackbone, MambaGpuTrainer,
    MambaWeights, TrainSessionCfg, WeightDtype,
};

const BATCH: usize = 2;
/// Wider than any model's d_model here but the widest, so every family
/// runs its input projection.
const INPUT_DIM: usize = 24;
/// Past the automatic threshold: `Auto` takes the parallel form.
const WINDOW: usize = 65;
const DECODE_STEPS: usize = 3;

const SCANS: [ScanMode; 3] = [ScanMode::Auto, ScanMode::Sequential, ScanMode::Parallel];
const PRECISIONS: [WeightDtype; 4] = [
    WeightDtype::F32,
    WeightDtype::Tf32,
    WeightDtype::Bf16,
    WeightDtype::F16,
];
const GEMMS: [GemmMode; 3] = [
    GemmMode::Deterministic,
    GemmMode::CublasFast,
    GemmMode::CublasPedantic,
];

/// The cells of one configuration and path. A window takes every
/// precision with every scan mode, the GEMM mode rotating so each pair of
/// the three settings meets at least once; the decode step is the
/// recurrence whatever the mode, so it takes every precision with every
/// GEMM mode.
fn cells(path: Path) -> Vec<(ScanMode, WeightDtype, GemmMode)> {
    let mut out = Vec::new();
    for (p, &precision) in PRECISIONS.iter().enumerate() {
        if path == Path::Decode {
            out.extend(GEMMS.map(|gemm| (ScanMode::Auto, precision, gemm)));
            continue;
        }
        for (s, &scan) in SCANS.iter().enumerate() {
            out.push((scan, precision, GEMMS[(p + s) % GEMMS.len()]));
        }
    }
    out
}

/// How far the device may land from the host, relative to the largest
/// value the host gives: the storage precision's rounding through a few
/// layers, and the vendor's TF32 under `CublasFast`.
fn tolerance(precision: WeightDtype, gemm: GemmMode) -> f32 {
    let storage: f32 = match precision {
        WeightDtype::F32 => 2e-3,
        WeightDtype::Tf32 => 3e-2,
        WeightDtype::F16 => 3e-2,
        WeightDtype::Bf16 => 1e-1,
    };
    if gemm == GemmMode::CublasFast {
        storage.max(3e-2)
    } else {
        storage
    }
}

/// How far a precision rounds a stored value, relative: half the gap
/// between its neighbours at one. f32 is the host's own arithmetic.
fn unit_roundoff(precision: WeightDtype) -> f32 {
    match precision {
        WeightDtype::F32 => 0.0,
        WeightDtype::Tf32 | WeightDtype::F16 => 2f32.powi(-11),
        WeightDtype::Bf16 => 2f32.powi(-8),
    }
}

/// The seeds of the host's nudged runs, one set of signs each.
const NUDGES: [u32; 2] = [1001, 1002];

/// How many times the host's own spread a cell may land from it.
const SPREAD_FACTOR: f32 = 4.0;

/// How many roundings of its precision a half cell may land from the host
/// computing in that precision. The census measured at most about two
/// (bf16) and three and a half (f16) on the decode and the prompt, where a
/// flipped rounding spreads furthest, and far less on training windows.
const LANE_ROUNDINGS: f32 = 8.0;

/// `v` with every value moved by `u` relative, up or down as `seed` draws.
fn nudged(v: &[f32], u: f32, seed: u32) -> Vec<f32> {
    v.iter()
        .zip(values(v.len(), seed))
        .map(|(x, s)| x * (1.0 + u * s.signum()))
        .collect()
}

/// A cell's distance from the host, how far the host moves itself when
/// its inputs move by one rounding of the cell's precision, and in a half
/// precision the cell's distance from the host computing in it.
struct Check {
    diff: f32,
    spread: f32,
    lane: Option<f32>,
}

/// Whether the host runs a lane of `precision` beside its f32 run: the
/// half precisions, whose GPU stores the host's lane rounds as it does.
fn has_lane(precision: WeightDtype) -> bool {
    matches!(precision, WeightDtype::Bf16 | WeightDtype::F16)
}

#[derive(Clone, Copy, Debug)]
enum Cfg {
    M1(MambaConfig),
    M2(Mamba2Config),
    Siso(Mamba3Config),
    Mimo(Mamba3MimoConfig),
}

enum Weights {
    M1(MambaWeights),
    M2(Mamba2Weights),
    Siso(Mamba3Weights),
    Mimo(Mamba3MimoWeights),
}

/// The host backbone of a family and one sequence's recurrence on it.
enum Host {
    M1(MambaBackbone),
    M2(Mamba2Backbone),
    Siso(Mamba3Backbone),
    Mimo(Mamba3MimoBackbone),
}

impl Cfg {
    fn validate(&self) -> Result<(), String> {
        match self {
            Cfg::M1(c) => c.validate(),
            Cfg::M2(c) => c.validate(),
            Cfg::Siso(c) => c.validate(),
            Cfg::Mimo(c) => c.validate(),
        }
    }

    fn d_model(&self) -> usize {
        match self {
            Cfg::M1(c) => c.d_model,
            Cfg::M2(c) => c.d_model,
            Cfg::Siso(c) => c.d_model,
            Cfg::Mimo(c) => c.d_model,
        }
    }

    fn admit(&self, req: &Request) -> bool {
        let verdict = match self {
            Cfg::M1(c) => mamba_rs::mamba1::route::admit(c, INPUT_DIM, req),
            Cfg::M2(c) => mamba_rs::mamba2::route::admit(c, INPUT_DIM, req),
            Cfg::Siso(c) => mamba_rs::mamba3_siso::route::admit(c, INPUT_DIM, req),
            Cfg::Mimo(c) => mamba_rs::mamba3_mimo::route::admit(c, INPUT_DIM, req),
        };
        verdict.is_ok()
    }

    fn weights(&self, seed: u64) -> Weights {
        match self {
            Cfg::M1(c) => Weights::M1(MambaWeights::init(c, INPUT_DIM, seed)),
            Cfg::M2(c) => Weights::M2(Mamba2Weights::init(c, INPUT_DIM, seed)),
            Cfg::Siso(c) => Weights::Siso(Mamba3Weights::init(c, INPUT_DIM, seed)),
            Cfg::Mimo(c) => Weights::Mimo(Mamba3MimoWeights::init(c, INPUT_DIM, seed)),
        }
    }

    /// The host backbone computing in `precision`.
    fn host(&self, w: &Weights, precision: WeightDtype) -> Host {
        let o = CpuOptions::default().dtype(precision);
        match (self, w) {
            (Cfg::M1(c), Weights::M1(w)) => {
                Host::M1(MambaBackbone::new_with(w.clone(), *c, INPUT_DIM, o).unwrap())
            }
            (Cfg::M2(c), Weights::M2(w)) => {
                Host::M2(Mamba2Backbone::new_with(w.clone(), *c, INPUT_DIM, o).unwrap())
            }
            (Cfg::Siso(c), Weights::Siso(w)) => {
                Host::Siso(Mamba3Backbone::new_with(w.clone(), *c, INPUT_DIM, o).unwrap())
            }
            (Cfg::Mimo(c), Weights::Mimo(w)) => {
                Host::Mimo(Mamba3MimoBackbone::new_with(w.clone(), *c, INPUT_DIM, o).unwrap())
            }
            _ => unreachable!("weights of another family"),
        }
    }

    fn gpu(&self, w: &Weights, opts: GpuOptions) -> Result<AnyGpuBackbone, String> {
        Ok(match (self, w) {
            (Cfg::M1(c), Weights::M1(w)) => {
                MambaGpuBackbone::new(w, *c, INPUT_DIM, BATCH, opts)?.into()
            }
            (Cfg::M2(c), Weights::M2(w)) => {
                Mamba2GpuBackbone::new(w, *c, INPUT_DIM, BATCH, opts)?.into()
            }
            (Cfg::Siso(c), Weights::Siso(w)) => {
                Mamba3GpuBackbone::new(w, *c, INPUT_DIM, BATCH, opts)?.into()
            }
            (Cfg::Mimo(c), Weights::Mimo(w)) => {
                Mamba3MimoGpuBackbone::new(w, *c, INPUT_DIM, BATCH, opts)?.into()
            }
            _ => unreachable!("weights of another family"),
        })
    }

    /// The host trainer computing in `precision`.
    fn host_trainer(&self, w: &Weights, precision: WeightDtype) -> Result<AnyCpuTrainer, String> {
        let s = TrainSessionCfg::new(INPUT_DIM, BATCH, WINDOW);
        let o = CpuOptions::default().dtype(precision);
        Ok(match (self, w) {
            (Cfg::M1(c), Weights::M1(w)) => MambaCpuTrainer::new_with(w, *c, s, o)?.into(),
            (Cfg::M2(c), Weights::M2(w)) => Mamba2CpuTrainer::new_with(w, *c, s, o)?.into(),
            (Cfg::Siso(c), Weights::Siso(w)) => Mamba3CpuTrainer::new_with(w, *c, s, o)?.into(),
            (Cfg::Mimo(c), Weights::Mimo(w)) => Mamba3MimoCpuTrainer::new_with(w, *c, s, o)?.into(),
            _ => unreachable!("weights of another family"),
        })
    }

    fn gpu_trainer(&self, w: &Weights, opts: GpuOptions) -> Result<AnyGpuTrainer, String> {
        let s = TrainSessionCfg::new(INPUT_DIM, BATCH, WINDOW);
        Ok(match (self, w) {
            (Cfg::M1(c), Weights::M1(w)) => MambaGpuTrainer::new(w, *c, s, opts)?.into(),
            (Cfg::M2(c), Weights::M2(w)) => Mamba2GpuTrainer::new(w, *c, s, opts)?.into(),
            (Cfg::Siso(c), Weights::Siso(w)) => Mamba3GpuTrainer::new(w, *c, s, opts)?.into(),
            (Cfg::Mimo(c), Weights::Mimo(w)) => Mamba3MimoGpuTrainer::new(w, *c, s, opts)?.into(),
            _ => unreachable!("weights of another family"),
        })
    }
}

impl Host {
    /// The rows of one sequence's `steps` tokens, `[steps][input_dim]`, one
    /// after the other from the empty state.
    fn run(&self, inputs: &[f32], steps: usize, d_model: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; steps * d_model];
        macro_rules! walk {
            ($bb:expr) => {{
                let mut state = $bb.alloc_state();
                let mut scratch = $bb.alloc_scratch();
                for t in 0..steps {
                    $bb.forward_step(
                        &inputs[t * INPUT_DIM..(t + 1) * INPUT_DIM],
                        &mut out[t * d_model..(t + 1) * d_model],
                        &mut state,
                        &mut scratch,
                    );
                }
            }};
        }
        match self {
            Host::M1(bb) => walk!(bb),
            Host::M2(bb) => walk!(bb),
            Host::Siso(bb) => walk!(bb),
            Host::Mimo(bb) => walk!(bb),
        }
        out
    }

    /// One sequence's prompt of `prompt` tokens run in one pass, as the
    /// GPU runs it, then one step: the prompt's last row and the step's.
    fn prompt(&self, inputs: &[f32], prompt: usize, d_model: usize) -> [Vec<f32>; 2] {
        let mut rows = vec![0.0f32; prompt * d_model];
        let mut after = vec![0.0f32; d_model];
        let (head, next) = inputs.split_at(prompt * INPUT_DIM);
        macro_rules! window {
            ($bb:expr, |$state:ident, $scratch:ident| $run:block) => {{
                let mut $state = $bb.alloc_state();
                let mut $scratch = $bb.alloc_scratch();
                $run
                $bb.forward_step(&next[..INPUT_DIM], &mut after, &mut $state, &mut $scratch);
            }};
        }
        match self {
            Host::M1(bb) => window!(bb, |state, scratch| {
                let mut pf = bb.alloc_prefill_scratch(prompt);
                bb.forward_prefill(
                    head,
                    &mut rows,
                    &mut state,
                    &mut pf,
                    prompt,
                    PrefillMode::Single,
                );
            }),
            Host::M2(bb) => window!(bb, |state, scratch| {
                bb.forward_sequence(head, &mut rows, &mut state, &mut scratch, prompt);
            }),
            Host::Siso(bb) => window!(bb, |state, scratch| {
                let mut pf = bb.alloc_prefill_scratch(prompt);
                bb.forward_prefill(
                    head,
                    &mut rows,
                    &mut state,
                    &mut pf,
                    prompt,
                    PrefillMode::Single,
                );
            }),
            Host::Mimo(bb) => window!(bb, |state, scratch| {
                bb.forward_sequence(head, &mut rows, &mut state, &mut scratch, prompt);
            }),
        }
        [rows[(prompt - 1) * d_model..].to_vec(), after]
    }
}

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// The largest difference relative to the largest value of `want`; a NaN
/// on either side is infinite.
fn rel_diff(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter().zip(want).fold(0.0f32, |m, (g, w)| {
        let d = (g - w).abs() / scale;
        if d.is_nan() { f32::INFINITY } else { m.max(d) }
    })
}

/// Sequence `b` of rows `[batch, steps, width]`.
fn sequence(rows: &[f32], b: usize, steps: usize, width: usize) -> &[f32] {
    &rows[b * steps * width..(b + 1) * steps * width]
}

fn configs() -> Vec<Cfg> {
    vec![
        // One feature wide: every projection to or from d_model is a
        // one-column GEMM.
        Cfg::M1(MambaConfig {
            d_model: 1,
            d_state: 1,
            d_conv: 1,
            expand: 2,
            n_layers: 2,
            ..MambaConfig::default()
        }),
        Cfg::M2(Mamba2Config {
            d_model: 1,
            headdim: 1,
            d_state: 1,
            d_conv: 1,
            n_layers: 2,
            ..Mamba2Config::default()
        }),
        Cfg::Siso(Mamba3Config {
            d_model: 1,
            headdim: 1,
            d_state: 2,
            n_layers: 2,
            ..Mamba3Config::default()
        }),
        Cfg::Mimo(Mamba3MimoConfig {
            d_model: 1,
            headdim: 1,
            d_state: 2,
            mimo_rank: 1,
            rope_fraction: 1.0,
            n_layers: 2,
            ..Mamba3MimoConfig::default()
        }),
        Cfg::M1(MambaConfig {
            d_model: 16,
            d_state: 1,
            d_conv: 1,
            expand: 1,
            n_layers: 2,
            ..MambaConfig::default()
        }),
        // FalconMamba: the dt, B and C parts of every x_proj row normalized.
        Cfg::M1(MambaConfig {
            d_model: 40,
            d_state: 12,
            d_conv: 4,
            expand: 2,
            n_layers: 2,
            bcdt_rms_eps: Some(1e-6),
            ..MambaConfig::default()
        }),
        // Mamba-1 with in_proj and out_proj biases, without the conv bias,
        // with a dt rank of its own and a gated MLP after every mixer.
        Cfg::M1(MambaConfig {
            d_model: 24,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            n_layers: 2,
            dt_rank: Some(5),
            proj_bias: true,
            conv_bias: false,
            d_intermediate: 100,
            ..MambaConfig::default()
        }),
        Cfg::M1(MambaConfig {
            d_model: 33,
            d_state: 7,
            d_conv: 3,
            expand: 3,
            n_layers: 2,
            ..MambaConfig::default()
        }),
        Cfg::M1(MambaConfig {
            d_model: 32,
            d_state: 65,
            d_conv: 9,
            expand: 2,
            n_layers: 2,
            ..MambaConfig::default()
        }),
        Cfg::M1(MambaConfig {
            d_model: 24,
            d_state: 256,
            d_conv: 4,
            expand: 2,
            n_layers: 2,
            ..MambaConfig::default()
        }),
        Cfg::M2(Mamba2Config {
            d_model: 32,
            expand: 1,
            headdim: 1,
            d_state: 1,
            d_conv: 1,
            n_layers: 2,
            ..Mamba2Config::default()
        }),
        Cfg::M2(Mamba2Config {
            d_model: 48,
            headdim: 16,
            d_ssm: 64,
            ngroups: 2,
            d_state: 8,
            d_conv: 3,
            d_has_hdim: true,
            proj_bias: true,
            conv_bias: false,
            norm_before_gate: true,
            d_intermediate: 100,
            n_layers: 2,
            ..Mamba2Config::default()
        }),
        Cfg::M2(Mamba2Config {
            d_model: 32,
            headdim: 8,
            d_state: 128,
            ngroups: 8,
            d_conv: 9,
            rmsnorm: false,
            n_layers: 2,
            ..Mamba2Config::default()
        }),
        Cfg::M2(Mamba2Config {
            d_model: 64,
            headdim: 128,
            d_state: 300,
            dt_limit: (0.001, 0.1),
            n_layers: 2,
            ..Mamba2Config::default()
        }),
        Cfg::Siso(Mamba3Config {
            d_model: 16,
            headdim: 1,
            d_state: 2,
            ngroups: 2,
            rope_fraction: 1.0,
            n_layers: 2,
            ..Mamba3Config::default()
        }),
        Cfg::Siso(Mamba3Config {
            d_model: 33,
            headdim: 33,
            d_state: 16,
            n_layers: 2,
            ..Mamba3Config::default()
        }),
        Cfg::Siso(Mamba3Config {
            d_model: 64,
            headdim: 64,
            d_state: 128,
            rope_fraction: 1.0,
            is_outproj_norm: true,
            d_intermediate: 100,
            n_layers: 2,
            ..Mamba3Config::default()
        }),
        Cfg::Siso(Mamba3Config {
            d_model: 32,
            headdim: 16,
            d_state: 256,
            a_floor: 0.0,
            n_layers: 2,
            ..Mamba3Config::default()
        }),
        Cfg::Mimo(Mamba3MimoConfig {
            d_model: 16,
            headdim: 1,
            d_state: 2,
            mimo_rank: 1,
            rope_fraction: 1.0,
            n_layers: 2,
            ..Mamba3MimoConfig::default()
        }),
        Cfg::Mimo(Mamba3MimoConfig {
            d_model: 40,
            headdim: 20,
            d_state: 6,
            mimo_rank: 3,
            ngroups: 2,
            rope_fraction: 1.0,
            n_layers: 2,
            ..Mamba3MimoConfig::default()
        }),
        Cfg::Mimo(Mamba3MimoConfig {
            d_model: 64,
            headdim: 64,
            d_state: 128,
            mimo_rank: 4,
            is_outproj_norm: true,
            d_intermediate: 100,
            n_layers: 2,
            ..Mamba3MimoConfig::default()
        }),
        Cfg::Mimo(Mamba3MimoConfig {
            d_model: 32,
            headdim: 16,
            d_state: 256,
            mimo_rank: 16,
            a_floor: 0.0,
            n_layers: 2,
            ..Mamba3MimoConfig::default()
        }),
    ]
}

fn request(path: Path, (scan, precision, gemm): (ScanMode, WeightDtype, GemmMode)) -> Request {
    Request {
        path,
        batch: BATCH,
        seq_len: if path == Path::Decode { 1 } else { WINDOW },
        scan,
        precision,
        gemm,
    }
}

fn options((scan, precision, gemm): (ScanMode, WeightDtype, GemmMode)) -> GpuOptions {
    GpuOptions::default()
        .dtype(precision)
        .mode(gemm)
        .scan_mode(scan)
}

/// Decode steps of every sequence, the host's rows per step.
fn check_decode(
    cfg: &Cfg,
    w: &Weights,
    (host, lane): (&Host, Option<&Host>),
    cell: (ScanMode, WeightDtype, GemmMode),
) -> Check {
    let d = cfg.d_model();
    let inputs = values(BATCH * DECODE_STEPS * INPUT_DIM, 7);
    // A host's rows of every step, the batch's rows of a step together
    // as the device gives them.
    let rows_of = |host: &Host, inputs: &[f32]| -> Vec<f32> {
        let runs: Vec<Vec<f32>> = (0..BATCH)
            .map(|b| {
                host.run(
                    sequence(inputs, b, DECODE_STEPS, INPUT_DIM),
                    DECODE_STEPS,
                    d,
                )
            })
            .collect();
        (0..DECODE_STEPS)
            .flat_map(|t| {
                runs.iter()
                    .flat_map(move |r| r[t * d..(t + 1) * d].to_vec())
            })
            .collect()
    };
    let host_rows = |inputs: &[f32]| rows_of(host, inputs);
    let step_of =
        |rows: &[f32], t: usize| -> Vec<f32> { rows[t * BATCH * d..(t + 1) * BATCH * d].to_vec() };
    let want = host_rows(&inputs);
    let lane_rows = lane.map(|l| rows_of(l, &inputs));
    let mut gpu = cfg
        .gpu(w, options(cell))
        .unwrap_or_else(|e| panic!("{cfg:?} {cell:?}: {e}"));
    let (mut diff, mut lane_diff) = (0.0f32, 0.0f32);
    for t in 0..DECODE_STEPS {
        let step: Vec<f32> = (0..BATCH)
            .flat_map(|b| {
                let s = sequence(&inputs, b, DECODE_STEPS, INPUT_DIM);
                s[t * INPUT_DIM..(t + 1) * INPUT_DIM].to_vec()
            })
            .collect();
        let mut got = vec![0.0f32; BATCH * d];
        gpu.step(&step, &mut got)
            .unwrap_or_else(|e| panic!("{cfg:?} {cell:?}: decode step {t}: {e}"));
        diff = diff.max(rel_diff(&got, &step_of(&want, t)));
        if let Some(rows) = &lane_rows {
            lane_diff = lane_diff.max(rel_diff(&got, &step_of(rows, t)));
        }
    }
    let spread = host_spread(cell.1, |u, seed| {
        let moved = host_rows(&nudged(&inputs, u, seed));
        (0..DECODE_STEPS)
            .map(|t| rel_diff(&step_of(&moved, t), &step_of(&want, t)))
            .fold(0.0f32, f32::max)
    });
    Check {
        diff,
        spread,
        lane: lane_rows.map(|_| lane_diff),
    }
}

/// A prompt of `WINDOW` tokens and the step after it, the host's rows for
/// the prompt's last token and for that step.
fn check_prompt(
    cfg: &Cfg,
    w: &Weights,
    (host, lane): (&Host, Option<&Host>),
    cell: (ScanMode, WeightDtype, GemmMode),
) -> Check {
    let d = cfg.d_model();
    let steps = WINDOW + 1;
    let inputs = values(BATCH * steps * INPUT_DIM, 11);
    // The host's rows for the prompt's last token and for the step after.
    let host_rows = |inputs: &[f32]| -> [Vec<f32>; 2] {
        let runs: Vec<Vec<f32>> = (0..BATCH)
            .map(|b| host.run(sequence(inputs, b, steps, INPUT_DIM), steps, d))
            .collect();
        [WINDOW - 1, WINDOW].map(|t| {
            runs.iter()
                .flat_map(|r| r[t * d..(t + 1) * d].to_vec())
                .collect()
        })
    };
    let want = host_rows(&inputs);
    let prompt: Vec<f32> = (0..BATCH)
        .flat_map(|b| sequence(&inputs, b, steps, INPUT_DIM)[..WINDOW * INPUT_DIM].to_vec())
        .collect();
    let next: Vec<f32> = (0..BATCH)
        .flat_map(|b| sequence(&inputs, b, steps, INPUT_DIM)[WINDOW * INPUT_DIM..].to_vec())
        .collect();
    let mut gpu = cfg
        .gpu(w, options(cell))
        .unwrap_or_else(|e| panic!("{cfg:?} {cell:?}: {e}"));
    gpu.prefill(&prompt, WINDOW)
        .unwrap_or_else(|e| panic!("{cfg:?} {cell:?}: prompt: {e}"));
    let mut last = vec![0.0f32; BATCH * d];
    gpu.download_temporal(&mut last).unwrap();
    let mut after = vec![0.0f32; BATCH * d];
    gpu.step(&next, &mut after)
        .unwrap_or_else(|e| panic!("{cfg:?} {cell:?}: the step after the prompt: {e}"));
    let diff = rel_diff(&last, &want[0]).max(rel_diff(&after, &want[1]));
    let spread = host_spread(cell.1, |u, seed| {
        let moved = host_rows(&nudged(&inputs, u, seed));
        rel_diff(&moved[0], &want[0]).max(rel_diff(&moved[1], &want[1]))
    });
    // The lane runs the prompt in one pass, as the device does.
    let lane = lane.map(|l| {
        let runs: Vec<[Vec<f32>; 2]> = (0..BATCH)
            .map(|b| l.prompt(sequence(&inputs, b, steps, INPUT_DIM), WINDOW, d))
            .collect();
        let row = |k: usize| -> Vec<f32> { runs.iter().flat_map(|r| r[k].clone()).collect() };
        rel_diff(&last, &row(0)).max(rel_diff(&after, &row(1)))
    });
    Check { diff, spread, lane }
}

/// One training window on both devices: the GPU's global gradient norm
/// against the host's, the window applied on both.
fn check_train(cfg: &Cfg, w: &Weights, cell: (ScanMode, WeightDtype, GemmMode)) -> Check {
    let measure = BackwardOpts::default().with_clip_max_norm(1e30);
    let input = values(BATCH * WINDOW * INPUT_DIM, 13);
    let d_temporal = values(BATCH * WINDOW * cfg.d_model(), 17);
    // The host's global gradient norm of the window, from the weights.
    let host_norm = |input: &[f32]| -> f32 {
        let mut cpu = cfg
            .host_trainer(w, WeightDtype::F32)
            .unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
        cpu.set_step_opts(measure).unwrap();
        let host = cpu.step(input, &d_temporal).unwrap();
        assert!(
            host.optimizer_stepped,
            "{cfg:?}: the host skipped its window"
        );
        host.grad_norm.unwrap()
    };
    let h = host_norm(&input);
    // The host trainer computing in the cell's precision, its f16 windows
    // through the loss scaler as on the device.
    let lane_norm = has_lane(cell.1).then(|| {
        let mut cpu = cfg
            .host_trainer(w, cell.1)
            .unwrap_or_else(|e| panic!("{cfg:?} {cell:?}: {e}"));
        cpu.set_step_opts(measure).unwrap();
        if cell.1 == WeightDtype::F16 {
            cpu.load_scaler_state(1024.0, 0).unwrap();
        }
        let mut m = cpu.step(&input, &d_temporal).unwrap();
        for _ in 0..16 {
            if m.overflow_skipped != Some(true) {
                break;
            }
            cpu.reset_state().unwrap();
            m = cpu.step(&input, &d_temporal).unwrap();
        }
        assert!(
            m.optimizer_stepped,
            "{cfg:?} {cell:?}: the host lane skipped"
        );
        m.grad_norm.unwrap()
    });
    let rel = |norm: f32| -> f32 {
        let d = ((norm - h) / h.max(1e-12)).abs();
        if d.is_nan() { f32::INFINITY } else { d }
    };
    let mut gpu = cfg
        .gpu_trainer(w, options(cell))
        .unwrap_or_else(|e| panic!("{cfg:?} {cell:?}: {e}"));
    gpu.set_step_opts(measure).unwrap();
    if cell.1 == WeightDtype::F16 {
        // Small models overflow f16 at the scaler's default start.
        gpu.load_scaler_state(1024.0, 0).unwrap();
    }
    // An f16 overflow skips the window and halves the scale, as torch's
    // GradScaler does, and leaves the weights where they were. The trainer
    // carries the recurrence across windows, so the state goes back to
    // zero and the same window runs again until the scale lets it through.
    // RMSNorm over one feature scales its gradient by up to 1 / sqrt(eps)
    // near zero, past f16 at the starting scale.
    let mut device = gpu
        .step(&input, &d_temporal)
        .unwrap_or_else(|e| panic!("{cfg:?} {cell:?}: train step: {e}"));
    for _ in 0..16 {
        if device.overflow_skipped != Some(true) {
            break;
        }
        gpu.reset_state().unwrap();
        device = gpu
            .step(&input, &d_temporal)
            .unwrap_or_else(|e| panic!("{cfg:?} {cell:?}: train step: {e}"));
    }
    assert!(
        device.optimizer_stepped,
        "{cfg:?} {cell:?}: the GPU skipped its window ({:?})",
        device.loss_scale
    );
    let spread = host_spread(cell.1, |u, seed| rel(host_norm(&nudged(&input, u, seed))));
    let g = device.grad_norm.unwrap();
    Check {
        diff: rel(g),
        spread,
        lane: lane_norm.map(|l| {
            let d = ((g - l) / l.max(1e-12)).abs();
            if d.is_nan() { f32::INFINITY } else { d }
        }),
    }
}

/// The host's largest distance from itself over the nudged runs, each
/// run's inputs moved by one rounding of `precision`; nothing for f32.
fn host_spread(precision: WeightDtype, run: impl Fn(f32, u32) -> f32) -> f32 {
    let u = unit_roundoff(precision);
    if u == 0.0 {
        return 0.0;
    }
    NUDGES
        .iter()
        .map(|&seed| run(u, seed))
        .fold(0.0f32, f32::max)
}

#[test]
fn every_cell_runs_and_agrees_with_the_host() {
    let mut failures = Vec::new();
    let mut held_to_spread = Vec::new();
    let mut lanes = Vec::new();
    let mut cells_run = 0;
    for (i, cfg) in configs().into_iter().enumerate() {
        cfg.validate().unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
        let w = cfg.weights(100 + i as u64);
        let host = cfg.host(&w, WeightDtype::F32);
        let lane_hosts: Vec<(WeightDtype, Host)> = PRECISIONS
            .into_iter()
            .filter(|&p| has_lane(p))
            .map(|p| (p, cfg.host(&w, p)))
            .collect();
        for path in Path::ALL {
            for cell in cells(path) {
                assert!(cfg.admit(&request(path, cell)), "{cfg:?} {path:?} {cell:?}");
                let lane = lane_hosts
                    .iter()
                    .find(|(p, _)| *p == cell.1)
                    .map(|(_, h)| h);
                let check = match path {
                    Path::Decode => check_decode(&cfg, &w, (&host, lane), cell),
                    Path::Prefill => check_prompt(&cfg, &w, (&host, lane), cell),
                    Path::Train => check_train(&cfg, &w, cell),
                };
                if let Some(l) = check.lane {
                    let line = format!(
                        "{cfg:?} {} {cell:?}: f32 host {:.3e}, lane {l:.3e}",
                        path.as_str(),
                        check.diff
                    );
                    let lane_tol = LANE_ROUNDINGS * unit_roundoff(cell.1);
                    if l > lane_tol {
                        failures.push(format!("{line} > {lane_tol:.1e} from the lane"));
                    }
                    lanes.push(line);
                }
                cells_run += 1;
                let base = tolerance(cell.1, cell.2);
                let tol = base.max(SPREAD_FACTOR * check.spread);
                let line = format!(
                    "{cfg:?} {} {cell:?}: {:.3e}, host spread {:.3e}",
                    path.as_str(),
                    check.diff,
                    check.spread
                );
                if tol > base {
                    held_to_spread.push(line.clone());
                }
                if check.diff > tol {
                    failures.push(format!("{line} > {tol:.0e}"));
                }
            }
        }
    }
    eprintln!("{cells_run} cells run");
    eprintln!(
        "{} half cells against the host lane:\n{}",
        lanes.len(),
        lanes.join("\n")
    );
    if !held_to_spread.is_empty() {
        eprintln!(
            "{} cells held to the host's own spread:\n{}",
            held_to_spread.len(),
            held_to_spread.join("\n")
        );
    }
    assert!(
        failures.is_empty(),
        "{} cells disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
