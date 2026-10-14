//! Several sequences packed into the rows of one window (varlen) train on
//! the GPU as they train alone, on both scans and in every precision:
//! changing one sequence's inputs and loss moves no bit of another's rows
//! or input gradient, a loss on one sequence leaves every other's input
//! gradient exactly zero, and an overflow in one sequence stays in it; the
//! packed rows and input gradients equal each sequence's own run from the
//! empty state bit for bit (a sequence alone keeps its window's rows, so
//! the GEMMs see the same shapes); a packing without a start is no packing;
//! the sequence before a row's first start continues the carried state and
//! the state after the window is the last sequence's; the host trainer
//! trains a packing as the GPU does; a captured step replays a packing as
//! the eager step runs it and refuses to drop or take one it was not
//! captured with. Mamba-1, which the reference does not pack, refuses a
//! packing.
//!
//! Run on a GPU box:
//!   cargo test --release --features cuda --test varlen_gpu -- --test-threads=1
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::gpu::GemmMode;
use mamba_rs::{
    AnyCpuTrainer, AnyGpuTrainer, BackwardOpts, CpuOptions, CpuTrainer, GpuOptions, GpuTrainer,
    Mamba2Config, Mamba2CpuTrainer, Mamba2GpuTrainer, Mamba2Weights, Mamba3Config,
    Mamba3CpuTrainer, Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoCpuTrainer,
    Mamba3MimoGpuTrainer, Mamba3MimoWeights, Mamba3Weights, MambaConfig, MambaGpuTrainer,
    MambaWeights, TrainSessionCfg, Varlen, WeightDtype,
};

const D_MODEL: usize = 32;
const INPUT_DIM: usize = 24;
/// A short row: a sequence of 5 tokens, then one of 7, inside one chunk.
const SEQ_LEN: usize = 12;
const CUT: usize = 5;
/// A long row of three sequences, the middle one crossing a chunk
/// boundary and none starting on one (every family's chunk is 64 tokens).
const LONG: usize = 160;
const LONG_STARTS: [usize; 2] = [40, 100];
const DTYPES: [WeightDtype; 4] = [
    WeightDtype::F32,
    WeightDtype::Tf32,
    WeightDtype::Bf16,
    WeightDtype::F16,
];
const SCANS: [ScanMode; 2] = [ScanMode::Sequential, ScanMode::Parallel];

fn m2_cfg() -> Mamba2Config {
    Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        d_intermediate: 40,
        ..Mamba2Config::default()
    }
}

fn siso_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        d_intermediate: 40,
        ..Mamba3Config::default()
    }
}

fn mimo_cfg() -> Mamba3MimoConfig {
    Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        d_intermediate: 40,
        ..Mamba3MimoConfig::default()
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

/// An output gradient of a loss's size, about 1e-2.
fn grad(n: usize, seed: u32) -> Vec<f32> {
    let mut d = values(n, seed);
    d.iter_mut().for_each(|v| *v *= 1.0 / 64.0);
    d
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn gpu_options(dtype: WeightDtype, scan: ScanMode) -> GpuOptions {
    GpuOptions::default()
        .dtype(dtype)
        .mode(GemmMode::Deterministic)
        .scan_mode(scan)
}

/// The packing families, in the order of [`gpu_trainer`]'s index.
const FAMILIES: [&str; 3] = ["Mamba-2", "Mamba-3 SISO", "Mamba-3 MIMO"];

/// The GPU trainer of family `i` over `batch` rows of `seq_len` positions,
/// computing in `dtype` on `scan`. An f16 trainer starts its loss scale at
/// 1024: at the default 65536 these small models overflow and a comparison
/// would read NaNs.
fn gpu_trainer(
    i: usize,
    (batch, seq_len): (usize, usize),
    dtype: WeightDtype,
    scan: ScanMode,
) -> AnyGpuTrainer {
    let s = TrainSessionCfg::new(INPUT_DIM, batch, seq_len);
    let o = gpu_options(dtype, scan);
    let mut t: AnyGpuTrainer = match i {
        0 => {
            let w = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 2);
            Mamba2GpuTrainer::new(&w, m2_cfg(), s, o).unwrap().into()
        }
        1 => {
            let w = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 3);
            Mamba3GpuTrainer::new(&w, siso_cfg(), s, o).unwrap().into()
        }
        _ => {
            let w = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 4);
            Mamba3MimoGpuTrainer::new(&w, mimo_cfg(), s, o)
                .unwrap()
                .into()
        }
    };
    t.load_scaler_state(1024.0, 0).unwrap();
    t
}

/// The host trainer of family `i`, f32, over the same weights.
fn cpu_trainer(i: usize, (batch, seq_len): (usize, usize)) -> AnyCpuTrainer {
    let s = TrainSessionCfg::new(INPUT_DIM, batch, seq_len);
    let o = CpuOptions::default().dtype(WeightDtype::F32);
    match i {
        0 => {
            let w = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 2);
            Mamba2CpuTrainer::new_with(&w, m2_cfg(), s, o)
                .unwrap()
                .into()
        }
        1 => {
            let w = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 3);
            Mamba3CpuTrainer::new_with(&w, siso_cfg(), s, o)
                .unwrap()
                .into()
        }
        _ => {
            let w = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 4);
            Mamba3MimoCpuTrainer::new_with(&w, mimo_cfg(), s, o)
                .unwrap()
                .into()
        }
    }
}

/// The forward of `input` and the accumulating backward of `d`: the rows,
/// the input gradient and the weight gradient.
fn run_raw(t: &mut AnyGpuTrainer, input: &[f32], d: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rows = vec![0.0f32; d.len()];
    let mut d_input = vec![0.0f32; input.len()];
    t.forward(input, &mut rows).unwrap();
    t.backward_step_input_grad(
        d,
        BackwardOpts::default().with_accumulate_only(true),
        &mut d_input,
    )
    .unwrap();
    (rows, d_input, t.download_grad_arena().unwrap())
}

/// [`run_raw`] with every value finite: a window that overflowed would
/// compare its NaNs.
fn run(t: &mut AnyGpuTrainer, input: &[f32], d: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let out = run_raw(t, input, d);
    for (what, v) in [
        ("rows", &out.0),
        ("input gradient", &out.1),
        ("weight gradient", &out.2),
    ] {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "the window's {what} overflowed"
        );
    }
    out
}

/// Every sequence of one row of `seq_len` positions starting at `starts`:
/// its first and end positions.
fn spans(starts: &[usize], seq_len: usize) -> Vec<(usize, usize)> {
    let mut cuts: Vec<usize> = std::iter::once(0)
        .chain(starts.iter().copied().filter(|&t| t > 0))
        .collect();
    cuts.push(seq_len);
    cuts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Changing one packed sequence's inputs and loss moves no bit of another
/// sequence's rows or input gradient, on both scans and in every
/// precision, the middle sequence crossing a chunk boundary.
#[test]
fn a_sequence_never_sees_another() {
    let packing = Varlen::from_starts(&[LONG_STARTS.to_vec()], LONG).unwrap();
    let (a, e) = (LONG_STARTS[0], LONG_STARTS[1]);
    let input = values(LONG * INPUT_DIM, 31);
    let d = grad(LONG * D_MODEL, 32);
    let mut input2 = input.clone();
    input2[a * INPUT_DIM..e * INPUT_DIM].copy_from_slice(&values((e - a) * INPUT_DIM, 33));
    let mut d2 = d.clone();
    d2[a * D_MODEL..e * D_MODEL].copy_from_slice(&grad((e - a) * D_MODEL, 34));
    for scan in SCANS {
        for dtype in DTYPES {
            for (i, family) in FAMILIES.iter().enumerate() {
                let mut t = gpu_trainer(i, (1, LONG), dtype, scan);
                t.set_varlen(Some(&packing)).unwrap();
                let (rows, d_input, _) = run(&mut t, &input, &d);
                t.reset_state().unwrap();
                let (rows2, d_input2, _) = run(&mut t, &input2, &d2);
                for (lo, hi) in [(0, a), (e, LONG)] {
                    assert_eq!(
                        bits(&rows[lo * D_MODEL..hi * D_MODEL]),
                        bits(&rows2[lo * D_MODEL..hi * D_MODEL]),
                        "{family} {dtype:?} {scan:?}: the rows of the sequence at {lo}..{hi}"
                    );
                    assert_eq!(
                        bits(&d_input[lo * INPUT_DIM..hi * INPUT_DIM]),
                        bits(&d_input2[lo * INPUT_DIM..hi * INPUT_DIM]),
                        "{family} {dtype:?} {scan:?}: the input gradient of the sequence at \
                         {lo}..{hi}"
                    );
                }
                assert_ne!(
                    bits(&rows[a * D_MODEL..e * D_MODEL]),
                    bits(&rows2[a * D_MODEL..e * D_MODEL]),
                    "{family} {dtype:?} {scan:?}: the changed sequence did not change"
                );
            }
        }
    }
}

#[test]
fn a_loss_on_one_sequence_leaves_the_others_untouched() {
    let packing = Varlen::from_starts(&[LONG_STARTS.to_vec()], LONG).unwrap();
    let last = LONG_STARTS[1];
    let input = values(LONG * INPUT_DIM, 3);
    // The loss on the last sequence only: without the packing its gradient
    // would reach the earlier sequences' inputs through the state.
    let mut d = grad(LONG * D_MODEL, 4);
    d[..last * D_MODEL].fill(0.0);
    for scan in SCANS {
        for dtype in DTYPES {
            for (i, family) in FAMILIES.iter().enumerate() {
                let mut t = gpu_trainer(i, (1, LONG), dtype, scan);
                t.set_varlen(Some(&packing)).unwrap();
                let (_, d_input, _) = run(&mut t, &input, &d);
                assert!(
                    d_input[..last * INPUT_DIM].iter().all(|&v| v == 0.0),
                    "{family} {dtype:?} {scan:?}: the loss on the last sequence reached another"
                );
                assert!(
                    d_input[last * INPUT_DIM..].iter().any(|&v| v != 0.0),
                    "{family} {dtype:?} {scan:?}: no gradient on the last sequence"
                );
                // Without the packing the earlier sequences' inputs do get one.
                t.set_varlen(None).unwrap();
                t.reset_state().unwrap();
                let (_, d_plain, _) = run(&mut t, &input, &d);
                assert!(
                    d_plain[..last * INPUT_DIM].iter().any(|&v| v != 0.0),
                    "{family} {dtype:?} {scan:?}: the unpacked control carries no gradient back"
                );
            }
        }
    }
}

/// An f16 window whose last sequence's loss overflows leaves the earlier
/// sequences' input gradients exactly zero, not NaN: the cut at a start
/// drops the overflowed carries instead of scaling them by zero.
#[test]
fn an_overflowing_loss_stays_in_its_sequence() {
    let packing = Varlen::from_starts(&[LONG_STARTS.to_vec()], LONG).unwrap();
    let last = LONG_STARTS[1];
    let input = values(LONG * INPUT_DIM, 8);
    let mut d = grad(LONG * D_MODEL, 9);
    d[..last * D_MODEL].fill(0.0);
    d[last * D_MODEL] = f32::MAX;
    for scan in SCANS {
        for (i, family) in FAMILIES.iter().enumerate() {
            let mut t = gpu_trainer(i, (1, LONG), WeightDtype::F16, scan);
            t.set_varlen(Some(&packing)).unwrap();
            let (_, d_input, _) = run_raw(&mut t, &input, &d);
            assert!(
                d_input[last * INPUT_DIM..].iter().any(|v| !v.is_finite()),
                "{family} {scan:?}: the last sequence did not overflow, so the test proves nothing"
            );
            assert!(
                d_input[..last * INPUT_DIM].iter().all(|&v| v == 0.0),
                "{family} {scan:?}: the overflow of the last sequence reached another"
            );
        }
    }
}

/// A sequence whose inputs overflow its activations leaves every later
/// sequence's rows and input gradient as they are with tame inputs: the
/// chunk's decay sums restart at each sequence, so an infinite step there
/// never meets another sequence's in a subtraction.
#[test]
fn an_overflowing_input_stays_in_its_sequence() {
    let packing = Varlen::from_starts(&[LONG_STARTS.to_vec()], LONG).unwrap();
    let (a, e) = (LONG_STARTS[0], LONG_STARTS[1]);
    let input = values(LONG * INPUT_DIM, 41);
    let mut wild = input.clone();
    wild[a * INPUT_DIM..e * INPUT_DIM].fill(f32::MAX);
    // The loss on the last sequence only, so the wild one's own gradient
    // stays out of the arena.
    let mut d = grad(LONG * D_MODEL, 42);
    d[..e * D_MODEL].fill(0.0);
    for scan in SCANS {
        for dtype in DTYPES {
            for (i, family) in FAMILIES.iter().enumerate() {
                let mut t = gpu_trainer(i, (1, LONG), dtype, scan);
                t.set_varlen(Some(&packing)).unwrap();
                let (rows, d_input, _) = run_raw(&mut t, &input, &d);
                t.reset_state().unwrap();
                let (rows_w, d_input_w, _) = run_raw(&mut t, &wild, &d);
                assert!(
                    rows_w[a * D_MODEL..e * D_MODEL]
                        .iter()
                        .any(|v| !v.is_finite()),
                    "{family} {dtype:?} {scan:?}: the wild sequence did not overflow, so the \
                     test proves nothing"
                );
                assert_eq!(
                    bits(&rows[e * D_MODEL..]),
                    bits(&rows_w[e * D_MODEL..]),
                    "{family} {dtype:?} {scan:?}: the wild sequence moved the next one's rows"
                );
                assert_eq!(
                    bits(&d_input[e * INPUT_DIM..]),
                    bits(&d_input_w[e * INPUT_DIM..]),
                    "{family} {dtype:?} {scan:?}: the wild sequence moved the next one's input \
                     gradient"
                );
                assert!(
                    rows[e * D_MODEL..].iter().all(|v| v.is_finite()),
                    "{family} {dtype:?} {scan:?}: the tame run overflowed"
                );
            }
        }
    }
}

/// Each sequence packed into a row runs as it does alone from the empty
/// state: its rows and input gradient bit for bit. The run alone keeps the
/// packed window's shape (the sequence first, a filler sequence after it),
/// so the GEMMs see the same rows; the chunked scan segments a sequence the
/// same way when it starts on a chunk or fits in one, as every sequence
/// here does on that scan, and the sequential scan takes any start.
#[test]
fn packed_sequences_run_as_they_do_alone() {
    let cases: [(&[usize], usize, &[ScanMode]); 4] = [
        (&[0, CUT], SEQ_LEN, &SCANS),
        (&[3, 9], SEQ_LEN, &SCANS),
        (&[64, 128], 192, &SCANS),
        (&LONG_STARTS, LONG, &[ScanMode::Sequential]),
    ];
    for (n, &(starts, seq_len, scans)) in cases.iter().enumerate() {
        let seed = 100 * n as u32;
        let packing = Varlen::from_starts(&[starts.to_vec()], seq_len).unwrap();
        let input = values(seq_len * INPUT_DIM, seed + 1);
        let d = grad(seq_len * D_MODEL, seed + 2);
        let filler_in = values(seq_len * INPUT_DIM, seed + 3);
        let filler_d = grad(seq_len * D_MODEL, seed + 4);
        for &scan in scans {
            for dtype in DTYPES {
                for (i, family) in FAMILIES.iter().enumerate() {
                    let mut t = gpu_trainer(i, (1, seq_len), dtype, scan);
                    t.set_varlen(Some(&packing)).unwrap();
                    let (rows, d_input, _) = run(&mut t, &input, &d);
                    for (a, e) in spans(starts, seq_len) {
                        let len = e - a;
                        // The sequence alone at the window's start, the filler
                        // after it.
                        let mut in_alone = filler_in.clone();
                        in_alone[..len * INPUT_DIM]
                            .copy_from_slice(&input[a * INPUT_DIM..e * INPUT_DIM]);
                        let mut d_alone = filler_d.clone();
                        d_alone[..len * D_MODEL].copy_from_slice(&d[a * D_MODEL..e * D_MODEL]);
                        let alone_starts: Vec<usize> =
                            if len < seq_len { vec![0, len] } else { vec![0] };
                        let alone_packing = Varlen::from_starts(&[alone_starts], seq_len).unwrap();
                        let mut t = gpu_trainer(i, (1, seq_len), dtype, scan);
                        t.set_varlen(Some(&alone_packing)).unwrap();
                        let (rows_s, d_input_s, _) = run(&mut t, &in_alone, &d_alone);
                        assert_eq!(
                            bits(&rows[a * D_MODEL..e * D_MODEL]),
                            bits(&rows_s[..len * D_MODEL]),
                            "{family} {dtype:?} {scan:?}: the rows of the sequence at {a}..{e}"
                        );
                        assert_eq!(
                            bits(&d_input[a * INPUT_DIM..e * INPUT_DIM]),
                            bits(&d_input_s[..len * INPUT_DIM]),
                            "{family} {dtype:?} {scan:?}: the input gradient of the sequence at \
                         {a}..{e}"
                        );
                    }
                }
            }
        }
    }
}

/// A packing whose every id is 0 (the carried sequence through the whole
/// window) is no packing at all: the same rows, input gradient and weight
/// gradient bit for bit.
#[test]
fn a_packing_without_a_start_is_no_packing() {
    let none = Varlen::from_seq_idx(vec![0; LONG], 1, LONG).unwrap();
    let input = values(LONG * INPUT_DIM, 51);
    let d = grad(LONG * D_MODEL, 52);
    for scan in SCANS {
        for dtype in DTYPES {
            for (i, family) in FAMILIES.iter().enumerate() {
                let mut plain = gpu_trainer(i, (1, LONG), dtype, scan);
                let mut packed = gpu_trainer(i, (1, LONG), dtype, scan);
                packed.set_varlen(Some(&none)).unwrap();
                let a = run(&mut plain, &input, &d);
                let b = run(&mut packed, &input, &d);
                for (what, x, y) in [
                    ("rows", &a.0, &b.0),
                    ("input gradient", &a.1, &b.1),
                    ("weight gradient", &a.2, &b.2),
                ] {
                    assert_eq!(bits(x), bits(y), "{family} {dtype:?} {scan:?}: the {what}");
                }
            }
        }
    }
}

/// The positions before a row's first start continue the state carried
/// from the window before, and those after it start from the empty state;
/// the state after the window is the last sequence's.
#[test]
fn the_carried_state_reaches_only_the_first_sequence() {
    let warm = values(SEQ_LEN * INPUT_DIM, 5);
    let input = values(SEQ_LEN * INPUT_DIM, 6);
    let next = values(SEQ_LEN * INPUT_DIM, 7);
    let packing = Varlen::from_starts(&[vec![CUT]], SEQ_LEN).unwrap();
    let tail = SEQ_LEN - CUT;
    // The second sequence alone keeps the window's shape: its tokens first,
    // a filler sequence after them.
    let mut in_b = values(SEQ_LEN * INPUT_DIM, 8);
    in_b[..tail * INPUT_DIM].copy_from_slice(&input[CUT * INPUT_DIM..]);
    let alone = Varlen::from_starts(&[vec![0, tail]], SEQ_LEN).unwrap();
    let rows_n = SEQ_LEN * D_MODEL;
    let shape = (1, SEQ_LEN);
    for scan in SCANS {
        for dtype in DTYPES {
            for (i, family) in FAMILIES.iter().enumerate() {
                let mut packed = gpu_trainer(i, shape, dtype, scan);
                let mut plain = gpu_trainer(i, shape, dtype, scan);
                let mut fresh_b = gpu_trainer(i, shape, dtype, scan);
                let mut sink = vec![0.0f32; rows_n];
                packed.forward(&warm, &mut sink).unwrap();
                plain.forward(&warm, &mut sink).unwrap();
                packed.set_varlen(Some(&packing)).unwrap();
                fresh_b.set_varlen(Some(&alone)).unwrap();
                let (mut rows, mut rows_plain, mut rows_b) = (
                    vec![0.0f32; rows_n],
                    vec![0.0f32; rows_n],
                    vec![0.0f32; rows_n],
                );
                packed.forward(&input, &mut rows).unwrap();
                plain.forward(&input, &mut rows_plain).unwrap();
                fresh_b.forward(&in_b, &mut rows_b).unwrap();
                assert_eq!(
                    bits(&rows[..CUT * D_MODEL]),
                    bits(&rows_plain[..CUT * D_MODEL]),
                    "{family} {dtype:?} {scan:?}: the carried sequence"
                );
                assert_eq!(
                    bits(&rows[CUT * D_MODEL..]),
                    bits(&rows_b[..tail * D_MODEL]),
                    "{family} {dtype:?} {scan:?}: the sequence that starts"
                );
                // The state after the window is the last sequence's: the
                // packed trainer's next window runs as one of a trainer that
                // loads the state the second sequence leaves alone, from a
                // window of its own length.
                let mut own = gpu_trainer(i, (1, tail), dtype, scan);
                let mut own_rows = vec![0.0f32; tail * D_MODEL];
                own.forward(&input[CUT * INPUT_DIM..], &mut own_rows)
                    .unwrap();
                let mut cont = gpu_trainer(i, shape, dtype, scan);
                cont.load_trainer_state(&own.trainer_state().unwrap())
                    .unwrap();
                packed.set_varlen(None).unwrap();
                let (mut a, mut b) = (vec![0.0f32; rows_n], vec![0.0f32; rows_n]);
                packed.forward(&next, &mut a).unwrap();
                cont.forward(&next, &mut b).unwrap();
                assert_eq!(
                    bits(&a),
                    bits(&b),
                    "{family} {dtype:?} {scan:?}: the state after the window"
                );
            }
        }
    }
}

/// The host trainer and the GPU trainer agree on a packed window, f32, on
/// both GPU scans: the rows and the input gradient to the f32 reorder of
/// their reductions, the weight gradient's norm likewise.
#[test]
fn the_host_trains_a_packing_as_the_gpu_does() {
    let packing = Varlen::from_starts(&[LONG_STARTS.to_vec()], LONG).unwrap();
    let input = values(LONG * INPUT_DIM, 61);
    let d = grad(LONG * D_MODEL, 62);
    let rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f32 = x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = x.iter().map(|a| a * a).sum();
        (num / den.max(1e-30)).sqrt()
    };
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for scan in SCANS {
        for (i, family) in FAMILIES.iter().enumerate() {
            let mut c = cpu_trainer(i, (1, LONG));
            let mut g = gpu_trainer(i, (1, LONG), WeightDtype::F32, scan);
            c.set_varlen(Some(&packing)).unwrap();
            g.set_varlen(Some(&packing)).unwrap();
            let mut rows_c = vec![0.0f32; LONG * D_MODEL];
            let mut d_input_c = vec![0.0f32; LONG * INPUT_DIM];
            c.forward(&input, &mut rows_c).unwrap();
            c.backward_step_input_grad(
                &d,
                BackwardOpts::default().with_accumulate_only(true),
                &mut d_input_c,
            )
            .unwrap();
            let grads_c = c.download_grad_arena().unwrap();
            let (rows_g, d_input_g, grads_g) = run(&mut g, &input, &d);
            for (what, err) in [
                ("rows", rel(&rows_c, &rows_g)),
                ("input gradient", rel(&d_input_c, &d_input_g)),
                (
                    "weight gradient norm",
                    (norm(&grads_c) - norm(&grads_g)).abs() / norm(&grads_c),
                ),
            ] {
                assert!(
                    err < 1e-3,
                    "{family} {scan:?}: the host and the GPU part on the {what} by {err:e}"
                );
            }
        }
    }
}

fn moments(t: &AnyGpuTrainer) -> (Vec<u32>, Vec<u32>) {
    let blob = t.optimizer_state().unwrap();
    (bits(&blob.m), bits(&blob.v))
}

/// A captured step replays a packing as the eager step runs it, a new
/// packing of the same shape included, and refuses to drop it; a step
/// captured without a packing refuses one. A capture takes the GEMMs of
/// one eager step before it.
#[test]
fn a_captured_step_replays_a_packing() {
    let first = Varlen::from_starts(&[LONG_STARTS.to_vec()], LONG).unwrap();
    let second = Varlen::from_starts(&[vec![0, 64, 65, 150]], LONG).unwrap();
    let input = values(LONG * INPUT_DIM, 71);
    let d = grad(LONG * D_MODEL, 72);
    for scan in SCANS {
        for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            for (i, family) in FAMILIES.iter().enumerate() {
                let mut graphed = gpu_trainer(i, (1, LONG), dtype, scan);
                let mut eager = gpu_trainer(i, (1, LONG), dtype, scan);
                graphed.set_varlen(Some(&first)).unwrap();
                eager.set_varlen(Some(&first)).unwrap();
                graphed.step(&input, &d).unwrap();
                eager.step(&input, &d).unwrap();
                graphed.capture_graph().unwrap();
                for packing in [&first, &second, &first] {
                    graphed.set_varlen(Some(packing)).unwrap();
                    eager.set_varlen(Some(packing)).unwrap();
                    let replayed = graphed.step(&input, &d).unwrap();
                    assert!(
                        replayed.graph_replayed,
                        "{family} {dtype:?} {scan:?}: the captured step did not replay"
                    );
                    eager.step(&input, &d).unwrap();
                    assert!(
                        moments(&graphed) == moments(&eager),
                        "{family} {dtype:?} {scan:?}: the replayed step parts from the eager one"
                    );
                }
                assert!(
                    graphed.set_varlen(None).is_err(),
                    "{family} {dtype:?} {scan:?}: a step captured with a packing dropped it"
                );
                graphed.drop_graph();
                graphed.set_varlen(None).unwrap();
                graphed.step(&input, &d).unwrap();
                graphed.capture_graph().unwrap();
                assert!(
                    graphed.set_varlen(Some(&first)).is_err(),
                    "{family} {dtype:?} {scan:?}: a step captured without a packing took one"
                );
            }
        }
    }
}

#[test]
fn mamba1_refuses_a_packing() {
    let cfg = MambaConfig {
        d_model: D_MODEL,
        n_layers: 1,
        ..MambaConfig::default()
    };
    let mut t = MambaGpuTrainer::new(
        &MambaWeights::init(&cfg, INPUT_DIM, 1),
        cfg,
        TrainSessionCfg::new(INPUT_DIM, 1, SEQ_LEN),
        GpuOptions::default(),
    )
    .unwrap();
    let packing = Varlen::from_starts(&[vec![CUT]], SEQ_LEN).unwrap();
    let err = t.set_varlen(Some(&packing)).unwrap_err();
    assert!(err.contains("no varlen"), "{err}");
    t.set_varlen(None).unwrap();
}
