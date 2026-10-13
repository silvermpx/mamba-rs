//! The GPU surface every family shares.
//!
//! A prompt run in one pass continues like the same tokens stepped one by
//! one, across two prompts in a row; a captured graph replays the eager
//! steps, and a captured prompt graph the eager prompts, bit for bit; the
//! trait and the run-time enum reach the family's
//! own methods; and the GPU language model of every family gives the CPU
//! one's logits and loads the language-model checkpoint.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::{
    AnyGpuBackbone, GpuBackbone, GpuOptions, Mamba2Backbone, Mamba2Config, Mamba2GpuBackbone,
    Mamba2Weights, Mamba3Backbone, Mamba3Config, Mamba3GpuBackbone, Mamba3MimoBackbone,
    Mamba3MimoConfig, Mamba3MimoGpuBackbone, Mamba3MimoWeights, Mamba3Weights, MambaBackbone,
    MambaConfig, MambaGpuBackbone, MambaWeights, ModelFamily, WeightDtype,
};

/// Every storage precision; each path runs in all four.
const DTYPES: [WeightDtype; 4] = [
    WeightDtype::F32,
    WeightDtype::Tf32,
    WeightDtype::Bf16,
    WeightDtype::F16,
];

const BATCH: usize = 2;
const D_MODEL: usize = 32;
/// Wider than the model, so every family runs its input projection.
const INPUT_DIM: usize = 24;

fn m1_cfg() -> MambaConfig {
    MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
        bcdt_rms_eps: None,
        dt_rank: None,
        proj_bias: false,
        conv_bias: true,
    }
}

fn m2_cfg() -> Mamba2Config {
    Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
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
        ..Mamba3Config::default()
    }
}

fn mimo_cfg() -> Mamba3MimoConfig {
    Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: 2,
        ..Mamba3MimoConfig::default()
    }
}

/// One backbone of every family, `input_dim` wide, in `dtype`.
fn every_family(input_dim: usize, dtype: WeightDtype) -> Vec<AnyGpuBackbone> {
    let opts = GpuOptions::default().dtype(dtype);
    let m1 = MambaWeights::init(&m1_cfg(), input_dim, 11);
    let m2 = Mamba2Weights::init(&m2_cfg(), input_dim, 12);
    let siso = Mamba3Weights::init(&siso_cfg(), input_dim, 13);
    let mimo = Mamba3MimoWeights::init(&mimo_cfg(), input_dim, 14);
    vec![
        MambaGpuBackbone::new(&m1, m1_cfg(), input_dim, BATCH, opts)
            .unwrap()
            .into(),
        Mamba2GpuBackbone::new(&m2, m2_cfg(), input_dim, BATCH, opts)
            .unwrap()
            .into(),
        Mamba3GpuBackbone::new(&siso, siso_cfg(), input_dim, BATCH, opts)
            .unwrap()
            .into(),
        Mamba3MimoGpuBackbone::new(&mimo, mimo_cfg(), input_dim, BATCH, opts)
            .unwrap()
            .into(),
    ]
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

/// Step `t` of host rows `[batch, seq_len, width]`: rows `[batch, width]`.
fn step_rows(rows: &[f32], seq_len: usize, width: usize, t: usize) -> Vec<f32> {
    (0..BATCH)
        .flat_map(|b| {
            let at = (b * seq_len + t) * width;
            rows[at..at + width].iter().copied()
        })
        .collect()
}

/// Two prompts `[batch, len, width]` concatenated along time.
fn concat_in_time(a: &[f32], a_len: usize, b: &[f32], b_len: usize, width: usize) -> Vec<f32> {
    (0..BATCH)
        .flat_map(|s| {
            let pa = &a[s * a_len * width..(s + 1) * a_len * width];
            let pb = &b[s * b_len * width..(s + 1) * b_len * width];
            pa.iter().chain(pb).copied()
        })
        .collect()
}

fn max_rel_diff(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Two prompts in one pass each, against every token stepped: the rows
/// after the prompts and the step after them agree in every family. The
/// second prompt continues from the state the first one left, so the
/// state crosses a prompt seam the way it crosses a step.
#[test]
fn prompts_continue_like_steps_in_every_family() {
    let (t1, t2) = (100, 37);
    let first = values(BATCH * t1 * INPUT_DIM, 1);
    let second = values(BATCH * t2 * INPUT_DIM, 2);
    let whole = concat_in_time(&first, t1, &second, t2, INPUT_DIM);
    let next = values(BATCH * INPUT_DIM, 3);
    for mut bb in every_family(INPUT_DIM, WeightDtype::F32) {
        let family = bb.family();
        let rows = BATCH * bb.d_model();

        bb.reset().unwrap();
        let mut stepped = vec![0.0f32; rows];
        for t in 0..t1 + t2 {
            bb.step(&step_rows(&whole, t1 + t2, INPUT_DIM, t), &mut stepped)
                .unwrap();
        }
        let mut after_steps = vec![0.0f32; rows];
        bb.step(&next, &mut after_steps).unwrap();

        bb.reset().unwrap();
        bb.prefill(&first, t1).unwrap();
        bb.prefill(&second, t2).unwrap();
        let mut prefilled = vec![0.0f32; rows];
        bb.download_temporal(&mut prefilled).unwrap();
        let mut after_prefill = vec![0.0f32; rows];
        bb.step(&next, &mut after_prefill).unwrap();

        let (last, then) = (
            max_rel_diff(&prefilled, &stepped),
            max_rel_diff(&after_prefill, &after_steps),
        );
        eprintln!("{family:?}: prompt rows {last:.2e}, next step {then:.2e}");
        assert!(last < 1e-3, "{family:?}: prompt rows differ by {last}");
        assert!(then < 1e-3, "{family:?}: the step after differs by {then}");
    }
}

/// An empty prompt leaves the rows and the state as they are in every
/// family.
#[test]
fn an_empty_prompt_changes_nothing_in_every_family() {
    let prompt = values(BATCH * 5 * INPUT_DIM, 4);
    let next = values(BATCH * INPUT_DIM, 5);
    for mut bb in every_family(INPUT_DIM, WeightDtype::F32) {
        let family = bb.family();
        let rows = BATCH * bb.d_model();
        bb.reset().unwrap();
        bb.prefill(&prompt, 5).unwrap();
        let mut before = vec![0.0f32; rows];
        bb.download_temporal(&mut before).unwrap();
        bb.prefill(&[], 0).unwrap();
        let mut after = vec![0.0f32; rows];
        bb.download_temporal(&mut after).unwrap();
        assert_eq!(bits(&after), bits(&before), "{family:?}: the rows moved");
        let mut stepped = vec![0.0f32; rows];
        bb.step(&next, &mut stepped).unwrap();
        bb.reset().unwrap();
        bb.prefill(&prompt, 5).unwrap();
        let mut control = vec![0.0f32; rows];
        bb.step(&next, &mut control).unwrap();
        assert_eq!(
            bits(&stepped),
            bits(&control),
            "{family:?}: the state moved"
        );
    }
}

/// A captured graph replays the eager steps bit for bit, and a prompt as
/// long as one run before the capture still runs after it.
#[test]
fn graph_replays_eager_steps_in_every_family() {
    let t = 64;
    let prompt = values(BATCH * t * D_MODEL, 4);
    for dtype in DTYPES {
        for mut bb in every_family(D_MODEL, dtype) {
            let family = bb.family();
            let rows = BATCH * bb.d_model();
            let steps: Vec<Vec<f32>> = (0..5).map(|i| values(BATCH * D_MODEL, 10 + i)).collect();
            let run = |bb: &mut AnyGpuBackbone| -> Vec<Vec<u32>> {
                bb.reset().unwrap();
                steps
                    .iter()
                    .map(|input| {
                        let mut out = vec![0.0f32; rows];
                        bb.step(input, &mut out).unwrap();
                        bits(&out)
                    })
                    .collect()
            };
            let prompt_rows = |bb: &mut AnyGpuBackbone| -> Vec<u32> {
                bb.reset().unwrap();
                bb.prefill(&prompt, t).unwrap();
                let mut out = vec![0.0f32; rows];
                bb.download_temporal(&mut out).unwrap();
                bits(&out)
            };
            let eager = run(&mut bb);
            let eager_prompt = prompt_rows(&mut bb);
            bb.capture_graph().unwrap();
            assert!(bb.has_graph(), "{family:?} {dtype:?}: no graph");
            assert_eq!(run(&mut bb), eager, "{family:?} {dtype:?}: replay");
            assert_eq!(
                prompt_rows(&mut bb),
                eager_prompt,
                "{family:?} {dtype:?}: the prompt after the capture"
            );
        }
    }
}

/// A captured graph does not stop a prompt longer than any run before it:
/// the prompt runs eagerly and gives the bits of a backbone that never
/// captured.
#[test]
fn a_longer_prompt_runs_after_the_capture_in_every_family() {
    let t = 600;
    let prompt = values(BATCH * t * D_MODEL, 9);
    for dtype in DTYPES {
        let captured = every_family(D_MODEL, dtype);
        let plain = every_family(D_MODEL, dtype);
        for (mut a, mut b) in captured.into_iter().zip(plain) {
            let family = a.family();
            let rows = BATCH * a.d_model();
            a.capture_graph().unwrap();
            let after = a.prefill(&prompt, t);
            assert!(
                after.is_ok(),
                "{family:?} {dtype:?}: a long prompt after the capture: {after:?}"
            );
            b.prefill(&prompt, t).unwrap();
            let (mut ra, mut rb) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            a.download_temporal(&mut ra).unwrap();
            b.download_temporal(&mut rb).unwrap();
            assert_eq!(bits(&ra), bits(&rb), "{family:?} {dtype:?}");
        }
    }
}

/// A model narrower than one warp decodes on the GPU like the CPU in every
/// family: the row norms reduce over a whole warp whatever the width, and
/// no family refuses the width.
#[test]
fn every_family_decodes_a_model_narrower_than_a_warp() {
    const DM: usize = 8;
    // One CPU decoder per sequence: the family's backbone with its own state.
    type Stepper = Box<dyn FnMut(&[f32], &mut [f32])>;
    macro_rules! stepper {
        ($bb:expr) => {{
            let bb = $bb;
            let mut state = bb.alloc_state();
            let mut scratch = bb.alloc_scratch();
            Box::new(move |x: &[f32], y: &mut [f32]| {
                bb.forward_step(x, y, &mut state, &mut scratch)
            }) as Stepper
        }};
    }
    let opts = GpuOptions::default().dtype(WeightDtype::F32);
    let m1 = MambaConfig {
        d_model: DM,
        ..m1_cfg()
    };
    let m2 = Mamba2Config {
        d_model: DM,
        ..m2_cfg()
    };
    let siso = Mamba3Config {
        d_model: DM,
        ..siso_cfg()
    };
    let mimo = Mamba3MimoConfig {
        d_model: DM,
        ..mimo_cfg()
    };
    let w1 = MambaWeights::init(&m1, INPUT_DIM, 21);
    let w2 = Mamba2Weights::init(&m2, INPUT_DIM, 22);
    let w3 = Mamba3Weights::init(&siso, INPUT_DIM, 23);
    let w4 = Mamba3MimoWeights::init(&mimo, INPUT_DIM, 24);
    let cases: Vec<(AnyGpuBackbone, Vec<Stepper>)> = vec![
        (
            MambaGpuBackbone::new(&w1, m1, INPUT_DIM, BATCH, opts)
                .unwrap()
                .into(),
            (0..BATCH)
                .map(|_| stepper!(MambaBackbone::new(w1.clone(), m1, INPUT_DIM).unwrap()))
                .collect(),
        ),
        (
            Mamba2GpuBackbone::new(&w2, m2, INPUT_DIM, BATCH, opts)
                .unwrap()
                .into(),
            (0..BATCH)
                .map(|_| stepper!(Mamba2Backbone::new(w2.clone(), m2, INPUT_DIM).unwrap()))
                .collect(),
        ),
        (
            Mamba3GpuBackbone::new(&w3, siso, INPUT_DIM, BATCH, opts)
                .unwrap()
                .into(),
            (0..BATCH)
                .map(|_| stepper!(Mamba3Backbone::new(w3.clone(), siso, INPUT_DIM).unwrap()))
                .collect(),
        ),
        (
            Mamba3MimoGpuBackbone::new(&w4, mimo, INPUT_DIM, BATCH, opts)
                .unwrap()
                .into(),
            (0..BATCH)
                .map(|_| stepper!(Mamba3MimoBackbone::new(w4.clone(), mimo, INPUT_DIM).unwrap()))
                .collect(),
        ),
    ];
    for (mut gpu, mut cpu) in cases {
        let family = gpu.family();
        for t in 0..6 {
            let input = values(BATCH * INPUT_DIM, 40 + t);
            let mut got = vec![0.0f32; BATCH * DM];
            gpu.step(&input, &mut got).unwrap();
            let mut want = vec![0.0f32; BATCH * DM];
            for (b, step) in cpu.iter_mut().enumerate() {
                step(
                    &input[b * INPUT_DIM..(b + 1) * INPUT_DIM],
                    &mut want[b * DM..(b + 1) * DM],
                );
            }
            let diff = max_rel_diff(&got, &want);
            assert!(
                diff < 1e-3,
                "{family:?} step {t}: GPU and CPU differ by {diff}"
            );
        }
    }
}

/// A capture leaves the backbone where the caller left it: after a prompt,
/// the rows of its last position and the step that follows are the bits of
/// a backbone that never captured.
#[test]
fn a_capture_keeps_the_prompt_position_in_every_family() {
    let t = 50;
    let prompt = values(BATCH * t * D_MODEL, 31);
    let next = values(BATCH * D_MODEL, 32);
    for dtype in DTYPES {
        let captured = every_family(D_MODEL, dtype);
        let plain = every_family(D_MODEL, dtype);
        for (mut a, mut b) in captured.into_iter().zip(plain) {
            let family = a.family();
            let rows = BATCH * a.d_model();
            b.prefill(&prompt, t).unwrap();
            let mut want_rows = vec![0.0f32; rows];
            b.download_temporal(&mut want_rows).unwrap();
            let mut want_next = vec![0.0f32; rows];
            b.step(&next, &mut want_next).unwrap();

            a.prefill(&prompt, t).unwrap();
            a.capture_graph().unwrap();
            let mut got_rows = vec![0.0f32; rows];
            a.download_temporal(&mut got_rows).unwrap();
            let mut got_next = vec![0.0f32; rows];
            a.step(&next, &mut got_next).unwrap();
            assert_eq!(
                bits(&got_rows),
                bits(&want_rows),
                "{family:?} {dtype:?}: the prompt rows after the capture"
            );
            assert_eq!(
                bits(&got_next),
                bits(&want_next),
                "{family:?} {dtype:?}: the step after the capture"
            );
        }
    }
}

/// A second capture replaces the first, and a cleared graph hands the
/// steps back to the eager path: every step keeps the bits of a backbone
/// that only ever ran eagerly.
#[test]
fn a_second_capture_and_a_clear_in_every_family() {
    for dtype in DTYPES {
        let graphs = every_family(D_MODEL, dtype);
        let plain = every_family(D_MODEL, dtype);
        for (mut a, mut b) in graphs.into_iter().zip(plain) {
            let family = a.family();
            let rows = BATCH * a.d_model();
            let step_both = |a: &mut AnyGpuBackbone, b: &mut AnyGpuBackbone, seed: u32| {
                let input = values(BATCH * D_MODEL, seed);
                let (mut ra, mut rb) = (vec![0.0f32; rows], vec![0.0f32; rows]);
                a.step(&input, &mut ra).unwrap();
                b.step(&input, &mut rb).unwrap();
                assert_eq!(bits(&ra), bits(&rb), "{family:?} {dtype:?}: step {seed}");
            };
            a.capture_graph().unwrap();
            step_both(&mut a, &mut b, 50);
            a.capture_graph().unwrap();
            assert!(
                a.has_graph(),
                "{family:?} {dtype:?}: no graph after recapture"
            );
            step_both(&mut a, &mut b, 51);
            a.clear_graph().unwrap();
            assert!(
                !a.has_graph(),
                "{family:?} {dtype:?}: graph after the clear"
            );
            step_both(&mut a, &mut b, 52);
        }
    }
}

/// A prompt pass captured as one graph gives the eager bits in every family
/// and precision, with and without an input projection: the capture on a
/// state mid-sequence leaves the rows and the state where they were,
/// prompts of the captured length match the eager backbone's rows and the
/// step after each, a prompt of another length runs eagerly and drops the
/// graph, and a prompt graph captured again works beside a decode graph.
#[test]
fn prompt_graph_replays_the_eager_prefill_in_every_family() {
    const LEN: usize = 37;
    for input_dim in [INPUT_DIM, D_MODEL] {
        for dtype in DTYPES {
            let graphs = every_family(input_dim, dtype);
            let plain = every_family(input_dim, dtype);
            for (mut a, mut b) in graphs.into_iter().zip(plain) {
                let label = format!("{:?} {dtype:?} input {input_dim}", a.family());
                let x = values(BATCH * input_dim, 5);
                let rows = |bb: &AnyGpuBackbone| {
                    let mut v = vec![0.0f32; BATCH * D_MODEL];
                    bb.download_temporal(&mut v).unwrap();
                    bits(&v)
                };
                let step_both = |a: &mut AnyGpuBackbone, b: &mut AnyGpuBackbone, what: &str| {
                    let (mut ra, mut rb) =
                        (vec![0.0f32; BATCH * D_MODEL], vec![0.0f32; BATCH * D_MODEL]);
                    a.step(&x, &mut ra).unwrap();
                    b.step(&x, &mut rb).unwrap();
                    assert_eq!(bits(&ra), bits(&rb), "{label}: {what}");
                };
                let prompt_both =
                    |a: &mut AnyGpuBackbone, b: &mut AnyGpuBackbone, len: usize, seed: u32| {
                        let prompt = values(BATCH * len * input_dim, seed);
                        a.prefill(&prompt, len).unwrap();
                        b.prefill(&prompt, len).unwrap();
                        assert_eq!(rows(a), rows(b), "{label}: prompt {seed} of {len}");
                    };
                step_both(&mut a, &mut b, "the first step");
                a.capture_prefill_graph(LEN).unwrap();
                assert!(a.has_prefill_graph(), "{label}");
                assert_eq!(rows(&a), rows(&b), "{label}: the capture moved the rows");
                step_both(&mut a, &mut b, "the step after the capture");
                for seed in 10..13 {
                    prompt_both(&mut a, &mut b, LEN, seed);
                    step_both(&mut a, &mut b, "the step after a replayed prompt");
                }
                assert!(a.has_prefill_graph(), "{label}: replays keep the graph");
                prompt_both(&mut a, &mut b, 5, 20);
                assert!(
                    !a.has_prefill_graph(),
                    "{label}: another length drops the graph"
                );
                step_both(&mut a, &mut b, "the step after an eager prompt");
                a.capture_prefill_graph(LEN).unwrap();
                a.capture_graph().unwrap();
                for seed in 30..32 {
                    prompt_both(&mut a, &mut b, LEN, seed);
                    step_both(&mut a, &mut b, "a replayed step after a replayed prompt");
                }
                a.clear_prefill_graph();
                assert!(!a.has_prefill_graph() && a.has_graph(), "{label}");
                prompt_both(&mut a, &mut b, LEN, 40);
            }
        }
    }
}

/// The trait and the enum hand every call to the family's own method: the
/// same model driven through both gives the same bits and reports the
/// same shape.
#[test]
fn trait_and_enum_reach_the_family_methods() {
    let mut own = every_family(INPUT_DIM, WeightDtype::F32);
    let mut via = every_family(INPUT_DIM, WeightDtype::F32);
    let want = [
        ModelFamily::Mamba1,
        ModelFamily::Mamba2,
        ModelFamily::Mamba3,
        ModelFamily::Mamba3Mimo,
    ];
    let input = values(BATCH * INPUT_DIM, 5);
    for ((a, b), family) in own.iter_mut().zip(via.iter_mut()).zip(want) {
        assert_eq!(a.family(), family);
        let rows = BATCH * a.d_model();
        let mut by_family = vec![0.0f32; rows];
        let mut by_trait = vec![0.0f32; rows];
        match a {
            AnyGpuBackbone::M1(bb) => bb.step(&input, &mut by_family),
            AnyGpuBackbone::M2(bb) => bb.step(&input, &mut by_family),
            AnyGpuBackbone::M3(bb) => bb.step(&input, &mut by_family),
            AnyGpuBackbone::M3Mimo(bb) => bb.step(&input, &mut by_family),
        }
        .unwrap();
        let dynamic: &mut dyn GpuBackbone = b;
        dynamic.step(&input, &mut by_trait).unwrap();
        assert_eq!(bits(&by_trait), bits(&by_family), "{family:?}");
        assert_eq!(
            (dynamic.batch(), dynamic.input_dim(), dynamic.d_model()),
            (BATCH, INPUT_DIM, D_MODEL),
            "{family:?}"
        );
        assert_eq!(dynamic.dtype(), WeightDtype::F32, "{family:?}");
        assert!(!dynamic.has_graph(), "{family:?}");
    }
}

/// A SISO prompt continued from a zeroed state gives the bits of the
/// stateless window: the language models reset before a prompt, and the
/// continued window's boundary fold must add nothing there.
#[test]
fn siso_prompt_from_zero_state_matches_the_stateless_window() {
    let t = 300;
    let cfg = siso_cfg();
    let weights = Mamba3Weights::init(&cfg, D_MODEL, 21);
    let prompt = values(BATCH * t * D_MODEL, 6);
    let next = values(BATCH * D_MODEL, 7);
    for dtype in DTYPES {
        let opts = GpuOptions::default().dtype(dtype);
        let mut bb = Mamba3GpuBackbone::new(&weights, cfg, D_MODEL, BATCH, opts).unwrap();
        let rows = BATCH * D_MODEL;
        let finish = |bb: &mut Mamba3GpuBackbone| -> (Vec<u32>, Vec<u32>) {
            let mut last = vec![0.0f32; rows];
            bb.download_temporal(&mut last).unwrap();
            let mut after = vec![0.0f32; rows];
            bb.step(&next, &mut after).unwrap();
            (bits(&last), bits(&after))
        };

        bb.reset().unwrap();
        bb.prefill(&prompt, t).unwrap();
        let continued = finish(&mut bb);

        bb.reset().unwrap();
        let mut pf = bb.alloc_prefill(t).unwrap();
        let rows_in = GpuBuffer::from_cpu(bb.stream(), &prompt).unwrap();
        bb.prefill_sequence(&mut pf, &rows_in, t, false).unwrap();
        let stateless = finish(&mut bb);

        assert_eq!(
            continued.0, stateless.0,
            "{dtype:?}: the prompt's last rows"
        );
        assert_eq!(continued.1, stateless.1, "{dtype:?}: the step after it");
    }
}

#[cfg(feature = "hf")]
mod language_model {
    use super::*;
    use mamba_rs::module::sample::SampleParams;
    use mamba_rs::{
        CpuOptions, Mamba2Backbone, Mamba3Backbone, Mamba3MimoBackbone, MambaBackbone, MambaGpuLM,
        MambaLM,
    };

    const VOCAB: usize = 90;
    /// The embedding rows, the vocabulary padded the way the loaders pad.
    const VOCAB_ROWS: usize = 96;

    fn models(opts: GpuOptions) -> Vec<(MambaLM, MambaGpuLM)> {
        models_with(opts, CpuOptions::default())
    }

    /// Every family's model on the CPU, its backbone in the precision
    /// `cpu` names, and on the GPU.
    fn models_with(opts: GpuOptions, cpu: CpuOptions) -> Vec<(MambaLM, MambaGpuLM)> {
        let embed = values(VOCAB_ROWS * D_MODEL, 30);
        let lm = |cpu: Result<MambaLM, String>, gpu: Result<MambaGpuLM, String>| {
            (cpu.unwrap(), gpu.unwrap())
        };
        let mut m1 = MambaWeights::init(&m1_cfg(), D_MODEL, 31);
        m1.input_proj_w.clear();
        m1.input_proj_b.clear();
        let m2 = Mamba2Weights::init(&m2_cfg(), D_MODEL, 32);
        let siso = Mamba3Weights::init(&siso_cfg(), D_MODEL, 33);
        let mimo = Mamba3MimoWeights::init(&mimo_cfg(), D_MODEL, 34);
        vec![
            lm(
                MambaBackbone::new_with(m1.clone(), m1_cfg(), D_MODEL, cpu)
                    .and_then(|bb| MambaLM::new(bb, embed.clone(), None, VOCAB)),
                MambaGpuBackbone::new(&m1, m1_cfg(), D_MODEL, 1, opts)
                    .and_then(|bb| MambaGpuLM::new(bb, embed.clone(), None, VOCAB)),
            ),
            lm(
                Mamba2Backbone::new_with(m2.clone(), m2_cfg(), D_MODEL, cpu)
                    .and_then(|bb| MambaLM::new(bb, embed.clone(), None, VOCAB)),
                Mamba2GpuBackbone::new(&m2, m2_cfg(), D_MODEL, 1, opts)
                    .and_then(|bb| MambaGpuLM::new(bb, embed.clone(), None, VOCAB)),
            ),
            lm(
                Mamba3Backbone::new_with(siso.clone(), siso_cfg(), D_MODEL, cpu)
                    .and_then(|bb| MambaLM::new(bb, embed.clone(), None, VOCAB)),
                Mamba3GpuBackbone::new(&siso, siso_cfg(), D_MODEL, 1, opts)
                    .and_then(|bb| MambaGpuLM::new(bb, embed.clone(), None, VOCAB)),
            ),
            lm(
                Mamba3MimoBackbone::new_with(mimo.clone(), mimo_cfg(), D_MODEL, cpu)
                    .and_then(|bb| MambaLM::new(bb, embed.clone(), None, VOCAB)),
                Mamba3MimoGpuBackbone::new(&mimo, mimo_cfg(), D_MODEL, 1, opts)
                    .and_then(|bb| MambaGpuLM::new(bb, embed.clone(), None, VOCAB)),
            ),
        ]
    }

    /// The relative l2 distance of `a` from `b`.
    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += f64::from(y).powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    }

    /// The bf16 and f16 models of every family against the CPU model whose
    /// backbone and head compute in the same precision: after a short
    /// prompt fed token by token and after a long one run in one pass, its
    /// logits land closer to the GPU's than the f32 CPU model's do.
    #[test]
    fn half_language_models_track_the_cpu_lane_in_every_family() {
        let params = SampleParams {
            max_tokens: 0,
            ..SampleParams::default()
        };
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let opts = GpuOptions::default().dtype(dtype);
            let lane = CpuOptions::default().dtype(dtype);
            let f32_models = models_with(opts, CpuOptions::default());
            let lane_models = models_with(opts, lane);
            for ((mut f32_cpu, mut gpu), (mut lane_cpu, _)) in
                f32_models.into_iter().zip(lane_models)
            {
                let family = gpu.family();
                for len in [12, 300] {
                    let prompt: Vec<u32> = (0..len).map(|i| (i * 37 % VOCAB) as u32).collect();
                    gpu.generate(&prompt, &params).unwrap();
                    f32_cpu.generate(&prompt, &params);
                    lane_cpu.generate(&prompt, &params);
                    let d_f32 = rel_l2(f32_cpu.last_logits(0), gpu.last_logits(0));
                    let d_lane = rel_l2(lane_cpu.last_logits(0), gpu.last_logits(0));
                    eprintln!(
                        "{family:?} {dtype:?} prompt {len}: logits vs CPU f32 {d_f32:.2e}, \
                         vs CPU lane {d_lane:.2e}"
                    );
                    assert!(
                        d_lane < d_f32,
                        "{family:?} {dtype:?} prompt {len}: the CPU lane ({d_lane:.3e}) is no \
                         closer than f32 ({d_f32:.3e})"
                    );
                }
            }
        }
    }

    /// The logits after a short prompt (fed token by token) and after a
    /// long one (one pass) match the CPU model's in every family.
    #[test]
    fn gpu_language_model_gives_the_cpu_logits_in_every_family() {
        let params = SampleParams {
            max_tokens: 0,
            ..SampleParams::default()
        };
        for (mut cpu, mut gpu) in models(GpuOptions::default()) {
            assert_eq!(cpu.family(), gpu.family());
            for len in [12, 300] {
                let prompt: Vec<u32> = (0..len).map(|i| (i * 37 % VOCAB) as u32).collect();
                cpu.generate(&prompt, &params);
                gpu.generate(&prompt, &params).unwrap();
                let diff = max_rel_diff(gpu.last_logits(0), cpu.last_logits(0));
                eprintln!("{:?} prompt {len}: logits {diff:.2e}", gpu.family());
                assert!(diff < 1e-3, "{:?} prompt {len}: {diff}", gpu.family());
            }
        }
    }

    /// A language-model checkpoint loads onto the device as the model it
    /// was saved from: in every family its logits are those of the GPU
    /// model built from the same parts, bit for bit.
    #[test]
    fn gpu_language_model_loads_a_checkpoint_in_every_family() {
        let params = SampleParams {
            max_tokens: 0,
            ..SampleParams::default()
        };
        let prompt: Vec<u32> = (0..12).map(|i| (i * 37 % VOCAB) as u32).collect();
        let bits = |l: &[f32]| l.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        for (cpu, mut gpu) in models(GpuOptions::default()) {
            let family = gpu.family();
            let path = std::env::temp_dir().join(format!(
                "mamba_rs_gpu_lm_checkpoint_{}_{family:?}.safetensors",
                std::process::id()
            ));
            cpu.save_checkpoint(&path).unwrap();
            let loaded = MambaGpuLM::from_checkpoint(&path, 1, GpuOptions::default());
            std::fs::remove_file(&path).ok();
            let mut loaded = loaded.unwrap();
            assert_eq!(loaded.family(), family);
            gpu.generate(&prompt, &params).unwrap();
            loaded.generate(&prompt, &params).unwrap();
            assert_eq!(
                bits(loaded.last_logits(0)),
                bits(gpu.last_logits(0)),
                "{family:?}"
            );
        }
    }

    /// Generation runs with a captured graph in every family, and a graph
    /// does not change the tokens: after a short prompt fed token by token,
    /// and after a long one, run in one pass by a model that captured its
    /// graph before it ever saw a prompt that long.
    #[test]
    fn gpu_language_model_generates_with_a_graph_in_every_family() {
        let params = SampleParams {
            temperature: 0.0,
            max_tokens: 8,
            ..SampleParams::default()
        };
        let short: Vec<u32> = (0..9).map(|i| (i * 11 % VOCAB) as u32).collect();
        let long: Vec<u32> = (0..300).map(|i| (i * 13 % VOCAB) as u32).collect();
        for dtype in DTYPES {
            let opts = GpuOptions::default().dtype(dtype);
            for ((_, mut graphed), (_, mut eager)) in models(opts).into_iter().zip(models(opts)) {
                let family = graphed.family();
                graphed.capture_graph().unwrap();
                for prompt in [&short, &long] {
                    let n = prompt.len();
                    let want = eager.generate(prompt, &params).unwrap();
                    let got = graphed.generate(prompt, &params);
                    assert!(got.is_ok(), "{family:?} {dtype:?} prompt {n}: {got:?}");
                    assert_eq!(got.unwrap(), want, "{family:?} {dtype:?} prompt {n}");
                    assert_eq!(want.len(), 8, "{family:?} {dtype:?}");
                }
            }
        }
    }
}
