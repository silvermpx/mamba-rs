//! Several sequences packed into the rows of one window (varlen) train on
//! the host as they train alone: the packed rows of every sequence equal
//! its own run from the empty state, and so does its input gradient, bit
//! for bit (on a BLAS backend, to the precision's rounding); the weight
//! gradients are the sum of the separate runs' to f32 rounding; a loss on one sequence leaves every other sequence's input
//! gradient exactly zero; the sequence before a row's first start
//! continues the carried state, and the state after the window is the last
//! sequence's. Every precision of the host trainer holds all of it.
//! Mamba-1, which the reference does not pack, refuses a packing.

use mamba_rs::{
    AnyCpuTrainer, BackwardOpts, CpuOptions, CpuTrainer, Mamba2Config, Mamba2CpuTrainer,
    Mamba2Weights, Mamba3Config, Mamba3CpuTrainer, Mamba3MimoConfig, Mamba3MimoCpuTrainer,
    Mamba3MimoWeights, Mamba3Weights, MambaConfig, MambaCpuTrainer, MambaWeights, TrainSessionCfg,
    TrainerState, Varlen, WeightDtype,
};

const D_MODEL: usize = 32;
const INPUT_DIM: usize = 24;
/// The packed row: a sequence of 5 tokens, then one of 7.
const SEQ_LEN: usize = 12;
const CUT: usize = 5;
const DTYPES: [WeightDtype; 4] = [
    WeightDtype::F32,
    WeightDtype::Tf32,
    WeightDtype::Bf16,
    WeightDtype::F16,
];

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

/// Whether a BLAS backend serves the host GEMMs. It blocks a GEMM by its
/// size, so a packed window and its sequences' own runs, which differ in
/// rows, round apart there: their sums agree to the precision's rounding,
/// not bit for bit as on the BLAS-free GEMMs, whose sums do not depend on
/// a call's rows.
const BLAS: bool = cfg!(any(feature = "accelerate", feature = "gemm-blas"));

/// The share of the largest value the two blockings may part by on a BLAS
/// backend: the f32 rounding for f32, the rounding of the operands for the
/// lanes that round them.
fn blas_tol(dtype: WeightDtype) -> f32 {
    match dtype {
        WeightDtype::F32 => 1e-5,
        WeightDtype::Tf32 => 1e-3,
        WeightDtype::F16 => 4e-3,
        WeightDtype::Bf16 => 2e-2,
    }
}

/// `got` equals `want` bit for bit, or on a BLAS backend within
/// [`blas_tol`] of the largest value.
fn assert_same(what: &str, dtype: WeightDtype, got: &[f32], want: &[f32]) {
    if BLAS {
        assert_eq!(got.len(), want.len(), "{what}");
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(want)
            .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
        assert!(
            worst <= blas_tol(dtype) * scale,
            "{what}: off by {worst:e} of {scale:e}"
        );
    } else {
        assert_eq!(bits(got), bits(want), "{what}");
    }
}

/// A trainer of each packing family over `batch` rows of `seq_len`
/// positions, computing in `dtype`. An f16 trainer starts its loss scale
/// at 1024: at the default 65536 these small models overflow and a
/// comparison would read NaNs.
fn trainers(
    (batch, seq_len): (usize, usize),
    dtype: WeightDtype,
) -> Vec<(&'static str, AnyCpuTrainer)> {
    let s = TrainSessionCfg::new(INPUT_DIM, batch, seq_len);
    let o = CpuOptions::default().dtype(dtype);
    let m2 = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 2);
    let siso = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 3);
    let mimo = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 4);
    let mut all: Vec<(&'static str, AnyCpuTrainer)> = vec![
        (
            "Mamba-2",
            Mamba2CpuTrainer::new_with(&m2, m2_cfg(), s, o)
                .unwrap()
                .into(),
        ),
        (
            "Mamba-3 SISO",
            Mamba3CpuTrainer::new_with(&siso, siso_cfg(), s, o)
                .unwrap()
                .into(),
        ),
        (
            "Mamba-3 MIMO",
            Mamba3MimoCpuTrainer::new_with(&mimo, mimo_cfg(), s, o)
                .unwrap()
                .into(),
        ),
    ];
    for (_, t) in &mut all {
        t.load_scaler_state(1024.0, 0).unwrap();
    }
    all
}

/// The trainer of family `i` over one row of `seq_len` positions.
fn trainer(i: usize, seq_len: usize, dtype: WeightDtype) -> AnyCpuTrainer {
    trainers((1, seq_len), dtype).swap_remove(i).1
}

/// The forward of `input` and the accumulating backward of `d`: the rows,
/// the input gradient and the weight gradient, every value finite (a
/// window that overflowed would compare its NaNs).
fn run(t: &mut AnyCpuTrainer, input: &[f32], d: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rows = vec![0.0f32; d.len()];
    let mut d_input = vec![0.0f32; input.len()];
    t.forward(input, &mut rows).unwrap();
    t.backward_step_input_grad(
        d,
        BackwardOpts::default().with_accumulate_only(true),
        &mut d_input,
    )
    .unwrap();
    let grads = t.download_grad_arena().unwrap();
    for (what, v) in [
        ("rows", &rows),
        ("input gradient", &d_input),
        ("weight gradient", &grads),
    ] {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "the window's {what} overflowed"
        );
    }
    (rows, d_input, grads)
}

/// Packs the sequences that start at `starts[b]` into row `b` of a window
/// of `seq_len` positions and checks each against its own run from the
/// empty state, in every precision.
fn check_packed_against_alone(starts: &[Vec<usize>], seq_len: usize, seed: u32) {
    let batch = starts.len();
    let input = values(batch * seq_len * INPUT_DIM, seed);
    let d = grad(batch * seq_len * D_MODEL, seed + 1);
    let packing = Varlen::from_starts(starts, seq_len).unwrap();
    // Every sequence as its first and end positions in the window's rows.
    // The positions before a row's first start continue the carried state,
    // the empty one of a fresh trainer, as a run of their own does.
    let mut bounds = Vec::new();
    for (b, row) in starts.iter().enumerate() {
        let mut cuts: Vec<usize> = std::iter::once(0)
            .chain(row.iter().copied().filter(|&t| t > 0))
            .collect();
        cuts.push(seq_len);
        bounds.extend(
            cuts.windows(2)
                .map(|w| (b * seq_len + w[0], b * seq_len + w[1])),
        );
    }
    for dtype in DTYPES {
        for (i, (family, mut packed)) in trainers((batch, seq_len), dtype).into_iter().enumerate() {
            packed.set_varlen(Some(&packing)).unwrap();
            let (rows, d_input, grads) = run(&mut packed, &input, &d);
            let mut sum = vec![0.0f32; grads.len()];
            for &(a, e) in &bounds {
                let (rows_s, d_input_s, grads_s) = run(
                    &mut trainer(i, e - a, dtype),
                    &input[a * INPUT_DIM..e * INPUT_DIM],
                    &d[a * D_MODEL..e * D_MODEL],
                );
                assert_same(
                    &format!("{family} {dtype:?}: the rows of the sequence at {a}..{e}"),
                    dtype,
                    &rows[a * D_MODEL..e * D_MODEL],
                    &rows_s,
                );
                assert_same(
                    &format!("{family} {dtype:?}: the input gradient of the sequence at {a}..{e}"),
                    dtype,
                    &d_input[a * INPUT_DIM..e * INPUT_DIM],
                    &d_input_s,
                );
                for (s, g) in sum.iter_mut().zip(&grads_s) {
                    *s += g;
                }
            }
            let scale = grads.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = grads
                .iter()
                .zip(&sum)
                .fold(0.0f32, |m, (g, s)| m.max((g - s).abs()));
            let tol = if BLAS {
                blas_tol(dtype).max(1e-5)
            } else {
                1e-5
            };
            assert!(
                worst <= tol * scale,
                "{family} {dtype:?}: the weight gradient is off the separate runs' sum by \
                 {worst:e} of {scale:e}"
            );
        }
    }
}

#[test]
fn packed_sequences_train_as_they_do_alone() {
    check_packed_against_alone(&[vec![0, CUT]], SEQ_LEN, 1);
    // Starts on the tapes' checkpoint boundaries (every 8 or 16 tokens), a
    // sequence of one token inside the row and one at its end.
    check_packed_against_alone(&[vec![0, 8, 16, 17, 32, 39]], 40, 11);
    // Two rows packed apart, the second continuing its carried sequence
    // until its first start.
    check_packed_against_alone(&[vec![0, CUT], vec![3, 9]], SEQ_LEN, 21);
}

#[test]
fn a_loss_on_one_sequence_leaves_the_other_untouched() {
    let input = values(SEQ_LEN * INPUT_DIM, 3);
    // The loss on the second sequence only: without the packing its
    // gradient would reach the first sequence's inputs through the state.
    let mut d = grad(SEQ_LEN * D_MODEL, 4);
    d[..CUT * D_MODEL].fill(0.0);
    let packing = Varlen::from_starts(&[vec![CUT]], SEQ_LEN).unwrap();
    for dtype in DTYPES {
        for (family, mut t) in trainers((1, SEQ_LEN), dtype) {
            t.set_varlen(Some(&packing)).unwrap();
            let (_, d_input, _) = run(&mut t, &input, &d);
            assert!(
                d_input[..CUT * INPUT_DIM].iter().all(|&v| v == 0.0),
                "{family} {dtype:?}: the loss on the second sequence reached the first"
            );
            assert!(
                d_input[CUT * INPUT_DIM..].iter().any(|&v| v != 0.0),
                "{family} {dtype:?}: no gradient on the second sequence"
            );
            // Without the packing the first sequence's inputs do get one.
            t.set_varlen(None).unwrap();
            t.reset_state().unwrap();
            let (_, d_plain, _) = run(&mut t, &input, &d);
            assert!(
                d_plain[..CUT * INPUT_DIM].iter().any(|&v| v != 0.0),
                "{family} {dtype:?}: the unpacked control carries no gradient back"
            );
        }
    }
}

/// An f16 window whose second sequence overflows leaves the first
/// sequence's input gradient exactly zero, not NaN: the cut at a start
/// drops the overflowed carries instead of scaling them by zero.
#[test]
fn an_overflow_in_one_sequence_stays_in_it() {
    let input = values(SEQ_LEN * INPUT_DIM, 8);
    // The loss on the second sequence only, one of its values past what
    // the scaled f16 gradient holds.
    let mut d = grad(SEQ_LEN * D_MODEL, 9);
    d[..CUT * D_MODEL].fill(0.0);
    d[CUT * D_MODEL] = f32::MAX;
    let packing = Varlen::from_starts(&[vec![CUT]], SEQ_LEN).unwrap();
    for (family, mut t) in trainers((1, SEQ_LEN), WeightDtype::F16) {
        t.set_varlen(Some(&packing)).unwrap();
        let mut rows = vec![0.0f32; SEQ_LEN * D_MODEL];
        let mut d_input = vec![0.0f32; SEQ_LEN * INPUT_DIM];
        t.forward(&input, &mut rows).unwrap();
        t.backward_step_input_grad(
            &d,
            BackwardOpts::default().with_accumulate_only(true),
            &mut d_input,
        )
        .unwrap();
        assert!(
            d_input[CUT * INPUT_DIM..].iter().any(|v| !v.is_finite()),
            "{family}: the second sequence did not overflow, so the test proves nothing"
        );
        assert!(
            d_input[..CUT * INPUT_DIM].iter().all(|&v| v == 0.0),
            "{family}: the overflow of the second sequence reached the first"
        );
    }
}

/// The positions before a row's first start continue the state carried
/// from the window before, and those after it start from the empty state.
#[test]
fn the_carried_state_reaches_only_the_first_sequence() {
    let warm = values(SEQ_LEN * INPUT_DIM, 5);
    let input = values(SEQ_LEN * INPUT_DIM, 6);
    let packing = Varlen::from_starts(&[vec![CUT]], SEQ_LEN).unwrap();
    let in_b = &input[CUT * INPUT_DIM..];
    let mut sink = vec![0.0f32; SEQ_LEN * D_MODEL];
    for dtype in DTYPES {
        let shape = (1, SEQ_LEN);
        for (i, (((family, mut packed), (_, mut plain)), (_, mut fresh_b))) in
            trainers(shape, dtype)
                .into_iter()
                .zip(trainers(shape, dtype))
                .zip(trainers((1, SEQ_LEN - CUT), dtype))
                .enumerate()
        {
            packed.forward(&warm, &mut sink).unwrap();
            plain.forward(&warm, &mut sink).unwrap();
            packed.set_varlen(Some(&packing)).unwrap();
            let (mut rows, mut rows_plain, mut rows_b) = (
                vec![0.0f32; SEQ_LEN * D_MODEL],
                vec![0.0f32; SEQ_LEN * D_MODEL],
                vec![0.0f32; (SEQ_LEN - CUT) * D_MODEL],
            );
            packed.forward(&input, &mut rows).unwrap();
            plain.forward(&input, &mut rows_plain).unwrap();
            fresh_b.forward(in_b, &mut rows_b).unwrap();
            assert_same(
                &format!("{family} {dtype:?}: the carried sequence"),
                dtype,
                &rows[..CUT * D_MODEL],
                &rows_plain[..CUT * D_MODEL],
            );
            assert_same(
                &format!("{family} {dtype:?}: the sequence that starts"),
                dtype,
                &rows[CUT * D_MODEL..],
                &rows_b,
            );
            // The state after the window is the last sequence's: the next
            // window of the packed trainer continues as the fresh run does.
            // The fresh run of the second sequence alone holds the state the
            // packed trainer carries; a trainer of the window's length loads
            // it.
            packed.set_varlen(None).unwrap();
            let next = values(SEQ_LEN * INPUT_DIM, 7);
            let (mut a, mut b) = (
                vec![0.0f32; SEQ_LEN * D_MODEL],
                vec![0.0f32; SEQ_LEN * D_MODEL],
            );
            packed.forward(&next, &mut a).unwrap();
            let state = fresh_b.trainer_state().unwrap();
            let mut cont = trainer(i, SEQ_LEN, dtype);
            cont.load_trainer_state(&state).unwrap();
            cont.forward(&next, &mut b).unwrap();
            assert_same(
                &format!("{family} {dtype:?}: the state after the window"),
                dtype,
                &a,
                &b,
            );
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
    let mut t = MambaCpuTrainer::new(
        &MambaWeights::init(&cfg, INPUT_DIM, 1),
        cfg,
        TrainSessionCfg::new(INPUT_DIM, 1, SEQ_LEN),
    )
    .unwrap();
    let packing = Varlen::from_starts(&[vec![CUT]], SEQ_LEN).unwrap();
    let err = t.set_varlen(Some(&packing)).unwrap_err();
    assert!(err.contains("no varlen"), "{err}");
    t.set_varlen(None).unwrap();
}

#[test]
fn packing_contract_backward_requires_the_boundaries_of_its_forward() {
    let input = values(SEQ_LEN * INPUT_DIM, 32);
    let mut d = grad(SEQ_LEN * D_MODEL, 33);
    d[..CUT * D_MODEL].fill(0.0);
    let mut rows = vec![0.0; d.len()];
    let mut dx = vec![0.0; input.len()];
    let packing = Varlen::from_starts(&[vec![CUT]], SEQ_LEN).unwrap();
    let mut ids = vec![0; SEQ_LEN];
    ids[CUT..].fill(i32::MAX);
    let relabeled = Varlen::from_seq_idx(ids, 1, SEQ_LEN).unwrap();
    let wrong_shape = Varlen::from_seq_idx(vec![0; SEQ_LEN], 2, SEQ_LEN / 2).unwrap();
    let uncut = Varlen::from_starts(&[vec![]], SEQ_LEN).unwrap();
    for (family, mut t) in trainers((1, SEQ_LEN), WeightDtype::F32) {
        t.set_varlen(Some(&packing)).unwrap();
        t.forward(&input, &mut rows).unwrap();
        assert!(t.set_varlen(Some(&wrong_shape)).is_err(), "{family}");
        t.set_varlen(Some(&relabeled)).unwrap();
        t.backward_step_input_grad(&d, BackwardOpts::default(), &mut dx)
            .unwrap();
        assert!(dx[..CUT * INPUT_DIM].iter().all(|&x| x == 0.0), "{family}");
        assert!(dx[CUT * INPUT_DIM..].iter().any(|&x| x != 0.0), "{family}");

        t.forward(&input, &mut rows).unwrap();
        t.set_varlen(None).unwrap();
        assert!(
            t.backward_step(&d, BackwardOpts::default()).is_err(),
            "{family} accepted a backward across a changed sequence boundary"
        );
        t.forward(&input, &mut rows).unwrap();
        t.set_varlen(Some(&uncut)).unwrap();
        t.backward_step_input_grad(&d, BackwardOpts::default(), &mut dx)
            .unwrap();
        assert!(dx[..CUT * INPUT_DIM].iter().any(|&x| x != 0.0), "{family}");
    }
}

#[test]
fn packing_contract_checkpoint_restores_the_next_windows() {
    let shape = (2, SEQ_LEN);
    let input = values(2 * SEQ_LEN * INPUT_DIM, 34);
    let d = grad(2 * SEQ_LEN * D_MODEL, 35);
    let warm = values(input.len(), 36);
    let mut ids = vec![0; 2 * SEQ_LEN];
    ids[CUT..SEQ_LEN].fill(i32::MAX);
    ids[SEQ_LEN..].fill(i32::MAX - 1);
    ids[SEQ_LEN + 2..].fill(i32::MAX);
    let packing = Varlen::from_seq_idx(ids, 2, SEQ_LEN).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trainer.safetensors");
    for ((family, mut original), (_, mut restored)) in trainers(shape, WeightDtype::F32)
        .into_iter()
        .zip(trainers(shape, WeightDtype::F32))
    {
        original.set_lr(0.0).unwrap();
        restored.set_lr(0.0).unwrap();
        for next in [Some(&packing), None] {
            original.set_varlen(next).unwrap();
            original.step(&warm, &d).unwrap();
            let saved = original.trainer_state().unwrap();
            assert_eq!(saved.packing.as_ref(), next, "{family}: configured labels");
            saved.save(&path).unwrap();
            let loaded = TrainerState::load(&path).unwrap();
            assert_eq!(saved, loaded, "{family}: the file");
            restored.load_trainer_state(&loaded).unwrap();
            let (y, dx, dw) = run(&mut original, &input, &d);
            let (yr, dxr, dwr) = run(&mut restored, &input, &d);
            for (name, a, b) in [
                ("output", y, yr),
                ("input gradient", dx, dxr),
                ("gradient", dw, dwr),
            ] {
                assert!(bits(&a) == bits(&b), "{family}: resumed {name}");
            }
            original.apply_window().unwrap();
            restored.apply_window().unwrap();
        }
    }
}

#[test]
fn packing_contract_checkpoint_requires_a_closed_accumulation_window() {
    let input = values(SEQ_LEN * INPUT_DIM, 37);
    let d = grad(SEQ_LEN * D_MODEL, 38);
    let mut rows = vec![0.0; d.len()];
    for (family, mut t) in trainers((1, SEQ_LEN), WeightDtype::F32) {
        t.forward(&input, &mut rows).unwrap();
        let checkpoint = t.trainer_state().unwrap();
        t.backward_step(&d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap();
        assert!(
            t.trainer_state().is_err(),
            "{family} checkpointed a window with unapplied gradients"
        );
        assert!(t.apply_window().unwrap().optimizer_stepped);
        t.load_trainer_state(&checkpoint).unwrap();
        assert!(
            t.backward_step(&d, BackwardOpts::default()).is_err(),
            "{family}"
        );
    }
}

#[test]
fn packing_contract_loading_replaces_pending_gradients_and_activations() {
    let input = values(SEQ_LEN * INPUT_DIM, 39);
    let d = grad(SEQ_LEN * D_MODEL, 40);
    let mut rows = vec![0.0; d.len()];
    for (family, mut t) in trainers((1, SEQ_LEN), WeightDtype::F32) {
        let checkpoint = t.trainer_state().unwrap();
        t.accumulate(&input, &d).unwrap();
        t.forward(&input, &mut rows).unwrap();
        t.load_trainer_state(&checkpoint).unwrap();
        assert!(t.apply_window().is_err(), "{family}: stale gradients");
        assert!(
            t.backward_step(&d, BackwardOpts::default()).is_err(),
            "{family}: stale activations"
        );
        assert_eq!(t.step(&input, &d).unwrap().step, 1, "{family}");
    }
}

#[test]
fn packing_contract_failed_load_preserves_the_pending_backward_and_scale() {
    let input = values(SEQ_LEN * INPUT_DIM, 41);
    let d = grad(SEQ_LEN * D_MODEL, 42);
    let mut rows = vec![0.0; d.len()];
    let wrong_shape = Varlen::from_starts(&[vec![0], vec![3]], SEQ_LEN / 2).unwrap();
    for (family, mut t) in trainers((1, SEQ_LEN), WeightDtype::F16) {
        t.forward(&input, &mut rows).unwrap();
        let checkpoint = t.trainer_state().unwrap();
        let mut invalid = checkpoint.clone();
        invalid.scaler = Some((4.0, 17));
        invalid.optimizer.v.pop();
        assert!(t.load_trainer_state(&invalid).is_err(), "{family}");
        assert_eq!(t.scaler_state(), checkpoint.scaler, "{family}");
        let mut invalid = checkpoint.clone();
        invalid.packing = Some(wrong_shape.clone());
        invalid.optimizer.step += 1;
        assert!(t.load_trainer_state(&invalid).is_err(), "{family}");
        assert_eq!(t.trainer_state().unwrap(), checkpoint, "{family}");
        t.backward_step(&d, BackwardOpts::default()).unwrap();
    }
}

#[test]
fn packing_contract_an_optimizer_step_invalidates_an_older_forward() {
    let input = values(SEQ_LEN * INPUT_DIM, 43);
    let d = grad(SEQ_LEN * D_MODEL, 44);
    let mut rows = vec![0.0; d.len()];
    for (family, mut t) in trainers((1, SEQ_LEN), WeightDtype::F32) {
        t.accumulate(&input, &d).unwrap();
        t.forward(&input, &mut rows).unwrap();
        assert!(t.apply_window().unwrap().optimizer_stepped);
        assert!(
            t.backward_step(&d, BackwardOpts::default()).is_err(),
            "{family} paired the new weights with activations of the old weights"
        );
        t.forward(&input, &mut rows).unwrap();
        t.backward_step(&d, BackwardOpts::default()).unwrap();
    }
}

#[test]
fn packing_contract_reads_legacy_files_and_rejects_malformed_ids() {
    use safetensors::{Dtype, SafeTensors, tensor::TensorView};

    let mut t = trainer(0, 2, WeightDtype::F32);
    let packing = Varlen::from_seq_idx(vec![0, i32::MAX], 1, 2).unwrap();
    t.set_varlen(Some(&packing)).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.safetensors");
    t.trainer_state().unwrap().save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    let (_, header) = SafeTensors::read_metadata(&bytes).unwrap();
    let metadata = header.metadata().as_ref().unwrap();
    let packed_format = &metadata["format"];
    let labels = file.tensor("seq_idx").unwrap();
    assert_eq!(labels.dtype(), Dtype::I32);
    assert_eq!(labels.shape(), &[1, 2]);
    let read = |format: &str, ids: Option<TensorView<'_>>| {
        let mut tensors = file.tensors();
        tensors.retain(|(name, _)| name != "seq_idx");
        if let Some(tensor) = ids {
            tensors.push(("seq_idx".into(), tensor));
        }
        let mut m = metadata.clone();
        m.insert("format".into(), format.into());
        TrainerState::load_from_bytes(&safetensors::serialize(tensors, Some(m)).unwrap())
    };
    let legacy = "mamba-rs-trainer-state";
    assert!(read(legacy, None).unwrap().packing.is_none());
    assert!(read(legacy, Some(labels.clone())).is_err());
    assert!(read(packed_format, None).is_err());
    let decreasing = [1i32, 0].map(i32::to_le_bytes);
    let negative = [-1i32, 0].map(i32::to_le_bytes);
    for (dtype, shape, data) in [
        (Dtype::F32, vec![1, 2], labels.data()),
        (Dtype::I32, vec![2], labels.data()),
        (Dtype::I32, vec![1, 2], decreasing.as_flattened()),
        (Dtype::I32, vec![1, 2], negative.as_flattened()),
    ] {
        let malformed = TensorView::new(dtype, shape, data).unwrap();
        assert!(read(packed_format, Some(malformed)).is_err());
    }
}
