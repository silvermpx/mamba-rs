//! The host trainer of every family, through the shared `CpuTrainer`
//! surface: a window of two calls steps on twice its gradient, bit for
//! bit; the bits do not depend on the thread pool; the input gradient
//! matches central differences; training lowers a loss; the clip and the
//! spike skip act on the applying call; a window closed by
//! `apply_step_full` gives the bits of an applying backward; and a run
//! resumes bit for bit through the family's checkpoint and the
//! trainer-state file.

use mamba_rs::config::ScanMode;
use mamba_rs::{
    AnyCpuTrainer, BackwardOpts, CpuTrainer, Mamba2Config, Mamba2CpuTrainer, Mamba2Weights,
    Mamba3Config, Mamba3CpuTrainer, Mamba3MimoConfig, Mamba3MimoCpuTrainer, Mamba3MimoWeights,
    Mamba3Weights, MambaConfig, MambaCpuTrainer, MambaWeights, ModelFamily, TrainSessionCfg,
    TrainerState,
};

const BATCH: usize = 3;
const SEQ_LEN: usize = 12;
const D_MODEL: usize = 32;
const INPUT_DIM: usize = 24;

fn session() -> TrainSessionCfg {
    TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ_LEN)
}

fn m1_cfg() -> MambaConfig {
    MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: ScanMode::Auto,
        rms_norm_eps: 1e-5,
    }
}

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
        d_intermediate: 48,
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn window(seed: u32) -> (Vec<f32>, Vec<f32>) {
    (
        values(BATCH * SEQ_LEN * INPUT_DIM, seed),
        values(BATCH * SEQ_LEN * D_MODEL, seed + 1000)
            .into_iter()
            .map(|v| v * 1e-2)
            .collect(),
    )
}

/// One trainer of each family, the same weights every call.
fn trainers(s: TrainSessionCfg) -> Vec<AnyCpuTrainer> {
    vec![
        MambaCpuTrainer::new(&MambaWeights::init(&m1_cfg(), INPUT_DIM, 1), m1_cfg(), s)
            .unwrap()
            .into(),
        Mamba2CpuTrainer::new(&Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 2), m2_cfg(), s)
            .unwrap()
            .into(),
        Mamba3CpuTrainer::new(
            &Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 3),
            siso_cfg(),
            s,
        )
        .unwrap()
        .into(),
        Mamba3MimoCpuTrainer::new(
            &Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 4),
            mimo_cfg(),
            s,
        )
        .unwrap()
        .into(),
    ]
}

/// The model's response to a fixed window from a zero state, after the
/// state is put back: a fingerprint of the weights.
fn response(t: &mut AnyCpuTrainer) -> Vec<u32> {
    let saved = t.trainer_state().unwrap();
    t.reset_state().unwrap();
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    t.forward(&window(77).0, &mut out).unwrap();
    // The forward's tape is dropped by the next forward; the recurrence
    // goes back as it was.
    t.load_trainer_state(&saved).unwrap();
    bits(&out)
}

/// A window of two calls on the same sequence steps on twice its
/// gradient: in f32 the window gives the bits of one call with the
/// gradient doubled.
#[test]
fn a_window_of_two_is_one_doubled_call_in_every_family() {
    let (input, d) = window(1);
    let d2: Vec<f32> = d.iter().map(|v| 2.0 * v).collect();
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    for (mut win, mut doubled) in trainers(session()).into_iter().zip(trainers(session())) {
        let family = win.family();
        win.forward(&input, &mut out).unwrap();
        let a = win
            .backward_step(&d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap();
        assert!(!a.optimizer_stepped, "{family:?}");
        win.reset_state().unwrap();
        win.forward(&input, &mut out).unwrap();
        let m = win.backward_step(&d, BackwardOpts::default()).unwrap();
        assert!(m.optimizer_stepped && m.step == 1, "{family:?}: {m:?}");

        doubled.forward(&input, &mut out).unwrap();
        doubled.backward_step(&d2, BackwardOpts::default()).unwrap();
        assert_eq!(
            win.optimizer_state().unwrap(),
            doubled.optimizer_state().unwrap(),
            "{family:?}: the moments"
        );
        assert_eq!(response(&mut win), response(&mut doubled), "{family:?}");
    }
}

/// Three steps on one thread and on four give the same bits: the batch's
/// gradient sums in chunks fixed by the batch alone.
#[test]
fn the_bits_do_not_depend_on_the_thread_pool() {
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                trainers(session())
                    .into_iter()
                    .map(|mut t| {
                        for k in 0..3 {
                            let (input, d) = window(10 + k);
                            t.step(&input, &d).unwrap();
                        }
                        (t.family(), response(&mut t), t.optimizer_state().unwrap())
                    })
                    .collect::<Vec<_>>()
            })
    };
    let (one, four) = (run(1), run(4));
    for (a, b) in one.iter().zip(&four) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1, "{:?}: the response", a.0);
        assert_eq!(a.2, b.2, "{:?}: the moments", a.0);
    }
}

/// The input gradient matches central differences of `sum(out * d)` over
/// the window's input, each forward from a zero state; the accumulating
/// backward leaves the weights alone.
#[test]
fn every_family_returns_the_input_gradient() {
    let (input, d) = window(3);
    let n_out = BATCH * SEQ_LEN * D_MODEL;
    let loss = |t: &mut AnyCpuTrainer, x: &[f32]| -> f64 {
        let mut out = vec![0.0f32; n_out];
        t.reset_state().unwrap();
        t.forward(x, &mut out).unwrap();
        out.iter()
            .zip(&d)
            .map(|(o, g)| f64::from(*o) * f64::from(*g))
            .sum()
    };
    for mut t in trainers(session()) {
        let family = t.family();
        let mut out = vec![0.0f32; n_out];
        t.reset_state().unwrap();
        t.forward(&input, &mut out).unwrap();
        let mut d_input = vec![0.0f32; input.len()];
        t.backward_step_input_grad(
            &d,
            BackwardOpts::default().with_accumulate_only(true),
            &mut d_input,
        )
        .unwrap();
        let scale = d_input.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "{family:?}: an empty input gradient");
        for &i in &[0, 7, INPUT_DIM + 3, input.len() / 2, input.len() - 1] {
            let h = 1e-2f32;
            let (mut plus, mut minus) = (input.clone(), input.clone());
            plus[i] += h;
            minus[i] -= h;
            let fd = (loss(&mut t, &plus) - loss(&mut t, &minus)) / f64::from(2.0 * h);
            let got = f64::from(d_input[i]);
            assert!(
                (fd - got).abs() <= 2e-3 + 5e-2 * got.abs(),
                "{family:?} input {i}: backward {got:e}, central difference {fd:e}"
            );
        }
    }
}

/// Fifty steps toward a fixed target lower the squared error in every
/// family.
#[test]
fn training_lowers_the_loss_in_every_family() {
    let s = TrainSessionCfg {
        lr: 3e-3,
        ..session()
    };
    let input = values(BATCH * SEQ_LEN * INPUT_DIM, 5);
    let target: Vec<f32> = values(BATCH * SEQ_LEN * D_MODEL, 6)
        .into_iter()
        .map(|v| 0.5 * v)
        .collect();
    let mut out = vec![0.0f32; target.len()];
    for mut t in trainers(s) {
        let family = t.family();
        let mut losses = Vec::new();
        for _ in 0..50 {
            t.reset_state().unwrap();
            t.forward(&input, &mut out).unwrap();
            let d: Vec<f32> = out
                .iter()
                .zip(&target)
                .map(|(o, y)| 2.0 * (o - y) / out.len() as f32)
                .collect();
            losses.push(
                out.iter()
                    .zip(&target)
                    .map(|(o, y)| (o - y) * (o - y))
                    .sum::<f32>()
                    / out.len() as f32,
            );
            t.backward_step(&d, BackwardOpts::default()).unwrap();
        }
        let (first, last) = (losses[0], losses[losses.len() - 1]);
        assert!(
            last < 0.7 * first,
            "{family:?}: the loss went from {first} to {last}"
        );
    }
}

/// The clip reports the gradient's norm and leaves a gradient under the
/// limit alone; a window whose norm passes the spike threshold takes no
/// step; a clip on an accumulating call is an error.
#[test]
fn clip_and_spike_skip_act_on_the_applying_call() {
    let (input, d) = window(8);
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    for ((mut plain, mut clipped), mut skipped) in trainers(session())
        .into_iter()
        .zip(trainers(session()))
        .zip(trainers(session()))
    {
        let family = plain.family();
        plain.forward(&input, &mut out).unwrap();
        plain.backward_step(&d, BackwardOpts::default()).unwrap();

        clipped.forward(&input, &mut out).unwrap();
        let m = clipped
            .backward_step(&d, BackwardOpts::default().with_clip_max_norm(1e9))
            .unwrap();
        let norm = m.grad_norm.expect("a clip reports the norm");
        assert!(norm > 0.0 && m.optimizer_stepped, "{family:?}: {m:?}");
        assert_eq!(
            response(&mut plain),
            response(&mut clipped),
            "{family:?}: a clip above the norm changes nothing"
        );

        skipped.forward(&input, &mut out).unwrap();
        let m = skipped
            .backward_step(&d, BackwardOpts::default().with_step_skip_above(norm * 0.5))
            .unwrap();
        assert!(!m.optimizer_stepped && m.step == 0, "{family:?}: {m:?}");
        assert_eq!(m.grad_norm, Some(norm), "{family:?}");

        skipped.forward(&input, &mut out).unwrap();
        let refused = skipped.backward_step(
            &d,
            BackwardOpts::default()
                .with_accumulate_only(true)
                .with_clip_max_norm(1.0),
        );
        assert!(
            refused.is_err(),
            "{family:?}: a clip on an accumulating call"
        );
    }
}

/// A window whose last micro-batch only accumulates and whose tail runs
/// alone through `apply_step_full` steps on the bits an applying
/// backward gives; its arena reads out and uploads back unchanged.
#[test]
fn apply_step_full_closes_a_window_as_an_applying_backward_does() {
    let (input, d) = window(9);
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    for (mut applying, mut split) in trainers(session()).into_iter().zip(trainers(session())) {
        let family = applying.family();
        applying.forward(&input, &mut out).unwrap();
        applying
            .backward_step(&d, BackwardOpts::default().with_clip_max_norm(0.1))
            .unwrap();

        split.forward(&input, &mut out).unwrap();
        split
            .backward_step(&d, BackwardOpts::default().with_accumulate_only(true))
            .unwrap();
        let arena = split.download_grad_arena().unwrap();
        split.upload_grad_arena(&arena).unwrap();
        let m = split.apply_step_full(Some(0.1), None, None).unwrap();
        assert!(m.optimizer_stepped && m.step == 1, "{family:?}: {m:?}");
        assert!(
            split.download_grad_arena().is_err(),
            "{family:?}: the window closed"
        );
        assert_eq!(response(&mut applying), response(&mut split), "{family:?}");
    }
}

/// Every family resumes bit for bit through files: the weights through
/// the family's checkpoint, the optimizer and the carried recurrence
/// through the trainer-state file. Two steps, both files written, a fresh
/// trainer built from them, two more steps: the bits of four unbroken
/// steps. A state of another family is refused.
/// The fused calls carry the step options and the window as the split
/// calls do: `step` with options set gives the bits of a forward and an
/// applying backward with them, and `accumulate` twice then `apply_window`
/// the bits of two accumulating backwards and the options' tail.
#[test]
fn the_fused_calls_take_the_step_options_and_the_window_in_every_family() {
    let opts = BackwardOpts::default()
        .with_clip_max_norm(0.05)
        .with_control_clip_max_norm(0.01);
    let acc = BackwardOpts::default().with_accumulate_only(true);
    let mut out = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
    for (mut fused, mut split) in trainers(session()).into_iter().zip(trainers(session())) {
        let family = fused.family();
        assert!(
            fused.set_step_opts(acc).is_err(),
            "{family:?}: accumulation as a step option"
        );
        assert!(fused.apply_window().is_err(), "{family:?}: no window");
        fused.set_step_opts(opts).unwrap();
        assert_eq!(fused.step_opts(), opts, "{family:?}");
        let (x, d) = window(70);
        let m = fused.step(&x, &d).unwrap();
        assert!(
            m.optimizer_stepped && m.grad_norm.is_some(),
            "{family:?}: {m:?}"
        );
        split.forward(&x, &mut out).unwrap();
        split.backward_step(&d, opts).unwrap();
        for k in 0..2 {
            let (x, d) = window(71 + k);
            let m = fused.accumulate(&x, &d).unwrap();
            assert!(!m.optimizer_stepped, "{family:?}: {m:?}");
            split.forward(&x, &mut out).unwrap();
            split.backward_step(&d, acc).unwrap();
        }
        let m = fused.apply_window().unwrap();
        assert!(
            m.optimizer_stepped && m.grad_norm.is_some() && m.step == 2,
            "{family:?}: {m:?}"
        );
        split
            .apply_step_full(
                opts.clip_max_norm,
                opts.step_skip_above,
                opts.control_clip_max_norm,
            )
            .unwrap();
        assert_eq!(response(&mut fused), response(&mut split), "{family:?}");
    }
}

#[test]
fn every_family_resumes_through_its_checkpoint_and_trainer_state_files() {
    let dir = std::env::temp_dir();
    let s = session();
    for (mut a, mut b) in trainers(s).into_iter().zip(trainers(s)) {
        let family = a.family();
        for k in 0..4 {
            let (input, d) = window(20 + k);
            a.step(&input, &d).unwrap();
        }
        for k in 0..2 {
            let (input, d) = window(20 + k);
            b.step(&input, &d).unwrap();
        }
        let name = format!("mamba_rs_cpu_resume_{}_{family:?}", std::process::id());
        let (wpath, spath) = (
            dir.join(format!("{name}_weights.safetensors")),
            dir.join(format!("{name}_state.safetensors")),
        );
        b.trainer_state().unwrap().save(&spath).unwrap();
        let mut c: AnyCpuTrainer = match &b {
            AnyCpuTrainer::M1(t) => {
                mamba_rs::mamba1::serialize::save(
                    &wpath,
                    &t.snapshot_master(),
                    &m1_cfg(),
                    INPUT_DIM,
                )
                .unwrap();
                let (w, cfg, _) = mamba_rs::mamba1::serialize::load(&wpath).unwrap();
                MambaCpuTrainer::new(&w, cfg, s).unwrap().into()
            }
            AnyCpuTrainer::M2(t) => {
                mamba_rs::mamba2::serialize::save(
                    &wpath,
                    &t.snapshot_master(),
                    &m2_cfg(),
                    INPUT_DIM,
                )
                .unwrap();
                let (w, cfg, _) = mamba_rs::mamba2::serialize::load(&wpath).unwrap();
                Mamba2CpuTrainer::new(&w, cfg, s).unwrap().into()
            }
            AnyCpuTrainer::M3(t) => {
                mamba_rs::mamba3_siso::serialize::save(
                    &wpath,
                    &t.snapshot_master(),
                    &siso_cfg(),
                    INPUT_DIM,
                )
                .unwrap();
                let (w, cfg, _) = mamba_rs::mamba3_siso::serialize::load(&wpath).unwrap();
                Mamba3CpuTrainer::new(&w, cfg, s).unwrap().into()
            }
            AnyCpuTrainer::M3Mimo(t) => {
                mamba_rs::mamba3_mimo::serialize::save(
                    &wpath,
                    &t.snapshot_master(),
                    &mimo_cfg(),
                    INPUT_DIM,
                )
                .unwrap();
                let (w, cfg, _) = mamba_rs::mamba3_mimo::serialize::load(&wpath).unwrap();
                Mamba3MimoCpuTrainer::new(&w, cfg, s).unwrap().into()
            }
        };
        let state = TrainerState::load(&spath);
        std::fs::remove_file(&wpath).ok();
        std::fs::remove_file(&spath).ok();
        let state = state.unwrap();
        assert_eq!(state, b.trainer_state().unwrap(), "{family:?}: the file");

        let mut foreign = state.clone();
        foreign.family = if family == ModelFamily::Mamba1 {
            ModelFamily::Mamba2
        } else {
            ModelFamily::Mamba1
        };
        assert!(
            c.load_trainer_state(&foreign).is_err(),
            "{family:?}: another family's state"
        );
        c.load_trainer_state(&state).unwrap();
        for k in 2..4 {
            let (input, d) = window(20 + k);
            c.step(&input, &d).unwrap();
        }
        assert_eq!(
            c.trainer_state().unwrap(),
            a.trainer_state().unwrap(),
            "{family:?}: the optimizer and recurrence"
        );
        assert_eq!(
            response(&mut c),
            response(&mut a),
            "{family:?}: the weights"
        );
    }
}

/// The reduced-precision host trainers of every family: an f16 window
/// that overflows is skipped whole and the loss scale backs off, the
/// masters untouched, and the next clean window steps; the scale travels
/// in the trainer state. bf16 needs no scaler and steps at once. Each lane
/// runs the same window twice to the same bits.
#[test]
fn half_lanes_scale_skip_and_step() {
    use mamba_rs::{CpuOptions, WeightDtype};
    let s = session();
    let m1 = MambaWeights::init(&m1_cfg(), INPUT_DIM, 1);
    let m2 = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 2);
    let siso = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 3);
    let mimo = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 4);
    let lanes = |dtype| -> Vec<AnyCpuTrainer> {
        let o = CpuOptions::default().dtype(dtype);
        vec![
            MambaCpuTrainer::new_with(&m1, m1_cfg(), s, o)
                .unwrap()
                .into(),
            Mamba3CpuTrainer::new_with(&siso, siso_cfg(), s, o)
                .unwrap()
                .into(),
            Mamba2CpuTrainer::new_with(&m2, m2_cfg(), s, o)
                .unwrap()
                .into(),
            Mamba3MimoCpuTrainer::new_with(&mimo, mimo_cfg(), s, o)
                .unwrap()
                .into(),
        ]
    };
    let input = values(BATCH * SEQ_LEN * INPUT_DIM, 41);
    let d_out = values(BATCH * SEQ_LEN * D_MODEL, 42);
    let mut rows = vec![0f32; BATCH * SEQ_LEN * D_MODEL];
    for mut t in lanes(WeightDtype::F16) {
        assert_eq!(t.dtype(), WeightDtype::F16);
        t.load_scaler_state(1024.0, 0).unwrap();
        let before = t.trainer_state().unwrap();
        let mut blown = d_out.clone();
        blown[0] = f32::MAX;
        t.forward(&input, &mut rows).unwrap();
        let m = t.backward_step(&blown, BackwardOpts::default()).unwrap();
        assert_eq!(
            (m.optimizer_stepped, m.overflow_skipped, m.loss_scale),
            (false, Some(true), Some(1024.0))
        );
        assert_eq!(t.scaler_state(), Some((512.0, 0)));
        assert_eq!(
            t.optimizer_state().unwrap().step,
            before.optimizer.step,
            "a skipped window moved the optimizer"
        );
        t.forward(&input, &mut rows).unwrap();
        let m = t.backward_step(&d_out, BackwardOpts::default()).unwrap();
        assert_eq!(
            (m.optimizer_stepped, m.overflow_skipped),
            (true, Some(false))
        );
        let state = t.trainer_state().unwrap();
        assert_eq!(state.scaler, Some((512.0, 1)));
    }
    for dtype in [WeightDtype::Bf16, WeightDtype::F16, WeightDtype::Tf32] {
        let mut a = lanes(dtype);
        let mut b = lanes(dtype);
        for (x, y) in a.iter_mut().zip(b.iter_mut()) {
            x.load_scaler_state(256.0, 0).unwrap();
            y.load_scaler_state(256.0, 0).unwrap();
            assert_eq!(x.scaler_state().is_some(), dtype == WeightDtype::F16);
            let (mut rx, mut ry) = (rows.clone(), rows.clone());
            x.forward(&input, &mut rx).unwrap();
            y.forward(&input, &mut ry).unwrap();
            assert_eq!(rx, ry, "{dtype:?}: the forward is not repeatable");
            let mx = x.backward_step(&d_out, BackwardOpts::default()).unwrap();
            let my = y.backward_step(&d_out, BackwardOpts::default()).unwrap();
            assert!(mx.optimizer_stepped && my.optimizer_stepped, "{dtype:?}");
            assert_eq!(
                x.optimizer_state().unwrap(),
                y.optimizer_state().unwrap(),
                "{dtype:?}: the step is not repeatable"
            );
        }
    }
}
