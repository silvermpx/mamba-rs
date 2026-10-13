//! The host trainer of every family trains as the GPU trainer does, in
//! each scan mode of the GPU: from the same weights, three steps with the
//! global and the control clip give master weights within f32 rounding of
//! the GPU's in f32, the same step counts, the same clip norms to f32
//! precision, and after every window the same carried state in the same
//! layout, so a trainer state either device saved loads into the other.
//! The steps run once with every window from the empty state and once
//! with the state carried across the windows, which takes the gradient of
//! the terms a window's first step reads from the window before. For
//! Mamba-2, whose host tensor order is the GPU arena's, the optimizer
//! moments agree too.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::{
    AnyCpuTrainer, AnyGpuTrainer, BackwardOpts, CpuTrainer, GpuOptions, GpuTrainer, Mamba2Config,
    Mamba2CpuTrainer, Mamba2GpuTrainer, Mamba2Weights, Mamba3Config, Mamba3CpuTrainer,
    Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoCpuTrainer, Mamba3MimoGpuTrainer,
    Mamba3MimoWeights, Mamba3Weights, MambaConfig, MambaCpuTrainer, MambaGpuTrainer, MambaWeights,
    ModelFamily, TrainSessionCfg, WeightDtype,
};

const BATCH: usize = 2;
const SEQ_LEN: usize = 24;
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
        bcdt_rms_eps: None,
    }
}

/// Mamba-1 with FalconMamba's dt, B and C norms.
fn falcon_cfg() -> MambaConfig {
    MambaConfig {
        bcdt_rms_eps: Some(1e-6),
        ..m1_cfg()
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

/// The largest elementwise difference; a NaN on either side is infinite.
fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| {
        let d = (x - y).abs();
        if d.is_nan() { f32::INFINITY } else { m.max(d) }
    })
}

/// The same family on both devices from the same weights, in f32, the GPU
/// in the scan mode `scan`.
fn pairs(scan: ScanMode) -> Vec<(AnyCpuTrainer, AnyGpuTrainer)> {
    let s = session();
    let opts = GpuOptions::default()
        .dtype(WeightDtype::F32)
        .scan_mode(scan);
    let m1 = MambaWeights::init(&m1_cfg(), INPUT_DIM, 1);
    let m2 = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 2);
    let siso = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 3);
    let mimo = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 4);
    vec![
        (
            MambaCpuTrainer::new(&m1, m1_cfg(), s).unwrap().into(),
            MambaGpuTrainer::new(&m1, m1_cfg(), s, opts).unwrap().into(),
        ),
        (
            Mamba2CpuTrainer::new(&m2, m2_cfg(), s).unwrap().into(),
            Mamba2GpuTrainer::new(&m2, m2_cfg(), s, opts)
                .unwrap()
                .into(),
        ),
        (
            Mamba3CpuTrainer::new(&siso, siso_cfg(), s).unwrap().into(),
            Mamba3GpuTrainer::new(&siso, siso_cfg(), s, opts)
                .unwrap()
                .into(),
        ),
        (
            Mamba3MimoCpuTrainer::new(&mimo, mimo_cfg(), s)
                .unwrap()
                .into(),
            Mamba3MimoGpuTrainer::new(&mimo, mimo_cfg(), s, opts)
                .unwrap()
                .into(),
        ),
    ]
}

/// The master weights of a pair, flat in each family's own order: the
/// same tensors on both sides.
fn masters(cpu: &AnyCpuTrainer, gpu: &AnyGpuTrainer) -> (Vec<f32>, Vec<f32>) {
    fn m2(w: &Mamba2Weights) -> Vec<f32> {
        let mut v = w.input_proj_w.clone();
        v.extend_from_slice(&w.input_proj_b);
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
                v.extend_from_slice(t);
            }
        }
        v.extend_from_slice(&w.norm_f_weight);
        v
    }
    fn mimo(w: &Mamba3MimoWeights) -> Vec<f32> {
        let mut v = w.input_proj_w.clone();
        v.extend_from_slice(&w.input_proj_b);
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
                &l.out_proj_w,
            ] {
                v.extend_from_slice(t);
            }
        }
        v.extend_from_slice(&w.norm_f_weight);
        v
    }
    fn m1(w: &MambaWeights) -> Vec<f32> {
        let mut v = w.input_proj_w.clone();
        v.extend_from_slice(&w.input_proj_b);
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
                v.extend_from_slice(t);
            }
        }
        v.extend_from_slice(&w.norm_f_weight);
        v
    }
    fn siso(w: &Mamba3Weights) -> Vec<f32> {
        let mut v = w.input_proj_w.clone();
        v.extend_from_slice(&w.input_proj_b);
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
                &l.norm_gate_weight,
                &l.out_proj_w,
            ] {
                v.extend_from_slice(t);
            }
        }
        v.extend_from_slice(&w.norm_f_weight);
        v
    }
    match (cpu, gpu) {
        (AnyCpuTrainer::M1(c), AnyGpuTrainer::M1(g)) => {
            (m1(&c.snapshot_master()), m1(&g.snapshot_master().unwrap()))
        }
        (AnyCpuTrainer::M2(c), AnyGpuTrainer::M2(g)) => {
            (m2(&c.snapshot_master()), m2(&g.snapshot_master().unwrap()))
        }
        (AnyCpuTrainer::M3(c), AnyGpuTrainer::M3(g)) => (
            siso(&c.snapshot_master()),
            siso(&g.snapshot_master().unwrap()),
        ),
        (AnyCpuTrainer::M3Mimo(c), AnyGpuTrainer::M3Mimo(g)) => (
            mimo(&c.snapshot_master()),
            mimo(&g.snapshot_master().unwrap()),
        ),
        _ => panic!("a pair of two families"),
    }
}

#[test]
fn every_family_trains_on_the_cpu_as_on_the_gpu() {
    for scan in [ScanMode::Sequential, ScanMode::Parallel] {
        for carry in [false, true] {
            for (cpu, gpu) in pairs(scan) {
                train_alike(cpu, gpu, scan, carry);
            }
        }
    }
}

/// Three steps on both devices, every window from the empty state or the
/// state carried across them, compared after each window and at the end.
fn train_alike(mut cpu: AnyCpuTrainer, mut gpu: AnyGpuTrainer, scan: ScanMode, carry: bool) {
    let opts = BackwardOpts::default()
        .with_clip_max_norm(0.05)
        .with_control_clip_max_norm(0.01);
    let n_out = BATCH * SEQ_LEN * D_MODEL;
    let family = cpu.family();
    assert_eq!(family, gpu.family());
    let case = format!(
        "{family:?} with the {scan:?} scan, {}",
        if carry {
            "the state carried"
        } else {
            "every window from the empty state"
        }
    );
    let (before, _) = masters(&cpu, &gpu);
    for k in 0..3u32 {
        let input = values(BATCH * SEQ_LEN * INPUT_DIM, 10 + k);
        let d: Vec<f32> = values(n_out, 20 + k).iter().map(|v| v * 1e-2).collect();
        let (mut oc, mut og) = (vec![0.0f32; n_out], vec![0.0f32; n_out]);
        if !carry {
            cpu.reset_state().unwrap();
            gpu.reset_state().unwrap();
        }
        cpu.forward(&input, &mut oc).unwrap();
        gpu.forward(&input, &mut og).unwrap();
        let out_diff = worst(&oc, &og);
        assert!(
            out_diff < 1e-4,
            "{case}, step {k}: outputs differ by {out_diff}"
        );
        let (sc, sg) = (cpu.trainer_state().unwrap(), gpu.trainer_state().unwrap());
        assert_eq!(sc.recurrent.len(), sg.recurrent.len(), "{case}");
        for ((name, vc), (name_g, vg)) in sc.recurrent.iter().zip(&sg.recurrent) {
            assert_eq!((name, vc.len()), (name_g, vg.len()), "{case}");
            let scale = vg.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            let diff = worst(vc, vg);
            assert!(
                diff <= 1e-4 * scale,
                "{case}, step {k}: the carried {name} differs by {diff} of {scale}"
            );
        }
        let mc = cpu.backward_step(&d, opts).unwrap();
        let mg = gpu.backward_step(&d, opts).unwrap();
        assert_eq!(mc.step, mg.step, "{case}, step {k}");
        assert_eq!(mc.optimizer_stepped, mg.optimizer_stepped, "{case}");
        let (nc, ng) = (mc.grad_norm.unwrap(), mg.grad_norm.unwrap());
        assert!(
            (nc - ng).abs() <= 1e-4 * ng.abs(),
            "{case}, step {k}: clip norms {nc} and {ng}"
        );
    }
    let (wc, wg) = masters(&cpu, &gpu);
    assert_eq!(wc.len(), wg.len(), "{case}");
    let moved = worst(&before, &wg);
    let diff = worst(&wc, &wg);
    eprintln!("{case}: weights moved {moved:e}, devices differ by {diff:e}");
    assert!(moved > 1e-4, "{case}: the weights did not move");
    assert!(
        diff < 2e-2 * moved,
        "{case}: the devices differ by {diff:e} where training moved {moved:e}"
    );
    if family == ModelFamily::Mamba2 {
        let (sc, sg) = (
            cpu.optimizer_state().unwrap(),
            gpu.optimizer_state().unwrap(),
        );
        assert_eq!((sc.m.len(), sc.step), (sg.m.len(), sg.step));
        let scale = sg.m.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let moments = worst(&sc.m, &sg.m);
        assert!(
            moments <= 1e-3 * scale,
            "{case}: the first moments differ by {moments:e} of {scale:e}"
        );
    }
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

/// A trainer's window: the input and the output gradient in, the rows of
/// the forward out, the global gradient norm of the applied step returned.
type Window<'a> = dyn FnMut(&[f32], &[f32], &mut [f32]) -> f32 + 'a;

/// One window of a pair: the rows of the forward and the global gradient
/// norm of the applied step.
fn window(cpu_or_gpu: &mut Window<'_>) -> (Vec<f32>, f32) {
    let input = values(BATCH * SEQ_LEN * INPUT_DIM, 31);
    let d_out = values(BATCH * SEQ_LEN * D_MODEL, 32);
    let mut rows = vec![0f32; BATCH * SEQ_LEN * D_MODEL];
    let norm = cpu_or_gpu(&input, &d_out, &mut rows);
    (rows, norm)
}

/// The bf16 and f16 GPU trainers of every family against the CPU
/// trainer of the same precision, which rounds the weights and the values
/// between the kernels where the GPU trainer stores them: its rows land
/// closer to the GPU's than the f32 CPU's do. (A TF32 trainer takes TF32
/// only on the GEMMs a board has a measured kernel for, so the CPU's TF32
/// lane, which rounds every product's operands, is not its twin.)
#[test]
fn half_lanes_track_the_cpu_lane() {
    use mamba_rs::CpuOptions;
    let s = session();
    let m1 = MambaWeights::init(&m1_cfg(), INPUT_DIM, 1);
    let falcon = MambaWeights::init(&falcon_cfg(), INPUT_DIM, 5);
    let m2 = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 2);
    let siso = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 3);
    let mimo = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 4);
    let measure = BackwardOpts::default().with_clip_max_norm(1e30);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let gpu_opts = GpuOptions::default().dtype(dtype);
        let lane = CpuOptions::default().dtype(dtype);
        let trio: Vec<(&str, [AnyCpuTrainer; 2], AnyGpuTrainer)> = vec![
            (
                "Mamba-1",
                [
                    MambaCpuTrainer::new(&m1, m1_cfg(), s).unwrap().into(),
                    MambaCpuTrainer::new_with(&m1, m1_cfg(), s, lane)
                        .unwrap()
                        .into(),
                ],
                MambaGpuTrainer::new(&m1, m1_cfg(), s, gpu_opts)
                    .unwrap()
                    .into(),
            ),
            (
                "FalconMamba",
                [
                    MambaCpuTrainer::new(&falcon, falcon_cfg(), s)
                        .unwrap()
                        .into(),
                    MambaCpuTrainer::new_with(&falcon, falcon_cfg(), s, lane)
                        .unwrap()
                        .into(),
                ],
                MambaGpuTrainer::new(&falcon, falcon_cfg(), s, gpu_opts)
                    .unwrap()
                    .into(),
            ),
            (
                "Mamba-2",
                [
                    Mamba2CpuTrainer::new(&m2, m2_cfg(), s).unwrap().into(),
                    Mamba2CpuTrainer::new_with(&m2, m2_cfg(), s, lane)
                        .unwrap()
                        .into(),
                ],
                Mamba2GpuTrainer::new(&m2, m2_cfg(), s, gpu_opts)
                    .unwrap()
                    .into(),
            ),
            (
                "SISO",
                [
                    Mamba3CpuTrainer::new(&siso, siso_cfg(), s).unwrap().into(),
                    Mamba3CpuTrainer::new_with(&siso, siso_cfg(), s, lane)
                        .unwrap()
                        .into(),
                ],
                Mamba3GpuTrainer::new(&siso, siso_cfg(), s, gpu_opts)
                    .unwrap()
                    .into(),
            ),
            (
                "MIMO",
                [
                    Mamba3MimoCpuTrainer::new(&mimo, mimo_cfg(), s)
                        .unwrap()
                        .into(),
                    Mamba3MimoCpuTrainer::new_with(&mimo, mimo_cfg(), s, lane)
                        .unwrap()
                        .into(),
                ],
                Mamba3MimoGpuTrainer::new(&mimo, mimo_cfg(), s, gpu_opts)
                    .unwrap()
                    .into(),
            ),
        ];
        for (label, [mut f32_host, mut lane_host], mut gpu) in trio {
            assert_eq!(lane_host.dtype(), dtype);
            for t in [&mut f32_host, &mut lane_host] {
                t.set_step_opts(measure).unwrap();
                t.load_scaler_state(256.0, 0).unwrap();
            }
            gpu.set_step_opts(measure).unwrap();
            gpu.load_scaler_state(256.0, 0).unwrap();
            let on_cpu = |t: &mut AnyCpuTrainer| {
                window(&mut |x, d, rows| {
                    t.forward(x, rows).unwrap();
                    let m = t
                        .backward_step(d, t.step_opts())
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert!(m.optimizer_stepped, "{label} {dtype:?}: the host skipped");
                    m.grad_norm.unwrap()
                })
            };
            let (f32_rows, f32_norm) = on_cpu(&mut f32_host);
            let (lane_rows, lane_norm) = on_cpu(&mut lane_host);
            let (gpu_rows, gpu_norm) = window(&mut |x, d, rows| {
                gpu.forward(x, rows).unwrap();
                let m = gpu.backward_step(d, gpu.step_opts()).unwrap();
                assert!(m.optimizer_stepped, "{label} {dtype:?}: the GPU skipped");
                m.grad_norm.unwrap()
            });
            let (d_f32, d_lane) = (rel_l2(&gpu_rows, &f32_rows), rel_l2(&gpu_rows, &lane_rows));
            eprintln!(
                "{label} {dtype:?}: rows GPU vs CPU f32 {d_f32:.2e}, vs CPU lane {d_lane:.2e}; \
                 gradient norm GPU {gpu_norm}, CPU f32 {f32_norm}, CPU lane {lane_norm}"
            );
            assert!(
                d_lane < d_f32,
                "{label} {dtype:?}: the CPU lane ({d_lane:.3e}) is no closer than f32 ({d_f32:.3e})"
            );
            assert_eq!(
                lane_host.scaler_state().is_some(),
                dtype == WeightDtype::F16
            );
        }
    }
}
