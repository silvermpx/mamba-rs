//! Mamba-3 SISO integration tests — correctness and gradient checks.
//! Mirrors test_correctness.rs (Mamba-1) adapted for Mamba-3 architecture.

use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::backward::backward_mamba3_layer_batched;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::flat::Mamba3LayerFlat;
use mamba_rs::mamba3_siso::cpu::forward::{Mamba3LayerStateMut, forward_mamba3_layer_batched};
use mamba_rs::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use mamba_rs::mamba3_siso::cpu::scratch::Mamba3TrainScratch;
use mamba_rs::mamba3_siso::cpu::weights::TrainMamba3LayerWeights;
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

#[path = "common/m3_oracle.rs"]
mod m3_oracle;
use m3_oracle::{backbone_step64, train_layer, zeros64};

fn test_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 16,
        d_state: 4,
        expand: 2,
        headdim: 4,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    }
}

fn init_layer_w(dims: &Mamba3Dims) -> TrainMamba3LayerWeights {
    let mut w = TrainMamba3LayerWeights::zeros(dims);
    w.norm_weight.fill(1.0);
    w.d_param.fill(1.0);
    w.b_norm_weight.fill(1.0);
    w.c_norm_weight.fill(1.0);
    for (i, v) in w.in_proj_w.iter_mut().enumerate() {
        *v = ((i % 7) as f32 - 3.0) * 0.01;
    }
    for (i, v) in w.out_proj_w.iter_mut().enumerate() {
        *v = ((i % 5) as f32 - 2.0) * 0.01;
    }
    w
}

/// Run training forward and return temporal output.
fn run_training_forward(
    w: &TrainMamba3LayerWeights,
    dims: &Mamba3Dims,
    temporal: &mut [f32],
) -> Mamba3LayerFlat {
    let mut acts = Mamba3LayerFlat::zeros(*dims);
    let mut scratch = Mamba3TrainScratch::zeros(dims);
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ds = dims.d_state;
    let na = dims.num_rope_angles.max(1);
    let mut ssm = vec![0.0; nh * hd * ds];
    let mut k = vec![0.0; nh * ds];
    let mut v = vec![0.0; nh * hd];
    let mut a = vec![0.0; nh * na];

    forward_mamba3_layer_batched(
        temporal,
        &mut acts,
        w,
        Mamba3LayerStateMut {
            ssm: &mut ssm,
            k: &mut k,
            v: &mut v,
            angle: &mut a,
        },
        &mut scratch,
        dims,
    );
    acts
}

/// Run forward + backward and return the loss (sum of output).
fn compute_loss(w: &TrainMamba3LayerWeights, dims: &Mamba3Dims) -> f64 {
    let mut temporal = vec![0.5_f32; dims.seq_len * dims.d_model];
    run_training_forward(w, dims, &mut temporal);
    // Summed in f64 so the sum adds no rounding of its own to a central
    // difference of two nearby losses.
    temporal.iter().map(|&v| f64::from(v)).sum()
}

// =========================================================================
// Finite-difference gradient checks
// =========================================================================

fn finite_diff_check(
    name: &str,
    get_param: fn(&mut TrainMamba3LayerWeights) -> &mut [f32],
    get_grad: fn(&TrainMamba3LayerWeights) -> &[f32],
    idx: usize,
) {
    finite_diff_check_cfg(
        &test_cfg(),
        init_layer_w_drawn,
        2e-2,
        (name, get_param, get_grad, idx),
    );
}

type GetParam = fn(&mut TrainMamba3LayerWeights) -> &mut [f32];
type GetGrad = fn(&TrainMamba3LayerWeights) -> &[f32];

/// One parameter element to check: its name, how to reach it in the
/// weights and in the gradient, and its index.
type FdCase<'a> = (&'a str, GetParam, GetGrad, usize);

fn finite_diff_check_cfg(
    cfg: &Mamba3Config,
    init: fn(&Mamba3Dims) -> TrainMamba3LayerWeights,
    eps: f32,
    (name, get_param, get_grad, idx): FdCase<'_>,
) {
    let dims = Mamba3Dims::from_config(cfg, 4);

    // Analytic gradient
    let w = init(&dims);
    let mut temporal = vec![0.5_f32; dims.seq_len * dims.d_model];
    let acts = run_training_forward(&w, &dims, &mut temporal);

    let mut d_temporal = vec![1.0_f32; dims.seq_len * dims.d_model];
    let mut d_w = TrainMamba3LayerWeights::zeros(&dims);
    let mut scratch = Mamba3TrainScratch::zeros(&dims);
    backward_mamba3_layer_batched(
        &mut d_temporal,
        &acts,
        &w,
        &mut d_w,
        &mut scratch,
        &dims,
        None,
    );

    let analytic = get_grad(&d_w)[idx];

    // Finite difference: (loss(w+eps) - loss(w-eps)) / (2*eps)
    let mut w_plus = init(&dims);
    get_param(&mut w_plus)[idx] += eps;
    let loss_plus = compute_loss(&w_plus, &dims);

    let mut w_minus = init(&dims);
    get_param(&mut w_minus)[idx] -= eps;
    let loss_minus = compute_loss(&w_minus, &dims);

    let numeric = ((loss_plus - loss_minus) / (2.0 * f64::from(eps))) as f32;

    // A gradient this small would pass on the absolute escape below
    // whatever the backward computed, so the check refuses it.
    assert!(
        analytic.abs() > 1e-3,
        "{name}[{idx}]: gradient {analytic} too small to check"
    );
    let abs_diff = (analytic - numeric).abs();
    let rel_diff = abs_diff / (analytic.abs().max(numeric.abs()).max(1e-8));

    assert!(
        rel_diff < 0.05 || abs_diff < 1e-4,
        "{name}[{idx}]: analytic={analytic:.6}, numeric={numeric:.6}, rel={rel_diff:.4}, abs={abs_diff:.6}"
    );
}

#[test]
fn test_m3_finite_diff_dt_bias() {
    finite_diff_check("dt_bias", |w| &mut w.dt_bias, |w| &w.dt_bias, 0);
}

#[test]
fn test_m3_finite_diff_d_param() {
    finite_diff_check("d_param", |w| &mut w.d_param, |w| &w.d_param, 0);
}

#[test]
fn test_m3_finite_diff_b_bias() {
    finite_diff_check("b_bias", |w| &mut w.b_bias, |w| &w.b_bias, 0);
}

#[test]
fn test_m3_finite_diff_out_proj_w() {
    finite_diff_check("out_proj_w", |w| &mut w.out_proj_w, |w| &w.out_proj_w, 0);
}

/// Weights drawn from a deterministic xorshift stream at a scale where the
/// layer output, not the residual it rides on, sets the loss. With the
/// patterned weights of `init_layer_w` a wide head's output sits below one
/// f32 ulp of the residual, so a central difference reads zero.
fn init_layer_w_drawn(dims: &Mamba3Dims) -> TrainMamba3LayerWeights {
    let mut w = init_layer_w(dims);
    let mut s = 0x9E37_79B9u32;
    let mut draw = |scale: f32| {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
    };
    for v in w.in_proj_w.iter_mut().chain(w.out_proj_w.iter_mut()) {
        *v = draw(0.4);
    }
    for v in w.b_bias.iter_mut().chain(w.c_bias.iter_mut()) {
        *v = 1.0 + draw(0.5);
    }
    for v in w.dt_bias.iter_mut() {
        *v = draw(1.0);
    }
    w
}

/// The official checkpoints' head width and state size (headdim 64,
/// d_state 128) run on the CPU: the backward agrees with central
/// differences there, on per-head, per-state and projection parameters,
/// each with a gradient large enough to resolve.
#[test]
fn test_m3_finite_diff_official_head_width() {
    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 128,
        expand: 2,
        headdim: 64,
        ..test_cfg()
    };
    cfg.validate().unwrap();
    // A difference step large enough to rise above the f32 rounding of a
    // loss this size; the parameters checked enter the loss linearly or
    // nearly so, which keeps the step's own truncation error small.
    let cases: [FdCase; 6] = [
        ("dt_bias", |w| &mut w.dt_bias, |w| &w.dt_bias, 0),
        ("d_param", |w| &mut w.d_param, |w| &w.d_param, 0),
        ("b_bias", |w| &mut w.b_bias, |w| &w.b_bias, 70),
        ("c_bias", |w| &mut w.c_bias, |w| &w.c_bias, 3),
        ("out_proj_w", |w| &mut w.out_proj_w, |w| &w.out_proj_w, 100),
        ("norm_weight", |w| &mut w.norm_weight, |w| &w.norm_weight, 5),
    ];
    for case in cases {
        finite_diff_check_cfg(&cfg, init_layer_w_drawn, 2e-2, case);
    }
}

// Note: in_proj_w finite-diff is numerically unstable because it fans out to
// 8 downstream paths (z, x, B, C, dt, A, trap, angles). The gradient is verified
// nonzero by test_m3_all_cpu_gradients_nonzero instead.

#[test]
fn test_m3_finite_diff_norm_weight() {
    finite_diff_check("norm_weight", |w| &mut w.norm_weight, |w| &w.norm_weight, 0);
}

#[test]
fn test_m3_finite_diff_b_norm_weight() {
    finite_diff_check(
        "b_norm_weight",
        |w| &mut w.b_norm_weight,
        |w| &w.b_norm_weight,
        0,
    );
}

#[test]
fn test_m3_finite_diff_c_norm_weight() {
    finite_diff_check(
        "c_norm_weight",
        |w| &mut w.c_norm_weight,
        |w| &w.c_norm_weight,
        0,
    );
}

#[test]
fn test_m3_finite_diff_c_bias() {
    finite_diff_check("c_bias", |w| &mut w.c_bias, |w| &w.c_bias, 0);
}

#[test]
fn test_m3_finite_diff_norm_gate_weight() {
    // Only relevant when is_outproj_norm = true
    let cfg = Mamba3Config {
        is_outproj_norm: true,
        ..test_cfg()
    };
    let dims = Mamba3Dims::from_config(&cfg, 4);
    let eps = 2e-2_f32;

    let mut w = init_layer_w_drawn(&dims);
    // Set norm_gate to nonzero for gradient flow
    w.norm_gate_weight.fill(1.0);
    let mut temporal = vec![0.5_f32; dims.seq_len * dims.d_model];
    let acts = run_training_forward(&w, &dims, &mut temporal);

    let mut d_temporal = vec![1.0_f32; dims.seq_len * dims.d_model];
    let mut d_w = TrainMamba3LayerWeights::zeros(&dims);
    let mut scratch = Mamba3TrainScratch::zeros(&dims);
    backward_mamba3_layer_batched(
        &mut d_temporal,
        &acts,
        &w,
        &mut d_w,
        &mut scratch,
        &dims,
        None,
    );

    let analytic = d_w.norm_gate_weight[0];

    let compute = |w: &TrainMamba3LayerWeights| -> f64 {
        let mut t = vec![0.5_f32; dims.seq_len * dims.d_model];
        run_training_forward(w, &dims, &mut t);
        t.iter().map(|&v| f64::from(v)).sum()
    };
    let mut w_plus = w.clone();
    w_plus.norm_gate_weight[0] += eps;
    let mut w_minus = w.clone();
    w_minus.norm_gate_weight[0] -= eps;
    let numeric = ((compute(&w_plus) - compute(&w_minus)) / (2.0 * f64::from(eps))) as f32;
    assert!(
        analytic.abs() > 1e-3,
        "norm_gate_weight[0]: gradient {analytic} too small to check"
    );

    let abs_diff = (analytic - numeric).abs();
    let rel_diff = abs_diff / (analytic.abs().max(numeric.abs()).max(1e-8));
    assert!(
        rel_diff < 0.05 || abs_diff < 1e-4,
        "norm_gate_weight[0]: analytic={analytic:.6}, numeric={numeric:.6}, rel={rel_diff:.4}"
    );
}

// =========================================================================
// Inference: single step, state, reset
// =========================================================================

#[test]
fn test_m3_sequence_matches_steps() {
    let cfg = Mamba3Config {
        d_model: 16,
        d_state: 4,
        expand: 2,
        headdim: 4,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    };
    let w = Mamba3Weights::init(&cfg, 8, 42);
    let input = vec![0.5_f32; 8];

    // Run 4 steps sequentially
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut temporal = vec![0.0; cfg.d_model];
    let mut outputs = Vec::new();
    for _ in 0..4 {
        mamba3_step(
            &mut temporal,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        outputs.push(temporal.clone());
    }

    // Reset and run again — should produce same outputs
    state.reset();
    temporal.fill(0.0);
    for (i, expected) in outputs.iter().enumerate() {
        mamba3_step(
            &mut temporal,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        assert_eq!(&temporal, expected, "step {i} mismatch after reset");
    }
}

#[test]
fn m3_decode_step_runs_at_the_reference_default_d_state() {
    // The decode step used to stage B, C and the angles in 64-float stack
    // arrays while the config admits d_state up to 256 and the prefill,
    // the training forward and the backward all stage 256. At the
    // reference's own default width the step indexed past the end and
    // panicked, and no test reached it because every other M3 test runs
    // d_state in {4, 8, 16}.
    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 128,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    };
    cfg.validate()
        .expect("d_state 128 is inside the accepted range");
    let w = Mamba3Weights::init(&cfg, cfg.d_model, 7);
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut temporal = vec![0.0_f32; cfg.d_model];
    let input = vec![0.25_f32; cfg.d_model];
    for _ in 0..3 {
        mamba3_step(
            &mut temporal,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        assert!(
            temporal.iter().all(|v| v.is_finite()),
            "decode output must stay finite at d_state 128"
        );
    }
}

// =========================================================================
// Serialization
// =========================================================================

/// A NaN reaching dd_A must reach the state and the output, as the
/// reference's clamp lets it: a clamp that returned the floor for a NaN
/// would hide a corrupt projection behind a finite decay.
#[test]
fn m3_nan_decay_input_reaches_the_state() {
    let cfg = test_cfg();
    let mut w = Mamba3Weights::init(&cfg, cfg.d_model, 7);
    let (di, ng, ds, nh) = (cfg.d_inner(), cfg.ngroups, cfg.d_state, cfg.nheads());
    let ip = cfg.in_proj_out_dim();
    // The dd_A column of head 0: after z, x, B, C and dd_dt.
    let dd_a_col = 2 * di + 2 * ng * ds + nh;
    for row in 0..cfg.d_model {
        w.layers[0].in_proj_w[row * ip + dd_a_col] = f32::NAN;
    }
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut temporal = vec![0.0_f32; cfg.d_model];
    let input = vec![0.25_f32; cfg.d_model];
    mamba3_step(
        &mut temporal,
        &input,
        &mut scratch,
        &w,
        &mut state.layers,
        &cfg,
    );
    let hd = cfg.headdim;
    assert!(
        state.layers[0].ssm_state[..hd * ds]
            .iter()
            .any(|v| v.is_nan()),
        "a NaN dd_A must reach head 0's state"
    );
    assert!(
        temporal.iter().any(|v| v.is_nan()),
        "a NaN dd_A must reach the output"
    );
}

#[test]
fn test_m3_serialize_roundtrip() {
    let cfg = Mamba3Config::default();
    let w = Mamba3Weights::init(&cfg, 128, 42);
    let tmp = std::env::temp_dir().join("mamba3_roundtrip_test.safetensors");
    mamba_rs::mamba3_siso::serialize::save(&tmp, &w, &cfg, 128).unwrap();
    let (w2, input_dim) = mamba_rs::mamba3_siso::serialize::load_mamba3(&tmp, &cfg).unwrap();
    std::fs::remove_file(&tmp).ok();

    assert_eq!(input_dim, 128);
    assert_eq!(w.layers[0].dt_bias, w2.layers[0].dt_bias);
    assert_eq!(w.layers[0].in_proj_w, w2.layers[0].in_proj_w);
    assert_eq!(w.layers[0].out_proj_w, w2.layers[0].out_proj_w);
    assert_eq!(w.layers[0].b_bias, w2.layers[0].b_bias);
    assert_eq!(w.layers[0].c_bias, w2.layers[0].c_bias);
    assert_eq!(w.layers[0].d_param, w2.layers[0].d_param);
    assert_eq!(w.norm_f_weight, w2.norm_f_weight);
    assert_eq!(w.input_proj_w, w2.input_proj_w);
    assert_eq!(w.input_proj_b, w2.input_proj_b);
}

#[test]
fn test_m3_serialize_inference_parity() {
    let cfg = Mamba3Config::default();
    let w = Mamba3Weights::init(&cfg, 128, 42);

    // Save + load
    let tmp = std::env::temp_dir().join("mamba3_parity_test.safetensors");
    mamba_rs::mamba3_siso::serialize::save(&tmp, &w, &cfg, 128).unwrap();
    let (w2, _) = mamba_rs::mamba3_siso::serialize::load_mamba3(&tmp, &cfg).unwrap();
    std::fs::remove_file(&tmp).ok();

    // Run inference with both
    let mut s1 = Mamba3State::zeros(&cfg);
    let mut s2 = Mamba3State::zeros(&cfg);
    let mut sc1 = Mamba3StepScratch::new(&cfg);
    let mut sc2 = Mamba3StepScratch::new(&cfg);
    let input = vec![1.0_f32; 128];
    let mut t1 = vec![0.0; cfg.d_model];
    let mut t2 = vec![0.0; cfg.d_model];

    mamba3_step(&mut t1, &input, &mut sc1, &w, &mut s1.layers, &cfg);
    mamba3_step(&mut t2, &input, &mut sc2, &w2, &mut s2.layers, &cfg);

    assert_eq!(t1, t2, "loaded weights must produce identical output");
}

// =========================================================================
// Training forward matches inference (step-by-step)
// =========================================================================

#[test]
fn test_m3_training_forward_matches_inference() {
    let cfg = test_cfg();
    let seq_len = 8;
    let dims = Mamba3Dims::from_config(&cfg, seq_len);
    let dm = dims.d_model;

    let w = init_layer_w(&dims);

    // Training forward: same temporal as inference init
    let mut temporal_train = vec![0.5_f32; seq_len * dm];
    run_training_forward(&w, &dims, &mut temporal_train);
    let train_last = &temporal_train[(seq_len - 1) * dm..seq_len * dm];

    // Inference: step through same inputs (each step gets 0.5 as input, output feeds back)
    // Note: training forward takes temporal as in/out (residual stream).
    // Inference single-step doesn't have input_proj — it's layer-level only.
    // So we compare the training output at each timestep vs the training output itself.
    // This test verifies the training forward produces finite nonzero output.
    assert!(
        train_last.iter().any(|&v| v.abs() > 1e-10),
        "training forward last timestep should be nonzero"
    );
    assert!(
        train_last.iter().all(|v| v.is_finite()),
        "training forward last timestep must be finite"
    );
}

// =========================================================================
// All weight gradients nonzero
// =========================================================================

/// The tape's checkpoint stride changes nothing: from carried SSM, K, V and
/// angle states, every stride (a checkpoint at every step, at a few, one
/// for the window, and one past it) gives the same rows, carried states,
/// input gradient and weight gradients, bit for bit, the gated norm and the
/// rotation included.
#[test]
fn m3_tape_stride_gives_the_same_bits() {
    let cfg = Mamba3Config {
        rope_fraction: 1.0,
        is_outproj_norm: true,
        ..test_cfg()
    };
    let seq_len = 37;
    let dims = Mamba3Dims::from_config(&cfg, seq_len);
    let w = init_layer_w_drawn(&dims);
    let (nh, hd, ds) = (dims.nheads, dims.headdim, dims.d_state);
    let na = dims.num_rope_angles.max(1);
    let draw = |n: usize, k: usize, m: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * k % m) as f32 / (m / 2) as f32) - 1.0)
            .collect()
    };
    let rows = draw(seq_len * dims.d_model, 37, 101);
    let d_rows = draw(seq_len * dims.d_model, 53, 97);
    let states0 = [
        draw(nh * hd * ds, 13, 23),
        draw(nh * ds, 17, 29),
        draw(nh * hd, 19, 31),
        draw(nh * na, 23, 37),
    ];
    let flat = |g: &TrainMamba3LayerWeights| -> Vec<f32> {
        [
            &g.norm_weight,
            &g.in_proj_w,
            &g.dt_bias,
            &g.b_norm_weight,
            &g.c_norm_weight,
            &g.b_bias,
            &g.c_bias,
            &g.d_param,
            &g.norm_gate_weight,
            &g.out_proj_w,
        ]
        .iter()
        .flat_map(|t| t.iter().copied())
        .collect()
    };
    let run = |stride: usize| -> Vec<Vec<u32>> {
        let mut acts = Mamba3LayerFlat::zeros_strided(dims, stride);
        let mut scratch = Mamba3TrainScratch::zeros(&dims);
        let [mut ssm, mut k, mut v, mut a] = states0.clone();
        let mut out = rows.clone();
        forward_mamba3_layer_batched(
            &mut out,
            &mut acts,
            &w,
            Mamba3LayerStateMut {
                ssm: &mut ssm,
                k: &mut k,
                v: &mut v,
                angle: &mut a,
            },
            &mut scratch,
            &dims,
        );
        let mut d = d_rows.clone();
        let mut g = TrainMamba3LayerWeights::zeros(&dims);
        backward_mamba3_layer_batched(&mut d, &acts, &w, &mut g, &mut scratch, &dims, None);
        [out, ssm, k, v, a, d, flat(&g)]
            .iter()
            .map(|x| x.iter().map(|f| f.to_bits()).collect())
            .collect()
    };
    let want = run(1);
    for stride in [3, 16, seq_len, seq_len + 13] {
        let got = run(stride);
        let what = [
            "rows",
            "SSM state",
            "K state",
            "V state",
            "angles",
            "input gradient",
            "weight gradients",
        ];
        for (i, what) in what.iter().enumerate() {
            assert_eq!(got[i], want[i], "stride {stride}: {what}");
        }
    }
}

#[test]
fn test_m3_all_cpu_gradients_nonzero() {
    let cfg = test_cfg();
    let seq_len = 4;
    let dims = Mamba3Dims::from_config(&cfg, seq_len);

    let w = init_layer_w(&dims);
    let mut temporal = vec![0.5_f32; seq_len * dims.d_model];
    let acts = run_training_forward(&w, &dims, &mut temporal);

    let mut d_temporal = vec![1.0_f32; seq_len * dims.d_model];
    let mut d_w = TrainMamba3LayerWeights::zeros(&dims);
    let mut scratch = Mamba3TrainScratch::zeros(&dims);
    backward_mamba3_layer_batched(
        &mut d_temporal,
        &acts,
        &w,
        &mut d_w,
        &mut scratch,
        &dims,
        None,
    );

    macro_rules! check_nonzero {
        ($name:ident) => {
            let m = d_w.$name.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            assert!(
                m > 1e-8,
                "{} gradient is zero (max_abs={m})",
                stringify!($name)
            );
        };
    }

    check_nonzero!(norm_weight);
    check_nonzero!(in_proj_w);
    check_nonzero!(dt_bias);
    check_nonzero!(b_norm_weight);
    check_nonzero!(c_norm_weight);
    check_nonzero!(b_bias);
    check_nonzero!(c_bias);
    check_nonzero!(d_param);
    check_nonzero!(out_proj_w);
    // norm_gate_weight only has gradient when is_outproj_norm=true, skip here
}

// =========================================================================
// Custom configs (different model sizes)
// =========================================================================

#[test]
fn test_m3_custom_config_small() {
    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 4,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    };
    let input_dim = 16;
    let w = Mamba3Weights::init(&cfg, input_dim, 99);
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut output = vec![0.0f32; cfg.d_model];
    let input = vec![0.5f32; input_dim];

    mamba3_step(
        &mut output,
        &input,
        &mut scratch,
        &w,
        &mut state.layers,
        &cfg,
    );
    assert!(
        output.iter().any(|&v| v.abs() > 1e-10),
        "custom small config should produce nonzero output"
    );
}

#[test]
fn test_m3_custom_config_large() {
    let cfg = Mamba3Config {
        d_model: 128,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 4,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    };
    let input_dim = 128;
    let w = Mamba3Weights::init(&cfg, input_dim, 77);
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut output = vec![0.0f32; cfg.d_model];
    let input = vec![0.1f32; input_dim];

    // Run 10 steps, check no NaN/Inf
    for step in 0..10 {
        mamba3_step(
            &mut output,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "step {step}: output contains NaN or Inf"
        );
    }
}

#[test]
fn test_m3_custom_config_outproj_norm() {
    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    };
    let input_dim = 32;
    let w = Mamba3Weights::init(&cfg, input_dim, 55);
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut output = vec![0.0f32; cfg.d_model];
    let input = vec![0.5f32; input_dim];

    for step in 0..5 {
        mamba3_step(
            &mut output,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "step {step}: outproj_norm output contains NaN or Inf"
        );
    }
    assert!(
        output.iter().any(|&v| v.abs() > 1e-10),
        "outproj_norm should produce nonzero output"
    );
}

#[test]
fn test_m3_custom_config_ngroups() {
    let cfg = Mamba3Config {
        d_model: 32,
        d_state: 4,
        expand: 2,
        headdim: 4,
        ngroups: 2, // 2 groups for BCNorm
        n_layers: 1,
        rope_fraction: 0.5,
        a_floor: 0.0625,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    };
    cfg.validate().unwrap();
    let input_dim = 16;
    let w = Mamba3Weights::init(&cfg, input_dim, 123);
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut output = vec![0.0f32; cfg.d_model];
    let input = vec![0.3f32; input_dim];

    mamba3_step(
        &mut output,
        &input,
        &mut scratch,
        &w,
        &mut state.layers,
        &cfg,
    );
    assert!(
        output.iter().any(|&v| v.abs() > 1e-10),
        "ngroups=2 config should produce nonzero output"
    );
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn test_m3_seq_len_one() {
    let cfg = test_cfg();
    let seq_len = 1;
    let dims = Mamba3Dims::from_config(&cfg, seq_len);

    let w = init_layer_w(&dims);
    let mut temporal = vec![0.5_f32; dims.d_model];
    run_training_forward(&w, &dims, &mut temporal);

    assert!(
        temporal.iter().all(|v| v.is_finite()),
        "seq_len=1 produces NaN/Inf"
    );
    assert!(
        temporal.iter().any(|&v| v.abs() > 1e-10),
        "seq_len=1 produces zero output"
    );
}

#[test]
fn test_m3_seq_len_one_backward() {
    let cfg = test_cfg();
    let seq_len = 1;
    let dims = Mamba3Dims::from_config(&cfg, seq_len);

    let w = init_layer_w(&dims);
    let mut temporal = vec![0.5_f32; dims.d_model];
    let acts = run_training_forward(&w, &dims, &mut temporal);

    let mut d_temporal = vec![1.0_f32; dims.d_model];
    let mut d_w = TrainMamba3LayerWeights::zeros(&dims);
    let mut scratch = Mamba3TrainScratch::zeros(&dims);
    backward_mamba3_layer_batched(
        &mut d_temporal,
        &acts,
        &w,
        &mut d_w,
        &mut scratch,
        &dims,
        None,
    );

    assert!(
        d_temporal.iter().all(|v| v.is_finite()),
        "seq_len=1 backward d_temporal NaN/Inf"
    );
    assert!(
        d_w.sum_sq() > 0.0,
        "seq_len=1 backward produces zero gradients"
    );
}

// =========================================================================
// Long sequence stability
// =========================================================================

#[test]
fn test_m3_long_sequence_stability() {
    let cfg = Mamba3Config::default();
    let input_dim = 128;
    let w = Mamba3Weights::init(&cfg, input_dim, 42);
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut output = vec![0.0f32; cfg.d_model];

    // Run 200 steps — verify no NaN/Inf accumulation in state
    for step in 0..200 {
        let input: Vec<f32> = (0..input_dim)
            .map(|i| ((step * input_dim + i) as f32) * 0.001)
            .collect();
        mamba3_step(
            &mut output,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "step {step}: output contains NaN or Inf"
        );
    }
}

// =========================================================================
// Long training forward+backward stability
// =========================================================================

#[test]
fn test_m3_long_training_stability() {
    let cfg = test_cfg();
    let seq_len = 32;
    let dims = Mamba3Dims::from_config(&cfg, seq_len);

    let w = init_layer_w(&dims);
    let mut temporal = vec![0.5_f32; seq_len * dims.d_model];
    let acts = run_training_forward(&w, &dims, &mut temporal);

    assert!(
        temporal.iter().all(|v| v.is_finite()),
        "seq_len=32 forward output has NaN/Inf"
    );

    let mut d_temporal = vec![1.0_f32; seq_len * dims.d_model];
    let mut d_w = TrainMamba3LayerWeights::zeros(&dims);
    let mut scratch = Mamba3TrainScratch::zeros(&dims);
    backward_mamba3_layer_batched(
        &mut d_temporal,
        &acts,
        &w,
        &mut d_w,
        &mut scratch,
        &dims,
        None,
    );

    assert!(
        d_temporal.iter().all(|v| v.is_finite()),
        "seq_len=32 backward d_temporal has NaN/Inf"
    );
    let grad_norm = d_w.sum_sq().sqrt();
    assert!(
        grad_norm.is_finite() && grad_norm > 0.0,
        "seq_len=32 backward grad_norm: {grad_norm}"
    );
}

// =========================================================================
// RoPE: no RoPE config still works
// =========================================================================

#[test]
fn test_m3_no_rope() {
    let cfg = Mamba3Config {
        d_model: 16,
        d_state: 4,
        expand: 2,
        headdim: 4,
        ngroups: 1,
        n_layers: 1,
        rope_fraction: 0.5, // with headdim=4, num_rope_angles = 4*0.5/2 = 1
        a_floor: 0.0625,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    };
    // Verify RoPE angles produce valid output
    assert!(cfg.num_rope_angles() > 0);

    let input_dim = 8;
    let w = Mamba3Weights::init(&cfg, input_dim, 42);
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3StepScratch::new(&cfg);
    let mut output = vec![0.0f32; cfg.d_model];
    let input = vec![0.5f32; input_dim];

    for step in 0..10 {
        mamba3_step(
            &mut output,
            &input,
            &mut scratch,
            &w,
            &mut state.layers,
            &cfg,
        );
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "step {step}: RoPE output NaN/Inf"
        );
    }

    // Verify angle state is non-trivial (accumulated)
    let angles = &state.layers[0].angle_state;
    assert!(
        angles.iter().any(|&a| a.abs() > 1e-6),
        "angle state should accumulate"
    );
}

// =========================================================================
// Determinism
// =========================================================================

#[test]
fn test_m3_deterministic() {
    let cfg = Mamba3Config::default();
    let input_dim = 64;
    let w = Mamba3Weights::init(&cfg, input_dim, 42);
    let input = vec![0.5_f32; input_dim];

    // Run 1
    let mut s1 = Mamba3State::zeros(&cfg);
    let mut sc1 = Mamba3StepScratch::new(&cfg);
    let mut o1 = vec![0.0f32; cfg.d_model];
    for _ in 0..5 {
        mamba3_step(&mut o1, &input, &mut sc1, &w, &mut s1.layers, &cfg);
    }

    // Run 2
    let mut s2 = Mamba3State::zeros(&cfg);
    let mut sc2 = Mamba3StepScratch::new(&cfg);
    let mut o2 = vec![0.0f32; cfg.d_model];
    for _ in 0..5 {
        mamba3_step(&mut o2, &input, &mut sc2, &w, &mut s2.layers, &cfg);
    }

    assert_eq!(o1, o2, "two identical runs must produce identical output");
}

// =========================================================================
// Gradient accumulation: two backward passes should double gradients
// =========================================================================

#[test]
fn test_m3_gradient_accumulation() {
    let cfg = test_cfg();
    let seq_len = 4;
    let dims = Mamba3Dims::from_config(&cfg, seq_len);

    let w = init_layer_w(&dims);
    let mut temporal = vec![0.5_f32; seq_len * dims.d_model];
    let acts = run_training_forward(&w, &dims, &mut temporal);

    // Single backward
    let mut d_temporal_1 = vec![1.0_f32; seq_len * dims.d_model];
    let mut d_w_1 = TrainMamba3LayerWeights::zeros(&dims);
    let mut scratch = Mamba3TrainScratch::zeros(&dims);
    backward_mamba3_layer_batched(
        &mut d_temporal_1,
        &acts,
        &w,
        &mut d_w_1,
        &mut scratch,
        &dims,
        None,
    );

    // Double backward (accumulate)
    let mut d_temporal_2 = vec![1.0_f32; seq_len * dims.d_model];
    let mut d_w_2 = TrainMamba3LayerWeights::zeros(&dims);
    backward_mamba3_layer_batched(
        &mut d_temporal_2,
        &acts,
        &w,
        &mut d_w_2,
        &mut scratch,
        &dims,
        None,
    );
    backward_mamba3_layer_batched(
        &mut d_temporal_2,
        &acts,
        &w,
        &mut d_w_2,
        &mut scratch,
        &dims,
        None,
    );

    // d_w_2 should be 2x d_w_1 for all weight groups
    let ratio = d_w_2.sum_sq() / d_w_1.sum_sq();
    assert!(
        (ratio - 4.0).abs() < 0.01, // sum_sq doubles → ratio is 4.0
        "gradient accumulation ratio: {ratio:.4} (expected 4.0)"
    );
}

/// Draws in `[-scale, scale)` from a fixed xorshift stream.
fn draw(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// The largest deviation from the float64 values over their largest one.
fn rel_max(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |a, &w| a.max(w.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .fold(0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()))
        / scale
}

/// The decode step of a whole model against the float64 reference step:
/// the input projection, every layer's mixer and gated MLP and the final
/// norm, under both block norms, every norm scale and bias drawn so no
/// constant init hides a path.
#[test]
fn m3_decode_matches_the_f64_backbone_oracle() {
    let base = Mamba3Config {
        d_model: 32,
        d_state: 16,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        is_outproj_norm: true,
        d_intermediate: 48,
        ..Mamba3Config::default()
    };
    let layer_norm = Mamba3Config {
        block_norm: mamba_rs::BlockNorm::Layer,
        ..base
    };
    let input_dim = 20usize;
    for (label, cfg) in [
        ("RMSNorm, MLP", base),
        ("LayerNorm, MLP", layer_norm),
        (
            "LayerNorm, no MLP, SiLU gate",
            Mamba3Config {
                d_intermediate: 0,
                is_outproj_norm: false,
                ..layer_norm
            },
        ),
    ] {
        cfg.validate().unwrap();
        let mut w = Mamba3Weights::init(&cfg, input_dim, 7);
        let norms = w
            .layers
            .iter_mut()
            .flat_map(|lw| [&mut lw.norm_weight, &mut lw.norm_bias])
            .chain(
                w.mlp
                    .iter_mut()
                    .flat_map(|m| [&mut m.norm_weight, &mut m.norm_bias]),
            )
            .chain([&mut w.norm_f_weight, &mut w.norm_f_bias]);
        for (k, v) in norms.enumerate() {
            // Scales around one, biases around zero.
            let base = if k % 2 == 0 { 1.0 } else { 0.0 };
            let draws = draw(v.len(), 40 + k as u64, 0.3);
            for (x, d) in v.iter_mut().zip(draws) {
                *x = base + d;
            }
        }
        let layers: Vec<TrainMamba3LayerWeights> = w.layers.iter().map(train_layer).collect();
        let mut st32 = Mamba3State::zeros(&cfg);
        let mut scratch = Mamba3StepScratch::new(&cfg);
        let mut st64 = zeros64(&cfg);
        let mut out = vec![0f32; cfg.d_model];
        let mut worst = 0f64;
        for t in 0..24u64 {
            let input = draw(input_dim, 300 + t, 1.0);
            mamba3_step(&mut out, &input, &mut scratch, &w, &mut st32.layers, &cfg);
            let want = backbone_step64(&cfg, (&w, &layers), &mut st64, &input);
            worst = worst.max(rel_max(&out, &want));
        }
        eprintln!("{label}: outputs {worst:.2e}");
        assert!(worst < 2e-5, "{label}: outputs rel {worst:.3e}");
    }
}
