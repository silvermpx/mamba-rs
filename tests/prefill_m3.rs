//! Full-sequence Mamba-3 CPU prefill contracts — the M3 twins of
//! `prefill.rs`, plus the param-labeled M1-vs-M3 head-to-head latency probe.

use mamba_rs::mamba_ssm::cpu::prefill::PrefillMode;
use mamba_rs::mamba3_siso::Mamba3LayerFlat;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::forward::{Mamba3LayerStateMut, forward_mamba3_layer_batched};
use mamba_rs::mamba3_siso::cpu::prefill::{
    Mamba3PrefillScratch, forward_mamba3_backbone_prefill, forward_mamba3_backbone_prefill_mode,
    prefill3_batch,
};
use mamba_rs::mamba3_siso::cpu::scratch::Mamba3Scratch;
use mamba_rs::mamba3_siso::cpu::weights::{TrainMamba3LayerWeights, TrainMamba3Weights};
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::ops::blas::sgemm_forward;
use mamba_rs::ops::fast_math::RMS_NORM_EPS;
use mamba_rs::ops::norms::rms_norm_inplace;

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

fn test_cfg() -> Mamba3Config {
    Mamba3Config {
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
    }
}

fn train_weights_from(w: &Mamba3Weights) -> TrainMamba3Weights {
    TrainMamba3Weights {
        input_proj_w: w.input_proj_w.clone(),
        input_proj_b: w.input_proj_b.clone(),
        layers: w
            .layers
            .iter()
            .map(|lw| TrainMamba3LayerWeights {
                norm_weight: lw.norm_weight.clone(),
                in_proj_w: lw.in_proj_w.clone(),
                dt_bias: lw.dt_bias.clone(),
                b_norm_weight: lw.b_norm_weight.clone(),
                c_norm_weight: lw.c_norm_weight.clone(),
                b_bias: lw.b_bias.clone(),
                c_bias: lw.c_bias.clone(),
                d_param: lw.d_param.clone(),
                norm_gate_weight: lw.norm_gate_weight.clone(),
                out_proj_w: lw.out_proj_w.clone(),
            })
            .collect(),
        norm_f_weight: w.norm_f_weight.clone(),
    }
}

fn assert_bits(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}]: {x} vs {y}");
    }
}

/// Prefill (zero state) must reproduce the training layer forwards
/// bit-for-bit: same input_proj GEMM, same per-layer math, same norm_f.
#[test]
fn m3_prefill_matches_training_layers_bitwise() {
    let cfg = test_cfg();
    let input_dim = 20usize;
    let seq_len = 8usize;
    let w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    let tw = train_weights_from(&w);
    let input = det(seq_len * input_dim, 0xAA, 0.05);
    let dims = Mamba3Dims::from_config(&cfg, seq_len);
    let dm = cfg.d_model;
    let (nh, hd, ds) = (cfg.nheads(), cfg.headdim, cfg.d_state);
    let na = cfg.num_rope_angles().max(1);

    // Reference: input_proj GEMM + training layer forwards + norm_f rows.
    let mut ref_out = vec![0.0f32; seq_len * dm];
    sgemm_forward(
        &mut ref_out,
        &input,
        &w.input_proj_w,
        Some(&w.input_proj_b),
        seq_len,
        input_dim,
        dm,
    );
    let mut phase = Mamba3Scratch::zeros(&dims);
    for lw in &tw.layers {
        let mut acts = Mamba3LayerFlat::zeros(dims);
        let mut ssm = vec![0.0f32; nh * hd * ds];
        let mut k = vec![0.0f32; nh * ds];
        let mut v = vec![0.0f32; nh * hd];
        let mut angle = vec![0.0f32; nh * na];
        forward_mamba3_layer_batched(
            &mut ref_out,
            &mut acts,
            lw,
            Mamba3LayerStateMut {
                ssm: &mut ssm,
                k: &mut k,
                v: &mut v,
                angle: &mut angle,
            },
            &mut phase,
            &dims,
        );
    }
    for row in ref_out.chunks_mut(dm) {
        rms_norm_inplace(row, &w.norm_f_weight, RMS_NORM_EPS);
    }

    // Prefill.
    let mut state = Mamba3State::zeros(&cfg);
    let mut scratch = Mamba3PrefillScratch::new(&dims, input_dim);
    let mut out = vec![0.0f32; seq_len * dm];
    forward_mamba3_backbone_prefill(&mut out, &input, &w, &mut state.layers, &mut scratch, &dims);

    assert_bits("m3 prefill vs training layers", &out, &ref_out);
}

/// Parallel mode must be bit-equal to Single — outputs AND carried states.
#[test]
fn m3_prefill_parallel_bit_equals_single() {
    let cfg = test_cfg();
    let input_dim = 20usize;
    let seq_len = 16usize;
    let w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    let input = det(seq_len * input_dim, 0xAB, 0.05);
    let dims = Mamba3Dims::from_config(&cfg, seq_len);

    let run = |mode: PrefillMode| -> (Vec<f32>, Mamba3State) {
        let mut state = Mamba3State::zeros(&cfg);
        let mut scratch = Mamba3PrefillScratch::new(&dims, input_dim);
        let mut out = vec![0.0f32; seq_len * cfg.d_model];
        forward_mamba3_backbone_prefill_mode(
            &mut out,
            &input,
            &w,
            &mut state.layers,
            &mut scratch,
            &dims,
            mode,
        );
        (out, state)
    };
    let (out_s, st_s) = run(PrefillMode::Single);
    let (out_p, st_p) = run(PrefillMode::Parallel);
    assert_bits("m3 parallel vs single output", &out_p, &out_s);
    for (l, (a, b)) in st_s.layers.iter().zip(st_p.layers.iter()).enumerate() {
        assert_bits(&format!("L{l} ssm_state"), &b.ssm_state, &a.ssm_state);
        assert_bits(&format!("L{l} k_state"), &b.k_state, &a.k_state);
        assert_bits(&format!("L{l} v_state"), &b.v_state, &a.v_state);
        assert_bits(&format!("L{l} angle_state"), &b.angle_state, &a.angle_state);
    }
}

/// After prefilling T tokens, the per-step decode path must continue as if
/// all T+1 tokens had gone through the step loop (tolerance: the step path
/// uses matvec accumulation orders, prefill uses SGEMM).
#[test]
fn m3_prefill_then_decode_handoff() {
    let cfg = test_cfg();
    let input_dim = cfg.d_model;
    let seq_len = 8usize;
    let w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    let dims = Mamba3Dims::from_config(&cfg, seq_len);
    let inputs = det((seq_len + 1) * input_dim, 0xAC, 0.05);
    let dm = cfg.d_model;

    // Reference: the pure step loop over T+1 tokens.
    let mut ref_state = Mamba3State::zeros(&cfg);
    let mut ref_scratch = mamba_rs::mamba3_siso::Mamba3StepScratch::new(&cfg);
    let mut ref_last = vec![0.0f32; dm];
    for t in 0..=seq_len {
        mamba_rs::mamba3_siso::mamba3_step(
            &mut ref_last,
            &inputs[t * input_dim..(t + 1) * input_dim],
            &mut ref_scratch,
            &w,
            &mut ref_state.layers,
            &cfg,
        );
    }

    // Prefill T, then decode token T via the step path.
    let mut state = Mamba3State::zeros(&cfg);
    let mut prefill_scratch = Mamba3PrefillScratch::new(&dims, input_dim);
    let mut prefill_out = vec![0.0f32; seq_len * dm];
    forward_mamba3_backbone_prefill(
        &mut prefill_out,
        &inputs[..seq_len * input_dim],
        &w,
        &mut state.layers,
        &mut prefill_scratch,
        &dims,
    );
    let mut step_scratch = mamba_rs::mamba3_siso::Mamba3StepScratch::new(&cfg);
    let mut step_out = vec![0.0f32; dm];
    mamba_rs::mamba3_siso::mamba3_step(
        &mut step_out,
        &inputs[seq_len * input_dim..],
        &mut step_scratch,
        &w,
        &mut state.layers,
        &cfg,
    );

    let mut dot = 0.0f64;
    let mut na_ = 0.0f64;
    let mut nb = 0.0f64;
    let mut worst = 0.0f32;
    for (&a, &b) in step_out.iter().zip(ref_last.iter()) {
        dot += a as f64 * b as f64;
        na_ += (a as f64).powi(2);
        nb += (b as f64).powi(2);
        let d = (a - b).abs();
        worst = worst.max(d / a.abs().max(b.abs()).max(1e-3));
    }
    let cos = dot / (na_.sqrt() * nb.sqrt());
    assert!(
        cos > 0.9999 && worst < 1e-2,
        "m3 handoff decode diverges from the step loop: cos={cos} max_rel={worst}"
    );
}

/// The batch helper must equal per-sample sequential prefills bit-for-bit.
#[test]
fn m3_prefill_batch_matches_sequential() {
    let cfg = test_cfg();
    let input_dim = 20usize;
    let seq_len = 8usize;
    let b = 3usize;
    let w = Mamba3Weights::init(&cfg, input_dim, 0xC0FFEE);
    let dims = Mamba3Dims::from_config(&cfg, seq_len);
    let dm = cfg.d_model;
    let inputs = det(b * seq_len * input_dim, 0xAE, 0.05);

    let mut seq_out = vec![0.0f32; b * seq_len * dm];
    for i in 0..b {
        let mut state = Mamba3State::zeros(&cfg);
        let mut scratch = Mamba3PrefillScratch::new(&dims, input_dim);
        forward_mamba3_backbone_prefill(
            &mut seq_out[i * seq_len * dm..(i + 1) * seq_len * dm],
            &inputs[i * seq_len * input_dim..(i + 1) * seq_len * input_dim],
            &w,
            &mut state.layers,
            &mut scratch,
            &dims,
        );
    }

    let mut states: Vec<Mamba3State> = (0..b).map(|_| Mamba3State::zeros(&cfg)).collect();
    let mut scratches: Vec<Mamba3PrefillScratch> = (0..b)
        .map(|_| Mamba3PrefillScratch::new(&dims, input_dim))
        .collect();
    let mut batch_out = vec![0.0f32; b * seq_len * dm];
    prefill3_batch(
        &mut batch_out,
        &inputs,
        &w,
        &mut states,
        &mut scratches,
        &dims,
    );

    assert_bits("m3 batch vs sequential", &batch_out, &seq_out);
}
