//! A backbone built without its final norm is a segment of a stack on the
//! GPU too, in every precision: its rows are its f32 residual stream, and
//! a second segment built with an identity input carries it on. Two
//! segments of two layers give a model of four layers bit for bit: the
//! trainers' rows and, after one applying step driven back through the
//! second segment's input gradient, every master weight; the backbones'
//! decode steps chained token by token.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::mlp::GatedMlpLayerWeights;
use mamba_rs::{
    AnyGpuBackbone, AnyGpuTrainer, BackwardOpts, GpuBackbone, GpuOptions, GpuTrainer, Mamba2Config,
    Mamba2GpuBackbone, Mamba2GpuTrainer, Mamba2Weights, Mamba3Config, Mamba3GpuBackbone,
    Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoGpuBackbone, Mamba3MimoGpuTrainer,
    Mamba3MimoWeights, Mamba3Weights, MambaConfig, MambaGpuBackbone, MambaGpuTrainer, MambaWeights,
    TrainSessionCfg, WeightDtype,
};

const BATCH: usize = 2;
const SEQ_LEN: usize = 16;
const D_MODEL: usize = 32;
const INPUT_DIM: usize = 24;
const LAYERS: usize = 4;
/// The layers of the first segment; the second takes the rest.
const CUT: usize = 2;
/// Decode steps chained through both segments.
const STEPS: u32 = 5;

const DTYPES: [WeightDtype; 3] = [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16];

fn m1_cfg() -> MambaConfig {
    MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        n_layers: LAYERS,
        scan_mode: ScanMode::Auto,
        d_intermediate: 40,
        block_norm: mamba_rs::BlockNorm::Layer,
        ..MambaConfig::default()
    }
}

fn m2_cfg() -> Mamba2Config {
    Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: LAYERS,
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
        n_layers: LAYERS,
        d_intermediate: 40,
        block_norm: mamba_rs::BlockNorm::Layer,
        ..Mamba3Config::default()
    }
}

fn mimo_cfg() -> Mamba3MimoConfig {
    Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers: LAYERS,
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The first `CUT` layers of `$w` with its input projection and without a
/// final norm, and the other layers with the final norm and an identity
/// input, with the configs that build them.
macro_rules! split {
    ($w:expr, $cfg:expr, $Cfg:ident) => {{
        let (w, cfg) = ($w, $cfg);
        let mut first = w.clone();
        first.layers.truncate(CUT);
        first.mlp.truncate(CUT);
        first.norm_f_weight.clear();
        first.norm_f_bias.clear();
        let mut second = w.clone();
        second.layers.drain(..CUT);
        second.mlp.drain(..CUT.min(second.mlp.len()));
        second.input_proj_w.clear();
        second.input_proj_b.clear();
        let first_cfg = $Cfg {
            n_layers: CUT,
            final_norm: false,
            ..cfg
        };
        let second_cfg = $Cfg {
            n_layers: LAYERS - CUT,
            ..cfg
        };
        (first, first_cfg, second, second_cfg)
    }};
}

/// The trained tensors of a model in one flat list.
macro_rules! flat {
    ($w:expr, [$($field:ident),*]) => {{
        let w = $w;
        let mut v = w.input_proj_w.clone();
        v.extend_from_slice(&w.input_proj_b);
        for l in &w.layers {
            $(v.extend_from_slice(&l.$field);)*
        }
        mlp_and_final(&mut v, &w.mlp, [&w.norm_f_weight, &w.norm_f_bias]);
        v
    }};
}

fn mlp_and_final(v: &mut Vec<f32>, mlp: &[GatedMlpLayerWeights], norm_f: [&Vec<f32>; 2]) {
    for m in mlp {
        for t in [&m.norm_weight, &m.norm_bias, &m.fc1_w, &m.fc2_w] {
            v.extend_from_slice(t);
        }
    }
    for t in norm_f {
        v.extend_from_slice(t);
    }
}

fn m1_flat(w: &MambaWeights) -> Vec<f32> {
    flat!(
        w,
        [
            norm_weight,
            norm_bias,
            in_proj_w,
            in_proj_b,
            conv1d_weight,
            conv1d_bias,
            x_proj_w,
            dt_proj_w,
            dt_proj_b,
            a_log,
            d_param,
            out_proj_w,
            out_proj_b
        ]
    )
}

fn m2_flat(w: &Mamba2Weights) -> Vec<f32> {
    flat!(
        w,
        [
            norm_weight,
            norm_bias,
            in_proj_w,
            in_proj_b,
            conv_w,
            conv_b,
            dt_bias,
            a_log,
            d_param,
            gate_norm_weight,
            out_proj_w,
            out_proj_b
        ]
    )
}

fn siso_flat(w: &Mamba3Weights) -> Vec<f32> {
    flat!(
        w,
        [
            norm_weight,
            norm_bias,
            in_proj_w,
            dt_bias,
            b_norm_weight,
            c_norm_weight,
            b_bias,
            c_bias,
            d_param,
            norm_gate_weight,
            out_proj_w
        ]
    )
}

fn mimo_flat(w: &Mamba3MimoWeights) -> Vec<f32> {
    flat!(
        w,
        [
            norm_weight,
            norm_bias,
            in_proj_w,
            dt_bias,
            b_norm_weight,
            c_norm_weight,
            b_bias,
            c_bias,
            d_param,
            mimo_x,
            mimo_z,
            mimo_o,
            out_proj_w,
            norm_gate_weight
        ]
    )
}

/// The whole model, its two segments, a segment trainer's flat master
/// weights, and the whole trainer's cut where the segments are.
struct Stack {
    whole: AnyGpuTrainer,
    first: AnyGpuTrainer,
    second: AnyGpuTrainer,
    masters: fn(&AnyGpuTrainer) -> Vec<f32>,
    cut: fn(&AnyGpuTrainer) -> (Vec<f32>, Vec<f32>),
}

/// The weights of every family, LayerNorm biases drawn away from zero so
/// the forward reads them.
fn weights() -> (
    MambaWeights,
    Mamba2Weights,
    Mamba3Weights,
    Mamba3MimoWeights,
) {
    let mut m1 = MambaWeights::init(&m1_cfg(), INPUT_DIM, 1);
    let mut siso = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 3);
    let biases = m1
        .layers
        .iter_mut()
        .map(|l| &mut l.norm_bias)
        .chain(m1.mlp.iter_mut().map(|m| &mut m.norm_bias))
        .chain(std::iter::once(&mut m1.norm_f_bias))
        .chain(siso.layers.iter_mut().map(|l| &mut l.norm_bias))
        .chain(siso.mlp.iter_mut().map(|m| &mut m.norm_bias))
        .chain(std::iter::once(&mut siso.norm_f_bias));
    for (k, b) in biases.enumerate() {
        let d: Vec<f32> = values(b.len(), 50 + k as u32)
            .iter()
            .map(|v| v * 0.2)
            .collect();
        b.copy_from_slice(&d);
    }
    (
        m1,
        Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 2),
        siso,
        Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 4),
    )
}

fn stacks(dtype: WeightDtype) -> Vec<(&'static str, Stack)> {
    let s = TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ_LEN);
    let s_id = TrainSessionCfg::new(D_MODEL, BATCH, SEQ_LEN);
    let o = GpuOptions::default().dtype(dtype);
    let (m1, m2, siso, mimo) = weights();
    let (m1_a, m1_ac, m1_b, m1_bc) = split!(&m1, m1_cfg(), MambaConfig);
    let (m2_a, m2_ac, m2_b, m2_bc) = split!(&m2, m2_cfg(), Mamba2Config);
    let (siso_a, siso_ac, siso_b, siso_bc) = split!(&siso, siso_cfg(), Mamba3Config);
    let (mimo_a, mimo_ac, mimo_b, mimo_bc) = split!(&mimo, mimo_cfg(), Mamba3MimoConfig);
    vec![
        (
            "Mamba-1",
            Stack {
                whole: MambaGpuTrainer::new(&m1, m1_cfg(), s, o).unwrap().into(),
                first: MambaGpuTrainer::new(&m1_a, m1_ac, s, o).unwrap().into(),
                second: MambaGpuTrainer::new(&m1_b, m1_bc, s_id, o).unwrap().into(),
                masters: |t| match t {
                    AnyGpuTrainer::M1(t) => m1_flat(&t.snapshot_master().unwrap()),
                    _ => unreachable!(),
                },
                cut: |t| match t {
                    AnyGpuTrainer::M1(t) => {
                        let w = t.snapshot_master().unwrap();
                        let (a, _, b, _) = split!(&w, m1_cfg(), MambaConfig);
                        (m1_flat(&a), m1_flat(&b))
                    }
                    _ => unreachable!(),
                },
            },
        ),
        (
            "Mamba-2",
            Stack {
                whole: Mamba2GpuTrainer::new(&m2, m2_cfg(), s, o).unwrap().into(),
                first: Mamba2GpuTrainer::new(&m2_a, m2_ac, s, o).unwrap().into(),
                second: Mamba2GpuTrainer::new(&m2_b, m2_bc, s_id, o).unwrap().into(),
                masters: |t| match t {
                    AnyGpuTrainer::M2(t) => m2_flat(&t.snapshot_master().unwrap()),
                    _ => unreachable!(),
                },
                cut: |t| match t {
                    AnyGpuTrainer::M2(t) => {
                        let w = t.snapshot_master().unwrap();
                        let (a, _, b, _) = split!(&w, m2_cfg(), Mamba2Config);
                        (m2_flat(&a), m2_flat(&b))
                    }
                    _ => unreachable!(),
                },
            },
        ),
        (
            "SISO",
            Stack {
                whole: Mamba3GpuTrainer::new(&siso, siso_cfg(), s, o)
                    .unwrap()
                    .into(),
                first: Mamba3GpuTrainer::new(&siso_a, siso_ac, s, o)
                    .unwrap()
                    .into(),
                second: Mamba3GpuTrainer::new(&siso_b, siso_bc, s_id, o)
                    .unwrap()
                    .into(),
                masters: |t| match t {
                    AnyGpuTrainer::M3(t) => siso_flat(&t.snapshot_master().unwrap()),
                    _ => unreachable!(),
                },
                cut: |t| match t {
                    AnyGpuTrainer::M3(t) => {
                        let w = t.snapshot_master().unwrap();
                        let (a, _, b, _) = split!(&w, siso_cfg(), Mamba3Config);
                        (siso_flat(&a), siso_flat(&b))
                    }
                    _ => unreachable!(),
                },
            },
        ),
        (
            "MIMO",
            Stack {
                whole: Mamba3MimoGpuTrainer::new(&mimo, mimo_cfg(), s, o)
                    .unwrap()
                    .into(),
                first: Mamba3MimoGpuTrainer::new(&mimo_a, mimo_ac, s, o)
                    .unwrap()
                    .into(),
                second: Mamba3MimoGpuTrainer::new(&mimo_b, mimo_bc, s_id, o)
                    .unwrap()
                    .into(),
                masters: |t| match t {
                    AnyGpuTrainer::M3Mimo(t) => mimo_flat(&t.snapshot_master().unwrap()),
                    _ => unreachable!(),
                },
                cut: |t| match t {
                    AnyGpuTrainer::M3Mimo(t) => {
                        let w = t.snapshot_master().unwrap();
                        let (a, _, b, _) = split!(&w, mimo_cfg(), Mamba3MimoConfig);
                        (mimo_flat(&a), mimo_flat(&b))
                    }
                    _ => unreachable!(),
                },
            },
        ),
    ]
}

#[test]
fn two_segments_train_as_the_whole_model_in_every_family_and_precision() {
    let input = values(BATCH * SEQ_LEN * INPUT_DIM, 7);
    let d_out: Vec<f32> = values(BATCH * SEQ_LEN * D_MODEL, 8)
        .iter()
        .map(|v| v * 1e-2)
        .collect();
    let n_out = BATCH * SEQ_LEN * D_MODEL;
    let opts = BackwardOpts::default();
    for dtype in DTYPES {
        for (family, mut st) in stacks(dtype) {
            let case = format!("{family} {dtype:?}");
            // An f16 scale the windows' gradients fit at; no effect in the
            // other precisions.
            for t in [&mut st.whole, &mut st.first, &mut st.second] {
                t.load_scaler_state(1024.0, 0).unwrap();
            }
            let (mut whole_rows, mut cut_rows, mut rows) = (
                vec![0.0f32; n_out],
                vec![0.0f32; n_out],
                vec![0.0f32; n_out],
            );
            st.whole.forward(&input, &mut whole_rows).unwrap();
            st.first.forward(&input, &mut cut_rows).unwrap();
            st.second.forward(&cut_rows, &mut rows).unwrap();
            assert_eq!(bits(&rows), bits(&whole_rows), "{case}: the rows");

            let m = st.whole.backward_step(&d_out, opts).unwrap();
            assert!(m.optimizer_stepped, "{case}: the whole model stepped");
            let mut d_cut = vec![0.0f32; n_out];
            let m = st
                .second
                .backward_step_input_grad(&d_out, opts, &mut d_cut)
                .unwrap();
            assert!(m.optimizer_stepped, "{case}: the second segment stepped");
            let m = st.first.backward_step(&d_cut, opts).unwrap();
            assert!(m.optimizer_stepped, "{case}: the first segment stepped");

            let (whole_first, whole_second) = (st.cut)(&st.whole);
            assert_eq!(
                bits(&(st.masters)(&st.first)),
                bits(&whole_first),
                "{case}: the first segment's weights after the step"
            );
            assert_eq!(
                bits(&(st.masters)(&st.second)),
                bits(&whole_second),
                "{case}: the second segment's weights after the step"
            );
        }
    }
}

/// The whole backbone and its two segments of every family.
fn backbones(dtype: WeightDtype) -> Vec<(&'static str, [AnyGpuBackbone; 3])> {
    let o = GpuOptions::default().dtype(dtype);
    let (m1, m2, siso, mimo) = weights();
    let (m1_a, m1_ac, m1_b, m1_bc) = split!(&m1, m1_cfg(), MambaConfig);
    let (m2_a, m2_ac, m2_b, m2_bc) = split!(&m2, m2_cfg(), Mamba2Config);
    let (siso_a, siso_ac, siso_b, siso_bc) = split!(&siso, siso_cfg(), Mamba3Config);
    let (mimo_a, mimo_ac, mimo_b, mimo_bc) = split!(&mimo, mimo_cfg(), Mamba3MimoConfig);
    vec![
        (
            "Mamba-1",
            [
                MambaGpuBackbone::new(&m1, m1_cfg(), INPUT_DIM, BATCH, o)
                    .unwrap()
                    .into(),
                MambaGpuBackbone::new(&m1_a, m1_ac, INPUT_DIM, BATCH, o)
                    .unwrap()
                    .into(),
                MambaGpuBackbone::new(&m1_b, m1_bc, D_MODEL, BATCH, o)
                    .unwrap()
                    .into(),
            ],
        ),
        (
            "Mamba-2",
            [
                Mamba2GpuBackbone::new(&m2, m2_cfg(), INPUT_DIM, BATCH, o)
                    .unwrap()
                    .into(),
                Mamba2GpuBackbone::new(&m2_a, m2_ac, INPUT_DIM, BATCH, o)
                    .unwrap()
                    .into(),
                Mamba2GpuBackbone::new(&m2_b, m2_bc, D_MODEL, BATCH, o)
                    .unwrap()
                    .into(),
            ],
        ),
        (
            "SISO",
            [
                Mamba3GpuBackbone::new(&siso, siso_cfg(), INPUT_DIM, BATCH, o)
                    .unwrap()
                    .into(),
                Mamba3GpuBackbone::new(&siso_a, siso_ac, INPUT_DIM, BATCH, o)
                    .unwrap()
                    .into(),
                Mamba3GpuBackbone::new(&siso_b, siso_bc, D_MODEL, BATCH, o)
                    .unwrap()
                    .into(),
            ],
        ),
        (
            "MIMO",
            [
                Mamba3MimoGpuBackbone::new(&mimo, mimo_cfg(), INPUT_DIM, BATCH, o)
                    .unwrap()
                    .into(),
                Mamba3MimoGpuBackbone::new(&mimo_a, mimo_ac, INPUT_DIM, BATCH, o)
                    .unwrap()
                    .into(),
                Mamba3MimoGpuBackbone::new(&mimo_b, mimo_bc, D_MODEL, BATCH, o)
                    .unwrap()
                    .into(),
            ],
        ),
    ]
}

/// Decode steps through the two segments, the first's f32 residual rows
/// feeding the second, give the whole backbone's rows, and a segment
/// reports its rows as f32 in every precision.
#[test]
fn two_segments_decode_as_the_whole_backbone_in_every_family_and_precision() {
    for dtype in DTYPES {
        for (family, [mut whole, mut first, mut second]) in backbones(dtype) {
            let case = format!("{family} {dtype:?}");
            assert_eq!(first.temporal_dtype(), WeightDtype::F32, "{case}");
            let (mut want, mut cut, mut got) = (
                vec![0.0f32; BATCH * D_MODEL],
                vec![0.0f32; BATCH * D_MODEL],
                vec![0.0f32; BATCH * D_MODEL],
            );
            for k in 0..STEPS {
                let x = values(BATCH * INPUT_DIM, 100 + k);
                whole.step(&x, &mut want).unwrap();
                first.step(&x, &mut cut).unwrap();
                second.step(&cut, &mut got).unwrap();
                assert_eq!(bits(&got), bits(&want), "{case}, step {k}");
            }
        }
    }
}
