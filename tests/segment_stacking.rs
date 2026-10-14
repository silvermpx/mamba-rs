//! A backbone built without its final norm is a segment of a stack: its
//! output is the residual stream after its last block, and a second
//! segment built with an identity input carries it on. In every family two
//! segments of two layers give a model of four layers bit for bit on the
//! host: the rows of the forward, and after one applying step, with the
//! second segment's input gradient driving the first segment's backward,
//! every master weight. A segment saves and loads through its family's
//! checkpoint, and a language model refuses one.

use mamba_rs::config::ScanMode;
use mamba_rs::mlp::GatedMlpLayerWeights;
use mamba_rs::{
    AnyCpuTrainer, BackwardOpts, CpuTrainer, Mamba2Config, Mamba2CpuTrainer, Mamba2Weights,
    Mamba3Config, Mamba3CpuTrainer, Mamba3MimoConfig, Mamba3MimoCpuTrainer, Mamba3MimoWeights,
    Mamba3Weights, MambaConfig, MambaCpuTrainer, MambaWeights, TrainSessionCfg,
};
#[cfg(feature = "hf")]
use mamba_rs::{Mamba2Backbone, MambaLM};

const BATCH: usize = 2;
const SEQ_LEN: usize = 12;
const D_MODEL: usize = 32;
const INPUT_DIM: usize = 24;
const LAYERS: usize = 4;
/// The layers of the first segment; the second takes the rest.
const CUT: usize = 2;

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

/// The trained tensors of a model, the input projection, then every
/// layer's mixer, every layer's MLP and the final norm, in one flat list.
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

/// The whole model, its two segments, the flat master weights of a
/// segment's trainer, and those of the whole model's trainer cut where the
/// segments are.
struct Stack {
    whole: AnyCpuTrainer,
    first: AnyCpuTrainer,
    second: AnyCpuTrainer,
    masters: fn(&AnyCpuTrainer) -> Vec<f32>,
    cut: fn(&AnyCpuTrainer) -> (Vec<f32>, Vec<f32>),
}

fn stacks() -> Vec<(&'static str, Stack)> {
    let s = TrainSessionCfg::new(INPUT_DIM, BATCH, SEQ_LEN);
    let s_id = TrainSessionCfg::new(D_MODEL, BATCH, SEQ_LEN);
    // The LayerNorm biases start at zero; drawn ones make the forward read
    // them.
    let mut m1 = MambaWeights::init(&m1_cfg(), INPUT_DIM, 1);
    for (k, b) in m1
        .layers
        .iter_mut()
        .map(|l| &mut l.norm_bias)
        .chain(m1.mlp.iter_mut().map(|m| &mut m.norm_bias))
        .chain(std::iter::once(&mut m1.norm_f_bias))
        .enumerate()
    {
        let d: Vec<f32> = values(b.len(), 50 + k as u32)
            .iter()
            .map(|v| v * 0.2)
            .collect();
        b.copy_from_slice(&d);
    }
    let (m1_a, m1_ac, m1_b, m1_bc) = split!(&m1, m1_cfg(), MambaConfig);
    let m2 = Mamba2Weights::init(&m2_cfg(), INPUT_DIM, 2);
    let (m2_a, m2_ac, m2_b, m2_bc) = split!(&m2, m2_cfg(), Mamba2Config);
    let siso = Mamba3Weights::init(&siso_cfg(), INPUT_DIM, 3);
    let (siso_a, siso_ac, siso_b, siso_bc) = split!(&siso, siso_cfg(), Mamba3Config);
    let mimo = Mamba3MimoWeights::init(&mimo_cfg(), INPUT_DIM, 4);
    let (mimo_a, mimo_ac, mimo_b, mimo_bc) = split!(&mimo, mimo_cfg(), Mamba3MimoConfig);
    vec![
        (
            "Mamba-1",
            Stack {
                whole: MambaCpuTrainer::new(&m1, m1_cfg(), s).unwrap().into(),
                first: MambaCpuTrainer::new(&m1_a, m1_ac, s).unwrap().into(),
                second: MambaCpuTrainer::new(&m1_b, m1_bc, s_id).unwrap().into(),
                masters: |t| match t {
                    AnyCpuTrainer::M1(t) => m1_flat(&t.snapshot_master()),
                    _ => unreachable!(),
                },
                cut: |t| match t {
                    AnyCpuTrainer::M1(t) => {
                        let (a, _, b, _) = split!(&t.snapshot_master(), m1_cfg(), MambaConfig);
                        (m1_flat(&a), m1_flat(&b))
                    }
                    _ => unreachable!(),
                },
            },
        ),
        (
            "Mamba-2",
            Stack {
                whole: Mamba2CpuTrainer::new(&m2, m2_cfg(), s).unwrap().into(),
                first: Mamba2CpuTrainer::new(&m2_a, m2_ac, s).unwrap().into(),
                second: Mamba2CpuTrainer::new(&m2_b, m2_bc, s_id).unwrap().into(),
                masters: |t| match t {
                    AnyCpuTrainer::M2(t) => m2_flat(&t.snapshot_master()),
                    _ => unreachable!(),
                },
                cut: |t| match t {
                    AnyCpuTrainer::M2(t) => {
                        let (a, _, b, _) = split!(&t.snapshot_master(), m2_cfg(), Mamba2Config);
                        (m2_flat(&a), m2_flat(&b))
                    }
                    _ => unreachable!(),
                },
            },
        ),
        (
            "SISO",
            Stack {
                whole: Mamba3CpuTrainer::new(&siso, siso_cfg(), s).unwrap().into(),
                first: Mamba3CpuTrainer::new(&siso_a, siso_ac, s).unwrap().into(),
                second: Mamba3CpuTrainer::new(&siso_b, siso_bc, s_id)
                    .unwrap()
                    .into(),
                masters: |t| match t {
                    AnyCpuTrainer::M3(t) => siso_flat(&t.snapshot_master()),
                    _ => unreachable!(),
                },
                cut: |t| match t {
                    AnyCpuTrainer::M3(t) => {
                        let (a, _, b, _) = split!(&t.snapshot_master(), siso_cfg(), Mamba3Config);
                        (siso_flat(&a), siso_flat(&b))
                    }
                    _ => unreachable!(),
                },
            },
        ),
        (
            "MIMO",
            Stack {
                whole: Mamba3MimoCpuTrainer::new(&mimo, mimo_cfg(), s)
                    .unwrap()
                    .into(),
                first: Mamba3MimoCpuTrainer::new(&mimo_a, mimo_ac, s)
                    .unwrap()
                    .into(),
                second: Mamba3MimoCpuTrainer::new(&mimo_b, mimo_bc, s_id)
                    .unwrap()
                    .into(),
                masters: |t| match t {
                    AnyCpuTrainer::M3Mimo(t) => mimo_flat(&t.snapshot_master()),
                    _ => unreachable!(),
                },
                cut: |t| match t {
                    AnyCpuTrainer::M3Mimo(t) => {
                        let (a, _, b, _) =
                            split!(&t.snapshot_master(), mimo_cfg(), Mamba3MimoConfig);
                        (mimo_flat(&a), mimo_flat(&b))
                    }
                    _ => unreachable!(),
                },
            },
        ),
    ]
}

#[test]
fn two_segments_train_as_the_whole_model_in_every_family() {
    let input = values(BATCH * SEQ_LEN * INPUT_DIM, 7);
    let d_out = values(BATCH * SEQ_LEN * D_MODEL, 8);
    let n_out = BATCH * SEQ_LEN * D_MODEL;
    let opts = BackwardOpts::default();
    for (family, mut st) in stacks() {
        let (mut whole_rows, mut cut_rows, mut rows) = (
            vec![0.0f32; n_out],
            vec![0.0f32; n_out],
            vec![0.0f32; n_out],
        );
        st.whole.forward(&input, &mut whole_rows).unwrap();
        st.first.forward(&input, &mut cut_rows).unwrap();
        st.second.forward(&cut_rows, &mut rows).unwrap();
        assert_eq!(bits(&rows), bits(&whole_rows), "{family}: the rows");

        let m = st.whole.backward_step(&d_out, opts).unwrap();
        assert!(m.optimizer_stepped, "{family}: the whole model stepped");
        let mut d_cut = vec![0.0f32; n_out];
        let m = st
            .second
            .backward_step_input_grad(&d_out, opts, &mut d_cut)
            .unwrap();
        assert!(m.optimizer_stepped, "{family}: the second segment stepped");
        let m = st.first.backward_step(&d_cut, opts).unwrap();
        assert!(m.optimizer_stepped, "{family}: the first segment stepped");

        let (whole_first, whole_second) = (st.cut)(&st.whole);
        assert_eq!(
            bits(&(st.masters)(&st.first)),
            bits(&whole_first),
            "{family}: the first segment's weights after the step"
        );
        assert_eq!(
            bits(&(st.masters)(&st.second)),
            bits(&whole_second),
            "{family}: the second segment's weights after the step"
        );
    }
}

/// A segment saves and loads through its family's checkpoint: the config
/// keeps `final_norm = false`, and the weights come back without a final
/// norm, every tensor with its bits.
#[test]
fn a_segment_saves_and_loads_in_every_family() {
    let dir = std::env::temp_dir().join(format!("mamba_rs_segments_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    macro_rules! round_trip {
        ($module:ident, $Weights:ident, $Cfg:ident, $cfg:expr, $flat:ident, $name:expr) => {{
            let w = $Weights::init(&$cfg, INPUT_DIM, 9);
            let (seg, seg_cfg, _, _) = split!(&w, $cfg, $Cfg);
            let path = dir.join($name);
            mamba_rs::$module::serialize::save(&path, &seg, &seg_cfg, INPUT_DIM).unwrap();
            let (back, back_cfg, input_dim) = mamba_rs::$module::serialize::load(&path).unwrap();
            assert_eq!(input_dim, INPUT_DIM, "{}", $name);
            assert!(!back_cfg.final_norm, "{}: the flag", $name);
            assert!(back.norm_f_weight.is_empty() && back.norm_f_bias.is_empty());
            assert_eq!(bits(&$flat(&back)), bits(&$flat(&seg)), "{}", $name);
            std::fs::remove_file(&path).unwrap();
        }};
    }
    round_trip!(
        mamba1,
        MambaWeights,
        MambaConfig,
        m1_cfg(),
        m1_flat,
        "m1.safetensors"
    );
    round_trip!(
        mamba2,
        Mamba2Weights,
        Mamba2Config,
        m2_cfg(),
        m2_flat,
        "m2.safetensors"
    );
    round_trip!(
        mamba3_siso,
        Mamba3Weights,
        Mamba3Config,
        siso_cfg(),
        siso_flat,
        "siso.safetensors"
    );
    round_trip!(
        mamba3_mimo,
        Mamba3MimoWeights,
        Mamba3MimoConfig,
        mimo_cfg(),
        mimo_flat,
        "mimo.safetensors"
    );
    let _ = std::fs::remove_dir(&dir);
}

/// A language model's head reads the final norm, so it refuses a backbone
/// built without one.
#[cfg(feature = "hf")]
#[test]
fn a_language_model_refuses_a_segment() {
    let w = Mamba2Weights::init(&m2_cfg(), D_MODEL, 2);
    let (seg, seg_cfg, _, _) = split!(&w, m2_cfg(), Mamba2Config);
    let backbone = Mamba2Backbone::new(seg, seg_cfg, D_MODEL).unwrap();
    let embed = values(10 * D_MODEL, 3);
    let err = MambaLM::new(backbone, embed, None, 10).err().unwrap();
    assert!(err.contains("final norm"), "{err}");
}
