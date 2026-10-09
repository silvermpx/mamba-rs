//! Loading the official Mamba-3 checkpoints (state-spaces/mamba3-*).
//!
//! These ship as `config.json` in the original mamba_ssm format (model
//! fields at the top level, the mixer settings under `ssm_cfg`) and a
//! `pytorch_model.bin` in bf16, with an optional safetensors conversion
//! beside it. Every layer is `x += mixer(norm(x)); x += mlp(norm2(x))`,
//! the embedding is tied to the head and there is no input projection.
//!
//! The key map is strict: every tensor must be one the model uses, every
//! one the model uses must be present, and every shape is checked, so a
//! checkpoint that does not match the architecture fails to load instead
//! of running with a missing branch.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::weights::{Mamba3LayerWeights, Mamba3Weights};
use crate::mlp::GatedMlpLayerWeights;

use super::pickle::TensorRef;
use super::torch_zip::TorchBin;

/// An official Mamba-3 checkpoint, ready for [`crate::module::MambaLM::from_mamba3`]
/// or the GPU language models.
pub struct HfMamba3Model {
    pub cfg: Mamba3Config,
    /// Layer weights with an empty (identity) input projection.
    pub weights: Mamba3Weights,
    /// The embedding `[vocab_padded, d_model]`, also the tied head.
    pub embed: Vec<f32>,
    /// The tokenizer's vocabulary size; rows past it are padding.
    pub vocab_size: usize,
}

/// An official Mamba-3 MIMO checkpoint, ready for the MIMO decode.
pub struct HfMamba3MimoModel {
    pub cfg: Mamba3MimoConfig,
    /// Layer weights with an empty (identity) input projection.
    pub weights: Mamba3MimoWeights,
    /// The embedding `[vocab_padded, d_model]`, also the tied head.
    pub embed: Vec<f32>,
    /// The tokenizer's vocabulary size; rows past it are padding.
    pub vocab_size: usize,
}

/// The parsed `config.json` of an official Mamba-3 checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct HfMamba3Config {
    pub cfg: Mamba3Config,
    /// The embedding rows, the vocabulary rounded up to the pad multiple.
    pub vocab_padded: usize,
    pub vocab_size: usize,
    pub tie_embeddings: bool,
    /// The checkpoint is Mamba-3 MIMO with this rank (SISO is `None`).
    pub mimo_rank: Option<usize>,
}

pub(super) fn field<'a>(obj: &'a Value, key: &str) -> Option<&'a Value> {
    obj.get(key).filter(|v| !v.is_null())
}

pub(super) fn usize_or(obj: &Value, key: &str, default: usize) -> Result<usize, String> {
    match field(obj, key) {
        None => Ok(default),
        Some(v) => v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| format!("config.json: {key} is not a non-negative integer")),
    }
}

pub(super) fn f32_or(obj: &Value, key: &str, default: f32) -> Result<f32, String> {
    match field(obj, key) {
        None => Ok(default),
        Some(v) => v
            .as_f64()
            .map(|n| n as f32)
            .ok_or_else(|| format!("config.json: {key} is not a number")),
    }
}

pub(super) fn bool_or(obj: &Value, key: &str, default: bool) -> Result<bool, String> {
    match field(obj, key) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| format!("config.json: {key} is not a bool")),
    }
}

/// Parse the original-format `config.json` of a Mamba-3 checkpoint. The
/// defaults are the reference's own (`MambaConfig` and `Mamba3.__init__`).
pub fn parse_mamba3_config(json: &[u8]) -> Result<HfMamba3Config, String> {
    let root: Value =
        serde_json::from_slice(json).map_err(|e| format!("config.json parse error: {e}"))?;
    let ssm =
        field(&root, "ssm_cfg").ok_or("config.json: no ssm_cfg (not a Mamba-3 checkpoint)")?;
    let layer = field(ssm, "layer").and_then(Value::as_str).unwrap_or("");
    if layer != "Mamba3" {
        return Err(format!(
            "config.json: ssm_cfg.layer is {layer:?}, not \"Mamba3\""
        ));
    }
    if field(&root, "attn_layer_idx")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
    {
        return Err("config.json: hybrid attention layers are not supported".into());
    }
    if !bool_or(&root, "rms_norm", true)? {
        return Err("config.json: rms_norm = false (LayerNorm) is not supported".into());
    }
    let eps = f32_or(&root, "norm_epsilon", 1e-5)?;
    // The reference hard-codes the mixer's B/C norm eps at 1e-5, while this
    // crate drives every norm from one eps; another block eps would be a
    // different model than the one loaded.
    if eps != 1e-5 {
        return Err(format!(
            "config.json: norm_epsilon {eps} differs from the mixer's fixed 1e-5"
        ));
    }
    let mimo_rank = if bool_or(ssm, "is_mimo", false)? {
        Some(usize_or(ssm, "mimo_rank", 4)?)
    } else {
        None
    };
    let vocab_size = usize_or(&root, "vocab_size", 50277)?;
    let pad = usize_or(&root, "pad_vocab_size_multiple", 8)?;
    let vocab_padded = if pad > 1 {
        vocab_size.div_ceil(pad) * pad
    } else {
        vocab_size
    };
    if vocab_size == 0 || vocab_padded > (1 << 24) {
        return Err(format!("config.json: vocab_size {vocab_size} out of range"));
    }
    let cfg = Mamba3Config {
        d_model: usize_or(&root, "d_model", 2560)?,
        n_layers: usize_or(&root, "n_layer", 64)?,
        d_intermediate: usize_or(&root, "d_intermediate", 0)?,
        d_state: usize_or(ssm, "d_state", 128)?,
        expand: usize_or(ssm, "expand", 2)?,
        headdim: usize_or(ssm, "headdim", 64)?,
        ngroups: usize_or(ssm, "ngroups", 1)?,
        rope_fraction: f32_or(ssm, "rope_fraction", 0.5)?,
        a_floor: f32_or(ssm, "A_floor", 1e-4)?,
        is_outproj_norm: bool_or(ssm, "is_outproj_norm", false)?,
        rms_norm_eps: eps,
        ..Mamba3Config::default()
    };
    cfg.validate()?;
    Ok(HfMamba3Config {
        cfg,
        vocab_padded,
        vocab_size,
        tie_embeddings: bool_or(&root, "tie_embeddings", true)?,
        mimo_rank,
    })
}

impl HfMamba3Config {
    /// The MIMO model this config describes, `None` for a SISO checkpoint.
    pub fn mimo_config(&self) -> Result<Option<Mamba3MimoConfig>, String> {
        let Some(mimo_rank) = self.mimo_rank else {
            return Ok(None);
        };
        let c = self.cfg;
        let cfg = Mamba3MimoConfig {
            d_model: c.d_model,
            d_state: c.d_state,
            expand: c.expand,
            headdim: c.headdim,
            ngroups: c.ngroups,
            n_layers: c.n_layers,
            mimo_rank,
            rope_fraction: c.rope_fraction,
            a_floor: c.a_floor,
            is_outproj_norm: c.is_outproj_norm,
            rms_norm_eps: c.rms_norm_eps,
            // The reference fixes the mixer's norms at 1e-5.
            ssm_norm_eps: 1e-5,
            d_intermediate: c.d_intermediate,
        };
        cfg.validate()?;
        Ok(Some(cfg))
    }
}

/// Tensors by name from either weight format the official repositories
/// carry.
pub(super) enum Source {
    TorchBin(TorchBin, HashMap<String, TensorRef>),
    Safetensors(Vec<u8>),
}

impl Source {
    pub(super) fn open(dir: &Path) -> Result<Self, String> {
        let st = dir.join("model.safetensors");
        if st.exists() {
            let bytes =
                std::fs::read(&st).map_err(|e| format!("cannot read {}: {e}", st.display()))?;
            return Ok(Self::Safetensors(bytes));
        }
        let bin = dir.join("pytorch_model.bin");
        let archive = TorchBin::open(&bin)?;
        let by_name = archive.tensors().iter().cloned().collect();
        Ok(Self::TorchBin(archive, by_name))
    }

    /// Every tensor name with its shape.
    fn shapes(&self) -> Result<Vec<(String, Vec<usize>)>, String> {
        match self {
            Self::TorchBin(archive, _) => Ok(archive
                .tensors()
                .iter()
                .map(|(n, t)| (n.clone(), t.shape.clone()))
                .collect()),
            Self::Safetensors(bytes) => {
                let st = safetensors::SafeTensors::deserialize(bytes)
                    .map_err(|e| format!("safetensors parse error: {e}"))?;
                Ok(st
                    .tensors()
                    .into_iter()
                    .map(|(n, t)| (n, t.shape().to_vec()))
                    .collect())
            }
        }
    }

    pub(super) fn read(&mut self, name: &str) -> Result<Vec<f32>, String> {
        match self {
            Self::TorchBin(archive, by_name) => {
                let t = by_name
                    .get(name)
                    .ok_or_else(|| format!("tensor {name} is missing"))?
                    .clone();
                archive.read_f32(&t)
            }
            Self::Safetensors(bytes) => {
                let st = safetensors::SafeTensors::deserialize(bytes)
                    .map_err(|e| format!("safetensors parse error: {e}"))?;
                let t = st.tensor(name).map_err(|e| format!("tensor {name}: {e}"))?;
                super::load::tensor_to_f32(t.data(), t.dtype())
            }
        }
    }
}

/// Transpose a PyTorch `[out, in]` weight to this crate's `[in, out]`.
pub(super) fn transpose(src: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; src.len()];
    for (o, row) in src.chunks_exact(inp).enumerate() {
        for (i, &v) in row.iter().enumerate() {
            dst[i * out + o] = v;
        }
    }
    dst
}

/// Read and parse a checkpoint directory's `config.json`.
pub fn read_mamba3_config(dir: &Path) -> Result<HfMamba3Config, String> {
    let json = std::fs::read(dir.join("config.json"))
        .map_err(|e| format!("cannot read config.json: {e}"))?;
    parse_mamba3_config(&json)
}

/// Check the checkpoint's tensors against the expected names and shapes:
/// nothing missing, nothing extra, every shape exact.
pub(super) fn check_tensors(
    src: &Source,
    expect: &HashMap<String, Vec<usize>>,
    tie_embeddings: bool,
) -> Result<(), String> {
    let mut seen = 0usize;
    for (name, shape) in src.shapes()? {
        if name == "lm_head.weight" {
            // A tied head is the embedding again; an untied one is a
            // different model this loader does not build.
            if !tie_embeddings {
                return Err("untied lm_head checkpoints are not supported".into());
            }
            continue;
        }
        match expect.get(&name) {
            Some(want) if *want == shape => seen += 1,
            Some(want) => {
                return Err(format!("tensor {name}: shape {shape:?}, expected {want:?}"));
            }
            None => return Err(format!("unexpected tensor {name} for this architecture")),
        }
    }
    if seen != expect.len() {
        let mut missing: Vec<&String> = expect.keys().collect();
        missing.sort();
        return Err(format!(
            "checkpoint holds {seen} of the {} tensors the model needs (first expected: {})",
            expect.len(),
            missing[0]
        ));
    }
    Ok(())
}

/// The tensors every Mamba-3 layer and the model around them carry, whatever
/// the mixer: embedding, final norm, the block norms and the MLP.
pub(super) fn expect_common(
    expect: &mut HashMap<String, Vec<usize>>,
    (n_layers, d, hidden, vocab_padded): (usize, usize, usize, usize),
) {
    expect.insert("backbone.embedding.weight".into(), vec![vocab_padded, d]);
    expect.insert("backbone.norm_f.weight".into(), vec![d]);
    for i in 0..n_layers {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        expect.insert(p("norm.weight"), vec![d]);
        if hidden > 0 {
            expect.insert(p("norm2.weight"), vec![d]);
            expect.insert(p("mlp.fc1.weight"), vec![2 * hidden, d]);
            expect.insert(p("mlp.fc2.weight"), vec![d, hidden]);
        }
    }
}

pub(super) fn read_mlp(
    src: &mut Source,
    i: usize,
    d: usize,
    hidden: usize,
) -> Result<GatedMlpLayerWeights, String> {
    let p = |n: &str| format!("backbone.layers.{i}.{n}");
    Ok(GatedMlpLayerWeights {
        norm_weight: src.read(&p("norm2.weight"))?,
        fc1_w: transpose(&src.read(&p("mlp.fc1.weight"))?, 2 * hidden, d),
        fc2_w: transpose(&src.read(&p("mlp.fc2.weight"))?, d, hidden),
    })
}

/// Load an official Mamba-3 MIMO checkpoint directory.
pub fn load_hf_mamba3_mimo(dir: &Path) -> Result<HfMamba3MimoModel, String> {
    let hf = read_mamba3_config(dir)?;
    let Some(cfg) = hf.mimo_config()? else {
        return Err("a Mamba-3 SISO checkpoint: load it with load_hf_mamba3".into());
    };
    let (d, di, ds, nh, r, p_len, ip) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.d_state,
        cfg.nheads(),
        cfg.mimo_rank,
        cfg.headdim,
        cfg.in_proj_out_dim(),
    );
    let hidden = cfg.mlp_hidden();
    let mut src = Source::open(dir)?;

    let mut expect: HashMap<String, Vec<usize>> = HashMap::new();
    expect_common(&mut expect, (cfg.n_layers, d, hidden, hf.vocab_padded));
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("backbone.layers.{i}.mixer.{n}");
        expect.insert(p("in_proj.weight"), vec![ip, d]);
        expect.insert(p("dt_bias"), vec![nh]);
        expect.insert(p("B_bias"), vec![nh, r, ds]);
        expect.insert(p("C_bias"), vec![nh, r, ds]);
        expect.insert(p("D"), vec![nh]);
        expect.insert(p("B_norm.weight"), vec![ds]);
        expect.insert(p("C_norm.weight"), vec![ds]);
        expect.insert(p("mimo_x"), vec![nh, r, p_len]);
        expect.insert(p("mimo_z"), vec![nh, r, p_len]);
        expect.insert(p("mimo_o"), vec![nh, r, p_len]);
        expect.insert(p("out_proj.weight"), vec![d, di]);
        if cfg.is_outproj_norm {
            expect.insert(p("norm.weight"), vec![di]);
        }
    }
    check_tensors(&src, &expect, hf.tie_embeddings)?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    let mut mlp = Vec::new();
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        layers.push(Mamba3MimoLayerWeights {
            norm_weight: src.read(&p("norm.weight"))?,
            in_proj_w: transpose(&src.read(&p("mixer.in_proj.weight"))?, ip, d),
            dt_bias: src.read(&p("mixer.dt_bias"))?,
            b_norm_weight: src.read(&p("mixer.B_norm.weight"))?,
            c_norm_weight: src.read(&p("mixer.C_norm.weight"))?,
            b_bias: src.read(&p("mixer.B_bias"))?,
            c_bias: src.read(&p("mixer.C_bias"))?,
            d_param: src.read(&p("mixer.D"))?,
            mimo_x: src.read(&p("mixer.mimo_x"))?,
            mimo_z: src.read(&p("mixer.mimo_z"))?,
            mimo_o: src.read(&p("mixer.mimo_o"))?,
            norm_gate_weight: if cfg.is_outproj_norm {
                src.read(&p("mixer.norm.weight"))?
            } else {
                vec![1.0; di]
            },
            out_proj_w: transpose(&src.read(&p("mixer.out_proj.weight"))?, d, di),
        });
        if hidden > 0 {
            mlp.push(read_mlp(&mut src, i, d, hidden)?);
        }
    }
    let weights = Mamba3MimoWeights {
        input_proj_w: Vec::new(),
        input_proj_b: Vec::new(),
        layers,
        norm_f_weight: src.read("backbone.norm_f.weight")?,
        mlp,
    };
    weights.validate(&cfg, d)?;
    let embed = src.read("backbone.embedding.weight")?;
    Ok(HfMamba3MimoModel {
        cfg,
        weights,
        embed,
        vocab_size: hf.vocab_size,
    })
}

/// Load an official Mamba-3 SISO checkpoint directory.
pub fn load_hf_mamba3(dir: &Path) -> Result<HfMamba3Model, String> {
    let hf = read_mamba3_config(dir)?;
    if let Some(rank) = hf.mimo_rank {
        return Err(format!(
            "a Mamba-3 MIMO (rank {rank}) checkpoint: load it with load_hf_mamba3_mimo"
        ));
    }
    let cfg = hf.cfg;
    let (d, di, ds, nh, ip) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.d_state,
        cfg.nheads(),
        cfg.in_proj_out_dim(),
    );
    let hidden = cfg.mlp_hidden();
    let mut src = Source::open(dir)?;

    // The expected tensors and their reference shapes.
    let mut expect: HashMap<String, Vec<usize>> = HashMap::new();
    expect_common(&mut expect, (cfg.n_layers, d, hidden, hf.vocab_padded));
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        expect.insert(p("mixer.in_proj.weight"), vec![ip, d]);
        expect.insert(p("mixer.dt_bias"), vec![nh]);
        expect.insert(p("mixer.B_bias"), vec![nh, 1, ds]);
        expect.insert(p("mixer.C_bias"), vec![nh, 1, ds]);
        expect.insert(p("mixer.D"), vec![nh]);
        expect.insert(p("mixer.B_norm.weight"), vec![ds]);
        expect.insert(p("mixer.C_norm.weight"), vec![ds]);
        expect.insert(p("mixer.out_proj.weight"), vec![d, di]);
        if cfg.is_outproj_norm {
            expect.insert(p("mixer.norm.weight"), vec![di]);
        }
    }
    check_tensors(&src, &expect, hf.tie_embeddings)?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    let mut mlp = Vec::new();
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        layers.push(Mamba3LayerWeights {
            norm_weight: src.read(&p("norm.weight"))?,
            in_proj_w: transpose(&src.read(&p("mixer.in_proj.weight"))?, ip, d),
            dt_bias: src.read(&p("mixer.dt_bias"))?,
            b_norm_weight: src.read(&p("mixer.B_norm.weight"))?,
            c_norm_weight: src.read(&p("mixer.C_norm.weight"))?,
            b_bias: src.read(&p("mixer.B_bias"))?,
            c_bias: src.read(&p("mixer.C_bias"))?,
            d_param: src.read(&p("mixer.D"))?,
            norm_gate_weight: if cfg.is_outproj_norm {
                src.read(&p("mixer.norm.weight"))?
            } else {
                vec![1.0; di]
            },
            out_proj_w: transpose(&src.read(&p("mixer.out_proj.weight"))?, d, di),
        });
        if hidden > 0 {
            mlp.push(read_mlp(&mut src, i, d, hidden)?);
        }
    }
    let weights = Mamba3Weights {
        input_proj_w: Vec::new(),
        input_proj_b: Vec::new(),
        layers,
        norm_f_weight: src.read("backbone.norm_f.weight")?,
        mlp,
    };
    weights.validate(&cfg, d)?;
    let embed = src.read("backbone.embedding.weight")?;
    Ok(HfMamba3Model {
        cfg,
        weights,
        embed,
        vocab_size: hf.vocab_size,
    })
}
