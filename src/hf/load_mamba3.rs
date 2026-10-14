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

use crate::config::BlockNorm;
use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::weights::{Mamba3LayerWeights, Mamba3Weights};
use crate::mlp::GatedMlpLayerWeights;

use super::config_json::{config_json, json_f32};
use super::source::Source;

/// An official Mamba-3 checkpoint, ready for [`crate::module::MambaLM::new`]
/// or the GPU language model.
pub struct HfMamba3Model {
    pub cfg: Mamba3Config,
    /// Layer weights with an empty (identity) input projection.
    pub weights: Mamba3Weights,
    /// The embedding `[vocab_padded, d_model]`, also the head when it is
    /// tied.
    pub embed: Vec<f32>,
    /// The untied head `[d_model, vocab_size]`; `None` ties the head to the
    /// embedding.
    pub lm_head: Option<Vec<f32>>,
    /// The tokenizer's vocabulary size; rows past it are padding.
    pub vocab_size: usize,
}

/// An official Mamba-3 MIMO checkpoint, ready for the MIMO decode.
pub struct HfMamba3MimoModel {
    pub cfg: Mamba3MimoConfig,
    /// Layer weights with an empty (identity) input projection.
    pub weights: Mamba3MimoWeights,
    /// The embedding `[vocab_padded, d_model]`, also the head when it is
    /// tied.
    pub embed: Vec<f32>,
    /// The untied head `[d_model, vocab_size]`; `None` ties the head to the
    /// embedding.
    pub lm_head: Option<Vec<f32>>,
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
        Some(v) => json_f32(v).ok_or_else(|| format!("config.json: {key} is not a number")),
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
    let root: Value = config_json(json)?;
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
        return Err(
            "config.json: attention layers (attn_layer_idx) are not loaded: attention stays \
             outside mamba-rs"
                .into(),
        );
    }
    // The block norms take norm_epsilon; the mixer's B/C and gate norms
    // keep the 1e-5 the reference fixes for them (ssm_norm_eps below).
    let eps = f32_or(&root, "norm_epsilon", 1e-5)?;
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
        block_norm: BlockNorm::from_rms_norm(bool_or(&root, "rms_norm", true)?),
        rms_norm_eps: eps,
        // The reference fixes the mixer's norms at 1e-5.
        ssm_norm_eps: 1e-5,
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
            block_norm: c.block_norm,
            rms_norm_eps: c.rms_norm_eps,
            ssm_norm_eps: c.ssm_norm_eps,
            d_intermediate: c.d_intermediate,
            scan_mode: c.scan_mode,
        };
        cfg.validate()?;
        Ok(Some(cfg))
    }
}

/// Transpose a PyTorch `[out, in]` weight to this crate's `[in, out]`, in
/// square tiles so both sides of a large head stream through the cache.
pub(super) fn transpose(src: &[f32], out: usize, inp: usize) -> Vec<f32> {
    const TILE: usize = 32;
    let mut dst = vec![0.0f32; src.len()];
    for o0 in (0..out).step_by(TILE) {
        let o1 = (o0 + TILE).min(out);
        for i0 in (0..inp).step_by(TILE) {
            let i1 = (i0 + TILE).min(inp);
            for o in o0..o1 {
                let row = &src[o * inp + i0..o * inp + i1];
                for (i, &v) in (i0..i1).zip(row) {
                    dst[i * out + o] = v;
                }
            }
        }
    }
    dst
}

/// An untied head, `[vocab_rows, d]` in the checkpoint, as the language
/// models take it: `[d, vocab]`, the padding rows past the vocabulary
/// dropped (they never win sampling).
pub(super) fn read_untied_head(
    src: &mut Source,
    (vocab, d): (usize, usize),
) -> Result<Vec<f32>, String> {
    let rows = src.read("lm_head.weight")?;
    let len = vocab
        .checked_mul(d)
        .filter(|&len| len <= rows.len())
        .ok_or("lm_head.weight holds fewer rows than the vocabulary")?;
    Ok(transpose(&rows[..len], vocab, d))
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
    for name in src.names() {
        // A tied head saved beside the embedding is the embedding again; an
        // untied one is in `expect` at its own shape.
        if name == "lm_head.weight" && tie_embeddings {
            continue;
        }
        let shape = src.shape(name).unwrap_or_default();
        match expect.get(name) {
            Some(want) if want.as_slice() == shape => seen += 1,
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

/// The name the original mamba_ssm format gives the embedding.
pub(super) const EMBEDDING: &str = "backbone.embedding.weight";

/// The tensors every layer and the model around them carry, whatever the
/// mixer: the embedding (under `embedding`), the head when it is untied,
/// the final norm, the block norms and the MLP.
/// The tensors every original-format checkpoint holds around its mixers:
/// the embedding and the head, the block norms (a LayerNorm's with its
/// bias) and the gated MLPs.
pub(super) fn expect_common(
    expect: &mut HashMap<String, Vec<usize>>,
    (n_layers, d, hidden, vocab_padded): (usize, usize, usize, usize),
    (tie_embeddings, embedding): (bool, &str),
    norm: BlockNorm,
) {
    expect.insert(embedding.into(), vec![vocab_padded, d]);
    if !tie_embeddings {
        expect.insert("lm_head.weight".into(), vec![vocab_padded, d]);
    }
    let mut norm_at = |name: String| {
        if !norm.rms_norm() {
            expect.insert(format!("{name}.bias"), vec![d]);
        }
        expect.insert(format!("{name}.weight"), vec![d]);
    };
    norm_at("backbone.norm_f".into());
    for i in 0..n_layers {
        norm_at(format!("backbone.layers.{i}.norm"));
        if hidden > 0 {
            norm_at(format!("backbone.layers.{i}.norm2"));
        }
    }
    for i in (0..n_layers).filter(|_| hidden > 0) {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        expect.insert(p("mlp.fc1.weight"), vec![2 * hidden, d]);
        expect.insert(p("mlp.fc2.weight"), vec![d, hidden]);
    }
}

/// A block norm's bias as the checkpoint stores it: LayerNorm's, empty for
/// RMSNorm.
pub(super) fn read_norm_bias(
    src: &mut Source,
    name: &str,
    norm: BlockNorm,
) -> Result<Vec<f32>, String> {
    if norm.rms_norm() {
        Ok(Vec::new())
    } else {
        src.read(&format!("{name}.bias"))
    }
}

pub(super) fn read_mlp(
    src: &mut Source,
    i: usize,
    (d, hidden): (usize, usize),
    norm: BlockNorm,
) -> Result<GatedMlpLayerWeights, String> {
    let p = |n: &str| format!("backbone.layers.{i}.{n}");
    Ok(GatedMlpLayerWeights {
        norm_weight: src.read(&p("norm2.weight"))?,
        norm_bias: read_norm_bias(src, &p("norm2"), norm)?,
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
    expect_common(
        &mut expect,
        (cfg.n_layers, d, hidden, hf.vocab_padded),
        (hf.tie_embeddings, EMBEDDING),
        cfg.block_norm,
    );
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
        let norm_bias = read_norm_bias(&mut src, &p("norm"), cfg.block_norm)?;
        layers.push(Mamba3MimoLayerWeights {
            norm_weight: src.read(&p("norm.weight"))?,
            norm_bias,
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
            mlp.push(read_mlp(&mut src, i, (d, hidden), cfg.block_norm)?);
        }
    }
    let weights = Mamba3MimoWeights {
        input_proj_w: Vec::new(),
        input_proj_b: Vec::new(),
        layers,
        norm_f_weight: src.read("backbone.norm_f.weight")?,
        norm_f_bias: read_norm_bias(&mut src, "backbone.norm_f", cfg.block_norm)?,
        mlp,
    };
    weights.validate(&cfg, d)?;
    let embed = src.read(EMBEDDING)?;
    let lm_head = (!hf.tie_embeddings)
        .then(|| read_untied_head(&mut src, (hf.vocab_size, d)))
        .transpose()?;
    Ok(HfMamba3MimoModel {
        cfg,
        weights,
        embed,
        lm_head,
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
    expect_common(
        &mut expect,
        (cfg.n_layers, d, hidden, hf.vocab_padded),
        (hf.tie_embeddings, EMBEDDING),
        cfg.block_norm,
    );
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
        let norm_bias = read_norm_bias(&mut src, &p("norm"), cfg.block_norm)?;
        layers.push(Mamba3LayerWeights {
            norm_weight: src.read(&p("norm.weight"))?,
            norm_bias,
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
            mlp.push(read_mlp(&mut src, i, (d, hidden), cfg.block_norm)?);
        }
    }
    let weights = Mamba3Weights {
        input_proj_w: Vec::new(),
        input_proj_b: Vec::new(),
        layers,
        norm_f_weight: src.read("backbone.norm_f.weight")?,
        norm_f_bias: read_norm_bias(&mut src, "backbone.norm_f", cfg.block_norm)?,
        mlp,
    };
    weights.validate(&cfg, d)?;
    let embed = src.read(EMBEDDING)?;
    let lm_head = (!hf.tie_embeddings)
        .then(|| read_untied_head(&mut src, (hf.vocab_size, d)))
        .transpose()?;
    Ok(HfMamba3Model {
        cfg,
        weights,
        embed,
        lm_head,
        vocab_size: hf.vocab_size,
    })
}
