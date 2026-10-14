//! Loading Mamba-2 checkpoints in either layout they ship in.
//!
//! The official state-spaces/mamba2-* repositories keep `config.json` in
//! the original mamba_ssm format (the mixer settings under `ssm_cfg`,
//! `"layer": "Mamba2"`) and a `pytorch_model.bin`. The transformers layout
//! (`"model_type": "mamba2"`: mistralai/Mamba-Codestral-7B-v0.1, the
//! converted mamba2-*-hf) names the fields its own way and the embedding
//! `backbone.embeddings.weight`, and ships safetensors, often sharded and
//! with an untied head. Neither has an input projection. The key map is as
//! strict as the Mamba-3 loader's: every tensor the model uses must be
//! present at its shape, and nothing else may be.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use super::config_json::{config_json, json_f32};
use super::load_mamba3::{
    EMBEDDING, bool_or, check_tensors, expect_common, f32_or, field, read_mlp, read_norm_bias,
    read_untied_head, transpose, usize_or,
};
use super::source::Source;
use crate::config::BlockNorm;
use crate::mamba2::config::Mamba2Config;
use crate::mamba2::weights::{Mamba2LayerWeights, Mamba2Weights};

/// An official Mamba-2 checkpoint, ready for the Mamba-2 decode.
pub struct HfMamba2Model {
    pub cfg: Mamba2Config,
    /// Layer weights with an empty (identity) input projection.
    pub weights: Mamba2Weights,
    /// The embedding `[vocab_padded, d_model]`, also the head when it is
    /// tied.
    pub embed: Vec<f32>,
    /// The untied head `[d_model, vocab_size]`; `None` ties the head to the
    /// embedding.
    pub lm_head: Option<Vec<f32>>,
    /// The tokenizer's vocabulary size; rows past it are padding.
    pub vocab_size: usize,
}

/// The parsed `config.json` of a Mamba-2 checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct HfMamba2Config {
    pub cfg: Mamba2Config,
    pub vocab_padded: usize,
    pub vocab_size: usize,
    pub tie_embeddings: bool,
    /// The name of the embedding tensor: `backbone.embedding.weight` in the
    /// original format, `backbone.embeddings.weight` in the transformers
    /// one.
    pub embedding: &'static str,
}

/// The name the transformers layout gives the embedding.
const HF_EMBEDDING: &str = "backbone.embeddings.weight";

/// Is this config a Mamba-2 one, in either layout?
pub fn is_mamba2_config(json: &[u8]) -> bool {
    config_json::<Value>(json).is_ok_and(|c| {
        c["ssm_cfg"]["layer"].as_str() == Some("Mamba2") || c["model_type"] == "mamba2"
    })
}

/// Parse the `config.json` of a Mamba-2 checkpoint in either layout. The
/// defaults are each format's own: the reference's `MambaConfig` and
/// `Mamba2.__init__` for the original one, transformers' `Mamba2Config`
/// for the other.
pub fn parse_mamba2_config(json: &[u8]) -> Result<HfMamba2Config, String> {
    let root: Value = config_json(json)?;
    if field(&root, "model_type").and_then(Value::as_str) == Some("mamba2") {
        return parse_transformers_config(&root);
    }
    let ssm = field(&root, "ssm_cfg").ok_or("config.json: no ssm_cfg")?;
    let layer = field(ssm, "layer").and_then(Value::as_str).unwrap_or("");
    if layer != "Mamba2" {
        return Err(format!(
            "config.json: ssm_cfg.layer is {layer:?}, not \"Mamba2\""
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
    let dt_limit = match field(ssm, "dt_limit") {
        None => (0.0, f32::INFINITY),
        Some(v) => {
            let pair = v
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or("config.json: dt_limit is not a pair")?;
            let num = |x: &Value| json_f32(x).ok_or("config.json: dt_limit holds a non-number");
            (num(&pair[0])?, num(&pair[1])?)
        }
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
    let cfg = Mamba2Config {
        d_model: usize_or(&root, "d_model", 2560)?,
        n_layers: usize_or(&root, "n_layer", 64)?,
        d_intermediate: usize_or(&root, "d_intermediate", 0)?,
        d_state: usize_or(ssm, "d_state", 128)?,
        d_conv: usize_or(ssm, "d_conv", 4)?,
        expand: usize_or(ssm, "expand", 2)?,
        headdim: usize_or(ssm, "headdim", 64)?,
        d_ssm: usize_or(ssm, "d_ssm", 0)?,
        ngroups: usize_or(ssm, "ngroups", 1)?,
        d_has_hdim: bool_or(ssm, "D_has_hdim", false)?,
        rmsnorm: bool_or(ssm, "rmsnorm", true)?,
        norm_before_gate: bool_or(ssm, "norm_before_gate", false)?,
        dt_limit,
        proj_bias: bool_or(ssm, "bias", false)?,
        conv_bias: bool_or(ssm, "conv_bias", true)?,
        block_norm: BlockNorm::from_rms_norm(bool_or(&root, "rms_norm", true)?),
        rms_norm_eps: f32_or(&root, "norm_epsilon", 1e-5)?,
        gated_norm_eps: 1e-5,
        scan_mode: crate::config::ScanMode::Auto,
    };
    cfg.validate()?;
    Ok(HfMamba2Config {
        cfg,
        vocab_padded,
        vocab_size,
        tie_embeddings: bool_or(&root, "tie_embeddings", true)?,
        embedding: EMBEDDING,
    })
}

/// The transformers layout's `config.json`. Its `vocab_size` is the
/// embedding's row count. Two of its fields do not describe the model the
/// weights run: `rms_norm`, since every transformers Mamba-2 has the
/// gated norm and RMSNorm block norms, and `norm_before_gate`, which
/// transformers never reads (it
/// gates before the norm as mamba_ssm does by default; Codestral's config
/// says `true` of weights that run with the gate first).
fn parse_transformers_config(root: &Value) -> Result<HfMamba2Config, String> {
    let d_model = usize_or(root, "hidden_size", 4096)?;
    let expand = usize_or(root, "expand", 2)?;
    let (heads, headdim) = (
        usize_or(root, "num_heads", 128)?,
        usize_or(root, "head_dim", 64)?,
    );
    let d_inner = d_model
        .checked_mul(expand)
        .ok_or("config.json: hidden_size * expand overflows")?;
    if heads.checked_mul(headdim) != Some(d_inner) {
        return Err(format!(
            "config.json: num_heads {heads} x head_dim {headdim} is not hidden_size {d_model} \
             x expand {expand}"
        ));
    }
    if field(root, "intermediate_size").is_some()
        && usize_or(root, "intermediate_size", 0)? != d_inner
    {
        return Err("config.json: intermediate_size is not hidden_size x expand".into());
    }
    let dt_limit = match field(root, "time_step_limit") {
        None => (0.0, f32::INFINITY),
        Some(v) => {
            let pair = v
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or("config.json: time_step_limit is not a pair")?;
            let num =
                |x: &Value| json_f32(x).ok_or("config.json: time_step_limit holds a non-number");
            (num(&pair[0])?, num(&pair[1])?)
        }
    };
    let vocab_size = usize_or(root, "vocab_size", 32768)?;
    if vocab_size == 0 || vocab_size > (1 << 24) {
        return Err(format!("config.json: vocab_size {vocab_size} out of range"));
    }
    let eps = f32_or(root, "layer_norm_epsilon", 1e-5)?;
    let cfg = Mamba2Config {
        d_model,
        n_layers: usize_or(root, "num_hidden_layers", 64)?,
        d_intermediate: 0,
        d_state: usize_or(root, "state_size", 128)?,
        d_conv: usize_or(root, "conv_kernel", 4)?,
        expand,
        headdim,
        d_ssm: 0,
        ngroups: usize_or(root, "n_groups", 8)?,
        d_has_hdim: false,
        rmsnorm: true,
        norm_before_gate: false,
        dt_limit,
        proj_bias: bool_or(root, "use_bias", false)?,
        conv_bias: bool_or(root, "use_conv_bias", true)?,
        block_norm: BlockNorm::Rms,
        rms_norm_eps: eps,
        gated_norm_eps: eps,
        scan_mode: crate::config::ScanMode::Auto,
    };
    cfg.validate()?;
    Ok(HfMamba2Config {
        cfg,
        vocab_padded: vocab_size,
        vocab_size,
        tie_embeddings: bool_or(root, "tie_word_embeddings", false)?,
        embedding: HF_EMBEDDING,
    })
}

/// Load a Mamba-2 checkpoint directory in either layout.
pub fn load_hf_mamba2(dir: &Path) -> Result<HfMamba2Model, String> {
    let json = std::fs::read(dir.join("config.json"))
        .map_err(|e| format!("cannot read config.json: {e}"))?;
    let hf = parse_mamba2_config(&json)?;
    let cfg = hf.cfg;
    let (d, di, ip, cd, nh, w) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.in_proj_out_dim(),
        cfg.conv_dim(),
        cfg.nheads(),
        cfg.d_conv,
    );
    let hidden = cfg.mlp_hidden();
    let d_len = if cfg.d_has_hdim { cfg.ssm_dim() } else { nh };
    let mut src = Source::open(dir)?;

    let mut expect: HashMap<String, Vec<usize>> = HashMap::new();
    expect_common(
        &mut expect,
        (cfg.n_layers, d, hidden, hf.vocab_padded),
        (hf.tie_embeddings, hf.embedding),
        cfg.block_norm,
    );
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("backbone.layers.{i}.mixer.{n}");
        expect.insert(p("in_proj.weight"), vec![ip, d]);
        expect.insert(p("conv1d.weight"), vec![cd, 1, w]);
        expect.insert(p("dt_bias"), vec![nh]);
        expect.insert(p("A_log"), vec![nh]);
        expect.insert(p("D"), vec![d_len]);
        expect.insert(p("out_proj.weight"), vec![d, di]);
        if cfg.conv_bias {
            expect.insert(p("conv1d.bias"), vec![cd]);
        }
        if cfg.proj_bias {
            expect.insert(p("in_proj.bias"), vec![ip]);
            expect.insert(p("out_proj.bias"), vec![d]);
        }
        if cfg.rmsnorm {
            expect.insert(p("norm.weight"), vec![cfg.ssm_dim()]);
        }
    }
    check_tensors(&src, &expect, hf.tie_embeddings)?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    let mut mlp = Vec::new();
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        let mut opt = |name: &str, on: bool| -> Result<Vec<f32>, String> {
            if on {
                src.read(&p(name))
            } else {
                Ok(Vec::new())
            }
        };
        let in_proj_b = opt("mixer.in_proj.bias", cfg.proj_bias)?;
        let out_proj_b = opt("mixer.out_proj.bias", cfg.proj_bias)?;
        let conv_b = opt("mixer.conv1d.bias", cfg.conv_bias)?;
        let gate_norm_weight = opt("mixer.norm.weight", cfg.rmsnorm)?;
        let norm_bias = read_norm_bias(&mut src, &p("norm"), cfg.block_norm)?;
        layers.push(Mamba2LayerWeights {
            norm_weight: src.read(&p("norm.weight"))?,
            norm_bias,
            in_proj_w: transpose(&src.read(&p("mixer.in_proj.weight"))?, ip, d),
            in_proj_b,
            conv_w: src.read(&p("mixer.conv1d.weight"))?,
            conv_b,
            dt_bias: src.read(&p("mixer.dt_bias"))?,
            a_log: src.read(&p("mixer.A_log"))?,
            d_param: src.read(&p("mixer.D"))?,
            gate_norm_weight,
            out_proj_w: transpose(&src.read(&p("mixer.out_proj.weight"))?, d, di),
            out_proj_b,
        });
        if hidden > 0 {
            mlp.push(read_mlp(&mut src, i, (d, hidden), cfg.block_norm)?);
        }
    }
    let weights = Mamba2Weights {
        input_proj_w: Vec::new(),
        input_proj_b: Vec::new(),
        layers,
        norm_f_weight: src.read("backbone.norm_f.weight")?,
        norm_f_bias: read_norm_bias(&mut src, "backbone.norm_f", cfg.block_norm)?,
        mlp,
    };
    weights.validate(&cfg, d)?;
    let embed = src.read(hf.embedding)?;
    let lm_head = (!hf.tie_embeddings)
        .then(|| read_untied_head(&mut src, (hf.vocab_size, d)))
        .transpose()?;
    Ok(HfMamba2Model {
        cfg,
        weights,
        embed,
        lm_head,
        vocab_size: hf.vocab_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_config_parses_with_the_reference_defaults() {
        let json = br#"{"d_model": 768, "d_intermediate": 0, "n_layer": 24,
            "vocab_size": 50277, "ssm_cfg": {"layer": "Mamba2"}, "attn_layer_idx": [],
            "rms_norm": true, "residual_in_fp32": true, "fused_add_norm": true,
            "pad_vocab_size_multiple": 16, "tie_embeddings": true}"#;
        assert!(is_mamba2_config(json));
        let hf = parse_mamba2_config(json).unwrap();
        assert_eq!(hf.vocab_padded, 50288);
        assert_eq!(hf.cfg.in_proj_out_dim(), 3352);
        assert_eq!(
            (hf.cfg.d_state, hf.cfg.headdim, hf.cfg.d_conv),
            (128, 64, 4)
        );
        assert!(hf.cfg.rmsnorm && hf.cfg.conv_bias && !hf.cfg.proj_bias);
    }
}
