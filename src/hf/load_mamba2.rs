//! Loading the official Mamba-2 checkpoints (state-spaces/mamba2-*).
//!
//! They ship as `config.json` in the original mamba_ssm format (the mixer
//! settings under `ssm_cfg`, `"layer": "Mamba2"`) and a `pytorch_model.bin`,
//! the embedding tied to the head and no input projection. The key map is
//! as strict as the Mamba-3 loader's: every tensor the model uses must be
//! present at its shape, and nothing else may be.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use super::load_mamba3::{
    Source, bool_or, check_tensors, expect_common, f32_or, field, read_mlp, transpose, usize_or,
};
use crate::mamba2::config::Mamba2Config;
use crate::mamba2::weights::{Mamba2LayerWeights, Mamba2Weights};

/// An official Mamba-2 checkpoint, ready for the Mamba-2 decode.
pub struct HfMamba2Model {
    pub cfg: Mamba2Config,
    /// Layer weights with an empty (identity) input projection.
    pub weights: Mamba2Weights,
    /// The embedding `[vocab_padded, d_model]`, also the tied head.
    pub embed: Vec<f32>,
    /// The tokenizer's vocabulary size; rows past it are padding.
    pub vocab_size: usize,
}

/// The parsed `config.json` of an official Mamba-2 checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct HfMamba2Config {
    pub cfg: Mamba2Config,
    pub vocab_padded: usize,
    pub vocab_size: usize,
    pub tie_embeddings: bool,
}

/// Is this original-format config a Mamba-2 one?
pub fn is_mamba2_config(json: &[u8]) -> bool {
    serde_json::from_slice::<Value>(json)
        .ok()
        .is_some_and(|c| c["ssm_cfg"]["layer"].as_str() == Some("Mamba2"))
}

/// Parse the original-format `config.json` of a Mamba-2 checkpoint. The
/// defaults are the reference's own (`MambaConfig` and `Mamba2.__init__`).
pub fn parse_mamba2_config(json: &[u8]) -> Result<HfMamba2Config, String> {
    let root: Value =
        serde_json::from_slice(json).map_err(|e| format!("config.json parse error: {e}"))?;
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
        return Err("config.json: hybrid attention layers are not supported".into());
    }
    if !bool_or(&root, "rms_norm", true)? {
        return Err("config.json: rms_norm = false (LayerNorm) is not supported".into());
    }
    let dt_limit = match field(ssm, "dt_limit") {
        None => (0.0, f32::INFINITY),
        Some(v) => {
            let pair = v
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or("config.json: dt_limit is not a pair")?;
            let num = |x: &Value| {
                x.as_f64()
                    .map(|n| n as f32)
                    .ok_or("config.json: dt_limit holds a non-number")
            };
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
        rms_norm_eps: f32_or(&root, "norm_epsilon", 1e-5)?,
        gated_norm_eps: 1e-5,
    };
    cfg.validate()?;
    Ok(HfMamba2Config {
        cfg,
        vocab_padded,
        vocab_size,
        tie_embeddings: bool_or(&root, "tie_embeddings", true)?,
    })
}

/// Load an official Mamba-2 checkpoint directory.
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
    expect_common(&mut expect, (cfg.n_layers, d, hidden, hf.vocab_padded));
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
        layers.push(Mamba2LayerWeights {
            norm_weight: src.read(&p("norm.weight"))?,
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
            mlp.push(read_mlp(&mut src, i, d, hidden)?);
        }
    }
    let weights = Mamba2Weights {
        input_proj_w: Vec::new(),
        input_proj_b: Vec::new(),
        layers,
        norm_f_weight: src.read("backbone.norm_f.weight")?,
        mlp,
    };
    weights.validate(&cfg, d)?;
    let embed = src.read("backbone.embedding.weight")?;
    Ok(HfMamba2Model {
        cfg,
        weights,
        embed,
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
