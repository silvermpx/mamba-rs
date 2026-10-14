//! Safetensors checkpoints of Mamba-2 weights.
//!
//! The file names every tensor after its place in the model:
//! - `input_proj.weight`, `input_proj.bias` (empty for a model without an
//!   input projection)
//! - `layers.{i}.norm.weight`, `layers.{i}.norm.bias` (LayerNorm models)
//! - `layers.{i}.in_proj.weight`, `layers.{i}.in_proj.bias`
//! - `layers.{i}.conv1d.weight`, `layers.{i}.conv1d.bias`
//! - `layers.{i}.dt_bias`, `layers.{i}.A_log`, `layers.{i}.D`
//! - `layers.{i}.gate_norm.weight`
//! - `layers.{i}.out_proj.weight`, `layers.{i}.out_proj.bias`
//! - `layers.{i}.norm2.weight`, `layers.{i}.norm2.bias`,
//!   `layers.{i}.mlp.fc1.weight`, `layers.{i}.mlp.fc2.weight` (models with
//!   a gated MLP; the bias for LayerNorm models)
//! - `norm_f.weight`, `norm_f.bias` (LayerNorm models)
//!
//! The metadata holds the whole configuration and the input width, so
//! [`load`] needs nothing but the file. Absent projection and conv biases
//! are empty tensors; an RMSNorm model writes no norm bias and no
//! `rms_norm` entry, so its files keep the bytes they had before
//! LayerNorm models existed.

use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::SafeTensors;

use super::config::Mamba2Config;
use super::weights::{Mamba2LayerWeights, Mamba2Weights};
use crate::mlp::GatedMlpLayerWeights;
use crate::ops::st_io::{
    TensorList, block_norm, block_norm_entry, meta, read_f32, read_meta, scan_mode,
};

/// The family marker in the metadata.
const VERSION: &str = "2";

/// Save the weights, the configuration and the input width to a
/// safetensors file.
pub fn save(
    path: &Path,
    weights: &Mamba2Weights,
    cfg: &Mamba2Config,
    input_dim: usize,
) -> Result<(), String> {
    let (list, meta) = checkpoint_parts(weights, cfg, input_dim)?;
    list.save(meta, path)
}

/// The tensors and the metadata [`save`] writes; the language-model
/// checkpoint adds its embedding and head to them.
pub(crate) fn checkpoint_parts<'a>(
    weights: &'a Mamba2Weights,
    cfg: &Mamba2Config,
    input_dim: usize,
) -> Result<(TensorList<'a>, HashMap<String, String>), String> {
    weights.validate(cfg, input_dim)?;
    let dm = cfg.d_model;
    let mut list = TensorList::new();
    list.push(
        "input_proj.weight".into(),
        &weights.input_proj_w,
        &[input_dim, dm],
    )?;
    list.push("input_proj.bias".into(), &weights.input_proj_b, &[dm])?;
    for (i, lw) in weights.layers.iter().enumerate() {
        let p = |n: &str| format!("layers.{i}.{n}");
        let rows = |v: &[f32]| v.len() / dm;
        list.push(p("norm.weight"), &lw.norm_weight, &[dm])?;
        if !lw.norm_bias.is_empty() {
            list.push(p("norm.bias"), &lw.norm_bias, &[dm])?;
        }
        list.push(
            p("in_proj.weight"),
            &lw.in_proj_w,
            &[dm, rows(&lw.in_proj_w)],
        )?;
        list.push(p("in_proj.bias"), &lw.in_proj_b, &[lw.in_proj_b.len()])?;
        list.push(p("conv1d.weight"), &lw.conv_w, &[lw.conv_w.len()])?;
        list.push(p("conv1d.bias"), &lw.conv_b, &[lw.conv_b.len()])?;
        list.push(p("dt_bias"), &lw.dt_bias, &[lw.dt_bias.len()])?;
        list.push(p("A_log"), &lw.a_log, &[lw.a_log.len()])?;
        list.push(p("D"), &lw.d_param, &[lw.d_param.len()])?;
        list.push(
            p("gate_norm.weight"),
            &lw.gate_norm_weight,
            &[lw.gate_norm_weight.len()],
        )?;
        list.push(
            p("out_proj.weight"),
            &lw.out_proj_w,
            &[rows(&lw.out_proj_w), dm],
        )?;
        list.push(p("out_proj.bias"), &lw.out_proj_b, &[lw.out_proj_b.len()])?;
    }
    let hidden = cfg.mlp_hidden();
    for (i, m) in weights.mlp.iter().enumerate() {
        let p = |n: &str| format!("layers.{i}.{n}");
        list.push(p("norm2.weight"), &m.norm_weight, &[dm])?;
        if !m.norm_bias.is_empty() {
            list.push(p("norm2.bias"), &m.norm_bias, &[dm])?;
        }
        list.push(p("mlp.fc1.weight"), &m.fc1_w, &[dm, 2 * hidden])?;
        list.push(p("mlp.fc2.weight"), &m.fc2_w, &[hidden, dm])?;
    }
    list.push("norm_f.weight".into(), &weights.norm_f_weight, &[dm])?;
    if !weights.norm_f_bias.is_empty() {
        list.push("norm_f.bias".into(), &weights.norm_f_bias, &[dm])?;
    }

    let mut meta = HashMap::from([
        ("mamba_version".to_string(), VERSION.to_string()),
        ("input_dim".to_string(), input_dim.to_string()),
        ("d_model".to_string(), cfg.d_model.to_string()),
        ("d_state".to_string(), cfg.d_state.to_string()),
        ("d_conv".to_string(), cfg.d_conv.to_string()),
        ("expand".to_string(), cfg.expand.to_string()),
        ("headdim".to_string(), cfg.headdim.to_string()),
        ("d_ssm".to_string(), cfg.d_ssm.to_string()),
        ("ngroups".to_string(), cfg.ngroups.to_string()),
        ("n_layers".to_string(), cfg.n_layers.to_string()),
        ("d_has_hdim".to_string(), cfg.d_has_hdim.to_string()),
        ("rmsnorm".to_string(), cfg.rmsnorm.to_string()),
        (
            "norm_before_gate".to_string(),
            cfg.norm_before_gate.to_string(),
        ),
        ("dt_limit_min".to_string(), cfg.dt_limit.0.to_string()),
        ("dt_limit_max".to_string(), cfg.dt_limit.1.to_string()),
        ("proj_bias".to_string(), cfg.proj_bias.to_string()),
        ("conv_bias".to_string(), cfg.conv_bias.to_string()),
        ("rms_norm_eps".to_string(), cfg.rms_norm_eps.to_string()),
        ("gated_norm_eps".to_string(), cfg.gated_norm_eps.to_string()),
        ("d_intermediate".to_string(), cfg.d_intermediate.to_string()),
        ("scan_mode".to_string(), format!("{:?}", cfg.scan_mode)),
    ]);
    meta.extend(block_norm_entry(cfg.block_norm));
    Ok((list, meta))
}

/// Load a checkpoint [`save`] wrote: the weights, the configuration and the
/// input width.
pub fn load(path: &Path) -> Result<(Mamba2Weights, Mamba2Config, usize), String> {
    let data = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    load_from_bytes(&data)
}

/// [`load`] from the bytes of a checkpoint.
pub fn load_from_bytes(data: &[u8]) -> Result<(Mamba2Weights, Mamba2Config, usize), String> {
    let m = read_meta(data)?;
    if m.get("mamba_version").map(String::as_str) != Some(VERSION) {
        return Err(format!(
            "not a Mamba-2 checkpoint (mamba_version = {:?})",
            m.get("mamba_version")
        ));
    }
    let cfg = Mamba2Config {
        d_model: meta(&m, "d_model")?,
        d_state: meta(&m, "d_state")?,
        d_conv: meta(&m, "d_conv")?,
        expand: meta(&m, "expand")?,
        headdim: meta(&m, "headdim")?,
        d_ssm: meta(&m, "d_ssm")?,
        ngroups: meta(&m, "ngroups")?,
        n_layers: meta(&m, "n_layers")?,
        d_has_hdim: meta(&m, "d_has_hdim")?,
        rmsnorm: meta(&m, "rmsnorm")?,
        norm_before_gate: meta(&m, "norm_before_gate")?,
        dt_limit: (meta(&m, "dt_limit_min")?, meta(&m, "dt_limit_max")?),
        proj_bias: meta(&m, "proj_bias")?,
        conv_bias: meta(&m, "conv_bias")?,
        block_norm: block_norm(&m)?,
        rms_norm_eps: meta(&m, "rms_norm_eps")?,
        gated_norm_eps: meta(&m, "gated_norm_eps")?,
        d_intermediate: meta(&m, "d_intermediate")?,
        scan_mode: scan_mode(&m)?,
    };
    cfg.validate()?;
    let input_dim: usize = meta(&m, "input_dim")?;

    let st = SafeTensors::deserialize(data).map_err(|e| format!("deserialize: {e:?}"))?;
    let get = |name: &str| read_f32(&st, name);
    // A norm's bias, which only a LayerNorm model's file holds.
    let layer_norm = !cfg.block_norm.rms_norm();
    let norm_bias = |name: &str| {
        if layer_norm {
            get(name)
        } else {
            Ok(Vec::new())
        }
    };
    let layers = (0..cfg.n_layers)
        .map(|i| {
            let p = |n: &str| format!("layers.{i}.{n}");
            Ok(Mamba2LayerWeights {
                norm_weight: get(&p("norm.weight"))?,
                norm_bias: norm_bias(&p("norm.bias"))?,
                in_proj_w: get(&p("in_proj.weight"))?,
                in_proj_b: get(&p("in_proj.bias"))?,
                conv_w: get(&p("conv1d.weight"))?,
                conv_b: get(&p("conv1d.bias"))?,
                dt_bias: get(&p("dt_bias"))?,
                a_log: get(&p("A_log"))?,
                d_param: get(&p("D"))?,
                gate_norm_weight: get(&p("gate_norm.weight"))?,
                out_proj_w: get(&p("out_proj.weight"))?,
                out_proj_b: get(&p("out_proj.bias"))?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mlp = if cfg.mlp_hidden() == 0 {
        Vec::new()
    } else {
        (0..cfg.n_layers)
            .map(|i| {
                let p = |n: &str| format!("layers.{i}.{n}");
                Ok(GatedMlpLayerWeights {
                    norm_weight: get(&p("norm2.weight"))?,
                    norm_bias: norm_bias(&p("norm2.bias"))?,
                    fc1_w: get(&p("mlp.fc1.weight"))?,
                    fc2_w: get(&p("mlp.fc2.weight"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?
    };
    let weights = Mamba2Weights {
        input_proj_w: get("input_proj.weight")?,
        input_proj_b: get("input_proj.bias")?,
        layers,
        norm_f_weight: get("norm_f.weight")?,
        norm_f_bias: norm_bias("norm_f.bias")?,
        mlp,
    };
    weights.validate(&cfg, input_dim)?;
    Ok((weights, cfg, input_dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(cfg: Mamba2Config, input_dim: usize, name: &str) {
        let mut w = Mamba2Weights::init(&cfg, input_dim, 9);
        // Distinct norm biases, so a bias read from the wrong tensor shows.
        let mut k = 0.0f32;
        let mut fill = |v: &mut Vec<f32>| {
            for b in v.iter_mut() {
                k += 0.01;
                *b = k;
            }
        };
        for lw in &mut w.layers {
            fill(&mut lw.norm_bias);
        }
        for m in &mut w.mlp {
            fill(&mut m.norm_bias);
        }
        fill(&mut w.norm_f_bias);
        let tag: String = name.chars().filter(char::is_ascii_alphanumeric).collect();
        let path = std::env::temp_dir().join(format!(
            "mamba2_serialize_{tag}_{}.safetensors",
            std::process::id()
        ));
        save(&path, &w, &cfg, input_dim).unwrap();
        // An RMSNorm model's file names no norm kind and no norm bias.
        let bytes = std::fs::read(&path).unwrap();
        let rms = cfg.block_norm.rms_norm();
        assert_eq!(
            read_meta(&bytes).unwrap().contains_key("rms_norm"),
            !rms,
            "{name}"
        );
        let st = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(st.tensor("norm_f.bias").is_ok(), !rms, "{name}");
        let (w2, cfg2, input_dim2) = load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(cfg2, cfg, "{name}");
        assert_eq!(input_dim2, input_dim, "{name}");
        assert_eq!(w2, w, "{name}");
    }

    /// Every configuration field and every tensor come back as written:
    /// the defaults, every switch flipped with an MLP, and a model without
    /// an input projection.
    #[test]
    fn checkpoints_round_trip() {
        let base = Mamba2Config {
            d_model: 64,
            d_state: 16,
            headdim: 16,
            n_layers: 2,
            ..Mamba2Config::default()
        };
        roundtrip(base, 40, "base");
        roundtrip(
            Mamba2Config {
                d_model: 96,
                headdim: 24,
                d_state: 20,
                ngroups: 2,
                d_conv: 3,
                d_has_hdim: true,
                dt_limit: (0.01, 0.2),
                proj_bias: true,
                rms_norm_eps: 1e-6,
                gated_norm_eps: 2e-5,
                scan_mode: crate::config::ScanMode::Sequential,
                ..base
            },
            40,
            "groups, D per channel, clamped dt, biases",
        );
        roundtrip(
            Mamba2Config {
                norm_before_gate: true,
                conv_bias: false,
                d_intermediate: 96,
                ..base
            },
            40,
            "norm before gate, no conv bias, MLP",
        );
        roundtrip(
            Mamba2Config {
                block_norm: crate::config::BlockNorm::Layer,
                d_intermediate: 96,
                ..base
            },
            40,
            "LayerNorm block norms with an MLP",
        );
        roundtrip(
            Mamba2Config {
                rmsnorm: false,
                d_ssm: 64,
                ..base
            },
            40,
            "no norm, inner MLP",
        );
        let mut identity = Mamba2Weights::init(&base, base.d_model, 3);
        identity.input_proj_w.clear();
        identity.input_proj_b.clear();
        let path = std::env::temp_dir().join(format!(
            "mamba2_serialize_identity_{}.safetensors",
            std::process::id()
        ));
        save(&path, &identity, &base, base.d_model).unwrap();
        let (w2, _, input_dim) = load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(input_dim, base.d_model);
        assert_eq!(w2, identity);
    }

    /// Another family's checkpoint is refused by name.
    #[test]
    fn other_families_are_refused() {
        let cfg = crate::mamba3_mimo::config::Mamba3MimoConfig {
            d_model: 64,
            n_layers: 1,
            ..Default::default()
        };
        let w = crate::mamba3_mimo::weights::Mamba3MimoWeights::init(&cfg, 64, 1);
        let path = std::env::temp_dir().join(format!(
            "mamba2_serialize_other_{}.safetensors",
            std::process::id()
        ));
        crate::mamba3_mimo::serialize::save(&path, &w, &cfg, 64).unwrap();
        let err = load(&path).map(|_| ()).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("not a Mamba-2 checkpoint"), "{err}");
    }
}
