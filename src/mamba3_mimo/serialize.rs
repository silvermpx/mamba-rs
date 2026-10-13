//! Safetensors checkpoints of Mamba-3 MIMO weights.
//!
//! The file names every tensor after its place in the model:
//! - `input_proj.weight`, `input_proj.bias` (empty for a model without an
//!   input projection)
//! - `layers.{i}.norm.weight`
//! - `layers.{i}.in_proj.weight`
//! - `layers.{i}.dt_bias`
//! - `layers.{i}.b_norm.weight`, `layers.{i}.c_norm.weight`
//! - `layers.{i}.b_bias`, `layers.{i}.c_bias`
//! - `layers.{i}.D`
//! - `layers.{i}.mimo_x`, `layers.{i}.mimo_z`, `layers.{i}.mimo_o`
//! - `layers.{i}.norm_gate.weight`
//! - `layers.{i}.out_proj.weight`
//! - `layers.{i}.norm2.weight`, `layers.{i}.mlp.fc1.weight`,
//!   `layers.{i}.mlp.fc2.weight` (models with a gated MLP)
//! - `norm_f.weight`
//!
//! The metadata holds the whole configuration and the input width, so
//! [`load`] needs nothing but the file.

use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::SafeTensors;

use super::config::Mamba3MimoConfig;
use super::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use crate::mlp::GatedMlpLayerWeights;
use crate::ops::st_io::{TensorList, meta, read_f32, read_meta, scan_mode};

/// The family marker in the metadata. The SISO checkpoints carry "3".
const VERSION: &str = "3-mimo";

/// Save the weights, the configuration and the input width to a
/// safetensors file.
pub fn save(
    path: &Path,
    weights: &Mamba3MimoWeights,
    cfg: &Mamba3MimoConfig,
    input_dim: usize,
) -> Result<(), String> {
    let (list, meta) = checkpoint_parts(weights, cfg, input_dim)?;
    list.save(meta, path)
}

/// The tensors and the metadata [`save`] writes; the language-model
/// checkpoint adds its embedding and head to them.
pub(crate) fn checkpoint_parts<'a>(
    weights: &'a Mamba3MimoWeights,
    cfg: &Mamba3MimoConfig,
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
        let flat = |v: &[f32]| [v.len()];
        list.push(p("norm.weight"), &lw.norm_weight, &[dm])?;
        list.push(
            p("in_proj.weight"),
            &lw.in_proj_w,
            &[dm, lw.in_proj_w.len() / dm],
        )?;
        list.push(p("dt_bias"), &lw.dt_bias, &flat(&lw.dt_bias))?;
        list.push(
            p("b_norm.weight"),
            &lw.b_norm_weight,
            &flat(&lw.b_norm_weight),
        )?;
        list.push(
            p("c_norm.weight"),
            &lw.c_norm_weight,
            &flat(&lw.c_norm_weight),
        )?;
        list.push(p("b_bias"), &lw.b_bias, &flat(&lw.b_bias))?;
        list.push(p("c_bias"), &lw.c_bias, &flat(&lw.c_bias))?;
        list.push(p("D"), &lw.d_param, &flat(&lw.d_param))?;
        list.push(p("mimo_x"), &lw.mimo_x, &flat(&lw.mimo_x))?;
        list.push(p("mimo_z"), &lw.mimo_z, &flat(&lw.mimo_z))?;
        list.push(p("mimo_o"), &lw.mimo_o, &flat(&lw.mimo_o))?;
        list.push(
            p("norm_gate.weight"),
            &lw.norm_gate_weight,
            &flat(&lw.norm_gate_weight),
        )?;
        list.push(
            p("out_proj.weight"),
            &lw.out_proj_w,
            &[lw.out_proj_w.len() / dm, dm],
        )?;
    }
    let hidden = cfg.mlp_hidden();
    for (i, m) in weights.mlp.iter().enumerate() {
        let p = |n: &str| format!("layers.{i}.{n}");
        list.push(p("norm2.weight"), &m.norm_weight, &[dm])?;
        list.push(p("mlp.fc1.weight"), &m.fc1_w, &[dm, 2 * hidden])?;
        list.push(p("mlp.fc2.weight"), &m.fc2_w, &[hidden, dm])?;
    }
    list.push("norm_f.weight".into(), &weights.norm_f_weight, &[dm])?;

    let meta = HashMap::from([
        ("mamba_version".to_string(), VERSION.to_string()),
        ("input_dim".to_string(), input_dim.to_string()),
        ("d_model".to_string(), cfg.d_model.to_string()),
        ("d_state".to_string(), cfg.d_state.to_string()),
        ("expand".to_string(), cfg.expand.to_string()),
        ("headdim".to_string(), cfg.headdim.to_string()),
        ("ngroups".to_string(), cfg.ngroups.to_string()),
        ("n_layers".to_string(), cfg.n_layers.to_string()),
        ("mimo_rank".to_string(), cfg.mimo_rank.to_string()),
        ("rope_fraction".to_string(), cfg.rope_fraction.to_string()),
        ("a_floor".to_string(), cfg.a_floor.to_string()),
        (
            "is_outproj_norm".to_string(),
            cfg.is_outproj_norm.to_string(),
        ),
        ("rms_norm_eps".to_string(), cfg.rms_norm_eps.to_string()),
        ("ssm_norm_eps".to_string(), cfg.ssm_norm_eps.to_string()),
        ("d_intermediate".to_string(), cfg.d_intermediate.to_string()),
        ("scan_mode".to_string(), format!("{:?}", cfg.scan_mode)),
    ]);
    Ok((list, meta))
}

/// Load a checkpoint [`save`] wrote: the weights, the configuration and the
/// input width.
pub fn load(path: &Path) -> Result<(Mamba3MimoWeights, Mamba3MimoConfig, usize), String> {
    let data = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    load_from_bytes(&data)
}

/// [`load`] from the bytes of a checkpoint.
pub fn load_from_bytes(
    data: &[u8],
) -> Result<(Mamba3MimoWeights, Mamba3MimoConfig, usize), String> {
    let m = read_meta(data)?;
    if m.get("mamba_version").map(String::as_str) != Some(VERSION) {
        return Err(format!(
            "not a Mamba-3 MIMO checkpoint (mamba_version = {:?})",
            m.get("mamba_version")
        ));
    }
    let cfg = Mamba3MimoConfig {
        d_model: meta(&m, "d_model")?,
        d_state: meta(&m, "d_state")?,
        expand: meta(&m, "expand")?,
        headdim: meta(&m, "headdim")?,
        ngroups: meta(&m, "ngroups")?,
        n_layers: meta(&m, "n_layers")?,
        mimo_rank: meta(&m, "mimo_rank")?,
        rope_fraction: meta(&m, "rope_fraction")?,
        a_floor: meta(&m, "a_floor")?,
        is_outproj_norm: meta(&m, "is_outproj_norm")?,
        rms_norm_eps: meta(&m, "rms_norm_eps")?,
        ssm_norm_eps: meta(&m, "ssm_norm_eps")?,
        d_intermediate: meta(&m, "d_intermediate")?,
        scan_mode: scan_mode(&m)?,
    };
    cfg.validate()?;
    let input_dim: usize = meta(&m, "input_dim")?;

    let st = SafeTensors::deserialize(data).map_err(|e| format!("deserialize: {e:?}"))?;
    let get = |name: &str| read_f32(&st, name);
    let layers = (0..cfg.n_layers)
        .map(|i| {
            let p = |n: &str| format!("layers.{i}.{n}");
            Ok(Mamba3MimoLayerWeights {
                norm_weight: get(&p("norm.weight"))?,
                in_proj_w: get(&p("in_proj.weight"))?,
                dt_bias: get(&p("dt_bias"))?,
                b_norm_weight: get(&p("b_norm.weight"))?,
                c_norm_weight: get(&p("c_norm.weight"))?,
                b_bias: get(&p("b_bias"))?,
                c_bias: get(&p("c_bias"))?,
                d_param: get(&p("D"))?,
                mimo_x: get(&p("mimo_x"))?,
                mimo_z: get(&p("mimo_z"))?,
                mimo_o: get(&p("mimo_o"))?,
                norm_gate_weight: get(&p("norm_gate.weight"))?,
                out_proj_w: get(&p("out_proj.weight"))?,
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
                    fc1_w: get(&p("mlp.fc1.weight"))?,
                    fc2_w: get(&p("mlp.fc2.weight"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?
    };
    let weights = Mamba3MimoWeights {
        input_proj_w: get("input_proj.weight")?,
        input_proj_b: get("input_proj.bias")?,
        layers,
        norm_f_weight: get("norm_f.weight")?,
        mlp,
    };
    weights.validate(&cfg, input_dim)?;
    Ok((weights, cfg, input_dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(cfg: Mamba3MimoConfig, input_dim: usize, name: &str) {
        let w = Mamba3MimoWeights::init(&cfg, input_dim, 9);
        let tag: String = name.chars().filter(char::is_ascii_alphanumeric).collect();
        let path = std::env::temp_dir().join(format!(
            "mimo_serialize_{tag}_{}.safetensors",
            std::process::id()
        ));
        save(&path, &w, &cfg, input_dim).unwrap();
        let (w2, cfg2, input_dim2) = load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(cfg2, cfg, "{name}");
        assert_eq!(input_dim2, input_dim, "{name}");
        assert_eq!(w2, w, "{name}");
    }

    /// Every configuration field and every tensor come back as written:
    /// the defaults, the other ranks with groups, an output norm and an
    /// MLP, and a model without an input projection.
    #[test]
    fn checkpoints_round_trip() {
        let base = Mamba3MimoConfig {
            d_model: 64,
            d_state: 16,
            headdim: 16,
            n_layers: 2,
            ..Mamba3MimoConfig::default()
        };
        roundtrip(base, 40, "rank 4");
        roundtrip(
            Mamba3MimoConfig {
                mimo_rank: 2,
                ngroups: 2,
                d_state: 24,
                ..base
            },
            40,
            "rank 2, groups 2",
        );
        roundtrip(
            Mamba3MimoConfig {
                mimo_rank: 1,
                rope_fraction: 1.0,
                d_intermediate: 160,
                is_outproj_norm: true,
                rms_norm_eps: 1e-6,
                ssm_norm_eps: 2e-5,
                scan_mode: crate::config::ScanMode::Parallel,
                ..base
            },
            40,
            "rank 1, full rope, MLP, output norm",
        );
        let mut identity = Mamba3MimoWeights::init(&base, base.d_model, 3);
        identity.input_proj_w.clear();
        identity.input_proj_b.clear();
        let path = std::env::temp_dir().join(format!(
            "mimo_serialize_identity_{}.safetensors",
            std::process::id()
        ));
        save(&path, &identity, &base, base.d_model).unwrap();
        let (w2, _, input_dim) = load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(input_dim, base.d_model);
        assert_eq!(w2, identity);
    }

    /// The SISO loader refuses a MIMO checkpoint, which shares its
    /// generation number.
    #[test]
    fn the_siso_loader_refuses_it() {
        let cfg = Mamba3MimoConfig {
            d_model: 64,
            n_layers: 1,
            ..Mamba3MimoConfig::default()
        };
        let w = Mamba3MimoWeights::init(&cfg, 64, 1);
        let path = std::env::temp_dir().join(format!(
            "mimo_serialize_siso_{}.safetensors",
            std::process::id()
        ));
        save(&path, &w, &cfg, 64).unwrap();
        let err = crate::mamba3_siso::serialize::load(&path)
            .map(|_| ())
            .unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("not a Mamba-3"), "{err}");
    }
}
