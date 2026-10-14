//! Safetensors serialization for Mamba-3 SISO weights.
//!
//! Tensor naming convention:
//! - `input_proj.weight`, `input_proj.bias`
//! - `layers.{i}.norm.weight`, `layers.{i}.norm.bias` (LayerNorm models)
//! - `layers.{i}.in_proj.weight`
//! - `layers.{i}.dt_bias`
//! - `layers.{i}.b_norm.weight`, `layers.{i}.c_norm.weight`
//! - `layers.{i}.b_bias`, `layers.{i}.c_bias`
//! - `layers.{i}.D`
//! - `layers.{i}.norm_gate.weight`
//! - `layers.{i}.out_proj.weight`
//! - `layers.{i}.norm2.weight`, `layers.{i}.norm2.bias`,
//!   `layers.{i}.mlp.fc1.weight`, `layers.{i}.mlp.fc2.weight` (models with
//!   a gated MLP; the bias for LayerNorm models)
//! - `norm_f.weight`, `norm_f.bias` (LayerNorm models)
//!
//! An RMSNorm model writes no norm bias and no `rms_norm` entry, so its
//! files keep the bytes they had before LayerNorm models existed.

use super::config::Mamba3Config;
use super::weights::{Mamba3LayerWeights, Mamba3Weights};
use crate::ops::st_io::{TensorList, block_norm, block_norm_entry, read_f32};
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// Save the weights, the configuration and the input width to a
/// safetensors file.
pub fn save(
    path: &Path,
    weights: &Mamba3Weights,
    cfg: &Mamba3Config,
    input_dim: usize,
) -> Result<(), String> {
    let (list, meta) = checkpoint_parts(weights, cfg, input_dim)?;
    list.save(meta, path)
}

/// The tensors and the metadata [`save`] writes; the language-model
/// checkpoint adds its embedding and head to them.
pub(crate) fn checkpoint_parts<'a>(
    weights: &'a Mamba3Weights,
    cfg: &Mamba3Config,
    input_dim: usize,
) -> Result<(TensorList<'a>, HashMap<String, String>), String> {
    weights.validate(cfg, input_dim)?;
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nh = cfg.nheads();
    let ip = cfg.in_proj_out_dim();

    let mut list = TensorList::new();
    list.push(
        "input_proj.weight".into(),
        &weights.input_proj_w,
        &[input_dim, dm],
    )?;
    list.push("input_proj.bias".into(), &weights.input_proj_b, &[dm])?;
    for (i, lw) in weights.layers.iter().enumerate() {
        let p = |n: &str| format!("layers.{i}.{n}");
        list.push(p("norm.weight"), &lw.norm_weight, &[dm])?;
        if !lw.norm_bias.is_empty() {
            list.push(p("norm.bias"), &lw.norm_bias, &[dm])?;
        }
        list.push(p("in_proj.weight"), &lw.in_proj_w, &[dm, ip])?;
        list.push(p("dt_bias"), &lw.dt_bias, &[nh])?;
        list.push(p("b_norm.weight"), &lw.b_norm_weight, &[ds])?;
        list.push(p("c_norm.weight"), &lw.c_norm_weight, &[ds])?;
        list.push(p("b_bias"), &lw.b_bias, &[nh, ds])?;
        list.push(p("c_bias"), &lw.c_bias, &[nh, ds])?;
        list.push(p("D"), &lw.d_param, &[nh])?;
        list.push(p("norm_gate.weight"), &lw.norm_gate_weight, &[di])?;
        list.push(p("out_proj.weight"), &lw.out_proj_w, &[di, dm])?;
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
        ("d_model".to_string(), dm.to_string()),
        ("d_state".to_string(), ds.to_string()),
        ("expand".to_string(), cfg.expand.to_string()),
        ("headdim".to_string(), cfg.headdim.to_string()),
        ("ngroups".to_string(), cfg.ngroups.to_string()),
        ("n_layers".to_string(), cfg.n_layers.to_string()),
        ("input_dim".to_string(), input_dim.to_string()),
        ("rope_fraction".to_string(), cfg.rope_fraction.to_string()),
        ("a_floor".to_string(), cfg.a_floor.to_string()),
        // Identity-critical: omitting is_outproj_norm loads a DIFFERENT
        // model (norm_gate silently on/off); eps re-defines every norm.
        (
            "is_outproj_norm".to_string(),
            cfg.is_outproj_norm.to_string(),
        ),
        ("rms_norm_eps".to_string(), cfg.rms_norm_eps.to_string()),
        ("ssm_norm_eps".to_string(), cfg.ssm_norm_eps.to_string()),
        // Identity-critical too: without it a model with an MLP would load
        // as one without.
        ("d_intermediate".to_string(), cfg.d_intermediate.to_string()),
        ("scan_mode".to_string(), format!("{:?}", cfg.scan_mode)),
        ("mamba_version".to_string(), "3".to_string()),
    ]);
    meta.extend(block_norm_entry(cfg.block_norm));
    Ok((list, meta))
}

/// Load Mamba-3 SISO weights from safetensors with a caller-supplied config.
///
/// Identity guard: if the file carries `is_outproj_norm` / `rms_norm_eps`
/// metadata (checkpoints written since 0.6) and it CONTRADICTS `cfg`, this
/// fails loudly — silently loading with a mismatched norm topology or eps
/// produces a different model with plausible-looking outputs.
pub fn load_mamba3(path: &Path, cfg: &Mamba3Config) -> Result<(Mamba3Weights, usize), String> {
    let data = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    weights_from_bytes(&data, cfg)
}

/// [`load_mamba3`] from the bytes of a checkpoint.
fn weights_from_bytes(data: &[u8], cfg: &Mamba3Config) -> Result<(Mamba3Weights, usize), String> {
    let (_, header) = SafeTensors::read_metadata(data).map_err(|e| format!("metadata: {e}"))?;
    let meta_map = header.metadata().as_ref().ok_or("no metadata")?;
    let input_dim: usize = meta_map
        .get("input_dim")
        .ok_or("missing input_dim")?
        .parse()
        .map_err(|e| format!("parse input_dim: {e}"))?;
    if let Some(v) = meta_map.get("is_outproj_norm") {
        let stored: bool = v
            .parse()
            .map_err(|e| format!("parse is_outproj_norm: {e}"))?;
        if stored != cfg.is_outproj_norm {
            return Err(format!(
                "checkpoint identity mismatch: file has is_outproj_norm={stored}, \
                 caller config says {} — these are different models",
                cfg.is_outproj_norm
            ));
        }
    }
    if let Some(v) = meta_map.get("rms_norm_eps") {
        let stored: f32 = v.parse().map_err(|e| format!("parse rms_norm_eps: {e}"))?;
        if stored != cfg.rms_norm_eps {
            return Err(format!(
                "checkpoint identity mismatch: file has rms_norm_eps={stored}, \
                 caller config says {} — every norm site differs",
                cfg.rms_norm_eps
            ));
        }
    }
    // A file written before the mixer norms had their own epsilon ran them
    // at the block norms'.
    if let Some(v) = meta_map
        .get("ssm_norm_eps")
        .or(meta_map.get("rms_norm_eps"))
    {
        let stored: f32 = v.parse().map_err(|e| format!("parse ssm_norm_eps: {e}"))?;
        if stored != cfg.ssm_norm_eps {
            return Err(format!(
                "checkpoint identity mismatch: file has ssm_norm_eps={stored}, \
                 caller config says {} — the B/C and gate norms differ",
                cfg.ssm_norm_eps
            ));
        }
    }
    let stored_norm = block_norm(meta_map)?;
    if stored_norm != cfg.block_norm {
        return Err(format!(
            "checkpoint identity mismatch: file has block norms {stored_norm:?}, caller \
             config says {:?} — every block norm differs",
            cfg.block_norm
        ));
    }
    let stored_mlp: usize = match meta_map.get("d_intermediate") {
        Some(v) => v
            .parse()
            .map_err(|e| format!("parse d_intermediate: {e}"))?,
        None => 0,
    };
    if stored_mlp != cfg.d_intermediate {
        return Err(format!(
            "checkpoint identity mismatch: file has d_intermediate={stored_mlp}, \
             caller config says {} — the MLP differs",
            cfg.d_intermediate
        ));
    }

    let st = SafeTensors::deserialize(data).map_err(|e| format!("deserialize: {e:?}"))?;
    let get = |name: &str| read_f32(&st, name);
    // A norm's bias, which only a LayerNorm model's file holds.
    let norm_bias = |name: &str| {
        if cfg.block_norm.rms_norm() {
            Ok(Vec::new())
        } else {
            get(name)
        }
    };

    let mut layers = Vec::new();
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("layers.{i}.{n}");
        layers.push(Mamba3LayerWeights {
            norm_weight: get(&p("norm.weight"))?,
            norm_bias: norm_bias(&p("norm.bias"))?,
            in_proj_w: get(&p("in_proj.weight"))?,
            dt_bias: get(&p("dt_bias"))?,
            b_norm_weight: get(&p("b_norm.weight"))?,
            c_norm_weight: get(&p("c_norm.weight"))?,
            b_bias: get(&p("b_bias"))?,
            c_bias: get(&p("c_bias"))?,
            d_param: get(&p("D"))?,
            norm_gate_weight: get(&p("norm_gate.weight"))?,
            out_proj_w: get(&p("out_proj.weight"))?,
        });
    }

    let mlp = if cfg.mlp_hidden() == 0 {
        Vec::new()
    } else {
        (0..cfg.n_layers)
            .map(|i| {
                let p = |n: &str| format!("layers.{i}.{n}");
                Ok(crate::mlp::GatedMlpLayerWeights {
                    norm_weight: get(&p("norm2.weight"))?,
                    norm_bias: norm_bias(&p("norm2.bias"))?,
                    fc1_w: get(&p("mlp.fc1.weight"))?,
                    fc2_w: get(&p("mlp.fc2.weight"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?
    };

    let w = Mamba3Weights {
        input_proj_w: get("input_proj.weight")?,
        input_proj_b: get("input_proj.bias")?,
        layers,
        norm_f_weight: get("norm_f.weight")?,
        norm_f_bias: norm_bias("norm_f.bias")?,
        mlp,
    };

    w.validate(cfg, input_dim)?;
    Ok((w, input_dim))
}

#[deprecated(since = "0.8.0", note = "use `serialize::save`")]
pub fn save_mamba3(
    path: &Path,
    weights: &Mamba3Weights,
    cfg: &Mamba3Config,
    input_dim: usize,
) -> Result<(), String> {
    save(path, weights, cfg, input_dim)
}

/// Load a checkpoint [`save`] wrote: the weights, the configuration read
/// from the file's metadata, and the input width. The caller supplies
/// nothing, so it cannot describe the model wrongly.
///
/// Checkpoints written before 0.6 lack `is_outproj_norm`, `rms_norm_eps`
/// and `scan_mode`; they load with the values those versions fixed
/// (false, 1e-5 and Auto), the model they trained as.
pub fn load(path: &Path) -> Result<(Mamba3Weights, Mamba3Config, usize), String> {
    let data = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    load_from_bytes(&data)
}

/// [`load`] from the bytes of a checkpoint.
pub fn load_from_bytes(data: &[u8]) -> Result<(Mamba3Weights, Mamba3Config, usize), String> {
    let (_, header) = SafeTensors::read_metadata(data).map_err(|e| format!("metadata: {e}"))?;
    let meta_map = header.metadata().as_ref().ok_or("no metadata")?;

    if meta_map.get("mamba_version").map(String::as_str) != Some("3") {
        return Err(format!(
            "not a Mamba-3 checkpoint (mamba_version = {:?})",
            meta_map.get("mamba_version")
        ));
    }

    fn req<T: std::str::FromStr>(
        meta: &std::collections::HashMap<String, String>,
        key: &str,
    ) -> Result<T, String>
    where
        T::Err: std::fmt::Display,
    {
        meta.get(key)
            .ok_or(format!("missing '{key}' in checkpoint metadata"))?
            .parse()
            .map_err(|e| format!("parse '{key}': {e}"))
    }

    let scan_mode = match meta_map.get("scan_mode").map(String::as_str) {
        Some("Sequential") => crate::config::ScanMode::Sequential,
        Some("Parallel") => crate::config::ScanMode::Parallel,
        Some("Auto") | None => crate::config::ScanMode::Auto,
        Some(other) => return Err(format!("unknown scan_mode '{other}' in metadata")),
    };
    let cfg = Mamba3Config {
        d_model: req(meta_map, "d_model")?,
        d_state: req(meta_map, "d_state")?,
        expand: req(meta_map, "expand")?,
        headdim: req(meta_map, "headdim")?,
        ngroups: req(meta_map, "ngroups")?,
        n_layers: req(meta_map, "n_layers")?,
        rope_fraction: req(meta_map, "rope_fraction")?,
        a_floor: req(meta_map, "a_floor")?,
        is_outproj_norm: match meta_map.get("is_outproj_norm") {
            Some(v) => v
                .parse()
                .map_err(|e| format!("parse is_outproj_norm: {e}"))?,
            None => false,
        },
        rms_norm_eps: match meta_map.get("rms_norm_eps") {
            Some(v) => v.parse().map_err(|e| format!("parse rms_norm_eps: {e}"))?,
            None => 1e-5,
        },
        // Files written before the mixer norms had their own epsilon used
        // the block norms' for them.
        ssm_norm_eps: match meta_map
            .get("ssm_norm_eps")
            .or(meta_map.get("rms_norm_eps"))
        {
            Some(v) => v.parse().map_err(|e| format!("parse ssm_norm_eps: {e}"))?,
            None => 1e-5,
        },
        scan_mode,
        block_norm: block_norm(meta_map)?,
        // Checkpoints from before the MLP carry no MLP.
        d_intermediate: match meta_map.get("d_intermediate") {
            Some(v) => v
                .parse()
                .map_err(|e| format!("parse d_intermediate: {e}"))?,
            None => 0,
        },
    };
    cfg.validate()?;

    let (w, input_dim) = weights_from_bytes(data, &cfg)?;
    Ok((w, cfg, input_dim))
}

#[deprecated(since = "0.8.0", note = "use `serialize::load`")]
pub fn load_mamba3_auto(path: &Path) -> Result<(Mamba3Weights, Mamba3Config, usize), String> {
    load(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_roundtrip() {
        let cfg = Mamba3Config::default();
        let w = Mamba3Weights::init(&cfg, 128, 42);
        let tmp = std::env::temp_dir().join("mamba3_test.safetensors");
        save(&tmp, &w, &cfg, 128).unwrap();
        let (w2, input_dim) = load_mamba3(&tmp, &cfg).unwrap();
        assert_eq!(input_dim, 128);
        assert_eq!(w.layers[0].dt_bias, w2.layers[0].dt_bias);
        assert_eq!(w.layers[0].in_proj_w, w2.layers[0].in_proj_w);
        assert_eq!(w.norm_f_weight, w2.norm_f_weight);
        std::fs::remove_file(tmp).ok();
    }

    /// A model without an input projection, as every HF checkpoint loads,
    /// saves and loads back as one: the empty tensors round-trip.
    #[test]
    fn identity_input_round_trips() {
        let cfg = Mamba3Config::default();
        let mut w = Mamba3Weights::init(&cfg, cfg.d_model, 42);
        w.input_proj_w.clear();
        w.input_proj_b.clear();
        let tmp = std::env::temp_dir().join("mamba3_identity_test.safetensors");
        save(&tmp, &w, &cfg, cfg.d_model).unwrap();
        let (w2, input_dim) = load_mamba3(&tmp, &cfg).unwrap();
        std::fs::remove_file(tmp).ok();
        assert_eq!(input_dim, cfg.d_model);
        assert!(w2.input_proj_w.is_empty() && w2.input_proj_b.is_empty());
        assert_eq!(w.layers[0].in_proj_w, w2.layers[0].in_proj_w);
    }

    /// A checkpoint fully describes its own model identity — the
    /// auto loader reconstructs the exact non-default config, and a caller
    /// supplying a CONTRADICTING config is rejected loudly instead of
    /// silently loading a different model.
    #[test]
    fn test_self_describing_checkpoint_roundtrip() {
        let cfg = Mamba3Config {
            is_outproj_norm: true,
            rms_norm_eps: 3e-6,
            ssm_norm_eps: 7e-6,
            scan_mode: crate::config::ScanMode::Sequential,
            ..Mamba3Config::default()
        };
        let w = Mamba3Weights::init(&cfg, 64, 7);
        let tmp = std::env::temp_dir().join("mamba3_test_g4.safetensors");
        save(&tmp, &w, &cfg, 64).unwrap();

        let (w2, cfg2, input_dim) = load(&tmp).unwrap();
        assert_eq!(cfg2, cfg, "metadata round-trip must restore the config");
        assert_eq!(input_dim, 64);
        assert_eq!(w.layers[0].dt_bias, w2.layers[0].dt_bias);

        let wrong_norm = Mamba3Config {
            is_outproj_norm: false,
            ..cfg
        };
        let err = load_mamba3(&tmp, &wrong_norm).err().unwrap();
        assert!(err.contains("is_outproj_norm"), "{err}");

        let wrong_eps = Mamba3Config {
            rms_norm_eps: 1e-5,
            ..cfg
        };
        let err = load_mamba3(&tmp, &wrong_eps).err().unwrap();
        assert!(err.contains("rms_norm_eps"), "{err}");

        let wrong_mixer_eps = Mamba3Config {
            ssm_norm_eps: 1e-5,
            ..cfg
        };
        let err = load_mamba3(&tmp, &wrong_mixer_eps).err().unwrap();
        assert!(err.contains("ssm_norm_eps"), "{err}");

        std::fs::remove_file(tmp).ok();
    }

    /// A file written before the mixer norms had their own epsilon ran
    /// them at the block norms': it loads with that epsilon for both.
    #[test]
    fn files_without_the_mixer_epsilon_load_with_the_block_one() {
        let cfg = Mamba3Config {
            rms_norm_eps: 3e-6,
            ssm_norm_eps: 3e-6,
            ..Mamba3Config::default()
        };
        let w = Mamba3Weights::init(&cfg, 64, 9);
        let (list, mut meta) = checkpoint_parts(&w, &cfg, 64).unwrap();
        meta.remove("ssm_norm_eps");
        let tmp = std::env::temp_dir().join(format!(
            "mamba3_test_old_eps_{}.safetensors",
            std::process::id()
        ));
        list.save(meta, &tmp).unwrap();
        let loaded = load(&tmp);
        std::fs::remove_file(&tmp).ok();
        let (_, cfg2, _) = loaded.unwrap();
        assert_eq!(cfg2, cfg);
    }

    /// A LayerNorm model saves its norms' biases and its kind, the auto
    /// loader rebuilds both, and a caller config of the other kind is
    /// refused.
    #[test]
    fn layer_norm_checkpoint_roundtrip() {
        let cfg = Mamba3Config {
            block_norm: crate::config::BlockNorm::Layer,
            d_intermediate: 200,
            ..Mamba3Config::default()
        };
        let mut w = Mamba3Weights::init(&cfg, 64, 13);
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
        let tmp = std::env::temp_dir().join(format!(
            "mamba3_test_layer_norm_{}.safetensors",
            std::process::id()
        ));
        save(&tmp, &w, &cfg, 64).unwrap();
        let (w2, cfg2, _) = load(&tmp).unwrap();
        assert_eq!(cfg2, cfg);
        assert_eq!(w2.norm_f_bias, w.norm_f_bias);
        for (a, b) in w.layers.iter().zip(&w2.layers) {
            assert_eq!(a.norm_bias, b.norm_bias);
        }
        for (a, b) in w.mlp.iter().zip(&w2.mlp) {
            assert_eq!(a.norm_bias, b.norm_bias);
        }
        let rms = Mamba3Config {
            block_norm: crate::config::BlockNorm::Rms,
            ..cfg
        };
        let err = load_mamba3(&tmp, &rms).err().unwrap();
        std::fs::remove_file(&tmp).ok();
        assert!(err.contains("block norms"), "{err}");
    }

    /// A model with a gated MLP saves its MLP tensors and its width, the
    /// auto loader rebuilds both, and a config without the MLP is refused
    /// rather than loading the mixers alone.
    #[test]
    fn test_mlp_checkpoint_roundtrip() {
        let cfg = Mamba3Config {
            d_intermediate: 200,
            ..Mamba3Config::default()
        };
        let w = Mamba3Weights::init(&cfg, 64, 11);
        assert_eq!(w.mlp.len(), cfg.n_layers);
        let tmp = std::env::temp_dir().join("mamba3_test_mlp.safetensors");
        save(&tmp, &w, &cfg, 64).unwrap();

        let (w2, cfg2, _) = load(&tmp).unwrap();
        assert_eq!(cfg2, cfg);
        for (a, b) in w.mlp.iter().zip(&w2.mlp) {
            assert_eq!(a.norm_weight, b.norm_weight);
            assert_eq!(a.fc1_w, b.fc1_w);
            assert_eq!(a.fc2_w, b.fc2_w);
        }

        let no_mlp = Mamba3Config {
            d_intermediate: 0,
            ..cfg
        };
        let err = load_mamba3(&tmp, &no_mlp).err().unwrap();
        assert!(err.contains("d_intermediate"), "{err}");

        std::fs::remove_file(tmp).ok();
    }
}
