//! Weight serialization via safetensors format (HuggingFace standard).
//!
//! Uses the `safetensors` crate for safe, zero-copy, framework-agnostic
//! weight storage. Compatible with Python `safetensors` library.
//!
//! Tensor naming convention:
//! - `input_proj.weight`, `input_proj.bias`
//! - `layers.{i}.norm.weight`
//! - `layers.{i}.in_proj.weight`
//! - `layers.{i}.conv1d.weight`, `layers.{i}.conv1d.bias`
//! - `layers.{i}.x_proj.weight`
//! - `layers.{i}.dt_proj.weight`, `layers.{i}.dt_proj.bias`
//! - `layers.{i}.a_log`
//! - `layers.{i}.D`
//! - `layers.{i}.out_proj.weight`
//! - `norm_f.weight`
//!
//! Metadata stores config as JSON: `d_model`, `d_state`, `d_conv`, `expand`,
//! `n_layers`, `input_dim`, and the family marker `mamba_version = "1"`.

use crate::config::ModelFamily;
use crate::mamba1::config::MambaConfig;
use crate::mamba1::weights::{MambaLayerWeights, MambaWeights};
use crate::ops::st_io::{TensorList, checkpoint_family};
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// The family marker in the metadata.
const VERSION: &str = "1";

/// Save weights to a `.safetensors` file.
pub fn save(
    path: &Path,
    weights: &MambaWeights,
    cfg: &MambaConfig,
    input_dim: usize,
) -> Result<(), String> {
    let (list, meta) = checkpoint_parts(weights, cfg, input_dim)?;
    list.save(meta, path)
}

/// The tensors and the metadata [`save`] writes; the language-model
/// checkpoint adds its embedding and head to them.
pub(crate) fn checkpoint_parts<'a>(
    weights: &'a MambaWeights,
    cfg: &MambaConfig,
    input_dim: usize,
) -> Result<(TensorList<'a>, HashMap<String, String>), String> {
    weights.validate(cfg, input_dim)?;
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let dc = cfg.d_conv;
    let dt_rank = cfg.dt_rank();
    let xdbl_dim = cfg.xdbl_dim();

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
        list.push(p("in_proj.weight"), &lw.in_proj_w, &[dm, 2 * di])?;
        if cfg.proj_bias {
            list.push(p("in_proj.bias"), &lw.in_proj_b, &[2 * di])?;
        }
        list.push(p("conv1d.weight"), &lw.conv1d_weight, &[di, dc])?;
        if cfg.conv_bias {
            list.push(p("conv1d.bias"), &lw.conv1d_bias, &[di])?;
        }
        list.push(p("x_proj.weight"), &lw.x_proj_w, &[di, xdbl_dim])?;
        list.push(p("dt_proj.weight"), &lw.dt_proj_w, &[dt_rank, di])?;
        list.push(p("dt_proj.bias"), &lw.dt_proj_b, &[di])?;
        list.push(p("a_log"), &lw.a_log, &[di, ds])?;
        list.push(p("D"), &lw.d_param, &[di])?;
        list.push(p("out_proj.weight"), &lw.out_proj_w, &[di, dm])?;
        if cfg.proj_bias {
            list.push(p("out_proj.bias"), &lw.out_proj_b, &[dm])?;
        }
    }
    list.push("norm_f.weight".into(), &weights.norm_f_weight, &[dm])?;

    // The configuration as JSON. scan_mode and the norm epsilons are part
    // of the checkpoint's numeric route: without them a model of another
    // eps would load with the default one, and a FalconMamba model as
    // Mamba-1.
    // The fields a model leaves at their defaults are not written, so a
    // file of such a model reads as it always did.
    let mut extra = cfg
        .bcdt_rms_eps
        .map(|eps| format!(r#","bcdt_rms_eps":{eps}"#))
        .unwrap_or_default();
    if let Some(rank) = cfg.dt_rank {
        extra.push_str(&format!(r#","dt_rank":{rank}"#));
    }
    if cfg.proj_bias {
        extra.push_str(r#","proj_bias":true"#);
    }
    if !cfg.conv_bias {
        extra.push_str(r#","conv_bias":false"#);
    }
    let meta = HashMap::from([
        ("mamba_version".to_string(), VERSION.to_string()),
        (
            "config".to_string(),
            format!(
                r#"{{"d_model":{},"d_state":{},"d_conv":{},"expand":{},"n_layers":{},"input_dim":{},"scan_mode":"{:?}","rms_norm_eps":{}{extra}}}"#,
                dm, ds, dc, cfg.expand, cfg.n_layers, input_dim, cfg.scan_mode, cfg.rms_norm_eps
            ),
        ),
    ]);
    Ok((list, meta))
}

/// Load weights from a `.safetensors` file.
///
/// Returns `(weights, config, input_dim)`.
pub fn load(path: &Path) -> Result<(MambaWeights, MambaConfig, usize), String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    load_from_bytes(&data)
}

/// Load weights from already-read `.safetensors` bytes — for consumers
/// that hash/verify the file first and should not read it twice.
pub fn load_from_bytes(data: &[u8]) -> Result<(MambaWeights, MambaConfig, usize), String> {
    // Read metadata first to get config
    let (_, meta) =
        SafeTensors::read_metadata(data).map_err(|e| format!("safetensors metadata: {e}"))?;
    let meta = meta
        .metadata()
        .as_ref()
        .ok_or("missing 'config' metadata")?;
    // Files written before the family tag carry none.
    if checkpoint_family(meta)? != ModelFamily::Mamba1 {
        return Err(format!(
            "not a Mamba-1 checkpoint (mamba_version = {:?})",
            meta.get("mamba_version")
        ));
    }
    let config_json = meta.get("config").ok_or("missing 'config' metadata")?;

    let st = SafeTensors::deserialize(data).map_err(|e| format!("safetensors parse: {e}"))?;

    // Minimal JSON parsing (no serde_json dependency)
    let parse_field = |json: &str, key: &str| -> Result<usize, String> {
        let pattern = format!("\"{}\":", key);
        let start = json
            .find(&pattern)
            .ok_or(format!("missing '{key}' in config metadata"))?
            + pattern.len();
        let end = json[start..]
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(json.len() - start);
        json[start..start + end]
            .parse::<usize>()
            .map_err(|e| format!("parse '{key}': {e}"))
    };

    let d_model = parse_field(config_json, "d_model")?;
    let d_state = parse_field(config_json, "d_state")?;
    let d_conv = parse_field(config_json, "d_conv")?;
    let expand = parse_field(config_json, "expand")?;
    let n_layers = parse_field(config_json, "n_layers")?;
    let input_dim = parse_field(config_json, "input_dim")?;

    // Route fields are OPTIONAL for backward compatibility with pre-0.5.2
    // checkpoints (which carry neither): absent -> library defaults, same
    // behavior as before; present -> restored exactly (M-B).
    let scan_mode = if config_json.contains(r#""scan_mode":"Sequential""#) {
        crate::config::ScanMode::Sequential
    } else if config_json.contains(r#""scan_mode":"Parallel""#) {
        crate::config::ScanMode::Parallel
    } else {
        crate::config::ScanMode::default()
    };
    // A float field of the config, `None` when the file has none.
    let float = |key: &str| -> Result<Option<f32>, String> {
        let tag = format!(r#""{key}":"#);
        config_json
            .find(&tag)
            .map(|p| {
                let tail = &config_json[p + tag.len()..];
                let end = tail
                    .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
                    .unwrap_or(tail.len());
                tail[..end]
                    .parse::<f32>()
                    .map_err(|e| format!("parse '{key}': {e}"))
            })
            .transpose()
    };
    let rms_norm_eps =
        float("rms_norm_eps")?.unwrap_or_else(|| MambaConfig::default().rms_norm_eps);

    let cfg = MambaConfig {
        d_model,
        d_state,
        d_conv,
        expand,
        n_layers,
        scan_mode,
        rms_norm_eps,
        bcdt_rms_eps: float("bcdt_rms_eps")?,
        dt_rank: config_json
            .contains(r#""dt_rank":"#)
            .then(|| parse_field(config_json, "dt_rank"))
            .transpose()?,
        proj_bias: config_json.contains(r#""proj_bias":true"#),
        conv_bias: !config_json.contains(r#""conv_bias":false"#),
    };
    cfg.validate()?;

    // Helper: read tensor as Vec<f32>. Decodes byte-wise: the safetensors
    // spec does not guarantee 4-byte alignment of the data section, so
    // casting the raw buffer to `*const f32` would be UB on a spec-valid
    // file written with an unpadded header.
    let read = |name: &str| -> Result<Vec<f32>, String> {
        let tv = st
            .tensor(name)
            .map_err(|e| format!("tensor '{name}': {e}"))?;
        if tv.dtype() != safetensors::Dtype::F32 {
            return Err(format!(
                "tensor '{name}': dtype {:?}, expected F32",
                tv.dtype()
            ));
        }
        let bytes = tv.data();
        if bytes.len() % 4 != 0 {
            return Err(format!(
                "tensor '{name}': byte length {} is not a multiple of 4",
                bytes.len()
            ));
        }
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };

    let input_proj_w = read("input_proj.weight")?;
    let input_proj_b = read("input_proj.bias")?;

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let a_log = read(&format!("layers.{i}.a_log"))?;
        let a_neg = a_log.iter().map(|v| -v.exp()).collect();
        // A bias the config lacks is not in the file.
        let optional = |on: bool, name: &str| -> Result<Vec<f32>, String> {
            if on {
                read(&format!("layers.{i}.{name}"))
            } else {
                Ok(Vec::new())
            }
        };
        layers.push(MambaLayerWeights {
            norm_weight: read(&format!("layers.{i}.norm.weight"))?,
            in_proj_w: read(&format!("layers.{i}.in_proj.weight"))?,
            in_proj_b: optional(cfg.proj_bias, "in_proj.bias")?,
            conv1d_weight: read(&format!("layers.{i}.conv1d.weight"))?,
            conv1d_bias: optional(cfg.conv_bias, "conv1d.bias")?,
            x_proj_w: read(&format!("layers.{i}.x_proj.weight"))?,
            dt_proj_w: read(&format!("layers.{i}.dt_proj.weight"))?,
            dt_proj_b: read(&format!("layers.{i}.dt_proj.bias"))?,
            a_log,
            a_neg,
            d_param: read(&format!("layers.{i}.D"))?,
            out_proj_w: read(&format!("layers.{i}.out_proj.weight"))?,
            out_proj_b: optional(cfg.proj_bias, "out_proj.bias")?,
        });
    }

    let norm_f_weight = read("norm_f.weight")?;

    let weights = MambaWeights {
        input_proj_w,
        input_proj_b,
        layers,
        norm_f_weight,
    };

    // Reject undersized/oversized tensors before they can reach the BLAS
    // FFI paths (which trust config-derived dims for their pointer math).
    weights.validate(&cfg, input_dim)?;

    Ok((weights, cfg, input_dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkpoint written before the family marker carries none and
    /// still loads; a file marked as another family does not.
    #[test]
    fn untagged_files_load_and_other_families_are_refused() {
        let cfg = MambaConfig::default();
        let w = MambaWeights::init(&cfg, 16, 7);
        let path = std::env::temp_dir().join(format!(
            "mamba1_untagged_{}.safetensors",
            std::process::id()
        ));
        let write = |tag: Option<&str>| {
            let (list, mut meta) = checkpoint_parts(&w, &cfg, 16).unwrap();
            match tag {
                Some(t) => meta.insert("mamba_version".into(), t.into()),
                None => meta.remove("mamba_version"),
            };
            list.save(meta, &path).unwrap();
            load(&path)
        };
        let untagged = write(None);
        let mamba2 = write(Some("2"));
        std::fs::remove_file(&path).ok();
        let (w2, cfg2, input_dim) = untagged.unwrap();
        assert_eq!((cfg2, input_dim), (cfg, 16));
        assert_eq!(w.layers[0].in_proj_w, w2.layers[0].in_proj_w);
        let err = mamba2.err().unwrap();
        assert!(err.contains("not a Mamba-1 checkpoint"), "{err}");
    }

    /// A model without an input projection, as every HF checkpoint loads,
    /// saves and loads back as one instead of failing on its empty tensors.
    #[test]
    fn identity_input_round_trips() {
        let cfg = MambaConfig::default();
        let mut w = MambaWeights::init(&cfg, cfg.d_model, 42);
        w.input_proj_w.clear();
        w.input_proj_b.clear();
        let path = std::env::temp_dir().join("mamba1_identity_test.safetensors");
        save(&path, &w, &cfg, cfg.d_model).unwrap();
        let loaded = load(&path);
        std::fs::remove_file(&path).ok();
        let (w2, cfg2, input_dim) = loaded.unwrap();
        assert_eq!((cfg2, input_dim), (cfg, cfg.d_model));
        assert!(w2.input_proj_w.is_empty() && w2.input_proj_b.is_empty());
        assert_eq!(w.layers[0].in_proj_w, w2.layers[0].in_proj_w);
    }

    /// A FalconMamba model keeps its mixer norm epsilon through a save and
    /// a load; a Mamba-1 file (none written) loads without it.
    #[test]
    fn falcon_mixer_norms_round_trip() {
        for bcdt_rms_eps in [Some(1e-6), Some(3.5e-5), None] {
            let cfg = MambaConfig {
                bcdt_rms_eps,
                ..MambaConfig::default()
            };
            let w = MambaWeights::init(&cfg, 16, 7);
            let path = std::env::temp_dir().join(format!(
                "mamba1_falcon_{}_{:?}.safetensors",
                std::process::id(),
                bcdt_rms_eps
            ));
            save(&path, &w, &cfg, 16).unwrap();
            let loaded = load(&path);
            std::fs::remove_file(&path).ok();
            let (_, cfg2, _) = loaded.unwrap();
            assert_eq!(cfg2, cfg);
        }
    }

    /// The projection biases, a missing conv bias and an explicit dt rank
    /// survive a save and a load, the tensors bit for bit; a file of a
    /// model without them stays as it was written before they existed.
    #[test]
    fn bias_and_rank_options_round_trip() {
        let variants = [
            (Some(5), true, false),
            (None, true, true),
            (Some(3), false, false),
            (None, false, true),
        ];
        for (dt_rank, proj_bias, conv_bias) in variants {
            let cfg = MambaConfig {
                dt_rank,
                proj_bias,
                conv_bias,
                ..MambaConfig::default()
            };
            let mut w = MambaWeights::init(&cfg, 16, 9);
            for lw in &mut w.layers {
                for (i, b) in lw
                    .in_proj_b
                    .iter_mut()
                    .chain(&mut lw.out_proj_b)
                    .enumerate()
                {
                    *b = i as f32 * 0.01 - 0.3;
                }
            }
            let (list, meta) = checkpoint_parts(&w, &cfg, 16).unwrap();
            let config_json = &meta["config"];
            if (dt_rank, proj_bias, conv_bias) == (None, false, true) {
                for key in ["dt_rank", "proj_bias", "conv_bias"] {
                    assert!(!config_json.contains(key), "{key} in {config_json}");
                }
            }
            let path = std::env::temp_dir().join(format!(
                "mamba1_options_{}_{dt_rank:?}_{proj_bias}_{conv_bias}.safetensors",
                std::process::id()
            ));
            list.save(meta, &path).unwrap();
            let loaded = load(&path);
            std::fs::remove_file(&path).ok();
            let (w2, cfg2, _) = loaded.unwrap();
            assert_eq!(cfg2, cfg);
            for (a, b) in w.layers.iter().zip(&w2.layers) {
                assert_eq!(a.in_proj_b, b.in_proj_b);
                assert_eq!(a.out_proj_b, b.out_proj_b);
                assert_eq!(a.conv1d_bias, b.conv1d_bias);
                assert_eq!(a.dt_proj_w, b.dt_proj_w);
            }
        }
    }
}
