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
//! `n_layers`, `input_dim`.

use crate::config::MambaConfig;
use crate::weights::{MambaLayerWeights, MambaWeights};
use safetensors::serialize_to_file;
use safetensors::tensor::{SafeTensors, TensorView};
use std::collections::HashMap;
use std::path::Path;

/// Save weights to a `.safetensors` file.
pub fn save(
    path: &Path,
    weights: &MambaWeights,
    cfg: &MambaConfig,
    input_dim: usize,
) -> Result<(), String> {
    let mut tensors: Vec<(&str, TensorView<'_>)> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    // Helper: create a 1D f32 tensor view from a slice
    let view = |data: &[f32], shape: &[usize]| -> TensorView<'_> {
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        TensorView::new(safetensors::Dtype::F32, shape.to_vec(), bytes).unwrap()
    };

    // input_proj
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let dc = cfg.d_conv;
    let dt_rank = cfg.dt_rank();
    let xdbl_dim = cfg.xdbl_dim();

    names.push("input_proj.weight".into());
    names.push("input_proj.bias".into());

    for i in 0..cfg.n_layers {
        names.push(format!("layers.{i}.norm.weight"));
        names.push(format!("layers.{i}.in_proj.weight"));
        names.push(format!("layers.{i}.conv1d.weight"));
        names.push(format!("layers.{i}.conv1d.bias"));
        names.push(format!("layers.{i}.x_proj.weight"));
        names.push(format!("layers.{i}.dt_proj.weight"));
        names.push(format!("layers.{i}.dt_proj.bias"));
        names.push(format!("layers.{i}.a_log"));
        names.push(format!("layers.{i}.D"));
        names.push(format!("layers.{i}.out_proj.weight"));
    }
    names.push("norm_f.weight".into());

    // Build tensor views (must live as long as tensors vec)
    let mut views: Vec<TensorView<'_>> = Vec::new();
    views.push(view(&weights.input_proj_w, &[input_dim, dm]));
    views.push(view(&weights.input_proj_b, &[dm]));

    for lw in &weights.layers {
        views.push(view(&lw.norm_weight, &[dm]));
        views.push(view(&lw.in_proj_w, &[dm, 2 * di]));
        views.push(view(&lw.conv1d_weight, &[di, dc]));
        views.push(view(&lw.conv1d_bias, &[di]));
        views.push(view(&lw.x_proj_w, &[di, xdbl_dim]));
        views.push(view(&lw.dt_proj_w, &[dt_rank, di]));
        views.push(view(&lw.dt_proj_b, &[di]));
        views.push(view(&lw.a_log, &[di, ds]));
        views.push(view(&lw.d_param, &[di]));
        views.push(view(&lw.out_proj_w, &[di, dm]));
    }
    views.push(view(&weights.norm_f_weight, &[dm]));

    // Build name-view pairs
    for (name, v) in names.iter().zip(views.into_iter()) {
        tensors.push((name.as_str(), v));
    }

    // Metadata: config as JSON
    let mut metadata = HashMap::new();
    metadata.insert(
        "config".to_string(),
        format!(
            r#"{{"d_model":{},"d_state":{},"d_conv":{},"expand":{},"n_layers":{},"input_dim":{}}}"#,
            dm, ds, dc, cfg.expand, cfg.n_layers, input_dim
        ),
    );

    serialize_to_file(tensors, Some(metadata), path).map_err(|e| format!("safetensors save: {e}"))
}

/// Load weights from a `.safetensors` file.
///
/// Returns `(weights, config, input_dim)`.
pub fn load(path: &Path) -> Result<(MambaWeights, MambaConfig, usize), String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    // Read metadata first to get config
    let (_, meta) =
        SafeTensors::read_metadata(&data).map_err(|e| format!("safetensors metadata: {e}"))?;
    let config_json = meta
        .metadata()
        .as_ref()
        .and_then(|m| m.get("config"))
        .ok_or("missing 'config' metadata")?;

    let st = SafeTensors::deserialize(&data).map_err(|e| format!("safetensors parse: {e}"))?;

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

    let cfg = MambaConfig {
        d_model,
        d_state,
        d_conv,
        expand,
        n_layers,
        ..Default::default()
    };
    cfg.validate()?;

    let _di = cfg.d_inner();
    let _dt_rank = cfg.dt_rank();
    let _xdbl_dim = cfg.xdbl_dim();

    // Helper: read tensor as Vec<f32>
    let read = |name: &str| -> Result<Vec<f32>, String> {
        let tv = st
            .tensor(name)
            .map_err(|e| format!("tensor '{name}': {e}"))?;
        let bytes = tv.data();
        if bytes.len() % 4 != 0 {
            return Err(format!("tensor '{name}': not aligned to f32"));
        }
        let floats =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
        Ok(floats.to_vec())
    };

    let input_proj_w = read("input_proj.weight")?;
    let input_proj_b = read("input_proj.bias")?;

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let a_log = read(&format!("layers.{i}.a_log"))?;
        let a_neg = a_log.iter().map(|v| -v.exp()).collect();
        layers.push(MambaLayerWeights {
            norm_weight: read(&format!("layers.{i}.norm.weight"))?,
            in_proj_w: read(&format!("layers.{i}.in_proj.weight"))?,
            conv1d_weight: read(&format!("layers.{i}.conv1d.weight"))?,
            conv1d_bias: read(&format!("layers.{i}.conv1d.bias"))?,
            x_proj_w: read(&format!("layers.{i}.x_proj.weight"))?,
            dt_proj_w: read(&format!("layers.{i}.dt_proj.weight"))?,
            dt_proj_b: read(&format!("layers.{i}.dt_proj.bias"))?,
            a_log,
            a_neg,
            d_param: read(&format!("layers.{i}.D"))?,
            out_proj_w: read(&format!("layers.{i}.out_proj.weight"))?,
        });
    }

    let norm_f_weight = read("norm_f.weight")?;

    let weights = MambaWeights {
        input_proj_w,
        input_proj_b,
        layers,
        norm_f_weight,
    };

    Ok((weights, cfg, input_dim))
}
