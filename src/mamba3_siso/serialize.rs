//! Safetensors serialization for Mamba-3 SISO weights.
//!
//! Tensor naming convention:
//! - `input_proj.weight`, `input_proj.bias`
//! - `layers.{i}.norm.weight`
//! - `layers.{i}.in_proj.weight`
//! - `layers.{i}.dt_bias`
//! - `layers.{i}.b_norm.weight`, `layers.{i}.c_norm.weight`
//! - `layers.{i}.b_bias`, `layers.{i}.c_bias`
//! - `layers.{i}.D`
//! - `layers.{i}.norm_gate.weight`
//! - `layers.{i}.out_proj.weight`
//! - `norm_f.weight`

use super::config::Mamba3Config;
use super::weights::{Mamba3LayerWeights, Mamba3Weights};
use safetensors::serialize_to_file;
use safetensors::tensor::{SafeTensors, TensorView};
use std::collections::HashMap;
use std::path::Path;

fn view<'a>(data: &'a [f32], shape: &[usize]) -> TensorView<'a> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    TensorView::new(safetensors::Dtype::F32, shape.to_vec(), bytes).unwrap()
}

/// Save Mamba-3 SISO weights to safetensors.
pub fn save_mamba3(
    path: &Path,
    weights: &Mamba3Weights,
    cfg: &Mamba3Config,
    input_dim: usize,
) -> Result<(), String> {
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nh = cfg.nheads();
    let ip = cfg.in_proj_out_dim();

    macro_rules! t {
        ($name:expr, $data:expr, $shape:expr) => {
            ($name.to_string(), view($data, $shape))
        };
    }

    let mut tensors: Vec<(String, TensorView<'_>)> = vec![
        t!("input_proj.weight", &weights.input_proj_w, &[input_dim, dm]),
        t!("input_proj.bias", &weights.input_proj_b, &[dm]),
    ];

    for (i, lw) in weights.layers.iter().enumerate() {
        let p = |n: &str| format!("layers.{i}.{n}");
        tensors.extend([
            t!(p("norm.weight"), &lw.norm_weight, &[dm]),
            t!(p("in_proj.weight"), &lw.in_proj_w, &[dm, ip]),
            t!(p("dt_bias"), &lw.dt_bias, &[nh]),
            t!(p("b_norm.weight"), &lw.b_norm_weight, &[ds]),
            t!(p("c_norm.weight"), &lw.c_norm_weight, &[ds]),
            t!(p("b_bias"), &lw.b_bias, &[nh, ds]),
            t!(p("c_bias"), &lw.c_bias, &[nh, ds]),
            t!(p("D"), &lw.d_param, &[nh]),
            t!(p("norm_gate.weight"), &lw.norm_gate_weight, &[di]),
            t!(p("out_proj.weight"), &lw.out_proj_w, &[di, dm]),
        ]);
    }

    tensors.push(t!("norm_f.weight", &weights.norm_f_weight, &[dm]));

    let meta = HashMap::from([
        ("d_model".to_string(), dm.to_string()),
        ("d_state".to_string(), ds.to_string()),
        ("expand".to_string(), cfg.expand.to_string()),
        ("headdim".to_string(), cfg.headdim.to_string()),
        ("ngroups".to_string(), cfg.ngroups.to_string()),
        ("n_layers".to_string(), cfg.n_layers.to_string()),
        ("input_dim".to_string(), input_dim.to_string()),
        ("rope_fraction".to_string(), cfg.rope_fraction.to_string()),
        ("a_floor".to_string(), cfg.a_floor.to_string()),
        ("mamba_version".to_string(), "3".to_string()),
    ]);

    let refs: Vec<_> = tensors
        .iter()
        .map(|(n, t)| (n.as_str(), t.clone()))
        .collect();
    serialize_to_file(refs, Some(meta), path).map_err(|e| format!("safetensors save failed: {e:?}"))
}

/// Load Mamba-3 SISO weights from safetensors.
pub fn load_mamba3(path: &Path, cfg: &Mamba3Config) -> Result<(Mamba3Weights, usize), String> {
    let data = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    let (_, header) = SafeTensors::read_metadata(&data).map_err(|e| format!("metadata: {e}"))?;
    let meta_map = header.metadata().as_ref().ok_or("no metadata")?;
    let input_dim: usize = meta_map
        .get("input_dim")
        .ok_or("missing input_dim")?
        .parse()
        .map_err(|e| format!("parse input_dim: {e}"))?;

    let st = SafeTensors::deserialize(&data).map_err(|e| format!("deserialize: {e:?}"))?;

    // Byte-wise decode: the safetensors spec does not guarantee 4-byte
    // alignment of the data section, so a `*const f32` cast on the raw
    // buffer would be UB for a spec-valid file with an unpadded header.
    let get = |name: &str| -> Result<Vec<f32>, String> {
        let t = st
            .tensor(name)
            .map_err(|e| format!("tensor '{name}': {e:?}"))?;
        if t.dtype() != safetensors::Dtype::F32 {
            return Err(format!(
                "tensor '{name}': dtype {:?}, expected F32",
                t.dtype()
            ));
        }
        let bytes = t.data();
        if bytes.len() % 4 != 0 {
            return Err(format!(
                "tensor '{name}': byte length {} is not a multiple of 4",
                bytes.len()
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };

    let mut layers = Vec::new();
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("layers.{i}.{n}");
        layers.push(Mamba3LayerWeights {
            norm_weight: get(&p("norm.weight"))?,
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

    let w = Mamba3Weights {
        input_proj_w: get("input_proj.weight")?,
        input_proj_b: get("input_proj.bias")?,
        layers,
        norm_f_weight: get("norm_f.weight")?,
    };

    w.validate(cfg, input_dim)?;
    Ok((w, input_dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_roundtrip() {
        let cfg = Mamba3Config::default();
        let w = Mamba3Weights::init(&cfg, 128, 42);
        let tmp = std::env::temp_dir().join("mamba3_test.safetensors");
        save_mamba3(&tmp, &w, &cfg, 128).unwrap();
        let (w2, input_dim) = load_mamba3(&tmp, &cfg).unwrap();
        assert_eq!(input_dim, 128);
        assert_eq!(w.layers[0].dt_bias, w2.layers[0].dt_bias);
        assert_eq!(w.layers[0].in_proj_w, w2.layers[0].in_proj_w);
        assert_eq!(w.norm_f_weight, w2.norm_f_weight);
        std::fs::remove_file(tmp).ok();
    }
}
