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
//! - `layers.{i}.norm2.weight`, `layers.{i}.mlp.fc1.weight`,
//!   `layers.{i}.mlp.fc2.weight` (models with a gated MLP)
//! - `norm_f.weight`

use super::config::Mamba3Config;
use super::weights::{Mamba3LayerWeights, Mamba3Weights};
use safetensors::serialize_to_file;
use safetensors::tensor::{SafeTensors, TensorView};
use std::collections::HashMap;
use std::path::Path;

fn view<'a>(data: &'a [f32], shape: &[usize]) -> TensorView<'a> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    TensorView::new(safetensors::Dtype::F32, shape.to_vec(), bytes).expect(
        "tensor view: bytes are sized from data.len() and every call site \
             passes a shape whose product equals data.len()",
    )
}

/// Save the weights, the configuration and the input width to a
/// safetensors file.
pub fn save(
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

    let hidden = cfg.mlp_hidden();
    for (i, m) in weights.mlp.iter().enumerate() {
        let p = |n: &str| format!("layers.{i}.{n}");
        tensors.extend([
            t!(p("norm2.weight"), &m.norm_weight, &[dm]),
            t!(p("mlp.fc1.weight"), &m.fc1_w, &[dm, 2 * hidden]),
            t!(p("mlp.fc2.weight"), &m.fc2_w, &[hidden, dm]),
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
        // Identity-critical: omitting is_outproj_norm loads a DIFFERENT
        // model (norm_gate silently on/off); eps re-defines every norm.
        (
            "is_outproj_norm".to_string(),
            cfg.is_outproj_norm.to_string(),
        ),
        ("rms_norm_eps".to_string(), cfg.rms_norm_eps.to_string()),
        // Identity-critical too: without it a model with an MLP would load
        // as one without.
        ("d_intermediate".to_string(), cfg.d_intermediate.to_string()),
        ("scan_mode".to_string(), format!("{:?}", cfg.scan_mode)),
        ("mamba_version".to_string(), "3".to_string()),
    ]);

    let refs: Vec<_> = tensors
        .iter()
        .map(|(n, t)| (n.as_str(), t.clone()))
        .collect();
    serialize_to_file(refs, Some(meta), path).map_err(|e| format!("safetensors save failed: {e:?}"))
}

/// Load Mamba-3 SISO weights from safetensors with a caller-supplied config.
///
/// Identity guard: if the file carries `is_outproj_norm` / `rms_norm_eps`
/// metadata (checkpoints written since 0.6) and it CONTRADICTS `cfg`, this
/// fails loudly — silently loading with a mismatched norm topology or eps
/// produces a different model with plausible-looking outputs.
pub fn load_mamba3(path: &Path, cfg: &Mamba3Config) -> Result<(Mamba3Weights, usize), String> {
    let data = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    let (_, header) = SafeTensors::read_metadata(&data).map_err(|e| format!("metadata: {e}"))?;
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
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

    let mlp = if cfg.mlp_hidden() == 0 {
        Vec::new()
    } else {
        (0..cfg.n_layers)
            .map(|i| {
                let p = |n: &str| format!("layers.{i}.{n}");
                Ok(crate::mlp::GatedMlpLayerWeights {
                    norm_weight: get(&p("norm2.weight"))?,
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
    let (_, header) = SafeTensors::read_metadata(&data).map_err(|e| format!("metadata: {e}"))?;
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
        scan_mode,
        // Checkpoints from before the MLP carry no MLP.
        d_intermediate: match meta_map.get("d_intermediate") {
            Some(v) => v
                .parse()
                .map_err(|e| format!("parse d_intermediate: {e}"))?,
            None => 0,
        },
    };
    cfg.validate()?;

    let (w, input_dim) = load_mamba3(path, &cfg)?;
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

    /// A checkpoint fully describes its own model identity — the
    /// auto loader reconstructs the exact non-default config, and a caller
    /// supplying a CONTRADICTING config is rejected loudly instead of
    /// silently loading a different model.
    #[test]
    fn test_self_describing_checkpoint_roundtrip() {
        let cfg = Mamba3Config {
            is_outproj_norm: true,
            rms_norm_eps: 3e-6,
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

        std::fs::remove_file(tmp).ok();
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
