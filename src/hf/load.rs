//! Top-level HuggingFace model loading.

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;

use crate::config::MambaConfig;
use crate::module::MambaBackbone;
use crate::weights::{MambaLayerWeights, MambaWeights};

use super::bf16::bf16_bytes_to_f32;
use super::config_json::{ModelFamily, parse_config_json};
use super::keys::{RemappedKey, remap_m1_hf_key};

/// Loaded HF Mamba model components.
pub struct HfModel {
    pub backbone: MambaBackbone,
    pub embed: Vec<f32>,
    pub lm_head: Option<Vec<f32>>,
    pub vocab_size: usize,
    pub vocab_size_padded: usize,
    pub d_model: usize,
}

/// Load a Mamba model from a HuggingFace checkpoint directory.
///
/// The directory must contain `config.json` and safetensors weight file(s).
pub fn load_hf(dir: &Path) -> Result<HfModel, String> {
    let config_path = dir.join("config.json");
    let config_bytes =
        std::fs::read(&config_path).map_err(|e| format!("cannot read config.json: {e}"))?;
    let hf_cfg = parse_config_json(&config_bytes)?;

    if hf_cfg.family != ModelFamily::Mamba1 {
        return Err(format!(
            "only Mamba SSM supported by this loader, got {:?}",
            hf_cfg.family
        ));
    }

    let shard_paths = discover_shards(dir)?;

    // config.json is untrusted input: bound the dims before sizing
    // allocations from their products (otherwise a hostile file can demand
    // a near-arbitrary allocation or wrap the product in release builds).
    if hf_cfg.vocab_size == 0 || hf_cfg.vocab_size > (1 << 24) {
        return Err(format!(
            "vocab_size {} out of supported range (1..=2^24)",
            hf_cfg.vocab_size
        ));
    }
    if hf_cfg.d_model == 0 || hf_cfg.d_model > (1 << 20) {
        return Err(format!(
            "d_model {} out of supported range (1..=2^20)",
            hf_cfg.d_model
        ));
    }
    let vocab_size_padded = (hf_cfg.vocab_size + 63) & !63;
    let embed_len = vocab_size_padded
        .checked_mul(hf_cfg.d_model)
        .ok_or("vocab_size_padded * d_model overflows usize")?;
    let mut embed = vec![0.0f32; embed_len];
    let mut lm_head: Option<Vec<f32>> = None;
    let mut norm_f_weight = vec![0.0f32; hf_cfg.d_model];
    let mut layer_weights: Vec<Option<LayerAccum>> = (0..hf_cfg.n_layers).map(|_| None).collect();

    for shard_path in &shard_paths {
        let file_bytes = std::fs::read(shard_path)
            .map_err(|e| format!("cannot read {}: {e}", shard_path.display()))?;
        let st = SafeTensors::deserialize(&file_bytes)
            .map_err(|e| format!("safetensors parse error in {}: {e}", shard_path.display()))?;

        for name in st.names() {
            let tensor = st
                .tensor(name)
                .map_err(|e| format!("cannot read tensor {name}: {e}"))?;

            match remap_m1_hf_key(name) {
                RemappedKey::Embedding => {
                    let data = tensor_to_f32(tensor.data(), tensor.dtype())?;
                    let expected = hf_cfg.vocab_size * hf_cfg.d_model;
                    if data.len() != expected {
                        return Err(format!(
                            "embedding shape: expected {expected} elements, got {}",
                            data.len()
                        ));
                    }
                    embed[..expected].copy_from_slice(&data);
                }
                RemappedKey::LmHead => {
                    let data = tensor_to_f32(tensor.data(), tensor.dtype())?;
                    lm_head = Some(data);
                }
                RemappedKey::NormF => {
                    let data = tensor_to_f32(tensor.data(), tensor.dtype())?;
                    if data.len() != hf_cfg.d_model {
                        return Err(format!(
                            "norm_f_weight: expected {}, got {}",
                            hf_cfg.d_model,
                            data.len()
                        ));
                    }
                    norm_f_weight.copy_from_slice(&data);
                }
                RemappedKey::Layer(idx, field) => {
                    if idx >= hf_cfg.n_layers {
                        return Err(format!("layer index {idx} >= n_layers {}", hf_cfg.n_layers));
                    }
                    let data = tensor_to_f32(tensor.data(), tensor.dtype())?;
                    let accum = layer_weights[idx].get_or_insert_with(LayerAccum::default);
                    accum.set(field, data)?;
                }
                RemappedKey::Unknown => {}
            }
        }
    }

    if !hf_cfg.tie_word_embeddings && lm_head.is_none() {
        return Err("tie_word_embeddings=false but lm_head.weight not found".into());
    }
    if hf_cfg.tie_word_embeddings {
        lm_head = None;
    }

    // Transpose untied lm_head from PyTorch [vocab_size, d_model] to [d_model, vocab_size].
    // Our matvec_forward expects W as [n_in, n_out] = [d_model, vocab_size].
    if let Some(ref mut lm) = lm_head {
        let expected = hf_cfg.vocab_size * hf_cfg.d_model;
        if lm.len() != expected {
            return Err(format!(
                "lm_head.weight: expected {expected} elements, got {}",
                lm.len()
            ));
        }
        *lm = transpose_2d(std::mem::take(lm), hf_cfg.vocab_size, hf_cfg.d_model);
    }

    let mamba_cfg = MambaConfig {
        d_model: hf_cfg.d_model,
        d_state: hf_cfg.d_state,
        d_conv: hf_cfg.d_conv,
        expand: hf_cfg.expand,
        n_layers: hf_cfg.n_layers,
        // Checkpoint-specified norm epsilon (FalconMamba: 1e-6) — plumbed
        // through every CPU/GPU inference norm via MambaConfig/MambaDims.
        rms_norm_eps: hf_cfg.rms_norm_eps,
        ..MambaConfig::default()
    };

    let layers: Vec<MambaLayerWeights> = layer_weights
        .into_iter()
        .enumerate()
        .map(|(i, opt)| {
            opt.ok_or(format!("layer {i}: no weights found"))
                .and_then(|a| a.into_layer_weights(&mamba_cfg, &hf_cfg))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let weights = MambaWeights {
        input_proj_w: vec![],
        input_proj_b: vec![],
        layers,
        norm_f_weight,
    };

    let backbone = MambaBackbone::from_weights_no_proj(mamba_cfg, weights)?;

    Ok(HfModel {
        backbone,
        embed,
        lm_head,
        vocab_size: hf_cfg.vocab_size,
        vocab_size_padded,
        d_model: hf_cfg.d_model,
    })
}

fn discover_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_bytes =
            std::fs::read(&index_path).map_err(|e| format!("cannot read index.json: {e}"))?;
        let index: serde_json::Value =
            serde_json::from_slice(&index_bytes).map_err(|e| format!("index.json parse: {e}"))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or("index.json missing weight_map")?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();

        Ok(shard_files.into_iter().map(|f| dir.join(f)).collect())
    } else {
        let single = dir.join("model.safetensors");
        if single.exists() {
            Ok(vec![single])
        } else {
            Err(format!(
                "no model.safetensors or index.json in {}",
                dir.display()
            ))
        }
    }
}

fn tensor_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f32>, String> {
    match dtype {
        safetensors::Dtype::F32 => {
            let floats: &[f32] = bytemuck::cast_slice(data);
            Ok(floats.to_vec())
        }
        safetensors::Dtype::BF16 => bf16_bytes_to_f32(data),
        safetensors::Dtype::F16 => {
            use half::f16;
            use half::slice::HalfFloatSliceExt;
            let f16_slice: &[f16] = bytemuck::cast_slice(data);
            let mut out = vec![0.0f32; f16_slice.len()];
            f16_slice.convert_to_f32_slice(&mut out);
            Ok(out)
        }
        other => Err(format!("unsupported dtype: {other:?}")),
    }
}

#[derive(Default)]
struct LayerAccum {
    fields: HashMap<&'static str, Vec<f32>>,
}

impl LayerAccum {
    fn set(&mut self, field: &'static str, data: Vec<f32>) -> Result<(), String> {
        if self.fields.contains_key(field) {
            return Err(format!("duplicate field: {field}"));
        }
        self.fields.insert(field, data);
        Ok(())
    }

    fn take(&mut self, field: &str) -> Result<Vec<f32>, String> {
        self.fields
            .remove(field)
            .ok_or_else(|| format!("missing field: {field}"))
    }

    fn into_layer_weights(
        mut self,
        cfg: &MambaConfig,
        hf_cfg: &crate::hf::config_json::HfMambaConfig,
    ) -> Result<MambaLayerWeights, String> {
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dr = cfg.dt_rank();
        let xd = cfg.xdbl_dim();

        // PyTorch nn.Linear(in, out) stores weight as [out, in] row-major.
        // Our BLAS convention: W is [n_in, n_out] row-major (y = x @ W).
        // All four linear weight matrices must be transposed from HF layout.
        //
        // `use_conv_bias=false` checkpoints omit `conv1d_bias` entirely;
        // we synthesize a zero bias so the conv1d kernel (which always
        // reads a `bias[di]` tensor) sees a no-op. Same for `use_bias=true`
        // scenarios where the projection biases are present (we don't
        // currently support those, but we warn loudly here rather than
        // silently take the default zero-bias path).
        let conv1d_bias = if hf_cfg.use_conv_bias {
            self.take("conv1d_bias")?
        } else {
            vec![0.0f32; di]
        };
        if hf_cfg.use_bias {
            return Err(
                "HF config has use_bias=true (non-default) — projection biases \
                 on in_proj / out_proj / x_proj / dt_proj are not wired into \
                 mamba-rs inference kernels. Open an issue if you need this."
                    .to_string(),
            );
        }
        Ok(MambaLayerWeights {
            norm_weight: self.take("norm_weight")?,
            // HF: [2*d_inner, d_model] -> need [d_model, 2*d_inner]
            in_proj_w: transpose_2d(self.take("in_proj_w")?, 2 * di, d),
            // HF stores depthwise conv1d weight as [d_inner, 1, d_conv]
            // (PyTorch `nn.Conv1d(groups=d_inner)` layout; the singleton
            // middle dim is the per-group in-channel). mamba-rs flattens
            // that to [d_inner, d_conv] row-major and the CUDA / CPU
            // kernels index as `weight[d * d_conv + k]` — bit-identical
            // to the squeezed PyTorch layout, so no transpose is needed.
            // Source: mamba_ssm `MambaBlock.conv1d` in selective_scan /
            // models/mixer_seq_simple.py; layout matches nn.Conv1d docs.
            conv1d_weight: self.take("conv1d_weight")?,
            conv1d_bias,
            // HF: [xdbl_dim, d_inner] -> need [d_inner, xdbl_dim]
            x_proj_w: transpose_2d(self.take("x_proj_w")?, xd, di),
            // HF: [d_inner, dt_rank] -> need [dt_rank, d_inner]
            dt_proj_w: transpose_2d(self.take("dt_proj_w")?, di, dr),
            dt_proj_b: self.take("dt_proj_b")?,
            a_log: self.take("a_log")?,
            a_neg: vec![0.0; di * ds],
            d_param: self.take("d_param")?,
            // HF: [d_model, d_inner] -> need [d_inner, d_model]
            out_proj_w: transpose_2d(self.take("out_proj_w")?, d, di),
        })
    }
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
///
/// PyTorch nn.Linear stores weights as `[out_features, in_features]`.
/// Our BLAS expects `[n_in, n_out]`. This converts between the two.
fn transpose_2d(src: Vec<f32>, rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}
