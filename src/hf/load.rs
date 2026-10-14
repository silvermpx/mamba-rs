//! Top-level HuggingFace model loading.

use std::collections::HashMap;
use std::path::Path;

use crate::mamba1::config::MambaConfig;
use crate::mamba1::weights::{MambaLayerWeights, MambaWeights};
use crate::module::MambaBackbone;

use super::config_json::{ModelFamily, parse_config_json};
use super::keys::{RemappedKey, remap_m1_hf_key};
use super::load_mamba3::read_mlp;
use super::source::Source;

/// Loaded HF Mamba model components.
pub struct HfModel {
    pub backbone: MambaBackbone,
    pub embed: Vec<f32>,
    pub lm_head: Option<Vec<f32>>,
    pub vocab_size: usize,
    pub vocab_size_padded: usize,
    pub d_model: usize,
}

/// Load a Mamba-1 checkpoint directory: `config.json` in the HF or the
/// original format, and the weights in any layout [`Source`] reads (one or
/// several safetensors files, or the `pytorch_model.bin` the original
/// repositories ship).
pub fn load_hf(dir: &Path) -> Result<HfModel, String> {
    let config_path = dir.join("config.json");
    let config_bytes =
        std::fs::read(&config_path).map_err(|e| format!("cannot read config.json: {e}"))?;
    let hf_cfg = parse_config_json(&config_bytes)?;

    if hf_cfg.family != ModelFamily::Mamba1 {
        return Err(format!(
            "this loader reads Mamba-1 checkpoints, got {:?} (Mamba-2 and Mamba-3 \
             load through hf::load_mamba2 and hf::load_mamba3)",
            hf_cfg.family
        ));
    }

    let mut src = Source::open(dir)?;

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
    if hf_cfg.n_layers == 0 || hf_cfg.n_layers > crate::config::envelope::MAX_LAYERS {
        return Err(format!(
            "n_layers {} out of supported range (1..=4096)",
            hf_cfg.n_layers
        ));
    }
    if hf_cfg.d_intermediate > crate::config::envelope::MAX_WIDTH {
        return Err(format!(
            "d_intermediate {} out of supported range (0..=2^20)",
            hf_cfg.d_intermediate
        ));
    }
    let vocab_size_padded = (hf_cfg.vocab_size + 63) & !63;
    let embed_len = vocab_size_padded
        .checked_mul(hf_cfg.d_model)
        .ok_or("vocab_size_padded * d_model overflows usize")?;
    let mut embed = vec![0.0f32; embed_len];
    let mut lm_head: Option<Vec<f32>> = None;
    let mut norm_f_weight = vec![0.0f32; hf_cfg.d_model];
    // A LayerNorm model's final bias; an RMSNorm one has none.
    let layer_norm = !hf_cfg.block_norm.rms_norm();
    let mut norm_f_bias: Option<Vec<f32>> = None;
    let (mut embed_found, mut norm_f_found) = (false, false);
    let mut layer_weights: Vec<Option<LayerAccum>> = (0..hf_cfg.n_layers).map(|_| None).collect();
    let hidden = crate::mlp::mlp_hidden(hf_cfg.d_intermediate);

    let names = src.names().to_vec();
    for name in &names {
        match remap_m1_hf_key(name) {
            RemappedKey::Embedding => {
                let data = src.read(name)?;
                let expected = hf_cfg.vocab_size * hf_cfg.d_model;
                if data.len() != expected {
                    return Err(format!(
                        "embedding shape: expected {expected} elements, got {}",
                        data.len()
                    ));
                }
                embed[..expected].copy_from_slice(&data);
                embed_found = true;
            }
            RemappedKey::LmHead => {
                // A tied head saved beside the embedding is not read.
                if !hf_cfg.tie_word_embeddings {
                    lm_head = Some(src.read(name)?);
                }
            }
            RemappedKey::NormF => {
                let data = src.read(name)?;
                if data.len() != hf_cfg.d_model {
                    return Err(format!(
                        "norm_f_weight: expected {}, got {}",
                        hf_cfg.d_model,
                        data.len()
                    ));
                }
                norm_f_weight.copy_from_slice(&data);
                norm_f_found = true;
            }
            RemappedKey::NormFBias => {
                if !layer_norm {
                    return Err(format!(
                        "unexpected tensor {name}: an RMSNorm model's final norm has no bias"
                    ));
                }
                let data = src.read(name)?;
                if data.len() != hf_cfg.d_model {
                    return Err(format!(
                        "norm_f_bias: expected {}, got {}",
                        hf_cfg.d_model,
                        data.len()
                    ));
                }
                norm_f_bias = Some(data);
            }
            RemappedKey::Layer(idx, field) => {
                if idx >= hf_cfg.n_layers {
                    return Err(format!("layer index {idx} >= n_layers {}", hf_cfg.n_layers));
                }
                let data = src.read(name)?;
                let accum = layer_weights[idx].get_or_insert_with(LayerAccum::default);
                accum.set(field, data)?;
            }
            // The gated MLP's tensors are read whole once every name is
            // known; here each is held to the shape the config gives it.
            RemappedKey::Mlp(idx, part) => {
                let d = hf_cfg.d_model;
                if hidden == 0 || idx >= hf_cfg.n_layers {
                    return Err(format!(
                        "unexpected tensor {name}: the config has no gated MLP in layer {idx}"
                    ));
                }
                if part == "norm2_bias" && !layer_norm {
                    return Err(format!(
                        "unexpected tensor {name}: an RMSNorm model's norm2 has no bias"
                    ));
                }
                let want = match part {
                    "norm2" | "norm2_bias" => vec![d],
                    "fc1" => vec![2 * hidden, d],
                    _ => vec![d, hidden],
                };
                if src.shape(name) != Some(want.as_slice()) {
                    return Err(format!(
                        "tensor {name}: shape {:?}, expected {want:?}",
                        src.shape(name)
                    ));
                }
            }
            // A tensor this model has no place for is a part it does
            // not model. Loading without it would run another network.
            RemappedKey::Unknown => {
                return Err(format!(
                    "unexpected tensor {name}: the checkpoint holds a part this \
                     Mamba-1 model does not have"
                ));
            }
        }
    }
    if !embed_found {
        return Err("the checkpoint has no embedding tensor".into());
    }
    if !norm_f_found {
        return Err("the checkpoint has no norm_f tensor".into());
    }
    let norm_f_bias = match (layer_norm, norm_f_bias) {
        (false, _) => Vec::new(),
        (true, Some(bias)) => bias,
        (true, None) => return Err("the LayerNorm checkpoint has no norm_f bias".into()),
    };

    if !hf_cfg.tie_word_embeddings && lm_head.is_none() {
        return Err("tie_word_embeddings=false but lm_head.weight not found".into());
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
        *lm = transpose_2d(
            std::mem::take(lm),
            (hf_cfg.vocab_size, hf_cfg.d_model),
            "lm_head.weight",
        )?;
    }

    let mamba_cfg = MambaConfig {
        d_model: hf_cfg.d_model,
        d_state: hf_cfg.d_state,
        d_conv: hf_cfg.d_conv,
        expand: hf_cfg.expand,
        n_layers: hf_cfg.n_layers,
        block_norm: hf_cfg.block_norm,
        // Checkpoint-specified norm epsilon (FalconMamba: 1e-6) — plumbed
        // through every CPU/GPU inference norm via MambaConfig/MambaDims.
        rms_norm_eps: hf_cfg.rms_norm_eps,
        bcdt_rms_eps: hf_cfg.bcdt_rms_eps,
        dt_rank: hf_cfg.dt_rank,
        proj_bias: hf_cfg.use_bias,
        conv_bias: hf_cfg.use_conv_bias,
        d_intermediate: hf_cfg.d_intermediate,
        ..MambaConfig::default()
    };

    let layers: Vec<MambaLayerWeights> = layer_weights
        .into_iter()
        .enumerate()
        .map(|(i, opt)| {
            opt.ok_or(format!("layer {i}: no weights found"))
                .and_then(|a| a.into_layer_weights(&mamba_cfg))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mlp = (0..hf_cfg.n_layers)
        .filter(|_| hidden > 0)
        .map(|i| read_mlp(&mut src, i, (hf_cfg.d_model, hidden), hf_cfg.block_norm))
        .collect::<Result<Vec<_>, _>>()?;

    let weights = MambaWeights {
        input_proj_w: vec![],
        input_proj_b: vec![],
        layers,
        norm_f_weight,
        norm_f_bias,
        mlp,
    };

    let backbone = MambaBackbone::new(weights, mamba_cfg, mamba_cfg.d_model)?;

    Ok(HfModel {
        backbone,
        embed,
        lm_head,
        vocab_size: hf_cfg.vocab_size,
        vocab_size_padded,
        d_model: hf_cfg.d_model,
    })
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

    /// [`Self::take`] for a tensor that must hold exactly `len` values.
    fn take_len(&mut self, field: &str, len: usize) -> Result<Vec<f32>, String> {
        let data = self.take(field)?;
        if data.len() != len {
            return Err(format!(
                "{field}: expected {len} values for this config, the checkpoint holds {}",
                data.len()
            ));
        }
        Ok(data)
    }

    fn into_layer_weights(mut self, cfg: &MambaConfig) -> Result<MambaLayerWeights, String> {
        let d = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dr = cfg.dt_rank();
        let xd = cfg.xdbl_dim();

        // PyTorch nn.Linear(in, out) stores weight as [out, in] row-major.
        // Our BLAS convention: W is [n_in, n_out] row-major (y = x @ W).
        // All four linear weight matrices must be transposed from HF layout.
        //
        // A bias the config turns off is absent from the checkpoint and
        // stays empty here, as the model holds it.
        let mut optional = |on: bool, field: &str, len: usize| {
            if on {
                self.take_len(field, len)
            } else {
                Ok(Vec::new())
            }
        };
        let norm_bias = optional(!cfg.block_norm.rms_norm(), "norm_bias", d)?;
        let in_proj_b = optional(cfg.proj_bias, "in_proj_b", 2 * di)?;
        let out_proj_b = optional(cfg.proj_bias, "out_proj_b", d)?;
        let conv1d_bias = optional(cfg.conv_bias, "conv1d_bias", di)?;
        let layer = MambaLayerWeights {
            norm_weight: self.take_len("norm_weight", d)?,
            norm_bias,
            // HF: [2*d_inner, d_model] -> need [d_model, 2*d_inner]
            in_proj_w: transpose_2d(self.take("in_proj_w")?, (2 * di, d), "in_proj_w")?,
            // HF stores depthwise conv1d weight as [d_inner, 1, d_conv]
            // (PyTorch `nn.Conv1d(groups=d_inner)` layout; the singleton
            // middle dim is the per-group in-channel). mamba-rs flattens
            // that to [d_inner, d_conv] row-major and the CUDA / CPU
            // kernels index as `weight[d * d_conv + k]` — bit-identical
            // to the squeezed PyTorch layout, so no transpose is needed.
            // Source: mamba_ssm `MambaBlock.conv1d` in selective_scan /
            // models/mixer_seq_simple.py; layout matches nn.Conv1d docs.
            conv1d_weight: self.take_len("conv1d_weight", di * cfg.d_conv)?,
            conv1d_bias,
            in_proj_b,
            // HF: [xdbl_dim, d_inner] -> need [d_inner, xdbl_dim]
            x_proj_w: transpose_2d(self.take("x_proj_w")?, (xd, di), "x_proj_w")?,
            // HF: [d_inner, dt_rank] -> need [dt_rank, d_inner]
            dt_proj_w: transpose_2d(self.take("dt_proj_w")?, (di, dr), "dt_proj_w")?,
            dt_proj_b: self.take_len("dt_proj_b", di)?,
            a_log: self.take_len("a_log", di * ds)?,
            a_neg: vec![0.0; di * ds],
            d_param: self.take_len("d_param", di)?,
            // HF: [d_model, d_inner] -> need [d_inner, d_model]
            out_proj_w: transpose_2d(self.take("out_proj_w")?, (d, di), "out_proj_w")?,
            out_proj_b,
        };
        // A tensor left over is a part the config says the model lacks.
        if let Some(field) = self.fields.keys().next() {
            return Err(format!(
                "the checkpoint holds {field}, a part this config's model does not have"
            ));
        }
        Ok(layer)
    }
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`, refusing
/// a tensor whose size is not that shape's.
///
/// PyTorch nn.Linear stores weights as `[out_features, in_features]`.
/// Our BLAS expects `[n_in, n_out]`. This converts between the two.
fn transpose_2d(
    src: Vec<f32>,
    (rows, cols): (usize, usize),
    name: &str,
) -> Result<Vec<f32>, String> {
    if src.len() != rows * cols {
        return Err(format!(
            "{name}: expected [{rows}, {cols}] for this config, the checkpoint holds {} values",
            src.len()
        ));
    }
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    Ok(dst)
}
