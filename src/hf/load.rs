//! Top-level HuggingFace model loading.

use std::collections::HashMap;
use std::path::Path;

use crate::mamba1::config::MambaConfig;
use crate::mamba1::weights::{MambaLayerWeights, MambaWeights};
use crate::module::MambaBackbone;

use super::config_json::{ModelFamily, parse_config_json};
use super::keys::{RemappedKey, remap_m1_hf_key};
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
    let vocab_size_padded = (hf_cfg.vocab_size + 63) & !63;
    let embed_len = vocab_size_padded
        .checked_mul(hf_cfg.d_model)
        .ok_or("vocab_size_padded * d_model overflows usize")?;
    let mut embed = vec![0.0f32; embed_len];
    let mut lm_head: Option<Vec<f32>> = None;
    let mut norm_f_weight = vec![0.0f32; hf_cfg.d_model];
    let (mut embed_found, mut norm_f_found) = (false, false);
    let mut layer_weights: Vec<Option<LayerAccum>> = (0..hf_cfg.n_layers).map(|_| None).collect();

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
            RemappedKey::Layer(idx, field) => {
                if idx >= hf_cfg.n_layers {
                    return Err(format!("layer index {idx} >= n_layers {}", hf_cfg.n_layers));
                }
                let data = src.read(name)?;
                let accum = layer_weights[idx].get_or_insert_with(LayerAccum::default);
                accum.set(field, data)?;
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
            self.take_len("conv1d_bias", di)?
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
            norm_weight: self.take_len("norm_weight", d)?,
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
        })
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
