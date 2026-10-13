//! The language-model checkpoint: one safetensors file with a backbone of
//! any family, its embedding, its untied head when it has one, and the
//! vocabulary size. The backbone's tensors and metadata are its family's
//! own checkpoint, so that family's loader reads the same file. Beside them
//! ride `embedding.weight` (`[vocab_rows, d_model]`, rows past the
//! vocabulary kept), `lm_head.weight` (`[d_model, vocab_size]`, the layout
//! the logits product reads, so the transpose of a Hugging Face head) and
//! the `vocab_size` metadata.

use std::path::Path;

use safetensors::tensor::SafeTensors;

use crate::config::ModelFamily;
use crate::ops::st_io::{checkpoint_family, meta, read_f32, read_f32_opt, read_meta};

use super::lm::AnyBackbone;
use super::{Mamba2Backbone, Mamba3Backbone, Mamba3MimoBackbone, MambaBackbone};

const EMBED: &str = "embedding.weight";
const HEAD: &str = "lm_head.weight";

/// A language model read from a checkpoint, before it is wrapped for the
/// host or the device.
pub(crate) struct LmParts {
    pub(crate) backbone: AnyBackbone,
    pub(crate) embed: Vec<f32>,
    pub(crate) lm_head: Option<Vec<f32>>,
    pub(crate) vocab_size: usize,
}

/// Write a language model to `path`.
pub(crate) fn save(
    path: &Path,
    backbone: &AnyBackbone,
    embed: &[f32],
    lm_head: Option<&[f32]>,
    vocab_size: usize,
) -> Result<(), String> {
    let (mut list, mut meta) = match backbone {
        AnyBackbone::M1(b) => {
            crate::mamba1::serialize::checkpoint_parts(b.weights(), b.config(), b.input_dim())?
        }
        AnyBackbone::M2(b) => {
            crate::mamba2::serialize::checkpoint_parts(b.weights(), b.config(), b.input_dim())?
        }
        AnyBackbone::M3(b) => {
            crate::mamba3_siso::serialize::checkpoint_parts(b.weights(), b.config(), b.input_dim())?
        }
        AnyBackbone::M3Mimo(b) => {
            crate::mamba3_mimo::serialize::checkpoint_parts(b.weights(), b.config(), b.input_dim())?
        }
    };
    let d_model = backbone.d_model();
    if d_model == 0 || !embed.len().is_multiple_of(d_model) {
        return Err(format!(
            "embed length {} is not a multiple of d_model {d_model}",
            embed.len()
        ));
    }
    list.push(EMBED.into(), embed, &[embed.len() / d_model, d_model])?;
    if let Some(head) = lm_head {
        list.push(HEAD.into(), head, &[d_model, vocab_size])?;
    }
    meta.insert("vocab_size".into(), vocab_size.to_string());
    list.save(meta, path)
}

/// Read a language model [`save`] wrote.
pub(crate) fn load(path: &Path) -> Result<LmParts, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let m = read_meta(&data)?;
    if !m.contains_key("vocab_size") {
        return Err(format!(
            "{} is not a language-model checkpoint: it has no vocab_size",
            path.display()
        ));
    }
    let vocab_size: usize = meta(&m, "vocab_size")?;
    let backbone: AnyBackbone = match checkpoint_family(&m)? {
        ModelFamily::Mamba1 => {
            let (w, cfg, input_dim) = crate::mamba1::serialize::load_from_bytes(&data)?;
            MambaBackbone::new(w, cfg, input_dim)?.into()
        }
        ModelFamily::Mamba2 => {
            let (w, cfg, input_dim) = crate::mamba2::serialize::load_from_bytes(&data)?;
            Mamba2Backbone::new(w, cfg, input_dim)?.into()
        }
        ModelFamily::Mamba3 => {
            let (w, cfg, input_dim) = crate::mamba3_siso::serialize::load_from_bytes(&data)?;
            Mamba3Backbone::new(w, cfg, input_dim)?.into()
        }
        ModelFamily::Mamba3Mimo => {
            let (w, cfg, input_dim) = crate::mamba3_mimo::serialize::load_from_bytes(&data)?;
            Mamba3MimoBackbone::new(w, cfg, input_dim)?.into()
        }
    };
    let st = SafeTensors::deserialize(&data).map_err(|e| format!("safetensors: {e:?}"))?;
    Ok(LmParts {
        backbone,
        embed: read_f32(&st, EMBED)?,
        lm_head: read_f32_opt(&st, HEAD)?,
        vocab_size,
    })
}
