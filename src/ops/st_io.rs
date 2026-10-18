//! What the safetensors serializers share: building the tensor list with
//! shapes, reading an f32 tensor back byte by byte, and reading the
//! configuration and the family from the file's metadata.

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use safetensors::serialize_to_file;
use safetensors::tensor::{SafeTensors, TensorView};

/// The tensors of one checkpoint, named, in the order they are written.
pub(crate) struct TensorList<'a> {
    tensors: Vec<(String, TensorView<'a>)>,
}

impl<'a> TensorList<'a> {
    pub(crate) fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    /// The elements of every tensor in the list.
    pub(crate) fn element_count(&self) -> usize {
        self.tensors
            .iter()
            .map(|(_, t)| t.shape().iter().product::<usize>())
            .sum()
    }

    /// Add `data` under `name` with `shape`, whose product must be its
    /// length. An empty tensor (an absent bias, an identity projection)
    /// is written with one dimension of zero.
    pub(crate) fn push(
        &mut self,
        name: String,
        data: &'a [f32],
        shape: &[usize],
    ) -> Result<(), String> {
        let shape: Vec<usize> = if data.is_empty() {
            vec![0]
        } else {
            shape.to_vec()
        };
        if shape.iter().product::<usize>() != data.len() {
            return Err(format!(
                "tensor '{name}': {} values do not fill the shape {shape:?}",
                data.len()
            ));
        }
        // The views borrow `data`; safetensors writes them little-endian,
        // which is the f32 layout in memory on every target this crate
        // builds for.
        let bytes: &'a [u8] = bytemuck_f32(data);
        let view = TensorView::new(safetensors::Dtype::F32, shape, bytes)
            .map_err(|e| format!("tensor '{name}': {e:?}"))?;
        self.tensors.push((name, view));
        Ok(())
    }

    /// The values of every tensor in the list.
    #[cfg(test)]
    pub(crate) fn elements(&self) -> usize {
        self.tensors.iter().map(|(_, v)| v.data().len() / 4).sum()
    }

    /// Write the tensors and `meta` to `path`.
    pub(crate) fn save(self, meta: HashMap<String, String>, path: &Path) -> Result<(), String> {
        serialize_to_file(self.tensors, Some(meta), path)
            .map_err(|e| format!("safetensors save failed: {e:?}"))
    }
}

/// The bytes of an f32 slice.
fn bytemuck_f32(data: &[f32]) -> &[u8] {
    // SAFETY: any f32 is four initialized bytes, u8 has alignment one, and
    // the length covers exactly the slice's memory.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

/// An f32 tensor of `st` as a vector. Decoded byte by byte: the format does
/// not promise four-byte alignment of the data section.
pub(crate) fn read_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>, String> {
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
}

/// [`read_f32`] of a tensor the file may not hold.
#[cfg(feature = "hf")]
pub(crate) fn read_f32_opt(st: &SafeTensors<'_>, name: &str) -> Result<Option<Vec<f32>>, String> {
    match st.tensor(name) {
        Err(safetensors::SafeTensorError::TensorNotFound(_)) => Ok(None),
        _ => read_f32(st, name).map(Some),
    }
}

/// The metadata of a safetensors file.
pub(crate) fn read_meta(data: &[u8]) -> Result<HashMap<String, String>, String> {
    let (_, header) = SafeTensors::read_metadata(data).map_err(|e| format!("metadata: {e}"))?;
    header
        .metadata()
        .clone()
        .ok_or_else(|| "the checkpoint carries no metadata".to_string())
}

/// A required metadata value, parsed.
pub(crate) fn meta<T: FromStr>(meta: &HashMap<String, String>, key: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    meta.get(key)
        .ok_or_else(|| format!("missing '{key}' in the checkpoint metadata"))?
        .parse()
        .map_err(|e| format!("parse '{key}': {e}"))
}

/// The block norms' kind as the metadata spells it: `rms_norm = false` for
/// LayerNorm, RMSNorm when the key is absent (every file written before the
/// option, and every RMSNorm model since, carries none).
pub(crate) fn block_norm(
    meta: &HashMap<String, String>,
) -> Result<crate::config::BlockNorm, String> {
    match meta.get("rms_norm").map(String::as_str) {
        None | Some("true") => Ok(crate::config::BlockNorm::Rms),
        Some("false") => Ok(crate::config::BlockNorm::Layer),
        Some(other) => Err(format!("unknown rms_norm '{other}' in the metadata")),
    }
}

/// The metadata entry of a non-default block norm kind: LayerNorm writes
/// `rms_norm = false`, RMSNorm nothing, so its files keep their bytes.
pub(crate) fn block_norm_entry(norm: crate::config::BlockNorm) -> Option<(String, String)> {
    (!norm.rms_norm()).then(|| ("rms_norm".to_string(), "false".to_string()))
}

/// Whether the backbone ends in its final norm: files written without the
/// entry, and every model built with the norm, do.
pub(crate) fn final_norm(meta: &HashMap<String, String>) -> Result<bool, String> {
    match meta.get("final_norm").map(String::as_str) {
        None | Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(other) => Err(format!("unknown final_norm '{other}' in the metadata")),
    }
}

/// The metadata entry of a backbone without its final norm, a segment of a
/// stack: a model with the norm writes nothing, so its files keep their
/// bytes.
pub(crate) fn final_norm_entry(final_norm: bool) -> Option<(String, String)> {
    (!final_norm).then(|| ("final_norm".to_string(), "false".to_string()))
}

/// Whether the layers read the sequence in both directions: files written
/// without the entry, and every model built one way, do not.
pub(crate) fn bidirectional(meta: &HashMap<String, String>) -> Result<bool, String> {
    match meta.get("bidirectional").map(String::as_str) {
        None | Some("false") => Ok(false),
        Some("true") => Ok(true),
        Some(other) => Err(format!("unknown bidirectional '{other}' in the metadata")),
    }
}

/// The metadata entry of a bidirectional model: a model read one way
/// writes nothing, so its files keep their bytes.
pub(crate) fn bidirectional_entry(bidirectional: bool) -> Option<(String, String)> {
    bidirectional.then(|| ("bidirectional".to_string(), "true".to_string()))
}

/// The scan mode as the metadata spells it.
pub(crate) fn scan_mode(meta: &HashMap<String, String>) -> Result<crate::config::ScanMode, String> {
    use crate::config::ScanMode;
    match meta.get("scan_mode").map(String::as_str) {
        Some("Sequential") => Ok(ScanMode::Sequential),
        Some("Parallel") => Ok(ScanMode::Parallel),
        Some("Auto") | None => Ok(ScanMode::Auto),
        Some(other) => Err(format!("unknown scan_mode '{other}' in the metadata")),
    }
}

/// The family a checkpoint's `mamba_version` names. Mamba-1 files written
/// before the marker carry none.
pub(crate) fn checkpoint_family(
    meta: &HashMap<String, String>,
) -> Result<crate::config::ModelFamily, String> {
    use crate::config::ModelFamily;
    match meta.get("mamba_version").map(String::as_str) {
        None | Some("1") => Ok(ModelFamily::Mamba1),
        Some("2") => Ok(ModelFamily::Mamba2),
        Some("3") => Ok(ModelFamily::Mamba3),
        Some("3-mimo") => Ok(ModelFamily::Mamba3Mimo),
        Some(other) => Err(format!("unknown mamba_version '{other}' in the metadata")),
    }
}
