//! The trainer-state sidecar: what a trainer needs beside its weights
//! to begin the next window, in one safetensors file. The AdamW moments are
//! the tensors `optimizer.m` and `optimizer.v`, the carried recurrence the
//! tensors `recurrent.<name>`, and the optional integer tensor `seq_idx`
//! holds the packing. The metadata holds the family, the
//! optimizer's step and settings, the f16 loss scale and the recurrence's
//! names. The weights go to the family's own checkpoint.

use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::SafeTensors;
use safetensors::{Dtype, SafeTensorError};

use crate::config::ModelFamily;
use crate::ops::st_io::{TensorList, meta, read_f32, read_meta};
use crate::optim::adamw::AdamWStateBlob;
use crate::optim::loss_scaler::DynamicLossScaler;
use crate::varlen::Varlen;

/// The format marker in the metadata.
const FORMAT: &str = "mamba-rs-trainer-state";
// Older readers must refuse a packed checkpoint: ignoring its boundaries
// would silently change the next training example.
const PACKED_FORMAT: &str = "mamba-rs-packed-trainer-state";
const SEQ_IDX: &str = "seq_idx";

/// The state for the next forward after a closed accumulation window.
/// A saved forward's activations are local to the trainer that ran it.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainerState {
    /// The family whose trainer wrote the state; a trainer of another
    /// family refuses it.
    pub family: ModelFamily,
    /// The AdamW moments, step counter and update settings.
    pub optimizer: AdamWStateBlob,
    /// The f16 loss scale and its growth counter; `None` in the other
    /// precisions.
    pub scaler: Option<(f32, u32)>,
    /// The carried recurrence, tensor by tensor, as the family names it.
    pub recurrent: Vec<(String, Vec<f32>)>,
    /// The packing of the next windows. Older files load as unpacked.
    pub packing: Option<Varlen>,
}

fn family_name(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Mamba1 => "mamba1",
        ModelFamily::Mamba2 => "mamba2",
        ModelFamily::Mamba3 => "mamba3",
        ModelFamily::Mamba3Mimo => "mamba3-mimo",
    }
}

fn family_of(name: &str) -> Result<ModelFamily, String> {
    match name {
        "mamba1" => Ok(ModelFamily::Mamba1),
        "mamba2" => Ok(ModelFamily::Mamba2),
        "mamba3" => Ok(ModelFamily::Mamba3),
        "mamba3-mimo" => Ok(ModelFamily::Mamba3Mimo),
        other => Err(format!("unknown family '{other}' in the trainer state")),
    }
}

fn check_packing_family(family: ModelFamily, packed: bool) -> Result<(), String> {
    if family == ModelFamily::Mamba1 && packed {
        return Err("Mamba1 has no varlen: a Mamba1 trainer state cannot carry a packing".into());
    }
    Ok(())
}

fn read_packing(st: &SafeTensors<'_>, packed: bool) -> Result<Option<Varlen>, String> {
    let tensor = match st.tensor(SEQ_IDX) {
        Err(SafeTensorError::TensorNotFound(_)) if !packed => return Ok(None),
        Err(e) => return Err(format!("tensor '{SEQ_IDX}': {e}")),
        Ok(_) if !packed => return Err("an unpacked trainer state carries sequence ids".into()),
        Ok(t) => t,
    };
    let [batch, seq_len] = tensor.shape() else {
        return Err(format!(
            "trainer packing shape {:?}, expected [batch, seq_len]",
            tensor.shape()
        ));
    };
    if tensor.dtype() != Dtype::I32 {
        return Err(format!(
            "trainer packing dtype {:?}, expected I32",
            tensor.dtype()
        ));
    }
    let ids = tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&bytes| i32::from_le_bytes(bytes))
        .collect();
    Varlen::from_seq_idx(ids, *batch, *seq_len).map(Some)
}

impl TrainerState {
    /// Write the state to a safetensors file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        check_packing_family(self.family, self.packing.is_some())?;
        let o = &self.optimizer;
        let mut list = TensorList::new();
        let format = if let Some(p) = &self.packing {
            list.push_i32(SEQ_IDX.into(), p.seq_idx(), [p.batch(), p.seq_len()])?;
            PACKED_FORMAT
        } else {
            FORMAT
        };
        list.push("optimizer.m".into(), &o.m, &[o.m.len()])?;
        list.push("optimizer.v".into(), &o.v, &[o.v.len()])?;
        for (name, values) in &self.recurrent {
            if name.is_empty() || name.contains(',') {
                return Err(format!("recurrent tensor name '{name}'"));
            }
            list.push(format!("recurrent.{name}"), values, &[values.len()])?;
        }
        let names: Vec<&str> = self.recurrent.iter().map(|(n, _)| n.as_str()).collect();
        let mut m: HashMap<String, String> = [
            ("format", format.to_string()),
            ("family", family_name(self.family).to_string()),
            ("step", o.step.to_string()),
            ("beta1", o.beta1.to_string()),
            ("beta2", o.beta2.to_string()),
            ("eps", o.eps.to_string()),
            ("weight_decay", o.weight_decay.to_string()),
            ("reference_no_decay", o.reference_no_decay.to_string()),
            ("recurrent", names.join(",")),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        if let Some((scale, tracker)) = self.scaler {
            m.insert("loss_scale".into(), scale.to_string());
            m.insert("growth_tracker".into(), tracker.to_string());
        }
        list.save(m, path)
    }

    /// Read a state [`Self::save`] wrote.
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
        Self::load_from_bytes(&data)
    }

    /// [`Self::load`] from the bytes of a file.
    pub fn load_from_bytes(data: &[u8]) -> Result<Self, String> {
        let m = read_meta(data)?;
        let packed = match m.get("format").map(String::as_str) {
            Some(FORMAT) => false,
            Some(PACKED_FORMAT) => true,
            _ => return Err("not a trainer state: unknown or missing format marker".into()),
        };
        let family = family_of(&meta::<String>(&m, "family")?)?;
        check_packing_family(family, packed)?;
        let st = SafeTensors::deserialize(data).map_err(|e| format!("safetensors: {e:?}"))?;
        let packing = read_packing(&st, packed)?;
        let optimizer = AdamWStateBlob {
            m: read_f32(&st, "optimizer.m")?,
            v: read_f32(&st, "optimizer.v")?,
            step: meta(&m, "step")?,
            beta1: meta(&m, "beta1")?,
            beta2: meta(&m, "beta2")?,
            eps: meta(&m, "eps")?,
            weight_decay: meta(&m, "weight_decay")?,
            reference_no_decay: meta(&m, "reference_no_decay")?,
        };
        let scaler = match m.get("loss_scale") {
            Some(_) => Some((meta(&m, "loss_scale")?, meta(&m, "growth_tracker")?)),
            None => None,
        };
        let names: String = meta(&m, "recurrent")?;
        let recurrent = names
            .split(',')
            .filter(|n| !n.is_empty())
            .map(|n| Ok((n.to_string(), read_f32(&st, &format!("recurrent.{n}"))?)))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            family,
            optimizer,
            scaler,
            recurrent,
            packing,
        })
    }

    /// Borrow the named tensor only if it fits the destination's recurrence.
    pub(crate) fn recurrent_tensor(&self, name: &str, len: usize) -> Result<&[f32], String> {
        let values = self
            .recurrent
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, values)| values.as_slice())
            .ok_or_else(|| format!("the trainer state has no recurrence tensor '{name}'"))?;
        if values.len() != len {
            return Err(format!(
                "recurrence '{name}': expected {len} values, got {}",
                values.len()
            ));
        }
        Ok(values)
    }

    /// Check the shared settings before a trainer restores its own tensors.
    pub(crate) fn check_window(
        &self,
        family: ModelFamily,
        batch: usize,
        seq_len: usize,
        moments: usize,
    ) -> Result<(), String> {
        if self.family != family {
            return Err(format!(
                "a trainer state of {:?} loaded into a {family:?} trainer",
                self.family
            ));
        }
        check_packing_family(family, self.packing.is_some())?;
        if let Some(p) = &self.packing {
            p.check_shape(batch, seq_len)?;
        }
        if let Some((scale, _)) = self.scaler {
            DynamicLossScaler::check_scale(scale)?;
        }
        if self.optimizer.m.len() != moments || self.optimizer.v.len() != moments {
            return Err(format!(
                "AdamW state holds {} and {} moments for a model of {moments} parameters",
                self.optimizer.m.len(),
                self.optimizer.v.len()
            ));
        }
        Ok(())
    }
}
