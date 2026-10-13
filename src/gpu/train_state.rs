//! The trainer-state sidecar: what a GPU trainer needs beside its weights
//! to resume bit for bit, in one safetensors file. The AdamW moments are
//! the tensors `optimizer.m` and `optimizer.v`, the carried recurrence the
//! tensors `recurrent.<name>`, and the metadata holds the family, the
//! optimizer's step and settings, the f16 loss scale and the recurrence's
//! names. The weights go to the family's own checkpoint.

use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::SafeTensors;

use crate::config::ModelFamily;
use crate::gpu::adamw::AdamWStateBlob;
use crate::ops::st_io::{TensorList, meta, read_f32, read_meta};

/// The format marker in the metadata.
const FORMAT: &str = "mamba-rs-trainer-state";

/// Everything a trainer needs beside its weights to resume bit for bit.
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

impl TrainerState {
    /// Write the state to a safetensors file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let o = &self.optimizer;
        let mut list = TensorList::new();
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
            ("format", FORMAT.to_string()),
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
        if m.get("format").map(String::as_str) != Some(FORMAT) {
            return Err("not a trainer state: the format marker is missing".into());
        }
        let st = SafeTensors::deserialize(data).map_err(|e| format!("safetensors: {e:?}"))?;
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
            family: family_of(&meta::<String>(&m, "family")?)?,
            optimizer,
            scaler,
            recurrent,
        })
    }

    /// The recurrence tensor `name`; an error when the state has none. The
    /// trainer that loads it checks its length.
    pub(crate) fn recurrent_tensor(&self, name: &str) -> Result<&[f32], String> {
        self.recurrent
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, values)| values.as_slice())
            .ok_or_else(|| format!("the trainer state has no recurrence tensor '{name}'"))
    }

    /// An error unless the state was written by a trainer of `family`.
    pub(crate) fn check_family(&self, family: ModelFamily) -> Result<(), String> {
        if self.family == family {
            Ok(())
        } else {
            Err(format!(
                "a trainer state of {:?} loaded into a {family:?} trainer",
                self.family
            ))
        }
    }
}
