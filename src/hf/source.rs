//! Tensors by name from the weight files of a checkpoint directory.
//!
//! A directory keeps its weights in one of four layouts, taken in this
//! order: safetensors shards (`model.safetensors.index.json`), one
//! `model.safetensors`, `pytorch_model.bin` shards
//! (`pytorch_model.bin.index.json`), or one `pytorch_model.bin`. Every file
//! is read tensor by tensor with positioned reads, never whole, so a load
//! holds the f32 weights and nothing the size of the file. The files are
//! untrusted input: every header length, offset and size is checked
//! against the file before anything is read or allocated from it.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use safetensors::Dtype;
use safetensors::tensor::TensorInfo;
use serde_json::Value;

use super::pickle::TensorRef;
use super::torch_zip::TorchBin;

/// The bound on a safetensors header, the format's own.
const MAX_SAFETENSORS_HEADER: u64 = 100_000_000;

/// The bound on a shard index: a weight map of a few thousand names.
const MAX_INDEX_BYTES: u64 = 64 << 20;

/// One safetensors file: the open file and where its data region starts.
struct SafetensorsFile {
    file: File,
    data_start: u64,
}

enum WeightFile {
    Safetensors(SafetensorsFile),
    Torch(TorchBin),
}

/// Adds one weight file, held to the tensors a shard index maps to it.
type AddFile = fn(&mut Source, &Path, Option<&HashSet<&str>>) -> Result<(), String>;

/// Where one tensor lives in its file.
#[derive(Clone, Debug)]
enum Stored {
    Safetensors { dtype: Dtype, begin: u64, len: u64 },
    Torch(TensorRef),
}

#[derive(Clone, Debug)]
struct Entry {
    file: usize,
    shape: Vec<usize>,
    stored: Stored,
}

/// The tensors of a checkpoint directory, read on demand as f32.
pub struct Source {
    files: Vec<WeightFile>,
    entries: HashMap<String, Entry>,
    /// The tensor names in file order, then each file's own order.
    names: Vec<String>,
}

impl Source {
    /// Open the weight files of `dir` and index their tensors.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let mut source = Self {
            files: Vec::new(),
            entries: HashMap::new(),
            names: Vec::new(),
        };
        let st_index = dir.join("model.safetensors.index.json");
        let st_single = dir.join("model.safetensors");
        let bin_index = dir.join("pytorch_model.bin.index.json");
        let bin_single = dir.join("pytorch_model.bin");
        if st_index.exists() {
            source.add_shards(dir, &st_index, Self::add_safetensors)?;
        } else if st_single.exists() {
            source.add_safetensors(&st_single, None)?;
        } else if bin_index.exists() {
            source.add_shards(dir, &bin_index, Self::add_torch)?;
        } else if bin_single.exists() {
            source.add_torch(&bin_single, None)?;
        } else {
            return Err(format!(
                "{}: no model.safetensors, pytorch_model.bin or shard index",
                dir.display()
            ));
        }
        Ok(source)
    }

    /// Every tensor name, in the order the files hold them.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The shape of tensor `name`, `None` when the checkpoint has no such
    /// tensor.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// Whether the checkpoint holds tensor `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Read tensor `name` as f32, row-major.
    pub fn read(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| format!("tensor {name} is missing"))?;
        match (&mut self.files[entry.file], &entry.stored) {
            (WeightFile::Safetensors(f), Stored::Safetensors { dtype, begin, len }) => {
                read_safetensor(f, (*dtype, *begin, *len), name)
            }
            (WeightFile::Torch(archive), Stored::Torch(t)) => archive.read_f32(t),
            _ => Err(format!("tensor {name}: index and file disagree")),
        }
    }

    fn insert(&mut self, name: String, entry: Entry) -> Result<(), String> {
        if self.entries.contains_key(&name) {
            return Err(format!("tensor {name} is stored twice"));
        }
        self.names.push(name.clone());
        self.entries.insert(name, entry);
        Ok(())
    }

    /// Add every file a shard index names, each holding exactly the tensors
    /// the index maps to it.
    fn add_shards(&mut self, dir: &Path, index: &Path, add: AddFile) -> Result<(), String> {
        let bytes = read_bounded(index, MAX_INDEX_BYTES)?;
        let root: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", index.display()))?;
        let map = root
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{}: no weight_map", index.display()))?;
        let mut by_file: Vec<(&str, HashSet<&str>)> = Vec::new();
        for (tensor, file) in map {
            let file = file
                .as_str()
                .ok_or_else(|| format!("{}: {tensor} maps to a non-string", index.display()))?;
            // A shard is a file beside the index, never a path elsewhere.
            if file.is_empty() || file.contains(['/', '\\']) || file.starts_with('.') {
                return Err(format!("{}: shard name {file:?}", index.display()));
            }
            match by_file.iter_mut().find(|(f, _)| *f == file) {
                Some((_, names)) => {
                    names.insert(tensor.as_str());
                }
                None => by_file.push((file, HashSet::from([tensor.as_str()]))),
            }
        }
        by_file.sort_by(|a, b| a.0.cmp(b.0));
        for (file, names) in by_file {
            add(self, &dir.join(file), Some(&names))?;
        }
        Ok(())
    }

    /// Add one safetensors file; with `expect`, the file must hold exactly
    /// those tensors.
    fn add_safetensors(
        &mut self,
        path: &Path,
        expect: Option<&HashSet<&str>>,
    ) -> Result<(), String> {
        let shown = path.display();
        let mut file = File::open(path).map_err(|e| format!("cannot open {shown}: {e}"))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("cannot stat {shown}: {e}"))?
            .len();
        let mut n_bytes = [0u8; 8];
        file.read_exact(&mut n_bytes)
            .map_err(|e| format!("{shown}: no safetensors header: {e}"))?;
        let header_len = u64::from_le_bytes(n_bytes);
        if header_len > MAX_SAFETENSORS_HEADER || 8 + header_len > file_len {
            return Err(format!("{shown}: header of {header_len} bytes"));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)
            .map_err(|e| format!("{shown}: header: {e}"))?;
        let header: serde_json::Map<String, Value> =
            serde_json::from_slice(&header).map_err(|e| format!("{shown}: header: {e}"))?;
        let data_start = 8 + header_len;
        let data_len = file_len - data_start;
        let index = self.files.len();
        let mut found = 0usize;
        let mut infos: Vec<(String, TensorInfo)> = Vec::with_capacity(header.len());
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let info: TensorInfo =
                serde_json::from_value(value).map_err(|e| format!("{shown}: {name}: {e}"))?;
            infos.push((name, info));
        }
        // The file's own order, by where each tensor's bytes start.
        infos.sort_by_key(|(_, info)| info.data_offsets.0);
        for (name, info) in infos {
            let (begin, end) = info.data_offsets;
            let numel = info
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| format!("{shown}: {name}: element count overflows"))?;
            let bytes = numel
                .checked_mul(info.dtype.bitsize())
                .map(|bits| bits.div_ceil(8))
                .ok_or_else(|| format!("{shown}: {name}: byte count overflows"))?;
            if begin > end || end as u64 > data_len || end - begin != bytes {
                return Err(format!(
                    "{shown}: {name}: offsets {begin}..{end} for {bytes} bytes of a \
                     {data_len}-byte data region"
                ));
            }
            if expect.is_some_and(|names| !names.contains(&name.as_str())) {
                return Err(format!(
                    "{shown}: {name} is not one the shard index maps to this file"
                ));
            }
            found += 1;
            let stored = Stored::Safetensors {
                dtype: info.dtype,
                begin: begin as u64,
                len: bytes as u64,
            };
            self.insert(
                name,
                Entry {
                    file: index,
                    shape: info.shape,
                    stored,
                },
            )?;
        }
        if let Some(names) = expect
            && found != names.len()
        {
            return Err(format!(
                "{shown}: holds {found} of the {} tensors the shard index maps to it",
                names.len()
            ));
        }
        self.files.push(WeightFile::Safetensors(SafetensorsFile {
            file,
            data_start,
        }));
        Ok(())
    }

    /// Add one `pytorch_model.bin`; with `expect`, the archive must hold
    /// exactly those tensors.
    fn add_torch(&mut self, path: &Path, expect: Option<&HashSet<&str>>) -> Result<(), String> {
        let archive = TorchBin::open(path)?;
        let index = self.files.len();
        let tensors = archive.tensors().to_vec();
        if let Some(names) = expect
            && tensors.len() != names.len()
        {
            return Err(format!(
                "{}: holds {} tensors, the shard index maps {} to it",
                path.display(),
                tensors.len(),
                names.len()
            ));
        }
        for (name, t) in tensors {
            if expect.is_some_and(|names| !names.contains(&name.as_str())) {
                return Err(format!(
                    "{}: {name} is not one the shard index maps to this file",
                    path.display()
                ));
            }
            self.insert(
                name,
                Entry {
                    file: index,
                    shape: t.shape.clone(),
                    stored: Stored::Torch(t),
                },
            )?;
        }
        self.files.push(WeightFile::Torch(archive));
        Ok(())
    }
}

/// Read a small file whole, refusing one past `bound` bytes.
fn read_bounded(path: &Path, bound: u64) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();
    if len > bound {
        return Err(format!("{}: {len} bytes, over the bound", path.display()));
    }
    let mut bytes = Vec::with_capacity(len as usize);
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(bytes)
}

/// One tensor of a safetensors file as f32. An f32 tensor is read straight
/// into its output; a half tensor is read and widened.
fn read_safetensor(
    f: &mut SafetensorsFile,
    (dtype, begin, len): (Dtype, u64, u64),
    name: &str,
) -> Result<Vec<f32>, String> {
    let len = usize::try_from(len).map_err(|_| format!("tensor {name} too large"))?;
    f.file
        .seek(SeekFrom::Start(f.data_start + begin))
        .map_err(|e| format!("tensor {name}: seek: {e}"))?;
    let read = |file: &mut File, buf: &mut [u8]| {
        file.read_exact(buf)
            .map_err(|e| format!("tensor {name}: read {len} bytes: {e}"))
    };
    match dtype {
        Dtype::F32 => {
            let mut out = vec![0f32; len / 4];
            read(&mut f.file, bytemuck::cast_slice_mut(&mut out))?;
            if cfg!(target_endian = "big") {
                out.iter_mut()
                    .for_each(|v| *v = f32::from_bits(u32::from_le(v.to_bits())));
            }
            Ok(out)
        }
        Dtype::BF16 | Dtype::F16 => {
            let mut raw = vec![0u16; len / 2];
            read(&mut f.file, bytemuck::cast_slice_mut(&mut raw))?;
            let widen = |h: u16| {
                let h = u16::from_le(h);
                match dtype {
                    Dtype::BF16 => f32::from_bits(u32::from(h) << 16),
                    _ => half::f16::from_bits(h).to_f32(),
                }
            };
            Ok(raw.into_iter().map(widen).collect())
        }
        other => Err(format!(
            "tensor {name}: dtype {other:?} is not a float weight"
        )),
    }
}
