//! Reading `pytorch_model.bin` (the zip archive `torch.save` writes)
//! without torch and without a zip dependency.
//!
//! The archive holds `<prefix>/data.pkl`, a pickled state dict whose
//! tensors point at raw storage entries `<prefix>/data/<key>`. `torch.save`
//! stores every entry uncompressed, so a tensor is a byte range of the
//! file, read with a positioned read and upcast to f32. Compressed entries
//! are refused. The file is untrusted input: every offset and length is
//! checked against the file before it is used, and allocations are sized
//! only after those checks.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::pickle::{TensorDtype, TensorRef, parse_state_dict};

const EOCD_SIG: u32 = 0x0605_4b50;
const ZIP64_LOCATOR_SIG: u32 = 0x0706_4b50;
const ZIP64_EOCD_SIG: u32 = 0x0606_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;

/// The bound on `data.pkl`: a state dict's pickle is a few hundred KB even
/// for billions of parameters, so anything larger is refused.
const MAX_PICKLE_BYTES: u64 = 64 << 20;

/// One stored archive entry: where its bytes start and how many there are.
#[derive(Clone, Copy, Debug)]
struct Entry {
    data_offset: u64,
    size: u64,
}

/// An opened `pytorch_model.bin`: the entry table and the parsed state dict.
pub struct TorchBin {
    file: File,
    entries: HashMap<String, Entry>,
    prefix: String,
    tensors: Vec<(String, TensorRef)>,
}

fn u16_at(b: &[u8], at: usize) -> Result<u16, String> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| "zip: record truncated".to_string())
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, String> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| "zip: record truncated".to_string())
}

fn u64_at(b: &[u8], at: usize) -> Result<u64, String> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
        .ok_or_else(|| "zip: record truncated".to_string())
}

fn read_at(file: &mut File, offset: u64, len: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("zip: seek to {offset}: {e}"))?;
    file.read_exact(&mut buf)
        .map_err(|e| format!("zip: read {len} bytes at {offset}: {e}"))?;
    Ok(buf)
}

impl TorchBin {
    /// Open the archive, read its central directory and parse `data.pkl`.
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut file =
            File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
            .len();
        let entries = read_central_directory(&mut file, file_len)?;

        let pickles: Vec<&String> = entries
            .keys()
            .filter(|n| n.ends_with("/data.pkl") || n.as_str() == "data.pkl")
            .collect();
        let [pkl_name] = pickles.as_slice() else {
            return Err(format!(
                "{}: expected exactly one data.pkl, found {}",
                path.display(),
                pickles.len()
            ));
        };
        let prefix = pkl_name
            .strip_suffix("data.pkl")
            .unwrap_or_default()
            .to_string();
        let pkl = entries[*pkl_name];
        if pkl.size > MAX_PICKLE_BYTES {
            return Err(format!("data.pkl is {} bytes, over the bound", pkl.size));
        }
        let pickle_bytes = read_at(&mut file, pkl.data_offset, pkl.size as usize)?;
        let tensors = parse_state_dict(&pickle_bytes)?;
        Ok(Self {
            file,
            entries,
            prefix,
            tensors,
        })
    }

    /// The state dict's tensors in pickle order: name and reference.
    pub fn tensors(&self) -> &[(String, TensorRef)] {
        &self.tensors
    }

    /// Read one tensor as f32, row-major, checked against its storage entry.
    pub fn read_f32(&mut self, tensor: &TensorRef) -> Result<Vec<f32>, String> {
        let name = format!("{}data/{}", self.prefix, tensor.storage_key);
        let entry = *self
            .entries
            .get(&name)
            .ok_or_else(|| format!("storage entry {name} is missing"))?;
        let elem = tensor.dtype.size_bytes() as u64;
        let numel = tensor.numel()? as u64;
        let start = (tensor.storage_offset as u64)
            .checked_mul(elem)
            .ok_or("tensor offset overflows")?;
        let len = numel.checked_mul(elem).ok_or("tensor size overflows")?;
        let end = start.checked_add(len).ok_or("tensor end overflows")?;
        if end > entry.size {
            return Err(format!(
                "tensor runs past its storage {name} ({end} > {} bytes)",
                entry.size
            ));
        }
        let raw = read_at(
            &mut self.file,
            entry.data_offset + start,
            usize::try_from(len).map_err(|_| "tensor too large")?,
        )?;
        Ok(match tensor.dtype {
            TensorDtype::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect(),
            TensorDtype::Bf16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f32::from_bits(u32::from(u16::from_le_bytes(c)) << 16))
                .collect(),
            TensorDtype::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| half::f16::from_bits(u16::from_le_bytes(c)).to_f32())
                .collect(),
        })
    }
}

/// Locate the end-of-central-directory record (zip64 aware) and read every
/// central-directory entry, resolving each one's data offset through its
/// local header. Only stored (uncompressed) entries are accepted.
fn read_central_directory(
    file: &mut File,
    file_len: u64,
) -> Result<HashMap<String, Entry>, String> {
    if file_len < 22 {
        return Err("zip: file too short".into());
    }
    // The EOCD record ends the file, followed by at most 65535 bytes of
    // comment.
    let tail_len = file_len.min(22 + 65535 + 20) as usize;
    let tail_start = file_len - tail_len as u64;
    let tail = read_at(file, tail_start, tail_len)?;
    let eocd = (0..=tail_len - 22)
        .rev()
        .find(|&i| u32_at(&tail, i).ok() == Some(EOCD_SIG))
        .ok_or("zip: end-of-central-directory record not found")?;
    let mut n_entries = u64::from(u16_at(&tail, eocd + 10)?);
    let mut cd_size = u64::from(u32_at(&tail, eocd + 12)?);
    let mut cd_offset = u64::from(u32_at(&tail, eocd + 16)?);
    if n_entries == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF {
        let loc = eocd.checked_sub(20).ok_or("zip: zip64 locator missing")?;
        if u32_at(&tail, loc)? != ZIP64_LOCATOR_SIG {
            return Err("zip: zip64 locator missing".into());
        }
        let z64_offset = u64_at(&tail, loc + 8)?;
        if z64_offset.checked_add(56).is_none_or(|e| e > file_len) {
            return Err("zip: zip64 record out of range".into());
        }
        let z64 = read_at(file, z64_offset, 56)?;
        if u32_at(&z64, 0)? != ZIP64_EOCD_SIG {
            return Err("zip: bad zip64 record".into());
        }
        n_entries = u64_at(&z64, 32)?;
        cd_size = u64_at(&z64, 40)?;
        cd_offset = u64_at(&z64, 48)?;
    }
    if cd_offset.checked_add(cd_size).is_none_or(|e| e > file_len) {
        return Err("zip: central directory out of range".into());
    }
    // Each entry takes at least 46 bytes, which bounds the count.
    if n_entries > cd_size / 46 {
        return Err("zip: entry count exceeds the central directory".into());
    }
    let cd = read_at(
        file,
        cd_offset,
        usize::try_from(cd_size).map_err(|_| "zip: central directory too large")?,
    )?;
    let mut entries = HashMap::with_capacity(n_entries as usize);
    let mut at = 0usize;
    for _ in 0..n_entries {
        if u32_at(&cd, at)? != CENTRAL_SIG {
            return Err("zip: bad central directory entry".into());
        }
        let method = u16_at(&cd, at + 10)?;
        let mut comp_size = u64::from(u32_at(&cd, at + 20)?);
        let mut uncomp_size = u64::from(u32_at(&cd, at + 24)?);
        let name_len = usize::from(u16_at(&cd, at + 28)?);
        let extra_len = usize::from(u16_at(&cd, at + 30)?);
        let comment_len = usize::from(u16_at(&cd, at + 32)?);
        let mut local_offset = u64::from(u32_at(&cd, at + 42)?);
        let name_bytes = cd
            .get(at + 46..at + 46 + name_len)
            .ok_or("zip: entry name truncated")?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| "zip: entry name is not UTF-8")?
            .to_string();
        let extra = cd
            .get(at + 46 + name_len..at + 46 + name_len + extra_len)
            .ok_or("zip: entry extra field truncated")?;
        // Zip64 extended information: 8-byte values for the fields that
        // read 0xFFFFFFFF, in the order uncompressed, compressed, offset.
        let mut e = 0usize;
        while e + 4 <= extra.len() {
            let id = u16_at(extra, e)?;
            let len = usize::from(u16_at(extra, e + 2)?);
            if id == 0x0001 {
                let mut f = e + 4;
                if uncomp_size == 0xFFFF_FFFF {
                    uncomp_size = u64_at(extra, f)?;
                    f += 8;
                }
                if comp_size == 0xFFFF_FFFF {
                    comp_size = u64_at(extra, f)?;
                    f += 8;
                }
                if local_offset == 0xFFFF_FFFF {
                    local_offset = u64_at(extra, f)?;
                }
            }
            e += 4 + len;
        }
        if method != 0 {
            return Err(format!("zip: entry {name} is compressed (method {method})"));
        }
        if comp_size != uncomp_size {
            return Err(format!("zip: stored entry {name} has mismatched sizes"));
        }
        if local_offset.checked_add(30).is_none_or(|e| e > file_len) {
            return Err(format!("zip: entry {name} header out of range"));
        }
        let local = read_at(file, local_offset, 30)?;
        if u32_at(&local, 0)? != LOCAL_SIG {
            return Err(format!("zip: bad local header for {name}"));
        }
        let data_offset =
            local_offset + 30 + u64::from(u16_at(&local, 26)?) + u64::from(u16_at(&local, 28)?);
        if data_offset
            .checked_add(comp_size)
            .is_none_or(|e| e > file_len)
        {
            return Err(format!("zip: entry {name} data out of range"));
        }
        entries.insert(
            name,
            Entry {
                data_offset,
                size: comp_size,
            },
        );
        at += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}
