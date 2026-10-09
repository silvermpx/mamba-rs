//! A restricted pickle machine for torch state dicts.
//!
//! `torch.save` pickles a state dict as an `OrderedDict` of tensors, each
//! rebuilt by `torch._utils._rebuild_tensor_v2` over a storage that the
//! pickle names by persistent id. This machine runs only the opcodes such a
//! pickle uses and resolves only those globals: `collections.OrderedDict`,
//! `torch._utils._rebuild_tensor_v2` and the bf16, f16 and f32 storage
//! types. Any other global, opcode or shape fails the load, so a hostile
//! `data.pkl` cannot run code; it can only be refused.

use std::collections::HashMap;

/// Element type of a stored tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDtype {
    F32,
    Bf16,
    F16,
}

impl TensorDtype {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 | Self::F16 => 2,
        }
    }
}

/// Where a state-dict tensor lives: its storage entry, element offset,
/// shape and (contiguous) strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRef {
    pub dtype: TensorDtype,
    pub storage_key: String,
    pub storage_offset: usize,
    pub shape: Vec<usize>,
}

impl TensorRef {
    pub fn numel(&self) -> Result<usize, String> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| "tensor element count overflows".to_string())
    }
}

#[derive(Clone, Debug)]
enum Value {
    None,
    /// A bool's value is never needed: `requires_grad` is the only one.
    Bool,
    Int(i64),
    Str(String),
    Tuple(Vec<Value>),
    Mark,
    OrderedDictType,
    RebuildTensor,
    StorageType(TensorDtype),
    Storage {
        dtype: TensorDtype,
        key: String,
    },
    Tensor(TensorRef),
    Dict(Vec<(Value, Value)>),
}

/// The bound on the pickle machine's stack and memo; a state dict needs a
/// few entries per tensor.
const MAX_ITEMS: usize = 1 << 20;

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.at.checked_add(n).ok_or("pickle: length overflows")?;
        let s = self
            .data
            .get(self.at..end)
            .ok_or("pickle: unexpected end of data")?;
        self.at = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn line(&mut self) -> Result<String, String> {
        let rest = self.data.get(self.at..).unwrap_or_default();
        let end = rest
            .iter()
            .position(|&c| c == b'\n')
            .ok_or("pickle: unterminated line")?;
        let s = std::str::from_utf8(&rest[..end])
            .map_err(|_| "pickle: line is not UTF-8")?
            .to_string();
        self.at += end + 1;
        Ok(s)
    }
}

fn pop(stack: &mut Vec<Value>) -> Result<Value, String> {
    stack
        .pop()
        .ok_or_else(|| "pickle: stack underflow".to_string())
}

fn pop_to_mark(stack: &mut Vec<Value>) -> Result<Vec<Value>, String> {
    let mark = stack
        .iter()
        .rposition(|v| matches!(v, Value::Mark))
        .ok_or("pickle: no mark on the stack")?;
    let items = stack.split_off(mark + 1);
    stack.pop();
    Ok(items)
}

fn as_usize(v: &Value, what: &str) -> Result<usize, String> {
    match v {
        Value::Int(i) if *i >= 0 => usize::try_from(*i).map_err(|_| format!("{what} too large")),
        _ => Err(format!("pickle: {what} is not a non-negative integer")),
    }
}

fn as_dims(v: &Value, what: &str) -> Result<Vec<usize>, String> {
    match v {
        Value::Tuple(items) => items.iter().map(|d| as_usize(d, what)).collect(),
        _ => Err(format!("pickle: {what} is not a tuple")),
    }
}

/// `_rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks)`.
fn rebuild_tensor(args: Vec<Value>) -> Result<Value, String> {
    let [storage, offset, size, stride, ..] = args.as_slice() else {
        return Err("pickle: _rebuild_tensor_v2 takes at least four arguments".into());
    };
    let Value::Storage { dtype, key } = storage else {
        return Err("pickle: tensor storage is not a storage".into());
    };
    let shape = as_dims(size, "tensor size")?;
    let stride = as_dims(stride, "tensor stride")?;
    if shape.len() != stride.len() {
        return Err("pickle: tensor size and stride differ in rank".into());
    }
    // Only contiguous row-major tensors occur in these checkpoints; any
    // other layout is refused rather than read wrongly.
    let mut expect = 1usize;
    for (&dim, &st) in shape.iter().zip(&stride).rev() {
        if dim > 1 && st != expect {
            return Err(format!(
                "pickle: non-contiguous tensor {shape:?} {stride:?}"
            ));
        }
        expect = expect
            .checked_mul(dim)
            .ok_or("pickle: tensor element count overflows")?;
    }
    Ok(Value::Tensor(TensorRef {
        dtype: *dtype,
        storage_key: key.clone(),
        storage_offset: as_usize(offset, "tensor offset")?,
        shape,
    }))
}

fn resolve_global(module: &str, name: &str) -> Result<Value, String> {
    match (module, name) {
        ("collections", "OrderedDict") => Ok(Value::OrderedDictType),
        ("torch._utils", "_rebuild_tensor_v2") => Ok(Value::RebuildTensor),
        ("torch", "FloatStorage") => Ok(Value::StorageType(TensorDtype::F32)),
        ("torch", "BFloat16Storage") => Ok(Value::StorageType(TensorDtype::Bf16)),
        ("torch", "HalfStorage") => Ok(Value::StorageType(TensorDtype::F16)),
        _ => Err(format!("pickle: global {module}.{name} is not allowed")),
    }
}

/// Run a state-dict pickle and return its tensors in order.
pub fn parse_state_dict(data: &[u8]) -> Result<Vec<(String, TensorRef)>, String> {
    let mut r = Reader { data, at: 0 };
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: HashMap<u32, Value> = HashMap::new();
    loop {
        if stack.len() > MAX_ITEMS || memo.len() > MAX_ITEMS {
            return Err("pickle: too many items".into());
        }
        let op = r.u8()?;
        match op {
            0x80 => {
                // PROTO
                let proto = r.u8()?;
                if proto > 5 {
                    return Err(format!("pickle: protocol {proto} not supported"));
                }
            }
            b'.' => {
                // STOP
                let top = pop(&mut stack)?;
                let Value::Dict(items) = top else {
                    return Err("pickle: the result is not a dict".into());
                };
                return items
                    .into_iter()
                    .map(|(k, v)| match (k, v) {
                        (Value::Str(name), Value::Tensor(t)) => Ok((name, t)),
                        _ => Err("pickle: state dict entry is not name -> tensor".to_string()),
                    })
                    .collect();
            }
            b'c' => {
                // GLOBAL
                let module = r.line()?;
                let name = r.line()?;
                stack.push(resolve_global(&module, &name)?);
            }
            b'q' => {
                // BINPUT
                let k = u32::from(r.u8()?);
                memo.insert(k, stack.last().ok_or("pickle: put on empty stack")?.clone());
            }
            b'r' => {
                // LONG_BINPUT
                let k = r.u32()?;
                memo.insert(k, stack.last().ok_or("pickle: put on empty stack")?.clone());
            }
            b'h' => {
                // BINGET
                let k = u32::from(r.u8()?);
                stack.push(memo.get(&k).ok_or("pickle: memo miss")?.clone());
            }
            b'j' => {
                // LONG_BINGET
                let k = r.u32()?;
                stack.push(memo.get(&k).ok_or("pickle: memo miss")?.clone());
            }
            b'(' => stack.push(Value::Mark),
            b')' => stack.push(Value::Tuple(Vec::new())),
            b't' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(Value::Tuple(items));
            }
            0x85 => {
                // TUPLE1
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(vec![a]));
            }
            0x86 => {
                // TUPLE2
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(vec![a, b]));
            }
            0x87 => {
                // TUPLE3
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(vec![a, b, c]));
            }
            b'J' => stack.push(Value::Int(i64::from(r.u32()? as i32))),
            b'K' => stack.push(Value::Int(i64::from(r.u8()?))),
            b'M' => stack.push(Value::Int(i64::from(r.u16()?))),
            0x8a => {
                // LONG1: little-endian two's complement of n bytes
                let n = usize::from(r.u8()?);
                if n > 8 {
                    return Err("pickle: integer wider than 64 bits".into());
                }
                let bytes = r.take(n)?;
                let mut v: i64 = 0;
                for (i, &b) in bytes.iter().enumerate() {
                    v |= i64::from(b) << (8 * i);
                }
                if n > 0 && n < 8 && bytes[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(Value::Int(v));
            }
            b'X' => {
                // BINUNICODE
                let n = r.u32()? as usize;
                let s =
                    std::str::from_utf8(r.take(n)?).map_err(|_| "pickle: string is not UTF-8")?;
                stack.push(Value::Str(s.to_string()));
            }
            0x8c => {
                // SHORT_BINUNICODE
                let n = usize::from(r.u8()?);
                let s =
                    std::str::from_utf8(r.take(n)?).map_err(|_| "pickle: string is not UTF-8")?;
                stack.push(Value::Str(s.to_string()));
            }
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool),
            0x89 => stack.push(Value::Bool),
            b'}' => stack.push(Value::Dict(Vec::new())),
            b'Q' => {
                // BINPERSID: ('storage', storage_type, key, location, numel)
                let pid = pop(&mut stack)?;
                let Value::Tuple(items) = pid else {
                    return Err("pickle: persistent id is not a tuple".into());
                };
                match items.as_slice() {
                    [
                        Value::Str(tag),
                        Value::StorageType(dtype),
                        Value::Str(key),
                        ..,
                    ] if tag == "storage" => {
                        if key.contains('/') || key.contains("..") {
                            return Err(format!("pickle: bad storage key {key:?}"));
                        }
                        stack.push(Value::Storage {
                            dtype: *dtype,
                            key: key.clone(),
                        });
                    }
                    _ => return Err("pickle: unsupported persistent id".into()),
                }
            }
            b'R' => {
                // REDUCE
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                let Value::Tuple(args) = args else {
                    return Err("pickle: reduce arguments are not a tuple".into());
                };
                match callable {
                    Value::OrderedDictType if args.is_empty() => {
                        stack.push(Value::Dict(Vec::new()))
                    }
                    Value::RebuildTensor => stack.push(rebuild_tensor(args)?),
                    _ => return Err("pickle: unsupported reduce".into()),
                }
            }
            b's' | b'u' => {
                // SETITEM / SETITEMS
                let items = if op == b's' {
                    let v = pop(&mut stack)?;
                    let k = pop(&mut stack)?;
                    vec![k, v]
                } else {
                    pop_to_mark(&mut stack)?
                };
                if items.len() % 2 != 0 {
                    return Err("pickle: odd number of dict items".into());
                }
                let Some(Value::Dict(dict)) = stack.last_mut() else {
                    return Err("pickle: setitems on a non-dict".into());
                };
                let mut it = items.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    dict.push((k, v));
                }
            }
            b'b' => {
                // BUILD: a state dict may carry `_metadata` this way; the
                // state is not needed and is dropped.
                pop(&mut stack)?;
                if !matches!(stack.last(), Some(Value::Dict(_))) {
                    return Err("pickle: build on a non-dict".into());
                }
            }
            0x95 => {
                // FRAME: the length is advisory.
                r.take(8)?;
            }
            0x94 => {
                // MEMOIZE
                let k = memo.len() as u32;
                memo.insert(
                    k,
                    stack
                        .last()
                        .ok_or("pickle: memoize on empty stack")?
                        .clone(),
                );
            }
            0x93 => {
                // STACK_GLOBAL
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                let (Value::Str(module), Value::Str(name)) = (module, name) else {
                    return Err("pickle: stack global names are not strings".into());
                };
                stack.push(resolve_global(&module, &name)?);
            }
            other => return Err(format!("pickle: opcode 0x{other:02x} not supported")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The opcode stream `torch.save` writes for a one-tensor state dict.
    fn one_tensor_pickle(global: &str) -> Vec<u8> {
        let mut p = vec![0x80, 2];
        p.extend_from_slice(b"ccollections\nOrderedDict\nq\x00)R");
        p.extend_from_slice(b"(X\x01\x00\x00\x00w");
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        p.extend_from_slice(b"((X\x07\x00\x00\x00storage");
        p.extend_from_slice(format!("ctorch\n{global}\n").as_bytes());
        p.extend_from_slice(b"X\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x06tQ");
        p.extend_from_slice(b"K\x00K\x02K\x03\x86K\x03K\x01\x86\x89h\x00)Rtq\x01Ru.");
        p
    }

    #[test]
    fn reads_a_torch_state_dict_entry() {
        let tensors = parse_state_dict(&one_tensor_pickle("BFloat16Storage")).unwrap();
        assert_eq!(
            tensors,
            vec![(
                "w".to_string(),
                TensorRef {
                    dtype: TensorDtype::Bf16,
                    storage_key: "0".into(),
                    storage_offset: 0,
                    shape: vec![2, 3],
                }
            )]
        );
    }

    #[test]
    fn refuses_globals_off_the_list() {
        let err = parse_state_dict(&one_tensor_pickle("IntStorage")).unwrap_err();
        assert!(err.contains("not allowed"), "{err}");
        let mut p = vec![0x80, 2];
        p.extend_from_slice(b"cos\nsystem\n.");
        assert!(parse_state_dict(&p).unwrap_err().contains("not allowed"));
    }

    #[test]
    fn refuses_truncated_and_unknown_input() {
        let full = one_tensor_pickle("FloatStorage");
        for cut in 0..full.len() {
            assert!(parse_state_dict(&full[..cut]).is_err(), "cut at {cut}");
        }
        assert!(parse_state_dict(&[0x80, 2, 0xff]).is_err());
    }
}
