//! The matrix every family serves: what a caller asks for (a path, a
//! window, a scan, a precision, a GEMM mode) and the one place that says
//! whether the family runs it on the GPU, and how, or why not.
//!
//! Each family's `admit` (`mamba1::route::admit` and its siblings) is a
//! pure host function: the census walks every configuration `validate`
//! accepts through it without a device, and the GPU constructors ask it
//! before they build anything, so the two cannot disagree. A refusal is a
//! malformed request, a capacity the 32-bit kernel arguments impose, which
//! is legitimate, or a kernel that does not serve the shape, which is a
//! defect the census ledger lists until a kernel serves it.

use crate::config::ScanMode;
use crate::precision::{GemmMode, WeightDtype};

/// The work a request asks a family for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Path {
    /// One token per sequence on the carried state.
    Decode,
    /// A prompt window into the carried state.
    Prefill,
    /// A training window: the forward, the backward and the update.
    Train,
}

impl Path {
    pub const ALL: [Path; 3] = [Path::Decode, Path::Prefill, Path::Train];

    pub fn as_str(self) -> &'static str {
        match self {
            Path::Decode => "decode",
            Path::Prefill => "prefill",
            Path::Train => "train",
        }
    }
}

/// One request: a path over windows of `seq_len` tokens of `batch`
/// sequences, with the scan mode, the storage precision and the GEMM mode
/// the caller set. A decode window is one token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Request {
    pub path: Path,
    pub batch: usize,
    pub seq_len: usize,
    pub scan: ScanMode,
    pub precision: WeightDtype,
    pub gemm: GemmMode,
}

/// The scan an admitted request runs over its window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scan {
    /// The recurrence, token by token.
    Sequential,
    /// The family's parallel form: the prefix scan (Mamba-1) or the
    /// chunked scan (Mamba-2, Mamba-3).
    Parallel,
}

impl Scan {
    pub fn of(parallel: bool) -> Self {
        if parallel {
            Scan::Parallel
        } else {
            Scan::Sequential
        }
    }

    /// The scan an explicit mode names; `None` for `Auto`, which leaves the
    /// choice to the family.
    pub fn named(mode: ScanMode) -> Option<Self> {
        match mode {
            ScanMode::Sequential => Some(Scan::Sequential),
            ScanMode::Parallel => Some(Scan::Parallel),
            ScanMode::Auto => None,
        }
    }
}

/// How a family runs an admitted request.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Plan {
    pub scan: Scan,
}

/// Why a family does not run a request.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Refusal {
    /// The request names no work: a zero batch or window, or a decode
    /// window of more than one token.
    Request { detail: String },
    /// The request needs more of `resource` than the 32-bit kernel
    /// arguments carry.
    Capacity {
        resource: &'static str,
        need: u64,
        have: u64,
    },
    /// No kernel serves the shape: a defect the census ledger lists until
    /// one does. `predicate` names the condition, `detail` the values.
    Kernel {
        predicate: &'static str,
        detail: String,
    },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Request { detail } => write!(f, "{detail}"),
            Refusal::Capacity {
                resource,
                need,
                have,
            } => write!(
                f,
                "{resource} {need} is past the {have} the GPU kernels take it in"
            ),
            Refusal::Kernel { predicate, detail } => {
                write!(f, "no GPU kernel serves {predicate}: {detail}")
            }
        }
    }
}

impl From<Refusal> for String {
    fn from(r: Refusal) -> String {
        r.to_string()
    }
}

/// The largest value a kernel takes as an `int`: `i32::MAX` less the
/// headroom kernels step past their bounds with. A loop counter strided by
/// the block ends up to a block past its bound, and a time row is padded by
/// a few elements, so a value right under `i32::MAX` would wrap them.
pub const INT_ARG_MAX: usize = i32::MAX as usize - (1 << 16);

/// Refuses the first of `args` past [`INT_ARG_MAX`]. Each is a dimension, a
/// row count or a row width a kernel takes as an `int`; element counts and
/// offsets are 64-bit, so the size of a tensor is bounded by memory alone.
pub(crate) fn int_args(args: &[(&'static str, usize)]) -> Result<(), Refusal> {
    match args.iter().find(|(_, v)| *v > INT_ARG_MAX) {
        Some(&(resource, need)) => Err(Refusal::Capacity {
            resource,
            need: need as u64,
            have: INT_ARG_MAX as u64,
        }),
        None => Ok(()),
    }
}

/// Refuses a request that names no work.
pub(crate) fn check_request(req: &Request) -> Result<(), Refusal> {
    if req.batch == 0 || req.seq_len == 0 {
        return Err(Refusal::Request {
            detail: format!(
                "a window needs at least one sequence and one token, got batch {} and \
                 seq_len {}",
                req.batch, req.seq_len
            ),
        });
    }
    if req.path == Path::Decode && req.seq_len != 1 {
        return Err(Refusal::Request {
            detail: format!(
                "a decode step is one token per sequence, got seq_len {}",
                req.seq_len
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(path: Path, batch: usize, seq_len: usize) -> Request {
        Request {
            path,
            batch,
            seq_len,
            scan: ScanMode::Auto,
            precision: WeightDtype::F32,
            gemm: GemmMode::Deterministic,
        }
    }

    #[test]
    fn an_empty_or_multi_token_decode_request_is_refused() {
        assert!(check_request(&req(Path::Train, 0, 4)).is_err());
        assert!(check_request(&req(Path::Prefill, 2, 0)).is_err());
        assert!(check_request(&req(Path::Decode, 2, 2)).is_err());
        assert!(check_request(&req(Path::Decode, 2, 1)).is_ok());
        assert!(check_request(&req(Path::Train, 1, 1)).is_ok());
    }

    #[test]
    fn the_first_int_argument_past_the_bound_names_itself() {
        assert_eq!(int_args(&[("a", INT_ARG_MAX), ("b", 1)]), Ok(()));
        let err = int_args(&[("a", 1), ("rows", INT_ARG_MAX + 1), ("c", usize::MAX)]);
        assert_eq!(
            err,
            Err(Refusal::Capacity {
                resource: "rows",
                need: INT_ARG_MAX as u64 + 1,
                have: INT_ARG_MAX as u64,
            })
        );
        assert!(String::from(err.unwrap_err()).contains("rows"));
    }
}
