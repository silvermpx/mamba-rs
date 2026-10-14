//! Several sequences of different lengths packed into the rows of one
//! window (varlen), as the reference trains Mamba-2 and Mamba-3 with
//! `seq_idx` and `cu_seqlens`: no state, conv history, previous-step term,
//! RoPE angle or gradient crosses from one sequence into the next.
//!
//! Every position carries the id of its sequence, non-decreasing along a
//! row. Id 0 is the sequence the trainer carried into the window from the
//! window before; any other id starts from an empty state. A position whose
//! id differs from the one before it (from 0 for the first position)
//! starts a sequence. After the window the trainer carries the last
//! sequence of each row on, so a sequence may run across windows and the
//! next window drops it by starting a sequence at its first position.
//! Mamba-1 has no varlen in the reference and refuses one.

/// The sequence ids of a window's positions, `[batch, seq_len]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Varlen {
    seq_idx: Vec<i32>,
    batch: usize,
    seq_len: usize,
}

impl Varlen {
    /// From the reference's `seq_idx`: `batch` rows of `seq_len` ids, each
    /// row non-decreasing and every id non-negative.
    pub fn from_seq_idx(seq_idx: Vec<i32>, batch: usize, seq_len: usize) -> Result<Self, String> {
        let want = batch
            .checked_mul(seq_len)
            .ok_or("varlen: batch * seq_len overflows usize")?;
        if seq_idx.len() != want {
            return Err(format!(
                "varlen: {} ids for {batch} rows of {seq_len} positions",
                seq_idx.len()
            ));
        }
        if seq_len > 0 {
            for (b, row) in seq_idx.chunks_exact(seq_len).enumerate() {
                if let Some(t) = row.iter().position(|&id| id < 0) {
                    return Err(format!("varlen: row {b} position {t} has a negative id"));
                }
                if let Some(t) = row.windows(2).position(|w| w[1] < w[0]) {
                    return Err(format!("varlen: row {b} decreases at position {}", t + 1));
                }
            }
        }
        Ok(Self {
            seq_idx,
            batch,
            seq_len,
        })
    }

    /// From the positions where sequences start in each row of `seq_len`
    /// positions: `starts[b]` ascending, each below `seq_len`. The
    /// positions before a row's first start continue the carried
    /// sequence; a start at 0 drops it.
    pub fn from_starts(starts: &[Vec<usize>], seq_len: usize) -> Result<Self, String> {
        let mut seq_idx = Vec::with_capacity(starts.len() * seq_len);
        for (b, row) in starts.iter().enumerate() {
            if row.windows(2).any(|w| w[1] <= w[0]) {
                return Err(format!("varlen: row {b}'s starts do not ascend"));
            }
            if row.last().is_some_and(|&t| t >= seq_len) {
                return Err(format!(
                    "varlen: row {b} starts a sequence past its {seq_len} positions"
                ));
            }
            let mut id = 0i32;
            let mut next = row.iter().peekable();
            for t in 0..seq_len {
                if next.next_if(|&&s| s == t).is_some() {
                    id = id
                        .checked_add(1)
                        .ok_or("varlen: more sequences than i32 ids")?;
                }
                seq_idx.push(id);
            }
        }
        Self::from_seq_idx(seq_idx, starts.len(), seq_len)
    }

    /// From the reference's `cu_seqlens` of one row: the cumulative lengths
    /// `0 = cu[0] < cu[1] < ... `, the last one the row's length. Every
    /// sequence starts from an empty state, the first one included.
    pub fn from_cu_seqlens(cu_seqlens: &[usize]) -> Result<Self, String> {
        let (&first, &last) = cu_seqlens
            .first()
            .zip(cu_seqlens.last())
            .ok_or("varlen: cu_seqlens is empty")?;
        if first != 0 {
            return Err("varlen: cu_seqlens starts past 0".into());
        }
        let starts: Vec<usize> = cu_seqlens[..cu_seqlens.len() - 1].to_vec();
        if starts.is_empty() && last > 0 {
            return Err("varlen: cu_seqlens holds no sequence".into());
        }
        Self::from_starts(&[starts], last)
    }

    /// Every position's sequence id, row after row.
    pub fn seq_idx(&self) -> &[i32] {
        &self.seq_idx
    }

    /// The rows.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The positions of a row.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// The ids of row `b`.
    pub fn row(&self, b: usize) -> &[i32] {
        &self.seq_idx[b * self.seq_len..(b + 1) * self.seq_len]
    }

    /// An error unless the packing covers `batch` rows of `seq_len`.
    pub(crate) fn check_shape(&self, batch: usize, seq_len: usize) -> Result<(), String> {
        if (self.batch, self.seq_len) != (batch, seq_len) {
            return Err(format!(
                "varlen: a packing of {} rows of {} positions for a window of {batch} rows of \
                 {seq_len}",
                self.batch, self.seq_len
            ));
        }
        Ok(())
    }
}

/// One row's packing as the host paths read it: whether each position
/// starts a sequence, and the first position of each position's sequence
/// (`isize::MIN` for the carried one, whose history reaches back before the
/// window).
#[derive(Clone, Debug, Default)]
pub(crate) struct RowStarts {
    start: Vec<bool>,
    first: Vec<isize>,
}

impl RowStarts {
    pub(crate) fn from_ids(row: &[i32]) -> Self {
        let mut first = Vec::with_capacity(row.len());
        let mut cur = isize::MIN;
        let start: Vec<bool> = (0..row.len())
            .map(|t| {
                let s = starts_at(row, t);
                if s {
                    cur = t as isize;
                }
                first.push(cur);
                s
            })
            .collect();
        Self { start, first }
    }

    /// Whether position `t` starts a sequence.
    #[inline(always)]
    pub(crate) fn starts(&self, t: usize) -> bool {
        self.start[t]
    }

    /// The first position of `t`'s sequence, `isize::MIN` for the carried
    /// one.
    #[inline(always)]
    pub(crate) fn first(&self, t: usize) -> isize {
        self.first[t]
    }
}

/// Whether position `t` of a row with ids `row` starts a sequence: its id
/// differs from the one before it, or from 0 at the first position.
#[inline(always)]
pub(crate) fn starts_at(row: &[i32], t: usize) -> bool {
    row[t] != if t == 0 { 0 } else { row[t - 1] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_build_the_reference_ids() {
        let v = Varlen::from_starts(&[vec![0, 3], vec![2]], 5).unwrap();
        assert_eq!(v.seq_idx(), &[1, 1, 1, 2, 2, 0, 0, 1, 1, 1]);
        assert!(starts_at(v.row(0), 0) && starts_at(v.row(0), 3));
        assert!(!starts_at(v.row(1), 0) && starts_at(v.row(1), 2));
        let cu = Varlen::from_cu_seqlens(&[0, 2, 5]).unwrap();
        assert_eq!(cu.seq_idx(), &[1, 1, 2, 2, 2]);
        let r = RowStarts::from_ids(v.row(1));
        assert_eq!(
            (0..5).map(|t| r.starts(t)).collect::<Vec<_>>(),
            [false, false, true, false, false]
        );
        assert_eq!(r.first(1), isize::MIN);
        assert_eq!(r.first(4), 2);
    }

    #[test]
    fn bad_packings_are_refused() {
        assert!(Varlen::from_seq_idx(vec![0, 1, 0], 1, 3).is_err());
        assert!(Varlen::from_seq_idx(vec![0, -1], 1, 2).is_err());
        assert!(Varlen::from_seq_idx(vec![0, 1], 1, 3).is_err());
        assert!(Varlen::from_starts(&[vec![2, 1]], 4).is_err());
        assert!(Varlen::from_starts(&[vec![4]], 4).is_err());
        assert!(Varlen::from_cu_seqlens(&[1, 3]).is_err());
        assert!(Varlen::from_cu_seqlens(&[]).is_err());
    }
}
