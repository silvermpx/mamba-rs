//! The pieces every family's bidirectional mode shares: which layers read
//! their sequences reversed, and the reversal of every sequence's rows.
//!
//! A bidirectional model pairs its layers. Both layers of a pair read the
//! pair's input; the first reads every sequence forward, the second
//! reversed: its normalised rows are reversed before its mixer and the
//! mixer's output is reversed back, so the norm, the projections and the
//! gate see each row where it stands and only the convolution and the scan
//! run the other way. The pair's output is `(x + y_forward) + y_reversed`.

use crate::varlen::RowStarts;

/// The second layer of each pair of a bidirectional model: it reads every
/// sequence reversed.
#[inline(always)]
pub(crate) fn reads_reversed(bidirectional: bool, layer: usize) -> bool {
    bidirectional && !layer.is_multiple_of(2)
}

/// The first layer of each pair of a bidirectional model: the pair's
/// input is the residual stream before it.
#[inline(always)]
pub(crate) fn opens_pair(bidirectional: bool, layer: usize) -> bool {
    bidirectional && layer.is_multiple_of(2)
}

/// The layers whose gated MLPs run after layer `layer`'s mixer: its own,
/// or in a bidirectional model both of a pair's after the pair's second
/// mixer and none after its first.
#[inline(always)]
pub(crate) fn mlps_after(bidirectional: bool, layer: usize) -> std::ops::Range<usize> {
    match (bidirectional, layer.is_multiple_of(2)) {
        (false, _) => layer..layer + 1,
        (true, true) => layer..layer,
        (true, false) => layer - 1..layer + 1,
    }
}

/// How a layer of a bidirectional pair takes its gradient: d loss / d (the
/// pair's output rows), in the layer's own order (reversed for the pair's
/// second layer), and whether the layer reads its sequences reversed.
#[derive(Clone, Copy)]
pub(crate) struct PairGrad<'a> {
    pub d_out: &'a [f32],
    pub reversed: bool,
}

/// Reverse the rows of every sequence of one batch row in place: `rows`
/// holds the batch row's `width`-wide rows, and `starts` its packing (one
/// sequence when `None`). A sequence spanning positions `[first, end)`
/// trades row `t` for row `first + end - 1 - t`.
pub(crate) fn reverse_rows(rows: &mut [f32], width: usize, starts: Option<&RowStarts>) {
    if width == 0 {
        return;
    }
    let seq_len = rows.len() / width;
    let mut first = 0;
    while first < seq_len {
        let end = match starts {
            None => seq_len,
            Some(s) => (first + 1..seq_len)
                .find(|&t| s.starts(t))
                .unwrap_or(seq_len),
        };
        reverse_span(&mut rows[first * width..end * width], width);
        first = end;
    }
}

/// Reverse the `width`-wide rows of one span in place.
fn reverse_span(span: &mut [f32], width: usize) {
    let n = span.len() / width;
    let (head, tail) = span.split_at_mut((n / 2) * width);
    // The middle row of an odd span stays; the rest trade across it.
    let tail_rows = tail.len() / width;
    let tail = &mut tail[(tail_rows - n / 2) * width..];
    for (a, b) in head
        .chunks_exact_mut(width)
        .zip(tail.chunks_exact_mut(width).rev())
    {
        a.swap_with_slice(b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::varlen::Varlen;

    fn rows(n: usize, width: usize) -> Vec<f32> {
        (0..n * width).map(|i| i as f32).collect()
    }

    fn row_ids(v: &[f32], width: usize) -> Vec<usize> {
        v.chunks_exact(width)
            .map(|r| r[0] as usize / width)
            .collect()
    }

    #[test]
    fn one_sequence_reverses_whole() {
        for n in [0usize, 1, 2, 5, 6] {
            let mut v = rows(n, 3);
            reverse_rows(&mut v, 3, None);
            assert_eq!(row_ids(&v, 3), (0..n).rev().collect::<Vec<_>>(), "{n} rows");
            reverse_rows(&mut v, 3, None);
            assert_eq!(v, rows(n, 3), "{n} rows twice");
        }
    }

    #[test]
    fn packed_sequences_reverse_in_their_own_spans() {
        let packing = Varlen::from_starts(&[vec![0, 3, 4]], 7).unwrap();
        let starts = RowStarts::from_ids(packing.row(0));
        let mut v = rows(7, 2);
        reverse_rows(&mut v, 2, Some(&starts));
        assert_eq!(row_ids(&v, 2), vec![2, 1, 0, 3, 6, 5, 4]);
        // The carried sequence before the first start is a span too.
        let packing = Varlen::from_starts(&[vec![2]], 5).unwrap();
        let starts = RowStarts::from_ids(packing.row(0));
        let mut v = rows(5, 2);
        reverse_rows(&mut v, 2, Some(&starts));
        assert_eq!(row_ids(&v, 2), vec![1, 0, 4, 3, 2]);
    }
}
