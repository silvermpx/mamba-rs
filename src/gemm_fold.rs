//! How a deterministic GEMM output is summed: where its reduction is cut
//! into pieces, how each piece chains its products, and how the pieces, the
//! bias and the sign of zero combine. The rules read only the shape of the
//! layer, never the number of rows, the card or the route, so a row writes
//! the same bits whatever batch it is in. Every kernel that serves an
//! output computes one of these folds; this module states them once and
//! evaluates them on the CPU, so a kernel's words can be checked against
//! the fold it claims.
//!
//! A deep or narrow layer is cut into pieces of 128 inputs, or of 64 when it
//! has fewer than 512 inputs (a narrow layer cut from 384). The depth
//! threshold, the sign of an exact zero and the place of the bias are open;
//! [`FoldRule`] carries each choice, the other cuts that were measured
//! included, so that every side of a choice can be measured.

use std::ops::Range;

/// The revision of the fold rules below. A change that moves any output
/// word bumps it.
pub const GEMM_CONTRACT_REVISION: u32 = 2;

/// The granule of the four-piece cut: piece bounds fall on whole groups of
/// this many inputs.
pub const PIECE_GRANULE: usize = 32;

/// The length of a piece under the batch-invariant forward's cut, for a
/// layer of [`CHUNK_DEEP_FROM`] inputs or more.
pub const CHUNK_INPUTS: usize = 128;

/// The length of a piece under the batch-invariant forward's cut for a
/// shallower layer, which only a narrow layer's cut reaches.
pub const CHUNK_INPUTS_SHALLOW: usize = 64;

/// The depth from which the batch-invariant forward's pieces are
/// [`CHUNK_INPUTS`] long.
pub const CHUNK_DEEP_FROM: usize = 512;

/// How a deep or narrow layer cuts its reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeepCut {
    /// Four pieces of whole 32-input groups, the last one shorter.
    FourPieces,
    /// Pieces of this many inputs, a whole number of 32-input groups, the
    /// last one ragged.
    Chunks(usize),
    /// Pieces of `shallow` inputs below `from` inputs and of `deep` inputs
    /// from there, each a whole number of 32-input groups, the last ragged.
    ChunksByDepth {
        shallow: usize,
        deep: usize,
        from: usize,
    },
}

/// From what depth a layer is cut at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CutThreshold {
    /// From 512 inputs, or from 384 when the layer has fewer than 128
    /// outputs and a chain tile has too few columns to fill the card.
    Deep512Narrow384,
    /// From 256 inputs, whatever the width.
    Deep256,
}

/// The sign an exactly-zero output is written with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZeroSign {
    /// Every exactly-zero output is +0.0.
    Positive,
    /// A zero keeps the sign its chains leave, except where a kernel's
    /// zero padding adds +0.0: after a one-piece chain whose reduction is
    /// not a whole number of 16-input tiles, and after the last of several
    /// pieces when its length is not a whole number of 32-input tiles.
    Grain,
}

/// Where the bias enters the sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BiasPlace {
    /// A one-piece chain starts from the bias; several pieces add it after
    /// the fold.
    SeedOnePiece,
    /// The bias is added after the fold for every piece count.
    AfterSum,
}

/// One complete choice of fold for exact f32 NN and NT outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FoldRule {
    pub cut: DeepCut,
    pub threshold: CutThreshold,
    pub zero: ZeroSign,
    pub bias: BiasPlace,
}

impl FoldRule {
    /// The proposed rule: four pieces from 512 inputs (384 for a narrow
    /// layer), every zero positive, the bias seeding a one-piece chain.
    pub const PROPOSED: Self = Self {
        cut: DeepCut::FourPieces,
        threshold: CutThreshold::Deep512Narrow384,
        zero: ZeroSign::Positive,
        bias: BiasPlace::SeedOnePiece,
    };

    /// 64-input chunks from the same depths, zeros as the kernels' padding
    /// leaves them.
    pub const CHUNKS_64: Self = Self {
        cut: DeepCut::Chunks(64),
        threshold: CutThreshold::Deep512Narrow384,
        zero: ZeroSign::Grain,
        bias: BiasPlace::SeedOnePiece,
    };

    /// 128-input pieces from the same depths, zeros as the kernels' padding
    /// leaves them.
    pub const CHUNKS_128: Self = Self {
        cut: DeepCut::Chunks(128),
        threshold: CutThreshold::Deep512Narrow384,
        zero: ZeroSign::Grain,
        bias: BiasPlace::SeedOnePiece,
    };

    /// 128-input pieces from 512 inputs and 64-input pieces for the narrow
    /// layers cut below, zeros as the kernels' padding leaves them.
    pub const CHUNKS_128_SHALLOW_64: Self = Self {
        cut: DeepCut::ChunksByDepth {
            shallow: CHUNK_INPUTS_SHALLOW,
            deep: CHUNK_INPUTS,
            from: CHUNK_DEEP_FROM,
        },
        threshold: CutThreshold::Deep512Narrow384,
        zero: ZeroSign::Grain,
        bias: BiasPlace::SeedOnePiece,
    };

    /// Four pieces from the same depths, zeros as the kernels' padding
    /// leaves them: the proposed cut on the kernels as they are.
    pub const FOUR_PIECES: Self = Self {
        cut: DeepCut::FourPieces,
        threshold: CutThreshold::Deep512Narrow384,
        zero: ZeroSign::Grain,
        bias: BiasPlace::SeedOnePiece,
    };
}

/// The fold every exact f32 forward follows. Over the 779 cut cells of the
/// census on an RTX 6000 Ada, with the in-CTA tiles keeping their sum in
/// registers and reading the piece length at launch and each cell at the
/// fastest same-bit route of each cut, 128-input pieces ran in 0.938 of
/// four pieces' time in geometric mean (faster by more than 3% in 397
/// cells, slower in 151), 64-input pieces in 0.967 and 256-input pieces in
/// 0.982; on an RTX 5090, without the register tiles, 128-input pieces ran
/// in 0.968. Among rules on the layer's shape alone, scored on 793 cut
/// cells against the fastest length per cell, 64-input pieces below 512
/// inputs and 128-input pieces from there came first (1.0506 of the best
/// in geometric mean, 128-input pieces everywhere 1.0510, 64 1.079, 256
/// 1.099, four pieces 1.109); what is left depends on the row count, which
/// no cut may read.
pub const BATCH_INVARIANT_FORWARD: FoldRule = FoldRule::CHUNKS_128_SHALLOW_64;

/// The piece length that cuts `r` inputs into at most `p` pieces on a
/// granule of `g`: `g * ceil(ceil(r / g) / p)`. Zero for an empty reduction.
pub fn piece_len(r: usize, p: usize, g: usize) -> usize {
    debug_assert!(p > 0 && g > 0);
    g * r.div_ceil(g).div_ceil(p.max(1))
}

/// The pieces of `r` inputs cut every `len`: `[0, len)`, `[len, 2 len)`,
/// ..., the last ending at `r`. No piece is empty, and an empty reduction
/// has none.
pub fn pieces(r: usize, len: usize) -> impl Iterator<Item = Range<usize>> {
    let len = len.max(1);
    (0..r.div_ceil(len)).map(move |q| q * len..((q + 1) * len).min(r))
}

/// Whether an NN or NT layer of `r` inputs to its outputs and `w` outputs
/// per row is cut under `threshold`.
pub fn is_cut(r: usize, w: usize, threshold: CutThreshold) -> bool {
    match threshold {
        CutThreshold::Deep512Narrow384 => r >= 512 || (w < 128 && r >= 384),
        CutThreshold::Deep256 => r >= 256,
    }
}

/// The piece length of an NN or NT output under `rule`, read from the
/// layer alone: the whole reduction when the layer is not cut.
pub fn exact_piece_len(r: usize, w: usize, rule: FoldRule) -> usize {
    if r == 0 || !is_cut(r, w, rule.threshold) {
        return r;
    }
    match rule.cut {
        DeepCut::FourPieces => piece_len(r, 4, PIECE_GRANULE),
        DeepCut::Chunks(len) => len,
        DeepCut::ChunksByDepth {
            shallow,
            deep,
            from,
        } => {
            if r < from {
                shallow
            } else {
                deep
            }
        }
    }
}

/// `x + 0.0`: -0.0 becomes +0.0, every other value stays.
#[inline]
pub fn close(x: f32) -> f32 {
    x + 0.0
}

/// One exact f32 NN or NT output under `rule`: the products `a(r) * b(r)`
/// taken in increasing `r` over a reduction of `r_len`, for a layer `width`
/// outputs wide. NN: `a(r) = x[m][r]`, `b(r) = w[r][j]`, width N. NT:
/// `a(r) = dy[m][r]`, `b(r) = w[i][r]`, width K, no bias.
pub fn exact_output(
    a: impl Fn(usize) -> f32,
    b: impl Fn(usize) -> f32,
    bias: Option<f32>,
    r_len: usize,
    width: usize,
    rule: FoldRule,
) -> f32 {
    let written = |s: f32| match rule.zero {
        ZeroSign::Positive => close(s),
        ZeroSign::Grain => s,
    };
    if r_len == 0 {
        return close(0.0 + bias.unwrap_or(0.0));
    }
    let len = exact_piece_len(r_len, width, rule);
    let count = r_len.div_ceil(len);
    let seed = match (count, rule.bias, bias) {
        (1, BiasPlace::SeedOnePiece, Some(bias)) => bias,
        _ => 0.0,
    };
    let mut sum = 0.0f32;
    for (q, piece) in pieces(r_len, len).enumerate() {
        let piece_len = piece.len();
        let mut acc = seed;
        for r in piece {
            acc = a(r).mul_add(b(r), acc);
        }
        if rule.zero == ZeroSign::Grain {
            let pads = if count == 1 {
                !r_len.is_multiple_of(16)
            } else {
                q + 1 == count && !piece_len.is_multiple_of(32)
            };
            if pads {
                acc += 0.0;
            }
        }
        sum = if q == 0 { acc } else { sum + acc };
    }
    if let Some(bias) = bias
        && (count > 1 || rule.bias == BiasPlace::AfterSum)
    {
        sum += bias;
    }
    written(sum)
}

/// The cut of an exact f32 TN weight gradient of `m` rows into a `k` x `n`
/// output, from the shape alone: the piece length over the rows, `m` for
/// one chain. `target` is the number of work units the cut aims for (a
/// shape constant, never a card's multiprocessor count).
pub fn exact_tn_piece_len(m: usize, k: usize, n: usize, target: usize) -> usize {
    if m == 0 || n < 2 {
        return m;
    }
    let units = if n >= 128 {
        k.div_ceil(128) * n.div_ceil(128)
    } else {
        k.div_ceil(64) * n.div_ceil(32)
    };
    if m < 256 || units >= target {
        return m;
    }
    let p = target.div_ceil(units.max(1)).min(m / 32);
    if p < 2 {
        return m;
    }
    piece_len(m, p, 16)
}

/// The cut of a TF32 TN weight gradient: sixteen pieces for a deep
/// gradient into a few output tiles, eight for a deep or mid-depth one into
/// a few more, one chain otherwise. The pieces fall on 32-row groups.
pub fn tf32_tn_piece_len(m: usize, k: usize, n: usize) -> usize {
    let tiles = k.div_ceil(64) * n.div_ceil(64);
    let p = if m >= 1024 && tiles <= 8 {
        16
    } else if m >= 2048 || (m >= 1024 && tiles <= 16) {
        8
    } else {
        1
    };
    if p == 1 { m } else { piece_len(m, p, 32) }
}

/// One exact f32 TN output summed over `pieces_len`-row pieces, each a
/// chain from +0.0 in increasing row order, the pieces added in order,
/// then added to the old value for an accumulating call; the written zero
/// is positive.
pub fn exact_tn_output(
    a: impl Fn(usize) -> f32,
    b: impl Fn(usize) -> f32,
    old: Option<f32>,
    m: usize,
    pieces_len: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for (q, piece) in pieces(m, pieces_len).enumerate() {
        let mut acc = 0.0f32;
        for r in piece {
            acc = a(r).mul_add(b(r), acc);
        }
        sum = if q == 0 { acc } else { sum + acc };
    }
    close(match old {
        Some(old) => old + sum,
        None => sum,
    })
}

/// A TN output with a single column: 32 lanes chain the rows `l, l + 32,
/// ...` from +0.0, then each lane adds its partner's value for partner
/// distances 16, 8, 4, 2 and 1, and lane 0 holds the sum.
pub fn tn_column_output(a: impl Fn(usize) -> f32, b: impl Fn(usize) -> f32, m: usize) -> f32 {
    let mut lanes = [0.0f32; 32];
    for (lane, value) in lanes.iter_mut().enumerate() {
        let mut r = lane;
        while r < m {
            *value = a(r).mul_add(b(r), *value);
            r += 32;
        }
    }
    for distance in [16, 8, 4, 2, 1] {
        let before = lanes;
        for (lane, value) in lanes.iter_mut().enumerate() {
            *value = before[lane] + before[lane ^ distance];
        }
    }
    lanes[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u32) -> impl FnMut() -> f32 {
        let mut state = seed | 1;
        move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        }
    }

    #[test]
    fn pieces_cover_the_reduction_once_in_order() {
        for r in 0..2_100usize {
            for p in [1usize, 2, 3, 4, 8, 16] {
                for g in [16usize, 32, 64] {
                    let len = piece_len(r, p, g);
                    let cut: Vec<_> = pieces(r, len).collect();
                    let mut next = 0;
                    for piece in &cut {
                        assert_eq!(piece.start, next, "r={r} p={p} g={g}");
                        assert!(!piece.is_empty(), "r={r} p={p} g={g}");
                        next = piece.end;
                    }
                    assert_eq!(next, r, "r={r} p={p} g={g}");
                    assert!(cut.len() <= p, "r={r} p={p} g={g}: {}", cut.len());
                    if r > 0 {
                        assert!(len.is_multiple_of(g));
                    }
                }
            }
        }
    }

    #[test]
    fn a_one_piece_fold_is_the_plain_chain() {
        let mut next = lcg(7);
        let a: Vec<f32> = (0..300).map(|_| next()).collect();
        let b: Vec<f32> = (0..300).map(|_| next()).collect();
        for r_len in [1usize, 15, 16, 17, 255, 300] {
            let bias = 0.375f32;
            let chain = a
                .iter()
                .zip(&b)
                .take(r_len)
                .fold(bias, |acc, (x, y)| x.mul_add(*y, acc));
            // 300 outputs wide keeps every depth here below the cut.
            let got = exact_output(
                |r| a[r],
                |r| b[r],
                Some(bias),
                r_len,
                300,
                FoldRule::PROPOSED,
            );
            assert_eq!(got.to_bits(), close(chain).to_bits(), "r_len={r_len}");
        }
    }

    #[test]
    fn the_proposed_cut_is_four_pieces_of_whole_groups() {
        assert_eq!(exact_piece_len(511, 128, FoldRule::PROPOSED), 511);
        assert_eq!(exact_piece_len(512, 128, FoldRule::PROPOSED), 128);
        assert_eq!(exact_piece_len(384, 127, FoldRule::PROPOSED), 96);
        assert_eq!(exact_piece_len(383, 127, FoldRule::PROPOSED), 383);
        assert_eq!(exact_piece_len(582, 64, FoldRule::PROPOSED), 160);
        assert_eq!(pieces(582, 160).count(), 4);
        assert_eq!(exact_piece_len(3000, 101, FoldRule::CHUNKS_64), 64);
        assert_eq!(pieces(3000, 64).count(), 47);
    }

    #[test]
    fn the_forward_cut_is_64_input_pieces_below_512_and_128_from_there() {
        let rule = BATCH_INVARIANT_FORWARD;
        assert_eq!(exact_piece_len(511, 128, rule), 511);
        assert_eq!(exact_piece_len(512, 128, rule), 128);
        assert_eq!(exact_piece_len(384, 127, rule), 64);
        assert_eq!(exact_piece_len(511, 101, rule), 64);
        assert_eq!(exact_piece_len(512, 101, rule), 128);
        assert_eq!(exact_piece_len(383, 127, rule), 383);
        assert_eq!(exact_piece_len(3000, 101, rule), 128);
        assert_eq!(pieces(3000, 128).count(), 24);
        assert_eq!(pieces(3000, 128).last(), Some(2944..3000));
    }

    #[test]
    fn closing_each_piece_writes_the_words_of_closing_once() {
        let mut next = lcg(11);
        let data: Vec<Vec<f32>> = vec![
            (0..2048).map(|_| next()).collect(),
            vec![0.0; 2048],
            vec![-0.0; 2048],
            (0..2048)
                .map(|i| if i % 2 == 0 { -1e-30 } else { 1e-30 })
                .collect(),
            (0..2048)
                .map(|i| if i % 3 == 0 { -0.0 } else { -1e-45 })
                .collect(),
        ];
        for a in &data {
            for b in &data {
                for r_len in [384usize, 512, 513, 1000, 2048] {
                    let len = exact_piece_len(r_len, 64, FoldRule::PROPOSED);
                    let mut each = 0.0f32;
                    let mut once = 0.0f32;
                    for (q, piece) in pieces(r_len, len).enumerate() {
                        let mut acc = 0.0f32;
                        for r in piece {
                            acc = a[r].mul_add(b[r], acc);
                        }
                        each = if q == 0 {
                            close(acc)
                        } else {
                            close(each + close(acc))
                        };
                        once = if q == 0 { acc } else { once + acc };
                    }
                    assert_eq!(close(each).to_bits(), close(once).to_bits());
                    let got = exact_output(|r| a[r], |r| b[r], None, r_len, 64, FoldRule::PROPOSED);
                    assert_eq!(got.to_bits(), close(once).to_bits());
                }
            }
        }
    }

    #[test]
    fn an_empty_reduction_writes_the_bias_or_positive_zero() {
        for rule in [
            FoldRule::PROPOSED,
            FoldRule::CHUNKS_64,
            FoldRule::CHUNKS_128,
            FoldRule::CHUNKS_128_SHALLOW_64,
        ] {
            assert_eq!(
                exact_output(|_| 1.0, |_| 1.0, None, 0, 8, rule).to_bits(),
                0
            );
            assert_eq!(
                exact_output(|_| 1.0, |_| 1.0, Some(-0.0), 0, 8, rule).to_bits(),
                0
            );
            assert_eq!(exact_output(|_| 1.0, |_| 1.0, Some(2.5), 0, 8, rule), 2.5);
        }
    }

    #[test]
    fn the_weight_gradient_cut_reads_the_shape_alone() {
        // Few rows or enough output tiles: one chain.
        assert_eq!(exact_tn_piece_len(255, 64, 64, 512), 255);
        assert_eq!(exact_tn_piece_len(4096, 4096, 2048, 512), 4096);
        // 288 units of a 3072 x 1536 output ask for two pieces.
        assert_eq!(exact_tn_piece_len(4096, 3072, 1536, 512), 2048);
        // 1024 rows into a 256 x 256 output: 4 units, so 128 pieces are
        // asked and 32 allowed, of 32 rows each.
        assert_eq!(exact_tn_piece_len(1024, 256, 256, 512), 32);
        assert_eq!(pieces(1024, 32).count(), 32);
        assert_eq!(tf32_tn_piece_len(4096, 128, 256), 256);
        assert_eq!(tf32_tn_piece_len(2048, 1024, 1024), 256);
        assert_eq!(tf32_tn_piece_len(512, 1024, 1024), 512);
    }

    #[test]
    fn a_single_column_folds_its_lanes_by_halving_distances() {
        let mut next = lcg(13);
        let a: Vec<f32> = (0..1000).map(|_| next()).collect();
        let b: Vec<f32> = (0..1000).map(|_| next()).collect();
        let got = tn_column_output(|r| a[r], |r| b[r], 1000);
        let mut lanes = [0.0f32; 32];
        for (r, (x, y)) in a.iter().zip(&b).enumerate() {
            lanes[r % 32] = x.mul_add(*y, lanes[r % 32]);
        }
        let mut width = 32;
        while width > 1 {
            width /= 2;
            let before = lanes;
            for (lane, value) in lanes.iter_mut().enumerate() {
                *value = before[lane] + before[lane ^ width];
            }
        }
        assert_eq!(got.to_bits(), lanes[0].to_bits());
    }
}
