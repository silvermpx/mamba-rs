//! The fold plan: how the flat gradient arena is partitioned into
//! per-rank shards and how the cross-rank sum is ordered.
//!
//! Determinism rests on two facts. IEEE-754 addition is deterministic
//! per operand pair; nondeterminism enters a reduction only through the
//! association order and the operand set. Both are fixed here as program
//! text: rank r owns one contiguous shard, receives the other ranks'
//! copies of THAT shard as pure byte movement, and sums the W addends
//! per element in strictly ascending logical-rank order. Transfer
//! chunking, arrival order, link topology, and library version cannot
//! appear in the arithmetic, so the reduced bits depend only on the
//! logical world size and the addend values.
//!
//! The mean is taken as sum-then-multiply by `1/W` AFTER the fold —
//! exact when W is a power of two. Pre-scaled averaging (each rank
//! multiplying its addend first) is banned: it couples rounding to the
//! world size at every element.

/// Contiguous shard of a length-`n` arena owned by one logical rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shard {
    pub start: usize,
    pub len: usize,
}

/// Partition `n` elements into `world` contiguous shards, remainder
/// spread one element each over the lowest ranks. Every element belongs
/// to exactly one shard; shards are ordered by rank; a shard can be
/// empty when `n < world`.
pub fn shard_plan(n: usize, world: usize) -> Vec<Shard> {
    assert!(world > 0, "world size must be positive");
    let base = n / world;
    let extra = n % world;
    let mut out = Vec::with_capacity(world);
    let mut start = 0usize;
    for r in 0..world {
        let len = base + usize::from(r < extra);
        out.push(Shard { start, len });
        start += len;
    }
    out
}

/// Reference reduction: fold `world` full-arena addends into `out`,
/// summing per element in strictly ascending logical-rank order, then
/// multiply by `1/world`. This host implementation IS the numeric
/// contract — every transport-backed reducer must produce these exact
/// bits (the GPU kernel mirrors the same per-element ascending fold).
pub fn reduce_mean_reference(addends: &[&[f32]], out: &mut [f32]) {
    let world = addends.len();
    assert!(world > 0, "need at least one addend");
    for a in addends {
        assert_eq!(a.len(), out.len(), "addend length mismatch");
    }
    let inv_w = 1.0f32 / world as f32;
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = addends[0][i];
        for a in &addends[1..] {
            acc += a[i];
        }
        *o = acc * inv_w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_cover_everything_once_in_rank_order() {
        for (n, w) in [(0, 1), (1, 4), (7, 3), (16, 4), (100, 8), (3, 8)] {
            let plan = shard_plan(n, w);
            assert_eq!(plan.len(), w);
            let mut cursor = 0usize;
            for s in &plan {
                assert_eq!(s.start, cursor, "shards must be contiguous in rank order");
                cursor += s.len;
            }
            assert_eq!(cursor, n, "shards must cover the arena exactly");
        }
    }

    #[test]
    fn remainder_goes_to_lowest_ranks() {
        let plan = shard_plan(10, 4);
        let lens: Vec<usize> = plan.iter().map(|s| s.len).collect();
        assert_eq!(lens, vec![3, 3, 2, 2]);
    }

    #[test]
    fn reference_fold_is_ascending_rank_order() {
        // Floating-point addition is not associative, so a fold in a
        // different order produces different bits on adversarial values.
        // Pin the ascending order by comparing against a hand-rolled
        // left fold and by showing a reversed fold differs.
        // (1e8 + -1e8) + 1 = 1, while (1 + -1e8) + 1e8 = 0: the small
        // addend survives only in ascending order here.
        let a = vec![1.0e8f32, 2.5];
        let b = vec![-1.0e8f32, 0.5];
        let c = vec![1.0f32, -1.0];
        let mut out = vec![0.0f32; 2];
        reduce_mean_reference(&[&a, &b, &c], &mut out);
        let manual0 = ((a[0] + b[0]) + c[0]) * (1.0 / 3.0);
        let manual1 = ((a[1] + b[1]) + c[1]) * (1.0 / 3.0);
        assert_eq!(out[0].to_bits(), manual0.to_bits());
        assert_eq!(out[1].to_bits(), manual1.to_bits());
        let reversed0 = ((c[0] + b[0]) + a[0]) * (1.0 / 3.0);
        assert_ne!(
            out[0].to_bits(),
            reversed0.to_bits(),
            "the test values must be order-sensitive, or this pin proves nothing"
        );
    }
}
