//! The reduction contract, proven without any GPU or communicator: the
//! sharded owner fold (what a real transport-backed reducer executes)
//! must produce bit-for-bit the straight-line ascending-rank reference,
//! independent of world size, arena length, and delivery order.
//!
//! Scope note on the delivery-order oracles: immunity is provided BY
//! CONSTRUCTION through slot-keyed placement (contributions land at
//! slot = source rank; the fold consumes slots in fixed program-text
//! order). The permutation tests therefore pin the mechanism — a mutant
//! that placed contributions by ARRIVAL index would fail them — not a
//! hypothetical reordering of the fold itself, which no code path can
//! express.

use mamba_rs::dist::EmulatedWorld;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            // Mix magnitudes so the fold order actually matters: large
            // and small addends interleave and rounding is order-
            // sensitive almost everywhere.
            let base = (s & 0xFFFF) as f32 / 65536.0 - 0.5;
            if s & 1 == 0 { base * 1.0e6 } else { base }
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn sharded_fold_matches_reference_across_worlds_and_lengths() {
    for world in [1usize, 2, 3, 4, 7, 8] {
        for n in [0usize, 1, 5, 64, 1000, 4097] {
            let ew = EmulatedWorld::new(world).unwrap();
            let arenas: Vec<Vec<f32>> = (0..world)
                .map(|r| det(n, 0x1000 + (world * 100 + r) as u32))
                .collect();
            let reference = ew.reference_mean(&arenas).unwrap();
            let mut sharded = arenas.clone();
            ew.all_reduce_mean(&mut sharded, None).unwrap();
            for (r, arena) in sharded.iter().enumerate() {
                assert_eq!(
                    bits(arena),
                    bits(&reference),
                    "world={world} n={n} rank={r}: sharded fold diverged from the reference"
                );
            }
        }
    }
}

#[test]
fn delivery_order_cannot_change_the_bits() {
    // A transport delivers peer contributions in arbitrary order; the
    // fold must sort by logical rank, so every permutation of delivery
    // produces identical bits.
    let world = 4usize;
    let n = 513usize;
    let ew = EmulatedWorld::new(world).unwrap();
    let arenas: Vec<Vec<f32>> = (0..world).map(|r| det(n, 0x77 + r as u32)).collect();

    let mut canonical = arenas.clone();
    ew.all_reduce_mean(&mut canonical, None).unwrap();

    for order in [
        vec![3, 2, 1, 0],
        vec![1, 3, 0, 2],
        vec![2, 0, 3, 1],
        vec![0, 1, 2, 3],
    ] {
        let mut permuted = arenas.clone();
        ew.all_reduce_mean(&mut permuted, Some(&order)).unwrap();
        assert_eq!(
            bits(&permuted[0]),
            bits(&canonical[0]),
            "delivery order {order:?} changed the reduced bits"
        );
    }
}

#[test]
fn two_rank_mean_of_identical_arenas_reproduces_input_bits() {
    // x + x doubles the exponent bit-exactly and the *0.5 scale shifts
    // it back, so a two-rank mean of identical arenas must reproduce
    // the input bits. (Larger worlds pass through intermediate sums
    // like 3x that legitimately round — no such claim is made there;
    // the power-of-two-W exactness statement is about the final scale
    // step only, which is a pure exponent shift.)
    let n = 257usize;
    let ew = EmulatedWorld::new(2).unwrap();
    let a = det(n, 0xAB);
    let mut arenas = vec![a.clone(), a.clone()];
    ew.all_reduce_mean(&mut arenas, None).unwrap();
    for (i, (&got, &want)) in arenas[0].iter().zip(a.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "element {i}: two-rank mean of identical arenas must be exact"
        );
    }
}

#[test]
fn same_addends_different_fold_tree_changes_bits() {
    // The trainer-level W-route test necessarily changes the data
    // schedule together with W; this pin isolates the FOLD TREE: the
    // same four addends reduced flat at W=4 vs pairwise (two W=2 folds,
    // then a W=2 fold of the halves, each with its own 1/W scale — the
    // hierarchical shape a W=2 world reducing pre-averaged pairs would
    // produce) must differ bitwise on order-sensitive values.
    let n = 257usize;
    let a: Vec<Vec<f32>> = (0..4).map(|r| det(n, 0xD00 + r as u32)).collect();

    let ew4 = EmulatedWorld::new(4).unwrap();
    let flat = ew4.reference_mean(&a).unwrap();

    let ew2 = EmulatedWorld::new(2).unwrap();
    let left = ew2.reference_mean(&a[0..2]).unwrap();
    let right = ew2.reference_mean(&a[2..4]).unwrap();
    let nested = ew2.reference_mean(&[left, right]).unwrap();

    let flat_bits: Vec<u32> = flat.iter().map(|x| x.to_bits()).collect();
    let nested_bits: Vec<u32> = nested.iter().map(|x| x.to_bits()).collect();
    assert_ne!(
        flat_bits, nested_bits,
        "flat W=4 fold and nested W=2 folds unexpectedly agree — the \
         adversarial inputs must be order-sensitive for this pin to hold"
    );
}
