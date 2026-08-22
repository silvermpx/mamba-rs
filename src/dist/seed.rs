//! The seed law: the derivations a data-parallel training harness must
//! draw ALL of its randomness through, so that no random decision ever
//! depends on the rank. The crate provides the law; honoring it is a
//! CALLER OBLIGATION — a harness that seeds anything outside these
//! derivations forfeits world-size invariance.
//!
//! - Weight init: derive from [`SeedLaw::init`] — rank-identical.
//! - Epoch order: a rank-identical global permutation from
//!   [`SeedLaw::epoch_order`]; a rank enters only as the strided slice
//!   `k % W == rank` taken AFTER the global order is fixed
//!   (`DistContext::shard`).
//! - Sample-level noise (augmentation): key on the global sample
//!   identity via [`SeedLaw::sample_noise`] — never on the rank — so
//!   the exact bytes seen at optimizer step k are world-size-invariant.
//!
//! Per-rank RNG streams are deliberately absent: they are the standard
//! way data-parallel training becomes irreproducible across world sizes.

/// Derives sub-seeds from one master seed. All derivations are pure
/// functions of their arguments; none takes a rank.
#[derive(Clone, Copy, Debug)]
pub struct SeedLaw {
    seed: u64,
}

/// One round of splitmix64 — the finalizer used for all derivations.
/// A bijective mix with full avalanche; the composed derivation is not
/// collision-free across tuples (no 64-bit hash is), it just makes
/// collisions no more likely than random.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Mix a domain tag and arguments into the master seed, guaranteeing a
/// nonzero result — xorshift-family generators collapse on state 0. The
/// clamp maps a zero mix to a fixed constant, which is one deliberate
/// extra collision in exchange for never seeding a dead generator.
fn derive(seed: u64, domain: u64, a: u64, b: u64) -> u64 {
    let mut z = splitmix64(seed ^ splitmix64(domain));
    z = splitmix64(z ^ splitmix64(a));
    z = splitmix64(z ^ splitmix64(b));
    if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z }
}

impl SeedLaw {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// The master seed this law derives from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Seed for weight/head initialization — identical on every rank.
    pub fn init(&self) -> u64 {
        derive(self.seed, 0x11, 0, 0)
    }

    /// Seed for the global epoch permutation — identical on every rank;
    /// the rank slices the permuted order afterwards, never reshuffles.
    pub fn epoch_order(&self, epoch: u64) -> u64 {
        derive(self.seed, 0x22, epoch, 0)
    }

    /// Seed for per-sample noise (augmentation), keyed on the GLOBAL
    /// sample identity so the same sample gets the same noise no matter
    /// which rank processes it or how many ranks exist.
    pub fn sample_noise(&self, epoch: u64, global_index: u64) -> u64 {
        derive(self.seed, 0x33, epoch, global_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivations_are_nonzero_and_domain_separated() {
        let law = SeedLaw::new(0);
        assert_ne!(law.init(), 0);
        assert_ne!(law.epoch_order(0), 0);
        assert_ne!(law.sample_noise(0, 0), 0);
        assert_ne!(law.init(), law.epoch_order(0));
        assert_ne!(law.epoch_order(0), law.epoch_order(1));
        assert_ne!(law.sample_noise(0, 1), law.sample_noise(1, 0));
    }

    #[test]
    fn same_inputs_same_outputs() {
        let a = SeedLaw::new(42);
        let b = SeedLaw::new(42);
        assert_eq!(a.init(), b.init());
        assert_eq!(a.epoch_order(7), b.epoch_order(7));
        assert_eq!(a.sample_noise(3, 999), b.sample_noise(3, 999));
    }
}
