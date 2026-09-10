//! The one digest law: FNV-1a over raw little-endian bytes.
//!
//! Every printer and every bit gate in the tree hashes through this
//! module, because two hash laws under one output prefix produce
//! incomparable numbers: a value recorded by a word-wise variant can
//! never be checked against a byte-wise one, and the mismatch looks
//! exactly like a real regression. Streaming absorption with label
//! separation replaces XOR-folding of per-part digests - XOR is linear,
//! so correlated changes can cancel; sequential absorption makes
//! cancellation impossible by construction and keeps the order of parts
//! part of the value.
//!
//! FNV-1a/64 is enough here: the adversary is an accidental code
//! change, not a forger.

pub struct Digest(u64);

impl Digest {
    pub fn new() -> Self {
        Digest(0xcbf29ce484222325)
    }

    pub fn absorb_bytes(&mut self, bytes: &[u8]) -> &mut Self {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
        self
    }

    /// Domain separation for multi-part digests: absorb a label before
    /// each part so identical parts under different roles cannot alias.
    pub fn absorb_label(&mut self, label: &str) -> &mut Self {
        self.absorb_bytes(label.as_bytes())
    }

    pub fn absorb_f32(&mut self, values: &[f32]) -> &mut Self {
        for v in values {
            self.absorb_bytes(&v.to_bits().to_le_bytes());
        }
        self
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}

impl Default for Digest {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot form of the law for an f32 slice.
pub fn fnv1a_f32(values: &[f32]) -> u64 {
    let mut d = Digest::new();
    d.absorb_f32(values);
    d.finish()
}
