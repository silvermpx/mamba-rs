//! Finite storage-bit corpus for Fixed F32/TF32 qualification.
//! Kept independent of CUDA so its coverage can be checked with native rustc.

/// Retains the test-only W4 discovery generator's exact bits. This finite
/// corpus supplements, and does not replace, separate NaN/Inf payload cases.
pub fn finite_full_mantissa_values(len: usize, mut state: u64) -> Vec<f32> {
    const SPECIAL: [u32; 10] = [
        0x0000_0000,
        0x8000_0000,
        0x3f80_1000,
        0xbf80_1000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x3f80_0fff,
        0x3f80_1001,
    ];
    (0..len)
        .map(|index| {
            if index < SPECIAL.len() {
                return f32::from_bits(SPECIAL[index]);
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sign = (state as u32) & 0x8000_0000;
            let exponent = (((state >> 32) as u32 % 48) + 103) << 23;
            let mantissa = (state as u32) & 0x007f_ffff;
            f32::from_bits(sign | exponent | mantissa)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::finite_full_mantissa_values;

    fn bits(len: usize, seed: u64) -> Vec<u32> {
        finite_full_mantissa_values(len, seed)
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    #[test]
    fn edge_prefix_preserves_signed_zero_subnormals_and_tf32_ties() {
        assert!(bits(0, 1).is_empty());
        let expected = [
            0x0000_0000,
            0x8000_0000,
            0x3f80_1000,
            0xbf80_1000,
            0x0000_0001,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x3f80_0fff,
            0x3f80_1001,
        ];
        for len in 1..=expected.len() {
            assert_eq!(bits(len, 0x89a0_0001), expected[..len]);
        }
    }

    #[test]
    fn finite_random_tail_exercises_all_low_tf32_rounding_bits() {
        let values = bits(4096, 0x89a0_0001);
        assert!(
            values
                .iter()
                .all(|&value| f32::from_bits(value).is_finite())
        );
        let random = &values[10..];
        assert!(random.iter().any(|&value| value >> 31 == 0));
        assert!(random.iter().any(|&value| value >> 31 == 1));
        assert_eq!(
            random
                .iter()
                .fold(0, |mask, &value| mask | (value & 0x1fff)),
            0x1fff
        );
        assert!(
            random
                .iter()
                .all(|&value| (103..=150).contains(&((value >> 23) & 0xff)))
        );
    }

    #[test]
    fn corpus_is_repeatable_prefix_stable_and_seed_sensitive() {
        let full = bits(4096, 0x89a0_0001);
        assert!(full == bits(4096, 0x89a0_0001), "repeat changed bits");
        assert!(
            bits(2039, 0x89a0_0001) == full[..2039],
            "prefix changed bits"
        );
        assert!(full[10..] != bits(4096, 0x89b0_0001)[10..], "seed ignored");
    }

    #[test]
    fn seeded_tail_retains_frozen_w4_generator_bits() {
        assert_eq!(
            bits(18, 0x89a0_0001)[10..],
            [
                0x33b1_6041,
                0xb9a3_32c1,
                0x3774_e4e4,
                0xb57c_b42d,
                0xc0e7_e605,
                0x3e31_0889,
                0x4b2d_68d8,
                0x3c6d_0409
            ],
        );
    }
}
