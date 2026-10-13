//! Rounding to the reduced precisions, bit for bit as the GPU rounds: bf16
//! and f16 to nearest with ties to even (the device's `__float2bfloat16_rn`
//! and `__float2half_rn`, the host's `half::*::from_f32` that uploads the
//! GPU's weights), and TF32 operands to nearest with ties away from zero
//! (`cvt.rna.tf32.f32`). Values stay f32: a rounded value is the f32 the
//! narrow value widens to, exactly, so the host's f32 kernels run on it
//! unchanged. A NaN stays a NaN.

use crate::precision::WeightDtype;

/// `x` rounded to bf16 and widened back, ties to even. Past the largest
/// bf16 it rounds to infinity, as the device does.
#[inline(always)]
pub(crate) fn round_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    if x.is_nan() {
        return f32::from_bits((b | 0x0040_0000) & 0xFFFF_0000);
    }
    f32::from_bits(b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
}

/// `x` rounded to TF32 (ten mantissa bits) and widened back, ties away
/// from zero. Subnormals round on their own grid the same way.
#[inline(always)]
pub(crate) fn round_tf32(x: f32) -> f32 {
    let b = x.to_bits();
    if x.is_nan() {
        return f32::from_bits((b | 0x0040_0000) & 0xFFFF_E000);
    }
    f32::from_bits(b.wrapping_add(0x1000) & 0xFFFF_E000)
}

/// The f16 bits of `x`, ties to even: infinity past the largest f16, the
/// subnormals on their 2^-24 grid, zero below half of it.
pub(crate) fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let exp = (b >> 23) & 0xFF;
    let man = b & 0x007F_FFFF;
    if exp == 0xFF {
        let nan = if man == 0 { 0 } else { 0x0200 };
        return (sign | 0x7C00 | nan | (man >> 13)) as u16;
    }
    let e = exp as i32 - 112;
    if e >= 0x1F {
        return (sign | 0x7C00) as u16;
    }
    if e <= 0 {
        if e < -10 {
            return sign as u16;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let round = 1u32 << (shift - 1);
        let up = u32::from(m & round != 0 && m & (3 * round - 1) != 0);
        return (sign | ((m >> shift) + up)) as u16;
    }
    let up = u32::from(man & 0x1000 != 0 && man & 0x2FFF != 0);
    ((sign | ((e as u32) << 10) | (man >> 13)) + up) as u16
}

/// The f32 an f16 widens to, exactly; a NaN keeps its payload, quieted.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let h = u32::from(h);
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let man = h & 0x03FF;
    if exp == 0x1F {
        let nan = if man == 0 {
            0
        } else {
            0x0040_0000 | (man << 13)
        };
        return f32::from_bits(sign | 0x7F80_0000 | nan);
    }
    if exp == 0 {
        // man * 2^-24, exact in f32.
        let v = man as f32 * f32::from_bits(0x3380_0000);
        return if sign == 0 { v } else { -v };
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
}

/// `x` rounded to f16 and widened back, ties to even.
#[inline(always)]
pub(crate) fn round_f16(x: f32) -> f32 {
    f16_to_f32(f16_bits(x))
}

/// `x` as `dtype` stores it between kernels: a bf16 or f16 value for the
/// half precisions, `x` itself for f32 and TF32.
#[inline(always)]
pub(crate) fn stored(dtype: WeightDtype, x: f32) -> f32 {
    match dtype {
        WeightDtype::Bf16 => round_bf16(x),
        WeightDtype::F16 => round_f16(x),
        WeightDtype::F32 | WeightDtype::Tf32 => x,
    }
}

/// Rounds `v` in place to what `dtype` stores between kernels: bf16 and
/// f16 values for the half precisions; f32 and TF32 store f32.
pub(crate) fn store(dtype: WeightDtype, v: &mut [f32]) {
    match dtype {
        WeightDtype::Bf16 => v.iter_mut().for_each(|x| *x = round_bf16(*x)),
        WeightDtype::F16 => v.iter_mut().for_each(|x| *x = round_f16(*x)),
        WeightDtype::F32 | WeightDtype::Tf32 => {}
    }
}

/// Rounds `v` in place to what a GEMM of `dtype` multiplies: TF32 operands
/// for `Tf32`, nothing for the others, whose operands are what they store.
pub(crate) fn operand(dtype: WeightDtype, v: &mut [f32]) {
    if dtype == WeightDtype::Tf32 {
        v.iter_mut().for_each(|x| *x = round_tf32(*x));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every 2^32 / step-th bit pattern, the specials and their neighbours.
    fn patterns(step: u32) -> impl Iterator<Item = u32> {
        let specials = [
            0u32,
            0x8000_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
            0x7F80_0001,
            0xFFFF_FFFF,
            0x7F7F_FFFF,
            0x0000_0001,
            0x0080_0000,
            0x007F_FFFF,
            0x477F_E000,
            0x477F_F000,
            0x477F_EFFF,
            0x3880_0000,
            0x3380_0000,
            0x3300_0000,
            0x3300_0001,
            0x3F80_1000,
            0x3F80_3000,
            0x3F80_8000,
            0x3F81_8000,
        ];
        (0..=u32::MAX / step)
            .map(move |i| i.wrapping_mul(step))
            .chain(specials)
    }

    fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn bf16_matches_the_half_crate() {
        for b in patterns(4099) {
            let x = f32::from_bits(b);
            let want = half::bf16::from_f32(x).to_f32();
            assert!(same(round_bf16(x), want), "{b:#010x}");
        }
    }

    #[test]
    fn f16_matches_the_half_crate() {
        for b in patterns(4099) {
            let x = f32::from_bits(b);
            let h = half::f16::from_f32(x);
            if !x.is_nan() {
                assert_eq!(f16_bits(x), h.to_bits(), "{b:#010x}");
            }
            assert!(same(round_f16(x), h.to_f32()), "{b:#010x}");
        }
        for h in 0..=u16::MAX {
            let want = half::f16::from_bits(h).to_f32();
            assert!(same(f16_to_f32(h), want), "{h:#06x}");
        }
    }

    /// Ties go away from zero, as `cvt.rna.tf32.f32` rounds; the values the
    /// device's TF32 golden pins.
    #[test]
    fn tf32_rounds_ties_away_from_zero() {
        let r = |b: u32| round_tf32(f32::from_bits(b)).to_bits();
        assert_eq!(r(0x3F80_1000), 0x3F80_2000);
        assert_eq!(r(0xBF80_1000), 0xBF80_2000);
        assert_eq!(r(0x3F80_0FFF), 0x3F80_0000);
        assert_eq!(r(0x3F80_3000), 0x3F80_4000);
        assert_eq!(r(0x7F7F_FFFF), 0x7F80_0000);
        assert_eq!(r(0x7F80_0000), 0x7F80_0000);
        assert!(round_tf32(f32::NAN).is_nan());
        for b in patterns(65521) {
            let x = f32::from_bits(b);
            if x.is_nan() {
                continue;
            }
            let got = round_tf32(x);
            assert_eq!(got.to_bits() & 0x1FFF, 0, "{b:#010x}");
            // Within half of the TF32 spacing of x.
            let ulp = f32::from_bits((b & 0x7F80_0000).max(0x0080_0000)) * 2f32.powi(-10);
            assert!(
                (got - x).abs() <= ulp / 2.0 || got.is_infinite(),
                "{b:#010x}"
            );
        }
    }

    #[test]
    fn slices_round_only_in_their_lane() {
        let v = [1.000_001_f32, -3.257_812_5, 65_519.0, 1e-7];
        for dtype in [WeightDtype::F32, WeightDtype::Tf32] {
            let mut s = v;
            store(dtype, &mut s);
            assert_eq!(s, v);
        }
        for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            let mut s = v;
            operand(dtype, &mut s);
            assert_eq!(s, v);
        }
        let mut s = v;
        store(WeightDtype::Bf16, &mut s);
        assert_eq!(s, v.map(round_bf16));
        for dtype in [
            WeightDtype::F32,
            WeightDtype::Tf32,
            WeightDtype::Bf16,
            WeightDtype::F16,
        ] {
            let mut s = v;
            store(dtype, &mut s);
            assert_eq!(s, v.map(|x| stored(dtype, x)));
        }
        let mut s = v;
        store(WeightDtype::F16, &mut s);
        assert_eq!(s, v.map(round_f16));
        let mut s = v;
        operand(WeightDtype::Tf32, &mut s);
        assert_eq!(s, v.map(round_tf32));
    }
}
