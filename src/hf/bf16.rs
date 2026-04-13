//! bf16 → f32 conversion utilities.

use half::bf16;
use half::slice::HalfFloatSliceExt;

/// Convert raw bf16 bytes to f32 vec.
///
/// `raw_bytes` must have length divisible by 2 (each bf16 is 2 bytes).
pub fn bf16_bytes_to_f32(raw_bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(format!(
            "bf16 buffer length {} is not even",
            raw_bytes.len()
        ));
    }
    let bf16_slice: &[bf16] = bytemuck::cast_slice(raw_bytes);
    let mut out = vec![0.0f32; bf16_slice.len()];
    bf16_slice.convert_to_f32_slice(&mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_zero_roundtrip() {
        let zero_bf16 = bf16::from_f32(0.0);
        let bytes = bytemuck::bytes_of(&zero_bf16);
        let result = bf16_bytes_to_f32(bytes).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_bf16_known_values() {
        let vals = [1.0f32, -1.0, 0.5, 3.14];
        let bf16_vals: Vec<bf16> = vals.iter().map(|&v| bf16::from_f32(v)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&bf16_vals);
        let result = bf16_bytes_to_f32(bytes).unwrap();
        assert_eq!(result.len(), 4);
        for (got, &expected) in result.iter().zip(&vals) {
            assert!(
                (got - expected).abs() < 0.02,
                "bf16 roundtrip: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn test_bf16_nan_inf() {
        let vals = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let bf16_vals: Vec<bf16> = vals.iter().map(|&v| bf16::from_f32(v)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&bf16_vals);
        let result = bf16_bytes_to_f32(bytes).unwrap();
        assert!(result[0].is_infinite() && result[0] > 0.0);
        assert!(result[1].is_infinite() && result[1] < 0.0);
        assert!(result[2].is_nan());
    }

    #[test]
    fn test_bf16_simd_matches_scalar() {
        let vals: Vec<f32> = (0..257).map(|i| i as f32 * 0.01 - 1.28).collect();
        let bf16_vals: Vec<bf16> = vals.iter().map(|&v| bf16::from_f32(v)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&bf16_vals);

        let simd_result = bf16_bytes_to_f32(bytes).unwrap();
        let scalar_result: Vec<f32> = bf16_vals.iter().map(|v| v.to_f32()).collect();

        assert_eq!(simd_result.len(), scalar_result.len());
        for (i, (s, r)) in simd_result.iter().zip(&scalar_result).enumerate() {
            assert_eq!(s, r, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_bf16_odd_length_errors() {
        let result = bf16_bytes_to_f32(&[0u8, 1, 2]);
        assert!(result.is_err());
    }
}
