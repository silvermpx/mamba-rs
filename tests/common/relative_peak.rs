#[path = "finite_pairs.rs"]
pub(crate) mod tensor;

/// Largest absolute error divided by the reference's peak magnitude or its tolerance floor.
pub fn relative_peak(got: &[f32], want: &[f64], floor: f64) -> f64 {
    let mut difference = 0.0_f64;
    let mut scale = floor;
    for (g, w) in tensor::pairs("relative peak error", got, want) {
        difference = difference.max((f64::from(g) - w).abs());
        scale = scale.max(w.abs());
    }
    if difference == 0.0 {
        0.0
    } else {
        difference / scale
    }
}
