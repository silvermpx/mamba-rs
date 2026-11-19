#[path = "finite_pairs.rs"]
pub(crate) mod tensor;

#[track_caller]
pub fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    named_rel_l2("relative L2", got, want)
}

#[track_caller]
pub fn named_rel_l2(label: &str, got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, w) in tensor::pairs(label, got, want) {
        num += (f64::from(g) - f64::from(w)).powi(2);
        den += f64::from(w).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}
