//! The distance used by the GPU parity suites, exercised on the host.

#[path = "common/rel_l2.rs"]
mod distance;
#[path = "common/gradient_distance.rs"]
mod gradients;
use distance::rel_l2;

#[test]
fn relative_distance_refuses_a_nonfinite_entry_on_either_side() {
    let finite = [1.0, 2.0, 3.0];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let corrupt = [1.0, bad, 3.0];
        for (got, want) in [(&corrupt, &finite), (&finite, &corrupt)] {
            assert!(std::panic::catch_unwind(|| rel_l2(got, want)).is_err());
        }
    }
}

#[test]
fn relative_distance_refuses_different_tensor_lengths() {
    let full = [3.0, 4.0];
    for (got, want) in [(&full[..], &full[..1]), (&full[..0], &full[..])] {
        assert!(std::panic::catch_unwind(|| rel_l2(got, want)).is_err());
    }
}

#[test]
fn relative_distance_is_normalized_by_the_reference_tensor() {
    assert_eq!(rel_l2(&[6.0, 8.0], &[3.0, 4.0]), 1.0);
    assert_eq!(rel_l2(&[], &[]), 0.0);
}

#[test]
fn gradient_distance_checks_even_absent_parameter_tensors() {
    let weights = || vec![("weight".into(), &[1.0][..]), ("bias".into(), &[][..])];
    let mut extra = weights();
    extra[1].1 = &[3.0];
    assert!(std::panic::catch_unwind(|| gradients::worst_gradient(extra, weights())).is_err());
    let mut missing = weights();
    missing.pop();
    assert!(std::panic::catch_unwind(|| gradients::worst_gradient(missing, weights())).is_err());
}
