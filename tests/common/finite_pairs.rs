/// Every element must have a finite counterpart before it contributes to a comparison.
#[track_caller]
pub fn pairs<'a, A: Copy + Into<f64>, B: Copy + Into<f64>>(
    label: &'a str,
    actual: &'a [A],
    expected: &'a [B],
) -> impl Iterator<Item = (A, B)> + 'a {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: tensor lengths differ"
    );
    actual
        .iter()
        .copied()
        .zip(expected.iter().copied())
        .enumerate()
        .map(move |(i, (a, b))| {
            let (x, y) = (a.into(), b.into());
            assert!(
                x.is_finite() && y.is_finite(),
                "{label}[{i}]: got {x}, expected {y}"
            );
            (a, b)
        })
}
