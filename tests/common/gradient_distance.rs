use crate::distance::named_rel_l2;

pub fn worst_gradient(got: Vec<(String, &[f32])>, want: Vec<(String, &[f32])>) -> (f64, String) {
    assert_eq!(got.len(), want.len(), "gradient tensor counts differ");
    let mut worst = (0.0, String::new());
    for ((name, values), (expected_name, reference)) in got.into_iter().zip(want) {
        assert_eq!(name, expected_name, "gradient tensors differ");
        let error = named_rel_l2(&name, values, reference);
        if error > worst.0 {
            worst = (error, name);
        }
    }
    worst
}
