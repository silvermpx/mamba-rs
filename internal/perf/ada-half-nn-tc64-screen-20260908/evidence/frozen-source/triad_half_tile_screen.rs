#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BracketOrder {
    Abba,
    Baab,
}

pub fn candidate_over_reference(raw: [f64; 4], order: BracketOrder) -> f64 {
    let (candidate, reference) = match order {
        BracketOrder::Abba => (0.5 * (raw[0] + raw[3]), 0.5 * (raw[1] + raw[2])),
        BracketOrder::Baab => (0.5 * (raw[1] + raw[2]), 0.5 * (raw[0] + raw[3])),
    };
    candidate / reference
}

pub fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[index])
}

pub fn retain_stratum(ratios: &[f64]) -> bool {
    if ratios.len() != 7
        || ratios
            .iter()
            .any(|ratio| !ratio.is_finite() || *ratio <= 0.0)
    {
        return false;
    }
    matches!(
        (percentile(ratios, 0.50), percentile(ratios, 0.95)),
        (Some(p50), Some(p95)) if p50 < 0.99 && p95 < 0.99
    )
}

pub fn retain_decision(strata: &[[f64; 2]]) -> bool {
    strata.len() == 4
        && strata.iter().all(|[p50, p95]| {
            p50.is_finite()
                && p95.is_finite()
                && *p50 > 0.0
                && *p95 > 0.0
                && *p50 < 0.99
                && *p95 < 0.99
        })
}

#[cfg(test)]
mod tests {
    use super::{BracketOrder, candidate_over_reference, retain_decision, retain_stratum};

    #[test]
    fn raw_brackets_preserve_direction_and_once7_thresholds() {
        assert_eq!(
            candidate_over_reference([8.0, 10.0, 10.0, 8.0], BracketOrder::Abba),
            0.8
        );
        assert_eq!(
            candidate_over_reference([10.0, 8.0, 8.0, 10.0], BracketOrder::Baab),
            0.8
        );
        assert!(retain_stratum(&[0.98; 7]));
        assert!(!retain_stratum(&[]));
        assert!(!retain_stratum(&[0.98; 6]));
        assert!(!retain_stratum(&[0.98; 8]));
        assert!(!retain_stratum(&[0.0; 7]));
        assert!(!retain_stratum(&[-0.1; 7]));
        assert!(!retain_stratum(&[f64::NAN; 7]));
        assert!(!retain_stratum(&[0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99]));
    }

    #[test]
    fn decision_requires_exactly_four_passing_strata() {
        assert!(retain_decision(&[[0.98, 0.989]; 4]));
        assert!(!retain_decision(&[[0.98, 0.989]; 3]));
        assert!(!retain_decision(&[[0.98, 0.989]; 5]));
        assert!(!retain_decision(&[
            [0.98, 0.989],
            [0.98, 0.989],
            [0.98, 0.99],
            [0.98, 0.989],
        ]));
    }
}
