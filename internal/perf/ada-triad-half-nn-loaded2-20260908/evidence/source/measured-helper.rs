#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BracketOrder {
    Abba,
    Baab,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaHalfArm {
    PortableTc64,
    PortableTc128,
    FixedSm89Tc128Pipeline,
    FixedSm89Tc128Swizzle,
    FixedSm89Tc128S3,
}

impl AdaHalfArm {
    pub const fn name(self) -> &'static str {
        match self {
            Self::PortableTc64 => "forced_tc64",
            Self::PortableTc128 => "forced_tc128",
            Self::FixedSm89Tc128Pipeline => "fixed_sm89_tc128_pipeline",
            Self::FixedSm89Tc128Swizzle => "fixed_sm89_tc128_swizzle",
            Self::FixedSm89Tc128S3 => "fixed_sm89_tc128_s3",
        }
    }

    pub const fn resources(self) -> (u32, i32, usize, u32) {
        match self {
            Self::PortableTc64 => (128, 36_864, 0, 1),
            Self::PortableTc128 => (256, 0, 71_680, 1),
            Self::FixedSm89Tc128Pipeline => (256, 0, 71_680, 1),
            Self::FixedSm89Tc128Swizzle => (256, 0, 69_632, 1),
            Self::FixedSm89Tc128S3 => (256, 0, 98_304, 1),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedHalfS3Params {
    pub alpha: f32,
    pub beta: f32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
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
    use super::{
        AdaHalfArm, BracketOrder, FixedHalfS3Params, candidate_over_reference, retain_decision,
        retain_stratum,
    };
    use std::mem::{offset_of, size_of};

    #[test]
    fn fixed_s3_arm_has_distinct_identity_and_resource_boundary() {
        assert_eq!(AdaHalfArm::PortableTc64.name(), "forced_tc64");
        assert_eq!(AdaHalfArm::PortableTc128.name(), "forced_tc128");
        assert_eq!(AdaHalfArm::FixedSm89Tc128S3.name(), "fixed_sm89_tc128_s3");
        assert_eq!(AdaHalfArm::PortableTc64.resources(), (128, 36_864, 0, 1));
        assert_eq!(AdaHalfArm::PortableTc128.resources(), (256, 0, 71_680, 1));
        assert_eq!(
            AdaHalfArm::FixedSm89Tc128S3.resources(),
            (256, 0, 98_304, 1)
        );
        assert_eq!(
            AdaHalfArm::FixedSm89Tc128Pipeline.name(),
            "fixed_sm89_tc128_pipeline"
        );
        assert_eq!(
            AdaHalfArm::FixedSm89Tc128Pipeline.resources(),
            (256, 0, 71_680, 1)
        );
        assert_eq!(
            AdaHalfArm::FixedSm89Tc128Swizzle.name(),
            "fixed_sm89_tc128_swizzle"
        );
        assert_eq!(
            AdaHalfArm::FixedSm89Tc128Swizzle.resources(),
            (256, 0, 69_632, 1)
        );
    }

    #[test]
    fn fixed_s3_parameter_bundle_is_exactly_eight_words() {
        assert_eq!(size_of::<FixedHalfS3Params>(), 32);
        assert_eq!(offset_of!(FixedHalfS3Params, alpha), 0);
        assert_eq!(offset_of!(FixedHalfS3Params, beta), 4);
        assert_eq!(offset_of!(FixedHalfS3Params, m), 8);
        assert_eq!(offset_of!(FixedHalfS3Params, n), 12);
        assert_eq!(offset_of!(FixedHalfS3Params, k), 16);
        assert_eq!(offset_of!(FixedHalfS3Params, lda), 20);
        assert_eq!(offset_of!(FixedHalfS3Params, ldb), 24);
        assert_eq!(offset_of!(FixedHalfS3Params, ldc), 28);
    }

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
