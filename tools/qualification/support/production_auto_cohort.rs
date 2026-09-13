use std::ffi::OsStr;

pub const STATE_CAPACITY_ENV: &str = "GEMM_BI_QUAL_STATE_CAP";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductionAutoInventory {
    Release070,
    Outliers071,
}

impl ProductionAutoInventory {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Release070 => "release_070",
            Self::Outliers071 => "outliers_071",
        }
    }

    const fn expected_counts(self) -> (usize, usize, usize) {
        match self {
            Self::Release070 => (66, 81, 324),
            Self::Outliers071 => (72, 93, 372),
        }
    }
}

pub fn parse_state_capacity(value: Option<&OsStr>) -> Result<usize, String> {
    match value {
        None => Ok(64),
        Some(value) => match value.to_str() {
            Some("16") => Ok(16),
            Some("64") => Ok(64),
            Some(value) => Err(format!(
                "{STATE_CAPACITY_ENV} must be exactly 16 or 64, got {value:?}"
            )),
            None => Err(format!("{STATE_CAPACITY_ENV} must be valid UTF-8")),
        },
    }
}

pub fn state_capacity_from_env() -> Result<usize, String> {
    parse_state_capacity(std::env::var_os(STATE_CAPACITY_ENV).as_deref())
}

pub fn render_cohort_fragment(
    inventory: ProductionAutoInventory,
    state_capacity: usize,
    cublas_workspace_bytes: usize,
) -> String {
    format!(
        "\"inventory\":\"{}\",\"state_capacity\":{state_capacity},\"cublas_workspace_bytes\":{cublas_workspace_bytes}",
        inventory.label()
    )
}

pub fn validate_completion_counts(
    inventory: ProductionAutoInventory,
    cells: usize,
    comparator_views: usize,
    records: usize,
) -> Result<(), String> {
    let expected = inventory.expected_counts();
    let actual = (cells, comparator_views, records);
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{} completion requires cells={}, views={}, records={}; got cells={cells}, views={comparator_views}, records={records}",
            inventory.label(),
            expected.0,
            expected.1,
            expected.2,
        ))
    }
}
