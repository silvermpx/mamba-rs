use crate::mamba_ssm::gpu::{dtype::WeightDtype, kernel_identity::ResolvedGemmOp};

const SOURCE: &str = include_str!("../../../../kernels/gemm_bi_triad/sm89_half.cu");

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sm89HalfRoute {
    NnM128N128Bk64S3,
    NtM128N128Bk64S3Bxor,
    NtM96N128Bk64S3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm89HalfKernelSpec {
    pub route: Sm89HalfRoute,
    pub op: ResolvedGemmOp,
    pub dtype: WeightDtype,
    pub symbol: &'static str,
    pub tile: (u32, u32),
    pub bk: u32,
    pub stages: u8,
    pub threads: u32,
    pub dynamic_shared_bytes: u32,
    pub register_cap: u32,
    pub occupancy_gate: u32,
}

const fn spec(
    route: Sm89HalfRoute,
    op: ResolvedGemmOp,
    dtype: WeightDtype,
    symbol: &'static str,
    tile: (u32, u32),
    threads: u32,
    dynamic_shared_bytes: u32,
    register_cap: u32,
) -> Sm89HalfKernelSpec {
    Sm89HalfKernelSpec {
        route,
        op,
        dtype,
        symbol,
        tile,
        bk: 64,
        stages: 3,
        threads,
        dynamic_shared_bytes,
        register_cap,
        occupancy_gate: 1,
    }
}

pub const SM89_HALF_KERNEL_SPECS: [Sm89HalfKernelSpec; 6] = [
    spec(
        Sm89HalfRoute::NnM128N128Bk64S3,
        ResolvedGemmOp::Nn,
        WeightDtype::F16,
        "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_f16",
        (128, 128),
        256,
        98_304,
        188,
    ),
    spec(
        Sm89HalfRoute::NnM128N128Bk64S3,
        ResolvedGemmOp::Nn,
        WeightDtype::Bf16,
        "gemm_bi_nn_sm89_m128n128_bk64_s3_v1_bf16",
        (128, 128),
        256,
        98_304,
        188,
    ),
    spec(
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
        ResolvedGemmOp::Nt,
        WeightDtype::F16,
        "gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_f16",
        (128, 128),
        256,
        98_304,
        167,
    ),
    spec(
        Sm89HalfRoute::NtM128N128Bk64S3Bxor,
        ResolvedGemmOp::Nt,
        WeightDtype::Bf16,
        "gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_bf16",
        (128, 128),
        256,
        98_304,
        167,
    ),
    spec(
        Sm89HalfRoute::NtM96N128Bk64S3,
        ResolvedGemmOp::Nt,
        WeightDtype::F16,
        "gemm_bi_nt_sm89_m96n128_bk64_s3_v1_f16",
        (96, 128),
        384,
        86_016,
        120,
    ),
    spec(
        Sm89HalfRoute::NtM96N128Bk64S3,
        ResolvedGemmOp::Nt,
        WeightDtype::Bf16,
        "gemm_bi_nt_sm89_m96n128_bk64_s3_v1_bf16",
        (96, 128),
        384,
        86_016,
        120,
    ),
];

/// Phase-1 production registry. The module is compiled, identified and
/// resource-qualified, but no AUTO request may enter it until a concrete
/// toolkit/device/artifact cohort is frozen in phase 2.
pub const SM89_HALF_AUTO_CELLS: &[(
    ResolvedGemmOp,
    WeightDtype,
    (usize, usize, usize),
    Sm89HalfRoute,
)] = &[];

pub(super) fn kernel_spec(
    route: Sm89HalfRoute,
    dtype: WeightDtype,
) -> Option<&'static Sm89HalfKernelSpec> {
    SM89_HALF_KERNEL_SPECS
        .iter()
        .find(|spec| spec.route == route && spec.dtype == dtype)
}

pub(super) fn compose_sm89_half_source() -> Result<String, String> {
    for spec in SM89_HALF_KERNEL_SPECS {
        let represented_once = match spec.route {
            Sm89HalfRoute::NtM128N128Bk64S3Bxor => {
                SOURCE
                    .matches("gemm_bi_nt_sm89_m128n128_bk64_s3_bxor_v1_##SUFFIX")
                    .count()
                    == 1
                    && SOURCE
                        .matches(match spec.dtype {
                            WeightDtype::F16 => "SM89_HALF_NT_S3_EXPORT(__half, f16)",
                            WeightDtype::Bf16 => "SM89_HALF_NT_S3_EXPORT(__nv_bfloat16, bf16)",
                            WeightDtype::F32 => {
                                return Err("TriadSm89Half owns no f32 export".into());
                            }
                        })
                        .count()
                        == 1
            }
            _ => SOURCE.matches(spec.symbol).count() == 1,
        };
        if !represented_once {
            return Err(format!(
                "TriadSm89Half source must export {} exactly once",
                spec.symbol
            ));
        }
    }
    for forbidden in [
        "_test_",
        "gemm_bi_nn_fixed_sm89_tc128_s3_v1_",
        "gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_",
    ] {
        if SOURCE.contains(forbidden) {
            return Err(format!(
                "TriadSm89Half source retained non-production marker {forbidden}"
            ));
        }
    }
    Ok(SOURCE.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_source_has_six_owned_exports_and_three_physical_families() {
        let source = compose_sm89_half_source().unwrap();
        assert_eq!(SM89_HALF_KERNEL_SPECS.len(), 6);
        assert_eq!(
            SM89_HALF_KERNEL_SPECS
                .iter()
                .map(|spec| spec.route)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        assert!(!source.contains("gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_"));
    }

    #[test]
    fn phase_one_auto_registry_is_empty() {
        assert!(SM89_HALF_AUTO_CELLS.is_empty());
    }
}
