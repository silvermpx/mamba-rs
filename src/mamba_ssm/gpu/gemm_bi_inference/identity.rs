//! Closed terminal identities for existing Inference launches; never a selector.
//! Context routes bind actual pointers. Physical observations additionally bind
//! prevalidated allocation identities; only the latter prove allocation liveness.
use super::super::context::{BiGemmFamily, GpuCtx};
use super::super::kernel_identity::{
    FramedSha256, ModuleKind, NumericContractSet, PhysicalGemmBackend, PhysicalLaunchObservation,
    PhysicalLaunchObserver, PolicyDtype, ResolvedGemmOp, ResolvedGemmRoute,
    ResolvedInstructionFamily, ResolvedInstructionShape, ResolvedKernelLaunch,
    ResolvedNumericContract, ResolvedOperandConversion, ResolvedOutputOwnership, SCHEDULE_REVISION,
    SM89_FIXED_COPYPLAN_ROUTE_REVISION, Sha256Digest, TUNING_TABLE_REVISION,
};
use cudarc::driver::{CudaFunction, LaunchConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(in crate::mamba_ssm::gpu) enum AbiKind {
    Legacy = 1,
    Tf32 = 2,
    Tf32Wide = 3,
    ExactF32 = 4,
    HalfSm89 = 5,
    Sm120Tf32 = 6,
    Sm120Half = 7,
    Sm120PostBias = 8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BiasDomain {
    Optional,
    Required,
    Forbidden,
}

#[derive(Clone, Copy)]
pub(in crate::mamba_ssm::gpu) struct TerminalSpec {
    pub(super) symbol: &'static str,
    storage: [PolicyDtype; 3],
    backend: PhysicalGemmBackend,
    tile: (u32, u32),
    bk: u32,
    stages: u8,
    threads: u32,
    shared: u32,
    abi: AbiKind,
    bias: BiasDomain,
}

macro_rules! terminal {
    ($symbol:literal, ($input:ident, $output:ident), $backend:ident,
     ($m:literal, $n:literal, $bk:literal, $stages:literal, $threads:literal, $shared:literal),
     $abi:ident, $bias:ident) => {
        TerminalSpec {
            symbol: $symbol,
            storage: [
                PolicyDtype::$input,
                PolicyDtype::$input,
                PolicyDtype::$output,
            ],
            backend: PhysicalGemmBackend::$backend,
            tile: ($m, $n),
            bk: $bk,
            stages: $stages,
            threads: $threads,
            shared: $shared,
            abi: AbiKind::$abi,
            bias: BiasDomain::$bias,
        }
    };
}

// Every spelling is literal and admitted separately, including pair-store and
// native half-to-F32. No suffix acceptance or dynamic symbol construction.
const TERMINALS: &[TerminalSpec] = &[
    terminal!(
        "f32_f32_s2",
        (F32, F32),
        InferenceScalarFma,
        (64, 64, 32, 2, 128, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "f32_f32_n128_s2",
        (F32, F32),
        InferenceScalarFma,
        (64, 128, 32, 2, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_sm89_f32_n64_copyplan",
        (F32, F32),
        ScalarFmaSm89FixedCopyPlan,
        (64, 64, 32, 2, 128, 0),
        ExactF32,
        Optional
    ),
    terminal!(
        "nn_sm89_tc128_f32out_s3_bf16",
        (Bf16, F32),
        InferenceMma16,
        (128, 128, 64, 3, 256, 98304),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_sm89_tc128_f32out_s3_f16",
        (F16, F32),
        InferenceMma16,
        (128, 128, 64, 3, 256, 98304),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_sm89_f32_m128n64_tail_copyplan",
        (F32, F32),
        InferenceScalarFma,
        (128, 64, 32, 2, 256, 0),
        ExactF32,
        Forbidden
    ),
    terminal!(
        "nn_sm120_f32_n64_copyplan",
        (F32, F32),
        InferenceScalarFma,
        (64, 64, 32, 2, 128, 0),
        ExactF32,
        Optional
    ),
    terminal!(
        "nn_sm120_f32_n64_copyplan_t256",
        (F32, F32),
        InferenceScalarFma,
        (64, 64, 32, 2, 256, 0),
        ExactF32,
        Optional
    ),
    terminal!(
        "nn_sm120_f32_n64_copyplan_m128n64_t256",
        (F32, F32),
        InferenceScalarFma,
        (128, 64, 32, 2, 256, 0),
        ExactF32,
        Optional
    ),
    terminal!(
        "nn_sm120_f32_n64_sliced",
        (F32, F32),
        InferenceScalarFma,
        (64, 64, 32, 2, 128, 0),
        ExactF32,
        Optional
    ),
    terminal!(
        "bf16_bf16",
        (Bf16, Bf16),
        InferenceWmma,
        (64, 64, 32, 1, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "matvec_bi_bf16_bf16",
        (Bf16, Bf16),
        FixedMatvecEightWarp,
        (1, 32, 0, 1, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "bf16_f32",
        (Bf16, F32),
        InferenceWmma,
        (64, 64, 32, 1, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "matvec_bi_bf16_f32",
        (Bf16, F32),
        FixedMatvecEightWarp,
        (1, 32, 0, 1, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc128_bf16",
        (Bf16, Bf16),
        InferenceMma16,
        (128, 128, 64, 2, 256, 71680),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc128_f32out_bf16",
        (Bf16, F32),
        InferenceMma16,
        (128, 128, 64, 2, 256, 71680),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tcw64_bf16",
        (Bf16, Bf16),
        InferenceMma16,
        (128, 128, 64, 2, 128, 65536),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tcwn64_bf16",
        (Bf16, Bf16),
        InferenceMma16,
        (128, 256, 64, 2, 256, 98304),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc64_bf16",
        (Bf16, Bf16),
        InferenceMma16,
        (64, 64, 64, 2, 128, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc64_f32out_bf16",
        (Bf16, F32),
        InferenceMma16,
        (64, 64, 64, 2, 128, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc16_bf16",
        (Bf16, Bf16),
        InferenceMma16,
        (16, 32, 64, 4, 128, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc16_f32out_bf16",
        (Bf16, F32),
        InferenceMma16,
        (16, 32, 64, 4, 128, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_sm89_tc128_pipeline_bf16",
        (Bf16, Bf16),
        InferenceMma16,
        (128, 128, 64, 2, 256, 71680),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_sm89_tc128_swizzle_bf16",
        (Bf16, Bf16),
        InferenceMma16,
        (128, 128, 64, 2, 256, 69632),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_sm89_tc128_s3_bf16",
        (Bf16, Bf16),
        InferenceMma16,
        (128, 128, 64, 3, 256, 98304),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_64x64_bk64_s2_bf16",
        (Bf16, Bf16),
        InferenceSm120TmaMma16,
        (64, 64, 64, 2, 128, 32896),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_64x64_bk64_s2_f32out_bf16",
        (Bf16, F32),
        InferenceSm120TmaMma16,
        (64, 64, 64, 2, 128, 32896),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_64x128_bk64_s2_bf16",
        (Bf16, Bf16),
        InferenceSm120TmaMma16,
        (64, 128, 64, 2, 256, 49280),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_64x128_bk64_s2_f32out_bf16",
        (Bf16, F32),
        InferenceSm120TmaMma16,
        (64, 128, 64, 2, 256, 49280),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x64_bk32_s3_bf16",
        (Bf16, Bf16),
        InferenceSm120TmaMma16,
        (128, 64, 32, 3, 256, 36992),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x64_bk32_s3_f32out_bf16",
        (Bf16, F32),
        InferenceSm120TmaMma16,
        (128, 64, 32, 3, 256, 36992),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x128_bk32_s2_bf16",
        (Bf16, Bf16),
        InferenceSm120TmaMma16,
        (128, 128, 32, 2, 256, 32896),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x128_bk32_s2_f32out_bf16",
        (Bf16, F32),
        InferenceSm120TmaMma16,
        (128, 128, 32, 2, 256, 32896),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x128_bk32_s3_bf16",
        (Bf16, Bf16),
        InferenceSm120TmaMma16,
        (128, 128, 32, 3, 256, 49280),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x128_bk32_s3_f32out_bf16",
        (Bf16, F32),
        InferenceSm120TmaMma16,
        (128, 128, 32, 3, 256, 49280),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm90a_wgmma_wg1_bf16",
        (Bf16, Bf16),
        InferenceSm90aWgmma,
        (64, 128, 64, 2, 128, 49152),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_sm100_tcgen_c4_bf16",
        (Bf16, Bf16),
        InferenceSm100Tcgen05,
        (128, 128, 64, 2, 128, 65536),
        Legacy,
        Optional
    ),
    terminal!(
        "f16_f16",
        (F16, F16),
        InferenceWmma,
        (64, 64, 32, 1, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "matvec_bi_f16_f16",
        (F16, F16),
        FixedMatvecEightWarp,
        (1, 32, 0, 1, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "f16_f32",
        (F16, F32),
        InferenceWmma,
        (64, 64, 32, 1, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "matvec_bi_f16_f32",
        (F16, F32),
        FixedMatvecEightWarp,
        (1, 32, 0, 1, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc128_f16",
        (F16, F16),
        InferenceMma16,
        (128, 128, 64, 2, 256, 71680),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc128_f32out_f16",
        (F16, F32),
        InferenceMma16,
        (128, 128, 64, 2, 256, 71680),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tcw64_f16",
        (F16, F16),
        InferenceMma16,
        (128, 128, 64, 2, 128, 65536),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tcwn64_f16",
        (F16, F16),
        InferenceMma16,
        (128, 256, 64, 2, 256, 98304),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc64_f16",
        (F16, F16),
        InferenceMma16,
        (64, 64, 64, 2, 128, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc64_f32out_f16",
        (F16, F32),
        InferenceMma16,
        (64, 64, 64, 2, 128, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc16_f16",
        (F16, F16),
        InferenceMma16,
        (16, 32, 64, 4, 128, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_tc16_f32out_f16",
        (F16, F32),
        InferenceMma16,
        (16, 32, 64, 4, 128, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_sm89_tc128_pipeline_f16",
        (F16, F16),
        InferenceMma16,
        (128, 128, 64, 2, 256, 71680),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_sm89_tc128_swizzle_f16",
        (F16, F16),
        InferenceMma16,
        (128, 128, 64, 2, 256, 69632),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_sm89_tc128_s3_f16",
        (F16, F16),
        InferenceMma16,
        (128, 128, 64, 3, 256, 98304),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_64x64_bk64_s2_f16",
        (F16, F16),
        InferenceSm120TmaMma16,
        (64, 64, 64, 2, 128, 32896),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_64x64_bk64_s2_f32out_f16",
        (F16, F32),
        InferenceSm120TmaMma16,
        (64, 64, 64, 2, 128, 32896),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_64x128_bk64_s2_f16",
        (F16, F16),
        InferenceSm120TmaMma16,
        (64, 128, 64, 2, 256, 49280),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_64x128_bk64_s2_f32out_f16",
        (F16, F32),
        InferenceSm120TmaMma16,
        (64, 128, 64, 2, 256, 49280),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x64_bk32_s3_f16",
        (F16, F16),
        InferenceSm120TmaMma16,
        (128, 64, 32, 3, 256, 36992),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x64_bk32_s3_f32out_f16",
        (F16, F32),
        InferenceSm120TmaMma16,
        (128, 64, 32, 3, 256, 36992),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x128_bk32_s2_f16",
        (F16, F16),
        InferenceSm120TmaMma16,
        (128, 128, 32, 2, 256, 32896),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x128_bk32_s2_f32out_f16",
        (F16, F32),
        InferenceSm120TmaMma16,
        (128, 128, 32, 2, 256, 32896),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x128_bk32_s3_f16",
        (F16, F16),
        InferenceSm120TmaMma16,
        (128, 128, 32, 3, 256, 49280),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_128x128_bk32_s3_f32out_f16",
        (F16, F32),
        InferenceSm120TmaMma16,
        (128, 128, 32, 3, 256, 49280),
        Sm120Half,
        Optional
    ),
    terminal!(
        "nn_sm90a_wgmma_wg1_f16",
        (F16, F16),
        InferenceSm90aWgmma,
        (64, 128, 64, 2, 128, 49152),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_sm100_tcgen_c4_f16",
        (F16, F16),
        InferenceSm100Tcgen05,
        (128, 128, 64, 2, 128, 65536),
        Legacy,
        Optional
    ),
    terminal!(
        "matvec_bi_f32_f32",
        (F32, F32),
        FixedMatvecEightWarp,
        (1, 32, 0, 1, 256, 0),
        Legacy,
        Optional
    ),
    terminal!(
        "nn_sm89_m64n64_bk64_s3_f16",
        (F16, F16),
        InferenceMma16,
        (64, 64, 64, 3, 128, 49152),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_sm89_m128n64_bk64_s2_f16",
        (F16, F16),
        InferenceMma16,
        (128, 64, 64, 2, 128, 49152),
        HalfSm89,
        Optional
    ),
    terminal!(
        "nn_tf32_m128n64_bk32_s2",
        (F32, F32),
        InferenceMmaTf32Rna,
        (128, 64, 32, 2, 256, 55296),
        Tf32,
        Optional
    ),
    terminal!(
        "nn_tf32_m128n64_bk32_s3",
        (F32, F32),
        InferenceMmaTf32Rna,
        (128, 64, 32, 3, 256, 82944),
        Tf32,
        Optional
    ),
    terminal!(
        "nn_tf32_m64n64_bk32_s2",
        (F32, F32),
        InferenceMmaTf32Rna,
        (64, 64, 32, 2, 128, 32768),
        Tf32,
        Optional
    ),
    terminal!(
        "nn_tf32_m64n64_bk32_s3",
        (F32, F32),
        InferenceMmaTf32Rna,
        (64, 64, 32, 3, 128, 55296),
        Tf32,
        Optional
    ),
    terminal!(
        "nn_tf32_m16n32_bk32_s4",
        (F32, F32),
        InferenceMmaTf32Rna,
        (16, 32, 32, 4, 128, 29696),
        Tf32,
        Optional
    ),
    terminal!(
        "nn_rna_wide_tf32_m128n128_bk32_s3",
        (F32, F32),
        InferenceMmaTf32Rna,
        (128, 128, 32, 3, 256, 98304),
        Tf32Wide,
        Optional
    ),
    terminal!(
        "nn_sm89_rna_tf32_m128n96_bk32_s3",
        (F32, F32),
        InferenceMmaTf32Rna,
        (128, 96, 32, 3, 256, 86016),
        Tf32Wide,
        Optional
    ),
    terminal!(
        "nn_sm80_mma_tf32_m128n128_bk32_s3",
        (F32, F32),
        MmaTf32Rna,
        (128, 128, 32, 3, 256, 98304),
        Tf32Wide,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_tf32_m128n64_bk32_s2",
        (F32, F32),
        InferenceSm120TmaMmaTf32Rna,
        (128, 64, 32, 2, 128, 49280),
        Sm120Tf32,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_tf32_m128n64_bk32_s3",
        (F32, F32),
        InferenceSm120TmaMmaTf32Rna,
        (128, 64, 32, 3, 256, 73856),
        Sm120Tf32,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_tf32_m64n128_bk32_s2",
        (F32, F32),
        InferenceSm120TmaMmaTf32Rna,
        (64, 128, 32, 2, 128, 49280),
        Sm120Tf32,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_tf32_m64n128_bk32_s3",
        (F32, F32),
        InferenceSm120TmaMmaTf32Rna,
        (64, 128, 32, 3, 256, 73856),
        Sm120Tf32,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_tf32_m64n64_bk32_s2_producer_warp",
        (F32, F32),
        InferenceSm120TmaMmaTf32Rna,
        (64, 64, 32, 2, 160, 32896),
        Sm120Tf32,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_tf32_m64n64_bk32_s2",
        (F32, F32),
        InferenceSm120TmaMmaTf32Rna,
        (64, 64, 32, 2, 128, 32896),
        Sm120Tf32,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store",
        (F32, F32),
        InferenceSm120TmaMmaTf32Rna,
        (64, 64, 32, 2, 128, 32896),
        Sm120Tf32,
        Optional
    ),
    terminal!(
        "nn_sm120_tma_fma_postbias_m128n64_bk16_s2",
        (F32, F32),
        InferenceSm120TmaFma,
        (128, 64, 16, 2, 128, 24592),
        Sm120PostBias,
        Required
    ),
    terminal!(
        "nn_sm120_tma_fma_postbias_m64n128_bk16_s2",
        (F32, F32),
        InferenceSm120TmaFma,
        (64, 128, 16, 2, 128, 24592),
        Sm120PostBias,
        Required
    ),
    terminal!(
        "nn_sm120_tma_fma_postbias_m128n96_bk16_s2",
        (F32, F32),
        InferenceSm120TmaFma,
        (128, 96, 16, 2, 256, 28688),
        Sm120PostBias,
        Required
    ),
    terminal!(
        "nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4",
        (F32, F32),
        InferenceSm120TmaFma,
        (128, 64, 16, 2, 128, 24592),
        Sm120PostBias,
        Required
    ),
    terminal!(
        "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2",
        (F32, F32),
        InferenceSm120TmaFma,
        (128, 64, 16, 2, 256, 24592),
        Sm120PostBias,
        Required
    ),
    terminal!(
        "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2",
        (F32, F32),
        InferenceSm120TmaFma,
        (128, 64, 16, 2, 256, 24592),
        Sm120PostBias,
        Forbidden
    ),
];

pub(in crate::mamba_ssm::gpu) fn terminal(symbol: &str) -> Option<&'static TerminalSpec> {
    TERMINALS.iter().find(|spec| spec.symbol == symbol)
}

/// The cached bridge keeps the existing prepared Triad identity. These are
/// its two actual NN/single-split entries, not aliases in the Fixed table.
pub(in crate::mamba_ssm::gpu) fn validate_cached_bridge(
    route: &ResolvedGemmRoute,
) -> Result<(), String> {
    if !matches!(
        route.symbol,
        "nn_sm120_tma_fma_m128n64_bk16_s2" | "nn_sm120_tma_fma_m64n128_bk16_s2"
    ) {
        return Err(
            "Inference exact-TMA bridge symbol is not one of its two prepared entries".into(),
        );
    }
    let spec = super::super::gemm_bi_triad::tf32_route_specs_all(ModuleKind::TriadSm120)
        .find(|spec| spec.symbol == route.symbol)
        .ok_or("missing prepared exact-TMA bridge spec")?;
    if route.backend != PhysicalGemmBackend::Sm120TmaFmaExact
        || route.numeric_contract != ResolvedNumericContract::ScalarFma
        || route.op != ResolvedGemmOp::Nn
        || route.dtype != PolicyDtype::F32
        || route.ownership != ResolvedOutputOwnership::OneCtaPerOutputTile
        || route.module_kind != spec.module_kind
        || route.instruction_family != spec.instruction_family
        || route.instruction_shape != spec.instruction_shape
        || route.operand_conversion != spec.operand_conversion
        || route.tile != spec.tile
        || route.bk != spec.bk
        || route.stages != spec.stages
        || route.threads != spec.threads
        || route.launch.block_dim != (spec.threads, 1, 1)
        || route.launch.shared_mem_bytes != spec.dynamic_shared_bytes
        || route.tensor_map_revision != spec.tensor_map_revision
        || route.tensor_maps_digest == [0; 32]
        || route.tuning_table_revision != super::super::gemm_bi_triad::F32_TF32_TUNING_REVISION
        || route.schedule_revision != spec.schedule_revision
        || route.strides != (route.shape.1, route.shape.2, route.shape.2)
    {
        return Err("Inference exact-TMA bridge contradicts its existing prepared identity".into());
    }
    let (m, k, n) = route.shape;
    let grid = u32::try_from(m.div_ceil(spec.tile.0 as usize))
        .ok()
        .zip(u32::try_from(n.div_ceil(spec.tile.1 as usize)).ok())
        .and_then(|(m, n)| m.checked_mul(n))
        .ok_or("exact-TMA bridge grid overflow")?;
    if m == 0 || k == 0 || n == 0 || route.launch.grid_dim != (grid, 1, 1) {
        return Err("Inference exact-TMA bridge shape/config changed".into());
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Arithmetic {
    numeric: ResolvedNumericContract,
    instruction: ResolvedInstructionFamily,
    shape: ResolvedInstructionShape,
    conversion: ResolvedOperandConversion,
}

impl TerminalSpec {
    fn arithmetic(&self) -> Arithmetic {
        use PhysicalGemmBackend as B;
        use ResolvedInstructionFamily as I;
        use ResolvedNumericContract as N;
        use ResolvedOperandConversion as C;
        let (numeric, instruction, (m, n, k), conversion) = match self.backend {
            B::InferenceScalarFma | B::ScalarFmaSm89FixedCopyPlan | B::InferenceSm120TmaFma => {
                (N::ScalarFmaPostDotBias, I::ScalarFma, (1, 1, 1), C::None)
            }
            B::InferenceWmma => (N::WmmaF32PostDotBias, I::WmmaApi, (16, 16, 16), C::None),
            B::InferenceMma16 | B::InferenceSm120TmaMma16 => {
                (N::MmaSyncF32, I::MmaSync, (16, 8, 16), C::None)
            }
            B::InferenceSm90aWgmma => (N::WgmmaF32, I::Wgmma, (64, 128, 16), C::None),
            B::InferenceSm100Tcgen05 => (N::Tcgen05F32, I::Tcgen05, (128, 128, 16), C::None),
            B::InferenceMmaTf32Rna => (
                N::MmaTf32Rna,
                I::MmaSync,
                (16, 8, 8),
                C::RegisterCvtRnaTf32F32,
            ),
            B::InferenceSm120TmaMmaTf32Rna => (
                N::Sm120TmaMmaTf32Rna,
                I::MmaSync,
                (16, 8, 8),
                C::TensorMapUint32ThenCvtRnaTf32F32,
            ),
            B::FixedMatvecEightWarp => (
                N::ScalarFmaEightWarpTreePostDotBias,
                I::ScalarFma,
                (1, 1, 1),
                C::None,
            ),
            B::MmaTf32Rna => (
                N::MmaTf32AddHalfUlp,
                I::MmaSync,
                (16, 8, 8),
                C::RegisterAddHalfUlpTf32,
            ),
            _ => unreachable!("closed Inference terminal table"),
        };
        Arithmetic {
            numeric,
            instruction,
            shape: ResolvedInstructionShape { m, n, k },
            conversion,
        }
    }

    fn module(&self) -> ModuleKind {
        if self.backend == PhysicalGemmBackend::MmaTf32Rna {
            ModuleKind::TriadSm80
        } else {
            ModuleKind::Fixed
        }
    }

    fn tuning_revision(&self) -> u16 {
        if self.backend == PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlan {
            SM89_FIXED_COPYPLAN_ROUTE_REVISION
        } else {
            TUNING_TABLE_REVISION
        }
    }

    fn uses_maps(&self) -> bool {
        matches!(
            self.abi,
            AbiKind::Sm120Tf32 | AbiKind::Sm120Half | AbiKind::Sm120PostBias
        )
    }

    pub(in crate::mamba_ssm::gpu) fn validate_route(
        &self,
        route: &ResolvedGemmRoute,
    ) -> Result<(), String> {
        let arithmetic = self.arithmetic();
        if route.symbol != self.symbol
            || route.op != ResolvedGemmOp::Nn
            || route.dtype != self.storage[0]
            || route.backend != self.backend
            || route.numeric_contract != arithmetic.numeric
            || route.instruction_family != arithmetic.instruction
            || route.instruction_shape != arithmetic.shape
            || route.operand_conversion != arithmetic.conversion
            || route.ownership != ResolvedOutputOwnership::OneCtaPerOutputTile
            || route.module_kind != self.module()
            || route.artifact.module_kind != self.module()
            || route.tile != self.tile
            || route.bk != self.bk
            || route.stages != self.stages
            || route.threads != self.threads
            || route.tuning_table_revision != self.tuning_revision()
            || route.schedule_revision != SCHEDULE_REVISION
            || route.strides != (route.shape.1, route.shape.2, route.shape.2)
        {
            return Err("Inference terminal route contradicts its exact symbol contract".into());
        }
        let (m, k, n) = route.shape;
        if m == 0 || n == 0 {
            return Err("Inference terminal cannot record empty output".into());
        }
        let m = u32::try_from(m).map_err(|_| "Inference M exceeds u32")?;
        let n = u32::try_from(n).map_err(|_| "Inference N exceeds u32")?;
        let (grid, shared) = if self.backend == PhysicalGemmBackend::FixedMatvecEightWarp {
            let shared = u32::try_from(k)
                .ok()
                .and_then(|k| k.checked_mul(width(self.storage[0]) as u32))
                .and_then(|bytes| bytes.checked_add(15))
                .ok_or("Inference matvec shared span exceeds u32")?
                & !15;
            ((n.div_ceil(32), m, 1), shared)
        } else {
            (
                (
                    m.div_ceil(self.tile.0)
                        .checked_mul(n.div_ceil(self.tile.1))
                        .ok_or("Inference grid exceeds u32")?,
                    1,
                    1,
                ),
                self.shared,
            )
        };
        if route.launch.grid_dim != grid
            || route.launch.block_dim != (self.threads, 1, 1)
            || route.launch.shared_mem_bytes != shared
            || route.tensor_map_revision != u16::from(self.uses_maps())
            || self.uses_maps() != (route.tensor_maps_digest != [0; 32])
        {
            return Err("Inference terminal launch geometry or maps changed".into());
        }
        Ok(())
    }

    pub(in crate::mamba_ssm::gpu) fn required_contract(
        &self,
        family: BiGemmFamily,
    ) -> Result<NumericContractSet, String> {
        use PhysicalGemmBackend as B;
        let expected_family = if self.backend == B::FixedMatvecEightWarp {
            BiGemmFamily::Triad
        } else {
            BiGemmFamily::Inference
        };
        if family != expected_family {
            return Err("Inference terminal has the wrong live GEMM family".into());
        }
        Ok(match self.backend {
            B::InferenceScalarFma | B::ScalarFmaSm89FixedCopyPlan | B::InferenceSm120TmaFma => {
                NumericContractSet::FIXED_SCALAR_FMA
            }
            B::InferenceWmma
            | B::InferenceMma16
            | B::InferenceSm90aWgmma
            | B::InferenceSm100Tcgen05
            | B::InferenceSm120TmaMma16 => NumericContractSet::FIXED_MMA_SYNC,
            B::InferenceMmaTf32Rna | B::InferenceSm120TmaMmaTf32Rna | B::MmaTf32Rna => {
                NumericContractSet::FIXED_DETERMINISTIC_TF32
            }
            B::FixedMatvecEightWarp => NumericContractSet::FIXED_MATVEC_TREE,
            _ => unreachable!("closed Inference terminal table"),
        })
    }
}

fn width(dtype: PolicyDtype) -> u64 {
    match dtype {
        PolicyDtype::F32 => 4,
        PolicyDtype::Bf16 | PolicyDtype::F16 => 2,
    }
}

pub(in crate::mamba_ssm::gpu) fn policy_dtype(dtype: super::WeightDtype) -> PolicyDtype {
    match dtype {
        super::WeightDtype::F32 | super::WeightDtype::Tf32 => PolicyDtype::F32,
        super::WeightDtype::Bf16 => PolicyDtype::Bf16,
        super::WeightDtype::F16 => PolicyDtype::F16,
    }
}

/// Actual scalar/parameter fields in CUDA ABI order; unused words are zero.
/// Opaque CUtensorMap fields are copied individually, never Rust padding.
#[derive(Clone, Copy)]
pub(in crate::mamba_ssm::gpu) struct Arguments {
    pub(in crate::mamba_ssm::gpu) pointers: [u64; 4], // C,A,B,bias
    pub(in crate::mamba_ssm::gpu) storage: [PolicyDtype; 3],
    pub(in crate::mamba_ssm::gpu) abi: AbiKind,
    pub(in crate::mamba_ssm::gpu) words: [u32; 10],
    pub(in crate::mamba_ssm::gpu) maps: Option<[[u64; 16]; 2]>,
    pub(in crate::mamba_ssm::gpu) auxiliary: [u64; 2],
}

impl Arguments {
    pub(in crate::mamba_ssm::gpu) fn legacy(
        pointers: [u64; 4],
        storage: [PolicyDtype; 3],
        scalars: [f32; 2],
        dimensions: [i32; 6],
    ) -> Self {
        let [m, n, k, lda, ldb, ldc] = dimensions;
        Self {
            pointers,
            storage,
            abi: AbiKind::Legacy,
            words: [
                scalars[0].to_bits(),
                scalars[1].to_bits(),
                m as u32,
                n as u32,
                k as u32,
                lda as u32,
                ldb as u32,
                ldc as u32,
                0,
                0,
            ],
            maps: None,
            auxiliary: [0; 2],
        }
    }

    fn dimensions(&self) -> Result<(usize, usize, usize), String> {
        let (m, k, n) = match self.abi {
            AbiKind::Legacy | AbiKind::ExactF32 | AbiKind::HalfSm89 => {
                (self.words[2], self.words[4], self.words[3])
            }
            AbiKind::Tf32 | AbiKind::Sm120Tf32 => (self.words[0], self.words[1], self.words[2]),
            AbiKind::Tf32Wide => (self.words[2], self.words[3], self.words[4]),
            AbiKind::Sm120Half => (self.words[6], self.words[7], self.words[8]),
            AbiKind::Sm120PostBias => (self.words[2], self.words[4], self.words[3]),
        };
        if [m, k, n].iter().any(|v| *v > i32::MAX as u32) {
            return Err("Inference ABI dimension exceeds i32".into());
        }
        Ok((m as usize, k as usize, n as usize))
    }

    fn scalar_bits(&self) -> [u32; 2] {
        match self.abi {
            AbiKind::Tf32 | AbiKind::Sm120Tf32 => [1f32.to_bits(), 0],
            AbiKind::Sm120Half => [self.words[4], self.words[5]],
            _ => [self.words[0], self.words[1]],
        }
    }

    fn validate(&self, route: &ResolvedGemmRoute) -> Result<(), String> {
        let spec = terminal(route.symbol).ok_or("unknown Inference terminal symbol")?;
        spec.validate_route(route)?;
        if self.storage != spec.storage
            || self.abi != spec.abi
            || self.dimensions()? != route.shape
            || self.maps.is_some() != spec.uses_maps()
            || self.auxiliary != [0; 2]
        {
            return Err("Inference storage/ABI/map/auxiliary binding changed".into());
        }
        let (m, k, n) = route.shape;
        let (m, k, n) = (m as u32, k as u32, n as u32);
        let [alpha, beta] = self.scalar_bits();
        // The inference entry submits this exact epilogue. The generic matvec
        // boundary binds its actual scalar arguments with the same forward contract.
        if alpha != 1f32.to_bits() || beta != 0 {
            return Err("Inference alpha/beta contract changed".into());
        }
        let expected = match self.abi {
            AbiKind::Legacy | AbiKind::ExactF32 | AbiKind::HalfSm89 => {
                [alpha, beta, m, n, k, k, n, n, 0, 0]
            }
            AbiKind::Tf32 => [m, k, n, k, n, n, 0, 0, 0, 0],
            AbiKind::Tf32Wide => [alpha, beta, m, k, n, k, n, n, 0, 0],
            AbiKind::Sm120Tf32 => [m, k, n, n, 0, 0, 0, 0, 0, 0],
            AbiKind::Sm120Half => [0, 0, 0, 0, alpha, beta, m, k, n, n],
            AbiKind::Sm120PostBias => [alpha, beta, m, n, k, n, 1, k.div_ceil(16), 0, 0],
        };
        if self.words != expected {
            return Err("Inference parameter or stride binding changed".into());
        }
        if (spec.bias == BiasDomain::Required && self.pointers[3] == 0)
            || (spec.bias == BiasDomain::Forbidden && self.pointers[3] != 0)
        {
            return Err("Inference terminal bias domain changed".into());
        }
        let spans = self.spans(route.shape)?;
        for (pointer, bytes) in self.pointers.into_iter().zip(spans) {
            if bytes != 0 && pointer == 0 {
                return Err("Inference nonempty operand is null".into());
            }
            pointer
                .checked_add(bytes)
                .ok_or("Inference pointer span overflows u64")?;
        }
        if route.tensor_maps_digest != self.maps_digest() {
            return Err("Inference tensor map binding changed".into());
        }
        Ok(())
    }

    fn spans(&self, (m, k, n): (usize, usize, usize)) -> Result<[u64; 4], String> {
        let bytes = |rows: usize, cols: usize, dtype: PolicyDtype| {
            rows.checked_mul(cols)
                .and_then(|elements| u64::try_from(elements).ok())
                .and_then(|elements| elements.checked_mul(width(dtype)))
                .ok_or_else(|| "Inference operand span overflows".to_string())
        };
        Ok([
            bytes(m, n, self.storage[2])?,
            bytes(m, k, self.storage[0])?,
            bytes(k, n, self.storage[1])?,
            if self.pointers[3] == 0 {
                0
            } else {
                bytes(1, n, PolicyDtype::F32)?
            },
        ])
    }

    fn maps_digest(&self) -> Sha256Digest {
        let Some(maps) = self.maps else {
            return [0; 32];
        };
        let mut hash = FramedSha256::new(b"mamba-rs.inference-nn-tensor-maps.v1");
        for map in maps {
            for field in map {
                hash = hash.required(b"opaque-map-field", &field.to_le_bytes());
            }
        }
        hash.finish()
    }

    fn digest(
        &self,
        route: &ResolvedGemmRoute,
        allocations: Option<[Sha256Digest; 4]>,
    ) -> Sha256Digest {
        let mut hash = FramedSha256::new(if allocations.is_some() {
            b"mamba-rs.inference-nn-gemm-allocation-arguments.v1"
        } else {
            b"mamba-rs.inference-nn-gemm-pointer-arguments.v1"
        });
        hash = hash
            .required(b"symbol", route.symbol.as_bytes())
            .required(b"backend", &[route.backend as u8])
            .required(b"op", &[route.op as u8])
            .required(b"storage", &self.storage.map(|d| d as u8))
            .required(b"abi", &[self.abi as u8])
            .required(
                b"null-mask",
                &[self
                    .pointers
                    .iter()
                    .enumerate()
                    .fold(0u8, |mask, (i, p)| mask | (u8::from(*p == 0) << i))],
            );
        for value in [
            route.shape.0,
            route.shape.1,
            route.shape.2,
            route.strides.0,
            route.strides.1,
            route.strides.2,
        ] {
            hash = hash.required(b"shape-stride", &(value as u64).to_le_bytes());
        }
        for word in self.words {
            hash = hash.required(b"abi-field", &word.to_le_bytes());
        }
        for scalar in self.scalar_bits() {
            hash = hash.required(b"scalar-bits", &scalar.to_le_bytes());
        }
        for pointer in self.auxiliary {
            hash = hash.required(b"null-auxiliary", &pointer.to_le_bytes());
        }
        if let Some(allocations) = allocations {
            for allocation in allocations {
                hash = hash.required(b"allocation", &allocation);
            }
        } else {
            for pointer in self.pointers {
                hash = hash.required(b"pointer", &pointer.to_le_bytes());
            }
        }
        hash.required(b"map-revision", &route.tensor_map_revision.to_le_bytes())
            .required(b"maps", &self.maps_digest())
            .finish()
    }

    pub(in crate::mamba_ssm::gpu) fn resolve<O: PhysicalLaunchObserver>(
        &self,
        observer: &O,
        route: &ResolvedGemmRoute,
        config: LaunchConfig,
    ) -> Result<Sha256Digest, String> {
        self.validate(route)?;
        if route.launch.grid_dim != config.grid_dim
            || route.launch.block_dim != config.block_dim
            || route.launch.shared_mem_bytes != config.shared_mem_bytes
            || route.launch.arguments_digest != self.digest(route, None)
        {
            return Err("Inference observed launch or pointer-bound arguments changed".into());
        }
        let spans = self.spans(route.shape)?;
        let mut allocations = [[0; 32]; 4];
        for i in 0..4 {
            // K0 preserves actual presence in the null mask and never fabricates
            // zero-length input allocation identities.
            if spans[i] != 0 {
                allocations[i] = observer.argument_identity_digest(self.pointers[i], spans[i])?;
            }
        }
        Ok(self.digest(route, Some(allocations)))
    }
}

/// Production fast-path seam; the closure must include every metadata lookup,
/// digest and allocation query introduced by recording.
pub(super) fn when_recording<T>(
    record: bool,
    build: impl FnOnce() -> Result<T, String>,
) -> Result<Option<T>, String> {
    if record { build().map(Some) } else { Ok(None) }
}

pub(in crate::mamba_ssm::gpu) fn observation<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &O,
    function: &CudaFunction,
    config: LaunchConfig,
    arguments: impl FnOnce() -> Arguments,
) -> Result<Option<PhysicalLaunchObservation>, String> {
    ctx.ensure_gemm_usable()?;
    let record = O::ENABLED || ctx.gemm_route_recording_active()?;
    when_recording(record, || {
        let args = arguments();
        let spec = TERMINALS
            .iter()
            .find(|spec| {
                ctx.kernels
                    .inference_terminal_function(spec.symbol)
                    .is_some_and(|loaded| std::ptr::eq(loaded, function))
            })
            .ok_or("selected function is not a loaded Inference terminal")?;
        let arithmetic = spec.arithmetic();
        let context = ctx.gemm_route();
        let (artifact, compiler) = if spec.module() == ModuleKind::Fixed {
            (
                ctx.kernels.artifact_set_identity().fixed,
                ctx.kernels.compiler_identity(),
            )
        } else {
            (
                ctx.kernels.artifact_set_identity().triad_sm80,
                ctx.kernels.triad_sm80_compiler_identity(),
            )
        };
        let mut route = ResolvedGemmRoute {
            op: ResolvedGemmOp::Nn,
            dtype: args.storage[0],
            backend: spec.backend,
            numeric_contract: arithmetic.numeric,
            instruction_family: arithmetic.instruction,
            instruction_shape: arithmetic.shape,
            operand_conversion: arithmetic.conversion,
            ownership: ResolvedOutputOwnership::OneCtaPerOutputTile,
            symbol: spec.symbol,
            module_kind: spec.module(),
            target: compiler.target,
            artifact,
            compiler,
            device: context.device,
            device_caps: context.device_caps,
            shape: args.dimensions()?,
            strides: (0, 0, 0),
            tile: spec.tile,
            bk: spec.bk,
            stages: spec.stages,
            threads: spec.threads,
            launch: ResolvedKernelLaunch {
                grid_dim: config.grid_dim,
                block_dim: config.block_dim,
                shared_mem_bytes: config.shared_mem_bytes,
                arguments_digest: [0; 32],
            },
            tensor_map_revision: u16::from(spec.uses_maps()),
            tensor_maps_digest: args.maps_digest(),
            resources_digest: [0; 32],
            tuning_table_revision: spec.tuning_revision(),
            schedule_revision: SCHEDULE_REVISION,
        };
        route.strides = (route.shape.1, route.shape.2, route.shape.2);
        args.validate(&route)?;
        route.launch.arguments_digest = args.digest(&route, None);
        ctx.validate_resolved_gemm_route(&route, "Inference terminal")?;
        ctx.record_resolved_gemm_route(route)?;
        let _ = observer;
        Ok(PhysicalLaunchObservation::inference(args, route))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba_ssm::gpu::kernel_identity::{
        ResolvedPhysicalLaunchSet, inference_test_support, resolve_physical_launch_observation,
        route_backend_contract_sets,
    };

    fn fixture(spec: &TerminalSpec, k: u32) -> (Arguments, ResolvedGemmRoute, LaunchConfig) {
        let mut route = inference_test_support::route();
        let arithmetic = spec.arithmetic();
        route.symbol = spec.symbol;
        route.backend = spec.backend;
        route.dtype = spec.storage[0];
        route.numeric_contract = arithmetic.numeric;
        route.instruction_family = arithmetic.instruction;
        route.instruction_shape = arithmetic.shape;
        route.operand_conversion = arithmetic.conversion;
        route.module_kind = spec.module();
        route.artifact.module_kind = spec.module();
        route.shape = (3, k as usize, 96);
        route.strides = (k as usize, 96, 96);
        route.tile = spec.tile;
        route.bk = spec.bk;
        route.stages = spec.stages;
        route.threads = spec.threads;
        route.tuning_table_revision = spec.tuning_revision();
        route.schedule_revision = SCHEDULE_REVISION;
        let words = match spec.abi {
            AbiKind::Legacy | AbiKind::ExactF32 | AbiKind::HalfSm89 => {
                [1f32.to_bits(), 0, 3, 96, k, k, 96, 96, 0, 0]
            }
            AbiKind::Tf32 => [3, k, 96, k, 96, 96, 0, 0, 0, 0],
            AbiKind::Tf32Wide => [1f32.to_bits(), 0, 3, k, 96, k, 96, 96, 0, 0],
            AbiKind::Sm120Tf32 => [3, k, 96, 96, 0, 0, 0, 0, 0, 0],
            AbiKind::Sm120Half => [0, 0, 0, 0, 1f32.to_bits(), 0, 3, k, 96, 96],
            AbiKind::Sm120PostBias => [1f32.to_bits(), 0, 3, 96, k, 96, 1, k.div_ceil(16), 0, 0],
        };
        let args = Arguments {
            pointers: [
                0x1000,
                if k == 0 { 0 } else { 0x10000 },
                if k == 0 { 0 } else { 0x20000 },
                if spec.bias == BiasDomain::Required {
                    0x30000
                } else {
                    0
                },
            ],
            storage: spec.storage,
            abi: spec.abi,
            words,
            maps: spec.uses_maps().then_some([[7; 16], [9; 16]]),
            auxiliary: [0; 2],
        };
        let matvec = spec.backend == PhysicalGemmBackend::FixedMatvecEightWarp;
        let config = LaunchConfig {
            grid_dim: if matvec {
                (3, 3, 1)
            } else {
                (
                    3u32.div_ceil(spec.tile.0) * 96u32.div_ceil(spec.tile.1),
                    1,
                    1,
                )
            },
            block_dim: (spec.threads, 1, 1),
            shared_mem_bytes: if matvec {
                (k * width(spec.storage[0]) as u32 + 15) & !15
            } else {
                spec.shared
            },
        };
        route.launch = ResolvedKernelLaunch {
            grid_dim: config.grid_dim,
            block_dim: config.block_dim,
            shared_mem_bytes: config.shared_mem_bytes,
            arguments_digest: [0; 32],
        };
        route.tensor_map_revision = u16::from(spec.uses_maps());
        route.tensor_maps_digest = args.maps_digest();
        route.launch.arguments_digest = args.digest(&route, None);
        (args, route, config)
    }

    fn bounded_observer(
        args: Arguments,
        shape: (usize, usize, usize),
    ) -> super::super::super::kernel_identity::RecordingPhysicalObserver {
        let spans = args.spans(shape).unwrap();
        inference_test_support::observer(move |pointer, bytes| {
            let index = args
                .pointers
                .iter()
                .position(|p| *p == pointer)
                .ok_or("unregistered argument")?;
            if bytes == 0 || bytes > spans[index] {
                return Err("argument exceeds live allocation".into());
            }
            Ok(FramedSha256::new(b"test-allocation")
                .required(b"id", &[index as u8 + 1])
                .required(b"bytes", &bytes.to_le_bytes())
                .finish())
        })
    }

    #[test]
    fn every_terminal_has_its_exact_storage_geometry_and_arithmetic_tuple() {
        // Literal census expectations from the approved vocabulary. This table
        // does not call a production selector or terminal constructor.
        type TerminalRow = (&'static str, u8, (u32, u32, u32, u8, u32, u32), [u8; 3]);
        let expected: &[TerminalRow] = &[
            ("f32_f32_s2", 31, (64, 64, 32, 2, 128, 0), [1, 1, 1]),
            ("f32_f32_n128_s2", 31, (64, 128, 32, 2, 256, 0), [1, 1, 1]),
            (
                "nn_sm89_f32_n64_copyplan",
                22,
                (64, 64, 32, 2, 128, 0),
                [1, 1, 1],
            ),
            (
                "nn_sm89_tc128_f32out_s3_bf16",
                33,
                (128, 128, 64, 3, 256, 98_304),
                [3, 3, 1],
            ),
            (
                "nn_sm89_tc128_f32out_s3_f16",
                33,
                (128, 128, 64, 3, 256, 98_304),
                [2, 2, 1],
            ),
            (
                "nn_sm89_f32_m128n64_tail_copyplan",
                31,
                (128, 64, 32, 2, 256, 0),
                [1, 1, 1],
            ),
            (
                "nn_sm120_f32_n64_copyplan",
                31,
                (64, 64, 32, 2, 128, 0),
                [1, 1, 1],
            ),
            (
                "nn_sm120_f32_n64_copyplan_t256",
                31,
                (64, 64, 32, 2, 256, 0),
                [1, 1, 1],
            ),
            (
                "nn_sm120_f32_n64_copyplan_m128n64_t256",
                31,
                (128, 64, 32, 2, 256, 0),
                [1, 1, 1],
            ),
            (
                "nn_sm120_f32_n64_sliced",
                31,
                (64, 64, 32, 2, 128, 0),
                [1, 1, 1],
            ),
            ("bf16_bf16", 32, (64, 64, 32, 1, 256, 0), [3, 3, 3]),
            ("matvec_bi_bf16_bf16", 40, (1, 32, 0, 1, 256, 0), [3, 3, 3]),
            ("bf16_f32", 32, (64, 64, 32, 1, 256, 0), [3, 3, 1]),
            ("matvec_bi_bf16_f32", 40, (1, 32, 0, 1, 256, 0), [3, 3, 1]),
            (
                "nn_tc128_bf16",
                33,
                (128, 128, 64, 2, 256, 71680),
                [3, 3, 3],
            ),
            (
                "nn_tc128_f32out_bf16",
                33,
                (128, 128, 64, 2, 256, 71680),
                [3, 3, 1],
            ),
            (
                "nn_tcw64_bf16",
                33,
                (128, 128, 64, 2, 128, 65536),
                [3, 3, 3],
            ),
            (
                "nn_tcwn64_bf16",
                33,
                (128, 256, 64, 2, 256, 98304),
                [3, 3, 3],
            ),
            ("nn_tc64_bf16", 33, (64, 64, 64, 2, 128, 0), [3, 3, 3]),
            (
                "nn_tc64_f32out_bf16",
                33,
                (64, 64, 64, 2, 128, 0),
                [3, 3, 1],
            ),
            ("nn_tc16_bf16", 33, (16, 32, 64, 4, 128, 0), [3, 3, 3]),
            (
                "nn_tc16_f32out_bf16",
                33,
                (16, 32, 64, 4, 128, 0),
                [3, 3, 1],
            ),
            (
                "nn_sm89_tc128_pipeline_bf16",
                33,
                (128, 128, 64, 2, 256, 71680),
                [3, 3, 3],
            ),
            (
                "nn_sm89_tc128_swizzle_bf16",
                33,
                (128, 128, 64, 2, 256, 69632),
                [3, 3, 3],
            ),
            (
                "nn_sm89_tc128_s3_bf16",
                33,
                (128, 128, 64, 3, 256, 98304),
                [3, 3, 3],
            ),
            (
                "nn_sm120_tma_64x64_bk64_s2_bf16",
                38,
                (64, 64, 64, 2, 128, 32896),
                [3, 3, 3],
            ),
            (
                "nn_sm120_tma_64x64_bk64_s2_f32out_bf16",
                38,
                (64, 64, 64, 2, 128, 32896),
                [3, 3, 1],
            ),
            (
                "nn_sm120_tma_64x128_bk64_s2_bf16",
                38,
                (64, 128, 64, 2, 256, 49280),
                [3, 3, 3],
            ),
            (
                "nn_sm120_tma_64x128_bk64_s2_f32out_bf16",
                38,
                (64, 128, 64, 2, 256, 49280),
                [3, 3, 1],
            ),
            (
                "nn_sm120_tma_128x64_bk32_s3_bf16",
                38,
                (128, 64, 32, 3, 256, 36992),
                [3, 3, 3],
            ),
            (
                "nn_sm120_tma_128x64_bk32_s3_f32out_bf16",
                38,
                (128, 64, 32, 3, 256, 36992),
                [3, 3, 1],
            ),
            (
                "nn_sm120_tma_128x128_bk32_s2_bf16",
                38,
                (128, 128, 32, 2, 256, 32896),
                [3, 3, 3],
            ),
            (
                "nn_sm120_tma_128x128_bk32_s2_f32out_bf16",
                38,
                (128, 128, 32, 2, 256, 32896),
                [3, 3, 1],
            ),
            (
                "nn_sm120_tma_128x128_bk32_s3_bf16",
                38,
                (128, 128, 32, 3, 256, 49280),
                [3, 3, 3],
            ),
            (
                "nn_sm120_tma_128x128_bk32_s3_f32out_bf16",
                38,
                (128, 128, 32, 3, 256, 49280),
                [3, 3, 1],
            ),
            (
                "nn_sm90a_wgmma_wg1_bf16",
                34,
                (64, 128, 64, 2, 128, 49152),
                [3, 3, 3],
            ),
            (
                "nn_sm100_tcgen_c4_bf16",
                35,
                (128, 128, 64, 2, 128, 65536),
                [3, 3, 3],
            ),
            ("f16_f16", 32, (64, 64, 32, 1, 256, 0), [2, 2, 2]),
            ("matvec_bi_f16_f16", 40, (1, 32, 0, 1, 256, 0), [2, 2, 2]),
            ("f16_f32", 32, (64, 64, 32, 1, 256, 0), [2, 2, 1]),
            ("matvec_bi_f16_f32", 40, (1, 32, 0, 1, 256, 0), [2, 2, 1]),
            ("nn_tc128_f16", 33, (128, 128, 64, 2, 256, 71680), [2, 2, 2]),
            (
                "nn_tc128_f32out_f16",
                33,
                (128, 128, 64, 2, 256, 71680),
                [2, 2, 1],
            ),
            ("nn_tcw64_f16", 33, (128, 128, 64, 2, 128, 65536), [2, 2, 2]),
            (
                "nn_tcwn64_f16",
                33,
                (128, 256, 64, 2, 256, 98304),
                [2, 2, 2],
            ),
            ("nn_tc64_f16", 33, (64, 64, 64, 2, 128, 0), [2, 2, 2]),
            ("nn_tc64_f32out_f16", 33, (64, 64, 64, 2, 128, 0), [2, 2, 1]),
            ("nn_tc16_f16", 33, (16, 32, 64, 4, 128, 0), [2, 2, 2]),
            ("nn_tc16_f32out_f16", 33, (16, 32, 64, 4, 128, 0), [2, 2, 1]),
            (
                "nn_sm89_tc128_pipeline_f16",
                33,
                (128, 128, 64, 2, 256, 71680),
                [2, 2, 2],
            ),
            (
                "nn_sm89_tc128_swizzle_f16",
                33,
                (128, 128, 64, 2, 256, 69632),
                [2, 2, 2],
            ),
            (
                "nn_sm89_tc128_s3_f16",
                33,
                (128, 128, 64, 3, 256, 98304),
                [2, 2, 2],
            ),
            (
                "nn_sm120_tma_64x64_bk64_s2_f16",
                38,
                (64, 64, 64, 2, 128, 32896),
                [2, 2, 2],
            ),
            (
                "nn_sm120_tma_64x64_bk64_s2_f32out_f16",
                38,
                (64, 64, 64, 2, 128, 32896),
                [2, 2, 1],
            ),
            (
                "nn_sm120_tma_64x128_bk64_s2_f16",
                38,
                (64, 128, 64, 2, 256, 49280),
                [2, 2, 2],
            ),
            (
                "nn_sm120_tma_64x128_bk64_s2_f32out_f16",
                38,
                (64, 128, 64, 2, 256, 49280),
                [2, 2, 1],
            ),
            (
                "nn_sm120_tma_128x64_bk32_s3_f16",
                38,
                (128, 64, 32, 3, 256, 36992),
                [2, 2, 2],
            ),
            (
                "nn_sm120_tma_128x64_bk32_s3_f32out_f16",
                38,
                (128, 64, 32, 3, 256, 36992),
                [2, 2, 1],
            ),
            (
                "nn_sm120_tma_128x128_bk32_s2_f16",
                38,
                (128, 128, 32, 2, 256, 32896),
                [2, 2, 2],
            ),
            (
                "nn_sm120_tma_128x128_bk32_s2_f32out_f16",
                38,
                (128, 128, 32, 2, 256, 32896),
                [2, 2, 1],
            ),
            (
                "nn_sm120_tma_128x128_bk32_s3_f16",
                38,
                (128, 128, 32, 3, 256, 49280),
                [2, 2, 2],
            ),
            (
                "nn_sm120_tma_128x128_bk32_s3_f32out_f16",
                38,
                (128, 128, 32, 3, 256, 49280),
                [2, 2, 1],
            ),
            (
                "nn_sm90a_wgmma_wg1_f16",
                34,
                (64, 128, 64, 2, 128, 49152),
                [2, 2, 2],
            ),
            (
                "nn_sm100_tcgen_c4_f16",
                35,
                (128, 128, 64, 2, 128, 65536),
                [2, 2, 2],
            ),
            ("matvec_bi_f32_f32", 40, (1, 32, 0, 1, 256, 0), [1, 1, 1]),
            (
                "nn_sm89_m64n64_bk64_s3_f16",
                33,
                (64, 64, 64, 3, 128, 49152),
                [2, 2, 2],
            ),
            (
                "nn_sm89_m128n64_bk64_s2_f16",
                33,
                (128, 64, 64, 2, 128, 49152),
                [2, 2, 2],
            ),
            (
                "nn_tf32_m128n64_bk32_s2",
                36,
                (128, 64, 32, 2, 256, 55296),
                [1, 1, 1],
            ),
            (
                "nn_tf32_m128n64_bk32_s3",
                36,
                (128, 64, 32, 3, 256, 82944),
                [1, 1, 1],
            ),
            (
                "nn_tf32_m64n64_bk32_s2",
                36,
                (64, 64, 32, 2, 128, 32768),
                [1, 1, 1],
            ),
            (
                "nn_tf32_m64n64_bk32_s3",
                36,
                (64, 64, 32, 3, 128, 55296),
                [1, 1, 1],
            ),
            (
                "nn_tf32_m16n32_bk32_s4",
                36,
                (16, 32, 32, 4, 128, 29696),
                [1, 1, 1],
            ),
            (
                "nn_rna_wide_tf32_m128n128_bk32_s3",
                36,
                (128, 128, 32, 3, 256, 98304),
                [1, 1, 1],
            ),
            (
                "nn_sm89_rna_tf32_m128n96_bk32_s3",
                36,
                (128, 96, 32, 3, 256, 86016),
                [1, 1, 1],
            ),
            (
                "nn_sm80_mma_tf32_m128n128_bk32_s3",
                5,
                (128, 128, 32, 3, 256, 98304),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_tf32_m128n64_bk32_s2",
                39,
                (128, 64, 32, 2, 128, 49280),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_tf32_m128n64_bk32_s3",
                39,
                (128, 64, 32, 3, 256, 73856),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_tf32_m64n128_bk32_s2",
                39,
                (64, 128, 32, 2, 128, 49280),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_tf32_m64n128_bk32_s3",
                39,
                (64, 128, 32, 3, 256, 73856),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_tf32_m64n64_bk32_s2_producer_warp",
                39,
                (64, 64, 32, 2, 160, 32896),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_tf32_m64n64_bk32_s2",
                39,
                (64, 64, 32, 2, 128, 32896),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store",
                39,
                (64, 64, 32, 2, 128, 32896),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_fma_postbias_m128n64_bk16_s2",
                37,
                (128, 64, 16, 2, 128, 24592),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_fma_postbias_m64n128_bk16_s2",
                37,
                (64, 128, 16, 2, 128, 24592),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_fma_postbias_m128n96_bk16_s2",
                37,
                (128, 96, 16, 2, 256, 28688),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4",
                37,
                (128, 64, 16, 2, 128, 24592),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2",
                37,
                (128, 64, 16, 2, 256, 24592),
                [1, 1, 1],
            ),
            (
                "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2",
                37,
                (128, 64, 16, 2, 256, 24592),
                [1, 1, 1],
            ),
        ];
        assert_eq!(TERMINALS.len(), 88);
        assert_eq!(expected.len(), TERMINALS.len());
        let mut seen = std::collections::BTreeSet::new();
        for &(symbol, backend, geometry, storage) in expected {
            assert!(seen.insert(symbol));
            let spec = terminal(symbol).unwrap();
            assert_eq!(spec.backend as u8, backend, "{symbol}");
            assert_eq!(
                (
                    spec.tile.0,
                    spec.tile.1,
                    spec.bk,
                    spec.stages,
                    spec.threads,
                    spec.shared
                ),
                geometry,
                "{symbol}"
            );
            assert_eq!(spec.storage.map(|d| d as u8), storage, "{symbol}");
            let arithmetic = spec.arithmetic();
            let expected_arithmetic = match backend {
                22 | 31 | 37 => (25, 1, (1, 1, 1), 0),
                32 => (26, 5, (16, 16, 16), 0),
                33 | 38 => (2, 2, (16, 8, 16), 0),
                34 => (3, 3, (64, 128, 16), 0),
                35 => (4, 4, (128, 128, 16), 0),
                36 => (5, 2, (16, 8, 8), 1),
                39 => (8, 2, (16, 8, 8), 3),
                40 => (27, 1, (1, 1, 1), 0),
                5 => (23, 2, (16, 8, 8), 4),
                _ => panic!("unexpected backend"),
            };
            assert_eq!(
                (
                    arithmetic.numeric as u8,
                    arithmetic.instruction as u8,
                    (arithmetic.shape.m, arithmetic.shape.n, arithmetic.shape.k),
                    arithmetic.conversion as u8
                ),
                expected_arithmetic,
                "{symbol}"
            );
            let (args, route, config) = fixture(spec, 64);
            args.validate(&route).unwrap();
            assert_eq!(
                route.module_kind,
                if backend == 5 {
                    ModuleKind::TriadSm80
                } else {
                    ModuleKind::Fixed
                }
            );
            assert_eq!(
                route.tuning_table_revision,
                if backend == 22 { 1 } else { 45 }
            );
            assert_eq!(route.schedule_revision, 8);
            let observer = bounded_observer(args, route.shape);
            let node = resolve_physical_launch_observation(
                &observer,
                PhysicalLaunchObservation::inference(args, route),
                config,
            )
            .unwrap();
            assert_ne!(node.launch.arguments_digest, route.launch.arguments_digest);
            assert_eq!(
                node.launch.arguments_digest,
                node.gemm_route().unwrap().launch.arguments_digest
            );
            ResolvedPhysicalLaunchSet::from_nodes(&[node]).unwrap();
        }
        assert!(terminal("f32_f32").is_none());
        assert!(terminal("nn_sm90a_wgmma_wg1_f32out_bf16").is_none());
        assert!(terminal("unrelated_f32out_bf16").is_none());
    }

    #[test]
    fn inference_bundle_admission_has_three_independent_closed_terminal_identities() {
        let cases = [
            (
                "nn_sm89_tc128_f32out_s3_bf16",
                PolicyDtype::Bf16,
                PhysicalGemmBackend::InferenceMma16,
                AbiKind::HalfSm89,
                (128, 128, 64, 3, 256, 98_304),
                BiasDomain::Optional,
            ),
            (
                "nn_sm89_tc128_f32out_s3_f16",
                PolicyDtype::F16,
                PhysicalGemmBackend::InferenceMma16,
                AbiKind::HalfSm89,
                (128, 128, 64, 3, 256, 98_304),
                BiasDomain::Optional,
            ),
            (
                "nn_sm89_f32_m128n64_tail_copyplan",
                PolicyDtype::F32,
                PhysicalGemmBackend::InferenceScalarFma,
                AbiKind::ExactF32,
                (128, 64, 32, 2, 256, 0),
                BiasDomain::Forbidden,
            ),
        ];
        for (symbol, input, backend, abi, geometry, bias) in cases {
            let spec = terminal(symbol).unwrap();
            assert_eq!(spec.storage, [input, input, PolicyDtype::F32], "{symbol}");
            assert_eq!(spec.backend, backend, "{symbol}");
            assert_eq!(spec.abi, abi, "{symbol}");
            assert_eq!(
                (
                    spec.tile.0,
                    spec.tile.1,
                    spec.bk,
                    spec.stages,
                    spec.threads,
                    spec.shared
                ),
                geometry,
                "{symbol}"
            );
            assert_eq!(spec.bias, bias, "{symbol}");
        }
    }

    #[test]
    fn every_terminal_rejects_individually_changed_route_and_abi_before_enqueue() {
        for spec in TERMINALS {
            let (args, route, config) = fixture(spec, 64);
            let observer = bounded_observer(args, route.shape);
            let mut route_mutations = Vec::new();
            let mut value = route;
            value.dtype = if route.dtype == PolicyDtype::F32 {
                PolicyDtype::F16
            } else {
                PolicyDtype::F32
            };
            route_mutations.push(value);
            let mut value = route;
            value.backend = PhysicalGemmBackend::ScalarFma;
            route_mutations.push(value);
            let mut value = route;
            value.numeric_contract = ResolvedNumericContract::ScalarFma;
            route_mutations.push(value);
            let mut value = route;
            value.module_kind = ModuleKind::TriadScalar;
            route_mutations.push(value);
            let mut value = route;
            value.instruction_shape.k += 1;
            route_mutations.push(value);
            let mut value = route;
            value.operand_conversion = ResolvedOperandConversion::TensorMapTfloat32;
            route_mutations.push(value);
            let mut value = route;
            value.tile.1 += 1;
            route_mutations.push(value);
            let mut value = route;
            value.bk += 1;
            route_mutations.push(value);
            let mut value = route;
            value.stages += 1;
            route_mutations.push(value);
            let mut value = route;
            value.launch.grid_dim.0 += 1;
            route_mutations.push(value);
            let mut value = route;
            value.launch.block_dim.0 += 1;
            route_mutations.push(value);
            let mut value = route;
            value.launch.shared_mem_bytes += 16;
            route_mutations.push(value);
            let mut value = route;
            value.strides.1 += 1;
            route_mutations.push(value);
            let mut value = route;
            value.tuning_table_revision += 1;
            route_mutations.push(value);
            let mut value = route;
            value.schedule_revision += 1;
            route_mutations.push(value);
            let mut value = route;
            value.symbol = if route.symbol == "f32_f32_s2" {
                "f32_f32_n128_s2"
            } else {
                "f32_f32_s2"
            };
            route_mutations.push(value);
            for changed in route_mutations {
                let mut enqueued = false;
                let result = resolve_physical_launch_observation(
                    &observer,
                    PhysicalLaunchObservation::inference(args, changed),
                    config,
                )
                .map(|_| {
                    enqueued = true;
                });
                assert!(
                    result.is_err() && !enqueued,
                    "route mutation accepted: {}",
                    spec.symbol
                );
            }
            let mut argument_mutations = Vec::new();
            for field in 0..10 {
                let mut value = args;
                value.words[field] ^= 1;
                argument_mutations.push(value);
            }
            for field in 0..3 {
                let mut value = args;
                value.storage[field] = if value.storage[field] == PolicyDtype::F32 {
                    PolicyDtype::Bf16
                } else {
                    PolicyDtype::F32
                };
                argument_mutations.push(value);
            }
            for field in 0..3 {
                let mut value = args;
                value.pointers[field] = 0;
                argument_mutations.push(value);
            }
            let mut value = args;
            value.pointers[0] = u64::MAX - 1;
            argument_mutations.push(value);
            let mut value = args;
            value.pointers[3] ^= 0x30000;
            argument_mutations.push(value);
            let mut value = args;
            value.auxiliary[0] = 1;
            argument_mutations.push(value);
            let mut value = args;
            value.auxiliary[1] = 1;
            argument_mutations.push(value);
            let mut value = args;
            value.abi = if args.abi == AbiKind::Legacy {
                AbiKind::Tf32
            } else {
                AbiKind::Legacy
            };
            argument_mutations.push(value);
            if let Some(maps) = args.maps {
                for map in 0..2 {
                    for field in 0..16 {
                        let mut value = args;
                        let mut changed = maps;
                        changed[map][field] ^= 1;
                        value.maps = Some(changed);
                        argument_mutations.push(value);
                    }
                }
                let mut value = args;
                value.maps = None;
                argument_mutations.push(value);
            } else {
                let mut value = args;
                value.maps = Some([[0; 16]; 2]);
                argument_mutations.push(value);
            }
            for changed in argument_mutations {
                let mut enqueued = false;
                let result = resolve_physical_launch_observation(
                    &observer,
                    PhysicalLaunchObservation::inference(changed, route),
                    config,
                )
                .map(|_| {
                    enqueued = true;
                });
                assert!(
                    result.is_err() && !enqueued,
                    "argument mutation accepted: {}",
                    spec.symbol
                );
            }
        }
    }

    #[test]
    fn pair_store_symbol_is_bound_even_when_geometry_is_identical() {
        let regular = terminal("nn_sm120_tma_tf32_m64n64_bk32_s2").unwrap();
        let pair = terminal("nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store").unwrap();
        let (args, route, config) = fixture(regular, 64);
        let (pair_args, pair_route, _) = fixture(pair, 64);
        assert_eq!(route.launch.grid_dim, pair_route.launch.grid_dim);
        assert_eq!(route.launch.block_dim, pair_route.launch.block_dim);
        assert_ne!(
            args.digest(&route, None),
            pair_args.digest(&pair_route, None)
        );
        let mut changed = route;
        changed.symbol = pair.symbol;
        let mut enqueued = false;
        let result = resolve_physical_launch_observation(
            &bounded_observer(args, route.shape),
            PhysicalLaunchObservation::inference(args, changed),
            config,
        )
        .map(|_| enqueued = true);
        assert!(result.is_err() && !enqueued);
    }

    #[test]
    fn cached_bridge_preserves_only_its_two_existing_triad_specs() {
        for (symbol, tile) in [
            ("nn_sm120_tma_fma_m128n64_bk16_s2", (128, 64)),
            ("nn_sm120_tma_fma_m64n128_bk16_s2", (64, 128)),
        ] {
            let spec =
                crate::mamba_ssm::gpu::gemm_bi_triad::tf32_route_specs_all(ModuleKind::TriadSm120)
                    .find(|spec| spec.symbol == symbol)
                    .unwrap();
            let mut route = inference_test_support::route();
            route.symbol = symbol;
            route.backend = PhysicalGemmBackend::Sm120TmaFmaExact;
            route.module_kind = ModuleKind::TriadSm120;
            route.artifact.module_kind = ModuleKind::TriadSm120;
            route.shape = (2048, 768, 2304);
            route.strides = (768, 2304, 2304);
            route.tile = tile;
            route.bk = 16;
            route.stages = 2;
            route.threads = 128;
            route.launch.grid_dim = (2048u32.div_ceil(tile.0) * 2304u32.div_ceil(tile.1), 1, 1);
            route.launch.block_dim = (128, 1, 1);
            route.launch.shared_mem_bytes = spec.dynamic_shared_bytes;
            route.tensor_map_revision = spec.tensor_map_revision;
            route.tensor_maps_digest = [3; 32];
            route.schedule_revision = spec.schedule_revision;
            validate_cached_bridge(&route).unwrap();
            let mut bad = route;
            bad.symbol = "nn_sm120_tma_fma_m64n64_bk16_s2";
            assert!(validate_cached_bridge(&bad).is_err());
            let mut bad = route;
            bad.numeric_contract = ResolvedNumericContract::ScalarFmaPostDotBias;
            assert!(validate_cached_bridge(&bad).is_err());
            let mut bad = route;
            bad.ownership = ResolvedOutputOwnership::OwnerCtaPerOutputTileFixedSplitFold;
            assert!(validate_cached_bridge(&bad).is_err());
            let mut bad = route;
            bad.module_kind = ModuleKind::Fixed;
            assert!(validate_cached_bridge(&bad).is_err());
            let mut bad = route;
            bad.launch.grid_dim.0 += 1;
            assert!(validate_cached_bridge(&bad).is_err());
            assert!(
                terminal(symbol).is_none(),
                "bridge must not get a Fixed alias"
            );
        }
    }

    #[test]
    fn allocation_spans_and_zero_reduction_use_actual_required_operands() {
        let spec = terminal("f32_f32_s2").unwrap();
        let (args, route, config) = fixture(spec, 64);
        for failing in [0x1000, 0x10000, 0x20000] {
            let observer = inference_test_support::observer(move |pointer, _| {
                if pointer == failing {
                    Err("insufficient allocation span".into())
                } else {
                    Ok([1; 32])
                }
            });
            let mut enqueued = false;
            let result = resolve_physical_launch_observation(
                &observer,
                PhysicalLaunchObservation::inference(args, route),
                config,
            )
            .map(|_| enqueued = true);
            assert!(result.is_err() && !enqueued);
        }
        let mut with_bias = args;
        with_bias.pointers[3] = 0x30000;
        let mut bias_route = route;
        bias_route.launch.arguments_digest = with_bias.digest(&bias_route, None);
        let observer = inference_test_support::observer(|pointer, _| {
            if pointer == 0x30000 {
                Err("insufficient bias allocation".into())
            } else {
                Ok([1; 32])
            }
        });
        assert!(
            resolve_physical_launch_observation(
                &observer,
                PhysicalLaunchObservation::inference(with_bias, bias_route),
                config
            )
            .is_err()
        );
        assert!(args.spans((usize::MAX, 64, 96)).is_err());
        assert!(args.spans((3, usize::MAX, 96)).is_err());
        let (mut zero, mut route, config) = fixture(spec, 0);
        for pointers in [[0, 0], [0x10000, 0x20000], [0, 0x20000]] {
            zero.pointers[1] = pointers[0];
            zero.pointers[2] = pointers[1];
            route.launch.arguments_digest = zero.digest(&route, None);
            let observer = inference_test_support::observer(|pointer, bytes| {
                assert_eq!(pointer, 0x1000, "K0 attempted to resolve an unused input");
                assert_eq!(bytes, 3 * 96 * 4);
                Ok([5; 32])
            });
            resolve_physical_launch_observation(
                &observer,
                PhysicalLaunchObservation::inference(zero, route),
                config,
            )
            .unwrap();
        }
    }

    #[test]
    fn allocation_bound_digest_updates_the_contained_route_and_rejects_stale_copy() {
        let (args, route, config) = fixture(terminal("bf16_f32").unwrap(), 64);
        let observer = bounded_observer(args, route.shape);
        let node = resolve_physical_launch_observation(
            &observer,
            PhysicalLaunchObservation::inference(args, route),
            config,
        )
        .unwrap();
        assert_eq!(
            node.gemm_route().unwrap().launch.arguments_digest,
            node.launch.arguments_digest
        );
        let mut stale = node;
        stale.launch.arguments_digest = route.launch.arguments_digest;
        assert!(ResolvedPhysicalLaunchSet::from_nodes(&[stale]).is_err());
        let mut wrong_logical = node;
        wrong_logical.logical_dtype = PolicyDtype::F32;
        assert!(ResolvedPhysicalLaunchSet::from_nodes(&[wrong_logical]).is_err());
    }

    #[test]
    fn disabled_recording_never_builds_metadata() {
        let called = std::cell::Cell::new(false);
        assert_eq!(
            when_recording(false, || {
                called.set(true);
                Ok(7)
            })
            .unwrap(),
            None
        );
        assert!(!called.get());
        assert_eq!(
            when_recording(true, || {
                called.set(true);
                Ok(7)
            })
            .unwrap(),
            Some(7)
        );
        assert!(called.get());
    }

    #[test]
    fn terminal_permissions_follow_live_family_and_not_the_triad_half_switch() {
        let mut policy = super::super::super::kernel_identity::GemmPolicy {
            batch_invariant: true,
            bi_tensor_cores: true,
            fast_gemm: false,
            cublas_tf32: false,
            f32_triad_policy: super::super::super::context::F32TriadPolicy::ExactScalarFma,
            half_triad_policy: super::super::super::context::HalfTriadPolicy::TiledParity,
            bi_gemm_family: BiGemmFamily::Inference,
        };
        for family in [BiGemmFamily::Inference, BiGemmFamily::Triad] {
            for tc in [false, true] {
                for allow_tf32 in [false, true] {
                    policy.bi_gemm_family = family;
                    policy.bi_tensor_cores = tc;
                    policy.f32_triad_policy = if allow_tf32 {
                        super::super::super::context::F32TriadPolicy::AllowDeterministicTf32
                    } else {
                        super::super::super::context::F32TriadPolicy::ExactScalarFma
                    };
                    let (_, contracts) = route_backend_contract_sets(policy);
                    for spec in TERMINALS {
                        let expected_family =
                            if spec.backend == PhysicalGemmBackend::FixedMatvecEightWarp {
                                BiGemmFamily::Triad
                            } else {
                                BiGemmFamily::Inference
                            };
                        let required = spec.required_contract(family);
                        if family != expected_family {
                            assert!(required.is_err());
                            continue;
                        }
                        let required = required.unwrap();
                        let tf32 = matches!(
                            spec.backend,
                            PhysicalGemmBackend::InferenceMmaTf32Rna
                                | PhysicalGemmBackend::InferenceSm120TmaMmaTf32Rna
                                | PhysicalGemmBackend::MmaTf32Rna
                        );
                        assert_eq!(
                            contracts.contains(required),
                            !tf32 || allow_tf32,
                            "{}",
                            spec.symbol
                        );
                    }
                    if family == BiGemmFamily::Inference {
                        assert_eq!(
                            contracts.contains(NumericContractSet::FIXED_DETERMINISTIC_TF32),
                            allow_tf32
                        );
                        assert!(!contracts.contains(NumericContractSet::TRIAD_DETERMINISTIC_TF32));
                    } else {
                        assert_eq!(contracts.contains(NumericContractSet::TRIAD_MMA_SYNC), tc);
                    }
                }
            }
        }
    }
}
