#![cfg(feature = "cuda")]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_int, c_void};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use common::gpu_quiet::QuietGpu;
use cudarc::driver::{CudaGraph, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    F32TriadShape, PhysicalPaddedNnLayout, PhysicalQualificationOffset,
    PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedGuardValidation,
    QualifiedPhysicalLaunch, QualifiedPhysicalLaunchEvidence, QualifiedPhysicalLaunchNode,
    SM80_TF32_ROUTE_SPECS, TcTile, Tf32KernelSpec, Tf32PhysicalRoute, Tf32PortableRoute,
    Tf32PortableStages, Tf32PortableTile, presize_physical_qualification_suite,
    qualify_physical_launch, tf32_route_specs,
};
#[cfg(feature = "cuda-cublaslt-qualification")]
use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
    Sm120Bk, Sm120ForcedRoute, Sm120LaunchOperands, Sm120MapRequest, Sm120Op, Sm120PhysicalRoute,
    Sm120PreparedLaunch, Sm120Shape, Sm120Stages, Sm120Tile, launch_sm120_tma_prepared,
    prepare_sm120_tensor_maps, prepare_sm120_tma_forced,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    FramedSha256, ModuleKind, PhysicalLaunchKind, PolicyDtype, ResolvedGemmOp, digest_hex,
};
use sha2::{Digest as _, Sha256};

const DEFAULT_WINDOWS: usize = 101;
const TARGET_WINDOW_MS: f64 = 5.0;
const FROZEN_ITERATION_FORMAT: &str = "gemm-bi-frozen-iterations.v1";
const FROZEN_ITERATION_SUITE: &str = "gemm_bi_canonical_performance";
const EDGE_FROZEN_ITERATION_SUITE: &str = "gemm_bi_deterministic_performance_edges";
const CANONICAL_INVENTORY_COUNT: usize = 652;
const CANONICAL_INVENTORY_DIGEST: &str =
    "cdfe923e3b00710e2cd52a9e942961b64d7f9d7b6b9ff24eebb202858526ac5d";
const FROZEN_ITERATION_HEADER: &str = "cell_id\teager_iterations\tgraph_iterations";
const QUALIFICATION_CELL_IDS_ENV: &str = "GEMM_BI_QUAL_CELL_IDS";
const EDGE_QUALIFICATION_CELL_IDS_ENV: &str = "GEMM_BI_EDGE_QUAL_CELL_IDS";
const CUBLAS_DENOMINATOR_CELL_IDS_ENV: &str = "GEMM_BI_CUBLAS_CELL_IDS";
const EDGE_INVENTORY_COUNT: usize = 533;
const EDGE_INVENTORY_DIGEST: &str =
    "d7f2e9d705b5a25e695059046ef73faf0f4d43ef3b187d70456813668c19be29";
const UNSUPPORTED_PADDED_INVENTORY_COUNT: usize = 16;
const UNSUPPORTED_PADDED_INVENTORY_DIGEST: &str =
    "4884091a6f0f85c207185f89a8ae0ba1ea0c55cbc0b08eda12c4c643fd415e8d";
const TF32_TOURNAMENT_OUTPUT_ENV: &str = "MAMBA_RS_TF32_NN_TOURNAMENT_JSONL";

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    dims: (usize, usize, usize),
}

const SHAPES: [Shape; 17] = [
    Shape {
        name: "sq64",
        dims: (64, 64, 64),
    },
    Shape {
        name: "underfill",
        dims: (256, 512, 384),
    },
    Shape {
        name: "large",
        dims: (2048, 3072, 768),
    },
    Shape {
        name: "large_deep",
        dims: (4096, 3072, 1536),
    },
    Shape {
        name: "rect_wide",
        dims: (512, 3072, 768),
    },
    Shape {
        name: "rect_tall",
        dims: (4096, 512, 768),
    },
    Shape {
        name: "thin_rows",
        dims: (16, 512, 2048),
    },
    Shape {
        name: "thin_cols",
        dims: (512, 16, 2048),
    },
    Shape {
        name: "thin_rows_tail",
        dims: (49, 65, 129),
    },
    Shape {
        name: "thin_cols_tail",
        dims: (65, 49, 129),
    },
    Shape {
        name: "all_tail",
        dims: (129, 131, 100),
    },
    Shape {
        name: "split_candidate",
        dims: (128, 8192, 128),
    },
    Shape {
        name: "d128_in_proj",
        dims: (1024, 128, 512),
    },
    Shape {
        name: "d128_out_proj",
        dims: (1024, 256, 128),
    },
    Shape {
        name: "d768_in_proj",
        dims: (2048, 768, 3072),
    },
    Shape {
        name: "d768_out_proj",
        dims: (2048, 1536, 768),
    },
    Shape {
        name: "prism_in_proj",
        dims: (4621, 384, 1928),
    },
];

const FORCED_SQUARE_SHAPES: [&str; 13] = [
    "underfill",
    "large",
    "large_deep",
    "rect_wide",
    "rect_tall",
    "all_tail",
    "thin_rows",
    "thin_cols",
    "d128_in_proj",
    "d128_out_proj",
    "d768_in_proj",
    "d768_out_proj",
    "prism_in_proj",
];
const FORCED_THIN_SHAPES: [&str; 5] = [
    "sq64",
    "underfill",
    "thin_rows",
    "thin_rows_tail",
    "all_tail",
];
const TF32_PROJECTION_SHAPES: [&str; 5] = [
    "d128_in_proj",
    "d128_out_proj",
    "d768_in_proj",
    "d768_out_proj",
    "prism_in_proj",
];
const SM120_TF32_HOT_SHAPES: [&str; 5] = [
    "large",
    "large_deep",
    "d768_in_proj",
    "d768_out_proj",
    "prism_in_proj",
];

#[derive(Clone, Copy)]
enum Route {
    F32Policy {
        policy: F32TriadPolicy,
    },
    HalfPolicy {
        dtype: WeightDtype,
        tensor_cores: bool,
    },
    HalfForced {
        dtype: WeightDtype,
        tile: TcTile,
    },
    Tf32Forced(&'static Tf32KernelSpec),
}

#[derive(Clone, Copy)]
struct Cell {
    route: Route,
    op: ResolvedGemmOp,
    shape: Shape,
    stride_class: StrideClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum EdgeCohort {
    Boundary,
    DeepK,
    Offset,
    DirectPaddedNn,
}

impl EdgeCohort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Boundary => "boundary",
            Self::DeepK => "deep_k",
            Self::Offset => "offset",
            Self::DirectPaddedNn => "padded_nn",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ElementOffsets {
    output: usize,
    a: usize,
    b: usize,
}

impl ElementOffsets {
    const ALIGNED: Self = Self {
        output: 0,
        a: 0,
        b: 0,
    };

    fn label(self) -> String {
        match (self.output, self.a, self.b) {
            (0, 1, 0) => "a_offset_1".into(),
            (0, 0, 1) => "b_offset_1".into(),
            (1, 0, 0) => "output_offset_1".into(),
            _ => format!("a{}_b{}_output{}", self.a, self.b, self.output),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PaddedNnLayout {
    label: &'static str,
    strides: (usize, usize, usize),
    offsets: ElementOffsets,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeLayout {
    Contiguous,
    Offset(ElementOffsets),
    DirectPaddedNn(PaddedNnLayout),
}

impl EdgeLayout {
    fn label(self) -> String {
        match self {
            Self::Contiguous => "contiguous".into(),
            Self::Offset(offsets) => offsets.label(),
            Self::DirectPaddedNn(layout) => layout.label.into(),
        }
    }
}

#[derive(Clone, Copy)]
struct EdgeCell {
    cohort: EdgeCohort,
    route: Route,
    op: ResolvedGemmOp,
    shape: Shape,
    layout: EdgeLayout,
}

#[derive(Clone, Copy)]
struct UnsupportedPaddedDescriptor {
    contract: &'static str,
    route: Route,
    op: ResolvedGemmOp,
    shape: Shape,
    layout: PaddedNnLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StrideClass {
    Contiguous,
}

impl StrideClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
        }
    }
}

struct Samples {
    values_us: Vec<f64>,
    iterations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CublasDenominatorMode {
    Fast,
    Pedantic,
}

impl CublasDenominatorMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Pedantic => "pedantic",
        }
    }
}

#[derive(Clone, Copy)]
struct CublasDenominatorCell {
    dtype: WeightDtype,
    op: ResolvedGemmOp,
    shape: Shape,
}

#[derive(Clone, Copy)]
struct CublasDenominatorGeometry {
    trans_a: cudarc::cublas::sys::cublasOperation_t,
    trans_b: cudarc::cublas::sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    lda: c_int,
    ldb: c_int,
    ldc: c_int,
}

struct CublasDenominatorBuffers {
    output: DtypedBuf,
    a: DtypedBuf,
    b: DtypedBuf,
}

#[derive(Clone, Copy)]
struct CublasDenominatorIterations {
    fast: usize,
    pedantic: usize,
}

#[cfg(feature = "cuda-cublaslt-qualification")]
mod cublaslt_qualification {
    use super::*;

    #[derive(Clone, Copy)]
    pub(super) struct Sm120PairedNnCell {
        pub(super) dtype: WeightDtype,
        pub(super) shape: Shape,
        pub(super) physical: Sm120PhysicalRoute,
        pub(super) alpha: f32,
        pub(super) beta: f32,
        pub(super) bias_present: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct LtMatrixLayoutSpec {
        pub(super) dtype: cudarc::cublaslt::sys::cudaDataType,
        pub(super) rows: u64,
        pub(super) columns: u64,
        pub(super) ld: i64,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct LtNnDescriptorSpec {
        pub(super) compute: cudarc::cublaslt::sys::cublasComputeType_t,
        pub(super) scale_dtype: cudarc::cublaslt::sys::cudaDataType,
        pub(super) epilogue: cudarc::cublaslt::sys::cublasLtEpilogue_t,
        pub(super) a: LtMatrixLayoutSpec,
        pub(super) b: LtMatrixLayoutSpec,
        pub(super) output: LtMatrixLayoutSpec,
        pub(super) alpha: f32,
        pub(super) beta: f32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum ComparatorOrder {
        CustomThenCublas,
        CublasThenCustom,
    }

    impl ComparatorOrder {
        pub(super) fn as_str(self) -> &'static str {
            match self {
                Self::CustomThenCublas => "custom_then_cublas",
                Self::CublasThenCustom => "cublas_then_custom",
            }
        }
    }

    pub(super) fn build_sm120_paired_nn_cells() -> Vec<Sm120PairedNnCell> {
        let mut cells = Vec::with_capacity(6);
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for shape in [
                Shape {
                    name: "large",
                    dims: (2048, 3072, 768),
                },
                Shape {
                    name: "large_deep",
                    dims: (4096, 3072, 1536),
                },
                Shape {
                    name: "d768_out_proj",
                    dims: (2048, 1536, 768),
                },
            ] {
                let physical = if shape.name == "large_deep" {
                    Sm120PhysicalRoute {
                        tile: Sm120Tile::M128N64,
                        bk: Sm120Bk::Bk32,
                        stages: Sm120Stages::S2,
                    }
                } else {
                    Sm120PhysicalRoute {
                        tile: Sm120Tile::M64N64,
                        bk: Sm120Bk::Bk64,
                        stages: Sm120Stages::S2,
                    }
                };
                cells.push(Sm120PairedNnCell {
                    dtype,
                    shape,
                    physical,
                    alpha: 1.0,
                    beta: 0.0,
                    bias_present: false,
                });
            }
        }
        cells
    }

    pub(super) fn sm120_paired_nn_cell_id(cell: Sm120PairedNnCell) -> &'static str {
        match (cell.dtype, cell.shape.name) {
            (WeightDtype::Bf16, "large") => "sm120_paired/bf16/nn/large",
            (WeightDtype::Bf16, "large_deep") => "sm120_paired/bf16/nn/large_deep",
            (WeightDtype::Bf16, "d768_out_proj") => "sm120_paired/bf16/nn/d768_out_proj",
            (WeightDtype::F16, "large") => "sm120_paired/f16/nn/large",
            (WeightDtype::F16, "large_deep") => "sm120_paired/f16/nn/large_deep",
            (WeightDtype::F16, "d768_out_proj") => "sm120_paired/f16/nn/d768_out_proj",
            _ => "sm120_paired/unsupported",
        }
    }

    pub(super) fn sm120_paired_nn_lt_descriptor(
        cell: Sm120PairedNnCell,
    ) -> Result<LtNnDescriptorSpec, String> {
        use cudarc::cublaslt::sys::{cublasComputeType_t, cublasLtEpilogue_t, cudaDataType_t};

        if cell.alpha.to_bits() != 1.0f32.to_bits()
            || cell.beta.to_bits() != 0.0f32.to_bits()
            || cell.bias_present
        {
            return Err("SM120 paired NN requires alpha=1, beta=0, and no bias".into());
        }
        let dtype = match cell.dtype {
            WeightDtype::Bf16 => cudaDataType_t::CUDA_R_16BF,
            WeightDtype::F16 => cudaDataType_t::CUDA_R_16F,
            WeightDtype::F32 => return Err("SM120 half NN pairing rejects F32 storage".into()),
        };
        let (m, k, n) = cell.shape.dims;
        let as_u64 =
            |value| u64::try_from(value).map_err(|_| "Lt dimension exceeds u64".to_owned());
        let as_i64 = |value| i64::try_from(value).map_err(|_| "Lt stride exceeds i64".to_owned());
        Ok(LtNnDescriptorSpec {
            compute: cublasComputeType_t::CUBLAS_COMPUTE_32F,
            scale_dtype: cudaDataType_t::CUDA_R_32F,
            epilogue: cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            a: LtMatrixLayoutSpec {
                dtype,
                rows: as_u64(n)?,
                columns: as_u64(k)?,
                ld: as_i64(n)?,
            },
            b: LtMatrixLayoutSpec {
                dtype,
                rows: as_u64(k)?,
                columns: as_u64(m)?,
                ld: as_i64(k)?,
            },
            output: LtMatrixLayoutSpec {
                dtype,
                rows: as_u64(n)?,
                columns: as_u64(m)?,
                ld: as_i64(n)?,
            },
            alpha: cell.alpha,
            beta: cell.beta,
        })
    }

    pub(super) fn sm120_paired_comparator_schedule() -> Vec<ComparatorOrder> {
        let mut schedule = Vec::with_capacity(202);
        for _ in 0..101 {
            schedule.push(ComparatorOrder::CustomThenCublas);
            schedule.push(ComparatorOrder::CublasThenCustom);
        }
        schedule
    }

    pub(super) fn sm120_paired_ratios(custom: &[f64], cublas: &[f64]) -> Result<Vec<f64>, String> {
        if custom.len() != 101 || cublas.len() != 101 {
            return Err("SM120 pairing requires exactly 101 samples per arm".into());
        }
        custom
            .iter()
            .zip(cublas)
            .enumerate()
            .map(|(index, (&custom, &cublas))| {
                let ratio = custom / cublas;
                if !custom.is_finite()
                    || custom <= 0.0
                    || !cublas.is_finite()
                    || cublas <= 0.0
                    || !ratio.is_finite()
                    || ratio <= 0.0
                {
                    return Err(format!("invalid SM120 paired sample at window {index}"));
                }
                Ok(ratio)
            })
            .collect()
    }

    pub(super) fn sm120_paired_ratio_p50(ratios: &[f64]) -> Result<f64, String> {
        if ratios.len() != 101
            || ratios
                .iter()
                .any(|ratio| !ratio.is_finite() || *ratio <= 0.0)
        {
            return Err("SM120 paired median requires exactly 101 positive finite ratios".into());
        }
        let mut sorted = ratios.to_vec();
        sorted.sort_by(f64::total_cmp);
        Ok(sorted[50])
    }

    pub(super) const SM120_PAIRED_WINDOWS_PER_ORDER: usize = 101;
    pub(super) const SM120_PAIRED_WARMUP_LAUNCHES: usize = 128;
    pub(super) const SM120_PAIRED_LT_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

    struct LtHandle {
        raw: cudarc::cublaslt::sys::cublasLtHandle_t,
    }

    impl LtHandle {
        fn new() -> Result<Self, String> {
            cudarc::cublaslt::result::create_handle()
                .map(|raw| Self { raw })
                .map_err(|error| format!("create cuBLASLt handle: {error:?}"))
        }
    }

    impl Drop for LtHandle {
        fn drop(&mut self) {
            let _ = unsafe { cudarc::cublaslt::result::destroy_handle(self.raw) };
        }
    }

    struct LtLayout {
        raw: cudarc::cublaslt::sys::cublasLtMatrixLayout_t,
    }

    impl LtLayout {
        fn new(spec: LtMatrixLayoutSpec) -> Result<Self, String> {
            cudarc::cublaslt::result::create_matrix_layout(
                spec.dtype,
                spec.rows,
                spec.columns,
                spec.ld,
            )
            .map(|raw| Self { raw })
            .map_err(|error| format!("create cuBLASLt matrix layout: {error:?}"))
        }
    }

    impl Drop for LtLayout {
        fn drop(&mut self) {
            let _ = unsafe { cudarc::cublaslt::result::destroy_matrix_layout(self.raw) };
        }
    }

    struct LtMatmulDesc {
        raw: cudarc::cublaslt::sys::cublasLtMatmulDesc_t,
    }

    impl LtMatmulDesc {
        fn new(spec: LtNnDescriptorSpec) -> Result<Self, String> {
            use cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE;

            let raw = cudarc::cublaslt::result::create_matmul_desc(spec.compute, spec.scale_dtype)
                .map_err(|error| format!("create cuBLASLt matmul descriptor: {error:?}"))?;
            let descriptor = Self { raw };
            descriptor.set_attribute(CUBLASLT_MATMUL_DESC_EPILOGUE, &spec.epilogue)?;
            Ok(descriptor)
        }

        fn set_attribute<T: Copy>(
            &self,
            attribute: cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t,
            value: &T,
        ) -> Result<(), String> {
            unsafe {
                cudarc::cublaslt::result::set_matmul_desc_attribute(
                    self.raw,
                    attribute,
                    value as *const T as *const c_void,
                    std::mem::size_of::<T>(),
                )
            }
            .map_err(|error| format!("set cuBLASLt matmul descriptor attribute: {error:?}"))
        }
    }

    impl Drop for LtMatmulDesc {
        fn drop(&mut self) {
            let _ = unsafe { cudarc::cublaslt::result::destroy_matmul_desc(self.raw) };
        }
    }

    struct LtPreference {
        raw: cudarc::cublaslt::sys::cublasLtMatmulPreference_t,
    }

    impl LtPreference {
        fn new(workspace_bytes: usize) -> Result<Self, String> {
            use cudarc::cublaslt::sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;

            let raw = cudarc::cublaslt::result::create_matmul_pref()
                .map_err(|error| format!("create cuBLASLt preference: {error:?}"))?;
            let preference = Self { raw };
            unsafe {
                cudarc::cublaslt::result::set_matmul_pref_attribute(
                    preference.raw,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspace_bytes as *const usize as *const c_void,
                    std::mem::size_of::<usize>(),
                )
            }
            .map_err(|error| format!("set cuBLASLt workspace preference: {error:?}"))?;
            Ok(preference)
        }
    }

    impl Drop for LtPreference {
        fn drop(&mut self) {
            let _ = unsafe { cudarc::cublaslt::result::destroy_matmul_pref(self.raw) };
        }
    }

    pub(super) struct Sm120PairedLtRuntime {
        handle: LtHandle,
        workspace: DtypedBuf,
    }

    impl Sm120PairedLtRuntime {
        pub(super) fn new(ctx: &GpuCtx) -> Result<Self, String> {
            let workspace_elements =
                SM120_PAIRED_LT_WORKSPACE_BYTES / WeightDtype::F32.size_bytes();
            Ok(Self {
                handle: LtHandle::new()?,
                workspace: DtypedBuf::zeros(&ctx.stream, workspace_elements, WeightDtype::F32)?,
            })
        }
    }

    #[derive(Clone, Copy)]
    struct Sm120PairedLtBindings {
        a: u64,
        b: u64,
        output: u64,
    }

    struct Sm120PairedLtPlan {
        descriptor: LtMatmulDesc,
        a_layout: LtLayout,
        b_layout: LtLayout,
        output_layout: LtLayout,
        algorithm: cudarc::cublaslt::sys::cublasLtMatmulAlgo_t,
        workspace_bytes: usize,
        waves: f32,
        bindings: Sm120PairedLtBindings,
        alpha: f32,
        beta: f32,
    }

    impl Sm120PairedLtPlan {
        fn prepare(
            runtime: &Sm120PairedLtRuntime,
            spec: LtNnDescriptorSpec,
            bindings: Sm120PairedLtBindings,
        ) -> Result<Self, String> {
            let descriptor = LtMatmulDesc::new(spec)?;
            let a_layout = LtLayout::new(spec.a)?;
            let b_layout = LtLayout::new(spec.b)?;
            let output_layout = LtLayout::new(spec.output)?;
            let preference = LtPreference::new(runtime.workspace.size_bytes())?;
            let heuristic = unsafe {
                cudarc::cublaslt::result::get_matmul_algo_heuristic(
                    runtime.handle.raw,
                    descriptor.raw,
                    a_layout.raw,
                    b_layout.raw,
                    output_layout.raw,
                    output_layout.raw,
                    preference.raw,
                )
            }
            .map_err(|error| format!("query cuBLASLt heuristic: {error:?}"))?;
            if heuristic.workspaceSize > runtime.workspace.size_bytes() {
                return Err(format!(
                    "cuBLASLt heuristic requires {} workspace bytes, only {} allocated",
                    heuristic.workspaceSize,
                    runtime.workspace.size_bytes()
                ));
            }
            Ok(Self {
                descriptor,
                a_layout,
                b_layout,
                output_layout,
                algorithm: heuristic.algo,
                workspace_bytes: heuristic.workspaceSize,
                waves: heuristic.wavesCount,
                bindings,
                alpha: spec.alpha,
                beta: spec.beta,
            })
        }

        fn launch(&self, ctx: &GpuCtx, runtime: &Sm120PairedLtRuntime) -> Result<(), String> {
            unsafe {
                cudarc::cublaslt::result::matmul(
                    runtime.handle.raw,
                    self.descriptor.raw,
                    &self.alpha as *const f32 as *const c_void,
                    &self.beta as *const f32 as *const c_void,
                    self.bindings.a as *const c_void,
                    self.a_layout.raw,
                    self.bindings.b as *const c_void,
                    self.b_layout.raw,
                    self.bindings.output as *const c_void,
                    self.output_layout.raw,
                    self.bindings.output as *mut c_void,
                    self.output_layout.raw,
                    &self.algorithm,
                    runtime.workspace.cached_ptr() as *mut c_void,
                    self.workspace_bytes,
                    ctx.stream.cu_stream() as cudarc::cublaslt::sys::cudaStream_t,
                )
            }
            .map_err(|error| format!("launch paired cuBLASLt matmul: {error:?}"))
        }
    }

    struct Sm120PairedNnRunner {
        cell: Sm120PairedNnCell,
        _a: DtypedBuf,
        _b: DtypedBuf,
        custom_output: DtypedBuf,
        cublas_output: DtypedBuf,
        custom: Sm120PreparedLaunch,
        lt: Sm120PairedLtPlan,
    }

    impl Sm120PairedNnRunner {
        fn prepare(
            ctx: &GpuCtx,
            runtime: &Sm120PairedLtRuntime,
            cell: Sm120PairedNnCell,
        ) -> Result<Self, String> {
            let (m, k, n) = cell.shape.dims;
            let a_elements = m.checked_mul(k).ok_or("paired A element count overflow")?;
            let b_elements = k.checked_mul(n).ok_or("paired B element count overflow")?;
            let output_elements = m
                .checked_mul(n)
                .ok_or("paired output element count overflow")?;
            let a = DtypedBuf::zeros(&ctx.stream, a_elements, cell.dtype)?;
            let b = DtypedBuf::zeros(&ctx.stream, b_elements, cell.dtype)?;
            let custom_output = DtypedBuf::zeros(&ctx.stream, output_elements, cell.dtype)?;
            let cublas_output = DtypedBuf::zeros(&ctx.stream, output_elements, cell.dtype)?;
            upload_sm120_paired_probe(ctx, &a, &b, m, k, n)?;
            let shape = Sm120Shape {
                m,
                k,
                n,
                lda: k,
                ldb: n,
                ldc: n,
            };
            let route = Sm120ForcedRoute {
                op: Sm120Op::Nn,
                dtype: cell.dtype,
                physical: cell.physical,
                shape,
            };
            let maps = prepare_sm120_tensor_maps(
                &ctx.stream,
                &ctx.kernels,
                Sm120MapRequest {
                    op: Sm120Op::Nn,
                    dtype: cell.dtype,
                    tile: cell.physical.tile,
                    bk: cell.physical.bk,
                    a_ptr: a.cached_ptr(),
                    b_ptr: b.cached_ptr(),
                    shape,
                },
            )?;
            let custom = prepare_sm120_tma_forced(
                &ctx.stream,
                &ctx.kernels,
                route,
                &maps,
                Sm120LaunchOperands {
                    output_ptr: custom_output.cached_ptr(),
                    bias_ptr: 0,
                    alpha: cell.alpha,
                    beta: cell.beta,
                },
            )?;
            let spec = sm120_paired_nn_lt_descriptor(cell)?;
            let lt = Sm120PairedLtPlan::prepare(
                runtime,
                spec,
                Sm120PairedLtBindings {
                    a: b.cached_ptr(),
                    b: a.cached_ptr(),
                    output: cublas_output.cached_ptr(),
                },
            )?;
            Ok(Self {
                cell,
                _a: a,
                _b: b,
                custom_output,
                cublas_output,
                custom,
                lt,
            })
        }

        fn launch_custom(&self, ctx: &GpuCtx) -> Result<(), String> {
            launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &self.custom).map(|_| ())
        }

        fn launch_cublas(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedLtRuntime,
        ) -> Result<(), String> {
            self.lt.launch(ctx, runtime)
        }
    }

    fn upload_sm120_paired_probe(
        ctx: &GpuCtx,
        a: &DtypedBuf,
        b: &DtypedBuf,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        let mut host = vec![0.0f32; m.checked_mul(k).ok_or("paired probe A overflow")?];
        for row in 0..m {
            host[row * k] = ((row % 7) as i32 - 3) as f32 * 0.125;
        }
        a.upload_f32(&ctx.stream, &host)?;

        host.clear();
        host.resize(k.checked_mul(n).ok_or("paired probe B overflow")?, 0.0);
        for column in 0..n {
            host[column] = ((column % 11) as i32 - 5) as f32 * 0.125;
        }
        b.upload_f32(&ctx.stream, &host)
    }

    fn validate_sm120_paired_semantics(
        ctx: &GpuCtx,
        runtime: &Sm120PairedLtRuntime,
        runner: &Sm120PairedNnRunner,
    ) -> Result<(), String> {
        runner.launch_custom(ctx)?;
        runner.launch_cublas(ctx, runtime)?;
        let elements = runner.custom_output.len_elems();
        let mut custom = vec![0.0f32; elements];
        let mut cublas = vec![0.0f32; elements];
        runner
            .custom_output
            .download_f32(&ctx.stream, &mut custom)?;
        runner
            .cublas_output
            .download_f32(&ctx.stream, &mut cublas)?;
        if let Some(index) = custom
            .iter()
            .zip(&cublas)
            .position(|(&lhs, &rhs)| lhs.to_bits() != rhs.to_bits())
        {
            return Err(format!(
                "paired nonzero semantic probe differs at {index}: custom={} cublas={}",
                custom[index], cublas[index]
            ));
        }
        Ok(())
    }

    #[derive(Clone, Copy)]
    struct Sm120PairedIterations {
        custom: usize,
        cublas: usize,
    }

    struct Sm120PairedOrderSamples {
        order: ComparatorOrder,
        custom_us: Vec<f64>,
        cublas_us: Vec<f64>,
    }

    impl Sm120PairedOrderSamples {
        fn new(order: ComparatorOrder) -> Self {
            Self {
                order,
                custom_us: Vec::with_capacity(SM120_PAIRED_WINDOWS_PER_ORDER),
                cublas_us: Vec::with_capacity(SM120_PAIRED_WINDOWS_PER_ORDER),
            }
        }

        fn push(&mut self, custom_us: f64, cublas_us: f64) -> Result<(), String> {
            let index = self.custom_us.len();
            validate_sample_us(custom_us, index)?;
            validate_sample_us(cublas_us, index)?;
            self.custom_us.push(custom_us);
            self.cublas_us.push(cublas_us);
            Ok(())
        }
    }

    fn measure_sm120_paired_arm_ms(
        ctx: &GpuCtx,
        runtime: &Sm120PairedLtRuntime,
        runner: &Sm120PairedNnRunner,
        custom: bool,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("paired benchmark iteration count must be positive".into());
        }
        let start = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record paired pilot start event: {error:?}"))?;
        for _ in 0..iterations {
            if custom {
                runner.launch_custom(ctx)?;
            } else {
                runner.launch_cublas(ctx, runtime)?;
            }
        }
        let end = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record paired pilot end event: {error:?}"))?;
        start
            .elapsed_ms(&end)
            .map(f64::from)
            .map_err(|error| format!("measure paired pilot events: {error:?}"))
    }

    fn calibrate_sm120_paired_iterations(
        ctx: &GpuCtx,
        runtime: &Sm120PairedLtRuntime,
        runner: &Sm120PairedNnRunner,
    ) -> Result<Sm120PairedIterations, String> {
        Ok(Sm120PairedIterations {
            custom: calibrate_measurement(|iterations| {
                measure_sm120_paired_arm_ms(ctx, runtime, runner, true, iterations)
            })?,
            cublas: calibrate_measurement(|iterations| {
                measure_sm120_paired_arm_ms(ctx, runtime, runner, false, iterations)
            })?,
        })
    }

    fn warm_sm120_paired_arms(
        ctx: &GpuCtx,
        runtime: &Sm120PairedLtRuntime,
        runner: &Sm120PairedNnRunner,
    ) -> Result<(), String> {
        for _ in 0..SM120_PAIRED_WARMUP_LAUNCHES {
            runner.launch_custom(ctx)?;
        }
        for _ in 0..SM120_PAIRED_WARMUP_LAUNCHES {
            runner.launch_cublas(ctx, runtime)?;
        }
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize paired warmups: {error:?}"))
    }

    fn enqueue_sm120_paired_arm(
        ctx: &GpuCtx,
        runtime: &Sm120PairedLtRuntime,
        runner: &Sm120PairedNnRunner,
        custom: bool,
        iterations: usize,
    ) -> Result<(), String> {
        for _ in 0..iterations {
            if custom {
                runner.launch_custom(ctx)?;
            } else {
                runner.launch_cublas(ctx, runtime)?;
            }
        }
        Ok(())
    }

    fn measure_sm120_paired_window(
        ctx: &GpuCtx,
        runtime: &Sm120PairedLtRuntime,
        runner: &Sm120PairedNnRunner,
        iterations: Sm120PairedIterations,
        order: ComparatorOrder,
    ) -> Result<(f64, f64), String> {
        let first_custom = order == ComparatorOrder::CustomThenCublas;
        let first_iterations = if first_custom {
            iterations.custom
        } else {
            iterations.cublas
        };
        let second_iterations = if first_custom {
            iterations.cublas
        } else {
            iterations.custom
        };
        let first_start = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record first paired start event: {error:?}"))?;
        enqueue_sm120_paired_arm(ctx, runtime, runner, first_custom, first_iterations)?;
        let first_end = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record first paired end event: {error:?}"))?;
        let second_start = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record second paired start event: {error:?}"))?;
        enqueue_sm120_paired_arm(ctx, runtime, runner, !first_custom, second_iterations)?;
        let second_end = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record second paired end event: {error:?}"))?;

        let first_ms = first_start
            .elapsed_ms(&first_end)
            .map(f64::from)
            .map_err(|error| format!("measure first paired arm: {error:?}"))?;
        let second_ms = second_start
            .elapsed_ms(&second_end)
            .map(f64::from)
            .map_err(|error| format!("measure second paired arm: {error:?}"))?;
        let first_us = first_ms * 1000.0 / first_iterations as f64;
        let second_us = second_ms * 1000.0 / second_iterations as f64;
        Ok(if first_custom {
            (first_us, second_us)
        } else {
            (second_us, first_us)
        })
    }

    fn collect_sm120_paired_samples(
        ctx: &GpuCtx,
        runtime: &Sm120PairedLtRuntime,
        runner: &Sm120PairedNnRunner,
        iterations: Sm120PairedIterations,
    ) -> Result<[Sm120PairedOrderSamples; 2], String> {
        let mut custom_first = Sm120PairedOrderSamples::new(ComparatorOrder::CustomThenCublas);
        let mut cublas_first = Sm120PairedOrderSamples::new(ComparatorOrder::CublasThenCustom);
        for order in sm120_paired_comparator_schedule() {
            let (custom_us, cublas_us) =
                measure_sm120_paired_window(ctx, runtime, runner, iterations, order)?;
            match order {
                ComparatorOrder::CustomThenCublas => custom_first.push(custom_us, cublas_us)?,
                ComparatorOrder::CublasThenCustom => cublas_first.push(custom_us, cublas_us)?,
            }
        }
        Ok([custom_first, cublas_first])
    }

    fn render_f64_samples(values: &[f64]) -> String {
        let mut rendered = String::new();
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                rendered.push(',');
            }
            write!(rendered, "{value:.9}").expect("String writes cannot fail");
        }
        rendered
    }

    fn render_lt_algorithm(algorithm: cudarc::cublaslt::sys::cublasLtMatmulAlgo_t) -> String {
        algorithm
            .data
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn emit_sm120_paired_record(
        ctx: &GpuCtx,
        runner: &Sm120PairedNnRunner,
        iterations: Sm120PairedIterations,
        samples: &Sm120PairedOrderSamples,
    ) -> Result<(), String> {
        let ratios = sm120_paired_ratios(&samples.custom_us, &samples.cublas_us)?;
        let ratio_p50 = sm120_paired_ratio_p50(&ratios)?;
        let identity = runner.custom.identity();
        let resources = runner.custom.resources();
        let (m, k, n) = runner.cell.shape.dims;
        let cc = ctx.stream.context().compute_capability().expect("CUDA CC");
        println!(
            concat!(
                "{{\"schema\":\"MambaBiSm120HalfNnPairedCublasLtV1\",",
                "\"suite\":\"gemm_bi_sm120_half_nn_paired_cublaslt_hot\",",
                "\"scope\":\"performance_only\",\"cell_id\":\"{}\",",
                "\"comparator_order\":\"{}\",\"cc\":\"{}.{}\",",
                "\"dtype\":\"{}\",\"op\":\"nn\",\"shape\":\"{}\",",
                "\"m\":{},\"k\":{},\"n\":{},\"alpha\":1.0,\"beta\":0.0,",
                "\"bias\":false,",
                "\"cublaslt_compute\":\"CUBLAS_COMPUTE_32F\",",
                "\"cublaslt_epilogue\":\"CUBLASLT_EPILOGUE_DEFAULT\",",
                "\"cublaslt_a_role\":\"row_major_b_as_col_major_n_by_k\",",
                "\"cublaslt_b_role\":\"row_major_a_as_col_major_k_by_m\",",
                "\"cublaslt_output_layout\":\"col_major_n_by_m_ld_n\",",
                "\"semantic_probe\":\"nonzero_outer_product_bit_exact\",",
                "\"custom_symbol\":\"{}\",\"custom_tile\":\"{:?}\",",
                "\"custom_bk\":\"{:?}\",\"custom_stages\":\"{:?}\",",
                "\"custom_registers_per_thread\":{},\"custom_shared_bytes\":{},",
                "\"custom_iterations\":{},\"cublas_iterations\":{},",
                "\"warmup_launches_per_arm\":{},\"windows\":{},",
                "\"cublaslt_workspace_bytes\":{},\"cublaslt_waves\":{:.9},",
                "\"cublaslt_algorithm\":[{}],\"ratio_orientation\":\"custom_over_cublas\",",
                "\"ratio_p50\":{:.9},\"custom_us\":[{}],\"cublas_us\":[{}],",
                "\"ratios\":[{}]}}"
            ),
            sm120_paired_nn_cell_id(runner.cell),
            samples.order.as_str(),
            cc.0,
            cc.1,
            dtype_name(runner.cell.dtype),
            runner.cell.shape.name,
            m,
            k,
            n,
            identity.symbol,
            identity.physical.tile,
            identity.physical.bk,
            identity.physical.stages,
            resources.registers_per_thread,
            resources.dynamic_shared_bytes,
            iterations.custom,
            iterations.cublas,
            SM120_PAIRED_WARMUP_LAUNCHES,
            samples.custom_us.len(),
            runner.lt.workspace_bytes,
            runner.lt.waves,
            render_lt_algorithm(runner.lt.algorithm),
            ratio_p50,
            render_f64_samples(&samples.custom_us),
            render_f64_samples(&samples.cublas_us),
            render_f64_samples(&ratios),
        );
        Ok(())
    }

    pub(super) fn run_sm120_paired_nn_cell(
        ctx: &GpuCtx,
        runtime: &Sm120PairedLtRuntime,
        cell: Sm120PairedNnCell,
    ) -> Result<(), String> {
        let runner = Sm120PairedNnRunner::prepare(ctx, runtime, cell)?;
        validate_sm120_paired_semantics(ctx, runtime, &runner)?;
        warm_sm120_paired_arms(ctx, runtime, &runner)?;
        let iterations = calibrate_sm120_paired_iterations(ctx, runtime, &runner)?;
        let samples = collect_sm120_paired_samples(ctx, runtime, &runner, iterations)?;
        for order_samples in &samples {
            emit_sm120_paired_record(ctx, &runner, iterations, order_samples)?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda-cublaslt-qualification")]
mod cublaslt_tn_nt_qualification {
    use super::*;
    use std::mem::MaybeUninit;

    pub(super) const SM120_PAIRED_FINAL_WINDOWS_PER_ORDER: usize = 101;
    pub(super) const SM120_PAIRED_PILOT_WINDOWS_PER_ORDER: usize = 11;
    pub(super) const SM120_PAIRED_WARMUP_LAUNCHES: usize = 128;
    pub(super) const SM120_PAIRED_LT_HEURISTIC_LIMIT: usize = 32;
    const SM120_PAIRED_LT_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

    #[derive(Clone, Copy)]
    pub(super) struct Sm120PairedTnNtCell {
        pub(super) dtype: WeightDtype,
        pub(super) op: Sm120Op,
        pub(super) shape: Shape,
        pub(super) physical: Sm120PhysicalRoute,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct Sm120PairedContract {
        pub(super) output_dtype: WeightDtype,
        pub(super) alpha: f32,
        pub(super) beta: f32,
        pub(super) bias_ptr: u64,
        pub(super) reset_before_window: bool,
        a_elements: usize,
        b_elements: usize,
        output_elements: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct LtMatrixLayoutSpec {
        pub(super) dtype: cudarc::cublaslt::sys::cudaDataType,
        pub(super) rows: u64,
        pub(super) columns: u64,
        pub(super) ld: i64,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct LtDescriptorSpec {
        pub(super) compute: cudarc::cublaslt::sys::cublasComputeType_t,
        pub(super) scale_dtype: cudarc::cublaslt::sys::cudaDataType,
        pub(super) epilogue: cudarc::cublaslt::sys::cublasLtEpilogue_t,
        pub(super) trans_a: cudarc::cublas::sys::cublasOperation_t,
        pub(super) trans_b: cudarc::cublas::sys::cublasOperation_t,
        pub(super) a: LtMatrixLayoutSpec,
        pub(super) b: LtMatrixLayoutSpec,
        pub(super) c: LtMatrixLayoutSpec,
        pub(super) d: LtMatrixLayoutSpec,
        pub(super) alpha: f32,
        pub(super) beta: f32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GemmExGeometry {
        pub(super) trans_a: cudarc::cublas::sys::cublasOperation_t,
        pub(super) trans_b: cudarc::cublas::sys::cublasOperation_t,
        pub(super) m: c_int,
        pub(super) n: c_int,
        pub(super) k: c_int,
        pub(super) lda: c_int,
        pub(super) ldb: c_int,
        pub(super) ldc: c_int,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(super) struct SemanticProbe {
        pub(super) a: Vec<f32>,
        pub(super) b: Vec<f32>,
        pub(super) seed: Vec<f32>,
        pub(super) expected: Vec<f32>,
        pub(super) timed_a: Vec<f32>,
        pub(super) timed_b: Vec<f32>,
        pub(super) timed_seed: Vec<f32>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum PairedOrder {
        CustomThenVendor,
        VendorThenCustom,
    }

    impl PairedOrder {
        pub(super) fn as_str(self) -> &'static str {
            match self {
                Self::CustomThenVendor => "custom_then_vendor",
                Self::VendorThenCustom => "vendor_then_custom",
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum VendorPlanKind {
        GemmEx,
        CublasLt(usize),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(super) struct VendorPilot {
        pub(super) kind: VendorPlanKind,
        pub(super) custom_then_vendor_us: Vec<f64>,
        pub(super) vendor_then_custom_us: Vec<f64>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum LtRejectionReason {
        Status,
        Workspace,
        Smoke,
    }

    #[derive(Clone, Debug)]
    struct LtRejection {
        heuristic_index: Option<usize>,
        reason: LtRejectionReason,
        detail: String,
    }

    #[derive(Clone, Copy)]
    struct LtCandidate {
        heuristic_index: usize,
        algorithm: cudarc::cublaslt::sys::cublasLtMatmulAlgo_t,
        workspace_bytes: usize,
        waves: f32,
    }

    pub(super) fn build_sm120_paired_tn_nt_cells() -> Vec<Sm120PairedTnNtCell> {
        let mut cells = Vec::with_capacity(12);
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for op in [Sm120Op::Tn, Sm120Op::Nt] {
                for shape in [
                    Shape {
                        name: "large",
                        dims: (2048, 3072, 768),
                    },
                    Shape {
                        name: "large_deep",
                        dims: (4096, 3072, 1536),
                    },
                    Shape {
                        name: "d768_out_proj",
                        dims: (2048, 1536, 768),
                    },
                ] {
                    let physical = match (op, shape.name) {
                        (Sm120Op::Tn, "large" | "d768_out_proj") => Sm120PhysicalRoute {
                            tile: Sm120Tile::M64N128,
                            bk: Sm120Bk::Bk32,
                            stages: Sm120Stages::S3,
                        },
                        (Sm120Op::Tn, "large_deep") => Sm120PhysicalRoute {
                            tile: Sm120Tile::M128N128,
                            bk: Sm120Bk::Bk32,
                            stages: Sm120Stages::S3,
                        },
                        (Sm120Op::Nt, "d768_out_proj") => Sm120PhysicalRoute {
                            tile: Sm120Tile::M64N64,
                            bk: Sm120Bk::Bk64,
                            stages: Sm120Stages::S2,
                        },
                        (Sm120Op::Nt, "large") => Sm120PhysicalRoute {
                            tile: Sm120Tile::M128N64,
                            bk: Sm120Bk::Bk32,
                            stages: Sm120Stages::S2,
                        },
                        (Sm120Op::Nt, "large_deep") => Sm120PhysicalRoute {
                            tile: Sm120Tile::M128N128,
                            bk: Sm120Bk::Bk32,
                            stages: Sm120Stages::S3,
                        },
                        _ => unreachable!("fixed TN/NT inventory"),
                    };
                    cells.push(Sm120PairedTnNtCell {
                        dtype,
                        op,
                        shape,
                        physical,
                    });
                }
            }
        }
        cells
    }

    pub(super) fn sm120_paired_tn_nt_cell_id(cell: Sm120PairedTnNtCell) -> &'static str {
        match (cell.dtype, cell.op, cell.shape.name) {
            (WeightDtype::Bf16, Sm120Op::Tn, "large") => "sm120_paired/bf16/tn/large",
            (WeightDtype::Bf16, Sm120Op::Tn, "large_deep") => "sm120_paired/bf16/tn/large_deep",
            (WeightDtype::Bf16, Sm120Op::Tn, "d768_out_proj") => {
                "sm120_paired/bf16/tn/d768_out_proj"
            }
            (WeightDtype::Bf16, Sm120Op::Nt, "large") => "sm120_paired/bf16/nt/large",
            (WeightDtype::Bf16, Sm120Op::Nt, "large_deep") => "sm120_paired/bf16/nt/large_deep",
            (WeightDtype::Bf16, Sm120Op::Nt, "d768_out_proj") => {
                "sm120_paired/bf16/nt/d768_out_proj"
            }
            (WeightDtype::F16, Sm120Op::Tn, "large") => "sm120_paired/f16/tn/large",
            (WeightDtype::F16, Sm120Op::Tn, "large_deep") => "sm120_paired/f16/tn/large_deep",
            (WeightDtype::F16, Sm120Op::Tn, "d768_out_proj") => "sm120_paired/f16/tn/d768_out_proj",
            (WeightDtype::F16, Sm120Op::Nt, "large") => "sm120_paired/f16/nt/large",
            (WeightDtype::F16, Sm120Op::Nt, "large_deep") => "sm120_paired/f16/nt/large_deep",
            (WeightDtype::F16, Sm120Op::Nt, "d768_out_proj") => "sm120_paired/f16/nt/d768_out_proj",
            _ => "sm120_paired/unsupported",
        }
    }

    pub(super) fn sm120_paired_contract(
        cell: Sm120PairedTnNtCell,
    ) -> Result<Sm120PairedContract, String> {
        let (m, k, n) = cell.shape.dims;
        let extent = |left: usize, right: usize, label: &str| {
            left.checked_mul(right)
                .ok_or_else(|| format!("paired {label} extent overflows usize"))
        };
        match (cell.dtype, cell.op) {
            (WeightDtype::Bf16 | WeightDtype::F16, Sm120Op::Tn) => Ok(Sm120PairedContract {
                output_dtype: WeightDtype::F32,
                alpha: 1.0,
                beta: 1.0,
                bias_ptr: 0,
                reset_before_window: true,
                a_elements: extent(m, k, "TN A")?,
                b_elements: extent(m, n, "TN B")?,
                output_elements: extent(k, n, "TN output")?,
            }),
            (WeightDtype::Bf16 | WeightDtype::F16, Sm120Op::Nt) => Ok(Sm120PairedContract {
                output_dtype: cell.dtype,
                alpha: 1.0,
                beta: 0.0,
                bias_ptr: 0,
                reset_before_window: false,
                a_elements: extent(m, n, "NT A")?,
                b_elements: extent(k, n, "NT B")?,
                output_elements: extent(m, k, "NT output")?,
            }),
            (WeightDtype::F32, _) => Err("paired TN/NT requires BF16 or F16 inputs".into()),
            (_, Sm120Op::Nn) => Err("paired TN/NT inventory rejects NN".into()),
        }
    }

    fn lt_dtype(dtype: WeightDtype) -> Result<cudarc::cublaslt::sys::cudaDataType, String> {
        use cudarc::cublaslt::sys::cudaDataType_t;
        match dtype {
            WeightDtype::Bf16 => Ok(cudaDataType_t::CUDA_R_16BF),
            WeightDtype::F16 => Ok(cudaDataType_t::CUDA_R_16F),
            WeightDtype::F32 => Ok(cudaDataType_t::CUDA_R_32F),
        }
    }

    pub(super) fn sm120_paired_lt_descriptor(
        cell: Sm120PairedTnNtCell,
    ) -> Result<LtDescriptorSpec, String> {
        use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
        use cudarc::cublaslt::sys::{cublasComputeType_t, cublasLtEpilogue_t, cudaDataType_t};

        let contract = sm120_paired_contract(cell)?;
        let input_dtype = lt_dtype(cell.dtype)?;
        let output_dtype = lt_dtype(contract.output_dtype)?;
        let (m, k, n) = cell.shape.dims;
        let u64_dim =
            |value| u64::try_from(value).map_err(|_| String::from("Lt dimension exceeds u64"));
        let i64_dim =
            |value| i64::try_from(value).map_err(|_| String::from("Lt stride exceeds i64"));
        let layout = |dtype, rows, columns, ld| {
            Ok::<_, String>(LtMatrixLayoutSpec {
                dtype,
                rows: u64_dim(rows)?,
                columns: u64_dim(columns)?,
                ld: i64_dim(ld)?,
            })
        };
        let (trans_a, trans_b, a, b, c) = match cell.op {
            Sm120Op::Tn => (
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                layout(input_dtype, n, m, n)?,
                layout(input_dtype, k, m, k)?,
                layout(output_dtype, n, k, n)?,
            ),
            Sm120Op::Nt => (
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                layout(input_dtype, n, k, n)?,
                layout(input_dtype, n, m, n)?,
                layout(output_dtype, k, m, k)?,
            ),
            Sm120Op::Nn => return Err("paired TN/NT descriptor rejects NN".into()),
        };
        Ok(LtDescriptorSpec {
            compute: cublasComputeType_t::CUBLAS_COMPUTE_32F,
            scale_dtype: cudaDataType_t::CUDA_R_32F,
            epilogue: cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            trans_a,
            trans_b,
            a,
            b,
            c,
            d: c,
            alpha: contract.alpha,
            beta: contract.beta,
        })
    }

    pub(super) fn sm120_paired_gemm_ex_geometry(cell: Sm120PairedTnNtCell) -> GemmExGeometry {
        let geometry = cublas_denominator_geometry(
            match cell.op {
                Sm120Op::Tn => ResolvedGemmOp::Tn,
                Sm120Op::Nt => ResolvedGemmOp::Nt,
                Sm120Op::Nn => ResolvedGemmOp::Nn,
            },
            cell.shape.dims,
        );
        GemmExGeometry {
            trans_a: geometry.trans_a,
            trans_b: geometry.trans_b,
            m: geometry.m,
            n: geometry.n,
            k: geometry.k,
            lda: geometry.lda,
            ldb: geometry.ldb,
            ldc: geometry.ldc,
        }
    }

    fn quantize(dtype: WeightDtype, value: f32) -> f32 {
        match dtype {
            WeightDtype::Bf16 => half::bf16::from_f32(value).to_f32(),
            WeightDtype::F16 => half::f16::from_f32(value).to_f32(),
            WeightDtype::F32 => value,
        }
    }

    pub(super) fn build_sm120_paired_semantic_probe(
        cell: Sm120PairedTnNtCell,
    ) -> Result<SemanticProbe, String> {
        let contract = sm120_paired_contract(cell)?;
        let (m, k, n) = cell.shape.dims;
        let mut a = vec![0.0; contract.a_elements];
        let mut b = vec![0.0; contract.b_elements];
        let mut seed = vec![0.0; contract.output_elements];
        let mut expected = vec![0.0; contract.output_elements];
        match cell.op {
            Sm120Op::Tn => {
                for row in 0..k {
                    a[row] = 0.25 + (row % 4) as f32 * 0.125;
                }
                for column in 0..n {
                    b[column] = -0.5 - (column % 4) as f32 * 0.125;
                }
                for (index, value) in seed.iter_mut().enumerate() {
                    *value = 1.0 + (index % 7) as f32 * 0.125;
                }
                for row in 0..k {
                    for column in 0..n {
                        let index = row * n + column;
                        expected[index] = seed[index] + a[row] * b[column];
                    }
                }
            }
            Sm120Op::Nt => {
                for row in 0..m {
                    a[row * n] = 0.25 + (row % 4) as f32 * 0.125;
                }
                for column in 0..k {
                    b[column * n] = -0.5 - (column % 4) as f32 * 0.125;
                }
                for row in 0..m {
                    for column in 0..k {
                        expected[row * k + column] =
                            quantize(cell.dtype, a[row * n] * b[column * n]);
                    }
                }
            }
            Sm120Op::Nn => return Err("paired TN/NT semantic probe rejects NN".into()),
        }
        Ok(SemanticProbe {
            timed_a: vec![0.0; a.len()],
            timed_b: vec![0.0; b.len()],
            timed_seed: seed.clone(),
            a,
            b,
            seed,
            expected,
        })
    }

    fn paired_schedule(windows_per_order: usize) -> Vec<PairedOrder> {
        let mut schedule = Vec::with_capacity(windows_per_order * 2);
        for _ in 0..windows_per_order {
            schedule.push(PairedOrder::CustomThenVendor);
            schedule.push(PairedOrder::VendorThenCustom);
        }
        schedule
    }

    pub(super) fn sm120_paired_final_schedule() -> Vec<PairedOrder> {
        paired_schedule(SM120_PAIRED_FINAL_WINDOWS_PER_ORDER)
    }

    pub(super) fn sm120_paired_pilot_schedule() -> Vec<PairedOrder> {
        paired_schedule(SM120_PAIRED_PILOT_WINDOWS_PER_ORDER)
    }

    pub(super) fn sm120_paired_ratios(custom: &[f64], vendor: &[f64]) -> Result<Vec<f64>, String> {
        if custom.len() != SM120_PAIRED_FINAL_WINDOWS_PER_ORDER
            || vendor.len() != SM120_PAIRED_FINAL_WINDOWS_PER_ORDER
        {
            return Err("SM120 TN/NT pairing requires exactly 101 samples per arm".into());
        }
        custom
            .iter()
            .zip(vendor)
            .enumerate()
            .map(|(index, (&custom, &vendor))| {
                let ratio = custom / vendor;
                if !custom.is_finite()
                    || custom <= 0.0
                    || !vendor.is_finite()
                    || vendor <= 0.0
                    || !ratio.is_finite()
                    || ratio <= 0.0
                {
                    return Err(format!(
                        "invalid SM120 TN/NT paired sample at window {index}"
                    ));
                }
                Ok(ratio)
            })
            .collect()
    }

    pub(super) fn sm120_paired_ratio_p50(ratios: &[f64]) -> Result<f64, String> {
        if ratios.len() != SM120_PAIRED_FINAL_WINDOWS_PER_ORDER
            || ratios
                .iter()
                .any(|ratio| !ratio.is_finite() || *ratio <= 0.0)
        {
            return Err("SM120 TN/NT median requires 101 positive finite ratios".into());
        }
        let mut sorted = ratios.to_vec();
        sorted.sort_by(f64::total_cmp);
        Ok(sorted[50])
    }

    fn even_median(values: &[f64]) -> Result<f64, String> {
        if values.is_empty()
            || values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("vendor pilot requires positive finite samples".into());
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let upper = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            Ok((sorted[upper - 1] + sorted[upper]) * 0.5)
        } else {
            Ok(sorted[upper])
        }
    }

    fn vendor_pilot_median(pilot: &VendorPilot) -> Result<f64, String> {
        if pilot.custom_then_vendor_us.len() != SM120_PAIRED_PILOT_WINDOWS_PER_ORDER
            || pilot.vendor_then_custom_us.len() != SM120_PAIRED_PILOT_WINDOWS_PER_ORDER
        {
            return Err("vendor pilot requires 11 samples in each comparator order".into());
        }
        let mut samples = pilot.custom_then_vendor_us.clone();
        samples.extend_from_slice(&pilot.vendor_then_custom_us);
        even_median(&samples)
    }

    pub(super) fn select_fastest_vendor_plan(
        pilots: &[VendorPilot],
    ) -> Result<VendorPlanKind, String> {
        if pilots
            .iter()
            .filter(|pilot| pilot.kind == VendorPlanKind::GemmEx)
            .count()
            != 1
        {
            return Err("vendor pilot must include GemmEx exactly once".into());
        }
        let mut fastest = None;
        for pilot in pilots {
            let median = vendor_pilot_median(pilot)?;
            if fastest.is_none_or(|(_, best)| median < best) {
                fastest = Some((pilot.kind, median));
            }
        }
        fastest
            .map(|(kind, _)| kind)
            .ok_or_else(|| "vendor pilot inventory is empty".into())
    }

    pub(super) fn reject_lt_heuristic(
        status_success: bool,
        workspace_bytes: usize,
        smoke_success: bool,
        workspace_limit: usize,
    ) -> Option<LtRejectionReason> {
        if !status_success {
            Some(LtRejectionReason::Status)
        } else if workspace_bytes > workspace_limit {
            Some(LtRejectionReason::Workspace)
        } else if !smoke_success {
            Some(LtRejectionReason::Smoke)
        } else {
            None
        }
    }

    struct LtHandle {
        raw: cudarc::cublaslt::sys::cublasLtHandle_t,
    }

    impl LtHandle {
        fn new() -> Result<Self, String> {
            cudarc::cublaslt::result::create_handle()
                .map(|raw| Self { raw })
                .map_err(|error| format!("create TN/NT cuBLASLt handle: {error:?}"))
        }
    }

    impl Drop for LtHandle {
        fn drop(&mut self) {
            let _ = unsafe { cudarc::cublaslt::result::destroy_handle(self.raw) };
        }
    }

    struct LtLayout {
        raw: cudarc::cublaslt::sys::cublasLtMatrixLayout_t,
    }

    impl LtLayout {
        fn new(spec: LtMatrixLayoutSpec) -> Result<Self, String> {
            cudarc::cublaslt::result::create_matrix_layout(
                spec.dtype,
                spec.rows,
                spec.columns,
                spec.ld,
            )
            .map(|raw| Self { raw })
            .map_err(|error| format!("create TN/NT cuBLASLt matrix layout: {error:?}"))
        }
    }

    impl Drop for LtLayout {
        fn drop(&mut self) {
            let _ = unsafe { cudarc::cublaslt::result::destroy_matrix_layout(self.raw) };
        }
    }

    struct LtMatmulDesc {
        raw: cudarc::cublaslt::sys::cublasLtMatmulDesc_t,
    }

    impl LtMatmulDesc {
        fn new(spec: LtDescriptorSpec) -> Result<Self, String> {
            use cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::{
                CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_MATMUL_DESC_TRANSA,
                CUBLASLT_MATMUL_DESC_TRANSB,
            };

            let raw = cudarc::cublaslt::result::create_matmul_desc(spec.compute, spec.scale_dtype)
                .map_err(|error| format!("create TN/NT cuBLASLt descriptor: {error:?}"))?;
            let descriptor = Self { raw };
            descriptor.set_attribute(CUBLASLT_MATMUL_DESC_EPILOGUE, &spec.epilogue)?;
            descriptor.set_attribute(CUBLASLT_MATMUL_DESC_TRANSA, &spec.trans_a)?;
            descriptor.set_attribute(CUBLASLT_MATMUL_DESC_TRANSB, &spec.trans_b)?;
            Ok(descriptor)
        }

        fn set_attribute<T: Copy>(
            &self,
            attribute: cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t,
            value: &T,
        ) -> Result<(), String> {
            unsafe {
                cudarc::cublaslt::result::set_matmul_desc_attribute(
                    self.raw,
                    attribute,
                    value as *const T as *const c_void,
                    std::mem::size_of::<T>(),
                )
            }
            .map_err(|error| format!("set TN/NT cuBLASLt descriptor attribute: {error:?}"))
        }
    }

    impl Drop for LtMatmulDesc {
        fn drop(&mut self) {
            let _ = unsafe { cudarc::cublaslt::result::destroy_matmul_desc(self.raw) };
        }
    }

    struct LtPreference {
        raw: cudarc::cublaslt::sys::cublasLtMatmulPreference_t,
    }

    impl LtPreference {
        fn new(workspace_bytes: usize) -> Result<Self, String> {
            use cudarc::cublaslt::sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;

            let raw = cudarc::cublaslt::result::create_matmul_pref()
                .map_err(|error| format!("create TN/NT cuBLASLt preference: {error:?}"))?;
            let preference = Self { raw };
            unsafe {
                cudarc::cublaslt::result::set_matmul_pref_attribute(
                    preference.raw,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspace_bytes as *const usize as *const c_void,
                    std::mem::size_of::<usize>(),
                )
            }
            .map_err(|error| format!("set TN/NT cuBLASLt preference: {error:?}"))?;
            Ok(preference)
        }
    }

    impl Drop for LtPreference {
        fn drop(&mut self) {
            let _ = unsafe { cudarc::cublaslt::result::destroy_matmul_pref(self.raw) };
        }
    }

    pub(super) struct Sm120PairedTnNtRuntime {
        handle: LtHandle,
        workspace: DtypedBuf,
    }

    impl Sm120PairedTnNtRuntime {
        pub(super) fn new(ctx: &GpuCtx) -> Result<Self, String> {
            let elements = SM120_PAIRED_LT_WORKSPACE_BYTES / WeightDtype::F32.size_bytes();
            Ok(Self {
                handle: LtHandle::new()?,
                workspace: DtypedBuf::zeros(&ctx.stream, elements, WeightDtype::F32)?,
            })
        }
    }

    struct LtPlanSet {
        spec: LtDescriptorSpec,
        descriptor: LtMatmulDesc,
        a_layout: LtLayout,
        b_layout: LtLayout,
        c_layout: LtLayout,
        d_layout: LtLayout,
        a_ptr: u64,
        b_ptr: u64,
    }

    impl LtPlanSet {
        fn new(spec: LtDescriptorSpec, a_ptr: u64, b_ptr: u64) -> Result<Self, String> {
            Ok(Self {
                spec,
                descriptor: LtMatmulDesc::new(spec)?,
                a_layout: LtLayout::new(spec.a)?,
                b_layout: LtLayout::new(spec.b)?,
                c_layout: LtLayout::new(spec.c)?,
                d_layout: LtLayout::new(spec.d)?,
                a_ptr,
                b_ptr,
            })
        }

        fn query_candidates(
            &self,
            runtime: &Sm120PairedTnNtRuntime,
        ) -> Result<(Vec<LtCandidate>, Vec<LtRejection>), String> {
            use cudarc::cublaslt::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS;

            let preference = LtPreference::new(runtime.workspace.size_bytes())?;
            let mut raw: [MaybeUninit<cudarc::cublaslt::sys::cublasLtMatmulHeuristicResult_t>;
                SM120_PAIRED_LT_HEURISTIC_LIMIT] =
                [MaybeUninit::uninit(); SM120_PAIRED_LT_HEURISTIC_LIMIT];
            let mut returned = 0;
            let status = unsafe {
                cudarc::cublaslt::sys::cublasLtMatmulAlgoGetHeuristic(
                    runtime.handle.raw,
                    self.descriptor.raw,
                    self.a_layout.raw,
                    self.b_layout.raw,
                    self.c_layout.raw,
                    self.d_layout.raw,
                    preference.raw,
                    SM120_PAIRED_LT_HEURISTIC_LIMIT as c_int,
                    raw.as_mut_ptr() as *mut cudarc::cublaslt::sys::cublasLtMatmulHeuristicResult_t,
                    &mut returned,
                )
            };
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(format!("cuBLASLt heuristic query status {status:?}"));
            }
            let returned = usize::try_from(returned)
                .map_err(|_| "cuBLASLt returned a negative heuristic count".to_string())?;
            if returned > SM120_PAIRED_LT_HEURISTIC_LIMIT {
                return Err(format!("cuBLASLt returned too many heuristics: {returned}"));
            }
            let mut candidates = Vec::with_capacity(returned);
            let mut rejections = Vec::new();
            for (heuristic_index, entry) in raw[..returned].iter().enumerate() {
                let heuristic = unsafe { entry.assume_init() };
                let reason = reject_lt_heuristic(
                    heuristic.state == CUBLAS_STATUS_SUCCESS,
                    heuristic.workspaceSize,
                    true,
                    runtime.workspace.size_bytes(),
                );
                if let Some(reason) = reason {
                    rejections.push(LtRejection {
                        heuristic_index: Some(heuristic_index),
                        reason,
                        detail: format!(
                            "state={:?},workspace_bytes={}",
                            heuristic.state, heuristic.workspaceSize
                        ),
                    });
                } else {
                    candidates.push(LtCandidate {
                        heuristic_index,
                        algorithm: heuristic.algo,
                        workspace_bytes: heuristic.workspaceSize,
                        waves: heuristic.wavesCount,
                    });
                }
            }
            Ok((candidates, rejections))
        }

        fn launch(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            candidate: LtCandidate,
            output_ptr: u64,
        ) -> Result<(), String> {
            unsafe {
                cudarc::cublaslt::result::matmul(
                    runtime.handle.raw,
                    self.descriptor.raw,
                    &self.spec.alpha as *const f32 as *const c_void,
                    &self.spec.beta as *const f32 as *const c_void,
                    self.a_ptr as *const c_void,
                    self.a_layout.raw,
                    self.b_ptr as *const c_void,
                    self.b_layout.raw,
                    output_ptr as *const c_void,
                    self.c_layout.raw,
                    output_ptr as *mut c_void,
                    self.d_layout.raw,
                    &candidate.algorithm,
                    runtime.workspace.cached_ptr() as *mut c_void,
                    candidate.workspace_bytes,
                    ctx.stream.cu_stream() as cudarc::cublaslt::sys::cudaStream_t,
                )
            }
            .map_err(|error| {
                format!(
                    "launch TN/NT cuBLASLt heuristic {}: {error:?}",
                    candidate.heuristic_index
                )
            })
        }
    }

    #[derive(Clone, Copy)]
    struct PairedIterations {
        custom: usize,
        vendor: usize,
    }

    #[derive(Clone, Copy)]
    struct PairConfig {
        kind: VendorPlanKind,
        iterations: PairedIterations,
    }

    struct OrderSamples {
        order: PairedOrder,
        custom_us: Vec<f64>,
        vendor_us: Vec<f64>,
    }

    impl OrderSamples {
        fn new(order: PairedOrder, capacity: usize) -> Self {
            Self {
                order,
                custom_us: Vec::with_capacity(capacity),
                vendor_us: Vec::with_capacity(capacity),
            }
        }

        fn push(&mut self, custom_us: f64, vendor_us: f64) -> Result<(), String> {
            let index = self.custom_us.len();
            validate_sample_us(custom_us, index)?;
            validate_sample_us(vendor_us, index)?;
            self.custom_us.push(custom_us);
            self.vendor_us.push(vendor_us);
            Ok(())
        }
    }

    struct Sm120PairedTnNtRunner {
        cell: Sm120PairedTnNtCell,
        contract: Sm120PairedContract,
        a: DtypedBuf,
        b: DtypedBuf,
        seed: DtypedBuf,
        custom_output: DtypedBuf,
        vendor_output: DtypedBuf,
        custom: Sm120PreparedLaunch,
        lt: Option<LtPlanSet>,
        lt_candidates: Vec<LtCandidate>,
        lt_rejections: Vec<LtRejection>,
        pilots: Vec<VendorPilot>,
        selected: VendorPlanKind,
    }

    impl Sm120PairedTnNtRunner {
        fn prepare(
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            cell: Sm120PairedTnNtCell,
        ) -> Result<Self, String> {
            let contract = sm120_paired_contract(cell)?;
            let probe = build_sm120_paired_semantic_probe(cell)?;
            let a = DtypedBuf::zeros(&ctx.stream, contract.a_elements, cell.dtype)?;
            let b = DtypedBuf::zeros(&ctx.stream, contract.b_elements, cell.dtype)?;
            let seed =
                DtypedBuf::zeros(&ctx.stream, contract.output_elements, contract.output_dtype)?;
            let custom_output =
                DtypedBuf::zeros(&ctx.stream, contract.output_elements, contract.output_dtype)?;
            let vendor_output =
                DtypedBuf::zeros(&ctx.stream, contract.output_elements, contract.output_dtype)?;
            a.upload_f32(&ctx.stream, &probe.a)?;
            b.upload_f32(&ctx.stream, &probe.b)?;
            seed.upload_f32(&ctx.stream, &probe.seed)?;
            let shape = Sm120Shape::contiguous(cell.op, cell.shape.dims);
            let route = Sm120ForcedRoute {
                op: cell.op,
                dtype: cell.dtype,
                physical: cell.physical,
                shape,
            };
            let maps = prepare_sm120_tensor_maps(
                &ctx.stream,
                &ctx.kernels,
                Sm120MapRequest {
                    op: cell.op,
                    dtype: cell.dtype,
                    tile: cell.physical.tile,
                    bk: cell.physical.bk,
                    a_ptr: a.cached_ptr(),
                    b_ptr: b.cached_ptr(),
                    shape,
                },
            )?;
            let custom = prepare_sm120_tma_forced(
                &ctx.stream,
                &ctx.kernels,
                route,
                &maps,
                Sm120LaunchOperands {
                    output_ptr: custom_output.cached_ptr(),
                    bias_ptr: contract.bias_ptr,
                    alpha: contract.alpha,
                    beta: contract.beta,
                },
            )?;
            let mut lt_rejections = Vec::new();
            let (lt, lt_candidates) = match sm120_paired_lt_descriptor(cell)
                .and_then(|spec| LtPlanSet::new(spec, b.cached_ptr(), a.cached_ptr()))
            {
                Ok(plan) => match plan.query_candidates(runtime) {
                    Ok((candidates, rejections)) => {
                        lt_rejections.extend(rejections);
                        (Some(plan), candidates)
                    }
                    Err(error) => {
                        lt_rejections.push(LtRejection {
                            heuristic_index: None,
                            reason: LtRejectionReason::Status,
                            detail: error,
                        });
                        (None, Vec::new())
                    }
                },
                Err(error) => {
                    lt_rejections.push(LtRejection {
                        heuristic_index: None,
                        reason: LtRejectionReason::Status,
                        detail: error,
                    });
                    (None, Vec::new())
                }
            };
            let mut runner = Self {
                cell,
                contract,
                a,
                b,
                seed,
                custom_output,
                vendor_output,
                custom,
                lt,
                lt_candidates,
                lt_rejections,
                pilots: Vec::new(),
                selected: VendorPlanKind::GemmEx,
            };
            runner.validate_semantics(ctx, runtime, &probe)?;
            runner.a.upload_f32(&ctx.stream, &probe.timed_a)?;
            runner.b.upload_f32(&ctx.stream, &probe.timed_b)?;
            runner.seed.upload_f32(&ctx.stream, &probe.timed_seed)?;
            runner.reset_both_outputs(ctx)?;
            runner.pilots = runner.collect_vendor_pilots(ctx, runtime)?;
            runner.selected = select_fastest_vendor_plan(&runner.pilots)?;
            Ok(runner)
        }

        fn copy_seed_to(&self, ctx: &GpuCtx, output: &DtypedBuf) -> Result<(), String> {
            if !self.contract.reset_before_window {
                return Ok(());
            }
            let status = unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    output.cached_ptr(),
                    self.seed.cached_ptr(),
                    self.seed.size_bytes(),
                    ctx.stream.cu_stream(),
                )
            };
            if status == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                Ok(())
            } else {
                Err(format!(
                    "reset paired TN output from immutable seed: {status:?}"
                ))
            }
        }

        fn reset_both_outputs(&self, ctx: &GpuCtx) -> Result<(), String> {
            self.copy_seed_to(ctx, &self.custom_output)?;
            self.copy_seed_to(ctx, &self.vendor_output)
        }

        fn launch_custom(&self, ctx: &GpuCtx) -> Result<(), String> {
            launch_sm120_tma_prepared(&ctx.stream, &ctx.kernels, &self.custom).map(|_| ())
        }

        fn launch_gemm_ex(&self, ctx: &GpuCtx) -> Result<(), String> {
            let geometry = sm120_paired_gemm_ex_geometry(self.cell);
            unsafe {
                cudarc::cublas::result::gemm_ex(
                    *ctx.blas.handle(),
                    geometry.trans_a,
                    geometry.trans_b,
                    geometry.m,
                    geometry.n,
                    geometry.k,
                    &self.contract.alpha as *const f32 as *const c_void,
                    self.b.cached_ptr() as *const c_void,
                    self.cell.dtype.cuda_data_type(),
                    geometry.lda,
                    self.a.cached_ptr() as *const c_void,
                    self.cell.dtype.cuda_data_type(),
                    geometry.ldb,
                    &self.contract.beta as *const f32 as *const c_void,
                    self.vendor_output.cached_ptr() as *mut c_void,
                    self.contract.output_dtype.cuda_data_type(),
                    geometry.ldc,
                    cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(|error| format!("launch paired GemmEx: {error:?}"))
        }

        fn lt_candidate(&self, heuristic_index: usize) -> Result<LtCandidate, String> {
            self.lt_candidates
                .iter()
                .copied()
                .find(|candidate| candidate.heuristic_index == heuristic_index)
                .ok_or_else(|| format!("unknown cuBLASLt heuristic {heuristic_index}"))
        }

        fn launch_vendor(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            kind: VendorPlanKind,
        ) -> Result<(), String> {
            match kind {
                VendorPlanKind::GemmEx => self.launch_gemm_ex(ctx),
                VendorPlanKind::CublasLt(heuristic_index) => {
                    let plan = self.lt.as_ref().ok_or("cuBLASLt plan is unavailable")?;
                    plan.launch(
                        ctx,
                        runtime,
                        self.lt_candidate(heuristic_index)?,
                        self.vendor_output.cached_ptr(),
                    )
                }
            }
        }

        fn download_output(&self, ctx: &GpuCtx, output: &DtypedBuf) -> Result<Vec<f32>, String> {
            let mut host = vec![0.0; self.contract.output_elements];
            output.download_f32(&ctx.stream, &mut host)?;
            Ok(host)
        }

        fn first_bit_difference(lhs: &[f32], rhs: &[f32]) -> Option<usize> {
            lhs.iter()
                .zip(rhs)
                .position(|(&left, &right)| left.to_bits() != right.to_bits())
        }

        fn validate_semantics(
            &mut self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            probe: &SemanticProbe,
        ) -> Result<(), String> {
            self.reset_both_outputs(ctx)?;
            self.launch_custom(ctx)?;
            let custom = self.download_output(ctx, &self.custom_output)?;
            if let Some(index) = Self::first_bit_difference(&custom, &probe.expected) {
                return Err(format!(
                    "{} custom semantic probe differs at {index}: got={} expected={}",
                    sm120_paired_tn_nt_cell_id(self.cell),
                    custom[index],
                    probe.expected[index]
                ));
            }
            self.copy_seed_to(ctx, &self.vendor_output)?;
            self.launch_gemm_ex(ctx)?;
            let gemm_ex = self.download_output(ctx, &self.vendor_output)?;
            if let Some(index) = Self::first_bit_difference(&custom, &gemm_ex) {
                return Err(format!(
                    "{} GemmEx semantic probe differs at {index}: custom={} vendor={}",
                    sm120_paired_tn_nt_cell_id(self.cell),
                    custom[index],
                    gemm_ex[index]
                ));
            }

            let candidates = self.lt_candidates.clone();
            let mut accepted = Vec::with_capacity(candidates.len());
            for candidate in candidates {
                self.copy_seed_to(ctx, &self.vendor_output)?;
                let smoke = self.lt.as_ref().ok_or("cuBLASLt plan disappeared")?.launch(
                    ctx,
                    runtime,
                    candidate,
                    self.vendor_output.cached_ptr(),
                );
                let smoke = smoke.and_then(|()| {
                    let output = self.download_output(ctx, &self.vendor_output)?;
                    if let Some(index) = Self::first_bit_difference(&custom, &output) {
                        Err(format!(
                            "bit mismatch at {index}: custom={} vendor={}",
                            custom[index], output[index]
                        ))
                    } else {
                        Ok(())
                    }
                });
                match smoke {
                    Ok(()) => accepted.push(candidate),
                    Err(error) => self.lt_rejections.push(LtRejection {
                        heuristic_index: Some(candidate.heuristic_index),
                        reason: LtRejectionReason::Smoke,
                        detail: error,
                    }),
                }
            }
            self.lt_candidates = accepted;
            Ok(())
        }

        fn enqueue_arm(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            custom: bool,
            kind: VendorPlanKind,
            iterations: usize,
        ) -> Result<(), String> {
            for _ in 0..iterations {
                if custom {
                    self.launch_custom(ctx)?;
                } else {
                    self.launch_vendor(ctx, runtime, kind)?;
                }
            }
            Ok(())
        }

        fn measure_arm_ms(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            custom: bool,
            kind: VendorPlanKind,
            iterations: usize,
        ) -> Result<f64, String> {
            if iterations == 0 {
                return Err("paired TN/NT iteration count must be positive".into());
            }
            if custom {
                self.copy_seed_to(ctx, &self.custom_output)?;
            } else {
                self.copy_seed_to(ctx, &self.vendor_output)?;
            }
            let start = ctx
                .stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("record paired TN/NT arm start: {error:?}"))?;
            self.enqueue_arm(ctx, runtime, custom, kind, iterations)?;
            let end = ctx
                .stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("record paired TN/NT arm end: {error:?}"))?;
            start
                .elapsed_ms(&end)
                .map(f64::from)
                .map_err(|error| format!("measure paired TN/NT arm: {error:?}"))
        }

        fn calibrate_kind(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            custom: bool,
            kind: VendorPlanKind,
        ) -> Result<usize, String> {
            calibrate_measurement(|iterations| {
                self.measure_arm_ms(ctx, runtime, custom, kind, iterations)
            })
        }

        fn measure_pair(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            config: PairConfig,
            order: PairedOrder,
        ) -> Result<(f64, f64), String> {
            self.reset_both_outputs(ctx)?;
            let first_custom = order == PairedOrder::CustomThenVendor;
            let first_iterations = if first_custom {
                config.iterations.custom
            } else {
                config.iterations.vendor
            };
            let second_iterations = if first_custom {
                config.iterations.vendor
            } else {
                config.iterations.custom
            };
            let first_start = ctx
                .stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("record first paired TN/NT start: {error:?}"))?;
            self.enqueue_arm(ctx, runtime, first_custom, config.kind, first_iterations)?;
            let first_end = ctx
                .stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("record first paired TN/NT end: {error:?}"))?;
            let second_start = ctx
                .stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("record second paired TN/NT start: {error:?}"))?;
            self.enqueue_arm(ctx, runtime, !first_custom, config.kind, second_iterations)?;
            let second_end = ctx
                .stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| format!("record second paired TN/NT end: {error:?}"))?;
            let first_us = first_start
                .elapsed_ms(&first_end)
                .map(f64::from)
                .map_err(|error| format!("measure first paired TN/NT arm: {error:?}"))?
                * 1000.0
                / first_iterations as f64;
            let second_us = second_start
                .elapsed_ms(&second_end)
                .map(f64::from)
                .map_err(|error| format!("measure second paired TN/NT arm: {error:?}"))?
                * 1000.0
                / second_iterations as f64;
            Ok(if first_custom {
                (first_us, second_us)
            } else {
                (second_us, first_us)
            })
        }

        fn collect_vendor_pilot(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            kind: VendorPlanKind,
            custom_iterations: usize,
        ) -> Result<VendorPilot, String> {
            let vendor_iterations = self.calibrate_kind(ctx, runtime, false, kind)?;
            let config = PairConfig {
                kind,
                iterations: PairedIterations {
                    custom: custom_iterations,
                    vendor: vendor_iterations,
                },
            };
            let mut custom_first = Vec::with_capacity(SM120_PAIRED_PILOT_WINDOWS_PER_ORDER);
            let mut vendor_first = Vec::with_capacity(SM120_PAIRED_PILOT_WINDOWS_PER_ORDER);
            for order in sm120_paired_pilot_schedule() {
                let (_, vendor_us) = self.measure_pair(ctx, runtime, config, order)?;
                validate_sample_us(vendor_us, custom_first.len() + vendor_first.len())?;
                match order {
                    PairedOrder::CustomThenVendor => custom_first.push(vendor_us),
                    PairedOrder::VendorThenCustom => vendor_first.push(vendor_us),
                }
            }
            Ok(VendorPilot {
                kind,
                custom_then_vendor_us: custom_first,
                vendor_then_custom_us: vendor_first,
            })
        }

        fn collect_vendor_pilots(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
        ) -> Result<Vec<VendorPilot>, String> {
            let custom_iterations =
                self.calibrate_kind(ctx, runtime, true, VendorPlanKind::GemmEx)?;
            let mut kinds = Vec::with_capacity(self.lt_candidates.len() + 1);
            kinds.push(VendorPlanKind::GemmEx);
            kinds.extend(
                self.lt_candidates
                    .iter()
                    .map(|candidate| VendorPlanKind::CublasLt(candidate.heuristic_index)),
            );
            kinds
                .into_iter()
                .map(|kind| self.collect_vendor_pilot(ctx, runtime, kind, custom_iterations))
                .collect()
        }

        fn warm_selected(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
        ) -> Result<(), String> {
            self.reset_both_outputs(ctx)?;
            for _ in 0..SM120_PAIRED_WARMUP_LAUNCHES {
                self.launch_custom(ctx)?;
            }
            for _ in 0..SM120_PAIRED_WARMUP_LAUNCHES {
                self.launch_vendor(ctx, runtime, self.selected)?;
            }
            ctx.stream
                .synchronize()
                .map_err(|error| format!("synchronize paired TN/NT warmups: {error:?}"))
        }

        fn calibrate_selected(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
        ) -> Result<PairedIterations, String> {
            Ok(PairedIterations {
                custom: self.calibrate_kind(ctx, runtime, true, self.selected)?,
                vendor: self.calibrate_kind(ctx, runtime, false, self.selected)?,
            })
        }

        fn collect_final_samples(
            &self,
            ctx: &GpuCtx,
            runtime: &Sm120PairedTnNtRuntime,
            iterations: PairedIterations,
        ) -> Result<[OrderSamples; 2], String> {
            let mut custom_first = OrderSamples::new(
                PairedOrder::CustomThenVendor,
                SM120_PAIRED_FINAL_WINDOWS_PER_ORDER,
            );
            let mut vendor_first = OrderSamples::new(
                PairedOrder::VendorThenCustom,
                SM120_PAIRED_FINAL_WINDOWS_PER_ORDER,
            );
            let config = PairConfig {
                kind: self.selected,
                iterations,
            };
            for order in sm120_paired_final_schedule() {
                let (custom_us, vendor_us) = self.measure_pair(ctx, runtime, config, order)?;
                match order {
                    PairedOrder::CustomThenVendor => custom_first.push(custom_us, vendor_us)?,
                    PairedOrder::VendorThenCustom => vendor_first.push(custom_us, vendor_us)?,
                }
            }
            Ok([custom_first, vendor_first])
        }
    }

    fn render_f64_samples(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn render_algorithm(algorithm: cudarc::cublaslt::sys::cublasLtMatmulAlgo_t) -> String {
        algorithm
            .data
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn json_escape(value: &str) -> String {
        let mut output = String::with_capacity(value.len());
        for character in value.chars() {
            match character {
                '"' => output.push_str("\\\""),
                '\\' => output.push_str("\\\\"),
                '\n' => output.push_str("\\n"),
                '\r' => output.push_str("\\r"),
                '\t' => output.push_str("\\t"),
                character if character.is_control() => {
                    write!(output, "\\u{:04x}", character as u32)
                        .expect("String writes cannot fail");
                }
                character => output.push(character),
            }
        }
        output
    }

    fn render_rejections(rejections: &[LtRejection]) -> String {
        rejections
            .iter()
            .map(|rejection| {
                let index = rejection
                    .heuristic_index
                    .map_or_else(|| "null".into(), |index| index.to_string());
                format!(
                    "{{\"heuristic_index\":{index},\"reason\":\"{:?}\",\"detail\":\"{}\"}}",
                    rejection.reason,
                    json_escape(&rejection.detail)
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn render_pilots(pilots: &[VendorPilot]) -> Result<String, String> {
        pilots
            .iter()
            .map(|pilot| {
                let api = match pilot.kind {
                    VendorPlanKind::GemmEx => "gemm_ex".to_string(),
                    VendorPlanKind::CublasLt(index) => format!("cublaslt:{index}"),
                };
                Ok(format!(
                    concat!(
                        "{{\"plan\":\"{}\",\"median_us\":{:.9},",
                        "\"custom_then_vendor_us\":[{}],",
                        "\"vendor_then_custom_us\":[{}]}}"
                    ),
                    api,
                    vendor_pilot_median(pilot)?,
                    render_f64_samples(&pilot.custom_then_vendor_us),
                    render_f64_samples(&pilot.vendor_then_custom_us),
                ))
            })
            .collect::<Result<Vec<_>, String>>()
            .map(|records| records.join(","))
    }

    fn operation_name(op: Sm120Op) -> &'static str {
        match op {
            Sm120Op::Nn => "nn",
            Sm120Op::Tn => "tn",
            Sm120Op::Nt => "nt",
        }
    }

    pub(super) fn sm120_paired_output_reset_name(op: Sm120Op) -> &'static str {
        match op {
            Sm120Op::Tn => "immutable_f32_seed_d2d_before_start_event",
            Sm120Op::Nt => "none_beta0",
            Sm120Op::Nn => "",
        }
    }

    fn operation_code(operation: cudarc::cublas::sys::cublasOperation_t) -> &'static str {
        use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
        match operation {
            CUBLAS_OP_N => "N",
            CUBLAS_OP_T => "T",
            _ => "unsupported",
        }
    }

    fn cuda_dtype_name(dtype: cudarc::cublaslt::sys::cudaDataType) -> &'static str {
        use cudarc::cublaslt::sys::cudaDataType_t::{CUDA_R_16BF, CUDA_R_16F, CUDA_R_32F};
        match dtype {
            CUDA_R_16BF => "CUDA_R_16BF",
            CUDA_R_16F => "CUDA_R_16F",
            CUDA_R_32F => "CUDA_R_32F",
            _ => "unsupported",
        }
    }

    fn render_layout(layout: LtMatrixLayoutSpec) -> String {
        format!(
            "{{\"dtype\":\"{}\",\"rows\":{},\"columns\":{},\"ld\":{}}}",
            cuda_dtype_name(layout.dtype),
            layout.rows,
            layout.columns,
            layout.ld
        )
    }

    struct RatioStats {
        p05: f64,
        p50: f64,
        p95: f64,
        median_interval_low: f64,
        median_interval_high: f64,
        min: f64,
        max: f64,
    }

    pub(super) fn sm120_paired_median_interval_95(ratios: &[f64]) -> Result<(f64, f64), String> {
        if ratios.len() != SM120_PAIRED_FINAL_WINDOWS_PER_ORDER
            || ratios.iter().any(|value| !value.is_finite())
        {
            return Err("median interval requires exactly 101 finite ratios".to_string());
        }
        let mut sorted = ratios.to_vec();
        sorted.sort_by(f64::total_cmp);
        Ok((sorted[40], sorted[60]))
    }

    pub(super) const fn sm120_paired_median_interval_coverage() -> f64 {
        0.953_955_933_070_657_2
    }

    fn ratio_stats(ratios: &[f64]) -> Result<RatioStats, String> {
        let p50 = sm120_paired_ratio_p50(ratios)?;
        let (median_interval_low, median_interval_high) = sm120_paired_median_interval_95(ratios)?;
        let mut sorted = ratios.to_vec();
        sorted.sort_by(f64::total_cmp);
        Ok(RatioStats {
            p05: percentile(&sorted, 0.05),
            p50,
            p95: percentile(&sorted, 0.95),
            median_interval_low,
            median_interval_high,
            min: sorted[0],
            max: sorted[100],
        })
    }

    fn emit_record(
        ctx: &GpuCtx,
        runner: &Sm120PairedTnNtRunner,
        iterations: PairedIterations,
        samples: &OrderSamples,
    ) -> Result<(), String> {
        let ratios = sm120_paired_ratios(&samples.custom_us, &samples.vendor_us)?;
        let stats = ratio_stats(&ratios)?;
        let descriptor = sm120_paired_lt_descriptor(runner.cell)?;
        let identity = runner.custom.identity();
        let resources = runner.custom.resources();
        let cc = ctx.stream.context().compute_capability().expect("CUDA CC");
        let (m, k, n) = runner.cell.shape.dims;
        let (api, heuristic_index, algorithm, workspace, waves) = match runner.selected {
            VendorPlanKind::GemmEx => (
                "gemm_ex",
                "null".to_string(),
                String::new(),
                0,
                "null".to_string(),
            ),
            VendorPlanKind::CublasLt(index) => {
                let candidate = runner.lt_candidate(index)?;
                (
                    "cublaslt",
                    index.to_string(),
                    render_algorithm(candidate.algorithm),
                    candidate.workspace_bytes,
                    format!("{:.9}", candidate.waves),
                )
            }
        };
        println!(
            concat!(
                "{{\"schema\":\"MambaBiSm120HalfTnNtPairedVendorV1\",",
                "\"suite\":\"gemm_bi_sm120_half_tn_nt_paired_vendor_hot\",",
                "\"scope\":\"performance_only\",\"cell_id\":\"{}\",",
                "\"comparator_order\":\"{}\",\"cc\":\"{}.{}\",",
                "\"rust_profile\":\"{}\",\"release_build\":{},",
                "\"dtype\":\"{}\",\"op\":\"{}\",\"shape\":\"{}\",",
                "\"m\":{},\"k\":{},\"n\":{},\"alpha\":1.0,\"beta\":{},",
                "\"bias\":false,\"input_dtype\":\"{}\",",
                "\"c_dtype\":\"{}\",\"d_dtype\":\"{}\",",
                "\"compute\":\"CUBLAS_COMPUTE_32F\",",
                "\"trans_a\":\"{}\",\"trans_b\":\"{}\",",
                "\"layout_a\":{},\"layout_b\":{},\"layout_c\":{},\"layout_d\":{},",
                "\"vendor_api\":\"{}\",\"vendor_algorithm\":\"{}\",",
                "\"vendor_heuristic_index\":{},\"vendor_algorithm_bytes\":[{}],",
                "\"vendor_workspace_bytes\":{},\"vendor_waves\":{},",
                "\"lt_heuristics_requested\":32,\"lt_rejections\":[{}],",
                "\"vendor_pilots\":[{}],\"vendor_selected_before_final\":true,",
                "\"semantic_probe\":\"one_active_term_full_output_bit_exact\",",
                "\"timed_product\":\"exactly_zero\",",
                "\"output_reset\":\"{}\",",
                "\"custom_symbol\":\"{}\",\"custom_tile\":\"{:?}\",",
                "\"custom_bk\":\"{:?}\",\"custom_stages\":\"{:?}\",",
                "\"custom_registers_per_thread\":{},\"custom_shared_bytes\":{},",
                "\"custom_iterations\":{},\"vendor_iterations\":{},",
                "\"target_window_ms\":{:.1},\"warmup_launches_per_arm\":{},",
                "\"windows\":{},\"ratio_orientation\":\"custom_over_vendor\",",
                "\"ratio_p05\":{:.9},\"ratio_p50\":{:.9},\"ratio_p95\":{:.9},",
                "\"ratio_median_interval_low\":{:.9},",
                "\"ratio_median_interval_high\":{:.9},",
                "\"ratio_min\":{:.9},\"ratio_max\":{:.9},",
                "\"custom_us\":[{}],\"vendor_us\":[{}],\"ratios\":[{}]}}"
            ),
            sm120_paired_tn_nt_cell_id(runner.cell),
            samples.order.as_str(),
            cc.0,
            cc.1,
            rust_profile(),
            !cfg!(debug_assertions),
            dtype_name(runner.cell.dtype),
            operation_name(runner.cell.op),
            runner.cell.shape.name,
            m,
            k,
            n,
            runner.contract.beta,
            cuda_dtype_name(descriptor.a.dtype),
            cuda_dtype_name(descriptor.c.dtype),
            cuda_dtype_name(descriptor.d.dtype),
            operation_code(descriptor.trans_a),
            operation_code(descriptor.trans_b),
            render_layout(descriptor.a),
            render_layout(descriptor.b),
            render_layout(descriptor.c),
            render_layout(descriptor.d),
            api,
            if api == "gemm_ex" {
                "CUBLAS_GEMM_DEFAULT"
            } else {
                "heuristic"
            },
            heuristic_index,
            algorithm,
            workspace,
            waves,
            render_rejections(&runner.lt_rejections),
            render_pilots(&runner.pilots)?,
            sm120_paired_output_reset_name(runner.cell.op),
            identity.symbol,
            identity.physical.tile,
            identity.physical.bk,
            identity.physical.stages,
            resources.registers_per_thread,
            resources.dynamic_shared_bytes,
            iterations.custom,
            iterations.vendor,
            TARGET_WINDOW_MS,
            SM120_PAIRED_WARMUP_LAUNCHES,
            samples.custom_us.len(),
            stats.p05,
            stats.p50,
            stats.p95,
            stats.median_interval_low,
            stats.median_interval_high,
            stats.min,
            stats.max,
            render_f64_samples(&samples.custom_us),
            render_f64_samples(&samples.vendor_us),
            render_f64_samples(&ratios),
        );
        Ok(())
    }

    pub(super) fn run_sm120_paired_tn_nt_cell(
        ctx: &GpuCtx,
        runtime: &Sm120PairedTnNtRuntime,
        cell: Sm120PairedTnNtCell,
    ) -> Result<(), String> {
        let runner = Sm120PairedTnNtRunner::prepare(ctx, runtime, cell)?;
        runner.warm_selected(ctx, runtime)?;
        let iterations = runner.calibrate_selected(ctx, runtime)?;
        let samples = runner.collect_final_samples(ctx, runtime, iterations)?;
        for order_samples in &samples {
            emit_record(ctx, &runner, iterations, order_samples)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrozenIterations {
    eager: usize,
    graph: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PathOrder {
    Ab,
    Ba,
}

impl PathOrder {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ab => "ab",
            Self::Ba => "ba",
        }
    }
}

#[derive(Clone, Copy)]
enum IterationSource {
    Calibrated,
    Frozen,
}

impl IterationSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Calibrated => "calibrated",
            Self::Frozen => "frozen",
        }
    }
}

struct RecordMeta<'a> {
    variant: &'a str,
    path_order: PathOrder,
    iteration_source: IterationSource,
    physical: &'a QualifiedPhysicalLaunchEvidence,
}

fn physical_kind_name(kind: PhysicalLaunchKind) -> &'static str {
    match kind {
        PhysicalLaunchKind::Gemm => "gemm",
        PhysicalLaunchKind::InputUpcast => "input_upcast",
        PhysicalLaunchKind::OutputDowncast => "output_downcast",
    }
}

fn module_kind_name(kind: ModuleKind) -> &'static str {
    match kind {
        ModuleKind::Fixed => "fixed",
        ModuleKind::TriadScalar => "triad_scalar",
        ModuleKind::TriadSm80 => "triad_sm80",
        ModuleKind::TriadSm90a => "triad_sm90a",
        ModuleKind::TriadSm100 => "triad_sm100",
        ModuleKind::TriadSm120 => "triad_sm120",
        ModuleKind::Mamba3Combined => "mamba3_combined",
    }
}

fn policy_dtype_name(dtype: PolicyDtype) -> &'static str {
    match dtype {
        PolicyDtype::F32 => "f32",
        PolicyDtype::Bf16 => "bf16",
        PolicyDtype::F16 => "f16",
    }
}

fn render_physical_node(node: &QualifiedPhysicalLaunchNode) -> String {
    let tile = node.tile.map_or_else(
        || "null".into(),
        |(rows, columns)| format!("[{rows},{columns}]"),
    );
    format!(
        concat!(
            "{{\"kind\":\"{}\",\"symbol\":\"{}\",\"module_owner\":\"{}\",",
            "\"logical_op\":\"{}\",\"logical_dtype\":\"{}\",",
            "\"execution_dtype\":\"{}\",\"shape\":[{},{},{}],",
            "\"strides\":[{},{},{}],\"tile\":{},",
            "\"grid\":[{},{},{}],\"block\":[{},{},{}],",
            "\"shared_mem_bytes\":{},\"arguments_digest\":\"{}\"}}"
        ),
        physical_kind_name(node.kind),
        node.symbol,
        module_kind_name(node.module_kind),
        op_name(node.logical_op),
        policy_dtype_name(node.logical_dtype),
        policy_dtype_name(node.execution_dtype),
        node.shape.0,
        node.shape.1,
        node.shape.2,
        node.strides.0,
        node.strides.1,
        node.strides.2,
        tile,
        node.launch.grid_dim.0,
        node.launch.grid_dim.1,
        node.launch.grid_dim.2,
        node.launch.block_dim.0,
        node.launch.block_dim.1,
        node.launch.block_dim.2,
        node.launch.shared_mem_bytes,
        hex_digest(node.launch.arguments_digest),
    )
}

fn render_physical_evidence_fields(evidence: &QualifiedPhysicalLaunchEvidence) -> String {
    let nodes = evidence
        .nodes()
        .iter()
        .map(render_physical_node)
        .collect::<Vec<_>>()
        .join(",");
    let physical_symbol = evidence
        .single_launch_symbol()
        .map_or_else(|| "null".into(), |symbol| format!("\"{symbol}\""));
    let tile = evidence.single_launch_tile().map_or_else(
        || "null".into(),
        |(rows, columns)| format!("[{rows},{columns}]"),
    );
    let execution_dtype = evidence.uniform_execution_dtype().map_or_else(
        || "null".into(),
        |dtype| format!("\"{}\"", policy_dtype_name(dtype)),
    );
    let module_owner = evidence.uniform_module_kind().map_or_else(
        || "null".into(),
        |kind| format!("\"{}\"", module_kind_name(kind)),
    );
    format!(
        concat!(
            "\"evidence_scope\":\"{}\",\"eager_graph_equal\":{},",
            "\"timed_request_digest\":\"{}\",",
            "\"physical_launch_count\":{},\"physical_launch_digest\":\"{}\",",
            "\"physical_symbol\":{},\"physical_tile\":{},",
            "\"physical_execution_dtype\":{},\"physical_module_owner\":{},",
            "\"physical_nodes\":[{}]"
        ),
        evidence.evidence_scope(),
        evidence.eager_graph_equal(),
        hex_digest(evidence.request_identity_digest()),
        evidence.launch_count(),
        hex_digest(evidence.launch_digest()),
        physical_symbol,
        tile,
        execution_dtype,
        module_owner,
        nodes,
    )
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut rendered = String::with_capacity(64);
    for byte in digest {
        write!(rendered, "{byte:02x}").expect("String writes cannot fail");
    }
    rendered
}

fn escape_json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                write!(output, "\\u{:04x}", character as u32).expect("String writes cannot fail");
            }
            character => output.push(character),
        }
    }
    output
}

fn op_name(op: ResolvedGemmOp) -> &'static str {
    match op {
        ResolvedGemmOp::Nn => "nn",
        ResolvedGemmOp::Tn => "tn",
        ResolvedGemmOp::Nt => "nt",
    }
}

fn dtype_name(dtype: WeightDtype) -> &'static str {
    match dtype {
        WeightDtype::F32 => "f32",
        WeightDtype::Bf16 => "bf16",
        WeightDtype::F16 => "f16",
    }
}

fn tile_name(tile: TcTile) -> &'static str {
    match tile {
        TcTile::Tile128 => "tile128",
        TcTile::Tile64 => "tile64",
        TcTile::Thin16 => "thin16",
        TcTile::Rect128x64 => "rect128x64",
    }
}

fn logical_route_name(route: Route) -> String {
    match route {
        Route::F32Policy {
            policy: F32TriadPolicy::ExactScalarFmaV1,
        } => "f32_policy_exact".into(),
        Route::F32Policy {
            policy: F32TriadPolicy::AllowDeterministicTf32V1,
        } => "f32_policy_allow_tf32".into(),
        Route::HalfPolicy {
            dtype,
            tensor_cores,
        } => format!(
            "{}_policy_{}",
            dtype_name(dtype),
            if tensor_cores { "tc" } else { "scalar" }
        ),
        Route::HalfForced { dtype, tile } => {
            format!("{}_forced_{}", dtype_name(dtype), tile_name(tile))
        }
        Route::Tf32Forced(spec) => tf32_logical_route_name(spec),
    }
}

fn tf32_logical_route_name(spec: &Tf32KernelSpec) -> String {
    format!(
        "f32_tf32_forced_sm80_mma_tf32_v1_m{}n{}_bk{}_s{}",
        spec.tile.0, spec.tile.1, spec.bk, spec.stages
    )
}

fn render_route_identity_fields(route: Route) -> String {
    format!("\"route\":\"{}\"", logical_route_name(route))
}

fn call_scope(route: Route) -> &'static str {
    match route {
        Route::F32Policy { .. } => "production_cached",
        Route::HalfPolicy { .. } => "production_policy",
        Route::HalfForced { .. } => "direct_forced",
        Route::Tf32Forced(_) => "prepared_forced",
    }
}

fn cell_id(cell: Cell) -> String {
    format!(
        "{}/{}/{}/{}",
        logical_route_name(cell.route),
        op_name(cell.op),
        cell.shape.name,
        cell.stride_class.as_str()
    )
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn parse_qualification_cell_ids(value: &str, cells: &[Cell]) -> Result<BTreeSet<String>, String> {
    if value.is_empty() {
        return Err(format!("{QUALIFICATION_CELL_IDS_ENV} must not be empty"));
    }
    if value.chars().any(char::is_whitespace) {
        return Err(format!(
            "{QUALIFICATION_CELL_IDS_ENV} must not contain whitespace"
        ));
    }

    let canonical = cells.iter().copied().map(cell_id).collect::<BTreeSet<_>>();
    let mut requested = BTreeSet::new();
    for id in value.split(',') {
        if id.is_empty() {
            return Err(format!(
                "{QUALIFICATION_CELL_IDS_ENV} must contain comma-separated non-empty cell IDs"
            ));
        }
        if !canonical.contains(id) {
            return Err(format!(
                "{QUALIFICATION_CELL_IDS_ENV} contains unknown cell ID {id:?}"
            ));
        }
        if !requested.insert(id.to_string()) {
            return Err(format!(
                "{QUALIFICATION_CELL_IDS_ENV} contains duplicate cell ID {id:?}"
            ));
        }
    }
    Ok(requested)
}

fn load_qualification_cell_ids(cells: &[Cell]) -> Result<Option<BTreeSet<String>>, String> {
    match std::env::var(QUALIFICATION_CELL_IDS_ENV) {
        Ok(value) => parse_qualification_cell_ids(&value, cells).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{QUALIFICATION_CELL_IDS_ENV} must be valid Unicode"
        )),
    }
}

fn select_qualification_cells(
    cells: &[Cell],
    requested: Option<&BTreeSet<String>>,
    shard: usize,
    shards: usize,
) -> Result<Vec<Cell>, String> {
    if shards == 0 || shard >= shards {
        return Err(format!("invalid shard {shard}/{shards}"));
    }
    let selected = cells
        .iter()
        .copied()
        .filter(|cell| {
            let id = cell_id(*cell);
            requested.is_none_or(|ids| ids.contains(&id))
                && stable_hash(&id) % shards as u64 == shard as u64
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(format!("shard {shard}/{shards} selected no cells"));
    }
    Ok(selected)
}

fn parse_edge_qualification_cell_ids(
    value: &str,
    cells: &[EdgeCell],
) -> Result<BTreeSet<String>, String> {
    if value.is_empty() {
        return Err(format!(
            "{EDGE_QUALIFICATION_CELL_IDS_ENV} must not be empty"
        ));
    }
    if value.chars().any(char::is_whitespace) {
        return Err(format!(
            "{EDGE_QUALIFICATION_CELL_IDS_ENV} must not contain whitespace"
        ));
    }
    let inventory = cells
        .iter()
        .copied()
        .map(edge_cell_id)
        .collect::<BTreeSet<_>>();
    let mut requested = BTreeSet::new();
    for id in value.split(',') {
        if id.is_empty() {
            return Err(format!(
                "{EDGE_QUALIFICATION_CELL_IDS_ENV} must contain comma-separated non-empty cell IDs"
            ));
        }
        if !inventory.contains(id) {
            return Err(format!(
                "{EDGE_QUALIFICATION_CELL_IDS_ENV} contains unknown cell ID {id:?}"
            ));
        }
        if !requested.insert(id.to_string()) {
            return Err(format!(
                "{EDGE_QUALIFICATION_CELL_IDS_ENV} contains duplicate cell ID {id:?}"
            ));
        }
    }
    Ok(requested)
}

fn load_edge_qualification_cell_ids(
    cells: &[EdgeCell],
) -> Result<Option<BTreeSet<String>>, String> {
    match std::env::var(EDGE_QUALIFICATION_CELL_IDS_ENV) {
        Ok(value) => parse_edge_qualification_cell_ids(&value, cells).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{EDGE_QUALIFICATION_CELL_IDS_ENV} must be valid Unicode"
        )),
    }
}

fn select_edge_qualification_cells(
    cells: &[EdgeCell],
    requested: Option<&BTreeSet<String>>,
    shard: usize,
    shards: usize,
) -> Result<Vec<EdgeCell>, String> {
    if shards == 0 || shard >= shards {
        return Err(format!("invalid edge shard {shard}/{shards}"));
    }
    let selected = cells
        .iter()
        .copied()
        .filter(|cell| {
            let id = edge_cell_id(*cell);
            requested.is_none_or(|ids| ids.contains(&id))
                && stable_hash(&id) % shards as u64 == shard as u64
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(format!("edge shard {shard}/{shards} selected no cells"));
    }
    Ok(selected)
}

fn parse_env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|_| format!("{name} must be an unsigned integer")),
        Err(_) => Ok(default),
    }
}

fn parse_path_order(value: &str) -> Result<PathOrder, String> {
    match value {
        "ab" => Ok(PathOrder::Ab),
        "ba" => Ok(PathOrder::Ba),
        _ => Err("GEMM_BI_QUAL_PATH_ORDER must be exactly ab or ba".into()),
    }
}

fn validate_variant(value: &str) -> Result<(), String> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(
            "GEMM_BI_QUAL_VARIANT must be a non-empty ASCII identifier using [A-Za-z0-9._-]".into(),
        );
    }
    Ok(())
}

fn validate_sample_us(value: f64, index: usize) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!(
            "performance sample {index} was not positive and finite: {value}"
        ))
    }
}

fn validate_run_profile(windows: usize, debug_build: bool) -> Result<(), String> {
    if windows >= 100 && debug_build {
        return Err(
            "performance runs with at least 100 windows require cargo test --release".into(),
        );
    }
    Ok(())
}

fn rust_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn validate_canonical_inventory(cells: &[Cell]) -> Result<(), String> {
    if cells.len() != CANONICAL_INVENTORY_COUNT {
        return Err(format!(
            "canonical inventory count changed: expected {CANONICAL_INVENTORY_COUNT}, got {}",
            cells.len()
        ));
    }
    let digest = hex_digest(canonical_inventory_digest(cells));
    if digest != CANONICAL_INVENTORY_DIGEST {
        return Err(format!(
            "canonical inventory digest changed: expected {CANONICAL_INVENTORY_DIGEST}, got {digest}"
        ));
    }
    Ok(())
}

fn parse_frozen_iterations(
    input: &str,
    cells: &[Cell],
) -> Result<BTreeMap<String, FrozenIterations>, String> {
    validate_canonical_inventory(cells)?;
    let mut lines = input.lines();
    for (line_number, expected) in [
        format!("format\t{FROZEN_ITERATION_FORMAT}"),
        format!("suite\t{FROZEN_ITERATION_SUITE}"),
        format!("inventory_count\t{CANONICAL_INVENTORY_COUNT}"),
        format!("inventory_digest\t{CANONICAL_INVENTORY_DIGEST}"),
    ]
    .into_iter()
    .enumerate()
    {
        let actual = lines.next().ok_or_else(|| {
            format!(
                "frozen iteration manifest is missing metadata line {}",
                line_number + 1
            )
        })?;
        if actual != expected {
            return Err(format!(
                "frozen iteration manifest line {} must be {expected:?}, got {actual:?}",
                line_number + 1
            ));
        }
    }
    if lines.next() != Some(FROZEN_ITERATION_HEADER) {
        return Err(format!(
            "frozen iteration manifest must start with {FROZEN_ITERATION_HEADER:?}"
        ));
    }
    let mut entries = BTreeMap::new();
    for (index, cell) in cells.iter().copied().enumerate() {
        let line_number = index + 6;
        let line = lines.next().ok_or_else(|| {
            format!(
                "frozen iteration manifest is missing canonical cell {index}: {}",
                cell_id(cell)
            )
        })?;
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 || fields[0].is_empty() {
            return Err(format!(
                "frozen iteration manifest line {line_number} must have three non-empty fields"
            ));
        }
        let expected_id = cell_id(cell);
        if fields[0] != expected_id {
            return Err(format!(
                "frozen iteration manifest canonical cell {index} must be {expected_id:?}, got {:?}",
                fields[0]
            ));
        }
        let eager = fields[1]
            .parse::<usize>()
            .map_err(|_| format!("invalid eager iterations on line {line_number}"))?;
        let graph = fields[2]
            .parse::<usize>()
            .map_err(|_| format!("invalid graph iterations on line {line_number}"))?;
        if eager == 0 || graph == 0 {
            return Err(format!(
                "frozen iterations must be positive on line {line_number}"
            ));
        }
        if entries
            .insert(fields[0].to_string(), FrozenIterations { eager, graph })
            .is_some()
        {
            return Err(format!(
                "duplicate frozen iteration cell {} on line {line_number}",
                fields[0]
            ));
        }
    }
    if let Some(extra) = lines.next() {
        return Err(format!(
            "frozen iteration manifest has an extra row after the canonical inventory: {extra:?}"
        ));
    }
    Ok(entries)
}

fn render_frozen_iterations(
    entries: &BTreeMap<String, FrozenIterations>,
    cells: &[Cell],
) -> Result<String, String> {
    validate_canonical_inventory(cells)?;
    if entries.len() != cells.len() {
        return Err(format!(
            "frozen iteration output requires all {} canonical cells, got {}",
            cells.len(),
            entries.len()
        ));
    }
    let mut output = format!(
        "format\t{FROZEN_ITERATION_FORMAT}\nsuite\t{FROZEN_ITERATION_SUITE}\ninventory_count\t{CANONICAL_INVENTORY_COUNT}\ninventory_digest\t{CANONICAL_INVENTORY_DIGEST}\n{FROZEN_ITERATION_HEADER}"
    );
    output.push('\n');
    for cell in cells.iter().copied() {
        let id = cell_id(cell);
        let iterations = entries
            .get(&id)
            .ok_or_else(|| format!("frozen iteration output has no canonical cell {id}"))?;
        if iterations.eager == 0 || iterations.graph == 0 {
            return Err(format!(
                "frozen iteration output has zero iterations for {id}"
            ));
        }
        writeln!(output, "{id}\t{}\t{}", iterations.eager, iterations.graph)
            .expect("String writes cannot fail");
    }
    Ok(output)
}

fn parse_edge_frozen_iterations(
    input: &str,
    cells: &[EdgeCell],
) -> Result<BTreeMap<String, FrozenIterations>, String> {
    validate_edge_inventory(cells)?;
    let mut lines = input.lines();
    for (line_number, expected) in [
        format!("format\t{FROZEN_ITERATION_FORMAT}"),
        format!("suite\t{EDGE_FROZEN_ITERATION_SUITE}"),
        format!("inventory_count\t{EDGE_INVENTORY_COUNT}"),
        format!("inventory_digest\t{EDGE_INVENTORY_DIGEST}"),
    ]
    .into_iter()
    .enumerate()
    {
        let actual = lines.next().ok_or_else(|| {
            format!(
                "edge frozen iteration manifest is missing metadata line {}",
                line_number + 1
            )
        })?;
        if actual != expected {
            return Err(format!(
                "edge frozen iteration manifest line {} must be {expected:?}, got {actual:?}",
                line_number + 1
            ));
        }
    }
    if lines.next() != Some(FROZEN_ITERATION_HEADER) {
        return Err(format!(
            "edge frozen iteration manifest must start with {FROZEN_ITERATION_HEADER:?}"
        ));
    }
    let mut entries = BTreeMap::new();
    for (index, cell) in cells.iter().copied().enumerate() {
        let line_number = index + 6;
        let line = lines.next().ok_or_else(|| {
            format!(
                "edge frozen iteration manifest is missing cell {index}: {}",
                edge_cell_id(cell)
            )
        })?;
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 || fields[0].is_empty() {
            return Err(format!(
                "edge frozen iteration manifest line {line_number} must have three non-empty fields"
            ));
        }
        let expected_id = edge_cell_id(cell);
        if fields[0] != expected_id {
            return Err(format!(
                "edge frozen iteration manifest cell {index} must be {expected_id:?}, got {:?}",
                fields[0]
            ));
        }
        let eager = fields[1]
            .parse::<usize>()
            .map_err(|_| format!("invalid eager iterations on line {line_number}"))?;
        let graph = fields[2]
            .parse::<usize>()
            .map_err(|_| format!("invalid graph iterations on line {line_number}"))?;
        if eager == 0 || graph == 0 {
            return Err(format!(
                "edge frozen iterations must be positive on line {line_number}"
            ));
        }
        if entries
            .insert(fields[0].to_string(), FrozenIterations { eager, graph })
            .is_some()
        {
            return Err(format!(
                "duplicate edge frozen iteration cell {} on line {line_number}",
                fields[0]
            ));
        }
    }
    if let Some(extra) = lines.next() {
        return Err(format!(
            "edge frozen iteration manifest has an extra row after the inventory: {extra:?}"
        ));
    }
    Ok(entries)
}

fn render_edge_frozen_iterations(
    entries: &BTreeMap<String, FrozenIterations>,
    cells: &[EdgeCell],
) -> Result<String, String> {
    validate_edge_inventory(cells)?;
    if entries.len() != cells.len() {
        return Err(format!(
            "edge frozen iteration output requires all {} cells, got {}",
            cells.len(),
            entries.len()
        ));
    }
    let mut output = format!(
        "format\t{FROZEN_ITERATION_FORMAT}\nsuite\t{EDGE_FROZEN_ITERATION_SUITE}\ninventory_count\t{EDGE_INVENTORY_COUNT}\ninventory_digest\t{EDGE_INVENTORY_DIGEST}\n{FROZEN_ITERATION_HEADER}"
    );
    output.push('\n');
    for cell in cells.iter().copied() {
        let id = edge_cell_id(cell);
        let iterations = entries
            .get(&id)
            .ok_or_else(|| format!("edge frozen iteration output has no cell {id}"))?;
        if iterations.eager == 0 || iterations.graph == 0 {
            return Err(format!(
                "edge frozen iteration output has zero iterations for {id}"
            ));
        }
        writeln!(output, "{id}\t{}\t{}", iterations.eager, iterations.graph)
            .expect("String writes cannot fail");
    }
    Ok(output)
}

fn load_frozen_iterations(
    cells: &[Cell],
) -> Result<Option<BTreeMap<String, FrozenIterations>>, String> {
    let path = match std::env::var("GEMM_BI_QUAL_ITERS_IN") {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    let input = std::fs::read_to_string(&path)
        .map_err(|error| format!("read frozen iterations {path}: {error}"))?;
    parse_frozen_iterations(&input, cells).map(Some)
}

fn load_edge_frozen_iterations(
    cells: &[EdgeCell],
) -> Result<Option<BTreeMap<String, FrozenIterations>>, String> {
    let path = match std::env::var("GEMM_BI_QUAL_ITERS_IN") {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    let input = std::fs::read_to_string(&path)
        .map_err(|error| format!("read edge frozen iterations {path}: {error}"))?;
    parse_edge_frozen_iterations(&input, cells).map(Some)
}

fn build_cublas_denominator_cells() -> Vec<CublasDenominatorCell> {
    let mut cells = Vec::with_capacity(SHAPES.len() * 3 * 3);
    for shape in SHAPES {
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
                cells.push(CublasDenominatorCell { dtype, op, shape });
            }
        }
    }
    cells
}

fn cublas_denominator_cell_id(cell: CublasDenominatorCell) -> String {
    format!(
        "cublas/{}/{}/{}",
        dtype_name(cell.dtype),
        op_name(cell.op),
        cell.shape.name
    )
}

fn parse_cublas_denominator_cell_ids(
    value: &str,
    cells: &[CublasDenominatorCell],
) -> Result<BTreeSet<String>, String> {
    if value.is_empty() {
        return Err(format!(
            "{CUBLAS_DENOMINATOR_CELL_IDS_ENV} must not be empty"
        ));
    }
    if value.chars().any(char::is_whitespace) {
        return Err(format!(
            "{CUBLAS_DENOMINATOR_CELL_IDS_ENV} must not contain whitespace"
        ));
    }
    let inventory = cells
        .iter()
        .copied()
        .map(cublas_denominator_cell_id)
        .collect::<BTreeSet<_>>();
    let mut requested = BTreeSet::new();
    for id in value.split(',') {
        if id.is_empty() {
            return Err(format!(
                "{CUBLAS_DENOMINATOR_CELL_IDS_ENV} must contain comma-separated non-empty cell IDs"
            ));
        }
        if !inventory.contains(id) {
            return Err(format!(
                "{CUBLAS_DENOMINATOR_CELL_IDS_ENV} contains unknown cell ID {id:?}"
            ));
        }
        if !requested.insert(id.to_string()) {
            return Err(format!(
                "{CUBLAS_DENOMINATOR_CELL_IDS_ENV} contains duplicate cell ID {id:?}"
            ));
        }
    }
    Ok(requested)
}

fn load_cublas_denominator_cell_ids(
    cells: &[CublasDenominatorCell],
) -> Result<Option<BTreeSet<String>>, String> {
    match std::env::var(CUBLAS_DENOMINATOR_CELL_IDS_ENV) {
        Ok(value) => parse_cublas_denominator_cell_ids(&value, cells).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{CUBLAS_DENOMINATOR_CELL_IDS_ENV} must be valid Unicode"
        )),
    }
}

fn select_cublas_denominator_cells(
    cells: &[CublasDenominatorCell],
    requested: Option<&BTreeSet<String>>,
    shard: usize,
    shards: usize,
) -> Result<Vec<CublasDenominatorCell>, String> {
    if shards == 0 || shard >= shards {
        return Err(format!("invalid cuBLAS denominator shard {shard}/{shards}"));
    }
    let selected = cells
        .iter()
        .copied()
        .filter(|cell| {
            let id = cublas_denominator_cell_id(*cell);
            requested.is_none_or(|ids| ids.contains(&id))
                && stable_hash(&id) % shards as u64 == shard as u64
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(format!(
            "cuBLAS denominator shard {shard}/{shards} selected no cells"
        ));
    }
    Ok(selected)
}

fn cublas_denominator_output_dtype(dtype: WeightDtype, op: ResolvedGemmOp) -> WeightDtype {
    if op == ResolvedGemmOp::Tn {
        WeightDtype::F32
    } else {
        dtype
    }
}

fn cublas_denominator_compute(
    dtype: WeightDtype,
    mode: CublasDenominatorMode,
) -> cudarc::cublas::sys::cublasComputeType_t {
    use cudarc::cublas::sys::cublasComputeType_t;

    match (dtype, mode) {
        (WeightDtype::F32, CublasDenominatorMode::Fast) => {
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        }
        (WeightDtype::Bf16 | WeightDtype::F16, CublasDenominatorMode::Fast) => {
            cublasComputeType_t::CUBLAS_COMPUTE_32F
        }
        (_, CublasDenominatorMode::Pedantic) => cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
    }
}

fn cublas_denominator_compute_name(
    dtype: WeightDtype,
    mode: CublasDenominatorMode,
) -> &'static str {
    match (dtype, mode) {
        (WeightDtype::F32, CublasDenominatorMode::Fast) => "32f_fast_tf32",
        (WeightDtype::Bf16 | WeightDtype::F16, CublasDenominatorMode::Fast) => "32f",
        (_, CublasDenominatorMode::Pedantic) => "32f_pedantic",
    }
}

fn cublas_denominator_geometry(
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
) -> CublasDenominatorGeometry {
    use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};

    let (m, k, n) = dims;
    let as_int = |value: usize| c_int::try_from(value).expect("cuBLAS dimension fits c_int");
    match op {
        ResolvedGemmOp::Nn => CublasDenominatorGeometry {
            trans_a: CUBLAS_OP_N,
            trans_b: CUBLAS_OP_N,
            m: as_int(n),
            n: as_int(m),
            k: as_int(k),
            lda: as_int(n),
            ldb: as_int(k),
            ldc: as_int(n),
        },
        ResolvedGemmOp::Tn => CublasDenominatorGeometry {
            trans_a: CUBLAS_OP_N,
            trans_b: CUBLAS_OP_T,
            m: as_int(n),
            n: as_int(k),
            k: as_int(m),
            lda: as_int(n),
            ldb: as_int(k),
            ldc: as_int(n),
        },
        ResolvedGemmOp::Nt => CublasDenominatorGeometry {
            trans_a: CUBLAS_OP_T,
            trans_b: CUBLAS_OP_N,
            m: as_int(k),
            n: as_int(m),
            k: as_int(n),
            lda: as_int(n),
            ldb: as_int(n),
            ldc: as_int(k),
        },
    }
}

fn build_cells() -> Vec<Cell> {
    let mut cells = Vec::new();
    for shape in SHAPES {
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            cells.push(Cell {
                route: Route::F32Policy {
                    policy: F32TriadPolicy::ExactScalarFmaV1,
                },
                op,
                shape,
                stride_class: StrideClass::Contiguous,
            });
            cells.push(Cell {
                route: Route::F32Policy {
                    policy: F32TriadPolicy::AllowDeterministicTf32V1,
                },
                op,
                shape,
                stride_class: StrideClass::Contiguous,
            });
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                for tensor_cores in [false, true] {
                    cells.push(Cell {
                        route: Route::HalfPolicy {
                            dtype,
                            tensor_cores,
                        },
                        op,
                        shape,
                        stride_class: StrideClass::Contiguous,
                    });
                }
            }
        }
    }

    for shape in SHAPES
        .into_iter()
        .filter(|shape| FORCED_SQUARE_SHAPES.contains(&shape.name))
    {
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                for tile in [TcTile::Tile64, TcTile::Tile128] {
                    cells.push(Cell {
                        route: Route::HalfForced { dtype, tile },
                        op,
                        shape,
                        stride_class: StrideClass::Contiguous,
                    });
                }
            }
        }
    }
    for shape in SHAPES
        .into_iter()
        .filter(|shape| FORCED_THIN_SHAPES.contains(&shape.name))
    {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            cells.push(Cell {
                route: Route::HalfForced {
                    dtype,
                    tile: TcTile::Thin16,
                },
                op: ResolvedGemmOp::Nn,
                shape,
                stride_class: StrideClass::Contiguous,
            });
        }
    }

    for spec in &SM80_TF32_ROUTE_SPECS {
        for shape_name in tf32_shape_names(spec) {
            let shape = SHAPES
                .into_iter()
                .find(|shape| shape.name == shape_name)
                .expect("TF32 performance shape");
            cells.push(Cell {
                route: Route::Tf32Forced(spec),
                op: spec.op,
                shape,
                stride_class: StrideClass::Contiguous,
            });
        }
    }
    cells
}

fn expected_cell_count() -> usize {
    let op_count = 3;
    let policy_cells = SHAPES.len() * op_count * (2 + 2 * 2);
    let forced_square_cells = SHAPES
        .iter()
        .filter(|shape| FORCED_SQUARE_SHAPES.contains(&shape.name))
        .count()
        * op_count
        * 2
        * 2;
    let forced_thin_cells = SHAPES
        .iter()
        .filter(|shape| FORCED_THIN_SHAPES.contains(&shape.name))
        .count()
        * 2;
    let forced_tf32_cells = SM80_TF32_ROUTE_SPECS
        .iter()
        .map(|spec| tf32_shape_names(spec).count())
        .sum::<usize>();
    policy_cells + forced_square_cells + forced_thin_cells + forced_tf32_cells
}

fn canonical_inventory_digest(cells: &[Cell]) -> [u8; 32] {
    let mut digest = FramedSha256::new(b"gemm-bi-canonical-performance-inventory.v1")
        .required(b"cell-count", &(cells.len() as u64).to_le_bytes());
    for (index, cell) in cells.iter().copied().enumerate() {
        let id = cell_id(cell);
        digest = digest
            .required(b"cell-index", &(index as u64).to_le_bytes())
            .required(b"cell-id", id.as_bytes());
    }
    digest.finish()
}

fn half_edge_routes() -> Vec<Route> {
    let mut routes = Vec::with_capacity(8);
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        routes.extend([
            Route::HalfPolicy {
                dtype,
                tensor_cores: true,
            },
            Route::HalfPolicy {
                dtype,
                tensor_cores: false,
            },
            Route::HalfForced {
                dtype,
                tile: TcTile::Tile64,
            },
            Route::HalfForced {
                dtype,
                tile: TcTile::Tile128,
            },
        ]);
    }
    routes
}

fn boundary_edge_routes() -> Vec<Route> {
    let mut routes = Vec::with_capacity(9);
    routes.push(Route::F32Policy {
        policy: F32TriadPolicy::ExactScalarFmaV1,
    });
    routes.extend(half_edge_routes());
    routes
}

fn edge_shape(dims: (usize, usize, usize)) -> Shape {
    Shape { name: "edge", dims }
}

fn build_edge_cells() -> Vec<EdgeCell> {
    let mut cells = Vec::with_capacity(EDGE_INVENTORY_COUNT);
    let boundary_shapes = [
        (31, 32, 32),
        (32, 32, 32),
        (128, 32, 1),
        (128, 32, 2),
        (128, 32, 127),
        (128, 32, 128),
        (128, 33, 128),
        (127, 513, 16_384),
        (128, 513, 16_384),
        (255, 16, 256),
        (256, 16, 256),
        (16, 192, 256),
        (16, 256, 256),
        (128, 512, 832),
        (129, 512, 576),
    ];
    for dims in boundary_shapes {
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            for route in boundary_edge_routes() {
                cells.push(EdgeCell {
                    cohort: EdgeCohort::Boundary,
                    route,
                    op,
                    shape: edge_shape(dims),
                    layout: EdgeLayout::Contiguous,
                });
            }
        }
    }

    for dims in [
        (127, 8192, 128),
        (128, 4096, 128),
        (128, 8191, 128),
        (128, 8192, 128),
        (128, 8193, 128),
        (129, 8192, 129),
    ] {
        for route in half_edge_routes() {
            cells.push(EdgeCell {
                cohort: EdgeCohort::DeepK,
                route,
                op: ResolvedGemmOp::Nn,
                shape: edge_shape(dims),
                layout: EdgeLayout::Contiguous,
            });
        }
    }

    let offsets = [
        ElementOffsets {
            output: 0,
            a: 1,
            b: 0,
        },
        ElementOffsets {
            output: 0,
            a: 0,
            b: 1,
        },
        ElementOffsets {
            output: 1,
            a: 0,
            b: 0,
        },
    ];
    for (op, dims) in [
        (ResolvedGemmOp::Nn, (129, 65, 129)),
        (ResolvedGemmOp::Tn, (65, 128, 128)),
        (ResolvedGemmOp::Nt, (129, 129, 72)),
    ] {
        for route in half_edge_routes() {
            for offsets in offsets {
                cells.push(EdgeCell {
                    cohort: EdgeCohort::Offset,
                    route,
                    op,
                    shape: edge_shape(dims),
                    layout: EdgeLayout::Offset(offsets),
                });
            }
        }
    }

    let padded_layouts = [
        PaddedNnLayout {
            label: "lda72_ldb136_ldc130",
            strides: (72, 136, 130),
            offsets: ElementOffsets::ALIGNED,
        },
        PaddedNnLayout {
            label: "lda72_ldb136_ldc131",
            strides: (72, 136, 131),
            offsets: ElementOffsets::ALIGNED,
        },
    ];
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        for tile in [TcTile::Tile64, TcTile::Tile128] {
            for layout in padded_layouts {
                cells.push(EdgeCell {
                    cohort: EdgeCohort::DirectPaddedNn,
                    route: Route::HalfForced { dtype, tile },
                    op: ResolvedGemmOp::Nn,
                    shape: edge_shape((129, 65, 129)),
                    layout: EdgeLayout::DirectPaddedNn(layout),
                });
            }
        }
    }
    cells
}

fn edge_shape_name(shape: Shape) -> String {
    format!("m{}_k{}_n{}", shape.dims.0, shape.dims.1, shape.dims.2)
}

fn edge_cell_id(cell: EdgeCell) -> String {
    format!(
        "edge/{}/{}/{}/{}/{}",
        cell.cohort.as_str(),
        logical_route_name(cell.route),
        op_name(cell.op),
        edge_shape_name(cell.shape),
        cell.layout.label(),
    )
}

fn edge_inventory_digest(cells: &[EdgeCell]) -> [u8; 32] {
    let mut digest = FramedSha256::new(b"gemm-bi-edge-performance-inventory.v1")
        .required(b"cell-count", &(cells.len() as u64).to_le_bytes());
    for (index, cell) in cells.iter().copied().enumerate() {
        let id = edge_cell_id(cell);
        digest = digest
            .required(b"cell-index", &(index as u64).to_le_bytes())
            .required(b"cell-id", id.as_bytes());
    }
    digest.finish()
}

fn validate_edge_inventory(cells: &[EdgeCell]) -> Result<(), String> {
    if cells.len() != EDGE_INVENTORY_COUNT {
        return Err(format!(
            "edge inventory count changed: expected {EDGE_INVENTORY_COUNT}, got {}",
            cells.len()
        ));
    }
    let unique = cells
        .iter()
        .copied()
        .map(edge_cell_id)
        .collect::<BTreeSet<_>>();
    if unique.len() != cells.len() {
        return Err("edge inventory contains duplicate cell identifiers".into());
    }
    let digest = hex_digest(edge_inventory_digest(cells));
    if digest != EDGE_INVENTORY_DIGEST {
        return Err(format!(
            "edge inventory digest changed: expected {EDGE_INVENTORY_DIGEST}, got {digest}"
        ));
    }
    Ok(())
}

fn unsupported_padded_layouts(op: ResolvedGemmOp) -> [PaddedNnLayout; 2] {
    match op {
        ResolvedGemmOp::Tn => [
            PaddedNnLayout {
                label: "lda136_ldb136_ldc129",
                strides: (136, 136, 129),
                offsets: ElementOffsets::ALIGNED,
            },
            PaddedNnLayout {
                label: "lda136_ldb136_ldc130",
                strides: (136, 136, 130),
                offsets: ElementOffsets::ALIGNED,
            },
        ],
        ResolvedGemmOp::Nt => [
            PaddedNnLayout {
                label: "lda80_ldb80_ldc130",
                strides: (80, 80, 130),
                offsets: ElementOffsets::ALIGNED,
            },
            PaddedNnLayout {
                label: "lda80_ldb80_ldc131",
                strides: (80, 80, 131),
                offsets: ElementOffsets::ALIGNED,
            },
        ],
        ResolvedGemmOp::Nn => unreachable!(),
    }
}

fn build_unsupported_padded_descriptors() -> Vec<UnsupportedPaddedDescriptor> {
    let mut descriptors = Vec::with_capacity(UNSUPPORTED_PADDED_INVENTORY_COUNT);
    for (op, dims) in [
        (ResolvedGemmOp::Tn, (65, 128, 128)),
        (ResolvedGemmOp::Nt, (129, 129, 72)),
    ] {
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for tile in [TcTile::Tile64, TcTile::Tile128] {
                for layout in unsupported_padded_layouts(op) {
                    descriptors.push(UnsupportedPaddedDescriptor {
                        contract: "unsupported_by_kernel_abi",
                        route: Route::HalfForced { dtype, tile },
                        op,
                        shape: edge_shape(dims),
                        layout,
                    });
                }
            }
        }
    }
    descriptors
}

fn unsupported_padded_id(descriptor: UnsupportedPaddedDescriptor) -> String {
    format!(
        "edge/unsupported_padded/{}/{}/{}/{}/{}",
        logical_route_name(descriptor.route),
        op_name(descriptor.op),
        edge_shape_name(descriptor.shape),
        descriptor.layout.label,
        descriptor.contract,
    )
}

fn unsupported_padded_inventory_digest(descriptors: &[UnsupportedPaddedDescriptor]) -> [u8; 32] {
    let mut digest = FramedSha256::new(b"gemm-bi-edge-unsupported-inventory.v1")
        .required(b"cell-count", &(descriptors.len() as u64).to_le_bytes());
    for (index, descriptor) in descriptors.iter().copied().enumerate() {
        let id = unsupported_padded_id(descriptor);
        digest = digest
            .required(b"cell-index", &(index as u64).to_le_bytes())
            .required(b"cell-id", id.as_bytes());
    }
    digest.finish()
}

fn validate_unsupported_padded_inventory(
    descriptors: &[UnsupportedPaddedDescriptor],
) -> Result<(), String> {
    if descriptors.len() != UNSUPPORTED_PADDED_INVENTORY_COUNT {
        return Err(format!(
            "unsupported padded inventory count changed: expected {UNSUPPORTED_PADDED_INVENTORY_COUNT}, got {}",
            descriptors.len()
        ));
    }
    let unique = descriptors
        .iter()
        .copied()
        .map(unsupported_padded_id)
        .collect::<BTreeSet<_>>();
    if unique.len() != descriptors.len() {
        return Err("unsupported padded inventory contains duplicate descriptors".into());
    }
    let digest = hex_digest(unsupported_padded_inventory_digest(descriptors));
    if digest != UNSUPPORTED_PADDED_INVENTORY_DIGEST {
        return Err(format!(
            "unsupported padded inventory digest changed: expected {UNSUPPORTED_PADDED_INVENTORY_DIGEST}, got {digest}"
        ));
    }
    Ok(())
}

fn unsupported_padded_descriptors_for_shard(
    shard: usize,
) -> Result<Vec<UnsupportedPaddedDescriptor>, String> {
    let descriptors = build_unsupported_padded_descriptors();
    validate_unsupported_padded_inventory(&descriptors)?;
    Ok(if shard == 0 { descriptors } else { Vec::new() })
}

fn render_unsupported_padded_descriptor(descriptor: UnsupportedPaddedDescriptor) -> String {
    format!(
        concat!(
            "{{\"schema\":\"MambaBiUnsupportedPerformanceCellV1\",",
            "\"descriptor_id\":\"{}\",\"contract\":\"{}\",",
            "\"route\":\"{}\",\"op\":\"{}\",\"shape\":\"{}\",",
            "\"stride_class\":\"{}\",\"strides\":[{},{},{}],",
            "\"offsets\":{{\"output\":{},\"a\":{},\"b\":{}}}}}"
        ),
        unsupported_padded_id(descriptor),
        descriptor.contract,
        logical_route_name(descriptor.route),
        op_name(descriptor.op),
        edge_shape_name(descriptor.shape),
        descriptor.layout.label,
        descriptor.layout.strides.0,
        descriptor.layout.strides.1,
        descriptor.layout.strides.2,
        descriptor.layout.offsets.output,
        descriptor.layout.offsets.a,
        descriptor.layout.offsets.b,
    )
}

fn tf32_shape_names(spec: &Tf32KernelSpec) -> impl Iterator<Item = &'static str> {
    let primary = match spec.tile {
        (128, 64) => [
            "large",
            "large_deep",
            "rect_tall",
            "all_tail",
            "split_candidate",
        ],
        (64, 64) => [
            "sq64",
            "underfill",
            "rect_wide",
            "all_tail",
            "split_candidate",
        ],
        (80, 32) => [
            "rect_wide",
            "large_deep",
            "thin_cols",
            "all_tail",
            "split_candidate",
        ],
        (16, 32) | (16, 16) => [
            "thin_rows",
            "thin_cols",
            "thin_rows_tail",
            "thin_cols_tail",
            "all_tail",
        ],
        tile => panic!("unexpected portable TF32 tile {tile:?}"),
    };
    primary.into_iter().chain(TF32_PROJECTION_SHAPES)
}

fn physical_qualification_route(route: Route) -> PhysicalQualificationRoute {
    match route {
        Route::F32Policy { policy } => PhysicalQualificationRoute::F32Policy(policy),
        Route::HalfPolicy {
            dtype,
            tensor_cores,
        } => PhysicalQualificationRoute::HalfPolicy {
            dtype,
            tensor_cores,
        },
        Route::HalfForced { dtype, tile } => PhysicalQualificationRoute::HalfForced { dtype, tile },
        Route::Tf32Forced(spec) => PhysicalQualificationRoute::Tf32Forced(spec.route),
    }
}

fn qualification_request(cell: Cell) -> PhysicalQualificationRequest {
    PhysicalQualificationRequest::contiguous(
        cell.op,
        cell.shape.dims,
        physical_qualification_route(cell.route),
    )
}

fn edge_qualification_request(cell: EdgeCell) -> Result<PhysicalQualificationRequest, String> {
    let route = physical_qualification_route(cell.route);
    match cell.layout {
        EdgeLayout::Contiguous => Ok(PhysicalQualificationRequest::contiguous(
            cell.op,
            cell.shape.dims,
            route,
        )),
        EdgeLayout::Offset(offsets) => {
            let offset = match (offsets.output, offsets.a, offsets.b) {
                (0, 1, 0) => PhysicalQualificationOffset::A,
                (0, 0, 1) => PhysicalQualificationOffset::B,
                (1, 0, 0) => PhysicalQualificationOffset::Output,
                _ => return Err("edge offset cell has a non-frozen offset layout".into()),
            };
            Ok(PhysicalQualificationRequest::one_element_offset(
                cell.op,
                cell.shape.dims,
                route,
                offset,
            ))
        }
        EdgeLayout::DirectPaddedNn(layout) => {
            let (dtype, tile) = match cell.route {
                Route::HalfForced { dtype, tile } => (dtype, tile),
                _ => return Err("direct padded edge cell requires a forced half route".into()),
            };
            let layout = match (layout.label, layout.strides) {
                ("lda72_ldb136_ldc130", (72, 136, 130)) => {
                    PhysicalPaddedNnLayout::Lda72Ldb136Ldc130
                }
                ("lda72_ldb136_ldc131", (72, 136, 131)) => {
                    PhysicalPaddedNnLayout::Lda72Ldb136Ldc131
                }
                _ => return Err("direct padded edge cell has a non-frozen stride layout".into()),
            };
            Ok(PhysicalQualificationRequest::direct_padded_nn(
                cell.shape.dims,
                dtype,
                tile,
                layout,
            ))
        }
    }
}

fn edge_call_scope(cell: EdgeCell) -> &'static str {
    if cell.cohort == EdgeCohort::DirectPaddedNn {
        "direct_kernel_forced"
    } else {
        call_scope(cell.route)
    }
}

fn prepare_cell(ctx: &GpuCtx, cell: Cell) -> Result<QualifiedPhysicalLaunch<'_>, String> {
    qualify_physical_launch(ctx, qualification_request(cell))
}

fn calibrate_measurement(
    mut measure: impl FnMut(usize) -> Result<f64, String>,
) -> Result<usize, String> {
    let probe_iterations = 16;
    let elapsed = measure(probe_iterations)?;
    let per_launch = elapsed / probe_iterations as f64;
    if !per_launch.is_finite() || per_launch <= 0.0 {
        return Err(format!(
            "pilot timing was not positive and finite: {per_launch}"
        ));
    }
    Ok((TARGET_WINDOW_MS / per_launch).ceil().clamp(1.0, 4096.0) as usize)
}

fn collect_paired_samples(
    ctx: &GpuCtx,
    iterations: FrozenIterations,
    windows: usize,
    order: PathOrder,
    physical: &mut QualifiedPhysicalLaunch<'_>,
) -> Result<(Samples, Samples), String> {
    match order {
        PathOrder::Ab => {
            physical.measure_eager_window_ms(ctx, 128)?;
            physical.measure_graph_window_ms(ctx, 128)?;
        }
        PathOrder::Ba => {
            physical.measure_graph_window_ms(ctx, 128)?;
            physical.measure_eager_window_ms(ctx, 128)?;
        }
    }
    let mut eager_values_us = Vec::with_capacity(windows);
    let mut graph_values_us = Vec::with_capacity(windows);
    for index in 0..windows {
        let sample_eager = |physical: &mut QualifiedPhysicalLaunch<'_>,
                            values: &mut Vec<f64>|
         -> Result<(), String> {
            let value = physical.measure_eager_window_ms(ctx, iterations.eager)? * 1000.0
                / iterations.eager as f64;
            validate_sample_us(value, index)?;
            values.push(value);
            Ok(())
        };
        let sample_graph =
            |physical: &QualifiedPhysicalLaunch<'_>, values: &mut Vec<f64>| -> Result<(), String> {
                let value = physical.measure_graph_window_ms(ctx, iterations.graph)? * 1000.0
                    / iterations.graph as f64;
                validate_sample_us(value, index)?;
                values.push(value);
                Ok(())
            };
        match order {
            PathOrder::Ab => {
                sample_eager(physical, &mut eager_values_us)?;
                sample_graph(physical, &mut graph_values_us)?;
            }
            PathOrder::Ba => {
                sample_graph(physical, &mut graph_values_us)?;
                sample_eager(physical, &mut eager_values_us)?;
            }
        }
    }
    Ok((
        Samples {
            values_us: eager_values_us,
            iterations: iterations.eager,
        },
        Samples {
            values_us: graph_values_us,
            iterations: iterations.graph,
        },
    ))
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn emit_record(ctx: &GpuCtx, cell: Cell, path: &str, samples: &Samples, meta: &RecordMeta<'_>) {
    let mut sorted = samples.values_us.clone();
    sorted.sort_by(f64::total_cmp);
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);
    let mut deviations = sorted
        .iter()
        .map(|value| (value - p50).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    let mad = percentile(&deviations, 0.50);
    let (m, k, n) = cell.shape.dims;
    let tflops = 2.0 * m as f64 * k as f64 * n as f64 / (p50 * 1.0e6);
    let mut raw = String::new();
    for (index, value) in samples.values_us.iter().enumerate() {
        if index != 0 {
            raw.push(',');
        }
        write!(raw, "{value:.9}").expect("String writes cannot fail");
    }
    let cc = ctx.stream.context().compute_capability().expect("CUDA CC");
    let route_identity = format!(
        "{},{}",
        render_route_identity_fields(cell.route),
        render_physical_evidence_fields(meta.physical),
    );
    println!(
        concat!(
            "{{\"schema\":\"MambaBiPerformanceCellV1\",",
            "\"variant\":\"{}\",\"cell_id\":\"{}\",\"path\":\"{}\",",
            "\"path_order\":\"{}\",\"iteration_source\":\"{}\",",
            "\"rust_profile\":\"{}\",\"release_build\":{},",
            "\"cc\":\"{}.{}\",{},",
            "\"call_scope\":\"{}\",\"op\":\"{}\",",
            "\"shape\":\"{}\",\"stride_class\":\"{}\",",
            "\"m\":{},\"k\":{},\"n\":{},",
            "\"iterations\":{},\"windows\":{},\"p50_us\":{:.9},",
            "\"p95_us\":{:.9},\"min_us\":{:.9},\"max_us\":{:.9},",
            "\"mad_us\":{:.9},\"tflops\":{:.9},\"samples_us\":[{}]}}"
        ),
        meta.variant,
        cell_id(cell),
        path,
        meta.path_order.as_str(),
        meta.iteration_source.as_str(),
        rust_profile(),
        !cfg!(debug_assertions),
        cc.0,
        cc.1,
        route_identity,
        call_scope(cell.route),
        op_name(cell.op),
        cell.shape.name,
        cell.stride_class.as_str(),
        m,
        k,
        n,
        samples.iterations,
        samples.values_us.len(),
        p50,
        p95,
        sorted[0],
        sorted[sorted.len() - 1],
        mad,
        tflops,
        raw,
    );
}

struct RunnerConfig<'a> {
    windows: usize,
    variant: &'a str,
    path_order: PathOrder,
    frozen: Option<FrozenIterations>,
}

fn performance_suite_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn run_cell(
    ctx: &GpuCtx,
    cell: Cell,
    config: RunnerConfig<'_>,
) -> Result<FrozenIterations, String> {
    run_cell_with_validation(ctx, cell, config, true)
}

fn run_prequalified_cell(
    ctx: &GpuCtx,
    cell: Cell,
    config: RunnerConfig<'_>,
) -> Result<FrozenIterations, String> {
    run_cell_with_validation(ctx, cell, config, false)
}

fn run_cell_with_validation(
    ctx: &GpuCtx,
    cell: Cell,
    config: RunnerConfig<'_>,
    validate_wrapper: bool,
) -> Result<FrozenIterations, String> {
    let mut physical = prepare_cell(ctx, cell)?;
    if validate_wrapper {
        physical.validate_timed_request(ctx, qualification_request(cell))?;
    }
    let (iterations, iteration_source) = match config.frozen {
        Some(iterations) => (iterations, IterationSource::Frozen),
        None => {
            let iterations = match config.path_order {
                PathOrder::Ab => FrozenIterations {
                    eager: calibrate_measurement(|iterations| {
                        physical.measure_eager_window_ms(ctx, iterations)
                    })?,
                    graph: calibrate_measurement(|iterations| {
                        physical.measure_graph_window_ms(ctx, iterations)
                    })?,
                },
                PathOrder::Ba => {
                    let graph = calibrate_measurement(|iterations| {
                        physical.measure_graph_window_ms(ctx, iterations)
                    })?;
                    let eager = calibrate_measurement(|iterations| {
                        physical.measure_eager_window_ms(ctx, iterations)
                    })?;
                    FrozenIterations { eager, graph }
                }
            };
            (iterations, IterationSource::Calibrated)
        }
    };
    let (eager, graph_samples) = collect_paired_samples(
        ctx,
        iterations,
        config.windows,
        config.path_order,
        &mut physical,
    )?;
    let meta = RecordMeta {
        variant: config.variant,
        path_order: config.path_order,
        iteration_source,
        physical: physical.evidence(),
    };
    match config.path_order {
        PathOrder::Ab => {
            emit_record(ctx, cell, "eager", &eager, &meta);
            emit_record(ctx, cell, "graph", &graph_samples, &meta);
        }
        PathOrder::Ba => {
            emit_record(ctx, cell, "graph", &graph_samples, &meta);
            emit_record(ctx, cell, "eager", &eager, &meta);
        }
    }
    Ok(iterations)
}

struct EdgeRecordMeta<'a> {
    variant: &'a str,
    path_order: PathOrder,
    iteration_source: IterationSource,
    physical: &'a QualifiedPhysicalLaunchEvidence,
    guards: QualifiedGuardValidation,
}

fn edge_layout_identity(cell: EdgeCell) -> ((usize, usize, usize), ElementOffsets) {
    match cell.layout {
        EdgeLayout::Contiguous => {
            let shape = F32TriadShape::contiguous(cell.op, cell.shape.dims);
            ((shape.lda, shape.ldb, shape.ldc), ElementOffsets::ALIGNED)
        }
        EdgeLayout::Offset(offsets) => {
            let shape = F32TriadShape::contiguous(cell.op, cell.shape.dims);
            ((shape.lda, shape.ldb, shape.ldc), offsets)
        }
        EdgeLayout::DirectPaddedNn(layout) => (layout.strides, layout.offsets),
    }
}

fn emit_edge_record(
    ctx: &GpuCtx,
    cell: EdgeCell,
    path: &str,
    samples: &Samples,
    meta: &EdgeRecordMeta<'_>,
) {
    let mut sorted = samples.values_us.clone();
    sorted.sort_by(f64::total_cmp);
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);
    let mut deviations = sorted
        .iter()
        .map(|value| (value - p50).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    let mad = percentile(&deviations, 0.50);
    let (m, k, n) = cell.shape.dims;
    let tflops = 2.0 * m as f64 * k as f64 * n as f64 / (p50 * 1.0e6);
    let mut raw = String::new();
    for (index, value) in samples.values_us.iter().enumerate() {
        if index != 0 {
            raw.push(',');
        }
        write!(raw, "{value:.9}").expect("String writes cannot fail");
    }
    let cc = ctx.stream.context().compute_capability().expect("CUDA CC");
    let route_identity = format!(
        "{},{}",
        render_route_identity_fields(cell.route),
        render_physical_evidence_fields(meta.physical),
    );
    let (strides, offsets) = edge_layout_identity(cell);
    println!(
        concat!(
            "{{\"schema\":\"MambaBiEdgePerformanceCellV1\",",
            "\"suite\":\"{}\",\"variant\":\"{}\",",
            "\"cell_id\":\"{}\",\"cohort\":\"{}\",\"path\":\"{}\",",
            "\"path_order\":\"{}\",\"iteration_source\":\"{}\",",
            "\"rust_profile\":\"{}\",\"release_build\":{},",
            "\"cc\":\"{}.{}\",{},",
            "\"call_scope\":\"{}\",\"op\":\"{}\",",
            "\"shape\":\"{}\",\"stride_class\":\"{}\",",
            "\"strides\":[{},{},{}],",
            "\"offsets\":{{\"output\":{},\"a\":{},\"b\":{}}},",
            "\"red_zones_validated\":true,\"guard_allocation_count\":{},",
            "\"guard_element_count\":{},",
            "\"m\":{},\"k\":{},\"n\":{},",
            "\"iterations\":{},\"windows\":{},\"p50_us\":{:.9},",
            "\"p95_us\":{:.9},\"min_us\":{:.9},\"max_us\":{:.9},",
            "\"mad_us\":{:.9},\"tflops\":{:.9},\"samples_us\":[{}]}}"
        ),
        EDGE_FROZEN_ITERATION_SUITE,
        meta.variant,
        edge_cell_id(cell),
        cell.cohort.as_str(),
        path,
        meta.path_order.as_str(),
        meta.iteration_source.as_str(),
        rust_profile(),
        !cfg!(debug_assertions),
        cc.0,
        cc.1,
        route_identity,
        edge_call_scope(cell),
        op_name(cell.op),
        edge_shape_name(cell.shape),
        cell.layout.label(),
        strides.0,
        strides.1,
        strides.2,
        offsets.output,
        offsets.a,
        offsets.b,
        meta.guards.allocation_count(),
        meta.guards.element_count(),
        m,
        k,
        n,
        samples.iterations,
        samples.values_us.len(),
        p50,
        p95,
        sorted[0],
        sorted[sorted.len() - 1],
        mad,
        tflops,
        raw,
    );
}

fn run_edge_cell(
    ctx: &GpuCtx,
    cell: EdgeCell,
    config: RunnerConfig<'_>,
) -> Result<FrozenIterations, String> {
    let request = edge_qualification_request(cell)?;
    let mut physical = qualify_physical_launch(ctx, request)?;
    physical.validate_timed_request(ctx, request)?;
    let (iterations, iteration_source) = match config.frozen {
        Some(iterations) => (iterations, IterationSource::Frozen),
        None => {
            let iterations = match config.path_order {
                PathOrder::Ab => FrozenIterations {
                    eager: calibrate_measurement(|iterations| {
                        physical.measure_eager_window_ms(ctx, iterations)
                    })?,
                    graph: calibrate_measurement(|iterations| {
                        physical.measure_graph_window_ms(ctx, iterations)
                    })?,
                },
                PathOrder::Ba => {
                    let graph = calibrate_measurement(|iterations| {
                        physical.measure_graph_window_ms(ctx, iterations)
                    })?;
                    let eager = calibrate_measurement(|iterations| {
                        physical.measure_eager_window_ms(ctx, iterations)
                    })?;
                    FrozenIterations { eager, graph }
                }
            };
            (iterations, IterationSource::Calibrated)
        }
    };
    let (eager, graph_samples) = collect_paired_samples(
        ctx,
        iterations,
        config.windows,
        config.path_order,
        &mut physical,
    )?;
    let guards = physical.validate_red_zones(ctx)?;
    let meta = EdgeRecordMeta {
        variant: config.variant,
        path_order: config.path_order,
        iteration_source,
        physical: physical.evidence(),
        guards,
    };
    match config.path_order {
        PathOrder::Ab => {
            emit_edge_record(ctx, cell, "eager", &eager, &meta);
            emit_edge_record(ctx, cell, "graph", &graph_samples, &meta);
        }
        PathOrder::Ba => {
            emit_edge_record(ctx, cell, "graph", &graph_samples, &meta);
            emit_edge_record(ctx, cell, "eager", &eager, &meta);
        }
    }
    Ok(iterations)
}

fn allocate_cublas_denominator_buffers(
    ctx: &GpuCtx,
    cell: CublasDenominatorCell,
) -> Result<CublasDenominatorBuffers, String> {
    let (m, k, n) = cell.shape.dims;
    let extent = |left: usize, right: usize, label: &str| {
        left.checked_mul(right)
            .ok_or_else(|| format!("cuBLAS denominator {label} extent overflows usize"))
    };
    let (a_elements, b_elements, output_elements) = match cell.op {
        ResolvedGemmOp::Nn => (
            extent(m, k, "M*K")?,
            extent(k, n, "K*N")?,
            extent(m, n, "M*N")?,
        ),
        ResolvedGemmOp::Tn => (
            extent(m, k, "M*K")?,
            extent(m, n, "M*N")?,
            extent(k, n, "K*N")?,
        ),
        ResolvedGemmOp::Nt => (
            extent(m, n, "M*N")?,
            extent(k, n, "K*N")?,
            extent(m, k, "M*K")?,
        ),
    };
    Ok(CublasDenominatorBuffers {
        output: DtypedBuf::zeros(
            &ctx.stream,
            output_elements,
            cublas_denominator_output_dtype(cell.dtype, cell.op),
        )?,
        a: DtypedBuf::zeros(&ctx.stream, a_elements, cell.dtype)?,
        b: DtypedBuf::zeros(&ctx.stream, b_elements, cell.dtype)?,
    })
}

fn launch_cublas_denominator(
    ctx: &GpuCtx,
    cell: CublasDenominatorCell,
    mode: CublasDenominatorMode,
    buffers: &CublasDenominatorBuffers,
) -> Result<(), String> {
    let geometry = cublas_denominator_geometry(cell.op, cell.shape.dims);
    let output_dtype = cublas_denominator_output_dtype(cell.dtype, cell.op);
    let alpha = 1.0f32;
    let beta = if cell.op == ResolvedGemmOp::Tn {
        1.0f32
    } else {
        0.0f32
    };
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *ctx.blas.handle(),
            geometry.trans_a,
            geometry.trans_b,
            geometry.m,
            geometry.n,
            geometry.k,
            &alpha as *const f32 as *const c_void,
            buffers.b.cached_ptr() as *const c_void,
            cell.dtype.cuda_data_type(),
            geometry.lda,
            buffers.a.cached_ptr() as *const c_void,
            cell.dtype.cuda_data_type(),
            geometry.ldb,
            &beta as *const f32 as *const c_void,
            buffers.output.cached_ptr() as *mut c_void,
            output_dtype.cuda_data_type(),
            geometry.ldc,
            cublas_denominator_compute(cell.dtype, mode),
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|error| {
            format!(
                "cuBLAS {} {} denominator launch failed: {error:?}",
                mode.as_str(),
                cublas_denominator_cell_id(cell)
            )
        })?;
    }
    Ok(())
}

fn measure_cublas_denominator_window_ms(
    ctx: &GpuCtx,
    cell: CublasDenominatorCell,
    mode: CublasDenominatorMode,
    buffers: &CublasDenominatorBuffers,
    iterations: usize,
) -> Result<f64, String> {
    if iterations == 0 {
        return Err("cuBLAS denominator iteration count must be positive".into());
    }
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("record cuBLAS denominator start event: {error:?}"))?;
    for _ in 0..iterations {
        launch_cublas_denominator(ctx, cell, mode, buffers)?;
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("record cuBLAS denominator end event: {error:?}"))?;
    start
        .elapsed_ms(&end)
        .map(f64::from)
        .map_err(|error| format!("measure cuBLAS denominator events: {error:?}"))
}

fn calibrate_cublas_denominator_iterations(
    ctx: &GpuCtx,
    cell: CublasDenominatorCell,
    buffers: &CublasDenominatorBuffers,
    order: PathOrder,
) -> Result<CublasDenominatorIterations, String> {
    let calibrate = |mode| {
        calibrate_measurement(|iterations| {
            measure_cublas_denominator_window_ms(ctx, cell, mode, buffers, iterations)
        })
    };
    match order {
        PathOrder::Ab => Ok(CublasDenominatorIterations {
            fast: calibrate(CublasDenominatorMode::Fast)?,
            pedantic: calibrate(CublasDenominatorMode::Pedantic)?,
        }),
        PathOrder::Ba => {
            let pedantic = calibrate(CublasDenominatorMode::Pedantic)?;
            let fast = calibrate(CublasDenominatorMode::Fast)?;
            Ok(CublasDenominatorIterations { fast, pedantic })
        }
    }
}

fn collect_cublas_denominator_samples(
    ctx: &GpuCtx,
    cell: CublasDenominatorCell,
    buffers: &CublasDenominatorBuffers,
    iterations: CublasDenominatorIterations,
    windows: usize,
    order: PathOrder,
) -> Result<(Samples, Samples), String> {
    let mut fast = Vec::with_capacity(windows);
    let mut pedantic = Vec::with_capacity(windows);
    let sample = |mode, count: usize, values: &mut Vec<f64>, index| {
        let value = measure_cublas_denominator_window_ms(ctx, cell, mode, buffers, count)? * 1000.0
            / count as f64;
        validate_sample_us(value, index)?;
        values.push(value);
        Ok::<(), String>(())
    };
    for index in 0..windows {
        match order {
            PathOrder::Ab => {
                sample(
                    CublasDenominatorMode::Fast,
                    iterations.fast,
                    &mut fast,
                    index,
                )?;
                sample(
                    CublasDenominatorMode::Pedantic,
                    iterations.pedantic,
                    &mut pedantic,
                    index,
                )?;
            }
            PathOrder::Ba => {
                sample(
                    CublasDenominatorMode::Pedantic,
                    iterations.pedantic,
                    &mut pedantic,
                    index,
                )?;
                sample(
                    CublasDenominatorMode::Fast,
                    iterations.fast,
                    &mut fast,
                    index,
                )?;
            }
        }
    }
    Ok((
        Samples {
            values_us: fast,
            iterations: iterations.fast,
        },
        Samples {
            values_us: pedantic,
            iterations: iterations.pedantic,
        },
    ))
}

fn emit_cublas_denominator_record(
    ctx: &GpuCtx,
    cell: CublasDenominatorCell,
    mode: CublasDenominatorMode,
    samples: &Samples,
    variant: &str,
    order: PathOrder,
) {
    let mut sorted = samples.values_us.clone();
    sorted.sort_by(f64::total_cmp);
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);
    let mut deviations = sorted
        .iter()
        .map(|value| (value - p50).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    let mad = percentile(&deviations, 0.50);
    let (m, k, n) = cell.shape.dims;
    let tflops = 2.0 * m as f64 * k as f64 * n as f64 / (p50 * 1.0e6);
    let mut raw = String::new();
    for (index, value) in samples.values_us.iter().enumerate() {
        if index != 0 {
            raw.push(',');
        }
        write!(raw, "{value:.9}").expect("String writes cannot fail");
    }
    let cc = ctx.stream.context().compute_capability().expect("CUDA CC");
    println!(
        concat!(
            "{{\"schema\":\"MambaBiCublasPerformanceCellV1\",",
            "\"suite\":\"gemm_bi_cublas_performance_denominators\",",
            "\"scope\":\"performance_only\",\"variant\":\"{}\",",
            "\"cell_id\":\"{}\",\"denominator\":\"cublas_{}\",",
            "\"arm_order\":\"{}\",\"iteration_source\":\"calibrated\",",
            "\"rust_profile\":\"{}\",\"release_build\":{},",
            "\"cc\":\"{}.{}\",\"op\":\"{}\",\"dtype\":\"{}\",",
            "\"shape\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
            "\"compute\":\"{}\",\"algorithm\":\"default\",",
            "\"iterations\":{},\"windows\":{},\"p50_us\":{:.9},",
            "\"p95_us\":{:.9},\"min_us\":{:.9},\"max_us\":{:.9},",
            "\"mad_us\":{:.9},\"tflops\":{:.9},\"samples_us\":[{}]}}"
        ),
        variant,
        cublas_denominator_cell_id(cell),
        mode.as_str(),
        order.as_str(),
        rust_profile(),
        !cfg!(debug_assertions),
        cc.0,
        cc.1,
        op_name(cell.op),
        dtype_name(cell.dtype),
        cell.shape.name,
        m,
        k,
        n,
        cublas_denominator_compute_name(cell.dtype, mode),
        samples.iterations,
        samples.values_us.len(),
        p50,
        p95,
        sorted[0],
        sorted[sorted.len() - 1],
        mad,
        tflops,
        raw,
    );
}

fn run_cublas_denominator_cell(
    ctx: &GpuCtx,
    cell: CublasDenominatorCell,
    windows: usize,
    variant: &str,
    order: PathOrder,
) -> Result<(), String> {
    let buffers = allocate_cublas_denominator_buffers(ctx, cell)?;
    let iterations = calibrate_cublas_denominator_iterations(ctx, cell, &buffers, order)?;
    let (fast, pedantic) =
        collect_cublas_denominator_samples(ctx, cell, &buffers, iterations, windows, order)?;
    match order {
        PathOrder::Ab => {
            emit_cublas_denominator_record(
                ctx,
                cell,
                CublasDenominatorMode::Fast,
                &fast,
                variant,
                order,
            );
            emit_cublas_denominator_record(
                ctx,
                cell,
                CublasDenominatorMode::Pedantic,
                &pedantic,
                variant,
                order,
            );
        }
        PathOrder::Ba => {
            emit_cublas_denominator_record(
                ctx,
                cell,
                CublasDenominatorMode::Pedantic,
                &pedantic,
                variant,
                order,
            );
            emit_cublas_denominator_record(
                ctx,
                cell,
                CublasDenominatorMode::Fast,
                &fast,
                variant,
                order,
            );
        }
    }
    Ok(())
}

mod tn_narrow_cublas_pair {
    use super::*;

    const DISCOVERY_WINDOWS: usize = 21;
    const FINAL_WINDOWS: usize = 101;
    const WARMUP_LAUNCHES: usize = 128;
    const SEMANTIC_REPEATS: usize = 10;
    const GUARD_ELEMENTS: usize = 16;
    const GUARD_BITS: u32 = 0x4f12_3456;
    const OUTPUT_ENV: &str = "MAMBA_RS_TN_NARROW_CUBLAS_JSONL";
    const FINAL_ENV: &str = "MAMBA_RS_TN_NARROW_CUBLAS_FINAL";
    const RECORD_SCHEMA: &str = "MambaBiF32TnNarrowTwoLaunchPairedCublasV1";
    const COMPLETION_SCHEMA: &str = "MambaBiF32TnNarrowTwoLaunchPairedCublasCompletionV1";

    #[derive(Clone, Copy)]
    pub(super) struct PairCell {
        pub(super) name: &'static str,
        pub(super) dims: (usize, usize, usize),
        pub(super) a_offset: usize,
        pub(super) b_offset: usize,
        pub(super) output_offset: usize,
    }

    const CELLS: [PairCell; 7] = [
        PairCell {
            name: "m256_k32_n2",
            dims: (256, 32, 2),
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        PairCell {
            name: "m1024_k47_n17",
            dims: (1_024, 47, 17),
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        PairCell {
            name: "m1024_k128_n25",
            dims: (1_024, 128, 25),
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        PairCell {
            name: "m4096_k64_n64",
            dims: (4_096, 64, 64),
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        PairCell {
            name: "m4096_k128_n96",
            dims: (4_096, 128, 96),
            a_offset: 0,
            b_offset: 0,
            output_offset: 0,
        },
        PairCell {
            name: "m4096_k256_n101_offset1",
            dims: (4_096, 256, 101),
            a_offset: 1,
            b_offset: 0,
            output_offset: 1,
        },
        PairCell {
            name: "m4111_k257_n127_offset1",
            dims: (4_111, 257, 127),
            a_offset: 0,
            b_offset: 1,
            output_offset: 1,
        },
    ];

    pub(super) const fn candidate_partial_symbol(cell: PairCell) -> &'static str {
        if cell.a_offset == 0 && cell.b_offset == 0 {
            "gemm_bi_tn_narrow_splitm_partial_aligned"
        } else {
            "gemm_bi_tn_narrow_splitm_partial"
        }
    }

    #[derive(Clone, Copy)]
    pub(super) struct CandidatePlan {
        pub(super) m_chunk: usize,
        pub(super) chunks: usize,
        grid_m: u32,
        grid_n: u32,
    }

    pub(super) fn candidate_plan(
        cell: PairCell,
        multiprocessors: u32,
    ) -> Result<CandidatePlan, String> {
        const SCRATCH_CAP_F32: usize = 1 << 23;
        const CHUNK_ALIGNMENT: usize = 16;
        let (m, k, n) = cell.dims;
        if m < 256 || !(2..=127).contains(&n) || multiprocessors == 0 {
            return Err(format!("{} is outside the TN narrow envelope", cell.name));
        }
        let grid_m = u32::try_from(k)
            .map_err(|_| "paired TN K exceeds u32::MAX")?
            .div_ceil(64);
        let grid_n = u32::try_from(n)
            .map_err(|_| "paired TN N exceeds u32::MAX")?
            .div_ceil(32);
        let base_blocks = grid_m
            .checked_mul(grid_n)
            .ok_or_else(|| "paired TN base grid overflows u32".to_string())?;
        let target_blocks = multiprocessors
            .checked_mul(2)
            .ok_or_else(|| "paired TN two-wave target overflows u32".to_string())?;
        if base_blocks == 0 || base_blocks >= target_blocks {
            return Err(format!("{} does not underfill two SM waves", cell.name));
        }
        let output_elements = k
            .checked_mul(n)
            .ok_or_else(|| "paired TN output extent overflows usize".to_string())?;
        let scratch_chunks = SCRATCH_CAP_F32 / output_elements;
        let desired_chunks = (target_blocks as usize)
            .div_ceil(base_blocks as usize)
            .min(scratch_chunks)
            .max(1);
        let raw_chunk = m.div_ceil(desired_chunks);
        let m_chunk = raw_chunk
            .checked_add(CHUNK_ALIGNMENT - 1)
            .ok_or_else(|| "paired TN chunk alignment overflows usize".to_string())?
            & !(CHUNK_ALIGNMENT - 1);
        let chunks = m.div_ceil(m_chunk);
        if chunks < 2
            || chunks
                .checked_mul(output_elements)
                .is_none_or(|elements| elements > SCRATCH_CAP_F32)
        {
            return Err(format!("{} cannot form a bounded split plan", cell.name));
        }
        Ok(CandidatePlan {
            m_chunk,
            chunks,
            grid_m,
            grid_n,
        })
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum PairOrder {
        CandidateFirst,
        CublasFirst,
    }

    impl PairOrder {
        fn as_str(self) -> &'static str {
            match self {
                Self::CandidateFirst => "candidate_first",
                Self::CublasFirst => "cublas_first",
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PairPath {
        Eager,
        Graph,
    }

    impl PairPath {
        fn as_str(self) -> &'static str {
            match self {
                Self::Eager => "eager",
                Self::Graph => "graph",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum PairArm {
        Candidate,
        Cublas,
    }

    pub(super) struct SemanticProbe {
        pub(super) x: Vec<f32>,
        pub(super) dy: Vec<f32>,
        pub(super) seed: Vec<f32>,
        pub(super) expected: Vec<f32>,
    }

    pub(super) fn cells() -> &'static [PairCell] {
        &CELLS
    }

    pub(super) fn denominator_cell(cell: PairCell) -> CublasDenominatorCell {
        CublasDenominatorCell {
            dtype: WeightDtype::F32,
            op: ResolvedGemmOp::Tn,
            shape: Shape {
                name: cell.name,
                dims: cell.dims,
            },
        }
    }

    pub(super) fn window_count(value: Option<&str>) -> Result<usize, String> {
        match value.filter(|value| !value.is_empty()) {
            None | Some("discovery") | Some("21") => Ok(DISCOVERY_WINDOWS),
            Some("1") | Some("true") | Some("final") | Some("101") => Ok(FINAL_WINDOWS),
            Some(value) => Err(format!(
                "MAMBA_RS_TN_NARROW_CUBLAS_FINAL must be discovery/21 or final/101, received {value:?}"
            )),
        }
    }

    pub(super) fn paired_schedule(windows: usize) -> Result<Vec<PairOrder>, String> {
        if !matches!(windows, DISCOVERY_WINDOWS | FINAL_WINDOWS) {
            return Err("paired TN narrow window count must be exactly 21 or 101".into());
        }
        let mut schedule = Vec::with_capacity(windows * 2);
        for _ in 0..windows {
            schedule.push(PairOrder::CandidateFirst);
            schedule.push(PairOrder::CublasFirst);
        }
        Ok(schedule)
    }

    pub(super) fn tn_reference(
        dims: (usize, usize, usize),
        x: &[f32],
        dy: &[f32],
        seed: &[f32],
    ) -> Result<Vec<f64>, String> {
        let (m, k, n) = dims;
        let mk = m
            .checked_mul(k)
            .ok_or_else(|| "paired TN M*K extent overflows usize".to_string())?;
        let mn = m
            .checked_mul(n)
            .ok_or_else(|| "paired TN M*N extent overflows usize".to_string())?;
        let kn = k
            .checked_mul(n)
            .ok_or_else(|| "paired TN K*N extent overflows usize".to_string())?;
        if x.len() != mk || dy.len() != mn || seed.len() != kn {
            return Err("paired TN semantic operand extent changed".into());
        }
        let mut expected = seed.iter().copied().map(f64::from).collect::<Vec<_>>();
        for row in 0..k {
            for column in 0..n {
                let mut sum = 0.0f64;
                for reduction in 0..m {
                    sum +=
                        f64::from(x[reduction * k + row]) * f64::from(dy[reduction * n + column]);
                }
                expected[row * n + column] += sum;
            }
        }
        Ok(expected)
    }

    pub(super) fn one_term_probe(dims: (usize, usize, usize)) -> Result<SemanticProbe, String> {
        let (m, k, n) = dims;
        if m == 0 || k == 0 || n == 0 {
            return Err("paired TN one-term probe requires positive dimensions".into());
        }
        let mut x = vec![0.0; m.checked_mul(k).ok_or("paired TN probe X overflow")?];
        let mut dy = vec![0.0; m.checked_mul(n).ok_or("paired TN probe dY overflow")?];
        let active = m / 2;
        let x_values = [0.25, -0.5, 0.75, -1.0];
        let dy_values = [0.5, -0.25, 0.75, -0.5];
        for row in 0..k {
            x[active * k + row] = x_values[row % x_values.len()];
        }
        for column in 0..n {
            dy[active * n + column] = dy_values[column % dy_values.len()];
        }
        let seed_values = [0.03125, -0.0625, 0.09375, -0.125];
        let seed = (0..k.checked_mul(n).ok_or("paired TN probe seed overflow")?)
            .map(|index| seed_values[index % seed_values.len()])
            .collect::<Vec<_>>();
        let expected = tn_reference(dims, &x, &dy, &seed)?
            .into_iter()
            .map(|value| value as f32)
            .collect();
        Ok(SemanticProbe {
            x,
            dy,
            seed,
            expected,
        })
    }

    pub(super) fn paired_ratios(
        candidate: &[f64],
        cublas: &[f64],
        expected: usize,
    ) -> Result<Vec<f64>, String> {
        if candidate.len() != expected || cublas.len() != expected {
            return Err(format!(
                "paired TN ratio requires {expected} samples per arm"
            ));
        }
        candidate
            .iter()
            .zip(cublas)
            .enumerate()
            .map(|(index, (&candidate, &cublas))| {
                let ratio = candidate / cublas;
                if !candidate.is_finite()
                    || candidate <= 0.0
                    || !cublas.is_finite()
                    || cublas <= 0.0
                    || !ratio.is_finite()
                    || ratio <= 0.0
                {
                    return Err(format!("invalid paired TN sample at window {index}"));
                }
                Ok(ratio)
            })
            .collect()
    }

    struct PairBuffers {
        candidate_output: GpuBuffer,
        candidate_seed: GpuBuffer,
        candidate_x: GpuBuffer,
        candidate_dy: GpuBuffer,
        candidate_scratch: GpuBuffer,
        cublas: CublasDenominatorBuffers,
        cublas_seed: GpuBuffer,
        cell: PairCell,
        candidate_plan: CandidatePlan,
        output_elements: usize,
        x_elements: usize,
        dy_elements: usize,
        scratch_elements: usize,
    }

    impl PairBuffers {
        fn new(ctx: &GpuCtx, cell: PairCell, multiprocessors: u32) -> Result<Self, String> {
            let (m, k, n) = cell.dims;
            let guarded = |offset: usize, elements: usize, label: &str| {
                elements
                    .checked_add(offset)
                    .ok_or_else(|| format!("paired TN {label} offset extent overflows usize"))?
                    .checked_add(GUARD_ELEMENTS)
                    .ok_or_else(|| format!("paired TN {label} guarded extent overflows usize"))
            };
            let x_elements = m
                .checked_mul(k)
                .ok_or_else(|| "paired TN X extent overflows usize".to_string())?;
            let dy_elements = m
                .checked_mul(n)
                .ok_or_else(|| "paired TN dY extent overflows usize".to_string())?;
            let output_elements = k
                .checked_mul(n)
                .ok_or_else(|| "paired TN dW extent overflows usize".to_string())?;
            let candidate_plan = candidate_plan(cell, multiprocessors)?;
            let scratch_elements = candidate_plan
                .chunks
                .checked_mul(output_elements)
                .ok_or_else(|| "paired TN scratch extent overflows usize".to_string())?;
            let guarded_x =
                vec![f32::from_bits(GUARD_BITS); guarded(cell.a_offset, x_elements, "X")?];
            let guarded_dy =
                vec![f32::from_bits(GUARD_BITS); guarded(cell.b_offset, dy_elements, "dY")?];
            let guarded_output = vec![
                f32::from_bits(GUARD_BITS);
                guarded(cell.output_offset, output_elements, "dW")?
            ];
            let guarded_scratch =
                vec![f32::from_bits(GUARD_BITS); guarded(0, scratch_elements, "scratch")?];
            let candidate_x = GpuBuffer::from_cpu(&ctx.stream, &guarded_x)?;
            let candidate_dy = GpuBuffer::from_cpu(&ctx.stream, &guarded_dy)?;
            let candidate_seed = GpuBuffer::from_cpu(&ctx.stream, &guarded_output)?;
            let candidate_output = GpuBuffer::from_cpu(&ctx.stream, &guarded_output)?;
            let candidate_scratch = GpuBuffer::from_cpu(&ctx.stream, &guarded_scratch)?;
            let cublas_seed = GpuBuffer::from_cpu(&ctx.stream, &guarded_output)?;
            let cublas = CublasDenominatorBuffers {
                output: DtypedBuf::zeros(&ctx.stream, guarded_output.len(), WeightDtype::F32)?,
                a: DtypedBuf::zeros(&ctx.stream, guarded_x.len(), WeightDtype::F32)?,
                b: DtypedBuf::zeros(&ctx.stream, guarded_dy.len(), WeightDtype::F32)?,
            };
            Ok(Self {
                candidate_output,
                candidate_seed,
                candidate_x,
                candidate_dy,
                candidate_scratch,
                cublas,
                cublas_seed,
                cell,
                candidate_plan,
                output_elements,
                x_elements,
                dy_elements,
                scratch_elements,
            })
        }

        fn guarded(values: &[f32], offset: usize) -> Vec<f32> {
            let mut guarded = Vec::with_capacity(offset + values.len() + GUARD_ELEMENTS);
            guarded.resize(offset, f32::from_bits(GUARD_BITS));
            guarded.extend_from_slice(values);
            guarded.resize(
                offset + values.len() + GUARD_ELEMENTS,
                f32::from_bits(GUARD_BITS),
            );
            guarded
        }

        fn upload_probe(&mut self, ctx: &GpuCtx, probe: &SemanticProbe) -> Result<(), String> {
            if probe.x.len() != self.x_elements
                || probe.dy.len() != self.dy_elements
                || probe.seed.len() != self.output_elements
            {
                return Err("paired TN probe extent does not match device buffers".into());
            }
            let x = Self::guarded(&probe.x, self.cell.a_offset);
            let dy = Self::guarded(&probe.dy, self.cell.b_offset);
            let seed = Self::guarded(&probe.seed, self.cell.output_offset);
            self.candidate_x.upload(&ctx.stream, &x)?;
            self.candidate_dy.upload(&ctx.stream, &dy)?;
            self.candidate_seed.upload(&ctx.stream, &seed)?;
            self.candidate_output.upload(&ctx.stream, &seed)?;
            self.cublas.a.upload_f32(&ctx.stream, &x)?;
            self.cublas.b.upload_f32(&ctx.stream, &dy)?;
            self.cublas.output.upload_f32(&ctx.stream, &seed)?;
            self.cublas_seed.upload(&ctx.stream, &seed)
        }

        fn reset(&mut self, ctx: &GpuCtx, arm: PairArm) -> Result<(), String> {
            match arm {
                PairArm::Candidate => self
                    .candidate_output
                    .copy_from_raw(&self.candidate_seed, &ctx.stream),
                PairArm::Cublas => copy_raw_f32(
                    &ctx.stream,
                    self.cublas.output.cached_ptr(),
                    self.cublas_seed.cached_ptr(),
                    self.cublas.output.len_elems(),
                ),
            }
        }

        fn output(&self, ctx: &GpuCtx, arm: PairArm) -> Result<Vec<f32>, String> {
            let values = match arm {
                PairArm::Candidate => self.candidate_output.to_cpu(&ctx.stream)?,
                PairArm::Cublas => {
                    let mut values = vec![0.0; self.cublas.output.len_elems()];
                    self.cublas.output.download_f32(&ctx.stream, &mut values)?;
                    values
                }
            };
            validate_guard(&values, self.cell.output_offset, self.output_elements, "dW")?;
            let start = self.cell.output_offset;
            Ok(values[start..start + self.output_elements].to_vec())
        }

        fn validate_guards(&self, ctx: &GpuCtx) -> Result<(), String> {
            let candidate_x = self.candidate_x.to_cpu(&ctx.stream)?;
            let candidate_dy = self.candidate_dy.to_cpu(&ctx.stream)?;
            let mut cublas_x = vec![0.0; self.cublas.a.len_elems()];
            let mut cublas_dy = vec![0.0; self.cublas.b.len_elems()];
            self.cublas.a.download_f32(&ctx.stream, &mut cublas_x)?;
            self.cublas.b.download_f32(&ctx.stream, &mut cublas_dy)?;
            validate_guard(
                &candidate_x,
                self.cell.a_offset,
                self.x_elements,
                "candidate X",
            )?;
            validate_guard(
                &candidate_dy,
                self.cell.b_offset,
                self.dy_elements,
                "candidate dY",
            )?;
            validate_guard(&cublas_x, self.cell.a_offset, self.x_elements, "cuBLAS X")?;
            validate_guard(
                &cublas_dy,
                self.cell.b_offset,
                self.dy_elements,
                "cuBLAS dY",
            )?;
            let candidate_seed = self.candidate_seed.to_cpu(&ctx.stream)?;
            let cublas_seed = self.cublas_seed.to_cpu(&ctx.stream)?;
            validate_guard(
                &candidate_seed,
                self.cell.output_offset,
                self.output_elements,
                "candidate seed",
            )?;
            validate_guard(
                &cublas_seed,
                self.cell.output_offset,
                self.output_elements,
                "cuBLAS seed",
            )?;
            let scratch = self.candidate_scratch.to_cpu(&ctx.stream)?;
            validate_guard(&scratch, 0, self.scratch_elements, "candidate scratch")?;
            self.output(ctx, PairArm::Candidate)?;
            self.output(ctx, PairArm::Cublas)?;
            Ok(())
        }
    }

    fn copy_raw_f32(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        destination: u64,
        source: u64,
        elements: usize,
    ) -> Result<(), String> {
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "paired TN D2D byte count overflows usize".to_string())?;
        let status = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                destination,
                source,
                bytes,
                stream.cu_stream(),
            )
        };
        if status == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("paired TN D2D reset failed: {status:?}"))
        }
    }

    fn validate_guard(
        values: &[f32],
        offset: usize,
        active: usize,
        label: &str,
    ) -> Result<(), String> {
        if values.len() != offset + active + GUARD_ELEMENTS {
            return Err(format!("paired TN {label} guard extent changed"));
        }
        if let Some((index, value)) = values[..offset]
            .iter()
            .chain(values[offset + active..].iter())
            .enumerate()
            .find(|(_, value)| value.to_bits() != GUARD_BITS)
        {
            return Err(format!(
                "paired TN {label} guard changed at red-zone element {index}: 0x{:08x}",
                value.to_bits()
            ));
        }
        Ok(())
    }

    fn launch_candidate(ctx: &GpuCtx, buffers: &PairBuffers) -> Result<(), String> {
        let cell = buffers.cell;
        let plan = buffers.candidate_plan;
        let (m, k, n) = cell.dims;
        let m_i = i32::try_from(m).map_err(|_| "paired TN M exceeds i32::MAX")?;
        let k_i = i32::try_from(k).map_err(|_| "paired TN K exceeds i32::MAX")?;
        let n_i = i32::try_from(n).map_err(|_| "paired TN N exceeds i32::MAX")?;
        let m_chunk_i =
            i32::try_from(plan.m_chunk).map_err(|_| "paired TN M chunk exceeds i32::MAX")?;
        let chunks_i =
            i32::try_from(plan.chunks).map_err(|_| "paired TN chunks exceed i32::MAX")?;
        let chunks_u32 =
            u32::try_from(plan.chunks).map_err(|_| "paired TN chunks exceed u32::MAX")?;
        let output = buffers
            .candidate_output
            .raw_ptr_at(&ctx.stream, cell.output_offset);
        let a = buffers.candidate_x.raw_ptr_at(&ctx.stream, cell.a_offset);
        let b = buffers.candidate_dy.raw_ptr_at(&ctx.stream, cell.b_offset);
        let scratch = buffers.candidate_scratch.raw_ptr(&ctx.stream);
        let alpha = 1.0f32;
        let partial_config = cudarc::driver::LaunchConfig {
            grid_dim: (plan.grid_m, plan.grid_n, chunks_u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let partial = match candidate_partial_symbol(cell) {
            "gemm_bi_tn_narrow_splitm_partial_aligned" => {
                &ctx.kernels.gemm_bi_tn_narrow_splitm_partial_aligned
            }
            "gemm_bi_tn_narrow_splitm_partial" => &ctx.kernels.gemm_bi_tn_narrow_splitm_partial,
            symbol => return Err(format!("unknown paired TN partial symbol {symbol}")),
        };
        let mut partial_builder = ctx.stream.launch_builder(partial);
        partial_builder.arg(&scratch);
        partial_builder.arg(&a);
        partial_builder.arg(&b);
        partial_builder.arg(&m_i);
        partial_builder.arg(&k_i);
        partial_builder.arg(&n_i);
        partial_builder.arg(&m_chunk_i);
        unsafe { partial_builder.launch(partial_config) }
            .map_err(|error| format!("launch raw paired TN partial: {error:?}"))?;

        let total = u32::try_from(buffers.output_elements)
            .map_err(|_| "paired TN reducer extent exceeds u32::MAX")?;
        let reduce_config = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut reducer = ctx
            .stream
            .launch_builder(&ctx.kernels.gemm_bi_splitm_reduce);
        reducer.arg(&output);
        reducer.arg(&scratch);
        reducer.arg(&alpha);
        reducer.arg(&k_i);
        reducer.arg(&n_i);
        reducer.arg(&chunks_i);
        unsafe { reducer.launch(reduce_config) }
            .map(|_| ())
            .map_err(|error| format!("launch raw paired TN reducer: {error:?}"))
    }

    fn pointer_at_f32(base: u64, offset: usize, label: &str) -> Result<u64, String> {
        let byte_offset = offset
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("paired TN {label} byte offset overflows usize"))?;
        base.checked_add(
            u64::try_from(byte_offset)
                .map_err(|_| format!("paired TN {label} byte offset exceeds u64::MAX"))?,
        )
        .ok_or_else(|| format!("paired TN {label} pointer overflows u64"))
    }

    fn launch_cublas(ctx: &GpuCtx, buffers: &PairBuffers) -> Result<(), String> {
        let cell = buffers.cell;
        let geometry = cublas_denominator_geometry(ResolvedGemmOp::Tn, cell.dims);
        let a = pointer_at_f32(buffers.cublas.a.cached_ptr(), cell.a_offset, "cuBLAS A")?;
        let b = pointer_at_f32(buffers.cublas.b.cached_ptr(), cell.b_offset, "cuBLAS B")?;
        let output = pointer_at_f32(
            buffers.cublas.output.cached_ptr(),
            cell.output_offset,
            "cuBLAS output",
        )?;
        let alpha = 1.0f32;
        let beta = 1.0f32;
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *ctx.blas.handle(),
                geometry.trans_a,
                geometry.trans_b,
                geometry.m,
                geometry.n,
                geometry.k,
                &alpha as *const f32 as *const c_void,
                b as *const c_void,
                WeightDtype::F32.cuda_data_type(),
                geometry.lda,
                a as *const c_void,
                WeightDtype::F32.cuda_data_type(),
                geometry.ldb,
                &beta as *const f32 as *const c_void,
                output as *mut c_void,
                WeightDtype::F32.cuda_data_type(),
                geometry.ldc,
                cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|error| {
                format!("cuBLAS FAST_TF32 paired TN {} failed: {error:?}", cell.name)
            })?;
        }
        Ok(())
    }

    fn validate_candidate_route(ctx: &GpuCtx, buffers: &PairBuffers) -> Result<String, String> {
        let cell = buffers.cell;
        let a = buffers.candidate_x.raw_ptr_at(&ctx.stream, cell.a_offset);
        let b = buffers.candidate_dy.raw_ptr_at(&ctx.stream, cell.b_offset);
        let observed_symbol = if a.is_multiple_of(16) && b.is_multiple_of(16) {
            "gemm_bi_tn_narrow_splitm_partial_aligned"
        } else {
            "gemm_bi_tn_narrow_splitm_partial"
        };
        let expected_symbol = candidate_partial_symbol(cell);
        if observed_symbol != expected_symbol {
            return Err(format!(
                "paired TN {} expected {expected_symbol}, but pointer classes select {observed_symbol}",
                cell.name
            ));
        }
        launch_candidate(ctx, buffers)?;
        let plan = buffers.candidate_plan;
        let digest = FramedSha256::new(b"mamba-bi-f32-tn-narrow-cublas-pair-route.v1")
            .required(b"launch-count", &2u64.to_le_bytes())
            .required(b"partial-symbol", expected_symbol.as_bytes())
            .required(b"reducer-symbol", b"gemm_bi_splitm_reduce")
            .required(b"m", &(cell.dims.0 as u64).to_le_bytes())
            .required(b"k", &(cell.dims.1 as u64).to_le_bytes())
            .required(b"n", &(cell.dims.2 as u64).to_le_bytes())
            .required(b"a-offset-f32", &(cell.a_offset as u64).to_le_bytes())
            .required(b"b-offset-f32", &(cell.b_offset as u64).to_le_bytes())
            .required(
                b"output-offset-f32",
                &(cell.output_offset as u64).to_le_bytes(),
            )
            .required(b"m-chunk", &(plan.m_chunk as u64).to_le_bytes())
            .required(b"chunks", &(plan.chunks as u64).to_le_bytes())
            .required(b"partial-grid-m", &plan.grid_m.to_le_bytes())
            .required(b"partial-grid-n", &plan.grid_n.to_le_bytes())
            .finish();
        Ok(digest_hex(&digest))
    }

    struct PairGraphs {
        candidate: CudaGraph,
        cublas: CudaGraph,
    }

    fn require_graph_node_count(
        graph: &CudaGraph,
        expected: usize,
        label: &str,
    ) -> Result<(), String> {
        let mut actual = 0usize;
        let status = unsafe {
            cudarc::driver::sys::cuGraphGetNodes(
                graph.cu_graph(),
                std::ptr::null_mut(),
                &mut actual,
            )
        };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("query paired TN {label} graph nodes: {status:?}"));
        }
        if actual != expected {
            return Err(format!(
                "paired TN {label} graph has {actual} nodes instead of {expected}"
            ));
        }
        Ok(())
    }

    fn capture_graphs(ctx: &GpuCtx, buffers: &PairBuffers) -> Result<PairGraphs, String> {
        launch_candidate(ctx, buffers)?;
        launch_cublas(ctx, buffers)?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize paired TN warm launch: {error:?}"))?;
        let candidate =
            unsafe { capture_into_graph(&ctx.stream, || launch_candidate(ctx, buffers)) }?;
        let cublas = unsafe { capture_into_graph(&ctx.stream, || launch_cublas(ctx, buffers)) }?;
        require_graph_node_count(&candidate, 2, "candidate")?;
        Ok(PairGraphs { candidate, cublas })
    }

    struct PairRuntime<'a> {
        ctx: &'a GpuCtx,
        cell: PairCell,
        buffers: &'a mut PairBuffers,
        graphs: &'a PairGraphs,
    }

    fn launch_arm(runtime: &PairRuntime<'_>, path: PairPath, arm: PairArm) -> Result<(), String> {
        match (path, arm) {
            (PairPath::Eager, PairArm::Candidate) => launch_candidate(runtime.ctx, runtime.buffers),
            (PairPath::Eager, PairArm::Cublas) => launch_cublas(runtime.ctx, runtime.buffers),
            (PairPath::Graph, PairArm::Candidate) => runtime
                .graphs
                .candidate
                .launch()
                .map_err(|error| format!("launch paired TN candidate graph: {error:?}")),
            (PairPath::Graph, PairArm::Cublas) => runtime
                .graphs
                .cublas
                .launch()
                .map_err(|error| format!("launch paired TN cuBLAS graph: {error:?}")),
        }
    }

    fn measure_window_ms(
        runtime: &mut PairRuntime<'_>,
        path: PairPath,
        arm: PairArm,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("paired TN iteration count must be positive".into());
        }
        runtime.buffers.reset(runtime.ctx, arm)?;
        let start = runtime
            .ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record paired TN start event: {error:?}"))?;
        for _ in 0..iterations {
            launch_arm(runtime, path, arm)?;
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record paired TN end event: {error:?}"))?;
        start
            .elapsed_ms(&end)
            .map(f64::from)
            .map_err(|error| format!("measure paired TN events: {error:?}"))
    }

    #[derive(Clone, Copy)]
    struct PairIterations {
        candidate: usize,
        cublas: usize,
    }

    fn calibrate_pair(
        runtime: &mut PairRuntime<'_>,
        path: PairPath,
    ) -> Result<PairIterations, String> {
        measure_window_ms(runtime, path, PairArm::Candidate, WARMUP_LAUNCHES)?;
        measure_window_ms(runtime, path, PairArm::Cublas, WARMUP_LAUNCHES)?;
        let candidate = calibrate_measurement(|iterations| {
            measure_window_ms(runtime, path, PairArm::Candidate, iterations)
        })?;
        let cublas = calibrate_measurement(|iterations| {
            measure_window_ms(runtime, path, PairArm::Cublas, iterations)
        })?;
        Ok(PairIterations { candidate, cublas })
    }

    #[derive(Default)]
    struct OrderSamples {
        candidate_us: Vec<f64>,
        cublas_us: Vec<f64>,
        ratios: Vec<f64>,
    }

    struct PairSamples {
        candidate_first: OrderSamples,
        cublas_first: OrderSamples,
    }

    fn collect_samples(
        runtime: &mut PairRuntime<'_>,
        path: PairPath,
        iterations: PairIterations,
        windows: usize,
    ) -> Result<PairSamples, String> {
        let mut candidate_first = OrderSamples::default();
        let mut cublas_first = OrderSamples::default();
        for order in paired_schedule(windows)? {
            let sample = |runtime: &mut PairRuntime<'_>, arm, iterations| {
                let elapsed = measure_window_ms(runtime, path, arm, iterations)?;
                Ok::<f64, String>(elapsed * 1_000.0 / iterations as f64)
            };
            let (candidate, cublas) = match order {
                PairOrder::CandidateFirst => (
                    sample(runtime, PairArm::Candidate, iterations.candidate)?,
                    sample(runtime, PairArm::Cublas, iterations.cublas)?,
                ),
                PairOrder::CublasFirst => {
                    let cublas = sample(runtime, PairArm::Cublas, iterations.cublas)?;
                    let candidate = sample(runtime, PairArm::Candidate, iterations.candidate)?;
                    (candidate, cublas)
                }
            };
            let target = match order {
                PairOrder::CandidateFirst => &mut candidate_first,
                PairOrder::CublasFirst => &mut cublas_first,
            };
            let index = target.candidate_us.len();
            validate_sample_us(candidate, index)?;
            validate_sample_us(cublas, index)?;
            target.candidate_us.push(candidate);
            target.cublas_us.push(cublas);
            target.ratios.push(candidate / cublas);
        }
        for samples in [&candidate_first, &cublas_first] {
            paired_ratios(&samples.candidate_us, &samples.cublas_us, windows)?;
        }
        Ok(PairSamples {
            candidate_first,
            cublas_first,
        })
    }

    fn dense_probe(dims: (usize, usize, usize)) -> Result<SemanticProbe, String> {
        let (m, k, n) = dims;
        let pattern = [0.125f32, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0];
        let make = |len: usize, salt: usize| {
            (0..len)
                .map(|index| pattern[(index.wrapping_mul(5).wrapping_add(salt)) % pattern.len()])
                .collect::<Vec<_>>()
        };
        let x = make(m.checked_mul(k).ok_or("paired TN dense X overflow")?, 1);
        let dy = make(m.checked_mul(n).ok_or("paired TN dense dY overflow")?, 3);
        let seed = make(k.checked_mul(n).ok_or("paired TN dense seed overflow")?, 6);
        let expected = tn_reference(dims, &x, &dy, &seed)?
            .into_iter()
            .map(|value| value as f32)
            .collect();
        Ok(SemanticProbe {
            x,
            dy,
            seed,
            expected,
        })
    }

    fn output_digest(values: &[f32]) -> String {
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    fn require_exact(label: &str, actual: &[f32], expected: &[f32]) -> Result<(), String> {
        if actual.len() != expected.len() {
            return Err(format!("paired TN {label} output extent changed"));
        }
        if let Some((index, (actual, expected))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
        {
            return Err(format!(
                "paired TN {label} bit mismatch at {index}: 0x{:08x} != 0x{:08x}",
                actual.to_bits(),
                expected.to_bits()
            ));
        }
        Ok(())
    }

    fn require_close(label: &str, actual: &[f32], expected: &[f32]) -> Result<(), String> {
        if actual.len() != expected.len() {
            return Err(format!("paired TN {label} output extent changed"));
        }
        if let Some((index, (actual, expected))) =
            actual
                .iter()
                .zip(expected)
                .enumerate()
                .find(|(_, (actual, expected))| {
                    let tolerance = 0.25 + 0.02 * actual.abs().max(expected.abs());
                    !actual.is_finite() || (*actual - *expected).abs() > tolerance
                })
        {
            return Err(format!(
                "paired TN {label} tolerance mismatch at {index}: {actual} != {expected}"
            ));
        }
        Ok(())
    }

    #[derive(Clone, PartialEq, Eq)]
    struct SemanticEvidence {
        one_term_candidate: String,
        one_term_cublas: String,
        dense_candidate: String,
        dense_cublas: String,
        dense_cross_arm_bit_exact: bool,
    }

    fn run_semantics(
        runtime: &mut PairRuntime<'_>,
        path: PairPath,
    ) -> Result<SemanticEvidence, String> {
        let one_term = one_term_probe(runtime.cell.dims)?;
        runtime.buffers.upload_probe(runtime.ctx, &one_term)?;
        let mut one_term_digests = Vec::new();
        for arm in [PairArm::Candidate, PairArm::Cublas] {
            runtime.buffers.reset(runtime.ctx, arm)?;
            launch_arm(runtime, path, arm)?;
            let output = runtime.buffers.output(runtime.ctx, arm)?;
            require_exact("one-term", &output, &one_term.expected)?;
            one_term_digests.push(output_digest(&output));
        }

        let dense = dense_probe(runtime.cell.dims)?;
        runtime.buffers.upload_probe(runtime.ctx, &dense)?;
        let mut candidate_digest = None;
        for _ in 0..SEMANTIC_REPEATS {
            runtime.buffers.reset(runtime.ctx, PairArm::Candidate)?;
            launch_arm(runtime, path, PairArm::Candidate)?;
            let output = runtime.buffers.output(runtime.ctx, PairArm::Candidate)?;
            require_close("deterministic dense", &output, &dense.expected)?;
            let digest = output_digest(&output);
            if candidate_digest
                .as_ref()
                .is_some_and(|expected| expected != &digest)
            {
                return Err(format!(
                    "paired TN {} dense candidate was not bit-exact across repeats",
                    path.as_str()
                ));
            }
            candidate_digest = Some(digest);
        }
        runtime.buffers.reset(runtime.ctx, PairArm::Cublas)?;
        launch_arm(runtime, path, PairArm::Cublas)?;
        let cublas = runtime.buffers.output(runtime.ctx, PairArm::Cublas)?;
        require_close("cuBLAS FAST_TF32 dense", &cublas, &dense.expected)?;
        let cublas_digest = output_digest(&cublas);
        let candidate_digest = candidate_digest.expect("semantic repeat count is positive");
        runtime.buffers.validate_guards(runtime.ctx)?;
        Ok(SemanticEvidence {
            one_term_candidate: one_term_digests[0].clone(),
            one_term_cublas: one_term_digests[1].clone(),
            dense_cross_arm_bit_exact: candidate_digest == cublas_digest,
            dense_candidate: candidate_digest,
            dense_cublas: cublas_digest,
        })
    }

    fn raw_json(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn json_escape(value: &str) -> String {
        let mut output = String::with_capacity(value.len());
        for character in value.chars() {
            match character {
                '"' => output.push_str("\\\""),
                '\\' => output.push_str("\\\\"),
                '\n' => output.push_str("\\n"),
                '\r' => output.push_str("\\r"),
                '\t' => output.push_str("\\t"),
                character if character.is_control() => {
                    write!(output, "\\u{:04x}", character as u32)
                        .expect("String writes cannot fail");
                }
                character => output.push(character),
            }
        }
        output
    }

    fn inventory_digest() -> String {
        let mut hash = FramedSha256::new(b"mamba-bi-f32-tn-narrow-cublas-pair.v1");
        for cell in CELLS {
            hash = hash
                .required(b"name", cell.name.as_bytes())
                .required(b"m", &cell.dims.0.to_le_bytes())
                .required(b"k", &cell.dims.1.to_le_bytes())
                .required(b"n", &cell.dims.2.to_le_bytes())
                .required(b"a-offset-f32", &cell.a_offset.to_le_bytes())
                .required(b"b-offset-f32", &cell.b_offset.to_le_bytes())
                .required(b"output-offset-f32", &cell.output_offset.to_le_bytes());
        }
        digest_hex(&hash.finish())
    }

    struct PairJsonlSink {
        path: PathBuf,
        writer: BufWriter<File>,
        digest: Sha256,
        records: usize,
    }

    impl PairJsonlSink {
        fn create_from_env() -> Result<Self, String> {
            let path = PathBuf::from(
                std::env::var(OUTPUT_ENV)
                    .map_err(|_| format!("{OUTPUT_ENV} must name a new JSONL evidence file"))?,
            );
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| format!("create paired TN evidence {path:?}: {error}"))?;
            Ok(Self {
                path,
                writer: BufWriter::new(file),
                digest: Sha256::new(),
                records: 0,
            })
        }

        fn write(&mut self, line: String) -> Result<(), String> {
            let mut bytes = line.into_bytes();
            bytes.push(b'\n');
            self.writer
                .write_all(&bytes)
                .map_err(|error| format!("write paired TN evidence {:?}: {error}", self.path))?;
            self.digest.update(&bytes);
            self.records += 1;
            Ok(())
        }

        fn record(&mut self, record: &PairRecord<'_>) -> Result<(), String> {
            let (m, k, n) = record.cell.dims;
            let arm = |order: PairOrder, samples: &OrderSamples| {
                format!(
                    concat!(
                        "{{\"order\":\"{}\",\"candidate_us\":[{}],",
                        "\"cublas_us\":[{}],\"candidate_over_cublas\":[{}]}}"
                    ),
                    order.as_str(),
                    raw_json(&samples.candidate_us),
                    raw_json(&samples.cublas_us),
                    raw_json(&samples.ratios),
                )
            };
            self.write(format!(
                concat!(
                    "{{\"schema\":\"{}\",\"scope\":\"performance_only\",",
                    "\"decision\":\"measurement_only_no_auto_admission\",",
                    "\"cell\":\"{}\",\"op\":\"tn\",\"dtype\":\"f32\",",
                    "\"alpha\":1.0,\"beta\":1.0,\"m\":{},\"k\":{},\"n\":{},",
                    "\"a_offset_f32\":{},\"b_offset_f32\":{},\"output_offset_f32\":{},",
                    "\"candidate\":\"forced_tn_narrow_splitm_two_launch\",",
                    "\"candidate_partial_symbol\":\"{}\",",
                    "\"candidate_reduce_symbol\":\"gemm_bi_splitm_reduce\",",
                    "\"candidate_m_chunk\":{},\"candidate_chunks\":{},",
                    "\"denominator\":\"cublas_fast_tf32\",\"path\":\"{}\",",
                    "\"windows_per_order\":{},\"warmups\":{},",
                    "\"candidate_iterations\":{},\"cublas_iterations\":{},",
                    "\"route_digest\":\"{}\",\"red_zones_validated\":true,",
                    "\"guard_elements_per_allocation\":{},",
                    "\"semantic\":{{\"one_term_candidate\":\"{}\",",
                    "\"one_term_cublas\":\"{}\",\"dense_candidate\":\"{}\",",
                    "\"dense_cublas\":\"{}\",\"dense_cross_arm_bit_exact\":{}}},",
                    "\"preflight\":\"{}\",\"postflight\":\"{}\",",
                    "\"orders\":[{},{}]}}"
                ),
                RECORD_SCHEMA,
                record.cell.name,
                m,
                k,
                n,
                record.cell.a_offset,
                record.cell.b_offset,
                record.cell.output_offset,
                candidate_partial_symbol(record.cell),
                record.plan.m_chunk,
                record.plan.chunks,
                record.path.as_str(),
                record.windows,
                WARMUP_LAUNCHES,
                record.iterations.candidate,
                record.iterations.cublas,
                record.route_digest,
                GUARD_ELEMENTS,
                record.semantic.one_term_candidate,
                record.semantic.one_term_cublas,
                record.semantic.dense_candidate,
                record.semantic.dense_cublas,
                record.semantic.dense_cross_arm_bit_exact,
                json_escape(record.preflight),
                json_escape(record.postflight),
                arm(PairOrder::CandidateFirst, &record.samples.candidate_first),
                arm(PairOrder::CublasFirst, &record.samples.cublas_first),
            ))
        }

        fn finish(
            mut self,
            windows: usize,
            quiet_gpu: &QuietGpu,
            device: &GpuDevice,
        ) -> Result<(), String> {
            let expected = CELLS.len() * 2;
            if self.records != expected {
                return Err(format!(
                    "incomplete paired TN evidence: got {}, expected {expected}",
                    self.records
                ));
            }
            let cohort_digest = format!("{:x}", self.digest.clone().finalize());
            self.write(format!(
                concat!(
                    "{{\"schema\":\"{}\",\"dataset_complete\":true,",
                    "\"measurement_only\":true,\"auto_admitted\":false,",
                    "\"decision\":\"measurement_only_no_auto_admission\",",
                    "\"gpu_uuid\":\"{}\",\"cc\":\"{}.{}\",",
                    "\"cells\":{},\"paths_per_cell\":2,\"windows_per_order\":{},",
                    "\"measurement_records\":{},\"total_jsonl_records\":{},",
                    "\"inventory_digest\":\"{}\",\"cohort_digest\":\"{}\"}}"
                ),
                COMPLETION_SCHEMA,
                quiet_gpu.uuid(),
                device.compute_capability.0,
                device.compute_capability.1,
                CELLS.len(),
                windows,
                expected,
                expected + 1,
                inventory_digest(),
                cohort_digest,
            ))?;
            self.writer
                .flush()
                .map_err(|error| format!("flush paired TN evidence {:?}: {error}", self.path))?;
            self.writer
                .get_ref()
                .sync_all()
                .map_err(|error| format!("sync paired TN evidence {:?}: {error}", self.path))
        }
    }

    struct PairRecord<'a> {
        cell: PairCell,
        path: PairPath,
        windows: usize,
        iterations: PairIterations,
        plan: CandidatePlan,
        route_digest: &'a str,
        semantic: &'a SemanticEvidence,
        samples: &'a PairSamples,
        preflight: &'a str,
        postflight: &'a str,
    }

    fn run_cell(
        ctx: &GpuCtx,
        quiet_gpu: &QuietGpu,
        cell: PairCell,
        multiprocessors: u32,
        windows: usize,
        sink: &mut PairJsonlSink,
    ) -> Result<(), String> {
        let mut buffers = PairBuffers::new(ctx, cell, multiprocessors)?;
        let dense = dense_probe(cell.dims)?;
        buffers.upload_probe(ctx, &dense)?;
        buffers.reset(ctx, PairArm::Candidate)?;
        let route_digest = validate_candidate_route(ctx, &buffers)?;
        buffers.reset(ctx, PairArm::Candidate)?;
        let graphs = capture_graphs(ctx, &buffers)?;
        let mut runtime = PairRuntime {
            ctx,
            cell,
            buffers: &mut buffers,
            graphs: &graphs,
        };
        let eager_semantic = run_semantics(&mut runtime, PairPath::Eager)?;
        let graph_semantic = run_semantics(&mut runtime, PairPath::Graph)?;
        if eager_semantic != graph_semantic {
            return Err("paired TN eager and graph semantic evidence differs".into());
        }
        runtime.buffers.upload_probe(ctx, &dense)?;
        for path in [PairPath::Eager, PairPath::Graph] {
            let label = format!("tn-narrow-pair/{}/{}/pre", cell.name, path.as_str());
            let preflight = quiet_gpu.require_cohort(&label)?;
            let iterations = calibrate_pair(&mut runtime, path)?;
            let samples = collect_samples(&mut runtime, path, iterations, windows)?;
            runtime.buffers.validate_guards(ctx)?;
            let label = format!("tn-narrow-pair/{}/{}/post", cell.name, path.as_str());
            let postflight = quiet_gpu.verify_post_cohort(&label)?;
            sink.record(&PairRecord {
                cell,
                path,
                windows,
                iterations,
                plan: runtime.buffers.candidate_plan,
                route_digest: &route_digest,
                semantic: &eager_semantic,
                samples: &samples,
                preflight: &preflight,
                postflight: &postflight,
            })?;
        }
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize paired TN graphs before drop: {error:?}"))
    }

    pub(super) fn run_suite() -> Result<(), String> {
        let windows = window_count(std::env::var(FINAL_ENV).ok().as_deref())?;
        validate_run_profile(windows, cfg!(debug_assertions))?;
        let mut sink = PairJsonlSink::create_from_env()?;
        let quiet_gpu = QuietGpu::for_cuda_ordinal(0)?;
        quiet_gpu.require_pre_context("tn-narrow-pair/pre-context")?;
        let device = GpuDevice::new(0)?;
        if device.compute_capability < (8, 0) {
            return Err("paired TN comparator requires SM80 or newer".into());
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        let multiprocessors = device.multiprocessor_count();
        for cell in CELLS {
            run_cell(&ctx, &quiet_gpu, cell, multiprocessors, windows, &mut sink)
                .map_err(|error| format!("{}: {error}", cell.name))?;
        }
        sink.finish(windows, &quiet_gpu, &device)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Tf32TournamentRoute {
    Direct,
    SplitK2,
    SplitK4,
}

impl Tf32TournamentRoute {
    const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct_m16n32_s4",
            Self::SplitK2 => "splitk2_m16n32_s4",
            Self::SplitK4 => "splitk4_m16n32_s4",
        }
    }

    const fn physical_route(self) -> Tf32PhysicalRoute {
        let portable = Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        };
        match self {
            Self::Direct => Tf32PhysicalRoute::MmaTf32RnaV1(portable),
            Self::SplitK2 => Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(portable),
            Self::SplitK4 => Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(portable),
        }
    }

    const fn symbol(self) -> &'static str {
        match self {
            Self::Direct => "gemm_bi_nn_sm80_mma_tf32_v1_m16n32_bk32_s4",
            Self::SplitK2 => "gemm_bi_nn_sm80_mma_tf32_splitk2_v1_m16n32_bk32_s4",
            Self::SplitK4 => "gemm_bi_nn_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4",
        }
    }

    const fn expected_grid(self) -> (u32, u32, u32) {
        match self {
            Self::Direct => (48, 1, 1),
            Self::SplitK2 => (12, 4, 2),
            Self::SplitK4 => (12, 4, 4),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tf32TournamentPath {
    Eager,
    Graph,
}

impl Tf32TournamentPath {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Graph => "graph",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tf32TournamentComparison {
    RoutePair {
        candidate: Tf32TournamentRoute,
        denominator: Tf32TournamentRoute,
    },
    CublasFastTf32 {
        candidate: Tf32TournamentRoute,
    },
}

impl Tf32TournamentComparison {
    const fn kind(self) -> &'static str {
        match self {
            Self::RoutePair { .. } => "deterministic_route_pair",
            Self::CublasFastTf32 { .. } => "cublas_fast_tf32_denominator",
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::RoutePair {
                candidate: Tf32TournamentRoute::Direct,
                denominator: Tf32TournamentRoute::SplitK2,
            } => "direct_m16n32_s4_vs_splitk2_m16n32_s4",
            Self::RoutePair {
                candidate: Tf32TournamentRoute::Direct,
                denominator: Tf32TournamentRoute::SplitK4,
            } => "direct_m16n32_s4_vs_splitk4_m16n32_s4",
            Self::RoutePair {
                candidate: Tf32TournamentRoute::SplitK2,
                denominator: Tf32TournamentRoute::SplitK4,
            } => "splitk2_m16n32_s4_vs_splitk4_m16n32_s4",
            Self::CublasFastTf32 {
                candidate: Tf32TournamentRoute::Direct,
            } => "direct_m16n32_s4_vs_cublas_fast_tf32",
            Self::CublasFastTf32 {
                candidate: Tf32TournamentRoute::SplitK2,
            } => "splitk2_m16n32_s4_vs_cublas_fast_tf32",
            Self::CublasFastTf32 {
                candidate: Tf32TournamentRoute::SplitK4,
            } => "splitk4_m16n32_s4_vs_cublas_fast_tf32",
            Self::RoutePair { .. } => "unsupported_route_pair",
        }
    }

    const fn candidate(self) -> Tf32TournamentRoute {
        match self {
            Self::RoutePair { candidate, .. } | Self::CublasFastTf32 { candidate } => candidate,
        }
    }

    const fn denominator_route(self) -> Option<Tf32TournamentRoute> {
        match self {
            Self::RoutePair { denominator, .. } => Some(denominator),
            Self::CublasFastTf32 { .. } => None,
        }
    }

    const fn denominator_name(self) -> &'static str {
        match self.denominator_route() {
            Some(route) => route.name(),
            None => "cublas_fast_tf32",
        }
    }

    const fn denominator_symbol(self) -> &'static str {
        match self.denominator_route() {
            Some(route) => route.symbol(),
            None => "cublasGemmEx",
        }
    }
}

const TF32_TOURNAMENT_ROUTES: [Tf32TournamentRoute; 3] = [
    Tf32TournamentRoute::Direct,
    Tf32TournamentRoute::SplitK2,
    Tf32TournamentRoute::SplitK4,
];
const TF32_TOURNAMENT_PATHS: [Tf32TournamentPath; 2] =
    [Tf32TournamentPath::Eager, Tf32TournamentPath::Graph];
const TF32_TOURNAMENT_ORDERS: [PathOrder; 2] = [PathOrder::Ab, PathOrder::Ba];
const TF32_TOURNAMENT_SHAPES: [Shape; 2] = [
    Shape {
        name: "m64_k1536_n384",
        dims: (64, 1_536, 384),
    },
    Shape {
        name: "m64_k833_n384",
        dims: (64, 833, 384),
    },
];
const TF32_TOURNAMENT_COMPARISONS: [Tf32TournamentComparison; 6] = [
    Tf32TournamentComparison::RoutePair {
        candidate: Tf32TournamentRoute::Direct,
        denominator: Tf32TournamentRoute::SplitK2,
    },
    Tf32TournamentComparison::RoutePair {
        candidate: Tf32TournamentRoute::Direct,
        denominator: Tf32TournamentRoute::SplitK4,
    },
    Tf32TournamentComparison::RoutePair {
        candidate: Tf32TournamentRoute::SplitK2,
        denominator: Tf32TournamentRoute::SplitK4,
    },
    Tf32TournamentComparison::CublasFastTf32 {
        candidate: Tf32TournamentRoute::Direct,
    },
    Tf32TournamentComparison::CublasFastTf32 {
        candidate: Tf32TournamentRoute::SplitK2,
    },
    Tf32TournamentComparison::CublasFastTf32 {
        candidate: Tf32TournamentRoute::SplitK4,
    },
];
const TF32_TOURNAMENT_ASSIGNMENTS: usize = 2;

const fn tf32_tournament_records_per_assignment() -> usize {
    TF32_TOURNAMENT_SHAPES.len()
        * TF32_TOURNAMENT_PATHS.len()
        * TF32_TOURNAMENT_ORDERS.len()
        * TF32_TOURNAMENT_COMPARISONS.len()
}

const fn tf32_tournament_expected_records() -> usize {
    tf32_tournament_records_per_assignment() * TF32_TOURNAMENT_ASSIGNMENTS
}

const fn tf32_tournament_total_jsonl_records() -> usize {
    tf32_tournament_expected_records() + 1
}

#[derive(Clone, Copy)]
struct Tf32TournamentIterations {
    candidate: usize,
    denominator: usize,
}

#[derive(Clone, Copy)]
struct Tf32TournamentCohort {
    path: Tf32TournamentPath,
    order: PathOrder,
    windows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32TournamentSemanticEvidence {
    candidate_one_term_digest: [u8; 32],
    denominator_one_term_digest: [u8; 32],
    candidate_dense_digest: [u8; 32],
    denominator_dense_digest: [u8; 32],
    dense_cross_arm_bit_exact: bool,
}

struct Tf32TournamentOneTermVectors {
    a: Vec<f32>,
    b: Vec<f32>,
    output: Vec<f32>,
    expected: Vec<u32>,
}

struct Tf32TournamentRuntime<'a> {
    primary_ctx: &'a GpuCtx,
    secondary_ctx: &'a GpuCtx,
    quiet_gpu: &'a QuietGpu,
    assignment: &'static str,
    assignment_preflight_snapshot: &'a str,
    multiprocessors: u32,
}

struct Tf32TournamentJsonlSink {
    path: PathBuf,
    writer: BufWriter<File>,
    digest: Sha256,
    records: usize,
}

impl Tf32TournamentJsonlSink {
    fn create_from_env() -> Result<Self, String> {
        let value = std::env::var_os(TF32_TOURNAMENT_OUTPUT_ENV)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("{TF32_TOURNAMENT_OUTPUT_ENV} must name a new JSONL file"))?;
        let path = PathBuf::from(value);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("create TF32 tournament evidence {path:?}: {error}"))?;
        if !file
            .metadata()
            .map_err(|error| format!("inspect TF32 tournament evidence {path:?}: {error}"))?
            .file_type()
            .is_file()
        {
            return Err(format!(
                "TF32 tournament evidence {path:?} is not a regular file"
            ));
        }
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            digest: Sha256::new(),
            records: 0,
        })
    }

    fn write(&mut self, record: String) -> Result<(), String> {
        let mut bytes = record.into_bytes();
        bytes.push(b'\n');
        self.writer
            .write_all(&bytes)
            .map_err(|error| format!("write TF32 tournament evidence {:?}: {error}", self.path))?;
        self.digest.update(&bytes);
        self.records += 1;
        Ok(())
    }

    fn finish(
        mut self,
        expected_records: usize,
        quiet_gpu: &QuietGpu,
        device: &GpuDevice,
    ) -> Result<(), String> {
        if self.records != expected_records {
            return Err(format!(
                "incomplete TF32 tournament evidence: got {} records, expected {expected_records}",
                self.records
            ));
        }
        let prior_digest = format!("{:x}", self.digest.clone().finalize());
        self.write(format!(
            "{{\"schema\":\"MambaBiTf32NnTournamentCompletionV1\",\"cohort_valid\":true,\"gpu_uuid\":\"{}\",\"cc\":\"{}.{}\",\"multiprocessors\":{},\"cohort_records\":{},\"total_jsonl_records\":{},\"cohort_digest\":\"{}\",\"decision\":\"measurement_only_no_auto_admission\"}}",
            quiet_gpu.uuid(),
            device.compute_capability.0,
            device.compute_capability.1,
            device.multiprocessor_count(),
            expected_records,
            tf32_tournament_total_jsonl_records(),
            prior_digest,
        ))?;
        self.writer
            .flush()
            .map_err(|error| format!("flush TF32 tournament evidence {:?}: {error}", self.path))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("sync TF32 tournament evidence {:?}: {error}", self.path))?;
        eprintln!(
            "TF32 tournament evidence complete: {:?}; records={}; digest={prior_digest}",
            self.path, self.records
        );
        Ok(())
    }
}

struct Tf32TournamentSamples {
    candidate_us: Vec<f64>,
    denominator_us: Vec<f64>,
    ratios: Vec<f64>,
}

struct Tf32TournamentSummary {
    candidate_p50: f64,
    candidate_p95: f64,
    denominator_p50: f64,
    denominator_p95: f64,
    ratio_p50: f64,
    ratio_p95: f64,
}

struct Tf32TournamentPhysicalPair<'a, 'q> {
    candidate: &'a mut QualifiedPhysicalLaunch<'q>,
    denominator: &'a mut QualifiedPhysicalLaunch<'q>,
}

#[derive(Clone, Copy)]
struct Tf32TournamentCublasComparator<'a> {
    cell: CublasDenominatorCell,
    buffers: &'a CublasDenominatorBuffers,
    graph: &'a CudaGraph,
}

struct Tf32TournamentCublasPair<'a, 'q> {
    candidate: &'a mut QualifiedPhysicalLaunch<'q>,
    comparator: Tf32TournamentCublasComparator<'a>,
}

struct Tf32TournamentRecordMeta<'a> {
    runtime: &'a Tf32TournamentRuntime<'a>,
    semantic: &'a Tf32TournamentSemanticEvidence,
    candidate: &'a QualifiedPhysicalLaunch<'a>,
    denominator: Option<&'a QualifiedPhysicalLaunch<'a>>,
    calibration_preflight: &'a str,
    preflight: &'a str,
    postflight: &'a str,
}

fn tf32_tournament_request(
    shape: Shape,
    route: Tf32TournamentRoute,
) -> PhysicalQualificationRequest {
    PhysicalQualificationRequest::contiguous(
        ResolvedGemmOp::Nn,
        shape.dims,
        PhysicalQualificationRoute::Tf32Forced(route.physical_route()),
    )
}

fn validate_tf32_tournament_manifest(
    route: Tf32TournamentRoute,
    physical: &QualifiedPhysicalLaunch<'_>,
) -> Result<(), String> {
    let evidence = physical.evidence();
    let nodes = evidence.nodes();
    if !evidence.eager_graph_equal()
        || evidence.launch_count() != 1
        || evidence.single_launch_symbol() != Some(route.symbol())
        || evidence.single_launch_tile() != Some((16, 32))
        || nodes[0].symbol != route.symbol()
        || nodes[0].tile != Some((16, 32))
        || nodes[0].launch.grid_dim != route.expected_grid()
        || nodes[0].launch.block_dim != (128, 1, 1)
        || nodes[0].launch.shared_mem_bytes != 29_696
    {
        return Err(format!(
            "TF32 tournament {} physical manifest changed: {:?}",
            route.name(),
            evidence.nodes()
        ));
    }
    Ok(())
}

fn measure_tf32_tournament_physical_window_ms(
    ctx: &GpuCtx,
    physical: &mut QualifiedPhysicalLaunch<'_>,
    path: Tf32TournamentPath,
    iterations: usize,
) -> Result<f64, String> {
    match path {
        Tf32TournamentPath::Eager => physical.measure_eager_window_ms(ctx, iterations),
        Tf32TournamentPath::Graph => physical.measure_graph_window_ms(ctx, iterations),
    }
}

fn measure_tf32_tournament_graph_window_ms(
    ctx: &GpuCtx,
    graph: &CudaGraph,
    iterations: usize,
) -> Result<f64, String> {
    if iterations == 0 {
        return Err("TF32 tournament graph iteration count must be positive".into());
    }
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("record TF32 tournament graph start event: {error:?}"))?;
    for _ in 0..iterations {
        graph
            .launch()
            .map_err(|error| format!("launch TF32 tournament graph: {error:?}"))?;
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("record TF32 tournament graph end event: {error:?}"))?;
    start
        .elapsed_ms(&end)
        .map(f64::from)
        .map_err(|error| format!("measure TF32 tournament graph events: {error:?}"))
}

fn measure_tf32_tournament_cublas_window_ms(
    runtime: &Tf32TournamentRuntime<'_>,
    comparator: Tf32TournamentCublasComparator<'_>,
    path: Tf32TournamentPath,
    iterations: usize,
) -> Result<f64, String> {
    match path {
        Tf32TournamentPath::Eager => measure_cublas_denominator_window_ms(
            runtime.secondary_ctx,
            comparator.cell,
            CublasDenominatorMode::Fast,
            comparator.buffers,
            iterations,
        ),
        Tf32TournamentPath::Graph => measure_tf32_tournament_graph_window_ms(
            runtime.secondary_ctx,
            comparator.graph,
            iterations,
        ),
    }
}

fn capture_tf32_tournament_cublas_graph(
    runtime: &Tf32TournamentRuntime<'_>,
    cell: CublasDenominatorCell,
    buffers: &CublasDenominatorBuffers,
) -> Result<CudaGraph, String> {
    // The secondary context, cuBLAS handle, and buffers outlive the captured graph.
    unsafe {
        capture_into_graph(&runtime.secondary_ctx.stream, || {
            launch_cublas_denominator(
                runtime.secondary_ctx,
                cell,
                CublasDenominatorMode::Fast,
                buffers,
            )
        })
    }
}

fn tf32_tournament_seeded_values(len: usize, salt: u64) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
    (0..len)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let signed = ((state.wrapping_add(index as u64) % 4093) as i32) - 2046;
            signed as f32 / 1024.0
        })
        .collect()
}

fn tf32_tournament_output_digest(bits: &[u32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for value in bits {
        digest.update(value.to_le_bytes());
    }
    digest.finalize().into()
}

fn tf32_tournament_split_partition_edges(
    k: usize,
    partitions: usize,
) -> Result<Vec<usize>, String> {
    if k == 0 || !matches!(partitions, 2 | 4) {
        return Err("TF32 tournament partition edges require K > 0 and 2 or 4 partitions".into());
    }
    let full_tiles = k.div_ceil(32);
    let tiles_per_partition = full_tiles.div_ceil(partitions);
    let mut edges = Vec::with_capacity(partitions * 2);
    for partition in 0..partitions {
        let begin = (partition * tiles_per_partition * 32).min(k - 1);
        let end = ((partition + 1) * tiles_per_partition * 32).min(k) - 1;
        edges.push(begin);
        edges.push(end.max(begin));
    }
    edges.dedup();
    Ok(edges)
}

fn validate_tf32_tournament_dense_partition_activation(shape: Shape) -> Result<(), String> {
    const SALT: u64 = 0x7b31_4d29;
    let (m, k, n) = shape.dims;
    let a = tf32_tournament_seeded_values(m * k, SALT ^ 0x2d);
    let b = tf32_tournament_seeded_values(k * n, SALT ^ 0x67);
    for partitions in [2, 4] {
        for reduction in tf32_tournament_split_partition_edges(k, partitions)? {
            if a[reduction] == 0.0 || b[reduction * n] == 0.0 {
                return Err(format!(
                    "{} has an inactive K={reduction} edge for split-K{partitions}",
                    shape.name
                ));
            }
        }
    }
    Ok(())
}

fn compare_tf32_tournament_dense_outputs(
    candidate: &[u32],
    denominator: &[u32],
) -> Result<bool, String> {
    if candidate.is_empty() || candidate.len() != denominator.len() {
        return Err("TF32 dense semantic outputs have different or empty extents".into());
    }
    let bit_exact = candidate == denominator;
    for (index, (&candidate_bits, &denominator_bits)) in
        candidate.iter().zip(denominator).enumerate()
    {
        let candidate_value = f32::from_bits(candidate_bits);
        let denominator_value = f32::from_bits(denominator_bits);
        if !candidate_value.is_finite() || !denominator_value.is_finite() {
            return Err(format!("TF32 dense semantic output {index} is not finite"));
        }
        let difference = (candidate_value - denominator_value).abs();
        let scale = candidate_value.abs().max(denominator_value.abs());
        let tolerance = 0.25 + 0.02 * scale;
        if difference > tolerance {
            return Err(format!(
                "TF32 dense semantic output {index} differs by {difference}, tolerance {tolerance}"
            ));
        }
    }
    Ok(bit_exact)
}

fn tf32_tournament_one_term_vectors(shape: Shape) -> Result<Tf32TournamentOneTermVectors, String> {
    let (m, k, n) = shape.dims;
    let mut a = vec![0.0; m.checked_mul(k).ok_or("TF32 semantic A overflow")?];
    let mut b = vec![0.0; k.checked_mul(n).ok_or("TF32 semantic B overflow")?];
    let output = vec![0.0; m.checked_mul(n).ok_or("TF32 semantic output overflow")?];
    for row in 0..m {
        a[row * k] = ((row % 7) as i32 - 3) as f32 * 0.125;
    }
    for (column, value) in b.iter_mut().take(n).enumerate() {
        *value = ((column % 11) as i32 - 5) as f32 * 0.125;
    }
    let mut expected = Vec::with_capacity(output.len());
    for row in 0..m {
        for &b_value in b.iter().take(n) {
            expected.push(a[row * k].mul_add(b_value, 0.0).to_bits());
        }
    }
    Ok(Tf32TournamentOneTermVectors {
        a,
        b,
        output,
        expected,
    })
}

fn require_tf32_tournament_output(
    label: &str,
    actual: &[u32],
    expected: &[u32],
) -> Result<(), String> {
    if actual == expected {
        return Ok(());
    }
    let index = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
        .unwrap_or(actual.len().min(expected.len()));
    Err(format!(
        "{label} semantic output differs at flat index {index}"
    ))
}

fn validate_tf32_tournament_route_pair_semantics(
    runtime: &Tf32TournamentRuntime<'_>,
    shape: Shape,
    comparison: Tf32TournamentComparison,
    mut pair: Tf32TournamentPhysicalPair<'_, '_>,
    path: Tf32TournamentPath,
) -> Result<Tf32TournamentSemanticEvidence, String> {
    let denominator_route = comparison
        .denominator_route()
        .ok_or("route-pair semantic probe requires a deterministic denominator")?;
    pair.candidate
        .seed_f32_nn_single_term_probe(runtime.primary_ctx)?;
    pair.denominator
        .seed_f32_nn_single_term_probe(runtime.secondary_ctx)?;
    measure_tf32_tournament_physical_window_ms(runtime.primary_ctx, pair.candidate, path, 1)?;
    measure_tf32_tournament_physical_window_ms(runtime.secondary_ctx, pair.denominator, path, 1)?;
    let candidate = pair.candidate.f32_output_bits(runtime.primary_ctx)?;
    let denominator = pair.denominator.f32_output_bits(runtime.secondary_ctx)?;
    let vectors = tf32_tournament_one_term_vectors(shape)?;
    require_tf32_tournament_output(comparison.candidate().name(), &candidate, &vectors.expected)?;
    require_tf32_tournament_output(denominator_route.name(), &denominator, &vectors.expected)?;
    let candidate_one_term_digest = tf32_tournament_output_digest(&candidate);
    let denominator_one_term_digest = tf32_tournament_output_digest(&denominator);
    seed_tf32_tournament_route_pair_dense_inputs(runtime, shape, &mut pair)?;
    measure_tf32_tournament_physical_window_ms(runtime.primary_ctx, pair.candidate, path, 1)?;
    measure_tf32_tournament_physical_window_ms(runtime.secondary_ctx, pair.denominator, path, 1)?;
    let candidate_dense = pair.candidate.f32_output_bits(runtime.primary_ctx)?;
    let denominator_dense = pair.denominator.f32_output_bits(runtime.secondary_ctx)?;
    let dense_cross_arm_bit_exact =
        compare_tf32_tournament_dense_outputs(&candidate_dense, &denominator_dense)?;
    Ok(Tf32TournamentSemanticEvidence {
        candidate_one_term_digest,
        denominator_one_term_digest,
        candidate_dense_digest: tf32_tournament_output_digest(&candidate_dense),
        denominator_dense_digest: tf32_tournament_output_digest(&denominator_dense),
        dense_cross_arm_bit_exact,
    })
}

fn validate_tf32_tournament_cublas_semantics(
    runtime: &Tf32TournamentRuntime<'_>,
    shape: Shape,
    candidate: &mut QualifiedPhysicalLaunch<'_>,
    comparator: Tf32TournamentCublasComparator<'_>,
    path: Tf32TournamentPath,
) -> Result<Tf32TournamentSemanticEvidence, String> {
    candidate.seed_f32_nn_single_term_probe(runtime.primary_ctx)?;
    let vectors = tf32_tournament_one_term_vectors(shape)?;
    comparator
        .buffers
        .a
        .upload_f32(&runtime.secondary_ctx.stream, &vectors.a)?;
    comparator
        .buffers
        .b
        .upload_f32(&runtime.secondary_ctx.stream, &vectors.b)?;
    comparator
        .buffers
        .output
        .upload_f32(&runtime.secondary_ctx.stream, &vectors.output)?;
    measure_tf32_tournament_physical_window_ms(runtime.primary_ctx, candidate, path, 1)?;
    match path {
        Tf32TournamentPath::Eager => launch_cublas_denominator(
            runtime.secondary_ctx,
            comparator.cell,
            CublasDenominatorMode::Fast,
            comparator.buffers,
        )?,
        Tf32TournamentPath::Graph => comparator
            .graph
            .launch()
            .map_err(|error| format!("launch TF32 tournament semantic graph: {error:?}"))?,
    }
    runtime
        .secondary_ctx
        .stream
        .synchronize()
        .map_err(|error| format!("synchronize TF32 tournament semantic probe: {error:?}"))?;
    let candidate_bits = candidate.f32_output_bits(runtime.primary_ctx)?;
    let mut denominator_values = vec![0.0; vectors.output.len()];
    comparator
        .buffers
        .output
        .download_f32(&runtime.secondary_ctx.stream, &mut denominator_values)?;
    let denominator_bits = denominator_values
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    require_tf32_tournament_output("deterministic route", &candidate_bits, &vectors.expected)?;
    require_tf32_tournament_output("cuBLAS FAST_TF32", &denominator_bits, &vectors.expected)?;
    let candidate_one_term_digest = tf32_tournament_output_digest(&candidate_bits);
    let denominator_one_term_digest = tf32_tournament_output_digest(&denominator_bits);
    seed_tf32_tournament_cublas_dense_inputs(runtime, shape, candidate, comparator.buffers)?;
    measure_tf32_tournament_physical_window_ms(runtime.primary_ctx, candidate, path, 1)?;
    match path {
        Tf32TournamentPath::Eager => launch_cublas_denominator(
            runtime.secondary_ctx,
            comparator.cell,
            CublasDenominatorMode::Fast,
            comparator.buffers,
        )?,
        Tf32TournamentPath::Graph => comparator
            .graph
            .launch()
            .map_err(|error| format!("launch TF32 tournament dense graph: {error:?}"))?,
    }
    runtime
        .secondary_ctx
        .stream
        .synchronize()
        .map_err(|error| format!("synchronize TF32 tournament dense probe: {error:?}"))?;
    let candidate_dense = candidate.f32_output_bits(runtime.primary_ctx)?;
    let mut denominator_dense_values = vec![0.0; vectors.output.len()];
    comparator
        .buffers
        .output
        .download_f32(&runtime.secondary_ctx.stream, &mut denominator_dense_values)?;
    let denominator_dense = denominator_dense_values
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let dense_cross_arm_bit_exact =
        compare_tf32_tournament_dense_outputs(&candidate_dense, &denominator_dense)?;
    Ok(Tf32TournamentSemanticEvidence {
        candidate_one_term_digest,
        denominator_one_term_digest,
        candidate_dense_digest: tf32_tournament_output_digest(&candidate_dense),
        denominator_dense_digest: tf32_tournament_output_digest(&denominator_dense),
        dense_cross_arm_bit_exact,
    })
}

fn seed_tf32_tournament_route_pair_dense_inputs(
    runtime: &Tf32TournamentRuntime<'_>,
    shape: Shape,
    pair: &mut Tf32TournamentPhysicalPair<'_, '_>,
) -> Result<(), String> {
    const SALT: u64 = 0x7b31_4d29;
    validate_tf32_tournament_dense_partition_activation(shape)?;
    pair.candidate
        .seed_f32_operands(runtime.primary_ctx, SALT)?;
    pair.denominator
        .seed_f32_operands(runtime.secondary_ctx, SALT)
}

fn seed_tf32_tournament_cublas_dense_inputs(
    runtime: &Tf32TournamentRuntime<'_>,
    shape: Shape,
    candidate: &mut QualifiedPhysicalLaunch<'_>,
    buffers: &CublasDenominatorBuffers,
) -> Result<(), String> {
    const SALT: u64 = 0x7b31_4d29;
    let (m, k, n) = shape.dims;
    validate_tf32_tournament_dense_partition_activation(shape)?;
    candidate.seed_f32_operands(runtime.primary_ctx, SALT)?;
    buffers.output.upload_f32(
        &runtime.secondary_ctx.stream,
        &tf32_tournament_seeded_values(m * n, SALT ^ 0x91),
    )?;
    buffers.a.upload_f32(
        &runtime.secondary_ctx.stream,
        &tf32_tournament_seeded_values(m * k, SALT ^ 0x2d),
    )?;
    buffers.b.upload_f32(
        &runtime.secondary_ctx.stream,
        &tf32_tournament_seeded_values(k * n, SALT ^ 0x67),
    )?;
    runtime
        .secondary_ctx
        .stream
        .synchronize()
        .map_err(|error| format!("synchronize TF32 tournament dense seed: {error:?}"))
}

fn collect_tf32_tournament_route_pair(
    runtime: &Tf32TournamentRuntime<'_>,
    pair: Tf32TournamentPhysicalPair<'_, '_>,
    iterations: Tf32TournamentIterations,
    cohort: Tf32TournamentCohort,
) -> Result<Tf32TournamentSamples, String> {
    let mut candidate_us = Vec::with_capacity(cohort.windows);
    let mut denominator_us = Vec::with_capacity(cohort.windows);
    let mut ratios = Vec::with_capacity(cohort.windows);
    for index in 0..cohort.windows {
        let mut candidate_sample = || {
            let sample = measure_tf32_tournament_physical_window_ms(
                runtime.primary_ctx,
                pair.candidate,
                cohort.path,
                iterations.candidate,
            )? * 1_000.0
                / iterations.candidate as f64;
            validate_sample_us(sample, index)?;
            Ok::<f64, String>(sample)
        };
        let mut denominator_sample = || {
            let sample = measure_tf32_tournament_physical_window_ms(
                runtime.secondary_ctx,
                pair.denominator,
                cohort.path,
                iterations.denominator,
            )? * 1_000.0
                / iterations.denominator as f64;
            validate_sample_us(sample, index)?;
            Ok::<f64, String>(sample)
        };
        let (candidate, denominator) = match cohort.order {
            PathOrder::Ab => (candidate_sample()?, denominator_sample()?),
            PathOrder::Ba => {
                let denominator = denominator_sample()?;
                let candidate = candidate_sample()?;
                (candidate, denominator)
            }
        };
        let ratio = candidate / denominator;
        validate_sample_us(ratio, index)?;
        candidate_us.push(candidate);
        denominator_us.push(denominator);
        ratios.push(ratio);
    }
    Ok(Tf32TournamentSamples {
        candidate_us,
        denominator_us,
        ratios,
    })
}

fn collect_tf32_tournament_cublas(
    runtime: &Tf32TournamentRuntime<'_>,
    pair: Tf32TournamentCublasPair<'_, '_>,
    iterations: Tf32TournamentIterations,
    cohort: Tf32TournamentCohort,
) -> Result<Tf32TournamentSamples, String> {
    let mut candidate_us = Vec::with_capacity(cohort.windows);
    let mut denominator_us = Vec::with_capacity(cohort.windows);
    let mut ratios = Vec::with_capacity(cohort.windows);
    for index in 0..cohort.windows {
        let mut candidate_sample = || {
            let sample = measure_tf32_tournament_physical_window_ms(
                runtime.primary_ctx,
                pair.candidate,
                cohort.path,
                iterations.candidate,
            )? * 1_000.0
                / iterations.candidate as f64;
            validate_sample_us(sample, index)?;
            Ok::<f64, String>(sample)
        };
        let denominator_sample = || {
            let sample = measure_tf32_tournament_cublas_window_ms(
                runtime,
                pair.comparator,
                cohort.path,
                iterations.denominator,
            )? * 1_000.0
                / iterations.denominator as f64;
            validate_sample_us(sample, index)?;
            Ok::<f64, String>(sample)
        };
        let (candidate, denominator) = match cohort.order {
            PathOrder::Ab => (candidate_sample()?, denominator_sample()?),
            PathOrder::Ba => denominator_sample().and_then(|denominator| {
                candidate_sample().map(|candidate| (candidate, denominator))
            })?,
        };
        let ratio = candidate / denominator;
        validate_sample_us(ratio, index)?;
        candidate_us.push(candidate);
        denominator_us.push(denominator);
        ratios.push(ratio);
    }
    Ok(Tf32TournamentSamples {
        candidate_us,
        denominator_us,
        ratios,
    })
}

fn summarize_tf32_tournament_samples(
    samples: &Tf32TournamentSamples,
) -> Result<Tf32TournamentSummary, String> {
    if samples.candidate_us.len() != DEFAULT_WINDOWS
        || samples.denominator_us.len() != DEFAULT_WINDOWS
        || samples.ratios.len() != DEFAULT_WINDOWS
    {
        return Err(format!(
            "TF32 tournament cohort requires {DEFAULT_WINDOWS} paired samples"
        ));
    }
    let percentiles = |values: &[f64]| {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        (percentile(&sorted, 0.50), percentile(&sorted, 0.95))
    };
    let (candidate_p50, candidate_p95) = percentiles(&samples.candidate_us);
    let (denominator_p50, denominator_p95) = percentiles(&samples.denominator_us);
    let (ratio_p50, ratio_p95) = percentiles(&samples.ratios);
    Ok(Tf32TournamentSummary {
        candidate_p50,
        candidate_p95,
        denominator_p50,
        denominator_p95,
        ratio_p50,
        ratio_p95,
    })
}

fn render_tf32_tournament_samples(values: &[f64]) -> String {
    let mut raw = String::new();
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            raw.push(',');
        }
        write!(raw, "{value:.9}").expect("String writes cannot fail");
    }
    raw
}

fn emit_tf32_tournament_evidence(
    shape: Shape,
    comparison: Tf32TournamentComparison,
    cohort: Tf32TournamentCohort,
    samples: &Tf32TournamentSamples,
    summary: &Tf32TournamentSummary,
    iterations: Tf32TournamentIterations,
    meta: Tf32TournamentRecordMeta<'_>,
) -> String {
    let (m, k, n) = shape.dims;
    let cc = meta
        .runtime
        .primary_ctx
        .stream
        .context()
        .compute_capability()
        .expect("TF32 tournament CUDA CC");
    let candidate_route = comparison.candidate();
    let candidate_evidence = meta.candidate.evidence();
    let denominator_evidence = meta.denominator.map(QualifiedPhysicalLaunch::evidence);
    let denominator_route_identity = denominator_evidence.map_or_else(
        || "cublasGemmEx/CUBLAS_COMPUTE_32F_FAST_TF32/CUBLAS_GEMM_DEFAULT".to_string(),
        |evidence| format!("{:?}", evidence.route_identity()),
    );
    let denominator_launch_digest = denominator_evidence.map_or_else(
        || "null".to_string(),
        |evidence| format!("\"{}\"", hex_digest(evidence.launch_digest())),
    );
    let denominator_request_digest = denominator_evidence.map_or_else(
        || "null".to_string(),
        |evidence| format!("\"{}\"", hex_digest(evidence.request_identity_digest())),
    );
    let denominator_grid = comparison.denominator_route().map_or_else(
        || "null".to_string(),
        |route| {
            let grid = route.expected_grid();
            format!("[{},{},{}]", grid.0, grid.1, grid.2)
        },
    );
    let candidate_grid = candidate_route.expected_grid();
    format!(
        concat!(
            "{{\"schema\":\"MambaBiTf32NnTournamentPerformanceV1\",",
            "\"suite\":\"gemm_bi_tf32_nn_route_tournament\",",
            "\"scope\":\"performance_only\",\"cohort_valid\":true,",
            "\"shape\":\"{}\",\"m\":{},\"k\":{},\"n\":{},",
            "\"comparison_kind\":\"{}\",\"comparison\":\"{}\",",
            "\"path\":\"{}\",\"order\":\"{}\",\"release_build\":true,",
            "\"cc\":\"{}.{}\",\"multiprocessors\":{},\"gpu_uuid\":\"{}\",",
            "\"context_assignment\":\"{}\",\"assignment_preflight\":\"{}\",",
            "\"calibration_preflight\":\"{}\",\"cohort_preflight\":\"{}\",",
            "\"cohort_postflight\":\"{}\",",
            "\"candidate_route\":\"{}\",\"candidate_symbol\":\"{}\",",
            "\"candidate_grid\":[{},{},{}],\"candidate_block\":[128,1,1],",
            "\"candidate_shared_mem_bytes\":29696,",
            "\"candidate_route_identity\":\"{}\",",
            "\"candidate_launch_digest\":\"{}\",\"candidate_request_digest\":\"{}\",",
            "\"denominator_route\":\"{}\",\"denominator_symbol\":\"{}\",",
            "\"denominator_grid\":{},\"denominator_route_identity\":\"{}\",",
            "\"denominator_launch_digest\":{},\"denominator_request_digest\":{},",
            "\"semantic_probe\":\"one_active_term_exact_and_dense_all_k_tolerant\",",
            "\"one_term_cross_arm_bit_exact\":true,",
            "\"dense_partition_edges_active\":true,",
            "\"dense_tolerance\":\"abs<=0.25+0.02*max_abs\",",
            "\"dense_cross_arm_bit_exact\":{},",
            "\"candidate_one_term_output_digest\":\"{}\",",
            "\"denominator_one_term_output_digest\":\"{}\",",
            "\"candidate_dense_output_digest\":\"{}\",",
            "\"denominator_dense_output_digest\":\"{}\",",
            "\"candidate_iterations\":{},\"denominator_iterations\":{},",
            "\"windows\":{},\"candidate_p50_us\":{:.9},",
            "\"candidate_p95_us\":{:.9},\"denominator_p50_us\":{:.9},",
            "\"denominator_p95_us\":{:.9},\"ratio_p50\":{:.9},",
            "\"ratio_p95\":{:.9},\"candidate_beats_p50\":{},",
            "\"candidate_beats_p95\":{},\"candidate_samples_us\":[{}],",
            "\"denominator_samples_us\":[{}],\"ratios\":[{}]}}"
        ),
        shape.name,
        m,
        k,
        n,
        comparison.kind(),
        comparison.name(),
        cohort.path.as_str(),
        cohort.order.as_str(),
        cc.0,
        cc.1,
        meta.runtime.multiprocessors,
        meta.runtime.quiet_gpu.uuid(),
        meta.runtime.assignment,
        escape_json_string(meta.runtime.assignment_preflight_snapshot),
        escape_json_string(meta.calibration_preflight),
        escape_json_string(meta.preflight),
        escape_json_string(meta.postflight),
        candidate_route.name(),
        candidate_route.symbol(),
        candidate_grid.0,
        candidate_grid.1,
        candidate_grid.2,
        escape_json_string(&format!("{:?}", candidate_evidence.route_identity())),
        hex_digest(candidate_evidence.launch_digest()),
        hex_digest(candidate_evidence.request_identity_digest()),
        comparison.denominator_name(),
        comparison.denominator_symbol(),
        denominator_grid,
        escape_json_string(&denominator_route_identity),
        denominator_launch_digest,
        denominator_request_digest,
        meta.semantic.dense_cross_arm_bit_exact,
        hex_digest(meta.semantic.candidate_one_term_digest),
        hex_digest(meta.semantic.denominator_one_term_digest),
        hex_digest(meta.semantic.candidate_dense_digest),
        hex_digest(meta.semantic.denominator_dense_digest),
        iterations.candidate,
        iterations.denominator,
        samples.ratios.len(),
        summary.candidate_p50,
        summary.candidate_p95,
        summary.denominator_p50,
        summary.denominator_p95,
        summary.ratio_p50,
        summary.ratio_p95,
        summary.ratio_p50 < 1.0,
        summary.ratio_p95 < 1.0,
        render_tf32_tournament_samples(&samples.candidate_us),
        render_tf32_tournament_samples(&samples.denominator_us),
        render_tf32_tournament_samples(&samples.ratios),
    )
}

fn run_tf32_tournament_route_pair_path(
    runtime: &Tf32TournamentRuntime<'_>,
    shape: Shape,
    comparison: Tf32TournamentComparison,
    pair: Tf32TournamentPhysicalPair<'_, '_>,
    semantic: &Tf32TournamentSemanticEvidence,
    path: Tf32TournamentPath,
    sink: &mut Tf32TournamentJsonlSink,
) -> Result<(), String> {
    let calibration_label = format!(
        "tf32/tournament/{}/{}/{}/calibration",
        shape.name,
        comparison.name(),
        path.as_str()
    );
    let calibration_preflight = runtime.quiet_gpu.require_cohort(&calibration_label)?;
    measure_tf32_tournament_physical_window_ms(runtime.primary_ctx, pair.candidate, path, 128)?;
    measure_tf32_tournament_physical_window_ms(runtime.secondary_ctx, pair.denominator, path, 128)?;
    let iterations = Tf32TournamentIterations {
        candidate: calibrate_measurement(|count| {
            measure_tf32_tournament_physical_window_ms(
                runtime.primary_ctx,
                pair.candidate,
                path,
                count,
            )
        })?,
        denominator: calibrate_measurement(|count| {
            measure_tf32_tournament_physical_window_ms(
                runtime.secondary_ctx,
                pair.denominator,
                path,
                count,
            )
        })?,
    };
    for order in TF32_TOURNAMENT_ORDERS {
        let cohort = Tf32TournamentCohort {
            path,
            order,
            windows: DEFAULT_WINDOWS,
        };
        let label = format!(
            "tf32/tournament/{}/{}/{}/{}",
            shape.name,
            comparison.name(),
            path.as_str(),
            order.as_str()
        );
        let preflight = runtime.quiet_gpu.require_cohort(&label)?;
        let samples = collect_tf32_tournament_route_pair(
            runtime,
            Tf32TournamentPhysicalPair {
                candidate: pair.candidate,
                denominator: pair.denominator,
            },
            iterations,
            cohort,
        )?;
        let postflight = runtime.quiet_gpu.verify_post_cohort(&label)?;
        let summary = summarize_tf32_tournament_samples(&samples)?;
        sink.write(emit_tf32_tournament_evidence(
            shape,
            comparison,
            cohort,
            &samples,
            &summary,
            iterations,
            Tf32TournamentRecordMeta {
                runtime,
                semantic,
                candidate: pair.candidate,
                denominator: Some(pair.denominator),
                calibration_preflight: &calibration_preflight,
                preflight: &preflight,
                postflight: &postflight,
            },
        ))?;
    }
    Ok(())
}

fn run_tf32_tournament_cublas_path(
    runtime: &Tf32TournamentRuntime<'_>,
    shape: Shape,
    comparison: Tf32TournamentComparison,
    pair: Tf32TournamentCublasPair<'_, '_>,
    semantic: &Tf32TournamentSemanticEvidence,
    path: Tf32TournamentPath,
    sink: &mut Tf32TournamentJsonlSink,
) -> Result<(), String> {
    let calibration_label = format!(
        "tf32/tournament/{}/{}/{}/calibration",
        shape.name,
        comparison.name(),
        path.as_str()
    );
    let calibration_preflight = runtime.quiet_gpu.require_cohort(&calibration_label)?;
    measure_tf32_tournament_physical_window_ms(runtime.primary_ctx, pair.candidate, path, 128)?;
    measure_tf32_tournament_cublas_window_ms(runtime, pair.comparator, path, 128)?;
    let iterations = Tf32TournamentIterations {
        candidate: calibrate_measurement(|count| {
            measure_tf32_tournament_physical_window_ms(
                runtime.primary_ctx,
                pair.candidate,
                path,
                count,
            )
        })?,
        denominator: calibrate_measurement(|count| {
            measure_tf32_tournament_cublas_window_ms(runtime, pair.comparator, path, count)
        })?,
    };
    for order in TF32_TOURNAMENT_ORDERS {
        let cohort = Tf32TournamentCohort {
            path,
            order,
            windows: DEFAULT_WINDOWS,
        };
        let label = format!(
            "tf32/tournament/{}/{}/{}/{}",
            shape.name,
            comparison.name(),
            path.as_str(),
            order.as_str()
        );
        let preflight = runtime.quiet_gpu.require_cohort(&label)?;
        let samples = collect_tf32_tournament_cublas(
            runtime,
            Tf32TournamentCublasPair {
                candidate: pair.candidate,
                comparator: pair.comparator,
            },
            iterations,
            cohort,
        )?;
        let postflight = runtime.quiet_gpu.verify_post_cohort(&label)?;
        let summary = summarize_tf32_tournament_samples(&samples)?;
        sink.write(emit_tf32_tournament_evidence(
            shape,
            comparison,
            cohort,
            &samples,
            &summary,
            iterations,
            Tf32TournamentRecordMeta {
                runtime,
                semantic,
                candidate: pair.candidate,
                denominator: None,
                calibration_preflight: &calibration_preflight,
                preflight: &preflight,
                postflight: &postflight,
            },
        ))?;
    }
    Ok(())
}

fn run_tf32_tournament_route_pair(
    runtime: &Tf32TournamentRuntime<'_>,
    shape: Shape,
    comparison: Tf32TournamentComparison,
    sink: &mut Tf32TournamentJsonlSink,
) -> Result<(), String> {
    let candidate_route = comparison.candidate();
    let denominator_route = comparison
        .denominator_route()
        .ok_or("route-pair comparison requires a deterministic denominator")?;
    let mut candidate = qualify_physical_launch(
        runtime.primary_ctx,
        tf32_tournament_request(shape, candidate_route),
    )?;
    let mut denominator = qualify_physical_launch(
        runtime.secondary_ctx,
        tf32_tournament_request(shape, denominator_route),
    )?;
    validate_tf32_tournament_manifest(candidate_route, &candidate)?;
    validate_tf32_tournament_manifest(denominator_route, &denominator)?;
    let eager_semantic = validate_tf32_tournament_route_pair_semantics(
        runtime,
        shape,
        comparison,
        Tf32TournamentPhysicalPair {
            candidate: &mut candidate,
            denominator: &mut denominator,
        },
        Tf32TournamentPath::Eager,
    )?;
    let graph_semantic = validate_tf32_tournament_route_pair_semantics(
        runtime,
        shape,
        comparison,
        Tf32TournamentPhysicalPair {
            candidate: &mut candidate,
            denominator: &mut denominator,
        },
        Tf32TournamentPath::Graph,
    )?;
    if eager_semantic != graph_semantic {
        return Err(format!(
            "{} eager and graph semantic digests differ",
            comparison.name()
        ));
    }
    seed_tf32_tournament_route_pair_dense_inputs(
        runtime,
        shape,
        &mut Tf32TournamentPhysicalPair {
            candidate: &mut candidate,
            denominator: &mut denominator,
        },
    )?;
    for path in TF32_TOURNAMENT_PATHS {
        run_tf32_tournament_route_pair_path(
            runtime,
            shape,
            comparison,
            Tf32TournamentPhysicalPair {
                candidate: &mut candidate,
                denominator: &mut denominator,
            },
            &eager_semantic,
            path,
            sink,
        )?;
    }
    Ok(())
}

fn run_tf32_tournament_cublas_comparison(
    runtime: &Tf32TournamentRuntime<'_>,
    shape: Shape,
    comparison: Tf32TournamentComparison,
    comparator: Tf32TournamentCublasComparator<'_>,
    sink: &mut Tf32TournamentJsonlSink,
) -> Result<(), String> {
    let route = comparison.candidate();
    let mut candidate =
        qualify_physical_launch(runtime.primary_ctx, tf32_tournament_request(shape, route))?;
    validate_tf32_tournament_manifest(route, &candidate)?;
    let eager_semantic = validate_tf32_tournament_cublas_semantics(
        runtime,
        shape,
        &mut candidate,
        comparator,
        Tf32TournamentPath::Eager,
    )?;
    let graph_semantic = validate_tf32_tournament_cublas_semantics(
        runtime,
        shape,
        &mut candidate,
        comparator,
        Tf32TournamentPath::Graph,
    )?;
    if eager_semantic != graph_semantic {
        return Err(format!(
            "{} eager and graph semantic digests differ",
            comparison.name()
        ));
    }
    seed_tf32_tournament_cublas_dense_inputs(runtime, shape, &mut candidate, comparator.buffers)?;
    for path in TF32_TOURNAMENT_PATHS {
        run_tf32_tournament_cublas_path(
            runtime,
            shape,
            comparison,
            Tf32TournamentCublasPair {
                candidate: &mut candidate,
                comparator,
            },
            &eager_semantic,
            path,
            sink,
        )?;
    }
    Ok(())
}

fn run_tf32_tournament_shape(
    runtime: &Tf32TournamentRuntime<'_>,
    shape: Shape,
    sink: &mut Tf32TournamentJsonlSink,
) -> Result<(), String> {
    for comparison in TF32_TOURNAMENT_COMPARISONS {
        if matches!(comparison, Tf32TournamentComparison::RoutePair { .. }) {
            run_tf32_tournament_route_pair(runtime, shape, comparison, sink)?;
        }
    }
    let cell = CublasDenominatorCell {
        dtype: WeightDtype::F32,
        op: ResolvedGemmOp::Nn,
        shape,
    };
    let buffers = allocate_cublas_denominator_buffers(runtime.secondary_ctx, cell)?;
    launch_cublas_denominator(
        runtime.secondary_ctx,
        cell,
        CublasDenominatorMode::Fast,
        &buffers,
    )?;
    runtime
        .secondary_ctx
        .stream
        .synchronize()
        .map_err(|error| format!("warm TF32 tournament cuBLAS handle: {error:?}"))?;
    let graph = capture_tf32_tournament_cublas_graph(runtime, cell, &buffers)?;
    let comparator = Tf32TournamentCublasComparator {
        cell,
        buffers: &buffers,
        graph: &graph,
    };
    for comparison in TF32_TOURNAMENT_COMPARISONS {
        if matches!(comparison, Tf32TournamentComparison::CublasFastTf32 { .. }) {
            run_tf32_tournament_cublas_comparison(runtime, shape, comparison, comparator, sink)?;
        }
    }
    runtime
        .secondary_ctx
        .stream
        .synchronize()
        .map_err(|error| format!("synchronize TF32 tournament graph before drop: {error:?}"))
}

fn run_tf32_tournament_assignment(
    device: &GpuDevice,
    quiet_gpu: &QuietGpu,
    assignment_preflight_snapshot: &str,
    primary_first: bool,
    sink: &mut Tf32TournamentJsonlSink,
) -> Result<(), String> {
    let first_ctx = GpuCtx::new(device)?;
    let second_ctx = GpuCtx::new(device)?;
    let (primary_ctx, secondary_ctx, assignment) = if primary_first {
        (&first_ctx, &second_ctx, "primary_context_first")
    } else {
        (&second_ctx, &first_ctx, "primary_context_second")
    };
    for ctx in [primary_ctx, secondary_ctx] {
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    }
    let requests = TF32_TOURNAMENT_SHAPES
        .into_iter()
        .flat_map(|shape| TF32_TOURNAMENT_ROUTES.map(|route| tf32_tournament_request(shape, route)))
        .collect::<Vec<_>>();
    presize_physical_qualification_suite(primary_ctx, &requests)?;
    presize_physical_qualification_suite(secondary_ctx, &requests)?;
    let runtime = Tf32TournamentRuntime {
        primary_ctx,
        secondary_ctx,
        quiet_gpu,
        assignment,
        assignment_preflight_snapshot,
        multiprocessors: device.multiprocessor_count(),
    };
    for shape in TF32_TOURNAMENT_SHAPES {
        run_tf32_tournament_shape(&runtime, shape, sink)
            .map_err(|error| format!("{} {assignment}: {error}", shape.name))?;
    }
    Ok(())
}

#[test]
#[ignore = "requires an otherwise idle qualified SM89 or SM120 GPU and emits the deterministic TF32 NN route tournament"]
fn gemm_bi_tf32_nn_route_tournament() {
    let _suite_guard = performance_suite_lock()
        .lock()
        .expect("lock serialized performance suite");
    validate_run_profile(DEFAULT_WINDOWS, cfg!(debug_assertions))
        .expect("TF32 tournament requires --release");
    let mut sink =
        Tf32TournamentJsonlSink::create_from_env().expect("new TF32 tournament evidence file");
    let quiet_gpu = QuietGpu::for_cuda_ordinal(0).expect("resolve CUDA device 0 UUID");
    let pre_context_snapshot = quiet_gpu
        .require_pre_context("tf32/tournament/pre-context")
        .expect("exclusive CUDA device 0 before context creation");
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(
        matches!(device.compute_capability, (8, 9) | (12, 0)),
        "TF32 tournament evidence is qualified only on exact SM89 and SM120"
    );
    run_tf32_tournament_assignment(&device, &quiet_gpu, &pre_context_snapshot, true, &mut sink)
        .expect("primary-first TF32 tournament assignment");
    let swapped_assignment_preflight = quiet_gpu
        .require_cohort("tf32/tournament/swapped-assignment")
        .expect("quiet CUDA device before swapped TF32 tournament assignment");
    run_tf32_tournament_assignment(
        &device,
        &quiet_gpu,
        &swapped_assignment_preflight,
        false,
        &mut sink,
    )
    .expect("primary-second TF32 tournament assignment");
    sink.finish(tf32_tournament_expected_records(), &quiet_gpu, &device)
        .unwrap();
}

#[test]
#[ignore = "requires an SM80+ GPU and emits performance-only cuBLAS denominators as JSONL"]
fn gemm_bi_cublas_performance_denominators() {
    let _suite_guard = performance_suite_lock()
        .lock()
        .expect("lock serialized performance suite");
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(
        device.compute_capability >= (8, 0),
        "SM80 or newer required"
    );
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(false);

    let shard = parse_env_usize("GEMM_BI_QUAL_SHARD", 0).expect("parse shard");
    let shards = parse_env_usize("GEMM_BI_QUAL_SHARDS", 1).expect("parse shard count");
    assert!(shards > 0, "shard count must be positive");
    assert!(shard < shards, "shard index must be below shard count");
    let windows =
        parse_env_usize("GEMM_BI_QUAL_WINDOWS", DEFAULT_WINDOWS).expect("parse window count");
    assert!(windows > 0, "window count must be positive");
    validate_run_profile(windows, cfg!(debug_assertions)).expect("validate Rust build profile");
    let variant = std::env::var("GEMM_BI_QUAL_VARIANT").unwrap_or_else(|_| "cublas".into());
    validate_variant(&variant).expect("validate variant");
    let order = std::env::var("GEMM_BI_QUAL_PATH_ORDER")
        .map_or_else(|_| Ok(PathOrder::Ab), |value| parse_path_order(&value))
        .expect("parse arm order");

    let cells = build_cublas_denominator_cells();
    assert_eq!(cells.len(), SHAPES.len() * 3 * 3);
    let requested =
        load_cublas_denominator_cell_ids(&cells).expect("parse exact cuBLAS denominator cell IDs");
    let selected = select_cublas_denominator_cells(&cells, requested.as_ref(), shard, shards)
        .expect("select cuBLAS denominator cells");
    for cell in selected {
        let id = cublas_denominator_cell_id(cell);
        run_cublas_denominator_cell(&ctx, cell, windows, &variant, order)
            .unwrap_or_else(|error| panic!("{id}: {error}"));
    }
}

#[test]
#[ignore = "requires an idle SM80+ GPU and emits the forced TN narrow two-launch paired fast-cuBLAS comparator"]
fn gemm_bi_tn_narrow_two_launch_paired_fast_cublas() {
    let _suite_guard = performance_suite_lock()
        .lock()
        .expect("lock serialized performance suite");
    tn_narrow_cublas_pair::run_suite().expect("run forced TN narrow paired comparator");
}

#[test]
#[ignore = "requires an SM80+ GPU and emits the deterministic Triad performance matrix as JSONL"]
fn gemm_bi_deterministic_performance_matrix() {
    let _suite_guard = performance_suite_lock()
        .lock()
        .expect("lock serialized performance suite");
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(
        device.compute_capability >= (8, 0),
        "SM80 or newer required"
    );
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_fast_gemm(false);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);

    let shard = parse_env_usize("GEMM_BI_QUAL_SHARD", 0).expect("parse shard");
    let shards = parse_env_usize("GEMM_BI_QUAL_SHARDS", 1).expect("parse shard count");
    let windows =
        parse_env_usize("GEMM_BI_QUAL_WINDOWS", DEFAULT_WINDOWS).expect("parse window count");
    assert!(windows > 0, "window count must be positive");
    validate_run_profile(windows, cfg!(debug_assertions)).expect("validate Rust build profile");
    let variant = std::env::var("GEMM_BI_QUAL_VARIANT").unwrap_or_else(|_| "baseline".into());
    validate_variant(&variant).expect("validate variant");
    let path_order = std::env::var("GEMM_BI_QUAL_PATH_ORDER")
        .map_or_else(|_| Ok(PathOrder::Ab), |value| parse_path_order(&value))
        .expect("parse path order");
    let cells = build_cells();
    assert_eq!(
        cells.len(),
        expected_cell_count(),
        "derived logical performance cell count"
    );
    let frozen_iterations =
        load_frozen_iterations(&cells).expect("load frozen canonical iterations");
    let requested_cells =
        load_qualification_cell_ids(&cells).expect("parse exact qualification cell IDs");
    let cells = select_qualification_cells(&cells, requested_cells.as_ref(), shard, shards)
        .expect("select qualification cells");
    let requests = cells
        .iter()
        .copied()
        .map(qualification_request)
        .collect::<Vec<_>>();
    presize_physical_qualification_suite(&ctx, &requests)
        .expect("pre-size qualification scratch before graph capture");
    let mut selected = 0;
    let mut used_iterations = BTreeMap::new();
    for cell in cells {
        let id = cell_id(cell);
        let frozen = frozen_iterations.as_ref().map(|manifest| {
            *manifest
                .get(&id)
                .unwrap_or_else(|| panic!("frozen iteration manifest has no cell {id}"))
        });
        let iterations = run_cell(
            &ctx,
            cell,
            RunnerConfig {
                windows,
                variant: &variant,
                path_order,
                frozen,
            },
        )
        .unwrap_or_else(|error| panic!("{id}: {error}"));
        assert!(
            used_iterations.insert(id, iterations).is_none(),
            "duplicate selected performance cell"
        );
        selected += 1;
    }
    assert!(selected > 0, "shard {shard}/{shards} selected no cells");
    if let Ok(path) = std::env::var("GEMM_BI_QUAL_ITERS_OUT") {
        let output = render_frozen_iterations(&used_iterations, &build_cells())
            .expect("render complete canonical frozen iteration manifest");
        std::fs::write(&path, output)
            .unwrap_or_else(|error| panic!("write frozen iterations {path}: {error}"));
    }
}

#[test]
#[ignore = "requires an SM120 GPU and emits forced deterministic TF32 performance JSONL"]
fn gemm_bi_sm120_tf32_forced_hot_performance() {
    let _suite_guard = performance_suite_lock()
        .lock()
        .expect("lock serialized performance suite");
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(
        matches!(device.compute_capability, (12, 0) | (12, 1)),
        "SM120 or SM121 required"
    );
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_fast_gemm(false);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);

    let windows =
        parse_env_usize("GEMM_BI_QUAL_WINDOWS", DEFAULT_WINDOWS).expect("parse window count");
    assert!(windows > 0, "window count must be positive");
    validate_run_profile(windows, cfg!(debug_assertions)).expect("validate Rust build profile");
    let variant = std::env::var("GEMM_BI_QUAL_VARIANT").unwrap_or_else(|_| "sm120".into());
    validate_variant(&variant).expect("validate variant");
    let path_order = std::env::var("GEMM_BI_QUAL_PATH_ORDER")
        .map_or_else(|_| Ok(PathOrder::Ab), |value| parse_path_order(&value))
        .expect("parse path order");
    let shapes = SHAPES
        .iter()
        .copied()
        .filter(|shape| SM120_TF32_HOT_SHAPES.contains(&shape.name));
    let cells = tf32_route_specs(ModuleKind::TriadSm120)
        .iter()
        .filter(|spec| spec.op == ResolvedGemmOp::Tn && !spec.route.is_exact_fma())
        .flat_map(|spec| {
            shapes.clone().map(|shape| Cell {
                route: Route::Tf32Forced(spec),
                op: ResolvedGemmOp::Tn,
                shape,
                stride_class: StrideClass::Contiguous,
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(cells.len(), 30, "SM120 TN TF32 hot cell count");
    let requests = cells
        .iter()
        .copied()
        .map(qualification_request)
        .collect::<Vec<_>>();
    presize_physical_qualification_suite(&ctx, &requests)
        .expect("pre-size SM120 TF32 performance scratch");
    for cell in cells {
        let id = cell_id(cell);
        run_prequalified_cell(
            &ctx,
            cell,
            RunnerConfig {
                windows,
                variant: &variant,
                path_order,
                frozen: None,
            },
        )
        .unwrap_or_else(|error| panic!("{id}: {error}"));
    }
}

#[test]
#[ignore = "requires an SM120 GPU and emits forced deterministic TF32 NN/NT performance JSONL"]
fn gemm_bi_sm120_tf32_forced_nn_nt_hot_performance() {
    let _suite_guard = performance_suite_lock()
        .lock()
        .expect("lock serialized performance suite");
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(
        matches!(device.compute_capability, (12, 0) | (12, 1)),
        "SM120 or SM121 required"
    );
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_fast_gemm(false);
    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);

    let windows =
        parse_env_usize("GEMM_BI_QUAL_WINDOWS", DEFAULT_WINDOWS).expect("parse window count");
    assert!(windows > 0, "window count must be positive");
    validate_run_profile(windows, cfg!(debug_assertions)).expect("validate Rust build profile");
    let variant = std::env::var("GEMM_BI_QUAL_VARIANT").unwrap_or_else(|_| "sm120".into());
    validate_variant(&variant).expect("validate variant");
    let path_order = std::env::var("GEMM_BI_QUAL_PATH_ORDER")
        .map_or_else(|_| Ok(PathOrder::Ab), |value| parse_path_order(&value))
        .expect("parse path order");
    let shapes = SHAPES
        .iter()
        .copied()
        .filter(|shape| SM120_TF32_HOT_SHAPES.contains(&shape.name))
        .collect::<Vec<_>>();
    let mut cells = Vec::with_capacity(55);
    for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Nt] {
        for spec in tf32_route_specs(ModuleKind::TriadSm120)
            .iter()
            .filter(|spec| spec.op == op && !spec.route.is_exact_fma())
        {
            for shape in shapes.iter().copied() {
                cells.push(Cell {
                    route: Route::Tf32Forced(spec),
                    op,
                    shape,
                    stride_class: StrideClass::Contiguous,
                });
            }
        }
    }
    assert_eq!(cells.len(), 55, "SM120 NN/NT TF32 hot cell count");
    let requests = cells
        .iter()
        .copied()
        .map(qualification_request)
        .collect::<Vec<_>>();
    presize_physical_qualification_suite(&ctx, &requests)
        .expect("pre-size SM120 TF32 performance scratch");
    for cell in cells {
        let id = cell_id(cell);
        run_prequalified_cell(
            &ctx,
            cell,
            RunnerConfig {
                windows,
                variant: &variant,
                path_order,
                frozen: None,
            },
        )
        .unwrap_or_else(|error| panic!("{id}: {error}"));
    }
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
#[ignore = "requires an SM120 GPU and emits paired half NN Triad/cuBLASLt performance JSONL"]
fn gemm_bi_sm120_half_nn_paired_cublaslt_hot() {
    use cublaslt_qualification::*;

    let _suite_guard = performance_suite_lock()
        .lock()
        .expect("lock serialized performance suite");
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(
        matches!(device.compute_capability, (12, 0) | (12, 1)),
        "SM120 or SM121 required"
    );
    assert!(
        !cfg!(debug_assertions),
        "paired performance benchmark requires --release"
    );
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_fast_gemm(false);
    let runtime = Sm120PairedLtRuntime::new(&ctx).expect("prepare paired cuBLASLt runtime");

    for cell in build_sm120_paired_nn_cells() {
        let id = sm120_paired_nn_cell_id(cell);
        run_sm120_paired_nn_cell(&ctx, &runtime, cell)
            .unwrap_or_else(|error| panic!("{id}: {error}"));
    }
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
#[ignore = "requires a physical CC 12.0 GPU and emits paired TN/NT vendor performance JSONL"]
fn gemm_bi_sm120_half_tn_nt_paired_vendor_hot() {
    use cublaslt_tn_nt_qualification::*;

    let _suite_guard = performance_suite_lock()
        .lock()
        .expect("lock serialized performance suite");
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert_eq!(
        device.compute_capability,
        (12, 0),
        "physical CC 12.0 is required"
    );
    assert!(
        !cfg!(debug_assertions),
        "paired TN/NT performance benchmark requires --release"
    );
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_fast_gemm(false);
    let runtime = Sm120PairedTnNtRuntime::new(&ctx).expect("prepare paired TN/NT vendor runtime");

    for cell in build_sm120_paired_tn_nt_cells() {
        let id = sm120_paired_tn_nt_cell_id(cell);
        run_sm120_paired_tn_nt_cell(&ctx, &runtime, cell)
            .unwrap_or_else(|error| panic!("{id}: {error}"));
    }
}

#[test]
#[ignore = "requires an SM80+ GPU and emits deterministic Triad edge qualification JSONL"]
fn gemm_bi_deterministic_performance_edges() {
    let _suite_guard = performance_suite_lock()
        .lock()
        .expect("lock serialized performance suite");
    let device = GpuDevice::new(0).expect("open CUDA device 0");
    assert!(
        device.compute_capability >= (8, 0),
        "SM80 or newer required"
    );
    let ctx = GpuCtx::new(&device).expect("create GPU context");
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_fast_gemm(false);
    ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);

    let shard = parse_env_usize("GEMM_BI_QUAL_SHARD", 0).expect("parse edge shard");
    let shards = parse_env_usize("GEMM_BI_QUAL_SHARDS", 1).expect("parse edge shard count");
    let windows =
        parse_env_usize("GEMM_BI_QUAL_WINDOWS", DEFAULT_WINDOWS).expect("parse window count");
    assert!(windows > 0, "window count must be positive");
    validate_run_profile(windows, cfg!(debug_assertions)).expect("validate Rust build profile");
    let variant = std::env::var("GEMM_BI_QUAL_VARIANT").unwrap_or_else(|_| "edge-baseline".into());
    validate_variant(&variant).expect("validate variant");
    let path_order = std::env::var("GEMM_BI_QUAL_PATH_ORDER")
        .map_or_else(|_| Ok(PathOrder::Ab), |value| parse_path_order(&value))
        .expect("parse path order");

    let inventory = build_edge_cells();
    validate_edge_inventory(&inventory).expect("validate edge inventory");
    let frozen_iterations =
        load_edge_frozen_iterations(&inventory).expect("load edge frozen iterations");
    let requested_cells =
        load_edge_qualification_cell_ids(&inventory).expect("parse exact edge cell IDs");
    let cells =
        select_edge_qualification_cells(&inventory, requested_cells.as_ref(), shard, shards)
            .expect("select edge qualification cells");
    let requests = cells
        .iter()
        .copied()
        .map(edge_qualification_request)
        .collect::<Result<Vec<_>, _>>()
        .expect("map edge qualification requests");
    presize_physical_qualification_suite(&ctx, &requests)
        .expect("pre-size edge qualification scratch before graph capture");

    let unsupported_descriptors = unsupported_padded_descriptors_for_shard(shard)
        .expect("validate unsupported padded descriptor inventory");
    for descriptor in unsupported_descriptors {
        println!("{}", render_unsupported_padded_descriptor(descriptor));
    }
    let mut used_iterations = BTreeMap::new();
    for cell in cells {
        let id = edge_cell_id(cell);
        let frozen = frozen_iterations.as_ref().map(|manifest| {
            *manifest
                .get(&id)
                .unwrap_or_else(|| panic!("edge frozen iteration manifest has no cell {id}"))
        });
        let iterations = run_edge_cell(
            &ctx,
            cell,
            RunnerConfig {
                windows,
                variant: &variant,
                path_order,
                frozen,
            },
        )
        .unwrap_or_else(|error| panic!("{id}: {error}"));
        assert!(
            used_iterations.insert(id, iterations).is_none(),
            "duplicate selected edge performance cell"
        );
    }
    if let Ok(path) = std::env::var("GEMM_BI_QUAL_ITERS_OUT") {
        let output = render_edge_frozen_iterations(&used_iterations, &inventory)
            .expect("render complete edge frozen iteration manifest");
        std::fs::write(&path, output)
            .unwrap_or_else(|error| panic!("write edge frozen iterations {path}: {error}"));
    }
}

#[test]
fn tf32_logical_route_survives_physical_symbol_rename() {
    let original = SM80_TF32_ROUTE_SPECS[0];
    let mut renamed = original;
    renamed.symbol = "gemm_bi_nn_sm80_mma_tf32_v1_m128n64_bk32_s2";

    assert_eq!(
        tf32_logical_route_name(&original),
        "f32_tf32_forced_sm80_mma_tf32_v1_m128n64_bk32_s2"
    );
    assert_eq!(
        tf32_logical_route_name(&original),
        tf32_logical_route_name(&renamed)
    );
}

#[test]
fn frozen_iteration_manifest_is_strict_and_round_trips() {
    let cells = build_cells();
    let entries = cells
        .iter()
        .copied()
        .map(|cell| {
            (
                cell_id(cell),
                FrozenIterations {
                    eager: 17,
                    graph: 31,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let rendered = render_frozen_iterations(&entries, &cells).expect("render frozen iterations");

    assert!(rendered.starts_with(concat!(
        "format\tgemm-bi-frozen-iterations.v1\n",
        "suite\tgemm_bi_canonical_performance\n",
        "inventory_count\t652\n",
        "inventory_digest\tcdfe923e3b00710e2cd52a9e942961b64d7f9d7b6b9ff24eebb202858526ac5d\n",
        "cell_id\teager_iterations\tgraph_iterations\n",
        "f32_policy_exact/nn/sq64/contiguous\t17\t31\n",
    )));
    let parsed = parse_frozen_iterations(&rendered, &cells).expect("parse frozen iterations");
    assert_eq!(parsed, entries);

    for changed in [
        rendered.replacen("suite\tgemm_bi_canonical_performance", "suite\tother", 1),
        rendered.replacen("inventory_count\t652", "inventory_count\t651", 1),
        rendered.replacen(CANONICAL_INVENTORY_DIGEST, &"0".repeat(64), 1),
    ] {
        assert!(parse_frozen_iterations(&changed, &cells).is_err());
    }

    let mut lines = rendered.lines().map(str::to_string).collect::<Vec<_>>();
    lines.swap(5, 6);
    assert!(parse_frozen_iterations(&(lines.join("\n") + "\n"), &cells).is_err());

    let mut missing = rendered.lines().map(str::to_string).collect::<Vec<_>>();
    missing.pop();
    assert!(parse_frozen_iterations(&(missing.join("\n") + "\n"), &cells).is_err());

    let mut extra = rendered.clone();
    extra.push_str("extra\t1\t1\n");
    assert!(parse_frozen_iterations(&extra, &cells).is_err());

    let mut duplicate = rendered.lines().map(str::to_string).collect::<Vec<_>>();
    duplicate[6] = duplicate[5].clone();
    assert!(parse_frozen_iterations(&(duplicate.join("\n") + "\n"), &cells).is_err());
}

#[test]
fn qualification_cell_selection_preserves_canonical_order_and_default_inventory() {
    let cells = build_cells();
    let first = "f32_policy_exact/nn/sq64/contiguous";
    let third = "bf16_policy_scalar/nn/sq64/contiguous";
    let requested = parse_qualification_cell_ids(&format!("{third},{first}"), &cells)
        .expect("parse exact qualification cells");
    let selected = select_qualification_cells(&cells, Some(&requested), 0, 1)
        .expect("select exact qualification cells");

    assert_eq!(
        selected.into_iter().map(cell_id).collect::<Vec<_>>(),
        [first, third]
    );

    let default = select_qualification_cells(&cells, None, 0, 1)
        .expect("select default qualification inventory");
    assert_eq!(default.len(), CANONICAL_INVENTORY_COUNT);
    assert_eq!(
        default.into_iter().map(cell_id).collect::<Vec<_>>(),
        cells.into_iter().map(cell_id).collect::<Vec<_>>()
    );
}

#[test]
fn qualification_cell_selection_composes_with_stable_shards() {
    let cells = build_cells();
    let exact = "f32_policy_exact/nn/sq64/contiguous";
    let allow = "f32_policy_allow_tf32/nn/sq64/contiguous";
    let bf16 = "bf16_policy_scalar/nn/sq64/contiguous";
    let requested = parse_qualification_cell_ids(&format!("{bf16},{allow},{exact}"), &cells)
        .expect("parse sharded qualification cells");

    let shard_zero =
        select_qualification_cells(&cells, Some(&requested), 0, 2).expect("select shard zero");
    assert_eq!(
        shard_zero.into_iter().map(cell_id).collect::<Vec<_>>(),
        [exact, allow]
    );
    let shard_one =
        select_qualification_cells(&cells, Some(&requested), 1, 2).expect("select shard one");
    assert_eq!(
        shard_one.into_iter().map(cell_id).collect::<Vec<_>>(),
        [bf16]
    );

    let exact_only = parse_qualification_cell_ids(exact, &cells).expect("parse one exact cell");
    assert!(select_qualification_cells(&cells, Some(&exact_only), 1, 2).is_err());
    assert!(select_qualification_cells(&cells, None, 0, 0).is_err());
    assert!(select_qualification_cells(&cells, None, 2, 2).is_err());
}

#[test]
fn qualification_cell_selection_rejects_ambiguous_or_noncanonical_inputs() {
    let cells = build_cells();
    let exact = "f32_policy_exact/nn/sq64/contiguous";
    let allow = "f32_policy_allow_tf32/nn/sq64/contiguous";
    let invalid = [
        "".to_string(),
        ",".to_string(),
        format!("{exact},"),
        format!(",{exact}"),
        format!("{exact},,{allow}"),
        format!("{exact},{exact}"),
        format!(" {exact}"),
        format!("{exact} "),
        format!("{exact},\t{allow}"),
        format!("{exact},\n{allow}"),
        "f32_policy_exact/nn/not_a_shape/contiguous".to_string(),
    ];

    for value in invalid {
        assert!(
            parse_qualification_cell_ids(&value, &cells).is_err(),
            "accepted invalid exact-cell selection {value:?}"
        );
    }
}

#[test]
fn path_order_and_variant_inputs_are_strict() {
    assert_eq!(parse_path_order("ab").expect("AB order"), PathOrder::Ab);
    assert_eq!(parse_path_order("ba").expect("BA order"), PathOrder::Ba);
    assert!(parse_path_order("AB").is_err());
    assert!(validate_variant("candidate-2.sm89").is_ok());
    assert!(validate_variant("").is_err());
    assert!(validate_variant("bad\"json").is_err());
    assert!(validate_variant("line\nbreak").is_err());
}

#[test]
fn tf32_tournament_route_contract_is_exact() {
    assert_eq!(
        TF32_TOURNAMENT_ROUTES,
        [
            Tf32TournamentRoute::Direct,
            Tf32TournamentRoute::SplitK2,
            Tf32TournamentRoute::SplitK4,
        ]
    );
    let contracts = TF32_TOURNAMENT_ROUTES.map(|route| {
        (
            route.name(),
            route.symbol(),
            route.expected_grid(),
            route.physical_route(),
        )
    });
    assert_eq!(
        contracts,
        [
            (
                "direct_m16n32_s4",
                "gemm_bi_nn_sm80_mma_tf32_v1_m16n32_bk32_s4",
                (48, 1, 1),
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
            ),
            (
                "splitk2_m16n32_s4",
                "gemm_bi_nn_sm80_mma_tf32_splitk2_v1_m16n32_bk32_s4",
                (12, 4, 2),
                Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
            ),
            (
                "splitk4_m16n32_s4",
                "gemm_bi_nn_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4",
                (12, 4, 4),
                Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
            ),
        ]
    );
}

#[test]
fn tf32_tournament_comparison_schedule_is_complete() {
    assert_eq!(
        TF32_TOURNAMENT_COMPARISONS,
        [
            Tf32TournamentComparison::RoutePair {
                candidate: Tf32TournamentRoute::Direct,
                denominator: Tf32TournamentRoute::SplitK2,
            },
            Tf32TournamentComparison::RoutePair {
                candidate: Tf32TournamentRoute::Direct,
                denominator: Tf32TournamentRoute::SplitK4,
            },
            Tf32TournamentComparison::RoutePair {
                candidate: Tf32TournamentRoute::SplitK2,
                denominator: Tf32TournamentRoute::SplitK4,
            },
            Tf32TournamentComparison::CublasFastTf32 {
                candidate: Tf32TournamentRoute::Direct,
            },
            Tf32TournamentComparison::CublasFastTf32 {
                candidate: Tf32TournamentRoute::SplitK2,
            },
            Tf32TournamentComparison::CublasFastTf32 {
                candidate: Tf32TournamentRoute::SplitK4,
            },
        ]
    );
    let kinds = TF32_TOURNAMENT_COMPARISONS.map(|comparison| comparison.kind());
    assert_eq!(
        kinds,
        [
            "deterministic_route_pair",
            "deterministic_route_pair",
            "deterministic_route_pair",
            "cublas_fast_tf32_denominator",
            "cublas_fast_tf32_denominator",
            "cublas_fast_tf32_denominator",
        ]
    );
}

#[test]
fn tf32_tournament_record_count_is_exact() {
    assert_eq!(
        TF32_TOURNAMENT_PATHS,
        [Tf32TournamentPath::Eager, Tf32TournamentPath::Graph]
    );
    assert_eq!(TF32_TOURNAMENT_SHAPES.len(), 2);
    assert_eq!(TF32_TOURNAMENT_ORDERS.len(), 2);
    assert_eq!(tf32_tournament_records_per_assignment(), 48);
    assert_eq!(tf32_tournament_expected_records(), 96);
    assert_eq!(tf32_tournament_total_jsonl_records(), 97);

    let mut records = BTreeSet::new();
    for shape in TF32_TOURNAMENT_SHAPES {
        for path in TF32_TOURNAMENT_PATHS {
            for order in TF32_TOURNAMENT_ORDERS {
                for comparison in TF32_TOURNAMENT_COMPARISONS {
                    records.insert(format!(
                        "{}/{}/{}/{}/{}",
                        shape.name,
                        path.as_str(),
                        order.as_str(),
                        comparison.kind(),
                        comparison.name()
                    ));
                }
            }
        }
    }
    assert_eq!(records.len(), 48);
}

#[test]
fn tf32_tournament_dense_semantic_rejects_missing_partitions() {
    let reference = [8.0f32.to_bits(), (-4.0f32).to_bits(), 0.5f32.to_bits()];
    let close = [7.9f32.to_bits(), (-4.1f32).to_bits(), 0.49f32.to_bits()];
    let missing_partitions = [2.0f32.to_bits(), (-1.0f32).to_bits(), 0.125f32.to_bits()];
    assert!(compare_tf32_tournament_dense_outputs(&reference, &reference).unwrap());
    assert!(!compare_tf32_tournament_dense_outputs(&reference, &close).unwrap());
    assert!(compare_tf32_tournament_dense_outputs(&reference, &missing_partitions).is_err());
    assert!(compare_tf32_tournament_dense_outputs(&reference, &reference[..2]).is_err());
    assert!(compare_tf32_tournament_dense_outputs(&reference, &[f32::NAN.to_bits(); 3]).is_err());

    assert_eq!(
        tf32_tournament_split_partition_edges(1_536, 4).unwrap(),
        [0, 383, 384, 767, 768, 1_151, 1_152, 1_535]
    );
    assert_eq!(
        tf32_tournament_split_partition_edges(833, 4).unwrap(),
        [0, 223, 224, 447, 448, 671, 672, 832]
    );
    for shape in TF32_TOURNAMENT_SHAPES {
        validate_tf32_tournament_dense_partition_activation(shape).unwrap();
    }
}

#[test]
fn sample_and_release_guards_reject_invalid_evidence() {
    assert!(validate_sample_us(1.25, 0).is_ok());
    assert!(validate_sample_us(0.0, 1).is_err());
    assert!(validate_sample_us(f64::NAN, 2).is_err());
    assert!(validate_run_profile(1, true).is_ok());
    assert!(validate_run_profile(DEFAULT_WINDOWS, false).is_ok());
    assert!(validate_run_profile(DEFAULT_WINDOWS, true).is_err());
}

#[test]
fn inventory_includes_cached_f32_and_forced_production_cells() {
    let cells = build_cells();
    let expected = expected_cell_count();
    assert_eq!(
        expected, CANONICAL_INVENTORY_COUNT,
        "frozen logical performance inventory"
    );
    assert_eq!(cells.len(), expected);
    assert_eq!(
        cells
            .iter()
            .map(|cell| cell_id(*cell))
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        expected
    );
    for shape in [
        "d128_in_proj",
        "d128_out_proj",
        "d768_in_proj",
        "d768_out_proj",
        "prism_in_proj",
    ] {
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                for tile in [TcTile::Tile64, TcTile::Tile128] {
                    assert!(cells.iter().any(|cell| {
                        cell.shape.name == shape
                            && cell.op == op
                            && matches!(
                                cell.route,
                                Route::HalfForced {
                                    dtype: found_dtype,
                                    tile: found_tile,
                                } if found_dtype == dtype && found_tile == tile
                            )
                    }));
                }
            }
        }
    }
    assert!(cells.iter().any(|cell| {
        cell.shape.name == "prism_in_proj"
            && cell.op == ResolvedGemmOp::Tn
            && matches!(cell.route, Route::F32Policy { .. })
    }));
}

#[test]
fn canonical_inventory_count_order_and_digest_are_frozen() {
    let cells = build_cells();
    let ids = cells.iter().copied().map(cell_id).collect::<Vec<_>>();
    assert_eq!(ids.len(), 652);
    assert_eq!(ids.iter().collect::<BTreeSet<_>>().len(), 652);
    assert_eq!(ids.first().unwrap(), "f32_policy_exact/nn/sq64/contiguous");
    assert_eq!(
        ids.last().unwrap(),
        "f32_tf32_forced_sm80_mma_tf32_v1_m16n16_bk32_s4/nt/prism_in_proj/contiguous"
    );
    assert_eq!(
        hex_digest(canonical_inventory_digest(&cells)),
        "cdfe923e3b00710e2cd52a9e942961b64d7f9d7b6b9ff24eebb202858526ac5d"
    );
}

#[test]
fn every_cell_has_physical_evidence() {
    let cells = build_cells();
    let f32_policy = cells
        .iter()
        .filter(|cell| matches!(cell.route, Route::F32Policy { .. }))
        .count();
    let half_policy = cells
        .iter()
        .filter(|cell| matches!(cell.route, Route::HalfPolicy { .. }))
        .count();
    let half_forced = cells
        .iter()
        .filter(|cell| matches!(cell.route, Route::HalfForced { .. }))
        .count();
    let tf32_forced = cells
        .iter()
        .filter(|cell| matches!(cell.route, Route::Tf32Forced(_)))
        .count();

    assert_eq!(
        (f32_policy, half_policy, half_forced, tf32_forced),
        (102, 204, 166, 180)
    );
    assert_eq!(f32_policy + half_policy + half_forced + tf32_forced, 652);
}

#[test]
fn f32_policy_comparison_uses_matching_production_call_scopes() {
    let cells = build_cells();
    let exact = cells
        .iter()
        .filter(|cell| {
            matches!(
                cell.route,
                Route::F32Policy {
                    policy: F32TriadPolicy::ExactScalarFmaV1,
                }
            )
        })
        .count();
    let allow = cells
        .iter()
        .filter(|cell| {
            matches!(
                cell.route,
                Route::F32Policy {
                    policy: F32TriadPolicy::AllowDeterministicTf32V1,
                }
            )
        })
        .count();
    let expected = SHAPES.len() * 3;

    assert_eq!(exact, expected);
    assert_eq!(allow, expected);
    assert!(cells.iter().all(|cell| {
        logical_route_name(cell.route) != "f32_scalar"
            && (!matches!(cell.route, Route::F32Policy { .. })
                || call_scope(cell.route) == "production_cached")
    }));
}

#[test]
fn every_tf32_candidate_covers_projection_and_prism_shapes() {
    let cells = build_cells();
    let projection_shapes = [
        "d128_in_proj",
        "d128_out_proj",
        "d768_in_proj",
        "d768_out_proj",
        "prism_in_proj",
    ];

    for spec in &SM80_TF32_ROUTE_SPECS {
        for shape in projection_shapes {
            assert!(cells.iter().any(|cell| {
                cell.op == spec.op
                    && cell.shape.name == shape
                    && matches!(
                        cell.route,
                        Route::Tf32Forced(found) if found.symbol == spec.symbol
                    )
            }));
        }
    }
}

#[test]
fn sm120_tf32_hot_suite_totals_follow_the_exact_route_inventory() {
    let shapes = SHAPES
        .iter()
        .filter(|shape| SM120_TF32_HOT_SHAPES.contains(&shape.name))
        .count();
    // The exact-F32 routes share the module but belong to the exact policy;
    // the TF32 hot suite times the TF32 routes only.
    let specs = tf32_route_specs(ModuleKind::TriadSm120);
    let routes = |op| {
        specs
            .iter()
            .filter(|spec| spec.op == op && !spec.route.is_exact_fma())
            .count()
    };
    assert_eq!(
        specs
            .iter()
            .filter(|spec| spec.route.is_exact_fma())
            .count(),
        12
    );

    assert_eq!(shapes, 5);
    assert_eq!(routes(ResolvedGemmOp::Nn), 6);
    assert_eq!(routes(ResolvedGemmOp::Tn), 7);
    assert_eq!(routes(ResolvedGemmOp::Nt), 5);
    assert_eq!(shapes * routes(ResolvedGemmOp::Tn), 35);
    assert_eq!(
        shapes * (routes(ResolvedGemmOp::Nn) + routes(ResolvedGemmOp::Nt)),
        55
    );
}

#[test]
fn performance_cell_keys_pin_the_contiguous_stride_class() {
    let cells = build_cells();

    assert!(cells.iter().all(|cell| {
        cell.stride_class == StrideClass::Contiguous && cell_id(*cell).ends_with("/contiguous")
    }));
    assert_eq!(
        cells
            .iter()
            .map(|cell| cell_id(*cell))
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        expected_cell_count()
    );
}

#[test]
fn edge_inventory_count_membership_order_and_digest_are_frozen() {
    let cells = build_edge_cells();
    let ids = cells.iter().copied().map(edge_cell_id).collect::<Vec<_>>();
    let cohort_counts = cells.iter().fold(BTreeMap::new(), |mut counts, cell| {
        *counts.entry(cell.cohort).or_insert(0usize) += 1;
        counts
    });

    assert_eq!(ids.len(), 533);
    assert_eq!(ids.iter().collect::<BTreeSet<_>>().len(), 533);
    assert_eq!(
        cohort_counts,
        BTreeMap::from([
            (EdgeCohort::Boundary, 405),
            (EdgeCohort::DeepK, 48),
            (EdgeCohort::Offset, 72),
            (EdgeCohort::DirectPaddedNn, 8),
        ])
    );
    assert_eq!(
        ids.first().map(String::as_str),
        Some("edge/boundary/f32_policy_exact/nn/m31_k32_n32/contiguous")
    );
    assert_eq!(
        ids.last().map(String::as_str),
        Some("edge/padded_nn/f16_forced_tile128/nn/m129_k65_n129/lda72_ldb136_ldc131")
    );
    assert_eq!(
        hex_digest(edge_inventory_digest(&cells)),
        EDGE_INVENTORY_DIGEST
    );
    validate_edge_inventory(&cells).expect("validate frozen edge inventory");
}

#[test]
fn edge_inventory_rejects_count_order_route_and_stride_label_mutations() {
    let cells = build_edge_cells();

    let mut missing = cells.clone();
    missing.pop();
    assert!(validate_edge_inventory(&missing).is_err());

    let mut reordered = cells.clone();
    reordered.swap(0, 1);
    assert!(validate_edge_inventory(&reordered).is_err());

    let mut route_changed = cells.clone();
    route_changed[0].route = Route::F32Policy {
        policy: F32TriadPolicy::AllowDeterministicTf32V1,
    };
    assert!(validate_edge_inventory(&route_changed).is_err());

    let padded = cells
        .iter()
        .position(|cell| cell.cohort == EdgeCohort::DirectPaddedNn)
        .expect("direct padded edge cell");
    let mut label_changed = cells;
    label_changed[padded].layout = EdgeLayout::DirectPaddedNn(PaddedNnLayout {
        label: "renamed_stride_class",
        ..match label_changed[padded].layout {
            EdgeLayout::DirectPaddedNn(layout) => layout,
            _ => unreachable!(),
        }
    });
    assert!(validate_edge_inventory(&label_changed).is_err());
}

#[test]
fn unsupported_padded_tn_nt_descriptors_are_explicit_and_untimed() {
    let descriptors = build_unsupported_padded_descriptors();
    let ids = descriptors
        .iter()
        .copied()
        .map(unsupported_padded_id)
        .collect::<Vec<_>>();

    assert_eq!(descriptors.len(), 16);
    assert_eq!(ids.iter().collect::<BTreeSet<_>>().len(), 16);
    assert!(descriptors.iter().all(|descriptor| {
        matches!(descriptor.op, ResolvedGemmOp::Tn | ResolvedGemmOp::Nt)
            && descriptor.contract == "unsupported_by_kernel_abi"
    }));
    assert_eq!(
        hex_digest(unsupported_padded_inventory_digest(&descriptors)),
        UNSUPPORTED_PADDED_INVENTORY_DIGEST
    );
    validate_unsupported_padded_inventory(&descriptors)
        .expect("validate unsupported padded inventory");
    for descriptor in descriptors {
        let rendered = render_unsupported_padded_descriptor(descriptor);
        for forbidden in [
            "samples_us",
            "iterations",
            "tflops",
            "path",
            "physical_nodes",
            "physical_launch_digest",
        ] {
            assert!(!rendered.contains(forbidden), "forbidden field {forbidden}");
        }
    }
}

#[test]
fn unsupported_padded_inventory_rejects_membership_order_and_layout_mutations() {
    let descriptors = build_unsupported_padded_descriptors();
    let mut missing = descriptors.clone();
    missing.pop();
    assert!(validate_unsupported_padded_inventory(&missing).is_err());

    let mut reordered = descriptors.clone();
    reordered.swap(0, 1);
    assert!(validate_unsupported_padded_inventory(&reordered).is_err());

    let mut route_changed = descriptors.clone();
    route_changed[0].route = Route::HalfForced {
        dtype: WeightDtype::F16,
        tile: TcTile::Tile64,
    };
    assert!(validate_unsupported_padded_inventory(&route_changed).is_err());

    let mut layout_changed = descriptors;
    layout_changed[0].layout.label = "renamed_unsupported_stride";
    assert!(validate_unsupported_padded_inventory(&layout_changed).is_err());
}

#[test]
fn unsupported_padded_inventory_is_emitted_once_across_shards() {
    let mut combined = Vec::new();
    for shard in 0..2 {
        combined.extend(
            unsupported_padded_descriptors_for_shard(shard)
                .expect("select unsupported descriptors for valid shard"),
        );
    }
    assert_eq!(combined.len(), UNSUPPORTED_PADDED_INVENTORY_COUNT);
    assert_eq!(
        combined
            .iter()
            .copied()
            .map(unsupported_padded_id)
            .collect::<BTreeSet<_>>()
            .len(),
        UNSUPPORTED_PADDED_INVENTORY_COUNT
    );
    validate_unsupported_padded_inventory(&combined)
        .expect("validate combined unsupported descriptor inventory");
}

#[test]
fn building_edge_inventory_does_not_change_the_canonical_contract() {
    let before = build_cells();
    let before_ids = before.iter().copied().map(cell_id).collect::<Vec<_>>();
    let before_digest = canonical_inventory_digest(&before);

    let _ = build_edge_cells();

    let after = build_cells();
    assert_eq!(
        after.iter().copied().map(cell_id).collect::<Vec<_>>(),
        before_ids
    );
    assert_eq!(canonical_inventory_digest(&after), before_digest);
    assert_eq!(hex_digest(before_digest), CANONICAL_INVENTORY_DIGEST);
}

#[test]
fn edge_frozen_iteration_manifest_is_independent_strict_and_ordered() {
    let cells = build_edge_cells();
    let entries = cells
        .iter()
        .copied()
        .map(|cell| {
            (
                edge_cell_id(cell),
                FrozenIterations {
                    eager: 23,
                    graph: 41,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let rendered = render_edge_frozen_iterations(&entries, &cells).expect("render edge iterations");
    assert!(rendered.starts_with(concat!(
        "format\tgemm-bi-frozen-iterations.v1\n",
        "suite\tgemm_bi_deterministic_performance_edges\n",
        "inventory_count\t533\n",
        "inventory_digest\td7f2e9d705b5a25e695059046ef73faf0f4d43ef3b187d70456813668c19be29\n",
        "cell_id\teager_iterations\tgraph_iterations\n",
        "edge/boundary/f32_policy_exact/nn/m31_k32_n32/contiguous\t23\t41\n",
    )));
    assert_eq!(
        parse_edge_frozen_iterations(&rendered, &cells).unwrap(),
        entries
    );

    let mut reordered = rendered.lines().map(str::to_string).collect::<Vec<_>>();
    reordered.swap(5, 6);
    assert!(parse_edge_frozen_iterations(&(reordered.join("\n") + "\n"), &cells).is_err());
    assert!(parse_frozen_iterations(&rendered, &build_cells()).is_err());

    for changed in [
        rendered.replacen(
            "suite\tgemm_bi_deterministic_performance_edges",
            "suite\tother",
            1,
        ),
        rendered.replacen("inventory_count\t533", "inventory_count\t532", 1),
        rendered.replacen(EDGE_INVENTORY_DIGEST, &"0".repeat(64), 1),
    ] {
        assert!(parse_edge_frozen_iterations(&changed, &cells).is_err());
    }
    let mut missing = rendered.lines().map(str::to_string).collect::<Vec<_>>();
    missing.pop();
    assert!(parse_edge_frozen_iterations(&(missing.join("\n") + "\n"), &cells).is_err());

    let mut extra = rendered.clone();
    extra.push_str("extra\t1\t1\n");
    assert!(parse_edge_frozen_iterations(&extra, &cells).is_err());

    let mut duplicate = rendered.lines().map(str::to_string).collect::<Vec<_>>();
    duplicate[6] = duplicate[5].clone();
    assert!(parse_edge_frozen_iterations(&(duplicate.join("\n") + "\n"), &cells).is_err());

    let zero = rendered.replacen("\t23\t41", "\t0\t41", 1);
    assert!(parse_edge_frozen_iterations(&zero, &cells).is_err());
}

#[test]
fn edge_runner_mapping_selection_and_call_scope_are_explicit() {
    let cells = build_edge_cells();
    let first = edge_cell_id(cells[0]);
    let last = edge_cell_id(cells[532]);
    let requested = parse_edge_qualification_cell_ids(&format!("{last},{first}"), &cells)
        .expect("parse edge cell selection");
    let selected =
        select_edge_qualification_cells(&cells, Some(&requested), 0, 1).expect("select edge cells");
    assert_eq!(
        selected.into_iter().map(edge_cell_id).collect::<Vec<_>>(),
        [first, last]
    );

    for cell in cells {
        edge_qualification_request(cell).expect("map edge qualification request");
        assert_eq!(
            edge_call_scope(cell),
            if cell.cohort == EdgeCohort::DirectPaddedNn {
                "direct_kernel_forced"
            } else {
                call_scope(cell.route)
            }
        );
    }
}

#[test]
fn cublas_denominator_inventory_is_full_unique_and_ordered() {
    let cells = build_cublas_denominator_cells();
    assert_eq!(cells.len(), SHAPES.len() * 3 * 3);

    let ids = cells
        .iter()
        .copied()
        .map(cublas_denominator_cell_id)
        .collect::<Vec<_>>();
    assert_eq!(ids.first().map(String::as_str), Some("cublas/f32/nn/sq64"));
    assert_eq!(
        ids.last().map(String::as_str),
        Some("cublas/f16/nt/prism_in_proj")
    );
    assert_eq!(ids.iter().collect::<BTreeSet<_>>().len(), ids.len());
}

#[test]
fn cublas_denominator_cell_selection_is_exact_and_preserves_order() {
    let cells = build_cublas_denominator_cells();
    let first = "cublas/bf16/tn/large";
    let second = "cublas/f16/tn/d768_out_proj";
    let requested = parse_cublas_denominator_cell_ids(&format!("{second},{first}"), &cells)
        .expect("parse exact cuBLAS denominator cells");
    let selected = select_cublas_denominator_cells(&cells, Some(&requested), 0, 1)
        .expect("select exact cuBLAS denominator cells");
    assert_eq!(
        selected
            .into_iter()
            .map(cublas_denominator_cell_id)
            .collect::<Vec<_>>(),
        [first, second]
    );

    for invalid in [
        "",
        "cublas/bf16/tn/large,cublas/bf16/tn/large",
        "cublas/bf16/tn/large ",
        "cublas/bf16/tn/not_a_shape",
    ] {
        assert!(parse_cublas_denominator_cell_ids(invalid, &cells).is_err());
    }
    assert!(select_cublas_denominator_cells(&cells, None, 0, 0).is_err());
}

#[test]
fn cublas_denominator_modes_pin_the_requested_speed_ceiling() {
    use cudarc::cublas::sys::cublasComputeType_t;

    assert_eq!(CublasDenominatorMode::Fast.as_str(), "fast");
    assert_eq!(CublasDenominatorMode::Pedantic.as_str(), "pedantic");
    assert_eq!(
        cublas_denominator_compute(WeightDtype::F32, CublasDenominatorMode::Fast),
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    );
    assert_eq!(
        cublas_denominator_compute(WeightDtype::F32, CublasDenominatorMode::Pedantic),
        cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC
    );
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        assert_eq!(
            cublas_denominator_compute(dtype, CublasDenominatorMode::Fast),
            cublasComputeType_t::CUBLAS_COMPUTE_32F
        );
        assert_eq!(
            cublas_denominator_compute(dtype, CublasDenominatorMode::Pedantic),
            cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC
        );
    }
}

#[test]
fn cublas_denominator_output_dtype_matches_the_training_contract() {
    for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
        assert_eq!(
            cublas_denominator_output_dtype(dtype, ResolvedGemmOp::Tn),
            WeightDtype::F32
        );
        assert_eq!(
            cublas_denominator_output_dtype(dtype, ResolvedGemmOp::Nn),
            dtype
        );
        assert_eq!(
            cublas_denominator_output_dtype(dtype, ResolvedGemmOp::Nt),
            dtype
        );
    }
}

#[test]
fn cublas_denominator_geometry_matches_row_major_nn_tn_nt() {
    use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};

    let nn = cublas_denominator_geometry(ResolvedGemmOp::Nn, (5, 7, 11));
    assert_eq!((nn.trans_a, nn.trans_b), (CUBLAS_OP_N, CUBLAS_OP_N));
    assert_eq!((nn.m, nn.n, nn.k), (11, 5, 7));
    assert_eq!((nn.lda, nn.ldb, nn.ldc), (11, 7, 11));

    let tn = cublas_denominator_geometry(ResolvedGemmOp::Tn, (5, 7, 11));
    assert_eq!((tn.trans_a, tn.trans_b), (CUBLAS_OP_N, CUBLAS_OP_T));
    assert_eq!((tn.m, tn.n, tn.k), (11, 7, 5));
    assert_eq!((tn.lda, tn.ldb, tn.ldc), (11, 7, 11));

    let nt = cublas_denominator_geometry(ResolvedGemmOp::Nt, (5, 7, 11));
    assert_eq!((nt.trans_a, nt.trans_b), (CUBLAS_OP_T, CUBLAS_OP_N));
    assert_eq!((nt.m, nt.n, nt.k), (7, 5, 11));
    assert_eq!((nt.lda, nt.ldb, nt.ldc), (11, 11, 7));
}

#[test]
fn tn_narrow_cublas_pair_inventory_and_measurement_modes_are_exact() {
    use tn_narrow_cublas_pair::{
        PairOrder, candidate_partial_symbol, candidate_plan, cells, denominator_cell,
        paired_schedule, window_count,
    };

    let cells = cells();
    assert_eq!(cells.len(), 7);
    assert_eq!(
        cells
            .iter()
            .map(|cell| {
                (
                    cell.name,
                    cell.dims,
                    (cell.a_offset, cell.b_offset, cell.output_offset),
                    candidate_partial_symbol(*cell),
                )
            })
            .collect::<Vec<_>>(),
        [
            (
                "m256_k32_n2",
                (256, 32, 2),
                (0, 0, 0),
                "gemm_bi_tn_narrow_splitm_partial_aligned",
            ),
            (
                "m1024_k47_n17",
                (1_024, 47, 17),
                (0, 0, 0),
                "gemm_bi_tn_narrow_splitm_partial_aligned",
            ),
            (
                "m1024_k128_n25",
                (1_024, 128, 25),
                (0, 0, 0),
                "gemm_bi_tn_narrow_splitm_partial_aligned",
            ),
            (
                "m4096_k64_n64",
                (4_096, 64, 64),
                (0, 0, 0),
                "gemm_bi_tn_narrow_splitm_partial_aligned",
            ),
            (
                "m4096_k128_n96",
                (4_096, 128, 96),
                (0, 0, 0),
                "gemm_bi_tn_narrow_splitm_partial_aligned",
            ),
            (
                "m4096_k256_n101_offset1",
                (4_096, 256, 101),
                (1, 0, 1),
                "gemm_bi_tn_narrow_splitm_partial",
            ),
            (
                "m4111_k257_n127_offset1",
                (4_111, 257, 127),
                (0, 1, 1),
                "gemm_bi_tn_narrow_splitm_partial",
            ),
        ]
    );
    for cell in cells {
        let denominator = denominator_cell(*cell);
        assert_eq!(denominator.dtype, WeightDtype::F32);
        assert_eq!(denominator.op, ResolvedGemmOp::Tn);
        assert_eq!(denominator.shape.dims, cell.dims);
        assert_eq!(
            cublas_denominator_compute(denominator.dtype, CublasDenominatorMode::Fast),
            cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        );
    }
    assert_eq!(
        cells
            .iter()
            .map(|cell| {
                let plan = candidate_plan(*cell, 170).unwrap();
                (plan.m_chunk, plan.chunks)
            })
            .collect::<Vec<_>>(),
        [
            (16, 16),
            (16, 64),
            (16, 64),
            (32, 128),
            (80, 52),
            (192, 22),
            (256, 17),
        ]
    );

    assert_eq!(window_count(None).unwrap(), 21);
    assert_eq!(window_count(Some("discovery")).unwrap(), 21);
    assert_eq!(window_count(Some("21")).unwrap(), 21);
    assert_eq!(window_count(Some("final")).unwrap(), 101);
    assert_eq!(window_count(Some("101")).unwrap(), 101);
    assert!(window_count(Some("100")).is_err());

    let schedule = paired_schedule(21).unwrap();
    assert_eq!(schedule.len(), 42);
    assert_eq!(schedule[0], PairOrder::CandidateFirst);
    assert_eq!(schedule[1], PairOrder::CublasFirst);
    assert_eq!(
        schedule
            .iter()
            .filter(|order| **order == PairOrder::CandidateFirst)
            .count(),
        21
    );
    assert_eq!(
        schedule
            .iter()
            .filter(|order| **order == PairOrder::CublasFirst)
            .count(),
        21
    );
    assert!(paired_schedule(0).is_err());
}

#[test]
fn tn_narrow_cublas_pair_semantic_math_and_ratios_are_explicit() {
    use tn_narrow_cublas_pair::{one_term_probe, paired_ratios, tn_reference};

    let probe = one_term_probe((2, 2, 2)).unwrap();
    assert_eq!(probe.x, [0.0, 0.0, 0.25, -0.5]);
    assert_eq!(probe.dy, [0.0, 0.0, 0.5, -0.25]);
    assert_eq!(probe.seed, [0.03125, -0.0625, 0.09375, -0.125]);
    assert_eq!(probe.expected, [0.15625, -0.125, -0.15625, 0.0]);
    assert!(probe.seed.iter().all(|value| *value != 0.0));

    let expected = tn_reference(
        (2, 2, 2),
        &[1.0, 2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0, 8.0],
        &[0.5, 1.0, 1.5, 2.0],
    )
    .unwrap();
    assert_eq!(expected, [26.5, 31.0, 39.5, 46.0]);
    assert!(tn_reference((2, 2, 2), &[1.0], &[1.0; 4], &[1.0; 4]).is_err());

    assert_eq!(
        paired_ratios(&[2.0, 3.0, 4.0], &[1.0, 1.5, 2.0], 3).unwrap(),
        [2.0, 2.0, 2.0]
    );
    assert!(paired_ratios(&[2.0], &[1.0], 3).is_err());
    assert!(paired_ratios(&[2.0, 3.0, 4.0], &[1.0, 0.0, 2.0], 3).is_err());
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_nn_inventory_is_exact_unique_and_pins_selected_routes() {
    use cublaslt_qualification::*;

    let cells = build_sm120_paired_nn_cells();
    assert_eq!(SM120_PAIRED_WINDOWS_PER_ORDER, 101);
    assert_eq!(SM120_PAIRED_WARMUP_LAUNCHES, 128);
    assert_eq!(SM120_PAIRED_LT_WORKSPACE_BYTES, 32 * 1024 * 1024);
    assert_eq!(cells.len(), 6);
    assert_eq!(
        cells
            .iter()
            .copied()
            .map(sm120_paired_nn_cell_id)
            .collect::<Vec<_>>(),
        [
            "sm120_paired/bf16/nn/large",
            "sm120_paired/bf16/nn/large_deep",
            "sm120_paired/bf16/nn/d768_out_proj",
            "sm120_paired/f16/nn/large",
            "sm120_paired/f16/nn/large_deep",
            "sm120_paired/f16/nn/d768_out_proj",
        ]
    );
    assert_eq!(
        cells
            .iter()
            .copied()
            .map(sm120_paired_nn_cell_id)
            .collect::<BTreeSet<_>>()
            .len(),
        6
    );

    for cell in cells {
        let expected = if cell.shape.name == "large_deep" {
            (Sm120Tile::M128N64, Sm120Bk::Bk32, Sm120Stages::S2)
        } else {
            (Sm120Tile::M64N64, Sm120Bk::Bk64, Sm120Stages::S2)
        };
        assert_eq!(
            (cell.physical.tile, cell.physical.bk, cell.physical.stages),
            expected
        );
        assert_eq!(cell.alpha.to_bits(), 1.0f32.to_bits());
        assert_eq!(cell.beta.to_bits(), 0.0f32.to_bits());
        assert!(!cell.bias_present);
    }
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_nn_cublaslt_descriptor_keeps_no_bias_transpose_identity() {
    use cublaslt_qualification::*;
    use cudarc::cublaslt::sys::{cublasComputeType_t, cublasLtEpilogue_t, cudaDataType_t};

    let cell = Sm120PairedNnCell {
        dtype: WeightDtype::Bf16,
        shape: Shape {
            name: "fixture",
            dims: (5, 7, 11),
        },
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        alpha: 1.0,
        beta: 0.0,
        bias_present: false,
    };
    let descriptor = sm120_paired_nn_lt_descriptor(cell).expect("valid Lt descriptor contract");
    assert_eq!(descriptor.compute, cublasComputeType_t::CUBLAS_COMPUTE_32F);
    assert_eq!(descriptor.scale_dtype, cudaDataType_t::CUDA_R_32F);
    assert_eq!(
        descriptor.epilogue,
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT
    );
    assert_eq!(
        (descriptor.a.rows, descriptor.a.columns, descriptor.a.ld),
        (11, 7, 11)
    );
    assert_eq!(descriptor.a.dtype, cudaDataType_t::CUDA_R_16BF);
    assert_eq!(
        (descriptor.b.rows, descriptor.b.columns, descriptor.b.ld),
        (7, 5, 7)
    );
    assert_eq!(descriptor.b.dtype, cudaDataType_t::CUDA_R_16BF);
    assert_eq!(
        (
            descriptor.output.rows,
            descriptor.output.columns,
            descriptor.output.ld
        ),
        (11, 5, 11)
    );
    assert_eq!(descriptor.output.dtype, cudaDataType_t::CUDA_R_16BF);
    assert_eq!(descriptor.alpha.to_bits(), 1.0f32.to_bits());
    assert_eq!(descriptor.beta.to_bits(), 0.0f32.to_bits());

    let mut f16_cell = cell;
    f16_cell.dtype = WeightDtype::F16;
    let f16 = sm120_paired_nn_lt_descriptor(f16_cell).expect("valid F16 Lt contract");
    assert_eq!(f16.a.dtype, cudaDataType_t::CUDA_R_16F);
    assert_eq!(f16.b.dtype, cudaDataType_t::CUDA_R_16F);
    assert_eq!(f16.output.dtype, cudaDataType_t::CUDA_R_16F);
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_nn_cublaslt_descriptor_rejects_unmatched_epilogues() {
    use cublaslt_qualification::*;

    let mut cell = build_sm120_paired_nn_cells()[0];
    cell.alpha = 0.75;
    assert!(sm120_paired_nn_lt_descriptor(cell).is_err());

    cell.alpha = 1.0;
    cell.beta = -0.25;
    assert!(sm120_paired_nn_lt_descriptor(cell).is_err());

    cell.beta = 0.0;
    cell.bias_present = true;
    assert!(sm120_paired_nn_lt_descriptor(cell).is_err());
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_nn_schedule_alternates_101_windows_per_comparator_order() {
    use cublaslt_qualification::*;

    let schedule = sm120_paired_comparator_schedule();
    assert_eq!(schedule.len(), 202);
    assert_eq!(
        schedule
            .iter()
            .filter(|order| **order == ComparatorOrder::CustomThenCublas)
            .count(),
        101
    );
    assert_eq!(
        schedule
            .iter()
            .filter(|order| **order == ComparatorOrder::CublasThenCustom)
            .count(),
        101
    );
    for pair in schedule.chunks_exact(2) {
        assert_eq!(
            pair,
            [
                ComparatorOrder::CustomThenCublas,
                ComparatorOrder::CublasThenCustom,
            ]
        );
    }
    assert_eq!(
        ComparatorOrder::CustomThenCublas.as_str(),
        "custom_then_cublas"
    );
    assert_eq!(
        ComparatorOrder::CublasThenCustom.as_str(),
        "cublas_then_custom"
    );
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_nn_ratios_preserve_window_pairing_and_custom_over_cublas_orientation() {
    use cublaslt_qualification::*;

    let custom = (1..=101)
        .map(|value| value as f64 * 2.0)
        .collect::<Vec<_>>();
    let cublas = (1..=101).map(|value| value as f64).collect::<Vec<_>>();
    let ratios = sm120_paired_ratios(&custom, &cublas).expect("valid paired ratios");
    assert_eq!(ratios, vec![2.0; 101]);
    assert_eq!(sm120_paired_ratio_p50(&ratios).unwrap(), 2.0);

    assert!(sm120_paired_ratios(&custom[..100], &cublas[..100]).is_err());
    let mut invalid = cublas;
    invalid[50] = 0.0;
    assert!(sm120_paired_ratios(&custom, &invalid).is_err());
    invalid[50] = f64::NAN;
    assert!(sm120_paired_ratios(&custom, &invalid).is_err());
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_tn_nt_inventory_is_exact_unique_and_pins_accepted_routes() {
    use cublaslt_tn_nt_qualification::*;

    let cells = build_sm120_paired_tn_nt_cells();
    assert_eq!(cells.len(), 12);
    assert_eq!(SM120_PAIRED_FINAL_WINDOWS_PER_ORDER, 101);
    assert_eq!(SM120_PAIRED_PILOT_WINDOWS_PER_ORDER, 11);
    assert_eq!(SM120_PAIRED_WARMUP_LAUNCHES, 128);
    assert_eq!(SM120_PAIRED_LT_HEURISTIC_LIMIT, 32);
    assert_eq!(
        cells
            .iter()
            .copied()
            .map(sm120_paired_tn_nt_cell_id)
            .collect::<Vec<_>>(),
        [
            "sm120_paired/bf16/tn/large",
            "sm120_paired/bf16/tn/large_deep",
            "sm120_paired/bf16/tn/d768_out_proj",
            "sm120_paired/bf16/nt/large",
            "sm120_paired/bf16/nt/large_deep",
            "sm120_paired/bf16/nt/d768_out_proj",
            "sm120_paired/f16/tn/large",
            "sm120_paired/f16/tn/large_deep",
            "sm120_paired/f16/tn/d768_out_proj",
            "sm120_paired/f16/nt/large",
            "sm120_paired/f16/nt/large_deep",
            "sm120_paired/f16/nt/d768_out_proj",
        ]
    );
    assert_eq!(
        cells
            .iter()
            .copied()
            .map(sm120_paired_tn_nt_cell_id)
            .collect::<BTreeSet<_>>()
            .len(),
        12
    );

    for cell in cells {
        let expected = match (cell.op, cell.shape.name) {
            (Sm120Op::Tn, "large" | "d768_out_proj") => {
                (Sm120Tile::M64N128, Sm120Bk::Bk32, Sm120Stages::S3)
            }
            (Sm120Op::Tn, "large_deep") => (Sm120Tile::M128N128, Sm120Bk::Bk32, Sm120Stages::S3),
            (Sm120Op::Nt, "d768_out_proj") => (Sm120Tile::M64N64, Sm120Bk::Bk64, Sm120Stages::S2),
            (Sm120Op::Nt, "large") => (Sm120Tile::M128N64, Sm120Bk::Bk32, Sm120Stages::S2),
            (Sm120Op::Nt, "large_deep") => (Sm120Tile::M128N128, Sm120Bk::Bk32, Sm120Stages::S3),
            _ => panic!("unexpected paired cell"),
        };
        assert_eq!(
            (cell.physical.tile, cell.physical.bk, cell.physical.stages),
            expected
        );
        let contract = sm120_paired_contract(cell).expect("valid paired contract");
        assert_eq!(contract.alpha.to_bits(), 1.0f32.to_bits());
        assert_eq!(contract.bias_ptr, 0);
        match cell.op {
            Sm120Op::Tn => {
                assert_eq!(contract.output_dtype, WeightDtype::F32);
                assert_eq!(contract.beta.to_bits(), 1.0f32.to_bits());
                assert!(contract.reset_before_window);
            }
            Sm120Op::Nt => {
                assert_eq!(contract.output_dtype, cell.dtype);
                assert_eq!(contract.beta.to_bits(), 0.0f32.to_bits());
                assert!(!contract.reset_before_window);
            }
            Sm120Op::Nn => panic!("NN is not in the TN/NT inventory"),
        }
    }
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_tn_nt_layouts_match_the_row_major_training_contracts() {
    use cublaslt_tn_nt_qualification::*;
    use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
    use cudarc::cublaslt::sys::{cublasComputeType_t, cudaDataType_t};

    let fixture = |op| Sm120PairedTnNtCell {
        dtype: WeightDtype::Bf16,
        op,
        shape: Shape {
            name: "fixture",
            dims: (5, 7, 11),
        },
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
    };

    let tn = sm120_paired_lt_descriptor(fixture(Sm120Op::Tn)).expect("TN Lt contract");
    assert_eq!(tn.compute, cublasComputeType_t::CUBLAS_COMPUTE_32F);
    assert_eq!((tn.trans_a, tn.trans_b), (CUBLAS_OP_N, CUBLAS_OP_T));
    assert_eq!(
        (tn.a.dtype, tn.a.rows, tn.a.columns, tn.a.ld),
        (cudaDataType_t::CUDA_R_16BF, 11, 5, 11)
    );
    assert_eq!(
        (tn.b.dtype, tn.b.rows, tn.b.columns, tn.b.ld),
        (cudaDataType_t::CUDA_R_16BF, 7, 5, 7)
    );
    assert_eq!(
        (tn.c.dtype, tn.c.rows, tn.c.columns, tn.c.ld),
        (cudaDataType_t::CUDA_R_32F, 11, 7, 11)
    );
    assert_eq!(tn.c, tn.d);
    assert_eq!(tn.alpha.to_bits(), 1.0f32.to_bits());
    assert_eq!(tn.beta.to_bits(), 1.0f32.to_bits());

    let nt = sm120_paired_lt_descriptor(fixture(Sm120Op::Nt)).expect("NT Lt contract");
    assert_eq!((nt.trans_a, nt.trans_b), (CUBLAS_OP_T, CUBLAS_OP_N));
    assert_eq!(
        (nt.a.dtype, nt.a.rows, nt.a.columns, nt.a.ld),
        (cudaDataType_t::CUDA_R_16BF, 11, 7, 11)
    );
    assert_eq!(
        (nt.b.dtype, nt.b.rows, nt.b.columns, nt.b.ld),
        (cudaDataType_t::CUDA_R_16BF, 11, 5, 11)
    );
    assert_eq!(
        (nt.c.dtype, nt.c.rows, nt.c.columns, nt.c.ld),
        (cudaDataType_t::CUDA_R_16BF, 7, 5, 7)
    );
    assert_eq!(nt.c, nt.d);
    assert_eq!(nt.beta.to_bits(), 0.0f32.to_bits());

    let tn_geometry = sm120_paired_gemm_ex_geometry(fixture(Sm120Op::Tn));
    assert_eq!(
        (
            tn_geometry.trans_a,
            tn_geometry.trans_b,
            tn_geometry.m,
            tn_geometry.n,
            tn_geometry.k,
            tn_geometry.lda,
            tn_geometry.ldb,
            tn_geometry.ldc,
        ),
        (CUBLAS_OP_N, CUBLAS_OP_T, 11, 7, 5, 11, 7, 11)
    );
    let nt_geometry = sm120_paired_gemm_ex_geometry(fixture(Sm120Op::Nt));
    assert_eq!(
        (
            nt_geometry.trans_a,
            nt_geometry.trans_b,
            nt_geometry.m,
            nt_geometry.n,
            nt_geometry.k,
            nt_geometry.lda,
            nt_geometry.ldb,
            nt_geometry.ldc,
        ),
        (CUBLAS_OP_T, CUBLAS_OP_N, 7, 5, 11, 11, 11, 7)
    );
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_tn_nt_semantic_probe_has_one_exact_term_and_stable_tn_seed() {
    use cublaslt_tn_nt_qualification::*;

    let fixture = |op, dtype| Sm120PairedTnNtCell {
        dtype,
        op,
        shape: Shape {
            name: "fixture",
            dims: (2, 3, 4),
        },
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
    };

    let tn = build_sm120_paired_semantic_probe(fixture(Sm120Op::Tn, WeightDtype::F16))
        .expect("TN semantic probe");
    assert_eq!(tn.a.iter().filter(|value| **value != 0.0).count(), 3);
    assert_eq!(tn.b.iter().filter(|value| **value != 0.0).count(), 4);
    assert_eq!(tn.seed.len(), 12);
    assert!(tn.seed.iter().all(|value| *value != 0.0));
    for row in 0..3 {
        for column in 0..4 {
            let index = row * 4 + column;
            assert_eq!(
                tn.expected[index].to_bits(),
                (tn.seed[index] + tn.a[row] * tn.b[column]).to_bits()
            );
        }
    }
    assert!(tn.timed_a.iter().all(|value| *value == 0.0));
    assert!(tn.timed_b.iter().all(|value| *value == 0.0));
    assert_eq!(tn.timed_seed, tn.seed);

    let nt = build_sm120_paired_semantic_probe(fixture(Sm120Op::Nt, WeightDtype::Bf16))
        .expect("NT semantic probe");
    assert_eq!(nt.a.iter().filter(|value| **value != 0.0).count(), 2);
    assert_eq!(nt.b.iter().filter(|value| **value != 0.0).count(), 3);
    assert!(nt.seed.iter().all(|value| *value == 0.0));
    for row in 0..2 {
        for column in 0..3 {
            let index = row * 3 + column;
            let exact = half::bf16::from_f32(nt.a[row * 4] * nt.b[column * 4]).to_f32();
            assert_eq!(nt.expected[index].to_bits(), exact.to_bits());
        }
    }
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_tn_nt_schedules_are_disjoint_both_order_windows() {
    use cublaslt_tn_nt_qualification::*;

    let final_schedule = sm120_paired_final_schedule();
    assert_eq!(final_schedule.len(), 202);
    assert_eq!(
        final_schedule
            .iter()
            .filter(|order| **order == PairedOrder::CustomThenVendor)
            .count(),
        101
    );
    assert_eq!(
        final_schedule
            .iter()
            .filter(|order| **order == PairedOrder::VendorThenCustom)
            .count(),
        101
    );
    for pair in final_schedule.chunks_exact(2) {
        assert_eq!(
            pair,
            [PairedOrder::CustomThenVendor, PairedOrder::VendorThenCustom]
        );
    }
    assert_eq!(PairedOrder::CustomThenVendor.as_str(), "custom_then_vendor");
    assert_eq!(PairedOrder::VendorThenCustom.as_str(), "vendor_then_custom");

    let pilot_schedule = sm120_paired_pilot_schedule();
    assert_eq!(pilot_schedule.len(), 22);
    for pair in pilot_schedule.chunks_exact(2) {
        assert_eq!(
            pair,
            [PairedOrder::CustomThenVendor, PairedOrder::VendorThenCustom]
        );
    }
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_tn_nt_ratios_preserve_pairing_and_custom_over_vendor_orientation() {
    use cublaslt_tn_nt_qualification::*;

    let custom = (1..=101)
        .map(|value| value as f64 * 0.75)
        .collect::<Vec<_>>();
    let vendor = (1..=101).map(|value| value as f64).collect::<Vec<_>>();
    let ratios = sm120_paired_ratios(&custom, &vendor).expect("valid paired ratios");
    assert_eq!(ratios, vec![0.75; 101]);
    assert_eq!(sm120_paired_ratio_p50(&ratios).unwrap(), 0.75);
    assert!(sm120_paired_ratios(&custom[..100], &vendor[..100]).is_err());
    let mut invalid = vendor;
    invalid[17] = f64::INFINITY;
    assert!(sm120_paired_ratios(&custom, &invalid).is_err());
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_median_interval_is_conservative_at_95_percent() {
    use cublaslt_tn_nt_qualification::*;

    let descending = (0..=100).rev().map(f64::from).collect::<Vec<_>>();
    assert_eq!(
        sm120_paired_median_interval_95(&descending).unwrap(),
        (40.0, 60.0)
    );
    assert!(sm120_paired_median_interval_95(&descending[..100]).is_err());

    let exact_coverage = 0.9539559330706572_f64;
    assert!(exact_coverage >= 0.95);
    assert!((sm120_paired_median_interval_coverage() - exact_coverage).abs() < f64::EPSILON);
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_vendor_selection_uses_only_valid_both_order_pilots() {
    use cublaslt_tn_nt_qualification::*;

    let pilot = |kind, first, second| VendorPilot {
        kind,
        custom_then_vendor_us: vec![first; 11],
        vendor_then_custom_us: vec![second; 11],
    };
    let pilots = [
        pilot(VendorPlanKind::GemmEx, 2.0, 2.2),
        pilot(VendorPlanKind::CublasLt(3), 1.9, 1.8),
        pilot(VendorPlanKind::CublasLt(7), 1.7, 1.6),
    ];
    assert_eq!(
        select_fastest_vendor_plan(&pilots).expect("select pilot winner"),
        VendorPlanKind::CublasLt(7)
    );

    assert!(select_fastest_vendor_plan(&pilots[1..]).is_err());
    let mut wrong_window_count = pilots.to_vec();
    wrong_window_count[0].vendor_then_custom_us.pop();
    assert!(select_fastest_vendor_plan(&wrong_window_count).is_err());
    let mut invalid_sample = pilots.to_vec();
    invalid_sample[2].custom_then_vendor_us[5] = 0.0;
    assert!(select_fastest_vendor_plan(&invalid_sample).is_err());

    assert_eq!(
        reject_lt_heuristic(false, 0, true, 1024),
        Some(LtRejectionReason::Status)
    );
    assert_eq!(
        reject_lt_heuristic(true, 2048, true, 1024),
        Some(LtRejectionReason::Workspace)
    );
    assert_eq!(
        reject_lt_heuristic(true, 1024, false, 1024),
        Some(LtRejectionReason::Smoke)
    );
    assert_eq!(reject_lt_heuristic(true, 1024, true, 1024), None);
}

#[cfg(feature = "cuda-cublaslt-qualification")]
#[test]
fn sm120_paired_output_reset_metadata_distinguishes_tn_from_nt() {
    use cublaslt_tn_nt_qualification::*;

    assert_eq!(
        sm120_paired_output_reset_name(Sm120Op::Tn),
        "immutable_f32_seed_d2d_before_start_event"
    );
    assert_eq!(sm120_paired_output_reset_name(Sm120Op::Nt), "none_beta0");
    assert!(sm120_paired_output_reset_name(Sm120Op::Nn).is_empty());
}
