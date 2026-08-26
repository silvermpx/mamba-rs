use super::super::blas::TypedPtr;
use super::super::dtype::WeightDtype;
use crate::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, CompilerIdentity, CudaTarget, DeviceCaps, DeviceIdentity, FramedSha256,
    ModuleKind, PhysicalGemmBackend, PolicyDtype, ResolvedGemmLaunchSet, ResolvedGemmOp,
    ResolvedGemmRoute, ResolvedInstructionFamily, ResolvedInstructionShape,
    ResolvedNumericContract, ResolvedOperandConversion, SCHEDULE_REVISION, Sha256Digest,
    TUNING_TABLE_REVISION,
};
use cudarc::driver::{DeviceRepr, sys};

pub type CUptr = cudarc::driver::sys::CUdeviceptr;

pub const F32_TF32_TUNING_REVISION: u16 = TUNING_TABLE_REVISION;
pub const TF32_TENSOR_MAP_REVISION: u16 = 1;
pub const TF32_SCHEDULE_REVISION: u16 = SCHEDULE_REVISION;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct F32TriadShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
}

impl F32TriadShape {
    pub fn contiguous(op: ResolvedGemmOp, dims: (usize, usize, usize)) -> Self {
        let (m, k, n) = dims;
        match op {
            ResolvedGemmOp::Nn | ResolvedGemmOp::Tn => Self {
                m,
                k,
                n,
                lda: k,
                ldb: n,
                ldc: n,
            },
            ResolvedGemmOp::Nt => Self {
                m,
                k,
                n,
                lda: n,
                ldb: n,
                ldc: k,
            },
        }
    }

    pub const fn output_rows(self, op: ResolvedGemmOp) -> usize {
        match op {
            ResolvedGemmOp::Nn | ResolvedGemmOp::Nt => self.m,
            ResolvedGemmOp::Tn => self.k,
        }
    }

    pub const fn output_columns(self, op: ResolvedGemmOp) -> usize {
        match op {
            ResolvedGemmOp::Nn | ResolvedGemmOp::Tn => self.n,
            ResolvedGemmOp::Nt => self.k,
        }
    }

    pub const fn reduction(self, op: ResolvedGemmOp) -> usize {
        match op {
            ResolvedGemmOp::Nn => self.k,
            ResolvedGemmOp::Tn => self.m,
            ResolvedGemmOp::Nt => self.n,
        }
    }

    pub fn validate(self, op: ResolvedGemmOp) -> Result<(), String> {
        let output_rows = self.output_rows(op);
        let output_columns = self.output_columns(op);
        if output_rows == 0 || output_columns == 0 {
            return Err(invalid_gemm_dimensions(format!(
                "output axes must be positive, got rows={output_rows} columns={output_columns}"
            )));
        }
        for (value, name) in [(self.m, "M"), (self.k, "K"), (self.n, "N")] {
            checked_i32(value, name)?;
        }
        for (lhs, rhs, name) in [
            (self.m, self.k, "M*K"),
            (self.m, self.n, "M*N"),
            (self.k, self.n, "K*N"),
        ] {
            let product = lhs.checked_mul(rhs).ok_or_else(|| {
                invalid_gemm_dimensions(format!("{name} overflows usize ({lhs} * {rhs})"))
            })?;
            checked_i32(product, name)?;
        }
        let (widths, rows) = match op {
            ResolvedGemmOp::Nn => ([self.k, self.n, self.n], [self.m, self.k, self.m]),
            ResolvedGemmOp::Tn => ([self.k, self.n, self.n], [self.m, self.m, self.k]),
            ResolvedGemmOp::Nt => ([self.n, self.n, self.k], [self.m, self.k, self.m]),
        };
        for ((stride, width, rows), name) in [
            ((self.lda, widths[0], rows[0]), "lda"),
            ((self.ldb, widths[1], rows[1]), "ldb"),
            ((self.ldc, widths[2], rows[2]), "ldc"),
        ] {
            if stride < width {
                return Err(invalid_gemm_dimensions(format!(
                    "{name}={stride} is smaller than the physical width {width}"
                )));
            }
            checked_i32(stride, name)?;
            if rows != 0 && width != 0 {
                let span = (rows - 1)
                    .checked_mul(stride)
                    .and_then(|offset| offset.checked_add(width))
                    .ok_or_else(|| {
                        invalid_gemm_dimensions(format!("{name} storage span overflows usize"))
                    })?;
                checked_i32(span, &format!("{name} storage"))?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct F32TriadRequest {
    pub op: ResolvedGemmOp,
    pub shape: F32TriadShape,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F32TriadOperands {
    pub output: CUptr,
    pub a: CUptr,
    pub b: CUptr,
    pub bias: Option<CUptr>,
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tf32PortableTile {
    M128N64,
    M64N64,
    M16N32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tf32PortableStages {
    S2,
    S3,
    S4,
}

impl Tf32PortableStages {
    pub const fn count(self) -> u8 {
        match self {
            Self::S2 => 2,
            Self::S3 => 3,
            Self::S4 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tf32PortableRoute {
    pub tile: Tf32PortableTile,
    pub stages: Tf32PortableStages,
}

impl Tf32PortableRoute {
    pub fn validate(self) -> Result<(), String> {
        if matches!(
            (self.tile, self.stages),
            (
                Tf32PortableTile::M128N64 | Tf32PortableTile::M64N64,
                Tf32PortableStages::S2 | Tf32PortableStages::S3
            ) | (Tf32PortableTile::M16N32, Tf32PortableStages::S4)
        ) {
            Ok(())
        } else {
            Err(format!("illegal portable TF32 route {self:?}"))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tf32Sm90aRoute {
    pub schedule: Sm90aWarpgroupSchedule,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tf32Sm100Route {
    pub tile: Sm100Tile,
    pub stages: Sm100Stages,
    pub schedule: Sm100Schedule,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tf32Sm120Tile {
    M128N64,
    M64N128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tf32Sm120Route {
    pub tile: Tf32Sm120Tile,
    pub stages: Sm120Stages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tf32PhysicalRoute {
    MmaTf32RnaV1(Tf32PortableRoute),
    Sm90aWgmmaTf32TmaV1(Tf32Sm90aRoute),
    Sm100Tcgen05Tf32TmaV1(Tf32Sm100Route),
    Sm120TmaMmaTf32RnaV1(Tf32Sm120Route),
}

impl Tf32PhysicalRoute {
    pub const fn module_kind(self) -> ModuleKind {
        match self {
            Self::MmaTf32RnaV1(_) => ModuleKind::TriadSm80,
            Self::Sm90aWgmmaTf32TmaV1(_) => ModuleKind::TriadSm90a,
            Self::Sm100Tcgen05Tf32TmaV1(_) => ModuleKind::TriadSm100,
            Self::Sm120TmaMmaTf32RnaV1(_) => ModuleKind::TriadSm120,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F32TriadSelection {
    ScalarFmaV1,
    Tf32(Tf32PhysicalRoute),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tf32QualifiedModule {
    pub module_kind: ModuleKind,
    pub target: CudaTarget,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
    pub device_caps: DeviceCaps,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F32TriadAvailability {
    pub portable: Option<Tf32QualifiedModule>,
    pub specialized: Option<Tf32QualifiedModule>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tf32KernelSpec {
    pub op: ResolvedGemmOp,
    pub route: Tf32PhysicalRoute,
    pub symbol: &'static str,
    pub module_kind: ModuleKind,
    pub instruction_family: ResolvedInstructionFamily,
    pub instruction_shape: ResolvedInstructionShape,
    pub operand_conversion: ResolvedOperandConversion,
    pub tile: (u32, u32),
    pub bk: u32,
    pub stages: u8,
    pub threads: u32,
    pub dynamic_shared_bytes: u32,
    pub tensor_map_revision: u16,
    pub schedule_revision: u16,
}

macro_rules! portable_tf32_specs {
    ($op:expr, $op_name:literal) => {
        [
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N64,
                    stages: Tf32PortableStages::S2,
                }),
                symbol: concat!("sgemm_bi_", $op_name, "_sm80_mma_tf32_v1_m128n64_bk32_s2"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32V1,
                tile: (128, 64),
                bk: 32,
                stages: 2,
                threads: 256,
                dynamic_shared_bytes: 55_296,
                tensor_map_revision: 0,
                schedule_revision: TF32_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N64,
                    stages: Tf32PortableStages::S3,
                }),
                symbol: concat!("sgemm_bi_", $op_name, "_sm80_mma_tf32_v1_m128n64_bk32_s3"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32V1,
                tile: (128, 64),
                bk: 32,
                stages: 3,
                threads: 256,
                dynamic_shared_bytes: 82_944,
                tensor_map_revision: 0,
                schedule_revision: TF32_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M64N64,
                    stages: Tf32PortableStages::S2,
                }),
                symbol: concat!("sgemm_bi_", $op_name, "_sm80_mma_tf32_v1_m64n64_bk32_s2"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32V1,
                tile: (64, 64),
                bk: 32,
                stages: 2,
                threads: 128,
                dynamic_shared_bytes: 36_864,
                tensor_map_revision: 0,
                schedule_revision: TF32_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M64N64,
                    stages: Tf32PortableStages::S3,
                }),
                symbol: concat!("sgemm_bi_", $op_name, "_sm80_mma_tf32_v1_m64n64_bk32_s3"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32V1,
                tile: (64, 64),
                bk: 32,
                stages: 3,
                threads: 128,
                dynamic_shared_bytes: 55_296,
                tensor_map_revision: 0,
                schedule_revision: TF32_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
                symbol: concat!("sgemm_bi_", $op_name, "_sm80_mma_tf32_v1_m16n32_bk32_s4"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32V1,
                tile: (16, 32),
                bk: 32,
                stages: 4,
                threads: 128,
                dynamic_shared_bytes: 29_696,
                tensor_map_revision: 0,
                schedule_revision: TF32_SCHEDULE_REVISION,
            },
        ]
    };
}

macro_rules! sm90a_tf32_specs {
    ($op:expr, $op_name:literal) => {
        [
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(Tf32Sm90aRoute {
                    schedule: Sm90aWarpgroupSchedule::Wg1,
                }),
                symbol: concat!(
                    "sgemm_bi_",
                    $op_name,
                    "_sm90a_wgmma_tf32_v1_m64n128_bk32_s3_wg1"
                ),
                module_kind: ModuleKind::TriadSm90a,
                instruction_family: ResolvedInstructionFamily::Wgmma,
                instruction_shape: ResolvedInstructionShape {
                    m: 64,
                    n: 128,
                    k: 8,
                },
                operand_conversion: ResolvedOperandConversion::TensorMapTfloat32V1,
                tile: (64, 128),
                bk: 32,
                stages: 3,
                threads: 128,
                dynamic_shared_bytes: 73_984,
                tensor_map_revision: TF32_TENSOR_MAP_REVISION,
                schedule_revision: TF32_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(Tf32Sm90aRoute {
                    schedule: Sm90aWarpgroupSchedule::Wg2,
                }),
                symbol: concat!(
                    "sgemm_bi_",
                    $op_name,
                    "_sm90a_wgmma_tf32_v1_m64n128_bk32_s3_wg2"
                ),
                module_kind: ModuleKind::TriadSm90a,
                instruction_family: ResolvedInstructionFamily::Wgmma,
                instruction_shape: ResolvedInstructionShape {
                    m: 64,
                    n: 128,
                    k: 8,
                },
                operand_conversion: ResolvedOperandConversion::TensorMapTfloat32V1,
                tile: (64, 128),
                bk: 32,
                stages: 3,
                threads: 256,
                dynamic_shared_bytes: 73_984,
                tensor_map_revision: TF32_TENSOR_MAP_REVISION,
                schedule_revision: TF32_SCHEDULE_REVISION,
            },
        ]
    };
}

macro_rules! sm100_tf32_spec {
    ($op:expr, $op_name:literal, $tile:expr, $n:literal, $stage:expr, $s:literal, $schedule:expr, $schedule_name:literal, $threads:literal, $shared:literal) => {
        Tf32KernelSpec {
            op: $op,
            route: Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(Tf32Sm100Route {
                tile: $tile,
                stages: $stage,
                schedule: $schedule,
            }),
            symbol: concat!(
                "sgemm_bi_",
                $op_name,
                "_sm100_tcgen_tf32_v1_m128n",
                $n,
                "_bk32_s",
                $s,
                "_",
                $schedule_name
            ),
            module_kind: ModuleKind::TriadSm100,
            instruction_family: ResolvedInstructionFamily::Tcgen05,
            instruction_shape: ResolvedInstructionShape {
                m: 128,
                n: $n,
                k: 8,
            },
            operand_conversion: ResolvedOperandConversion::TensorMapTfloat32V1,
            tile: (128, $n),
            bk: 32,
            stages: $s,
            threads: $threads,
            dynamic_shared_bytes: $shared,
            tensor_map_revision: TF32_TENSOR_MAP_REVISION,
            schedule_revision: TF32_SCHEDULE_REVISION,
        }
    };
}

macro_rules! sm100_tf32_specs {
    ($op:expr, $op_name:literal) => {
        [
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N64,
                64,
                Sm100Stages::S2,
                2,
                Sm100Schedule::C4,
                "c4",
                128,
                49_408
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N64,
                64,
                Sm100Stages::S2,
                2,
                Sm100Schedule::P8,
                "p8",
                256,
                49_408
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N64,
                64,
                Sm100Stages::S3,
                3,
                Sm100Schedule::C4,
                "c4",
                128,
                73_984
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N64,
                64,
                Sm100Stages::S3,
                3,
                Sm100Schedule::P8,
                "p8",
                256,
                73_984
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N64,
                64,
                Sm100Stages::S4,
                4,
                Sm100Schedule::C4,
                "c4",
                128,
                98_560
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N64,
                64,
                Sm100Stages::S4,
                4,
                Sm100Schedule::P8,
                "p8",
                256,
                98_560
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N128,
                128,
                Sm100Stages::S2,
                2,
                Sm100Schedule::C4,
                "c4",
                128,
                65_792
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N128,
                128,
                Sm100Stages::S2,
                2,
                Sm100Schedule::P8,
                "p8",
                256,
                65_792
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N128,
                128,
                Sm100Stages::S3,
                3,
                Sm100Schedule::C4,
                "c4",
                128,
                98_560
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N128,
                128,
                Sm100Stages::S3,
                3,
                Sm100Schedule::P8,
                "p8",
                256,
                98_560
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N128,
                128,
                Sm100Stages::S4,
                4,
                Sm100Schedule::C4,
                "c4",
                128,
                131_328
            ),
            sm100_tf32_spec!(
                $op,
                $op_name,
                Sm100Tile::M128N128,
                128,
                Sm100Stages::S4,
                4,
                Sm100Schedule::P8,
                "p8",
                256,
                131_328
            ),
        ]
    };
}

macro_rules! sm120_tf32_spec {
    ($op:expr, $op_name:literal, $tile:expr, $tile_name:literal, $stage:expr, $s:literal, $shared:literal) => {
        Tf32KernelSpec {
            op: $op,
            route: Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route {
                tile: $tile,
                stages: $stage,
            }),
            symbol: concat!(
                "sgemm_bi_",
                $op_name,
                "_sm120_tma_mma_tf32_v1_",
                $tile_name,
                "_bk32_s",
                $s
            ),
            module_kind: ModuleKind::TriadSm120,
            instruction_family: ResolvedInstructionFamily::MmaSync,
            instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
            operand_conversion: ResolvedOperandConversion::TensorMapUint32ThenCvtRnaTf32F32V1,
            tile: match $tile {
                Tf32Sm120Tile::M128N64 => (128, 64),
                Tf32Sm120Tile::M64N128 => (64, 128),
            },
            bk: 32,
            stages: $s,
            threads: 256,
            dynamic_shared_bytes: $shared,
            tensor_map_revision: TF32_TENSOR_MAP_REVISION,
            schedule_revision: TF32_SCHEDULE_REVISION,
        }
    };
}

macro_rules! sm120_tf32_specs {
    ($op:expr, $op_name:literal) => {
        [
            sm120_tf32_spec!(
                $op,
                $op_name,
                Tf32Sm120Tile::M128N64,
                "m128n64",
                Sm120Stages::S2,
                2,
                49_280
            ),
            sm120_tf32_spec!(
                $op,
                $op_name,
                Tf32Sm120Tile::M128N64,
                "m128n64",
                Sm120Stages::S3,
                3,
                73_856
            ),
            sm120_tf32_spec!(
                $op,
                $op_name,
                Tf32Sm120Tile::M64N128,
                "m64n128",
                Sm120Stages::S2,
                2,
                49_280
            ),
            sm120_tf32_spec!(
                $op,
                $op_name,
                Tf32Sm120Tile::M64N128,
                "m64n128",
                Sm120Stages::S3,
                3,
                73_856
            ),
        ]
    };
}

const SM80_TF32_NN: [Tf32KernelSpec; 5] = portable_tf32_specs!(ResolvedGemmOp::Nn, "nn");
const SM80_TF32_TN: [Tf32KernelSpec; 5] = portable_tf32_specs!(ResolvedGemmOp::Tn, "tn");
const SM80_TF32_NT: [Tf32KernelSpec; 5] = portable_tf32_specs!(ResolvedGemmOp::Nt, "nt");
pub const SM80_TF32_ROUTE_SPECS: [Tf32KernelSpec; 15] = [
    SM80_TF32_NN[0],
    SM80_TF32_NN[1],
    SM80_TF32_NN[2],
    SM80_TF32_NN[3],
    SM80_TF32_NN[4],
    SM80_TF32_TN[0],
    SM80_TF32_TN[1],
    SM80_TF32_TN[2],
    SM80_TF32_TN[3],
    SM80_TF32_TN[4],
    SM80_TF32_NT[0],
    SM80_TF32_NT[1],
    SM80_TF32_NT[2],
    SM80_TF32_NT[3],
    SM80_TF32_NT[4],
];

const SM90A_TF32_NN: [Tf32KernelSpec; 2] = sm90a_tf32_specs!(ResolvedGemmOp::Nn, "nn");
const SM90A_TF32_TN: [Tf32KernelSpec; 2] = sm90a_tf32_specs!(ResolvedGemmOp::Tn, "tn");
const SM90A_TF32_NT: [Tf32KernelSpec; 2] = sm90a_tf32_specs!(ResolvedGemmOp::Nt, "nt");
pub const SM90A_TF32_ROUTE_SPECS: [Tf32KernelSpec; 6] = [
    SM90A_TF32_NN[0],
    SM90A_TF32_NN[1],
    SM90A_TF32_TN[0],
    SM90A_TF32_TN[1],
    SM90A_TF32_NT[0],
    SM90A_TF32_NT[1],
];

const SM100_TF32_NN: [Tf32KernelSpec; 12] = sm100_tf32_specs!(ResolvedGemmOp::Nn, "nn");
const SM100_TF32_TN: [Tf32KernelSpec; 12] = sm100_tf32_specs!(ResolvedGemmOp::Tn, "tn");
const SM100_TF32_NT: [Tf32KernelSpec; 12] = sm100_tf32_specs!(ResolvedGemmOp::Nt, "nt");
pub const SM100_TF32_ROUTE_SPECS: [Tf32KernelSpec; 36] = [
    SM100_TF32_NN[0],
    SM100_TF32_NN[1],
    SM100_TF32_NN[2],
    SM100_TF32_NN[3],
    SM100_TF32_NN[4],
    SM100_TF32_NN[5],
    SM100_TF32_NN[6],
    SM100_TF32_NN[7],
    SM100_TF32_NN[8],
    SM100_TF32_NN[9],
    SM100_TF32_NN[10],
    SM100_TF32_NN[11],
    SM100_TF32_TN[0],
    SM100_TF32_TN[1],
    SM100_TF32_TN[2],
    SM100_TF32_TN[3],
    SM100_TF32_TN[4],
    SM100_TF32_TN[5],
    SM100_TF32_TN[6],
    SM100_TF32_TN[7],
    SM100_TF32_TN[8],
    SM100_TF32_TN[9],
    SM100_TF32_TN[10],
    SM100_TF32_TN[11],
    SM100_TF32_NT[0],
    SM100_TF32_NT[1],
    SM100_TF32_NT[2],
    SM100_TF32_NT[3],
    SM100_TF32_NT[4],
    SM100_TF32_NT[5],
    SM100_TF32_NT[6],
    SM100_TF32_NT[7],
    SM100_TF32_NT[8],
    SM100_TF32_NT[9],
    SM100_TF32_NT[10],
    SM100_TF32_NT[11],
];

const SM120_TF32_NN: [Tf32KernelSpec; 4] = sm120_tf32_specs!(ResolvedGemmOp::Nn, "nn");
const SM120_TF32_TN: [Tf32KernelSpec; 4] = sm120_tf32_specs!(ResolvedGemmOp::Tn, "tn");
const SM120_TF32_NT: [Tf32KernelSpec; 4] = sm120_tf32_specs!(ResolvedGemmOp::Nt, "nt");
pub const SM120_TF32_ROUTE_SPECS: [Tf32KernelSpec; 12] = [
    SM120_TF32_NN[0],
    SM120_TF32_NN[1],
    SM120_TF32_NN[2],
    SM120_TF32_NN[3],
    SM120_TF32_TN[0],
    SM120_TF32_TN[1],
    SM120_TF32_TN[2],
    SM120_TF32_TN[3],
    SM120_TF32_NT[0],
    SM120_TF32_NT[1],
    SM120_TF32_NT[2],
    SM120_TF32_NT[3],
];

pub fn tf32_route_specs(module_kind: ModuleKind) -> &'static [Tf32KernelSpec] {
    match module_kind {
        ModuleKind::TriadSm80 => &SM80_TF32_ROUTE_SPECS,
        ModuleKind::TriadSm90a => &SM90A_TF32_ROUTE_SPECS,
        ModuleKind::TriadSm100 => &SM100_TF32_ROUTE_SPECS,
        ModuleKind::TriadSm120 => &SM120_TF32_ROUTE_SPECS,
        ModuleKind::Fixed | ModuleKind::TriadScalar | ModuleKind::Mamba3Combined => &[],
    }
}

pub fn tf32_module_symbols(module_kind: ModuleKind) -> impl ExactSizeIterator<Item = &'static str> {
    tf32_route_specs(module_kind).iter().map(|spec| spec.symbol)
}

pub fn tf32_kernel_spec(
    op: ResolvedGemmOp,
    route: Tf32PhysicalRoute,
) -> Result<&'static Tf32KernelSpec, String> {
    if let Tf32PhysicalRoute::MmaTf32RnaV1(portable) = route {
        portable.validate()?;
    }
    let mut matches = tf32_route_specs(route.module_kind())
        .iter()
        .filter(|spec| spec.op == op && spec.route == route);
    let Some(spec) = matches.next() else {
        return Err(format!("no TF32 kernel matches {op:?}/{route:?}"));
    };
    if matches.next().is_some() {
        Err(format!("duplicate TF32 kernels match {op:?}/{route:?}"))
    } else {
        Ok(spec)
    }
}

pub const SM90A_DYNAMIC_SHARED_BYTES: u32 = 73_984;
pub const SM90A_TILE: (u32, u32, u32) = (64, 128, 64);
pub const SM90A_STAGES: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm90aOp {
    Nn,
    Tn,
    Nt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm90aWarpgroupSchedule {
    Wg1,
    Wg2,
}

impl Sm90aWarpgroupSchedule {
    pub const fn threads(self) -> u32 {
        match self {
            Self::Wg1 => 128,
            Self::Wg2 => 256,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm90aShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
}

impl Sm90aShape {
    pub fn contiguous(op: Sm90aOp, dims: (usize, usize, usize)) -> Self {
        let (m, k, n) = dims;
        match op {
            Sm90aOp::Nn | Sm90aOp::Tn => Self {
                m,
                k,
                n,
                lda: k,
                ldb: n,
                ldc: n,
            },
            Sm90aOp::Nt => Self {
                m,
                k,
                n,
                lda: n,
                ldb: n,
                ldc: k,
            },
        }
    }

    pub fn validate(self, op: Sm90aOp) -> Result<(), String> {
        for (value, name) in [(self.m, "M"), (self.k, "K"), (self.n, "N")] {
            if value == 0 {
                return Err(invalid_gemm_dimensions(format!("{name} must be positive")));
            }
            checked_i32(value, name)?;
        }
        let (a_width, b_width, c_width) = match op {
            Sm90aOp::Nn | Sm90aOp::Tn => (self.k, self.n, self.n),
            Sm90aOp::Nt => (self.n, self.n, self.k),
        };
        for (stride, width, name) in [
            (self.lda, a_width, "lda"),
            (self.ldb, b_width, "ldb"),
            (self.ldc, c_width, "ldc"),
        ] {
            if stride < width {
                return Err(invalid_gemm_dimensions(format!(
                    "{name}={stride} is smaller than the physical width {width}"
                )));
            }
            checked_i32(stride, name)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm90aForcedRoute {
    pub op: Sm90aOp,
    pub dtype: WeightDtype,
    pub schedule: Sm90aWarpgroupSchedule,
    pub shape: Sm90aShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm90aNumericContract {
    WgmmaV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm90aRouteIdentity {
    pub numeric_contract: Sm90aNumericContract,
    pub op: Sm90aOp,
    pub dtype: WeightDtype,
    pub schedule: Sm90aWarpgroupSchedule,
    pub shape: Sm90aShape,
    pub tile: (u32, u32, u32),
    pub stages: u8,
    pub cluster: (u8, u8, u8),
    pub symbol: &'static str,
    pub module_kind: ModuleKind,
    pub exact_target: &'static str,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
    pub tensor_maps_digest: Sha256Digest,
    pub resources_digest: Sha256Digest,
    pub tuning_revision: u16,
}

impl Sm90aRouteIdentity {
    pub fn ensure_current(self, live: Self, prefix: &str) -> Result<(), String> {
        if self == live {
            Ok(())
        } else {
            Err(format!(
                "{prefix}: SM90a route changed since capture; re-capture before replay"
            ))
        }
    }
}

impl Sm90aForcedRoute {
    pub fn symbol(self) -> &'static str {
        match (self.op, self.dtype, self.schedule) {
            (Sm90aOp::Nn, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg1) => {
                "sgemm_bi_nn_sm90a_wgmma_wg1_bf16"
            }
            (Sm90aOp::Nn, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg1) => {
                "sgemm_bi_nn_sm90a_wgmma_wg1_f16"
            }
            (Sm90aOp::Tn, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg1) => {
                "sgemm_bi_tn_sm90a_wgmma_wg1_bf16"
            }
            (Sm90aOp::Tn, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg1) => {
                "sgemm_bi_tn_sm90a_wgmma_wg1_f16"
            }
            (Sm90aOp::Nt, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg1) => {
                "sgemm_bi_nt_sm90a_wgmma_wg1_bf16"
            }
            (Sm90aOp::Nt, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg1) => {
                "sgemm_bi_nt_sm90a_wgmma_wg1_f16"
            }
            (Sm90aOp::Nn, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg2) => {
                "sgemm_bi_nn_sm90a_wgmma_wg2_bf16"
            }
            (Sm90aOp::Nn, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg2) => {
                "sgemm_bi_nn_sm90a_wgmma_wg2_f16"
            }
            (Sm90aOp::Tn, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg2) => {
                "sgemm_bi_tn_sm90a_wgmma_wg2_bf16"
            }
            (Sm90aOp::Tn, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg2) => {
                "sgemm_bi_tn_sm90a_wgmma_wg2_f16"
            }
            (Sm90aOp::Nt, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg2) => {
                "sgemm_bi_nt_sm90a_wgmma_wg2_bf16"
            }
            (Sm90aOp::Nt, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg2) => {
                "sgemm_bi_nt_sm90a_wgmma_wg2_f16"
            }
            (_, WeightDtype::F32, _) => unreachable!("f32 has no SM90a WGMMA route"),
        }
    }
}

pub const SM100_TENSOR_MAP_REVISION: u16 = 1;
pub const SM100_TUNING_REVISION: u16 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm100Op {
    Nn,
    Tn,
    Nt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm100Tile {
    M128N64,
    M128N128,
}

impl Sm100Tile {
    pub const fn output_rows(self) -> u32 {
        128
    }

    pub const fn output_columns(self) -> u32 {
        match self {
            Self::M128N64 => 64,
            Self::M128N128 => 128,
        }
    }

    pub const fn tmem_columns(self) -> u32 {
        self.output_columns()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm100Stages {
    S2,
    S3,
    S4,
}

impl Sm100Stages {
    pub const fn count(self) -> u8 {
        match self {
            Self::S2 => 2,
            Self::S3 => 3,
            Self::S4 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm100Schedule {
    C4,
    P8,
}

impl Sm100Schedule {
    pub const fn threads(self) -> u32 {
        match self {
            Self::C4 => 128,
            Self::P8 => 256,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm100PhysicalRoute {
    pub tile: Sm100Tile,
    pub stages: Sm100Stages,
    pub schedule: Sm100Schedule,
}

impl Sm100PhysicalRoute {
    pub const fn dynamic_shared_bytes(self) -> u32 {
        match (self.tile, self.stages) {
            (Sm100Tile::M128N64, Sm100Stages::S2) => 49_408,
            (Sm100Tile::M128N64, Sm100Stages::S3) => 73_984,
            (Sm100Tile::M128N64, Sm100Stages::S4) => 98_560,
            (Sm100Tile::M128N128, Sm100Stages::S2) => 65_792,
            (Sm100Tile::M128N128, Sm100Stages::S3) => 98_560,
            (Sm100Tile::M128N128, Sm100Stages::S4) => 131_328,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm100Shape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
}

impl Sm100Shape {
    pub fn contiguous(op: Sm100Op, dims: (usize, usize, usize)) -> Self {
        let (m, k, n) = dims;
        match op {
            Sm100Op::Nn | Sm100Op::Tn => Self {
                m,
                k,
                n,
                lda: k,
                ldb: n,
                ldc: n,
            },
            Sm100Op::Nt => Self {
                m,
                k,
                n,
                lda: n,
                ldb: n,
                ldc: k,
            },
        }
    }

    pub fn validate(self, op: Sm100Op) -> Result<(), String> {
        for (value, name) in [(self.m, "M"), (self.k, "K"), (self.n, "N")] {
            if value == 0 {
                return Err(invalid_gemm_dimensions(format!("{name} must be positive")));
            }
            checked_i32(value, name)?;
        }
        let (a_width, b_width, c_width) = match op {
            Sm100Op::Nn | Sm100Op::Tn => (self.k, self.n, self.n),
            Sm100Op::Nt => (self.n, self.n, self.k),
        };
        for (stride, width, name) in [
            (self.lda, a_width, "lda"),
            (self.ldb, b_width, "ldb"),
            (self.ldc, c_width, "ldc"),
        ] {
            if stride < width {
                return Err(invalid_gemm_dimensions(format!(
                    "{name}={stride} is smaller than the physical width {width}"
                )));
            }
            checked_i32(stride, name)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm100ForcedRoute {
    pub op: Sm100Op,
    pub dtype: WeightDtype,
    pub physical: Sm100PhysicalRoute,
    pub shape: Sm100Shape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm100TargetKind {
    Family,
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm100TargetCandidate {
    pub device_cc: (i32, i32),
    pub nvrtc_arch: &'static str,
    pub ptx_target: &'static str,
    pub kind: Sm100TargetKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm100KernelSpec {
    pub op: Sm100Op,
    pub dtype: WeightDtype,
    pub physical: Sm100PhysicalRoute,
    pub symbol: &'static str,
    pub threads: u32,
    pub dynamic_shared_bytes: u32,
    pub bk: u32,
}

macro_rules! sm100_spec {
    ($op:expr, $dtype:expr, $tile:expr, $stages:expr, $schedule:expr, $symbol:literal) => {{
        let physical = Sm100PhysicalRoute {
            tile: $tile,
            stages: $stages,
            schedule: $schedule,
        };
        Sm100KernelSpec {
            op: $op,
            dtype: $dtype,
            physical,
            symbol: $symbol,
            threads: physical.schedule.threads(),
            dynamic_shared_bytes: physical.dynamic_shared_bytes(),
            bk: 64,
        }
    }};
}

pub const SM100_KERNEL_SPECS: [Sm100KernelSpec; 72] = [
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n64_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_nn_sm100_tcgen_m128n128_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n64_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_tn_sm100_tcgen_m128n128_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n64_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "sgemm_bi_nt_sm100_tcgen_m128n128_bk64_s4_p8_f16"
    ),
];

impl Sm100ForcedRoute {
    pub fn kernel_spec(self) -> Result<&'static Sm100KernelSpec, String> {
        SM100_KERNEL_SPECS
            .iter()
            .find(|spec| {
                spec.op == self.op && spec.dtype == self.dtype && spec.physical == self.physical
            })
            .ok_or_else(|| "no SM100 TCGEN kernel matches the forced route".to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm100NumericContract {
    Tcgen05F32V1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm100RouteIdentity {
    pub numeric_contract: Sm100NumericContract,
    pub op: Sm100Op,
    pub dtype: WeightDtype,
    pub physical: Sm100PhysicalRoute,
    pub shape: Sm100Shape,
    pub symbol: &'static str,
    pub module_kind: ModuleKind,
    pub target: Sm100TargetCandidate,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
    pub tensor_map_revision: u16,
    pub tensor_maps_digest: Sha256Digest,
    pub resources_digest: Sha256Digest,
    pub tuning_revision: u16,
}

impl Sm100RouteIdentity {
    pub fn ensure_current(self, live: Self, prefix: &str) -> Result<(), String> {
        if self == live {
            Ok(())
        } else {
            Err(format!(
                "{prefix}: SM100 route changed since preparation; re-capture before replay"
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm90aMapRequest {
    pub op: Sm90aOp,
    pub dtype: WeightDtype,
    pub a_ptr: CUptr,
    pub b_ptr: CUptr,
    pub shape: Sm90aShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Sm90aTensorMapKey {
    base: CUptr,
    global_dimensions: [u64; 2],
    outer_byte_stride: u64,
    box_dimensions: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Sm90aAllocationIdentity {
    context_handle: usize,
    allocation_base: CUptr,
    allocation_bytes: u64,
    offset_bytes: u64,
    required_bytes: u64,
    buffer_id: u64,
}

impl Sm90aAllocationIdentity {
    fn query(
        pointer: CUptr,
        required_bytes: u64,
        expected_context: usize,
        backend: &str,
        name: &str,
    ) -> Result<Self, String> {
        if pointer == 0 || required_bytes == 0 {
            return Err(format!("{backend} {name} allocation must be non-empty"));
        }
        let mut context: sys::CUcontext = std::ptr::null_mut();
        let mut buffer_id = 0_u64;
        let mut allocation_base = 0_u64;
        let mut allocation_bytes = 0_usize;
        let mut attributes = [
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_BUFFER_ID,
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_SIZE,
        ];
        let mut outputs = [
            std::ptr::from_mut(&mut context).cast(),
            std::ptr::from_mut(&mut buffer_id).cast(),
            std::ptr::from_mut(&mut allocation_base).cast(),
            std::ptr::from_mut(&mut allocation_bytes).cast(),
        ];
        let result = unsafe {
            sys::cuPointerGetAttributes(
                attributes.len() as u32,
                attributes.as_mut_ptr(),
                outputs.as_mut_ptr(),
                pointer,
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "query {backend} {name} allocation identity: {result:?}"
            ));
        }
        let context_handle = context as usize;
        if context_handle != expected_context {
            return Err(format!(
                "{backend} {name} allocation belongs to a different CUDA context"
            ));
        }
        let allocation_bytes = u64::try_from(allocation_bytes)
            .map_err(|_| format!("{backend} {name} allocation size exceeds u64::MAX"))?;
        let offset_bytes = pointer
            .checked_sub(allocation_base)
            .ok_or_else(|| format!("{backend} {name} pointer precedes its allocation"))?;
        let end = offset_bytes
            .checked_add(required_bytes)
            .ok_or_else(|| format!("{backend} {name} byte range overflows u64"))?;
        if end > allocation_bytes {
            return Err(format!(
                "{backend} {name} requires {required_bytes} bytes at offset {offset_bytes}, allocation has {allocation_bytes} bytes"
            ));
        }
        Ok(Self {
            context_handle,
            allocation_base,
            allocation_bytes,
            offset_bytes,
            required_bytes,
            buffer_id,
        })
    }

    fn append_digest(self, digest: FramedSha256) -> FramedSha256 {
        digest
            .required(
                b"context-handle",
                &(self.context_handle as u64).to_le_bytes(),
            )
            .required(b"allocation-base", &self.allocation_base.to_le_bytes())
            .required(b"allocation-bytes", &self.allocation_bytes.to_le_bytes())
            .required(b"offset-bytes", &self.offset_bytes.to_le_bytes())
            .required(b"required-bytes", &self.required_bytes.to_le_bytes())
            .required(b"buffer-id", &self.buffer_id.to_le_bytes())
    }
}

impl Sm90aTensorMapKey {
    fn validate(self, backend: &str) -> Result<(), String> {
        if self.base == 0 || !self.base.is_multiple_of(16) {
            return Err(format!(
                "{backend} tensor-map base must be non-null and 16-byte aligned"
            ));
        }
        if self.global_dimensions.contains(&0) {
            return Err(format!("{backend} tensor-map dimensions must be positive"));
        }
        if self.outer_byte_stride == 0
            || !self.outer_byte_stride.is_multiple_of(16)
            || self.outer_byte_stride >= (1_u64 << 40)
        {
            return Err(format!(
                "{backend} tensor-map outer byte stride must be a positive multiple of 16 below 2^40"
            ));
        }
        let row_bytes = self.global_dimensions[0]
            .checked_mul(2)
            .ok_or_else(|| format!("{backend} tensor-map row width overflows u64"))?;
        if self.outer_byte_stride < row_bytes {
            return Err(format!(
                "{backend} tensor-map outer stride is smaller than its inner dimension"
            ));
        }
        if self.box_dimensions.contains(&0) || self.box_dimensions.iter().any(|&dim| dim > 256) {
            return Err(format!(
                "{backend} tensor-map box dimensions must be in 1..=256"
            ));
        }
        let inner_bytes = self.box_dimensions[0]
            .checked_mul(2)
            .ok_or_else(|| format!("{backend} tensor-map box width overflows u32"))?;
        if inner_bytes > 128 || !inner_bytes.is_multiple_of(16) {
            return Err(format!(
                "{backend} SW128 inner box span must be a multiple of 16 bytes and at most 128 bytes"
            ));
        }
        Ok(())
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm90aTensorMap(sys::CUtensorMap);

unsafe impl DeviceRepr for Sm90aTensorMap {}

const _: () = {
    assert!(std::mem::size_of::<Sm90aTensorMap>() == std::mem::size_of::<sys::CUtensorMap>());
    assert!(std::mem::align_of::<Sm90aTensorMap>() == std::mem::align_of::<sys::CUtensorMap>());
};

impl Sm90aTensorMap {
    fn encode(key: Sm90aTensorMapKey, backend: &str) -> Result<Self, String> {
        key.validate(backend)?;
        let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
        let element_strides = [1_u32, 1_u32];
        let global_strides = [key.outer_byte_stride];
        unsafe {
            sys::cuTensorMapEncodeTiled(
                raw.as_mut_ptr(),
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
                2,
                key.base as usize as *mut std::ffi::c_void,
                key.global_dimensions.as_ptr(),
                global_strides.as_ptr(),
                key.box_dimensions.as_ptr(),
                element_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
            .result()
            .map_err(|error| format!("cuTensorMapEncodeTiled failed: {error:?}"))?;
            Ok(Self(raw.assume_init()))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm90aPreparedTensorMaps {
    pub(super) a: Sm90aTensorMap,
    pub(super) b: Sm90aTensorMap,
    pub(super) keys: [Sm90aTensorMapKey; 2],
    pub(super) request: Sm90aMapRequest,
    pub(super) binding: Sm90aMapBinding,
    pub(super) allocations: [Sm90aAllocationIdentity; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sm90aMapBinding {
    pub context_handle: usize,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
}

impl Sm90aPreparedTensorMaps {
    pub fn identity_digest(&self) -> Sha256Digest {
        let dtype = match self.request.dtype {
            WeightDtype::F32 => 0,
            WeightDtype::F16 => 1,
            WeightDtype::Bf16 => 2,
        };
        let op = match self.request.op {
            Sm90aOp::Nn => 0,
            Sm90aOp::Tn => 1,
            Sm90aOp::Nt => 2,
        };
        let mut digest = FramedSha256::new(b"sm90a-tensor-map-pair.v2")
            .required(b"op", &[op])
            .required(b"dtype", &[dtype]);
        for key in self.keys {
            digest = digest
                .required(b"base", &key.base.to_le_bytes())
                .required(b"global-0", &key.global_dimensions[0].to_le_bytes())
                .required(b"global-1", &key.global_dimensions[1].to_le_bytes())
                .required(b"outer-stride", &key.outer_byte_stride.to_le_bytes())
                .required(b"box-0", &key.box_dimensions[0].to_le_bytes())
                .required(b"box-1", &key.box_dimensions[1].to_le_bytes());
        }
        for (index, map) in [self.a, self.b].into_iter().enumerate() {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    std::ptr::from_ref(&map.0).cast::<u8>(),
                    std::mem::size_of::<sys::CUtensorMap>(),
                )
            };
            digest = digest
                .required(b"descriptor-index", &(index as u64).to_le_bytes())
                .required(b"encoded-descriptor", bytes);
        }
        for allocation in self.allocations {
            digest = allocation.append_digest(digest);
        }
        digest.finish()
    }

    pub(super) fn matches_binding(&self, binding: Sm90aMapBinding) -> bool {
        self.binding == binding
    }

    pub(super) fn validate_live_allocations(&self) -> Result<(), String> {
        let live = sm90a_allocation_identities(self.keys, self.binding.context_handle)?;
        if live != self.allocations {
            return Err(
                "SM90a input allocation identity changed since tensor-map preparation".into(),
            );
        }
        Ok(())
    }
}

pub(super) fn sm90a_tensor_map_keys(
    request: Sm90aMapRequest,
) -> Result<[Sm90aTensorMapKey; 2], String> {
    request.shape.validate(request.op)?;
    if !matches!(request.dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM90a WGMMA tensor maps require bf16 or f16 operands".into());
    }
    let shape = request.shape;
    let stride_bytes = |stride: usize, name: &str| {
        u64::try_from(
            stride
                .checked_mul(2)
                .ok_or_else(|| format!("{name} byte stride overflows usize"))?,
        )
        .map_err(|_| format!("{name} byte stride exceeds u64::MAX"))
    };
    let (a_dimensions, b_dimensions, a_box, b_box) = match request.op {
        Sm90aOp::Nn => (
            [shape.k as u64, shape.m as u64],
            [shape.n as u64, shape.k as u64],
            [64, 64],
            [64, 64],
        ),
        Sm90aOp::Tn => (
            [shape.k as u64, shape.m as u64],
            [shape.n as u64, shape.m as u64],
            [64, 64],
            [64, 64],
        ),
        Sm90aOp::Nt => (
            [shape.n as u64, shape.m as u64],
            [shape.n as u64, shape.k as u64],
            [64, 64],
            [64, 128],
        ),
    };
    let keys = [
        Sm90aTensorMapKey {
            base: request.a_ptr,
            global_dimensions: a_dimensions,
            outer_byte_stride: stride_bytes(shape.lda, "A")?,
            box_dimensions: a_box,
        },
        Sm90aTensorMapKey {
            base: request.b_ptr,
            global_dimensions: b_dimensions,
            outer_byte_stride: stride_bytes(shape.ldb, "B")?,
            box_dimensions: b_box,
        },
    ];
    for key in keys {
        key.validate("SM90a")?;
    }
    Ok(keys)
}

pub fn validate_sm90a_map_request(request: Sm90aMapRequest) -> Result<(), String> {
    sm90a_tensor_map_keys(request).map(|_| ())
}

pub(super) fn encode_sm90a_tensor_maps(
    keys: [Sm90aTensorMapKey; 2],
    request: Sm90aMapRequest,
    binding: Sm90aMapBinding,
    allocations: [Sm90aAllocationIdentity; 2],
) -> Result<Sm90aPreparedTensorMaps, String> {
    Ok(Sm90aPreparedTensorMaps {
        a: Sm90aTensorMap::encode(keys[0], "SM90a")?,
        b: Sm90aTensorMap::encode(keys[1], "SM90a")?,
        keys,
        request,
        binding,
        allocations,
    })
}

pub(super) fn sm90a_allocation_identities(
    keys: [Sm90aTensorMapKey; 2],
    context_handle: usize,
) -> Result<[Sm90aAllocationIdentity; 2], String> {
    let required = |key: Sm90aTensorMapKey| {
        let rows = key.global_dimensions[1];
        let row_bytes = key.global_dimensions[0]
            .checked_mul(2)
            .ok_or_else(|| "SM90a tensor-map row bytes overflow u64".to_string())?;
        rows.checked_sub(1)
            .and_then(|rows| rows.checked_mul(key.outer_byte_stride))
            .and_then(|prefix| prefix.checked_add(row_bytes))
            .ok_or_else(|| "SM90a tensor-map allocation span overflows u64".to_string())
    };
    Ok([
        Sm90aAllocationIdentity::query(
            keys[0].base,
            required(keys[0])?,
            context_handle,
            "SM90a",
            "A",
        )?,
        Sm90aAllocationIdentity::query(
            keys[1].base,
            required(keys[1])?,
            context_handle,
            "SM90a",
            "B",
        )?,
    ])
}

pub(super) fn sm90a_resources_digest(
    route: Sm90aForcedRoute,
    operands: Sm90aLaunchOperands,
    context_handle: usize,
    tensor_maps_digest: Sha256Digest,
) -> Result<Sha256Digest, String> {
    let (rows, columns, element_bytes) = match route.op {
        Sm90aOp::Nn => (route.shape.m, route.shape.n, 2_u64),
        Sm90aOp::Tn => (route.shape.k, route.shape.n, 4_u64),
        Sm90aOp::Nt => (route.shape.m, route.shape.k, 2_u64),
    };
    let row_bytes = u64::try_from(columns)
        .ok()
        .and_then(|columns| columns.checked_mul(element_bytes))
        .ok_or_else(|| "SM90a output row bytes overflow u64".to_string())?;
    let stride_bytes = u64::try_from(route.shape.ldc)
        .ok()
        .and_then(|stride| stride.checked_mul(element_bytes))
        .ok_or_else(|| "SM90a output stride bytes overflow u64".to_string())?;
    let output_bytes = u64::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_sub(1))
        .and_then(|rows| rows.checked_mul(stride_bytes))
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or_else(|| "SM90a output allocation span overflows u64".to_string())?;
    let output = Sm90aAllocationIdentity::query(
        operands.output_ptr,
        output_bytes,
        context_handle,
        "SM90a",
        "output",
    )?;
    let mut digest = output
        .append_digest(FramedSha256::new(b"sm90a-launch-resources.v2"))
        .required(b"tensor-maps", &tensor_maps_digest)
        .required(b"m", &(route.shape.m as u64).to_le_bytes())
        .required(b"k", &(route.shape.k as u64).to_le_bytes())
        .required(b"n", &(route.shape.n as u64).to_le_bytes())
        .required(b"lda", &(route.shape.lda as u64).to_le_bytes())
        .required(b"ldb", &(route.shape.ldb as u64).to_le_bytes())
        .required(b"ldc", &(route.shape.ldc as u64).to_le_bytes())
        .required(b"output-rows", &(rows as u64).to_le_bytes())
        .required(b"output-columns", &(columns as u64).to_le_bytes())
        .required(b"output-element-bytes", &element_bytes.to_le_bytes())
        .required(b"bias-present", &[u8::from(operands.bias_ptr != 0)])
        .required(b"alpha", &operands.alpha.to_bits().to_le_bytes())
        .required(b"beta", &operands.beta.to_bits().to_le_bytes());
    if operands.bias_ptr != 0 {
        let bias_bytes = u64::try_from(columns)
            .ok()
            .and_then(|columns| columns.checked_mul(4))
            .ok_or_else(|| "SM90a bias allocation span overflows u64".to_string())?;
        let bias = Sm90aAllocationIdentity::query(
            operands.bias_ptr,
            bias_bytes,
            context_handle,
            "SM90a",
            "bias",
        )?;
        digest = bias.append_digest(digest);
    }
    Ok(digest.finish())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm90aLaunchOperands {
    pub output_ptr: CUptr,
    pub bias_ptr: CUptr,
    pub alpha: f32,
    pub beta: f32,
}

pub type Sm100TensorMap = Sm90aTensorMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm100MapRequest {
    pub op: Sm100Op,
    pub dtype: WeightDtype,
    pub tile: Sm100Tile,
    pub a_ptr: CUptr,
    pub b_ptr: CUptr,
    pub shape: Sm100Shape,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm100LaunchOperands {
    pub output_ptr: CUptr,
    pub bias_ptr: CUptr,
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Sm100TensorOrigins {
    pub a_x: i32,
    pub a_y: i32,
    pub b_x: i32,
    pub b_y: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm100KernelParams {
    pub a_x: i32,
    pub a_y: i32,
    pub b_x: i32,
    pub b_y: i32,
    pub alpha: f32,
    pub beta: f32,
    pub m: i32,
    pub k: i32,
    pub n: i32,
    pub ldc: i32,
}

unsafe impl DeviceRepr for Sm100KernelParams {}

const _: () = {
    assert!(std::mem::size_of::<Sm100KernelParams>() == 40);
    assert!(std::mem::align_of::<Sm100KernelParams>() == 4);
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sm100MapBinding {
    pub context_handle: usize,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
    pub target: Sm100TargetCandidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm100PreparedTensorMaps {
    pub(super) a: Sm100TensorMap,
    pub(super) b: Sm100TensorMap,
    pub(super) keys: [Sm90aTensorMapKey; 2],
    pub(super) request: Sm100MapRequest,
    pub(super) binding: Sm100MapBinding,
    pub(super) allocations: [Sm90aAllocationIdentity; 2],
    pub(super) origins: Sm100TensorOrigins,
}

impl Sm100PreparedTensorMaps {
    pub fn identity_digest(&self) -> Sha256Digest {
        let dtype = dtype_tag(self.request.dtype);
        let op = match self.request.op {
            Sm100Op::Nn => 0,
            Sm100Op::Tn => 1,
            Sm100Op::Nt => 2,
        };
        let tile = match self.request.tile {
            Sm100Tile::M128N64 => 64_u32,
            Sm100Tile::M128N128 => 128_u32,
        };
        let mut digest = FramedSha256::new(b"sm100-tensor-map-pair.v1")
            .required(b"op", &[op])
            .required(b"dtype", &[dtype])
            .required(b"tile-columns", &tile.to_le_bytes())
            .required(
                b"tensor-map-revision",
                &SM100_TENSOR_MAP_REVISION.to_le_bytes(),
            )
            .required(b"a-origin-x", &self.origins.a_x.to_le_bytes())
            .required(b"a-origin-y", &self.origins.a_y.to_le_bytes())
            .required(b"b-origin-x", &self.origins.b_x.to_le_bytes())
            .required(b"b-origin-y", &self.origins.b_y.to_le_bytes());
        for key in self.keys {
            digest = append_tensor_map_key(digest, key);
        }
        for (index, map) in [self.a, self.b].into_iter().enumerate() {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    std::ptr::from_ref(&map.0).cast::<u8>(),
                    std::mem::size_of::<sys::CUtensorMap>(),
                )
            };
            digest = digest
                .required(b"descriptor-index", &(index as u64).to_le_bytes())
                .required(b"encoded-descriptor", bytes);
        }
        for allocation in self.allocations {
            digest = allocation.append_digest(digest);
        }
        digest.finish()
    }

    pub(super) fn matches_binding(&self, binding: Sm100MapBinding) -> bool {
        self.binding == binding
    }

    pub(super) fn validate_live_allocations(&self) -> Result<(), String> {
        let plan = sm100_tensor_map_plan(self.request, self.binding.context_handle)?;
        if plan.keys != self.keys
            || plan.allocations != self.allocations
            || plan.origins != self.origins
        {
            return Err(
                "SM100 input allocation or tensor-map origin changed since preparation".into(),
            );
        }
        Ok(())
    }
}

fn dtype_tag(dtype: WeightDtype) -> u8 {
    match dtype {
        WeightDtype::F32 => 0,
        WeightDtype::F16 => 1,
        WeightDtype::Bf16 => 2,
    }
}

fn append_tensor_map_key(digest: FramedSha256, key: Sm90aTensorMapKey) -> FramedSha256 {
    digest
        .required(b"base", &key.base.to_le_bytes())
        .required(b"global-0", &key.global_dimensions[0].to_le_bytes())
        .required(b"global-1", &key.global_dimensions[1].to_le_bytes())
        .required(b"outer-stride", &key.outer_byte_stride.to_le_bytes())
        .required(b"box-0", &key.box_dimensions[0].to_le_bytes())
        .required(b"box-1", &key.box_dimensions[1].to_le_bytes())
}

#[derive(Clone, Copy)]
struct Sm100OperandLayout {
    pointer: CUptr,
    stride: usize,
    width: usize,
    rows: usize,
    issued_coordinate_max: [usize; 2],
    box_dimensions: [u32; 2],
    name: &'static str,
}

fn validate_sm100_issued_coordinates(
    layout: Sm100OperandLayout,
    origin: (u64, u64),
) -> Result<(), String> {
    for (axis, origin, issued) in [
        ("x", origin.0, layout.issued_coordinate_max[0]),
        ("y", origin.1, layout.issued_coordinate_max[1]),
    ] {
        let issued = u64::try_from(issued).map_err(|_| {
            format!(
                "SM100 {} issued {axis} coordinate exceeds u64::MAX",
                layout.name
            )
        })?;
        let coordinate = origin.checked_add(issued).ok_or_else(|| {
            format!(
                "SM100 {} issued {axis} coordinate overflows u64",
                layout.name
            )
        })?;
        i32::try_from(coordinate).map_err(|_| {
            format!(
                "SM100 {} issued {axis} coordinate exceeds i32::MAX after applying the subview origin",
                layout.name
            )
        })?;
    }
    Ok(())
}

fn sm100_operand_layouts(request: Sm100MapRequest) -> [Sm100OperandLayout; 2] {
    let shape = request.shape;
    let columns = request.tile.output_columns();
    let last_tile = |extent: usize, tile: usize| extent.saturating_sub(1) / tile * tile;
    let output_rows = if request.op == Sm100Op::Tn {
        shape.k
    } else {
        shape.m
    };
    let output_columns = if request.op == Sm100Op::Nt {
        shape.k
    } else {
        shape.n
    };
    let reduction = match request.op {
        Sm100Op::Nn => shape.k,
        Sm100Op::Tn => shape.m,
        Sm100Op::Nt => shape.n,
    };
    let last_output_row = last_tile(output_rows, 128);
    let last_output_column = last_tile(output_columns, columns as usize);
    let last_reduction = last_tile(reduction, 64);
    let second_half = if columns == 128 { 64 } else { 0 };
    match request.op {
        Sm100Op::Nn => [
            Sm100OperandLayout {
                pointer: request.a_ptr,
                stride: shape.lda,
                width: shape.k,
                rows: shape.m,
                issued_coordinate_max: [last_reduction, last_output_row],
                box_dimensions: [64, 128],
                name: "A",
            },
            Sm100OperandLayout {
                pointer: request.b_ptr,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.k,
                issued_coordinate_max: [last_output_column + second_half, last_reduction],
                box_dimensions: [64, 64],
                name: "B",
            },
        ],
        Sm100Op::Tn => [
            Sm100OperandLayout {
                pointer: request.a_ptr,
                stride: shape.lda,
                width: shape.k,
                rows: shape.m,
                issued_coordinate_max: [last_output_row + 64, last_reduction],
                box_dimensions: [64, 64],
                name: "A",
            },
            Sm100OperandLayout {
                pointer: request.b_ptr,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.m,
                issued_coordinate_max: [last_output_column + second_half, last_reduction],
                box_dimensions: [64, 64],
                name: "B",
            },
        ],
        Sm100Op::Nt => [
            Sm100OperandLayout {
                pointer: request.a_ptr,
                stride: shape.lda,
                width: shape.n,
                rows: shape.m,
                issued_coordinate_max: [last_reduction, last_output_row],
                box_dimensions: [64, 128],
                name: "A",
            },
            Sm100OperandLayout {
                pointer: request.b_ptr,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.k,
                issued_coordinate_max: [last_reduction, last_output_column],
                box_dimensions: [64, columns],
                name: "B",
            },
        ],
    }
}

fn validate_sm100_request(request: Sm100MapRequest) -> Result<(), String> {
    request.shape.validate(request.op)?;
    if !matches!(request.dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM100 TCGEN tensor maps require bf16 or f16 operands".into());
    }
    for layout in sm100_operand_layouts(request) {
        if layout.pointer == 0 || !layout.pointer.is_multiple_of(2) {
            return Err(format!(
                "SM100 {} logical pointer must be non-null and element aligned",
                layout.name
            ));
        }
        let stride_bytes = layout
            .stride
            .checked_mul(2)
            .ok_or_else(|| format!("SM100 {} byte stride overflows usize", layout.name))?;
        if !stride_bytes.is_multiple_of(16) {
            return Err(format!(
                "SM100 {} byte stride must be a multiple of 16",
                layout.name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn sm100_tensor_map_keys(
    request: Sm100MapRequest,
) -> Result<[Sm90aTensorMapKey; 2], String> {
    validate_sm100_request(request)?;
    let layouts = sm100_operand_layouts(request);
    let make_key = |layout: Sm100OperandLayout| -> Result<Sm90aTensorMapKey, String> {
        Ok(Sm90aTensorMapKey {
            base: layout.pointer,
            global_dimensions: [layout.width as u64, layout.rows as u64],
            outer_byte_stride: u64::try_from(
                layout
                    .stride
                    .checked_mul(2)
                    .ok_or_else(|| format!("SM100 {} byte stride overflows usize", layout.name))?,
            )
            .map_err(|_| format!("SM100 {} byte stride exceeds u64::MAX", layout.name))?,
            box_dimensions: layout.box_dimensions,
        })
    };
    let keys = [make_key(layouts[0])?, make_key(layouts[1])?];
    for key in keys {
        key.validate("SM100")?;
    }
    Ok(keys)
}

pub fn validate_sm100_map_request(request: Sm100MapRequest) -> Result<(), String> {
    validate_sm100_request(request)
}

pub(super) fn encode_sm100_tensor_maps(
    keys: [Sm90aTensorMapKey; 2],
    request: Sm100MapRequest,
    binding: Sm100MapBinding,
    allocations: [Sm90aAllocationIdentity; 2],
    origins: Sm100TensorOrigins,
) -> Result<Sm100PreparedTensorMaps, String> {
    Ok(Sm100PreparedTensorMaps {
        a: Sm100TensorMap::encode(keys[0], "SM100")?,
        b: Sm100TensorMap::encode(keys[1], "SM100")?,
        keys,
        request,
        binding,
        allocations,
        origins,
    })
}

pub(super) struct Sm100TensorMapPlan {
    pub keys: [Sm90aTensorMapKey; 2],
    pub allocations: [Sm90aAllocationIdentity; 2],
    pub origins: Sm100TensorOrigins,
}

fn sm100_subview_plan(
    layout: Sm100OperandLayout,
    context_handle: usize,
) -> Result<(Sm90aTensorMapKey, Sm90aAllocationIdentity, (i32, i32)), String> {
    let initial =
        Sm90aAllocationIdentity::query(layout.pointer, 2, context_handle, "SM100", layout.name)?;
    if !initial.offset_bytes.is_multiple_of(2) {
        return Err(format!(
            "SM100 {} subview offset is not element aligned",
            layout.name
        ));
    }
    let stride = u64::try_from(layout.stride)
        .map_err(|_| format!("SM100 {} stride exceeds u64::MAX", layout.name))?;
    let element_offset = initial.offset_bytes / 2;
    let origin_x = element_offset % stride;
    let origin_y = element_offset / stride;
    let width = u64::try_from(layout.width)
        .map_err(|_| format!("SM100 {} width exceeds u64::MAX", layout.name))?;
    let rows = u64::try_from(layout.rows)
        .map_err(|_| format!("SM100 {} rows exceed u64::MAX", layout.name))?;
    let logical_end_x = origin_x
        .checked_add(width)
        .ok_or_else(|| format!("SM100 {} column origin overflows u64", layout.name))?;
    if logical_end_x > stride {
        return Err(format!(
            "SM100 {} subview row wraps across its declared stride",
            layout.name
        ));
    }
    let logical_end_y = origin_y
        .checked_add(rows)
        .ok_or_else(|| format!("SM100 {} row origin overflows u64", layout.name))?;
    let origin = (
        i32::try_from(origin_x)
            .map_err(|_| format!("SM100 {} column origin exceeds i32::MAX", layout.name))?,
        i32::try_from(origin_y)
            .map_err(|_| format!("SM100 {} row origin exceeds i32::MAX", layout.name))?,
    );
    i32::try_from(logical_end_x)
        .map_err(|_| format!("SM100 {} global width exceeds i32::MAX", layout.name))?;
    i32::try_from(logical_end_y)
        .map_err(|_| format!("SM100 {} global rows exceed i32::MAX", layout.name))?;
    validate_sm100_issued_coordinates(layout, (origin_x, origin_y))?;
    let required_bytes = matrix_span_bytes(
        layout.rows,
        layout.width,
        layout.stride,
        2,
        &format!("SM100 {} subview", layout.name),
    )?;
    let allocation = Sm90aAllocationIdentity::query(
        layout.pointer,
        required_bytes,
        context_handle,
        "SM100",
        layout.name,
    )?;
    if allocation.allocation_base != initial.allocation_base
        || allocation.allocation_bytes != initial.allocation_bytes
        || allocation.offset_bytes != initial.offset_bytes
        || allocation.buffer_id != initial.buffer_id
    {
        return Err(format!(
            "SM100 {} allocation identity changed during tensor-map preparation",
            layout.name
        ));
    }
    let key = Sm90aTensorMapKey {
        base: allocation.allocation_base,
        global_dimensions: [logical_end_x, logical_end_y],
        outer_byte_stride: stride
            .checked_mul(2)
            .ok_or_else(|| format!("SM100 {} byte stride overflows u64", layout.name))?,
        box_dimensions: layout.box_dimensions,
    };
    key.validate("SM100")?;
    Ok((key, allocation, origin))
}

pub(super) fn sm100_tensor_map_plan(
    request: Sm100MapRequest,
    context_handle: usize,
) -> Result<Sm100TensorMapPlan, String> {
    validate_sm100_request(request)?;
    let layouts = sm100_operand_layouts(request);
    let (a_key, a_allocation, (a_x, a_y)) = sm100_subview_plan(layouts[0], context_handle)?;
    let (b_key, b_allocation, (b_x, b_y)) = sm100_subview_plan(layouts[1], context_handle)?;
    Ok(Sm100TensorMapPlan {
        keys: [a_key, b_key],
        allocations: [a_allocation, b_allocation],
        origins: Sm100TensorOrigins { a_x, a_y, b_x, b_y },
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sm100LaunchResourceSnapshot {
    output: Sm90aAllocationIdentity,
    bias: Option<Sm90aAllocationIdentity>,
}

impl Sm100LaunchResourceSnapshot {
    pub(super) fn query(
        route: Sm100ForcedRoute,
        operands: Sm100LaunchOperands,
        context_handle: usize,
    ) -> Result<Self, String> {
        let (rows, columns, element_bytes) = sm100_output_layout(route);
        let output_bytes = matrix_span_bytes(
            rows,
            columns,
            route.shape.ldc,
            element_bytes,
            "SM100 output",
        )?;
        let output = Sm90aAllocationIdentity::query(
            operands.output_ptr,
            output_bytes,
            context_handle,
            "SM100",
            "output",
        )?;
        let bias = if operands.bias_ptr == 0 {
            None
        } else {
            let bytes = u64::try_from(columns)
                .ok()
                .and_then(|columns| columns.checked_mul(4))
                .ok_or_else(|| "SM100 bias allocation span overflows u64".to_string())?;
            Some(Sm90aAllocationIdentity::query(
                operands.bias_ptr,
                bytes,
                context_handle,
                "SM100",
                "bias",
            )?)
        };
        Ok(Self { output, bias })
    }

    pub(super) fn digest(
        self,
        route: Sm100ForcedRoute,
        operands: Sm100LaunchOperands,
        tensor_maps_digest: Sha256Digest,
    ) -> Sha256Digest {
        let mut digest = self
            .output
            .append_digest(FramedSha256::new(b"sm100-launch-resources.v1"))
            .required(b"tensor-maps", &tensor_maps_digest)
            .required(b"m", &(route.shape.m as u64).to_le_bytes())
            .required(b"k", &(route.shape.k as u64).to_le_bytes())
            .required(b"n", &(route.shape.n as u64).to_le_bytes())
            .required(b"lda", &(route.shape.lda as u64).to_le_bytes())
            .required(b"ldb", &(route.shape.ldb as u64).to_le_bytes())
            .required(b"ldc", &(route.shape.ldc as u64).to_le_bytes())
            .required(b"output-pointer", &operands.output_ptr.to_le_bytes())
            .required(b"bias-pointer", &operands.bias_ptr.to_le_bytes())
            .required(b"alpha", &operands.alpha.to_bits().to_le_bytes())
            .required(b"beta", &operands.beta.to_bits().to_le_bytes());
        if let Some(bias) = self.bias {
            digest = bias.append_digest(digest);
        }
        digest.finish()
    }
}

fn sm100_output_layout(route: Sm100ForcedRoute) -> (usize, usize, u64) {
    match route.op {
        Sm100Op::Nn => (route.shape.m, route.shape.n, 2),
        Sm100Op::Tn => (route.shape.k, route.shape.n, 4),
        Sm100Op::Nt => (route.shape.m, route.shape.k, 2),
    }
}

fn matrix_span_bytes(
    rows: usize,
    columns: usize,
    stride: usize,
    element_bytes: u64,
    name: &str,
) -> Result<u64, String> {
    let row_bytes = u64::try_from(columns)
        .ok()
        .and_then(|columns| columns.checked_mul(element_bytes))
        .ok_or_else(|| format!("{name} row bytes overflow u64"))?;
    let stride_bytes = u64::try_from(stride)
        .ok()
        .and_then(|stride| stride.checked_mul(element_bytes))
        .ok_or_else(|| format!("{name} stride bytes overflow u64"))?;
    u64::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_sub(1))
        .and_then(|rows| rows.checked_mul(stride_bytes))
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or_else(|| format!("{name} allocation span overflows u64"))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm100PreparedLaunch {
    pub(super) route: Sm100ForcedRoute,
    pub(super) maps: Sm100PreparedTensorMaps,
    pub(super) operands: Sm100LaunchOperands,
    pub(super) params: Sm100KernelParams,
    pub(super) identity: Sm100RouteIdentity,
    pub(super) resources: Sm100LaunchResourceSnapshot,
}

impl Sm100PreparedLaunch {
    pub fn identity(&self) -> Sm100RouteIdentity {
        self.identity
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::mamba_ssm::gpu) struct GemmDims {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub m_i32: i32,
    pub k_i32: i32,
    pub n_i32: i32,
    pub mk: usize,
    pub mn: usize,
    pub kn: usize,
    pub(super) m_u32: u32,
    pub(super) k_u32: u32,
    pub(super) n_u32: u32,
    pub(super) mk_u32: u32,
    pub(super) mn_u32: u32,
    pub(super) kn_u32: u32,
}

impl GemmDims {
    pub(super) fn checked(
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    ) -> Result<Self, String> {
        Self::checked_storage(m, k, n, [lda, ldb, ldc], [k, n, n], [m, k, m])
    }

    pub(in crate::mamba_ssm::gpu) fn nn(
        dims: (usize, usize, usize),
        lda: usize,
    ) -> Result<Self, String> {
        Self::checked(dims.0, dims.1, dims.2, lda, dims.2, dims.2)
    }

    pub(in crate::mamba_ssm::gpu) fn tn(dims: (usize, usize, usize)) -> Result<Self, String> {
        Self::checked_storage(
            dims.0,
            dims.1,
            dims.2,
            [dims.1, dims.2, dims.2],
            [dims.1, dims.2, dims.2],
            [dims.0, dims.0, dims.1],
        )
    }

    pub(in crate::mamba_ssm::gpu) fn nt(dims: (usize, usize, usize)) -> Result<Self, String> {
        Self::checked_storage(
            dims.0,
            dims.1,
            dims.2,
            [dims.2, dims.2, dims.1],
            [dims.2, dims.2, dims.1],
            [dims.0, dims.1, dims.0],
        )
    }

    fn checked_storage(
        m: usize,
        k: usize,
        n: usize,
        strides: [usize; 3],
        widths: [usize; 3],
        row_counts: [usize; 3],
    ) -> Result<Self, String> {
        let product = |lhs: usize, rhs: usize, name: &str| {
            lhs.checked_mul(rhs).ok_or_else(|| {
                invalid_gemm_dimensions(format!("{name} overflows usize ({lhs} * {rhs})"))
            })
        };
        let mk = product(m, k, "M*K")?;
        let mn = product(m, n, "M*N")?;
        let kn = product(k, n, "K*N")?;

        if m == 0 || k == 0 || n == 0 {
            return Err(invalid_gemm_dimensions(format!(
                "axes must be positive, got M={m} K={k} N={n}"
            )));
        }

        let axis_i32 = |value: usize, name: &str| {
            i32::try_from(value)
                .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds i32::MAX")))
        };
        let m_i32 = axis_i32(m, "M")?;
        let k_i32 = axis_i32(k, "K")?;
        let n_i32 = axis_i32(n, "N")?;
        for (value, name) in [(mk, "M*K"), (mn, "M*N"), (kn, "K*N")] {
            axis_i32(value, name)?;
        }

        let [lda, ldb, ldc] = strides;
        let [lda_min, ldb_min, ldc_min] = widths;
        let [a_rows, b_rows, c_rows] = row_counts;
        for (value, minimum, name) in [
            (lda, lda_min, "lda"),
            (ldb, ldb_min, "ldb"),
            (ldc, ldc_min, "ldc"),
        ] {
            if value == 0 || value < minimum {
                return Err(invalid_gemm_dimensions(format!(
                    "{name}={value} is smaller than the physical width {minimum}"
                )));
            }
        }
        let lda = axis_i32(lda, "lda")?;
        let ldb = axis_i32(ldb, "ldb")?;
        let ldc = axis_i32(ldc, "ldc")?;

        for (rows, stride, width, name) in [
            (a_rows, strides[0], widths[0], "A storage"),
            (b_rows, strides[1], widths[1], "B storage"),
            (c_rows, strides[2], widths[2], "C storage"),
        ] {
            let span = rows
                .checked_sub(1)
                .and_then(|last_row| last_row.checked_mul(stride))
                .and_then(|offset| offset.checked_add(width))
                .ok_or_else(|| invalid_gemm_dimensions(format!("{name} span overflows usize")))?;
            axis_i32(span, name)?;
        }

        let to_u32 = |value: usize, name: &str| {
            u32::try_from(value)
                .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds u32::MAX")))
        };
        Ok(Self {
            m,
            k,
            n,
            lda,
            ldb,
            ldc,
            m_i32,
            k_i32,
            n_i32,
            mk,
            mn,
            kn,
            m_u32: to_u32(m, "M")?,
            k_u32: to_u32(k, "K")?,
            n_u32: to_u32(n, "N")?,
            mk_u32: to_u32(mk, "M*K")?,
            mn_u32: to_u32(mn, "M*N")?,
            kn_u32: to_u32(kn, "K*N")?,
        })
    }

    pub(super) fn tuple(self) -> (usize, usize, usize) {
        (self.m, self.k, self.n)
    }
}

pub(super) fn invalid_gemm_dimensions(reason: impl std::fmt::Display) -> String {
    format!("invalid GEMM dimensions: {reason}")
}

pub(super) fn checked_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value)
        .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds u32::MAX")))
}

pub(super) fn checked_i32(value: usize, name: &str) -> Result<i32, String> {
    i32::try_from(value)
        .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds i32::MAX")))
}

pub(super) fn checked_usize(value: u32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .map_err(|_| invalid_gemm_dimensions(format!("{name}={value} exceeds usize::MAX")))
}

pub(super) fn checked_tile_grid(
    rows: u32,
    row_tile: u32,
    cols: u32,
    col_tile: u32,
) -> Result<u32, String> {
    rows.div_ceil(row_tile)
        .checked_mul(cols.div_ceil(col_tile))
        .ok_or_else(|| invalid_gemm_dimensions("tile grid overflows u32"))
}

pub(super) fn checked_grid_product(lhs: u32, rhs: u32, depth: u32) -> Result<u32, String> {
    lhs.checked_mul(rhs)
        .and_then(|value| value.checked_mul(depth))
        .ok_or_else(|| invalid_gemm_dimensions("launch grid overflows u32"))
}

pub(super) fn checked_u32_product(lhs: u32, rhs: u32, name: &str) -> Result<u32, String> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| invalid_gemm_dimensions(format!("{name} overflows u32")))
}

pub(super) fn checked_mul3(
    lhs: usize,
    middle: usize,
    rhs: usize,
    name: &str,
) -> Result<usize, String> {
    lhs.checked_mul(middle)
        .and_then(|value| value.checked_mul(rhs))
        .ok_or_else(|| invalid_gemm_dimensions(format!("{name} overflows usize")))
}

pub(super) fn checked_byte_offset(
    elements: usize,
    element_bytes: usize,
    name: &str,
) -> Result<u64, String> {
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| invalid_gemm_dimensions(format!("{name} byte offset overflows usize")))?;
    u64::try_from(bytes)
        .map_err(|_| invalid_gemm_dimensions(format!("{name} byte offset exceeds u64::MAX")))
}

pub(super) fn checked_ptr_add(base: u64, offset: u64, name: &str) -> Result<u64, String> {
    base.checked_add(offset)
        .ok_or_else(|| invalid_gemm_dimensions(format!("{name} pointer offset overflows u64")))
}

pub(super) fn validate_bias_preseed(
    alpha: f32,
    bias_ptr: CUptr,
    route: &str,
) -> Result<(), String> {
    if bias_ptr != 0 && alpha != 1.0 {
        return Err(format!(
            "{route}: bias pre-seeding requires alpha == 1.0, got {alpha}"
        ));
    }
    Ok(())
}

/// Operand bundle for the TC NN forward (`Y = X @ W + bias`).
pub struct TcFwdOperands {
    pub y: TypedPtr,
    pub x: TypedPtr,
    pub w: TypedPtr,
    /// f32 bias pointer, 0 = none.
    pub bias_ptr: CUptr,
}

/// Strided scalar-forward operands shared by every deterministic NN bucket.
#[derive(Clone, Copy)]
pub struct SgemmFwdSubOperands {
    pub x_ptr: CUptr,
    pub lda: usize,
    pub w_ptr: CUptr,
    /// f32 bias pointer, 0 = none.
    pub bias_ptr: CUptr,
}

pub const SM120_TENSOR_MAP_REVISION: u16 = 1;
pub const SM120_TUNING_REVISION: u16 = 0;
pub const SM120_SCHEDULE_REVISION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm120Op {
    Nn,
    Tn,
    Nt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm120Tile {
    M64N64,
    M128N64,
    M64N128,
    M128N128,
}

impl Sm120Tile {
    pub const fn output_rows(self) -> u32 {
        match self {
            Self::M64N64 | Self::M64N128 => 64,
            Self::M128N64 | Self::M128N128 => 128,
        }
    }

    pub const fn output_columns(self) -> u32 {
        match self {
            Self::M64N64 | Self::M128N64 => 64,
            Self::M64N128 | Self::M128N128 => 128,
        }
    }

    pub const fn compute_warps(self) -> u32 {
        self.output_rows() / 32 * (self.output_columns() / 32)
    }

    pub const fn threads(self) -> u32 {
        self.compute_warps() * 32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm120Bk {
    Bk32,
    Bk64,
}

impl Sm120Bk {
    pub const fn elements(self) -> u32 {
        match self {
            Self::Bk32 => 32,
            Self::Bk64 => 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm120Stages {
    S2,
    S3,
}

impl Sm120Stages {
    pub const fn count(self) -> u8 {
        match self {
            Self::S2 => 2,
            Self::S3 => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm120PhysicalRoute {
    pub tile: Sm120Tile,
    pub bk: Sm120Bk,
    pub stages: Sm120Stages,
}

impl Sm120PhysicalRoute {
    pub const fn dynamic_shared_bytes(self) -> u32 {
        let payload =
            (self.tile.output_rows() + self.tile.output_columns()) * self.bk.elements() * 2;
        payload * self.stages.count() as u32 + 128
    }

    pub const fn expected_transaction_bytes(self) -> u32 {
        (self.tile.output_rows() + self.tile.output_columns()) * self.bk.elements() * 2
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm120Shape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
}

impl Sm120Shape {
    pub fn contiguous(op: Sm120Op, dims: (usize, usize, usize)) -> Self {
        let (m, k, n) = dims;
        match op {
            Sm120Op::Nn | Sm120Op::Tn => Self {
                m,
                k,
                n,
                lda: k,
                ldb: n,
                ldc: n,
            },
            Sm120Op::Nt => Self {
                m,
                k,
                n,
                lda: n,
                ldb: n,
                ldc: k,
            },
        }
    }

    pub fn validate(self, op: Sm120Op) -> Result<(), String> {
        for (value, name) in [(self.m, "M"), (self.k, "K"), (self.n, "N")] {
            if value == 0 {
                return Err(invalid_gemm_dimensions(format!("{name} must be positive")));
            }
            checked_i32(value, name)?;
        }
        let (a_width, b_width, c_width) = match op {
            Sm120Op::Nn | Sm120Op::Tn => (self.k, self.n, self.n),
            Sm120Op::Nt => (self.n, self.n, self.k),
        };
        for (stride, width, name) in [
            (self.lda, a_width, "lda"),
            (self.ldb, b_width, "ldb"),
            (self.ldc, c_width, "ldc"),
        ] {
            if stride < width {
                return Err(invalid_gemm_dimensions(format!(
                    "{name}={stride} is smaller than the physical width {width}"
                )));
            }
            checked_i32(stride, name)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm120ForcedRoute {
    pub op: Sm120Op,
    pub dtype: WeightDtype,
    pub physical: Sm120PhysicalRoute,
    pub shape: Sm120Shape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm120TargetCandidate {
    pub device_cc: (i32, i32),
    pub nvrtc_arch: &'static str,
    pub ptx_target: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm120KernelSpec {
    pub op: Sm120Op,
    pub dtype: WeightDtype,
    pub physical: Sm120PhysicalRoute,
    pub symbol: &'static str,
    pub threads: u32,
    pub dynamic_shared_bytes: u32,
    pub empty_barrier_arrivals: u32,
    pub full_barrier_arrivals: u32,
    pub cluster: (u8, u8, u8),
    pub warp_tile: (u32, u32),
    pub expected_transaction_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm120KernelResources {
    pub threads: u32,
    pub dynamic_shared_bytes: u32,
    pub max_threads_per_block: u32,
    pub local_bytes: u32,
    pub spill_store_bytes: u32,
    pub spill_load_bytes: u32,
    pub registers_per_thread: u32,
    pub active_blocks_per_sm: u32,
}

macro_rules! sm120_spec {
    ($op:expr, $dtype:expr, $tile:expr, $bk:expr, $stages:expr, $symbol:expr) => {{
        let physical = Sm120PhysicalRoute {
            tile: $tile,
            bk: $bk,
            stages: $stages,
        };
        Sm120KernelSpec {
            op: $op,
            dtype: $dtype,
            physical,
            symbol: $symbol,
            threads: physical.tile.threads(),
            dynamic_shared_bytes: physical.dynamic_shared_bytes(),
            empty_barrier_arrivals: physical.tile.threads(),
            full_barrier_arrivals: 1,
            cluster: (1, 1, 1),
            warp_tile: (32, 32),
            expected_transaction_bytes: physical.expected_transaction_bytes(),
        }
    }};
}

macro_rules! sm120_specs {
    ($(($op:expr, $op_name:literal, $tile:expr, $tile_name:literal)),+ $(,)?) => {
        [$(
            sm120_spec!($op, WeightDtype::Bf16, $tile, Sm120Bk::Bk32, Sm120Stages::S2,
                concat!("sgemm_bi_", $op_name, "_sm120_tma_", $tile_name, "_bk32_s2_bf16")),
            sm120_spec!($op, WeightDtype::F16, $tile, Sm120Bk::Bk32, Sm120Stages::S2,
                concat!("sgemm_bi_", $op_name, "_sm120_tma_", $tile_name, "_bk32_s2_f16")),
            sm120_spec!($op, WeightDtype::Bf16, $tile, Sm120Bk::Bk32, Sm120Stages::S3,
                concat!("sgemm_bi_", $op_name, "_sm120_tma_", $tile_name, "_bk32_s3_bf16")),
            sm120_spec!($op, WeightDtype::F16, $tile, Sm120Bk::Bk32, Sm120Stages::S3,
                concat!("sgemm_bi_", $op_name, "_sm120_tma_", $tile_name, "_bk32_s3_f16")),
            sm120_spec!($op, WeightDtype::Bf16, $tile, Sm120Bk::Bk64, Sm120Stages::S2,
                concat!("sgemm_bi_", $op_name, "_sm120_tma_", $tile_name, "_bk64_s2_bf16")),
            sm120_spec!($op, WeightDtype::F16, $tile, Sm120Bk::Bk64, Sm120Stages::S2,
                concat!("sgemm_bi_", $op_name, "_sm120_tma_", $tile_name, "_bk64_s2_f16")),
            sm120_spec!($op, WeightDtype::Bf16, $tile, Sm120Bk::Bk64, Sm120Stages::S3,
                concat!("sgemm_bi_", $op_name, "_sm120_tma_", $tile_name, "_bk64_s3_bf16")),
            sm120_spec!($op, WeightDtype::F16, $tile, Sm120Bk::Bk64, Sm120Stages::S3,
                concat!("sgemm_bi_", $op_name, "_sm120_tma_", $tile_name, "_bk64_s3_f16")),
        )+]
    };
}

pub const SM120_KERNEL_SPECS: [Sm120KernelSpec; 96] = sm120_specs!(
    (Sm120Op::Nn, "nn", Sm120Tile::M64N64, "64x64"),
    (Sm120Op::Nn, "nn", Sm120Tile::M128N64, "128x64"),
    (Sm120Op::Nn, "nn", Sm120Tile::M64N128, "64x128"),
    (Sm120Op::Nn, "nn", Sm120Tile::M128N128, "128x128"),
    (Sm120Op::Tn, "tn", Sm120Tile::M64N64, "64x64"),
    (Sm120Op::Tn, "tn", Sm120Tile::M128N64, "128x64"),
    (Sm120Op::Tn, "tn", Sm120Tile::M64N128, "64x128"),
    (Sm120Op::Tn, "tn", Sm120Tile::M128N128, "128x128"),
    (Sm120Op::Nt, "nt", Sm120Tile::M64N64, "64x64"),
    (Sm120Op::Nt, "nt", Sm120Tile::M128N64, "128x64"),
    (Sm120Op::Nt, "nt", Sm120Tile::M64N128, "64x128"),
    (Sm120Op::Nt, "nt", Sm120Tile::M128N128, "128x128"),
);

impl Sm120ForcedRoute {
    pub fn kernel_spec(self) -> Result<&'static Sm120KernelSpec, String> {
        SM120_KERNEL_SPECS
            .iter()
            .find(|spec| {
                spec.op == self.op && spec.dtype == self.dtype && spec.physical == self.physical
            })
            .ok_or_else(|| "no SM120 TMA kernel matches the forced route".to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm120NumericContract {
    TmaMma16F32V1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm120RouteIdentity {
    pub numeric_contract: Sm120NumericContract,
    pub op: Sm120Op,
    pub dtype: WeightDtype,
    pub physical: Sm120PhysicalRoute,
    pub shape: Sm120Shape,
    pub symbol: &'static str,
    pub module_kind: ModuleKind,
    pub target: Sm120TargetCandidate,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
    pub device_caps: crate::mamba_ssm::gpu::kernel_identity::DeviceCaps,
    pub tensor_map_revision: u16,
    pub tensor_maps_digest: Sha256Digest,
    pub resources_digest: Sha256Digest,
    pub tuning_revision: u16,
    pub schedule_revision: u16,
}

impl Sm120RouteIdentity {
    pub fn ensure_current(self, live: Self, prefix: &str) -> Result<(), String> {
        if self == live {
            Ok(())
        } else {
            Err(format!(
                "{prefix}: SM120 route changed since preparation; re-capture before replay"
            ))
        }
    }

    pub fn resolved_route(self) -> Result<ResolvedGemmRoute, String> {
        let route = Sm120ForcedRoute {
            op: self.op,
            dtype: self.dtype,
            physical: self.physical,
            shape: self.shape,
        };
        route.shape.validate(route.op)?;
        let spec = route.kernel_spec()?;
        if self.symbol != spec.symbol {
            return Err("SM120 identity symbol does not match its physical route".into());
        }
        if self.module_kind != ModuleKind::TriadSm120
            || self.artifact.module_kind != ModuleKind::TriadSm120
        {
            return Err("SM120 identity does not name the SM120 TRIAD module".into());
        }
        if self.compiler.target.as_str() != self.target.nvrtc_arch
            || self.device.target.as_str() != self.target.ptx_target
        {
            return Err("SM120 identity target transaction is inconsistent".into());
        }
        let device_cc = (
            u32::try_from(self.target.device_cc.0)
                .map_err(|_| "SM120 target has a negative CC major".to_string())?,
            u32::try_from(self.target.device_cc.1)
                .map_err(|_| "SM120 target has a negative CC minor".to_string())?,
        );
        if self.device.compute_capability != device_cc
            || self.device_caps.compute_capability != device_cc
            || self.device_caps.nvrtc_version != self.compiler.nvrtc_version
            || self.device_caps.accepted_target != Some(self.compiler.target)
            || !self.device_caps.tensor_map_access
            || self.device_caps.optin_shared_bytes < spec.dynamic_shared_bytes
        {
            return Err("SM120 identity device capabilities are inconsistent".into());
        }
        let op = match self.op {
            Sm120Op::Nn => ResolvedGemmOp::Nn,
            Sm120Op::Tn => ResolvedGemmOp::Tn,
            Sm120Op::Nt => ResolvedGemmOp::Nt,
        };
        let dtype = match self.dtype {
            WeightDtype::F16 => PolicyDtype::F16,
            WeightDtype::Bf16 => PolicyDtype::Bf16,
            WeightDtype::F32 => return Err("SM120 TMA route requires f16 or bf16".into()),
        };
        Ok(ResolvedGemmRoute {
            op,
            dtype,
            backend: PhysicalGemmBackend::Sm120TmaMma16V1,
            numeric_contract: ResolvedNumericContract::MmaSyncF32V1,
            instruction_family: ResolvedInstructionFamily::MmaSync,
            instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 16 },
            operand_conversion: ResolvedOperandConversion::None,
            symbol: self.symbol,
            module_kind: self.module_kind,
            target: self.compiler.target,
            artifact: self.artifact,
            compiler: self.compiler,
            device: self.device,
            device_caps: self.device_caps,
            shape: (self.shape.m, self.shape.k, self.shape.n),
            strides: (self.shape.lda, self.shape.ldb, self.shape.ldc),
            tile: (
                self.physical.tile.output_rows(),
                self.physical.tile.output_columns(),
            ),
            bk: self.physical.bk.elements(),
            stages: self.physical.stages.count(),
            threads: spec.threads,
            tensor_map_revision: self.tensor_map_revision,
            tensor_maps_digest: self.tensor_maps_digest,
            resources_digest: self.resources_digest,
            tuning_table_revision: self.tuning_revision,
            schedule_revision: self.schedule_revision,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm120MapRequest {
    pub op: Sm120Op,
    pub dtype: WeightDtype,
    pub tile: Sm120Tile,
    pub bk: Sm120Bk,
    pub a_ptr: CUptr,
    pub b_ptr: CUptr,
    pub shape: Sm120Shape,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm120LaunchOperands {
    pub output_ptr: CUptr,
    pub bias_ptr: CUptr,
    pub alpha: f32,
    pub beta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm120KernelParams {
    pub a_x: i32,
    pub a_y: i32,
    pub b_x: i32,
    pub b_y: i32,
    pub alpha: f32,
    pub beta: f32,
    pub m: i32,
    pub k: i32,
    pub n: i32,
    pub ldc: i32,
}

unsafe impl DeviceRepr for Sm120KernelParams {}

const _: () = {
    assert!(std::mem::size_of::<Sm120KernelParams>() == 40);
    assert!(std::mem::align_of::<Sm120KernelParams>() == 4);
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Sm120Swizzle {
    Bytes64,
    Bytes128,
}

impl Sm120Swizzle {
    const fn from_bk(bk: Sm120Bk) -> Self {
        match bk {
            Sm120Bk::Bk32 => Self::Bytes64,
            Sm120Bk::Bk64 => Self::Bytes128,
        }
    }

    const fn bytes(self) -> u32 {
        match self {
            Self::Bytes64 => 64,
            Self::Bytes128 => 128,
        }
    }

    const fn driver(self) -> sys::CUtensorMapSwizzle {
        match self {
            Self::Bytes64 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
            Self::Bytes128 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Sm120TensorMapKey {
    base: CUptr,
    global_dimensions: [u64; 2],
    outer_byte_stride: u64,
    box_dimensions: [u32; 2],
    swizzle: Sm120Swizzle,
}

impl Sm120TensorMapKey {
    fn validate(self) -> Result<(), String> {
        if self.base == 0 || !self.base.is_multiple_of(128) {
            return Err(
                "SM120 swizzled tensor-map base must be non-null and 128-byte aligned".into(),
            );
        }
        if self.global_dimensions.contains(&0) {
            return Err("SM120 tensor-map dimensions must be positive".into());
        }
        if self.outer_byte_stride == 0
            || !self.outer_byte_stride.is_multiple_of(16)
            || self.outer_byte_stride >= (1_u64 << 40)
        {
            return Err(
                "SM120 tensor-map outer byte stride must be a positive multiple of 16 below 2^40"
                    .into(),
            );
        }
        let row_bytes = self.global_dimensions[0]
            .checked_mul(2)
            .ok_or_else(|| "SM120 tensor-map row width overflows u64".to_string())?;
        if self.outer_byte_stride < row_bytes {
            return Err("SM120 tensor-map outer stride is smaller than its inner dimension".into());
        }
        if self.box_dimensions.contains(&0) || self.box_dimensions.iter().any(|&dim| dim > 256) {
            return Err("SM120 tensor-map box dimensions must be in 1..=256".into());
        }
        let inner_bytes = self.box_dimensions[0]
            .checked_mul(2)
            .ok_or_else(|| "SM120 tensor-map box width overflows u32".to_string())?;
        if inner_bytes != self.swizzle.bytes() {
            return Err(format!(
                "SM120 tensor-map inner box span must equal its {}-byte swizzle",
                self.swizzle.bytes()
            ));
        }
        Ok(())
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm120TensorMap(sys::CUtensorMap);

unsafe impl DeviceRepr for Sm120TensorMap {}

const _: () = {
    assert!(std::mem::size_of::<Sm120TensorMap>() == std::mem::size_of::<sys::CUtensorMap>());
    assert!(std::mem::align_of::<Sm120TensorMap>() == std::mem::align_of::<sys::CUtensorMap>());
};

impl Sm120TensorMap {
    fn encode(key: Sm120TensorMapKey) -> Result<Self, String> {
        key.validate()?;
        let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
        let element_strides = [1_u32, 1_u32];
        let global_strides = [key.outer_byte_stride];
        unsafe {
            sys::cuTensorMapEncodeTiled(
                raw.as_mut_ptr(),
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
                2,
                key.base as usize as *mut std::ffi::c_void,
                key.global_dimensions.as_ptr(),
                global_strides.as_ptr(),
                key.box_dimensions.as_ptr(),
                element_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                key.swizzle.driver(),
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
            .result()
            .map_err(|error| format!("SM120 cuTensorMapEncodeTiled failed: {error:?}"))?;
            Ok(Self(raw.assume_init()))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Sm120TensorOrigins {
    pub a_x: i32,
    pub a_y: i32,
    pub b_x: i32,
    pub b_y: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sm120MapBinding {
    pub context_handle: usize,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
    pub device_caps: crate::mamba_ssm::gpu::kernel_identity::DeviceCaps,
    pub target: Sm120TargetCandidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm120PreparedTensorMaps {
    pub(super) a: Sm120TensorMap,
    pub(super) b: Sm120TensorMap,
    pub(super) keys: [Sm120TensorMapKey; 2],
    pub(super) request: Sm120MapRequest,
    pub(super) binding: Sm120MapBinding,
    pub(super) allocations: [Sm90aAllocationIdentity; 2],
    pub(super) origins: Sm120TensorOrigins,
}

impl Sm120PreparedTensorMaps {
    pub fn identity_digest(&self) -> Sha256Digest {
        let op = match self.request.op {
            Sm120Op::Nn => 0,
            Sm120Op::Tn => 1,
            Sm120Op::Nt => 2,
        };
        let mut digest = FramedSha256::new(b"sm120-tensor-map-pair.v1")
            .required(b"op", &[op])
            .required(b"dtype", &[dtype_tag(self.request.dtype)])
            .required(b"tile-m", &self.request.tile.output_rows().to_le_bytes())
            .required(b"tile-n", &self.request.tile.output_columns().to_le_bytes())
            .required(b"bk", &self.request.bk.elements().to_le_bytes())
            .required(
                b"tensor-map-revision",
                &SM120_TENSOR_MAP_REVISION.to_le_bytes(),
            )
            .required(b"a-origin-x", &self.origins.a_x.to_le_bytes())
            .required(b"a-origin-y", &self.origins.a_y.to_le_bytes())
            .required(b"b-origin-x", &self.origins.b_x.to_le_bytes())
            .required(b"b-origin-y", &self.origins.b_y.to_le_bytes());
        for key in self.keys {
            digest = digest
                .required(b"base", &key.base.to_le_bytes())
                .required(b"global-0", &key.global_dimensions[0].to_le_bytes())
                .required(b"global-1", &key.global_dimensions[1].to_le_bytes())
                .required(b"outer-stride", &key.outer_byte_stride.to_le_bytes())
                .required(b"box-0", &key.box_dimensions[0].to_le_bytes())
                .required(b"box-1", &key.box_dimensions[1].to_le_bytes())
                .required(b"swizzle", &key.swizzle.bytes().to_le_bytes());
        }
        for (index, map) in [self.a, self.b].into_iter().enumerate() {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    std::ptr::from_ref(&map.0).cast::<u8>(),
                    std::mem::size_of::<sys::CUtensorMap>(),
                )
            };
            digest = digest
                .required(b"descriptor-index", &(index as u64).to_le_bytes())
                .required(b"encoded-descriptor", bytes);
        }
        for allocation in self.allocations {
            digest = allocation.append_digest(digest);
        }
        digest.finish()
    }

    pub(super) fn matches_binding(&self, binding: Sm120MapBinding) -> bool {
        self.binding == binding
    }

    pub(super) fn validate_live_allocations(&self) -> Result<(), String> {
        let plan = sm120_tensor_map_plan(self.request, self.binding.context_handle)?;
        if plan.keys != self.keys
            || plan.allocations != self.allocations
            || plan.origins != self.origins
        {
            return Err(
                "SM120 input allocation or tensor-map origin changed since preparation".into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Sm120OperandLayout {
    pointer: CUptr,
    stride: usize,
    width: usize,
    rows: usize,
    issued_coordinate_max: [usize; 2],
    box_dimensions: [u32; 2],
    swizzle: Sm120Swizzle,
    name: &'static str,
}

fn sm120_operand_layouts(request: Sm120MapRequest) -> [Sm120OperandLayout; 2] {
    let shape = request.shape;
    let tile_m = request.tile.output_rows() as usize;
    let tile_n = request.tile.output_columns() as usize;
    let bk = request.bk.elements() as usize;
    let swizzle = Sm120Swizzle::from_bk(request.bk);
    let last_tile = |extent: usize, tile: usize| extent.saturating_sub(1) / tile * tile;
    let output_rows = if request.op == Sm120Op::Tn {
        shape.k
    } else {
        shape.m
    };
    let output_columns = if request.op == Sm120Op::Nt {
        shape.k
    } else {
        shape.n
    };
    let reduction = match request.op {
        Sm120Op::Nn => shape.k,
        Sm120Op::Tn => shape.m,
        Sm120Op::Nt => shape.n,
    };
    let last_output_row = last_tile(output_rows, tile_m);
    let last_output_column = last_tile(output_columns, tile_n);
    let last_reduction = last_tile(reduction, bk);
    match request.op {
        Sm120Op::Nn => [
            Sm120OperandLayout {
                pointer: request.a_ptr,
                stride: shape.lda,
                width: shape.k,
                rows: shape.m,
                issued_coordinate_max: [last_reduction, last_output_row],
                box_dimensions: [request.bk.elements(), request.tile.output_rows()],
                swizzle,
                name: "A",
            },
            Sm120OperandLayout {
                pointer: request.b_ptr,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.k,
                issued_coordinate_max: [last_output_column + tile_n - bk, last_reduction],
                box_dimensions: [request.bk.elements(), request.bk.elements()],
                swizzle,
                name: "B",
            },
        ],
        Sm120Op::Tn => [
            Sm120OperandLayout {
                pointer: request.a_ptr,
                stride: shape.lda,
                width: shape.k,
                rows: shape.m,
                issued_coordinate_max: [last_output_row + tile_m - bk, last_reduction],
                box_dimensions: [request.bk.elements(), request.bk.elements()],
                swizzle,
                name: "A",
            },
            Sm120OperandLayout {
                pointer: request.b_ptr,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.m,
                issued_coordinate_max: [last_output_column + tile_n - bk, last_reduction],
                box_dimensions: [request.bk.elements(), request.bk.elements()],
                swizzle,
                name: "B",
            },
        ],
        Sm120Op::Nt => [
            Sm120OperandLayout {
                pointer: request.a_ptr,
                stride: shape.lda,
                width: shape.n,
                rows: shape.m,
                issued_coordinate_max: [last_reduction, last_output_row],
                box_dimensions: [request.bk.elements(), request.tile.output_rows()],
                swizzle,
                name: "A",
            },
            Sm120OperandLayout {
                pointer: request.b_ptr,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.k,
                issued_coordinate_max: [last_reduction, last_output_column],
                box_dimensions: [request.bk.elements(), request.tile.output_columns()],
                swizzle,
                name: "B",
            },
        ],
    }
}

fn validate_sm120_issued_coordinates(
    layout: Sm120OperandLayout,
    origin: (u64, u64),
) -> Result<(), String> {
    for (axis, origin, issued) in [
        ("x", origin.0, layout.issued_coordinate_max[0]),
        ("y", origin.1, layout.issued_coordinate_max[1]),
    ] {
        let issued = u64::try_from(issued).map_err(|_| {
            format!(
                "SM120 {} issued {axis} coordinate exceeds u64::MAX",
                layout.name
            )
        })?;
        let coordinate = origin.checked_add(issued).ok_or_else(|| {
            format!(
                "SM120 {} issued {axis} coordinate overflows u64",
                layout.name
            )
        })?;
        i32::try_from(coordinate).map_err(|_| {
            format!(
                "SM120 {} issued {axis} coordinate exceeds i32::MAX after applying the subview origin",
                layout.name
            )
        })?;
    }
    Ok(())
}

fn validate_sm120_request(request: Sm120MapRequest) -> Result<(), String> {
    request.shape.validate(request.op)?;
    if !matches!(request.dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM120 TMA tensor maps require bf16 or f16 operands".into());
    }
    for layout in sm120_operand_layouts(request) {
        if layout.pointer == 0 || !layout.pointer.is_multiple_of(2) {
            return Err(format!(
                "SM120 {} logical pointer must be non-null and element aligned",
                layout.name
            ));
        }
        let stride_bytes = layout
            .stride
            .checked_mul(2)
            .ok_or_else(|| format!("SM120 {} byte stride overflows usize", layout.name))?;
        if !stride_bytes.is_multiple_of(16) {
            return Err(format!(
                "SM120 {} byte stride must be a multiple of 16",
                layout.name
            ));
        }
    }
    Ok(())
}

pub fn validate_sm120_map_request(request: Sm120MapRequest) -> Result<(), String> {
    validate_sm120_request(request)
}

pub(super) struct Sm120TensorMapPlan {
    pub keys: [Sm120TensorMapKey; 2],
    pub allocations: [Sm90aAllocationIdentity; 2],
    pub origins: Sm120TensorOrigins,
}

fn sm120_subview_plan(
    layout: Sm120OperandLayout,
    context_handle: usize,
) -> Result<(Sm120TensorMapKey, Sm90aAllocationIdentity, (i32, i32)), String> {
    let initial =
        Sm90aAllocationIdentity::query(layout.pointer, 2, context_handle, "SM120", layout.name)?;
    if !initial.offset_bytes.is_multiple_of(2) {
        return Err(format!(
            "SM120 {} subview offset is not element aligned",
            layout.name
        ));
    }
    let stride = u64::try_from(layout.stride)
        .map_err(|_| format!("SM120 {} stride exceeds u64::MAX", layout.name))?;
    let element_offset = initial.offset_bytes / 2;
    let origin_x = element_offset % stride;
    let origin_y = element_offset / stride;
    let width = u64::try_from(layout.width)
        .map_err(|_| format!("SM120 {} width exceeds u64::MAX", layout.name))?;
    let rows = u64::try_from(layout.rows)
        .map_err(|_| format!("SM120 {} rows exceeds u64::MAX", layout.name))?;
    let logical_end_x = origin_x
        .checked_add(width)
        .ok_or_else(|| format!("SM120 {} column origin overflows u64", layout.name))?;
    if logical_end_x > stride {
        return Err(format!(
            "SM120 {} subview row wraps across its declared stride",
            layout.name
        ));
    }
    let logical_end_y = origin_y
        .checked_add(rows)
        .ok_or_else(|| format!("SM120 {} row origin overflows u64", layout.name))?;
    let origin = (
        i32::try_from(origin_x)
            .map_err(|_| format!("SM120 {} column origin exceeds i32::MAX", layout.name))?,
        i32::try_from(origin_y)
            .map_err(|_| format!("SM120 {} row origin exceeds i32::MAX", layout.name))?,
    );
    i32::try_from(logical_end_x)
        .map_err(|_| format!("SM120 {} global width exceeds i32::MAX", layout.name))?;
    i32::try_from(logical_end_y)
        .map_err(|_| format!("SM120 {} global rows exceed i32::MAX", layout.name))?;
    validate_sm120_issued_coordinates(layout, (origin_x, origin_y))?;
    let required_bytes = matrix_span_bytes(
        layout.rows,
        layout.width,
        layout.stride,
        2,
        &format!("SM120 {} subview", layout.name),
    )?;
    let allocation = Sm90aAllocationIdentity::query(
        layout.pointer,
        required_bytes,
        context_handle,
        "SM120",
        layout.name,
    )?;
    if allocation.allocation_base != initial.allocation_base
        || allocation.allocation_bytes != initial.allocation_bytes
        || allocation.offset_bytes != initial.offset_bytes
        || allocation.buffer_id != initial.buffer_id
    {
        return Err(format!(
            "SM120 {} allocation identity changed during tensor-map preparation",
            layout.name
        ));
    }
    let key = Sm120TensorMapKey {
        base: allocation.allocation_base,
        global_dimensions: [logical_end_x, logical_end_y],
        outer_byte_stride: stride
            .checked_mul(2)
            .ok_or_else(|| format!("SM120 {} byte stride overflows u64", layout.name))?,
        box_dimensions: layout.box_dimensions,
        swizzle: layout.swizzle,
    };
    key.validate()?;
    Ok((key, allocation, origin))
}

pub(super) fn sm120_tensor_map_plan(
    request: Sm120MapRequest,
    context_handle: usize,
) -> Result<Sm120TensorMapPlan, String> {
    validate_sm120_request(request)?;
    let layouts = sm120_operand_layouts(request);
    let (a_key, a_allocation, (a_x, a_y)) = sm120_subview_plan(layouts[0], context_handle)?;
    let (b_key, b_allocation, (b_x, b_y)) = sm120_subview_plan(layouts[1], context_handle)?;
    Ok(Sm120TensorMapPlan {
        keys: [a_key, b_key],
        allocations: [a_allocation, b_allocation],
        origins: Sm120TensorOrigins { a_x, a_y, b_x, b_y },
    })
}

pub(super) fn encode_sm120_tensor_maps(
    keys: [Sm120TensorMapKey; 2],
    request: Sm120MapRequest,
    binding: Sm120MapBinding,
    allocations: [Sm90aAllocationIdentity; 2],
    origins: Sm120TensorOrigins,
) -> Result<Sm120PreparedTensorMaps, String> {
    Ok(Sm120PreparedTensorMaps {
        a: Sm120TensorMap::encode(keys[0])?,
        b: Sm120TensorMap::encode(keys[1])?,
        keys,
        request,
        binding,
        allocations,
        origins,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sm120LaunchResourceSnapshot {
    output: Sm90aAllocationIdentity,
    bias: Option<Sm90aAllocationIdentity>,
}

impl Sm120LaunchResourceSnapshot {
    pub(super) fn query(
        route: Sm120ForcedRoute,
        operands: Sm120LaunchOperands,
        context_handle: usize,
    ) -> Result<Self, String> {
        let (rows, columns, element_bytes) = sm120_output_layout(route);
        let output_bytes = matrix_span_bytes(
            rows,
            columns,
            route.shape.ldc,
            element_bytes,
            "SM120 output",
        )?;
        let output = Sm90aAllocationIdentity::query(
            operands.output_ptr,
            output_bytes,
            context_handle,
            "SM120",
            "output",
        )?;
        let bias = if operands.bias_ptr == 0 {
            None
        } else {
            let bytes = u64::try_from(columns)
                .ok()
                .and_then(|columns| columns.checked_mul(4))
                .ok_or_else(|| "SM120 bias allocation span overflows u64".to_string())?;
            Some(Sm90aAllocationIdentity::query(
                operands.bias_ptr,
                bytes,
                context_handle,
                "SM120",
                "bias",
            )?)
        };
        Ok(Self { output, bias })
    }

    pub(super) fn digest(
        self,
        route: Sm120ForcedRoute,
        operands: Sm120LaunchOperands,
        tensor_maps_digest: Sha256Digest,
    ) -> Sha256Digest {
        let mut digest = self
            .output
            .append_digest(FramedSha256::new(b"sm120-launch-resources.v1"))
            .required(b"tensor-maps", &tensor_maps_digest)
            .required(b"m", &(route.shape.m as u64).to_le_bytes())
            .required(b"k", &(route.shape.k as u64).to_le_bytes())
            .required(b"n", &(route.shape.n as u64).to_le_bytes())
            .required(b"lda", &(route.shape.lda as u64).to_le_bytes())
            .required(b"ldb", &(route.shape.ldb as u64).to_le_bytes())
            .required(b"ldc", &(route.shape.ldc as u64).to_le_bytes())
            .required(b"tile-m", &route.physical.tile.output_rows().to_le_bytes())
            .required(
                b"tile-n",
                &route.physical.tile.output_columns().to_le_bytes(),
            )
            .required(b"bk", &route.physical.bk.elements().to_le_bytes())
            .required(b"stages", &[route.physical.stages.count()])
            .required(b"output-pointer", &operands.output_ptr.to_le_bytes())
            .required(b"bias-pointer", &operands.bias_ptr.to_le_bytes())
            .required(b"alpha", &operands.alpha.to_bits().to_le_bytes())
            .required(b"beta", &operands.beta.to_bits().to_le_bytes());
        if let Some(bias) = self.bias {
            digest = bias.append_digest(digest);
        }
        digest.finish()
    }
}

fn sm120_output_layout(route: Sm120ForcedRoute) -> (usize, usize, u64) {
    match route.op {
        Sm120Op::Nn => (route.shape.m, route.shape.n, 2),
        Sm120Op::Tn => (route.shape.k, route.shape.n, 4),
        Sm120Op::Nt => (route.shape.m, route.shape.k, 2),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm120PreparedLaunch {
    pub(super) stream_handle: usize,
    pub(super) route: Sm120ForcedRoute,
    pub(super) maps: Sm120PreparedTensorMaps,
    pub(super) operands: Sm120LaunchOperands,
    pub(super) params: Sm120KernelParams,
    pub(super) identity: Sm120RouteIdentity,
    pub(super) resolved_launch_set: ResolvedGemmLaunchSet,
    pub(super) resources: Sm120LaunchResourceSnapshot,
    pub(super) kernel_resources: Sm120KernelResources,
}

impl Sm120PreparedLaunch {
    pub fn identity(&self) -> Sm120RouteIdentity {
        self.identity
    }

    pub fn resources(&self) -> Sm120KernelResources {
        self.kernel_resources
    }

    pub fn resolved_launch_set(&self) -> ResolvedGemmLaunchSet {
        self.resolved_launch_set
    }
}

#[cfg(test)]
mod tests {
    use super::{
        F32TriadShape, GemmDims, SM120_KERNEL_SPECS, Sm90aMapRequest, Sm90aOp, Sm90aShape,
        Sm100MapRequest, Sm100Op, Sm100Shape, Sm100Tile, Sm120Bk, Sm120Op, Sm120Stages, Sm120Tile,
        checked_grid_product, checked_u32, sm90a_tensor_map_keys, sm100_operand_layouts,
        sm100_tensor_map_keys, tf32_kernel_spec, tf32_route_specs, validate_bias_preseed,
        validate_sm100_issued_coordinates,
    };
    use crate::mamba_ssm::gpu::dtype::WeightDtype;
    use crate::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, ResolvedGemmOp, ResolvedInstructionFamily, ResolvedInstructionShape,
        ResolvedOperandConversion,
    };

    #[test]
    fn f32_zero_reduction_validation_is_op_normalized() {
        let cases = [
            (
                ResolvedGemmOp::Nn,
                F32TriadShape::contiguous(ResolvedGemmOp::Nn, (3, 0, 5)),
            ),
            (
                ResolvedGemmOp::Tn,
                F32TriadShape::contiguous(ResolvedGemmOp::Tn, (0, 3, 5)),
            ),
            (
                ResolvedGemmOp::Nt,
                F32TriadShape::contiguous(ResolvedGemmOp::Nt, (3, 5, 0)),
            ),
        ];
        for (op, shape) in cases {
            shape.validate(op).unwrap();
            assert_eq!(shape.reduction(op), 0);
            assert_eq!(shape.output_rows(op), 3);
            assert_eq!(shape.output_columns(op), 5);
        }

        for (op, dims) in [
            (ResolvedGemmOp::Nn, (0, 1, 1)),
            (ResolvedGemmOp::Nn, (1, 1, 0)),
            (ResolvedGemmOp::Tn, (1, 0, 1)),
            (ResolvedGemmOp::Tn, (1, 1, 0)),
            (ResolvedGemmOp::Nt, (0, 1, 1)),
            (ResolvedGemmOp::Nt, (1, 0, 1)),
        ] {
            assert!(F32TriadShape::contiguous(op, dims).validate(op).is_err());
        }
    }

    #[test]
    fn f32_shape_rejects_bad_strides_and_i32_overflow() {
        let mut nn = F32TriadShape::contiguous(ResolvedGemmOp::Nn, (7, 11, 13));
        nn.lda = 10;
        assert!(nn.validate(ResolvedGemmOp::Nn).is_err());

        let mut tn = F32TriadShape::contiguous(ResolvedGemmOp::Tn, (7, 11, 13));
        tn.ldc = 12;
        assert!(tn.validate(ResolvedGemmOp::Tn).is_err());

        let mut nt = F32TriadShape::contiguous(ResolvedGemmOp::Nt, (7, 11, 13));
        nt.ldc = 10;
        assert!(nt.validate(ResolvedGemmOp::Nt).is_err());

        let too_large = i32::MAX as usize + 1;
        let oversized = F32TriadShape::contiguous(ResolvedGemmOp::Nn, (too_large, 0, 1));
        assert!(oversized.validate(ResolvedGemmOp::Nn).is_err());
    }

    #[test]
    fn tf32_route_spec_inventories_are_exact_and_unique() {
        let expected = [
            (ModuleKind::TriadSm80, 15),
            (ModuleKind::TriadSm90a, 6),
            (ModuleKind::TriadSm100, 36),
            (ModuleKind::TriadSm120, 12),
        ];
        let mut all_symbols = std::collections::BTreeSet::new();
        for (module_kind, count) in expected {
            let specs = tf32_route_specs(module_kind);
            assert_eq!(specs.len(), count, "{module_kind:?}");
            for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
                assert_eq!(
                    specs.iter().filter(|spec| spec.op == op).count(),
                    count / 3,
                    "{module_kind:?}/{op:?}"
                );
            }
            for spec in specs {
                assert_eq!(spec.module_kind, module_kind);
                assert_eq!(tf32_kernel_spec(spec.op, spec.route).unwrap(), spec);
                match spec.route {
                    super::Tf32PhysicalRoute::MmaTf32RnaV1(_)
                    | super::Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_) => {
                        assert_eq!(spec.instruction_family, ResolvedInstructionFamily::MmaSync);
                        assert_eq!(
                            spec.instruction_shape,
                            ResolvedInstructionShape { m: 16, n: 8, k: 8 }
                        );
                    }
                    super::Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(_) => {
                        assert_eq!(spec.instruction_family, ResolvedInstructionFamily::Wgmma);
                        assert_eq!(
                            spec.instruction_shape,
                            ResolvedInstructionShape {
                                m: 64,
                                n: 128,
                                k: 8,
                            }
                        );
                    }
                    super::Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(_) => {
                        assert_eq!(spec.instruction_family, ResolvedInstructionFamily::Tcgen05);
                        assert_eq!(spec.instruction_shape.k, 8);
                        assert_eq!(
                            (
                                u32::from(spec.instruction_shape.m),
                                u32::from(spec.instruction_shape.n)
                            ),
                            spec.tile
                        );
                    }
                }
                assert_ne!(spec.operand_conversion, ResolvedOperandConversion::None);
                assert!(all_symbols.insert(spec.symbol), "duplicate {}", spec.symbol);
            }
        }
        assert_eq!(all_symbols.len(), 69);
        assert!(tf32_route_specs(ModuleKind::Fixed).is_empty());
        assert!(tf32_route_specs(ModuleKind::TriadScalar).is_empty());
        assert!(tf32_route_specs(ModuleKind::Mamba3Combined).is_empty());
    }

    fn sm90a_request(op: Sm90aOp) -> Sm90aMapRequest {
        Sm90aMapRequest {
            op,
            dtype: WeightDtype::Bf16,
            a_ptr: 0x1000,
            b_ptr: 0x2000,
            shape: Sm90aShape {
                m: 65,
                k: 127,
                n: 129,
                lda: if op == Sm90aOp::Nt { 136 } else { 128 },
                ldb: 136,
                ldc: if op == Sm90aOp::Nt { 130 } else { 132 },
            },
        }
    }

    fn sm100_request(op: Sm100Op, tile: Sm100Tile) -> Sm100MapRequest {
        Sm100MapRequest {
            op,
            dtype: WeightDtype::Bf16,
            tile,
            a_ptr: 0x1010,
            b_ptr: 0x2020,
            shape: Sm100Shape {
                m: 65,
                k: 127,
                n: 129,
                lda: if op == Sm100Op::Nt { 136 } else { 128 },
                ldb: 136,
                ldc: if op == Sm100Op::Nt { 128 } else { 136 },
            },
        }
    }

    fn assert_invalid_error(error: String) {
        assert!(error.starts_with("invalid GEMM dimensions"), "{error}");
        assert!(!error.starts_with("UNCOVERED"), "{error}");
    }

    fn assert_invalid(result: Result<GemmDims, String>) {
        assert_invalid_error(result.expect_err("dimensions must be rejected"));
    }

    #[test]
    fn gemm_dims_reject_zero_axes() {
        for dims in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert_invalid(GemmDims::nn(dims, dims.1.max(1)));
        }
    }

    #[test]
    fn gemm_dims_accept_i32_max_boundary() {
        let limit = i32::MAX as usize;
        let dims = GemmDims::checked(limit, 1, 1, 1, 1, 1).unwrap();

        assert_eq!(dims.m_i32, i32::MAX);
        assert_eq!(dims.mk, limit);
        assert_eq!(dims.mn, limit);
        assert_eq!(dims.kn, 1);
    }

    #[test]
    fn gemm_dims_reject_axis_above_i32_max() {
        let too_large = i32::MAX as usize + 1;
        assert_invalid(GemmDims::checked(too_large, 1, 1, 1, 1, 1));
    }

    #[test]
    fn gemm_dims_reject_product_overflow() {
        assert_invalid(GemmDims::checked(usize::MAX, 2, 1, 2, 1, 1));
    }

    #[test]
    fn gemm_dims_reject_device_total_overflow() {
        let limit = i32::MAX as usize;
        assert_invalid(GemmDims::checked(limit, 2, 1, 2, 1, 1));
    }

    #[test]
    fn gemm_dims_reject_grid_conversion_overflow() {
        assert_invalid_error(
            checked_u32(u32::MAX as usize + 1, "grid axis")
                .expect_err("an oversized grid axis must be rejected"),
        );
        assert_invalid_error(
            checked_grid_product(u32::MAX, 2, 1)
                .expect_err("an overflowing grid product must be rejected"),
        );
    }

    #[test]
    fn sm100_i32_max_host_and_device_tile_math_agree() {
        let extent = i32::MAX as u32;
        for tile in [64_u32, 128] {
            let host_tiles = extent.div_ceil(tile);
            let device_tiles = 1 + (extent - 1) / tile;
            assert_eq!(host_tiles, device_tiles);
            let last_origin = (device_tiles - 1) * tile;
            assert!(last_origin <= extent);
            assert!(i32::try_from(last_origin).is_ok());
        }
    }

    #[test]
    fn sm90a_map_keys_freeze_each_operand_layout() {
        let nn = sm90a_tensor_map_keys(sm90a_request(Sm90aOp::Nn)).unwrap();
        assert_eq!(nn[0].global_dimensions, [127, 65]);
        assert_eq!(nn[0].outer_byte_stride, 256);
        assert_eq!(nn[0].box_dimensions, [64, 64]);
        assert_eq!(nn[1].global_dimensions, [129, 127]);
        assert_eq!(nn[1].outer_byte_stride, 272);
        assert_eq!(nn[1].box_dimensions, [64, 64]);

        let tn = sm90a_tensor_map_keys(sm90a_request(Sm90aOp::Tn)).unwrap();
        assert_eq!(tn[0].global_dimensions, [127, 65]);
        assert_eq!(tn[0].outer_byte_stride, 256);
        assert_eq!(tn[0].box_dimensions, [64, 64]);
        assert_eq!(tn[1].global_dimensions, [129, 65]);
        assert_eq!(tn[1].outer_byte_stride, 272);
        assert_eq!(tn[1].box_dimensions, [64, 64]);

        let nt = sm90a_tensor_map_keys(sm90a_request(Sm90aOp::Nt)).unwrap();
        assert_eq!(nt[0].global_dimensions, [129, 65]);
        assert_eq!(nt[0].outer_byte_stride, 272);
        assert_eq!(nt[0].box_dimensions, [64, 64]);
        assert_eq!(nt[1].global_dimensions, [129, 127]);
        assert_eq!(nt[1].outer_byte_stride, 272);
        assert_eq!(nt[1].box_dimensions, [64, 128]);
    }

    #[test]
    fn sm90a_map_keys_reject_bad_strides_and_shapes() {
        let mut request = sm90a_request(Sm90aOp::Nn);
        request.shape.lda = 127;
        let error = sm90a_tensor_map_keys(request).unwrap_err();
        assert!(error.contains("outer byte stride"), "{error}");

        request = sm90a_request(Sm90aOp::Nn);
        request.shape.ldb = 129;
        let error = sm90a_tensor_map_keys(request).unwrap_err();
        assert!(error.contains("outer byte stride"), "{error}");

        request = sm90a_request(Sm90aOp::Nn);
        request.shape.m = 0;
        assert!(sm90a_tensor_map_keys(request).is_err());

        request = sm90a_request(Sm90aOp::Nn);
        request.shape.n = i32::MAX as usize + 1;
        assert!(sm90a_tensor_map_keys(request).is_err());
    }

    #[test]
    fn sm100_map_keys_freeze_each_operand_layout_and_tile() {
        let nn = sm100_tensor_map_keys(sm100_request(Sm100Op::Nn, Sm100Tile::M128N64)).unwrap();
        assert_eq!(nn[0].global_dimensions, [127, 65]);
        assert_eq!(nn[0].outer_byte_stride, 256);
        assert_eq!(nn[0].box_dimensions, [64, 128]);
        assert_eq!(nn[1].global_dimensions, [129, 127]);
        assert_eq!(nn[1].outer_byte_stride, 272);
        assert_eq!(nn[1].box_dimensions, [64, 64]);

        let tn = sm100_tensor_map_keys(sm100_request(Sm100Op::Tn, Sm100Tile::M128N128)).unwrap();
        assert_eq!(tn[0].global_dimensions, [127, 65]);
        assert_eq!(tn[0].box_dimensions, [64, 64]);
        assert_eq!(tn[1].global_dimensions, [129, 65]);
        assert_eq!(tn[1].box_dimensions, [64, 64]);

        let nt64 = sm100_tensor_map_keys(sm100_request(Sm100Op::Nt, Sm100Tile::M128N64)).unwrap();
        assert_eq!(nt64[0].global_dimensions, [129, 65]);
        assert_eq!(nt64[0].outer_byte_stride, 272);
        assert_eq!(nt64[0].box_dimensions, [64, 128]);
        assert_eq!(nt64[1].global_dimensions, [129, 127]);
        assert_eq!(nt64[1].outer_byte_stride, 272);
        assert_eq!(nt64[1].box_dimensions, [64, 64]);

        let nt128 = sm100_tensor_map_keys(sm100_request(Sm100Op::Nt, Sm100Tile::M128N128)).unwrap();
        assert_eq!(nt128[1].box_dimensions, [64, 128]);
    }

    #[test]
    fn sm100_map_keys_reject_bad_strides_dtypes_and_shapes() {
        let mut request = sm100_request(Sm100Op::Nn, Sm100Tile::M128N64);
        request.shape.lda = 127;
        let error = sm100_tensor_map_keys(request).unwrap_err();
        assert!(error.contains("byte stride"), "{error}");

        request = sm100_request(Sm100Op::Nt, Sm100Tile::M128N128);
        request.shape.ldc = 126;
        assert!(sm100_tensor_map_keys(request).is_err());

        request = sm100_request(Sm100Op::Nn, Sm100Tile::M128N64);
        request.dtype = WeightDtype::F32;
        assert!(sm100_tensor_map_keys(request).is_err());

        for axis in 0..3 {
            request = sm100_request(Sm100Op::Tn, Sm100Tile::M128N64);
            match axis {
                0 => request.shape.m = 0,
                1 => request.shape.k = 0,
                _ => request.shape.n = 0,
            }
            assert!(sm100_tensor_map_keys(request).is_err());
        }
    }

    #[test]
    fn sm100_subview_origins_cover_secondary_tma_coordinates() {
        let tn = sm100_operand_layouts(sm100_request(Sm100Op::Tn, Sm100Tile::M128N64));
        validate_sm100_issued_coordinates(tn[0], (i32::MAX as u64 - 64, 0)).unwrap();
        let error = validate_sm100_issued_coordinates(tn[0], (i32::MAX as u64 - 63, 0))
            .expect_err("TN's secondary A coordinate must remain representable");
        assert!(
            error.contains("issued x coordinate exceeds i32::MAX"),
            "{error}"
        );

        let nn = sm100_operand_layouts(sm100_request(Sm100Op::Nn, Sm100Tile::M128N128));
        assert_eq!(nn[1].issued_coordinate_max[0], 192);
    }

    #[test]
    fn sm120_inventory_covers_all_ninety_six_physical_routes_once() {
        assert_eq!(SM120_KERNEL_SPECS.len(), 96);
        let mut symbols = std::collections::BTreeSet::new();
        for spec in SM120_KERNEL_SPECS {
            assert!(symbols.insert(spec.symbol), "duplicate {}", spec.symbol);
            assert!(spec.symbol.starts_with("sgemm_bi_"), "{}", spec.symbol);
        }

        for op in [Sm120Op::Nn, Sm120Op::Tn, Sm120Op::Nt] {
            for dtype in [WeightDtype::F16, WeightDtype::Bf16] {
                for tile in [
                    Sm120Tile::M64N64,
                    Sm120Tile::M128N64,
                    Sm120Tile::M64N128,
                    Sm120Tile::M128N128,
                ] {
                    for bk in [Sm120Bk::Bk32, Sm120Bk::Bk64] {
                        for stages in [Sm120Stages::S2, Sm120Stages::S3] {
                            assert_eq!(
                                SM120_KERNEL_SPECS
                                    .iter()
                                    .filter(|spec| {
                                        spec.op == op
                                            && spec.dtype == dtype
                                            && spec.physical.tile == tile
                                            && spec.physical.bk == bk
                                            && spec.physical.stages == stages
                                    })
                                    .count(),
                                1,
                                "missing or duplicate {op:?}/{dtype:?}/{tile:?}/{bk:?}/{stages:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            SM120_KERNEL_SPECS
                .iter()
                .all(|spec| spec.dtype != WeightDtype::F32)
        );
    }

    #[test]
    fn sm120_physical_metadata_matches_literal_resource_contract() {
        let expected = [
            (
                Sm120Tile::M64N64,
                Sm120Bk::Bk32,
                Sm120Stages::S2,
                128,
                16_512,
            ),
            (
                Sm120Tile::M64N64,
                Sm120Bk::Bk32,
                Sm120Stages::S3,
                128,
                24_704,
            ),
            (
                Sm120Tile::M64N64,
                Sm120Bk::Bk64,
                Sm120Stages::S2,
                128,
                32_896,
            ),
            (
                Sm120Tile::M64N64,
                Sm120Bk::Bk64,
                Sm120Stages::S3,
                128,
                49_280,
            ),
            (
                Sm120Tile::M128N64,
                Sm120Bk::Bk32,
                Sm120Stages::S2,
                256,
                24_704,
            ),
            (
                Sm120Tile::M128N64,
                Sm120Bk::Bk32,
                Sm120Stages::S3,
                256,
                36_992,
            ),
            (
                Sm120Tile::M128N64,
                Sm120Bk::Bk64,
                Sm120Stages::S2,
                256,
                49_280,
            ),
            (
                Sm120Tile::M128N64,
                Sm120Bk::Bk64,
                Sm120Stages::S3,
                256,
                73_856,
            ),
            (
                Sm120Tile::M64N128,
                Sm120Bk::Bk32,
                Sm120Stages::S2,
                256,
                24_704,
            ),
            (
                Sm120Tile::M64N128,
                Sm120Bk::Bk32,
                Sm120Stages::S3,
                256,
                36_992,
            ),
            (
                Sm120Tile::M64N128,
                Sm120Bk::Bk64,
                Sm120Stages::S2,
                256,
                49_280,
            ),
            (
                Sm120Tile::M64N128,
                Sm120Bk::Bk64,
                Sm120Stages::S3,
                256,
                73_856,
            ),
            (
                Sm120Tile::M128N128,
                Sm120Bk::Bk32,
                Sm120Stages::S2,
                512,
                32_896,
            ),
            (
                Sm120Tile::M128N128,
                Sm120Bk::Bk32,
                Sm120Stages::S3,
                512,
                49_280,
            ),
            (
                Sm120Tile::M128N128,
                Sm120Bk::Bk64,
                Sm120Stages::S2,
                512,
                65_664,
            ),
            (
                Sm120Tile::M128N128,
                Sm120Bk::Bk64,
                Sm120Stages::S3,
                512,
                98_432,
            ),
        ];

        for (tile, bk, stages, threads, shared) in expected {
            let matching: Vec<_> = SM120_KERNEL_SPECS
                .iter()
                .filter(|spec| {
                    spec.physical.tile == tile
                        && spec.physical.bk == bk
                        && spec.physical.stages == stages
                })
                .collect();
            assert_eq!(matching.len(), 6, "{tile:?}/{bk:?}/{stages:?}");
            for spec in matching {
                assert_eq!(spec.threads, threads, "{}", spec.symbol);
                assert_eq!(spec.dynamic_shared_bytes, shared, "{}", spec.symbol);
            }
        }
    }

    #[test]
    fn gemm_dims_reject_bad_nn_strides() {
        for strides in [(2, 5, 5), (3, 4, 5), (3, 5, 4), (0, 5, 5)] {
            assert_invalid(GemmDims::checked(
                2,
                strides.0.max(3),
                5,
                strides.0,
                strides.1,
                strides.2,
            ));
        }
        assert_invalid(GemmDims::checked(2, 1, 1, i32::MAX as usize, 1, 1));
        assert_invalid(GemmDims::checked(1, 1, 1, i32::MAX as usize + 1, 1, 1));
    }

    #[test]
    fn gemm_dims_preserve_tn_storage_strides() {
        let dims = GemmDims::tn((2, 3, 5)).unwrap();
        assert_eq!((dims.lda, dims.ldb, dims.ldc), (3, 5, 5));
    }

    #[test]
    fn gemm_dims_reject_bad_tn_strides() {
        assert_invalid(GemmDims::checked_storage(
            2,
            3,
            5,
            [2, 5, 5],
            [3, 5, 5],
            [2, 2, 3],
        ));
    }

    #[test]
    fn gemm_dims_preserve_nt_storage_strides() {
        let dims = GemmDims::nt((2, 3, 5)).unwrap();
        assert_eq!((dims.lda, dims.ldb, dims.ldc), (5, 5, 3));
    }

    #[test]
    fn gemm_dims_reject_bad_nt_strides() {
        assert_invalid(GemmDims::checked_storage(
            2,
            3,
            5,
            [4, 5, 3],
            [5, 5, 3],
            [2, 3, 2],
        ));
    }

    #[test]
    fn bias_preseed_rejects_non_identity_alpha() {
        let error = validate_bias_preseed(0.5, 1, "triad-test")
            .expect_err("bias pre-seeding must reject alpha != 1");
        assert!(error.contains("alpha == 1.0"), "{error}");
        validate_bias_preseed(0.5, 0, "triad-test").unwrap();
        validate_bias_preseed(1.0, 1, "triad-test").unwrap();
    }
}
