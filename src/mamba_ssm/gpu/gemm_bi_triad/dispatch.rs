use super::super::dtype::WeightDtype;
use super::super::{
    context::F32TriadPolicy,
    kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        ModuleKind, NUMERIC_ABI_REVISION, SCHEDULE_REVISION,
    },
};
use super::contract::{
    F32_TF32_TUNING_REVISION, F32TriadAvailability, F32TriadOperands, F32TriadRequest,
    F32TriadSelection, F32TriadShape, Sm90aForcedRoute, Sm90aOp, Sm90aShape,
    Sm90aWarpgroupSchedule, Sm100ForcedRoute, Sm100Op, Sm100TargetCandidate, Sm100TargetKind,
    Sm120Bk, Sm120FmaRoute, Sm120FmaTile, Sm120ForcedRoute, Sm120LaunchOperands, Sm120MapRequest,
    Sm120Op, Sm120PhysicalRoute, Sm120Shape, Sm120Stages, Sm120TargetCandidate, Sm120Tile,
    Tf32PhysicalRoute, Tf32QualifiedModule, tf32_kernel_spec, validate_sm120_map_request,
};
use super::contract::{GemmDims, checked_mul3, checked_tile_grid, checked_usize};
use crate::mamba_ssm::gpu::kernel_identity::DeviceCaps;
use crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32ExactShape {
    output_rows: usize,
    output_columns: usize,
    reduction: usize,
}

impl Tf32ExactShape {
    fn matches_contiguous(self, request: F32TriadRequest) -> bool {
        let rows = request.shape.output_rows(request.op);
        let columns = request.shape.output_columns(request.op);
        let reduction = request.shape.reduction(request.op);
        if (rows, columns, reduction) != (self.output_rows, self.output_columns, self.reduction) {
            return false;
        }
        let dims = match request.op {
            crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn => {
                (rows, reduction, columns)
            }
            crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn => {
                (reduction, rows, columns)
            }
            crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt => {
                (rows, columns, reduction)
            }
        };
        request.shape == super::contract::F32TriadShape::contiguous(request.op, dims)
    }
}

/// The archived cells record whether their exact strides use scalar or vector
/// staging. Every class still requires the concrete epilogue and aligned base
/// pointers that were present during qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tf32AutoOperandGate {
    /// TN/NT forbids bias and this exact stride tuple forces scalar staging,
    /// so four-byte-aligned production pointers stay in the measured path.
    RequestContractSafe,
    /// NN permits bias, but the archived point measured `bias=None` only.
    RequiresNoBiasEvidence,
    /// The route can stage asynchronously, but pointer-offset evidence is absent.
    RequiresVectorAlignmentEvidence,
    /// Both bias and pointer-alignment evidence are absent.
    RequiresNoBiasAndVectorAlignmentEvidence,
}

fn tf32_auto_operands_match(op: ResolvedGemmOp, operands: F32TriadOperands) -> bool {
    let epilogue_matches = match op {
        ResolvedGemmOp::Nn | ResolvedGemmOp::Nt => {
            operands.alpha.to_bits() == 1.0_f32.to_bits()
                && operands.beta.to_bits() == 0.0_f32.to_bits()
                && operands.bias.is_none()
        }
        ResolvedGemmOp::Tn => {
            operands.alpha.to_bits() == 1.0_f32.to_bits()
                && operands.beta.to_bits() == 1.0_f32.to_bits()
                && operands.bias.is_none()
        }
    };
    epilogue_matches
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(16))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32AutoQualificationIdentity {
    module_kind: ModuleKind,
    module_target: &'static str,
    device_target: &'static str,
    compute_capability: (u32, u32),
    multiprocessor_count: u32,
    nvrtc_version: (i32, i32),
    driver_api_version: i32,
    driver_build_sources: u8,
    optin_shared_bytes: u32,
    tensor_map_access: bool,
    compile_key: [u8; 32],
    artifact_digest: [u8; 32],
    source_digest: [u8; 32],
    invocation_digest: [u8; 32],
    header_manifest_digest: [u8; 32],
    nvrtc_library_domain: [u8; 32],
    driver_build_digest: [u8; 32],
}

impl Tf32AutoQualificationIdentity {
    fn matches(self, module: Tf32QualifiedModule) -> bool {
        self.mismatch(module).is_none()
    }

    /// The first identity field the bound module does not satisfy, or `None`
    /// when the cohort applies to it. A decline on this path used to be a bare
    /// `None`; the field name is what tells a wrong board, a moved toolkit and
    /// an edited kernel apart.
    fn mismatch(self, module: Tf32QualifiedModule) -> Option<&'static str> {
        let checks: [(&'static str, bool); 31] = [
            ("module kind", module.module_kind == self.module_kind),
            (
                "module target",
                module.target.as_str() == self.module_target,
            ),
            (
                "artifact module kind",
                module.artifact.module_kind == self.module_kind,
            ),
            (
                "artifact kind",
                module.artifact.artifact_kind == ArtifactKind::Ptx,
            ),
            (
                "compile key",
                module.artifact.compile_key == self.compile_key,
            ),
            (
                "artifact digest",
                module.artifact.artifact_digest == self.artifact_digest,
            ),
            (
                "source digest",
                module.compiler.source_digest == self.source_digest,
            ),
            (
                "invocation digest",
                module.compiler.invocation_digest == self.invocation_digest,
            ),
            (
                "header manifest digest",
                module.compiler.header_manifest_digest == self.header_manifest_digest,
            ),
            (
                "compiler target",
                module.compiler.target.as_str() == self.module_target,
            ),
            (
                "nvrtc version",
                module.compiler.nvrtc_version == self.nvrtc_version,
            ),
            (
                "nvrtc library domain",
                module.compiler.nvrtc_library_domain == self.nvrtc_library_domain,
            ),
            ("nvrtc library known", module.compiler.nvrtc_library_known),
            (
                "compiler output kind",
                module.compiler.output_kind == ArtifactKind::Ptx,
            ),
            (
                "composer revision",
                module.compiler.composer_revision == COMPOSER_REVISION,
            ),
            (
                "compiler revision",
                module.compiler.compiler_revision == COMPILER_REVISION,
            ),
            (
                "numeric abi revision",
                module.compiler.numeric_abi_revision == NUMERIC_ABI_REVISION,
            ),
            (
                "schedule revision",
                module.compiler.schedule_revision == SCHEDULE_REVISION,
            ),
            (
                "device compute capability",
                module.device.compute_capability == self.compute_capability,
            ),
            (
                "multiprocessor count",
                module.device.multiprocessor_count == self.multiprocessor_count,
            ),
            (
                "device target",
                module.device.target.as_str() == self.device_target,
            ),
            (
                "driver api version",
                module.device.driver.api_version == self.driver_api_version,
            ),
            (
                "driver build sources",
                module.device.driver.build_sources == self.driver_build_sources,
            ),
            (
                "driver build digest",
                module.device.driver.build_digest == self.driver_build_digest,
            ),
            (
                "caps compute capability",
                module.device_caps.compute_capability == self.compute_capability,
            ),
            (
                "caps nvrtc version",
                module.device_caps.nvrtc_version == self.nvrtc_version,
            ),
            (
                "accepted target",
                module
                    .device_caps
                    .accepted_target
                    .is_some_and(|target| target.as_str() == self.module_target),
            ),
            (
                "opt-in shared bytes",
                module.device_caps.optin_shared_bytes == self.optin_shared_bytes,
            ),
            (
                "tensor map access",
                module.device_caps.tensor_map_access == self.tensor_map_access,
            ),
            ("cohort compute capability", true),
            ("cohort multiprocessor count", true),
        ];
        checks
            .into_iter()
            .find(|(_, holds)| !holds)
            .map(|(field, _)| field)
    }
}

const SM89_TF32_QUALIFIED_TUNING_REVISION: u16 = F32_TF32_TUNING_REVISION;
const SM89_TF32_QUALIFICATION_IDENTITY: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm80,
        module_target: "sm_89",
        device_target: "sm_89",
        compute_capability: (8, 9),
        multiprocessor_count: 142,
        nvrtc_version: (13, 2),
        driver_api_version: 13_020,
        driver_build_sources: 7,
        optin_shared_bytes: 101_376,
        tensor_map_access: false,
        compile_key: [
            200, 155, 40, 187, 13, 40, 216, 221, 209, 116, 242, 247, 73, 127, 161, 55, 56, 132, 90,
            93, 215, 45, 64, 180, 36, 103, 141, 71, 68, 237, 217, 207,
        ],
        artifact_digest: [
            248, 184, 177, 29, 225, 252, 108, 18, 6, 155, 40, 243, 158, 125, 57, 69, 204, 35, 73,
            234, 102, 238, 121, 182, 218, 142, 17, 200, 185, 70, 157, 2,
        ],
        source_digest: [
            123, 6, 93, 13, 99, 156, 70, 127, 222, 185, 224, 146, 228, 74, 32, 36, 95, 28, 144,
            127, 173, 220, 249, 144, 110, 139, 173, 76, 241, 66, 192, 127,
        ],
        invocation_digest: [
            200, 155, 40, 187, 13, 40, 216, 221, 209, 116, 242, 247, 73, 127, 161, 55, 56, 132, 90,
            93, 215, 45, 64, 180, 36, 103, 141, 71, 68, 237, 217, 207,
        ],
        header_manifest_digest: [
            10, 239, 1, 191, 15, 19, 213, 142, 41, 133, 167, 118, 99, 219, 167, 154, 128, 174, 212,
            209, 216, 181, 220, 151, 93, 30, 190, 48, 22, 197, 192, 110,
        ],
        nvrtc_library_domain: [
            208, 49, 165, 62, 185, 114, 53, 183, 15, 98, 246, 82, 147, 45, 177, 189, 247, 40, 234,
            34, 156, 140, 168, 9, 213, 60, 95, 253, 145, 100, 38, 135,
        ],
        driver_build_digest: [
            209, 237, 197, 165, 188, 62, 16, 162, 104, 142, 33, 86, 141, 204, 189, 229, 226, 128,
            220, 57, 20, 77, 216, 30, 133, 56, 67, 202, 152, 178, 208, 225,
        ],
    };

const SM120_TF32_QUALIFIED_TUNING_REVISION: u16 = F32_TF32_TUNING_REVISION;
const SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm80,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 2),
        driver_api_version: 13_020,
        driver_build_sources: 7,
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
        compile_key: [
            152, 216, 153, 132, 10, 117, 104, 1, 208, 62, 15, 125, 19, 15, 218, 83, 180, 144, 4,
            144, 211, 131, 159, 120, 140, 116, 131, 206, 41, 62, 77, 183,
        ],
        artifact_digest: [
            240, 103, 177, 251, 174, 24, 81, 83, 102, 109, 200, 106, 209, 179, 67, 191, 217, 255,
            129, 130, 70, 139, 203, 226, 105, 153, 35, 119, 142, 224, 129, 125,
        ],
        source_digest: [
            123, 6, 93, 13, 99, 156, 70, 127, 222, 185, 224, 146, 228, 74, 32, 36, 95, 28, 144,
            127, 173, 220, 249, 144, 110, 139, 173, 76, 241, 66, 192, 127,
        ],
        invocation_digest: [
            152, 216, 153, 132, 10, 117, 104, 1, 208, 62, 15, 125, 19, 15, 218, 83, 180, 144, 4,
            144, 211, 131, 159, 120, 140, 116, 131, 206, 41, 62, 77, 183,
        ],
        header_manifest_digest: [
            10, 239, 1, 191, 15, 19, 213, 142, 41, 133, 167, 118, 99, 219, 167, 154, 128, 174, 212,
            209, 216, 181, 220, 151, 93, 30, 190, 48, 22, 197, 192, 110,
        ],
        nvrtc_library_domain: [
            14, 13, 195, 250, 169, 151, 174, 150, 68, 46, 246, 47, 252, 2, 100, 13, 54, 26, 92, 80,
            224, 180, 229, 223, 42, 5, 188, 9, 134, 254, 98, 65,
        ],
        driver_build_digest: [
            187, 232, 57, 127, 110, 241, 26, 80, 106, 80, 45, 18, 125, 94, 184, 39, 69, 165, 21,
            171, 100, 249, 178, 144, 29, 171, 78, 131, 75, 241, 144, 216,
        ],
    };
const SM120_TF32_QUALIFICATION_IDENTITY_CUDA_12_8: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (12, 8),
        driver_api_version: 13_020,
        driver_build_sources: 7,
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
        compile_key: [
            166, 206, 169, 76, 240, 169, 84, 100, 217, 7, 12, 212, 25, 4, 118, 118, 179, 222, 227,
            156, 49, 139, 185, 110, 235, 81, 26, 54, 11, 54, 128, 33,
        ],
        artifact_digest: [
            91, 106, 28, 188, 79, 11, 154, 143, 66, 25, 178, 253, 11, 152, 42, 219, 121, 139, 185,
            24, 232, 18, 21, 222, 72, 132, 54, 144, 222, 61, 247, 171,
        ],
        source_digest: [
            36, 109, 46, 53, 235, 5, 156, 214, 150, 236, 116, 50, 93, 74, 34, 51, 18, 250, 96, 95,
            232, 79, 245, 143, 23, 246, 99, 141, 153, 184, 24, 44,
        ],
        invocation_digest: [
            166, 206, 169, 76, 240, 169, 84, 100, 217, 7, 12, 212, 25, 4, 118, 118, 179, 222, 227,
            156, 49, 139, 185, 110, 235, 81, 26, 54, 11, 54, 128, 33,
        ],
        header_manifest_digest: [
            254, 128, 56, 210, 41, 106, 84, 102, 144, 148, 104, 242, 181, 2, 247, 203, 59, 245, 94,
            236, 49, 74, 64, 150, 164, 59, 2, 159, 109, 139, 29, 48,
        ],
        nvrtc_library_domain: [
            38, 176, 163, 160, 32, 68, 255, 203, 193, 105, 63, 216, 62, 146, 97, 190, 255, 166,
            146, 164, 251, 207, 227, 172, 94, 157, 140, 135, 152, 11, 177, 85,
        ],
        driver_build_digest: [
            187, 232, 57, 127, 110, 241, 26, 80, 106, 80, 45, 18, 125, 94, 184, 39, 69, 165, 21,
            171, 100, 249, 178, 144, 29, 171, 78, 131, 75, 241, 144, 216,
        ],
    };

const SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_0: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 0),
        driver_api_version: 13_020,
        driver_build_sources: 7,
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
        compile_key: [
            218, 36, 138, 227, 73, 184, 235, 221, 246, 237, 231, 11, 39, 108, 209, 191, 166, 32,
            167, 221, 251, 169, 63, 201, 34, 81, 127, 134, 136, 166, 71, 79,
        ],
        artifact_digest: [
            81, 129, 108, 141, 121, 6, 25, 109, 73, 170, 104, 7, 116, 44, 14, 147, 172, 23, 50,
            242, 58, 150, 71, 46, 170, 185, 125, 196, 70, 239, 181, 184,
        ],
        source_digest: [
            36, 109, 46, 53, 235, 5, 156, 214, 150, 236, 116, 50, 93, 74, 34, 51, 18, 250, 96, 95,
            232, 79, 245, 143, 23, 246, 99, 141, 153, 184, 24, 44,
        ],
        invocation_digest: [
            218, 36, 138, 227, 73, 184, 235, 221, 246, 237, 231, 11, 39, 108, 209, 191, 166, 32,
            167, 221, 251, 169, 63, 201, 34, 81, 127, 134, 136, 166, 71, 79,
        ],
        header_manifest_digest: [
            244, 255, 248, 65, 139, 210, 195, 70, 200, 110, 198, 240, 127, 94, 116, 23, 20, 124, 1,
            59, 240, 56, 205, 193, 109, 243, 231, 108, 243, 248, 10, 217,
        ],
        nvrtc_library_domain: [
            112, 155, 145, 195, 107, 251, 14, 217, 102, 238, 105, 173, 200, 214, 248, 127, 241, 16,
            238, 207, 61, 251, 80, 96, 54, 127, 24, 60, 230, 20, 235, 13,
        ],
        driver_build_digest: [
            187, 232, 57, 127, 110, 241, 26, 80, 106, 80, 45, 18, 125, 94, 184, 39, 69, 165, 21,
            171, 100, 249, 178, 144, 29, 171, 78, 131, 75, 241, 144, 216,
        ],
    };

/// Frozen CUDA 13.2 qualification identity.
const SM120_TF32_QUALIFICATION_IDENTITY: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 2),
        driver_api_version: 13_020,
        driver_build_sources: 7,
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
        compile_key: [
            112, 238, 69, 136, 32, 242, 124, 85, 78, 231, 125, 63, 137, 201, 144, 207, 0, 93, 126,
            62, 86, 109, 196, 135, 234, 127, 136, 146, 46, 195, 54, 69,
        ],
        artifact_digest: [
            80, 206, 171, 198, 77, 105, 232, 87, 90, 121, 116, 211, 68, 150, 5, 152, 12, 177, 180,
            45, 222, 248, 149, 42, 206, 55, 37, 7, 60, 217, 4, 65,
        ],
        source_digest: [
            101, 223, 188, 193, 100, 34, 155, 115, 243, 247, 111, 161, 80, 22, 226, 138, 58, 223,
            154, 210, 50, 128, 48, 235, 31, 230, 217, 126, 227, 208, 26, 175,
        ],
        invocation_digest: [
            112, 238, 69, 136, 32, 242, 124, 85, 78, 231, 125, 63, 137, 201, 144, 207, 0, 93, 126,
            62, 86, 109, 196, 135, 234, 127, 136, 146, 46, 195, 54, 69,
        ],
        header_manifest_digest: [
            144, 90, 202, 198, 154, 11, 239, 32, 177, 45, 241, 189, 47, 183, 11, 111, 139, 211,
            143, 197, 48, 236, 2, 191, 245, 242, 177, 36, 191, 141, 145, 87,
        ],
        nvrtc_library_domain: [
            14, 13, 195, 250, 169, 151, 174, 150, 68, 46, 246, 47, 252, 2, 100, 13, 54, 26, 92, 80,
            224, 180, 229, 223, 42, 5, 188, 9, 134, 254, 98, 65,
        ],
        driver_build_digest: [
            198, 167, 186, 45, 24, 253, 128, 109, 152, 54, 59, 29, 126, 209, 127, 60, 52, 112, 16,
            227, 178, 168, 2, 40, 61, 220, 25, 31, 187, 87, 122, 188,
        ],
    };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32AutoCell {
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    shape: Tf32ExactShape,
    route: Tf32PhysicalRoute,
    tuning_revision: u16,
    operand_gate: Tf32AutoOperandGate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tf32AutoEvidenceCohort {
    identity: Tf32AutoQualificationIdentity,
    cells: &'static [Tf32AutoCell],
}

const fn sm89_tf32_cell(
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    output_rows: usize,
    output_columns: usize,
    reduction: usize,
    tile: super::contract::Tf32PortableTile,
    stages: super::contract::Tf32PortableStages,
    operand_gate: Tf32AutoOperandGate,
) -> Tf32AutoCell {
    Tf32AutoCell {
        op,
        shape: Tf32ExactShape {
            output_rows,
            output_columns,
            reduction,
        },
        route: Tf32PhysicalRoute::MmaTf32RnaV1(super::contract::Tf32PortableRoute { tile, stages }),
        tuning_revision: SM89_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate,
    }
}

/// A measured SM89 cell whose winner is not the plain portable route: the
/// split-K arms are physical routes of the same portable family.
const fn sm89_tf32_route_cell(
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    dims: (usize, usize, usize),
    route: Tf32PhysicalRoute,
    operand_gate: Tf32AutoOperandGate,
) -> Tf32AutoCell {
    let (output_rows, output_columns, reduction) = dims;
    Tf32AutoCell {
        op,
        shape: Tf32ExactShape {
            output_rows,
            output_columns,
            reduction,
        },
        route,
        tuning_revision: SM89_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate,
    }
}

use super::contract::Tf32PortableStages::{S2, S3, S4};
use super::contract::Tf32PortableTile::{M16N32, M64N64, M128N64};
use crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::{Nn, Nt, Tn};
use Tf32AutoOperandGate::{
    RequestContractSafe, RequiresNoBiasAndVectorAlignmentEvidence, RequiresNoBiasEvidence,
    RequiresVectorAlignmentEvidence,
};

/// Exact SM89 TF32 evidence inventory. Production preparation supplies the
/// concrete epilogue and pointers required to match an archived cell. The
/// request-only API therefore stays on the scalar route.
const SM89_TF32_EVIDENCE_CELLS: &[Tf32AutoCell] = &[
    sm89_tf32_cell(
        Nn,
        16,
        2048,
        512,
        M16N32,
        S4,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(Nn, 49, 129, 65, M16N32, S4, RequiresNoBiasEvidence),
    sm89_tf32_cell(Nn, 65, 129, 49, M16N32, S4, RequiresNoBiasEvidence),
    sm89_tf32_cell(Nn, 129, 100, 131, M16N32, S4, RequiresNoBiasEvidence),
    sm89_tf32_cell(
        Nn,
        512,
        768,
        3072,
        M64N64,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        1024,
        128,
        256,
        M16N32,
        S4,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        1024,
        512,
        128,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        M128N64,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        2048,
        768,
        3072,
        M128N64,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        4096,
        768,
        512,
        M128N64,
        S3,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        4096,
        1536,
        3072,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        16,
        2048,
        512,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(Tn, 49, 129, 65, M16N32, S4, RequestContractSafe),
    sm89_tf32_cell(Tn, 65, 129, 49, M16N32, S4, RequestContractSafe),
    sm89_tf32_cell(
        Tn,
        128,
        512,
        1024,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(Tn, 131, 100, 129, M16N32, S4, RequestContractSafe),
    sm89_tf32_cell(
        Tn,
        256,
        128,
        1024,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        512,
        384,
        256,
        M64N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        3072,
        768,
        512,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        8192,
        128,
        128,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        16,
        512,
        2048,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(Nt, 49, 65, 129, M16N32, S4, RequestContractSafe),
    sm89_tf32_cell(Nt, 65, 49, 129, M16N32, S4, RequestContractSafe),
    sm89_tf32_cell(
        Nt,
        128,
        8192,
        128,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        512,
        16,
        2048,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        512,
        3072,
        768,
        M64N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        1024,
        256,
        128,
        M16N32,
        S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        2048,
        3072,
        768,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        4096,
        512,
        768,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        768,
        3072,
        2048,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        1536,
        768,
        2048,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        384,
        1928,
        4621,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        3072,
        1536,
        4096,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        4096,
        3072,
        1536,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Tn,
        3072,
        768,
        2048,
        M64N64,
        S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_route_cell(
        Nt,
        (1024, 128, 512),
        Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(super::contract::Tf32PortableRoute {
            tile: M16N32,
            stages: S3,
        }),
        RequiresVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nn,
        10400,
        1536,
        384,
        M64N64,
        S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm89_tf32_cell(
        Nt,
        10400,
        384,
        1536,
        M128N64,
        S3,
        RequiresVectorAlignmentEvidence,
    ),
];

const fn sm120_tf32_streamk_cell(
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    output_rows: usize,
    output_columns: usize,
    reduction: usize,
    stages: super::contract::Tf32Sm120Stages,
    operand_gate: Tf32AutoOperandGate,
) -> Tf32AutoCell {
    Tf32AutoCell {
        op,
        shape: Tf32ExactShape {
            output_rows,
            output_columns,
            reduction,
        },
        route: Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(super::contract::Tf32Sm120Route {
            tile: super::contract::Tf32Sm120Tile::M64N128,
            stages,
        }),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate,
    }
}

const fn sm120_tf32_cell(
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    output_rows: usize,
    output_columns: usize,
    reduction: usize,
    tile: super::contract::Tf32Sm120Tile,
    stages: super::contract::Tf32Sm120Stages,
    operand_gate: Tf32AutoOperandGate,
) -> Tf32AutoCell {
    Tf32AutoCell {
        op,
        shape: Tf32ExactShape {
            output_rows,
            output_columns,
            reduction,
        },
        route: Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(super::contract::Tf32Sm120Route {
            tile,
            stages,
        }),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate,
    }
}

/// Exact CUDA 12.8 SM120 TF32 evidence inventory from the production
/// projection suite. This is intentionally literal: CUDA 12.8 and 13.2 chose
/// different routes for TN d768 input projection.
const SM120_TF32_EVIDENCE_CELLS_CUDA_12_8: &[Tf32AutoCell] = &[
    sm120_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        768,
        3072,
        2048,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        1536,
        768,
        2048,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        384,
        1928,
        4621,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S4,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
];

/// Exact CUDA 13.0 SM120 TF32 evidence inventory from the production
/// projection suite. Routes are bound to the exact 13.0 artifact and device
/// identity rather than inferred from either neighboring toolchain cohort.
const SM120_TF32_EVIDENCE_CELLS_CUDA_13_0: &[Tf32AutoCell] = &[
    sm120_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        768,
        3072,
        2048,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        1536,
        768,
        2048,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        384,
        1928,
        4621,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
];

/// Exact CUDA 13.2 SM120 TF32 evidence inventory from the production
/// projection suite. Operand-aware dispatch remains scalar unless the full
/// artifact and device identity matches this cohort.
const SM120_TF32_EVIDENCE_CELLS: &[Tf32AutoCell] = &[
    Tf32AutoCell {
        op: Tn,
        shape: Tf32ExactShape {
            output_rows: 512,
            output_columns: 384,
            reduction: 256,
        },
        route: Tf32PhysicalRoute::MmaTf32RnaV1(super::contract::Tf32PortableRoute {
            tile: M16N32,
            stages: S4,
        }),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    sm120_tf32_cell(
        Nn,
        512,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M80N32Bk64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        768,
        3072,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        1536,
        768,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        super::contract::Tf32Sm120Tile::M128N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        384,
        1928,
        4621,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
];

/// Frozen CUDA 13.2 qualification identity from the 595.84 driver build. The
/// 13.2 cohort above was taken on 595.91, and two of its nine cells chose a
/// different route there, so the two builds carry separate evidence.
const SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84: Tf32AutoQualificationIdentity =
    Tf32AutoQualificationIdentity {
        module_kind: ModuleKind::TriadSm120,
        module_target: "compute_120",
        device_target: "sm_120",
        compute_capability: (12, 0),
        multiprocessor_count: 170,
        nvrtc_version: (13, 2),
        driver_api_version: 13_020,
        driver_build_sources: 7,
        optin_shared_bytes: 101_376,
        tensor_map_access: true,
        compile_key: [
            164, 111, 255, 220, 20, 165, 216, 226, 182, 171, 219, 3, 7, 114, 247, 176, 17, 171, 94,
            27, 103, 33, 223, 150, 13, 39, 77, 154, 160, 146, 90, 252,
        ],
        artifact_digest: [
            84, 105, 236, 114, 208, 179, 3, 173, 38, 68, 104, 40, 127, 52, 184, 89, 101, 109, 115,
            160, 208, 54, 150, 90, 148, 230, 54, 213, 21, 212, 71, 227,
        ],
        source_digest: [
            125, 146, 67, 184, 153, 20, 74, 5, 79, 185, 32, 107, 196, 155, 195, 167, 59, 100, 79,
            61, 54, 211, 230, 24, 212, 155, 175, 154, 80, 133, 233, 145,
        ],
        invocation_digest: [
            164, 111, 255, 220, 20, 165, 216, 226, 182, 171, 219, 3, 7, 114, 247, 176, 17, 171, 94,
            27, 103, 33, 223, 150, 13, 39, 77, 154, 160, 146, 90, 252,
        ],
        header_manifest_digest: [
            101, 83, 218, 3, 23, 116, 215, 159, 181, 30, 190, 53, 234, 120, 114, 18, 88, 118, 25,
            76, 202, 62, 69, 221, 223, 33, 204, 218, 138, 5, 236, 133,
        ],
        nvrtc_library_domain: [
            220, 223, 96, 48, 189, 148, 19, 101, 183, 103, 158, 213, 226, 50, 226, 19, 235, 24,
            147, 98, 12, 178, 250, 224, 154, 245, 36, 3, 79, 180, 23, 212,
        ],
        driver_build_digest: [
            142, 150, 68, 239, 130, 136, 131, 5, 214, 222, 50, 93, 180, 249, 96, 244, 193, 247,
            254, 61, 169, 46, 223, 96, 180, 53, 238, 116, 173, 98, 33, 178,
        ],
    };

/// Exact SM120 TF32 evidence inventory measured on the 595.84 driver build:
/// nineteen projection cells over the 101-window selector protocol. Two of the
/// twenty-one shapes measured, both 1024x256x128 output projections, admitted
/// no TF32 winner at all and are absent by measurement, not by omission.
const SM120_TF32_EVIDENCE_CELLS_CUDA_13_2_DRIVER_595_84: &[Tf32AutoCell] = &[
    sm120_tf32_cell(
        Nn,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        768,
        3072,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        1536,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        1536,
        768,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        1536,
        768,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4621,
        1928,
        384,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        384,
        1928,
        4621,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4621,
        384,
        1928,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        4096,
        1536,
        3072,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        3072,
        1536,
        4096,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        4096,
        3072,
        1536,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nn,
        2048,
        768,
        3072,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_streamk_cell(
        Tn,
        3072,
        768,
        2048,
        super::contract::Tf32Sm120Stages::S3,
        RequiresVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Nt,
        2048,
        3072,
        768,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
    Tf32AutoCell {
        op: Nn,
        shape: Tf32ExactShape {
            output_rows: 1024,
            output_columns: 512,
            reduction: 128,
        },
        route: Tf32PhysicalRoute::MmaTf32RnaV1(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M64N64,
            stages: super::contract::Tf32PortableStages::S3,
        }),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate: RequiresNoBiasAndVectorAlignmentEvidence,
    },
    Tf32AutoCell {
        op: Tn,
        shape: Tf32ExactShape {
            output_rows: 128,
            output_columns: 512,
            reduction: 1024,
        },
        route: Tf32PhysicalRoute::MmaTf32RnaV1(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N32,
            stages: super::contract::Tf32PortableStages::S4,
        }),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    Tf32AutoCell {
        op: Nt,
        shape: Tf32ExactShape {
            output_rows: 1024,
            output_columns: 128,
            reduction: 512,
        },
        route: Tf32PhysicalRoute::MmaTf32RnaV1(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N16,
            stages: super::contract::Tf32PortableStages::S4,
        }),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    Tf32AutoCell {
        op: Tn,
        shape: Tf32ExactShape {
            output_rows: 256,
            output_columns: 128,
            reduction: 1024,
        },
        route: Tf32PhysicalRoute::MmaTf32RnaV1(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M16N16,
            stages: super::contract::Tf32PortableStages::S4,
        }),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    // The deep reductions of the training batch and the split candidate
    // (internal/perf/sm120-requal-deepk-20260903): no split-K arm won, and the
    // TN batch projection admitted no TF32 route at all.
    sm120_tf32_cell(
        Nn,
        10400,
        1536,
        384,
        super::contract::Tf32Sm120Tile::M64N64,
        super::contract::Tf32Sm120Stages::S2,
        RequiresNoBiasAndVectorAlignmentEvidence,
    ),
    sm120_tf32_cell(
        Tn,
        128,
        128,
        8192,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S4,
        RequiresVectorAlignmentEvidence,
    ),
    Tf32AutoCell {
        op: Nt,
        shape: Tf32ExactShape {
            output_rows: 128,
            output_columns: 8192,
            reduction: 128,
        },
        route: Tf32PhysicalRoute::MmaTf32RnaV1(super::contract::Tf32PortableRoute {
            tile: super::contract::Tf32PortableTile::M64N64,
            stages: super::contract::Tf32PortableStages::S2,
        }),
        tuning_revision: SM120_TF32_QUALIFIED_TUNING_REVISION,
        operand_gate: RequiresVectorAlignmentEvidence,
    },
    sm120_tf32_cell(
        Nt,
        10400,
        384,
        1536,
        super::contract::Tf32Sm120Tile::M64N128,
        super::contract::Tf32Sm120Stages::S2,
        RequiresVectorAlignmentEvidence,
    ),
];

const SM120_TF32_EVIDENCE_COHORTS: &[Tf32AutoEvidenceCohort] = &[
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY_CUDA_12_8,
        cells: SM120_TF32_EVIDENCE_CELLS_CUDA_12_8,
    },
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_0,
        cells: SM120_TF32_EVIDENCE_CELLS_CUDA_13_0,
    },
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY,
        cells: SM120_TF32_EVIDENCE_CELLS,
    },
    Tf32AutoEvidenceCohort {
        identity: SM120_TF32_QUALIFICATION_IDENTITY_CUDA_13_2_DRIVER_595_84,
        cells: SM120_TF32_EVIDENCE_CELLS_CUDA_13_2_DRIVER_595_84,
    },
];

fn matching_tf32_cohort(
    module: Tf32QualifiedModule,
    cohorts: &[Tf32AutoEvidenceCohort],
) -> Option<&Tf32AutoEvidenceCohort> {
    cohorts
        .iter()
        .find(|cohort| cohort.identity.matches(module))
}

fn measured_tf32_cell(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    tuning_revision: u16,
    cells: &[Tf32AutoCell],
) -> Option<Tf32PhysicalRoute> {
    cells
        .iter()
        .find(|cell| {
            cell.tuning_revision == tuning_revision
                && cell.op == request.op
                && cell.shape.matches_contiguous(request)
                && tf32_auto_operands_match(request.op, operands)
        })
        .map(|cell| cell.route)
}

fn measured_tf32_route_with_operands(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    availability: F32TriadAvailability,
    tuning_revision: u16,
) -> Option<Tf32PhysicalRoute> {
    if let Some(module) = availability.specialized
        && matching_tf32_cohort(module, SM120_TF32_EVIDENCE_COHORTS).is_none()
    {
        static NO_COHORT: std::sync::Once = std::sync::Once::new();
        crate::mamba_ssm::gpu::diagnostics::warn_once(&NO_COHORT, || {
            let newest = SM120_TF32_EVIDENCE_COHORTS
                .last()
                .and_then(|cohort| cohort.identity.mismatch(module))
                .unwrap_or("no cohort is frozen");
            format!(
                "no SM120 TF32 evidence cohort matches this stack (newest cohort differs at: \
                 {newest}); the exact f32 family serves every TF32 request until a \
                 requalification is frozen"
            )
        });
    }
    if let Some(module) = availability.specialized
        && let Some(cohort) = matching_tf32_cohort(module, SM120_TF32_EVIDENCE_COHORTS)
    {
        let route = measured_tf32_cell(request, operands, tuning_revision, cohort.cells)?;
        if cohort.identity == SM120_TF32_QUALIFICATION_IDENTITY
            && route.module_kind() == ModuleKind::TriadSm80
            && !availability.portable.is_some_and(|portable| {
                SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2.matches(portable)
            })
        {
            return None;
        }
        return Some(route);
    }
    let portable = availability.portable?;
    if let Some(field) = SM89_TF32_QUALIFICATION_IDENTITY.mismatch(portable) {
        if availability.specialized.is_none() {
            static STALE: std::sync::Once = std::sync::Once::new();
            crate::mamba_ssm::gpu::diagnostics::warn_once(&STALE, || {
                format!(
                    "the portable TF32 evidence cohort does not match this stack (differs at: \
                     {field}); the exact f32 family serves every TF32 request until a \
                     requalification is frozen"
                )
            });
        }
        return None;
    }
    measured_tf32_cell(request, operands, tuning_revision, SM89_TF32_EVIDENCE_CELLS)
}

pub fn resolve_f32_triad_auto(
    policy: F32TriadPolicy,
    request: F32TriadRequest,
    availability: F32TriadAvailability,
) -> Result<F32TriadSelection, String> {
    resolve_f32_triad_auto_impl(policy, request, None, availability)
}

pub(super) fn resolve_f32_triad_auto_with_operands(
    policy: F32TriadPolicy,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    availability: F32TriadAvailability,
) -> Result<F32TriadSelection, String> {
    resolve_f32_triad_auto_impl(policy, request, Some(operands), availability)
}

fn resolve_f32_triad_auto_impl(
    policy: F32TriadPolicy,
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> Result<F32TriadSelection, String> {
    request.shape.validate(request.op)?;
    match policy {
        F32TriadPolicy::ExactScalarFmaV1 => {
            Ok(exact_or_scalar_selection(request, operands, availability))
        }
        F32TriadPolicy::AllowDeterministicTf32V1 => {
            if availability.portable.is_none() && availability.specialized.is_none() {
                return Ok(F32TriadSelection::ScalarFmaV1);
            }
            let route = match operands {
                Some(operands) => measured_tf32_route_with_operands(
                    request,
                    operands,
                    availability,
                    F32_TF32_TUNING_REVISION,
                ),
                None => None,
            };
            let Some(route) = route else {
                return Ok(exact_or_scalar_selection(request, operands, availability));
            };
            match resolve_tf32_forced(request, availability, route) {
                Ok(route) => Ok(F32TriadSelection::Tf32(route)),
                Err(reason) => {
                    static DECLINED: std::sync::Once = std::sync::Once::new();
                    crate::mamba_ssm::gpu::diagnostics::warn_once(&DECLINED, || {
                        format!(
                            "deterministic TF32 route {route:?} is measured for this shape but \
                             does not bind on this stack ({reason}); the exact family serves \
                             instead"
                        )
                    });
                    Ok(exact_or_scalar_selection(request, operands, availability))
                }
            }
        }
    }
}

/// The exact-F32 family is the floor under both policies. A shape with no
/// measured TF32 route still runs on the SM120 exact routes when the operands
/// admit them, and only falls through to the plain scalar chain when they do
/// not: allowing TF32 must never select something slower than forbidding it.
pub(super) fn exact_or_scalar_selection(
    request: F32TriadRequest,
    operands: Option<F32TriadOperands>,
    availability: F32TriadAvailability,
) -> F32TriadSelection {
    operands
        .and_then(|operands| sm120_fma_exact_route(request, operands, availability))
        .map_or(
            F32TriadSelection::ScalarFmaV1,
            F32TriadSelection::ExactSm120Fma,
        )
}

pub fn resolve_tf32_forced(
    request: F32TriadRequest,
    availability: F32TriadAvailability,
    route: Tf32PhysicalRoute,
) -> Result<Tf32PhysicalRoute, String> {
    request.shape.validate(request.op)?;
    let (module_kind, dynamic_shared_bytes) = match route {
        Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_) => {
            let spec = super::contract::tf32_splitk_spec(request.op, route)?;
            (ModuleKind::TriadSm80, spec.dynamic_shared_bytes)
        }
        _ => {
            let spec = tf32_kernel_spec(request.op, route)?;
            (spec.module_kind, spec.dynamic_shared_bytes)
        }
    };
    let binding = match route {
        Tf32PhysicalRoute::MmaTf32RnaV1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_) => availability.portable,
        Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(_)
        | Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(_)
        | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_)
        | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_)
        | Tf32PhysicalRoute::Sm120TmaFmaExactV1(_) => availability.specialized,
    }
    .ok_or_else(|| format!("forced TF32 route {route:?} has no qualified module"))?;
    ensure_tf32_binding_contract(binding, module_kind, dynamic_shared_bytes, route)?;
    Ok(route)
}

fn ensure_tf32_binding_contract(
    binding: Tf32QualifiedModule,
    module_kind: ModuleKind,
    dynamic_shared_bytes: u32,
    route: Tf32PhysicalRoute,
) -> Result<(), String> {
    if binding.module_kind != module_kind
        || binding.module_kind != route.module_kind()
        || binding.artifact.module_kind != binding.module_kind
    {
        return Err(format!(
            "forced TF32 route {route:?} has the wrong module identity"
        ));
    }
    if binding.target != binding.compiler.target
        || binding.device_caps.accepted_target != Some(binding.target)
        || binding.compiler.output_kind != binding.artifact.artifact_kind
        || binding.artifact.compile_key != binding.compiler.invocation_digest
        || binding.compiler.composer_revision != COMPOSER_REVISION
        || binding.compiler.compiler_revision != COMPILER_REVISION
        || binding.compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
        || binding.compiler.schedule_revision != SCHEDULE_REVISION
    {
        return Err(format!(
            "forced TF32 route {route:?} has an inconsistent compiler or artifact identity"
        ));
    }
    if binding.device.compute_capability != binding.device_caps.compute_capability
        || binding.compiler.nvrtc_version != binding.device_caps.nvrtc_version
    {
        return Err(format!(
            "forced TF32 route {route:?} has an inconsistent device identity"
        ));
    }
    if !target_admits_route(binding, route) {
        return Err(format!(
            "forced TF32 route {route:?} is not admitted by target {}",
            binding.target.as_str()
        ));
    }
    if !matches!(
        route,
        Tf32PhysicalRoute::MmaTf32RnaV1(_)
            | Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
            | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
            | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_)
    ) && !binding.device_caps.tensor_map_access
    {
        return Err(format!(
            "forced TF32 route {route:?} requires tensor-map access"
        ));
    }
    if binding.device_caps.optin_shared_bytes < dynamic_shared_bytes {
        return Err(format!(
            "forced TF32 route {route:?} needs {} shared bytes, device admits {}",
            dynamic_shared_bytes, binding.device_caps.optin_shared_bytes
        ));
    }
    Ok(())
}

fn target_admits_route(binding: Tf32QualifiedModule, route: Tf32PhysicalRoute) -> bool {
    let cc = binding.device.compute_capability;
    let target = binding.target.as_str();
    let device_target = binding.device.target.as_str();
    match route {
        Tf32PhysicalRoute::MmaTf32RnaV1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_) => matches!(
            (cc, target, device_target),
            ((8, 0), "sm_80", "sm_80")
                | ((8, 6), "sm_86", "sm_86")
                | ((8, 7), "sm_87", "sm_87")
                | ((8, 9), "sm_89", "sm_89")
                | ((9, 0), "sm_90a", "sm_90a")
                | ((10, 0), "sm_100a", "sm_100a")
                | ((10, 1), "sm_101a", "sm_101a")
                | ((10, 3), "sm_103a", "sm_103a")
                | ((11, 0), "sm_110", "sm_110")
                | ((12, 0), "compute_120", "sm_120")
                | ((12, 1), "compute_121", "sm_121")
                | ((12, 1), "compute_120", "sm_120")
        ),
        Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(_) => {
            (cc, target, device_target) == ((9, 0), "sm_90a", "sm_90a")
        }
        Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(_) => matches!(
            (cc, target, device_target),
            ((10, 0), "compute_100f", "sm_100f")
                | ((10, 0), "compute_100a", "sm_100a")
                | ((10, 3), "compute_103f", "sm_103f")
                | ((10, 3), "compute_103a", "sm_103a")
                | ((11, 0), "compute_110f", "sm_110f")
                | ((11, 0), "compute_110a", "sm_110a")
        ),
        Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_)
        | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_)
        | Tf32PhysicalRoute::Sm120TmaFmaExactV1(_) => matches!(
            (cc, target, device_target),
            ((12, 0), "compute_120", "sm_120")
                | ((12, 1), "compute_121", "sm_121")
                | ((12, 1), "compute_120", "sm_120")
        ),
    }
}

/// One exact-F32 SM120 cell measured under the exact policy: the shape in
/// the performance-matrix convention with contiguous strides, and the arm
/// that won its official 101-window qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sm120FmaMeasuredCell {
    op: ResolvedGemmOp,
    shape: F32TriadShape,
    route: Sm120FmaRoute,
}

const fn sm120_fma_cell(
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    tile: Sm120FmaTile,
    kvec: bool,
    splits: u8,
) -> Sm120FmaMeasuredCell {
    let (m, k, n) = dims;
    let (lda, ldb, ldc) = match op {
        ResolvedGemmOp::Nn => (k, n, n),
        ResolvedGemmOp::Tn => (k, n, n),
        ResolvedGemmOp::Nt => (n, n, k),
    };
    Sm120FmaMeasuredCell {
        op,
        shape: F32TriadShape {
            m,
            k,
            n,
            lda,
            ldb,
            ldc,
        },
        route: Sm120FmaRoute { tile, kvec, splits },
    }
}

/// The measured cells, each carrying the arm that won its qualification:
/// the nine research projections against the scalar production route
/// (1.22x to 1.60x, internal/perf/sm120-exact-nn-wide-20260903), and the
/// classifier serve and training-batch projections against what the exact
/// policy resolved before them (1.21x to 2.07x,
/// internal/perf/sm120-exact-wide-product-20260903).
const SM120_FMA_MEASURED_CELLS_CC120_170: [Sm120FmaMeasuredCell; 18] = [
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (2_048, 768, 3_072),
        Sm120FmaTile::M128N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (2_048, 1_536, 768),
        Sm120FmaTile::M128N64,
        false,
        4,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (4_621, 384, 1_928),
        Sm120FmaTile::M64N128,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (2_048, 768, 3_072),
        Sm120FmaTile::M64N128,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (2_048, 1_536, 768),
        Sm120FmaTile::M128N64,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (4_621, 384, 1_928),
        Sm120FmaTile::M128N64,
        false,
        5,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (2_048, 768, 3_072),
        Sm120FmaTile::M128N64,
        true,
        5,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (2_048, 1_536, 768),
        Sm120FmaTile::M64N128,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (4_621, 384, 1_928),
        Sm120FmaTile::M128N64,
        true,
        3,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (4_621, 768, 384),
        Sm120FmaTile::M64N64,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (4_621, 1_024, 384),
        Sm120FmaTile::M64N64,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (10_400, 384, 384),
        Sm120FmaTile::M128N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nn,
        (10_400, 768, 384),
        Sm120FmaTile::M128N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (10_400, 384, 1_536),
        Sm120FmaTile::M64N128,
        false,
        5,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Tn,
        (4_096, 3_072, 1_536),
        Sm120FmaTile::M64N128,
        false,
        2,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (10_400, 384, 1_536),
        Sm120FmaTile::M128N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (10_400, 384, 384),
        Sm120FmaTile::M64N64,
        false,
        1,
    ),
    sm120_fma_cell(
        ResolvedGemmOp::Nt,
        (4_621, 768, 384),
        Sm120FmaTile::M64N64,
        false,
        1,
    ),
];

/// Shapes past the measured cells take the exact family when they fill the
/// device with whole tiles on their own: at least one tile per
/// multiprocessor, no split, and a reduction long enough for the pipeline
/// to matter. The floor was three; the product-shape screen showed the
/// family beating the scalar route at 1.3 tiles per multiprocessor (1.68x
/// on the 4621x384 output) and at 2.9 (1.81x on the 10400x384 output).
const SM120_FMA_GENERIC_TILES_PER_MULTIPROCESSOR: usize = 1;
const SM120_FMA_GENERIC_MIN_EDGE: usize = 128;
const SM120_FMA_GENERIC_MIN_REDUCTION: usize = 256;

fn sm120_fma_generic_route(
    request: F32TriadRequest,
    multiprocessor_count: u32,
) -> Option<Sm120FmaRoute> {
    let (tile, kvec) = match request.op {
        ResolvedGemmOp::Nn => (Sm120FmaTile::M128N64, false),
        ResolvedGemmOp::Tn => (Sm120FmaTile::M64N128, false),
        ResolvedGemmOp::Nt => (Sm120FmaTile::M128N64, true),
    };
    let rows = request.shape.output_rows(request.op);
    let columns = request.shape.output_columns(request.op);
    let reduction = request.shape.reduction(request.op);
    if rows < SM120_FMA_GENERIC_MIN_EDGE
        || columns < SM120_FMA_GENERIC_MIN_EDGE
        || reduction < SM120_FMA_GENERIC_MIN_REDUCTION
    {
        return None;
    }
    let (bm, bn) = tile.dims();
    let tiles = rows
        .div_ceil(bm as usize)
        .checked_mul(columns.div_ceil(bn as usize))?;
    let required =
        (multiprocessor_count as usize).checked_mul(SM120_FMA_GENERIC_TILES_PER_MULTIPROCESSOR)?;
    (tiles >= required).then_some(Sm120FmaRoute {
        tile,
        kvec,
        splits: 1,
    })
}

fn f32_pointer_is_tma_aligned(pointer: super::contract::CUptr) -> bool {
    pointer != 0 && pointer.is_multiple_of(16)
}

/// The exact-F32 SM120 route for a request under the exact policy: the
/// specialized module must be bound on a CC 12.0 device, every operand must
/// sit on a 16-byte boundary with float4 leading dimensions (the tensor
/// maps demand it), and the shape must be a measured cell or fill the
/// device on its own. Anything else keeps the scalar routes.
pub(super) fn sm120_fma_exact_route(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    availability: F32TriadAvailability,
) -> Option<Sm120FmaRoute> {
    let qualified = availability.specialized?;
    // The kernels are built for both minors of the family and the route
    // admission already names sm_121, so the gate follows the kernel rather
    // than the board the cells happen to be measured on.
    if qualified.module_kind != ModuleKind::TriadSm120
        || !matches!(qualified.device.compute_capability, (12, 0) | (12, 1))
        || !qualified.device_caps.tensor_map_access
    {
        return None;
    }
    let shape = request.shape;
    if shape.reduction(request.op) == 0
        || !f32_pointer_is_tma_aligned(operands.a)
        || !f32_pointer_is_tma_aligned(operands.b)
        || !f32_pointer_is_tma_aligned(operands.output)
        || !shape.lda.is_multiple_of(4)
        || !shape.ldb.is_multiple_of(4)
    {
        return None;
    }
    let measured = SM120_FMA_MEASURED_CELLS_CC120_170
        .iter()
        .find(|cell| cell.op == request.op && cell.shape == shape)
        .map(|cell| cell.route)
        .filter(|_| qualified.device.multiprocessor_count == 170);
    let route = measured
        .or_else(|| sm120_fma_generic_route(request, qualified.device.multiprocessor_count))?;
    super::launch::sm120_fma_launch_plan(request, route).ok()?;
    Some(route)
}

pub const SM90A_AUTO_CELLS: &[Sm90aForcedRoute] = &[];
pub const SM100_AUTO_CELLS_CC100: &[Sm100ForcedRoute] = &[];
pub const SM100_AUTO_CELLS_CC103: &[Sm100ForcedRoute] = &[];
/// Automatic CC 11.0 routes. Empty until a board measures them.
pub const SM100_AUTO_CELLS_CC110: &[Sm100ForcedRoute] = &[];

/// An automatic SM100 request: the measured table names the tile, the
/// caller only the operation, operands and shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm100AutoRequest {
    pub op: Sm100Op,
    pub dtype: WeightDtype,
    pub shape: super::contract::Sm100Shape,
    pub a_ptr: super::contract::CUptr,
    pub b_ptr: super::contract::CUptr,
    pub operands: super::contract::Sm100LaunchOperands,
}

/// The measured SM100 cells of a board's compute capability. A minor with
/// no table has no cell to find and declines like an uncovered shape.
pub fn sm100_auto_cells(device_cc: (i32, i32)) -> &'static [Sm100ForcedRoute] {
    match device_cc {
        (10, 0) => SM100_AUTO_CELLS_CC100,
        (10, 3) => SM100_AUTO_CELLS_CC103,
        (11, 0) => SM100_AUTO_CELLS_CC110,
        _ => &[],
    }
}

/// Resolves a measured SM100 cell for the request, or declines to the
/// portable caller. The table is per compute capability and stays empty
/// until a board of that capability qualifies its cells.
pub fn resolve_sm100_auto(
    device_cc: (i32, i32),
    module_target: Option<Sm100TargetCandidate>,
    request: Sm100AutoRequest,
) -> Option<Sm100ForcedRoute> {
    resolve_sm100_auto_from_cells(
        sm100_auto_cells(device_cc),
        device_cc,
        module_target,
        request,
    )
}

fn resolve_sm100_auto_from_cells(
    cells: &[Sm100ForcedRoute],
    device_cc: (i32, i32),
    module_target: Option<Sm100TargetCandidate>,
    request: Sm100AutoRequest,
) -> Option<Sm100ForcedRoute> {
    // A measured cell first; anything the table does not cover takes the wave
    // rule: the wide tile only when the output still fills a device with it,
    // deeper stages and the larger schedule only when the reduction is deep
    // enough to feed them. The map, operand and admission checks below apply
    // to both sources equally.
    let route = cells
        .iter()
        .copied()
        .find(|cell| {
            cell.op == request.op && cell.dtype == request.dtype && cell.shape == request.shape
        })
        .unwrap_or_else(|| {
            let (rows, columns, reduction) = match request.op {
                Sm100Op::Nn => (request.shape.m, request.shape.n, request.shape.k),
                Sm100Op::Tn => (request.shape.k, request.shape.n, request.shape.m),
                Sm100Op::Nt => (request.shape.m, request.shape.k, request.shape.n),
            };
            // No board of this family is qualified yet, so the tile floor is
            // the family's own SM counts: the smallest announced part carries
            // well over a hundred multiprocessors, and 128 wide tiles cover it.
            let wide_tiles = rows.div_ceil(128).saturating_mul(columns.div_ceil(128));
            let tile = if columns >= 128 && wide_tiles >= 128 {
                super::contract::Sm100Tile::M128N128
            } else {
                super::contract::Sm100Tile::M128N64
            };
            let stages = if reduction >= 2048 {
                super::contract::Sm100Stages::S4
            } else if reduction >= 512 {
                super::contract::Sm100Stages::S3
            } else {
                super::contract::Sm100Stages::S2
            };
            let schedule = if reduction >= 1024 {
                super::contract::Sm100Schedule::P8
            } else {
                super::contract::Sm100Schedule::C4
            };
            Sm100ForcedRoute {
                op: request.op,
                dtype: request.dtype,
                physical: super::contract::Sm100PhysicalRoute {
                    tile,
                    stages,
                    schedule,
                },
                shape: request.shape,
            }
        });
    super::contract::validate_sm100_map_request(super::contract::Sm100MapRequest {
        op: request.op,
        dtype: request.dtype,
        tile: route.physical.tile,
        a_ptr: request.a_ptr,
        b_ptr: request.b_ptr,
        shape: request.shape,
    })
    .ok()?;
    if !sm100_auto_operands_supported(request.op, request.operands) {
        return None;
    }
    (resolve_sm100_forced(device_cc, module_target, route).ok()? == Some(route)).then_some(route)
}

/// The epilogue contexts the measured SM100 cells were qualified under:
/// the same law as the SM120 tiles, until a board measures otherwise.
fn sm100_auto_operands_supported(
    op: Sm100Op,
    operands: super::contract::Sm100LaunchOperands,
) -> bool {
    let output_alignment = if op == Sm100Op::Tn { 4 } else { 2 };
    operands.output_ptr != 0
        && operands.output_ptr.is_multiple_of(output_alignment)
        && (operands.bias_ptr == 0 || operands.bias_ptr.is_multiple_of(4))
        && match op {
            Sm100Op::Nn => operands.bias_ptr == 0 || operands.alpha == 1.0,
            Sm100Op::Tn => operands.bias_ptr == 0 && operands.beta == 1.0,
            Sm100Op::Nt => operands.bias_ptr == 0 && operands.beta == 0.0,
        }
}

/// An automatic SM90a request: the measured table names the warpgroup
/// schedule, the caller only the operation, operands and shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm90aAutoRequest {
    pub op: Sm90aOp,
    pub dtype: WeightDtype,
    pub shape: Sm90aShape,
    pub a_ptr: super::contract::CUptr,
    pub b_ptr: super::contract::CUptr,
    pub operands: super::contract::Sm90aLaunchOperands,
}

/// Resolves a measured SM90a cell for the request, or declines to the
/// portable caller. The table stays empty until a CC 9.0 board qualifies
/// its cells.
pub fn resolve_sm90a_auto(
    device_cc: (i32, i32),
    module_available: bool,
    request: Sm90aAutoRequest,
) -> Option<Sm90aForcedRoute> {
    resolve_sm90a_auto_from_cells(SM90A_AUTO_CELLS, device_cc, module_available, request)
}

fn resolve_sm90a_auto_from_cells(
    cells: &[Sm90aForcedRoute],
    device_cc: (i32, i32),
    module_available: bool,
    request: Sm90aAutoRequest,
) -> Option<Sm90aForcedRoute> {
    // A measured cell first; anything the table does not cover takes the wave
    // rule, so a Hopper board runs its own wgmma kernels rather than the
    // portable tiles. The second warpgroup of Wg2 is a dedicated TMA producer:
    // it earns its threads on a deep reduction and costs occupancy on a
    // shallow one, so the reduction depth is the axis the rule reads. The map,
    // operand and admission checks below apply to both sources equally.
    let route = cells
        .iter()
        .copied()
        .find(|cell| {
            cell.op == request.op && cell.dtype == request.dtype && cell.shape == request.shape
        })
        .unwrap_or_else(|| {
            let reduction = match request.op {
                Sm90aOp::Nn => request.shape.k,
                Sm90aOp::Tn => request.shape.m,
                Sm90aOp::Nt => request.shape.n,
            };
            let schedule = if reduction >= 1024 {
                Sm90aWarpgroupSchedule::Wg2
            } else {
                Sm90aWarpgroupSchedule::Wg1
            };
            Sm90aForcedRoute {
                op: request.op,
                dtype: request.dtype,
                schedule,
                shape: request.shape,
            }
        });
    super::contract::validate_sm90a_map_request(super::contract::Sm90aMapRequest {
        op: request.op,
        dtype: request.dtype,
        a_ptr: request.a_ptr,
        b_ptr: request.b_ptr,
        shape: request.shape,
    })
    .ok()?;
    if !sm90a_auto_operands_supported(request.op, request.operands) {
        return None;
    }
    let resolved = resolve_sm90a_forced(
        device_cc,
        module_available,
        route.op,
        route.dtype,
        route.schedule,
        route.shape,
    )
    .ok()?;
    (resolved == Some(route)).then_some(route)
}

/// The epilogue contexts the measured SM90a cells were qualified under:
/// the same law as the SM120 tiles, until a board measures otherwise.
fn sm90a_auto_operands_supported(
    op: Sm90aOp,
    operands: super::contract::Sm90aLaunchOperands,
) -> bool {
    let output_alignment = if op == Sm90aOp::Tn { 4 } else { 2 };
    operands.output_ptr != 0
        && operands.output_ptr.is_multiple_of(output_alignment)
        && (operands.bias_ptr == 0 || operands.bias_ptr.is_multiple_of(4))
        && match op {
            Sm90aOp::Nn => operands.bias_ptr == 0 || operands.alpha == 1.0,
            Sm90aOp::Tn => operands.bias_ptr == 0 && operands.beta == 1.0,
            Sm90aOp::Nt => operands.bias_ptr == 0 && operands.beta == 0.0,
        }
}
/// Measured BF16/F16 NN/TN/NT cells eligible for automatic CC 12.0 dispatch.
///
/// These routes are exact shape-and-stride matches; this is not a heuristic
/// table and it must not be reused for another compute-capability minor.
pub const SM120_AUTO_CELLS_CC120: &[Sm120ForcedRoute] = &[
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 1536,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 3072,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 3072,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 1536,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 3072,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 3072,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 1536,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 3072,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 3072,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 1536,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 3072,
            ldb: 768,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 3072,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 768,
            ldb: 768,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 768,
            ldb: 768,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 1536,
            ldb: 1536,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 1536,
            n: 768,
            lda: 768,
            ldb: 768,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 3072,
            n: 768,
            lda: 768,
            ldb: 768,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4096,
            k: 3072,
            n: 1536,
            lda: 1536,
            ldb: 1536,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 768,
            ldb: 3072,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 768,
            ldb: 3072,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 768,
            ldb: 3072,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 768,
            ldb: 3072,
            ldc: 3072,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 384,
            ldb: 1928,
            ldc: 1928,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 3072,
            ldb: 3072,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 1928,
            ldb: 1928,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 2048,
            k: 768,
            n: 3072,
            lda: 3072,
            ldb: 3072,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 384,
            n: 1928,
            lda: 1928,
            ldb: 1928,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 1024,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 384,
            ldb: 1536,
            ldc: 1536,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Tn,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 768,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 1024,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 1536,
            ldb: 1536,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::Bf16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 768,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 768,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 4621,
            k: 1024,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 1024,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 384,
            n: 1536,
            lda: 1536,
            ldb: 1536,
            ldc: 384,
        },
    },
    Sm120ForcedRoute {
        op: Sm120Op::Nt,
        dtype: WeightDtype::F16,
        physical: Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        },
        shape: Sm120Shape {
            m: 10400,
            k: 768,
            n: 384,
            lda: 384,
            ldb: 384,
            ldc: 768,
        },
    },
];
/// Automatic CC 12.1 routes. Empty until separate physical evidence exists.
pub const SM120_AUTO_CELLS_CC121: &[Sm120ForcedRoute] = &[];

/// Logical operands for an SM120 automatic route lookup.
///
/// The selected physical route comes solely from the per-minor qualified
/// table; callers never nominate a tile, BK, or pipeline stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::mamba_ssm::gpu) struct Sm120AutoRequest {
    pub op: Sm120Op,
    pub dtype: WeightDtype,
    pub shape: Sm120Shape,
    pub a_ptr: super::contract::CUptr,
    pub b_ptr: super::contract::CUptr,
    pub operands: Sm120LaunchOperands,
    /// Live multiprocessor count of the board: the neighbour rule refuses a
    /// tile whose grid cannot fill one wave of it.
    pub multiprocessors: u32,
}

/// Resolves only a measured, minor-specific SM120 cell. Every unsupported
/// capability or request detail declines to the portable caller with `None`.
pub(super) fn resolve_sm120_auto(
    caps: DeviceCaps,
    module_target: Option<Sm120TargetCandidate>,
    request: Sm120AutoRequest,
) -> Option<Sm120ForcedRoute> {
    if !crate::mamba_ssm::gpu::device::is_sm120_family(caps.compute_capability) {
        return None;
    }
    // A minor without a measured table has no cell to find and declines the
    // way an uncovered shape does.
    let cells = match caps.compute_capability {
        (12, 0) => SM120_AUTO_CELLS_CC120,
        (12, 1) => SM120_AUTO_CELLS_CC121,
        _ => &[],
    };
    resolve_sm120_auto_from_cells(cells, caps, module_target, request)
}

/// How far, per axis and in natural-log units, a shape may sit from the
/// nearest measured cell and still take that cell's tile: a factor of eight
/// in output rows, output columns and reduction. Inside that band the
/// leave-one-out check over the sixty measured cells loses 8 percent on
/// average to the best tile (internal/perf/sm120-half-census-20260903b);
/// outside it the portable tensor-core tiles serve, as they did before.
const SM120_AUTO_NEIGHBOUR_LOG_BOUND: f64 = 2.079_441_541_679_836;

/// Output rows, output columns and reduction length of an SM120 request,
/// the three axes a tile choice depends on.
fn sm120_auto_geometry(op: Sm120Op, shape: Sm120Shape) -> [f64; 3] {
    let (rows, columns, reduction) = match op {
        Sm120Op::Nn => (shape.m, shape.n, shape.k),
        Sm120Op::Tn => (shape.k, shape.n, shape.m),
        Sm120Op::Nt => (shape.m, shape.k, shape.n),
    };
    [rows as f64, columns as f64, reduction as f64]
}

/// The tile of the nearest measured cell of the same operation and dtype,
/// when one lies within the neighbour band. Ties keep table order, so the
/// choice is deterministic for a given table.
fn nearest_sm120_cell(
    cells: &[Sm120ForcedRoute],
    op: Sm120Op,
    dtype: WeightDtype,
    shape: Sm120Shape,
    multiprocessors: u32,
) -> Option<Sm120PhysicalRoute> {
    let target = sm120_auto_geometry(op, shape).map(f64::ln);
    let mut best: Option<(f64, &Sm120ForcedRoute)> = None;
    for cell in cells {
        if cell.op != op || cell.dtype != dtype {
            continue;
        }
        let axes = sm120_auto_geometry(op, cell.shape)
            .map(f64::ln)
            .iter()
            .zip(target.iter())
            .map(|(cell_axis, target_axis)| (cell_axis - target_axis).abs())
            .collect::<Vec<_>>();
        if axes
            .iter()
            .any(|axis| *axis > SM120_AUTO_NEIGHBOUR_LOG_BOUND)
        {
            continue;
        }
        let distance = axes.iter().map(|axis| axis * axis).sum::<f64>().sqrt();
        if best.is_none_or(|(best_distance, _)| distance < best_distance) {
            best = Some((distance, cell));
        }
    }
    let (_, neighbour) = best?;
    let mut physical = neighbour.physical;
    // A neighbour measured on a larger output may carry a tile whose grid
    // leaves most of this board idle. When the shape fills less than one
    // wave with that tile and less than half of what the neighbour filled,
    // the smallest tile serves instead: the census shows it winning every
    // cell that far into underfill. The pipeline depth and reduction step
    // stay the neighbour's. A neighbour that won underfilled itself keeps
    // its tile for shapes that are underfilled the same way.
    let grid = |cell_shape: Sm120Shape| {
        let [rows, columns, _] = sm120_auto_geometry(op, cell_shape);
        (rows / f64::from(physical.tile.output_rows())).ceil()
            * (columns / f64::from(physical.tile.output_columns())).ceil()
    };
    let target_grid = grid(shape);
    if target_grid < f64::from(multiprocessors) && target_grid * 2.0 < grid(neighbour.shape) {
        physical.tile = Sm120Tile::M64N64;
    }
    Some(physical)
}

fn resolve_sm120_auto_from_cells(
    cells: &[Sm120ForcedRoute],
    caps: DeviceCaps,
    module_target: Option<Sm120TargetCandidate>,
    request: Sm120AutoRequest,
) -> Option<Sm120ForcedRoute> {
    let measured = cells.iter().copied().find(|cell| {
        cell.op == request.op && cell.dtype == request.dtype && cell.shape == request.shape
    });
    // A shape no cell names takes the tile of the nearest measured cell, so
    // every shape in the measured band has a path; the exact cell still wins
    // where one exists.
    let route = match measured {
        Some(cell) => cell,
        None => Sm120ForcedRoute {
            op: request.op,
            dtype: request.dtype,
            physical: nearest_sm120_cell(
                cells,
                request.op,
                request.dtype,
                request.shape,
                request.multiprocessors,
            )?,
            shape: request.shape,
        },
    };
    let maps = Sm120MapRequest {
        op: request.op,
        dtype: request.dtype,
        tile: route.physical.tile,
        bk: route.physical.bk,
        a_ptr: request.a_ptr,
        b_ptr: request.b_ptr,
        shape: request.shape,
    };
    validate_sm120_map_request(maps).ok()?;
    if !sm120_auto_operands_supported(request.op, request.operands) {
        return None;
    }
    match sm120_forced_decline(caps, module_target, route) {
        Ok(None) => Some(route),
        Ok(Some(reason)) => {
            static DECLINED: std::sync::Once = std::sync::Once::new();
            crate::mamba_ssm::gpu::diagnostics::warn_once(&DECLINED, || {
                format!(
                    "SM120 half cell {route:?} is measured for this shape but this board \
                     declines it ({reason}); the portable tensor-core tiles serve instead"
                )
            });
            None
        }
        Err(error) => {
            static INVALID: std::sync::Once = std::sync::Once::new();
            crate::mamba_ssm::gpu::diagnostics::warn_once(&INVALID, || {
                format!("SM120 half cell {route:?} is not a launchable route: {error}")
            });
            None
        }
    }
}

fn sm120_auto_operands_supported(op: Sm120Op, operands: Sm120LaunchOperands) -> bool {
    let output_alignment = if op == Sm120Op::Tn { 4 } else { 2 };
    operands.output_ptr != 0
        && operands.output_ptr.is_multiple_of(output_alignment)
        && (operands.bias_ptr == 0 || operands.bias_ptr.is_multiple_of(4))
        && match op {
            Sm120Op::Nn => operands.bias_ptr == 0 || operands.alpha == 1.0,
            Sm120Op::Tn => operands.bias_ptr == 0 && operands.beta == 1.0,
            Sm120Op::Nt => operands.bias_ptr == 0 && operands.beta == 0.0,
        }
}

const SM100_CC100_TARGETS: [Sm100TargetCandidate; 2] = [
    Sm100TargetCandidate {
        device_cc: (10, 0),
        nvrtc_arch: "compute_100f",
        ptx_target: "sm_100f",
        kind: Sm100TargetKind::Family,
    },
    Sm100TargetCandidate {
        device_cc: (10, 0),
        nvrtc_arch: "compute_100a",
        ptx_target: "sm_100a",
        kind: Sm100TargetKind::Exact,
    },
];

const SM100_CC103_TARGETS: [Sm100TargetCandidate; 2] = [
    Sm100TargetCandidate {
        device_cc: (10, 3),
        nvrtc_arch: "compute_103f",
        ptx_target: "sm_103f",
        kind: Sm100TargetKind::Family,
    },
    Sm100TargetCandidate {
        device_cc: (10, 3),
        nvrtc_arch: "compute_103a",
        ptx_target: "sm_103a",
        kind: Sm100TargetKind::Exact,
    },
];

const SM100_CC110_TARGETS: [Sm100TargetCandidate; 2] = [
    Sm100TargetCandidate {
        device_cc: (11, 0),
        nvrtc_arch: "compute_110f",
        ptx_target: "sm_110f",
        kind: Sm100TargetKind::Family,
    },
    Sm100TargetCandidate {
        device_cc: (11, 0),
        nvrtc_arch: "compute_110a",
        ptx_target: "sm_110a",
        kind: Sm100TargetKind::Exact,
    },
];

pub fn sm100_target_candidates(cc: (i32, i32)) -> &'static [Sm100TargetCandidate] {
    match cc {
        (10, 0) => &SM100_CC100_TARGETS,
        (10, 3) => &SM100_CC103_TARGETS,
        (11, 0) => &SM100_CC110_TARGETS,
        _ => &[],
    }
}

/// The SM100 candidates a toolkit can compile and that the contract has
/// verified. Family-specific targets (`compute_100f` and siblings) and the
/// CC 10.3 targets arrived with CUDA 12.9, the CC 11.0 targets with CUDA
/// 13.2, and CUDA 12.8 assembles the tcgen allocation with a different
/// instruction pairing than the one the contract freezes; below 12.9 the
/// family is not offered at all rather than run unverified.
pub fn sm100_target_candidates_for_nvrtc(
    cc: (i32, i32),
    nvrtc_version: (i32, i32),
) -> Vec<Sm100TargetCandidate> {
    sm100_target_candidates(cc)
        .iter()
        .copied()
        .filter(|candidate| {
            let floor = match candidate.device_cc {
                (11, 0) => (13, 2),
                _ => (12, 9),
            };
            nvrtc_version >= floor
        })
        .collect()
}

const SM120_CC120_TARGETS: [Sm120TargetCandidate; 1] = [Sm120TargetCandidate {
    device_cc: (12, 0),
    nvrtc_arch: "compute_120",
    ptx_target: "sm_120",
}];

const SM120_CC121_FALLBACK_TARGETS: [Sm120TargetCandidate; 1] = [Sm120TargetCandidate {
    device_cc: (12, 1),
    nvrtc_arch: "compute_120",
    ptx_target: "sm_120",
}];

const SM120_CC121_TARGETS: [Sm120TargetCandidate; 2] = [
    Sm120TargetCandidate {
        device_cc: (12, 1),
        nvrtc_arch: "compute_121",
        ptx_target: "sm_121",
    },
    Sm120TargetCandidate {
        device_cc: (12, 1),
        nvrtc_arch: "compute_120",
        ptx_target: "sm_120",
    },
];

/// Returns compiler targets valid for the exact device and NVRTC version.
///
/// An empty slice means the specialized module must decline to the portable
/// Triad path. Target support does not by itself qualify an automatic cell.
pub fn sm120_target_candidates(
    cc: (i32, i32),
    nvrtc: (i32, i32),
) -> &'static [Sm120TargetCandidate] {
    match (cc, nvrtc) {
        ((12, 0), version) if version >= (12, 8) => &SM120_CC120_TARGETS,
        ((12, 1), version) if version >= (12, 9) => &SM120_CC121_TARGETS,
        ((12, 1), version) if version >= (12, 8) => &SM120_CC121_FALLBACK_TARGETS,
        _ => &[],
    }
}

/// Validates a caller-selected SM120 route against device, target, and resources.
///
/// This forced resolver exists for qualification and census tooling. Production
/// typed dispatch uses the private minor-specific automatic resolver instead.
pub fn resolve_sm120_forced(
    caps: DeviceCaps,
    module_target: Option<Sm120TargetCandidate>,
    route: Sm120ForcedRoute,
) -> Result<Option<Sm120ForcedRoute>, String> {
    Ok(sm120_forced_decline(caps, module_target, route)?
        .is_none()
        .then_some(route))
}

/// The first board-level reason `route` cannot launch here, or `None` when
/// it can. [`resolve_sm120_forced`] collapses this into its option; the
/// automatic path reports it, so a wrong board is told apart from a missing
/// cell.
pub(super) fn sm120_forced_decline(
    caps: DeviceCaps,
    module_target: Option<Sm120TargetCandidate>,
    route: Sm120ForcedRoute,
) -> Result<Option<&'static str>, String> {
    route.shape.validate(route.op)?;
    if !matches!(route.dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM120 TMA supports bf16 and f16 operands only".into());
    }
    let spec = route.kernel_spec()?;
    let Some(module_target) = module_target else {
        return Ok(Some("no SM120 module is bound"));
    };
    let device_cc = (
        i32::try_from(caps.compute_capability.0)
            .map_err(|_| "SM120 device CC major exceeds i32::MAX".to_string())?,
        i32::try_from(caps.compute_capability.1)
            .map_err(|_| "SM120 device CC minor exceeds i32::MAX".to_string())?,
    );
    let accepted = caps
        .accepted_target
        .map(|target| target.as_str().to_owned());
    Ok(if module_target.device_cc != device_cc {
        Some("the bound SM120 module was compiled for another compute capability")
    } else if !sm120_target_candidates(device_cc, caps.nvrtc_version).contains(&module_target) {
        Some("this toolkit has no SM120 target for the device")
    } else if accepted.as_deref() != Some(module_target.nvrtc_arch) {
        Some("the driver accepted a different target than the bound module")
    } else if !caps.tensor_map_access {
        Some("the driver exposes no tensor-map access")
    } else if caps.optin_shared_bytes < spec.dynamic_shared_bytes {
        Some("the device opt-in shared memory is below the kernel's staging")
    } else {
        None
    })
}

pub fn resolve_sm100_forced(
    device_cc: (i32, i32),
    module_target: Option<Sm100TargetCandidate>,
    route: Sm100ForcedRoute,
) -> Result<Option<Sm100ForcedRoute>, String> {
    route.shape.validate(route.op)?;
    if !matches!(route.dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM100 TCGEN supports bf16 and f16 operands only".into());
    }
    route.kernel_spec()?;
    let Some(module_target) = module_target else {
        return Ok(None);
    };
    if module_target.device_cc != device_cc
        || !sm100_target_candidates(device_cc).contains(&module_target)
    {
        return Ok(None);
    }
    Ok(Some(route))
}

pub fn resolve_sm90a_forced(
    device_cc: (i32, i32),
    module_available: bool,
    op: Sm90aOp,
    dtype: WeightDtype,
    schedule: Sm90aWarpgroupSchedule,
    shape: Sm90aShape,
) -> Result<Option<Sm90aForcedRoute>, String> {
    shape.validate(op)?;
    if !matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM90a WGMMA supports bf16 and f16 operands only".into());
    }
    if device_cc != (9, 0) || !module_available {
        return Ok(None);
    }
    Ok(Some(Sm90aForcedRoute {
        op,
        dtype,
        schedule,
        shape,
    }))
}

// ── Split-M TN partition heuristic ──

/// Scratch cap for split-M partials, in f32 elements. Must not exceed the
/// `splitk_scratch` allocation in kernels.rs.
pub(super) const SPLITM_TN_SCRATCH_CAP: usize = 1 << 23;
/// m_chunk alignment (BK of the TN tile).
pub(super) const SPLITM_TN_BK_ALIGN: u32 = 16;

/// Decide the split-M factor for the TN (dW) kernel on underfilled grids.
/// Returns `(m_chunk, f_final)` or `None` when the plain kernel is fine.
#[inline]
pub(super) fn splitm_tn_partition(
    batch: usize,
    n_in: usize,
    n_out: usize,
    multiprocessor_count: u32,
) -> Result<Option<(usize, usize)>, String> {
    // No n_in floor: the partial kernel predicates K_out < 128 exactly
    // like the plain kernel, and a small-K dW against a large batch
    // reduction underfills the grid without the split (K_out=24, N=768
    // ran six CTAs). The split changes the dW summation order versus
    // the plain kernel; run-to-run and per-shape determinism hold — the
    // partition is a pure function of the shape and immutable device size.
    if !(n_out >= 128 && batch >= 256) {
        return Ok(None);
    }
    let policy = scalar_wave_policy(multiprocessor_count)?;
    let target_blocks = wave_target_blocks(
        multiprocessor_count,
        policy.tn_split_m_wave_numerator,
        policy.tn_split_m_wave_denominator,
    )?;
    let batch_u32 = u32::try_from(batch).map_err(|_| "TN split-M batch exceeds u32::MAX")?;
    let k_tiles = u32::try_from(n_in)
        .map_err(|_| "TN split-M input width exceeds u32::MAX")?
        .div_ceil(128);
    let n_tiles = u32::try_from(n_out)
        .map_err(|_| "TN split-M output width exceeds u32::MAX")?
        .div_ceil(128);
    let base_blocks = k_tiles
        .checked_mul(n_tiles)
        .ok_or_else(|| "TN split-M base grid overflows u32".to_string())?;
    if base_blocks == 0 || base_blocks >= target_blocks {
        return Ok(None);
    }
    let f_grid = target_blocks.div_ceil(base_blocks);
    let output_elements = n_in
        .checked_mul(n_out)
        .ok_or_else(|| "TN split-M output size overflows usize".to_string())?;
    let f_scratch_cap = u32::try_from(SPLITM_TN_SCRATCH_CAP / output_elements)
        .map_err(|_| "TN split-M scratch factor exceeds u32::MAX")?;
    let f = f_grid.min(f_scratch_cap).max(1);
    let m_chunk_raw = batch_u32.div_ceil(f);
    let m_chunk = m_chunk_raw
        .checked_add(SPLITM_TN_BK_ALIGN - 1)
        .ok_or_else(|| "TN split-M aligned chunk overflows u32".to_string())?
        & !(SPLITM_TN_BK_ALIGN - 1);
    let f_final = batch_u32.div_ceil(m_chunk);
    let scratch_elements = usize::try_from(f_final)
        .map_err(|_| "TN split-M partition count exceeds usize")?
        .checked_mul(output_elements)
        .ok_or_else(|| "TN split-M scratch size overflows usize".to_string())?;
    if f_final < 2 || scratch_elements > SPLITM_TN_SCRATCH_CAP {
        return Ok(None);
    }
    Ok(Some((
        usize::try_from(m_chunk).map_err(|_| "TN split-M chunk exceeds usize")?,
        usize::try_from(f_final).map_err(|_| "TN split-M partition count exceeds usize")?,
    )))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TnNarrowSplitMCell {
    shape: F32TriadShape,
    m_chunk: usize,
    chunks: usize,
}

const TN_NARROW_SPLITM_SM120_CC120_170_CELLS: [TnNarrowSplitMCell; 3] = [
    TnNarrowSplitMCell {
        shape: F32TriadShape {
            m: 1024,
            k: 47,
            n: 17,
            lda: 47,
            ldb: 17,
            ldc: 17,
        },
        m_chunk: 32,
        chunks: 32,
    },
    TnNarrowSplitMCell {
        shape: F32TriadShape {
            m: 1024,
            k: 128,
            n: 25,
            lda: 128,
            ldb: 25,
            ldc: 25,
        },
        m_chunk: 32,
        chunks: 32,
    },
    TnNarrowSplitMCell {
        shape: F32TriadShape {
            m: 4096,
            k: 64,
            n: 64,
            lda: 64,
            ldb: 64,
            ldc: 64,
        },
        m_chunk: 48,
        chunks: 86,
    },
];

fn qualified_tn_narrow_splitm_cell(request: F32TriadRequest) -> Option<TnNarrowSplitMCell> {
    (request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn).then_some(())?;
    TN_NARROW_SPLITM_SM120_CC120_170_CELLS
        .iter()
        .copied()
        .find(|cell| cell.shape == request.shape)
}

/// The facts a tuned scalar cell is admitted on. The f32 policy is not one
/// of them: a tuned exact-FMA route is the floor the TF32 policy falls back
/// to, so it serves under either policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScalarLaunchFacts {
    pub scalar_artifact: ArtifactIdentity,
    pub scalar_compiler: CompilerIdentity,
    pub compute_capability: (u32, u32),
    pub multiprocessor_count: u32,
}

fn qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts: ScalarLaunchFacts) -> bool {
    facts.compute_capability == (12, 0)
        && facts.multiprocessor_count == 170
        && facts.scalar_artifact.module_kind == ModuleKind::TriadScalar
        && facts.scalar_artifact.artifact_kind == facts.scalar_compiler.output_kind
        && facts.scalar_artifact.compile_key == facts.scalar_compiler.invocation_digest
        && facts.scalar_artifact.artifact_digest != [0; 32]
        && facts.scalar_compiler.invocation_digest != [0; 32]
        && facts.scalar_compiler.target.as_str() == "compute_120"
        && facts.scalar_compiler.nvrtc_version == (13, 2)
        && facts.scalar_compiler.nvrtc_library_known
        && facts.scalar_compiler.nvrtc_library_domain != [0; 32]
}

const NN_M64N64_SM120_CC120_170_NVRTC132_CELLS: [F32TriadShape; 7] = [
    F32TriadShape {
        m: 2048,
        k: 3072,
        n: 768,
        lda: 3072,
        ldb: 768,
        ldc: 768,
    },
    F32TriadShape {
        m: 4096,
        k: 3072,
        n: 1536,
        lda: 3072,
        ldb: 1536,
        ldc: 1536,
    },
    F32TriadShape {
        m: 512,
        k: 3072,
        n: 768,
        lda: 3072,
        ldb: 768,
        ldc: 768,
    },
    F32TriadShape {
        m: 4096,
        k: 512,
        n: 768,
        lda: 512,
        ldb: 768,
        ldc: 768,
    },
    F32TriadShape {
        m: 2048,
        k: 768,
        n: 3072,
        lda: 768,
        ldb: 3072,
        ldc: 3072,
    },
    F32TriadShape {
        m: 2048,
        k: 1536,
        n: 768,
        lda: 1536,
        ldb: 768,
        ldc: 768,
    },
    F32TriadShape {
        m: 4621,
        k: 384,
        n: 1928,
        lda: 384,
        ldb: 1928,
        ldc: 1928,
    },
];

fn qualified_nn_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn
        && NN_M64N64_SM120_CC120_170_NVRTC132_CELLS.contains(&request.shape)
}

fn qualified_nn_m64n64_operands(operands: F32TriadOperands) -> bool {
    const ALIGNMENT: u64 = 16;
    operands.bias.is_none()
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 0.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(ALIGNMENT))
}

const NN_M32N64_SPLITK32_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 128,
    k: 8_192,
    n: 128,
    lda: 8_192,
    ldb: 128,
    ldc: 128,
};

fn qualified_nn_m32n64_splitk32_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn
        && request.shape == NN_M32N64_SPLITK32_SM120_CC120_170_NVRTC132_CELL
}

fn qualified_nn_m32n64_splitk32_operands(operands: F32TriadOperands) -> bool {
    qualified_nn_m64n64_operands(operands)
}

const NT_D768_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 768,
    n: 3072,
    lda: 3072,
    ldb: 3072,
    ldc: 768,
};

const NT_D768_OUT_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 2048,
    k: 1536,
    n: 768,
    lda: 768,
    ldb: 768,
    ldc: 1536,
};

const NT_D768_OUT_TRANSPOSE_ELEMENTS: usize = 1_179_648;

const NT_LARGE_DEEP_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 4096,
    k: 3072,
    n: 1536,
    lda: 1536,
    ldb: 1536,
    ldc: 3072,
};

const NT_PRISM_VECTOR_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 4621,
    k: 384,
    n: 1928,
    lda: 1928,
    ldb: 1928,
    ldc: 384,
};

const NT_PRISM_TRANSPOSE_ELEMENTS: usize = 740_352;

const NT_D128_OUT_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 1024,
    k: 256,
    n: 128,
    lda: 128,
    ldb: 128,
    ldc: 256,
};

const NT_D128_OUT_TRANSPOSE_ELEMENTS: usize = 32_768;

const TN_M16N16_SPLITM16_SM120_CC120_170_NVRTC132_CELL: F32TriadShape = F32TriadShape {
    m: 256,
    k: 512,
    n: 384,
    lda: 512,
    ldb: 384,
    ldc: 384,
};

fn qualified_nt_m2n16_splitk32_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape
            == F32TriadShape::contiguous(
                crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt,
                (512, 16, 2_048),
            )
}

fn qualified_tn_m16n16_splitm16_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn
        && request.shape == TN_M16N16_SPLITM16_SM120_CC120_170_NVRTC132_CELL
}

fn qualified_tn_m16n16_splitm16_operands(operands: F32TriadOperands) -> bool {
    const ALIGNMENT: u64 = 16;
    operands.bias.is_none()
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 1.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(ALIGNMENT))
}

fn qualified_nt_d768_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL
}

fn qualified_nt_d768_out_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D768_OUT_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_D768_OUT_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_large_deep_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_LARGE_DEEP_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL
        && request.shape.k.checked_mul(request.shape.n)
            == Some(super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS)
}

fn qualified_nt_prism_vector_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_PRISM_VECTOR_SM120_CC120_170_NVRTC132_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_PRISM_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_d128_out_transpose_m64n64_request(request: F32TriadRequest) -> bool {
    request.op == crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt
        && request.shape == NT_D128_OUT_TRANSPOSE_M64N64_SM120_CC120_170_NVRTC132_CELL
        && request.shape.k.checked_mul(request.shape.n) == Some(NT_D128_OUT_TRANSPOSE_ELEMENTS)
}

fn qualified_nt_d768_transpose_m64n64_operands(operands: F32TriadOperands) -> bool {
    const ALIGNMENT: u64 = 16;
    operands.bias.is_none()
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 0.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(ALIGNMENT))
}

fn qualified_tn_narrow_splitm_operands(operands: F32TriadOperands) -> bool {
    let alignment = std::mem::align_of::<f32>() as u64;
    operands.bias.is_none()
        && operands.alpha.to_bits() == 1.0_f32.to_bits()
        && operands.beta.to_bits() == 1.0_f32.to_bits()
        && [operands.output, operands.a, operands.b]
            .into_iter()
            .all(|pointer| pointer != 0 && pointer.is_multiple_of(alignment))
}

/// Resolves the scalar physical launch plan from request, operands, and the
/// complete device/compiler policy facts available to both prepared and raw
/// launch paths. Unsupported evidence cells retain the ordinary scalar plan.
pub(super) fn scalar_launch_plan(
    facts: ScalarLaunchFacts,
    request: F32TriadRequest,
    operands: F32TriadOperands,
) -> Result<ScalarDispatchPlan, String> {
    let fallback = scalar_dispatch_plan(request, facts.multiprocessor_count)?;
    if fallback
        == (ScalarDispatchPlan::NtSplitKMain {
            n_main: 2_048,
            n_tail: 0,
        })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_m2n16_splitk32_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtM2N16SplitK32Qualified);
    }
    if fallback == (ScalarDispatchPlan::NnFinal { slim: false })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nn_m64n64_request(request)
        && qualified_nn_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NnM64N64Qualified);
    }
    if fallback == ScalarDispatchPlan::NnSplitKThin
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nn_m32n64_splitk32_request(request)
        && qualified_nn_m32n64_splitk32_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NnM32N64SplitK32Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_d768_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768TransposeM64N64Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_d768_out_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: false })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_large_deep_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified);
    }
    if fallback == (ScalarDispatchPlan::NtFinal { slim: true })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_prism_vector_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtPrismVectorQualified);
    }
    if fallback
        == (ScalarDispatchPlan::NtSplitKMain {
            n_main: 128,
            n_tail: 0,
        })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_nt_d128_out_transpose_m64n64_request(request)
        && qualified_nt_d768_transpose_m64n64_operands(operands)
    {
        return Ok(ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified);
    }
    if fallback
        == (ScalarDispatchPlan::TnSplitM {
            m_chunk: 16,
            chunks: 16,
        })
        && qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        && qualified_tn_m16n16_splitm16_request(request)
        && qualified_tn_m16n16_splitm16_operands(operands)
    {
        return Ok(ScalarDispatchPlan::TnM16N16SplitM16Qualified);
    }
    if fallback != ScalarDispatchPlan::TnNarrow
        || !qualified_scalar_sm120_cc120_170_nvrtc132_environment(facts)
        || !qualified_tn_narrow_splitm_operands(operands)
    {
        return Ok(fallback);
    }
    let Some(cell) = qualified_tn_narrow_splitm_cell(request) else {
        return Ok(fallback);
    };
    Ok(ScalarDispatchPlan::TnNarrowSplitM {
        m_chunk: cell.m_chunk,
        chunks: cell.chunks,
    })
}

/// Minimum N (output cols) before the dispatcher switches from Slim-N tiles
/// to Big-N tiles. Below this, Slim-N (BN=64) packs better; above it Big-N
/// (BN=128) wins on wave occupancy. The threshold once gated a cuBLAS
/// fallback; that fallback is gone, so this is only a tile-pick boundary.
pub(super) const GEMM_CUSTOM_MIN: usize = 128;

/// Boundary between Slim-N and Big tile variants (by output N dimension).
pub(super) const GEMM_SLIM_MAX: usize = 512;

/// v6.5 Phase C-1.5av: separate Slim Split-K NT-via-T n_in cap for backward dx.
/// The forward Slim NN path uses N as output dim → GEMM_SLIM_MAX=512 bounds
/// wave-fill correctness there. But NT-via-T backward dx reads n_in (input dim
/// of original forward), and the kernel itself tiles arbitrary n_in via N-axis
/// tiling — the 512 cap is conservative, not load-bearing. v6.5 multi-step
/// Production input widths can exceed 512; 768 keeps those shapes on Slim
/// Split-K NT-via-T
/// with F=4 K-tile partials (576 blocks vs plain Big NT 144 blocks).
/// Determinism preserved: F is shape-keyed (function of n_out, not batch).
pub(super) const GEMM_SLIM_NT_NIN_MAX: usize = 768;

/// M threshold below which we force Slim-N even for N ≥ 129 (wave underfill protection).
/// At M < 512, Big tile BM=128 gives ≤4 M-blocks; adding N-blocks via Slim's BN=64 (vs Big's BN=128)
/// doubles grid to reduce wave underfill on Ada's 142 SMs. Only matters when N ≥ 129 (otherwise slim already chosen).
pub(super) const GEMM_M_SLIM_FORCE: usize = 512;

/// Single source of truth for Split-K/M scratch buffer cap, in f32 elements.
/// Must match `splitk_scratch` allocation in `kernels.rs` (1 << 23 = 8M f32 = 32 MB).
/// All Split-K dispatch gates (NN fwd, NT bwd_dx, Split-M TN bwd_dw) read this.
pub(super) const SPLITK_SCRATCH_CAP: usize = 1 << 23;

/// One device-scope completion counter per fused TF32 split-K output tile.
pub(super) const TF32_SPLITK_COUNTER_CAP: usize = 1 << 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScalarDispatchPlan {
    NnUltraThin,
    NnNarrowSmall,
    NnNarrow,
    NnGemv,
    NnSplitKThinTail { k_main: usize, k_tail: usize },
    NnSplitKThin,
    NnSplitKSlim { chunks: u32 },
    NnM32N64SplitK32Qualified,
    NnM64N64Qualified,
    NnFinal { slim: bool },
    TnGemv,
    TnNarrow,
    TnNarrowSplitM { m_chunk: usize, chunks: usize },
    TnSplitM { m_chunk: usize, chunks: usize },
    TnM16N16SplitM16Qualified,
    TnFinal { slim: bool },
    NtNarrow,
    NtSmallBatchWide,
    NtGemv,
    NtSplitKTail { k_main: usize, k_tail: usize },
    NtSplitKMain { n_main: usize, n_tail: usize },
    NtSplitKSlim { chunks: u32 },
    NtMidBatchWide,
    NtM2N16SplitK32Qualified,
    NtD768TransposeM64N64Qualified,
    NtD768OutTransposeM64N64Qualified,
    NtLargeDeepTransposeM64N64Qualified,
    NtPrismVectorQualified,
    NtD128OutTransposeM64N64Qualified,
    NtFinal { slim: bool },
}

impl ScalarDispatchPlan {
    pub(super) const fn needs_split_scratch(self) -> bool {
        matches!(
            self,
            Self::NnSplitKThinTail { .. }
                | Self::NnSplitKThin
                | Self::NnSplitKSlim { .. }
                | Self::NnM32N64SplitK32Qualified
                | Self::TnNarrowSplitM { .. }
                | Self::TnSplitM { .. }
                | Self::NtSplitKTail { .. }
                | Self::NtSplitKMain { .. }
                | Self::NtSplitKSlim { .. }
        )
    }

    pub(super) const fn needs_transpose_scratch(self) -> bool {
        matches!(
            self,
            Self::NtSplitKTail { .. }
                | Self::NtSplitKMain { .. }
                | Self::NtSplitKSlim { .. }
                | Self::NtD768TransposeM64N64Qualified
                | Self::NtD768OutTransposeM64N64Qualified
                | Self::NtLargeDeepTransposeM64N64Qualified
                | Self::NtPrismVectorQualified
                | Self::NtD128OutTransposeM64N64Qualified
        )
    }
}

fn slim_final(output_rows: usize, output_columns: usize) -> bool {
    output_columns <= GEMM_SLIM_MAX
        || (output_rows < GEMM_M_SLIM_FORCE && output_columns >= GEMM_CUSTOM_MIN)
}

fn scalar_wave_policy(
    multiprocessor_count: u32,
) -> Result<crate::mamba_ssm::gpu::kernel_identity::ScalarWavePolicyV1, String> {
    if multiprocessor_count == 0 {
        return Err("scalar dispatch requires a nonzero multiprocessor count".into());
    }
    let policy = crate::mamba_ssm::gpu::kernel_identity::ScalarWavePolicyV1::current();
    if policy.thin_split_wave_denominator == 0
        || policy.slim_split_wave_denominator == 0
        || policy.tn_split_m_wave_denominator == 0
    {
        return Err("scalar dispatch wave policy has a zero denominator".into());
    }
    Ok(policy)
}

fn grid_below_waves(
    blocks: u32,
    multiprocessor_count: u32,
    numerator: u64,
    denominator: u64,
) -> Result<bool, String> {
    let blocks = u64::from(blocks)
        .checked_mul(denominator)
        .ok_or_else(|| "scalar dispatch block-wave comparison overflows u64".to_string())?;
    let target = u64::from(multiprocessor_count)
        .checked_mul(numerator)
        .ok_or_else(|| "scalar dispatch SM-wave comparison overflows u64".to_string())?;
    Ok(blocks < target)
}

fn wave_target_blocks(
    multiprocessor_count: u32,
    numerator: u64,
    denominator: u64,
) -> Result<u32, String> {
    let scaled = u64::from(multiprocessor_count)
        .checked_mul(numerator)
        .ok_or_else(|| "scalar dispatch target wave count overflows u64".to_string())?;
    u32::try_from(scaled.div_ceil(denominator))
        .map_err(|_| "scalar dispatch target block count exceeds u32::MAX".into())
}

fn scalar_nn_plan(dims: GemmDims, multiprocessor_count: u32) -> Result<ScalarDispatchPlan, String> {
    let policy = scalar_wave_policy(multiprocessor_count)?;
    let (batch, n_in, n_out) = dims.tuple();
    if (1..32).contains(&batch) && (32..=2048).contains(&n_in) && n_out >= 32 {
        return Ok(ScalarDispatchPlan::NnUltraThin);
    }
    if (2..=127).contains(&n_out) && (1..=64).contains(&batch) {
        return Ok(ScalarDispatchPlan::NnNarrowSmall);
    }
    if (2..=127).contains(&n_out) {
        return Ok(ScalarDispatchPlan::NnNarrow);
    }
    if n_out == 1 && n_in >= 32 {
        return Ok(ScalarDispatchPlan::NnGemv);
    }
    let plain_slim_blocks = checked_tile_grid(dims.m_u32, 128, dims.n_u32, 64)?;
    let underfill = grid_below_waves(
        plain_slim_blocks,
        multiprocessor_count,
        policy.thin_split_wave_numerator,
        policy.thin_split_wave_denominator,
    )?;
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && underfill
    {
        let k_tail = n_in % 32;
        let k_main = n_in - k_tail;
        if k_main >= 32
            && checked_mul3(k_main / 32, batch, n_out, "NN K-tail scratch")? <= SPLITK_SCRATCH_CAP
        {
            return Ok(ScalarDispatchPlan::NnSplitKThinTail { k_main, k_tail });
        }
    }
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 32
        && n_in.is_multiple_of(32)
        && checked_mul3(n_in / 32, batch, n_out, "NN split-K scratch")? <= SPLITK_SCRATCH_CAP
        && underfill
    {
        return Ok(ScalarDispatchPlan::NnSplitKThin);
    }
    if batch > 1024
        && (128..=GEMM_SLIM_MAX).contains(&n_out)
        && n_in >= 64
        && n_in.is_multiple_of(32)
    {
        let chunks = dims.k_u32.div_ceil(64);
        if chunks >= 6
            && checked_mul3(
                checked_usize(chunks, "NN slim split-K chunks")?,
                batch,
                n_out,
                "NN slim split-K scratch",
            )? <= SPLITK_SCRATCH_CAP
            && plain_slim_blocks > 0
            && grid_below_waves(
                plain_slim_blocks,
                multiprocessor_count,
                policy.slim_split_wave_numerator,
                policy.slim_split_wave_denominator,
            )?
        {
            return Ok(ScalarDispatchPlan::NnSplitKSlim { chunks });
        }
    }
    if batch < 128 && n_out >= 128 {
        return Ok(ScalarDispatchPlan::NnNarrow);
    }
    if batch >= GEMM_CUSTOM_MIN && n_out >= GEMM_CUSTOM_MIN {
        return Ok(ScalarDispatchPlan::NnFinal {
            slim: slim_final(batch, n_out),
        });
    }
    Err(format!(
        "UNCOVERED scalar NN route M={batch} K={n_in} N={n_out}"
    ))
}

pub(in crate::mamba_ssm::gpu) fn tc_half_policy_prefers_scalar_forward(
    compute_capability: (u32, u32),
    dims: (usize, usize, usize),
    multiprocessor_count: u32,
) -> Result<bool, String> {
    let policy = super::super::kernel_identity::Sm80TcPolicyV3::current();
    if compute_capability != policy.deep_split_k_compute_capability
        || dims.2 != policy.deep_split_k_output_columns
    {
        return Ok(false);
    }
    let plan = scalar_nn_plan(GemmDims::nn(dims, dims.1)?, multiprocessor_count)?;
    Ok(match plan {
        ScalarDispatchPlan::NnSplitKThinTail { .. } => {
            dims.1 >= policy.deep_split_k_tail_min_reduction
        }
        ScalarDispatchPlan::NnSplitKThin => dims.1 >= policy.deep_split_k_aligned_min_reduction,
        _ => false,
    })
}

fn scalar_tn_plan(dims: GemmDims, multiprocessor_count: u32) -> Result<ScalarDispatchPlan, String> {
    scalar_wave_policy(multiprocessor_count)?;
    let (batch, n_in, n_out) = dims.tuple();
    if n_out == 1 && n_in >= 4 && batch >= 32 {
        return Ok(ScalarDispatchPlan::TnGemv);
    }
    if (2..=127).contains(&n_out) {
        return Ok(ScalarDispatchPlan::TnNarrow);
    }
    if let Some((m_chunk, chunks)) = splitm_tn_partition(batch, n_in, n_out, multiprocessor_count)?
    {
        return Ok(ScalarDispatchPlan::TnSplitM { m_chunk, chunks });
    }
    if n_in >= 1 && n_out >= GEMM_CUSTOM_MIN {
        return Ok(ScalarDispatchPlan::TnFinal {
            slim: slim_final(n_in, n_out),
        });
    }
    Err(format!(
        "UNCOVERED scalar TN route M={batch} K={n_in} N={n_out}"
    ))
}

fn scalar_nt_plan(dims: GemmDims, multiprocessor_count: u32) -> Result<ScalarDispatchPlan, String> {
    let policy = scalar_wave_policy(multiprocessor_count)?;
    let (batch, n_in, n_out) = dims.tuple();
    if (2..=127).contains(&n_out) {
        return Ok(ScalarDispatchPlan::NtNarrow);
    }
    if n_out == 1 {
        return Ok(ScalarDispatchPlan::NtGemv);
    }
    if matches!((batch, n_in, n_out), (512, 16, 2048) | (16, 512, 2048)) {
        return Ok(ScalarDispatchPlan::NtSplitKMain {
            n_main: 2048,
            n_tail: 0,
        });
    }
    let plain_slim_blocks = checked_tile_grid(dims.m_u32, 128, dims.k_u32, 64)?;
    let underfill = grid_below_waves(
        plain_slim_blocks,
        multiprocessor_count,
        policy.thin_split_wave_numerator,
        policy.thin_split_wave_denominator,
    )?;
    if batch < 32 && n_out >= 128 {
        return Ok(ScalarDispatchPlan::NtSmallBatchWide);
    }
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_out.is_multiple_of(32)
        && underfill
    {
        let k_tail = n_in % 32;
        let k_main = n_in - k_tail;
        if k_main >= 32
            && k_main.checked_mul(n_out).is_some_and(|elements| {
                elements <= super::contract::SCALAR_GENERIC_TRANSPOSE_ROUTE_CAP_ELEMENTS
            })
            && checked_mul3(n_out / 32, batch, k_main, "NT K-tail scratch")? <= SPLITK_SCRATCH_CAP
        {
            return Ok(ScalarDispatchPlan::NtSplitKTail { k_main, k_tail });
        }
    }
    let n_tail = n_out % 32;
    let n_main = n_out - n_tail;
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_main >= 32
        && dims.kn <= super::contract::SCALAR_GENERIC_TRANSPOSE_ROUTE_CAP_ELEMENTS
        && checked_mul3(n_main / 32, batch, n_in, "NT split-K scratch")? <= SPLITK_SCRATCH_CAP
        && underfill
    {
        return Ok(ScalarDispatchPlan::NtSplitKMain { n_main, n_tail });
    }
    if batch > 1024
        && (128..=GEMM_SLIM_NT_NIN_MAX).contains(&n_in)
        && n_out >= 64
        && n_out.is_multiple_of(32)
        && dims.kn <= super::contract::SCALAR_GENERIC_TRANSPOSE_ROUTE_CAP_ELEMENTS
    {
        let chunks = dims.n_u32.div_ceil(64);
        if chunks >= 2
            && checked_mul3(
                checked_usize(chunks, "NT slim split-K chunks")?,
                batch,
                n_in,
                "NT slim split-K scratch",
            )? <= SPLITK_SCRATCH_CAP
            && plain_slim_blocks > 0
            && grid_below_waves(
                plain_slim_blocks,
                multiprocessor_count,
                policy.slim_split_wave_numerator,
                policy.slim_split_wave_denominator,
            )?
        {
            return Ok(ScalarDispatchPlan::NtSplitKSlim { chunks });
        }
    }
    if (32..128).contains(&batch) && n_out >= 128 {
        return Ok(ScalarDispatchPlan::NtMidBatchWide);
    }
    if batch >= GEMM_CUSTOM_MIN {
        return Ok(ScalarDispatchPlan::NtFinal {
            slim: slim_final(batch, n_in),
        });
    }
    Err(format!(
        "UNCOVERED scalar NT route M={batch} K={n_in} N={n_out}"
    ))
}

pub(super) fn scalar_dispatch_plan(
    request: F32TriadRequest,
    multiprocessor_count: u32,
) -> Result<ScalarDispatchPlan, String> {
    request.shape.validate(request.op)?;
    scalar_wave_policy(multiprocessor_count)?;
    let dims = (request.shape.m, request.shape.k, request.shape.n);
    match request.op {
        crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn => {
            scalar_nn_plan(GemmDims::nn(dims, request.shape.lda)?, multiprocessor_count)
        }
        crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Tn => {
            scalar_tn_plan(GemmDims::tn(dims)?, multiprocessor_count)
        }
        crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nt => {
            scalar_nt_plan(GemmDims::nt(dims)?, multiprocessor_count)
        }
    }
}

pub(super) fn nn_routes_to_big(
    batch: usize,
    n_in: usize,
    n_out: usize,
    multiprocessor_count: u32,
) -> bool {
    GemmDims::nn((batch, n_in, n_out), n_in)
        .and_then(|dims| scalar_nn_plan(dims, multiprocessor_count))
        .is_ok_and(|plan| matches!(plan, ScalarDispatchPlan::NnFinal { slim: false }))
}

pub(super) fn tn_routes_to_big(
    batch: usize,
    n_in: usize,
    n_out: usize,
    multiprocessor_count: u32,
) -> bool {
    GemmDims::tn((batch, n_in, n_out))
        .and_then(|dims| scalar_tn_plan(dims, multiprocessor_count))
        .is_ok_and(|plan| matches!(plan, ScalarDispatchPlan::TnFinal { slim: false }))
}

pub(super) fn nt_routes_to_big(
    batch: usize,
    n_in: usize,
    n_out: usize,
    multiprocessor_count: u32,
) -> bool {
    GemmDims::nt((batch, n_in, n_out))
        .and_then(|dims| scalar_nt_plan(dims, multiprocessor_count))
        .is_ok_and(|plan| matches!(plan, ScalarDispatchPlan::NtFinal { slim: false }))
}

#[cfg(test)]
mod scalar_wave_policy_tests {
    use super::{ScalarDispatchPlan, ScalarLaunchFacts, scalar_dispatch_plan, scalar_launch_plan};
    use crate::mamba_ssm::gpu::gemm_bi_triad::{F32TriadOperands, F32TriadRequest, F32TriadShape};
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, ModuleKind, NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    };

    const RTX_6000_ADA_SMS: u32 = 142;

    fn plan(
        op: ResolvedGemmOp,
        dims: (usize, usize, usize),
        multiprocessor_count: u32,
    ) -> Result<ScalarDispatchPlan, String> {
        scalar_dispatch_plan(
            F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, dims),
            },
            multiprocessor_count,
        )
    }

    fn tn_admission_compiler(
        target: &str,
        nvrtc_version: (i32, i32),
        library_domain: [u8; 32],
        library_known: bool,
    ) -> CompilerIdentity {
        CompilerIdentity {
            source_digest: [1; 32],
            invocation_digest: [2; 32],
            header_manifest_digest: [3; 32],
            target: CudaTarget::new(target).unwrap(),
            nvrtc_version,
            nvrtc_library_domain: library_domain,
            nvrtc_library_known: library_known,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        }
    }

    fn tn_admission_facts() -> ScalarLaunchFacts {
        let compiler = tn_admission_compiler("compute_120", (13, 2), [4; 32], true);
        ScalarLaunchFacts {
            scalar_artifact: ArtifactIdentity {
                module_kind: ModuleKind::TriadScalar,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: compiler.invocation_digest,
                artifact_digest: [6; 32],
            },
            scalar_compiler: compiler,
            compute_capability: (12, 0),
            multiprocessor_count: 170,
        }
    }

    fn tn_admission_request(dims: (usize, usize, usize)) -> F32TriadRequest {
        F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
        }
    }

    fn tn_admission_operands() -> F32TriadOperands {
        F32TriadOperands {
            output: 0x3000,
            a: 0x1000,
            b: 0x2000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        }
    }

    fn admitted_tn_plan(
        facts: ScalarLaunchFacts,
        request: F32TriadRequest,
        operands: F32TriadOperands,
    ) -> ScalarDispatchPlan {
        scalar_launch_plan(facts, request, operands).unwrap()
    }

    fn nn_qualified_request(dims: (usize, usize, usize)) -> F32TriadRequest {
        F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
        }
    }

    fn nn_qualified_operands() -> F32TriadOperands {
        F32TriadOperands {
            output: 0x3000,
            a: 0x1000,
            b: 0x2000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        }
    }

    #[test]
    fn scalar_dispatch_rejects_zero_multiprocessors() {
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let error = plan(op, (128, 128, 128), 0)
                .expect_err("scalar dispatch requires a nonzero multiprocessor count");
            assert!(error.contains("multiprocessor count"), "{error}");
        }
    }

    #[test]
    fn nn_thin_split_admission_uses_the_live_multiprocessor_count() {
        let dims = (512, 64, 2048);
        assert_eq!(
            plan(ResolvedGemmOp::Nn, dims, 108).unwrap(),
            ScalarDispatchPlan::NnFinal { slim: false }
        );
        assert_eq!(
            plan(ResolvedGemmOp::Nn, dims, RTX_6000_ADA_SMS).unwrap(),
            ScalarDispatchPlan::NnSplitKThin
        );
    }

    #[test]
    fn nt_split_admission_uses_the_live_multiprocessor_count() {
        let dims = (512, 2048, 128);
        assert_eq!(
            plan(ResolvedGemmOp::Nt, dims, 108).unwrap(),
            ScalarDispatchPlan::NtFinal { slim: false }
        );
        assert_eq!(
            plan(ResolvedGemmOp::Nt, dims, RTX_6000_ADA_SMS).unwrap(),
            ScalarDispatchPlan::NtSplitKMain {
                n_main: 128,
                n_tail: 0,
            }
        );
    }

    #[test]
    fn nt_generic_split_admission_keeps_the_pre_large_deep_scratch_boundary() {
        assert_eq!(
            plan(ResolvedGemmOp::Nt, (32, 2_304, 2_048), 170).unwrap(),
            ScalarDispatchPlan::NtMidBatchWide
        );
    }

    #[test]
    fn nt_split_admission_promotes_the_two_qualified_thin_output_cells() {
        for multiprocessor_count in [20, 56, 82, 108, 120, 128, 132, 142, 148, 170] {
            for dims in [(512, 16, 2048), (16, 512, 2048)] {
                assert_eq!(
                    plan(ResolvedGemmOp::Nt, dims, multiprocessor_count).unwrap(),
                    ScalarDispatchPlan::NtSplitKMain {
                        n_main: 2048,
                        n_tail: 0,
                    },
                    "{dims:?} {multiprocessor_count} SMs"
                );
            }
        }

        for (dims, expected) in [
            ((512, 17, 2048), ScalarDispatchPlan::NtFinal { slim: true }),
            ((17, 512, 2048), ScalarDispatchPlan::NtSmallBatchWide),
            ((512, 16, 2016), ScalarDispatchPlan::NtFinal { slim: true }),
            ((16, 512, 2016), ScalarDispatchPlan::NtSmallBatchWide),
        ] {
            assert_eq!(
                plan(ResolvedGemmOp::Nt, dims, 170).unwrap(),
                expected,
                "neighbor {dims:?}"
            );
        }
    }

    #[test]
    fn nt_m2n16_exact_cell_requires_shape_operands_and_qualified_environment() {
        let facts = tn_admission_facts();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (512, 16, 2_048)),
        };
        let operands = nn_qualified_operands();
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NtM2N16SplitK32Qualified
        );
        for dims in [
            (511, 16, 2_048),
            (513, 16, 2_048),
            (512, 15, 2_048),
            (512, 17, 2_048),
            (512, 16, 2_047),
            (512, 16, 2_049),
        ] {
            let neighbor = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
            };
            assert_ne!(
                scalar_launch_plan(facts, neighbor, operands).unwrap(),
                ScalarDispatchPlan::NtM2N16SplitK32Qualified,
                "neighbor {dims:?}"
            );
        }
        for mutation in [
            F32TriadOperands {
                alpha: -1.0,
                ..operands
            },
            F32TriadOperands {
                beta: 1.0,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(facts, request, mutation).unwrap(),
                ScalarDispatchPlan::NtM2N16SplitK32Qualified
            );
        }
        let mut mutations = [facts; 3];
        mutations[0].compute_capability = (8, 9);
        mutations[1].multiprocessor_count = 169;
        mutations[2].scalar_compiler.nvrtc_version = (13, 0);
        for mutation in mutations {
            assert_ne!(
                scalar_launch_plan(mutation, request, operands).unwrap(),
                ScalarDispatchPlan::NtM2N16SplitK32Qualified
            );
        }
    }

    #[test]
    fn nn_m32n64_splitk32_exact_cell_is_fail_closed() {
        let facts = tn_admission_facts();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (128, 8_192, 128)),
        };
        let operands = nn_qualified_operands();
        let fallback = scalar_dispatch_plan(request, facts.multiprocessor_count).unwrap();
        assert_eq!(fallback, ScalarDispatchPlan::NnSplitKThin);
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NnM32N64SplitK32Qualified
        );

        for dims in [
            (127, 8_192, 128),
            (129, 8_192, 128),
            (128, 8_191, 128),
            (128, 8_193, 128),
            (128, 8_192, 127),
            (128, 8_192, 129),
        ] {
            let neighbor = F32TriadRequest {
                op: ResolvedGemmOp::Nn,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
            };
            assert_ne!(
                scalar_launch_plan(facts, neighbor, operands).unwrap(),
                ScalarDispatchPlan::NnM32N64SplitK32Qualified,
                "neighbor {dims:?}"
            );
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            let strided = F32TriadRequest {
                op: ResolvedGemmOp::Nn,
                shape,
            };
            assert_ne!(
                scalar_launch_plan(facts, strided, operands).unwrap(),
                ScalarDispatchPlan::NnM32N64SplitK32Qualified
            );
        }
        for op in [ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let wrong_op = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (128, 8_192, 128)),
            };
            assert_ne!(
                scalar_launch_plan(facts, wrong_op, operands).unwrap(),
                ScalarDispatchPlan::NnM32N64SplitK32Qualified
            );
        }
        for mutation in [
            F32TriadOperands {
                alpha: -1.0,
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
        ] {
            assert_eq!(
                scalar_launch_plan(facts, request, mutation).unwrap(),
                fallback
            );
        }

        let mut mutations = [facts; 13];
        mutations[0].compute_capability = (8, 9);
        mutations[1].compute_capability = (12, 1);
        mutations[2].multiprocessor_count = 169;
        mutations[3].scalar_compiler.nvrtc_version = (13, 0);
        mutations[4].scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        mutations[5].scalar_compiler.nvrtc_library_known = false;
        mutations[6].scalar_compiler.nvrtc_library_domain = [0; 32];
        mutations[7].scalar_artifact.compile_key[0] ^= 1;
        mutations[8].scalar_artifact.artifact_digest = [0; 32];
        mutations[9].scalar_artifact.module_kind = ModuleKind::TriadSm80;
        mutations[10].scalar_compiler.invocation_digest = [0; 32];
        mutations[11].scalar_artifact.artifact_kind = ArtifactKind::Cubin;
        mutations[12].scalar_compiler.output_kind = ArtifactKind::Cubin;
        for mutation in mutations {
            assert_eq!(
                scalar_launch_plan(mutation, request, operands).unwrap(),
                fallback
            );
        }
    }

    #[test]
    fn tn_split_m_partition_targets_two_live_waves() {
        let dims = (4096, 128, 128);
        assert_eq!(
            plan(ResolvedGemmOp::Tn, dims, 108).unwrap(),
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 32,
                chunks: 128,
            }
        );
        assert_eq!(
            plan(ResolvedGemmOp::Tn, dims, RTX_6000_ADA_SMS).unwrap(),
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 16,
                chunks: 256,
            }
        );
    }

    #[test]
    fn tn_small_input_splitm_gap_is_covered_by_the_generic_member() {
        let plan = plan(ResolvedGemmOp::Tn, (4096, 24, 768), RTX_6000_ADA_SMS).unwrap();
        assert!(matches!(
            plan,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 96,
                chunks: 43,
            }
        ));
    }

    #[test]
    fn tn_narrow_split_m_selector_admits_only_the_three_exact_cells() {
        for (dims, expected) in [
            ((1024, 47, 17), (32, 32)),
            ((1024, 128, 25), (32, 32)),
            ((4096, 64, 64), (48, 86)),
        ] {
            assert_eq!(
                admitted_tn_plan(
                    tn_admission_facts(),
                    tn_admission_request(dims),
                    tn_admission_operands(),
                ),
                ScalarDispatchPlan::TnNarrowSplitM {
                    m_chunk: expected.0,
                    chunks: expected.1,
                },
                "{dims:?}"
            );
        }

        let request = tn_admission_request((1024, 47, 17));
        let operands = tn_admission_operands();
        for admitted in [
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                b: operands.b + 4,
                ..operands
            },
        ] {
            assert_eq!(
                admitted_tn_plan(tn_admission_facts(), request, admitted),
                ScalarDispatchPlan::TnNarrowSplitM {
                    m_chunk: 32,
                    chunks: 32,
                }
            );
        }

        for dims in [
            (256, 32, 2),
            (4096, 128, 96),
            (4096, 256, 101),
            (4111, 257, 127),
        ] {
            assert_eq!(
                admitted_tn_plan(
                    tn_admission_facts(),
                    tn_admission_request(dims),
                    tn_admission_operands(),
                ),
                ScalarDispatchPlan::TnNarrow,
                "rejected tournament cell {dims:?}"
            );
        }
    }

    #[test]
    fn tn_narrow_split_m_selector_admits_from_independent_scalar_facts() {
        assert_eq!(
            admitted_tn_plan(
                tn_admission_facts(),
                tn_admission_request((1024, 47, 17)),
                tn_admission_operands(),
            ),
            ScalarDispatchPlan::TnNarrowSplitM {
                m_chunk: 32,
                chunks: 32,
            }
        );
    }

    #[test]
    fn tn_narrow_split_m_selector_requires_the_exact_device_and_compiler_domain() {
        let request = tn_admission_request((1024, 47, 17));
        let operands = tn_admission_operands();
        let fallback = ScalarDispatchPlan::TnNarrow;

        let mut wrong_arch = tn_admission_facts();
        wrong_arch.compute_capability = (12, 1);
        assert_eq!(admitted_tn_plan(wrong_arch, request, operands), fallback);

        let mut wrong_sm_count = tn_admission_facts();
        wrong_sm_count.multiprocessor_count = 169;
        assert_eq!(
            admitted_tn_plan(wrong_sm_count, request, operands),
            fallback
        );

        let mut wrong_nvrtc = tn_admission_facts();
        wrong_nvrtc.scalar_compiler.nvrtc_version = (13, 1);
        assert_eq!(admitted_tn_plan(wrong_nvrtc, request, operands), fallback);

        let mut unknown_library = tn_admission_facts();
        unknown_library.scalar_compiler.nvrtc_library_known = false;
        assert_eq!(
            admitted_tn_plan(unknown_library, request, operands),
            fallback
        );

        let mut wrong_compile_key = tn_admission_facts();
        wrong_compile_key.scalar_artifact.compile_key = [8; 32];
        assert_eq!(
            admitted_tn_plan(wrong_compile_key, request, operands),
            fallback
        );

        let mut wrong_target = tn_admission_facts();
        wrong_target.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        assert_eq!(admitted_tn_plan(wrong_target, request, operands), fallback);
    }

    #[test]
    fn tn_narrow_split_m_selector_rejects_neighboring_requests_and_epilogues() {
        let facts = tn_admission_facts();
        let request = tn_admission_request((1024, 47, 17));
        let operands = tn_admission_operands();
        let fallback = ScalarDispatchPlan::TnNarrow;

        let nn_request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: request.shape,
        };
        assert_eq!(
            admitted_tn_plan(facts, nn_request, operands),
            ScalarDispatchPlan::NnNarrow
        );

        for dims in [(1023, 47, 17), (1024, 48, 17), (1024, 47, 18)] {
            assert_eq!(
                admitted_tn_plan(facts, tn_admission_request(dims), operands),
                fallback,
                "shape neighbor {dims:?}"
            );
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_eq!(
                admitted_tn_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Tn,
                        shape,
                    },
                    operands,
                ),
                fallback,
                "stride neighbor {shape:?}"
            );
        }
        for rejected in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: 0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands {
                output: operands.output + 2,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands {
                a: operands.a + 2,
                ..operands
            },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                b: operands.b + 2,
                ..operands
            },
        ] {
            assert_eq!(admitted_tn_plan(facts, request, rejected), fallback);
        }
    }

    #[test]
    fn nn_m64n64_selector_admits_only_the_seven_measured_cells() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let cells = [
            (2_048, 3_072, 768),
            (4_096, 3_072, 1_536),
            (512, 3_072, 768),
            (4_096, 512, 768),
            (2_048, 768, 3_072),
            (2_048, 1_536, 768),
            (4_621, 384, 1_928),
        ];

        for dims in cells {
            let request = nn_qualified_request(dims);
            let plan = scalar_launch_plan(facts, request, operands).unwrap();
            assert_eq!(format!("{plan:?}"), "NnM64N64Qualified", "cell {dims:?}");

            let dimensions = [dims.0, dims.1, dims.2];
            for axis in 0..3 {
                for delta in [-1_isize, 1] {
                    let mut neighbor = dimensions;
                    neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                    let neighbor = (neighbor[0], neighbor[1], neighbor[2]);
                    let plan = scalar_launch_plan(facts, nn_qualified_request(neighbor), operands)
                        .unwrap();
                    assert_ne!(
                        format!("{plan:?}"),
                        "NnM64N64Qualified",
                        "shape neighbor {neighbor:?}"
                    );
                }
            }

            for shape in [
                F32TriadShape {
                    lda: request.shape.lda + 1,
                    ..request.shape
                },
                F32TriadShape {
                    ldb: request.shape.ldb + 1,
                    ..request.shape
                },
                F32TriadShape {
                    ldc: request.shape.ldc + 1,
                    ..request.shape
                },
            ] {
                let plan = scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nn,
                        shape,
                    },
                    operands,
                )
                .unwrap();
                assert_ne!(
                    format!("{plan:?}"),
                    "NnM64N64Qualified",
                    "stride mutation {shape:?}"
                );
            }
        }
    }

    #[test]
    fn nt_d768_transpose_m64n64_selector_is_an_exact_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 768, 3_072)),
        };
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NtD768TransposeM64N64Qualified
        );
        assert!(ScalarDispatchPlan::NtD768TransposeM64N64Qualified.needs_transpose_scratch());
        assert!(!ScalarDispatchPlan::NtD768TransposeM64N64Qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    ScalarDispatchPlan::NtD768TransposeM64N64Qualified
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            let mutated = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape,
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (2_048, 768, 3_072)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            );
        }
    }

    #[test]
    fn nt_d768_out_transpose_m64n64_selector_is_a_distinct_exact_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
        );
        assert_ne!(
            ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified,
            ScalarDispatchPlan::NtD768TransposeM64N64Qualified
        );
        assert!(ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified.needs_transpose_scratch());
        assert!(!ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape,
                    },
                    operands,
                )
                .unwrap(),
                ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (2_048, 1_536, 768)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
            );
        }
    }

    #[test]
    fn nt_large_deep_transpose_m64n64_selector_is_an_exact_capacity_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_096, 3_072, 1_536)),
        };
        assert_eq!(
            request.shape.k.checked_mul(request.shape.n),
            Some(super::super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS)
        );
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
        );
        assert!(ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified.needs_transpose_scratch());
        assert!(!ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape,
                    },
                    operands,
                )
                .unwrap(),
                ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (4_096, 3_072, 1_536)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
            );
        }
    }

    #[test]
    fn nt_large_deep_transpose_m64n64_rejects_environment_and_operand_mutations() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_096, 3_072, 1_536)),
        };
        let operands = nn_qualified_operands();
        let qualified = ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified;
        let mut fact_mutations = Vec::new();

        let mut facts = tn_admission_facts();
        facts.compute_capability = (12, 1);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.multiprocessor_count = 169;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 1);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_domain = [0; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.invocation_digest = [0; 32];
        facts.scalar_artifact.compile_key = [0; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.module_kind = ModuleKind::TriadSm80;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_kind = ArtifactKind::Cubin;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_digest = [0; 32];
        fact_mutations.push(facts);

        for facts in fact_mutations {
            assert_ne!(
                scalar_launch_plan(facts, request, operands).unwrap(),
                qualified
            );
        }
        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 12,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(tn_admission_facts(), request, mutation).unwrap(),
                qualified
            );
        }
    }

    #[test]
    fn nt_prism_vector_selector_is_an_exact_slim_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_621, 384, 1_928)),
        };
        let qualified = ScalarDispatchPlan::NtPrismVectorQualified;
        assert_eq!(
            scalar_dispatch_plan(request, 170).unwrap(),
            ScalarDispatchPlan::NtFinal { slim: true }
        );
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            qualified
        );
        assert!(qualified.needs_transpose_scratch());
        assert!(!qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    qualified,
                    "dimension neighbor {neighbor:?}"
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: request.shape.lda + 1,
                ..request.shape
            },
            F32TriadShape {
                ldb: request.shape.ldb + 1,
                ..request.shape
            },
            F32TriadShape {
                ldc: request.shape.ldc + 1,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape,
                    },
                    operands,
                )
                .unwrap(),
                qualified,
                "stride neighbor {shape:?}"
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (4_621, 384, 1_928)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                qualified
            );
        }
    }

    #[test]
    fn nt_d128_out_transpose_m64n64_selector_is_an_exact_splitk_cell() {
        let facts = tn_admission_facts();
        let operands = nn_qualified_operands();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (1_024, 256, 128)),
        };
        let qualified = ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified;
        assert_eq!(
            scalar_dispatch_plan(request, 170).unwrap(),
            ScalarDispatchPlan::NtSplitKMain {
                n_main: 128,
                n_tail: 0,
            }
        );
        assert_eq!(
            scalar_launch_plan(facts, request, operands).unwrap(),
            qualified
        );
        assert!(qualified.needs_transpose_scratch());
        assert!(!qualified.needs_split_scratch());

        let dimensions = [request.shape.m, request.shape.k, request.shape.n];
        for axis in 0..3 {
            for delta in [-1_isize, 1] {
                let mut neighbor = dimensions;
                neighbor[axis] = neighbor[axis].checked_add_signed(delta).unwrap();
                let neighbor = F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: F32TriadShape::contiguous(
                        ResolvedGemmOp::Nt,
                        (neighbor[0], neighbor[1], neighbor[2]),
                    ),
                };
                assert_ne!(
                    scalar_launch_plan(facts, neighbor, operands).unwrap(),
                    qualified
                );
            }
        }
        for shape in [
            F32TriadShape {
                lda: 129,
                ..request.shape
            },
            F32TriadShape {
                ldb: 129,
                ..request.shape
            },
            F32TriadShape {
                ldc: 257,
                ..request.shape
            },
        ] {
            assert_ne!(
                scalar_launch_plan(
                    facts,
                    F32TriadRequest {
                        op: ResolvedGemmOp::Nt,
                        shape
                    },
                    operands,
                )
                .unwrap(),
                qualified
            );
        }
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            let mutated = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (1_024, 256, 128)),
            };
            assert_ne!(
                scalar_launch_plan(facts, mutated, operands).unwrap(),
                qualified
            );
        }

        let mut fact_mutations = Vec::new();
        let mut wrong = facts;
        wrong.compute_capability = (12, 1);
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.multiprocessor_count = 169;
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.scalar_compiler.nvrtc_version = (13, 1);
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.scalar_compiler.nvrtc_library_domain = [0; 32];
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.scalar_compiler.invocation_digest = [0; 32];
        wrong.scalar_artifact.compile_key = [0; 32];
        fact_mutations.push(wrong);
        let mut wrong = facts;
        wrong.scalar_artifact.artifact_digest = [0; 32];
        fact_mutations.push(wrong);
        for wrong in fact_mutations {
            assert_ne!(
                scalar_launch_plan(wrong, request, operands).unwrap(),
                qualified
            );
        }
        for mutation in [
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
        ] {
            assert_ne!(
                scalar_launch_plan(facts, request, mutation).unwrap(),
                qualified
            );
        }
    }

    #[test]
    fn nt_prism_vector_rejects_environment_and_operand_mutations() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_621, 384, 1_928)),
        };
        let operands = nn_qualified_operands();
        let qualified = ScalarDispatchPlan::NtPrismVectorQualified;
        let mut fact_mutations = Vec::new();

        let mut facts = tn_admission_facts();
        facts.compute_capability = (12, 1);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.compute_capability = (8, 9);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.multiprocessor_count = 169;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 1);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 3);
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_domain = [0; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.invocation_digest = [0; 32];
        facts.scalar_artifact.compile_key = [0; 32];
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.module_kind = ModuleKind::TriadSm80;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_kind = ArtifactKind::Cubin;
        fact_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_digest = [0; 32];
        fact_mutations.push(facts);

        for facts in fact_mutations {
            assert_ne!(
                scalar_launch_plan(facts, request, operands).unwrap(),
                qualified
            );
        }
        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 12,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(tn_admission_facts(), request, mutation).unwrap(),
                qualified
            );
        }
        for pointer in [operands.output, operands.a, operands.b] {
            let shifted = F32TriadOperands {
                output: if pointer == operands.output {
                    pointer + 16
                } else {
                    operands.output
                },
                a: if pointer == operands.a {
                    pointer + 16
                } else {
                    operands.a
                },
                b: if pointer == operands.b {
                    pointer + 16
                } else {
                    operands.b
                },
                ..operands
            };
            assert_eq!(
                scalar_launch_plan(tn_admission_facts(), request, shifted).unwrap(),
                qualified
            );
        }
    }

    #[test]
    fn nt_d768_out_transpose_m64n64_requires_qualified_environment_and_operands() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        let operands = nn_qualified_operands();
        let plan = ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified;
        let mut mutations = Vec::new();
        let mut facts = tn_admission_facts();
        facts.compute_capability = (8, 9);
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.multiprocessor_count = 169;
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_89").unwrap();
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 1);
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_domain = [0; 32];
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_digest = [0; 32];
        mutations.push(facts);
        for facts in mutations {
            assert_ne!(scalar_launch_plan(facts, request, operands).unwrap(), plan);
        }

        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 12,
                ..operands
            },
        ] {
            assert_ne!(
                scalar_launch_plan(tn_admission_facts(), request, mutation).unwrap(),
                plan
            );
        }
        for aligned in [
            F32TriadOperands {
                output: operands.output + 16,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 16,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 16,
                ..operands
            },
        ] {
            assert_eq!(
                scalar_launch_plan(tn_admission_facts(), request, aligned).unwrap(),
                plan
            );
        }
    }

    #[test]
    fn nt_d768_transpose_m64n64_requires_qualified_environment_and_operands() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 768, 3_072)),
        };
        let operands = nn_qualified_operands();
        let mut facts_mutations = Vec::new();
        let mut facts = tn_admission_facts();
        facts.compute_capability = (12, 1);
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.multiprocessor_count = 169;
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_version = (13, 1);
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_domain = [0; 32];
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.invocation_digest = [0; 32];
        facts.scalar_artifact.compile_key = [0; 32];
        facts_mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_digest = [0; 32];
        facts_mutations.push(facts);
        for facts in facts_mutations {
            assert_ne!(
                scalar_launch_plan(facts, request, operands).unwrap(),
                ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            );
        }

        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands { b: 0, ..operands },
        ] {
            assert_ne!(
                scalar_launch_plan(tn_admission_facts(), request, mutation).unwrap(),
                ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            );
        }
    }

    #[test]
    fn nn_m64n64_selector_requires_the_exact_scalar_environment() {
        let request = nn_qualified_request((2_048, 768, 3_072));
        let operands = nn_qualified_operands();

        let mut mutations = Vec::new();
        let mut facts = tn_admission_facts();
        facts.compute_capability = (12, 1);
        mutations.push(facts);
        for count in [169, 171] {
            let mut facts = tn_admission_facts();
            facts.multiprocessor_count = count;
            mutations.push(facts);
        }
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.module_kind = ModuleKind::TriadSm80;
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.artifact_kind = ArtifactKind::Cubin;
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_artifact.compile_key = [9; 32];
        mutations.push(facts);
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.target = CudaTarget::new("compute_121").unwrap();
        mutations.push(facts);
        for version in [(12, 8), (13, 1), (13, 3)] {
            let mut facts = tn_admission_facts();
            facts.scalar_compiler.nvrtc_version = version;
            mutations.push(facts);
        }
        let mut facts = tn_admission_facts();
        facts.scalar_compiler.nvrtc_library_known = false;
        mutations.push(facts);

        for facts in mutations {
            let plan = scalar_launch_plan(facts, request, operands).unwrap();
            assert_ne!(
                format!("{plan:?}"),
                "NnM64N64Qualified",
                "environment mutation {facts:?}"
            );
        }
    }

    #[test]
    fn nn_m64n64_selector_requires_the_measured_operand_domain() {
        let facts = tn_admission_facts();
        let request = nn_qualified_request((2_048, 768, 3_072));
        let operands = nn_qualified_operands();

        for admitted in [
            F32TriadOperands {
                output: operands.output + 16,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 16,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 16,
                ..operands
            },
        ] {
            let plan = scalar_launch_plan(facts, request, admitted).unwrap();
            assert_eq!(format!("{plan:?}"), "NnM64N64Qualified");
        }

        for mutation in [
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits(1.0_f32.to_bits() + 1),
                ..operands
            },
            F32TriadOperands {
                beta: f32::from_bits(1),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
            F32TriadOperands {
                output: 0,
                ..operands
            },
            F32TriadOperands { a: 0, ..operands },
            F32TriadOperands { b: 0, ..operands },
            F32TriadOperands {
                output: operands.output + 4,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 4,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 4,
                ..operands
            },
            F32TriadOperands {
                output: operands.output + 8,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 8,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 8,
                ..operands
            },
            F32TriadOperands {
                output: operands.output + 12,
                ..operands
            },
            F32TriadOperands {
                a: operands.a + 12,
                ..operands
            },
            F32TriadOperands {
                b: operands.b + 12,
                ..operands
            },
        ] {
            let plan = scalar_launch_plan(facts, request, mutation).unwrap();
            assert_ne!(
                format!("{plan:?}"),
                "NnM64N64Qualified",
                "operand mutation {mutation:?}"
            );
        }

        for op in [ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let request = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (2_048, 768, 3_072)),
            };
            let plan = scalar_launch_plan(facts, request, operands).unwrap();
            assert_ne!(format!("{plan:?}"), "NnM64N64Qualified", "op {op:?}");
        }
    }

    #[test]
    fn nn_slim_split_admission_uses_three_live_waves() {
        let dims = (2560, 384, 512);
        assert_eq!(
            plan(ResolvedGemmOp::Nn, dims, 48).unwrap(),
            ScalarDispatchPlan::NnFinal { slim: true }
        );
        assert_eq!(
            plan(ResolvedGemmOp::Nn, dims, RTX_6000_ADA_SMS).unwrap(),
            ScalarDispatchPlan::NnSplitKSlim { chunks: 6 }
        );
    }

    #[test]
    fn live_sm_policy_preserves_the_frozen_142_sm_scalar_table() {
        let cases = [
            (
                ResolvedGemmOp::Nn,
                (1, 32, 32),
                ScalarDispatchPlan::NnUltraThin,
            ),
            (
                ResolvedGemmOp::Nn,
                (32, 64, 64),
                ScalarDispatchPlan::NnNarrowSmall,
            ),
            (
                ResolvedGemmOp::Nn,
                (128, 64, 64),
                ScalarDispatchPlan::NnNarrow,
            ),
            (ResolvedGemmOp::Nn, (128, 32, 1), ScalarDispatchPlan::NnGemv),
            (
                ResolvedGemmOp::Nn,
                (32, 33, 128),
                ScalarDispatchPlan::NnSplitKThinTail {
                    k_main: 32,
                    k_tail: 1,
                },
            ),
            (
                ResolvedGemmOp::Nn,
                (32, 32, 128),
                ScalarDispatchPlan::NnSplitKThin,
            ),
            (
                ResolvedGemmOp::Nn,
                (2560, 384, 512),
                ScalarDispatchPlan::NnSplitKSlim { chunks: 6 },
            ),
            (
                ResolvedGemmOp::Nn,
                (2048, 128, 512),
                ScalarDispatchPlan::NnFinal { slim: true },
            ),
            (
                ResolvedGemmOp::Nn,
                (2048, 128, 1024),
                ScalarDispatchPlan::NnFinal { slim: false },
            ),
            (ResolvedGemmOp::Tn, (32, 4, 1), ScalarDispatchPlan::TnGemv),
            (ResolvedGemmOp::Tn, (1, 1, 2), ScalarDispatchPlan::TnNarrow),
            (
                ResolvedGemmOp::Tn,
                (256, 128, 128),
                ScalarDispatchPlan::TnSplitM {
                    m_chunk: 16,
                    chunks: 16,
                },
            ),
            (
                ResolvedGemmOp::Tn,
                (128, 128, 128),
                ScalarDispatchPlan::TnFinal { slim: true },
            ),
            (
                ResolvedGemmOp::Tn,
                (128, 1024, 1024),
                ScalarDispatchPlan::TnFinal { slim: false },
            ),
            (
                ResolvedGemmOp::Nt,
                (32, 95, 127),
                ScalarDispatchPlan::NtNarrow,
            ),
            (
                ResolvedGemmOp::Nt,
                (31, 95, 128),
                ScalarDispatchPlan::NtSmallBatchWide,
            ),
            (ResolvedGemmOp::Nt, (1, 7, 1), ScalarDispatchPlan::NtGemv),
            (
                ResolvedGemmOp::Nt,
                (32, 95, 128),
                ScalarDispatchPlan::NtSplitKTail {
                    k_main: 64,
                    k_tail: 31,
                },
            ),
            (
                ResolvedGemmOp::Nt,
                (32, 96, 128),
                ScalarDispatchPlan::NtSplitKMain {
                    n_main: 128,
                    n_tail: 0,
                },
            ),
            (
                ResolvedGemmOp::Nt,
                (2048, 512, 256),
                ScalarDispatchPlan::NtSplitKSlim { chunks: 4 },
            ),
            (
                ResolvedGemmOp::Nt,
                (32, 95, 129),
                ScalarDispatchPlan::NtMidBatchWide,
            ),
            (
                ResolvedGemmOp::Nt,
                (2048, 512, 129),
                ScalarDispatchPlan::NtFinal { slim: true },
            ),
            (
                ResolvedGemmOp::Nt,
                (2048, 1024, 129),
                ScalarDispatchPlan::NtFinal { slim: false },
            ),
        ];

        for (op, dims, expected) in cases {
            assert_eq!(
                plan(op, dims, RTX_6000_ADA_SMS).unwrap(),
                expected,
                "{op:?} {dims:?}"
            );
        }
    }
}

/// Which tensor-core tile variant a TC entry point launched. Returned on
/// success so callers and tests can detect a route that silently failed to
/// launch its selected kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcTile {
    /// 128x128 CTA tile, 256 threads / 8 warps (`gemm_bi_*_tc_*`).
    Tile128,
    /// 64x64 CTA tile, 128 threads / 4 warps (`gemm_bi_*_tc64_*`).
    Tile64,
    /// 16x32 CTA tile, 128 threads / 4 warps, 4-stage cp.async
    /// (`gemm_bi_nn_tc16_*`) - the decode rung of the ladder. NN
    /// forward only; picked by `tc_pick_tile_forward` for the small-M
    /// and narrow-N bands.
    Thin16,
    /// 128x64 CTA tile, 256 threads / 8 warps, BK32 with three stages.
    /// Forced TN dW experiment only; automatic policy never selects it.
    Rect128x64,
}

#[derive(Clone, Copy)]
struct TcGeometry {
    rows: usize,
    columns: usize,
    reduction: usize,
}

fn tc_grid_ctas(rows: usize, columns: usize, tile_rows: u64, tile_columns: u64) -> Option<u64> {
    let rows = u64::try_from(rows).ok()?.div_ceil(tile_rows);
    let columns = u64::try_from(columns).ok()?.div_ceil(tile_columns);
    rows.checked_mul(columns)
}

fn tc_pick_square_tile(
    op: Option<super::super::kernel_identity::PolicyOp>,
    geometry: TcGeometry,
    multiprocessor_count: u32,
) -> Option<TcTile> {
    let policy = super::super::kernel_identity::Sm80TcPolicyV3::current();
    if multiprocessor_count == 0
        || geometry.rows < policy.square_tile_min
        || geometry.columns < policy.square_tile_min
    {
        return None;
    }
    if geometry.rows < policy.large_tile_min || geometry.columns < policy.large_tile_min {
        return Some(TcTile::Tile64);
    }

    let rows128 = u64::try_from(geometry.rows).ok()?.div_ceil(128);
    let columns128 = u64::try_from(geometry.columns).ok()?.div_ceil(128);
    let tiles128 = rows128.checked_mul(columns128)?;
    let sms = u64::from(multiprocessor_count);

    if op == Some(super::super::kernel_identity::PolicyOp::Dw) {
        let short = rows128.min(columns128);
        let long = rows128.max(columns128);
        let rectangular = long >= short.checked_mul(policy.tn_rectangular_min_aspect)?;
        let at_most_one_wave = tiles128.checked_mul(policy.tn_rectangular_wave_denominator)?
            <= sms.checked_mul(policy.tn_rectangular_wave_numerator)?;
        if rectangular && at_most_one_wave {
            return Some(TcTile::Tile64);
        }
    }

    let short_reduction_tile128 = op.is_none()
        && geometry.reduction <= policy.forward_short_reduction_max
        && tiles128.checked_mul(policy.forward_short_reduction_wave_denominator)?
            >= sms.checked_mul(policy.forward_short_reduction_wave_numerator)?;
    let base_tile128 = tiles128.checked_mul(policy.tile128_base_wave_denominator)?
        >= sms.checked_mul(policy.tile128_base_wave_numerator)?;
    if short_reduction_tile128 || base_tile128 {
        Some(TcTile::Tile128)
    } else {
        Some(TcTile::Tile64)
    }
}

/// The forward thin tile covers the narrow rows or columns below Tile64.
/// Its per-element MMA order matches the square tiles, so crossing this
/// scheduling boundary preserves the forward numeric contract.
pub(super) fn tc_pick_tile_forward(
    dims: (usize, usize, usize),
    multiprocessor_count: u32,
) -> Option<TcTile> {
    let (rows, reduction, columns) = dims;
    let policy = super::super::kernel_identity::Sm80TcPolicyV3::current();
    if multiprocessor_count == 0
        || rows == 0
        || reduction == 0
        || columns < policy.forward_min_columns
    {
        return None;
    }
    if rows <= policy.forward_thin_max_rows
        || (policy.forward_thin_below_square_columns && columns < policy.square_tile_min)
    {
        return Some(TcTile::Thin16);
    }

    let grid64 = tc_grid_ctas(rows, columns, 64, 64)?;
    let sms = u64::from(multiprocessor_count);
    let underfilled = grid64.checked_mul(policy.forward_underfill_wave_denominator)?
        < sms.checked_mul(policy.forward_underfill_wave_numerator)?;
    let measured_column_band = columns < policy.large_tile_min
        || (rows >= policy.large_tile_min && columns >= policy.forward_underfill_min_columns);
    if reduction <= policy.forward_underfill_max_reduction && underfilled && measured_column_band {
        return Some(TcTile::Thin16);
    }

    tc_pick_square_tile(
        None,
        TcGeometry {
            rows,
            columns,
            reduction,
        },
        multiprocessor_count,
    )
}

pub(super) fn tc_pick_tile_backward(
    op: super::super::kernel_identity::PolicyOp,
    dims: (usize, usize, usize),
    multiprocessor_count: u32,
) -> Option<TcTile> {
    let (batch, n_in, n_out) = dims;
    let geometry = match op {
        super::super::kernel_identity::PolicyOp::Dw => TcGeometry {
            rows: n_in,
            columns: n_out,
            reduction: batch,
        },
        super::super::kernel_identity::PolicyOp::Dx => TcGeometry {
            rows: batch,
            columns: n_in,
            reduction: n_out,
        },
    };
    let policy = super::super::kernel_identity::Sm80TcPolicyV3::current();
    if multiprocessor_count == 0
        || (policy.reject_zero_axes
            && (geometry.rows == 0 || geometry.columns == 0 || geometry.reduction == 0))
    {
        return None;
    }
    if geometry.rows >= policy.square_tile_min && geometry.columns >= policy.square_tile_min {
        return tc_pick_square_tile(Some(op), geometry, multiprocessor_count);
    }

    let one_long_axis =
        geometry.rows >= policy.square_tile_min || geometry.columns >= policy.square_tile_min;
    let grid64 = tc_grid_ctas(geometry.rows, geometry.columns, 64, 64)?;
    (one_long_axis
        && geometry.reduction >= policy.backward_tail_min_reduction
        && grid64 >= policy.backward_tail_min_tile64_ctas)
        .then_some(TcTile::Tile64)
}

pub(super) fn tc_pick_tile_backward_for_device(
    op: super::super::kernel_identity::PolicyOp,
    dims: (usize, usize, usize),
    multiprocessor_count: u32,
    compute_capability: (i32, i32),
) -> Option<TcTile> {
    let portable = tc_pick_tile_backward(op, dims, multiprocessor_count);
    if compute_capability != (8, 9) || multiprocessor_count == 0 {
        return portable;
    }

    let (batch, n_in, n_out) = dims;
    let geometry = match op {
        super::super::kernel_identity::PolicyOp::Dw => TcGeometry {
            rows: n_in,
            columns: n_out,
            reduction: batch,
        },
        super::super::kernel_identity::PolicyOp::Dx => TcGeometry {
            rows: batch,
            columns: n_in,
            reduction: n_out,
        },
    };
    if geometry.rows < 128 || geometry.columns < 128 || geometry.reduction == 0 {
        return portable;
    }

    let tiles128 = tc_grid_ctas(geometry.rows, geometry.columns, 128, 128)?;
    let sms = u64::from(multiprocessor_count);
    match op {
        super::super::kernel_identity::PolicyOp::Dw => {
            let four_waves = sms.checked_mul(4)?;
            if geometry.reduction <= 128 && tiles128 >= four_waves {
                Some(TcTile::Tile128)
            } else {
                Some(TcTile::Tile64)
            }
        }
        super::super::kernel_identity::PolicyOp::Dx => {
            let enough_tile128_work = tiles128.checked_mul(2)? >= sms;
            let short = geometry.rows.min(geometry.columns);
            let long = geometry.rows.max(geometry.columns);
            let elongated = long.checked_mul(2)? >= short.checked_mul(3)?;
            if enough_tile128_work && (geometry.reduction >= 1024 || elongated) {
                Some(TcTile::Tile128)
            } else {
                Some(TcTile::Tile64)
            }
        }
    }
}

#[cfg(test)]
mod tc_policy_tests {
    use super::{
        TcTile, tc_half_policy_prefers_scalar_forward, tc_pick_tile_backward,
        tc_pick_tile_backward_for_device, tc_pick_tile_forward,
    };
    use crate::mamba_ssm::gpu::kernel_identity::{
        FramedSha256, PolicyOp, digest_hex, gemm_dispatch_policy_digest,
    };

    const RTX_6000_ADA_SMS: u32 = 142;

    #[test]
    fn sm89_half_policy_prefers_proven_split_k_scalar_boundaries_only() {
        let prefers_scalar =
            |dims| tc_half_policy_prefers_scalar_forward((8, 9), dims, RTX_6000_ADA_SMS).unwrap();

        for dims in [
            (127, 511, 128),
            (128, 511, 128),
            (129, 511, 128),
            (128, 513, 128),
            (128, 1023, 128),
            (128, 1024, 128),
            (128, 16_384, 128),
            (128, 16_385, 128),
        ] {
            assert!(prefers_scalar(dims), "approved SM89 scalar cell {dims:?}");
            assert_eq!(
                tc_pick_tile_forward(dims, RTX_6000_ADA_SMS),
                Some(TcTile::Tile64),
                "forced TC selection stays available for {dims:?}"
            );
        }

        for dims in [
            (128, 510, 128),
            (128, 512, 128),
            (128, 768, 128),
            (128, 1023, 127),
            (128, 1023, 129),
            (128, 16_416, 128),
        ] {
            assert!(
                !prefers_scalar(dims),
                "unapproved SM89 scalar cell {dims:?}"
            );
        }
    }

    #[test]
    fn deep_split_k_scalar_preference_is_sm89_specific() {
        for compute_capability in [(8, 0), (8, 6), (8, 7), (9, 0), (10, 0), (12, 0)] {
            assert!(
                !tc_half_policy_prefers_scalar_forward(
                    compute_capability,
                    (128, 8192, 128),
                    RTX_6000_ADA_SMS,
                )
                .unwrap(),
                "unmeasured architecture {compute_capability:?} must keep TC"
            );
        }
    }

    fn selector_tile_name(tile: Option<TcTile>) -> &'static str {
        match tile {
            None => "none",
            Some(TcTile::Thin16) => "thin16",
            Some(TcTile::Tile64) => "tile64",
            Some(TcTile::Tile128) => "tile128",
            Some(TcTile::Rect128x64) => "rect128x64",
        }
    }

    fn selector_outcome_label(outcome: [Option<TcTile>; 3]) -> String {
        format!(
            "{},{},{}",
            selector_tile_name(outcome[0]),
            selector_tile_name(outcome[1]),
            selector_tile_name(outcome[2]),
        )
    }

    fn selector_outcome_digest(
        multiprocessor_count: u32,
        shapes: &[(usize, usize, usize); 15],
        outcomes: &[[Option<TcTile>; 3]; 15],
    ) -> [u8; 32] {
        let mut digest = FramedSha256::new(b"gemm-bi-edge-selector-outcomes.v1")
            .required(b"multiprocessor-count", &multiprocessor_count.to_le_bytes())
            .required(b"outcome-count", &45_u64.to_le_bytes());
        let mut index = 0_u64;
        for (shape, outcome) in shapes.iter().zip(outcomes) {
            let shape = format!("m{}_k{}_n{}", shape.0, shape.1, shape.2);
            for (op, tile) in ["nn", "tn", "nt"].into_iter().zip(outcome) {
                digest = digest
                    .required(b"outcome-index", &index.to_le_bytes())
                    .required(b"shape", shape.as_bytes())
                    .required(b"op", op.as_bytes())
                    .required(b"tile", selector_tile_name(*tile).as_bytes());
                index += 1;
            }
        }
        digest.finish()
    }

    #[test]
    fn selector_policy_identity_is_pinned_for_sm80_plus_device_sizes() {
        for (multiprocessor_count, expected) in [
            (
                48,
                "3d78b978fd63011482be287553dff74fc8ad5ed44c3cbb6211eb847876920492",
            ),
            (
                80,
                "f9b667ff684bb9c2059b699328e28a3c72f880c6368484faa2bc259d3765c5a5",
            ),
            (
                108,
                "8ce633125ebffc7baa5c3648a854af37f1f323fdbb9e1ec4c5b84bc6655ba4f4",
            ),
            (
                120,
                "ba5987deabc8a8aa0aecbf566edbee4b14911abafeba381ad3f05a0a3b9c1e78",
            ),
            (
                142,
                "9a418830be2f351bc6655d07b58d29605cab6df64ceedab30dc854bdda8758be",
            ),
        ] {
            assert_eq!(
                digest_hex(&gemm_dispatch_policy_digest(multiprocessor_count)),
                expected,
                "{multiprocessor_count} SM policy identity"
            );
        }
    }

    #[test]
    fn edge_selector_outcomes_are_pinned_for_sm80_plus_device_sizes() {
        let shapes = [
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
        for (multiprocessor_count, expected, expected_digest) in [
            (
                48,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "tile64,none,tile64",
                    "tile64,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "tile64,tile128,tile64",
                    "tile64,tile64,tile64",
                ],
                "63b5e7e9a2e9077eb0be28dfc9182cceb0346ae5a599e50432b5f73d4d803089",
            ),
            (
                80,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "tile64,none,tile64",
                    "tile64,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "tile64,tile64,tile64",
                    "tile64,tile64,tile64",
                ],
                "f7171ad4ff1d4aca745d7fa851a906e32e22eba1532c22f28205b9459c16c5ef",
            ),
            (
                108,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "thin16,none,tile64",
                    "thin16,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "tile64,tile64,tile64",
                    "tile64,tile64,tile64",
                ],
                "ce7836babc059753e82b0a771cd8cd2db2033ae4e4842783f4d990e64ed90dcb",
            ),
            (
                120,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "thin16,none,tile64",
                    "thin16,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "tile64,tile64,tile64",
                    "tile64,tile64,tile64",
                ],
                "9ffbbbac14c9bbdbfcc10270ac51d054d24402d02d52bfaf39e9fae2ebfa0090",
            ),
            (
                142,
                [
                    "thin16,none,none",
                    "thin16,none,none",
                    "none,none,none",
                    "none,none,none",
                    "thin16,none,none",
                    "tile64,none,none",
                    "tile64,none,none",
                    "tile64,tile128,tile64",
                    "tile128,tile128,tile64",
                    "thin16,none,tile64",
                    "thin16,tile64,tile64",
                    "thin16,tile64,none",
                    "thin16,tile64,tile64",
                    "thin16,tile64,tile64",
                    "tile64,tile64,tile64",
                ],
                "9c5fe15f7ed7bc426d35f2fce3017ed489aef0958ef7b7a049b66419e9e71bc1",
            ),
        ] {
            let outcomes = shapes.map(|dims| {
                [
                    tc_pick_tile_forward(dims, multiprocessor_count),
                    tc_pick_tile_backward(PolicyOp::Dw, dims, multiprocessor_count),
                    tc_pick_tile_backward(PolicyOp::Dx, dims, multiprocessor_count),
                ]
            });
            assert_eq!(outcomes.map(selector_outcome_label), expected);
            assert_eq!(
                digest_hex(&selector_outcome_digest(
                    multiprocessor_count,
                    &shapes,
                    &outcomes,
                )),
                expected_digest,
            );
        }
    }

    #[test]
    fn edge_selector_outcome_digest_rejects_order_and_route_mutations() {
        let mut shapes = [
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
        let mut outcomes = shapes.map(|dims| {
            [
                tc_pick_tile_forward(dims, 142),
                tc_pick_tile_backward(PolicyOp::Dw, dims, 142),
                tc_pick_tile_backward(PolicyOp::Dx, dims, 142),
            ]
        });
        let frozen = selector_outcome_digest(142, &shapes, &outcomes);

        outcomes[0][0] = Some(TcTile::Tile64);
        assert_ne!(selector_outcome_digest(142, &shapes, &outcomes), frozen);
        outcomes[0][0] = Some(TcTile::Thin16);
        shapes.swap(0, 1);
        assert_ne!(selector_outcome_digest(142, &shapes, &outcomes), frozen);
    }

    fn nn(dims: (usize, usize, usize)) -> Option<TcTile> {
        tc_pick_tile_forward(dims, RTX_6000_ADA_SMS)
    }

    fn dw(dims: (usize, usize, usize)) -> Option<TcTile> {
        tc_pick_tile_backward(PolicyOp::Dw, dims, RTX_6000_ADA_SMS)
    }

    fn dx(dims: (usize, usize, usize)) -> Option<TcTile> {
        tc_pick_tile_backward(PolicyOp::Dx, dims, RTX_6000_ADA_SMS)
    }

    #[test]
    fn nn_policy_uses_underfill_and_short_reduction_wave_rules() {
        assert_eq!(nn((129, 131, 100)), Some(TcTile::Thin16));
        assert_eq!(nn((256, 512, 384)), Some(TcTile::Thin16));
        assert_eq!(nn((512, 16, 2048)), Some(TcTile::Tile128));
    }

    #[test]
    fn nn_short_reduction_boundary_is_inclusive_at_k16() {
        assert_eq!(nn((128, 16, 8064)), Some(TcTile::Tile128));
        assert_eq!(nn((128, 17, 8064)), Some(TcTile::Tile64));
    }

    #[test]
    fn nn_underfill_reduction_boundary_is_inclusive_at_k512() {
        assert_eq!(nn((129, 512, 100)), Some(TcTile::Thin16));
        assert_eq!(nn((129, 513, 100)), Some(TcTile::Tile64));
    }

    #[test]
    fn nn_underfill_grid_boundary_is_strict_between_26_and_27_ctas() {
        assert_eq!(nn((128, 512, 832)), Some(TcTile::Thin16));
        assert_eq!(nn((129, 512, 576)), Some(TcTile::Tile64));
    }

    #[test]
    fn square_half_wave_boundary_is_inclusive_between_70_and_71_ctas() {
        assert_eq!(nn((128, 17, 8960)), Some(TcTile::Tile64));
        assert_eq!(nn((128, 17, 9088)), Some(TcTile::Tile128));
    }

    #[test]
    fn tn_policy_uses_tile64_for_one_wave_rectangles() {
        assert_eq!(dw((2048, 768, 3072)), Some(TcTile::Tile64));
        assert_eq!(dw((2048, 3072, 768)), Some(TcTile::Tile64));
        assert_eq!(dw((512, 3072, 768)), Some(TcTile::Tile64));

        assert_eq!(dw((2048, 1536, 768)), Some(TcTile::Tile128));
        assert_eq!(dw((4096, 3072, 1536)), Some(TcTile::Tile128));
    }

    #[test]
    fn sm89_backward_square_policy_matches_forced_matrix_envelopes() {
        let sm89 = (8, 9);
        let sm80 = (8, 0);

        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dw,
                (2048, 1536, 768),
                RTX_6000_ADA_SMS,
                sm89,
            ),
            Some(TcTile::Tile64),
        );
        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dw,
                (4096, 3072, 1536),
                RTX_6000_ADA_SMS,
                sm89,
            ),
            Some(TcTile::Tile64),
        );
        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dx,
                (2048, 1536, 768),
                RTX_6000_ADA_SMS,
                sm89,
            ),
            Some(TcTile::Tile64),
        );
        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dx,
                (2048, 3072, 768),
                RTX_6000_ADA_SMS,
                sm89,
            ),
            Some(TcTile::Tile128),
        );
        assert_eq!(
            tc_pick_tile_backward_for_device(
                PolicyOp::Dw,
                (4096, 3072, 1536),
                RTX_6000_ADA_SMS,
                sm80,
            ),
            tc_pick_tile_backward(PolicyOp::Dw, (4096, 3072, 1536), RTX_6000_ADA_SMS,),
        );
    }

    #[test]
    fn tn_rectangular_aspect_uses_the_ceil_divided_tile_grid() {
        assert_eq!(dw((1024, 513, 2432)), Some(TcTile::Tile128));
        assert_eq!(dw((1024, 640, 2433)), Some(TcTile::Tile64));
    }

    #[test]
    fn backward_tail_envelope_replaces_exact_shape_admission() {
        assert_eq!(dw((512, 16, 2048)), Some(TcTile::Tile64));
        assert_eq!(dx((512, 16, 2048)), Some(TcTile::Tile64));
        assert_eq!(dx((16, 512, 2048)), Some(TcTile::Tile64));

        assert_eq!(dw((255, 16, 2048)), None);
        assert_eq!(dx((16, 192, 256)), None);
        assert_eq!(dx((16, 256, 256)), Some(TcTile::Tile64));
        assert_eq!(dw((256, 63, 63)), None);
        assert_eq!(dw((256, 0, 256)), None);
    }

    #[test]
    fn backward_tail_reduction_boundary_is_inclusive_at_256() {
        assert_eq!(dw((255, 16, 256)), None);
        assert_eq!(dw((256, 16, 256)), Some(TcTile::Tile64));
    }

    #[test]
    fn one_sub_128_axis_never_uses_the_tile128_square_rule() {
        assert_eq!(nn((127, 513, 16384)), Some(TcTile::Tile64));
        assert_eq!(dw((512, 127, 16384)), Some(TcTile::Tile64));
    }

    #[test]
    fn policy_v3_keeps_long_reduction_and_full_tile_controls() {
        assert_eq!(nn((128, 8192, 128)), Some(TcTile::Tile64));
        assert_eq!(nn((1024, 256, 128)), Some(TcTile::Tile64));
        assert_eq!(nn((2048, 768, 3072)), Some(TcTile::Tile128));

        assert_eq!(dx((1024, 128, 512)), Some(TcTile::Tile64));
        assert_eq!(dx((2048, 768, 3072)), Some(TcTile::Tile128));
        assert_eq!(dx((4096, 3072, 1536)), Some(TcTile::Tile128));
    }
}

#[cfg(test)]
mod sm120_tests {
    use super::{
        SM120_AUTO_CELLS_CC120, SM120_AUTO_CELLS_CC121, Sm120AutoRequest, resolve_sm120_auto,
        sm120_target_candidates,
    };
    use crate::mamba_ssm::gpu::dtype::WeightDtype;
    use crate::mamba_ssm::gpu::gemm_bi_triad::contract::{
        Sm120Bk, Sm120ForcedRoute, Sm120LaunchOperands, Sm120Op, Sm120PhysicalRoute, Sm120Shape,
        Sm120Stages, Sm120Tile,
    };
    use crate::mamba_ssm::gpu::kernel_identity::{CudaTarget, DeviceCaps};

    fn targets(cc: (i32, i32), nvrtc: (i32, i32)) -> Vec<(&'static str, &'static str)> {
        sm120_target_candidates(cc, nvrtc)
            .iter()
            .map(|candidate| (candidate.nvrtc_arch, candidate.ptx_target))
            .collect()
    }

    fn caps(
        compute_capability: (u32, u32),
        nvrtc_version: (i32, i32),
        accepted_target: Option<&str>,
    ) -> DeviceCaps {
        DeviceCaps {
            compute_capability,
            nvrtc_version,
            accepted_target: accepted_target.map(|target| CudaTarget::new(target).unwrap()),
            optin_shared_bytes: 101_376,
            tensor_map_access: true,
        }
    }

    fn route(
        op: Sm120Op,
        dtype: WeightDtype,
        dims: (usize, usize, usize),
        physical: Sm120PhysicalRoute,
    ) -> Sm120ForcedRoute {
        Sm120ForcedRoute {
            op,
            dtype,
            physical,
            shape: Sm120Shape::contiguous(op, dims),
        }
    }

    fn qualified_cc120_routes() -> [Sm120ForcedRoute; 60] {
        let m64n64_bk64_s2 = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        };
        let m128n64_bk32_s2 = Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        };
        let m64n128_bk32_s3 = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        };
        let m128n128_bk32_s3 = Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S3,
        };
        let m128n64_bk64_s2 = Sm120PhysicalRoute {
            tile: Sm120Tile::M128N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S2,
        };
        let m128n128_bk64_s3 = Sm120PhysicalRoute {
            tile: Sm120Tile::M128N128,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        };
        let m64n64_bk64_s3 = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N64,
            bk: Sm120Bk::Bk64,
            stages: Sm120Stages::S3,
        };
        let m64n128_bk32_s2 = Sm120PhysicalRoute {
            tile: Sm120Tile::M64N128,
            bk: Sm120Bk::Bk32,
            stages: Sm120Stages::S2,
        };
        let projection = (2048, 1536, 768);
        let large = (2048, 3072, 768);
        let deep = (4096, 3072, 1536);
        let input_projection = (2048, 768, 3072);
        let prism = (4621, 384, 1928);
        let out_projection = (4621, 768, 384);
        let input_projection_wide = (4621, 1024, 384);
        let batch_input_projection = (10400, 384, 384);
        let batch_in_projection = (10400, 384, 1536);
        let batch_out_projection = (10400, 768, 384);
        [
            route(Sm120Op::Nn, WeightDtype::Bf16, projection, m64n64_bk64_s2),
            route(Sm120Op::Nn, WeightDtype::Bf16, large, m64n64_bk64_s2),
            route(Sm120Op::Nn, WeightDtype::Bf16, deep, m128n64_bk32_s2),
            route(Sm120Op::Nn, WeightDtype::F16, projection, m64n64_bk64_s2),
            route(Sm120Op::Nn, WeightDtype::F16, large, m64n64_bk64_s2),
            route(Sm120Op::Nn, WeightDtype::F16, deep, m128n64_bk32_s2),
            route(Sm120Op::Tn, WeightDtype::Bf16, projection, m64n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::Bf16, large, m64n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::Bf16, deep, m128n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::F16, projection, m64n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::F16, large, m64n128_bk32_s3),
            route(Sm120Op::Tn, WeightDtype::F16, deep, m128n128_bk32_s3),
            route(Sm120Op::Nt, WeightDtype::Bf16, projection, m64n64_bk64_s2),
            route(Sm120Op::Nt, WeightDtype::Bf16, large, m128n64_bk32_s2),
            route(Sm120Op::Nt, WeightDtype::Bf16, deep, m128n128_bk32_s3),
            route(Sm120Op::Nt, WeightDtype::F16, projection, m64n64_bk64_s2),
            route(Sm120Op::Nt, WeightDtype::F16, large, m128n64_bk32_s2),
            route(Sm120Op::Nt, WeightDtype::F16, deep, m128n128_bk32_s3),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                input_projection,
                m128n64_bk32_s2,
            ),
            route(Sm120Op::Nn, WeightDtype::Bf16, prism, m128n64_bk64_s2),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                input_projection,
                m128n64_bk32_s2,
            ),
            route(Sm120Op::Nn, WeightDtype::F16, prism, m128n64_bk64_s2),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                input_projection,
                m64n128_bk32_s3,
            ),
            route(Sm120Op::Tn, WeightDtype::Bf16, prism, m64n128_bk32_s3),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                input_projection,
                m64n128_bk32_s3,
            ),
            route(Sm120Op::Tn, WeightDtype::F16, prism, m64n128_bk32_s3),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                input_projection,
                m64n64_bk64_s2,
            ),
            route(Sm120Op::Nt, WeightDtype::Bf16, prism, m128n128_bk64_s3),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                input_projection,
                m64n64_bk64_s2,
            ),
            route(Sm120Op::Nt, WeightDtype::F16, prism, m128n128_bk64_s3),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                out_projection,
                m64n64_bk64_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                input_projection_wide,
                m64n64_bk64_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                batch_input_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                batch_in_projection,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::Bf16,
                batch_out_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                out_projection,
                m64n64_bk64_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                input_projection_wide,
                m64n64_bk64_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                batch_input_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                batch_in_projection,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nn,
                WeightDtype::F16,
                batch_out_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                out_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                input_projection_wide,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                batch_input_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                batch_in_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::Bf16,
                batch_out_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                out_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                input_projection_wide,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                batch_input_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                batch_in_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Tn,
                WeightDtype::F16,
                batch_out_projection,
                m64n64_bk64_s3,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                out_projection,
                m64n128_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                input_projection_wide,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                batch_input_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                batch_in_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::Bf16,
                batch_out_projection,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                out_projection,
                m64n128_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                input_projection_wide,
                m128n128_bk32_s3,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                batch_input_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                batch_in_projection,
                m128n64_bk32_s2,
            ),
            route(
                Sm120Op::Nt,
                WeightDtype::F16,
                batch_out_projection,
                m128n128_bk32_s3,
            ),
        ]
    }

    #[test]
    fn a_shape_off_the_table_takes_the_nearest_measured_tile_inside_the_band() {
        use super::{SM120_AUTO_CELLS_CC120, nearest_sm120_cell};
        // The tall rectangle sits nearest the 4621x384-output serve
        // projections and takes their tiles, one per operation.
        let tall = Sm120Shape::contiguous(Sm120Op::Nn, (4096, 512, 768));
        assert_eq!(
            nearest_sm120_cell(
                SM120_AUTO_CELLS_CC120,
                Sm120Op::Nn,
                WeightDtype::Bf16,
                tall,
                170
            ),
            Some(Sm120PhysicalRoute {
                tile: Sm120Tile::M64N64,
                bk: Sm120Bk::Bk64,
                stages: Sm120Stages::S2,
            })
        );
        let tall_tn = Sm120Shape::contiguous(Sm120Op::Tn, (4096, 512, 768));
        assert_eq!(
            nearest_sm120_cell(
                SM120_AUTO_CELLS_CC120,
                Sm120Op::Tn,
                WeightDtype::F16,
                tall_tn,
                170,
            ),
            Some(Sm120PhysicalRoute {
                tile: Sm120Tile::M64N64,
                bk: Sm120Bk::Bk64,
                stages: Sm120Stages::S3,
            })
        );
        // A measured shape is its own nearest cell.
        let measured = Sm120Shape::contiguous(Sm120Op::Nt, (2048, 3072, 768));
        assert_eq!(
            nearest_sm120_cell(
                SM120_AUTO_CELLS_CC120,
                Sm120Op::Nt,
                WeightDtype::Bf16,
                measured,
                170,
            ),
            Some(Sm120PhysicalRoute {
                tile: Sm120Tile::M128N64,
                bk: Sm120Bk::Bk32,
                stages: Sm120Stages::S2,
            })
        );
        // A narrow output whose neighbour carries a 128x128 tile cannot fill
        // one wave with it and takes the smallest tile instead.
        let narrow = Sm120Shape::contiguous(Sm120Op::Nt, (1024, 128, 512));
        let picked = nearest_sm120_cell(
            SM120_AUTO_CELLS_CC120,
            Sm120Op::Nt,
            WeightDtype::Bf16,
            narrow,
            170,
        )
        .expect("the narrow projection is inside the band");
        assert_eq!(picked.tile, Sm120Tile::M64N64);
        // A tiny square is more than a factor of eight from every cell on at
        // least one axis and gets no tile: the portable tiles serve it.
        let tiny = Sm120Shape::contiguous(Sm120Op::Nn, (64, 64, 64));
        assert_eq!(
            nearest_sm120_cell(
                SM120_AUTO_CELLS_CC120,
                Sm120Op::Nn,
                WeightDtype::Bf16,
                tiny,
                170
            ),
            None
        );
    }

    fn request_for(route: Sm120ForcedRoute) -> Sm120AutoRequest {
        let beta = if route.op == Sm120Op::Tn { 1.0 } else { 0.0 };
        Sm120AutoRequest {
            op: route.op,
            dtype: route.dtype,
            shape: route.shape,
            a_ptr: 0x1_0000,
            b_ptr: 0x2_0000,
            multiprocessors: 170,
            operands: Sm120LaunchOperands {
                output_ptr: 0x3_0000,
                bias_ptr: 0,
                alpha: 1.0,
                beta,
            },
        }
    }

    #[test]
    fn sm120_auto_table_matches_the_qualified_cc120_inventory() {
        assert_eq!(SM120_AUTO_CELLS_CC120, qualified_cc120_routes());
        assert!(SM120_AUTO_CELLS_CC121.is_empty());
    }

    #[test]
    fn sm120_auto_selects_every_qualified_cc120_cell() {
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        for expected in qualified_cc120_routes() {
            assert_eq!(
                resolve_sm120_auto(
                    caps((12, 0), (12, 8), Some("compute_120")),
                    Some(target),
                    request_for(expected),
                ),
                Some(expected)
            );
        }
    }

    #[test]
    fn sm120_auto_declines_unsupported_operands_for_each_operation() {
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        for op in [Sm120Op::Nn, Sm120Op::Tn, Sm120Op::Nt] {
            let route = qualified_cc120_routes()
                .into_iter()
                .find(|route| route.op == op)
                .expect("qualified route for operation");
            let mut request = request_for(route);
            match op {
                Sm120Op::Nn => {
                    request.operands.bias_ptr = 0x4_0000;
                    request.operands.alpha = 0.5;
                }
                Sm120Op::Tn => request.operands.beta = 0.0,
                Sm120Op::Nt => request.operands.beta = 1.0,
            }
            assert_eq!(
                resolve_sm120_auto(
                    caps((12, 0), (12, 8), Some("compute_120")),
                    Some(target),
                    request,
                ),
                None,
                "unsupported {op:?} operands"
            );
        }
    }

    #[test]
    fn sm120_auto_declines_a_nonqualified_shape() {
        let route = qualified_cc120_routes()[6];
        let mut request = request_for(route);
        request.shape.m += 1;
        let target = sm120_target_candidates((12, 0), (12, 8))[0];

        // One row off a measured cell is inside the neighbour band: the
        // request takes that cell's tile on its own shape.
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_120")),
                Some(target),
                request,
            ),
            Some(Sm120ForcedRoute {
                op: route.op,
                dtype: route.dtype,
                physical: route.physical,
                shape: request.shape,
            })
        );

        // Far outside the band nothing is measured and the request declines.
        let mut far = request_for(route);
        far.shape = Sm120Shape::contiguous(route.op, (64, 64, 64));
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_120")),
                Some(target),
                far,
            ),
            None
        );

        let mut unsupported_dtype = request_for(route);
        unsupported_dtype.dtype = WeightDtype::F32;
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_120")),
                Some(target),
                unsupported_dtype,
            ),
            None
        );
    }

    #[test]
    fn sm120_auto_declines_unaligned_inputs_and_outputs() {
        let route = qualified_cc120_routes()[6];
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        for operand in 0..4 {
            let mut request = request_for(route);
            match operand {
                0 => request.a_ptr += 2,
                1 => request.b_ptr += 2,
                2 => request.operands.output_ptr += 2,
                _ => request.operands.output_ptr = 0,
            }
            assert_eq!(
                resolve_sm120_auto(
                    caps((12, 0), (12, 8), Some("compute_120")),
                    Some(target),
                    request,
                ),
                None,
                "invalid operand {operand}"
            );
        }

        let nn_route = qualified_cc120_routes()[0];
        let mut request = request_for(nn_route);
        request.operands.bias_ptr = 0x4_0002;
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_120")),
                Some(target),
                request,
            ),
            None
        );
    }

    #[test]
    fn sm120_auto_declines_without_tensor_map_capability() {
        let route = qualified_cc120_routes()[6];
        let request = request_for(route);
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        let mut unavailable = caps((12, 0), (12, 8), Some("compute_120"));
        unavailable.tensor_map_access = false;

        assert_eq!(resolve_sm120_auto(unavailable, Some(target), request), None);

        let mut insufficient_shared = caps((12, 0), (12, 8), Some("compute_120"));
        insufficient_shared.optin_shared_bytes = route
            .kernel_spec()
            .expect("qualified route specification")
            .dynamic_shared_bytes
            - 1;
        assert_eq!(
            resolve_sm120_auto(insufficient_shared, Some(target), request),
            None
        );
        assert_eq!(
            resolve_sm120_auto(caps((12, 0), (12, 8), Some("compute_120")), None, request),
            None
        );
        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 7), Some("compute_120")),
                Some(target),
                request,
            ),
            None
        );
    }

    #[test]
    fn sm120_auto_declines_a_mismatched_accepted_target() {
        let route = qualified_cc120_routes()[6];
        let request = request_for(route);
        let target = sm120_target_candidates((12, 0), (12, 8))[0];

        assert_eq!(
            resolve_sm120_auto(
                caps((12, 0), (12, 8), Some("compute_121")),
                Some(target),
                request,
            ),
            None
        );
    }

    #[test]
    fn sm120_auto_declines_without_qualified_minor_table() {
        let request = request_for(qualified_cc120_routes()[6]);
        let target = sm120_target_candidates((12, 1), (12, 9))[0];

        assert_eq!(
            resolve_sm120_auto(
                caps((12, 1), (12, 9), Some("compute_121")),
                Some(target),
                request
            ),
            None
        );
    }

    #[test]
    fn sm120_candidates_follow_toolkit_support_and_generic_compatibility() {
        assert!(targets((12, 0), (12, 7)).is_empty());
        assert!(targets((12, 1), (12, 7)).is_empty());
        assert!(targets((11, 0), (13, 2)).is_empty());
        assert!(targets((12, 2), (13, 2)).is_empty());

        assert_eq!(targets((12, 0), (12, 8)), [("compute_120", "sm_120")]);
        assert_eq!(targets((12, 1), (12, 8)), [("compute_120", "sm_120")]);
        assert_eq!(
            targets((12, 1), (12, 9)),
            [("compute_121", "sm_121"), ("compute_120", "sm_120")]
        );
        assert_eq!(
            targets((12, 1), (13, 2)),
            [("compute_121", "sm_121"), ("compute_120", "sm_120")]
        );
    }
}

#[cfg(test)]
mod tf32_tests {
    use super::{
        SM89_TF32_EVIDENCE_CELLS, SM89_TF32_QUALIFICATION_IDENTITY, SM120_TF32_EVIDENCE_CELLS,
        SM120_TF32_EVIDENCE_COHORTS, Tf32AutoEvidenceCohort, Tf32AutoOperandGate,
        matching_tf32_cohort, measured_tf32_cell, measured_tf32_route_with_operands,
        resolve_f32_triad_auto, resolve_f32_triad_auto_with_operands, resolve_tf32_forced,
    };
    use crate::mamba_ssm::gpu::context::F32TriadPolicy;
    use crate::mamba_ssm::gpu::gemm_bi_triad::contract::{
        F32_TF32_TUNING_REVISION, F32TriadAvailability, F32TriadOperands, F32TriadRequest,
        F32TriadSelection, F32TriadShape, TF32_NT_SPLITK8_S3_SPEC, Tf32PhysicalRoute,
        Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile, Tf32QualifiedModule,
        Tf32Sm90aRoute, Tf32Sm100Route, Tf32Sm120Route, Tf32Sm120Stages, Tf32Sm120Tile,
    };
    use crate::mamba_ssm::gpu::gemm_bi_triad::{
        Sm90aWarpgroupSchedule, Sm100Schedule, Sm100Stages, Sm100Tile,
    };
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, ModuleKind, NUMERIC_ABI_REVISION,
        ResolvedGemmOp, SCHEDULE_REVISION, TUNING_TABLE_REVISION, digest_hex,
    };

    fn qualified_module(
        module_kind: ModuleKind,
        target_name: &str,
        device_target_name: &str,
        compute_capability: (u32, u32),
        tensor_map_access: bool,
        optin_shared_bytes: u32,
    ) -> Tf32QualifiedModule {
        let target = CudaTarget::new(target_name).unwrap();
        let device_target = CudaTarget::new(device_target_name).unwrap();
        let nvrtc_version = (13, 2);
        Tf32QualifiedModule {
            module_kind,
            target,
            artifact: ArtifactIdentity {
                module_kind,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: [4; 32],
                artifact_digest: [2; 32],
            },
            compiler: CompilerIdentity {
                source_digest: [3; 32],
                invocation_digest: [4; 32],
                header_manifest_digest: [5; 32],
                target,
                nvrtc_version,
                nvrtc_library_domain: [6; 32],
                nvrtc_library_known: true,
                output_kind: ArtifactKind::Ptx,
                composer_revision: COMPOSER_REVISION,
                compiler_revision: COMPILER_REVISION,
                numeric_abi_revision: NUMERIC_ABI_REVISION,
                schedule_revision: SCHEDULE_REVISION,
            },
            device: DeviceIdentity {
                compute_capability,
                multiprocessor_count: 142,
                target: device_target,
                driver: DriverIdentity {
                    api_version: 13_020,
                    build_sources: 1,
                    build_digest: [7; 32],
                },
            },
            device_caps: DeviceCaps {
                compute_capability,
                nvrtc_version,
                accepted_target: Some(target),
                optin_shared_bytes,
                tensor_map_access,
            },
        }
    }

    fn request(op: ResolvedGemmOp) -> F32TriadRequest {
        F32TriadRequest {
            op,
            shape: F32TriadShape::contiguous(op, (128, 256, 128)),
        }
    }

    fn normalized_request(
        op: ResolvedGemmOp,
        output_rows: usize,
        output_columns: usize,
        reduction: usize,
    ) -> F32TriadRequest {
        let dims = match op {
            ResolvedGemmOp::Nn => (output_rows, reduction, output_columns),
            ResolvedGemmOp::Tn => (reduction, output_rows, output_columns),
            ResolvedGemmOp::Nt => (output_rows, output_columns, reduction),
        };
        F32TriadRequest {
            op,
            shape: F32TriadShape::contiguous(op, dims),
        }
    }

    fn sm89_availability() -> F32TriadAvailability {
        let mut portable = qualified_module(
            ModuleKind::TriadSm80,
            "sm_89",
            "sm_89",
            (8, 9),
            false,
            101_376,
        );
        portable.artifact.compile_key = SM89_TF32_QUALIFICATION_IDENTITY.compile_key;
        portable.artifact.artifact_digest = SM89_TF32_QUALIFICATION_IDENTITY.artifact_digest;
        portable.compiler.source_digest = SM89_TF32_QUALIFICATION_IDENTITY.source_digest;
        portable.compiler.invocation_digest = SM89_TF32_QUALIFICATION_IDENTITY.invocation_digest;
        portable.compiler.header_manifest_digest =
            SM89_TF32_QUALIFICATION_IDENTITY.header_manifest_digest;
        portable.compiler.nvrtc_library_domain =
            SM89_TF32_QUALIFICATION_IDENTITY.nvrtc_library_domain;
        portable.device.driver.build_sources = 7;
        portable.device.driver.build_digest = SM89_TF32_QUALIFICATION_IDENTITY.driver_build_digest;
        F32TriadAvailability {
            portable: Some(portable),
            specialized: None,
        }
    }

    /// TF32 fail-closed: a request or operand set that drifts off a measured
    /// cell may still run on the exact family, which carries its own
    /// qualification, but it must never reach a TF32 route.
    fn assert_no_tf32_route(selection: F32TriadSelection) {
        assert!(
            !matches!(selection, F32TriadSelection::Tf32(_)),
            "drifted request selected {selection:?}"
        );
    }

    fn sm120_availability_for(
        identity: super::Tf32AutoQualificationIdentity,
    ) -> F32TriadAvailability {
        let specialized = qualified_module_for_auto_identity(identity);
        F32TriadAvailability {
            portable: None,
            specialized: Some(specialized),
        }
    }

    fn qualified_module_for_auto_identity(
        identity: super::Tf32AutoQualificationIdentity,
    ) -> Tf32QualifiedModule {
        let mut module = qualified_module(
            identity.module_kind,
            identity.module_target,
            identity.device_target,
            identity.compute_capability,
            identity.tensor_map_access,
            identity.optin_shared_bytes,
        );
        module.artifact.compile_key = identity.compile_key;
        module.artifact.artifact_digest = identity.artifact_digest;
        module.compiler.source_digest = identity.source_digest;
        module.compiler.invocation_digest = identity.invocation_digest;
        module.compiler.header_manifest_digest = identity.header_manifest_digest;
        module.compiler.nvrtc_version = identity.nvrtc_version;
        module.compiler.nvrtc_library_domain = identity.nvrtc_library_domain;
        module.device.multiprocessor_count = identity.multiprocessor_count;
        module.device.driver.api_version = identity.driver_api_version;
        module.device.driver.build_sources = identity.driver_build_sources;
        module.device.driver.build_digest = identity.driver_build_digest;
        module.device_caps.nvrtc_version = identity.nvrtc_version;
        module
    }

    fn sm120_cohort(nvrtc_version: (i32, i32)) -> Tf32AutoEvidenceCohort {
        SM120_TF32_EVIDENCE_COHORTS
            .iter()
            .copied()
            .find(|cohort| cohort.identity.nvrtc_version == nvrtc_version)
            .unwrap_or_else(|| panic!("missing frozen SM120 TF32 CUDA {nvrtc_version:?} cohort"))
    }

    fn sm120_identity_mutations() -> [fn(&mut Tf32QualifiedModule); 29] {
        [
            |module| module.module_kind = ModuleKind::Fixed,
            |module| module.target = CudaTarget::new("compute_121").unwrap(),
            |module| module.artifact.module_kind = ModuleKind::Fixed,
            |module| module.artifact.artifact_kind = ArtifactKind::Cubin,
            |module| module.artifact.compile_key[0] ^= 1,
            |module| module.artifact.artifact_digest[0] ^= 1,
            |module| module.compiler.source_digest[0] ^= 1,
            |module| module.compiler.invocation_digest[0] ^= 1,
            |module| module.compiler.header_manifest_digest[0] ^= 1,
            |module| module.compiler.target = CudaTarget::new("compute_121").unwrap(),
            |module| module.compiler.nvrtc_version.1 ^= 1,
            |module| module.compiler.nvrtc_library_domain[0] ^= 1,
            |module| module.compiler.nvrtc_library_known = false,
            |module| module.compiler.output_kind = ArtifactKind::Cubin,
            |module| module.compiler.composer_revision ^= 1,
            |module| module.compiler.compiler_revision ^= 1,
            |module| module.compiler.numeric_abi_revision ^= 1,
            |module| module.compiler.schedule_revision ^= 1,
            |module| module.device.compute_capability = (12, 1),
            |module| module.device.multiprocessor_count -= 1,
            |module| module.device.target = CudaTarget::new("sm_121").unwrap(),
            |module| module.device.driver.api_version -= 1,
            |module| module.device.driver.build_sources -= 1,
            |module| module.device.driver.build_digest[0] ^= 1,
            |module| module.device_caps.compute_capability = (12, 1),
            |module| module.device_caps.nvrtc_version.1 ^= 1,
            |module| module.device_caps.accepted_target = None,
            |module| module.device_caps.optin_shared_bytes -= 1,
            |module| module.device_caps.tensor_map_access = false,
        ]
    }

    fn portable_sm120_coupled_mutations() -> [fn(&mut Tf32QualifiedModule); 6] {
        [
            |module| {
                module.artifact.compile_key[0] ^= 1;
                module.compiler.invocation_digest[0] ^= 1;
            },
            |module| {
                let target = CudaTarget::new("compute_121").unwrap();
                module.target = target;
                module.compiler.target = target;
                module.device_caps.accepted_target = Some(target);
            },
            |module| {
                module.artifact.artifact_kind = ArtifactKind::Cubin;
                module.compiler.output_kind = ArtifactKind::Cubin;
            },
            |module| {
                module.device.compute_capability = (12, 1);
                module.device_caps.compute_capability = (12, 1);
            },
            |module| {
                module.compiler.nvrtc_version = (13, 1);
                module.device_caps.nvrtc_version = (13, 1);
            },
            |module| {
                module.module_kind = ModuleKind::Fixed;
                module.artifact.module_kind = ModuleKind::Fixed;
            },
        ]
    }

    #[derive(Clone, Copy, Debug)]
    enum Sm120ManifestRoute {
        Tiled(Tf32Sm120Tile, Tf32Sm120Stages),
        StreamK(Tf32Sm120Tile, Tf32Sm120Stages),
    }

    impl Sm120ManifestRoute {
        fn route(self) -> Tf32PhysicalRoute {
            match self {
                Self::Tiled(tile, stages) => {
                    Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route { tile, stages })
                }
                Self::StreamK(tile, stages) => {
                    Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(Tf32Sm120Route { tile, stages })
                }
            }
        }
    }

    type Sm120RouteManifestEntry = (ResolvedGemmOp, usize, usize, usize, Sm120ManifestRoute);

    fn assert_sm120_route_manifest<const N: usize>(
        cells: &[super::Tf32AutoCell],
        expected: [Sm120RouteManifestEntry; N],
    ) {
        assert_eq!(cells.len(), expected.len());
        for (cell, (op, rows, columns, reduction, route)) in cells.iter().zip(expected) {
            assert_eq!(cell.op, op);
            assert_eq!(
                cell.shape,
                super::Tf32ExactShape {
                    output_rows: rows,
                    output_columns: columns,
                    reduction,
                }
            );
            assert_eq!(cell.route, route.route());
        }
    }

    #[test]
    fn sm120_tf32_cuda_12_8_identity_matches_the_literal_qualification_manifest() {
        let identity = sm120_cohort((12, 8)).identity;
        assert_eq!(identity.module_kind, ModuleKind::TriadSm120);
        assert_eq!(identity.module_target, "compute_120");
        assert_eq!(identity.device_target, "sm_120");
        assert_eq!(identity.compute_capability, (12, 0));
        assert_eq!(identity.multiprocessor_count, 170);
        assert_eq!(identity.nvrtc_version, (12, 8));
        assert_eq!(identity.driver_api_version, 13_020);
        assert_eq!(identity.driver_build_sources, 7);
        assert_eq!(identity.optin_shared_bytes, 101_376);
        assert!(identity.tensor_map_access);
        assert_eq!(
            digest_hex(&identity.compile_key),
            "a6cea94cf0a95464d9070cd419047676b3dee39c318bb96eeb511a360b368021"
        );
        assert_eq!(
            digest_hex(&identity.artifact_digest),
            "5b6a1cbc4f0b9a8f4219b2fd0b982adb798bb918e81215de48843690de3df7ab"
        );
        assert_eq!(
            digest_hex(&identity.source_digest),
            "246d2e35eb059cd696ec74325d4a223312fa605fe84ff58f17f6638d99b8182c"
        );
        assert_eq!(
            digest_hex(&identity.invocation_digest),
            "a6cea94cf0a95464d9070cd419047676b3dee39c318bb96eeb511a360b368021"
        );
        assert_eq!(
            digest_hex(&identity.header_manifest_digest),
            "fe8038d2296a5466909468f2b502f7cb3bf55eec314a4096a43b029f6d8b1d30"
        );
        assert_eq!(
            digest_hex(&identity.nvrtc_library_domain),
            "26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155"
        );
        assert_eq!(
            digest_hex(&identity.driver_build_digest),
            "bbe8397f6ef11a506a502d127d5eb82745a515ab64f9b2901dab4e834bf190d8"
        );

        let module = qualified_module_for_auto_identity(identity);
        assert_eq!(module.artifact.module_kind, ModuleKind::TriadSm120);
        assert_eq!(module.artifact.artifact_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.target.as_str(), "compute_120");
        assert!(module.compiler.nvrtc_library_known);
        assert_eq!(module.compiler.output_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.composer_revision, 1);
        assert_eq!(module.compiler.compiler_revision, 2);
        assert_eq!(module.compiler.numeric_abi_revision, 5);
        assert_eq!(module.compiler.schedule_revision, 8);
        assert_eq!(module.device_caps.compute_capability, (12, 0));
        assert_eq!(module.device_caps.nvrtc_version, (12, 8));
        assert_eq!(
            module
                .device_caps
                .accepted_target
                .expect("accepted CUDA 12.8 target")
                .as_str(),
            "compute_120"
        );
        assert_eq!(module.device_caps.optin_shared_bytes, 101_376);
        assert!(module.device_caps.tensor_map_access);
    }

    #[test]
    fn sm120_tf32_cuda_12_8_route_manifest_is_literal_and_complete() {
        let cohort = sm120_cohort((12, 8));
        assert_sm120_route_manifest(
            cohort.cells,
            [
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    3072,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    768,
                    3072,
                    2048,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    768,
                    3072,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    768,
                    1536,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    1536,
                    768,
                    2048,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S4),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    1536,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    4621,
                    1928,
                    384,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    384,
                    1928,
                    4621,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S4),
                ),
                (
                    ResolvedGemmOp::Nt,
                    4621,
                    384,
                    1928,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
            ],
        );
        for cell in cohort.cells {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    operands,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
                F32TriadSelection::Tf32(cell.route),
            );
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
            );
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_0_identity_matches_the_literal_qualification_manifest() {
        let identity = sm120_cohort((13, 0)).identity;
        assert_eq!(identity.module_kind, ModuleKind::TriadSm120);
        assert_eq!(identity.module_target, "compute_120");
        assert_eq!(identity.device_target, "sm_120");
        assert_eq!(identity.compute_capability, (12, 0));
        assert_eq!(identity.multiprocessor_count, 170);
        assert_eq!(identity.nvrtc_version, (13, 0));
        assert_eq!(identity.driver_api_version, 13_020);
        assert_eq!(identity.driver_build_sources, 7);
        assert_eq!(identity.optin_shared_bytes, 101_376);
        assert!(identity.tensor_map_access);
        assert_eq!(
            digest_hex(&identity.compile_key),
            "da248ae349b8ebddf6ede70b276cd1bfa620a7ddfba93fc922517f8688a6474f"
        );
        assert_eq!(
            digest_hex(&identity.artifact_digest),
            "51816c8d7906196d49aa6807742c0e93ac1732f23a96472eaab97dc446efb5b8"
        );
        assert_eq!(
            digest_hex(&identity.source_digest),
            "246d2e35eb059cd696ec74325d4a223312fa605fe84ff58f17f6638d99b8182c"
        );
        assert_eq!(
            digest_hex(&identity.invocation_digest),
            "da248ae349b8ebddf6ede70b276cd1bfa620a7ddfba93fc922517f8688a6474f"
        );
        assert_eq!(
            digest_hex(&identity.header_manifest_digest),
            "f4fff8418bd2c346c86ec6f07f5e7417147c013bf038cdc16df3e76cf3f80ad9"
        );
        assert_eq!(
            digest_hex(&identity.nvrtc_library_domain),
            "709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d"
        );
        assert_eq!(
            digest_hex(&identity.driver_build_digest),
            "bbe8397f6ef11a506a502d127d5eb82745a515ab64f9b2901dab4e834bf190d8"
        );

        let module = qualified_module_for_auto_identity(identity);
        assert_eq!(module.artifact.module_kind, ModuleKind::TriadSm120);
        assert_eq!(module.artifact.artifact_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.target.as_str(), "compute_120");
        assert!(module.compiler.nvrtc_library_known);
        assert_eq!(module.compiler.output_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.composer_revision, 1);
        assert_eq!(module.compiler.compiler_revision, 2);
        assert_eq!(module.compiler.numeric_abi_revision, 5);
        assert_eq!(module.compiler.schedule_revision, 8);
        assert_eq!(module.device_caps.compute_capability, (12, 0));
        assert_eq!(module.device_caps.nvrtc_version, (13, 0));
        assert_eq!(
            module
                .device_caps
                .accepted_target
                .expect("accepted CUDA 13.0 target")
                .as_str(),
            "compute_120"
        );
        assert_eq!(module.device_caps.optin_shared_bytes, 101_376);
        assert!(module.device_caps.tensor_map_access);
    }

    #[test]
    fn sm120_tf32_cuda_13_0_route_manifest_is_literal_and_complete() {
        let cohort = sm120_cohort((13, 0));
        assert_sm120_route_manifest(
            cohort.cells,
            [
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    3072,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    768,
                    3072,
                    2048,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    768,
                    3072,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    768,
                    1536,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    1536,
                    768,
                    2048,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    1536,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    4621,
                    1928,
                    384,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    384,
                    1928,
                    4621,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nt,
                    4621,
                    384,
                    1928,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
            ],
        );
        for cell in cohort.cells {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    operands,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
                F32TriadSelection::Tf32(cell.route),
            );
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
            );
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_0_all_nine_routes_fail_closed_on_request_or_operand_drift() {
        let cohort = sm120_cohort((13, 0));
        assert_eq!(cohort.cells.len(), 9);
        for cell in cohort.cells {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };

            for (axis, value) in [
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            ]
            .into_iter()
            .enumerate()
            {
                for changed in [value - 1, value + 1] {
                    let mut shape = [
                        cell.shape.output_rows,
                        cell.shape.output_columns,
                        cell.shape.reduction,
                    ];
                    shape[axis] = changed;
                    assert_no_tf32_route(
                        resolve_f32_triad_auto_with_operands(
                            F32TriadPolicy::AllowDeterministicTf32V1,
                            normalized_request(cell.op, shape[0], shape[1], shape[2]),
                            operands,
                            sm120_availability_for(cohort.identity),
                        )
                        .unwrap(),
                    );
                }
            }

            for stride in 0..3 {
                let mut noncontiguous = request;
                match stride {
                    0 => noncontiguous.shape.lda += 1,
                    1 => noncontiguous.shape.ldb += 1,
                    _ => noncontiguous.shape.ldc += 1,
                }
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        noncontiguous,
                        operands,
                        sm120_availability_for(cohort.identity),
                    )
                    .unwrap(),
                );
            }

            for rejected in [
                F32TriadOperands {
                    output: 0,
                    ..operands
                },
                F32TriadOperands { a: 0, ..operands },
                F32TriadOperands { b: 0, ..operands },
                F32TriadOperands {
                    output: 0x1004,
                    ..operands
                },
                F32TriadOperands {
                    a: 0x2004,
                    ..operands
                },
                F32TriadOperands {
                    b: 0x3004,
                    ..operands
                },
                F32TriadOperands {
                    bias: Some(0x4000),
                    ..operands
                },
                F32TriadOperands {
                    alpha: 0.5,
                    ..operands
                },
                F32TriadOperands {
                    beta: if cell.op == ResolvedGemmOp::Tn {
                        0.0
                    } else {
                        1.0
                    },
                    ..operands
                },
            ] {
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        request,
                        rejected,
                        sm120_availability_for(cohort.identity),
                    )
                    .unwrap(),
                );
            }

            // The exact policy owns the exact-F32 SM120 family: a measured
            // cell resolves to its qualified arm whatever the TF32
            // qualification identity says, everything else stays scalar.
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::ExactScalarFmaV1,
                    request,
                    operands,
                    sm120_availability_for(cohort.identity),
                )
                .unwrap(),
                expected_exact_selection(
                    request,
                    operands,
                    sm120_availability_for(cohort.identity)
                ),
            );
        }
    }

    /// The exact-policy selection the measured-cell table implies: the
    /// official arm for a measured shape on a bound CC 12.0 module with
    /// tensor-map-aligned operands, otherwise the scalar routes.
    fn expected_exact_selection(
        request: F32TriadRequest,
        operands: F32TriadOperands,
        availability: F32TriadAvailability,
    ) -> F32TriadSelection {
        super::sm120_fma_exact_route(request, operands, availability).map_or(
            F32TriadSelection::ScalarFmaV1,
            F32TriadSelection::ExactSm120Fma,
        )
    }

    #[test]
    fn exact_policy_measured_cells_resolve_to_their_official_arms() {
        let identity = sm120_cohort((13, 2)).identity;
        let availability = sm120_availability_for(identity);
        for cell in super::SM120_FMA_MEASURED_CELLS_CC120_170 {
            let request = F32TriadRequest {
                op: cell.op,
                shape: cell.shape,
            };
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::ExactScalarFmaV1,
                    request,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ExactSm120Fma(cell.route),
                "{:?} {:?}",
                cell.op,
                cell.shape
            );
            // A misaligned operand or leading dimension keeps the scalar
            // routes: the tensor maps cannot describe it.
            let misaligned = F32TriadOperands {
                a: 0x2004,
                ..operands
            };
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::ExactScalarFmaV1,
                    request,
                    misaligned,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
            );
            // Under the TF32 policy a measured cell takes its TF32 route when
            // the cohort has one and the exact family otherwise; it never
            // falls through to the bare scalar route.
            assert!(!matches!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1
                    | F32TriadSelection::Tf32(super::Tf32PhysicalRoute::Sm120TmaFmaExactV1(_))
            ));
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_2_route_manifest_is_literal_and_complete() {
        let cells = sm120_cohort((13, 2)).cells;
        assert_eq!(
            cells[0],
            super::Tf32AutoCell {
                op: ResolvedGemmOp::Tn,
                shape: super::Tf32ExactShape {
                    output_rows: 512,
                    output_columns: 384,
                    reduction: 256,
                },
                route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
                tuning_revision: F32_TF32_TUNING_REVISION,
                operand_gate: Tf32AutoOperandGate::RequiresVectorAlignmentEvidence,
            }
        );
        assert_sm120_route_manifest(
            &cells[1..],
            [
                (
                    ResolvedGemmOp::Nn,
                    512,
                    768,
                    3072,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M80N32Bk64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    3072,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    768,
                    3072,
                    2048,
                    Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    768,
                    3072,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    2048,
                    768,
                    1536,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    1536,
                    768,
                    2048,
                    Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
                ),
                (
                    ResolvedGemmOp::Nt,
                    2048,
                    1536,
                    768,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Nn,
                    4621,
                    1928,
                    384,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
                ),
                (
                    ResolvedGemmOp::Tn,
                    384,
                    1928,
                    4621,
                    Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
                ),
                (
                    ResolvedGemmOp::Nt,
                    4621,
                    384,
                    1928,
                    Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                ),
            ],
        );
    }

    #[test]
    fn tf32_tn_underfill_qualification_uses_current_tuning_revision() {
        assert_eq!(TUNING_TABLE_REVISION, 38);
        assert_eq!(F32_TF32_TUNING_REVISION, 38);
    }

    #[test]
    fn sm120_tf32_cuda_13_2_tn_underfill_cell_is_exact_and_fail_closed() {
        let identity = sm120_cohort((13, 2)).identity;
        let request = normalized_request(ResolvedGemmOp::Tn, 512, 384, 256);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };
        let route = Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        let availability = || {
            let mut value = sm120_availability_for(identity);
            value.portable = Some(qualified_module_for_auto_identity(
                super::SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2,
            ));
            value
        };
        let resolve = |policy, request, operands, availability| {
            resolve_f32_triad_auto_with_operands(policy, request, operands, availability).unwrap()
        };

        assert_eq!(
            resolve(
                F32TriadPolicy::AllowDeterministicTf32V1,
                request,
                operands,
                availability(),
            ),
            F32TriadSelection::Tf32(route),
        );
        assert_eq!(
            resolve(
                F32TriadPolicy::ExactScalarFmaV1,
                request,
                operands,
                availability(),
            ),
            F32TriadSelection::ScalarFmaV1,
        );

        for rejected in [
            normalized_request(ResolvedGemmOp::Tn, 511, 384, 256),
            normalized_request(ResolvedGemmOp::Tn, 513, 384, 256),
            normalized_request(ResolvedGemmOp::Tn, 512, 383, 256),
            normalized_request(ResolvedGemmOp::Tn, 512, 385, 256),
            normalized_request(ResolvedGemmOp::Tn, 512, 384, 255),
            normalized_request(ResolvedGemmOp::Tn, 512, 384, 257),
            normalized_request(ResolvedGemmOp::Nn, 512, 384, 256),
            normalized_request(ResolvedGemmOp::Nt, 512, 384, 256),
        ] {
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    rejected,
                    operands,
                    availability(),
                ),
                F32TriadSelection::ScalarFmaV1,
            );
        }
        for mutate in [
            |shape: &mut F32TriadShape| shape.lda += 1,
            |shape: &mut F32TriadShape| shape.ldb += 1,
            |shape: &mut F32TriadShape| shape.ldc += 1,
        ] {
            let mut rejected = request;
            mutate(&mut rejected.shape);
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    rejected,
                    operands,
                    availability(),
                ),
                F32TriadSelection::ScalarFmaV1,
            );
        }
        for rejected in [
            F32TriadOperands {
                output: 0x1004,
                ..operands
            },
            F32TriadOperands {
                a: 0x2004,
                ..operands
            },
            F32TriadOperands {
                b: 0x3004,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: -0.0,
                ..operands
            },
            F32TriadOperands {
                beta: 0.0,
                ..operands
            },
        ] {
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    rejected,
                    availability(),
                ),
                F32TriadSelection::ScalarFmaV1,
            );
        }
        let mut unavailable = availability();
        unavailable.portable = None;
        assert_eq!(
            resolve(
                F32TriadPolicy::AllowDeterministicTf32V1,
                request,
                operands,
                unavailable,
            ),
            F32TriadSelection::ScalarFmaV1,
        );
        for mutate in sm120_identity_mutations() {
            let mut rejected = availability();
            mutate(rejected.specialized.as_mut().unwrap());
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    operands,
                    rejected,
                ),
                F32TriadSelection::ScalarFmaV1,
            );
        }
        for mutate in sm120_identity_mutations()
            .into_iter()
            .chain(portable_sm120_coupled_mutations())
        {
            let mut rejected = availability();
            mutate(rejected.portable.as_mut().unwrap());
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    operands,
                    rejected,
                ),
                F32TriadSelection::ScalarFmaV1,
            );
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_2_rect_wide_cell_is_exact_and_fail_closed() {
        let cohort = sm120_cohort((13, 2));
        let request = normalized_request(ResolvedGemmOp::Nn, 512, 768, 3072);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let route = Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route {
            tile: Tf32Sm120Tile::M80N32Bk64,
            stages: Tf32Sm120Stages::S2,
        });
        let resolve = |policy, request, operands, availability| {
            resolve_f32_triad_auto_with_operands(policy, request, operands, availability).unwrap()
        };

        assert_eq!(
            resolve(
                F32TriadPolicy::AllowDeterministicTf32V1,
                request,
                operands,
                sm120_availability_for(cohort.identity),
            ),
            F32TriadSelection::Tf32(route),
        );
        assert_eq!(
            resolve_f32_triad_auto(
                F32TriadPolicy::AllowDeterministicTf32V1,
                request,
                sm120_availability_for(cohort.identity),
            )
            .unwrap(),
            F32TriadSelection::ScalarFmaV1,
        );

        for (name, rejected) in [
            (
                "M-1",
                normalized_request(ResolvedGemmOp::Nn, 511, 768, 3072),
            ),
            (
                "M+1",
                normalized_request(ResolvedGemmOp::Nn, 513, 768, 3072),
            ),
            (
                "N-1",
                normalized_request(ResolvedGemmOp::Nn, 512, 767, 3072),
            ),
            (
                "N+1",
                normalized_request(ResolvedGemmOp::Nn, 512, 769, 3072),
            ),
            (
                "K-1",
                normalized_request(ResolvedGemmOp::Nn, 512, 768, 3071),
            ),
            (
                "K+1",
                normalized_request(ResolvedGemmOp::Nn, 512, 768, 3073),
            ),
            ("TN", normalized_request(ResolvedGemmOp::Tn, 512, 768, 3072)),
            ("NT", normalized_request(ResolvedGemmOp::Nt, 512, 768, 3072)),
        ] {
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    rejected,
                    operands,
                    sm120_availability_for(cohort.identity),
                ),
                F32TriadSelection::ScalarFmaV1,
                "rect-wide {name} mutation was admitted",
            );
        }

        for (name, rejected) in [
            (
                "null output",
                F32TriadOperands {
                    output: 0,
                    ..operands
                },
            ),
            ("null A", F32TriadOperands { a: 0, ..operands }),
            ("null B", F32TriadOperands { b: 0, ..operands }),
            (
                "output alignment",
                F32TriadOperands {
                    output: 0x1004,
                    ..operands
                },
            ),
            (
                "A alignment",
                F32TriadOperands {
                    a: 0x2004,
                    ..operands
                },
            ),
            (
                "B alignment",
                F32TriadOperands {
                    b: 0x3004,
                    ..operands
                },
            ),
            (
                "alpha",
                F32TriadOperands {
                    alpha: 0.5,
                    ..operands
                },
            ),
            (
                "beta sign",
                F32TriadOperands {
                    beta: -0.0,
                    ..operands
                },
            ),
            (
                "bias",
                F32TriadOperands {
                    bias: Some(0x4000),
                    ..operands
                },
            ),
        ] {
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    rejected,
                    sm120_availability_for(cohort.identity),
                ),
                F32TriadSelection::ScalarFmaV1,
                "rect-wide {name} mutation was admitted",
            );
        }
        assert_eq!(
            resolve(
                F32TriadPolicy::ExactScalarFmaV1,
                request,
                operands,
                sm120_availability_for(cohort.identity),
            ),
            F32TriadSelection::ScalarFmaV1,
        );

        for (field, mutate) in sm120_identity_mutations().into_iter().enumerate() {
            let mut availability = sm120_availability_for(cohort.identity);
            mutate(availability.specialized.as_mut().unwrap());
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    operands,
                    availability,
                ),
                F32TriadSelection::ScalarFmaV1,
                "rect-wide environment mutation {field} was admitted",
            );
        }
        for version in [(12, 8), (13, 0)] {
            assert_eq!(
                resolve(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    operands,
                    sm120_availability_for(sm120_cohort(version).identity),
                ),
                F32TriadSelection::ScalarFmaV1,
                "rect-wide cell leaked into CUDA {version:?}",
            );
        }
    }

    #[test]
    fn sm120_tf32_cuda_12_8_identity_rejects_all_29_single_field_mutations() {
        let cohort = sm120_cohort((12, 8));
        let exact = qualified_module_for_auto_identity(cohort.identity);
        assert_eq!(
            matching_tf32_cohort(exact, SM120_TF32_EVIDENCE_COHORTS),
            Some(&cohort),
        );
        for (field, mutate) in sm120_identity_mutations().into_iter().enumerate() {
            let mut module = exact;
            mutate(&mut module);
            assert!(
                matching_tf32_cohort(module, SM120_TF32_EVIDENCE_COHORTS).is_none(),
                "12.8 identity mutation {field} was admitted",
            );
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_0_identity_rejects_all_29_single_field_mutations() {
        let cohort = sm120_cohort((13, 0));
        let exact = qualified_module_for_auto_identity(cohort.identity);
        assert_eq!(
            matching_tf32_cohort(exact, SM120_TF32_EVIDENCE_COHORTS),
            Some(&cohort),
        );
        for (field, mutate) in sm120_identity_mutations().into_iter().enumerate() {
            let mut module = exact;
            mutate(&mut module);
            assert!(
                matching_tf32_cohort(module, SM120_TF32_EVIDENCE_COHORTS).is_none(),
                "13.0 identity mutation {field} was admitted",
            );
        }
    }

    #[test]
    fn sm120_tf32_real_cohorts_bind_routes_and_reject_cross_version_splices() {
        let cuda_12_8 = sm120_cohort((12, 8));
        let cuda_13_0 = sm120_cohort((13, 0));
        let cuda_13_2 = sm120_cohort((13, 2));
        let request = normalized_request(ResolvedGemmOp::Tn, 768, 3072, 2048);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };
        assert_eq!(
            measured_tf32_cell(request, operands, F32_TF32_TUNING_REVISION, cuda_12_8.cells,),
            Some(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route {
                tile: Tf32Sm120Tile::M64N128,
                stages: Tf32Sm120Stages::S2,
            },)),
        );
        assert_eq!(
            measured_tf32_cell(request, operands, F32_TF32_TUNING_REVISION, cuda_13_0.cells,),
            Some(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route {
                tile: Tf32Sm120Tile::M128N64,
                stages: Tf32Sm120Stages::S2,
            },)),
        );
        assert_eq!(
            measured_tf32_cell(request, operands, F32_TF32_TUNING_REVISION, cuda_13_2.cells,),
            Some(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(
                Tf32Sm120Route {
                    tile: Tf32Sm120Tile::M64N128,
                    stages: Tf32Sm120Stages::S3,
                },
            )),
        );

        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32V1,
                request,
                operands,
                sm120_availability_for(cuda_13_0.identity),
            )
            .unwrap(),
            F32TriadSelection::Tf32(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route {
                tile: Tf32Sm120Tile::M128N64,
                stages: Tf32Sm120Stages::S2,
            },)),
        );
        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32V1,
                request,
                operands,
                sm120_availability_for(cuda_12_8.identity),
            )
            .unwrap(),
            F32TriadSelection::Tf32(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route {
                tile: Tf32Sm120Tile::M64N128,
                stages: Tf32Sm120Stages::S2,
            },)),
        );
        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32V1,
                request,
                operands,
                sm120_availability_for(cuda_13_2.identity),
            )
            .unwrap(),
            F32TriadSelection::Tf32(Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(
                Tf32Sm120Route {
                    tile: Tf32Sm120Tile::M64N128,
                    stages: Tf32Sm120Stages::S3,
                },
            )),
        );

        for source in [cuda_12_8, cuda_13_0, cuda_13_2] {
            for destination in [cuda_12_8, cuda_13_0, cuda_13_2] {
                if source.identity.nvrtc_version == destination.identity.nvrtc_version {
                    continue;
                }
                let mut spliced = qualified_module_for_auto_identity(source.identity);
                spliced.compiler.nvrtc_version = destination.identity.nvrtc_version;
                spliced.device_caps.nvrtc_version = destination.identity.nvrtc_version;
                assert!(
                    matching_tf32_cohort(spliced, SM120_TF32_EVIDENCE_COHORTS).is_none(),
                    "CUDA {:?} body admitted with CUDA {:?} versions",
                    source.identity.nvrtc_version,
                    destination.identity.nvrtc_version,
                );
            }
        }
    }

    /// A cohort is frozen against the module source it was measured on. Editing
    /// any kernel of that module changes the source digest, and every cohort
    /// whose digest no longer matches is silently unreachable: the TF32 policy
    /// resolves nothing and falls through to the exact floor with no error.
    /// At least one cohort has to still describe this tree, or the whole
    /// deterministic TF32 family is dead code until someone requalifies it.
    #[test]
    fn at_least_one_sm120_tf32_cohort_matches_this_tree() {
        let live = super::super::modules::module_source_digest(ModuleKind::TriadSm120)
            .expect("compose the SM120 module source");
        let stale: Vec<usize> = SM120_TF32_EVIDENCE_COHORTS
            .iter()
            .enumerate()
            .filter(|(_, cohort)| cohort.identity.source_digest != live)
            .map(|(index, _)| index)
            .collect();
        assert!(
            stale.len() < SM120_TF32_EVIDENCE_COHORTS.len(),
            "every SM120 TF32 cohort is frozen against a source this tree no \
             longer contains: requalify one against the current kernels or \
             delete the table. Stale cohort indices: {stale:?}",
        );
    }

    #[test]
    fn sm120_tf32_evidence_cohorts_are_exact_and_unambiguous() {
        assert!(!SM120_TF32_EVIDENCE_COHORTS.is_empty());
        assert_eq!(
            SM120_TF32_EVIDENCE_COHORTS
                .iter()
                .filter(|cohort| cohort.identity.nvrtc_version == (12, 8))
                .count(),
            1,
        );
        assert_eq!(
            SM120_TF32_EVIDENCE_COHORTS
                .iter()
                .filter(|cohort| cohort.identity.nvrtc_version == (13, 0))
                .count(),
            1,
        );
        // One NVRTC version can carry several cohorts: a cohort is a whole
        // stack, and two driver builds of the same toolkit chose different
        // routes. What must stay unique is the driver build behind each one.
        assert!(
            SM120_TF32_EVIDENCE_COHORTS
                .iter()
                .filter(|cohort| cohort.identity.nvrtc_version == (13, 2))
                .count()
                >= 1,
        );
        assert!(
            SM120_TF32_EVIDENCE_COHORTS
                .iter()
                .all(|cohort| cohort.identity.nvrtc_version != (13, 1))
        );
        for (index, cohort) in SM120_TF32_EVIDENCE_COHORTS.iter().enumerate() {
            for other in SM120_TF32_EVIDENCE_COHORTS.iter().skip(index + 1) {
                assert!(
                    cohort.identity.nvrtc_version != other.identity.nvrtc_version
                        || cohort.identity.driver_build_digest
                            != other.identity.driver_build_digest,
                    "cohort {index} shares a stack with a later cohort",
                );
            }
        }
        for (index, cohort) in SM120_TF32_EVIDENCE_COHORTS.iter().enumerate() {
            assert_eq!(cohort.identity.module_kind, ModuleKind::TriadSm120);
            assert!(!cohort.cells.is_empty());
            assert_eq!(
                SM120_TF32_EVIDENCE_COHORTS
                    .iter()
                    .filter(|candidate| {
                        candidate
                            .identity
                            .matches(qualified_module_for_auto_identity(cohort.identity))
                    })
                    .count(),
                1,
                "cohort {index} did not match exactly once",
            );
            assert_eq!(
                matching_tf32_cohort(
                    qualified_module_for_auto_identity(cohort.identity),
                    SM120_TF32_EVIDENCE_COHORTS,
                ),
                Some(cohort),
            );
            for other in &SM120_TF32_EVIDENCE_COHORTS[index + 1..] {
                assert_ne!(cohort.identity, other.identity);
            }
        }
    }

    #[test]
    fn sm120_tf32_operand_aware_resolution_admits_only_archived_cells() {
        assert_eq!(SM120_TF32_EVIDENCE_CELLS.len(), 11);
        assert_eq!(
            SM120_TF32_EVIDENCE_CELLS
                .iter()
                .filter(|cell| matches!(
                    cell.operand_gate,
                    Tf32AutoOperandGate::RequiresNoBiasAndVectorAlignmentEvidence
                ))
                .count(),
            4
        );
        assert_eq!(
            SM120_TF32_EVIDENCE_CELLS
                .iter()
                .filter(|cell| matches!(
                    cell.operand_gate,
                    Tf32AutoOperandGate::RequiresVectorAlignmentEvidence
                ))
                .count(),
            7
        );
        let unique_keys = SM120_TF32_EVIDENCE_CELLS
            .iter()
            .map(|cell| {
                (
                    cell.op as u8,
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique_keys.len(), SM120_TF32_EVIDENCE_CELLS.len());
        let expected = [
            (
                ResolvedGemmOp::Nn,
                512,
                768,
                3072,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M80N32Bk64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                3072,
                768,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Tn,
                768,
                3072,
                2048,
                Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                768,
                3072,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                1536,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Tn,
                1536,
                768,
                2048,
                Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                1536,
                768,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Nn,
                4621,
                1928,
                384,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M128N64, Tf32Sm120Stages::S2),
            ),
            (
                ResolvedGemmOp::Tn,
                384,
                1928,
                4621,
                Sm120ManifestRoute::StreamK(Tf32Sm120Tile::M64N128, Tf32Sm120Stages::S3),
            ),
            (
                ResolvedGemmOp::Nt,
                4621,
                384,
                1928,
                Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
            ),
        ];
        assert_eq!(
            SM120_TF32_EVIDENCE_CELLS[0].route,
            Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                tile: Tf32PortableTile::M16N32,
                stages: Tf32PortableStages::S4,
            })
        );
        for (cell, (op, rows, columns, reduction, route)) in
            SM120_TF32_EVIDENCE_CELLS[1..].iter().zip(expected)
        {
            assert_eq!(cell.op, op);
            assert_eq!(
                cell.shape,
                super::Tf32ExactShape {
                    output_rows: rows,
                    output_columns: columns,
                    reduction,
                }
            );
            assert_eq!(cell.route, route.route());
        }
        for cohort in SM120_TF32_EVIDENCE_COHORTS {
            let mut exact_availability = sm120_availability_for(cohort.identity);
            exact_availability.portable = Some(qualified_module_for_auto_identity(
                super::SM120_TF32_PORTABLE_QUALIFICATION_IDENTITY_CUDA_13_2,
            ));
            for cell in cohort.cells {
                assert!(
                    matches!(
                        (cell.op, cell.operand_gate),
                        (
                            ResolvedGemmOp::Nn,
                            Tf32AutoOperandGate::RequiresNoBiasAndVectorAlignmentEvidence
                        ) | (
                            ResolvedGemmOp::Tn,
                            Tf32AutoOperandGate::RequiresVectorAlignmentEvidence
                        ) | (
                            ResolvedGemmOp::Nt,
                            Tf32AutoOperandGate::RequiresVectorAlignmentEvidence
                        )
                    ),
                    "SM120 cell gate must agree with the strict shared operand predicate: {cell:?}",
                );
                let request = normalized_request(
                    cell.op,
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                );
                let operands = F32TriadOperands {
                    output: 0x1000,
                    a: 0x2000,
                    b: 0x3000,
                    bias: None,
                    alpha: 1.0,
                    beta: if cell.op == ResolvedGemmOp::Tn {
                        1.0
                    } else {
                        0.0
                    },
                };
                assert_eq!(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        request,
                        operands,
                        exact_availability,
                    )
                    .unwrap(),
                    F32TriadSelection::Tf32(cell.route),
                );
                assert_eq!(
                    resolve_f32_triad_auto(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        request,
                        exact_availability,
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFmaV1,
                );

                for (axis, value) in [
                    cell.shape.output_rows,
                    cell.shape.output_columns,
                    cell.shape.reduction,
                ]
                .into_iter()
                .enumerate()
                {
                    for changed in [value - 1, value + 1] {
                        let mut shape = [
                            cell.shape.output_rows,
                            cell.shape.output_columns,
                            cell.shape.reduction,
                        ];
                        shape[axis] = changed;
                        assert_no_tf32_route(
                            resolve_f32_triad_auto_with_operands(
                                F32TriadPolicy::AllowDeterministicTf32V1,
                                normalized_request(cell.op, shape[0], shape[1], shape[2]),
                                operands,
                                exact_availability,
                            )
                            .unwrap(),
                        );
                    }
                }

                for stride in 0..3 {
                    let mut noncontiguous = request;
                    match stride {
                        0 => noncontiguous.shape.lda += 1,
                        1 => noncontiguous.shape.ldb += 1,
                        _ => noncontiguous.shape.ldc += 1,
                    }
                    assert_no_tf32_route(
                        resolve_f32_triad_auto_with_operands(
                            F32TriadPolicy::AllowDeterministicTf32V1,
                            noncontiguous,
                            operands,
                            exact_availability,
                        )
                        .unwrap(),
                    );
                }

                for rejected in [
                    F32TriadOperands {
                        output: 0,
                        ..operands
                    },
                    F32TriadOperands { a: 0, ..operands },
                    F32TriadOperands { b: 0, ..operands },
                    F32TriadOperands {
                        output: 0x1004,
                        ..operands
                    },
                    F32TriadOperands {
                        a: 0x2004,
                        ..operands
                    },
                    F32TriadOperands {
                        b: 0x3004,
                        ..operands
                    },
                    F32TriadOperands {
                        bias: Some(0x4000),
                        ..operands
                    },
                    F32TriadOperands {
                        alpha: 0.5,
                        ..operands
                    },
                    F32TriadOperands {
                        beta: if cell.op == ResolvedGemmOp::Tn {
                            0.0
                        } else {
                            1.0
                        },
                        ..operands
                    },
                ] {
                    assert_no_tf32_route(
                        resolve_f32_triad_auto_with_operands(
                            F32TriadPolicy::AllowDeterministicTf32V1,
                            request,
                            rejected,
                            exact_availability,
                        )
                        .unwrap(),
                    );
                }
            }

            let unmeasured = normalized_request(ResolvedGemmOp::Tn, 385, 1928, 4621);
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: 1.0,
            };
            assert_no_tf32_route(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    unmeasured,
                    operands,
                    exact_availability,
                )
                .unwrap(),
            );

            let measured = normalized_request(ResolvedGemmOp::Tn, 384, 1928, 4621);
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::ExactScalarFmaV1,
                    measured,
                    operands,
                    exact_availability,
                )
                .unwrap(),
                expected_exact_selection(measured, operands, exact_availability),
            );

            for mutate in sm120_identity_mutations() {
                let mut availability = exact_availability;
                mutate(availability.specialized.as_mut().unwrap());
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        measured,
                        operands,
                        availability,
                    )
                    .unwrap(),
                );
            }
            for version in [(12, 8), (13, 0), (13, 1), (13, 2), (13, 3)]
                .into_iter()
                .filter(|version| *version != cohort.identity.nvrtc_version)
            {
                let mut availability = exact_availability;
                let module = availability.specialized.as_mut().unwrap();
                module.compiler.nvrtc_version = version;
                module.device_caps.nvrtc_version = version;
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        measured,
                        operands,
                        availability,
                    )
                    .unwrap(),
                );
            }
            for multiprocessors in [169, 171] {
                let mut availability = exact_availability;
                availability
                    .specialized
                    .as_mut()
                    .unwrap()
                    .device
                    .multiprocessor_count = multiprocessors;
                assert_no_tf32_route(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        measured,
                        operands,
                        availability,
                    )
                    .unwrap(),
                );
            }
        }
    }

    #[test]
    fn sm120_tf32_cuda_13_2_identity_matches_the_literal_qualification_manifest() {
        let identity = sm120_cohort((13, 2)).identity;
        assert_eq!(identity.module_kind, ModuleKind::TriadSm120);
        assert_eq!(identity.module_target, "compute_120");
        assert_eq!(identity.device_target, "sm_120");
        assert_eq!(identity.compute_capability, (12, 0));
        assert_eq!(identity.multiprocessor_count, 170);
        assert_eq!(identity.nvrtc_version, (13, 2));
        assert_eq!(identity.driver_api_version, 13_020);
        assert_eq!(identity.driver_build_sources, 7);
        assert_eq!(identity.optin_shared_bytes, 101_376);
        assert!(identity.tensor_map_access);
        assert_eq!(
            digest_hex(&identity.compile_key),
            "70ee458820f27c554ee77d3f89c990cf005d7e3e566dc487ea7f88922ec33645"
        );
        assert_eq!(
            digest_hex(&identity.artifact_digest),
            "50ceabc64d69e8575a7974d3449605980cb1b42ddef8952ace3725073cd90441"
        );
        assert_eq!(
            digest_hex(&identity.source_digest),
            "65dfbcc164229b73f3f76fa15016e28a3adf9ad2328030eb1fe6d97ee3d01aaf"
        );
        assert_eq!(
            digest_hex(&identity.invocation_digest),
            "70ee458820f27c554ee77d3f89c990cf005d7e3e566dc487ea7f88922ec33645"
        );
        assert_eq!(
            digest_hex(&identity.header_manifest_digest),
            "905acac69a0bef20b12df1bd2fb70b6f8bd38fc530ec02bff5f2b124bf8d9157"
        );
        assert_eq!(
            digest_hex(&identity.nvrtc_library_domain),
            "0e0dc3faa997ae96442ef62ffc02640d361a5c50e0b4e5df2a05bc0986fe6241"
        );
        assert_eq!(
            digest_hex(&identity.driver_build_digest),
            "c6a7ba2d18fd806d98363b1d7ed17f3c347010e3b2a802283ddc191fbb577abc"
        );

        let module = qualified_module_for_auto_identity(identity);
        assert_eq!(module.artifact.module_kind, ModuleKind::TriadSm120);
        assert_eq!(module.artifact.artifact_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.target.as_str(), "compute_120");
        assert!(module.compiler.nvrtc_library_known);
        assert_eq!(module.compiler.output_kind, ArtifactKind::Ptx);
        assert_eq!(module.compiler.composer_revision, 1);
        assert_eq!(module.compiler.compiler_revision, 2);
        assert_eq!(module.compiler.numeric_abi_revision, 5);
        assert_eq!(module.compiler.schedule_revision, 8);
        assert_eq!(module.device_caps.compute_capability, (12, 0));
        assert_eq!(module.device_caps.nvrtc_version, (13, 2));
        assert_eq!(
            module
                .device_caps
                .accepted_target
                .expect("accepted CUDA 13.2 target")
                .as_str(),
            "compute_120"
        );
        assert_eq!(module.device_caps.optin_shared_bytes, 101_376);
        assert!(module.device_caps.tensor_map_access);
    }

    const SCALAR_STRIDE_CELLS: [(ResolvedGemmOp, usize, usize, usize); 5] = [
        (ResolvedGemmOp::Tn, 49, 129, 65),
        (ResolvedGemmOp::Tn, 65, 129, 49),
        (ResolvedGemmOp::Tn, 131, 100, 129),
        (ResolvedGemmOp::Nt, 49, 65, 129),
        (ResolvedGemmOp::Nt, 65, 49, 129),
    ];

    const OPERAND_GATED_CELLS: [(ResolvedGemmOp, usize, usize, usize); 29] = [
        (ResolvedGemmOp::Nn, 16, 2048, 512),
        (ResolvedGemmOp::Nn, 49, 129, 65),
        (ResolvedGemmOp::Nn, 65, 129, 49),
        (ResolvedGemmOp::Nn, 129, 100, 131),
        (ResolvedGemmOp::Nn, 512, 768, 3072),
        (ResolvedGemmOp::Nn, 1024, 128, 256),
        (ResolvedGemmOp::Nn, 1024, 512, 128),
        (ResolvedGemmOp::Nn, 2048, 768, 1536),
        (ResolvedGemmOp::Nn, 2048, 768, 3072),
        (ResolvedGemmOp::Nn, 2048, 3072, 768),
        (ResolvedGemmOp::Nn, 4096, 768, 512),
        (ResolvedGemmOp::Nn, 4096, 1536, 3072),
        (ResolvedGemmOp::Nn, 4621, 1928, 384),
        (ResolvedGemmOp::Tn, 16, 2048, 512),
        (ResolvedGemmOp::Tn, 128, 512, 1024),
        (ResolvedGemmOp::Tn, 256, 128, 1024),
        (ResolvedGemmOp::Tn, 512, 384, 256),
        (ResolvedGemmOp::Tn, 3072, 768, 512),
        (ResolvedGemmOp::Tn, 8192, 128, 128),
        (ResolvedGemmOp::Nt, 16, 512, 2048),
        (ResolvedGemmOp::Nt, 128, 8192, 128),
        (ResolvedGemmOp::Nt, 512, 16, 2048),
        (ResolvedGemmOp::Nt, 512, 3072, 768),
        (ResolvedGemmOp::Nt, 1024, 256, 128),
        (ResolvedGemmOp::Nt, 2048, 768, 3072),
        (ResolvedGemmOp::Nt, 2048, 1536, 768),
        (ResolvedGemmOp::Nt, 2048, 3072, 768),
        (ResolvedGemmOp::Nt, 4096, 512, 768),
        (ResolvedGemmOp::Nt, 4621, 384, 1928),
    ];

    #[test]
    fn sm89_tf32_request_only_cells_remain_scalar() {
        for (op, rows, columns, reduction) in SCALAR_STRIDE_CELLS {
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    normalized_request(op, rows, columns, reduction),
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
                "request-only resolution promoted {op:?} {rows}x{columns}x{reduction}",
            );
        }
    }

    #[test]
    fn sm89_tf32_scalar_stride_cells_have_no_bucket_or_layout_bleed() {
        for (op, rows, columns, reduction) in SCALAR_STRIDE_CELLS {
            let request = normalized_request(op, rows, columns, reduction);
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
            };
            let expected = SM89_TF32_EVIDENCE_CELLS
                .iter()
                .find(|cell| {
                    cell.op == op
                        && cell.shape.output_rows == rows
                        && cell.shape.output_columns == columns
                        && cell.shape.reduction == reduction
                })
                .unwrap()
                .route;
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    operands,
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::Tf32(expected),
            );
            for (axis, value) in [rows, columns, reduction].into_iter().enumerate() {
                for changed in [value - 1, value + 1] {
                    let mut normalized = [rows, columns, reduction];
                    normalized[axis] = changed;
                    assert_eq!(
                        resolve_f32_triad_auto_with_operands(
                            F32TriadPolicy::AllowDeterministicTf32V1,
                            normalized_request(op, normalized[0], normalized[1], normalized[2]),
                            operands,
                            sm89_availability(),
                        )
                        .unwrap(),
                        F32TriadSelection::ScalarFmaV1,
                        "bucket bleed for {op:?} {rows}x{columns}x{reduction} axis {axis}",
                    );
                }
            }

            for stride in 0..3 {
                let mut noncontiguous = normalized_request(op, rows, columns, reduction);
                match stride {
                    0 => noncontiguous.shape.lda += 1,
                    1 => noncontiguous.shape.ldb += 1,
                    _ => noncontiguous.shape.ldc += 1,
                }
                assert_eq!(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        noncontiguous,
                        operands,
                        sm89_availability(),
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFmaV1,
                    "layout bleed for {op:?} {rows}x{columns}x{reduction} stride {stride}",
                );
            }
        }
    }

    #[test]
    fn sm89_tf32_cells_are_topology_target_and_revision_exact() {
        let selected = normalized_request(ResolvedGemmOp::Tn, 49, 129, 65);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };
        let mut wrong_sm_count = sm89_availability();
        wrong_sm_count
            .portable
            .as_mut()
            .unwrap()
            .device
            .multiprocessor_count = 141;
        let mut wrong_sm_count_high = sm89_availability();
        wrong_sm_count_high
            .portable
            .as_mut()
            .unwrap()
            .device
            .multiprocessor_count = 143;
        let cases = [
            F32TriadAvailability::default(),
            wrong_sm_count,
            wrong_sm_count_high,
            F32TriadAvailability {
                portable: Some(qualified_module(
                    ModuleKind::TriadSm80,
                    "sm_89",
                    "sm_89",
                    (8, 8),
                    false,
                    99_000,
                )),
                specialized: None,
            },
            F32TriadAvailability {
                portable: Some(qualified_module(
                    ModuleKind::TriadSm80,
                    "sm_90a",
                    "sm_90a",
                    (9, 0),
                    false,
                    99_000,
                )),
                specialized: None,
            },
            F32TriadAvailability {
                portable: Some(qualified_module(
                    ModuleKind::TriadSm80,
                    "sm_80",
                    "sm_80",
                    (8, 9),
                    false,
                    99_000,
                )),
                specialized: None,
            },
            F32TriadAvailability {
                portable: Some(qualified_module(
                    ModuleKind::TriadSm90a,
                    "sm_89",
                    "sm_89",
                    (8, 9),
                    false,
                    99_000,
                )),
                specialized: None,
            },
        ];
        for availability in cases {
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    selected,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
            );
        }
        assert_eq!(
            measured_tf32_route_with_operands(
                selected,
                operands,
                sm89_availability(),
                F32_TF32_TUNING_REVISION - 1,
            ),
            None,
        );
    }

    #[test]
    fn sm89_tf32_cells_are_nvrtc_13_2_exact() {
        let selected = normalized_request(ResolvedGemmOp::Tn, 49, 129, 65);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };
        for version in [(13, 1), (13, 0), (12, 8), (0, 0)] {
            let mut availability = sm89_availability();
            let portable = availability.portable.as_mut().unwrap();
            portable.compiler.nvrtc_version = version;
            portable.device_caps.nvrtc_version = version;
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    selected,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
                "NVRTC {version:?} escaped the exact compiler key",
            );
        }

        let mut unknown_library = sm89_availability();
        unknown_library
            .portable
            .as_mut()
            .unwrap()
            .compiler
            .nvrtc_library_known = false;
        assert_eq!(
            resolve_f32_triad_auto_with_operands(
                F32TriadPolicy::AllowDeterministicTf32V1,
                selected,
                operands,
                unknown_library,
            )
            .unwrap(),
            F32TriadSelection::ScalarFmaV1,
        );

        for (compiler, device_caps) in [((13, 2), (13, 1)), ((13, 1), (13, 2))] {
            let mut availability = sm89_availability();
            let portable = availability.portable.as_mut().unwrap();
            portable.compiler.nvrtc_version = compiler;
            portable.device_caps.nvrtc_version = device_caps;
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    selected,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
                "compiler/device NVRTC mismatch {compiler:?}/{device_caps:?} escaped",
            );
        }
    }

    #[test]
    fn sm89_tf32_operand_gated_cells_remain_scalar() {
        for (op, rows, columns, reduction) in OPERAND_GATED_CELLS {
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    normalized_request(op, rows, columns, reduction),
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
                "operand-gated {op:?} {rows}x{columns}x{reduction} was promoted",
            );
        }
    }

    #[test]
    fn sm89_tf32_auto_fails_closed_without_operands_alignment_or_exact_identity() {
        for cell in SM89_TF32_EVIDENCE_CELLS {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
                "request-only resolution promoted {:?}",
                cell.shape,
            );

            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if cell.op == ResolvedGemmOp::Tn {
                    1.0
                } else {
                    0.0
                },
            };
            for rejected in [
                F32TriadOperands {
                    output: 0x1004,
                    ..operands
                },
                F32TriadOperands {
                    a: 0x2004,
                    ..operands
                },
                F32TriadOperands {
                    b: 0x3004,
                    ..operands
                },
            ] {
                assert_eq!(
                    resolve_f32_triad_auto_with_operands(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        request,
                        rejected,
                        sm89_availability(),
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFmaV1,
                    "misaligned operand promoted {:?}",
                    cell.shape,
                );
            }
        }

        let request = normalized_request(ResolvedGemmOp::Nn, 49, 129, 65);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let mutations: [fn(&mut Tf32QualifiedModule); 29] = [
            |module| module.module_kind = ModuleKind::Fixed,
            |module| module.target = CudaTarget::new("sm_80").unwrap(),
            |module| module.artifact.module_kind = ModuleKind::Fixed,
            |module| module.artifact.artifact_kind = ArtifactKind::Cubin,
            |module| module.artifact.compile_key[0] ^= 1,
            |module| module.artifact.artifact_digest[0] ^= 1,
            |module| module.compiler.source_digest[0] ^= 1,
            |module| module.compiler.invocation_digest[0] ^= 1,
            |module| module.compiler.header_manifest_digest[0] ^= 1,
            |module| module.compiler.target = CudaTarget::new("sm_80").unwrap(),
            |module| module.compiler.nvrtc_version = (13, 1),
            |module| module.compiler.nvrtc_library_domain[0] ^= 1,
            |module| module.compiler.nvrtc_library_known = false,
            |module| module.compiler.output_kind = ArtifactKind::Cubin,
            |module| module.compiler.composer_revision ^= 1,
            |module| module.compiler.compiler_revision ^= 1,
            |module| module.compiler.numeric_abi_revision ^= 1,
            |module| module.compiler.schedule_revision ^= 1,
            |module| module.device.compute_capability = (8, 8),
            |module| module.device.multiprocessor_count -= 1,
            |module| module.device.target = CudaTarget::new("sm_80").unwrap(),
            |module| module.device.driver.api_version -= 1,
            |module| module.device.driver.build_sources -= 1,
            |module| module.device.driver.build_digest[0] ^= 1,
            |module| module.device_caps.compute_capability = (8, 8),
            |module| module.device_caps.nvrtc_version = (13, 1),
            |module| module.device_caps.accepted_target = None,
            |module| module.device_caps.optin_shared_bytes -= 1,
            |module| module.device_caps.tensor_map_access = true,
        ];
        for mutate in mutations {
            let mut availability = sm89_availability();
            mutate(availability.portable.as_mut().unwrap());
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    operands,
                    availability,
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
                "mutated qualification identity was admitted",
            );
        }
    }

    #[test]
    fn sm89_tf32_operand_aware_resolution_admits_exact_evidence() {
        for (op, rows, columns, reduction) in
            SCALAR_STRIDE_CELLS.into_iter().chain(OPERAND_GATED_CELLS)
        {
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
            };
            assert!(matches!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    normalized_request(op, rows, columns, reduction),
                    operands,
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::Tf32(_)
            ));
        }
    }

    #[test]
    fn sm89_tf32_operand_aware_resolution_rejects_semantic_and_alignment_drift() {
        let request = normalized_request(ResolvedGemmOp::Nn, 2048, 768, 1536);
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        for rejected in [
            F32TriadOperands {
                output: 0x1004,
                ..operands
            },
            F32TriadOperands {
                a: 0x2004,
                ..operands
            },
            F32TriadOperands {
                b: 0x3004,
                ..operands
            },
            F32TriadOperands {
                bias: Some(0x4000),
                ..operands
            },
            F32TriadOperands {
                alpha: f32::from_bits((-0.0_f32).to_bits()),
                ..operands
            },
            F32TriadOperands {
                beta: -0.0,
                ..operands
            },
        ] {
            assert_eq!(
                resolve_f32_triad_auto_with_operands(
                    F32TriadPolicy::AllowDeterministicTf32V1,
                    request,
                    rejected,
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
            );
        }
    }

    #[test]
    fn sm89_tf32_evidence_inventory_encodes_the_missing_operand_gates() {
        assert_eq!(SM89_TF32_EVIDENCE_CELLS.len(), 43);
        let mut gate_counts = [0_usize; 4];
        for cell in SM89_TF32_EVIDENCE_CELLS {
            let request = normalized_request(
                cell.op,
                cell.shape.output_rows,
                cell.shape.output_columns,
                cell.shape.reduction,
            );
            let async_staging_shape =
                request.shape.lda.is_multiple_of(4) && request.shape.ldb.is_multiple_of(4);
            match cell.operand_gate {
                Tf32AutoOperandGate::RequestContractSafe => {
                    gate_counts[0] += 1;
                    assert_ne!(cell.op, ResolvedGemmOp::Nn);
                    assert!(!async_staging_shape);
                }
                Tf32AutoOperandGate::RequiresNoBiasEvidence => {
                    gate_counts[1] += 1;
                    assert_eq!(cell.op, ResolvedGemmOp::Nn);
                    assert!(!async_staging_shape);
                }
                Tf32AutoOperandGate::RequiresVectorAlignmentEvidence => {
                    gate_counts[2] += 1;
                    assert_ne!(cell.op, ResolvedGemmOp::Nn);
                    assert!(async_staging_shape);
                }
                Tf32AutoOperandGate::RequiresNoBiasAndVectorAlignmentEvidence => {
                    gate_counts[3] += 1;
                    assert_eq!(cell.op, ResolvedGemmOp::Nn);
                    assert!(async_staging_shape);
                }
            }
        }
        assert_eq!(gate_counts, [5, 3, 24, 11]);
    }

    #[test]
    fn sm89_tf32_evidence_inventory_matches_all_qualified_routes() {
        use Tf32AutoOperandGate::{
            RequestContractSafe as Safe, RequiresNoBiasAndVectorAlignmentEvidence as BiasAlign,
            RequiresNoBiasEvidence as Bias, RequiresVectorAlignmentEvidence as Align,
        };
        use Tf32PortableStages::{S2, S3, S4};
        use Tf32PortableTile::{M16N32, M64N64, M128N64};

        let expected = [
            (
                ResolvedGemmOp::Nn,
                16,
                2048,
                512,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                49,
                129,
                65,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Bias,
            ),
            (
                ResolvedGemmOp::Nn,
                65,
                129,
                49,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Bias,
            ),
            (
                ResolvedGemmOp::Nn,
                129,
                100,
                131,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Bias,
            ),
            (
                ResolvedGemmOp::Nn,
                512,
                768,
                3072,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                1024,
                128,
                256,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                1024,
                512,
                128,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                1536,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                768,
                3072,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                2048,
                3072,
                768,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                4096,
                768,
                512,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                4096,
                1536,
                3072,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nn,
                4621,
                1928,
                384,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Tn,
                16,
                2048,
                512,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                49,
                129,
                65,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Tn,
                65,
                129,
                49,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Tn,
                128,
                512,
                1024,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                131,
                100,
                129,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Tn,
                256,
                128,
                1024,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                512,
                384,
                256,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                3072,
                768,
                512,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                8192,
                128,
                128,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                16,
                512,
                2048,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                49,
                65,
                129,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Nt,
                65,
                49,
                129,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Safe,
            ),
            (
                ResolvedGemmOp::Nt,
                128,
                8192,
                128,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                512,
                16,
                2048,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                512,
                3072,
                768,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                1024,
                256,
                128,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S4,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                768,
                3072,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                1536,
                768,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                2048,
                3072,
                768,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                4096,
                512,
                768,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                4621,
                384,
                1928,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                768,
                3072,
                2048,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                1536,
                768,
                2048,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                384,
                1928,
                4621,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                3072,
                1536,
                4096,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                4096,
                3072,
                1536,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Tn,
                3072,
                768,
                2048,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nt,
                1024,
                128,
                512,
                Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(Tf32PortableRoute {
                    tile: M16N32,
                    stages: S3,
                }),
                Align,
            ),
            (
                ResolvedGemmOp::Nn,
                10400,
                1536,
                384,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M64N64,
                    stages: S2,
                }),
                BiasAlign,
            ),
            (
                ResolvedGemmOp::Nt,
                10400,
                384,
                1536,
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: M128N64,
                    stages: S3,
                }),
                Align,
            ),
        ];

        assert_eq!(SM89_TF32_EVIDENCE_CELLS.len(), expected.len());
        for (cell, (op, rows, columns, reduction, route, operand_gate)) in
            SM89_TF32_EVIDENCE_CELLS.iter().zip(expected)
        {
            assert_eq!(cell.op, op);
            assert_eq!(
                cell.shape,
                super::Tf32ExactShape {
                    output_rows: rows,
                    output_columns: columns,
                    reduction,
                }
            );
            assert_eq!(cell.route, route);
            assert_eq!(cell.operand_gate, operand_gate);
        }
    }

    #[test]
    fn exact_scalar_policy_dominates_the_sm89_tf32_table() {
        for (op, rows, columns, reduction) in
            SCALAR_STRIDE_CELLS.into_iter().chain(OPERAND_GATED_CELLS)
        {
            assert_eq!(
                resolve_f32_triad_auto(
                    F32TriadPolicy::ExactScalarFmaV1,
                    normalized_request(op, rows, columns, reduction),
                    sm89_availability(),
                )
                .unwrap(),
                F32TriadSelection::ScalarFmaV1,
            );
        }
    }

    #[test]
    fn exact_and_unmeasured_auto_resolution_stay_scalar() {
        let portable = qualified_module(
            ModuleKind::TriadSm80,
            "sm_89",
            "sm_89",
            (8, 9),
            false,
            99_000,
        );
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            for availability in [
                F32TriadAvailability::default(),
                F32TriadAvailability {
                    portable: Some(portable),
                    specialized: None,
                },
            ] {
                assert_eq!(
                    resolve_f32_triad_auto(
                        F32TriadPolicy::ExactScalarFmaV1,
                        request(op),
                        availability,
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFmaV1
                );
                assert_eq!(
                    resolve_f32_triad_auto(
                        F32TriadPolicy::AllowDeterministicTf32V1,
                        request(op),
                        availability,
                    )
                    .unwrap(),
                    F32TriadSelection::ScalarFmaV1
                );
            }
        }
    }

    #[test]
    fn forced_resolution_accepts_each_exact_qualified_family() {
        let cases = [
            (
                Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
                F32TriadAvailability {
                    portable: Some(qualified_module(
                        ModuleKind::TriadSm80,
                        "sm_89",
                        "sm_89",
                        (8, 9),
                        false,
                        29_696,
                    )),
                    specialized: None,
                },
            ),
            (
                Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(Tf32Sm90aRoute {
                    schedule: Sm90aWarpgroupSchedule::Wg2,
                }),
                F32TriadAvailability {
                    portable: None,
                    specialized: Some(qualified_module(
                        ModuleKind::TriadSm90a,
                        "sm_90a",
                        "sm_90a",
                        (9, 0),
                        true,
                        73_984,
                    )),
                },
            ),
            (
                Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(Tf32Sm100Route {
                    tile: Sm100Tile::M128N128,
                    stages: Sm100Stages::S4,
                    schedule: Sm100Schedule::P8,
                }),
                F32TriadAvailability {
                    portable: None,
                    specialized: Some(qualified_module(
                        ModuleKind::TriadSm100,
                        "compute_100a",
                        "sm_100a",
                        (10, 0),
                        true,
                        131_328,
                    )),
                },
            ),
            (
                Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route {
                    tile: Tf32Sm120Tile::M64N128,
                    stages: Tf32Sm120Stages::S3,
                }),
                F32TriadAvailability {
                    portable: None,
                    specialized: Some(qualified_module(
                        ModuleKind::TriadSm120,
                        "compute_120",
                        "sm_120",
                        (12, 0),
                        true,
                        73_856,
                    )),
                },
            ),
        ];
        for (route, availability) in cases {
            assert_eq!(
                resolve_tf32_forced(request(ResolvedGemmOp::Nn), availability, route).unwrap(),
                route
            );
        }
    }

    #[test]
    fn portable_sm110_accepts_the_generic_target_transaction() {
        let route = Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        let availability = F32TriadAvailability {
            portable: Some(qualified_module(
                ModuleKind::TriadSm80,
                "sm_110",
                "sm_110",
                (11, 0),
                false,
                29_696,
            )),
            specialized: None,
        };

        assert_eq!(
            resolve_tf32_forced(request(ResolvedGemmOp::Nn), availability, route).unwrap(),
            route
        );
    }

    #[test]
    fn portable_sm101a_accepts_the_exact_target_transaction() {
        let route = Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        let availability = F32TriadAvailability {
            portable: Some(qualified_module(
                ModuleKind::TriadSm80,
                "sm_101a",
                "sm_101a",
                (10, 1),
                false,
                29_696,
            )),
            specialized: None,
        };

        assert_eq!(
            resolve_tf32_forced(request(ResolvedGemmOp::Nn), availability, route).unwrap(),
            route
        );
    }

    #[test]
    fn nt_splitk8_is_forced_only_and_does_not_change_auto_selection() {
        let selected = normalized_request(ResolvedGemmOp::Nt, 64, 384, 1_536);
        assert_eq!(
            resolve_tf32_forced(selected, sm89_availability(), TF32_NT_SPLITK8_S3_SPEC.route,)
                .unwrap(),
            TF32_NT_SPLITK8_S3_SPEC.route
        );
        assert_eq!(
            resolve_f32_triad_auto(
                F32TriadPolicy::AllowDeterministicTf32V1,
                selected,
                sm89_availability(),
            )
            .unwrap(),
            F32TriadSelection::ScalarFmaV1
        );
    }

    #[test]
    fn specialized_sm110_accepts_only_feature_target_transactions() {
        let route = Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(Tf32Sm100Route {
            tile: Sm100Tile::M128N64,
            stages: Sm100Stages::S2,
            schedule: Sm100Schedule::C4,
        });
        for (compiler_target, device_target) in
            [("compute_110f", "sm_110f"), ("compute_110a", "sm_110a")]
        {
            let availability = F32TriadAvailability {
                portable: None,
                specialized: Some(qualified_module(
                    ModuleKind::TriadSm100,
                    compiler_target,
                    device_target,
                    (11, 0),
                    true,
                    49_408,
                )),
            };
            assert_eq!(
                resolve_tf32_forced(request(ResolvedGemmOp::Nn), availability, route).unwrap(),
                route
            );
        }

        let generic = F32TriadAvailability {
            portable: None,
            specialized: Some(qualified_module(
                ModuleKind::TriadSm100,
                "sm_110",
                "sm_110",
                (11, 0),
                true,
                49_408,
            )),
        };
        assert!(resolve_tf32_forced(request(ResolvedGemmOp::Nn), generic, route).is_err());
    }

    #[test]
    fn forced_resolution_rejects_incoherent_or_unavailable_bindings() {
        let route = Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(Tf32Sm100Route {
            tile: Sm100Tile::M128N128,
            stages: Sm100Stages::S4,
            schedule: Sm100Schedule::C4,
        });
        let valid = qualified_module(
            ModuleKind::TriadSm100,
            "compute_100a",
            "sm_100a",
            (10, 0),
            true,
            131_328,
        );
        assert!(
            resolve_tf32_forced(
                request(ResolvedGemmOp::Nn),
                F32TriadAvailability::default(),
                route
            )
            .is_err()
        );
        for invalid in [
            Tf32QualifiedModule {
                module_kind: ModuleKind::TriadSm90a,
                ..valid
            },
            Tf32QualifiedModule {
                target: CudaTarget::new("compute_103a").unwrap(),
                ..valid
            },
            Tf32QualifiedModule {
                device: DeviceIdentity {
                    compute_capability: (10, 3),
                    ..valid.device
                },
                ..valid
            },
            Tf32QualifiedModule {
                device_caps: DeviceCaps {
                    tensor_map_access: false,
                    ..valid.device_caps
                },
                ..valid
            },
            Tf32QualifiedModule {
                device_caps: DeviceCaps {
                    optin_shared_bytes: 131_327,
                    ..valid.device_caps
                },
                ..valid
            },
            Tf32QualifiedModule {
                compiler: CompilerIdentity {
                    numeric_abi_revision: 0,
                    ..valid.compiler
                },
                ..valid
            },
        ] {
            assert!(
                resolve_tf32_forced(
                    request(ResolvedGemmOp::Nn),
                    F32TriadAvailability {
                        portable: None,
                        specialized: Some(invalid),
                    },
                    route,
                )
                .is_err(),
                "accepted {invalid:?}"
            );
        }

        let illegal = Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S2,
        });
        assert!(
            resolve_tf32_forced(
                request(ResolvedGemmOp::Nn),
                F32TriadAvailability {
                    portable: Some(qualified_module(
                        ModuleKind::TriadSm80,
                        "sm_89",
                        "sm_89",
                        (8, 9),
                        false,
                        99_000,
                    )),
                    specialized: None,
                },
                illegal,
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod sm100_toolkit_tests {
    use super::super::contract::Sm100TargetKind;
    use super::sm100_target_candidates_for_nvrtc;

    #[test]
    fn older_toolkits_are_offered_only_the_targets_they_can_name() {
        assert!(sm100_target_candidates_for_nvrtc((10, 0), (12, 8)).is_empty());

        let on_12_9 = sm100_target_candidates_for_nvrtc((10, 0), (12, 9));
        assert_eq!(on_12_9.len(), 2);
        assert_eq!(on_12_9[0].nvrtc_arch, "compute_100f");
        assert_eq!(on_12_9[0].kind, Sm100TargetKind::Family);
        assert_eq!(on_12_9[1].kind, Sm100TargetKind::Exact);

        assert!(sm100_target_candidates_for_nvrtc((10, 3), (12, 8)).is_empty());
        assert_eq!(sm100_target_candidates_for_nvrtc((10, 3), (12, 9)).len(), 2);
        assert!(sm100_target_candidates_for_nvrtc((11, 0), (13, 0)).is_empty());
        assert_eq!(sm100_target_candidates_for_nvrtc((11, 0), (13, 2)).len(), 2);
        assert!(sm100_target_candidates_for_nvrtc((12, 0), (13, 2)).is_empty());
    }
}

#[cfg(test)]
mod sm100_sm90a_auto_tests {
    use super::super::contract::{
        Sm90aLaunchOperands, Sm90aShape, Sm100LaunchOperands, Sm100PhysicalRoute, Sm100Schedule,
        Sm100Shape, Sm100Stages, Sm100Tile,
    };
    use super::*;

    fn sm100_request(op: Sm100Op, dims: (usize, usize, usize)) -> Sm100AutoRequest {
        let beta = if op == Sm100Op::Tn { 1.0 } else { 0.0 };
        Sm100AutoRequest {
            op,
            dtype: WeightDtype::Bf16,
            shape: Sm100Shape::contiguous(op, dims),
            a_ptr: 0x1_0000,
            b_ptr: 0x2_0000,
            operands: Sm100LaunchOperands {
                output_ptr: 0x3_0000,
                bias_ptr: 0,
                alpha: 1.0,
                beta,
            },
        }
    }

    fn sm100_cell(op: Sm100Op, dims: (usize, usize, usize)) -> Sm100ForcedRoute {
        Sm100ForcedRoute {
            op,
            dtype: WeightDtype::Bf16,
            physical: Sm100PhysicalRoute {
                tile: Sm100Tile::M128N64,
                stages: Sm100Stages::S2,
                schedule: Sm100Schedule::C4,
            },
            shape: Sm100Shape::contiguous(op, dims),
        }
    }

    #[test]
    fn sm100_tables_are_empty_and_the_wave_rule_serves_the_family() {
        let request = sm100_request(Sm100Op::Nn, (2048, 768, 3072));
        // The measured tables stay empty until a board qualifies its cells,
        // and the wave rule serves the family's boards in the meantime: the
        // output fills a device with wide tiles and the reduction feeds three
        // stages but not the larger schedule.
        let expected = Sm100PhysicalRoute {
            tile: Sm100Tile::M128N128,
            stages: Sm100Stages::S3,
            schedule: Sm100Schedule::C4,
        };
        for device_cc in [(10, 0), (10, 3), (11, 0)] {
            assert!(sm100_auto_cells(device_cc).is_empty());
            let target = sm100_target_candidates(device_cc).first().copied();
            let resolved = resolve_sm100_auto(device_cc, target, request)
                .unwrap_or_else(|| panic!("wave rule must serve {device_cc:?}"));
            assert_eq!(resolved.physical, expected);
            assert_eq!(resolved.shape, request.shape);
        }
        // Off the family nothing resolves, wave rule or not.
        for device_cc in [(12, 0), (9, 0)] {
            assert!(sm100_auto_cells(device_cc).is_empty());
            let target = sm100_target_candidates(device_cc).first().copied();
            assert_eq!(resolve_sm100_auto(device_cc, target, request), None);
        }
    }

    #[test]
    fn a_measured_sm100_cell_wins_on_its_board_and_the_rule_takes_the_rest() {
        let dims = (2048, 768, 3072);
        let cells = [sm100_cell(Sm100Op::Nn, dims)];
        let target = sm100_target_candidates((10, 0))[0];
        let request = sm100_request(Sm100Op::Nn, dims);
        assert_eq!(
            resolve_sm100_auto_from_cells(&cells, (10, 0), Some(target), request),
            Some(cells[0])
        );
        // No bound module, another board, a shape one row off, or an
        // epilogue outside the measured law each decline.
        assert_eq!(
            resolve_sm100_auto_from_cells(&cells, (10, 0), None, request),
            None
        );
        assert_eq!(
            resolve_sm100_auto_from_cells(&cells, (10, 3), Some(target), request),
            None
        );
        // A shape one row off the cell is not a decline any more: it takes
        // the wave rule instead of the measured route.
        let mut off = request;
        off.shape.m += 1;
        let drifted = resolve_sm100_auto_from_cells(&cells, (10, 0), Some(target), off)
            .expect("the wave rule serves the drifted shape");
        assert_ne!(drifted, cells[0]);
        assert_eq!(drifted.shape, off.shape);
        let mut biased = request;
        biased.operands.bias_ptr = 0x4_0000;
        biased.operands.alpha = 2.0;
        assert_eq!(
            resolve_sm100_auto_from_cells(&cells, (10, 0), Some(target), biased),
            None
        );
    }

    fn sm90a_request(op: Sm90aOp, dims: (usize, usize, usize)) -> Sm90aAutoRequest {
        let beta = if op == Sm90aOp::Tn { 1.0 } else { 0.0 };
        Sm90aAutoRequest {
            op,
            dtype: WeightDtype::Bf16,
            shape: Sm90aShape::contiguous(op, dims),
            a_ptr: 0x1_0000,
            b_ptr: 0x2_0000,
            operands: Sm90aLaunchOperands {
                output_ptr: 0x3_0000,
                bias_ptr: 0,
                alpha: 1.0,
                beta,
            },
        }
    }

    #[test]
    fn the_sm90a_wave_rule_serves_hopper_and_measured_cells_stay_scoped() {
        let dims = (2048, 768, 3072);
        let request = sm90a_request(Sm90aOp::Nn, dims);
        assert!(SM90A_AUTO_CELLS.is_empty());
        // With the table empty the wave rule serves Hopper: this reduction is
        // too shallow to feed a dedicated producer warpgroup.
        assert_eq!(
            resolve_sm90a_auto((9, 0), true, request).map(|route| route.schedule),
            Some(Sm90aWarpgroupSchedule::Wg1)
        );
        let mut deep = sm90a_request(Sm90aOp::Nn, (2048, 2048, 3072));
        deep.operands.beta = 0.0;
        assert_eq!(
            resolve_sm90a_auto((9, 0), true, deep).map(|route| route.schedule),
            Some(Sm90aWarpgroupSchedule::Wg2)
        );
        assert_eq!(resolve_sm90a_auto((12, 0), true, request), None);
        let cell = Sm90aForcedRoute {
            op: Sm90aOp::Nn,
            dtype: WeightDtype::Bf16,
            schedule: Sm90aWarpgroupSchedule::Wg1,
            shape: Sm90aShape::contiguous(Sm90aOp::Nn, dims),
        };
        assert_eq!(
            resolve_sm90a_auto_from_cells(&[cell], (9, 0), true, request),
            Some(cell)
        );
        assert_eq!(
            resolve_sm90a_auto_from_cells(&[cell], (9, 0), false, request),
            None
        );
        assert_eq!(
            resolve_sm90a_auto_from_cells(&[cell], (12, 0), true, request),
            None
        );
        let mut biased = request;
        biased.operands.bias_ptr = 0x4_0000;
        biased.operands.alpha = 2.0;
        assert_eq!(
            resolve_sm90a_auto_from_cells(&[cell], (9, 0), true, biased),
            None
        );
    }
}
