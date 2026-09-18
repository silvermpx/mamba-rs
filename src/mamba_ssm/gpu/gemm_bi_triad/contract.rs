use super::super::blas::TypedPtr;
use super::super::buffers::{ManagedAllocationEpochStamp, managed_allocation_epoch_for_ranges};
use super::super::dtype::WeightDtype;
use crate::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, CompilerIdentity, CudaTarget, DeviceCaps, DeviceIdentity, FramedSha256,
    ModuleKind, PhysicalGemmBackend, PolicyDtype, ResolvedGemmLaunchSet, ResolvedGemmOp,
    ResolvedGemmRoute, ResolvedInstructionFamily, ResolvedInstructionShape, ResolvedKernelLaunch,
    ResolvedNumericContract, ResolvedOperandConversion, ResolvedOutputOwnership, SCHEDULE_REVISION,
    Sha256Digest, TUNING_TABLE_REVISION,
};
use cudarc::driver::{CudaContext, DeviceRepr, sys};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
static ALLOCATION_IDENTITY_QUERY_COUNT: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
pub(super) fn reset_allocation_identity_query_count() {
    ALLOCATION_IDENTITY_QUERY_COUNT.store(0, Ordering::Release);
}

#[cfg(test)]
pub(super) fn allocation_identity_query_count() -> u64 {
    ALLOCATION_IDENTITY_QUERY_COUNT.load(Ordering::Acquire)
}

pub type CUptr = cudarc::driver::sys::CUdeviceptr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct AllocationDomain {
    pub(super) context_handle: usize,
    pub(super) device_ordinal: i32,
}

impl AllocationDomain {
    pub(super) fn from_context(context: &CudaContext) -> Result<Self, String> {
        let context_handle = context.cu_ctx() as usize;
        if context_handle == 0 {
            return Err("CUDA allocation domain requires a non-null context".into());
        }
        let device_ordinal = i32::try_from(context.ordinal())
            .map_err(|_| "CUDA device ordinal exceeds i32::MAX".to_string())?;
        Ok(Self {
            context_handle,
            device_ordinal,
        })
    }
}

fn validate_allocation_domain(
    expected: AllocationDomain,
    associated_context: Option<usize>,
    actual_device_ordinal: i32,
    backend: &str,
    name: &str,
) -> Result<(), String> {
    if associated_context.is_some_and(|context| context != expected.context_handle) {
        return Err(format!(
            "{backend} {name} allocation belongs to a different CUDA context"
        ));
    }
    if actual_device_ordinal != expected.device_ordinal {
        return Err(format!(
            "{backend} {name} allocation belongs to CUDA device {actual_device_ordinal}, expected CUDA device {}",
            expected.device_ordinal
        ));
    }
    Ok(())
}

pub const F32_TF32_TUNING_REVISION: u16 = TUNING_TABLE_REVISION;
pub const TF32_TENSOR_MAP_REVISION: u16 = 1;
pub const TF32_SCHEDULE_REVISION: u16 = SCHEDULE_REVISION;
pub const TF32_PORTABLE_SCHEDULE_REVISION: u16 = SCHEDULE_REVISION;
pub const SM89_FINALIST_TUNING_REVISION: u16 = 4;
pub const ZERO_REDUCTION_MAP_REVISION: u16 = 1;
pub const ZERO_REDUCTION_DIGEST_DOMAIN: &[u8] = b"tf32-zero-reduction-maps.v1";
pub const SCALAR_BIG_NT_DYNAMIC_SHARED_BYTES: u32 = 33_376;
pub const SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES: u32 = 17_408;
pub const SCALAR_NN_M32N64_SPLITK32_THREADS: u32 = 128;
pub const SCALAR_NN_M32N64_SPLITK32_DYNAMIC_SHARED_BYTES: u32 = 0;
pub const SCALAR_NN_M32N64_SPLITK32_STATIC_SHARED_BYTES: usize = 13_312;
pub const SCALAR_NN_M32N64_SPLITK32_REGISTER_CAP: i32 = 64;
pub const SCALAR_NN_M32N64_SPLITK32_MIN_ACTIVE_BLOCKS: u32 = 4;
pub const SCALAR_NT_M2N16_THREADS: u32 = 64;
pub const SCALAR_NT_M2N16_DYNAMIC_SHARED_BYTES: u32 = 17_984;
pub const SCALAR_NT_M2N16_STATIC_SHARED_BYTES: usize = 0;
pub const SCALAR_NT_M2N16_REGISTER_CAP: i32 = 112;
pub const SCALAR_NT_M2N16_MIN_ACTIVE_BLOCKS: u32 = 4;
pub const SCALAR_TN_M16N16_THREADS: u32 = 64;
pub const SCALAR_TN_M16N16_DYNAMIC_SHARED_BYTES: u32 = 4_096;
pub const SCALAR_TN_M16N16_STATIC_SHARED_BYTES: usize = 0;
pub const SCALAR_TN_M16N16_REGISTER_CAP: i32 = 112;
pub const SCALAR_TN_M16N16_MIN_ACTIVE_BLOCKS: u32 = 8;
pub const SCALAR_GENERIC_TRANSPOSE_ROUTE_CAP_ELEMENTS: usize = 1 << 22;
// Sized for the deepest transposed operand a qualified route stages: the
// 4096-row product's 4096 x 3072 weight-gradient input. The buffer is
// allocated on first use, so contexts that never transpose pay nothing.
pub const SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS: usize = 12_582_912;
pub const SCALAR_NT_D768_TRANSPOSE_THREADS: u32 = 512;
pub const SCALAR_NT_D768_TRANSPOSE_STATIC_SHARED_BYTES: usize = 4_224;
pub const SCALAR_NT_D768_TRANSPOSE_DYNAMIC_SHARED_BYTES: usize = 0;
pub const SCALAR_NT_D768_TRANSPOSE_REGISTER_CAP: i32 = 28;
pub const SCALAR_NT_D768_TRANSPOSE_MIN_ACTIVE_BLOCKS: u32 = 2;

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
#[repr(u8)]
pub enum Tf32TensorMapFormat {
    Tfloat32 = 1,
    Uint32 = 2,
    /// Exact F32 words, no swizzle, dense boxes as the exact routes read them.
    Uint32Dense = 3,
    /// Exact F32 words behind the 64-byte swizzle of the NT B stage.
    Uint32Swizzle64 = 4,
}

impl Tf32TensorMapFormat {
    const fn is_exact_fma(self) -> bool {
        matches!(self, Self::Uint32Dense | Self::Uint32Swizzle64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tf32TensorMapKey {
    base: CUptr,
    global_dimensions: [u64; 2],
    outer_byte_stride: u64,
    box_dimensions: [u32; 2],
    format: Tf32TensorMapFormat,
}

impl Tf32TensorMapKey {
    fn validate(self) -> Result<(), String> {
        if self.base == 0 || !self.base.is_multiple_of(16) {
            return Err("TF32 tensor-map base must be non-null and 16-byte aligned".into());
        }
        if self.global_dimensions.contains(&0) {
            return Err("TF32 tensor-map dimensions must be positive".into());
        }
        if self.outer_byte_stride == 0
            || !self.outer_byte_stride.is_multiple_of(16)
            || self.outer_byte_stride >= (1_u64 << 40)
        {
            return Err(
                "TF32 tensor-map outer byte stride must be a positive multiple of 16 below 2^40"
                    .into(),
            );
        }
        let row_bytes = self.global_dimensions[0]
            .checked_mul(4)
            .ok_or_else(|| "TF32 tensor-map row width overflows u64".to_string())?;
        if self.outer_byte_stride < row_bytes {
            return Err("TF32 tensor-map outer stride is smaller than its inner dimension".into());
        }
        if self.format.is_exact_fma() {
            let inner_bytes = self.box_dimensions[0] * 4;
            let swizzle_span = match self.format {
                Tf32TensorMapFormat::Uint32Swizzle64 => 64,
                _ => 256 * 4,
            };
            if self.box_dimensions[0] == 0
                || self.box_dimensions[1] == 0
                || self.box_dimensions[1] > 256
                || !inner_bytes.is_multiple_of(16)
                || inner_bytes > swizzle_span
            {
                return Err(
                    "exact-F32 box dimensions must be [16-byte multiples within the swizzle span, 1..=256]"
                        .into(),
                );
            }
            return Ok(());
        }
        if self.box_dimensions[0] != 32
            || self.box_dimensions[1] == 0
            || self.box_dimensions[1] > 256
        {
            return Err("TF32 SW128 box dimensions must be [32, 1..=256]".into());
        }
        Ok(())
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tf32TensorMap(sys::CUtensorMap);

unsafe impl DeviceRepr for Tf32TensorMap {}

const _: () = {
    assert!(std::mem::size_of::<Tf32TensorMap>() == std::mem::size_of::<sys::CUtensorMap>());
    assert!(std::mem::align_of::<Tf32TensorMap>() == std::mem::align_of::<sys::CUtensorMap>());
};

impl Tf32TensorMap {
    fn encode(key: Tf32TensorMapKey) -> Result<Self, String> {
        key.validate()?;
        let data_type = match key.format {
            Tf32TensorMapFormat::Tfloat32 => {
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_TFLOAT32
            }
            Tf32TensorMapFormat::Uint32
            | Tf32TensorMapFormat::Uint32Dense
            | Tf32TensorMapFormat::Uint32Swizzle64 => {
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32
            }
        };
        // The exact routes read dense stages and promote L2 lines, the TF32
        // routes keep their frozen 128-byte swizzle and promotion.
        let (swizzle, promotion) = match key.format {
            Tf32TensorMapFormat::Uint32Dense => (
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            ),
            Tf32TensorMapFormat::Uint32Swizzle64 => (
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            ),
            Tf32TensorMapFormat::Tfloat32 | Tf32TensorMapFormat::Uint32 => (
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            ),
        };
        let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
        let element_strides = [1_u32, 1_u32];
        let global_strides = [key.outer_byte_stride];
        unsafe {
            sys::cuTensorMapEncodeTiled(
                raw.as_mut_ptr(),
                data_type,
                2,
                key.base as usize as *mut std::ffi::c_void,
                key.global_dimensions.as_ptr(),
                global_strides.as_ptr(),
                key.box_dimensions.as_ptr(),
                element_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle,
                promotion,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
            .result()
            .map_err(|error| format!("TF32 cuTensorMapEncodeTiled failed: {error:?}"))?;
            Ok(Self(raw.assume_init()))
        }
    }

    fn is_zero(self) -> bool {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&self.0).cast::<u8>(),
                std::mem::size_of::<sys::CUtensorMap>(),
            )
        }
        .iter()
        .all(|&byte| byte == 0)
    }
}

pub fn zeroed_tensor_map_sentinel() -> Tf32TensorMap {
    let raw = unsafe { std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed().assume_init() };
    Tf32TensorMap(raw)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tf32TensorOrigins {
    pub a_x: i32,
    pub a_y: i32,
    pub b_x: i32,
    pub b_y: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tf32MapBinding {
    pub(super) allocation_domain: AllocationDomain,
    pub qualified: Tf32QualifiedModule,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct F32EncodedTensorMaps {
    a: Tf32TensorMap,
    b: Tf32TensorMap,
    keys: [Tf32TensorMapKey; 2],
    request: F32TriadRequest,
    route: Tf32PhysicalRoute,
    binding: Tf32MapBinding,
    allocations: [Sm90aAllocationIdentity; 2],
    origins: Tf32TensorOrigins,
    format: Tf32TensorMapFormat,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct F32ZeroReductionTensorMaps {
    a: Tf32TensorMap,
    b: Tf32TensorMap,
    request: F32TriadRequest,
    route: Option<Tf32PhysicalRoute>,
    binding: Option<Tf32MapBinding>,
    format: Tf32TensorMapFormat,
    revision: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum F32PreparedTensorMaps {
    Encoded {
        data: Box<F32EncodedTensorMaps>,
    },
    ZeroReduction {
        data: Box<F32ZeroReductionTensorMaps>,
    },
}

impl F32PreparedTensorMaps {
    pub(super) fn zero_reduction(
        request: F32TriadRequest,
        route: Option<Tf32PhysicalRoute>,
        binding: Option<Tf32MapBinding>,
        format: Tf32TensorMapFormat,
    ) -> Self {
        Self::ZeroReduction {
            data: Box::new(F32ZeroReductionTensorMaps {
                a: zeroed_tensor_map_sentinel(),
                b: zeroed_tensor_map_sentinel(),
                request,
                route,
                binding,
                format,
                revision: ZERO_REDUCTION_MAP_REVISION,
            }),
        }
    }

    pub fn maps(&self) -> [Tf32TensorMap; 2] {
        match self {
            Self::Encoded { data } => [data.a, data.b],
            Self::ZeroReduction { data } => [data.a, data.b],
        }
    }

    pub fn origins(&self) -> Tf32TensorOrigins {
        match self {
            Self::Encoded { data } => data.origins,
            Self::ZeroReduction { .. } => Tf32TensorOrigins::default(),
        }
    }

    pub fn request(&self) -> F32TriadRequest {
        match self {
            Self::Encoded { data } => data.request,
            Self::ZeroReduction { data } => data.request,
        }
    }

    pub fn binding(&self) -> Option<Tf32MapBinding> {
        match self {
            Self::Encoded { data } => Some(data.binding),
            Self::ZeroReduction { data } => data.binding,
        }
    }

    fn identity_digest_header(&self, domain: &[u8]) -> FramedSha256 {
        let request = self.request();
        FramedSha256::new(domain)
            .required(b"op", &[self.request().op as u8])
            .required(b"m", &(request.shape.m as u64).to_le_bytes())
            .required(b"k", &(request.shape.k as u64).to_le_bytes())
            .required(b"n", &(request.shape.n as u64).to_le_bytes())
            .required(b"lda", &(request.shape.lda as u64).to_le_bytes())
            .required(b"ldb", &(request.shape.ldb as u64).to_le_bytes())
            .required(b"ldc", &(request.shape.ldc as u64).to_le_bytes())
            .required(
                b"output-rows",
                &(request.shape.output_rows(request.op) as u64).to_le_bytes(),
            )
            .required(
                b"output-columns",
                &(request.shape.output_columns(request.op) as u64).to_le_bytes(),
            )
            .required(
                b"reduction",
                &(request.shape.reduction(request.op) as u64).to_le_bytes(),
            )
    }

    pub fn identity_digest(&self) -> Sha256Digest {
        let digest = self.identity_digest_header(b"tf32-tensor-map-pair.v1");
        match self {
            Self::Encoded { data } => append_tf32_encoded_map_identity(
                digest,
                Tf32EncodedMapIdentity {
                    revision: TF32_TENSOR_MAP_REVISION,
                    maps: [data.a, data.b],
                    keys: data.keys,
                    allocations: data.allocations,
                    origins: data.origins,
                    format: data.format,
                    route: data.route,
                },
            )
            .finish(),
            Self::ZeroReduction { data } => {
                let digest = digest
                    .required(b"mode", ZERO_REDUCTION_DIGEST_DOMAIN)
                    .required(b"revision", &data.revision.to_le_bytes())
                    .required(b"format", &[data.format as u8]);
                let digest = match data.route {
                    Some(route) => append_tf32_route_digest(digest, route),
                    None => digest.required(b"executor-family", b"scalar-fma-v1"),
                };
                digest
                    .required(
                        b"tensor-map-size",
                        &(std::mem::size_of::<sys::CUtensorMap>() as u64).to_le_bytes(),
                    )
                    .required(
                        b"tensor-map-align",
                        &(std::mem::align_of::<sys::CUtensorMap>() as u64).to_le_bytes(),
                    )
                    .finish()
            }
        }
    }

    pub(super) fn physical_identity_digest(&self) -> Sha256Digest {
        match self {
            Self::Encoded { data } => append_tf32_encoded_map_physical_identity(
                self.identity_digest_header(b"tf32-tensor-map-pair-physical.v1"),
                Tf32EncodedMapIdentity {
                    revision: TF32_TENSOR_MAP_REVISION,
                    maps: [data.a, data.b],
                    keys: data.keys,
                    allocations: data.allocations,
                    origins: data.origins,
                    format: data.format,
                    route: data.route,
                },
            )
            .finish(),
            Self::ZeroReduction { .. } => self.identity_digest(),
        }
    }

    pub fn matches_binding(&self, binding: Tf32MapBinding) -> bool {
        self.binding() == Some(binding)
    }

    pub fn validate_live_allocations(&self) -> Result<(), String> {
        match self {
            Self::Encoded { data } => {
                for (index, allocation) in data.allocations.into_iter().enumerate() {
                    let name = if index == 0 { "A" } else { "B" };
                    if allocation.requery("TF32", name)? != allocation {
                        return Err(
                            "TF32 input allocation identity changed since tensor-map preparation"
                                .into(),
                        );
                    }
                }
                Ok(())
            }
            Self::ZeroReduction { data } => {
                if data.revision != ZERO_REDUCTION_MAP_REVISION
                    || data.request.shape.reduction(data.request.op) != 0
                    || !data.a.is_zero()
                    || !data.b.is_zero()
                {
                    return Err("TF32 zero-reduction tensor-map sentinel changed".into());
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Tf32EncodedMapIdentity {
    revision: u16,
    maps: [Tf32TensorMap; 2],
    keys: [Tf32TensorMapKey; 2],
    allocations: [Sm90aAllocationIdentity; 2],
    origins: Tf32TensorOrigins,
    format: Tf32TensorMapFormat,
    route: Tf32PhysicalRoute,
}

fn append_tf32_encoded_map_identity(
    digest: FramedSha256,
    identity: Tf32EncodedMapIdentity,
) -> FramedSha256 {
    let mut digest = append_tf32_route_digest(
        digest
            .required(b"mode", b"encoded-v1")
            .required(b"tensor-map-revision", &identity.revision.to_le_bytes())
            .required(b"format", &[identity.format as u8])
            .required(b"a-x", &identity.origins.a_x.to_le_bytes())
            .required(b"a-y", &identity.origins.a_y.to_le_bytes())
            .required(b"b-x", &identity.origins.b_x.to_le_bytes())
            .required(b"b-y", &identity.origins.b_y.to_le_bytes())
            .required(
                b"tensor-map-size",
                &(std::mem::size_of::<sys::CUtensorMap>() as u64).to_le_bytes(),
            )
            .required(
                b"tensor-map-align",
                &(std::mem::align_of::<sys::CUtensorMap>() as u64).to_le_bytes(),
            ),
        identity.route,
    );
    for index in 0..2 {
        let key = identity.keys[index];
        let map = identity.maps[index];
        let map_bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&map.0).cast::<u8>(),
                std::mem::size_of::<sys::CUtensorMap>(),
            )
        };
        digest = digest
            .required(b"map-index", &(index as u64).to_le_bytes())
            .required(b"key-index", &(index as u64).to_le_bytes())
            .required(b"key-format", &[key.format as u8])
            .required(b"base", &key.base.to_le_bytes())
            .required(b"global-0", &key.global_dimensions[0].to_le_bytes())
            .required(b"global-1", &key.global_dimensions[1].to_le_bytes())
            .required(b"outer-stride", &key.outer_byte_stride.to_le_bytes())
            .required(b"box-0", &key.box_dimensions[0].to_le_bytes())
            .required(b"box-1", &key.box_dimensions[1].to_le_bytes())
            .required(b"allocation-index", &(index as u64).to_le_bytes());
        digest = identity.allocations[index]
            .append_digest(digest)
            .required(b"descriptor-index", &(index as u64).to_le_bytes())
            .required(b"encoded-descriptor", map_bytes);
    }
    digest
}

fn append_tf32_encoded_map_physical_identity(
    digest: FramedSha256,
    identity: Tf32EncodedMapIdentity,
) -> FramedSha256 {
    let mut digest = append_tf32_route_digest(
        digest
            .required(b"mode", b"encoded-physical-v1")
            .required(b"tensor-map-revision", &identity.revision.to_le_bytes())
            .required(b"format", &[identity.format as u8])
            .required(b"a-x", &identity.origins.a_x.to_le_bytes())
            .required(b"a-y", &identity.origins.a_y.to_le_bytes())
            .required(b"b-x", &identity.origins.b_x.to_le_bytes())
            .required(b"b-y", &identity.origins.b_y.to_le_bytes())
            .required(
                b"tensor-map-size",
                &(std::mem::size_of::<sys::CUtensorMap>() as u64).to_le_bytes(),
            )
            .required(
                b"tensor-map-align",
                &(std::mem::align_of::<sys::CUtensorMap>() as u64).to_le_bytes(),
            ),
        identity.route,
    );
    for index in 0..2 {
        let key = identity.keys[index];
        digest = digest
            .required(b"map-index", &(index as u64).to_le_bytes())
            .required(b"key-index", &(index as u64).to_le_bytes())
            .required(b"key-format", &[key.format as u8])
            .required(b"global-0", &key.global_dimensions[0].to_le_bytes())
            .required(b"global-1", &key.global_dimensions[1].to_le_bytes())
            .required(b"outer-stride", &key.outer_byte_stride.to_le_bytes())
            .required(b"box-0", &key.box_dimensions[0].to_le_bytes())
            .required(b"box-1", &key.box_dimensions[1].to_le_bytes())
            .required(b"allocation-index", &(index as u64).to_le_bytes())
            .required(
                b"physical-allocation",
                &identity.allocations[index].physical_digest(),
            );
    }
    digest
}

pub(super) fn append_tf32_route_digest(
    digest: FramedSha256,
    route: Tf32PhysicalRoute,
) -> FramedSha256 {
    match route {
        Tf32PhysicalRoute::MmaTf32Rna(route) => {
            let tile = match route.tile {
                Tf32PortableTile::M128N64 => 1,
                Tf32PortableTile::M64N64 => 2,
                Tf32PortableTile::M16N32 => 3,
                Tf32PortableTile::M16N16 => 4,
                Tf32PortableTile::M32N32 => 5,
                Tf32PortableTile::M128N128 => 6,
            };
            digest
                .required(b"route-family", &[1])
                .required(b"tile", &[tile])
                .required(b"stages", &[route.stages.count()])
        }
        Tf32PhysicalRoute::MmaTf32RnaSplitK4(route) => {
            let tile = match route.tile {
                Tf32PortableTile::M128N64 => 1,
                Tf32PortableTile::M64N64 => 2,
                Tf32PortableTile::M16N32 => 3,
                Tf32PortableTile::M16N16 => 4,
                Tf32PortableTile::M32N32 => 5,
                Tf32PortableTile::M128N128 => 6,
            };
            digest
                .required(b"route-family", &[5])
                .required(b"tile", &[tile])
                .required(b"stages", &[route.stages.count()])
                .required(b"partitions", &[4])
        }
        Tf32PhysicalRoute::MmaTf32RnaSplitK2(route) => {
            let tile = match route.tile {
                Tf32PortableTile::M128N64 => 1,
                Tf32PortableTile::M64N64 => 2,
                Tf32PortableTile::M16N32 => 3,
                Tf32PortableTile::M16N16 => 4,
                Tf32PortableTile::M32N32 => 5,
                Tf32PortableTile::M128N128 => 6,
            };
            digest
                .required(b"route-family", &[6])
                .required(b"tile", &[tile])
                .required(b"stages", &[route.stages.count()])
                .required(b"partitions", &[2])
        }
        Tf32PhysicalRoute::MmaTf32RnaSplitK8(route) => {
            let tile = match route.tile {
                Tf32PortableTile::M128N64 => 1,
                Tf32PortableTile::M64N64 => 2,
                Tf32PortableTile::M16N32 => 3,
                Tf32PortableTile::M16N16 => 4,
                Tf32PortableTile::M32N32 => 5,
                Tf32PortableTile::M128N128 => 6,
            };
            digest
                .required(b"route-family", &[7])
                .required(b"tile", &[tile])
                .required(b"stages", &[route.stages.count()])
                .required(b"partitions", &[8])
        }
        Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(route) => digest
            .required(b"route-family", &[2])
            .required(b"schedule-threads", &route.schedule.threads().to_le_bytes()),
        Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(route) => digest
            .required(b"route-family", &[3])
            .required(b"tile-columns", &route.tile.output_columns().to_le_bytes())
            .required(b"stages", &[route.stages.count()])
            .required(b"schedule-threads", &route.schedule.threads().to_le_bytes()),
        Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(route) => {
            let tile = match route.tile {
                Tf32Sm120Tile::M128N64 => 1,
                Tf32Sm120Tile::M64N128 => 2,
                Tf32Sm120Tile::M64N64 => 3,
                Tf32Sm120Tile::M80N32Bk64 => 4,
            };
            digest
                .required(b"route-family", &[4])
                .required(b"tile", &[tile])
                .required(b"stages", &[route.stages.count()])
        }
        Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(route) => {
            let tile = match route.tile {
                Tf32Sm120Tile::M128N64 => 1,
                Tf32Sm120Tile::M64N128 => 2,
                Tf32Sm120Tile::M64N64 => 3,
                Tf32Sm120Tile::M80N32Bk64 => 4,
            };
            digest
                .required(b"route-family", &[8])
                .required(b"tile", &[tile])
                .required(b"stages", &[route.stages.count()])
        }
        Tf32PhysicalRoute::Sm120TmaFmaExact(route) => digest
            .required(b"route-family", &[9])
            .required(b"tile", &[route.tile.digest_code()])
            .required(b"kvec", &[u8::from(route.kvec)])
            .required(b"splits", &[route.splits]),
        Tf32PhysicalRoute::Sm89MmaTf32Compact8 => digest.required(b"route-family", &[10]),
        Tf32PhysicalRoute::Sm89TnPreRnaN96 => digest.required(b"route-family", &[11]),
        Tf32PhysicalRoute::Sm89TnPreRnaM64N64 => digest.required(b"route-family", &[12]),
        Tf32PhysicalRoute::Sm89NnDirectN96 => digest.required(b"route-family", &[13]),
        Tf32PhysicalRoute::Sm89NnN96 => digest.required(b"route-family", &[14]),
        Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2 => digest.required(b"route-family", &[15]),
        Tf32PhysicalRoute::Sm89NtALdmatrixN96 => digest.required(b"route-family", &[16]),
        Tf32PhysicalRoute::Sm89NtRnaM144N96S2 => digest.required(b"route-family", &[17]),
        Tf32PhysicalRoute::Sm89NtRowstageM128N192S2 => digest.required(b"route-family", &[18]),
        Tf32PhysicalRoute::Sm89TnDirectM192N192S2 => digest.required(b"route-family", &[19]),
        Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2 => digest.required(b"route-family", &[20]),
        Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3 => digest.required(b"route-family", &[21]),
    }
}

pub(super) struct Tf32TensorMapPlan {
    pub keys: [Tf32TensorMapKey; 2],
    pub allocations: [Sm90aAllocationIdentity; 2],
    pub origins: Tf32TensorOrigins,
    pub format: Tf32TensorMapFormat,
}

#[derive(Clone, Copy)]
struct Tf32OperandLayout {
    pointer: CUptr,
    stride: usize,
    width: usize,
    rows: usize,
    issued_coordinate_max: [usize; 2],
    box_dimensions: [u32; 2],
    name: &'static str,
}

fn tf32_last_tile_start(extent: usize, tile: usize, name: &str) -> Result<usize, String> {
    if tile == 0 {
        return Err(format!("TF32 {name} tile extent must be positive"));
    }
    let last = extent
        .checked_sub(1)
        .ok_or_else(|| format!("TF32 {name} logical extent must be positive"))?;
    (last / tile)
        .checked_mul(tile)
        .ok_or_else(|| format!("TF32 {name} last tile start overflows usize"))
}

fn tf32_tail_plane_start(
    extent: usize,
    tile: usize,
    plane: usize,
    name: &str,
) -> Result<usize, String> {
    let tail_offset = tile
        .checked_sub(plane)
        .ok_or_else(|| format!("TF32 {name} tile is smaller than its issue plane"))?;
    tf32_last_tile_start(extent, tile, name)?
        .checked_add(tail_offset)
        .ok_or_else(|| format!("TF32 {name} tail-plane start overflows usize"))
}

fn tf32_operand_layouts(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: Tf32PhysicalRoute,
) -> Result<[Tf32OperandLayout; 2], String> {
    request.shape.validate(request.op)?;
    if request.shape.reduction(request.op) == 0 {
        return Err("zero-reduction TF32 routes use the mapless sentinel".into());
    }
    if matches!(route, Tf32PhysicalRoute::MmaTf32Rna(_)) {
        return Err("portable TF32 does not use tensor maps".into());
    }
    if route.is_exact_fma() {
        return sm120_fma_operand_layouts(request, operands, route);
    }
    let spec = tf32_kernel_spec(request.op, route)?;
    if spec.map_bk != 32 || !spec.bk.is_multiple_of(spec.map_bk) {
        return Err(format!(
            "unsupported TF32 logical/map BK={}/{}",
            spec.bk, spec.map_bk
        ));
    }
    let shape = request.shape;
    let tile_rows =
        usize::try_from(spec.tile.0).map_err(|_| "TF32 tile rows exceed usize::MAX".to_string())?;
    let tile_columns = usize::try_from(spec.tile.1)
        .map_err(|_| "TF32 tile columns exceed usize::MAX".to_string())?;
    let reduction_tile = usize::try_from(spec.map_bk)
        .map_err(|_| "TF32 reduction tile exceeds usize::MAX".to_string())?;
    let layouts = match request.op {
        ResolvedGemmOp::Nn => [
            Tf32OperandLayout {
                pointer: operands.a,
                stride: shape.lda,
                width: shape.k,
                rows: shape.m,
                issued_coordinate_max: [
                    tf32_last_tile_start(shape.k, reduction_tile, "NN A reduction")?,
                    tf32_last_tile_start(shape.m, tile_rows, "NN A rows")?,
                ],
                box_dimensions: [spec.map_bk, spec.tile.0],
                name: "A",
            },
            Tf32OperandLayout {
                pointer: operands.b,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.k,
                issued_coordinate_max: [
                    tf32_tail_plane_start(shape.n, tile_columns, reduction_tile, "NN B columns")?,
                    tf32_last_tile_start(shape.k, reduction_tile, "NN B reduction")?,
                ],
                box_dimensions: [spec.map_bk, spec.map_bk],
                name: "B",
            },
        ],
        ResolvedGemmOp::Tn => [
            Tf32OperandLayout {
                pointer: operands.a,
                stride: shape.lda,
                width: shape.k,
                rows: shape.m,
                issued_coordinate_max: [
                    tf32_tail_plane_start(shape.k, tile_rows, reduction_tile, "TN A columns")?,
                    tf32_last_tile_start(shape.m, reduction_tile, "TN A reduction")?,
                ],
                box_dimensions: [spec.map_bk, spec.map_bk],
                name: "A",
            },
            Tf32OperandLayout {
                pointer: operands.b,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.m,
                issued_coordinate_max: [
                    tf32_tail_plane_start(shape.n, tile_columns, reduction_tile, "TN B columns")?,
                    tf32_last_tile_start(shape.m, reduction_tile, "TN B reduction")?,
                ],
                box_dimensions: [spec.map_bk, spec.map_bk],
                name: "B",
            },
        ],
        ResolvedGemmOp::Nt => [
            Tf32OperandLayout {
                pointer: operands.a,
                stride: shape.lda,
                width: shape.n,
                rows: shape.m,
                issued_coordinate_max: [
                    tf32_last_tile_start(shape.n, reduction_tile, "NT A reduction")?,
                    tf32_last_tile_start(shape.m, tile_rows, "NT A rows")?,
                ],
                box_dimensions: [spec.map_bk, spec.tile.0],
                name: "A",
            },
            Tf32OperandLayout {
                pointer: operands.b,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.k,
                issued_coordinate_max: [
                    tf32_last_tile_start(shape.n, reduction_tile, "NT B reduction")?,
                    tf32_last_tile_start(shape.k, tile_columns, "NT B columns")?,
                ],
                box_dimensions: [spec.map_bk, spec.tile.1],
                name: "B",
            },
        ],
    };
    Ok(layouts)
}

fn validate_tf32_issued_coordinates(
    layout: Tf32OperandLayout,
    origin: (u64, u64),
) -> Result<(), String> {
    for (axis, origin, issued) in [
        ("x", origin.0, layout.issued_coordinate_max[0]),
        ("y", origin.1, layout.issued_coordinate_max[1]),
    ] {
        let issued = u64::try_from(issued).map_err(|_| {
            format!(
                "TF32 {} issued {axis} coordinate exceeds u64::MAX",
                layout.name
            )
        })?;
        let coordinate = origin.checked_add(issued).ok_or_else(|| {
            format!(
                "TF32 {} issued {axis} coordinate overflows u64",
                layout.name
            )
        })?;
        i32::try_from(coordinate).map_err(|_| {
            format!(
                "TF32 {} issued {axis} coordinate exceeds i32::MAX after applying the subview origin",
                layout.name
            )
        })?;
    }
    Ok(())
}

/// Operand layouts of the exact-F32 routes. Kernel-side terms: C[M][N] over
/// the reduction K; NN reads A [M][K] and B [K][N], TN reads A [K][M] and B
/// [K][N], NT reads A [M][K] and B [N][K]. Boxes are one 16-wide k tile by a
/// block tile edge, so every coordinate the kernel issues is a multiple of a
/// box edge.
fn sm120_fma_operand_layouts(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: Tf32PhysicalRoute,
) -> Result<[Tf32OperandLayout; 2], String> {
    let spec = tf32_kernel_spec(request.op, route)?;
    let shape = request.shape;
    let (bm, bn) = spec.tile;
    let bk = SM120_FMA_BK;
    let rows_out = shape.output_rows(request.op);
    let columns_out = shape.output_columns(request.op);
    let reduction = shape.reduction(request.op);
    let last_k = tf32_last_tile_start(reduction, bk as usize, "exact-F32 reduction")?;
    let last_rows = tf32_last_tile_start(rows_out, bm as usize, "exact-F32 rows")?;
    let last_columns = tf32_last_tile_start(columns_out, bn as usize, "exact-F32 columns")?;
    let (a, b) = match request.op {
        ResolvedGemmOp::Nn => (
            Tf32OperandLayout {
                pointer: operands.a,
                stride: shape.lda,
                width: shape.k,
                rows: shape.m,
                issued_coordinate_max: [last_k, last_rows],
                box_dimensions: [bk, bm],
                name: "A",
            },
            Tf32OperandLayout {
                pointer: operands.b,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.k,
                issued_coordinate_max: [last_columns, last_k],
                box_dimensions: [bn, bk],
                name: "B",
            },
        ),
        ResolvedGemmOp::Tn => (
            Tf32OperandLayout {
                pointer: operands.a,
                stride: shape.lda,
                width: shape.k,
                rows: shape.m,
                issued_coordinate_max: [last_rows, last_k],
                box_dimensions: [bm, bk],
                name: "A",
            },
            Tf32OperandLayout {
                pointer: operands.b,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.m,
                issued_coordinate_max: [last_columns, last_k],
                box_dimensions: [bn, bk],
                name: "B",
            },
        ),
        ResolvedGemmOp::Nt => (
            Tf32OperandLayout {
                pointer: operands.a,
                stride: shape.lda,
                width: shape.n,
                rows: shape.m,
                issued_coordinate_max: [last_k, last_rows],
                box_dimensions: [bk, bm],
                name: "A",
            },
            Tf32OperandLayout {
                pointer: operands.b,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.k,
                issued_coordinate_max: [last_k, last_columns],
                box_dimensions: [bk, bn],
                name: "B",
            },
        ),
    };
    Ok([a, b])
}

/// The exact-F32 map of one operand: the pointer is the map base, the map
/// spans exactly the operand, and the allocation identity is kept for the
/// live check before every launch.
fn sm120_fma_map_plan(
    layout: Tf32OperandLayout,
    allocation_domain: AllocationDomain,
    format: Tf32TensorMapFormat,
) -> Result<(Tf32TensorMapKey, Sm90aAllocationIdentity), String> {
    if !layout.pointer.is_multiple_of(16) {
        return Err(format!(
            "exact-F32 {} TMA pointer must be 16-byte aligned",
            layout.name
        ));
    }
    let outer_byte_stride = u64::try_from(
        layout
            .stride
            .checked_mul(4)
            .ok_or_else(|| format!("exact-F32 {} byte stride overflows usize", layout.name))?,
    )
    .map_err(|_| format!("exact-F32 {} byte stride exceeds u64::MAX", layout.name))?;
    let width = u64::try_from(layout.width)
        .map_err(|_| format!("exact-F32 {} width exceeds u64::MAX", layout.name))?;
    let rows = u64::try_from(layout.rows)
        .map_err(|_| format!("exact-F32 {} rows exceed u64::MAX", layout.name))?;
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| format!("exact-F32 {} row bytes overflow u64", layout.name))?;
    let required_bytes = rows
        .checked_sub(1)
        .and_then(|prefix_rows| prefix_rows.checked_mul(outer_byte_stride))
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or_else(|| format!("exact-F32 {} allocation span overflows u64", layout.name))?;
    let allocation = Sm90aAllocationIdentity::query(
        layout.pointer,
        required_bytes,
        allocation_domain,
        "exact-F32",
        layout.name,
    )?;
    validate_tf32_issued_coordinates(layout, (0, 0))?;
    let key = Tf32TensorMapKey {
        base: layout.pointer,
        global_dimensions: [width, rows],
        outer_byte_stride,
        box_dimensions: layout.box_dimensions,
        format,
    };
    Ok((key, allocation))
}

fn tf32_subview_plan(
    layout: Tf32OperandLayout,
    allocation_domain: AllocationDomain,
    format: Tf32TensorMapFormat,
) -> Result<(Tf32TensorMapKey, Sm90aAllocationIdentity, (i32, i32)), String> {
    if !layout.pointer.is_multiple_of(16) {
        return Err(format!(
            "TF32 {} TMA pointer must be 16-byte aligned",
            layout.name
        ));
    }
    let outer_byte_stride = u64::try_from(
        layout
            .stride
            .checked_mul(4)
            .ok_or_else(|| format!("TF32 {} byte stride overflows usize", layout.name))?,
    )
    .map_err(|_| format!("TF32 {} byte stride exceeds u64::MAX", layout.name))?;
    let width = u64::try_from(layout.width)
        .map_err(|_| format!("TF32 {} width exceeds u64::MAX", layout.name))?;
    let rows = u64::try_from(layout.rows)
        .map_err(|_| format!("TF32 {} rows exceed u64::MAX", layout.name))?;
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| format!("TF32 {} row bytes overflow u64", layout.name))?;
    let required_bytes = rows
        .checked_sub(1)
        .and_then(|prefix_rows| prefix_rows.checked_mul(outer_byte_stride))
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or_else(|| format!("TF32 {} allocation span overflows u64", layout.name))?;
    let logical = Sm90aAllocationIdentity::query(
        layout.pointer,
        required_bytes,
        allocation_domain,
        "TF32",
        layout.name,
    )?;
    if !logical.offset_bytes.is_multiple_of(4) {
        return Err(format!(
            "TF32 {} subview offset must be element aligned",
            layout.name
        ));
    }
    let origin_y = logical.offset_bytes / outer_byte_stride;
    let origin_x = (logical.offset_bytes % outer_byte_stride) / 4;
    let stride_elements = outer_byte_stride / 4;
    if origin_x
        .checked_add(width)
        .is_none_or(|end| end > stride_elements)
    {
        return Err(format!(
            "TF32 {} subview crosses its physical row",
            layout.name
        ));
    }
    let global_dimensions = [
        origin_x
            .checked_add(width)
            .ok_or_else(|| format!("TF32 {} inner dimension overflows u64", layout.name))?,
        origin_y
            .checked_add(rows)
            .ok_or_else(|| format!("TF32 {} outer dimension overflows u64", layout.name))?,
    ];
    validate_tf32_issued_coordinates(layout, (origin_x, origin_y))?;
    let key = Tf32TensorMapKey {
        base: logical.allocation_base,
        global_dimensions,
        outer_byte_stride,
        box_dimensions: layout.box_dimensions,
        format,
    };
    key.validate()?;
    Ok((
        key,
        logical,
        (
            i32::try_from(origin_x)
                .map_err(|_| format!("TF32 {} x origin exceeds i32::MAX", layout.name))?,
            i32::try_from(origin_y)
                .map_err(|_| format!("TF32 {} y origin exceeds i32::MAX", layout.name))?,
        ),
    ))
}

pub(super) fn tf32_tensor_map_plan(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: Tf32PhysicalRoute,
    allocation_domain: AllocationDomain,
) -> Result<Tf32TensorMapPlan, String> {
    let format = match route {
        Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(_) | Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(_) => {
            Tf32TensorMapFormat::Tfloat32
        }
        Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(_)
        | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_) => Tf32TensorMapFormat::Uint32,
        Tf32PhysicalRoute::Sm120TmaFmaExact(_) => Tf32TensorMapFormat::Uint32Dense,
        Tf32PhysicalRoute::MmaTf32Rna(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_)
        | Tf32PhysicalRoute::Sm89MmaTf32Compact8
        | Tf32PhysicalRoute::Sm89TnPreRnaN96
        | Tf32PhysicalRoute::Sm89TnPreRnaM64N64
        | Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2
        | Tf32PhysicalRoute::Sm89NnDirectN96
        | Tf32PhysicalRoute::Sm89NnN96
        | Tf32PhysicalRoute::Sm89NtALdmatrixN96
        | Tf32PhysicalRoute::Sm89NtRnaM144N96S2
        | Tf32PhysicalRoute::Sm89NtRowstageM128N192S2
        | Tf32PhysicalRoute::Sm89TnDirectM192N192S2
        | Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2
        | Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3 => {
            return Err("portable TF32 does not use tensor maps".into());
        }
    };
    let layouts = tf32_operand_layouts(request, operands, route)?;
    if route.is_exact_fma() {
        // The exact kernels take no subview origin: the map base is the
        // operand pointer itself and every coordinate starts at zero.
        let b_format = if request.op == ResolvedGemmOp::Nt {
            Tf32TensorMapFormat::Uint32Swizzle64
        } else {
            format
        };
        let (a_key, a_allocation) = sm120_fma_map_plan(layouts[0], allocation_domain, format)?;
        let (b_key, b_allocation) = sm120_fma_map_plan(layouts[1], allocation_domain, b_format)?;
        let keys = [a_key, b_key];
        for key in keys {
            key.validate()?;
        }
        return Ok(Tf32TensorMapPlan {
            allocations: [a_allocation, b_allocation],
            keys,
            origins: Tf32TensorOrigins::default(),
            format,
        });
    }
    let (a_key, a_allocation, (a_x, a_y)) =
        tf32_subview_plan(layouts[0], allocation_domain, format)?;
    let (b_key, b_allocation, (b_x, b_y)) =
        tf32_subview_plan(layouts[1], allocation_domain, format)?;
    let keys = [a_key, b_key];
    for key in keys {
        key.validate()?;
    }
    Ok(Tf32TensorMapPlan {
        allocations: [a_allocation, b_allocation],
        keys,
        origins: Tf32TensorOrigins { a_x, a_y, b_x, b_y },
        format,
    })
}

pub(super) fn encode_tf32_tensor_maps(
    plan: Tf32TensorMapPlan,
    request: F32TriadRequest,
    route: Tf32PhysicalRoute,
    binding: Tf32MapBinding,
) -> Result<F32PreparedTensorMaps, String> {
    Ok(F32PreparedTensorMaps::Encoded {
        data: Box::new(F32EncodedTensorMaps {
            a: Tf32TensorMap::encode(plan.keys[0])?,
            b: Tf32TensorMap::encode(plan.keys[1])?,
            keys: plan.keys,
            request,
            route,
            binding,
            allocations: plan.allocations,
            origins: plan.origins,
            format: plan.format,
        }),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct F32LaunchResourceSnapshot {
    output: Sm90aAllocationIdentity,
    bias: Option<Sm90aAllocationIdentity>,
    inputs: Option<[Sm90aAllocationIdentity; 2]>,
    split_scratch: Option<Sm90aAllocationIdentity>,
    transpose_scratch: Option<Sm90aAllocationIdentity>,
    coordination_scratch: Option<Sm90aAllocationIdentity>,
}

impl F32LaunchResourceSnapshot {
    pub(super) fn physical_digest(self) -> Sha256Digest {
        let mut digest = FramedSha256::new(b"f32-triad-physical-resources.v1")
            .required(b"output", &self.output.physical_digest());
        for (role, identity) in [
            (b"bias".as_slice(), self.bias),
            (b"A".as_slice(), self.inputs.map(|inputs| inputs[0])),
            (b"B".as_slice(), self.inputs.map(|inputs| inputs[1])),
            (b"split-scratch".as_slice(), self.split_scratch),
            (b"transpose-scratch".as_slice(), self.transpose_scratch),
            (
                b"coordination-scratch".as_slice(),
                self.coordination_scratch,
            ),
        ] {
            digest = match identity {
                Some(identity) => digest.optional(role, Some(&identity.physical_digest())),
                None => digest.optional(role, None),
            };
        }
        digest.finish()
    }

    pub(super) fn managed_epoch(self) -> Option<ManagedAllocationEpochStamp> {
        let mut ranges = [(0_u64, 0_u64); 4];
        let mut count = 0;
        for identity in [
            Some(self.output),
            self.bias,
            self.inputs.map(|inputs| inputs[0]),
            self.inputs.map(|inputs| inputs[1]),
        ]
        .into_iter()
        .flatten()
        {
            if identity.allocation_domain != self.output.allocation_domain {
                return None;
            }
            ranges[count] = (
                identity
                    .allocation_base
                    .checked_add(identity.offset_bytes)?,
                identity.required_bytes,
            );
            count += 1;
        }
        managed_allocation_epoch_for_ranges(
            self.output.allocation_domain.context_handle,
            &ranges[..count],
        )
    }

    pub(super) fn query_output(
        request: F32TriadRequest,
        operands: F32TriadOperands,
        allocation_domain: AllocationDomain,
    ) -> Result<Self, String> {
        let rows = request.shape.output_rows(request.op);
        let columns = request.shape.output_columns(request.op);
        let output_bytes =
            matrix_span_bytes(rows, columns, request.shape.ldc, 4, "f32 Triad output")?;
        let output = Sm90aAllocationIdentity::query(
            operands.output,
            output_bytes,
            allocation_domain,
            "f32 Triad",
            "output",
        )?;
        let bias = match operands.bias {
            Some(pointer) => {
                let bytes = u64::try_from(columns)
                    .ok()
                    .and_then(|columns| columns.checked_mul(4))
                    .ok_or_else(|| "f32 Triad bias allocation span overflows u64".to_string())?;
                Some(Sm90aAllocationIdentity::query(
                    pointer,
                    bytes,
                    allocation_domain,
                    "f32 Triad",
                    "bias",
                )?)
            }
            None => None,
        };
        if bias.is_some_and(|bias| output.requested_range_overlaps(bias)) {
            return Err("f32 Triad output overlaps the requested bias range".into());
        }
        Ok(Self {
            output,
            bias,
            inputs: None,
            split_scratch: None,
            transpose_scratch: None,
            coordination_scratch: None,
        })
    }

    pub(super) fn with_inputs(
        mut self,
        request: F32TriadRequest,
        operands: F32TriadOperands,
        allocation_domain: AllocationDomain,
    ) -> Result<Self, String> {
        let shape = request.shape;
        let ((a_rows, a_columns), (b_rows, b_columns)) = match request.op {
            ResolvedGemmOp::Nn => ((shape.m, shape.k), (shape.k, shape.n)),
            ResolvedGemmOp::Tn => ((shape.m, shape.k), (shape.m, shape.n)),
            ResolvedGemmOp::Nt => ((shape.m, shape.n), (shape.k, shape.n)),
        };
        let a_bytes = matrix_span_bytes(a_rows, a_columns, shape.lda, 4, "f32 Triad A")?;
        let b_bytes = matrix_span_bytes(b_rows, b_columns, shape.ldb, 4, "f32 Triad B")?;
        let inputs = [
            Sm90aAllocationIdentity::query(
                operands.a,
                a_bytes,
                allocation_domain,
                "f32 Triad",
                "A",
            )?,
            Sm90aAllocationIdentity::query(
                operands.b,
                b_bytes,
                allocation_domain,
                "f32 Triad",
                "B",
            )?,
        ];
        for (input, name) in [(inputs[0], "A"), (inputs[1], "B")] {
            if self.output.requested_range_overlaps(input) {
                return Err(format!(
                    "f32 Triad output overlaps the requested {name} input range"
                ));
            }
        }
        self.inputs = Some(inputs);
        Ok(self)
    }

    pub(super) fn with_scratch(
        mut self,
        split: Option<(CUptr, u64)>,
        transpose: Option<(CUptr, u64)>,
        coordination: Option<(CUptr, u64)>,
        allocation_domain: AllocationDomain,
    ) -> Result<Self, String> {
        self.split_scratch = split
            .map(|(pointer, bytes)| {
                Sm90aAllocationIdentity::query(
                    pointer,
                    bytes,
                    allocation_domain,
                    "f32 Triad",
                    "split scratch",
                )
            })
            .transpose()?;
        self.transpose_scratch = transpose
            .map(|(pointer, bytes)| {
                Sm90aAllocationIdentity::query(
                    pointer,
                    bytes,
                    allocation_domain,
                    "f32 Triad",
                    "transpose scratch",
                )
            })
            .transpose()?;
        self.coordination_scratch = coordination
            .map(|(pointer, bytes)| {
                Sm90aAllocationIdentity::query(
                    pointer,
                    bytes,
                    allocation_domain,
                    "f32 Triad",
                    "coordination scratch",
                )
            })
            .transpose()?;
        Ok(self)
    }

    pub(super) fn validate_live(self) -> Result<(), String> {
        for (identity, name) in [
            (Some(self.output), "output"),
            (self.bias, "bias"),
            (self.inputs.map(|inputs| inputs[0]), "A"),
            (self.inputs.map(|inputs| inputs[1]), "B"),
            (self.split_scratch, "split scratch"),
            (self.transpose_scratch, "transpose scratch"),
            (self.coordination_scratch, "coordination scratch"),
        ] {
            if let Some(identity) = identity
                && identity.requery("f32 Triad", name)? != identity
            {
                return Err(format!(
                    "f32 Triad {name} allocation identity changed since preparation"
                ));
            }
        }
        Ok(())
    }

    pub(super) fn transpose_scratch_identity(self) -> Option<Sm90aAllocationIdentity> {
        self.transpose_scratch
    }

    pub(super) fn digest(
        self,
        request: F32TriadRequest,
        operands: F32TriadOperands,
        tensor_maps_digest: Sha256Digest,
    ) -> Sha256Digest {
        let bias_pointer = operands.bias.map(CUptr::to_le_bytes);
        let mut digest = self
            .output
            .append_digest(FramedSha256::new(b"f32-triad-launch-resources.v1"))
            .required(b"op", &[request.op as u8])
            .required(b"m", &(request.shape.m as u64).to_le_bytes())
            .required(b"k", &(request.shape.k as u64).to_le_bytes())
            .required(b"n", &(request.shape.n as u64).to_le_bytes())
            .required(b"lda", &(request.shape.lda as u64).to_le_bytes())
            .required(b"ldb", &(request.shape.ldb as u64).to_le_bytes())
            .required(b"ldc", &(request.shape.ldc as u64).to_le_bytes())
            .required(b"tensor-maps", &tensor_maps_digest)
            .required(b"output-pointer", &operands.output.to_le_bytes())
            .optional(
                b"bias-pointer",
                bias_pointer.as_ref().map(<[u8; 8]>::as_slice),
            )
            .required(b"alpha", &operands.alpha.to_bits().to_le_bytes())
            .required(b"beta", &operands.beta.to_bits().to_le_bytes());
        for identity in [
            self.bias,
            self.inputs.map(|inputs| inputs[0]),
            self.inputs.map(|inputs| inputs[1]),
            self.split_scratch,
            self.transpose_scratch,
            self.coordination_scratch,
        ]
        .into_iter()
        .flatten()
        {
            digest = identity.append_digest(digest);
        }
        digest.finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tf32PortableTile {
    M128N64,
    M64N64,
    M16N32,
    M16N16,
    M32N32,
    /// The wide eight-warp tile of the portable extension fragment; NN only,
    /// absent from the CC 12.x composition of the portable module.
    M128N128,
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
                | (Tf32PortableTile::M16N16, Tf32PortableStages::S4)
                | (Tf32PortableTile::M128N128, Tf32PortableStages::S3)
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
    M64N64,
    M80N32Bk64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tf32Sm120Stages {
    S2,
    S3,
    S4,
}

impl Tf32Sm120Stages {
    pub const fn count(self) -> u8 {
        match self {
            Self::S2 => 2,
            Self::S3 => 3,
            Self::S4 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tf32Sm120Route {
    pub tile: Tf32Sm120Tile,
    pub stages: Tf32Sm120Stages,
}

/// The reduction tile of the exact-F32 SM120 routes.
pub const SM120_FMA_BK: u32 = 16;
/// Register gate of the exact-F32 SM120 routes: the plain arms keep three
/// blocks per multiprocessor, the NT k-vector arms two.
pub const SM120_FMA_REGISTER_CAP: u32 = 168;
pub const SM120_FMA_KVEC_REGISTER_CAP: u32 = 255;

/// Block tiles of the exact-F32 SM120 routes: an 8x8 register microtile,
/// BK 16, two TMA stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm120FmaTile {
    M128N64,
    M64N128,
    M64N64,
}

impl Sm120FmaTile {
    pub const fn dims(self) -> (u32, u32) {
        match self {
            Self::M128N64 => (128, 64),
            Self::M64N128 => (64, 128),
            Self::M64N64 => (64, 64),
        }
    }

    pub const fn threads(self) -> u32 {
        match self {
            Self::M128N64 | Self::M64N128 => 128,
            Self::M64N64 => 64,
        }
    }

    /// Two dense stages plus one mbarrier per stage.
    pub const fn dynamic_shared_bytes(self) -> u32 {
        let (bm, bn) = self.dims();
        2 * (bm * SM120_FMA_BK + SM120_FMA_BK * bn) * 4 + 2 * 8
    }

    pub const fn min_blocks(self, kvec: bool) -> u32 {
        match (self, kvec) {
            (Self::M64N64, false) => 5,
            (Self::M64N64, true) => 4,
            (_, false) => 3,
            (_, true) => 2,
        }
    }

    const fn digest_code(self) -> u8 {
        match self {
            Self::M128N64 => 1,
            Self::M64N128 => 2,
            Self::M64N64 => 3,
        }
    }
}

/// One exact-F32 SM120 route: the block tile, whether an NT arm keeps its B
/// fragment as float4 along k, and how many reduction splits the launch
/// deals. One split reproduces the scalar chain bit for bit; more splits
/// fold fixed-order partials.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm120FmaRoute {
    pub tile: Sm120FmaTile,
    pub kvec: bool,
    pub splits: u8,
}

impl Sm120FmaRoute {
    pub fn validate(self, op: ResolvedGemmOp) -> Result<(), String> {
        if self.splits == 0 {
            return Err("exact-F32 SM120 route needs at least one split".into());
        }
        if self.kvec && op != ResolvedGemmOp::Nt {
            return Err("the k-vector B fragment only exists for NT".into());
        }
        Ok(())
    }

    /// The route with its split count normalized to one: the kernel is the
    /// same, only the launch deals differently.
    pub const fn spec_key(self) -> Self {
        Self {
            tile: self.tile,
            kvec: self.kvec,
            splits: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tf32PhysicalRoute {
    MmaTf32Rna(Tf32PortableRoute),
    MmaTf32RnaSplitK2(Tf32PortableRoute),
    MmaTf32RnaSplitK4(Tf32PortableRoute),
    MmaTf32RnaSplitK8(Tf32PortableRoute),
    Sm90aWgmmaTf32Tma(Tf32Sm90aRoute),
    Sm100Tcgen05Tf32Tma(Tf32Sm100Route),
    Sm120TmaMmaTf32Rna(Tf32Sm120Route),
    Sm120TmaMmaTf32RnaStreamKV1(Tf32Sm120Route),
    /// Exact F32 through the scalar FMA chain, fed by TMA; lives in the
    /// SM120 module next to the TF32 routes but never rounds an operand.
    Sm120TmaFmaExact(Sm120FmaRoute),
    /// The isolated Ada NT finalist. It retains the portable RNA/MMA
    /// numerical contract, but owns a distinct exact-SM89 module and epoch.
    Sm89MmaTf32Compact8,
    /// Ada TN after an explicit, separately observed RNA+transpose of A.
    Sm89TnPreRnaN96,
    /// Ada TN Prism after an explicit, separately observed RNA+transpose of A.
    Sm89TnPreRnaM64N64,
    /// Ada TN Prism M64xN96/BK32/S2 after an explicit RNA+transpose of A.
    Sm89TnPreRnaM64N96S2,
    /// Ada NN Prism with retained add-half conversion and direct full-tile stores.
    Sm89NnDirectN96,
    /// Ada NN d768-out with retained add-half conversion and shared epilogue.
    Sm89NnN96,
    /// Ada NT d768-in direct A-ldmatrix N96 winner.
    Sm89NtALdmatrixN96,
    /// Ada NT Prism 144x96 wide tile with RNA operand rounding.
    Sm89NtRnaM144N96S2,
    /// Ada NT deep-cell 128x192 wide tile with row-staged stores.
    Sm89NtRowstageM128N192S2,
    /// Ada TN deep-cell 192x192 wide tile reading the saved input directly.
    Sm89TnDirectM192N192S2,
    /// Ada TN d768-in 96x192 wide tile after the explicit RNA+transpose of A.
    Sm89TnPreRnaM96N192S2,
    /// Ada TN d768-out 96x96 three-stage tile after the explicit RNA+transpose of A.
    Sm89TnPreRnaM96N96S3,
}

impl Tf32PhysicalRoute {
    pub const fn module_kind(self) -> ModuleKind {
        match self {
            Self::MmaTf32Rna(_)
            | Self::MmaTf32RnaSplitK2(_)
            | Self::MmaTf32RnaSplitK4(_)
            | Self::MmaTf32RnaSplitK8(_) => ModuleKind::TriadSm80,
            Self::Sm90aWgmmaTf32Tma(_) => ModuleKind::TriadSm90a,
            Self::Sm100Tcgen05Tf32Tma(_) => ModuleKind::TriadSm100,
            Self::Sm120TmaMmaTf32Rna(_)
            | Self::Sm120TmaMmaTf32RnaStreamKV1(_)
            | Self::Sm120TmaFmaExact(_) => ModuleKind::TriadSm120,
            Self::Sm89MmaTf32Compact8 => ModuleKind::TriadSm89Finalist,
            Self::Sm89TnPreRnaN96
            | Self::Sm89TnPreRnaM64N64
            | Self::Sm89TnPreRnaM64N96S2
            | Self::Sm89NnDirectN96
            | Self::Sm89NnN96 => ModuleKind::TriadSm89Tf32Joint,
            Self::Sm89NtALdmatrixN96
            | Self::Sm89NtRnaM144N96S2
            | Self::Sm89NtRowstageM128N192S2
            | Self::Sm89TnDirectM192N192S2
            | Self::Sm89TnPreRnaM96N192S2
            | Self::Sm89TnPreRnaM96N96S3 => ModuleKind::TriadSm89Tf32Joint,
        }
    }

    /// True for the exact-F32 FMA routes, which the TF32 policy never
    /// selects and the exact policy owns.
    pub const fn is_exact_fma(self) -> bool {
        matches!(self, Self::Sm120TmaFmaExact(_))
    }

    /// The exact-F32 route payload, if this is one.
    pub const fn exact_fma(self) -> Option<Sm120FmaRoute> {
        match self {
            Self::Sm120TmaFmaExact(route) => Some(route),
            _ => None,
        }
    }

    /// The route as the kernel inventory names it: an exact-F32 route with
    /// its split count normalized to one.
    pub const fn spec_key(self) -> Self {
        match self {
            Self::Sm120TmaFmaExact(route) => Self::Sm120TmaFmaExact(route.spec_key()),
            other => other,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F32TriadSelection {
    ScalarFma,
    Tf32(Tf32PhysicalRoute),
    /// A route measured on another board, offered here with the portable
    /// route of the same numeric contract it must reproduce; the launcher
    /// admits it at first use only when every output word matches
    /// (`gemm_bi_triad::proof`).
    Tf32Proof {
        candidate: Tf32PhysicalRoute,
        reference: Tf32PhysicalRoute,
    },
    /// The exact policy's own SM120 route: scalar FMA numerics behind TMA,
    /// never a TF32 selection.
    ExactSm120Fma(Sm120FmaRoute),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sm120FmaExclusions(u16);

impl Sm120FmaExclusions {
    pub(crate) fn from_routes(routes: &[(ResolvedGemmOp, Sm120FmaRoute)]) -> Result<Self, String> {
        let mut bits = 0_u16;
        for &(op, route) in routes {
            let key = Tf32PhysicalRoute::Sm120TmaFmaExact(route.spec_key());
            let index = SM120_FMA_ROUTE_SPECS
                .iter()
                .position(|spec| spec.op == op && spec.route == key)
                .ok_or_else(|| format!("{op:?} exact-F32 route {route:?} is not inventoried"))?;
            bits |= 1_u16 << index;
        }
        Ok(Self(bits))
    }

    pub(crate) fn is_excluded(self, op: ResolvedGemmOp, route: Sm120FmaRoute) -> bool {
        let key = Tf32PhysicalRoute::Sm120TmaFmaExact(route.spec_key());
        SM120_FMA_ROUTE_SPECS
            .iter()
            .position(|spec| spec.op == op && spec.route == key)
            .is_some_and(|index| self.0 & (1_u16 << index) != 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tf32QualifiedModule {
    pub module_kind: ModuleKind,
    pub target: CudaTarget,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
    pub device_caps: DeviceCaps,
    /// Exact-F32 SM120 symbols rejected by this binding's Driver resource
    /// gates. An empty set preserves synthetic and non-SM120 bindings.
    pub sm120_fma_exclusions: Sm120FmaExclusions,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F32TriadAvailability {
    pub portable: Option<Tf32QualifiedModule>,
    pub specialized: Option<Tf32QualifiedModule>,
    pub finalist: Option<Tf32QualifiedModule>,
    pub joint: Option<Tf32QualifiedModule>,
    /// Multiprocessors of the board the modules were bound on; zero when
    /// unknown, which keeps the automatic selection to exact measured cells.
    pub multiprocessors: u32,
}

/// Kernel parameters of the exact-F32 SM120 routes: kernel-side (M, N, K),
/// the output stride, and the reduction split the launch deals.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm120FmaKernelParams {
    pub alpha: f32,
    pub beta: f32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub ldc: i32,
    pub splits: i32,
    pub tiles_per_split: i32,
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
    pub map_bk: u32,
    pub stages: u8,
    pub threads: u32,
    pub dynamic_shared_bytes: u32,
    pub tensor_map_revision: u16,
    pub schedule_revision: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tf32SplitKSpec {
    pub op: ResolvedGemmOp,
    pub route: Tf32PhysicalRoute,
    pub symbol: &'static str,
    pub tile: (u32, u32),
    pub bk: u32,
    pub stages: u8,
    pub partitions: u32,
    pub threads: u32,
    pub dynamic_shared_bytes: u32,
    pub register_cap: u32,
    pub occupancy_gate: u32,
}

pub const TF32_SPLITK2_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Nn,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK2(Tf32PortableRoute {
        tile: Tf32PortableTile::M16N32,
        stages: Tf32PortableStages::S4,
    }),
    symbol: "nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4",
    tile: (16, 32),
    bk: 32,
    stages: 4,
    partitions: 2,
    threads: 128,
    dynamic_shared_bytes: 29_696,
    register_cap: 128,
    occupancy_gate: 3,
};

pub const TF32_SPLITK4_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Nn,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
        tile: Tf32PortableTile::M16N32,
        stages: Tf32PortableStages::S4,
    }),
    symbol: "nn_sm80_mma_tf32_splitk4_m16n32_bk32_s4",
    tile: (16, 32),
    bk: 32,
    stages: 4,
    partitions: 4,
    threads: 128,
    dynamic_shared_bytes: 29_696,
    register_cap: 128,
    occupancy_gate: 3,
};

pub const TF32_NT_SPLITK4_S3_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Nt,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
        tile: Tf32PortableTile::M16N32,
        stages: Tf32PortableStages::S3,
    }),
    symbol: "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s3",
    tile: (16, 32),
    bk: 32,
    stages: 3,
    partitions: 4,
    threads: 128,
    dynamic_shared_bytes: 20_736,
    register_cap: 96,
    occupancy_gate: 3,
};

pub const TF32_NT_SPLITK4_S4_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Nt,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
        tile: Tf32PortableTile::M16N32,
        stages: Tf32PortableStages::S4,
    }),
    symbol: "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s4",
    tile: (16, 32),
    bk: 32,
    stages: 4,
    partitions: 4,
    threads: 128,
    dynamic_shared_bytes: 27_648,
    register_cap: 96,
    occupancy_gate: 3,
};

pub const TF32_NT_SPLITK8_S3_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Nt,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
        tile: Tf32PortableTile::M32N32,
        stages: Tf32PortableStages::S3,
    }),
    symbol: "nt_sm80_mma_tf32_splitk8_m32n32_bk32_s3",
    tile: (32, 32),
    bk: 32,
    stages: 3,
    partitions: 8,
    threads: 128,
    dynamic_shared_bytes: 27_648,
    register_cap: 96,
    occupancy_gate: 3,
};

pub const TF32_NT_SPLITK8_S4_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Nt,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
        tile: Tf32PortableTile::M32N32,
        stages: Tf32PortableStages::S4,
    }),
    symbol: "nt_sm80_mma_tf32_splitk8_m32n32_bk32_s4",
    tile: (32, 32),
    bk: 32,
    stages: 4,
    partitions: 8,
    threads: 128,
    dynamic_shared_bytes: 36_864,
    register_cap: 96,
    occupancy_gate: 2,
};

pub const TF32_SPLITK_CANDIDATE_SPECS: [Tf32SplitKSpec; 6] = [
    TF32_SPLITK2_SPEC,
    TF32_SPLITK4_SPEC,
    TF32_NT_SPLITK4_S3_SPEC,
    TF32_NT_SPLITK4_S4_SPEC,
    TF32_NT_SPLITK8_S3_SPEC,
    TF32_NT_SPLITK8_S4_SPEC,
];

pub const TF32_TN_SPLITK8_M64_S2_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Tn,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
        tile: Tf32PortableTile::M64N64,
        stages: Tf32PortableStages::S2,
    }),
    symbol: "tn_sm80_mma_tf32_splitk8_m64n64_bk32_s2",
    tile: (64, 64),
    bk: 32,
    stages: 2,
    partitions: 8,
    threads: 128,
    dynamic_shared_bytes: 36_864,
    register_cap: 128,
    occupancy_gate: 2,
};
pub const TF32_TN_SPLITK8_M64_S3_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Tn,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
        tile: Tf32PortableTile::M64N64,
        stages: Tf32PortableStages::S3,
    }),
    symbol: "tn_sm80_mma_tf32_splitk8_m64n64_bk32_s3",
    tile: (64, 64),
    bk: 32,
    stages: 3,
    partitions: 8,
    threads: 128,
    dynamic_shared_bytes: 55_296,
    register_cap: 128,
    occupancy_gate: 1,
};
pub const TF32_TN_SPLITK8_M32_S3_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Tn,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
        tile: Tf32PortableTile::M32N32,
        stages: Tf32PortableStages::S3,
    }),
    symbol: "tn_sm80_mma_tf32_splitk8_m32n32_bk32_s3",
    tile: (32, 32),
    bk: 32,
    stages: 3,
    partitions: 8,
    threads: 128,
    dynamic_shared_bytes: 30_720,
    register_cap: 96,
    occupancy_gate: 3,
};
pub const TF32_TN_SPLITK8_M32_S4_SPEC: Tf32SplitKSpec = Tf32SplitKSpec {
    op: ResolvedGemmOp::Tn,
    route: Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
        tile: Tf32PortableTile::M32N32,
        stages: Tf32PortableStages::S4,
    }),
    symbol: "tn_sm80_mma_tf32_splitk8_m32n32_bk32_s4",
    tile: (32, 32),
    bk: 32,
    stages: 4,
    partitions: 8,
    threads: 128,
    dynamic_shared_bytes: 40_960,
    register_cap: 96,
    occupancy_gate: 2,
};
/// The TN split-K candidates live in the extension fragment, composed for
/// the sm80-family targets only; the CC 12.x portable module never carries
/// them (see `sm80_target_composes_extensions`).
pub const TF32_SPLITK_EXTENSION_SPECS: [Tf32SplitKSpec; 4] = [
    TF32_TN_SPLITK8_M64_S2_SPEC,
    TF32_TN_SPLITK8_M64_S3_SPEC,
    TF32_TN_SPLITK8_M32_S3_SPEC,
    TF32_TN_SPLITK8_M32_S4_SPEC,
];

/// The split-K candidates the portable module carries when `extensions`
/// says its target composes the extension fragments.
pub fn tf32_splitk_specs_for(extensions: bool) -> impl Iterator<Item = &'static Tf32SplitKSpec> {
    TF32_SPLITK_CANDIDATE_SPECS.iter().chain(
        if extensions {
            &TF32_SPLITK_EXTENSION_SPECS[..]
        } else {
            &[]
        }
        .iter(),
    )
}

/// Every split-K candidate the portable module can carry on any target.
pub fn tf32_splitk_specs_all() -> impl Iterator<Item = &'static Tf32SplitKSpec> {
    tf32_splitk_specs_for(true)
}

pub fn tf32_splitk_spec(
    op: ResolvedGemmOp,
    route: Tf32PhysicalRoute,
) -> Result<&'static Tf32SplitKSpec, String> {
    tf32_splitk_specs_all()
        .find(|spec| spec.op == op && spec.route == route)
        .ok_or_else(|| format!("no TF32 split-K kernel matches {op:?}/{route:?}"))
}

pub fn tf32_splitk_partition_bounds(
    reduction: usize,
    partitions: u32,
    partition: u32,
) -> Result<(usize, usize), String> {
    const BK: usize = 32;
    if !matches!(partitions, 2 | 4 | 8) {
        return Err(format!(
            "TF32 split-K partition count {partitions} is unsupported"
        ));
    }
    if partition >= partitions {
        return Err(format!(
            "TF32 split-K partition {partition} is out of range for {partitions} partitions"
        ));
    }
    let tiles = reduction.div_ceil(BK);
    let partitions = partitions as usize;
    let partition = partition as usize;
    let tiles_per_partition = tiles.div_ceil(partitions);
    let begin_tile = tiles_per_partition
        .checked_mul(partition)
        .ok_or_else(|| "TF32 split-K begin tile overflows usize".to_string())?
        .min(tiles);
    let end_tile = tiles_per_partition
        .checked_mul(partition + 1)
        .ok_or_else(|| "TF32 split-K end tile overflows usize".to_string())?
        .min(tiles);
    Ok((
        begin_tile
            .checked_mul(BK)
            .ok_or_else(|| "TF32 split-K begin offset overflows usize".to_string())?
            .min(reduction),
        end_tile
            .checked_mul(BK)
            .ok_or_else(|| "TF32 split-K end offset overflows usize".to_string())?
            .min(reduction),
    ))
}

const fn portable_tf32_dynamic_shared_bytes(
    op: ResolvedGemmOp,
    tile: Tf32PortableTile,
    stages: Tf32PortableStages,
) -> u32 {
    let stage_bytes = match (op, tile) {
        (ResolvedGemmOp::Nn | ResolvedGemmOp::Nt, Tf32PortableTile::M128N64) => 27_648,
        (ResolvedGemmOp::Tn, Tf32PortableTile::M128N64) => 26_624,
        (_, Tf32PortableTile::M64N64) => 18_432,
        (ResolvedGemmOp::Nn, Tf32PortableTile::M16N32) => 7_424,
        (ResolvedGemmOp::Tn, Tf32PortableTile::M16N32) => 8_192,
        (ResolvedGemmOp::Nt, Tf32PortableTile::M16N32) => 6_912,
        (ResolvedGemmOp::Nn, Tf32PortableTile::M16N16) => 5_376,
        (ResolvedGemmOp::Tn, Tf32PortableTile::M16N16) => 6_144,
        (ResolvedGemmOp::Nt, Tf32PortableTile::M16N16) => 4_608,
        (ResolvedGemmOp::Tn, Tf32PortableTile::M32N32) => 10_240,
        (_, Tf32PortableTile::M32N32) => 9_216,
        // Unpadded XOR-swizzled stages: 128 x 32 + 32 x 128 floats.
        (_, Tf32PortableTile::M128N128) => 32_768,
    };
    stage_bytes * stages.count() as u32
}

macro_rules! portable_tf32_specs {
    ($op:expr, $op_name:literal) => {
        [
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N64,
                    stages: Tf32PortableStages::S2,
                }),
                symbol: concat!($op_name, "_sm80_mma_tf32_m128n64_bk32_s2"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32,
                tile: (128, 64),
                bk: 32,
                map_bk: 32,
                stages: 2,
                threads: 256,
                dynamic_shared_bytes: portable_tf32_dynamic_shared_bytes(
                    $op,
                    Tf32PortableTile::M128N64,
                    Tf32PortableStages::S2,
                ),
                tensor_map_revision: 0,
                schedule_revision: TF32_PORTABLE_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M128N64,
                    stages: Tf32PortableStages::S3,
                }),
                symbol: concat!($op_name, "_sm80_mma_tf32_m128n64_bk32_s3"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32,
                tile: (128, 64),
                bk: 32,
                map_bk: 32,
                stages: 3,
                threads: 256,
                dynamic_shared_bytes: portable_tf32_dynamic_shared_bytes(
                    $op,
                    Tf32PortableTile::M128N64,
                    Tf32PortableStages::S3,
                ),
                tensor_map_revision: 0,
                schedule_revision: TF32_PORTABLE_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M64N64,
                    stages: Tf32PortableStages::S2,
                }),
                symbol: concat!($op_name, "_sm80_mma_tf32_m64n64_bk32_s2"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32,
                tile: (64, 64),
                bk: 32,
                map_bk: 32,
                stages: 2,
                threads: 128,
                dynamic_shared_bytes: portable_tf32_dynamic_shared_bytes(
                    $op,
                    Tf32PortableTile::M64N64,
                    Tf32PortableStages::S2,
                ),
                tensor_map_revision: 0,
                schedule_revision: TF32_PORTABLE_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M64N64,
                    stages: Tf32PortableStages::S3,
                }),
                symbol: concat!($op_name, "_sm80_mma_tf32_m64n64_bk32_s3"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32,
                tile: (64, 64),
                bk: 32,
                map_bk: 32,
                stages: 3,
                threads: 128,
                dynamic_shared_bytes: portable_tf32_dynamic_shared_bytes(
                    $op,
                    Tf32PortableTile::M64N64,
                    Tf32PortableStages::S3,
                ),
                tensor_map_revision: 0,
                schedule_revision: TF32_PORTABLE_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
                symbol: concat!($op_name, "_sm80_mma_tf32_m16n32_bk32_s4"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32,
                tile: (16, 32),
                bk: 32,
                map_bk: 32,
                stages: 4,
                threads: 128,
                dynamic_shared_bytes: portable_tf32_dynamic_shared_bytes(
                    $op,
                    Tf32PortableTile::M16N32,
                    Tf32PortableStages::S4,
                ),
                tensor_map_revision: 0,
                schedule_revision: TF32_PORTABLE_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N16,
                    stages: Tf32PortableStages::S4,
                }),
                symbol: concat!($op_name, "_sm80_mma_tf32_m16n16_bk32_s4"),
                module_kind: ModuleKind::TriadSm80,
                instruction_family: ResolvedInstructionFamily::MmaSync,
                instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
                operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32,
                tile: (16, 16),
                bk: 32,
                map_bk: 32,
                stages: 4,
                threads: 64,
                dynamic_shared_bytes: portable_tf32_dynamic_shared_bytes(
                    $op,
                    Tf32PortableTile::M16N16,
                    Tf32PortableStages::S4,
                ),
                tensor_map_revision: 0,
                schedule_revision: TF32_PORTABLE_SCHEDULE_REVISION,
            },
        ]
    };
}

macro_rules! sm90a_tf32_specs {
    ($op:expr, $op_name:literal) => {
        [
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(Tf32Sm90aRoute {
                    schedule: Sm90aWarpgroupSchedule::Wg1,
                }),
                symbol: concat!($op_name, "_sm90a_wgmma_tf32_m64n128_bk32_s3_wg1"),
                module_kind: ModuleKind::TriadSm90a,
                instruction_family: ResolvedInstructionFamily::Wgmma,
                instruction_shape: ResolvedInstructionShape {
                    m: 64,
                    n: 128,
                    k: 8,
                },
                operand_conversion: ResolvedOperandConversion::TensorMapTfloat32,
                tile: (64, 128),
                bk: 32,
                map_bk: 32,
                stages: 3,
                threads: 128,
                dynamic_shared_bytes: 73_984,
                tensor_map_revision: TF32_TENSOR_MAP_REVISION,
                schedule_revision: TF32_SCHEDULE_REVISION,
            },
            Tf32KernelSpec {
                op: $op,
                route: Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(Tf32Sm90aRoute {
                    schedule: Sm90aWarpgroupSchedule::Wg2,
                }),
                symbol: concat!($op_name, "_sm90a_wgmma_tf32_m64n128_bk32_s3_wg2"),
                module_kind: ModuleKind::TriadSm90a,
                instruction_family: ResolvedInstructionFamily::Wgmma,
                instruction_shape: ResolvedInstructionShape {
                    m: 64,
                    n: 128,
                    k: 8,
                },
                operand_conversion: ResolvedOperandConversion::TensorMapTfloat32,
                tile: (64, 128),
                bk: 32,
                map_bk: 32,
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
            route: Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(Tf32Sm100Route {
                tile: $tile,
                stages: $stage,
                schedule: $schedule,
            }),
            symbol: concat!(
                $op_name,
                "_sm100_tcgen_tf32_m128n",
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
            operand_conversion: ResolvedOperandConversion::TensorMapTfloat32,
            tile: (128, $n),
            bk: 32,
            map_bk: 32,
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
            route: Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
                tile: $tile,
                stages: $stage,
            }),
            symbol: concat!($op_name, "_sm120_tma_mma_tf32_", $tile_name, "_bk32_s", $s),
            module_kind: ModuleKind::TriadSm120,
            instruction_family: ResolvedInstructionFamily::MmaSync,
            instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
            operand_conversion: ResolvedOperandConversion::TensorMapUint32ThenCvtRnaTf32F32,
            tile: match $tile {
                Tf32Sm120Tile::M128N64 => (128, 64),
                Tf32Sm120Tile::M64N128 => (64, 128),
                Tf32Sm120Tile::M64N64 => (64, 64),
                Tf32Sm120Tile::M80N32Bk64 => (80, 32),
            },
            bk: 32,
            map_bk: 32,
            stages: $s,
            threads: match $tile {
                Tf32Sm120Tile::M64N64 => 128,
                Tf32Sm120Tile::M128N64 | Tf32Sm120Tile::M64N128 => 256,
                Tf32Sm120Tile::M80N32Bk64 => 160,
            },
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
                Tf32Sm120Stages::S2,
                2,
                49_280
            ),
            sm120_tf32_spec!(
                $op,
                $op_name,
                Tf32Sm120Tile::M128N64,
                "m128n64",
                Tf32Sm120Stages::S3,
                3,
                73_856
            ),
            sm120_tf32_spec!(
                $op,
                $op_name,
                Tf32Sm120Tile::M64N128,
                "m64n128",
                Tf32Sm120Stages::S2,
                2,
                49_280
            ),
            sm120_tf32_spec!(
                $op,
                $op_name,
                Tf32Sm120Tile::M64N128,
                "m64n128",
                Tf32Sm120Stages::S3,
                3,
                73_856
            ),
            sm120_tf32_spec!(
                $op,
                $op_name,
                Tf32Sm120Tile::M64N64,
                "m64n64",
                Tf32Sm120Stages::S2,
                2,
                32_896
            ),
        ]
    };
}

const SM80_TF32_NN: [Tf32KernelSpec; 6] = portable_tf32_specs!(ResolvedGemmOp::Nn, "nn");
const SM80_TF32_TN: [Tf32KernelSpec; 6] = portable_tf32_specs!(ResolvedGemmOp::Tn, "tn");
const SM80_TF32_NT: [Tf32KernelSpec; 6] = portable_tf32_specs!(ResolvedGemmOp::Nt, "nt");

pub const SM89_FINALIST_TF32_ROUTE_SPECS: [Tf32KernelSpec; 1] = [Tf32KernelSpec {
    op: ResolvedGemmOp::Nt,
    route: Tf32PhysicalRoute::Sm89MmaTf32Compact8,
    symbol: super::sm89_finalist_source::SM89_FINALIST_SYMBOL,
    module_kind: ModuleKind::TriadSm89Finalist,
    instruction_family: ResolvedInstructionFamily::MmaSync,
    instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
    operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32,
    tile: (128, 64),
    bk: 32,
    map_bk: 32,
    stages: 2,
    threads: 256,
    dynamic_shared_bytes: 49_152,
    tensor_map_revision: 0,
    schedule_revision: TF32_SCHEDULE_REVISION,
}];

/// Scoped route epoch for the isolated Ada retained-winner TF32 module.
pub const SM89_TF32_JOINT_TUNING_REVISION: u16 = 3;

pub const SM89_TF32_JOINT_ROUTE_SPECS: [Tf32KernelSpec; 11] = [
    Tf32KernelSpec {
        op: ResolvedGemmOp::Nn,
        route: Tf32PhysicalRoute::Sm89NnDirectN96,
        symbol: super::sm89_tf32_joint_source::NN_ADD_HALF_DIRECT_N96_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::RegisterAddHalfUlpTf32,
        tile: (128, 96),
        bk: 32,
        map_bk: 32,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 86_016,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Nn,
        route: Tf32PhysicalRoute::Sm89NnN96,
        symbol: super::sm89_tf32_joint_source::NN_ADD_HALF_N96_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::RegisterAddHalfUlpTf32,
        tile: (128, 96),
        bk: 32,
        map_bk: 32,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 86_016,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Tn,
        route: Tf32PhysicalRoute::Sm89TnPreRnaN96,
        symbol: super::sm89_tf32_joint_source::TN_PRE_RNA_N96_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::PreRnaAThenRegisterCvtRnaBV1,
        tile: (128, 96),
        bk: 32,
        map_bk: 32,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 86_016,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Nt,
        route: Tf32PhysicalRoute::Sm89NtALdmatrixN96,
        symbol: super::sm89_tf32_joint_source::NT_A_LDMATRIX_N96_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::RegisterAddHalfUlpTf32,
        tile: (128, 96),
        bk: 32,
        map_bk: 32,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 86_016,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Tn,
        route: Tf32PhysicalRoute::Sm89TnPreRnaM64N64,
        symbol: super::sm89_tf32_joint_source::TN_PRE_RNA_M64N64_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::PreRnaAThenRegisterCvtRnaBV1,
        tile: (64, 64),
        bk: 32,
        map_bk: 32,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 49_152,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Tn,
        route: Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2,
        symbol: super::sm89_tf32_joint_source::TN_PRE_RNA_M64N96_S2_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::PreRnaAThenRegisterCvtRnaBV1,
        tile: (64, 96),
        bk: 32,
        map_bk: 32,
        stages: 2,
        threads: 256,
        dynamic_shared_bytes: 40_960,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Nt,
        route: Tf32PhysicalRoute::Sm89NtRnaM144N96S2,
        symbol: super::sm89_tf32_joint_source::NT_RNA_M144N96_S2_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32,
        tile: (144, 96),
        bk: 32,
        map_bk: 32,
        stages: 2,
        threads: 384,
        dynamic_shared_bytes: 61_440,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Nt,
        route: Tf32PhysicalRoute::Sm89NtRowstageM128N192S2,
        symbol: super::sm89_tf32_joint_source::NT_ROWSTAGE_M128N192_S2_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::RegisterAddHalfUlpTf32,
        tile: (128, 192),
        bk: 32,
        map_bk: 32,
        stages: 2,
        threads: 256,
        dynamic_shared_bytes: 81_920,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Tn,
        route: Tf32PhysicalRoute::Sm89TnDirectM192N192S2,
        symbol: super::sm89_tf32_joint_source::TN_DIRECT_M192N192_S2_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32,
        tile: (192, 192),
        bk: 32,
        map_bk: 32,
        stages: 2,
        threads: 384,
        dynamic_shared_bytes: 98_304,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Tn,
        route: Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2,
        symbol: super::sm89_tf32_joint_source::TN_PRE_RNA_M96N192_S2_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::PreRnaAThenRegisterCvtRnaBV1,
        tile: (96, 192),
        bk: 32,
        map_bk: 32,
        stages: 2,
        threads: 384,
        dynamic_shared_bytes: 73_728,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
    Tf32KernelSpec {
        op: ResolvedGemmOp::Tn,
        route: Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3,
        symbol: super::sm89_tf32_joint_source::TN_PRE_RNA_M96N96_S3_SYMBOL,
        module_kind: ModuleKind::TriadSm89Tf32Joint,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::PreRnaAThenRegisterCvtRnaBV1,
        tile: (96, 96),
        bk: 32,
        map_bk: 32,
        stages: 3,
        threads: 256,
        dynamic_shared_bytes: 73_728,
        tensor_map_revision: 0,
        schedule_revision: TF32_SCHEDULE_REVISION,
    },
];
/// The portable extension fragment's TF32 routes: composed into the module
/// for every sm80-family target except CC 12.x, so they sit outside
/// [`SM80_TF32_ROUTE_SPECS`] and join it through `tf32_route_specs_for`.
pub const SM80_TF32_WIDE_ROUTE_SPECS: [Tf32KernelSpec; 1] = [Tf32KernelSpec {
    op: ResolvedGemmOp::Nn,
    route: Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
        tile: Tf32PortableTile::M128N128,
        stages: Tf32PortableStages::S3,
    }),
    symbol: "nn_sm80_mma_tf32_m128n128_bk32_s3",
    module_kind: ModuleKind::TriadSm80,
    instruction_family: ResolvedInstructionFamily::MmaSync,
    instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
    operand_conversion: ResolvedOperandConversion::RegisterAddHalfUlpTf32,
    tile: (128, 128),
    bk: 32,
    map_bk: 32,
    stages: 3,
    threads: 256,
    dynamic_shared_bytes: portable_tf32_dynamic_shared_bytes(
        ResolvedGemmOp::Nn,
        Tf32PortableTile::M128N128,
        Tf32PortableStages::S3,
    ),
    tensor_map_revision: 0,
    schedule_revision: TF32_PORTABLE_SCHEDULE_REVISION,
}];

/// Whether the portable module compiled for `arch` composes the extension
/// fragments (the tc64 TN stream-K twin and the wide TF32 tile). CC 12.x
/// boards run the SM120 kernels and keep their portable module byte-identical
/// to the one their TF32 cohort's portable twin was frozen against.
pub fn sm80_target_composes_extensions(arch: &str) -> bool {
    !matches!(arch, "sm_120" | "compute_120" | "sm_121" | "compute_121")
}

/// The same fact from a board's compute capability, for harnesses that hold
/// the device rather than the module target.
pub fn portable_extensions_composed_for_cc(compute_capability: (u32, u32)) -> bool {
    compute_capability.0 != 12
}

/// The extension routes of a module: the portable module's wide tile, and
/// nothing for the others.
pub fn tf32_extension_route_specs(module_kind: ModuleKind) -> &'static [Tf32KernelSpec] {
    match module_kind {
        ModuleKind::TriadSm80 => &SM80_TF32_WIDE_ROUTE_SPECS,
        _ => &[],
    }
}

/// The routes a module carries when `extensions` says its target composes
/// the extension fragments.
pub fn tf32_route_specs_for(
    module_kind: ModuleKind,
    extensions: bool,
) -> impl Iterator<Item = &'static Tf32KernelSpec> {
    tf32_route_specs(module_kind).iter().chain(
        if extensions {
            tf32_extension_route_specs(module_kind)
        } else {
            &[]
        }
        .iter(),
    )
}

/// Every route a module can carry on any target.
pub fn tf32_route_specs_all(
    module_kind: ModuleKind,
) -> impl Iterator<Item = &'static Tf32KernelSpec> {
    tf32_route_specs_for(module_kind, true)
}

pub const SM80_TF32_ROUTE_SPECS: [Tf32KernelSpec; 18] = [
    SM80_TF32_NN[0],
    SM80_TF32_NN[1],
    SM80_TF32_NN[2],
    SM80_TF32_NN[3],
    SM80_TF32_NN[4],
    SM80_TF32_NN[5],
    SM80_TF32_TN[0],
    SM80_TF32_TN[1],
    SM80_TF32_TN[2],
    SM80_TF32_TN[3],
    SM80_TF32_TN[4],
    SM80_TF32_TN[5],
    SM80_TF32_NT[0],
    SM80_TF32_NT[1],
    SM80_TF32_NT[2],
    SM80_TF32_NT[3],
    SM80_TF32_NT[4],
    SM80_TF32_NT[5],
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

const SM120_TF32_NN: [Tf32KernelSpec; 5] = sm120_tf32_specs!(ResolvedGemmOp::Nn, "nn");
const SM120_TF32_TN: [Tf32KernelSpec; 5] = sm120_tf32_specs!(ResolvedGemmOp::Tn, "tn");
const SM120_TF32_NT: [Tf32KernelSpec; 5] = sm120_tf32_specs!(ResolvedGemmOp::Nt, "nt");
const SM120_TF32_TN_M64N128_S4_PAIR: Tf32KernelSpec = Tf32KernelSpec {
    op: ResolvedGemmOp::Tn,
    route: Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
        tile: Tf32Sm120Tile::M64N128,
        stages: Tf32Sm120Stages::S4,
    }),
    symbol: "tn_sm120_tma_mma_tf32_m64n128_bk32_s4_pair",
    module_kind: ModuleKind::TriadSm120,
    instruction_family: ResolvedInstructionFamily::MmaSync,
    instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
    operand_conversion: ResolvedOperandConversion::TensorMapUint32ThenCvtRnaTf32F32,
    tile: (64, 128),
    bk: 32,
    map_bk: 32,
    stages: 4,
    threads: 256,
    dynamic_shared_bytes: 98_432,
    tensor_map_revision: TF32_TENSOR_MAP_REVISION,
    schedule_revision: TF32_SCHEDULE_REVISION,
};

const SM120_TF32_NN_M80N32_BK64_S2: Tf32KernelSpec = Tf32KernelSpec {
    op: ResolvedGemmOp::Nn,
    route: Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
        tile: Tf32Sm120Tile::M80N32Bk64,
        stages: Tf32Sm120Stages::S2,
    }),
    symbol: "nn_sm120_tma_mma_tf32_m80n32_bk64_s2",
    module_kind: ModuleKind::TriadSm120,
    instruction_family: ResolvedInstructionFamily::MmaSync,
    instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
    operand_conversion: ResolvedOperandConversion::TensorMapUint32ThenCvtRnaTf32F32,
    tile: (80, 32),
    bk: 64,
    map_bk: 32,
    stages: 2,
    threads: 160,
    dynamic_shared_bytes: 57_472,
    tensor_map_revision: TF32_TENSOR_MAP_REVISION,
    schedule_revision: TF32_SCHEDULE_REVISION,
};

/// Fragment-order slab floats per CTA slot and slots per CTA of the stream-K
/// workspace: every thread of the 256 keeps 32 accumulators, and a CTA never
/// publishes more than one slab per tile it does not own.
pub const SM120_TF32_STREAMK_SLAB_FLOATS: usize = 256 * 32;
pub const SM120_TF32_STREAMK_SLOTS_PER_CTA: usize = 2;

const SM120_TF32_TN_M64N128_S3_PAIR_STREAMK: Tf32KernelSpec = Tf32KernelSpec {
    op: ResolvedGemmOp::Tn,
    route: Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(Tf32Sm120Route {
        tile: Tf32Sm120Tile::M64N128,
        stages: Tf32Sm120Stages::S3,
    }),
    symbol: "tn_sm120_tma_mma_tf32_m64n128_bk32_s3_pair_streamk",
    module_kind: ModuleKind::TriadSm120,
    instruction_family: ResolvedInstructionFamily::MmaSync,
    instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
    operand_conversion: ResolvedOperandConversion::TensorMapUint32ThenCvtRnaTf32F32,
    tile: (64, 128),
    bk: 32,
    map_bk: 32,
    stages: 3,
    threads: 256,
    dynamic_shared_bytes: 73_856,
    tensor_map_revision: TF32_TENSOR_MAP_REVISION,
    schedule_revision: TF32_SCHEDULE_REVISION,
};

const fn sm120_fma_spec(
    op: ResolvedGemmOp,
    tile: Sm120FmaTile,
    kvec: bool,
    symbol: &'static str,
) -> Tf32KernelSpec {
    Tf32KernelSpec {
        op,
        route: Tf32PhysicalRoute::Sm120TmaFmaExact(Sm120FmaRoute {
            tile,
            kvec,
            splits: 1,
        }),
        symbol,
        module_kind: ModuleKind::TriadSm120,
        instruction_family: ResolvedInstructionFamily::ScalarFma,
        instruction_shape: ResolvedInstructionShape { m: 1, n: 1, k: 1 },
        operand_conversion: ResolvedOperandConversion::None,
        tile: tile.dims(),
        bk: SM120_FMA_BK,
        map_bk: SM120_FMA_BK,
        stages: 2,
        threads: tile.threads(),
        dynamic_shared_bytes: tile.dynamic_shared_bytes(),
        tensor_map_revision: TF32_TENSOR_MAP_REVISION,
        schedule_revision: TF32_SCHEDULE_REVISION,
    }
}

/// The exact-F32 SM120 inventory: three tiles per operation plus the NT
/// k-vector arms.
pub const SM120_FMA_ROUTE_SPECS: [Tf32KernelSpec; 12] = [
    sm120_fma_spec(
        ResolvedGemmOp::Nn,
        Sm120FmaTile::M128N64,
        false,
        "nn_sm120_tma_fma_m128n64_bk16_s2",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Nn,
        Sm120FmaTile::M64N128,
        false,
        "nn_sm120_tma_fma_m64n128_bk16_s2",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Nn,
        Sm120FmaTile::M64N64,
        false,
        "nn_sm120_tma_fma_m64n64_bk16_s2",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Tn,
        Sm120FmaTile::M128N64,
        false,
        "tn_sm120_tma_fma_m128n64_bk16_s2",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Tn,
        Sm120FmaTile::M64N128,
        false,
        "tn_sm120_tma_fma_m64n128_bk16_s2",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Tn,
        Sm120FmaTile::M64N64,
        false,
        "tn_sm120_tma_fma_m64n64_bk16_s2",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Nt,
        Sm120FmaTile::M128N64,
        false,
        "nt_sm120_tma_fma_m128n64_bk16_s2",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Nt,
        Sm120FmaTile::M64N128,
        false,
        "nt_sm120_tma_fma_m64n128_bk16_s2",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Nt,
        Sm120FmaTile::M64N64,
        false,
        "nt_sm120_tma_fma_m64n64_bk16_s2",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Nt,
        Sm120FmaTile::M128N64,
        true,
        "nt_sm120_tma_fma_m128n64_bk16_s2_kvec",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Nt,
        Sm120FmaTile::M64N128,
        true,
        "nt_sm120_tma_fma_m64n128_bk16_s2_kvec",
    ),
    sm120_fma_spec(
        ResolvedGemmOp::Nt,
        Sm120FmaTile::M64N64,
        true,
        "nt_sm120_tma_fma_m64n64_bk16_s2_kvec",
    ),
];

pub const SM120_TF32_ROUTE_SPECS: [Tf32KernelSpec; 30] = [
    SM120_TF32_NN[0],
    SM120_TF32_NN[1],
    SM120_TF32_NN[2],
    SM120_TF32_NN[3],
    SM120_TF32_NN[4],
    SM120_TF32_NN_M80N32_BK64_S2,
    SM120_TF32_TN[0],
    SM120_TF32_TN[1],
    SM120_TF32_TN[2],
    SM120_TF32_TN[3],
    SM120_TF32_TN[4],
    SM120_TF32_TN_M64N128_S4_PAIR,
    SM120_TF32_TN_M64N128_S3_PAIR_STREAMK,
    SM120_TF32_NT[0],
    SM120_TF32_NT[1],
    SM120_TF32_NT[2],
    SM120_TF32_NT[3],
    SM120_TF32_NT[4],
    SM120_FMA_ROUTE_SPECS[0],
    SM120_FMA_ROUTE_SPECS[1],
    SM120_FMA_ROUTE_SPECS[2],
    SM120_FMA_ROUTE_SPECS[3],
    SM120_FMA_ROUTE_SPECS[4],
    SM120_FMA_ROUTE_SPECS[5],
    SM120_FMA_ROUTE_SPECS[6],
    SM120_FMA_ROUTE_SPECS[7],
    SM120_FMA_ROUTE_SPECS[8],
    SM120_FMA_ROUTE_SPECS[9],
    SM120_FMA_ROUTE_SPECS[10],
    SM120_FMA_ROUTE_SPECS[11],
];

pub fn tf32_route_specs(module_kind: ModuleKind) -> &'static [Tf32KernelSpec] {
    match module_kind {
        ModuleKind::TriadSm80 => &SM80_TF32_ROUTE_SPECS,
        ModuleKind::TriadSm89Finalist => &SM89_FINALIST_TF32_ROUTE_SPECS,
        ModuleKind::TriadSm89Tf32Joint => &SM89_TF32_JOINT_ROUTE_SPECS,
        ModuleKind::TriadSm90a => &SM90A_TF32_ROUTE_SPECS,
        ModuleKind::TriadSm100 => &SM100_TF32_ROUTE_SPECS,
        ModuleKind::TriadSm120 => &SM120_TF32_ROUTE_SPECS,
        ModuleKind::Fixed
        | ModuleKind::TriadScalar
        | ModuleKind::TriadSm89Half
        | ModuleKind::TriadSm89ExactF32
        | ModuleKind::TriadSm89ExactF32D128
        | ModuleKind::Mamba3Combined => &[],
    }
}

pub fn tf32_module_symbols(module_kind: ModuleKind) -> impl ExactSizeIterator<Item = &'static str> {
    tf32_route_specs(module_kind).iter().map(|spec| spec.symbol)
}

pub fn tf32_kernel_spec(
    op: ResolvedGemmOp,
    route: Tf32PhysicalRoute,
) -> Result<&'static Tf32KernelSpec, String> {
    match route {
        Tf32PhysicalRoute::MmaTf32Rna(portable) => portable.validate()?,
        Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8(_) => {
            return Err(format!(
                "TF32 split-K route {op:?}/{route:?} has a fused-kernel specification"
            ));
        }
        _ => {}
    }
    let key = route.spec_key();
    let mut matches =
        tf32_route_specs_all(route.module_kind()).filter(|spec| spec.op == op && spec.route == key);
    let Some(spec) = matches.next() else {
        return Err(format!("no TF32 kernel matches {op:?}/{route:?}"));
    };
    if matches.next().is_some() {
        Err(format!("duplicate TF32 kernels match {op:?}/{route:?}"))
    } else {
        Ok(spec)
    }
}

pub(super) fn build_sm90a_tf32_descriptor(
    shared_address: u32,
    leading_offset: u32,
    stride_offset: u32,
) -> u64 {
    let mut descriptor = (u64::from(shared_address) >> 4) & 0x3fff;
    descriptor |= u64::from(leading_offset & 0x3fff) << 16;
    descriptor |= u64::from(stride_offset & 0x3fff) << 32;
    descriptor |= 1_u64 << 46;
    descriptor | (2_u64 << 61)
}

pub(super) fn decode_sm90a_tf32_descriptor(descriptor: u64) -> (u32, u32, u32) {
    (
        u32::try_from((descriptor & 0x3fff) << 4).expect("14-bit shared address fits u32"),
        u32::try_from((descriptor >> 16) & 0x3fff).expect("14-bit leading offset fits u32"),
        u32::try_from((descriptor >> 32) & 0x3fff).expect("14-bit stride offset fits u32"),
    )
}

pub(super) fn sm100_tf32_instruction_descriptor(
    op: ResolvedGemmOp,
    columns: u32,
) -> Result<u32, String> {
    match (op, columns) {
        (ResolvedGemmOp::Nt, 64) => Ok(0x08100910),
        (ResolvedGemmOp::Nn, 64) => Ok(0x08110910),
        (ResolvedGemmOp::Tn, 64) => Ok(0x08118910),
        (ResolvedGemmOp::Nt, 128) => Ok(0x08200910),
        (ResolvedGemmOp::Nn, 128) => Ok(0x08210910),
        (ResolvedGemmOp::Tn, 128) => Ok(0x08218910),
        (_, columns) => Err(format!(
            "unsupported SM100 TF32 instruction width {columns}"
        )),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm90aForcedRoute {
    pub op: Sm90aOp,
    pub dtype: WeightDtype,
    pub schedule: Sm90aWarpgroupSchedule,
    pub shape: Sm90aShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm90aNumericContract {
    Wgmma,
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
                "nn_sm90a_wgmma_wg1_bf16"
            }
            (Sm90aOp::Nn, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg1) => {
                "nn_sm90a_wgmma_wg1_f16"
            }
            (Sm90aOp::Tn, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg1) => {
                "tn_sm90a_wgmma_wg1_bf16"
            }
            (Sm90aOp::Tn, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg1) => {
                "tn_sm90a_wgmma_wg1_f16"
            }
            (Sm90aOp::Nt, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg1) => {
                "nt_sm90a_wgmma_wg1_bf16"
            }
            (Sm90aOp::Nt, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg1) => {
                "nt_sm90a_wgmma_wg1_f16"
            }
            (Sm90aOp::Nn, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg2) => {
                "nn_sm90a_wgmma_wg2_bf16"
            }
            (Sm90aOp::Nn, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg2) => {
                "nn_sm90a_wgmma_wg2_f16"
            }
            (Sm90aOp::Tn, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg2) => {
                "tn_sm90a_wgmma_wg2_bf16"
            }
            (Sm90aOp::Tn, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg2) => {
                "tn_sm90a_wgmma_wg2_f16"
            }
            (Sm90aOp::Nt, WeightDtype::Bf16, Sm90aWarpgroupSchedule::Wg2) => {
                "nt_sm90a_wgmma_wg2_bf16"
            }
            (Sm90aOp::Nt, WeightDtype::F16, Sm90aWarpgroupSchedule::Wg2) => {
                "nt_sm90a_wgmma_wg2_f16"
            }
            (_, WeightDtype::F32 | WeightDtype::Tf32, _) => {
                unreachable!("f32 has no SM90a WGMMA route")
            }
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
        "nn_sm100_tcgen_m128n64_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n64_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n64_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n64_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n64_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n64_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n64_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n64_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n64_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n64_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n64_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n64_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n128_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n128_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n128_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n128_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n128_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n128_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n128_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n128_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n128_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "nn_sm100_tcgen_m128n128_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n128_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "nn_sm100_tcgen_m128n128_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n64_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n64_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n64_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n64_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n64_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n64_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n64_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n64_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n64_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n64_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n64_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n64_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n128_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n128_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n128_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n128_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n128_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n128_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n128_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n128_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n128_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "tn_sm100_tcgen_m128n128_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n128_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Tn,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "tn_sm100_tcgen_m128n128_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n64_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n64_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n64_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n64_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n64_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n64_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n64_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n64_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n64_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n64_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n64_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N64,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n64_bk64_s4_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n128_bk64_s2_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n128_bk64_s2_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n128_bk64_s2_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S2,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n128_bk64_s2_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n128_bk64_s3_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n128_bk64_s3_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n128_bk64_s3_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S3,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n128_bk64_s3_p8_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n128_bk64_s4_c4_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::C4,
        "nt_sm100_tcgen_m128n128_bk64_s4_c4_f16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::Bf16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n128_bk64_s4_p8_bf16"
    ),
    sm100_spec!(
        Sm100Op::Nt,
        WeightDtype::F16,
        Sm100Tile::M128N128,
        Sm100Stages::S4,
        Sm100Schedule::P8,
        "nt_sm100_tcgen_m128n128_bk64_s4_p8_f16"
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
    Tcgen05F32,
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
    allocation_domain: AllocationDomain,
    allocation_base: CUptr,
    allocation_bytes: u64,
    offset_bytes: u64,
    required_bytes: u64,
    buffer_id: u64,
}

impl Sm90aAllocationIdentity {
    pub(super) fn matches_requested_range(self, pointer: CUptr, required_bytes: u64) -> bool {
        self.allocation_base.checked_add(self.offset_bytes) == Some(pointer)
            && self.required_bytes == required_bytes
    }

    fn requested_range_overlaps(self, other: Self) -> bool {
        if self.allocation_domain != other.allocation_domain
            || self.allocation_base != other.allocation_base
            || self.buffer_id != other.buffer_id
        {
            return false;
        }
        let self_end = self.offset_bytes.saturating_add(self.required_bytes);
        let other_end = other.offset_bytes.saturating_add(other.required_bytes);
        self.offset_bytes < other_end && other.offset_bytes < self_end
    }

    pub(super) fn query(
        pointer: CUptr,
        required_bytes: u64,
        expected_domain: AllocationDomain,
        backend: &str,
        name: &str,
    ) -> Result<Self, String> {
        #[cfg(test)]
        ALLOCATION_IDENTITY_QUERY_COUNT.fetch_add(1, Ordering::Relaxed);
        if pointer == 0 || required_bytes == 0 {
            return Err(format!("{backend} {name} allocation must be non-empty"));
        }
        let mut context: sys::CUcontext = std::ptr::null_mut();
        let mut device_ordinal = 0_i32;
        let mut buffer_id = 0_u64;
        let mut allocation_base = 0_u64;
        let mut allocation_bytes = 0_usize;
        let mut attributes = [
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_BUFFER_ID,
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_SIZE,
        ];
        let mut outputs = [
            std::ptr::from_mut(&mut context).cast(),
            std::ptr::from_mut(&mut device_ordinal).cast(),
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
        let associated_context = (!context.is_null()).then_some(context as usize);
        validate_allocation_domain(
            expected_domain,
            associated_context,
            device_ordinal,
            backend,
            name,
        )?;
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
            allocation_domain: expected_domain,
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
                b"device-ordinal",
                &self.allocation_domain.device_ordinal.to_le_bytes(),
            )
            .required(b"allocation-base", &self.allocation_base.to_le_bytes())
            .required(b"allocation-bytes", &self.allocation_bytes.to_le_bytes())
            .required(b"offset-bytes", &self.offset_bytes.to_le_bytes())
            .required(b"required-bytes", &self.required_bytes.to_le_bytes())
            .required(b"buffer-id", &self.buffer_id.to_le_bytes())
    }

    pub(super) fn physical_subrange(self, pointer: CUptr, required_bytes: u64) -> Option<Self> {
        let offset_bytes = pointer.checked_sub(self.allocation_base)?;
        let end = offset_bytes.checked_add(required_bytes)?;
        if required_bytes == 0 || end > self.allocation_bytes {
            return None;
        }
        Some(Self {
            offset_bytes,
            required_bytes,
            ..self
        })
    }

    pub(super) fn physical_digest(self) -> Sha256Digest {
        FramedSha256::new(b"physical-allocation-identity.v1")
            .required(
                b"device-ordinal",
                &self.allocation_domain.device_ordinal.to_le_bytes(),
            )
            .required(b"allocation-bytes", &self.allocation_bytes.to_le_bytes())
            .required(b"offset-bytes", &self.offset_bytes.to_le_bytes())
            .required(b"required-bytes", &self.required_bytes.to_le_bytes())
            .required(b"buffer-id", &self.buffer_id.to_le_bytes())
            .finish()
    }

    fn requery(self, backend: &str, name: &str) -> Result<Self, String> {
        let pointer = self
            .allocation_base
            .checked_add(self.offset_bytes)
            .ok_or_else(|| format!("{backend} {name} live pointer overflows u64"))?;
        Self::query(
            pointer,
            self.required_bytes,
            self.allocation_domain,
            backend,
            name,
        )
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
    pub allocation_domain: AllocationDomain,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
}

impl Sm90aPreparedTensorMaps {
    /// The managed-allocation epoch of the operands this preparation took
    /// in together with the output and bias the route touches, when all of
    /// them are managed; a later epoch means the preparation no longer
    /// describes live memory.
    pub(super) fn managed_epoch(
        &self,
        route: Sm90aForcedRoute,
        operands: Sm90aLaunchOperands,
    ) -> Result<Option<ManagedAllocationEpochStamp>, String> {
        let (output, bias) =
            sm90a_launch_allocations(route, operands, self.binding.allocation_domain)?;
        let mut ranges = Vec::with_capacity(4);
        for allocation in self
            .allocations
            .iter()
            .chain(std::iter::once(&output))
            .chain(bias.iter())
        {
            ranges.push((allocation.allocation_base, allocation.allocation_bytes));
        }
        Ok(managed_allocation_epoch_for_ranges(
            self.binding.allocation_domain.context_handle,
            &ranges,
        ))
    }

    pub fn identity_digest(&self) -> Sha256Digest {
        let dtype = match self.request.dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => 0,
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
        let live = sm90a_allocation_identities(self.keys, self.binding.allocation_domain)?;
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
    allocation_domain: AllocationDomain,
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
            allocation_domain,
            "SM90a",
            "A",
        )?,
        Sm90aAllocationIdentity::query(
            keys[1].base,
            required(keys[1])?,
            allocation_domain,
            "SM90a",
            "B",
        )?,
    ])
}

/// The output and bias allocations an SM90a launch touches beside its
/// tensor maps, identified over the span the route actually writes.
pub(super) fn sm90a_launch_allocations(
    route: Sm90aForcedRoute,
    operands: Sm90aLaunchOperands,
    allocation_domain: AllocationDomain,
) -> Result<(Sm90aAllocationIdentity, Option<Sm90aAllocationIdentity>), String> {
    let (rows, columns, element_bytes) = sm90a_output_geometry(route);
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
        allocation_domain,
        "SM90a",
        "output",
    )?;
    let bias = if operands.bias_ptr != 0 {
        let bias_bytes = u64::try_from(columns)
            .ok()
            .and_then(|columns| columns.checked_mul(4))
            .ok_or_else(|| "SM90a bias allocation span overflows u64".to_string())?;
        Some(Sm90aAllocationIdentity::query(
            operands.bias_ptr,
            bias_bytes,
            allocation_domain,
            "SM90a",
            "bias",
        )?)
    } else {
        None
    };
    Ok((output, bias))
}

fn sm90a_output_geometry(route: Sm90aForcedRoute) -> (usize, usize, u64) {
    match route.op {
        Sm90aOp::Nn => (route.shape.m, route.shape.n, 2_u64),
        Sm90aOp::Tn => (route.shape.k, route.shape.n, 4_u64),
        Sm90aOp::Nt => (route.shape.m, route.shape.k, 2_u64),
    }
}

pub(super) fn sm90a_resources_digest(
    route: Sm90aForcedRoute,
    operands: Sm90aLaunchOperands,
    allocation_domain: AllocationDomain,
    tensor_maps_digest: Sha256Digest,
) -> Result<Sha256Digest, String> {
    let (rows, columns, element_bytes) = sm90a_output_geometry(route);
    let (output, bias) = sm90a_launch_allocations(route, operands, allocation_domain)?;
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
    if let Some(bias) = bias {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sm100MapBinding {
    pub allocation_domain: AllocationDomain,
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
        let plan = sm100_tensor_map_plan(self.request, self.binding.allocation_domain)?;
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
        WeightDtype::F32 | WeightDtype::Tf32 => 0,
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
    allocation_domain: AllocationDomain,
) -> Result<(Sm90aTensorMapKey, Sm90aAllocationIdentity, (i32, i32)), String> {
    let initial =
        Sm90aAllocationIdentity::query(layout.pointer, 2, allocation_domain, "SM100", layout.name)?;
    if !initial.offset_bytes.is_multiple_of(2) {
        return Err(format!(
            "SM100 {} subview offset is not element aligned",
            layout.name
        ));
    }
    let stride = u64::try_from(layout.stride)
        .map_err(|_| format!("SM100 {} stride exceeds u64::MAX", layout.name))?;
    let element_offset = initial.offset_bytes / 2;
    let width = u64::try_from(layout.width)
        .map_err(|_| format!("SM100 {} width exceeds u64::MAX", layout.name))?;
    let rows = u64::try_from(layout.rows)
        .map_err(|_| format!("SM100 {} rows exceed u64::MAX", layout.name))?;
    // Described from the allocation base while the operand's row fits the
    // allocation grid; a matrix sliced out of a flat arena starts inside a
    // row of that grid and is described from its own first element instead.
    let allocation_origin_x = element_offset % stride;
    let fits_allocation_grid = allocation_origin_x
        .checked_add(width)
        .is_some_and(|end| end <= stride);
    let (base, origin_x, origin_y) = if fits_allocation_grid {
        (
            initial.allocation_base,
            allocation_origin_x,
            element_offset / stride,
        )
    } else if layout.pointer.is_multiple_of(16) {
        (layout.pointer, 0, 0)
    } else {
        return Err(format!(
            "SM100 {} subview starts inside a row of its allocation and is not 16-byte aligned",
            layout.name
        ));
    };
    let logical_end_x = origin_x
        .checked_add(width)
        .ok_or_else(|| format!("SM100 {} column origin overflows u64", layout.name))?;
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
        allocation_domain,
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
        base,
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
    allocation_domain: AllocationDomain,
) -> Result<Sm100TensorMapPlan, String> {
    validate_sm100_request(request)?;
    let layouts = sm100_operand_layouts(request);
    let (a_key, a_allocation, (a_x, a_y)) = sm100_subview_plan(layouts[0], allocation_domain)?;
    let (b_key, b_allocation, (b_x, b_y)) = sm100_subview_plan(layouts[1], allocation_domain)?;
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
        allocation_domain: AllocationDomain,
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
            allocation_domain,
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
                allocation_domain,
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
    pub(super) params: [u32; 10],
    pub(super) identity: Sm100RouteIdentity,
    pub(super) resources: Sm100LaunchResourceSnapshot,
}

impl Sm100PreparedLaunch {
    /// The managed-allocation epoch every operand of this preparation was
    /// taken in, when all of them are managed; a later epoch means the
    /// preparation no longer describes live memory.
    pub(super) fn managed_epoch(&self) -> Option<ManagedAllocationEpochStamp> {
        let mut ranges = Vec::with_capacity(4);
        for allocation in self
            .maps
            .allocations
            .iter()
            .chain(std::iter::once(&self.resources.output))
            .chain(self.resources.bias.iter())
        {
            ranges.push((allocation.allocation_base, allocation.allocation_bytes));
        }
        managed_allocation_epoch_for_ranges(
            self.maps.binding.allocation_domain.context_handle,
            &ranges,
        )
    }

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

/// Exact by-value parameter bundle shared by the eager and prepared-graph
/// launchers for the SM89 half NN export.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub(in crate::mamba_ssm::gpu) struct Sm89HalfNnParams {
    pub(crate) alpha: f32,
    pub(crate) beta: f32,
    pub(crate) m: i32,
    pub(crate) n: i32,
    pub(crate) k: i32,
    pub(crate) lda: i32,
    pub(crate) ldb: i32,
    pub(crate) ldc: i32,
}

unsafe impl cudarc::driver::DeviceRepr for Sm89HalfNnParams {}

/// Strided scalar-forward operands shared by every deterministic NN bucket.
#[derive(Clone, Copy)]
pub struct GemmBiFwdSubOperands {
    pub x_ptr: CUptr,
    pub lda: usize,
    pub w_ptr: CUptr,
    /// f32 bias pointer, 0 = none.
    pub bias_ptr: CUptr,
}

/// Tensor-map layout revision sealed into SM120 route identities.
pub const SM120_TENSOR_MAP_REVISION: u16 = 3;
/// Qualified automatic-route table revision for SM120.
pub const SM120_TUNING_REVISION: u16 = 0;
/// Device schedule revision sealed into SM120 route identities.
pub const SM120_SCHEDULE_REVISION: u16 = 9;

/// Logical GEMM operation implemented by the SM120 Triad module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm120Op {
    /// Forward matrix product, `Y = X @ W`.
    Nn,
    /// Weight-gradient product, `dW += X^T @ dY`.
    Tn,
    /// Input-gradient product, `dX = dY @ W^T`.
    Nt,
}

/// Output tile owned by one SM120 thread block.
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

/// Reduction-slab width for an SM120 physical route.
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

/// Number of TMA pipeline stages for an SM120 physical route.
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

/// How the CTAs of one SM120 launch divide the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Sm120Schedule {
    /// One CTA per output tile over the whole reduction.
    Tiled = 0,
    /// A persistent grid of one CTA per multiprocessor over contiguous
    /// ranges of (tile, k-tile) units, partial slabs folded in a fixed
    /// order; TN only, where the training batch reduces ten thousand rows
    /// deep over a few dozen tiles.
    StreamK = 1,
}

/// Complete physical schedule selected for one SM120 kernel launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm120PhysicalRoute {
    pub tile: Sm120Tile,
    pub bk: Sm120Bk,
    pub stages: Sm120Stages,
    pub schedule: Sm120Schedule,
}

impl Sm120PhysicalRoute {
    pub const fn wide_m_warp(self) -> bool {
        matches!(self.tile, Sm120Tile::M128N128) && matches!(self.bk, Sm120Bk::Bk32)
    }

    pub const fn compute_warps(self) -> u32 {
        if self.wide_m_warp() {
            8
        } else {
            self.tile.compute_warps()
        }
    }

    pub const fn threads(self) -> u32 {
        self.compute_warps() * 32
    }

    pub const fn warp_tile(self) -> (u32, u32) {
        if self.wide_m_warp() {
            (64, 32)
        } else {
            (32, 32)
        }
    }

    pub const fn dynamic_shared_bytes(self) -> u32 {
        let payload =
            (self.tile.output_rows() + self.tile.output_columns()) * self.bk.elements() * 2;
        payload * self.stages.count() as u32 + 128
    }

    pub const fn expected_transaction_bytes(self) -> u32 {
        (self.tile.output_rows() + self.tile.output_columns()) * self.bk.elements() * 2
    }
}

/// Logical GEMM extents and physical row-major strides.
///
/// Use [`Sm120Shape::contiguous`] to derive operation-correct strides.
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

/// Explicit SM120 route used by qualification and kernel census tooling.
///
/// Normal training code uses the typed GEMM entry points, which select only
/// measured cells from the minor-specific automatic table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm120ForcedRoute {
    pub op: Sm120Op,
    pub dtype: WeightDtype,
    pub physical: Sm120PhysicalRoute,
    pub shape: Sm120Shape,
}

/// NVRTC and device-code target pair accepted for an SM120 module build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sm120TargetCandidate {
    pub device_cc: (i32, i32),
    pub nvrtc_arch: &'static str,
    pub ptx_target: &'static str,
}

/// Static launch and resource contract for one exported SM120 kernel.
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

/// Driver-reported resources checked before an SM120 route can be promoted.
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
    ($op:expr, $dtype:expr, $tile:expr, $bk:expr, $stages:expr, $symbol:expr) => {
        sm120_spec!(
            $op,
            $dtype,
            $tile,
            $bk,
            $stages,
            Sm120Schedule::Tiled,
            $symbol
        )
    };
    ($op:expr, $dtype:expr, $tile:expr, $bk:expr, $stages:expr, $schedule:expr, $symbol:expr) => {{
        let physical = Sm120PhysicalRoute {
            tile: $tile,
            bk: $bk,
            stages: $stages,
            schedule: $schedule,
        };
        Sm120KernelSpec {
            op: $op,
            dtype: $dtype,
            physical,
            symbol: $symbol,
            threads: physical.threads(),
            dynamic_shared_bytes: physical.dynamic_shared_bytes(),
            empty_barrier_arrivals: physical.compute_warps(),
            full_barrier_arrivals: 1,
            cluster: (1, 1, 1),
            warp_tile: physical.warp_tile(),
            expected_transaction_bytes: physical.expected_transaction_bytes(),
        }
    }};
}

macro_rules! sm120_specs {
    ($(($op:expr, $op_name:literal, $tile:expr, $tile_name:literal)),+ $(,)?) => {
        [$(
            sm120_spec!($op, WeightDtype::Bf16, $tile, Sm120Bk::Bk32, Sm120Stages::S2,
                concat!($op_name, "_sm120_tma_", $tile_name, "_bk32_s2_bf16")),
            sm120_spec!($op, WeightDtype::F16, $tile, Sm120Bk::Bk32, Sm120Stages::S2,
                concat!($op_name, "_sm120_tma_", $tile_name, "_bk32_s2_f16")),
            sm120_spec!($op, WeightDtype::Bf16, $tile, Sm120Bk::Bk32, Sm120Stages::S3,
                concat!($op_name, "_sm120_tma_", $tile_name, "_bk32_s3_bf16")),
            sm120_spec!($op, WeightDtype::F16, $tile, Sm120Bk::Bk32, Sm120Stages::S3,
                concat!($op_name, "_sm120_tma_", $tile_name, "_bk32_s3_f16")),
            sm120_spec!($op, WeightDtype::Bf16, $tile, Sm120Bk::Bk64, Sm120Stages::S2,
                concat!($op_name, "_sm120_tma_", $tile_name, "_bk64_s2_bf16")),
            sm120_spec!($op, WeightDtype::F16, $tile, Sm120Bk::Bk64, Sm120Stages::S2,
                concat!($op_name, "_sm120_tma_", $tile_name, "_bk64_s2_f16")),
            sm120_spec!($op, WeightDtype::Bf16, $tile, Sm120Bk::Bk64, Sm120Stages::S3,
                concat!($op_name, "_sm120_tma_", $tile_name, "_bk64_s3_bf16")),
            sm120_spec!($op, WeightDtype::F16, $tile, Sm120Bk::Bk64, Sm120Stages::S3,
                concat!($op_name, "_sm120_tma_", $tile_name, "_bk64_s3_f16")),
        )+]
    };
}

/// Full forced SM120 census inventory; automatic dispatch uses a measured subset.
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

/// The stream-K half kernels: TN over the 64x64 BK64 S3 body, the tile the
/// half table picks for the training batch, one per operand dtype.
pub const SM120_STREAMK_KERNEL_SPECS: [Sm120KernelSpec; 2] = [
    sm120_spec!(
        Sm120Op::Tn,
        WeightDtype::Bf16,
        Sm120Tile::M64N64,
        Sm120Bk::Bk64,
        Sm120Stages::S3,
        Sm120Schedule::StreamK,
        "tn_sm120_tma_64x64_bk64_s3_streamk_bf16"
    ),
    sm120_spec!(
        Sm120Op::Tn,
        WeightDtype::F16,
        Sm120Tile::M64N64,
        Sm120Bk::Bk64,
        Sm120Stages::S3,
        Sm120Schedule::StreamK,
        "tn_sm120_tma_64x64_bk64_s3_streamk_f16"
    ),
];

/// Every SM120 half kernel the module exports: the tiled census inventory
/// followed by the stream-K bodies.
pub fn sm120_kernel_specs() -> impl Iterator<Item = &'static Sm120KernelSpec> {
    SM120_KERNEL_SPECS
        .iter()
        .chain(SM120_STREAMK_KERNEL_SPECS.iter())
}

impl Sm120ForcedRoute {
    pub fn kernel_spec(self) -> Result<&'static Sm120KernelSpec, String> {
        sm120_kernel_specs()
            .find(|spec| {
                spec.op == self.op && spec.dtype == self.dtype && spec.physical == self.physical
            })
            .ok_or_else(|| "no SM120 TMA kernel matches the forced route".to_string())
    }
}

/// Arithmetic contract sealed into every SM120 route identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sm120NumericContract {
    /// BF16/F16 operands, `mma.sync` FP32 accumulation, deterministic route order.
    TmaMma16F32,
    /// The same operands and accumulation over the stream-K schedule: the
    /// reduction is split across a persistent grid and the partial slabs
    /// fold in a fixed order, so the bits are stable for a shape on a
    /// device but differ from the tiled ladder's.
    TmaMma16F32StreamKV1,
}

impl Sm120NumericContract {
    /// The contract a physical route's schedule carries.
    pub const fn for_schedule(schedule: Sm120Schedule) -> Self {
        match schedule {
            Sm120Schedule::Tiled => Self::TmaMma16F32,
            Sm120Schedule::StreamK => Self::TmaMma16F32StreamKV1,
        }
    }
}

/// Replay-stable identity of a prepared SM120 route and all of its bindings.
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
        if self.tuning_revision != SM120_TUNING_REVISION
            || self.schedule_revision != SM120_SCHEDULE_REVISION
        {
            return Err("SM120 identity revision is stale".into());
        }
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
            WeightDtype::F32 | WeightDtype::Tf32 => {
                return Err("SM120 TMA route requires f16 or bf16".into());
            }
        };
        let (output_rows, output_columns) = match self.op {
            Sm120Op::Nn => (self.shape.m, self.shape.n),
            Sm120Op::Tn => (self.shape.k, self.shape.n),
            Sm120Op::Nt => (self.shape.m, self.shape.k),
        };
        let grid = checked_grid_product(
            checked_u32(output_rows, "SM120 route output rows")?
                .div_ceil(self.physical.tile.output_rows()),
            checked_u32(output_columns, "SM120 route output columns")?
                .div_ceil(self.physical.tile.output_columns()),
            1,
        )?;
        let arguments_digest = FramedSha256::new(b"sm120-kernel-arguments.v1")
            .required(b"symbol", self.symbol.as_bytes())
            .required(b"op", &[op as u8])
            .required(b"dtype", &[dtype as u8])
            .required(b"m", &(self.shape.m as u64).to_le_bytes())
            .required(b"k", &(self.shape.k as u64).to_le_bytes())
            .required(b"n", &(self.shape.n as u64).to_le_bytes())
            .required(b"lda", &(self.shape.lda as u64).to_le_bytes())
            .required(b"ldb", &(self.shape.ldb as u64).to_le_bytes())
            .required(b"ldc", &(self.shape.ldc as u64).to_le_bytes())
            .required(b"tuning-revision", &self.tuning_revision.to_le_bytes())
            .required(b"schedule-revision", &self.schedule_revision.to_le_bytes())
            .required(b"tensor-maps", &self.tensor_maps_digest)
            .required(b"resources", &self.resources_digest)
            .finish();
        Ok(ResolvedGemmRoute {
            op,
            dtype,
            backend: PhysicalGemmBackend::Sm120TmaMma16,
            numeric_contract: match self.physical.schedule {
                Sm120Schedule::Tiled => ResolvedNumericContract::MmaSyncF32,
                Sm120Schedule::StreamK => ResolvedNumericContract::MmaSyncF32StreamKFixedOrder,
            },
            instruction_family: ResolvedInstructionFamily::MmaSync,
            instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 16 },
            operand_conversion: ResolvedOperandConversion::None,
            ownership: match self.physical.schedule {
                Sm120Schedule::Tiled => ResolvedOutputOwnership::OneCtaPerOutputTile,
                Sm120Schedule::StreamK => {
                    ResolvedOutputOwnership::OwnerCtaPerOutputTileStreamKFixedOrder
                }
            },
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
            launch: ResolvedKernelLaunch {
                grid_dim: (grid, 1, 1),
                block_dim: (spec.threads, 1, 1),
                shared_mem_bytes: spec.dynamic_shared_bytes,
                arguments_digest,
            },
            tensor_map_revision: self.tensor_map_revision,
            tensor_maps_digest: self.tensor_maps_digest,
            resources_digest: self.resources_digest,
            tuning_table_revision: TUNING_TABLE_REVISION,
            schedule_revision: self.schedule_revision,
        })
    }
}

/// Inputs required to validate and encode the two SM120 TMA tensor maps.
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

/// Output, optional bias, and scalar operands for a prepared SM120 launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm120LaunchOperands {
    pub output_ptr: CUptr,
    pub bias_ptr: CUptr,
    pub alpha: f32,
    pub beta: f32,
}

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
/// Driver-compatible encoded tensor map passed by value to an SM120 kernel.
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
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
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
    pub allocation_domain: AllocationDomain,
    pub artifact: ArtifactIdentity,
    pub compiler: CompilerIdentity,
    pub device: DeviceIdentity,
    pub device_caps: crate::mamba_ssm::gpu::kernel_identity::DeviceCaps,
    pub target: Sm120TargetCandidate,
}

/// Validated tensor-map pair bound to one CUDA context and module identity.
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
        let plan = sm120_tensor_map_plan(self.request, self.binding.allocation_domain)?;
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
    let nn_wide_b = request.op == Sm120Op::Nn
        && request.tile == Sm120Tile::M128N64
        && request.bk == Sm120Bk::Bk32;
    let tn_wide_ab = request.op == Sm120Op::Tn
        && request.tile != Sm120Tile::M128N128
        && request.bk == Sm120Bk::Bk32;
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
                issued_coordinate_max: [
                    if nn_wide_b {
                        last_output_column
                    } else {
                        last_output_column + tile_n - bk
                    },
                    last_reduction,
                ],
                box_dimensions: if nn_wide_b {
                    [request.tile.output_columns(), request.bk.elements()]
                } else {
                    [request.bk.elements(), request.bk.elements()]
                },
                swizzle: if nn_wide_b {
                    Sm120Swizzle::Bytes128
                } else {
                    swizzle
                },
                name: "B",
            },
        ],
        Sm120Op::Tn => [
            Sm120OperandLayout {
                pointer: request.a_ptr,
                stride: shape.lda,
                width: shape.k,
                rows: shape.m,
                issued_coordinate_max: [
                    if tn_wide_ab {
                        last_output_row + tile_m - 64
                    } else {
                        last_output_row + tile_m - bk
                    },
                    last_reduction,
                ],
                box_dimensions: if tn_wide_ab {
                    [64, request.bk.elements()]
                } else {
                    [request.bk.elements(), request.bk.elements()]
                },
                swizzle: if tn_wide_ab {
                    Sm120Swizzle::Bytes128
                } else {
                    swizzle
                },
                name: "A",
            },
            Sm120OperandLayout {
                pointer: request.b_ptr,
                stride: shape.ldb,
                width: shape.n,
                rows: shape.m,
                issued_coordinate_max: [
                    if tn_wide_ab {
                        last_output_column + tile_n - 64
                    } else {
                        last_output_column + tile_n - bk
                    },
                    last_reduction,
                ],
                box_dimensions: if tn_wide_ab {
                    [64, request.bk.elements()]
                } else {
                    [request.bk.elements(), request.bk.elements()]
                },
                swizzle: if tn_wide_ab {
                    Sm120Swizzle::Bytes128
                } else {
                    swizzle
                },
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
        if layout.pointer == 0 || !layout.pointer.is_multiple_of(16) {
            return Err(format!(
                "SM120 {} TMA logical pointer must be non-null and 16-byte aligned",
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

/// Validates an SM120 map request without touching the CUDA driver.
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
    allocation_domain: AllocationDomain,
) -> Result<(Sm120TensorMapKey, Sm90aAllocationIdentity, (i32, i32)), String> {
    let initial =
        Sm90aAllocationIdentity::query(layout.pointer, 2, allocation_domain, "SM120", layout.name)?;
    if !initial.offset_bytes.is_multiple_of(2) {
        return Err(format!(
            "SM120 {} subview offset is not element aligned",
            layout.name
        ));
    }
    let stride = u64::try_from(layout.stride)
        .map_err(|_| format!("SM120 {} stride exceeds u64::MAX", layout.name))?;
    let element_offset = initial.offset_bytes / 2;
    let width = u64::try_from(layout.width)
        .map_err(|_| format!("SM120 {} width exceeds u64::MAX", layout.name))?;
    let rows = u64::try_from(layout.rows)
        .map_err(|_| format!("SM120 {} rows exceeds u64::MAX", layout.name))?;
    // A subview is normally described from its allocation base with the
    // operand's first element as the tensor-map origin, which keeps any
    // element-aligned pointer usable. A matrix sliced out of a flat arena
    // starts inside a row of that grid, so its columns would run past the
    // declared stride; such a subview is described from its own first
    // element instead, which the tensor map accepts at swizzle alignment.
    let allocation_origin_x = element_offset % stride;
    let fits_allocation_grid = allocation_origin_x
        .checked_add(width)
        .is_some_and(|end| end <= stride);
    let (base, origin_x, origin_y) = if fits_allocation_grid {
        (
            initial.allocation_base,
            allocation_origin_x,
            element_offset / stride,
        )
    } else if layout.pointer.is_multiple_of(128) {
        (layout.pointer, 0, 0)
    } else {
        return Err(format!(
            "SM120 {} subview starts inside a row of its allocation and is not 128-byte aligned",
            layout.name
        ));
    };
    let logical_end_x = origin_x
        .checked_add(width)
        .ok_or_else(|| format!("SM120 {} column origin overflows u64", layout.name))?;
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
        allocation_domain,
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
        base,
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
    allocation_domain: AllocationDomain,
) -> Result<Sm120TensorMapPlan, String> {
    validate_sm120_request(request)?;
    let layouts = sm120_operand_layouts(request);
    let (a_key, a_allocation, (a_x, a_y)) = sm120_subview_plan(layouts[0], allocation_domain)?;
    let (b_key, b_allocation, (b_x, b_y)) = sm120_subview_plan(layouts[1], allocation_domain)?;
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
        allocation_domain: AllocationDomain,
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
            allocation_domain,
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
                allocation_domain,
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
            .required(b"schedule", &[route.physical.schedule as u8])
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

/// Fully prepared forced SM120 launch for direct execution or graph capture.
///
/// Preparation is eager-only. Call [`crate::mamba_ssm::gpu::gemm_bi_triad::validate_sm120_graph_replay`]
/// before replaying a captured launch after any allocation or module change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sm120PreparedLaunch {
    pub(super) stream_handle: usize,
    pub(super) route: Sm120ForcedRoute,
    pub(super) maps: Sm120PreparedTensorMaps,
    pub(super) operands: Sm120LaunchOperands,
    pub(super) params: [u32; 10],
    pub(super) identity: Sm120RouteIdentity,
    pub(super) resolved_launch_set: ResolvedGemmLaunchSet,
    pub(super) resources: Sm120LaunchResourceSnapshot,
    pub(super) kernel_resources: Sm120KernelResources,
}

impl Sm120PreparedLaunch {
    /// Returns the complete physical route and binding identity.
    pub fn identity(&self) -> Sm120RouteIdentity {
        self.identity
    }

    /// Returns the driver resource census recorded during preparation.
    pub fn resources(&self) -> Sm120KernelResources {
        self.kernel_resources
    }

    /// Returns the resolved launch set sealed into this preparation.
    pub fn resolved_launch_set(&self) -> ResolvedGemmLaunchSet {
        self.resolved_launch_set
    }

    pub(super) fn managed_epoch(&self) -> Option<ManagedAllocationEpochStamp> {
        let mut ranges = Vec::with_capacity(4);
        for allocation in self
            .maps
            .allocations
            .iter()
            .chain(std::iter::once(&self.resources.output))
            .chain(self.resources.bias.iter())
        {
            ranges.push((allocation.allocation_base, allocation.allocation_bytes));
        }
        managed_allocation_epoch_for_ranges(
            self.maps.binding.allocation_domain.context_handle,
            &ranges,
        )
    }
}

/// Schedule revision of the SM100 half routes: no schedule has been tuned
/// yet, so the observer sees the first one.
pub const SM100_SCHEDULE_REVISION: u16 = 0;
/// Schedule revision of the SM90a half routes: the warpgroup schedule is
/// the route itself, and no other has been tuned.
pub const SM90A_SCHEDULE_REVISION: u16 = 0;
/// Tensor-map revision of the SM90a half routes: the first encoding.
pub const SM90A_TENSOR_MAP_REVISION: u16 = 1;

impl Sm100RouteIdentity {
    /// The route this identity names, in the form the launch observer and
    /// the graph replay record; `device_caps` is the board the module was
    /// bound on. Mirrors the SM120 conversion field for field so that the
    /// three specialized families are observed alike.
    pub fn resolved_route(self, device_caps: DeviceCaps) -> Result<ResolvedGemmRoute, String> {
        let route = Sm100ForcedRoute {
            op: self.op,
            dtype: self.dtype,
            physical: self.physical,
            shape: self.shape,
        };
        route.shape.validate(route.op)?;
        let spec = route.kernel_spec()?;
        if self.symbol != spec.symbol {
            return Err("SM100 identity symbol does not match its physical route".into());
        }
        if self.module_kind != ModuleKind::TriadSm100
            || self.artifact.module_kind != ModuleKind::TriadSm100
        {
            return Err("SM100 identity does not name the SM100 TRIAD module".into());
        }
        if self.compiler.target.as_str() != self.target.nvrtc_arch
            || self.device.target.as_str() != self.target.ptx_target
        {
            return Err("SM100 identity target transaction is inconsistent".into());
        }
        let op = match self.op {
            Sm100Op::Nn => ResolvedGemmOp::Nn,
            Sm100Op::Tn => ResolvedGemmOp::Tn,
            Sm100Op::Nt => ResolvedGemmOp::Nt,
        };
        let dtype = match self.dtype {
            WeightDtype::F16 => PolicyDtype::F16,
            WeightDtype::Bf16 => PolicyDtype::Bf16,
            WeightDtype::F32 | WeightDtype::Tf32 => {
                return Err("SM100 TCGEN route requires f16 or bf16".into());
            }
        };
        let (output_rows, output_columns) = match self.op {
            Sm100Op::Nn => (self.shape.m, self.shape.n),
            Sm100Op::Tn => (self.shape.k, self.shape.n),
            Sm100Op::Nt => (self.shape.m, self.shape.k),
        };
        let grid = checked_grid_product(
            checked_u32(output_rows, "SM100 route output rows")?
                .div_ceil(self.physical.tile.output_rows()),
            checked_u32(output_columns, "SM100 route output columns")?
                .div_ceil(self.physical.tile.output_columns()),
            1,
        )?;
        let arguments_digest = FramedSha256::new(b"sm100-kernel-arguments.v1")
            .required(b"symbol", self.symbol.as_bytes())
            .required(b"op", &[op as u8])
            .required(b"dtype", &[dtype as u8])
            .required(b"m", &(self.shape.m as u64).to_le_bytes())
            .required(b"k", &(self.shape.k as u64).to_le_bytes())
            .required(b"n", &(self.shape.n as u64).to_le_bytes())
            .required(b"lda", &(self.shape.lda as u64).to_le_bytes())
            .required(b"ldb", &(self.shape.ldb as u64).to_le_bytes())
            .required(b"ldc", &(self.shape.ldc as u64).to_le_bytes())
            .required(b"tuning-revision", &self.tuning_revision.to_le_bytes())
            .required(b"tensor-maps", &self.tensor_maps_digest)
            .required(b"resources", &self.resources_digest)
            .finish();
        let tile_columns = self.physical.tile.output_columns();
        Ok(ResolvedGemmRoute {
            op,
            dtype,
            backend: PhysicalGemmBackend::Sm100Tcgen05,
            numeric_contract: ResolvedNumericContract::Tcgen05F32,
            instruction_family: ResolvedInstructionFamily::Tcgen05,
            instruction_shape: ResolvedInstructionShape {
                m: 128,
                n: u16::try_from(tile_columns)
                    .map_err(|_| format!("SM100 tile width {tile_columns} exceeds u16"))?,
                k: 16,
            },
            operand_conversion: ResolvedOperandConversion::None,
            ownership: ResolvedOutputOwnership::OneCtaPerOutputTile,
            symbol: self.symbol,
            module_kind: self.module_kind,
            target: self.compiler.target,
            artifact: self.artifact,
            compiler: self.compiler,
            device: self.device,
            device_caps,
            shape: (self.shape.m, self.shape.k, self.shape.n),
            strides: (self.shape.lda, self.shape.ldb, self.shape.ldc),
            tile: (
                self.physical.tile.output_rows(),
                self.physical.tile.output_columns(),
            ),
            bk: spec.bk,
            stages: self.physical.stages.count(),
            threads: spec.threads,
            launch: ResolvedKernelLaunch {
                grid_dim: (grid, 1, 1),
                block_dim: (spec.threads, 1, 1),
                shared_mem_bytes: spec.dynamic_shared_bytes,
                arguments_digest,
            },
            tensor_map_revision: self.tensor_map_revision,
            tensor_maps_digest: self.tensor_maps_digest,
            resources_digest: self.resources_digest,
            tuning_table_revision: TUNING_TABLE_REVISION,
            schedule_revision: SM100_SCHEDULE_REVISION,
        })
    }
}

impl Sm90aRouteIdentity {
    /// The route this identity names, in the form the launch observer and
    /// the graph replay record; `device_caps` is the board the module was
    /// bound on.
    pub fn resolved_route(self, device_caps: DeviceCaps) -> Result<ResolvedGemmRoute, String> {
        self.shape.validate(self.op)?;
        if self.module_kind != ModuleKind::TriadSm90a
            || self.artifact.module_kind != ModuleKind::TriadSm90a
        {
            return Err("SM90a identity does not name the SM90a TRIAD module".into());
        }
        if self.compiler.target.as_str() != self.exact_target
            || self.device.target.as_str() != self.exact_target
        {
            return Err("SM90a identity target transaction is inconsistent".into());
        }
        let op = match self.op {
            Sm90aOp::Nn => ResolvedGemmOp::Nn,
            Sm90aOp::Tn => ResolvedGemmOp::Tn,
            Sm90aOp::Nt => ResolvedGemmOp::Nt,
        };
        let dtype = match self.dtype {
            WeightDtype::F16 => PolicyDtype::F16,
            WeightDtype::Bf16 => PolicyDtype::Bf16,
            WeightDtype::F32 | WeightDtype::Tf32 => {
                return Err("SM90a WGMMA route requires f16 or bf16".into());
            }
        };
        let (output_rows, output_columns) = match self.op {
            Sm90aOp::Nn => (self.shape.m, self.shape.n),
            Sm90aOp::Tn => (self.shape.k, self.shape.n),
            Sm90aOp::Nt => (self.shape.m, self.shape.k),
        };
        let (tile_rows, tile_columns, tile_reduction) = self.tile;
        let grid = checked_grid_product(
            checked_u32(output_rows, "SM90a route output rows")?.div_ceil(tile_rows),
            checked_u32(output_columns, "SM90a route output columns")?.div_ceil(tile_columns),
            1,
        )?;
        let threads = self.schedule.threads();
        let arguments_digest = FramedSha256::new(b"sm90a-kernel-arguments.v1")
            .required(b"symbol", self.symbol.as_bytes())
            .required(b"op", &[op as u8])
            .required(b"dtype", &[dtype as u8])
            .required(b"m", &(self.shape.m as u64).to_le_bytes())
            .required(b"k", &(self.shape.k as u64).to_le_bytes())
            .required(b"n", &(self.shape.n as u64).to_le_bytes())
            .required(b"lda", &(self.shape.lda as u64).to_le_bytes())
            .required(b"ldb", &(self.shape.ldb as u64).to_le_bytes())
            .required(b"ldc", &(self.shape.ldc as u64).to_le_bytes())
            .required(b"tuning-revision", &self.tuning_revision.to_le_bytes())
            .required(b"tensor-maps", &self.tensor_maps_digest)
            .required(b"resources", &self.resources_digest)
            .finish();
        Ok(ResolvedGemmRoute {
            op,
            dtype,
            backend: PhysicalGemmBackend::Sm90aWgmma,
            numeric_contract: ResolvedNumericContract::WgmmaF32,
            instruction_family: ResolvedInstructionFamily::Wgmma,
            instruction_shape: ResolvedInstructionShape {
                m: 64,
                n: u16::try_from(tile_columns)
                    .map_err(|_| format!("SM90a tile width {tile_columns} exceeds u16"))?,
                k: 16,
            },
            operand_conversion: ResolvedOperandConversion::None,
            ownership: ResolvedOutputOwnership::OneCtaPerOutputTile,
            symbol: self.symbol,
            module_kind: self.module_kind,
            target: self.compiler.target,
            artifact: self.artifact,
            compiler: self.compiler,
            device: self.device,
            device_caps,
            shape: (self.shape.m, self.shape.k, self.shape.n),
            strides: (self.shape.lda, self.shape.ldb, self.shape.ldc),
            tile: (tile_rows, tile_columns),
            bk: tile_reduction,
            stages: self.stages,
            threads,
            launch: ResolvedKernelLaunch {
                grid_dim: (grid, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: SM90A_DYNAMIC_SHARED_BYTES,
                arguments_digest,
            },
            tensor_map_revision: SM90A_TENSOR_MAP_REVISION,
            tensor_maps_digest: self.tensor_maps_digest,
            resources_digest: self.resources_digest,
            tuning_table_revision: TUNING_TABLE_REVISION,
            schedule_revision: SM90A_SCHEDULE_REVISION,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AllocationDomain, F32EncodedTensorMaps, F32LaunchResourceSnapshot, F32PreparedTensorMaps,
        F32TriadOperands, F32TriadRequest, F32TriadShape, F32ZeroReductionTensorMaps, GemmDims,
        SCALAR_NN_M32N64_SPLITK32_DYNAMIC_SHARED_BYTES,
        SCALAR_NN_M32N64_SPLITK32_MIN_ACTIVE_BLOCKS, SCALAR_NN_M32N64_SPLITK32_REGISTER_CAP,
        SCALAR_NN_M32N64_SPLITK32_STATIC_SHARED_BYTES, SCALAR_NN_M32N64_SPLITK32_THREADS,
        SCALAR_TN_M16N16_DYNAMIC_SHARED_BYTES, SCALAR_TN_M16N16_MIN_ACTIVE_BLOCKS,
        SCALAR_TN_M16N16_REGISTER_CAP, SCALAR_TN_M16N16_STATIC_SHARED_BYTES,
        SCALAR_TN_M16N16_THREADS, SM120_KERNEL_SPECS, Sm90aAllocationIdentity, Sm90aMapRequest,
        Sm90aOp, Sm90aShape, Sm90aWarpgroupSchedule, Sm100MapRequest, Sm100Op, Sm100Schedule,
        Sm100Shape, Sm100Stages, Sm100Tile, Sm120Bk, Sm120Op, Sm120Stages, Sm120Tile,
        TF32_NT_SPLITK4_S4_SPEC, TF32_NT_SPLITK8_S3_SPEC, TF32_PORTABLE_SCHEDULE_REVISION,
        TF32_SCHEDULE_REVISION, TF32_SPLITK_CANDIDATE_SPECS, TF32_SPLITK_EXTENSION_SPECS,
        TF32_SPLITK2_SPEC, TF32_SPLITK4_SPEC, TF32_TENSOR_MAP_REVISION,
        TF32_TN_SPLITK8_M32_S3_SPEC, Tf32EncodedMapIdentity, Tf32MapBinding, Tf32OperandLayout,
        Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile,
        Tf32QualifiedModule, Tf32Sm90aRoute, Tf32Sm100Route, Tf32Sm120Route, Tf32Sm120Stages,
        Tf32Sm120Tile, Tf32TensorMap, Tf32TensorMapFormat, Tf32TensorMapKey, Tf32TensorOrigins,
        ZERO_REDUCTION_MAP_REVISION, append_tf32_encoded_map_identity, build_sm90a_tf32_descriptor,
        checked_grid_product, checked_u32, decode_sm90a_tf32_descriptor, sm90a_tensor_map_keys,
        sm100_operand_layouts, sm100_tensor_map_keys, sm100_tf32_instruction_descriptor,
        tf32_kernel_spec, tf32_operand_layouts, tf32_route_specs, tf32_splitk_partition_bounds,
        tf32_splitk_spec, tf32_splitk_specs_all, tf32_splitk_specs_for, tf32_subview_plan,
        validate_allocation_domain, validate_bias_preseed, validate_sm100_issued_coordinates,
        validate_tf32_issued_coordinates, zeroed_tensor_map_sentinel,
    };

    const TEST_ALLOCATION_DOMAIN: AllocationDomain = AllocationDomain {
        context_handle: 0x1234,
        device_ordinal: 2,
    };

    fn allocation_identity(offset_bytes: u64, required_bytes: u64) -> Sm90aAllocationIdentity {
        Sm90aAllocationIdentity {
            allocation_domain: TEST_ALLOCATION_DOMAIN,
            allocation_base: 0x1000,
            allocation_bytes: 0x1000,
            offset_bytes,
            required_bytes,
            buffer_id: 7,
        }
    }

    #[test]
    fn allocation_identity_detects_requested_span_overlap() {
        let output = allocation_identity(64, 128);
        assert!(output.requested_range_overlaps(allocation_identity(0, 65)));
        assert!(output.requested_range_overlaps(allocation_identity(191, 32)));
        assert!(output.requested_range_overlaps(allocation_identity(96, 16)));
    }

    #[test]
    fn tma_tensor_map_rejects_a_logical_pointer_below_its_alignment() {
        let layout = Tf32OperandLayout {
            pointer: 0x1004,
            stride: 64,
            width: 32,
            rows: 32,
            issued_coordinate_max: [0, 0],
            box_dimensions: [32, 32],
            name: "A",
        };
        let error = tf32_subview_plan(layout, TEST_ALLOCATION_DOMAIN, Tf32TensorMapFormat::Uint32)
            .expect_err("TMA must reject a four-byte-aligned logical pointer");
        assert!(error.contains("TMA pointer must be 16-byte aligned"));
    }

    #[test]
    fn allocation_identity_accepts_adjacent_or_distinct_allocations() {
        let output = allocation_identity(64, 128);
        assert!(!output.requested_range_overlaps(allocation_identity(0, 64)));
        assert!(!output.requested_range_overlaps(allocation_identity(192, 32)));
        assert!(!output.requested_range_overlaps(Sm90aAllocationIdentity {
            buffer_id: 8,
            ..allocation_identity(96, 16)
        }));
    }

    #[test]
    fn allocation_domain_accepts_matching_associated_context_and_device() {
        validate_allocation_domain(TEST_ALLOCATION_DOMAIN, Some(0x1234), 2, "test", "A").unwrap();
    }

    #[test]
    fn allocation_domain_rejects_mismatched_associated_context() {
        let error =
            validate_allocation_domain(TEST_ALLOCATION_DOMAIN, Some(0x5678), 2, "test", "A")
                .expect_err("an associated allocation from another context must be rejected");
        assert!(error.contains("different CUDA context"), "{error}");
    }

    #[test]
    fn allocation_domain_rejects_mismatched_device_after_context_match() {
        let error =
            validate_allocation_domain(TEST_ALLOCATION_DOMAIN, Some(0x1234), 3, "test", "A")
                .expect_err("an associated allocation from another device must be rejected");
        assert!(error.contains("CUDA device 3"), "{error}");
        assert!(error.contains("expected CUDA device 2"), "{error}");
    }

    #[test]
    fn allocation_domain_accepts_null_context_only_on_matching_device() {
        validate_allocation_domain(TEST_ALLOCATION_DOMAIN, None, 2, "test", "A").unwrap();
    }

    #[test]
    fn allocation_domain_rejects_null_context_on_mismatched_device() {
        let error = validate_allocation_domain(TEST_ALLOCATION_DOMAIN, None, 3, "test", "A")
            .expect_err("a contextless allocation from another device must be rejected");
        assert!(error.contains("CUDA device 3"), "{error}");
        assert!(error.contains("expected CUDA device 2"), "{error}");
    }
    use crate::mamba_ssm::gpu::buffers::register_managed_allocation_range;
    use crate::mamba_ssm::gpu::dtype::WeightDtype;
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, ModuleKind, NUMERIC_ABI_REVISION,
        ResolvedGemmOp, ResolvedInstructionFamily, ResolvedInstructionShape,
        ResolvedOperandConversion, SCHEDULE_REVISION,
    };

    #[test]
    fn scalar_tn_m16n16_resource_contract_is_exact() {
        assert_eq!(SCALAR_TN_M16N16_THREADS, 64);
        assert_eq!(SCALAR_TN_M16N16_DYNAMIC_SHARED_BYTES, 4_096);
        assert_eq!(SCALAR_TN_M16N16_STATIC_SHARED_BYTES, 0);
        assert_eq!(SCALAR_TN_M16N16_REGISTER_CAP, 112);
        assert_eq!(SCALAR_TN_M16N16_MIN_ACTIVE_BLOCKS, 8);
    }

    #[test]
    fn scalar_nn_m32n64_splitk32_resource_contract_is_exact() {
        assert_eq!(SCALAR_NN_M32N64_SPLITK32_THREADS, 128);
        assert_eq!(SCALAR_NN_M32N64_SPLITK32_DYNAMIC_SHARED_BYTES, 0);
        assert_eq!(SCALAR_NN_M32N64_SPLITK32_STATIC_SHARED_BYTES, 13_312);
        assert_eq!(SCALAR_NN_M32N64_SPLITK32_REGISTER_CAP, 64);
        assert_eq!(SCALAR_NN_M32N64_SPLITK32_MIN_ACTIVE_BLOCKS, 4);
    }

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
            let reduction = || shape.reduction(op);
            let output_rows = || shape.output_rows(op);
            let output_columns = || shape.output_columns(op);
            assert_eq!(reduction(), 0);
            assert_eq!(output_rows(), 3);
            assert_eq!(output_columns(), 5);
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
            (ModuleKind::TriadSm80, 18, [6, 6, 6]),
            (ModuleKind::TriadSm89Finalist, 1, [0, 0, 1]),
            (ModuleKind::TriadSm89Tf32Joint, 11, [2, 6, 3]),
            (ModuleKind::TriadSm90a, 6, [2, 2, 2]),
            (ModuleKind::TriadSm100, 36, [12, 12, 12]),
            (ModuleKind::TriadSm120, 30, [9, 10, 11]),
        ];
        let expected_total = expected.iter().map(|(_, count, _)| count).sum::<usize>();
        let mut all_symbols = std::collections::BTreeSet::new();
        for (module_kind, count, op_counts) in expected {
            let specs = tf32_route_specs(module_kind);
            assert_eq!(specs.len(), count, "{module_kind:?}");
            for (op, op_count) in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt]
                .into_iter()
                .zip(op_counts)
            {
                assert_eq!(
                    specs.iter().filter(|spec| spec.op == op).count(),
                    op_count,
                    "{module_kind:?}/{op:?}"
                );
            }
            for spec in specs {
                assert_eq!(spec.module_kind, module_kind);
                assert_eq!(tf32_kernel_spec(spec.op, spec.route).unwrap(), spec);
                match spec.route {
                    super::Tf32PhysicalRoute::MmaTf32Rna(_)
                    | super::Tf32PhysicalRoute::MmaTf32RnaSplitK2(_)
                    | super::Tf32PhysicalRoute::MmaTf32RnaSplitK4(_)
                    | super::Tf32PhysicalRoute::MmaTf32RnaSplitK8(_)
                    | super::Tf32PhysicalRoute::Sm89MmaTf32Compact8
                    | super::Tf32PhysicalRoute::Sm89TnPreRnaN96
                    | super::Tf32PhysicalRoute::Sm89TnPreRnaM64N64
                    | super::Tf32PhysicalRoute::Sm89TnPreRnaM64N96S2
                    | super::Tf32PhysicalRoute::Sm89NnDirectN96
                    | super::Tf32PhysicalRoute::Sm89NnN96
                    | super::Tf32PhysicalRoute::Sm89NtALdmatrixN96
                    | super::Tf32PhysicalRoute::Sm89NtRnaM144N96S2
                    | super::Tf32PhysicalRoute::Sm89NtRowstageM128N192S2
                    | super::Tf32PhysicalRoute::Sm89TnDirectM192N192S2
                    | super::Tf32PhysicalRoute::Sm89TnPreRnaM96N192S2
                    | super::Tf32PhysicalRoute::Sm89TnPreRnaM96N96S3
                    | super::Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(_)
                    | super::Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_) => {
                        assert_eq!(spec.instruction_family, ResolvedInstructionFamily::MmaSync);
                        assert_eq!(
                            spec.instruction_shape,
                            ResolvedInstructionShape { m: 16, n: 8, k: 8 }
                        );
                    }
                    super::Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(_) => {
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
                    super::Tf32PhysicalRoute::Sm120TmaFmaExact(_) => {
                        assert_eq!(
                            spec.instruction_family,
                            ResolvedInstructionFamily::ScalarFma
                        );
                        assert_eq!(
                            spec.instruction_shape,
                            ResolvedInstructionShape { m: 1, n: 1, k: 1 }
                        );
                        assert_eq!(spec.operand_conversion, ResolvedOperandConversion::None);
                        assert_eq!(spec.bk, super::SM120_FMA_BK);
                        assert_eq!(spec.stages, 2);
                    }
                    super::Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(_) => {
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
                if !spec.route.is_exact_fma() {
                    assert_ne!(spec.operand_conversion, ResolvedOperandConversion::None);
                }
                assert!(all_symbols.insert(spec.symbol), "duplicate {}", spec.symbol);
            }
        }
        assert_eq!(all_symbols.len(), expected_total);
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn] {
            assert!(
                tf32_kernel_spec(op, super::Tf32PhysicalRoute::Sm89MmaTf32Compact8).is_err(),
                "the NT-only finalist unexpectedly accepts {op:?}"
            );
        }
        assert!(tf32_route_specs(ModuleKind::Fixed).is_empty());
        assert!(tf32_route_specs(ModuleKind::TriadScalar).is_empty());
        assert!(tf32_route_specs(ModuleKind::Mamba3Combined).is_empty());
    }

    #[test]
    fn rect_wide_spec_keeps_logical_bk64_and_bk32_tensor_maps() {
        let route = Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
            tile: Tf32Sm120Tile::M80N32Bk64,
            stages: Tf32Sm120Stages::S2,
        });
        let spec = tf32_kernel_spec(ResolvedGemmOp::Nn, route).unwrap();
        assert_eq!(spec.symbol, "nn_sm120_tma_mma_tf32_m80n32_bk64_s2");
        assert_eq!(spec.tile, (80, 32));
        assert_eq!((spec.bk, spec.map_bk), (64, 32));
        assert_eq!(
            (spec.stages, spec.threads, spec.dynamic_shared_bytes),
            (2, 160, 57_472)
        );

        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (512, 3_072, 768)),
        };
        let layouts = tf32_operand_layouts(
            request,
            F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: 0.0,
            },
            route,
        )
        .unwrap();
        assert_eq!(layouts[0].box_dimensions, [32, 80]);
        assert_eq!(layouts[1].box_dimensions, [32, 32]);
        assert_eq!(
            (layouts[0].width, layouts[0].rows, layouts[0].stride),
            (3_072, 512, 3_072)
        );
        assert_eq!(
            (layouts[1].width, layouts[1].rows, layouts[1].stride),
            (768, 3_072, 768)
        );

        for (op, stages) in [
            (ResolvedGemmOp::Tn, Tf32Sm120Stages::S2),
            (ResolvedGemmOp::Nt, Tf32Sm120Stages::S2),
            (ResolvedGemmOp::Nn, Tf32Sm120Stages::S3),
            (ResolvedGemmOp::Nn, Tf32Sm120Stages::S4),
        ] {
            assert!(
                tf32_kernel_spec(
                    op,
                    Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
                        tile: Tf32Sm120Tile::M80N32Bk64,
                        stages,
                    }),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn portable_tf32_nn_splitk2_and_splitk4_are_distinct_bit_families() {
        let splitk2 = Tf32PhysicalRoute::MmaTf32RnaSplitK2(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        let route = Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        let splitk2_spec = tf32_splitk_spec(ResolvedGemmOp::Nn, splitk2).unwrap();
        assert_eq!(splitk2_spec, &TF32_SPLITK2_SPEC);
        assert_eq!(splitk2_spec.route, splitk2);
        assert_eq!(
            splitk2_spec.symbol,
            "nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4"
        );
        assert_eq!(splitk2_spec.tile, (16, 32));
        assert_eq!(splitk2_spec.bk, 32);
        assert_eq!(splitk2_spec.stages, 4);
        assert_eq!(splitk2_spec.partitions, 2);
        assert_eq!(splitk2_spec.threads, 128);
        assert_eq!(splitk2_spec.dynamic_shared_bytes, 29_696);

        let spec = tf32_splitk_spec(ResolvedGemmOp::Nn, route).unwrap();
        assert_eq!(spec, &TF32_SPLITK4_SPEC);
        assert_eq!(spec.route, route);
        assert_eq!(spec.symbol, "nn_sm80_mma_tf32_splitk4_m16n32_bk32_s4");
        assert_eq!(spec.tile, (16, 32));
        assert_eq!(spec.bk, 32);
        assert_eq!(spec.stages, 4);
        assert_eq!(spec.partitions, 4);
        assert_eq!(spec.threads, 128);
        assert_eq!(spec.dynamic_shared_bytes, 29_696);

        assert_eq!(&TF32_SPLITK_CANDIDATE_SPECS[..2], &[*splitk2_spec, *spec]);
        assert!(
            tf32_splitk_spec(
                ResolvedGemmOp::Nn,
                Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S3,
                }),
            )
            .is_err()
        );

        assert_eq!(
            (0..2)
                .map(|partition| tf32_splitk_partition_bounds(384, 2, partition).unwrap())
                .collect::<Vec<_>>(),
            [(0, 192), (192, 384)]
        );
        assert_eq!(
            (0..2)
                .map(|partition| tf32_splitk_partition_bounds(833, 2, partition).unwrap())
                .collect::<Vec<_>>(),
            [(0, 448), (448, 833)]
        );
        assert_eq!(
            (0..2)
                .map(|partition| tf32_splitk_partition_bounds(17, 2, partition).unwrap())
                .collect::<Vec<_>>(),
            [(0, 17), (17, 17)]
        );
        assert_eq!(
            (0..4)
                .map(|partition| tf32_splitk_partition_bounds(384, 4, partition).unwrap())
                .collect::<Vec<_>>(),
            [(0, 96), (96, 192), (192, 288), (288, 384)]
        );
        assert_eq!(
            (0..4)
                .map(|partition| tf32_splitk_partition_bounds(833, 4, partition).unwrap())
                .collect::<Vec<_>>(),
            [(0, 224), (224, 448), (448, 672), (672, 833)]
        );
        assert!(tf32_splitk_partition_bounds(384, 2, 2).is_err());
        assert!(tf32_splitk_partition_bounds(384, 4, 4).is_err());
        assert!(tf32_splitk_partition_bounds(384, 3, 0).is_err());
        assert!(tf32_splitk_spec(ResolvedGemmOp::Tn, splitk2).is_err());
        assert!(tf32_splitk_spec(ResolvedGemmOp::Nt, splitk2).is_err());
        assert!(tf32_splitk_spec(ResolvedGemmOp::Tn, route).is_err());
        assert_eq!(
            tf32_splitk_spec(ResolvedGemmOp::Nt, route).unwrap(),
            &TF32_NT_SPLITK4_S4_SPEC
        );
        assert!(
            tf32_splitk_spec(
                ResolvedGemmOp::Nn,
                Tf32PhysicalRoute::MmaTf32Rna(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
            )
            .is_err()
        );
    }

    #[test]
    fn portable_tf32_splitk_inventory_covers_nn_and_nt_candidates_exactly() {
        let actual = TF32_SPLITK_CANDIDATE_SPECS
            .iter()
            .map(|spec| {
                (
                    spec.op,
                    spec.symbol,
                    spec.tile,
                    spec.stages,
                    spec.partitions,
                    spec.dynamic_shared_bytes,
                    spec.register_cap,
                    spec.occupancy_gate,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            [
                (
                    ResolvedGemmOp::Nn,
                    "nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4",
                    (16, 32),
                    4,
                    2,
                    29_696,
                    128,
                    3,
                ),
                (
                    ResolvedGemmOp::Nn,
                    "nn_sm80_mma_tf32_splitk4_m16n32_bk32_s4",
                    (16, 32),
                    4,
                    4,
                    29_696,
                    128,
                    3,
                ),
                (
                    ResolvedGemmOp::Nt,
                    "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s3",
                    (16, 32),
                    3,
                    4,
                    20_736,
                    96,
                    3,
                ),
                (
                    ResolvedGemmOp::Nt,
                    "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s4",
                    (16, 32),
                    4,
                    4,
                    27_648,
                    96,
                    3,
                ),
                (
                    ResolvedGemmOp::Nt,
                    "nt_sm80_mma_tf32_splitk8_m32n32_bk32_s3",
                    (32, 32),
                    3,
                    8,
                    27_648,
                    96,
                    3,
                ),
                (
                    ResolvedGemmOp::Nt,
                    "nt_sm80_mma_tf32_splitk8_m32n32_bk32_s4",
                    (32, 32),
                    4,
                    8,
                    36_864,
                    96,
                    2,
                ),
            ]
        );
    }

    #[test]
    fn portable_tf32_splitk_extension_inventory_covers_tn_candidates_exactly() {
        let actual = TF32_SPLITK_EXTENSION_SPECS
            .iter()
            .map(|spec| {
                (
                    spec.op,
                    spec.symbol,
                    spec.tile,
                    spec.stages,
                    spec.partitions,
                    spec.dynamic_shared_bytes,
                    spec.register_cap,
                    spec.occupancy_gate,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            [
                (
                    ResolvedGemmOp::Tn,
                    "tn_sm80_mma_tf32_splitk8_m64n64_bk32_s2",
                    (64, 64),
                    2,
                    8,
                    36_864,
                    128,
                    2,
                ),
                (
                    ResolvedGemmOp::Tn,
                    "tn_sm80_mma_tf32_splitk8_m64n64_bk32_s3",
                    (64, 64),
                    3,
                    8,
                    55_296,
                    128,
                    1,
                ),
                (
                    ResolvedGemmOp::Tn,
                    "tn_sm80_mma_tf32_splitk8_m32n32_bk32_s3",
                    (32, 32),
                    3,
                    8,
                    30_720,
                    96,
                    3,
                ),
                (
                    ResolvedGemmOp::Tn,
                    "tn_sm80_mma_tf32_splitk8_m32n32_bk32_s4",
                    (32, 32),
                    4,
                    8,
                    40_960,
                    96,
                    2,
                ),
            ]
        );
        assert!(TF32_SPLITK_EXTENSION_SPECS.iter().all(|spec| spec.route
            == Tf32PhysicalRoute::MmaTf32RnaSplitK8(Tf32PortableRoute {
                tile: match spec.tile {
                    (64, 64) => Tf32PortableTile::M64N64,
                    _ => Tf32PortableTile::M32N32,
                },
                stages: match spec.stages {
                    2 => Tf32PortableStages::S2,
                    3 => Tf32PortableStages::S3,
                    _ => Tf32PortableStages::S4,
                },
            })));
        assert_eq!(tf32_splitk_specs_for(false).count(), 6);
        assert_eq!(tf32_splitk_specs_all().count(), 10);
        assert!(tf32_splitk_specs_for(false).all(|spec| spec.op != ResolvedGemmOp::Tn));
    }

    #[test]
    fn portable_tf32_splitk_lookup_and_partitioning_are_operation_aware() {
        let p4 = Tf32PhysicalRoute::MmaTf32RnaSplitK4(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        assert_eq!(
            tf32_splitk_spec(ResolvedGemmOp::Nn, p4).unwrap().symbol,
            "nn_sm80_mma_tf32_splitk4_m16n32_bk32_s4"
        );
        assert_eq!(
            tf32_splitk_spec(ResolvedGemmOp::Nt, p4).unwrap().symbol,
            "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s4"
        );

        let p8 = TF32_NT_SPLITK8_S3_SPEC.route;
        assert_eq!(
            tf32_splitk_spec(ResolvedGemmOp::Nt, p8).unwrap(),
            &TF32_NT_SPLITK8_S3_SPEC
        );
        assert!(tf32_splitk_spec(ResolvedGemmOp::Nn, p8).is_err());
        assert_eq!(
            tf32_splitk_spec(ResolvedGemmOp::Tn, p8).unwrap(),
            &TF32_TN_SPLITK8_M32_S3_SPEC
        );
        assert_eq!(
            (0..8)
                .map(|partition| tf32_splitk_partition_bounds(1_536, 8, partition).unwrap())
                .collect::<Vec<_>>(),
            [
                (0, 192),
                (192, 384),
                (384, 576),
                (576, 768),
                (768, 960),
                (960, 1_152),
                (1_152, 1_344),
                (1_344, 1_536),
            ]
        );
        assert!(tf32_splitk_partition_bounds(1_536, 8, 8).is_err());
    }

    #[test]
    fn portable_tf32_shared_bytes_are_op_specific() {
        for (op, expected) in [
            (
                ResolvedGemmOp::Nn,
                [55_296, 82_944, 36_864, 55_296, 29_696, 21_504],
            ),
            (
                ResolvedGemmOp::Tn,
                [53_248, 79_872, 36_864, 55_296, 32_768, 24_576],
            ),
            (
                ResolvedGemmOp::Nt,
                [55_296, 82_944, 36_864, 55_296, 27_648, 18_432],
            ),
        ] {
            let actual = tf32_route_specs(ModuleKind::TriadSm80)
                .iter()
                .filter(|spec| spec.op == op)
                .map(|spec| spec.dynamic_shared_bytes)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{op:?} portable TF32 shared bytes");
        }
    }

    #[test]
    fn portable_tf32_schedule_revision_tracks_the_current_layout() {
        assert_eq!(SCHEDULE_REVISION, 8);
        assert_eq!(TF32_PORTABLE_SCHEDULE_REVISION, SCHEDULE_REVISION);
        assert_eq!(TF32_SCHEDULE_REVISION, SCHEDULE_REVISION);
        for module_kind in [
            ModuleKind::TriadSm80,
            ModuleKind::TriadSm90a,
            ModuleKind::TriadSm100,
            ModuleKind::TriadSm120,
        ] {
            assert!(
                tf32_route_specs(module_kind)
                    .iter()
                    .all(|spec| spec.schedule_revision == SCHEDULE_REVISION),
                "{module_kind:?} TF32 schedule revision drifted"
            );
        }
    }

    #[test]
    fn tf32_descriptor_oracles_match_production_builders() {
        for (shared_address, leading_offset, stride_offset, expected_raw, expected_decoded) in [
            (0, 1, 64, 0x4000404000010000, (0, 1, 64)),
            (0, 0, 64, 0x4000404000000000, (0, 0, 64)),
            (0, 256, 64, 0x4000404001000000, (0, 256, 64)),
            (0x1230, 192, 320, 0x4000414000c00123, (0x1230, 192, 320)),
            (0x40000, 0x4001, 0x7fff, 0x40007fff00010000, (0, 1, 0x3fff)),
            (
                0x3fff0,
                0x3fff,
                0x3fff,
                0x40007fff3fff3fff,
                (0x3fff0, 0x3fff, 0x3fff),
            ),
        ] {
            let descriptor =
                build_sm90a_tf32_descriptor(shared_address, leading_offset, stride_offset);
            assert_eq!(descriptor, expected_raw);
            assert_eq!(decode_sm90a_tf32_descriptor(descriptor), expected_decoded);
        }

        for (op, columns, expected) in [
            (ResolvedGemmOp::Nt, 64, 0x08100910),
            (ResolvedGemmOp::Nn, 64, 0x08110910),
            (ResolvedGemmOp::Tn, 64, 0x08118910),
            (ResolvedGemmOp::Nt, 128, 0x08200910),
            (ResolvedGemmOp::Nn, 128, 0x08210910),
            (ResolvedGemmOp::Tn, 128, 0x08218910),
        ] {
            assert_eq!(
                sm100_tf32_instruction_descriptor(op, columns).unwrap(),
                expected,
                "{op:?}/{columns}"
            );
        }
        assert!(sm100_tf32_instruction_descriptor(ResolvedGemmOp::Nn, 32).is_err());
    }

    fn sm120_tf32_sw128_offset(plane_base: u32, logical_row: u32, element: u32) -> u32 {
        let chunk = element / 4;
        let element_in_vector = element & 3;
        let phase = (plane_base / 128) % 8;
        let physical_chunk = chunk ^ ((logical_row + phase) % 8);
        plane_base + logical_row * 128 + physical_chunk * 16 + element_in_vector * 4
    }

    #[test]
    fn sm120_sw128_oracle_matches_production_decode() {
        const CHUNK_PERMUTATIONS: [[u32; 8]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 3, 2, 5, 4, 7, 6],
            [2, 3, 0, 1, 6, 7, 4, 5],
            [3, 2, 1, 0, 7, 6, 5, 4],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 4, 7, 6, 1, 0, 3, 2],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ];

        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            for phase in 0_u32..8 {
                let plane_base = phase * 128;
                for logical_row in 0_u32..32 {
                    let swizzle = ((logical_row + phase) % 8) as usize;
                    let mut addresses = std::collections::BTreeSet::new();
                    for element in 0_u32..32 {
                        let chunk = (element / 4) as usize;
                        let expected = plane_base
                            + logical_row * 128
                            + CHUNK_PERMUTATIONS[swizzle][chunk] * 16
                            + (element & 3) * 4;
                        let actual = sm120_tf32_sw128_offset(plane_base, logical_row, element);
                        assert_eq!(
                            actual, expected,
                            "{op:?}/phase={phase}/row={logical_row}/element={element}"
                        );
                        assert!(addresses.insert(actual - plane_base));
                    }
                    let expected_addresses: std::collections::BTreeSet<_> = (0_u32..32)
                        .map(|element| logical_row * 128 + element * 4)
                        .collect();
                    assert_eq!(
                        addresses, expected_addresses,
                        "{op:?}/phase={phase}/row={logical_row}"
                    );
                }
            }
        }
    }

    fn tf32_epilogue_reference(
        op: ResolvedGemmOp,
        accumulator: f32,
        old_output: f32,
        alpha: f32,
        beta: f32,
    ) -> f32 {
        match op {
            ResolvedGemmOp::Nn => {
                let value = if alpha == 1.0 {
                    accumulator
                } else {
                    alpha * accumulator
                };
                if beta == 0.0 {
                    value
                } else {
                    beta.mul_add(old_output, value)
                }
            }
            ResolvedGemmOp::Tn => alpha.mul_add(accumulator, old_output),
            ResolvedGemmOp::Nt => {
                if alpha == 1.0 {
                    accumulator
                } else {
                    alpha * accumulator
                }
            }
        }
    }

    #[test]
    fn tf32_epilogue_matches_scalar_reference_bits() {
        let just_above_one = f32::from_bits(0x3f800001);
        let negative_rounded_product = f32::from_bits(0xbf800002);
        for (op, accumulator, old_output, alpha, beta, expected) in [
            (
                ResolvedGemmOp::Nn,
                negative_rounded_product,
                just_above_one,
                1.0,
                just_above_one,
                0x28800000,
            ),
            (
                ResolvedGemmOp::Nn,
                just_above_one,
                0.0,
                just_above_one,
                0.0,
                0x3f800002,
            ),
            (
                ResolvedGemmOp::Tn,
                just_above_one,
                negative_rounded_product,
                just_above_one,
                1.0,
                0x28800000,
            ),
            (
                ResolvedGemmOp::Nt,
                just_above_one,
                37.0,
                just_above_one,
                0.0,
                0x3f800002,
            ),
        ] {
            assert_eq!(
                tf32_epilogue_reference(op, accumulator, old_output, alpha, beta).to_bits(),
                expected,
                "{op:?}"
            );
        }

        assert_eq!(
            tf32_epilogue_reference(ResolvedGemmOp::Nt, -0.0, 19.0, 1.0, 0.0).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert!(
            tf32_epilogue_reference(ResolvedGemmOp::Nn, f32::INFINITY, 0.0, 1.0, 0.0).is_infinite()
        );
        assert!(tf32_epilogue_reference(ResolvedGemmOp::Nt, f32::NAN, 0.0, 1.0, 0.0).is_nan());
        for poisoned_old_output in [f32::NAN, f32::INFINITY] {
            assert_eq!(
                tf32_epilogue_reference(
                    ResolvedGemmOp::Nn,
                    negative_rounded_product,
                    poisoned_old_output,
                    1.0,
                    0.0,
                )
                .to_bits(),
                negative_rounded_product.to_bits()
            );
        }
        assert!(
            tf32_epilogue_reference(
                ResolvedGemmOp::Tn,
                f32::INFINITY,
                f32::NEG_INFINITY,
                1.0,
                1.0,
            )
            .is_nan()
        );
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

    fn tf32_sm100_route() -> Tf32PhysicalRoute {
        Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(Tf32Sm100Route {
            tile: Sm100Tile::M128N128,
            stages: Sm100Stages::S2,
            schedule: Sm100Schedule::C4,
        })
    }

    fn tf32_test_binding() -> Tf32MapBinding {
        let target = CudaTarget::new("sm_100a").unwrap();
        let nvrtc_version = (13, 2);
        Tf32MapBinding {
            allocation_domain: AllocationDomain {
                context_handle: 1,
                device_ordinal: 0,
            },
            qualified: Tf32QualifiedModule {
                module_kind: ModuleKind::TriadSm100,
                target,
                artifact: ArtifactIdentity {
                    module_kind: ModuleKind::TriadSm100,
                    artifact_kind: ArtifactKind::Ptx,
                    compile_key: [1; 32],
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
                    compute_capability: (10, 0),
                    multiprocessor_count: 132,
                    target,
                    driver: DriverIdentity {
                        api_version: 13_020,
                        build_sources: 1,
                        build_digest: [7; 32],
                    },
                },
                device_caps: DeviceCaps {
                    compute_capability: (10, 0),
                    nvrtc_version,
                    accepted_target: Some(target),
                    optin_shared_bytes: 228_000,
                    tensor_map_access: true,
                },
                sm120_fma_exclusions: Default::default(),
            },
        }
    }

    fn tf32_request(op: ResolvedGemmOp) -> F32TriadRequest {
        F32TriadRequest {
            op,
            shape: F32TriadShape {
                m: 65,
                k: 127,
                n: 129,
                lda: if op == ResolvedGemmOp::Nt { 132 } else { 128 },
                ldb: 132,
                ldc: if op == ResolvedGemmOp::Nt { 128 } else { 132 },
            },
        }
    }

    fn tf32_operands() -> F32TriadOperands {
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
    fn specialized_tf32_issued_coordinates_include_tail_plane_starts() {
        let routes = [
            (
                Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(Tf32Sm90aRoute {
                    schedule: Sm90aWarpgroupSchedule::Wg1,
                }),
                [
                    [[96, 64], [224, 96]],
                    [[96, 64], [224, 64]],
                    [[128, 64], [128, 0]],
                ],
            ),
            (
                tf32_sm100_route(),
                [
                    [[96, 0], [224, 96]],
                    [[96, 64], [224, 64]],
                    [[128, 0], [128, 0]],
                ],
            ),
            (
                Tf32PhysicalRoute::Sm100Tcgen05Tf32Tma(Tf32Sm100Route {
                    tile: Sm100Tile::M128N64,
                    stages: Sm100Stages::S2,
                    schedule: Sm100Schedule::C4,
                }),
                [
                    [[96, 0], [160, 96]],
                    [[96, 64], [160, 64]],
                    [[128, 0], [128, 64]],
                ],
            ),
            (
                Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
                    tile: Tf32Sm120Tile::M128N64,
                    stages: Tf32Sm120Stages::S2,
                }),
                [
                    [[96, 0], [160, 96]],
                    [[96, 64], [160, 64]],
                    [[128, 0], [128, 64]],
                ],
            ),
            (
                Tf32PhysicalRoute::Sm120TmaMmaTf32Rna(Tf32Sm120Route {
                    tile: Tf32Sm120Tile::M64N128,
                    stages: Tf32Sm120Stages::S2,
                }),
                [
                    [[96, 64], [224, 96]],
                    [[96, 64], [224, 64]],
                    [[128, 64], [128, 0]],
                ],
            ),
        ];
        let operations = [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt];
        for (route, expected_by_op) in routes {
            for (op, expected_maxima) in operations.into_iter().zip(expected_by_op) {
                let layouts =
                    tf32_operand_layouts(tf32_request(op), tf32_operands(), route).unwrap();
                assert_eq!(
                    layouts.map(|layout| layout.issued_coordinate_max),
                    expected_maxima
                );
                for layout in layouts {
                    for axis in 0..2 {
                        let issued = layout.issued_coordinate_max[axis] as u64;
                        let boundary = i32::MAX as u64 - issued;
                        let accepted_origin = if axis == 0 {
                            (boundary, 0)
                        } else {
                            (0, boundary)
                        };
                        validate_tf32_issued_coordinates(layout, accepted_origin).unwrap();
                        let rejected_origin = if axis == 0 {
                            (boundary + 1, 0)
                        } else {
                            (0, boundary + 1)
                        };
                        let error = validate_tf32_issued_coordinates(layout, rejected_origin)
                            .expect_err(
                                "the last specialized TMA issue must remain in the i32 domain",
                            );
                        let axis_name = if axis == 0 { "x" } else { "y" };
                        assert!(
                            error.contains(&format!(
                                "issued {axis_name} coordinate exceeds i32::MAX"
                            )),
                            "{error}"
                        );
                    }
                }
            }
        }
    }

    fn descriptor_with_first_byte(value: u8) -> Tf32TensorMap {
        let mut map = zeroed_tensor_map_sentinel();
        unsafe {
            *std::ptr::from_mut(&mut map.0).cast::<u8>() = value;
        }
        map
    }

    fn zero_reduction_fixture(a: Tf32TensorMap, b: Tf32TensorMap) -> F32PreparedTensorMaps {
        F32PreparedTensorMaps::ZeroReduction {
            data: Box::new(F32ZeroReductionTensorMaps {
                a,
                b,
                request: F32TriadRequest {
                    op: ResolvedGemmOp::Nn,
                    shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (65, 0, 129)),
                },
                route: Some(tf32_sm100_route()),
                binding: Some(tf32_test_binding()),
                format: Tf32TensorMapFormat::Tfloat32,
                revision: ZERO_REDUCTION_MAP_REVISION,
            }),
        }
    }

    #[test]
    fn zero_reduction_identity_excludes_stored_descriptor_bytes() {
        let zero = zeroed_tensor_map_sentinel();
        let canonical = zero_reduction_fixture(zero, zero);
        let poisoned = zero_reduction_fixture(
            descriptor_with_first_byte(0x5a),
            descriptor_with_first_byte(0xa5),
        );

        assert_eq!(canonical.identity_digest(), poisoned.identity_digest());
    }

    #[test]
    fn zero_reduction_validation_rejects_nonzero_sentinels() {
        let zero = zeroed_tensor_map_sentinel();
        zero_reduction_fixture(zero, zero)
            .validate_live_allocations()
            .unwrap();
        let error = zero_reduction_fixture(descriptor_with_first_byte(1), zero)
            .validate_live_allocations()
            .expect_err("nonzero map storage must not pass mapless validation");
        assert!(error.contains("zero-reduction tensor-map sentinel changed"));
    }

    fn encoded_identity_fixture() -> Tf32EncodedMapIdentity {
        let keys = [
            Tf32TensorMapKey {
                base: 0x1000,
                global_dimensions: [127, 65],
                outer_byte_stride: 512,
                box_dimensions: [32, 128],
                format: Tf32TensorMapFormat::Tfloat32,
            },
            Tf32TensorMapKey {
                base: 0x2000,
                global_dimensions: [129, 127],
                outer_byte_stride: 528,
                box_dimensions: [32, 32],
                format: Tf32TensorMapFormat::Tfloat32,
            },
        ];
        let allocations = [
            Sm90aAllocationIdentity {
                allocation_domain: AllocationDomain {
                    context_handle: 1,
                    device_ordinal: 0,
                },
                allocation_base: 0x1000,
                allocation_bytes: 1 << 20,
                offset_bytes: 16,
                required_bytes: 32_000,
                buffer_id: 11,
            },
            Sm90aAllocationIdentity {
                allocation_domain: AllocationDomain {
                    context_handle: 1,
                    device_ordinal: 0,
                },
                allocation_base: 0x2000,
                allocation_bytes: 1 << 20,
                offset_bytes: 32,
                required_bytes: 64_000,
                buffer_id: 22,
            },
        ];
        Tf32EncodedMapIdentity {
            revision: TF32_TENSOR_MAP_REVISION,
            maps: [descriptor_with_first_byte(1), descriptor_with_first_byte(2)],
            keys,
            allocations,
            origins: Tf32TensorOrigins {
                a_x: 4,
                a_y: 1,
                b_x: 8,
                b_y: 2,
            },
            format: Tf32TensorMapFormat::Tfloat32,
            route: tf32_sm100_route(),
        }
    }

    fn encoded_prepared_fixture(identity: Tf32EncodedMapIdentity) -> F32PreparedTensorMaps {
        F32PreparedTensorMaps::Encoded {
            data: Box::new(F32EncodedTensorMaps {
                a: identity.maps[0],
                b: identity.maps[1],
                keys: identity.keys,
                request: tf32_request(ResolvedGemmOp::Nn),
                route: identity.route,
                binding: tf32_test_binding(),
                allocations: identity.allocations,
                origins: identity.origins,
                format: identity.format,
            }),
        }
    }

    fn encoded_identity_digest(identity: Tf32EncodedMapIdentity) -> [u8; 32] {
        append_tf32_encoded_map_identity(
            crate::mamba_ssm::gpu::kernel_identity::FramedSha256::new(b"tf32-encoded-map-test.v1"),
            identity,
        )
        .finish()
    }

    fn allocation_identity_digest(identity: Sm90aAllocationIdentity) -> [u8; 32] {
        identity
            .append_digest(crate::mamba_ssm::gpu::kernel_identity::FramedSha256::new(
                b"allocation-identity-test.v1",
            ))
            .finish()
    }

    #[test]
    fn allocation_identity_digest_excludes_raw_context_and_tracks_logical_identity() {
        let identity = encoded_identity_fixture().allocations[0];
        let expected = allocation_identity_digest(identity);
        assert_eq!(
            allocation_identity_digest(Sm90aAllocationIdentity {
                allocation_domain: AllocationDomain {
                    context_handle: identity.allocation_domain.context_handle + 1,
                    ..identity.allocation_domain
                },
                ..identity
            }),
            expected,
            "a raw CUDA context address must not enter a stable allocation digest"
        );
        let changes = [
            Sm90aAllocationIdentity {
                allocation_domain: AllocationDomain {
                    device_ordinal: identity.allocation_domain.device_ordinal + 1,
                    ..identity.allocation_domain
                },
                ..identity
            },
            Sm90aAllocationIdentity {
                allocation_base: identity.allocation_base + 1,
                ..identity
            },
            Sm90aAllocationIdentity {
                allocation_bytes: identity.allocation_bytes + 1,
                ..identity
            },
            Sm90aAllocationIdentity {
                offset_bytes: identity.offset_bytes + 1,
                ..identity
            },
            Sm90aAllocationIdentity {
                required_bytes: identity.required_bytes + 1,
                ..identity
            },
            Sm90aAllocationIdentity {
                buffer_id: identity.buffer_id + 1,
                ..identity
            },
        ];
        for changed in changes {
            assert_ne!(allocation_identity_digest(changed), expected);
        }
    }

    #[test]
    fn physical_allocation_digest_excludes_addresses_and_tracks_subview_generation() {
        let identity = encoded_identity_fixture().allocations[0];
        let expected = identity.physical_digest();
        assert_eq!(
            Sm90aAllocationIdentity {
                allocation_domain: AllocationDomain {
                    context_handle: identity.allocation_domain.context_handle + 1,
                    ..identity.allocation_domain
                },
                ..identity
            }
            .physical_digest(),
            expected,
            "a raw CUDA context address must not enter stable physical evidence"
        );
        assert_eq!(
            Sm90aAllocationIdentity {
                allocation_base: identity.allocation_base + 0x10_0000,
                ..identity
            }
            .physical_digest(),
            expected
        );
        for changed in [
            Sm90aAllocationIdentity {
                offset_bytes: identity.offset_bytes + 2,
                ..identity
            },
            Sm90aAllocationIdentity {
                required_bytes: identity.required_bytes + 2,
                ..identity
            },
            Sm90aAllocationIdentity {
                buffer_id: identity.buffer_id + 1,
                ..identity
            },
        ] {
            assert_ne!(changed.physical_digest(), expected);
        }
    }

    #[test]
    fn physical_f32_resources_bind_roles_nullness_and_scratch_provenance() {
        let first = encoded_identity_fixture().allocations[0];
        let second = Sm90aAllocationIdentity {
            allocation_base: first.allocation_base + 0x20_0000,
            buffer_id: first.buffer_id + 1,
            ..first
        };
        let baseline = F32LaunchResourceSnapshot {
            output: first,
            bias: None,
            inputs: Some([first, second]),
            split_scratch: Some(first),
            transpose_scratch: Some(second),
            coordination_scratch: Some(first),
        };
        let expected = baseline.physical_digest();
        for changed in [
            F32LaunchResourceSnapshot {
                inputs: Some([second, first]),
                ..baseline
            },
            F32LaunchResourceSnapshot {
                inputs: None,
                ..baseline
            },
            F32LaunchResourceSnapshot {
                bias: Some(first),
                ..baseline
            },
            F32LaunchResourceSnapshot {
                split_scratch: None,
                ..baseline
            },
            F32LaunchResourceSnapshot {
                transpose_scratch: None,
                ..baseline
            },
            F32LaunchResourceSnapshot {
                coordination_scratch: None,
                ..baseline
            },
            F32LaunchResourceSnapshot {
                split_scratch: Some(Sm90aAllocationIdentity {
                    buffer_id: first.buffer_id + 9,
                    ..first
                }),
                ..baseline
            },
            F32LaunchResourceSnapshot {
                split_scratch: Some(Sm90aAllocationIdentity {
                    offset_bytes: first.offset_bytes + 4,
                    ..first
                }),
                ..baseline
            },
            F32LaunchResourceSnapshot {
                transpose_scratch: Some(Sm90aAllocationIdentity {
                    buffer_id: second.buffer_id + 9,
                    ..second
                }),
                ..baseline
            },
            F32LaunchResourceSnapshot {
                transpose_scratch: Some(Sm90aAllocationIdentity {
                    offset_bytes: second.offset_bytes + 4,
                    ..second
                }),
                ..baseline
            },
            F32LaunchResourceSnapshot {
                coordination_scratch: Some(Sm90aAllocationIdentity {
                    buffer_id: first.buffer_id + 11,
                    ..first
                }),
                ..baseline
            },
        ] {
            assert_ne!(changed.physical_digest(), expected);
        }
    }

    #[test]
    fn f32_resource_epoch_tracks_external_owners_but_not_context_scratch() {
        let domain = AllocationDomain {
            context_handle: 0x2201,
            device_ordinal: 0,
        };
        let identity = |base, offset, required, buffer_id| Sm90aAllocationIdentity {
            allocation_domain: domain,
            allocation_base: base,
            allocation_bytes: 4096,
            offset_bytes: offset,
            required_bytes: required,
            buffer_id,
        };
        let output = register_managed_allocation_range(domain.context_handle, 0x50_0000, 4096)
            .expect("register output");
        let a = register_managed_allocation_range(domain.context_handle, 0x60_0000, 4096)
            .expect("register A");
        let b = register_managed_allocation_range(domain.context_handle, 0x70_0000, 4096)
            .expect("register B");
        let resources = F32LaunchResourceSnapshot {
            output: identity(0x50_0000, 128, 512, 1),
            bias: None,
            inputs: Some([
                identity(0x60_0000, 0, 1024, 2),
                identity(0x70_0000, 64, 2048, 3),
            ]),
            split_scratch: Some(identity(0x80_0000, 0, 4096, 4)),
            transpose_scratch: None,
            coordination_scratch: Some(identity(0x90_0000, 0, 4096, 5)),
        };
        let stamp = resources
            .managed_epoch()
            .expect("all external resources are managed");

        assert!(stamp.is_current());
        drop(a);
        assert!(!stamp.is_current());
        drop((output, b));
    }

    #[test]
    fn encoded_tf32_map_identity_frames_every_graph_bound_resource() {
        let identity = encoded_identity_fixture();
        let expected = encoded_identity_digest(identity);
        let independently_framed = match std::mem::align_of::<Tf32TensorMap>() {
            64 => [
                19, 209, 35, 117, 240, 122, 6, 14, 164, 7, 107, 111, 251, 130, 246, 150, 204, 91,
                75, 158, 173, 20, 98, 38, 66, 74, 26, 62, 191, 152, 255, 48,
            ],
            128 => [
                147, 96, 49, 185, 154, 29, 14, 4, 210, 164, 178, 208, 31, 14, 109, 171, 111, 126,
                154, 240, 246, 2, 219, 27, 222, 128, 166, 16, 164, 242, 182, 61,
            ],
            alignment => panic!("unexpected CUtensorMap alignment {alignment}"),
        };
        assert_eq!(expected, independently_framed);

        let mut revision = identity;
        revision.revision += 1;
        assert_ne!(encoded_identity_digest(revision), expected);

        let mut format = identity;
        format.format = Tf32TensorMapFormat::Uint32;
        assert_ne!(encoded_identity_digest(format), expected);

        let mut origin = identity;
        origin.origins.b_y += 1;
        assert_ne!(encoded_identity_digest(origin), expected);

        let mut allocation = identity;
        allocation.allocations[1].buffer_id += 1;
        assert_ne!(encoded_identity_digest(allocation), expected);

        let mut allocation_context = identity;
        allocation_context.allocations[0]
            .allocation_domain
            .context_handle += 1;
        assert_eq!(
            encoded_identity_digest(allocation_context),
            expected,
            "a raw CUDA context address is not graph identity"
        );

        let mut allocation_device = identity;
        allocation_device.allocations[0]
            .allocation_domain
            .device_ordinal += 1;
        assert_ne!(encoded_identity_digest(allocation_device), expected);

        let mut key = identity;
        key.keys[0].global_dimensions[1] += 1;
        assert_ne!(encoded_identity_digest(key), expected);

        let mut descriptor = identity;
        descriptor.maps[0] = descriptor_with_first_byte(3);
        assert_ne!(encoded_identity_digest(descriptor), expected);

        let mut order = identity;
        order.maps.swap(0, 1);
        order.keys.swap(0, 1);
        order.allocations.swap(0, 1);
        assert_ne!(encoded_identity_digest(order), expected);
    }

    #[test]
    fn prepared_encoded_identity_tracks_stored_descriptor_bytes() {
        let identity = encoded_identity_fixture();
        let expected = encoded_prepared_fixture(identity).identity_digest();
        let mut changed = identity;
        changed.maps[1] = descriptor_with_first_byte(0x7f);

        assert_ne!(
            encoded_prepared_fixture(changed).identity_digest(),
            expected
        );
    }

    #[test]
    fn physical_encoded_identity_excludes_addresses_and_descriptor_storage() {
        let identity = encoded_identity_fixture();
        let prepared = encoded_prepared_fixture(identity);
        let expected = prepared.physical_identity_digest();
        let graph_identity = prepared.identity_digest();

        for changed in [
            {
                let mut changed = identity;
                changed.maps[0] = descriptor_with_first_byte(0x7f);
                changed
            },
            {
                let mut changed = identity;
                changed.keys[0].base += 0x10_0000;
                changed
            },
            {
                let mut changed = identity;
                changed.allocations[0].allocation_base += 0x10_0000;
                changed
            },
        ] {
            let changed = encoded_prepared_fixture(changed);
            assert_eq!(changed.physical_identity_digest(), expected);
            assert_ne!(changed.identity_digest(), graph_identity);
        }

        let mut changed_context = identity;
        changed_context.allocations[0]
            .allocation_domain
            .context_handle += 1;
        let changed_context = encoded_prepared_fixture(changed_context);
        assert_eq!(changed_context.physical_identity_digest(), expected);
        assert_eq!(changed_context.identity_digest(), graph_identity);

        for changed in [
            {
                let mut changed = identity;
                changed.keys[1].box_dimensions[0] += 1;
                changed
            },
            {
                let mut changed = identity;
                changed.origins.b_y += 1;
                changed
            },
            {
                let mut changed = identity;
                changed.allocations[1].offset_bytes += 4;
                changed
            },
            {
                let mut changed = identity;
                changed.allocations[1].required_bytes += 4;
                changed
            },
            {
                let mut changed = identity;
                changed.allocations[1].buffer_id += 1;
                changed
            },
        ] {
            assert_ne!(
                encoded_prepared_fixture(changed).physical_identity_digest(),
                expected
            );
        }
    }

    #[test]
    fn sm120_inventory_covers_all_ninety_six_physical_routes_once() {
        assert_eq!(SM120_KERNEL_SPECS.len(), 96);
        let mut symbols = std::collections::BTreeSet::new();
        for spec in SM120_KERNEL_SPECS {
            assert!(symbols.insert(spec.symbol), "duplicate {}", spec.symbol);
            let op_prefix = match spec.op {
                Sm120Op::Nn => "nn_",
                Sm120Op::Tn => "tn_",
                Sm120Op::Nt => "nt_",
            };
            assert!(spec.symbol.starts_with(op_prefix), "{}", spec.symbol);
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
                256,
                32_896,
            ),
            (
                Sm120Tile::M128N128,
                Sm120Bk::Bk32,
                Sm120Stages::S3,
                256,
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
