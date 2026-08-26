use super::super::blas::TypedPtr;
use super::super::dtype::WeightDtype;
use crate::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, CompilerIdentity, DeviceIdentity, FramedSha256, ModuleKind, Sha256Digest,
};
use cudarc::driver::{DeviceRepr, sys};

pub(super) type CUptr = cudarc::driver::sys::CUdeviceptr;

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
        name: &str,
    ) -> Result<Self, String> {
        if pointer == 0 || required_bytes == 0 {
            return Err(format!("SM90a {name} allocation must be non-empty"));
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
                "query SM90a {name} allocation identity: {result:?}"
            ));
        }
        let context_handle = context as usize;
        if context_handle != expected_context {
            return Err(format!(
                "SM90a {name} allocation belongs to a different CUDA context"
            ));
        }
        let allocation_bytes = u64::try_from(allocation_bytes)
            .map_err(|_| format!("SM90a {name} allocation size exceeds u64::MAX"))?;
        let offset_bytes = pointer
            .checked_sub(allocation_base)
            .ok_or_else(|| format!("SM90a {name} pointer precedes its allocation"))?;
        let end = offset_bytes
            .checked_add(required_bytes)
            .ok_or_else(|| format!("SM90a {name} byte range overflows u64"))?;
        if end > allocation_bytes {
            return Err(format!(
                "SM90a {name} requires {required_bytes} bytes at offset {offset_bytes}, allocation has {allocation_bytes} bytes"
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
    fn validate(self) -> Result<(), String> {
        if self.base == 0 || !self.base.is_multiple_of(16) {
            return Err("SM90a tensor-map base must be non-null and 16-byte aligned".into());
        }
        if self.global_dimensions.contains(&0) {
            return Err("SM90a tensor-map dimensions must be positive".into());
        }
        if self.outer_byte_stride == 0
            || !self.outer_byte_stride.is_multiple_of(16)
            || self.outer_byte_stride >= (1_u64 << 40)
        {
            return Err(
                "SM90a tensor-map outer byte stride must be a positive multiple of 16 below 2^40"
                    .into(),
            );
        }
        let row_bytes = self.global_dimensions[0]
            .checked_mul(2)
            .ok_or_else(|| "SM90a tensor-map row width overflows u64".to_string())?;
        if self.outer_byte_stride < row_bytes {
            return Err("SM90a tensor-map outer stride is smaller than its inner dimension".into());
        }
        if self.box_dimensions.contains(&0) || self.box_dimensions.iter().any(|&dim| dim > 256) {
            return Err("SM90a tensor-map box dimensions must be in 1..=256".into());
        }
        let inner_bytes = self.box_dimensions[0]
            .checked_mul(2)
            .ok_or_else(|| "SM90a tensor-map box width overflows u32".to_string())?;
        if inner_bytes > 128 || !inner_bytes.is_multiple_of(16) {
            return Err(
                "SM90a SW128 inner box span must be a multiple of 16 bytes and at most 128 bytes"
                    .into(),
            );
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
    fn encode(key: Sm90aTensorMapKey) -> Result<Self, String> {
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
        key.validate()?;
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
        a: Sm90aTensorMap::encode(keys[0])?,
        b: Sm90aTensorMap::encode(keys[1])?,
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
        Sm90aAllocationIdentity::query(keys[0].base, required(keys[0])?, context_handle, "A")?,
        Sm90aAllocationIdentity::query(keys[1].base, required(keys[1])?, context_handle, "B")?,
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
        let bias =
            Sm90aAllocationIdentity::query(operands.bias_ptr, bias_bytes, context_handle, "bias")?;
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

#[cfg(test)]
mod tests {
    use super::{
        GemmDims, Sm90aMapRequest, Sm90aOp, Sm90aShape, checked_grid_product, checked_u32,
        sm90a_tensor_map_keys, validate_bias_preseed,
    };
    use crate::mamba_ssm::gpu::dtype::WeightDtype;

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
