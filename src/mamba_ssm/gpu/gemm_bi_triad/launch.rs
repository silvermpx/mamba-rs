use super::super::buffers::{
    GpuBuffer, ManagedAllocationEpochStamp, managed_allocation_epoch_for_ranges,
};
use super::super::context::{GpuCtx, HalfTriadPolicy};
use super::super::kernels::MambaKernels as GpuKernels;
use super::contract::*;
use super::dispatch::*;
use crate::mamba_ssm::gpu::kernel_identity::{
    FramedSha256, GemmPolicy, GemmRouteIdentity, ModuleKind, NoPhysicalObserver,
    PhysicalCudaLaunchError, PhysicalGemmBackend, PhysicalLaunchObservation,
    PhysicalLaunchObserver, PolicyDtype, RecordingPhysicalObserver, ResolvedGemmLaunchSet,
    ResolvedGemmLaunchSetBuilder, ResolvedGemmOp, ResolvedGemmRoute, ResolvedInstructionFamily,
    ResolvedInstructionShape, ResolvedKernelLaunch, ResolvedNumericContract,
    ResolvedOperandConversion, ResolvedOutputOwnership, ResolvedPhysicalKernelLaunch,
    SCHEDULE_REVISION, Sha256Digest, TUNING_TABLE_REVISION, build_resolved_gemm_launch_set,
    build_zero_reduction_route_identity, enqueue_prepared_physical_launch,
    enqueue_with_physical_observation, prepare_recording_physical_observer,
    resolve_physical_launch_observation,
};
use cudarc::driver::{
    CudaFunction, CudaStream, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg,
};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(in crate::mamba_ssm::gpu) struct PhysicalArgumentRange {
    pub(crate) pointer: CUptr,
    pub(crate) required_bytes: u64,
}

pub(in crate::mamba_ssm::gpu) fn prepare_physical_observer(
    ctx: &GpuCtx,
    capacity: usize,
    ranges: &[PhysicalArgumentRange],
) -> Result<RecordingPhysicalObserver, String> {
    let allocation_domain =
        validated_allocation_domain(&ctx.stream, &ctx.kernels, "physical half launch")?;
    let mut argument_allocations = Vec::new();
    argument_allocations
        .try_reserve_exact(ranges.len())
        .map_err(|error| format!("reserve physical argument identities: {error}"))?;
    for range in ranges {
        argument_allocations.push(Sm90aAllocationIdentity::query(
            range.pointer,
            range.required_bytes,
            allocation_domain,
            "physical half launch",
            "argument",
        )?);
    }
    if argument_allocations.is_empty() {
        return Err("physical launch observer requires prevalidated arguments".into());
    }
    let mut managed_ranges = Vec::new();
    managed_ranges
        .try_reserve_exact(ranges.len())
        .map_err(|error| format!("reserve physical allocation liveness ranges: {error}"))?;
    managed_ranges.extend(
        ranges
            .iter()
            .map(|range| (range.pointer, range.required_bytes)),
    );
    let managed_epoch =
        managed_allocation_epoch_for_ranges(allocation_domain.context_handle, &managed_ranges);
    let argument_allocations = argument_allocations.into_boxed_slice();
    let observer = prepare_recording_physical_observer(
        ctx,
        capacity,
        managed_epoch,
        move |pointer, required_bytes| {
            argument_allocations
                .iter()
                .find_map(|identity| identity.physical_subrange(pointer, required_bytes))
                .map(Sm90aAllocationIdentity::physical_digest)
                .ok_or_else(|| {
                    format!(
                        "physical launch argument range is not covered by its prevalidated allocation: pointer={pointer:#x}, bytes={required_bytes}"
                    )
                })
        },
    )?;
    observer.validate_start(capacity)?;
    Ok(observer)
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Sm80Tf32KernelParams {
    alpha: f32,
    beta: f32,
    m: i32,
    k: i32,
    n: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for Sm80Tf32KernelParams {}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct SgbZeroReductionParams {
    alpha: f32,
    beta: f32,
    m: i32,
    k: i32,
    n: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for SgbZeroReductionParams {}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct SgbNnM64N64Params {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for SgbNnM64N64Params {}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Sm90aTf32KernelParams {
    a_x: i32,
    a_y: i32,
    b_x: i32,
    b_y: i32,
    alpha: f32,
    beta: f32,
    m: i32,
    k: i32,
    n: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for Sm90aTf32KernelParams {}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Sm100KernelParams {
    a_x: i32,
    a_y: i32,
    b_x: i32,
    b_y: i32,
    alpha: f32,
    beta: f32,
    m: i32,
    k: i32,
    n: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for Sm100KernelParams {}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Sm120KernelParams {
    a_x: i32,
    a_y: i32,
    b_x: i32,
    b_y: i32,
    alpha: f32,
    beta: f32,
    m: i32,
    k: i32,
    n: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for Sm120KernelParams {}
unsafe impl DeviceRepr for Sm120FmaKernelParams {}

pub(super) const GEMM_BI_ZERO_REDUCTION_PARAMS_SIZE: usize =
    std::mem::size_of::<SgbZeroReductionParams>();

pub(super) const fn tf32_kernel_params_size(module_kind: ModuleKind) -> Option<usize> {
    match module_kind {
        ModuleKind::TriadSm80 => Some(std::mem::size_of::<Sm80Tf32KernelParams>()),
        ModuleKind::TriadSm90a => Some(std::mem::size_of::<Sm90aTf32KernelParams>()),
        ModuleKind::TriadSm100 => Some(std::mem::size_of::<Sm100KernelParams>()),
        ModuleKind::TriadSm120 => Some(std::mem::size_of::<Sm120KernelParams>()),
        _ => None,
    }
}

impl Sm100KernelParams {
    fn into_words(self) -> [u32; 10] {
        [
            self.a_x as u32,
            self.a_y as u32,
            self.b_x as u32,
            self.b_y as u32,
            self.alpha.to_bits(),
            self.beta.to_bits(),
            self.m as u32,
            self.k as u32,
            self.n as u32,
            self.ldc as u32,
        ]
    }

    fn from_words(words: [u32; 10]) -> Self {
        Self {
            a_x: words[0] as i32,
            a_y: words[1] as i32,
            b_x: words[2] as i32,
            b_y: words[3] as i32,
            alpha: f32::from_bits(words[4]),
            beta: f32::from_bits(words[5]),
            m: words[6] as i32,
            k: words[7] as i32,
            n: words[8] as i32,
            ldc: words[9] as i32,
        }
    }
}

impl Sm120KernelParams {
    fn into_words(self) -> [u32; 10] {
        [
            self.a_x as u32,
            self.a_y as u32,
            self.b_x as u32,
            self.b_y as u32,
            self.alpha.to_bits(),
            self.beta.to_bits(),
            self.m as u32,
            self.k as u32,
            self.n as u32,
            self.ldc as u32,
        ]
    }

    fn from_words(words: [u32; 10]) -> Self {
        Self {
            a_x: words[0] as i32,
            a_y: words[1] as i32,
            b_x: words[2] as i32,
            b_y: words[3] as i32,
            alpha: f32::from_bits(words[4]),
            beta: f32::from_bits(words[5]),
            m: words[6] as i32,
            k: words[7] as i32,
            n: words[8] as i32,
            ldc: words[9] as i32,
        }
    }
}

fn prepare_f32_maps_with<Plan, Encode>(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: Tf32PhysicalRoute,
    binding: Tf32MapBinding,
    plan: Plan,
    encode: Encode,
) -> Result<F32PreparedTensorMaps, String>
where
    Plan: FnOnce(
        F32TriadRequest,
        F32TriadOperands,
        Tf32PhysicalRoute,
    ) -> Result<Tf32TensorMapPlan, String>,
    Encode: FnOnce(Tf32TensorMapPlan) -> Result<F32PreparedTensorMaps, String>,
{
    if request.shape.reduction(request.op) == 0 {
        let format = match route {
            Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_)
            | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_) => Tf32TensorMapFormat::Uint32V1,
            _ => Tf32TensorMapFormat::Tfloat32V1,
        };
        return Ok(F32PreparedTensorMaps::zero_reduction(
            request,
            Some(route),
            Some(binding),
            format,
        ));
    }
    encode(plan(request, operands, route)?)
}

macro_rules! abi_const_assert_eq {
    ($left:expr, $right:expr) => {
        const _: [(); $right] = [(); $left];
    };
}

abi_const_assert_eq!(std::mem::size_of::<Sm80Tf32KernelParams>(), 32);
abi_const_assert_eq!(std::mem::align_of::<Sm80Tf32KernelParams>(), 4);
abi_const_assert_eq!(std::mem::offset_of!(Sm80Tf32KernelParams, alpha), 0);
abi_const_assert_eq!(std::mem::offset_of!(Sm80Tf32KernelParams, beta), 4);
abi_const_assert_eq!(std::mem::offset_of!(Sm80Tf32KernelParams, m), 8);
abi_const_assert_eq!(std::mem::offset_of!(Sm80Tf32KernelParams, k), 12);
abi_const_assert_eq!(std::mem::offset_of!(Sm80Tf32KernelParams, n), 16);
abi_const_assert_eq!(std::mem::offset_of!(Sm80Tf32KernelParams, lda), 20);
abi_const_assert_eq!(std::mem::offset_of!(Sm80Tf32KernelParams, ldb), 24);
abi_const_assert_eq!(std::mem::offset_of!(Sm80Tf32KernelParams, ldc), 28);
abi_const_assert_eq!(std::mem::size_of::<SgbZeroReductionParams>(), 32);
abi_const_assert_eq!(std::mem::align_of::<SgbZeroReductionParams>(), 4);
abi_const_assert_eq!(std::mem::offset_of!(SgbZeroReductionParams, alpha), 0);
abi_const_assert_eq!(std::mem::offset_of!(SgbZeroReductionParams, beta), 4);
abi_const_assert_eq!(std::mem::offset_of!(SgbZeroReductionParams, m), 8);
abi_const_assert_eq!(std::mem::offset_of!(SgbZeroReductionParams, k), 12);
abi_const_assert_eq!(std::mem::offset_of!(SgbZeroReductionParams, n), 16);
abi_const_assert_eq!(std::mem::offset_of!(SgbZeroReductionParams, lda), 20);
abi_const_assert_eq!(std::mem::offset_of!(SgbZeroReductionParams, ldb), 24);
abi_const_assert_eq!(std::mem::offset_of!(SgbZeroReductionParams, ldc), 28);
abi_const_assert_eq!(std::mem::size_of::<SgbNnM64N64Params>(), 32);
abi_const_assert_eq!(std::mem::align_of::<SgbNnM64N64Params>(), 4);
abi_const_assert_eq!(std::mem::offset_of!(SgbNnM64N64Params, alpha), 0);
abi_const_assert_eq!(std::mem::offset_of!(SgbNnM64N64Params, beta), 4);
abi_const_assert_eq!(std::mem::offset_of!(SgbNnM64N64Params, m), 8);
abi_const_assert_eq!(std::mem::offset_of!(SgbNnM64N64Params, n), 12);
abi_const_assert_eq!(std::mem::offset_of!(SgbNnM64N64Params, k), 16);
abi_const_assert_eq!(std::mem::offset_of!(SgbNnM64N64Params, lda), 20);
abi_const_assert_eq!(std::mem::offset_of!(SgbNnM64N64Params, ldb), 24);
abi_const_assert_eq!(std::mem::offset_of!(SgbNnM64N64Params, ldc), 28);
abi_const_assert_eq!(std::mem::size_of::<Sm90aTf32KernelParams>(), 40);
abi_const_assert_eq!(std::mem::align_of::<Sm90aTf32KernelParams>(), 4);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, a_x), 0);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, a_y), 4);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, b_x), 8);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, b_y), 12);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, alpha), 16);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, beta), 20);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, m), 24);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, k), 28);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, n), 32);
abi_const_assert_eq!(std::mem::offset_of!(Sm90aTf32KernelParams, ldc), 36);
abi_const_assert_eq!(std::mem::size_of::<Sm100KernelParams>(), 40);
abi_const_assert_eq!(std::mem::align_of::<Sm100KernelParams>(), 4);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, a_x), 0);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, a_y), 4);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, b_x), 8);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, b_y), 12);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, alpha), 16);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, beta), 20);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, m), 24);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, k), 28);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, n), 32);
abi_const_assert_eq!(std::mem::offset_of!(Sm100KernelParams, ldc), 36);
abi_const_assert_eq!(std::mem::size_of::<Sm120KernelParams>(), 40);
abi_const_assert_eq!(std::mem::align_of::<Sm120KernelParams>(), 4);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, a_x), 0);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, a_y), 4);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, b_x), 8);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, b_y), 12);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, alpha), 16);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, beta), 20);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, m), 24);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, k), 28);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, n), 32);
abi_const_assert_eq!(std::mem::offset_of!(Sm120KernelParams, ldc), 36);

#[derive(Clone, Copy)]
enum PreparedTf32Params {
    Sm80(Sm80Tf32KernelParams),
    Sm90a(Sm90aTf32KernelParams),
    Sm100(Sm100KernelParams),
    Sm120(Sm120KernelParams),
    Sm120Fma(Sm120FmaKernelParams),
}

#[derive(Clone, Copy)]
struct Tf32RawLaunch<'a> {
    operands: F32TriadOperands,
    route: Tf32PhysicalRoute,
    maps: Option<&'a F32PreparedTensorMaps>,
    params: PreparedTf32Params,
    config: LaunchConfig,
    zero_reduction: bool,
    symbol: &'static str,
    observation: Option<PhysicalLaunchObservation>,
    /// The slab and flag buffers of a stream-K route; a single-CTA
    /// qualification probe deals whole tiles only and passes none.
    streamk: Option<Tf32StreamKWorkspace>,
}

#[derive(Clone, Copy, Debug)]
struct Tf32StreamKWorkspace {
    // Context-owned buffers shared with the split-K launches; the recorded
    // stream gate keeps every use on one ordered CUDA stream.
    partial: CUptr,
    flags: CUptr,
}

#[derive(Clone, Copy, Debug)]
struct Tf32StreamKLaunchPlan {
    grid: u32,
    partial_elements: usize,
    flag_elements: usize,
}

/// The persistent grid of a stream-K launch and the workspace it needs. The
/// kernel waits on lower CTAs, so the grid is one CTA per multiprocessor,
/// which the device keeps resident together; the dealing of units depends on
/// the grid, so the same device gives the same bits.
fn tf32_streamk_launch_plan(
    request: F32TriadRequest,
    spec: &Tf32KernelSpec,
    multiprocessor_count: u32,
) -> Result<Tf32StreamKLaunchPlan, String> {
    request.shape.validate(request.op)?;
    if request.op != spec.op {
        return Err(format!(
            "stream-K TF32 specification {:?} does not match {:?}",
            spec.op, request.op
        ));
    }
    if multiprocessor_count == 0 {
        return Err("stream-K TF32 launch requires at least one multiprocessor".into());
    }
    let rows = request.shape.output_rows(request.op);
    let columns = request.shape.output_columns(request.op);
    let tiles = rows
        .div_ceil(spec.tile.0 as usize)
        .checked_mul(columns.div_ceil(spec.tile.1 as usize))
        .ok_or_else(|| invalid_gemm_dimensions("stream-K TF32 tile count overflows usize"))?;
    let k_tiles = request
        .shape
        .reduction(request.op)
        .div_ceil(spec.bk as usize)
        .max(1);
    let units = tiles
        .checked_mul(k_tiles)
        .ok_or_else(|| invalid_gemm_dimensions("stream-K TF32 unit count overflows usize"))?;
    if units > i32::MAX as usize {
        return Err(invalid_gemm_dimensions(
            "stream-K TF32 unit count exceeds the kernel's 32-bit range",
        ));
    }
    let grid = multiprocessor_count;
    let partial_elements = (grid as usize)
        .checked_mul(SM120_TF32_STREAMK_SLOTS_PER_CTA)
        .and_then(|slots| slots.checked_mul(SM120_TF32_STREAMK_SLAB_FLOATS))
        .ok_or_else(|| invalid_gemm_dimensions("stream-K TF32 slab extent overflows usize"))?;
    if partial_elements > SPLITK_SCRATCH_CAP {
        return Err(invalid_gemm_dimensions(
            "stream-K TF32 slabs exceed the fixed workspace",
        ));
    }
    let flag_elements = (grid as usize) * SM120_TF32_STREAMK_SLOTS_PER_CTA;
    if flag_elements > TF32_SPLITK_COUNTER_CAP {
        return Err(invalid_gemm_dimensions(
            "stream-K TF32 flags exceed the fixed counter workspace",
        ));
    }
    Ok(Tf32StreamKLaunchPlan {
        grid,
        partial_elements,
        flag_elements,
    })
}

enum PreparedF32Kind {
    Scalar(ScalarDispatchPlan),
    ScalarZero {
        maps: F32PreparedTensorMaps,
        params: SgbZeroReductionParams,
        config: cudarc::driver::LaunchConfig,
    },
    Tf32 {
        route: Tf32PhysicalRoute,
        maps: Option<F32PreparedTensorMaps>,
        params: PreparedTf32Params,
        config: cudarc::driver::LaunchConfig,
    },
    Tf32SplitK {
        route: Tf32PhysicalRoute,
        params: Sm80Tf32KernelParams,
        plan: Tf32SplitKLaunchPlan,
        workspace: Tf32SplitKWorkspace,
    },
    Tf32StreamK {
        route: Tf32PhysicalRoute,
        maps: F32PreparedTensorMaps,
        params: PreparedTf32Params,
        config: cudarc::driver::LaunchConfig,
        plan: Tf32StreamKLaunchPlan,
        workspace: Tf32StreamKWorkspace,
    },
}

pub(in crate::mamba_ssm::gpu) struct PreparedF32TriadLaunch {
    context_token: u64,
    stream_token: usize,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    resources: F32LaunchResourceSnapshot,
    managed_epoch: Option<ManagedAllocationEpochStamp>,
    routes: Box<[ResolvedGemmRoute]>,
    resolved_launch_set: ResolvedGemmLaunchSet,
    kind: PreparedF32Kind,
}

impl PreparedF32TriadLaunch {
    pub(in crate::mamba_ssm::gpu) fn physical_graph_request(&self) -> F32TriadRequest {
        self.request
    }

    pub(in crate::mamba_ssm::gpu) fn physical_graph_operands(&self) -> F32TriadOperands {
        self.operands
    }

    pub(in crate::mamba_ssm::gpu) fn physical_graph_is_direct(&self) -> bool {
        matches!(
            self.kind,
            PreparedF32Kind::ScalarZero { .. }
                | PreparedF32Kind::Tf32 { .. }
                | PreparedF32Kind::Tf32SplitK { .. }
                | PreparedF32Kind::Tf32StreamK { .. }
        )
    }

    pub(in crate::mamba_ssm::gpu) fn physical_graph_launch_count(&self) -> usize {
        self.routes.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum F32PreparedSelection {
    Automatic,
    ExactScalar,
    Forced(Tf32PhysicalRoute),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PreparedF32Key {
    context_token: u64,
    policy: GemmPolicy,
    selection: F32PreparedSelection,
    request: F32TriadRequest,
    output: CUptr,
    a: CUptr,
    b: CUptr,
    bias: Option<CUptr>,
    alpha_bits: u32,
    beta_bits: u32,
}

impl PreparedF32Key {
    fn new(
        context_token: u64,
        policy: GemmPolicy,
        selection: F32PreparedSelection,
        request: F32TriadRequest,
        operands: F32TriadOperands,
    ) -> Self {
        Self {
            context_token,
            policy,
            selection,
            request,
            output: operands.output,
            a: operands.a,
            b: operands.b,
            bias: operands.bias,
            alpha_bits: operands.alpha.to_bits(),
            beta_bits: operands.beta.to_bits(),
        }
    }
}

#[derive(Default)]
pub(crate) struct F32PreparedLaunchCache {
    entries: HashMap<PreparedF32Key, Box<PreparedF32TriadLaunch>>,
}

// Large enough for dozens of layers with distinct forward and backward
// resources, while keeping dynamic-shape metadata bounded per context.
const F32_PREPARED_CACHE_LIMIT: usize = 1024;
const SM120_PREPARED_CACHE_LIMIT: usize = 1024;

fn make_room_in_bounded_cache<K, V>(
    entries: &mut HashMap<K, V>,
    incoming: &K,
    limit: usize,
    mut keep: impl FnMut(&V) -> bool,
) where
    K: Eq + std::hash::Hash,
{
    entries.retain(|_, value| keep(value));
    if !entries.contains_key(incoming) && entries.len() >= limit {
        entries.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sm120PreparedKey {
    context: GemmRouteIdentity,
    route: Sm120ForcedRoute,
    a: CUptr,
    b: CUptr,
    output: CUptr,
    bias: CUptr,
    alpha_bits: u32,
    beta_bits: u32,
}

impl std::hash::Hash for Sm120PreparedKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.context.hash(state);
        self.route.op.hash(state);
        match self.route.dtype {
            WeightDtype::F32 => 0_u8,
            WeightDtype::F16 => 1,
            WeightDtype::Bf16 => 2,
        }
        .hash(state);
        self.route.physical.hash(state);
        self.route.shape.hash(state);
        self.a.hash(state);
        self.b.hash(state);
        self.output.hash(state);
        self.bias.hash(state);
        self.alpha_bits.hash(state);
        self.beta_bits.hash(state);
    }
}

impl Sm120PreparedKey {
    fn new(context: GemmRouteIdentity, route: Sm120ForcedRoute, request: Sm120AutoRequest) -> Self {
        Self {
            context,
            route,
            a: request.a_ptr,
            b: request.b_ptr,
            output: request.operands.output_ptr,
            bias: request.operands.bias_ptr,
            alpha_bits: request.operands.alpha.to_bits(),
            beta_bits: request.operands.beta.to_bits(),
        }
    }
}

struct Sm120PreparedCacheEntry {
    prepared: Box<Sm120PreparedLaunch>,
    managed_epoch: Option<ManagedAllocationEpochStamp>,
}

#[derive(Default)]
pub(crate) struct Sm120PreparedLaunchCache {
    entries: HashMap<Sm120PreparedKey, Sm120PreparedCacheEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sm120ManagedEpochState {
    Missing,
    Current,
    Stale,
    Untracked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sm120CacheAction {
    UsePrepared,
    Validate,
    Prepare,
    CaptureMissing,
    CaptureStale,
    CaptureUntracked,
}

fn sm120_cache_action(
    capturing: bool,
    hit: bool,
    epoch: Sm120ManagedEpochState,
) -> Sm120CacheAction {
    if !hit {
        return if capturing {
            Sm120CacheAction::CaptureMissing
        } else {
            Sm120CacheAction::Prepare
        };
    }
    if !capturing {
        return Sm120CacheAction::Validate;
    }
    match epoch {
        Sm120ManagedEpochState::Current => Sm120CacheAction::UsePrepared,
        Sm120ManagedEpochState::Stale => Sm120CacheAction::CaptureStale,
        Sm120ManagedEpochState::Untracked | Sm120ManagedEpochState::Missing => {
            Sm120CacheAction::CaptureUntracked
        }
    }
}

fn sm120_capture_cache_error(action: Sm120CacheAction) -> &'static str {
    match action {
        Sm120CacheAction::CaptureMissing => {
            "prepared SM120 Triad cache entry is missing during graph capture; run eager warmup again"
        }
        Sm120CacheAction::CaptureStale => {
            "prepared SM120 Triad allocation epoch changed during graph capture; run eager warmup again"
        }
        Sm120CacheAction::CaptureUntracked => {
            "prepared SM120 Triad automatic capture requires managed allocations; run eager warmup again"
        }
        Sm120CacheAction::UsePrepared | Sm120CacheAction::Validate | Sm120CacheAction::Prepare => {
            "SM120 cache action is not a capture error"
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::mamba_ssm::gpu) struct Sm120AutoBranchSeal {
    pub(in crate::mamba_ssm::gpu) route: Sm120ForcedRoute,
}

impl Sm120PreparedLaunchCache {
    fn managed_epoch_state(entry: Option<&Sm120PreparedCacheEntry>) -> Sm120ManagedEpochState {
        match entry {
            None => Sm120ManagedEpochState::Missing,
            Some(entry) => match entry.managed_epoch.as_ref() {
                Some(epoch) if epoch.is_current() => Sm120ManagedEpochState::Current,
                Some(_) => Sm120ManagedEpochState::Stale,
                None => Sm120ManagedEpochState::Untracked,
            },
        }
    }

    fn ensure_sm120_prepared(
        &mut self,
        ctx: &GpuCtx,
        key: Sm120PreparedKey,
        route: Sm120ForcedRoute,
        request: Sm120AutoRequest,
    ) -> Result<&Sm120PreparedLaunch, String> {
        let capturing = ctx
            .stream
            .capture_status()
            .map_err(|error| format!("query SM120 TMA capture status: {error:?}"))?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let epoch = Self::managed_epoch_state(self.entries.get(&key));
        let action = sm120_cache_action(capturing, self.entries.contains_key(&key), epoch);
        match action {
            Sm120CacheAction::UsePrepared => {
                return Ok(&self.entries.get(&key).expect("cache hit above").prepared);
            }
            Sm120CacheAction::CaptureMissing => {
                return Err(sm120_capture_cache_error(action).into());
            }
            Sm120CacheAction::CaptureStale | Sm120CacheAction::CaptureUntracked => {
                return Err(sm120_capture_cache_error(action).into());
            }
            Sm120CacheAction::Validate => {
                let validation = validate_sm120_graph_replay(
                    &ctx.stream,
                    &ctx.kernels,
                    &self.entries.get(&key).expect("cache hit above").prepared,
                );
                if validation.is_ok() {
                    return Ok(&self
                        .entries
                        .get(&key)
                        .expect("validated cache hit")
                        .prepared);
                }
                self.entries.remove(&key);
            }
            Sm120CacheAction::Prepare => {}
        }

        let maps = prepare_sm120_tensor_maps(
            &ctx.stream,
            &ctx.kernels,
            Sm120MapRequest {
                op: route.op,
                dtype: route.dtype,
                tile: route.physical.tile,
                bk: route.physical.bk,
                a_ptr: request.a_ptr,
                b_ptr: request.b_ptr,
                shape: route.shape,
            },
        )?;
        let prepared =
            prepare_sm120_tma_forced(&ctx.stream, &ctx.kernels, route, &maps, request.operands)?;
        let entry = Sm120PreparedCacheEntry {
            managed_epoch: prepared.managed_epoch(),
            prepared: Box::new(prepared),
        };
        make_room_in_bounded_cache(
            &mut self.entries,
            &key,
            SM120_PREPARED_CACHE_LIMIT,
            |cached| {
                cached
                    .managed_epoch
                    .as_ref()
                    .is_none_or(ManagedAllocationEpochStamp::is_current)
            },
        );
        if !self.entries.contains_key(&key) {
            self.entries
                .try_reserve(1)
                .map_err(|error| format!("reserve prepared SM120 TMA cache: {error}"))?;
        }
        self.entries.insert(key, entry);
        Ok(&self
            .entries
            .get(&key)
            .expect("prepared SM120 TMA cache entry was inserted above")
            .prepared)
    }
}

const SM100_PREPARED_CACHE_LIMIT: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Sm100PreparedKey {
    context: GemmRouteIdentity,
    route: Sm100ForcedRoute,
    a: CUptr,
    b: CUptr,
    output: CUptr,
    bias: CUptr,
    alpha_bits: u32,
    beta_bits: u32,
}

impl Eq for Sm100PreparedKey {}

impl std::hash::Hash for Sm100PreparedKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.context.hash(state);
        self.route.op.hash(state);
        match self.route.dtype {
            WeightDtype::F32 => 0_u8,
            WeightDtype::F16 => 1,
            WeightDtype::Bf16 => 2,
        }
        .hash(state);
        self.route.physical.hash(state);
        self.route.shape.hash(state);
        self.a.hash(state);
        self.b.hash(state);
        self.output.hash(state);
        self.bias.hash(state);
        self.alpha_bits.hash(state);
        self.beta_bits.hash(state);
    }
}

impl Sm100PreparedKey {
    fn new(context: GemmRouteIdentity, route: Sm100ForcedRoute, request: Sm100AutoRequest) -> Self {
        Self {
            context,
            route,
            a: request.a_ptr,
            b: request.b_ptr,
            output: request.operands.output_ptr,
            bias: request.operands.bias_ptr,
            alpha_bits: request.operands.alpha.to_bits(),
            beta_bits: request.operands.beta.to_bits(),
        }
    }
}

struct Sm100PreparedCacheEntry {
    prepared: Box<Sm100PreparedLaunch>,
    managed_epoch: Option<ManagedAllocationEpochStamp>,
}

/// Prepared SM100 launches, keyed the way the SM120 cache is: an eager
/// launch prepares and caches, a capture replays a prepared entry or fails
/// closed.
#[derive(Default)]
pub(crate) struct Sm100PreparedLaunchCache {
    entries: HashMap<Sm100PreparedKey, Sm100PreparedCacheEntry>,
}

fn sm100_capture_cache_error(action: Sm120CacheAction) -> &'static str {
    match action {
        Sm120CacheAction::CaptureMissing => {
            "prepared SM100 Triad cache entry is missing during graph capture; run eager warmup again"
        }
        Sm120CacheAction::CaptureStale => {
            "prepared SM100 Triad allocation epoch changed during graph capture; run eager warmup again"
        }
        Sm120CacheAction::CaptureUntracked => {
            "prepared SM100 Triad automatic capture requires managed allocations; run eager warmup again"
        }
        Sm120CacheAction::UsePrepared | Sm120CacheAction::Validate | Sm120CacheAction::Prepare => {
            "SM100 cache action is not a capture error"
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::mamba_ssm::gpu) struct Sm100AutoBranchSeal {
    pub(in crate::mamba_ssm::gpu) route: Sm100ForcedRoute,
}

impl Sm100PreparedLaunchCache {
    fn managed_epoch_state(entry: Option<&Sm100PreparedCacheEntry>) -> Sm120ManagedEpochState {
        match entry {
            None => Sm120ManagedEpochState::Missing,
            Some(entry) => match entry.managed_epoch.as_ref() {
                Some(epoch) if epoch.is_current() => Sm120ManagedEpochState::Current,
                Some(_) => Sm120ManagedEpochState::Stale,
                None => Sm120ManagedEpochState::Untracked,
            },
        }
    }

    fn ensure_sm100_prepared(
        &mut self,
        ctx: &GpuCtx,
        key: Sm100PreparedKey,
        route: Sm100ForcedRoute,
        request: Sm100AutoRequest,
    ) -> Result<&Sm100PreparedLaunch, String> {
        let capturing = ctx
            .stream
            .capture_status()
            .map_err(|error| format!("query SM100 TCGEN capture status: {error:?}"))?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let epoch = Self::managed_epoch_state(self.entries.get(&key));
        let action = sm120_cache_action(capturing, self.entries.contains_key(&key), epoch);
        match action {
            Sm120CacheAction::UsePrepared => {
                return Ok(&self.entries.get(&key).expect("cache hit above").prepared);
            }
            Sm120CacheAction::CaptureMissing
            | Sm120CacheAction::CaptureStale
            | Sm120CacheAction::CaptureUntracked => {
                return Err(sm100_capture_cache_error(action).into());
            }
            Sm120CacheAction::Validate => {
                let validation = validate_sm100_graph_replay(
                    &ctx.stream,
                    &ctx.kernels,
                    &self.entries.get(&key).expect("cache hit above").prepared,
                );
                if validation.is_ok() {
                    return Ok(&self
                        .entries
                        .get(&key)
                        .expect("validated cache hit")
                        .prepared);
                }
                self.entries.remove(&key);
            }
            Sm120CacheAction::Prepare => {}
        }

        let maps = prepare_sm100_tensor_maps(
            &ctx.stream,
            &ctx.kernels,
            Sm100MapRequest {
                op: route.op,
                dtype: route.dtype,
                tile: route.physical.tile,
                a_ptr: request.a_ptr,
                b_ptr: request.b_ptr,
                shape: route.shape,
            },
        )?;
        let prepared =
            prepare_sm100_tcgen_forced(&ctx.stream, &ctx.kernels, route, &maps, request.operands)?;
        let entry = Sm100PreparedCacheEntry {
            managed_epoch: prepared.managed_epoch(),
            prepared: Box::new(prepared),
        };
        make_room_in_bounded_cache(
            &mut self.entries,
            &key,
            SM100_PREPARED_CACHE_LIMIT,
            |cached| {
                cached
                    .managed_epoch
                    .as_ref()
                    .is_none_or(ManagedAllocationEpochStamp::is_current)
            },
        );
        if !self.entries.contains_key(&key) {
            self.entries
                .try_reserve(1)
                .map_err(|error| format!("reserve prepared SM100 TCGEN cache: {error}"))?;
        }
        self.entries.insert(key, entry);
        Ok(&self
            .entries
            .get(&key)
            .expect("prepared SM100 TCGEN cache entry was inserted above")
            .prepared)
    }
}

fn sm100_policy_dtype(dtype: WeightDtype) -> Result<PolicyDtype, String> {
    match dtype {
        WeightDtype::Bf16 => Ok(PolicyDtype::Bf16),
        WeightDtype::F16 => Ok(PolicyDtype::F16),
        WeightDtype::F32 => Err("SM100 automatic route requires BF16 or F16".into()),
    }
}

unsafe fn enqueue_sm100_tcgen_prepared_observed<O: PhysicalLaunchObserver>(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    prepared: &Sm100PreparedLaunch,
    observer: &mut O,
    observation: Option<PhysicalLaunchObservation>,
) -> Result<(), String> {
    validate_sm100_prepared_binding(stream, kernels, prepared)?;
    let spec = prepared.route.kernel_spec()?;
    if spec.symbol != prepared.identity.symbol {
        return Err("SM100 prepared symbol no longer matches its physical route".into());
    }
    let function = kernels
        .sm100_function(spec.symbol)
        .ok_or_else(|| format!("SM100 kernel {} is unavailable", spec.symbol))?;
    let (rows, columns) = match prepared.route.op {
        Sm100Op::Nn => (prepared.route.shape.m, prepared.route.shape.n),
        Sm100Op::Tn => (prepared.route.shape.k, prepared.route.shape.n),
        Sm100Op::Nt => (prepared.route.shape.m, prepared.route.shape.k),
    };
    let rows = checked_u32(rows, "SM100 output rows")?;
    let columns = checked_u32(columns, "SM100 output columns")?;
    let grid = checked_grid_product(
        rows.div_ceil(prepared.route.physical.tile.output_rows()),
        columns.div_ceil(prepared.route.physical.tile.output_columns()),
        1,
    )?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (spec.threads, 1, 1),
        shared_mem_bytes: spec.dynamic_shared_bytes,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&prepared.operands.output_ptr);
    builder.arg(&prepared.maps.a);
    builder.arg(&prepared.maps.b);
    builder.arg(&prepared.operands.bias_ptr);
    let params = Sm100KernelParams::from_words(prepared.params);
    builder.arg(&params);
    unsafe { enqueue_with_physical_observation(observer, &mut builder, config, observation) }
        .map_err(|error| error.with_driver_context(format_args!("launch {}", spec.symbol)))
}

/// The automatic SM100 launch: a measured cell of the board's capability
/// prepares once, launches under the observer and is replayed by graphs;
/// every other request declines to the portable caller.
pub(in crate::mamba_ssm::gpu) fn launch_sm100_auto_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &mut O,
    request: Sm100AutoRequest,
) -> Result<Option<Sm100AutoBranchSeal>, String> {
    let Some(target) = ctx.kernels.sm100_target_candidate() else {
        return Ok(None);
    };
    let Some(route) = resolve_sm100_auto(target.device_cc, Some(target), request) else {
        static NO_CELL: std::sync::Once = std::sync::Once::new();
        crate::mamba_ssm::gpu::diagnostics::warn_once(&NO_CELL, || {
            format!(
                "no measured SM100 cell for {:?} {:?} {:?}; the portable tensor-core tiles serve \
                 it (reported once; later uncovered shapes are silent)",
                request.op, request.dtype, request.shape
            )
        });
        return Ok(None);
    };
    let key = Sm100PreparedKey::new(ctx.gemm_route(), route, request);
    let caps = query_specialized_device_caps(
        &ctx.stream,
        target.nvrtc_arch,
        crate::mamba_ssm::gpu::kernels::nvrtc_version(),
    )?;
    ctx.with_sm100_prepared_launches(|cache| {
        let prepared = cache.ensure_sm100_prepared(ctx, key, route, request)?;
        let resolved = prepared.identity().resolved_route(caps)?;
        unsafe {
            enqueue_sm100_tcgen_prepared_observed(
                &ctx.stream,
                &ctx.kernels,
                prepared,
                observer,
                Some(PhysicalLaunchObservation::gemm(
                    sm100_policy_dtype(route.dtype)?,
                    None,
                    resolved,
                )),
            )
        }?;
        ctx.record_resolved_gemm_route(resolved)?;
        Ok(Some(Sm100AutoBranchSeal { route }))
    })
}

/// The prepared SM100 launch a graph capture replays: the entry the eager
/// warmup cached, validated against the live binding.
pub(in crate::mamba_ssm::gpu) fn prepare_sm100_auto_graph_sequence<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &O,
    request: Sm100AutoRequest,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    let target = ctx
        .kernels
        .sm100_target_candidate()
        .ok_or_else(|| "prepared SM100 graph route has no module target".to_string())?;
    let route = resolve_sm100_auto(target.device_cc, Some(target), request)
        .ok_or_else(|| "prepared SM100 graph request is no longer qualified".to_string())?;
    let key = Sm100PreparedKey::new(ctx.gemm_route(), route, request);
    let caps = query_specialized_device_caps(
        &ctx.stream,
        target.nvrtc_arch,
        crate::mamba_ssm::gpu::kernels::nvrtc_version(),
    )?;
    ctx.with_sm100_prepared_launches(|cache| {
        let prepared = &cache
            .entries
            .get(&key)
            .ok_or_else(|| {
                "prepared SM100 graph cache entry is missing; run eager warmup again".to_string()
            })?
            .prepared;
        validate_sm100_graph_replay(&ctx.stream, &ctx.kernels, prepared)?;
        let resolved = prepared.identity().resolved_route(caps)?;
        let config = LaunchConfig {
            grid_dim: resolved.launch.grid_dim,
            block_dim: resolved.launch.block_dim,
            shared_mem_bytes: resolved.launch.shared_mem_bytes,
        };
        let observation =
            PhysicalLaunchObservation::gemm(sm100_policy_dtype(route.dtype)?, None, resolved);
        let node = resolve_physical_launch_observation(observer, observation, config)?;
        let function = ctx
            .kernels
            .sm100_function(resolved.symbol)
            .ok_or_else(|| format!("qualified SM100 symbol {} is unavailable", resolved.symbol))?
            .clone();
        let mut arguments = PhysicalScalarKernelArguments::new();
        arguments.push(prepared.operands.output_ptr)?;
        arguments.push(prepared.maps.a)?;
        arguments.push(prepared.maps.b)?;
        arguments.push(prepared.operands.bias_ptr)?;
        arguments.push(Sm100KernelParams::from_words(prepared.params))?;
        validate_sm100_graph_replay(&ctx.stream, &ctx.kernels, prepared)?;
        Ok(PreparedTriadPhysicalGraphSequence {
            launches: vec![PreparedTriadPhysicalGraphLaunch {
                function,
                config,
                node,
                arguments: Box::new(arguments),
            }]
            .into_boxed_slice(),
        })
    })
}

const SM90A_PREPARED_CACHE_LIMIT: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Sm90aPreparedKey {
    context: GemmRouteIdentity,
    route: Sm90aForcedRoute,
    a: CUptr,
    b: CUptr,
    output: CUptr,
    bias: CUptr,
    alpha_bits: u32,
    beta_bits: u32,
}

impl Eq for Sm90aPreparedKey {}

impl std::hash::Hash for Sm90aPreparedKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.context.hash(state);
        self.route.op.hash(state);
        match self.route.dtype {
            WeightDtype::F32 => 0_u8,
            WeightDtype::F16 => 1,
            WeightDtype::Bf16 => 2,
        }
        .hash(state);
        self.route.schedule.hash(state);
        self.route.shape.hash(state);
        self.a.hash(state);
        self.b.hash(state);
        self.output.hash(state);
        self.bias.hash(state);
        self.alpha_bits.hash(state);
        self.beta_bits.hash(state);
    }
}

impl Sm90aPreparedKey {
    fn new(context: GemmRouteIdentity, route: Sm90aForcedRoute, request: Sm90aAutoRequest) -> Self {
        Self {
            context,
            route,
            a: request.a_ptr,
            b: request.b_ptr,
            output: request.operands.output_ptr,
            bias: request.operands.bias_ptr,
            alpha_bits: request.operands.alpha.to_bits(),
            beta_bits: request.operands.beta.to_bits(),
        }
    }
}

/// A prepared SM90a launch: the encoded tensor maps, the identity the
/// eager launch recorded, and the route and operands that produced them.
struct Sm90aPreparedCacheEntry {
    maps: Box<Sm90aPreparedTensorMaps>,
    identity: Sm90aRouteIdentity,
    route: Sm90aForcedRoute,
    operands: Sm90aLaunchOperands,
    managed_epoch: Option<ManagedAllocationEpochStamp>,
}

/// Prepared SM90a launches, keyed the way the SM120 cache is: an eager
/// launch prepares and caches, a capture replays a prepared entry or fails
/// closed.
#[derive(Default)]
pub(crate) struct Sm90aPreparedLaunchCache {
    entries: HashMap<Sm90aPreparedKey, Sm90aPreparedCacheEntry>,
}

fn sm90a_capture_cache_error(action: Sm120CacheAction) -> &'static str {
    match action {
        Sm120CacheAction::CaptureMissing => {
            "prepared SM90a Triad cache entry is missing during graph capture; run eager warmup again"
        }
        Sm120CacheAction::CaptureStale => {
            "prepared SM90a Triad allocation epoch changed during graph capture; run eager warmup again"
        }
        Sm120CacheAction::CaptureUntracked => {
            "prepared SM90a Triad automatic capture requires managed allocations; run eager warmup again"
        }
        Sm120CacheAction::UsePrepared | Sm120CacheAction::Validate | Sm120CacheAction::Prepare => {
            "SM90a cache action is not a capture error"
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::mamba_ssm::gpu) struct Sm90aAutoBranchSeal {
    pub(in crate::mamba_ssm::gpu) route: Sm90aForcedRoute,
}

impl Sm90aPreparedLaunchCache {
    fn managed_epoch_state(entry: Option<&Sm90aPreparedCacheEntry>) -> Sm120ManagedEpochState {
        match entry {
            None => Sm120ManagedEpochState::Missing,
            Some(entry) => match entry.managed_epoch.as_ref() {
                Some(epoch) if epoch.is_current() => Sm120ManagedEpochState::Current,
                Some(_) => Sm120ManagedEpochState::Stale,
                None => Sm120ManagedEpochState::Untracked,
            },
        }
    }

    fn ensure_sm90a_prepared(
        &mut self,
        ctx: &GpuCtx,
        key: Sm90aPreparedKey,
        route: Sm90aForcedRoute,
        request: Sm90aAutoRequest,
    ) -> Result<&Sm90aPreparedCacheEntry, String> {
        let capturing = ctx
            .stream
            .capture_status()
            .map_err(|error| format!("query SM90a WGMMA capture status: {error:?}"))?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let epoch = Self::managed_epoch_state(self.entries.get(&key));
        let action = sm120_cache_action(capturing, self.entries.contains_key(&key), epoch);
        match action {
            Sm120CacheAction::UsePrepared => {
                return Ok(self.entries.get(&key).expect("cache hit above"));
            }
            Sm120CacheAction::CaptureMissing
            | Sm120CacheAction::CaptureStale
            | Sm120CacheAction::CaptureUntracked => {
                return Err(sm90a_capture_cache_error(action).into());
            }
            Sm120CacheAction::Validate => {
                let entry = self.entries.get(&key).expect("cache hit above");
                let validation = validate_sm90a_graph_replay(
                    &ctx.stream,
                    &ctx.kernels,
                    entry.route,
                    &entry.maps,
                    entry.operands,
                    entry.identity,
                );
                if validation.is_ok() {
                    return Ok(self.entries.get(&key).expect("validated cache hit"));
                }
                self.entries.remove(&key);
            }
            Sm120CacheAction::Prepare => {}
        }

        let maps = prepare_sm90a_tensor_maps(
            &ctx.stream,
            &ctx.kernels,
            Sm90aMapRequest {
                op: route.op,
                dtype: route.dtype,
                a_ptr: request.a_ptr,
                b_ptr: request.b_ptr,
                shape: route.shape,
            },
        )?;
        let identity =
            sm90a_forced_identity(&ctx.stream, &ctx.kernels, route, &maps, request.operands)?;
        let managed_epoch = maps.managed_epoch(route, request.operands)?;
        let entry = Sm90aPreparedCacheEntry {
            maps: Box::new(maps),
            identity,
            route,
            operands: request.operands,
            managed_epoch,
        };
        make_room_in_bounded_cache(
            &mut self.entries,
            &key,
            SM90A_PREPARED_CACHE_LIMIT,
            |cached| {
                cached
                    .managed_epoch
                    .as_ref()
                    .is_none_or(ManagedAllocationEpochStamp::is_current)
            },
        );
        if !self.entries.contains_key(&key) {
            self.entries
                .try_reserve(1)
                .map_err(|error| format!("reserve prepared SM90a WGMMA cache: {error}"))?;
        }
        self.entries.insert(key, entry);
        Ok(self
            .entries
            .get(&key)
            .expect("prepared SM90a WGMMA cache entry was inserted above"))
    }
}

fn sm90a_policy_dtype(dtype: WeightDtype) -> Result<PolicyDtype, String> {
    match dtype {
        WeightDtype::Bf16 => Ok(PolicyDtype::Bf16),
        WeightDtype::F16 => Ok(PolicyDtype::F16),
        WeightDtype::F32 => Err("SM90a automatic route requires BF16 or F16".into()),
    }
}

/// The SM90a launch configuration of a route: one CTA per 64 by 128 output
/// tile, the schedule's threads, the module's dynamic shared memory.
fn sm90a_launch_config(route: Sm90aForcedRoute) -> Result<cudarc::driver::LaunchConfig, String> {
    let (rows, columns) = match route.op {
        Sm90aOp::Nn => (route.shape.m, route.shape.n),
        Sm90aOp::Tn => (route.shape.k, route.shape.n),
        Sm90aOp::Nt => (route.shape.m, route.shape.k),
    };
    let rows = checked_u32(rows, "SM90a output rows")?;
    let columns = checked_u32(columns, "SM90a output columns")?;
    let grid = checked_grid_product(
        rows.div_ceil(SM90A_TILE.0),
        columns.div_ceil(SM90A_TILE.1),
        1,
    )?;
    Ok(cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (route.schedule.threads(), 1, 1),
        shared_mem_bytes: SM90A_DYNAMIC_SHARED_BYTES,
    })
}

unsafe fn enqueue_sm90a_wgmma_prepared_observed<O: PhysicalLaunchObserver>(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    entry: &Sm90aPreparedCacheEntry,
    observer: &mut O,
    observation: Option<PhysicalLaunchObservation>,
) -> Result<(), String> {
    let live = sm90a_forced_identity(stream, kernels, entry.route, &entry.maps, entry.operands)?;
    entry
        .identity
        .ensure_current(live, "SM90a automatic launch")?;
    let symbol = entry.route.symbol();
    let function = kernels
        .sm90a_function(symbol)
        .ok_or_else(|| format!("SM90a kernel {symbol} is unavailable"))?;
    let config = sm90a_launch_config(entry.route)?;
    let m = checked_i32(entry.route.shape.m, "M")?;
    let k = checked_i32(entry.route.shape.k, "K")?;
    let n = checked_i32(entry.route.shape.n, "N")?;
    let ldc = checked_i32(entry.route.shape.ldc, "ldc")?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&entry.operands.output_ptr);
    builder.arg(&entry.maps.a);
    builder.arg(&entry.maps.b);
    builder.arg(&entry.operands.bias_ptr);
    builder.arg(&entry.operands.alpha);
    builder.arg(&entry.operands.beta);
    builder.arg(&m);
    builder.arg(&k);
    builder.arg(&n);
    builder.arg(&ldc);
    unsafe { enqueue_with_physical_observation(observer, &mut builder, config, observation) }
        .map_err(|error| error.with_driver_context(format_args!("launch {symbol}")))
}

/// The automatic SM90a launch: a Hopper board's measured cell or wave-rule
/// schedule prepares once, launches under the observer and is replayed by
/// graphs; every other board declines to the portable caller.
pub(in crate::mamba_ssm::gpu) fn launch_sm90a_auto_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &mut O,
    request: Sm90aAutoRequest,
) -> Result<Option<Sm90aAutoBranchSeal>, String> {
    if !ctx.kernels.has_sm90a_wgmma() {
        return Ok(None);
    }
    let device_cc = ctx
        .stream
        .context()
        .compute_capability()
        .map_err(|error| format!("query compute capability for the SM90a branch: {error:?}"))?;
    let Some(route) = resolve_sm90a_auto(device_cc, true, request) else {
        static NO_ROUTE: std::sync::Once = std::sync::Once::new();
        crate::mamba_ssm::gpu::diagnostics::warn_once(&NO_ROUTE, || {
            format!(
                "no SM90a route for {:?} {:?} {:?}; the portable tensor-core tiles serve it \
                 (reported once; later unserved shapes are silent)",
                request.op, request.dtype, request.shape
            )
        });
        return Ok(None);
    };
    let key = Sm90aPreparedKey::new(ctx.gemm_route(), route, request);
    let caps = query_specialized_device_caps(
        &ctx.stream,
        "sm_90a",
        crate::mamba_ssm::gpu::kernels::nvrtc_version(),
    )?;
    ctx.with_sm90a_prepared_launches(|cache| {
        let entry = cache.ensure_sm90a_prepared(ctx, key, route, request)?;
        let resolved = entry.identity.resolved_route(caps)?;
        unsafe {
            enqueue_sm90a_wgmma_prepared_observed(
                &ctx.stream,
                &ctx.kernels,
                entry,
                observer,
                Some(PhysicalLaunchObservation::gemm(
                    sm90a_policy_dtype(route.dtype)?,
                    None,
                    resolved,
                )),
            )
        }?;
        ctx.record_resolved_gemm_route(resolved)?;
        Ok(Some(Sm90aAutoBranchSeal { route }))
    })
}

/// The prepared SM90a launch a graph capture replays: the entry the eager
/// warmup cached, validated against the live binding.
pub(in crate::mamba_ssm::gpu) fn prepare_sm90a_auto_graph_sequence<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &O,
    request: Sm90aAutoRequest,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    if !ctx.kernels.has_sm90a_wgmma() {
        return Err("prepared SM90a graph route has no module".into());
    }
    let device_cc = ctx
        .stream
        .context()
        .compute_capability()
        .map_err(|error| format!("query compute capability for the SM90a graph: {error:?}"))?;
    let route = resolve_sm90a_auto(device_cc, true, request)
        .ok_or_else(|| "prepared SM90a graph request is no longer qualified".to_string())?;
    let key = Sm90aPreparedKey::new(ctx.gemm_route(), route, request);
    let caps = query_specialized_device_caps(
        &ctx.stream,
        "sm_90a",
        crate::mamba_ssm::gpu::kernels::nvrtc_version(),
    )?;
    ctx.with_sm90a_prepared_launches(|cache| {
        let entry = cache.entries.get(&key).ok_or_else(|| {
            "prepared SM90a graph cache entry is missing; run eager warmup again".to_string()
        })?;
        validate_sm90a_graph_replay(
            &ctx.stream,
            &ctx.kernels,
            entry.route,
            &entry.maps,
            entry.operands,
            entry.identity,
        )?;
        let resolved = entry.identity.resolved_route(caps)?;
        let config = LaunchConfig {
            grid_dim: resolved.launch.grid_dim,
            block_dim: resolved.launch.block_dim,
            shared_mem_bytes: resolved.launch.shared_mem_bytes,
        };
        let observation =
            PhysicalLaunchObservation::gemm(sm90a_policy_dtype(route.dtype)?, None, resolved);
        let node = resolve_physical_launch_observation(observer, observation, config)?;
        let function = ctx
            .kernels
            .sm90a_function(resolved.symbol)
            .ok_or_else(|| format!("qualified SM90a symbol {} is unavailable", resolved.symbol))?
            .clone();
        let mut arguments = PhysicalScalarKernelArguments::new();
        arguments.push(entry.operands.output_ptr)?;
        arguments.push(entry.maps.a)?;
        arguments.push(entry.maps.b)?;
        arguments.push(entry.operands.bias_ptr)?;
        arguments.push(entry.operands.alpha)?;
        arguments.push(entry.operands.beta)?;
        arguments.push(checked_i32(entry.route.shape.m, "M")?)?;
        arguments.push(checked_i32(entry.route.shape.k, "K")?)?;
        arguments.push(checked_i32(entry.route.shape.n, "N")?)?;
        arguments.push(checked_i32(entry.route.shape.ldc, "ldc")?)?;
        Ok(PreparedTriadPhysicalGraphSequence {
            launches: vec![PreparedTriadPhysicalGraphLaunch {
                function,
                config,
                node,
                arguments: Box::new(arguments),
            }]
            .into_boxed_slice(),
        })
    })
}

pub(in crate::mamba_ssm::gpu) struct ScalarLaunchControl<'a> {
    ctx: &'a GpuCtx,
    routes: &'a [ResolvedGemmRoute],
    plan: ScalarDispatchPlan,
    operands: F32TriadOperands,
    next: usize,
}

fn validate_prepared_scalar_binding(
    expected_symbol: &str,
    expected: ResolvedKernelLaunch,
    symbol: &str,
    config: cudarc::driver::LaunchConfig,
) -> Result<Sha256Digest, String> {
    if expected.arguments_digest == [0; 32] {
        return Err(format!(
            "prepared scalar kernel {expected_symbol} has a zero argument digest"
        ));
    }
    if expected_symbol != symbol
        || expected.grid_dim != config.grid_dim
        || expected.block_dim != config.block_dim
        || expected.shared_mem_bytes != config.shared_mem_bytes
    {
        return Err(format!(
            "prepared scalar kernel {expected_symbol} physical launch changed before {symbol}"
        ));
    }
    Ok(expected.arguments_digest)
}

impl ScalarLaunchControl<'_> {
    fn bind(
        &mut self,
        symbol: &str,
        config: cudarc::driver::LaunchConfig,
    ) -> Result<ResolvedGemmRoute, String> {
        let route = self
            .routes
            .get(self.next)
            .ok_or_else(|| format!("unexpected scalar kernel {symbol} after prepared route end"))?;
        let _frozen_arguments_digest =
            validate_prepared_scalar_binding(route.symbol, route.launch, symbol, config)?;
        self.ctx.record_resolved_gemm_route(*route)?;
        self.next += 1;
        Ok(*route)
    }

    fn finish(&self) -> Result<(), String> {
        if self.next == self.routes.len() {
            Ok(())
        } else {
            Err(format!(
                "prepared scalar route expected {} kernels but enqueued {}",
                self.routes.len(),
                self.next
            ))
        }
    }

    pub(super) fn operands(&self) -> F32TriadOperands {
        self.operands
    }

    pub(super) fn plan(&self) -> ScalarDispatchPlan {
        self.plan
    }

    fn validate_operands(&self, actual: F32TriadOperands) -> Result<(), String> {
        let expected = self.operands;
        if actual.output != expected.output
            || actual.a != expected.a
            || actual.b != expected.b
            || actual.bias != expected.bias
            || actual.alpha.to_bits() != expected.alpha.to_bits()
            || actual.beta.to_bits() != expected.beta.to_bits()
        {
            return Err("scalar operands changed after f32 Triad preparation".into());
        }
        Ok(())
    }
}

trait ScalarLaunchController {
    fn enqueue(
        &mut self,
        symbol: &'static str,
        config: LaunchConfig,
        builder: &mut ScalarLaunchArgs<'_>,
    ) -> Result<(), PhysicalCudaLaunchError>;
    fn plan(&self) -> ScalarDispatchPlan;
    fn operands(&self) -> F32TriadOperands;
    fn validate_operands(&self, actual: F32TriadOperands) -> Result<(), String>;
    fn prepares_physical_graph(&self) -> bool {
        false
    }
}

impl ScalarLaunchController for ScalarLaunchControl<'_> {
    #[inline(always)]
    fn enqueue(
        &mut self,
        symbol: &'static str,
        config: LaunchConfig,
        builder: &mut ScalarLaunchArgs<'_>,
    ) -> Result<(), PhysicalCudaLaunchError> {
        self.bind(symbol, config)
            .map_err(PhysicalCudaLaunchError::from)?;
        let mut observer = NoPhysicalObserver;
        unsafe {
            enqueue_with_physical_observation(&mut observer, builder.launch_args(), config, None)
        }
    }

    fn plan(&self) -> ScalarDispatchPlan {
        ScalarLaunchControl::plan(self)
    }

    fn operands(&self) -> F32TriadOperands {
        ScalarLaunchControl::operands(self)
    }

    fn validate_operands(&self, actual: F32TriadOperands) -> Result<(), String> {
        ScalarLaunchControl::validate_operands(self, actual)
    }
}

struct PhysicalScalarLaunchControl<'a, O> {
    base: ScalarLaunchControl<'a>,
    logical_dtype: PolicyDtype,
    physical_resources_digest: Sha256Digest,
    observer: &'a mut O,
}

impl<O: PhysicalLaunchObserver> PhysicalScalarLaunchControl<'_, O> {
    fn finish(&self) -> Result<(), String> {
        self.base.finish()
    }
}

impl<O: PhysicalLaunchObserver> ScalarLaunchController for PhysicalScalarLaunchControl<'_, O> {
    #[inline(always)]
    fn enqueue(
        &mut self,
        symbol: &'static str,
        config: LaunchConfig,
        builder: &mut ScalarLaunchArgs<'_>,
    ) -> Result<(), PhysicalCudaLaunchError> {
        let route = self
            .base
            .bind(symbol, config)
            .map_err(PhysicalCudaLaunchError::from)?;
        unsafe {
            enqueue_with_physical_observation(
                self.observer,
                builder.launch_args(),
                config,
                Some(PhysicalLaunchObservation::gemm(
                    self.logical_dtype,
                    Some(self.physical_resources_digest),
                    route,
                )),
            )
        }
    }

    fn plan(&self) -> ScalarDispatchPlan {
        self.base.plan()
    }

    fn operands(&self) -> F32TriadOperands {
        self.base.operands()
    }

    fn validate_operands(&self, actual: F32TriadOperands) -> Result<(), String> {
        self.base.validate_operands(actual)
    }
}

const PHYSICAL_SCALAR_MAX_KERNEL_ARGUMENTS: usize = 16;
const PHYSICAL_SCALAR_MAX_ARGUMENT_BYTES: usize = 128;

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct PhysicalScalarKernelArgument {
    bytes: [u8; PHYSICAL_SCALAR_MAX_ARGUMENT_BYTES],
}

unsafe impl DeviceRepr for PhysicalScalarKernelArgument {}

impl PhysicalScalarKernelArgument {
    fn encode<T: Copy>(value: T) -> Result<Self, String> {
        let width = std::mem::size_of::<T>();
        if width > PHYSICAL_SCALAR_MAX_ARGUMENT_BYTES {
            return Err(format!(
                "physical scalar graph argument uses {width} bytes; maximum is {PHYSICAL_SCALAR_MAX_ARGUMENT_BYTES}"
            ));
        }
        let mut encoded = Self {
            bytes: [0; PHYSICAL_SCALAR_MAX_ARGUMENT_BYTES],
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(&value).cast::<u8>(),
                encoded.bytes.as_mut_ptr(),
                width,
            );
        }
        Ok(encoded)
    }
}

struct PhysicalScalarKernelArguments {
    values: [PhysicalScalarKernelArgument; PHYSICAL_SCALAR_MAX_KERNEL_ARGUMENTS],
    len: usize,
}

impl PhysicalScalarKernelArguments {
    fn new() -> Self {
        Self {
            values: [PhysicalScalarKernelArgument {
                bytes: [0; PHYSICAL_SCALAR_MAX_ARGUMENT_BYTES],
            }; PHYSICAL_SCALAR_MAX_KERNEL_ARGUMENTS],
            len: 0,
        }
    }

    fn push<T: Copy>(&mut self, value: T) -> Result<(), String> {
        let slot = self
            .values
            .get_mut(self.len)
            .ok_or_else(|| "physical scalar graph argument capacity exceeded".to_string())?;
        *slot = PhysicalScalarKernelArgument::encode(value)?;
        self.len += 1;
        Ok(())
    }

    fn values(&self) -> &[PhysicalScalarKernelArgument] {
        &self.values[..self.len]
    }
}

struct ScalarLaunchArgs<'a> {
    builder: LaunchArgs<'a>,
    prepared_function: Option<CudaFunction>,
    prepared_arguments: Option<Box<PhysicalScalarKernelArguments>>,
    preparation_error: Option<String>,
}

impl<'a> ScalarLaunchArgs<'a> {
    fn new(
        stream: &'a Arc<CudaStream>,
        function: &'a CudaFunction,
        prepare_physical_graph: bool,
    ) -> Self {
        Self {
            builder: stream.launch_builder(function),
            prepared_function: prepare_physical_graph.then(|| function.clone()),
            prepared_arguments: prepare_physical_graph
                .then(|| Box::new(PhysicalScalarKernelArguments::new())),
            preparation_error: None,
        }
    }

    fn capture_argument<T: Copy>(&mut self, argument: T) {
        if let Some(arguments) = self.prepared_arguments.as_mut()
            && self.preparation_error.is_none()
            && let Err(error) = arguments.push(argument)
        {
            self.preparation_error = Some(error);
        }
    }

    fn arg<T: Copy + DeviceRepr>(&mut self, argument: &'a T) -> &mut Self {
        self.capture_argument(*argument);
        self.builder.arg(argument);
        self
    }

    fn arg_buffer(&mut self, buffer: &'a GpuBuffer) -> &mut Self {
        self.capture_argument(buffer.cached_ptr());
        self.builder.arg(buffer.inner());
        self
    }

    fn arg_buffer_mut(&mut self, buffer: &'a mut GpuBuffer) -> &mut Self {
        self.capture_argument(buffer.cached_ptr());
        self.builder.arg(buffer.inner_mut());
        self
    }

    fn launch_args(&mut self) -> &mut LaunchArgs<'a> {
        &mut self.builder
    }

    fn take_prepared(
        &mut self,
    ) -> Result<(CudaFunction, Box<PhysicalScalarKernelArguments>), String> {
        if let Some(error) = self.preparation_error.take() {
            return Err(error);
        }
        let function = self.prepared_function.take().ok_or_else(|| {
            "scalar launch arguments were not configured for physical graph preparation".to_string()
        })?;
        let arguments = self
            .prepared_arguments
            .take()
            .ok_or_else(|| "scalar launch argument storage was already consumed".to_string())?;
        Ok((function, arguments))
    }
}

fn scalar_launch_builder<'a, C: ScalarLaunchController>(
    stream: &'a Arc<CudaStream>,
    function: &'a CudaFunction,
    control: &Option<&mut C>,
) -> ScalarLaunchArgs<'a> {
    let prepare = control
        .as_deref()
        .is_some_and(ScalarLaunchController::prepares_physical_graph);
    ScalarLaunchArgs::new(stream, function, prepare)
}

struct PreparedTriadPhysicalGraphLaunch {
    function: CudaFunction,
    config: LaunchConfig,
    node: ResolvedPhysicalKernelLaunch,
    arguments: Box<PhysicalScalarKernelArguments>,
}

pub(in crate::mamba_ssm::gpu) struct PreparedTriadPhysicalGraphSequence {
    launches: Box<[PreparedTriadPhysicalGraphLaunch]>,
}

impl PreparedTriadPhysicalGraphSequence {
    pub(in crate::mamba_ssm::gpu) fn len(&self) -> usize {
        self.launches.len()
    }

    pub(in crate::mamba_ssm::gpu) fn bind<'a>(
        &'a self,
        stream: &'a Arc<CudaStream>,
    ) -> Result<BoundTriadPhysicalGraphSequence<'a>, String> {
        let mut launches = Vec::new();
        launches
            .try_reserve_exact(self.launches.len())
            .map_err(|error| format!("reserve bound scalar physical launches: {error}"))?;
        for launch in &self.launches {
            let mut builder = stream.launch_builder(&launch.function);
            for argument in launch.arguments.values() {
                builder.arg(argument);
            }
            launches.push(BoundTriadPhysicalGraphLaunch {
                builder,
                config: launch.config,
                node: launch.node,
            });
        }
        if launches.len() != self.launches.len() || launches.capacity() != self.launches.len() {
            return Err("bound scalar physical launch backing capacity is not exact".into());
        }
        Ok(BoundTriadPhysicalGraphSequence { launches })
    }
}

struct BoundTriadPhysicalGraphLaunch<'a> {
    builder: LaunchArgs<'a>,
    config: LaunchConfig,
    node: ResolvedPhysicalKernelLaunch,
}

pub(in crate::mamba_ssm::gpu) struct BoundTriadPhysicalGraphSequence<'a> {
    launches: Vec<BoundTriadPhysicalGraphLaunch<'a>>,
}

impl BoundTriadPhysicalGraphSequence<'_> {
    #[inline(always)]
    pub(in crate::mamba_ssm::gpu) unsafe fn enqueue(
        &mut self,
        observer: &mut RecordingPhysicalObserver,
    ) -> Result<(), PhysicalCudaLaunchError> {
        for launch in &mut self.launches {
            unsafe {
                enqueue_prepared_physical_launch(
                    observer,
                    &mut launch.builder,
                    launch.config,
                    launch.node,
                )?;
            }
        }
        Ok(())
    }
}

struct PreparedPhysicalScalarLaunchControl<'a, O> {
    base: ScalarLaunchControl<'a>,
    logical_dtype: PolicyDtype,
    physical_resources_digest: Sha256Digest,
    observer: &'a O,
    launches: Vec<PreparedTriadPhysicalGraphLaunch>,
}

impl<O: PhysicalLaunchObserver> PreparedPhysicalScalarLaunchControl<'_, O> {
    fn finish(self) -> Result<PreparedTriadPhysicalGraphSequence, String> {
        self.base.finish()?;
        if self.launches.len() != self.base.routes.len()
            || self.launches.capacity() != self.base.routes.len()
        {
            return Err("prepared scalar physical launch capacity is not exact".into());
        }
        Ok(PreparedTriadPhysicalGraphSequence {
            launches: self.launches.into_boxed_slice(),
        })
    }
}

impl<O: PhysicalLaunchObserver> ScalarLaunchController
    for PreparedPhysicalScalarLaunchControl<'_, O>
{
    #[inline(always)]
    fn enqueue(
        &mut self,
        symbol: &'static str,
        config: LaunchConfig,
        builder: &mut ScalarLaunchArgs<'_>,
    ) -> Result<(), PhysicalCudaLaunchError> {
        let route = self
            .base
            .bind(symbol, config)
            .map_err(PhysicalCudaLaunchError::from)?;
        let observation = PhysicalLaunchObservation::gemm(
            self.logical_dtype,
            Some(self.physical_resources_digest),
            route,
        );
        let node = resolve_physical_launch_observation(self.observer, observation, config)
            .map_err(PhysicalCudaLaunchError::from)?;
        let (function, arguments) = builder
            .take_prepared()
            .map_err(PhysicalCudaLaunchError::from)?;
        self.launches.push(PreparedTriadPhysicalGraphLaunch {
            function,
            config,
            node,
            arguments,
        });
        Ok(())
    }

    fn plan(&self) -> ScalarDispatchPlan {
        self.base.plan()
    }

    fn operands(&self) -> F32TriadOperands {
        self.base.operands()
    }

    fn validate_operands(&self, actual: F32TriadOperands) -> Result<(), String> {
        self.base.validate_operands(actual)
    }

    fn prepares_physical_graph(&self) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScalarNodeSpec {
    symbol: &'static str,
    tile: (u32, u32),
    bk: u32,
    stages: u8,
    launch: ResolvedKernelLaunch,
}

fn scalar_node_count(plan: ScalarDispatchPlan) -> usize {
    match plan {
        ScalarDispatchPlan::NtSplitKTail { k_tail, .. } => 3 + k_tail,
        ScalarDispatchPlan::NnSplitKThinTail { .. }
        | ScalarDispatchPlan::NnSplitKThin
        | ScalarDispatchPlan::NnSplitKSlim { .. }
        | ScalarDispatchPlan::NnM32N64SplitK32Qualified
        | ScalarDispatchPlan::TnNarrowSplitM { .. }
        | ScalarDispatchPlan::TnSplitM { .. } => 2,
        ScalarDispatchPlan::NtD768TransposeM64N64Qualified
        | ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
        | ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
        | ScalarDispatchPlan::NtPrismVectorQualified
        | ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified => 2,
        ScalarDispatchPlan::NtSplitKMain { .. } | ScalarDispatchPlan::NtSplitKSlim { .. } => 3,
        _ => 1,
    }
}

fn scalar_plan_requires_zero_beta(plan: ScalarDispatchPlan) -> bool {
    matches!(
        plan,
        ScalarDispatchPlan::NnSplitKThinTail { .. }
            | ScalarDispatchPlan::NnSplitKThin
            | ScalarDispatchPlan::NnSplitKSlim { .. }
            | ScalarDispatchPlan::NnM32N64SplitK32Qualified
            | ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            | ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
            | ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
            | ScalarDispatchPlan::NtPrismVectorQualified
            | ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified
            | ScalarDispatchPlan::NtM2N16SplitK32Qualified
    )
}

fn scalar_plan_fields(plan: ScalarDispatchPlan) -> (u8, u64, u64) {
    match plan {
        ScalarDispatchPlan::NnUltraThin => (1, 0, 0),
        ScalarDispatchPlan::NnNarrowSmall => (2, 0, 0),
        ScalarDispatchPlan::NnNarrow => (3, 0, 0),
        ScalarDispatchPlan::NnGemv => (4, 0, 0),
        ScalarDispatchPlan::NnSplitKThinTail { k_main, k_tail } => {
            (5, k_main as u64, k_tail as u64)
        }
        ScalarDispatchPlan::NnSplitKThin => (6, 0, 0),
        ScalarDispatchPlan::NnSplitKSlim { chunks } => (7, chunks as u64, 0),
        ScalarDispatchPlan::NnM32N64SplitK32Qualified => (36, 0, 0),
        ScalarDispatchPlan::NnM64N64Qualified => (24, 0, 0),
        ScalarDispatchPlan::NnFinal { slim } => (8, u64::from(slim), 0),
        ScalarDispatchPlan::TnGemv => (9, 0, 0),
        ScalarDispatchPlan::TnNarrow => (10, 0, 0),
        ScalarDispatchPlan::TnNarrowSplitM { m_chunk, chunks } => {
            (22, m_chunk as u64, chunks as u64)
        }
        ScalarDispatchPlan::TnSplitM { m_chunk, chunks } => (11, m_chunk as u64, chunks as u64),
        ScalarDispatchPlan::TnM16N16SplitM16Qualified => (35, 0, 0),
        ScalarDispatchPlan::TnFinal { slim } => (12, u64::from(slim), 0),
        ScalarDispatchPlan::NtNarrow => (13, 0, 0),
        ScalarDispatchPlan::NtSmallBatchWide => (14, 0, 0),
        ScalarDispatchPlan::NtGemv => (15, 0, 0),
        ScalarDispatchPlan::NtSplitKTail { k_main, k_tail } => (16, k_main as u64, k_tail as u64),
        ScalarDispatchPlan::NtSplitKMain { n_main, n_tail } => (17, n_main as u64, n_tail as u64),
        ScalarDispatchPlan::NtSplitKSlim { chunks } => (18, chunks as u64, 0),
        ScalarDispatchPlan::NtMidBatchWide => (19, 0, 0),
        ScalarDispatchPlan::NtM2N16SplitK32Qualified => (31, 0, 0),
        ScalarDispatchPlan::NtD768TransposeM64N64Qualified => (25, 0, 0),
        ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified => (26, 0, 0),
        ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified => (27, 0, 0),
        ScalarDispatchPlan::NtPrismVectorQualified => (30, 0, 0),
        ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified => (29, 0, 0),
        ScalarDispatchPlan::NtFinal { slim } => (20, u64::from(slim), 0),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ScalarArgumentLayout {
    output_offset: u64,
    a_offset: u64,
    b_offset: u64,
    null_pointer_mask: u64,
    output_column: Option<u64>,
}

fn scalar_argument_layout(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    plan: ScalarDispatchPlan,
    index: usize,
) -> ScalarArgumentLayout {
    let bias_null = u64::from(operands.bias.is_none());
    match (plan, index) {
        (ScalarDispatchPlan::NnSplitKThinTail { .. }, 0)
        | (ScalarDispatchPlan::NnSplitKThin, 0)
        | (ScalarDispatchPlan::NnSplitKSlim { .. }, 0)
        | (ScalarDispatchPlan::NnM32N64SplitK32Qualified, 0) => ScalarArgumentLayout::default(),
        (ScalarDispatchPlan::NnSplitKThinTail { k_main, .. }, 1) => ScalarArgumentLayout {
            a_offset: k_main as u64 * 4,
            b_offset: k_main as u64 * request.shape.n as u64 * 4,
            null_pointer_mask: bias_null << 2,
            ..ScalarArgumentLayout::default()
        },
        (ScalarDispatchPlan::NnSplitKThin, 1)
        | (ScalarDispatchPlan::NnSplitKSlim { .. }, 1)
        | (ScalarDispatchPlan::NnM32N64SplitK32Qualified, 1) => ScalarArgumentLayout {
            null_pointer_mask: 0b11000 | (bias_null << 2),
            ..ScalarArgumentLayout::default()
        },
        (ScalarDispatchPlan::NnM64N64Qualified, _) => ScalarArgumentLayout {
            null_pointer_mask: bias_null << 3,
            ..ScalarArgumentLayout::default()
        },
        (ScalarDispatchPlan::NtM2N16SplitK32Qualified, _) => ScalarArgumentLayout::default(),
        (ScalarDispatchPlan::TnM16N16SplitM16Qualified, _) => ScalarArgumentLayout::default(),
        (ScalarDispatchPlan::NtD768TransposeM64N64Qualified, 1)
        | (ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified, 1)
        | (ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified, 1)
        | (ScalarDispatchPlan::NtPrismVectorQualified, 1)
        | (ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified, 1) => ScalarArgumentLayout {
            null_pointer_mask: bias_null << 3,
            ..ScalarArgumentLayout::default()
        },
        (ScalarDispatchPlan::NtD768TransposeM64N64Qualified, 0)
        | (ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified, 0)
        | (ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified, 0)
        | (ScalarDispatchPlan::NtPrismVectorQualified, 0)
        | (ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified, 0) => {
            ScalarArgumentLayout::default()
        }
        (ScalarDispatchPlan::TnGemv, _)
        | (ScalarDispatchPlan::TnNarrow, _)
        | (ScalarDispatchPlan::TnNarrowSplitM { .. }, _)
        | (ScalarDispatchPlan::TnSplitM { .. }, _)
        | (ScalarDispatchPlan::TnFinal { .. }, _)
        | (ScalarDispatchPlan::NtNarrow, _)
        | (ScalarDispatchPlan::NtSmallBatchWide, _)
        | (ScalarDispatchPlan::NtGemv, _)
        | (ScalarDispatchPlan::NtMidBatchWide, _)
        | (ScalarDispatchPlan::NtFinal { .. }, _) => ScalarArgumentLayout::default(),
        (ScalarDispatchPlan::NtSplitKTail { k_main, .. }, tail) if tail >= 3 => {
            let column = k_main + tail - 3;
            ScalarArgumentLayout {
                b_offset: column as u64 * request.shape.n as u64 * 4,
                output_column: Some(column as u64),
                ..ScalarArgumentLayout::default()
            }
        }
        (ScalarDispatchPlan::NtSplitKMain { n_main, n_tail }, 2) if n_tail > 0 => {
            ScalarArgumentLayout {
                a_offset: n_main as u64 * 4,
                b_offset: n_main as u64 * request.shape.k as u64 * 4,
                null_pointer_mask: 0b100,
                ..ScalarArgumentLayout::default()
            }
        }
        (ScalarDispatchPlan::NtSplitKMain { .. }, 2)
        | (ScalarDispatchPlan::NtSplitKTail { .. }, 2)
        | (ScalarDispatchPlan::NtSplitKSlim { .. }, 2) => ScalarArgumentLayout {
            null_pointer_mask: 0b11100,
            ..ScalarArgumentLayout::default()
        },
        (ScalarDispatchPlan::NtSplitKMain { .. }, _)
        | (ScalarDispatchPlan::NtSplitKTail { .. }, _)
        | (ScalarDispatchPlan::NtSplitKSlim { .. }, _) => ScalarArgumentLayout::default(),
        _ => ScalarArgumentLayout {
            null_pointer_mask: bias_null << 3,
            ..ScalarArgumentLayout::default()
        },
    }
}

fn scalar_arguments_digest(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    plan: ScalarDispatchPlan,
    index: usize,
    symbol: &str,
) -> Sha256Digest {
    let (plan_tag, plan_a, plan_b) = scalar_plan_fields(plan);
    let layout = scalar_argument_layout(request, operands, plan, index);
    let output_column = layout.output_column.map(u64::to_le_bytes);
    FramedSha256::new(b"triad-scalar-kernel-arguments.v1")
        .required(b"symbol", symbol.as_bytes())
        .required(b"node-index", &(index as u64).to_le_bytes())
        .required(b"plan", &[plan_tag])
        .required(b"plan-a", &plan_a.to_le_bytes())
        .required(b"plan-b", &plan_b.to_le_bytes())
        .required(b"m", &(request.shape.m as u64).to_le_bytes())
        .required(b"k", &(request.shape.k as u64).to_le_bytes())
        .required(b"n", &(request.shape.n as u64).to_le_bytes())
        .required(b"lda", &(request.shape.lda as u64).to_le_bytes())
        .required(b"ldb", &(request.shape.ldb as u64).to_le_bytes())
        .required(b"ldc", &(request.shape.ldc as u64).to_le_bytes())
        .required(b"alpha", &operands.alpha.to_bits().to_le_bytes())
        .required(b"beta", &operands.beta.to_bits().to_le_bytes())
        .required(b"output-offset", &layout.output_offset.to_le_bytes())
        .required(b"a-offset", &layout.a_offset.to_le_bytes())
        .required(b"b-offset", &layout.b_offset.to_le_bytes())
        .required(
            b"null-pointer-mask",
            &layout.null_pointer_mask.to_le_bytes(),
        )
        .optional(
            b"output-column",
            output_column.as_ref().map(<[u8; 8]>::as_slice),
        )
        .finish()
}

fn push_scalar_node(
    nodes: &mut Vec<ScalarNodeSpec>,
    context: (F32TriadRequest, F32TriadOperands, ScalarDispatchPlan),
    symbol: &'static str,
    tile: (u32, u32),
    bk_stages: (u32, u8),
    config: cudarc::driver::LaunchConfig,
) {
    let (request, operands, plan) = context;
    let index = nodes.len();
    nodes.push(ScalarNodeSpec {
        symbol,
        tile,
        bk: bk_stages.0,
        stages: bk_stages.1,
        launch: ResolvedKernelLaunch {
            grid_dim: config.grid_dim,
            block_dim: config.block_dim,
            shared_mem_bytes: config.shared_mem_bytes,
            arguments_digest: scalar_arguments_digest(request, operands, plan, index, symbol),
        },
    });
}

fn f32_base_is_vector_aligned(pointer: CUptr) -> bool {
    pointer & 15 == 0
}

fn scalar_tn_kernel_symbol(plan: ScalarDispatchPlan, operands: F32TriadOperands) -> &'static str {
    match plan {
        ScalarDispatchPlan::TnNarrowSplitM { .. }
            if f32_base_is_vector_aligned(operands.a) && f32_base_is_vector_aligned(operands.b) =>
        {
            "gemm_bi_tn_narrow_splitm_partial_aligned"
        }
        ScalarDispatchPlan::TnNarrowSplitM { .. } => "gemm_bi_tn_narrow_splitm_partial",
        ScalarDispatchPlan::TnSplitM { .. }
            if f32_base_is_vector_aligned(operands.a) && f32_base_is_vector_aligned(operands.b) =>
        {
            "gemm_bi_tn_splitm_partial_aligned"
        }
        ScalarDispatchPlan::TnSplitM { .. } => "gemm_bi_tn_splitm_partial",
        ScalarDispatchPlan::TnFinal { slim: false }
            if f32_base_is_vector_aligned(operands.output)
                && f32_base_is_vector_aligned(operands.a)
                && f32_base_is_vector_aligned(operands.b) =>
        {
            "gemm_bi_tn_aligned"
        }
        ScalarDispatchPlan::TnFinal { slim: false } => "gemm_bi_tn",
        ScalarDispatchPlan::TnFinal { slim: true } => "gemm_bi_tn_slim",
        _ => unreachable!("TN vector-aligned symbol requested for a non-Big TN plan"),
    }
}

fn scalar_physical_nodes(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    plan: ScalarDispatchPlan,
) -> Result<Box<[ScalarNodeSpec]>, String> {
    let context = (request, operands, plan);
    let shape = request.shape;
    let m = checked_u32(shape.m, "scalar M")?;
    let k = checked_u32(shape.k, "scalar K")?;
    let n = checked_u32(shape.n, "scalar N")?;
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(scalar_node_count(plan))
        .map_err(|error| format!("reserve scalar physical plan: {error}"))?;
    let cfg = |grid_dim, block_dim, shared_mem_bytes| cudarc::driver::LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes,
    };
    match plan {
        ScalarDispatchPlan::NnUltraThin => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_nn_ultra_thin",
            (1, 32),
            (32, 1),
            cfg((n.div_ceil(32), m, 1), (256, 1, 1), k * 4),
        ),
        ScalarDispatchPlan::NnNarrowSmall => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_nn_narrow_small",
            (16, 16),
            (16, 1),
            cfg(
                (
                    checked_grid_product(m.div_ceil(16), n.div_ceil(16), 1)?,
                    1,
                    1,
                ),
                (64, 1, 1),
                0,
            ),
        ),
        ScalarDispatchPlan::NnNarrow => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_nn_narrow",
            (64, 32),
            (16, 1),
            cfg(
                (
                    checked_grid_product(m.div_ceil(64), n.div_ceil(32), 1)?,
                    1,
                    1,
                ),
                (128, 1, 1),
                0,
            ),
        ),
        ScalarDispatchPlan::NnGemv => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_nn_gemv",
            (4, 1),
            (32, 1),
            cfg((m.div_ceil(4), 1, 1), (128, 1, 1), 0),
        ),
        ScalarDispatchPlan::NnSplitKThinTail { .. } | ScalarDispatchPlan::NnSplitKThin => {
            let k_main = match plan {
                ScalarDispatchPlan::NnSplitKThinTail { k_main, .. } => {
                    checked_u32(k_main, "NN K main")?
                }
                _ => k,
            };
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_nn_splitk32_partial",
                (32, 64),
                (32, 1),
                cfg(
                    (
                        checked_grid_product(m.div_ceil(32), n.div_ceil(64), k_main / 32)?,
                        1,
                        1,
                    ),
                    (128, 1, 1),
                    0,
                ),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_splitk_reduce",
                (1, 1),
                (1, 1),
                cfg(
                    (
                        checked_u32_product(m, n, "NN reducer outputs")?.div_ceil(256),
                        1,
                        1,
                    ),
                    (256, 1, 1),
                    0,
                ),
            );
        }
        ScalarDispatchPlan::NnSplitKSlim { chunks } => {
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_nn_splitk_slim_partial",
                (128, 64),
                (32, 1),
                cfg(
                    (
                        checked_grid_product(m.div_ceil(128), n.div_ceil(64), 1)?,
                        1,
                        chunks,
                    ),
                    (128, 1, 1),
                    0,
                ),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_splitk_reduce",
                (1, 1),
                (1, 1),
                cfg(
                    (
                        checked_u32_product(m, n, "NN slim reducer outputs")?.div_ceil(256),
                        1,
                        1,
                    ),
                    (256, 1, 1),
                    0,
                ),
            );
        }
        ScalarDispatchPlan::NnM32N64SplitK32Qualified => {
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_nn_splitk32_m32n64_exact_v1",
                (32, 64),
                (32, 1),
                cfg((2_048, 1, 1), (128, 1, 1), 0),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_splitk_reduce",
                (1, 1),
                (1, 1),
                cfg((64, 1, 1), (256, 1, 1), 0),
            );
        }
        ScalarDispatchPlan::NnM64N64Qualified => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_nn_m64n64_bk16_s2_v1",
            (64, 64),
            (16, 2),
            cfg(
                (
                    checked_grid_product(m.div_ceil(64), n.div_ceil(64), 1)?,
                    1,
                    1,
                ),
                (128, 1, 1),
                super::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES,
            ),
        ),
        ScalarDispatchPlan::NnFinal { slim } => {
            let bn = if slim { 64 } else { 128 };
            push_scalar_node(
                &mut nodes,
                context,
                if slim {
                    "gemm_bi_nn_slim"
                } else {
                    "gemm_bi_nn"
                },
                (128, bn),
                (if slim { 32 } else { 16 }, if slim { 1 } else { 2 }),
                cfg(
                    (
                        checked_grid_product(m.div_ceil(128), n.div_ceil(bn), 1)?,
                        1,
                        1,
                    ),
                    (if slim { 128 } else { 256 }, 1, 1),
                    if slim { 0 } else { 34 * 1024 },
                ),
            );
        }
        ScalarDispatchPlan::TnGemv => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_tn_gemv",
            (4, 1),
            (32, 1),
            cfg((k.div_ceil(4), 1, 1), (128, 1, 1), 0),
        ),
        ScalarDispatchPlan::TnNarrow => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_tn_narrow",
            (64, 32),
            (16, 1),
            cfg(
                (
                    checked_grid_product(k.div_ceil(64), n.div_ceil(32), 1)?,
                    1,
                    1,
                ),
                (128, 1, 1),
                0,
            ),
        ),
        ScalarDispatchPlan::TnNarrowSplitM { chunks, .. } => {
            push_scalar_node(
                &mut nodes,
                context,
                scalar_tn_kernel_symbol(plan, operands),
                (64, 32),
                (16, 1),
                cfg(
                    (
                        k.div_ceil(64),
                        n.div_ceil(32),
                        checked_u32(chunks, "TN narrow split-M chunks")?,
                    ),
                    (128, 1, 1),
                    0,
                ),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_splitm_reduce",
                (1, 1),
                (1, 1),
                cfg(
                    (
                        checked_u32_product(k, n, "TN narrow reducer outputs")?.div_ceil(256),
                        1,
                        1,
                    ),
                    (256, 1, 1),
                    0,
                ),
            );
        }
        ScalarDispatchPlan::TnM16N16SplitM16Qualified => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1",
            (16, 16),
            (16, 2),
            cfg(
                (
                    checked_grid_product(k.div_ceil(16), n.div_ceil(16), 1)?,
                    1,
                    1,
                ),
                (super::contract::SCALAR_TN_M16N16_THREADS, 1, 1),
                super::contract::SCALAR_TN_M16N16_DYNAMIC_SHARED_BYTES,
            ),
        ),
        ScalarDispatchPlan::TnSplitM { chunks, .. } => {
            push_scalar_node(
                &mut nodes,
                context,
                scalar_tn_kernel_symbol(plan, operands),
                (128, 128),
                (16, 1),
                cfg(
                    (
                        checked_grid_product(k.div_ceil(128), n.div_ceil(128), 1)?,
                        1,
                        checked_u32(chunks, "TN split-M chunks")?,
                    ),
                    (256, 1, 1),
                    0,
                ),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_splitm_reduce",
                (1, 1),
                (1, 1),
                cfg(
                    (
                        checked_u32_product(k, n, "TN reducer outputs")?.div_ceil(256),
                        1,
                        1,
                    ),
                    (256, 1, 1),
                    0,
                ),
            );
        }
        ScalarDispatchPlan::TnFinal { slim } => {
            let bn = if slim { 64 } else { 128 };
            push_scalar_node(
                &mut nodes,
                context,
                scalar_tn_kernel_symbol(plan, operands),
                (128, bn),
                (if slim { 32 } else { 16 }, if slim { 1 } else { 2 }),
                cfg(
                    (
                        checked_grid_product(k.div_ceil(128), n.div_ceil(bn), 1)?,
                        1,
                        1,
                    ),
                    (if slim { 128 } else { 256 }, 1, 1),
                    if slim { 0 } else { 34 * 1024 },
                ),
            );
        }
        ScalarDispatchPlan::NtNarrow
        | ScalarDispatchPlan::NtSmallBatchWide
        | ScalarDispatchPlan::NtMidBatchWide => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_nt_narrow",
            (64, 32),
            (16, 1),
            cfg(
                (
                    checked_grid_product(m.div_ceil(64), k.div_ceil(32), 1)?,
                    1,
                    1,
                ),
                (128, 1, 1),
                0,
            ),
        ),
        ScalarDispatchPlan::NtGemv => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_nt_gemv",
            (1, 1),
            (1, 1),
            cfg(
                (
                    checked_u32_product(m, k, "NT GEMV outputs")?.div_ceil(256),
                    1,
                    1,
                ),
                (256, 1, 1),
                0,
            ),
        ),
        ScalarDispatchPlan::NtM2N16SplitK32Qualified => push_scalar_node(
            &mut nodes,
            context,
            "gemm_bi_nt_m2n16_bk64_splitk32_v1",
            (2, 16),
            (64, 2),
            cfg(
                (256, 1, 1),
                (super::contract::SCALAR_NT_M2N16_THREADS, 1, 1),
                super::contract::SCALAR_NT_M2N16_DYNAMIC_SHARED_BYTES,
            ),
        ),
        ScalarDispatchPlan::NtD768TransposeM64N64Qualified
        | ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
        | ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
        | ScalarDispatchPlan::NtPrismVectorQualified
        | ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified => {
            let m64_symbol = if plan == ScalarDispatchPlan::NtPrismVectorQualified {
                "gemm_bi_nn_prism_m64n64_bk16_s2_v1"
            } else {
                "gemm_bi_nn_m64n64_bk16_s2_v1"
            };
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_transpose_f32_32x16_d768_v1",
                (32, 32),
                (1, 1),
                cfg((n.div_ceil(32), k.div_ceil(32), 1), (32, 16, 1), 0),
            );
            push_scalar_node(
                &mut nodes,
                context,
                m64_symbol,
                (64, 64),
                (16, 2),
                cfg(
                    (
                        checked_grid_product(m.div_ceil(64), k.div_ceil(64), 1)?,
                        1,
                        1,
                    ),
                    (128, 1, 1),
                    super::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES,
                ),
            );
        }
        ScalarDispatchPlan::NtSplitKTail { k_main, k_tail } => {
            let k_main_u32 = checked_u32(k_main, "NT K main")?;
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_transpose_f32_2d",
                (32, 32),
                (1, 1),
                cfg((n.div_ceil(32), k_main_u32.div_ceil(32), 1), (32, 32, 1), 0),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_nn_splitk32_partial",
                (32, 64),
                (32, 1),
                cfg(
                    (
                        checked_grid_product(m.div_ceil(32), k_main_u32.div_ceil(64), n / 32)?,
                        1,
                        1,
                    ),
                    (128, 1, 1),
                    0,
                ),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_splitk_reduce",
                (1, 1),
                (1, 1),
                cfg(
                    (
                        checked_u32_product(m, k_main_u32, "NT K-tail reducer outputs")?
                            .div_ceil(256),
                        1,
                        1,
                    ),
                    (256, 1, 1),
                    0,
                ),
            );
            for _ in 0..k_tail {
                push_scalar_node(
                    &mut nodes,
                    context,
                    "gemm_bi_dx_col_gemv",
                    (1, 1),
                    (1, 1),
                    cfg((m.div_ceil(128), 1, 1), (128, 1, 1), 0),
                );
            }
        }
        ScalarDispatchPlan::NtSplitKMain { n_main, .. } => {
            let n_main = checked_u32(n_main, "NT N main")?;
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_transpose_f32_2d",
                (32, 32),
                (1, 1),
                cfg((n.div_ceil(32), k.div_ceil(32), 1), (32, 32, 1), 0),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_nn_splitk32_partial",
                (32, 64),
                (32, 1),
                cfg(
                    (
                        checked_grid_product(m.div_ceil(32), k.div_ceil(64), n_main / 32)?,
                        1,
                        1,
                    ),
                    (128, 1, 1),
                    0,
                ),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_splitk_reduce",
                (1, 1),
                (1, 1),
                cfg(
                    (
                        checked_u32_product(m, k, "NT reducer outputs")?.div_ceil(256),
                        1,
                        1,
                    ),
                    (256, 1, 1),
                    0,
                ),
            );
        }
        ScalarDispatchPlan::NtSplitKSlim { chunks } => {
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_transpose_f32_2d",
                (32, 32),
                (1, 1),
                cfg((n.div_ceil(32), k.div_ceil(32), 1), (32, 32, 1), 0),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_nn_splitk_slim_partial",
                (128, 64),
                (32, 1),
                cfg(
                    (
                        checked_grid_product(m.div_ceil(128), k.div_ceil(64), 1)?,
                        1,
                        chunks,
                    ),
                    (128, 1, 1),
                    0,
                ),
            );
            push_scalar_node(
                &mut nodes,
                context,
                "gemm_bi_splitk_reduce",
                (1, 1),
                (1, 1),
                cfg(
                    (
                        checked_u32_product(m, k, "NT slim reducer outputs")?.div_ceil(256),
                        1,
                        1,
                    ),
                    (256, 1, 1),
                    0,
                ),
            );
        }
        ScalarDispatchPlan::NtFinal { slim } => {
            let bn = if slim { 64 } else { 128 };
            push_scalar_node(
                &mut nodes,
                context,
                if slim {
                    "gemm_bi_nt_slim"
                } else {
                    "gemm_bi_nt"
                },
                (128, bn),
                (if slim { 32 } else { 16 }, if slim { 1 } else { 2 }),
                cfg(
                    (
                        checked_grid_product(m.div_ceil(128), k.div_ceil(bn), 1)?,
                        1,
                        1,
                    ),
                    (if slim { 128 } else { 256 }, 1, 1),
                    if slim {
                        0
                    } else {
                        SCALAR_BIG_NT_DYNAMIC_SHARED_BYTES
                    },
                ),
            );
        }
    }
    debug_assert_eq!(nodes.len(), scalar_node_count(plan));
    Ok(nodes.into_boxed_slice())
}

fn scalar_route_contract(
    symbol: &str,
) -> (
    PhysicalGemmBackend,
    ResolvedNumericContract,
    ResolvedOutputOwnership,
) {
    if matches!(
        symbol,
        "gemm_bi_nn_splitk32_partial"
            | "gemm_bi_nn_splitk_slim_partial"
            | "gemm_bi_nn_splitk32_m32n64_exact_v1"
    ) {
        return (
            PhysicalGemmBackend::ScalarFmaSplitKPartialV1,
            ResolvedNumericContract::ScalarFmaSplitKPartialV1,
            ResolvedOutputOwnership::OneCtaPerOutputTilePerSplitKPartitionV1,
        );
    }
    if symbol == "gemm_bi_splitk_reduce" {
        return (
            PhysicalGemmBackend::ScalarFmaSplitKF32ReduceV1,
            ResolvedNumericContract::ScalarFmaSplitKF32ReduceV1,
            ResolvedOutputOwnership::OneThreadPerOutputElementFixedSplitKReduceV1,
        );
    }
    if symbol.contains("_tn_narrow_splitm_partial") {
        return (
            PhysicalGemmBackend::ScalarFmaTnNarrowSplitMPartialV1,
            ResolvedNumericContract::ScalarFmaTnNarrowSplitMPartialV1,
            ResolvedOutputOwnership::OneCtaPerOutputTileV1,
        );
    }
    if symbol == "gemm_bi_splitm_reduce" {
        return (
            PhysicalGemmBackend::ScalarFmaTnSplitMF64ReduceV1,
            ResolvedNumericContract::ScalarFmaTnSplitMF64ReduceV1,
            ResolvedOutputOwnership::OneThreadPerOutputElementFixedSplitMReduceV1,
        );
    }
    if symbol == "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1" {
        return (
            PhysicalGemmBackend::ScalarFmaTnSplitMF64ReduceV1,
            ResolvedNumericContract::ScalarFmaTnSplitMF64ReduceV1,
            ResolvedOutputOwnership::OneCtaPerOutputTileV1,
        );
    }
    (
        PhysicalGemmBackend::ScalarFmaV1,
        ResolvedNumericContract::ScalarFmaV1,
        ResolvedOutputOwnership::OneCtaPerOutputTileV1,
    )
}

fn scalar_resolved_routes(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    resources_digest: Sha256Digest,
    nodes: &[ScalarNodeSpec],
) -> Result<Box<[ResolvedGemmRoute]>, String> {
    let context = ctx.gemm_route();
    let compiler = ctx.kernels.triad_scalar_compiler_identity();
    let mut routes = Vec::new();
    routes
        .try_reserve_exact(nodes.len())
        .map_err(|error| format!("reserve scalar route plan: {error}"))?;
    for node in nodes {
        let (backend, numeric_contract, ownership) = scalar_route_contract(node.symbol);
        routes.push(ResolvedGemmRoute {
            op: request.op,
            dtype: PolicyDtype::F32,
            backend,
            numeric_contract,
            instruction_family: ResolvedInstructionFamily::ScalarFma,
            instruction_shape: ResolvedInstructionShape { m: 1, n: 1, k: 1 },
            operand_conversion: ResolvedOperandConversion::None,
            ownership,
            symbol: node.symbol,
            module_kind: ModuleKind::TriadScalar,
            target: compiler.target,
            artifact: context.artifacts.triad_scalar,
            compiler,
            device: context.device,
            device_caps: context.device_caps,
            shape: (request.shape.m, request.shape.k, request.shape.n),
            strides: (request.shape.lda, request.shape.ldb, request.shape.ldc),
            tile: node.tile,
            bk: node.bk,
            stages: node.stages,
            threads: node.launch.block_dim.0 * node.launch.block_dim.1 * node.launch.block_dim.2,
            launch: node.launch,
            tensor_map_revision: 0,
            tensor_maps_digest: [0; 32],
            resources_digest,
            tuning_table_revision: TUNING_TABLE_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        });
    }
    Ok(routes.into_boxed_slice())
}

fn validate_f32_triad_operands(
    request: F32TriadRequest,
    operands: F32TriadOperands,
) -> Result<(), String> {
    let alignment = std::mem::align_of::<f32>() as CUptr;
    if operands.output == 0 || !operands.output.is_multiple_of(alignment) {
        return Err("f32 Triad output pointer must be non-null and 4-byte aligned".into());
    }
    if operands
        .bias
        .is_some_and(|bias| bias == 0 || !bias.is_multiple_of(alignment))
    {
        return Err("f32 Triad bias pointer must be non-null and 4-byte aligned".into());
    }
    if request.shape.reduction(request.op) != 0 {
        for (name, pointer) in [("A", operands.a), ("B", operands.b)] {
            if pointer == 0 || !pointer.is_multiple_of(alignment) {
                return Err(format!(
                    "f32 Triad {name} pointer must be non-null and 4-byte aligned"
                ));
            }
        }
    }
    match request.op {
        ResolvedGemmOp::Nn if operands.bias.is_some() && operands.alpha != 1.0 => {
            Err("f32 Triad NN bias requires alpha == 1.0".into())
        }
        ResolvedGemmOp::Tn if operands.bias.is_some() || operands.beta != 1.0 => {
            Err("f32 Triad TN requires no bias and beta == 1.0".into())
        }
        ResolvedGemmOp::Nt if operands.bias.is_some() || operands.beta != 0.0 => {
            Err("f32 Triad NT requires no bias and beta == 0.0".into())
        }
        _ => Ok(()),
    }
}

fn require_f32_preparation_outside_capture(ctx: &GpuCtx) -> Result<(), String> {
    let status = ctx
        .stream
        .capture_status()
        .map_err(|error| format!("query f32 Triad capture status: {error:?}"))?;
    if status != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        return Err("f32 Triad launch must be prepared before graph capture".into());
    }
    Ok(())
}

fn validated_allocation_domain(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    backend: &str,
) -> Result<AllocationDomain, String> {
    let allocation_domain = AllocationDomain::from_context(stream.context())?;
    let kernel_domain = kernels.allocation_domain();
    if allocation_domain != kernel_domain {
        return Err(format!(
            "{backend} stream allocation domain {allocation_domain:?} does not match kernel module allocation domain {kernel_domain:?}"
        ));
    }
    Ok(allocation_domain)
}

fn f32_map_binding(ctx: &GpuCtx, route: Tf32PhysicalRoute) -> Result<Tf32MapBinding, String> {
    let qualified = match route {
        Tf32PhysicalRoute::MmaTf32RnaV1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_) => {
            ctx.kernels.f32_triad_availability().portable
        }
        _ => ctx.kernels.f32_triad_availability().specialized,
    }
    .ok_or_else(|| format!("TF32 route {route:?} has no qualified module"))?;
    Ok(Tf32MapBinding {
        allocation_domain: validated_allocation_domain(&ctx.stream, &ctx.kernels, "TF32")?,
        qualified,
    })
}

fn prepare_specialized_tf32_maps(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: Tf32PhysicalRoute,
    binding: Tf32MapBinding,
) -> Result<F32PreparedTensorMaps, String> {
    if matches!(route, Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(_)) {
        let descriptor = build_sm90a_tf32_descriptor(0, 0, 64);
        if decode_sm90a_tf32_descriptor(descriptor) != (0, 0, 64) {
            return Err("SM90a TF32 descriptor constants changed".into());
        }
    }
    if matches!(route, Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(_)) {
        sm100_tf32_instruction_descriptor(request.op, tf32_kernel_spec(request.op, route)?.tile.1)?;
    }
    let capturing = ctx
        .stream
        .capture_status()
        .map_err(|error| format!("query CUDA capture status: {error:?}"))?
        != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    prepare_f32_maps_with(
        request,
        operands,
        route,
        binding,
        |request, operands, route| {
            tf32_tensor_map_plan(request, operands, route, binding.allocation_domain)
        },
        |plan| {
            ctx.kernels
                .triad_kernels()
                .prepare_tf32_tensor_maps(request, route, plan, capturing, binding)
        },
    )
}

fn tf32_params(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    origins: Tf32TensorOrigins,
    route: Tf32PhysicalRoute,
) -> Result<PreparedTf32Params, String> {
    let shape = request.shape;
    let m = checked_i32(shape.m, "M")?;
    let k = checked_i32(shape.k, "K")?;
    let n = checked_i32(shape.n, "N")?;
    let ldc = checked_i32(shape.ldc, "ldc")?;
    Ok(match route {
        Tf32PhysicalRoute::MmaTf32RnaV1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
        | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_) => {
            PreparedTf32Params::Sm80(Sm80Tf32KernelParams {
                alpha: operands.alpha,
                beta: operands.beta,
                m,
                k,
                n,
                lda: checked_i32(shape.lda, "lda")?,
                ldb: checked_i32(shape.ldb, "ldb")?,
                ldc,
            })
        }
        Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(_) => {
            PreparedTf32Params::Sm90a(Sm90aTf32KernelParams {
                a_x: origins.a_x,
                a_y: origins.a_y,
                b_x: origins.b_x,
                b_y: origins.b_y,
                alpha: operands.alpha,
                beta: operands.beta,
                m,
                k,
                n,
                ldc,
            })
        }
        Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(_) => {
            PreparedTf32Params::Sm100(Sm100KernelParams {
                a_x: origins.a_x,
                a_y: origins.a_y,
                b_x: origins.b_x,
                b_y: origins.b_y,
                alpha: operands.alpha,
                beta: operands.beta,
                m,
                k,
                n,
                ldc,
            })
        }
        Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_)
        | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_) => {
            PreparedTf32Params::Sm120(Sm120KernelParams {
                a_x: origins.a_x,
                a_y: origins.a_y,
                b_x: origins.b_x,
                b_y: origins.b_y,
                alpha: operands.alpha,
                beta: operands.beta,
                m,
                k,
                n,
                ldc,
            })
        }
        Tf32PhysicalRoute::Sm120TmaFmaExactV1(exact) => {
            // The exact kernels speak in output rows, output columns and the
            // reduction, whatever the operation stores as m, k and n.
            let plan = sm120_fma_launch_plan(request, exact)?;
            PreparedTf32Params::Sm120Fma(Sm120FmaKernelParams {
                alpha: operands.alpha,
                beta: operands.beta,
                m: checked_i32(shape.output_rows(request.op), "exact-F32 rows")?,
                n: checked_i32(shape.output_columns(request.op), "exact-F32 columns")?,
                k: checked_i32(shape.reduction(request.op), "exact-F32 reduction")?,
                ldc,
                splits: i32::from(exact.splits),
                tiles_per_split: checked_i32(plan.tiles_per_split, "exact-F32 tiles per split")?,
            })
        }
    })
}

/// The launch of an exact-F32 SM120 route: one unit per (tile, split), the
/// slabs the non-owning splits publish, and one flag per slab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sm120FmaLaunchPlan {
    pub tiles: usize,
    pub tiles_per_split: usize,
    pub units: u32,
    pub slab_elements: usize,
    pub flag_elements: usize,
}

pub(super) fn sm120_fma_launch_plan(
    request: F32TriadRequest,
    route: Sm120FmaRoute,
) -> Result<Sm120FmaLaunchPlan, String> {
    request.shape.validate(request.op)?;
    route.validate(request.op)?;
    let (bm, bn) = route.tile.dims();
    let rows = request.shape.output_rows(request.op);
    let columns = request.shape.output_columns(request.op);
    let reduction = request.shape.reduction(request.op);
    if reduction == 0 {
        return Err("exact-F32 SM120 route requires a nonzero reduction".into());
    }
    let tiles = rows
        .div_ceil(bm as usize)
        .checked_mul(columns.div_ceil(bn as usize))
        .ok_or_else(|| invalid_gemm_dimensions("exact-F32 tile count overflows usize"))?;
    let splits = usize::from(route.splits);
    let k_tiles = reduction.div_ceil(SM120_FMA_BK as usize);
    let tiles_per_split = k_tiles.div_ceil(splits);
    if tiles_per_split
        .checked_mul(splits - 1)
        .is_none_or(|covered| covered >= k_tiles)
    {
        return Err(invalid_gemm_dimensions(
            "exact-F32 split leaves a reduction range empty",
        ));
    }
    let units = tiles
        .checked_mul(splits)
        .and_then(|units| u32::try_from(units).ok())
        .filter(|units| *units <= i32::MAX as u32)
        .ok_or_else(|| {
            invalid_gemm_dimensions("exact-F32 unit count exceeds the kernel's range")
        })?;
    let slab_elements = tiles
        .checked_mul((bm * bn) as usize)
        .and_then(|slab| slab.checked_mul(splits - 1))
        .ok_or_else(|| invalid_gemm_dimensions("exact-F32 slab extent overflows usize"))?;
    if slab_elements > SPLITK_SCRATCH_CAP {
        return Err(invalid_gemm_dimensions(
            "exact-F32 slabs exceed the fixed workspace",
        ));
    }
    let flag_elements = tiles * (splits - 1);
    if flag_elements > TF32_SPLITK_COUNTER_CAP {
        return Err(invalid_gemm_dimensions(
            "exact-F32 flags exceed the fixed counter workspace",
        ));
    }
    Ok(Sm120FmaLaunchPlan {
        tiles,
        tiles_per_split,
        units,
        slab_elements,
        flag_elements,
    })
}

#[derive(Clone, Copy)]
struct Tf32LaunchDigests {
    maps: Sha256Digest,
    resources: Sha256Digest,
    arguments: Sha256Digest,
}

#[derive(Clone, Copy, Debug)]
struct Tf32SplitKLaunchPlan {
    scratch_elements: usize,
    counter_elements: usize,
    fused: cudarc::driver::LaunchConfig,
}

#[derive(Clone, Copy)]
struct Tf32SplitKWorkspace {
    // These context-owned buffers are shared by prepared split-K launches. The
    // recorded-stream gate below keeps every use on one ordered CUDA stream, so
    // workspace lifetimes cannot overlap across concurrent streams.
    partial: CUptr,
    counters: CUptr,
}

fn tf32_splitk_launch_plan(
    request: F32TriadRequest,
    spec: &Tf32SplitKSpec,
) -> Result<Tf32SplitKLaunchPlan, String> {
    request.shape.validate(request.op)?;
    if request.op != spec.op {
        return Err(format!(
            "portable TF32 split-K specification {:?} does not match {:?}",
            spec.op, request.op
        ));
    }
    let reduction = request.shape.reduction(request.op);
    if reduction == 0 {
        return Err("portable TF32 split-K requires a nonzero reduction".into());
    }
    let (_, covered) =
        tf32_splitk_partition_bounds(reduction, spec.partitions, spec.partitions - 1)?;
    if covered != reduction {
        return Err("portable TF32 split-K partitions do not cover the reduction".into());
    }
    let rows = request.shape.output_rows(request.op);
    let columns = request.shape.output_columns(request.op);
    let output_elements = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid_gemm_dimensions("TF32 split-K output size overflows usize"))?;
    let scratch_elements = output_elements
        .checked_mul(spec.partitions as usize)
        .ok_or_else(|| invalid_gemm_dimensions("TF32 split-K scratch size overflows usize"))?;
    if scratch_elements > SPLITK_SCRATCH_CAP {
        return Err(invalid_gemm_dimensions(
            "TF32 split-K scratch exceeds the fixed workspace",
        ));
    }
    let rows = checked_u32(rows, "TF32 split-K output rows")?;
    let columns = checked_u32(columns, "TF32 split-K output columns")?;
    let row_tiles = rows.div_ceil(spec.tile.0);
    if row_tiles > 65_535 {
        return Err(invalid_gemm_dimensions(
            "TF32 split-K row grid exceeds the CUDA y-dimension limit",
        ));
    }
    let column_tiles = columns.div_ceil(spec.tile.1);
    let counter_elements = usize::try_from(row_tiles)
        .ok()
        .and_then(|rows| {
            usize::try_from(column_tiles)
                .ok()
                .and_then(|columns| rows.checked_mul(columns))
        })
        .ok_or_else(|| invalid_gemm_dimensions("TF32 split-K counter size overflows usize"))?;
    if counter_elements > TF32_SPLITK_COUNTER_CAP {
        return Err(invalid_gemm_dimensions(
            "TF32 split-K counters exceed the fixed coordination workspace",
        ));
    }
    Ok(Tf32SplitKLaunchPlan {
        scratch_elements,
        counter_elements,
        fused: cudarc::driver::LaunchConfig {
            grid_dim: (column_tiles, row_tiles, spec.partitions),
            block_dim: (spec.threads, 1, 1),
            shared_mem_bytes: spec.dynamic_shared_bytes,
        },
    })
}

fn tf32_splitk_arguments_digest(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    symbol: &str,
    node_index: u8,
    partitions: u32,
    scratch_elements: usize,
) -> Sha256Digest {
    let domain: &[u8] = match partitions {
        2 => b"triad-tf32-split-k2-kernel-arguments.v2",
        4 => b"triad-tf32-split-k4-kernel-arguments.v2",
        8 => b"triad-tf32-split-k8-kernel-arguments.v2",
        _ => b"triad-tf32-invalid-split-k-kernel-arguments.v1",
    };
    let partition_bytes = [partitions as u8];
    FramedSha256::new(domain)
        .required(b"symbol", symbol.as_bytes())
        .required(b"node-index", &[node_index])
        .required(b"op", &[request.op as u8])
        .required(b"partitions", &partition_bytes)
        .required(b"m", &(request.shape.m as u64).to_le_bytes())
        .required(b"k", &(request.shape.k as u64).to_le_bytes())
        .required(b"n", &(request.shape.n as u64).to_le_bytes())
        .required(b"lda", &(request.shape.lda as u64).to_le_bytes())
        .required(b"ldb", &(request.shape.ldb as u64).to_le_bytes())
        .required(b"ldc", &(request.shape.ldc as u64).to_le_bytes())
        .required(b"alpha", &operands.alpha.to_bits().to_le_bytes())
        .required(b"beta", &operands.beta.to_bits().to_le_bytes())
        .required(b"bias-null", &[u8::from(operands.bias.is_none())])
        .required(
            b"scratch-elements",
            &(scratch_elements as u64).to_le_bytes(),
        )
        .finish()
}

fn tf32_splitk_resolved_routes(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    spec: &Tf32SplitKSpec,
    binding: Tf32MapBinding,
    resources_digest: Sha256Digest,
    plan: Tf32SplitKLaunchPlan,
) -> Box<[ResolvedGemmRoute]> {
    let qualified = binding.qualified;
    let shape = (request.shape.m, request.shape.k, request.shape.n);
    let strides = (request.shape.lda, request.shape.ldb, request.shape.ldc);
    let (backend, numeric_contract, ownership) = match spec.route {
        Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_) => (
            PhysicalGemmBackend::MmaTf32RnaSplitK2V1,
            ResolvedNumericContract::MmaTf32RnaSplitK2V1,
            ResolvedOutputOwnership::LastCtaPerOutputTileFixedSplitK2ReduceV1,
        ),
        Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_) => (
            PhysicalGemmBackend::MmaTf32RnaSplitK4V1,
            ResolvedNumericContract::MmaTf32RnaSplitK4V1,
            ResolvedOutputOwnership::LastCtaPerOutputTileFixedSplitK4ReduceV1,
        ),
        Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_) => (
            PhysicalGemmBackend::MmaTf32RnaSplitK8V1,
            ResolvedNumericContract::MmaTf32RnaSplitK8V1,
            ResolvedOutputOwnership::LastCtaPerOutputTileFixedSplitK8ReduceV1,
        ),
        _ => unreachable!("split-K spec admitted a non-split route"),
    };
    let fused = ResolvedGemmRoute {
        op: request.op,
        dtype: PolicyDtype::F32,
        backend,
        numeric_contract,
        instruction_family: ResolvedInstructionFamily::MmaSync,
        instruction_shape: ResolvedInstructionShape { m: 16, n: 8, k: 8 },
        operand_conversion: ResolvedOperandConversion::RegisterCvtRnaTf32F32V1,
        ownership,
        symbol: spec.symbol,
        module_kind: ModuleKind::TriadSm80,
        target: qualified.target,
        artifact: qualified.artifact,
        compiler: qualified.compiler,
        device: qualified.device,
        device_caps: qualified.device_caps,
        shape,
        strides,
        tile: spec.tile,
        bk: spec.bk,
        stages: spec.stages,
        threads: spec.threads,
        launch: ResolvedKernelLaunch {
            grid_dim: plan.fused.grid_dim,
            block_dim: plan.fused.block_dim,
            shared_mem_bytes: plan.fused.shared_mem_bytes,
            arguments_digest: tf32_splitk_arguments_digest(
                request,
                operands,
                spec.symbol,
                0,
                spec.partitions,
                plan.scratch_elements,
            ),
        },
        tensor_map_revision: 0,
        tensor_maps_digest: [0; 32],
        resources_digest,
        tuning_table_revision: F32_TF32_TUNING_REVISION,
        schedule_revision: SCHEDULE_REVISION,
    };
    vec![fused].into_boxed_slice()
}

fn tf32_resolved_route(
    request: F32TriadRequest,
    spec: &Tf32KernelSpec,
    binding: Tf32MapBinding,
    digests: Tf32LaunchDigests,
    zero_reduction: bool,
    config: cudarc::driver::LaunchConfig,
) -> ResolvedGemmRoute {
    let (backend, numeric_contract) = match spec.route {
        Tf32PhysicalRoute::MmaTf32RnaV1(_) => (
            PhysicalGemmBackend::MmaTf32RnaV1,
            ResolvedNumericContract::MmaTf32RnaV1,
        ),
        Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_) => (
            PhysicalGemmBackend::MmaTf32RnaSplitK2V1,
            ResolvedNumericContract::MmaTf32RnaSplitK2V1,
        ),
        Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_) => (
            PhysicalGemmBackend::MmaTf32RnaSplitK4V1,
            ResolvedNumericContract::MmaTf32RnaSplitK4V1,
        ),
        Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_) => (
            PhysicalGemmBackend::MmaTf32RnaSplitK8V1,
            ResolvedNumericContract::MmaTf32RnaSplitK8V1,
        ),
        Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(_) => (
            PhysicalGemmBackend::Sm90aWgmmaTf32TmaV1,
            ResolvedNumericContract::Sm90aWgmmaTf32TmaV1,
        ),
        Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(_) => (
            PhysicalGemmBackend::Sm100Tcgen05Tf32TmaV1,
            ResolvedNumericContract::Sm100Tcgen05Tf32TmaV1,
        ),
        Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_) => (
            PhysicalGemmBackend::Sm120TmaMmaTf32RnaV1,
            ResolvedNumericContract::Sm120TmaMmaTf32RnaV1,
        ),
        Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_) => (
            PhysicalGemmBackend::Sm120TmaMmaTf32RnaStreamKV1,
            ResolvedNumericContract::Sm120TmaMmaTf32RnaStreamKV1,
        ),
        Tf32PhysicalRoute::Sm120TmaFmaExactV1(exact) => (
            PhysicalGemmBackend::Sm120TmaFmaExactV1,
            sm120_fma_numeric_contract(exact),
        ),
    };
    let ownership = match spec.route {
        Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_) => {
            ResolvedOutputOwnership::OwnerCtaPerOutputTileStreamKFixedOrderV1
        }
        Tf32PhysicalRoute::Sm120TmaFmaExactV1(exact) => sm120_fma_ownership(exact),
        _ => ResolvedOutputOwnership::OneCtaPerOutputTileV1,
    };
    ResolvedGemmRoute {
        op: request.op,
        dtype: PolicyDtype::F32,
        backend,
        numeric_contract: if zero_reduction {
            ResolvedNumericContract::ZeroReductionEpilogueF32V1
        } else {
            numeric_contract
        },
        instruction_family: if zero_reduction {
            ResolvedInstructionFamily::ScalarFma
        } else {
            spec.instruction_family
        },
        instruction_shape: if zero_reduction {
            ResolvedInstructionShape { m: 1, n: 1, k: 1 }
        } else {
            spec.instruction_shape
        },
        operand_conversion: if zero_reduction {
            ResolvedOperandConversion::None
        } else {
            spec.operand_conversion
        },
        ownership,
        symbol: spec.symbol,
        module_kind: spec.module_kind,
        target: binding.qualified.target,
        artifact: binding.qualified.artifact,
        compiler: binding.qualified.compiler,
        device: binding.qualified.device,
        device_caps: binding.qualified.device_caps,
        shape: (request.shape.m, request.shape.k, request.shape.n),
        strides: (request.shape.lda, request.shape.ldb, request.shape.ldc),
        tile: spec.tile,
        bk: spec.bk,
        stages: spec.stages,
        threads: spec.threads,
        launch: ResolvedKernelLaunch {
            grid_dim: config.grid_dim,
            block_dim: config.block_dim,
            shared_mem_bytes: config.shared_mem_bytes,
            arguments_digest: digests.arguments,
        },
        tensor_map_revision: if zero_reduction {
            ZERO_REDUCTION_MAP_REVISION
        } else {
            spec.tensor_map_revision
        },
        tensor_maps_digest: digests.maps,
        resources_digest: digests.resources,
        tuning_table_revision: F32_TF32_TUNING_REVISION,
        schedule_revision: spec.schedule_revision,
    }
}

fn scalar_zero_symbol(op: ResolvedGemmOp) -> &'static str {
    match op {
        ResolvedGemmOp::Nn => "gemm_bi_nn_zero_reduction_v1",
        ResolvedGemmOp::Tn => "gemm_bi_tn_zero_reduction_v1",
        ResolvedGemmOp::Nt => "gemm_bi_nt_zero_reduction_v1",
    }
}

fn scalar_zero_route(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    maps_digest: Sha256Digest,
    resources_digest: Sha256Digest,
    config: cudarc::driver::LaunchConfig,
    arguments_digest: Sha256Digest,
) -> ResolvedGemmRoute {
    let context = ctx.gemm_route();
    let compiler = ctx.kernels.triad_scalar_compiler_identity();
    ResolvedGemmRoute {
        op: request.op,
        dtype: PolicyDtype::F32,
        backend: PhysicalGemmBackend::ScalarFmaV1,
        numeric_contract: ResolvedNumericContract::ZeroReductionEpilogueF32V1,
        instruction_family: ResolvedInstructionFamily::ScalarFma,
        instruction_shape: ResolvedInstructionShape { m: 1, n: 1, k: 1 },
        operand_conversion: ResolvedOperandConversion::None,
        ownership: ResolvedOutputOwnership::OneCtaPerOutputTileV1,
        symbol: scalar_zero_symbol(request.op),
        module_kind: ModuleKind::TriadScalar,
        target: compiler.target,
        artifact: context.artifacts.triad_scalar,
        compiler,
        device: context.device,
        device_caps: context.device_caps,
        shape: (request.shape.m, request.shape.k, request.shape.n),
        strides: (request.shape.lda, request.shape.ldb, request.shape.ldc),
        tile: (1, 1),
        bk: 0,
        stages: 1,
        threads: 256,
        launch: ResolvedKernelLaunch {
            grid_dim: config.grid_dim,
            block_dim: config.block_dim,
            shared_mem_bytes: config.shared_mem_bytes,
            arguments_digest,
        },
        tensor_map_revision: ZERO_REDUCTION_MAP_REVISION,
        tensor_maps_digest: maps_digest,
        resources_digest,
        tuning_table_revision: TUNING_TABLE_REVISION,
        schedule_revision: SCHEDULE_REVISION,
    }
}

struct ScalarScratchResources {
    split: Option<(CUptr, u64)>,
    transpose: Option<(CUptr, u64)>,
}

fn scalar_transpose_scratch_elements(
    request: F32TriadRequest,
    plan: ScalarDispatchPlan,
) -> Result<Option<usize>, String> {
    if !plan.needs_transpose_scratch() {
        return Ok(None);
    }
    if request.op != ResolvedGemmOp::Nt {
        return Err("scalar transpose scratch is only valid for NT plans".into());
    }
    let rows = match plan {
        ScalarDispatchPlan::NtSplitKTail { k_main, .. } => k_main,
        ScalarDispatchPlan::NtSplitKMain { .. }
        | ScalarDispatchPlan::NtSplitKSlim { .. }
        | ScalarDispatchPlan::NtD768TransposeM64N64Qualified
        | ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
        | ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
        | ScalarDispatchPlan::NtPrismVectorQualified
        | ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified => request.shape.k,
        _ => return Err("scalar plan declares unsupported transpose scratch".into()),
    };
    let elements = rows
        .checked_mul(request.shape.n)
        .ok_or_else(|| "scalar transpose scratch extent overflows usize".to_string())?;
    if elements > super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS {
        return Err(format!(
            "scalar transpose scratch requires {elements} f32 elements, capacity is {}",
            super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS
        ));
    }
    Ok(Some(elements))
}

fn scalar_launch_facts(kernels: &GpuKernels) -> ScalarLaunchFacts {
    ScalarLaunchFacts {
        scalar_artifact: kernels.artifact_set_identity().triad_scalar,
        scalar_compiler: kernels.triad_scalar_compiler_identity(),
        compute_capability: kernels.triad_scalar_compute_capability(),
        multiprocessor_count: kernels.multiprocessor_count(),
    }
}

fn scalar_scratch_resources(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    plan: ScalarDispatchPlan,
) -> Result<ScalarScratchResources, String> {
    use cudarc::driver::DevicePtr;
    let split = if plan.needs_split_scratch() {
        let buffer = ctx.kernels.splitk_scratch_buf(&ctx.stream)?;
        let (pointer, _) = buffer.device_ptr(&ctx.stream);
        Some((pointer, (SPLITK_SCRATCH_CAP as u64) * 4))
    } else {
        None
    };
    let transpose = if plan.needs_transpose_scratch() {
        let buffer = ctx.kernels.transpose_scratch_buf(&ctx.stream)?;
        let required = scalar_transpose_scratch_elements(request, plan)?
            .ok_or_else(|| "scalar transpose plan lost its scratch extent".to_string())?;
        if buffer.len() != super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS
            || required > buffer.len()
        {
            return Err(format!(
                "scalar transpose scratch allocation has {} f32 elements, requires {required} with exact capacity {}",
                buffer.len(),
                super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS
            ));
        }
        let (pointer, _) = buffer.device_ptr(&ctx.stream);
        Some((
            pointer,
            u64::try_from(super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS)
                .map_err(|_| "scalar transpose scratch capacity overflows u64")?
                .checked_mul(4)
                .ok_or_else(|| {
                    "scalar transpose scratch byte capacity overflows u64".to_string()
                })?,
        ))
    } else {
        None
    };
    Ok(ScalarScratchResources { split, transpose })
}

fn prepare_scalar_f32(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    output_resources: F32LaunchResourceSnapshot,
) -> Result<PreparedF32TriadLaunch, String> {
    let plan = scalar_launch_plan(scalar_launch_facts(&ctx.kernels), request, operands)?;
    prepare_scalar_f32_with_plan(ctx, request, operands, output_resources, plan)
}

fn prepare_scalar_f32_with_plan(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    output_resources: F32LaunchResourceSnapshot,
    plan: ScalarDispatchPlan,
) -> Result<PreparedF32TriadLaunch, String> {
    if operands.beta != 0.0 && scalar_plan_requires_zero_beta(plan) {
        return Err("selected exact scalar plan requires beta == 0".into());
    }
    let allocation_domain = validated_allocation_domain(&ctx.stream, &ctx.kernels, "f32 Triad")?;
    let resources = output_resources.with_inputs(request, operands, allocation_domain)?;
    let scratch = scalar_scratch_resources(ctx, request, plan)?;
    let resources =
        resources.with_scratch(scratch.split, scratch.transpose, None, allocation_domain)?;
    let resources_digest = resources.digest(request, operands, [0; 32]);
    let nodes = scalar_physical_nodes(request, operands, plan)?;
    let routes = scalar_resolved_routes(ctx, request, resources_digest, &nodes)?;
    let resolved_launch_set = build_resolved_gemm_launch_set(&routes)?;
    let managed_epoch = resources.managed_epoch();
    Ok(PreparedF32TriadLaunch {
        context_token: ctx.instance_token(),
        stream_token: ctx.stream_token(),
        request,
        operands,
        resources,
        managed_epoch,
        routes,
        resolved_launch_set,
        kind: PreparedF32Kind::Scalar(plan),
    })
}

fn prepare_scalar_zero_f32(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    resources: F32LaunchResourceSnapshot,
    maps: F32PreparedTensorMaps,
) -> Result<PreparedF32TriadLaunch, String> {
    let maps_digest = maps.identity_digest();
    let resources_digest = resources.digest(request, operands, maps_digest);
    let rows = request.shape.output_rows(request.op);
    let columns = request.shape.output_columns(request.op);
    let total = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid_gemm_dimensions("zero-reduction output size overflows usize"))?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (
            checked_u32(total.div_ceil(256), "zero-reduction grid.x")?,
            1,
            1,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let symbol = scalar_zero_symbol(request.op);
    let null_pointer_mask = 0b110_u64 | (u64::from(operands.bias.is_none()) << 3);
    let arguments_digest = FramedSha256::new(b"triad-scalar-zero-reduction-arguments.v1")
        .required(b"symbol", symbol.as_bytes())
        .required(b"op", &[request.op as u8])
        .required(b"m", &(request.shape.m as u64).to_le_bytes())
        .required(b"k", &(request.shape.k as u64).to_le_bytes())
        .required(b"n", &(request.shape.n as u64).to_le_bytes())
        .required(b"lda", &(request.shape.lda as u64).to_le_bytes())
        .required(b"ldb", &(request.shape.ldb as u64).to_le_bytes())
        .required(b"ldc", &(request.shape.ldc as u64).to_le_bytes())
        .required(b"alpha", &operands.alpha.to_bits().to_le_bytes())
        .required(b"beta", &operands.beta.to_bits().to_le_bytes())
        .required(b"output-offset", &0_u64.to_le_bytes())
        .required(b"null-pointer-mask", &null_pointer_mask.to_le_bytes())
        .finish();
    let route = scalar_zero_route(
        ctx,
        request,
        maps_digest,
        resources_digest,
        config,
        arguments_digest,
    );
    let routes = vec![route].into_boxed_slice();
    let resolved_launch_set = build_zero_reduction_route_identity(routes[0])?;
    let params = SgbZeroReductionParams {
        alpha: operands.alpha,
        beta: operands.beta,
        m: checked_i32(request.shape.m, "M")?,
        k: checked_i32(request.shape.k, "K")?,
        n: checked_i32(request.shape.n, "N")?,
        lda: checked_i32(request.shape.lda, "lda")?,
        ldb: checked_i32(request.shape.ldb, "ldb")?,
        ldc: checked_i32(request.shape.ldc, "ldc")?,
    };
    let managed_epoch = resources.managed_epoch();
    Ok(PreparedF32TriadLaunch {
        context_token: ctx.instance_token(),
        stream_token: ctx.stream_token(),
        request,
        operands,
        resources,
        managed_epoch,
        routes,
        resolved_launch_set,
        kind: PreparedF32Kind::ScalarZero {
            maps,
            params,
            config,
        },
    })
}

fn prepare_tf32_f32(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    output_resources: F32LaunchResourceSnapshot,
    route: Tf32PhysicalRoute,
) -> Result<PreparedF32TriadLaunch, String> {
    let availability = ctx.kernels.f32_triad_availability();
    let route = resolve_tf32_forced(request, availability, route).map_err(|error| {
        ctx.kernels
            .tf32_qualification_rejection(route)
            .map(|rejection| {
                format!("forced TF32 route {route:?} failed module qualification: {rejection}")
            })
            .unwrap_or(error)
    })?;
    if matches!(
        route,
        Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
            | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
            | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_)
    ) {
        return prepare_tf32_splitk_f32(ctx, request, operands, output_resources, route);
    }
    if matches!(route, Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_)) {
        return prepare_tf32_streamk_f32(ctx, request, operands, output_resources, route);
    }
    if route.is_exact_fma() {
        return prepare_sm120_fma_f32(ctx, request, operands, output_resources, route);
    }
    let spec = tf32_kernel_spec(request.op, route)?;
    let binding = f32_map_binding(ctx, route)?;
    let allocation_domain = binding.allocation_domain;
    let zero_reduction = request.shape.reduction(request.op) == 0;
    let resources = if zero_reduction {
        output_resources
    } else {
        output_resources.with_inputs(request, operands, allocation_domain)?
    };
    let maps = if zero_reduction {
        Some(F32PreparedTensorMaps::zero_reduction(
            request,
            Some(route),
            Some(binding),
            match route {
                Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_)
                | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_) => {
                    Tf32TensorMapFormat::Uint32V1
                }
                _ => Tf32TensorMapFormat::Tfloat32V1,
            },
        ))
    } else if matches!(route, Tf32PhysicalRoute::MmaTf32RnaV1(_)) {
        None
    } else {
        Some(prepare_specialized_tf32_maps(
            ctx, request, operands, route, binding,
        )?)
    };
    let origins = maps
        .as_ref()
        .map(F32PreparedTensorMaps::origins)
        .unwrap_or_default();
    let maps_digest = maps
        .as_ref()
        .map(F32PreparedTensorMaps::identity_digest)
        .unwrap_or([0; 32]);
    let resources_digest = resources.digest(request, operands, maps_digest);
    let rows = checked_u32(request.shape.output_rows(request.op), "TF32 output rows")?;
    let columns = checked_u32(
        request.shape.output_columns(request.op),
        "TF32 output columns",
    )?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (
            checked_tile_grid(rows, spec.tile.0, columns, spec.tile.1)?,
            1,
            1,
        ),
        block_dim: (spec.threads, 1, 1),
        shared_mem_bytes: spec.dynamic_shared_bytes,
    };
    let arguments_digest =
        tf32_kernel_arguments_digest(request, operands, spec.symbol, maps_digest);
    let resolved = tf32_resolved_route(
        request,
        spec,
        binding,
        Tf32LaunchDigests {
            maps: maps_digest,
            resources: resources_digest,
            arguments: arguments_digest,
        },
        zero_reduction,
        config,
    );
    let routes = vec![resolved].into_boxed_slice();
    let resolved_launch_set = build_resolved_gemm_launch_set(&routes)?;
    let managed_epoch = resources.managed_epoch();
    Ok(PreparedF32TriadLaunch {
        context_token: ctx.instance_token(),
        stream_token: ctx.stream_token(),
        request,
        operands,
        resources,
        managed_epoch,
        routes,
        resolved_launch_set,
        kind: PreparedF32Kind::Tf32 {
            route,
            maps,
            params: tf32_params(request, operands, origins, route)?,
            config,
        },
    })
}

fn prepare_tf32_streamk_f32(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    output_resources: F32LaunchResourceSnapshot,
    route: Tf32PhysicalRoute,
) -> Result<PreparedF32TriadLaunch, String> {
    use cudarc::driver::DevicePtr;

    let spec = tf32_kernel_spec(request.op, route)?;
    // An empty reduction walks the persistent grid over the epilogue alone:
    // no operand maps, no inputs, the same placeholder maps as the tiled
    // SM120 routes.
    let zero_reduction = request.shape.reduction(request.op) == 0;
    let plan = tf32_streamk_launch_plan(request, spec, ctx.kernels.multiprocessor_count())?;
    let binding = f32_map_binding(ctx, route)?;
    let allocation_domain = binding.allocation_domain;
    let resources = if zero_reduction {
        output_resources
    } else {
        output_resources.with_inputs(request, operands, allocation_domain)?
    };
    let scratch_buffer = ctx.kernels.splitk_scratch_buf(&ctx.stream)?;
    let (partial, _) = scratch_buffer.device_ptr(&ctx.stream);
    let flag_buffer = ctx
        .kernels
        .triad_kernels()
        .tf32_splitk_counter_buf(&ctx.stream)?;
    let (flags, _) = flag_buffer.device_ptr(&ctx.stream);
    let resources = resources.with_scratch(
        Some((partial, (SPLITK_SCRATCH_CAP as u64) * 4)),
        None,
        Some((flags, (TF32_SPLITK_COUNTER_CAP as u64) * 4)),
        allocation_domain,
    )?;
    let maps = if zero_reduction {
        F32PreparedTensorMaps::zero_reduction(
            request,
            Some(route),
            Some(binding),
            Tf32TensorMapFormat::Uint32V1,
        )
    } else {
        prepare_specialized_tf32_maps(ctx, request, operands, route, binding)?
    };
    let origins = maps.origins();
    let maps_digest = maps.identity_digest();
    let resources_digest = resources.digest(request, operands, maps_digest);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (plan.grid, 1, 1),
        block_dim: (spec.threads, 1, 1),
        shared_mem_bytes: spec.dynamic_shared_bytes,
    };
    let arguments_digest =
        tf32_kernel_arguments_digest(request, operands, spec.symbol, maps_digest);
    let resolved = tf32_resolved_route(
        request,
        spec,
        binding,
        Tf32LaunchDigests {
            maps: maps_digest,
            resources: resources_digest,
            arguments: arguments_digest,
        },
        zero_reduction,
        config,
    );
    let routes = vec![resolved].into_boxed_slice();
    let resolved_launch_set = build_resolved_gemm_launch_set(&routes)?;
    let managed_epoch = resources.managed_epoch();
    Ok(PreparedF32TriadLaunch {
        context_token: ctx.instance_token(),
        stream_token: ctx.stream_token(),
        request,
        operands,
        resources,
        managed_epoch,
        routes,
        resolved_launch_set,
        kind: PreparedF32Kind::Tf32StreamK {
            route,
            maps,
            params: tf32_params(request, operands, origins, route)?,
            config,
            plan,
            workspace: Tf32StreamKWorkspace { partial, flags },
        },
    })
}

/// Prepares an exact-F32 SM120 launch. It reuses the stream-K shape of a
/// prepared launch: tensor maps, a slab and flag workspace, one grid of
/// units. With one split the workspace goes unused and the chain matches
/// the scalar route bit for bit.
fn prepare_sm120_fma_f32(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    output_resources: F32LaunchResourceSnapshot,
    route: Tf32PhysicalRoute,
) -> Result<PreparedF32TriadLaunch, String> {
    use cudarc::driver::DevicePtr;

    let exact = route
        .exact_fma()
        .ok_or_else(|| "exact-F32 preparation requires an exact route".to_string())?;
    let spec = tf32_kernel_spec(request.op, route)?;
    let plan = sm120_fma_launch_plan(request, exact)?;
    let binding = f32_map_binding(ctx, route)?;
    let allocation_domain = binding.allocation_domain;
    let resources = output_resources.with_inputs(request, operands, allocation_domain)?;
    let (resources, partial, flags) = if exact.splits == 1 {
        debug_assert_eq!(plan.slab_elements, 0);
        debug_assert_eq!(plan.flag_elements, 0);
        (resources, 0, 0)
    } else {
        let scratch_buffer = ctx.kernels.splitk_scratch_buf(&ctx.stream)?;
        let (partial, _) = scratch_buffer.device_ptr(&ctx.stream);
        let flag_buffer = ctx
            .kernels
            .triad_kernels()
            .tf32_splitk_counter_buf(&ctx.stream)?;
        let (flags, _) = flag_buffer.device_ptr(&ctx.stream);
        let resources = resources.with_scratch(
            Some((partial, (SPLITK_SCRATCH_CAP as u64) * 4)),
            None,
            Some((flags, (TF32_SPLITK_COUNTER_CAP as u64) * 4)),
            allocation_domain,
        )?;
        (resources, partial, flags)
    };
    let maps = prepare_specialized_tf32_maps(ctx, request, operands, route, binding)?;
    let origins = maps.origins();
    let maps_digest = maps.identity_digest();
    let resources_digest = resources.digest(request, operands, maps_digest);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (plan.units, 1, 1),
        block_dim: (spec.threads, 1, 1),
        shared_mem_bytes: spec.dynamic_shared_bytes,
    };
    let arguments_digest =
        tf32_kernel_arguments_digest(request, operands, spec.symbol, maps_digest);
    let resolved = tf32_resolved_route(
        request,
        spec,
        binding,
        Tf32LaunchDigests {
            maps: maps_digest,
            resources: resources_digest,
            arguments: arguments_digest,
        },
        false,
        config,
    );
    let resolved = ResolvedGemmRoute {
        numeric_contract: sm120_fma_numeric_contract(exact),
        ownership: sm120_fma_ownership(exact),
        ..resolved
    };
    let routes = vec![resolved].into_boxed_slice();
    let resolved_launch_set = build_resolved_gemm_launch_set(&routes)?;
    let managed_epoch = resources.managed_epoch();
    Ok(PreparedF32TriadLaunch {
        context_token: ctx.instance_token(),
        stream_token: ctx.stream_token(),
        request,
        operands,
        resources,
        managed_epoch,
        routes,
        resolved_launch_set,
        kind: PreparedF32Kind::Tf32StreamK {
            route,
            maps,
            params: tf32_params(request, operands, origins, route)?,
            config,
            plan: Tf32StreamKLaunchPlan {
                grid: plan.units,
                partial_elements: plan.slab_elements,
                flag_elements: plan.flag_elements,
            },
            workspace: Tf32StreamKWorkspace { partial, flags },
        },
    })
}

/// One split keeps the scalar chain; more splits fold fixed-order partials.
fn sm120_fma_numeric_contract(route: Sm120FmaRoute) -> ResolvedNumericContract {
    if route.splits == 1 {
        ResolvedNumericContract::ScalarFmaV1
    } else {
        ResolvedNumericContract::ScalarFmaFixedSplitFoldV1
    }
}

fn sm120_fma_ownership(route: Sm120FmaRoute) -> ResolvedOutputOwnership {
    if route.splits == 1 {
        ResolvedOutputOwnership::OneCtaPerOutputTileV1
    } else {
        ResolvedOutputOwnership::OwnerCtaPerOutputTileFixedSplitFoldV1
    }
}

fn prepare_tf32_splitk_f32(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    output_resources: F32LaunchResourceSnapshot,
    route: Tf32PhysicalRoute,
) -> Result<PreparedF32TriadLaunch, String> {
    use cudarc::driver::DevicePtr;

    let spec = tf32_splitk_spec(request.op, route)?;
    let plan = tf32_splitk_launch_plan(request, spec)?;
    let binding = f32_map_binding(ctx, route)?;
    let allocation_domain = binding.allocation_domain;
    let resources = output_resources.with_inputs(request, operands, allocation_domain)?;
    let scratch_buffer = ctx.kernels.splitk_scratch_buf(&ctx.stream)?;
    let (scratch, _) = scratch_buffer.device_ptr(&ctx.stream);
    let counter_buffer = ctx
        .kernels
        .triad_kernels()
        .tf32_splitk_counter_buf(&ctx.stream)?;
    let (counters, _) = counter_buffer.device_ptr(&ctx.stream);
    let resources = resources.with_scratch(
        Some((scratch, (SPLITK_SCRATCH_CAP as u64) * 4)),
        None,
        Some((counters, (TF32_SPLITK_COUNTER_CAP as u64) * 4)),
        allocation_domain,
    )?;
    let resources_digest = resources.digest(request, operands, [0; 32]);
    let routes =
        tf32_splitk_resolved_routes(request, operands, spec, binding, resources_digest, plan);
    let resolved_launch_set = build_resolved_gemm_launch_set(&routes)?;
    let PreparedTf32Params::Sm80(params) =
        tf32_params(request, operands, Tf32TensorOrigins::default(), route)?
    else {
        return Err("portable TF32 split-K resolved a non-SM80 parameter ABI".into());
    };
    let managed_epoch = resources.managed_epoch();
    Ok(PreparedF32TriadLaunch {
        context_token: ctx.instance_token(),
        stream_token: ctx.stream_token(),
        request,
        operands,
        resources,
        managed_epoch,
        routes,
        resolved_launch_set,
        kind: PreparedF32Kind::Tf32SplitK {
            route,
            params,
            plan,
            workspace: Tf32SplitKWorkspace {
                partial: scratch,
                counters,
            },
        },
    })
}

fn tf32_kernel_arguments_digest(
    request: F32TriadRequest,
    operands: F32TriadOperands,
    symbol: &str,
    maps_digest: Sha256Digest,
) -> Sha256Digest {
    FramedSha256::new(b"triad-tf32-kernel-arguments.v2")
        .required(b"symbol", symbol.as_bytes())
        .required(b"op", &[request.op as u8])
        .required(b"m", &(request.shape.m as u64).to_le_bytes())
        .required(b"k", &(request.shape.k as u64).to_le_bytes())
        .required(b"n", &(request.shape.n as u64).to_le_bytes())
        .required(b"lda", &(request.shape.lda as u64).to_le_bytes())
        .required(b"ldb", &(request.shape.ldb as u64).to_le_bytes())
        .required(b"ldc", &(request.shape.ldc as u64).to_le_bytes())
        .required(b"alpha", &operands.alpha.to_bits().to_le_bytes())
        .required(b"beta", &operands.beta.to_bits().to_le_bytes())
        .required(b"bias-null", &[u8::from(operands.bias.is_none())])
        .required(b"tensor-maps", &maps_digest)
        .finish()
}

fn physical_prepared_f32_route(
    prepared: &PreparedF32TriadLaunch,
    route: ResolvedGemmRoute,
) -> ResolvedGemmRoute {
    let mut physical = route;
    physical.resources_digest = prepared.resources.physical_digest();
    if let PreparedF32Kind::Tf32 { maps, .. } = &prepared.kind {
        let maps_digest = maps
            .as_ref()
            .map(F32PreparedTensorMaps::physical_identity_digest)
            .unwrap_or([0; 32]);
        physical.tensor_maps_digest = maps_digest;
    }
    physical
}

pub(in crate::mamba_ssm::gpu) fn prepare_f32_triad(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
) -> Result<PreparedF32TriadLaunch, String> {
    request.shape.validate(request.op)?;
    validate_f32_triad_operands(request, operands)?;
    require_f32_preparation_outside_capture(ctx)?;
    let allocation_domain = validated_allocation_domain(&ctx.stream, &ctx.kernels, "f32 Triad")?;
    let output_resources =
        F32LaunchResourceSnapshot::query_output(request, operands, allocation_domain)?;
    let reduction = request.shape.reduction(request.op);
    let zero_reduction = reduction == 0;
    if zero_reduction {
        let maps = F32PreparedTensorMaps::zero_reduction(
            request,
            None,
            None,
            Tf32TensorMapFormat::Tfloat32V1,
        );
        let prepared = prepare_scalar_zero_f32(ctx, request, operands, output_resources, maps)?;
        return Ok(prepared);
    }
    match resolve_f32_triad_auto_with_operands(
        ctx.f32_triad_policy(),
        request,
        operands,
        ctx.kernels.f32_triad_availability(),
    )? {
        F32TriadSelection::ScalarFmaV1 => {
            prepare_scalar_f32(ctx, request, operands, output_resources)
        }
        F32TriadSelection::Tf32(route) => {
            // A measured TF32 route whose kernel this toolkit could not serve
            // declines to the exact floor, with the reason said once.
            if let Ok(spec) = super::contract::tf32_kernel_spec(request.op, route)
                && let Some(reason) = ctx
                    .kernels
                    .triad_kernels()
                    .tf32_symbol_exclusion(spec.symbol)
            {
                static EXCLUDED: std::sync::Once = std::sync::Once::new();
                crate::mamba_ssm::gpu::diagnostics::warn_once(&EXCLUDED, || {
                    format!(
                        "TF32 route {} is excluded on this toolkit ({reason}); the exact \
                         family serves this shape",
                        spec.symbol
                    )
                });
                return match super::dispatch::exact_or_scalar_selection(
                    request,
                    Some(operands),
                    ctx.kernels.f32_triad_availability(),
                ) {
                    F32TriadSelection::ExactSm120Fma(route) => prepare_tf32_f32(
                        ctx,
                        request,
                        operands,
                        output_resources,
                        Tf32PhysicalRoute::Sm120TmaFmaExactV1(route),
                    ),
                    _ => prepare_scalar_f32(ctx, request, operands, output_resources),
                };
            }
            prepare_tf32_f32(ctx, request, operands, output_resources, route)
        }
        F32TriadSelection::ExactSm120Fma(route) => prepare_tf32_f32(
            ctx,
            request,
            operands,
            output_resources,
            Tf32PhysicalRoute::Sm120TmaFmaExactV1(route),
        ),
    }
}

fn prepare_exact_scalar_f32_triad(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
) -> Result<PreparedF32TriadLaunch, String> {
    request.shape.validate(request.op)?;
    validate_f32_triad_operands(request, operands)?;
    require_f32_preparation_outside_capture(ctx)?;
    let allocation_domain = validated_allocation_domain(&ctx.stream, &ctx.kernels, "f32 Triad")?;
    let output_resources =
        F32LaunchResourceSnapshot::query_output(request, operands, allocation_domain)?;
    if request.shape.reduction(request.op) == 0 {
        let maps = F32PreparedTensorMaps::zero_reduction(
            request,
            None,
            None,
            Tf32TensorMapFormat::Tfloat32V1,
        );
        return prepare_scalar_zero_f32(ctx, request, operands, output_resources, maps);
    }
    prepare_scalar_f32(ctx, request, operands, output_resources)
}

impl F32PreparedLaunchCache {
    fn prepare(
        ctx: &GpuCtx,
        selection: F32PreparedSelection,
        request: F32TriadRequest,
        operands: F32TriadOperands,
    ) -> Result<PreparedF32TriadLaunch, String> {
        match selection {
            F32PreparedSelection::Automatic => prepare_f32_triad(ctx, request, operands),
            F32PreparedSelection::ExactScalar => {
                prepare_exact_scalar_f32_triad(ctx, request, operands)
            }
            F32PreparedSelection::Forced(route) => {
                prepare_f32_triad_forced(ctx, request, operands, route)
            }
        }
    }

    fn ensure_prepared(
        &mut self,
        ctx: &GpuCtx,
        selection: F32PreparedSelection,
        request: F32TriadRequest,
        operands: F32TriadOperands,
    ) -> Result<&PreparedF32TriadLaunch, String> {
        let key = PreparedF32Key::new(
            ctx.instance_token(),
            ctx.gemm_policy(),
            selection,
            request,
            operands,
        );
        let stale = match self.entries.get_mut(&key) {
            Some(prepared) => {
                let mut managed_epoch = prepared.managed_epoch.take();
                let validation = refresh_cached_validation(
                    &mut managed_epoch,
                    || {
                        ctx.stream
                            .capture_status()
                            .map(|status| {
                                status
                                    != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
                            })
                            .map_err(|error| {
                                format!("query f32 Triad capture status: {error:?}")
                            })
                    },
                    || {
                        let refreshed = prepared.resources.managed_epoch();
                        validate_prepared_f32_triad(ctx, prepared)?;
                        Ok(refreshed)
                    },
                );
                prepared.managed_epoch = managed_epoch;
                validation.is_err()
            }
            None => false,
        };
        if !self.entries.contains_key(&key) || stale {
            let capturing = ctx
                .stream
                .capture_status()
                .map_err(|error| format!("query f32 Triad capture status: {error:?}"))?
                != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            if capturing {
                let reason = if stale { "stale" } else { "missing" };
                return Err(format!(
                    "prepared f32 Triad cache entry is {reason} during graph capture; run eager warmup again"
                ));
            }
            let prepared = Self::prepare(ctx, selection, request, operands)?;
            make_room_in_bounded_cache(
                &mut self.entries,
                &key,
                F32_PREPARED_CACHE_LIMIT,
                |cached| {
                    cached
                        .managed_epoch
                        .as_ref()
                        .is_none_or(ManagedAllocationEpochStamp::is_current)
                },
            );
            if !self.entries.contains_key(&key) {
                self.entries
                    .try_reserve(1)
                    .map_err(|error| format!("reserve prepared f32 Triad cache: {error}"))?;
            }
            self.entries.insert(key, Box::new(prepared));
        }
        Ok(self
            .entries
            .get(&key)
            .expect("prepared f32 Triad cache entry was inserted above"))
    }

    fn launch<Scalar>(
        &mut self,
        ctx: &GpuCtx,
        selection: F32PreparedSelection,
        request: F32TriadRequest,
        operands: F32TriadOperands,
        scalar: Scalar,
    ) -> Result<(), String>
    where
        Scalar: FnOnce(&mut ScalarLaunchControl<'_>) -> Result<(), String>,
    {
        let prepared = self.ensure_prepared(ctx, selection, request, operands)?;
        unsafe { enqueue_validated_prepared_f32_triad(ctx, prepared, scalar) }
    }

    fn launch_observed<O, Scalar>(
        &mut self,
        ctx: &GpuCtx,
        selection: F32PreparedSelection,
        request: F32TriadRequest,
        operands: F32TriadOperands,
        physical: (&mut O, PolicyDtype),
        scalar: Scalar,
    ) -> Result<(), String>
    where
        O: PhysicalLaunchObserver,
        Scalar: FnOnce(&mut PhysicalScalarLaunchControl<'_, O>) -> Result<(), String>,
    {
        let prepared = self.ensure_prepared(ctx, selection, request, operands)?;
        unsafe {
            enqueue_validated_prepared_f32_triad_observed(
                ctx, prepared, physical.0, physical.1, scalar,
            )
        }
    }
}

fn refresh_cached_validation<Capture, Validate>(
    stamp: &mut Option<ManagedAllocationEpochStamp>,
    capture_status: Capture,
    validate: Validate,
) -> Result<(), String>
where
    Capture: FnOnce() -> Result<bool, String>,
    Validate: FnOnce() -> Result<Option<ManagedAllocationEpochStamp>, String>,
{
    if stamp
        .as_ref()
        .is_some_and(ManagedAllocationEpochStamp::is_current)
    {
        return Ok(());
    }
    if stamp.is_some() && capture_status()? {
        return Err(
            "prepared f32 Triad allocation epoch changed during graph capture; run eager warmup again"
                .into(),
        );
    }
    let refreshed = validate()?;
    if refreshed
        .as_ref()
        .is_some_and(|candidate| !candidate.is_current())
    {
        return Err("managed CUDA allocation epoch changed during f32 Triad validation".into());
    }
    *stamp = refreshed;
    Ok(())
}

fn launch_cached_f32_triad<Scalar>(
    ctx: &GpuCtx,
    selection: F32PreparedSelection,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    scalar: Scalar,
) -> Result<(), String>
where
    Scalar: FnOnce(&mut ScalarLaunchControl<'_>) -> Result<(), String>,
{
    ctx.with_f32_prepared_launches(|cache| cache.launch(ctx, selection, request, operands, scalar))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FixedSm120ExactTmaTile {
    M128N64,
    M64N128,
}

/// Reuses the qualified Triad SM120 exact-TMA implementation from the Fixed
/// family without entering Triad AUTO. The prepared-cache key seals the
/// literal physical route, pointers, policy and shape, so graph capture keeps
/// the same descriptor binding as the eager warmup.
pub(crate) fn launch_cached_fixed_sm120_exact_tma(
    ctx: &GpuCtx,
    shape: (usize, usize, usize),
    operands: F32TriadOperands,
    tile: FixedSm120ExactTmaTile,
) -> Result<bool, String> {
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Nn,
        shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, shape),
    };
    let route = Tf32PhysicalRoute::Sm120TmaFmaExactV1(Sm120FmaRoute {
        tile: match tile {
            FixedSm120ExactTmaTile::M128N64 => Sm120FmaTile::M128N64,
            FixedSm120ExactTmaTile::M64N128 => Sm120FmaTile::M64N128,
        },
        kvec: false,
        splits: 1,
    });
    if resolve_tf32_forced(request, ctx.kernels.f32_triad_availability(), route).is_err() {
        return Ok(false);
    }
    let spec = tf32_kernel_spec(request.op, route)?;
    if ctx
        .kernels
        .triad_kernels()
        .tf32_function(spec.symbol)
        .is_none()
    {
        return Ok(false);
    }
    launch_cached_f32_triad(
        ctx,
        F32PreparedSelection::Forced(route),
        request,
        operands,
        |_| Err("Fixed exact-TMA bridge unexpectedly entered the scalar fallback".into()),
    )?;
    Ok(true)
}

pub(in crate::mamba_ssm::gpu) fn with_cached_f32_triad_prepared<R>(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    use_prepared: impl FnOnce(&PreparedF32TriadLaunch) -> Result<R, String>,
) -> Result<R, String> {
    ctx.with_f32_prepared_launches(|cache| {
        let prepared =
            cache.ensure_prepared(ctx, F32PreparedSelection::Automatic, request, operands)?;
        use_prepared(prepared)
    })
}

fn launch_cached_f32_triad_observed<O, Scalar>(
    ctx: &GpuCtx,
    selection: F32PreparedSelection,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    physical: (&mut O, PolicyDtype),
    scalar: Scalar,
) -> Result<(), String>
where
    O: PhysicalLaunchObserver,
    Scalar: FnOnce(&mut PhysicalScalarLaunchControl<'_, O>) -> Result<(), String>,
{
    ctx.with_f32_prepared_launches(|cache| {
        cache.launch_observed(ctx, selection, request, operands, physical, scalar)
    })
}

fn launch_cached_f32_forward_selected(
    ctx: &GpuCtx,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: CUptr,
    bias_ptr: CUptr,
    dims: (usize, usize, usize),
    selection: F32PreparedSelection,
) -> Result<(), String> {
    let shape = F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims);
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Nn,
        shape,
    };
    let reduction_is_zero = shape.reduction(request.op) == 0;
    let output = y.raw_ptr(&ctx.stream);
    let x_ptr = if reduction_is_zero {
        0
    } else {
        x.raw_ptr(&ctx.stream)
    };
    let w_ptr = if reduction_is_zero { 0 } else { w_ptr };
    let operands = F32TriadOperands {
        output,
        a: x_ptr,
        b: w_ptr,
        bias: (bias_ptr != 0).then_some(bias_ptr),
        alpha: 1.0,
        beta: 0.0,
    };
    let scalar_operands = GemmBiFwdSubOperands {
        x_ptr,
        lda: shape.lda,
        w_ptr,
        bias_ptr,
    };
    launch_cached_f32_triad(ctx, selection, request, operands, |control| {
        gemm_bi_forward_sub_with_control(
            &ctx.stream,
            &ctx.kernels,
            y,
            &scalar_operands,
            dims,
            Some(control),
        )
    })
}

pub(crate) fn launch_cached_f32_forward(
    ctx: &GpuCtx,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: CUptr,
    bias_ptr: CUptr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    launch_cached_f32_forward_selected(
        ctx,
        y,
        x,
        w_ptr,
        bias_ptr,
        dims,
        F32PreparedSelection::Automatic,
    )
}

#[derive(Clone, Copy)]
pub(crate) struct ScalarFallbackPhysicalContext {
    pub(crate) dims: (usize, usize, usize),
    pub(crate) dtype: WeightDtype,
}

enum ScalarPhysicalGraphDispatch<'a> {
    Nn {
        y: &'a mut GpuBuffer,
        x: &'a GpuBuffer,
        w_ptr: CUptr,
        bias_ptr: CUptr,
        dims: (usize, usize, usize),
    },
    Tn {
        dw_ptr: CUptr,
        dy: &'a GpuBuffer,
        x_saved: &'a GpuBuffer,
        dims: (usize, usize, usize),
    },
    Nt {
        dx: &'a mut GpuBuffer,
        dy: &'a GpuBuffer,
        w_ptr: CUptr,
        dims: (usize, usize, usize),
    },
}

fn prepare_exact_scalar_physical_graph_sequence<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &O,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    logical_dtype: PolicyDtype,
    dispatch: ScalarPhysicalGraphDispatch<'_>,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    ctx.with_f32_prepared_launches(|cache| {
        let prepared =
            cache.ensure_prepared(ctx, F32PreparedSelection::ExactScalar, request, operands)?;
        prepare_prepared_scalar_physical_graph_sequence(
            ctx,
            observer,
            prepared,
            logical_dtype,
            dispatch,
        )
    })
}

fn prepare_prepared_scalar_physical_graph_sequence<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &O,
    prepared: &PreparedF32TriadLaunch,
    logical_dtype: PolicyDtype,
    dispatch: ScalarPhysicalGraphDispatch<'_>,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    validate_prepared_f32_triad(ctx, prepared)?;
    let PreparedF32Kind::Scalar(plan) = prepared.kind else {
        return Err("prepared scalar graph package resolved a non-scalar F32 launch".into());
    };
    let mut launches = Vec::new();
    launches
        .try_reserve_exact(prepared.routes.len())
        .map_err(|error| format!("reserve prepared scalar physical launches: {error}"))?;
    let base = ScalarLaunchControl {
        ctx,
        routes: &prepared.routes,
        plan,
        operands: prepared.operands,
        next: 0,
    };
    let mut control = PreparedPhysicalScalarLaunchControl {
        base,
        logical_dtype,
        physical_resources_digest: prepared.resources.physical_digest(),
        observer,
        launches,
    };
    match dispatch {
        ScalarPhysicalGraphDispatch::Nn {
            y,
            x,
            w_ptr,
            bias_ptr,
            dims,
        } => {
            let scalar_operands = GemmBiFwdSubOperands {
                x_ptr: prepared.operands.a,
                lda: prepared.request.shape.lda,
                w_ptr,
                bias_ptr,
            };
            gemm_bi_forward_sub_with_control(
                &ctx.stream,
                &ctx.kernels,
                y,
                &scalar_operands,
                dims,
                Some(&mut control),
            )?;
            if x.raw_ptr(&ctx.stream) != prepared.operands.a {
                return Err("prepared scalar NN input binding changed".into());
            }
        }
        ScalarPhysicalGraphDispatch::Tn {
            dw_ptr,
            dy,
            x_saved,
            dims,
        } => gemm_bi_backward_dw_with_control(
            &ctx.stream,
            &ctx.kernels,
            dw_ptr,
            dy,
            x_saved,
            dims,
            Some(&mut control),
        )?,
        ScalarPhysicalGraphDispatch::Nt {
            dx,
            dy,
            w_ptr,
            dims,
        } => gemm_bi_backward_dx_with_control(
            &ctx.stream,
            &ctx.kernels,
            dx,
            dy,
            w_ptr,
            dims,
            Some(&mut control),
        )?,
    }
    control.finish()
}

pub(in crate::mamba_ssm::gpu) fn prepare_prepared_f32_forward_graph_sequence<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &O,
    prepared: &PreparedF32TriadLaunch,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    if prepared.request.op != ResolvedGemmOp::Nn {
        return Err("prepared F32 forward graph sequence requires an NN launch".into());
    }
    prepare_prepared_scalar_physical_graph_sequence(
        ctx,
        observer,
        prepared,
        PolicyDtype::F32,
        ScalarPhysicalGraphDispatch::Nn {
            y,
            x,
            w_ptr: prepared.operands.b,
            bias_ptr: prepared.operands.bias.unwrap_or(0),
            dims: (
                prepared.request.shape.m,
                prepared.request.shape.k,
                prepared.request.shape.n,
            ),
        },
    )
}

pub(in crate::mamba_ssm::gpu) fn prepare_prepared_f32_backward_dw_graph_sequence<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &O,
    prepared: &PreparedF32TriadLaunch,
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    if prepared.request.op != ResolvedGemmOp::Tn {
        return Err("prepared F32 backward-dW graph sequence requires a TN launch".into());
    }
    prepare_prepared_scalar_physical_graph_sequence(
        ctx,
        observer,
        prepared,
        PolicyDtype::F32,
        ScalarPhysicalGraphDispatch::Tn {
            dw_ptr: prepared.operands.output,
            dy,
            x_saved,
            dims: (
                prepared.request.shape.m,
                prepared.request.shape.k,
                prepared.request.shape.n,
            ),
        },
    )
}

pub(in crate::mamba_ssm::gpu) fn prepare_prepared_f32_backward_dx_graph_sequence<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &O,
    prepared: &PreparedF32TriadLaunch,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    if prepared.request.op != ResolvedGemmOp::Nt {
        return Err("prepared F32 backward-dX graph sequence requires an NT launch".into());
    }
    prepare_prepared_scalar_physical_graph_sequence(
        ctx,
        observer,
        prepared,
        PolicyDtype::F32,
        ScalarPhysicalGraphDispatch::Nt {
            dx,
            dy,
            w_ptr: prepared.operands.b,
            dims: (
                prepared.request.shape.m,
                prepared.request.shape.k,
                prepared.request.shape.n,
            ),
        },
    )
}

pub(in crate::mamba_ssm::gpu) fn prepare_prepared_f32_direct_graph_sequence<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &O,
    prepared: &PreparedF32TriadLaunch,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    validate_prepared_f32_triad(ctx, prepared)?;
    if let PreparedF32Kind::Tf32SplitK {
        params,
        plan,
        workspace,
        ..
    } = &prepared.kind
    {
        return prepare_tf32_splitk_direct_graph_sequence(
            ctx,
            observer,
            prepared,
            params,
            *plan,
            workspace.partial,
            workspace.counters,
        );
    }
    if prepared.routes.len() != 1 {
        return Err("prepared direct F32 graph sequence requires exactly one route".into());
    }
    let route = physical_prepared_f32_route(prepared, prepared.routes[0]);
    let config = LaunchConfig {
        grid_dim: route.launch.grid_dim,
        block_dim: route.launch.block_dim,
        shared_mem_bytes: route.launch.shared_mem_bytes,
    };
    let observation = PhysicalLaunchObservation::gemm(PolicyDtype::F32, None, route);
    let node = resolve_physical_launch_observation(observer, observation, config)?;
    let output = prepared.operands.output;
    let bias = prepared.operands.bias.unwrap_or(0);
    let mut arguments = PhysicalScalarKernelArguments::new();
    let function = match &prepared.kind {
        PreparedF32Kind::Scalar(_) => {
            return Err(
                "prepared scalar F32 graph sequence requires its operation-level producer".into(),
            );
        }
        PreparedF32Kind::ScalarZero { params, .. } => {
            let null_input = 0_u64;
            arguments.push(output)?;
            arguments.push(null_input)?;
            arguments.push(null_input)?;
            arguments.push(bias)?;
            arguments.push(*params)?;
            let kernels = ctx.kernels.triad_kernels();
            match prepared.request.op {
                ResolvedGemmOp::Nn => kernels.gemm_bi_nn_zero_reduction.clone(),
                ResolvedGemmOp::Tn => kernels.gemm_bi_tn_zero_reduction.clone(),
                ResolvedGemmOp::Nt => kernels.gemm_bi_nt_zero_reduction.clone(),
            }
        }
        PreparedF32Kind::Tf32 {
            route: physical_route,
            maps,
            params,
            ..
        } => {
            match (*physical_route, *params) {
                (Tf32PhysicalRoute::MmaTf32RnaV1(_), PreparedTf32Params::Sm80(params)) => {
                    let reduction_is_zero =
                        prepared.request.shape.reduction(prepared.request.op) == 0;
                    arguments.push(output)?;
                    arguments.push(if reduction_is_zero {
                        0
                    } else {
                        prepared.operands.a
                    })?;
                    arguments.push(if reduction_is_zero {
                        0
                    } else {
                        prepared.operands.b
                    })?;
                    arguments.push(bias)?;
                    arguments.push(params)?;
                }
                (Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(_), PreparedTf32Params::Sm90a(params)) => {
                    let maps = maps
                        .as_ref()
                        .ok_or_else(|| "SM90a TF32 graph sequence has no tensor maps".to_string())?
                        .maps();
                    arguments.push(output)?;
                    arguments.push(maps[0])?;
                    arguments.push(maps[1])?;
                    arguments.push(bias)?;
                    arguments.push(params)?;
                }
                (
                    Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(_),
                    PreparedTf32Params::Sm100(params),
                ) => {
                    let maps = maps
                        .as_ref()
                        .ok_or_else(|| "SM100 TF32 graph sequence has no tensor maps".to_string())?
                        .maps();
                    arguments.push(output)?;
                    arguments.push(maps[0])?;
                    arguments.push(maps[1])?;
                    arguments.push(bias)?;
                    arguments.push(params)?;
                }
                (Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_), PreparedTf32Params::Sm120(params)) => {
                    let maps = maps
                        .as_ref()
                        .ok_or_else(|| "SM120 TF32 graph sequence has no tensor maps".to_string())?
                        .maps();
                    arguments.push(output)?;
                    arguments.push(maps[0])?;
                    arguments.push(maps[1])?;
                    arguments.push(bias)?;
                    arguments.push(params)?;
                }
                _ => return Err("prepared TF32 graph route and parameter ABI disagree".into()),
            }
            ctx.kernels
                .triad_kernels()
                .tf32_function(route.symbol)
                .ok_or_else(|| format!("qualified TF32 symbol {} is unavailable", route.symbol))?
                .clone()
        }
        PreparedF32Kind::Tf32SplitK { .. } => {
            return Err("prepared TF32 split-K graph sequence was not expanded".into());
        }
        PreparedF32Kind::Tf32StreamK {
            maps,
            params,
            workspace,
            ..
        } => {
            let maps = maps.maps();
            arguments.push(output)?;
            arguments.push(workspace.partial)?;
            arguments.push(workspace.flags)?;
            arguments.push(maps[0])?;
            arguments.push(maps[1])?;
            arguments.push(bias)?;
            match params {
                PreparedTf32Params::Sm120(params) => arguments.push(*params)?,
                PreparedTf32Params::Sm120Fma(params) => arguments.push(*params)?,
                _ => {
                    return Err("prepared TF32 stream-K route and parameter ABI disagree".into());
                }
            }
            ctx.kernels
                .triad_kernels()
                .tf32_function(route.symbol)
                .ok_or_else(|| format!("qualified TF32 symbol {} is unavailable", route.symbol))?
                .clone()
        }
    };
    Ok(PreparedTriadPhysicalGraphSequence {
        launches: vec![PreparedTriadPhysicalGraphLaunch {
            function,
            config,
            node,
            arguments: Box::new(arguments),
        }]
        .into_boxed_slice(),
    })
}

fn prepare_tf32_splitk_direct_graph_sequence<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &O,
    prepared: &PreparedF32TriadLaunch,
    params: &Sm80Tf32KernelParams,
    plan: Tf32SplitKLaunchPlan,
    scratch: CUptr,
    counters: CUptr,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    let [resolved] = prepared.routes.as_ref() else {
        return Err("prepared fused TF32 split-K graph sequence requires one route".into());
    };
    let config = plan.fused;
    let route = physical_prepared_f32_route(prepared, *resolved);
    if route.launch.grid_dim != config.grid_dim
        || route.launch.block_dim != config.block_dim
        || route.launch.shared_mem_bytes != config.shared_mem_bytes
    {
        return Err("prepared fused TF32 split-K graph launch configuration changed".into());
    }
    let observation = PhysicalLaunchObservation::gemm(PolicyDtype::F32, None, route);
    let node = resolve_physical_launch_observation(observer, observation, config)?;
    let mut arguments = PhysicalScalarKernelArguments::new();
    arguments.push(prepared.operands.output)?;
    arguments.push(scratch)?;
    arguments.push(counters)?;
    arguments.push(prepared.operands.a)?;
    arguments.push(prepared.operands.b)?;
    arguments.push(prepared.operands.bias.unwrap_or(0))?;
    arguments.push(*params)?;
    let function = ctx
        .kernels
        .triad_kernels()
        .tf32_splitk_function(route.symbol)
        .ok_or_else(|| {
            format!(
                "qualified fused TF32 split-K symbol {} is unavailable",
                route.symbol
            )
        })?
        .clone();
    Ok(PreparedTriadPhysicalGraphSequence {
        launches: vec![PreparedTriadPhysicalGraphLaunch {
            function,
            config,
            node,
            arguments: Box::new(arguments),
        }]
        .into_boxed_slice(),
    })
}

pub(in crate::mamba_ssm::gpu) fn prepare_exact_scalar_f32_forward_graph_sequence<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &O,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: CUptr,
    bias_ptr: CUptr,
    physical: ScalarFallbackPhysicalContext,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    let ScalarFallbackPhysicalContext { dims, dtype } = physical;
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Nn,
        shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
    };
    let operands = F32TriadOperands {
        output: y.raw_ptr(&ctx.stream),
        a: x.raw_ptr(&ctx.stream),
        b: w_ptr,
        bias: (bias_ptr != 0).then_some(bias_ptr),
        alpha: 1.0,
        beta: 0.0,
    };
    prepare_exact_scalar_physical_graph_sequence(
        ctx,
        observer,
        request,
        operands,
        half_policy_dtype(dtype)?,
        ScalarPhysicalGraphDispatch::Nn {
            y,
            x,
            w_ptr,
            bias_ptr,
            dims,
        },
    )
}

pub(in crate::mamba_ssm::gpu) fn prepare_exact_scalar_f32_backward_dw_graph_sequence<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &O,
    dw_ptr: CUptr,
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    physical: ScalarFallbackPhysicalContext,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    let ScalarFallbackPhysicalContext { dims, dtype } = physical;
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Tn,
        shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims),
    };
    let operands = F32TriadOperands {
        output: dw_ptr,
        a: x_saved.raw_ptr(&ctx.stream),
        b: dy.raw_ptr(&ctx.stream),
        bias: None,
        alpha: 1.0,
        beta: 1.0,
    };
    prepare_exact_scalar_physical_graph_sequence(
        ctx,
        observer,
        request,
        operands,
        half_policy_dtype(dtype)?,
        ScalarPhysicalGraphDispatch::Tn {
            dw_ptr,
            dy,
            x_saved,
            dims,
        },
    )
}

pub(in crate::mamba_ssm::gpu) fn prepare_exact_scalar_f32_backward_dx_graph_sequence<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &O,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
    w_ptr: CUptr,
    physical: ScalarFallbackPhysicalContext,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    let ScalarFallbackPhysicalContext { dims, dtype } = physical;
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Nt,
        shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
    };
    let operands = F32TriadOperands {
        output: dx.raw_ptr(&ctx.stream),
        a: dy.raw_ptr(&ctx.stream),
        b: w_ptr,
        bias: None,
        alpha: 1.0,
        beta: 0.0,
    };
    prepare_exact_scalar_physical_graph_sequence(
        ctx,
        observer,
        request,
        operands,
        half_policy_dtype(dtype)?,
        ScalarPhysicalGraphDispatch::Nt {
            dx,
            dy,
            w_ptr,
            dims,
        },
    )
}

pub(in crate::mamba_ssm::gpu) fn record_physical_exact_scalar_f32_forward<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &mut O,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: CUptr,
    bias_ptr: CUptr,
    physical: ScalarFallbackPhysicalContext,
) -> Result<(), String> {
    let ScalarFallbackPhysicalContext { dims, dtype } = physical;
    let shape = F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims);
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Nn,
        shape,
    };
    let operands = F32TriadOperands {
        output: y.raw_ptr(&ctx.stream),
        a: x.raw_ptr(&ctx.stream),
        b: w_ptr,
        bias: (bias_ptr != 0).then_some(bias_ptr),
        alpha: 1.0,
        beta: 0.0,
    };
    let scalar_operands = GemmBiFwdSubOperands {
        x_ptr: operands.a,
        lda: shape.lda,
        w_ptr,
        bias_ptr,
    };
    if O::ENABLED {
        let logical_dtype = half_policy_dtype(dtype)?;
        return launch_cached_f32_triad_observed(
            ctx,
            F32PreparedSelection::ExactScalar,
            request,
            operands,
            (observer, logical_dtype),
            |control| {
                gemm_bi_forward_sub_with_control(
                    &ctx.stream,
                    &ctx.kernels,
                    y,
                    &scalar_operands,
                    dims,
                    Some(control),
                )
            },
        );
    }
    launch_cached_f32_triad(
        ctx,
        F32PreparedSelection::ExactScalar,
        request,
        operands,
        |control| {
            gemm_bi_forward_sub_with_control(
                &ctx.stream,
                &ctx.kernels,
                y,
                &scalar_operands,
                dims,
                Some(control),
            )
        },
    )
}

fn launch_cached_f32_backward_dw_selected(
    ctx: &GpuCtx,
    dw_ptr: CUptr,
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    dims: (usize, usize, usize),
    selection: F32PreparedSelection,
) -> Result<(), String> {
    let shape = F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims);
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Tn,
        shape,
    };
    let reduction_is_zero = shape.reduction(request.op) == 0;
    let operands = F32TriadOperands {
        output: dw_ptr,
        a: if reduction_is_zero {
            0
        } else {
            x_saved.raw_ptr(&ctx.stream)
        },
        b: if reduction_is_zero {
            0
        } else {
            dy.raw_ptr(&ctx.stream)
        },
        bias: None,
        alpha: 1.0,
        beta: 1.0,
    };
    launch_cached_f32_triad(ctx, selection, request, operands, |control| {
        gemm_bi_backward_dw_with_control(
            &ctx.stream,
            &ctx.kernels,
            dw_ptr,
            dy,
            x_saved,
            dims,
            Some(control),
        )
    })
}

pub(crate) fn launch_cached_f32_backward_dw(
    ctx: &GpuCtx,
    dw_ptr: CUptr,
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    launch_cached_f32_backward_dw_selected(
        ctx,
        dw_ptr,
        dy,
        x_saved,
        dims,
        F32PreparedSelection::Automatic,
    )
}

pub(in crate::mamba_ssm::gpu) fn record_physical_exact_scalar_f32_backward_dw<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &mut O,
    dw_ptr: CUptr,
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    physical: ScalarFallbackPhysicalContext,
) -> Result<(), String> {
    let ScalarFallbackPhysicalContext { dims, dtype } = physical;
    let shape = F32TriadShape::contiguous(ResolvedGemmOp::Tn, dims);
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Tn,
        shape,
    };
    let operands = F32TriadOperands {
        output: dw_ptr,
        a: x_saved.raw_ptr(&ctx.stream),
        b: dy.raw_ptr(&ctx.stream),
        bias: None,
        alpha: 1.0,
        beta: 1.0,
    };
    if O::ENABLED {
        let logical_dtype = half_policy_dtype(dtype)?;
        return launch_cached_f32_triad_observed(
            ctx,
            F32PreparedSelection::ExactScalar,
            request,
            operands,
            (observer, logical_dtype),
            |control| {
                gemm_bi_backward_dw_with_control(
                    &ctx.stream,
                    &ctx.kernels,
                    dw_ptr,
                    dy,
                    x_saved,
                    dims,
                    Some(control),
                )
            },
        );
    }
    launch_cached_f32_triad(
        ctx,
        F32PreparedSelection::ExactScalar,
        request,
        operands,
        |control| {
            gemm_bi_backward_dw_with_control(
                &ctx.stream,
                &ctx.kernels,
                dw_ptr,
                dy,
                x_saved,
                dims,
                Some(control),
            )
        },
    )
}

fn launch_cached_f32_backward_dx_selected(
    ctx: &GpuCtx,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
    w_ptr: CUptr,
    dims: (usize, usize, usize),
    selection: F32PreparedSelection,
) -> Result<(), String> {
    let shape = F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims);
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Nt,
        shape,
    };
    let reduction_is_zero = shape.reduction(request.op) == 0;
    let operands = F32TriadOperands {
        output: dx.raw_ptr(&ctx.stream),
        a: if reduction_is_zero {
            0
        } else {
            dy.raw_ptr(&ctx.stream)
        },
        b: if reduction_is_zero { 0 } else { w_ptr },
        bias: None,
        alpha: 1.0,
        beta: 0.0,
    };
    launch_cached_f32_triad(ctx, selection, request, operands, |control| {
        gemm_bi_backward_dx_with_control(
            &ctx.stream,
            &ctx.kernels,
            dx,
            dy,
            w_ptr,
            dims,
            Some(control),
        )
    })
}

pub(crate) fn launch_cached_f32_backward_dx(
    ctx: &GpuCtx,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
    w_ptr: CUptr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    launch_cached_f32_backward_dx_selected(
        ctx,
        dx,
        dy,
        w_ptr,
        dims,
        F32PreparedSelection::Automatic,
    )
}

pub(in crate::mamba_ssm::gpu) fn record_physical_exact_scalar_f32_backward_dx<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &mut O,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
    w_ptr: CUptr,
    physical: ScalarFallbackPhysicalContext,
) -> Result<(), String> {
    let ScalarFallbackPhysicalContext { dims, dtype } = physical;
    let shape = F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims);
    let request = F32TriadRequest {
        op: ResolvedGemmOp::Nt,
        shape,
    };
    let operands = F32TriadOperands {
        output: dx.raw_ptr(&ctx.stream),
        a: dy.raw_ptr(&ctx.stream),
        b: w_ptr,
        bias: None,
        alpha: 1.0,
        beta: 0.0,
    };
    if O::ENABLED {
        let logical_dtype = half_policy_dtype(dtype)?;
        return launch_cached_f32_triad_observed(
            ctx,
            F32PreparedSelection::ExactScalar,
            request,
            operands,
            (observer, logical_dtype),
            |control| {
                gemm_bi_backward_dx_with_control(
                    &ctx.stream,
                    &ctx.kernels,
                    dx,
                    dy,
                    w_ptr,
                    dims,
                    Some(control),
                )
            },
        );
    }
    launch_cached_f32_triad(
        ctx,
        F32PreparedSelection::ExactScalar,
        request,
        operands,
        |control| {
            gemm_bi_backward_dx_with_control(
                &ctx.stream,
                &ctx.kernels,
                dx,
                dy,
                w_ptr,
                dims,
                Some(control),
            )
        },
    )
}

pub(in crate::mamba_ssm::gpu) fn prepare_f32_triad_forced(
    ctx: &GpuCtx,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: Tf32PhysicalRoute,
) -> Result<PreparedF32TriadLaunch, String> {
    request.shape.validate(request.op)?;
    validate_f32_triad_operands(request, operands)?;
    require_f32_preparation_outside_capture(ctx)?;
    let allocation_domain = validated_allocation_domain(&ctx.stream, &ctx.kernels, "f32 Triad")?;
    let output_resources =
        F32LaunchResourceSnapshot::query_output(request, operands, allocation_domain)?;
    prepare_tf32_f32(ctx, request, operands, output_resources, route)
}

fn validate_prepared_f32_triad(
    ctx: &GpuCtx,
    prepared: &PreparedF32TriadLaunch,
) -> Result<(), String> {
    if prepared.context_token != ctx.instance_token() {
        return Err("prepared f32 Triad launch belongs to another GPU context".into());
    }
    if prepared.stream_token != ctx.stream_token() {
        return Err("prepared f32 Triad launch belongs to another CUDA stream".into());
    }
    prepared.resources.validate_live()?;
    match &prepared.kind {
        PreparedF32Kind::Scalar(_) => {}
        PreparedF32Kind::ScalarZero { maps, .. } => maps.validate_live_allocations()?,
        PreparedF32Kind::Tf32 { route, maps, .. } => {
            if let Some(maps) = maps {
                maps.validate_live_allocations()?;
                let binding = f32_map_binding(ctx, *route)?;
                if !maps.matches_binding(binding) {
                    return Err("prepared TF32 tensor-map binding changed before launch".into());
                }
            }
        }
        PreparedF32Kind::Tf32SplitK { route, plan, .. } => {
            f32_map_binding(ctx, *route)?;
            validate_tf32_splitk_prepared_layout(prepared, *route, *plan)?;
        }
        PreparedF32Kind::Tf32StreamK {
            route, maps, plan, ..
        } => {
            maps.validate_live_allocations()?;
            let binding = f32_map_binding(ctx, *route)?;
            if !maps.matches_binding(binding) {
                return Err("prepared TF32 tensor-map binding changed before launch".into());
            }
            validate_tf32_streamk_prepared_layout(ctx, prepared, *route, *plan)?;
        }
    }
    let mut live = ResolvedGemmLaunchSetBuilder::new(prepared.routes.len())?;
    for route in &prepared.routes {
        ctx.validate_resolved_gemm_route(route, "prepared f32 Triad")?;
        live.push(route)?;
    }
    prepared
        .resolved_launch_set
        .ensure_current(live.finish()?, "prepared f32 Triad")
}

pub(in crate::mamba_ssm::gpu) fn validate_prepared_f32_triad_for_timing(
    ctx: &GpuCtx,
    prepared: &PreparedF32TriadLaunch,
) -> Result<(), String> {
    validate_prepared_f32_triad(ctx, prepared)
}

fn validate_tf32_streamk_prepared_layout(
    ctx: &GpuCtx,
    prepared: &PreparedF32TriadLaunch,
    route: Tf32PhysicalRoute,
    plan: Tf32StreamKLaunchPlan,
) -> Result<(), String> {
    let spec = tf32_kernel_spec(prepared.request.op, route)?;
    let [resolved] = prepared.routes.as_ref() else {
        return Err("prepared TF32 stream-K launch requires exactly one route".into());
    };
    if resolved.symbol != spec.symbol {
        return Err("prepared TF32 stream-K route changed".into());
    }
    let expected = match route.exact_fma() {
        Some(exact) => {
            let plan = sm120_fma_launch_plan(prepared.request, exact)?;
            Tf32StreamKLaunchPlan {
                grid: plan.units,
                partial_elements: plan.slab_elements,
                flag_elements: plan.flag_elements,
            }
        }
        None => {
            tf32_streamk_launch_plan(prepared.request, spec, ctx.kernels.multiprocessor_count())?
        }
    };
    if expected.grid != plan.grid
        || expected.partial_elements != plan.partial_elements
        || expected.flag_elements != plan.flag_elements
    {
        return Err("prepared TF32 stream-K launch plan changed".into());
    }
    if resolved.launch.grid_dim != (plan.grid, 1, 1)
        || resolved.launch.block_dim != (spec.threads, 1, 1)
        || resolved.launch.shared_mem_bytes != spec.dynamic_shared_bytes
    {
        return Err("prepared TF32 stream-K launch configuration changed".into());
    }
    Ok(())
}

fn validate_tf32_splitk_prepared_layout(
    prepared: &PreparedF32TriadLaunch,
    route: Tf32PhysicalRoute,
    plan: Tf32SplitKLaunchPlan,
) -> Result<(), String> {
    let spec = tf32_splitk_spec(prepared.request.op, route)?;
    let [fused] = prepared.routes.as_ref() else {
        return Err("prepared fused TF32 split-K launch requires exactly one route".into());
    };
    if fused.symbol != spec.symbol {
        return Err("prepared fused TF32 split-K route changed".into());
    }
    let expected = tf32_splitk_launch_plan(prepared.request, spec)?;
    if expected.scratch_elements != plan.scratch_elements
        || expected.counter_elements != plan.counter_elements
    {
        return Err("prepared fused TF32 split-K workspace extent changed".into());
    }
    if expected.fused.grid_dim != plan.fused.grid_dim
        || expected.fused.block_dim != plan.fused.block_dim
        || expected.fused.shared_mem_bytes != plan.fused.shared_mem_bytes
        || fused.launch.grid_dim != plan.fused.grid_dim
        || fused.launch.block_dim != plan.fused.block_dim
        || fused.launch.shared_mem_bytes != plan.fused.shared_mem_bytes
    {
        return Err("prepared fused TF32 split-K launch configuration changed".into());
    }
    Ok(())
}

unsafe fn enqueue_scalar_zero_f32<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    prepared: &PreparedF32TriadLaunch,
    params: &SgbZeroReductionParams,
    config: cudarc::driver::LaunchConfig,
    observer: &mut O,
    observation: Option<PhysicalLaunchObservation>,
) -> Result<(), String> {
    let kernels = ctx.kernels.triad_kernels();
    let function = match prepared.request.op {
        ResolvedGemmOp::Nn => &kernels.gemm_bi_nn_zero_reduction,
        ResolvedGemmOp::Tn => &kernels.gemm_bi_tn_zero_reduction,
        ResolvedGemmOp::Nt => &kernels.gemm_bi_nt_zero_reduction,
    };
    let output = prepared.operands.output;
    let null_input = 0_u64;
    let bias = prepared.operands.bias.unwrap_or(0);
    let mut builder = ctx.stream.launch_builder(function);
    builder.arg(&output);
    builder.arg(&null_input);
    builder.arg(&null_input);
    builder.arg(&bias);
    builder.arg(params);
    unsafe { enqueue_with_physical_observation(observer, &mut builder, config, observation) }
        .map_err(|error| error.with_driver_context(format_args!("{}", prepared.routes[0].symbol)))
}

unsafe fn enqueue_tf32_f32<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    prepared: &PreparedF32TriadLaunch,
    launch: Tf32RawLaunch<'_>,
    observer: &mut O,
) -> Result<(), String> {
    let function = ctx
        .kernels
        .triad_kernels()
        .tf32_function(prepared.routes[0].symbol)
        .ok_or_else(|| {
            format!(
                "qualified TF32 symbol {} is unavailable",
                prepared.routes[0].symbol
            )
        })?;
    unsafe { enqueue_tf32_raw(&ctx.stream, function, launch, observer) }
}

unsafe fn enqueue_tf32_splitk_f32<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    prepared: &PreparedF32TriadLaunch,
    params: &Sm80Tf32KernelParams,
    plan: Tf32SplitKLaunchPlan,
    workspace: Tf32SplitKWorkspace,
    observer: &mut O,
    logical_dtype: Option<PolicyDtype>,
) -> Result<(), String> {
    let [resolved] = prepared.routes.as_ref() else {
        return Err("prepared fused TF32 split-K launch requires exactly one route".into());
    };
    let function = ctx
        .kernels
        .triad_kernels()
        .tf32_splitk_function(resolved.symbol)
        .ok_or_else(|| {
            format!(
                "qualified fused TF32 split-K symbol {} is unavailable",
                resolved.symbol
            )
        })?;
    ctx.record_resolved_gemm_route(*resolved)?;
    let physical = physical_prepared_f32_route(prepared, *resolved);
    let observation =
        logical_dtype.map(|dtype| PhysicalLaunchObservation::gemm(dtype, None, physical));
    let output = prepared.operands.output;
    let bias = prepared.operands.bias.unwrap_or(0);
    let mut builder = ctx.stream.launch_builder(function);
    builder.arg(&output);
    builder.arg(&workspace.partial);
    builder.arg(&workspace.counters);
    builder.arg(&prepared.operands.a);
    builder.arg(&prepared.operands.b);
    builder.arg(&bias);
    builder.arg(params);
    unsafe { enqueue_with_physical_observation(observer, &mut builder, plan.fused, observation) }
        .map_err(|error| error.with_driver_context(format_args!("{}", resolved.symbol)))
}

unsafe fn enqueue_tf32_raw<O: PhysicalLaunchObserver>(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    launch: Tf32RawLaunch<'_>,
    observer: &mut O,
) -> Result<(), String> {
    let output = launch.operands.output;
    let bias = launch.operands.bias.unwrap_or(0);
    match (launch.route, launch.params) {
        (Tf32PhysicalRoute::MmaTf32RnaV1(_), PreparedTf32Params::Sm80(params)) => {
            let a = if launch.zero_reduction {
                0
            } else {
                launch.operands.a
            };
            let b = if launch.zero_reduction {
                0
            } else {
                launch.operands.b
            };
            let mut builder = stream.launch_builder(function);
            builder.arg(&output);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&bias);
            builder.arg(&params);
            unsafe {
                enqueue_with_physical_observation(
                    observer,
                    &mut builder,
                    launch.config,
                    launch.observation,
                )
            }
            .map_err(|error| error.with_driver_context(format_args!("{}", launch.symbol)))
        }
        (Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(_), PreparedTf32Params::Sm90a(params)) => {
            let maps = launch
                .maps
                .ok_or_else(|| "SM90a TF32 launch has no tensor maps".to_string())?
                .maps();
            let mut builder = stream.launch_builder(function);
            builder.arg(&output);
            builder.arg(&maps[0]);
            builder.arg(&maps[1]);
            builder.arg(&bias);
            builder.arg(&params);
            unsafe {
                enqueue_with_physical_observation(
                    observer,
                    &mut builder,
                    launch.config,
                    launch.observation,
                )
            }
            .map_err(|error| error.with_driver_context(format_args!("{}", launch.symbol)))
        }
        (Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(_), PreparedTf32Params::Sm100(params)) => {
            let maps = launch
                .maps
                .ok_or_else(|| "SM100 TF32 launch has no tensor maps".to_string())?
                .maps();
            let mut builder = stream.launch_builder(function);
            builder.arg(&output);
            builder.arg(&maps[0]);
            builder.arg(&maps[1]);
            builder.arg(&bias);
            builder.arg(&params);
            unsafe {
                enqueue_with_physical_observation(
                    observer,
                    &mut builder,
                    launch.config,
                    launch.observation,
                )
            }
            .map_err(|error| error.with_driver_context(format_args!("{}", launch.symbol)))
        }
        (
            Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_)
            | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_),
            PreparedTf32Params::Sm120(_),
        )
        | (Tf32PhysicalRoute::Sm120TmaFmaExactV1(_), PreparedTf32Params::Sm120Fma(_)) => {
            let maps = launch
                .maps
                .ok_or_else(|| "SM120 TF32 launch has no tensor maps".to_string())?
                .maps();
            // The stream-K and exact-F32 kernels take their slab and flag
            // buffers ahead of the tensor maps. A workspace-less launch is
            // legal only when nothing is ever folded: the single-CTA stream-K
            // qualification probe, which deals whole tiles only, and any
            // exact-F32 launch with one split, which keeps the scalar chain.
            let workspace_first = match launch.route {
                Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_) => {
                    if launch.streamk.is_none() && launch.config.grid_dim != (1, 1, 1) {
                        return Err(
                            "SM120 TF32 stream-K launch without a workspace must be one CTA".into(),
                        );
                    }
                    true
                }
                Tf32PhysicalRoute::Sm120TmaFmaExactV1(exact) => {
                    if launch.streamk.is_none() && exact.splits != 1 {
                        return Err("exact-F32 SM120 split launch requires a workspace".into());
                    }
                    true
                }
                _ => false,
            };
            let workspace = launch.streamk.unwrap_or(Tf32StreamKWorkspace {
                partial: 0,
                flags: 0,
            });
            let params = launch.params;
            let mut builder = stream.launch_builder(function);
            builder.arg(&output);
            if workspace_first {
                builder.arg(&workspace.partial);
                builder.arg(&workspace.flags);
            }
            builder.arg(&maps[0]);
            builder.arg(&maps[1]);
            builder.arg(&bias);
            match &params {
                PreparedTf32Params::Sm120(params) => {
                    builder.arg(params);
                }
                PreparedTf32Params::Sm120Fma(params) => {
                    builder.arg(params);
                }
                _ => return Err("prepared TF32 route and parameter ABI disagree".into()),
            }
            unsafe {
                enqueue_with_physical_observation(
                    observer,
                    &mut builder,
                    launch.config,
                    launch.observation,
                )
            }
            .map_err(|error| error.with_driver_context(format_args!("{}", launch.symbol)))
        }
        _ => Err("prepared TF32 route and parameter ABI disagree".into()),
    }
}

/// Enqueues one already-qualified TF32 function during module initialization.
///
/// # Safety
///
/// All raw pointers and optional tensor maps must belong to `stream`'s CUDA
/// context and remain live until the stream has completed the launch.
pub(super) unsafe fn enqueue_tf32_qualification_probe(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    request: F32TriadRequest,
    operands: F32TriadOperands,
    route: Tf32PhysicalRoute,
    maps: Option<&F32PreparedTensorMaps>,
    config: LaunchConfig,
) -> Result<(), String> {
    let spec = tf32_kernel_spec(request.op, route)?;
    let origins = maps.map(F32PreparedTensorMaps::origins).unwrap_or_default();
    let params = tf32_params(request, operands, origins, route)?;
    let mut observer = NoPhysicalObserver;
    unsafe {
        enqueue_tf32_raw(
            stream,
            function,
            Tf32RawLaunch {
                operands,
                route,
                maps,
                params,
                config,
                zero_reduction: request.shape.reduction(request.op) == 0,
                symbol: spec.symbol,
                observation: None,
                streamk: None,
            },
            &mut observer,
        )
    }
}

/// Enqueues a launch whose route, resources, module, and ABI were frozen by
/// [`prepare_f32_triad`]. The scalar closure must use the controlled scalar
/// entry points, which bind observation and CUDA enqueue as one operation.
///
/// # Safety
///
/// Every prepared device allocation and controlled scalar `GpuBuffer` must
/// remain alive and bound to the same CUDA context and stream until execution
/// completes. Identity validation rejects changed live allocations, but it
/// cannot make a dangling raw CUDA pointer safe.
pub(in crate::mamba_ssm::gpu) unsafe fn launch_prepared_f32_triad(
    ctx: &GpuCtx,
    prepared: &PreparedF32TriadLaunch,
    scalar: impl FnOnce(&mut ScalarLaunchControl<'_>) -> Result<(), String>,
) -> Result<(), String> {
    validate_prepared_f32_triad(ctx, prepared)?;
    unsafe { enqueue_validated_prepared_f32_triad(ctx, prepared, scalar) }
}

/// Enqueues a cache entry that was validated immediately before this call or
/// built from the same live resources on the current stream.
///
/// # Safety
///
/// The caller must uphold [`launch_prepared_f32_triad`]'s lifetime contract
/// and must not let the prepared resources change after validation.
pub(in crate::mamba_ssm::gpu) unsafe fn enqueue_validated_prepared_f32_triad(
    ctx: &GpuCtx,
    prepared: &PreparedF32TriadLaunch,
    scalar: impl FnOnce(&mut ScalarLaunchControl<'_>) -> Result<(), String>,
) -> Result<(), String> {
    match &prepared.kind {
        PreparedF32Kind::Scalar(plan) => {
            let mut control = ScalarLaunchControl {
                ctx,
                routes: &prepared.routes,
                plan: *plan,
                operands: prepared.operands,
                next: 0,
            };
            scalar(&mut control)?;
            control.finish()
        }
        PreparedF32Kind::ScalarZero { params, config, .. } => {
            ctx.record_resolved_gemm_route(prepared.routes[0])?;
            let mut observer = NoPhysicalObserver;
            unsafe { enqueue_scalar_zero_f32(ctx, prepared, params, *config, &mut observer, None) }
        }
        PreparedF32Kind::Tf32 { .. } | PreparedF32Kind::Tf32StreamK { .. } => {
            let (route, maps, params, config, streamk) = tf32_prepared_kernel_parts(prepared)?;
            ctx.record_resolved_gemm_route(prepared.routes[0])?;
            let mut observer = NoPhysicalObserver;
            unsafe {
                enqueue_tf32_f32(
                    ctx,
                    prepared,
                    Tf32RawLaunch {
                        operands: prepared.operands,
                        route,
                        maps,
                        params,
                        config: *config,
                        zero_reduction: streamk.is_none()
                            && prepared.request.shape.reduction(prepared.request.op) == 0,
                        symbol: prepared.routes[0].symbol,
                        observation: None,
                        streamk,
                    },
                    &mut observer,
                )
            }
        }
        PreparedF32Kind::Tf32SplitK {
            params,
            plan,
            workspace,
            ..
        } => {
            let mut observer = NoPhysicalObserver;
            unsafe {
                enqueue_tf32_splitk_f32(
                    ctx,
                    prepared,
                    params,
                    *plan,
                    *workspace,
                    &mut observer,
                    None,
                )
            }
        }
    }
}

/// The parts of a prepared tiled or stream-K TF32 launch that its raw launch
/// package is built from: the route, its tensor maps, the kernel parameters,
/// the frozen launch configuration, and the stream-K workspace if any.
type Tf32PreparedKernelParts<'a> = (
    Tf32PhysicalRoute,
    Option<&'a F32PreparedTensorMaps>,
    PreparedTf32Params,
    &'a cudarc::driver::LaunchConfig,
    Option<Tf32StreamKWorkspace>,
);

fn tf32_prepared_kernel_parts(
    prepared: &PreparedF32TriadLaunch,
) -> Result<Tf32PreparedKernelParts<'_>, String> {
    match &prepared.kind {
        PreparedF32Kind::Tf32 {
            route,
            maps,
            params,
            config,
        } => Ok((*route, maps.as_ref(), *params, config, None)),
        PreparedF32Kind::Tf32StreamK {
            route,
            maps,
            params,
            config,
            workspace,
            ..
        } => Ok((*route, Some(maps), *params, config, Some(*workspace))),
        _ => Err("prepared launch is not a TF32 kernel launch".into()),
    }
}

unsafe fn enqueue_validated_prepared_f32_triad_observed<O, Scalar>(
    ctx: &GpuCtx,
    prepared: &PreparedF32TriadLaunch,
    observer: &mut O,
    logical_dtype: PolicyDtype,
    scalar: Scalar,
) -> Result<(), String>
where
    O: PhysicalLaunchObserver,
    Scalar: FnOnce(&mut PhysicalScalarLaunchControl<'_, O>) -> Result<(), String>,
{
    let physical_resources_digest = prepared.resources.physical_digest();
    match &prepared.kind {
        PreparedF32Kind::Scalar(plan) => {
            let base = ScalarLaunchControl {
                ctx,
                routes: &prepared.routes,
                plan: *plan,
                operands: prepared.operands,
                next: 0,
            };
            let mut control = PhysicalScalarLaunchControl {
                base,
                logical_dtype,
                physical_resources_digest,
                observer,
            };
            scalar(&mut control)?;
            control.finish()
        }
        PreparedF32Kind::ScalarZero { params, config, .. } => {
            let route = prepared.routes[0];
            ctx.record_resolved_gemm_route(route)?;
            unsafe {
                enqueue_scalar_zero_f32(
                    ctx,
                    prepared,
                    params,
                    *config,
                    observer,
                    Some(PhysicalLaunchObservation::gemm(
                        logical_dtype,
                        Some(physical_resources_digest),
                        route,
                    )),
                )
            }
        }
        PreparedF32Kind::Tf32 { .. } | PreparedF32Kind::Tf32StreamK { .. } => {
            let (route, maps, params, config, streamk) = tf32_prepared_kernel_parts(prepared)?;
            let resolved = prepared.routes[0];
            let physical = physical_prepared_f32_route(prepared, resolved);
            ctx.record_resolved_gemm_route(resolved)?;
            unsafe {
                enqueue_tf32_f32(
                    ctx,
                    prepared,
                    Tf32RawLaunch {
                        operands: prepared.operands,
                        route,
                        maps,
                        params,
                        config: *config,
                        zero_reduction: streamk.is_none()
                            && prepared.request.shape.reduction(prepared.request.op) == 0,
                        symbol: resolved.symbol,
                        observation: Some(PhysicalLaunchObservation::gemm(
                            logical_dtype,
                            None,
                            physical,
                        )),
                        streamk,
                    },
                    observer,
                )
            }
        }
        PreparedF32Kind::Tf32SplitK {
            params,
            plan,
            workspace,
            ..
        } => unsafe {
            enqueue_tf32_splitk_f32(
                ctx,
                prepared,
                params,
                *plan,
                *workspace,
                observer,
                Some(logical_dtype),
            )
        },
    }
}

fn require_sm90a_tensor_map_access(stream: &Arc<cudarc::driver::CudaStream>) -> Result<(), String> {
    let supported = stream
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
        )
        .map_err(|error| format!("query CUDA tensor-map support: {error:?}"))?;
    if supported == 0 {
        return Err("SM90a WGMMA requires CUDA tensor-map access support".into());
    }
    Ok(())
}

fn specialized_device_identity(
    compute_capability: (u32, u32),
    multiprocessor_count: u32,
    target: crate::mamba_ssm::gpu::kernel_identity::CudaTarget,
    driver: crate::mamba_ssm::gpu::kernel_identity::DriverIdentity,
) -> crate::mamba_ssm::gpu::kernel_identity::DeviceIdentity {
    crate::mamba_ssm::gpu::kernel_identity::DeviceIdentity {
        compute_capability,
        multiprocessor_count,
        target,
        driver,
    }
}

fn sm90a_map_binding(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
) -> Result<Sm90aMapBinding, String> {
    require_sm90a_tensor_map_access(stream)?;
    let allocation_domain = validated_allocation_domain(stream, kernels, "SM90a")?;
    let compiler = kernels
        .sm90a_compiler_identity()
        .filter(|compiler| compiler.target.as_str() == "sm_90a")
        .ok_or_else(|| "exact-sm_90a compiler identity is unavailable".to_string())?;
    let artifact = kernels
        .artifact_set_identity()
        .specialized
        .filter(|artifact| {
            artifact.module_kind == crate::mamba_ssm::gpu::kernel_identity::ModuleKind::TriadSm90a
        })
        .ok_or_else(|| "exact-sm_90a artifact identity is unavailable".to_string())?;
    Ok(Sm90aMapBinding {
        allocation_domain,
        artifact,
        compiler,
        device: sm90a_device_identity(stream, kernels)?,
    })
}

fn sm90a_device_identity(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
) -> Result<crate::mamba_ssm::gpu::kernel_identity::DeviceIdentity, String> {
    let (major, minor) = stream
        .context()
        .compute_capability()
        .map_err(|error| format!("query CUDA compute capability: {error:?}"))?;
    Ok(specialized_device_identity(
        (
            u32::try_from(major).map_err(|_| format!("negative CUDA CC major {major}"))?,
            u32::try_from(minor).map_err(|_| format!("negative CUDA CC minor {minor}"))?,
        ),
        kernels.multiprocessor_count(),
        crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new("sm_90a")?,
        crate::mamba_ssm::gpu::kernel_identity::query_driver_identity()?,
    ))
}

pub fn prepare_sm90a_tensor_maps(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    request: Sm90aMapRequest,
) -> Result<Sm90aPreparedTensorMaps, String> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| format!("bind CUDA context for tensor-map encoding: {error:?}"))?;
    let binding = sm90a_map_binding(stream, kernels)?;
    if stream.context().compute_capability().ok() != Some((9, 0)) || !kernels.has_sm90a_wgmma() {
        return Err("SM90a tensor maps require a loaded exact-sm_90a module".into());
    }
    let keys = sm90a_tensor_map_keys(request)?;
    let allocations = sm90a_allocation_identities(keys, binding.allocation_domain)?;
    let capturing = stream
        .capture_status()
        .map_err(|error| format!("query CUDA capture status: {error:?}"))?
        != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    kernels.prepare_sm90a_tensor_maps(request, keys, allocations, capturing, binding)
}

fn sm90a_forced_identity(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    route: Sm90aForcedRoute,
    maps: &Sm90aPreparedTensorMaps,
    operands: Sm90aLaunchOperands,
) -> Result<Sm90aRouteIdentity, String> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| format!("bind CUDA context for SM90a launch: {error:?}"))?;
    route.shape.validate(route.op)?;
    let binding = sm90a_map_binding(stream, kernels)?;
    if !maps.matches_binding(binding) {
        return Err(
            "SM90a prepared tensor maps belong to a different CUDA context or module".into(),
        );
    }
    maps.validate_live_allocations()?;
    if maps.request.op != route.op
        || maps.request.dtype != route.dtype
        || maps.request.shape != route.shape
    {
        return Err("SM90a prepared tensor maps do not match the forced route".into());
    }
    let resolved = resolve_sm90a_forced(
        (
            i32::try_from(binding.device.compute_capability.0)
                .map_err(|_| "CUDA CC major exceeds i32::MAX".to_string())?,
            i32::try_from(binding.device.compute_capability.1)
                .map_err(|_| "CUDA CC minor exceeds i32::MAX".to_string())?,
        ),
        kernels.has_sm90a_wgmma(),
        route.op,
        route.dtype,
        route.schedule,
        route.shape,
    )?;
    if resolved != Some(route) {
        return Err("SM90a forced route is unavailable; use the resolved baseline".into());
    }
    if operands.output_ptr == 0 {
        return Err("SM90a output pointer must be non-null".into());
    }
    let output_alignment = if route.op == Sm90aOp::Tn { 4 } else { 2 };
    if !operands.output_ptr.is_multiple_of(output_alignment) {
        return Err(format!(
            "SM90a output pointer must be {output_alignment}-byte aligned"
        ));
    }
    if operands.bias_ptr != 0 && !operands.bias_ptr.is_multiple_of(4) {
        return Err("SM90a bias pointer must be 4-byte aligned".into());
    }
    match route.op {
        Sm90aOp::Nn => {}
        Sm90aOp::Tn if operands.bias_ptr != 0 || operands.beta != 1.0 => {
            return Err("SM90a TN requires no bias and beta == 1.0".into());
        }
        Sm90aOp::Nt if operands.bias_ptr != 0 || operands.beta != 0.0 => {
            return Err("SM90a NT requires no bias and beta == 0.0".into());
        }
        _ => {}
    }
    let tensor_maps_digest = maps.identity_digest();
    Ok(Sm90aRouteIdentity {
        numeric_contract: Sm90aNumericContract::WgmmaV1,
        op: route.op,
        dtype: route.dtype,
        schedule: route.schedule,
        shape: route.shape,
        tile: SM90A_TILE,
        stages: SM90A_STAGES,
        cluster: (1, 1, 1),
        symbol: route.symbol(),
        module_kind: crate::mamba_ssm::gpu::kernel_identity::ModuleKind::TriadSm90a,
        exact_target: "sm_90a",
        artifact: binding.artifact,
        compiler: binding.compiler,
        device: binding.device,
        tensor_maps_digest,
        resources_digest: sm90a_resources_digest(
            route,
            operands,
            binding.allocation_domain,
            tensor_maps_digest,
        )?,
        tuning_revision: 0,
    })
}

pub fn validate_sm90a_graph_replay(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    route: Sm90aForcedRoute,
    maps: &Sm90aPreparedTensorMaps,
    operands: Sm90aLaunchOperands,
    captured: Sm90aRouteIdentity,
) -> Result<(), String> {
    let live = sm90a_forced_identity(stream, kernels, route, maps, operands)?;
    captured.ensure_current(live, "SM90a graph replay")
}

pub fn launch_sm90a_wgmma_forced(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    route: Sm90aForcedRoute,
    maps: &Sm90aPreparedTensorMaps,
    operands: Sm90aLaunchOperands,
) -> Result<Sm90aRouteIdentity, String> {
    let identity = sm90a_forced_identity(stream, kernels, route, maps, operands)?;
    let function = kernels
        .sm90a_function(route.symbol())
        .ok_or_else(|| format!("SM90a kernel {} is unavailable", route.symbol()))?;
    let config = sm90a_launch_config(route)?;
    let m = checked_i32(route.shape.m, "M")?;
    let k = checked_i32(route.shape.k, "K")?;
    let n = checked_i32(route.shape.n, "N")?;
    let ldc = checked_i32(route.shape.ldc, "ldc")?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&operands.output_ptr);
    builder.arg(&maps.a);
    builder.arg(&maps.b);
    builder.arg(&operands.bias_ptr);
    builder.arg(&operands.alpha);
    builder.arg(&operands.beta);
    builder.arg(&m);
    builder.arg(&k);
    builder.arg(&n);
    builder.arg(&ldc);
    unsafe { builder.launch(config) }
        .map(|_| identity)
        .map_err(|error| format!("launch {}: {error:?}", route.symbol()))
}

fn require_sm100_tensor_map_access(stream: &Arc<cudarc::driver::CudaStream>) -> Result<(), String> {
    let supported = stream
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
        )
        .map_err(|error| format!("query CUDA tensor-map support: {error:?}"))?;
    if supported == 0 {
        return Err("SM100 TCGEN requires CUDA tensor-map access support".into());
    }
    Ok(())
}

fn sm100_map_binding(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
) -> Result<Sm100MapBinding, String> {
    require_sm100_tensor_map_access(stream)?;
    let allocation_domain = validated_allocation_domain(stream, kernels, "SM100")?;
    let target = kernels
        .sm100_target_candidate()
        .ok_or_else(|| "SM100 target identity is unavailable".to_string())?;
    let compiler = kernels
        .sm100_compiler_identity()
        .filter(|compiler| compiler.target.as_str() == target.nvrtc_arch)
        .ok_or_else(|| "SM100 compiler identity is unavailable".to_string())?;
    let artifact = kernels
        .artifact_set_identity()
        .specialized
        .filter(|artifact| {
            artifact.module_kind == crate::mamba_ssm::gpu::kernel_identity::ModuleKind::TriadSm100
        })
        .ok_or_else(|| "SM100 artifact identity is unavailable".to_string())?;
    let (major, minor) = stream
        .context()
        .compute_capability()
        .map_err(|error| format!("query CUDA compute capability: {error:?}"))?;
    if (major, minor) != target.device_cc {
        return Err("SM100 module target does not match the CUDA device minor".into());
    }
    let compute_capability = (
        u32::try_from(major).map_err(|_| format!("negative CUDA CC major {major}"))?,
        u32::try_from(minor).map_err(|_| format!("negative CUDA CC minor {minor}"))?,
    );
    let device = specialized_device_identity(
        compute_capability,
        kernels.multiprocessor_count(),
        crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new(
            crate::mamba_ssm::gpu::device::GpuDevice::resolve_nvrtc_target(compute_capability)?,
        )?,
        crate::mamba_ssm::gpu::kernel_identity::query_driver_identity()?,
    );
    Ok(Sm100MapBinding {
        allocation_domain,
        artifact,
        compiler,
        device,
        target,
    })
}

pub fn prepare_sm100_tensor_maps(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    request: Sm100MapRequest,
) -> Result<Sm100PreparedTensorMaps, String> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| format!("bind CUDA context for SM100 tensor maps: {error:?}"))?;
    if stream
        .capture_status()
        .map_err(|error| format!("query CUDA capture status: {error:?}"))?
        != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        return Err("SM100 tensor maps must be prepared before graph capture".into());
    }
    let binding = sm100_map_binding(stream, kernels)?;
    if !kernels.has_sm100_tcgen() {
        return Err("SM100 tensor maps require a complete specialized module".into());
    }
    let plan = sm100_tensor_map_plan(request, binding.allocation_domain)?;
    kernels.prepare_sm100_tensor_maps(
        request,
        plan.keys,
        plan.allocations,
        plan.origins,
        false,
        binding,
    )
}

pub fn prepare_sm100_tcgen_forced(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    route: Sm100ForcedRoute,
    maps: &Sm100PreparedTensorMaps,
    operands: Sm100LaunchOperands,
) -> Result<Sm100PreparedLaunch, String> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| format!("bind CUDA context for SM100 preparation: {error:?}"))?;
    if stream
        .capture_status()
        .map_err(|error| format!("query CUDA capture status: {error:?}"))?
        != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        return Err("SM100 launch must be prepared before graph capture".into());
    }
    route.shape.validate(route.op)?;
    let binding = sm100_map_binding(stream, kernels)?;
    if !maps.matches_binding(binding) {
        return Err(
            "SM100 prepared tensor maps belong to a different CUDA context or module".into(),
        );
    }
    maps.validate_live_allocations()?;
    if maps.request.op != route.op
        || maps.request.dtype != route.dtype
        || maps.request.tile != route.physical.tile
        || maps.request.shape != route.shape
    {
        return Err("SM100 tensor maps do not match the forced physical route".into());
    }
    if resolve_sm100_forced(binding.target.device_cc, Some(binding.target), route)? != Some(route) {
        return Err("SM100 forced route is unavailable; use the resolved baseline".into());
    }
    validate_sm100_operands(route, operands)?;
    let spec = route.kernel_spec()?;
    let tensor_maps_digest = maps.identity_digest();
    let resources = Sm100LaunchResourceSnapshot::query(route, operands, binding.allocation_domain)?;
    let params = Sm100KernelParams {
        a_x: maps.origins.a_x,
        a_y: maps.origins.a_y,
        b_x: maps.origins.b_x,
        b_y: maps.origins.b_y,
        alpha: operands.alpha,
        beta: operands.beta,
        m: checked_i32(route.shape.m, "M")?,
        k: checked_i32(route.shape.k, "K")?,
        n: checked_i32(route.shape.n, "N")?,
        ldc: checked_i32(route.shape.ldc, "ldc")?,
    };
    let identity = Sm100RouteIdentity {
        numeric_contract: Sm100NumericContract::Tcgen05F32V1,
        op: route.op,
        dtype: route.dtype,
        physical: route.physical,
        shape: route.shape,
        symbol: spec.symbol,
        module_kind: crate::mamba_ssm::gpu::kernel_identity::ModuleKind::TriadSm100,
        target: binding.target,
        artifact: binding.artifact,
        compiler: binding.compiler,
        device: binding.device,
        tensor_map_revision: SM100_TENSOR_MAP_REVISION,
        tensor_maps_digest,
        resources_digest: resources.digest(route, operands, tensor_maps_digest),
        tuning_revision: SM100_TUNING_REVISION,
    };
    Ok(Sm100PreparedLaunch {
        route,
        maps: *maps,
        operands,
        params: params.into_words(),
        identity,
        resources,
    })
}

fn validate_sm100_operands(
    route: Sm100ForcedRoute,
    operands: Sm100LaunchOperands,
) -> Result<(), String> {
    if operands.output_ptr == 0 {
        return Err("SM100 output pointer must be non-null".into());
    }
    let output_alignment = if route.op == Sm100Op::Tn { 4 } else { 2 };
    if !operands.output_ptr.is_multiple_of(output_alignment) {
        return Err(format!(
            "SM100 output pointer must be {output_alignment}-byte aligned"
        ));
    }
    if operands.bias_ptr != 0 && !operands.bias_ptr.is_multiple_of(4) {
        return Err("SM100 bias pointer must be 4-byte aligned".into());
    }
    match route.op {
        Sm100Op::Nn if operands.bias_ptr != 0 && operands.alpha != 1.0 => {
            Err("SM100 NN bias seeding requires alpha == 1.0".into())
        }
        Sm100Op::Tn if operands.bias_ptr != 0 || operands.beta != 1.0 => {
            Err("SM100 TN requires no bias and beta == 1.0".into())
        }
        Sm100Op::Nt if operands.bias_ptr != 0 || operands.beta != 0.0 => {
            Err("SM100 NT requires no bias and beta == 0.0".into())
        }
        _ => Ok(()),
    }
}

fn validate_sm100_prepared_binding(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    prepared: &Sm100PreparedLaunch,
) -> Result<(), String> {
    if AllocationDomain::from_context(stream.context())? != prepared.maps.binding.allocation_domain
        || kernels.allocation_domain() != prepared.maps.binding.allocation_domain
        || kernels.sm100_compiler_identity() != Some(prepared.identity.compiler)
        || kernels.artifact_set_identity().specialized != Some(prepared.identity.artifact)
        || kernels.sm100_target_candidate() != Some(prepared.identity.target)
        || prepared.maps.binding.artifact != prepared.identity.artifact
        || prepared.maps.binding.compiler != prepared.identity.compiler
        || prepared.maps.binding.device != prepared.identity.device
        || prepared.maps.binding.target != prepared.identity.target
    {
        return Err("SM100 prepared launch no longer matches its module context".into());
    }
    Ok(())
}

pub fn validate_sm100_graph_replay(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    prepared: &Sm100PreparedLaunch,
) -> Result<(), String> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| format!("bind CUDA context for SM100 replay guard: {error:?}"))?;
    validate_sm100_prepared_binding(stream, kernels, prepared)?;
    prepared.maps.validate_live_allocations()?;
    let live = Sm100LaunchResourceSnapshot::query(
        prepared.route,
        prepared.operands,
        prepared.maps.binding.allocation_domain,
    )?;
    if live != prepared.resources {
        return Err("SM100 graph replay allocation identity changed since capture".into());
    }
    let digest = live.digest(
        prepared.route,
        prepared.operands,
        prepared.maps.identity_digest(),
    );
    if digest != prepared.identity.resources_digest {
        return Err("SM100 graph replay resource identity changed since capture".into());
    }
    Ok(())
}

pub fn launch_sm100_tcgen_prepared(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    prepared: &Sm100PreparedLaunch,
) -> Result<Sm100RouteIdentity, String> {
    validate_sm100_prepared_binding(stream, kernels, prepared)?;
    let spec = prepared.route.kernel_spec()?;
    if spec.symbol != prepared.identity.symbol {
        return Err("SM100 prepared symbol no longer matches its physical route".into());
    }
    let function = kernels
        .sm100_function(spec.symbol)
        .ok_or_else(|| format!("SM100 kernel {} is unavailable", spec.symbol))?;
    let (rows, columns) = match prepared.route.op {
        Sm100Op::Nn => (prepared.route.shape.m, prepared.route.shape.n),
        Sm100Op::Tn => (prepared.route.shape.k, prepared.route.shape.n),
        Sm100Op::Nt => (prepared.route.shape.m, prepared.route.shape.k),
    };
    let rows = checked_u32(rows, "SM100 output rows")?;
    let columns = checked_u32(columns, "SM100 output columns")?;
    let grid = checked_grid_product(
        rows.div_ceil(prepared.route.physical.tile.output_rows()),
        columns.div_ceil(prepared.route.physical.tile.output_columns()),
        1,
    )?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (spec.threads, 1, 1),
        shared_mem_bytes: spec.dynamic_shared_bytes,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&prepared.operands.output_ptr);
    builder.arg(&prepared.maps.a);
    builder.arg(&prepared.maps.b);
    builder.arg(&prepared.operands.bias_ptr);
    let params = Sm100KernelParams::from_words(prepared.params);
    builder.arg(&params);
    unsafe { builder.launch(config) }
        .map(|_| prepared.identity)
        .map_err(|error| format!("launch {}: {error:?}", spec.symbol))
}

/// Encodes and binds SM120 tensor maps before graph capture begins.
///
/// This is a low-level qualification/census API. Ordinary typed GEMM calls
/// populate and reuse the prepared-launch cache automatically.
pub fn prepare_sm120_tensor_maps(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    request: Sm120MapRequest,
) -> Result<Sm120PreparedTensorMaps, String> {
    if stream
        .capture_status()
        .map_err(|error| format!("query CUDA capture status: {error:?}"))?
        != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        return Err("SM120 tensor maps must be prepared before graph capture".into());
    }
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| format!("bind CUDA context for SM120 tensor maps: {error:?}"))?;
    let binding = sm120_map_binding(stream, kernels)?;
    if !kernels.has_sm120_tma_mma16() {
        return Err("SM120 tensor maps require a complete specialized module".into());
    }
    let plan = sm120_tensor_map_plan(request, binding.allocation_domain)?;
    kernels.prepare_sm120_tensor_maps(
        request,
        plan.keys,
        plan.allocations,
        plan.origins,
        false,
        binding,
    )
}

fn sm120_map_binding(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
) -> Result<Sm120MapBinding, String> {
    let allocation_domain = validated_allocation_domain(stream, kernels, "SM120")?;
    let tensor_map_access = stream
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
        )
        .map_err(|error| format!("query SM120 tensor-map support: {error:?}"))?
        != 0;
    let optin_shared = stream
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .map_err(|error| format!("query SM120 opt-in shared memory: {error:?}"))?;
    let (major, minor) = stream
        .context()
        .compute_capability()
        .map_err(|error| format!("query SM120 compute capability: {error:?}"))?;
    let device_cc = (major, minor);
    let compiler = kernels
        .sm120_compiler_identity()
        .ok_or_else(|| "SM120 compiler identity is unavailable".to_string())?;
    let target = kernels
        .sm120_target_candidate()
        .ok_or_else(|| "SM120 generic target identity is unavailable".to_string())?;
    if target.device_cc != device_cc {
        return Err("SM120 module target does not match the CUDA device minor".into());
    }
    if compiler.target.as_str() != target.nvrtc_arch {
        return Err("SM120 compiler target does not match the accepted transaction".into());
    }
    let artifact = kernels
        .artifact_set_identity()
        .specialized
        .filter(|artifact| {
            artifact.module_kind == crate::mamba_ssm::gpu::kernel_identity::ModuleKind::TriadSm120
        })
        .ok_or_else(|| "SM120 artifact identity is unavailable".to_string())?;
    let compute_capability = (
        u32::try_from(major).map_err(|_| format!("negative CUDA CC major {major}"))?,
        u32::try_from(minor).map_err(|_| format!("negative CUDA CC minor {minor}"))?,
    );
    let accepted_target =
        crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new(target.nvrtc_arch)?;
    let device_caps = crate::mamba_ssm::gpu::kernel_identity::DeviceCaps {
        compute_capability,
        nvrtc_version: compiler.nvrtc_version,
        accepted_target: Some(accepted_target),
        optin_shared_bytes: u32::try_from(optin_shared)
            .map_err(|_| format!("negative SM120 opt-in shared memory {optin_shared}"))?,
        tensor_map_access,
    };
    if kernels.sm120_device_caps() != Some(device_caps) {
        return Err("SM120 device capabilities changed since module qualification".into());
    }
    let device = specialized_device_identity(
        compute_capability,
        kernels.multiprocessor_count(),
        crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new(target.ptx_target)?,
        crate::mamba_ssm::gpu::kernel_identity::query_driver_identity()?,
    );
    Ok(Sm120MapBinding {
        allocation_domain,
        artifact,
        compiler,
        device,
        device_caps,
        target,
    })
}

/// Prepares one already-resolved forced SM120 route for launch or capture.
///
/// The route, maps, allocations, module, resources, and scalar arguments are
/// sealed into the returned identity. Preparation is rejected during capture.
pub fn prepare_sm120_tma_forced(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    route: Sm120ForcedRoute,
    maps: &Sm120PreparedTensorMaps,
    operands: Sm120LaunchOperands,
) -> Result<Sm120PreparedLaunch, String> {
    if stream
        .capture_status()
        .map_err(|error| format!("query CUDA capture status: {error:?}"))?
        != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        return Err("SM120 launch must be prepared before graph capture".into());
    }
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| format!("bind CUDA context for SM120 preparation: {error:?}"))?;
    route.shape.validate(route.op)?;
    let binding = sm120_map_binding(stream, kernels)?;
    if !maps.matches_binding(binding) {
        return Err(
            "SM120 prepared tensor maps belong to a different CUDA context or module".into(),
        );
    }
    maps.validate_live_allocations()?;
    if maps.request.op != route.op
        || maps.request.dtype != route.dtype
        || maps.request.tile != route.physical.tile
        || maps.request.bk != route.physical.bk
        || maps.request.shape != route.shape
    {
        return Err("SM120 tensor maps do not match the forced physical route".into());
    }
    if resolve_sm120_forced(binding.device_caps, Some(binding.target), route)? != Some(route) {
        return Err("SM120 forced route is unavailable; use the resolved baseline".into());
    }
    validate_sm120_operands(route, operands)?;
    let spec = route.kernel_spec()?;
    let kernel_resources = kernels
        .sm120_kernel_resources(spec.symbol)
        .ok_or_else(|| format!("SM120 resource census for {} is unavailable", spec.symbol))?;
    let tensor_maps_digest = maps.identity_digest();
    let resources = Sm120LaunchResourceSnapshot::query(route, operands, binding.allocation_domain)?;
    let params = Sm120KernelParams {
        a_x: maps.origins.a_x,
        a_y: maps.origins.a_y,
        b_x: maps.origins.b_x,
        b_y: maps.origins.b_y,
        alpha: operands.alpha,
        beta: operands.beta,
        m: checked_i32(route.shape.m, "M")?,
        k: checked_i32(route.shape.k, "K")?,
        n: checked_i32(route.shape.n, "N")?,
        ldc: checked_i32(route.shape.ldc, "ldc")?,
    };
    let identity = Sm120RouteIdentity {
        numeric_contract: Sm120NumericContract::for_schedule(route.physical.schedule),
        op: route.op,
        dtype: route.dtype,
        physical: route.physical,
        shape: route.shape,
        symbol: spec.symbol,
        module_kind: crate::mamba_ssm::gpu::kernel_identity::ModuleKind::TriadSm120,
        target: binding.target,
        artifact: binding.artifact,
        compiler: binding.compiler,
        device: binding.device,
        device_caps: binding.device_caps,
        tensor_map_revision: SM120_TENSOR_MAP_REVISION,
        tensor_maps_digest,
        resources_digest: resources.digest(route, operands, tensor_maps_digest),
        tuning_revision: SM120_TUNING_REVISION,
        schedule_revision: SM120_SCHEDULE_REVISION,
    };
    let resolved_launch_set = build_resolved_gemm_launch_set(&[identity.resolved_route()?])?;
    Ok(Sm120PreparedLaunch {
        stream_handle: stream.cu_stream() as usize,
        route,
        maps: *maps,
        operands,
        params: params.into_words(),
        identity,
        resolved_launch_set,
        resources,
        kernel_resources,
    })
}

fn validate_sm120_operands(
    route: Sm120ForcedRoute,
    operands: Sm120LaunchOperands,
) -> Result<(), String> {
    if operands.output_ptr == 0 {
        return Err("SM120 output pointer must be non-null".into());
    }
    let output_alignment = if route.op == Sm120Op::Tn { 4 } else { 2 };
    if !operands.output_ptr.is_multiple_of(output_alignment) {
        return Err(format!(
            "SM120 output pointer must be {output_alignment}-byte aligned"
        ));
    }
    if operands.bias_ptr != 0 && !operands.bias_ptr.is_multiple_of(4) {
        return Err("SM120 bias pointer must be 4-byte aligned".into());
    }
    match route.op {
        Sm120Op::Nn if operands.bias_ptr != 0 && operands.alpha != 1.0 => {
            Err("SM120 NN bias seeding requires alpha == 1.0".into())
        }
        Sm120Op::Tn if operands.bias_ptr != 0 || operands.beta != 1.0 => {
            Err("SM120 TN requires no bias and beta == 1.0".into())
        }
        Sm120Op::Nt if operands.bias_ptr != 0 || operands.beta != 0.0 => {
            Err("SM120 NT requires no bias and beta == 0.0".into())
        }
        _ => Ok(()),
    }
}

fn validate_sm120_prepared_binding(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    prepared: &Sm120PreparedLaunch,
) -> Result<(), String> {
    if stream.cu_stream() as usize != prepared.stream_handle {
        return Err("SM120 prepared launch belongs to a different CUDA stream".into());
    }
    if AllocationDomain::from_context(stream.context())? != prepared.maps.binding.allocation_domain
        || kernels.allocation_domain() != prepared.maps.binding.allocation_domain
        || kernels.sm120_compiler_identity() != Some(prepared.identity.compiler)
        || kernels.artifact_set_identity().specialized != Some(prepared.identity.artifact)
        || prepared.maps.binding.artifact != prepared.identity.artifact
        || prepared.maps.binding.compiler != prepared.identity.compiler
        || prepared.maps.binding.device != prepared.identity.device
        || prepared.maps.binding.device_caps != prepared.identity.device_caps
        || prepared.maps.binding.target != prepared.identity.target
    {
        return Err("SM120 prepared launch no longer matches its module context".into());
    }
    Ok(())
}

/// Revalidates every prepared SM120 binding before a captured graph is replayed.
///
/// Pointer reuse, allocation-generation changes, module changes, and stale
/// route revisions fail closed and require eager warmup plus recapture.
pub fn validate_sm120_graph_replay(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    prepared: &Sm120PreparedLaunch,
) -> Result<(), String> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| format!("bind CUDA context for SM120 replay guard: {error:?}"))?;
    validate_sm120_prepared_binding(stream, kernels, prepared)?;
    let binding = sm120_map_binding(stream, kernels)?;
    if binding != prepared.maps.binding {
        return Err("SM120 graph replay device or module identity changed since capture".into());
    }
    prepared.maps.validate_live_allocations()?;
    let live = Sm120LaunchResourceSnapshot::query(
        prepared.route,
        prepared.operands,
        prepared.maps.binding.allocation_domain,
    )?;
    if live != prepared.resources {
        return Err("SM120 graph replay allocation identity changed since capture".into());
    }
    let digest = live.digest(
        prepared.route,
        prepared.operands,
        prepared.maps.identity_digest(),
    );
    if digest != prepared.identity.resources_digest {
        return Err("SM120 graph replay resource identity changed since capture".into());
    }
    let live_identity = Sm120RouteIdentity {
        artifact: binding.artifact,
        compiler: binding.compiler,
        device: binding.device,
        device_caps: binding.device_caps,
        tensor_maps_digest: prepared.maps.identity_digest(),
        resources_digest: digest,
        ..prepared.identity
    };
    let live_launch_set = build_resolved_gemm_launch_set(&[live_identity.resolved_route()?])?;
    prepared
        .resolved_launch_set()
        .ensure_current(live_launch_set, "SM120 graph replay")?;
    Ok(())
}

/// Launches a previously prepared forced SM120 route without changing its plan.
///
/// Normal typed calls should use the production GEMM entry points; this direct
/// surface is retained for physical qualification and kernel census tests.
pub fn launch_sm120_tma_prepared(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    prepared: &Sm120PreparedLaunch,
) -> Result<Sm120RouteIdentity, String> {
    let mut observer = NoPhysicalObserver;
    unsafe { enqueue_sm120_tma_prepared_observed(stream, kernels, prepared, &mut observer, None) }
        .map(|()| prepared.identity)
}

unsafe fn enqueue_sm120_tma_prepared_observed<O: PhysicalLaunchObserver>(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    prepared: &Sm120PreparedLaunch,
    observer: &mut O,
    observation: Option<PhysicalLaunchObservation>,
) -> Result<(), String> {
    validate_sm120_prepared_binding(stream, kernels, prepared)?;
    let spec = prepared.route.kernel_spec()?;
    if spec.symbol != prepared.identity.symbol {
        return Err("SM120 prepared symbol no longer matches its physical route".into());
    }
    let function = kernels
        .sm120_function(spec.symbol)
        .ok_or_else(|| format!("SM120 kernel {} is unavailable", spec.symbol))?;
    let (rows, columns) = match prepared.route.op {
        Sm120Op::Nn => (prepared.route.shape.m, prepared.route.shape.n),
        Sm120Op::Tn => (prepared.route.shape.k, prepared.route.shape.n),
        Sm120Op::Nt => (prepared.route.shape.m, prepared.route.shape.k),
    };
    let rows = checked_u32(rows, "SM120 output rows")?;
    let columns = checked_u32(columns, "SM120 output columns")?;
    let grid = sm120_launch_grid(kernels, prepared.route, spec, rows, columns)?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (spec.threads, 1, 1),
        shared_mem_bytes: spec.dynamic_shared_bytes,
    };
    let workspace = sm120_streamk_workspace(stream, kernels, prepared.route, spec, grid)?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&prepared.operands.output_ptr);
    if let Some(workspace) = &workspace {
        builder.arg(&workspace.partial);
        builder.arg(&workspace.flags);
    }
    builder.arg(&prepared.maps.a);
    builder.arg(&prepared.maps.b);
    builder.arg(&prepared.operands.bias_ptr);
    let params = Sm120KernelParams::from_words(prepared.params);
    builder.arg(&params);
    unsafe { enqueue_with_physical_observation(observer, &mut builder, config, observation) }
        .map_err(|error| error.with_driver_context(format_args!("launch {}", spec.symbol)))
}

/// The grid of an SM120 launch: one CTA per output tile, or the persistent
/// grid of one CTA per multiprocessor under the stream-K schedule.
fn sm120_launch_grid(
    kernels: &GpuKernels,
    route: Sm120ForcedRoute,
    spec: &Sm120KernelSpec,
    rows: u32,
    columns: u32,
) -> Result<u32, String> {
    match route.physical.schedule {
        Sm120Schedule::Tiled => checked_grid_product(
            rows.div_ceil(route.physical.tile.output_rows()),
            columns.div_ceil(route.physical.tile.output_columns()),
            1,
        ),
        Sm120Schedule::StreamK => {
            let grid = kernels.multiprocessor_count();
            if grid == 0 {
                return Err("SM120 stream-K launch requires at least one multiprocessor".into());
            }
            let reduction = match route.op {
                Sm120Op::Nn => route.shape.k,
                Sm120Op::Tn => route.shape.m,
                Sm120Op::Nt => route.shape.n,
            };
            let k_tiles = reduction
                .div_ceil(route.physical.bk.elements() as usize)
                .max(1);
            let tiles = (rows.div_ceil(route.physical.tile.output_rows()) as usize)
                .checked_mul(columns.div_ceil(route.physical.tile.output_columns()) as usize)
                .ok_or_else(|| "SM120 stream-K tile count overflows usize".to_string())?;
            let units = tiles
                .checked_mul(k_tiles)
                .ok_or_else(|| "SM120 stream-K unit count overflows usize".to_string())?;
            if units > i32::MAX as usize {
                return Err("SM120 stream-K unit count exceeds the kernel's 32-bit range".into());
            }
            let _ = spec;
            Ok(grid)
        }
    }
}

/// The partial-slab and flag workspace a stream-K launch reads and writes;
/// `None` under the tiled schedule.
struct Sm120StreamKWorkspace {
    partial: CUptr,
    flags: CUptr,
}

fn sm120_streamk_workspace(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    route: Sm120ForcedRoute,
    spec: &Sm120KernelSpec,
    grid: u32,
) -> Result<Option<Sm120StreamKWorkspace>, String> {
    use cudarc::driver::DevicePtr;
    if route.physical.schedule != Sm120Schedule::StreamK {
        return Ok(None);
    }
    // One slab per (CTA, slot): every thread holds MAtoms x 16 accumulators.
    let m_atoms = if route.physical.wide_m_warp() { 4 } else { 2 };
    let slab_floats = (spec.threads as usize) * m_atoms * 16;
    let partial_floats = (grid as usize)
        .checked_mul(SM120_TF32_STREAMK_SLOTS_PER_CTA)
        .and_then(|slots| slots.checked_mul(slab_floats))
        .ok_or_else(|| "SM120 stream-K slab extent overflows usize".to_string())?;
    if partial_floats > SPLITK_SCRATCH_CAP {
        return Err("SM120 stream-K slabs exceed the fixed workspace".into());
    }
    if (grid as usize) * SM120_TF32_STREAMK_SLOTS_PER_CTA > TF32_SPLITK_COUNTER_CAP {
        return Err("SM120 stream-K flags exceed the fixed counter workspace".into());
    }
    let (partial, _) = kernels.splitk_scratch_buf(stream)?.device_ptr(stream);
    let (flags, _) = kernels
        .triad_kernels()
        .tf32_splitk_counter_buf(stream)?
        .device_ptr(stream);
    Ok(Some(Sm120StreamKWorkspace { partial, flags }))
}

fn sm120_policy_dtype(dtype: WeightDtype) -> Result<PolicyDtype, String> {
    match dtype {
        WeightDtype::Bf16 => Ok(PolicyDtype::Bf16),
        WeightDtype::F16 => Ok(PolicyDtype::F16),
        WeightDtype::F32 => Err("SM120 automatic route requires BF16 or F16".into()),
    }
}

pub(in crate::mamba_ssm::gpu) fn launch_sm120_auto_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &mut O,
    request: Sm120AutoRequest,
) -> Result<Option<Sm120AutoBranchSeal>, String> {
    // Off the SM120 family the tiles are not compiled and silence is the
    // designed answer; on the family every decline is reported once.
    let family = ctx.kernels.triad_kernels().serves_sm120_family();
    let Some(caps) = ctx.kernels.sm120_device_caps() else {
        if family {
            static NO_CAPS: std::sync::Once = std::sync::Once::new();
            crate::mamba_ssm::gpu::diagnostics::warn_once(&NO_CAPS, || {
                "this SM120 board reports no SM120 device capabilities; the portable \
                 tensor-core tiles serve every half GEMM"
                    .to_string()
            });
        }
        return Ok(None);
    };
    let Some(target) = ctx.kernels.sm120_target_candidate() else {
        if family {
            static NO_TARGET: std::sync::Once = std::sync::Once::new();
            crate::mamba_ssm::gpu::diagnostics::warn_once(&NO_TARGET, || {
                "this SM120 board has no bound SM120 module target; the portable \
                 tensor-core tiles serve every half GEMM"
                    .to_string()
            });
        }
        return Ok(None);
    };
    let Some(route) = resolve_sm120_auto(caps, Some(target), request) else {
        if family {
            static NO_CELL: std::sync::Once = std::sync::Once::new();
            crate::mamba_ssm::gpu::diagnostics::warn_once(&NO_CELL, || {
                format!(
                    "no measured SM120 half cell for {:?} {:?} {:?}; the portable tensor-core \
                     tiles serve it (reported once; later uncovered shapes are silent)",
                    request.op, request.dtype, request.shape
                )
            });
        }
        return Ok(None);
    };
    let key = Sm120PreparedKey::new(ctx.gemm_route(), route, request);
    ctx.with_sm120_prepared_launches(|cache| {
        let prepared = cache.ensure_sm120_prepared(ctx, key, route, request)?;
        let identity = prepared.identity();
        let resolved = identity.resolved_route()?;
        unsafe {
            enqueue_sm120_tma_prepared_observed(
                &ctx.stream,
                &ctx.kernels,
                prepared,
                observer,
                Some(PhysicalLaunchObservation::gemm(
                    sm120_policy_dtype(route.dtype)?,
                    None,
                    resolved,
                )),
            )
        }?;
        ctx.record_resolved_gemm_route(resolved)?;
        Ok(Some(Sm120AutoBranchSeal { route }))
    })
}

pub(in crate::mamba_ssm::gpu) fn prepare_sm120_auto_graph_sequence<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &O,
    request: Sm120AutoRequest,
) -> Result<PreparedTriadPhysicalGraphSequence, String> {
    let caps = ctx
        .kernels
        .sm120_device_caps()
        .ok_or_else(|| "prepared SM120 graph route has no device capabilities".to_string())?;
    let target = ctx
        .kernels
        .sm120_target_candidate()
        .ok_or_else(|| "prepared SM120 graph route has no module target".to_string())?;
    let route = resolve_sm120_auto(caps, Some(target), request)
        .ok_or_else(|| "prepared SM120 graph request is no longer qualified".to_string())?;
    let key = Sm120PreparedKey::new(ctx.gemm_route(), route, request);
    ctx.with_sm120_prepared_launches(|cache| {
        let prepared = &cache
            .entries
            .get(&key)
            .ok_or_else(|| {
                "prepared SM120 graph cache entry is missing; run eager warmup again".to_string()
            })?
            .prepared;
        validate_sm120_graph_replay(&ctx.stream, &ctx.kernels, prepared)?;
        let resolved = prepared.identity().resolved_route()?;
        let config = LaunchConfig {
            grid_dim: resolved.launch.grid_dim,
            block_dim: resolved.launch.block_dim,
            shared_mem_bytes: resolved.launch.shared_mem_bytes,
        };
        let observation =
            PhysicalLaunchObservation::gemm(sm120_policy_dtype(route.dtype)?, None, resolved);
        let node = resolve_physical_launch_observation(observer, observation, config)?;
        let function = ctx
            .kernels
            .sm120_function(resolved.symbol)
            .ok_or_else(|| format!("qualified SM120 symbol {} is unavailable", resolved.symbol))?
            .clone();
        let mut arguments = PhysicalScalarKernelArguments::new();
        arguments.push(prepared.operands.output_ptr)?;
        let spec = prepared.route.kernel_spec()?;
        if let Some(workspace) = sm120_streamk_workspace(
            &ctx.stream,
            &ctx.kernels,
            prepared.route,
            spec,
            resolved.launch.grid_dim.0,
        )? {
            arguments.push(workspace.partial)?;
            arguments.push(workspace.flags)?;
        }
        arguments.push(prepared.maps.a)?;
        arguments.push(prepared.maps.b)?;
        arguments.push(prepared.operands.bias_ptr)?;
        arguments.push(Sm120KernelParams::from_words(prepared.params))?;
        validate_sm120_graph_replay(&ctx.stream, &ctx.kernels, prepared)?;
        Ok(PreparedTriadPhysicalGraphSequence {
            launches: vec![PreparedTriadPhysicalGraphLaunch {
                function,
                config,
                node,
                arguments: Box::new(arguments),
            }]
            .into_boxed_slice(),
        })
    })
}

/// Batched linear forward on GPU: `Y[B,N] = X[B,K] @ W[K,N] + bias[N]`.
///
/// cuBLAS computes: `Y^T[N,B] = W^T[N,K] @ X^T[K,B]` (column-major).
/// With row-major data, this is equivalent to: `Y[B,N] = X[B,K] @ W[K,N]`.
///
/// Bias is broadcast via pre-fill + beta=1.0 accumulate.
///
/// # Arguments
/// - `y`: output `[B * N]`, overwritten
/// - `x`: input `[B * K]`
/// - `w`: weights `[K * N]`
/// - `bias`: optional `[N]`, broadcast to each row
/// - `batch`: B (number of samples)
/// - `n_in`: K (input dimension)
/// - `n_out`: N (output dimension)
pub fn gemm_bi_forward(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: CUptr,
    bias_ptr: CUptr, // 0 = no bias
    dims: (usize, usize, usize),
) -> Result<(), String> {
    GemmDims::nn(dims, dims.1)?;
    let x_ptr = {
        use cudarc::driver::DevicePtr;
        let (ptr, _r) = x.inner().device_ptr(stream);
        ptr
    };
    let operands = GemmBiFwdSubOperands {
        x_ptr,
        lda: dims.1,
        w_ptr,
        bias_ptr,
    };
    gemm_bi_forward_sub(stream, kernels, y, &operands, dims)
}

/// [`gemm_bi_forward`] over a STRIDED X operand: `x_ptr` is the first
/// element of an [M, K] sub-matrix whose row stride is `lda` elements
/// (lda >= K). Every bucket's kernel already takes lda and addresses A
/// as `row * lda + col`, so a sub-matrix read is the same per-output
/// ascending-K FMA chain as a gathered copy — bit-identical operands,
/// gather kernel deleted at the call site.
pub fn gemm_bi_forward_sub(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    y: &mut GpuBuffer,
    operands: &GemmBiFwdSubOperands,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    gemm_bi_forward_sub_with_control(
        stream,
        kernels,
        y,
        operands,
        dims,
        None::<&mut ScalarLaunchControl<'_>>,
    )
}

fn enqueue_scalar_forward<C: ScalarLaunchController>(
    control: &mut Option<&mut C>,
    _request: F32TriadRequest,
    _operands: F32TriadOperands,
    symbol: &'static str,
    config: cudarc::driver::LaunchConfig,
    builder: &mut ScalarLaunchArgs<'_>,
) -> Result<(), PhysicalCudaLaunchError> {
    if let Some(control) = control.as_deref_mut() {
        return control.enqueue(symbol, config, builder);
    }
    let mut observer = NoPhysicalObserver;
    unsafe { enqueue_with_physical_observation(&mut observer, builder.launch_args(), config, None) }
}

fn gemm_bi_forward_sub_with_control<C: ScalarLaunchController>(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    y: &mut GpuBuffer,
    operands: &GemmBiFwdSubOperands,
    dims: (usize, usize, usize),
    mut control: Option<&mut C>,
) -> Result<(), String> {
    let GemmBiFwdSubOperands {
        x_ptr,
        lda,
        w_ptr,
        bias_ptr,
    } = *operands;
    let checked_dims = GemmDims::nn(dims, lda)?;
    let (batch, n_in, n_out) = checked_dims.tuple();
    let request = F32TriadRequest {
        op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp::Nn,
        shape: F32TriadShape {
            m: batch,
            k: n_in,
            n: n_out,
            lda,
            ldb: n_out,
            ldc: n_out,
        },
    };
    let lda_i = checked_dims.lda;
    let alpha = control
        .as_ref()
        .map(|control| control.operands().alpha)
        .unwrap_or(1.0);
    let beta = control
        .as_ref()
        .map(|control| control.operands().beta)
        .unwrap_or(0.0);
    use cudarc::driver::DevicePtr;
    let (output, _) = y.inner().device_ptr(stream);
    let actual_operands = F32TriadOperands {
        output,
        a: x_ptr,
        b: w_ptr,
        bias: (bias_ptr != 0).then_some(bias_ptr),
        alpha,
        beta,
    };
    if let Some(control) = control.as_deref() {
        control.validate_operands(actual_operands)?;
    }
    let scalar_plan = match control.as_ref() {
        Some(prepared) => prepared.plan(),
        None => scalar_launch_plan(scalar_launch_facts(kernels), request, actual_operands)?,
    };
    validate_bias_preseed(alpha, bias_ptr, "gemm_bi_forward_sub")?;
    // Shape-A Ultra-Thin-M NN dispatch: batch ∈ [1, 31] (actor inference rollout).
    // Covers shapes that fall through Split-K (min 32) and Big/Slim (min 128).
    // Grid: (ceil(N/32), M, 1). smem = K*4 bytes ≤ 8 KB (K ≤ 2048) — within the
    // 48 KB default dynamic-smem limit on sm_80+.
    // Non-mod-32 N handled by kernel's `col < N` predication (tail tile partial).
    // K up to 2048 covers SimbaV2 w2 forward (K=2048 N=512 when batch < 32).
    if matches!(scalar_plan, ScalarDispatchPlan::NnUltraThin) {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_i = checked_dims.k_i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_dims.n_u32.div_ceil(32), checked_dims.m_u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: checked_u32_product(
                checked_dims.k_u32,
                checked_u32(std::mem::size_of::<f32>(), "f32 byte width")?,
                "ultra-thin shared memory",
            )?,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_nn_ultra_thin, &control);
        builder.arg_buffer_mut(y);
        builder.arg(&x_ptr);
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&lda_i); // lda
        builder.arg(&n_i); // ldb
        builder.arg(&n_i); // ldc
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_nn_ultra_thin",
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_nn_ultra_thin forward"))
        })?;
        return Ok(());
    }

    // Narrow-N NN small-tile dispatch: N∈[2..127] AND batch ≤ 64.
    // Representative production shape: M=64, K=512, N=25. Tile
    // NBM=16 NBN=16 NBK=16, 64 threads (2 warps). At M=64 N=25 grid is
    // ceil(64/16) × ceil(25/16) = 4 × 2 = 8 CTAs (vs 1 for the big-tile
    // narrow kernel). Per-output FMA chain is byte-identical to
    // gemm_bi_nn_narrow regardless of tile — same ascending K __fmaf_rn,
    // same bias pre-seed at K=0, same scalar N-tail epilogue. ZERO ULP
    // downstream drift; CPU mirror (narrow_nn_sgemm_nn in blas_bi.rs) is
    // tile-agnostic and matches both GPU variants.
    if matches!(scalar_plan, ScalarDispatchPlan::NnNarrowSmall) {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_i = checked_dims.k_i32;
        let post_op: i32 = 0;
        let num_pid_m = checked_dims.m_u32.div_ceil(16);
        let num_pid_n = checked_dims.n_u32.div_ceil(16);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_grid_product(num_pid_m, num_pid_n, 1)?, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_nn_narrow_small, &control);
        builder.arg_buffer_mut(y);
        builder.arg(&x_ptr);
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&lda_i);
        builder.arg(&n_i);
        builder.arg(&n_i);
        builder.arg(&post_op);
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_nn_narrow_small",
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_nn_narrow_small forward"))
        })?;
        return Ok(());
    }

    // Narrow-N NN dispatch: N∈[2..127], batch > 64.
    // Tile BM=64 BN=32 BK=16, 128 threads, 2x2 warps. Scalar N-epilogue.
    // Kernel has M-predication (`if (g_row >= M) continue;`) and N-predication
    // (`if (g_col >= N) continue;`) → safe for any batch and any N via tile count.
    // Covers test-config shapes (M=32, K=32..64, N=32..64) that otherwise fall
    // to cuBLAS (non-deterministic, violates zero-cuBLAS contract).
    if matches!(scalar_plan, ScalarDispatchPlan::NnNarrow) {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_i = checked_dims.k_i32;
        let post_op: i32 = 0;
        let num_pid_m = checked_dims.m_u32.div_ceil(64);
        let num_pid_n = checked_dims.n_u32.div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_grid_product(num_pid_m, num_pid_n, 1)?, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_nn_narrow, &control);
        builder.arg_buffer_mut(y);
        builder.arg(&x_ptr);
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&lda_i);
        builder.arg(&n_i);
        builder.arg(&n_i);
        builder.arg(&post_op);
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_nn_narrow",
            cfg,
            &mut builder,
        )
        .map_err(|error| error.with_driver_context(format_args!("gemm_bi_nn_narrow forward")))?;
        return Ok(());
    }

    // GEMV-N1 dispatch: N=1 output (actor mean/log_std heads).
    // 4 rows/block, warp-shuffle K-reduction, deterministic batch-invariant.
    //
    // batch lower bound relaxed 4 → 1. Kernel
    // gemm_bi_nn_gemv has `if (row >= M) return;` predication in scalar.cu.
    // so M<4 is safe — partial last block. Closes single-env eval gap
    // (M=1 N=1 K=512 was hitting cuBLAS-fallback panic in an eval-parity test).
    // Determinism preserved (kernel unchanged; same warp-shuffle butterfly).
    if matches!(scalar_plan, ScalarDispatchPlan::NnGemv) {
        let m_i = checked_dims.m_i32;
        let k_i = checked_dims.k_i32;
        let ldy_i: i32 = 1;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_dims.m_u32.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_nn_gemv, &control);
        builder.arg_buffer_mut(y);
        builder.arg(&x_ptr);
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&lda_i);
        builder.arg(&ldy_i);
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_nn_gemv",
            cfg,
            &mut builder,
        )
        .map_err(|error| error.with_driver_context(format_args!("gemm_bi_nn_gemv forward")))?;
        return Ok(());
    }

    // Split-K Thin-M NN + K-tail dispatch for M<128 shapes with non-%32 K.
    // Decompose K = K_main + K_tail, where K_main = K - K%32 (multiple of 32),
    // K_tail = K%32 (1..31). Main is processed by the existing Split-K NN kernel
    // with lda = full K (so it reads only columns [0..K_main) of each row).
    // Tail is folded into the reducer as Σ_k X[m, K_main+k] · W[K_main+k, n].
    // Universal: works for K ∈ {33..65535} with any K%32 ≠ 0 — covers K=129,
    // K=257 (SALE action), K=385, K=513, K=642 (hypermlp forward), etc.
    // Use module-level SPLITK_SCRATCH_CAP (was per-function duplicated).
    //
    // Phase C-1.5bo: Slim NN underfill guard. Even with cap≤1024, wide-N shapes
    // (n_out=2048 at batch=1024) give Slim NN base_blocks=8*32=256 = 1.8 waves —
    // already saturated; Split-K's partial+reducer (DRAM scratch round-trip) is
    // strictly negative. The live one-wave threshold admits true underfill only.
    // It is tighter than the Slim Split-K's three-wave threshold because
    // Thin-M's BM=32 produces 4× the tile count of Slim NN's BM=128.
    // Replaces the implicit "batch≤1024" guard with an explicit M_tiles*N_tiles
    // check that doesn't rely on cap-relax envelope.
    if let ScalarDispatchPlan::NnSplitKThinTail { k_main, k_tail } = scalar_plan {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_chunks = checked_i32(k_main / 32, "NN K-tail chunks")?;
        let num_pid_m = checked_dims.m_u32.div_ceil(32);
        let num_pid_n = checked_dims.n_u32.div_ceil(64);
        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                checked_grid_product(
                    num_pid_m,
                    num_pid_n,
                    checked_u32(k_main / 32, "NN K-tail chunks")?,
                )?,
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let partial_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };
        // Main Split-K partial on A columns [0..k_main), B rows [0..k_main).
        let mut pb = scalar_launch_builder(stream, &kernels.gemm_bi_nn_splitk32_partial, &control);
        pb.arg(&partial_ptr);
        pb.arg(&x_ptr);
        pb.arg(&w_ptr);
        pb.arg(&m_i);
        pb.arg(&n_i);
        pb.arg(&k_chunks);
        pb.arg(&lda_i);
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_nn_splitk32_partial",
            partial_cfg,
            &mut pb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_nn_splitk32_partial (K-tail main)"))
        })?;

        // Tail fold via reducer: tail_cnt iterations of X[m, K_main+k] · W[K_main+k, n].
        let total = checked_dims.mn_u32;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let zero_i32: i32 = 0;
        let tail_cnt_i = checked_i32(k_tail, "NN K-tail count")?;
        let x_tail_ptr = checked_ptr_add(
            x_ptr,
            checked_byte_offset(k_main, std::mem::size_of::<f32>(), "X tail")?,
            "X tail",
        )?;
        let w_tail_ptr = checked_ptr_add(
            w_ptr,
            checked_byte_offset(
                k_main.checked_mul(n_out).ok_or_else(|| {
                    invalid_gemm_dimensions("W tail element offset overflows usize")
                })?,
                std::mem::size_of::<f32>(),
                "W tail",
            )?,
            "W tail",
        )?;
        let x_tail_stride_i = lda_i; // stride between X[m, k_main] rows = the A row stride
        let mut rb = scalar_launch_builder(stream, &kernels.gemm_bi_splitk_reduce, &control);
        rb.arg_buffer_mut(y);
        rb.arg(&partial_ptr);
        rb.arg(&bias_ptr);
        rb.arg(&x_tail_ptr);
        rb.arg(&w_tail_ptr);
        rb.arg(&alpha);
        rb.arg(&m_i);
        rb.arg(&n_i);
        rb.arg(&k_chunks);
        rb.arg(&x_tail_stride_i);
        rb.arg(&zero_i32); // out_col_stride default = N
        rb.arg(&tail_cnt_i);
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_splitk_reduce",
            reduce_cfg,
            &mut rb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_splitk_reduce (K-tail)"))
        })?;
        return Ok(());
    }

    // Split-K Thin-M NN dispatch and its exact qualified M32N64 partial.
    // K is split into 32-wide chunks, partial GEMMs run per (m,n,kc) block,
    // and the same deterministic reducer consumes the unchanged partial layout.
    //
    // Envelope: M ∈ [32, 127], N ∈ [64, 512], K % 32 == 0, K ≥ 32,
    // and partial_size = K_CHUNKS*M*N ≤ 2M floats (scratch capacity).
    //
    // Phase C-1.5bo: same Slim NN underfill guard as K-tail variant above.
    if matches!(
        scalar_plan,
        ScalarDispatchPlan::NnSplitKThin | ScalarDispatchPlan::NnM32N64SplitK32Qualified
    ) {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_chunks = checked_i32(n_in / 32, "NN split-K chunks")?;

        // Partial kernel launch: grid = M_tiles × N_tiles × K_CHUNKS
        let num_pid_m = checked_dims.m_u32.div_ceil(32);
        let num_pid_n = checked_dims.n_u32.div_ceil(64);
        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                checked_grid_product(
                    num_pid_m,
                    num_pid_n,
                    checked_u32(n_in / 32, "NN split-K chunks")?,
                )?,
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let partial_ptr = {
            // Raw device pointer to pre-allocated scratch (1 MB, shared across calls).
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };
        let (partial_function, partial_symbol) =
            if scalar_plan == ScalarDispatchPlan::NnM32N64SplitK32Qualified {
                (
                    &kernels.gemm_bi_nn_splitk32_m32n64_exact_v1,
                    "gemm_bi_nn_splitk32_m32n64_exact_v1",
                )
            } else {
                (
                    &kernels.gemm_bi_nn_splitk32_partial,
                    "gemm_bi_nn_splitk32_partial",
                )
            };
        let mut pb = scalar_launch_builder(stream, partial_function, &control);
        pb.arg(&partial_ptr);
        pb.arg(&x_ptr);
        pb.arg(&w_ptr);
        pb.arg(&m_i);
        pb.arg(&n_i);
        pb.arg(&k_chunks);
        pb.arg(&lda_i);
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            partial_symbol,
            partial_cfg,
            &mut pb,
        )
        .map_err(|error| error.with_driver_context(format_args!("{partial_symbol}")))?;

        // Reduce kernel launch: grid covers M*N outputs, 256 threads/block.
        let total = checked_dims.mn_u32;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let null_tail: u64 = 0;
        let zero_i32: i32 = 0;
        let mut rb = scalar_launch_builder(stream, &kernels.gemm_bi_splitk_reduce, &control);
        rb.arg_buffer_mut(y);
        rb.arg(&partial_ptr);
        rb.arg(&bias_ptr);
        rb.arg(&null_tail); // x_tail_ptr (none)
        rb.arg(&null_tail); // w_tail_ptr (none)
        rb.arg(&alpha);
        rb.arg(&m_i);
        rb.arg(&n_i);
        rb.arg(&k_chunks);
        rb.arg(&zero_i32); // x_tail_stride (unused)
        rb.arg(&zero_i32); // out_col_stride = N (default)
        rb.arg(&zero_i32); // tail_cnt = 0 (no tail)
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_splitk_reduce",
            reduce_cfg,
            &mut rb,
        )
        .map_err(|error| error.with_driver_context(format_args!("gemm_bi_splitk_reduce")))?;
        return Ok(());
    }

    // Split-K Slim NN for fat-M shapes (M > 1024) that underfill
    // the Slim grid. Targets Mamba layer shapes at b=64 seq=33 → M=2112.
    // Tile BM=128 BN=64 BK=32 (same as gemm_bi_nn_slim) — each fc's K-slice
    // has identical per-block FMA order to Slim NN on that K-range. Reducer
    // gemm_bi_splitk_reduce applies alpha + bias, overwrites y.
    //
    // Determinism: K_CHUNK is a COMPILE-TIME CONSTANT → F = ceil(K / K_CHUNK)
    // depends ONLY on K. Same K always produces same F (and same per-fc
    // k-range) regardless of M, N, batch, stream, or SM scheduling.
    // Reducer f32 ascending-fc order (`gemm_bi_splitk_reduce`; only the
    // split-M TN reducer is f64). Batch-invariant by construction.
    //
    // Gate ordering: fires AFTER Thin-M cap (batch > 1024) so it never steals
    // shapes Thin-M handles well (M ≤ 1024 has 4× BM=32 tiles vs 1× BM=128,
    // better fill under Thin-M). Only fat-M Mamba shapes land here.
    //
    // K_CHUNK choice: 64 (2× BK=32) — gives F=2 for K=128 (Mamba in_proj),
    // F=4 for K=256 (out_proj). Since z_s left the Mamba input, the input_proj
    // K is `obs_dim` (was `obs_dim + emb = 384`), which falls below the F≥6 gate
    // below — Mamba input_proj routes via regular Slim NN dispatch now.
    // Sweet spot: small enough to split K=128, big enough that each fc does
    // 2+ BK iterations to amortize kernel launch overhead.
    //
    // F must not depend on base_blocks (M_tiles*N_tiles/SMs): that makes equal
    // logical reductions at different batch sizes use different chunk counts,
    // changes reduction order, and causes bit drift.
    const SPLITK_SLIM_K_CHUNK: u32 = 64; // PURE K-BASED, batch-invariant
    if let ScalarDispatchPlan::NnSplitKSlim { chunks: f_final } = scalar_plan {
        // F ≥ 6 (K ≥ 384). Ncu profiles:
        //   - SALE L1 (M=4224 K=128 N=128, F=2): reducer 35% of time → moved
        //     out of splitk_slim in a prior commit (F≥4 gate).
        //   - SALE L2/L3 (M=4224 K=256 N=256, F=4): splitk_reduce kernel is
        //     DRAM-bound at 83% throughput, partial kernel SM at 30%. The
        //     132 M-N output tiles (≥ 128 SMs) already saturate without
        //     K-split → splitk_slim only adds reducer overhead. Moved out
        //     of splitk_slim at the F≥6 raise.
        //   - (Historical) Mamba input_proj (M=3840 K=384 N=128, F=6): 60 M-N
        //     tiles < SM count, F=6 wave-fill was a win. With z_s dropped
        //     from the Mamba input, K=obs_dim falls below the F≥6 gate
        //     and routes via regular Slim NN. Gate retained for any future
        //     K∈[384,512] fat-M shape.
        //
        // After this gate raise, shapes with K < 384 fall to the regular
        // Slim NN dispatch below (single kernel, no reducer overhead).
        // The CPU mirror uses this same underfill threshold.
        let base_blocks = checked_tile_grid(checked_dims.m_u32, 128, checked_dims.n_u32, 64)?;
        let k_chunk = SPLITK_SLIM_K_CHUNK;
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_i = checked_dims.k_i32;
        let ldb_i = checked_dims.n_i32; // B is [K, N], row-major
        let k_chunk_i = checked_i32(
            checked_usize(k_chunk, "NN slim split-K chunk")?,
            "NN slim split-K chunk",
        )?;

        let partial_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };

        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (base_blocks, 1, f_final),
            block_dim: (128, 1, 1), // Slim tile: 128 threads
            shared_mem_bytes: 0,    // static smem
        };
        let mut pb =
            scalar_launch_builder(stream, &kernels.gemm_bi_nn_splitk_slim_partial, &control);
        pb.arg(&partial_ptr);
        pb.arg(&x_ptr);
        pb.arg(&w_ptr);
        pb.arg(&m_i);
        pb.arg(&n_i);
        pb.arg(&k_i);
        pb.arg(&lda_i);
        pb.arg(&ldb_i);
        pb.arg(&k_chunk_i);
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_nn_splitk_slim_partial",
            partial_cfg,
            &mut pb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_nn_splitk_slim_partial"))
        })?;

        let total = checked_dims.mn_u32;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let null_tail: u64 = 0;
        let zero_i32_local: i32 = 0;
        let f_i = checked_i32(
            checked_usize(f_final, "NN slim split-K chunks")?,
            "NN slim split-K chunks",
        )?;
        let mut rb = scalar_launch_builder(stream, &kernels.gemm_bi_splitk_reduce, &control);
        rb.arg_buffer_mut(y);
        rb.arg(&partial_ptr);
        rb.arg(&bias_ptr);
        rb.arg(&null_tail); // x_tail_ptr (none)
        rb.arg(&null_tail); // w_tail_ptr (none)
        rb.arg(&alpha);
        rb.arg(&m_i);
        rb.arg(&n_i);
        rb.arg(&f_i); // K_chunks = F
        rb.arg(&zero_i32_local); // x_tail_stride (unused)
        rb.arg(&zero_i32_local); // out_col_stride default = N
        rb.arg(&zero_i32_local); // tail_cnt = 0 (K % 32 == 0 enforced)
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_splitk_reduce",
            reduce_cfg,
            &mut rb,
        )
        .map_err(|error| error.with_driver_context(format_args!("gemm_bi_splitk_reduce (slim)")))?;
        return Ok(());
    }

    if matches!(scalar_plan, ScalarDispatchPlan::NnM64N64Qualified) {
        let params = SgbNnM64N64Params {
            alpha,
            beta,
            m: checked_dims.m_i32,
            n: checked_dims.n_i32,
            k: checked_dims.k_i32,
            lda: lda_i,
            ldb: checked_dims.n_i32,
            ldc: checked_dims.n_i32,
        };
        let total_tiles = checked_tile_grid(checked_dims.m_u32, 64, checked_dims.n_u32, 64)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: super::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES,
        };
        let mut builder =
            scalar_launch_builder(stream, &kernels.gemm_bi_nn_m64n64_bk16_s2_v1, &control);
        builder.arg_buffer_mut(y);
        builder.arg(&x_ptr);
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&params);
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            "gemm_bi_nn_m64n64_bk16_s2_v1",
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_nn_m64n64_bk16_s2_v1 forward"))
        })?;
        return Ok(());
    }

    // Custom deterministic GEMM.
    // Envelope: M ≥ 128, N ≥ 128, K ≥ 1. Non-%4 N handled by kernel scalar N-epilogue.
    // Non-%4 K handled by kernel scalar K-fallback (runtime lda%4 check).
    // K<BK: kernel's scalar bounds check zero-fills smem for dotIdx≥K; wastes a few FMAs
    // but correct (handles Mamba-1 dt_proj K=4,8). dropped `n_in >= 16` guard.
    if let ScalarDispatchPlan::NnFinal { slim } = scalar_plan {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_i = checked_dims.k_i32;
        let (func, bn) = if slim {
            (&kernels.gemm_bi_nn_slim, 64)
        } else {
            (&kernels.gemm_bi_nn, 128)
        };
        // Opt1: Big uses 256 threads/block for TLP; Slim stays 128.
        let threads = if slim { 128u32 } else { 256u32 };
        // Big NN uses dynamic smem for its 2-stage cp.async pipeline.
        // Slim still uses static smem (single-stage). Set shared_mem_bytes only for Big.
        let smem_bytes: u32 = if slim { 0 } else { 34 * 1024 };
        // The kernel is data-parallel, one tile per CTA, so the grid covers
        // every output tile without a persistent-CTA cap.
        let total_tiles = checked_tile_grid(checked_dims.m_u32, 128, checked_dims.n_u32, bn)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let mut builder = scalar_launch_builder(stream, func, &control);
        builder.arg_buffer_mut(y);
        builder.arg(&x_ptr);
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&lda_i); // lda (A row stride; = K for contiguous X)
        builder.arg(&n_i); // ldb = n_out (B is [K, N])
        builder.arg(&n_i); // ldc = n_out (C is [M, N])
        enqueue_scalar_forward(
            &mut control,
            request,
            actual_operands,
            if slim {
                "gemm_bi_nn_slim"
            } else {
                "gemm_bi_nn"
            },
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!(
                "gemm_bi_nn{} forward",
                if slim { "_slim" } else { "" }
            ))
        })?;
        return Ok(());
    }

    // The zero-cuBLAS contract requires every training path to route through
    // custom deterministic kernels. A reachable cuBLAS fallback breaks
    // CPU↔GPU parity and is non-deterministic. Panic loudly so missing
    // dispatch coverage is caught at first hit, not as a silent training
    // regression months later.
    panic!(
        "gpu_gemm_bi_forward: cuBLAS fallback hit (shape M={batch} K={n_in} N={n_out}). \
         The zero-cuBLAS contract requires every shape to route through a custom \
         kernel — add a dispatcher branch in this function for this shape."
    );
}

#[cfg(test)]
fn scalar_backward_launch_plan(
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    multiprocessor_count: u32,
) -> Result<(GemmDims, F32TriadRequest, ScalarDispatchPlan), String> {
    let (checked_dims, request) = scalar_backward_request(op, dims)?;
    let plan = scalar_dispatch_plan(request, multiprocessor_count)?;
    Ok((checked_dims, request, plan))
}

fn scalar_backward_request(
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
) -> Result<(GemmDims, F32TriadRequest), String> {
    let checked_dims = match op {
        ResolvedGemmOp::Tn => GemmDims::tn(dims)?,
        ResolvedGemmOp::Nt => GemmDims::nt(dims)?,
        ResolvedGemmOp::Nn => return Err("backward scalar launch plan requires TN or NT".into()),
    };
    let request = F32TriadRequest {
        op,
        shape: F32TriadShape::contiguous(op, dims),
    };
    Ok((checked_dims, request))
}

fn enqueue_scalar_backward<C: ScalarLaunchController>(
    control: &mut Option<&mut C>,
    _request: F32TriadRequest,
    _operands: F32TriadOperands,
    symbol: &'static str,
    config: cudarc::driver::LaunchConfig,
    builder: &mut ScalarLaunchArgs<'_>,
) -> Result<(), PhysicalCudaLaunchError> {
    if let Some(control) = control.as_deref_mut() {
        return control.enqueue(symbol, config, builder);
    }
    let mut observer = NoPhysicalObserver;
    unsafe { enqueue_with_physical_observation(&mut observer, builder.launch_args(), config, None) }
}

/// Weight gradient: `dW[K,N] += X^T[K,B] @ dY[B,N]` (accumulated, beta=1.0).
///
/// cuBLAS: `dW^T[N,K] += dY^T[N,B] @ X[B,K]`
/// In col-major: A=dY (transa=N gives `dY^T[N,B]`), B=X_saved (transb=T gives `X[B,K]`)
/// gemm(N, T, N, K, B, 1.0, dY, N, X_saved, K, 1.0, dW, N)
///
/// Note: beta=1.0 for gradient accumulation.
pub fn gemm_bi_backward_dw(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dw_ptr: CUptr, // accumulated in place (+=)
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    gemm_bi_backward_dw_with_control(
        stream,
        kernels,
        dw_ptr,
        dy,
        x_saved,
        dims,
        None::<&mut ScalarLaunchControl<'_>>,
    )
}

fn gemm_bi_backward_dw_with_control<C: ScalarLaunchController>(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dw_ptr: CUptr,
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    dims: (usize, usize, usize),
    mut control: Option<&mut C>,
) -> Result<(), String> {
    let (checked_dims, request) = scalar_backward_request(ResolvedGemmOp::Tn, dims)?;
    let (batch, n_in, n_out) = checked_dims.tuple();
    let alpha = control
        .as_ref()
        .map(|control| control.operands().alpha)
        .unwrap_or(1.0);
    let operands = F32TriadOperands {
        output: dw_ptr,
        a: x_saved.raw_ptr(stream),
        b: dy.raw_ptr(stream),
        bias: None,
        alpha,
        beta: 1.0,
    };
    let scalar_plan = match control.as_ref() {
        Some(prepared) => prepared.plan(),
        None => scalar_launch_plan(scalar_launch_facts(kernels), request, operands)?,
    };
    if let Some(control) = control.as_ref() {
        let prepared = control.operands();
        if prepared.output != operands.output
            || prepared.a != operands.a
            || prepared.b != operands.b
            || prepared.bias != operands.bias
            || prepared.alpha.to_bits() != operands.alpha.to_bits()
            || prepared.beta.to_bits() != operands.beta.to_bits()
        {
            return Err("prepared TN operands differ from the physical launch arguments".into());
        }
    }
    // GEMV-N1 TN dispatch: dW[K,1] += X^T[K,M] @ dY[M,1]
    if matches!(scalar_plan, ScalarDispatchPlan::TnGemv) {
        let m_i = checked_dims.m_i32;
        let k_i = checked_dims.k_i32;
        let lda_i = checked_dims.k_i32;
        let ldy_i: i32 = 1;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_dims.k_u32.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_tn_gemv, &control);
        builder.arg(&dw_ptr);
        builder.arg_buffer(x_saved);
        builder.arg_buffer(dy);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&lda_i);
        builder.arg(&ldy_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_tn_gemv",
            cfg,
            &mut builder,
        )
        .map_err(|error| error.with_driver_context(format_args!("gemm_bi_tn_gemv backward_dw")))?;
        return Ok(());
    }

    // Narrow-N TN dispatch: N∈[2..127], including the N∈[49..127] gap
    // where slim/big kernels (N>=128) don't apply.
    // The gate starts at N=2; N=1 is handled by the GEMV route above.
    // Kernel has `if (g_row >= K_out) continue;` and N-tile predication via
    // `div_ceil(N, 32)` blocks → safe for any n_in and any N.
    // Relaxed to n_in>=1, batch>=1 covers test shapes (M=32, K=32..64, N=32..64)
    // that otherwise fall to cuBLAS (zero-cuBLAS contract violation).
    if matches!(scalar_plan, ScalarDispatchPlan::TnNarrow) {
        let m_i = checked_dims.m_i32;
        let k_i = checked_dims.k_i32;
        let n_i = checked_dims.n_i32;
        let num_pid_m = checked_dims.k_u32.div_ceil(64);
        let num_pid_n = checked_dims.n_u32.div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_grid_product(num_pid_m, num_pid_n, 1)?, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_tn_narrow, &control);
        builder.arg(&dw_ptr);
        builder.arg_buffer(x_saved);
        builder.arg_buffer(dy);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&n_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_tn_narrow",
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_tn_narrow backward_dw"))
        })?;
        return Ok(());
    }

    if let ScalarDispatchPlan::TnNarrowSplitM {
        m_chunk,
        chunks: f_final,
    } = scalar_plan
    {
        let grid_m = checked_dims.k_u32.div_ceil(64);
        let grid_n = checked_dims.n_u32.div_ceil(32);
        let m_i = checked_dims.m_i32;
        let k_i = checked_dims.k_i32;
        let n_i = checked_dims.n_i32;
        let m_chunk_i = checked_i32(m_chunk, "TN narrow split-M chunk")?;
        let f_i = checked_i32(f_final, "TN narrow split-M partitions")?;
        let f_u32 = checked_u32(f_final, "TN narrow split-M partitions")?;
        let partial_ptr = {
            use cudarc::driver::DevicePtr;
            kernels.splitk_scratch_buf(stream)?.device_ptr(stream).0
        };
        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_m, grid_n, f_u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let symbol = scalar_tn_kernel_symbol(scalar_plan, operands);
        let function = if symbol == "gemm_bi_tn_narrow_splitm_partial_aligned" {
            &kernels.gemm_bi_tn_narrow_splitm_partial_aligned
        } else {
            &kernels.gemm_bi_tn_narrow_splitm_partial
        };
        let mut partial_builder = scalar_launch_builder(stream, function, &control);
        partial_builder.arg(&partial_ptr);
        partial_builder.arg_buffer(x_saved);
        partial_builder.arg_buffer(dy);
        partial_builder.arg(&m_i);
        partial_builder.arg(&k_i);
        partial_builder.arg(&n_i);
        partial_builder.arg(&m_chunk_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            symbol,
            partial_cfg,
            &mut partial_builder,
        )
        .map_err(|error| error.with_driver_context(format_args!("{symbol}")))?;

        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_dims.kn_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut reducer = scalar_launch_builder(stream, &kernels.gemm_bi_splitm_reduce, &control);
        reducer.arg(&dw_ptr);
        reducer.arg(&partial_ptr);
        reducer.arg(&alpha);
        reducer.arg(&k_i);
        reducer.arg(&n_i);
        reducer.arg(&f_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_splitm_reduce",
            reduce_cfg,
            &mut reducer,
        )
        .map_err(|error| error.with_driver_context(format_args!("gemm_bi_splitm_reduce")))?;
        return Ok(());
    }

    if scalar_plan == ScalarDispatchPlan::TnM16N16SplitM16Qualified {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                checked_tile_grid(checked_dims.k_u32, 16, checked_dims.n_u32, 16)?,
                1,
                1,
            ),
            block_dim: (super::contract::SCALAR_TN_M16N16_THREADS, 1, 1),
            shared_mem_bytes: super::contract::SCALAR_TN_M16N16_DYNAMIC_SHARED_BYTES,
        };
        let mut builder = scalar_launch_builder(
            stream,
            &kernels.gemm_bi_tn_m16n16_bk16_s2_splitm16_v1,
            &control,
        );
        builder.arg(&dw_ptr);
        builder.arg_buffer(x_saved);
        builder.arg_buffer(dy);
        builder.arg(&alpha);
        builder.arg(&checked_dims.m_i32);
        builder.arg(&checked_dims.k_i32);
        builder.arg(&checked_dims.n_i32);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1",
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!(
                "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1 backward_dw"
            ))
        })?;
        return Ok(());
    }

    // Split-M TN dispatch: M-axis split for underfilled Big TN grids.
    // CUTLASS parallel-split + deterministic ascending-fc reducer.
    //
    // The immutable device size fixes (m_chunk, f_final) at preparation and
    // capture, while the two-wave target scales the partial grid across SM80+.
    // Run-to-run and eager-vs-graph bits stay exact within that resolved route.
    // Backward_dw is intentionally NOT batch-invariant (sums over M) but
    // for each fixed batch the (m_chunk, f_final) is deterministic.
    if let ScalarDispatchPlan::TnSplitM {
        m_chunk,
        chunks: f_final,
    } = scalar_plan
    {
        let base_blocks = checked_tile_grid(checked_dims.k_u32, 128, checked_dims.n_u32, 128)?;
        let m_i = checked_dims.m_i32;
        let k_i = checked_dims.k_i32;
        let n_i = checked_dims.n_i32;
        let m_chunk_i = checked_i32(m_chunk, "TN split-M chunk")?;
        let f_i = checked_i32(f_final, "TN split-M partitions")?;
        let f_final_u32 = checked_u32(f_final, "TN split-M partitions")?;

        let partial_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };

        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (base_blocks, 1, f_final_u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let partial_symbol = scalar_tn_kernel_symbol(scalar_plan, operands);
        let partial_function = if partial_symbol == "gemm_bi_tn_splitm_partial_aligned" {
            &kernels.gemm_bi_tn_splitm_partial_aligned
        } else {
            &kernels.gemm_bi_tn_splitm_partial
        };
        let mut pb = scalar_launch_builder(stream, partial_function, &control);
        pb.arg(&partial_ptr);
        pb.arg_buffer(x_saved);
        pb.arg_buffer(dy);
        pb.arg(&m_i);
        pb.arg(&k_i);
        pb.arg(&n_i);
        pb.arg(&m_chunk_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            partial_symbol,
            partial_cfg,
            &mut pb,
        )
        .map_err(|error| error.with_driver_context(format_args!("{partial_symbol}")))?;

        let total = checked_dims.kn_u32;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut rb = scalar_launch_builder(stream, &kernels.gemm_bi_splitm_reduce, &control);
        rb.arg(&dw_ptr);
        rb.arg(&partial_ptr);
        rb.arg(&alpha);
        rb.arg(&k_i);
        rb.arg(&n_i);
        rb.arg(&f_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_splitm_reduce",
            reduce_cfg,
            &mut rb,
        )
        .map_err(|error| error.with_driver_context(format_args!("gemm_bi_splitm_reduce")))?;
        return Ok(());
    }

    // Custom: dW[K,N] += X^T[K,M] @ dY[M,N]
    // Envelope: K_out ≥ 1, N ≥ 128. Kernel A-load is scalar per-row (handles non-%4 M),
    // B-load has runtime N%4 scalar fallback. K scalar fallback handles non-%4 K.
    // Dropped `n_in >= 128` — kernel grid handles K_out<128 correctly;
    // covers Mamba-1 dt_proj backward (K_out=8).
    if let ScalarDispatchPlan::TnFinal { slim } = scalar_plan {
        let m_i = checked_dims.m_i32;
        let k_i = checked_dims.k_i32;
        let n_i = checked_dims.n_i32;
        let symbol = scalar_tn_kernel_symbol(scalar_plan, operands);
        let (func, bn) = match symbol {
            "gemm_bi_tn_slim" => (&kernels.gemm_bi_tn_slim, 64),
            "gemm_bi_tn_aligned" => (&kernels.gemm_bi_tn_aligned, 128),
            "gemm_bi_tn" => (&kernels.gemm_bi_tn, 128),
            _ => unreachable!("unexpected Big TN symbol"),
        };
        // Opt1: Big uses 256 threads/block; Slim stays 128.
        let threads = if slim { 128u32 } else { 256u32 };
        // Big TN uses dynamic smem for 2-stage cp.async; Slim stays static.
        let smem_bytes: u32 = if slim { 0 } else { 34 * 1024 };
        // The data-parallel grid launches one CTA per output tile.
        let total_tiles = checked_tile_grid(checked_dims.k_u32, 128, checked_dims.n_u32, bn)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let mut builder = scalar_launch_builder(stream, func, &control);
        builder.arg(&dw_ptr);
        builder.arg_buffer(x_saved);
        builder.arg_buffer(dy);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&n_i);
        enqueue_scalar_backward(&mut control, request, operands, symbol, cfg, &mut builder)
            .map_err(|error| {
                error.with_driver_context(format_args!(
                    "gemm_bi_tn{} backward_dw",
                    if slim { "_slim" } else { "" }
                ))
            })?;
        return Ok(());
    }

    // The zero-cuBLAS contract has no fallback beyond this point.
    panic!(
        "gpu_gemm_bi_backward_dw: cuBLAS fallback hit (shape M={batch} K={n_in} N={n_out}). \
         The zero-cuBLAS contract requires every shape to route through a custom \
         kernel — add a dispatcher branch in this function for this shape."
    );
}

/// Input gradient: `dX[B,K] = dY[B,N] @ W^T[N,K]` (overwritten, beta=0.0).
///
/// cuBLAS: `dX^T[K,B] = W[K,N] @ dY^T[N,B]`
/// But we want dX row-major, so:
/// `dX^T[K,B] = W[K,N](as col-major=W^T[N,K]) @ dY^T[N,B]`
///
/// Actually, row-major trick:
/// For C = A @ B^T in row-major:
/// C^T = B @ A^T in col-major
/// gemm(T, N, K, B, N, 1.0, W, N, dY, N, 0.0, dX, K)
pub fn gemm_bi_backward_dx(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
    w_ptr: CUptr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    gemm_bi_backward_dx_with_control(
        stream,
        kernels,
        dx,
        dy,
        w_ptr,
        dims,
        None::<&mut ScalarLaunchControl<'_>>,
    )
}

fn gemm_bi_backward_dx_with_control<C: ScalarLaunchController>(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
    w_ptr: CUptr,
    dims: (usize, usize, usize),
    mut control: Option<&mut C>,
) -> Result<(), String> {
    let (checked_dims, request) = scalar_backward_request(ResolvedGemmOp::Nt, dims)?;
    let (batch, n_in, n_out) = checked_dims.tuple();
    let alpha = control
        .as_ref()
        .map(|control| control.operands().alpha)
        .unwrap_or(1.0);
    let operands = F32TriadOperands {
        output: dx.raw_ptr(stream),
        a: dy.raw_ptr(stream),
        b: w_ptr,
        bias: None,
        alpha,
        beta: 0.0,
    };
    let scalar_plan = match control.as_ref() {
        Some(prepared) => prepared.plan(),
        None => scalar_launch_plan(scalar_launch_facts(kernels), request, operands)?,
    };
    if let Some(control) = control.as_ref() {
        let prepared = control.operands();
        if prepared.output != operands.output
            || prepared.a != operands.a
            || prepared.b != operands.b
            || prepared.bias != operands.bias
            || prepared.alpha.to_bits() != operands.alpha.to_bits()
            || prepared.beta.to_bits() != operands.beta.to_bits()
        {
            return Err("prepared NT operands differ from the physical launch arguments".into());
        }
    }
    if scalar_plan == ScalarDispatchPlan::NtM2N16SplitK32Qualified {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (256, 1, 1),
            block_dim: (super::contract::SCALAR_NT_M2N16_THREADS, 1, 1),
            shared_mem_bytes: super::contract::SCALAR_NT_M2N16_DYNAMIC_SHARED_BYTES,
        };
        let dy_ptr = dy.raw_ptr(stream);
        let mut builder =
            scalar_launch_builder(stream, &kernels.gemm_bi_nt_m2n16_bk64_splitk32_v1, &control);
        builder.arg_buffer_mut(dx);
        builder.arg(&dy_ptr);
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&checked_dims.m_i32);
        builder.arg(&checked_dims.n_i32);
        builder.arg(&checked_dims.k_i32);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_nt_m2n16_bk64_splitk32_v1",
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!(
                "gemm_bi_nt_m2n16_bk64_splitk32_v1 backward_dx"
            ))
        })?;
        return Ok(());
    }
    if matches!(
        scalar_plan,
        ScalarDispatchPlan::NtD768TransposeM64N64Qualified
            | ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
            | ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
            | ScalarDispatchPlan::NtPrismVectorQualified
            | ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified
    ) {
        use cudarc::driver::DevicePtr;
        let transpose_scratch = kernels.transpose_scratch_buf(stream)?;
        let required =
            scalar_transpose_scratch_elements(request, scalar_plan)?.ok_or_else(|| {
                "qualified scalar NT transpose plan lost its scratch extent".to_string()
            })?;
        if transpose_scratch.len() != super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS
            || required > transpose_scratch.len()
        {
            return Err(format!(
                "qualified scalar NT transpose scratch has {} f32 elements, requires {required} with exact capacity {}",
                transpose_scratch.len(),
                super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS
            ));
        }
        let (w_t_ptr, _) = transpose_scratch.device_ptr(stream);
        let rows = checked_dims.k_i32;
        let columns = checked_dims.n_i32;
        let transpose_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                checked_dims.n_u32.div_ceil(32),
                checked_dims.k_u32.div_ceil(32),
                1,
            ),
            block_dim: (32, 16, 1),
            shared_mem_bytes: 0,
        };
        let mut transpose = scalar_launch_builder(
            stream,
            &kernels.gemm_bi_transpose_f32_32x16_d768_v1,
            &control,
        );
        transpose.arg(&w_t_ptr);
        transpose.arg(&w_ptr);
        transpose.arg(&rows);
        transpose.arg(&columns);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_transpose_f32_32x16_d768_v1",
            transpose_cfg,
            &mut transpose,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!(
                "gemm_bi_transpose_f32_32x16_d768_v1 backward_dx"
            ))
        })?;

        let params = SgbNnM64N64Params {
            alpha,
            beta: 0.0,
            m: checked_dims.m_i32,
            n: checked_dims.k_i32,
            k: checked_dims.n_i32,
            lda: checked_dims.n_i32,
            ldb: checked_dims.k_i32,
            ldc: checked_dims.k_i32,
        };
        let m64_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                checked_grid_product(
                    checked_dims.m_u32.div_ceil(64),
                    checked_dims.k_u32.div_ceil(64),
                    1,
                )?,
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: super::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES,
        };
        let bias = 0_u64;
        let dy_ptr = dy.raw_ptr(stream);
        let (m64_function, m64_symbol) =
            if scalar_plan == ScalarDispatchPlan::NtPrismVectorQualified {
                (
                    &kernels.gemm_bi_nn_prism_m64n64_bk16_s2_v1,
                    "gemm_bi_nn_prism_m64n64_bk16_s2_v1",
                )
            } else {
                (
                    &kernels.gemm_bi_nn_m64n64_bk16_s2_v1,
                    "gemm_bi_nn_m64n64_bk16_s2_v1",
                )
            };
        let mut m64 = scalar_launch_builder(stream, m64_function, &control);
        m64.arg_buffer_mut(dx);
        m64.arg(&dy_ptr);
        m64.arg(&w_t_ptr);
        m64.arg(&bias);
        m64.arg(&params);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            m64_symbol,
            m64_cfg,
            &mut m64,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!(
                "{m64_symbol} qualified NT transpose backward_dx"
            ))
        })?;
        return Ok(());
    }
    // Narrow-N NT dispatch: N∈[2..127], including the N∈[49..127] gap
    // where slim/big kernels (N>=128) don't apply.
    // The gate starts at N=2; N=1 is handled by the column-GEMV route.
    // Kernel has `if (g_row >= M) continue;` M-predication → safe for any batch.
    // Relaxed to n_in>=1, batch>=1 covers test-config (M=32, K=32..64, N=32..64)
    // that otherwise falls to cuBLAS (zero-cuBLAS contract violation).
    if matches!(scalar_plan, ScalarDispatchPlan::NtNarrow) {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_i = checked_dims.k_i32;
        let num_pid_m = checked_dims.m_u32.div_ceil(64);
        let num_pid_n = checked_dims.k_u32.div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_grid_product(num_pid_m, num_pid_n, 1)?, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_nt_narrow, &control);
        builder.arg_buffer_mut(dx);
        builder.arg_buffer(dy);
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_nt_narrow",
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_nt_narrow backward_dx"))
        })?;
        return Ok(());
    }

    // Small-batch wide-N NT dispatch.
    // Gap: batch ∈ [1, 31], N >= 128 — Narrow NT capped at N=127, Split-K
    // NT-via-T requires batch >= 32, Big/Slim NT requires batch >= 128.
    // Solution: reuse gemm_bi_nt_narrow kernel — N is reduction-axis, kernel
    // iterates `for nIdx in [0, N) by NBK=16` in scalar.cu, with no upper
    // bound on N. Tile dims (BM=64, BN=32) fit any small batch; M/K_out
    // predication inside kernel handles partial last block.
    // Determinism: kernel unchanged → bit-exact with the N<=127 path.
    // Production unaffected: training uses batch=128 (Big/Slim path).
    // Closes test_gpu_correctness M=4 K=32 N=128 cuBLAS-fallback panic.
    if matches!(scalar_plan, ScalarDispatchPlan::NtSmallBatchWide) {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_i = checked_dims.k_i32;
        let num_pid_m = checked_dims.m_u32.div_ceil(64);
        let num_pid_n = checked_dims.k_u32.div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_grid_product(num_pid_m, num_pid_n, 1)?, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_nt_narrow, &control);
        builder.arg_buffer_mut(dx);
        builder.arg_buffer(dy);
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_nt_narrow",
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!(
                "gemm_bi_nt_narrow (small-batch wide-N) backward_dx"
            ))
        })?;
        return Ok(());
    }

    // GEMV-N1 NT dispatch: dX[M,K] = dY[M,1] @ W^T[1,K] (outer product)
    // batch lower bound relaxed 4 → 1.
    // Kernel gemm_bi_nt_gemv computes per-element dX[m,k] = alpha*dY[m]*W[k]
    // with total = M*K threads and `if (tid >= total) return;` predication
    // in scalar.cu — safe for M<4. Closes the single-env eval gap.
    if matches!(scalar_plan, ScalarDispatchPlan::NtGemv) {
        let m_i = checked_dims.m_i32;
        let k_i = checked_dims.k_i32;
        let ldx_i = checked_dims.k_i32;
        let ldy_i: i32 = 1;
        let total = checked_dims.mk_u32;
        let block = 256u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_nt_gemv, &control);
        builder.arg_buffer_mut(dx);
        builder.arg_buffer(dy);
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&ldx_i);
        builder.arg(&ldy_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_nt_gemv",
            cfg,
            &mut builder,
        )
        .map_err(|error| error.with_driver_context(format_args!("gemm_bi_nt_gemv backward_dx")))?;
        return Ok(());
    }

    // Split-K NT-via-transpose + K-tail for M<128 shapes with K_out%32 != 0.
    // Covers SALE action backward_dx (K_out=257, tail=1), SimbaV2 hyperplane
    // backward_dx (K_out=642, tail=2), and any K_out%32 ∈ {1..31}: main is the
    // first K_out - (K_out%32) rows (multiple of 32 → %4 safe for vectorized
    // stores), tail is K_out%32 columns filled via sequential dx_col_gemv calls.
    // Same transpose_scratch (4,718,592 f32 = 18 MiB) + splitk_scratch
    // (8 M f32) as the main
    // NT-via-T path below, so envelope caps match: K_out ≤ 4096, N ≤ 2048.
    //
    // Phase C-1.5bo: Slim NT-via-T underfill guard. Slim NT tile BM=128, BN=64
    // along K_out (= n_in here — backward dx output column axis). At one live
    // wave Slim NT already saturates; Split-K transpose + partial + reducer adds
    // DRAM round-trips for no occupancy benefit. This matches the forward NN
    // guard because both use the same Slim BM=128 geometry.
    if let ScalarDispatchPlan::NtSplitKTail {
        k_main,
        k_tail: k_tail_cnt,
    } = scalar_plan
    {
        // The W cap matches transpose_scratch and the partial cap matches splitk_scratch.
        // (the GPU splitk_scratch capacity). Earlier hardcoded `1<<23` partial
        // cap was tighter than the underlying scratch (1<<23) and caused k_tail
        // to fall through at batch=1024 (partial=10.5M > 8M cap) while CPU has
        // no cap caused catastrophic dispatch divergence for a large-batch
        // K-tail dX shape (max_ulp=2.1M on synthetic LCG). Lifting matches the
        // actual scratch sizes — bit-exact + no perf regression (k_tail is the
        // optimal path; the previous cap unnecessarily routed to slower default).
        // Step 1: transpose W[0..k_main, :] → W_T[N, k_main] into scratch.
        let rows_i = checked_i32(k_main, "NT K-tail rows")?;
        let cols_i = checked_dims.n_i32;
        let t_grid_x = checked_dims.n_u32.div_ceil(32);
        let t_grid_y = checked_u32(k_main, "NT K-tail rows")?.div_ceil(32);
        let t_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (t_grid_x, t_grid_y, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        };
        let w_t_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.transpose_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };
        let mut tb = scalar_launch_builder(stream, &kernels.gemm_bi_transpose_f32_2d, &control);
        tb.arg(&w_t_ptr);
        tb.arg(&w_ptr);
        tb.arg(&rows_i);
        tb.arg(&cols_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_transpose_f32_2d",
            t_cfg,
            &mut tb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_transpose_f32_2d (K-tail)"))
        })?;

        // Step 2: Split-K NN partial — A=dY, B=W_T, output [M, k_main].
        let m_i = checked_dims.m_i32;
        let k_main_i = checked_i32(k_main, "NT K-tail columns")?;
        let k_chunks = checked_i32(n_out / 32, "NT K-tail chunks")?;
        let lda_dy_i = checked_dims.n_i32;
        let num_pid_m = checked_dims.m_u32.div_ceil(32);
        let num_pid_n = checked_u32(k_main, "NT K-tail columns")?.div_ceil(64);
        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                checked_grid_product(
                    num_pid_m,
                    num_pid_n,
                    checked_u32(n_out / 32, "NT K-tail chunks")?,
                )?,
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let partial_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };
        let mut pb = scalar_launch_builder(stream, &kernels.gemm_bi_nn_splitk32_partial, &control);
        pb.arg(&partial_ptr);
        pb.arg_buffer(dy);
        pb.arg(&w_t_ptr);
        pb.arg(&m_i);
        pb.arg(&k_main_i);
        pb.arg(&k_chunks);
        pb.arg(&lda_dy_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_nn_splitk32_partial",
            partial_cfg,
            &mut pb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_nn_splitk32_partial (NT K-tail main)"))
        })?;

        // Step 3: reducer writes dX[:, 0..k_main] with stride n_in.
        let null_tail: u64 = 0;
        let null_bias: u64 = 0;
        let zero_i32: i32 = 0;
        let out_stride_i = checked_dims.k_i32;
        let total_main = checked_u32(
            batch
                .checked_mul(k_main)
                .ok_or_else(|| invalid_gemm_dimensions("NT K-tail output total overflows usize"))?,
            "NT K-tail output total",
        )?;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_main.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut rb = scalar_launch_builder(stream, &kernels.gemm_bi_splitk_reduce, &control);
        rb.arg_buffer_mut(dx);
        rb.arg(&partial_ptr);
        rb.arg(&null_bias);
        rb.arg(&null_tail);
        rb.arg(&null_tail);
        rb.arg(&alpha);
        rb.arg(&m_i);
        rb.arg(&k_main_i);
        rb.arg(&k_chunks);
        rb.arg(&zero_i32);
        rb.arg(&out_stride_i); // dX row stride = n_in (K_out full)
        rb.arg(&zero_i32); // tail_cnt = 0 (tail handled by separate gemv)
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_splitk_reduce",
            reduce_cfg,
            &mut rb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_splitk_reduce (NT K-tail main)"))
        })?;

        // Step 4: loop over tail columns. For each k in [0, k_tail_cnt):
        // dX[:, k_main + k] = Σ_n dY[m, n] · W[k_main + k, n]. Each call is
        // one gemv; tail_cnt ≤ 31 so total overhead is bounded. Sequential
        // (not parallel) to keep kernel launches small and deterministic.
        let w_base_ptr = w_ptr;
        let n_i = checked_dims.n_i32;
        let block = 128u32;
        let tail_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_dims.m_u32.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        for k in 0..k_tail_cnt {
            let k_tail_col = k_main + k;
            let row_elements = k_tail_col
                .checked_mul(n_out)
                .ok_or_else(|| invalid_gemm_dimensions("W tail row offset overflows usize"))?;
            let w_tail_row_ptr = checked_ptr_add(
                w_base_ptr,
                checked_byte_offset(row_elements, std::mem::size_of::<f32>(), "W tail row")?,
                "W tail row",
            )?;
            let col_idx_i = checked_i32(k_tail_col, "NT K-tail column")?;
            let mut gb = scalar_launch_builder(stream, &kernels.gemm_bi_dx_col_gemv, &control);
            gb.arg_buffer_mut(dx);
            gb.arg_buffer(dy);
            gb.arg(&w_tail_row_ptr);
            gb.arg(&m_i);
            gb.arg(&n_i);
            gb.arg(&col_idx_i);
            gb.arg(&out_stride_i);
            enqueue_scalar_backward(
                &mut control,
                request,
                operands,
                "gemm_bi_dx_col_gemv",
                tail_cfg,
                &mut gb,
            )
            .map_err(|error| {
                error.with_driver_context(format_args!("gemm_bi_dx_col_gemv (NT K-tail col={})", k))
            })?;
        }
        return Ok(());
    }

    // Split-K NT-via-transpose dispatch for underfilled backward-dX projections.
    // Strategy: transpose W[K_out, N] → W_T[N, K_out], then dX = dY @ W_T via the
    // existing NN Split-K kernel. This route is faster than dedicated NT for
    // the underfilled thin shapes it admits.
    //
    // A.2 — generalised to support n_out%32 != 0 by folding the N-tail (residue
    // after the largest 32-aligned prefix) into the reducer's `tail_cnt` arg.
    // The reducer in scalar.cu already supports tail folding: for each
    // (m, n) cell it appends `Σ_{k<tail_cnt} x_tail[m,k] * w_tail[k,n]` after
    // the K_CHUNKS partial reduce. For NT-via-T post-transpose the tail is along
    // the reduction axis (= original n_out), so:
    //   x_tail_ptr     = dY[:, n_main]              (stride n_out, full dY width)
    //   w_tail_ptr     = W_T[n_main, :]             (stride n_in)
    //   x_tail_stride  = n_out
    //   tail_cnt       = n_out % 32
    // For n_out%32==0 the tail is empty (tail_cnt=0) and behaviour matches the
    // pre-A.2 main path bit-exactly. For n_out%32 != 0 (e.g. production hit
    // M=36 K=128 N=796 → tail=28) the formerly-uncovered shape now lands here
    // with full custom-kernel coverage and no cuBLAS fallback.
    //
    // General envelope: M ∈ [32, 1024], K_out ∈ [64, 4096], K_out % 4 == 0,
    // N ∈ [32, 2048], n_in % 32 == 0 (the K-tail bwd_dx route above covers
    // n_in%32 != 0 separately; combined K-tail + N-tail is rare and falls
    // through to the ordinary scalar route).
    // The selector also admits the measured (512, 16, 2048) and
    // (16, 512, 2048) cells as exact exceptions. They do not broaden this
    // envelope to neighboring M or K_out values.
    if let ScalarDispatchPlan::NtSplitKMain {
        n_main: n_main_nt,
        n_tail: n_tail_nt,
    } = scalar_plan
    {
        // Step 1: transpose full W[n_in=K_out, n_out=N] → W_T[N, K_out] into
        // scratch (full width, including the tail rows W_T[n_main..n_out, :]).
        let rows_i = checked_dims.k_i32;
        let cols_i = checked_dims.n_i32;
        let t_grid_x = checked_dims.n_u32.div_ceil(32);
        let t_grid_y = checked_dims.k_u32.div_ceil(32);
        let t_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (t_grid_x, t_grid_y, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        };
        let w_t_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.transpose_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };
        let mut tb = scalar_launch_builder(stream, &kernels.gemm_bi_transpose_f32_2d, &control);
        tb.arg(&w_t_ptr);
        tb.arg(&w_ptr);
        tb.arg(&rows_i);
        tb.arg(&cols_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_transpose_f32_2d",
            t_cfg,
            &mut tb,
        )
        .map_err(|error| error.with_driver_context(format_args!("gemm_bi_transpose_f32_2d")))?;

        // Step 2: NN Split-K partial on the n_main (32-aligned) prefix.
        // partial = dY[M, n_main] @ W_T[n_main, K_out], reduction over n_main.
        // lda_i = n_out (full dY row stride) — partial reads only the first
        // k_chunks*32 = n_main columns per row, leaving the tail for step 3.
        let m_i = checked_dims.m_i32;
        let k_out_i = checked_dims.k_i32;
        let k_chunks = checked_i32(n_main_nt / 32, "NT split-K chunks")?;

        let num_pid_m = checked_dims.m_u32.div_ceil(32);
        let num_pid_n = checked_dims.k_u32.div_ceil(64);
        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                checked_grid_product(
                    num_pid_m,
                    num_pid_n,
                    checked_u32(n_main_nt / 32, "NT split-K chunks")?,
                )?,
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let partial_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };
        let lda_i = checked_dims.n_i32; // dY row stride = N_full (NOT n_main)
        let mut pb = scalar_launch_builder(stream, &kernels.gemm_bi_nn_splitk32_partial, &control);
        pb.arg(&partial_ptr);
        pb.arg_buffer(dy);
        pb.arg(&w_t_ptr);
        pb.arg(&m_i);
        pb.arg(&k_out_i);
        pb.arg(&k_chunks);
        pb.arg(&lda_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_nn_splitk32_partial",
            partial_cfg,
            &mut pb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!(
                "gemm_bi_nn_splitk32_partial (NT-via-T N-tail)"
            ))
        })?;

        // Step 3: reducer with N-tail fold. Computes
        //   dX[m,k] = Σ_{c<k_chunks} partial[c][m,k]               (chunk sum, ascending c)
        //          + Σ_{i<tail_cnt} dY[m, n_main+i] · W_T[n_main+i, k]   (tail, ascending i)
        // Each BK32 partial has its own rounded accumulator; the reducer then
        // adds those partials and tail FMAs in ascending order. This is a
        // distinct frozen deterministic scalar family, not the CPU sequential
        // accumulation family.
        let null_bias: u64 = 0;
        let total = checked_dims.mk_u32;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let zero_i32: i32 = 0;
        let tail_cnt_i = checked_i32(n_tail_nt, "NT reduction tail")?;
        let dy_tail_stride_i = checked_dims.n_i32; // dY row stride
        // x_tail_ptr = dY[:, n_main] (offset n_main floats into base).
        // w_tail_ptr = W_T[n_main, :] (offset n_main * n_in floats into W_T base).
        let (dy_tail_ptr, wt_tail_ptr): (u64, u64) = if n_tail_nt > 0 {
            use cudarc::driver::DevicePtr;
            let (dy_base, _r_dy) = dy.inner().device_ptr(stream);
            let dyp = checked_ptr_add(
                dy_base,
                checked_byte_offset(n_main_nt, std::mem::size_of::<f32>(), "dY tail")?,
                "dY tail",
            )?;
            let wt_elements = n_main_nt.checked_mul(n_in).ok_or_else(|| {
                invalid_gemm_dimensions("transposed W tail offset overflows usize")
            })?;
            let wtp = checked_ptr_add(
                w_t_ptr,
                checked_byte_offset(wt_elements, std::mem::size_of::<f32>(), "transposed W tail")?,
                "transposed W tail",
            )?;
            (dyp, wtp)
        } else {
            (0, 0)
        };
        let mut rb = scalar_launch_builder(stream, &kernels.gemm_bi_splitk_reduce, &control);
        rb.arg_buffer_mut(dx);
        rb.arg(&partial_ptr);
        rb.arg(&null_bias);
        rb.arg(&dy_tail_ptr);
        rb.arg(&wt_tail_ptr);
        rb.arg(&alpha);
        rb.arg(&m_i);
        rb.arg(&k_out_i);
        rb.arg(&k_chunks);
        rb.arg(&dy_tail_stride_i);
        rb.arg(&zero_i32); // out_col_stride default = N (= K_out = n_in)
        rb.arg(&tail_cnt_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_splitk_reduce",
            reduce_cfg,
            &mut rb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_splitk_reduce (NT-via-T N-tail)"))
        })?;
        return Ok(());
    }

    // Split-K Slim NN via transpose for fat-M bwd_dx shapes
    // (M > 1024). Mirrors the M<128 NT-via-T above but uses Slim Split-K partial
    // (BM=128 BN=64) for better arithmetic intensity on fat-M Mamba shapes.
    //
    // Transformation: dX[M, K_out] = dY[M, N] @ W^T[N, K_out]
    //   After transposing W[K_out, N] → W_T[N, K_out], becomes NN:
    //   dX[M, K_out] = dY[M, N] @ W_T[N, K_out]
    //   Kernel params: M=batch, N=K_out (n_in), K=N (n_out, reduction axis).
    //
    // Batch-invariance: K_CHUNK = compile-time constant → F = ceil(N / K_CHUNK)
    // is a pure function of N. Same N always produces same F for any batch.
    //
    // Fires AFTER M<=1024 NT-via-T gate so never steals shapes Thin-M handles.
    const SLIM_NT_K_CHUNK: u32 = 64;
    if let ScalarDispatchPlan::NtSplitKSlim { chunks: f_final } = scalar_plan {
        let base_blocks = checked_grid_product(
            checked_dims.m_u32.div_ceil(128),
            checked_dims.k_u32.div_ceil(64),
            1,
        )?;
        let k_chunk = SLIM_NT_K_CHUNK;
        // Step 1: transpose W[n_in=K_out, n_out=N] → W_T[N, K_out] into scratch.
        let rows_i = checked_dims.k_i32;
        let cols_i = checked_dims.n_i32;
        let t_grid_x = checked_dims.n_u32.div_ceil(32);
        let t_grid_y = checked_dims.k_u32.div_ceil(32);
        let t_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (t_grid_x, t_grid_y, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        };
        let w_t_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.transpose_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };
        let mut tb = scalar_launch_builder(stream, &kernels.gemm_bi_transpose_f32_2d, &control);
        tb.arg(&w_t_ptr);
        tb.arg(&w_ptr);
        tb.arg(&rows_i);
        tb.arg(&cols_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_transpose_f32_2d",
            t_cfg,
            &mut tb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_transpose_f32_2d (slim NT)"))
        })?;

        // Step 2: Slim Split-K NN partial on (dY, W_T) with K_chunk split.
        let m_i = checked_dims.m_i32;
        let k_out_i = checked_dims.k_i32; // NN's "N" = K_out
        let k_full_i = checked_dims.n_i32; // NN's "K" = n_out (reduction axis)
        let lda_i = checked_dims.n_i32; // dY stride = n_out
        let ldb_i = checked_dims.k_i32; // W_T stride = K_out
        let k_chunk_i = checked_i32(
            checked_usize(k_chunk, "NT slim split-K chunk")?,
            "NT slim split-K chunk",
        )?;

        let partial_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch_buf(stream)?.device_ptr(stream);
            ptr
        };

        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (base_blocks, 1, f_final),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut pb =
            scalar_launch_builder(stream, &kernels.gemm_bi_nn_splitk_slim_partial, &control);
        pb.arg(&partial_ptr);
        pb.arg_buffer(dy);
        pb.arg(&w_t_ptr);
        pb.arg(&m_i);
        pb.arg(&k_out_i);
        pb.arg(&k_full_i);
        pb.arg(&lda_i);
        pb.arg(&ldb_i);
        pb.arg(&k_chunk_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_nn_splitk_slim_partial",
            partial_cfg,
            &mut pb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_nn_splitk_slim_partial (slim NT)"))
        })?;

        // Step 3: reducer writes dX (beta=0, no bias, alpha=1).
        let null_bias: u64 = 0;
        let null_tail: u64 = 0;
        let zero_i32_nt: i32 = 0;
        let f_i = checked_i32(
            checked_usize(f_final, "NT slim split-K chunks")?,
            "NT slim split-K chunks",
        )?;
        let total = checked_dims.mk_u32;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut rb = scalar_launch_builder(stream, &kernels.gemm_bi_splitk_reduce, &control);
        rb.arg_buffer_mut(dx);
        rb.arg(&partial_ptr);
        rb.arg(&null_bias);
        rb.arg(&null_tail);
        rb.arg(&null_tail);
        rb.arg(&alpha);
        rb.arg(&m_i);
        rb.arg(&k_out_i);
        rb.arg(&f_i);
        rb.arg(&zero_i32_nt);
        rb.arg(&zero_i32_nt);
        rb.arg(&zero_i32_nt);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_splitk_reduce",
            reduce_cfg,
            &mut rb,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!("gemm_bi_splitk_reduce (slim NT)"))
        })?;
        return Ok(());
    }

    // ===== Gap-fill: thin-batch wide-N shapes not caught by specialized branches =====
    // Closes dispatcher gap at (batch ∈ [32..128), N >= 128) that:
    //   - Narrow NT (line ~932) caps at N=127
    //   - Small-batch wide-N (line ~967) caps at batch < 32
    //   - Split-K NT-via-T requires N % 32 == 0 (n_out=194 with %32=2 falls)
    //   - Big NT requires batch >= 128
    // Order: AFTER all splitk attempts (so it never steals their coverage),
    // BEFORE big-NT. Re-uses `gemm_bi_nt_narrow` kernel (BM=64, BN=32 along K_out,
    // N as reduction axis with `nIdx in [0,N) by NBK=16` — no upper bound on N).
    //
    // Determinism: per-output ascending-N FMA chain — bit-identical to CPU
    // mirror `narrow_nt_sgemm_nt` regardless of tile grid. Same kernel as the
    // small-batch-<32 branch above, so byte-identical FMA path.
    //
    // Perf: ~50% tile fill at boundary (batch padded to BM=64) — acceptable
    // for a gap-fill vs cuBLAS panic / non-determinism.
    if matches!(scalar_plan, ScalarDispatchPlan::NtMidBatchWide) {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_i = checked_dims.k_i32;
        let num_pid_m = checked_dims.m_u32.div_ceil(64);
        let num_pid_n = checked_dims.k_u32.div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_grid_product(num_pid_m, num_pid_n, 1)?, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = scalar_launch_builder(stream, &kernels.gemm_bi_nt_narrow, &control);
        builder.arg_buffer_mut(dx);
        builder.arg_buffer(dy);
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        enqueue_scalar_backward(
            &mut control,
            request,
            operands,
            "gemm_bi_nt_narrow",
            cfg,
            &mut builder,
        )
        .map_err(|error| {
            error.with_driver_context(format_args!(
                "gemm_bi_nt_narrow (gap-fill mid-batch wide-N)"
            ))
        })?;
        return Ok(());
    }

    // Custom: dX[M,K] = dY[M,N] @ W^T[N,K]
    // Envelope: M ≥ 128, K_out ≥ 1. Kernel has scalar N-fallback for non-%4 N,
    // scalar K-fallback for non-%4 K_out.
    // Dropped `n_in >= 128` — covers Mamba-1 dt_proj backward_dx (K_out=8).
    if let ScalarDispatchPlan::NtFinal { slim } = scalar_plan {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_i = checked_dims.k_i32;
        let (func, bn) = if slim {
            (&kernels.gemm_bi_nt_slim, 64)
        } else {
            (&kernels.gemm_bi_nt, 128)
        };
        // Opt1: Big uses 256 threads/block; Slim stays 128.
        let threads = if slim { 128u32 } else { 256u32 };
        // Big NT keeps two A stages while B moves through raw and compute layouts.
        let smem_bytes = if slim {
            0
        } else {
            SCALAR_BIG_NT_DYNAMIC_SHARED_BYTES
        };
        // The data-parallel grid launches one CTA per output tile.
        let total_tiles = checked_tile_grid(checked_dims.m_u32, 128, checked_dims.k_u32, bn)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let mut builder = scalar_launch_builder(stream, func, &control);
        builder.arg_buffer_mut(dx);
        builder.arg_buffer(dy);
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        let symbol = if slim {
            "gemm_bi_nt_slim"
        } else {
            "gemm_bi_nt"
        };
        enqueue_scalar_backward(&mut control, request, operands, symbol, cfg, &mut builder)
            .map_err(|error| {
                error.with_driver_context(format_args!(
                    "gemm_bi_nt{} backward_dx",
                    if slim { "_slim" } else { "" }
                ))
            })?;
        return Ok(());
    }

    // The zero-cuBLAS contract has no fallback beyond this point.
    panic!(
        "gpu_gemm_bi_backward_dx: cuBLAS fallback hit (shape M={batch} K={n_in} N={n_out}). \
         The zero-cuBLAS contract requires every shape to route through a custom \
         kernel — add a dispatcher branch in this function for this shape."
    );
}

// ============================================================================
// Typed (bf16/f16) dispatch — typed sync-load buckets.
// ============================================================================
// Same bucket geometry and launch configs as the f32 dispatcher above; the
// typed kernels are bit-identical to "upcast inputs to f32, run the f32
// kernel". Buckets not covered by native typed kernels return Err:
// callers must not silently fall back to non-deterministic cuBLAS.

use super::super::blas::{HalfPhysicalTraceRequest, TypedPtr};
use super::super::dtype::WeightDtype;

/// Typed launchers accept only the homogeneous bf16/f16 contracts.
fn require_half(dt: WeightDtype, what: &str) -> Result<(), String> {
    if dt == WeightDtype::F32 {
        return Err(format!(
            "gemm_bi typed dispatch: {what} is f32 — use the f32 entry points"
        ));
    }
    Ok(())
}

/// How a half kernel reduces one output tile: the tiled kernels keep one
/// owner CTA and one reduction order per tile; the stream-K kernel folds
/// per-CTA partial slabs in a fixed order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HalfSchedule {
    Tiled,
    StreamKFixedOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HalfKernelIdentity {
    symbol: &'static str,
    module_kind: ModuleKind,
    schedule: HalfSchedule,
}

impl HalfKernelIdentity {
    fn resolve(base: &str, dtype: WeightDtype) -> Result<Self, String> {
        let module_kind = if matches!(
            base,
            "gemm_bi_nn_tc"
                | "gemm_bi_nn_tc64"
                | "gemm_bi_nn_tc16"
                | "gemm_bi_tn_tc"
                | "gemm_bi_tn_tc64"
                | "gemm_bi_tn_tc64_streamk"
                | "gemm_bi_tn_tc128x64"
                | "gemm_bi_nt_tc"
                | "gemm_bi_nt_tc64"
        ) {
            ModuleKind::TriadSm80
        } else {
            ModuleKind::TriadScalar
        };
        let schedule = if base == "gemm_bi_tn_tc64_streamk" {
            HalfSchedule::StreamKFixedOrder
        } else {
            HalfSchedule::Tiled
        };
        let symbol = match (base, dtype) {
            ("gemm_bi_nn_gemv", WeightDtype::Bf16) => "gemm_bi_nn_gemv_bf16",
            ("gemm_bi_nn_gemv", WeightDtype::F16) => "gemm_bi_nn_gemv_f16",
            ("gemm_bi_nn_ultra_thin", WeightDtype::Bf16) => "gemm_bi_nn_ultra_thin_bf16",
            ("gemm_bi_nn_ultra_thin", WeightDtype::F16) => "gemm_bi_nn_ultra_thin_f16",
            ("gemm_bi_nn_narrow", WeightDtype::Bf16) => "gemm_bi_nn_narrow_bf16",
            ("gemm_bi_nn_narrow", WeightDtype::F16) => "gemm_bi_nn_narrow_f16",
            ("gemm_bi_nn_narrow_small", WeightDtype::Bf16) => "gemm_bi_nn_narrow_small_bf16",
            ("gemm_bi_nn_narrow_small", WeightDtype::F16) => "gemm_bi_nn_narrow_small_f16",
            ("gemm_bi_nn_big", WeightDtype::Bf16) => "gemm_bi_nn_big_bf16",
            ("gemm_bi_nn_big", WeightDtype::F16) => "gemm_bi_nn_big_f16",
            ("gemm_bi_tn_gemv", WeightDtype::Bf16) => "gemm_bi_tn_gemv_bf16",
            ("gemm_bi_tn_gemv", WeightDtype::F16) => "gemm_bi_tn_gemv_f16",
            ("gemm_bi_tn_narrow", WeightDtype::Bf16) => "gemm_bi_tn_narrow_bf16",
            ("gemm_bi_tn_narrow", WeightDtype::F16) => "gemm_bi_tn_narrow_f16",
            ("gemm_bi_tn_big", WeightDtype::Bf16) => "gemm_bi_tn_big_bf16",
            ("gemm_bi_tn_big", WeightDtype::F16) => "gemm_bi_tn_big_f16",
            ("gemm_bi_nt_gemv", WeightDtype::Bf16) => "gemm_bi_nt_gemv_bf16",
            ("gemm_bi_nt_gemv", WeightDtype::F16) => "gemm_bi_nt_gemv_f16",
            ("gemm_bi_nt_narrow", WeightDtype::Bf16) => "gemm_bi_nt_narrow_bf16",
            ("gemm_bi_nt_narrow", WeightDtype::F16) => "gemm_bi_nt_narrow_f16",
            ("gemm_bi_nt_big", WeightDtype::Bf16) => "gemm_bi_nt_big_bf16",
            ("gemm_bi_nt_big", WeightDtype::F16) => "gemm_bi_nt_big_f16",
            ("gemm_bi_nn_tc", WeightDtype::Bf16) => "gemm_bi_nn_tc_bf16",
            ("gemm_bi_nn_tc", WeightDtype::F16) => "gemm_bi_nn_tc_f16",
            ("gemm_bi_nn_tc64", WeightDtype::Bf16) => "gemm_bi_nn_tc64_bf16",
            ("gemm_bi_nn_tc64", WeightDtype::F16) => "gemm_bi_nn_tc64_f16",
            ("gemm_bi_nn_tc16", WeightDtype::Bf16) => "gemm_bi_nn_tc16_bf16",
            ("gemm_bi_nn_tc16", WeightDtype::F16) => "gemm_bi_nn_tc16_f16",
            ("gemm_bi_tn_tc", WeightDtype::Bf16) => "gemm_bi_tn_tc_bf16",
            ("gemm_bi_tn_tc", WeightDtype::F16) => "gemm_bi_tn_tc_f16",
            ("gemm_bi_tn_tc64", WeightDtype::Bf16) => "gemm_bi_tn_tc64_bf16",
            ("gemm_bi_tn_tc64", WeightDtype::F16) => "gemm_bi_tn_tc64_f16",
            ("gemm_bi_tn_tc64_streamk", WeightDtype::Bf16) => "gemm_bi_tn_tc64_streamk_bf16",
            ("gemm_bi_tn_tc64_streamk", WeightDtype::F16) => "gemm_bi_tn_tc64_streamk_f16",
            ("gemm_bi_tn_tc128x64", WeightDtype::Bf16) => "gemm_bi_tn_tc128x64_bf16",
            ("gemm_bi_tn_tc128x64", WeightDtype::F16) => "gemm_bi_tn_tc128x64_f16",
            ("gemm_bi_nt_tc", WeightDtype::Bf16) => "gemm_bi_nt_tc_bf16",
            ("gemm_bi_nt_tc", WeightDtype::F16) => "gemm_bi_nt_tc_f16",
            ("gemm_bi_nt_tc64", WeightDtype::Bf16) => "gemm_bi_nt_tc64_bf16",
            ("gemm_bi_nt_tc64", WeightDtype::F16) => "gemm_bi_nt_tc64_f16",
            (_, WeightDtype::F32) => {
                return Err("half kernel identity does not accept f32".into());
            }
            _ => return Err(format!("unknown half kernel base {base}")),
        };
        Ok(Self {
            symbol,
            module_kind,
            schedule,
        })
    }

    #[cfg(test)]
    fn validate(self, module_kind: ModuleKind, symbol: &str) -> Result<(), String> {
        if module_kind != self.module_kind || symbol != self.symbol {
            return Err(
                "half kernel identity does not match its exact symbol and module owner".into(),
            );
        }
        Ok(())
    }
}

struct HalfLaunchEnvironment<'a, O> {
    stream: &'a Arc<CudaStream>,
    kernels: &'a GpuKernels,
    observer: O,
}

impl<'a> HalfLaunchEnvironment<'a, NoPhysicalObserver> {
    fn production(stream: &'a Arc<CudaStream>, kernels: &'a GpuKernels) -> Self {
        Self {
            stream,
            kernels,
            observer: NoPhysicalObserver,
        }
    }
}

impl<'a, O: PhysicalLaunchObserver> HalfLaunchEnvironment<'a, O> {
    fn observed(ctx: &'a GpuCtx, observer: O) -> Self {
        Self {
            stream: &ctx.stream,
            kernels: &ctx.kernels,
            observer,
        }
    }
}

const _: [(); 2 * std::mem::size_of::<usize>()] =
    [(); std::mem::size_of::<HalfLaunchEnvironment<'static, NoPhysicalObserver>>()];

#[derive(Clone, Copy)]
struct HalfGemmArguments {
    output: CUptr,
    a: CUptr,
    b: CUptr,
    bias: CUptr,
}

#[derive(Clone, Copy)]
struct HalfGemmObservation {
    base: &'static str,
    op: ResolvedGemmOp,
    dtype: WeightDtype,
    dims: (usize, usize, usize),
    strides: (usize, usize, usize),
    tile: (u32, u32),
    bk_stages: (u32, u8),
    arguments: HalfGemmArguments,
}

impl HalfGemmObservation {
    fn shape(self) -> F32TriadShape {
        F32TriadShape {
            m: self.dims.0,
            k: self.dims.1,
            n: self.dims.2,
            lda: self.strides.0,
            ldb: self.strides.1,
            ldc: self.strides.2,
        }
    }
}

#[derive(Clone, Copy)]
struct HalfKernelChoice<'a> {
    base: &'static str,
    function: &'a CudaFunction,
}

impl<'a> HalfKernelChoice<'a> {
    fn new(base: &'static str, function: &'a CudaFunction) -> Self {
        Self { base, function }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::mamba_ssm::gpu) struct HalfNativeBranchSeal {
    pub(in crate::mamba_ssm::gpu) base: &'static str,
    pub(in crate::mamba_ssm::gpu) op: ResolvedGemmOp,
    pub(in crate::mamba_ssm::gpu) dtype: WeightDtype,
    pub(in crate::mamba_ssm::gpu) dims: (usize, usize, usize),
    pub(in crate::mamba_ssm::gpu) strides: (usize, usize, usize),
    pub(in crate::mamba_ssm::gpu) tile: (u32, u32),
    pub(in crate::mamba_ssm::gpu) bk_stages: (u32, u8),
    pub(in crate::mamba_ssm::gpu) grid_dim: (u32, u32, u32),
    pub(in crate::mamba_ssm::gpu) block_dim: (u32, u32, u32),
    pub(in crate::mamba_ssm::gpu) shared_mem_bytes: u32,
}

fn half_policy_dtype(dtype: WeightDtype) -> Result<PolicyDtype, String> {
    match dtype {
        WeightDtype::Bf16 => Ok(PolicyDtype::Bf16),
        WeightDtype::F16 => Ok(PolicyDtype::F16),
        WeightDtype::F32 => Err("half physical launch identity does not accept f32".into()),
    }
}

fn half_gemm_arguments_digest<O: PhysicalLaunchObserver>(
    observer: &O,
    observation: HalfGemmObservation,
    identity: HalfKernelIdentity,
    dtype: PolicyDtype,
) -> Result<Sha256Digest, String> {
    let shape = observation.shape();
    let element_bytes = u64::try_from(observation.dtype.size_bytes())
        .map_err(|_| "half element width exceeds u64::MAX".to_string())?;
    let matrix_bytes = |elements: usize, name: &str| {
        u64::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| format!("physical half {name} span overflows u64"))
    };
    let output_element_bytes = if observation.op == ResolvedGemmOp::Tn {
        4
    } else {
        element_bytes
    };
    let output_elements = shape
        .output_rows(observation.op)
        .checked_mul(shape.output_columns(observation.op))
        .ok_or_else(|| "physical half output span overflows usize".to_string())?;
    let output_bytes = u64::try_from(output_elements)
        .ok()
        .and_then(|elements| elements.checked_mul(output_element_bytes))
        .ok_or_else(|| "physical half output span overflows u64".to_string())?;
    let elements = |rows: usize, columns: usize, name: &str| {
        rows.checked_mul(columns)
            .ok_or_else(|| format!("physical half {name} span overflows usize"))
    };
    let (a_elements, b_elements) = match observation.op {
        ResolvedGemmOp::Nn => (
            elements(shape.m, shape.k, "A")?,
            elements(shape.k, shape.n, "B")?,
        ),
        ResolvedGemmOp::Tn => (
            elements(shape.m, shape.k, "A")?,
            elements(shape.m, shape.n, "B")?,
        ),
        ResolvedGemmOp::Nt => (
            elements(shape.m, shape.n, "A")?,
            elements(shape.k, shape.n, "B")?,
        ),
    };
    let output_identity =
        observer.argument_identity_digest(observation.arguments.output, output_bytes)?;
    let a_identity = observer
        .argument_identity_digest(observation.arguments.a, matrix_bytes(a_elements, "A")?)?;
    let b_identity = observer
        .argument_identity_digest(observation.arguments.b, matrix_bytes(b_elements, "B")?)?;
    let bias_identity = if observation.arguments.bias == 0 {
        None
    } else {
        let bytes = u64::try_from(shape.output_columns(observation.op))
            .ok()
            .and_then(|columns| columns.checked_mul(4))
            .ok_or_else(|| "physical half bias span overflows u64".to_string())?;
        Some(observer.argument_identity_digest(observation.arguments.bias, bytes)?)
    };
    let alpha = 1.0_f32;
    let beta = if observation.op == ResolvedGemmOp::Tn {
        1.0_f32
    } else {
        0.0_f32
    };
    Ok(FramedSha256::new(b"triad-half-kernel-arguments.v2")
        .required(b"symbol", identity.symbol.as_bytes())
        .required(b"op", &[observation.op as u8])
        .required(b"dtype", &[dtype as u8])
        .required(b"m", &(shape.m as u64).to_le_bytes())
        .required(b"k", &(shape.k as u64).to_le_bytes())
        .required(b"n", &(shape.n as u64).to_le_bytes())
        .required(b"lda", &(shape.lda as u64).to_le_bytes())
        .required(b"ldb", &(shape.ldb as u64).to_le_bytes())
        .required(b"ldc", &(shape.ldc as u64).to_le_bytes())
        .required(b"alpha", &alpha.to_bits().to_le_bytes())
        .required(b"beta", &beta.to_bits().to_le_bytes())
        .required(
            b"null-pointer-mask",
            &(u64::from(observation.arguments.bias == 0) << 3).to_le_bytes(),
        )
        .required(b"output-allocation", &output_identity)
        .required(b"a-allocation", &a_identity)
        .required(b"b-allocation", &b_identity)
        .optional(
            b"bias-allocation",
            bias_identity.as_ref().map(Sha256Digest::as_slice),
        )
        .required(b"tile-m", &observation.tile.0.to_le_bytes())
        .required(b"tile-n", &observation.tile.1.to_le_bytes())
        .required(b"bk", &observation.bk_stages.0.to_le_bytes())
        .required(b"stages", &[observation.bk_stages.1])
        .finish())
}

fn half_gemm_resources_digest(identity: HalfKernelIdentity, config: LaunchConfig) -> Sha256Digest {
    FramedSha256::new(b"triad-half-kernel-resources.v1")
        .required(b"symbol", identity.symbol.as_bytes())
        .required(b"module-kind", &[identity.module_kind as u8])
        .required(
            b"threads",
            &(config.block_dim.0 * config.block_dim.1 * config.block_dim.2).to_le_bytes(),
        )
        .required(
            b"dynamic-shared-memory-bytes",
            &config.shared_mem_bytes.to_le_bytes(),
        )
        .finish()
}

fn resolved_half_gemm_route<O: PhysicalLaunchObserver>(
    context: GemmRouteIdentity,
    kernels: &GpuKernels,
    observer: &O,
    observation: HalfGemmObservation,
    identity: HalfKernelIdentity,
    config: LaunchConfig,
) -> Result<ResolvedGemmRoute, String> {
    let dtype = half_policy_dtype(observation.dtype)?;
    let shape = observation.shape();
    let (compiler, artifact, backend, numeric_contract, instruction_family, instruction_shape) =
        match identity.module_kind {
            ModuleKind::TriadScalar => (
                kernels.triad_scalar_compiler_identity(),
                context.artifacts.triad_scalar,
                PhysicalGemmBackend::ScalarFmaV1,
                ResolvedNumericContract::ScalarFmaV1,
                ResolvedInstructionFamily::ScalarFma,
                ResolvedInstructionShape { m: 1, n: 1, k: 1 },
            ),
            ModuleKind::TriadSm80 => (
                kernels.triad_sm80_compiler_identity(),
                context.artifacts.triad_sm80,
                PhysicalGemmBackend::Sm80Mma16V1,
                match identity.schedule {
                    HalfSchedule::Tiled => ResolvedNumericContract::MmaSyncF32V1,
                    HalfSchedule::StreamKFixedOrder => {
                        ResolvedNumericContract::MmaSyncF32StreamKFixedOrderV1
                    }
                },
                ResolvedInstructionFamily::MmaSync,
                ResolvedInstructionShape { m: 16, n: 8, k: 16 },
            ),
            module_kind => {
                return Err(format!(
                    "half physical GEMM has unsupported module owner {module_kind:?}"
                ));
            }
        };
    let launch = ResolvedKernelLaunch {
        grid_dim: config.grid_dim,
        block_dim: config.block_dim,
        shared_mem_bytes: config.shared_mem_bytes,
        arguments_digest: half_gemm_arguments_digest(observer, observation, identity, dtype)?,
    };
    Ok(ResolvedGemmRoute {
        op: observation.op,
        dtype,
        backend,
        numeric_contract,
        instruction_family,
        instruction_shape,
        operand_conversion: ResolvedOperandConversion::None,
        ownership: match identity.schedule {
            HalfSchedule::Tiled => ResolvedOutputOwnership::OneCtaPerOutputTileV1,
            HalfSchedule::StreamKFixedOrder => {
                ResolvedOutputOwnership::OwnerCtaPerOutputTileStreamKFixedOrderV1
            }
        },
        symbol: identity.symbol,
        module_kind: identity.module_kind,
        target: compiler.target,
        artifact,
        compiler,
        device: context.device,
        device_caps: context.device_caps,
        shape: (shape.m, shape.k, shape.n),
        strides: (shape.lda, shape.ldb, shape.ldc),
        tile: observation.tile,
        bk: observation.bk_stages.0,
        stages: observation.bk_stages.1,
        threads: launch.block_dim.0 * launch.block_dim.1 * launch.block_dim.2,
        launch,
        tensor_map_revision: 0,
        tensor_maps_digest: [0; 32],
        resources_digest: half_gemm_resources_digest(identity, config),
        tuning_table_revision: TUNING_TABLE_REVISION,
        schedule_revision: SCHEDULE_REVISION,
    })
}

fn resolve_half_gemm_observation<O: PhysicalLaunchObserver>(
    observer: &O,
    kernels: &GpuKernels,
    observation: HalfGemmObservation,
    config: LaunchConfig,
) -> Result<PhysicalLaunchObservation, String> {
    let context = observer
        .route_context()
        .ok_or_else(|| "recording half launch requires a GEMM route context".to_string())?;
    let identity = HalfKernelIdentity::resolve(observation.base, observation.dtype)?;
    if !context.policy.batch_invariant
        || context.policy.bi_gemm_family != crate::mamba_ssm::gpu::context::BiGemmFamily::Triad
    {
        return Err(
            "recording a half launch requires the live batch-invariant Triad policy".into(),
        );
    }
    if identity.module_kind == ModuleKind::TriadSm80 && !context.policy.bi_tensor_cores {
        return Err(
            "recording a forced half Tensor Core launch requires the live Triad Tensor Core policy"
                .into(),
        );
    }
    let route =
        resolved_half_gemm_route(context, kernels, observer, observation, identity, config)?;
    Ok(PhysicalLaunchObservation::gemm(
        half_policy_dtype(observation.dtype)?,
        None,
        route,
    ))
}

pub(in crate::mamba_ssm::gpu) struct PreparedHalfGraphIdentity {
    function: CudaFunction,
    config: LaunchConfig,
    node: ResolvedPhysicalKernelLaunch,
    base: &'static str,
}

impl PreparedHalfGraphIdentity {
    pub(in crate::mamba_ssm::gpu) fn into_parts(
        self,
    ) -> (
        CudaFunction,
        LaunchConfig,
        ResolvedPhysicalKernelLaunch,
        &'static str,
    ) {
        (self.function, self.config, self.node, self.base)
    }
}

pub(in crate::mamba_ssm::gpu) fn prepare_native_half_graph_identity<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &O,
    expected: ResolvedPhysicalKernelLaunch,
    request: HalfPhysicalTraceRequest,
) -> Result<PreparedHalfGraphIdentity, String> {
    let suffix = match request.dtype {
        WeightDtype::Bf16 => "_bf16",
        WeightDtype::F16 => "_f16",
        WeightDtype::F32 => return Err("native half graph identity does not accept f32".into()),
    };
    let base = expected
        .symbol()
        .strip_suffix(suffix)
        .ok_or_else(|| "native half graph symbol has the wrong dtype suffix".to_string())?;
    let choice = match base {
        "gemm_bi_nn_gemv" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nn_gemv_typed.get(request.dtype))
        }
        "gemm_bi_nn_ultra_thin" => HalfKernelChoice::new(
            base,
            ctx.kernels.gemm_bi_nn_ultra_thin_typed.get(request.dtype),
        ),
        "gemm_bi_nn_narrow_small" => HalfKernelChoice::new(
            base,
            ctx.kernels.gemm_bi_nn_narrow_small_typed.get(request.dtype),
        ),
        "gemm_bi_nn_narrow" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nn_narrow_typed.get(request.dtype))
        }
        "gemm_bi_nn_big" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nn_big_typed.get(request.dtype))
        }
        "gemm_bi_tn_gemv" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_tn_gemv_typed.get(request.dtype))
        }
        "gemm_bi_tn_narrow" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_tn_narrow_typed.get(request.dtype))
        }
        "gemm_bi_tn_big" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_tn_big_typed.get(request.dtype))
        }
        "gemm_bi_nt_gemv" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nt_gemv_typed.get(request.dtype))
        }
        "gemm_bi_nt_narrow" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nt_narrow_typed.get(request.dtype))
        }
        "gemm_bi_nt_big" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nt_big_typed.get(request.dtype))
        }
        "gemm_bi_nn_tc" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nn_tc_typed.get(request.dtype))
        }
        "gemm_bi_nn_tc64" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nn_tc64_typed.get(request.dtype))
        }
        "gemm_bi_nn_tc16" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nn_tc16_typed.get(request.dtype))
        }
        "gemm_bi_tn_tc" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_tn_tc_typed.get(request.dtype))
        }
        "gemm_bi_tn_tc64" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_tn_tc64_typed.get(request.dtype))
        }
        "gemm_bi_tn_tc128x64" => HalfKernelChoice::new(
            base,
            ctx.kernels.gemm_bi_tn_tc128x64_typed.get(request.dtype),
        ),
        "gemm_bi_tn_tc64_streamk" => HalfKernelChoice::new(
            base,
            ctx.kernels
                .gemm_bi_tn_tc64_streamk_typed
                .as_ref()
                .ok_or_else(|| {
                    "the portable stream-K dW kernel is not composed for this target".to_string()
                })?
                .get(request.dtype),
        ),
        "gemm_bi_nt_tc" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nt_tc_typed.get(request.dtype))
        }
        "gemm_bi_nt_tc64" => {
            HalfKernelChoice::new(base, ctx.kernels.gemm_bi_nt_tc64_typed.get(request.dtype))
        }
        _ => return Err("native half graph symbol is not a prepared typed route".into()),
    };
    let launch = expected.launch();
    let config = LaunchConfig {
        grid_dim: launch.grid_dim,
        block_dim: launch.block_dim,
        shared_mem_bytes: launch.shared_mem_bytes,
    };
    let route = expected
        .gemm_route()
        .ok_or_else(|| "native half graph node has no GEMM route".to_string())?;
    let observation = HalfGemmObservation {
        base: choice.base,
        op: request.op,
        dtype: request.dtype,
        dims: request.dims,
        strides: request.nn_strides.unwrap_or_else(|| {
            let shape = F32TriadShape::contiguous(request.op, request.dims);
            (shape.lda, shape.ldb, shape.ldc)
        }),
        tile: expected
            .tile()
            .ok_or_else(|| "native half graph node has no tile".to_string())?,
        bk_stages: (route.bk, route.stages),
        arguments: HalfGemmArguments {
            output: request.output,
            a: request.a,
            b: request.b,
            bias: if request.op == ResolvedGemmOp::Nn {
                request.bias
            } else {
                0
            },
        },
    };
    let resolved = resolve_half_gemm_observation(observer, &ctx.kernels, observation, config)?;
    let node = resolve_physical_launch_observation(observer, resolved, config)?;
    Ok(PreparedHalfGraphIdentity {
        function: choice.function.clone(),
        config,
        node,
        base: choice.base,
    })
}

#[inline(always)]
unsafe fn enqueue_half_gemm<O: PhysicalLaunchObserver>(
    observer: &mut O,
    kernels: &GpuKernels,
    builder: &mut LaunchArgs<'_>,
    config: LaunchConfig,
    observation: HalfGemmObservation,
    driver_context: std::fmt::Arguments<'_>,
) -> Result<HalfNativeBranchSeal, String> {
    let physical_observation = if O::ENABLED {
        Some(resolve_half_gemm_observation(
            observer,
            kernels,
            observation,
            config,
        )?)
    } else {
        None
    };
    unsafe { enqueue_with_physical_observation(observer, builder, config, physical_observation) }
        .map_err(|error| error.with_driver_context(driver_context))?;
    Ok(HalfNativeBranchSeal {
        base: observation.base,
        op: observation.op,
        dtype: observation.dtype,
        dims: observation.dims,
        strides: observation.strides,
        tile: observation.tile,
        bk_stages: observation.bk_stages,
        grid_dim: config.grid_dim,
        block_dim: config.block_dim,
        shared_mem_bytes: config.shared_mem_bytes,
    })
}
impl TcTile {
    /// CTA tile edge in output elements.
    /// Output-tile extents `(bm, bn)` - the ladder is not square.
    fn extents(self) -> (u32, u32) {
        match self {
            TcTile::Tile128 => (128, 128),
            TcTile::Tile64 => (64, 64),
            TcTile::Thin16 => (16, 32),
            TcTile::Rect128x64 => (128, 64),
            TcTile::Tile64StreamK => (64, 64),
        }
    }

    /// CTA thread count (must match `__launch_bounds__` of the kernels).
    fn block_dim(self) -> u32 {
        match self {
            TcTile::Tile128 => 256,
            TcTile::Tile64 | TcTile::Thin16 | TcTile::Tile64StreamK => 128,
            TcTile::Rect128x64 => 256,
        }
    }

    fn bk_stages(self) -> (u32, u8) {
        match self {
            TcTile::Tile128 | TcTile::Tile64 | TcTile::Tile64StreamK => (64, 2),
            TcTile::Thin16 => (64, 4),
            TcTile::Rect128x64 => (32, 3),
        }
    }

    /// 1-D launch config over the output tile grid `rows x cols`.
    /// `dyn_bytes128`: the Tile128 kernel's dynamic-smem footprint (the
    /// BK=64 staging exceeds the 48 KB static cap, so the 128-tile family
    /// uses `extern __shared__`; per-op: NN 71 680, TN 69 632, NT 73 728 —
    /// must stay <= the MAX_DYNAMIC_SHARED opt-in set at load,
    /// modules.rs). The Tile64 family stays on static smem (36 864 B).
    fn launch_cfg(
        self,
        rows: usize,
        cols: usize,
        dyn_bytes128: u32,
    ) -> Result<cudarc::driver::LaunchConfig, String> {
        if self == TcTile::Tile64StreamK {
            return Err(
                "Tile64StreamK launches a persistent grid; use the stream-K dW path".into(),
            );
        }
        let (bm, bn) = self.extents();
        let total_tiles = checked_tile_grid(
            checked_u32(rows, "tile rows")?,
            bm,
            checked_u32(cols, "tile columns")?,
            bn,
        )?;
        Ok(cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (self.block_dim(), 1, 1),
            shared_mem_bytes: match self {
                TcTile::Tile128 => dyn_bytes128,
                TcTile::Tile64 | TcTile::Thin16 | TcTile::Tile64StreamK => 0,
                TcTile::Rect128x64 => 0,
            },
        })
    }
}

/// Slab floats of one stream-K (CTA, slot) pair: 128 threads by 32
/// accumulators, matching `GEMM_BI_TC64_STREAMK_ACCUMULATORS` in sm80.cu.
const SM80_STREAMK_SLAB_FLOATS: usize = 128 * 32;
/// Slots per CTA: a range can end inside its first tile or a later one.
const SM80_STREAMK_SLOTS_PER_CTA: usize = 2;

/// The persistent grid of the tc64 TN stream-K kernel: one CTA per
/// multiprocessor, never more CTAs than (tile, slab) units, never zero.
pub fn sm80_streamk_grid(kernels: &GpuKernels, dims: (usize, usize, usize)) -> Result<u32, String> {
    let checked = GemmDims::tn(dims)?;
    let (batch, n_in, n_out) = checked.tuple();
    let tiles = checked_tile_grid(
        checked_u32(n_in, "tile rows")?,
        64,
        checked_u32(n_out, "tile columns")?,
        64,
    )?;
    let slabs = checked_u32(batch, "reduction rows")?.div_ceil(64);
    let units = u64::from(tiles) * u64::from(slabs);
    let multiprocessors = u64::from(kernels.multiprocessor_count().max(1));
    u32::try_from(units.min(multiprocessors).max(1))
        .map_err(|_| "stream-K grid exceeds u32".to_string())
}

/// The partial-slab and flag pointers of the stream-K dW kernel, from the
/// fixed split-K workspaces; the extents are checked against their caps so
/// a grid the workspace cannot hold fails here, not in the kernel.
pub fn sm80_streamk_workspace(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    grid: u32,
) -> Result<(CUptr, CUptr), String> {
    use cudarc::driver::DevicePtr;
    let slots = (grid as usize)
        .checked_mul(SM80_STREAMK_SLOTS_PER_CTA)
        .ok_or_else(|| "sm80 stream-K slot count overflows usize".to_string())?;
    let partial_floats = slots
        .checked_mul(SM80_STREAMK_SLAB_FLOATS)
        .ok_or_else(|| "sm80 stream-K slab extent overflows usize".to_string())?;
    if partial_floats > SPLITK_SCRATCH_CAP {
        return Err("sm80 stream-K slabs exceed the fixed workspace".into());
    }
    if slots > TF32_SPLITK_COUNTER_CAP {
        return Err("sm80 stream-K flags exceed the fixed counter workspace".into());
    }
    let (partial, _) = kernels.splitk_scratch_buf(stream)?.device_ptr(stream);
    let (flags, _) = kernels
        .triad_kernels()
        .tf32_splitk_counter_buf(stream)?
        .device_ptr(stream);
    Ok((partial, flags))
}

/// Tensor-core NN forward (`bi_tensor_cores` tier):
/// `Y = X @ W + bias` via mma.sync.m16n8k16 with f32 accumulation.
/// SEPARATE numeric contract from the scalar triad (TC reduction tree, not
/// the ascending-K FMA chain) — deterministic and batch-invariant across
/// ALL M (each element's full K-reduction lives in one warp, independent of
/// grid shape; the Thin16/Tile64/Tile128 rungs are bit-identical per
/// element). Covers every M at N >= 32 (the Thin16 column floor), K >= 1;
/// Err otherwise. Returns the tile variant that actually launched.
pub fn gemm_bi_forward_tc(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    y: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: CUptr,
    dims: (usize, usize, usize),
) -> Result<TcTile, String> {
    let mut environment = HalfLaunchEnvironment::production(stream, kernels);
    gemm_bi_forward_tc_in(&mut environment, y, x, w, bias_ptr, dims).map(|(tile, _)| tile)
}

fn gemm_bi_forward_tc_in<O: PhysicalLaunchObserver>(
    environment: &mut HalfLaunchEnvironment<'_, O>,
    y: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: CUptr,
    dims: (usize, usize, usize),
) -> Result<(TcTile, HalfNativeBranchSeal), String> {
    GemmDims::nn(dims, dims.1)?;
    let tile = tc_pick_tile_forward(dims, environment.kernels.multiprocessor_count()).ok_or_else(|| {
        let (batch, n_in, n_out) = dims;
        format!(
            "UNCOVERED gemm_bi_forward_tc: shape M={batch} K={n_in} N={n_out} below the TC tile gate"
        )
    })?;
    let ops = TcFwdOperands { y, x, w, bias_ptr };
    let seal = gemm_bi_forward_tc_with_tile_in(
        environment,
        &ops,
        F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
        tile,
    )?;
    Ok((tile, seal))
}

/// Forced-tile TC NN forward. Exposed so the cross-tile bit-identity
/// contract (Tile64 == Tile128 per element) is directly testable; the
/// auto-routing entry is [`gemm_bi_forward_tc`]. The caller must respect
/// the tile's gate (M and N >= tile edge is NOT required — both kernels
/// predicate tails — but M >= 64 && N >= 64 keeps warps useful).
pub fn gemm_bi_forward_tc_with_tile(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    ops: &TcFwdOperands,
    dims: (usize, usize, usize),
    tile: TcTile,
) -> Result<(), String> {
    let mut environment = HalfLaunchEnvironment::production(stream, kernels);
    gemm_bi_forward_tc_with_tile_in(
        &mut environment,
        ops,
        F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
        tile,
    )
    .map(drop)
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_forward_tc_with_tile_shape(
    ctx: &GpuCtx,
    ops: &TcFwdOperands,
    shape: F32TriadShape,
    tile: TcTile,
) -> Result<(), String> {
    let mut environment = HalfLaunchEnvironment::production(&ctx.stream, &ctx.kernels);
    gemm_bi_forward_tc_with_tile_in(&mut environment, ops, shape, tile).map(drop)
}

fn gemm_bi_forward_tc_with_tile_in<O: PhysicalLaunchObserver>(
    environment: &mut HalfLaunchEnvironment<'_, O>,
    ops: &TcFwdOperands,
    shape: F32TriadShape,
    tile: TcTile,
) -> Result<HalfNativeBranchSeal, String> {
    shape.validate(ResolvedGemmOp::Nn)?;
    let dims = (shape.m, shape.k, shape.n);
    let checked_dims = GemmDims::nn(dims, shape.lda)?;
    let (batch, _n_in, n_out) = checked_dims.tuple();
    require_half(ops.y.dtype, "output")?;
    if ops.x.dtype != ops.y.dtype || ops.w.dtype != ops.y.dtype {
        return Err("gemm_bi_forward_tc: mixed dtypes not supported".into());
    }
    let dt = ops.y.dtype;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    validate_bias_preseed(alpha, ops.bias_ptr, "gemm_bi_forward_tc")?;
    let m_i = checked_dims.m_i32;
    let n_i = checked_dims.n_i32;
    let k_i = checked_dims.k_i32;
    let cfg = tile.launch_cfg(batch, n_out, 71_680)?;
    let choice = match tile {
        TcTile::Tile128 => HalfKernelChoice::new(
            "gemm_bi_nn_tc",
            environment.kernels.gemm_bi_nn_tc_typed.get(dt),
        ),
        TcTile::Tile64 => HalfKernelChoice::new(
            "gemm_bi_nn_tc64",
            environment.kernels.gemm_bi_nn_tc64_typed.get(dt),
        ),
        TcTile::Thin16 => HalfKernelChoice::new(
            "gemm_bi_nn_tc16",
            environment.kernels.gemm_bi_nn_tc16_typed.get(dt),
        ),
        TcTile::Rect128x64 => {
            return Err("Rect128x64 is a forced TN dW tile; NN has no rectangular route".into());
        }
        TcTile::Tile64StreamK => {
            return Err("Tile64StreamK is a TN dW schedule; NN has no stream-K route".into());
        }
    };
    let mut b = environment.stream.launch_builder(choice.function);
    b.arg(&ops.y.ptr);
    b.arg(&ops.x.ptr);
    b.arg(&ops.w.ptr);
    b.arg(&ops.bias_ptr);
    b.arg(&alpha);
    b.arg(&beta);
    b.arg(&m_i);
    b.arg(&n_i);
    b.arg(&k_i);
    let lda_i = checked_i32(shape.lda, "lda")?;
    let ldb_i = checked_i32(shape.ldb, "ldb")?;
    let ldc_i = checked_i32(shape.ldc, "ldc")?;
    b.arg(&lda_i);
    b.arg(&ldb_i);
    b.arg(&ldc_i);
    unsafe {
        enqueue_half_gemm(
            &mut environment.observer,
            environment.kernels,
            &mut b,
            cfg,
            HalfGemmObservation {
                base: choice.base,
                op: ResolvedGemmOp::Nn,
                dtype: dt,
                dims,
                strides: (shape.lda, shape.ldb, shape.ldc),
                tile: tile.extents(),
                bk_stages: tile.bk_stages(),
                arguments: HalfGemmArguments {
                    output: ops.y.ptr,
                    a: ops.x.ptr,
                    b: ops.w.ptr,
                    bias: ops.bias_ptr,
                },
            },
            format_args!("gemm_bi_nn_tc ({tile:?})"),
        )
    }
}

/// Tensor-core TN dW: `dW[K,N] += X^T @ dY` via mma.sync with f32
/// accumulate straight into the f32 master gradient. Same TC contract as
/// [`gemm_bi_forward_tc`]. Large outputs keep the square-tile policy;
/// qualified one-axis tails use Tile64. Returns the tile that launched.
pub fn gemm_bi_backward_dw_tc(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dw_ptr: CUptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<TcTile, String> {
    let mut environment = HalfLaunchEnvironment::production(stream, kernels);
    gemm_bi_backward_dw_tc_in(
        &mut environment,
        dw_ptr,
        dy,
        x_saved,
        dims,
        HalfTriadPolicy::TiledParityV1,
    )
    .map(|(tile, _)| tile)
}

fn gemm_bi_backward_dw_tc_in<O: PhysicalLaunchObserver>(
    environment: &mut HalfLaunchEnvironment<'_, O>,
    dw_ptr: CUptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
    half_policy: HalfTriadPolicy,
) -> Result<(TcTile, HalfNativeBranchSeal), String> {
    let checked_dims = GemmDims::tn(dims)?;
    let (batch, n_in, n_out) = checked_dims.tuple();
    let compute_capability = environment
        .stream
        .context()
        .compute_capability()
        .map_err(|error| format!("query CUDA compute capability: {error:?}"))?;
    // Tile geometry keys on (K_out, N). Tail admission also keeps the
    // reduction length in its frozen performance key.
    let tile = tc_pick_tile_backward_for_device(
        super::super::kernel_identity::PolicyOp::Dw,
        dims,
        environment.kernels.multiprocessor_count(),
        compute_capability,
        half_policy,
    ).ok_or_else(|| {
        format!(
            "UNCOVERED gemm_bi_backward_dw_tc: shape M={batch} K={n_in} N={n_out} outside the automatic TC route"
        )
    })?;
    let seal = gemm_bi_backward_dw_tc_with_tile_in(environment, dw_ptr, dy, x_saved, dims, tile)?;
    Ok((tile, seal))
}

/// Forced-tile TC TN dW (see [`gemm_bi_forward_tc_with_tile`]).
pub fn gemm_bi_backward_dw_tc_with_tile(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dw_ptr: CUptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
    tile: TcTile,
) -> Result<(), String> {
    let mut environment = HalfLaunchEnvironment::production(stream, kernels);
    gemm_bi_backward_dw_tc_with_tile_in(&mut environment, dw_ptr, dy, x_saved, dims, tile).map(drop)
}

fn gemm_bi_backward_dw_tc_with_tile_in<O: PhysicalLaunchObserver>(
    environment: &mut HalfLaunchEnvironment<'_, O>,
    dw_ptr: CUptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
    tile: TcTile,
) -> Result<HalfNativeBranchSeal, String> {
    let checked_dims = GemmDims::tn(dims)?;
    let (_batch, n_in, n_out) = checked_dims.tuple();
    require_half(dy.dtype, "dY")?;
    if dy.dtype != x_saved.dtype {
        return Err("gemm_bi_backward_dw_tc: mixed dtypes not supported".into());
    }
    let dt = dy.dtype;
    let alpha: f32 = 1.0;
    let m_red_i = checked_dims.m_i32;
    let k_out_i = checked_dims.k_i32;
    let n_i = checked_dims.n_i32;
    let (cfg, choice, workspace) = match tile {
        TcTile::Tile128 => (
            tile.launch_cfg(n_in, n_out, 69_632)?,
            HalfKernelChoice::new(
                "gemm_bi_tn_tc",
                environment.kernels.gemm_bi_tn_tc_typed.get(dt),
            ),
            None,
        ),
        TcTile::Tile64 => (
            tile.launch_cfg(n_in, n_out, 69_632)?,
            HalfKernelChoice::new(
                "gemm_bi_tn_tc64",
                environment.kernels.gemm_bi_tn_tc64_typed.get(dt),
            ),
            None,
        ),
        TcTile::Rect128x64 => (
            tile.launch_cfg(n_in, n_out, 69_632)?,
            HalfKernelChoice::new(
                "gemm_bi_tn_tc128x64",
                environment.kernels.gemm_bi_tn_tc128x64_typed.get(dt),
            ),
            None,
        ),
        TcTile::Tile64StreamK => {
            let grid = sm80_streamk_grid(environment.kernels, dims)?;
            let workspace = sm80_streamk_workspace(environment.stream, environment.kernels, grid)?;
            (
                cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (tile.block_dim(), 1, 1),
                    shared_mem_bytes: 0,
                },
                HalfKernelChoice::new(
                    "gemm_bi_tn_tc64_streamk",
                    environment
                        .kernels
                        .gemm_bi_tn_tc64_streamk_typed
                        .as_ref()
                        .ok_or_else(|| {
                            "the portable stream-K dW kernel is not composed for this target"
                                .to_string()
                        })?
                        .get(dt),
                ),
                Some(workspace),
            )
        }
        TcTile::Thin16 => {
            return Err("Thin16 is an NN-forward rung; the TN dW path has no thin tile".into());
        }
    };
    let mut b = environment.stream.launch_builder(choice.function);
    b.arg(&dw_ptr);
    b.arg(&x_saved.ptr);
    b.arg(&dy.ptr);
    b.arg(&alpha);
    b.arg(&m_red_i);
    b.arg(&k_out_i);
    b.arg(&n_i);
    if let Some((partial, flags)) = &workspace {
        b.arg(partial);
        b.arg(flags);
    }
    unsafe {
        enqueue_half_gemm(
            &mut environment.observer,
            environment.kernels,
            &mut b,
            cfg,
            HalfGemmObservation {
                base: choice.base,
                op: ResolvedGemmOp::Tn,
                dtype: dt,
                dims,
                strides: (dims.1, dims.2, dims.2),
                tile: tile.extents(),
                bk_stages: tile.bk_stages(),
                arguments: HalfGemmArguments {
                    output: dw_ptr,
                    a: x_saved.ptr,
                    b: dy.ptr,
                    bias: 0,
                },
            },
            format_args!("gemm_bi_tn_tc ({tile:?})"),
        )
    }
}

/// Tensor-core NT dX: `dX[M,K] = dY @ W^T` via mma.sync, typed RNE
/// overwrite. Same TC contract as [`gemm_bi_forward_tc`]. Covers
/// large outputs plus qualified one-axis tails. Returns the tile that launched.
pub fn gemm_bi_backward_dx_tc(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<TcTile, String> {
    let mut environment = HalfLaunchEnvironment::production(stream, kernels);
    gemm_bi_backward_dx_tc_in(&mut environment, dx, dy, w, dims).map(|(tile, _)| tile)
}

fn gemm_bi_backward_dx_tc_in<O: PhysicalLaunchObserver>(
    environment: &mut HalfLaunchEnvironment<'_, O>,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(TcTile, HalfNativeBranchSeal), String> {
    let checked_dims = GemmDims::nt(dims)?;
    let (batch, n_in, n_out) = checked_dims.tuple();
    let compute_capability = environment
        .stream
        .context()
        .compute_capability()
        .map_err(|error| format!("query CUDA compute capability: {error:?}"))?;
    let tile = tc_pick_tile_backward_for_device(
        super::super::kernel_identity::PolicyOp::Dx,
        dims,
        environment.kernels.multiprocessor_count(),
        compute_capability,
        HalfTriadPolicy::TiledParityV1,
    ).ok_or_else(|| {
        format!(
            "UNCOVERED gemm_bi_backward_dx_tc: shape M={batch} K={n_in} N={n_out} outside the automatic TC route"
        )
    })?;
    let seal = gemm_bi_backward_dx_tc_with_tile_in(environment, dx, dy, w, dims, tile)?;
    Ok((tile, seal))
}

/// Forced-tile TC NT dX (see [`gemm_bi_forward_tc_with_tile`]).
pub fn gemm_bi_backward_dx_tc_with_tile(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
    tile: TcTile,
) -> Result<(), String> {
    let mut environment = HalfLaunchEnvironment::production(stream, kernels);
    gemm_bi_backward_dx_tc_with_tile_in(&mut environment, dx, dy, w, dims, tile).map(drop)
}

fn gemm_bi_backward_dx_tc_with_tile_in<O: PhysicalLaunchObserver>(
    environment: &mut HalfLaunchEnvironment<'_, O>,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
    tile: TcTile,
) -> Result<HalfNativeBranchSeal, String> {
    let checked_dims = GemmDims::nt(dims)?;
    let (batch, n_in, _n_out) = checked_dims.tuple();
    require_half(dx.dtype, "dX")?;
    if dx.dtype != dy.dtype || dy.dtype != w.dtype {
        return Err("gemm_bi_backward_dx_tc: mixed dtypes not supported".into());
    }
    let dt = dx.dtype;
    let alpha: f32 = 1.0;
    let m_i = checked_dims.m_i32;
    let n_i = checked_dims.n_i32;
    let k_out_i = checked_dims.k_i32;
    let cfg = tile.launch_cfg(batch, n_in, 73_728)?;
    let choice = match tile {
        TcTile::Tile128 => HalfKernelChoice::new(
            "gemm_bi_nt_tc",
            environment.kernels.gemm_bi_nt_tc_typed.get(dt),
        ),
        TcTile::Tile64 => HalfKernelChoice::new(
            "gemm_bi_nt_tc64",
            environment.kernels.gemm_bi_nt_tc64_typed.get(dt),
        ),
        TcTile::Thin16 => {
            return Err("Thin16 is an NN-forward rung; the NT dX path has no thin tile".into());
        }
        TcTile::Rect128x64 => {
            return Err("Rect128x64 is a forced TN dW tile; NT has no rectangular route".into());
        }
        TcTile::Tile64StreamK => {
            return Err("Tile64StreamK is a TN dW schedule; NT has no stream-K route".into());
        }
    };
    let mut b = environment.stream.launch_builder(choice.function);
    b.arg(&dx.ptr);
    b.arg(&dy.ptr);
    b.arg(&w.ptr);
    b.arg(&alpha);
    b.arg(&m_i);
    b.arg(&n_i);
    b.arg(&k_out_i);
    unsafe {
        enqueue_half_gemm(
            &mut environment.observer,
            environment.kernels,
            &mut b,
            cfg,
            HalfGemmObservation {
                base: choice.base,
                op: ResolvedGemmOp::Nt,
                dtype: dt,
                dims,
                strides: (dims.2, dims.2, dims.1),
                tile: tile.extents(),
                bk_stages: tile.bk_stages(),
                arguments: HalfGemmArguments {
                    output: dx.ptr,
                    a: dy.ptr,
                    b: w.ptr,
                    bias: 0,
                },
            },
            format_args!("gemm_bi_nt_tc ({tile:?})"),
        )
    }
}

/// Launches a native typed NN bucket without the full-policy fallback ladder.
///
/// This low-level qualification surface returns `UNCOVERED` when the shape has
/// no native bucket. Application code should use
/// [`crate::mamba_ssm::gpu::blas::gemm_bi_forward_typed`] instead.
pub fn gemm_bi_forward_typed_native(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    y: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: CUptr, // f32, 0 = none
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let mut environment = HalfLaunchEnvironment::production(stream, kernels);
    gemm_bi_forward_typed_in(&mut environment, y, x, w, bias_ptr, dims).map(drop)
}

fn gemm_bi_forward_typed_in<O: PhysicalLaunchObserver>(
    environment: &mut HalfLaunchEnvironment<'_, O>,
    y: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: CUptr,
    dims: (usize, usize, usize),
) -> Result<HalfNativeBranchSeal, String> {
    let checked_dims = GemmDims::nn(dims, dims.1)?;
    let (batch, n_in, n_out) = checked_dims.tuple();
    require_half(y.dtype, "output")?;
    if x.dtype != y.dtype || w.dtype != y.dtype {
        return Err("gemm_bi_forward_typed_native: mixed dtypes not supported".into());
    }
    let dt = y.dtype;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    validate_bias_preseed(alpha, bias_ptr, "gemm_bi_forward_typed_native")?;
    let m_i = checked_dims.m_i32;
    let n_i = checked_dims.n_i32;
    let k_i = checked_dims.k_i32;

    // GEMV N=1.
    if n_out == 1 && batch >= 1 && n_in >= 32 {
        let lda_i = checked_dims.k_i32;
        let ldy_i: i32 = 1;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_dims.m_u32.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let choice = HalfKernelChoice::new(
            "gemm_bi_nn_gemv",
            environment.kernels.gemm_bi_nn_gemv_typed.get(dt),
        );
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&y.ptr);
        b.arg(&x.ptr);
        b.arg(&w.ptr);
        b.arg(&bias_ptr);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&m_i);
        b.arg(&k_i);
        b.arg(&lda_i);
        b.arg(&ldy_i);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Nn,
                    dtype: dt,
                    dims,
                    strides: (dims.1, dims.2, dims.2),
                    tile: (4, 1),
                    bk_stages: (32, 1),
                    arguments: HalfGemmArguments {
                        output: y.ptr,
                        a: x.ptr,
                        b: w.ptr,
                        bias: bias_ptr,
                    },
                },
                format_args!("gemm_bi_nn_gemv typed"),
            )
        }?;
        return Ok(seal);
    }

    // Ultra-thin M (1..32).
    if (1..32).contains(&batch) && (32..=2048).contains(&n_in) && n_out >= 32 {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_dims.n_u32.div_ceil(32), checked_dims.m_u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: checked_u32_product(
                checked_dims.k_u32,
                checked_u32(std::mem::size_of::<f32>(), "f32 byte width")?,
                "typed ultra-thin shared memory",
            )?,
        };
        let choice = HalfKernelChoice::new(
            "gemm_bi_nn_ultra_thin",
            environment.kernels.gemm_bi_nn_ultra_thin_typed.get(dt),
        );
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&y.ptr);
        b.arg(&x.ptr);
        b.arg(&w.ptr);
        b.arg(&bias_ptr);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_i);
        b.arg(&k_i);
        b.arg(&n_i);
        b.arg(&n_i);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Nn,
                    dtype: dt,
                    dims,
                    strides: (dims.1, dims.2, dims.2),
                    tile: (1, 32),
                    bk_stages: (32, 1),
                    arguments: HalfGemmArguments {
                        output: y.ptr,
                        a: x.ptr,
                        b: w.ptr,
                        bias: bias_ptr,
                    },
                },
                format_args!("gemm_bi_nn_ultra_thin typed"),
            )
        }?;
        return Ok(seal);
    }

    // Narrow N (2..=127): small tile for batch <= 64, big-narrow otherwise.
    if (2..=127).contains(&n_out) && batch >= 1 && n_in >= 1 {
        let post_op: i32 = 0;
        let small = batch <= 64;
        let (grid, block, choice) = if small {
            (
                checked_tile_grid(checked_dims.m_u32, 16, checked_dims.n_u32, 16)?,
                64u32,
                HalfKernelChoice::new(
                    "gemm_bi_nn_narrow_small",
                    environment.kernels.gemm_bi_nn_narrow_small_typed.get(dt),
                ),
            )
        } else {
            (
                checked_tile_grid(checked_dims.m_u32, 64, checked_dims.n_u32, 32)?,
                128u32,
                HalfKernelChoice::new(
                    "gemm_bi_nn_narrow",
                    environment.kernels.gemm_bi_nn_narrow_typed.get(dt),
                ),
            )
        };
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&y.ptr);
        b.arg(&x.ptr);
        b.arg(&w.ptr);
        b.arg(&bias_ptr);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_i);
        b.arg(&k_i);
        b.arg(&n_i);
        b.arg(&n_i);
        b.arg(&post_op);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Nn,
                    dtype: dt,
                    dims,
                    strides: (dims.1, dims.2, dims.2),
                    tile: if small { (16, 16) } else { (64, 32) },
                    bk_stages: (16, 1),
                    arguments: HalfGemmArguments {
                        output: y.ptr,
                        a: x.ptr,
                        b: w.ptr,
                        bias: bias_ptr,
                    },
                },
                format_args!("gemm_bi_nn_narrow typed"),
            )
        }?;
        return Ok(seal);
    }

    // Native typed Big NN fires exactly
    // where the f32 cascade would run Big (predicate-mirrored gates).
    if nn_routes_to_big(
        batch,
        n_in,
        n_out,
        environment.kernels.multiprocessor_count(),
    ) {
        let total_tiles = checked_tile_grid(checked_dims.m_u32, 128, checked_dims.n_u32, 128)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 34 * 1024,
        };
        let choice = HalfKernelChoice::new(
            "gemm_bi_nn_big",
            environment.kernels.gemm_bi_nn_big_typed.get(dt),
        );
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&y.ptr);
        b.arg(&x.ptr);
        b.arg(&w.ptr);
        b.arg(&bias_ptr);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_i);
        b.arg(&k_i);
        b.arg(&n_i);
        b.arg(&n_i);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Nn,
                    dtype: dt,
                    dims,
                    strides: (dims.1, dims.2, dims.2),
                    tile: (128, 128),
                    bk_stages: (16, 2),
                    arguments: HalfGemmArguments {
                        output: y.ptr,
                        a: x.ptr,
                        b: w.ptr,
                        bias: bias_ptr,
                    },
                },
                format_args!("gemm_bi_nn_big typed"),
            )
        }?;
        return Ok(seal);
    }

    Err(format!(
        "UNCOVERED gemm_bi_forward_typed_native: Big/Slim buckets not yet implemented — \
         shape M={batch} K={n_in} N={n_out}. Disable the batch-invariant flag for \
         this configuration."
    ))
}

/// Launches a native typed TN bucket into the FP32 master gradient.
///
/// This low-level qualification surface returns `UNCOVERED` when the shape has
/// no native bucket. Application code should use
/// [`crate::mamba_ssm::gpu::blas::gemm_bi_backward_dw_typed`] instead.
pub fn gemm_bi_backward_dw_typed_native(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dw_ptr: CUptr, // f32 master, accumulated
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let mut environment = HalfLaunchEnvironment::production(stream, kernels);
    gemm_bi_backward_dw_typed_in(&mut environment, dw_ptr, dy, x_saved, dims).map(drop)
}

fn gemm_bi_backward_dw_typed_in<O: PhysicalLaunchObserver>(
    environment: &mut HalfLaunchEnvironment<'_, O>,
    dw_ptr: CUptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<HalfNativeBranchSeal, String> {
    let checked_dims = GemmDims::tn(dims)?;
    let (batch, n_in, n_out) = checked_dims.tuple();
    require_half(dy.dtype, "dY")?;
    if x_saved.dtype != dy.dtype {
        return Err("gemm_bi_backward_dw_typed_native: mixed dtypes not supported".into());
    }
    let dt = dy.dtype;
    let alpha: f32 = 1.0;

    // GEMV N=1.
    if n_out == 1 && n_in >= 4 && batch >= 32 {
        let m_i = checked_dims.m_i32;
        let k_i = checked_dims.k_i32;
        let lda_i = checked_dims.k_i32;
        let ldy_i: i32 = 1;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (checked_dims.k_u32.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let choice = HalfKernelChoice::new(
            "gemm_bi_tn_gemv",
            environment.kernels.gemm_bi_tn_gemv_typed.get(dt),
        );
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&dw_ptr);
        b.arg(&x_saved.ptr);
        b.arg(&dy.ptr);
        b.arg(&alpha);
        b.arg(&m_i);
        b.arg(&k_i);
        b.arg(&lda_i);
        b.arg(&ldy_i);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Tn,
                    dtype: dt,
                    dims,
                    strides: (dims.1, dims.2, dims.2),
                    tile: (4, 1),
                    bk_stages: (32, 1),
                    arguments: HalfGemmArguments {
                        output: dw_ptr,
                        a: x_saved.ptr,
                        b: dy.ptr,
                        bias: 0,
                    },
                },
                format_args!("gemm_bi_tn_gemv typed"),
            )
        }?;
        return Ok(seal);
    }

    // Narrow N (2..=127).
    if (2..=127).contains(&n_out) && batch >= 1 && n_in >= 1 {
        let m_red_i = checked_dims.m_i32;
        let k_out_i = checked_dims.k_i32;
        let n_i = checked_dims.n_i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                checked_tile_grid(checked_dims.k_u32, 64, checked_dims.n_u32, 32)?,
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let choice = HalfKernelChoice::new(
            "gemm_bi_tn_narrow",
            environment.kernels.gemm_bi_tn_narrow_typed.get(dt),
        );
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&dw_ptr);
        b.arg(&x_saved.ptr);
        b.arg(&dy.ptr);
        b.arg(&alpha);
        b.arg(&m_red_i);
        b.arg(&k_out_i);
        b.arg(&n_i);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Tn,
                    dtype: dt,
                    dims,
                    strides: (dims.1, dims.2, dims.2),
                    tile: (64, 32),
                    bk_stages: (16, 1),
                    arguments: HalfGemmArguments {
                        output: dw_ptr,
                        a: x_saved.ptr,
                        b: dy.ptr,
                        bias: 0,
                    },
                },
                format_args!("gemm_bi_tn_narrow typed"),
            )
        }?;
        return Ok(seal);
    }

    // Native typed Big TN keeps dW in f32 with += accumulation.
    if tn_routes_to_big(
        batch,
        n_in,
        n_out,
        environment.kernels.multiprocessor_count(),
    ) {
        let alpha: f32 = 1.0;
        let m_red_i = checked_dims.m_i32;
        let k_out_i = checked_dims.k_i32;
        let n_i = checked_dims.n_i32;
        let total_tiles = checked_tile_grid(checked_dims.k_u32, 128, checked_dims.n_u32, 128)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 34 * 1024,
        };
        let choice = HalfKernelChoice::new(
            "gemm_bi_tn_big",
            environment.kernels.gemm_bi_tn_big_typed.get(dt),
        );
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&dw_ptr);
        b.arg(&x_saved.ptr);
        b.arg(&dy.ptr);
        b.arg(&alpha);
        b.arg(&m_red_i);
        b.arg(&k_out_i);
        b.arg(&n_i);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Tn,
                    dtype: dt,
                    dims,
                    strides: (dims.1, dims.2, dims.2),
                    tile: (128, 128),
                    bk_stages: (16, 2),
                    arguments: HalfGemmArguments {
                        output: dw_ptr,
                        a: x_saved.ptr,
                        b: dy.ptr,
                        bias: 0,
                    },
                },
                format_args!("gemm_bi_tn_big typed"),
            )
        }?;
        return Ok(seal);
    }

    Err(format!(
        "UNCOVERED gemm_bi_backward_dw_typed_native: split-M/Slim buckets are upcast-fallback territory — \
         shape M={batch} K={n_in} N={n_out}."
    ))
}

/// Launches a native typed NT bucket and overwrites `dX`.
///
/// This low-level qualification surface returns `UNCOVERED` when the shape has
/// no native bucket. Application code should use
/// [`crate::mamba_ssm::gpu::blas::gemm_bi_backward_dx_typed`] instead.
pub fn gemm_bi_backward_dx_typed_native(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let mut environment = HalfLaunchEnvironment::production(stream, kernels);
    gemm_bi_backward_dx_typed_in(&mut environment, dx, dy, w, dims).map(drop)
}

fn gemm_bi_backward_dx_typed_in<O: PhysicalLaunchObserver>(
    environment: &mut HalfLaunchEnvironment<'_, O>,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<HalfNativeBranchSeal, String> {
    let checked_dims = GemmDims::nt(dims)?;
    let (batch, n_in, n_out) = checked_dims.tuple();
    require_half(dx.dtype, "dX")?;
    if dy.dtype != dx.dtype || w.dtype != dx.dtype {
        return Err("gemm_bi_backward_dx_typed_native: mixed dtypes not supported".into());
    }
    let dt = dx.dtype;
    let alpha: f32 = 1.0;

    // GEMV N=1 (outer product).
    if n_out == 1 && batch >= 1 && n_in >= 1 {
        let m_i = checked_dims.m_i32;
        let k_i = checked_dims.k_i32;
        let ldx_i = checked_dims.k_i32;
        let ldy_i: i32 = 1;
        let total = checked_dims.mk_u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let choice = HalfKernelChoice::new(
            "gemm_bi_nt_gemv",
            environment.kernels.gemm_bi_nt_gemv_typed.get(dt),
        );
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&dx.ptr);
        b.arg(&dy.ptr);
        b.arg(&w.ptr);
        b.arg(&alpha);
        b.arg(&m_i);
        b.arg(&k_i);
        b.arg(&ldx_i);
        b.arg(&ldy_i);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Nt,
                    dtype: dt,
                    dims,
                    strides: (dims.2, dims.2, dims.1),
                    tile: (1, 1),
                    bk_stages: (1, 1),
                    arguments: HalfGemmArguments {
                        output: dx.ptr,
                        a: dy.ptr,
                        b: w.ptr,
                        bias: 0,
                    },
                },
                format_args!("gemm_bi_nt_gemv typed"),
            )
        }?;
        return Ok(seal);
    }

    // Narrow reduction N (2..=127).
    if (2..=127).contains(&n_out) && batch >= 1 && n_in >= 1 {
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_out_i = checked_dims.k_i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                checked_tile_grid(checked_dims.m_u32, 64, checked_dims.k_u32, 32)?,
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let choice = HalfKernelChoice::new(
            "gemm_bi_nt_narrow",
            environment.kernels.gemm_bi_nt_narrow_typed.get(dt),
        );
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&dx.ptr);
        b.arg(&dy.ptr);
        b.arg(&w.ptr);
        b.arg(&alpha);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_out_i);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Nt,
                    dtype: dt,
                    dims,
                    strides: (dims.2, dims.2, dims.1),
                    tile: (64, 32),
                    bk_stages: (16, 1),
                    arguments: HalfGemmArguments {
                        output: dx.ptr,
                        a: dy.ptr,
                        b: w.ptr,
                        bias: 0,
                    },
                },
                format_args!("gemm_bi_nt_narrow typed"),
            )
        }?;
        return Ok(seal);
    }

    // Native typed Big NT overwrites the typed dX output.
    if nt_routes_to_big(
        batch,
        n_in,
        n_out,
        environment.kernels.multiprocessor_count(),
    ) {
        let alpha: f32 = 1.0;
        let m_i = checked_dims.m_i32;
        let n_i = checked_dims.n_i32;
        let k_out_i = checked_dims.k_i32;
        let total_tiles = checked_tile_grid(checked_dims.m_u32, 128, checked_dims.k_u32, 128)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 34 * 1024,
        };
        let choice = HalfKernelChoice::new(
            "gemm_bi_nt_big",
            environment.kernels.gemm_bi_nt_big_typed.get(dt),
        );
        let mut b = environment.stream.launch_builder(choice.function);
        b.arg(&dx.ptr);
        b.arg(&dy.ptr);
        b.arg(&w.ptr);
        b.arg(&alpha);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_out_i);
        let seal = unsafe {
            enqueue_half_gemm(
                &mut environment.observer,
                environment.kernels,
                &mut b,
                cfg,
                HalfGemmObservation {
                    base: choice.base,
                    op: ResolvedGemmOp::Nt,
                    dtype: dt,
                    dims,
                    strides: (dims.2, dims.2, dims.1),
                    tile: (128, 128),
                    bk_stages: (16, 2),
                    arguments: HalfGemmArguments {
                        output: dx.ptr,
                        a: dy.ptr,
                        b: w.ptr,
                        bias: 0,
                    },
                },
                format_args!("gemm_bi_nt_big typed"),
            )
        }?;
        return Ok(seal);
    }

    Err(format!(
        "UNCOVERED gemm_bi_backward_dx_typed_native: split-N/Slim buckets are upcast-fallback territory — \
         shape M={batch} K={n_in} N={n_out}."
    ))
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_forward_tc_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &mut O,
    ops: &TcFwdOperands,
    dims: (usize, usize, usize),
) -> Result<(TcTile, HalfNativeBranchSeal), String> {
    let mut environment = HalfLaunchEnvironment::observed(ctx, observer);
    gemm_bi_forward_tc_in(&mut environment, ops.y, ops.x, ops.w, ops.bias_ptr, dims)
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_forward_tc_with_tile_observed<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &mut O,
    ops: &TcFwdOperands,
    shape: F32TriadShape,
    tile: TcTile,
) -> Result<HalfNativeBranchSeal, String> {
    let mut environment = HalfLaunchEnvironment::observed(ctx, observer);
    gemm_bi_forward_tc_with_tile_in(&mut environment, ops, shape, tile)
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_backward_dw_tc_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &mut O,
    dw_ptr: CUptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(TcTile, HalfNativeBranchSeal), String> {
    let mut environment = HalfLaunchEnvironment::observed(ctx, observer);
    gemm_bi_backward_dw_tc_in(
        &mut environment,
        dw_ptr,
        dy,
        x_saved,
        dims,
        ctx.half_triad_policy(),
    )
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_backward_dw_tc_with_tile_observed<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &mut O,
    dw_ptr: CUptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
    tile: TcTile,
) -> Result<HalfNativeBranchSeal, String> {
    let mut environment = HalfLaunchEnvironment::observed(ctx, observer);
    gemm_bi_backward_dw_tc_with_tile_in(&mut environment, dw_ptr, dy, x_saved, dims, tile)
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_backward_dx_tc_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &mut O,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(TcTile, HalfNativeBranchSeal), String> {
    let mut environment = HalfLaunchEnvironment::observed(ctx, observer);
    gemm_bi_backward_dx_tc_in(&mut environment, dx, dy, w, dims)
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_backward_dx_tc_with_tile_observed<
    O: PhysicalLaunchObserver,
>(
    ctx: &GpuCtx,
    observer: &mut O,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
    tile: TcTile,
) -> Result<HalfNativeBranchSeal, String> {
    let mut environment = HalfLaunchEnvironment::observed(ctx, observer);
    gemm_bi_backward_dx_tc_with_tile_in(&mut environment, dx, dy, w, dims, tile)
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_forward_typed_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &mut O,
    ops: &TcFwdOperands,
    dims: (usize, usize, usize),
) -> Result<HalfNativeBranchSeal, String> {
    let mut environment = HalfLaunchEnvironment::observed(ctx, observer);
    gemm_bi_forward_typed_in(&mut environment, ops.y, ops.x, ops.w, ops.bias_ptr, dims)
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_backward_dw_typed_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &mut O,
    dw_ptr: CUptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<HalfNativeBranchSeal, String> {
    let mut environment = HalfLaunchEnvironment::observed(ctx, observer);
    gemm_bi_backward_dw_typed_in(&mut environment, dw_ptr, dy, x_saved, dims)
}

pub(in crate::mamba_ssm::gpu) fn gemm_bi_backward_dx_typed_observed<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    observer: &mut O,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<HalfNativeBranchSeal, String> {
    let mut environment = HalfLaunchEnvironment::observed(ctx, observer);
    gemm_bi_backward_dx_typed_in(&mut environment, dx, dy, w, dims)
}

#[cfg(test)]
mod prepared_f32_launch_tests {
    use super::*;
    use crate::mamba_ssm::gpu::buffers::{
        managed_allocation_epoch_for_ranges, register_managed_allocation_range,
    };
    use crate::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, HalfTriadPolicy};
    use crate::mamba_ssm::gpu::device::GpuDevice;
    use crate::mamba_ssm::gpu::gemm_bi_triad::contract::{
        TF32_NT_SPLITK4_S3_SPEC, TF32_NT_SPLITK4_S4_SPEC, TF32_NT_SPLITK8_S3_SPEC,
        TF32_NT_SPLITK8_S4_SPEC, TF32_SPLITK2_SPEC, TF32_SPLITK4_SPEC,
        allocation_identity_query_count, reset_allocation_identity_query_count,
    };
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, BackendSet, COMPILER_REVISION, COMPOSER_REVISION,
        CompilerIdentity, CudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, GemmPolicy,
        GemmRouteIdentity, ModuleKind, NUMERIC_ABI_REVISION, NumericContractSet, POLICY_REVISION,
        ResolvedGemmOp, SCHEDULE_REVISION, TUNING_TABLE_REVISION, build_artifact_set,
    };
    use std::cell::Cell;

    fn scalar_fixture(
        op: ResolvedGemmOp,
        dims: (usize, usize, usize),
    ) -> (F32TriadRequest, F32TriadOperands, ScalarDispatchPlan) {
        let request = F32TriadRequest {
            op,
            shape: F32TriadShape::contiguous(op, dims),
        };
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: (op == ResolvedGemmOp::Nn).then_some(0x4000),
            alpha: 1.0,
            beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
        };
        let plan = scalar_dispatch_plan(request, 142).unwrap();
        (request, operands, plan)
    }

    #[test]
    fn tf32_splitk_launch_plan_freezes_one_fused_node_and_workspaces() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (64, 1_536, 384)),
        };
        let plan = tf32_splitk_launch_plan(request, &TF32_SPLITK4_SPEC).unwrap();
        assert_eq!(plan.scratch_elements, 98_304);
        assert_eq!(plan.counter_elements, 48);
        assert_eq!(plan.fused.grid_dim, (12, 4, 4));
        assert_eq!(plan.fused.block_dim, (128, 1, 1));
        assert_eq!(plan.fused.shared_mem_bytes, 29_696);

        let oversized = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (4_096, 128, 1_024)),
        };
        assert!(tf32_splitk_launch_plan(oversized, &TF32_SPLITK4_SPEC).is_err());
        let oversized_row_grid = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (1_048_561, 1, 1)),
        };
        assert!(tf32_splitk_launch_plan(oversized_row_grid, &TF32_SPLITK4_SPEC).is_err());
        let wrong_op = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (64, 1_536, 384)),
        };
        assert!(tf32_splitk_launch_plan(wrong_op, &TF32_SPLITK4_SPEC).is_err());
    }

    #[test]
    fn tf32_splitk2_launch_plan_and_route_freeze_two_partitions() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (64, 833, 384)),
        };
        let operands = scalar_fixture(ResolvedGemmOp::Nn, (64, 833, 384)).1;
        let plan = tf32_splitk_launch_plan(request, &TF32_SPLITK2_SPEC).unwrap();
        assert_eq!(plan.scratch_elements, 49_152);
        assert_eq!(plan.counter_elements, 48);
        assert_eq!(plan.fused.grid_dim, (12, 4, 2));
        assert_eq!(plan.fused.block_dim, (128, 1, 1));
        assert_eq!(plan.fused.shared_mem_bytes, 29_696);

        let splitk2 = tf32_splitk_resolved_routes(
            request,
            operands,
            &TF32_SPLITK2_SPEC,
            portable_binding(),
            [9; 32],
            plan,
        );
        let splitk4_plan = tf32_splitk_launch_plan(request, &TF32_SPLITK4_SPEC).unwrap();
        let splitk4 = tf32_splitk_resolved_routes(
            request,
            operands,
            &TF32_SPLITK4_SPEC,
            portable_binding(),
            [9; 32],
            splitk4_plan,
        );
        assert_eq!(splitk2.len(), 1);
        assert_eq!(splitk2[0].backend, PhysicalGemmBackend::MmaTf32RnaSplitK2V1);
        assert_eq!(
            splitk2[0].numeric_contract,
            ResolvedNumericContract::MmaTf32RnaSplitK2V1
        );
        assert_eq!(
            splitk2[0].ownership,
            ResolvedOutputOwnership::LastCtaPerOutputTileFixedSplitK2ReduceV1
        );
        assert_ne!(splitk2[0].backend, splitk4[0].backend);
        assert_ne!(splitk2[0].numeric_contract, splitk4[0].numeric_contract);
        assert_ne!(splitk2[0].ownership, splitk4[0].ownership);
        assert_ne!(
            splitk2[0].launch.arguments_digest,
            splitk4[0].launch.arguments_digest
        );
    }

    #[test]
    fn tf32_nt_splitk_live_and_reverse_plans_freeze_p4_and_p8_geometry() {
        let cases = [
            ((64, 384, 1_536), (12, 4, 4), (12, 2, 8)),
            ((384, 64, 1_536), (2, 24, 4), (2, 12, 8)),
        ];
        for (dims, p4_grid, p8_grid) in cases {
            let request = F32TriadRequest {
                op: ResolvedGemmOp::Nt,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
            };
            for spec in [&TF32_NT_SPLITK4_S3_SPEC, &TF32_NT_SPLITK4_S4_SPEC] {
                let plan = tf32_splitk_launch_plan(request, spec).unwrap();
                assert_eq!(plan.scratch_elements, 98_304);
                assert_eq!(plan.counter_elements, 48);
                assert_eq!(plan.fused.grid_dim, p4_grid);
                assert_eq!(plan.fused.block_dim, (128, 1, 1));
                assert_eq!(plan.fused.shared_mem_bytes, spec.dynamic_shared_bytes);
            }
            for spec in [&TF32_NT_SPLITK8_S3_SPEC, &TF32_NT_SPLITK8_S4_SPEC] {
                let plan = tf32_splitk_launch_plan(request, spec).unwrap();
                assert_eq!(plan.scratch_elements, 196_608);
                assert_eq!(plan.counter_elements, 24);
                assert_eq!(plan.fused.grid_dim, p8_grid);
                assert_eq!(plan.fused.block_dim, (128, 1, 1));
                assert_eq!(plan.fused.shared_mem_bytes, spec.dynamic_shared_bytes);
            }
        }
    }

    #[test]
    fn tf32_nt_splitk8_has_distinct_numeric_and_ownership_identity() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (64, 384, 1_536)),
        };
        let operands = scalar_fixture(ResolvedGemmOp::Nt, (64, 384, 1_536)).1;
        let p4_plan = tf32_splitk_launch_plan(request, &TF32_NT_SPLITK4_S3_SPEC).unwrap();
        let p8_plan = tf32_splitk_launch_plan(request, &TF32_NT_SPLITK8_S3_SPEC).unwrap();
        let p4 = tf32_splitk_resolved_routes(
            request,
            operands,
            &TF32_NT_SPLITK4_S3_SPEC,
            portable_binding(),
            [9; 32],
            p4_plan,
        );
        let p8 = tf32_splitk_resolved_routes(
            request,
            operands,
            &TF32_NT_SPLITK8_S3_SPEC,
            portable_binding(),
            [9; 32],
            p8_plan,
        );
        assert_eq!(p8[0].backend, PhysicalGemmBackend::MmaTf32RnaSplitK8V1);
        assert_eq!(
            p8[0].numeric_contract,
            ResolvedNumericContract::MmaTf32RnaSplitK8V1
        );
        assert_eq!(
            p8[0].ownership,
            ResolvedOutputOwnership::LastCtaPerOutputTileFixedSplitK8ReduceV1
        );
        assert_eq!(p8[0].tile, (32, 32));
        assert_eq!(p8[0].launch.grid_dim, (12, 2, 8));
        assert_ne!(p4[0].backend, p8[0].backend);
        assert_ne!(p4[0].numeric_contract, p8[0].numeric_contract);
        assert_ne!(p4[0].ownership, p8[0].ownership);
        assert_ne!(p4[0].launch.arguments_digest, p8[0].launch.arguments_digest);
    }

    fn sm120_context_fixture() -> GemmRouteIdentity {
        let compiler = CompilerIdentity {
            source_digest: [1; 32],
            invocation_digest: [2; 32],
            header_manifest_digest: [3; 32],
            target: CudaTarget::new("compute_120").unwrap(),
            nvrtc_version: (12, 8),
            nvrtc_library_domain: [4; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        let artifact = |module_kind, seed| ArtifactIdentity {
            module_kind,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: [seed; 32],
            artifact_digest: [seed + 1; 32],
        };
        GemmRouteIdentity {
            policy: GemmPolicy {
                batch_invariant: true,
                bi_tensor_cores: true,
                fast_gemm: false,
                cublas_tf32: false,
                f32_triad_policy: F32TriadPolicy::ExactScalarFmaV1,
                half_triad_policy: HalfTriadPolicy::TiledParityV1,
                bi_gemm_family: BiGemmFamily::Triad,
            },
            backend_set: BackendSet::TRIAD,
            numeric_contracts: NumericContractSet::TRIAD_MMA_SYNC_V1,
            compiler,
            artifacts: build_artifact_set(&[
                artifact(ModuleKind::Fixed, 5),
                artifact(ModuleKind::TriadScalar, 7),
                artifact(ModuleKind::TriadSm80, 9),
                artifact(ModuleKind::TriadSm120, 11),
            ])
            .unwrap(),
            policy_revision: POLICY_REVISION,
            policy_hash: [13; 32],
            device: DeviceIdentity {
                compute_capability: (12, 0),
                multiprocessor_count: 24,
                target: CudaTarget::new("sm_120").unwrap(),
                driver: DriverIdentity {
                    api_version: 12_800,
                    build_sources: 1,
                    build_digest: [14; 32],
                },
            },
            device_caps: DeviceCaps {
                compute_capability: (12, 0),
                nvrtc_version: (12, 8),
                accepted_target: Some(CudaTarget::new("compute_120").unwrap()),
                optin_shared_bytes: 101_376,
                tensor_map_access: true,
            },
            tuning_table_revision: TUNING_TABLE_REVISION,
            schedule_set_revision: SCHEDULE_REVISION,
            state_capacity: 64,
        }
    }

    fn sm120_key_fixture() -> (Sm120ForcedRoute, Sm120AutoRequest, Sm120PreparedKey) {
        let route = SM120_AUTO_CELLS_CC120[0];
        let request = Sm120AutoRequest {
            op: route.op,
            dtype: route.dtype,
            shape: route.shape,
            a_ptr: 0x1_0000,
            b_ptr: 0x2_0000,
            multiprocessors: 170,
            half_policy: HalfTriadPolicy::TiledParityV1,
            operands: Sm120LaunchOperands {
                output_ptr: 0x3_0000,
                bias_ptr: 0x4_0000,
                alpha: 1.0,
                beta: 0.0,
            },
        };
        let key = Sm120PreparedKey::new(sm120_context_fixture(), route, request);
        (route, request, key)
    }

    #[test]
    fn f32_prepared_key_seals_context_request_pointers_and_scalar_bits() {
        let context = sm120_context_fixture();
        let context_token = 7;
        let policy = context.policy;
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (49, 65, 129)),
        };
        let operands = F32TriadOperands {
            output: 0x1_0000,
            a: 0x2_0000,
            b: 0x3_0000,
            bias: Some(0x4_0000),
            alpha: 1.0,
            beta: 0.0,
        };
        let expected = PreparedF32Key::new(
            context_token,
            policy,
            F32PreparedSelection::Automatic,
            request,
            operands,
        );
        let assert_distinct = |context_token, policy, selection, request, operands| {
            assert_ne!(
                PreparedF32Key::new(context_token, policy, selection, request, operands),
                expected
            );
        };

        assert_distinct(
            context_token + 1,
            policy,
            F32PreparedSelection::Automatic,
            request,
            operands,
        );
        for changed_policy in [
            GemmPolicy {
                batch_invariant: !policy.batch_invariant,
                ..policy
            },
            GemmPolicy {
                bi_tensor_cores: !policy.bi_tensor_cores,
                ..policy
            },
            GemmPolicy {
                fast_gemm: !policy.fast_gemm,
                ..policy
            },
            GemmPolicy {
                cublas_tf32: !policy.cublas_tf32,
                ..policy
            },
            GemmPolicy {
                f32_triad_policy: F32TriadPolicy::AllowDeterministicTf32V1,
                ..policy
            },
            GemmPolicy {
                half_triad_policy: HalfTriadPolicy::AllowStreamKFixedOrderV1,
                ..policy
            },
            GemmPolicy {
                bi_gemm_family: BiGemmFamily::Fixed,
                ..policy
            },
        ] {
            assert_distinct(
                context_token,
                changed_policy,
                F32PreparedSelection::Automatic,
                request,
                operands,
            );
        }
        assert_distinct(
            context_token,
            policy,
            F32PreparedSelection::ExactScalar,
            request,
            operands,
        );
        let forced = Tf32PhysicalRoute::Sm120TmaFmaExactV1(Sm120FmaRoute {
            tile: Sm120FmaTile::M128N64,
            kvec: false,
            splits: 1,
        });
        assert_distinct(
            context_token,
            policy,
            F32PreparedSelection::Forced(forced),
            request,
            operands,
        );
        assert_ne!(
            PreparedF32Key::new(
                context_token,
                policy,
                F32PreparedSelection::Forced(forced),
                request,
                operands,
            ),
            PreparedF32Key::new(
                context_token,
                policy,
                F32PreparedSelection::Forced(Tf32PhysicalRoute::Sm120TmaFmaExactV1(
                    Sm120FmaRoute {
                        tile: Sm120FmaTile::M64N128,
                        kvec: false,
                        splits: 1,
                    },
                )),
                request,
                operands,
            ),
        );
        let changed_request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (49, 65, 129)),
        };
        assert_distinct(
            context_token,
            policy,
            F32PreparedSelection::Automatic,
            changed_request,
            operands,
        );
        for pointer in 0..4 {
            let mut changed = operands;
            match pointer {
                0 => changed.output += 16,
                1 => changed.a += 16,
                2 => changed.b += 16,
                _ => changed.bias = changed.bias.map(|bias| bias + 16),
            }
            assert_distinct(
                context_token,
                policy,
                F32PreparedSelection::Automatic,
                request,
                changed,
            );
        }
        let changed_bias_nullness = F32TriadOperands {
            bias: None,
            ..operands
        };
        assert_distinct(
            context_token,
            policy,
            F32PreparedSelection::Automatic,
            request,
            changed_bias_nullness,
        );
        let changed_alpha = F32TriadOperands {
            alpha: f32::from_bits(operands.alpha.to_bits() ^ 1),
            ..operands
        };
        assert_distinct(
            context_token,
            policy,
            F32PreparedSelection::Automatic,
            request,
            changed_alpha,
        );
        let changed_beta = F32TriadOperands {
            beta: -0.0,
            ..operands
        };
        assert_distinct(
            context_token,
            policy,
            F32PreparedSelection::Automatic,
            request,
            changed_beta,
        );
    }

    #[test]
    fn sm120_prepared_key_seals_route_context_pointers_and_scalar_bits() {
        let (route, request, expected) = sm120_key_fixture();
        let context = sm120_context_fixture();
        let assert_distinct = |route, request, context| {
            assert_ne!(Sm120PreparedKey::new(context, route, request), expected);
        };

        let mut changed_route = route;
        changed_route.physical.stages = Sm120Stages::S3;
        assert_distinct(changed_route, request, context);
        let mut changed_context = context;
        changed_context.policy_hash[0] ^= 1;
        assert_distinct(route, request, changed_context);
        for pointer in 0..4 {
            let mut changed = request;
            match pointer {
                0 => changed.a_ptr += 128,
                1 => changed.b_ptr += 128,
                2 => changed.operands.output_ptr += 128,
                _ => changed.operands.bias_ptr += 128,
            }
            assert_distinct(route, changed, context);
        }
        let mut changed_alpha = request;
        changed_alpha.operands.alpha = f32::from_bits(request.operands.alpha.to_bits() ^ 1);
        assert_distinct(route, changed_alpha, context);
        let mut changed_beta = request;
        changed_beta.operands.beta = -0.0;
        assert_distinct(route, changed_beta, context);
    }

    #[test]
    fn sm120_cache_action_is_prepared_only_and_fail_closed_during_capture() {
        assert_eq!(
            sm120_cache_action(true, false, Sm120ManagedEpochState::Missing),
            Sm120CacheAction::CaptureMissing
        );
        assert_eq!(
            sm120_cache_action(true, true, Sm120ManagedEpochState::Stale),
            Sm120CacheAction::CaptureStale
        );
        assert_eq!(
            sm120_cache_action(true, true, Sm120ManagedEpochState::Untracked),
            Sm120CacheAction::CaptureUntracked
        );
        assert_eq!(
            sm120_cache_action(true, true, Sm120ManagedEpochState::Current),
            Sm120CacheAction::UsePrepared
        );
        assert_eq!(
            sm120_cache_action(false, false, Sm120ManagedEpochState::Missing),
            Sm120CacheAction::Prepare
        );
        assert_eq!(
            sm120_cache_action(false, true, Sm120ManagedEpochState::Stale),
            Sm120CacheAction::Validate
        );
        assert_eq!(
            sm120_capture_cache_error(Sm120CacheAction::CaptureMissing),
            "prepared SM120 Triad cache entry is missing during graph capture; run eager warmup again"
        );
        assert_eq!(
            sm120_capture_cache_error(Sm120CacheAction::CaptureStale),
            "prepared SM120 Triad allocation epoch changed during graph capture; run eager warmup again"
        );
        assert_eq!(
            sm120_capture_cache_error(Sm120CacheAction::CaptureUntracked),
            "prepared SM120 Triad automatic capture requires managed allocations; run eager warmup again"
        );
    }

    #[test]
    fn physical_tf32_route_preserves_semantic_arguments_while_projecting_resources() {
        let source = include_str!("launch.rs");
        let start = source
            .find("fn physical_prepared_f32_route(")
            .expect("physical F32 route helper");
        let end = source[start..]
            .find("pub(in crate::mamba_ssm::gpu) fn prepare_f32_triad(")
            .map(|offset| start + offset)
            .expect("prepare entry after physical route helper");
        let helper = &source[start..end];
        let required = [
            "physical.resources_digest = prepared.resources.physical_digest();",
            ".map(F32PreparedTensorMaps::physical_identity_digest)",
            "physical.tensor_maps_digest = maps_digest;",
        ];
        for fragment in required {
            assert!(helper.contains(fragment), "missing {fragment}");
            let mutated = helper.replacen(fragment, "physical_digest_step_removed();", 1);
            assert!(
                required.iter().any(|required| !mutated.contains(required)),
                "physical TF32 route accepted removal of {fragment}"
            );
        }
        assert!(
            !helper.contains("physical.launch.arguments_digest ="),
            "physical TF32 projection must preserve the semantic kernel-argument digest"
        );
        let production = &source[..source.find("#[cfg(test)]").expect("test module boundary")];
        assert_eq!(
            production
                .matches("physical_prepared_f32_route(prepared,")
                .count(),
            4,
            "eager observation and direct graph preparation must share the physical route"
        );
    }

    #[test]
    fn tf32_argument_digests_bind_bias_nullness() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (64, 1_536, 384)),
        };
        let without_bias = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let with_bias = F32TriadOperands {
            bias: Some(0x4000),
            ..without_bias
        };
        assert_ne!(
            tf32_kernel_arguments_digest(request, without_bias, "tf32", [0; 32]),
            tf32_kernel_arguments_digest(request, with_bias, "tf32", [0; 32])
        );
        assert_ne!(
            tf32_splitk_arguments_digest(request, without_bias, "splitk", 0, 4, 98_304),
            tf32_splitk_arguments_digest(request, with_bias, "splitk", 0, 4, 98_304)
        );
    }

    #[test]
    fn specialized_device_identity_preserves_the_loaded_kernel_sm_count() {
        let target = CudaTarget::new("sm_100").unwrap();
        let driver = DriverIdentity {
            api_version: 13_020,
            build_sources: 1,
            build_digest: [9; 32],
        };

        let identity = specialized_device_identity((10, 0), 73, target, driver);

        assert_eq!(
            identity,
            DeviceIdentity {
                compute_capability: (10, 0),
                multiprocessor_count: 73,
                target,
                driver,
            }
        );
    }

    #[test]
    fn current_managed_epoch_skips_cached_full_validation() {
        let registration = register_managed_allocation_range(0x3301, 0x90_0000, 4096).unwrap();
        let mut stamp = Some(
            managed_allocation_epoch_for_ranges(0x3301, &[(0x90_0000, 4096)])
                .expect("managed allocation stamp"),
        );
        let validations = Cell::new(0);

        refresh_cached_validation(
            &mut stamp,
            || panic!("stable epoch must not query capture status"),
            || {
                validations.set(validations.get() + 1);
                Ok(None)
            },
        )
        .unwrap();

        assert_eq!(validations.get(), 0);
        drop(registration);
    }

    #[test]
    fn changed_managed_epoch_refreshes_once_outside_capture() {
        let first = register_managed_allocation_range(0x3302, 0xa0_0000, 4096).unwrap();
        let mut stamp = Some(
            managed_allocation_epoch_for_ranges(0x3302, &[(0xa0_0000, 4096)])
                .expect("first managed allocation stamp"),
        );
        drop(first);
        let second = register_managed_allocation_range(0x3302, 0xa0_0000, 4096).unwrap();
        let refreshed = managed_allocation_epoch_for_ranges(0x3302, &[(0xa0_0000, 4096)])
            .expect("replacement managed allocation stamp");
        let validations = Cell::new(0);

        refresh_cached_validation(
            &mut stamp,
            || Ok(false),
            || {
                validations.set(validations.get() + 1);
                Ok(Some(refreshed))
            },
        )
        .unwrap();

        assert_eq!(validations.get(), 1);
        assert!(stamp.as_ref().is_some_and(|stamp| stamp.is_current()));
        drop(second);
    }

    #[test]
    fn changed_managed_epoch_requires_warmup_during_capture() {
        let registration = register_managed_allocation_range(0x3303, 0xb0_0000, 4096).unwrap();
        let mut stamp = Some(
            managed_allocation_epoch_for_ranges(0x3303, &[(0xb0_0000, 4096)])
                .expect("managed allocation stamp"),
        );
        drop(registration);

        let error = refresh_cached_validation(
            &mut stamp,
            || Ok(true),
            || panic!("capture must not run full allocation validation"),
        )
        .expect_err("stale managed epoch must reject capture");

        assert!(error.contains("eager warmup"), "{error}");
    }

    #[test]
    fn bounded_cache_sweep_removes_stale_entries_before_insertion() {
        let mut entries = HashMap::from([(1_u32, true), (2, false)]);

        make_room_in_bounded_cache(&mut entries, &3, F32_PREPARED_CACHE_LIMIT, |current| {
            *current
        });

        assert_eq!(entries, HashMap::from([(1, true)]));
    }

    #[test]
    fn bounded_cache_clears_before_a_distinct_entry_exceeds_the_limit() {
        let mut entries = (0..F32_PREPARED_CACHE_LIMIT)
            .map(|key| (key, true))
            .collect::<HashMap<_, _>>();

        make_room_in_bounded_cache(
            &mut entries,
            &F32_PREPARED_CACHE_LIMIT,
            F32_PREPARED_CACHE_LIMIT,
            |current| *current,
        );

        assert!(entries.is_empty());
    }

    #[test]
    fn untracked_cached_resources_keep_full_validation() {
        let mut stamp = None;
        let validations = Cell::new(0);
        for _ in 0..2 {
            refresh_cached_validation(
                &mut stamp,
                || panic!("untracked validation keeps the existing capture behavior"),
                || {
                    validations.set(validations.get() + 1);
                    Ok(None)
                },
            )
            .unwrap();
        }
        assert_eq!(validations.get(), 2);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn managed_f32_cache_hit_skips_allocation_identity_queries() {
        let device = GpuDevice::new(0).expect("open CUDA device");
        let ctx = GpuCtx::new(&device).expect("create GPU context");
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        let dims = (8, 8, 8);
        let x = GpuBuffer::from_cpu(&ctx.stream, &vec![0.25; dims.0 * dims.1]).expect("allocate X");
        let w = GpuBuffer::from_cpu(&ctx.stream, &vec![0.5; dims.1 * dims.2]).expect("allocate W");
        let mut y = GpuBuffer::zeros(&ctx.stream, dims.0 * dims.2).expect("allocate Y");
        ctx.stream.synchronize().expect("finish allocations");

        reset_allocation_identity_query_count();
        launch_cached_f32_forward(&ctx, &mut y, &x, w.cached_ptr(), 0, dims)
            .expect("warm managed f32 cache");
        assert!(allocation_identity_query_count() >= 3);

        reset_allocation_identity_query_count();
        launch_cached_f32_forward(&ctx, &mut y, &x, w.cached_ptr(), 0, dims)
            .expect("hit managed f32 cache");
        assert_eq!(allocation_identity_query_count(), 0);
        ctx.stream.synchronize().expect("finish cached launches");
    }

    fn alignment_fixture_values(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 17 + salt * 13) % 31) as f32 / 32.0 - 0.5)
            .collect()
    }

    fn assert_managed_nn_b_subview(policy: F32TriadPolicy) {
        let device = GpuDevice::new(0).expect("open CUDA device");
        assert_eq!(device.compute_capability, (8, 9), "Ada alignment gate");
        let ctx = GpuCtx::new(&device).expect("create GPU context");
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(policy);
        let dims = (1024, 16, 128);
        let x = GpuBuffer::from_cpu(&ctx.stream, &alignment_fixture_values(dims.0 * dims.1, 3))
            .expect("allocate X");
        let weights = alignment_fixture_values(dims.1 * dims.2, 11);
        let aligned_w = GpuBuffer::from_cpu(&ctx.stream, &weights).expect("allocate aligned W");
        let mut shifted_weights = vec![123.0; weights.len() + 1];
        shifted_weights[1..].copy_from_slice(&weights);
        let shifted_w =
            GpuBuffer::from_cpu(&ctx.stream, &shifted_weights).expect("allocate shifted W");
        let mut expected =
            GpuBuffer::zeros(&ctx.stream, dims.0 * dims.2).expect("allocate expected output");
        let mut actual =
            GpuBuffer::zeros(&ctx.stream, dims.0 * dims.2).expect("allocate actual output");
        ctx.stream
            .synchronize()
            .expect("finish fixture allocations");

        launch_cached_f32_forward(&ctx, &mut expected, &x, aligned_w.cached_ptr(), 0, dims)
            .expect("launch aligned reference");
        ctx.stream.synchronize().expect("finish aligned reference");
        let expected = expected
            .to_cpu(&ctx.stream)
            .expect("download aligned output");

        let trace = ctx
            .record_eager_gemm_trace(|| {
                launch_cached_f32_forward(
                    &ctx,
                    &mut actual,
                    &x,
                    shifted_w.raw_ptr_at(&ctx.stream, 1),
                    0,
                    dims,
                )
            })
            .expect("record shifted-B launch");
        let symbols = trace
            .routes()
            .iter()
            .map(|route| route.symbol)
            .collect::<Vec<_>>();
        assert_eq!(symbols, ["gemm_bi_nn_slim"]);
        ctx.stream.synchronize().expect("finish shifted-B launch");
        let actual = actual
            .to_cpu(&ctx.stream)
            .expect("download shifted-B output");
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires an SM89 CUDA device"]
    fn exact_scalar_prepared_nn_accepts_managed_b_plus_one() {
        assert_managed_nn_b_subview(F32TriadPolicy::ExactScalarFmaV1);
    }

    #[test]
    #[ignore = "requires an SM89 CUDA device"]
    fn automatic_scalar_prepared_nn_accepts_managed_b_plus_one() {
        assert_managed_nn_b_subview(F32TriadPolicy::AllowDeterministicTf32V1);
    }

    #[test]
    #[ignore = "requires an SM89 CUDA device"]
    fn operand_aware_auto_prepares_and_repeats_the_measured_tf32_route() {
        let device = GpuDevice::new(0).expect("open CUDA device");
        assert_eq!(device.compute_capability, (8, 9), "Ada route gate");
        let ctx = GpuCtx::new(&device).expect("create GPU context");
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        let dims = (49, 65, 129);
        let a = GpuBuffer::from_cpu(&ctx.stream, &alignment_fixture_values(dims.0 * dims.1, 5))
            .expect("allocate A");
        let b = GpuBuffer::from_cpu(&ctx.stream, &alignment_fixture_values(dims.1 * dims.2, 17))
            .expect("allocate B");
        let mut output = GpuBuffer::zeros(&ctx.stream, dims.0 * dims.2).expect("allocate output");
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
        };
        let operands = F32TriadOperands {
            output: output.cached_ptr(),
            a: a.cached_ptr(),
            b: b.cached_ptr(),
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let prepared = prepare_f32_triad(&ctx, request, operands).expect("prepare TF32 AUTO");
        assert!(matches!(
            prepared.kind,
            PreparedF32Kind::Tf32 {
                route: Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
                    tile: Tf32PortableTile::M16N32,
                    stages: Tf32PortableStages::S4,
                }),
                ..
            }
        ));
        assert_eq!(
            prepared
                .routes
                .iter()
                .map(|route| route.symbol)
                .collect::<Vec<_>>(),
            ["gemm_bi_nn_sm80_mma_tf32_v1_m16n32_bk32_s4"]
        );

        let mut repeat_bits = None;
        for _ in 0..2 {
            output
                .upload(&ctx.stream, &vec![0.0; dims.0 * dims.2])
                .expect("reset output");
            unsafe {
                launch_prepared_f32_triad(&ctx, &prepared, |_| {
                    Err("TF32 AUTO unexpectedly requested scalar execution".into())
                })
            }
            .expect("launch prepared TF32 AUTO");
            ctx.stream.synchronize().expect("finish TF32 AUTO");
            let bits = output
                .to_cpu(&ctx.stream)
                .expect("download TF32 AUTO output")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            if let Some(expected) = repeat_bits.replace(bits.clone()) {
                assert_eq!(bits, expected, "TF32 AUTO repeat changed output bits");
            }
        }
    }

    fn launch_scalar_nn_slim_raw(
        ctx: &GpuCtx,
        output: CUptr,
        a: CUptr,
        b: CUptr,
        dims: (usize, usize, usize),
        config: cudarc::driver::LaunchConfig,
    ) -> Result<(), String> {
        let (m, k, n) = dims;
        let bias = 0_u64;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let m = i32::try_from(m).expect("M fits i32");
        let k = i32::try_from(k).expect("K fits i32");
        let n = i32::try_from(n).expect("N fits i32");
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.gemm_bi_nn_slim);
        builder.arg(&output);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&bias);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m);
        builder.arg(&n);
        builder.arg(&k);
        builder.arg(&k);
        builder.arg(&n);
        builder.arg(&n);
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|error| format!("launch gemm_bi_nn_slim alignment fixture: {error:?}"))
    }

    #[test]
    #[ignore = "requires an SM89 CUDA device"]
    fn exact_scalar_prepared_nn_accepts_managed_output_plus_one() {
        let device = GpuDevice::new(0).expect("open CUDA device");
        assert_eq!(device.compute_capability, (8, 9), "Ada alignment gate");
        let ctx = GpuCtx::new(&device).expect("create GPU context");
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        let dims = (1024, 16, 128);
        let a = GpuBuffer::from_cpu(&ctx.stream, &alignment_fixture_values(dims.0 * dims.1, 5))
            .expect("allocate A");
        let b = GpuBuffer::from_cpu(&ctx.stream, &alignment_fixture_values(dims.1 * dims.2, 17))
            .expect("allocate B");
        let aligned =
            GpuBuffer::zeros(&ctx.stream, dims.0 * dims.2).expect("allocate aligned output");
        let shifted = GpuBuffer::zeros(&ctx.stream, 1 + dims.0 * dims.2 + 8)
            .expect("allocate shifted output");
        ctx.stream
            .synchronize()
            .expect("finish fixture allocations");

        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
        };
        let output = shifted.raw_ptr_at(&ctx.stream, 1);
        let operands = F32TriadOperands {
            output,
            a: a.cached_ptr(),
            b: b.cached_ptr(),
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let prepared = prepare_f32_triad(&ctx, request, operands)
            .expect("prepare exact scalar output subview");
        let symbols = prepared
            .routes
            .iter()
            .map(|route| route.symbol)
            .collect::<Vec<_>>();
        assert_eq!(symbols, ["gemm_bi_nn_slim"]);
        assert!(matches!(
            prepared.kind,
            PreparedF32Kind::Scalar(ScalarDispatchPlan::NnFinal { slim: true })
        ));
        let route = prepared.routes[0];
        let config = cudarc::driver::LaunchConfig {
            grid_dim: route.launch.grid_dim,
            block_dim: route.launch.block_dim,
            shared_mem_bytes: route.launch.shared_mem_bytes,
        };

        launch_scalar_nn_slim_raw(
            &ctx,
            aligned.cached_ptr(),
            a.cached_ptr(),
            b.cached_ptr(),
            dims,
            config,
        )
        .expect("launch aligned output reference");
        ctx.stream
            .synchronize()
            .expect("finish aligned output reference");
        let expected = aligned
            .to_cpu(&ctx.stream)
            .expect("download aligned output");

        launch_scalar_nn_slim_raw(&ctx, output, a.cached_ptr(), b.cached_ptr(), dims, config)
            .expect("launch shifted output");
        ctx.stream
            .synchronize()
            .expect("finish shifted output launch");
        let shifted = shifted
            .to_cpu(&ctx.stream)
            .expect("download shifted output");
        assert_eq!(shifted[0].to_bits(), 0.0_f32.to_bits());
        assert!(
            shifted[1 + dims.0 * dims.2..]
                .iter()
                .all(|value| value.to_bits() == 0.0_f32.to_bits())
        );
        assert_eq!(
            shifted[1..1 + dims.0 * dims.2]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum ScalarAlignmentOperand {
        Output,
        A,
        B,
        Bias,
    }

    #[derive(Clone, Copy, Debug)]
    struct ScalarAlignmentCase {
        op: ResolvedGemmOp,
        dims: (usize, usize, usize),
        slim: bool,
        shifted: ScalarAlignmentOperand,
    }

    #[derive(Clone, Copy)]
    struct ScalarAlignmentPointers {
        output: CUptr,
        a: CUptr,
        b: CUptr,
        bias: Option<CUptr>,
    }

    fn scalar_alignment_extents(
        op: ResolvedGemmOp,
        dims: (usize, usize, usize),
    ) -> (usize, usize, usize, usize) {
        let (m, k, n) = dims;
        match op {
            ResolvedGemmOp::Nn => (m * k, k * n, m * n, n),
            ResolvedGemmOp::Tn => (m * k, m * n, k * n, 0),
            ResolvedGemmOp::Nt => (m * n, k * n, m * k, 0),
        }
    }

    fn launch_scalar_alignment_case(
        ctx: &GpuCtx,
        case: ScalarAlignmentCase,
        pointers: ScalarAlignmentPointers,
    ) -> Result<(), String> {
        let (m, k, n) = case.dims;
        let m_i = i32::try_from(m).expect("M fits i32");
        let k_i = i32::try_from(k).expect("K fits i32");
        let n_i = i32::try_from(n).expect("N fits i32");
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let bias = pointers.bias.unwrap_or(0);
        let bn = if case.slim { 64_u32 } else { 128_u32 };
        let threads = if case.slim { 128_u32 } else { 256_u32 };
        let (rows, columns) = match case.op {
            ResolvedGemmOp::Nn => (m, n),
            ResolvedGemmOp::Tn => (k, n),
            ResolvedGemmOp::Nt => (m, k),
        };
        let grid = u32::try_from(rows.div_ceil(128) * columns.div_ceil(bn as usize))
            .expect("alignment fixture grid fits u32");
        let shared_mem_bytes = match (case.op, case.slim) {
            (_, true) => 0,
            (ResolvedGemmOp::Nt, false) => SCALAR_BIG_NT_DYNAMIC_SHARED_BYTES,
            (_, false) => 34 * 1024,
        };
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes,
        };
        let function = match (case.op, case.slim) {
            (ResolvedGemmOp::Nn, false) => &ctx.kernels.gemm_bi_nn,
            (ResolvedGemmOp::Nn, true) => &ctx.kernels.gemm_bi_nn_slim,
            (ResolvedGemmOp::Tn, false) => &ctx.kernels.gemm_bi_tn,
            (ResolvedGemmOp::Tn, true) => &ctx.kernels.gemm_bi_tn_slim,
            (ResolvedGemmOp::Nt, false) => &ctx.kernels.gemm_bi_nt,
            (ResolvedGemmOp::Nt, true) => &ctx.kernels.gemm_bi_nt_slim,
        };
        let mut builder = ctx.stream.launch_builder(function);
        builder.arg(&pointers.output);
        builder.arg(&pointers.a);
        builder.arg(&pointers.b);
        if case.op == ResolvedGemmOp::Nn {
            builder.arg(&bias);
            builder.arg(&alpha);
            builder.arg(&beta);
            builder.arg(&m_i);
            builder.arg(&n_i);
            builder.arg(&k_i);
            builder.arg(&k_i);
            builder.arg(&n_i);
            builder.arg(&n_i);
        } else if case.op == ResolvedGemmOp::Tn {
            builder.arg(&alpha);
            builder.arg(&m_i);
            builder.arg(&k_i);
            builder.arg(&n_i);
        } else {
            builder.arg(&alpha);
            builder.arg(&m_i);
            builder.arg(&n_i);
            builder.arg(&k_i);
        }
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|error| format!("launch scalar alignment case {case:?}: {error:?}"))
    }

    fn shifted_alignment_storage(values: &[f32], guard: f32) -> Vec<f32> {
        let mut storage = Vec::with_capacity(values.len() + 2);
        storage.push(guard);
        storage.extend_from_slice(values);
        storage.push(guard);
        storage
    }

    fn assert_scalar_alignment_case(ctx: &GpuCtx, case: ScalarAlignmentCase) {
        let (a_len, b_len, output_len, bias_len) = scalar_alignment_extents(case.op, case.dims);
        let a_values = alignment_fixture_values(a_len, 31);
        let b_values = alignment_fixture_values(b_len, 37);
        let output_values = if case.op == ResolvedGemmOp::Tn {
            alignment_fixture_values(output_len, 41)
        } else {
            vec![0.0; output_len]
        };
        let bias_values = alignment_fixture_values(bias_len, 43);
        let guard = 19.25_f32;

        let reference_a =
            GpuBuffer::from_cpu(&ctx.stream, &a_values).expect("allocate reference A");
        let reference_b =
            GpuBuffer::from_cpu(&ctx.stream, &b_values).expect("allocate reference B");
        let reference_bias = (!bias_values.is_empty()).then(|| {
            GpuBuffer::from_cpu(&ctx.stream, &bias_values).expect("allocate reference bias")
        });
        let reference_output =
            GpuBuffer::from_cpu(&ctx.stream, &output_values).expect("allocate reference output");

        let mut actual_a = GpuBuffer::from_cpu(
            &ctx.stream,
            &if matches!(case.shifted, ScalarAlignmentOperand::A) {
                shifted_alignment_storage(&a_values, guard)
            } else {
                a_values.clone()
            },
        )
        .expect("allocate actual A");
        let mut actual_b = GpuBuffer::from_cpu(
            &ctx.stream,
            &if matches!(case.shifted, ScalarAlignmentOperand::B) {
                shifted_alignment_storage(&b_values, guard)
            } else {
                b_values.clone()
            },
        )
        .expect("allocate actual B");
        let mut actual_output = GpuBuffer::from_cpu(
            &ctx.stream,
            &if matches!(case.shifted, ScalarAlignmentOperand::Output) {
                shifted_alignment_storage(&output_values, guard)
            } else {
                output_values.clone()
            },
        )
        .expect("allocate actual output");
        let mut actual_bias = (!bias_values.is_empty()).then(|| {
            GpuBuffer::from_cpu(
                &ctx.stream,
                &if matches!(case.shifted, ScalarAlignmentOperand::Bias) {
                    shifted_alignment_storage(&bias_values, guard)
                } else {
                    bias_values.clone()
                },
            )
            .expect("allocate actual bias")
        });

        launch_scalar_alignment_case(
            ctx,
            case,
            ScalarAlignmentPointers {
                output: reference_output.cached_ptr(),
                a: reference_a.cached_ptr(),
                b: reference_b.cached_ptr(),
                bias: reference_bias.as_ref().map(GpuBuffer::cached_ptr),
            },
        )
        .expect("launch aligned scalar reference");
        ctx.stream.synchronize().expect("finish scalar reference");
        let expected = reference_output
            .to_cpu(&ctx.stream)
            .expect("download scalar reference");

        let actual_a_pointer = actual_a.cached_ptr()
            + u64::from(matches!(case.shifted, ScalarAlignmentOperand::A)) * 4;
        let actual_b_pointer = actual_b.cached_ptr()
            + u64::from(matches!(case.shifted, ScalarAlignmentOperand::B)) * 4;
        let actual_output_pointer = actual_output.cached_ptr()
            + u64::from(matches!(case.shifted, ScalarAlignmentOperand::Output)) * 4;
        let actual_bias_pointer = actual_bias.as_ref().map(|bias| {
            bias.cached_ptr() + u64::from(matches!(case.shifted, ScalarAlignmentOperand::Bias)) * 4
        });
        let mut first = None;
        for _ in 0..2 {
            actual_a
                .upload(
                    &ctx.stream,
                    &if matches!(case.shifted, ScalarAlignmentOperand::A) {
                        shifted_alignment_storage(&a_values, guard)
                    } else {
                        a_values.clone()
                    },
                )
                .expect("restore actual A");
            actual_b
                .upload(
                    &ctx.stream,
                    &if matches!(case.shifted, ScalarAlignmentOperand::B) {
                        shifted_alignment_storage(&b_values, guard)
                    } else {
                        b_values.clone()
                    },
                )
                .expect("restore actual B");
            actual_output
                .upload(
                    &ctx.stream,
                    &if matches!(case.shifted, ScalarAlignmentOperand::Output) {
                        shifted_alignment_storage(&output_values, guard)
                    } else {
                        output_values.clone()
                    },
                )
                .expect("restore actual output");
            if let Some(bias) = actual_bias.as_mut() {
                bias.upload(
                    &ctx.stream,
                    &if matches!(case.shifted, ScalarAlignmentOperand::Bias) {
                        shifted_alignment_storage(&bias_values, guard)
                    } else {
                        bias_values.clone()
                    },
                )
                .expect("restore actual bias");
            }
            launch_scalar_alignment_case(
                ctx,
                case,
                ScalarAlignmentPointers {
                    output: actual_output_pointer,
                    a: actual_a_pointer,
                    b: actual_b_pointer,
                    bias: actual_bias_pointer,
                },
            )
            .expect("launch shifted scalar case");
            ctx.stream
                .synchronize()
                .expect("finish shifted scalar case");
            let storage = actual_output
                .to_cpu(&ctx.stream)
                .expect("download shifted scalar output");
            let active = if matches!(case.shifted, ScalarAlignmentOperand::Output) {
                assert_eq!(storage[0].to_bits(), guard.to_bits(), "{case:?} prefix");
                assert_eq!(
                    storage[output_len + 1].to_bits(),
                    guard.to_bits(),
                    "{case:?} suffix"
                );
                storage[1..output_len + 1].to_vec()
            } else {
                storage
            };
            assert_eq!(
                active
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{case:?}"
            );
            if let Some(previous) = first.replace(active) {
                assert_eq!(
                    first.as_ref().expect("second run"),
                    &previous,
                    "{case:?} is not deterministic"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires an SM89 CUDA device"]
    fn scalar_f32_big_and_slim_accept_every_four_byte_aligned_external_base() {
        let device = GpuDevice::new(0).expect("open CUDA device");
        assert_eq!(device.compute_capability, (8, 9), "Ada alignment gate");
        let ctx = GpuCtx::new(&device).expect("create GPU context");
        let dims = (128, 64, 128);
        for slim in [false, true] {
            for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
                let operands: &[ScalarAlignmentOperand] = match op {
                    ResolvedGemmOp::Nn => &[
                        ScalarAlignmentOperand::Output,
                        ScalarAlignmentOperand::A,
                        ScalarAlignmentOperand::B,
                        ScalarAlignmentOperand::Bias,
                    ],
                    ResolvedGemmOp::Tn | ResolvedGemmOp::Nt => &[
                        ScalarAlignmentOperand::Output,
                        ScalarAlignmentOperand::A,
                        ScalarAlignmentOperand::B,
                    ],
                };
                for &shifted in operands {
                    assert_scalar_alignment_case(
                        &ctx,
                        ScalarAlignmentCase {
                            op,
                            dims,
                            slim,
                            shifted,
                        },
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_physical_plan_freezes_maximum_node_counts_and_neighbors() {
        for (op, dims, expected) in [
            (ResolvedGemmOp::Nn, (32, 32, 128), 2),
            (ResolvedGemmOp::Tn, (256, 128, 128), 2),
            (ResolvedGemmOp::Nt, (32, 95, 128), 34),
            (ResolvedGemmOp::Nt, (32, 96, 128), 3),
            (ResolvedGemmOp::Nt, (31, 95, 128), 1),
            (ResolvedGemmOp::Nt, (32, 95, 127), 1),
            (ResolvedGemmOp::Nt, (32, 95, 129), 1),
        ] {
            let (request, operands, plan) = scalar_fixture(op, dims);
            let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
            assert_eq!(nodes.len(), expected, "{op:?} {dims:?} {plan:?}");
        }
    }

    #[test]
    fn scalar_nt_thin_output_cells_have_the_qualified_splitk_node_identity() {
        for (dims, expected_grids) in [
            ((512, 16, 2048), [(64, 1, 1), (1024, 1, 1), (32, 1, 1)]),
            ((16, 512, 2048), [(64, 16, 1), (512, 1, 1), (32, 1, 1)]),
        ] {
            let (request, operands, plan) = scalar_fixture(ResolvedGemmOp::Nt, dims);
            assert_eq!(
                plan,
                ScalarDispatchPlan::NtSplitKMain {
                    n_main: 2048,
                    n_tail: 0,
                }
            );
            let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
            assert_eq!(nodes.len(), 3);
            assert_eq!(
                nodes.iter().map(|node| node.symbol).collect::<Vec<_>>(),
                [
                    "gemm_bi_transpose_f32_2d",
                    "gemm_bi_nn_splitk32_partial",
                    "gemm_bi_splitk_reduce",
                ]
            );
            assert_eq!(
                nodes
                    .iter()
                    .map(|node| node.launch.grid_dim)
                    .collect::<Vec<_>>(),
                expected_grids
            );
            for index in 0..nodes.len() {
                assert!(
                    nodes[..index].iter().all(|node| {
                        node.launch.arguments_digest != nodes[index].launch.arguments_digest
                    }),
                    "{dims:?} physical argument digest collision at node {index}"
                );
            }
            assert_ne!(
                scalar_route_contract(nodes[1].symbol),
                scalar_route_contract(nodes[2].symbol),
                "{dims:?} partial and reducer identities"
            );
        }
    }

    #[test]
    fn scalar_big_nt_uses_its_exact_shared_memory_contract() {
        for (op, dims, expected_plan, symbol, shared_mem_bytes) in [
            (
                ResolvedGemmOp::Nn,
                (2048, 128, 1024),
                ScalarDispatchPlan::NnFinal { slim: false },
                "gemm_bi_nn",
                34 * 1024,
            ),
            (
                ResolvedGemmOp::Tn,
                (128, 1024, 1024),
                ScalarDispatchPlan::TnFinal { slim: false },
                "gemm_bi_tn_aligned",
                34 * 1024,
            ),
            (
                ResolvedGemmOp::Nt,
                (2048, 1024, 129),
                ScalarDispatchPlan::NtFinal { slim: false },
                "gemm_bi_nt",
                33_376,
            ),
            (
                ResolvedGemmOp::Nt,
                (2048, 512, 129),
                ScalarDispatchPlan::NtFinal { slim: true },
                "gemm_bi_nt_slim",
                0,
            ),
        ] {
            let (request, operands, plan) = scalar_fixture(op, dims);
            assert_eq!(plan, expected_plan);
            let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
            assert_eq!(nodes.len(), 1);
            assert_eq!(nodes[0].symbol, symbol);
            assert_eq!(nodes[0].launch.shared_mem_bytes, shared_mem_bytes);
        }
    }

    #[test]
    fn scalar_nn_m64n64_plan_has_the_qualified_seven_cell_identity() {
        let operands = F32TriadOperands {
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let mut digests = Vec::new();
        for (dims, expected_grid) in [
            ((2_048, 3_072, 768), 384),
            ((4_096, 3_072, 1_536), 1_536),
            ((512, 3_072, 768), 96),
            ((4_096, 512, 768), 768),
            ((2_048, 768, 3_072), 1_536),
            ((2_048, 1_536, 768), 384),
            ((4_621, 384, 1_928), 2_263),
        ] {
            let request = F32TriadRequest {
                op: ResolvedGemmOp::Nn,
                shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
            };
            let nodes =
                scalar_physical_nodes(request, operands, ScalarDispatchPlan::NnM64N64Qualified)
                    .unwrap();
            assert_eq!(nodes.len(), 1);
            assert_eq!(nodes[0].symbol, "gemm_bi_nn_m64n64_bk16_s2_v1");
            assert_eq!(nodes[0].tile, (64, 64));
            assert_eq!((nodes[0].bk, nodes[0].stages), (16, 2));
            assert_eq!(nodes[0].launch.grid_dim, (expected_grid, 1, 1));
            assert_eq!(nodes[0].launch.block_dim, (128, 1, 1));
            assert_eq!(nodes[0].launch.shared_mem_bytes, 17_408);
            assert_ne!(nodes[0].launch.arguments_digest, [0; 32]);
            assert!(!digests.contains(&nodes[0].launch.arguments_digest));
            digests.push(nodes[0].launch.arguments_digest);
        }
        assert_eq!(
            scalar_plan_fields(ScalarDispatchPlan::NnM64N64Qualified),
            (24, 0, 0)
        );
        assert_eq!(
            scalar_argument_layout(
                F32TriadRequest {
                    op: ResolvedGemmOp::Nn,
                    shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (2_048, 768, 3_072),),
                },
                operands,
                ScalarDispatchPlan::NnM64N64Qualified,
                0,
            )
            .null_pointer_mask,
            0b1000,
        );
    }

    #[test]
    fn scalar_nn_m32n64_splitk32_plan_has_two_exact_ordered_nodes() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (128, 8_192, 128)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nn, 1.0);
        let plan = ScalarDispatchPlan::NnM32N64SplitK32Qualified;
        let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes.iter().map(|node| node.symbol).collect::<Vec<_>>(),
            [
                "gemm_bi_nn_splitk32_m32n64_exact_v1",
                "gemm_bi_splitk_reduce",
            ]
        );
        assert_eq!(nodes[0].tile, (32, 64));
        assert_eq!((nodes[0].bk, nodes[0].stages), (32, 1));
        assert_eq!(nodes[0].launch.grid_dim, (2_048, 1, 1));
        assert_eq!(nodes[0].launch.block_dim, (128, 1, 1));
        assert_eq!(nodes[0].launch.shared_mem_bytes, 0);
        assert_eq!(nodes[1].launch.grid_dim, (64, 1, 1));
        assert_eq!(nodes[1].launch.block_dim, (256, 1, 1));
        assert_eq!(nodes[1].launch.shared_mem_bytes, 0);
        assert_ne!(nodes[0].launch.arguments_digest, [0; 32]);
        assert_ne!(nodes[1].launch.arguments_digest, [0; 32]);
        assert_ne!(
            nodes[0].launch.arguments_digest,
            nodes[1].launch.arguments_digest
        );
        assert_eq!(scalar_plan_fields(plan), (36, 0, 0));
        assert!(scalar_plan_requires_zero_beta(plan));
        assert!(plan.needs_split_scratch());
        assert!(!plan.needs_transpose_scratch());
        let partial_elements = request.shape.k / 32 * request.shape.m * request.shape.n;
        assert_eq!(partial_elements, 4_194_304);
        assert!(partial_elements <= SPLITK_SCRATCH_CAP);
        assert_eq!(
            scalar_route_contract(nodes[0].symbol),
            (
                PhysicalGemmBackend::ScalarFmaSplitKPartialV1,
                ResolvedNumericContract::ScalarFmaSplitKPartialV1,
                ResolvedOutputOwnership::OneCtaPerOutputTilePerSplitKPartitionV1,
            )
        );
    }

    #[test]
    fn scalar_nt_d768_transpose_plan_has_two_exact_ordered_nodes() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 768, 3_072)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 0.0);
        let plan = ScalarDispatchPlan::NtD768TransposeM64N64Qualified;
        let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes.iter().map(|node| node.symbol).collect::<Vec<_>>(),
            [
                "gemm_bi_transpose_f32_32x16_d768_v1",
                "gemm_bi_nn_m64n64_bk16_s2_v1",
            ]
        );
        assert_eq!(nodes[0].tile, (32, 32));
        assert_eq!((nodes[0].bk, nodes[0].stages), (1, 1));
        assert_eq!(nodes[0].launch.grid_dim, (96, 24, 1));
        assert_eq!(nodes[0].launch.block_dim, (32, 16, 1));
        assert_eq!(nodes[0].launch.shared_mem_bytes, 0);
        assert_eq!(nodes[1].tile, (64, 64));
        assert_eq!((nodes[1].bk, nodes[1].stages), (16, 2));
        assert_eq!(nodes[1].launch.grid_dim, (384, 1, 1));
        assert_eq!(nodes[1].launch.block_dim, (128, 1, 1));
        assert_eq!(
            nodes[1].launch.shared_mem_bytes,
            crate::mamba_ssm::gpu::gemm_bi_triad::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES
        );
        assert_ne!(nodes[0].launch.arguments_digest, [0; 32]);
        assert_ne!(nodes[1].launch.arguments_digest, [0; 32]);
        assert_ne!(
            nodes[0].launch.arguments_digest,
            nodes[1].launch.arguments_digest
        );
        assert_eq!(scalar_plan_fields(plan), (25, 0, 0));
        assert!(scalar_plan_requires_zero_beta(plan));
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 0),
            ScalarArgumentLayout::default()
        );
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 1).null_pointer_mask,
            0b1000
        );
    }

    #[test]
    fn scalar_nt_d768_out_transpose_plan_has_distinct_tag_and_exact_nodes() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 0.0);
        let plan = ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified;
        let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes.iter().map(|node| node.symbol).collect::<Vec<_>>(),
            [
                "gemm_bi_transpose_f32_32x16_d768_v1",
                "gemm_bi_nn_m64n64_bk16_s2_v1",
            ]
        );
        assert_eq!(nodes[0].launch.grid_dim, (24, 48, 1));
        assert_eq!(nodes[0].launch.block_dim, (32, 16, 1));
        assert_eq!(nodes[0].launch.shared_mem_bytes, 0);
        assert_eq!(nodes[1].launch.grid_dim, (768, 1, 1));
        assert_eq!(nodes[1].launch.block_dim, (128, 1, 1));
        assert_eq!(
            nodes[1].launch.shared_mem_bytes,
            crate::mamba_ssm::gpu::gemm_bi_triad::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES
        );
        assert_ne!(nodes[0].launch.arguments_digest, [0; 32]);
        assert_ne!(nodes[1].launch.arguments_digest, [0; 32]);
        assert_ne!(
            nodes[0].launch.arguments_digest,
            nodes[1].launch.arguments_digest
        );
        assert_eq!(scalar_plan_fields(plan), (26, 0, 0));
        assert!(scalar_plan_requires_zero_beta(plan));
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 0),
            ScalarArgumentLayout::default()
        );
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 1).null_pointer_mask,
            0b1000
        );
    }

    #[test]
    fn scalar_nt_large_deep_transpose_plan_has_the_qualified_revision_and_two_exact_nodes() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_096, 3_072, 1_536)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let plan = ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified;
        let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes.iter().map(|node| node.symbol).collect::<Vec<_>>(),
            [
                "gemm_bi_transpose_f32_32x16_d768_v1",
                "gemm_bi_nn_m64n64_bk16_s2_v1",
            ]
        );
        assert_eq!(nodes[0].tile, (32, 32));
        assert_eq!((nodes[0].bk, nodes[0].stages), (1, 1));
        assert_eq!(nodes[0].launch.grid_dim, (48, 96, 1));
        assert_eq!(nodes[0].launch.block_dim, (32, 16, 1));
        assert_eq!(nodes[0].launch.shared_mem_bytes, 0);
        assert_eq!(nodes[1].tile, (64, 64));
        assert_eq!((nodes[1].bk, nodes[1].stages), (16, 2));
        assert_eq!(nodes[1].launch.grid_dim, (3_072, 1, 1));
        assert_eq!(nodes[1].launch.block_dim, (128, 1, 1));
        assert_eq!(
            nodes[1].launch.shared_mem_bytes,
            crate::mamba_ssm::gpu::gemm_bi_triad::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES
        );
        assert_ne!(nodes[0].launch.arguments_digest, [0; 32]);
        assert_ne!(nodes[1].launch.arguments_digest, [0; 32]);
        assert_ne!(
            nodes[0].launch.arguments_digest,
            nodes[1].launch.arguments_digest
        );
        assert_eq!(scalar_plan_fields(plan), (27, 0, 0));
        assert!(scalar_plan_requires_zero_beta(plan));
        assert_eq!(scalar_node_count(plan), 2);
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 0),
            ScalarArgumentLayout::default()
        );
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 1).null_pointer_mask,
            0b1000
        );
    }

    #[test]
    fn scalar_nt_prism_vector_plan_has_the_qualified_revision_and_two_exact_nodes() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_621, 384, 1_928)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let plan = ScalarDispatchPlan::NtPrismVectorQualified;
        let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes.iter().map(|node| node.symbol).collect::<Vec<_>>(),
            [
                "gemm_bi_transpose_f32_32x16_d768_v1",
                "gemm_bi_nn_prism_m64n64_bk16_s2_v1",
            ]
        );
        assert_eq!(nodes[0].tile, (32, 32));
        assert_eq!((nodes[0].bk, nodes[0].stages), (1, 1));
        assert_eq!(nodes[0].launch.grid_dim, (61, 12, 1));
        assert_eq!(nodes[0].launch.block_dim, (32, 16, 1));
        assert_eq!(nodes[0].launch.shared_mem_bytes, 0);
        assert_eq!(nodes[1].tile, (64, 64));
        assert_eq!((nodes[1].bk, nodes[1].stages), (16, 2));
        assert_eq!(nodes[1].launch.grid_dim, (438, 1, 1));
        assert_eq!(nodes[1].launch.block_dim, (128, 1, 1));
        assert_eq!(
            nodes[1].launch.shared_mem_bytes,
            crate::mamba_ssm::gpu::gemm_bi_triad::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES
        );
        assert_ne!(nodes[0].launch.arguments_digest, [0; 32]);
        assert_ne!(nodes[1].launch.arguments_digest, [0; 32]);
        assert_ne!(
            nodes[0].launch.arguments_digest,
            nodes[1].launch.arguments_digest
        );
        assert_eq!(scalar_plan_fields(plan), (30, 0, 0));
        assert!(scalar_plan_requires_zero_beta(plan));
        assert_eq!(scalar_node_count(plan), 2);
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 0),
            ScalarArgumentLayout::default()
        );
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 1).null_pointer_mask,
            0b1000
        );
        assert_eq!(
            scalar_transpose_scratch_elements(request, plan).unwrap(),
            Some(740_352)
        );
        assert_eq!(
            crate::mamba_ssm::gpu::gemm_bi_triad::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS,
            4_718_592
        );
    }

    #[test]
    fn scalar_nt_d128_out_transpose_plan_has_the_qualified_revision_and_two_exact_nodes() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (1_024, 256, 128)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let plan = ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified;
        let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes.iter().map(|node| node.symbol).collect::<Vec<_>>(),
            [
                "gemm_bi_transpose_f32_32x16_d768_v1",
                "gemm_bi_nn_m64n64_bk16_s2_v1",
            ]
        );
        assert_eq!(
            (nodes[0].tile, nodes[0].bk, nodes[0].stages),
            ((32, 32), 1, 1)
        );
        assert_eq!(nodes[0].launch.grid_dim, (4, 8, 1));
        assert_eq!(nodes[0].launch.block_dim, (32, 16, 1));
        assert_eq!(nodes[0].launch.shared_mem_bytes, 0);
        assert_eq!(
            (nodes[1].tile, nodes[1].bk, nodes[1].stages),
            ((64, 64), 16, 2)
        );
        assert_eq!(nodes[1].launch.grid_dim, (64, 1, 1));
        assert_eq!(nodes[1].launch.block_dim, (128, 1, 1));
        assert_eq!(
            nodes[1].launch.shared_mem_bytes,
            crate::mamba_ssm::gpu::gemm_bi_triad::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES
        );
        assert_ne!(nodes[0].launch.arguments_digest, [0; 32]);
        assert_ne!(nodes[1].launch.arguments_digest, [0; 32]);
        assert_ne!(
            nodes[0].launch.arguments_digest,
            nodes[1].launch.arguments_digest
        );
        assert_eq!(scalar_plan_fields(plan), (29, 0, 0));
        assert!(scalar_plan_requires_zero_beta(plan));
        assert_eq!(scalar_node_count(plan), 2);
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 0),
            ScalarArgumentLayout::default()
        );
        assert_eq!(
            scalar_argument_layout(request, operands, plan, 1).null_pointer_mask,
            0b1000
        );
        assert_eq!(
            scalar_transpose_scratch_elements(request, plan).unwrap(),
            Some(32_768)
        );
    }

    #[test]
    fn prepared_scalar_binding_uses_frozen_digest_and_rejects_symbol_config_or_order_changes() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (1_024, 256, 128)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let nodes = scalar_physical_nodes(
            request,
            operands,
            ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified,
        )
        .unwrap();
        assert_eq!(nodes.len(), 2);
        for node in &nodes {
            let config = cudarc::driver::LaunchConfig {
                grid_dim: node.launch.grid_dim,
                block_dim: node.launch.block_dim,
                shared_mem_bytes: node.launch.shared_mem_bytes,
            };
            assert_eq!(
                validate_prepared_scalar_binding(node.symbol, node.launch, node.symbol, config)
                    .unwrap(),
                node.launch.arguments_digest
            );
            assert_ne!(node.launch.arguments_digest, [0; 32]);

            for wrong_config in [
                cudarc::driver::LaunchConfig {
                    grid_dim: (config.grid_dim.0 + 1, config.grid_dim.1, config.grid_dim.2),
                    ..config
                },
                cudarc::driver::LaunchConfig {
                    block_dim: (
                        config.block_dim.0 + 1,
                        config.block_dim.1,
                        config.block_dim.2,
                    ),
                    ..config
                },
                cudarc::driver::LaunchConfig {
                    shared_mem_bytes: config.shared_mem_bytes + 4,
                    ..config
                },
            ] {
                assert!(
                    validate_prepared_scalar_binding(
                        node.symbol,
                        node.launch,
                        node.symbol,
                        wrong_config,
                    )
                    .is_err()
                );
            }
        }
        let first_config = cudarc::driver::LaunchConfig {
            grid_dim: nodes[0].launch.grid_dim,
            block_dim: nodes[0].launch.block_dim,
            shared_mem_bytes: nodes[0].launch.shared_mem_bytes,
        };
        assert!(
            validate_prepared_scalar_binding(
                nodes[0].symbol,
                nodes[0].launch,
                nodes[1].symbol,
                first_config,
            )
            .is_err()
        );
        let mut zero_digest = nodes[0].launch;
        zero_digest.arguments_digest = [0; 32];
        assert!(
            validate_prepared_scalar_binding(
                nodes[0].symbol,
                zero_digest,
                nodes[0].symbol,
                first_config,
            )
            .is_err()
        );
    }

    #[test]
    fn prepared_scalar_enqueue_does_not_rehash_frozen_arguments() {
        let source = include_str!("launch.rs");
        for (start, end) in [
            (
                "fn enqueue_scalar_forward",
                "fn gemm_bi_forward_sub_with_control",
            ),
            ("fn enqueue_scalar_backward", "/// Weight gradient"),
        ] {
            let body = source
                .split_once(start)
                .and_then(|(_, tail)| tail.split_once(end).map(|(body, _)| body))
                .expect("controlled scalar enqueue body");
            assert!(!body.contains("scalar_arguments_digest"));
            assert!(body.contains("control.enqueue(symbol, config, builder)"));
        }
    }

    #[test]
    fn prepared_cache_lookup_does_not_build_full_route_identity() {
        let source = include_str!("launch.rs");
        let body = source
            .split_once("fn ensure_prepared(")
            .and_then(|(_, tail)| tail.split_once("fn launch<Scalar>(").map(|(body, _)| body))
            .expect("prepared cache lookup body");
        assert!(!body.contains("ctx.gemm_route()"));
        assert!(body.contains("ctx.instance_token()"));
        assert!(body.contains("ctx.gemm_policy()"));
    }

    #[test]
    fn prepared_backward_enqueue_skips_fallback_dispatch() {
        let source = include_str!("launch.rs");
        for (start, end) in [
            ("fn gemm_bi_backward_dw_with_control", "/// Input gradient"),
            (
                "fn gemm_bi_backward_dx_with_control",
                "/// Typed input gradient",
            ),
        ] {
            let body = source
                .split_once(start)
                .and_then(|(_, tail)| tail.split_once(end).map(|(body, _)| body))
                .expect("prepared backward enqueue body");
            assert!(!body.contains("scalar_backward_launch_plan"));
            assert!(body.contains("scalar_backward_request"));
            assert!(body.contains("Some(prepared) => prepared.plan()"));
        }
    }

    #[test]
    fn scalar_nt_d128_out_raw_and_prepared_identity_match() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (1_024, 256, 128)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let raw_plan = scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        let prepared_plan =
            scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        assert_eq!(
            raw_plan,
            ScalarDispatchPlan::NtD128OutTransposeM64N64Qualified
        );
        assert_eq!(prepared_plan, raw_plan);
        assert_eq!(
            scalar_physical_nodes(request, operands, prepared_plan).unwrap(),
            scalar_physical_nodes(request, operands, raw_plan).unwrap()
        );
    }

    #[test]
    fn scalar_tn_m16n16_raw_and_prepared_one_node_identity_match() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, (256, 512, 384)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Tn, 1.0);
        let raw_plan = scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        let prepared_plan =
            scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        assert_eq!(raw_plan, ScalarDispatchPlan::TnM16N16SplitM16Qualified);
        assert_eq!(prepared_plan, raw_plan);

        let raw_nodes = scalar_physical_nodes(request, operands, raw_plan).unwrap();
        let prepared_nodes = scalar_physical_nodes(request, operands, prepared_plan).unwrap();
        assert_eq!(prepared_nodes, raw_nodes);
        assert_eq!(raw_nodes.len(), scalar_node_count(raw_plan));
        assert_eq!(raw_nodes.len(), 1);
        let node = raw_nodes[0];
        assert_eq!(node.symbol, "gemm_bi_tn_m16n16_bk16_s2_splitm16_v1");
        assert_eq!(node.tile, (16, 16));
        assert_eq!((node.bk, node.stages), (16, 2));
        assert_eq!(node.launch.grid_dim, (768, 1, 1));
        assert_eq!(node.launch.block_dim, (64, 1, 1));
        assert_eq!(node.launch.shared_mem_bytes, 4_096);
        assert_ne!(node.launch.arguments_digest, [0; 32]);
        assert_eq!(scalar_plan_fields(raw_plan), (35, 0, 0));
        assert_eq!(
            scalar_argument_layout(request, operands, raw_plan, 0),
            ScalarArgumentLayout::default()
        );
        assert_eq!(
            scalar_route_contract(node.symbol),
            (
                PhysicalGemmBackend::ScalarFmaTnSplitMF64ReduceV1,
                ResolvedNumericContract::ScalarFmaTnSplitMF64ReduceV1,
                ResolvedOutputOwnership::OneCtaPerOutputTileV1,
            )
        );
        assert!(!raw_plan.needs_transpose_scratch());
        assert!(!raw_plan.needs_split_scratch());
    }

    #[test]
    fn scalar_nn_m32n64_splitk32_raw_and_prepared_identity_match() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (128, 8_192, 128)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nn, 1.0);
        let raw_plan = scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        let prepared_plan =
            scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        assert_eq!(raw_plan, ScalarDispatchPlan::NnM32N64SplitK32Qualified);
        assert_eq!(prepared_plan, raw_plan);

        let raw_nodes = scalar_physical_nodes(request, operands, raw_plan).unwrap();
        let prepared_nodes = scalar_physical_nodes(request, operands, prepared_plan).unwrap();
        assert_eq!(prepared_nodes, raw_nodes);
        assert_eq!(raw_nodes.len(), scalar_node_count(raw_plan));
        assert_eq!(raw_nodes.len(), 2);
        assert_eq!(
            raw_nodes.iter().map(|node| node.symbol).collect::<Vec<_>>(),
            [
                "gemm_bi_nn_splitk32_m32n64_exact_v1",
                "gemm_bi_splitk_reduce",
            ]
        );
        assert!(
            raw_nodes
                .iter()
                .all(|node| node.launch.arguments_digest != [0; 32])
        );
        assert_ne!(
            raw_nodes[0].launch.arguments_digest,
            raw_nodes[1].launch.arguments_digest
        );
    }

    #[test]
    fn scalar_nt_m2n16_raw_and_prepared_one_node_identity_match() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (512, 16, 2_048)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let raw_plan = scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        let prepared_plan =
            scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        assert_eq!(raw_plan, ScalarDispatchPlan::NtM2N16SplitK32Qualified);
        assert_eq!(prepared_plan, raw_plan);

        let raw_nodes = scalar_physical_nodes(request, operands, raw_plan).unwrap();
        let prepared_nodes = scalar_physical_nodes(request, operands, prepared_plan).unwrap();
        assert_eq!(prepared_nodes, raw_nodes);
        assert_eq!(raw_nodes.len(), scalar_node_count(raw_plan));
        assert_eq!(raw_nodes.len(), 1);
        let node = raw_nodes[0];
        assert_eq!(node.symbol, "gemm_bi_nt_m2n16_bk64_splitk32_v1");
        assert_eq!(node.tile, (2, 16));
        assert_eq!((node.bk, node.stages), (64, 2));
        assert_eq!(node.launch.grid_dim, (256, 1, 1));
        assert_eq!(node.launch.block_dim, (64, 1, 1));
        assert_eq!(node.launch.shared_mem_bytes, 17_984);
        assert_ne!(node.launch.arguments_digest, [0; 32]);
        assert_eq!(scalar_plan_fields(raw_plan), (31, 0, 0));
        assert!(!raw_plan.needs_transpose_scratch());
        assert!(!raw_plan.needs_split_scratch());
    }

    #[test]
    fn scalar_nt_prism_raw_and_prepared_identity_match() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_621, 384, 1_928)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let raw_plan = scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        let prepared_plan =
            scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        assert_eq!(raw_plan, ScalarDispatchPlan::NtPrismVectorQualified);
        assert_eq!(prepared_plan, raw_plan);
        assert_eq!(
            scalar_physical_nodes(request, operands, prepared_plan).unwrap(),
            scalar_physical_nodes(request, operands, raw_plan).unwrap()
        );
    }

    #[test]
    fn scalar_nt_large_deep_raw_and_prepared_identity_match() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_096, 3_072, 1_536)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let raw_plan = scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        let prepared_plan =
            scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        assert_eq!(
            raw_plan,
            ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified
        );
        assert_eq!(prepared_plan, raw_plan);
        assert_eq!(
            scalar_physical_nodes(request, operands, prepared_plan).unwrap(),
            scalar_physical_nodes(request, operands, raw_plan).unwrap()
        );
    }

    #[test]
    fn scalar_transpose_workspace_extent_is_checked_per_plan() {
        let large_deep = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (4_096, 3_072, 1_536)),
        };
        assert_eq!(
            scalar_transpose_scratch_elements(
                large_deep,
                ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified,
            )
            .unwrap(),
            Some(crate::mamba_ssm::gpu::gemm_bi_triad::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS)
        );
        let oversized = F32TriadRequest {
            shape: F32TriadShape {
                k: large_deep.shape.k + 1,
                ..large_deep.shape
            },
            ..large_deep
        };
        assert!(
            scalar_transpose_scratch_elements(
                oversized,
                ScalarDispatchPlan::NtLargeDeepTransposeM64N64Qualified,
            )
            .is_err()
        );
        assert_eq!(
            scalar_transpose_scratch_elements(
                large_deep,
                ScalarDispatchPlan::NtFinal { slim: false },
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn scalar_nt_d768_out_raw_and_prepared_paths_freeze_the_same_identity() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nt,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nt, (2_048, 1_536, 768)),
        };
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let raw_plan = scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        let prepared_plan =
            scalar_launch_plan(scalar_tn_admission_facts(), request, operands).unwrap();
        assert_eq!(
            raw_plan,
            ScalarDispatchPlan::NtD768OutTransposeM64N64Qualified
        );
        assert_eq!(prepared_plan, raw_plan);

        let raw_nodes = scalar_physical_nodes(request, operands, raw_plan).unwrap();
        let prepared_nodes = scalar_physical_nodes(request, operands, prepared_plan).unwrap();
        assert_eq!(prepared_nodes, raw_nodes);
        assert_eq!(raw_nodes.len(), scalar_node_count(raw_plan));
        assert_eq!(raw_nodes.len(), 2);
        assert_eq!(scalar_plan_fields(raw_plan), (26, 0, 0));
        assert!(raw_plan.needs_transpose_scratch());
        assert!(!raw_plan.needs_split_scratch());
    }

    #[test]
    fn scalar_node_identity_distinguishes_equal_symbol_sequences() {
        let left = scalar_fixture(ResolvedGemmOp::Nn, (32, 64, 128));
        let right = scalar_fixture(ResolvedGemmOp::Nn, (32, 96, 128));
        let left_nodes = scalar_physical_nodes(left.0, left.1, left.2).unwrap();
        let right_nodes = scalar_physical_nodes(right.0, right.1, right.2).unwrap();
        let left_symbols: Vec<_> = left_nodes.iter().map(|node| node.symbol).collect();
        let right_symbols: Vec<_> = right_nodes.iter().map(|node| node.symbol).collect();
        assert_eq!(left_symbols, right_symbols);
        assert_ne!(left_nodes, right_nodes);
        assert_ne!(
            left_nodes[0].launch.arguments_digest,
            right_nodes[0].launch.arguments_digest
        );
    }

    #[test]
    fn scalar_splitk_symbols_have_stage_specific_route_contracts() {
        let partial = (
            PhysicalGemmBackend::ScalarFmaSplitKPartialV1,
            ResolvedNumericContract::ScalarFmaSplitKPartialV1,
            ResolvedOutputOwnership::OneCtaPerOutputTilePerSplitKPartitionV1,
        );
        assert_eq!(partial.0 as u8, 17);
        assert_eq!(partial.1 as u8, 16);
        assert_eq!(partial.2 as u8, 9);
        for symbol in [
            "gemm_bi_nn_splitk32_partial",
            "gemm_bi_nn_splitk_slim_partial",
        ] {
            assert_eq!(scalar_route_contract(symbol), partial, "{symbol}");
        }

        let reducer = (
            PhysicalGemmBackend::ScalarFmaSplitKF32ReduceV1,
            ResolvedNumericContract::ScalarFmaSplitKF32ReduceV1,
            ResolvedOutputOwnership::OneThreadPerOutputElementFixedSplitKReduceV1,
        );
        assert_eq!(reducer.0 as u8, 18);
        assert_eq!(reducer.1 as u8, 17);
        assert_eq!(reducer.2 as u8, 10);
        assert_eq!(scalar_route_contract("gemm_bi_splitk_reduce"), reducer);
        assert_ne!(partial, reducer);
        assert_ne!(
            partial,
            scalar_route_contract("gemm_bi_nn_splitk32_partial_typo")
        );
    }

    #[test]
    fn scalar_argument_layout_tracks_the_physical_pointer_abi() {
        let (nn_request, mut nn_operands, nn_plan) =
            scalar_fixture(ResolvedGemmOp::Nn, (32, 32, 128));
        nn_operands.bias = None;
        assert!(scalar_plan_requires_zero_beta(nn_plan));
        assert_eq!(
            scalar_argument_layout(nn_request, nn_operands, nn_plan, 0),
            ScalarArgumentLayout::default()
        );
        assert_eq!(
            scalar_argument_layout(nn_request, nn_operands, nn_plan, 1).null_pointer_mask,
            0b11100
        );

        let (tn_request, tn_operands, tn_plan) =
            scalar_fixture(ResolvedGemmOp::Tn, (256, 128, 128));
        assert!(!scalar_plan_requires_zero_beta(tn_plan));
        for index in 0..scalar_node_count(tn_plan) {
            assert_eq!(
                scalar_argument_layout(tn_request, tn_operands, tn_plan, index),
                ScalarArgumentLayout::default()
            );
        }

        let (nt_request, nt_operands, nt_plan) = scalar_fixture(ResolvedGemmOp::Nt, (32, 95, 128));
        assert_eq!(
            scalar_argument_layout(nt_request, nt_operands, nt_plan, 2).null_pointer_mask,
            0b11100
        );
        let tail = scalar_argument_layout(nt_request, nt_operands, nt_plan, 3);
        assert_eq!(tail.output_offset, 0);
        assert_eq!(tail.output_column, Some(64));
        assert_eq!(tail.b_offset, 64 * 128 * 4);
        assert_eq!(tail.null_pointer_mask, 0);
    }

    fn portable_binding() -> Tf32MapBinding {
        let target = CudaTarget::new("sm_89").unwrap();
        let nvrtc_version = (13, 2);
        Tf32MapBinding {
            allocation_domain: AllocationDomain {
                context_handle: 7,
                device_ordinal: 0,
            },
            qualified: Tf32QualifiedModule {
                module_kind: ModuleKind::TriadSm80,
                target,
                artifact: ArtifactIdentity {
                    module_kind: ModuleKind::TriadSm80,
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
                    compute_capability: (8, 9),
                    multiprocessor_count: 142,
                    target,
                    driver: DriverIdentity {
                        api_version: 13_020,
                        build_sources: 1,
                        build_digest: [7; 32],
                    },
                },
                device_caps: DeviceCaps {
                    compute_capability: (8, 9),
                    nvrtc_version,
                    accepted_target: Some(target),
                    optin_shared_bytes: 99_000,
                    tensor_map_access: false,
                },
            },
        }
    }

    fn rect_wide_sm120_binding() -> Tf32MapBinding {
        let mut binding = portable_binding();
        let module_target = CudaTarget::new("compute_120").unwrap();
        let device_target = CudaTarget::new("sm_120").unwrap();
        binding.qualified.module_kind = ModuleKind::TriadSm120;
        binding.qualified.target = module_target;
        binding.qualified.artifact.module_kind = ModuleKind::TriadSm120;
        binding.qualified.compiler.target = module_target;
        binding.qualified.device.compute_capability = (12, 0);
        binding.qualified.device.multiprocessor_count = 170;
        binding.qualified.device.target = device_target;
        binding.qualified.device_caps.compute_capability = (12, 0);
        binding.qualified.device_caps.accepted_target = Some(module_target);
        binding.qualified.device_caps.tensor_map_access = true;
        binding
    }

    fn tn_underfill_portable_binding() -> Tf32MapBinding {
        let mut binding = portable_binding();
        let module_target = CudaTarget::new("compute_120").unwrap();
        let device_target = CudaTarget::new("sm_120").unwrap();
        binding.qualified.target = module_target;
        binding.qualified.compiler.target = module_target;
        binding.qualified.device.compute_capability = (12, 0);
        binding.qualified.device.multiprocessor_count = 170;
        binding.qualified.device.target = device_target;
        binding.qualified.device_caps.compute_capability = (12, 0);
        binding.qualified.device_caps.accepted_target = Some(module_target);
        binding
    }

    #[test]
    fn tn_underfill_route_freezes_physical_and_graph_identity() {
        let route = Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        });
        let spec = tf32_kernel_spec(ResolvedGemmOp::Tn, route).unwrap();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Tn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Tn, (256, 512, 384)),
        };
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (384, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 32_768,
        };
        let resolved = tf32_resolved_route(
            request,
            spec,
            tn_underfill_portable_binding(),
            Tf32LaunchDigests {
                maps: [0; 32],
                resources: [2; 32],
                arguments: [3; 32],
            },
            false,
            config,
        );

        assert_eq!(
            resolved.symbol,
            "gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4"
        );
        assert_eq!(resolved.module_kind, ModuleKind::TriadSm80);
        assert_eq!(resolved.backend, PhysicalGemmBackend::MmaTf32RnaV1);
        assert_eq!(
            resolved.numeric_contract,
            ResolvedNumericContract::MmaTf32RnaV1
        );
        assert_eq!(resolved.shape, (256, 512, 384));
        assert_eq!(resolved.strides, (512, 384, 384));
        assert_eq!(resolved.tile, (16, 32));
        assert_eq!(resolved.bk, 32);
        assert_eq!(resolved.stages, 4);
        assert_eq!(resolved.threads, 128);
        assert_eq!(resolved.launch.grid_dim, config.grid_dim);
        assert_eq!(resolved.launch.block_dim, config.block_dim);
        assert_eq!(resolved.launch.shared_mem_bytes, config.shared_mem_bytes);
        assert_eq!(resolved.tensor_maps_digest, [0; 32]);
        assert_eq!(resolved.resources_digest, [2; 32]);
        assert_eq!(resolved.launch.arguments_digest, [3; 32]);
        assert_eq!(resolved.tuning_table_revision, 40);

        let eager = build_resolved_gemm_launch_set(&[resolved]).unwrap();
        let graph = build_resolved_gemm_launch_set(&[resolved]).unwrap();
        assert_eq!(eager, graph);
        assert_ne!(eager.ordered_digest, [0; 32]);
        let mut mutated = resolved;
        mutated.launch.arguments_digest[0] ^= 1;
        assert_ne!(
            eager.ordered_digest,
            build_resolved_gemm_launch_set(&[mutated])
                .unwrap()
                .ordered_digest
        );
    }

    #[test]
    fn rect_wide_route_freezes_physical_and_graph_identity() {
        let route = Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(Tf32Sm120Route {
            tile: Tf32Sm120Tile::M80N32Bk64,
            stages: Tf32Sm120Stages::S2,
        });
        let spec = tf32_kernel_spec(ResolvedGemmOp::Nn, route).unwrap();
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (512, 3_072, 768)),
        };
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (168, 1, 1),
            block_dim: (160, 1, 1),
            shared_mem_bytes: 57_472,
        };
        let resolved = tf32_resolved_route(
            request,
            spec,
            rect_wide_sm120_binding(),
            Tf32LaunchDigests {
                maps: [1; 32],
                resources: [2; 32],
                arguments: [3; 32],
            },
            false,
            config,
        );

        assert_eq!(
            resolved.symbol,
            "gemm_bi_nn_sm120_tma_mma_tf32_v1_m80n32_bk64_s2"
        );
        assert_eq!(resolved.module_kind, ModuleKind::TriadSm120);
        assert_eq!(resolved.backend, PhysicalGemmBackend::Sm120TmaMmaTf32RnaV1);
        assert_eq!(
            resolved.numeric_contract,
            ResolvedNumericContract::Sm120TmaMmaTf32RnaV1
        );
        assert_eq!(resolved.shape, (512, 3_072, 768));
        assert_eq!(resolved.strides, (3_072, 768, 768));
        assert_eq!(resolved.tile, (80, 32));
        assert_eq!(resolved.bk, 64);
        assert_eq!(resolved.stages, 2);
        assert_eq!(resolved.threads, 160);
        assert_eq!(resolved.launch.grid_dim, config.grid_dim);
        assert_eq!(resolved.launch.block_dim, config.block_dim);
        assert_eq!(resolved.launch.shared_mem_bytes, config.shared_mem_bytes);
        assert_eq!(resolved.tensor_maps_digest, [1; 32]);
        assert_eq!(resolved.resources_digest, [2; 32]);
        assert_eq!(resolved.launch.arguments_digest, [3; 32]);
        assert_eq!(resolved.tuning_table_revision, 40);

        let launch_set = build_resolved_gemm_launch_set(&[resolved]).unwrap();
        assert_eq!(launch_set.launch_count, 1);
        assert_ne!(launch_set.ordered_digest, [0; 32]);
        let mut mutated = resolved;
        mutated.launch.arguments_digest[0] ^= 1;
        let mutated_set = build_resolved_gemm_launch_set(&[mutated]).unwrap();
        assert_ne!(launch_set.ordered_digest, mutated_set.ordered_digest);
    }

    fn zero_request(op: ResolvedGemmOp) -> F32TriadRequest {
        let dims = match op {
            ResolvedGemmOp::Nn => (3, 0, 5),
            ResolvedGemmOp::Tn => (0, 3, 5),
            ResolvedGemmOp::Nt => (3, 5, 0),
        };
        F32TriadRequest {
            op,
            shape: F32TriadShape::contiguous(op, dims),
        }
    }

    fn portable_route() -> Tf32PhysicalRoute {
        Tf32PhysicalRoute::MmaTf32RnaV1(Tf32PortableRoute {
            tile: Tf32PortableTile::M16N32,
            stages: Tf32PortableStages::S4,
        })
    }

    #[test]
    fn tf32_kernel_param_abi_matches_cuda() {
        macro_rules! check_abi {
            ($ty:ty, $size:literal, $($field:ident => $offset:literal),+ $(,)?) => {{
                assert_eq!(std::mem::size_of::<$ty>(), $size);
                assert_eq!(std::mem::align_of::<$ty>(), 4);
                $(assert_eq!(std::mem::offset_of!($ty, $field), $offset);)+
            }};
        }
        check_abi!(Sm80Tf32KernelParams, 32,
            alpha => 0, beta => 4, m => 8, k => 12, n => 16, lda => 20, ldb => 24, ldc => 28);
        check_abi!(Sm90aTf32KernelParams, 40,
            a_x => 0, a_y => 4, b_x => 8, b_y => 12, alpha => 16, beta => 20,
            m => 24, k => 28, n => 32, ldc => 36);
        check_abi!(Sm100KernelParams, 40,
            a_x => 0, a_y => 4, b_x => 8, b_y => 12, alpha => 16, beta => 20,
            m => 24, k => 28, n => 32, ldc => 36);
        check_abi!(Sm120KernelParams, 40,
            a_x => 0, a_y => 4, b_x => 8, b_y => 12, alpha => 16, beta => 20,
            m => 24, k => 28, n => 32, ldc => 36);
    }

    #[test]
    fn f32_triad_operand_validation_rejects_null_and_misaligned_pointers() {
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let (request, operands, _) = scalar_fixture(op, (3, 4, 5));
            validate_f32_triad_operands(request, operands)
                .unwrap_or_else(|error| panic!("aligned {op:?} operands were rejected: {error}"));

            validate_f32_triad_operands(
                request,
                F32TriadOperands {
                    b: operands.b + std::mem::size_of::<f32>() as u64,
                    ..operands
                },
            )
            .unwrap_or_else(|error| {
                panic!("4-byte-aligned B+1 f32 {op:?} operand was rejected: {error}")
            });

            for (name, misaligned) in [
                (
                    "A",
                    F32TriadOperands {
                        a: operands.a + 2,
                        ..operands
                    },
                ),
                (
                    "B",
                    F32TriadOperands {
                        b: operands.b + 2,
                        ..operands
                    },
                ),
            ] {
                let error = validate_f32_triad_operands(request, misaligned)
                    .expect_err("misaligned f32 inputs must be rejected");
                assert!(error.contains(&format!("{name} pointer")), "{error}");
            }

            let zero_reduction_operands = F32TriadOperands {
                a: 0,
                b: 0,
                ..operands
            };
            validate_f32_triad_operands(zero_request(op), zero_reduction_operands)
                .unwrap_or_else(|error| panic!("K=0 {op:?} operands were rejected: {error}"));
        }

        let (request, operands, _) = scalar_fixture(ResolvedGemmOp::Nn, (3, 4, 5));
        for output in [0, 0x1002] {
            let error =
                validate_f32_triad_operands(request, F32TriadOperands { output, ..operands })
                    .expect_err("null and misaligned outputs must be rejected");
            assert!(error.contains("output pointer"), "{error}");
        }
        for bias in [0, 0x2002] {
            let error = validate_f32_triad_operands(
                request,
                F32TriadOperands {
                    bias: Some(bias),
                    ..operands
                },
            )
            .expect_err("null and misaligned biases must be rejected");
            assert!(error.contains("bias pointer"), "{error}");
        }
    }

    #[test]
    fn zero_reduction_preparation_never_queries_input_allocations() {
        let a_input_queries = Cell::new(0);
        let b_input_queries = Cell::new(0);
        let tensor_map_plan_queries = Cell::new(0);
        let tensor_map_encodes = Cell::new(0);
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let request = zero_request(op);
            let maps = prepare_f32_maps_with(
                request,
                F32TriadOperands {
                    output: 0x3000,
                    a: 0,
                    b: 0,
                    bias: None,
                    alpha: 1.0,
                    beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
                },
                portable_route(),
                portable_binding(),
                |_, _, _| {
                    a_input_queries.set(a_input_queries.get() + 1);
                    b_input_queries.set(b_input_queries.get() + 1);
                    tensor_map_plan_queries.set(tensor_map_plan_queries.get() + 1);
                    Err("K=0 must not build a tensor-map plan".into())
                },
                |_| {
                    tensor_map_encodes.set(tensor_map_encodes.get() + 1);
                    Err("K=0 must not encode tensor maps".into())
                },
            )
            .unwrap();
            assert!(matches!(
                maps,
                F32PreparedTensorMaps::ZeroReductionV1 { .. }
            ));
        }
        assert_eq!(a_input_queries.get(), 0);
        assert_eq!(b_input_queries.get(), 0);
        assert_eq!(tensor_map_plan_queries.get(), 0);
        assert_eq!(tensor_map_encodes.get(), 0);

        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (3, 4, 5)),
        };
        let error = prepare_f32_maps_with(
            request,
            F32TriadOperands {
                output: 0x3000,
                a: 0x1000,
                b: 0x2000,
                bias: None,
                alpha: 1.0,
                beta: 0.0,
            },
            portable_route(),
            portable_binding(),
            |_, _, _| {
                a_input_queries.set(a_input_queries.get() + 1);
                b_input_queries.set(b_input_queries.get() + 1);
                tensor_map_plan_queries.set(tensor_map_plan_queries.get() + 1);
                Err("nonzero control reached the plan boundary".into())
            },
            |_| {
                tensor_map_encodes.set(tensor_map_encodes.get() + 1);
                Err("nonzero control must stop at the plan boundary".into())
            },
        )
        .unwrap_err();
        assert!(error.contains("plan boundary"));
        assert_eq!(a_input_queries.get(), 1);
        assert_eq!(b_input_queries.get(), 1);
        assert_eq!(tensor_map_plan_queries.get(), 1);
        assert_eq!(tensor_map_encodes.get(), 0);
    }

    fn scalar_test_operands(op: ResolvedGemmOp, alpha: f32) -> F32TriadOperands {
        F32TriadOperands {
            output: 0x3000,
            a: 0x1000,
            b: 0x2000,
            bias: None,
            alpha,
            beta: if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
        }
    }

    fn scalar_tn_admission_facts() -> ScalarLaunchFacts {
        let compiler = CompilerIdentity {
            source_digest: [1; 32],
            invocation_digest: [2; 32],
            header_manifest_digest: [3; 32],
            target: CudaTarget::new("compute_120").unwrap(),
            nvrtc_version: (13, 2),
            nvrtc_library_domain: [4; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        ScalarLaunchFacts {
            scalar_artifact: ArtifactIdentity {
                module_kind: ModuleKind::TriadScalar,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: compiler.invocation_digest,
                artifact_digest: [5; 32],
            },
            scalar_compiler: compiler,
            compute_capability: (12, 0),
            multiprocessor_count: 170,
        }
    }

    #[test]
    fn backward_launch_plan_uses_the_canonical_op_and_contiguous_abi() {
        let (tn_dims, tn_request, tn_plan) =
            scalar_backward_launch_plan(ResolvedGemmOp::Tn, (32, 4, 1), 142).unwrap();
        assert_eq!(
            scalar_backward_request(ResolvedGemmOp::Tn, (32, 4, 1)).unwrap(),
            (tn_dims, tn_request)
        );
        assert_eq!(tn_dims.tuple(), (32, 4, 1));
        assert_eq!(
            tn_request.shape,
            F32TriadShape {
                m: 32,
                k: 4,
                n: 1,
                lda: 4,
                ldb: 1,
                ldc: 1,
            }
        );
        assert_eq!(tn_plan, ScalarDispatchPlan::TnGemv);

        let (nt_dims, nt_request, nt_plan) =
            scalar_backward_launch_plan(ResolvedGemmOp::Nt, (1, 7, 1), 142).unwrap();
        assert_eq!(
            scalar_backward_request(ResolvedGemmOp::Nt, (1, 7, 1)).unwrap(),
            (nt_dims, nt_request)
        );
        assert_eq!(nt_dims.tuple(), (1, 7, 1));
        assert_eq!(
            nt_request.shape,
            F32TriadShape {
                m: 1,
                k: 7,
                n: 1,
                lda: 1,
                ldb: 1,
                ldc: 7,
            }
        );
        assert_eq!(nt_plan, ScalarDispatchPlan::NtGemv);
    }

    #[test]
    fn tn_splitm_plan_matches_enqueue_geometry_at_the_batch_boundary() {
        let (_, request, plan) =
            scalar_backward_launch_plan(ResolvedGemmOp::Tn, (256, 128, 128), 142).unwrap();
        assert_eq!(
            plan,
            ScalarDispatchPlan::TnSplitM {
                m_chunk: 16,
                chunks: 16,
            }
        );
        let nodes =
            scalar_physical_nodes(request, scalar_test_operands(ResolvedGemmOp::Tn, 1.0), plan)
                .unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].symbol, "gemm_bi_tn_splitm_partial_aligned");
        assert_eq!(nodes[0].launch.grid_dim, (1, 1, 16));
        assert_eq!(nodes[0].launch.block_dim, (256, 1, 1));
        assert_eq!(nodes[1].symbol, "gemm_bi_splitm_reduce");
        assert_eq!(nodes[1].launch.grid_dim, (64, 1, 1));
        assert_eq!(nodes[1].launch.block_dim, (256, 1, 1));

        let (_, below_request, below_plan) =
            scalar_backward_launch_plan(ResolvedGemmOp::Tn, (255, 128, 128), 142).unwrap();
        assert_eq!(below_plan, ScalarDispatchPlan::TnFinal { slim: true });
        let below = scalar_physical_nodes(
            below_request,
            scalar_test_operands(ResolvedGemmOp::Tn, 1.0),
            below_plan,
        )
        .unwrap();
        assert_eq!(below.len(), 1);
        assert_eq!(below[0].symbol, "gemm_bi_tn_slim");
    }

    #[test]
    fn tn_narrow_splitm_qualified_manifests_have_exact_two_node_geometry() {
        for (dims, partition, partial_grid, reducer_grid) in [
            ((1024, 47, 17), (32, 32), (1, 1, 32), (4, 1, 1)),
            ((1024, 128, 25), (32, 32), (2, 1, 32), (13, 1, 1)),
            ((4096, 64, 64), (48, 86), (1, 2, 86), (16, 1, 1)),
        ] {
            let (_, request, automatic) =
                scalar_backward_launch_plan(ResolvedGemmOp::Tn, dims, 170).unwrap();
            assert_eq!(automatic, ScalarDispatchPlan::TnNarrow, "{dims:?}");
            let aligned = scalar_test_operands(ResolvedGemmOp::Tn, 1.0);
            let plan = scalar_launch_plan(scalar_tn_admission_facts(), request, aligned).unwrap();
            assert_eq!(
                plan,
                ScalarDispatchPlan::TnNarrowSplitM {
                    m_chunk: partition.0,
                    chunks: partition.1,
                },
                "{dims:?}"
            );

            let aligned_nodes = scalar_physical_nodes(request, aligned, plan).unwrap();
            assert_eq!(aligned_nodes.len(), 2, "{dims:?}");
            assert_eq!(
                aligned_nodes[0].symbol, "gemm_bi_tn_narrow_splitm_partial_aligned",
                "{dims:?}"
            );
            assert_eq!(aligned_nodes[0].launch.grid_dim, partial_grid, "{dims:?}");
            assert_eq!(aligned_nodes[0].launch.block_dim, (128, 1, 1), "{dims:?}");
            assert_eq!(aligned_nodes[1].symbol, "gemm_bi_splitm_reduce", "{dims:?}");
            assert_eq!(aligned_nodes[1].launch.grid_dim, reducer_grid, "{dims:?}");
            assert_eq!(aligned_nodes[1].launch.block_dim, (256, 1, 1), "{dims:?}");

            for unaligned in [
                F32TriadOperands {
                    a: aligned.a + std::mem::size_of::<f32>() as u64,
                    ..aligned
                },
                F32TriadOperands {
                    b: aligned.b + std::mem::size_of::<f32>() as u64,
                    ..aligned
                },
                F32TriadOperands {
                    a: aligned.a + std::mem::size_of::<f32>() as u64,
                    b: aligned.b + std::mem::size_of::<f32>() as u64,
                    ..aligned
                },
            ] {
                let unaligned_nodes = scalar_physical_nodes(request, unaligned, plan).unwrap();
                assert_eq!(unaligned_nodes.len(), 2, "{dims:?}");
                assert_eq!(
                    unaligned_nodes[0].symbol, "gemm_bi_tn_narrow_splitm_partial",
                    "{dims:?}"
                );
                assert_eq!(unaligned_nodes[0].launch.grid_dim, partial_grid, "{dims:?}");
                assert_eq!(
                    unaligned_nodes[1].symbol, "gemm_bi_splitm_reduce",
                    "{dims:?}"
                );
                assert_eq!(unaligned_nodes[1].launch.grid_dim, reducer_grid, "{dims:?}");
            }

            let output_offset = F32TriadOperands {
                output: aligned.output + std::mem::size_of::<f32>() as u64,
                ..aligned
            };
            let output_offset_nodes = scalar_physical_nodes(request, output_offset, plan).unwrap();
            assert_eq!(
                output_offset_nodes[0].symbol, "gemm_bi_tn_narrow_splitm_partial_aligned",
                "{dims:?}"
            );
        }
    }

    #[test]
    fn tn_big_and_splitm_physical_symbols_separate_aligned_hot_paths() {
        for (dims, aligned_symbol, fallback_symbol) in [
            (
                (256, 128, 128),
                "gemm_bi_tn_splitm_partial_aligned",
                "gemm_bi_tn_splitm_partial",
            ),
            ((128, 1024, 1024), "gemm_bi_tn_aligned", "gemm_bi_tn"),
        ] {
            let (_, request, plan) =
                scalar_backward_launch_plan(ResolvedGemmOp::Tn, dims, 142).unwrap();
            let aligned = scalar_test_operands(ResolvedGemmOp::Tn, 1.0);
            let mut unaligned = aligned;
            unaligned.a += std::mem::size_of::<f32>() as u64;

            let aligned_nodes = scalar_physical_nodes(request, aligned, plan).unwrap();
            let unaligned_nodes = scalar_physical_nodes(request, unaligned, plan).unwrap();
            assert_eq!(aligned_nodes[0].symbol, aligned_symbol, "{dims:?}");
            assert_eq!(unaligned_nodes[0].symbol, fallback_symbol, "{dims:?}");
            assert_ne!(
                aligned_nodes[0].launch.arguments_digest,
                unaligned_nodes[0].launch.arguments_digest,
                "{dims:?}"
            );
        }
    }

    #[test]
    fn nt_k_tail_34_node_plan_preserves_repeated_symbol_arguments_and_neighbors() {
        let (_, request, plan) =
            scalar_backward_launch_plan(ResolvedGemmOp::Nt, (32, 95, 128), 142).unwrap();
        assert_eq!(
            plan,
            ScalarDispatchPlan::NtSplitKTail {
                k_main: 64,
                k_tail: 31,
            }
        );
        let operands = scalar_test_operands(ResolvedGemmOp::Nt, 1.0);
        let nodes = scalar_physical_nodes(request, operands, plan).unwrap();
        assert_eq!(nodes.len(), 34);
        assert_eq!(nodes[0].symbol, "gemm_bi_transpose_f32_2d");
        assert_eq!(nodes[0].launch.grid_dim, (4, 2, 1));
        assert_eq!(nodes[0].launch.block_dim, (32, 32, 1));
        assert_eq!(nodes[1].symbol, "gemm_bi_nn_splitk32_partial");
        assert_eq!(nodes[1].launch.grid_dim, (4, 1, 1));
        assert_eq!(nodes[2].symbol, "gemm_bi_splitk_reduce");
        assert_eq!(nodes[2].launch.grid_dim, (8, 1, 1));
        for node in &nodes[3..] {
            assert_eq!(node.symbol, "gemm_bi_dx_col_gemv");
            assert_eq!(node.launch.grid_dim, (1, 1, 1));
            assert_eq!(node.launch.block_dim, (128, 1, 1));
        }
        for (index, node) in nodes[3..].iter().enumerate() {
            assert!(
                nodes[3..index + 3]
                    .iter()
                    .all(|prior| prior.launch.arguments_digest != node.launch.arguments_digest),
                "tail node {index} collided with an earlier physical argument digest"
            );
        }

        let (_, tail_30_request, tail_30_plan) =
            scalar_backward_launch_plan(ResolvedGemmOp::Nt, (32, 94, 128), 142).unwrap();
        assert_eq!(
            scalar_physical_nodes(tail_30_request, operands, tail_30_plan)
                .unwrap()
                .len(),
            33
        );
        let (_, aligned_request, aligned_plan) =
            scalar_backward_launch_plan(ResolvedGemmOp::Nt, (32, 96, 128), 142).unwrap();
        assert!(matches!(
            aligned_plan,
            ScalarDispatchPlan::NtSplitKMain {
                n_main: 128,
                n_tail: 0
            }
        ));
        assert_eq!(
            scalar_physical_nodes(aligned_request, operands, aligned_plan)
                .unwrap()
                .len(),
            3
        );

        for (dims, expected_plan) in [
            ((31, 95, 128), ScalarDispatchPlan::NtSmallBatchWide),
            ((32, 95, 127), ScalarDispatchPlan::NtNarrow),
            ((32, 95, 129), ScalarDispatchPlan::NtMidBatchWide),
        ] {
            let (_, neighbor_request, neighbor_plan) =
                scalar_backward_launch_plan(ResolvedGemmOp::Nt, dims, 142).unwrap();
            assert_eq!(neighbor_plan, expected_plan, "neighbor {dims:?}");
            let neighbor =
                scalar_physical_nodes(neighbor_request, operands, neighbor_plan).unwrap();
            assert_eq!(neighbor.len(), 1, "neighbor {dims:?}");
            assert_eq!(neighbor[0].symbol, "gemm_bi_nt_narrow", "neighbor {dims:?}");
        }

        let alpha_mutation =
            scalar_physical_nodes(request, scalar_test_operands(ResolvedGemmOp::Nt, 2.0), plan)
                .unwrap();
        assert_ne!(
            nodes[0].launch.arguments_digest,
            alpha_mutation[0].launch.arguments_digest
        );
    }
}

#[cfg(test)]
mod half_physical_trace_tests {
    use super::*;

    #[test]
    fn no_physical_observer_is_zero_sized_and_never_records() {
        assert_eq!(std::mem::size_of::<NoPhysicalObserver>(), 0);
        assert_eq!(
            std::mem::size_of::<HalfLaunchEnvironment<'static, NoPhysicalObserver>>(),
            2 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn half_kernel_identity_uses_exact_dtype_suffix_and_module_owner() {
        for (base, module_kind) in [
            ("gemm_bi_nn_gemv", ModuleKind::TriadScalar),
            ("gemm_bi_nn_ultra_thin", ModuleKind::TriadScalar),
            ("gemm_bi_nn_narrow", ModuleKind::TriadScalar),
            ("gemm_bi_nn_narrow_small", ModuleKind::TriadScalar),
            ("gemm_bi_nn_big", ModuleKind::TriadScalar),
            ("gemm_bi_tn_gemv", ModuleKind::TriadScalar),
            ("gemm_bi_tn_narrow", ModuleKind::TriadScalar),
            ("gemm_bi_tn_big", ModuleKind::TriadScalar),
            ("gemm_bi_nt_gemv", ModuleKind::TriadScalar),
            ("gemm_bi_nt_narrow", ModuleKind::TriadScalar),
            ("gemm_bi_nt_big", ModuleKind::TriadScalar),
            ("gemm_bi_nn_tc", ModuleKind::TriadSm80),
            ("gemm_bi_nn_tc64", ModuleKind::TriadSm80),
            ("gemm_bi_nn_tc16", ModuleKind::TriadSm80),
            ("gemm_bi_tn_tc", ModuleKind::TriadSm80),
            ("gemm_bi_tn_tc64", ModuleKind::TriadSm80),
            ("gemm_bi_tn_tc128x64", ModuleKind::TriadSm80),
            ("gemm_bi_nt_tc", ModuleKind::TriadSm80),
            ("gemm_bi_nt_tc64", ModuleKind::TriadSm80),
        ] {
            for (dtype, suffix) in [(WeightDtype::Bf16, "_bf16"), (WeightDtype::F16, "_f16")] {
                let identity = HalfKernelIdentity::resolve(base, dtype).unwrap();
                assert_eq!(identity.symbol, format!("{base}{suffix}"));
                assert_eq!(identity.module_kind, module_kind);
            }
        }

        let tensor_core = HalfKernelIdentity::resolve("gemm_bi_nn_tc64", WeightDtype::F16).unwrap();
        assert!(
            tensor_core
                .validate(ModuleKind::TriadScalar, tensor_core.symbol)
                .is_err()
        );
        assert!(
            tensor_core
                .validate(tensor_core.module_kind, "gemm_bi_nn_tc64")
                .is_err()
        );
        assert!(HalfKernelIdentity::resolve("gemm_bi_nn_tc64", WeightDtype::F32).is_err());
        assert!(HalfKernelIdentity::resolve("gemm_bi_unknown", WeightDtype::Bf16).is_err());
    }

    #[test]
    fn rect128x64_tn_forced_identity_is_exact() {
        let tile = TcTile::Rect128x64;
        assert_eq!(tile.extents(), (128, 64));
        assert_eq!(tile.block_dim(), 256);
        assert_eq!(tile.bk_stages(), (32, 3));
        let config = tile.launch_cfg(1536, 768, 69_632).unwrap();
        assert_eq!(config.grid_dim, (144, 1, 1));
        assert_eq!(config.block_dim, (256, 1, 1));
        assert_eq!(config.shared_mem_bytes, 0);

        for (dtype, symbol) in [
            (WeightDtype::Bf16, "gemm_bi_tn_tc128x64_bf16"),
            (WeightDtype::F16, "gemm_bi_tn_tc128x64_f16"),
        ] {
            let identity = HalfKernelIdentity::resolve("gemm_bi_tn_tc128x64", dtype).unwrap();
            assert_eq!(identity.symbol, symbol);
            assert_eq!(identity.module_kind, ModuleKind::TriadSm80);
        }
    }
}

#[cfg(test)]
#[path = "scalar_nt_tests.rs"]
mod scalar_nt_tests;

#[cfg(test)]
#[path = "scalar_nn_tn_tests.rs"]
mod scalar_nn_tn_tests;

#[cfg(test)]
mod sm120_api_tests {
    use super::super::super::kernels::MambaKernels;
    use super::super::contract::{
        Sm120ForcedRoute, Sm120LaunchOperands, Sm120MapRequest, Sm120PreparedLaunch,
        Sm120PreparedTensorMaps, Sm120RouteIdentity,
    };
    use super::{
        launch_sm120_tma_prepared, prepare_sm120_tensor_maps, prepare_sm120_tma_forced,
        validate_sm120_graph_replay,
    };
    use std::sync::Arc;

    type Stream = Arc<cudarc::driver::CudaStream>;
    type PrepareMaps =
        fn(&Stream, &MambaKernels, Sm120MapRequest) -> Result<Sm120PreparedTensorMaps, String>;
    type PrepareLaunch = fn(
        &Stream,
        &MambaKernels,
        Sm120ForcedRoute,
        &Sm120PreparedTensorMaps,
        Sm120LaunchOperands,
    ) -> Result<Sm120PreparedLaunch, String>;
    type LaunchPrepared =
        fn(&Stream, &MambaKernels, &Sm120PreparedLaunch) -> Result<Sm120RouteIdentity, String>;
    type ValidateReplay = fn(&Stream, &MambaKernels, &Sm120PreparedLaunch) -> Result<(), String>;

    const _: PrepareMaps = prepare_sm120_tensor_maps;
    const _: PrepareLaunch = prepare_sm120_tma_forced;
    const _: LaunchPrepared = launch_sm120_tma_prepared;
    const _: ValidateReplay = validate_sm120_graph_replay;
}

/// The board capabilities a specialized module was bound on, as the launch
/// observer records them: the compute capability, the opt-in shared memory
/// and tensor-map access of the device, and the target the driver accepted.
pub fn query_specialized_device_caps(
    stream: &Arc<cudarc::driver::CudaStream>,
    nvrtc_arch: &'static str,
    nvrtc_version: (i32, i32),
) -> Result<crate::mamba_ssm::gpu::kernel_identity::DeviceCaps, String> {
    let ctx = stream.context();
    let (major, minor) = ctx
        .compute_capability()
        .map_err(|error| format!("query compute capability for {nvrtc_arch}: {error:?}"))?;
    let optin_shared = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .map_err(|error| format!("query opt-in shared memory for {nvrtc_arch}: {error:?}"))?;
    let tensor_map_access = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
        )
        .map_err(|error| format!("query tensor-map support for {nvrtc_arch}: {error:?}"))?
        != 0;
    Ok(crate::mamba_ssm::gpu::kernel_identity::DeviceCaps {
        compute_capability: (
            u32::try_from(major).map_err(|_| format!("negative CUDA CC major {major}"))?,
            u32::try_from(minor).map_err(|_| format!("negative CUDA CC minor {minor}"))?,
        ),
        nvrtc_version,
        accepted_target: Some(crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new(
            nvrtc_arch,
        )?),
        optin_shared_bytes: u32::try_from(optin_shared)
            .map_err(|_| format!("negative opt-in shared memory {optin_shared}"))?,
        tensor_map_access,
    })
}
