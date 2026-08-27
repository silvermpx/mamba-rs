use super::super::dtype::WeightDtype;
use super::super::kernels::MambaKernels as GpuKernels;
use super::super::{
    context::F32TriadPolicy,
    kernel_identity::{
        COMPILER_REVISION, COMPOSER_REVISION, ModuleKind, NUMERIC_ABI_REVISION, SCHEDULE_REVISION,
    },
};
use super::contract::{
    F32_TF32_TUNING_REVISION, F32TriadAvailability, F32TriadRequest, F32TriadSelection,
    Sm90aForcedRoute, Sm90aOp, Sm90aShape, Sm90aWarpgroupSchedule, Sm100ForcedRoute,
    Sm100TargetCandidate, Sm100TargetKind, Sm120ForcedRoute, Sm120TargetCandidate, Tf32KernelSpec,
    Tf32PhysicalRoute, Tf32QualifiedModule, tf32_kernel_spec,
};
use super::contract::{GemmDims, checked_mul3, checked_tile_grid, checked_usize};
use crate::mamba_ssm::gpu::kernel_identity::DeviceCaps;

#[derive(Clone, Copy)]
struct Tf32ShapeBucket {
    output_rows: (usize, usize),
    output_columns: (usize, usize),
    reduction: (usize, usize),
}

impl Tf32ShapeBucket {
    fn contains(self, request: F32TriadRequest) -> bool {
        let rows = request.shape.output_rows(request.op);
        let columns = request.shape.output_columns(request.op);
        let reduction = request.shape.reduction(request.op);
        (self.output_rows.0..=self.output_rows.1).contains(&rows)
            && (self.output_columns.0..=self.output_columns.1).contains(&columns)
            && (self.reduction.0..=self.reduction.1).contains(&reduction)
    }
}

#[derive(Clone, Copy)]
struct Tf32AutoCell {
    op: crate::mamba_ssm::gpu::kernel_identity::ResolvedGemmOp,
    shape_bucket: Tf32ShapeBucket,
    target: &'static str,
    route: Tf32PhysicalRoute,
    tuning_revision: u16,
}

const SM89_TF32_AUTO_CELLS: &[Tf32AutoCell] = &[];

fn measured_tf32_route(
    request: F32TriadRequest,
    availability: F32TriadAvailability,
    tuning_revision: u16,
) -> Option<Tf32PhysicalRoute> {
    let portable = availability.portable?;
    if portable.device.compute_capability != (8, 9) || portable.module_kind != ModuleKind::TriadSm80
    {
        return None;
    }
    SM89_TF32_AUTO_CELLS
        .iter()
        .find(|cell| {
            cell.tuning_revision == tuning_revision
                && cell.op == request.op
                && cell.target == portable.target.as_str()
                && cell.shape_bucket.contains(request)
        })
        .map(|cell| cell.route)
}

pub fn resolve_f32_triad_auto(
    policy: F32TriadPolicy,
    request: F32TriadRequest,
    availability: F32TriadAvailability,
) -> Result<F32TriadSelection, String> {
    request.shape.validate(request.op)?;
    match policy {
        F32TriadPolicy::ExactScalarFmaV1 => Ok(F32TriadSelection::ScalarFmaV1),
        F32TriadPolicy::AllowDeterministicTf32V1 => {
            if availability.portable.is_none() && availability.specialized.is_none() {
                return Ok(F32TriadSelection::ScalarFmaV1);
            }
            let Some(route) = measured_tf32_route(request, availability, F32_TF32_TUNING_REVISION)
            else {
                return Ok(F32TriadSelection::ScalarFmaV1);
            };
            match resolve_tf32_forced(request, availability, route) {
                Ok(route) => Ok(F32TriadSelection::Tf32(route)),
                Err(_) => Ok(F32TriadSelection::ScalarFmaV1),
            }
        }
    }
}

pub fn resolve_tf32_forced(
    request: F32TriadRequest,
    availability: F32TriadAvailability,
    route: Tf32PhysicalRoute,
) -> Result<Tf32PhysicalRoute, String> {
    request.shape.validate(request.op)?;
    let spec = tf32_kernel_spec(request.op, route)?;
    let binding = match route {
        Tf32PhysicalRoute::MmaTf32RnaV1(_) => availability.portable,
        Tf32PhysicalRoute::Sm90aWgmmaTf32TmaV1(_)
        | Tf32PhysicalRoute::Sm100Tcgen05Tf32TmaV1(_)
        | Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_) => availability.specialized,
    }
    .ok_or_else(|| format!("forced TF32 route {route:?} has no qualified module"))?;
    ensure_tf32_binding(binding, spec, route)?;
    Ok(route)
}

fn ensure_tf32_binding(
    binding: Tf32QualifiedModule,
    spec: &Tf32KernelSpec,
    route: Tf32PhysicalRoute,
) -> Result<(), String> {
    if binding.module_kind != spec.module_kind
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
    if !matches!(route, Tf32PhysicalRoute::MmaTf32RnaV1(_))
        && !binding.device_caps.tensor_map_access
    {
        return Err(format!(
            "forced TF32 route {route:?} requires tensor-map access"
        ));
    }
    if binding.device_caps.optin_shared_bytes < spec.dynamic_shared_bytes {
        return Err(format!(
            "forced TF32 route {route:?} needs {} shared bytes, device admits {}",
            spec.dynamic_shared_bytes, binding.device_caps.optin_shared_bytes
        ));
    }
    Ok(())
}

fn target_admits_route(binding: Tf32QualifiedModule, route: Tf32PhysicalRoute) -> bool {
    let cc = binding.device.compute_capability;
    let target = binding.target.as_str();
    let device_target = binding.device.target.as_str();
    match route {
        Tf32PhysicalRoute::MmaTf32RnaV1(_) => matches!(
            (cc, target, device_target),
            ((8, 0), "sm_80", "sm_80")
                | ((8, 6), "sm_86", "sm_86")
                | ((8, 7), "sm_87", "sm_87")
                | ((8, 9), "sm_89", "sm_89")
                | ((9, 0), "sm_90a", "sm_90a")
                | ((10, 0), "sm_100a", "sm_100a")
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
        Tf32PhysicalRoute::Sm120TmaMmaTf32RnaV1(_) => matches!(
            (cc, target, device_target),
            ((12, 0), "compute_120", "sm_120")
                | ((12, 1), "compute_121", "sm_121")
                | ((12, 1), "compute_120", "sm_120")
        ),
    }
}

pub const SM90A_AUTO_CELLS: &[Sm90aForcedRoute] = &[];
pub const SM100_AUTO_CELLS_CC100: &[Sm100ForcedRoute] = &[];
pub const SM100_AUTO_CELLS_CC103: &[Sm100ForcedRoute] = &[];
pub const SM120_AUTO_CELLS_CC120: &[Sm120ForcedRoute] = &[];
pub const SM120_AUTO_CELLS_CC121: &[Sm120ForcedRoute] = &[];

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

pub fn sm100_target_candidates(cc: (i32, i32)) -> &'static [Sm100TargetCandidate] {
    match cc {
        (10, 0) => &SM100_CC100_TARGETS,
        (10, 3) => &SM100_CC103_TARGETS,
        _ => &[],
    }
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

pub fn resolve_sm120_forced(
    caps: DeviceCaps,
    module_target: Option<Sm120TargetCandidate>,
    route: Sm120ForcedRoute,
) -> Result<Option<Sm120ForcedRoute>, String> {
    route.shape.validate(route.op)?;
    if !matches!(route.dtype, WeightDtype::Bf16 | WeightDtype::F16) {
        return Err("SM120 TMA supports bf16 and f16 operands only".into());
    }
    let spec = route.kernel_spec()?;
    let Some(module_target) = module_target else {
        return Ok(None);
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
    if module_target.device_cc != device_cc
        || !sm120_target_candidates(device_cc, caps.nvrtc_version).contains(&module_target)
        || accepted.as_deref() != Some(module_target.nvrtc_arch)
        || !caps.tensor_map_access
        || caps.optin_shared_bytes < spec.dynamic_shared_bytes
    {
        return Ok(None);
    }
    Ok(Some(route))
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

// ── Split-M TN partition heuristic (ported from SQV-RS blas_bi.rs) ──

/// Target CTA count factor for the split-M TN partition: aim to fill the
/// GPU with at least this many blocks when the base (K-tile × N-tile) grid
/// underfills it.
pub(super) const SPLITM_TN_TARGET_GRID_FACTOR: u32 = 284;
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
) -> Option<(usize, usize)> {
    // No n_in floor: the partial kernel predicates K_out < 128 exactly
    // like the plain kernel, and a small-K dW against a large batch
    // reduction underfills the grid without the split (K_out=24, N=768
    // ran six CTAs). The split changes the dW summation order versus
    // the plain kernel; run-to-run and per-shape determinism hold — the
    // partition is a pure function of (batch, n_in, n_out).
    if !(n_out >= 128 && batch >= 256) {
        return None;
    }
    let batch_u32 = u32::try_from(batch).ok()?;
    let k_tiles = u32::try_from(n_in).ok()?.div_ceil(128);
    let n_tiles = u32::try_from(n_out).ok()?.div_ceil(128);
    let base_blocks = k_tiles.checked_mul(n_tiles)?;
    if base_blocks == 0 || base_blocks >= SPLITM_TN_TARGET_GRID_FACTOR {
        return None;
    }
    let f_grid = SPLITM_TN_TARGET_GRID_FACTOR.div_ceil(base_blocks);
    let output_elements = n_in.checked_mul(n_out)?;
    let f_scratch_cap = u32::try_from(SPLITM_TN_SCRATCH_CAP / output_elements).ok()?;
    let f = f_grid.min(f_scratch_cap).max(1);
    let m_chunk_raw = batch_u32.div_ceil(f);
    let m_chunk = m_chunk_raw.checked_add(SPLITM_TN_BK_ALIGN - 1)? & !(SPLITM_TN_BK_ALIGN - 1);
    let f_final = batch_u32.div_ceil(m_chunk);
    let scratch_elements = usize::try_from(f_final)
        .ok()?
        .checked_mul(output_elements)?;
    if f_final < 2 || scratch_elements > SPLITM_TN_SCRATCH_CAP {
        return None;
    }
    Some((
        usize::try_from(m_chunk).ok()?,
        usize::try_from(f_final).ok()?,
    ))
}

/// Minimum N (output cols) before the dispatcher switches from Slim-N tiles
/// to Big-N tiles. Below this, Slim-N (BN=64) packs better; above it Big-N
/// (BN=128) wins on wave occupancy. Historic name (`SGEMM_CUSTOM_MIN`) is a
/// leftover from when the threshold gated a cuBLAS fallback — the fallback
/// is gone (zero-cuBLAS contract), the constant remains as a tile-pick
/// boundary only.
pub(super) const SGEMM_CUSTOM_MIN: usize = 128;

/// Boundary between Slim-N and Big tile variants (by output N dimension).
pub(super) const SGEMM_SLIM_MAX: usize = 512;

/// v6.5 Phase C-1.5av: separate Slim Split-K NT-via-T n_in cap for backward dx.
/// The forward Slim NN path uses N as output dim → SGEMM_SLIM_MAX=512 bounds
/// wave-fill correctness there. But NT-via-T backward dx reads n_in (input dim
/// of original forward), and the kernel itself tiles arbitrary n_in via N-axis
/// tiling — the 512 cap is conservative, not load-bearing. v6.5 multi-step
/// critic_in_proj has n_in = critic_in = d_model + action_dim + 2*emb = 641
/// at default config. Bumping to 768 lets this shape hit Slim Split-K NT-via-T
/// with F=4 K-tile partials (576 blocks vs plain Big NT 144 blocks).
/// Determinism preserved: F is shape-keyed (function of n_out, not batch).
pub(super) const SGEMM_SLIM_NT_NIN_MAX: usize = 768;

/// M threshold below which we force Slim-N even for N ≥ 129 (wave underfill protection).
/// At M < 512, Big tile BM=128 gives ≤4 M-blocks; adding N-blocks via Slim's BN=64 (vs Big's BN=128)
/// doubles grid to reduce wave underfill on Ada's 142 SMs. Only matters when N ≥ 129 (otherwise slim already chosen).
pub(super) const SGEMM_M_SLIM_FORCE: usize = 512;

/// Single source of truth for Split-K/M scratch buffer cap, in f32 elements.
/// Must match `splitk_scratch` allocation in `kernels.rs` (1 << 23 = 8M f32 = 32 MB).
/// All Split-K dispatch gates (NN fwd, NT bwd_dx, Split-M TN bwd_dw) read this.
pub(super) const SPLITK_SCRATCH_CAP: usize = 1 << 23;

/// SM count for dispatch wave-fill heuristics. Calibrated for Ada RTX 6000 (142 SMs).
/// Over-shoot on smaller GPUs (A100=108) is correctness-safe — Split-K gates fire
/// slightly more aggressively. Keep this as the single source of truth for the
/// current frozen dispatch table.
pub(super) const NUM_SMS: u32 = 142;

/// Pick (kernel function, BN tile size) with M-aware wave-quantization fix.
/// Slim-N for narrow output, or for small M with wide N.
/// Later buckets extend this dispatcher with narrow / GEMV / small-K buckets.
pub(super) fn dispatch_slim_or_big<'k>(
    _kernels: &'k GpuKernels,
    m: usize,
    n_out: usize,
    func_slim: &'k cudarc::driver::CudaFunction,
    func_big: &'k cudarc::driver::CudaFunction,
) -> (&'k cudarc::driver::CudaFunction, u32) {
    let slim = n_out <= SGEMM_SLIM_MAX || (m < SGEMM_M_SLIM_FORCE && n_out >= SGEMM_CUSTOM_MIN);
    let func = if slim { func_slim } else { func_big };
    let bn: u32 = if slim { 64 } else { 128 };
    (func, bn)
}

// ---------------------------------------------------------------------------
// f32-cascade routing predicates. Each returns true iff the f32
// dispatcher would run the BIG kernel (BN=128, 2-stage 33 KB smem) for this
// shape — i.e. NO earlier bucket in the cascade claims it AND the final
// slim/big split picks Big. The typed dispatch uses these so a native typed
// Big kernel fires exactly where the f32 reference runs the same FMA chain;
// any drift between a predicate and the real cascade shows up as a bit
// mismatch in tests/gemm_bi_typed_parity.rs.
// ---------------------------------------------------------------------------

/// NN forward: mirrors `sgemm_bi_forward` (gemv, ultra-thin, narrow tiers,
/// split-K thin-M K-tail/main, split-K slim, gap-fill, then big/slim).
pub(super) fn nn_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    let Ok(dims) = GemmDims::nn((batch, n_in, n_out), n_in) else {
        return false;
    };
    if n_out == 1 {
        return false; // gemv (or panic tail) — never Big
    }
    if (1..32).contains(&batch) && (32..=2048).contains(&n_in) && n_out >= 32 {
        return false; // ultra-thin
    }
    if (2..=127).contains(&n_out) {
        return false; // narrow tiers
    }
    let Ok(plain_slim_blocks) = checked_tile_grid(dims.m_u32, 128, dims.n_u32, 64) else {
        return false;
    };
    let underfill = plain_slim_blocks < NUM_SMS;
    // split-K thin-M K-tail
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && underfill
    {
        let k_main = n_in - n_in % 32;
        if k_main >= 32
            && checked_mul3(k_main / 32, batch, n_out, "NN route scratch")
                .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        {
            return false;
        }
    }
    // split-K thin-M main
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 32
        && n_in.is_multiple_of(32)
        && checked_mul3(n_in / 32, batch, n_out, "NN route scratch")
            .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        && underfill
    {
        return false;
    }
    // split-K slim
    if batch > 1024
        && (128..=SGEMM_SLIM_MAX).contains(&n_out)
        && n_in >= 64
        && n_in.is_multiple_of(32)
    {
        let f_final = dims.k_u32.div_ceil(64);
        if f_final >= 6
            && checked_mul3(
                checked_usize(f_final, "NN route chunks").unwrap_or(usize::MAX),
                batch,
                n_out,
                "NN route scratch",
            )
            .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        {
            let base_blocks =
                checked_tile_grid(dims.m_u32, 128, dims.n_u32, 64).unwrap_or(u32::MAX);
            if base_blocks > 0 && base_blocks < 3 * NUM_SMS {
                return false;
            }
        }
    }
    if batch < 128 {
        return false; // gap-fill territory
    }
    if !(batch >= SGEMM_CUSTOM_MIN && n_out >= SGEMM_CUSTOM_MIN) {
        return false;
    }
    let slim = n_out <= SGEMM_SLIM_MAX || (batch < SGEMM_M_SLIM_FORCE && n_out >= SGEMM_CUSTOM_MIN);
    !slim
}

/// TN dW: mirrors `sgemm_bi_backward_dw` (gemv, narrow, split-M, big/slim
/// keyed on output rows = `n_in`).
pub(super) fn tn_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    if GemmDims::tn((batch, n_in, n_out)).is_err() {
        return false;
    }
    if n_out == 1 || (2..=127).contains(&n_out) {
        return false; // gemv / narrow
    }
    if splitm_tn_partition(batch, n_in, n_out).is_some() {
        return false;
    }
    if !(n_in >= 1 && n_out >= SGEMM_CUSTOM_MIN) {
        return false;
    }
    let slim = n_out <= SGEMM_SLIM_MAX || (n_in < SGEMM_M_SLIM_FORCE && n_out >= SGEMM_CUSTOM_MIN);
    !slim
}

/// NT dX: mirrors `sgemm_bi_backward_dx` (narrow, col-gemv, gemv, split-N
/// K-tail/main, split-N slim, gap-fill, big/slim keyed on (`batch`, `n_in`)).
pub(super) fn nt_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    let Ok(dims) = GemmDims::nt((batch, n_in, n_out)) else {
        return false;
    };
    if (2..=127).contains(&n_out) {
        return false; // NT narrow (small reduction N)
    }
    if batch < 32 && n_out >= 128 {
        return false; // dx_col_gemv
    }
    if n_out == 1 {
        return false; // NT gemv
    }
    const SPLITK_NT_TRANSPOSE_CAP: usize = 1 << 22;
    let Ok(plain_slim_blocks) = checked_tile_grid(dims.m_u32, 128, dims.k_u32, 64) else {
        return false;
    };
    let underfill = plain_slim_blocks < NUM_SMS;
    // split-N K-tail
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_out.is_multiple_of(32)
        && underfill
    {
        let k_main = n_in - n_in % 32;
        if k_main >= 32
            && k_main
                .checked_mul(n_out)
                .is_some_and(|elements| elements <= SPLITK_NT_TRANSPOSE_CAP)
            && checked_mul3(n_out / 32, batch, k_main, "NT route scratch")
                .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        {
            return false;
        }
    }
    // split-N main
    let n_main = n_out - n_out % 32;
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in.is_multiple_of(4)
        && n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_main >= 32
        && dims.kn <= SPLITK_NT_TRANSPOSE_CAP
        && checked_mul3(n_main / 32, batch, n_in, "NT route scratch")
            .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        && underfill
    {
        return false;
    }
    // split-N slim
    if batch > 1024
        && (128..=SGEMM_SLIM_NT_NIN_MAX).contains(&n_in)
        && n_out >= 64
        && n_out.is_multiple_of(32)
        && dims.kn <= SPLITK_NT_TRANSPOSE_CAP
    {
        let f_final = dims.n_u32.div_ceil(64);
        if f_final >= 2
            && checked_mul3(
                checked_usize(f_final, "NT route chunks").unwrap_or(usize::MAX),
                batch,
                n_in,
                "NT route scratch",
            )
            .is_ok_and(|elements| elements <= SPLITK_SCRATCH_CAP)
        {
            let base_blocks =
                checked_tile_grid(dims.m_u32, 128, dims.k_u32, 64).unwrap_or(u32::MAX);
            if base_blocks > 0 && base_blocks < 3 * NUM_SMS {
                return false;
            }
        }
    }
    if (32..128).contains(&batch) {
        return false; // gap-fill NT
    }
    if !(batch >= SGEMM_CUSTOM_MIN && n_in >= 1) {
        return false;
    }
    let slim = n_in <= SGEMM_SLIM_MAX || (batch < SGEMM_M_SLIM_FORCE && n_in >= SGEMM_CUSTOM_MIN);
    !slim
}
/// Which tensor-core tile variant a TC entry point launched. Returned on
/// success so callers and tests can assert launch reality (0.4.0 lesson:
/// a kernel that silently never fires must be impossible to miss).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcTile {
    /// 128x128 CTA tile, 256 threads / 8 warps (`sgemm_bi_*_tc_*`).
    Tile128,
    /// 64x64 CTA tile, 128 threads / 4 warps (`sgemm_bi_*_tc64_*`).
    Tile64,
    /// 16x32 CTA tile, 128 threads / 4 warps, 4-stage cp.async
    /// (`sgemm_bi_nn_tc16_*`) - the decode rung of the ladder. NN
    /// forward only; picked by `tc_pick_tile_forward` for the small-M
    /// and narrow-N bands.
    Thin16,
}

/// Prefer the smaller tile while a 128x128 launch has too few independent
/// CTAs. Both tile families issue the same ascending MMA reduction for an
/// output element, so this changes occupancy without changing its bits.
pub const TC64_PREFER_MAX_TILES128: u32 =
    super::super::kernel_identity::LegacySm80Policy::current().tile128_prefer_min_tiles;

/// Choose between the square tiles once both output axes reach one Tile64.
pub(super) fn tc_pick_tile_large(rows: usize, cols: usize) -> Option<TcTile> {
    let policy = super::super::kernel_identity::LegacySm80Policy::current();
    if rows >= policy.large_tile_min && cols >= policy.large_tile_min {
        let tiles128 = u32::try_from(rows)
            .ok()?
            .div_ceil(128)
            .checked_mul(u32::try_from(cols).ok()?.div_ceil(128))?;
        if tiles128 >= TC64_PREFER_MAX_TILES128 {
            return Some(TcTile::Tile128);
        }
        return Some(TcTile::Tile64);
    }
    if rows >= policy.square_tile_min && cols >= policy.square_tile_min {
        return Some(TcTile::Tile64);
    }
    None
}

/// The forward thin tile covers the narrow rows or columns below Tile64.
/// Its per-element MMA order matches the square tiles, so crossing this
/// scheduling boundary preserves the forward numeric contract.
pub(super) fn tc_pick_tile_forward(rows: usize, cols: usize) -> Option<TcTile> {
    let policy = super::super::kernel_identity::LegacySm80Policy::current();
    if cols < policy.forward_min_columns {
        return None;
    }
    if rows <= policy.forward_thin_max_rows
        || (policy.forward_thin_below_square_columns && cols < policy.square_tile_min)
    {
        return Some(TcTile::Thin16);
    }
    tc_pick_tile_large(rows, cols)
}

/// Tile64 predicates both output tails, but two short axes would waste the
/// whole square tile and remain on the scalar fallback.
pub(super) fn tc_pick_tile_backward_bridge(rows: usize, cols: usize) -> Option<TcTile> {
    let policy = super::super::kernel_identity::LegacySm80Policy::current();
    if policy.reject_zero_axes && (rows == 0 || cols == 0) {
        return None;
    }
    if rows >= policy.square_tile_min && cols >= policy.square_tile_min {
        return tc_pick_tile_large(rows, cols);
    }
    if policy.backward_one_axis_tile64
        && (rows >= policy.square_tile_min || cols >= policy.square_tile_min)
    {
        return Some(TcTile::Tile64);
    }
    (!policy.backward_two_small_fallback).then_some(TcTile::Tile64)
}

/// Large shapes keep the existing square-tile policy. A one-axis tail is
/// selected automatically only when its full operation is frozen above.
pub(super) fn tc_pick_tile_backward(
    op: super::super::kernel_identity::PolicyOp,
    dtype: WeightDtype,
    dims: (usize, usize, usize),
) -> Option<TcTile> {
    let (batch, n_in, n_out) = dims;
    let (rows, cols) = match op {
        super::super::kernel_identity::PolicyOp::Dw => (n_in, n_out),
        super::super::kernel_identity::PolicyOp::Dx => (batch, n_in),
    };
    let tile = tc_pick_tile_backward_bridge(rows, cols)?;
    let policy = super::super::kernel_identity::LegacySm80Policy::current();
    if rows >= policy.square_tile_min && cols >= policy.square_tile_min {
        return Some(tile);
    }

    let dtype = match dtype {
        WeightDtype::F32 => super::super::kernel_identity::PolicyDtype::F32,
        WeightDtype::F16 => super::super::kernel_identity::PolicyDtype::F16,
        WeightDtype::Bf16 => super::super::kernel_identity::PolicyDtype::Bf16,
    };
    super::super::kernel_identity::LegacySm80Policy::current()
        .admits(op, dtype, dims)
        .then_some(tile)
}

#[cfg(test)]
mod sm120_tests {
    use super::{SM120_AUTO_CELLS_CC120, SM120_AUTO_CELLS_CC121, sm120_target_candidates};

    fn targets(cc: (i32, i32), nvrtc: (i32, i32)) -> Vec<(&'static str, &'static str)> {
        sm120_target_candidates(cc, nvrtc)
            .iter()
            .map(|candidate| (candidate.nvrtc_arch, candidate.ptx_target))
            .collect()
    }

    #[test]
    fn sm120_auto_tables_stay_empty_until_each_device_minor_is_qualified() {
        assert!(SM120_AUTO_CELLS_CC120.is_empty());
        assert!(SM120_AUTO_CELLS_CC121.is_empty());
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
    use super::{resolve_f32_triad_auto, resolve_tf32_forced};
    use crate::mamba_ssm::gpu::context::F32TriadPolicy;
    use crate::mamba_ssm::gpu::gemm_bi_triad::contract::{
        F32TriadAvailability, F32TriadRequest, F32TriadSelection, F32TriadShape, Tf32PhysicalRoute,
        Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile, Tf32QualifiedModule,
        Tf32Sm90aRoute, Tf32Sm100Route, Tf32Sm120Route, Tf32Sm120Tile,
    };
    use crate::mamba_ssm::gpu::gemm_bi_triad::{
        Sm90aWarpgroupSchedule, Sm100Schedule, Sm100Stages, Sm100Tile, Sm120Stages,
    };
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, ModuleKind, NUMERIC_ABI_REVISION,
        ResolvedGemmOp, SCHEDULE_REVISION,
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
                    stages: Sm120Stages::S3,
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
