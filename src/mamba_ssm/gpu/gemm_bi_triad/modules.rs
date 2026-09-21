use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, HashMap},
    ffi::{CString, c_void},
    sync::{Arc, Mutex},
};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig,
};

use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::kernel_identity::{
    ArtifactIdentity, ArtifactKind, CompilerIdentity, CudaTarget, FramedSha256, ModuleKind,
    ResolvedGemmOp, Sha256Digest,
};

use super::super::buffers::{GpuBuffer, cu_memcpy_dtoh_raw, cu_memcpy_htod_raw};
use super::super::kernels::{
    CudaModuleAnchors, FixedSm120FmaPostbiasKernels, HalfKernel, cuda_include_paths,
    kernel_cache_dir, nvrtc_version,
};

pub(crate) mod inference_bundle;

const TF32_EXCEPTIONAL_PROBE_BITS: [u32; 10] = [
    0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001, 0x00000001, 0x007fffff,
    0x00800000, 0x7f7fffff,
];

fn qualify_tf32_conversion_artifact<Launch, Download>(
    artifact: ArtifactIdentity,
    mut launch: Launch,
    mut download: Download,
) -> Result<Sha256Digest, String>
where
    Launch: FnMut() -> Result<(), String>,
    Download: FnMut() -> Result<Vec<u32>, String>,
{
    launch().map_err(|error| format!("first TF32 exceptional probe launch: {error}"))?;
    let first =
        download().map_err(|error| format!("first TF32 exceptional probe download: {error}"))?;
    if first.len() != TF32_EXCEPTIONAL_PROBE_BITS.len() {
        return Err(format!(
            "first TF32 exceptional probe returned {} words, expected {}",
            first.len(),
            TF32_EXCEPTIONAL_PROBE_BITS.len()
        ));
    }

    launch().map_err(|error| format!("second TF32 exceptional probe launch: {error}"))?;
    let second =
        download().map_err(|error| format!("second TF32 exceptional probe download: {error}"))?;
    if second.len() != TF32_EXCEPTIONAL_PROBE_BITS.len() {
        return Err(format!(
            "second TF32 exceptional probe returned {} words, expected {}",
            second.len(),
            TF32_EXCEPTIONAL_PROBE_BITS.len()
        ));
    }
    if first != second {
        return Err("TF32 exceptional probe output changed between launches".into());
    }

    let count = u64::try_from(TF32_EXCEPTIONAL_PROBE_BITS.len())
        .map_err(|_| "TF32 exceptional probe length exceeds u64::MAX".to_string())?;
    let mut digest = FramedSha256::new(b"tf32-exceptional-conversion-artifact.v1")
        .required(b"module-kind", &[artifact.module_kind as u8])
        .required(b"artifact-kind", &[artifact.artifact_kind as u8])
        .required(b"compile-key", &artifact.compile_key)
        .required(b"artifact-digest", &artifact.artifact_digest)
        .required(b"input-count", &count.to_le_bytes());
    for (index, bits) in TF32_EXCEPTIONAL_PROBE_BITS.iter().copied().enumerate() {
        let index = u64::try_from(index)
            .map_err(|_| "TF32 exceptional input index exceeds u64::MAX".to_string())?;
        digest = digest
            .required(b"input-index", &index.to_le_bytes())
            .required(b"input-bits", &bits.to_le_bytes());
    }
    digest = digest.required(b"output-count", &count.to_le_bytes());
    for (index, bits) in first.iter().copied().enumerate() {
        let index = u64::try_from(index)
            .map_err(|_| "TF32 exceptional output index exceeds u64::MAX".to_string())?;
        digest = digest
            .required(b"output-index", &index.to_le_bytes())
            .required(b"output-bits", &bits.to_le_bytes());
    }
    Ok(digest.finish())
}

fn retain_tf32_candidate<T, Binding>(
    functions: HashMap<&'static str, T>,
    binding: Binding,
    qualification: Result<Sha256Digest, String>,
) -> Result<(HashMap<&'static str, T>, Option<Binding>), String> {
    qualification?;
    if functions.is_empty() {
        Ok((HashMap::new(), None))
    } else {
        Ok((functions, Some(binding)))
    }
}

fn retain_specialized_tf32_candidate<T>(
    functions: HashMap<&'static str, T>,
    mut binding: super::contract::Tf32QualifiedModule,
    exclusions: &[Tf32SymbolExclusion],
    qualification: Result<Sha256Digest, String>,
) -> Result<
    (
        HashMap<&'static str, T>,
        Option<super::contract::Tf32QualifiedModule>,
    ),
    String,
> {
    binding.sm120_fma_exclusions = sm120_fma_exclusions(exclusions);
    retain_tf32_candidate(functions, binding, qualification)
}

fn retain_forced_only_functions<T>(
    functions: Result<(HashMap<&'static str, T>, Vec<Tf32SymbolExclusion>), String>,
) -> Result<(HashMap<&'static str, T>, Vec<Tf32SymbolExclusion>), String> {
    functions
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tf32DriverParameterAbi {
    offset: usize,
    size: usize,
}

impl Tf32DriverParameterAbi {
    pub(crate) fn offset(self) -> usize {
        self.offset
    }

    pub(crate) fn size(self) -> usize {
        self.size
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Tf32DriverAbi {
    parameter_count: usize,
    parameters: Box<[Tf32DriverParameterAbi]>,
}

impl Tf32DriverAbi {
    fn checked(parameter_count: usize, parameters: Vec<(usize, usize)>) -> Result<Self, String> {
        if parameter_count == 0 {
            return Err("TF32 Driver ABI has no parameters".into());
        }
        if parameter_count != parameters.len() {
            return Err(format!(
                "TF32 Driver ABI count is {parameter_count}, but {} layouts were queried",
                parameters.len()
            ));
        }
        if parameter_count > 64 {
            return Err(format!(
                "TF32 Driver ABI reports an implausible parameter count {parameter_count}"
            ));
        }

        let mut previous_end = 0;
        let mut checked = Vec::with_capacity(parameter_count);
        for (index, (offset, size)) in parameters.into_iter().enumerate() {
            if size == 0 {
                return Err(format!("TF32 Driver ABI parameter {index} has zero size"));
            }
            if index == 0 && offset != 0 {
                return Err(format!(
                    "TF32 Driver ABI first parameter starts at offset {offset}"
                ));
            }
            if offset < previous_end {
                return Err(format!(
                    "TF32 Driver ABI parameter {index} overlaps its predecessor"
                ));
            }
            previous_end = offset.checked_add(size).ok_or_else(|| {
                format!("TF32 Driver ABI parameter {index} extent overflows usize")
            })?;
            checked.push(Tf32DriverParameterAbi { offset, size });
        }
        Ok(Self {
            parameter_count,
            parameters: checked.into_boxed_slice(),
        })
    }

    pub(crate) fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub(crate) fn parameters(&self) -> &[Tf32DriverParameterAbi] {
        &self.parameters
    }

    pub(crate) fn tsv_record(&self, symbol: &str) -> Result<String, String> {
        if symbol.is_empty()
            || !symbol
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err("TF32 Driver ABI symbol is not safe for TSV output".into());
        }
        let layout = self
            .parameters()
            .iter()
            .map(|parameter| format!("{}:{}", parameter.offset(), parameter.size()))
            .collect::<Vec<_>>()
            .join(",");
        Ok(format!(
            "{symbol}\t{}\tptx_contract+cuFuncGetParamInfo_terminal_probe\t{layout}",
            self.parameter_count()
        ))
    }
}

fn merge_tf32_driver_abi(
    mut portable: BTreeMap<&'static str, Tf32DriverAbi>,
    specialized: Option<BTreeMap<&'static str, Tf32DriverAbi>>,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    for (symbol, abi) in specialized.into_iter().flatten() {
        if portable.insert(symbol, abi).is_some() {
            return Err(format!(
                "TF32 Driver ABI symbol {symbol} belongs to more than one module"
            ));
        }
    }
    Ok(portable)
}

fn merge_optional_finalist_driver_abi(
    portable: BTreeMap<&'static str, Tf32DriverAbi>,
    finalist: Option<BTreeMap<&'static str, Tf32DriverAbi>>,
) -> (BTreeMap<&'static str, Tf32DriverAbi>, Option<String>) {
    let fallback = portable.clone();
    match merge_tf32_driver_abi(portable, finalist) {
        Ok(merged) => (merged, None),
        Err(error) => (fallback, Some(error)),
    }
}

struct DriverModule {
    raw: Option<cudarc::driver::sys::CUmodule>,
}

impl DriverModule {
    fn load(ctx: &CudaContext, ptx: &str) -> Result<Self, String> {
        ctx.bind_to_thread()
            .map_err(|error| format!("bind CUDA context for Driver ABI census: {error:?}"))?;
        let image = CString::new(ptx)
            .map_err(|_| "canonical PTX contains an interior NUL byte".to_string())?;
        let raw = unsafe {
            cudarc::driver::result::module::load_data(image.as_ptr().cast::<c_void>())
        }
        .map_err(|error| format!("load temporary module for Driver ABI census: {error:?}"))?;
        Ok(Self { raw: Some(raw) })
    }

    fn raw(&self) -> cudarc::driver::sys::CUmodule {
        self.raw.expect("live DriverModule")
    }

    fn unload(mut self) -> Result<(), String> {
        let raw = self.raw.take().expect("live DriverModule");
        unsafe { cudarc::driver::result::module::unload(raw) }
            .map_err(|error| format!("unload temporary Driver ABI module: {error:?}"))
    }
}

impl Drop for DriverModule {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            let _ = unsafe { cudarc::driver::result::module::unload(raw) };
        }
    }
}

fn driver_call(result: cudarc::driver::sys::CUresult, operation: &str) -> Result<(), String> {
    if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!(
            "{operation}: {:?}",
            cudarc::driver::result::DriverError(result)
        ))
    }
}

fn driver_proc_address(symbol: &str, cuda_version: i32) -> Result<*mut c_void, String> {
    let symbol = CString::new(symbol).expect("static CUDA Driver symbol");
    let mut address = std::ptr::null_mut();
    let mut status =
        cudarc::driver::sys::CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    let result = unsafe {
        cudarc::driver::sys::cuGetProcAddress_v2(
            symbol.as_ptr(),
            &mut address,
            cuda_version,
            cudarc::driver::sys::CUdriverProcAddress_flags::CU_GET_PROC_ADDRESS_DEFAULT as u64,
            &mut status,
        )
    };
    driver_call(result, &format!("resolve {}", symbol.to_string_lossy()))?;
    if status != cudarc::driver::sys::CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SUCCESS
        || address.is_null()
    {
        return Err(format!(
            "resolve {} returned {status:?} at address {address:p}",
            symbol.to_string_lossy(),
        ));
    }
    Ok(address)
}

const TF32_DRIVER_PARAMETER_COUNT: usize = 5;
/// The stream-K kernel takes its slab and flag buffers ahead of the tensor
/// maps: output, slabs, flags, two maps, bias, parameters.
const TF32_STREAMK_DRIVER_PARAMETER_COUNT: usize = 7;

fn tf32_driver_parameter_count(module_kind: ModuleKind, symbol: &str) -> usize {
    let streamk = super::contract::tf32_route_specs_all(module_kind).any(|spec| {
        spec.symbol == symbol
            && (spec.route.is_exact_fma()
                || matches!(
                    spec.route,
                    super::contract::Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_)
                ))
    });
    if streamk {
        TF32_STREAMK_DRIVER_PARAMETER_COUNT
    } else {
        TF32_DRIVER_PARAMETER_COUNT
    }
}

fn query_driver_parameter_abi(
    label: &str,
    parameter_count: usize,
    mut get_parameter_info: impl FnMut(usize, &mut usize, &mut usize) -> cudarc::driver::sys::CUresult,
) -> Result<Tf32DriverAbi, String> {
    let mut parameters = Vec::with_capacity(parameter_count);
    for index in 0..parameter_count {
        let mut offset = 0;
        let mut size = 0;
        let result = get_parameter_info(index, &mut offset, &mut size);
        driver_call(result, &format!("cuFuncGetParamInfo {label}[{index}]"))?;
        parameters.push((offset, size));
    }

    let mut extra_offset = 0;
    let mut extra_size = 0;
    let extra = get_parameter_info(parameter_count, &mut extra_offset, &mut extra_size);
    if extra == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!(
            "{label} exposes more than {parameter_count} Driver ABI parameters"
        ));
    }
    if extra != cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE {
        driver_call(
            extra,
            &format!("cuFuncGetParamInfo {label}[{parameter_count}] sentinel"),
        )?;
    }
    Tf32DriverAbi::checked(parameters.len(), parameters)
        .map_err(|error| format!("{label}: {error}"))
}

/// The Driver ABI of one TF32 route symbol: the tiled routes expose the
/// five-parameter contract, the stream-K route the seven-parameter one.
fn query_tf32_driver_parameter_abi(
    label: &str,
    parameter_count: usize,
    get_parameter_info: impl FnMut(usize, &mut usize, &mut usize) -> cudarc::driver::sys::CUresult,
) -> Result<Tf32DriverAbi, String> {
    query_driver_parameter_abi(label, parameter_count, get_parameter_info)
}

fn census_tf32_driver_abi(
    ctx: &CudaContext,
    module_kind: ModuleKind,
    extensions: bool,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    let symbols: Vec<&'static str> = super::contract::tf32_route_specs_for(module_kind, extensions)
        .map(|spec| spec.symbol)
        .collect();
    if symbols.is_empty() {
        return Ok(BTreeMap::new());
    }

    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;

    let module = DriverModule::load(ctx, ptx)?;
    let get_parameter_info: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for symbol in symbols {
        let name = CString::new(symbol).expect("static TF32 symbol");
        let function = unsafe { cudarc::driver::result::module::get_function(module.raw(), name) }
            .map_err(|error| format!("load {module_kind:?}/{symbol} for Driver ABI: {error:?}"))?;
        let label = format!("{module_kind:?}/{symbol}");
        let abi = query_tf32_driver_parameter_abi(
            &label,
            tf32_driver_parameter_count(module_kind, symbol),
            |index, offset, size| unsafe { get_parameter_info(function, index, offset, size) },
        )?;
        if census.insert(symbol, abi).is_some() {
            return Err(format!(
                "{module_kind:?} Driver ABI census contains duplicate symbol {symbol}"
            ));
        }
    }
    module.unload()?;
    Ok(census)
}

fn census_tf32_splitk_driver_abi(
    ctx: &CudaContext,
    module_kind: ModuleKind,
    extensions: bool,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    if module_kind != ModuleKind::TriadSm80 {
        return Ok(BTreeMap::new());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;

    let module = DriverModule::load(ctx, ptx)?;
    let get_parameter_info: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for spec in super::contract::tf32_splitk_specs_for(extensions) {
        let symbol = spec.symbol;
        let name = CString::new(symbol).expect("static TF32 split-K symbol");
        let function = unsafe { cudarc::driver::result::module::get_function(module.raw(), name) }
            .map_err(|error| format!("load {module_kind:?}/{symbol} for Driver ABI: {error:?}"))?;
        let label = format!("{module_kind:?}/{symbol}");
        let abi = query_driver_parameter_abi(&label, 7, |index, offset, size| unsafe {
            get_parameter_info(function, index, offset, size)
        })?;
        if census.insert(symbol, abi).is_some() {
            return Err(format!(
                "{module_kind:?} split-K Driver ABI census contains duplicate symbol {symbol}"
            ));
        }
    }
    module.unload()?;
    Ok(census)
}

fn census_all_tf32_driver_abi(
    ctx: &CudaContext,
    module_kind: ModuleKind,
    extensions: bool,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    let mut census = census_tf32_driver_abi(ctx, module_kind, extensions, ptx)?;
    for (symbol, abi) in census_tf32_splitk_driver_abi(ctx, module_kind, extensions, ptx)? {
        if census.insert(symbol, abi).is_some() {
            return Err(format!(
                "{module_kind:?} Driver ABI census contains duplicate symbol {symbol}"
            ));
        }
    }
    Ok(census)
}

fn complete_tf32_driver_abi(
    module_kind: ModuleKind,
    extensions: bool,
    census: &BTreeMap<&'static str, Tf32DriverAbi>,
) -> bool {
    let production: Vec<&'static str> =
        super::contract::tf32_route_specs_for(module_kind, extensions)
            .map(|spec| spec.symbol)
            .collect();
    let production_complete =
        !production.is_empty() && production.iter().all(|symbol| census.contains_key(symbol));
    let splitk_complete = module_kind != ModuleKind::TriadSm80
        || super::contract::tf32_splitk_specs_for(extensions)
            .map(|spec| spec.symbol)
            .all(|symbol| census.contains_key(symbol));
    production_complete && splitk_complete
}

pub(crate) struct CompileModuleRequest<'a> {
    pub ctx: &'a Arc<CudaContext>,
    pub arch: &'static str,
    pub state_cap: usize,
    pub module_kind: ModuleKind,
}

struct FixedSm89FinalistDriverAbi {
    rna_n96: Result<Tf32DriverAbi, String>,
    half_m64n64_s3: Result<Tf32DriverAbi, String>,
    half_m128n64_s2: Result<Tf32DriverAbi, String>,
}

impl FixedSm89FinalistDriverAbi {
    fn rejected(reason: String) -> Self {
        Self {
            rna_n96: Err(reason.clone()),
            half_m64n64_s3: Err(reason.clone()),
            half_m128n64_s2: Err(reason),
        }
    }
}

pub(crate) struct CompiledModule {
    pub module: Arc<CudaModule>,
    pub compiler_identity: CompilerIdentity,
    pub artifact_identity: ArtifactIdentity,
    tf32_qualified: bool,
    /// The first step that kept the module's TF32 routes from qualifying.
    tf32_qualification_error: Option<String>,
    tf32_driver_abi: BTreeMap<&'static str, Tf32DriverAbi>,
    sm89_half_driver_abi: Result<BTreeMap<&'static str, Result<Tf32DriverAbi, String>>, String>,
    sm89_exact_f32_driver_abi:
        Result<BTreeMap<&'static str, Result<Tf32DriverAbi, String>>, String>,
    sm89_exact_f32_d128_driver_abi:
        Result<BTreeMap<&'static str, Result<Tf32DriverAbi, String>>, String>,
    sm89_tf32_joint_driver_abi:
        Result<BTreeMap<&'static str, Result<Tf32DriverAbi, String>>, String>,
    /// Separate from Triad TF32 qualification: the optional Ada Fixed half
    /// extension has the same generic Driver layout representation only.
    fixed_sm89_half_driver_abi: Result<BTreeMap<&'static str, Tf32DriverAbi>, String>,
    fixed_sm89_half_swizzle_driver_abi: Result<BTreeMap<&'static str, Tf32DriverAbi>, String>,
    fixed_sm89_half_s3_driver_abi: Result<BTreeMap<&'static str, Tf32DriverAbi>, String>,
    fixed_sm89_rna_wide_driver_abi: Result<Tf32DriverAbi, String>,
    fixed_sm89_finalist_driver_abi: FixedSm89FinalistDriverAbi,
    fixed_sm89_exact_n64_driver_abi: Result<BTreeMap<&'static str, Tf32DriverAbi>, String>,
    fixed_sm89_cells_driver_abi: Result<BTreeMap<&'static str, Tf32DriverAbi>, String>,
    fixed_sm120_exact_n64_driver_abi: Result<BTreeMap<&'static str, Tf32DriverAbi>, String>,
    fixed_sm120_sliced_driver_abi: Result<BTreeMap<&'static str, Tf32DriverAbi>, String>,
    fixed_sm120_postbias_driver_abi: Result<BTreeMap<&'static str, Tf32DriverAbi>, String>,
    inference_sm89_driver_abi: inference_bundle::InferenceSm89DriverAbiCensus,
}

/// A TF32 symbol the loaded module cannot serve on this toolkit: the
/// compiler spilled it to local memory, or it exceeds its register or thread
/// gate. The rest of the module serves; a route to this symbol declines to
/// the exact family with the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Tf32SymbolExclusion {
    pub symbol: &'static str,
    pub reason: String,
}

/// Whether one loaded TF32 kernel meets its resource gates. Every violation
/// is a per-symbol exclusion, never a module rejection: one kernel a toolkit
/// spills must not darken the family.
pub(super) fn tf32_symbol_admission(
    symbol: &str,
    local_bytes: u32,
    local_bytes_pin: u32,
    registers: u32,
    register_cap: u32,
    max_threads: i32,
    threads: i32,
) -> Result<(), String> {
    // A spill is normally a mistuned kernel, so the pin is zero for all but
    // the one tile whose measurement recorded one: a 192x192 tile over 384
    // threads holds 96 accumulators per thread, and the register file leaves
    // 168 registers each, so the rest goes to local memory. That tile was
    // measured with the spill and still takes its cell by a quarter, so the
    // spill is pinned rather than forbidden and a larger one still fails.
    if local_bytes > local_bytes_pin {
        return Err(format!(
            "{symbol} uses {local_bytes} bytes of Driver JIT local memory on this toolkit, \
             above the {local_bytes_pin} bytes its measurement recorded"
        ));
    }
    if registers > register_cap {
        return Err(format!(
            "{symbol} uses {registers} registers, above its {register_cap}-register gate"
        ));
    }
    if max_threads < threads {
        return Err(format!("{symbol} cannot launch {threads} threads"));
    }
    Ok(())
}

pub(crate) struct QualifiedSpecializedModule {
    module: CompiledModule,
    functions: HashMap<&'static str, CudaFunction>,
    tf32_functions: HashMap<&'static str, CudaFunction>,
    tf32_rejection: Option<String>,
    tf32_excluded: Vec<Tf32SymbolExclusion>,
    sm120_target: Option<super::contract::Sm120TargetCandidate>,
    sm120_device_caps: Option<crate::mamba_ssm::gpu::kernel_identity::DeviceCaps>,
    sm120_resources: HashMap<&'static str, super::contract::Sm120KernelResources>,
}

pub(crate) struct Sm120ArtifactSet {
    pub fixed: CompiledModule,
    pub scalar: CompiledModule,
    pub sm80: CompiledModule,
    pub specialized: Option<QualifiedSpecializedModule>,
}

const SCALAR_GROUP_M_MACRO: &str = "GEMM_BI_GROUP_M";

fn scalar_group_m_option(arch: &str) -> String {
    let group_m = match arch {
        "sm_80" | "sm_86" | "sm_87" => 8,
        _ => 16,
    };
    format!("-D{SCALAR_GROUP_M_MACRO}={group_m}")
}

/// The compile-time identity of one module, computed the way
/// [`compile_module`] computes it but without a device: the composed source,
/// the NVRTC invocation, the PTX it produces and the header closure it read.
/// The qualification tools print these for every toolkit so the frozen
/// cohorts can be re-pinned after a source change that moves no bits.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct MintedModuleIdentity {
    pub module_kind: ModuleKind,
    pub target: &'static str,
    pub state_cap: usize,
    pub device_cc: Option<(i32, i32)>,
    pub nvrtc_version: (i32, i32),
    pub source_digest: Sha256Digest,
    pub compile_key: Option<Sha256Digest>,
    pub invocation_digest: Sha256Digest,
    pub artifact_digest: Sha256Digest,
    pub header_manifest_digest: Sha256Digest,
    pub nvrtc_library_domain: Sha256Digest,
}

#[doc(hidden)]
pub fn mint_module_identity(
    module_kind: ModuleKind,
    arch: &'static str,
    state_cap: usize,
    device_cc: Option<(i32, i32)>,
) -> Result<MintedModuleIdentity, String> {
    validate_module_target(module_kind, arch)?;
    let nvrtc = nvrtc_version();
    let combined = compose_compile_module_source(module_kind, device_cc, arch, state_cap, nvrtc)?;
    if let Some(directory) = std::env::var_os("MAMBA_RS_MINT_DUMP") {
        let file = std::path::Path::new(&directory).join(format!(
            "{module_kind:?}-{arch}-cap{state_cap}-nvrtc{}.{}.cu",
            nvrtc.0, nvrtc.1
        ));
        std::fs::write(&file, combined.as_bytes())
            .map_err(|error| format!("could not write {file:?}: {error}"))?;
    }
    let mut option_strings = vec![
        "--fmad=true".to_string(),
        "--extra-device-vectorization".to_string(),
        "-DNDEBUG".to_string(),
        scalar_group_m_option(arch),
    ];
    if module_kind == ModuleKind::Fixed {
        option_strings.push(format!("-DMAMBA_RS_STATE_CAP={state_cap}"));
    }
    option_strings.extend(
        crate::mamba_ssm::gpu::kernel_identity::deterministic_nvrtc_options(nvrtc, "1295072049"),
    );
    let include_paths = cuda_include_paths();
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(arch),
        options: option_strings.clone(),
        include_paths: include_paths.clone(),
        ..Default::default()
    };
    let nvrtc_library_domain = crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain();
    let header_manifest = crate::mamba_ssm::gpu::kernel_identity::header_manifest(
        combined.as_bytes(),
        &include_paths,
    );
    let mut argv = vec![format!("--gpu-architecture={arch}").into_bytes()];
    argv.extend(option_strings.iter().map(|value| value.as_bytes().to_vec()));
    let key_material = crate::mamba_ssm::gpu::kernel_identity::CompileKeyMaterial {
        module_kind,
        source: combined.as_bytes().to_vec(),
        target: arch.as_bytes().to_vec(),
        argv,
        header_manifest: header_manifest.clone(),
        nvrtc_version: nvrtc,
        nvrtc_library_domain: nvrtc_library_domain.clone(),
        output_kind: ArtifactKind::Ptx,
        composer_revision: crate::mamba_ssm::gpu::kernel_identity::COMPOSER_REVISION,
        compiler_revision: crate::mamba_ssm::gpu::kernel_identity::COMPILER_REVISION,
        numeric_abi_revision: crate::mamba_ssm::gpu::kernel_identity::NUMERIC_ABI_REVISION,
        schedule_revision: crate::mamba_ssm::gpu::kernel_identity::SCHEDULE_REVISION,
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(&combined, opts).map_err(|error| {
        format!(
            "{module_kind:?} NVRTC compile failed: {}",
            format!("{error:?}").replace("\\n", "\n")
        )
    })?;
    let ptx_image = ptx
        .as_bytes()
        .ok_or_else(|| format!("{module_kind:?} NVRTC returned no PTX image"))?;
    let ptx_source = crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(ptx_image)?;
    validate_module_ptx(module_kind, arch, &ptx_source)?;
    Ok(MintedModuleIdentity {
        module_kind,
        target: arch,
        state_cap,
        device_cc,
        nvrtc_version: nvrtc,
        source_digest: FramedSha256::bytes(combined.as_bytes()),
        compile_key: key_material.digest(),
        invocation_digest: key_material.invocation_digest(),
        artifact_digest: FramedSha256::bytes(ptx_source.as_bytes()),
        header_manifest_digest: FramedSha256::new(b"cuda-header-manifest.v1")
            .optional(b"manifest", header_manifest.as_deref())
            .finish(),
        nvrtc_library_domain: FramedSha256::new(b"nvrtc-library-set-identity.v2")
            .optional(b"domain", nvrtc_library_domain.as_deref())
            .finish(),
    })
}

pub(crate) fn compile_module(request: CompileModuleRequest<'_>) -> Result<CompiledModule, String> {
    validate_module_target(request.module_kind, request.arch)?;
    let nvrtc = nvrtc_version();
    let device_cc = if request.module_kind == ModuleKind::Fixed {
        request.ctx.compute_capability().ok()
    } else {
        None
    };
    let combined = compose_compile_module_source(
        request.module_kind,
        device_cc,
        request.arch,
        request.state_cap,
        nvrtc,
    )?;
    if request.module_kind == ModuleKind::TriadScalar
        && !combined.contains(&format!("#ifndef {SCALAR_GROUP_M_MACRO}"))
    {
        return Err("scalar L2 swizzle macro is missing from the composed source".into());
    }
    let mut option_strings = vec![
        "--fmad=true".to_string(),
        "--extra-device-vectorization".to_string(),
        "-DNDEBUG".to_string(),
        scalar_group_m_option(request.arch),
    ];
    // The state cap sizes the SSM kernels' register arrays; only the Fixed
    // module carries those kernels. A GEMM module compiled with it would
    // change its compile key with every model's d_state for nothing.
    if request.module_kind == ModuleKind::Fixed {
        option_strings.push(format!("-DMAMBA_RS_STATE_CAP={}", request.state_cap));
    }
    option_strings.extend(
        crate::mamba_ssm::gpu::kernel_identity::deterministic_nvrtc_options(nvrtc, "1295072049"),
    );
    let include_paths = cuda_include_paths();
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(request.arch),
        options: option_strings.clone(),
        include_paths: include_paths.clone(),
        ..Default::default()
    };
    let nvrtc_library_domain = crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain();
    let header_manifest = crate::mamba_ssm::gpu::kernel_identity::header_manifest(
        combined.as_bytes(),
        &include_paths,
    );
    // The include paths are a fact about this machine, not about the
    // kernels: the header manifest already digests every header the source
    // reaches, so the key stays the same wherever the toolkit is installed.
    let mut argv = vec![format!("--gpu-architecture={}", request.arch).into_bytes()];
    argv.extend(option_strings.iter().map(|value| value.as_bytes().to_vec()));
    let key_material = crate::mamba_ssm::gpu::kernel_identity::CompileKeyMaterial {
        module_kind: request.module_kind,
        source: combined.as_bytes().to_vec(),
        target: request.arch.as_bytes().to_vec(),
        argv,
        header_manifest: header_manifest.clone(),
        nvrtc_version: nvrtc,
        nvrtc_library_domain: nvrtc_library_domain.clone(),
        output_kind: ArtifactKind::Ptx,
        composer_revision: crate::mamba_ssm::gpu::kernel_identity::COMPOSER_REVISION,
        compiler_revision: crate::mamba_ssm::gpu::kernel_identity::COMPILER_REVISION,
        numeric_abi_revision: crate::mamba_ssm::gpu::kernel_identity::NUMERIC_ABI_REVISION,
        schedule_revision: crate::mamba_ssm::gpu::kernel_identity::SCHEDULE_REVISION,
    };
    let invocation_digest = key_material.invocation_digest();
    let cache_key = key_material.digest();
    let cache_path = cache_key.and_then(|key| {
        kernel_cache_dir().map(|directory| {
            directory.join(format!(
                "mamba-kernels-v1-{}.bin",
                crate::mamba_ssm::gpu::kernel_identity::digest_hex(&key)
            ))
        })
    });

    let trace = std::env::var_os("MAMBA_RS_CACHE_TRACE").is_some();
    if trace {
        eprintln!(
            "mamba-rs cache trace: {:?} arch={} key={} path={:?} source_len={}",
            request.module_kind,
            request.arch,
            cache_key.map_or_else(
                || "none".to_string(),
                |key| { crate::mamba_ssm::gpu::kernel_identity::digest_hex(&key) }
            ),
            cache_path,
            combined.len()
        );
    }
    let mut loaded = None;
    if let (Some(path), Some(key)) = (&cache_path, cache_key)
        && let Some(hit) =
            crate::mamba_ssm::gpu::kernel_identity::read_cache(path, key, ArtifactKind::Ptx)
        && let Ok(src) =
            crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_from_cache(hit.payload)
        && validate_module_ptx(request.module_kind, request.arch, &src).is_ok()
        && let Ok(module) = request
            .ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(src.clone()))
        && crate::mamba_ssm::gpu::kernel_identity::cache_hit_header_closure_is_current(
            combined.as_bytes(),
            &include_paths,
            &header_manifest,
        )
        && nvrtc_library_domain
            .as_deref()
            .is_some_and(crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain_is_current)
    {
        let extensions = module_composes_extensions(request.module_kind, request.arch);
        let census = census_all_tf32_driver_abi(request.ctx, request.module_kind, extensions, &src);
        let sm89_half_abi =
            census_sm89_half_driver_abi(request.ctx, request.module_kind, request.arch, &src);
        let sm89_exact_f32_abi =
            census_sm89_exact_f32_driver_abi(request.ctx, request.module_kind, request.arch, &src);
        let sm89_exact_f32_d128_abi = census_sm89_exact_f32_d128_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let sm89_tf32_joint_abi =
            census_sm89_tf32_joint_driver_abi(request.ctx, request.module_kind, request.arch, &src);
        let fixed_half_abi =
            census_fixed_sm89_half_driver_abi(request.ctx, request.module_kind, request.arch, &src);
        let fixed_half_swizzle_abi = census_fixed_sm89_half_swizzle_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let fixed_half_s3_abi = census_fixed_sm89_half_s3_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let fixed_rna_wide_abi = census_fixed_sm89_rna_wide_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let fixed_finalist_abi = census_fixed_sm89_finalist_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let fixed_exact_n64_abi = census_fixed_sm89_exact_n64_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let fixed_cells_abi = census_fixed_sm89_cells_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let fixed_sm120_exact_n64_abi = census_fixed_sm120_exact_n64_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let fixed_sm120_sliced_abi = census_fixed_sm120_sliced_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let fixed_sm120_postbias_abi = census_fixed_sm120_postbias_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let inference_sm89_driver_abi = inference_bundle::census_inference_sm89_driver_abi(
            request.ctx,
            request.module_kind,
            request.arch,
            &src,
        );
        let validation = validate_tf32_specialization(request.module_kind, request.arch, &src);
        let (tf32_driver_abi, tf32_qualification_error) =
            tf32_qualification_verdict(request.module_kind, extensions, census, validation);
        if trace {
            eprintln!(
                "mamba-rs cache trace: {:?} hit artifact={}",
                request.module_kind,
                crate::mamba_ssm::gpu::kernel_identity::digest_hex(&hit.artifact_digest)
            );
        }
        loaded = Some((
            module,
            hit.artifact_digest,
            tf32_qualification_error,
            tf32_driver_abi,
            sm89_half_abi,
            sm89_exact_f32_abi,
            sm89_exact_f32_d128_abi,
            sm89_tf32_joint_abi,
            fixed_half_abi,
            fixed_half_swizzle_abi,
            fixed_half_s3_abi,
            fixed_rna_wide_abi,
            fixed_finalist_abi,
            fixed_exact_n64_abi,
            fixed_cells_abi,
            fixed_sm120_exact_n64_abi,
            fixed_sm120_sliced_abi,
            fixed_sm120_postbias_abi,
            inference_sm89_driver_abi,
        ));
    }

    let (
        module,
        artifact_digest,
        tf32_qualification_error,
        tf32_driver_abi,
        sm89_half_driver_abi,
        sm89_exact_f32_driver_abi,
        sm89_exact_f32_d128_driver_abi,
        sm89_tf32_joint_driver_abi,
        fixed_sm89_half_driver_abi,
        fixed_sm89_half_swizzle_driver_abi,
        fixed_sm89_half_s3_driver_abi,
        fixed_sm89_rna_wide_driver_abi,
        fixed_sm89_finalist_driver_abi,
        fixed_sm89_exact_n64_driver_abi,
        fixed_sm89_cells_driver_abi,
        fixed_sm120_exact_n64_driver_abi,
        fixed_sm120_sliced_driver_abi,
        fixed_sm120_postbias_driver_abi,
        inference_sm89_driver_abi,
    ) = match loaded {
        Some(value) => value,
        None => {
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(&combined, opts).map_err(|error| {
                format!(
                    "{:?} NVRTC compile failed: {}",
                    request.module_kind,
                    format!("{error:?}").replace("\\n", "\n")
                )
            })?;
            let ptx_image = ptx
                .as_bytes()
                .ok_or_else(|| format!("{:?} NVRTC returned no PTX image", request.module_kind))?;
            let ptx_source =
                crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(ptx_image)?;
            validate_module_ptx(request.module_kind, request.arch, &ptx_source)?;
            let extensions = module_composes_extensions(request.module_kind, request.arch);
            let census = census_all_tf32_driver_abi(
                request.ctx,
                request.module_kind,
                extensions,
                &ptx_source,
            );
            let sm89_half_abi = census_sm89_half_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let sm89_exact_f32_abi = census_sm89_exact_f32_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let sm89_exact_f32_d128_abi = census_sm89_exact_f32_d128_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let sm89_tf32_joint_abi = census_sm89_tf32_joint_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_half_abi = census_fixed_sm89_half_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_half_swizzle_abi = census_fixed_sm89_half_swizzle_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_half_s3_abi = census_fixed_sm89_half_s3_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_rna_wide_abi = census_fixed_sm89_rna_wide_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_finalist_abi = census_fixed_sm89_finalist_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_exact_n64_abi = census_fixed_sm89_exact_n64_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_cells_abi = census_fixed_sm89_cells_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_sm120_exact_n64_abi = census_fixed_sm120_exact_n64_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_sm120_sliced_abi = census_fixed_sm120_sliced_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let fixed_sm120_postbias_abi = census_fixed_sm120_postbias_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let inference_sm89_driver_abi = inference_bundle::census_inference_sm89_driver_abi(
                request.ctx,
                request.module_kind,
                request.arch,
                &ptx_source,
            );
            let validation =
                validate_tf32_specialization(request.module_kind, request.arch, &ptx_source);
            let (tf32_driver_abi, tf32_qualification_error) =
                tf32_qualification_verdict(request.module_kind, extensions, census, validation);
            if !crate::mamba_ssm::gpu::kernel_identity::header_manifest_is_current(
                combined.as_bytes(),
                &include_paths,
                &header_manifest,
            ) {
                return Err(format!(
                    "{:?} CUDA headers changed during compilation; retry initialization",
                    request.module_kind
                ));
            }
            if let Some(domain) = nvrtc_library_domain.as_deref()
                && !crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain_is_current(domain)
            {
                return Err(format!(
                    "{:?} NVRTC libraries changed during compilation; retry initialization",
                    request.module_kind
                ));
            }
            let artifact_digest = FramedSha256::bytes(ptx_source.as_bytes());
            if trace {
                eprintln!(
                    "mamba-rs cache trace: {:?} miss compiled artifact={} ptx_len={} publish={}",
                    request.module_kind,
                    crate::mamba_ssm::gpu::kernel_identity::digest_hex(&artifact_digest),
                    ptx_source.len(),
                    cache_path.is_some() && cache_key.is_some()
                );
                if let Some(directory) = std::env::var_os("MAMBA_RS_CACHE_TRACE_DIR") {
                    let digest =
                        crate::mamba_ssm::gpu::kernel_identity::digest_hex(&artifact_digest);
                    let file = std::path::Path::new(&directory).join(format!(
                        "{:?}-{}.ptx",
                        request.module_kind,
                        &digest[..12]
                    ));
                    if let Err(error) = std::fs::write(&file, ptx_source.as_bytes()) {
                        eprintln!("mamba-rs cache trace: could not write {file:?}: {error}");
                    }
                }
            }
            if let (Some(path), Some(key)) = (&cache_path, cache_key) {
                crate::mamba_ssm::gpu::kernel_identity::publish_cache(
                    path,
                    key,
                    ArtifactKind::Ptx,
                    ptx_source.as_bytes(),
                );
            }
            let module = request
                .ctx
                .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
                .map_err(|error| {
                    format!("{:?} module load failed: {error:?}", request.module_kind)
                })?;
            (
                module,
                artifact_digest,
                tf32_qualification_error,
                tf32_driver_abi,
                sm89_half_abi,
                sm89_exact_f32_abi,
                sm89_exact_f32_d128_abi,
                sm89_tf32_joint_abi,
                fixed_half_abi,
                fixed_half_swizzle_abi,
                fixed_half_s3_abi,
                fixed_rna_wide_abi,
                fixed_finalist_abi,
                fixed_exact_n64_abi,
                fixed_cells_abi,
                fixed_sm120_exact_n64_abi,
                fixed_sm120_sliced_abi,
                fixed_sm120_postbias_abi,
                inference_sm89_driver_abi,
            )
        }
    };

    let compiler_identity = CompilerIdentity {
        source_digest: FramedSha256::bytes(combined.as_bytes()),
        invocation_digest,
        header_manifest_digest: FramedSha256::new(b"cuda-header-manifest.v1")
            .optional(b"manifest", header_manifest.as_deref())
            .finish(),
        target: CudaTarget::new(request.arch)?,
        nvrtc_version: nvrtc,
        nvrtc_library_domain: FramedSha256::new(b"nvrtc-library-set-identity.v2")
            .optional(b"domain", nvrtc_library_domain.as_deref())
            .finish(),
        nvrtc_library_known: nvrtc_library_domain.is_some(),
        output_kind: ArtifactKind::Ptx,
        composer_revision: crate::mamba_ssm::gpu::kernel_identity::COMPOSER_REVISION,
        compiler_revision: crate::mamba_ssm::gpu::kernel_identity::COMPILER_REVISION,
        numeric_abi_revision: crate::mamba_ssm::gpu::kernel_identity::NUMERIC_ABI_REVISION,
        schedule_revision: crate::mamba_ssm::gpu::kernel_identity::SCHEDULE_REVISION,
    };
    Ok(CompiledModule {
        module,
        compiler_identity,
        artifact_identity: ArtifactIdentity {
            module_kind: request.module_kind,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: invocation_digest,
            artifact_digest,
        },
        tf32_qualified: tf32_qualification_error.is_none(),
        tf32_qualification_error,
        tf32_driver_abi,
        sm89_half_driver_abi,
        sm89_exact_f32_driver_abi,
        sm89_exact_f32_d128_driver_abi,
        sm89_tf32_joint_driver_abi,
        fixed_sm89_half_driver_abi,
        fixed_sm89_half_swizzle_driver_abi,
        fixed_sm89_half_s3_driver_abi,
        fixed_sm89_rna_wide_driver_abi,
        fixed_sm89_finalist_driver_abi,
        fixed_sm89_exact_n64_driver_abi,
        fixed_sm89_cells_driver_abi,
        fixed_sm120_exact_n64_driver_abi,
        fixed_sm120_sliced_driver_abi,
        fixed_sm120_postbias_driver_abi,
        inference_sm89_driver_abi,
    })
}

/// The TF32 routes of a module qualify when the specialization validates
/// and the Driver ABI census covers every production symbol; otherwise the
/// first failing step is kept as the reason.
fn tf32_qualification_verdict(
    module_kind: ModuleKind,
    extensions: bool,
    census: Result<BTreeMap<&'static str, Tf32DriverAbi>, String>,
    validation: Result<(), String>,
) -> (BTreeMap<&'static str, Tf32DriverAbi>, Option<String>) {
    let (tf32_driver_abi, census_error) = match census {
        Ok(census) => (census, None),
        Err(error) => (BTreeMap::new(), Some(error)),
    };
    let error = validation.err().or(census_error).or_else(|| {
        (!complete_tf32_driver_abi(module_kind, extensions, &tf32_driver_abi))
            .then(|| format!("{module_kind:?} TF32 Driver ABI census is incomplete"))
    });
    (tf32_driver_abi, error)
}

pub(crate) fn compile_sm100_optional(
    ctx: &Arc<CudaContext>,
    state_cap: usize,
    device_cc: (i32, i32),
) -> Option<QualifiedSpecializedModule> {
    select_sm100_candidate(
        &super::dispatch::sm100_target_candidates_for_nvrtc(device_cc, nvrtc_version()),
        |candidate| probe_sm100_target(ctx, candidate),
        |candidate| {
            compile_module(CompileModuleRequest {
                ctx,
                arch: candidate.nvrtc_arch,
                state_cap,
                module_kind: ModuleKind::TriadSm100,
            })
        },
        qualify_specialized_module,
    )
}

fn sm100_target_candidates(
    device_cc: (i32, i32),
) -> &'static [super::contract::Sm100TargetCandidate] {
    super::dispatch::sm100_target_candidates(device_cc)
}

fn select_sm100_candidate<T, U>(
    candidates: &[super::contract::Sm100TargetCandidate],
    mut probe: impl FnMut(super::contract::Sm100TargetCandidate) -> Result<(), String>,
    mut compile: impl FnMut(super::contract::Sm100TargetCandidate) -> Result<T, String>,
    mut qualify: impl FnMut(T) -> Result<U, String>,
) -> Option<U> {
    for &candidate in candidates {
        let (stage, error) = match probe(candidate) {
            Err(error) => ("probe", error),
            Ok(()) => match compile(candidate) {
                Err(error) => ("compile", error),
                Ok(compiled) => match qualify(compiled) {
                    Err(error) => ("qualify", error),
                    Ok(qualified) => return Some(qualified),
                },
            },
        };
        static REJECTED: std::sync::Once = std::sync::Once::new();
        crate::mamba_ssm::gpu::diagnostics::warn_once(&REJECTED, || {
            format!(
                "SM100 candidate {candidate:?} rejected at {stage}: {error}; the next \
                 candidate or the portable kernels serve"
            )
        });
    }
    None
}

pub(crate) fn compile_sm120_artifact_set(
    ctx: &Arc<CudaContext>,
    state_cap: usize,
    device_cc: (i32, i32),
    nvrtc: (i32, i32),
) -> Option<Sm120ArtifactSet> {
    let candidates = super::dispatch::sm120_target_candidates(device_cc, nvrtc);
    if candidates.is_empty() {
        static NO_TARGET: std::sync::Once = std::sync::Once::new();
        crate::mamba_ssm::gpu::diagnostics::warn_once(&NO_TARGET, || {
            format!(
                "CUDA {}.{} has no SM120 target for compute capability {}.{}; the SM120 \
                 kernels are not compiled and the portable kernels serve",
                nvrtc.0, nvrtc.1, device_cc.0, device_cc.1
            )
        });
        return None;
    }
    let mut baseline = None;
    for &candidate in candidates {
        let specialized = compile_module(CompileModuleRequest {
            ctx,
            arch: candidate.nvrtc_arch,
            state_cap,
            module_kind: ModuleKind::TriadSm120,
        })
        .and_then(|module| {
            if module.compiler_identity.target.as_str() != candidate.nvrtc_arch {
                return Err("SM120 artifact transaction mixed CUDA targets".into());
            }
            qualify_specialized_module(module)
        });
        let (fixed, scalar, sm80) = match compile_sm120_baseline(ctx, state_cap, candidate) {
            Ok(baseline) => baseline,
            Err(error) => {
                static BASELINE: std::sync::Once = std::sync::Once::new();
                crate::mamba_ssm::gpu::diagnostics::warn_once(&BASELINE, || {
                    format!(
                        "the baseline modules failed to compile for {}: {error}",
                        candidate.nvrtc_arch
                    )
                });
                continue;
            }
        };
        let specialized = match specialized {
            Ok(specialized) => Some(specialized),
            Err(error) => {
                static REJECTED: std::sync::Once = std::sync::Once::new();
                crate::mamba_ssm::gpu::diagnostics::warn_once(&REJECTED, || {
                    format!(
                        "the SM120 module was rejected for {}: {error}; the portable kernels \
                         serve every GEMM on this board",
                        candidate.nvrtc_arch
                    )
                });
                None
            }
        };
        if let Some(mut specialized) = specialized {
            let complete = crate::mamba_ssm::gpu::kernel_identity::build_artifact_set(&[
                fixed.artifact_identity,
                scalar.artifact_identity,
                sm80.artifact_identity,
                specialized.module.artifact_identity,
            ]);
            let caps = query_sm120_device_caps(ctx, candidate, nvrtc);
            let resources = snapshot_sm120_resources(&specialized.functions);
            match (complete, caps, resources) {
                (Ok(_), Ok(caps), Ok(resources)) => {
                    specialized.sm120_target = Some(candidate);
                    specialized.sm120_device_caps = Some(caps);
                    specialized.sm120_resources = resources;
                    return Some(Sm120ArtifactSet {
                        fixed,
                        scalar,
                        sm80,
                        specialized: Some(specialized),
                    });
                }
                (complete, caps, resources) => {
                    let error = complete
                        .err()
                        .or_else(|| caps.err())
                        .or_else(|| resources.err())
                        .unwrap_or_default();
                    static UNBOUND: std::sync::Once = std::sync::Once::new();
                    crate::mamba_ssm::gpu::diagnostics::warn_once(&UNBOUND, || {
                        format!(
                            "the SM120 module compiled for {} but could not be bound \
                             ({error}); the portable kernels serve every GEMM on this board",
                            candidate.nvrtc_arch
                        )
                    });
                }
            }
        }
        if baseline.is_none() {
            baseline = Some(Sm120ArtifactSet {
                fixed,
                scalar,
                sm80,
                specialized: None,
            });
        }
    }
    baseline
}

fn compile_sm120_baseline(
    ctx: &Arc<CudaContext>,
    state_cap: usize,
    candidate: super::contract::Sm120TargetCandidate,
) -> Result<(CompiledModule, CompiledModule, CompiledModule), String> {
    let compile = |module_kind| {
        compile_module(CompileModuleRequest {
            ctx,
            arch: candidate.nvrtc_arch,
            state_cap,
            module_kind,
        })
    };
    let fixed = compile(ModuleKind::Fixed)?;
    let scalar = compile(ModuleKind::TriadScalar)?;
    let sm80 = compile(ModuleKind::TriadSm80)?;
    for module in [&fixed, &scalar, &sm80] {
        if module.compiler_identity.target.as_str() != candidate.nvrtc_arch {
            return Err("SM120 baseline artifact transaction mixed CUDA targets".into());
        }
    }
    crate::mamba_ssm::gpu::kernel_identity::build_artifact_set(&[
        fixed.artifact_identity,
        scalar.artifact_identity,
        sm80.artifact_identity,
    ])?;
    Ok((fixed, scalar, sm80))
}

fn query_sm120_device_caps(
    ctx: &Arc<CudaContext>,
    candidate: super::contract::Sm120TargetCandidate,
    nvrtc: (i32, i32),
) -> Result<crate::mamba_ssm::gpu::kernel_identity::DeviceCaps, String> {
    let (major, minor) = ctx
        .compute_capability()
        .map_err(|error| format!("query SM120 compute capability: {error:?}"))?;
    if (major, minor) != candidate.device_cc {
        return Err("SM120 candidate does not match the CUDA device minor".into());
    }
    let optin_shared = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .map_err(|error| format!("query SM120 opt-in shared memory: {error:?}"))?;
    let tensor_map_access = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
        )
        .map_err(|error| format!("query SM120 tensor-map support: {error:?}"))?
        != 0;
    Ok(crate::mamba_ssm::gpu::kernel_identity::DeviceCaps {
        compute_capability: (
            u32::try_from(major).map_err(|_| format!("negative CUDA CC major {major}"))?,
            u32::try_from(minor).map_err(|_| format!("negative CUDA CC minor {minor}"))?,
        ),
        nvrtc_version: nvrtc,
        accepted_target: Some(crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new(
            candidate.nvrtc_arch,
        )?),
        optin_shared_bytes: u32::try_from(optin_shared)
            .map_err(|_| format!("negative SM120 opt-in shared memory {optin_shared}"))?,
        tensor_map_access,
    })
}

fn snapshot_sm120_resources(
    functions: &HashMap<&'static str, CudaFunction>,
) -> Result<HashMap<&'static str, super::contract::Sm120KernelResources>, String> {
    let mut resources = HashMap::new();
    for spec in super::contract::sm120_kernel_specs() {
        let function = functions
            .get(spec.symbol)
            .ok_or_else(|| format!("SM120 resource census is missing {}", spec.symbol))?;
        let max_threads_per_block = u32::try_from(
            function
                .max_threads_per_block()
                .map_err(|error| format!("query {} max threads: {error:?}", spec.symbol))?,
        )
        .map_err(|_| format!("{} reports a negative max thread count", spec.symbol))?;
        let local_bytes = u32::try_from(
            function
                .local_size_bytes()
                .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?,
        )
        .map_err(|_| format!("{} reports negative local memory", spec.symbol))?;
        let registers_per_thread = u32::try_from(
            function
                .num_regs()
                .map_err(|error| format!("query {} registers: {error:?}", spec.symbol))?,
        )
        .map_err(|_| format!("{} reports a negative register count", spec.symbol))?;
        let active_blocks_per_sm = function
            .occupancy_max_active_blocks_per_multiprocessor(
                spec.threads,
                spec.dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
        resources.insert(
            spec.symbol,
            super::contract::Sm120KernelResources {
                threads: spec.threads,
                dynamic_shared_bytes: spec.dynamic_shared_bytes,
                max_threads_per_block,
                local_bytes,
                spill_store_bytes: 0,
                spill_load_bytes: 0,
                registers_per_thread,
                active_blocks_per_sm,
            },
        );
    }
    Ok(resources)
}

#[cfg(test)]
fn select_sm120_candidate<T>(
    candidates: &[super::contract::Sm120TargetCandidate],
    mut resolve: impl FnMut(
        super::contract::Sm120TargetCandidate,
    ) -> Result<(super::contract::Sm120TargetCandidate, T), String>,
) -> Option<(super::contract::Sm120TargetCandidate, T)> {
    for &candidate in candidates {
        let Ok((resolved, value)) = resolve(candidate) else {
            continue;
        };
        if resolved == candidate {
            return Some((resolved, value));
        }
    }
    None
}

const SM100_PROBE_SOURCE: &str = r#"
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

#if __CUDACC_VER_MAJOR__ >= 13
struct alignas(128) CUtensorMap {
#else
struct alignas(64) CUtensorMap {
#endif
    unsigned long long opaque[16];
};

static_assert(sizeof(CUtensorMap) == 128, "unexpected tensor map ABI");

static __device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

static __device__ __forceinline__ unsigned long long tcgen_desc(
    const void* p, unsigned lbo, unsigned sbo) {
    unsigned long long d =
        (static_cast<unsigned long long>(smem_addr(p)) >> 4) & 0x3fffULL;
    d |= static_cast<unsigned long long>(lbo & 0x3fffU) << 16;
    d |= static_cast<unsigned long long>(sbo & 0x3fffU) << 32;
    d |= 1ULL << 46;
    d |= 2ULL << 61;
    return d;
}

static __device__ __forceinline__ void wait_barrier(
    unsigned bar, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "p, [%1], %2; "
            "selp.b32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(bar), "r"(phase) : "memory");
    } while (!ready);
}

static __device__ __forceinline__ void issue_mma(
    unsigned tmem, unsigned long long a, unsigned long long b,
    unsigned idesc, unsigned accumulate) {
    unsigned zero = 0;
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %8, 0; "
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
        "{%4, %5, %6, %7}, p; }"
        :: "r"(tmem), "l"(a), "l"(b), "r"(idesc),
           "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(accumulate)
        : "memory");
}

extern "C" __global__ void tcgen05_probe(
    unsigned* out,
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b) {
    extern __shared__ __align__(1024) unsigned char smem[];
    __shared__ unsigned tmem_base;
    __shared__ __align__(8) unsigned long long barriers[2];

    unsigned full = smem_addr(&barriers[0]);
    unsigned done = smem_addr(&barriers[1]);
    unsigned warp = threadIdx.x >> 5;

    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"(full) : "memory");
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], 1;"
            :: "r"(done) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == 0) {
        unsigned dst = smem_addr(&tmem_base);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;"
            :: "r"(dst), "r"(128U) : "memory");
    }
    __syncthreads();

    unsigned values[8] = {
        threadIdx.x, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3,
        threadIdx.x + 4, threadIdx.x + 5, threadIdx.x + 6, threadIdx.x + 7,
    };
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "
        "{%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(tmem_base), "r"(values[0]), "r"(values[1]), "r"(values[2]),
           "r"(values[3]), "r"(values[4]), "r"(values[5]), "r"(values[6]),
           "r"(values[7]) : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned dst = smem_addr(smem + 1024);
        unsigned bar = full;
        unsigned long long a_map =
            reinterpret_cast<unsigned long long>(&map_a);
        unsigned long long b_map =
            reinterpret_cast<unsigned long long>(&map_b);
        int x = 0;
        int y = 0;
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
            "_, [%0], 32768;"
            :: "r"(bar) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile."
            "mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(dst), "l"(a_map), "r"(x), "r"(y), "r"(bar) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile."
            "mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(dst + 16384), "l"(b_map), "r"(x), "r"(y), "r"(bar)
            : "memory");

        wait_barrier(full, 0);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

        unsigned long long a_k = tcgen_desc(smem + 1024, 1, 64);
        unsigned long long a_mn = tcgen_desc(smem + 1024, 512, 64);
        unsigned long long b_k = tcgen_desc(smem + 17408, 1, 64);
        unsigned long long b_mn64 = tcgen_desc(smem + 17408, 0, 64);
        unsigned long long b_mn128 = tcgen_desc(smem + 17408, 512, 64);

        issue_mma(tmem_base, a_k, b_k, 0x08100010U, 1);
        issue_mma(tmem_base, a_k, b_mn64, 0x08110010U, 1);
        issue_mma(tmem_base, a_mn, b_mn64, 0x08118010U, 1);
        issue_mma(tmem_base, a_k, b_k, 0x08200010U, 1);
        issue_mma(tmem_base, a_k, b_mn128, 0x08210010U, 1);
        issue_mma(tmem_base, a_mn, b_mn128, 0x08218010U, 1);
        issue_mma(tmem_base, a_k, b_k, 0x08100490U, 1);
        issue_mma(tmem_base, a_k, b_mn64, 0x08110490U, 1);
        issue_mma(tmem_base, a_mn, b_mn64, 0x08118490U, 1);
        issue_mma(tmem_base, a_k, b_k, 0x08200490U, 1);
        issue_mma(tmem_base, a_k, b_mn128, 0x08210490U, 1);
        issue_mma(tmem_base, a_mn, b_mn128, 0x08218490U, 1);

        asm volatile(
            "tcgen05.commit.cta_group::1."
            "mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :: "r"(done) : "memory");
        wait_barrier(done, 0);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    }
    __syncthreads();

    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]),
          "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
          "=r"(values[6]), "=r"(values[7])
        : "r"(tmem_base) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    out[threadIdx.x] = values[0];
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    if (warp == 0) {
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
            ::: "memory");
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base), "r"(128U) : "memory");
    }
}

#endif
"#;

fn probe_sm100_target(
    ctx: &Arc<CudaContext>,
    candidate: super::contract::Sm100TargetCandidate,
) -> Result<(), String> {
    let options = cudarc::nvrtc::CompileOptions {
        arch: Some(candidate.nvrtc_arch),
        options: vec!["--fmad=true".to_string(), "-DNDEBUG".to_string()],
        ..Default::default()
    };
    let image =
        cudarc::nvrtc::compile_ptx_with_opts(SM100_PROBE_SOURCE, options).map_err(|error| {
            format!(
                "TriadSm100 target probe {} failed: {}",
                candidate.nvrtc_arch,
                format!("{error:?}").replace("\\n", "\n")
            )
        })?;
    let bytes = image
        .as_bytes()
        .ok_or_else(|| "TriadSm100 target probe returned no PTX image".to_string())?;
    let ptx = crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(bytes)?;
    validate_sm100_probe_ptx(candidate.nvrtc_arch, &ptx)?;
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
        .map_err(|error| format!("TriadSm100 target probe load failed: {error:?}"))?;
    module
        .load_function("tcgen05_probe")
        .map_err(|error| format!("TriadSm100 target probe symbol failed: {error:?}"))?;
    Ok(())
}

fn validate_module_target(kind: ModuleKind, arch: &str) -> Result<(), String> {
    if kind == ModuleKind::TriadSm80 && sm80_ptx_target(arch).is_none() {
        return Err(format!(
            "TriadSm80 requires an admitted SM80+ portable target, got {arch}"
        ));
    }
    if kind == ModuleKind::TriadSm90a && arch != "sm_90a" {
        return Err(format!(
            "TriadSm90a requires exact target sm_90a, got {arch}"
        ));
    }
    if kind == ModuleKind::InferenceSm89Cells && !fixed_portable_overlay_composed(arch) {
        return Err(format!(
            "InferenceSm89Cells requires an SM80-tier target outside the CC 12.x family, got {arch}"
        ));
    }
    if kind == ModuleKind::TriadSm89Finalist && sm80_ptx_target(arch).is_none() {
        return Err(format!(
            "TriadSm89Finalist requires an admitted SM80+ portable target, got {arch}"
        ));
    }
    if kind == ModuleKind::TriadSm89Half && sm80_ptx_target(arch).is_none() {
        return Err(format!(
            "TriadSm89Half requires an admitted SM80+ portable target, got {arch}"
        ));
    }
    if kind == ModuleKind::TriadSm89ExactF32 && sm80_ptx_target(arch).is_none() {
        return Err(format!(
            "TriadSm89ExactF32 requires an admitted SM80+ portable target, got {arch}"
        ));
    }
    if kind == ModuleKind::TriadSm89ExactF32D128 && sm80_ptx_target(arch).is_none() {
        return Err(format!(
            "TriadSm89ExactF32D128 requires an admitted SM80+ portable target, got {arch}"
        ));
    }
    if kind == ModuleKind::TriadSm89Tf32Joint && sm80_ptx_target(arch).is_none() {
        return Err(format!(
            "TriadSm89Tf32Joint requires an admitted SM80+ portable target, got {arch}"
        ));
    }
    if kind == ModuleKind::TriadSm100 && sm100_target_for_arch(arch).is_none() {
        return Err(format!(
            "TriadSm100 requires an admitted compute_100f/a, compute_103f/a, compute_107f/a, or compute_110f/a target, got {arch}"
        ));
    }
    if kind == ModuleKind::TriadSm120 && !matches!(arch, "compute_120" | "compute_121") {
        return Err(format!(
            "TriadSm120 requires generic target compute_120 or compute_121, got {arch}"
        ));
    }
    Ok(())
}

fn validate_tf32_ptx_inventory(
    module_kind: ModuleKind,
    extensions: bool,
    ptx: &str,
) -> Result<(), String> {
    let kernel_specs: Vec<_> =
        super::contract::tf32_route_specs_for(module_kind, extensions).collect();
    if kernel_specs.is_empty() {
        return Ok(());
    }
    let expected: BTreeSet<_> = kernel_specs
        .iter()
        .map(|kernel_spec| kernel_spec.symbol)
        .collect();
    if expected.len() != kernel_specs.len() {
        return Err(format!(
            "{module_kind:?} TF32 contract contains duplicate symbols"
        ));
    }
    let symbols = ptx_entry_symbols(ptx)?;
    // The split-K kernels are compiled into the same module but are named by
    // `Tf32SplitKSpec`, never by a route spec, so they are not part of this
    // inventory and must not read as foreign entries.
    let actual: Vec<_> = symbols
        .iter()
        .map(String::as_str)
        .filter(|symbol| {
            (symbol.contains("_tf32_") || symbol.contains("_tma_fma_"))
                && !symbol.contains("_splitk")
        })
        .collect();
    let unique: BTreeSet<_> = actual.iter().copied().collect();
    if actual.len() != unique.len() || unique != expected {
        let missing = expected.difference(&unique).copied().collect::<Vec<_>>();
        let foreign = unique.difference(&expected).copied().collect::<Vec<_>>();
        return Err(format!(
            "{module_kind:?} TF32 PTX inventory is incomplete, duplicated, or contains foreign \
             entries: missing {missing:?}, foreign {foreign:?}"
        ));
    }
    Ok(())
}

fn validate_sm89_finalist_ptx_inventory(ptx: &str) -> Result<(), String> {
    let original = "nt_sm80_mma_tf32_m128n64_bk32_s2";
    let mut expected = super::contract::tf32_module_symbols(ModuleKind::TriadSm80)
        .filter(|&symbol| symbol != original)
        .collect::<BTreeSet<_>>();
    if !expected.insert(super::sm89_finalist_source::SM89_FINALIST_SYMBOL) {
        return Err("TriadSm89Finalist contract contains a duplicate symbol".into());
    }
    let symbols = ptx_entry_symbols(ptx)?;
    // The split-K kernels ride along in the same source but belong to the
    // extension contract, which the finalist module never serves.
    let actual = symbols
        .iter()
        .map(String::as_str)
        .filter(|symbol| {
            (symbol.contains("_tf32_") || symbol.contains("_tma_fma_"))
                && !symbol.contains("_splitk")
        })
        .collect::<Vec<_>>();
    let unique = actual.iter().copied().collect::<BTreeSet<_>>();
    if actual.len() != unique.len() || unique != expected {
        let missing = expected.difference(&unique).copied().collect::<Vec<_>>();
        let foreign = unique.difference(&expected).copied().collect::<Vec<_>>();
        return Err(format!(
            "TriadSm89Finalist TF32 PTX inventory is incomplete, duplicated, or foreign: missing {missing:?}, foreign {foreign:?}"
        ));
    }
    Ok(())
}

fn validate_sm89_finalist_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    if sm80_ptx_target(arch).is_none_or(|target| ptx_target(ptx).ok().as_deref() != Some(target)) {
        return Err(
            "TriadSm89Finalist requires matching SM80+ portable source and PTX targets".into(),
        );
    }
    validate_sm89_finalist_ptx_inventory(ptx)?;
    let parsed = parse_ptx(ptx)?;
    let entry = parsed_ptx_entry_ref(&parsed, super::sm89_finalist_source::SM89_FINALIST_SYMBOL)?;
    require_ptx_entry_tokens(
        "TriadSm89Finalist",
        entry,
        &[
            "cvt.rna.tf32.f32",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        ],
    )?;
    if ptx_has_unquoted_token(&entry.body, |token| {
        token.starts_with("wgmma.")
            || token.starts_with("tcgen05.")
            || token.starts_with("cp.async.bulk.tensor.")
            || token.contains("tensormap")
    }) {
        return Err("TriadSm89Finalist contains a foreign TMA or tensor-core family".into());
    }
    Ok(())
}

fn validate_sm89_half_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    if sm80_ptx_target(arch).is_none_or(|target| ptx_target(ptx).ok().as_deref() != Some(target)) {
        return Err("TriadSm89Half requires matching SM80+ portable source and PTX targets".into());
    }
    let expected = super::sm89_half_source::runtime_kernel_specs()
        .map(|spec| spec.symbol)
        .collect::<BTreeSet<_>>();
    let symbols = ptx_entry_symbols(ptx)?;
    // The module composes nothing but its own kernels, so every PTX entry
    // must be one of the planned symbols.
    let actual = symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let unique = actual.iter().copied().collect::<BTreeSet<_>>();
    if actual.len() != unique.len() || unique != expected {
        return Err("TriadSm89Half PTX inventory is incomplete, duplicated, or foreign".into());
    }
    let parsed = parse_ptx(ptx)?;
    for spec in super::sm89_half_source::runtime_kernel_specs() {
        let entry = parsed_ptx_entry_ref(&parsed, spec.symbol)?;
        let mma = if spec.dtype == crate::mamba_ssm::gpu::dtype::WeightDtype::Bf16 {
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        } else {
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        };
        let (async_copy, loads, opposite_x4, opposite_x2) = match spec.route {
            super::sm89_half_source::Sm89HalfRuntimeRoute::Legacy(
                super::sm89_half_source::Sm89HalfRoute::NnM128N128Bk64S3,
            ) => (
                "cp.async.cg.shared.global",
                [
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
                ],
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
            ),
            super::sm89_half_source::Sm89HalfRuntimeRoute::Legacy(
                super::sm89_half_source::Sm89HalfRoute::TnM64N64Bk64S2CompactBxor
                | super::sm89_half_source::Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
            )
            | super::sm89_half_source::Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72
            | super::sm89_half_source::Sm89HalfRuntimeRoute::TnRelayM64N64Bk64S3 => (
                "cp.async.ca.shared.global",
                [
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
                ],
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
            ),
            super::sm89_half_source::Sm89HalfRuntimeRoute::TnD128InM32N16Bk64S4
            | super::sm89_half_source::Sm89HalfRuntimeRoute::TnD128OutM32N16Bk64S4 => (
                "cp.async.cg.shared.global",
                [
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
                ],
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
            ),
            super::sm89_half_source::Sm89HalfRuntimeRoute::NtSmallM16N64Bk64S4 => (
                "cp.async.ca.shared.global",
                [
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
                ],
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            ),
            super::sm89_half_source::Sm89HalfRuntimeRoute::NnSmallM16N64Bk64S4 => (
                "cp.async.ca.shared.global",
                [
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
                ],
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
            ),
            super::sm89_half_source::Sm89HalfRuntimeRoute::Legacy(
                super::sm89_half_source::Sm89HalfRoute::NtM128N128Bk64S3Bxor
                | super::sm89_half_source::Sm89HalfRoute::NtM96N128Bk64S3,
            ) => (
                "cp.async.cg.shared.global",
                [
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
                ],
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            ),
        };
        require_ptx_entry_tokens("TriadSm89Half", entry, &[async_copy, mma])?;
        require_ptx_entry_tokens("TriadSm89Half", entry, &loads)?;
        if ptx_has_unquoted_token(&entry.body, |token| {
            token == opposite_x4 || token == opposite_x2
        }) {
            return Err(format!(
                "TriadSm89Half/{} PTX contains a route-incompatible ldmatrix form",
                spec.symbol
            ));
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
                || token.starts_with("wgmma.")
                || token.starts_with("tcgen05.")
                || token.starts_with("cp.async.bulk.tensor.")
                || token.starts_with("ld.local.")
                || token.starts_with("st.local.")
        }) {
            return Err(format!(
                "TriadSm89Half {} contains a forbidden instruction family",
                spec.symbol
            ));
        }
    }
    Ok(())
}

fn validate_sm89_exact_f32_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    use super::sm89_exact_f32_source::Sm89ExactF32KernelKind;

    if sm80_ptx_target(arch).is_none_or(|target| ptx_target(ptx).ok().as_deref() != Some(target)) {
        return Err(
            "TriadSm89ExactF32 requires matching SM80+ portable source and PTX targets".into(),
        );
    }
    let expected = super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS
        .iter()
        .map(|spec| spec.symbol)
        .collect::<BTreeSet<_>>();
    let symbols = ptx_entry_symbols(ptx)?;
    let actual = symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let unique = actual.iter().copied().collect::<BTreeSet<_>>();
    if actual.len() != unique.len() || unique != expected {
        return Err("TriadSm89ExactF32 PTX inventory is incomplete, duplicated, or foreign".into());
    }

    let parsed = parse_ptx(ptx)?;
    for spec in super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS {
        let entry = parsed_ptx_entry_ref(&parsed, spec.symbol)?;
        let parameters = entry
            .text
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{} has no PTX parameter list", spec.symbol))?;
        let declarations = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect::<Vec<_>>();
        let abi_valid = match spec.kind {
            Sm89ExactF32KernelKind::DualChunkFusedFinalize => {
                declarations.len() == 4
                    && declarations[..3]
                        .iter()
                        .all(|line| line.starts_with(".param .u64 "))
                    && declarations[3].starts_with(".param .align 4 .b8 ")
                    && declarations[3].contains("[32]")
            }
            Sm89ExactF32KernelKind::DirectSplitMRaw => {
                declarations.len() == 7
                    && declarations[..3]
                        .iter()
                        .all(|line| line.starts_with(".param .u64 "))
                    && declarations[3..]
                        .iter()
                        .all(|line| line.starts_with(".param .u32 "))
            }
        };
        if !abi_valid {
            return Err(format!(
                "{} has the wrong static PTX parameter ABI",
                spec.symbol
            ));
        }

        let async_copy = match spec.kind {
            Sm89ExactF32KernelKind::DualChunkFusedFinalize => "cp.async.cg.shared.global",
            Sm89ExactF32KernelKind::DirectSplitMRaw => "cp.async.ca.shared.global",
        };
        require_ptx_entry_tokens("TriadSm89ExactF32", entry, &["fma.rn.f32", async_copy])?;
        match spec.kind {
            Sm89ExactF32KernelKind::DualChunkFusedFinalize => require_ptx_entry_tokens(
                "TriadSm89ExactF32 fused finalize",
                entry,
                &["add.rn.f64", "mul.rn.f64", "cvt.rn.f32.f64"],
            )?,
            Sm89ExactF32KernelKind::DirectSplitMRaw => {
                if ptx_has_unquoted_token(&entry.body, |token| {
                    matches!(token, "add.rn.f64" | "mul.rn.f64" | "cvt.rn.f32.f64")
                }) {
                    return Err(format!(
                        "{} raw partial unexpectedly contains the fused FP64 finalize",
                        spec.symbol
                    ));
                }
            }
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token == ".local"
                || token.starts_with("ld.local")
                || token.starts_with("st.local")
                || token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
                || token.starts_with("mma.")
                || token.starts_with("wgmma.")
                || token.starts_with("tcgen05.")
                || token.starts_with("cp.async.bulk.tensor.")
        }) {
            return Err(format!(
                "TriadSm89ExactF32 {} contains a forbidden instruction family",
                spec.symbol
            ));
        }
    }
    Ok(())
}

fn validate_sm89_exact_f32_d128_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    if sm80_ptx_target(arch).is_none_or(|target| ptx_target(ptx).ok().as_deref() != Some(target)) {
        return Err(
            "TriadSm89ExactF32D128 requires matching SM80+ portable source and PTX targets".into(),
        );
    }
    let expected = super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS
        .iter()
        .map(|spec| spec.symbol)
        .collect::<BTreeSet<_>>();
    let symbols = ptx_entry_symbols(ptx)?;
    let actual = symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let unique = actual.iter().copied().collect::<BTreeSet<_>>();
    if actual.len() != unique.len() || unique != expected {
        return Err(
            "TriadSm89ExactF32D128 PTX inventory is incomplete, duplicated, or foreign".into(),
        );
    }

    let parsed = parse_ptx(ptx)?;
    for spec in super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS {
        let entry = parsed_ptx_entry_ref(&parsed, spec.symbol)?;
        let parameters = entry
            .text
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{} has no PTX parameter list", spec.symbol))?;
        let declarations = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect::<Vec<_>>();
        let abi_valid = declarations.len() == 7
            && declarations[..3]
                .iter()
                .all(|line| line.starts_with(".param .u64 "))
            && declarations[3].starts_with(".param .f32 ")
            && declarations[4..]
                .iter()
                .all(|line| line.starts_with(".param .u32 "));
        if !abi_valid {
            return Err(format!(
                "{} has the wrong static PTX parameter ABI",
                spec.symbol
            ));
        }
        require_ptx_entry_tokens(
            "TriadSm89ExactF32D128 direct fold",
            entry,
            &[
                "fma.rn.f32",
                "add.rn.f64",
                "mul.rn.f64",
                "cvt.rn.f32.f64",
                "cp.async.cg.shared.global",
            ],
        )?;
        if ptx_has_unquoted_token(&entry.body, |token| {
            token == ".local"
                || token.starts_with("ld.local")
                || token.starts_with("st.local")
                || token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
                || token.starts_with("mma.")
                || token.starts_with("wmma.")
                || token.starts_with("wgmma.")
                || token.starts_with("tcgen05.")
                || token.starts_with("cp.async.bulk")
                || token.starts_with("cp.reduce.async.bulk")
                || token.contains("tensormap")
        }) {
            return Err(format!(
                "TriadSm89ExactF32D128 {} contains a forbidden instruction family",
                spec.symbol
            ));
        }
    }
    Ok(())
}

fn validate_sm89_tf32_joint_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    use super::sm89_tf32_joint_source::Sm89Tf32JointKernelKind;

    if sm80_ptx_target(arch).is_none_or(|target| ptx_target(ptx).ok().as_deref() != Some(target)) {
        return Err(
            "TriadSm89Tf32Joint requires matching SM80+ portable source and PTX targets".into(),
        );
    }
    let expected = super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS
        .iter()
        .map(|spec| spec.symbol)
        .collect::<BTreeSet<_>>();
    let symbols = ptx_entry_symbols(ptx)?;
    let actual = symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let unique = actual.iter().copied().collect::<BTreeSet<_>>();
    if actual.len() != unique.len() || unique != expected {
        return Err(
            "TriadSm89Tf32Joint PTX inventory is incomplete, duplicated, or foreign".into(),
        );
    }

    let parsed = parse_ptx(ptx)?;
    for spec in super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS {
        let entry = parsed_ptx_entry_ref(&parsed, spec.symbol)?;
        let parameters = entry
            .text
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{} has no PTX parameter list", spec.symbol))?;
        let declarations = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect::<Vec<_>>();
        let abi_valid = match spec.kind {
            Sm89Tf32JointKernelKind::TnPreRnaTranspose32x32 => {
                declarations.len() == 3
                    && declarations[..2]
                        .iter()
                        .all(|line| line.starts_with(".param .u64 "))
                    && declarations[2].starts_with(".param .align 4 .b8 ")
                    && declarations[2].contains("[12]")
            }
            _ => {
                declarations.len() == 5
                    && declarations[..4]
                        .iter()
                        .all(|line| line.starts_with(".param .u64 "))
                    && declarations[4].starts_with(".param .align 4 .b8 ")
                    && declarations[4].contains("[32]")
            }
        };
        if !abi_valid {
            return Err(format!(
                "{} has the wrong static PTX parameter ABI",
                spec.symbol
            ));
        }

        match spec.kind {
            Sm89Tf32JointKernelKind::TnPreRnaTranspose32x32 => {
                require_ptx_entry_tokens(
                    "TriadSm89Tf32Joint transpose",
                    entry,
                    &["cvt.rna.tf32.f32"],
                )?;
                if ptx_has_unquoted_token(&entry.body, |token| {
                    token.starts_with("mma.") || token.starts_with("cp.async.")
                }) {
                    return Err(format!(
                        "{} transpose contains a foreign GEMM instruction",
                        spec.symbol
                    ));
                }
            }
            Sm89Tf32JointKernelKind::NtRnaM144N96Bk32S2 => {
                require_ptx_entry_tokens(
                    "TriadSm89Tf32Joint NT RNA wide GEMM",
                    entry,
                    &[
                        "cp.async.cg.shared.global.L2::128B",
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                        "cvt.rna.tf32.f32",
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                    ],
                )?;
            }
            Sm89Tf32JointKernelKind::NtRowstageM128N192Bk32S2 => {
                require_ptx_entry_tokens(
                    "TriadSm89Tf32Joint NT row-staged wide GEMM",
                    entry,
                    &[
                        "cp.async.cg.shared.global.L2::128B",
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                    ],
                )?;
                if ptx_has_unquoted_token(&entry.body, |token| token == "cvt.rna.tf32.f32") {
                    return Err(format!(
                        "{} add-half NT route unexpectedly uses pre-RNA conversion",
                        spec.symbol
                    ));
                }
            }
            Sm89Tf32JointKernelKind::TnDirectM192N192Bk32S2 => {
                require_ptx_entry_tokens(
                    "TriadSm89Tf32Joint TN direct wide GEMM",
                    entry,
                    &[
                        "cp.async.cg.shared.global.L2::128B",
                        "cvt.rna.tf32.f32",
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                    ],
                )?;
            }
            Sm89Tf32JointKernelKind::TnPreRnaM128N96Bk32S3
            | Sm89Tf32JointKernelKind::TnPreRnaM64N64Bk32S3
            | Sm89Tf32JointKernelKind::TnPreRnaM64N96Bk32S2
            | Sm89Tf32JointKernelKind::TnPreRnaM96N192Bk32S2
            | Sm89Tf32JointKernelKind::TnPreRnaM96N96Bk32S3 => {
                require_ptx_entry_tokens(
                    "TriadSm89Tf32Joint pre-RNA GEMM",
                    entry,
                    &[
                        "cp.async.cg.shared.global.L2::128B",
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                        "cvt.rna.tf32.f32",
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                    ],
                )?;
            }
            Sm89Tf32JointKernelKind::NtALdmatrixM128N96Bk32S3 => {
                require_ptx_entry_tokens(
                    "TriadSm89Tf32Joint NT ldmatrix GEMM",
                    entry,
                    &[
                        "cp.async.cg.shared.global.L2::128B",
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                    ],
                )?;
                if ptx_has_unquoted_token(&entry.body, |token| token == "cvt.rna.tf32.f32") {
                    return Err(format!(
                        "{} add-half NT route unexpectedly uses pre-RNA conversion",
                        spec.symbol
                    ));
                }
            }
            Sm89Tf32JointKernelKind::NnAddHalfDirectM128N96Bk32S3
            | Sm89Tf32JointKernelKind::NnAddHalfM128N96Bk32S3 => {
                require_ptx_entry_tokens(
                    "TriadSm89Tf32Joint add-half GEMM",
                    entry,
                    &[
                        "cp.async.cg.shared.global.L2::128B",
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                    ],
                )?;
                if ptx_has_unquoted_token(&entry.body, |token| token == "cvt.rna.tf32.f32") {
                    return Err(format!(
                        "{} add-half route unexpectedly uses pre-RNA conversion",
                        spec.symbol
                    ));
                }
            }
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token == ".local"
                || token.starts_with("ld.local")
                || token.starts_with("st.local")
                || token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
                || token.starts_with("wgmma.")
                || token.starts_with("tcgen05.")
                || token.starts_with("cp.async.bulk.tensor.")
        }) {
            return Err(format!(
                "TriadSm89Tf32Joint {} contains a forbidden instruction family",
                spec.symbol
            ));
        }
    }
    Ok(())
}

fn validate_module_ptx(module_kind: ModuleKind, arch: &str, ptx: &str) -> Result<(), String> {
    match module_kind {
        ModuleKind::Fixed => {
            validate_fixed_tf32_ptx(arch, ptx)?;
            validate_fixed_sm89_rna_wide_ptx(arch, ptx)?;
            validate_fixed_sm89_finalist_ptx(arch, ptx)?;
            validate_fixed_sm89_half_ptx(arch, ptx)?;
            validate_fixed_sm89_half_swizzle_ptx(arch, ptx)?;
            validate_fixed_sm89_half_s3_ptx(arch, ptx)?;
            validate_fixed_sm89_exact_n64_ptx(arch, ptx)?;
            validate_fixed_sm89_cells_ptx(ModuleKind::Fixed, arch, ptx)?;
            validate_fixed_sm120_exact_n64_ptx(arch, ptx)?;
            validate_fixed_sm120_sliced_ptx(arch, ptx)?;
            validate_fixed_sm120_postbias_ptx(arch, ptx)
        }
        ModuleKind::TriadScalar => {
            validate_exact_ptx_exports("TriadScalar", SCALAR_SYMBOLS.len(), SCALAR_SYMBOLS, ptx)?;
            validate_scalar_zero_reduction_ptx(ptx)?;
            validate_scalar_nn_m32n64_splitk32_ptx(ptx)?;
            validate_scalar_nt_m2n16_ptx(ptx)?;
            validate_scalar_tn_m16n16_ptx(ptx)?;
            validate_tn_narrow_splitm_partial_ptx(ptx)?;
            validate_tn_splitm_partial_ptx(ptx)
        }
        ModuleKind::TriadSm80 => validate_sm80_ptx(arch, ptx),
        ModuleKind::TriadSm89Finalist => validate_sm89_finalist_ptx(arch, ptx),
        ModuleKind::TriadSm89Half => validate_sm89_half_ptx(arch, ptx),
        ModuleKind::TriadSm89ExactF32 => validate_sm89_exact_f32_ptx(arch, ptx),
        ModuleKind::TriadSm89ExactF32D128 => validate_sm89_exact_f32_d128_ptx(arch, ptx),
        ModuleKind::TriadSm89Tf32Joint => validate_sm89_tf32_joint_ptx(arch, ptx),
        ModuleKind::TriadSm90a => validate_sm90a_ptx(ptx),
        ModuleKind::TriadSm100 => validate_sm100_ptx(arch, ptx),
        ModuleKind::TriadSm120 => validate_sm120_ptx(arch, ptx),
        ModuleKind::InferenceSm89Cells => {
            validate_fixed_sm89_cells_ptx(ModuleKind::InferenceSm89Cells, arch, ptx)
        }
        ModuleKind::Mamba3Combined => Ok(()),
    }
}

const FIXED_TF32_SYMBOLS: [&str; 5] = [
    "nn_tf32_m128n64_bk32_s2",
    "nn_tf32_m128n64_bk32_s3",
    "nn_tf32_m64n64_bk32_s2",
    "nn_tf32_m64n64_bk32_s3",
    "nn_tf32_m16n32_bk32_s4",
];

pub(crate) const FIXED_SM89_RNA_WIDE_SYMBOL: &str = "nn_rna_wide_tf32_m128n128_bk32_s3";
const FIXED_SM89_RNA_WIDE_SHARED_BYTES: u32 = 98_304;
const FIXED_SM89_RNA_WIDE_THREADS: u32 = 256;
const FIXED_SM89_RNA_WIDE_REGISTER_CAP: u32 = 224;

pub(crate) const FIXED_SM89_RNA_N96_SYMBOL: &str = "nn_sm89_rna_tf32_m128n96_bk32_s3";
pub(crate) const FIXED_SM89_HALF_M64N64_S3_SYMBOL: &str = "nn_sm89_m64n64_bk64_s3_f16";
pub(crate) const FIXED_SM89_HALF_M128N64_S2_SYMBOL: &str = "nn_sm89_m128n64_bk64_s2_f16";

const FIXED_SM89_RNA_N96_SHARED_BYTES: u32 = 86_016;
const FIXED_SM89_RNA_N96_THREADS: u32 = 256;
const FIXED_SM89_RNA_N96_REGISTER_CAP: u32 = 136;
const FIXED_SM89_HALF_N64_SHARED_BYTES: u32 = 49_152;
const FIXED_SM89_HALF_N64_THREADS: u32 = 128;
const FIXED_SM89_HALF_M64N64_S3_REGISTER_CAP: u32 = 110;
const FIXED_SM89_HALF_M128N64_S2_REGISTER_CAP: u32 = 132;

/// Whether the Fixed module composed for `arch` carries the Ada-found
/// inference overlay: every sm_80-tier target except the CC 12.x family,
/// whose Fixed module stays byte-identical to the one its own frozen
/// cohorts were minted against.
pub(crate) fn fixed_portable_overlay_composed(arch: &str) -> bool {
    sm80_ptx_target(arch).is_some()
        && !matches!(arch, "sm_120" | "compute_120" | "sm_121" | "compute_121")
}

fn fixed_sm89_rna_wide_composed(arch: &str) -> bool {
    fixed_portable_overlay_composed(arch)
}

fn validate_fixed_sm89_rna_wide_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let parsed = parse_ptx(ptx)?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .filter(|entry| entry.symbol.contains("_rna_wide_tf32_"))
        .collect();
    let expected = usize::from(fixed_sm89_rna_wide_composed(arch));
    if actual.len() != expected || (expected == 1 && actual[0].symbol != FIXED_SM89_RNA_WIDE_SYMBOL)
    {
        return Err(format!(
            "Fixed SM89 RNA-wide PTX inventory is incomplete, duplicated, or foreign on {arch}"
        ));
    }
    if expected == 0 {
        return Ok(());
    }
    if super::super::gemm_bi_inference::FIXED_TF32_WIDE_PARAMS_SIZE != 32 {
        return Err("Fixed RNA-wide host parameter ABI drifted".into());
    }
    let entry = actual[0];
    let header = entry
        .text
        .split_once('{')
        .map(|(header, _)| header)
        .ok_or_else(|| format!("{} has no PTX body", entry.symbol))?;
    let tokens = ptx_tokens(header);
    let text: Vec<_> = tokens.iter().map(|token| token.text).collect();
    let begin = text
        .iter()
        .position(|token| *token == "(")
        .ok_or_else(|| format!("{} has no PTX parameters", entry.symbol))?;
    let end = text
        .iter()
        .position(|token| *token == ")")
        .ok_or_else(|| format!("{} has no PTX parameter end", entry.symbol))?;
    let declarations: Vec<_> = text[begin + 1..end].split(|token| *token == ",").collect();
    let pointer = |declaration: &&[&str]| {
        (declaration.len() == 3 && declaration[..2] == [".param", ".u64"])
            || (declaration.len() == 6
                && declaration[..5] == [".param", ".u64", ".ptr", ".align", "1"])
    };
    if declarations.len() != 5
        || !declarations[..4].iter().all(pointer)
        || declarations[4].len() != 8
        || declarations[4][..4] != [".param", ".align", "4", ".b8"]
        || declarations[4][5..] != ["[", "32", "]"]
    {
        return Err(format!(
            "{} requires four pointers and an align-4 32-byte bundle",
            entry.symbol
        ));
    }
    for (directive, expected_value) in [(".maxntid", "256"), (".minnctapersm", "1")] {
        let positions: Vec<_> = text
            .iter()
            .enumerate()
            .filter_map(|(index, token)| (*token == directive).then_some(index))
            .collect();
        if positions.len() != 1 || text.get(positions[0] + 1).copied() != Some(expected_value) {
            return Err(format!(
                "{} has the wrong {directive} launch bound",
                entry.symbol
            ));
        }
    }
    for required in [
        "cvt.rna.tf32.f32",
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        "cp.async.commit_group",
        "cp.async.wait_group",
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
    ] {
        if !ptx_has_unquoted_token(&entry.body, |token| token == required) {
            return Err(format!("{} is missing {required}", entry.symbol));
        }
    }
    if !ptx_has_unquoted_token(&entry.body, |token| {
        token.starts_with("cp.async.cg.shared.global")
    }) {
        return Err(format!(
            "{} is missing cp.async.cg.shared.global",
            entry.symbol
        ));
    }
    if ptx_has_unquoted_token(&entry.body, |token| {
        token == ".local"
            || token.starts_with("ld.local")
            || token.starts_with("st.local")
            || token.starts_with("atom.")
            || token.starts_with("atom::")
            || token.starts_with("red.")
            || token.starts_with("red::")
            || token.starts_with("redux.")
    }) {
        return Err(format!(
            "{} contains local memory, a numeric atomic, or a reduction",
            entry.symbol
        ));
    }
    Ok(())
}

const FIXED_SM89_CELL_PREFIXES: [&str; 5] = [
    "nn_sm89_m112n128_",
    "nn_sm89_m128n144_",
    "nn_sm89_m128n96_",
    "nn_sm89_m64n288_",
    "nn_sm89_m64n96_",
];

fn validate_fixed_sm89_cell_entry_abi(entry: &ParsedPtxEntry, symbol: &str) -> Result<(), String> {
    let parameters = entry
        .text
        .split_once('(')
        .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
        .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
    let declarations = parameters
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with(".param "))
        .collect::<Vec<_>>();
    if declarations.len() != 5
        || !declarations[..4]
            .iter()
            .all(|line| line.starts_with(".param .u64 "))
        || !declarations[4].starts_with(".param .align 4 .b8 ")
        || !declarations[4].contains("[32]")
    {
        return Err(format!(
            "{symbol} has the wrong four-pointer/32-byte bundle PTX ABI"
        ));
    }
    Ok(())
}

/// The Ada inference cells travel with the portable overlay: every cell
/// symbol is present exactly once where the overlay is composed and absent
/// elsewhere, on its family's instruction contract.
fn validate_fixed_sm89_cells_ptx(kind: ModuleKind, arch: &str, ptx: &str) -> Result<(), String> {
    use super::super::gemm_bi_inference::sm89_cells::{SM89_CELL_SPECS, Sm89CellFamily};

    let parsed = parse_ptx(ptx)?;
    let expected: BTreeSet<_> =
        if kind == ModuleKind::InferenceSm89Cells && fixed_portable_overlay_composed(arch) {
            SM89_CELL_SPECS.iter().map(|spec| spec.symbol).collect()
        } else {
            BTreeSet::new()
        };
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .map(|entry| entry.symbol.as_str())
        .filter(|symbol| {
            FIXED_SM89_CELL_PREFIXES
                .iter()
                .any(|prefix| symbol.starts_with(prefix))
        })
        .collect();
    let unique: BTreeSet<_> = actual.iter().copied().collect();
    if actual.len() != unique.len() || unique != expected {
        return Err(format!(
            "{kind:?} SM89 cell PTX inventory is incomplete, duplicated, or foreign on {arch}"
        ));
    }
    for spec in SM89_CELL_SPECS
        .iter()
        .filter(|spec| expected.contains(spec.symbol))
    {
        let entry = parsed_ptx_entry_ref(&parsed, spec.symbol)?;
        validate_fixed_sm89_cell_entry_abi(entry, spec.symbol)?;
        let half_mma = if spec.input == crate::mamba_ssm::gpu::dtype::WeightDtype::Bf16 {
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        } else {
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        };
        let required: &[&str] = match spec.family {
            Sm89CellFamily::ExactFma => &["fma.rn.f32", "cp.async.cg.shared.global"],
            Sm89CellFamily::HalfMma => &[
                half_mma,
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
                "cp.async.cg.shared.global",
            ],
            Sm89CellFamily::Tf32Mma => &[
                "cvt.rna.tf32.f32",
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                "cp.async.cg.shared.global",
            ],
        };
        require_ptx_entry_tokens("Fixed SM89 cell", entry, required)?;
        if ptx_has_unquoted_token(&entry.body, |token| {
            token == ".local"
                || token.starts_with("ld.local")
                || token.starts_with("st.local")
                || token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
                || token.starts_with("wgmma.")
                || token.starts_with("tcgen05.")
                || token.starts_with("cp.async.bulk")
        }) {
            return Err(format!(
                "Fixed SM89 cell {} contains a forbidden instruction family",
                spec.symbol
            ));
        }
    }
    Ok(())
}

fn validate_fixed_sm89_finalist_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let parsed = parse_ptx(ptx)?;
    let expected: BTreeSet<_> = if fixed_portable_overlay_composed(arch) {
        [
            FIXED_SM89_RNA_N96_SYMBOL,
            FIXED_SM89_HALF_M64N64_S3_SYMBOL,
            FIXED_SM89_HALF_M128N64_S2_SYMBOL,
        ]
        .into_iter()
        .collect()
    } else {
        BTreeSet::new()
    };
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .map(|entry| entry.symbol.as_str())
        .filter(|symbol| {
            symbol.contains("_sm89_rna_tf32_m128n96_")
                || symbol.contains("_sm89_m64n64_bk64_s3_")
                || symbol.contains("_sm89_m128n64_bk64_s2_")
        })
        .collect();
    let unique: BTreeSet<_> = actual.iter().copied().collect();
    if actual.len() != unique.len() || unique != expected {
        return Err(format!(
            "Fixed SM89 finalist PTX inventory is incomplete, duplicated, or foreign on {arch}"
        ));
    }
    Ok(())
}

fn validate_fixed_sm89_finalist_entry_ptx(
    parsed: &ParsedPtx,
    symbol: &'static str,
) -> Result<(), String> {
    let entry = parsed_ptx_entry_ref(parsed, symbol)?;
    let header = entry
        .text
        .split_once('{')
        .map(|(header, _)| header)
        .ok_or_else(|| format!("{symbol} has no PTX body"))?;
    let tokens = ptx_tokens(header);
    let text: Vec<_> = tokens.iter().map(|token| token.text).collect();
    let begin = text
        .iter()
        .position(|token| *token == "(")
        .ok_or_else(|| format!("{symbol} has no PTX parameters"))?;
    let end = text
        .iter()
        .position(|token| *token == ")")
        .ok_or_else(|| format!("{symbol} has no PTX parameter end"))?;
    let declarations: Vec<_> = text[begin + 1..end].split(|token| *token == ",").collect();
    let pointer = |declaration: &&[&str]| {
        (declaration.len() == 3 && declaration[..2] == [".param", ".u64"])
            || (declaration.len() == 6
                && declaration[..5] == [".param", ".u64", ".ptr", ".align", "1"])
    };
    if declarations.len() != 5
        || !declarations[..4].iter().all(pointer)
        || declarations[4].len() != 8
        || declarations[4][..4] != [".param", ".align", "4", ".b8"]
        || declarations[4][5..] != ["[", "32", "]"]
    {
        return Err(format!(
            "{symbol} requires four pointers and an align-4 32-byte bundle"
        ));
    }
    let n96 = symbol == FIXED_SM89_RNA_N96_SYMBOL;
    for (directive, expected_value) in [
        (".maxntid", if n96 { "256" } else { "128" }),
        (".minnctapersm", if n96 { "1" } else { "2" }),
    ] {
        let positions: Vec<_> = text
            .iter()
            .enumerate()
            .filter_map(|(index, token)| (*token == directive).then_some(index))
            .collect();
        if positions.len() != 1 || text.get(positions[0] + 1).copied() != Some(expected_value) {
            return Err(format!("{symbol} has the wrong {directive} launch bound"));
        }
    }
    let mma = if n96 {
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
    } else {
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
    };
    for required in [mma, "cp.async.commit_group", "cp.async.wait_group"] {
        if !ptx_has_unquoted_token(&entry.body, |token| token == required) {
            return Err(format!("{symbol} is missing {required}"));
        }
    }
    if n96 && !ptx_has_unquoted_token(&entry.body, |token| token == "cvt.rna.tf32.f32") {
        return Err(format!("{symbol} is missing cvt.rna.tf32.f32"));
    }
    if !ptx_has_unquoted_token(&entry.body, |token| {
        token.starts_with("cp.async.cg.shared.global")
    }) {
        return Err(format!("{symbol} is missing cp.async.cg.shared.global"));
    }
    if ptx_has_unquoted_token(&entry.body, |token| {
        token == ".local"
            || token.starts_with("ld.local")
            || token.starts_with("st.local")
            || token.starts_with("atom.")
            || token.starts_with("atom::")
            || token.starts_with("red.")
            || token.starts_with("red::")
            || token.starts_with("redux.")
    }) {
        return Err(format!(
            "{symbol} contains local memory, a numeric atomic, or a reduction"
        ));
    }
    Ok(())
}

const FIXED_SM120_TF32_SYMBOLS: [&str; 7] = [
    "nn_sm120_tma_tf32_m128n64_bk32_s2",
    "nn_sm120_tma_tf32_m128n64_bk32_s3",
    "nn_sm120_tma_tf32_m64n128_bk32_s2",
    "nn_sm120_tma_tf32_m64n128_bk32_s3",
    "nn_sm120_tma_tf32_m64n64_bk32_s2_producer_warp",
    "nn_sm120_tma_tf32_m64n64_bk32_s2",
    "nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store",
];

const FIXED_SM120_HALF_BASES: [&str; 5] = [
    "nn_sm120_tma_64x64_bk64_s2",
    "nn_sm120_tma_64x128_bk64_s2",
    "nn_sm120_tma_128x64_bk32_s3",
    "nn_sm120_tma_128x128_bk32_s2",
    "nn_sm120_tma_128x128_bk32_s3",
];

const FIXED_SM89_HALF_SYMBOLS: [&str; 2] =
    ["nn_sm89_tc128_pipeline_bf16", "nn_sm89_tc128_pipeline_f16"];
const FIXED_SM89_HALF_SHARED_BYTES: u32 = 71_680;
const FIXED_SM89_HALF_THREADS: u32 = 256;
const FIXED_SM89_HALF_REGISTER_CAP: u32 = 224;
const FIXED_SM89_HALF_SWIZZLE_SYMBOLS: [&str; 2] =
    ["nn_sm89_tc128_swizzle_bf16", "nn_sm89_tc128_swizzle_f16"];
const FIXED_SM89_HALF_SWIZZLE_SHARED_BYTES: u32 = 69_632;
const FIXED_SM89_HALF_SWIZZLE_THREADS: u32 = 256;
const FIXED_SM89_HALF_SWIZZLE_REGISTER_CAP: u32 = 224;

const FIXED_SM89_HALF_S3_SYMBOLS: [&str; 2] = ["nn_sm89_tc128_s3_bf16", "nn_sm89_tc128_s3_f16"];
const FIXED_SM89_HALF_S3_SHARED_BYTES: u32 = 98_304;
const FIXED_SM89_HALF_S3_THREADS: u32 = 256;
// Production NVRTC BF16/F16: 182 on CUDA12.8/13.0, 188 on CUDA13.2.
const FIXED_SM89_HALF_S3_REGISTER_CAP: u32 = 188;

fn fixed_sm89_half_composed(arch: &str) -> bool {
    fixed_portable_overlay_composed(arch)
}

fn validate_fixed_sm89_half_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let parsed = parse_ptx(ptx)?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .map(|entry| entry.symbol.as_str())
        .filter(|symbol| symbol.starts_with("nn_sm89_tc128_pipeline"))
        .collect();
    let expected: BTreeSet<_> = if fixed_sm89_half_composed(arch) {
        FIXED_SM89_HALF_SYMBOLS.into_iter().collect()
    } else {
        BTreeSet::new()
    };
    let unique: BTreeSet<_> = actual.iter().copied().collect();
    if actual.len() != unique.len() || unique != expected {
        return Err(format!(
            "Fixed SM89 half pipeline PTX inventory is incomplete, duplicated, or foreign on {arch}"
        ));
    }
    for symbol in expected {
        let entry = parsed_ptx_entry_ref(&parsed, symbol)?;
        let parameters = entry
            .text
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
        let declarations: Vec<_> = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect();
        if declarations.len() != 5
            || !declarations[..4]
                .iter()
                .all(|line| line.starts_with(".param .u64 "))
            || !declarations[4].starts_with(".param .align 4 .b8 ")
            || !declarations[4].contains("[32]")
        {
            return Err(format!(
                "{symbol} requires four pointers and an align-4 32-byte bundle"
            ));
        }
        let mma = if symbol.ends_with("_bf16") {
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        } else {
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        };
        for required in [
            mma,
            "cp.async.cg.shared.global",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
        ] {
            if !ptx_has_unquoted_token(&entry.body, |token| token == required) {
                return Err(format!("{symbol} is missing {required}"));
            }
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
        }) {
            return Err(format!("{symbol} contains a numeric atomic or reduction"));
        }
    }
    Ok(())
}

fn validate_fixed_sm89_half_swizzle_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let parsed = parse_ptx(ptx)?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .map(|entry| entry.symbol.as_str())
        .filter(|symbol| symbol.starts_with("nn_sm89_tc128_swizzle"))
        .collect();
    let expected: BTreeSet<_> = if fixed_sm89_half_composed(arch) {
        FIXED_SM89_HALF_SWIZZLE_SYMBOLS.into_iter().collect()
    } else {
        BTreeSet::new()
    };
    let unique: BTreeSet<_> = actual.iter().copied().collect();
    if actual.len() != unique.len() || unique != expected {
        return Err(format!(
            "Fixed SM89 half swizzle PTX inventory is incomplete, duplicated, or foreign on {arch}"
        ));
    }
    for symbol in expected {
        let entry = parsed_ptx_entry_ref(&parsed, symbol)?;
        let parameters = entry
            .text
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
        let declarations: Vec<_> = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect();
        if declarations.len() != 5
            || !declarations[..4]
                .iter()
                .all(|line| line.starts_with(".param .u64 "))
            || !declarations[4].starts_with(".param .align 4 .b8 ")
            || !declarations[4].contains("[32]")
        {
            return Err(format!(
                "{symbol} requires four pointers and an align-4 32-byte bundle"
            ));
        }
        let mma = if symbol.ends_with("_bf16") {
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        } else {
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        };
        for required in [
            mma,
            "cp.async.cg.shared.global",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
        ] {
            if !ptx_has_unquoted_token(&entry.body, |token| token == required) {
                return Err(format!("{symbol} is missing {required}"));
            }
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token == ".local"
                || token.starts_with("ld.local")
                || token.starts_with("st.local")
                || token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
        }) {
            return Err(format!(
                "{symbol} contains local memory, a numeric atomic, or a reduction"
            ));
        }
    }
    Ok(())
}

fn validate_fixed_sm89_half_s3_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let parsed = parse_ptx(ptx)?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .map(|entry| entry.symbol.as_str())
        .filter(|symbol| symbol.starts_with("nn_sm89_tc128_s3"))
        .collect();
    let expected: BTreeSet<_> = if fixed_sm89_half_composed(arch) {
        FIXED_SM89_HALF_S3_SYMBOLS.into_iter().collect()
    } else {
        BTreeSet::new()
    };
    let unique: BTreeSet<_> = actual.iter().copied().collect();
    if actual.len() != unique.len() || unique != expected {
        return Err(format!(
            "Fixed SM89 half s3 PTX inventory is incomplete, duplicated, or foreign on {arch}"
        ));
    }
    for symbol in expected {
        let entry = parsed_ptx_entry_ref(&parsed, symbol)?;
        let parameters = entry
            .text
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
        let declarations: Vec<_> = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect();
        if declarations.len() != 5
            || !declarations[..4]
                .iter()
                .all(|line| line.starts_with(".param .u64 "))
            || !declarations[4].starts_with(".param .align 4 .b8 ")
            || !declarations[4].contains("[32]")
        {
            return Err(format!(
                "{symbol} requires four pointers and an align-4 32-byte bundle"
            ));
        }
        let mma = if symbol.ends_with("_bf16") {
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        } else {
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        };
        for required in [
            mma,
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "bar.sync",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
        ] {
            if !ptx_has_unquoted_token(&entry.body, |token| token == required) {
                return Err(format!("{symbol} is missing {required}"));
            }
        }
        let schedule = ptx_tokens(&entry.body)
            .iter()
            .filter(|token| !token.text.starts_with(char::from(34)))
            .map(|token| token.text)
            .collect::<Vec<_>>()
            .join(" ");
        for wait in ["cp.async.wait_group 0 ;", "cp.async.wait_group 1 ;"] {
            if !schedule.contains(wait) {
                return Err(format!("{symbol} is missing {wait}"));
            }
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token == ".local"
                || token.starts_with("ld.local")
                || token.starts_with("st.local")
                || token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
        }) {
            return Err(format!(
                "{symbol} contains local memory, a numeric atomic, or a reduction"
            ));
        }
    }
    Ok(())
}

fn validate_fixed_sm89_half_driver_abi(symbol: &str, abi: &Tf32DriverAbi) -> Result<(), String> {
    const EXPECTED: [(usize, usize); 5] = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    if super::super::gemm_bi_inference::FIXED_SM89_HALF_PARAMS_SIZE != 32 {
        return Err("Fixed SM89 half host parameter ABI drifted".into());
    }
    if abi.parameter_count() != EXPECTED.len()
        || !abi
            .parameters()
            .iter()
            .zip(EXPECTED)
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{symbol} has the wrong live five-argument/64-byte Driver ABI"
        ));
    }
    Ok(())
}

fn validate_sm89_half_driver_abi(
    spec: &super::sm89_half_source::Sm89HalfRuntimeSpec,
    abi: &Tf32DriverAbi,
) -> Result<(), String> {
    const NN: [(usize, usize); 5] = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    const NT: [(usize, usize); 7] = [(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)];
    // The relay adds the hand-off slab and the flag word to the TN list.
    const RELAY: [(usize, usize); 9] = [
        (0, 8),
        (8, 8),
        (16, 8),
        (24, 4),
        (28, 4),
        (32, 4),
        (36, 4),
        (40, 8),
        (48, 8),
    ];
    let tn = super::sm89_half_tn_source::HALF_TN_DRIVER_ABI
        .map(|(offset, size)| (offset as usize, size as usize));
    let expected: &[(usize, usize)] =
        if spec.schedule == super::sm89_half_source::Sm89HalfSchedule::Relay {
            &RELAY
        } else {
            match spec.op {
                ResolvedGemmOp::Nn => &NN,
                ResolvedGemmOp::Tn => &tn,
                ResolvedGemmOp::Nt => &NT,
            }
        };
    if abi.parameter_count() != expected.len()
        || !abi
            .parameters()
            .iter()
            .zip(expected.iter().copied())
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{} has the wrong live Driver parameter ABI",
            spec.symbol
        ));
    }
    Ok(())
}

fn census_sm89_half_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Result<Tf32DriverAbi, String>>, String> {
    if kind != ModuleKind::TriadSm89Half {
        return Ok(BTreeMap::new());
    }
    if sm80_ptx_target(arch).is_none() {
        return Err("TriadSm89Half Driver ABI census requires an sm_80-tier target".into());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for spec in super::sm89_half_source::runtime_kernel_specs() {
        let abi = (|| {
            let function = unsafe {
                cudarc::driver::result::module::get_function(
                    module.raw(),
                    CString::new(spec.symbol).unwrap(),
                )
            }
            .map_err(|error| {
                format!(
                    "load TriadSm89Half/{} for Driver ABI: {error:?}",
                    spec.symbol
                )
            })?;
            let count = if spec.schedule == super::sm89_half_source::Sm89HalfSchedule::Relay {
                9
            } else {
                match spec.op {
                    ResolvedGemmOp::Nn => 5,
                    ResolvedGemmOp::Tn => {
                        super::sm89_half_tn_source::HALF_TN_TERMINAL_ARGUMENT as usize
                    }
                    ResolvedGemmOp::Nt => 7,
                }
            };
            let abi =
                query_driver_parameter_abi(spec.symbol, count, |index, offset, size| unsafe {
                    get(function, index, offset, size)
                })?;
            validate_sm89_half_driver_abi(&spec, &abi)?;
            Ok(abi)
        })();
        if census.insert(spec.symbol, abi).is_some() {
            return Err(format!(
                "duplicate TriadSm89Half ABI symbol {}",
                spec.symbol
            ));
        }
    }
    module.unload()?;
    Ok(census)
}

fn validate_sm89_exact_f32_driver_abi(
    spec: &super::sm89_exact_f32_source::Sm89ExactF32KernelSpec,
    abi: &Tf32DriverAbi,
) -> Result<(), String> {
    use super::sm89_exact_f32_source::Sm89ExactF32KernelKind;

    const FUSED: [(usize, usize); 4] = [(0, 8), (8, 8), (16, 8), (24, 32)];
    const RAW: [(usize, usize); 7] = [(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)];
    let expected: &[(usize, usize)] = match spec.kind {
        Sm89ExactF32KernelKind::DualChunkFusedFinalize => &FUSED,
        Sm89ExactF32KernelKind::DirectSplitMRaw => &RAW,
    };
    let extent = expected
        .last()
        .map(|(offset, size)| offset + size)
        .unwrap_or_default();
    if abi.parameter_count() != spec.abi_parameter_count as usize
        || extent != spec.abi_parameter_bytes as usize
        || !abi
            .parameters()
            .iter()
            .zip(expected.iter().copied())
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{} has the wrong live Driver parameter ABI",
            spec.symbol
        ));
    }
    Ok(())
}

fn census_sm89_exact_f32_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Result<Tf32DriverAbi, String>>, String> {
    if kind != ModuleKind::TriadSm89ExactF32 {
        return Ok(BTreeMap::new());
    }
    if sm80_ptx_target(arch).is_none() {
        return Err("TriadSm89ExactF32 Driver ABI census requires an sm_80-tier target".into());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for spec in &super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS {
        let abi = (|| {
            let function = unsafe {
                cudarc::driver::result::module::get_function(
                    module.raw(),
                    CString::new(spec.symbol).unwrap(),
                )
            }
            .map_err(|error| {
                format!(
                    "load TriadSm89ExactF32/{} for Driver ABI: {error:?}",
                    spec.symbol
                )
            })?;
            let count = spec.abi_parameter_count as usize;
            let abi =
                query_driver_parameter_abi(spec.symbol, count, |index, offset, size| unsafe {
                    get(function, index, offset, size)
                })?;
            validate_sm89_exact_f32_driver_abi(spec, &abi)?;
            Ok(abi)
        })();
        if census.insert(spec.symbol, abi).is_some() {
            return Err(format!(
                "duplicate TriadSm89ExactF32 ABI symbol {}",
                spec.symbol
            ));
        }
    }
    module.unload()?;
    Ok(census)
}

fn validate_sm89_exact_f32_d128_driver_abi(
    spec: &super::sm89_exact_f32_d128_source::Sm89ExactF32D128KernelSpec,
    abi: &Tf32DriverAbi,
) -> Result<(), String> {
    let expected = super::sm89_exact_f32_d128_source::DIRECT_FOLD_DRIVER_ABI;
    let extent = expected
        .last()
        .map(|(offset, size)| offset + size)
        .unwrap_or_default();
    if abi.parameter_count() != spec.abi_parameter_count as usize
        || extent != spec.abi_parameter_bytes
        || !abi
            .parameters()
            .iter()
            .zip(expected)
            .all(|(actual, (offset, size))| {
                (actual.offset(), actual.size()) == (offset as usize, size as usize)
            })
    {
        return Err(format!(
            "{} has the wrong live Driver parameter ABI",
            spec.symbol
        ));
    }
    Ok(())
}

fn census_sm89_exact_f32_d128_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Result<Tf32DriverAbi, String>>, String> {
    if kind != ModuleKind::TriadSm89ExactF32D128 {
        return Ok(BTreeMap::new());
    }
    if sm80_ptx_target(arch).is_none() {
        return Err("TriadSm89ExactF32D128 Driver ABI census requires an sm_80-tier target".into());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for spec in &super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS {
        let abi = (|| {
            let function = unsafe {
                cudarc::driver::result::module::get_function(
                    module.raw(),
                    CString::new(spec.symbol).unwrap(),
                )
            }
            .map_err(|error| {
                format!(
                    "load TriadSm89ExactF32D128/{} for Driver ABI: {error:?}",
                    spec.symbol
                )
            })?;
            let abi = query_driver_parameter_abi(
                spec.symbol,
                spec.abi_parameter_count as usize,
                |index, offset, size| unsafe { get(function, index, offset, size) },
            )?;
            validate_sm89_exact_f32_d128_driver_abi(spec, &abi)?;
            Ok(abi)
        })();
        if census.insert(spec.symbol, abi).is_some() {
            return Err(format!(
                "duplicate TriadSm89ExactF32D128 ABI symbol {}",
                spec.symbol
            ));
        }
    }
    module.unload()?;
    Ok(census)
}

fn validate_sm89_tf32_joint_driver_abi(
    spec: &super::sm89_tf32_joint_source::Sm89Tf32JointKernelSpec,
    abi: &Tf32DriverAbi,
) -> Result<(), String> {
    let extent = spec
        .abi_parameters
        .last()
        .map(|parameter| parameter.offset + parameter.size)
        .unwrap_or_default();
    if abi.parameter_count() != spec.abi_parameters.len()
        || extent != spec.abi_parameter_bytes
        || spec.terminal_argument as usize != spec.abi_parameters.len()
        || !abi
            .parameters()
            .iter()
            .zip(spec.abi_parameters.iter())
            .all(|(actual, expected)| {
                actual.offset() == expected.offset as usize
                    && actual.size() == expected.size as usize
            })
    {
        return Err(format!(
            "{} has the wrong live Driver parameter ABI",
            spec.symbol
        ));
    }
    Ok(())
}

fn census_sm89_tf32_joint_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Result<Tf32DriverAbi, String>>, String> {
    if kind != ModuleKind::TriadSm89Tf32Joint {
        return Ok(BTreeMap::new());
    }
    if sm80_ptx_target(arch).is_none() {
        return Err("TriadSm89Tf32Joint Driver ABI census requires an sm_80-tier target".into());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for spec in &super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS {
        let abi = (|| {
            let function = unsafe {
                cudarc::driver::result::module::get_function(
                    module.raw(),
                    CString::new(spec.symbol).unwrap(),
                )
            }
            .map_err(|error| {
                format!(
                    "load TriadSm89Tf32Joint/{} for Driver ABI: {error:?}",
                    spec.symbol
                )
            })?;
            let abi = query_driver_parameter_abi(
                spec.symbol,
                spec.abi_parameters.len(),
                |index, offset, size| unsafe { get(function, index, offset, size) },
            )?;
            validate_sm89_tf32_joint_driver_abi(spec, &abi)?;
            Ok(abi)
        })();
        if census.insert(spec.symbol, abi).is_some() {
            return Err(format!(
                "duplicate TriadSm89Tf32Joint ABI symbol {}",
                spec.symbol
            ));
        }
    }
    module.unload()?;
    Ok(census)
}

fn census_fixed_sm89_half_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    if kind != ModuleKind::Fixed || !fixed_sm89_half_composed(arch) {
        return Ok(BTreeMap::new());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for symbol in FIXED_SM89_HALF_SYMBOLS {
        let function = unsafe {
            cudarc::driver::result::module::get_function(
                module.raw(),
                CString::new(symbol).unwrap(),
            )
        }
        .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
        let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| unsafe {
            get(function, index, offset, size)
        })?;
        validate_fixed_sm89_half_driver_abi(symbol, &abi)?;
        census.insert(symbol, abi);
    }
    module.unload()?;
    Ok(census)
}

fn validate_fixed_sm89_cell_driver_abi(symbol: &str, abi: &Tf32DriverAbi) -> Result<(), String> {
    const EXPECTED: [(usize, usize); 5] = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    if super::super::gemm_bi_inference::FIXED_SM89_HALF_PARAMS_SIZE != 32 {
        return Err("Fixed SM89 cell host parameter ABI drifted".into());
    }
    if abi.parameter_count() != EXPECTED.len()
        || !abi
            .parameters()
            .iter()
            .zip(EXPECTED)
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{symbol} has the wrong live five-argument/64-byte Driver ABI"
        ));
    }
    Ok(())
}

fn census_fixed_sm89_cells_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    if kind != ModuleKind::InferenceSm89Cells || !fixed_portable_overlay_composed(arch) {
        return Ok(BTreeMap::new());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for symbol in super::super::gemm_bi_inference::sm89_cells::SM89_CELL_SYMBOLS {
        let function = unsafe {
            cudarc::driver::result::module::get_function(
                module.raw(),
                CString::new(symbol).unwrap(),
            )
        }
        .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
        let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| unsafe {
            get(function, index, offset, size)
        })?;
        validate_fixed_sm89_cell_driver_abi(symbol, &abi)?;
        census.insert(symbol, abi);
    }
    module.unload()?;
    Ok(census)
}

/// Binds every Ada inference cell the module compiled; a cell that misses
/// its resource or ABI contract is left out with its reason, the others
/// stay bound.
pub(crate) fn load_fixed_sm89_cells(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (
    HashMap<&'static str, CudaFunction>,
    Vec<(&'static str, String)>,
) {
    use super::super::gemm_bi_inference::sm89_cells::SM89_CELL_SPECS;

    let mut functions = HashMap::new();
    let mut rejections = Vec::new();
    let composed = module.artifact_identity.module_kind == ModuleKind::InferenceSm89Cells
        && fixed_portable_overlay_composed(module.compiler_identity.target.as_str())
        && ctx.compute_capability().is_ok_and(|cc| cc.0 >= 8);
    if !composed {
        return (functions, rejections);
    }
    for spec in SM89_CELL_SPECS.iter() {
        let symbol = spec.symbol;
        let loaded = (|| -> Result<CudaFunction, String> {
            let shared_cap = ctx
                .attribute(
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                )
                .map_err(|error| format!("query {symbol} opt-in shared capacity: {error:?}"))?;
            if shared_cap < spec.dynamic_shared_bytes as i32 {
                return Err(format!(
                    "{symbol} requires {} shared bytes, device permits {shared_cap}",
                    spec.dynamic_shared_bytes
                ));
            }
            let abi = module
                .fixed_sm89_cells_driver_abi
                .as_ref()
                .map_err(Clone::clone)?
                .get(symbol)
                .ok_or_else(|| format!("{symbol} has no Driver ABI census entry"))?;
            validate_fixed_sm89_cell_driver_abi(symbol, abi)?;
            let function = load_function(&module.module, ModuleKind::InferenceSm89Cells, symbol)?;
            set_dynamic_shared(&function, symbol, spec.dynamic_shared_bytes as i32)?;
            let query_error = |label, error| format!("query {symbol} {label}: {error:?}");
            let local_bytes = u32::try_from(
                function
                    .local_size_bytes()
                    .map_err(|e| query_error("local bytes", e))?,
            )
            .map_err(|_| format!("{symbol} returned negative local memory"))?;
            let registers = u32::try_from(
                function
                    .num_regs()
                    .map_err(|e| query_error("registers", e))?,
            )
            .map_err(|_| format!("{symbol} returned negative registers"))?;
            let static_shared_bytes = u32::try_from(
                function
                    .shared_size_bytes()
                    .map_err(|e| query_error("static shared bytes", e))?,
            )
            .map_err(|_| format!("{symbol} returned negative static shared memory"))?;
            if static_shared_bytes != 0 {
                return Err(format!(
                    "{symbol} uses {static_shared_bytes} static shared bytes, expected zero"
                ));
            }
            let max_threads = function
                .max_threads_per_block()
                .map_err(|e| query_error("max threads", e))?;
            tf32_symbol_admission(
                symbol,
                local_bytes,
                0,
                registers,
                spec.register_cap,
                max_threads,
                spec.threads as i32,
            )?;
            let active_blocks = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    spec.threads,
                    spec.dynamic_shared_bytes as usize,
                    None,
                )
                .map_err(|e| query_error("occupancy", e))?;
            if active_blocks < spec.occupancy_gate {
                return Err(format!(
                    "{symbol} occupancy {active_blocks} misses its {}-CTA gate",
                    spec.occupancy_gate
                ));
            }
            Ok(function)
        })();
        match loaded {
            Ok(function) => {
                functions.insert(symbol, function);
            }
            Err(reason) => rejections.push((symbol, reason)),
        }
    }
    (functions, rejections)
}

fn validate_fixed_sm89_half_swizzle_driver_abi(
    symbol: &str,
    abi: &Tf32DriverAbi,
) -> Result<(), String> {
    const EXPECTED: [(usize, usize); 5] = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    if super::super::gemm_bi_inference::FIXED_SM89_HALF_SWIZZLE_PARAMS_SIZE != 32 {
        return Err("Fixed SM89 half swizzle host parameter ABI drifted".into());
    }
    if abi.parameter_count() != EXPECTED.len()
        || !abi
            .parameters()
            .iter()
            .zip(EXPECTED)
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{symbol} has the wrong live five-argument/64-byte Driver ABI"
        ));
    }
    Ok(())
}

fn census_fixed_sm89_half_swizzle_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    if kind != ModuleKind::Fixed || !fixed_sm89_half_composed(arch) {
        return Ok(BTreeMap::new());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for symbol in FIXED_SM89_HALF_SWIZZLE_SYMBOLS {
        let function = unsafe {
            cudarc::driver::result::module::get_function(
                module.raw(),
                CString::new(symbol).unwrap(),
            )
        }
        .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
        let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| unsafe {
            get(function, index, offset, size)
        })?;
        validate_fixed_sm89_half_swizzle_driver_abi(symbol, &abi)?;
        census.insert(symbol, abi);
    }
    module.unload()?;
    Ok(census)
}

fn validate_fixed_sm89_half_s3_driver_abi(symbol: &str, abi: &Tf32DriverAbi) -> Result<(), String> {
    const EXPECTED: [(usize, usize); 5] = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    if super::super::gemm_bi_inference::FIXED_SM89_HALF_PARAMS_SIZE != 32 {
        return Err("Fixed SM89 half s3 host parameter ABI drifted".into());
    }
    if abi.parameter_count() != EXPECTED.len()
        || !abi
            .parameters()
            .iter()
            .zip(EXPECTED)
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{symbol} has the wrong live five-argument/64-byte Driver ABI"
        ));
    }
    Ok(())
}

fn census_fixed_sm89_half_s3_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    if kind != ModuleKind::Fixed || !fixed_sm89_half_composed(arch) {
        return Ok(BTreeMap::new());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for symbol in FIXED_SM89_HALF_S3_SYMBOLS {
        let function = unsafe {
            cudarc::driver::result::module::get_function(
                module.raw(),
                CString::new(symbol).unwrap(),
            )
        }
        .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
        let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| unsafe {
            get(function, index, offset, size)
        })?;
        validate_fixed_sm89_half_s3_driver_abi(symbol, &abi)?;
        census.insert(symbol, abi);
    }
    module.unload()?;
    Ok(census)
}

fn validate_fixed_sm89_rna_wide_driver_abi(abi: &Tf32DriverAbi) -> Result<(), String> {
    const EXPECTED: [(usize, usize); 5] = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    if super::super::gemm_bi_inference::FIXED_TF32_WIDE_PARAMS_SIZE != 32 {
        return Err("Fixed SM89 RNA-wide host parameter ABI drifted".into());
    }
    if abi.parameter_count() != EXPECTED.len()
        || !abi
            .parameters()
            .iter()
            .zip(EXPECTED)
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{FIXED_SM89_RNA_WIDE_SYMBOL} has the wrong live five-argument/64-byte Driver ABI"
        ));
    }
    Ok(())
}

fn census_fixed_sm89_rna_wide_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<Tf32DriverAbi, String> {
    if kind != ModuleKind::Fixed || !fixed_sm89_rna_wide_composed(arch) {
        return Err("Fixed SM89 RNA-wide is not composed for this module/target".into());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let function = unsafe {
        cudarc::driver::result::module::get_function(
            module.raw(),
            CString::new(FIXED_SM89_RNA_WIDE_SYMBOL).unwrap(),
        )
    }
    .map_err(|error| {
        format!("load Fixed/{FIXED_SM89_RNA_WIDE_SYMBOL} for Driver ABI: {error:?}")
    })?;
    let abi = query_driver_parameter_abi(
        FIXED_SM89_RNA_WIDE_SYMBOL,
        5,
        |index, offset, size| unsafe { get(function, index, offset, size) },
    )?;
    validate_fixed_sm89_rna_wide_driver_abi(&abi)?;
    module.unload()?;
    Ok(abi)
}

fn census_fixed_sm89_finalist_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> FixedSm89FinalistDriverAbi {
    if kind != ModuleKind::Fixed || !fixed_portable_overlay_composed(arch) {
        return FixedSm89FinalistDriverAbi::rejected(
            "Fixed SM89 finalists are not composed for this module/target".into(),
        );
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = match DriverModule::load(ctx, ptx) {
        Ok(module) => module,
        Err(error) => return FixedSm89FinalistDriverAbi::rejected(error),
    };
    let get: GetParamInfo = match driver_proc_address("cuFuncGetParamInfo", 12_040) {
        Ok(get) => unsafe { std::mem::transmute::<*mut std::ffi::c_void, GetParamInfo>(get) },
        Err(error) => {
            let _ = module.unload();
            return FixedSm89FinalistDriverAbi::rejected(error);
        }
    };
    let parsed = match parse_ptx(ptx) {
        Ok(parsed) => parsed,
        Err(error) => {
            let _ = module.unload();
            return FixedSm89FinalistDriverAbi::rejected(error);
        }
    };
    let query = |symbol: &'static str| -> Result<Tf32DriverAbi, String> {
        validate_fixed_sm89_finalist_entry_ptx(&parsed, symbol)?;
        let function = unsafe {
            cudarc::driver::result::module::get_function(
                module.raw(),
                CString::new(symbol).unwrap(),
            )
        }
        .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
        let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| unsafe {
            get(function, index, offset, size)
        })?;
        validate_fixed_sm89_finalist_driver_abi(symbol, &abi)?;
        Ok(abi)
    };
    let census = FixedSm89FinalistDriverAbi {
        rna_n96: query(FIXED_SM89_RNA_N96_SYMBOL),
        half_m64n64_s3: query(FIXED_SM89_HALF_M64N64_S3_SYMBOL),
        half_m128n64_s2: query(FIXED_SM89_HALF_M128N64_S2_SYMBOL),
    };
    if let Err(error) = module.unload() {
        return FixedSm89FinalistDriverAbi::rejected(format!(
            "unload Fixed SM89 finalist ABI census: {error}"
        ));
    }
    census
}

fn validate_fixed_sm89_finalist_driver_abi(
    symbol: &str,
    abi: &Tf32DriverAbi,
) -> Result<(), String> {
    const EXPECTED: [(usize, usize); 5] = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    let host_size = if symbol == FIXED_SM89_RNA_N96_SYMBOL {
        super::super::gemm_bi_inference::FIXED_TF32_WIDE_PARAMS_SIZE
    } else {
        super::super::gemm_bi_inference::FIXED_SM89_HALF_PARAMS_SIZE
    };
    if host_size != 32 {
        return Err(format!(
            "{symbol} host parameter ABI drifted to {host_size} bytes"
        ));
    }
    if abi.parameter_count() != EXPECTED.len()
        || !abi
            .parameters()
            .iter()
            .zip(EXPECTED)
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{symbol} has the wrong live five-argument/64-byte Driver ABI"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedSm89FinalistResources {
    local_bytes: u32,
    registers: u32,
    static_shared_bytes: u32,
    max_threads: i32,
    active_blocks: u32,
}

fn validate_fixed_sm89_finalist_resources(
    symbol: &str,
    resources: FixedSm89FinalistResources,
) -> Result<(), String> {
    let (register_cap, threads, expected_blocks, shared_bytes) = match symbol {
        FIXED_SM89_RNA_N96_SYMBOL => (
            FIXED_SM89_RNA_N96_REGISTER_CAP,
            FIXED_SM89_RNA_N96_THREADS,
            1,
            FIXED_SM89_RNA_N96_SHARED_BYTES,
        ),
        FIXED_SM89_HALF_M64N64_S3_SYMBOL => (
            FIXED_SM89_HALF_M64N64_S3_REGISTER_CAP,
            FIXED_SM89_HALF_N64_THREADS,
            2,
            FIXED_SM89_HALF_N64_SHARED_BYTES,
        ),
        FIXED_SM89_HALF_M128N64_S2_SYMBOL => (
            FIXED_SM89_HALF_M128N64_S2_REGISTER_CAP,
            FIXED_SM89_HALF_N64_THREADS,
            2,
            FIXED_SM89_HALF_N64_SHARED_BYTES,
        ),
        _ => return Err(format!("unknown Fixed SM89 finalist {symbol}")),
    };
    tf32_symbol_admission(
        symbol,
        resources.local_bytes,
        0,
        resources.registers,
        register_cap,
        resources.max_threads,
        threads as i32,
    )?;
    if resources.static_shared_bytes != 0 {
        return Err(format!(
            "{symbol} uses {} static shared bytes, expected zero",
            resources.static_shared_bytes
        ));
    }
    if resources.active_blocks != expected_blocks {
        return Err(format!(
            "{symbol} has {} active CTAs at {shared_bytes} dynamic shared bytes, expected {expected_blocks}",
            resources.active_blocks
        ));
    }
    Ok(())
}

fn load_fixed_sm89_finalist(
    ctx: &CudaContext,
    module: &CompiledModule,
    symbol: &'static str,
) -> (Option<CudaFunction>, Option<String>) {
    let admitted = (|| -> Result<CudaFunction, String> {
        if module.artifact_identity.module_kind != ModuleKind::Fixed
            || !fixed_portable_overlay_composed(module.compiler_identity.target.as_str())
            || ctx
                .compute_capability()
                .map_err(|error| format!("query {symbol} CC: {error:?}"))?
                .0
                < 8
        {
            return Err(format!(
                "Fixed SM89 finalist {symbol} is composed only with the portable overlay"
            ));
        }
        let (abi, shared_bytes, threads) = match symbol {
            FIXED_SM89_RNA_N96_SYMBOL => (
                &module.fixed_sm89_finalist_driver_abi.rna_n96,
                FIXED_SM89_RNA_N96_SHARED_BYTES,
                FIXED_SM89_RNA_N96_THREADS,
            ),
            FIXED_SM89_HALF_M64N64_S3_SYMBOL => (
                &module.fixed_sm89_finalist_driver_abi.half_m64n64_s3,
                FIXED_SM89_HALF_N64_SHARED_BYTES,
                FIXED_SM89_HALF_N64_THREADS,
            ),
            FIXED_SM89_HALF_M128N64_S2_SYMBOL => (
                &module.fixed_sm89_finalist_driver_abi.half_m128n64_s2,
                FIXED_SM89_HALF_N64_SHARED_BYTES,
                FIXED_SM89_HALF_N64_THREADS,
            ),
            _ => return Err(format!("unknown Fixed SM89 finalist {symbol}")),
        };
        let shared_cap = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(|error| format!("query {symbol} opt-in shared capacity: {error:?}"))?;
        if shared_cap < shared_bytes as i32 {
            return Err(format!(
                "{symbol} requires {shared_bytes} shared bytes, device permits {shared_cap}"
            ));
        }
        validate_fixed_sm89_finalist_driver_abi(symbol, abi.as_ref().map_err(Clone::clone)?)?;
        let function = load_function(&module.module, ModuleKind::Fixed, symbol)?;
        set_dynamic_shared(&function, symbol, shared_bytes as i32)?;
        let query_error = |label, error| format!("query {symbol} {label}: {error:?}");
        let resources = FixedSm89FinalistResources {
            local_bytes: u32::try_from(
                function
                    .local_size_bytes()
                    .map_err(|e| query_error("local bytes", e))?,
            )
            .map_err(|_| format!("{symbol} returned negative local memory"))?,
            registers: u32::try_from(
                function
                    .num_regs()
                    .map_err(|e| query_error("registers", e))?,
            )
            .map_err(|_| format!("{symbol} returned negative registers"))?,
            static_shared_bytes: u32::try_from(
                function
                    .shared_size_bytes()
                    .map_err(|e| query_error("static shared bytes", e))?,
            )
            .map_err(|_| format!("{symbol} returned negative static shared memory"))?,
            max_threads: function
                .max_threads_per_block()
                .map_err(|e| query_error("max threads", e))?,
            active_blocks: function
                .occupancy_max_active_blocks_per_multiprocessor(
                    threads,
                    shared_bytes as usize,
                    None,
                )
                .map_err(|e| query_error("occupancy", e))?,
        };
        validate_fixed_sm89_finalist_resources(symbol, resources)?;
        Ok(function)
    })();
    match admitted {
        Ok(function) => (Some(function), None),
        Err(reason) => (None, Some(reason)),
    }
}

pub(crate) fn load_fixed_sm89_rna_n96(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<CudaFunction>, Option<String>) {
    load_fixed_sm89_finalist(ctx, module, FIXED_SM89_RNA_N96_SYMBOL)
}

pub(crate) fn load_fixed_sm89_half_m64n64_s3(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<CudaFunction>, Option<String>) {
    load_fixed_sm89_finalist(ctx, module, FIXED_SM89_HALF_M64N64_S3_SYMBOL)
}

pub(crate) fn load_fixed_sm89_half_m128n64_s2(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<CudaFunction>, Option<String>) {
    load_fixed_sm89_finalist(ctx, module, FIXED_SM89_HALF_M128N64_S2_SYMBOL)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedSm89RnaWideResources {
    local_bytes: u32,
    registers: u32,
    static_shared_bytes: u32,
    max_threads: i32,
    active_blocks: u32,
}

fn validate_fixed_sm89_rna_wide_resources(
    resources: FixedSm89RnaWideResources,
) -> Result<(), String> {
    tf32_symbol_admission(
        FIXED_SM89_RNA_WIDE_SYMBOL,
        resources.local_bytes,
        0,
        resources.registers,
        FIXED_SM89_RNA_WIDE_REGISTER_CAP,
        resources.max_threads,
        FIXED_SM89_RNA_WIDE_THREADS as i32,
    )?;
    if resources.static_shared_bytes != 0 {
        return Err(format!(
            "{FIXED_SM89_RNA_WIDE_SYMBOL} uses {} static shared bytes, expected zero",
            resources.static_shared_bytes
        ));
    }
    if resources.active_blocks < 1 {
        return Err(format!(
            "{FIXED_SM89_RNA_WIDE_SYMBOL} has no resident CTA at {} dynamic shared bytes",
            FIXED_SM89_RNA_WIDE_SHARED_BYTES
        ));
    }
    Ok(())
}

pub(crate) fn load_fixed_sm89_rna_wide(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<CudaFunction>, Option<String>) {
    let admitted = (|| -> Result<CudaFunction, String> {
        if module.artifact_identity.module_kind != ModuleKind::Fixed
            || !fixed_sm89_rna_wide_composed(module.compiler_identity.target.as_str())
            || ctx
                .compute_capability()
                .map_err(|error| format!("query Fixed RNA-wide CC: {error:?}"))?
                .0
                < 8
        {
            return Err("Fixed SM89 RNA-wide is composed only with the portable overlay".into());
        }
        let shared_cap = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        ).map_err(|error| format!("query Fixed RNA-wide opt-in shared capacity: {error:?}"))?;
        if shared_cap < FIXED_SM89_RNA_WIDE_SHARED_BYTES as i32 {
            return Err(format!(
                "Fixed SM89 RNA-wide requires {} shared bytes, device permits {shared_cap}",
                FIXED_SM89_RNA_WIDE_SHARED_BYTES
            ));
        }
        validate_fixed_sm89_rna_wide_driver_abi(
            module
                .fixed_sm89_rna_wide_driver_abi
                .as_ref()
                .map_err(Clone::clone)?,
        )?;
        let function = load_function(
            &module.module,
            ModuleKind::Fixed,
            FIXED_SM89_RNA_WIDE_SYMBOL,
        )?;
        set_dynamic_shared(
            &function,
            FIXED_SM89_RNA_WIDE_SYMBOL,
            FIXED_SM89_RNA_WIDE_SHARED_BYTES as i32,
        )?;
        let local_bytes = u32::try_from(
            function
                .local_size_bytes()
                .map_err(|error| format!("query RNA-wide local bytes: {error:?}"))?,
        )
        .map_err(|_| "Fixed SM89 RNA-wide returned negative local memory")?;
        let registers = u32::try_from(
            function
                .num_regs()
                .map_err(|error| format!("query RNA-wide registers: {error:?}"))?,
        )
        .map_err(|_| "Fixed SM89 RNA-wide returned negative registers")?;
        let static_shared_bytes = u32::try_from(
            function
                .shared_size_bytes()
                .map_err(|error| format!("query RNA-wide static shared bytes: {error:?}"))?,
        )
        .map_err(|_| "Fixed SM89 RNA-wide returned negative static shared memory")?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|error| format!("query RNA-wide max threads: {error:?}"))?;
        let active_blocks = function
            .occupancy_max_active_blocks_per_multiprocessor(
                FIXED_SM89_RNA_WIDE_THREADS,
                FIXED_SM89_RNA_WIDE_SHARED_BYTES as usize,
                None,
            )
            .map_err(|error| format!("query RNA-wide occupancy: {error:?}"))?;
        validate_fixed_sm89_rna_wide_resources(FixedSm89RnaWideResources {
            local_bytes,
            registers,
            static_shared_bytes,
            max_threads,
            active_blocks,
        })?;
        Ok(function)
    })();
    match admitted {
        Ok(function) => (Some(function), None),
        Err(reason) => (None, Some(reason)),
    }
}

/// Admit both homogeneous-half exports together. Unknown targets, ABI drift,
/// resource exclusions and Driver-query failures keep the optional holder
/// absent; they do not disable the mandatory portable Fixed kernels.
pub(crate) fn load_fixed_sm89_half_pipeline(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<HalfKernel>, Option<String>) {
    let admitted = (|| -> Result<HalfKernel, String> {
        if module.artifact_identity.module_kind != ModuleKind::Fixed
            || !fixed_sm89_half_composed(module.compiler_identity.target.as_str())
            || ctx
                .compute_capability()
                .map_err(|error| format!("query Fixed half CC: {error:?}"))?
                .0
                < 8
        {
            return Err(
                "Fixed SM89 half pipeline is composed only with the portable overlay".into(),
            );
        }
        let shared_cap = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        ).map_err(|error| format!("query Fixed half opt-in shared capacity: {error:?}"))?;
        if shared_cap < FIXED_SM89_HALF_SHARED_BYTES as i32 {
            return Err(format!(
                "Fixed SM89 half requires {} shared bytes, device permits {shared_cap}",
                FIXED_SM89_HALF_SHARED_BYTES
            ));
        }
        let abi = module
            .fixed_sm89_half_driver_abi
            .as_ref()
            .map_err(Clone::clone)?;
        let mut functions = Vec::with_capacity(2);
        for symbol in FIXED_SM89_HALF_SYMBOLS {
            validate_fixed_sm89_half_driver_abi(
                symbol,
                abi.get(symbol)
                    .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?,
            )?;
            let function = load_function(&module.module, ModuleKind::Fixed, symbol)?;
            set_dynamic_shared(&function, symbol, FIXED_SM89_HALF_SHARED_BYTES as i32)?;
            let local = function
                .local_size_bytes()
                .map_err(|error| format!("query {symbol} local bytes: {error:?}"))?;
            let registers = function
                .num_regs()
                .map_err(|error| format!("query {symbol} registers: {error:?}"))?;
            let threads = function
                .max_threads_per_block()
                .map_err(|error| format!("query {symbol} threads: {error:?}"))?;
            let local =
                u32::try_from(local).map_err(|_| format!("{symbol} negative local memory"))?;
            let registers =
                u32::try_from(registers).map_err(|_| format!("{symbol} negative registers"))?;
            tf32_symbol_admission(
                symbol,
                local,
                0,
                registers,
                FIXED_SM89_HALF_REGISTER_CAP,
                threads,
                FIXED_SM89_HALF_THREADS as i32,
            )?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    FIXED_SM89_HALF_THREADS,
                    FIXED_SM89_HALF_SHARED_BYTES as usize,
                    None,
                )
                .map_err(|error| format!("query {symbol} occupancy: {error:?}"))?;
            if occupancy == 0 {
                return Err(format!(
                    "{symbol} has no resident CTA at its required shared-memory size"
                ));
            }
            functions.push(function);
        }
        let mut functions = functions.into_iter();
        Ok(HalfKernel {
            bf16: functions.next().unwrap(),
            f16: functions.next().unwrap(),
        })
    })();
    match admitted {
        Ok(functions) => (Some(functions), None),
        Err(reason) => (None, Some(reason)),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedSm89HalfSwizzleResources {
    local_bytes: u32,
    registers: u32,
    static_shared_bytes: u32,
    max_threads: i32,
    active_blocks: u32,
}

fn validate_fixed_sm89_half_swizzle_resources(
    symbol: &str,
    resources: FixedSm89HalfSwizzleResources,
) -> Result<(), String> {
    tf32_symbol_admission(
        symbol,
        resources.local_bytes,
        0,
        resources.registers,
        FIXED_SM89_HALF_SWIZZLE_REGISTER_CAP,
        resources.max_threads,
        FIXED_SM89_HALF_SWIZZLE_THREADS as i32,
    )?;
    if resources.static_shared_bytes != 0 {
        return Err(format!(
            "{symbol} uses {} static shared bytes, expected zero",
            resources.static_shared_bytes
        ));
    }
    if resources.active_blocks < 1 {
        return Err(format!(
            "{symbol} has no resident CTA at {} dynamic shared bytes",
            FIXED_SM89_HALF_SWIZZLE_SHARED_BYTES
        ));
    }
    Ok(())
}

fn validate_fixed_sm89_half_swizzle_shared_capacity(shared_cap: i32) -> Result<(), String> {
    if shared_cap < FIXED_SM89_HALF_SWIZZLE_SHARED_BYTES as i32 {
        return Err(format!(
            "Fixed SM89 half swizzle requires {} shared bytes, device permits {shared_cap}",
            FIXED_SM89_HALF_SWIZZLE_SHARED_BYTES
        ));
    }
    Ok(())
}

/// Admit the swizzled homogeneous-half pair independently from the incumbent
/// pipeline. A failure here records only the swizzle rejection reason.
pub(crate) fn load_fixed_sm89_half_swizzle(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<HalfKernel>, Option<String>) {
    let admitted = (|| -> Result<HalfKernel, String> {
        if module.artifact_identity.module_kind != ModuleKind::Fixed
            || !fixed_sm89_half_composed(module.compiler_identity.target.as_str())
            || ctx
                .compute_capability()
                .map_err(|error| format!("query Fixed half swizzle CC: {error:?}"))?
                .0
                < 8
        {
            return Err(
                "Fixed SM89 half swizzle is composed only with the portable overlay".into(),
            );
        }
        let shared_cap = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        ).map_err(|error| format!("query Fixed half swizzle opt-in shared capacity: {error:?}"))?;
        validate_fixed_sm89_half_swizzle_shared_capacity(shared_cap)?;
        let abi = module
            .fixed_sm89_half_swizzle_driver_abi
            .as_ref()
            .map_err(Clone::clone)?;
        let mut functions = Vec::with_capacity(2);
        for symbol in FIXED_SM89_HALF_SWIZZLE_SYMBOLS {
            validate_fixed_sm89_half_swizzle_driver_abi(
                symbol,
                abi.get(symbol)
                    .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?,
            )?;
            let function = load_function(&module.module, ModuleKind::Fixed, symbol)?;
            set_dynamic_shared(
                &function,
                symbol,
                FIXED_SM89_HALF_SWIZZLE_SHARED_BYTES as i32,
            )?;
            let local_bytes = u32::try_from(
                function
                    .local_size_bytes()
                    .map_err(|error| format!("query {symbol} local bytes: {error:?}"))?,
            )
            .map_err(|_| format!("{symbol} returned negative local memory"))?;
            let registers = u32::try_from(
                function
                    .num_regs()
                    .map_err(|error| format!("query {symbol} registers: {error:?}"))?,
            )
            .map_err(|_| format!("{symbol} returned negative registers"))?;
            let static_shared_bytes = u32::try_from(
                function
                    .shared_size_bytes()
                    .map_err(|error| format!("query {symbol} static shared bytes: {error:?}"))?,
            )
            .map_err(|_| format!("{symbol} returned negative static shared memory"))?;
            let max_threads = function
                .max_threads_per_block()
                .map_err(|error| format!("query {symbol} max threads: {error:?}"))?;
            let active_blocks = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    FIXED_SM89_HALF_SWIZZLE_THREADS,
                    FIXED_SM89_HALF_SWIZZLE_SHARED_BYTES as usize,
                    None,
                )
                .map_err(|error| format!("query {symbol} occupancy: {error:?}"))?;
            validate_fixed_sm89_half_swizzle_resources(
                symbol,
                FixedSm89HalfSwizzleResources {
                    local_bytes,
                    registers,
                    static_shared_bytes,
                    max_threads,
                    active_blocks,
                },
            )?;
            functions.push(function);
        }
        let mut functions = functions.into_iter();
        Ok(HalfKernel {
            bf16: functions.next().unwrap(),
            f16: functions.next().unwrap(),
        })
    })();
    match admitted {
        Ok(functions) => (Some(functions), None),
        Err(reason) => (None, Some(reason)),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedSm89HalfS3Resources {
    local_bytes: u32,
    registers: u32,
    static_shared_bytes: u32,
    max_threads: i32,
    active_blocks: u32,
}

fn validate_fixed_sm89_half_s3_resources(
    symbol: &str,
    resources: FixedSm89HalfS3Resources,
) -> Result<(), String> {
    tf32_symbol_admission(
        symbol,
        resources.local_bytes,
        0,
        resources.registers,
        FIXED_SM89_HALF_S3_REGISTER_CAP,
        resources.max_threads,
        FIXED_SM89_HALF_S3_THREADS as i32,
    )?;
    if resources.static_shared_bytes != 0 {
        return Err(format!(
            "{symbol} uses {} static shared bytes, expected zero",
            resources.static_shared_bytes
        ));
    }
    if resources.active_blocks < 1 {
        return Err(format!(
            "{symbol} has no resident CTA at {} dynamic shared bytes",
            FIXED_SM89_HALF_S3_SHARED_BYTES
        ));
    }
    Ok(())
}

fn validate_fixed_sm89_half_s3_shared_capacity(shared_cap: i32) -> Result<(), String> {
    if shared_cap < FIXED_SM89_HALF_S3_SHARED_BYTES as i32 {
        return Err(format!(
            "Fixed SM89 half s3 requires {} shared bytes, device permits {shared_cap}",
            FIXED_SM89_HALF_S3_SHARED_BYTES
        ));
    }
    Ok(())
}

/// Admit the three-stage homogeneous-half pair independently from both
/// two-stage holders. A failure here records only the S3 rejection reason.
pub(crate) fn load_fixed_sm89_half_s3(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<HalfKernel>, Option<String>) {
    let admitted = (|| -> Result<HalfKernel, String> {
        if module.artifact_identity.module_kind != ModuleKind::Fixed
            || !fixed_sm89_half_composed(module.compiler_identity.target.as_str())
            || ctx
                .compute_capability()
                .map_err(|error| format!("query Fixed half s3 CC: {error:?}"))?
                .0
                < 8
        {
            return Err("Fixed SM89 half s3 is composed only with the portable overlay".into());
        }
        let shared_cap = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        ).map_err(|error| format!("query Fixed half s3 opt-in shared capacity: {error:?}"))?;
        validate_fixed_sm89_half_s3_shared_capacity(shared_cap)?;
        let abi = module
            .fixed_sm89_half_s3_driver_abi
            .as_ref()
            .map_err(Clone::clone)?;
        let mut functions = Vec::with_capacity(2);
        for symbol in FIXED_SM89_HALF_S3_SYMBOLS {
            validate_fixed_sm89_half_s3_driver_abi(
                symbol,
                abi.get(symbol)
                    .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?,
            )?;
            let function = load_function(&module.module, ModuleKind::Fixed, symbol)?;
            set_dynamic_shared(&function, symbol, FIXED_SM89_HALF_S3_SHARED_BYTES as i32)?;
            let local_bytes = u32::try_from(
                function
                    .local_size_bytes()
                    .map_err(|error| format!("query {symbol} local bytes: {error:?}"))?,
            )
            .map_err(|_| format!("{symbol} returned negative local memory"))?;
            let registers = u32::try_from(
                function
                    .num_regs()
                    .map_err(|error| format!("query {symbol} registers: {error:?}"))?,
            )
            .map_err(|_| format!("{symbol} returned negative registers"))?;
            let static_shared_bytes = u32::try_from(
                function
                    .shared_size_bytes()
                    .map_err(|error| format!("query {symbol} static shared bytes: {error:?}"))?,
            )
            .map_err(|_| format!("{symbol} returned negative static shared memory"))?;
            let max_threads = function
                .max_threads_per_block()
                .map_err(|error| format!("query {symbol} max threads: {error:?}"))?;
            let active_blocks = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    FIXED_SM89_HALF_S3_THREADS,
                    FIXED_SM89_HALF_S3_SHARED_BYTES as usize,
                    None,
                )
                .map_err(|error| format!("query {symbol} occupancy: {error:?}"))?;
            validate_fixed_sm89_half_s3_resources(
                symbol,
                FixedSm89HalfS3Resources {
                    local_bytes,
                    registers,
                    static_shared_bytes,
                    max_threads,
                    active_blocks,
                },
            )?;
            functions.push(function);
        }
        let mut functions = functions.into_iter();
        Ok(HalfKernel {
            bf16: functions.next().unwrap(),
            f16: functions.next().unwrap(),
        })
    })();
    match admitted {
        Ok(functions) => (Some(functions), None),
        Err(reason) => (None, Some(reason)),
    }
}

const FIXED_SM89_EXACT_N64_SYMBOL: &str = "nn_sm89_f32_n64_copyplan";
const FIXED_SM89_EXACT_N64_THREADS: u32 = 128;
const FIXED_SM89_EXACT_N64_STATIC_SHARED: i32 = 32_768;

fn fixed_sm89_exact_n64_composed(arch: &str) -> bool {
    fixed_portable_overlay_composed(arch)
}

const FIXED_SM120_EXACT_N64_SYMBOL: &str = "nn_sm120_f32_n64_copyplan";
const FIXED_SM120_COPYPLAN_T256_SYMBOL: &str = "nn_sm120_f32_n64_copyplan_t256";
const FIXED_SM120_COPYPLAN_M128_T256_SYMBOL: &str = "nn_sm120_f32_n64_copyplan_m128n64_t256";

fn fixed_sm120_exact_n64_composed(arch: &str) -> bool {
    arch == "compute_120"
}

const FIXED_SM120_SLICED_SYMBOL: &str = "nn_sm120_f32_n64_sliced";
const FIXED_SM120_POSTBIAS_SYMBOLS: [&str; 6] = [
    "nn_sm120_tma_fma_postbias_m128n64_bk16_s2",
    "nn_sm120_tma_fma_postbias_m64n128_bk16_s2",
    "nn_sm120_tma_fma_postbias_m128n96_bk16_s2",
    "nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4",
    "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2",
    "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2",
];
const FIXED_SM120_POSTBIAS_REGISTER_CAP: i32 = 168;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedSm120PostbiasLaunchContract {
    threads: u32,
    dynamic_shared: usize,
    min_active_blocks: u32,
}

fn fixed_sm120_postbias_launch_contract(
    symbol: &str,
) -> Result<FixedSm120PostbiasLaunchContract, String> {
    match symbol {
        "nn_sm120_tma_fma_postbias_m128n64_bk16_s2"
        | "nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4"
        | "nn_sm120_tma_fma_postbias_m64n128_bk16_s2" => Ok(FixedSm120PostbiasLaunchContract {
            threads: 128,
            dynamic_shared: 24_592,
            min_active_blocks: 3,
        }),
        "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2"
        | "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2" => Ok(FixedSm120PostbiasLaunchContract {
            threads: 256,
            dynamic_shared: 24_592,
            min_active_blocks: 3,
        }),
        "nn_sm120_tma_fma_postbias_m128n96_bk16_s2" => Ok(FixedSm120PostbiasLaunchContract {
            threads: 256,
            dynamic_shared: 28_688,
            min_active_blocks: 3,
        }),
        _ => Err(format!(
            "{symbol} has no Fixed SM120 post-dot-bias launch contract"
        )),
    }
}

fn fixed_sm120_tensor_map_alignment_for_cuda_major(cuda_major: i32) -> Result<usize, String> {
    match cuda_major {
        12 => Ok(64),
        13 => Ok(128),
        major => Err(format!("unsupported CUDA tensor-map ABI major {major}")),
    }
}

fn validate_fixed_sm120_postbias_ptx_for_cuda_major(
    arch: &str,
    ptx: &str,
    cuda_major: i32,
) -> Result<(), String> {
    let parsed = parse_ptx(ptx)?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .filter(|entry| entry.symbol.contains("_sm120_tma_fma_"))
        .collect();
    if !fixed_sm120_exact_n64_composed(arch) {
        return if actual.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "Fixed SM120 post-dot-bias exports are foreign on {arch}"
            ))
        };
    }
    let expected: BTreeSet<_> = FIXED_SM120_POSTBIAS_SYMBOLS.into_iter().collect();
    let symbols: Vec<_> = actual.iter().map(|entry| entry.symbol.as_str()).collect();
    let unique: BTreeSet<_> = symbols.iter().copied().collect();
    if symbols.len() != expected.len() || unique != expected {
        return Err("Fixed SM120 exact-FMA requires exactly its six unique v1 exports".into());
    }
    let tensor_map_alignment =
        fixed_sm120_tensor_map_alignment_for_cuda_major(cuda_major)?.to_string();

    for entry in actual {
        let symbol = entry.symbol.as_str();
        let launch = fixed_sm120_postbias_launch_contract(symbol)?;
        let header = entry
            .text
            .split_once('{')
            .map(|(header, _)| header)
            .ok_or_else(|| format!("{symbol} has no PTX body"))?;
        let tokens = ptx_tokens(header);
        let text: Vec<_> = tokens.iter().map(|token| token.text).collect();
        let begin = text
            .iter()
            .position(|token| *token == "(")
            .ok_or_else(|| format!("{symbol} has no PTX parameters"))?;
        let end = text
            .iter()
            .position(|token| *token == ")")
            .ok_or_else(|| format!("{symbol} has no PTX parameter end"))?;
        if end <= begin {
            return Err(format!("{symbol} has malformed PTX parameters"));
        }
        let declarations: Vec<_> = text[begin + 1..end].split(|token| *token == ",").collect();
        let u64_pointer = |decl: &&[&str]| {
            (decl.len() == 3 && decl[..2] == [".param", ".u64"])
                || (decl.len() == 6 && decl[..5] == [".param", ".u64", ".ptr", ".align", "1"])
        };
        let tensor_map = |decl: &&[&str]| {
            decl.len() == 8
                && decl[0] == ".param"
                && decl[1] == ".align"
                && decl[2] == tensor_map_alignment
                && decl[3] == ".b8"
                && decl[5..] == ["[", "128", "]"]
        };
        let bundle = |decl: &&[&str]| {
            decl.len() == 8
                && decl[..4] == [".param", ".align", "4", ".b8"]
                && decl[5..] == ["[", "32", "]"]
        };
        if declarations.len() != 7
            || !declarations[..3].iter().all(u64_pointer)
            || !declarations[3..5].iter().all(tensor_map)
            || !u64_pointer(&declarations[5])
            || !bundle(&declarations[6])
        {
            return Err(format!(
                "{symbol} requires three pointers, two by-value tensor maps, a bias pointer, and an align-4 32-byte bundle"
            ));
        }
        for directive in [".maxntid", ".minnctapersm"] {
            let positions: Vec<_> = text
                .iter()
                .enumerate()
                .filter_map(|(index, token)| (*token == directive).then_some(index))
                .collect();
            if positions.len() != 1 {
                return Err(format!("{symbol} has the wrong {directive} launch bound"));
            }
            let values: Vec<_> = text[positions[0] + 1..]
                .iter()
                .copied()
                .take_while(|token| !token.starts_with('.'))
                .collect();
            let valid = match directive {
                ".maxntid" => {
                    let threads = launch.threads.to_string();
                    values == [threads.as_str()] || values == [threads.as_str(), ",", "1", ",", "1"]
                }
                _ => values == [launch.min_active_blocks.to_string().as_str()],
            };
            if !valid {
                return Err(format!("{symbol} has the wrong {directive} launch bound"));
            }
        }
        for required in [
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
            "fma.rn.f32",
            "mul.rn.f32",
        ] {
            if !ptx_has_unquoted_token(&entry.body, |token| token == required) {
                return Err(format!("{symbol} is missing {required}"));
            }
        }
        let has_bias = !symbol.contains("_nobias_");
        if has_bias != ptx_has_unquoted_token(&entry.body, |token| token == "add.rn.f32") {
            return Err(format!("{symbol} has the wrong bias epilogue"));
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token == ".local"
                || token.starts_with("ld.local")
                || token.starts_with("st.local")
                || token.starts_with("mma.")
                || token.starts_with("wmma.")
                || token.split('.').any(|part| part == "tf32" || part == "ftz")
                || token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
                || token == "call"
                || token.starts_with("call.")
                || token == ".callprototype"
                || token == ".calltargets"
        }) {
            return Err(format!(
                "{symbol} contains local, tensor, reduction, FTZ, or device-call work"
            ));
        }
    }
    Ok(())
}

fn validate_fixed_sm120_postbias_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    validate_fixed_sm120_postbias_ptx_for_cuda_major(arch, ptx, nvrtc_version().0)
}

fn validate_fixed_sm120_sliced_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let parsed = parse_ptx(ptx)?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .filter(|entry| entry.symbol.starts_with("nn_sm120_f32_n64_sliced"))
        .collect();
    if !fixed_sm120_exact_n64_composed(arch) {
        return if actual.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "Fixed SM120 sliced N64 copy-plan export is foreign on {arch}"
            ))
        };
    }
    if actual.len() != 1 || actual[0].symbol != FIXED_SM120_SLICED_SYMBOL {
        return Err("Fixed SM120 sliced N64 requires exactly its unique v1 export".into());
    }
    let entry = actual[0];
    let symbol = FIXED_SM120_SLICED_SYMBOL;
    let header = entry
        .text
        .split_once('{')
        .map(|(header, _)| header)
        .ok_or_else(|| format!("{symbol} has no PTX body"))?;
    let tokens = ptx_tokens(header);
    let text: Vec<_> = tokens.iter().map(|token| token.text).collect();
    let begin = text
        .iter()
        .position(|token| *token == "(")
        .ok_or_else(|| format!("{symbol} has no PTX parameters"))?;
    let end = text
        .iter()
        .position(|token| *token == ")")
        .ok_or_else(|| format!("{symbol} has no PTX parameter end"))?;
    if end <= begin {
        return Err(format!("{symbol} has malformed PTX parameters"));
    }
    let declarations: Vec<_> = text[begin + 1..end].split(|token| *token == ",").collect();
    let pointer_decl = |decl: &&[&str]| {
        (decl.len() == 3 && decl[..2] == [".param", ".u64"])
            || (decl.len() == 6 && decl[..5] == [".param", ".u64", ".ptr", ".align", "1"])
    };
    if declarations.len() != 5
        || !declarations[..4].iter().all(pointer_decl)
        || declarations[4].len() != 8
        || declarations[4][..4] != [".param", ".align", "4", ".b8"]
        || declarations[4][5..] != ["[", "32", "]"]
    {
        return Err(format!(
            "{symbol} requires four pointers and an align-4 32-byte bundle"
        ));
    }
    for directive in [".maxntid", ".minnctapersm"] {
        let positions: Vec<_> = text
            .iter()
            .enumerate()
            .filter_map(|(index, token)| (*token == directive).then_some(index))
            .collect();
        if positions.len() != 1 {
            return Err(format!("{symbol} has the wrong {directive} launch bound"));
        }
        let values: Vec<_> = text[positions[0] + 1..]
            .iter()
            .copied()
            .take_while(|token| !token.starts_with('.'))
            .collect();
        let valid = match directive {
            ".maxntid" => values == ["128"] || values == ["128", ",", "1", ",", "1"],
            _ => values == ["2"],
        };
        if !valid {
            return Err(format!("{symbol} has the wrong {directive} launch bound"));
        }
    }
    for required in [
        "fma.rn.f32",
        "cp.async.cg.shared.global",
        "cp.async.commit_group",
        "cp.async.wait_group",
    ] {
        if !ptx_has_unquoted_token(&entry.body, |token| token == required) {
            return Err(format!("{symbol} is missing {required}"));
        }
    }
    if ptx_has_unquoted_token(&entry.body, |token| {
        token == ".local"
            || token.starts_with("ld.local")
            || token.starts_with("st.local")
            || token.starts_with("mma.")
            || token.starts_with("wmma.")
            || token.split('.').any(|part| part == "tf32" || part == "ftz")
            || token.starts_with("atom.")
            || token.starts_with("atom::")
            || token.starts_with("red.")
            || token.starts_with("red::")
            || token.starts_with("redux.")
    }) {
        return Err(format!(
            "{symbol} contains local, tensor, reduction, or FTZ work"
        ));
    }
    Ok(())
}

fn validate_fixed_sm120_exact_n64_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let parsed = parse_ptx(ptx)?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .filter(|entry| entry.symbol.starts_with("nn_sm120_f32_n64_copyplan"))
        .collect();
    if !fixed_sm120_exact_n64_composed(arch) {
        return if actual.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "Fixed SM120 exact N64 copy-plan export is foreign on {arch}"
            ))
        };
    }
    let expected: BTreeSet<_> = [
        FIXED_SM120_EXACT_N64_SYMBOL,
        FIXED_SM120_COPYPLAN_T256_SYMBOL,
        FIXED_SM120_COPYPLAN_M128_T256_SYMBOL,
    ]
    .into_iter()
    .collect();
    if actual.len() != 3
        || actual
            .iter()
            .map(|entry| entry.symbol.as_str())
            .collect::<BTreeSet<_>>()
            != expected
    {
        return Err(
            "Fixed SM120 exact N64 requires exactly its control and two T256 exports".into(),
        );
    }
    for entry in actual {
        let symbol = entry.symbol.as_str();
        let (threads, min_blocks) = if symbol == FIXED_SM120_COPYPLAN_T256_SYMBOL {
            ("256", "3")
        } else if symbol == FIXED_SM120_COPYPLAN_M128_T256_SYMBOL {
            ("256", "2")
        } else {
            ("128", "2")
        };
        let header = entry
            .text
            .split_once('{')
            .map(|(header, _)| header)
            .ok_or_else(|| format!("{symbol} has no PTX body"))?;
        let tokens = ptx_tokens(header);
        let text: Vec<_> = tokens.iter().map(|token| token.text).collect();
        let begin = text
            .iter()
            .position(|token| *token == "(")
            .ok_or_else(|| format!("{symbol} has no PTX parameters"))?;
        let end = text
            .iter()
            .position(|token| *token == ")")
            .ok_or_else(|| format!("{symbol} has no PTX parameter end"))?;
        if end <= begin {
            return Err(format!("{symbol} has malformed PTX parameters"));
        }
        let declarations: Vec<_> = text[begin + 1..end].split(|token| *token == ",").collect();
        let pointer_decl = |decl: &&[&str]| {
            (decl.len() == 3 && decl[..2] == [".param", ".u64"])
                || (decl.len() == 6 && decl[..5] == [".param", ".u64", ".ptr", ".align", "1"])
        };
        if declarations.len() != 5
            || !declarations[..4].iter().all(pointer_decl)
            || declarations[4].len() != 8
            || declarations[4][..4] != [".param", ".align", "4", ".b8"]
            || declarations[4][5..] != ["[", "32", "]"]
        {
            return Err(format!(
                "{symbol} requires four pointers and an align-4 32-byte bundle"
            ));
        }
        for directive in [".maxntid", ".minnctapersm"] {
            let positions: Vec<_> = text
                .iter()
                .enumerate()
                .filter_map(|(index, token)| (*token == directive).then_some(index))
                .collect();
            if positions.len() != 1 {
                return Err(format!("{symbol} has the wrong {directive} launch bound"));
            }
            let values: Vec<_> = text[positions[0] + 1..]
                .iter()
                .copied()
                .take_while(|token| !token.starts_with('.'))
                .collect();
            let valid = match directive {
                ".maxntid" => values == [threads] || values == [threads, ",", "1", ",", "1"],
                _ => values == [min_blocks],
            };
            if !valid {
                return Err(format!("{symbol} has the wrong {directive} launch bound"));
            }
        }
        for required in [
            "fma.rn.f32",
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
        ] {
            if !ptx_has_unquoted_token(&entry.body, |token| token == required) {
                return Err(format!("{symbol} is missing {required}"));
            }
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token == ".local"
                || token.starts_with("ld.local")
                || token.starts_with("st.local")
                || token.starts_with("mma.")
                || token.starts_with("wmma.")
                || token.split('.').any(|part| part == "tf32" || part == "ftz")
                || token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
        }) {
            return Err(format!(
                "{symbol} contains local, tensor, reduction, or FTZ work"
            ));
        }
    }
    Ok(())
}

fn validate_fixed_sm89_exact_n64_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let parsed = parse_ptx(ptx)?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .filter(|entry| entry.symbol.starts_with("nn_sm89_f32_n64_copyplan"))
        .collect();
    if !fixed_sm89_exact_n64_composed(arch) {
        return if actual.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "Fixed exact N64 copy-plan export is foreign on {arch}"
            ))
        };
    }
    if actual.len() != 1 || actual[0].symbol != FIXED_SM89_EXACT_N64_SYMBOL {
        return Err("Fixed SM89 exact N64 requires exactly its unique v1 export".into());
    }
    let entry = actual[0];
    let symbol = FIXED_SM89_EXACT_N64_SYMBOL;
    let header = entry
        .text
        .split_once('{')
        .map(|(header, _)| header)
        .ok_or_else(|| format!("{symbol} has no PTX body"))?;
    let tokens = ptx_tokens(header);
    let text: Vec<_> = tokens.iter().map(|token| token.text).collect();
    let begin = text
        .iter()
        .position(|token| *token == "(")
        .ok_or_else(|| format!("{symbol} has no PTX parameters"))?;
    let end = text
        .iter()
        .position(|token| *token == ")")
        .ok_or_else(|| format!("{symbol} has no PTX parameter end"))?;
    if end <= begin {
        return Err(format!("{symbol} has malformed PTX parameters"));
    }
    let declarations: Vec<_> = text[begin + 1..end].split(|token| *token == ",").collect();
    if declarations.len() != 5
        || !declarations[..4]
            .iter()
            .all(|decl| decl.len() == 3 && decl[..2] == [".param", ".u64"])
        || declarations[4].len() != 8
        || declarations[4][..4] != [".param", ".align", "4", ".b8"]
        || declarations[4][5..] != ["[", "32", "]"]
    {
        return Err(format!(
            "{symbol} requires four pointers and an align-4 32-byte bundle"
        ));
    }
    // Inspect directives, not incidental numbers or quoted source locations.
    for directive in [".maxntid", ".minnctapersm"] {
        let positions: Vec<_> = text
            .iter()
            .enumerate()
            .filter_map(|(index, token)| (*token == directive).then_some(index))
            .collect();
        if positions.len() != 1 {
            return Err(format!("{symbol} has the wrong {directive} launch bound"));
        }
        let values: Vec<_> = text[positions[0] + 1..]
            .iter()
            .copied()
            .take_while(|token| !token.starts_with('.'))
            .collect();
        let valid = match directive {
            ".maxntid" => values == ["128"] || values == ["128", ",", "1", ",", "1"],
            _ => values == ["2"],
        };
        if !valid {
            return Err(format!("{symbol} has the wrong {directive} launch bound"));
        }
    }
    for required in [
        "fma.rn.f32",
        "cp.async.cg.shared.global",
        "cp.async.commit_group",
        "cp.async.wait_group",
    ] {
        if !ptx_has_unquoted_token(&entry.body, |token| token == required) {
            return Err(format!("{symbol} is missing {required}"));
        }
    }
    if ptx_has_unquoted_token(&entry.body, |token| {
        token == ".local"
            || token.starts_with("ld.local")
            || token.starts_with("st.local")
            || token.starts_with("mma.")
            || token.starts_with("wmma.")
            || token.split('.').any(|part| part == "tf32" || part == "ftz")
            || token.starts_with("atom.")
            || token.starts_with("atom::")
            || token.starts_with("red.")
            || token.starts_with("red::")
            || token.starts_with("redux.")
    }) {
        return Err(format!(
            "{symbol} contains local, tensor, reduction, or FTZ work"
        ));
    }
    Ok(())
}

fn validate_fixed_sm89_exact_n64_driver_abi(
    symbol: &str,
    abi: &Tf32DriverAbi,
) -> Result<(), String> {
    const EXPECTED: [(usize, usize); 5] = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    if abi.parameter_count() != EXPECTED.len()
        || !abi
            .parameters()
            .iter()
            .zip(EXPECTED)
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{symbol} has the wrong live five-argument/64-byte Driver ABI"
        ));
    }
    Ok(())
}

fn census_fixed_sm89_exact_n64_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    if kind != ModuleKind::Fixed || !fixed_sm89_exact_n64_composed(arch) {
        return Ok(BTreeMap::new());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let symbol = FIXED_SM89_EXACT_N64_SYMBOL;
    let function = unsafe {
        cudarc::driver::result::module::get_function(module.raw(), CString::new(symbol).unwrap())
    }
    .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
    let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| unsafe {
        get(function, index, offset, size)
    })?;
    validate_fixed_sm89_exact_n64_driver_abi(symbol, &abi)?;
    module.unload()?;
    Ok(BTreeMap::from([(symbol, abi)]))
}

fn census_fixed_sm120_exact_n64_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    if kind != ModuleKind::Fixed || !fixed_sm120_exact_n64_composed(arch) {
        return Ok(BTreeMap::new());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for symbol in [
        FIXED_SM120_EXACT_N64_SYMBOL,
        FIXED_SM120_COPYPLAN_T256_SYMBOL,
        FIXED_SM120_COPYPLAN_M128_T256_SYMBOL,
    ] {
        let function = unsafe {
            cudarc::driver::result::module::get_function(
                module.raw(),
                CString::new(symbol).unwrap(),
            )
        }
        .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
        let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| unsafe {
            get(function, index, offset, size)
        })?;
        validate_fixed_sm89_exact_n64_driver_abi(symbol, &abi)?;
        census.insert(symbol, abi);
    }
    module.unload()?;
    Ok(census)
}

fn census_fixed_sm120_sliced_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    if kind != ModuleKind::Fixed || !fixed_sm120_exact_n64_composed(arch) {
        return Ok(BTreeMap::new());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let symbol = FIXED_SM120_SLICED_SYMBOL;
    let function = unsafe {
        cudarc::driver::result::module::get_function(module.raw(), CString::new(symbol).unwrap())
    }
    .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
    let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| unsafe {
        get(function, index, offset, size)
    })?;
    validate_fixed_sm89_exact_n64_driver_abi(symbol, &abi)?;
    module.unload()?;
    Ok(BTreeMap::from([(symbol, abi)]))
}

fn validate_fixed_sm120_postbias_driver_abi_for_cuda_major(
    symbol: &str,
    abi: &Tf32DriverAbi,
    cuda_major: i32,
) -> Result<(), String> {
    const CUDA_12: [(usize, usize); 7] = [
        (0, 8),
        (8, 8),
        (16, 8),
        (64, 128),
        (192, 128),
        (320, 8),
        (328, 32),
    ];
    const CUDA_13: [(usize, usize); 7] = [
        (0, 8),
        (8, 8),
        (16, 8),
        (128, 128),
        (256, 128),
        (384, 8),
        (392, 32),
    ];
    let (expected, terminal_bytes) = match cuda_major {
        12 => (&CUDA_12, 360),
        13 => (&CUDA_13, 424),
        major => return Err(format!("unsupported CUDA tensor-map ABI major {major}")),
    };
    if abi.parameter_count() != expected.len()
        || !abi
            .parameters()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual.offset(), actual.size()) == *expected)
    {
        return Err(format!(
            "{symbol} has the wrong live seven-argument/{terminal_bytes}-byte Driver ABI"
        ));
    }
    Ok(())
}

fn validate_fixed_sm120_postbias_driver_abi(
    symbol: &str,
    abi: &Tf32DriverAbi,
) -> Result<(), String> {
    validate_fixed_sm120_postbias_driver_abi_for_cuda_major(symbol, abi, nvrtc_version().0)
}

fn census_fixed_sm120_postbias_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<BTreeMap<&'static str, Tf32DriverAbi>, String> {
    if kind != ModuleKind::Fixed || !fixed_sm120_exact_n64_composed(arch) {
        return Ok(BTreeMap::new());
    }
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let module = DriverModule::load(ctx, ptx)?;
    let get: GetParamInfo =
        unsafe { std::mem::transmute(driver_proc_address("cuFuncGetParamInfo", 12_040)?) };
    let mut census = BTreeMap::new();
    for symbol in FIXED_SM120_POSTBIAS_SYMBOLS {
        let function = unsafe {
            cudarc::driver::result::module::get_function(
                module.raw(),
                CString::new(symbol).unwrap(),
            )
        }
        .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
        let abi = query_driver_parameter_abi(symbol, 7, |index, offset, size| unsafe {
            get(function, index, offset, size)
        })?;
        validate_fixed_sm120_postbias_driver_abi(symbol, &abi)?;
        census.insert(symbol, abi);
    }
    module.unload()?;
    Ok(census)
}

#[derive(Clone, Copy)]
struct FixedSm89ExactN64Resources {
    local_bytes: i32,
    registers: i32,
    static_shared_bytes: i32,
    max_threads: i32,
    active_blocks: u32,
    preferred_carveout: i32,
}

#[derive(Clone, Copy)]
struct FixedSm120PostbiasResources {
    local_bytes: i32,
    registers: i32,
    max_threads: i32,
    active_blocks: u32,
}

fn validate_fixed_sm120_postbias_resources(
    symbol: &str,
    resources: FixedSm120PostbiasResources,
) -> Result<(), String> {
    let launch = fixed_sm120_postbias_launch_contract(symbol)?;
    let register_cap = match symbol {
        // Three 256-thread CTAs must fit within 65,536 registers.
        "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2"
        | "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2" => 85,
        _ => FIXED_SM120_POSTBIAS_REGISTER_CAP,
    };
    if resources.local_bytes != 0
        || !(1..=register_cap).contains(&resources.registers)
        || resources.max_threads < launch.threads as i32
        || resources.active_blocks < launch.min_active_blocks
    {
        return Err(format!(
            "{symbol} resource admission declined: local={} registers={} max_threads={} active_blocks={}",
            resources.local_bytes,
            resources.registers,
            resources.max_threads,
            resources.active_blocks,
        ));
    }
    Ok(())
}

fn validate_fixed_sm89_exact_n64_resources(
    resources: FixedSm89ExactN64Resources,
) -> Result<(), String> {
    if resources.local_bytes != 0
        || !(1..=160).contains(&resources.registers)
        || resources.static_shared_bytes != FIXED_SM89_EXACT_N64_STATIC_SHARED
        || resources.max_threads < FIXED_SM89_EXACT_N64_THREADS as i32
        || resources.active_blocks < 3
        || resources.preferred_carveout != 100
    {
        return Err(format!(
            "{} resource admission declined: local={} registers={} static_shared={} max_threads={} active_blocks={} carveout={}",
            FIXED_SM89_EXACT_N64_SYMBOL,
            resources.local_bytes,
            resources.registers,
            resources.static_shared_bytes,
            resources.max_threads,
            resources.active_blocks,
            resources.preferred_carveout,
        ));
    }
    Ok(())
}

#[cfg(test)]
fn validate_fixed_sm120_exact_n64_resources(
    resources: FixedSm89ExactN64Resources,
) -> Result<(), String> {
    validate_fixed_sm120_exact_n64_resources_for(FIXED_SM120_EXACT_N64_SYMBOL, resources)
}

fn validate_fixed_sm120_exact_n64_resources_for(
    symbol: &str,
    resources: FixedSm89ExactN64Resources,
) -> Result<(), String> {
    let (register_cap, threads, shared, blocks) = if symbol == FIXED_SM120_COPYPLAN_M128_T256_SYMBOL
    {
        // Prefer <=120; hard two-CTA budget: 2 * 256 * 128 = 65,536 registers.
        (128, 256, 49_152, 2)
    } else if symbol == FIXED_SM120_COPYPLAN_T256_SYMBOL {
        (85, 256, 32_768, 3)
    } else {
        (160, 128, 32_768, 3)
    };
    if resources.local_bytes != 0
        || !(1..=register_cap).contains(&resources.registers)
        || resources.static_shared_bytes != shared
        || resources.max_threads < threads
        || resources.active_blocks < blocks
        || resources.preferred_carveout != 100
    {
        return Err(format!(
            "{} resource admission declined: local={} registers={} static_shared={} max_threads={} active_blocks={} carveout={}",
            symbol,
            resources.local_bytes,
            resources.registers,
            resources.static_shared_bytes,
            resources.max_threads,
            resources.active_blocks,
            resources.preferred_carveout,
        ));
    }
    Ok(())
}

/// Optional Fixed-owned SM120 exact-FMA post-dot-bias tiles. ABI and control
/// resource failures decline this holder without affecting mandatory Fixed
/// kernels or exact copy-plan routes. K4 and T256 resource failures decline
/// only the corresponding force-only twin.
pub(crate) fn load_fixed_sm120_fma_postbias(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<FixedSm120FmaPostbiasKernels>, Option<String>) {
    let admitted = (|| -> Result<FixedSm120FmaPostbiasKernels, String> {
        if module.artifact_identity.module_kind != ModuleKind::Fixed
            || module.compiler_identity.target.as_str() != "compute_120"
            || ctx
                .compute_capability()
                .map_err(|error| format!("query Fixed SM120 postbias CC: {error:?}"))?
                != (12, 0)
        {
            return Err(
                "Fixed SM120 post-dot-bias is only composed for compute_120 and admitted on CC12.0"
                    .into(),
            );
        }
        let census = module
            .fixed_sm120_postbias_driver_abi
            .as_ref()
            .map_err(Clone::clone)?;
        let load = |symbol| -> Result<CudaFunction, String> {
            let launch = fixed_sm120_postbias_launch_contract(symbol)?;
            validate_fixed_sm120_postbias_driver_abi(
                symbol,
                census
                    .get(symbol)
                    .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?,
            )?;
            let function = load_function(&module.module, ModuleKind::Fixed, symbol)?;
            set_dynamic_shared(
                &function,
                symbol,
                i32::try_from(launch.dynamic_shared)
                    .map_err(|_| format!("{symbol} dynamic shared memory exceeds i32::MAX"))?,
            )?;
            let query_error = |label, error| format!("query {symbol} {label}: {error:?}");
            let resources = FixedSm120PostbiasResources {
                local_bytes: function
                    .local_size_bytes()
                    .map_err(|error| query_error("local", error))?,
                registers: function
                    .num_regs()
                    .map_err(|error| query_error("registers", error))?,
                max_threads: function
                    .max_threads_per_block()
                    .map_err(|error| query_error("threads", error))?,
                active_blocks: function
                    .occupancy_max_active_blocks_per_multiprocessor(
                        launch.threads,
                        launch.dynamic_shared,
                        None,
                    )
                    .map_err(|error| query_error("occupancy", error))?,
            };
            validate_fixed_sm120_postbias_resources(symbol, resources)?;
            Ok(function)
        };
        let m128n64 = load(FIXED_SM120_POSTBIAS_SYMBOLS[0])?;
        let m64n128 = load(FIXED_SM120_POSTBIAS_SYMBOLS[1])?;
        let m128n96 = load(FIXED_SM120_POSTBIAS_SYMBOLS[2])?;
        // A force-only candidate resource miss must not disable qualified AUTO.
        let (m128n64_k4, m128n64_k4_rejection) = match load(FIXED_SM120_POSTBIAS_SYMBOLS[3]) {
            Ok(function) => (Some(function), None),
            Err(reason) => (None, Some(reason)),
        };
        let (m128n64_t256, m128n64_t256_rejection) = match load(FIXED_SM120_POSTBIAS_SYMBOLS[4]) {
            Ok(function) => (Some(function), None),
            Err(reason) => (None, Some(reason)),
        };
        let (nobias_m128n64_t256, nobias_m128n64_t256_rejection) =
            match load(FIXED_SM120_POSTBIAS_SYMBOLS[5]) {
                Ok(function) => (Some(function), None),
                Err(reason) => (None, Some(reason)),
            };
        Ok(FixedSm120FmaPostbiasKernels {
            m128n64,
            m64n128,
            m128n96,
            m128n64_k4,
            m128n64_k4_rejection,
            m128n64_t256,
            m128n64_t256_rejection,
            nobias_m128n64_t256,
            nobias_m128n64_t256_rejection,
        })
    })();
    match admitted {
        Ok(kernels) => (Some(kernels), None),
        Err(reason) => (None, Some(reason)),
    }
}

/// Candidate-only admission/configuration; never changes incumbent attributes.
/// Resource/Driver failures retain a reason and leave mandatory Fixed routes live.
pub(crate) fn load_fixed_sm89_f32_n64_copyplan(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<CudaFunction>, Option<String>) {
    let admitted = (|| -> Result<CudaFunction, String> {
        if module.artifact_identity.module_kind != ModuleKind::Fixed
            || !fixed_sm89_exact_n64_composed(module.compiler_identity.target.as_str())
            || ctx
                .compute_capability()
                .map_err(|error| format!("query Fixed exact N64 CC: {error:?}"))?
                .0
                < 8
        {
            return Err(
                "Fixed exact N64 copy-plan is composed only with the portable overlay".into(),
            );
        }
        let symbol = FIXED_SM89_EXACT_N64_SYMBOL;
        let census = module
            .fixed_sm89_exact_n64_driver_abi
            .as_ref()
            .map_err(Clone::clone)?;
        validate_fixed_sm89_exact_n64_driver_abi(
            symbol,
            census
                .get(symbol)
                .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?,
        )?;
        let function = load_function(&module.module, ModuleKind::Fixed, symbol)?;
        let carveout = cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
        function
            .set_attribute(
                carveout,
                cudarc::driver::sys::CUshared_carveout::CU_SHAREDMEM_CARVEOUT_MAX_SHARED as i32,
            )
            .map_err(|error| format!("configure {symbol} MaxShared: {error:?}"))?;
        let query_error = |label, error| format!("query {symbol} {label}: {error:?}");
        let resources = FixedSm89ExactN64Resources {
            local_bytes: function
                .local_size_bytes()
                .map_err(|e| query_error("local", e))?,
            registers: function
                .num_regs()
                .map_err(|e| query_error("registers", e))?,
            static_shared_bytes: function
                .shared_size_bytes()
                .map_err(|e| query_error("static shared", e))?,
            max_threads: function
                .max_threads_per_block()
                .map_err(|e| query_error("threads", e))?,
            active_blocks: function
                .occupancy_max_active_blocks_per_multiprocessor(
                    FIXED_SM89_EXACT_N64_THREADS,
                    0,
                    None,
                )
                .map_err(|e| query_error("occupancy", e))?,
            preferred_carveout: function
                .get_attribute(carveout)
                .map_err(|e| query_error("carveout", e))?,
        };
        validate_fixed_sm89_exact_n64_resources(resources)?;
        Ok(function)
    })();
    match admitted {
        Ok(function) => (Some(function), None),
        Err(reason) => (None, Some(reason)),
    }
}

/// SM120 candidate-only admission/configuration. Incumbent attributes are
/// query-only and every failure leaves the existing exact routes available.
pub(crate) fn load_fixed_sm120_f32_n64_copyplan(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<CudaFunction>, Option<String>) {
    load_fixed_sm120_f32_n64_copyplan_for(ctx, module, FIXED_SM120_EXACT_N64_SYMBOL, 128)
}

pub(crate) fn load_fixed_sm120_f32_n64_copyplan_t256(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<CudaFunction>, Option<String>) {
    load_fixed_sm120_f32_n64_copyplan_for(ctx, module, FIXED_SM120_COPYPLAN_T256_SYMBOL, 256)
}

pub(crate) fn load_fixed_sm120_f32_m128n64_copyplan_t256(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<CudaFunction>, Option<String>) {
    load_fixed_sm120_f32_n64_copyplan_for(ctx, module, FIXED_SM120_COPYPLAN_M128_T256_SYMBOL, 256)
}

fn load_fixed_sm120_f32_n64_copyplan_for(
    ctx: &CudaContext,
    module: &CompiledModule,
    symbol: &'static str,
    threads: u32,
) -> (Option<CudaFunction>, Option<String>) {
    let admitted = (|| -> Result<CudaFunction, String> {
        if module.artifact_identity.module_kind != ModuleKind::Fixed
            || !fixed_sm120_exact_n64_composed(module.compiler_identity.target.as_str())
            || ctx
                .compute_capability()
                .map_err(|error| format!("query Fixed SM120 exact N64 CC: {error:?}"))?
                != (12, 0)
        {
            return Err(
                "Fixed SM120 exact N64 copy-plan is only composed for compute_120 and admitted on CC12.0"
                    .into(),
            );
        }
        let census = module
            .fixed_sm120_exact_n64_driver_abi
            .as_ref()
            .map_err(Clone::clone)?;
        validate_fixed_sm89_exact_n64_driver_abi(
            symbol,
            census
                .get(symbol)
                .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?,
        )?;
        let function = load_function(&module.module, ModuleKind::Fixed, symbol)?;
        let carveout = cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
        function
            .set_attribute(
                carveout,
                cudarc::driver::sys::CUshared_carveout::CU_SHAREDMEM_CARVEOUT_MAX_SHARED as i32,
            )
            .map_err(|error| format!("configure {symbol} MaxShared: {error:?}"))?;
        let query_error = |label, error| format!("query {symbol} {label}: {error:?}");
        let resources = FixedSm89ExactN64Resources {
            local_bytes: function
                .local_size_bytes()
                .map_err(|e| query_error("local", e))?,
            registers: function
                .num_regs()
                .map_err(|e| query_error("registers", e))?,
            static_shared_bytes: function
                .shared_size_bytes()
                .map_err(|e| query_error("static shared", e))?,
            max_threads: function
                .max_threads_per_block()
                .map_err(|e| query_error("threads", e))?,
            active_blocks: function
                .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
                .map_err(|e| query_error("occupancy", e))?,
            preferred_carveout: function
                .get_attribute(carveout)
                .map_err(|e| query_error("carveout", e))?,
        };
        validate_fixed_sm120_exact_n64_resources_for(symbol, resources)?;
        Ok(function)
    })();
    match admitted {
        Ok(function) => (Some(function), None),
        Err(reason) => (None, Some(reason)),
    }
}

/// Independent optional sliced route; configures only its own function.
pub(crate) fn load_fixed_sm120_f32_n64_sliced(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> (Option<CudaFunction>, Option<String>) {
    let admitted = (|| -> Result<CudaFunction, String> {
        if module.artifact_identity.module_kind != ModuleKind::Fixed
            || !fixed_sm120_exact_n64_composed(module.compiler_identity.target.as_str())
            || ctx
                .compute_capability()
                .map_err(|error| format!("query Fixed SM120 sliced N64 CC: {error:?}"))?
                != (12, 0)
        {
            return Err(
                "Fixed SM120 sliced N64 copy-plan is only composed for compute_120 and admitted on CC12.0"
                    .into(),
            );
        }
        let symbol = FIXED_SM120_SLICED_SYMBOL;
        let census = module
            .fixed_sm120_sliced_driver_abi
            .as_ref()
            .map_err(Clone::clone)?;
        validate_fixed_sm89_exact_n64_driver_abi(
            symbol,
            census
                .get(symbol)
                .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?,
        )?;
        let function = load_function(&module.module, ModuleKind::Fixed, symbol)?;
        let carveout = cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
        function
            .set_attribute(
                carveout,
                cudarc::driver::sys::CUshared_carveout::CU_SHAREDMEM_CARVEOUT_MAX_SHARED as i32,
            )
            .map_err(|error| format!("configure {symbol} MaxShared: {error:?}"))?;
        let query_error = |label, error| format!("query {symbol} {label}: {error:?}");
        let resources = FixedSm89ExactN64Resources {
            local_bytes: function
                .local_size_bytes()
                .map_err(|e| query_error("local", e))?,
            registers: function
                .num_regs()
                .map_err(|e| query_error("registers", e))?,
            static_shared_bytes: function
                .shared_size_bytes()
                .map_err(|e| query_error("static shared", e))?,
            max_threads: function
                .max_threads_per_block()
                .map_err(|e| query_error("threads", e))?,
            active_blocks: function
                .occupancy_max_active_blocks_per_multiprocessor(
                    FIXED_SM89_EXACT_N64_THREADS,
                    0,
                    None,
                )
                .map_err(|e| query_error("occupancy", e))?,
            preferred_carveout: function
                .get_attribute(carveout)
                .map_err(|e| query_error("carveout", e))?,
        };
        validate_fixed_sm120_exact_n64_resources_for(FIXED_SM120_SLICED_SYMBOL, resources)?;
        Ok(function)
    })();
    match admitted {
        Ok(function) => (Some(function), None),
        Err(reason) => (None, Some(reason)),
    }
}

fn validate_fixed_tf32_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let owns_sm120 = matches!(arch, "sm_120" | "sm_121" | "compute_120" | "compute_121");
    let mut expected = FIXED_TF32_SYMBOLS.to_vec();
    if fixed_sm89_rna_wide_composed(arch) {
        expected.push(FIXED_SM89_RNA_WIDE_SYMBOL);
        expected.push(FIXED_SM89_RNA_N96_SYMBOL);
    }
    if owns_sm120 {
        expected.extend(FIXED_SM120_TF32_SYMBOLS);
    }
    let symbols = ptx_entry_symbols(ptx)?;
    let actual: BTreeSet<_> = symbols
        .iter()
        .map(String::as_str)
        .filter(|symbol| symbol.contains("_tf32_"))
        .collect();
    let expected: BTreeSet<_> = expected.into_iter().collect();
    if actual != expected {
        return Err("Fixed TF32 PTX inventory is incomplete or contains foreign entries".into());
    }
    if super::super::gemm_bi_inference::FIXED_TF32_PARAMS_SIZE != 24 {
        return Err("Fixed portable TF32 host parameter ABI drifted".into());
    }
    let portable_bundle = ".param .align 4 .b8 ";
    for symbol in FIXED_TF32_SYMBOLS {
        let entry = ptx_entry(ptx, symbol)?;
        let parameters = entry
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
        let declarations: Vec<_> = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect();
        if declarations.len() != 5
            || !declarations[..4]
                .iter()
                .all(|line| line.starts_with(".param .u64 "))
            || !declarations[4].starts_with(portable_bundle)
            || !declarations[4].contains("[24]")
        {
            return Err(format!("{symbol} has the wrong five-parameter ABI"));
        }
        for required in [
            "cvt.rna.tf32.f32",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        ] {
            if !entry.contains(required) {
                return Err(format!("{symbol} is missing {required}"));
            }
        }
    }
    if owns_sm120 {
        let map_size = super::super::gemm_bi_inference::FIXED_TENSOR_MAP_SIZE;
        let map_alignment = super::super::gemm_bi_inference::FIXED_TENSOR_MAP_ALIGN;
        let expected_alignment = match nvrtc_version().0 {
            12 => 64,
            13 => 128,
            major => return Err(format!("unsupported CUDA tensor-map ABI major {major}")),
        };
        if map_size != 128 || map_alignment != expected_alignment {
            return Err(format!(
                "Fixed tensor-map ABI mismatch: size={map_size} align={map_alignment}, expected 128/{expected_alignment}"
            ));
        }
        if super::super::gemm_bi_inference::FIXED_SM120_TF32_PARAMS_SIZE != 16 {
            return Err("Fixed SM120 TF32 host parameter ABI drifted".into());
        }
        if super::super::gemm_bi_inference::FIXED_SM120_HALF_PARAMS_SIZE != 40 {
            return Err("Fixed SM120 half host parameter ABI drifted".into());
        }
        let map = format!(".param .align {expected_alignment} .b8 ");
        for symbol in FIXED_SM120_TF32_SYMBOLS {
            let entry = ptx_entry(ptx, symbol)?;
            let parameters = entry
                .split_once('(')
                .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
                .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
            let declarations: Vec<_> = parameters
                .lines()
                .map(str::trim)
                .filter(|line| line.starts_with(".param "))
                .collect();
            let is_map = |line: &str| line.starts_with(&map) && line.contains("[128]");
            if declarations.len() != 5
                || !declarations[0].starts_with(".param .u64 ")
                || !is_map(declarations[1])
                || !is_map(declarations[2])
                || !declarations[3].starts_with(".param .u64 ")
                || !declarations[4].starts_with(".param .align 4 .b8 ")
                || !declarations[4].contains("[16]")
            {
                return Err(format!("{symbol} has the wrong five-parameter TMA ABI"));
            }
            for required in [
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                "cvt.rna.tf32.f32",
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
            ] {
                if !entry.contains(required) {
                    return Err(format!("{symbol} is missing {required}"));
                }
            }
        }
        let expected_half: BTreeSet<_> = FIXED_SM120_HALF_BASES
            .iter()
            .flat_map(|base| {
                [
                    format!("{base}_bf16"),
                    format!("{base}_f16"),
                    format!("{base}_f32out_bf16"),
                    format!("{base}_f32out_f16"),
                ]
            })
            .collect();
        let actual_half: BTreeSet<_> = symbols
            .iter()
            .filter(|symbol| {
                symbol.starts_with("nn_sm120_tma_")
                    && !symbol.contains("_tf32_")
                    && (symbol.ends_with("_bf16") || symbol.ends_with("_f16"))
            })
            .cloned()
            .collect();
        if actual_half != expected_half {
            return Err(
                "Fixed SM120 half PTX inventory is incomplete or contains foreign entries".into(),
            );
        }
        for symbol in &expected_half {
            let entry = ptx_entry(ptx, symbol)?;
            let parameters = entry
                .split_once('(')
                .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
                .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
            let declarations: Vec<_> = parameters
                .lines()
                .map(str::trim)
                .filter(|line| line.starts_with(".param "))
                .collect();
            let is_map = |line: &str| line.starts_with(&map) && line.contains("[128]");
            if declarations.len() != 5
                || !declarations[0].starts_with(".param .u64 ")
                || !is_map(declarations[1])
                || !is_map(declarations[2])
                || !declarations[3].starts_with(".param .u64 ")
                || !declarations[4].starts_with(".param .align 4 .b8 ")
                || !declarations[4].contains("[40]")
            {
                return Err(format!("{symbol} has the wrong five-parameter TMA ABI"));
            }
            for required in [
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                "mbarrier.arrive.release.cta.shared::cta.b64",
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            ] {
                if !entry.contains(required) {
                    return Err(format!("{symbol} is missing {required}"));
                }
            }
            let mma = if symbol.ends_with("_bf16") {
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
            } else {
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            };
            if !entry.contains(mma) {
                return Err(format!("{symbol} is missing {mma}"));
            }
        }
    }
    let stripped = strip_ptx_comments(ptx)?;
    for symbol in expected {
        let entry = ptx_entry(&stripped, symbol)?;
        if ptx_has_unquoted_token(&entry, |token| {
            token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
        }) {
            return Err(format!(
                "Fixed TF32 {symbol} contains a reduction instruction"
            ));
        }
    }
    if owns_sm120 {
        for base in FIXED_SM120_HALF_BASES {
            for suffix in ["_bf16", "_f16", "_f32out_bf16", "_f32out_f16"] {
                let symbol = format!("{base}{suffix}");
                let entry = ptx_entry(&stripped, &symbol)?;
                if ptx_has_unquoted_token(&entry, |token| {
                    token.starts_with("atom.")
                        || token.starts_with("atom::")
                        || token.starts_with("red.")
                        || token.starts_with("red::")
                        || token.starts_with("redux.")
                }) {
                    return Err(format!(
                        "Fixed SM120 half {symbol} contains a reduction instruction"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_scalar_zero_reduction_ptx(ptx: &str) -> Result<(), String> {
    let expected: BTreeSet<_> = SCALAR_ZERO_REDUCTION_SYMBOLS.iter().copied().collect();
    let symbols = ptx_entry_symbols(ptx)?;
    let actual: Vec<_> = symbols
        .iter()
        .map(String::as_str)
        .filter(|symbol| symbol.ends_with("_zero_reduction"))
        .collect();
    let unique: BTreeSet<_> = actual.iter().copied().collect();
    if actual.len() != unique.len() || unique != expected {
        return Err(
            "TriadScalar zero-reduction PTX inventory is incomplete, duplicated, or foreign".into(),
        );
    }
    let host_size = super::launch::GEMM_BI_ZERO_REDUCTION_PARAMS_SIZE;
    if host_size != 32 {
        return Err(format!(
            "Rust zero-reduction parameter size is {host_size}, expected 32"
        ));
    }
    for symbol in SCALAR_ZERO_REDUCTION_SYMBOLS {
        let entry = ptx_entry(ptx, symbol)?;
        let parameters = entry
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
        let declarations: Vec<_> = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect();
        let pointers_are_u64 = declarations
            .get(..4)
            .is_some_and(|pointers| pointers.iter().all(|line| line.starts_with(".param .u64 ")));
        let bundle_is_exact = declarations
            .get(4)
            .is_some_and(|line| line.starts_with(".param .align 4 .b8 ") && line.contains("[32]"));
        if declarations.len() != 5 || !pointers_are_u64 || !bundle_is_exact {
            return Err(format!("{symbol} has the wrong five-parameter ABI"));
        }
    }
    Ok(())
}

fn validate_scalar_nt_m2n16_ptx(ptx: &str) -> Result<(), String> {
    const SYMBOL: &str = "nt_m2n16_bk64_splitk32";
    let entry = ptx_entry(ptx, SYMBOL)?;
    let parameters = entry
        .split_once('(')
        .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
        .ok_or_else(|| format!("{SYMBOL} has no PTX parameter list"))?;
    let declarations: Vec<_> = parameters
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with(".param "))
        .collect();
    if declarations.len() != 7
        || !declarations[..3]
            .iter()
            .all(|line| line.starts_with(".param .u64 "))
        || !declarations[3].starts_with(".param .f32 ")
        || !declarations[4..]
            .iter()
            .all(|line| line.starts_with(".param .u32 "))
    {
        return Err(format!("{SYMBOL} has the wrong seven-parameter ABI"));
    }
    let body = ptx_entry_body(ptx, SYMBOL)?;
    for required in ["fma.rn.f32", "add.rn.f32", "mul.rn.f32"] {
        if !body.contains(required) {
            return Err(format!("{SYMBOL} is missing {required}"));
        }
    }
    if ptx_has_unquoted_token(&body, |token| {
        token.starts_with("atom.")
            || token.starts_with("atom::")
            || token.starts_with("red.")
            || token.starts_with("red::")
            || token.starts_with("redux.")
            || token.starts_with("mma.")
            || token.contains(".ftz")
            || token == "call"
            || token.starts_with("call.")
    }) {
        return Err(format!(
            "{SYMBOL} contains a forbidden PTX instruction family"
        ));
    }
    Ok(())
}

fn validate_scalar_nn_m32n64_splitk32_ptx(ptx: &str) -> Result<(), String> {
    const SYMBOL: &str = "nn_splitk32_m32n64_exact";
    let entry = ptx_entry(ptx, SYMBOL)?;
    let parameters = entry
        .split_once('(')
        .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
        .ok_or_else(|| format!("{SYMBOL} has no PTX parameter list"))?;
    let declarations: Vec<_> = parameters
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with(".param "))
        .collect();
    if declarations.len() != 7
        || !declarations[..3]
            .iter()
            .all(|line| line.starts_with(".param .u64 "))
        || !declarations[3..]
            .iter()
            .all(|line| line.starts_with(".param .u32 "))
    {
        return Err(format!("{SYMBOL} has the wrong seven-parameter ABI"));
    }
    let body = ptx_entry_body(ptx, SYMBOL)?;
    if !body.contains("fma.rn.f32") {
        return Err(format!("{SYMBOL} is missing fma.rn.f32"));
    }
    if ptx_has_unquoted_token(&body, |token| {
        token.starts_with("atom.")
            || token.starts_with("atom::")
            || token.starts_with("red.")
            || token.starts_with("red::")
            || token.starts_with("redux.")
            || token.starts_with("mma.")
            || token.contains(".ftz")
            || token == "call"
            || token.starts_with("call.")
    }) {
        return Err(format!(
            "{SYMBOL} contains a forbidden PTX instruction family"
        ));
    }
    Ok(())
}

fn validate_scalar_tn_m16n16_ptx(ptx: &str) -> Result<(), String> {
    const SYMBOL: &str = "tn_m16n16_bk16_s2_splitm16";
    let entry = ptx_entry(ptx, SYMBOL)?;
    let parameters = entry
        .split_once('(')
        .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
        .ok_or_else(|| format!("{SYMBOL} has no PTX parameter list"))?;
    let declarations: Vec<_> = parameters
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with(".param "))
        .collect();
    if declarations.len() != 7
        || !declarations[..3]
            .iter()
            .all(|line| line.starts_with(".param .u64 "))
        || !declarations[3].starts_with(".param .f32 ")
        || !declarations[4..]
            .iter()
            .all(|line| line.starts_with(".param .u32 "))
    {
        return Err(format!("{SYMBOL} has the wrong seven-parameter ABI"));
    }
    let body = ptx_entry_body(ptx, SYMBOL)?;
    let mut previous = 0;
    for required in [
        "fma.rn.f32",
        "add.rn.f64",
        "mul.rn.f64",
        "cvt.rn.f32.f64",
        "add.rn.f32",
    ] {
        let offset = body[previous..]
            .find(required)
            .map(|offset| previous + offset)
            .ok_or_else(|| format!("{SYMBOL} is missing ordered {required}"))?;
        previous = offset + required.len();
    }
    if ptx_has_unquoted_token(&body, |token| {
        token.starts_with("atom.")
            || token.starts_with("atom::")
            || token.starts_with("red.")
            || token.starts_with("red::")
            || token.starts_with("redux.")
            || token.starts_with("mma.")
            || token.contains(".ftz")
            || token == "call"
            || token.starts_with("call.")
    }) {
        return Err(format!(
            "{SYMBOL} contains a forbidden PTX instruction family"
        ));
    }
    Ok(())
}

fn validate_tf32_specialization(
    module_kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> Result<(), String> {
    if !matches!(
        module_kind,
        ModuleKind::TriadSm80
            | ModuleKind::TriadSm90a
            | ModuleKind::TriadSm100
            | ModuleKind::TriadSm120
            | ModuleKind::TriadSm89Finalist
    ) {
        return Err(format!("{module_kind:?} does not own TF32 kernels"));
    }
    let extensions = module_kind == ModuleKind::TriadSm80 && sm80_target_composes_streamk(arch);
    if module_kind == ModuleKind::TriadSm89Finalist {
        validate_sm89_finalist_ptx_inventory(ptx)?;
    } else {
        validate_tf32_ptx_inventory(module_kind, extensions, ptx)?;
    }
    validate_tf32_parameter_abi(module_kind, extensions, ptx, nvrtc_version().0)?;
    validate_tf32_host_abi(module_kind, extensions, ptx)?;
    validate_tf32_feature_instructions(module_kind, extensions, ptx)?;
    if module_kind == ModuleKind::TriadSm80 {
        validate_tf32_splitk_ptx(extensions, ptx)?;
    }
    Ok(())
}

fn validate_tf32_host_abi(
    module_kind: ModuleKind,
    extensions: bool,
    ptx: &str,
) -> Result<(), String> {
    let map_size = std::mem::size_of::<cudarc::driver::sys::CUtensorMap>();
    let map_alignment = std::mem::align_of::<cudarc::driver::sys::CUtensorMap>();
    if map_size != 128 || !matches!(map_alignment, 64 | 128) {
        return Err(format!(
            "unsupported Rust CUtensorMap ABI size={map_size} align={map_alignment}"
        ));
    }
    let host_cuda_major = if map_alignment == 64 { 12 } else { 13 };
    if nvrtc_version().0 != host_cuda_major {
        return Err(format!(
            "Rust CUtensorMap ABI is CUDA {host_cuda_major}, runtime NVRTC is CUDA {}",
            nvrtc_version().0
        ));
    }
    let parameter_size = super::launch::tf32_kernel_params_size(module_kind)
        .ok_or_else(|| format!("{module_kind:?} does not own a TF32 parameter ABI"))?;
    let expected_size = if matches!(
        module_kind,
        ModuleKind::TriadSm80 | ModuleKind::TriadSm89Finalist
    ) {
        32
    } else {
        40
    };
    if parameter_size != expected_size {
        return Err(format!(
            "{module_kind:?} Rust TF32 parameter size is {parameter_size}, expected {expected_size}"
        ));
    }
    validate_tf32_parameter_abi(module_kind, extensions, ptx, host_cuda_major)
}

/// Whether a module of the sm_80 instruction tier (the Ada-found triad
/// modules among them) compiles for `arch` on a board of `device_cc`: any
/// admitted portable target on an SM80-or-newer board.
pub(crate) fn sm80_tier_module_compiles(arch: &str, device_cc: Option<(i32, i32)>) -> bool {
    sm80_ptx_target(arch).is_some() && device_cc.is_some_and(|(major, _)| major >= 8)
}

pub(super) fn sm80_ptx_target(arch: &str) -> Option<&'static str> {
    match arch {
        "sm_80" => Some("sm_80"),
        "sm_86" => Some("sm_86"),
        "sm_87" => Some("sm_87"),
        "sm_89" => Some("sm_89"),
        "sm_90" => Some("sm_90"),
        "sm_90a" => Some("sm_90a"),
        "sm_100" => Some("sm_100"),
        "sm_100a" => Some("sm_100a"),
        "sm_101a" => Some("sm_101a"),
        "sm_103a" => Some("sm_103a"),
        "sm_107a" => Some("sm_107a"),
        "sm_110" => Some("sm_110"),
        "sm_110a" => Some("sm_110a"),
        "sm_120" | "compute_120" => Some("sm_120"),
        "sm_121" | "compute_121" => Some("sm_121"),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct PtxToken<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct ParsedPtxEntry {
    symbol: String,
    text: String,
    body: String,
}

#[derive(Debug)]
struct ParsedPtxFunction {
    symbol: String,
    body: Option<String>,
}

#[derive(Debug)]
struct ParsedPtx {
    target: Option<String>,
    entries: Vec<ParsedPtxEntry>,
    functions: Vec<ParsedPtxFunction>,
}

fn strip_ptx_comments(ptx: &str) -> Result<String, String> {
    let mut stripped = ptx.as_bytes().to_vec();
    let mut cursor = 0;
    while cursor < stripped.len() {
        if stripped[cursor] == b'"' {
            cursor += 1;
            let mut closed = false;
            while cursor < stripped.len() {
                match stripped[cursor] {
                    b'\\' => cursor = cursor.saturating_add(2),
                    b'"' => {
                        cursor += 1;
                        closed = true;
                        break;
                    }
                    _ => cursor += 1,
                }
            }
            if !closed {
                return Err("PTX contains an unterminated string literal".into());
            }
            continue;
        }
        if stripped.get(cursor..cursor + 2) == Some(b"//") {
            while cursor < stripped.len() && stripped[cursor] != b'\n' {
                stripped[cursor] = b' ';
                cursor += 1;
            }
            continue;
        }
        if stripped.get(cursor..cursor + 2) == Some(b"/*") {
            stripped[cursor] = b' ';
            stripped[cursor + 1] = b' ';
            cursor += 2;
            let mut closed = false;
            while cursor < stripped.len() {
                if stripped.get(cursor..cursor + 2) == Some(b"*/") {
                    stripped[cursor] = b' ';
                    stripped[cursor + 1] = b' ';
                    cursor += 2;
                    closed = true;
                    break;
                }
                if stripped[cursor] != b'\n' {
                    stripped[cursor] = b' ';
                }
                cursor += 1;
            }
            if !closed {
                return Err("PTX contains an unterminated block comment".into());
            }
            continue;
        }
        cursor += 1;
    }
    String::from_utf8(stripped).map_err(|_| "comment-stripped PTX is not UTF-8".to_string())
}

fn ptx_tokens(ptx: &str) -> Vec<PtxToken<'_>> {
    let bytes = ptx.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
            continue;
        }
        let start = cursor;
        if bytes[cursor] == b'"' {
            cursor += 1;
            while cursor < bytes.len() {
                match bytes[cursor] {
                    b'\\' => cursor = cursor.saturating_add(2),
                    b'"' => {
                        cursor += 1;
                        break;
                    }
                    _ => cursor += 1,
                }
            }
        } else if matches!(
            bytes[cursor],
            b'(' | b')' | b'{' | b'}' | b'[' | b']' | b',' | b';'
        ) {
            cursor += 1;
        } else {
            cursor += 1;
            while cursor < bytes.len()
                && !bytes[cursor].is_ascii_whitespace()
                && !matches!(
                    bytes[cursor],
                    b'(' | b')' | b'{' | b'}' | b'[' | b']' | b',' | b';' | b'"'
                )
            {
                cursor += 1;
            }
        }
        tokens.push(PtxToken {
            text: &ptx[start..cursor],
            start,
            end: cursor,
        });
    }
    tokens
}

fn ptx_has_unquoted_token(ptx: &str, mut predicate: impl FnMut(&str) -> bool) -> bool {
    ptx_tokens(ptx)
        .into_iter()
        .filter(|token| !token.text.starts_with('"'))
        .any(|token| predicate(token.text))
}

fn ptx_atomic_inc_limits(body: &str) -> Result<Vec<u32>, String> {
    let tokens = ptx_tokens(body);
    let mut limits = Vec::new();
    for (opcode_index, token) in tokens.iter().enumerate() {
        if token.text != "atom.global.inc.u32" {
            continue;
        }
        let mut bracket_depth = 0_usize;
        let mut operands = vec![Vec::new()];
        let mut terminated = false;
        for operand in tokens.iter().skip(opcode_index + 1) {
            match operand.text {
                "[" => {
                    bracket_depth += 1;
                    operands.last_mut().unwrap().push(operand.text);
                }
                "]" => {
                    bracket_depth = bracket_depth
                        .checked_sub(1)
                        .ok_or_else(|| "TF32 split-K atomicInc has an unmatched ']'".to_owned())?;
                    operands.last_mut().unwrap().push(operand.text);
                }
                "," if bracket_depth == 0 => operands.push(Vec::new()),
                ";" if bracket_depth == 0 => {
                    terminated = true;
                    break;
                }
                _ => operands.last_mut().unwrap().push(operand.text),
            }
        }
        if !terminated || bracket_depth != 0 || operands.len() != 3 || operands[2].len() != 1 {
            return Err("TF32 split-K atomicInc must have three well-formed operands".into());
        }
        let limit = operands[2][0]
            .parse::<u32>()
            .map_err(|_| "TF32 split-K atomicInc limit must be a decimal immediate".to_owned())?;
        limits.push(limit);
    }
    Ok(limits)
}

fn require_last_block_protocol_order(
    label: &str,
    entry: &ParsedPtxEntry,
    store_opcodes: &[&str],
    load_opcodes: &[&str],
) -> Result<(), String> {
    let tokens = ptx_tokens(&entry.body);
    let positions = |opcodes: &[&str]| {
        tokens
            .iter()
            .filter(|token| opcodes.contains(&token.text))
            .map(|token| token.start)
            .collect::<Vec<_>>()
    };
    let stores = positions(store_opcodes);
    let fences = positions(&["membar.gl"]);
    let atomics = positions(&["atom.global.inc.u32"]);
    let loads = positions(load_opcodes);
    if stores.is_empty() || fences.len() != 1 || atomics.len() != 1 || loads.is_empty() {
        return Err(format!(
            "{label}/{} has an incomplete last-block protocol",
            entry.symbol
        ));
    }
    let last_store = stores.into_iter().max().unwrap();
    let fence = fences[0];
    let atomic = atomics[0];
    let first_load = loads.into_iter().min().unwrap();
    if !(last_store < fence && fence < atomic && atomic < first_load) {
        return Err(format!(
            "{label}/{} must publish partials before the fence, signal completion after the fence, and reload only after the atomic",
            entry.symbol
        ));
    }
    Ok(())
}

fn is_ptx_symbol(token: &str) -> bool {
    let mut bytes = token.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'$'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$'))
}

fn matching_ptx_token(
    tokens: &[PtxToken<'_>],
    start: usize,
    open: &str,
    close: &str,
) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate().skip(start) {
        if token.text == open {
            depth += 1;
        } else if token.text == close {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn ptx_entry_signature_end(
    tokens: &[PtxToken<'_>],
    entry: usize,
    symbol: &str,
) -> Result<usize, String> {
    let after_symbol = entry + 2;
    match tokens.get(after_symbol).map(|token| token.text) {
        Some("(") => matching_ptx_token(tokens, after_symbol, "(", ")")
            .map(|close| close + 1)
            .ok_or_else(|| format!("PTX entry {symbol} has an unclosed parameter list")),
        Some("{") => Ok(after_symbol),
        Some(directive) if directive.starts_with('.') => Ok(after_symbol),
        Some(_) => Err(format!("PTX entry {symbol} has a malformed signature")),
        None => Err(format!("PTX entry {symbol} has no body")),
    }
}

fn ptx_entry_body_open(
    tokens: &[PtxToken<'_>],
    mut cursor: usize,
    symbol: &str,
) -> Result<usize, String> {
    loop {
        let token = tokens
            .get(cursor)
            .ok_or_else(|| format!("PTX entry {symbol} has no body"))?;
        if token.text == "{" {
            return Ok(cursor);
        }
        if matches!(
            token.text,
            ";" | "}" | ".entry" | ".extern" | ".func" | ".target"
        ) || !token.text.starts_with('.')
        {
            return Err(format!("PTX entry {symbol} has no body"));
        }

        if token.text == ".pragma" {
            tokens
                .get(cursor + 1)
                .filter(|value| value.text.starts_with('"') && value.text.ends_with('"'))
                .ok_or_else(|| format!("PTX entry {symbol} has a malformed pragma"))?;
            tokens
                .get(cursor + 2)
                .filter(|terminator| terminator.text == ";")
                .ok_or_else(|| format!("PTX entry {symbol} has a malformed pragma"))?;
            cursor += 3;
            continue;
        }

        cursor += 1;
        while let Some(operand) = tokens.get(cursor) {
            if operand.text == "{" || operand.text.starts_with('.') {
                break;
            }
            if operand.text == "}" {
                return Err(format!("PTX entry {symbol} has no body"));
            }
            cursor += 1;
            if operand.text == ";" {
                break;
            }
        }
    }
}

fn reject_nested_ptx_module_directives(
    tokens: &[PtxToken<'_>],
    body_open: usize,
    body_close: usize,
    owner: &str,
) -> Result<(), String> {
    if let Some(directive) = tokens[body_open + 1..body_close]
        .iter()
        .find(|token| matches!(token.text, ".entry" | ".func" | ".target"))
    {
        return Err(format!(
            "PTX {owner} contains nested module directive {}",
            directive.text
        ));
    }
    Ok(())
}

fn parse_ptx_function(
    ptx: &str,
    tokens: &[PtxToken<'_>],
    function: usize,
) -> Result<(ParsedPtxFunction, usize), String> {
    let mut cursor = function + 1;
    if tokens.get(cursor).is_some_and(|token| token.text == "(") {
        cursor = matching_ptx_token(tokens, cursor, "(", ")")
            .map(|close| close + 1)
            .ok_or_else(|| "PTX function has an unclosed return parameter list".to_string())?;
    }
    let symbol = tokens
        .get(cursor)
        .filter(|token| is_ptx_symbol(token.text))
        .ok_or_else(|| "PTX function has no valid symbol".to_string())?;
    cursor += 1;
    if tokens.get(cursor).is_some_and(|token| token.text == "(") {
        cursor = matching_ptx_token(tokens, cursor, "(", ")")
            .map(|close| close + 1)
            .ok_or_else(|| format!("PTX function {} has unclosed parameters", symbol.text))?;
    }

    let mut modifier = function;
    let mut is_extern = false;
    while modifier > 0 && tokens[modifier - 1].text.starts_with('.') {
        modifier -= 1;
        is_extern |= tokens[modifier].text == ".extern";
    }
    while let Some(token) = tokens.get(cursor) {
        match token.text {
            "{" => {
                if is_extern {
                    return Err(format!("extern PTX function {} has a body", symbol.text));
                }
                let body_close = matching_ptx_token(tokens, cursor, "{", "}")
                    .ok_or_else(|| format!("PTX function {} has an unclosed body", symbol.text))?;
                reject_nested_ptx_module_directives(tokens, cursor, body_close, "function")?;
                return Ok((
                    ParsedPtxFunction {
                        symbol: symbol.text.to_owned(),
                        body: Some(ptx[tokens[cursor].end..tokens[body_close].start].to_owned()),
                    },
                    body_close + 1,
                ));
            }
            ";" => {
                return Ok((
                    ParsedPtxFunction {
                        symbol: symbol.text.to_owned(),
                        body: None,
                    },
                    cursor + 1,
                ));
            }
            "}" | ".entry" | ".func" | ".target" => {
                return Err(format!("PTX function {} has no valid body", symbol.text));
            }
            _ => cursor += 1,
        }
    }
    Err(format!(
        "PTX function {} has no body or declaration terminator",
        symbol.text
    ))
}

fn parse_ptx(ptx: &str) -> Result<ParsedPtx, String> {
    let stripped = strip_ptx_comments(ptx)?;
    let tokens = ptx_tokens(&stripped);

    let mut target = None;
    let mut entries = Vec::new();
    let mut functions = Vec::new();
    let mut cursor = 0;
    while cursor < tokens.len() {
        match tokens[cursor].text {
            ".target" => {
                let value = tokens
                    .get(cursor + 1)
                    .filter(|token| is_ptx_symbol(token.text))
                    .ok_or_else(|| "PTX has a malformed target directive".to_string())?;
                if target.replace(value.text.to_owned()).is_some() {
                    return Err("PTX contains duplicate target directives".into());
                }
                cursor += 2;
                continue;
            }
            ".func" => {
                let (function, next) = parse_ptx_function(&stripped, &tokens, cursor)?;
                functions.push(function);
                cursor = next;
                continue;
            }
            "{" => {
                let close = matching_ptx_token(&tokens, cursor, "{", "}")
                    .ok_or_else(|| "PTX contains an unclosed module-scope brace".to_string())?;
                reject_nested_ptx_module_directives(&tokens, cursor, close, "module scope")?;
                cursor = close + 1;
                continue;
            }
            "}" => return Err("PTX contains an unmatched module-scope closing brace".into()),
            ".entry" => {}
            _ => {
                cursor += 1;
                continue;
            }
        }
        let mut modifier = cursor;
        while modifier > 0 && tokens[modifier - 1].text.starts_with('.') {
            modifier -= 1;
            if tokens[modifier].text == ".extern" {
                return Err("PTX contains an extern entry directive".into());
            }
        }
        let symbol = tokens
            .get(cursor + 1)
            .filter(|token| is_ptx_symbol(token.text))
            .ok_or_else(|| "PTX has an entry directive without a valid symbol".to_string())?;
        let signature_end = ptx_entry_signature_end(&tokens, cursor, symbol.text)?;
        let body_open = ptx_entry_body_open(&tokens, signature_end, symbol.text)?;
        let body_open_token = tokens
            .get(body_open)
            .ok_or_else(|| format!("PTX entry {} has no body", symbol.text))?;
        let body_close = matching_ptx_token(&tokens, body_open, "{", "}")
            .ok_or_else(|| format!("PTX entry {} has an unclosed body", symbol.text))?;
        reject_nested_ptx_module_directives(
            &tokens,
            body_open,
            body_close,
            &format!("entry {}", symbol.text),
        )?;
        entries.push(ParsedPtxEntry {
            symbol: symbol.text.to_owned(),
            text: stripped[tokens[cursor].start..tokens[body_close].end].to_owned(),
            body: stripped[body_open_token.end..tokens[body_close].start].to_owned(),
        });
        cursor = body_close + 1;
    }
    Ok(ParsedPtx {
        target,
        entries,
        functions,
    })
}

fn ptx_entry_symbols(ptx: &str) -> Result<Vec<String>, String> {
    Ok(parse_ptx(ptx)?
        .entries
        .into_iter()
        .map(|entry| entry.symbol)
        .collect())
}

fn validate_exact_ptx_exports(
    label: &str,
    expected_count: usize,
    expected: &[&str],
    ptx: &str,
) -> Result<ParsedPtx, String> {
    let expected_entries = expected.len();
    let mut expected_counts = BTreeMap::new();
    for &symbol in expected {
        *expected_counts.entry(symbol).or_insert(0_usize) += 1;
    }
    let expected_duplicates: Vec<_> = expected_counts
        .iter()
        .filter_map(|(&symbol, &count)| (count > 1).then_some(symbol))
        .collect();
    let expected: BTreeSet<_> = expected_counts.into_keys().collect();
    let parsed = parse_ptx(ptx).map_err(|error| format!("{label} PTX parse failed: {error}"))?;
    let actual: Vec<_> = parsed
        .entries
        .iter()
        .map(|entry| entry.symbol.as_str())
        .collect();
    let actual_entries = actual.len();
    let actual_unique: BTreeSet<_> = actual.iter().copied().collect();
    let mut actual_counts = BTreeMap::new();
    for &symbol in &actual {
        *actual_counts.entry(symbol).or_insert(0_usize) += 1;
    }
    let actual_duplicates: Vec<_> = actual_counts
        .into_iter()
        .filter_map(|(symbol, count)| (count > 1).then_some(symbol))
        .collect();
    let missing: Vec<_> = expected.difference(&actual_unique).copied().collect();
    let foreign: Vec<_> = actual_unique.difference(&expected).copied().collect();
    if expected_entries == expected_count
        && actual_entries == expected_count
        && actual_duplicates.is_empty()
        && missing.is_empty()
        && foreign.is_empty()
        && expected_duplicates.is_empty()
    {
        return Ok(parsed);
    }
    Err(format!(
        "{label} PTX export mismatch: expected_count={expected_count}; expected_entries={}; actual_entries={}; expected_duplicates={expected_duplicates:?}; actual_duplicates={actual_duplicates:?}; missing={missing:?}; foreign={foreign:?}",
        expected_entries, actual_entries
    ))
}

fn parsed_ptx_entry(ptx: &str, symbol: &str) -> Result<ParsedPtxEntry, String> {
    let mut matches = parse_ptx(ptx)?
        .entries
        .into_iter()
        .filter(|entry| entry.symbol == symbol);
    let entry = matches
        .next()
        .ok_or_else(|| format!("TF32 PTX is missing entry {symbol}"))?;
    if matches.next().is_some() {
        return Err(format!("TF32 PTX contains duplicate entry {symbol}"));
    }
    Ok(entry)
}

fn parsed_ptx_entry_ref<'a>(
    parsed: &'a ParsedPtx,
    symbol: &str,
) -> Result<&'a ParsedPtxEntry, String> {
    let mut matches = parsed.entries.iter().filter(|entry| entry.symbol == symbol);
    let entry = matches
        .next()
        .ok_or_else(|| format!("PTX is missing entry {symbol}"))?;
    if matches.next().is_some() {
        return Err(format!("PTX contains duplicate entry {symbol}"));
    }
    Ok(entry)
}

fn parsed_ptx_function_ref<'a>(
    parsed: &'a ParsedPtx,
    symbol: &str,
) -> Result<&'a ParsedPtxFunction, String> {
    let mut matches = parsed
        .functions
        .iter()
        .filter(|function| function.symbol == symbol);
    let function = matches
        .next()
        .ok_or_else(|| format!("PTX is missing function {symbol}"))?;
    if matches.next().is_some() {
        return Err(format!("PTX contains ambiguous function {symbol}"));
    }
    Ok(function)
}

fn ptx_direct_call_targets(body: &str) -> Result<Vec<String>, String> {
    let tokens = ptx_tokens(body);
    let mut targets = Vec::new();
    for (call, token) in tokens.iter().enumerate() {
        if token.text != "call" && !token.text.starts_with("call.") {
            continue;
        }
        if token.text != "call.uni" {
            return Err(format!(
                "PTX contains unsupported call opcode {}",
                token.text
            ));
        }
        let mut cursor = call + 1;
        if tokens.get(cursor).is_some_and(|token| token.text == "(") {
            cursor = matching_ptx_token(&tokens, cursor, "(", ")")
                .map(|close| close + 1)
                .ok_or_else(|| "PTX call has unclosed return arguments".to_string())?;
            if !tokens.get(cursor).is_some_and(|token| token.text == ",") {
                return Err("PTX call has no target separator".into());
            }
            cursor += 1;
        }
        let target = tokens
            .get(cursor)
            .filter(|target| is_ptx_symbol(target.text))
            .ok_or_else(|| "PTX call has no direct target".to_string())?;
        cursor += 1;
        if !tokens.get(cursor).is_some_and(|token| token.text == ",") {
            return Err(format!(
                "PTX call to {} has no argument separator",
                target.text
            ));
        }
        cursor += 1;
        if !tokens.get(cursor).is_some_and(|token| token.text == "(") {
            return Err(format!("PTX call to {} has no argument list", target.text));
        }
        cursor = matching_ptx_token(&tokens, cursor, "(", ")")
            .map(|close| close + 1)
            .ok_or_else(|| format!("PTX call to {} has unclosed arguments", target.text))?;
        if !tokens.get(cursor).is_some_and(|token| token.text == ";") {
            return Err(format!("PTX call to {} has no terminator", target.text));
        }
        targets.push(target.text.to_owned());
    }
    Ok(targets)
}

fn ptx_entry(ptx: &str, symbol: &str) -> Result<String, String> {
    Ok(parsed_ptx_entry(ptx, symbol)?.text)
}

fn ptx_entry_body(ptx: &str, symbol: &str) -> Result<String, String> {
    Ok(parsed_ptx_entry(ptx, symbol)?.body)
}

fn validate_tf32_parameter_abi(
    module_kind: ModuleKind,
    extensions: bool,
    ptx: &str,
    cuda_major: i32,
) -> Result<(), String> {
    let map_alignment = match cuda_major {
        12 => 64,
        13 => 128,
        _ => {
            return Err(format!(
                "unsupported CUDA tensor-map ABI major {cuda_major}"
            ));
        }
    };
    let parsed = parse_ptx(ptx)?;
    for kernel_spec in super::contract::tf32_route_specs_for(module_kind, extensions) {
        let entry = &parsed_ptx_entry_ref(&parsed, kernel_spec.symbol)?.text;
        let parameters = entry
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{} has no PTX parameter list", kernel_spec.symbol))?;
        let declarations: Vec<_> = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect();
        let bundle_size = if matches!(
            module_kind,
            ModuleKind::TriadSm80 | ModuleKind::TriadSm89Finalist
        ) {
            32
        } else {
            40
        };
        let bundle = ".param .align 4 .b8 ";
        let is_bundle =
            |line: &str| line.starts_with(bundle) && line.contains(&format!("[{bundle_size}]"));
        let is_u64 = |line: &str| line.starts_with(".param .u64 ");
        let map = format!(".param .align {map_alignment} .b8 ");
        let is_map = |line: &str| line.starts_with(&map) && line.contains("[128]");
        let seven_parameter_bundle = if kernel_spec.route.is_exact_fma() {
            Some(32)
        } else if matches!(
            kernel_spec.route,
            super::contract::Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_)
        ) {
            Some(bundle_size)
        } else {
            None
        };
        if let Some(bundle_size) = seven_parameter_bundle {
            // Output, slabs, flags, two tensor maps, bias, parameter bundle.
            let is_bundle =
                |line: &str| line.starts_with(bundle) && line.contains(&format!("[{bundle_size}]"));
            if declarations.len() != 7 {
                return Err(format!(
                    "{} must have seven ABI parameters",
                    kernel_spec.symbol
                ));
            }
            if !is_u64(declarations[0])
                || !is_u64(declarations[1])
                || !is_u64(declarations[2])
                || !is_map(declarations[3])
                || !is_map(declarations[4])
                || !is_u64(declarations[5])
                || !is_bundle(declarations[6])
            {
                return Err(format!(
                    "{} has the wrong stream-K tensor-map ABI",
                    kernel_spec.symbol
                ));
            }
            continue;
        }
        if declarations.len() != 5 {
            return Err(format!(
                "{} must have five ABI parameters",
                kernel_spec.symbol
            ));
        }
        if !is_bundle(declarations[4]) {
            return Err(format!(
                "{} has the wrong parameter bundle ABI",
                kernel_spec.symbol
            ));
        }
        if matches!(
            module_kind,
            ModuleKind::TriadSm80 | ModuleKind::TriadSm89Finalist
        ) {
            if !declarations[..4].iter().all(|line| is_u64(line)) {
                return Err(format!(
                    "{} has the wrong pointer parameter ABI",
                    kernel_spec.symbol
                ));
            }
        } else if !is_u64(declarations[0])
            || !is_map(declarations[1])
            || !is_map(declarations[2])
            || !is_u64(declarations[3])
        {
            return Err(format!(
                "{} has the wrong tensor-map ABI",
                kernel_spec.symbol
            ));
        }
    }
    Ok(())
}

fn validate_tf32_splitk_ptx(extensions: bool, ptx: &str) -> Result<(), String> {
    let expected = super::contract::tf32_splitk_specs_for(extensions)
        .map(|spec| spec.symbol)
        .collect::<BTreeSet<_>>();
    let symbols = ptx_entry_symbols(ptx)?;
    let actual = symbols
        .iter()
        .map(String::as_str)
        .filter(|symbol| symbol.contains("_tf32_splitk"))
        .collect::<Vec<_>>();
    let unique = actual.iter().copied().collect::<BTreeSet<_>>();
    if actual.len() != unique.len() || unique != expected {
        return Err(
            "TriadSm80 TF32 split-K PTX inventory is incomplete, duplicated, or foreign".into(),
        );
    }
    let parsed = parse_ptx(ptx)?;
    for spec in super::contract::tf32_splitk_specs_for(extensions) {
        let symbol = spec.symbol;
        let entry = parsed_ptx_entry_ref(&parsed, symbol)?;
        let parameters = entry
            .text
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
        let declarations = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect::<Vec<_>>();
        let pointers_are_u64 = declarations
            .get(..6)
            .is_some_and(|pointers| pointers.iter().all(|line| line.starts_with(".param .u64 ")));
        let bundle_is_exact = declarations
            .get(6)
            .is_some_and(|line| line.starts_with(".param .align 4 .b8 ") && line.contains("[32]"));
        if declarations.len() != 7 || !pointers_are_u64 || !bundle_is_exact {
            return Err(format!("{symbol} has the wrong seven-parameter ABI"));
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            (token.starts_with("atom.") && !token.starts_with("atom.global.inc.u32"))
                || token.starts_with("red.")
                || token.starts_with("redux.")
        }) {
            return Err(format!(
                "{symbol} contains a forbidden synchronization or reduction opcode"
            ));
        }
        let limits = ptx_atomic_inc_limits(&entry.body)?;
        let expected_limit = spec.partitions - 1;
        if limits != [expected_limit] {
            return Err(format!(
                "{symbol} must contain exactly one atomicInc with immediate limit {expected_limit}, got {limits:?}"
            ));
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token.starts_with("div.") || token.starts_with("rem.")
        }) {
            return Err(format!(
                "TriadSm80 TF32 split-K fused kernel {symbol} contains runtime division"
            ));
        }
        require_ptx_entry_tokens(
            "TriadSm80 TF32 split-K fused kernel",
            entry,
            &[
                "cvt.rna.tf32.f32",
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                "atom.global.inc.u32",
                "membar.gl",
                "add.rn.f32",
                "mul.rn.f32",
                "st.global.cg.f32",
                "st.global.cg.v2.f32",
                "ld.global.cg.f32",
                "ld.global.cg.v2.f32",
            ],
        )?;
        if spec.op == ResolvedGemmOp::Nn {
            require_ptx_entry_tokens("TriadSm80 TF32 split-K NN epilogue", entry, &["fma.rn.f32"])?;
        }
        require_last_block_protocol_order(
            "TriadSm80 TF32 split-K fused kernel",
            entry,
            &["st.global.cg.f32", "st.global.cg.v2.f32"],
            &["ld.global.cg.f32", "ld.global.cg.v2.f32"],
        )?;
    }
    Ok(())
}

fn validate_tn_narrow_splitm_partial_ptx(ptx: &str) -> Result<(), String> {
    validate_tn_splitm_partial_ptx_cohort(
        ptx,
        "tn_narrow_splitm_partial",
        SCALAR_TN_NARROW_SPLITM_PARTIAL_SYMBOLS,
        "TriadScalar TN narrow split-M partial",
        true,
    )
}

fn validate_tn_splitm_partial_ptx(ptx: &str) -> Result<(), String> {
    validate_tn_splitm_partial_ptx_cohort(
        ptx,
        "tn_splitm_partial",
        SCALAR_TN_SPLITM_PARTIAL_SYMBOLS,
        "TriadScalar TN split-M partial",
        false,
    )
}

fn validate_tn_splitm_partial_ptx_cohort(
    ptx: &str,
    symbol_marker: &str,
    expected_symbols: &[&str],
    label: &str,
    forbid_runtime_tile_division: bool,
) -> Result<(), String> {
    let symbols = ptx_entry_symbols(ptx)?;
    let actual = symbols
        .iter()
        .map(String::as_str)
        .filter(|symbol| symbol.starts_with(symbol_marker))
        .collect::<BTreeSet<_>>();
    let expected = expected_symbols.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!("{label} PTX inventory is incomplete or foreign"));
    }
    let parsed = parse_ptx(ptx)?;
    for symbol in expected_symbols {
        let entry = parsed_ptx_entry_ref(&parsed, symbol)?;
        let parameters = entry
            .text
            .split_once('(')
            .and_then(|(_, tail)| tail.split_once("\n)").map(|(head, _)| head))
            .ok_or_else(|| format!("{symbol} has no PTX parameter list"))?;
        let declarations = parameters
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(".param "))
            .collect::<Vec<_>>();
        let pointers_are_u64 = declarations
            .get(..3)
            .is_some_and(|items| items.iter().all(|line| line.starts_with(".param .u64 ")));
        let scalars_are_u32 = declarations
            .get(3..)
            .is_some_and(|items| items.iter().all(|line| line.starts_with(".param .u32 ")));
        if declarations.len() != 7 || !pointers_are_u64 || !scalars_are_u32 {
            return Err(format!("{symbol} has the wrong seven-parameter ABI"));
        }
        if ptx_has_unquoted_token(&entry.body, |token| {
            token.starts_with("atom.") || token.starts_with("red.") || token.starts_with("redux.")
        }) {
            return Err(format!("{symbol} contains an atomic or reduction opcode"));
        }
        if forbid_runtime_tile_division
            && ptx_has_unquoted_token(&entry.body, |token| {
                token.starts_with("div.") || token.starts_with("rem.")
            })
        {
            return Err(format!("{symbol} contains runtime tile division"));
        }
        require_ptx_entry_tokens(label, entry, &["fma.rn.f32"])?;
    }
    Ok(())
}

fn validate_sm80_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let expected = sm80_ptx_target(arch)
        .ok_or_else(|| format!("TriadSm80 has no portable target candidate for {arch}"))?;
    let actual = ptx_target(ptx)?;
    if actual != expected {
        return Err(format!(
            "TriadSm80 PTX target is {actual}, expected {expected}"
        ));
    }
    // The stream-K fragment is present exactly on the targets that compose
    // it: a missing kernel there or a stray one elsewhere is a composition
    // defect, not a runtime surprise.
    let symbols = ptx_entry_symbols(ptx)?;
    let composed = sm80_target_composes_streamk(arch);
    for symbol in SM80_STREAMK_SYMBOLS {
        let present = symbols.iter().any(|entry| entry == symbol);
        if present != composed {
            return Err(format!(
                "TriadSm80 PTX for {arch} {} {symbol}",
                if composed { "lacks" } else { "carries" }
            ));
        }
    }
    Ok(())
}

fn ptx_target(ptx: &str) -> Result<String, String> {
    parse_ptx(ptx)?
        .target
        .ok_or_else(|| "specialized PTX has no target directive".to_string())
}

fn require_ptx_entry_tokens(
    label: &str,
    entry: &ParsedPtxEntry,
    required: &[&str],
) -> Result<(), String> {
    require_ptx_scope_tokens(label, &entry.symbol, &entry.body, required)
}

fn require_ptx_scope_tokens(
    label: &str,
    scope: &str,
    body: &str,
    required: &[&str],
) -> Result<(), String> {
    for &instruction in required {
        if !ptx_has_unquoted_token(body, |token| token == instruction) {
            return Err(format!("{label}/{scope} PTX is missing {instruction}"));
        }
    }
    Ok(())
}

fn require_ptx_entry_opcode(
    label: &str,
    entry: &ParsedPtxEntry,
    description: &str,
    mut predicate: impl FnMut(&str) -> bool,
) -> Result<(), String> {
    if ptx_has_unquoted_token(&entry.body, |token| predicate(token)) {
        return Ok(());
    }
    Err(format!(
        "{label}/{} PTX is missing {description}",
        entry.symbol
    ))
}

fn reject_ptx_entry_tokens(
    label: &str,
    entry: &ParsedPtxEntry,
    forbidden: &[&str],
) -> Result<(), String> {
    for &instruction in forbidden {
        if ptx_has_unquoted_token(&entry.body, |token| token == instruction) {
            return Err(format!(
                "{label}/{} PTX contains incompatible core {instruction}",
                entry.symbol
            ));
        }
    }
    Ok(())
}

fn validate_sm90a_wg2_producer_features(parsed: &ParsedPtx) -> Result<(), String> {
    const PRODUCER: &[&str] = &[
        "setmaxnreg.dec.sync.aligned.u32",
        "bar.sync",
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "ret",
    ];
    let mut group_targets = Vec::new();
    for pair in SM90A_SYMBOLS[6..].as_chunks::<2>().0 {
        let mut pair_target = None;
        for &symbol in pair {
            let entry = parsed_ptx_entry_ref(parsed, symbol)?;
            let calls = ptx_direct_call_targets(&entry.body)
                .map_err(|error| format!("TriadSm90a/{symbol}: {error}"))?;
            if calls.len() != 1 {
                return Err(format!(
                    "TriadSm90a/{symbol} has {} direct calls, expected one producer call",
                    calls.len()
                ));
            }
            let target = &calls[0];
            let function = parsed_ptx_function_ref(parsed, target)?;
            let body = function
                .body
                .as_deref()
                .ok_or_else(|| format!("TriadSm90a producer {target} has no body"))?;
            require_ptx_scope_tokens("TriadSm90a producer", target, body, PRODUCER)?;
            if let Some(expected) = &pair_target
                && expected != target
            {
                return Err(format!(
                    "TriadSm90a WG2 pair {pair:?} calls different producers {expected} and {target}"
                ));
            }
            pair_target = Some(target.clone());
        }
        group_targets.push(
            pair_target
                .ok_or_else(|| "TriadSm90a WG2 metadata contains an empty pair".to_string())?,
        );
    }
    if group_targets.iter().collect::<BTreeSet<_>>().len() != group_targets.len() {
        return Err("TriadSm90a WG2 operation groups must call three distinct producers".into());
    }
    Ok(())
}

fn validate_sm90a_entry_features(parsed: &ParsedPtx) -> Result<(), String> {
    const BF16_CORE: &str = "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16";
    const F16_CORE: &str = "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16";
    const TF32_CORE: &str = "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32";
    const COMMON: &[&str] = &[
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
        "wgmma.fence.sync.aligned",
        "wgmma.commit_group.sync.aligned",
        "wgmma.wait_group.sync.aligned",
    ];
    const PRODUCER: &[&str] = &[
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
    ];

    for (index, &symbol) in SM90A_SYMBOLS.iter().enumerate() {
        let entry = parsed_ptx_entry_ref(parsed, symbol)?;
        require_ptx_entry_tokens("TriadSm90a", entry, COMMON)?;
        let (core, forbidden) = if index % 2 == 0 {
            (BF16_CORE, [F16_CORE, TF32_CORE])
        } else {
            (F16_CORE, [BF16_CORE, TF32_CORE])
        };
        require_ptx_entry_tokens("TriadSm90a", entry, &[core])?;
        reject_ptx_entry_tokens("TriadSm90a", entry, &forbidden)?;
        if index < 6 {
            require_ptx_entry_tokens("TriadSm90a", entry, PRODUCER)?;
        } else {
            require_ptx_entry_tokens("TriadSm90a", entry, &["setmaxnreg.inc.sync.aligned.u32"])?;
        }
    }
    validate_sm90a_wg2_producer_features(parsed)?;

    for spec in super::contract::tf32_route_specs(ModuleKind::TriadSm90a) {
        let entry = parsed_ptx_entry_ref(parsed, spec.symbol)?;
        require_ptx_entry_tokens("TriadSm90a", entry, COMMON)?;
        require_ptx_entry_tokens("TriadSm90a", entry, PRODUCER)?;
        require_ptx_entry_tokens("TriadSm90a", entry, &[TF32_CORE])?;
        reject_ptx_entry_tokens("TriadSm90a", entry, &[BF16_CORE, F16_CORE])?;
        let super::contract::Tf32PhysicalRoute::Sm90aWgmmaTf32Tma(route) = spec.route else {
            return Err(format!(
                "TriadSm90a/{} has foreign route metadata",
                spec.symbol
            ));
        };
        if route.schedule == super::contract::Sm90aWarpgroupSchedule::Wg2 {
            require_ptx_entry_tokens(
                "TriadSm90a",
                entry,
                &[
                    "setmaxnreg.inc.sync.aligned.u32",
                    "setmaxnreg.dec.sync.aligned.u32",
                ],
            )?;
        }
    }
    Ok(())
}

fn validate_sm90a_ptx(ptx: &str) -> Result<(), String> {
    let target = ptx_target(ptx)?;
    if target != "sm_90a" {
        return Err(format!(
            "TriadSm90a PTX target is {target}, expected sm_90a"
        ));
    }
    let mut expected = SM90A_SYMBOLS.to_vec();
    expected.extend(super::contract::tf32_module_symbols(ModuleKind::TriadSm90a));
    let parsed = validate_exact_ptx_exports("TriadSm90a", 18, &expected, ptx)?;
    validate_sm90a_entry_features(&parsed)?;
    let ptx = strip_ptx_comments(ptx)?;
    if ptx_has_unquoted_token(&ptx, |token| {
        token.starts_with("atom.")
            || token.starts_with("red.")
            || token.starts_with("atom::")
            || token.starts_with("red::")
            || token == "cvt.rna.tf32.f32"
    }) {
        return Err("TriadSm90a PTX contains a forbidden instruction family".into());
    }
    Ok(())
}

fn sm100_target_for_arch(arch: &str) -> Option<super::contract::Sm100TargetCandidate> {
    [(10, 0), (10, 3), (10, 7), (11, 0)]
        .into_iter()
        .flat_map(sm100_target_candidates)
        .copied()
        .find(|target| target.nvrtc_arch == arch)
}

fn validate_sm100_entry_features(parsed: &ParsedPtx) -> Result<(), String> {
    const COMMON: &[&str] = &[
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64",
        "tcgen05.fence::before_thread_sync",
        "tcgen05.fence::after_thread_sync",
        "tcgen05.ld.sync.aligned.32x32b.x8.b32",
        "tcgen05.wait::ld.sync.aligned",
    ];
    const STORE: &[&str] = &[
        "tcgen05.st.sync.aligned.32x32b.x8.b32",
        "tcgen05.wait::st.sync.aligned",
    ];

    for spec in &super::contract::SM100_KERNEL_SPECS {
        let entry = parsed_ptx_entry_ref(parsed, spec.symbol)?;
        require_ptx_entry_tokens("TriadSm100", entry, COMMON)?;
        require_ptx_entry_tokens("TriadSm100", entry, &["tcgen05.mma.cta_group::1.kind::f16"])?;
        reject_ptx_entry_tokens(
            "TriadSm100",
            entry,
            &["tcgen05.mma.cta_group::1.kind::tf32"],
        )?;
        if spec.op == super::contract::Sm100Op::Nn {
            require_ptx_entry_tokens("TriadSm100", entry, STORE)?;
        }
    }
    for spec in super::contract::tf32_route_specs(ModuleKind::TriadSm100) {
        let entry = parsed_ptx_entry_ref(parsed, spec.symbol)?;
        require_ptx_entry_tokens("TriadSm100", entry, COMMON)?;
        require_ptx_entry_tokens(
            "TriadSm100",
            entry,
            &["tcgen05.mma.cta_group::1.kind::tf32"],
        )?;
        reject_ptx_entry_tokens("TriadSm100", entry, &["tcgen05.mma.cta_group::1.kind::f16"])?;
        if spec.op == ResolvedGemmOp::Nn {
            require_ptx_entry_tokens("TriadSm100", entry, STORE)?;
        }
    }
    Ok(())
}

fn validate_sm100_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let candidate = sm100_target_for_arch(arch)
        .ok_or_else(|| format!("TriadSm100 has no target candidate for {arch}"))?;
    let actual = ptx_target(ptx)?;
    if actual != candidate.ptx_target {
        return Err(format!(
            "TriadSm100 PTX target is {actual}, expected {}",
            candidate.ptx_target
        ));
    }
    let mut expected: Vec<_> = super::contract::SM100_KERNEL_SPECS
        .iter()
        .map(|spec| spec.symbol)
        .collect();
    expected.extend(super::contract::tf32_module_symbols(ModuleKind::TriadSm100));
    let parsed = validate_exact_ptx_exports("TriadSm100", 108, &expected, ptx)?;
    validate_sm100_entry_features(&parsed)?;
    let ptx = strip_ptx_comments(ptx)?;
    validate_sm100_forbidden_instructions(&ptx)
}

fn validate_sm100_probe_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let candidate = sm100_target_for_arch(arch)
        .ok_or_else(|| format!("TriadSm100 has no target candidate for {arch}"))?;
    let actual = ptx_target(ptx)?;
    if actual != candidate.ptx_target {
        return Err(format!(
            "TriadSm100 probe PTX target is {actual}, expected {}",
            candidate.ptx_target
        ));
    }
    validate_exact_ptx_exports("TriadSm100 probe", 1, &["tcgen05_probe"], ptx)?;
    let body = ptx_entry_body(ptx, "tcgen05_probe")?;
    let ptx = strip_ptx_comments(ptx)?;
    validate_sm100_feature_instructions(&body, &ptx)
}

fn validate_sm100_feature_instructions(
    required_scope: &str,
    forbidden_scope: &str,
) -> Result<(), String> {
    let required_tokens = ptx_tokens(required_scope);
    for instruction in [
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
        "tcgen05.mma.cta_group::1.kind::f16",
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64",
        "tcgen05.fence::before_thread_sync",
        "tcgen05.fence::after_thread_sync",
        "tcgen05.ld.sync.aligned.32x32b.x8.b32",
        "tcgen05.wait::ld.sync.aligned",
        "tcgen05.st.sync.aligned.32x32b.x8.b32",
        "tcgen05.wait::st.sync.aligned",
    ] {
        if !required_tokens
            .iter()
            .any(|token| token.text == instruction)
        {
            return Err(format!("TriadSm100 PTX is missing {instruction}"));
        }
    }
    validate_sm100_forbidden_instructions(forbidden_scope)
}

fn validate_sm100_forbidden_instructions(forbidden_scope: &str) -> Result<(), String> {
    if ptx_has_unquoted_token(forbidden_scope, |token| {
        token.starts_with("atom.")
            || token.starts_with("red.")
            || token.starts_with("atom::")
            || token.starts_with("red::")
            || token.starts_with("tcgen05.ld.red")
            || token.starts_with("wgmma.")
            || token == "cvt.rna.tf32.f32"
            || (token.starts_with("tcgen05.") && token.contains("cta_group::2"))
            || token.contains(".multicast")
    }) {
        return Err("TriadSm100 PTX contains a forbidden instruction family".into());
    }
    if ptx_has_unquoted_token(forbidden_scope, |token| {
        token == "call"
            || token.starts_with("call.")
            || token == ".callprototype"
            || token == ".calltargets"
    }) {
        return Err("TriadSm100 PTX contains a device call instruction".into());
    }
    for symbol in [
        "cudaLaunchDevice",
        "cudaGetParameterBuffer",
        "cudaDeviceSynchronize",
        "__cudaPushCallConfiguration",
        "__cudaPopCallConfiguration",
        "malloc",
        "free",
        "operator new",
        "operator delete",
    ] {
        if ptx_has_unquoted_token(forbidden_scope, |token| token == symbol) {
            return Err(format!(
                "TriadSm100 PTX contains forbidden device-runtime symbol {symbol}"
            ));
        }
    }
    Ok(())
}

fn sm120_ptx_target(arch: &str) -> Option<&'static str> {
    match arch {
        "compute_120" => Some("sm_120"),
        "compute_121" => Some("sm_121"),
        _ => None,
    }
}

fn validate_sm120_entry_features(parsed: &ParsedPtx) -> Result<(), String> {
    const BF16_CORE: &str = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";
    const F16_CORE: &str = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32";
    const TF32_CORE: &str = "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32";
    const COMMON: &[&str] = &[
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "mbarrier.init.shared::cta.b64",
        "fence.mbarrier_init.release.cluster",
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
    ];

    for spec in super::contract::sm120_kernel_specs() {
        let entry = parsed_ptx_entry_ref(parsed, spec.symbol)?;
        require_ptx_entry_tokens("TriadSm120", entry, COMMON)?;
        require_ptx_entry_tokens(
            "TriadSm120",
            entry,
            &["mbarrier.arrive.release.cta.shared::cta.b64"],
        )?;
        let (core, forbidden) = match spec.dtype {
            super::super::dtype::WeightDtype::Bf16 => (BF16_CORE, [F16_CORE, TF32_CORE]),
            super::super::dtype::WeightDtype::F16 => (F16_CORE, [BF16_CORE, TF32_CORE]),
            super::super::dtype::WeightDtype::F32 | super::super::dtype::WeightDtype::Tf32 => {
                return Err(format!(
                    "TriadSm120/{} has unsupported typed route metadata",
                    spec.symbol
                ));
            }
        };
        require_ptx_entry_tokens("TriadSm120", entry, &[core])?;
        reject_ptx_entry_tokens("TriadSm120", entry, &forbidden)?;
        let loads = match spec.op {
            super::contract::Sm120Op::Nn => [
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            ],
            super::contract::Sm120Op::Tn => [
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            ],
            super::contract::Sm120Op::Nt => [
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
            ],
        };
        require_ptx_entry_tokens("TriadSm120", entry, &loads)?;
        require_ptx_entry_opcode("TriadSm120", entry, "st.global opcode", |token| {
            token.starts_with("st.global.")
        })?;
    }
    for spec in super::contract::tf32_route_specs(ModuleKind::TriadSm120) {
        let entry = parsed_ptx_entry_ref(parsed, spec.symbol)?;
        require_ptx_entry_tokens("TriadSm120", entry, COMMON)?;
        if spec.route.is_exact_fma() {
            // The exact routes multiply in scalar FMA and never round an
            // operand: no tensor-core instruction and no TF32 conversion.
            require_ptx_entry_tokens("TriadSm120", entry, &["fma.rn.f32"])?;
            reject_ptx_entry_tokens(
                "TriadSm120",
                entry,
                &[BF16_CORE, F16_CORE, TF32_CORE, "cvt.rna.tf32.f32"],
            )?;
        } else {
            require_ptx_entry_tokens("TriadSm120", entry, &["cvt.rna.tf32.f32", TF32_CORE])?;
            reject_ptx_entry_tokens("TriadSm120", entry, &[BF16_CORE, F16_CORE])?;
        }
        require_ptx_entry_opcode("TriadSm120", entry, "st.global opcode", |token| {
            token.starts_with("st.global.")
        })?;
    }
    Ok(())
}

fn validate_sm120_ptx(arch: &str, ptx: &str) -> Result<(), String> {
    let expected = sm120_ptx_target(arch)
        .ok_or_else(|| format!("TriadSm120 has no generic target candidate for {arch}"))?;
    let actual = ptx_target(ptx)?;
    if actual != expected {
        return Err(format!(
            "TriadSm120 PTX target is {actual}, expected {expected}"
        ));
    }
    let mut expected: Vec<_> = super::contract::sm120_kernel_specs()
        .map(|spec| spec.symbol)
        .collect();
    expected.extend(super::contract::tf32_module_symbols(ModuleKind::TriadSm120));
    let parsed = validate_exact_ptx_exports("TriadSm120", 128, &expected, ptx)?;
    validate_sm120_entry_features(&parsed)?;
    let ptx = strip_ptx_comments(ptx)?;
    if ptx_has_unquoted_token(&ptx, |token| {
        token.starts_with("atom.")
            || token.starts_with("atom::")
            || token.starts_with("red.")
            || token.starts_with("red::")
            || token.starts_with("redux.")
            || token.starts_with("tcgen05.")
            || token.starts_with("wgmma.")
            || token.starts_with("setmaxnreg.")
            || token.contains(".multicast")
            || (token.starts_with("tcgen05.") && token.contains("cta_group::2"))
            || token.contains(".shared::cluster")
            || token.starts_with("multimem.")
            || token.starts_with("mapa.")
            || token.starts_with("clusterlaunchcontrol.")
            || token.starts_with("griddepcontrol.")
    }) {
        return Err("TriadSm120 PTX contains a forbidden instruction family".into());
    }
    if ptx_has_unquoted_token(&ptx, |token| {
        token == "call"
            || token.starts_with("call.")
            || token == ".callprototype"
            || token == ".calltargets"
    }) {
        return Err("TriadSm120 PTX contains a device call instruction".into());
    }
    for symbol in [
        "cudaLaunchDevice",
        "cudaGetParameterBuffer",
        "cudaDeviceSynchronize",
        "__cudaPushCallConfiguration",
        "__cudaPopCallConfiguration",
        "malloc",
        "free",
        "operator new",
        "operator delete",
    ] {
        if ptx_has_unquoted_token(&ptx, |token| token == symbol) {
            return Err(format!("TriadSm120 PTX contains forbidden symbol {symbol}"));
        }
    }
    Ok(())
}

fn validate_tf32_feature_instructions(
    module_kind: ModuleKind,
    extensions: bool,
    ptx: &str,
) -> Result<(), String> {
    use crate::mamba_ssm::gpu::kernel_identity::{
        ResolvedInstructionFamily, ResolvedOperandConversion,
    };

    let expected_contract = match module_kind {
        ModuleKind::TriadSm80 | ModuleKind::TriadSm89Finalist => (
            ResolvedInstructionFamily::MmaSync,
            ResolvedOperandConversion::RegisterCvtRnaTf32F32,
        ),
        ModuleKind::TriadSm90a => (
            ResolvedInstructionFamily::Wgmma,
            ResolvedOperandConversion::TensorMapTfloat32,
        ),
        ModuleKind::TriadSm100 => (
            ResolvedInstructionFamily::Tcgen05,
            ResolvedOperandConversion::TensorMapTfloat32,
        ),
        ModuleKind::TriadSm120 => (
            ResolvedInstructionFamily::MmaSync,
            ResolvedOperandConversion::TensorMapUint32ThenCvtRnaTf32F32,
        ),
        _ => return Ok(()),
    };
    // The portable module's wide extension tiles may round in registers by
    // the half-ulp add instead of cvt.rna.tf32.f32; every other route keeps
    // the module's one contract.
    let contract_allowed = |kernel_spec: &super::contract::Tf32KernelSpec| {
        let contract = (
            kernel_spec.instruction_family,
            kernel_spec.operand_conversion,
        );
        contract == expected_contract
            || (module_kind == ModuleKind::TriadSm80
                && contract
                    == (
                        ResolvedInstructionFamily::MmaSync,
                        ResolvedOperandConversion::RegisterAddHalfUlpTf32,
                    ))
    };
    if super::contract::tf32_route_specs_all(module_kind)
        .filter(|kernel_spec| !kernel_spec.route.is_exact_fma())
        .any(|kernel_spec| !contract_allowed(kernel_spec))
    {
        return Err(format!(
            "{module_kind:?} TF32 route metadata has the wrong conversion contract"
        ));
    }
    if super::contract::tf32_route_specs_all(module_kind)
        .filter(|kernel_spec| kernel_spec.route.is_exact_fma())
        .any(|kernel_spec| {
            (
                kernel_spec.instruction_family,
                kernel_spec.operand_conversion,
            ) != (
                ResolvedInstructionFamily::ScalarFma,
                ResolvedOperandConversion::None,
            )
        })
    {
        return Err(format!(
            "{module_kind:?} exact-F32 route metadata has the wrong conversion contract"
        ));
    }
    let required: &[&str] = match module_kind {
        ModuleKind::TriadSm80 | ModuleKind::TriadSm89Finalist | ModuleKind::TriadSm120 => &[
            "cvt.rna.tf32.f32",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        ],
        ModuleKind::TriadSm90a => &[
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
            "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32",
        ],
        ModuleKind::TriadSm100 => &[
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
            "tcgen05.mma.cta_group::1.kind::tf32",
        ],
        _ => &[],
    };
    const EXACT_REQUIRED: &[&str] = &[
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        "fma.rn.f32",
    ];
    const ADD_HALF_ULP_REQUIRED: &[&str] = &["mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"];
    let parsed = parse_ptx(ptx)?;
    for kernel_spec in super::contract::tf32_route_specs_for(module_kind, extensions) {
        let entry = &parsed_ptx_entry_ref(&parsed, kernel_spec.symbol)?.text;
        let add_half_ulp =
            kernel_spec.operand_conversion == ResolvedOperandConversion::RegisterAddHalfUlpTf32;
        let required = if kernel_spec.route.is_exact_fma() {
            EXACT_REQUIRED
        } else if add_half_ulp {
            ADD_HALF_ULP_REQUIRED
        } else {
            required
        };
        for instruction in required {
            if !ptx_has_unquoted_token(entry, |token| token == *instruction) {
                return Err(format!(
                    "{module_kind:?}/{} TF32 PTX is missing {instruction}",
                    kernel_spec.symbol
                ));
            }
        }
        if add_half_ulp && ptx_has_unquoted_token(entry, |token| token == "cvt.rna.tf32.f32") {
            return Err(format!(
                "{module_kind:?}/{} rounds by the half-ulp add and must not also convert by cvt.rna.tf32.f32",
                kernel_spec.symbol
            ));
        }
        if ptx_has_unquoted_token(entry, |token| {
            token.starts_with("atom.")
                || token.starts_with("atom::")
                || token.starts_with("red.")
                || token.starts_with("red::")
                || token.starts_with("redux.")
        }) {
            return Err(format!(
                "{module_kind:?}/{} TF32 PTX contains a numeric atomic or reduction",
                kernel_spec.symbol
            ));
        }
    }
    let ptx = strip_ptx_comments(ptx)?;
    if module_kind == ModuleKind::TriadSm90a
        && ptx_has_unquoted_token(&ptx, |token| token == "cvt.rna.tf32.f32")
    {
        return Err("TriadSm90a must convert TF32 through TFLOAT32 tensor maps".into());
    }
    if module_kind == ModuleKind::TriadSm100
        && ptx_has_unquoted_token(&ptx, |token| {
            token == "cvt.rna.tf32.f32" || token.starts_with("wgmma.")
        })
    {
        return Err("TriadSm100 contains a foreign TF32 instruction family".into());
    }
    if module_kind == ModuleKind::TriadSm120
        && ptx_has_unquoted_token(&ptx, |token| token.starts_with("tcgen05."))
    {
        return Err("TriadSm120 must not contain TCGEN05 instructions".into());
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct SourceFragment {
    logical_name: &'static str,
    source: &'static str,
    allowed_quoted_includes: &'static [&'static str],
}

const TYPED_PRELUDE: SourceFragment = SourceFragment {
    logical_name: "kernels/_typed_prelude.cuh",
    source: include_str!("../../../../kernels/_typed_prelude.cuh"),
    allowed_quoted_includes: &[],
};

// Each `logical_name` is the file's real path and becomes a compiler-visible
// `#line` boundary, so it is part of the source, compile-key, and artifact
// identities: moving a file re-pins every cohort that composes it.
const FIXED_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    SourceFragment {
        logical_name: "kernels/mamba_ssm.cu",
        source: include_str!("../../../../kernels/mamba_ssm.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/mamba_ssm_parallel.cu",
        source: include_str!("../../../../kernels/mamba_ssm_parallel.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/conv1d.cu",
        source: include_str!("../../../../kernels/conv1d.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/activations.cu",
        source: include_str!("../../../../kernels/activations.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/norms.cu",
        source: include_str!("../../../../kernels/norms.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/elementwise.cu",
        source: include_str!("../../../../kernels/elementwise.cu"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/loss_scaler.cu",
        source: include_str!("../../../../kernels/loss_scaler.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/grad_clip.cu",
        source: include_str!("../../../../kernels/grad_clip.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/adamw.cu",
        source: include_str!("../../../../kernels/adamw.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/common.cuh",
        source: include_str!("../../../../kernels/gemm_bi_inference/common.cuh"),
        allowed_quoted_includes: &["_typed_prelude.cuh"],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/ffma.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/ffma.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/tf32.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/tf32.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/sm120/tf32.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/sm120/tf32.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/sm120/tma.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/sm120/tma.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/wmma_legacy.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/wmma_legacy.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/matvec.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/matvec.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/mma16.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/mma16.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/tcw64.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/tcw64.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/sm90a/wgmma.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/sm90a/wgmma.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_inference/sm100/tcgen05.cu",
        source: include_str!("../../../../kernels/gemm_bi_inference/sm100/tcgen05.cu"),
        allowed_quoted_includes: &[],
    },
];

const FIXED_SM89_HALF_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/half_pipeline.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/half_pipeline.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_EXACT_N64_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/f32_n64_copyplan.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/f32_n64_copyplan.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_RNA_WIDE_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/tf32_rna_wide.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/tf32_rna_wide.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_HALF_SWIZZLE_LAYOUT_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_HALF_SWIZZLE_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/half_swizzle.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/half_swizzle.cu"),
    allowed_quoted_includes: &["sm89_half_swizzle_layout.cuh"],
};

const FIXED_SM89_HALF_S3_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/half_s3.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/half_s3.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_RNA_N96_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/tf32_rna_n96.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/tf32_rna_n96.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_HALF_N64_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/half_n64.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/half_n64.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_CELLS_COMMON_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/cells_common.cuh",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/cells_common.cuh"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_CELLS_F32_SIMT_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/cells_f32_simt.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/cells_f32_simt.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_CELLS_HALF_MMA_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/cells_half_mma.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/cells_half_mma.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM89_CELLS_TF32_MMA_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm80/cells_tf32_mma.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm80/cells_tf32_mma.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM120_EXACT_N64_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm120/f32_n64_copyplan.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm120/f32_n64_copyplan.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM120_SLICED_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm120/f32_n64_sliced.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm120/f32_n64_sliced.cu"),
    allowed_quoted_includes: &[],
};

const FIXED_SM120_POSTBIAS_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_inference/sm120/f32_postbias.cu",
    source: include_str!("../../../../kernels/gemm_bi_inference/sm120/f32_postbias.cu"),
    allowed_quoted_includes: &[],
};

const TRIAD_CONTRACT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_triad/contract.cuh",
    source: include_str!("../../../../kernels/gemm_bi_triad/contract.cuh"),
    allowed_quoted_includes: &[],
};

const TRIAD_COMMON: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_triad/common.cuh",
    source: include_str!("../../../../kernels/gemm_bi_triad/common.cuh"),
    allowed_quoted_includes: &[],
};

const TRIAD_EPILOGUE: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_triad/epilogue.cuh",
    source: include_str!("../../../../kernels/gemm_bi_triad/epilogue.cuh"),
    allowed_quoted_includes: &[],
};

const SCALAR_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    TRIAD_CONTRACT,
    TRIAD_COMMON,
    TRIAD_EPILOGUE,
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/scalar.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/scalar.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/scalar_nn_m64n64.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/scalar_nn_m64n64.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/scalar_nn_splitk_m32n64.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/scalar_nn_splitk_m32n64.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/scalar_nt_d768_transpose.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/scalar_nt_d768_transpose.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/scalar_nt_m2n16.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/scalar_nt_m2n16.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/scalar_tn_m16n16.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/scalar_tn_m16n16.cu"),
        allowed_quoted_includes: &[],
    },
];

const SM80_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    TRIAD_CONTRACT,
    TRIAD_COMMON,
    TRIAD_EPILOGUE,
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/mma16.cuh",
        source: include_str!("../../../../kernels/gemm_bi_triad/mma16.cuh"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/sm80/mma.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/sm80/mma.cu"),
        allowed_quoted_includes: &[],
    },
];

/// The tc64 TN dW stream-K twin. Composed after the portable fragments for
/// every sm80-family target except CC 12.x (see
/// [`sm80_target_composes_streamk`]).
const SM80_STREAMK_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_triad/sm80/streamk.cu",
    source: include_str!("../../../../kernels/gemm_bi_triad/sm80/streamk.cu"),
    allowed_quoted_includes: &[],
};

/// The wide deterministic TF32 tile (NN, 128 x 128, eight computing
/// warps), composed with the stream-K fragment on the same targets.
const SM80_TF32_WIDE_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_triad/sm80/tf32_wide.cu",
    source: include_str!("../../../../kernels/gemm_bi_triad/sm80/tf32_wide.cu"),
    allowed_quoted_includes: &[],
};

/// The TN split-K family (eight partitions on the m64n64 and m32n32 tiles),
/// composed with the other two extension fragments on the same targets.
const SM80_TN_SPLITK_SOURCE_FRAGMENT: SourceFragment = SourceFragment {
    logical_name: "kernels/gemm_bi_triad/sm80/tn_splitk.cu",
    source: include_str!("../../../../kernels/gemm_bi_triad/sm80/tn_splitk.cu"),
    allowed_quoted_includes: &[],
};

/// Whether the portable module compiled for `arch` carries the extension
/// fragments (the tc64 TN stream-K kernels, the wide TF32 tile and the TN
/// split-K family). CC 12.x
/// boards run the SM120 kernels, and leaving the fragments out keeps their
/// portable module byte-identical to the one their TF32 cohort's portable
/// twin was frozen against.
pub(super) fn sm80_target_composes_streamk(arch: &str) -> bool {
    sm80_ptx_target(arch).is_some() && super::contract::sm80_target_composes_extensions(arch)
}

/// Whether the module of `kind` compiled for `arch` composes the extension
/// fragments: only the portable module, and only on sm80-family targets.
fn module_composes_extensions(kind: ModuleKind, arch: &str) -> bool {
    kind == ModuleKind::TriadSm80 && sm80_target_composes_streamk(arch)
}

const SM90A_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    TRIAD_CONTRACT,
    TRIAD_COMMON,
    TRIAD_EPILOGUE,
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/sm90a/wgmma.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/sm90a/wgmma.cu"),
        allowed_quoted_includes: &[],
    },
];

const SM100_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    TRIAD_CONTRACT,
    TRIAD_COMMON,
    TRIAD_EPILOGUE,
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/sm100/tcgen05.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/sm100/tcgen05.cu"),
        allowed_quoted_includes: &[],
    },
];

const SM120_SOURCE_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    TRIAD_CONTRACT,
    TRIAD_COMMON,
    TRIAD_EPILOGUE,
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/sm120/tma.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/sm120/tma.cu"),
        allowed_quoted_includes: &[],
    },
    SourceFragment {
        logical_name: "kernels/gemm_bi_triad/sm120/exact.cu",
        source: include_str!("../../../../kernels/gemm_bi_triad/sm120/exact.cu"),
        allowed_quoted_includes: &[],
    },
];

pub(super) const SCALAR_ZERO_REDUCTION_SYMBOLS: &[&str] = &[
    "nn_zero_reduction",
    "tn_zero_reduction",
    "nt_zero_reduction",
];

pub(super) const SCALAR_TN_NARROW_SPLITM_PARTIAL_SYMBOLS: &[&str] = &[
    "tn_narrow_splitm_partial",
    "tn_narrow_splitm_partial_aligned",
];

pub(super) const SCALAR_TN_SPLITM_PARTIAL_SYMBOLS: &[&str] =
    &["tn_splitm_partial", "tn_splitm_partial_aligned"];

pub(super) const SCALAR_SYMBOLS: &[&str] = &[
    "nn_big",
    "nn_m64n64_bk16_s2",
    "nn_splitk32_m32n64_exact",
    "nn_prism_m64n64_bk16_s2",
    "nn_zero_reduction",
    "tn_big",
    "tn_aligned",
    "tn_zero_reduction",
    "tn_narrow_splitm_partial",
    "tn_narrow_splitm_partial_aligned",
    "tn_splitm_partial",
    "tn_splitm_partial_aligned",
    "tn_m16n16_bk16_s2_splitm16",
    "splitm_reduce",
    "nt_big",
    "nt_m2n16_bk64_splitk32",
    "nt_zero_reduction",
    "nn_slim",
    "nn_splitk_slim_partial",
    "tn_slim",
    "nt_slim",
    "nn_ultra_thin",
    "nn_gemv",
    "tn_gemv",
    "nt_gemv",
    "nn_narrow",
    "nn_narrow_small",
    "tn_narrow",
    "nt_narrow",
    "nn_splitk32_partial",
    "splitk_reduce",
    "dx_col_gemv",
    "transpose_f32_2d",
    "transpose_f32_32x16_d768",
    "nn_gemv_bf16",
    "nn_gemv_f16",
    "tn_gemv_bf16",
    "tn_gemv_f16",
    "nt_gemv_bf16",
    "nt_gemv_f16",
    "nn_ultra_thin_bf16",
    "nn_ultra_thin_f16",
    "nn_narrow_bf16",
    "nn_narrow_f16",
    "nn_narrow_small_bf16",
    "nn_narrow_small_f16",
    "tn_narrow_bf16",
    "tn_narrow_f16",
    "nt_narrow_bf16",
    "nt_narrow_f16",
    "nn_big_bf16",
    "nn_big_f16",
    "tn_big_bf16",
    "tn_big_f16",
    "nt_big_bf16",
    "nt_big_f16",
];

pub(super) const SM80_SYMBOLS: &[&str] = &[
    "nn_tc_bf16",
    "nn_tc_f16",
    "tn_tc_bf16",
    "tn_tc_f16",
    "nt_tc_bf16",
    "nt_tc_f16",
    "nn_tc64_bf16",
    "nn_tc64_f16",
    "nn_tc16_bf16",
    "nn_tc16_f16",
    "tn_tc64_bf16",
    "tn_tc64_f16",
    "tn_tc128x64_bf16",
    "tn_tc128x64_f16",
    "nt_tc64_bf16",
    "nt_tc64_f16",
];

/// Exports of the stream-K fragment; present only where
/// [`sm80_target_composes_streamk`] holds.
pub(super) const SM80_STREAMK_SYMBOLS: &[&str] = &["tn_tc64_streamk_bf16", "tn_tc64_streamk_f16"];

pub const SM90A_SYMBOLS: &[&str] = &[
    "nn_sm90a_wgmma_wg1_bf16",
    "nn_sm90a_wgmma_wg1_f16",
    "tn_sm90a_wgmma_wg1_bf16",
    "tn_sm90a_wgmma_wg1_f16",
    "nt_sm90a_wgmma_wg1_bf16",
    "nt_sm90a_wgmma_wg1_f16",
    "nn_sm90a_wgmma_wg2_bf16",
    "nn_sm90a_wgmma_wg2_f16",
    "tn_sm90a_wgmma_wg2_bf16",
    "tn_sm90a_wgmma_wg2_f16",
    "nt_sm90a_wgmma_wg2_bf16",
    "nt_sm90a_wgmma_wg2_f16",
];

/// The Ada-measured inference cells: their own module, composed for every
/// board the Fixed overlay serves, so their artifact identity is their own.
const INFERENCE_SM89_CELLS_FRAGMENTS: &[SourceFragment] = &[
    TYPED_PRELUDE,
    FIXED_SM89_CELLS_COMMON_FRAGMENT,
    FIXED_SM89_CELLS_F32_SIMT_FRAGMENT,
    FIXED_SM89_CELLS_HALF_MMA_FRAGMENT,
    FIXED_SM89_CELLS_TF32_MMA_FRAGMENT,
];

fn module_fragments(kind: ModuleKind) -> Result<&'static [SourceFragment], String> {
    match kind {
        ModuleKind::Fixed => Ok(FIXED_SOURCE_FRAGMENTS),
        ModuleKind::InferenceSm89Cells => Ok(INFERENCE_SM89_CELLS_FRAGMENTS),
        ModuleKind::TriadScalar => Ok(SCALAR_SOURCE_FRAGMENTS),
        ModuleKind::TriadSm80 => Ok(SM80_SOURCE_FRAGMENTS),
        ModuleKind::TriadSm90a => Ok(SM90A_SOURCE_FRAGMENTS),
        ModuleKind::TriadSm100 => Ok(SM100_SOURCE_FRAGMENTS),
        ModuleKind::TriadSm120 => Ok(SM120_SOURCE_FRAGMENTS),
        _ => Err(format!("no source fragments for {kind:?}")),
    }
}

/// The composed source the compiler sees for `kind` on `arch`: the module's
/// frozen base and only that target's admitted extension fragments.
fn compose_module_source_for(kind: ModuleKind, arch: &str) -> Result<String, String> {
    if kind == ModuleKind::TriadSm89Finalist {
        validate_module_target(kind, arch)?;
        return super::sm89_finalist_source::compose_sm89_finalist_source();
    }
    if kind == ModuleKind::TriadSm89Half {
        validate_module_target(kind, arch)?;
        return super::sm89_half_source::compose_sm89_half_source();
    }
    if kind == ModuleKind::TriadSm89ExactF32 {
        validate_module_target(kind, arch)?;
        if FramedSha256::bytes(super::sm89_exact_f32_source::OWNER_TEMPLATE.as_bytes())
            != super::sm89_exact_f32_source::OWNER_SHA256_BYTES
        {
            return Err("TriadSm89ExactF32 CUDA owner SHA-256 changed".into());
        }
        super::sm89_exact_f32_source::validate_source()?;
        return super::sm89_exact_f32_source::compose_source();
    }
    if kind == ModuleKind::TriadSm89ExactF32D128 {
        if sm80_ptx_target(arch).is_none() {
            return Err(format!(
                "TriadSm89ExactF32D128 requires an admitted SM80+ portable target, got {arch}"
            ));
        }
        if FramedSha256::bytes(super::sm89_exact_f32_d128_source::OWNER_TEMPLATE.as_bytes())
            != super::sm89_exact_f32_d128_source::OWNER_SHA256_BYTES
        {
            return Err("TriadSm89ExactF32D128 CUDA owner SHA-256 changed".into());
        }
        super::sm89_exact_f32_d128_source::validate_source()?;
        return super::sm89_exact_f32_d128_source::compose_source();
    }
    if kind == ModuleKind::TriadSm89Tf32Joint {
        validate_module_target(kind, arch)?;
        let owner_digest = FramedSha256::bytes(super::sm89_tf32_joint_source::SOURCE.as_bytes());
        if crate::mamba_ssm::gpu::kernel_identity::digest_hex(&owner_digest)
            != super::sm89_tf32_joint_source::SOURCE_SHA256
        {
            return Err("TriadSm89Tf32Joint CUDA owner SHA-256 changed".into());
        }
        let primitive_digest =
            FramedSha256::bytes(super::sm89_tf32_joint_source::PRIMITIVES.as_bytes());
        if crate::mamba_ssm::gpu::kernel_identity::digest_hex(&primitive_digest)
            != super::sm89_tf32_joint_source::PRIMITIVES_SHA256
        {
            return Err("TriadSm89Tf32Joint primitive SHA-256 changed".into());
        }
        for (name, source, digest) in super::sm89_tf32_joint_source::WIDE_FRAGMENTS {
            let observed = crate::mamba_ssm::gpu::kernel_identity::digest_hex(
                &FramedSha256::bytes(source.as_bytes()),
            );
            if observed != digest {
                return Err(format!(
                    "TriadSm89Tf32Joint {name} fragment SHA-256 changed: {observed}"
                ));
            }
        }
        super::sm89_tf32_joint_source::validate_source()?;
        return super::sm89_tf32_joint_source::compose_source();
    }
    let base = module_fragments(kind)?;
    if kind == ModuleKind::Fixed && fixed_sm89_half_composed(arch) {
        let mut fragments = base.to_vec();
        fragments.push(FIXED_SM89_HALF_SOURCE_FRAGMENT);
        fragments.push(FIXED_SM89_EXACT_N64_SOURCE_FRAGMENT);
        fragments.push(FIXED_SM89_RNA_WIDE_SOURCE_FRAGMENT);
        fragments.push(FIXED_SM89_HALF_SWIZZLE_LAYOUT_FRAGMENT);
        fragments.push(FIXED_SM89_HALF_SWIZZLE_SOURCE_FRAGMENT);
        fragments.push(FIXED_SM89_HALF_S3_SOURCE_FRAGMENT);
        fragments.push(FIXED_SM89_RNA_N96_SOURCE_FRAGMENT);
        fragments.push(FIXED_SM89_HALF_N64_SOURCE_FRAGMENT);
        return compose_fragments(&fragments);
    }
    if kind == ModuleKind::Fixed && arch == "compute_120" {
        let mut fragments = base.to_vec();
        fragments.push(FIXED_SM120_EXACT_N64_SOURCE_FRAGMENT);
        fragments.push(FIXED_SM120_SLICED_SOURCE_FRAGMENT);
        fragments.push(FIXED_SM120_POSTBIAS_SOURCE_FRAGMENT);
        return compose_fragments(&fragments);
    }
    if kind == ModuleKind::TriadSm80 && sm80_target_composes_streamk(arch) {
        let mut fragments = base.to_vec();
        fragments.push(SM80_STREAMK_SOURCE_FRAGMENT);
        fragments.push(SM80_TF32_WIDE_SOURCE_FRAGMENT);
        fragments.push(SM80_TN_SPLITK_SOURCE_FRAGMENT);
        return compose_fragments(&fragments);
    }
    compose_fragments(base)
}

fn compose_compile_module_source(
    kind: ModuleKind,
    device_cc: Option<(i32, i32)>,
    arch: &str,
    state_cap: usize,
    nvrtc: (i32, i32),
) -> Result<String, String> {
    let mut combined = compose_module_source_for(kind, arch)?;
    if kind == ModuleKind::Fixed {
        combined = super::super::fold_transport::compose_fixed_source(
            combined, device_cc, arch, state_cap, nvrtc,
        )?;
        combined = crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compose_fixed_source(
            combined, device_cc, arch, state_cap, nvrtc,
        )?;
    }
    Ok(combined)
}

/// The fullest composition of `kind`: every fragment any target composes.
/// Source scans read this one; the compiler takes the per-target form.
#[cfg(test)]
fn compose_module_source(kind: ModuleKind) -> Result<String, String> {
    compose_module_source_for(
        kind,
        if kind == ModuleKind::Fixed {
            "sm_89"
        } else {
            "sm_80"
        },
    )
}

/// Digest of the target-specific base composition. Non-Fixed modules compile
/// these exact bytes; Fixed compilation additionally applies its two overlays.
#[cfg(test)]
pub(super) fn module_source_digest(kind: ModuleKind, arch: &str) -> Result<[u8; 32], String> {
    Ok(FramedSha256::bytes(
        compose_module_source_for(kind, arch)?.as_bytes(),
    ))
}

/// Digest the exact post-overlay bytes supplied to NVRTC, without a CUDA
/// context. This delegates to the same composer as `compile_module`.
#[cfg(test)]
pub(super) fn module_source_digest_for_compile(
    kind: ModuleKind,
    device_cc: Option<(i32, i32)>,
    arch: &str,
    state_cap: usize,
    nvrtc: (i32, i32),
) -> Result<[u8; 32], String> {
    Ok(FramedSha256::bytes(
        compose_compile_module_source(kind, device_cc, arch, state_cap, nvrtc)?.as_bytes(),
    ))
}

fn compose_fragments(fragments: &[SourceFragment]) -> Result<String, String> {
    let mut composed = String::new();
    for fragment in fragments {
        validate_fragment(fragment)?;
        composed.push_str("#line 1 \"");
        composed.push_str(fragment.logical_name);
        composed.push_str("\"\n");
        append_without_local_includes(&mut composed, fragment)?;
        if !composed.ends_with('\n') {
            composed.push('\n');
        }
    }
    Ok(composed)
}

fn validate_fragment(fragment: &SourceFragment) -> Result<(), String> {
    if fragment.logical_name.contains('"') || fragment.logical_name.contains('\n') {
        return Err(format!(
            "invalid logical source name {}",
            fragment.logical_name
        ));
    }
    if fragment.logical_name.starts_with("kernels/gemm_bi_triad/")
        && fragment.logical_name.ends_with(".cuh")
        && fragment.source.contains("extern \"C\" __global__")
    {
        return Err(format!(
            "triad header {} exports a global kernel",
            fragment.logical_name
        ));
    }
    Ok(())
}

fn append_without_local_includes(
    output: &mut String,
    fragment: &SourceFragment,
) -> Result<(), String> {
    let mut physical = fragment.source.split_inclusive('\n').peekable();
    while let Some(first) = physical.next() {
        let mut raw = first.to_owned();
        let mut logical = first.strip_suffix('\n').unwrap_or(first).to_owned();
        while logical.ends_with('\\') {
            logical.pop();
            let Some(next) = physical.next() else {
                return Err(format!(
                    "{} ends inside a preprocessor continuation",
                    fragment.logical_name
                ));
            };
            raw.push_str(next);
            logical.push_str(next.strip_suffix('\n').unwrap_or(next));
        }

        match include_directive(&logical).map_err(|reason| {
            format!(
                "{} contains invalid include directive: {reason}",
                fragment.logical_name
            )
        })? {
            Some(IncludeDirective::Quoted(target)) => {
                if target.ends_with(".cu") {
                    return Err(format!(
                        "{} includes forbidden CUDA source {target}",
                        fragment.logical_name
                    ));
                }
                if !fragment.allowed_quoted_includes.contains(&target.as_str()) {
                    return Err(format!(
                        "{} contains unlisted quoted include {target}",
                        fragment.logical_name
                    ));
                }
            }
            Some(IncludeDirective::Angle(target)) => {
                if target.ends_with(".cu") {
                    return Err(format!(
                        "{} includes forbidden CUDA source {target}",
                        fragment.logical_name
                    ));
                }
                output.push_str(&raw);
            }
            None => output.push_str(&raw),
        }
    }
    Ok(())
}

enum IncludeDirective {
    Quoted(String),
    Angle(String),
}

fn include_directive(logical_line: &str) -> Result<Option<IncludeDirective>, &'static str> {
    let bytes = logical_line.as_bytes();
    let Some(mut cursor) = skip_space_and_comments(bytes, 0) else {
        return Ok(None);
    };
    if bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    } else if bytes.get(cursor..cursor + 2) == Some(b"%:") {
        cursor += 2;
    } else {
        return Ok(None);
    }
    cursor = skip_space_and_comments(bytes, cursor).ok_or("unterminated comment after '#'")?;
    let start = cursor;
    while bytes.get(cursor).is_some_and(u8::is_ascii_alphabetic) {
        cursor += 1;
    }
    let directive = &bytes[start..cursor];
    if directive == b"import" {
        return Err("include-like import directive is forbidden");
    }
    if directive != b"include" {
        return Ok(None);
    }
    cursor = skip_space_and_comments(bytes, cursor).ok_or("unterminated comment after include")?;
    match bytes.get(cursor) {
        Some(b'"') => {
            cursor += 1;
            let target_start = cursor;
            while let Some(byte) = bytes.get(cursor) {
                match byte {
                    b'"' => {
                        if !include_trailing_is_trivia(bytes, cursor + 1) {
                            return Err("trailing tokens after quoted include target");
                        }
                        let target = String::from_utf8(bytes[target_start..cursor].to_vec())
                            .map_err(|_| "quoted include target is not UTF-8")?;
                        return Ok(Some(IncludeDirective::Quoted(target)));
                    }
                    b'\\' => return Err("escaped quoted include target"),
                    _ => cursor += 1,
                }
            }
            Err("unterminated quoted include target")
        }
        Some(b'<') => {
            cursor += 1;
            let target_start = cursor;
            while let Some(byte) = bytes.get(cursor) {
                if *byte == b'>' {
                    if !include_trailing_is_trivia(bytes, cursor + 1) {
                        return Err("trailing tokens after angle include target");
                    }
                    let target = String::from_utf8(bytes[target_start..cursor].to_vec())
                        .map_err(|_| "angle include target is not UTF-8")?;
                    return Ok(Some(IncludeDirective::Angle(target)));
                }
                cursor += 1;
            }
            Err("unterminated angle include target")
        }
        Some(_) => Err("macro include target"),
        None => Err("missing include target"),
    }
}

fn include_trailing_is_trivia(bytes: &[u8], mut cursor: usize) -> bool {
    loop {
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        if cursor == bytes.len() || bytes.get(cursor..cursor + 2) == Some(b"//") {
            return true;
        }
        if bytes.get(cursor..cursor + 2) != Some(b"/*") {
            return false;
        }
        let Some(remainder) = bytes.get(cursor + 2..) else {
            return false;
        };
        let Some(end) = remainder.windows(2).position(|window| window == b"*/") else {
            return false;
        };
        cursor += end + 4;
    }
}

fn skip_space_and_comments(bytes: &[u8], mut cursor: usize) -> Option<usize> {
    loop {
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        if bytes.get(cursor..cursor + 2) != Some(b"/*") {
            return Some(cursor);
        }
        let remainder = bytes.get(cursor + 2..)?;
        let end = remainder.windows(2).position(|window| window == b"*/")?;
        cursor += end + 4;
    }
}

fn resolve_owned_symbol<T>(
    symbol: &str,
    scalar_get: impl FnOnce(&str) -> Result<T, String>,
    sm80_get: impl FnOnce(&str) -> Result<T, String>,
) -> Result<T, String> {
    if SCALAR_SYMBOLS.contains(&symbol) {
        scalar_get(symbol).map_err(|error| format!("TriadScalar symbol {symbol}: {error}"))
    } else if SM80_SYMBOLS.contains(&symbol) || SM80_STREAMK_SYMBOLS.contains(&symbol) {
        sm80_get(symbol).map_err(|error| format!("TriadSm80 symbol {symbol}: {error}"))
    } else {
        Err(format!("no triad module owns symbol {symbol}"))
    }
}

/// Loaded deterministic training GEMM triad.
///
/// Scalar and Tensor Core handles remain in separate CUDA modules. Keeping the
/// modules and their identities beside the handles makes a route impossible to
/// outlive, or be replayed against, a different compiled artifact.
type Sm90aMapCacheKey = (
    [super::contract::Sm90aTensorMapKey; 2],
    [super::contract::Sm90aAllocationIdentity; 2],
    super::contract::Sm90aOp,
    u8,
    super::contract::Sm90aShape,
);
type Sm90aMapCache = Mutex<HashMap<Sm90aMapCacheKey, super::contract::Sm90aPreparedTensorMaps>>;
type Sm100MapCacheKey = (
    [super::contract::Sm90aTensorMapKey; 2],
    [super::contract::Sm90aAllocationIdentity; 2],
    super::contract::Sm100TensorOrigins,
    super::contract::Sm100Op,
    u8,
    super::contract::Sm100Tile,
    super::contract::Sm100Shape,
);
type Sm100MapCache = Mutex<HashMap<Sm100MapCacheKey, super::contract::Sm100PreparedTensorMaps>>;
type Sm120MapCacheKey = (
    [super::contract::Sm120TensorMapKey; 2],
    [super::contract::Sm90aAllocationIdentity; 2],
    super::contract::Sm120TensorOrigins,
    super::contract::Sm120Op,
    u8,
    super::contract::Sm120Tile,
    super::contract::Sm120Bk,
    super::contract::Sm120Shape,
);
type Sm120MapCache = Mutex<HashMap<Sm120MapCacheKey, super::contract::Sm120PreparedTensorMaps>>;
type Tf32MapCacheKey = (
    [super::contract::Tf32TensorMapKey; 2],
    [super::contract::Sm90aAllocationIdentity; 2],
    super::contract::F32TriadRequest,
    super::contract::Tf32PhysicalRoute,
);
type Tf32MapCache = Mutex<HashMap<Tf32MapCacheKey, super::contract::F32PreparedTensorMaps>>;

pub(crate) fn qualify_specialized_module(
    module: CompiledModule,
) -> Result<QualifiedSpecializedModule, String> {
    let functions = match module.artifact_identity.module_kind {
        ModuleKind::TriadSm90a => load_sm90a_functions(&module),
        ModuleKind::TriadSm100 => load_sm100_functions(&module),
        ModuleKind::TriadSm120 => load_sm120_functions(&module),
        kind => Err(format!("unsupported specialized triad module {kind:?}")),
    }?;
    let (tf32_functions, tf32_excluded, tf32_rejection) = if module.tf32_qualified {
        match load_tf32_functions(&module) {
            Ok((functions, excluded)) => (functions, excluded, None),
            Err(error) => (HashMap::new(), Vec::new(), Some(error)),
        }
    } else {
        (
            HashMap::new(),
            Vec::new(),
            module.tf32_qualification_error.clone(),
        )
    };
    Ok(QualifiedSpecializedModule {
        module,
        functions,
        tf32_functions,
        tf32_rejection,
        tf32_excluded,
        sm120_target: None,
        sm120_device_caps: None,
        sm120_resources: HashMap::new(),
    })
}

fn retain_sm89_half_symbol<T>(
    functions: &mut HashMap<&'static str, T>,
    exclusions: &mut Vec<Tf32SymbolExclusion>,
    symbol: &'static str,
    loaded: Result<T, String>,
) -> Result<(), String> {
    match loaded {
        Ok(function) => {
            if functions.insert(symbol, function).is_some() {
                return Err(format!("duplicate TriadSm89Half function {symbol}"));
            }
        }
        Err(reason) => exclusions.push(Tf32SymbolExclusion { symbol, reason }),
    }
    Ok(())
}

fn sm89_half_abi_for_symbol<'a>(
    census: &'a BTreeMap<&'static str, Result<Tf32DriverAbi, String>>,
    symbol: &str,
) -> Result<&'a Tf32DriverAbi, String> {
    census
        .get(symbol)
        .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?
        .as_ref()
        .map_err(Clone::clone)
}

fn sm89_half_runtime_entry<'a, T>(
    functions: &'a HashMap<&'static str, T>,
    module_available: bool,
    symbol: &str,
) -> Option<&'a T> {
    super::sm89_half_source::runtime_kernel_spec(symbol)?;
    module_available.then_some(())?;
    functions.get(symbol)
}

fn load_sm89_half_functions(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> Result<
    (
        HashMap<&'static str, CudaFunction>,
        Vec<Tf32SymbolExclusion>,
    ),
    String,
> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm89Half
        || sm80_ptx_target(module.compiler_identity.target.as_str()).is_none()
        || ctx
            .compute_capability()
            .map_err(|error| format!("query TriadSm89Half CC: {error:?}"))?
            .0
            < 8
    {
        return Err("TriadSm89Half requires an SM80+ portable binding".into());
    }
    let optin_shared = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .map_err(|error| format!("query TriadSm89Half opt-in shared memory: {error:?}"))?;
    let optin_shared = u32::try_from(optin_shared)
        .map_err(|_| format!("negative TriadSm89Half opt-in shared memory {optin_shared}"))?;
    let abi = module.sm89_half_driver_abi.as_ref().map_err(Clone::clone)?;
    let mut functions = HashMap::new();
    let mut exclusions = Vec::new();
    for spec in super::sm89_half_source::runtime_kernel_specs() {
        let loaded = (|| {
            validate_sm89_half_driver_abi(&spec, sm89_half_abi_for_symbol(abi, spec.symbol)?)?;
            let total_shared_bytes = spec
                .static_shared_bytes
                .checked_add(spec.dynamic_shared_bytes)
                .ok_or_else(|| format!("{} shared memory overflows u32", spec.symbol))?;
            if optin_shared < total_shared_bytes {
                return Err(format!(
                    "{} needs {total_shared_bytes} total shared bytes, device exposes {optin_shared}",
                    spec.symbol
                ));
            }
            let function = load_function(&module.module, ModuleKind::TriadSm89Half, spec.symbol)?;
            set_dynamic_shared(
                &function,
                spec.symbol,
                i32::try_from(spec.dynamic_shared_bytes)
                    .map_err(|_| format!("{} shared memory exceeds i32::MAX", spec.symbol))?,
            )?;
            let local_bytes = u32::try_from(
                function
                    .local_size_bytes()
                    .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?,
            )
            .map_err(|_| format!("{} returned negative local memory", spec.symbol))?;
            let registers = u32::try_from(
                function
                    .num_regs()
                    .map_err(|error| format!("query {} registers: {error:?}", spec.symbol))?,
            )
            .map_err(|_| format!("{} returned negative register count", spec.symbol))?;
            let static_shared_bytes =
                u32::try_from(function.shared_size_bytes().map_err(|error| {
                    format!("query {} static shared memory: {error:?}", spec.symbol)
                })?)
                .map_err(|_| format!("{} returned negative static shared memory", spec.symbol))?;
            if static_shared_bytes != spec.static_shared_bytes {
                return Err(format!(
                    "{} uses {static_shared_bytes} static shared bytes, expected {}",
                    spec.symbol, spec.static_shared_bytes
                ));
            }
            let max_threads = function
                .max_threads_per_block()
                .map_err(|error| format!("query {} max threads: {error:?}", spec.symbol))?;
            tf32_symbol_admission(
                spec.symbol,
                local_bytes,
                0,
                registers,
                spec.register_cap,
                max_threads,
                i32::try_from(spec.threads)
                    .map_err(|_| format!("{} thread count exceeds i32::MAX", spec.symbol))?,
            )?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    spec.threads,
                    spec.dynamic_shared_bytes as usize,
                    None,
                )
                .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
            if occupancy < spec.occupancy_gate {
                return Err(format!(
                    "{} occupancy {occupancy} misses its {}-CTA gate",
                    spec.symbol, spec.occupancy_gate
                ));
            }
            Ok(function)
        })();
        retain_sm89_half_symbol(&mut functions, &mut exclusions, spec.symbol, loaded)?;
    }
    if functions.len() + exclusions.len() != super::sm89_half_source::runtime_kernel_specs().count()
    {
        return Err("TriadSm89Half lost a symbol while applying resource gates".into());
    }
    Ok((functions, exclusions))
}

fn retain_sm89_exact_f32_symbol<T>(
    functions: &mut HashMap<&'static str, T>,
    exclusions: &mut Vec<Tf32SymbolExclusion>,
    symbol: &'static str,
    loaded: Result<T, String>,
) -> Result<(), String> {
    match loaded {
        Ok(function) => {
            if functions.insert(symbol, function).is_some() {
                return Err(format!("duplicate TriadSm89ExactF32 function {symbol}"));
            }
        }
        Err(reason) => exclusions.push(Tf32SymbolExclusion { symbol, reason }),
    }
    Ok(())
}

fn sm89_exact_f32_abi_for_symbol<'a>(
    census: &'a BTreeMap<&'static str, Result<Tf32DriverAbi, String>>,
    symbol: &str,
) -> Result<&'a Tf32DriverAbi, String> {
    census
        .get(symbol)
        .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?
        .as_ref()
        .map_err(Clone::clone)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sm89ExactF32ResourceFacts {
    local_bytes: u32,
    registers: u32,
    static_shared_bytes: u32,
    max_threads: i32,
    occupancy: u32,
}

fn validate_sm89_exact_f32_resources(
    spec: &super::sm89_exact_f32_source::Sm89ExactF32KernelSpec,
    facts: Sm89ExactF32ResourceFacts,
) -> Result<(), String> {
    if facts.static_shared_bytes != spec.static_shared_bytes {
        return Err(format!(
            "{} uses {} static shared bytes, expected {}",
            spec.symbol, facts.static_shared_bytes, spec.static_shared_bytes
        ));
    }
    tf32_symbol_admission(
        spec.symbol,
        facts.local_bytes,
        0,
        facts.registers,
        spec.register_cap,
        facts.max_threads,
        i32::try_from(spec.block.0)
            .map_err(|_| format!("{} thread count exceeds i32::MAX", spec.symbol))?,
    )?;
    if facts.occupancy < spec.occupancy_gate {
        return Err(format!(
            "{} occupancy {} misses its {}-CTA gate",
            spec.symbol, facts.occupancy, spec.occupancy_gate
        ));
    }
    Ok(())
}

fn load_sm89_exact_f32_functions(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> Result<
    (
        HashMap<&'static str, CudaFunction>,
        Vec<Tf32SymbolExclusion>,
    ),
    String,
> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm89ExactF32
        || sm80_ptx_target(module.compiler_identity.target.as_str()).is_none()
        || ctx
            .compute_capability()
            .map_err(|error| format!("query TriadSm89ExactF32 CC: {error:?}"))?
            .0
            < 8
    {
        return Err("TriadSm89ExactF32 requires an SM80+ portable binding".into());
    }
    let abi = module
        .sm89_exact_f32_driver_abi
        .as_ref()
        .map_err(Clone::clone)?;
    let mut functions = HashMap::new();
    let mut exclusions = Vec::new();
    for spec in &super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS {
        let loaded = (|| {
            validate_sm89_exact_f32_driver_abi(
                spec,
                sm89_exact_f32_abi_for_symbol(abi, spec.symbol)?,
            )?;
            let function =
                load_function(&module.module, ModuleKind::TriadSm89ExactF32, spec.symbol)?;
            let local_bytes = u32::try_from(
                function
                    .local_size_bytes()
                    .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?,
            )
            .map_err(|_| format!("{} returned negative local memory", spec.symbol))?;
            let registers = u32::try_from(
                function
                    .num_regs()
                    .map_err(|error| format!("query {} registers: {error:?}", spec.symbol))?,
            )
            .map_err(|_| format!("{} returned negative register count", spec.symbol))?;
            let static_shared_bytes =
                u32::try_from(function.shared_size_bytes().map_err(|error| {
                    format!("query {} static shared memory: {error:?}", spec.symbol)
                })?)
                .map_err(|_| format!("{} returned negative static shared memory", spec.symbol))?;
            let max_threads = function
                .max_threads_per_block()
                .map_err(|error| format!("query {} max threads: {error:?}", spec.symbol))?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    spec.block.0,
                    spec.dynamic_shared_bytes as usize,
                    None,
                )
                .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
            validate_sm89_exact_f32_resources(
                spec,
                Sm89ExactF32ResourceFacts {
                    local_bytes,
                    registers,
                    static_shared_bytes,
                    max_threads,
                    occupancy,
                },
            )?;
            Ok(function)
        })();
        retain_sm89_exact_f32_symbol(&mut functions, &mut exclusions, spec.symbol, loaded)?;
    }
    if functions.len() + exclusions.len()
        != super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS.len()
    {
        return Err("TriadSm89ExactF32 lost a symbol while applying resource gates".into());
    }
    Ok((functions, exclusions))
}

fn retain_sm89_exact_f32_d128_symbol<T>(
    functions: &mut HashMap<&'static str, T>,
    exclusions: &mut Vec<Tf32SymbolExclusion>,
    symbol: &'static str,
    loaded: Result<T, String>,
) -> Result<(), String> {
    match loaded {
        Ok(function) => {
            if functions.insert(symbol, function).is_some() {
                return Err(format!("duplicate TriadSm89ExactF32D128 function {symbol}"));
            }
        }
        Err(reason) => exclusions.push(Tf32SymbolExclusion { symbol, reason }),
    }
    Ok(())
}

fn sm89_exact_f32_d128_abi_for_symbol<'a>(
    census: &'a BTreeMap<&'static str, Result<Tf32DriverAbi, String>>,
    symbol: &str,
) -> Result<&'a Tf32DriverAbi, String> {
    census
        .get(symbol)
        .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?
        .as_ref()
        .map_err(Clone::clone)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sm89ExactF32D128ResourceFacts {
    local_bytes: u32,
    registers: u32,
    static_shared_bytes: u32,
    max_threads: i32,
    occupancy: u32,
}

fn validate_sm89_exact_f32_d128_resources(
    spec: &super::sm89_exact_f32_d128_source::Sm89ExactF32D128KernelSpec,
    facts: Sm89ExactF32D128ResourceFacts,
) -> Result<(), String> {
    if facts.static_shared_bytes != spec.static_shared_bytes {
        return Err(format!(
            "{} uses {} static shared bytes, expected {}",
            spec.symbol, facts.static_shared_bytes, spec.static_shared_bytes
        ));
    }
    tf32_symbol_admission(
        spec.symbol,
        facts.local_bytes,
        0,
        facts.registers,
        spec.register_cap,
        facts.max_threads,
        i32::try_from(spec.block.0 * spec.block.1 * spec.block.2)
            .map_err(|_| format!("{} thread count exceeds i32::MAX", spec.symbol))?,
    )?;
    if facts.occupancy < spec.occupancy_gate {
        return Err(format!(
            "{} occupancy {} misses its {}-CTA gate",
            spec.symbol, facts.occupancy, spec.occupancy_gate
        ));
    }
    Ok(())
}

fn load_sm89_exact_f32_d128_functions(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> Result<
    (
        HashMap<&'static str, CudaFunction>,
        Vec<Tf32SymbolExclusion>,
    ),
    String,
> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm89ExactF32D128
        || sm80_ptx_target(module.compiler_identity.target.as_str()).is_none()
        || ctx
            .compute_capability()
            .map_err(|error| format!("query TriadSm89ExactF32D128 CC: {error:?}"))?
            .0
            < 8
    {
        return Err("TriadSm89ExactF32D128 requires an SM80+ portable binding".into());
    }
    let abi = module
        .sm89_exact_f32_d128_driver_abi
        .as_ref()
        .map_err(Clone::clone)?;
    let mut functions = HashMap::new();
    let mut exclusions = Vec::new();
    for spec in &super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS {
        let loaded = (|| {
            validate_sm89_exact_f32_d128_driver_abi(
                spec,
                sm89_exact_f32_d128_abi_for_symbol(abi, spec.symbol)?,
            )?;
            let function = load_function(
                &module.module,
                ModuleKind::TriadSm89ExactF32D128,
                spec.symbol,
            )?;
            let local_bytes = u32::try_from(
                function
                    .local_size_bytes()
                    .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?,
            )
            .map_err(|_| format!("{} returned negative local memory", spec.symbol))?;
            let registers = u32::try_from(
                function
                    .num_regs()
                    .map_err(|error| format!("query {} registers: {error:?}", spec.symbol))?,
            )
            .map_err(|_| format!("{} returned negative register count", spec.symbol))?;
            let static_shared_bytes =
                u32::try_from(function.shared_size_bytes().map_err(|error| {
                    format!("query {} static shared memory: {error:?}", spec.symbol)
                })?)
                .map_err(|_| format!("{} returned negative static shared memory", spec.symbol))?;
            let max_threads = function
                .max_threads_per_block()
                .map_err(|error| format!("query {} max threads: {error:?}", spec.symbol))?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    spec.block.0 * spec.block.1 * spec.block.2,
                    spec.dynamic_shared_bytes as usize,
                    None,
                )
                .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
            validate_sm89_exact_f32_d128_resources(
                spec,
                Sm89ExactF32D128ResourceFacts {
                    local_bytes,
                    registers,
                    static_shared_bytes,
                    max_threads,
                    occupancy,
                },
            )?;
            Ok(function)
        })();
        retain_sm89_exact_f32_d128_symbol(&mut functions, &mut exclusions, spec.symbol, loaded)?;
    }
    if functions.len() + exclusions.len()
        != super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS.len()
    {
        return Err("TriadSm89ExactF32D128 lost a symbol while applying resource gates".into());
    }
    Ok((functions, exclusions))
}

fn retain_sm89_tf32_joint_symbol<T>(
    functions: &mut HashMap<&'static str, T>,
    exclusions: &mut Vec<Tf32SymbolExclusion>,
    symbol: &'static str,
    loaded: Result<T, String>,
) -> Result<(), String> {
    match loaded {
        Ok(function) => {
            if functions.insert(symbol, function).is_some() {
                return Err(format!("duplicate TriadSm89Tf32Joint function {symbol}"));
            }
        }
        Err(reason) => exclusions.push(Tf32SymbolExclusion { symbol, reason }),
    }
    Ok(())
}

fn sm89_tf32_joint_abi_for_symbol<'a>(
    census: &'a BTreeMap<&'static str, Result<Tf32DriverAbi, String>>,
    symbol: &str,
) -> Result<&'a Tf32DriverAbi, String> {
    census
        .get(symbol)
        .ok_or_else(|| format!("{symbol} has no live Driver ABI census"))?
        .as_ref()
        .map_err(Clone::clone)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sm89Tf32JointResourceFacts {
    local_bytes: u32,
    registers: u32,
    static_shared_bytes: u32,
    max_threads: i32,
    occupancy: u32,
}

fn validate_sm89_tf32_joint_resources(
    spec: &super::sm89_tf32_joint_source::Sm89Tf32JointKernelSpec,
    facts: Sm89Tf32JointResourceFacts,
) -> Result<(), String> {
    if facts.static_shared_bytes != spec.static_shared_bytes {
        return Err(format!(
            "{} uses {} static shared bytes, expected {}",
            spec.symbol, facts.static_shared_bytes, spec.static_shared_bytes
        ));
    }
    tf32_symbol_admission(
        spec.symbol,
        facts.local_bytes,
        spec.local_bytes,
        facts.registers,
        spec.register_cap,
        facts.max_threads,
        i32::try_from(spec.minimum_max_threads)
            .map_err(|_| format!("{} max-thread floor exceeds i32::MAX", spec.symbol))?,
    )?;
    if let Some(gate) = spec.minimum_active_blocks
        && facts.occupancy < gate
    {
        return Err(format!(
            "{} occupancy {} misses its {}-CTA gate",
            spec.symbol, facts.occupancy, gate
        ));
    }
    Ok(())
}

fn load_sm89_tf32_joint_functions(
    ctx: &CudaContext,
    module: &CompiledModule,
) -> Result<
    (
        HashMap<&'static str, CudaFunction>,
        Vec<Tf32SymbolExclusion>,
    ),
    String,
> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm89Tf32Joint
        || sm80_ptx_target(module.compiler_identity.target.as_str()).is_none()
        || ctx
            .compute_capability()
            .map_err(|error| format!("query TriadSm89Tf32Joint CC: {error:?}"))?
            .0
            < 8
    {
        return Err("TriadSm89Tf32Joint requires an SM80+ portable binding".into());
    }
    let optin_shared = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .map_err(|error| format!("query TriadSm89Tf32Joint opt-in shared memory: {error:?}"))?;
    let optin_shared = u32::try_from(optin_shared)
        .map_err(|_| format!("negative TriadSm89Tf32Joint opt-in shared memory {optin_shared}"))?;
    let abi = module
        .sm89_tf32_joint_driver_abi
        .as_ref()
        .map_err(Clone::clone)?;
    let mut functions = HashMap::new();
    let mut exclusions = Vec::new();
    for spec in &super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS {
        let loaded = (|| {
            validate_sm89_tf32_joint_driver_abi(
                spec,
                sm89_tf32_joint_abi_for_symbol(abi, spec.symbol)?,
            )?;
            if optin_shared < spec.dynamic_shared_bytes {
                return Err(format!(
                    "{} needs {} dynamic shared bytes, device exposes {optin_shared}",
                    spec.symbol, spec.dynamic_shared_bytes
                ));
            }
            let function =
                load_function(&module.module, ModuleKind::TriadSm89Tf32Joint, spec.symbol)?;
            if spec.dynamic_shared_bytes != 0 {
                set_dynamic_shared(
                    &function,
                    spec.symbol,
                    i32::try_from(spec.dynamic_shared_bytes)
                        .map_err(|_| format!("{} shared memory exceeds i32::MAX", spec.symbol))?,
                )?;
            }
            let local_bytes = u32::try_from(
                function
                    .local_size_bytes()
                    .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?,
            )
            .map_err(|_| format!("{} returned negative local memory", spec.symbol))?;
            let registers = u32::try_from(
                function
                    .num_regs()
                    .map_err(|error| format!("query {} registers: {error:?}", spec.symbol))?,
            )
            .map_err(|_| format!("{} returned negative register count", spec.symbol))?;
            let static_shared_bytes =
                u32::try_from(function.shared_size_bytes().map_err(|error| {
                    format!("query {} static shared memory: {error:?}", spec.symbol)
                })?)
                .map_err(|_| format!("{} returned negative static shared memory", spec.symbol))?;
            let max_threads = function
                .max_threads_per_block()
                .map_err(|error| format!("query {} max threads: {error:?}", spec.symbol))?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(
                    spec.block.0 * spec.block.1 * spec.block.2,
                    spec.dynamic_shared_bytes as usize,
                    None,
                )
                .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
            validate_sm89_tf32_joint_resources(
                spec,
                Sm89Tf32JointResourceFacts {
                    local_bytes,
                    registers,
                    static_shared_bytes,
                    max_threads,
                    occupancy,
                },
            )?;
            Ok(function)
        })();
        retain_sm89_tf32_joint_symbol(&mut functions, &mut exclusions, spec.symbol, loaded)?;
    }
    if functions.len() + exclusions.len()
        != super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS.len()
    {
        return Err("TriadSm89Tf32Joint lost a symbol while applying resource gates".into());
    }
    Ok((functions, exclusions))
}

pub struct GemmBiKernels {
    _modules: CudaModuleAnchors,
    allocation_domain: super::contract::AllocationDomain,
    compute_capability: (u32, u32),
    multiprocessor_count: u32,
    scalar_compiler_identity: CompilerIdentity,
    sm80_compiler_identity: CompilerIdentity,
    finalist_compiler_identity: Option<CompilerIdentity>,
    sm89_half_compiler_identity: Option<CompilerIdentity>,
    sm89_exact_f32_compiler_identity: Option<CompilerIdentity>,
    sm89_exact_f32_d128_compiler_identity: Option<CompilerIdentity>,
    sm89_tf32_joint_compiler_identity: Option<CompilerIdentity>,
    specialized_compiler_identity: Option<CompilerIdentity>,
    artifact_set_identity: crate::mamba_ssm::gpu::kernel_identity::ArtifactSetIdentity,
    f32_triad_availability: super::contract::F32TriadAvailability,
    tf32_driver_abi: BTreeMap<&'static str, Tf32DriverAbi>,
    portable_tf32_functions: HashMap<&'static str, CudaFunction>,
    tf32_splitk_functions: HashMap<&'static str, CudaFunction>,
    finalist_tf32_functions: HashMap<&'static str, CudaFunction>,
    sm89_half_functions: HashMap<&'static str, CudaFunction>,
    sm89_half_exclusions: Vec<Tf32SymbolExclusion>,
    sm89_half_relay_resident_ctas: u32,
    sm89_exact_f32_functions: HashMap<&'static str, CudaFunction>,
    sm89_exact_f32_exclusions: Vec<Tf32SymbolExclusion>,
    sm89_exact_f32_d128_functions: HashMap<&'static str, CudaFunction>,
    sm89_exact_f32_d128_exclusions: Vec<Tf32SymbolExclusion>,
    sm89_tf32_joint_functions: HashMap<&'static str, CudaFunction>,
    sm89_tf32_joint_exclusions: Vec<Tf32SymbolExclusion>,
    specialized_tf32_functions: HashMap<&'static str, CudaFunction>,
    portable_tf32_rejection: Option<String>,
    finalist_tf32_rejection: Option<String>,
    sm89_half_rejection: Option<String>,
    sm89_exact_f32_rejection: Option<String>,
    sm89_exact_f32_d128_rejection: Option<String>,
    sm89_tf32_joint_rejection: Option<String>,
    specialized_tf32_rejection: Option<String>,
    tf32_excluded_symbols: Vec<Tf32SymbolExclusion>,
    specialized_functions: HashMap<&'static str, CudaFunction>,
    sm120_target: Option<super::contract::Sm120TargetCandidate>,
    sm120_device_caps: Option<crate::mamba_ssm::gpu::kernel_identity::DeviceCaps>,
    sm120_resources: HashMap<&'static str, super::contract::Sm120KernelResources>,
    tf32_tensor_maps: Tf32MapCache,
    sm90a_tensor_maps: Sm90aMapCache,
    sm100_tensor_maps: Sm100MapCache,
    sm120_tensor_maps: Sm120MapCache,

    pub gemm_bi_nn: CudaFunction,
    pub gemm_bi_nn_m64n64_bk16_s2: CudaFunction,
    pub gemm_bi_nn_splitk32_m32n64_exact: CudaFunction,
    pub gemm_bi_nn_prism_m64n64_bk16_s2: CudaFunction,
    pub gemm_bi_tn: CudaFunction,
    pub gemm_bi_tn_aligned: CudaFunction,
    pub gemm_bi_tn_m16n16_bk16_s2_splitm16: CudaFunction,
    pub gemm_bi_nt: CudaFunction,
    pub gemm_bi_nt_m2n16_bk64_splitk32: CudaFunction,
    pub gemm_bi_nn_slim: CudaFunction,
    pub gemm_bi_tn_slim: CudaFunction,
    pub gemm_bi_nt_slim: CudaFunction,
    pub gemm_bi_nn_ultra_thin: CudaFunction,
    pub gemm_bi_nn_gemv: CudaFunction,
    pub gemm_bi_tn_gemv: CudaFunction,
    pub gemm_bi_nt_gemv: CudaFunction,
    pub gemm_bi_nn_narrow: CudaFunction,
    pub gemm_bi_nn_narrow_small: CudaFunction,
    pub gemm_bi_tn_narrow: CudaFunction,
    pub gemm_bi_tn_narrow_splitm_partial: CudaFunction,
    pub gemm_bi_tn_narrow_splitm_partial_aligned: CudaFunction,
    pub gemm_bi_nt_narrow: CudaFunction,
    pub gemm_bi_nn_splitk32_partial: CudaFunction,
    pub gemm_bi_splitk_reduce: CudaFunction,
    pub gemm_bi_tn_splitm_partial: CudaFunction,
    pub gemm_bi_tn_splitm_partial_aligned: CudaFunction,
    pub gemm_bi_splitm_reduce: CudaFunction,
    pub gemm_bi_nn_splitk_slim_partial: CudaFunction,
    pub gemm_bi_transpose_f32_2d: CudaFunction,
    pub gemm_bi_transpose_f32_32x16_d768: CudaFunction,
    pub gemm_bi_dx_col_gemv: CudaFunction,
    pub gemm_bi_nn_zero_reduction: CudaFunction,
    pub gemm_bi_tn_zero_reduction: CudaFunction,
    pub gemm_bi_nt_zero_reduction: CudaFunction,

    pub gemm_bi_nn_gemv_typed: HalfKernel,
    pub gemm_bi_tn_gemv_typed: HalfKernel,
    pub gemm_bi_nt_gemv_typed: HalfKernel,
    pub gemm_bi_nn_ultra_thin_typed: HalfKernel,
    pub gemm_bi_nn_narrow_typed: HalfKernel,
    pub gemm_bi_nn_narrow_small_typed: HalfKernel,
    pub gemm_bi_tn_narrow_typed: HalfKernel,
    pub gemm_bi_nt_narrow_typed: HalfKernel,
    pub gemm_bi_nn_big_typed: HalfKernel,
    pub gemm_bi_tn_big_typed: HalfKernel,
    pub gemm_bi_nt_big_typed: HalfKernel,
    pub gemm_bi_nn_tc_typed: HalfKernel,
    pub gemm_bi_tn_tc_typed: HalfKernel,
    pub gemm_bi_nt_tc_typed: HalfKernel,
    pub gemm_bi_nn_tc64_typed: HalfKernel,
    pub gemm_bi_nn_tc16_typed: HalfKernel,
    pub gemm_bi_tn_tc64_typed: HalfKernel,
    /// The tc64 TN dW body over the persistent stream-K grid; `None` on the
    /// targets whose portable module does not compose it (CC 12.x).
    pub gemm_bi_tn_tc64_streamk_typed: Option<HalfKernel>,
    /// How many CTAs of the stream-K kernel one multiprocessor holds at
    /// once, from the driver's occupancy query at load (0 when the kernel
    /// is not composed). The persistent grid may not exceed that many per
    /// multiprocessor: a CTA waits on lower CTAs and can only do so safely
    /// while every CTA of the grid is resident.
    pub tc64_streamk_resident_ctas: u32,
    pub gemm_bi_tn_tc128x64_typed: HalfKernel,
    pub gemm_bi_nt_tc64_typed: HalfKernel,

    splitk_scratch: std::sync::OnceLock<CudaSlice<f32>>,
    tf32_splitk_counters: std::sync::OnceLock<CudaSlice<u32>>,
    transpose_scratch: std::sync::OnceLock<GpuBuffer>,
}

/// The compiled modules a Triad kernel set is assembled from. Every optional
/// module travels with the text of its compile rejection, so a set can say
/// why a route is missing instead of silently lacking it.
pub(crate) struct GemmBiModuleSet {
    pub fixed_artifact: ArtifactIdentity,
    pub scalar: CompiledModule,
    pub sm80: CompiledModule,
    pub finalist: Option<CompiledModule>,
    pub finalist_compile_rejection: Option<String>,
    pub sm89_half: Option<CompiledModule>,
    pub sm89_half_compile_rejection: Option<String>,
    pub sm89_exact_f32: Option<CompiledModule>,
    pub sm89_exact_f32_compile_rejection: Option<String>,
    pub sm89_exact_f32_d128: Option<CompiledModule>,
    pub sm89_exact_f32_d128_compile_rejection: Option<String>,
    pub sm89_tf32_joint: Option<CompiledModule>,
    pub sm89_tf32_joint_compile_rejection: Option<String>,
    pub specialized: Option<QualifiedSpecializedModule>,
    /// The inference cells module's identity, when the board composed it.
    pub sm89_cells: Option<ArtifactIdentity>,
}

impl GemmBiKernels {
    pub(crate) fn load(ctx: &Arc<CudaContext>, modules: GemmBiModuleSet) -> Result<Self, String> {
        let GemmBiModuleSet {
            fixed_artifact,
            scalar,
            sm80,
            finalist,
            finalist_compile_rejection,
            sm89_half,
            sm89_half_compile_rejection,
            sm89_exact_f32,
            sm89_exact_f32_compile_rejection,
            sm89_exact_f32_d128,
            sm89_exact_f32_d128_compile_rejection,
            sm89_tf32_joint,
            sm89_tf32_joint_compile_rejection,
            specialized,
            sm89_cells,
        } = modules;
        let allocation_domain = super::contract::AllocationDomain::from_context(ctx)?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|error| format!("query scalar compute capability: {error:?}"))?;
        let compute_capability = (
            u32::try_from(major).map_err(|_| format!("negative CUDA CC major {major}"))?,
            u32::try_from(minor).map_err(|_| format!("negative CUDA CC minor {minor}"))?,
        );
        let multiprocessor_count = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|error| format!("query multiprocessor count: {error:?}"))?;
        let multiprocessor_count = u32::try_from(multiprocessor_count)
            .map_err(|_| format!("negative multiprocessor count {multiprocessor_count}"))?;
        if multiprocessor_count == 0 {
            return Err("CUDA device reported zero multiprocessors".into());
        }
        let mut artifacts = vec![
            fixed_artifact,
            scalar.artifact_identity,
            sm80.artifact_identity,
        ];
        // The common SM89-measured modules and a board's own architecture
        // module serve side by side: the board's table decides per shape.
        // The push order is the identity's canonical order, and the Ada
        // set (no architecture module) hashes exactly as before.
        if let Some(finalist) = finalist.as_ref() {
            artifacts.push(finalist.artifact_identity);
        }
        if let Some(specialized) = specialized.as_ref() {
            artifacts.push(specialized.module.artifact_identity);
        }
        if let Some(sm89_half) = sm89_half.as_ref() {
            artifacts.push(sm89_half.artifact_identity);
        }
        if let Some(sm89_exact_f32) = sm89_exact_f32.as_ref() {
            artifacts.push(sm89_exact_f32.artifact_identity);
        }
        if let Some(sm89_exact_f32_d128) = sm89_exact_f32_d128.as_ref() {
            artifacts.push(sm89_exact_f32_d128.artifact_identity);
        }
        if let Some(sm89_tf32_joint) = sm89_tf32_joint.as_ref() {
            artifacts.push(sm89_tf32_joint.artifact_identity);
        }
        if let Some(sm89_cells) = sm89_cells {
            artifacts.push(sm89_cells);
        }
        let artifact_set_identity =
            crate::mamba_ssm::gpu::kernel_identity::build_artifact_set(&artifacts)?;
        let mut finalist_tf32_rejection = finalist_compile_rejection;
        let (merged_portable_finalist_abi, finalist_abi_rejection) =
            merge_optional_finalist_driver_abi(
                sm80.tf32_driver_abi.clone(),
                finalist
                    .as_ref()
                    .map(|finalist| finalist.tf32_driver_abi.clone()),
            );
        let finalist_abi_merged = finalist_abi_rejection.is_none();
        if let Some(error) = finalist_abi_rejection {
            finalist_tf32_rejection = Some(error);
        }
        let tf32_driver_abi = merge_tf32_driver_abi(
            merged_portable_finalist_abi,
            specialized
                .as_ref()
                .map(|specialized| specialized.module.tf32_driver_abi.clone()),
        )?;
        let mut portable_tf32_rejection = None;
        let mut tf32_excluded_symbols = Vec::new();
        let (portable_tf32_functions, portable) = if sm80.tf32_qualified {
            let qualification = (|| {
                let (functions, excluded) = load_tf32_functions(&sm80)?;
                tf32_excluded_symbols.extend(excluded);
                let binding = qualify_tf32_module_binding(ctx, &sm80)?;
                let artifact = qualify_loaded_tf32_artifact(
                    ctx,
                    allocation_domain,
                    &sm80,
                    binding,
                    &functions,
                );
                retain_tf32_candidate(functions, binding, artifact)
            })();
            match qualification {
                Ok(qualified) => qualified,
                Err(error) => {
                    portable_tf32_rejection = Some(error);
                    (HashMap::new(), None)
                }
            }
        } else {
            portable_tf32_rejection = sm80.tf32_qualification_error.clone();
            (HashMap::new(), None)
        };
        let (tf32_splitk_functions, splitk_excluded) =
            retain_forced_only_functions(load_tf32_splitk_functions(&sm80))?;
        tf32_excluded_symbols.extend(splitk_excluded);
        let finalist_qualification = finalist
            .as_ref()
            .filter(|_| finalist_abi_merged)
            .filter(|module| {
                if module.tf32_qualified {
                    true
                } else {
                    finalist_tf32_rejection = module.tf32_qualification_error.clone();
                    false
                }
            })
            .map(|module| {
                let (functions, excluded) = load_tf32_functions(module)?;
                tf32_excluded_symbols.extend(excluded);
                let binding = qualify_tf32_module_binding(ctx, module)?;
                let artifact = qualify_loaded_tf32_artifact(
                    ctx,
                    allocation_domain,
                    module,
                    binding,
                    &functions,
                );
                retain_tf32_candidate(functions, binding, artifact)
            })
            .transpose();
        let (finalist_tf32_functions, finalist_binding) = match finalist_qualification {
            Ok(Some(qualified)) => qualified,
            Ok(None) => (HashMap::new(), None),
            Err(error) => {
                finalist_tf32_rejection = Some(error);
                (HashMap::new(), None)
            }
        };
        if let Some(specialized) = specialized.as_ref() {
            tf32_excluded_symbols.extend(specialized.tf32_excluded.iter().cloned());
        }
        let mut specialized_tf32_rejection = specialized
            .as_ref()
            .and_then(|specialized| specialized.tf32_rejection.clone());
        let specialized_qualification = specialized
            .as_ref()
            .filter(|_| specialized_tf32_rejection.is_none())
            .filter(|specialized| !specialized.tf32_functions.is_empty())
            .map(|specialized| {
                let binding = qualify_tf32_module_binding(ctx, &specialized.module)?;
                let functions = specialized.tf32_functions.clone();
                let artifact = qualify_loaded_tf32_artifact(
                    ctx,
                    allocation_domain,
                    &specialized.module,
                    binding,
                    &functions,
                );
                retain_specialized_tf32_candidate(
                    functions,
                    binding,
                    &specialized.tf32_excluded,
                    artifact,
                )
            })
            .transpose();
        let (specialized_tf32_functions, specialized_binding) = match specialized_qualification {
            Ok(Some(qualified)) => qualified,
            Ok(None) => (HashMap::new(), None),
            Err(error) => {
                specialized_tf32_rejection = Some(error);
                (HashMap::new(), None)
            }
        };
        let mut sm89_half_rejection = sm89_half_compile_rejection;
        let (sm89_half_functions, sm89_half_exclusions) = match sm89_half.as_ref() {
            Some(module) => match load_sm89_half_functions(ctx, module) {
                Ok(loaded) => loaded,
                Err(error) => {
                    sm89_half_rejection = Some(error);
                    (HashMap::new(), Vec::new())
                }
            },
            None => (HashMap::new(), Vec::new()),
        };
        let sm89_half_relay_resident_ctas = sm89_half_relay_resident(&sm89_half_functions)?;
        let mut sm89_exact_f32_rejection = sm89_exact_f32_compile_rejection;
        let (sm89_exact_f32_functions, sm89_exact_f32_exclusions) = match sm89_exact_f32.as_ref() {
            Some(module) => match load_sm89_exact_f32_functions(ctx, module) {
                Ok(loaded) => loaded,
                Err(error) => {
                    sm89_exact_f32_rejection = Some(error);
                    (HashMap::new(), Vec::new())
                }
            },
            None => (HashMap::new(), Vec::new()),
        };
        let mut sm89_exact_f32_d128_rejection = sm89_exact_f32_d128_compile_rejection;
        let (sm89_exact_f32_d128_functions, sm89_exact_f32_d128_exclusions) =
            match sm89_exact_f32_d128.as_ref() {
                Some(module) => match load_sm89_exact_f32_d128_functions(ctx, module) {
                    Ok(loaded) => loaded,
                    Err(error) => {
                        sm89_exact_f32_d128_rejection = Some(error);
                        (HashMap::new(), Vec::new())
                    }
                },
                None => (HashMap::new(), Vec::new()),
            };
        let mut sm89_tf32_joint_rejection = sm89_tf32_joint_compile_rejection;
        let (sm89_tf32_joint_functions, sm89_tf32_joint_exclusions) = match sm89_tf32_joint.as_ref()
        {
            Some(module) => match load_sm89_tf32_joint_functions(ctx, module) {
                Ok(loaded) => loaded,
                Err(error) => {
                    sm89_tf32_joint_rejection = Some(error);
                    (HashMap::new(), Vec::new())
                }
            },
            None => (HashMap::new(), Vec::new()),
        };
        let joint_binding = match sm89_tf32_joint
            .as_ref()
            .filter(|_| !sm89_tf32_joint_functions.is_empty())
        {
            Some(module) => match qualify_tf32_module_binding(ctx, module) {
                Ok(binding) => Some(binding),
                Err(error) => {
                    sm89_tf32_joint_rejection = Some(error);
                    None
                }
            },
            None => None,
        };
        let f32_triad_availability = super::contract::F32TriadAvailability {
            portable,
            specialized: specialized_binding,
            finalist: finalist_binding,
            joint: joint_binding,
            multiprocessors: multiprocessor_count,
        };
        let load = |name: &str| load_owned_function(name, &scalar.module, &sm80.module);
        let load_half = |base: &str| load_owned_half(base, &scalar.module, &sm80.module);
        let load_half_dynsmem = |base: &str, bytes: i32| {
            let kernel = load_half(base)?;
            set_half_dynamic_shared(&kernel, base, bytes)?;
            Ok::<HalfKernel, String>(kernel)
        };

        let gemm_bi_tn_tc64_streamk_typed =
            if sm80_target_composes_streamk(sm80.compiler_identity.target.as_str()) {
                Some(load_half("tn_tc64_streamk")?)
            } else {
                None
            };
        let tc64_streamk_resident_ctas = match &gemm_bi_tn_tc64_streamk_typed {
            Some(kernel) => streamk_resident_ctas(kernel)?,
            None => 0,
        };
        let gemm_bi_nn = load("nn_big")?;
        set_dynamic_shared(&gemm_bi_nn, "nn_big", 34 * 1024)?;
        let gemm_bi_nn_m64n64_bk16_s2 = load("nn_m64n64_bk16_s2")?;
        set_dynamic_shared(
            &gemm_bi_nn_m64n64_bk16_s2,
            "nn_m64n64_bk16_s2",
            super::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES as i32,
        )?;
        let gemm_bi_nn_splitk32_m32n64_exact = load("nn_splitk32_m32n64_exact")?;
        let gemm_bi_nn_prism_m64n64_bk16_s2 = load("nn_prism_m64n64_bk16_s2")?;
        set_dynamic_shared(
            &gemm_bi_nn_prism_m64n64_bk16_s2,
            "nn_prism_m64n64_bk16_s2",
            super::contract::SCALAR_NN_M64N64_DYNAMIC_SHARED_BYTES as i32,
        )?;
        let gemm_bi_tn = load("tn_big")?;
        set_dynamic_shared(&gemm_bi_tn, "tn_big", 34 * 1024)?;
        let gemm_bi_tn_aligned = load("tn_aligned")?;
        set_dynamic_shared(&gemm_bi_tn_aligned, "tn_aligned", 34 * 1024)?;
        let gemm_bi_tn_m16n16_bk16_s2_splitm16 = load("tn_m16n16_bk16_s2_splitm16")?;
        set_dynamic_shared(
            &gemm_bi_tn_m16n16_bk16_s2_splitm16,
            "tn_m16n16_bk16_s2_splitm16",
            super::contract::SCALAR_TN_M16N16_DYNAMIC_SHARED_BYTES as i32,
        )?;
        let gemm_bi_nt = load("nt_big")?;
        set_dynamic_shared(
            &gemm_bi_nt,
            "nt_big",
            super::contract::SCALAR_BIG_NT_DYNAMIC_SHARED_BYTES as i32,
        )?;
        let gemm_bi_nt_m2n16_bk64_splitk32 = load("nt_m2n16_bk64_splitk32")?;
        set_dynamic_shared(
            &gemm_bi_nt_m2n16_bk64_splitk32,
            "nt_m2n16_bk64_splitk32",
            super::contract::SCALAR_NT_M2N16_DYNAMIC_SHARED_BYTES as i32,
        )?;
        let gemm_bi_transpose_f32_32x16_d768 = load("transpose_f32_32x16_d768")?;
        let scalar_compiler = scalar.compiler_identity;
        let scalar_artifact = scalar.artifact_identity;
        if qualified_scalar_resource_environment(
            compute_capability,
            multiprocessor_count,
            scalar_compiler,
            scalar_artifact,
        ) {
            qualify_scalar_nt_d768_transpose(&gemm_bi_transpose_f32_32x16_d768)?;
            qualify_scalar_nn_m32n64_splitk32(&gemm_bi_nn_splitk32_m32n64_exact)?;
            qualify_scalar_nt_m2n16(&gemm_bi_nt_m2n16_bk64_splitk32)?;
            qualify_scalar_tn_m16n16(&gemm_bi_tn_m16n16_bk16_s2_splitm16)?;
        }

        let specialized_functions = specialized
            .as_ref()
            .map(|specialized| specialized.functions.clone())
            .unwrap_or_default();
        let sm120_target = specialized
            .as_ref()
            .and_then(|specialized| specialized.sm120_target);
        let sm120_device_caps = specialized
            .as_ref()
            .and_then(|specialized| specialized.sm120_device_caps);
        let sm120_resources = specialized
            .as_ref()
            .map(|specialized| specialized.sm120_resources.clone())
            .unwrap_or_default();
        let mut anchors = vec![scalar.module.clone(), sm80.module.clone()];
        if let Some(finalist) = finalist.as_ref() {
            anchors.push(finalist.module.clone());
        }
        if let Some(sm89_half) = sm89_half.as_ref() {
            anchors.push(sm89_half.module.clone());
        }
        if let Some(sm89_exact_f32) = sm89_exact_f32.as_ref() {
            anchors.push(sm89_exact_f32.module.clone());
        }
        if let Some(sm89_exact_f32_d128) = sm89_exact_f32_d128.as_ref() {
            anchors.push(sm89_exact_f32_d128.module.clone());
        }
        if let Some(sm89_tf32_joint) = sm89_tf32_joint.as_ref() {
            anchors.push(sm89_tf32_joint.module.clone());
        }
        if let Some(specialized) = specialized.as_ref() {
            anchors.push(specialized.module.module.clone());
        }

        Ok(Self {
            _modules: CudaModuleAnchors::new(anchors),
            allocation_domain,
            compute_capability,
            multiprocessor_count,
            scalar_compiler_identity: scalar.compiler_identity,
            sm80_compiler_identity: sm80.compiler_identity,
            finalist_compiler_identity: finalist.as_ref().map(|module| module.compiler_identity),
            sm89_half_compiler_identity: sm89_half.as_ref().map(|module| module.compiler_identity),
            sm89_exact_f32_compiler_identity: sm89_exact_f32
                .as_ref()
                .map(|module| module.compiler_identity),
            sm89_exact_f32_d128_compiler_identity: sm89_exact_f32_d128
                .as_ref()
                .map(|module| module.compiler_identity),
            sm89_tf32_joint_compiler_identity: sm89_tf32_joint
                .as_ref()
                .map(|module| module.compiler_identity),
            specialized_compiler_identity: specialized
                .as_ref()
                .map(|specialized| specialized.module.compiler_identity),
            artifact_set_identity,
            f32_triad_availability,
            tf32_driver_abi,
            portable_tf32_functions,
            tf32_splitk_functions,
            finalist_tf32_functions,
            sm89_half_functions,
            sm89_half_exclusions,
            sm89_half_relay_resident_ctas,
            sm89_exact_f32_functions,
            sm89_exact_f32_exclusions,
            sm89_exact_f32_d128_functions,
            sm89_exact_f32_d128_exclusions,
            sm89_tf32_joint_functions,
            sm89_tf32_joint_exclusions,
            tf32_excluded_symbols,
            specialized_tf32_functions,
            portable_tf32_rejection,
            finalist_tf32_rejection,
            sm89_half_rejection,
            sm89_exact_f32_rejection,
            sm89_exact_f32_d128_rejection,
            sm89_tf32_joint_rejection,
            specialized_tf32_rejection,
            specialized_functions,
            sm120_target,
            sm120_device_caps,
            sm120_resources,
            tf32_tensor_maps: Mutex::new(HashMap::new()),
            sm90a_tensor_maps: Mutex::new(HashMap::new()),
            sm100_tensor_maps: Mutex::new(HashMap::new()),
            sm120_tensor_maps: Mutex::new(HashMap::new()),
            gemm_bi_nn,
            gemm_bi_nn_m64n64_bk16_s2,
            gemm_bi_nn_splitk32_m32n64_exact,
            gemm_bi_nn_prism_m64n64_bk16_s2,
            gemm_bi_tn,
            gemm_bi_tn_aligned,
            gemm_bi_tn_m16n16_bk16_s2_splitm16,
            gemm_bi_nt,
            gemm_bi_nt_m2n16_bk64_splitk32,
            gemm_bi_nn_slim: load("nn_slim")?,
            gemm_bi_tn_slim: load("tn_slim")?,
            gemm_bi_nt_slim: load("nt_slim")?,
            gemm_bi_nn_ultra_thin: load("nn_ultra_thin")?,
            gemm_bi_nn_gemv: load("nn_gemv")?,
            gemm_bi_tn_gemv: load("tn_gemv")?,
            gemm_bi_nt_gemv: load("nt_gemv")?,
            gemm_bi_nn_narrow: load("nn_narrow")?,
            gemm_bi_nn_narrow_small: load("nn_narrow_small")?,
            gemm_bi_tn_narrow: load("tn_narrow")?,
            gemm_bi_tn_narrow_splitm_partial: load("tn_narrow_splitm_partial")?,
            gemm_bi_tn_narrow_splitm_partial_aligned: load("tn_narrow_splitm_partial_aligned")?,
            gemm_bi_nt_narrow: load("nt_narrow")?,
            gemm_bi_nn_splitk32_partial: load("nn_splitk32_partial")?,
            gemm_bi_splitk_reduce: load("splitk_reduce")?,
            gemm_bi_tn_splitm_partial: load("tn_splitm_partial")?,
            gemm_bi_tn_splitm_partial_aligned: load("tn_splitm_partial_aligned")?,
            gemm_bi_splitm_reduce: load("splitm_reduce")?,
            gemm_bi_nn_splitk_slim_partial: load("nn_splitk_slim_partial")?,
            gemm_bi_transpose_f32_2d: load("transpose_f32_2d")?,
            gemm_bi_transpose_f32_32x16_d768,
            gemm_bi_dx_col_gemv: load("dx_col_gemv")?,
            gemm_bi_nn_zero_reduction: load("nn_zero_reduction")?,
            gemm_bi_tn_zero_reduction: load("tn_zero_reduction")?,
            gemm_bi_nt_zero_reduction: load("nt_zero_reduction")?,
            gemm_bi_nn_gemv_typed: load_half("nn_gemv")?,
            gemm_bi_tn_gemv_typed: load_half("tn_gemv")?,
            gemm_bi_nt_gemv_typed: load_half("nt_gemv")?,
            gemm_bi_nn_ultra_thin_typed: load_half("nn_ultra_thin")?,
            gemm_bi_nn_narrow_typed: load_half("nn_narrow")?,
            gemm_bi_nn_narrow_small_typed: load_half("nn_narrow_small")?,
            gemm_bi_tn_narrow_typed: load_half("tn_narrow")?,
            gemm_bi_nt_narrow_typed: load_half("nt_narrow")?,
            gemm_bi_nn_big_typed: load_half_dynsmem("nn_big", 34 * 1024)?,
            gemm_bi_tn_big_typed: load_half_dynsmem("tn_big", 34 * 1024)?,
            gemm_bi_nt_big_typed: load_half_dynsmem("nt_big", 34 * 1024)?,
            gemm_bi_nn_tc_typed: load_half_dynsmem("nn_tc", 75_776)?,
            gemm_bi_tn_tc_typed: load_half_dynsmem("tn_tc", 75_776)?,
            gemm_bi_nt_tc_typed: load_half_dynsmem("nt_tc", 75_776)?,
            gemm_bi_nn_tc64_typed: load_half("nn_tc64")?,
            gemm_bi_nn_tc16_typed: load_half("nn_tc16")?,
            gemm_bi_tn_tc64_typed: load_half("tn_tc64")?,
            gemm_bi_tn_tc64_streamk_typed,
            tc64_streamk_resident_ctas,
            gemm_bi_tn_tc128x64_typed: load_half("tn_tc128x64")?,
            gemm_bi_nt_tc64_typed: load_half("nt_tc64")?,
            splitk_scratch: std::sync::OnceLock::new(),
            tf32_splitk_counters: std::sync::OnceLock::new(),
            transpose_scratch: std::sync::OnceLock::new(),
        })
    }

    pub fn artifact_set_identity(
        &self,
    ) -> crate::mamba_ssm::gpu::kernel_identity::ArtifactSetIdentity {
        self.artifact_set_identity
    }

    pub(crate) fn multiprocessor_count(&self) -> u32 {
        self.multiprocessor_count
    }

    pub(crate) fn tc64_streamk_resident_ctas(&self) -> u32 {
        self.tc64_streamk_resident_ctas
    }

    pub(crate) fn sm89_half_relay_resident_ctas(&self) -> u32 {
        self.sm89_half_relay_resident_ctas
    }

    /// Whether the board the kernels were bound on belongs to the SM120
    /// family, from the identity taken at load: the launch path reads it
    /// without a driver query.
    pub(crate) fn serves_sm120_family(&self) -> bool {
        crate::mamba_ssm::gpu::device::is_sm120_family(self.compute_capability())
    }

    pub(crate) fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    pub fn f32_triad_availability(&self) -> super::contract::F32TriadAvailability {
        self.f32_triad_availability
    }

    /// Why the specialized TF32 module (SM90a/SM100/SM120) is not bound, if
    /// it is not: the first qualification step that rejected it.
    pub(crate) fn specialized_tf32_rejection(&self) -> Option<&str> {
        self.specialized_tf32_rejection.as_deref()
    }

    pub(crate) fn finalist_tf32_rejection(&self) -> Option<&str> {
        self.finalist_tf32_rejection.as_deref()
    }

    pub(crate) fn sm89_half_rejection(&self) -> Option<&str> {
        self.sm89_half_rejection.as_deref()
    }

    pub(crate) fn sm89_half_exclusions(&self) -> &[Tf32SymbolExclusion] {
        &self.sm89_half_exclusions
    }

    pub(crate) fn sm89_exact_f32_rejection(&self) -> Option<&str> {
        self.sm89_exact_f32_rejection.as_deref()
    }

    pub(crate) fn sm89_exact_f32_exclusions(&self) -> &[Tf32SymbolExclusion] {
        &self.sm89_exact_f32_exclusions
    }

    pub(crate) fn sm89_exact_f32_d128_rejection(&self) -> Option<&str> {
        self.sm89_exact_f32_d128_rejection.as_deref()
    }

    pub(crate) fn sm89_exact_f32_d128_exclusions(&self) -> &[Tf32SymbolExclusion] {
        &self.sm89_exact_f32_d128_exclusions
    }

    pub(crate) fn sm89_tf32_joint_rejection(&self) -> Option<&str> {
        self.sm89_tf32_joint_rejection.as_deref()
    }

    pub(crate) fn sm89_tf32_joint_exclusions(&self) -> &[Tf32SymbolExclusion] {
        &self.sm89_tf32_joint_exclusions
    }

    /// Why the portable SM80 TF32 routes are not bound, if they are not.
    pub(crate) fn portable_tf32_rejection(&self) -> Option<&str> {
        self.portable_tf32_rejection.as_deref()
    }

    /// The TF32 symbols this toolkit could not serve, each with its reason.
    pub(crate) fn tf32_excluded_symbols(&self) -> &[Tf32SymbolExclusion] {
        &self.tf32_excluded_symbols
    }

    /// Why `symbol` is excluded on this toolkit, if it is.
    pub(crate) fn tf32_symbol_exclusion(&self, symbol: &str) -> Option<&str> {
        self.tf32_excluded_symbols
            .iter()
            .find(|exclusion| exclusion.symbol == symbol)
            .map(|exclusion| exclusion.reason.as_str())
    }

    pub(crate) fn tf32_qualification_rejection(
        &self,
        route: super::contract::Tf32PhysicalRoute,
    ) -> Option<&str> {
        match route.module_kind() {
            ModuleKind::TriadSm80 => self.portable_tf32_rejection.as_deref(),
            ModuleKind::TriadSm89Finalist => self.finalist_tf32_rejection.as_deref(),
            ModuleKind::TriadSm89Tf32Joint => self.sm89_tf32_joint_rejection.as_deref(),
            ModuleKind::TriadSm90a | ModuleKind::TriadSm100 | ModuleKind::TriadSm120 => {
                self.specialized_tf32_rejection.as_deref()
            }
            _ => None,
        }
    }

    pub(crate) fn tf32_driver_abi(&self, symbol: &str) -> Option<&Tf32DriverAbi> {
        self.tf32_driver_abi.get(symbol)
    }

    pub(crate) fn tf32_function(&self, symbol: &str) -> Option<&CudaFunction> {
        if super::sm89_tf32_joint_source::kernel_spec(symbol).is_some() {
            return self.sm89_tf32_joint_function(symbol);
        }
        self.tf32_driver_abi(symbol)?;
        self.portable_tf32_functions
            .get(symbol)
            .or_else(|| self.finalist_tf32_functions.get(symbol))
            .or_else(|| self.specialized_tf32_functions.get(symbol))
    }

    pub(crate) fn tf32_splitk_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.f32_triad_availability.portable?;
        self.tf32_driver_abi(symbol)?;
        self.tf32_splitk_functions.get(symbol)
    }

    pub fn scalar_compiler_identity(&self) -> CompilerIdentity {
        self.scalar_compiler_identity
    }

    pub fn sm80_compiler_identity(&self) -> CompilerIdentity {
        self.sm80_compiler_identity
    }

    pub fn sm89_finalist_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.finalist_compiler_identity
    }

    pub fn sm89_half_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.artifact_set_identity
            .sm89_half
            .and(self.sm89_half_compiler_identity)
    }

    pub fn sm89_half_function(
        &self,
        route: super::sm89_half_source::Sm89HalfRoute,
        dtype: WeightDtype,
    ) -> Option<&CudaFunction> {
        let spec = super::sm89_half_source::kernel_spec(route, dtype)?;
        self.sm89_half_compiler_identity()?;
        self.sm89_half_functions.get(spec.symbol)
    }

    pub(in crate::mamba_ssm::gpu) fn sm89_half_runtime_function(
        &self,
        symbol: &str,
    ) -> Option<&CudaFunction> {
        sm89_half_runtime_entry(
            &self.sm89_half_functions,
            self.sm89_half_compiler_identity().is_some(),
            symbol,
        )
    }

    pub fn sm89_exact_f32_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.artifact_set_identity
            .sm89_exact_f32
            .and(self.sm89_exact_f32_compiler_identity)
    }

    pub fn sm89_exact_f32_function(&self, symbol: &str) -> Option<&CudaFunction> {
        super::sm89_exact_f32_source::kernel_spec(symbol)?;
        self.sm89_exact_f32_compiler_identity()?;
        self.sm89_exact_f32_functions.get(symbol)
    }

    pub fn sm89_exact_f32_d128_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.artifact_set_identity
            .sm89_exact_f32_d128
            .and(self.sm89_exact_f32_d128_compiler_identity)
    }

    pub fn sm89_exact_f32_d128_function(&self, symbol: &str) -> Option<&CudaFunction> {
        super::sm89_exact_f32_d128_source::kernel_spec(symbol)?;
        self.sm89_exact_f32_d128_compiler_identity()?;
        self.sm89_exact_f32_d128_functions.get(symbol)
    }

    pub fn sm89_tf32_joint_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.artifact_set_identity
            .sm89_tf32_joint
            .and(self.sm89_tf32_joint_compiler_identity)
    }

    pub fn sm89_tf32_joint_function(&self, symbol: &str) -> Option<&CudaFunction> {
        super::sm89_tf32_joint_source::kernel_spec(symbol)?;
        self.sm89_tf32_joint_compiler_identity()?;
        self.sm89_tf32_joint_functions.get(symbol)
    }

    pub fn sm90a_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.artifact_set_identity
            .specialized
            .filter(|artifact| artifact.module_kind == ModuleKind::TriadSm90a)
            .and(self.specialized_compiler_identity)
    }

    pub fn sm100_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.artifact_set_identity
            .specialized
            .filter(|artifact| artifact.module_kind == ModuleKind::TriadSm100)
            .and(self.specialized_compiler_identity)
    }

    pub fn sm100_target_candidate(&self) -> Option<super::contract::Sm100TargetCandidate> {
        self.sm100_compiler_identity()
            .and_then(|compiler| sm100_target_for_arch(compiler.target.as_str()))
    }

    pub fn sm120_compiler_identity(&self) -> Option<CompilerIdentity> {
        self.artifact_set_identity
            .specialized
            .filter(|artifact| artifact.module_kind == ModuleKind::TriadSm120)
            .and(self.specialized_compiler_identity)
    }

    pub fn sm120_target_candidate(&self) -> Option<super::contract::Sm120TargetCandidate> {
        self.sm120_target
    }

    pub fn sm120_device_caps(&self) -> Option<crate::mamba_ssm::gpu::kernel_identity::DeviceCaps> {
        self.sm120_device_caps
    }

    pub fn has_sm90a_wgmma(&self) -> bool {
        self.sm90a_compiler_identity().is_some()
            && self.specialized_functions.len() == SM90A_SYMBOLS.len()
    }

    pub fn has_sm100_tcgen(&self) -> bool {
        self.sm100_compiler_identity().is_some()
            && self.specialized_functions.len() == super::contract::SM100_KERNEL_SPECS.len()
    }

    pub fn has_sm120_tma_mma16(&self) -> bool {
        self.sm120_compiler_identity().is_some()
            && self.specialized_functions.len() == super::contract::sm120_kernel_specs().count()
            && self.sm120_target.is_some()
            && self.sm120_device_caps.is_some()
            && self.sm120_resources.len() == super::contract::sm120_kernel_specs().count()
    }

    pub(super) fn allocation_domain(&self) -> super::contract::AllocationDomain {
        self.allocation_domain
    }

    pub(super) fn sm90a_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.has_sm90a_wgmma()
            .then(|| self.specialized_functions.get(symbol))
            .flatten()
    }

    pub(super) fn sm100_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.has_sm100_tcgen()
            .then(|| self.specialized_functions.get(symbol))
            .flatten()
    }

    pub(super) fn sm120_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.has_sm120_tma_mma16()
            .then(|| self.specialized_functions.get(symbol))
            .flatten()
    }

    pub(super) fn sm120_kernel_resources(
        &self,
        symbol: &str,
    ) -> Option<super::contract::Sm120KernelResources> {
        self.has_sm120_tma_mma16()
            .then(|| self.sm120_resources.get(symbol).copied())
            .flatten()
    }

    pub(super) fn prepare_tf32_tensor_maps(
        &self,
        request: super::contract::F32TriadRequest,
        route: super::contract::Tf32PhysicalRoute,
        plan: super::contract::Tf32TensorMapPlan,
        capturing: bool,
        binding: super::contract::Tf32MapBinding,
    ) -> Result<super::contract::F32PreparedTensorMaps, String> {
        let expected = match route.module_kind() {
            ModuleKind::TriadSm80 => self.f32_triad_availability.portable,
            ModuleKind::TriadSm89Finalist => self.f32_triad_availability.finalist,
            ModuleKind::TriadSm89Tf32Joint => self.f32_triad_availability.joint,
            _ => self.f32_triad_availability.specialized,
        };
        if binding.allocation_domain != self.allocation_domain
            || expected != Some(binding.qualified)
            || binding.qualified.module_kind != route.module_kind()
            || binding.qualified.artifact.module_kind != route.module_kind()
        {
            return Err("TF32 tensor-map binding does not match its CUDA module context".into());
        }
        let cache_key = (plan.keys, plan.allocations, request, route);
        let mut cache = self
            .tf32_tensor_maps
            .lock()
            .map_err(|_| "TF32 tensor-map cache is poisoned".to_string())?;
        cache.retain(|(keys, allocations, _, cached_route), _| {
            *cached_route != route || *keys != plan.keys || *allocations == plan.allocations
        });
        if let Some(maps) = cache.get(&cache_key) {
            return Ok(maps.clone());
        }
        if capturing {
            return Err("TF32 tensor-map cache miss during graph capture".into());
        }
        let maps = super::contract::encode_tf32_tensor_maps(plan, request, route, binding)?;
        cache.insert(cache_key, maps.clone());
        Ok(maps)
    }

    pub(super) fn prepare_sm90a_tensor_maps(
        &self,
        request: super::contract::Sm90aMapRequest,
        keys: [super::contract::Sm90aTensorMapKey; 2],
        allocations: [super::contract::Sm90aAllocationIdentity; 2],
        capturing: bool,
        binding: super::contract::Sm90aMapBinding,
    ) -> Result<super::contract::Sm90aPreparedTensorMaps, String> {
        if binding.allocation_domain != self.allocation_domain
            || Some(binding.compiler) != self.sm90a_compiler_identity()
            || Some(binding.artifact) != self.artifact_set_identity.specialized
        {
            return Err("SM90a tensor-map binding does not match its CUDA module context".into());
        }
        let mut cache = self
            .sm90a_tensor_maps
            .lock()
            .map_err(|_| "SM90a tensor-map cache is poisoned".to_string())?;
        let dtype = match request.dtype {
            super::super::dtype::WeightDtype::F32 | super::super::dtype::WeightDtype::Tf32 => 0,
            super::super::dtype::WeightDtype::F16 => 1,
            super::super::dtype::WeightDtype::Bf16 => 2,
        };
        let cache_key = (keys, allocations, request.op, dtype, request.shape);
        cache.retain(|(cached_keys, cached_allocations, _, _, _), _| {
            *cached_keys != keys || *cached_allocations == allocations
        });
        if let Some(maps) = cache.get(&cache_key) {
            return Ok(*maps);
        }
        if capturing {
            return Err("SM90a tensor-map cache miss during graph capture".into());
        }
        let maps = super::contract::encode_sm90a_tensor_maps(keys, request, binding, allocations)?;
        cache.insert(cache_key, maps);
        Ok(maps)
    }

    pub(super) fn prepare_sm100_tensor_maps(
        &self,
        request: super::contract::Sm100MapRequest,
        keys: [super::contract::Sm90aTensorMapKey; 2],
        allocations: [super::contract::Sm90aAllocationIdentity; 2],
        origins: super::contract::Sm100TensorOrigins,
        capturing: bool,
        binding: super::contract::Sm100MapBinding,
    ) -> Result<super::contract::Sm100PreparedTensorMaps, String> {
        if binding.allocation_domain != self.allocation_domain
            || Some(binding.compiler) != self.sm100_compiler_identity()
            || Some(binding.artifact) != self.artifact_set_identity.specialized
            || Some(binding.target) != self.sm100_target_candidate()
        {
            return Err("SM100 tensor-map binding does not match its CUDA module context".into());
        }
        let mut cache = self
            .sm100_tensor_maps
            .lock()
            .map_err(|_| "SM100 tensor-map cache is poisoned".to_string())?;
        let dtype = match request.dtype {
            super::super::dtype::WeightDtype::F32 | super::super::dtype::WeightDtype::Tf32 => 0,
            super::super::dtype::WeightDtype::F16 => 1,
            super::super::dtype::WeightDtype::Bf16 => 2,
        };
        let cache_key = (
            keys,
            allocations,
            origins,
            request.op,
            dtype,
            request.tile,
            request.shape,
        );
        cache.retain(|(cached_keys, cached_allocations, _, _, _, _, _), _| {
            *cached_keys != keys || *cached_allocations == allocations
        });
        if let Some(maps) = cache.get(&cache_key) {
            return Ok(*maps);
        }
        if capturing {
            return Err("SM100 tensor-map cache miss during graph capture".into());
        }
        let maps = super::contract::encode_sm100_tensor_maps(
            keys,
            request,
            binding,
            allocations,
            origins,
        )?;
        cache.insert(cache_key, maps);
        Ok(maps)
    }

    pub(super) fn prepare_sm120_tensor_maps(
        &self,
        request: super::contract::Sm120MapRequest,
        keys: [super::contract::Sm120TensorMapKey; 2],
        allocations: [super::contract::Sm90aAllocationIdentity; 2],
        origins: super::contract::Sm120TensorOrigins,
        capturing: bool,
        binding: super::contract::Sm120MapBinding,
    ) -> Result<super::contract::Sm120PreparedTensorMaps, String> {
        if binding.allocation_domain != self.allocation_domain
            || Some(binding.compiler) != self.sm120_compiler_identity()
            || Some(binding.artifact) != self.artifact_set_identity.specialized
            || binding.target.nvrtc_arch != binding.compiler.target.as_str()
        {
            return Err("SM120 tensor-map binding does not match its CUDA module context".into());
        }
        let mut cache = self
            .sm120_tensor_maps
            .lock()
            .map_err(|_| "SM120 tensor-map cache is poisoned".to_string())?;
        let dtype = match request.dtype {
            super::super::dtype::WeightDtype::F32 | super::super::dtype::WeightDtype::Tf32 => 0,
            super::super::dtype::WeightDtype::F16 => 1,
            super::super::dtype::WeightDtype::Bf16 => 2,
        };
        let cache_key = (
            keys,
            allocations,
            origins,
            request.op,
            dtype,
            request.tile,
            request.bk,
            request.shape,
        );
        cache.retain(|(cached_keys, cached_allocations, ..), _| {
            *cached_keys != keys || *cached_allocations == allocations
        });
        if let Some(maps) = cache.get(&cache_key) {
            return Ok(*maps);
        }
        if capturing {
            return Err("SM120 tensor-map cache miss during graph capture".into());
        }
        let maps = super::contract::encode_sm120_tensor_maps(
            keys,
            request,
            binding,
            allocations,
            origins,
        )?;
        cache.insert(cache_key, maps);
        Ok(maps)
    }

    pub fn splitk_scratch_buf(&self, stream: &Arc<CudaStream>) -> Result<&CudaSlice<f32>, String> {
        if self.splitk_scratch.get().is_none() {
            let buffer = stream
                .alloc_zeros::<f32>(1 << 23)
                .map_err(|error| format!("splitk_scratch alloc: {error:?}"))?;
            let _ = self.splitk_scratch.set(buffer);
        }
        self.splitk_scratch
            .get()
            .ok_or_else(|| "splitk_scratch cell empty after init".to_string())
    }

    pub fn tf32_splitk_counter_buf(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<&CudaSlice<u32>, String> {
        if self.tf32_splitk_counters.get().is_none() {
            let buffer = stream
                .alloc_zeros::<u32>(super::dispatch::TF32_SPLITK_COUNTER_CAP)
                .map_err(|error| format!("tf32_splitk_counters alloc: {error:?}"))?;
            let _ = self.tf32_splitk_counters.set(buffer);
        }
        self.tf32_splitk_counters
            .get()
            .ok_or_else(|| "tf32_splitk_counters cell empty after init".to_string())
    }

    pub fn transpose_scratch_buf(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<&CudaSlice<f32>, String> {
        if self.transpose_scratch.get().is_none() {
            let buffer = GpuBuffer::zeros(
                stream,
                super::contract::SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS,
            )
            .map_err(|error| format!("transpose_scratch alloc: {error}"))?;
            let _ = self.transpose_scratch.set(buffer);
        }
        self.transpose_scratch
            .get()
            .map(GpuBuffer::inner)
            .ok_or_else(|| "transpose_scratch cell empty after init".to_string())
    }
}

fn load_owned_function(
    name: &str,
    scalar: &Arc<CudaModule>,
    sm80: &Arc<CudaModule>,
) -> Result<CudaFunction, String> {
    resolve_owned_symbol(
        name,
        |symbol| load_function(scalar, ModuleKind::TriadScalar, symbol),
        |symbol| load_function(sm80, ModuleKind::TriadSm80, symbol),
    )
}

fn load_function(
    module: &Arc<CudaModule>,
    kind: ModuleKind,
    name: &str,
) -> Result<CudaFunction, String> {
    module
        .load_function(name)
        .map_err(|error| format!("{kind:?} kernel {name} not found: {error:?}"))
}

type Tf32LoadedFunctions = (
    HashMap<&'static str, CudaFunction>,
    Vec<Tf32SymbolExclusion>,
);

fn load_tf32_functions(module: &CompiledModule) -> Result<Tf32LoadedFunctions, String> {
    let module_kind = module.artifact_identity.module_kind;
    let extensions = module_kind == ModuleKind::TriadSm80
        && sm80_target_composes_streamk(module.compiler_identity.target.as_str());
    let specs: Vec<_> = super::contract::tf32_route_specs_for(module_kind, extensions).collect();
    if specs.is_empty() {
        return Err(format!("{module_kind:?} has no TF32 symbol inventory"));
    }
    let mut functions = HashMap::with_capacity(specs.len());
    let mut excluded = Vec::new();
    for kernel_spec in &specs {
        let function = load_function(&module.module, module_kind, kernel_spec.symbol)?;
        let shared = i32::try_from(kernel_spec.dynamic_shared_bytes)
            .map_err(|_| format!("{} shared memory exceeds i32::MAX", kernel_spec.symbol))?;
        set_dynamic_shared(&function, kernel_spec.symbol, shared)?;
        let local_bytes =
            u32::try_from(function.local_size_bytes().map_err(|error| {
                format!("query {} local memory: {error:?}", kernel_spec.symbol)
            })?)
            .map_err(|_| format!("{} returned negative local memory", kernel_spec.symbol))?;
        let static_shared_bytes = if module_kind == ModuleKind::TriadSm89Finalist {
            Some(
                u32::try_from(function.shared_size_bytes().map_err(|error| {
                    format!(
                        "query {} static shared memory: {error:?}",
                        kernel_spec.symbol
                    )
                })?)
                .map_err(|_| {
                    format!(
                        "{} returned negative static shared memory",
                        kernel_spec.symbol
                    )
                })?,
            )
        } else {
            None
        };
        if static_shared_bytes.is_some_and(|bytes| bytes != 0) {
            excluded.push(Tf32SymbolExclusion {
                symbol: kernel_spec.symbol,
                reason: format!(
                    "{} uses {} bytes of static shared memory, expected zero",
                    kernel_spec.symbol,
                    static_shared_bytes.unwrap_or_default()
                ),
            });
            continue;
        }
        let registers = u32::try_from(
            function
                .num_regs()
                .map_err(|error| format!("query {} registers: {error:?}", kernel_spec.symbol))?,
        )
        .map_err(|_| format!("{} returned a negative register count", kernel_spec.symbol))?;
        let register_cap = tf32_register_cap(module_kind, kernel_spec.symbol)?;
        let threads = i32::try_from(kernel_spec.threads)
            .map_err(|_| format!("{} thread count exceeds i32::MAX", kernel_spec.symbol))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|error| format!("query {} max threads: {error:?}", kernel_spec.symbol))?;
        if let Err(reason) = tf32_symbol_admission(
            kernel_spec.symbol,
            local_bytes,
            0,
            registers,
            register_cap,
            max_threads,
            threads,
        ) {
            excluded.push(Tf32SymbolExclusion {
                symbol: kernel_spec.symbol,
                reason,
            });
            continue;
        }
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                kernel_spec.threads,
                kernel_spec.dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("query {} occupancy: {error:?}", kernel_spec.symbol))?;
        let required_occupancy = tf32_required_occupancy(module_kind, kernel_spec.symbol);
        if occupancy < required_occupancy {
            excluded.push(Tf32SymbolExclusion {
                symbol: kernel_spec.symbol,
                reason: format!(
                    "{} occupancy {occupancy} misses its {required_occupancy}-CTA gate",
                    kernel_spec.symbol
                ),
            });
            continue;
        }
        if functions.insert(kernel_spec.symbol, function).is_some() {
            return Err(format!("duplicate TF32 function {}", kernel_spec.symbol));
        }
    }
    if functions.len() + excluded.len() != specs.len() {
        return Err(format!(
            "{module_kind:?} did not load its complete TF32 inventory"
        ));
    }
    if functions.is_empty() {
        return Err(format!(
            "{module_kind:?} has no TF32 kernel this toolkit can serve: {}",
            excluded
                .iter()
                .map(|exclusion| exclusion.reason.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    Ok((functions, excluded))
}

fn sm120_fma_exclusions(exclusions: &[Tf32SymbolExclusion]) -> super::contract::Sm120FmaExclusions {
    let routes = super::contract::SM120_FMA_ROUTE_SPECS
        .iter()
        .filter(|spec| {
            exclusions
                .iter()
                .any(|exclusion| exclusion.symbol == spec.symbol)
        })
        .map(|spec| {
            (
                spec.op,
                spec.route
                    .exact_fma()
                    .expect("SM120 exact-F32 inventory contains only exact routes"),
            )
        })
        .collect::<Vec<_>>();
    super::contract::Sm120FmaExclusions::from_routes(&routes)
        .expect("SM120 exact-F32 exclusions come from the literal inventory")
}

fn load_tf32_splitk_functions(module: &CompiledModule) -> Result<Tf32LoadedFunctions, String> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm80 {
        return Err("portable TF32 split-K requires the TriadSm80 module".into());
    }
    if !module.tf32_qualified {
        return Ok((HashMap::new(), Vec::new()));
    }
    let extensions = module_composes_extensions(
        ModuleKind::TriadSm80,
        module.compiler_identity.target.as_str(),
    );
    let specs: Vec<_> = super::contract::tf32_splitk_specs_for(extensions).collect();
    let mut functions = HashMap::with_capacity(specs.len());
    let mut excluded = Vec::new();
    for spec in &specs {
        let (symbol, threads, dynamic_shared_bytes, register_cap, occupancy_gate) = (
            spec.symbol,
            spec.threads,
            spec.dynamic_shared_bytes,
            spec.register_cap,
            spec.occupancy_gate,
        );
        module.tf32_driver_abi.get(symbol).ok_or_else(|| {
            format!("{symbol} has no live CUDA Driver parameter ABI census entry")
        })?;
        let function = load_function(&module.module, ModuleKind::TriadSm80, symbol)?;
        set_dynamic_shared(
            &function,
            symbol,
            i32::try_from(dynamic_shared_bytes)
                .map_err(|_| format!("{symbol} shared memory exceeds i32::MAX"))?,
        )?;
        let local_bytes = u32::try_from(
            function
                .local_size_bytes()
                .map_err(|error| format!("query {symbol} local memory: {error:?}"))?,
        )
        .map_err(|_| format!("{symbol} returned negative local memory"))?;
        let registers = u32::try_from(
            function
                .num_regs()
                .map_err(|error| format!("query {symbol} registers: {error:?}"))?,
        )
        .map_err(|_| format!("{symbol} returned a negative register count"))?;
        let threads_i32 = i32::try_from(threads)
            .map_err(|_| format!("{symbol} thread count exceeds i32::MAX"))?;
        let max_threads = function
            .max_threads_per_block()
            .map_err(|error| format!("query {symbol} max threads: {error:?}"))?;
        if let Err(reason) = tf32_symbol_admission(
            symbol,
            local_bytes,
            0,
            registers,
            register_cap,
            max_threads,
            threads_i32,
        ) {
            excluded.push(Tf32SymbolExclusion { symbol, reason });
            continue;
        }
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                threads,
                dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("query {symbol} occupancy: {error:?}"))?;
        if occupancy < occupancy_gate {
            excluded.push(Tf32SymbolExclusion {
                symbol,
                reason: format!(
                    "{symbol} occupancy {occupancy} misses its {occupancy_gate}-CTA gate"
                ),
            });
            continue;
        }
        if functions.insert(symbol, function).is_some() {
            return Err(format!("duplicate TF32 split-K function {symbol}"));
        }
    }
    if functions.len() + excluded.len() != specs.len() {
        return Err("TriadSm80 did not load its complete TF32 split-K inventory".into());
    }
    Ok((functions, excluded))
}

fn tf32_required_occupancy(module_kind: ModuleKind, symbol: &str) -> u32 {
    if module_kind == ModuleKind::TriadSm89Finalist {
        2
    } else if module_kind == ModuleKind::TriadSm90a && symbol.ends_with("_wg1") {
        3
    } else {
        1
    }
}

fn tf32_register_cap(module_kind: ModuleKind, symbol: &str) -> Result<u32, String> {
    const SM120_TAG33_SYMBOL: &str = "nn_sm120_tma_mma_tf32_m80n32_bk64_s2";

    match module_kind {
        // The compiled kernel uses 125 registers on CUDA 12.8 and 13.0 and
        // 121 on 13.2, with no local memory and two resident CTAs.
        ModuleKind::TriadSm89Finalist
            if symbol == super::sm89_finalist_source::SM89_FINALIST_SYMBOL =>
        {
            Ok(125)
        }
        ModuleKind::TriadSm80 if symbol.contains("_m128n64_") => Ok(192),
        // The wide tile holds the same 64-accumulator microtile per thread as
        // the 128x64 body plus a second fragment set, on all eight warps and
        // one CTA per multiprocessor.
        ModuleKind::TriadSm80 if symbol.contains("_m128n128_") => Ok(224),
        ModuleKind::TriadSm80 if symbol.contains("_m64n64_") => Ok(128),
        ModuleKind::TriadSm80 if symbol.contains("_m16n32_") || symbol.contains("_m16n16_") => {
            Ok(96)
        }
        ModuleKind::TriadSm90a if symbol.ends_with("_wg1") => Ok(168),
        ModuleKind::TriadSm90a if symbol.ends_with("_wg2") => Ok(128),
        ModuleKind::TriadSm120 if symbol == SM120_TAG33_SYMBOL => Ok(80),
        // The stream-K kernel keeps one CTA per multiprocessor by design and
        // spends the register file on its pipeline state and boundary code.
        ModuleKind::TriadSm120 if symbol.ends_with("_pair_streamk") => Ok(240),
        // The exact FMA routes hold a 64-accumulator microtile; the NT
        // k-vector arms also keep a float4 B fragment per column.
        ModuleKind::TriadSm120 if symbol.contains("_tma_fma_") && symbol.ends_with("_kvec") => {
            Ok(super::contract::SM120_FMA_KVEC_REGISTER_CAP)
        }
        ModuleKind::TriadSm120 if symbol.contains("_tma_fma_") => {
            Ok(super::contract::SM120_FMA_REGISTER_CAP)
        }
        ModuleKind::TriadSm100 | ModuleKind::TriadSm120 => Ok(128),
        _ => Err(format!(
            "no TF32 register gate for {module_kind:?}/{symbol}"
        )),
    }
}

#[derive(Clone, Copy)]
struct Tf32DriverJitLocalMemoryFacts<'a> {
    module_kind: ModuleKind,
    symbol: &'a str,
    target: CudaTarget,
    nvrtc_version: (i32, i32),
    nvrtc_library_known: bool,
    nvrtc_library_current: bool,
}

impl<'a> Tf32DriverJitLocalMemoryFacts<'a> {
    fn from_compiler(module_kind: ModuleKind, symbol: &'a str, compiler: CompilerIdentity) -> Self {
        let current_domain = crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain();
        let current_digest = FramedSha256::new(b"nvrtc-library-set-identity.v2")
            .optional(b"domain", current_domain.as_deref())
            .finish();
        Self {
            module_kind,
            symbol,
            target: compiler.target,
            nvrtc_version: compiler.nvrtc_version,
            nvrtc_library_known: compiler.nvrtc_library_known,
            nvrtc_library_current: current_domain.is_some()
                && current_digest == compiler.nvrtc_library_domain,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Tf32DriverJitLocalMemoryAdmission {
    pub observed_bytes: u32,
    pub approved_cap_bytes: u32,
}

pub(super) fn validate_tf32_driver_jit_local_memory(
    local_bytes: u32,
    module_kind: ModuleKind,
    symbol: &str,
    compiler: CompilerIdentity,
) -> Result<Tf32DriverJitLocalMemoryAdmission, String> {
    validate_tf32_driver_jit_local_memory_facts(
        local_bytes,
        Tf32DriverJitLocalMemoryFacts::from_compiler(module_kind, symbol, compiler),
    )
}

fn validate_tf32_driver_jit_local_memory_facts(
    local_bytes: u32,
    facts: Tf32DriverJitLocalMemoryFacts<'_>,
) -> Result<Tf32DriverJitLocalMemoryAdmission, String> {
    if local_bytes == 0 {
        return Ok(Tf32DriverJitLocalMemoryAdmission {
            observed_bytes: 0,
            approved_cap_bytes: 0,
        });
    }
    Err(format!(
        "{} uses {local_bytes} bytes of Driver JIT local memory; the current module requires zero: module={:?}, target={}, NVRTC={}.{}, library-known={}, library-current={}",
        facts.symbol,
        facts.module_kind,
        facts.target.as_str(),
        facts.nvrtc_version.0,
        facts.nvrtc_version.1,
        facts.nvrtc_library_known,
        facts.nvrtc_library_current
    ))
}

fn qualify_loaded_tf32_artifact(
    ctx: &Arc<CudaContext>,
    allocation_domain: super::contract::AllocationDomain,
    module: &CompiledModule,
    binding: super::contract::Tf32QualifiedModule,
    functions: &HashMap<&'static str, CudaFunction>,
) -> Result<Sha256Digest, String> {
    let stream = ctx.default_stream();
    let capture_status = stream
        .capture_status()
        .map_err(|error| format!("query TF32 artifact probe capture status: {error:?}"))?;
    if capture_status != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        return Err("TF32 artifact probe cannot run during CUDA stream capture".into());
    }

    let module_kind = module.artifact_identity.module_kind;
    if binding.module_kind != module_kind || binding.artifact != module.artifact_identity {
        return Err("TF32 artifact probe binding does not match its compiled module".into());
    }
    let finalist = module_kind == ModuleKind::TriadSm89Finalist;
    let spec = if finalist {
        super::contract::tf32_kernel_spec(
            ResolvedGemmOp::Nt,
            super::contract::Tf32PhysicalRoute::Sm89MmaTf32Compact8,
        )?
    } else {
        super::contract::tf32_route_specs(module_kind)
            .iter()
            .find(|spec| spec.op == ResolvedGemmOp::Nn)
            .ok_or_else(|| format!("{module_kind:?} has no NN TF32 artifact probe route"))?
    };
    let function = functions.get(spec.symbol).ok_or_else(|| {
        format!(
            "{module_kind:?} TF32 artifact probe function {} is unavailable",
            spec.symbol
        )
    })?;

    let result = (|| -> Result<Sha256Digest, String> {
        let a_host = vec![0x3f800000_u32, 0, 0, 0];
        let (b_host, request) = if finalist {
            let mut b = vec![0_u32; TF32_EXCEPTIONAL_PROBE_BITS.len() * 4];
            for (index, bits) in TF32_EXCEPTIONAL_PROBE_BITS.iter().copied().enumerate() {
                b[index * 4] = bits;
            }
            let len = TF32_EXCEPTIONAL_PROBE_BITS.len();
            (
                b,
                super::contract::F32TriadRequest {
                    op: ResolvedGemmOp::Nt,
                    shape: super::contract::F32TriadShape {
                        m: 1,
                        k: len,
                        n: 1,
                        lda: 4,
                        ldb: 4,
                        ldc: len,
                    },
                },
            )
        } else {
            let mut b = vec![0_u32; 12];
            b[..TF32_EXCEPTIONAL_PROBE_BITS.len()].copy_from_slice(&TF32_EXCEPTIONAL_PROBE_BITS);
            (
                b,
                super::contract::F32TriadRequest {
                    op: ResolvedGemmOp::Nn,
                    shape: super::contract::F32TriadShape {
                        m: 1,
                        k: 1,
                        n: TF32_EXCEPTIONAL_PROBE_BITS.len(),
                        lda: 4,
                        ldb: 12,
                        ldc: TF32_EXCEPTIONAL_PROBE_BITS.len(),
                    },
                },
            )
        };
        let a = stream
            .clone_htod(&a_host)
            .map_err(|error| format!("allocate TF32 artifact probe A: {error:?}"))?;
        let b = stream
            .clone_htod(&b_host)
            .map_err(|error| format!("allocate TF32 artifact probe B: {error:?}"))?;
        let mut output = stream
            .alloc_zeros::<u32>(TF32_EXCEPTIONAL_PROBE_BITS.len())
            .map_err(|error| format!("allocate TF32 artifact probe output: {error:?}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("initialize TF32 artifact probe allocations: {error:?}"))?;

        let (a_ptr, a_guard) = a.device_ptr(&stream);
        let (b_ptr, b_guard) = b.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let operands = super::contract::F32TriadOperands {
            output: output_ptr,
            a: a_ptr,
            b: b_ptr,
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        let maps = if matches!(
            module_kind,
            ModuleKind::TriadSm80 | ModuleKind::TriadSm89Finalist
        ) {
            None
        } else {
            let map_binding = super::contract::Tf32MapBinding {
                allocation_domain,
                qualified: binding,
            };
            let plan = super::contract::tf32_tensor_map_plan(
                request,
                operands,
                spec.route,
                allocation_domain,
            )?;
            Some(super::contract::encode_tf32_tensor_maps(
                plan,
                request,
                spec.route,
                map_binding,
            )?)
        };
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (spec.threads, 1, 1),
            shared_mem_bytes: spec.dynamic_shared_bytes,
        };
        const CANARIES: [u32; 2] = [0x3f123456, 0xbf654321];
        let next_run = Cell::new(0_usize);
        let active_canary = Cell::new(None::<u32>);
        let qualification = qualify_tf32_conversion_artifact(
            module.artifact_identity,
            || {
                let run = next_run.get();
                let canary = CANARIES
                    .get(run)
                    .copied()
                    .ok_or_else(|| "TF32 artifact probe launched more than twice".to_string())?;
                let canary_words = [canary; TF32_EXCEPTIONAL_PROBE_BITS.len()];
                cu_memcpy_htod_raw(&stream, output_ptr, bytemuck::cast_slice(&canary_words))?;
                unsafe {
                    super::launch::enqueue_tf32_qualification_probe(
                        &stream,
                        function,
                        request,
                        operands,
                        spec.route,
                        maps.as_ref(),
                        config,
                    )
                }?;
                active_canary.set(Some(canary));
                next_run.set(run + 1);
                Ok(())
            },
            || {
                let canary = active_canary.get().ok_or_else(|| {
                    "TF32 artifact probe download has no active launch".to_string()
                })?;
                let mut words = [0_u32; TF32_EXCEPTIONAL_PROBE_BITS.len()];
                cu_memcpy_dtoh_raw(&stream, output_ptr, bytemuck::cast_slice_mut(&mut words))?;
                if let Some(index) = words.iter().position(|&bits| bits == canary) {
                    return Err(format!(
                        "TF32 artifact probe output slot {index} retained its canary"
                    ));
                }
                active_canary.set(None);
                Ok(words.to_vec())
            },
        );
        drop(maps);
        drop(output_guard);
        drop(b_guard);
        drop(a_guard);
        qualification
    })();
    let cleanup = stream
        .synchronize()
        .map_err(|error| format!("TF32 artifact probe cleanup: {error:?}"));
    match (result, cleanup) {
        (Ok(digest), Ok(())) => Ok(digest),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(error), Err(cleanup)) => Err(format!("{error}; {cleanup}")),
    }
}

fn qualify_tf32_module_binding(
    ctx: &Arc<CudaContext>,
    module: &CompiledModule,
) -> Result<super::contract::Tf32QualifiedModule, String> {
    let module_kind = module.artifact_identity.module_kind;
    let compiler_target = module.compiler_identity.target.as_str();
    let (major, minor) = ctx
        .compute_capability()
        .map_err(|error| format!("query TF32 compute capability: {error:?}"))?;
    let device_cc = (
        u32::try_from(major).map_err(|_| format!("negative CUDA CC major {major}"))?,
        u32::try_from(minor).map_err(|_| format!("negative CUDA CC minor {minor}"))?,
    );
    let ptx_target = qualified_ptx_target(module_kind, compiler_target, (major, minor))?;
    let optin_shared = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .map_err(|error| format!("query TF32 opt-in shared memory: {error:?}"))?;
    let tensor_map_access = match ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
    ) {
        Ok(value) => value != 0,
        Err(_)
            if matches!(
                module_kind,
                ModuleKind::TriadSm80
                    | ModuleKind::TriadSm89Finalist
                    | ModuleKind::TriadSm89Tf32Joint
            ) =>
        {
            false
        }
        Err(error) => return Err(format!("query TF32 tensor-map support: {error:?}")),
    };
    if !matches!(
        module_kind,
        ModuleKind::TriadSm80 | ModuleKind::TriadSm89Finalist | ModuleKind::TriadSm89Tf32Joint
    ) && !tensor_map_access
    {
        return Err(format!("{module_kind:?} requires tensor-map access"));
    }
    let target = CudaTarget::new(compiler_target)?;
    let multiprocessor_count = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .map_err(|error| format!("query TF32 multiprocessor count: {error:?}"))?;
    let multiprocessor_count = u32::try_from(multiprocessor_count)
        .map_err(|_| format!("negative TF32 multiprocessor count {multiprocessor_count}"))?;
    if multiprocessor_count == 0 {
        return Err("CUDA device reported zero multiprocessors".into());
    }
    let binding = super::contract::Tf32QualifiedModule {
        module_kind,
        target,
        artifact: module.artifact_identity,
        compiler: module.compiler_identity,
        device: crate::mamba_ssm::gpu::kernel_identity::DeviceIdentity {
            compute_capability: device_cc,
            multiprocessor_count,
            target: CudaTarget::new(ptx_target)?,
            driver: crate::mamba_ssm::gpu::kernel_identity::query_driver_identity()?,
        },
        device_caps: crate::mamba_ssm::gpu::kernel_identity::DeviceCaps {
            compute_capability: device_cc,
            nvrtc_version: module.compiler_identity.nvrtc_version,
            accepted_target: Some(target),
            optin_shared_bytes: u32::try_from(optin_shared)
                .map_err(|_| format!("negative TF32 opt-in shared memory {optin_shared}"))?,
            tensor_map_access,
        },
        sm120_fma_exclusions: Default::default(),
    };
    Ok(binding)
}

fn qualified_ptx_target(
    module_kind: ModuleKind,
    compiler_target: &str,
    device_cc: (i32, i32),
) -> Result<&'static str, String> {
    match module_kind {
        ModuleKind::TriadSm80 => {
            let expected = portable_target_for_device(device_cc)?;
            let actual = sm80_ptx_target(compiler_target)
                .ok_or_else(|| format!("TriadSm80 target {compiler_target} is not admitted"))?;
            (actual == expected || (device_cc == (12, 1) && actual == "sm_120"))
                .then_some(actual)
                .ok_or_else(|| {
                    format!("TriadSm80 target {compiler_target} does not own CC {device_cc:?}")
                })
        }
        ModuleKind::TriadSm89Finalist | ModuleKind::TriadSm89Tf32Joint => {
            let expected = portable_target_for_device(device_cc)?;
            let actual = sm80_ptx_target(compiler_target).ok_or_else(|| {
                format!("{module_kind:?} target {compiler_target} is not admitted")
            })?;
            (actual == expected || (device_cc == (12, 1) && actual == "sm_120"))
                .then_some(actual)
                .ok_or_else(|| {
                    format!(
                        "{module_kind:?} target {compiler_target} does not own CC {device_cc:?}"
                    )
                })
        }
        ModuleKind::TriadSm90a if device_cc == (9, 0) && compiler_target == "sm_90a" => {
            Ok("sm_90a")
        }
        ModuleKind::TriadSm100 => sm100_target_for_arch(compiler_target)
            .filter(|candidate| candidate.device_cc == device_cc)
            .map(|candidate| candidate.ptx_target)
            .ok_or_else(|| {
                format!("TriadSm100 target {compiler_target} does not own CC {device_cc:?}")
            }),
        ModuleKind::TriadSm120 => {
            super::dispatch::sm120_target_candidates(device_cc, nvrtc_version())
                .iter()
                .find(|candidate| candidate.nvrtc_arch == compiler_target)
                .map(|candidate| candidate.ptx_target)
                .ok_or_else(|| {
                    format!("TriadSm120 target {compiler_target} does not own CC {device_cc:?}")
                })
        }
        _ => Err(format!("{module_kind:?} is not a qualified TF32 module")),
    }
}

fn portable_target_for_device(device_cc: (i32, i32)) -> Result<&'static str, String> {
    match device_cc {
        (8, 0) => Ok("sm_80"),
        (8, 6) => Ok("sm_86"),
        (8, 7) => Ok("sm_87"),
        (8, 9) => Ok("sm_89"),
        (9, 0) => Ok("sm_90a"),
        (10, 0) => Ok("sm_100a"),
        (10, 1) => Ok("sm_101a"),
        (10, 3) => Ok("sm_103a"),
        (10, 7) => Ok("sm_107a"),
        (11, 0) => Ok("sm_110a"),
        (12, 0) => Ok("sm_120"),
        (12, 1) => Ok("sm_121"),
        _ => Err(format!("no portable TF32 target for CC {device_cc:?}")),
    }
}

fn load_sm90a_functions(
    module: &CompiledModule,
) -> Result<HashMap<&'static str, CudaFunction>, String> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm90a
        || module.compiler_identity.target.as_str() != "sm_90a"
    {
        return Err("specialized triad module is not exact-target TriadSm90a".into());
    }
    let mut functions = HashMap::new();
    for &symbol in SM90A_SYMBOLS {
        let function = load_function(&module.module, ModuleKind::TriadSm90a, symbol)?;
        set_dynamic_shared(
            &function,
            symbol,
            super::contract::SM90A_DYNAMIC_SHARED_BYTES as i32,
        )?;
        if function
            .local_size_bytes()
            .map_err(|error| format!("query {symbol} local memory: {error:?}"))?
            != 0
        {
            return Err(format!("{symbol} spills to local memory"));
        }
        let wg2 = symbol.contains("_wg2_");
        let threads = if wg2 { 256 } else { 128 };
        if function
            .max_threads_per_block()
            .map_err(|error| format!("query {symbol} max threads: {error:?}"))?
            < threads
        {
            return Err(format!("{symbol} cannot launch {threads} threads"));
        }
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                threads as u32,
                super::contract::SM90A_DYNAMIC_SHARED_BYTES as usize,
                None,
            )
            .map_err(|error| format!("query {symbol} occupancy: {error:?}"))?;
        if occupancy < if wg2 { 1 } else { 3 } {
            return Err(format!(
                "{symbol} occupancy {occupancy} misses its schedule gate"
            ));
        }
        if !wg2
            && function
                .occupancy_available_dynamic_smem_per_block(3, 128)
                .map_err(|error| format!("query {symbol} shared-memory capacity: {error:?}"))?
                < super::contract::SM90A_DYNAMIC_SHARED_BYTES as usize
        {
            return Err(format!("{symbol} cannot sustain three 73984-byte CTAs"));
        }
        functions.insert(symbol, function);
    }
    Ok(functions)
}

fn load_sm100_functions(
    module: &CompiledModule,
) -> Result<HashMap<&'static str, CudaFunction>, String> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm100
        || sm100_target_for_arch(module.compiler_identity.target.as_str()).is_none()
    {
        return Err("specialized triad module is not a valid TriadSm100 target".into());
    }
    let mut functions = HashMap::new();
    for spec in super::contract::SM100_KERNEL_SPECS {
        let function = load_function(&module.module, ModuleKind::TriadSm100, spec.symbol)?;
        let shared = i32::try_from(spec.dynamic_shared_bytes)
            .map_err(|_| format!("{} shared memory exceeds i32::MAX", spec.symbol))?;
        set_dynamic_shared(&function, spec.symbol, shared)?;
        if function
            .local_size_bytes()
            .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?
            != 0
        {
            return Err(format!("{} spills to local memory", spec.symbol));
        }
        let threads = i32::try_from(spec.threads)
            .map_err(|_| format!("{} thread count exceeds i32::MAX", spec.symbol))?;
        if function
            .max_threads_per_block()
            .map_err(|error| format!("query {} max threads: {error:?}", spec.symbol))?
            < threads
        {
            return Err(format!(
                "{} cannot launch {} threads",
                spec.symbol, spec.threads
            ));
        }
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                spec.threads,
                spec.dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
        if occupancy < 1 {
            return Err(format!("{} has zero launch occupancy", spec.symbol));
        }
        functions.insert(spec.symbol, function);
    }
    if functions.len() != super::contract::SM100_KERNEL_SPECS.len() {
        return Err("TriadSm100 did not load its complete symbol inventory".into());
    }
    Ok(functions)
}

fn load_sm120_functions(
    module: &CompiledModule,
) -> Result<HashMap<&'static str, CudaFunction>, String> {
    if module.artifact_identity.module_kind != ModuleKind::TriadSm120
        || sm120_ptx_target(module.compiler_identity.target.as_str()).is_none()
    {
        return Err("specialized triad module is not a valid TriadSm120 target".into());
    }
    let mut functions = HashMap::new();
    for spec in super::contract::sm120_kernel_specs() {
        let function = load_function(&module.module, ModuleKind::TriadSm120, spec.symbol)?;
        let shared = i32::try_from(spec.dynamic_shared_bytes)
            .map_err(|_| format!("{} shared memory exceeds i32::MAX", spec.symbol))?;
        set_dynamic_shared(&function, spec.symbol, shared)?;
        if function
            .local_size_bytes()
            .map_err(|error| format!("query {} local memory: {error:?}", spec.symbol))?
            != 0
        {
            return Err(format!("{} spills to local memory", spec.symbol));
        }
        let threads = i32::try_from(spec.threads)
            .map_err(|_| format!("{} thread count exceeds i32::MAX", spec.symbol))?;
        if function
            .max_threads_per_block()
            .map_err(|error| format!("query {} max threads: {error:?}", spec.symbol))?
            < threads
        {
            return Err(format!(
                "{} cannot launch {} threads",
                spec.symbol, spec.threads
            ));
        }
        let occupancy = function
            .occupancy_max_active_blocks_per_multiprocessor(
                spec.threads,
                spec.dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
        if occupancy < 1 {
            return Err(format!("{} has zero launch occupancy", spec.symbol));
        }
        functions.insert(spec.symbol, function);
    }
    if functions.len() != super::contract::sm120_kernel_specs().count() {
        return Err("TriadSm120 did not load its complete symbol inventory".into());
    }
    Ok(functions)
}

fn load_owned_half(
    base: &str,
    scalar: &Arc<CudaModule>,
    sm80: &Arc<CudaModule>,
) -> Result<HalfKernel, String> {
    Ok(HalfKernel {
        bf16: load_owned_function(&format!("{base}_bf16"), scalar, sm80)?,
        f16: load_owned_function(&format!("{base}_f16"), scalar, sm80)?,
    })
}

fn set_dynamic_shared(function: &CudaFunction, name: &str, bytes: i32) -> Result<(), String> {
    function
        .set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes,
        )
        .map_err(|error| format!("set MAX_DYNAMIC_SHARED for {name}: {error:?}"))
}

fn qualified_scalar_resource_environment(
    compute_capability: (u32, u32),
    multiprocessor_count: u32,
    compiler: CompilerIdentity,
    artifact: ArtifactIdentity,
) -> bool {
    compute_capability == (12, 0)
        && multiprocessor_count == 170
        && compiler.target.as_str() == "compute_120"
        && compiler.nvrtc_version == (13, 2)
        && compiler.nvrtc_library_known
        && compiler.nvrtc_library_domain != [0; 32]
        && compiler.invocation_digest != [0; 32]
        && artifact.module_kind == ModuleKind::TriadScalar
        && artifact.artifact_kind == compiler.output_kind
        && artifact.compile_key == compiler.invocation_digest
        && artifact.artifact_digest != [0; 32]
}

fn qualify_scalar_nt_d768_transpose(function: &CudaFunction) -> Result<(), String> {
    let registers = function
        .num_regs()
        .map_err(|error| format!("query d768 transpose registers: {error:?}"))?;
    let local = function
        .local_size_bytes()
        .map_err(|error| format!("query d768 transpose local bytes: {error:?}"))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|error| format!("query d768 transpose static shared bytes: {error:?}"))?;
    let active_blocks = function
        .occupancy_max_active_blocks_per_multiprocessor(
            super::contract::SCALAR_NT_D768_TRANSPOSE_THREADS,
            super::contract::SCALAR_NT_D768_TRANSPOSE_DYNAMIC_SHARED_BYTES,
            None,
        )
        .map_err(|error| format!("query d768 transpose occupancy: {error:?}"))?;
    if registers > super::contract::SCALAR_NT_D768_TRANSPOSE_REGISTER_CAP
        || local != 0
        || static_shared as usize != super::contract::SCALAR_NT_D768_TRANSPOSE_STATIC_SHARED_BYTES
        || active_blocks < super::contract::SCALAR_NT_D768_TRANSPOSE_MIN_ACTIVE_BLOCKS
    {
        return Err(format!(
            "d768 transpose resource qualification failed: registers={registers} local={local} static_shared={static_shared} active_blocks={active_blocks}"
        ));
    }
    Ok(())
}

fn qualify_scalar_nt_m2n16(function: &CudaFunction) -> Result<(), String> {
    let registers = function
        .num_regs()
        .map_err(|error| format!("query M2N16 registers: {error:?}"))?;
    let local = function
        .local_size_bytes()
        .map_err(|error| format!("query M2N16 local bytes: {error:?}"))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|error| format!("query M2N16 static shared bytes: {error:?}"))?;
    let active_blocks = function
        .occupancy_max_active_blocks_per_multiprocessor(
            super::contract::SCALAR_NT_M2N16_THREADS,
            super::contract::SCALAR_NT_M2N16_DYNAMIC_SHARED_BYTES as usize,
            None,
        )
        .map_err(|error| format!("query M2N16 occupancy: {error:?}"))?;
    if registers > super::contract::SCALAR_NT_M2N16_REGISTER_CAP
        || local != 0
        || static_shared as usize != super::contract::SCALAR_NT_M2N16_STATIC_SHARED_BYTES
        || active_blocks < super::contract::SCALAR_NT_M2N16_MIN_ACTIVE_BLOCKS
    {
        return Err(format!(
            "M2N16 resource qualification failed: registers={registers} local={local} static_shared={static_shared} active_blocks={active_blocks}"
        ));
    }
    Ok(())
}

fn qualify_scalar_nn_m32n64_splitk32(function: &CudaFunction) -> Result<(), String> {
    let registers = function
        .num_regs()
        .map_err(|error| format!("query NN M32N64 Split-K registers: {error:?}"))?;
    let local = function
        .local_size_bytes()
        .map_err(|error| format!("query NN M32N64 Split-K local bytes: {error:?}"))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|error| format!("query NN M32N64 Split-K static shared bytes: {error:?}"))?;
    let active_blocks = function
        .occupancy_max_active_blocks_per_multiprocessor(
            super::contract::SCALAR_NN_M32N64_SPLITK32_THREADS,
            super::contract::SCALAR_NN_M32N64_SPLITK32_DYNAMIC_SHARED_BYTES as usize,
            None,
        )
        .map_err(|error| format!("query NN M32N64 Split-K occupancy: {error:?}"))?;
    if registers > super::contract::SCALAR_NN_M32N64_SPLITK32_REGISTER_CAP
        || local != 0
        || static_shared as usize != super::contract::SCALAR_NN_M32N64_SPLITK32_STATIC_SHARED_BYTES
        || active_blocks < super::contract::SCALAR_NN_M32N64_SPLITK32_MIN_ACTIVE_BLOCKS
    {
        return Err(format!(
            "NN M32N64 Split-K resource qualification failed: registers={registers} local={local} static_shared={static_shared} active_blocks={active_blocks}"
        ));
    }
    Ok(())
}

fn qualify_scalar_tn_m16n16(function: &CudaFunction) -> Result<(), String> {
    let registers = function
        .num_regs()
        .map_err(|error| format!("query TN M16N16 registers: {error:?}"))?;
    let local = function
        .local_size_bytes()
        .map_err(|error| format!("query TN M16N16 local bytes: {error:?}"))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|error| format!("query TN M16N16 static shared bytes: {error:?}"))?;
    let active_blocks = function
        .occupancy_max_active_blocks_per_multiprocessor(
            super::contract::SCALAR_TN_M16N16_THREADS,
            super::contract::SCALAR_TN_M16N16_DYNAMIC_SHARED_BYTES as usize,
            None,
        )
        .map_err(|error| format!("query TN M16N16 occupancy: {error:?}"))?;
    if registers > super::contract::SCALAR_TN_M16N16_REGISTER_CAP
        || local != 0
        || static_shared as usize != super::contract::SCALAR_TN_M16N16_STATIC_SHARED_BYTES
        || active_blocks < super::contract::SCALAR_TN_M16N16_MIN_ACTIVE_BLOCKS
    {
        return Err(format!(
            "TN M16N16 resource qualification failed: registers={registers} local={local} static_shared={static_shared} active_blocks={active_blocks}"
        ));
    }
    Ok(())
}

fn set_half_dynamic_shared(kernel: &HalfKernel, name: &str, bytes: i32) -> Result<(), String> {
    set_dynamic_shared(&kernel.bf16, name, bytes)?;
    set_dynamic_shared(&kernel.f16, name, bytes)
}

/// The CTAs of the stream-K kernel one multiprocessor holds at once, the
/// Resident CTAs per multiprocessor of the half relay, the smaller of its
/// two dtypes. The relay's grid may never exceed what stays resident: a CTA
/// waits on the flag of the CTA below it, and a grid larger than the board
/// holds could leave that lower CTA unscheduled. Zero when the relay is not
/// loaded, which is also when no request can reach it.
fn sm89_half_relay_resident(
    functions: &HashMap<&'static str, CudaFunction>,
) -> Result<u32, String> {
    let mut resident = u32::MAX;
    for spec in super::sm89_half_source::runtime_kernel_specs()
        .filter(|spec| spec.schedule == super::sm89_half_source::Sm89HalfSchedule::Relay)
    {
        let Some(function) = functions.get(spec.symbol) else {
            return Ok(0);
        };
        let blocks = function
            .occupancy_max_active_blocks_per_multiprocessor(
                spec.threads,
                spec.dynamic_shared_bytes as usize,
                None,
            )
            .map_err(|error| format!("query {} occupancy: {error:?}", spec.symbol))?;
        resident = resident.min(blocks);
    }
    Ok(if resident == u32::MAX { 0 } else { resident })
}

/// smaller of its two half variants; the kernel's shared tiles are static,
/// so the query carries no dynamic bytes.
fn streamk_resident_ctas(kernel: &HalfKernel) -> Result<u32, String> {
    let mut resident = u32::MAX;
    for (name, function) in [("bf16", &kernel.bf16), ("f16", &kernel.f16)] {
        let blocks = function
            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
            .map_err(|error| format!("query tn_tc64_streamk_{name} occupancy: {error:?}"))?;
        resident = resident.min(blocks);
    }
    Ok(resident.max(1))
}

#[cfg(test)]
mod tests {
    use std::{
        cell::{Cell, RefCell},
        collections::{BTreeSet, HashMap, VecDeque},
    };

    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget as KernelCudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, ModuleKind,
        NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    };

    use super::{
        SCALAR_GROUP_M_MACRO, SCALAR_SYMBOLS as PRODUCTION_SCALAR_SYMBOLS,
        SM80_SYMBOLS as PRODUCTION_SM80_SYMBOLS, SM90A_SYMBOLS, SM100_PROBE_SOURCE, SourceFragment,
        TF32_DRIVER_PARAMETER_COUNT, Tf32DriverAbi, Tf32DriverJitLocalMemoryFacts,
        compose_fragments, compose_module_source, compose_module_source_for,
        merge_optional_finalist_driver_abi, merge_tf32_driver_abi, parse_ptx,
        portable_target_for_device, ptx_entry, qualified_ptx_target,
        qualified_scalar_resource_environment, qualify_tf32_conversion_artifact,
        query_tf32_driver_parameter_abi, resolve_owned_symbol, retain_forced_only_functions,
        retain_tf32_candidate, scalar_group_m_option, select_sm100_candidate,
        select_sm120_candidate, sm100_target_candidates, tf32_register_cap,
        tf32_required_occupancy, validate_exact_ptx_exports, validate_module_ptx,
        validate_module_target, validate_sm89_finalist_ptx_inventory, validate_sm90a_ptx,
        validate_sm100_probe_ptx, validate_sm100_ptx, validate_sm120_ptx,
        validate_tf32_driver_jit_local_memory_facts, validate_tf32_feature_instructions,
        validate_tf32_parameter_abi, validate_tf32_ptx_inventory, validate_tf32_splitk_ptx,
        validate_tn_narrow_splitm_partial_ptx,
    };

    fn digest(hex: &str) -> [u8; 32] {
        assert_eq!(hex.len(), 64);
        std::array::from_fn(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap())
    }

    #[test]
    fn triad_retained_identity_fixed_source_digest_uses_complete_compile_composition() {
        let mut moved = Vec::new();
        for (nvrtc, state_cap, expected) in [
            (
                (12, 8),
                16,
                "e46af1e315d323db6abf5ad17569367150b63205a9f75173aaf8081db45e0e90",
            ),
            (
                (12, 8),
                64,
                "73111d550cd33778cba12f27eb38cc8b30e417b40b6469f3f36ddc3815e0dcab",
            ),
            (
                (13, 0),
                16,
                "e46af1e315d323db6abf5ad17569367150b63205a9f75173aaf8081db45e0e90",
            ),
            (
                (13, 0),
                64,
                "73111d550cd33778cba12f27eb38cc8b30e417b40b6469f3f36ddc3815e0dcab",
            ),
            (
                (13, 2),
                16,
                "e46af1e315d323db6abf5ad17569367150b63205a9f75173aaf8081db45e0e90",
            ),
            (
                (13, 2),
                64,
                "73111d550cd33778cba12f27eb38cc8b30e417b40b6469f3f36ddc3815e0dcab",
            ),
        ] {
            let live = super::module_source_digest_for_compile(
                ModuleKind::Fixed,
                Some((8, 9)),
                "sm_89",
                state_cap,
                nvrtc,
            )
            .unwrap();
            if live != digest(expected) {
                moved.push(format!(
                    "CUDA {nvrtc:?} cap{state_cap}: {}",
                    crate::mamba_ssm::gpu::kernel_identity::digest_hex(&live)
                ));
            }
        }
        assert!(
            moved.is_empty(),
            "the Fixed compile composition moved; refreeze every cohort from the live values:\n{}",
            moved.join("\n")
        );
    }

    #[test]
    fn inference_source_bundle_appends_each_retained_export_once() {
        let symbols = [
            "nn_sm89_tc128_f32out_s3_bf16",
            "nn_sm89_tc128_f32out_s3_f16",
            "nn_sm89_f32_m128n64_tail_copyplan",
        ];
        for cap in [16, 64] {
            for nvrtc in [(12, 8), (13, 0), (13, 2)] {
                let base = compose_module_source_for(ModuleKind::Fixed, "sm_89").unwrap();
                let result =
                    crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compose_fixed_source(
                        base.clone(),
                        Some((8, 9)),
                        "sm_89",
                        cap,
                        nvrtc,
                    )
                    .unwrap();
                assert!(result.starts_with(&base));
                for symbol in symbols {
                    assert_eq!(result.matches(&format!("void {symbol}(")).count(), 1);
                }
            }
        }
    }

    #[test]
    fn inference_source_bundle_preserves_the_complete_input_outside_its_envelope() {
        let cases = [
            ("missing CC", None, "sm_89", 16, (12, 8), false),
            ("CC 7.5", Some((7, 5)), "sm_89", 16, (12, 8), false),
            ("CC 12.0", Some((12, 0)), "sm_89", 16, (12, 8), false),
            (
                "compute_89 target",
                Some((8, 9)),
                "compute_89",
                16,
                (12, 8),
                false,
            ),
            (
                "compute_120 target",
                Some((8, 9)),
                "compute_120",
                16,
                (12, 8),
                false,
            ),
            ("capacity 0", Some((8, 9)), "sm_89", 0, (12, 8), false),
            ("capacity 8", Some((8, 9)), "sm_89", 8, (12, 8), false),
            ("capacity 32", Some((8, 9)), "sm_89", 32, (12, 8), false),
            ("capacity 128", Some((8, 9)), "sm_89", 128, (12, 8), false),
            ("capacity 256", Some((8, 9)), "sm_89", 256, (12, 8), false),
            ("NVRTC 0.0", Some((8, 9)), "sm_89", 16, (0, 0), false),
            ("NVRTC 12.7", Some((8, 9)), "sm_89", 16, (12, 7), false),
            ("NVRTC 13.1", Some((8, 9)), "sm_89", 16, (13, 1), false),
            ("NVRTC 13.3", Some((8, 9)), "sm_89", 16, (13, 3), false),
        ];

        for (name, device_cc, target, state_cap, nvrtc, expected_supported) in cases {
            assert_eq!(
                crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compiler_supported(
                    device_cc, target, state_cap, nvrtc,
                ),
                expected_supported,
                "wrong compiler support for {name}"
            );
            let input = compose_module_source_for(ModuleKind::Fixed, target).unwrap();
            let result =
                crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compose_fixed_source(
                    input.clone(),
                    device_cc,
                    target,
                    state_cap,
                    nvrtc,
                )
                .unwrap();
            assert_eq!(result, input, "composer changed bytes for {name}");
        }
        for (name, device_cc, target) in [
            ("CC 8.0", Some((8, 0)), "sm_89"),
            ("sm_80 target", Some((8, 9)), "sm_80"),
            ("CC 9.0 on sm_90", Some((9, 0)), "sm_90"),
        ] {
            assert!(
                crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compiler_supported(
                    device_cc,
                    target,
                    16,
                    (12, 8),
                ),
                "wrong compiler support for {name}"
            );
            let input = compose_module_source_for(ModuleKind::Fixed, target).unwrap();
            let result =
                crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compose_fixed_source(
                    input.clone(),
                    device_cc,
                    target,
                    16,
                    (12, 8),
                )
                .unwrap();
            assert!(
                result.len() > input.len() && result.starts_with(&input),
                "the retained members must follow the complete input for {name}"
            );
        }
    }

    #[test]
    fn inference_source_bundle_rejects_duplicate_retained_exports() {
        let base = compose_module_source_for(ModuleKind::Fixed, "sm_89").unwrap();
        let extended =
            crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compose_fixed_source(
                base,
                Some((8, 9)),
                "sm_89",
                16,
                (12, 8),
            )
            .unwrap();
        let duplicate =
            crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compose_fixed_source(
                extended,
                Some((8, 9)),
                "sm_89",
                16,
                (12, 8),
            );
        assert!(duplicate.is_err());
    }

    #[test]
    fn inference_source_bundle_appends_after_the_complete_fold_overlay() {
        let symbols = [
            "nn_sm89_tc128_f32out_s3_bf16",
            "nn_sm89_tc128_f32out_s3_f16",
            "nn_sm89_f32_m128n64_tail_copyplan",
        ];
        for state_cap in [16, 64] {
            for nvrtc in [(12, 8), (13, 0), (13, 2)] {
                let base = compose_module_source_for(ModuleKind::Fixed, "sm_89").unwrap();
                let folded = crate::mamba_ssm::gpu::fold_transport::compose_fixed_source(
                    base,
                    Some((8, 9)),
                    "sm_89",
                    state_cap,
                    nvrtc,
                )
                .unwrap();
                let final_source =
                    crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compose_fixed_source(
                        folded.clone(),
                        Some((8, 9)),
                        "sm_89",
                        state_cap,
                        nvrtc,
                    )
                    .unwrap();
                assert!(final_source.starts_with(&folded));
                for symbol in symbols {
                    assert_eq!(final_source.matches(&format!("void {symbol}(")).count(), 1);
                }
            }
        }
    }

    #[test]
    fn sm89_half_resource_failure_excludes_only_the_bad_small16_dtype() {
        let bf16 = super::super::sm89_half_tn_source::SMALL16_BF16_SYMBOL;
        let f16 = super::super::sm89_half_tn_source::SMALL16_F16_SYMBOL;

        for (missing, sibling) in [(bf16, f16), (f16, bf16)] {
            let mut functions = HashMap::new();
            let mut exclusions = Vec::new();
            super::retain_sm89_half_symbol(
                &mut functions,
                &mut exclusions,
                missing,
                Err("registers exceed the retained gate".into()),
            )
            .unwrap();
            super::retain_sm89_half_symbol(&mut functions, &mut exclusions, sibling, Ok(13_u8))
                .unwrap();

            assert_eq!(
                super::sm89_half_runtime_entry(&functions, true, missing),
                None
            );
            assert_eq!(
                super::sm89_half_runtime_entry(&functions, true, sibling),
                Some(&13)
            );
            assert_eq!(
                super::sm89_half_runtime_entry(&functions, false, sibling),
                None
            );
            assert_eq!(
                super::sm89_half_runtime_entry(&functions, true, "unknown_half_symbol"),
                None
            );
            assert_eq!(
                exclusions,
                [super::Tf32SymbolExclusion {
                    symbol: missing,
                    reason: "registers exceed the retained gate".into(),
                }]
            );
        }
    }

    #[test]
    fn sm89_half_driver_abi_failure_excludes_only_its_own_symbol() {
        let nn =
            Tf32DriverAbi::checked(5, vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]).unwrap();
        let nt = Tf32DriverAbi::checked(
            7,
            vec![(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)],
        )
        .unwrap();
        let census = std::collections::BTreeMap::from([
            ("nn_bf16", Ok(nn.clone())),
            ("nn_f16", Err("nn_f16 Driver ABI mismatch".to_string())),
            ("nt_bf16", Ok(nt)),
        ]);

        assert_eq!(
            super::sm89_half_abi_for_symbol(&census, "nn_bf16").unwrap(),
            &nn
        );
        assert_eq!(
            super::sm89_half_abi_for_symbol(&census, "nn_f16").unwrap_err(),
            "nn_f16 Driver ABI mismatch"
        );
        assert!(super::sm89_half_abi_for_symbol(&census, "nt_bf16").is_ok());

        let mut functions = HashMap::new();
        let mut exclusions = Vec::new();
        for (index, symbol) in ["nn_bf16", "nn_f16", "nt_bf16"].into_iter().enumerate() {
            let loaded = super::sm89_half_abi_for_symbol(&census, symbol).map(|_| index as u8);
            super::retain_sm89_half_symbol(&mut functions, &mut exclusions, symbol, loaded)
                .unwrap();
        }
        assert_eq!(
            functions,
            HashMap::from([("nn_bf16", 0_u8), ("nt_bf16", 2_u8)])
        );
        assert_eq!(exclusions.len(), 1);
        assert_eq!(exclusions[0].symbol, "nn_f16");
        assert_eq!(exclusions[0].reason, "nn_f16 Driver ABI mismatch");
    }

    #[test]
    fn triad_retained_half_small16_uses_the_exact_seven_argument_driver_abi() {
        let valid = Tf32DriverAbi::checked(
            7,
            vec![(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)],
        )
        .unwrap();
        let short =
            Tf32DriverAbi::checked(6, vec![(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4)])
                .unwrap();
        for symbol in [
            super::super::sm89_half_tn_source::SMALL16_BF16_SYMBOL,
            super::super::sm89_half_tn_source::SMALL16_F16_SYMBOL,
        ] {
            let spec = super::super::sm89_half_source::runtime_kernel_spec(symbol).unwrap();
            super::validate_sm89_half_driver_abi(&spec, &valid).unwrap();
            assert!(super::validate_sm89_half_driver_abi(&spec, &short).is_err());
        }
    }

    fn sm89_half_validator_test_loads(
        route: super::super::sm89_half_source::Sm89HalfRuntimeRoute,
    ) -> [&'static str; 2] {
        use super::super::sm89_half_source::{Sm89HalfRoute, Sm89HalfRuntimeRoute};

        match route {
            Sm89HalfRuntimeRoute::Legacy(Sm89HalfRoute::NnM128N128Bk64S3) => [
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            ],
            Sm89HalfRuntimeRoute::Legacy(
                Sm89HalfRoute::TnM64N64Bk64S2CompactBxor | Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
            )
            | Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72
            | Sm89HalfRuntimeRoute::TnD128InM32N16Bk64S4
            | Sm89HalfRuntimeRoute::TnD128OutM32N16Bk64S4
            | Sm89HalfRuntimeRoute::TnRelayM64N64Bk64S3 => [
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            ],
            Sm89HalfRuntimeRoute::NtSmallM16N64Bk64S4 => [
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
            ],
            Sm89HalfRuntimeRoute::NnSmallM16N64Bk64S4 => [
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
            ],
            Sm89HalfRuntimeRoute::Legacy(
                Sm89HalfRoute::NtM128N128Bk64S3Bxor | Sm89HalfRoute::NtM96N128Bk64S3,
            ) => [
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
            ],
        }
    }

    fn sm89_half_validator_opposite_x2(
        route: super::super::sm89_half_source::Sm89HalfRuntimeRoute,
    ) -> &'static str {
        use super::super::sm89_half_source::{Sm89HalfRoute, Sm89HalfRuntimeRoute};

        match route {
            Sm89HalfRuntimeRoute::Legacy(Sm89HalfRoute::NnM128N128Bk64S3) => {
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
            }
            Sm89HalfRuntimeRoute::Legacy(
                Sm89HalfRoute::TnM64N64Bk64S2CompactBxor | Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
            )
            | Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72
            | Sm89HalfRuntimeRoute::TnD128InM32N16Bk64S4
            | Sm89HalfRuntimeRoute::TnD128OutM32N16Bk64S4
            | Sm89HalfRuntimeRoute::TnRelayM64N64Bk64S3
            | Sm89HalfRuntimeRoute::NnSmallM16N64Bk64S4 => {
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
            }
            Sm89HalfRuntimeRoute::Legacy(
                Sm89HalfRoute::NtM128N128Bk64S3Bxor | Sm89HalfRoute::NtM96N128Bk64S3,
            )
            | Sm89HalfRuntimeRoute::NtSmallM16N64Bk64S4 => {
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"
            }
        }
    }

    fn sm89_half_validator_opposite_x4(
        route: super::super::sm89_half_source::Sm89HalfRuntimeRoute,
    ) -> &'static str {
        use super::super::sm89_half_source::{Sm89HalfRoute, Sm89HalfRuntimeRoute};

        match route {
            Sm89HalfRuntimeRoute::Legacy(
                Sm89HalfRoute::TnM64N64Bk64S2CompactBxor | Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
            )
            | Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72
            | Sm89HalfRuntimeRoute::TnD128InM32N16Bk64S4
            | Sm89HalfRuntimeRoute::TnD128OutM32N16Bk64S4
            | Sm89HalfRuntimeRoute::TnRelayM64N64Bk64S3 => {
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            }
            Sm89HalfRuntimeRoute::Legacy(
                Sm89HalfRoute::NnM128N128Bk64S3
                | Sm89HalfRoute::NtM128N128Bk64S3Bxor
                | Sm89HalfRoute::NtM96N128Bk64S3,
            )
            | Sm89HalfRuntimeRoute::NtSmallM16N64Bk64S4
            | Sm89HalfRuntimeRoute::NnSmallM16N64Bk64S4 => {
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            }
        }
    }

    fn sm89_half_validator_test_entry(
        spec: super::super::sm89_half_source::Sm89HalfRuntimeSpec,
    ) -> String {
        let [a_load, b_load] = sm89_half_validator_test_loads(spec.route);
        let mma = match spec.dtype {
            crate::mamba_ssm::gpu::dtype::WeightDtype::Bf16 => {
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
            }
            crate::mamba_ssm::gpu::dtype::WeightDtype::F16 => {
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            }
            crate::mamba_ssm::gpu::dtype::WeightDtype::F32
            | crate::mamba_ssm::gpu::dtype::WeightDtype::Tf32 => unreachable!(),
        };
        format!(
            ".visible .entry {}() {{\n\
             .reg .b32 %r<8>;\n.reg .b64 %rd<2>;\n.reg .f32 %f<8>;\n\
             {} [%r0], [%rd0], 16;\n\
             {a_load} {{%r0,%r1,%r2,%r3}}, [%r4];\n\
             {b_load} {{%r4,%r5}}, [%r6];\n\
             {mma} {{%f0,%f1,%f2,%f3}}, {{%r0,%r1,%r2,%r3}}, {{%r4,%r5}}, {{%f0,%f1,%f2,%f3}};\n\
             ret;\n}}\n",
            spec.symbol,
            sm89_half_validator_test_copy(spec.route),
        )
    }

    fn sm89_half_validator_test_copy(
        route: super::super::sm89_half_source::Sm89HalfRuntimeRoute,
    ) -> &'static str {
        use super::super::sm89_half_source::{Sm89HalfRoute, Sm89HalfRuntimeRoute};

        match route {
            Sm89HalfRuntimeRoute::Legacy(
                Sm89HalfRoute::TnM64N64Bk64S2CompactBxor | Sm89HalfRoute::TnM64N64Bk64S2RegpipeVec2,
            )
            | Sm89HalfRuntimeRoute::TnSmall16Bk64S2Ldb72
            | Sm89HalfRuntimeRoute::TnRelayM64N64Bk64S3
            | Sm89HalfRuntimeRoute::NtSmallM16N64Bk64S4
            | Sm89HalfRuntimeRoute::NnSmallM16N64Bk64S4 => "cp.async.ca.shared.global",
            Sm89HalfRuntimeRoute::Legacy(
                Sm89HalfRoute::NnM128N128Bk64S3
                | Sm89HalfRoute::NtM128N128Bk64S3Bxor
                | Sm89HalfRoute::NtM96N128Bk64S3,
            )
            | Sm89HalfRuntimeRoute::TnD128InM32N16Bk64S4
            | Sm89HalfRuntimeRoute::TnD128OutM32N16Bk64S4 => "cp.async.cg.shared.global",
        }
    }

    fn sm89_half_validator_test_ptx() -> String {
        super::super::sm89_half_source::runtime_kernel_specs().fold(
            String::from(".version 8.5\n.target sm_89\n.address_size 64\n"),
            |mut ptx, spec| {
                ptx.push_str(&sm89_half_validator_test_entry(spec));
                ptx
            },
        )
    }

    #[test]
    fn sm89_half_validator_accepts_exact_route_specific_ldmatrix_pairs() {
        super::validate_sm89_half_ptx("sm_89", &sm89_half_validator_test_ptx())
            .expect("the twelve exact route-specific half entries must validate");
    }

    #[test]
    fn triad_retained_half_validator_accepts_the_closed_twelve_symbol_inventory() {
        let baseline = sm89_half_validator_test_ptx();
        for symbol in [
            super::super::sm89_half_tn_source::SMALL16_BF16_SYMBOL,
            super::super::sm89_half_tn_source::SMALL16_F16_SYMBOL,
        ] {
            let missing = baseline.replace(
                &sm89_half_validator_test_entry(
                    super::super::sm89_half_source::runtime_kernel_spec(symbol).unwrap(),
                ),
                "",
            );
            super::validate_sm89_half_ptx("sm_89", &missing)
                .expect_err("the closed half inventory must reject a missing small16 sibling");
        }
    }

    #[test]
    fn sm89_half_validator_rejects_missing_or_swapped_ldmatrix_per_spec() {
        let baseline = sm89_half_validator_test_ptx();
        for spec in super::super::sm89_half_source::runtime_kernel_specs() {
            let entry = sm89_half_validator_test_entry(spec);
            let [required_x4, required_x2] = sm89_half_validator_test_loads(spec.route);

            let missing_x4_entry = entry.replacen(required_x4, "not_the_required_ldmatrix", 1);
            let missing_x4 = baseline.replacen(&entry, &missing_x4_entry, 1);
            super::validate_sm89_half_ptx("sm_89", &missing_x4).expect_err(&format!(
                "{:?}/{:?} must reject when {required_x4} is absent",
                spec.route, spec.dtype
            ));

            let opposite_x2 = sm89_half_validator_opposite_x2(spec.route);
            let swapped_x2_entry = entry.replacen(required_x2, opposite_x2, 1);
            assert!(swapped_x2_entry.contains("ldmatrix.sync.aligned"));
            let swapped_x2 = baseline.replacen(&entry, &swapped_x2_entry, 1);
            super::validate_sm89_half_ptx("sm_89", &swapped_x2).expect_err(&format!(
                "{:?}/{:?} must reject {opposite_x2} in place of {required_x2}",
                spec.route, spec.dtype
            ));
        }
    }

    #[test]
    fn sm89_half_validator_rejects_opposite_x2_coexisting_per_spec() {
        let baseline = sm89_half_validator_test_ptx();
        for spec in super::super::sm89_half_source::runtime_kernel_specs() {
            let entry = sm89_half_validator_test_entry(spec);
            let opposite_x2 = sm89_half_validator_opposite_x2(spec.route);
            let coexisting_entry = entry.replacen(
                "ret;",
                &format!("{opposite_x2} {{%r4,%r5}}, [%r6];\nret;"),
                1,
            );
            assert!(
                sm89_half_validator_test_loads(spec.route)
                    .into_iter()
                    .all(|required| coexisting_entry.contains(required))
            );
            let coexisting = baseline.replacen(&entry, &coexisting_entry, 1);
            super::validate_sm89_half_ptx("sm_89", &coexisting).expect_err(&format!(
                "{:?}/{:?} must reject coexisting opposite {opposite_x2}",
                spec.route, spec.dtype
            ));

            let opposite_x4 = sm89_half_validator_opposite_x4(spec.route);
            let coexisting_entry = entry.replacen(
                "ret;",
                &format!("{opposite_x4} {{%r0,%r1,%r2,%r3}}, [%r4];\nret;"),
                1,
            );
            let coexisting = baseline.replacen(&entry, &coexisting_entry, 1);
            super::validate_sm89_half_ptx("sm_89", &coexisting).expect_err(&format!(
                "{:?}/{:?} must reject coexisting opposite {opposite_x4}",
                spec.route, spec.dtype
            ));
        }
    }

    fn sm89_exact_f32_test_entry(
        spec: super::super::sm89_exact_f32_source::Sm89ExactF32KernelSpec,
    ) -> String {
        use super::super::sm89_exact_f32_source::Sm89ExactF32KernelKind;

        let parameters = match spec.kind {
            Sm89ExactF32KernelKind::DualChunkFusedFinalize => {
                ".param .u64 p0,\n.param .u64 p1,\n.param .u64 p2,\n.param .align 4 .b8 p3[32]"
            }
            Sm89ExactF32KernelKind::DirectSplitMRaw => {
                ".param .u64 p0,\n.param .u64 p1,\n.param .u64 p2,\n.param .u32 p3,\n.param .u32 p4,\n.param .u32 p5,\n.param .u32 p6"
            }
        };
        let (async_copy, finalize) = match spec.kind {
            Sm89ExactF32KernelKind::DualChunkFusedFinalize => (
                "cp.async.cg.shared.global",
                "add.rn.f64 %fd0, %fd1, %fd2;\nmul.rn.f64 %fd0, %fd0, %fd1;\ncvt.rn.f32.f64 %f0, %fd0;\n",
            ),
            Sm89ExactF32KernelKind::DirectSplitMRaw => ("cp.async.ca.shared.global", ""),
        };
        format!(
            ".visible .entry {}(\n{}\n)\n{{\n.reg .b32 %r<4>;\n.reg .b64 %rd<2>;\n.reg .f32 %f<4>;\n.reg .f64 %fd<3>;\n{} [%r0], [%rd0], 16;\nfma.rn.f32 %f0, %f1, %f2, %f3;\n{}ret;\n}}\n",
            spec.symbol, parameters, async_copy, finalize
        )
    }

    fn sm89_exact_f32_test_ptx() -> String {
        super::super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS
            .iter()
            .fold(
                String::from(".version 8.5\n.target sm_89\n.address_size 64\n"),
                |mut ptx, &spec| {
                    ptx.push_str(&sm89_exact_f32_test_entry(spec));
                    ptx
                },
            )
    }

    #[test]
    fn sm89_exact_f32_composition_binds_the_frozen_owner_and_exact_target() {
        assert!(super::super::contract::tf32_route_specs(ModuleKind::TriadSm89ExactF32).is_empty());
        let owner_digest = crate::mamba_ssm::gpu::kernel_identity::FramedSha256::bytes(
            super::super::sm89_exact_f32_source::OWNER_TEMPLATE.as_bytes(),
        );
        assert_eq!(
            owner_digest,
            super::super::sm89_exact_f32_source::OWNER_SHA256_BYTES
        );
        assert_eq!(
            crate::mamba_ssm::gpu::kernel_identity::digest_hex(&owner_digest),
            super::super::sm89_exact_f32_source::OWNER_SHA256
        );
        let source = compose_module_source_for(ModuleKind::TriadSm89ExactF32, "sm_89").unwrap();
        assert_eq!(
            super::module_source_digest(ModuleKind::TriadSm89ExactF32, "sm_89").unwrap(),
            crate::mamba_ssm::gpu::kernel_identity::FramedSha256::bytes(source.as_bytes())
        );
        assert!(
            super::super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS
                .iter()
                .all(|spec| source.matches(&format!("{}(", spec.symbol)).count() == 1)
        );
        assert!(compose_module_source_for(ModuleKind::TriadSm89ExactF32, "compute_89").is_err());
    }

    #[test]
    fn sm89_exact_f32_validator_accepts_only_the_three_route_specific_entries() {
        let baseline = sm89_exact_f32_test_ptx();
        super::validate_sm89_exact_f32_ptx("sm_89", &baseline).unwrap();
        assert!(super::validate_sm89_exact_f32_ptx("compute_89", &baseline).is_err());
        assert!(
            super::validate_sm89_exact_f32_ptx(
                "sm_89",
                &baseline.replace(".target sm_89", ".target sm_80")
            )
            .is_err()
        );
        for &spec in &super::super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS {
            let entry = sm89_exact_f32_test_entry(spec);
            assert!(
                super::validate_sm89_exact_f32_ptx("sm_89", &baseline.replacen(&entry, "", 1))
                    .is_err(),
                "accepted missing {}",
                spec.symbol
            );
            assert!(
                super::validate_sm89_exact_f32_ptx("sm_89", &(baseline.clone() + &entry)).is_err(),
                "accepted duplicate {}",
                spec.symbol
            );
        }
        for foreign in [
            ".visible .entry tn_sm89_f32_d128_foreign() { ret; }\n",
            ".visible .entry unrelated_callable_export() { ret; }\n",
        ] {
            assert!(
                super::validate_sm89_exact_f32_ptx("sm_89", &(baseline.clone() + foreign)).is_err()
            );
        }
    }

    #[test]
    fn sm89_exact_f32_validator_rejects_each_abi_and_instruction_drift() {
        let baseline = sm89_exact_f32_test_ptx();
        for &spec in &super::super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS {
            let entry = sm89_exact_f32_test_entry(spec);
            let wrong_abi_entry = entry.replacen(".param .u64 p0", ".param .u32 p0", 1);
            assert!(
                super::validate_sm89_exact_f32_ptx(
                    "sm_89",
                    &baseline.replacen(&entry, &wrong_abi_entry, 1)
                )
                .is_err(),
                "accepted ABI drift for {}",
                spec.symbol
            );
            for instruction in ["fma.rn.f32", "cp.async."] {
                let wrong = if instruction == "cp.async." {
                    entry.replacen("cp.async.", "not_async.", 1)
                } else {
                    entry.replacen(instruction, "not_fma", 1)
                };
                assert!(
                    super::validate_sm89_exact_f32_ptx(
                        "sm_89",
                        &baseline.replacen(&entry, &wrong, 1)
                    )
                    .is_err(),
                    "accepted missing {instruction} for {}",
                    spec.symbol
                );
            }
            for forbidden in ["ld.local.u32", "atom.global.add.f32", "mma.sync.aligned"] {
                let wrong = entry.replacen("ret;", &format!("{forbidden} %r0;\nret;"), 1);
                assert!(
                    super::validate_sm89_exact_f32_ptx(
                        "sm_89",
                        &baseline.replacen(&entry, &wrong, 1)
                    )
                    .is_err(),
                    "accepted {forbidden} for {}",
                    spec.symbol
                );
            }
        }
    }

    #[test]
    fn sm89_exact_f32_driver_abi_and_resources_are_per_symbol() {
        use super::super::sm89_exact_f32_source::Sm89ExactF32KernelKind;

        let mut functions = HashMap::new();
        let mut exclusions = Vec::new();
        for (index, spec) in super::super::sm89_exact_f32_source::SM89_EXACT_F32_KERNEL_SPECS
            .iter()
            .enumerate()
        {
            let layout = match spec.kind {
                Sm89ExactF32KernelKind::DualChunkFusedFinalize => {
                    vec![(0, 8), (8, 8), (16, 8), (24, 32)]
                }
                Sm89ExactF32KernelKind::DirectSplitMRaw => {
                    vec![(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)]
                }
            };
            let abi = Tf32DriverAbi::checked(spec.abi_parameter_count as usize, layout).unwrap();
            super::validate_sm89_exact_f32_driver_abi(spec, &abi).unwrap();
            let mut wrong = abi.clone();
            wrong.parameters[0].size = 4;
            assert!(super::validate_sm89_exact_f32_driver_abi(spec, &wrong).is_err());

            let valid = super::Sm89ExactF32ResourceFacts {
                local_bytes: 0,
                registers: spec.register_cap,
                static_shared_bytes: spec.static_shared_bytes,
                max_threads: spec.block.0 as i32,
                occupancy: spec.occupancy_gate,
            };
            super::validate_sm89_exact_f32_resources(spec, valid).unwrap();
            for invalid in [
                super::Sm89ExactF32ResourceFacts {
                    local_bytes: 4,
                    ..valid
                },
                super::Sm89ExactF32ResourceFacts {
                    registers: spec.register_cap + 1,
                    ..valid
                },
                super::Sm89ExactF32ResourceFacts {
                    static_shared_bytes: spec.static_shared_bytes + 4,
                    ..valid
                },
                super::Sm89ExactF32ResourceFacts {
                    max_threads: spec.block.0 as i32 - 1,
                    ..valid
                },
                super::Sm89ExactF32ResourceFacts {
                    occupancy: spec.occupancy_gate - 1,
                    ..valid
                },
            ] {
                assert!(super::validate_sm89_exact_f32_resources(spec, invalid).is_err());
            }

            let loaded = if index == 1 {
                Err("resource gate failed".to_string())
            } else {
                Ok(index as u8)
            };
            super::retain_sm89_exact_f32_symbol(
                &mut functions,
                &mut exclusions,
                spec.symbol,
                loaded,
            )
            .unwrap();
        }
        assert_eq!(functions.len(), 2);
        assert_eq!(exclusions.len(), 1);
        assert_eq!(
            exclusions[0].symbol,
            super::super::sm89_exact_f32_source::D768_OUT_RAW_SYMBOL
        );
    }

    fn sm89_exact_f32_d128_test_entry(symbol: &str) -> String {
        format!(
            ".visible .entry {symbol}(\n.param .u64 p0,\n.param .u64 p1,\n.param .u64 p2,\n.param .f32 p3,\n.param .u32 p4,\n.param .u32 p5,\n.param .u32 p6\n)\n{{\n.reg .b32 %r<4>;\n.reg .b64 %rd<2>;\n.reg .f32 %f<4>;\n.reg .f64 %fd<3>;\ncp.async.cg.shared.global [%r0], [%rd0], 16;\nfma.rn.f32 %f0, %f1, %f2, %f3;\nadd.rn.f64 %fd0, %fd1, %fd2;\nmul.rn.f64 %fd0, %fd0, %fd1;\ncvt.rn.f32.f64 %f0, %fd0;\nret;\n}}\n"
        )
    }

    fn sm89_exact_f32_d128_test_ptx() -> String {
        super::super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS
            .iter()
            .fold(
                String::from(".version 8.5\n.target sm_89\n.address_size 64\n"),
                |mut ptx, spec| {
                    ptx.push_str(&sm89_exact_f32_d128_test_entry(spec.symbol));
                    ptx
                },
            )
    }

    #[test]
    fn sm89_exact_f32_d128_composition_is_sealed_and_target_exact() {
        let source = compose_module_source_for(ModuleKind::TriadSm89ExactF32D128, "sm_89")
            .expect("compose sealed d128 module");
        assert_eq!(
            super::module_source_digest(ModuleKind::TriadSm89ExactF32D128, "sm_89").unwrap(),
            crate::mamba_ssm::gpu::kernel_identity::FramedSha256::bytes(source.as_bytes())
        );
        super::super::sm89_exact_f32_d128_source::validate_source().unwrap();
        assert!(
            compose_module_source_for(ModuleKind::TriadSm89ExactF32D128, "compute_89").is_err()
        );
    }

    #[test]
    fn sm89_exact_f32_d128_validator_requires_exact_float_alpha_and_instruction_contract() {
        let baseline = sm89_exact_f32_d128_test_ptx();
        super::validate_sm89_exact_f32_d128_ptx("sm_89", &baseline).unwrap();
        assert!(super::validate_sm89_exact_f32_d128_ptx("compute_89", &baseline).is_err());
        assert!(
            super::validate_sm89_exact_f32_d128_ptx(
                "sm_89",
                &baseline.replacen(".target sm_89", ".target sm_90", 1),
            )
            .is_err()
        );
        let first = super::super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS[0];
        assert!(
            super::validate_sm89_exact_f32_d128_ptx(
                "sm_89",
                &format!("{baseline}{}", sm89_exact_f32_d128_test_entry(first.symbol)),
            )
            .is_err(),
            "accepted a duplicate d128 export"
        );
        assert!(
            super::validate_sm89_exact_f32_d128_ptx(
                "sm_89",
                &format!(
                    "{baseline}{}",
                    sm89_exact_f32_d128_test_entry("tn_sm89_f32_d128_foreign")
                ),
            )
            .is_err(),
            "accepted a foreign d128 export"
        );
        for spec in super::super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS {
            let entry = sm89_exact_f32_d128_test_entry(spec.symbol);
            assert!(
                super::validate_sm89_exact_f32_d128_ptx("sm_89", &baseline.replacen(&entry, "", 1))
                    .is_err()
            );
            let missing_parameter = entry.replacen(".param .u32 p6\n", "", 1);
            assert!(
                super::validate_sm89_exact_f32_d128_ptx(
                    "sm_89",
                    &baseline.replacen(&entry, &missing_parameter, 1),
                )
                .is_err(),
                "accepted a six-parameter ABI for {}",
                spec.symbol
            );
            for (declaration, wrong_type) in [
                (".param .u64 p0", ".param .u32 p0"),
                (".param .u64 p1", ".param .u32 p1"),
                (".param .u64 p2", ".param .u32 p2"),
                (".param .f32 p3", ".param .u32 p3"),
                (".param .u32 p4", ".param .u64 p4"),
                (".param .u32 p5", ".param .u64 p5"),
                (".param .u32 p6", ".param .u64 p6"),
            ] {
                let wrong = entry.replacen(declaration, wrong_type, 1);
                assert!(
                    super::validate_sm89_exact_f32_d128_ptx(
                        "sm_89",
                        &baseline.replacen(&entry, &wrong, 1),
                    )
                    .is_err(),
                    "accepted {wrong_type} for {}",
                    spec.symbol
                );
            }
            for required in [
                "fma.rn.f32",
                "add.rn.f64",
                "mul.rn.f64",
                "cvt.rn.f32.f64",
                "cp.async.cg.shared.global",
            ] {
                let wrong = entry.replacen(required, "missing.instruction", 1);
                assert!(
                    super::validate_sm89_exact_f32_d128_ptx(
                        "sm_89",
                        &baseline.replacen(&entry, &wrong, 1)
                    )
                    .is_err(),
                    "accepted missing {required} for {}",
                    spec.symbol
                );
            }
            for forbidden in [
                "ld.local.u32",
                "atom.global.add.f32",
                "atom::sc.global.add.f32",
                "red.global.add.f32",
                "red::gpu.global.add.f32",
                "redux.sync.add.u32",
                "mma.sync.aligned",
                "wgmma.mma_async",
                "wmma.mma.sync",
                "tcgen05.mma",
                "cp.async.bulk.tensor",
                "cp.reduce.async.bulk",
                "tensormap.replace.tile.global_address.shared::cta.b1024.b64",
            ] {
                let wrong = entry.replacen("ret;", &format!("{forbidden} %r0;\nret;"), 1);
                assert!(
                    super::validate_sm89_exact_f32_d128_ptx(
                        "sm_89",
                        &baseline.replacen(&entry, &wrong, 1)
                    )
                    .is_err(),
                    "accepted {forbidden} for {}",
                    spec.symbol
                );
            }
        }
    }

    #[test]
    fn sm89_exact_f32_d128_abi_and_resource_failures_exclude_only_one_symbol() {
        for spec in super::super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS {
            let abi = Tf32DriverAbi::checked(
                7,
                vec![(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (36, 4)],
            )
            .unwrap();
            super::validate_sm89_exact_f32_d128_driver_abi(&spec, &abi).unwrap();
            for malformed in [
                Tf32DriverAbi::checked(6, vec![(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4)])
                    .unwrap(),
                Tf32DriverAbi::checked(
                    7,
                    vec![(0, 8), (8, 8), (16, 8), (24, 4), (32, 4), (36, 4), (40, 4)],
                )
                .unwrap(),
                Tf32DriverAbi::checked(
                    7,
                    vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 4), (36, 4), (40, 4)],
                )
                .unwrap(),
            ] {
                assert!(
                    super::validate_sm89_exact_f32_d128_driver_abi(&spec, &malformed).is_err(),
                    "accepted malformed live ABI for {}: {malformed:?}",
                    spec.symbol
                );
            }
            let valid = super::Sm89ExactF32D128ResourceFacts {
                local_bytes: 0,
                registers: spec.register_cap,
                static_shared_bytes: 0,
                max_threads: 256,
                occupancy: 2,
            };
            super::validate_sm89_exact_f32_d128_resources(&spec, valid).unwrap();
            for malformed in [
                super::Sm89ExactF32D128ResourceFacts {
                    local_bytes: 4,
                    ..valid
                },
                super::Sm89ExactF32D128ResourceFacts {
                    registers: spec.register_cap + 1,
                    ..valid
                },
                super::Sm89ExactF32D128ResourceFacts {
                    static_shared_bytes: 4,
                    ..valid
                },
                super::Sm89ExactF32D128ResourceFacts {
                    max_threads: 255,
                    ..valid
                },
                super::Sm89ExactF32D128ResourceFacts {
                    occupancy: 1,
                    ..valid
                },
            ] {
                assert!(
                    super::validate_sm89_exact_f32_d128_resources(&spec, malformed).is_err(),
                    "accepted malformed resources for {}: {malformed:?}",
                    spec.symbol
                );
            }
        }

        for (excluded_index, failure_kind) in [(0, "live ABI"), (1, "resource")] {
            let mut functions = HashMap::new();
            let mut exclusions = Vec::new();
            for (index, spec) in
                super::super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS
                    .iter()
                    .enumerate()
            {
                let loaded = if index != excluded_index {
                    Ok(index as u8)
                } else if failure_kind == "live ABI" {
                    let malformed = Tf32DriverAbi::checked(
                        6,
                        vec![(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4)],
                    )
                    .unwrap();
                    Err(
                        super::validate_sm89_exact_f32_d128_driver_abi(spec, &malformed)
                            .unwrap_err(),
                    )
                } else {
                    Err(super::validate_sm89_exact_f32_d128_resources(
                        spec,
                        super::Sm89ExactF32D128ResourceFacts {
                            local_bytes: 0,
                            registers: spec.register_cap + 1,
                            static_shared_bytes: 0,
                            max_threads: 64,
                            occupancy: 8,
                        },
                    )
                    .unwrap_err())
                };
                super::retain_sm89_exact_f32_d128_symbol(
                    &mut functions,
                    &mut exclusions,
                    spec.symbol,
                    loaded,
                )
                .unwrap();
            }
            assert_eq!(functions.len(), 1);
            assert_eq!(exclusions.len(), 1);
            assert_eq!(
                exclusions[0].symbol,
                super::super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS
                    [excluded_index]
                    .symbol
            );
            assert!(
                functions.contains_key(
                    super::super::sm89_exact_f32_d128_source::SM89_EXACT_F32_D128_KERNEL_SPECS
                        [1 - excluded_index]
                        .symbol
                ),
                "{failure_kind} exclusion removed the sibling"
            );
        }
    }

    fn sm89_tf32_joint_test_entry(
        spec: super::super::sm89_tf32_joint_source::Sm89Tf32JointKernelSpec,
    ) -> String {
        use super::super::sm89_tf32_joint_source::Sm89Tf32JointKernelKind;

        let parameters = match spec.kind {
            Sm89Tf32JointKernelKind::TnPreRnaTranspose32x32 => {
                ".param .u64 p0,\n.param .u64 p1,\n.param .align 4 .b8 p2[12]"
            }
            _ => {
                ".param .u64 p0,\n.param .u64 p1,\n.param .u64 p2,\n.param .u64 p3,\n.param .align 4 .b8 p4[32]"
            }
        };
        let body = match spec.kind {
            Sm89Tf32JointKernelKind::TnPreRnaTranspose32x32 => "cvt.rna.tf32.f32 %r0, %f0;\n",
            Sm89Tf32JointKernelKind::NtRnaM144N96Bk32S2 => {
                "cp.async.cg.shared.global.L2::128B [%r0], [%rd0], 16;\nldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%r4];\ncvt.rna.tf32.f32 %r0, %f0;\nmma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};\n"
            }
            Sm89Tf32JointKernelKind::NtRowstageM128N192Bk32S2 => {
                "cp.async.cg.shared.global.L2::128B [%r0], [%rd0], 16;\nldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%r4];\nmma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};\n"
            }
            Sm89Tf32JointKernelKind::TnDirectM192N192Bk32S2 => {
                "cp.async.cg.shared.global.L2::128B [%r0], [%rd0], 16;\ncvt.rna.tf32.f32 %r0, %f0;\nmma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};\n"
            }
            Sm89Tf32JointKernelKind::TnPreRnaM128N96Bk32S3
            | Sm89Tf32JointKernelKind::TnPreRnaM64N64Bk32S3
            | Sm89Tf32JointKernelKind::TnPreRnaM64N96Bk32S2
            | Sm89Tf32JointKernelKind::TnPreRnaM96N192Bk32S2
            | Sm89Tf32JointKernelKind::TnPreRnaM96N96Bk32S3 => {
                "cp.async.cg.shared.global.L2::128B [%r0], [%rd0], 16;\nldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%r4];\ncvt.rna.tf32.f32 %r0, %f0;\nmma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};\n"
            }
            Sm89Tf32JointKernelKind::NtALdmatrixM128N96Bk32S3 => {
                "cp.async.cg.shared.global.L2::128B [%r0], [%rd0], 16;\nldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%r4];\nldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r0,%r1}, [%r4];\nmma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};\n"
            }
            Sm89Tf32JointKernelKind::NnAddHalfDirectM128N96Bk32S3
            | Sm89Tf32JointKernelKind::NnAddHalfM128N96Bk32S3 => {
                "cp.async.cg.shared.global.L2::128B [%r0], [%rd0], 16;\nldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%r4];\nmma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};\n"
            }
        };
        format!(
            ".visible .entry {}(\n{}\n)\n{{\n.reg .b32 %r<8>;\n.reg .b64 %rd<2>;\n.reg .f32 %f<4>;\n{}ret;\n}}\n",
            spec.symbol, parameters, body
        )
    }

    fn sm89_tf32_joint_test_ptx() -> String {
        super::super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS
            .iter()
            .fold(
                String::from(".version 8.5\n.target sm_89\n.address_size 64\n"),
                |mut ptx, &spec| {
                    ptx.push_str(&sm89_tf32_joint_test_entry(spec));
                    ptx
                },
            )
    }

    #[test]
    fn sm89_tf32_joint_composition_is_isolated_and_exact_target_only() {
        assert_eq!(
            super::super::contract::tf32_route_specs(ModuleKind::TriadSm89Tf32Joint),
            &super::super::contract::SM89_TF32_JOINT_ROUTE_SPECS,
            "the joint artifact must expose exactly its twelve GEMM routes; the thirteenth export is the TN input transform"
        );
        let source = compose_module_source_for(ModuleKind::TriadSm89Tf32Joint, "sm_89").unwrap();
        assert_eq!(
            super::module_source_digest(ModuleKind::TriadSm89Tf32Joint, "sm_89").unwrap(),
            crate::mamba_ssm::gpu::kernel_identity::FramedSha256::bytes(source.as_bytes())
        );
        assert!(source.starts_with(super::super::sm89_tf32_joint_source::PRIMITIVES));
        assert_eq!(
            super::super::sm89_tf32_joint_source::export_inventory(&source).unwrap(),
            super::super::sm89_tf32_joint_source::SM89_TF32_JOINT_SYMBOLS
        );
        assert!(
            super::super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS
                .iter()
                .all(|spec| source.matches(&format!("{}(", spec.symbol)).count() == 1)
        );
        assert!(compose_module_source_for(ModuleKind::TriadSm89Tf32Joint, "compute_89").is_err());
    }

    #[test]
    fn sm89_tf32_joint_validator_accepts_only_seven_typed_entries() {
        let baseline = sm89_tf32_joint_test_ptx();
        super::validate_sm89_tf32_joint_ptx("sm_89", &baseline).unwrap();
        assert!(super::validate_sm89_tf32_joint_ptx("compute_89", &baseline).is_err());
        assert!(
            super::validate_sm89_tf32_joint_ptx(
                "sm_89",
                &baseline.replace(".target sm_89", ".target sm_80")
            )
            .is_err()
        );
        for &spec in &super::super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS {
            let entry = sm89_tf32_joint_test_entry(spec);
            assert!(
                super::validate_sm89_tf32_joint_ptx("sm_89", &baseline.replacen(&entry, "", 1))
                    .is_err(),
                "accepted missing {}",
                spec.symbol
            );
            assert!(
                super::validate_sm89_tf32_joint_ptx("sm_89", &(baseline.clone() + &entry)).is_err(),
                "accepted duplicate {}",
                spec.symbol
            );
        }
        for foreign in [
            ".visible .entry tn_sm89_tf32_foreign() { ret; }\n",
            ".visible .entry unrelated_callable_export() { ret; }\n",
        ] {
            assert!(
                super::validate_sm89_tf32_joint_ptx("sm_89", &(baseline.clone() + foreign))
                    .is_err()
            );
        }
    }

    #[test]
    fn sm89_tf32_joint_validator_rejects_route_specific_drift() {
        use super::super::sm89_tf32_joint_source::Sm89Tf32JointKernelKind;

        let baseline = sm89_tf32_joint_test_ptx();
        for &spec in &super::super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS {
            let entry = sm89_tf32_joint_test_entry(spec);
            let wrong_abi_entry = entry.replacen(".param .u64 p0", ".param .u32 p0", 1);
            assert!(
                super::validate_sm89_tf32_joint_ptx(
                    "sm_89",
                    &baseline.replacen(&entry, &wrong_abi_entry, 1)
                )
                .is_err(),
                "accepted ABI drift for {}",
                spec.symbol
            );
            let required = match spec.kind {
                Sm89Tf32JointKernelKind::TnPreRnaTranspose32x32 => "cvt.rna.tf32.f32",
                _ => "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
            };
            let missing = entry.replacen(required, "missing.required.instruction", 1);
            assert!(
                super::validate_sm89_tf32_joint_ptx(
                    "sm_89",
                    &baseline.replacen(&entry, &missing, 1)
                )
                .is_err(),
                "accepted missing {required} for {}",
                spec.symbol
            );
            let forbidden = entry.replacen("ret;", "atom.global.add.f32 %f0;\nret;", 1);
            assert!(
                super::validate_sm89_tf32_joint_ptx(
                    "sm_89",
                    &baseline.replacen(&entry, &forbidden, 1)
                )
                .is_err(),
                "accepted atomic instruction for {}",
                spec.symbol
            );
        }
    }

    #[test]
    fn sm89_tf32_joint_driver_abi_and_resources_are_per_symbol() {
        let mut functions = HashMap::new();
        let mut exclusions = Vec::new();
        for (index, spec) in super::super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS
            .iter()
            .enumerate()
        {
            let layout = spec
                .abi_parameters
                .iter()
                .map(|parameter| (parameter.offset as usize, parameter.size as usize))
                .collect();
            let abi = Tf32DriverAbi::checked(spec.abi_parameters.len(), layout).unwrap();
            super::validate_sm89_tf32_joint_driver_abi(spec, &abi).unwrap();
            let mut wrong = abi.clone();
            wrong.parameters[0].size = 4;
            assert!(super::validate_sm89_tf32_joint_driver_abi(spec, &wrong).is_err());

            let valid = super::Sm89Tf32JointResourceFacts {
                local_bytes: spec.local_bytes,
                registers: spec.register_cap,
                static_shared_bytes: spec.static_shared_bytes,
                max_threads: spec.minimum_max_threads as i32,
                occupancy: spec.minimum_active_blocks.unwrap_or(0),
            };
            super::validate_sm89_tf32_joint_resources(spec, valid).unwrap();
            for invalid in [
                super::Sm89Tf32JointResourceFacts {
                    local_bytes: spec.local_bytes + 4,
                    ..valid
                },
                super::Sm89Tf32JointResourceFacts {
                    registers: spec.register_cap + 1,
                    ..valid
                },
                super::Sm89Tf32JointResourceFacts {
                    static_shared_bytes: spec.static_shared_bytes + 4,
                    ..valid
                },
                super::Sm89Tf32JointResourceFacts {
                    max_threads: spec.minimum_max_threads as i32 - 1,
                    ..valid
                },
            ] {
                assert!(super::validate_sm89_tf32_joint_resources(spec, invalid).is_err());
            }
            if let Some(gate) = spec.minimum_active_blocks {
                assert!(
                    super::validate_sm89_tf32_joint_resources(
                        spec,
                        super::Sm89Tf32JointResourceFacts {
                            occupancy: gate - 1,
                            ..valid
                        }
                    )
                    .is_err()
                );
            }

            let loaded =
                if spec.symbol == super::super::sm89_tf32_joint_source::TN_PRE_RNA_M64N64_SYMBOL {
                    Err("resource gate failed".to_string())
                } else {
                    Ok(index as u8)
                };
            super::retain_sm89_tf32_joint_symbol(
                &mut functions,
                &mut exclusions,
                spec.symbol,
                loaded,
            )
            .unwrap();
        }
        assert_eq!(
            functions.len(),
            super::super::sm89_tf32_joint_source::SM89_TF32_JOINT_KERNEL_SPECS.len() - 1
        );
        assert_eq!(exclusions.len(), 1);
        assert_eq!(
            exclusions[0].symbol,
            super::super::sm89_tf32_joint_source::TN_PRE_RNA_M64N64_SYMBOL
        );
    }

    #[test]
    fn sm89_finalist_inventory_replaces_one_legacy_entry_and_rejects_foreign_families() {
        let source = compose_module_source_for(ModuleKind::TriadSm89Finalist, "sm_89").unwrap();
        assert_eq!(
            super::module_source_digest(ModuleKind::TriadSm89Finalist, "sm_89").unwrap(),
            crate::mamba_ssm::gpu::kernel_identity::FramedSha256::bytes(source.as_bytes()),
            "the finalist compiler identity must bind the complete generated source"
        );
        assert!(compose_module_source_for(ModuleKind::TriadSm89Finalist, "compute_89").is_err());

        let original = "nt_sm80_mma_tf32_m128n64_bk32_s2";
        let mut symbols = super::super::contract::tf32_module_symbols(ModuleKind::TriadSm80)
            .filter(|&symbol| symbol != original)
            .collect::<Vec<_>>();
        symbols.push(super::super::sm89_finalist_source::SM89_FINALIST_SYMBOL);
        let complete = symbols.iter().fold(
            String::from(".version 8.9\n.target sm_89\n"),
            |mut ptx, symbol| {
                ptx.push_str(&format!(".visible .entry {symbol}() {{ ret; }}\n"));
                ptx
            },
        );
        validate_sm89_finalist_ptx_inventory(&complete).unwrap();
        assert!(
            validate_sm89_finalist_ptx_inventory(&complete.replacen(
                super::super::sm89_finalist_source::SM89_FINALIST_SYMBOL,
                original,
                1,
            ))
            .is_err()
        );
        assert!(
            validate_sm89_finalist_ptx_inventory(&format!(
                "{complete}.visible .entry nn_sm90a_wgmma_tf32_foreign() {{ ret; }}\n"
            ))
            .is_err()
        );
        assert!(
            validate_sm89_finalist_ptx_inventory(&format!(
                "{complete}.visible .entry {}() {{ ret; }}\n",
                super::super::sm89_finalist_source::SM89_FINALIST_SYMBOL
            ))
            .is_err()
        );
    }

    #[test]
    fn scalar_resource_environment_rejects_zero_identity_domains() {
        let compiler = CompilerIdentity {
            source_digest: [1; 32],
            invocation_digest: [2; 32],
            header_manifest_digest: [3; 32],
            target: KernelCudaTarget::new("compute_120").unwrap(),
            nvrtc_version: (13, 2),
            nvrtc_library_domain: [4; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        let artifact = ArtifactIdentity {
            module_kind: ModuleKind::TriadScalar,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: compiler.invocation_digest,
            artifact_digest: [5; 32],
        };
        assert!(qualified_scalar_resource_environment(
            (12, 0),
            170,
            compiler,
            artifact
        ));

        let mut zero_library = compiler;
        zero_library.nvrtc_library_domain = [0; 32];
        assert!(!qualified_scalar_resource_environment(
            (12, 0),
            170,
            zero_library,
            artifact
        ));
        let mut zero_compile = compiler;
        zero_compile.invocation_digest = [0; 32];
        let mut zero_key = artifact;
        zero_key.compile_key = [0; 32];
        assert!(!qualified_scalar_resource_environment(
            (12, 0),
            170,
            zero_compile,
            zero_key
        ));
        let mut zero_artifact = artifact;
        zero_artifact.artifact_digest = [0; 32];
        assert!(!qualified_scalar_resource_environment(
            (12, 0),
            170,
            compiler,
            zero_artifact
        ));
    }

    fn tf32_test_artifact(artifact_digest: [u8; 32]) -> ArtifactIdentity {
        ArtifactIdentity {
            module_kind: ModuleKind::TriadSm80,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: [0x31; 32],
            artifact_digest,
        }
    }

    fn opaque_tf32_outputs() -> Vec<u32> {
        vec![
            0x0000_0000,
            0x8000_0000,
            0x7fc0_0001,
            0x7f80_0001,
            0x7f80_0000,
            0xff80_0000,
            0x0000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f12_3456,
        ]
    }

    #[test]
    fn a_symbol_failing_a_resource_gate_is_excluded_with_its_reason() {
        use super::tf32_symbol_admission;
        assert!(tf32_symbol_admission("k", 0, 0, 120, 128, 1024, 256).is_ok());
        let spill = tf32_symbol_admission("k", 8, 0, 120, 128, 1024, 256).unwrap_err();
        assert!(
            spill.contains("8 bytes of Driver JIT local memory"),
            "{spill}"
        );
        // A symbol whose measurement recorded a spill keeps that much and
        // no more.
        assert!(tf32_symbol_admission("k", 88, 88, 120, 128, 1024, 256).is_ok());
        let over = tf32_symbol_admission("k", 92, 88, 120, 128, 1024, 256).unwrap_err();
        assert!(over.contains("above the 88 bytes"), "{over}");
        let registers = tf32_symbol_admission("k", 0, 0, 129, 128, 1024, 256).unwrap_err();
        assert!(registers.contains("129 registers"), "{registers}");
        let threads = tf32_symbol_admission("k", 0, 0, 120, 128, 128, 256).unwrap_err();
        assert!(threads.contains("cannot launch 256 threads"), "{threads}");
    }

    #[test]
    fn sm120_exact_reachability_loader_masks_only_exact_inventory_symbols() {
        use super::{Tf32SymbolExclusion, retain_specialized_tf32_candidate};
        use crate::mamba_ssm::gpu::gemm_bi_triad::contract::{
            F32TriadAvailability, SM120_FMA_ROUTE_SPECS, SM120_TF32_ROUTE_SPECS, Sm120FmaRoute,
            Sm120FmaTile, Tf32QualifiedModule,
        };

        let measured = Sm120FmaRoute {
            tile: Sm120FmaTile::M64N128,
            kvec: false,
            splits: 2,
        };
        let excluded_symbol = "nt_sm120_tma_fma_m64n128_bk16_s2";
        let exclusions = [
            Tf32SymbolExclusion {
                symbol: excluded_symbol,
                reason: "test exact exclusion".into(),
            },
            Tf32SymbolExclusion {
                symbol: "nt_sm120_tma_mma_tf32_m64n64_bk32_s2",
                reason: "test non-exact exclusion".into(),
            },
        ];
        let target = KernelCudaTarget::new("compute_120").unwrap();
        let device_target = KernelCudaTarget::new("sm_120").unwrap();
        let compiler = CompilerIdentity {
            source_digest: [0x21; 32],
            invocation_digest: [0x22; 32],
            header_manifest_digest: [0x23; 32],
            target,
            nvrtc_version: (13, 0),
            nvrtc_library_domain: [0x24; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        let binding = Tf32QualifiedModule {
            module_kind: ModuleKind::TriadSm120,
            target,
            artifact: ArtifactIdentity {
                module_kind: ModuleKind::TriadSm120,
                artifact_kind: ArtifactKind::Ptx,
                compile_key: compiler.invocation_digest,
                artifact_digest: [0x25; 32],
            },
            compiler,
            device: DeviceIdentity {
                compute_capability: (12, 0),
                multiprocessor_count: 170,
                target: device_target,
                driver: DriverIdentity {
                    api_version: 13_000,
                    build_sources: 1,
                    build_digest: [0x26; 32],
                },
            },
            device_caps: DeviceCaps {
                compute_capability: (12, 0),
                nvrtc_version: (13, 0),
                accepted_target: Some(target),
                optin_shared_bytes: 104_448,
                tensor_map_access: true,
            },
            sm120_fma_exclusions: Default::default(),
        };
        let mut functions = SM120_FMA_ROUTE_SPECS
            .iter()
            .filter(|spec| spec.symbol != excluded_symbol)
            .enumerate()
            .map(|(index, spec)| (spec.symbol, index as u8))
            .collect::<HashMap<_, _>>();
        let tf32_sibling = SM120_TF32_ROUTE_SPECS
            .iter()
            .find(|spec| !spec.route.is_exact_fma() && spec.op == ResolvedGemmOp::Nn)
            .unwrap()
            .symbol;
        functions.insert(tf32_sibling, 0xfe);
        let expected_functions = functions.clone();
        let (functions, specialized) =
            retain_specialized_tf32_candidate(functions, binding, &exclusions, Ok([0x27; 32]))
                .unwrap();
        let availability = F32TriadAvailability {
            specialized,
            ..Default::default()
        };
        let retained = availability
            .specialized
            .expect("one excluded symbol must not unbind the specialized module");

        assert_eq!(functions, expected_functions);
        assert!(functions.contains_key(tf32_sibling));
        assert!(
            retained
                .sm120_fma_exclusions
                .is_excluded(ResolvedGemmOp::Nt, measured)
        );
        let mut accepted_exact = 0;
        for spec in SM120_FMA_ROUTE_SPECS {
            let route = spec.route.exact_fma().unwrap();
            if spec.symbol == excluded_symbol {
                assert!(retained.sm120_fma_exclusions.is_excluded(spec.op, route));
            } else {
                accepted_exact += 1;
                assert!(functions.contains_key(spec.symbol), "{}", spec.symbol);
                assert!(
                    !retained.sm120_fma_exclusions.is_excluded(spec.op, route),
                    "{}",
                    spec.symbol
                );
            }
        }
        assert_eq!(accepted_exact, 11);
    }

    #[test]
    fn tf32_driver_jit_local_memory_admission_requires_zero() {
        let symbol = "nn_sm120_tma_mma_tf32_m64n128_bk32_s3";
        let qualified = Tf32DriverJitLocalMemoryFacts {
            module_kind: ModuleKind::TriadSm120,
            symbol,
            target: crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new("compute_120").unwrap(),
            nvrtc_version: (13, 2),
            nvrtc_library_known: true,
            nvrtc_library_current: true,
        };

        validate_tf32_driver_jit_local_memory_facts(0, qualified).unwrap();
        for local_bytes in [1, 15, 16, 17] {
            let error = validate_tf32_driver_jit_local_memory_facts(local_bytes, qualified)
                .expect_err("nonzero local-memory size must fail");
            assert!(error.contains(symbol), "{error}");
            assert!(error.contains(&local_bytes.to_string()), "{error}");
        }
    }

    #[test]
    fn tf32_register_caps_freeze_the_rect_wide_symbol_without_weakening_generic_caps() {
        const TAG33: &str = "nn_sm120_tma_mma_tf32_m80n32_bk64_s2";

        assert_eq!(
            tf32_register_cap(
                ModuleKind::TriadSm89Finalist,
                super::super::sm89_finalist_source::SM89_FINALIST_SYMBOL,
            ),
            Ok(125)
        );
        assert_eq!(
            tf32_required_occupancy(
                ModuleKind::TriadSm89Finalist,
                super::super::sm89_finalist_source::SM89_FINALIST_SYMBOL,
            ),
            2
        );
        assert!(
            tf32_register_cap(ModuleKind::TriadSm89Finalist, "foreign_finalist_symbol").is_err()
        );

        assert_eq!(tf32_register_cap(ModuleKind::TriadSm120, TAG33), Ok(80));
        assert_eq!(
            tf32_register_cap(
                ModuleKind::TriadSm120,
                "nn_sm120_tma_mma_tf32_m64n64_bk32_s2"
            ),
            Ok(128)
        );
        for mutated in [
            "nn_sm120_tma_mma_tf32_m80n32_bk64_s3",
            "nn_sm120_tma_mma_tf32_m80n32_bk32_s2",
            "tn_sm120_tma_mma_tf32_m80n32_bk64_s2",
            "nn_sm120_tma_mma_tf32_m80n32_bk64_s2_exp",
        ] {
            assert_eq!(
                tf32_register_cap(ModuleKind::TriadSm120, mutated),
                Ok(128),
                "nearby symbol must retain the generic SM120 cap: {mutated}"
            );
        }
        assert_eq!(tf32_register_cap(ModuleKind::TriadSm100, TAG33), Ok(128));
    }

    #[test]
    fn tf32_exceptional_qualifier_calls_launch_and_download_twice_in_order() {
        let trace = RefCell::new(Vec::new());
        let output = opaque_tf32_outputs();
        let downloads = RefCell::new(VecDeque::from([output.clone(), output]));

        let digest = qualify_tf32_conversion_artifact(
            tf32_test_artifact([0x42; 32]),
            || {
                trace.borrow_mut().push("launch");
                Ok(())
            },
            || {
                trace.borrow_mut().push("download");
                downloads
                    .borrow_mut()
                    .pop_front()
                    .ok_or_else(|| "missing injected output".to_string())
            },
        )
        .expect("stable artifact output must qualify");

        assert_eq!(
            *trace.borrow(),
            ["launch", "download", "launch", "download"]
        );
        assert_eq!(downloads.borrow().len(), 0);
        assert_ne!(digest, [0; 32]);
    }

    #[test]
    fn tf32_exceptional_qualifier_rejects_length_and_repeat_drift() {
        let stable = opaque_tf32_outputs();
        let mut changed = stable.clone();
        changed[6] ^= 1;
        for (label, first, second, expected) in [
            ("short first", stable[..9].to_vec(), stable.clone(), "first"),
            (
                "long first",
                [stable.as_slice(), &[0xdead_beef]].concat(),
                stable.clone(),
                "first",
            ),
            (
                "short second",
                stable.clone(),
                stable[..9].to_vec(),
                "second",
            ),
            (
                "long second",
                stable.clone(),
                [stable.as_slice(), &[0xdead_beef]].concat(),
                "second",
            ),
            ("bit drift", stable.clone(), changed, "changed"),
        ] {
            let downloads = RefCell::new(VecDeque::from([first, second]));
            let error = qualify_tf32_conversion_artifact(
                tf32_test_artifact([0x42; 32]),
                || Ok(()),
                || {
                    downloads
                        .borrow_mut()
                        .pop_front()
                        .ok_or_else(|| "missing injected output".to_string())
                },
            )
            .expect_err(label);
            assert!(error.contains(expected), "{label}: {error}");
        }
    }

    #[test]
    fn tf32_exceptional_qualifier_propagates_callbacks() {
        let error = qualify_tf32_conversion_artifact(
            tf32_test_artifact([0x42; 32]),
            || Err("injected first launch".into()),
            || panic!("download followed a failed launch"),
        )
        .expect_err("first launch failure must propagate");
        assert!(error.contains("injected first launch"), "{error}");

        let error = qualify_tf32_conversion_artifact(
            tf32_test_artifact([0x42; 32]),
            || Ok(()),
            || Err("injected first download".into()),
        )
        .expect_err("first download failure must propagate");
        assert!(error.contains("injected first download"), "{error}");

        let launches = Cell::new(0);
        let output = opaque_tf32_outputs();
        let error = qualify_tf32_conversion_artifact(
            tf32_test_artifact([0x42; 32]),
            || {
                let next = launches.get() + 1;
                launches.set(next);
                if next == 2 {
                    Err("injected second launch".into())
                } else {
                    Ok(())
                }
            },
            || Ok(output.clone()),
        )
        .expect_err("second launch failure must propagate");
        assert!(error.contains("injected second launch"), "{error}");

        let downloads = Cell::new(0);
        let error = qualify_tf32_conversion_artifact(
            tf32_test_artifact([0x42; 32]),
            || Ok(()),
            || {
                let next = downloads.get() + 1;
                downloads.set(next);
                if next == 2 {
                    Err("injected second download".into())
                } else {
                    Ok(output.clone())
                }
            },
        )
        .expect_err("second download failure must propagate");
        assert!(error.contains("injected second download"), "{error}");
    }

    #[test]
    fn tf32_exceptional_digest_is_artifact_scoped_and_outputs_are_opaque() {
        let qualify = |artifact| {
            let output = opaque_tf32_outputs();
            let downloads = RefCell::new(VecDeque::from([output.clone(), output]));
            qualify_tf32_conversion_artifact(
                artifact,
                || Ok(()),
                || {
                    downloads
                        .borrow_mut()
                        .pop_front()
                        .ok_or_else(|| "missing injected output".to_string())
                },
            )
            .expect("opaque stable outputs must qualify")
        };

        let first = qualify(tf32_test_artifact([0x42; 32]));
        assert_eq!(first, qualify(tf32_test_artifact([0x42; 32])));
        assert_ne!(first, qualify(tf32_test_artifact([0x43; 32])));
    }

    #[test]
    fn specialized_tf32_qualification_error_is_not_reported_as_unavailable() {
        let functions = HashMap::from([("tf32", 7_u8)]);
        let rejected_symbol = "nn_sm120_mma_tf32_m16n32_bk16_s4";
        let qualification_error = format!(
            "specialized TF32 symbol {rejected_symbol} rejected: registers 129 exceed limit 128"
        );
        let error =
            retain_tf32_candidate(functions.clone(), 9_u8, Err(qualification_error.clone()))
                .expect_err("specialized qualification failure must abort module initialization");
        assert_eq!(error, qualification_error);
        assert!(error.contains(rejected_symbol), "{error}");
        assert!(error.contains("registers 129 exceed limit 128"), "{error}");

        let (kept_functions, kept_binding) =
            retain_tf32_candidate(functions.clone(), 9_u8, Ok([0x55; 32])).unwrap();
        assert_eq!(kept_functions, functions);
        assert_eq!(kept_binding, Some(9));

        let (empty_functions, empty_binding) =
            retain_tf32_candidate(HashMap::<&'static str, u8>::new(), 9_u8, Ok([0x55; 32]))
                .unwrap();
        assert!(empty_functions.is_empty());
        assert_eq!(empty_binding, None);
    }

    #[test]
    fn specialized_tf32_function_load_preserves_symbol_and_resource_error() {
        let rejected_symbol = "nt_sm80_mma_tf32_splitk4_m16n32_bk32_s4";
        let load_error = format!(
            "load specialized TF32 symbol {rejected_symbol}: dynamic shared memory 65536 exceeds device limit 49152"
        );
        let error = retain_forced_only_functions::<u8>(Err(load_error.clone()))
            .expect_err("specialized function-load failure must abort module initialization");
        assert_eq!(error, load_error);
        assert!(error.contains(rejected_symbol), "{error}");
        assert!(
            error.contains("dynamic shared memory 65536 exceeds device limit 49152"),
            "{error}"
        );

        let functions = HashMap::from([("forced", 7_u8)]);
        assert_eq!(
            retain_forced_only_functions(Ok((functions.clone(), Vec::new())))
                .unwrap()
                .0,
            functions
        );
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum CudaTarget {
        Compute110f,
        Compute110a,
        Sm110f,
        Sm110a,
    }

    fn synthetic_tf32_ptx(module_kind: ModuleKind) -> String {
        let target = match module_kind {
            ModuleKind::TriadSm80 => "sm_80",
            ModuleKind::TriadSm90a => "sm_90a",
            ModuleKind::TriadSm100 => "sm_100f",
            ModuleKind::TriadSm120 => "sm_120",
            _ => panic!("no TF32 inventory for {module_kind:?}"),
        };
        let mut ptx = format!(".version 9.0\n.target {target}\n");
        for kernel_spec in super::super::contract::tf32_route_specs(module_kind) {
            ptx.push_str(&format!(".entry {}(\n) {{}}\n", kernel_spec.symbol));
        }
        ptx
    }

    fn synthetic_tf32_abi_ptx(module_kind: ModuleKind, map_alignment: usize) -> String {
        let mut ptx = String::new();
        for kernel_spec in super::super::contract::tf32_route_specs(module_kind) {
            ptx.push_str(&format!(".visible .entry {}(\n", kernel_spec.symbol));
            if matches!(
                module_kind,
                ModuleKind::TriadSm80 | ModuleKind::TriadSm89Finalist
            ) {
                for parameter in 0..4 {
                    ptx.push_str(&format!(".param .u64 p{parameter},\n"));
                }
                ptx.push_str(".param .align 4 .b8 bundle[32]\n) {}\n");
            } else {
                ptx.push_str(".param .u64 output,\n");
                let exact = kernel_spec.route.is_exact_fma();
                if exact
                    || matches!(
                        kernel_spec.route,
                        super::super::contract::Tf32PhysicalRoute::Sm120TmaMmaTf32RnaStreamKV1(_)
                    )
                {
                    ptx.push_str(".param .u64 partial,\n.param .u64 flags,\n");
                }
                ptx.push_str(&format!(
                    ".param .align {map_alignment} .b8 a_map[128],\n\
                     .param .align {map_alignment} .b8 b_map[128],\n"
                ));
                ptx.push_str(".param .u64 bias,\n");
                let bundle = if exact { 32 } else { 40 };
                ptx.push_str(&format!(".param .align 4 .b8 bundle[{bundle}]\n) {{}}\n"));
            }
        }
        ptx
    }

    fn synthetic_splitk_ptx() -> String {
        let parameters = ".param .u64 p0,\n.param .u64 p1,\n.param .u64 p2,\n.param .u64 p3,\n.param .u64 p4,\n.param .u64 p5,\n.param .align 4 .b8 bundle[32]";
        let mut ptx = ".version 9.0\n.target sm_80\n".to_string();
        for spec in super::super::contract::TF32_SPLITK_CANDIDATE_SPECS {
            let epilogue = if spec.op == ResolvedGemmOp::Nn {
                " fma.rn.f32 %f5, %f1, %f2, %f3;"
            } else {
                ""
            };
            let body = format!(
                "cvt.rna.tf32.f32 %r1, %f1; mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32; st.global.cg.f32 [%rd2], %f1; st.global.cg.v2.f32 [%rd2], {{%f1, %f2}}; membar.gl; atom.global.inc.u32 %r2, [%rd1], {}; ld.global.cg.f32 %f1, [%rd2]; ld.global.cg.v2.f32 {{%f1, %f2}}, [%rd2]; add.rn.f32 %f1, %f2, %f3; mul.rn.f32 %f4, %f1, %f2;{epilogue} ret;",
                spec.partitions - 1,
            );
            ptx.push_str(&format!(
                ".visible .entry {}(\n{}\n) {{ {} }}\n",
                spec.symbol, parameters, body,
            ));
        }
        ptx
    }

    fn mutate_splitk_entry(ptx: &str, symbol: &str, from: &str, to: &str) -> String {
        let entry = ptx_entry(ptx, symbol).unwrap();
        let changed = entry.replacen(from, to, 1);
        assert_ne!(changed, entry, "{symbol} fixture did not contain {from}");
        ptx.replacen(&entry, &changed, 1)
    }

    #[test]
    fn splitk_candidate_ptx_inventory_and_parameter_abi_are_exact() {
        let valid = synthetic_splitk_ptx();
        validate_tf32_splitk_ptx(false, &valid).unwrap();
        for spec in super::super::contract::TF32_SPLITK_CANDIDATE_SPECS {
            let missing = valid.replacen(spec.symbol, "removed_splitk_fused", 1);
            assert!(validate_tf32_splitk_ptx(false, &missing).is_err());
        }
        let foreign = valid.replace(
            ".version 9.0",
            ".version 9.0\n.visible .entry foreign_tf32_splitk2_kernel() { ret; }",
        );
        assert!(validate_tf32_splitk_ptx(false, &foreign).is_err());
        let wrong_bundle = valid.replacen("bundle[32]", "bundle[40]", 1);
        assert!(validate_tf32_splitk_ptx(false, &wrong_bundle).is_err());
        let float_atomic = valid.replacen("atom.global.inc.u32", "atom.global.add.f32", 1);
        assert!(validate_tf32_splitk_ptx(false, &float_atomic).is_err());
        let missing_counter = valid.replacen("atom.global.inc.u32", "add.u32", 1);
        assert!(validate_tf32_splitk_ptx(false, &missing_counter).is_err());
        let duplicate_counter = valid.replacen(
            "atom.global.inc.u32 %r2, [%rd1], 1;",
            "atom.global.inc.u32 %r2, [%rd1], 1; atom.global.inc.u32 %r3, [%rd1], 1;",
            1,
        );
        assert!(validate_tf32_splitk_ptx(false, &duplicate_counter).is_err());
        let wrong_k2_limit = valid.replacen(
            "atom.global.inc.u32 %r2, [%rd1], 1;",
            "atom.global.inc.u32 %r2, [%rd1], 3;",
            1,
        );
        assert!(validate_tf32_splitk_ptx(false, &wrong_k2_limit).is_err());
        let register_limit = valid.replacen(
            "atom.global.inc.u32 %r2, [%rd1], 1;",
            "atom.global.inc.u32 %r2, [%rd1], %r9;",
            1,
        );
        assert!(validate_tf32_splitk_ptx(false, &register_limit).is_err());
        let formatted_limit = valid.replacen(
            "atom.global.inc.u32 %r2, [%rd1], 1;",
            "atom.global.inc.u32\n    %r2, [ %rd1 ], 1 ;",
            1,
        );
        validate_tf32_splitk_ptx(false, &formatted_limit).unwrap();
        for opcode in [
            "st.global.cg.f32",
            "st.global.cg.v2.f32",
            "ld.global.cg.f32",
            "ld.global.cg.v2.f32",
        ] {
            let stale_visibility = valid.replacen(opcode, &opcode.replace(".cg", ".wb"), 1);
            assert!(validate_tf32_splitk_ptx(false, &stale_visibility).is_err());
        }
        let missing_fence = valid.replacen("membar.gl", "bar.sync 0", 1);
        assert!(validate_tf32_splitk_ptx(false, &missing_fence).is_err());
        let atomic_before_fence = valid.replacen(
            "membar.gl; atom.global.inc.u32 %r2, [%rd1], 1;",
            "atom.global.inc.u32 %r2, [%rd1], 1; membar.gl;",
            1,
        );
        assert!(validate_tf32_splitk_ptx(false, &atomic_before_fence).is_err());
        let reload_before_atomic = valid.replacen(
            "atom.global.inc.u32 %r2, [%rd1], 1; ld.global.cg.f32 %f1, [%rd2];",
            "ld.global.cg.f32 %f1, [%rd2]; atom.global.inc.u32 %r2, [%rd1], 1;",
            1,
        );
        assert!(validate_tf32_splitk_ptx(false, &reload_before_atomic).is_err());
        let division = valid.replacen(
            "fma.rn.f32 %f5, %f1, %f2, %f3; ret;",
            "fma.rn.f32 %f5, %f1, %f2, %f3; div.u32 %r4, %r2, %r3; ret;",
            1,
        );
        assert!(validate_tf32_splitk_ptx(false, &division).is_err());
        let fused_division = valid.replacen(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32; st.global.cg.f32",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32; div.u32 %r4, %r2, %r3; st.global.cg.f32",
            1,
        );
        assert!(validate_tf32_splitk_ptx(false, &fused_division).is_err());

        for spec in super::super::contract::TF32_SPLITK_CANDIDATE_SPECS
            .iter()
            .filter(|spec| spec.op == ResolvedGemmOp::Nt)
        {
            let missing_fence = mutate_splitk_entry(&valid, spec.symbol, "membar.gl", "bar.sync 0");
            assert!(validate_tf32_splitk_ptx(false, &missing_fence).is_err());

            let atomic_before_fence = mutate_splitk_entry(
                &valid,
                spec.symbol,
                "membar.gl; atom.global.inc.u32",
                "atom.global.inc.u32 %r7, [%rd7], 0; membar.gl; atom.global.inc.u32",
            );
            assert!(validate_tf32_splitk_ptx(false, &atomic_before_fence).is_err());

            let reload_before_atomic = mutate_splitk_entry(
                &valid,
                spec.symbol,
                "atom.global.inc.u32",
                "ld.global.cg.f32 %f7, [%rd7]; atom.global.inc.u32",
            );
            assert!(validate_tf32_splitk_ptx(false, &reload_before_atomic).is_err());

            let float_atomic = mutate_splitk_entry(
                &valid,
                spec.symbol,
                "atom.global.inc.u32",
                "atom.global.add.f32",
            );
            assert!(validate_tf32_splitk_ptx(false, &float_atomic).is_err());

            let division = mutate_splitk_entry(
                &valid,
                spec.symbol,
                "mul.rn.f32",
                "div.u32 %r7, %r8, %r9; mul.rn.f32",
            );
            assert!(validate_tf32_splitk_ptx(false, &division).is_err());
        }

        for spec in super::super::contract::TF32_SPLITK_CANDIDATE_SPECS
            .iter()
            .filter(|spec| spec.partitions == 8)
        {
            let wrong_limit = mutate_splitk_entry(
                &valid,
                spec.symbol,
                "atom.global.inc.u32 %r2, [%rd1], 7;",
                "atom.global.inc.u32 %r2, [%rd1], 3;",
            );
            assert!(validate_tf32_splitk_ptx(false, &wrong_limit).is_err());
        }
    }

    #[test]
    fn scalar_tn_narrow_splitm_partial_ptx_abi_is_fail_closed() {
        let parameters = ".param .u64 partial,\n.param .u64 a,\n.param .u64 b,\n.param .u32 m,\n.param .u32 k,\n.param .u32 n,\n.param .u32 chunk";
        let body =
            "fma.rn.f32 %f1, %f2, %f3, %f4; st.global.v4.f32 [%rd1], {%f1, %f2, %f3, %f4}; ret;";
        let mut valid = ".version 9.0\n.target sm_80\n".to_string();
        for symbol in super::SCALAR_TN_NARROW_SPLITM_PARTIAL_SYMBOLS {
            valid.push_str(&format!(
                ".visible .entry {symbol}(\n{parameters}\n) {{ {body} }}\n"
            ));
        }
        validate_tn_narrow_splitm_partial_ptx(&valid).unwrap();
        for symbol in super::SCALAR_TN_NARROW_SPLITM_PARTIAL_SYMBOLS {
            assert!(
                validate_tn_narrow_splitm_partial_ptx(&valid.replacen(
                    symbol,
                    "removed_tn_narrow_splitm_partial",
                    1,
                ))
                .is_err()
            );
        }
        assert!(
            validate_tn_narrow_splitm_partial_ptx(&valid.replacen(
                ".param .u32 chunk",
                ".param .u64 chunk",
                1,
            ))
            .is_err()
        );
        assert!(
            validate_tn_narrow_splitm_partial_ptx(&valid.replacen("fma.rn.f32", "mul.rn.f32", 1,))
                .is_err()
        );
        let runtime_division = valid.replacen("ret;", "rem.s32 %r4, %r5, %r6; ret;", 1);
        assert!(validate_tn_narrow_splitm_partial_ptx(&runtime_division).is_err());
    }

    fn synthetic_scalar_splitm_module_ptx() -> String {
        let zero_parameters = ".param .u64 output,\n.param .u64 a,\n.param .u64 b,\n.param .u64 bias,\n.param .align 4 .b8 bundle[32]";
        let partial_parameters = ".param .u64 partial,\n.param .u64 a,\n.param .u64 b,\n.param .u32 m,\n.param .u32 k,\n.param .u32 n,\n.param .u32 chunk";
        let partial_body =
            "fma.rn.f32 %f1, %f2, %f3, %f4; st.global.v4.f32 [%rd1], {%f1, %f2, %f3, %f4}; ret;";
        let generic_partial_body = "div.u32 %r1, %r2, %r3; rem.u32 %r4, %r2, %r3; fma.rn.f32 %f1, %f2, %f3, %f4; st.global.v4.f32 [%rd1], {%f1, %f2, %f3, %f4}; ret;";
        let mut ptx = ".version 9.0\n.target sm_80\n".to_string();
        for symbol in super::SCALAR_ZERO_REDUCTION_SYMBOLS {
            ptx.push_str(&format!(
                ".visible .entry {symbol}(\n{zero_parameters}\n) {{ ret; }}\n"
            ));
        }
        for symbol in super::SCALAR_TN_NARROW_SPLITM_PARTIAL_SYMBOLS {
            ptx.push_str(&format!(
                ".visible .entry {symbol}(\n{partial_parameters}\n) {{ {partial_body} }}\n"
            ));
        }
        for symbol in ["tn_splitm_partial", "tn_splitm_partial_aligned"] {
            ptx.push_str(&format!(
                ".visible .entry {symbol}(\n{partial_parameters}\n) {{ {generic_partial_body} }}\n"
            ));
        }
        for symbol in super::SCALAR_SYMBOLS {
            if super::SCALAR_ZERO_REDUCTION_SYMBOLS.contains(symbol)
                || super::SCALAR_TN_NARROW_SPLITM_PARTIAL_SYMBOLS.contains(symbol)
                || super::SCALAR_TN_SPLITM_PARTIAL_SYMBOLS.contains(symbol)
            {
                continue;
            }
            if *symbol == "nt_m2n16_bk64_splitk32" {
                let parameters = ".param .u64 output,\n.param .u64 a,\n.param .u64 b,\n.param .f32 alpha,\n.param .u32 m,\n.param .u32 n,\n.param .u32 k_out";
                let body = "fma.rn.f32 %f1, %f2, %f3, %f4; add.rn.f32 %f5, %f1, %f4; mul.rn.f32 %f6, %f5, %f2; ret;";
                ptx.push_str(&format!(
                    ".visible .entry {symbol}(\n{parameters}\n) {{ {body} }}\n"
                ));
                continue;
            }
            if *symbol == "nn_splitk32_m32n64_exact" {
                let parameters = ".param .u64 partial,\n.param .u64 a,\n.param .u64 b,\n.param .u32 m,\n.param .u32 n,\n.param .u32 chunks,\n.param .u32 lda";
                let body = "fma.rn.f32 %f1, %f2, %f3, %f4; ret;";
                ptx.push_str(&format!(
                    ".visible .entry {symbol}(\n{parameters}\n) {{ {body} }}\n"
                ));
                continue;
            }
            if *symbol == "tn_m16n16_bk16_s2_splitm16" {
                let parameters = ".param .u64 output,\n.param .u64 a,\n.param .u64 b,\n.param .f32 alpha,\n.param .u32 m,\n.param .u32 k,\n.param .u32 n";
                let body = "fma.rn.f32 %f1, %f2, %f3, %f4; add.rn.f64 %fd1, %fd2, %fd3; mul.rn.f64 %fd4, %fd1, %fd2; cvt.rn.f32.f64 %f6, %fd4; add.rn.f32 %f5, %f1, %f6; ret;";
                ptx.push_str(&format!(
                    ".visible .entry {symbol}(\n{parameters}\n) {{ {body} }}\n"
                ));
                continue;
            }
            ptx.push_str(&format!(".visible .entry {symbol}() {{ ret; }}\n"));
        }
        ptx
    }

    #[test]
    fn triad_scalar_module_validation_rejects_m2n16_contract_mutations() {
        const SYMBOL: &str = "nt_m2n16_bk64_splitk32";
        let valid = synthetic_scalar_splitm_module_ptx();
        validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &valid).unwrap();

        for (label, from, to) in [
            (
                "seven-parameter ABI mutation",
                ".param .f32 alpha",
                ".param .u32 alpha",
            ),
            ("missing FFMA", "fma.rn.f32", "mad.rn.f32"),
            ("missing rounded add", "add.rn.f32", "sub.rn.f32"),
            ("missing rounded multiply", "mul.rn.f32", "div.rn.f32"),
            (
                "atomic reduction",
                "ret;",
                "atom.global.add.f32 %f7, [%rd7], %f1; ret;",
            ),
            (
                "red reduction",
                "ret;",
                "red.global.add.f32 [%rd7], %f1; ret;",
            ),
            (
                "redux reduction",
                "ret;",
                "redux.sync.add.u32 %r7, %r8, 0xffffffff; ret;",
            ),
            (
                "MMA instruction",
                "ret;",
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32; ret;",
            ),
            (
                "flush-to-zero instruction",
                "ret;",
                "add.rn.ftz.f32 %f7, %f1, %f2; ret;",
            ),
            ("device call", "ret;", "call.uni (); ret;"),
        ] {
            let mutated = mutate_splitk_entry(&valid, SYMBOL, from, to);
            assert!(
                validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &mutated).is_err(),
                "TriadScalar module validation accepted {label} in {SYMBOL}"
            );
        }
    }

    #[test]
    fn triad_scalar_module_validation_rejects_nn_m32n64_splitk32_mutations() {
        const SYMBOL: &str = "nn_splitk32_m32n64_exact";
        let valid = synthetic_scalar_splitm_module_ptx();
        validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &valid).unwrap();

        for (label, from, to) in [
            (
                "seven-parameter ABI mutation",
                ".param .u32 lda",
                ".param .u64 lda",
            ),
            ("missing FFMA", "fma.rn.f32", "mad.rn.f32"),
            (
                "atomic reduction",
                "ret;",
                "atom.global.add.f32 %f7, [%rd7], %f1; ret;",
            ),
            (
                "red reduction",
                "ret;",
                "red.global.add.f32 [%rd7], %f1; ret;",
            ),
            (
                "redux reduction",
                "ret;",
                "redux.sync.add.u32 %r7, %r8, 0xffffffff; ret;",
            ),
            (
                "MMA instruction",
                "ret;",
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32; ret;",
            ),
            (
                "flush-to-zero instruction",
                "ret;",
                "add.rn.ftz.f32 %f7, %f1, %f2; ret;",
            ),
            ("device call", "ret;", "call.uni (); ret;"),
        ] {
            let mutated = mutate_splitk_entry(&valid, SYMBOL, from, to);
            assert!(
                validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &mutated).is_err(),
                "TriadScalar module validation accepted {label} in {SYMBOL}"
            );
        }
    }

    #[test]
    fn triad_scalar_module_validation_rejects_tn_m16n16_contract_mutations() {
        const SYMBOL: &str = "tn_m16n16_bk16_s2_splitm16";
        let valid = synthetic_scalar_splitm_module_ptx();
        validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &valid).unwrap();

        for (label, from, to) in [
            (
                "seven-parameter ABI mutation",
                ".param .f32 alpha",
                ".param .u32 alpha",
            ),
            ("missing FFMA", "fma.rn.f32", "mad.rn.f32"),
            ("missing rounded f64 add", "add.rn.f64", "sub.rn.f64"),
            ("missing rounded f64 multiply", "mul.rn.f64", "div.rn.f64"),
            (
                "missing rounded f64 conversion",
                "cvt.rn.f32.f64",
                "cvt.rz.f32.f64",
            ),
            ("missing rounded output add", "add.rn.f32", "sub.rn.f32"),
            (
                "reordered reducer scale",
                "add.rn.f64 %fd1, %fd2, %fd3; mul.rn.f64 %fd4, %fd1, %fd2;",
                "mul.rn.f64 %fd4, %fd1, %fd2; add.rn.f64 %fd1, %fd2, %fd3;",
            ),
            (
                "atomic reduction",
                "ret;",
                "atom.global.add.f32 %f7, [%rd7], %f1; ret;",
            ),
            (
                "red reduction",
                "ret;",
                "red.global.add.f32 [%rd7], %f1; ret;",
            ),
            (
                "redux reduction",
                "ret;",
                "redux.sync.add.u32 %r7, %r8, 0xffffffff; ret;",
            ),
            (
                "MMA instruction",
                "ret;",
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32; ret;",
            ),
            (
                "flush-to-zero instruction",
                "ret;",
                "add.rn.ftz.f32 %f7, %f1, %f2; ret;",
            ),
            ("device call", "ret;", "call.uni (); ret;"),
        ] {
            let mutated = mutate_splitk_entry(&valid, SYMBOL, from, to);
            assert!(
                validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &mutated).is_err(),
                "TriadScalar module validation accepted {label} in {SYMBOL}"
            );
        }
    }

    #[test]
    fn triad_scalar_module_validation_rejects_generic_splitm_contract_mutations() {
        let valid = synthetic_scalar_splitm_module_ptx();
        validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &valid).unwrap();
        for symbol in ["tn_splitm_partial", "tn_splitm_partial_aligned"] {
            let missing = valid.replacen(symbol, "removed_tn_splitm_partial", 1);
            assert!(
                validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &missing).is_err(),
                "TriadScalar module validation accepted missing {symbol}"
            );
            for (label, from, to) in [
                (
                    "seven-parameter ABI mutation",
                    ".param .u32 chunk",
                    ".param .u64 chunk",
                ),
                ("missing FFMA", "fma.rn.f32", "mul.rn.f32"),
                (
                    "atomic reduction",
                    "ret;",
                    "atom.global.add.f32 %f5, [%rd2], %f1; ret;",
                ),
                (
                    "red reduction",
                    "ret;",
                    "red.global.add.f32 [%rd2], %f1; ret;",
                ),
                (
                    "redux reduction",
                    "ret;",
                    "redux.sync.add.u32 %r7, %r8, 0xffffffff; ret;",
                ),
            ] {
                let mutated = mutate_splitk_entry(&valid, symbol, from, to);
                assert!(
                    validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &mutated).is_err(),
                    "TriadScalar module validation accepted {label} in {symbol}"
                );
            }
        }
    }

    #[test]
    fn triad_scalar_module_validation_requires_the_exact_whole_module_exports() {
        const M64N64_SYMBOL: &str = "nn_m64n64_bk16_s2";
        const M32N64_SPLITK32_SYMBOL: &str = "nn_splitk32_m32n64_exact";
        const PRISM_M64N64_SYMBOL: &str = "nn_prism_m64n64_bk16_s2";
        const D768_TRANSPOSE_SYMBOL: &str = "transpose_f32_32x16_d768";
        const TN_M16N16_SYMBOL: &str = "tn_m16n16_bk16_s2_splitm16";
        let valid = synthetic_scalar_splitm_module_ptx();
        assert_eq!(super::SCALAR_SYMBOLS.len(), 56);
        validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &valid)
            .expect("complete TriadScalar export inventory");

        let mut foreign = valid.clone();
        foreign.push_str(".visible .entry foreign_scalar_kernel() { ret; }\n");
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &foreign)
            .expect_err("TriadScalar module validation accepted a foreign export");
        assert!(error.contains("foreign_scalar_kernel"), "{error}");

        let missing = valid.replacen(
            &format!(".visible .entry {M64N64_SYMBOL}()"),
            ".visible .entry removed_m64n64_kernel()",
            1,
        );
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &missing)
            .expect_err("TriadScalar module validation accepted a missing M64N64 export");
        assert!(error.contains("removed_m64n64_kernel"), "{error}");

        let missing = valid.replacen(M32N64_SPLITK32_SYMBOL, "removed_m32n64_splitk32_kernel", 1);
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &missing)
            .expect_err("TriadScalar module validation accepted a missing M32N64 Split-K export");
        assert!(error.contains("removed_m32n64_splitk32_kernel"), "{error}");

        let duplicate = format!("{valid}.visible .entry {M64N64_SYMBOL}() {{ ret; }}\n");
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &duplicate)
            .expect_err("TriadScalar module validation accepted a duplicate M64N64 export");
        assert!(error.contains(M64N64_SYMBOL), "{error}");

        let missing = valid.replacen(
            &format!(".visible .entry {PRISM_M64N64_SYMBOL}()"),
            ".visible .entry removed_prism_m64n64_kernel()",
            1,
        );
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &missing)
            .expect_err("TriadScalar module validation accepted a missing prism M64N64 export");
        assert!(error.contains(PRISM_M64N64_SYMBOL), "{error}");

        let duplicate = format!("{valid}.visible .entry {PRISM_M64N64_SYMBOL}() {{ ret; }}\n");
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &duplicate)
            .expect_err("TriadScalar module validation accepted a duplicate prism M64N64 export");
        assert!(error.contains(PRISM_M64N64_SYMBOL), "{error}");

        let missing = valid.replacen(
            &format!(".visible .entry {D768_TRANSPOSE_SYMBOL}()"),
            ".visible .entry removed_d768_transpose_kernel()",
            1,
        );
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &missing)
            .expect_err("TriadScalar module validation accepted a missing d768 transpose export");
        assert!(error.contains(D768_TRANSPOSE_SYMBOL), "{error}");

        let duplicate = format!("{valid}.visible .entry {D768_TRANSPOSE_SYMBOL}() {{ ret; }}\n");
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &duplicate)
            .expect_err("TriadScalar module validation accepted a duplicate d768 transpose export");
        assert!(error.contains(D768_TRANSPOSE_SYMBOL), "{error}");

        let missing = valid.replacen(
            &format!(".visible .entry {TN_M16N16_SYMBOL}("),
            ".visible .entry removed_tn_m16n16_kernel(",
            1,
        );
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &missing)
            .expect_err("TriadScalar module validation accepted a missing TN M16N16 export");
        assert!(error.contains(TN_M16N16_SYMBOL), "{error}");

        let duplicate = format!("{valid}.visible .entry {TN_M16N16_SYMBOL}() {{ ret; }}\n");
        let error = validate_module_ptx(ModuleKind::TriadScalar, "sm_80", &duplicate)
            .expect_err("TriadScalar module validation accepted a duplicate TN M16N16 export");
        assert!(error.contains(TN_M16N16_SYMBOL), "{error}");
    }

    fn whole_module_symbols(module_kind: ModuleKind) -> Vec<&'static str> {
        let mut symbols = match module_kind {
            ModuleKind::TriadSm90a => SM90A_SYMBOLS.to_vec(),
            ModuleKind::TriadSm100 => super::super::contract::SM100_KERNEL_SPECS
                .iter()
                .map(|spec| spec.symbol)
                .collect(),
            ModuleKind::TriadSm120 => super::super::contract::sm120_kernel_specs()
                .map(|spec| spec.symbol)
                .collect(),
            _ => panic!("no whole-module fixture for {module_kind:?}"),
        };
        symbols.extend(super::super::contract::tf32_module_symbols(module_kind));
        symbols
    }

    fn representative_entry_instructions(
        module_kind: ModuleKind,
        symbol: &str,
    ) -> Vec<&'static str> {
        let mut instructions = match module_kind {
            ModuleKind::TriadSm90a => vec![
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
                "wgmma.fence.sync.aligned",
                "wgmma.commit_group.sync.aligned",
                "wgmma.wait_group.sync.aligned",
            ],
            ModuleKind::TriadSm100 => vec![
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
                "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64",
                "tcgen05.fence::before_thread_sync",
                "tcgen05.fence::after_thread_sync",
                "tcgen05.ld.sync.aligned.32x32b.x8.b32",
                "tcgen05.wait::ld.sync.aligned",
            ],
            ModuleKind::TriadSm120 => vec![
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                "mbarrier.init.shared::cta.b64",
                "fence.mbarrier_init.release.cluster",
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
                "st.global.b32",
            ],
            _ => panic!("no representative body for {module_kind:?}"),
        };

        match module_kind {
            ModuleKind::TriadSm90a => {
                let tf32 = super::super::contract::tf32_module_symbols(module_kind)
                    .any(|expected| expected == symbol);
                if tf32 || symbol.contains("_wg1") {
                    instructions.extend([
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
                    ]);
                }
                if symbol.contains("_wg2") {
                    instructions.push("setmaxnreg.inc.sync.aligned.u32");
                    if tf32 {
                        instructions.push("setmaxnreg.dec.sync.aligned.u32");
                    }
                }
                instructions.push(if tf32 {
                    "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32"
                } else if symbol.ends_with("_bf16") {
                    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16"
                } else {
                    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16"
                });
            }
            ModuleKind::TriadSm100 => {
                instructions.push(if symbol.contains("_tf32_") {
                    "tcgen05.mma.cta_group::1.kind::tf32"
                } else {
                    "tcgen05.mma.cta_group::1.kind::f16"
                });
                if symbol.starts_with("nn_") {
                    instructions.extend([
                        "tcgen05.st.sync.aligned.32x32b.x8.b32",
                        "tcgen05.wait::st.sync.aligned",
                    ]);
                }
            }
            ModuleKind::TriadSm120 => {
                if symbol.contains("_tma_fma_") {
                    instructions.push("fma.rn.f32");
                } else if symbol.contains("_tf32_") {
                    instructions.extend([
                        "cvt.rna.tf32.f32",
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                    ]);
                } else {
                    instructions.push("mbarrier.arrive.release.cta.shared::cta.b64");
                    let dtype = if symbol.ends_with("_bf16") {
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
                    } else {
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                    };
                    instructions.push(dtype);
                    instructions.extend(if symbol.starts_with("tn_") {
                        [
                            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
                        ]
                    } else if symbol.starts_with("nt_") {
                        [
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                            "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
                        ]
                    } else {
                        [
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
                        ]
                    });
                }
            }
            _ => unreachable!(),
        }
        instructions
    }

    fn sm90a_wg2_producer_symbol(entry: &str) -> Option<&'static str> {
        SM90A_SYMBOLS[6..]
            .iter()
            .position(|&symbol| symbol == entry)
            .map(|index| {
                [
                    "_ZL18sm90a_wg2_producerILi0EEv",
                    "_ZL18sm90a_wg2_producerILi1EEv",
                    "_ZL18sm90a_wg2_producerILi2EEv",
                ][index / 2]
            })
    }

    fn sm90a_wg2_producer_instructions() -> [&'static str; 5] {
        [
            "setmaxnreg.dec.sync.aligned.u32",
            "bar.sync",
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
        ]
    }

    fn sm90a_wg2_producer_function(symbol: &str, body: &[&str]) -> String {
        let body = body
            .iter()
            .map(|instruction| format!("    {instruction};\n"))
            .collect::<String>();
        format!(".visible .func {symbol}() {{\n{body}    ret;\n}}\n")
    }

    fn append_sm90a_wg2_producer_functions(ptx: &mut String) {
        for symbol in [
            "_ZL18sm90a_wg2_producerILi0EEv",
            "_ZL18sm90a_wg2_producerILi1EEv",
            "_ZL18sm90a_wg2_producerILi2EEv",
        ] {
            ptx.push_str(&sm90a_wg2_producer_function(
                symbol,
                &sm90a_wg2_producer_instructions(),
            ));
        }
    }

    fn whole_module_entry(symbol: &str, body: &[&str]) -> String {
        let mut body = body
            .iter()
            .map(|instruction| format!("    {instruction};\n"))
            .collect::<String>();
        if let Some(target) = sm90a_wg2_producer_symbol(symbol) {
            body.push_str(&format!("    call.uni {target}, ();\n"));
        }
        format!(".entry {symbol}(\n) {{\n{body}}}\n")
    }

    fn whole_module_fixture(module_kind: ModuleKind, target: &str) -> String {
        let symbols = whole_module_symbols(module_kind);
        let expected_count = match module_kind {
            ModuleKind::TriadSm90a => 18,
            ModuleKind::TriadSm100 => 108,
            ModuleKind::TriadSm120 => 128,
            _ => unreachable!(),
        };
        assert_eq!(symbols.len(), expected_count);

        let mut ptx = format!(".version 9.0\n.target {target}\n");
        for symbol in symbols {
            let body = representative_entry_instructions(module_kind, symbol);
            ptx.push_str(&whole_module_entry(symbol, &body));
        }
        if module_kind == ModuleKind::TriadSm90a {
            append_sm90a_wg2_producer_functions(&mut ptx);
        }
        ptx
    }

    fn whole_module_with_bodies(
        module_kind: ModuleKind,
        target: &str,
        mut body: impl FnMut(&str) -> Vec<&'static str>,
    ) -> String {
        let mut ptx = format!(".version 9.0\n.target {target}\n");
        for symbol in whole_module_symbols(module_kind) {
            ptx.push_str(&whole_module_entry(symbol, &body(symbol)));
        }
        ptx
    }

    fn omnibus_entry_instructions(module_kind: ModuleKind) -> Vec<&'static str> {
        let mut instructions = BTreeSet::new();
        for symbol in whole_module_symbols(module_kind) {
            instructions.extend(representative_entry_instructions(module_kind, symbol));
        }
        instructions.into_iter().collect()
    }

    fn validate_specialized_fixture(
        module_kind: ModuleKind,
        arch: &str,
        ptx: &str,
    ) -> Result<(), String> {
        match module_kind {
            ModuleKind::TriadSm90a => validate_sm90a_ptx(ptx),
            ModuleKind::TriadSm100 => validate_sm100_ptx(arch, ptx),
            ModuleKind::TriadSm120 => validate_sm120_ptx(arch, ptx),
            _ => panic!("no specialized fixture validator for {module_kind:?}"),
        }
    }

    fn sm100_probe_instructions() -> [&'static str; 14] {
        [
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
            "tcgen05.mma.cta_group::1.kind::f16",
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64",
            "tcgen05.fence::before_thread_sync",
            "tcgen05.fence::after_thread_sync",
            "tcgen05.ld.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::ld.sync.aligned",
            "tcgen05.st.sync.aligned.32x32b.x8.b32",
            "tcgen05.wait::st.sync.aligned",
        ]
    }

    fn sm100_probe_fixture(body: &str) -> String {
        format!(".version 9.0\n.target sm_100f\n.visible .entry tcgen05_probe()\n{{\n{body}\n}}\n")
    }

    fn assert_whole_module_export_mutations_fail(
        module_kind: ModuleKind,
        target: &str,
        validate: impl Fn(&str) -> Result<(), String>,
    ) {
        let valid = whole_module_fixture(module_kind, target);
        validate(&valid).expect("complete whole-module fixture must pass");
        let tf32 = super::super::contract::tf32_module_symbols(module_kind)
            .next()
            .expect("specialized module TF32 export");
        let entry = whole_module_entry(tf32, &representative_entry_instructions(module_kind, tf32));

        let missing = valid.replacen(&entry, "", 1);
        let error = validate(&missing).expect_err("removed TF32 export must fail");
        assert!(error.contains(tf32), "{error}");

        let typed = match module_kind {
            ModuleKind::TriadSm90a => SM90A_SYMBOLS[0],
            ModuleKind::TriadSm100 => super::super::contract::SM100_KERNEL_SPECS[0].symbol,
            ModuleKind::TriadSm120 => super::super::contract::SM120_KERNEL_SPECS[0].symbol,
            _ => unreachable!(),
        };
        let typed_entry = whole_module_entry(
            typed,
            &representative_entry_instructions(module_kind, typed),
        );
        let missing = valid.replacen(&typed_entry, "", 1);
        let error = validate(&missing).expect_err("removed typed export must fail");
        assert!(error.contains(typed), "{error}");

        let duplicate = format!("{valid}\n{entry}");
        let error = validate(&duplicate).expect_err("duplicate TF32 export must fail");
        assert!(error.contains(tf32), "{error}");

        let foreign = format!("{valid}\n.entry harmless_foreign_export(\n) {{}}\n");
        let error = validate(&foreign).expect_err("foreign export must fail");
        assert!(error.contains("harmless_foreign_export"), "{error}");
    }

    #[test]
    fn specialized_whole_module_validators_require_exact_export_sets() {
        assert_whole_module_export_mutations_fail(ModuleKind::TriadSm90a, "sm_90a", |ptx| {
            validate_sm90a_ptx(ptx)
        });
        assert_whole_module_export_mutations_fail(ModuleKind::TriadSm100, "sm_100f", |ptx| {
            validate_sm100_ptx("compute_100f", ptx)
        });
        assert_whole_module_export_mutations_fail(ModuleKind::TriadSm120, "sm_121", |ptx| {
            validate_sm120_ptx("compute_121", ptx)
        });
    }

    #[test]
    fn sm90a_wg2_exports_require_exact_producer_linkage_and_helper_protocol() {
        let valid = whole_module_fixture(ModuleKind::TriadSm90a, "sm_90a");
        validate_sm90a_ptx(&valid).expect("complete SM90a producer fixture");
        let entry_symbol = SM90A_SYMBOLS[6];
        let paired_symbol = SM90A_SYMBOLS[7];
        let target = sm90a_wg2_producer_symbol(entry_symbol).unwrap();
        let foreign_target = sm90a_wg2_producer_symbol(SM90A_SYMBOLS[8]).unwrap();
        let call = format!("    call.uni {target}, ();\n");
        let function = sm90a_wg2_producer_function(target, &sm90a_wg2_producer_instructions());

        let missing_call = valid.replacen(&call, "", 1);
        let wrong_target = valid.replacen(
            &format!("    call.uni {target}, ();\n"),
            &format!("    call.uni {foreign_target}, ();\n"),
            1,
        );
        let missing_helper = valid.replacen(&function, "", 1);
        let duplicate_helper = format!("{valid}\n{function}");
        let stub_helper =
            valid.replacen(&function, &sm90a_wg2_producer_function(target, &["ret"]), 1);
        let mut incomplete = sm90a_wg2_producer_instructions().to_vec();
        incomplete.retain(|instruction| {
            *instruction
                != "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        });
        let incomplete_helper = sm90a_wg2_producer_function(target, &incomplete);
        let unrelated_helper = sm90a_wg2_producer_function(
            "unrelated_protocol_holder",
            &sm90a_wg2_producer_instructions(),
        );
        let unrelated_protocol = format!(
            "{}\n{unrelated_helper}",
            valid.replacen(&function, &incomplete_helper, 1)
        );
        let paired_call = format!(
            "    call.uni {}, ();\n",
            sm90a_wg2_producer_symbol(paired_symbol).unwrap()
        );
        let duplicate_call =
            valid.replacen(&paired_call, &format!("{paired_call}{paired_call}"), 1);

        for (label, malformed) in [
            ("missing call", missing_call),
            ("wrong call target", wrong_target),
            ("missing helper", missing_helper),
            ("duplicate helper", duplicate_helper),
            ("stub helper", stub_helper),
            ("protocol in unrelated helper", unrelated_protocol),
            ("duplicate producer call", duplicate_call),
        ] {
            assert!(
                validate_sm90a_ptx(&malformed).is_err(),
                "SM90a validator accepted {label}"
            );
        }
    }

    #[test]
    fn specialized_validators_reject_instruction_spoofs_in_unused_functions() {
        for (module_kind, arch, target) in [
            (ModuleKind::TriadSm90a, "compute_90a", "sm_90a"),
            (ModuleKind::TriadSm100, "compute_100f", "sm_100f"),
            (ModuleKind::TriadSm120, "compute_121", "sm_121"),
        ] {
            let mut spoofed = whole_module_with_bodies(module_kind, target, |_| vec!["ret"]);
            let omnibus = omnibus_entry_instructions(module_kind);
            spoofed.push_str(".visible .func unused_feature_spoof() {\n");
            for instruction in omnibus {
                spoofed.push_str(&format!("    {instruction};\n"));
            }
            spoofed.push_str("}\n");

            validate_specialized_fixture(module_kind, arch, &spoofed)
                .expect_err("unused functions and quoted metadata must not supply export features");
        }
    }

    #[test]
    fn specialized_validators_reject_instruction_spoofs_in_quoted_metadata() {
        for (module_kind, arch, target) in [
            (ModuleKind::TriadSm90a, "compute_90a", "sm_90a"),
            (ModuleKind::TriadSm100, "compute_100f", "sm_100f"),
            (ModuleKind::TriadSm120, "compute_121", "sm_121"),
        ] {
            let mut spoofed = whole_module_with_bodies(module_kind, target, |_| vec!["ret"]);
            for (index, instruction) in omnibus_entry_instructions(module_kind)
                .into_iter()
                .enumerate()
            {
                spoofed.push_str(&format!(".file {index} \"{instruction}\"\n"));
                spoofed.push_str(&format!(".pragma \"{instruction}\";\n"));
            }
            validate_specialized_fixture(module_kind, arch, &spoofed)
                .expect_err("quoted metadata must not supply export features");
        }
    }

    #[test]
    fn specialized_validators_require_core_compute_in_every_export_body() {
        for (module_kind, arch, target) in [
            (ModuleKind::TriadSm90a, "compute_90a", "sm_90a"),
            (ModuleKind::TriadSm100, "compute_100f", "sm_100f"),
            (ModuleKind::TriadSm120, "compute_121", "sm_121"),
        ] {
            let first = whole_module_symbols(module_kind)[0];
            let omnibus = omnibus_entry_instructions(module_kind);
            let one_real = whole_module_with_bodies(module_kind, target, |symbol| {
                if symbol == first {
                    omnibus.clone()
                } else {
                    vec!["ret"]
                }
            });
            validate_specialized_fixture(module_kind, arch, &one_real)
                .expect_err("one real kernel must not validate a module of stub exports");
        }
    }

    #[test]
    fn specialized_validators_reject_wrong_core_compute_for_one_export() {
        let sm90a_tf32 = super::super::contract::tf32_module_symbols(ModuleKind::TriadSm90a)
            .next()
            .unwrap();
        let sm100_tf32 = super::super::contract::tf32_module_symbols(ModuleKind::TriadSm100)
            .next()
            .unwrap();
        let sm120_tf32 = super::super::contract::tf32_module_symbols(ModuleKind::TriadSm120)
            .next()
            .unwrap();
        for (module_kind, arch, target, symbol, correct, wrong) in [
            (
                ModuleKind::TriadSm90a,
                "compute_90a",
                "sm_90a",
                SM90A_SYMBOLS[0],
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16",
            ),
            (
                ModuleKind::TriadSm100,
                "compute_100f",
                "sm_100f",
                super::super::contract::SM100_KERNEL_SPECS[0].symbol,
                "tcgen05.mma.cta_group::1.kind::f16",
                "tcgen05.mma.cta_group::1.kind::tf32",
            ),
            (
                ModuleKind::TriadSm120,
                "compute_121",
                "sm_121",
                super::super::contract::SM120_KERNEL_SPECS[0].symbol,
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            ),
            (
                ModuleKind::TriadSm90a,
                "compute_90a",
                "sm_90a",
                sm90a_tf32,
                "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32",
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16",
            ),
            (
                ModuleKind::TriadSm100,
                "compute_100f",
                "sm_100f",
                sm100_tf32,
                "tcgen05.mma.cta_group::1.kind::tf32",
                "tcgen05.mma.cta_group::1.kind::f16",
            ),
            (
                ModuleKind::TriadSm120,
                "compute_121",
                "sm_121",
                sm120_tf32,
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            ),
        ] {
            let valid = whole_module_fixture(module_kind, target);
            let correct_entry = whole_module_entry(
                symbol,
                &representative_entry_instructions(module_kind, symbol),
            );
            let wrong_entry = correct_entry.replacen(correct, wrong, 1);
            let wrong_core = valid.replacen(&correct_entry, &wrong_entry, 1);
            validate_specialized_fixture(module_kind, arch, &wrong_core)
                .expect_err("one export with the wrong compute instruction must fail");
        }
    }

    #[test]
    fn specialized_validators_reject_additive_incompatible_core_in_same_entry() {
        let sm90a_tf32 = super::super::contract::tf32_module_symbols(ModuleKind::TriadSm90a)
            .next()
            .unwrap();
        let sm100_tf32 = super::super::contract::tf32_module_symbols(ModuleKind::TriadSm100)
            .next()
            .unwrap();
        let sm120_tf32 = super::super::contract::tf32_module_symbols(ModuleKind::TriadSm120)
            .next()
            .unwrap();
        for (module_kind, arch, target, symbol, sibling) in [
            (
                ModuleKind::TriadSm90a,
                "compute_90a",
                "sm_90a",
                SM90A_SYMBOLS[0],
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16",
            ),
            (
                ModuleKind::TriadSm90a,
                "compute_90a",
                "sm_90a",
                SM90A_SYMBOLS[1],
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
            ),
            (
                ModuleKind::TriadSm90a,
                "compute_90a",
                "sm_90a",
                sm90a_tf32,
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
            ),
            (
                ModuleKind::TriadSm100,
                "compute_100f",
                "sm_100f",
                super::super::contract::SM100_KERNEL_SPECS[0].symbol,
                "tcgen05.mma.cta_group::1.kind::tf32",
            ),
            (
                ModuleKind::TriadSm100,
                "compute_100f",
                "sm_100f",
                sm100_tf32,
                "tcgen05.mma.cta_group::1.kind::f16",
            ),
            (
                ModuleKind::TriadSm120,
                "compute_121",
                "sm_121",
                super::super::contract::SM120_KERNEL_SPECS[0].symbol,
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            ),
            (
                ModuleKind::TriadSm120,
                "compute_121",
                "sm_121",
                super::super::contract::SM120_KERNEL_SPECS[1].symbol,
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            ),
            (
                ModuleKind::TriadSm120,
                "compute_121",
                "sm_121",
                sm120_tf32,
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            ),
        ] {
            let valid = whole_module_fixture(module_kind, target);
            let entry = whole_module_entry(
                symbol,
                &representative_entry_instructions(module_kind, symbol),
            );
            let additive_entry = entry.replacen("}\n", &format!("    {sibling};\n}}\n"), 1);
            let additive = valid.replacen(&entry, &additive_entry, 1);
            assert!(
                validate_specialized_fixture(module_kind, arch, &additive).is_err(),
                "{module_kind:?}/{symbol} accepted sibling core {sibling}"
            );
        }
    }

    #[test]
    fn specialized_validators_require_protocol_in_each_export_body() {
        for (module_kind, arch, target, symbol, protocol) in [
            (
                ModuleKind::TriadSm90a,
                "compute_90a",
                "sm_90a",
                SM90A_SYMBOLS[0],
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64",
            ),
            (
                ModuleKind::TriadSm100,
                "compute_100f",
                "sm_100f",
                super::super::contract::SM100_KERNEL_SPECS[0].symbol,
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
            ),
            (
                ModuleKind::TriadSm120,
                "compute_121",
                "sm_121",
                super::super::contract::SM120_KERNEL_SPECS[0].symbol,
                "mbarrier.init.shared::cta.b64",
            ),
        ] {
            let valid = whole_module_fixture(module_kind, target);
            let entry = whole_module_entry(
                symbol,
                &representative_entry_instructions(module_kind, symbol),
            );
            let missing_protocol = entry.replacen(&format!("    {protocol};\n"), "", 1);
            let malformed = valid.replacen(&entry, &missing_protocol, 1);
            validate_specialized_fixture(module_kind, arch, &malformed)
                .expect_err("a protocol token in another export must not repair this export");
        }
    }

    #[test]
    fn specialized_validators_ignore_quoted_and_commented_core_spoofs() {
        for (module_kind, arch, target, symbol, core) in [
            (
                ModuleKind::TriadSm90a,
                "compute_90a",
                "sm_90a",
                SM90A_SYMBOLS[0],
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
            ),
            (
                ModuleKind::TriadSm100,
                "compute_100f",
                "sm_100f",
                super::super::contract::SM100_KERNEL_SPECS[0].symbol,
                "tcgen05.mma.cta_group::1.kind::f16",
            ),
            (
                ModuleKind::TriadSm120,
                "compute_121",
                "sm_121",
                super::super::contract::SM120_KERNEL_SPECS[0].symbol,
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            ),
        ] {
            let valid = whole_module_fixture(module_kind, target);
            let entry = whole_module_entry(
                symbol,
                &representative_entry_instructions(module_kind, symbol),
            );
            let spoof = format!("    .pragma \"{core}\";\n    // {core}\n");
            let quoted = entry.replacen(&format!("    {core};\n"), &spoof, 1);
            let malformed = valid.replacen(&entry, &quoted, 1);
            validate_specialized_fixture(module_kind, arch, &malformed)
                .expect_err("quoted and commented core spellings must not count");
        }
    }

    #[test]
    fn specialized_base_admission_rejects_foreign_tf32_instruction_families() {
        for (module_kind, arch, target, forbidden) in [
            (
                ModuleKind::TriadSm90a,
                "compute_90a",
                "sm_90a",
                "cvt.rna.tf32.f32",
            ),
            (
                ModuleKind::TriadSm100,
                "compute_100f",
                "sm_100f",
                "cvt.rna.tf32.f32",
            ),
            (
                ModuleKind::TriadSm120,
                "compute_121",
                "sm_121",
                "tcgen05.mma.cta_group::1.kind::tf32",
            ),
        ] {
            let symbol = super::super::contract::tf32_module_symbols(module_kind)
                .next()
                .unwrap();
            let valid = whole_module_fixture(module_kind, target);
            let entry = whole_module_entry(
                symbol,
                &representative_entry_instructions(module_kind, symbol),
            );
            let foreign_entry = entry.replacen("}\n", &format!("    {forbidden};\n}}\n"), 1);
            let foreign = valid.replacen(&entry, &foreign_entry, 1);
            validate_specialized_fixture(module_kind, arch, &foreign)
                .expect_err("foreign TF32 instructions must make base admission fail");
        }
    }

    #[test]
    fn specialized_validators_accept_real_per_entry_representative_bodies() {
        for (module_kind, arch, target) in [
            (ModuleKind::TriadSm90a, "compute_90a", "sm_90a"),
            (ModuleKind::TriadSm100, "compute_100f", "sm_100f"),
            (ModuleKind::TriadSm120, "compute_121", "sm_121"),
        ] {
            let valid = whole_module_fixture(module_kind, target);
            validate_specialized_fixture(module_kind, arch, &valid)
                .unwrap_or_else(|error| panic!("{module_kind:?} representative PTX: {error}"));
        }
    }

    #[test]
    fn exact_export_set_rejects_duplicate_expected_symbols_with_a_useful_diff() {
        let ptx = ".version 9.0\n.target sm_90a\n.entry repeated_export(\n) {}\n";
        let error = validate_exact_ptx_exports(
            "duplicate-expected fixture",
            1,
            &["repeated_export", "repeated_export"],
            ptx,
        )
        .expect_err("duplicate expected exports must fail");
        assert!(error.contains("expected_duplicates"), "{error}");
        assert!(error.contains("repeated_export"), "{error}");
    }

    #[test]
    fn exact_export_set_rejects_wrong_expected_cardinality() {
        let ptx = ".version 9.0\n.target sm_90a\n.entry only_export(\n) {}\n";
        let error =
            validate_exact_ptx_exports("wrong-cardinality fixture", 2, &["only_export"], ptx)
                .expect_err("wrong expected export cardinality must fail");
        assert!(error.contains("expected_count=2"), "{error}");
        assert!(error.contains("expected_entries=1"), "{error}");
    }

    #[test]
    fn exact_export_parser_ignores_directives_in_line_and_block_comments() {
        let valid = whole_module_fixture(ModuleKind::TriadSm90a, "sm_90a");
        for spoof in [
            "// .entry harmless_line_spoof() {}",
            "/* .entry harmless_block_spoof() {} */",
        ] {
            validate_sm90a_ptx(&format!("{valid}\n{spoof}\n"))
                .unwrap_or_else(|error| panic!("comment spoof must be ignored: {error}"));
        }

        let symbol = SM90A_SYMBOLS[0];
        let entry = whole_module_entry(
            symbol,
            &representative_entry_instructions(ModuleKind::TriadSm90a, symbol),
        );
        let missing = valid.replacen(&entry, "", 1);
        for spoof in [
            format!("// .entry {symbol}() {{}}"),
            format!("/* .entry {symbol}() {{}} */"),
        ] {
            validate_sm90a_ptx(&format!("{missing}\n{spoof}\n"))
                .expect_err("commented expected export must not repair the inventory");
        }
    }

    #[test]
    fn exact_export_parser_accepts_ptx_whitespace_and_intervening_comments() {
        let valid = whole_module_fixture(ModuleKind::TriadSm90a, "sm_90a");
        let symbol = SM90A_SYMBOLS[0];
        let ordinary = format!(".entry {symbol}(");
        let spaced = format!(".entry\t/* directive gap */\n{symbol}\t(");
        validate_sm90a_ptx(&valid.replacen(&ordinary, &spaced, 1))
            .expect("PTX whitespace and comments must separate entry tokens");
    }

    #[test]
    fn exact_export_parser_rejects_obfuscated_duplicates_and_malformed_entries() {
        let valid = whole_module_fixture(ModuleKind::TriadSm90a, "sm_90a");
        let symbol = SM90A_SYMBOLS[0];
        let duplicate = format!("{valid}\n.entry/* gap */{symbol}() {{}}\n");
        validate_sm90a_ptx(&duplicate).expect_err("obfuscated duplicate export must fail");

        let ordinary = whole_module_entry(
            symbol,
            &representative_entry_instructions(ModuleKind::TriadSm90a, symbol),
        );
        let malformed = valid.replacen(&ordinary, &format!(".entry {symbol}(\n"), 1);
        validate_sm90a_ptx(&malformed).expect_err("unclosed entry directive must fail");

        let external = valid.replacen(&ordinary, &format!(".extern .entry {symbol}(\n) {{}}\n"), 1);
        validate_sm90a_ptx(&external).expect_err("extern entry must not count as an export");
    }

    #[test]
    fn exact_export_parser_accepts_parameterless_entries() {
        let ptx = ".version 9.0\n.target sm_90a\n.visible .func helper() { ret; }\n.visible .entry only_export { .pragma \"}\"; ret; }\n";
        validate_exact_ptx_exports("parameterless fixture", 1, &["only_export"], ptx)
            .expect("valid parameterless entry must be counted and bounded");
    }

    #[test]
    fn exact_export_parser_accepts_entry_scoped_pragma_before_body() {
        let ptx = ".version 9.0\n.target sm_90a\n.visible .entry only_export()\n.pragma \"nounroll\";\n{ ret; }\n.visible .func helper() { ret; }\n";
        validate_exact_ptx_exports("entry pragma fixture", 1, &["only_export"], ptx)
            .expect("entry-scoped pragma must not terminate the entry");

        let malformed =
            ".version 9.0\n.target sm_90a\n.entry only_export .func helper() { ret; }\n";
        validate_exact_ptx_exports("function confusion fixture", 1, &["only_export"], malformed)
            .expect_err("function directive must not supply an entry body");
    }

    #[test]
    fn exact_export_parser_rejects_nested_module_directives_and_unbalanced_scopes() {
        let expected = ["only_export"];
        for ptx in [
            ".version 9.0\n.target sm_90a\n.visible .func helper() { .entry only_export() { ret; } }\n",
            ".version 9.0\n.visible .func helper() { .target sm_90a; ret; }\n.entry only_export() { ret; }\n",
            ".version 9.0\n.target sm_90a\n.visible .func helper() { ret;\n.entry only_export() { ret; }\n",
            ".version 9.0\n.target sm_90a\n.entry only_export() { ret; }\n}\n",
            ".version 9.0\n.target sm_90a\n.visible .func outer() { .func nested() { ret; } }\n.entry only_export() { ret; }\n",
        ] {
            validate_exact_ptx_exports("nested-scope fixture", 1, &expected, ptx)
                .expect_err("nested directives and unbalanced scopes must fail closed");
        }

        let nested_target =
            ".version 9.0\n.target sm_90a\n.entry only_export() { .target sm_90a; ret; }\n";
        assert!(
            parse_ptx(nested_target).is_err(),
            "an entry-local target directive must be rejected"
        );
    }

    #[test]
    fn target_parser_ignores_comments_and_accepts_ptx_whitespace() {
        let valid = whole_module_fixture(ModuleKind::TriadSm90a, "sm_90a");
        let spaced = valid.replacen(".target sm_90a", ".target\t/* target gap */\nsm_90a", 1);
        validate_sm90a_ptx(&spaced).expect("PTX whitespace must separate target tokens");

        let spoofed = valid.replacen(".target sm_90a", "/*\n.target sm_90a\n*/\n.target sm_90", 1);
        validate_sm90a_ptx(&spoofed).expect_err("comment target must not spoof the real target");

        let duplicate = valid.replacen(".target sm_90a", ".target sm_90a\n.target sm_90a", 1);
        validate_sm90a_ptx(&duplicate).expect_err("duplicate target directive must fail");
    }

    #[test]
    fn tf32_contract_and_loader_inventories_match_cuda_exports() {
        for (module_kind, expected) in [
            (ModuleKind::TriadSm80, 18),
            (ModuleKind::TriadSm90a, 6),
            (ModuleKind::TriadSm100, 36),
            (ModuleKind::TriadSm120, 30),
        ] {
            let kernel_specs = super::super::contract::tf32_route_specs(module_kind);
            let symbols: BTreeSet<_> =
                super::super::contract::tf32_module_symbols(module_kind).collect();
            assert_eq!(kernel_specs.len(), expected);
            assert_eq!(symbols.len(), expected);

            let source = compose_module_source(module_kind).unwrap();
            let load_function = |symbol: &str| source.contains(symbol);
            for kernel_spec in kernel_specs {
                assert!(load_function(kernel_spec.symbol));
            }
        }
        // The portable extension routes are composed for an sm80-family
        // target and absent from the CC 12.x composition.
        let extensions = super::super::contract::tf32_extension_route_specs(ModuleKind::TriadSm80);
        assert_eq!(extensions.len(), 1);
        let sm89 = compose_module_source_for(ModuleKind::TriadSm80, "sm_89").unwrap();
        let cc12 = compose_module_source_for(ModuleKind::TriadSm80, "compute_120").unwrap();
        for spec in extensions {
            assert!(sm89.contains(spec.symbol), "{}", spec.symbol);
            assert!(!cc12.contains(spec.symbol), "{}", spec.symbol);
        }
        assert_eq!(
            super::super::contract::tf32_route_specs_for(ModuleKind::TriadSm80, true).count(),
            19
        );
        assert_eq!(
            super::super::contract::tf32_route_specs_for(ModuleKind::TriadSm80, false).count(),
            18
        );
    }

    #[test]
    fn tf32_compiled_inventory_rejects_every_partial_module() {
        for module_kind in [
            ModuleKind::TriadSm80,
            ModuleKind::TriadSm90a,
            ModuleKind::TriadSm100,
            ModuleKind::TriadSm120,
        ] {
            let complete = synthetic_tf32_ptx(module_kind);
            validate_tf32_ptx_inventory(module_kind, false, &complete).unwrap();

            let mut symbols: BTreeSet<_> =
                super::super::contract::tf32_module_symbols(module_kind).collect();
            let removed = symbols.pop_first().unwrap();
            assert!(!symbols.remove(removed));
            let partial = complete.replacen(
                &format!(".entry {removed}("),
                ".entry removed_tf32_symbol(",
                1,
            );
            assert!(validate_tf32_ptx_inventory(module_kind, false, &partial).is_err());

            let duplicate = format!("{complete}\n.entry {removed}(\n) {{}}\n");
            assert!(validate_tf32_ptx_inventory(module_kind, false, &duplicate).is_err());
        }
    }

    #[test]
    fn tf32_compiled_inventory_admits_the_split_k_kernels_beside_the_routes() {
        // The split-K kernels ride in the portable module and carry the same
        // TF32 token as a route symbol; counting them as foreign entries
        // unbinds every portable TF32 route and sends TF32 work to the exact
        // scalar kernels without failing anything.
        let module_kind = ModuleKind::TriadSm80;
        let mut ptx = synthetic_tf32_ptx(module_kind);
        for spec in super::super::contract::tf32_splitk_specs_for(true) {
            ptx.push_str(&format!("\n.entry {}(\n) {{}}\n", spec.symbol));
        }
        validate_tf32_ptx_inventory(module_kind, false, &ptx).unwrap();
    }

    #[test]
    fn sm110_feature_candidates_exclude_ordinary_sm110() {
        let candidates = sm100_target_candidates((11, 0));
        let expected = [("compute_110f", "sm_110f"), ("compute_110a", "sm_110a")];
        let identity_names = [
            (CudaTarget::Compute110f, CudaTarget::Sm110f),
            (CudaTarget::Compute110a, CudaTarget::Sm110a),
        ];
        assert_eq!(identity_names.len(), expected.len());
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| (candidate.nvrtc_arch, candidate.ptx_target))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.nvrtc_arch != "compute_110")
        );
        assert!(validate_module_target(ModuleKind::TriadSm100, "compute_110").is_err());
        assert!(validate_module_target(ModuleKind::TriadSm100, "sm_110").is_err());
    }

    #[test]
    fn sm107_feature_candidates_exclude_ordinary_sm107() {
        let candidates = sm100_target_candidates((10, 7));
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| (candidate.nvrtc_arch, candidate.ptx_target))
                .collect::<Vec<_>>(),
            [("compute_107f", "sm_107f"), ("compute_107a", "sm_107a")]
        );
        assert!(validate_module_target(ModuleKind::TriadSm100, "compute_107f").is_ok());
        assert!(validate_module_target(ModuleKind::TriadSm100, "compute_107").is_err());
        assert!(validate_module_target(ModuleKind::TriadSm100, "sm_107").is_err());
        assert_eq!(super::sm80_ptx_target("sm_107a"), Some("sm_107a"));
        assert_eq!(portable_target_for_device((10, 7)), Ok("sm_107a"));
    }

    #[test]
    fn tf32_parameter_abi_tracks_cuda_12_and_13_tensor_map_alignment() {
        let portable = synthetic_tf32_abi_ptx(ModuleKind::TriadSm80, 64);
        validate_tf32_parameter_abi(ModuleKind::TriadSm80, false, &portable, 12).unwrap();
        validate_tf32_parameter_abi(ModuleKind::TriadSm80, false, &portable, 13).unwrap();
        let finalist = synthetic_tf32_abi_ptx(ModuleKind::TriadSm89Finalist, 64);
        validate_tf32_parameter_abi(ModuleKind::TriadSm89Finalist, false, &finalist, 12).unwrap();
        validate_tf32_parameter_abi(ModuleKind::TriadSm89Finalist, false, &finalist, 13).unwrap();

        for module_kind in [
            ModuleKind::TriadSm90a,
            ModuleKind::TriadSm100,
            ModuleKind::TriadSm120,
        ] {
            let cuda12 = synthetic_tf32_abi_ptx(module_kind, 64);
            let cuda13 = synthetic_tf32_abi_ptx(module_kind, 128);
            validate_tf32_parameter_abi(module_kind, false, &cuda12, 12).unwrap();
            validate_tf32_parameter_abi(module_kind, false, &cuda13, 13).unwrap();
            assert!(validate_tf32_parameter_abi(module_kind, false, &cuda12, 13).is_err());
            assert!(validate_tf32_parameter_abi(module_kind, false, &cuda13, 12).is_err());
        }
        assert!(validate_tf32_parameter_abi(ModuleKind::TriadSm90a, false, "", 14).is_err());
    }

    #[test]
    fn driver_abi_validates_and_formats_live_layout() {
        let abi =
            Tf32DriverAbi::checked(5, vec![(0, 8), (128, 128), (256, 128), (384, 8), (392, 40)])
                .expect("valid CUDA 13 tensor-map ABI");

        assert_eq!(abi.parameter_count(), 5);
        assert_eq!(
            abi.parameters()
                .iter()
                .map(|parameter| (parameter.offset(), parameter.size()))
                .collect::<Vec<_>>(),
            [(0, 8), (128, 128), (256, 128), (384, 8), (392, 40)]
        );
        assert_eq!(
            abi.tsv_record("nn_tf32_sm100_m128n256_s2")
                .expect("safe symbol"),
            "nn_tf32_sm100_m128n256_s2\t5\tptx_contract+cuFuncGetParamInfo_terminal_probe\t0:8,128:128,256:128,384:8,392:40"
        );
    }

    #[test]
    fn driver_abi_rejects_malformed_driver_results() {
        for (label, count, parameters) in [
            ("empty", 0, vec![]),
            ("count", 2, vec![(0, 8)]),
            ("first offset", 1, vec![(8, 8)]),
            ("zero size", 1, vec![(0, 0)]),
            ("overlap", 2, vec![(0, 16), (8, 8)]),
            ("overflow", 2, vec![(0, 8), (16, usize::MAX)]),
        ] {
            assert!(
                Tf32DriverAbi::checked(count, parameters).is_err(),
                "{label}"
            );
        }

        let abi = Tf32DriverAbi::checked(1, vec![(0, 8)]).unwrap();
        for symbol in ["", "bad\tsymbol", "bad\nsymbol", "bad\rsymbol"] {
            assert!(abi.tsv_record(symbol).is_err(), "{symbol:?}");
        }
    }

    #[test]
    fn driver_abi_live_query_requires_five_parameters_and_a_terminal_probe() {
        use cudarc::driver::sys::CUresult;

        let layout = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
        let mut queried = Vec::new();
        let abi = query_tf32_driver_parameter_abi(
            "TriadSm80/test",
            TF32_DRIVER_PARAMETER_COUNT,
            |index, offset, size| {
                queried.push(index);
                if let Some((value_offset, value_size)) = layout.get(index).copied() {
                    *offset = value_offset;
                    *size = value_size;
                    CUresult::CUDA_SUCCESS
                } else {
                    CUresult::CUDA_ERROR_INVALID_VALUE
                }
            },
        )
        .expect("five live parameters followed by the terminal probe");
        assert_eq!(queried, [0, 1, 2, 3, 4, 5]);
        assert_eq!(
            abi.parameters()
                .iter()
                .map(|parameter| (parameter.offset(), parameter.size()))
                .collect::<Vec<_>>(),
            layout
        );

        let split_layout = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 8), (40, 8), (48, 32)];
        let split =
            query_tf32_driver_parameter_abi("TriadSm80/split-K", 7, |index, offset, size| {
                if let Some((value_offset, value_size)) = split_layout.get(index).copied() {
                    *offset = value_offset;
                    *size = value_size;
                    CUresult::CUDA_SUCCESS
                } else {
                    CUresult::CUDA_ERROR_INVALID_VALUE
                }
            })
            .expect("seven live split-K parameters followed by the terminal probe");
        assert_eq!(split.parameter_count(), 7);
        assert_eq!(
            split
                .parameters()
                .iter()
                .map(|parameter| (parameter.offset(), parameter.size()))
                .collect::<Vec<_>>(),
            split_layout
        );

        let mut queried = Vec::new();
        let missing = query_tf32_driver_parameter_abi(
            "TriadSm80/missing",
            TF32_DRIVER_PARAMETER_COUNT,
            |index, offset, size| {
                queried.push(index);
                *offset = index * 8;
                *size = 8;
                if index == 3 {
                    CUresult::CUDA_ERROR_INVALID_VALUE
                } else {
                    CUresult::CUDA_SUCCESS
                }
            },
        )
        .expect_err("a required parameter may not terminate the ABI");
        assert_eq!(queried, [0, 1, 2, 3]);
        assert!(missing.contains("TriadSm80/missing[3]"));

        let extra = query_tf32_driver_parameter_abi(
            "TriadSm80/extra",
            TF32_DRIVER_PARAMETER_COUNT,
            |index, offset, size| {
                *offset = index * 8;
                *size = 8;
                CUresult::CUDA_SUCCESS
            },
        )
        .expect_err("a sixth live parameter must fail the ABI census");
        assert!(extra.contains("more than 5 Driver ABI parameters"));

        let sentinel = query_tf32_driver_parameter_abi(
            "TriadSm80/sentinel",
            TF32_DRIVER_PARAMETER_COUNT,
            |index, offset, size| {
                *offset = index * 8;
                *size = 8;
                if index == 5 {
                    CUresult::CUDA_ERROR_INVALID_CONTEXT
                } else {
                    CUresult::CUDA_SUCCESS
                }
            },
        )
        .expect_err("a non-terminal Driver error must not be accepted");
        assert!(sentinel.contains("TriadSm80/sentinel[5] sentinel"));
    }

    #[test]
    fn driver_abi_merge_rejects_cross_module_symbol_aliases() {
        let portable = std::collections::BTreeMap::from([(
            "portable",
            Tf32DriverAbi::checked(1, vec![(0, 8)]).unwrap(),
        )]);
        let specialized = std::collections::BTreeMap::from([(
            "specialized",
            Tf32DriverAbi::checked(1, vec![(0, 8)]).unwrap(),
        )]);
        let merged = merge_tf32_driver_abi(portable.clone(), Some(specialized))
            .expect("disjoint inventories");
        assert_eq!(
            merged.keys().copied().collect::<Vec<_>>(),
            ["portable", "specialized"]
        );

        let duplicate = std::collections::BTreeMap::from([(
            "portable",
            Tf32DriverAbi::checked(1, vec![(0, 8)]).unwrap(),
        )]);
        assert!(merge_tf32_driver_abi(portable.clone(), Some(duplicate.clone())).is_err());

        let (retained, finalist_rejection) =
            merge_optional_finalist_driver_abi(portable.clone(), Some(duplicate));
        assert_eq!(
            retained, portable,
            "an optional ABI conflict removed portable"
        );
        assert!(
            finalist_rejection
                .as_deref()
                .is_some_and(|reason| reason.contains("belongs to more than one module")),
            "the optional finalist conflict was not recorded"
        );
    }

    const FIXED_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/mamba_ssm.cu",
        "kernels/mamba_ssm_parallel.cu",
        "kernels/conv1d.cu",
        "kernels/activations.cu",
        "kernels/norms.cu",
        "kernels/elementwise.cu",
        "kernels/loss_scaler.cu",
        "kernels/grad_clip.cu",
        "kernels/adamw.cu",
        "kernels/gemm_bi_inference/common.cuh",
        "kernels/gemm_bi_inference/ffma.cu",
        "kernels/gemm_bi_inference/tf32.cu",
        "kernels/gemm_bi_inference/sm120/tf32.cu",
        "kernels/gemm_bi_inference/sm120/tma.cu",
        "kernels/gemm_bi_inference/wmma_legacy.cu",
        "kernels/gemm_bi_inference/matvec.cu",
        "kernels/gemm_bi_inference/mma16.cu",
        "kernels/gemm_bi_inference/tcw64.cu",
        "kernels/gemm_bi_inference/sm90a/wgmma.cu",
        "kernels/gemm_bi_inference/sm100/tcgen05.cu",
        "kernels/gemm_bi_inference/sm80/half_pipeline.cu",
        "kernels/gemm_bi_inference/sm80/f32_n64_copyplan.cu",
        "kernels/gemm_bi_inference/sm80/tf32_rna_wide.cu",
        "kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh",
        "kernels/gemm_bi_inference/sm80/half_swizzle.cu",
        "kernels/gemm_bi_inference/sm80/half_s3.cu",
        "kernels/gemm_bi_inference/sm80/tf32_rna_n96.cu",
        "kernels/gemm_bi_inference/sm80/half_n64.cu",
    ];

    const SCALAR_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/gemm_bi_triad/contract.cuh",
        "kernels/gemm_bi_triad/common.cuh",
        "kernels/gemm_bi_triad/epilogue.cuh",
        "kernels/gemm_bi_triad/scalar.cu",
        "kernels/gemm_bi_triad/scalar_nn_m64n64.cu",
        "kernels/gemm_bi_triad/scalar_nn_splitk_m32n64.cu",
        "kernels/gemm_bi_triad/scalar_nt_d768_transpose.cu",
        "kernels/gemm_bi_triad/scalar_nt_m2n16.cu",
        "kernels/gemm_bi_triad/scalar_tn_m16n16.cu",
    ];

    const SM80_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/gemm_bi_triad/contract.cuh",
        "kernels/gemm_bi_triad/common.cuh",
        "kernels/gemm_bi_triad/epilogue.cuh",
        "kernels/gemm_bi_triad/mma16.cuh",
        "kernels/gemm_bi_triad/sm80/mma.cu",
        "kernels/gemm_bi_triad/sm80/streamk.cu",
        "kernels/gemm_bi_triad/sm80/tf32_wide.cu",
        "kernels/gemm_bi_triad/sm80/tn_splitk.cu",
    ];

    const SM90A_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/gemm_bi_triad/contract.cuh",
        "kernels/gemm_bi_triad/common.cuh",
        "kernels/gemm_bi_triad/epilogue.cuh",
        "kernels/gemm_bi_triad/sm90a/wgmma.cu",
    ];

    const SM100_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/gemm_bi_triad/contract.cuh",
        "kernels/gemm_bi_triad/common.cuh",
        "kernels/gemm_bi_triad/epilogue.cuh",
        "kernels/gemm_bi_triad/sm100/tcgen05.cu",
    ];

    const SM120_FRAGMENTS: &[&str] = &[
        "kernels/_typed_prelude.cuh",
        "kernels/gemm_bi_triad/contract.cuh",
        "kernels/gemm_bi_triad/common.cuh",
        "kernels/gemm_bi_triad/epilogue.cuh",
        "kernels/gemm_bi_triad/sm120/tma.cu",
        "kernels/gemm_bi_triad/sm120/exact.cu",
    ];

    const SCALAR_SYMBOLS: &[&str] = &[
        "nn_big",
        "nn_m64n64_bk16_s2",
        "nn_splitk32_m32n64_exact",
        "nn_prism_m64n64_bk16_s2",
        "nn_zero_reduction",
        "tn_big",
        "tn_aligned",
        "tn_zero_reduction",
        "tn_narrow_splitm_partial",
        "tn_narrow_splitm_partial_aligned",
        "tn_splitm_partial",
        "tn_splitm_partial_aligned",
        "tn_m16n16_bk16_s2_splitm16",
        "splitm_reduce",
        "nt_big",
        "nt_m2n16_bk64_splitk32",
        "nt_zero_reduction",
        "nn_slim",
        "nn_splitk_slim_partial",
        "tn_slim",
        "nt_slim",
        "nn_ultra_thin",
        "nn_gemv",
        "tn_gemv",
        "nt_gemv",
        "nn_narrow",
        "nn_narrow_small",
        "tn_narrow",
        "nt_narrow",
        "nn_splitk32_partial",
        "splitk_reduce",
        "dx_col_gemv",
        "transpose_f32_2d",
        "transpose_f32_32x16_d768",
        "nn_gemv_bf16",
        "nn_gemv_f16",
        "tn_gemv_bf16",
        "tn_gemv_f16",
        "nt_gemv_bf16",
        "nt_gemv_f16",
        "nn_ultra_thin_bf16",
        "nn_ultra_thin_f16",
        "nn_narrow_bf16",
        "nn_narrow_f16",
        "nn_narrow_small_bf16",
        "nn_narrow_small_f16",
        "tn_narrow_bf16",
        "tn_narrow_f16",
        "nt_narrow_bf16",
        "nt_narrow_f16",
        "nn_big_bf16",
        "nn_big_f16",
        "tn_big_bf16",
        "tn_big_f16",
        "nt_big_bf16",
        "nt_big_f16",
    ];

    const SM80_SYMBOLS: &[&str] = &[
        "nn_tc_bf16",
        "nn_tc_f16",
        "tn_tc_bf16",
        "tn_tc_f16",
        "nt_tc_bf16",
        "nt_tc_f16",
        "nn_tc64_bf16",
        "nn_tc64_f16",
        "nn_tc16_bf16",
        "nn_tc16_f16",
        "tn_tc64_bf16",
        "tn_tc64_f16",
        "tn_tc128x64_bf16",
        "tn_tc128x64_f16",
        "nt_tc64_bf16",
        "nt_tc64_f16",
    ];

    fn assert_composition(kind: ModuleKind, expected_names: &[&str]) {
        let first = compose_module_source(kind).unwrap();
        let second = compose_module_source(kind).unwrap();
        assert_eq!(first, second, "{kind:?} composition changed between calls");
        assert!(first.ends_with('\n'), "{kind:?} has no terminal newline");
        assert!(!first.contains(env!("CARGO_MANIFEST_DIR")));
        let boundaries: Vec<_> = first
            .lines()
            .filter_map(|line| line.strip_prefix("#line 1 \"")?.strip_suffix('"'))
            .collect();
        assert_eq!(boundaries, expected_names, "{kind:?} source boundaries");
    }

    #[test]
    fn module_sources_have_exact_deterministic_boundaries() {
        assert_composition(ModuleKind::Fixed, FIXED_FRAGMENTS);
        let fixed_cc12 = super::compose_module_source_for(ModuleKind::Fixed, "compute_120")
            .expect("compose the Fixed module for compute_120");
        let fixed_boundaries: Vec<_> = fixed_cc12
            .lines()
            .filter_map(|line| line.strip_prefix("#line 1 \"")?.strip_suffix('"'))
            .collect();
        let mut expected_fixed_cc12 = FIXED_FRAGMENTS
            .iter()
            .copied()
            .filter(|name| {
                !matches!(
                    *name,
                    "kernels/gemm_bi_inference/sm80/half_pipeline.cu"
                        | "kernels/gemm_bi_inference/sm80/f32_n64_copyplan.cu"
                        | "kernels/gemm_bi_inference/sm80/tf32_rna_wide.cu"
                        | "kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh"
                        | "kernels/gemm_bi_inference/sm80/half_swizzle.cu"
                        | "kernels/gemm_bi_inference/sm80/half_s3.cu"
                        | "kernels/gemm_bi_inference/sm80/tf32_rna_n96.cu"
                        | "kernels/gemm_bi_inference/sm80/half_n64.cu"
                )
            })
            .collect::<Vec<_>>();
        expected_fixed_cc12.push("kernels/gemm_bi_inference/sm120/f32_n64_copyplan.cu");
        expected_fixed_cc12.push("kernels/gemm_bi_inference/sm120/f32_n64_sliced.cu");
        expected_fixed_cc12.push("kernels/gemm_bi_inference/sm120/f32_postbias.cu");
        assert_eq!(fixed_boundaries, expected_fixed_cc12);
        assert_composition(ModuleKind::TriadScalar, SCALAR_FRAGMENTS);
        assert_composition(ModuleKind::TriadSm80, SM80_FRAGMENTS);
        let portable_cc12 = super::compose_module_source_for(ModuleKind::TriadSm80, "compute_120")
            .expect("compose the portable module for compute_120");
        let boundaries: Vec<_> = portable_cc12
            .lines()
            .filter_map(|line| line.strip_prefix("#line 1 \"")?.strip_suffix('"'))
            .collect();
        assert_eq!(
            boundaries,
            &SM80_FRAGMENTS[..SM80_FRAGMENTS.len() - 3],
            "the CC 12.x portable module must not compose the extension fragments"
        );
        assert_composition(ModuleKind::TriadSm90a, SM90A_FRAGMENTS);
        assert_composition(ModuleKind::TriadSm100, SM100_FRAGMENTS);
        assert_composition(ModuleKind::TriadSm120, SM120_FRAGMENTS);

        assert!(
            !std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("kernels/gemm_bi_triad.cu")
                .exists(),
            "obsolete root triad monolith must not return"
        );
    }

    #[test]
    fn inference_composed_source_identity_is_frozen() {
        let cases = [
            (
                "base",
                compose_fragments(super::FIXED_SOURCE_FRAGMENTS).unwrap(),
                590_184,
                "8fea22457053245c3465295d3cd69717452fd44dce2628a1830bedc1dfe4fefe",
            ),
            (
                "sm_80",
                compose_module_source_for(ModuleKind::Fixed, "sm_80").unwrap(),
                697_604,
                "ed45b82ec0a49deaede546707e230c537e6fd45c2a19d2ae89fd1ae358c9c317",
            ),
            (
                "sm_89",
                compose_module_source_for(ModuleKind::Fixed, "sm_89").unwrap(),
                697_604,
                "ed45b82ec0a49deaede546707e230c537e6fd45c2a19d2ae89fd1ae358c9c317",
            ),
            (
                "compute_120",
                compose_module_source_for(ModuleKind::Fixed, "compute_120").unwrap(),
                693_372,
                "c88e385a33c4e91bebaa0633106e90f64a5280f5bd7967e8c95df6d140bb30c8",
            ),
        ];

        let mut moved = Vec::new();
        for (name, source, expected_len, expected_sha256) in cases {
            let sha256 = crate::mamba_ssm::gpu::kernel_identity::digest_hex(
                &super::FramedSha256::bytes(source.as_bytes()),
            );
            if source.len() != expected_len || sha256 != expected_sha256 {
                moved.push(format!("{name}: length {} sha256 {sha256}", source.len()));
            }
        }
        assert!(
            moved.is_empty(),
            "the Fixed composed source moved; refreeze every target from the live values \
             (and requalify the cohorts that pin it):\n{}",
            moved.join("\n")
        );
    }

    #[test]
    fn scalar_group_m_option_matches_the_source_guard() {
        let source = compose_module_source(ModuleKind::TriadScalar).unwrap();
        assert!(source.contains(&format!("#ifndef {SCALAR_GROUP_M_MACRO}")));
        assert_eq!(scalar_group_m_option("sm_80"), "-DGEMM_BI_GROUP_M=8");
        assert_eq!(scalar_group_m_option("sm_87"), "-DGEMM_BI_GROUP_M=8");
        assert_eq!(scalar_group_m_option("sm_89"), "-DGEMM_BI_GROUP_M=16");
        assert_eq!(scalar_group_m_option("sm_120"), "-DGEMM_BI_GROUP_M=16");
    }

    #[test]
    fn scalar_big_nt_source_uses_raw_to_conflict_free_b_staging() {
        let source = compose_module_source(ModuleKind::TriadScalar).unwrap();
        let start = source.find("void nt_big(").expect("Big NT entry");
        let end = source[start..]
            .find("void nn_slim(")
            .map(|offset| start + offset)
            .expect("Slim NN entry after Big NT");
        let kernel = &source[start..end];

        for required in [
            "constexpr int B_RAW_STAGE = GEMM_BI_SCALAR_BN * GEMM_BI_SCALAR_BK;",
            "constexpr int B_COMPUTE_STAGE = 2072;",
            "float* Braw = smem + K_PIPE * A_STAGE;",
            "float* Bcompute = Braw + B_RAW_STAGE;",
            "static_assert(TOTAL_SMEM_BYTES == 33376",
            "reinterpret_cast<const uint4*>(Braw",
            "reinterpret_cast<unsigned int*>(Bcompute)",
            "cp.async.ca.shared.global [%0], [%1], 16, %2;",
            "cp.async.ca.shared.global [%0], [%1], 4, %2;",
            "(N & 3) == 0",
            "(threadIdx.x >> 2) + _half * (GEMM_BI_SCALAR_BN / 2)",
            "(threadIdx.x & 3) * 4",
            "((n_local) * GEMM_BI_SCALAR_BN + ((n_local) >> 2) * B_COMPUTE_GROUP_PAD + (k_local))",
        ] {
            assert!(
                kernel.contains(required),
                "missing Big NT staging fragment: {required}"
            );
        }
        assert!(kernel.contains("? B + (long long)_g_k * N + _g_n : B"));
        assert_eq!(kernel.matches("for (int dotIdx = 0;").count(), 1);
        assert_eq!(kernel.matches("threadResults[idx] = __fmaf_rn(").count(), 1);
    }

    #[test]
    fn scalar_splitm_reducer_declares_a_portable_four_cta_bound() {
        let source = compose_module_source(ModuleKind::TriadScalar).unwrap();
        let entry = source
            .find("void splitm_reduce(")
            .expect("split-M reducer entry");
        let declaration = &source[entry.saturating_sub(160)..entry];
        assert!(declaration.contains("__launch_bounds__(256, 4)"));
        assert!(!declaration.contains("__launch_bounds__(256, 8)"));
    }

    #[test]
    fn scalar_f32_vector_paths_require_aligned_external_bases() {
        let source = compose_module_source(ModuleKind::TriadScalar).unwrap();
        let require = |body: &str, guards: &[&str]| -> Result<(), String> {
            for guard in guards {
                if !body.contains(guard) {
                    return Err(format!("missing scalar f32 vector guard {guard}"));
                }
            }
            Ok(())
        };

        for (start_name, end_name, guards) in [
            (
                "void nn_big(",
                "void tn_impl(",
                &["is_aligned_16(B)", "is_aligned_16(C)"][..],
            ),
            (
                "void tn_impl(",
                "void tn_big(",
                &["is_aligned_16(A)", "is_aligned_16(B)", "is_aligned_16(C)"][..],
            ),
            (
                "void tn_splitm_partial_impl(",
                "void tn_splitm_partial(",
                &["is_aligned_16(A)", "is_aligned_16(B)"][..],
            ),
            (
                "void nt_big(",
                "void nn_slim(",
                &["is_aligned_16(B)", "is_aligned_16(C)"][..],
            ),
            (
                "void nn_slim(",
                "void nn_splitk_slim_partial(",
                &["is_aligned_16(B)", "is_aligned_16(C)"][..],
            ),
            (
                "void nn_splitk_slim_partial(",
                "void tn_slim(",
                &["is_aligned_16(B)"][..],
            ),
            (
                "void tn_slim(",
                "void nt_slim(",
                &["is_aligned_16(A)", "is_aligned_16(B)", "is_aligned_16(C)"][..],
            ),
            (
                "void nt_slim(",
                "void nn_ultra_thin(",
                &["is_aligned_16(C)"][..],
            ),
            (
                "void nn_narrow(",
                "void nn_narrow_small(",
                &["is_aligned_16(B)"][..],
            ),
            (
                "void nn_narrow_small(",
                "void tn_narrow_splitm_impl(",
                &["is_aligned_16(B)"][..],
            ),
            (
                "void tn_narrow(",
                "void nt_narrow(",
                &["is_aligned_16(A)", "is_aligned_16(B)"][..],
            ),
            (
                "void nt_narrow(",
                "void nn_splitk32_partial(",
                &["is_aligned_16(B)"][..],
            ),
            (
                "void nn_splitk32_partial(",
                "void splitk_reduce(",
                &["is_aligned_16(B)"][..],
            ),
        ] {
            let start = source.find(start_name).expect("scalar f32 kernel start");
            let end = source[start..]
                .find(end_name)
                .map(|offset| start + offset)
                .expect("next scalar f32 kernel");
            let body = &source[start..end];
            require(body, guards).unwrap_or_else(|error| panic!("{start_name}: {error}"));
            for guard in guards {
                let mutated = body.replacen(guard, "gemm_bi_alignment_guard_removed()", 1);
                assert!(
                    require(&mutated, guards).is_err(),
                    "{start_name} accepted removal of {guard}"
                );
            }
        }
        for instantiation in [
            "tn_impl<false>(C, A, B, alpha, M_red, K_out, N);",
            "tn_impl<true>(C, A, B, alpha, M_red, K_out, N);",
            "tn_splitm_partial_impl<false>(",
            "tn_splitm_partial_impl<true>(",
        ] {
            assert!(
                source.contains(instantiation),
                "missing scalar TN alignment specialization {instantiation}"
            );
        }
    }

    #[test]
    fn quoted_include_allowlist_is_exact_and_fail_closed() {
        let accepted = compose_fragments(&[SourceFragment {
            logical_name: "kernels/synthetic.cu",
            source: "#include \"common.cuh\"\n#include <cuda_fp16.h>\nint value;\n",
            allowed_quoted_includes: &["common.cuh"],
        }])
        .unwrap();
        assert_eq!(
            accepted,
            "#line 1 \"kernels/synthetic.cu\"\n#include <cuda_fp16.h>\nint value;\n"
        );

        for source in [
            "#include \"forbidden.cuh\"\n",
            "# include \"forbidden.cuh\"\n",
            "#/**/include/**/\"forbidden.cuh\"\n",
            "%:include \"forbidden.cuh\"\n",
            "#inc\\\nlude \"forbidden.cuh\"\n",
        ] {
            let error = compose_fragments(&[SourceFragment {
                logical_name: "kernels/synthetic.cu",
                source,
                allowed_quoted_includes: &["common.cuh"],
            }])
            .expect_err("unlisted quoted include must fail closed");
            assert!(error.contains("kernels/synthetic.cu"), "{error}");
            assert!(error.contains("forbidden.cuh"), "{error}");
        }

        let error = compose_fragments(&[SourceFragment {
            logical_name: "kernels/synthetic.cu",
            source: "#include \"nested.cu\"\n",
            allowed_quoted_includes: &["nested.cu"],
        }])
        .expect_err("a CUDA source must not include another CUDA source");
        assert!(error.contains("nested.cu"), "{error}");

        for source in [
            "#define HEADER \"forbidden.cuh\"\n#include HEADER\n",
            "#include <nested.cu>\n",
            "#include \"unterminated.cuh\n",
            "#include \"escaped\\\".cuh\"\n",
            "#import \"forbidden.cuh\"\n",
            "%:import \"forbidden.cuh\"\n",
            "#include_next \"forbidden.cuh\"\n",
            "%:include_next \"forbidden.cuh\"\n",
        ] {
            let error = compose_fragments(&[SourceFragment {
                logical_name: "kernels/synthetic.cu",
                source,
                allowed_quoted_includes: &[],
            }])
            .expect_err("an unclassifiable or CUDA-source include must fail closed");
            assert!(error.contains("kernels/synthetic.cu"), "{error}");
            assert!(error.contains("include"), "{error}");
        }

        let error = compose_fragments(&[SourceFragment {
            logical_name: "kernels/synthetic.cu",
            source: "#include \"common.cuh\" garbage\n",
            allowed_quoted_includes: &["common.cuh"],
        }])
        .expect_err("trailing include tokens must fail closed");
        assert!(error.contains("include"), "{error}");

        let error = compose_fragments(&[SourceFragment {
            logical_name: "kernels/gemm_bi_triad/synthetic.cuh",
            source: "extern \"C\" __global__ void forbidden() {}\n",
            allowed_quoted_includes: &[],
        }])
        .expect_err("a triad header must not export a global kernel");
        assert!(error.contains("synthetic.cuh"), "{error}");

        let lookalikes = "// #include \"forbidden.cuh\"\n\
            const char* text = \"#include \\\"forbidden.cuh\\\"\";\n\
            #include <cuda_bf16.h>\n";
        let composed = compose_fragments(&[SourceFragment {
            logical_name: "kernels/lookalikes.cu",
            source: lookalikes,
            allowed_quoted_includes: &[],
        }])
        .unwrap();
        assert!(composed.contains("#include <cuda_bf16.h>"));
        assert!(composed.contains("const char* text"));
    }

    #[test]
    fn triad_symbol_inventories_are_exact_disjoint_and_owned() {
        assert_eq!(PRODUCTION_SCALAR_SYMBOLS, SCALAR_SYMBOLS);
        assert_eq!(PRODUCTION_SM80_SYMBOLS, SM80_SYMBOLS);

        let scalar: BTreeSet<_> = SCALAR_SYMBOLS.iter().copied().collect();
        let sm80: BTreeSet<_> = SM80_SYMBOLS.iter().copied().collect();
        assert_eq!(scalar.len(), 56);
        assert_eq!(sm80.len(), 16);
        let sm80_streamk: BTreeSet<_> = super::SM80_STREAMK_SYMBOLS.iter().copied().collect();
        assert_eq!(sm80_streamk.len(), 2);
        assert!(sm80_streamk.is_disjoint(&sm80));
        assert!(sm80_streamk.is_disjoint(&scalar));
        // The fragment instantiates its kernels through one macro per dtype.
        let fragment = super::SM80_STREAMK_SOURCE_FRAGMENT.source;
        assert!(fragment.contains("void tn_tc64_streamk_##SUFFIX("));
        for suffix in ["bf16", "f16"] {
            assert!(
                fragment.contains(&format!("GEMM_BI_DEFINE_GEMM_BI_TN_TC64_STREAMK({suffix},")),
                "the stream-K fragment does not instantiate the {suffix} kernel"
            );
        }
        for arch in [
            "sm_80", "sm_86", "sm_87", "sm_89", "sm_90a", "sm_100a", "sm_103a", "sm_107a",
            "sm_110a",
        ] {
            assert!(super::sm80_target_composes_streamk(arch), "{arch}");
        }
        for arch in [
            "sm_120",
            "compute_120",
            "sm_121",
            "compute_121",
            "sm_75",
            "",
        ] {
            assert!(!super::sm80_target_composes_streamk(arch), "{arch}");
        }
        assert_eq!(SM90A_SYMBOLS.len(), 12);
        let sm90a: BTreeSet<_> = SM90A_SYMBOLS.iter().copied().collect();
        let sm100: BTreeSet<_> = super::super::contract::SM100_KERNEL_SPECS
            .iter()
            .map(|spec| spec.symbol)
            .collect();
        assert_eq!(sm90a.len(), 12);
        assert_eq!(sm100.len(), 72);
        assert!(scalar.is_disjoint(&sm80));
        assert!(scalar.is_disjoint(&sm90a));
        assert!(scalar.is_disjoint(&sm100));
        assert!(sm80.is_disjoint(&sm90a));
        assert!(sm80.is_disjoint(&sm100));
        assert!(sm90a.is_disjoint(&sm100));
        assert_eq!(scalar.union(&sm80).count(), 72);
        assert!(
            scalar
                .union(&sm80)
                .copied()
                .all(|name| !carries_family_prefix(name))
        );
    }

    /// A kernel is named by what it computes; the family it belongs to is
    /// the module that composes it, never part of the symbol.
    fn carries_family_prefix(name: &str) -> bool {
        name.starts_with("gemm_bi_")
            || name.starts_with("fixed_")
            || name.starts_with("inference_")
            || name.starts_with("triad_")
    }

    #[test]
    fn triad_cuda_sources_and_export_inventories_carry_no_family_prefix() {
        let legacy_prefix = ["s", "gemm_bi_"].concat();
        let module_kinds = [
            ModuleKind::TriadScalar,
            ModuleKind::TriadSm80,
            ModuleKind::TriadSm90a,
            ModuleKind::TriadSm100,
            ModuleKind::TriadSm120,
        ];
        for module_kind in module_kinds {
            let source = compose_module_source(module_kind).expect("compose Triad CUDA source");
            // Prose may name a test file or an old symbol, and the #line boundaries
            // carry the family folder path; only code identifiers count.
            let code = source
                .lines()
                .filter(|line| !line.trim_start().starts_with("#line "))
                .map(|line| line.split("//").next().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n");
            let identifiers = code
                .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
                .collect::<Vec<_>>();
            assert!(
                !identifiers
                    .iter()
                    .any(|identifier| identifier.starts_with(&legacy_prefix)),
                "{module_kind:?} source retains the legacy S-prefixed ABI"
            );
            assert!(
                !identifiers
                    .iter()
                    .any(|identifier| identifier.contains("_SGEMM_BI_")),
                "{module_kind:?} source retains a legacy SGEMM macro identifier"
            );
            let offending = identifiers
                .iter()
                .find(|identifier| carries_family_prefix(identifier));
            assert!(
                offending.is_none(),
                "{module_kind:?} source names a kernel after its family: {offending:?}"
            );
        }

        let inventories = [
            ("scalar", PRODUCTION_SCALAR_SYMBOLS.to_vec()),
            ("sm80", PRODUCTION_SM80_SYMBOLS.to_vec()),
            ("sm90a", SM90A_SYMBOLS.to_vec()),
            (
                "sm100",
                super::super::contract::SM100_KERNEL_SPECS
                    .iter()
                    .map(|spec| spec.symbol)
                    .collect(),
            ),
            (
                "sm120",
                super::super::contract::sm120_kernel_specs()
                    .map(|spec| spec.symbol)
                    .collect(),
            ),
        ];
        for (inventory, symbols) in inventories {
            assert!(
                symbols.iter().all(|symbol| !carries_family_prefix(symbol)),
                "{inventory} export inventory names a kernel after its family"
            );
        }

        for module_kind in [
            ModuleKind::TriadSm80,
            ModuleKind::TriadSm90a,
            ModuleKind::TriadSm100,
            ModuleKind::TriadSm120,
        ] {
            assert!(
                super::super::contract::tf32_module_symbols(module_kind)
                    .all(|symbol| !carries_family_prefix(symbol)),
                "{module_kind:?} TF32 export inventory names a kernel after its family"
            );
        }
    }

    #[test]
    fn sm90a_module_requires_the_exact_architecture_target() {
        validate_module_target(ModuleKind::TriadSm90a, "sm_90a").unwrap();
        for target in ["sm_80", "sm_89", "sm_90", "sm_100a", "sm_120"] {
            let error = validate_module_target(ModuleKind::TriadSm90a, target).unwrap_err();
            assert!(error.contains("exact target sm_90a"), "{error}");
        }
        validate_module_target(ModuleKind::TriadSm80, "sm_90a").unwrap();
    }

    #[test]
    fn portable_sm103a_target_is_admitted_and_owns_live_cc103_topology() {
        validate_module_target(ModuleKind::TriadSm80, "sm_103a").unwrap();
        assert_eq!(portable_target_for_device((10, 3)), Ok("sm_103a"));
        assert_eq!(
            qualified_ptx_target(ModuleKind::TriadSm80, "sm_103a", (10, 3)),
            Ok("sm_103a")
        );
        assert!(qualified_ptx_target(ModuleKind::TriadSm80, "sm_103", (10, 3)).is_err());
    }

    #[test]
    fn portable_sm101a_target_is_admitted_and_owns_live_cc101_topology() {
        validate_module_target(ModuleKind::TriadSm80, "sm_101a").unwrap();
        assert_eq!(portable_target_for_device((10, 1)), Ok("sm_101a"));
        assert_eq!(
            qualified_ptx_target(ModuleKind::TriadSm80, "sm_101a", (10, 1)),
            Ok("sm_101a")
        );
        assert!(qualified_ptx_target(ModuleKind::TriadSm80, "sm_101", (10, 1)).is_err());
    }

    #[test]
    fn sm100_module_accepts_only_exact_feature_targets() {
        for target in [
            "compute_100f",
            "compute_100a",
            "compute_103f",
            "compute_103a",
        ] {
            validate_module_target(ModuleKind::TriadSm100, target).unwrap();
        }
        for target in [
            "compute_100",
            "compute_103",
            "sm_100a",
            "sm_103a",
            "compute_120f",
        ] {
            assert!(validate_module_target(ModuleKind::TriadSm100, target).is_err());
        }
    }

    #[test]
    fn sm100_probe_is_small_feature_complete_and_validated_fail_closed() {
        assert!(SM100_PROBE_SOURCE.contains("tcgen05_probe"));
        assert!(SM100_PROBE_SOURCE.contains("cp.async.bulk.tensor.2d"));
        assert!(SM100_PROBE_SOURCE.contains("tcgen05.mma.cta_group::1.kind::f16"));
        assert!(!SM100_PROBE_SOURCE.contains("nn_sm100"));

        let valid = sm100_probe_fixture(&sm100_probe_instructions().join("\n"));
        validate_sm100_probe_ptx("compute_100f", &valid).unwrap();

        assert!(validate_sm100_probe_ptx("compute_100a", &valid).is_err());
        assert!(
            validate_sm100_probe_ptx("compute_100f", &valid.replace("tcgen05_probe", "wrong"))
                .is_err()
        );
        let duplicate = format!("{valid}\n.entry tcgen05_probe(\n) {{}}\n");
        assert!(validate_sm100_probe_ptx("compute_100f", &duplicate).is_err());
        let foreign = format!("{valid}\n.entry harmless_foreign_export(\n) {{}}\n");
        let error = validate_sm100_probe_ptx("compute_100f", &foreign)
            .expect_err("SM100 probe foreign export must fail");
        assert!(error.contains("harmless_foreign_export"), "{error}");
        assert!(
            validate_sm100_probe_ptx(
                "compute_100f",
                &valid.replace("tcgen05.wait::st.sync.aligned", "")
            )
            .is_err()
        );
        assert!(
            validate_sm100_probe_ptx("compute_100f", &format!("{valid}\natom.global.add.f32"))
                .is_err()
        );
    }

    #[test]
    fn sm100_probe_requires_opcodes_inside_its_comment_stripped_body() {
        let empty = sm100_probe_fixture("");
        validate_sm100_probe_ptx("compute_100f", &empty).expect_err("empty probe body must fail");

        let line_comments = sm100_probe_instructions()
            .iter()
            .map(|instruction| format!("// {instruction}"))
            .collect::<Vec<_>>()
            .join("\n");
        validate_sm100_probe_ptx("compute_100f", &sm100_probe_fixture(&line_comments))
            .expect_err("opcodes in line comments must not qualify the probe");

        let block_comments = sm100_probe_instructions()
            .iter()
            .map(|instruction| format!("/* {instruction} */"))
            .collect::<Vec<_>>()
            .join("\n");
        validate_sm100_probe_ptx("compute_100f", &sm100_probe_fixture(&block_comments))
            .expect_err("opcodes in block comments must not qualify the probe");

        let quoted = sm100_probe_instructions()
            .iter()
            .map(|instruction| format!(".pragma \"{instruction}\";"))
            .collect::<Vec<_>>()
            .join("\n");
        validate_sm100_probe_ptx("compute_100f", &sm100_probe_fixture(&quoted))
            .expect_err("opcode spellings in string operands must not qualify the probe");

        let unused = format!(
            "{}\n.visible .func unused_feature_holder()\n{{\n{}\n}}\n",
            empty,
            sm100_probe_instructions().join("\n")
        );
        validate_sm100_probe_ptx("compute_100f", &unused)
            .expect_err("opcodes in an unused function must not qualify the probe");
    }

    #[test]
    fn sm100_probe_rejects_malformed_and_extern_entry_directives() {
        let valid = sm100_probe_fixture(&sm100_probe_instructions().join("\n"));
        let malformed = valid
            .strip_suffix("}\n")
            .expect("coherent probe fixture suffix");
        validate_sm100_probe_ptx("compute_100f", malformed)
            .expect_err("unclosed probe body must fail");

        let external = valid.replacen(
            ".visible .entry tcgen05_probe",
            ".visible .extern .entry tcgen05_probe",
            1,
        );
        validate_sm100_probe_ptx("compute_100f", &external)
            .expect_err("extern probe declaration must fail");
    }

    #[test]
    fn sm100_probe_compiles_for_every_exact_feature_target() {
        for (requested, emitted) in [
            ("compute_100f", "sm_100f"),
            ("compute_100a", "sm_100a"),
            ("compute_103f", "sm_103f"),
            ("compute_103a", "sm_103a"),
            ("compute_107f", "sm_107f"),
            ("compute_107a", "sm_107a"),
            ("compute_110f", "sm_110f"),
            ("compute_110a", "sm_110a"),
        ] {
            // Family-specific targets and CC 10.3 need CUDA 12.9; CC 11.0
            // needs CUDA 13.2 and CC 10.7 CUDA 13.4. An older toolkit cannot
            // name them at all.
            let nvrtc = super::nvrtc_version();
            if (requested.contains("110") && nvrtc < (13, 2))
                || (requested.contains("107") && nvrtc < (13, 4))
                || ((requested.ends_with('f') || requested.contains("103")) && nvrtc < (12, 9))
            {
                continue;
            }
            let options = cudarc::nvrtc::CompileOptions {
                arch: Some(requested),
                options: vec!["--fmad=true".to_string(), "-DNDEBUG".to_string()],
                ..Default::default()
            };
            let image = cudarc::nvrtc::compile_ptx_with_opts(SM100_PROBE_SOURCE, options)
                .unwrap_or_else(|error| panic!("SM100 probe failed for {requested}: {error}"));
            let ptx = crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(
                image.as_bytes().expect("SM100 probe PTX image"),
            )
            .expect("SM100 probe PTX must be canonical UTF-8");
            assert!(
                ptx.lines()
                    .any(|line| line.trim() == format!(".target {emitted}"))
            );
            validate_sm100_probe_ptx(requested, &ptx).unwrap();
        }
    }

    #[test]
    fn sm100_candidate_fallback_survives_family_probe_failure() {
        let probes = std::cell::RefCell::new(Vec::new());
        let compiles = std::cell::RefCell::new(Vec::new());
        let qualified = std::cell::RefCell::new(Vec::new());
        let selected = select_sm100_candidate(
            super::super::dispatch::sm100_target_candidates((10, 0)),
            |candidate| {
                probes.borrow_mut().push(candidate.nvrtc_arch);
                if candidate.kind == super::super::contract::Sm100TargetKind::Family {
                    Err("injected family probe failure".into())
                } else {
                    Ok(())
                }
            },
            |candidate| {
                compiles.borrow_mut().push(candidate.nvrtc_arch);
                Ok(candidate)
            },
            |candidate| {
                qualified.borrow_mut().push(candidate.nvrtc_arch);
                Ok(candidate)
            },
        )
        .expect("exact candidate fallback");
        assert_eq!(selected.nvrtc_arch, "compute_100a");
        assert_eq!(*probes.borrow(), ["compute_100f", "compute_100a"]);
        assert_eq!(*compiles.borrow(), ["compute_100a"]);
        assert_eq!(*qualified.borrow(), ["compute_100a"]);
    }

    #[test]
    fn sm100_candidate_fallback_survives_family_compile_failure() {
        let compiles = std::cell::RefCell::new(Vec::new());
        let qualified = std::cell::RefCell::new(Vec::new());
        let selected = select_sm100_candidate(
            super::super::dispatch::sm100_target_candidates((10, 0)),
            |_| Ok(()),
            |candidate| {
                compiles.borrow_mut().push(candidate.nvrtc_arch);
                if candidate.kind == super::super::contract::Sm100TargetKind::Family {
                    Err("injected family compile failure".into())
                } else {
                    Ok(candidate)
                }
            },
            |candidate| {
                qualified.borrow_mut().push(candidate.nvrtc_arch);
                Ok(candidate)
            },
        )
        .expect("exact candidate fallback");
        assert_eq!(selected.nvrtc_arch, "compute_100a");
        assert_eq!(*compiles.borrow(), ["compute_100f", "compute_100a"]);
        assert_eq!(*qualified.borrow(), ["compute_100a"]);
    }

    #[test]
    fn sm100_candidate_fallback_survives_family_qualification_failure() {
        let qualified = std::cell::RefCell::new(Vec::new());
        let selected = select_sm100_candidate(
            super::super::dispatch::sm100_target_candidates((10, 0)),
            |_| Ok(()),
            Ok,
            |candidate| {
                qualified.borrow_mut().push(candidate.nvrtc_arch);
                if candidate.kind == super::super::contract::Sm100TargetKind::Family {
                    Err("injected family qualification failure".into())
                } else {
                    Ok(candidate)
                }
            },
        )
        .expect("exact candidate fallback");
        assert_eq!(selected.nvrtc_arch, "compute_100a");
        assert_eq!(*qualified.borrow(), ["compute_100f", "compute_100a"]);
    }

    #[test]
    fn sm120_cc121_resolution_rejects_a_mixed_target_artifact_transaction() {
        let candidates = super::super::dispatch::sm120_target_candidates((12, 1), (13, 2));
        let attempts = std::cell::RefCell::new(Vec::new());
        let selected = select_sm120_candidate(candidates, |requested| {
            attempts.borrow_mut().push(requested.nvrtc_arch);
            if requested.nvrtc_arch == "compute_121" {
                let wrong = super::super::dispatch::sm120_target_candidates((12, 1), (12, 8))[0];
                Ok((wrong, "mixed artifact set"))
            } else {
                Ok((requested, "same-target artifact set"))
            }
        })
        .expect("compute_120 same-target transaction");

        assert_eq!(selected.0.nvrtc_arch, "compute_120");
        assert_eq!(selected.1, "same-target artifact set");
        assert_eq!(*attempts.borrow(), ["compute_121", "compute_120"]);
    }

    #[test]
    fn sm120_module_accepts_only_generic_targets() {
        for target in ["compute_120", "compute_121"] {
            validate_module_target(ModuleKind::TriadSm120, target).unwrap();
        }
        for target in [
            "sm_120",
            "sm_121",
            "sm_120a",
            "sm_120f",
            "sm_121a",
            "sm_121f",
            "compute_120a",
            "compute_120f",
            "compute_121a",
            "compute_121f",
        ] {
            assert!(
                validate_module_target(ModuleKind::TriadSm120, target).is_err(),
                "TriadSm120 accepted {target}"
            );
        }
    }

    #[test]
    fn sm120_ptx_validation_is_complete_and_fail_closed() {
        let valid = whole_module_fixture(ModuleKind::TriadSm120, "sm_121");

        validate_sm120_ptx("compute_121", &valid).unwrap();
        assert!(validate_sm120_ptx("compute_120", &valid).is_err());
        let first = super::super::contract::SM120_KERNEL_SPECS[0].symbol;
        assert!(
            validate_sm120_ptx(
                "compute_121",
                &valid.replacen(&format!(".entry {first}("), ".entry missing(", 1),
            )
            .is_err()
        );
        for forbidden in [
            "atom.global.add.f32",
            "red.global.add.f32",
            "tcgen05.mma.cta_group::1.kind::f16",
            "tcgen05.mma.cta_group::1.kind::tf32",
            "wgmma.mma_async.sync.aligned",
            "setmaxnreg.inc.sync.aligned.u32",
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.multicast",
            "call.uni (_), cudaLaunchDevice, ();",
            ".extern .func malloc;",
        ] {
            assert!(
                validate_sm120_ptx("compute_121", &format!("{valid}\n{forbidden}")).is_err(),
                "validator accepted {forbidden}"
            );
        }
    }

    #[test]
    fn sm100_production_ptx_validator_rejects_partial_or_mixed_artifacts() {
        let valid = whole_module_fixture(ModuleKind::TriadSm100, "sm_100f");
        validate_sm100_ptx("compute_100f", &valid).unwrap();

        assert!(validate_sm100_ptx("compute_100a", &valid).is_err());
        let first = super::super::contract::SM100_KERNEL_SPECS[0].symbol;
        assert!(
            validate_sm100_ptx(
                "compute_100f",
                &valid.replacen(&format!(".entry {first}("), ".entry missing(", 1),
            )
            .is_err()
        );
        assert!(
            validate_sm100_ptx(
                "compute_100f",
                &format!("{valid}\nwgmma.fence.sync.aligned")
            )
            .is_err()
        );
        for forbidden in [
            "call.uni (_), cudaLaunchDevice, ();",
            ".callprototype ()_();",
            ".extern .func malloc;",
            ".extern .func free;",
            ".extern .func cudaGetParameterBuffer;",
        ] {
            assert!(
                validate_sm100_ptx("compute_100f", &format!("{valid}\n{forbidden}")).is_err(),
                "validator accepted {forbidden}"
            );
        }
    }

    #[test]
    fn forbidden_scan_distinguishes_harmless_strings_from_real_tokens() {
        for (module_kind, target, validate) in [
            (
                ModuleKind::TriadSm100,
                "sm_100f",
                validate_sm100_ptx as fn(&str, &str) -> Result<(), String>,
            ),
            (
                ModuleKind::TriadSm120,
                "sm_120",
                validate_sm120_ptx as fn(&str, &str) -> Result<(), String>,
            ),
        ] {
            let arch = match module_kind {
                ModuleKind::TriadSm100 => "compute_100f",
                ModuleKind::TriadSm120 => "compute_120",
                _ => unreachable!(),
            };
            let valid = whole_module_fixture(module_kind, target);
            let harmless = format!(
                "{valid}\n.file 7 \"free cudaLaunchDevice multicast shared::cluster\"\n.pragma \"operator new call.uni atom.global.add.f32 tcgen05.ld.red\";\n"
            );
            validate(arch, &harmless).unwrap_or_else(|error| {
                panic!("{module_kind:?} rejected harmless strings: {error}")
            });

            for forbidden in [
                "atom.global.add.f32 %f1, [%rd1], %f2;",
                ".extern .func free();",
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.multicast::cluster;",
            ] {
                validate(arch, &format!("{valid}\n{forbidden}\n"))
                    .expect_err("real forbidden opcode or runtime symbol must fail");
            }
        }
    }

    #[test]
    fn tf32_forbidden_scan_distinguishes_strings_from_real_opcodes() {
        for (module_kind, required, forbidden) in [
            (
                ModuleKind::TriadSm90a,
                [
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                    "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32",
                ],
                "cvt.rna.tf32.f32",
            ),
            (
                ModuleKind::TriadSm100,
                [
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                    "tcgen05.mma.cta_group::1.kind::tf32",
                ],
                "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32",
            ),
            (
                ModuleKind::TriadSm120,
                [
                    "cvt.rna.tf32.f32",
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                ],
                "tcgen05.mma.cta_group::1.kind::tf32",
            ),
        ] {
            let mut ptx = ".version 9.0\n.target sm_90a\n".to_string();
            for symbol in super::super::contract::tf32_module_symbols(module_kind) {
                let body: &[&str] = if symbol.contains("_tma_fma_") {
                    &[
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                        "fma.rn.f32",
                    ]
                } else {
                    &required
                };
                ptx.push_str(&format!(".entry {symbol}() {{\n{}\n}}\n", body.join("\n")));
            }
            validate_tf32_feature_instructions(
                module_kind,
                false,
                &format!("{ptx}\n.file 9 \"{forbidden}\"\n.pragma \"{forbidden}\";\n"),
            )
            .unwrap_or_else(|error| panic!("{module_kind:?} rejected harmless strings: {error}"));
            validate_tf32_feature_instructions(
                module_kind,
                false,
                &format!("{ptx}\n{forbidden};\n"),
            )
            .expect_err("real foreign TF32 opcode must fail");
        }
    }

    // These fixtures exercise the production composer and PTX admission
    // boundary. They are parser fixtures, not executable CUDA programs.
    const FIXED_SM89_HALF_TEST_SYMBOLS: [&str; 2] =
        ["nn_sm89_tc128_pipeline_bf16", "nn_sm89_tc128_pipeline_f16"];
    const FIXED_SM89_HALF_SWIZZLE_TEST_SYMBOLS: [&str; 2] =
        ["nn_sm89_tc128_swizzle_bf16", "nn_sm89_tc128_swizzle_f16"];
    const FIXED_SM89_HALF_S3_TEST_SYMBOLS: [&str; 2] =
        ["nn_sm89_tc128_s3_bf16", "nn_sm89_tc128_s3_f16"];
    const FIXED_SM89_RNA_N96_TEST_SYMBOL: &str = "nn_sm89_rna_tf32_m128n96_bk32_s3";
    const FIXED_SM89_HALF_M64N64_S3_TEST_SYMBOL: &str = "nn_sm89_m64n64_bk64_s3_f16";
    const FIXED_SM89_HALF_M128N64_S2_TEST_SYMBOL: &str = "nn_sm89_m128n64_bk64_s2_f16";

    fn fixed_sm89_half_test_base_ptx() -> String {
        let mut ptx = ".version 8.7\n.target sm_89\n.address_size 64\n".to_string();
        for symbol in [
            "nn_tf32_m128n64_bk32_s2",
            "nn_tf32_m128n64_bk32_s3",
            "nn_tf32_m64n64_bk32_s2",
            "nn_tf32_m64n64_bk32_s3",
            "nn_tf32_m16n32_bk32_s4",
        ] {
            ptx.push_str(&format!(
                ".visible .entry {symbol}(\n\
                 .param .u64 c,\n.param .u64 a,\n.param .u64 b,\n.param .u64 bias,\n\
                 .param .align 4 .b8 params[24]\n) {{\n\
                 cvt.rna.tf32.f32 %r0, %f0;\n\
                 mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n\
                 {{%f0,%f1,%f2,%f3}}, {{%r0,%r1,%r2,%r3}}, {{%r4,%r5}}, {{%f0,%f1,%f2,%f3}};\n\
                 ret;\n}}\n"
            ));
        }
        ptx
    }

    fn fixed_sm89_half_test_entry(symbol: &str, dtype: &str) -> String {
        format!(
            ".visible .entry {symbol}(\n\
             .param .u64 c,\n.param .u64 a,\n.param .u64 b,\n.param .u64 bias,\n\
             .param .align 4 .b8 params[32]\n) {{\n\
             cp.async.cg.shared.global [%r0], [%rd0], 16, %r1;\n\
             cp.async.commit_group;\ncp.async.wait_group 0;\nbar.sync 0;\n\
             ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r0,%r1,%r2,%r3}}, [%r4];\n\
             ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{%r4,%r5}}, [%r6];\n\
             mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32\n\
             {{%f0,%f1,%f2,%f3}}, {{%r0,%r1,%r2,%r3}}, {{%r4,%r5}}, {{%f0,%f1,%f2,%f3}};\n\
             mul.rn.f32 %f0, %f0, %f4;\nfma.rn.f32 %f0, %f1, %f2, %f0;\n\
             cvt.rn.{dtype}.f32 %rs0, %f0;\n\
             st.global.v4.u32 [%rd0], {{%r0,%r1,%r2,%r3}};\nret;\n}}\n"
        )
    }

    fn fixed_sm89_half_s3_test_entry(symbol: &str, dtype: &str) -> String {
        fixed_sm89_half_test_entry(symbol, dtype).replace(
            "cp.async.wait_group 0;",
            "cp.async.wait_group 1;\ncp.async.wait_group 0;",
        )
    }

    fn fixed_sm89_finalist_half_test_entry(symbol: &str, s3: bool) -> String {
        let entry = if s3 {
            fixed_sm89_half_s3_test_entry(symbol, "f16")
        } else {
            fixed_sm89_half_test_entry(symbol, "f16")
        };
        entry.replacen(") {\n", ")\n.maxntid 128, 1, 1\n.minnctapersm 2\n{\n", 1)
    }

    fn fixed_sm89_cell_test_entry(
        spec: &super::super::super::gemm_bi_inference::sm89_cells::Sm89CellSpec,
    ) -> String {
        use super::super::super::gemm_bi_inference::sm89_cells::Sm89CellFamily;
        let body = match spec.family {
            Sm89CellFamily::ExactFma => {
                "cp.async.cg.shared.global [%r0], [%rd0], 16;\nfma.rn.f32 %f0, %f1, %f2, %f3;".to_string()
            }
            Sm89CellFamily::HalfMma => format!(
                "cp.async.cg.shared.global [%r0], [%rd0], 16;\n\
                 ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r0,%r1,%r2,%r3}}, [%r4];\n\
                 ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{%r4,%r5}}, [%r6];\n\
                 mma.sync.aligned.m16n8k16.row.col.f32.{dt}.{dt}.f32 {{%f0,%f1,%f2,%f3}}, {{%r0,%r1,%r2,%r3}}, {{%r4,%r5}}, {{%f0,%f1,%f2,%f3}};",
                dt = if spec.input == crate::mamba_ssm::gpu::dtype::WeightDtype::Bf16 { "bf16" } else { "f16" }
            ),
            Sm89CellFamily::Tf32Mma => "cp.async.cg.shared.global [%r0], [%rd0], 16;\n\
                 cvt.rna.tf32.f32 %r0, %f0;\n\
                 mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};"
                .to_string(),
        };
        format!(
            ".visible .entry {}(\n.param .u64 c,\n.param .u64 a,\n.param .u64 b,\n.param .u64 bias,\n.param .align 4 .b8 params[32]\n)\n{{\n.reg .b32 %r<8>;\n.reg .b64 %rd<2>;\n.reg .f32 %f<8>;\n{body}\nret;\n}}\n",
            spec.symbol
        )
    }

    fn fixed_sm89_cells_test_entries() -> String {
        super::super::super::gemm_bi_inference::sm89_cells::SM89_CELL_SPECS
            .iter()
            .map(fixed_sm89_cell_test_entry)
            .collect()
    }

    fn inference_sm89_cells_test_ptx() -> String {
        fixed_sm89_half_test_base_ptx() + &fixed_sm89_cells_test_entries()
    }

    #[test]
    fn inference_sm89_cells_ptx_inventory_is_pinned_to_its_own_module() {
        use ModuleKind::{Fixed, InferenceSm89Cells};
        let cells = inference_sm89_cells_test_ptx();
        let entries = fixed_sm89_cells_test_entries();
        let without = cells.replacen(&entries, "", 1);
        super::validate_fixed_sm89_cells_ptx(InferenceSm89Cells, "sm_89", &cells).unwrap();
        super::validate_fixed_sm89_cells_ptx(InferenceSm89Cells, "sm_89", &without)
            .expect_err("every cell is mandatory where the overlay is composed");
        super::validate_fixed_sm89_cells_ptx(Fixed, "sm_89", &without).unwrap();
        super::validate_fixed_sm89_cells_ptx(Fixed, "sm_89", &cells)
            .expect_err("the Fixed module carries no cell");
        for spec in super::super::super::gemm_bi_inference::sm89_cells::SM89_CELL_SPECS.iter() {
            let entry = fixed_sm89_cell_test_entry(spec);
            super::validate_fixed_sm89_cells_ptx(
                InferenceSm89Cells,
                "sm_89",
                &cells.replacen(&entry, "", 1),
            )
            .expect_err("a missing cell must reject");
            super::validate_fixed_sm89_cells_ptx(
                InferenceSm89Cells,
                "sm_89",
                &format!("{cells}{entry}"),
            )
            .expect_err("a duplicated cell must reject");
            let narrow = entry.replacen("params[32]", "params[24]", 1);
            super::validate_fixed_sm89_cells_ptx(
                InferenceSm89Cells,
                "sm_89",
                &cells.replacen(&entry, &narrow, 1),
            )
            .expect_err("the 32-byte bundle ABI is pinned");
            let unrolled = entry.replacen("cp.async.cg.shared.global", "not.the.copy", 1);
            super::validate_fixed_sm89_cells_ptx(
                InferenceSm89Cells,
                "sm_89",
                &cells.replacen(&entry, &unrolled, 1),
            )
            .expect_err("the staging copy is pinned");
            let spilled = entry.replacen("ret;", "ld.local.u32 %r0, [%rd0]; ret;", 1);
            super::validate_fixed_sm89_cells_ptx(
                InferenceSm89Cells,
                "sm_89",
                &cells.replacen(&entry, &spilled, 1),
            )
            .expect_err("local memory rejects");
        }
        for target in ["sm_80", "sm_90a"] {
            let retargeted = cells.replace(".target sm_89", &format!(".target {target}"));
            super::validate_fixed_sm89_cells_ptx(InferenceSm89Cells, target, &retargeted).unwrap();
            super::validate_fixed_sm89_cells_ptx(
                InferenceSm89Cells,
                target,
                &retargeted.replacen(&entries, "", 1),
            )
            .expect_err("the cells are mandatory on every portable target");
        }
        for target in ["compute_89", "sm_120", "compute_120"] {
            let retargeted = cells.replace(".target sm_89", &format!(".target {target}"));
            super::validate_fixed_sm89_cells_ptx(
                InferenceSm89Cells,
                target,
                &retargeted.replacen(&entries, "", 1),
            )
            .unwrap();
            super::validate_fixed_sm89_cells_ptx(InferenceSm89Cells, target, &retargeted)
                .expect_err("the cells are foreign where the overlay is not composed");
        }
    }

    fn fixed_sm89_half_test_ptx() -> String {
        let mut ptx = fixed_sm89_half_test_base_ptx();
        ptx.push_str(&fixed_sm89_half_test_entry(
            FIXED_SM89_HALF_TEST_SYMBOLS[0],
            "bf16",
        ));
        ptx.push_str(&fixed_sm89_half_test_entry(
            FIXED_SM89_HALF_TEST_SYMBOLS[1],
            "f16",
        ));
        ptx.push_str(&fixed_sm89_exact_n64_test_entry(
            FIXED_SM89_EXACT_N64_TEST_SYMBOL,
        ));
        ptx.push_str(&fixed_sm89_rna_wide_test_entry());
        ptx.push_str(&fixed_sm89_half_test_entry(
            FIXED_SM89_HALF_SWIZZLE_TEST_SYMBOLS[0],
            "bf16",
        ));
        ptx.push_str(&fixed_sm89_half_test_entry(
            FIXED_SM89_HALF_SWIZZLE_TEST_SYMBOLS[1],
            "f16",
        ));
        for (symbol, dtype) in FIXED_SM89_HALF_S3_TEST_SYMBOLS
            .into_iter()
            .zip(["bf16", "f16"])
        {
            ptx.push_str(&fixed_sm89_half_s3_test_entry(symbol, dtype));
        }
        ptx.push_str(&fixed_sm89_rna_wide_test_entry().replace(
            FIXED_SM89_RNA_WIDE_TEST_SYMBOL,
            FIXED_SM89_RNA_N96_TEST_SYMBOL,
        ));
        ptx.push_str(&fixed_sm89_finalist_half_test_entry(
            FIXED_SM89_HALF_M64N64_S3_TEST_SYMBOL,
            true,
        ));
        ptx.push_str(&fixed_sm89_finalist_half_test_entry(
            FIXED_SM89_HALF_M128N64_S2_TEST_SYMBOL,
            false,
        ));
        ptx
    }

    #[test]
    fn fixed_sm89_half_pipeline_composes_the_ada_only_extension() {
        let base = compose_fragments(super::FIXED_SOURCE_FRAGMENTS).unwrap();
        let ada = compose_module_source_for(ModuleKind::Fixed, "sm_89").unwrap();
        let extension = ada
            .strip_prefix(&base)
            .expect("Ada must preserve the exact existing Fixed source prefix");
        let boundaries: Vec<_> = extension
            .lines()
            .filter_map(|line| line.strip_prefix("#line 1 \"")?.strip_suffix('"'))
            .collect();
        assert_eq!(
            boundaries,
            [
                "kernels/gemm_bi_inference/sm80/half_pipeline.cu",
                "kernels/gemm_bi_inference/sm80/f32_n64_copyplan.cu",
                "kernels/gemm_bi_inference/sm80/tf32_rna_wide.cu",
                "kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh",
                "kernels/gemm_bi_inference/sm80/half_swizzle.cu",
                "kernels/gemm_bi_inference/sm80/half_s3.cu",
                "kernels/gemm_bi_inference/sm80/tf32_rna_n96.cu",
                "kernels/gemm_bi_inference/sm80/half_n64.cu",
            ],
            "Ada must retain the half extension before the exact N64 extension"
        );
        let prior_suffix = compose_fragments(&[
            super::FIXED_SM89_HALF_SOURCE_FRAGMENT,
            super::FIXED_SM89_EXACT_N64_SOURCE_FRAGMENT,
            super::FIXED_SM89_RNA_WIDE_SOURCE_FRAGMENT,
            super::FIXED_SM89_HALF_SWIZZLE_LAYOUT_FRAGMENT,
            super::FIXED_SM89_HALF_SWIZZLE_SOURCE_FRAGMENT,
            super::FIXED_SM89_HALF_S3_SOURCE_FRAGMENT,
        ])
        .unwrap();
        assert!(
            extension.starts_with(&prior_suffix),
            "Ada composition must preserve every pre-finalist suffix byte"
        );
        for symbol in [
            "nn_sm89_rna_tf32_m128n96_bk32_s3",
            "nn_sm89_m64n64_bk64_s3_f16",
            "nn_sm89_m128n64_bk64_s2_f16",
        ] {
            assert!(ada.contains(symbol), "Fixed/sm_89 omitted {symbol}");
        }
    }

    #[test]
    fn fixed_sm89_half_pipeline_travels_with_the_portable_tier_and_skips_the_cc12_family() {
        let base = compose_fragments(super::FIXED_SOURCE_FRAGMENTS).unwrap();
        let ada = compose_module_source_for(ModuleKind::Fixed, "sm_89").unwrap();
        assert!(ada.len() > base.len() && ada.starts_with(&base));
        for target in [
            "sm_80", "sm_86", "sm_87", "sm_90", "sm_90a", "sm_100a", "sm_103a", "sm_107a",
            "sm_110a",
        ] {
            assert_eq!(
                compose_module_source_for(ModuleKind::Fixed, target)
                    .unwrap()
                    .as_bytes(),
                ada.as_bytes(),
                "Fixed composition differs from the Ada bytes on portable target {target}"
            );
        }
        for target in ["sm_120", "sm_121", "compute_121"] {
            assert_eq!(
                compose_module_source_for(ModuleKind::Fixed, target)
                    .unwrap()
                    .as_bytes(),
                base.as_bytes(),
                "Fixed composition changed on CC 12 target {target}"
            );
        }
    }

    #[test]
    fn fixed_sm89_half_pipeline_does_not_enter_any_triad_composition() {
        for target in ["sm_89", "sm_120", "compute_120", "sm_121", "compute_121"] {
            for kind in [
                ModuleKind::TriadScalar,
                ModuleKind::TriadSm80,
                ModuleKind::TriadSm120,
            ] {
                let source = compose_module_source_for(kind, target).unwrap();
                let boundaries: Vec<_> = source
                    .lines()
                    .filter_map(|line| line.strip_prefix("#line 1 \"")?.strip_suffix('"'))
                    .collect();
                assert!(
                    !boundaries.contains(&"kernels/gemm_bi_inference/sm80/half_pipeline.cu"),
                    "Fixed half extension leaked into {kind:?}/{target}"
                );
            }
        }
    }

    #[test]
    fn fixed_sm89_half_pipeline_ptx_requires_both_typed_exports() {
        validate_module_ptx(ModuleKind::Fixed, "sm_89", &fixed_sm89_half_test_ptx())
            .expect("both typed exports with the five-argument ABI must be accepted");
        validate_module_ptx(ModuleKind::Fixed, "sm_89", &fixed_sm89_half_test_base_ptx())
            .expect_err("an Ada Fixed module missing both pipeline exports must fail admission");
        for missing in 0..2 {
            let mut ptx = fixed_sm89_half_test_base_ptx();
            let retained = 1 - missing;
            ptx.push_str(&fixed_sm89_half_test_entry(
                FIXED_SM89_HALF_TEST_SYMBOLS[retained],
                if retained == 0 { "bf16" } else { "f16" },
            ));
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
                .expect_err("a partially compiled half holder must not be admitted");
        }
    }

    #[test]
    fn fixed_sm89_half_s3_ptx_inventory_is_required() {
        let old_inventory = fixed_sm89_half_test_ptx();
        let mut without_s3 = old_inventory;
        for (symbol, dtype) in FIXED_SM89_HALF_S3_TEST_SYMBOLS
            .into_iter()
            .zip(["bf16", "f16"])
        {
            without_s3 = without_s3.replace(&fixed_sm89_half_s3_test_entry(symbol, dtype), "");
        }
        validate_module_ptx(ModuleKind::Fixed, "sm_89", &without_s3)
            .expect_err("Ada Fixed must reject an artifact missing both S3 exports");
    }

    #[test]
    fn fixed_sm89_half_pipeline_ptx_rejects_duplicate_or_foreign_exports() {
        for extra in [
            FIXED_SM89_HALF_TEST_SYMBOLS[0],
            "nn_sm89_tc128_pipeline_f32",
            "nn_sm89_tc128_pipeline_vec_bf16",
        ] {
            let ptx = fixed_sm89_half_test_ptx() + &fixed_sm89_half_test_entry(extra, "bf16");
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
                .expect_err("the admitted half extension owns exactly two unique exports");
        }
    }

    #[test]
    fn fixed_sm89_half_pipeline_ptx_is_required_on_every_portable_tier_target() {
        for target in [
            "sm_80", "sm_86", "sm_87", "sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_110a",
        ] {
            let base = fixed_sm89_half_test_base_ptx()
                .replace(".target sm_89", &format!(".target {target}"));
            validate_module_ptx(ModuleKind::Fixed, target, &base).expect_err(
                "the portable overlay makes the Ada exports mandatory on every portable target",
            );
            let ptx =
                fixed_sm89_half_test_ptx().replace(".target sm_89", &format!(".target {target}"));
            validate_module_ptx(ModuleKind::Fixed, target, &ptx)
                .expect("the complete overlay inventory must be admitted on every portable target");
        }
    }

    #[test]
    fn fixed_sm89_half_pipeline_ptx_rejects_pointer_or_bundle_abi_drift() {
        let baseline = fixed_sm89_half_test_ptx();
        validate_module_ptx(ModuleKind::Fixed, "sm_89", &baseline).unwrap();
        let symbol = FIXED_SM89_HALF_TEST_SYMBOLS[0];
        let entry = fixed_sm89_half_test_entry(symbol, "bf16");
        for malformed in [
            entry.replacen(".param .u64 a,", ".param .u32 a,", 1),
            entry.replacen(".param .u64 bias,", ".param .u32 bias,", 1),
            entry.replacen("params[32]", "params[24]", 1),
            entry.replacen("params[32]", "params[28]", 1),
            entry.replacen("params[32]", "params[36]", 1),
            entry.replacen(".param .align 4 .b8 params", ".param .align 8 .b8 params", 1),
            entry.replacen("params[32]\n)", "params[32],\n.param .u64 scratch\n)", 1),
            entry.replacen(
                ".param .align 4 .b8 params[32]",
                ".param .f32 alpha,\n.param .f32 beta,\n.param .u32 m,\n.param .u32 n,\n.param .u32 k,\n.param .u32 lda,\n.param .u32 ldb,\n.param .u32 ldc",
                1,
            ),
        ] {
            let ptx = baseline.replacen(&entry, &malformed, 1);
            assert_ne!(ptx, baseline, "ABI fixture mutation must affect the typed entry");
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
                .expect_err("half pipeline requires four u64 pointers and one align-4 32-byte bundle");
        }
    }

    #[test]
    fn fixed_sm89_half_pipeline_ptx_requires_native_typed_mma_and_async_staging() {
        let baseline = fixed_sm89_half_test_ptx();
        validate_module_ptx(ModuleKind::Fixed, "sm_89", &baseline).unwrap();
        for (index, dtype) in [(0, "bf16"), (1, "f16")] {
            let entry = fixed_sm89_half_test_entry(FIXED_SM89_HALF_TEST_SYMBOLS[index], dtype);
            for instruction in [
                format!("mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32"),
                "cp.async.cg.shared.global".to_string(),
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16".to_string(),
            ] {
                let malformed = entry.replace(&instruction, "not_the_required_instruction");
                let ptx = baseline.replacen(&entry, &malformed, 1);
                validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
                    .expect_err("half dtype and asynchronous pipeline instructions are mandatory");
            }
        }
    }

    #[test]
    fn fixed_sm89_half_pipeline_ptx_rejects_numeric_atomics_and_reductions() {
        let baseline = fixed_sm89_half_test_ptx();
        for instruction in [
            "atom.global.add.f32 %f0, [%rd0], %f1;",
            "red.global.add.f32 [%rd0], %f1;",
            "redux.sync.add.s32 %r0, %r1, -1;",
        ] {
            let entry = fixed_sm89_half_test_entry(FIXED_SM89_HALF_TEST_SYMBOLS[0], "bf16");
            let malformed = entry.replacen("ret;", &format!("{instruction}\nret;"), 1);
            let ptx = baseline.replacen(&entry, &malformed, 1);
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
                .expect_err("the fixed half chain must not acquire numeric reductions");
        }
    }

    #[test]
    fn fixed_sm89_half_pipeline_driver_abi_requires_exact_offsets_sizes_and_terminal_probe() {
        use super::{query_driver_parameter_abi, validate_fixed_sm89_half_driver_abi};
        use cudarc::driver::sys::CUresult;
        let symbol = FIXED_SM89_HALF_TEST_SYMBOLS[0];
        let layout = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
        let mut queries = Vec::new();
        let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| {
            queries.push(index);
            if let Some((parameter_offset, parameter_size)) = layout.get(index) {
                *offset = *parameter_offset;
                *size = *parameter_size;
                CUresult::CUDA_SUCCESS
            } else {
                CUresult::CUDA_ERROR_INVALID_VALUE
            }
        })
        .expect("exact five-argument Driver ABI");
        assert_eq!(queries, [0, 1, 2, 3, 4, 5]);
        validate_fixed_sm89_half_driver_abi(symbol, &abi).unwrap();
        for layout in [
            vec![(0, 8), (8, 8), (16, 8), (24, 8)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 28)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (36, 32)],
            vec![(0, 4), (8, 8), (16, 8), (24, 8), (32, 32)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32), (64, 4)],
        ] {
            let malformed = Tf32DriverAbi::checked(layout.len(), layout).unwrap();
            validate_fixed_sm89_half_driver_abi(symbol, &malformed)
                .expect_err("valid generic layouts must still match the exact half ABI");
        }
        query_driver_parameter_abi(symbol, 5, |index, offset, size| {
            *offset = index * 8;
            *size = 8;
            CUresult::CUDA_SUCCESS
        })
        .expect_err("a sixth successful Driver parameter query must reject");
    }

    #[test]
    fn fixed_sm89_half_s3_driver_abi_requires_exact_offsets_sizes_and_terminal_probe() {
        use super::{query_driver_parameter_abi, validate_fixed_sm89_half_s3_driver_abi};
        use cudarc::driver::sys::CUresult;
        let symbol = FIXED_SM89_HALF_S3_TEST_SYMBOLS[0];
        let layout = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
        let mut queries = Vec::new();
        let abi = query_driver_parameter_abi(symbol, 5, |index, offset, size| {
            queries.push(index);
            if let Some((parameter_offset, parameter_size)) = layout.get(index) {
                *offset = *parameter_offset;
                *size = *parameter_size;
                CUresult::CUDA_SUCCESS
            } else {
                CUresult::CUDA_ERROR_INVALID_VALUE
            }
        })
        .expect("exact five-argument Driver ABI");
        assert_eq!(queries, [0, 1, 2, 3, 4, 5]);
        validate_fixed_sm89_half_s3_driver_abi(symbol, &abi).unwrap();
        for layout in [
            vec![(0, 8), (8, 8), (16, 8), (24, 8)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 28)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (36, 32)],
            vec![(0, 4), (8, 8), (16, 8), (24, 8), (32, 32)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32), (64, 4)],
        ] {
            let malformed = Tf32DriverAbi::checked(layout.len(), layout).unwrap();
            validate_fixed_sm89_half_s3_driver_abi(symbol, &malformed)
                .expect_err("valid generic layouts must still match the exact half ABI");
        }
        query_driver_parameter_abi(symbol, 5, |index, offset, size| {
            *offset = index * 8;
            *size = 8;
            CUresult::CUDA_SUCCESS
        })
        .expect_err("a sixth successful Driver parameter query must reject");
    }

    #[test]
    fn fixed_sm89_half_swizzle_driver_abi_and_resources_are_strict() {
        let symbol = FIXED_SM89_HALF_SWIZZLE_TEST_SYMBOLS[0];
        let abi =
            Tf32DriverAbi::checked(5, vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]).unwrap();
        super::validate_fixed_sm89_half_swizzle_driver_abi(symbol, &abi).unwrap();
        let valid = super::FixedSm89HalfSwizzleResources {
            local_bytes: 0,
            registers: super::FIXED_SM89_HALF_SWIZZLE_REGISTER_CAP,
            static_shared_bytes: 0,
            max_threads: 256,
            active_blocks: 1,
        };
        super::validate_fixed_sm89_half_swizzle_resources(symbol, valid).unwrap();
        super::validate_fixed_sm89_half_swizzle_shared_capacity(69_632).unwrap();
        super::validate_fixed_sm89_half_swizzle_shared_capacity(69_631)
            .expect_err("one byte below the opt-in shared requirement must reject");
        for malformed in [
            super::FixedSm89HalfSwizzleResources {
                local_bytes: 1,
                ..valid
            },
            super::FixedSm89HalfSwizzleResources {
                registers: super::FIXED_SM89_HALF_SWIZZLE_REGISTER_CAP + 1,
                ..valid
            },
            super::FixedSm89HalfSwizzleResources {
                static_shared_bytes: 1,
                ..valid
            },
            super::FixedSm89HalfSwizzleResources {
                max_threads: 255,
                ..valid
            },
            super::FixedSm89HalfSwizzleResources {
                active_blocks: 0,
                ..valid
            },
        ] {
            super::validate_fixed_sm89_half_swizzle_resources(symbol, malformed)
                .expect_err("each physical resource gate must reject independently");
        }
        for malformed in [
            vec![(0, 8), (8, 8), (16, 8), (24, 8)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 28)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (36, 32)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32), (64, 4)],
        ] {
            let malformed = Tf32DriverAbi::checked(malformed.len(), malformed).unwrap();
            super::validate_fixed_sm89_half_swizzle_driver_abi(symbol, &malformed)
                .expect_err("swizzle requires the exact five-argument Driver ABI");
        }
    }

    #[test]
    fn fixed_sm89_half_swizzle_ptx_is_all_or_nothing_and_instruction_exact() {
        let baseline = fixed_sm89_half_test_ptx();
        super::validate_fixed_sm89_half_swizzle_ptx("sm_89", &baseline).unwrap();
        for (symbol, dtype) in FIXED_SM89_HALF_SWIZZLE_TEST_SYMBOLS
            .into_iter()
            .zip(["bf16", "f16"])
        {
            let entry = fixed_sm89_half_test_entry(symbol, dtype);
            super::validate_fixed_sm89_half_swizzle_ptx("sm_89", &baseline.replacen(&entry, "", 1))
                .expect_err("both homogeneous-half swizzle exports are mandatory");
            for required in [
                format!("mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32"),
                "cp.async.cg.shared.global".to_string(),
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16".to_string(),
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16".to_string(),
            ] {
                let malformed = entry.replace(&required, "not_the_required_instruction");
                super::validate_fixed_sm89_half_swizzle_ptx(
                    "sm_89",
                    &baseline.replacen(&entry, &malformed, 1),
                )
                .expect_err("swizzle dtype/staging/fragment instruction drift must reject");
            }
            for mutation in [
                entry.replace("params[32]", "params[28]"),
                entry.replacen("ret;", ".local .b8 spill[16]; ret;", 1),
                entry.replacen("ret;", "atom.global.add.f32 %f0, [%rd0], %f1; ret;", 1),
            ] {
                super::validate_fixed_sm89_half_swizzle_ptx(
                    "sm_89",
                    &baseline.replacen(&entry, &mutation, 1),
                )
                .expect_err("swizzle ABI/local/atomic drift must reject");
            }
        }
        let duplicate = baseline.clone()
            + &fixed_sm89_half_test_entry(FIXED_SM89_HALF_SWIZZLE_TEST_SYMBOLS[0], "bf16");
        super::validate_fixed_sm89_half_swizzle_ptx("sm_89", &duplicate)
            .expect_err("duplicate swizzle export must reject");
        let foreign = baseline.clone()
            + &fixed_sm89_half_test_entry("nn_sm89_tc128_swizzle_decoy_bf16", "bf16");
        super::validate_fixed_sm89_half_swizzle_ptx("sm_89", &foreign)
            .expect_err("foreign swizzle export must reject");
        for target in ["sm_80", "sm_90a"] {
            super::validate_fixed_sm89_half_swizzle_ptx(target, "")
                .expect_err("swizzle exports are mandatory on every portable target");
            super::validate_fixed_sm89_half_swizzle_ptx(target, &baseline).unwrap();
        }
        for target in ["compute_120", "sm_120"] {
            super::validate_fixed_sm89_half_swizzle_ptx(target, "").unwrap();
            super::validate_fixed_sm89_half_swizzle_ptx(target, &baseline)
                .expect_err("swizzle exports must reject on the CC 12 family");
        }
    }

    #[test]
    fn fixed_sm89_half_s3_driver_abi_and_resources_are_strict() {
        let symbol = FIXED_SM89_HALF_S3_TEST_SYMBOLS[0];
        let abi =
            Tf32DriverAbi::checked(5, vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]).unwrap();
        super::validate_fixed_sm89_half_s3_driver_abi(symbol, &abi).unwrap();
        let valid = super::FixedSm89HalfS3Resources {
            local_bytes: 0,
            registers: super::FIXED_SM89_HALF_S3_REGISTER_CAP,
            static_shared_bytes: 0,
            max_threads: 256,
            active_blocks: 1,
        };
        super::validate_fixed_sm89_half_s3_resources(symbol, valid).unwrap();
        super::validate_fixed_sm89_half_s3_shared_capacity(98_304).unwrap();
        super::validate_fixed_sm89_half_s3_shared_capacity(98_303)
            .expect_err("one byte below the opt-in shared requirement must reject");
        for malformed in [
            super::FixedSm89HalfS3Resources {
                local_bytes: 1,
                ..valid
            },
            super::FixedSm89HalfS3Resources {
                registers: super::FIXED_SM89_HALF_S3_REGISTER_CAP + 1,
                ..valid
            },
            super::FixedSm89HalfS3Resources {
                static_shared_bytes: 1,
                ..valid
            },
            super::FixedSm89HalfS3Resources {
                max_threads: 255,
                ..valid
            },
            super::FixedSm89HalfS3Resources {
                active_blocks: 0,
                ..valid
            },
        ] {
            super::validate_fixed_sm89_half_s3_resources(symbol, malformed)
                .expect_err("each physical resource gate must reject independently");
        }
        for malformed in [
            vec![(0, 8), (8, 8), (16, 8), (24, 8)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 28)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (36, 32)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32), (64, 4)],
        ] {
            let malformed = Tf32DriverAbi::checked(malformed.len(), malformed).unwrap();
            super::validate_fixed_sm89_half_s3_driver_abi(symbol, &malformed)
                .expect_err("s3 requires the exact five-argument Driver ABI");
        }
    }

    #[test]
    fn fixed_sm89_half_s3_ptx_is_all_or_nothing_and_instruction_exact() {
        let baseline = fixed_sm89_half_test_ptx();
        super::validate_fixed_sm89_half_s3_ptx("sm_89", &baseline).unwrap();
        for (symbol, dtype) in FIXED_SM89_HALF_S3_TEST_SYMBOLS
            .into_iter()
            .zip(["bf16", "f16"])
        {
            let entry = fixed_sm89_half_s3_test_entry(symbol, dtype);
            super::validate_fixed_sm89_half_s3_ptx("sm_89", &baseline.replacen(&entry, "", 1))
                .expect_err("both homogeneous-half s3 exports are mandatory");
            for required in [
                format!("mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32"),
                "cp.async.cg.shared.global".to_string(),
                "cp.async.commit_group".to_string(),
                "cp.async.wait_group 0".to_string(),
                "cp.async.wait_group 1".to_string(),
                "bar.sync".to_string(),
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16".to_string(),
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16".to_string(),
            ] {
                let malformed = entry.replace(&required, "not_the_required_instruction");
                super::validate_fixed_sm89_half_s3_ptx(
                    "sm_89",
                    &baseline.replacen(&entry, &malformed, 1),
                )
                .expect_err("s3 dtype/staging/fragment instruction drift must reject");
            }
            for mutation in [
                entry.replace("params[32]", "params[28]"),
                entry.replace("cp.async.wait_group 1;", "cp.async.wait_group 10;"),
                entry.replace("cp.async.wait_group 0;", "cp.async.wait_group 01;"),
                entry.replacen("ret;", ".local .b8 spill[16]; ret;", 1),
                entry.replacen("ret;", "ld.local.u32 %r0, [%rd0]; ret;", 1),
                entry.replacen("ret;", "st.local.u32 [%rd0], %r0; ret;", 1),
                entry.replacen("ret;", "red.global.add.f32 [%rd0], %f1; ret;", 1),
                entry.replacen("ret;", "redux.sync.add.s32 %r0, %r1, -1; ret;", 1),
                entry.replacen("ret;", "atom.global.add.f32 %f0, [%rd0], %f1; ret;", 1),
            ] {
                super::validate_fixed_sm89_half_s3_ptx(
                    "sm_89",
                    &baseline.replacen(&entry, &mutation, 1),
                )
                .expect_err("s3 ABI/local/atomic drift must reject");
            }
        }
        let duplicate = baseline.clone()
            + &fixed_sm89_half_s3_test_entry(FIXED_SM89_HALF_S3_TEST_SYMBOLS[0], "bf16");
        super::validate_fixed_sm89_half_s3_ptx("sm_89", &duplicate)
            .expect_err("duplicate s3 export must reject");
        let foreign = baseline.clone()
            + &fixed_sm89_half_s3_test_entry("nn_sm89_tc128_s3_decoy_bf16", "bf16");
        super::validate_fixed_sm89_half_s3_ptx("sm_89", &foreign)
            .expect_err("foreign s3 export must reject");
        for target in ["sm_80", "sm_90a"] {
            super::validate_fixed_sm89_half_s3_ptx(target, "")
                .expect_err("s3 exports are mandatory on every portable target");
            super::validate_fixed_sm89_half_s3_ptx(target, &baseline).unwrap();
        }
        for target in ["compute_120", "sm_120"] {
            super::validate_fixed_sm89_half_s3_ptx(target, "").unwrap();
            super::validate_fixed_sm89_half_s3_ptx(target, &baseline)
                .expect_err("s3 exports must reject on the CC 12 family");
        }
    }

    const FIXED_SM89_RNA_WIDE_TEST_SYMBOL: &str = "nn_rna_wide_tf32_m128n128_bk32_s3";
    const FIXED_SM89_EXACT_N64_TEST_SYMBOL: &str = "nn_sm89_f32_n64_copyplan";
    const FIXED_SM120_EXACT_N64_TEST_SYMBOL: &str = "nn_sm120_f32_n64_copyplan";
    const FIXED_SM120_COPYPLAN_T256_TEST_SYMBOL: &str = "nn_sm120_f32_n64_copyplan_t256";

    fn fixed_sm89_rna_wide_test_entry() -> String {
        format!(
            ".visible .entry {FIXED_SM89_RNA_WIDE_TEST_SYMBOL}(\n\
             .param .u64 c,\n.param .u64 a,\n.param .u64 b,\n.param .u64 bias,\n\
             .param .align 4 .b8 params[32]\n)\n\
             .maxntid 256, 1, 1\n.minnctapersm 1\n{{\n\
             .extern .shared .align 16 .b8 dynamic_smem[];\n\
             cp.async.cg.shared.global [%r0], [%rd0], 16, %r1;\n\
             cp.async.commit_group;\ncp.async.wait_group 0;\nbar.sync 0;\n\
             ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r0,%r1,%r2,%r3}}, [%r4];\n\
             cvt.rna.tf32.f32 %r0, %f0;\n\
             mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n\
             {{%f0,%f1,%f2,%f3}}, {{%r0,%r1,%r2,%r3}}, {{%r4,%r5}}, {{%f0,%f1,%f2,%f3}};\n\
             mul.rn.f32 %f0, %f0, %f4;\nfma.rn.f32 %f0, %f1, %f2, %f0;\n\
             st.global.f32 [%rd0], %f0;\nret;\n}}\n"
        )
    }

    #[test]
    fn fixed_sm89_rna_wide_composer_follows_the_portable_overlay() {
        let source = compose_module_source_for(ModuleKind::Fixed, "sm_89").unwrap();
        assert!(
            source.contains(FIXED_SM89_RNA_WIDE_TEST_SYMBOL),
            "production Fixed/sm_89 composition is missing the RNA-wide export"
        );
        let mut retained = super::FIXED_SOURCE_FRAGMENTS.to_vec();
        retained.extend([
            super::FIXED_SM89_HALF_SOURCE_FRAGMENT,
            super::FIXED_SM89_EXACT_N64_SOURCE_FRAGMENT,
        ]);
        let before = compose_fragments(&retained).unwrap();
        let old_suffix = compose_fragments(&[
            super::FIXED_SM89_RNA_WIDE_SOURCE_FRAGMENT,
            super::FIXED_SM89_HALF_SWIZZLE_LAYOUT_FRAGMENT,
            super::FIXED_SM89_HALF_SWIZZLE_SOURCE_FRAGMENT,
            super::FIXED_SM89_HALF_S3_SOURCE_FRAGMENT,
        ])
        .unwrap();
        assert!(
            source
                .strip_prefix(&before)
                .unwrap()
                .starts_with(&old_suffix),
            "RNA-wide and the later swizzle twin must follow the prior Ada Fixed bytes"
        );
        for target in [
            "sm_80", "sm_86", "sm_87", "sm_90", "sm_90a", "sm_100a", "sm_103a", "sm_107a",
            "sm_110a",
        ] {
            assert!(
                compose_module_source_for(ModuleKind::Fixed, target)
                    .unwrap()
                    .contains(FIXED_SM89_RNA_WIDE_TEST_SYMBOL),
                "RNA-wide is missing from the portable overlay on Fixed/{target}"
            );
        }
        for target in [
            "compute_89",
            "sm_120",
            "sm_121",
            "compute_120",
            "compute_121",
        ] {
            assert!(
                !compose_module_source_for(ModuleKind::Fixed, target)
                    .unwrap()
                    .contains(FIXED_SM89_RNA_WIDE_TEST_SYMBOL),
                "RNA-wide leaked into Fixed/{target}"
            );
        }
        for target in ["sm_89", "sm_120", "compute_120", "sm_121", "compute_121"] {
            for kind in [
                ModuleKind::TriadScalar,
                ModuleKind::TriadSm80,
                ModuleKind::TriadSm90a,
                ModuleKind::TriadSm100,
                ModuleKind::TriadSm120,
            ] {
                assert!(
                    !compose_module_source_for(kind, target)
                        .unwrap()
                        .contains(FIXED_SM89_RNA_WIDE_TEST_SYMBOL),
                    "RNA-wide leaked into {kind:?}/{target}"
                );
            }
        }
    }

    #[test]
    fn fixed_sm89_rna_wide_validator_accepts_the_exact_rna_pipeline_entry() {
        let ptx = fixed_sm89_half_test_ptx();
        validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
            .expect("the exact five-argument RNA-wide entry must be admitted on Fixed/sm_89");
    }

    #[test]
    fn fixed_sm89_rna_wide_validator_rejects_inventory_abi_and_body_drift() {
        let baseline = fixed_sm89_half_test_ptx();
        let entry = fixed_sm89_rna_wide_test_entry();
        for malformed in [
            baseline.replacen(&entry, "", 1),
            format!("{baseline}{entry}"),
            baseline.replacen(
                &entry,
                &entry.replacen("rna_wide_tf32", "rna_wide_tf32_decoy", 1),
                1,
            ),
        ] {
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &malformed)
                .expect_err("missing, duplicate, or foreign RNA-wide inventory must reject");
        }
        for malformed_entry in [
            entry.replacen(".param .u64 a,", ".param .u32 a,", 1),
            entry.replacen("params[32]", "params[24]", 1),
            entry.replacen(".align 4 .b8 params", ".align 8 .b8 params", 1),
            entry.replacen("params[32]\n)", "params[32],\n.param .u64 scratch\n)", 1),
            entry.replacen("cvt.rna.tf32.f32", "cvt.rn.f32.f32", 1),
            entry.replacen(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                "not_the_required_mma",
                1,
            ),
            entry.replacen("cp.async.cg.shared.global", "not_the_required_async", 1),
            entry.replacen("cp.async.commit_group", "not_the_required_commit", 1),
            entry.replacen("cp.async.wait_group", "not_the_required_wait", 1),
            entry.replacen(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                "not_the_required_ldmatrix",
                1,
            ),
            entry.replacen(".maxntid 256", ".maxntid 128", 1),
            entry.replacen("ret;", ".local .b8 spill[16]; ret;", 1),
            entry.replacen("ret;", "atom.global.add.f32 %f0, [%rd0], %f1; ret;", 1),
            entry.replacen("ret;", "red.global.add.f32 [%rd0], %f1; ret;", 1),
            entry.replacen("ret;", "redux.sync.add.s32 %r0, %r1, -1; ret;", 1),
        ] {
            let malformed = baseline.replacen(&entry, &malformed_entry, 1);
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &malformed)
                .expect_err("RNA-wide inventory, ABI, pipeline, and reduction drift must reject");
        }
        for target in ["sm_80", "sm_90a"] {
            super::validate_fixed_sm89_rna_wide_ptx(target, &entry)
                .expect("the RNA-wide entry travels with the portable overlay");
        }
        for target in ["compute_89", "sm_120", "compute_120"] {
            super::validate_fixed_sm89_rna_wide_ptx(target, &entry)
                .expect_err("RNA-wide entry is foreign outside the portable overlay");
        }
    }

    #[test]
    fn fixed_sm89_rna_wide_driver_abi_and_resource_gates_are_strict() {
        let abi =
            Tf32DriverAbi::checked(5, vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]).unwrap();
        super::validate_fixed_sm89_rna_wide_driver_abi(&abi).unwrap();
        for layout in [
            vec![(0, 8), (8, 8), (16, 8), (24, 8)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 28)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (36, 32)],
            vec![(0, 4), (8, 8), (16, 8), (24, 8), (32, 32)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32), (64, 4)],
        ] {
            let malformed = Tf32DriverAbi::checked(layout.len(), layout).unwrap();
            super::validate_fixed_sm89_rna_wide_driver_abi(&malformed).unwrap_err();
        }
        let admitted = super::FixedSm89RnaWideResources {
            local_bytes: 0,
            registers: 224,
            static_shared_bytes: 0,
            max_threads: 256,
            active_blocks: 1,
        };
        super::validate_fixed_sm89_rna_wide_resources(admitted).unwrap();
        for rejected in [
            super::FixedSm89RnaWideResources {
                local_bytes: 1,
                ..admitted
            },
            super::FixedSm89RnaWideResources {
                registers: 225,
                ..admitted
            },
            super::FixedSm89RnaWideResources {
                static_shared_bytes: 1,
                ..admitted
            },
            super::FixedSm89RnaWideResources {
                max_threads: 255,
                ..admitted
            },
            super::FixedSm89RnaWideResources {
                active_blocks: 0,
                ..admitted
            },
        ] {
            super::validate_fixed_sm89_rna_wide_resources(rejected).unwrap_err();
        }
    }

    #[test]
    fn fixed_sm89_finalist_ptx_abi_and_resources_fail_per_symbol() {
        let baseline = fixed_sm89_half_test_ptx();
        super::validate_fixed_sm89_finalist_ptx("sm_89", &baseline).unwrap();
        let parsed = super::parse_ptx(&baseline).unwrap();
        for symbol in [
            FIXED_SM89_RNA_N96_TEST_SYMBOL,
            FIXED_SM89_HALF_M64N64_S3_TEST_SYMBOL,
            FIXED_SM89_HALF_M128N64_S2_TEST_SYMBOL,
        ] {
            super::validate_fixed_sm89_finalist_entry_ptx(&parsed, symbol).unwrap();
        }
        let n96_entry = fixed_sm89_rna_wide_test_entry().replace(
            FIXED_SM89_RNA_WIDE_TEST_SYMBOL,
            FIXED_SM89_RNA_N96_TEST_SYMBOL,
        );
        for invalid_inventory in [
            baseline.replacen(&n96_entry, "", 1),
            format!("{baseline}{n96_entry}"),
            baseline.replacen(
                FIXED_SM89_RNA_N96_TEST_SYMBOL,
                "nn_sm89_rna_tf32_decoy_m128n96_bk32_s3",
                1,
            ),
        ] {
            super::validate_fixed_sm89_finalist_ptx("sm_89", &invalid_inventory)
                .expect_err("missing, duplicate, or foreign finalist inventory must reject");
        }

        let d_entry =
            fixed_sm89_finalist_half_test_entry(FIXED_SM89_HALF_M64N64_S3_TEST_SYMBOL, true);
        let malformed_d = d_entry.replacen(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            ".local .b8 spill[16]",
            1,
        );
        let one_bad = baseline.replacen(&d_entry, &malformed_d, 1);
        super::validate_fixed_sm89_finalist_ptx("sm_89", &one_bad)
            .expect("global integrity owns inventory, not optional per-symbol eligibility");
        let parsed = super::parse_ptx(&one_bad).unwrap();
        super::validate_fixed_sm89_finalist_entry_ptx(
            &parsed,
            FIXED_SM89_HALF_M64N64_S3_TEST_SYMBOL,
        )
        .expect_err("one malformed optional entry must reject itself");
        for sibling in [
            FIXED_SM89_RNA_N96_TEST_SYMBOL,
            FIXED_SM89_HALF_M128N64_S2_TEST_SYMBOL,
        ] {
            super::validate_fixed_sm89_finalist_entry_ptx(&parsed, sibling)
                .expect("one malformed optional entry must retain its siblings");
        }

        let abi =
            Tf32DriverAbi::checked(5, vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]).unwrap();
        let malformed_abi =
            Tf32DriverAbi::checked(5, vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 28)]).unwrap();
        super::validate_fixed_sm89_finalist_driver_abi(
            FIXED_SM89_HALF_M64N64_S3_TEST_SYMBOL,
            &malformed_abi,
        )
        .expect_err("one malformed live ABI must reject that holder");
        super::validate_fixed_sm89_finalist_driver_abi(FIXED_SM89_RNA_N96_TEST_SYMBOL, &abi)
            .expect("an independent sibling ABI remains admitted");

        for (symbol, registers, active_blocks) in [
            (FIXED_SM89_RNA_N96_TEST_SYMBOL, 136, 1),
            (FIXED_SM89_HALF_M64N64_S3_TEST_SYMBOL, 110, 2),
            (FIXED_SM89_HALF_M128N64_S2_TEST_SYMBOL, 132, 2),
        ] {
            let admitted = super::FixedSm89FinalistResources {
                local_bytes: 0,
                registers,
                static_shared_bytes: 0,
                max_threads: if active_blocks == 1 { 256 } else { 128 },
                active_blocks,
            };
            super::validate_fixed_sm89_finalist_resources(symbol, admitted).unwrap();
            for rejected in [
                super::FixedSm89FinalistResources {
                    local_bytes: 1,
                    ..admitted
                },
                super::FixedSm89FinalistResources {
                    registers: registers + 1,
                    ..admitted
                },
                super::FixedSm89FinalistResources {
                    static_shared_bytes: 1,
                    ..admitted
                },
                super::FixedSm89FinalistResources {
                    max_threads: admitted.max_threads - 1,
                    ..admitted
                },
                super::FixedSm89FinalistResources {
                    active_blocks: active_blocks + 1,
                    ..admitted
                },
            ] {
                super::validate_fixed_sm89_finalist_resources(symbol, rejected)
                    .expect_err("each optional holder resource gate must reject independently");
            }
        }
    }

    fn fixed_sm120_copyplan_t256_test_entry() -> String {
        fixed_sm89_exact_n64_test_entry(FIXED_SM120_COPYPLAN_T256_TEST_SYMBOL)
            .replace(".maxntid 128", ".maxntid 256")
            .replace(".minnctapersm 2", ".minnctapersm 3")
    }

    fn fixed_sm120_copyplan_m128_test_entry() -> String {
        fixed_sm89_exact_n64_test_entry("nn_sm120_f32_n64_copyplan_m128n64_t256")
            .replace(".maxntid 128", ".maxntid 256")
    }

    #[test]
    fn fixed_sm120_copyplan_m128n64_t256_requires_own_export_and_two_cta_resources() {
        let symbol = "nn_sm120_f32_n64_copyplan_m128n64_t256";
        let source = compose_module_source_for(ModuleKind::Fixed, "compute_120").unwrap();
        assert!(
            source.contains(symbol),
            "missing M128N64 T256 CopyPlan export"
        );
    }

    #[test]
    fn fixed_sm120_copyplan_m128n64_t256_resource_contract_is_independent() {
        let symbol = "nn_sm120_f32_n64_copyplan_m128n64_t256";
        let admitted = super::FixedSm89ExactN64Resources {
            local_bytes: 0,
            registers: 128,
            static_shared_bytes: 49_152,
            max_threads: 256,
            active_blocks: 2,
            preferred_carveout: 100,
        };
        super::validate_fixed_sm120_exact_n64_resources_for(symbol, admitted).unwrap();
        for resources in [
            super::FixedSm89ExactN64Resources {
                registers: 129,
                ..admitted
            },
            super::FixedSm89ExactN64Resources {
                local_bytes: 1,
                ..admitted
            },
            super::FixedSm89ExactN64Resources {
                static_shared_bytes: 32_768,
                ..admitted
            },
            super::FixedSm89ExactN64Resources {
                max_threads: 128,
                ..admitted
            },
            super::FixedSm89ExactN64Resources {
                active_blocks: 1,
                ..admitted
            },
        ] {
            assert!(
                super::validate_fixed_sm120_exact_n64_resources_for(symbol, resources).is_err()
            );
        }
    }

    #[test]
    fn fixed_sm120_copyplan_t256_resources_require_three_ctas_without_spills() {
        let admitted = super::FixedSm89ExactN64Resources {
            local_bytes: 0,
            registers: 85,
            static_shared_bytes: 32_768,
            max_threads: 256,
            active_blocks: 3,
            preferred_carveout: 100,
        };
        super::validate_fixed_sm120_exact_n64_resources_for(
            FIXED_SM120_COPYPLAN_T256_TEST_SYMBOL,
            admitted,
        )
        .unwrap();
        for resources in [
            super::FixedSm89ExactN64Resources {
                registers: 86,
                ..admitted
            },
            super::FixedSm89ExactN64Resources {
                local_bytes: 1,
                ..admitted
            },
            super::FixedSm89ExactN64Resources {
                max_threads: 128,
                ..admitted
            },
            super::FixedSm89ExactN64Resources {
                active_blocks: 2,
                ..admitted
            },
        ] {
            assert!(
                super::validate_fixed_sm120_exact_n64_resources_for(
                    FIXED_SM120_COPYPLAN_T256_TEST_SYMBOL,
                    resources
                )
                .is_err()
            );
        }
    }

    #[test]
    fn fixed_sm120_copyplan_t256_has_own_fixed_export_and_launch_bounds() {
        let source = compose_module_source_for(ModuleKind::Fixed, "compute_120").unwrap();
        assert!(
            source.contains(FIXED_SM120_COPYPLAN_T256_TEST_SYMBOL),
            "missing T256 CopyPlan export"
        );
    }

    #[test]
    fn fixed_sm120_copyplan_t256_ptx_retains_control_and_requires_twin() {
        let control = fixed_sm89_exact_n64_test_entry(FIXED_SM120_EXACT_N64_TEST_SYMBOL);
        let twin = fixed_sm120_copyplan_t256_test_entry();
        let wide = fixed_sm120_copyplan_m128_test_entry();
        let valid = format!("{control}{twin}{wide}");
        super::validate_fixed_sm120_exact_n64_ptx("compute_120", &valid).unwrap();
        for malformed in [
            control.clone(),
            twin.clone(),
            valid.replace(&wide, ""),
            valid.replace(&wide, &wide.replace(".maxntid 256", ".maxntid 128")),
            valid.replace(&wide, &wide.replace(".minnctapersm 2", ".minnctapersm 3")),
            format!("{valid}{twin}"),
            valid.replace(&twin, &twin.replace(".maxntid 256", ".maxntid 128")),
            format!(
                "{control}{wide}{}",
                twin.replace(".minnctapersm 3", ".minnctapersm 2")
            ),
        ] {
            assert!(super::validate_fixed_sm120_exact_n64_ptx("compute_120", &malformed).is_err());
        }
        for arch in ["sm_89", "sm_120", "compute_121"] {
            assert!(super::validate_fixed_sm120_exact_n64_ptx(arch, &twin).is_err());
        }
    }
    const FIXED_SM120_POSTBIAS_TEST_SYMBOLS: [&str; 6] = [
        "nn_sm120_tma_fma_postbias_m128n64_bk16_s2",
        "nn_sm120_tma_fma_postbias_m64n128_bk16_s2",
        "nn_sm120_tma_fma_postbias_m128n96_bk16_s2",
        "nn_sm120_tma_fma_postbias_m128n64_bk16_s2_k4",
        "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2",
        "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2",
    ];

    fn fixed_sm120_postbias_test_entry(symbol: &str, tensor_map_alignment: usize) -> String {
        let threads = if symbol.contains("_m128n96_") || symbol.contains("_t256_") {
            256
        } else {
            128
        };
        let epilogue = if symbol.contains("_nobias_") {
            "st.global.f32 [%rd0], %f3;"
        } else {
            "add.rn.f32 %f5, %f3, %f6;\nst.global.f32 [%rd0], %f5;"
        };
        format!(
            ".visible .entry {symbol}(\n\
             .param .u64 output,\n.param .u64 slabs,\n.param .u64 flags,\n\
             .param .align {tensor_map_alignment} .b8 a_map[128],\n\
             .param .align {tensor_map_alignment} .b8 b_map[128],\n.param .u64 bias,\n\
             .param .align 4 .b8 params[32]\n)\n\
             .maxntid {threads}, 1, 1\n.minnctapersm 3\n{{\n\
             cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes;\n\
             fma.rn.f32 %f0, %f1, %f2, %f0;\n\
             mul.rn.f32 %f3, %f0, %f4;\n{epilogue}\nret;\n}}\n"
        )
    }

    fn fixed_sm120_postbias_test_ptx(tensor_map_alignment: usize) -> String {
        FIXED_SM120_POSTBIAS_TEST_SYMBOLS
            .iter()
            .map(|symbol| fixed_sm120_postbias_test_entry(symbol, tensor_map_alignment))
            .collect()
    }

    #[test]
    fn fixed_sm120_postbias_composition_is_fixed_owned() {
        let compute120 = compose_module_source_for(ModuleKind::Fixed, "compute_120").unwrap();
        for symbol in FIXED_SM120_POSTBIAS_TEST_SYMBOLS {
            assert_eq!(
                compute120
                    .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
                    .filter(|token| *token == symbol)
                    .count(),
                1,
                "missing or duplicate Fixed export {symbol}"
            );
        }
        for target in ["sm_89", "sm_120", "sm_121", "compute_121"] {
            let source = compose_module_source_for(ModuleKind::Fixed, target).unwrap();
            for symbol in FIXED_SM120_POSTBIAS_TEST_SYMBOLS {
                assert!(
                    !source.contains(symbol),
                    "foreign Fixed target {target} exports {symbol}"
                );
            }
        }
        for kind in [
            ModuleKind::TriadScalar,
            ModuleKind::TriadSm80,
            ModuleKind::TriadSm90a,
            ModuleKind::TriadSm100,
            ModuleKind::TriadSm120,
        ] {
            let source = compose_module_source_for(kind, "compute_120").unwrap();
            for symbol in FIXED_SM120_POSTBIAS_TEST_SYMBOLS {
                assert!(!source.contains(symbol), "foreign {kind:?} export {symbol}");
            }
        }
    }

    #[test]
    fn fixed_sm120_postbias_ptx_is_exact_target_scoped_and_fail_closed() {
        let valid = fixed_sm120_postbias_test_ptx(128);
        super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &valid, 13).unwrap();
        for arch in ["sm_89", "sm_120", "compute_121", "sm_121"] {
            super::validate_fixed_sm120_postbias_ptx_for_cuda_major(arch, "", 13).unwrap();
            super::validate_fixed_sm120_postbias_ptx_for_cuda_major(arch, &valid, 13).unwrap_err();
        }
        let first = fixed_sm120_postbias_test_entry(FIXED_SM120_POSTBIAS_TEST_SYMBOLS[0], 128);
        let t256 = fixed_sm120_postbias_test_entry(FIXED_SM120_POSTBIAS_TEST_SYMBOLS[4], 128);
        for malformed in [
            valid.replacen(&first, "", 1),
            valid.replacen(&t256, "", 1),
            valid.replacen(&t256, &t256.replace(".maxntid 256", ".maxntid 128"), 1),
            valid.replacen(&t256, &t256.replace(".u64 bias", ".u32 bias"), 1),
            format!("{valid}{first}"),
            valid.replacen("postbias_m128n64", "postbias_m128n64_decoy", 1),
            valid.replacen("nn_sm120", "tn_sm120", 1),
            valid.replacen(".param .u64 flags", ".param .u32 flags", 1),
            valid.replacen(".align 128 .b8 a_map", ".align 64 .b8 a_map", 1),
            valid.replacen(".align 128 .b8 a_map[128]", ".align 128 .b8 a_map[64]", 1),
            valid.replacen(".align 4 .b8 params[32]", ".align 8 .b8 params[32]", 1),
            valid.replacen(".maxntid 128, 1, 1", ".maxntid 64, 1, 1", 1),
            valid.replacen(".maxntid 256, 1, 1", ".maxntid 128, 1, 1", 1),
            valid.replacen(".minnctapersm 3", ".minnctapersm 2", 1),
        ] {
            super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &malformed, 13)
                .unwrap_err();
        }
        for (from, to) in [
            (
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                "mov.u32",
            ),
            ("fma.rn.f32", "mad.rn.f32"),
            ("mul.rn.f32", "mul.rz.f32"),
            ("add.rn.f32", "add.rz.f32"),
        ] {
            let malformed = valid.replacen(from, to, 1);
            super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &malformed, 13)
                .unwrap_err();
        }
        for forbidden in [
            ".local .align 4 .b8 spill[16];",
            "ld.local.f32 %f0, [%rd0];",
            "st.local.f32 [%rd0], %f0;",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32;",
            "wmma.mma.sync.aligned.row.col.f32.f32;",
            "cvt.rna.tf32.f32 %r0, %f0;",
            "add.rn.ftz.f32 %f0, %f1, %f2;",
            "atom.global.add.f32 %f0, [%rd0], %f1;",
            "red.global.add.f32 [%rd0], %f1;",
            "redux.sync.add.s32 %r0, %r1, -1;",
            "call.uni helper, ();",
        ] {
            let malformed = valid.replacen("ret;", &format!("{forbidden}\nret;"), 1);
            super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &malformed, 13)
                .unwrap_err();
        }
    }

    #[test]
    fn fixed_sm120_postbias_ptx_tracks_cuda_12_and_13_tensor_map_alignment() {
        let cuda12 = fixed_sm120_postbias_test_ptx(64);
        let cuda13 = fixed_sm120_postbias_test_ptx(128);
        super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &cuda12, 12)
            .unwrap();
        super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &cuda13, 13)
            .unwrap();
        assert!(
            super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &cuda12, 13)
                .is_err()
        );
        assert!(
            super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &cuda13, 12)
                .is_err()
        );
        assert!(
            super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &cuda13, 14)
                .is_err()
        );
    }

    #[test]
    fn fixed_sm120_nobias_ptx_requires_own_export_and_scale_only_epilogue() {
        let valid = fixed_sm120_postbias_test_ptx(128);
        let symbol = "nn_sm120_tma_fma_nobias_m128n64_t256_bk16_s2";
        let entry = fixed_sm120_postbias_test_entry(symbol, 128);
        super::validate_fixed_sm120_postbias_ptx_for_cuda_major("compute_120", &valid, 13).unwrap();
        for malformed in [
            valid.replace(&entry, ""),
            format!("{valid}{entry}"),
            valid.replace(&entry, &entry.replace(".maxntid 256", ".maxntid 128")),
            valid.replace(&entry, &entry.replace(".u64 bias", ".u32 bias")),
            valid.replace(&entry, &entry.replace("mul.rn.f32", "mul.rz.f32")),
            valid.replace(
                &entry,
                &entry.replace("ret;", "add.rn.f32 %f0, %f1, %f2;\nret;"),
            ),
        ] {
            assert!(
                super::validate_fixed_sm120_postbias_ptx_for_cuda_major(
                    "compute_120",
                    &malformed,
                    13,
                )
                .is_err()
            );
        }
        for arch in ["sm_89", "sm_120", "compute_121"] {
            assert!(
                super::validate_fixed_sm120_postbias_ptx_for_cuda_major(arch, &entry, 13).is_err()
            );
        }
    }

    #[test]
    fn fixed_sm120_postbias_t256_requires_three_ctas_within_register_budget() {
        use super::{
            FixedSm120PostbiasResources, fixed_sm120_postbias_launch_contract,
            validate_fixed_sm120_postbias_resources,
        };
        let symbol = "nn_sm120_tma_fma_postbias_m128n64_t256_bk16_s2";
        let launch = fixed_sm120_postbias_launch_contract(symbol).unwrap();
        assert_eq!(launch.threads, 256);
        assert_eq!(launch.dynamic_shared, 24_592);
        assert_eq!(launch.min_active_blocks, 3);
        let admitted = FixedSm120PostbiasResources {
            local_bytes: 0,
            registers: 85,
            max_threads: 256,
            active_blocks: 3,
        };
        validate_fixed_sm120_postbias_resources(symbol, admitted).unwrap();
        for rejected in [
            FixedSm120PostbiasResources {
                registers: 86,
                ..admitted
            },
            FixedSm120PostbiasResources {
                registers: 0,
                ..admitted
            },
            FixedSm120PostbiasResources {
                local_bytes: 1,
                ..admitted
            },
            FixedSm120PostbiasResources {
                max_threads: 128,
                ..admitted
            },
            FixedSm120PostbiasResources {
                active_blocks: 2,
                ..admitted
            },
        ] {
            assert!(validate_fixed_sm120_postbias_resources(symbol, rejected).is_err());
        }
    }

    #[test]
    fn fixed_sm120_postbias_driver_abi_and_resources_are_strict() {
        use super::{
            FixedSm120PostbiasLaunchContract, FixedSm120PostbiasResources,
            fixed_sm120_postbias_launch_contract, validate_fixed_sm120_postbias_resources,
        };
        use cudarc::driver::sys::CUresult;

        assert_eq!(
            super::FIXED_SM120_POSTBIAS_REGISTER_CAP as u32,
            super::super::contract::SM120_FMA_REGISTER_CAP
        );
        let layout = [
            (0, 8),
            (8, 8),
            (16, 8),
            (128, 128),
            (256, 128),
            (384, 8),
            (392, 32),
        ];
        let mut queried = Vec::new();
        let abi = super::query_driver_parameter_abi("Fixed/postbias", 7, |index, offset, size| {
            queried.push(index);
            if let Some((parameter_offset, parameter_size)) = layout.get(index) {
                *offset = *parameter_offset;
                *size = *parameter_size;
                CUresult::CUDA_SUCCESS
            } else {
                CUresult::CUDA_ERROR_INVALID_VALUE
            }
        })
        .unwrap();
        assert_eq!(queried, [0, 1, 2, 3, 4, 5, 6, 7]);
        for symbol in FIXED_SM120_POSTBIAS_TEST_SYMBOLS {
            let expected_launch = if symbol.contains("_m128n96_") {
                FixedSm120PostbiasLaunchContract {
                    threads: 256,
                    dynamic_shared: 28_688,
                    min_active_blocks: 3,
                }
            } else if symbol.contains("_t256_") {
                FixedSm120PostbiasLaunchContract {
                    threads: 256,
                    dynamic_shared: 24_592,
                    min_active_blocks: 3,
                }
            } else {
                FixedSm120PostbiasLaunchContract {
                    threads: 128,
                    dynamic_shared: 24_592,
                    min_active_blocks: 3,
                }
            };
            assert_eq!(
                fixed_sm120_postbias_launch_contract(symbol).unwrap(),
                expected_launch
            );
            let admitted = FixedSm120PostbiasResources {
                local_bytes: 0,
                registers: if symbol.contains("_t256_") { 85 } else { 168 },
                max_threads: expected_launch.threads as i32,
                active_blocks: expected_launch.min_active_blocks,
            };
            super::validate_fixed_sm120_postbias_driver_abi_for_cuda_major(symbol, &abi, 13)
                .unwrap();
            validate_fixed_sm120_postbias_resources(symbol, admitted).unwrap();
            for resources in [
                FixedSm120PostbiasResources {
                    local_bytes: 1,
                    ..admitted
                },
                FixedSm120PostbiasResources {
                    registers: 0,
                    ..admitted
                },
                FixedSm120PostbiasResources {
                    registers: admitted.registers + 1,
                    ..admitted
                },
                FixedSm120PostbiasResources {
                    max_threads: expected_launch.threads as i32 - 1,
                    ..admitted
                },
                FixedSm120PostbiasResources {
                    active_blocks: expected_launch.min_active_blocks - 1,
                    ..admitted
                },
            ] {
                assert!(validate_fixed_sm120_postbias_resources(symbol, resources).is_err());
            }
        }
        assert!(fixed_sm120_postbias_launch_contract("unknown").is_err());
        for layout in [
            vec![(0, 8), (8, 8), (16, 8), (128, 128), (256, 128), (384, 8)],
            vec![
                (0, 8),
                (8, 8),
                (16, 8),
                (128, 128),
                (256, 128),
                (384, 8),
                (392, 28),
            ],
            vec![
                (0, 8),
                (8, 8),
                (16, 8),
                (128, 128),
                (256, 128),
                (384, 8),
                (396, 32),
            ],
        ] {
            let malformed = Tf32DriverAbi::checked(layout.len(), layout).unwrap();
            super::validate_fixed_sm120_postbias_driver_abi_for_cuda_major(
                FIXED_SM120_POSTBIAS_TEST_SYMBOLS[0],
                &malformed,
                13,
            )
            .unwrap_err();
        }
    }

    #[test]
    fn fixed_sm120_postbias_driver_abi_tracks_cuda_12_and_13_tensor_map_alignment() {
        let symbol = FIXED_SM120_POSTBIAS_TEST_SYMBOLS[0];
        let cuda12 = Tf32DriverAbi::checked(
            7,
            vec![
                (0, 8),
                (8, 8),
                (16, 8),
                (64, 128),
                (192, 128),
                (320, 8),
                (328, 32),
            ],
        )
        .unwrap();
        let cuda13 = Tf32DriverAbi::checked(
            7,
            vec![
                (0, 8),
                (8, 8),
                (16, 8),
                (128, 128),
                (256, 128),
                (384, 8),
                (392, 32),
            ],
        )
        .unwrap();

        super::validate_fixed_sm120_postbias_driver_abi_for_cuda_major(symbol, &cuda12, 12)
            .unwrap();
        super::validate_fixed_sm120_postbias_driver_abi_for_cuda_major(symbol, &cuda13, 13)
            .unwrap();
        assert!(
            super::validate_fixed_sm120_postbias_driver_abi_for_cuda_major(symbol, &cuda12, 13)
                .is_err()
        );
        assert!(
            super::validate_fixed_sm120_postbias_driver_abi_for_cuda_major(symbol, &cuda13, 12)
                .is_err()
        );
        assert!(
            super::validate_fixed_sm120_postbias_driver_abi_for_cuda_major(symbol, &cuda13, 14)
                .is_err()
        );
    }

    #[test]
    fn fixed_sm120_sliced_composition_is_fixed_compute120_only() {
        let symbol = "nn_sm120_f32_n64_sliced";
        let before = compose_module_source_for(ModuleKind::Fixed, "compute_120").unwrap();
        assert!(
            before.contains(symbol),
            "new sliced export is missing from actual production composition"
        );
        let mut retained = super::FIXED_SOURCE_FRAGMENTS.to_vec();
        retained.push(super::FIXED_SM120_EXACT_N64_SOURCE_FRAGMENT);
        let old = compose_fragments(&retained).unwrap();
        assert_eq!(
            before.strip_prefix(&old).unwrap(),
            compose_fragments(&[
                super::FIXED_SM120_SLICED_SOURCE_FRAGMENT,
                super::FIXED_SM120_POSTBIAS_SOURCE_FRAGMENT,
            ])
            .unwrap()
        );
        assert_ne!(
            super::FramedSha256::bytes(before.as_bytes()),
            super::FramedSha256::bytes(old.as_bytes()),
            "Fixed source identity must invalidate old cached graphs"
        );
        for arch in ["sm_80", "sm_89", "sm_120", "sm_121", "compute_121"] {
            assert!(
                !compose_module_source_for(ModuleKind::Fixed, arch)
                    .unwrap()
                    .contains(symbol)
            );
        }
        for kind in [
            ModuleKind::TriadScalar,
            ModuleKind::TriadSm80,
            ModuleKind::TriadSm120,
        ] {
            assert!(
                !compose_module_source_for(kind, "compute_120")
                    .unwrap()
                    .contains(symbol)
            );
        }
    }

    #[test]
    fn fixed_sm120_sliced_ptx_fails_closed_on_abi_body_inventory_and_target() {
        let symbol = "nn_sm120_f32_n64_sliced";
        let valid = fixed_sm89_exact_n64_test_entry(symbol);
        super::validate_fixed_sm120_sliced_ptx("compute_120", &valid).unwrap();
        for malformed in [
            String::new(),
            format!("{valid}{valid}"),
            valid.replace("sliced", "sliced_decoy"),
            valid.replace("params[32]", "params[36]"),
            valid.replace("fma.rn.f32", "add.rn.f32"),
            valid.replace("cp.async.wait_group", "mov.u32"),
            valid.replace("ret;", ".local .b8 spill[16]; ret;"),
        ] {
            super::validate_fixed_sm120_sliced_ptx("compute_120", &malformed)
                .expect_err("sliced ABI/body/inventory drift must reject");
        }
        for arch in ["sm_89", "sm_120", "compute_121"] {
            super::validate_fixed_sm120_sliced_ptx(arch, &valid)
                .expect_err("foreign sliced target");
            super::validate_fixed_sm120_sliced_ptx(arch, "").unwrap();
        }
    }

    fn fixed_sm89_exact_n64_test_entry(symbol: &str) -> String {
        format!(
            ".visible .entry {symbol}(\n\
             .param .u64 c,\n.param .u64 a,\n.param .u64 b,\n.param .u64 bias,\n\
             .param .align 4 .b8 params[32]\n)\n\
             .maxntid 128, 1, 1\n.minnctapersm 2\n{{\n\
             .shared .align 16 .b8 a_stage[16384];\n\
             .shared .align 16 .b8 b_stage[16384];\n\
             cp.async.cg.shared.global [%r0], [%rd0], 16, %r1;\n\
             cp.async.commit_group;\ncp.async.wait_group 0;\nbar.sync 0;\n\
             fma.rn.f32 %f0, %f1, %f2, %f0;\n\
             mul.rn.f32 %f0, %f0, %f3;\nadd.rn.f32 %f0, %f0, %f4;\n\
             st.global.f32 [%rd0], %f0;\nret;\n}}\n"
        )
    }

    fn fixed_sm89_exact_n64_test_ptx() -> String {
        fixed_sm89_half_test_ptx()
    }

    #[test]
    fn fixed_sm89_exact_n64_composition_is_target_scoped() {
        let mut retained = super::FIXED_SOURCE_FRAGMENTS.to_vec();
        retained.push(super::FIXED_SM89_HALF_SOURCE_FRAGMENT);
        let before = compose_fragments(&retained).unwrap();
        let after = compose_module_source_for(ModuleKind::Fixed, "sm_89").unwrap();
        let new = after
            .strip_prefix(&before)
            .expect("old Ada Fixed bytes changed");
        let boundaries: Vec<_> = new
            .lines()
            .filter_map(|line| line.strip_prefix("#line 1 \"")?.strip_suffix('"'))
            .collect();
        assert_eq!(
            boundaries,
            [
                "kernels/gemm_bi_inference/sm80/f32_n64_copyplan.cu",
                "kernels/gemm_bi_inference/sm80/tf32_rna_wide.cu",
                "kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh",
                "kernels/gemm_bi_inference/sm80/half_swizzle.cu",
                "kernels/gemm_bi_inference/sm80/half_s3.cu",
                "kernels/gemm_bi_inference/sm80/tf32_rna_n96.cu",
                "kernels/gemm_bi_inference/sm80/half_n64.cu",
            ]
        );
    }

    #[test]
    fn fixed_sm120_exact_n64_composition_is_compute120_only() {
        let production = compose_module_source_for(ModuleKind::Fixed, "compute_120").unwrap();
        assert!(
            production.contains(FIXED_SM120_EXACT_N64_TEST_SYMBOL),
            "the actual CC12.0 NVRTC target must compose the SM120 exact-N64 route"
        );
        for arch in ["sm_89", "compute_89", "sm_120", "sm_121", "compute_121"] {
            assert!(
                !compose_module_source_for(ModuleKind::Fixed, arch)
                    .unwrap()
                    .contains(FIXED_SM120_EXACT_N64_TEST_SYMBOL),
                "unqualified target {arch} must not compose the SM120 exact-N64 route"
            );
        }
    }

    #[test]
    fn fixed_sm120_exact_n64_export_is_rejected_on_foreign_targets() {
        for arch in ["sm_80", "sm_89", "sm_120", "compute_121"] {
            let ptx = fixed_sm89_exact_n64_test_entry(FIXED_SM120_EXACT_N64_TEST_SYMBOL);
            super::validate_fixed_sm120_exact_n64_ptx(arch, &ptx)
                .expect_err("an SM120 exact-N64 export on a foreign target must be rejected");
        }
        let mut ada = fixed_sm89_exact_n64_test_ptx();
        ada.push_str(&fixed_sm89_exact_n64_test_entry(
            FIXED_SM120_EXACT_N64_TEST_SYMBOL,
        ));
        validate_module_ptx(ModuleKind::Fixed, "sm_89", &ada)
            .expect_err("the complete Ada module must reject the foreign SM120 export");
    }

    #[test]
    fn fixed_sm120_exact_n64_ptx_requires_unique_five_argument_exact_body() {
        let valid = format!(
            "{}{}{}",
            fixed_sm89_exact_n64_test_entry(FIXED_SM120_EXACT_N64_TEST_SYMBOL),
            fixed_sm120_copyplan_t256_test_entry(),
            fixed_sm120_copyplan_m128_test_entry()
        );
        super::validate_fixed_sm120_exact_n64_ptx("compute_120", &valid).unwrap();
        let nvrtc_pointer_qualified = valid.replace(".param .u64 ", ".param .u64 .ptr .align 1 ");
        super::validate_fixed_sm120_exact_n64_ptx("compute_120", &nvrtc_pointer_qualified)
            .expect("compute_120 NVRTC pointer qualifiers preserve the four-u64 ABI");
        for malformed in [
            String::new(),
            format!("{valid}{valid}"),
            fixed_sm89_exact_n64_test_entry("nn_sm120_f32_n64_copyplan_decoy"),
            valid.replacen(".param .align 4 .b8 params[32]", ".param .u64 params", 1),
            valid.replacen("fma.rn.f32", "add.rn.f32", 1),
            valid.replacen("cp.async.commit_group", "mov.u32", 1),
        ] {
            super::validate_fixed_sm120_exact_n64_ptx("compute_120", &malformed)
                .expect_err("missing, duplicate, foreign, ABI-drifted or reordered work must fail");
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_retains_all_other_fixed_composed_bytes() {
        let before = compose_fragments(super::FIXED_SOURCE_FRAGMENTS).unwrap();
        let ada = compose_module_source_for(ModuleKind::Fixed, "sm_89").unwrap();
        for arch in [
            "sm_80", "sm_86", "sm_87", "sm_90", "sm_90a", "sm_100a", "sm_103a", "sm_107a",
            "sm_110a",
        ] {
            assert_eq!(
                compose_module_source_for(ModuleKind::Fixed, arch).unwrap(),
                ada,
                "{arch}"
            );
        }
        for arch in ["compute_89", "sm_120", "sm_121", "compute_121"] {
            assert_eq!(
                compose_module_source_for(ModuleKind::Fixed, arch).unwrap(),
                before,
                "{arch}"
            );
        }
        let compute_120 = compose_module_source_for(ModuleKind::Fixed, "compute_120").unwrap();
        assert_eq!(
            compute_120.strip_prefix(&before).unwrap(),
            compose_fragments(&[
                super::FIXED_SM120_EXACT_N64_SOURCE_FRAGMENT,
                super::FIXED_SM120_SLICED_SOURCE_FRAGMENT,
                super::FIXED_SM120_POSTBIAS_SOURCE_FRAGMENT,
            ])
            .unwrap(),
            "the SM120 extension is the only intended compute_120 Fixed suffix"
        );
    }

    #[test]
    fn fixed_sm89_exact_n64_retains_every_triad_composition() {
        for arch in ["sm_89", "sm_120", "compute_120", "sm_121", "compute_121"] {
            for kind in [
                ModuleKind::TriadScalar,
                ModuleKind::TriadSm80,
                ModuleKind::TriadSm90a,
                ModuleKind::TriadSm100,
                ModuleKind::TriadSm120,
            ] {
                let mut retained = super::module_fragments(kind).unwrap().to_vec();
                if kind == ModuleKind::TriadSm80 && super::sm80_target_composes_streamk(arch) {
                    retained.extend([
                        super::SM80_STREAMK_SOURCE_FRAGMENT,
                        super::SM80_TF32_WIDE_SOURCE_FRAGMENT,
                        super::SM80_TN_SPLITK_SOURCE_FRAGMENT,
                    ]);
                }
                assert_eq!(
                    compose_module_source_for(kind, arch).unwrap(),
                    compose_fragments(&retained).unwrap(),
                    "{kind:?}/{arch}"
                );
            }
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_ptx_requires_its_unique_export() {
        validate_module_ptx(ModuleKind::Fixed, "sm_89", &fixed_sm89_exact_n64_test_ptx()).unwrap();
        let missing = fixed_sm89_exact_n64_test_ptx().replacen(
            &fixed_sm89_exact_n64_test_entry(FIXED_SM89_EXACT_N64_TEST_SYMBOL),
            "",
            1,
        );
        validate_module_ptx(ModuleKind::Fixed, "sm_89", &missing)
            .expect_err("Ada Fixed must contain the new exact N64 export");
        for extra in [
            FIXED_SM89_EXACT_N64_TEST_SYMBOL,
            "nn_sm89_f32_n64_copyplan_f16",
            "nn_sm89_f32_n64_copyplan_decoy",
        ] {
            let ptx = fixed_sm89_exact_n64_test_ptx() + &fixed_sm89_exact_n64_test_entry(extra);
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
                .expect_err("new exact N64 inventory is one unaliased F32 export");
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_ptx_rejects_wrong_pointer_and_bundle_abi() {
        let entry = fixed_sm89_exact_n64_test_entry(FIXED_SM89_EXACT_N64_TEST_SYMBOL);
        for malformed in [
            entry.replacen(".param .u64 a,", ".param .u32 a,", 1),
            entry.replacen("params[32]", "params[24]", 1),
            entry.replacen(".align 4 .b8 params", ".align 8 .b8 params", 1),
            entry.replacen("params[32]\n)", "params[32],\n.param .u64 scratch\n)", 1),
            entry.replacen(".param .align 4 .b8 params[32]",
                ".param .f32 alpha,\n.param .f32 beta,\n.param .u32 m,\n.param .u32 n,\n.param .u32 k,\n.param .u32 lda,\n.param .u32 ldb,\n.param .u32 ldc", 1),
        ] {
            let ptx = fixed_sm89_exact_n64_test_ptx().replacen(&entry, &malformed, 1);
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
                .expect_err("new exact N64 requires four pointers and one align4/size32 bundle");
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_ptx_requires_async_fma_and_launch_bounds() {
        let entry = fixed_sm89_exact_n64_test_entry(FIXED_SM89_EXACT_N64_TEST_SYMBOL);
        for (from, to) in [
            ("fma.rn.f32", "not_fma"),
            ("cp.async.cg.shared.global", "not_async"),
            ("cp.async.commit_group", "not_commit"),
            ("cp.async.wait_group", "not_wait"),
            (".maxntid 128, 1, 1", ".maxntid 256, 1, 1"),
            (".minnctapersm 2", ".minnctapersm 1"),
        ] {
            let malformed = entry.replacen(from, to, 1);
            let ptx = fixed_sm89_exact_n64_test_ptx().replacen(&entry, &malformed, 1);
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
                .expect_err("exact N64 pipeline and launch bounds are mandatory");
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_ptx_rejects_local_tensor_reduction_and_ftz_work() {
        let entry = fixed_sm89_exact_n64_test_entry(FIXED_SM89_EXACT_N64_TEST_SYMBOL);
        for instruction in [
            ".local .align 4 .b8 spill[16];",
            "ld.local.f32 %f0, [%rd0];",
            "st.local.f32 [%rd0], %f0;",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32;",
            "wmma.mma.sync.aligned.row.col.f32.f32;",
            "cvt.rna.tf32.f32 %r0, %f0;",
            "atom.global.add.f32 %f0, [%rd0], %f1;",
            "red.global.add.f32 [%rd0], %f1;",
            "redux.sync.add.s32 %r0, %r1, -1;",
            "fma.rn.ftz.f32 %f0, %f1, %f2, %f0;",
        ] {
            let malformed = entry.replacen("ret;", &format!("{instruction}\nret;"), 1);
            let ptx = fixed_sm89_exact_n64_test_ptx().replacen(&entry, &malformed, 1);
            validate_module_ptx(ModuleKind::Fixed, "sm_89", &ptx)
                .expect_err("exact N64 must not use local storage or a different numeric family");
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_ptx_travels_with_the_portable_overlay() {
        for arch in [
            "sm_80", "sm_86", "sm_87", "sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_110a",
        ] {
            let complete =
                fixed_sm89_half_test_ptx().replace(".target sm_89", &format!(".target {arch}"));
            validate_module_ptx(ModuleKind::Fixed, arch, &complete).unwrap();
            let entry = fixed_sm89_exact_n64_test_entry(FIXED_SM89_EXACT_N64_TEST_SYMBOL);
            validate_module_ptx(ModuleKind::Fixed, arch, &complete.replacen(&entry, "", 1))
                .expect_err("exact N64 is mandatory wherever the portable overlay is composed");
        }
        let before = fixed_sm89_half_test_base_ptx();
        validate_module_ptx(ModuleKind::Fixed, "compute_89", &before).unwrap();
        let ptx = before + &fixed_sm89_exact_n64_test_entry(FIXED_SM89_EXACT_N64_TEST_SYMBOL);
        validate_module_ptx(ModuleKind::Fixed, "compute_89", &ptx)
            .expect_err("exact N64 must not appear where the overlay is not composed");
    }

    #[test]
    fn fixed_sm89_exact_n64_driver_abi_requires_five_offsets_and_terminal_probe() {
        use cudarc::driver::sys::CUresult;
        let symbol = FIXED_SM89_EXACT_N64_TEST_SYMBOL;
        let layout = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
        let mut queries = Vec::new();
        let abi = super::query_driver_parameter_abi(symbol, 5, |index, offset, size| {
            queries.push(index);
            if let Some((parameter_offset, parameter_size)) = layout.get(index) {
                *offset = *parameter_offset;
                *size = *parameter_size;
                CUresult::CUDA_SUCCESS
            } else {
                CUresult::CUDA_ERROR_INVALID_VALUE
            }
        })
        .unwrap();
        assert_eq!(queries, [0, 1, 2, 3, 4, 5]);
        super::validate_fixed_sm89_exact_n64_driver_abi(symbol, &abi).unwrap();
        for layout in [
            vec![(0, 8), (8, 8), (16, 8), (24, 8)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 28)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (36, 32)],
            vec![(0, 4), (8, 8), (16, 8), (24, 8), (32, 32)],
            vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32), (64, 4)],
            // The unchanged standalone twelve-argument signature is not production ABI.
            vec![
                (0, 8),
                (8, 8),
                (16, 8),
                (24, 8),
                (32, 4),
                (36, 4),
                (40, 4),
                (44, 4),
                (48, 4),
                (52, 4),
                (56, 4),
                (60, 4),
            ],
        ] {
            let malformed = Tf32DriverAbi::checked(layout.len(), layout).unwrap();
            super::validate_fixed_sm89_exact_n64_driver_abi(symbol, &malformed).unwrap_err();
        }
        super::query_driver_parameter_abi(symbol, 5, |index, offset, size| {
            *offset = index * 8;
            *size = 8;
            CUresult::CUDA_SUCCESS
        })
        .expect_err("sixth successful Driver query must reject");
    }

    #[test]
    fn fixed_sm89_exact_n64_live_resource_gate_is_strict() {
        use super::{FixedSm89ExactN64Resources, validate_fixed_sm89_exact_n64_resources};
        let admitted = FixedSm89ExactN64Resources {
            local_bytes: 0,
            registers: 135,
            static_shared_bytes: 32_768,
            max_threads: 128,
            active_blocks: 3,
            preferred_carveout: 100,
        };
        validate_fixed_sm89_exact_n64_resources(admitted).unwrap();
        validate_fixed_sm89_exact_n64_resources(FixedSm89ExactN64Resources {
            registers: 160,
            ..admitted
        })
        .unwrap();
        for resources in [
            FixedSm89ExactN64Resources {
                local_bytes: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                local_bytes: 1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                registers: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                registers: 0,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                registers: 161,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                static_shared_bytes: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                static_shared_bytes: 32_767,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                static_shared_bytes: 32_769,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                max_threads: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                max_threads: 127,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                active_blocks: 0,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                active_blocks: 2,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                preferred_carveout: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                preferred_carveout: 0,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                preferred_carveout: 101,
                ..admitted
            },
        ] {
            let reason = validate_fixed_sm89_exact_n64_resources(resources).unwrap_err();
            assert!(reason.contains(FIXED_SM89_EXACT_N64_TEST_SYMBOL));
        }
    }

    #[test]
    fn fixed_sm120_sliced_live_resource_gate_is_strict() {
        use super::{FixedSm89ExactN64Resources, validate_fixed_sm120_exact_n64_resources};
        let admitted = FixedSm89ExactN64Resources {
            local_bytes: 0,
            registers: 113,
            static_shared_bytes: 32_768,
            max_threads: 128,
            active_blocks: 3,
            preferred_carveout: 100,
        };
        validate_fixed_sm120_exact_n64_resources(admitted).unwrap();
        validate_fixed_sm120_exact_n64_resources(FixedSm89ExactN64Resources {
            registers: 160,
            ..admitted
        })
        .unwrap();
        for resources in [
            FixedSm89ExactN64Resources {
                local_bytes: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                local_bytes: 1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                registers: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                registers: 0,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                registers: 161,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                static_shared_bytes: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                static_shared_bytes: 32_767,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                static_shared_bytes: 32_769,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                max_threads: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                max_threads: 127,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                active_blocks: 0,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                active_blocks: 2,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                preferred_carveout: -1,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                preferred_carveout: 0,
                ..admitted
            },
            FixedSm89ExactN64Resources {
                preferred_carveout: 101,
                ..admitted
            },
        ] {
            let reason = validate_fixed_sm120_exact_n64_resources(resources).unwrap_err();
            assert!(reason.contains(FIXED_SM120_EXACT_N64_TEST_SYMBOL));
        }
    }

    #[test]
    fn fixed_sm89_exact_n64_ptx_rejects_cc12_exports_without_nvrtc() {
        let entry = fixed_sm89_exact_n64_test_entry(FIXED_SM89_EXACT_N64_TEST_SYMBOL);
        for arch in ["sm_120", "compute_120", "sm_121", "compute_121"] {
            super::validate_fixed_sm89_exact_n64_ptx(arch, "").unwrap();
            super::validate_fixed_sm89_exact_n64_ptx(arch, &entry).unwrap_err();
        }
    }
    #[test]
    fn owned_symbol_resolution_never_falls_back_to_fixed() {
        let scalar_calls = std::cell::Cell::new(0);
        let sm80_calls = std::cell::Cell::new(0);
        let value: usize = resolve_owned_symbol(
            "nn_big",
            |name| {
                scalar_calls.set(scalar_calls.get() + 1);
                Ok(name.len())
            },
            |_| panic!("scalar symbol consulted the SM80 module"),
        )
        .unwrap();
        assert_eq!(value, "nn_big".len());
        assert_eq!(scalar_calls.get(), 1);
        assert_eq!(sm80_calls.get(), 0);

        let missing: Result<(), String> = resolve_owned_symbol(
            "nn_tc_bf16",
            |_| panic!("SM80 symbol consulted the scalar module"),
            |name| {
                sm80_calls.set(sm80_calls.get() + 1);
                Err(format!("absent {name}"))
            },
        );
        let error = missing.expect_err("missing owned symbol must abort initialization");
        assert!(error.contains("TriadSm80"), "{error}");
        assert!(error.contains("nn_tc_bf16"), "{error}");
        assert_eq!(sm80_calls.get(), 1);
    }
}
