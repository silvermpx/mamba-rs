use cudarc::driver::{CudaContext, CudaFunction};

use crate::mamba_ssm::gpu::kernel_identity::{ArtifactKind, FramedSha256, ModuleKind};

use super::{CompiledModule, Tf32DriverAbi};

pub(super) const HALF_BF16_SYMBOL: &str = "nn_sm89_tc128_f32out_s3_bf16";
pub(super) const HALF_F16_SYMBOL: &str = "nn_sm89_tc128_f32out_s3_f16";
pub(super) const EXACT_SYMBOL: &str = "nn_sm89_f32_m128n64_tail_copyplan";

pub(crate) struct InferenceSm89Bundle {
    pub(crate) half_f32out_s3_bf16: Result<CudaFunction, String>,
    pub(crate) half_f32out_s3_f16: Result<CudaFunction, String>,
    pub(crate) exact_m128n64_tail: Result<CudaFunction, String>,
}

impl InferenceSm89Bundle {
    fn rejected(reason: String) -> Self {
        Self {
            half_f32out_s3_bf16: Err(reason.clone()),
            half_f32out_s3_f16: Err(reason.clone()),
            exact_m128n64_tail: Err(reason),
        }
    }
}

pub(super) struct InferenceSm89DriverAbiCensus {
    pub(super) half_f32out_s3_bf16: Result<Tf32DriverAbi, String>,
    pub(super) half_f32out_s3_f16: Result<Tf32DriverAbi, String>,
    pub(super) exact_m128n64_tail: Result<Tf32DriverAbi, String>,
}

impl InferenceSm89DriverAbiCensus {
    pub(super) fn rejected(reason: String) -> Self {
        Self {
            half_f32out_s3_bf16: Err(reason.clone()),
            half_f32out_s3_f16: Err(reason.clone()),
            exact_m128n64_tail: Err(reason),
        }
    }
}

struct InferenceSm89PtxAdmission {
    half_f32out_s3_bf16: Result<(), String>,
    half_f32out_s3_f16: Result<(), String>,
    exact_m128n64_tail: Result<(), String>,
}

#[derive(Clone, Copy)]
struct InferenceSm89EnvelopeFacts<'a> {
    module_kind: ModuleKind,
    artifact_kind: ArtifactKind,
    compiler_output_kind: ArtifactKind,
    compile_key_matches: bool,
    target: &'a str,
    state_cap: usize,
    nvrtc: (i32, i32),
    nvrtc_library_known: bool,
    nvrtc_library_current: bool,
    device_cc: Option<(i32, i32)>,
}

#[derive(Clone, Copy)]
struct InferenceSm89ResourceFacts {
    local_bytes: i32,
    registers: i32,
    static_shared_bytes: i32,
    max_threads: i32,
    optin_shared_bytes: i32,
    active_blocks: u32,
}

impl InferenceSm89PtxAdmission {
    fn rejected(reason: String) -> Self {
        Self {
            half_f32out_s3_bf16: Err(reason.clone()),
            half_f32out_s3_f16: Err(reason.clone()),
            exact_m128n64_tail: Err(reason),
        }
    }
}

fn validate_inference_sm89_ptx(ptx: &str) -> InferenceSm89PtxAdmission {
    // The retained members are named outright. They used to carry a family
    // token that told them from the rest of the module, and a name prefix is
    // no longer able to draw that line: the module exports every kernel of
    // the family. Each member is addressed by its exact name, so a kernel
    // that is not one of them is served by the route that owns it and is not
    // this bundle's business; only a duplicated member is a real hazard.
    const MEMBERS: [&str; 3] = [HALF_BF16_SYMBOL, HALF_F16_SYMBOL, EXACT_SYMBOL];
    let parsed = match super::parse_ptx(ptx) {
        Ok(parsed) => parsed,
        Err(error) => {
            return InferenceSm89PtxAdmission::rejected(format!(
                "retained Inference PTX parse failed: {error}"
            ));
        }
    };
    let mut counts = std::collections::BTreeMap::new();
    for entry in parsed
        .entries
        .iter()
        .filter(|entry| MEMBERS.contains(&entry.symbol.as_str()))
    {
        *counts.entry(entry.symbol.as_str()).or_insert(0_usize) += 1;
    }
    let duplicate = counts
        .iter()
        .find_map(|(&symbol, &count)| (count != 1).then_some(symbol));
    if let Some(symbol) = duplicate {
        return InferenceSm89PtxAdmission::rejected(format!(
            "retained Inference PTX carries duplicate export {symbol}"
        ));
    }

    InferenceSm89PtxAdmission {
        half_f32out_s3_bf16: validate_half_entry(
            &parsed,
            HALF_BF16_SYMBOL,
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        ),
        half_f32out_s3_f16: validate_half_entry(
            &parsed,
            HALF_F16_SYMBOL,
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
        ),
        exact_m128n64_tail: validate_exact_entry(&parsed),
    }
}

fn validate_entry_header(entry: &super::ParsedPtxEntry, min_blocks: &str) -> Result<(), String> {
    let symbol = entry.symbol.as_str();
    let header = entry
        .text
        .split_once('{')
        .map(|(header, _)| header)
        .ok_or_else(|| format!("{symbol} has no PTX body"))?;
    let tokens = super::ptx_tokens(header);
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
            "{symbol} requires four u64 pointers and an align-4 32-byte bundle"
        ));
    }
    for (directive, expected) in [(".maxntid", "256"), (".minnctapersm", min_blocks)] {
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
        let valid = if directive == ".maxntid" {
            values == [expected] || values == [expected, ",", "1", ",", "1"]
        } else {
            values == [expected]
        };
        if !valid {
            return Err(format!("{symbol} has the wrong {directive} launch bound"));
        }
    }
    Ok(())
}

fn has_wait_group(body: &str, immediate: &str) -> bool {
    super::ptx_tokens(body)
        .windows(2)
        .any(|tokens| tokens[0].text == "cp.async.wait_group" && tokens[1].text == immediate)
}

fn has_async_copy(body: &str) -> bool {
    super::ptx_has_unquoted_token(body, |token| {
        ["cp.async.ca.shared.global", "cp.async.cg.shared.global"]
            .iter()
            .any(|family| {
                token == *family
                    || token
                        .strip_prefix(family)
                        .is_some_and(|suffix| suffix.starts_with('.'))
            })
    })
}

fn forbidden_common_token(token: &str) -> bool {
    token == ".local"
        || token.starts_with("ld.local")
        || token.starts_with("st.local")
        || token.starts_with("atom.")
        || token.starts_with("atom::")
        || token.starts_with("red.")
        || token.starts_with("red::")
        || token.starts_with("redux.")
        || token == "bar.red"
        || token.starts_with("bar.red.")
        || token == "call"
        || token.starts_with("call.")
        || token == ".callprototype"
        || token == ".calltargets"
}

fn validate_half_entry(
    parsed: &super::ParsedPtx,
    symbol: &str,
    expected_mma: &str,
) -> Result<(), String> {
    let entry = super::parsed_ptx_entry_ref(parsed, symbol)?;
    validate_entry_header(entry, "1")?;
    for required in [
        "cp.async.commit_group",
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16",
        expected_mma,
    ] {
        if !super::ptx_has_unquoted_token(&entry.body, |token| token == required) {
            return Err(format!("{symbol} is missing {required}"));
        }
    }
    if !has_async_copy(&entry.body)
        || !has_wait_group(&entry.body, "0")
        || !has_wait_group(&entry.body, "1")
        || !super::ptx_has_unquoted_token(&entry.body, |token| token == "bar.sync")
    {
        return Err(format!(
            "{symbol} has an incomplete async-copy or barrier protocol"
        ));
    }
    if super::ptx_has_unquoted_token(&entry.body, |token| {
        forbidden_common_token(token)
            || ((token.starts_with("mma.") || token.starts_with("wmma.")) && token != expected_mma)
            || token.starts_with("wgmma.")
            || token.starts_with("tcgen05.")
            || token.split('.').any(|part| part == "tf32" || part == "ftz")
    }) {
        return Err(format!(
            "{symbol} contains local, foreign tensor, FTZ, atomic, reduction, or call work"
        ));
    }
    Ok(())
}

fn validate_exact_entry(parsed: &super::ParsedPtx) -> Result<(), String> {
    let entry = super::parsed_ptx_entry_ref(parsed, EXACT_SYMBOL)?;
    validate_entry_header(entry, "2")?;
    for required in ["fma.rn.f32", "cp.async.commit_group"] {
        if !super::ptx_has_unquoted_token(&entry.body, |token| token == required) {
            return Err(format!("{EXACT_SYMBOL} is missing {required}"));
        }
    }
    if !has_async_copy(&entry.body) || !has_wait_group(&entry.body, "0") {
        return Err(format!(
            "{EXACT_SYMBOL} has an incomplete async-copy protocol"
        ));
    }
    if super::ptx_has_unquoted_token(&entry.body, |token| {
        forbidden_common_token(token)
            || token.starts_with("mma.")
            || token.starts_with("wmma.")
            || token.starts_with("wgmma.")
            || token.starts_with("tcgen05.")
            || token.split('.').any(|part| part == "tf32" || part == "ftz")
    }) {
        return Err(format!(
            "{EXACT_SYMBOL} contains local, tensor, FTZ, atomic, reduction, or call work"
        ));
    }
    Ok(())
}

fn validate_inference_sm89_driver_abi(
    symbol: &str,
    abi: &Tf32DriverAbi,
    host_size: usize,
    host_align: usize,
) -> Result<(), String> {
    const EXPECTED: [(usize, usize); 5] = [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)];
    if host_size != 32
        || host_align != 4
        || abi.parameter_count() != EXPECTED.len()
        || !abi
            .parameters()
            .iter()
            .zip(EXPECTED)
            .all(|(actual, expected)| (actual.offset(), actual.size()) == expected)
    {
        return Err(format!(
            "{symbol} has the wrong live five-argument/64-byte Driver ABI or host bundle layout"
        ));
    }
    Ok(())
}

fn validate_inference_sm89_envelope(facts: InferenceSm89EnvelopeFacts<'_>) -> Result<(), String> {
    if facts.module_kind != ModuleKind::Fixed
        || facts.artifact_kind != ArtifactKind::Ptx
        || facts.compiler_output_kind != ArtifactKind::Ptx
        || !facts.compile_key_matches
        || !facts.nvrtc_library_known
        || !facts.nvrtc_library_current
        || !crate::mamba_ssm::gpu::gemm_bi_inference::source_bundle::compiler_supported(
            facts.device_cc,
            facts.target,
            facts.state_cap,
            facts.nvrtc,
        )
    {
        return Err("retained Inference loader compiler/artifact envelope declined".into());
    }
    Ok(())
}

fn validate_inference_sm89_half_resources(
    symbol: &str,
    resources: InferenceSm89ResourceFacts,
) -> Result<(), String> {
    if resources.local_bytes != 0
        || !(1..=188).contains(&resources.registers)
        || resources.static_shared_bytes != 0
        || resources.max_threads < 256
        || resources.optin_shared_bytes < 98_304
        || resources.active_blocks < 1
    {
        return Err(format!(
            "{symbol} resource admission declined: local={} registers={} static_shared={} max_threads={} optin_shared={} active_blocks={}",
            resources.local_bytes,
            resources.registers,
            resources.static_shared_bytes,
            resources.max_threads,
            resources.optin_shared_bytes,
            resources.active_blocks,
        ));
    }
    Ok(())
}

fn validate_inference_sm89_exact_resources(
    resources: InferenceSm89ResourceFacts,
) -> Result<(), String> {
    if resources.local_bytes != 0
        || !(1..=128).contains(&resources.registers)
        || resources.static_shared_bytes != 49_152
        || resources.max_threads < 256
        || resources.active_blocks < 2
    {
        return Err(format!(
            "{EXACT_SYMBOL} resource admission declined: local={} registers={} static_shared={} max_threads={} active_blocks={}",
            resources.local_bytes,
            resources.registers,
            resources.static_shared_bytes,
            resources.max_threads,
            resources.active_blocks,
        ));
    }
    Ok(())
}

pub(super) fn census_inference_sm89_driver_abi(
    ctx: &CudaContext,
    kind: ModuleKind,
    arch: &str,
    ptx: &str,
) -> InferenceSm89DriverAbiCensus {
    if kind != ModuleKind::Fixed || !super::fixed_portable_overlay_composed(arch) {
        return InferenceSm89DriverAbiCensus::rejected(
            "retained Inference ABI census requires a Fixed module with the portable overlay"
                .into(),
        );
    }
    let admission = validate_inference_sm89_ptx(ptx);
    let module = match super::DriverModule::load(ctx, ptx) {
        Ok(module) => module,
        Err(error) => return InferenceSm89DriverAbiCensus::rejected(error),
    };
    type GetParamInfo = unsafe extern "C" fn(
        cudarc::driver::sys::CUfunction,
        usize,
        *mut usize,
        *mut usize,
    ) -> cudarc::driver::sys::CUresult;
    let get: GetParamInfo = match super::driver_proc_address("cuFuncGetParamInfo", 12_040) {
        Ok(get) => unsafe { std::mem::transmute::<*mut std::ffi::c_void, GetParamInfo>(get) },
        Err(error) => {
            let _ = module.unload();
            return InferenceSm89DriverAbiCensus::rejected(error);
        }
    };
    let query = |symbol: &'static str,
                 ptx_admission: Result<(), String>,
                 host_size: usize|
     -> Result<Tf32DriverAbi, String> {
        ptx_admission?;
        let name = std::ffi::CString::new(symbol)
            .map_err(|_| format!("retained Inference symbol {symbol} contains NUL"))?;
        let function = unsafe { cudarc::driver::result::module::get_function(module.raw(), name) }
            .map_err(|error| format!("load Fixed/{symbol} for Driver ABI: {error:?}"))?;
        let abi = super::query_driver_parameter_abi(symbol, 5, |index, offset, size| unsafe {
            get(function, index, offset, size)
        })?;
        validate_inference_sm89_driver_abi(symbol, &abi, host_size, 4)?;
        Ok(abi)
    };
    let census = InferenceSm89DriverAbiCensus {
        half_f32out_s3_bf16: query(
            HALF_BF16_SYMBOL,
            admission.half_f32out_s3_bf16,
            super::super::super::gemm_bi_inference::FIXED_SM89_HALF_PARAMS_SIZE,
        ),
        half_f32out_s3_f16: query(
            HALF_F16_SYMBOL,
            admission.half_f32out_s3_f16,
            super::super::super::gemm_bi_inference::FIXED_SM89_HALF_PARAMS_SIZE,
        ),
        exact_m128n64_tail: query(
            EXACT_SYMBOL,
            admission.exact_m128n64_tail,
            super::super::super::gemm_bi_inference::FIXED_SM89_EXACT_F32_PARAMS_SIZE,
        ),
    };
    if let Err(error) = module.unload() {
        return InferenceSm89DriverAbiCensus::rejected(format!(
            "unload retained Inference ABI census: {error}"
        ));
    }
    census
}

fn current_nvrtc_library_matches(module: &CompiledModule) -> bool {
    let current = crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain();
    nvrtc_library_identity_matches(
        module.compiler_identity.nvrtc_library_domain,
        current.as_deref(),
        crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain_is_current,
    )
}

fn nvrtc_library_identity_matches(
    expected_digest: [u8; 32],
    raw_domain: Option<&[u8]>,
    is_current: impl FnOnce(&[u8]) -> bool,
) -> bool {
    let Some(raw_domain) = raw_domain else {
        return false;
    };
    let digest = FramedSha256::new(b"nvrtc-library-set-identity.v2")
        .optional(b"domain", Some(raw_domain))
        .finish();
    digest == expected_digest && is_current(raw_domain)
}

fn load_half_member(
    ctx: &CudaContext,
    module: &CompiledModule,
    symbol: &'static str,
    abi: &Result<Tf32DriverAbi, String>,
) -> Result<CudaFunction, String> {
    validate_inference_sm89_driver_abi(
        symbol,
        abi.as_ref().map_err(Clone::clone)?,
        super::super::super::gemm_bi_inference::FIXED_SM89_HALF_PARAMS_SIZE,
        4,
    )?;
    let optin_shared_bytes = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .map_err(|error| format!("query {symbol} opt-in shared capacity: {error:?}"))?;
    let function = super::load_function(&module.module, ModuleKind::Fixed, symbol)?;
    super::set_dynamic_shared(&function, symbol, 98_304)?;
    set_max_shared_carveout(&function, symbol)?;
    let query_error = |label, error| format!("query {symbol} {label}: {error:?}");
    let resources = InferenceSm89ResourceFacts {
        local_bytes: function
            .local_size_bytes()
            .map_err(|error| query_error("local", error))?,
        registers: function
            .num_regs()
            .map_err(|error| query_error("registers", error))?,
        static_shared_bytes: function
            .shared_size_bytes()
            .map_err(|error| query_error("static shared", error))?,
        max_threads: function
            .max_threads_per_block()
            .map_err(|error| query_error("threads", error))?,
        optin_shared_bytes,
        active_blocks: function
            .occupancy_max_active_blocks_per_multiprocessor(256, 98_304, None)
            .map_err(|error| query_error("occupancy", error))?,
    };
    validate_inference_sm89_half_resources(symbol, resources)?;
    Ok(function)
}

fn load_exact_member(
    module: &CompiledModule,
    abi: &Result<Tf32DriverAbi, String>,
) -> Result<CudaFunction, String> {
    validate_inference_sm89_driver_abi(
        EXACT_SYMBOL,
        abi.as_ref().map_err(Clone::clone)?,
        super::super::super::gemm_bi_inference::FIXED_SM89_EXACT_F32_PARAMS_SIZE,
        4,
    )?;
    let function = super::load_function(&module.module, ModuleKind::Fixed, EXACT_SYMBOL)?;
    set_max_shared_carveout(&function, EXACT_SYMBOL)?;
    let query_error = |label, error| format!("query {EXACT_SYMBOL} {label}: {error:?}");
    let resources = InferenceSm89ResourceFacts {
        local_bytes: function
            .local_size_bytes()
            .map_err(|error| query_error("local", error))?,
        registers: function
            .num_regs()
            .map_err(|error| query_error("registers", error))?,
        static_shared_bytes: function
            .shared_size_bytes()
            .map_err(|error| query_error("static shared", error))?,
        max_threads: function
            .max_threads_per_block()
            .map_err(|error| query_error("threads", error))?,
        optin_shared_bytes: 0,
        active_blocks: function
            .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)
            .map_err(|error| query_error("occupancy", error))?,
    };
    validate_inference_sm89_exact_resources(resources)?;
    Ok(function)
}

fn set_max_shared_carveout(function: &CudaFunction, symbol: &str) -> Result<(), String> {
    function
        .set_attribute(
            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            cudarc::driver::sys::CUshared_carveout::CU_SHAREDMEM_CARVEOUT_MAX_SHARED as i32,
        )
        .map_err(|error| format!("configure {symbol} MaxShared: {error:?}"))
}

pub(crate) fn load_inference_sm89_bundle(
    ctx: &CudaContext,
    module: &CompiledModule,
    state_cap: usize,
) -> InferenceSm89Bundle {
    let device_cc = ctx.compute_capability().ok();
    let envelope = InferenceSm89EnvelopeFacts {
        module_kind: module.artifact_identity.module_kind,
        artifact_kind: module.artifact_identity.artifact_kind,
        compiler_output_kind: module.compiler_identity.output_kind,
        compile_key_matches: module.artifact_identity.compile_key
            == module.compiler_identity.invocation_digest,
        target: module.compiler_identity.target.as_str(),
        state_cap,
        nvrtc: module.compiler_identity.nvrtc_version,
        nvrtc_library_known: module.compiler_identity.nvrtc_library_known,
        nvrtc_library_current: current_nvrtc_library_matches(module),
        device_cc,
    };
    if let Err(reason) = validate_inference_sm89_envelope(envelope) {
        return InferenceSm89Bundle::rejected(reason);
    }
    InferenceSm89Bundle {
        half_f32out_s3_bf16: load_half_member(
            ctx,
            module,
            HALF_BF16_SYMBOL,
            &module.inference_sm89_driver_abi.half_f32out_s3_bf16,
        ),
        half_f32out_s3_f16: load_half_member(
            ctx,
            module,
            HALF_F16_SYMBOL,
            &module.inference_sm89_driver_abi.half_f32out_s3_f16,
        ),
        exact_m128n64_tail: load_exact_member(
            module,
            &module.inference_sm89_driver_abi.exact_m128n64_tail,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half_entry(symbol: &str, mma: &str) -> String {
        format!(
            ".visible .entry {symbol}(\n\
             .param .u64 p0, .param .u64 p1, .param .u64 p2, .param .u64 p3,\n\
             .param .align 4 .b8 params[32]\n\
             ) .maxntid 256, 1, 1 .minnctapersm 1 {{\n\
             cp.async.cg.shared.global [%r0], [%rd0], 16;\n\
             cp.async.commit_group;\n\
             cp.async.wait_group 1;\n\
             cp.async.wait_group 0;\n\
             bar.sync 0;\n\
             ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r0,%r1,%r2,%r3}}, [%r4];\n\
             ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{%r4,%r5}}, [%r6];\n\
             {mma} {{%f0,%f1,%f2,%f3}}, {{%r0,%r1,%r2,%r3}}, {{%r4,%r5}}, {{%f0,%f1,%f2,%f3}};\n\
             ret;\n\
             }}\n"
        )
    }

    fn exact_entry() -> String {
        format!(
            ".visible .entry {EXACT_SYMBOL}(\n\
             .param .u64 p0, .param .u64 p1, .param .u64 p2, .param .u64 p3,\n\
             .param .align 4 .b8 params[32]\n\
             ) .maxntid 256, 1, 1 .minnctapersm 2 {{\n\
             cp.async.cg.shared.global [%r0], [%rd0], 16;\n\
             cp.async.commit_group;\n\
             cp.async.wait_group 0;\n\
             bar.sync 0;\n\
             fma.rn.f32 %f0, %f1, %f2, %f0;\n\
             ret;\n\
             }}\n"
        )
    }

    fn valid_ptx() -> String {
        format!(
            ".version 8.4\n.target sm_89\n.address_size 64\n\
             .visible .entry nn_sm89_tc128_pipeline_bf16() .maxntid 32 {{ ret; }}\n{}{}{}",
            half_entry(
                HALF_BF16_SYMBOL,
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
            ),
            half_entry(
                HALF_F16_SYMBOL,
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            ),
            exact_entry(),
        )
    }

    fn assert_all_rejected(admission: InferenceSm89PtxAdmission) {
        assert!(admission.half_f32out_s3_bf16.is_err());
        assert!(admission.half_f32out_s3_f16.is_err());
        assert!(admission.exact_m128n64_tail.is_err());
    }

    #[test]
    fn inference_bundle_admission_accepts_literal_members_and_ignores_old_fixed_exports() {
        let admission = validate_inference_sm89_ptx(&valid_ptx());
        assert!(admission.half_f32out_s3_bf16.is_ok());
        assert!(admission.half_f32out_s3_f16.is_ok());
        assert!(admission.exact_m128n64_tail.is_ok());
    }

    #[test]
    fn inference_bundle_admission_missing_or_bad_member_preserves_valid_siblings() {
        let missing = valid_ptx().replace(
            &half_entry(
                HALF_F16_SYMBOL,
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            ),
            "",
        );
        let admission = validate_inference_sm89_ptx(&missing);
        assert!(admission.half_f32out_s3_bf16.is_ok());
        assert!(admission.half_f32out_s3_f16.is_err());
        assert!(admission.exact_m128n64_tail.is_ok());

        let contaminated = valid_ptx().replacen(
            "fma.rn.f32 %f0, %f1, %f2, %f0;",
            "fma.rn.f32 %f0, %f1, %f2, %f0;\nld.local.u32 %r0, [%rd0];",
            1,
        );
        let admission = validate_inference_sm89_ptx(&contaminated);
        assert!(admission.half_f32out_s3_bf16.is_ok());
        assert!(admission.half_f32out_s3_f16.is_ok());
        assert!(admission.exact_m128n64_tail.is_err());
    }

    #[test]
    fn inference_bundle_admission_rejects_duplicate_and_malformed_inventory_globally() {
        let duplicate = format!(
            "{}{}",
            valid_ptx(),
            half_entry(
                HALF_BF16_SYMBOL,
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
            )
        );
        assert_all_rejected(validate_inference_sm89_ptx(&duplicate));

        // A member exported under another name is a member that is missing:
        // the bundle addresses each one by its exact name, so the renamed
        // kernel is never launched and its siblings stay admissible.
        let renamed = valid_ptx().replace(HALF_BF16_SYMBOL, "nn_sm89_unreviewed_decoy_bf16");
        let admission = validate_inference_sm89_ptx(&renamed);
        assert!(admission.half_f32out_s3_bf16.is_err());
        assert!(admission.half_f32out_s3_f16.is_ok());
        assert!(admission.exact_m128n64_tail.is_ok());

        let malformed = format!("{}\n.visible .entry {HALF_BF16_SYMBOL}", valid_ptx());
        assert_all_rejected(validate_inference_sm89_ptx(&malformed));
    }

    #[test]
    fn inference_bundle_admission_does_not_accept_commented_or_quoted_fake_opcodes() {
        let fake = valid_ptx().replacen(
            "fma.rn.f32 %f0, %f1, %f2, %f0;",
            "mov.u32 %r0, %r1;\n// fma.rn.f32\n.file 1 \"fma.rn.f32\"",
            1,
        );
        let admission = validate_inference_sm89_ptx(&fake);
        assert!(admission.exact_m128n64_tail.is_err());

        let fake = valid_ptx()
            .replacen(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
                "mov.u32",
                1,
            )
            .replacen(
                "mov.u32 {%f0,%f1,%f2,%f3}",
                "// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\nmov.u32 {%f0,%f1,%f2,%f3}",
                1,
            );
        assert!(
            validate_inference_sm89_ptx(&fake)
                .half_f32out_s3_bf16
                .is_err()
        );
    }

    #[test]
    fn inference_bundle_admission_requires_real_global_to_shared_async_copies() {
        let half = valid_ptx().replacen(
            "cp.async.cg.shared.global [%r0], [%rd0], 16;",
            "cp.async.mbarrier.arrive.noinc.shared.b64 [%r0];",
            1,
        );
        let admission = validate_inference_sm89_ptx(&half);
        assert!(admission.half_f32out_s3_bf16.is_err());
        assert!(admission.half_f32out_s3_f16.is_ok());
        assert!(admission.exact_m128n64_tail.is_ok());

        let valid_exact = exact_entry();
        let invalid_exact = valid_exact.replacen(
            "cp.async.cg.shared.global [%r0], [%rd0], 16;",
            "cp.async.mbarrier.arrive.noinc.shared.b64 [%r0];",
            1,
        );
        let exact = valid_ptx().replace(&valid_exact, &invalid_exact);
        let admission = validate_inference_sm89_ptx(&exact);
        assert!(admission.half_f32out_s3_bf16.is_ok());
        assert!(admission.half_f32out_s3_f16.is_ok());
        assert!(admission.exact_m128n64_tail.is_err());
    }

    #[test]
    fn inference_bundle_admission_rejects_barrier_reductions_as_sync_evidence_or_work() {
        let half = valid_ptx().replacen("bar.sync 0;", "bar.red.popc.u32 %r0, 0, %p0;", 1);
        let admission = validate_inference_sm89_ptx(&half);
        assert!(admission.half_f32out_s3_bf16.is_err());
        assert!(admission.half_f32out_s3_f16.is_ok());
        assert!(admission.exact_m128n64_tail.is_ok());

        let valid_exact = exact_entry();
        let invalid_exact = valid_exact.replacen("bar.sync 0;", "bar.red.popc.u32 %r0, 0, %p0;", 1);
        let exact = valid_ptx().replace(&valid_exact, &invalid_exact);
        let admission = validate_inference_sm89_ptx(&exact);
        assert!(admission.half_f32out_s3_bf16.is_ok());
        assert!(admission.half_f32out_s3_f16.is_ok());
        assert!(admission.exact_m128n64_tail.is_err());
    }

    #[test]
    fn inference_bundle_admission_rejects_wrong_parameters_alignment_and_launch_bounds() {
        let cases = [
            valid_ptx().replacen(".param .u64 p3,", "", 1),
            valid_ptx().replacen(
                ".param .align 4 .b8 params[32]",
                ".param .align 8 .b8 params[32]",
                1,
            ),
            valid_ptx().replacen(".maxntid 256, 1, 1", ".maxntid 128, 1, 1", 1),
            valid_ptx().replacen(".minnctapersm 1", ".minnctapersm 2", 1),
            valid_ptx().replacen("params[32]", "params[28]", 1),
        ];
        for ptx in cases {
            assert!(
                validate_inference_sm89_ptx(&ptx)
                    .half_f32out_s3_bf16
                    .is_err()
            );
        }
        let exact = valid_ptx().replacen(".minnctapersm 2", ".minnctapersm 1", 1);
        assert!(
            validate_inference_sm89_ptx(&exact)
                .exact_m128n64_tail
                .is_err()
        );
    }

    #[test]
    fn inference_bundle_admission_rejects_local_tensor_ftz_atomic_and_call_contamination() {
        let exact_forbidden = [
            "ld.local.u32 %r0, [%rd0];",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0}, {%r0}, {%r1}, {%f0};",
            "fma.rn.ftz.f32 %f0, %f1, %f2, %f0;",
            "atom.global.add.u32 %r0, [%rd0], 1;",
            "red.global.add.u32 [%rd0], 1;",
            "call.uni helper, ();",
        ];
        for forbidden in exact_forbidden {
            let ptx = valid_ptx().replacen(
                "fma.rn.f32 %f0, %f1, %f2, %f0;",
                &format!("fma.rn.f32 %f0, %f1, %f2, %f0;\n{forbidden}"),
                1,
            );
            assert!(
                validate_inference_sm89_ptx(&ptx)
                    .exact_m128n64_tail
                    .is_err(),
                "{forbidden}"
            );
        }

        for forbidden in [
            ".local .align 4 .b8 spill[4];",
            "atom.global.add.u32 %r0, [%rd0], 1;",
            "redux.sync.add.u32 %r0, %r1, 0xffffffff;",
            "call.uni helper, ();",
        ] {
            let ptx = valid_ptx().replacen(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
                &format!("{forbidden}\nmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"),
                1,
            );
            assert!(
                validate_inference_sm89_ptx(&ptx)
                    .half_f32out_s3_bf16
                    .is_err(),
                "{forbidden}"
            );
        }
    }

    #[test]
    fn inference_bundle_admission_enforces_the_live_five_argument_driver_abi() {
        let valid =
            Tf32DriverAbi::checked(5, vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]).unwrap();
        assert!(validate_inference_sm89_driver_abi(HALF_BF16_SYMBOL, &valid, 32, 4).is_ok());

        let wrong_layout =
            Tf32DriverAbi::checked(5, vec![(0, 8), (8, 8), (16, 8), (24, 8), (32, 28)]).unwrap();
        assert!(
            validate_inference_sm89_driver_abi(HALF_BF16_SYMBOL, &wrong_layout, 32, 4).is_err()
        );
        assert!(validate_inference_sm89_driver_abi(HALF_BF16_SYMBOL, &valid, 28, 4).is_err());
        assert!(validate_inference_sm89_driver_abi(HALF_BF16_SYMBOL, &valid, 32, 8).is_err());
    }

    #[test]
    fn inference_bundle_admission_enforces_half_resource_boundaries() {
        let valid = InferenceSm89ResourceFacts {
            local_bytes: 0,
            registers: 188,
            static_shared_bytes: 0,
            max_threads: 256,
            optin_shared_bytes: 98_304,
            active_blocks: 1,
        };
        assert!(validate_inference_sm89_half_resources(HALF_BF16_SYMBOL, valid).is_ok());
        for invalid in [
            InferenceSm89ResourceFacts {
                local_bytes: 1,
                ..valid
            },
            InferenceSm89ResourceFacts {
                registers: 0,
                ..valid
            },
            InferenceSm89ResourceFacts {
                registers: 189,
                ..valid
            },
            InferenceSm89ResourceFacts {
                static_shared_bytes: 1,
                ..valid
            },
            InferenceSm89ResourceFacts {
                max_threads: 255,
                ..valid
            },
            InferenceSm89ResourceFacts {
                optin_shared_bytes: 98_303,
                ..valid
            },
            InferenceSm89ResourceFacts {
                active_blocks: 0,
                ..valid
            },
        ] {
            assert!(validate_inference_sm89_half_resources(HALF_BF16_SYMBOL, invalid).is_err());
        }
    }

    #[test]
    fn inference_bundle_admission_enforces_exact_resource_boundaries() {
        let valid = InferenceSm89ResourceFacts {
            local_bytes: 0,
            registers: 128,
            static_shared_bytes: 49_152,
            max_threads: 256,
            optin_shared_bytes: 0,
            active_blocks: 2,
        };
        assert!(validate_inference_sm89_exact_resources(valid).is_ok());
        for invalid in [
            InferenceSm89ResourceFacts {
                local_bytes: 1,
                ..valid
            },
            InferenceSm89ResourceFacts {
                registers: 0,
                ..valid
            },
            InferenceSm89ResourceFacts {
                registers: 129,
                ..valid
            },
            InferenceSm89ResourceFacts {
                static_shared_bytes: 49_151,
                ..valid
            },
            InferenceSm89ResourceFacts {
                max_threads: 255,
                ..valid
            },
            InferenceSm89ResourceFacts {
                active_blocks: 1,
                ..valid
            },
        ] {
            assert!(validate_inference_sm89_exact_resources(invalid).is_err());
        }
    }

    #[test]
    fn inference_bundle_admission_declines_outside_the_exact_compiler_envelope() {
        let valid = InferenceSm89EnvelopeFacts {
            module_kind: ModuleKind::Fixed,
            artifact_kind: ArtifactKind::Ptx,
            compiler_output_kind: ArtifactKind::Ptx,
            compile_key_matches: true,
            target: "sm_89",
            state_cap: 16,
            nvrtc: (12, 8),
            nvrtc_library_known: true,
            nvrtc_library_current: true,
            device_cc: Some((8, 9)),
        };
        assert!(validate_inference_sm89_envelope(valid).is_ok());
        assert!(
            validate_inference_sm89_envelope(InferenceSm89EnvelopeFacts {
                device_cc: Some((8, 6)),
                ..valid
            })
            .is_ok(),
            "an sm_80-tier board outside the frozen evidence takes the proof path"
        );
        for invalid in [
            InferenceSm89EnvelopeFacts {
                module_kind: ModuleKind::TriadSm80,
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                artifact_kind: ArtifactKind::Cubin,
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                compiler_output_kind: ArtifactKind::Cubin,
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                compile_key_matches: false,
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                target: "compute_89",
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                state_cap: 32,
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                nvrtc: (13, 1),
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                nvrtc_library_known: false,
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                nvrtc_library_current: false,
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                device_cc: Some((7, 5)),
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                device_cc: Some((12, 0)),
                ..valid
            },
            InferenceSm89EnvelopeFacts {
                device_cc: None,
                ..valid
            },
        ] {
            assert!(validate_inference_sm89_envelope(invalid).is_err());
        }
    }

    #[test]
    fn inference_bundle_admission_rechecks_nvrtc_library_freshness() {
        let raw_domain = b"literal-test-NVRTC-domain";
        let expected_digest = FramedSha256::new(b"nvrtc-library-set-identity.v2")
            .optional(b"domain", Some(raw_domain.as_slice()))
            .finish();
        let checks = std::cell::Cell::new(0);
        assert!(nvrtc_library_identity_matches(
            expected_digest,
            Some(raw_domain),
            |_| {
                checks.set(checks.get() + 1);
                true
            },
        ));
        assert_eq!(checks.get(), 1);

        assert!(!nvrtc_library_identity_matches(
            expected_digest,
            Some(raw_domain),
            |_| false,
        ));
        assert!(!nvrtc_library_identity_matches(
            expected_digest,
            None,
            |_| true,
        ));
        assert!(!nvrtc_library_identity_matches(
            [0x5a; 32],
            Some(raw_domain),
            |_| true,
        ));
    }
}
