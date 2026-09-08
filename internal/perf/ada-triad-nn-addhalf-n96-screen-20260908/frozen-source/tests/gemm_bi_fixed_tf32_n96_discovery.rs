#![cfg(feature = "cuda")]

use std::collections::BTreeSet;
use std::ffi::CStr;
use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, sys,
};
use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_fixed::{
    FixedFwdOperands, FixedShape, FixedTile, fixed_forward, fixed_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    NUMERIC_ABI_REVISION, SCHEDULE_REVISION, TUNING_TABLE_REVISION, digest_hex,
};
use sha2::{Digest as _, Sha256};

#[path = "support/triad_nn_n96_source.rs"]
mod triad_nn_n96_source;

const N96_BM: usize = 128;
const N96_BN: usize = 96;
const N96_BK: usize = 32;
const N96_STAGES: usize = 3;
const N96_SHARED: usize = 86_016;
const N96_SYMBOL: &str = "gemm_bi_nn_fixed_rna_tf32_exp_m128n96_bk32_s3";
const FAST_SYMBOL: &str =
    "_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4EEvNT_6ParamsE";
const CUDA_SOURCE: &str = include_str!("gemm_bi_fixed_tf32_n96_discovery.cu");
const GUARD: usize = 32;
const GUARD_BITS: u32 = 0x7fc0_96e0;
const POISON_BITS: u32 = 0x7fc0_96c0;

fn n96_stage_offsets(stage: usize) -> (usize, usize) {
    (stage * N96_BM * N96_BK * 4, stage * N96_BK * N96_BN * 4)
}

fn n96_dynamic_shared_bytes() -> usize {
    N96_STAGES * N96_BK * (N96_BM + N96_BN) * 4
}

fn n96_a_copy(thread: usize, slice: usize) -> (usize, usize) {
    let linear = thread + slice * 256;
    (linear / 8, (linear % 8) * 4)
}

fn n96_b_copy(thread: usize, slice: usize) -> (usize, usize) {
    let linear = thread + slice * 256;
    (linear / 24, (linear % 24) * 4)
}

fn n96_thread_outputs(warp: usize, lane: usize) -> Vec<(usize, usize)> {
    let warp_m = (warp >> 2) * 64;
    let warp_n = (warp & 3) * 24;
    let group = lane >> 2;
    let thread = lane & 3;
    let mut outputs = Vec::with_capacity(48);
    for m_atom in 0..4 {
        for n_atom in 0..3 {
            for element in 0..4 {
                outputs.push((
                    warp_m + m_atom * 16 + group + usize::from(element >= 2) * 8,
                    warp_n + n_atom * 8 + 2 * thread + (element & 1),
                ));
            }
        }
    }
    outputs
}

#[test]
fn n96_stage_offsets_keep_a_and_b_rings_asymmetric() {
    assert_eq!(n96_stage_offsets(0), (0, 0));
    assert_eq!(n96_stage_offsets(1), (16_384, 12_288));
    assert_eq!(n96_stage_offsets(2), (32_768, 24_576));
    assert_eq!(n96_dynamic_shared_bytes(), 86_016);
}

#[test]
fn n96_copy_plan_covers_each_a_and_b_vector_once() {
    let mut a = BTreeSet::new();
    for slice in 0..4 {
        for thread in 0..256 {
            let (row, k) = n96_a_copy(thread, slice);
            assert!(row < N96_BM);
            assert!(k < N96_BK);
            assert_eq!(k % 4, 0);
            assert!(a.insert((row, k)));
        }
    }
    assert_eq!(a.len(), 1_024);
    assert_eq!(a.first(), Some(&(0, 0)));
    assert_eq!(a.last(), Some(&(127, 28)));
    assert_eq!(n96_a_copy(7, 0), (0, 28));
    assert_eq!(n96_a_copy(8, 0), (1, 0));
    assert_eq!(n96_a_copy(255, 3), (127, 28));

    let mut b = BTreeSet::new();
    for slice in 0..3 {
        for thread in 0..256 {
            let (row, column) = n96_b_copy(thread, slice);
            assert!(row < N96_BK);
            assert!(column < N96_BN);
            assert_eq!(column % 4, 0);
            assert!(b.insert((row, column)));
        }
    }
    assert_eq!(b.len(), 768);
    assert_eq!(b.first(), Some(&(0, 0)));
    assert_eq!(b.last(), Some(&(31, 92)));
    assert_eq!(n96_b_copy(23, 0), (0, 92));
    assert_eq!(n96_b_copy(24, 0), (1, 0));
    assert_eq!(n96_b_copy(255, 0), (10, 60));
    assert_eq!(n96_b_copy(0, 1), (10, 64));
    assert_eq!(n96_b_copy(255, 2), (31, 92));
}

#[test]
fn n96_warps_partition_every_output_exactly_once() {
    let mut outputs = BTreeSet::new();
    for warp in 0..8 {
        for lane in 0..32 {
            let owned = n96_thread_outputs(warp, lane);
            assert_eq!(owned.len(), 48);
            for (row, column) in owned {
                assert!(row < N96_BM);
                assert!(column < N96_BN);
                assert!(
                    outputs.insert((row, column)),
                    "duplicate output {:?}",
                    (row, column)
                );
            }
        }
    }
    assert_eq!(outputs.len(), 128 * 96);
    assert_eq!(outputs.first(), Some(&(0, 0)));
    assert_eq!(outputs.last(), Some(&(127, 95)));
}

#[test]
#[ignore = "bounded CUDA13.2/CC8.9 N96 correctness; set MAMBA_FIXED_N96_DISCOVERY=1"]
fn n96_small_bits_graph_and_resources() {
    run_correctness().unwrap();
}

#[test]
#[ignore = "bounded CUDA13.2/CC8.9 N96 timing; set MAMBA_FIXED_N96_DISCOVERY=1"]
fn n96_e0_short_paired_discovery() {
    run_timing().unwrap();
}

#[derive(Clone, Copy)]
#[repr(C)]
struct N96Params {
    alpha: f32,
    beta: f32,
    m: i32,
    k: i32,
    n: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for N96Params {}

const _: () = {
    assert!(std::mem::size_of::<N96Params>() == 32);
    assert!(std::mem::align_of::<N96Params>() == 4);
};

#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    shape: FixedShape,
    bias: bool,
}

const E0: Case = Case {
    label: "e0",
    shape: FixedShape {
        m: 2048,
        k: 2304,
        n: 768,
    },
    bias: false,
};

const CORRECTNESS_CASES: [Case; 4] = [
    E0,
    Case {
        label: "e1",
        bias: true,
        ..E0
    },
    Case {
        label: "tail0",
        shape: FixedShape {
            m: 129,
            k: 36,
            n: 100,
        },
        bias: false,
    },
    Case {
        label: "tail1",
        shape: FixedShape {
            m: 129,
            k: 36,
            n: 100,
        },
        bias: true,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Candidate,
    ProductionRna,
    FastTf32,
}

impl Arm {
    const fn name(self) -> &'static str {
        match self {
            Self::Candidate => "candidate_n96",
            Self::ProductionRna => "production_rna_m128n128",
            Self::FastTf32 => "fast_tf32",
        }
    }
}

struct Runtime {
    _device: GpuDevice,
    ctx: GpuCtx,
    _module: Arc<CudaModule>,
    candidate: CudaFunction,
    candidate_source_sha: String,
    candidate_ptx_sha: String,
}

struct GuardedF32 {
    buffer: GpuBuffer,
    baseline: Vec<f32>,
    offset: usize,
    len: usize,
}

impl GuardedF32 {
    fn new(stream: &Arc<CudaStream>, active: Vec<f32>) -> Result<Self, String> {
        let len = active.len();
        let mut baseline = vec![f32::from_bits(GUARD_BITS); GUARD + len + GUARD];
        baseline[GUARD..GUARD + len].copy_from_slice(&active);
        Ok(Self {
            buffer: GpuBuffer::from_cpu(stream, &baseline)?,
            baseline,
            offset: GUARD,
            len,
        })
    }

    fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        self.buffer.raw_ptr_at(stream, self.offset)
    }

    fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
        self.buffer.upload(stream, &self.baseline)
    }

    fn all_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
        let values = self.buffer.to_cpu(stream)?;
        for (index, value) in values[..self.offset]
            .iter()
            .chain(&values[self.offset + self.len..])
            .enumerate()
        {
            if value.to_bits() != GUARD_BITS {
                return Err(format!("{label} changed red-zone element {index}"));
            }
        }
        Ok(values[self.offset..self.offset + self.len]
            .iter()
            .map(|value| value.to_bits())
            .collect())
    }

    fn unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
        let values = self.buffer.to_cpu(stream)?;
        if values
            .iter()
            .zip(&self.baseline)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err(format!("{label} input or guard changed"));
        }
        Ok(())
    }
}

struct Fixture {
    a: GuardedF32,
    b: GuardedF32,
    bias: GuardedF32,
    candidate: GuardedF32,
    production: GuardedF32,
    fast: GuardedF32,
}

impl Fixture {
    fn new(runtime: &Runtime, case: Case) -> Result<Self, String> {
        let shape = case.shape;
        let output_len = shape.m * shape.n;
        Ok(Self {
            a: GuardedF32::new(
                &runtime.ctx.stream,
                probe_values(shape.m * shape.k, 0xa096_0001),
            )?,
            b: GuardedF32::new(
                &runtime.ctx.stream,
                probe_values(shape.k * shape.n, 0xb096_0002),
            )?,
            bias: GuardedF32::new(&runtime.ctx.stream, probe_values(shape.n, 0xb1a5_0096))?,
            candidate: GuardedF32::new(
                &runtime.ctx.stream,
                vec![f32::from_bits(POISON_BITS); output_len],
            )?,
            production: GuardedF32::new(
                &runtime.ctx.stream,
                vec![f32::from_bits(POISON_BITS); output_len],
            )?,
            fast: GuardedF32::new(
                &runtime.ctx.stream,
                vec![f32::from_bits(POISON_BITS); output_len],
            )?,
        })
    }

    fn output(&self, arm: Arm) -> &GuardedF32 {
        match arm {
            Arm::Candidate => &self.candidate,
            Arm::ProductionRna => &self.production,
            Arm::FastTf32 => &self.fast,
        }
    }

    fn output_mut(&mut self, arm: Arm) -> &mut GuardedF32 {
        match arm {
            Arm::Candidate => &mut self.candidate,
            Arm::ProductionRna => &mut self.production,
            Arm::FastTf32 => &mut self.fast,
        }
    }

    fn operands(&self, runtime: &Runtime, case: Case, arm: Arm) -> FixedFwdOperands {
        let typed = |pointer| TypedPtr {
            ptr: pointer,
            dtype: WeightDtype::F32,
        };
        FixedFwdOperands {
            c: typed(self.output(arm).ptr(&runtime.ctx.stream)),
            x: typed(self.a.ptr(&runtime.ctx.stream)),
            w: typed(self.b.ptr(&runtime.ctx.stream)),
            bias_ptr: case.bias.then(|| self.bias.ptr(&runtime.ctx.stream)),
        }
    }

    fn reset(&mut self, runtime: &Runtime, arm: Arm) -> Result<(), String> {
        self.output_mut(arm).reset(&runtime.ctx.stream)
    }

    fn validate_inputs(&self, runtime: &Runtime) -> Result<(), String> {
        self.a.unchanged(&runtime.ctx.stream, "A")?;
        self.b.unchanged(&runtime.ctx.stream, "B")?;
        self.bias.unchanged(&runtime.ctx.stream, "bias")
    }
}

fn probe_values(len: usize, mut state: u64) -> Vec<f32> {
    let explicit = [
        0x0000_0000,
        0x8000_0000,
        0x3f80_1000,
        0xbf80_1000,
        0x0000_0001,
        0x8000_0001,
    ];
    (0..len)
        .map(|index| {
            if index < explicit.len() {
                return f32::from_bits(explicit[index]);
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let signed = ((state.wrapping_add(index as u64) % 2049) as i32) - 1024;
            signed as f32 / 1024.0
        })
        .collect()
}

fn compose_source() -> String {
    let prelude = include_str!("../kernels/_typed_prelude.cuh");
    let common = include_str!("../kernels/gemm_bi_fixed/common.cuh")
        .lines()
        .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
        .collect::<Vec<_>>()
        .join("\n");
    let tf32 = include_str!("../kernels/gemm_bi_fixed/tf32.cu");
    [prelude, &common, tf32, CUDA_SOURCE].join("\n")
}

fn new_runtime() -> Result<Runtime, String> {
    if std::env::var("MAMBA_FIXED_N96_DISCOVERY").as_deref() != Ok("1") {
        return Err("set MAMBA_FIXED_N96_DISCOVERY=1".into());
    }
    let device = GpuDevice::new(0)?;
    if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
        return Err(format!(
            "N96 discovery requires the 142-SM Ada lane, found {:?}/{}",
            device.compute_capability,
            device.multiprocessor_count()
        ));
    }
    let ctx = GpuCtx::new(&device)?;
    let compiler = ctx.kernels.compiler_identity();
    let artifact = ctx.kernels.artifact_set_identity().fixed;
    if compiler.nvrtc_version != (13, 2)
        || compiler.target.as_str() != "sm_89"
        || !compiler.nvrtc_library_known
        || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
        || compiler.schedule_revision != SCHEDULE_REVISION
        || TUNING_TABLE_REVISION != 44
        || NUMERIC_ABI_REVISION != 5
        || SCHEDULE_REVISION != 8
        || artifact.compile_key != compiler.invocation_digest
    {
        return Err(format!(
            "N96 discovery lost accepted CUDA13.2 Fixed identity: {compiler:?} {artifact:?}"
        ));
    }
    let source = compose_source();
    let candidate_source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
    let options = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_89"),
        options: vec![
            "--fmad=true".into(),
            "--extra-device-vectorization".into(),
            "-DNDEBUG".into(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, options)
        .map_err(|error| format!("compile N96 discovery: {error:?}"))?;
    let ptx_source = ptx.to_src();
    let candidate_ptx_sha = format!("{:x}", Sha256::digest(ptx_source.as_bytes()));
    let module = device
        .context()
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
        .map_err(|error| format!("load N96 discovery: {error:?}"))?;
    let candidate = module
        .load_function(N96_SYMBOL)
        .map_err(|error| format!("load {N96_SYMBOL}: {error:?}"))?;
    candidate
        .set_attribute(
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            N96_SHARED as i32,
        )
        .map_err(|error| format!("set N96 dynamic shared: {error:?}"))?;
    let runtime = Runtime {
        _device: device,
        ctx,
        _module: module,
        candidate,
        candidate_source_sha,
        candidate_ptx_sha,
    };
    validate_resources(&runtime)?;
    println!(
        concat!(
            "{{\"schema\":\"MambaBiFixedTf32N96DiscoveryIdentityV1\",",
            "\"cc\":\"8.9\",\"sm_count\":142,\"nvrtc\":[13,2],",
            "\"tuning_revision\":44,\"numeric_revision\":5,\"schedule_revision\":8,",
            "\"fixed_source_sha\":\"{}\",\"fixed_invocation_sha\":\"{}\",",
            "\"fixed_artifact_sha\":\"{}\",\"candidate_source_sha\":\"{}\",",
            "\"candidate_ptx_sha\":\"{}\"}}"
        ),
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        digest_hex(&artifact.artifact_digest),
        runtime.candidate_source_sha,
        runtime.candidate_ptx_sha,
    );
    Ok(runtime)
}

fn validate_resources(runtime: &Runtime) -> Result<(), String> {
    let registers = runtime
        .candidate
        .num_regs()
        .map_err(|error| format!("query N96 registers: {error:?}"))?;
    let local = runtime
        .candidate
        .local_size_bytes()
        .map_err(|error| format!("query N96 local bytes: {error:?}"))?;
    let static_shared = runtime
        .candidate
        .shared_size_bytes()
        .map_err(|error| format!("query N96 static shared: {error:?}"))?;
    let max_dynamic = runtime
        .candidate
        .get_attribute(
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        )
        .map_err(|error| format!("query N96 maximum dynamic shared: {error:?}"))?;
    let occupancy = runtime
        .candidate
        .occupancy_max_active_blocks_per_multiprocessor(256, N96_SHARED, None)
        .map_err(|error| format!("query N96 occupancy: {error:?}"))?;
    if registers <= 0
        || registers > 255
        || local != 0
        || static_shared != 0
        || max_dynamic < N96_SHARED as i32
        || occupancy != 1
    {
        return Err(format!(
            "N96 resource gate failed: regs={registers} local={local} static={static_shared} max_dynamic={max_dynamic} occupancy={occupancy}"
        ));
    }
    println!(
        "{{\"schema\":\"MambaBiFixedTf32N96DiscoveryResourceV1\",\"symbol\":\"{N96_SYMBOL}\",\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":{N96_SHARED},\"occupancy_blocks_per_sm\":{occupancy}}}"
    );
    Ok(())
}

fn configure(runtime: &Runtime, arm: Arm) -> Result<(), String> {
    runtime.ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    runtime.ctx.set_bi_tensor_cores(false);
    runtime
        .ctx
        .set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
    match arm {
        Arm::Candidate | Arm::ProductionRna => {
            runtime.ctx.set_batch_invariant(true);
            runtime.ctx.set_fast_gemm(false);
        }
        Arm::FastTf32 => {
            runtime.ctx.set_batch_invariant(false);
            runtime.ctx.set_fast_gemm(true);
            if !runtime.ctx.tf32() {
                return Err("FAST_TF32 comparator is disabled".into());
            }
        }
    }
    Ok(())
}

fn candidate_config(shape: FixedShape) -> Result<LaunchConfig, String> {
    let blocks = shape.m.div_ceil(N96_BM) * shape.n.div_ceil(N96_BN);
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(blocks).map_err(|_| "N96 grid exceeds u32")?,
            1,
            1,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: N96_SHARED as u32,
    })
}

fn launch(runtime: &Runtime, fixture: &Fixture, case: Case, arm: Arm) -> Result<(), String> {
    configure(runtime, arm)?;
    let operands = fixture.operands(runtime, case, arm);
    match arm {
        Arm::Candidate => {
            let output = operands.c.ptr;
            let a = operands.x.ptr;
            let b = operands.w.ptr;
            let bias = operands.bias_ptr.unwrap_or(0);
            let params = N96Params {
                alpha: 1.0,
                beta: 0.0,
                m: i32::try_from(case.shape.m).map_err(|_| "M exceeds i32")?,
                k: i32::try_from(case.shape.k).map_err(|_| "K exceeds i32")?,
                n: i32::try_from(case.shape.n).map_err(|_| "N exceeds i32")?,
                lda: i32::try_from(case.shape.k).map_err(|_| "lda exceeds i32")?,
                ldb: i32::try_from(case.shape.n).map_err(|_| "ldb exceeds i32")?,
                ldc: i32::try_from(case.shape.n).map_err(|_| "ldc exceeds i32")?,
            };
            let mut builder = runtime.ctx.stream.launch_builder(&runtime.candidate);
            builder.arg(&output);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&bias);
            builder.arg(&params);
            unsafe { builder.launch(candidate_config(case.shape)?) }
                .map(|_| ())
                .map_err(|error| format!("launch N96 candidate: {error:?}"))
        }
        Arm::ProductionRna => fixed_forward_with_tile(
            &runtime.ctx,
            operands,
            case.shape,
            FixedTile::Tf32RnaM128N128S3,
        ),
        Arm::FastTf32 => gpu_gemm_typed_forward_raw(
            &runtime.ctx,
            operands.c,
            operands.x,
            operands.w,
            operands.bias_ptr,
            (case.shape.m, case.shape.k, case.shape.n),
        ),
    }
}

fn capture(
    runtime: &Runtime,
    fixture: &Fixture,
    case: Case,
    arm: Arm,
) -> Result<CudaGraph, String> {
    unsafe { capture_into_graph(&runtime.ctx.stream, || launch(runtime, fixture, case, arm)) }
}

fn output_bits(
    runtime: &Runtime,
    fixture: &Fixture,
    arm: Arm,
    label: &str,
) -> Result<Vec<u32>, String> {
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|error| format!("synchronize {label}: {error:?}"))?;
    fixture.output(arm).all_bits(&runtime.ctx.stream, label)
}

fn assert_n96_graph(
    graph: &CudaGraph,
    shape: FixedShape,
    expected_symbol: &str,
) -> Result<(), String> {
    assert_single_tf32_graph(graph, shape, expected_symbol, N96_BM, N96_BN, N96_SHARED)
}

fn assert_single_tf32_graph(
    graph: &CudaGraph,
    shape: FixedShape,
    expected_symbol: &str,
    tile_m: usize,
    tile_n: usize,
    shared_bytes: usize,
) -> Result<(), String> {
    let mut count = 0usize;
    if unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) }
        != sys::CUresult::CUDA_SUCCESS
        || count != 1
    {
        return Err(format!("N96 graph node count {count}, expected one"));
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    if unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count) }
        != sys::CUresult::CUDA_SUCCESS
    {
        return Err("query N96 graph nodes failed".into());
    }
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    if unsafe { sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) }
        != sys::CUresult::CUDA_SUCCESS
    {
        return Err("query N96 graph parameters failed".into());
    }
    let mut name = std::ptr::null();
    if unsafe { sys::cuFuncGetName(&mut name, params.func) } != sys::CUresult::CUDA_SUCCESS
        || name.is_null()
    {
        return Err("query N96 graph symbol failed".into());
    }
    let symbol = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|error| format!("N96 symbol UTF-8: {error}"))?;
    let expected_grid = shape.m.div_ceil(tile_m) * shape.n.div_ceil(tile_n);
    if symbol != expected_symbol
        || (params.gridDimX, params.gridDimY, params.gridDimZ) != (expected_grid as u32, 1, 1)
        || (params.blockDimX, params.blockDimY, params.blockDimZ) != (256, 1, 1)
        || params.sharedMemBytes != shared_bytes as u32
    {
        return Err(format!(
            "N96 physical mismatch symbol={symbol} grid={:?} block={:?} shared={}",
            (params.gridDimX, params.gridDimY, params.gridDimZ),
            (params.blockDimX, params.blockDimY, params.blockDimZ),
            params.sharedMemBytes
        ));
    }
    for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
        .into_iter()
        .enumerate()
    {
        let mut offset = 0;
        let mut size = 0;
        if unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) }
            != sys::CUresult::CUDA_SUCCESS
            || (offset, size) != expected
        {
            return Err(format!("N96 ABI parameter {index}: {:?}", (offset, size)));
        }
    }
    let mut offset = 0;
    let mut size = 0;
    if unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) }
        != sys::CUresult::CUDA_ERROR_INVALID_VALUE
    {
        return Err("N96 ABI accepted a sixth argument".into());
    }
    Ok(())
}

fn run_arm_bits(
    runtime: &Runtime,
    fixture: &mut Fixture,
    case: Case,
    arm: Arm,
) -> Result<Vec<u32>, String> {
    fixture.reset(runtime, arm)?;
    launch(runtime, fixture, case, arm)?;
    let bits = output_bits(runtime, fixture, arm, arm.name())?;
    if bits.iter().any(|word| *word == POISON_BITS) {
        return Err(format!("{} retained poison", arm.name()));
    }
    Ok(bits)
}

fn check_case(runtime: &Runtime, case: Case) -> Result<Vec<u32>, String> {
    let mut fixture = Fixture::new(runtime, case)?;
    let production = run_arm_bits(runtime, &mut fixture, case, Arm::ProductionRna)?;
    let candidate = run_arm_bits(runtime, &mut fixture, case, Arm::Candidate)?;
    if candidate != production {
        return Err(format!(
            "{} candidate bits differ from production RNA",
            case.label
        ));
    }
    for repeat in 0..3 {
        for arm in [Arm::ProductionRna, Arm::Candidate] {
            if run_arm_bits(runtime, &mut fixture, case, arm)? != production {
                return Err(format!(
                    "{} {} eager repeat {repeat} changed bits",
                    case.label,
                    arm.name()
                ));
            }
        }
    }
    let production_graph = capture(runtime, &fixture, case, Arm::ProductionRna)?;
    let candidate_graph = capture(runtime, &fixture, case, Arm::Candidate)?;
    assert_n96_graph(&candidate_graph, case.shape, N96_SYMBOL)?;
    for replay in 0..3 {
        for (arm, graph) in [
            (Arm::ProductionRna, &production_graph),
            (Arm::Candidate, &candidate_graph),
        ] {
            fixture.reset(runtime, arm)?;
            graph
                .launch()
                .map_err(|error| format!("{} graph replay: {error:?}", arm.name()))?;
            if output_bits(runtime, &fixture, arm, arm.name())? != production {
                return Err(format!(
                    "{} {} graph replay {replay} changed bits",
                    case.label,
                    arm.name()
                ));
            }
        }
    }
    fixture.validate_inputs(runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedTf32N96DiscoveryBitsV1\",\"case\":\"{}\",\"shape\":[{},{},{}],\"bias\":{},\"eager_repeats\":3,\"graph_repeats\":3,\"passed\":true}}",
        case.label, case.shape.m, case.shape.k, case.shape.n, case.bias
    );
    Ok(production)
}

fn check_e_prefix(runtime: &Runtime, full: &[u32]) -> Result<(), String> {
    let case = Case {
        label: "e_prefix_m1",
        shape: FixedShape { m: 1, ..E0.shape },
        bias: false,
    };
    let mut fixture = Fixture::new(runtime, E0)?;
    let expected = &full[..case.shape.n];
    for arm in [Arm::ProductionRna, Arm::Candidate] {
        fixture.reset(runtime, arm)?;
        launch(runtime, &fixture, case, arm)?;
        let bits = output_bits(runtime, &fixture, arm, arm.name())?;
        if &bits[..case.shape.n] != expected
            || bits[case.shape.n..].iter().any(|word| *word != POISON_BITS)
        {
            return Err(format!("{} prefix or suffix mismatch", arm.name()));
        }
        let graph = capture(runtime, &fixture, case, arm)?;
        if arm == Arm::Candidate {
            assert_n96_graph(&graph, case.shape, N96_SYMBOL)?;
        }
        for replay in 0..3 {
            fixture.reset(runtime, arm)?;
            graph
                .launch()
                .map_err(|error| format!("{} prefix graph: {error:?}", arm.name()))?;
            let bits = output_bits(runtime, &fixture, arm, arm.name())?;
            if &bits[..case.shape.n] != expected
                || bits[case.shape.n..].iter().any(|word| *word != POISON_BITS)
            {
                return Err(format!(
                    "{} prefix graph replay {replay} mismatch",
                    arm.name()
                ));
            }
        }
    }
    fixture.validate_inputs(runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedTf32N96DiscoveryPrefixV1\",\"full_m\":2048,\"prefix_m\":1,\"k\":2304,\"n\":768,\"graph_repeats\":3,\"passed\":true}}"
    );
    Ok(())
}

fn assert_actual_auto(runtime: &Runtime, case: Case) -> Result<(), String> {
    let mut fixture = Fixture::new(runtime, case)?;
    fixture.reset(runtime, Arm::ProductionRna)?;
    configure(runtime, Arm::ProductionRna)?;
    let operands = fixture.operands(runtime, case, Arm::ProductionRna);
    let selected = fixed_forward(
        &runtime.ctx,
        operands.c,
        operands.x,
        operands.w,
        operands.bias_ptr,
        (case.shape.m, case.shape.k, case.shape.n),
    )?;
    if selected != FixedTile::Tf32RnaM128N128S3 {
        return Err(format!("E AUTO selected {selected:?}"));
    }
    output_bits(runtime, &fixture, Arm::ProductionRna, "actual AUTO")?;
    Ok(())
}

fn run_correctness() -> Result<(), String> {
    let runtime = new_runtime()?;
    assert_actual_auto(&runtime, E0)?;
    let e0_bits = check_case(&runtime, E0)?;
    check_e_prefix(&runtime, &e0_bits)?;
    for case in &CORRECTNESS_CASES[1..] {
        check_case(&runtime, *case)?;
    }
    println!(
        "{{\"schema\":\"MambaBiFixedTf32N96DiscoveryCorrectnessCompleteV1\",\"cases\":5,\"passed\":true}}"
    );
    Ok(())
}

fn measure_window(
    runtime: &Runtime,
    fixture: &Fixture,
    case: Case,
    arm: Arm,
    iterations: usize,
) -> Result<f64, String> {
    let start = runtime
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("record {} start: {error:?}", arm.name()))?;
    for _ in 0..iterations {
        launch(runtime, fixture, case, arm)?;
    }
    let end = runtime
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("record {} end: {error:?}", arm.name()))?;
    let us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|error| format!("measure {}: {error:?}", arm.name()))?,
    ) * 1000.0
        / iterations as f64;
    if !us.is_finite() || us <= 0.0 {
        return Err(format!("invalid {} sample {us}", arm.name()));
    }
    Ok(us)
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * quantile).round() as usize]
}

fn timed_comparison(
    runtime: &Runtime,
    fixture: &Fixture,
    comparator: Arm,
    windows: usize,
) -> Result<(f64, f64), String> {
    for _ in 0..128 {
        launch(runtime, fixture, E0, Arm::Candidate)?;
        launch(runtime, fixture, E0, comparator)?;
    }
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{} warmup: {error:?}", comparator.name()))?;
    let candidate_pilot = measure_window(runtime, fixture, E0, Arm::Candidate, 16)?;
    let comparator_pilot = measure_window(runtime, fixture, E0, comparator, 16)?;
    let iterations = |pilot: f64| (5000.0 / pilot).round().clamp(1.0, 4096.0) as usize;
    let candidate_iterations = iterations(candidate_pilot);
    let comparator_iterations = iterations(comparator_pilot);
    let mut all_ratios = Vec::with_capacity(windows * 2);
    for (order, candidate_first) in [("abba", true), ("baab", false)] {
        let mut candidate_samples = Vec::with_capacity(windows);
        let mut comparator_samples = Vec::with_capacity(windows);
        let mut ratios = Vec::with_capacity(windows);
        for _ in 0..windows {
            let (
                candidate_first_us,
                comparator_first_us,
                comparator_second_us,
                candidate_second_us,
            ) = if candidate_first {
                (
                    measure_window(runtime, fixture, E0, Arm::Candidate, candidate_iterations)?,
                    measure_window(runtime, fixture, E0, comparator, comparator_iterations)?,
                    measure_window(runtime, fixture, E0, comparator, comparator_iterations)?,
                    measure_window(runtime, fixture, E0, Arm::Candidate, candidate_iterations)?,
                )
            } else {
                let comparator_first_us =
                    measure_window(runtime, fixture, E0, comparator, comparator_iterations)?;
                let candidate_first_us =
                    measure_window(runtime, fixture, E0, Arm::Candidate, candidate_iterations)?;
                let candidate_second_us =
                    measure_window(runtime, fixture, E0, Arm::Candidate, candidate_iterations)?;
                let comparator_second_us =
                    measure_window(runtime, fixture, E0, comparator, comparator_iterations)?;
                (
                    candidate_first_us,
                    comparator_first_us,
                    comparator_second_us,
                    candidate_second_us,
                )
            };
            let candidate_us = (candidate_first_us + candidate_second_us) * 0.5;
            let comparator_us = (comparator_first_us + comparator_second_us) * 0.5;
            let ratio = candidate_us / comparator_us;
            candidate_samples.push(candidate_us);
            comparator_samples.push(comparator_us);
            ratios.push(ratio);
            all_ratios.push(ratio);
        }
        println!(
            "{{\"schema\":\"MambaBiFixedTf32N96DiscoveryTimingV1\",\"cell\":\"e0\",\"shape\":[2048,2304,768],\"bias\":false,\"candidate\":\"candidate_n96\",\"comparator\":\"{}\",\"order\":\"{}\",\"windows\":{},\"candidate_iterations\":{},\"comparator_iterations\":{},\"candidate_p50_us\":{},\"candidate_p95_us\":{},\"comparator_p50_us\":{},\"comparator_p95_us\":{},\"ratio_p50\":{},\"ratio_p95\":{},\"candidate_samples_us\":{:?},\"comparator_samples_us\":{:?},\"ratios\":{:?}}}",
            comparator.name(),
            order,
            windows,
            candidate_iterations,
            comparator_iterations,
            percentile(&candidate_samples, 0.5),
            percentile(&candidate_samples, 0.95),
            percentile(&comparator_samples, 0.5),
            percentile(&comparator_samples, 0.95),
            percentile(&ratios, 0.5),
            percentile(&ratios, 0.95),
            candidate_samples,
            comparator_samples,
            ratios,
        );
    }
    Ok((percentile(&all_ratios, 0.5), percentile(&all_ratios, 0.95)))
}

fn validate_fast_graph(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
    let graph = capture(runtime, fixture, E0, Arm::FastTf32)?;
    let mut count = 0usize;
    unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) };
    if count != 1 {
        return Err(format!("Fast graph has {count} nodes"));
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count) };
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    unsafe { sys::cuGraphKernelNodeGetParams_v2(nodes[0], &mut params) };
    let mut name = std::ptr::null();
    unsafe { sys::cuFuncGetName(&mut name, params.func) };
    if name.is_null() {
        return Err("Fast graph symbol is null".into());
    }
    let symbol = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|error| format!("Fast symbol UTF-8: {error}"))?;
    if symbol != FAST_SYMBOL {
        return Err(format!("unexpected Fast TF32 symbol {symbol}"));
    }
    println!(
        "{{\"schema\":\"MambaBiFixedTf32N96DiscoveryFastPhysicalV1\",\"symbol\":\"{symbol}\",\"grid\":[{},{},{}],\"block\":[{},{},{}],\"shared_bytes\":{}}}",
        params.gridDimX,
        params.gridDimY,
        params.gridDimZ,
        params.blockDimX,
        params.blockDimY,
        params.blockDimZ,
        params.sharedMemBytes,
    );
    Ok(())
}

fn run_timing() -> Result<(), String> {
    let windows = match std::env::var("MAMBA_FIXED_N96_WINDOWS") {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|error| format!("parse MAMBA_FIXED_N96_WINDOWS: {error}"))?,
        Err(std::env::VarError::NotPresent) => 7,
        Err(error) => return Err(format!("read MAMBA_FIXED_N96_WINDOWS: {error}")),
    };
    if !matches!(windows, 7 | 21) {
        return Err(format!("N96 windows must be 7 or 21, got {windows}"));
    }
    let runtime = new_runtime()?;
    assert_actual_auto(&runtime, E0)?;
    let fixture = Fixture::new(&runtime, E0)?;
    validate_fast_graph(&runtime, &fixture)?;
    let own = timed_comparison(&runtime, &fixture, Arm::ProductionRna, windows)?;
    let fast = timed_comparison(&runtime, &fixture, Arm::FastTf32, windows)?;
    fixture.validate_inputs(&runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedTf32N96DiscoveryCompleteV1\",\"windows\":{windows},\"own_ratio_p50\":{},\"own_ratio_p95\":{},\"fast_ratio_p50\":{},\"fast_ratio_p95\":{},\"own_win\":{},\"passed\":true}}",
        own.0,
        own.1,
        fast.0,
        fast.1,
        own.0 < 1.0 && own.1 < 1.0,
    );
    Ok(())
}

mod triad_nn_add_half_screen {
    use super::*;

    const ENV: &str = "MAMBA_TRIAD_NN_N96_DISCOVERY";
    const CURRENT_WIDE_SYMBOL: &str = "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3";
    const TARGET: Case = Case {
        label: "triad_nn_d768_out",
        shape: FixedShape {
            m: 2_048,
            k: 1_536,
            n: 768,
        },
        bias: false,
    };
    const TAIL: Case = Case {
        label: "triad_nn_n96_tail",
        shape: FixedShape {
            m: 129,
            k: 36,
            n: 100,
        },
        bias: false,
    };
    const K0: Case = Case {
        label: "triad_nn_n96_k0",
        shape: FixedShape {
            m: 129,
            k: 0,
            n: 100,
        },
        bias: false,
    };
    const WINDOWS: usize = 7;
    const WARMUPS: usize = 32;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TriadArm {
        Candidate,
        CurrentWide,
        ActualAuto,
    }

    impl TriadArm {
        const fn name(self) -> &'static str {
            match self {
                Self::Candidate => "add_half_n96",
                Self::CurrentWide => "current_add_half_wide",
                Self::ActualAuto => "actual_triad_auto",
            }
        }
    }

    fn output(fixture: &Fixture, arm: TriadArm) -> &GuardedF32 {
        match arm {
            TriadArm::Candidate => &fixture.candidate,
            TriadArm::CurrentWide => &fixture.production,
            TriadArm::ActualAuto => &fixture.fast,
        }
    }

    fn output_mut(fixture: &mut Fixture, arm: TriadArm) -> &mut GuardedF32 {
        match arm {
            TriadArm::Candidate => &mut fixture.candidate,
            TriadArm::CurrentWide => &mut fixture.production,
            TriadArm::ActualAuto => &mut fixture.fast,
        }
    }

    fn operands(
        runtime: &Runtime,
        fixture: &Fixture,
        case: Case,
        arm: TriadArm,
    ) -> FixedFwdOperands {
        let typed = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        let (a, b) = if case.shape.k == 0 {
            (0, 0)
        } else {
            (
                fixture.a.ptr(&runtime.ctx.stream),
                fixture.b.ptr(&runtime.ctx.stream),
            )
        };
        FixedFwdOperands {
            c: typed(output(fixture, arm).ptr(&runtime.ctx.stream)),
            x: typed(a),
            w: typed(b),
            bias_ptr: None,
        }
    }

    fn compose_source() -> Result<String, String> {
        let prelude = include_str!("../kernels/_typed_prelude.cuh");
        let common = include_str!("../kernels/gemm_bi_fixed/common.cuh")
            .lines()
            .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
            .collect::<Vec<_>>()
            .join("\n");
        let tf32 = include_str!("../kernels/gemm_bi_fixed/tf32.cu");
        let fixed_n96 = include_str!("../kernels/gemm_bi_fixed/tf32_rna_n96.cu");
        let candidate = triad_nn_n96_source::compose_triad_nn_n96_source(fixed_n96)?;
        Ok([prelude, &common, tf32, &candidate].join("\n"))
    }

    fn new_runtime() -> Result<Runtime, String> {
        if std::env::var(ENV).as_deref() != Ok("1") {
            return Err(format!("set {ENV}=1"));
        }
        let device = GpuDevice::new(0)?;
        if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
            return Err(format!(
                "Triad NN N96 discovery requires CC8.9/142 SM, found {:?}/{}",
                device.compute_capability,
                device.multiprocessor_count()
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_batch_invariant(true);
        ctx.set_bi_tensor_cores(true);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        let compiler = ctx.kernels.compiler_identity();
        let artifacts = ctx.kernels.artifact_set_identity();
        if compiler.nvrtc_version != (13, 2)
            || compiler.target.as_str() != "sm_89"
            || !compiler.nvrtc_library_known
            || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
            || compiler.schedule_revision != SCHEDULE_REVISION
            || TUNING_TABLE_REVISION != 45
            || NUMERIC_ABI_REVISION != 5
            || SCHEDULE_REVISION != 8
            || artifacts.fixed.compile_key != compiler.invocation_digest
            || artifacts.fixed.artifact_digest == [0; 32]
            || artifacts.triad_sm80.artifact_digest == [0; 32]
        {
            return Err(format!(
                "Triad NN N96 discovery lost the accepted CUDA13.2 artifacts: compiler={compiler:?} artifacts={artifacts:?}"
            ));
        }
        let source = compose_source()?;
        let candidate_source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("sm_89"),
            options: vec![
                "--fmad=true".into(),
                "--extra-device-vectorization".into(),
                "-DNDEBUG".into(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, options)
            .map_err(|error| format!("compile Triad NN add-half N96: {error:?}"))?;
        let ptx_source = ptx.to_src();
        let candidate_ptx_sha = format!("{:x}", Sha256::digest(ptx_source.as_bytes()));
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
            .map_err(|error| format!("load Triad NN add-half N96: {error:?}"))?;
        let candidate = module
            .load_function(triad_nn_n96_source::TRIAD_NN_N96_SYMBOL)
            .map_err(|error| {
                format!(
                    "load {}: {error:?}",
                    triad_nn_n96_source::TRIAD_NN_N96_SYMBOL
                )
            })?;
        candidate
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                N96_SHARED as i32,
            )
            .map_err(|error| format!("set Triad NN N96 dynamic shared: {error:?}"))?;
        let runtime = Runtime {
            _device: device,
            ctx,
            _module: module,
            candidate,
            candidate_source_sha,
            candidate_ptx_sha,
        };
        validate_resources(&runtime)?;
        println!(
            concat!(
                "{{\"schema\":\"MambaBiTriadNnN96IdentityV1\",",
                "\"shape\":[2048,1536,768],\"cc\":\"8.9\",\"sm_count\":142,",
                "\"nvrtc\":[13,2],\"conversion\":\"add_half_ulp_tf32_v1\",",
                "\"candidate_symbol\":\"{}\",\"current_symbol\":\"{}\",",
                "\"candidate_source_sha\":\"{}\",\"candidate_ptx_sha\":\"{}\",",
                "\"fixed_artifact_sha\":\"{}\",\"triad_sm80_artifact_sha\":\"{}\"}}"
            ),
            triad_nn_n96_source::TRIAD_NN_N96_SYMBOL,
            CURRENT_WIDE_SYMBOL,
            runtime.candidate_source_sha,
            runtime.candidate_ptx_sha,
            digest_hex(&artifacts.fixed.artifact_digest),
            digest_hex(&artifacts.triad_sm80.artifact_digest),
        );
        Ok(runtime)
    }

    fn validate_resources(runtime: &Runtime) -> Result<(), String> {
        let registers = runtime
            .candidate
            .num_regs()
            .map_err(|error| format!("Triad NN N96 registers: {error:?}"))?;
        let local = runtime
            .candidate
            .local_size_bytes()
            .map_err(|error| format!("Triad NN N96 local bytes: {error:?}"))?;
        let static_shared = runtime
            .candidate
            .shared_size_bytes()
            .map_err(|error| format!("Triad NN N96 static shared: {error:?}"))?;
        let occupancy = runtime
            .candidate
            .occupancy_max_active_blocks_per_multiprocessor(256, N96_SHARED, None)
            .map_err(|error| format!("Triad NN N96 occupancy: {error:?}"))?;
        println!(
            "{{\"schema\":\"MambaBiTriadNnN96ResourceV1\",\"symbol\":\"{}\",\"threads\":256,\"registers\":{registers},\"local_bytes\":{local},\"static_shared_bytes\":{static_shared},\"dynamic_shared_bytes\":{N96_SHARED},\"occupancy\":{occupancy}}}",
            triad_nn_n96_source::TRIAD_NN_N96_SYMBOL,
        );
        if !(1..=136).contains(&registers) || local != 0 || static_shared != 0 || occupancy != 1 {
            return Err(format!(
                "Triad NN N96 resource gate failed: registers={registers} local={local} static={static_shared} occupancy={occupancy}"
            ));
        }
        Ok(())
    }

    fn configure(runtime: &Runtime, arm: TriadArm) {
        runtime
            .ctx
            .set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        runtime.ctx.set_batch_invariant(true);
        runtime.ctx.set_fast_gemm(false);
        runtime.ctx.set_bi_tensor_cores(true);
        runtime.ctx.set_bi_gemm_family(match arm {
            TriadArm::CurrentWide => BiGemmFamily::Fixed,
            TriadArm::Candidate | TriadArm::ActualAuto => BiGemmFamily::Triad,
        });
    }

    fn launch(
        runtime: &Runtime,
        fixture: &Fixture,
        case: Case,
        arm: TriadArm,
    ) -> Result<(), String> {
        configure(runtime, arm);
        let operands = operands(runtime, fixture, case, arm);
        match arm {
            TriadArm::Candidate => {
                let params = N96Params {
                    alpha: 1.0,
                    beta: 0.0,
                    m: i32::try_from(case.shape.m).map_err(|_| "M exceeds i32")?,
                    k: i32::try_from(case.shape.k).map_err(|_| "K exceeds i32")?,
                    n: i32::try_from(case.shape.n).map_err(|_| "N exceeds i32")?,
                    lda: i32::try_from(case.shape.k).map_err(|_| "lda exceeds i32")?,
                    ldb: i32::try_from(case.shape.n).map_err(|_| "ldb exceeds i32")?,
                    ldc: i32::try_from(case.shape.n).map_err(|_| "ldc exceeds i32")?,
                };
                let output = operands.c.ptr;
                let a = operands.x.ptr;
                let b = operands.w.ptr;
                let bias = 0_u64;
                let mut builder = runtime.ctx.stream.launch_builder(&runtime.candidate);
                builder.arg(&output);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&bias);
                builder.arg(&params);
                unsafe { builder.launch(candidate_config(case.shape)?) }
                    .map(|_| ())
                    .map_err(|error| format!("launch Triad NN N96: {error:?}"))
            }
            TriadArm::CurrentWide => fixed_forward_with_tile(
                &runtime.ctx,
                operands,
                case.shape,
                FixedTile::Tf32M128N128S3,
            ),
            TriadArm::ActualAuto => gpu_gemm_typed_forward_raw(
                &runtime.ctx,
                operands.c,
                operands.x,
                operands.w,
                None,
                (case.shape.m, case.shape.k, case.shape.n),
            ),
        }
    }

    fn reset(runtime: &Runtime, fixture: &mut Fixture, arm: TriadArm) -> Result<(), String> {
        output_mut(fixture, arm).reset(&runtime.ctx.stream)
    }

    fn bits(
        runtime: &Runtime,
        fixture: &Fixture,
        arm: TriadArm,
        label: &str,
    ) -> Result<Vec<u32>, String> {
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize {label}: {error:?}"))?;
        output(fixture, arm).all_bits(&runtime.ctx.stream, label)
    }

    fn run_eager(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: TriadArm,
    ) -> Result<Vec<u32>, String> {
        reset(runtime, fixture, arm)?;
        launch(runtime, fixture, case, arm)?;
        let result = bits(runtime, fixture, arm, arm.name())?;
        if result.iter().any(|word| *word == POISON_BITS) {
            return Err(format!("{} retained poison", arm.name()));
        }
        fixture.validate_inputs(runtime)?;
        Ok(result)
    }

    fn capture_arm(
        runtime: &Runtime,
        fixture: &mut Fixture,
        case: Case,
        arm: TriadArm,
    ) -> Result<CudaGraph, String> {
        if arm == TriadArm::ActualAuto {
            run_eager(runtime, fixture, case, arm)?;
        }
        unsafe { capture_into_graph(&runtime.ctx.stream, || launch(runtime, fixture, case, arm)) }
    }

    fn run_graph(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: TriadArm,
        graph: &CudaGraph,
    ) -> Result<Vec<u32>, String> {
        reset(runtime, fixture, arm)?;
        graph
            .launch()
            .map_err(|error| format!("launch {} graph: {error:?}", arm.name()))?;
        let result = bits(runtime, fixture, arm, arm.name())?;
        if result.iter().any(|word| *word == POISON_BITS) {
            return Err(format!("{} graph retained poison", arm.name()));
        }
        fixture.validate_inputs(runtime)?;
        Ok(result)
    }

    fn fixture_with_values(
        runtime: &Runtime,
        case: Case,
        a_values: Vec<f32>,
        b_values: Vec<f32>,
    ) -> Result<Fixture, String> {
        let output_len = case.shape.m * case.shape.n;
        Ok(Fixture {
            a: GuardedF32::new(&runtime.ctx.stream, a_values)?,
            b: GuardedF32::new(&runtime.ctx.stream, b_values)?,
            bias: GuardedF32::new(&runtime.ctx.stream, vec![0.0; case.shape.n])?,
            candidate: GuardedF32::new(
                &runtime.ctx.stream,
                vec![f32::from_bits(POISON_BITS); output_len],
            )?,
            production: GuardedF32::new(
                &runtime.ctx.stream,
                vec![f32::from_bits(POISON_BITS); output_len],
            )?,
            fast: GuardedF32::new(
                &runtime.ctx.stream,
                vec![f32::from_bits(POISON_BITS); output_len],
            )?,
        })
    }

    fn ordinary_fixture(runtime: &Runtime, case: Case) -> Result<Fixture, String> {
        fixture_with_values(
            runtime,
            case,
            probe_values(case.shape.m * case.shape.k, 0xa096_89a1),
            probe_values(case.shape.k * case.shape.n, 0xb096_89a2),
        )
    }

    fn exceptional_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let mut a = vec![0.0; TAIL.shape.m * TAIL.shape.k];
        for row in 0..TAIL.shape.m {
            a[row * TAIL.shape.k] = 1.0;
        }
        let mut b = vec![0.0; TAIL.shape.k * TAIL.shape.n];
        for (column, raw) in [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0x7fc0_1234,
            0xffff_ffff,
        ]
        .into_iter()
        .enumerate()
        {
            b[column] = f32::from_bits(raw);
        }
        fixture_with_values(runtime, TAIL, a, b)
    }

    fn check_pair(
        runtime: &Runtime,
        case: Case,
        mut fixture: Fixture,
        label: &str,
    ) -> Result<Vec<u32>, String> {
        let expected = run_eager(runtime, &mut fixture, case, TriadArm::CurrentWide)?;
        let candidate = run_eager(runtime, &mut fixture, case, TriadArm::Candidate)?;
        if candidate != expected {
            return Err(format!(
                "{label} candidate differs from current add-half wide"
            ));
        }
        let current_graph = capture_arm(runtime, &mut fixture, case, TriadArm::CurrentWide)?;
        let candidate_graph = capture_arm(runtime, &mut fixture, case, TriadArm::Candidate)?;
        assert_single_tf32_graph(
            &current_graph,
            case.shape,
            CURRENT_WIDE_SYMBOL,
            128,
            128,
            98_304,
        )?;
        assert_n96_graph(
            &candidate_graph,
            case.shape,
            triad_nn_n96_source::TRIAD_NN_N96_SYMBOL,
        )?;
        for repeat in 0..2 {
            for (arm, graph) in [
                (TriadArm::CurrentWide, &current_graph),
                (TriadArm::Candidate, &candidate_graph),
            ] {
                if run_eager(runtime, &mut fixture, case, arm)? != expected
                    || run_graph(runtime, &mut fixture, arm, graph)? != expected
                {
                    return Err(format!(
                        "{label} {} repeat {repeat} changed bits",
                        arm.name()
                    ));
                }
            }
        }
        println!(
            "{{\"schema\":\"MambaBiTriadNnN96BitsV1\",\"case\":\"{label}\",\"shape\":[{},{},{}],\"eager_repeats\":2,\"graph_repeats\":2,\"passed\":true}}",
            case.shape.m, case.shape.k, case.shape.n,
        );
        Ok(expected)
    }

    fn prepare_target(
        runtime: &Runtime,
    ) -> Result<(Fixture, CudaGraph, CudaGraph, Vec<u32>), String> {
        let mut fixture = ordinary_fixture(runtime, TARGET)?;
        let expected = run_eager(runtime, &mut fixture, TARGET, TriadArm::CurrentWide)?;
        if run_eager(runtime, &mut fixture, TARGET, TriadArm::Candidate)? != expected
            || run_eager(runtime, &mut fixture, TARGET, TriadArm::ActualAuto)? != expected
        {
            return Err("target candidate/current/AUTO bits differ".into());
        }
        let candidate_graph = capture_arm(runtime, &mut fixture, TARGET, TriadArm::Candidate)?;
        let auto_graph = capture_arm(runtime, &mut fixture, TARGET, TriadArm::ActualAuto)?;
        assert_n96_graph(
            &candidate_graph,
            TARGET.shape,
            triad_nn_n96_source::TRIAD_NN_N96_SYMBOL,
        )?;
        assert_single_tf32_graph(
            &auto_graph,
            TARGET.shape,
            CURRENT_WIDE_SYMBOL,
            128,
            128,
            98_304,
        )?;
        for repeat in 0..2 {
            for (arm, graph) in [
                (TriadArm::Candidate, &candidate_graph),
                (TriadArm::ActualAuto, &auto_graph),
            ] {
                if run_eager(runtime, &mut fixture, TARGET, arm)? != expected
                    || run_graph(runtime, &mut fixture, arm, graph)? != expected
                {
                    return Err(format!(
                        "target {} repeat {repeat} changed bits",
                        arm.name()
                    ));
                }
            }
        }
        Ok((fixture, candidate_graph, auto_graph, expected))
    }

    #[derive(Clone, Copy)]
    enum Path {
        Eager,
        Graph,
    }

    impl Path {
        const fn name(self) -> &'static str {
            match self {
                Self::Eager => "eager",
                Self::Graph => "graph",
            }
        }
    }

    fn measure_one(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: TriadArm,
        graph: &CudaGraph,
        path: Path,
        expected: &[u32],
    ) -> Result<f64, String> {
        reset(runtime, fixture, arm)?;
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record {} start: {error:?}", arm.name()))?;
        match path {
            Path::Eager => launch(runtime, fixture, TARGET, arm)?,
            Path::Graph => graph
                .launch()
                .map_err(|error| format!("launch {} graph: {error:?}", arm.name()))?,
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record {} end: {error:?}", arm.name()))?;
        let elapsed_us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("measure {}: {error:?}", arm.name()))?,
        ) * 1_000.0;
        if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
            return Err(format!("invalid {} sample {elapsed_us}", arm.name()));
        }
        if bits(runtime, fixture, arm, arm.name())? != expected {
            return Err(format!("{} timing observation changed bits", arm.name()));
        }
        fixture.validate_inputs(runtime)?;
        Ok(elapsed_us)
    }

    fn json_raw(raw: &[[f64; 4]]) -> String {
        format!(
            "[{}]",
            raw.iter()
                .map(|v| format!("[{:.9},{:.9},{:.9},{:.9}]", v[0], v[1], v[2], v[3]))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn screen_stratum(
        runtime: &Runtime,
        fixture: &mut Fixture,
        candidate_graph: &CudaGraph,
        auto_graph: &CudaGraph,
        expected: &[u32],
        path: Path,
        order: AdaBracketOrder,
    ) -> Result<[f64; 2], String> {
        for _ in 0..WARMUPS {
            measure_one(
                runtime,
                fixture,
                TriadArm::Candidate,
                candidate_graph,
                path,
                expected,
            )?;
            measure_one(
                runtime,
                fixture,
                TriadArm::ActualAuto,
                auto_graph,
                path,
                expected,
            )?;
        }
        let arms = match order {
            AdaBracketOrder::Abba => [
                (TriadArm::ActualAuto, auto_graph),
                (TriadArm::Candidate, candidate_graph),
                (TriadArm::Candidate, candidate_graph),
                (TriadArm::ActualAuto, auto_graph),
            ],
            AdaBracketOrder::Baab => [
                (TriadArm::Candidate, candidate_graph),
                (TriadArm::ActualAuto, auto_graph),
                (TriadArm::ActualAuto, auto_graph),
                (TriadArm::Candidate, candidate_graph),
            ],
        };
        let mut raw = Vec::with_capacity(WINDOWS);
        let mut ratios = Vec::with_capacity(WINDOWS);
        for _ in 0..WINDOWS {
            let mut observation = [0.0; 4];
            for (index, (arm, graph)) in arms.into_iter().enumerate() {
                observation[index] = measure_one(runtime, fixture, arm, graph, path, expected)?;
            }
            let (candidate, auto) = match order {
                AdaBracketOrder::Abba => (
                    0.5 * (observation[1] + observation[2]),
                    0.5 * (observation[0] + observation[3]),
                ),
                AdaBracketOrder::Baab => (
                    0.5 * (observation[0] + observation[3]),
                    0.5 * (observation[1] + observation[2]),
                ),
            };
            ratios.push(candidate / auto);
            raw.push(observation);
        }
        let p50 = percentile(&ratios, 0.5);
        let p95 = percentile(&ratios, 0.95);
        let order_name = match order {
            AdaBracketOrder::Abba => "ABBA",
            AdaBracketOrder::Baab => "BAAB",
        };
        println!(
            "{{\"schema\":\"MambaBiTriadNnN96ScreenV1\",\"cell\":\"d768_out_proj\",\"shape\":[2048,1536,768],\"candidate\":\"add_half_n96\",\"comparator\":\"actual_triad_auto\",\"candidate_symbol\":\"{}\",\"comparator_symbol\":\"{CURRENT_WIDE_SYMBOL}\",\"path\":\"{}\",\"order\":\"{order_name}\",\"windows\":{WINDOWS},\"warmups_per_arm\":{WARMUPS},\"logical_gemms_per_observation\":1,\"raw_observations_us\":{},\"ratio_direction\":\"candidate_over_auto\",\"ratio_p50\":{p50:.9},\"ratio_p95\":{p95:.9}}}",
            triad_nn_n96_source::TRIAD_NN_N96_SYMBOL,
            path.name(),
            json_raw(&raw),
        );
        Ok([p50, p95])
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC8.9/142-SM CUDA13.2 GPU; set MAMBA_TRIAD_NN_N96_DISCOVERY=1"]
    fn triad_nn_d768_out_add_half_n96_discovery_once7() -> Result<(), String> {
        assert!(
            !cfg!(debug_assertions),
            "Triad NN N96 discovery requires --release"
        );
        let runtime = new_runtime()?;
        check_pair(&runtime, TAIL, ordinary_fixture(&runtime, TAIL)?, "tail")?;
        check_pair(
            &runtime,
            TAIL,
            exceptional_fixture(&runtime)?,
            "exceptional",
        )?;
        check_pair(&runtime, K0, ordinary_fixture(&runtime, K0)?, "k0_null_ab")?;
        let (mut fixture, candidate_graph, auto_graph, expected) = prepare_target(&runtime)?;
        let mut strata = Vec::with_capacity(4);
        for path in [Path::Eager, Path::Graph] {
            for order in [AdaBracketOrder::Abba, AdaBracketOrder::Baab] {
                strata.push(screen_stratum(
                    &runtime,
                    &mut fixture,
                    &candidate_graph,
                    &auto_graph,
                    &expected,
                    path,
                    order,
                )?);
            }
        }
        let retain = strata.len() == 4
            && strata.iter().all(|[p50, p95]| {
                p50.is_finite()
                    && p95.is_finite()
                    && *p50 > 0.0
                    && *p95 > 0.0
                    && *p50 < 0.99
                    && *p95 < 0.99
            });
        let strata_json = format!(
            "[{}]",
            strata
                .iter()
                .map(|[p50, p95]| format!("[{p50:.9},{p95:.9}]"))
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "{{\"schema\":\"MambaBiTriadNnN96DecisionV1\",\"cell\":\"d768_out_proj\",\"shape\":[2048,1536,768],\"candidate\":\"add_half_n96\",\"comparator\":\"actual_triad_auto\",\"strata_order\":[\"eager/ABBA\",\"eager/BAAB\",\"graph/ABBA\",\"graph/BAAB\"],\"strata_fields\":[\"ratio_p50\",\"ratio_p95\"],\"strata\":{strata_json},\"ratio_direction\":\"candidate_over_auto\",\"retain\":{retain},\"decision\":\"{}\",\"promotion\":false}}",
            if retain { "advance" } else { "stop_no_retry" },
        );
        if !retain {
            return Err("Triad NN add-half N96 failed one or more <0.99 strata".into());
        }
        Ok(())
    }
}
