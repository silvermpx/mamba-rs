//! Test-only CUDA 13.2 screening harness for three Ada homogeneous-half arms.
#![cfg(feature = "cuda")]

use std::ffi::{CStr, c_int, c_void};
use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaGraph, CudaModule, DeviceRepr, LaunchConfig, PushKernelArg, sys,
};
use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_inference::{
    InferenceFwdOperands, InferenceShape, InferenceTile, inference_forward,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    NUMERIC_ABI_REVISION, SCHEDULE_REVISION, TUNING_TABLE_REVISION, digest_hex,
};
use sha2::{Digest as _, Sha256};

const CUDA_SOURCE: &str = include_str!("gemm_bi_fixed_half_batch_discovery.cu");
const GUARD: usize = 64;
const GUARD_BITS: u16 = 0x7e4d;
const POISON_BITS: u16 = 0x7e31;

#[derive(Clone, Copy)]
struct Cell {
    label: &'static str,
    dtype: WeightDtype,
    shape: InferenceShape,
    symbol: &'static str,
    function: usize,
    bm: usize,
    bn: usize,
    threads: u32,
    shared: u32,
    occupancy: u32,
    control: InferenceTile,
}

const CELLS: [Cell; 4] = [
    Cell {
        label: "b0_bf16_compact_s3",
        dtype: WeightDtype::Bf16,
        shape: InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        symbol: "gemm_bi_nn_fixed_half_batch_s3_compact_bf16",
        function: 0,
        bm: 128,
        bn: 128,
        threads: 256,
        shared: 98_304,
        occupancy: 1,
        control: InferenceTile::Tc128Sm89S3,
    },
    Cell {
        label: "b0_f16_compact_s3",
        dtype: WeightDtype::F16,
        shape: InferenceShape {
            m: 4621,
            k: 768,
            n: 2304,
        },
        symbol: "gemm_bi_nn_fixed_half_batch_s3_compact_f16",
        function: 1,
        bm: 128,
        bn: 128,
        threads: 256,
        shared: 98_304,
        occupancy: 1,
        control: InferenceTile::Tc128Sm89S3,
    },
    Cell {
        label: "d0_f16_m64n64_bk64_s3",
        dtype: WeightDtype::F16,
        shape: InferenceShape {
            m: 2048,
            k: 768,
            n: 2304,
        },
        symbol: "gemm_bi_nn_fixed_half_batch_m64n64_bk64_s3_f16",
        function: 2,
        bm: 64,
        bn: 64,
        threads: 128,
        shared: 49_152,
        occupancy: 2,
        control: InferenceTile::Tc128Sm89Swizzle,
    },
    Cell {
        label: "e0_f16_m128n64_bk64_s2",
        dtype: WeightDtype::F16,
        shape: InferenceShape {
            m: 2048,
            k: 2304,
            n: 768,
        },
        symbol: "gemm_bi_nn_fixed_half_batch_m128n64_bk64_s2_f16",
        function: 3,
        bm: 128,
        bn: 64,
        threads: 128,
        shared: 49_152,
        occupancy: 2,
        control: InferenceTile::Tc128Sm89Pipeline,
    },
];

#[repr(C)]
#[derive(Clone, Copy)]
struct RawHalfParams([u32; 8]);
unsafe impl DeviceRepr for RawHalfParams {}

struct Runtime {
    _device: GpuDevice,
    ctx: GpuCtx,
    _module: Arc<CudaModule>,
    functions: [Result<CudaFunction, String>; 4],
    source_sha: String,
    ptx_sha: String,
}

struct GuardedHalf {
    buffer: GpuByteBuffer,
    baseline: Vec<u16>,
    active: usize,
}

impl GuardedHalf {
    fn new(ctx: &GpuCtx, values: &[u16]) -> Result<Self, String> {
        let mut baseline = vec![GUARD_BITS; GUARD + values.len() + GUARD];
        baseline[GUARD..GUARD + values.len()].copy_from_slice(values);
        let mut buffer = GpuByteBuffer::zeros(&ctx.stream, baseline.len() * 2)?;
        buffer.upload_bytes(&ctx.stream, bytemuck::cast_slice(&baseline))?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("guard upload: {error:?}"))?;
        Ok(Self {
            buffer,
            baseline,
            active: values.len(),
        })
    }

    fn ptr(&self) -> u64 {
        self.buffer.cached_ptr() + (GUARD * 2) as u64
    }

    fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
        self.buffer
            .upload_bytes(&ctx.stream, bytemuck::cast_slice(&self.baseline))
    }

    fn raw(&self, ctx: &GpuCtx) -> Result<Vec<u16>, String> {
        let mut bytes = vec![0; self.buffer.len_bytes()];
        ctx.stream
            .memcpy_dtoh(self.buffer.inner(), &mut bytes)
            .map_err(|error| format!("half download: {error:?}"))?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("half sync: {error:?}"))?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_le_bytes)
            .collect())
    }

    fn active_and_guards(&self, ctx: &GpuCtx, label: &str) -> Result<Vec<u16>, String> {
        let raw = self.raw(ctx)?;
        if raw[..GUARD]
            .iter()
            .chain(&raw[GUARD + self.active..])
            .any(|&x| x != GUARD_BITS)
        {
            return Err(format!("{label} changed a redzone"));
        }
        Ok(raw[GUARD..GUARD + self.active].to_vec())
    }

    fn unchanged(&self, ctx: &GpuCtx, label: &str) -> Result<(), String> {
        if self.raw(ctx)? != self.baseline {
            return Err(format!("{label} or its redzone changed"));
        }
        Ok(())
    }
}

struct Fixture {
    a: GuardedHalf,
    b: GuardedHalf,
    candidate: GuardedHalf,
    control: GuardedHalf,
    fast: GuardedHalf,
}

impl Fixture {
    fn new(runtime: &Runtime, cell: Cell, exceptional: bool) -> Result<Self, String> {
        let a = half_corpus(
            cell.shape.m * cell.shape.k,
            cell.dtype,
            0xa11c_e001,
            exceptional,
        );
        let b = half_corpus(
            cell.shape.k * cell.shape.n,
            cell.dtype,
            0xb11c_e002,
            exceptional,
        );
        let poison = vec![POISON_BITS; cell.shape.m * cell.shape.n];
        Ok(Self {
            a: GuardedHalf::new(&runtime.ctx, &a)?,
            b: GuardedHalf::new(&runtime.ctx, &b)?,
            candidate: GuardedHalf::new(&runtime.ctx, &poison)?,
            control: GuardedHalf::new(&runtime.ctx, &poison)?,
            fast: GuardedHalf::new(&runtime.ctx, &poison)?,
        })
    }

    fn output(&self, arm: Arm) -> &GuardedHalf {
        match arm {
            Arm::Candidate => &self.candidate,
            Arm::Control => &self.control,
            Arm::Fast => &self.fast,
        }
    }
    fn output_mut(&mut self, arm: Arm) -> &mut GuardedHalf {
        match arm {
            Arm::Candidate => &mut self.candidate,
            Arm::Control => &mut self.control,
            Arm::Fast => &mut self.fast,
        }
    }
    fn operands(&self, cell: Cell, arm: Arm) -> InferenceFwdOperands {
        let typed = |ptr| TypedPtr {
            ptr,
            dtype: cell.dtype,
        };
        InferenceFwdOperands {
            c: typed(self.output(arm).ptr()),
            x: typed(self.a.ptr()),
            w: typed(self.b.ptr()),
            bias_ptr: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Candidate,
    Control,
    Fast,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Control => "actual_auto",
            Self::Fast => "fast_half",
        }
    }
}

fn half_corpus(len: usize, dtype: WeightDtype, mut state: u64, exceptional: bool) -> Vec<u16> {
    let f16_special = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x3c01, 0xbc01, 0x3555, 0xb555,
    ];
    let bf16_special = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x3f81, 0xbf81, 0x3eab, 0xbeab,
    ];
    (0..len)
        .map(|index| {
            if exceptional && index < 256 {
                return match dtype {
                    WeightDtype::Bf16 => bf16_special[index % bf16_special.len()],
                    WeightDtype::F16 => f16_special[index % f16_special.len()],
                    WeightDtype::F32 => unreachable!(),
                };
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = ((state.wrapping_add(index as u64) % 4095) as i32 - 2047) as f32 / 8192.0;
            match dtype {
                WeightDtype::Bf16 => half::bf16::from_f32(value).to_bits(),
                WeightDtype::F16 => half::f16::from_f32(value).to_bits(),
                WeightDtype::F32 => unreachable!(),
            }
        })
        .collect()
}

fn strip_local_include(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim().starts_with("#include \""))
        .collect::<Vec<_>>()
        .join("\n")
}

fn compose_source() -> String {
    let prelude = include_str!("../kernels/_typed_prelude.cuh");
    let common = strip_local_include(include_str!("../kernels/gemm_bi_inference/common.cuh"));
    let layout = include_str!("../kernels/gemm_bi_inference/sm89_half_swizzle_layout.cuh");
    let swizzle = strip_local_include(include_str!(
        "../kernels/gemm_bi_inference/sm89_half_swizzle.cu"
    ));
    let s3 = include_str!("../kernels/gemm_bi_inference/sm89_half_s3.cu");
    [prelude, &common, layout, &swizzle, s3, CUDA_SOURCE].join("\n")
}

fn new_runtime() -> Result<Runtime, String> {
    if std::env::var("MAMBA_FIXED_HALF_BATCH_DISCOVERY").as_deref() != Ok("1") {
        return Err("set MAMBA_FIXED_HALF_BATCH_DISCOVERY=1".into());
    }
    let device = GpuDevice::new(0)?;
    if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
        return Err(format!(
            "half batch requires 142-SM Ada, got {:?}/{}",
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
        || TUNING_TABLE_REVISION != 44
        || NUMERIC_ABI_REVISION != 5
        || SCHEDULE_REVISION != 8
        || artifact.compile_key != compiler.invocation_digest
    {
        return Err(format!(
            "lost accepted CUDA13.2 Fixed identity: {compiler:?} {artifact:?}"
        ));
    }
    let source = compose_source();
    let source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        source,
        cudarc::nvrtc::CompileOptions {
            arch: Some("sm_89"),
            options: vec![
                "--fmad=true".into(),
                "--extra-device-vectorization".into(),
                "-DNDEBUG".into(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        },
    )
    .map_err(|error| format!("compile half batch: {error:?}"))?;
    let ptx_source = ptx.to_src();
    let ptx_sha = format!("{:x}", Sha256::digest(ptx_source.as_bytes()));
    let module = device
        .context()
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
        .map_err(|error| format!("load half batch: {error:?}"))?;
    let mut loaded = Vec::new();
    for cell in CELLS {
        let function = module
            .load_function(cell.symbol)
            .map_err(|error| format!("load {}: {error:?}", cell.symbol))
            .and_then(|function| {
                function.set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                cell.shared as i32,
                ).map_err(|error| format!("set {} shared: {error:?}", cell.label))?;
                Ok(function)
            });
        loaded.push(function);
    }
    let functions: [Result<CudaFunction, String>; 4] =
        loaded.try_into().map_err(|_| "half function count")?;
    let runtime = Runtime {
        _device: device,
        ctx,
        _module: module,
        functions,
        source_sha,
        ptx_sha,
    };
    println!(
        concat!(
            "{{\"schema\":\"MambaBiFixedHalfBatchIdentityV1\",",
            "\"cc\":\"8.9\",\"sm_count\":142,\"nvrtc\":[13,2],\"tuning_revision\":44,",
            "\"numeric_revision\":5,\"schedule_revision\":8,\"fixed_source_sha\":\"{}\",",
            "\"fixed_invocation_sha\":\"{}\",\"fixed_artifact_sha\":\"{}\",",
            "\"candidate_source_sha\":\"{}\",\"candidate_ptx_sha\":\"{}\"}}"
        ),
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        digest_hex(&artifact.artifact_digest),
        runtime.source_sha,
        runtime.ptx_sha
    );
    Ok(runtime)
}

fn validate_resource(runtime: &Runtime, cell: Cell) -> Result<(), String> {
    let function = runtime.functions[cell.function]
        .as_ref()
        .map_err(Clone::clone)?;
    let registers = function
        .num_regs()
        .map_err(|e| format!("{} regs: {e:?}", cell.label))?;
    let local = function
        .local_size_bytes()
        .map_err(|e| format!("{} local: {e:?}", cell.label))?;
    let static_shared = function
        .shared_size_bytes()
        .map_err(|e| format!("{} static shared: {e:?}", cell.label))?;
    let active = function
        .occupancy_max_active_blocks_per_multiprocessor(cell.threads, cell.shared as usize, None)
        .map_err(|e| format!("{} occupancy: {e:?}", cell.label))?;
    if !(1..=255).contains(&registers)
        || local != 0
        || static_shared != 0
        || active != cell.occupancy
    {
        return Err(format!(
            "{} resource reject regs={registers} local={local} static={static_shared} active={active}",
            cell.label
        ));
    }
    println!(
        "{{\"schema\":\"MambaBiFixedHalfBatchResourceV1\",\"cell\":\"{}\",\"symbol\":\"{}\",\"registers\":{},\"local_bytes\":{},\"static_shared_bytes\":{},\"dynamic_shared_bytes\":{},\"active_ctas\":{}}}",
        cell.label, cell.symbol, registers, local, static_shared, cell.shared, active
    );
    Ok(())
}

fn configure(ctx: &GpuCtx, arm: Arm) {
    ctx.set_bi_gemm_family(BiGemmFamily::Inference);
    ctx.set_bi_tensor_cores(false);
    ctx.set_batch_invariant(arm != Arm::Fast);
    ctx.set_fast_gemm(arm == Arm::Fast);
}

fn launch(runtime: &Runtime, fixture: &Fixture, cell: Cell, arm: Arm) -> Result<(), String> {
    configure(&runtime.ctx, arm);
    let ops = fixture.operands(cell, arm);
    match arm {
        Arm::Candidate => {
            if cell.function >= 2 {
                let legal = (cell.function == 2
                    && cell.shape
                        == (InferenceShape {
                            m: 2048,
                            k: 768,
                            n: 2304,
                        })
                    && cell.shape.k >= 128)
                    || (cell.function == 3
                        && cell.shape
                            == (InferenceShape {
                                m: 2048,
                                k: 2304,
                                n: 768,
                            })
                        && cell.shape.k >= 64);
                if !legal {
                    return Err(format!(
                        "{} rejected outside its full aligned exact cell",
                        cell.label
                    ));
                }
            }
            let params = RawHalfParams([
                1.0f32.to_bits(),
                0.0f32.to_bits(),
                cell.shape.m as u32,
                cell.shape.n as u32,
                cell.shape.k as u32,
                cell.shape.k as u32,
                cell.shape.n as u32,
                cell.shape.n as u32,
            ]);
            let bias = 0u64;
            let function = runtime.functions[cell.function]
                .as_ref()
                .map_err(Clone::clone)?;
            let mut builder = runtime.ctx.stream.launch_builder(function);
            builder
                .arg(&ops.c.ptr)
                .arg(&ops.x.ptr)
                .arg(&ops.w.ptr)
                .arg(&bias)
                .arg(&params);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (
                        (cell.shape.m.div_ceil(cell.bm) * cell.shape.n.div_ceil(cell.bn)) as u32,
                        1,
                        1,
                    ),
                    block_dim: (cell.threads, 1, 1),
                    shared_mem_bytes: cell.shared,
                })
            }
            .map(|_| ())
            .map_err(|error| format!("{} candidate launch: {error:?}", cell.label))
        }
        Arm::Control => {
            let selected = inference_forward(
                &runtime.ctx,
                ops.c,
                ops.x,
                ops.w,
                None,
                (cell.shape.m, cell.shape.k, cell.shape.n),
            )?;
            if selected != cell.control {
                return Err(format!(
                    "{} AUTO selected {selected:?}, expected {:?}",
                    cell.label, cell.control
                ));
            }
            Ok(())
        }
        Arm::Fast => gpu_gemm_typed_forward_raw(
            &runtime.ctx,
            ops.c,
            ops.x,
            ops.w,
            None,
            (cell.shape.m, cell.shape.k, cell.shape.n),
        ),
    }
}

fn graph(
    runtime: &Runtime,
    fixture: &Fixture,
    cell: Cell,
    arm: Arm,
    operations: usize,
) -> Result<CudaGraph, String> {
    unsafe {
        capture_into_graph(&runtime.ctx.stream, || {
            for _ in 0..operations {
                launch(runtime, fixture, cell, arm)?;
            }
            Ok(())
        })
    }
}

fn assert_candidate_graph(graph: &CudaGraph, fixture: &Fixture, cell: Cell) -> Result<(), String> {
    let mut count = 0;
    if unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) }
        != sys::CUresult::CUDA_SUCCESS
        || count != 1
    {
        return Err(format!("{} candidate graph nodes={count}", cell.label));
    }
    let mut node = std::ptr::null_mut();
    unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) };
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) };
    let mut name = std::ptr::null();
    unsafe { sys::cuFuncGetName(&mut name, params.func) };
    if name.is_null() {
        return Err(format!("{} null graph symbol", cell.label));
    }
    let symbol = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|e| e.to_string())?;
    let grid = (cell.shape.m.div_ceil(cell.bm) * cell.shape.n.div_ceil(cell.bn)) as u32;
    if symbol != cell.symbol
        || params.gridDimX != grid
        || params.blockDimX != cell.threads
        || params.sharedMemBytes != cell.shared
    {
        return Err(format!(
            "{} physical reject symbol={symbol} grid={} block={} shared={}",
            cell.label, params.gridDimX, params.blockDimX, params.sharedMemBytes
        ));
    }
    for (index, expected) in [(0, 8), (8, 8), (16, 8), (24, 8), (32, 32)]
        .into_iter()
        .enumerate()
    {
        let (mut offset, mut size) = (0, 0);
        if unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) }
            != sys::CUresult::CUDA_SUCCESS
            || (offset, size) != expected
        {
            return Err(format!("{} ABI {index}: {offset}/{size}", cell.label));
        }
    }
    let (mut offset, mut size) = (0, 0);
    if unsafe { sys::cuFuncGetParamInfo(params.func, 5, &mut offset, &mut size) }
        != sys::CUresult::CUDA_ERROR_INVALID_VALUE
    {
        return Err(format!("{} accepted sixth argument", cell.label));
    }
    if params.kernelParams.is_null() {
        return Err(format!("{} null graph argument vector", cell.label));
    }
    let ops = fixture.operands(cell, Arm::Candidate);
    for (index, expected) in [ops.c.ptr, ops.x.ptr, ops.w.ptr, 0].into_iter().enumerate() {
        let pointer = unsafe { *params.kernelParams.add(index) };
        if pointer.is_null() || unsafe { pointer.cast::<u64>().read_unaligned() } != expected {
            return Err(format!("{} graph pointer argument {index}", cell.label));
        }
    }
    let expected_bundle = [
        1.0f32.to_bits(),
        0.0f32.to_bits(),
        cell.shape.m as u32,
        cell.shape.n as u32,
        cell.shape.k as u32,
        cell.shape.k as u32,
        cell.shape.n as u32,
        cell.shape.n as u32,
    ];
    let bundle = unsafe { *params.kernelParams.add(4) };
    if bundle.is_null() || unsafe { bundle.cast::<[u32; 8]>().read_unaligned() } != expected_bundle
    {
        return Err(format!("{} graph parameter bundle", cell.label));
    }
    Ok(())
}

fn pedantic_reference(
    runtime: &Runtime,
    fixture: &Fixture,
    cell: Cell,
) -> Result<Vec<f32>, String> {
    let reference = GpuBuffer::zeros(&runtime.ctx.stream, cell.shape.m * cell.shape.n)?;
    let ops = InferenceFwdOperands {
        c: TypedPtr {
            ptr: reference.cached_ptr(),
            dtype: WeightDtype::F32,
        },
        x: TypedPtr {
            ptr: fixture.a.ptr(),
            dtype: cell.dtype,
        },
        w: TypedPtr {
            ptr: fixture.b.ptr(),
            dtype: cell.dtype,
        },
        bias_ptr: None,
    };
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *runtime.ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cell.shape.n as c_int,
            cell.shape.m as c_int,
            cell.shape.k as c_int,
            &alpha as *const f32 as *const c_void,
            ops.w.ptr as *const c_void,
            ops.w.dtype.cuda_data_type(),
            cell.shape.n as c_int,
            ops.x.ptr as *const c_void,
            ops.x.dtype.cuda_data_type(),
            cell.shape.k as c_int,
            &beta as *const f32 as *const c_void,
            ops.c.ptr as *mut c_void,
            ops.c.dtype.cuda_data_type(),
            cell.shape.n as c_int,
            cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .map_err(|error| format!("{} PEDANTIC: {error:?}", cell.label))?;
    reference.to_cpu(&runtime.ctx.stream)
}

fn normalized_error(bits: &[u16], reference: &[f32], dtype: WeightDtype) -> Result<f64, String> {
    let mut max_error = 0.0f64;
    let mut scale = 0.0f64;
    for (&bits, &gold) in bits.iter().zip(reference) {
        let actual = match dtype {
            WeightDtype::Bf16 => half::bf16::from_bits(bits).to_f32(),
            WeightDtype::F16 => half::f16::from_bits(bits).to_f32(),
            WeightDtype::F32 => unreachable!(),
        };
        if !actual.is_finite() || !gold.is_finite() {
            return Err("nonfinite numerical result".into());
        }
        max_error = max_error.max(f64::from((actual - gold).abs()));
        scale = scale.max(f64::from(gold.abs()));
    }
    if scale == 0.0 {
        return Err("all-zero PEDANTIC reference".into());
    }
    Ok(max_error / scale)
}

fn run_bits(
    runtime: &Runtime,
    fixture: &mut Fixture,
    cell: Cell,
    arm: Arm,
) -> Result<Vec<u16>, String> {
    fixture.output_mut(arm).reset(&runtime.ctx)?;
    launch(runtime, fixture, cell, arm)?;
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|e| format!("{} sync: {e:?}", arm.name()))?;
    let bits = fixture
        .output(arm)
        .active_and_guards(&runtime.ctx, arm.name())?;
    if bits.contains(&POISON_BITS) {
        return Err(format!("{} retained poison", arm.name()));
    }
    Ok(bits)
}

fn correctness(runtime: &Runtime, cell: Cell) -> Result<(), String> {
    validate_resource(runtime, cell)?;
    for exceptional in [false, true] {
        let mut fixture = Fixture::new(runtime, cell, exceptional)?;
        let expected = run_bits(runtime, &mut fixture, cell, Arm::Control)?;
        let candidate = run_bits(runtime, &mut fixture, cell, Arm::Candidate)?;
        if candidate != expected {
            return Err(format!(
                "{} candidate/control bits differ exceptional={exceptional}",
                cell.label
            ));
        }
        let reference = pedantic_reference(runtime, &fixture, cell)?;
        let error = normalized_error(&candidate, &reference, cell.dtype)?;
        let tolerance = if cell.dtype == WeightDtype::Bf16 {
            0.01
        } else {
            0.0025
        };
        if error > tolerance {
            return Err(format!(
                "{} normalized error {error} > {tolerance}",
                cell.label
            ));
        }
        for repeat in 0..3 {
            if run_bits(runtime, &mut fixture, cell, Arm::Candidate)? != expected {
                return Err(format!("{} eager repeat {repeat}", cell.label));
            }
        }
        let candidate_graph = graph(runtime, &fixture, cell, Arm::Candidate, 1)?;
        assert_candidate_graph(&candidate_graph, &fixture, cell)?;
        for replay in 0..3 {
            fixture.candidate.reset(&runtime.ctx)?;
            candidate_graph
                .launch()
                .map_err(|e| format!("{} graph {replay}: {e:?}", cell.label))?;
            let bits = fixture
                .candidate
                .active_and_guards(&runtime.ctx, cell.label)?;
            if bits != expected {
                return Err(format!("{} graph replay {replay}", cell.label));
            }
        }
        fixture.a.unchanged(&runtime.ctx, "A")?;
        fixture.b.unchanged(&runtime.ctx, "B")?;
        println!(
            "{{\"schema\":\"MambaBiFixedHalfBatchCorrectnessV1\",\"cell\":\"{}\",\"exceptional\":{},\"normalized_error\":{},\"tolerance\":{},\"eager_repeats\":3,\"graph_repeats\":3,\"passed\":true}}",
            cell.label, exceptional, error, tolerance
        );
    }
    Ok(())
}

fn event_us(
    runtime: &Runtime,
    mut launch: impl FnMut() -> Result<(), String>,
) -> Result<f64, String> {
    let start = runtime
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| format!("event start: {e:?}"))?;
    launch()?;
    let end = runtime
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| format!("event end: {e:?}"))?;
    let us = f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|e| format!("event elapsed: {e:?}"))?,
    ) * 1000.0
        / 20.0;
    if !us.is_finite() || us <= 0.0 {
        return Err(format!("invalid timing {us}"));
    }
    Ok(us)
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn timed_pair(
    runtime: &Runtime,
    fixture: &Fixture,
    cell: Cell,
    comparator: Arm,
    path: &str,
    start_parity: usize,
) -> Result<(f64, f64), String> {
    let candidate_graph = if path == "graph" {
        Some(graph(runtime, fixture, cell, Arm::Candidate, 20)?)
    } else {
        None
    };
    let comparator_graph = if path == "graph" {
        Some(graph(runtime, fixture, cell, comparator, 20)?)
    } else {
        None
    };
    let run = |arm: Arm| -> Result<(), String> {
        if path == "graph" {
            if arm == Arm::Candidate {
                candidate_graph
                    .as_ref()
                    .unwrap()
                    .launch()
                    .map_err(|e| format!("candidate graph: {e:?}"))
            } else {
                comparator_graph
                    .as_ref()
                    .unwrap()
                    .launch()
                    .map_err(|e| format!("comparator graph: {e:?}"))
            }
        } else {
            for _ in 0..20 {
                launch(runtime, fixture, cell, arm)?;
            }
            Ok(())
        }
    };
    for _ in 0..64 {
        launch(runtime, fixture, cell, Arm::Candidate)?;
        launch(runtime, fixture, cell, comparator)?;
    }
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|e| format!("warmup: {e:?}"))?;
    let mut ratios = Vec::new();
    for window in 0..7 {
        let abba = (window + start_parity) & 1 == 0;
        let order = if abba { "abba" } else { "baab" };
        let sequence = if abba {
            [Arm::Candidate, comparator, comparator, Arm::Candidate]
        } else {
            [comparator, Arm::Candidate, Arm::Candidate, comparator]
        };
        let mut raw = Vec::new();
        for arm in sequence {
            raw.push((arm, event_us(runtime, || run(arm))?));
        }
        let candidate = raw
            .iter()
            .filter(|x| x.0 == Arm::Candidate)
            .map(|x| x.1)
            .sum::<f64>()
            / 2.0;
        let control = raw
            .iter()
            .filter(|x| x.0 == comparator)
            .map(|x| x.1)
            .sum::<f64>()
            / 2.0;
        ratios.push(candidate / control);
        println!(
            "{{\"schema\":\"MambaBiFixedHalfBatchTimingV1\",\"cell\":\"{}\",\"comparator\":\"{}\",\"path\":\"{}\",\"start_parity\":{},\"order\":\"{}\",\"window\":{},\"operations\":20,\"raw_us\":{:?},\"ratio\":{}}}",
            cell.label,
            comparator.name(),
            path,
            start_parity,
            order,
            window,
            raw.iter().map(|x| x.1).collect::<Vec<_>>(),
            candidate / control
        );
    }
    Ok((quantile(&ratios, 0.5), quantile(&ratios, 0.95)))
}

fn screen(runtime: &Runtime, cell: Cell) -> Result<(), String> {
    let fixture = Fixture::new(runtime, cell, false)?;
    let mut own = Vec::new();
    let mut fast = Vec::new();
    for path in ["eager", "graph"] {
        for start in 0..2 {
            own.push(timed_pair(
                runtime,
                &fixture,
                cell,
                Arm::Control,
                path,
                start,
            )?);
            fast.push(timed_pair(runtime, &fixture, cell, Arm::Fast, path, start)?);
        }
    }
    let worst_median = own.iter().map(|x| x.0).max_by(f64::total_cmp).unwrap();
    let worst_p95 = own.iter().map(|x| x.1).max_by(f64::total_cmp).unwrap();
    let threshold = if cell.function < 2 { 0.985 } else { 0.97 };
    let advance = worst_median < 1.0 && worst_p95 <= 1.02 && worst_median <= threshold;
    let fast_groups = fast
        .iter()
        .map(|(median, p95)| format!("[{median},{p95}]"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"schema\":\"MambaBiFixedHalfBatchScreenV1\",\"cell\":\"{}\",\"windows\":7,\"worst_auto_median\":{},\"worst_auto_p95\":{},\"advance_threshold\":{},\"advance\":{},\"fast_groups\":[{}]}}",
        cell.label, worst_median, worst_p95, threshold, advance, fast_groups
    );
    Ok(())
}

#[test]
#[ignore = "bounded CUDA13.2/CC8.9 three-arm half correctness and short7 screen"]
fn fixed_half_three_arm_batch_discovery() {
    let runtime = new_runtime().expect("half batch runtime");
    let mut accepted = Vec::new();
    for cell in CELLS {
        match correctness(&runtime, cell) {
            Ok(()) => accepted.push(cell),
            Err(error) => println!(
                "{{\"schema\":\"MambaBiFixedHalfBatchRejectV1\",\"cell\":\"{}\",\"phase\":\"correctness_resource\",\"error\":{:?}}}",
                cell.label, error
            ),
        }
    }
    for cell in accepted {
        if let Err(error) = screen(&runtime, cell) {
            println!(
                "{{\"schema\":\"MambaBiFixedHalfBatchRejectV1\",\"cell\":\"{}\",\"phase\":\"timing\",\"error\":{:?}}}",
                cell.label, error
            );
        }
    }
}
