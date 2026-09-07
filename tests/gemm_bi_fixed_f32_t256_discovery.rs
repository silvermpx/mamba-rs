#![cfg(feature = "cuda")]

use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, sys,
};
use mamba_rs::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_fixed::{
    FixedFwdOperands, FixedShape, FixedTile, fixed_forward, fixed_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::kernel_identity::{
    NUMERIC_ABI_REVISION, SCHEDULE_REVISION, TUNING_TABLE_REVISION, digest_hex,
};
use sha2::{Digest as _, Sha256};

const SM120_SOURCE: &str = include_str!("../kernels/gemm_bi_fixed/sm120_f32_n64_copyplan.cu");
const T256_START: &str =
    "// Force-only T256 CopyPlan twin: four async vectors and a 4x4 microtile per thread.";
const T256_END: &str =
    "// Force-only M128N64 T256 CopyPlan twin: four A and two B vectors per thread.";
const EXPECTED_SOURCE_SHA: &str =
    "5015fbfcd457e92759f2f37a9093175d7e37c941fdd1d8d91258fedb06dddb84";
const EXPECTED_FRAGMENT_SHA: &str =
    "8db567604e13ff8afba012235b59b58c6579b5c4857be4bb7164dbb639a38541";
const SYMBOL: &str = "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_t256_discovery_v1";
const OLD_SYMBOL: &str = "gemm_bi_nn_fixed_sm120_f32_n64_copyplan_t256_v1";
const GUARD: usize = 32;
const GUARD_BITS: u32 = 0x7fc0_f320;
const POISON_BITS: u32 = 0x7fc0_2560;

fn t256_fragment() -> &'static str {
    assert_eq!(SM120_SOURCE.matches(T256_START).count(), 1);
    assert_eq!(SM120_SOURCE.matches(T256_END).count(), 1);
    let start = SM120_SOURCE.find(T256_START).unwrap();
    let end = SM120_SOURCE.find(T256_END).unwrap();
    &SM120_SOURCE[start..end]
}

fn a_copy(thread: usize, slice: usize) -> (usize, usize) {
    let linear = thread + slice * 256;
    (linear / 8, (linear % 8) * 4)
}

fn b_copy(thread: usize, slice: usize) -> (usize, usize) {
    let linear = thread + slice * 256;
    (linear / 16, (linear % 16) * 4)
}

fn outputs(thread: usize) -> Vec<(usize, usize)> {
    let tx = thread & 15;
    let ty = thread >> 4;
    let mut result = Vec::with_capacity(16);
    for i in 0..4 {
        for j in 0..4 {
            result.push((ty * 4 + i, tx * 4 + j));
        }
    }
    result
}

fn compose_source() -> String {
    let prelude = include_str!("../kernels/_typed_prelude.cuh");
    let common = include_str!("../kernels/gemm_bi_fixed/common.cuh")
        .lines()
        .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
        .collect::<Vec<_>>()
        .join("\n");
    let params = r#"
struct FixedSm120ExactF32Params {
    float alpha, beta;
    int m, n, k, lda, ldb, ldc;
};
static_assert(sizeof(FixedSm120ExactF32Params) == 32, "exact N64 parameter size");
static_assert(alignof(FixedSm120ExactF32Params) == 4, "exact N64 parameter alignment");
"#;
    let candidate = t256_fragment().replace(OLD_SYMBOL, SYMBOL);
    [prelude, &common, params, &candidate].join("\n")
}

#[test]
fn f32_t256_extracts_the_pinned_existing_body() {
    assert_eq!(
        format!("{:x}", Sha256::digest(SM120_SOURCE)),
        EXPECTED_SOURCE_SHA
    );
    let fragment = t256_fragment();
    assert_eq!(
        format!("{:x}", Sha256::digest(fragment)),
        EXPECTED_FRAGMENT_SHA
    );
    assert_eq!(fragment.matches("extern \"C\" __global__").count(), 1);
    assert!(fragment.contains("__launch_bounds__(SM120_EXACT_N64_CP_T256_THREADS, 3)"));
    assert!(fragment.contains("float acc[4][4]"));
    assert!(fragment.contains("for (int kk = 0; kk < SM120_EXACT_N64_CP_T256_BK; kk++)"));
    assert!(!fragment.contains("mma.sync"));
    assert!(!fragment.contains("atomic"));
}

#[test]
fn f32_t256_copy_plan_covers_each_vector_once() {
    let mut a = BTreeSet::new();
    let mut b = BTreeSet::new();
    for slice in 0..2 {
        for thread in 0..256 {
            assert!(a.insert(a_copy(thread, slice)));
            assert!(b.insert(b_copy(thread, slice)));
        }
    }
    assert_eq!(a.len(), 64 * 8);
    assert_eq!(a.first(), Some(&(0, 0)));
    assert_eq!(a.last(), Some(&(63, 28)));
    assert_eq!(b.len(), 32 * 16);
    assert_eq!(b.first(), Some(&(0, 0)));
    assert_eq!(b.last(), Some(&(31, 60)));
}

#[test]
fn f32_t256_threads_partition_the_output() {
    let mut all = BTreeSet::new();
    for thread in 0..256 {
        let owned = outputs(thread);
        assert_eq!(owned.len(), 16);
        for coordinate in owned {
            assert!(coordinate.0 < 64 && coordinate.1 < 64);
            assert!(all.insert(coordinate));
        }
    }
    assert_eq!(all.len(), 64 * 64);
}

#[test]
#[ignore = "bounded CUDA13.2/CC8.9 exact-F32 T256 correctness discovery"]
fn f32_t256_small_bits_prefix_tail_and_resources() {
    run_correctness().unwrap();
}

#[test]
#[ignore = "bounded CUDA13.2/CC8.9 exact-F32 T256 paired discovery"]
fn f32_t256_b0_paired_discovery() {
    run_timing().unwrap();
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Params {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe impl DeviceRepr for Params {}

const _: () = {
    assert!(std::mem::size_of::<Params>() == 32);
    assert!(std::mem::align_of::<Params>() == 4);
};

#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    shape: FixedShape,
    bias: bool,
}

const B0: Case = Case {
    label: "b0",
    shape: FixedShape {
        m: 4621,
        k: 768,
        n: 2304,
    },
    bias: false,
};

const TAIL0: Case = Case {
    label: "tail0",
    shape: FixedShape {
        m: 65,
        k: 36,
        n: 68,
    },
    bias: false,
};

#[derive(Clone, Copy, Debug)]
enum Arm {
    Candidate,
    Current,
    Fast,
}

impl Arm {
    const fn name(self) -> &'static str {
        match self {
            Self::Candidate => "candidate_t256",
            Self::Current => "current_copyplan",
            Self::Fast => "fast_tf32",
        }
    }
}

struct Runtime {
    _device: GpuDevice,
    ctx: GpuCtx,
    _module: Arc<CudaModule>,
    candidate: CudaFunction,
    source_sha: String,
    ptx_sha: String,
}

struct Guarded {
    buffer: GpuBuffer,
    baseline: Vec<f32>,
    offset: usize,
    len: usize,
}

impl Guarded {
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

    fn bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
        let values = self.buffer.to_cpu(stream)?;
        if values[..self.offset]
            .iter()
            .chain(&values[self.offset + self.len..])
            .any(|v| v.to_bits() != GUARD_BITS)
        {
            return Err(format!("{label} changed an output guard"));
        }
        Ok(values[self.offset..self.offset + self.len]
            .iter()
            .map(|v| v.to_bits())
            .collect())
    }

    fn unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
        let values = self.buffer.to_cpu(stream)?;
        if values
            .iter()
            .zip(&self.baseline)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(format!("{label} or its guards changed"));
        }
        Ok(())
    }
}

struct Fixture {
    a: Guarded,
    b: Guarded,
    bias: Guarded,
    candidate: Guarded,
    current: Guarded,
    fast: Guarded,
}

fn finite_values(len: usize, mut state: u64) -> Vec<f32> {
    let explicit = [
        0u32,
        0x8000_0000,
        1,
        0x8000_0001,
        0x3f80_0001,
        0xbf80_0001,
        0x3f80_1000,
        0xbf80_1000,
    ];
    (0..len)
        .map(|i| {
            if i < explicit.len() {
                return f32::from_bits(explicit[i]);
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sign = ((state >> 63) as u32) << 31;
            let exponent = (((state >> 23) % 250) as u32 + 2) << 23;
            f32::from_bits(sign | exponent | (state as u32 & 0x7f_ffff))
                * f32::from_bits(0x0380_0000)
        })
        .collect()
}

impl Fixture {
    fn new(runtime: &Runtime, shape: FixedShape) -> Result<Self, String> {
        let output = vec![f32::from_bits(POISON_BITS); shape.m * shape.n];
        Ok(Self {
            a: Guarded::new(
                &runtime.ctx.stream,
                finite_values(shape.m * shape.k, 0xa256_1),
            )?,
            b: Guarded::new(
                &runtime.ctx.stream,
                finite_values(shape.k * shape.n, 0xb256_2),
            )?,
            bias: Guarded::new(&runtime.ctx.stream, finite_values(shape.n, 0xb1a5_256))?,
            candidate: Guarded::new(&runtime.ctx.stream, output.clone())?,
            current: Guarded::new(&runtime.ctx.stream, output.clone())?,
            fast: Guarded::new(&runtime.ctx.stream, output)?,
        })
    }

    fn output(&self, arm: Arm) -> &Guarded {
        match arm {
            Arm::Candidate => &self.candidate,
            Arm::Current => &self.current,
            Arm::Fast => &self.fast,
        }
    }
    fn output_mut(&mut self, arm: Arm) -> &mut Guarded {
        match arm {
            Arm::Candidate => &mut self.candidate,
            Arm::Current => &mut self.current,
            Arm::Fast => &mut self.fast,
        }
    }
    fn reset(&mut self, runtime: &Runtime, arm: Arm) -> Result<(), String> {
        self.output_mut(arm).reset(&runtime.ctx.stream)
    }
    fn operands(&self, runtime: &Runtime, case: Case, arm: Arm) -> FixedFwdOperands {
        let typed = |ptr| TypedPtr {
            ptr,
            dtype: WeightDtype::F32,
        };
        FixedFwdOperands {
            c: typed(self.output(arm).ptr(&runtime.ctx.stream)),
            x: typed(self.a.ptr(&runtime.ctx.stream)),
            w: typed(self.b.ptr(&runtime.ctx.stream)),
            bias_ptr: case.bias.then(|| self.bias.ptr(&runtime.ctx.stream)),
        }
    }
    fn validate_inputs(&self, runtime: &Runtime) -> Result<(), String> {
        self.a.unchanged(&runtime.ctx.stream, "A")?;
        self.b.unchanged(&runtime.ctx.stream, "B")?;
        self.bias.unchanged(&runtime.ctx.stream, "bias")
    }
}

fn new_runtime() -> Result<Runtime, String> {
    if std::env::var("MAMBA_FIXED_F32_T256_DISCOVERY").as_deref() != Ok("1") {
        return Err("set MAMBA_FIXED_F32_T256_DISCOVERY=1".into());
    }
    let device = GpuDevice::new(0)?;
    if device.compute_capability != (8, 9) || device.multiprocessor_count() != 142 {
        return Err(format!(
            "requires 142-SM CC8.9, found {:?}/{}",
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
        || compiler.numeric_abi_revision != 5
        || compiler.schedule_revision != 8
        || TUNING_TABLE_REVISION != 44
        || NUMERIC_ABI_REVISION != 5
        || SCHEDULE_REVISION != 8
        || artifact.compile_key != compiler.invocation_digest
    {
        return Err(format!("lost frozen identity: {compiler:?} {artifact:?}"));
    }
    let source = compose_source();
    let source_sha = format!("{:x}", Sha256::digest(source.as_bytes()));
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
        .map_err(|e| format!("compile T256: {e:?}"))?;
    let ptx_source = ptx.to_src();
    let ptx_sha = format!("{:x}", Sha256::digest(ptx_source.as_bytes()));
    let module = device
        .context()
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
        .map_err(|e| format!("load T256: {e:?}"))?;
    let candidate = module
        .load_function(SYMBOL)
        .map_err(|e| format!("load {SYMBOL}: {e:?}"))?;
    let runtime = Runtime {
        _device: device,
        ctx,
        _module: module,
        candidate,
        source_sha,
        ptx_sha,
    };
    validate_resources(&runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedF32T256IdentityV1\",\"fixed_source_sha\":\"{}\",\"fixed_invocation_sha\":\"{}\",\"fixed_artifact_sha\":\"{}\",\"candidate_source_sha\":\"{}\",\"candidate_ptx_sha\":\"{}\"}}",
        digest_hex(&compiler.source_digest),
        digest_hex(&compiler.invocation_digest),
        digest_hex(&artifact.artifact_digest),
        runtime.source_sha,
        runtime.ptx_sha
    );
    Ok(runtime)
}

fn validate_resources(runtime: &Runtime) -> Result<(), String> {
    let regs = runtime
        .candidate
        .num_regs()
        .map_err(|e| format!("regs: {e:?}"))?;
    let local = runtime
        .candidate
        .local_size_bytes()
        .map_err(|e| format!("local: {e:?}"))?;
    let shared = runtime
        .candidate
        .shared_size_bytes()
        .map_err(|e| format!("shared: {e:?}"))?;
    let occupancy = runtime
        .candidate
        .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)
        .map_err(|e| format!("occupancy: {e:?}"))?;
    if regs <= 0 || regs > 85 || local != 0 || shared != 32_768 || occupancy != 3 {
        return Err(format!(
            "resource gate regs={regs} local={local} shared={shared} occupancy={occupancy}"
        ));
    }
    println!(
        "{{\"schema\":\"MambaBiFixedF32T256ResourceV1\",\"registers\":{regs},\"local_bytes\":{local},\"static_shared_bytes\":{shared},\"occupancy_blocks_per_sm\":{occupancy}}}"
    );
    Ok(())
}

fn configure(runtime: &Runtime, arm: Arm) -> Result<(), String> {
    runtime.ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
    runtime.ctx.set_bi_tensor_cores(false);
    runtime
        .ctx
        .set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
    runtime.ctx.set_batch_invariant(!matches!(arm, Arm::Fast));
    runtime.ctx.set_fast_gemm(matches!(arm, Arm::Fast));
    if matches!(arm, Arm::Fast) && !runtime.ctx.tf32() {
        return Err("FAST_TF32 disabled".into());
    }
    Ok(())
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
            let p = Params {
                alpha: 1.0,
                beta: 0.0,
                m: case.shape.m as i32,
                n: case.shape.n as i32,
                k: case.shape.k as i32,
                lda: case.shape.k as i32,
                ldb: case.shape.n as i32,
                ldc: case.shape.n as i32,
            };
            let blocks = case.shape.m.div_ceil(64) * case.shape.n.div_ceil(64);
            let cfg = LaunchConfig {
                grid_dim: (u32::try_from(blocks).map_err(|_| "grid u32")?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = runtime.ctx.stream.launch_builder(&runtime.candidate);
            builder.arg(&output);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&bias);
            builder.arg(&p);
            unsafe { builder.launch(cfg) }
                .map(|_| ())
                .map_err(|e| format!("candidate launch: {e:?}"))
        }
        Arm::Current => fixed_forward_with_tile(
            &runtime.ctx,
            operands,
            case.shape,
            FixedTile::F32Sm89N64CopyPlan,
        ),
        Arm::Fast => gpu_gemm_typed_forward_raw(
            &runtime.ctx,
            operands.c,
            operands.x,
            operands.w,
            operands.bias_ptr,
            (case.shape.m, case.shape.k, case.shape.n),
        ),
    }
}

fn run_bits(
    runtime: &Runtime,
    fixture: &mut Fixture,
    case: Case,
    arm: Arm,
) -> Result<Vec<u32>, String> {
    fixture.reset(runtime, arm)?;
    launch(runtime, fixture, case, arm)?;
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let bits = fixture.output(arm).bits(&runtime.ctx.stream, arm.name())?;
    if bits.iter().any(|x| *x == POISON_BITS) {
        return Err(format!("{} retained poison", arm.name()));
    }
    Ok(bits)
}

fn check_case(runtime: &Runtime, case: Case) -> Result<Vec<u32>, String> {
    let mut fixture = Fixture::new(runtime, case.shape)?;
    let current = run_bits(runtime, &mut fixture, case, Arm::Current)?;
    let candidate = run_bits(runtime, &mut fixture, case, Arm::Candidate)?;
    if candidate != current {
        return Err(format!("{} bits differ", case.label));
    }
    for _ in 0..2 {
        if run_bits(runtime, &mut fixture, case, Arm::Candidate)? != current {
            return Err(format!("{} repeat differs", case.label));
        }
    }
    fixture.validate_inputs(runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedF32T256BitsV1\",\"case\":\"{}\",\"shape\":[{},{},{}],\"bias\":{},\"passed\":true}}",
        case.label, case.shape.m, case.shape.k, case.shape.n, case.bias
    );
    Ok(current)
}

fn run_correctness() -> Result<(), String> {
    let runtime = new_runtime()?;
    let mut auto_fixture = Fixture::new(&runtime, B0.shape)?;
    auto_fixture.reset(&runtime, Arm::Current)?;
    configure(&runtime, Arm::Current)?;
    let o = auto_fixture.operands(&runtime, B0, Arm::Current);
    let selected = fixed_forward(
        &runtime.ctx,
        o.c,
        o.x,
        o.w,
        o.bias_ptr,
        (B0.shape.m, B0.shape.k, B0.shape.n),
    )?;
    if selected != FixedTile::F32Sm89N64CopyPlan {
        return Err(format!("AUTO selected {selected:?}"));
    }
    let full = check_case(&runtime, B0)?;
    let mut prefix = Fixture::new(&runtime, B0.shape)?;
    let one = Case {
        label: "prefix_m1",
        shape: FixedShape { m: 1, ..B0.shape },
        bias: false,
    };
    for arm in [Arm::Current, Arm::Candidate] {
        prefix.reset(&runtime, arm)?;
        launch(&runtime, &prefix, one, arm)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("prefix sync:{e:?}"))?;
        let bits = prefix.output(arm).bits(&runtime.ctx.stream, arm.name())?;
        if bits[..B0.shape.n] != full[..B0.shape.n]
            || bits[B0.shape.n..].iter().any(|x| *x != POISON_BITS)
        {
            return Err(format!("{} prefix differs", arm.name()));
        }
    }
    for bias in [false, true] {
        check_case(
            &runtime,
            Case {
                label: if bias { "tail1" } else { "tail0" },
                bias,
                ..TAIL0
            },
        )?;
    }
    prefix.validate_inputs(&runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedF32T256CorrectnessCompleteV1\",\"cases\":4,\"passed\":true}}"
    );
    Ok(())
}

fn measure(runtime: &Runtime, fixture: &Fixture, arm: Arm, iters: usize) -> Result<f64, String> {
    let start = runtime
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| format!("event:{e:?}"))?;
    for _ in 0..iters {
        launch(runtime, fixture, B0, arm)?;
    }
    let end = runtime
        .ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| format!("event:{e:?}"))?;
    Ok(f64::from(
        start
            .elapsed_ms(&end)
            .map_err(|e| format!("elapsed:{e:?}"))?,
    ) * 1000.0
        / iters as f64)
}

fn pct(v: &[f64], q: f64) -> f64 {
    let mut x = v.to_vec();
    x.sort_by(f64::total_cmp);
    x[((x.len() - 1) as f64 * q).round() as usize]
}

fn compare(
    runtime: &Runtime,
    fixture: &Fixture,
    other: Arm,
    windows: usize,
) -> Result<(f64, f64), String> {
    for _ in 0..64 {
        launch(runtime, fixture, B0, Arm::Candidate)?;
        launch(runtime, fixture, B0, other)?;
    }
    runtime
        .ctx
        .stream
        .synchronize()
        .map_err(|e| format!("warm:{e:?}"))?;
    let ci = (5000.0 / measure(runtime, fixture, Arm::Candidate, 8)?)
        .round()
        .clamp(1.0, 4096.0) as usize;
    let oi = (5000.0 / measure(runtime, fixture, other, 8)?)
        .round()
        .clamp(1.0, 4096.0) as usize;
    let mut all = Vec::new();
    for (order, cf) in [("abba", true), ("baab", false)] {
        let mut cs = Vec::new();
        let mut os = Vec::new();
        let mut rs = Vec::new();
        for _ in 0..windows {
            let (c1, o1, o2, c2) = if cf {
                (
                    measure(runtime, fixture, Arm::Candidate, ci)?,
                    measure(runtime, fixture, other, oi)?,
                    measure(runtime, fixture, other, oi)?,
                    measure(runtime, fixture, Arm::Candidate, ci)?,
                )
            } else {
                let o1 = measure(runtime, fixture, other, oi)?;
                let c1 = measure(runtime, fixture, Arm::Candidate, ci)?;
                let c2 = measure(runtime, fixture, Arm::Candidate, ci)?;
                let o2 = measure(runtime, fixture, other, oi)?;
                (c1, o1, o2, c2)
            };
            let c = (c1 + c2) * 0.5;
            let o = (o1 + o2) * 0.5;
            cs.push(c);
            os.push(o);
            rs.push(c / o);
            all.push(c / o);
        }
        println!(
            "{{\"schema\":\"MambaBiFixedF32T256TimingV1\",\"comparator\":\"{}\",\"order\":\"{}\",\"windows\":{},\"candidate_iterations\":{},\"comparator_iterations\":{},\"candidate_p50_us\":{},\"comparator_p50_us\":{},\"ratio_p50\":{},\"ratio_p95\":{},\"candidate_samples_us\":{:?},\"comparator_samples_us\":{:?},\"ratios\":{:?}}}",
            other.name(),
            order,
            windows,
            ci,
            oi,
            pct(&cs, 0.5),
            pct(&os, 0.5),
            pct(&rs, 0.5),
            pct(&rs, 0.95),
            cs,
            os,
            rs
        );
    }
    Ok((pct(&all, 0.5), pct(&all, 0.95)))
}

fn run_timing() -> Result<(), String> {
    let windows = std::env::var("MAMBA_FIXED_F32_T256_WINDOWS")
        .unwrap_or_else(|_| "7".into())
        .parse::<usize>()
        .map_err(|e| format!("windows:{e}"))?;
    if !matches!(windows, 7 | 21) {
        return Err("windows must be7or21".into());
    }
    let runtime = new_runtime()?;
    let fixture = Fixture::new(&runtime, B0.shape)?;
    let own = compare(&runtime, &fixture, Arm::Current, windows)?;
    let fast = compare(&runtime, &fixture, Arm::Fast, windows)?;
    fixture.validate_inputs(&runtime)?;
    println!(
        "{{\"schema\":\"MambaBiFixedF32T256CompleteV1\",\"windows\":{},\"own_p50\":{},\"own_p95\":{},\"fast_p50\":{},\"fast_p95\":{},\"own_win\":{},\"passed\":true}}",
        windows,
        own.0,
        own.1,
        fast.0,
        fast.1,
        own.0 < 1.0 && own.1 < 1.0
    );
    Ok(())
}
