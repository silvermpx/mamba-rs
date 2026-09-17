//! Separate SM120 exact-N64 production paired qualification.
//! Helper semantics are copied from the frozen Ada wrapper; no Ada source is changed.
#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::process::Command;
use std::thread;
use std::time::Duration;

use cudarc::driver::{CudaGraph, PushKernelArg};
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::blas::TypedPtr;
use mamba_rs::mamba_ssm::gpu::buffers::DtypedBuf;
use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::gemm_bi_inference::{
    InferenceFwdOperands, InferenceShape, InferenceTile, inference_forward,
    inference_forward_f32_legacy_baseline, inference_forward_with_tile,
};
use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{TUNING_TABLE_REVISION, digest_hex};

#[path = "support/fixed_sm120_exact_n64_admission.rs"]
mod exact_n64_admission;

#[test]
#[ignore = "requires exclusive SM120, actual NVRTC candidate admission and observed-unlocked paired evidence"]
fn fixed_sm120_exact_n64_paired_admission() {
    exact_n64_admission::run();
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

fn sm120_exact_json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn sm120_exact_environment_preflight(label: &str) -> Result<(), String> {
    let applications = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|error| format!("run compute-application preflight: {error}"))?;
    if !applications.status.success() {
        return Err(format!(
            "compute-application preflight exited with {}",
            applications.status
        ));
    }
    let own_pid = std::process::id();
    let competing_pids = String::from_utf8_lossy(&applications.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .filter(|&pid| pid != own_pid)
        .collect::<Vec<_>>();
    if !competing_pids.is_empty() {
        return Err(format!(
            "competing compute applications: {competing_pids:?}"
        ));
    }

    let mut last_snapshot = None;
    for _ in 0..50 {
        let snapshot = Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,clocks.sm,temperature.gpu,pstate",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|error| format!("run GPU telemetry preflight: {error}"))?;
        if !snapshot.status.success() {
            return Err(format!(
                "GPU telemetry preflight exited with {}",
                snapshot.status
            ));
        }
        let line = String::from_utf8_lossy(&snapshot.stdout)
            .lines()
            .next()
            .ok_or("GPU telemetry preflight returned no rows")?
            .to_owned();
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 6 {
            return Err(format!(
                "GPU telemetry preflight returned malformed row {line:?}"
            ));
        }
        let gpu_util = fields[0]
            .parse::<u32>()
            .map_err(|error| format!("parse GPU utilization {:?}: {error}", fields[0]))?;
        let memory_util = fields[1]
            .parse::<u32>()
            .map_err(|error| format!("parse memory utilization {:?}: {error}", fields[1]))?;
        let used_mib = fields[2]
            .parse::<u32>()
            .map_err(|error| format!("parse used memory {:?}: {error}", fields[2]))?;
        let sm_clock_mhz = fields[3]
            .parse::<u32>()
            .map_err(|error| format!("parse SM clock {:?}: {error}", fields[3]))?;
        let temperature_c = fields[4]
            .parse::<u32>()
            .map_err(|error| format!("parse GPU temperature {:?}: {error}", fields[4]))?;
        let pstate = fields[5];
        last_snapshot = Some(format!(
            "gpu_util={gpu_util}% memory_util={memory_util}% used={used_mib}MiB sm_clock={sm_clock_mhz}MHz temperature={temperature_c}C pstate={pstate}"
        ));
        if gpu_util <= 1 && memory_util <= 1 && sm_clock_mhz > 0 && temperature_c > 0 {
            eprintln!(
                "SM120 exact-N64 preflight {label}: {} (the process's own allocations are excluded from the launch-time <=128 MiB gate)",
                last_snapshot.as_deref().unwrap_or("telemetry unavailable")
            );
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(format!(
        "GPU did not return to <=1% compute and memory utilization; last snapshot: {}",
        last_snapshot.as_deref().unwrap_or("none")
    ))
}

fn configure_sm120_exact_custom(ctx: &GpuCtx, policy: F32TriadPolicy) {
    ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
    ctx.route_controls().set_family(BiGemmFamily::Inference);
    ctx.route_controls().set_tensor_cores(false);
    ctx.route_controls().set_f32_policy(policy);
}

fn launch_sm120_exact_auto(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
) -> InferenceTile {
    inference_forward(
        ctx,
        operands.c,
        operands.x,
        operands.w,
        operands.bias_ptr,
        (shape.m, shape.k, shape.n),
    )
    .expect("production Fixed AUTO launch")
}

fn sm120_exact_vendor_launch(
    ctx: &GpuCtx,
    operands: InferenceFwdOperands,
    shape: InferenceShape,
    compute: cudarc::cublas::sys::cublasComputeType_t,
) {
    use std::ffi::{c_int, c_void};

    let beta = if let Some(bias_ptr) = operands.bias_ptr {
        // Match the production vendor epilogue, including its timed broadcast.
        // A half output rounds the broadcast bias to its storage dtype before
        // GEMM; the independent reference below keeps its output in F32.
        let kernel = match operands.c.dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => &ctx.kernels.bias_broadcast,
            dtype => ctx.kernels.bias_broadcast_typed.get(dtype),
        };
        let rows = shape.m as c_int;
        let cols = shape.n as c_int;
        let mut launch = ctx.stream.launch_builder(kernel);
        launch.arg(&operands.c.ptr);
        launch.arg(&bias_ptr);
        launch.arg(&rows);
        launch.arg(&cols);
        unsafe { launch.launch(mamba_rs::mamba_ssm::gpu::launch::grid_1d(shape.m * shape.n)) }
            .expect("SM120 exact vendor bias broadcast");
        1.0f32
    } else {
        0.0f32
    };
    let alpha = 1.0f32;
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            shape.n as c_int,
            shape.m as c_int,
            shape.k as c_int,
            &alpha as *const f32 as *const c_void,
            operands.w.ptr as *const c_void,
            operands.w.dtype.cuda_data_type(),
            shape.n as c_int,
            operands.x.ptr as *const c_void,
            operands.x.dtype.cuda_data_type(),
            shape.k as c_int,
            &beta as *const f32 as *const c_void,
            operands.c.ptr as *mut c_void,
            operands.c.dtype.cuda_data_type(),
            shape.n as c_int,
            compute,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .expect("SM120 exact explicit-compute vendor GEMM");
    }
}

fn sm120_exact_event_window_us(ctx: &GpuCtx, iterations: usize, mut launch: impl FnMut()) -> f64 {
    let start = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("SM120 exact paired window start");
    for _ in 0..iterations {
        launch();
    }
    let end = ctx
        .stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .expect("SM120 exact paired window end");
    let elapsed = f64::from(
        start
            .elapsed_ms(&end)
            .expect("SM120 exact paired event timing"),
    ) * 1000.0
        / iterations as f64;
    assert!(
        elapsed.is_finite() && elapsed > 0.0,
        "invalid SM120 exact timing: {elapsed}"
    );
    elapsed
}
