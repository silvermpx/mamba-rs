//! Shared CUDA Graph capture helper.
//!
//! Wraps the begin_capture / run-body / end_capture dance in a single
//! function so call sites don't repeat the ~25 lines of error-handling
//! boilerplate (pre-sync, end-on-error to restore stream, ok_or on the
//! end_capture Option).
//!
//! Used by every captured pipeline in the crate:
//! - M1 inference (`inference::GpuInferenceEngine::capture_graph`)
//! - M3 mixed + mixed-native inference (`inference::GpuInferenceMixed*::capture_graph*`)
//! - M1 training step (`training_graph::GpuMambaTrainingStepGraph::capture`)
//!
//! ## Why pre-sync
//! Without it, a freshly-allocated buffer used inside the body can race
//! against an in-flight HtoD that was issued before capture started. The
//! "130m race lesson" — see `GpuMambaBackboneMixedActs::new`.
//!
//! ## Why end-on-error
//! `cuStreamBeginCapture` puts the stream into a sticky capture mode. If
//! the body errors and we return without `end_capture`, every subsequent
//! op on the stream silently fails. So we ALWAYS call `end_capture`,
//! discard its result on the error path, then propagate the body error.

use std::sync::Arc;

use cudarc::driver::{CudaGraph, CudaStream};

use super::blas::{BoundPhysicalGraphLaunches, PreparedPhysicalGraphPackage};
use super::context::{BiGemmFamily, GpuCtx};
use super::kernel_identity::{
    CapturedGemmGraphPlan, CapturedPhysicalGraphPlan, PhysicalCudaLaunchError,
    PreparedGemmCaptureManifest, RecordingPhysicalObserver, ResolvedPhysicalKernelLaunch,
    ResolvedPhysicalLaunchSet, finish_recording_physical_capture,
};

/// Capture all CUDA work issued by `body` on `stream` into a CUDA Graph.
///
/// Mirrors the pattern in `inference::GpuInferenceEngine::capture_graph`,
/// extracted so it doesn't need to be reimplemented per pipeline.
///
/// # Safety
///
/// Every allocation, module, function, library handle, workspace, context,
/// and stream observed by `body` must remain valid and at the same address
/// until the returned graph is destroyed and all launches have completed.
/// The caller must also synchronize before releasing any captured resource.
pub unsafe fn capture_into_graph<F>(stream: &Arc<CudaStream>, body: F) -> Result<CudaGraph, String>
where
    F: FnOnce() -> Result<(), String>,
{
    stream
        .synchronize()
        .map_err(|e| format!("pre-capture sync: {e:?}"))?;

    stream
        .begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .map_err(|e| format!("begin_capture: {e:?}"))?;

    let body_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));

    // ALWAYS end capture, even on body error — a stream stuck in capture
    // mode silently breaks every subsequent op.
    let end_result = stream.end_capture(
        cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );

    // Combine both error paths: if BOTH body and end_capture failed, we want
    // the caller to see both — otherwise an `?`-shortcircuit on body_result
    // would silently drop a stream-corrupting end_capture failure.
    let body_result = match body_result {
        Ok(result) => result,
        Err(payload) => {
            drop(end_result);
            std::panic::resume_unwind(payload);
        }
    };
    match (body_result, end_result) {
        (Ok(()), Ok(Some(g))) => {
            // Pre-upload the instantiated graph so the FIRST replay does
            // not pay the device-upload latency inside a timed region
            // (the M3 inference engine already did this; the trainers
            // inherited the un-uploaded variant).
            g.upload().map_err(|e| format!("graph upload: {e:?}"))?;
            Ok(g)
        }
        (Ok(()), Ok(None)) => Err("end_capture returned no graph (empty body?)".to_string()),
        (Ok(()), Err(e)) => Err(format!("end_capture: {e:?}")),
        (Err(b), Ok(_)) => Err(format!("body: {b}")),
        (Err(b), Err(e)) => Err(format!(
            "body: {b}; end_capture ALSO failed (stream may be in invalid state): {e:?}"
        )),
    }
}

pub(crate) unsafe fn capture_into_graph_with_gemm_plan<F>(
    ctx: &GpuCtx,
    route_capacity: usize,
    manifest: &PreparedGemmCaptureManifest,
    body: F,
) -> Result<(CudaGraph, Option<CapturedGemmGraphPlan>), String>
where
    F: FnOnce() -> Result<(), String>,
{
    ctx.ensure_gemm_usable()?;
    manifest.validate_capture_request(ctx.gemm_route(), route_capacity)?;
    let recording = ctx.begin_gemm_route_recording(route_capacity)?;
    let graph = unsafe { capture_into_graph(&ctx.stream, body) }?;
    let plan = recording.finish_against_manifest(manifest)?;
    Ok((graph, plan))
}

fn physical_capture_body_error(error: PhysicalCudaLaunchError) -> String {
    match error {
        #[cfg(test)]
        PhysicalCudaLaunchError::Prepared(error) => error.to_string(),
        PhysicalCudaLaunchError::Identity(error) => error,
        PhysicalCudaLaunchError::Driver(error) => {
            format!("prepared physical CUDA enqueue: {error:?}")
        }
    }
}

unsafe fn capture_prepared_physical_launches(
    stream: &Arc<CudaStream>,
    launches: &mut BoundPhysicalGraphLaunches<'_>,
    observer: &mut RecordingPhysicalObserver,
) -> Result<CudaGraph, String> {
    stream
        .synchronize()
        .map_err(|error| format!("pre-capture sync: {error:?}"))?;
    stream
        .begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .map_err(|error| format!("begin_capture: {error:?}"))?;

    let body_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        launches.enqueue(observer)
    }));
    let end_result = stream.end_capture(
        cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    let body_result = match body_result {
        Ok(result) => result,
        Err(payload) => {
            drop(end_result);
            std::panic::resume_unwind(payload);
        }
    };
    match (body_result, end_result) {
        (Ok(()), Ok(Some(graph))) => {
            graph
                .upload()
                .map_err(|error| format!("graph upload: {error:?}"))?;
            Ok(graph)
        }
        (Ok(()), Ok(None)) => Err("end_capture returned no graph for physical package".into()),
        (Ok(()), Err(error)) => Err(format!("end_capture: {error:?}")),
        (Err(body), Ok(_)) => Err(format!("body: {}", physical_capture_body_error(body))),
        (Err(body), Err(end)) => Err(format!(
            "body: {}; end_capture ALSO failed (stream may be in invalid state): {end:?}",
            physical_capture_body_error(body)
        )),
    }
}

pub(super) struct CapturedPhysicalGraph {
    graph: CudaGraph,
    plan: CapturedPhysicalGraphPlan,
}

impl CapturedPhysicalGraph {
    fn new(graph: CudaGraph, plan: CapturedPhysicalGraphPlan) -> Self {
        Self { graph, plan }
    }

    pub(super) fn nodes(&self) -> &[ResolvedPhysicalKernelLaunch] {
        self.plan.nodes()
    }

    pub(super) fn launches(&self) -> ResolvedPhysicalLaunchSet {
        self.plan.launches()
    }

    #[cfg(test)]
    pub(super) fn launch(&self, ctx: &GpuCtx, label: &str) -> Result<(), String> {
        self.plan.validate_replay(ctx, label)?;
        self.graph
            .launch()
            .map_err(|error| format!("{label}: launch physical graph: {error:?}"))
    }

    pub(super) fn measure_prevalidated(
        &self,
        ctx: &GpuCtx,
        iterations: usize,
        label: &str,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err(format!(
                "{label}: timed graph iteration count must be positive"
            ));
        }
        self.plan.validate_replay(ctx, label)?;
        let start = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("{label}: record graph start event: {error:?}"))?;
        let mut failure = None;
        for _ in 0..iterations {
            if let Err(error) = self.graph.launch() {
                failure = Some(error);
                break;
            }
        }
        let end = match ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        {
            Ok(end) => end,
            Err(error) => {
                return Err(physical_graph_timing_error(
                    ctx,
                    format!("{label}: record graph end event: {error:?}"),
                ));
            }
        };
        if let Some(error) = failure {
            return Err(physical_graph_timing_error(
                ctx,
                format!("{label}: launch physical graph: {error:?}"),
            ));
        }
        let elapsed = start
            .elapsed_ms(&end)
            .map(f64::from)
            .map_err(|error| format!("{label}: measure physical graph events: {error:?}"));
        elapsed.map_err(|primary| physical_graph_timing_error(ctx, primary))
    }
}

fn physical_graph_timing_error(ctx: &GpuCtx, primary: String) -> String {
    match ctx.stream.synchronize() {
        Ok(()) => primary,
        Err(cleanup) => format!("{primary}; cleanup synchronize failed: {cleanup:?}"),
    }
}

pub(super) unsafe fn capture_into_graph_with_physical_plan(
    mut package: PreparedPhysicalGraphPackage<'_>,
) -> Result<CapturedPhysicalGraph, String> {
    package.validate()?;
    let mut observer = package.take_observer()?;
    let ctx = package.context();
    let manifest = package.manifest();
    observer.validate_capture_start(ctx, manifest)?;
    let mut launches = package.bind_launches()?;
    package.validate()?;
    observer.validate_capture_start(ctx, manifest)?;
    ctx.freeze_graph_scratch();
    let graph_result =
        unsafe { capture_prepared_physical_launches(&ctx.stream, &mut launches, &mut observer) };
    drop(launches);
    #[cfg(test)]
    package.apply_post_capture_test_mutation();
    let binding_result = observer.validate_capture_binding(ctx);
    let graph = match (graph_result, binding_result) {
        (Ok(graph), Ok(())) => graph,
        (Err(capture), Ok(())) => return Err(capture),
        (Ok(_), Err(binding)) => return Err(binding),
        (Err(capture), Err(binding)) => {
            return Err(format!(
                "{capture}; physical graph post-capture validation also failed: {binding}"
            ));
        }
    };
    let plan = finish_recording_physical_capture(observer, ctx, manifest)?;
    Ok(CapturedPhysicalGraph::new(graph, plan))
}

pub(crate) fn require_f32_triad_graph_plan(
    ctx: &GpuCtx,
    logical_f32: bool,
    plan: Option<&CapturedGemmGraphPlan>,
    label: &str,
) -> Result<(), String> {
    if ctx.batch_invariant()
        && ctx.bi_gemm_family() == BiGemmFamily::Triad
        && logical_f32
        && plan.is_none()
    {
        return Err(format!(
            "{label}: logical-f32 Triad graph captured no resolved GEMM route"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod model_gemm_guard_tests {
    use super::*;
    use crate::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
    use crate::mamba_ssm::gpu::buffers::DtypedBuf;
    use crate::mamba_ssm::gpu::context::GemmMode;
    use crate::mamba_ssm::gpu::device::GpuDevice;
    use crate::mamba_ssm::gpu::dtype::WeightDtype;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn deterministic_inference_half_work_rejects_empty_graph_plan() {
        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new_with_mode(&device, GemmMode::Deterministic).unwrap();
        ctx.set_bi_gemm_family(BiGemmFamily::Inference);
        let dtype = WeightDtype::Bf16;
        let x = DtypedBuf::zeros(&ctx.stream, 32, dtype).unwrap();
        let w = DtypedBuf::zeros(&ctx.stream, 32 * 16, dtype).unwrap();
        let y = DtypedBuf::zeros(&ctx.stream, 16, dtype).unwrap();
        x.upload_f32(&ctx.stream, &[1.0; 32]).unwrap();
        w.upload_f32(&ctx.stream, &[1.0; 32 * 16]).unwrap();
        crate::mamba_ssm::gpu::gemm_bi_inference::prepare_inference_arch_rung(&ctx).unwrap();
        gpu_gemm_typed_forward_raw(
            &ctx,
            TypedPtr {
                ptr: y.cached_ptr(),
                dtype,
            },
            TypedPtr {
                ptr: x.cached_ptr(),
                dtype,
            },
            TypedPtr {
                ptr: w.cached_ptr(),
                dtype,
            },
            None,
            (1, 32, 16),
        )
        .unwrap();
        let mut output = [0.0; 16];
        y.download_f32(&ctx.stream, &mut output).unwrap();
        assert_eq!(output, [32.0; 16], "real nonzero half GEMM workload");
        // Even granting the old guard its strongest boolean cannot make it
        // protect this deterministic Inference workload.
        require_f32_triad_graph_plan(&ctx, true, None, "half workload")
            .expect_err("nonzero deterministic half GEMM work requires a plan");
    }
}
