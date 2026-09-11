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
use super::context::{GemmMode, GpuCtx};
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

/// Require a GEMM inventory for nonempty deterministic model work, regardless
/// of family or storage dtype. Vendor-only and zero-GEMM graphs may omit it.
pub(crate) fn require_deterministic_gemm_graph_plan(
    ctx: &GpuCtx,
    has_gemm_work: bool,
    plan: Option<&CapturedGemmGraphPlan>,
    label: &str,
) -> Result<(), String> {
    if ctx.gemm_mode() == GemmMode::Deterministic && has_gemm_work && plan.is_none() {
        return Err(format!(
            "{label}: nonzero deterministic GEMM graph captured no resolved GEMM route"
        ));
    }
    Ok(())
}

/// Validate context health and the GEMM-only inventory before submitting a
/// graph. Callers retain their logical-route and fixed-buffer checks.
pub(crate) fn with_validated_gemm_graph_launch(
    ctx: &GpuCtx,
    has_gemm_work: bool,
    plan: Option<&CapturedGemmGraphPlan>,
    label: &str,
    launch: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    ctx.ensure_gemm_usable()?;
    require_deterministic_gemm_graph_plan(ctx, has_gemm_work, plan, label)?;
    match plan {
        Some(plan) => plan.with_validated_launch(ctx, label, launch),
        None => launch(),
    }
}

#[cfg(test)]
pub(crate) mod model_gemm_guard_tests {
    use super::*;
    use crate::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
    use crate::mamba_ssm::gpu::buffers::DtypedBuf;
    use crate::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode};
    use crate::mamba_ssm::gpu::device::GpuDevice;
    use crate::mamba_ssm::gpu::dtype::WeightDtype;
    use crate::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, PolicyDtype, RecordedGemmTrace, ResolvedGemmOp,
    };
    use std::cell::Cell;

    pub(crate) fn configure(ctx: &GpuCtx, family: BiGemmFamily, tc: bool) {
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        ctx.set_bi_gemm_family(family);
        ctx.set_bi_tensor_cores(tc);
        ctx.set_f32_triad_policy(crate::mamba_ssm::gpu::context::F32TriadPolicy::ExactScalarFmaV1);
        ctx.set_half_triad_policy(crate::mamba_ssm::gpu::context::HalfTriadPolicy::TiledParityV1);
    }

    pub(crate) fn assert_inventory(
        ctx: &GpuCtx,
        trace: &RecordedGemmTrace,
        manifest: PreparedGemmCaptureManifest,
        plan: &CapturedGemmGraphPlan,
        expected: &[(usize, usize, usize)],
    ) {
        assert_eq!(manifest, trace.manifest());
        assert_eq!(Some(plan.launches), manifest.launches);
        assert_eq!(plan.routes(), trace.routes());
        let mut groups = Vec::new();
        for route in plan.routes() {
            assert_eq!(route.op, ResolvedGemmOp::Nn);
            ctx.validate_resolved_gemm_route(route, "model inventory")
                .unwrap();
            assert_ne!(route.launch.arguments_digest, [0; 32]);
            if groups.last() != Some(&route.shape) {
                groups.push(route.shape);
            }
        }
        assert_eq!(groups, expected, "ordered projection groups");
        if ctx.bi_gemm_family() == BiGemmFamily::Inference {
            assert_eq!(
                plan.routes().len(),
                expected.len(),
                "small direct Inference fixture"
            );
        }
    }

    pub(crate) fn assert_plan_mutations(ctx: &GpuCtx, plan: &CapturedGemmGraphPlan) {
        let calls = Cell::new(0);
        with_validated_gemm_graph_launch(ctx, true, Some(plan), "positive", || {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
        for change in 0..7 {
            let mut routes = plan.routes().to_vec();
            match change {
                0 => {
                    routes.remove(routes.len() / 2);
                }
                1 => routes.swap(0, 1),
                2 => routes[0].symbol = "invalid_model_projection",
                3 => routes[0].module_kind = ModuleKind::TriadScalar,
                4 => routes[0].dtype = PolicyDtype::F16,
                5 => routes[0].launch.arguments_digest[0] ^= 1,
                6 => routes[0].tensor_maps_digest[0] ^= 1,
                _ => unreachable!(),
            }
            // Every mutation moves the routes away from the recorded launch
            // digest, so the plan cannot even be built with it.
            assert!(
                CapturedGemmGraphPlan::new(
                    plan.context,
                    plan.launches,
                    routes.clone().into_boxed_slice()
                )
                .is_err(),
                "mutation {change} must not build a plan under the recorded digest"
            );
            // A plan whose digest matches the mutated routes is internally
            // consistent; the ones that name a route the live policy cannot
            // serve must still be refused at replay, before any work runs.
            let launches =
                crate::mamba_ssm::gpu::kernel_identity::build_resolved_gemm_launch_set(&routes)
                    .unwrap();
            let rebuilt =
                CapturedGemmGraphPlan::new(plan.context, launches, routes.into_boxed_slice())
                    .unwrap();
            let policy_visible = matches!(change, 2..=4);
            let result =
                with_validated_gemm_graph_launch(ctx, true, Some(&rebuilt), "tampered", || {
                    calls.set(calls.get() + 1);
                    Ok(())
                });
            if policy_visible {
                assert!(result.is_err(), "mutation {change} must fail at replay");
                assert_eq!(calls.get(), 1, "mutation {change} submitted work");
            } else {
                assert!(result.is_ok(), "mutation {change} is a consistent plan");
                calls.set(1);
            }
        }
        let family = ctx.bi_gemm_family();
        ctx.set_bi_gemm_family(if family == BiGemmFamily::Inference {
            BiGemmFamily::Triad
        } else {
            BiGemmFamily::Inference
        });
        assert!(
            with_validated_gemm_graph_launch(ctx, true, Some(plan), "family", || {
                calls.set(calls.get() + 1);
                Ok(())
            })
            .is_err()
        );
        ctx.set_bi_gemm_family(family);
        ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
        assert!(
            with_validated_gemm_graph_launch(ctx, true, Some(plan), "mode", || {
                calls.set(calls.get() + 1);
                Ok(())
            })
            .is_err()
        );
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        assert_eq!(calls.get(), 1);
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn model_graph_guard_zero_vendor_and_poison_controls() {
        let device = GpuDevice::new(0).unwrap();
        let ctx = GpuCtx::new_with_mode(&device, GemmMode::Deterministic).unwrap();
        let calls = Cell::new(0);
        for mode in [
            GemmMode::Deterministic,
            GemmMode::CublasFast,
            GemmMode::CublasPedantic,
        ] {
            ctx.set_gemm_mode(mode).unwrap();
            with_validated_gemm_graph_launch(
                &ctx,
                mode != GemmMode::Deterministic,
                None,
                "positive",
                || {
                    calls.set(calls.get() + 1);
                    Ok(())
                },
            )
            .unwrap();
        }
        assert_eq!(calls.get(), 3);
        ctx.poison_gemm_for_test();
        // The vendor None-plan branch must not bypass context health.
        assert!(
            with_validated_gemm_graph_launch(&ctx, true, None, "poison", || {
                calls.set(calls.get() + 1);
                Ok(())
            })
            .unwrap_err()
            .contains("unusable")
        );
        assert_eq!(calls.get(), 3);
    }

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
        require_deterministic_gemm_graph_plan(&ctx, true, None, "half workload")
            .expect_err("nonzero deterministic half GEMM work requires a plan");
    }
}
