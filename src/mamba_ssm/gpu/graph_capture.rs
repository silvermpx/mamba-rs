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

/// Capture all CUDA work issued by `body` on `stream` into a CUDA Graph.
///
/// Mirrors the pattern in `inference::GpuInferenceEngine::capture_graph`,
/// extracted so it doesn't need to be reimplemented per pipeline.
pub fn capture_into_graph<F>(stream: &Arc<CudaStream>, body: F) -> Result<CudaGraph, String>
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

    let body_result = body();

    // ALWAYS end capture, even on body error — a stream stuck in capture
    // mode silently breaks every subsequent op.
    let end_result = stream.end_capture(
        cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );

    body_result?;
    end_result
        .map_err(|e| format!("end_capture: {e:?}"))?
        .ok_or_else(|| "end_capture returned no graph (empty body?)".to_string())
}
