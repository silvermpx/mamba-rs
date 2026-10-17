//! SISO kernels frozen as they were released (`fixtures/*_reference.cu`),
//! compiled as the SISO module compiles its own: the same prelude, the
//! packed-row helpers, the numeric options and the state capacity, so each
//! frozen kernel gives the bits it gave when it was the production one.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule};
use mamba_rs::gpu::kernels::cuda_include_paths;

/// The module of a fixture's frozen kernels (`fixture` is its source) for
/// `arch`, their register arrays sized for `state_cap` states.
pub fn frozen_module(
    ctx: &Arc<CudaContext>,
    arch: &'static str,
    state_cap: usize,
    fixture: &str,
) -> Arc<CudaModule> {
    let source = [
        include_str!("../../kernels/_typed_prelude.cuh"),
        include_str!("../../kernels/_varlen.cuh"),
        fixture,
    ]
    .join("\n");
    let (mut major, mut minor) = (0, 0);
    let status = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    assert_eq!(status, cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS);
    let mut options = vec![
        "--fmad=true".to_owned(),
        "--extra-device-vectorization".to_owned(),
        "-DNDEBUG".to_owned(),
        format!("-DMAMBA_RS_STATE_CAP={state_cap}"),
    ];
    if (major, minor) >= (12, 9) {
        options.push("--frandom-seed=1295203121".to_owned());
    }
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        &source,
        cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options,
            include_paths: cuda_include_paths(),
            ..Default::default()
        },
    )
    .expect("frozen kernels compilation");
    ctx.load_module(ptx).expect("load frozen kernels")
}
