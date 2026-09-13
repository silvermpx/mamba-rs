//! Freeze the changed fragment while sharing the unchanged module context.

use cudarc::driver::CudaModule;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::kernel_identity::{FramedSha256, canonical_ptx_image, digest_hex};
use mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths;
use std::sync::Arc;

pub(super) fn compile(device: &GpuDevice, cap: usize) -> Arc<CudaModule> {
    let sources = [
        include_str!("../../../../kernels/_typed_prelude.cuh"),
        include_str!("../../../../kernels/mamba3_siso.cu"),
        include_str!("../../../../kernels/mamba3_ops.cu"),
        include_str!("legacy_chunked.cu"),
        include_str!("../../../../kernels/norms.cu"),
        include_str!("../../../../kernels/elementwise.cu"),
        include_str!("../../../../kernels/adamw.cu"),
    ];
    let digest = |bytes: &[u8]| digest_hex(&FramedSha256::bytes(bytes));
    assert_eq!(
        digest(sources[3].as_bytes()),
        "d4333722fc3fb3a2ef5fc76672ae4863a8275ccc5909444bec45efdba60c35b1"
    );
    for (i, source) in sources.iter().enumerate() {
        eprintln!(
            "LEGACY_FRAGMENT index={i} sha256={}",
            digest(source.as_bytes())
        );
    }
    let source = sources
        .iter()
        .map(|s| {
            s.lines()
                .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let (mut major, mut minor) = (0, 0);
    assert_eq!(
        unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) },
        cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS
    );
    assert!([(12, 8), (13, 0), (13, 2)].contains(&(major, minor)));
    let mut options = vec![
        "--fmad=true".to_owned(),
        "--extra-device-vectorization".to_owned(),
        "-DNDEBUG".to_owned(),
        format!("-DMAMBA_RS_STATE_CAP={cap}"),
    ];
    if (major, minor) >= (12, 9) {
        options.push("--frandom-seed=1295203121".to_owned());
    }
    eprintln!(
        "LEGACY_MODULE cap={cap} nvrtc={major}.{minor} target={} sha256={} flags={options:?}",
        device.nvrtc_target(),
        digest(source.as_bytes())
    );
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        source,
        cudarc::nvrtc::CompileOptions {
            arch: Some(device.nvrtc_target()),
            options,
            include_paths: cuda_include_paths(),
            ..Default::default()
        },
    )
    .unwrap();
    let canonical = canonical_ptx_image(ptx.as_bytes().expect("NVRTC PTX")).unwrap();
    eprintln!("LEGACY_ARTIFACT sha256={}", digest(canonical.as_bytes()));
    device
        .context()
        .load_module(cudarc::nvrtc::Ptx::from_src(canonical))
        .unwrap()
}
