//! Qualify compiled fold routes against the released parallel backward scan.

#![cfg(feature = "cuda")]

#[path = "support/m1_fold/raw.rs"]
mod raw;

use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::kernel_identity::digest_hex;
use mamba_rs::mamba_ssm::gpu::kernels::MambaKernels;

#[test]
#[ignore = "requires an Ada device and a qualified native compiler"]
fn compiled_fold_routes_keep_legacy_and_specialized_entries_available() {
    let device = GpuDevice::new(0).unwrap();
    assert_eq!(device.context().compute_capability().unwrap(), (8, 9));
    assert_eq!(device.nvrtc_target(), "sm_89");
    unsafe {
        device.context().disable_event_tracking();
    }
    let caps = std::env::var("M1_FOLD_CAPS").unwrap_or_else(|_| "16,32,64".to_owned());
    for cap in caps.split(',').map(|value| value.parse::<usize>().unwrap()) {
        assert!([16, 32, 64].contains(&cap));
        let kernels =
            MambaKernels::compile_with_state_cap(device.context(), device.nvrtc_target(), cap)
                .unwrap();
        let artifacts = kernels.artifact_set_identity();
        eprintln!(
            "PRODUCTION_FIXED_ARTIFACT state_cap={} module_kind={:?} artifact_kind={:?} compile_key={} artifact_digest={} ordered_set_digest={}",
            kernels.state_cap,
            artifacts.fixed.module_kind,
            artifacts.fixed.artifact_kind,
            digest_hex(&artifacts.fixed.compile_key),
            digest_hex(&artifacts.fixed.artifact_digest),
            digest_hex(&artifacts.ordered_digest),
        );
        for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            let legacy = kernels.ssm_parallel_bwd_fold_typed.get(dtype);
            let short = kernels.ssm_parallel_bwd_fold_for_shape(dtype, 8, 33, 768, 16);
            let long = kernels.ssm_parallel_bwd_fold_for_shape(dtype, 8, 1300, 768, 16);
            let admitted = cap == 16 && dtype != WeightDtype::F32;
            assert_eq!(!std::ptr::eq(short, legacy), admitted);
            assert_eq!(!std::ptr::eq(long, legacy), admitted);
            assert_eq!(!std::ptr::eq(short, long), admitted);
            for (batch, time, inner, state) in [
                (1, 1025, 4, 16),
                (8, 513, 768, 16),
                (8, 1300, 768, 32),
                (0, 33, 768, 16),
                (8, 0, 768, 16),
                (8, 33, 767, 16),
            ] {
                assert!(std::ptr::eq(
                    kernels.ssm_parallel_bwd_fold_for_shape(dtype, batch, time, inner, state),
                    legacy,
                ));
            }
        }
        if cap == 16 {
            raw::qualify(&device, &kernels);
        } else {
            // Capacity 32 compiles the legacy Fixed composition unchanged.
            // Capacity 64 carries the retained inference overlay on top of it,
            // so its source is the qualified capacity-64 identity that the
            // dispatcher's cohorts bind, not the legacy base.
            let expected = if cap == 64 {
                "3b3db58ef62aab3cdfe7157fbd68f157212ff4a6aa0cf144d0f041b1ad0bbc00"
            } else {
                "457e2e4076342df786917ac1b0aeb2ed60fc2279ddd6dc644109cbbddb0c8348"
            };
            assert_eq!(
                mamba_rs::mamba_ssm::gpu::kernel_identity::digest_hex(
                    &kernels.compiler_identity().source_digest
                ),
                expected,
            );
        }
    }
}
