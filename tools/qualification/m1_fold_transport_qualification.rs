//! Qualify compiled fold routes against the released parallel backward scan.

#![cfg(feature = "cuda")]

#[path = "support/m1_fold/raw.rs"]
mod raw;

use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::gpu::dtype::WeightDtype;
use mamba_rs::gpu::kernel_identity::digest_hex;
use mamba_rs::gpu::kernels::MambaKernels;

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
    let mut mamba1_sources = std::collections::BTreeMap::new();
    let mut inference_compilers = Vec::new();
    for cap in caps.split(',').map(|value| value.parse::<usize>().unwrap()) {
        assert!([16, 32, 64].contains(&cap));
        let kernels =
            MambaKernels::compile_with_state_cap(device.context(), device.nvrtc_target(), cap)
                .unwrap();
        let artifacts = kernels.artifact_set_identity();
        eprintln!(
            "PRODUCTION_INFERENCE_ARTIFACT state_cap={} module_kind={:?} artifact_kind={:?} compile_key={} artifact_digest={} ordered_set_digest={}",
            kernels.state_cap,
            artifacts.inference.module_kind,
            artifacts.inference.artifact_kind,
            digest_hex(&artifacts.inference.compile_key),
            digest_hex(&artifacts.inference.artifact_digest),
            digest_hex(&artifacts.ordered_digest),
        );
        let mamba1 = kernels.mamba1_artifact_identity();
        eprintln!(
            "PRODUCTION_MAMBA1_ARTIFACT state_cap={} module_kind={:?} artifact_kind={:?} compile_key={} artifact_digest={} source_digest={}",
            kernels.state_cap,
            mamba1.module_kind,
            mamba1.artifact_kind,
            digest_hex(&mamba1.compile_key),
            digest_hex(&mamba1.artifact_digest),
            digest_hex(&kernels.mamba1_compiler_identity().source_digest),
        );
        mamba1_sources.insert(cap, kernels.mamba1_compiler_identity().source_digest);
        inference_compilers.push(kernels.compiler_identity());
        for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
            let legacy = kernels.ssm_parallel_bwd_fold_typed.i32.get(dtype);
            let short = kernels
                .ssm_parallel_bwd_fold_launch(dtype, 8, 33, 768, 16)
                .0;
            let long = kernels
                .ssm_parallel_bwd_fold_launch(dtype, 8, 1300, 768, 16)
                .0;
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
                    kernels
                        .ssm_parallel_bwd_fold_launch(dtype, batch, time, inner, state)
                        .0,
                    legacy,
                ));
            }
        }
        if cap == 16 {
            raw::qualify(&device, &kernels);
        }
    }
    // The fold clones compose into the Mamba1 module at capacity 16 only;
    // every other capacity compiles the plain Mamba1 composition, whose
    // source carries no capacity (the capacity is a compile option).
    if let (Some(plain32), Some(plain64)) = (mamba1_sources.get(&32), mamba1_sources.get(&64)) {
        assert_eq!(plain32, plain64);
    }
    if let (Some(folded), Some(plain)) = (mamba1_sources.get(&16), mamba1_sources.get(&32)) {
        assert_ne!(folded, plain);
    }
    // The Inference module does not depend on the state capacity at all.
    assert!(
        inference_compilers
            .windows(2)
            .all(|pair| pair[0] == pair[1])
    );
}
