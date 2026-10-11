//! The import paths the 0.7 README and guides taught.
//!
//! 0.8 moved Mamba-1 into `mamba1`, split the GPU engine the families share
//! out into `gpu`, and renamed the GPU types after their family. The old
//! module paths stay as hidden re-exports and the old names as deprecated
//! aliases for one release. These tests hold every path the 0.7
//! documentation used to the item it names in 0.8, so a path that stops
//! resolving fails the build of this file.

use std::any::TypeId;

fn same<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

#[test]
fn the_cpu_types_of_0_7_keep_their_paths() {
    assert!(same::<mamba_rs::config::MambaConfig, mamba_rs::MambaConfig>());
    assert!(same::<mamba_rs::state::MambaState, mamba_rs::MambaState>());
    assert!(same::<
        mamba_rs::weights::MambaWeights,
        mamba_rs::MambaWeights,
    >());
    assert!(same::<
        mamba_rs::mamba_ssm::MambaConfig,
        mamba_rs::mamba1::MambaConfig,
    >());
    assert!(same::<
        mamba_rs::inference::PrefillMode,
        mamba_rs::mamba1::cpu::prefill::PrefillMode,
    >());
    assert!(same::<
        mamba_rs::module::MambaBackbone,
        mamba_rs::MambaBackbone,
    >());
    assert!(same::<
        mamba_rs::module::sample::SampleParams,
        mamba_rs::SampleParams,
    >());
    assert!(same::<
        mamba_rs::mamba3_siso::config::Mamba3Config,
        mamba_rs::Mamba3Config,
    >());
    assert!(same::<
        mamba_rs::mamba3_siso::state::Mamba3State,
        mamba_rs::Mamba3State,
    >());
    assert!(same::<
        mamba_rs::mamba3_siso::weights::Mamba3Weights,
        mamba_rs::Mamba3Weights,
    >());
    assert!(same::<
        mamba_rs::mamba3_siso::cpu::inference::Mamba3StepScratch,
        mamba_rs::Mamba3StepScratch,
    >());
}

#[test]
#[expect(
    deprecated,
    reason = "save_mamba3 is the 0.7 name, deprecated in 0.8, and the test holds that it still resolves"
)]
fn the_cpu_functions_of_0_7_keep_their_paths() {
    let _ = mamba_rs::serialize::save;
    let _ = mamba_rs::serialize::load;
    let _ = mamba_rs::mamba3_siso::serialize::save_mamba3;
    let _ = mamba_rs::mamba3_siso::serialize::load_mamba3;
    let _ = mamba_rs::mamba3_siso::cpu::inference::mamba3_step;
    let _ = mamba_rs::mamba_step;
}

#[cfg(feature = "cuda")]
#[test]
#[expect(
    deprecated,
    reason = "GpuMambaBackbone and MambaTrainer are the 0.7 names, deprecated in 0.8, and the test holds that they still resolve"
)]
fn the_gpu_types_of_0_7_keep_their_paths() {
    use mamba_rs::gpu::inference::GpuMambaBackbone;
    use mamba_rs::mamba_ssm::gpu::trainer::{BackwardOpts, MambaTrainer, TrainSessionCfg};

    assert!(same::<GpuMambaBackbone, mamba_rs::MambaGpuBackbone>());
    assert!(same::<MambaTrainer, mamba_rs::MambaGpuTrainer>());
    assert!(same::<BackwardOpts, mamba_rs::BackwardOpts>());
    assert!(same::<TrainSessionCfg, mamba_rs::TrainSessionCfg>());
    assert!(same::<mamba_rs::mamba_ssm::gpu::GemmMode, mamba_rs::GemmMode>());
    assert!(same::<
        mamba_rs::mamba_ssm::gpu::dtype::WeightDtype,
        mamba_rs::WeightDtype,
    >());
    assert!(same::<
        mamba_rs::mamba_ssm::gpu::context::GpuCtx,
        mamba_rs::gpu::context::GpuCtx,
    >());
    assert!(same::<
        mamba_rs::mamba_ssm::gpu::device::GpuDevice,
        mamba_rs::gpu::device::GpuDevice,
    >());
    #[cfg(feature = "hf")]
    let _ = TypeId::of::<mamba_rs::module::gpu_lm::GpuMambaLM>();
}
