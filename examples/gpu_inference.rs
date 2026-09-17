//! Mamba SSM GPU inference example — requires `cuda` feature + NVIDIA GPU.
//!
//! ```bash
//! cargo run --release --example gpu_inference --features cuda
//! ```
//!
//! The minimal GPU decode path. Storage dtype and GEMM mode are separate
//! choices: `WeightDtype::Bf16` stores the weights in bf16 (half the VRAM of
//! f32) while every kernel still accumulates in f32, and
//! `GemmMode::Deterministic` runs the projections on the crate's own
//! fixed-order kernels (the Inference family for a model context), so a
//! row's output does not depend on the batch it is in. The example then
//! switches a second backbone to `GemmMode::CublasFast` to show the
//! fallible mode change; the two cuBLAS modes are explicit alternatives,
//! see `mamba_rs::mamba_ssm::gpu::GemmMode`.

fn main() {
    #[cfg(feature = "cuda")]
    {
        use mamba_rs::WeightDtype;
        use mamba_rs::gpu::inference::GpuMambaBackbone;
        use mamba_rs::mamba_ssm::gpu::GemmMode;
        use mamba_rs::{MambaBackbone, MambaConfig};

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;

        let cpu_backbone = MambaBackbone::init(cfg, input_dim, 42);
        println!(
            "Mamba SSM: {} layers, d_model={}, d_inner={}, {} params",
            cpu_backbone.n_layers(),
            cpu_backbone.config().d_model,
            cpu_backbone.config().d_inner(),
            cpu_backbone.param_count(),
        );

        // Storage dtype and GEMM mode are chosen explicitly here; the
        // no-mode constructors read MAMBA_RS_GEMM_MODE instead and default
        // to Deterministic when it is absent.
        let mut gpu_bb = GpuMambaBackbone::new_with_dtype_and_mode(
            0, // GPU ordinal
            cpu_backbone.weights(),
            *cpu_backbone.config(),
            input_dim,
            1, // batch
            WeightDtype::Bf16,
            GemmMode::Deterministic,
        )
        .expect("GPU init failed");
        println!(
            "storage {:?}, mode {:?}",
            gpu_bb.dtype(),
            gpu_bb.ctx().gemm_mode()
        );

        // Capture the per-step execution plan: one graph launch per step
        // afterwards. The captured graph records the numeric route; a mode
        // or family change after capture is refused at replay.
        gpu_bb.capture_graph().expect("graph capture failed");
        println!("CUDA Graph captured");

        let mut output = vec![0.0f32; cfg.d_model];

        // Run 10 inference steps
        for step in 0..10 {
            let input = vec![0.1 * step as f32; input_dim];
            gpu_bb.step(&input, &mut output).expect("step failed");

            let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("step {step}: output L2 norm = {norm:.6}");
        }

        // Reset state for new sequence
        gpu_bb.reset().expect("reset failed");
        println!("state reset");

        // The mode can also be changed on a live context. The change is
        // fallible: it is refused while a graph is being captured, and a
        // graph captured before the change refuses to replay after it, so
        // change the mode before capturing.
        let mut vendor_bb = GpuMambaBackbone::new_with_dtype_and_mode(
            0,
            cpu_backbone.weights(),
            *cpu_backbone.config(),
            input_dim,
            1,
            WeightDtype::Bf16,
            GemmMode::Deterministic,
        )
        .expect("GPU init failed");
        vendor_bb
            .ctx()
            .set_gemm_mode(GemmMode::CublasFast)
            .expect("mode change failed");
        vendor_bb.capture_graph().expect("graph capture failed");
        vendor_bb
            .step(&vec![0.1; input_dim], &mut output)
            .expect("step failed");
        println!("second backbone: mode {:?}", vendor_bb.ctx().gemm_mode());
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the `cuda` feature:");
        eprintln!("  cargo run --example gpu_inference --features cuda");
    }
}
