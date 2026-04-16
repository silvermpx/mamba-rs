//! Mamba SSM GPU inference example — requires `cuda` feature + NVIDIA GPU.
//!
//! ```bash
//! cargo run --release --example gpu_inference --features cuda
//! ```
//!
//! Shows the minimal GPU path for both f32 and bf16 storage. The API is
//! identical — `new_with_dtype(..., WeightDtype::Bf16)` swaps in the
//! native bf16 pipeline (weight VRAM halved, ~30% faster decode). Compute
//! stays f32 regardless of storage dtype; batch-invariant GEMM ensures
//! logits are bit-identical across batch sizes.

fn main() {
    #[cfg(feature = "cuda")]
    {
        use mamba_rs::WeightDtype;
        use mamba_rs::gpu::inference::GpuMambaBackbone;
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

        // Create GPU backbone. `new_with_dtype` accepts F32 / Bf16 / F16;
        // bf16 halves weight VRAM and speeds up decode by ~30% on Ada.
        let mut gpu_bb = GpuMambaBackbone::new_with_dtype(
            0, // GPU ordinal
            cpu_backbone.weights(),
            *cpu_backbone.config(),
            input_dim,
            1, // batch
            WeightDtype::Bf16,
        )
        .expect("GPU init failed");

        // Capture the per-step execution plan — one cuGraphLaunch per step
        // thereafter, ~2-5× faster than eager dispatch at batch=1.
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
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the `cuda` feature:");
        eprintln!("  cargo run --example gpu_inference --features cuda");
    }
}
