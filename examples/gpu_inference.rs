//! GPU inference example — requires `cuda` feature and NVIDIA GPU.
//!
//! ```bash
//! cargo run --example gpu_inference --features cuda
//! ```

fn main() {
    #[cfg(feature = "cuda")]
    {
        use mamba_rs::gpu::inference::GpuMambaBackbone;
        use mamba_rs::{MambaBackbone, MambaConfig};

        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;

        // Initialize with random weights (paper defaults)
        let cpu_backbone = MambaBackbone::init(cfg, input_dim, 42);
        println!(
            "Mamba: {} layers, d_model={}, d_inner={}, {} params",
            cpu_backbone.n_layers(),
            cpu_backbone.config().d_model,
            cpu_backbone.config().d_inner(),
            cpu_backbone.param_count(),
        );

        // Create GPU backbone (uploads weights, compiles kernels)
        let mut gpu_bb = GpuMambaBackbone::new(
            0, // GPU device 0
            cpu_backbone.weights(),
            *cpu_backbone.config(),
            input_dim,
            1, // batch = 1
        )
        .expect("GPU init failed");

        // Optional: capture CUDA Graph for ~2-5x speedup
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
