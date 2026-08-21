//! Mamba-3 SISO GPU inference example — requires `cuda` feature and NVIDIA GPU.
//!
//! ```bash
//! cargo run --example mamba3_gpu_inference --features cuda
//! ```

fn main() {
    #[cfg(feature = "cuda")]
    {
        use mamba_rs::mamba3_siso::config::Mamba3Config;
        use mamba_rs::mamba3_siso::gpu::inference::GpuMamba3Backbone;
        use mamba_rs::mamba3_siso::weights::Mamba3Weights;

        let cfg = Mamba3Config::default();
        let input_dim = cfg.d_model;

        let weights = Mamba3Weights::init(&cfg, input_dim, 42);
        println!(
            "Mamba-3 SISO: d_model={}, layers={}, nheads={}, headdim={}, d_state={}",
            cfg.d_model,
            cfg.n_layers,
            cfg.nheads(),
            cfg.headdim,
            cfg.d_state,
        );

        // Create GPU backbone (uploads weights, compiles 47 CUDA kernels)
        let mut gpu = GpuMamba3Backbone::new(
            0, // GPU device 0
            &weights, cfg, input_dim, 1, // batch = 1
        )
        .expect("GPU init failed");

        // Optional: capture CUDA Graph for ~1.6x speedup
        gpu.capture_graph().expect("graph capture failed");
        println!("CUDA Graph captured");

        let mut output = vec![0.0f32; cfg.d_model];

        // Run 10 inference steps
        for step in 0..10 {
            let input = vec![0.1 * step as f32; input_dim];
            gpu.step(&input, &mut output).expect("step failed");

            let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("step {step}: output L2 norm = {norm:.6}");
        }

        // Reset state (episode boundary)
        gpu.reset().expect("reset failed");
        println!("\nState reset. Ready for next episode.");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the `cuda` feature:");
        eprintln!("  cargo run --example mamba3_gpu_inference --features cuda");
    }
}
