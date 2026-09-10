//! End-to-end Mamba-1 inference latency on the inference GEMM family, at
//! the shapes of the preserved `tests/m1_gpu_benchmark.rs` inference sections
//! (seed 42, `MambaConfig::default()`, batches 1/4/16/64/128, input 0.1,
//! twenty warmups, the original eager and graph iteration counts).
//!
//! One copy per endpoint, identical except for the family's name (`Fixed` on
//! the old main, `Inference` on the release), dropped into `examples/` and built
//! with `--features cuda`. Both f32 and bf16 storage are timed.
//! It only uses the backbone construction, step, reset and capture entry
//! points both trees share. The context is configured through the setters
//! both trees share as well; on the release tree they are the deprecated
//! adapters, so the release build prints their deprecation warnings, which is
//! the intended proof that both endpoints select the same route.
//!
//! Before any timing, each batch runs a fixed number of steps three ways:
//! eager twice (the bits must repeat), once through the captured graph (the
//! bits must equal the eager bits), and on the CPU reference (cosine and
//! relative-L2 tolerance of the existing GPU-versus-CPU step parity test).
//! Every output digest is printed so the two endpoints can be compared
//! offline; a digest difference between them is a finding to investigate,
//! not something this adapter resolves.

#[cfg(feature = "cuda")]
mod baseline {
    use std::time::Instant;

    use mamba_rs::config::MambaConfig;
    use mamba_rs::mamba_ssm::gpu::context::BiGemmFamily;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::inference::GpuMambaBackbone;
    use mamba_rs::module::MambaBackbone;
    use mamba_rs::weights::MambaWeights;

    const BATCHES: [usize; 5] = [1, 4, 16, 64, 128];
    const WARMUP: usize = 20;
    const CHECK_STEPS: usize = 8;
    const INPUT_VALUE: f32 = 0.1;
    const SEED: u64 = 42;
    const FAMILY: BiGemmFamily = BiGemmFamily::Inference;
    const DTYPES: [WeightDtype; 2] = [WeightDtype::F32, WeightDtype::Bf16];
    const GEMM_CONTROLS: [&str; 5] = [
        "MAMBA_RS_GEMM_MODE",
        "MAMBA_RS_BATCH_INVARIANT",
        "MAMBA_RS_BI_GEMM_FAMILY",
        "MAMBA_RS_BI_TENSOR_CORES",
        "MAMBA_RS_FAST_GEMM",
    ];

    fn fnv1a(values: &[f32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for value in values {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn refuse_ambient_controls() {
        for name in GEMM_CONTROLS {
            assert!(
                std::env::var_os(name).is_none(),
                "{name} is set; the runner must clear every GEMM control so the adapter alone selects the route"
            );
        }
    }

    fn configure(bb: &GpuMambaBackbone, label: &str) {
        let ctx = bb.ctx();
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(FAMILY);
        ctx.set_bi_tensor_cores(false);
        ctx.set_fast_gemm(false);
        assert!(ctx.batch_invariant(), "{label}: batch-invariant route");
        assert_eq!(ctx.bi_gemm_family(), FAMILY, "{label}: inference family");
        assert!(!ctx.bi_tensor_cores(), "{label}: scalar tier");
        assert!(!ctx.fast_gemm(), "{label}: no fast vendor compute");
        println!(
            "CONTEXT arm={label} batch_invariant={} family={:?} tensor_cores={} fast_gemm={} tf32={}",
            ctx.batch_invariant(),
            ctx.bi_gemm_family(),
            ctx.bi_tensor_cores(),
            ctx.fast_gemm(),
            ctx.tf32()
        );
    }

    /// Run the fixed-step check from a reset state and return one digest per
    /// step; every value must be finite.
    fn fixed_steps(
        bb: &mut GpuMambaBackbone,
        input: &[f32],
        output: &mut [f32],
        label: &str,
    ) -> Vec<u64> {
        bb.reset().unwrap();
        (0..CHECK_STEPS)
            .map(|step| {
                bb.step(input, output).unwrap();
                assert!(
                    output.iter().all(|value| value.is_finite()),
                    "{label}: non-finite output at step {step}"
                );
                fnv1a(output)
            })
            .collect()
    }

    fn cpu_reference_matches(
        cfg: MambaConfig,
        weights: &MambaWeights,
        dtype: WeightDtype,
        batch: usize,
        input: &[f32],
        eager: &[Vec<f32>],
    ) {
        let cpu = MambaBackbone::from_weights(cfg, weights.clone()).unwrap();
        // The f32 GPU step must land on the f32 CPU reference; a half GPU
        // step carries bf16/f16 storage rounding through every layer.
        let (cos_floor, rel_ceiling) = if dtype == WeightDtype::F32 {
            (0.999, 2e-2)
        } else {
            (0.99, 1e-1)
        };
        let mut states: Vec<_> = (0..batch).map(|_| cpu.alloc_state()).collect();
        let mut scratches: Vec<_> = (0..batch).map(|_| cpu.alloc_scratch()).collect();
        let mut cpu_out = vec![0.0f32; batch * cfg.d_model];
        for (step, gpu_out) in eager.iter().enumerate() {
            cpu.forward_step_batch(input, &mut cpu_out, &mut states, &mut scratches);
            let (mut dot, mut na, mut nb, mut num, mut den) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for (&x, &y) in cpu_out.iter().zip(gpu_out.iter()) {
                dot += f64::from(x) * f64::from(y);
                na += f64::from(x) * f64::from(x);
                nb += f64::from(y) * f64::from(y);
                num += f64::from(x - y) * f64::from(x - y);
                den += f64::from(x) * f64::from(x);
            }
            let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
            let rel = (num / den.max(1e-30)).sqrt();
            println!("CPU_PARITY batch={batch} step={step} cos={cos:.6} rel_l2={rel:.3e}");
            assert!(
                cos > cos_floor && rel < rel_ceiling,
                "batch {batch} step {step}: GPU step diverged from the CPU reference (cos={cos:.6}, rel_l2={rel:.3e})"
            );
        }
    }

    fn timed(bb: &mut GpuMambaBackbone, input: &[f32], output: &mut [f32], iters: usize) -> f64 {
        for _ in 0..WARMUP {
            bb.step(input, output).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            bb.step(input, output).unwrap();
        }
        t0.elapsed().as_micros() as f64 / iters as f64
    }

    pub fn run() {
        refuse_ambient_controls();
        let cfg = MambaConfig::default();
        let input_dim = cfg.d_model;
        let base_weights = MambaWeights::init(&cfg, input_dim, SEED);
        println!(
            "MODEL d_model={} layers={} d_inner={} d_state={} params={}",
            cfg.d_model,
            cfg.n_layers,
            cfg.d_inner(),
            cfg.d_state,
            MambaBackbone::init(cfg, input_dim, SEED).param_count()
        );

        for dtype in DTYPES {
            // The half-precision GPU path requires an identity input
            // projection (an empty projection on the GPU side); the CPU
            // reference gets the same identity written out as a matrix.
            let mut weights = base_weights.clone();
            let mut cpu_weights = base_weights.clone();
            if dtype != WeightDtype::F32 {
                weights.input_proj_w.clear();
                weights.input_proj_b.clear();
                let d = cfg.d_model;
                cpu_weights.input_proj_w = (0..d * d)
                    .map(|i| if i / d == i % d { 1.0 } else { 0.0 })
                    .collect();
                cpu_weights.input_proj_b = vec![0.0; d];
            }
            for &b in &BATCHES {
                let input = vec![INPUT_VALUE; b * input_dim];
                let mut output = vec![0.0f32; b * cfg.d_model];

                let mut eager_bb =
                    GpuMambaBackbone::new_with_dtype(0, &weights, cfg, input_dim, b, dtype)
                        .unwrap();
                configure(&eager_bb, "eager");
                let mut eager_outputs = Vec::with_capacity(CHECK_STEPS);
                eager_bb.reset().unwrap();
                for step in 0..CHECK_STEPS {
                    eager_bb.step(&input, &mut output).unwrap();
                    assert!(
                        output.iter().all(|value| value.is_finite()),
                        "eager: non-finite output at step {step}"
                    );
                    eager_outputs.push(output.clone());
                }
                let eager_digests: Vec<u64> = eager_outputs.iter().map(|out| fnv1a(out)).collect();
                let repeat_digests =
                    fixed_steps(&mut eager_bb, &input, &mut output, "eager repeat");
                assert_eq!(
                    eager_digests, repeat_digests,
                    "batch {b}: eager bits changed between repeats"
                );
                for (step, digest) in eager_digests.iter().enumerate() {
                    println!(
                        "DIGEST arm=eager dtype={dtype:?} batch={b} step={step} fnv1a={digest:016x}"
                    );
                }
                cpu_reference_matches(cfg, &cpu_weights, dtype, b, &input, &eager_outputs);

                let mut graph_bb =
                    GpuMambaBackbone::new_with_dtype(0, &weights, cfg, input_dim, b, dtype)
                        .unwrap();
                configure(&graph_bb, "graph");
                graph_bb.capture_graph().unwrap();
                assert!(
                    graph_bb.has_graph(),
                    "batch {b}: graph capture did not install a graph"
                );
                let graph_digests = fixed_steps(&mut graph_bb, &input, &mut output, "graph");
                assert_eq!(
                    eager_digests, graph_digests,
                    "batch {b}: graph bits differ from eager bits"
                );
                for (step, digest) in graph_digests.iter().enumerate() {
                    println!(
                        "DIGEST arm=graph dtype={dtype:?} batch={b} step={step} fnv1a={digest:016x}"
                    );
                }

                eager_bb.reset().unwrap();
                let eager_iters = if b <= 4 { 5000 } else { 2000 };
                let eager_us = timed(&mut eager_bb, &input, &mut output, eager_iters);
                println!(
                    "TIMING arm=eager dtype={dtype:?} batch={b} iters={eager_iters} us_per_step={eager_us:.3}"
                );

                graph_bb.reset().unwrap();
                let graph_iters = if b <= 4 { 10000 } else { 5000 };
                let graph_us = timed(&mut graph_bb, &input, &mut output, graph_iters);
                println!(
                    "TIMING arm=graph dtype={dtype:?} batch={b} iters={graph_iters} us_per_step={graph_us:.3}"
                );
            }
        }
        println!("DONE");
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    baseline::run();
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("build with --features cuda");
        std::process::exit(2);
    }
}
