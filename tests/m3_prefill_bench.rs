//! One-pass prompt prefill latency at a production-scale shape
//! (T=4621, 24 layers, d_model=384 — a document-page classify serve
//! shape). Run manually, release build:
//!
//! `cargo test --features cuda,hf,gemm-blas --release --test m3_prefill_bench -- --ignored --nocapture`

#![cfg(feature = "cuda")]

use std::time::Instant;

use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::inference::GpuMamba3Backbone;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 65536.0 - 0.5
        })
        .collect()
}

#[test]
#[ignore]
fn m3_prefill_latency_at_serve_shape() {
    let cfg = Mamba3Config {
        d_model: 384,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 24,
        rope_fraction: 0.5,
        a_floor: 1e-4,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    };
    let t = 4621usize;
    let dm = cfg.d_model;
    let mut w = Mamba3Weights::init(&cfg, dm, 42);
    w.input_proj_w.clear();
    w.input_proj_b.clear();

    let mut bb = GpuMamba3Backbone::new_with_dtype(0, &w, cfg, dm, 1, WeightDtype::F32).unwrap();
    let stream = bb.stream().clone();
    let mut gpu_input = GpuBuffer::zeros(&stream, t * dm).unwrap();
    stream.synchronize().unwrap();
    gpu_input.upload(&stream, &det(t * dm, 0xA1)).unwrap();
    let mut prefill = bb.alloc_prefill(t).unwrap();

    for _ in 0..3 {
        bb.prefill_sequence(&mut prefill, &gpu_input, t, false)
            .unwrap();
    }
    stream.synchronize().unwrap();

    let iters = 20usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        bb.prefill_sequence(&mut prefill, &gpu_input, t, false)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "prefill T={t} layers={} dm={dm}: {:.2} ms/prefill ({:.1} prefills/s)",
        cfg.n_layers,
        1e3 * dt / iters as f64,
        iters as f64 / dt
    );
}

/// Training step latency at a multi-chunk shape (the chunked backward is
/// the target). Run manually, release build.
#[test]
#[ignore]
fn m3_train_step_at_multichunk_shape() {
    use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;

    let cfg = Mamba3Config {
        d_model: 384,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 24,
        rope_fraction: 0.5,
        a_floor: 1e-4,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    };
    // Shape rides env so the CAMPAIGN shape is measurable without a
    // recompile (same knobs as the M1 campaign arm).
    let get = |k: &str, d: usize| -> usize {
        match std::env::var(k) {
            Err(_) => d,
            // Strict: a typo in a shape knob must fail, not silently
            // measure the default shape under the requested label.
            Ok(v) => v
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("{k}={v:?} did not parse: {e}")),
        }
    };
    let (batch, seq_len) = (get("MAMBA_RS_BENCH_B", 1), get("MAMBA_RS_BENCH_T", 256));
    let n = batch * seq_len * cfg.d_model;

    for dtype in [WeightDtype::F32, WeightDtype::Bf16] {
        // The f32 forward always runs the input-projection GEMM (eye
        // weights = identity semantics); the mixed pipeline wants the
        // identity branch (cleared weights).
        let mut w = Mamba3Weights::init(&cfg, cfg.d_model, 42);
        if matches!(dtype, WeightDtype::F32) {
            let dm = cfg.d_model;
            w.input_proj_w = (0..dm * dm)
                .map(|i| if i / dm == i % dm { 1.0 } else { 0.0 })
                .collect();
            w.input_proj_b = vec![0.0; dm];
        } else {
            w.input_proj_w.clear();
            w.input_proj_b.clear();
        }
        let mut tr =
            Mamba3Trainer::new_with_dtype(0, &w, cfg, cfg.d_model, batch, seq_len, dtype).unwrap();
        let input = det(n, 0x91);
        let d_temporal = det(n, 0x92);
        for _ in 0..3 {
            tr.step(&input, &d_temporal).unwrap();
        }
        tr.ctx().stream.synchronize().unwrap();
        let iters = 20usize;
        let t0 = Instant::now();
        for _ in 0..iters {
            tr.step(&input, &d_temporal).unwrap();
        }
        // Without the sync both loops read host enqueue time, not the
        // step wall — the device may still be several steps behind.
        tr.ctx().stream.synchronize().unwrap();
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "train step {dtype:?} B={batch} T={seq_len} layers={}: {:.2} ms/step (eager)",
            cfg.n_layers,
            1e3 * dt / iters as f64
        );
        // Graph lane too — the M1 table is graph-mode; an eager-only M3
        // number was never comparable with it.
        tr.capture_graph().unwrap();
        for _ in 0..5 {
            tr.step(&input, &d_temporal).unwrap();
        }
        tr.ctx().stream.synchronize().unwrap();
        let t1 = Instant::now();
        for _ in 0..iters {
            tr.step(&input, &d_temporal).unwrap();
        }
        tr.ctx().stream.synchronize().unwrap();
        let dt = t1.elapsed().as_secs_f64();
        eprintln!(
            "train step {dtype:?} B={batch} T={seq_len} layers={}: {:.2} ms/step (graph)",
            cfg.n_layers,
            1e3 * dt / iters as f64
        );
    }
}

/// The G8 page measurement: pooled-graph replay latency per page at the
/// prism serve shape (non-identity 1024->384 input projection, T=4621),
/// across the three numeric routes that matter: today's deterministic
/// f32 serve (Fixed family), the typed bf16 lane on the batch-invariant
/// tensor-core ladder, and non-deterministic cuBLAS f32 as the speed
/// reference.
#[test]
#[ignore]
fn m3_pooled_page_bench_typed_vs_f32() {
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
    use mamba_rs::mamba3_siso::gpu::prefill::{
        Mamba3Prefill, Mamba3PrefillPooledGraph, Mamba3PrefillRun,
    };
    use mamba_rs::mamba3_siso::gpu::state::{GpuMamba3Dims, GpuMamba3StateBufs};
    use mamba_rs::mamba3_siso::gpu::weights::{
        GpuMamba3MixedWeights, GpuMamba3WeightsInf, Mamba3WeightsView,
    };

    let cfg = Mamba3Config {
        d_model: 384,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 24,
        rope_fraction: 0.5,
        a_floor: 1e-4,
        is_outproj_norm: true,
        ..Mamba3Config::default()
    };
    let input_dim = 1024usize;
    let t = 4621usize;
    let dm = cfg.d_model;
    let w = Mamba3Weights::init(&cfg, input_dim, 42);
    let input = det(t * input_dim, 0xA1);

    let dims = GpuMamba3Dims {
        batch: 1,
        d_model: dm,
        d_inner: cfg.d_inner(),
        d_state: cfg.d_state,
        nheads: cfg.nheads(),
        headdim: cfg.headdim,
        ngroups: cfg.ngroups,
        in_proj_dim: cfg.in_proj_out_dim(),
        seq_len: t,
        mamba_input_dim: input_dim,
        n_layers: cfg.n_layers,
        n_angles: cfg.num_rope_angles(),
        a_floor: cfg.a_floor,
        is_outproj_norm: cfg.is_outproj_norm,
        rms_norm_eps: cfg.rms_norm_eps,
        use_parallel_scan: true,
    };

    // (label, dtype, batch_invariant, tc, family)
    let arms: &[(&str, WeightDtype, bool, bool, BiGemmFamily)] = &[
        (
            "f32 fixed (serve today)",
            WeightDtype::F32,
            true,
            false,
            BiGemmFamily::Fixed,
        ),
        (
            "bf16 ladder (serve route)",
            WeightDtype::Bf16,
            true,
            true,
            BiGemmFamily::Triad,
        ),
        (
            "f32 cuBLAS (non-det ref)",
            WeightDtype::F32,
            false,
            false,
            BiGemmFamily::Triad,
        ),
    ];

    for &(label, dtype, bi, tc, family) in arms {
        let device = GpuDevice::new(0).expect("cuda device");
        let ctx = GpuCtx::new(&device).expect("ctx");
        ctx.set_batch_invariant(bi);
        ctx.set_bi_gemm_family(family);
        if tc {
            ctx.set_bi_tensor_cores(true);
        }
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = Mamba3Kernels::compile(device.context(), arch).expect("m3 kernels");
        let fw;
        let mw;
        let view: &dyn Mamba3WeightsView = if dtype == WeightDtype::F32 {
            fw = GpuMamba3WeightsInf::from_cpu(&ctx.stream, &w, input_dim).unwrap();
            &fw
        } else {
            mw = GpuMamba3MixedWeights::from_cpu(&ctx.stream, &w, dtype).unwrap();
            &mw
        };
        let gpu_input = GpuBuffer::from_cpu(&ctx.stream, &input).unwrap();
        let mut prefill = Mamba3Prefill::new_with_dtype(&ctx.stream, &dims, dtype).unwrap();
        let mut last_hidden = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
        let mut pooled = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
        let nl = cfg.n_layers;
        let nh = cfg.nheads();
        let mut ssm = GpuBuffer::zeros(&ctx.stream, nl * nh * cfg.headdim * cfg.d_state).unwrap();
        let mut kst = GpuBuffer::zeros(&ctx.stream, nl * nh * cfg.d_state).unwrap();
        let mut vst = GpuBuffer::zeros(&ctx.stream, nl * nh * cfg.headdim).unwrap();
        let mut ast =
            GpuBuffer::zeros(&ctx.stream, nl * nh * cfg.num_rope_angles().max(1)).unwrap();

        let graph = Mamba3PrefillPooledGraph::capture(
            &mut prefill,
            &Mamba3PrefillRun {
                ctx: &ctx,
                kernels: &kernels,
                dims: &dims,
                weights: view,
                mamba_input: &gpu_input,
                identity_proj: false,
                carry_state: false,
            },
            GpuMamba3StateBufs {
                ssm: &mut ssm,
                k: &mut kst,
                v: &mut vst,
                angle: &mut ast,
            },
            &mut last_hidden,
            &mut pooled,
        )
        .unwrap();

        let states = GpuMamba3StateBufs {
            ssm: &mut ssm,
            k: &mut kst,
            v: &mut vst,
            angle: &mut ast,
        };
        for _ in 0..3 {
            graph
                .replay(&ctx, &kernels, view, &gpu_input, &states, &pooled)
                .unwrap();
        }
        ctx.stream.synchronize().unwrap();
        let iters = 40usize;
        let t0 = Instant::now();
        for _ in 0..iters {
            graph
                .replay(&ctx, &kernels, view, &gpu_input, &states, &pooled)
                .unwrap();
        }
        ctx.stream.synchronize().unwrap();
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "{label:>28}: {:.2} ms/page ({:.1} pages/s)",
            1e3 * dt / iters as f64,
            iters as f64 / dt
        );
    }
}
