//! One-pass prompt prefill latency at a production-scale shape
//! (T=4621, 24 layers, d_model=384 — a document-page classify serve
//! shape). Run manually, release build:
//!
//! `cargo test --features cuda,hf,gemm-blas,qualification --release --test m3_prefill_bench m3_prefill_latency_at_serve_shape -- --exact --ignored --nocapture --test-threads=1`

#![cfg(feature = "cuda")]

use std::time::Instant;

use mamba_rs::mamba_ssm::gpu::GemmMode;
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

fn training_dtypes(value: Option<&str>) -> Result<&'static [WeightDtype], String> {
    match value {
        None => Ok(&[WeightDtype::F32, WeightDtype::Bf16]),
        Some("all") => Ok(&[WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16]),
        Some("f32") => Ok(&[WeightDtype::F32]),
        Some("tf32") => Ok(&[WeightDtype::Tf32]),
        Some("bf16") => Ok(&[WeightDtype::Bf16]),
        Some("f16") => Ok(&[WeightDtype::F16]),
        Some(value) => Err(format!("unknown MAMBA_RS_BENCH_DTYPE={value:?}")),
    }
}

fn warm_up_f16_training(
    graph_replayed: bool,
    mut step: impl FnMut() -> Result<mamba_rs::mamba_ssm::gpu::trainer::StepMetrics, String>,
) -> Result<(usize, usize, f32), String> {
    let mut skipped = 0;
    let mut consecutive_clean = 0;
    let mut last_scale = 0.0;
    // Initial loss-scale backoff can outlast a fixed warmup count. Require
    // successful optimizer updates before timing, but bound unstable runs.
    for attempts in 1..=128 {
        let metrics = step()?;
        if metrics.graph_replayed != graph_replayed {
            return Err("F16 warmup used the wrong eager/graph execution path".into());
        }
        last_scale = metrics
            .loss_scale
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .ok_or("F16 warmup did not report a valid loss scale")?;
        if metrics
            .overflow_skipped
            .ok_or("F16 warmup did not report overflow status")?
        {
            skipped += 1;
            consecutive_clean = 0;
        } else {
            consecutive_clean += 1;
        }
        if consecutive_clean == 8 {
            return Ok((attempts, skipped, last_scale));
        }
    }
    Err(format!(
        "F16 warmup failed to reach 8 consecutive successful updates: attempts=128 skipped={skipped} consecutive_clean={consecutive_clean} last_used_loss_scale={last_scale}"
    ))
}

#[test]
fn training_dtype_filter_selects_only_the_requested_lane() {
    for (name, dtype) in [
        ("f32", WeightDtype::F32),
        ("bf16", WeightDtype::Bf16),
        ("f16", WeightDtype::F16),
    ] {
        assert_eq!(training_dtypes(Some(name)).unwrap(), [dtype]);
    }
    assert_eq!(
        training_dtypes(None).unwrap(),
        [WeightDtype::F32, WeightDtype::Bf16],
    );
    assert_eq!(
        training_dtypes(Some("all")).unwrap(),
        [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16],
    );
}

#[test]
fn training_dtype_filter_rejects_a_misspelled_lane() {
    for value in ["", "bf61", "tf32", "f16,bf16"] {
        assert!(training_dtypes(Some(value)).is_err(), "{value}");
    }
}

#[test]
fn f16_training_warmup_restarts_after_overflow() {
    use mamba_rs::mamba_ssm::gpu::loss_scaler::DynamicLossScaler;
    use mamba_rs::mamba_ssm::gpu::trainer::StepMetrics;

    for graph_replayed in [false, true] {
        let mut scaler = DynamicLossScaler::new();
        let mut attempts = 0;
        let mut updates = 0;
        let result = warm_up_f16_training(graph_replayed, || {
            attempts += 1;
            let overflow = attempts <= 4 || attempts == 7;
            let scale = scaler.scale();
            scaler.update(overflow);
            updates += u64::from(!overflow);
            Ok(StepMetrics {
                step: updates,
                graph_replayed,
                loss_scale: Some(scale),
                overflow_skipped: Some(overflow),
            })
        })
        .unwrap();
        assert_eq!(result, (15, 5, 2048.0));
        assert_eq!(attempts, 15);
        assert_eq!(updates, 10);
    }
}

#[test]
fn f16_training_warmup_stops_at_the_attempt_limit() {
    use mamba_rs::mamba_ssm::gpu::trainer::StepMetrics;

    for intermittent in [false, true] {
        let mut attempts = 0usize;
        let result = warm_up_f16_training(false, || {
            attempts += 1;
            assert!(attempts <= 128, "warmup exceeded its finite limit");
            Ok(StepMetrics {
                step: 0,
                graph_replayed: false,
                loss_scale: Some(1.0),
                overflow_skipped: Some(!intermittent || attempts.is_multiple_of(8)),
            })
        });
        assert!(result.is_err());
        assert_eq!(attempts, 128);
    }
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
    eprintln!(
        "prefill route: mode={:?} family={:?} route={:?} tensor_cores={} f32_policy={:?}",
        bb.ctx().gemm_mode(),
        bb.ctx().route_controls().family(),
        bb.ctx().gemm_route(),
        bb.ctx().route_controls().tensor_cores(),
        bb.ctx().route_controls().f32_policy(),
    );
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
/// `MAMBA_RS_BENCH_DTYPE=f32|tf32|bf16|f16|all` selects lanes; unset retains F32/BF16.
/// `MAMBA_RS_BENCH_B`, `MAMBA_RS_BENCH_T` and `MAMBA_RS_BENCH_ITERS` override
/// batch, sequence length and timed steps; defaults are 1, 256 and 20.
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
    // Shape rides env so the PRODUCTION shape is measurable without a
    // recompile (same knobs as the M1 free-shape arm).
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
    let iters = get("MAMBA_RS_BENCH_ITERS", 20);
    assert!(
        batch > 0 && seq_len > 0 && iters > 0,
        "benchmark dimensions and iterations must be positive"
    );
    let n = batch * seq_len * cfg.d_model;

    let requested_dtype = std::env::var("MAMBA_RS_BENCH_DTYPE").ok();
    for &dtype in training_dtypes(requested_dtype.as_deref()).unwrap() {
        // The f32 forward always runs the input-projection GEMM (eye
        // weights = identity semantics); the mixed pipeline wants the
        // identity branch (cleared weights).
        let mut w = Mamba3Weights::init(&cfg, cfg.d_model, 42);
        if matches!(dtype, WeightDtype::F32 | WeightDtype::Tf32) {
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
        eprintln!(
            "train route: dtype={dtype:?} scan={:?} parallel={} mode={:?} family={:?} route={:?} tensor_cores_allowed={} f32_policy={:?} B={batch} T={seq_len} layers={} state_cap={} iters={iters}",
            cfg.scan_mode,
            cfg.train_use_parallel_scan(),
            tr.ctx().gemm_mode(),
            tr.ctx().route_controls().family(),
            tr.ctx().gemm_route(),
            tr.ctx().route_controls().tensor_cores(),
            tr.ctx().route_controls().f32_policy(),
            cfg.n_layers,
            tr.ctx().state_cap(),
        );
        let input = det(n, 0x91);
        let d_temporal = det(n, 0x92);
        if dtype == WeightDtype::F16 {
            let (attempts, skipped, scale) =
                warm_up_f16_training(false, || tr.step(&input, &d_temporal)).unwrap();
            eprintln!(
                "train warmup F16 eager attempts={attempts} skipped={skipped} consecutive_clean=8 last_used_loss_scale={scale}"
            );
        } else {
            for _ in 0..3 {
                tr.step(&input, &d_temporal).unwrap();
            }
        }
        tr.ctx().stream.synchronize().unwrap();
        let mut eager_skips = 0usize;
        let mut eager_scale = None;
        let t0 = Instant::now();
        for _ in 0..iters {
            let metrics = tr.step(&input, &d_temporal).unwrap();
            assert!(!metrics.graph_replayed, "eager window replayed a graph");
            eager_skips += usize::from(metrics.overflow_skipped == Some(true));
            eager_scale = metrics.loss_scale;
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
        eprintln!(
            "train metrics {dtype:?} eager skipped={eager_skips}/{iters} last_used_loss_scale={eager_scale:?}"
        );
        // Graph arm too: the M1 table is graph-mode; an eager-only M3
        // number was never comparable with it.
        tr.capture_graph().unwrap();
        if dtype == WeightDtype::F16 {
            let (attempts, skipped, scale) =
                warm_up_f16_training(true, || tr.step(&input, &d_temporal)).unwrap();
            eprintln!(
                "train warmup F16 graph attempts={attempts} skipped={skipped} consecutive_clean=8 last_used_loss_scale={scale}"
            );
        } else {
            for _ in 0..5 {
                tr.step(&input, &d_temporal).unwrap();
            }
        }
        tr.ctx().stream.synchronize().unwrap();
        let mut graph_skips = 0usize;
        let mut graph_scale = None;
        let t1 = Instant::now();
        for _ in 0..iters {
            let metrics = tr.step(&input, &d_temporal).unwrap();
            assert!(metrics.graph_replayed, "graph window used eager execution");
            graph_skips += usize::from(metrics.overflow_skipped == Some(true));
            graph_scale = metrics.loss_scale;
        }
        tr.ctx().stream.synchronize().unwrap();
        let dt = t1.elapsed().as_secs_f64();
        eprintln!(
            "train step {dtype:?} B={batch} T={seq_len} layers={}: {:.2} ms/step (graph)",
            cfg.n_layers,
            1e3 * dt / iters as f64
        );
        eprintln!(
            "train metrics {dtype:?} graph skipped={graph_skips}/{iters} last_used_loss_scale={graph_scale:?}"
        );
    }
}

/// Page measurement: pooled-graph replay latency per page at the
/// classifier serve shape (non-identity 1024->384 input projection, T=4621),
/// across the three numeric routes that matter: today's deterministic
/// f32 serve (Inference family), the typed bf16 path on the batch-invariant
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

    // (label, dtype, mode, tc, family)
    let arms: &[(&str, WeightDtype, GemmMode, bool, BiGemmFamily)] = &[
        (
            "f32 fixed (serve today)",
            WeightDtype::F32,
            GemmMode::Deterministic,
            false,
            BiGemmFamily::Inference,
        ),
        (
            "bf16 fixed (serve route)",
            WeightDtype::Bf16,
            GemmMode::Deterministic,
            false,
            BiGemmFamily::Inference,
        ),
        (
            "f32 cuBLAS (non-det ref)",
            WeightDtype::F32,
            GemmMode::CublasFast,
            false,
            BiGemmFamily::Triad,
        ),
    ];

    for &(label, dtype, mode, tc, family) in arms {
        let device = GpuDevice::new(0).expect("cuda device");
        let ctx = GpuCtx::new(&device).expect("ctx");
        ctx.set_gemm_mode(mode).unwrap();
        ctx.route_controls().set_family(family);
        if tc {
            ctx.route_controls().set_tensor_cores(true);
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

        prefill
            .run_full(
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
                mamba_rs::mamba3_siso::gpu::prefill::Mamba3PrefillOutputs {
                    last_hidden: &mut last_hidden,
                    full_temporal: None,
                    pooled_sum: Some(&mut pooled),
                },
            )
            .expect("pooled graph warmup");

        // The benchmark keeps every captured allocation alive through the graph.
        let graph = unsafe {
            Mamba3PrefillPooledGraph::capture(
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
        }
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
