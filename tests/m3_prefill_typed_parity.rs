//! The TYPED (bf16/f16) m3 prefill acceptance gate - the G8 lane of the
//! 0.6.10 program.
//!
//! T1: the typed prefill's post-norm_f temporal must equal the MIXED
//!     trainer forward's temporal BIT FOR BIT (the prefill mirrors
//!     `forward_mixed` launch for launch; the prism checkpoint was
//!     trained through that exact chain).
//! T2: the pooled column sum stays a pure-f32 ascending-t fold in the
//!     typed lane too (a bf16 accumulator over thousands of rows would
//!     saturate its mantissa - this pins the f32 colsum invariant).
//! T3: last_hidden equals the temporal's final row.
//! T5: the pooled CUDA-graph replay equals the eager pooled path, and a
//!     replay against a DIFFERENT weights container is refused (the
//!     graph baked the captured container's pointers in).
//! L2: per-sample batch invariance of the pooled row on the serve route
//!     (batch_invariant + tensor cores): a page's pooled sum is
//!     bit-identical alone and inside a batch.
//! T7: the device `cast_f32_to_bf16` equals the host `half::bf16::from_f32`
//!     RNE bit for bit - which is what makes "the serve runs the
//!     trainer's compute shadow" a literal statement (trainer casts on
//!     device, serve casts on host).
#![cfg(feature = "cuda")]

use cudarc::driver::PushKernelArg;
use mamba_rs::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use mamba_rs::mamba_ssm::gpu::trainer::TrainSessionCfg;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::prefill::{
    Mamba3Prefill, Mamba3PrefillOutputs, Mamba3PrefillPooledGraph, Mamba3PrefillRun,
};
use mamba_rs::mamba3_siso::gpu::state::{GpuMamba3Dims, GpuMamba3StateBufs};
use mamba_rs::mamba3_siso::gpu::trainer::Mamba3Trainer;
use mamba_rs::mamba3_siso::gpu::weights::{GpuMamba3MixedWeights, GpuMamba3WeightsInf};
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn tiny_cfg() -> Mamba3Config {
    Mamba3Config {
        d_model: 32,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 1e-4,
        is_outproj_norm: false,
        ..Mamba3Config::default()
    }
}

fn gpu_dims(cfg: &Mamba3Config, input_dim: usize, batch: usize, seq_len: usize) -> GpuMamba3Dims {
    GpuMamba3Dims {
        batch,
        d_model: cfg.d_model,
        d_inner: cfg.d_inner(),
        d_state: cfg.d_state,
        nheads: cfg.nheads(),
        headdim: cfg.headdim,
        ngroups: cfg.ngroups,
        in_proj_dim: cfg.in_proj_out_dim(),
        seq_len,
        mamba_input_dim: input_dim,
        n_layers: cfg.n_layers,
        n_angles: cfg.num_rope_angles(),
        a_floor: cfg.a_floor,
        is_outproj_norm: cfg.is_outproj_norm,
        rms_norm_eps: cfg.rms_norm_eps,
        use_parallel_scan: true,
    }
}

fn det_input(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFFFF) as f32 / 16777216.0) * 2.0 - 1.0
        })
        .collect()
}

fn assert_bitwise(tag: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    let bad = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(bad, 0, "{tag}: {bad}/{} values differ bitwise", got.len());
}

struct StateSet {
    ssm: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    a: GpuBuffer,
}

impl StateSet {
    fn zeros(ctx: &GpuCtx, cfg: &Mamba3Config, batch: usize) -> Self {
        let nl = cfg.n_layers;
        let nh = cfg.nheads();
        StateSet {
            ssm: GpuBuffer::zeros(&ctx.stream, batch * nl * nh * cfg.headdim * cfg.d_state)
                .unwrap(),
            k: GpuBuffer::zeros(&ctx.stream, batch * nl * nh * cfg.d_state).unwrap(),
            v: GpuBuffer::zeros(&ctx.stream, batch * nl * nh * cfg.headdim).unwrap(),
            a: GpuBuffer::zeros(&ctx.stream, batch * nl * nh * cfg.num_rope_angles().max(1))
                .unwrap(),
        }
    }
    fn bufs(&mut self) -> GpuMamba3StateBufs<'_> {
        GpuMamba3StateBufs {
            ssm: &mut self.ssm,
            k: &mut self.k,
            v: &mut self.v,
            angle: &mut self.a,
        }
    }
}

/// T1 + T2 + T3, per typed dtype, with a NON-identity input projection
/// (input_dim != d_model - the prism shape class; the typed prefill casts
/// the input itself).
#[test]
fn typed_prefill_matches_mixed_trainer_forward() {
    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
        let cfg = tiny_cfg();
        let input_dim = 24usize;
        let seq_len = 192usize;
        let dm = cfg.d_model;
        let w = Mamba3Weights::init(&cfg, input_dim, 0x00C0_FFEE);
        let input = det_input(seq_len * input_dim, 77);

        // MIXED trainer-forward reference temporal: the exact numeric
        // chain the prism checkpoint was optimized against.
        let mut trainer = Mamba3Trainer::new_full(
            0,
            &w,
            cfg,
            TrainSessionCfg {
                input_dim,
                batch: 1,
                seq_len,
                lr: 1e-5,
                weight_decay: 0.0,
            },
            dtype,
        )
        .expect("m3 mixed trainer");
        let mut reference = vec![0.0f32; seq_len * dm];
        trainer.reset_state().unwrap();
        trainer.forward(&input, &mut reference).unwrap();
        // Vacuous-green guard: a zeroed reference would bitwise-match a
        // zeroed prefill output without testing anything.
        let nonzero = reference.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > reference.len() / 2,
            "trainer reference is {nonzero}/{} nonzero - harness defect",
            reference.len()
        );

        // Typed prefill on a fresh context against the mixed container.
        let device = GpuDevice::new(0).expect("cuda device");
        let ctx = GpuCtx::new(&device).expect("ctx");
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = Mamba3Kernels::compile(device.context(), arch).expect("m3 kernels");
        let dims = gpu_dims(&cfg, input_dim, 1, seq_len);
        let mw = GpuMamba3MixedWeights::from_cpu(&ctx.stream, &w, dtype).unwrap();
        let gpu_input = GpuBuffer::from_cpu(&ctx.stream, &input).unwrap();
        let mut prefill = Mamba3Prefill::new_with_dtype(&ctx.stream, &dims, dtype).unwrap();
        let mut last_hidden = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
        let mut full = GpuBuffer::zeros(&ctx.stream, seq_len * dm).unwrap();
        let mut pooled = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
        let mut states = StateSet::zeros(&ctx, &cfg, 1);

        prefill
            .run_full(
                &Mamba3PrefillRun {
                    ctx: &ctx,
                    kernels: &kernels,
                    dims: &dims,
                    weights: &mw,
                    mamba_input: &gpu_input,
                    identity_proj: false,
                    carry_state: false,
                },
                states.bufs(),
                Mamba3PrefillOutputs {
                    last_hidden: &mut last_hidden,
                    full_temporal: Some(&mut full),
                    pooled_sum: Some(&mut pooled),
                },
            )
            .unwrap();
        ctx.stream.synchronize().unwrap();

        let mut full_h = vec![0.0f32; seq_len * dm];
        full.download(&ctx.stream, &mut full_h).unwrap();
        let mut pooled_h = vec![0.0f32; dm];
        pooled.download(&ctx.stream, &mut pooled_h).unwrap();
        let mut last_h = vec![0.0f32; dm];
        last_hidden.download(&ctx.stream, &mut last_h).unwrap();

        assert_bitwise(
            &format!("{dtype:?} full_temporal vs mixed trainer forward"),
            &full_h,
            &reference,
        );
        assert_bitwise(
            &format!("{dtype:?} last_hidden vs final temporal row"),
            &last_h,
            &full_h[(seq_len - 1) * dm..],
        );
        let mut host_sum = vec![0.0f32; dm];
        for row in full_h.chunks(dm) {
            for (acc, &v) in host_sum.iter_mut().zip(row) {
                *acc += v;
            }
        }
        assert_bitwise(
            &format!("{dtype:?} pooled_sum vs ascending-t host fold"),
            &pooled_h,
            &host_sum,
        );
    }
}

/// T5: typed pooled graph replay == eager, and the replay guard refuses a
/// different weights container.
#[test]
fn typed_pooled_graph_replay_and_container_guard() {
    let dtype = WeightDtype::Bf16;
    let cfg = tiny_cfg();
    let input_dim = 24usize;
    let seq_len = 128usize;
    let dm = cfg.d_model;
    let w = Mamba3Weights::init(&cfg, input_dim, 0xB16_B00B);
    let device = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&device).expect("ctx");
    let arch = GpuDevice::nvrtc_arch(device.compute_capability);
    let kernels = Mamba3Kernels::compile(device.context(), arch).expect("m3 kernels");
    let dims = gpu_dims(&cfg, input_dim, 1, seq_len);
    let mw = GpuMamba3MixedWeights::from_cpu(&ctx.stream, &w, dtype).unwrap();
    let fw = GpuMamba3WeightsInf::from_cpu(&ctx.stream, &w, input_dim).unwrap();
    let mut gpu_input =
        GpuBuffer::from_cpu(&ctx.stream, &det_input(seq_len * input_dim, 5)).unwrap();
    let mut prefill = Mamba3Prefill::new_with_dtype(&ctx.stream, &dims, dtype).unwrap();
    let mut last_hidden = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
    let mut pooled = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
    let mut states = StateSet::zeros(&ctx, &cfg, 1);

    // Every captured allocation remains alive through graph destruction.
    let graph = unsafe {
        Mamba3PrefillPooledGraph::capture(
            &mut prefill,
            &Mamba3PrefillRun {
                ctx: &ctx,
                kernels: &kernels,
                dims: &dims,
                weights: &mw,
                mamba_input: &gpu_input,
                identity_proj: false,
                carry_state: false,
            },
            states.bufs(),
            &mut last_hidden,
            &mut pooled,
        )
    }
    .unwrap();

    // Negative first: a replay against the OTHER container must refuse.
    let err = graph
        .replay(&ctx, &kernels, &fw, &gpu_input, &states.bufs(), &pooled)
        .unwrap_err();
    assert!(
        err.contains("weights container"),
        "container-guard error text: {err}"
    );

    // Replay == eager on a fresh page.
    let page2 = det_input(seq_len * input_dim, 6);
    gpu_input.upload(&ctx.stream, &page2).unwrap();
    graph
        .replay(&ctx, &kernels, &mw, &gpu_input, &states.bufs(), &pooled)
        .unwrap();
    ctx.stream.synchronize().unwrap();
    let mut pooled_graph = vec![0.0f32; dm];
    pooled.download(&ctx.stream, &mut pooled_graph).unwrap();

    prefill
        .run_full(
            &Mamba3PrefillRun {
                ctx: &ctx,
                kernels: &kernels,
                dims: &dims,
                weights: &mw,
                mamba_input: &gpu_input,
                identity_proj: false,
                carry_state: false,
            },
            states.bufs(),
            Mamba3PrefillOutputs {
                last_hidden: &mut last_hidden,
                full_temporal: None,
                pooled_sum: Some(&mut pooled),
            },
        )
        .unwrap();
    ctx.stream.synchronize().unwrap();
    let mut pooled_eager = vec![0.0f32; dm];
    pooled.download(&ctx.stream, &mut pooled_eager).unwrap();
    assert_bitwise("typed pooled graph vs eager", &pooled_graph, &pooled_eager);
}

/// L2 on the serve route: a page's pooled row is bit-identical alone
/// (B=1) and inside a batch (B=2), under batch_invariant + tensor cores -
/// the strict all-M contract the G4 ladder established.
#[test]
fn typed_pooled_batch_invariance_on_the_serve_route() {
    let dtype = WeightDtype::Bf16;
    let cfg = tiny_cfg();
    let input_dim = 24usize;
    let seq_len = 160usize;
    let dm = cfg.d_model;
    let w = Mamba3Weights::init(&cfg, input_dim, 0xFACE);
    let device = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&device).expect("ctx");
    ctx.set_batch_invariant(true);
    ctx.set_bi_tensor_cores(true);
    let arch = GpuDevice::nvrtc_arch(device.compute_capability);
    let kernels = Mamba3Kernels::compile(device.context(), arch).expect("m3 kernels");
    let mw = GpuMamba3MixedWeights::from_cpu(&ctx.stream, &w, dtype).unwrap();

    let page_a = det_input(seq_len * input_dim, 41);
    let page_b = det_input(seq_len * input_dim, 42);

    let pooled_single = |page: &[f32]| -> Vec<f32> {
        let dims = gpu_dims(&cfg, input_dim, 1, seq_len);
        let gpu_input = GpuBuffer::from_cpu(&ctx.stream, page).unwrap();
        let mut prefill = Mamba3Prefill::new_with_dtype(&ctx.stream, &dims, dtype).unwrap();
        let mut last_hidden = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
        let mut pooled = GpuBuffer::zeros(&ctx.stream, dm).unwrap();
        let mut states = StateSet::zeros(&ctx, &cfg, 1);
        prefill
            .run_full(
                &Mamba3PrefillRun {
                    ctx: &ctx,
                    kernels: &kernels,
                    dims: &dims,
                    weights: &mw,
                    mamba_input: &gpu_input,
                    identity_proj: false,
                    carry_state: false,
                },
                states.bufs(),
                Mamba3PrefillOutputs {
                    last_hidden: &mut last_hidden,
                    full_temporal: None,
                    pooled_sum: Some(&mut pooled),
                },
            )
            .unwrap();
        ctx.stream.synchronize().unwrap();
        let mut h = vec![0.0f32; dm];
        pooled.download(&ctx.stream, &mut h).unwrap();
        h
    };
    let solo_a = pooled_single(&page_a);
    let solo_b = pooled_single(&page_b);

    let dims2 = gpu_dims(&cfg, input_dim, 2, seq_len);
    let mut both = page_a.clone();
    both.extend_from_slice(&page_b);
    let gpu_input = GpuBuffer::from_cpu(&ctx.stream, &both).unwrap();
    let mut prefill = Mamba3Prefill::new_with_dtype(&ctx.stream, &dims2, dtype).unwrap();
    let mut last_hidden = GpuBuffer::zeros(&ctx.stream, 2 * dm).unwrap();
    let mut pooled = GpuBuffer::zeros(&ctx.stream, 2 * dm).unwrap();
    let mut states = StateSet::zeros(&ctx, &cfg, 2);
    prefill
        .run_full(
            &Mamba3PrefillRun {
                ctx: &ctx,
                kernels: &kernels,
                dims: &dims2,
                weights: &mw,
                mamba_input: &gpu_input,
                identity_proj: false,
                carry_state: false,
            },
            states.bufs(),
            Mamba3PrefillOutputs {
                last_hidden: &mut last_hidden,
                full_temporal: None,
                pooled_sum: Some(&mut pooled),
            },
        )
        .unwrap();
    ctx.stream.synchronize().unwrap();
    let mut batched = vec![0.0f32; 2 * dm];
    pooled.download(&ctx.stream, &mut batched).unwrap();

    assert_bitwise("sample A pooled: solo vs batched", &batched[..dm], &solo_a);
    assert_bitwise("sample B pooled: solo vs batched", &batched[dm..], &solo_b);
}

/// T7: the device f32->bf16 cast (the trainer's compute-shadow writer
/// path) equals the host RNE cast (the serve's upload path) bit for bit,
/// including subnormals, exact ties, zeros and infinities. NaN inputs
/// only need to stay NaN (weights are finite; payload canonicalization
/// may differ).
#[test]
fn device_cast_f32_to_bf16_matches_host_rne() {
    let device = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&device).expect("ctx");
    let arch = GpuDevice::nvrtc_arch(device.compute_capability);
    let kernels = Mamba3Kernels::compile(device.context(), arch).expect("m3 kernels");

    let mut vals: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE / 2.0,
        1.0e-45,
        3.4e38,
        // Exact-tie mantissas around the bf16 rounding boundary.
        f32::from_bits(0x3F80_8000),
        f32::from_bits(0x3F81_8000),
        f32::from_bits(0x3F80_8001),
    ];
    let mut s = 0x5EEDu64;
    for _ in 0..4096 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        vals.push(f32::from_bits((s as u32) & 0x7FBF_FFFF));
        vals.push(-f32::from_bits((s >> 32) as u32 & 0x7FBF_FFFF));
    }
    let n = vals.len();

    let src = GpuBuffer::from_cpu(&ctx.stream, &vals).unwrap();
    let dst = DtypedBuf::zeros(&ctx.stream, n, WeightDtype::Bf16).unwrap();
    {
        let n_i = n as i32;
        let d = dst.cached_ptr();
        let sp = src.cached_ptr();
        let mut b = ctx.stream.launch_builder(&kernels.cast_f32_to_bf16);
        b.arg(&d);
        b.arg(&sp);
        b.arg(&n_i);
        unsafe { b.launch(mamba_rs::mamba_ssm::gpu::launch::grid_1d(n)) }.unwrap();
    }
    let mut widened = vec![0.0f32; n];
    dst.download_f32(&ctx.stream, &mut widened).unwrap();

    let mut bad = 0usize;
    for (i, &v) in vals.iter().enumerate() {
        let host = half::bf16::from_f32(v).to_f32();
        let dev = widened[i];
        if v.is_nan() {
            assert!(dev.is_nan(), "NaN input {i} did not stay NaN on device");
            continue;
        }
        if host.to_bits() != dev.to_bits() {
            bad += 1;
            if bad <= 5 {
                eprintln!(
                    "cast mismatch at {i}: in={v:e} host={:#010x} dev={:#010x}",
                    host.to_bits(),
                    dev.to_bits()
                );
            }
        }
    }
    assert_eq!(bad, 0, "{bad}/{n} device casts differ from host RNE");
}
