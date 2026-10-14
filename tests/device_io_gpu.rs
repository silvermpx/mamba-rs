//! Rows that stay on the device give the bits of rows that cross the host,
//! in every family, precision and scan, with an input projection and with
//! an identity input.
//!
//! A trainer: the rows a forward leaves at `temporal_ptr`, the input
//! gradient at `d_input_ptr`, the optimizer moments after an applying
//! backward, a step and an accumulation window, eager and captured. Two
//! segments joined on the device, the first's rows copied into the
//! second's input and the second's input gradient into the first's output
//! gradient, train as the same stack joined through the host.
//!
//! A backbone: a prompt and decode steps from its own input buffers, eager
//! and replayed, leave the host's rows, every position of the prompt
//! included, with and without the final norm. Two segments joined on the
//! device, every position of a prompt and each step's rows copied from the
//! first into the second, give the whole backbone's rows.
//!
//! Run on a GPU box:
//!   cargo test --release --features cuda --test device_io_gpu -- --test-threads=1
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::CudaStream;
use cudarc::driver::sys::{
    CUdeviceptr, CUresult, cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2,
};
use mamba_rs::config::ScanMode;
use mamba_rs::{
    AnyGpuBackbone, AnyGpuTrainer, BackwardOpts, GemmMode, GpuBackbone, GpuOptions, GpuTrainer,
    Mamba2Config, Mamba2GpuBackbone, Mamba2GpuTrainer, Mamba2Weights, Mamba3Config,
    Mamba3GpuBackbone, Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoGpuBackbone,
    Mamba3MimoGpuTrainer, Mamba3MimoWeights, Mamba3Weights, MambaConfig, MambaGpuBackbone,
    MambaGpuTrainer, MambaWeights, TrainSessionCfg, WeightDtype,
};

const D_MODEL: usize = 32;
const INPUT_DIM: usize = 24;
const BATCH: usize = 2;
/// Past one chunk of the chunked scans, so the state passing runs.
const SEQ_LEN: usize = 80;
/// The layers of a model, and of each segment of a split one.
const LAYERS: usize = 2;
/// Decode steps after a prompt.
const STEPS: u32 = 3;
const DTYPES: [WeightDtype; 4] = [
    WeightDtype::F32,
    WeightDtype::Tf32,
    WeightDtype::Bf16,
    WeightDtype::F16,
];
const SCANS: [ScanMode; 2] = [ScanMode::Sequential, ScanMode::Parallel];
const FAMILIES: [&str; 4] = ["Mamba-1", "Mamba-2", "Mamba-3 SISO", "Mamba-3 MIMO"];

fn m1_cfg(n_layers: usize, final_norm: bool) -> MambaConfig {
    MambaConfig {
        d_model: D_MODEL,
        d_state: 8,
        n_layers,
        d_intermediate: 40,
        final_norm,
        ..MambaConfig::default()
    }
}

fn m2_cfg(n_layers: usize, final_norm: bool) -> Mamba2Config {
    Mamba2Config {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers,
        d_intermediate: 40,
        final_norm,
        ..Mamba2Config::default()
    }
}

fn siso_cfg(n_layers: usize, final_norm: bool) -> Mamba3Config {
    Mamba3Config {
        d_model: D_MODEL,
        d_state: 8,
        expand: 2,
        headdim: 8,
        ngroups: 1,
        n_layers,
        d_intermediate: 40,
        final_norm,
        ..Mamba3Config::default()
    }
}

fn mimo_cfg(n_layers: usize, final_norm: bool) -> Mamba3MimoConfig {
    Mamba3MimoConfig {
        d_model: D_MODEL,
        d_state: 16,
        headdim: 16,
        n_layers,
        d_intermediate: 40,
        final_norm,
        ..Mamba3MimoConfig::default()
    }
}

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// An output gradient of a loss's size, about 1e-2.
fn grad(n: usize, seed: u32) -> Vec<f32> {
    values(n, seed).iter().map(|v| v / 64.0).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// How a test's model is built: the family, the precision, the scan,
/// whether it ends in its final norm and whether its input is the
/// residual stream itself (no input projection).
#[derive(Clone, Copy)]
struct Build {
    family: usize,
    dtype: WeightDtype,
    scan: ScanMode,
    final_norm: bool,
    identity_input: bool,
}

impl Build {
    fn input_dim(self) -> usize {
        if self.identity_input {
            D_MODEL
        } else {
            INPUT_DIM
        }
    }

    fn opts(self) -> GpuOptions {
        GpuOptions::default()
            .dtype(self.dtype)
            .mode(GemmMode::Deterministic)
            .scan_mode(self.scan)
    }
}

/// The weights of `$cfg` from `$seed` for the build `$b`: its input width,
/// its input projection and its final norm.
macro_rules! weights {
    ($W:ident, $cfg:expr, $b:expr, $seed:expr) => {{
        let (cfg, b) = ($cfg, $b);
        let mut w = $W::init(&cfg, b.input_dim(), $seed);
        if b.identity_input {
            w.input_proj_w.clear();
            w.input_proj_b.clear();
        }
        if !b.final_norm {
            w.norm_f_weight.clear();
            w.norm_f_bias.clear();
        }
        (w, cfg)
    }};
}

/// The trainer `b` describes, from the weights seed `seed`. An f16 trainer
/// starts its loss scale at 1024, which these windows' gradients fit.
fn trainer(b: Build, seed: u64) -> AnyGpuTrainer {
    let (s, o) = (
        TrainSessionCfg::new(b.input_dim(), BATCH, SEQ_LEN),
        b.opts(),
    );
    let mut t: AnyGpuTrainer = match b.family {
        0 => {
            let (w, cfg) = weights!(MambaWeights, m1_cfg(LAYERS, b.final_norm), b, seed);
            MambaGpuTrainer::new(&w, cfg, s, o).unwrap().into()
        }
        1 => {
            let (w, cfg) = weights!(Mamba2Weights, m2_cfg(LAYERS, b.final_norm), b, seed);
            Mamba2GpuTrainer::new(&w, cfg, s, o).unwrap().into()
        }
        2 => {
            let (w, cfg) = weights!(Mamba3Weights, siso_cfg(LAYERS, b.final_norm), b, seed);
            Mamba3GpuTrainer::new(&w, cfg, s, o).unwrap().into()
        }
        _ => {
            let (w, cfg) = weights!(Mamba3MimoWeights, mimo_cfg(LAYERS, b.final_norm), b, seed);
            Mamba3MimoGpuTrainer::new(&w, cfg, s, o).unwrap().into()
        }
    };
    t.load_scaler_state(1024.0, 0).unwrap();
    t
}

/// The backbone `b` describes, from the weights seed `seed`.
fn backbone(b: Build, seed: u64) -> AnyGpuBackbone {
    let (idim, o) = (b.input_dim(), b.opts());
    match b.family {
        0 => {
            let (w, cfg) = weights!(MambaWeights, m1_cfg(LAYERS, b.final_norm), b, seed);
            MambaGpuBackbone::new(&w, cfg, idim, BATCH, o)
                .unwrap()
                .into()
        }
        1 => {
            let (w, cfg) = weights!(Mamba2Weights, m2_cfg(LAYERS, b.final_norm), b, seed);
            Mamba2GpuBackbone::new(&w, cfg, idim, BATCH, o)
                .unwrap()
                .into()
        }
        2 => {
            let (w, cfg) = weights!(Mamba3Weights, siso_cfg(LAYERS, b.final_norm), b, seed);
            Mamba3GpuBackbone::new(&w, cfg, idim, BATCH, o)
                .unwrap()
                .into()
        }
        _ => {
            let (w, cfg) = weights!(Mamba3MimoWeights, mimo_cfg(LAYERS, b.final_norm), b, seed);
            Mamba3MimoGpuBackbone::new(&w, cfg, idim, BATCH, o)
                .unwrap()
                .into()
        }
    }
}

/// A backbone of `2 * LAYERS` layers in `b`'s family, precision and scan,
/// and its two segments: the first `LAYERS` layers with the input
/// projection and without the final norm, the rest with an identity input
/// and the final norm.
fn split_backbones(b: Build, seed: u64) -> [AnyGpuBackbone; 3] {
    let o = b.opts();
    macro_rules! split {
        ($W:ident, $B:ident, $cfg:ident) => {{
            let cfg = $cfg(2 * LAYERS, true);
            let w = $W::init(&cfg, INPUT_DIM, seed);
            let mut first = w.clone();
            first.layers.truncate(LAYERS);
            first.mlp.truncate(LAYERS);
            first.norm_f_weight.clear();
            first.norm_f_bias.clear();
            let mut second = w.clone();
            second.layers.drain(..LAYERS);
            second.mlp.drain(..LAYERS.min(second.mlp.len()));
            second.input_proj_w.clear();
            second.input_proj_b.clear();
            [
                $B::new(&w, cfg, INPUT_DIM, BATCH, o).unwrap().into(),
                $B::new(&first, $cfg(LAYERS, false), INPUT_DIM, BATCH, o)
                    .unwrap()
                    .into(),
                $B::new(&second, $cfg(LAYERS, true), D_MODEL, BATCH, o)
                    .unwrap()
                    .into(),
            ]
        }};
    }
    match b.family {
        0 => split!(MambaWeights, MambaGpuBackbone, m1_cfg),
        1 => split!(Mamba2Weights, Mamba2GpuBackbone, m2_cfg),
        2 => split!(Mamba3Weights, Mamba3GpuBackbone, siso_cfg),
        _ => split!(Mamba3MimoWeights, Mamba3MimoGpuBackbone, mimo_cfg),
    }
}

fn check(r: CUresult, what: &str) {
    assert_eq!(r, CUresult::CUDA_SUCCESS, "{what}");
}

/// Host rows into the device buffer at `dst`, on `stream`.
fn put(stream: &Arc<CudaStream>, dst: CUdeviceptr, v: &[f32]) {
    let bytes: &[u8] = bytemuck::cast_slice(v);
    check(
        unsafe {
            cuMemcpyHtoDAsync_v2(dst, bytes.as_ptr().cast(), bytes.len(), stream.cu_stream())
        },
        "host to device",
    );
    stream.synchronize().unwrap();
}

/// `len` bytes of the device buffer at `src`, after `stream`.
fn get_bytes(stream: &Arc<CudaStream>, src: CUdeviceptr, len: usize) -> Vec<u8> {
    stream.synchronize().unwrap();
    let mut out = vec![0u8; len];
    check(
        unsafe { cuMemcpyDtoHAsync_v2(out.as_mut_ptr().cast(), src, len, stream.cu_stream()) },
        "device to host",
    );
    stream.synchronize().unwrap();
    out
}

/// `n` floats of the device buffer at `src`, after `stream`.
fn get(stream: &Arc<CudaStream>, src: CUdeviceptr, n: usize) -> Vec<f32> {
    stream.synchronize().unwrap();
    let mut out = vec![0f32; n];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
    check(
        unsafe {
            cuMemcpyDtoHAsync_v2(
                bytes.as_mut_ptr().cast(),
                src,
                bytes.len(),
                stream.cu_stream(),
            )
        },
        "device to host",
    );
    stream.synchronize().unwrap();
    out
}

/// `len` bytes from one model's device buffer into another's, on the
/// destination's stream after both streams: the caller's layer between two
/// segments, here the identity.
fn copy(
    (from, src): (&Arc<CudaStream>, CUdeviceptr),
    (to, dst): (&Arc<CudaStream>, CUdeviceptr),
    len: usize,
) {
    from.synchronize().unwrap();
    check(
        unsafe { cuMemcpyDtoDAsync_v2(dst, src, len, to.cu_stream()) },
        "device to device",
    );
    to.synchronize().unwrap();
}

fn moments(t: &AnyGpuTrainer) -> (Vec<u32>, Vec<u32>) {
    let blob = t.optimizer_state().unwrap();
    (bits(&blob.m), bits(&blob.v))
}

/// Every build with the final norm, with and without an input projection.
fn builds() -> Vec<Build> {
    let mut all = Vec::new();
    for scan in SCANS {
        for dtype in DTYPES {
            for family in 0..FAMILIES.len() {
                for identity_input in [false, true] {
                    all.push(Build {
                        family,
                        dtype,
                        scan,
                        final_norm: true,
                        identity_input,
                    });
                }
            }
        }
    }
    all
}

fn label(b: Build) -> String {
    format!(
        "{} {:?} {:?}{}{}",
        FAMILIES[b.family],
        b.dtype,
        b.scan,
        if b.identity_input {
            " identity input"
        } else {
            ""
        },
        if b.final_norm { "" } else { " no final norm" }
    )
}

#[test]
fn device_windows_train_as_host_windows() {
    let n_out = BATCH * SEQ_LEN * D_MODEL;
    let opts = BackwardOpts::default();
    for b in builds() {
        let case = label(b);
        let n_in = BATCH * SEQ_LEN * b.input_dim();
        let (mut host, mut dev) = (trainer(b, 7), trainer(b, 7));
        let stream = dev.ctx().stream.clone();
        for window in 0..2u32 {
            let input = values(n_in, 10 + window);
            let d = grad(n_out, 20 + window);
            let mut rows = vec![0f32; n_out];
            host.forward(&input, &mut rows).unwrap();
            put(&stream, dev.input_ptr(), &input);
            dev.forward_on_device().unwrap();
            assert_eq!(
                bits(&rows),
                bits(&get(&stream, dev.temporal_ptr(), n_out)),
                "{case}: window {window}'s rows"
            );
            put(&stream, dev.d_temporal_ptr(), &d);
            if window == 0 {
                let mut d_input = vec![0f32; n_in];
                host.backward_step_input_grad(&d, opts, &mut d_input)
                    .unwrap();
                dev.backward_step_input_grad_on_device(opts).unwrap();
                assert_eq!(
                    bits(&d_input),
                    bits(&get(&stream, dev.d_input_ptr(), n_in)),
                    "{case}: the input gradient"
                );
            } else {
                host.backward_step(&d, opts).unwrap();
                dev.backward_step_on_device(opts).unwrap();
            }
            assert!(
                moments(&host) == moments(&dev),
                "{case}: the moments after window {window}"
            );
        }
    }
}

#[test]
fn device_steps_and_windows_train_as_host_ones() {
    let n_out = BATCH * SEQ_LEN * D_MODEL;
    for b in builds() {
        let case = label(b);
        let n_in = BATCH * SEQ_LEN * b.input_dim();
        let (mut host, mut dev) = (trainer(b, 9), trainer(b, 9));
        let stream = dev.ctx().stream.clone();
        let feed = |dev: &AnyGpuTrainer, k: u32| {
            put(&stream, dev.input_ptr(), &values(n_in, 30 + k));
            put(&stream, dev.d_temporal_ptr(), &grad(n_out, 40 + k));
        };
        // Two eager steps, then an accumulation window of two micro-batches.
        for k in 0..2 {
            host.step(&values(n_in, 30 + k), &grad(n_out, 40 + k))
                .unwrap();
            feed(&dev, k);
            dev.step_on_device().unwrap();
            assert!(moments(&host) == moments(&dev), "{case}: step {k}");
        }
        for k in 2..4 {
            host.accumulate(&values(n_in, 30 + k), &grad(n_out, 40 + k))
                .unwrap();
            feed(&dev, k);
            dev.accumulate_on_device().unwrap();
        }
        host.apply_window().unwrap();
        dev.apply_window().unwrap();
        assert!(moments(&host) == moments(&dev), "{case}: the window");
        // Captured steps and windows replay what the device rows hold, with
        // no upload.
        host.capture_graph().unwrap();
        dev.capture_graph().unwrap();
        for k in 4..6 {
            let h = host
                .step(&values(n_in, 30 + k), &grad(n_out, 40 + k))
                .unwrap();
            feed(&dev, k);
            let m = dev.step_on_device().unwrap();
            assert!(h.graph_replayed && m.graph_replayed, "{case}: the replay");
            assert!(moments(&host) == moments(&dev), "{case}: captured step {k}");
        }
        for k in 6..8 {
            host.accumulate(&values(n_in, 30 + k), &grad(n_out, 40 + k))
                .unwrap();
            feed(&dev, k);
            dev.accumulate_on_device().unwrap();
        }
        host.apply_window().unwrap();
        dev.apply_window().unwrap();
        assert!(
            moments(&host) == moments(&dev),
            "{case}: the captured window"
        );
    }
}

#[test]
fn a_stack_trains_on_the_device_as_through_the_host() {
    let n_out = BATCH * SEQ_LEN * D_MODEL;
    let opts = BackwardOpts::default();
    for scan in SCANS {
        for dtype in DTYPES {
            for family in 0..FAMILIES.len() {
                let first = Build {
                    family,
                    dtype,
                    scan,
                    final_norm: false,
                    identity_input: false,
                };
                let second = Build {
                    final_norm: true,
                    identity_input: true,
                    ..first
                };
                let case = label(first);
                let n_in = BATCH * SEQ_LEN * INPUT_DIM;
                let (mut h1, mut h2) = (trainer(first, 3), trainer(second, 4));
                let (mut d1, mut d2) = (trainer(first, 3), trainer(second, 4));
                let (s1, s2) = (d1.ctx().stream.clone(), d2.ctx().stream.clone());
                for window in 0..2u32 {
                    let input = values(n_in, 50 + window);
                    let d = grad(n_out, 60 + window);
                    // Through the host.
                    let (mut cut, mut rows) = (vec![0f32; n_out], vec![0f32; n_out]);
                    h1.forward(&input, &mut cut).unwrap();
                    h2.forward(&cut, &mut rows).unwrap();
                    let mut d_cut = vec![0f32; n_out];
                    h2.backward_step_input_grad(&d, opts, &mut d_cut).unwrap();
                    h1.backward_step(&d_cut, opts).unwrap();
                    // On the device, the host touching only the window's ends.
                    put(&s1, d1.input_ptr(), &input);
                    d1.forward_on_device().unwrap();
                    copy((&s1, d1.temporal_ptr()), (&s2, d2.input_ptr()), n_out * 4);
                    d2.forward_on_device().unwrap();
                    assert_eq!(
                        bits(&rows),
                        bits(&get(&s2, d2.temporal_ptr(), n_out)),
                        "{case}: window {window}'s rows"
                    );
                    put(&s2, d2.d_temporal_ptr(), &d);
                    d2.backward_step_input_grad_on_device(opts).unwrap();
                    copy(
                        (&s2, d2.d_input_ptr()),
                        (&s1, d1.d_temporal_ptr()),
                        n_out * 4,
                    );
                    d1.backward_step_on_device(opts).unwrap();
                    assert!(
                        moments(&h1) == moments(&d1) && moments(&h2) == moments(&d2),
                        "{case}: the moments after window {window}"
                    );
                }
            }
        }
    }
}

/// Every position's rows of the backbone's last prompt, as stored.
fn prompt_rows(bb: &AnyGpuBackbone) -> Vec<u8> {
    let len = BATCH * SEQ_LEN * D_MODEL * bb.prompt_rows_dtype().size_bytes();
    get_bytes(bb.stream(), bb.prompt_rows_ptr().unwrap(), len)
}

/// The rows at `temporal_ptr` are each sequence's last prompt position:
/// its bytes in one precision, rounded to the half type where the
/// temporal rows are narrower.
fn check_last_rows(bb: &AnyGpuBackbone, case: &str) {
    let (pd, td) = (bb.prompt_rows_dtype(), bb.temporal_dtype());
    let rows = prompt_rows(bb);
    let temporal = get_bytes(
        bb.stream(),
        bb.temporal_ptr(),
        BATCH * D_MODEL * td.size_bytes(),
    );
    let (pw, tw) = (D_MODEL * pd.size_bytes(), D_MODEL * td.size_bytes());
    for s in 0..BATCH {
        let last = &rows[(s * SEQ_LEN + SEQ_LEN - 1) * pw..][..pw];
        let want: Vec<u8> = if pd == td {
            last.to_vec()
        } else {
            assert_eq!(pd, WeightDtype::F32, "{case}: the wider prompt rows");
            last.as_chunks::<4>()
                .0
                .iter()
                .flat_map(|b| {
                    let x = f32::from_le_bytes(*b);
                    match td {
                        WeightDtype::Bf16 => half::bf16::from_f32(x).to_bits().to_le_bytes(),
                        _ => half::f16::from_f32(x).to_bits().to_le_bytes(),
                    }
                })
                .collect()
        };
        assert_eq!(
            &temporal[s * tw..][..tw],
            &want[..],
            "{case}: sequence {s}'s last prompt position"
        );
    }
}

#[test]
fn device_prompts_and_steps_run_as_host_ones() {
    let mut all = builds();
    all.extend(builds().into_iter().map(|b| Build {
        final_norm: false,
        ..b
    }));
    for b in all {
        let case = label(b);
        let n_in = BATCH * SEQ_LEN * b.input_dim();
        let (mut host, mut dev) = (backbone(b, 5), backbone(b, 5));
        for replay in [false, true] {
            if replay {
                // The captures run on zero rows: the device rows go in after.
                for bb in [&mut host, &mut dev] {
                    bb.reset().unwrap();
                    bb.capture_prefill_graph(SEQ_LEN).unwrap();
                    bb.capture_graph().unwrap();
                }
            }
            let what = if replay { "replayed" } else { "eager" };
            let x = values(n_in, 70 + u32::from(replay));
            host.prefill(&x, SEQ_LEN).unwrap();
            let p = dev.prompt_input_ptr(SEQ_LEN).unwrap();
            put(dev.stream(), p, &x);
            dev.prefill_on_device(SEQ_LEN).unwrap();
            assert_eq!(dev.has_prefill_graph(), replay, "{case}: {what} prompt");
            assert!(
                prompt_rows(&host) == prompt_rows(&dev),
                "{case}: the {what} prompt's rows"
            );
            let (mut want, mut got) = (vec![0f32; BATCH * D_MODEL], vec![0f32; BATCH * D_MODEL]);
            host.download_temporal(&mut want).unwrap();
            dev.download_temporal(&mut got).unwrap();
            assert_eq!(
                bits(&want),
                bits(&got),
                "{case}: the {what} prompt's last rows"
            );
            check_last_rows(&dev, &case);
            for k in 0..STEPS {
                let x = values(BATCH * b.input_dim(), 80 + k);
                host.step(&x, &mut want).unwrap();
                put(dev.stream(), dev.step_input_ptr(), &x);
                dev.step_on_device().unwrap();
                dev.download_temporal(&mut got).unwrap();
                assert_eq!(bits(&want), bits(&got), "{case}: {what} step {k}");
            }
        }
    }
}

#[test]
fn two_segments_joined_on_the_device_run_as_the_whole_backbone() {
    let n_rows = BATCH * SEQ_LEN * D_MODEL;
    for scan in SCANS {
        for dtype in DTYPES {
            for family in 0..FAMILIES.len() {
                let b = Build {
                    family,
                    dtype,
                    scan,
                    final_norm: true,
                    identity_input: false,
                };
                let case = label(b);
                let [mut whole, mut first, mut second] = split_backbones(b, 11);
                assert_eq!(first.prompt_rows_dtype(), WeightDtype::F32, "{case}");
                assert_eq!(first.temporal_dtype(), WeightDtype::F32, "{case}");
                let x = values(BATCH * SEQ_LEN * INPUT_DIM, 90);
                whole.prefill(&x, SEQ_LEN).unwrap();
                first.prefill(&x, SEQ_LEN).unwrap();
                let dst = second.prompt_input_ptr(SEQ_LEN).unwrap();
                copy(
                    (first.stream(), first.prompt_rows_ptr().unwrap()),
                    (second.stream(), dst),
                    n_rows * 4,
                );
                second.prefill_on_device(SEQ_LEN).unwrap();
                assert!(
                    prompt_rows(&whole) == prompt_rows(&second),
                    "{case}: the prompt's rows"
                );
                let (mut want, mut got) =
                    (vec![0f32; BATCH * D_MODEL], vec![0f32; BATCH * D_MODEL]);
                whole.download_temporal(&mut want).unwrap();
                second.download_temporal(&mut got).unwrap();
                assert_eq!(bits(&want), bits(&got), "{case}: the prompt's last rows");
                for k in 0..STEPS {
                    let x = values(BATCH * INPUT_DIM, 100 + k);
                    whole.step(&x, &mut want).unwrap();
                    first.step_gpu_only(&x).unwrap();
                    copy(
                        (first.stream(), first.temporal_ptr()),
                        (second.stream(), second.step_input_ptr()),
                        BATCH * D_MODEL * 4,
                    );
                    second.step_on_device().unwrap();
                    second.download_temporal(&mut got).unwrap();
                    assert_eq!(bits(&want), bits(&got), "{case}: step {k}");
                }
            }
        }
    }
}
