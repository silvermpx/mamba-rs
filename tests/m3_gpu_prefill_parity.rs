//! GPU chunked prefill acceptance: the prompt-window pass must agree with
//! the CPU prefill oracle, and a window split in two with state carry must
//! reproduce the unsplit window — the seam test that directly pins the
//! trapezoidal boundary fold (a missing or wrong fold shows up as a
//! first-position error orders of magnitude above float noise).
#![cfg(feature = "cuda")]

use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::prefill::{Mamba3PrefillScratch, forward_mamba3_backbone_prefill};
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::prefill::{Mamba3Prefill, Mamba3PrefillRun};
use mamba_rs::mamba3_siso::gpu::state::{GpuMamba3Dims, GpuMamba3StateBufs};
use mamba_rs::mamba3_siso::gpu::weights::GpuMamba3WeightsInf;
use mamba_rs::mamba3_siso::state::Mamba3State;
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

fn gpu_dims(cfg: &Mamba3Config, batch: usize, seq_len: usize) -> GpuMamba3Dims {
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
        mamba_input_dim: cfg.d_model,
        n_layers: cfg.n_layers,
        n_angles: cfg.num_rope_angles(),
        a_floor: cfg.a_floor,
        is_outproj_norm: cfg.is_outproj_norm,
        rms_norm_eps: cfg.rms_norm_eps,
        use_parallel_scan: true,
    }
}

/// Deterministic pseudo-random input (xorshift, seed-stable across runs).
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

/// Identity-input weights: the projection is cleared on BOTH sides so the
/// CPU oracle and the GPU run feed the same `[T * d_model]` values.
fn identity_weights(cfg: &Mamba3Config, seed: u64) -> Mamba3Weights {
    let mut w = Mamba3Weights::init(cfg, cfg.d_model, seed);
    w.input_proj_w.clear();
    w.input_proj_b.clear();
    w
}

struct GpuRig {
    ctx: GpuCtx,
    kernels: Mamba3Kernels,
}

fn rig() -> GpuRig {
    let device = GpuDevice::new(0).expect("cuda device");
    let ctx = GpuCtx::new(&device).expect("ctx");
    let arch = GpuDevice::nvrtc_arch(device.compute_capability);
    let kernels = Mamba3Kernels::compile(device.context(), arch).expect("m3 kernels");
    GpuRig { ctx, kernels }
}

struct GpuStates {
    ssm: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    angle: GpuBuffer,
}

impl GpuStates {
    fn zeros(rig: &GpuRig, dims: &GpuMamba3Dims) -> Self {
        let b = dims.batch;
        let nl = dims.n_layers;
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ds = dims.d_state;
        let na = dims.n_angles.max(1);
        let s = &rig.ctx.stream;
        Self {
            ssm: GpuBuffer::zeros(s, b * nl * nh * hd * ds).unwrap(),
            k: GpuBuffer::zeros(s, b * nl * nh * ds).unwrap(),
            v: GpuBuffer::zeros(s, b * nl * nh * hd).unwrap(),
            angle: GpuBuffer::zeros(s, b * nl * nh * na).unwrap(),
        }
    }

    fn bufs(&mut self) -> GpuMamba3StateBufs<'_> {
        GpuMamba3StateBufs {
            ssm: &mut self.ssm,
            k: &mut self.k,
            v: &mut self.v,
            angle: &mut self.angle,
        }
    }

    fn download(&self, rig: &GpuRig) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        rig.ctx.stream.synchronize().unwrap();
        let mut ssm = vec![0.0; self.ssm.len()];
        let mut k = vec![0.0; self.k.len()];
        let mut v = vec![0.0; self.v.len()];
        let mut angle = vec![0.0; self.angle.len()];
        self.ssm.download(&rig.ctx.stream, &mut ssm).unwrap();
        self.k.download(&rig.ctx.stream, &mut k).unwrap();
        self.v.download(&rig.ctx.stream, &mut v).unwrap();
        self.angle.download(&rig.ctx.stream, &mut angle).unwrap();
        (ssm, k, v, angle)
    }
}

fn gpu_prefill(
    rig: &GpuRig,
    cfg: &Mamba3Config,
    w: &Mamba3Weights,
    input: &[f32],
    seq_len: usize,
    states: &mut GpuStates,
    carry_state: bool,
) -> Vec<f32> {
    let dims = gpu_dims(cfg, 1, seq_len);
    let gw = GpuMamba3WeightsInf::from_cpu(&rig.ctx.stream, w, cfg.d_model).unwrap();
    let gpu_input = GpuBuffer::from_cpu(&rig.ctx.stream, input).unwrap();
    let mut prefill = Mamba3Prefill::new(&rig.ctx.stream, &dims).unwrap();
    let mut last_hidden = GpuBuffer::zeros(&rig.ctx.stream, cfg.d_model).unwrap();
    prefill
        .run(
            &Mamba3PrefillRun {
                ctx: &rig.ctx,
                kernels: &rig.kernels,
                dims: &dims,
                weights: &gw,
                mamba_input: &gpu_input,
                identity_proj: true,
                carry_state,
            },
            states.bufs(),
            &mut last_hidden,
        )
        .unwrap();
    rig.ctx.stream.synchronize().unwrap();
    let mut out = vec![0.0; cfg.d_model];
    last_hidden.download(&rig.ctx.stream, &mut out).unwrap();
    out
}

fn cpu_oracle(
    cfg: &Mamba3Config,
    w: &Mamba3Weights,
    input: &[f32],
    seq_len: usize,
) -> (Vec<f32>, Mamba3State) {
    let dims = Mamba3Dims::from_config(cfg, seq_len);
    let mut state = Mamba3State::zeros(cfg);
    let mut scratch = Mamba3PrefillScratch::new(&dims, cfg.d_model);
    let mut out = vec![0.0f32; seq_len * cfg.d_model];
    forward_mamba3_backbone_prefill(&mut out, input, w, &mut state.layers, &mut scratch, &dims);
    let last = out[(seq_len - 1) * cfg.d_model..].to_vec();
    (last, state)
}

fn assert_close(tag: &str, a: &[f32], b: &[f32], rel_tol: f32) {
    assert_eq!(a.len(), b.len(), "{tag}: length");
    let (mut dot, mut na, mut nb, mut num, mut den) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
        num += ((x - y) as f64) * ((x - y) as f64);
        den += (y as f64) * (y as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    let rel = (num / den.max(1e-30)).sqrt();
    assert!(
        cos > 0.9999 && rel < rel_tol as f64,
        "{tag}: cos={cos:.6} rel_l2={rel:.3e} (tol {rel_tol:.1e})"
    );
}

/// Chunked GPU prefill vs the CPU prefill oracle across whole-chunk,
/// partial-tail and sub-chunk window lengths.
#[test]
fn gpu_prefill_matches_cpu_oracle() {
    let cfg = tiny_cfg();
    let w = identity_weights(&cfg, 7);
    let rig = rig();
    for seq_len in [1usize, 63, 64, 65, 127, 192] {
        let input = det_input(seq_len * cfg.d_model, 42 + seq_len as u64);
        let (cpu_last, cpu_state) = cpu_oracle(&cfg, &w, &input, seq_len);
        let mut states = GpuStates::zeros(&rig, &gpu_dims(&cfg, 1, seq_len));
        let gpu_last = gpu_prefill(&rig, &cfg, &w, &input, seq_len, &mut states, false);
        assert_close(
            &format!("last_hidden T={seq_len}"),
            &gpu_last,
            &cpu_last,
            1e-3,
        );
        let (ssm, k, v, _angle) = states.download(&rig);
        let nl = cfg.n_layers;
        let per_ssm = ssm.len() / nl;
        let per_k = k.len() / nl;
        let per_v = v.len() / nl;
        for l in 0..nl {
            let ls = &cpu_state.layers[l];
            assert_close(
                &format!("ssm_state T={seq_len} L{l}"),
                &ssm[l * per_ssm..(l + 1) * per_ssm],
                &ls.ssm_state,
                1e-3,
            );
            assert_close(
                &format!("k_state T={seq_len} L{l}"),
                &k[l * per_k..(l + 1) * per_k],
                &ls.k_state,
                1e-3,
            );
            assert_close(
                &format!("v_state T={seq_len} L{l}"),
                &v[l * per_v..(l + 1) * per_v],
                &ls.v_state,
                1e-3,
            );
        }
    }
}

/// THE seam test: a window split in two with state carry must reproduce
/// the unsplit window. A missing (or mis-signed) trapezoidal boundary
/// fold fails this by orders of magnitude; float reassociation across the
/// different chunkings stays far below the tolerance. Both a chunk-aligned
/// and an unaligned split are pinned.
#[test]
fn gpu_prefill_split_window_reproduces_full_window() {
    let cfg = tiny_cfg();
    let w = identity_weights(&cfg, 11);
    let rig = rig();
    let total = 192usize;
    let input = det_input(total * cfg.d_model, 99);
    for split in [128usize, 100] {
        let mut full_states = GpuStates::zeros(&rig, &gpu_dims(&cfg, 1, total));
        let full_last = gpu_prefill(&rig, &cfg, &w, &input, total, &mut full_states, false);
        let (f_ssm, f_k, f_v, f_angle) = full_states.download(&rig);

        let mut states = GpuStates::zeros(&rig, &gpu_dims(&cfg, 1, total));
        let head = &input[..split * cfg.d_model];
        let tail = &input[split * cfg.d_model..];
        let _ = gpu_prefill(&rig, &cfg, &w, head, split, &mut states, false);
        let split_last = gpu_prefill(&rig, &cfg, &w, tail, total - split, &mut states, true);
        let (s_ssm, s_k, s_v, s_angle) = states.download(&rig);

        assert_close(
            &format!("split={split} last_hidden"),
            &split_last,
            &full_last,
            1e-4,
        );
        assert_close(&format!("split={split} ssm"), &s_ssm, &f_ssm, 1e-4);
        assert_close(&format!("split={split} k"), &s_k, &f_k, 1e-4);
        assert_close(&format!("split={split} v"), &s_v, &f_v, 1e-4);
        assert_close(&format!("split={split} angle"), &s_angle, &f_angle, 1e-4);
    }
}
