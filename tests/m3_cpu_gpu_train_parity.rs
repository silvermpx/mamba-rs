//! Mamba-3 SISO CPU ↔ GPU f32 training parity.
//!
//! The M3 GPU pipeline's correctness anchors were all GPU-internal
//! (typed-vs-f32 unit parity, eager-vs-graph, chunked-vs-sequential),
//! while the finite-diff-validated CPU implementation was never compared
//! against the GPU. This test closes that gap:
//!
//!   1. Identical weights via `Mamba3Weights::init` from one seed,
//!      uploaded to the GPU (`GpuMamba3Weights::from_cpu`) and cloned
//!      into CPU `TrainMamba3Weights`.
//!   2. One forward + backward on each path with identical input and
//!      upstream gradient, all recurrent state starting at zero.
//!      The CPU path has no backbone-level entry point, so the test
//!      composes it the same way the GPU backbone does: input_proj
//!      SGEMM → each layer through `parallel_mamba3_forward`, followed by
//!      its gated MLP (`gated_mlp_rows`) when the model has one → norm_f
//!      RMSNorm, and the mirrored backward (norm_f bwd → per layer the
//!      MLP's `gated_mlp_rows_backward`, then `parallel_mamba3_backward`
//!      → input_proj bwd).
//!   3. Per-tensor comparison in the canonical `GpuMamba3Grads.flat`
//!      ordering (src/mamba3_siso/gpu/weights.rs): cosine >= 0.9999 and
//!      relative L2 error < 1e-3 per tensor — loose enough for the f32
//!      reduction-order difference between the sequential CPU scan and
//!      the batched GPU kernels, tight enough to catch a sign flip or a
//!      scale-factor bug. Element-level max-rel-err (atol-masked) is
//!      asserted at the same loose 5e-2 bound as the M1 twin test
//!      (tests/cpu_gpu_train_parity.rs).
//!
//! Both GPU scan paths are exercised:
//!   - sequential SSM (`use_parallel_scan = false`, T = 64,
//!     is_outproj_norm = false → SiLU gate branch)
//!   - chunked parallel scan (`use_parallel_scan = true`, T = 256 =
//!     4 chunks, is_outproj_norm = true → RMSNormGated branch)
//!   - the chunked backward's tiled dQKV stage, at heads and states the
//!     warp kernel cannot hold (d_state 256, headdim 48, 64 and 100)
//!   - the sequential scan's block-tier kernels (headdim 48 and 64)
//!   - the gated output norm past 1024 channels and on a head of 48
//!   - the gated MLP after every mixer, on both scans
//!   - LayerNorm block norms (`rms_norm = false`) with their biases
//!   - heads wider than a block, the gated norm past 1024 channels a head
//!
//! The chunked window is stateless by design (chunk 0 starts from zero
//! state) which matches the zero-state CPU run exactly.

#![cfg(feature = "cuda")]

#[path = "common/arch.rs"]
mod arch;

use mamba_rs::gpu::GemmMode;
use mamba_rs::gpu::buffers::GpuBuffer;
use mamba_rs::gpu::context::GpuCtx;
use mamba_rs::gpu::device::GpuDevice;
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::flat::Mamba3LayerFlat;
use mamba_rs::mamba3_siso::cpu::parallel::{
    Mamba3States, parallel_mamba3_backward, parallel_mamba3_forward,
};
use mamba_rs::mamba3_siso::cpu::weights::{TrainMamba3LayerWeights, TrainMamba3Weights};
use mamba_rs::mamba3_siso::gpu::backward::gpu_backward_mamba3_backbone;
use mamba_rs::mamba3_siso::gpu::forward::gpu_forward_mamba3_backbone;
use mamba_rs::mamba3_siso::gpu::kernels::Mamba3Kernels;
use mamba_rs::mamba3_siso::gpu::state::{
    GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3Scratch, GpuMamba3StateBufs, M3Exec,
};
use mamba_rs::mamba3_siso::gpu::weights::{GpuMamba3Grads, GpuMamba3Weights};
use mamba_rs::mamba3_siso::weights::Mamba3Weights;
use mamba_rs::mlp::GatedMlpLayerWeights;
use mamba_rs::mlp::cpu::{
    GatedMlpBackwardScratch, GatedMlpRowsScratch, gated_mlp_rows, gated_mlp_rows_backward,
    mlp_norm_row,
};
use mamba_rs::ops::blas::{sgemm_backward, sgemm_forward};
use mamba_rs::ops::fast_math::RMS_NORM_EPS;
use mamba_rs::ops::norms::{layer_norm_inplace, layer_norm_row_backward};

/// Per-tensor directional agreement floor.
const COS_MIN: f32 = 0.9999;
/// Per-tensor relative L2 error ceiling (f32 reorder headroom).
const REL_L2_MAX: f32 = 1e-3;
/// Element-level max-rel-err ceiling at atol=1e-3 masking — same loose
/// bound as the M1 twin test; per-element cancellation makes a tighter
/// bound flaky while cos/rel-L2 above stay the strict checks.
const MAX_REL_MAX: f32 = 5e-2;

fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

// ─── Scenario plumbing ──────────────────────────────────────────────────

struct Scenario {
    cfg: Mamba3Config,
    batch: usize,
    seq_len: usize,
    use_parallel_scan: bool,
}

impl Scenario {
    /// Mamba input dim — identity-sized input_proj (input_dim = d_model).
    fn input_dim(&self) -> usize {
        self.cfg.d_model
    }

    fn bt(&self) -> usize {
        self.batch * self.seq_len
    }

    fn gpu_dims(&self) -> GpuMamba3Dims {
        let cfg = &self.cfg;
        GpuMamba3Dims {
            batch: self.batch,
            d_model: cfg.d_model,
            d_inner: cfg.d_inner(),
            d_state: cfg.d_state,
            nheads: cfg.nheads(),
            headdim: cfg.headdim,
            ngroups: cfg.ngroups,
            in_proj_dim: cfg.in_proj_out_dim(),
            seq_len: self.seq_len,
            mamba_input_dim: self.input_dim(),
            n_layers: cfg.n_layers,
            n_angles: cfg.num_rope_angles(),
            a_floor: cfg.a_floor,
            rms_norm_eps: cfg.rms_norm_eps,
            ssm_norm_eps: cfg.ssm_norm_eps,
            block_norm: cfg.block_norm,
            is_outproj_norm: cfg.is_outproj_norm,
            use_parallel_scan: self.use_parallel_scan,
        }
    }
}

fn train_weights_from(w: &Mamba3Weights) -> TrainMamba3Weights {
    TrainMamba3Weights {
        input_proj_w: w.input_proj_w.clone(),
        input_proj_b: w.input_proj_b.clone(),
        layers: w
            .layers
            .iter()
            .map(|lw| TrainMamba3LayerWeights {
                norm_weight: lw.norm_weight.clone(),
                norm_bias: lw.norm_bias.clone(),
                in_proj_w: lw.in_proj_w.clone(),
                dt_bias: lw.dt_bias.clone(),
                b_norm_weight: lw.b_norm_weight.clone(),
                c_norm_weight: lw.c_norm_weight.clone(),
                b_bias: lw.b_bias.clone(),
                c_bias: lw.c_bias.clone(),
                d_param: lw.d_param.clone(),
                norm_gate_weight: lw.norm_gate_weight.clone(),
                out_proj_w: lw.out_proj_w.clone(),
            })
            .collect(),
        norm_f_weight: w.norm_f_weight.clone(),
        norm_f_bias: w.norm_f_bias.clone(),
    }
}

/// Flatten CPU grads into the `GpuMamba3Grads.flat` linear order
/// (src/mamba3_siso/gpu/weights.rs `GpuMamba3Grads::new`), the host
/// trainer's: the gated MLPs come after the mixers, norm_f last.
fn flatten_cpu_grads(g: &TrainMamba3Weights, mlp: &[GatedMlpLayerWeights]) -> Vec<f32> {
    let mut out = Vec::new();
    out.extend_from_slice(&g.input_proj_w);
    out.extend_from_slice(&g.input_proj_b);
    for l in &g.layers {
        out.extend_from_slice(&l.norm_weight);
        out.extend_from_slice(&l.norm_bias);
        out.extend_from_slice(&l.in_proj_w);
        out.extend_from_slice(&l.dt_bias);
        out.extend_from_slice(&l.b_norm_weight);
        out.extend_from_slice(&l.c_norm_weight);
        out.extend_from_slice(&l.b_bias);
        out.extend_from_slice(&l.c_bias);
        out.extend_from_slice(&l.d_param);
        out.extend_from_slice(&l.norm_gate_weight);
        out.extend_from_slice(&l.out_proj_w);
    }
    for m in mlp {
        out.extend_from_slice(&m.norm_weight);
        out.extend_from_slice(&m.norm_bias);
        out.extend_from_slice(&m.fc1_w);
        out.extend_from_slice(&m.fc2_w);
    }
    out.extend_from_slice(&g.norm_f_weight);
    out.extend_from_slice(&g.norm_f_bias);
    out
}

/// Per-tensor segment table mirroring `flatten_cpu_grads` / `GpuMamba3Grads`.
fn grad_segments(cfg: &Mamba3Config, input_dim: usize) -> Vec<(String, usize)> {
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nh = cfg.nheads();
    let ip = cfg.in_proj_out_dim();
    // A LayerNorm's bias follows its scale; an RMSNorm has none.
    let nb = cfg.block_norm.bias_len(dm);
    let mut segs = vec![
        ("input_proj_w".into(), input_dim * dm),
        ("input_proj_b".into(), dm),
    ];
    for li in 0..cfg.n_layers {
        segs.extend([
            (format!("L{li}.norm_weight"), dm),
            (format!("L{li}.norm_bias"), nb),
            (format!("L{li}.in_proj_w"), dm * ip),
            (format!("L{li}.dt_bias"), nh),
            (format!("L{li}.b_norm_weight"), ds),
            (format!("L{li}.c_norm_weight"), ds),
            (format!("L{li}.b_bias"), nh * ds),
            (format!("L{li}.c_bias"), nh * ds),
            (format!("L{li}.d_param"), nh),
            (format!("L{li}.norm_gate_weight"), di),
            (format!("L{li}.out_proj_w"), di * dm),
        ]);
    }
    let h = cfg.mlp_hidden();
    for li in (0..cfg.n_layers).filter(|_| h > 0) {
        segs.extend([
            (format!("L{li}.mlp.norm_weight"), dm),
            (format!("L{li}.mlp.norm_bias"), nb),
            (format!("L{li}.mlp.fc1_w"), dm * 2 * h),
            (format!("L{li}.mlp.fc2_w"), h * dm),
        ]);
    }
    segs.push(("norm_f_weight".into(), dm));
    segs.push(("norm_f_bias".into(), nb));
    segs.retain(|&(_, len)| len > 0);
    segs
}

// ─── Metrics ────────────────────────────────────────────────────────────

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na < 1e-20 || nb < 1e-20 {
        // Both-zero tensors (e.g. norm_gate_weight with SiLU gating) agree.
        return 1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// ||a - b||_2 / ||a||_2 with a both-zero guard.
fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut nd = 0.0f64;
    let mut na = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        nd += (x as f64 - y as f64) * (x as f64 - y as f64);
        na += x as f64 * x as f64;
    }
    if na < 1e-20 {
        return if nd < 1e-20 { 0.0 } else { f32::INFINITY };
    }
    (nd.sqrt() / na.sqrt()) as f32
}

/// Max relative error ignoring elements whose magnitude falls below `atol`
/// on both sides (f32 round-off of a near-zero gradient is 100% relative
/// error without the floor).
fn max_rel_err_masked(a: &[f32], b: &[f32], atol: f32) -> f32 {
    let mut worst = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.abs().max(y.abs()) < atol {
            continue;
        }
        let d = (x - y).abs();
        worst = worst.max(d / x.abs().max(y.abs()).max(atol));
    }
    worst
}

/// Per-segment parity assertion with diagnostics that name the offending
/// tensor instead of a flat-buffer index.
fn assert_parity(label: &str, cpu: &[f32], gpu: &[f32], segs: &[(String, usize)]) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "{label}: length mismatch CPU={} GPU={}",
        cpu.len(),
        gpu.len()
    );
    let seg_total: usize = segs.iter().map(|(_, l)| l).sum();
    assert_eq!(seg_total, cpu.len(), "{label}: segment table mismatch");

    let mut off = 0usize;
    for (name, len) in segs {
        let c = &cpu[off..off + len];
        let g = &gpu[off..off + len];
        let cs = cos_sim(c, g);
        let rl2 = rel_l2(c, g);
        let mre = max_rel_err_masked(c, g, 1e-3);
        eprintln!("{label} {name:<22} cos={cs:.6} rel_l2={rl2:.3e} max_rel@1e-3={mre:.3e} n={len}");
        assert!(cs >= COS_MIN, "{label} {name}: cos_sim {cs:.6} < {COS_MIN}");
        assert!(
            rl2 < REL_L2_MAX,
            "{label} {name}: rel_l2 {rl2:.3e} >= {REL_L2_MAX:.0e}"
        );
        assert!(
            mre < MAX_REL_MAX,
            "{label} {name}: max_rel_err@1e-3 {mre:.3e} >= {MAX_REL_MAX:.0e}"
        );
        off += len;
    }
}

// ─── CPU reference path ─────────────────────────────────────────────────

/// Row-wise RMSNorm forward (norm_f). Mirrors the GPU `rmsnorm_forward`
/// kernel math: rms = sqrt(mean(x²) + 1e-5), y = x / rms * w.
fn rmsnorm_rows_fwd(out: &mut [f32], rms: &mut [f32], x: &[f32], weight: &[f32]) {
    let dm = weight.len();
    for ((o_row, x_row), r) in out.chunks_mut(dm).zip(x.chunks(dm)).zip(rms.iter_mut()) {
        let sum_sq: f32 = x_row.iter().map(|&v| v * v).sum();
        *r = (sum_sq / dm as f32 + RMS_NORM_EPS).sqrt();
        let inv = 1.0 / *r;
        for ((o, &xv), &w) in o_row.iter_mut().zip(x_row).zip(weight) {
            *o = xv * inv * w;
        }
    }
}

/// Row-wise RMSNorm backward (norm_f). Mirrors the GPU `rmsnorm_backward`
/// kernel: dx = (w·dy − x̂·mean(dy·x̂·w)) / rms, dw += dy·x̂.
fn rmsnorm_rows_bwd(
    grads: (&mut [f32], &mut [f32]),
    dy: &[f32],
    x: &[f32],
    norm: (&[f32], &[f32]),
) {
    let (dx, dw) = grads;
    let (weight, rms) = norm;
    let dm = weight.len();
    for ((dx_row, dy_row), (x_row, &r)) in dx
        .chunks_mut(dm)
        .zip(dy.chunks(dm))
        .zip(x.chunks(dm).zip(rms))
    {
        let inv = 1.0 / r;
        let mut rowdot = 0.0f32;
        for ((&dyv, &xv), &w) in dy_row.iter().zip(x_row).zip(weight) {
            rowdot += dyv * xv * inv * w;
        }
        let mean_dy_y = rowdot / dm as f32;
        for (d, ((&dyv, &xv), (&w, dwv))) in dx_row.iter_mut().zip(
            dy_row
                .iter()
                .zip(x_row)
                .zip(weight.iter().zip(dw.iter_mut())),
        ) {
            let x_hat = xv * inv;
            *dwv += dyv * x_hat;
            *d = (w * dyv - x_hat * mean_dy_y) * inv;
        }
    }
}

/// CPU full-backbone forward + backward composed exactly like the GPU
/// backbone: input_proj → layers → norm_f, then the mirrored backward.
/// Returns (forward output, gradients incl. input_proj/norm_f, the gated
/// MLPs' gradients).
fn run_cpu(
    scn: &Scenario,
    (tw, mlp): (&TrainMamba3Weights, &[GatedMlpLayerWeights]),
    input: &[f32],
    d_out: &[f32],
) -> (Vec<f32>, TrainMamba3Weights, Vec<GatedMlpLayerWeights>) {
    let dims = Mamba3Dims::from_config(&scn.cfg, scn.seq_len);
    let bt = scn.bt();
    let dm = dims.d_model;
    let (nh, hd, ds) = (dims.nheads, dims.headdim, dims.d_state);
    let nl = dims.n_layers;
    let na = dims.num_rope_angles.max(1);
    let input_dim = scn.input_dim();

    // Forward: input_proj SGEMM (with bias) — matches GPU F0.
    let mut temporal = vec![0.0f32; bt * dm];
    sgemm_forward(
        &mut temporal,
        input,
        &tw.input_proj_w,
        Some(&tw.input_proj_b),
        bt,
        input_dim,
        dm,
    );

    // Forward: the layers one at a time (per-sample, zero initial state on
    // every buffer), each followed by its gated MLP when the model has one:
    // `x += mixer(norm(x)); x += mlp(norm2(x))`.
    let one = Mamba3Dims::from_config(
        &Mamba3Config {
            n_layers: 1,
            ..scn.cfg
        },
        scn.seq_len,
    );
    let single = |l: usize| TrainMamba3Weights {
        input_proj_w: Vec::new(),
        input_proj_b: Vec::new(),
        layers: vec![tw.layers[l].clone()],
        norm_f_weight: Vec::new(),
        norm_f_bias: Vec::new(),
    };
    let eps = scn.cfg.rms_norm_eps;
    let mut layer_acts = Vec::with_capacity(nl);
    let mut mlp_in = Vec::with_capacity(mlp.len());
    let mut mlp_scratch = GatedMlpRowsScratch::default();
    for l in 0..nl {
        let mut acts: Vec<Vec<Mamba3LayerFlat>> = (0..scn.batch)
            .map(|_| vec![Mamba3LayerFlat::zeros(one)])
            .collect();
        let mut ssm = vec![0.0f32; scn.batch * nh * hd * ds];
        let mut k_st = vec![0.0f32; scn.batch * nh * ds];
        let mut v_st = vec![0.0f32; scn.batch * nh * hd];
        let mut angle_st = vec![0.0f32; scn.batch * nh * na];
        parallel_mamba3_forward(
            &mut temporal,
            &mut acts,
            Mamba3States {
                ssm: &mut ssm,
                k: &mut k_st,
                v: &mut v_st,
                angle: &mut angle_st,
            },
            &single(l),
            &one,
            scn.batch,
        );
        layer_acts.push(acts);
        if let Some(m) = mlp.get(l) {
            mlp_in.push(temporal.clone());
            gated_mlp_rows(
                &mut temporal,
                bt,
                m,
                &mut mlp_scratch,
                sgemm_forward,
                |dst, src| {
                    for (d_row, s_row) in dst.chunks_exact_mut(dm).zip(src.chunks_exact(dm)) {
                        mlp_norm_row(d_row, s_row, m, eps);
                    }
                },
            );
        }
    }

    // Forward: norm_f, an RMSNorm or a LayerNorm (its statistics are the
    // row's [std, mean]).
    let norm_f_input = temporal.clone();
    let layer_norm = !tw.norm_f_bias.is_empty();
    let mut norm_f_stats = vec![0.0f32; bt * if layer_norm { 2 } else { 1 }];
    let mut out = vec![0.0f32; bt * dm];
    if layer_norm {
        out.copy_from_slice(&norm_f_input);
        for (row, st) in out
            .chunks_exact_mut(dm)
            .zip(norm_f_stats.as_chunks_mut::<2>().0)
        {
            *st = layer_norm_inplace(row, (&tw.norm_f_weight, &tw.norm_f_bias), eps);
        }
    } else {
        rmsnorm_rows_fwd(
            &mut out,
            &mut norm_f_stats,
            &norm_f_input,
            &tw.norm_f_weight,
        );
    }

    // Backward: norm_f → layers → input_proj.
    let mut grads = TrainMamba3Weights::zeros(&dims, input_dim);
    let mut d_temporal = vec![0.0f32; bt * dm];
    if layer_norm {
        for (((dx, dy), x), st) in d_temporal
            .chunks_exact_mut(dm)
            .zip(d_out.chunks_exact(dm))
            .zip(norm_f_input.chunks_exact(dm))
            .zip(norm_f_stats.as_chunks::<2>().0)
        {
            layer_norm_row_backward(
                dx,
                (&mut grads.norm_f_weight, &mut grads.norm_f_bias),
                (dy, x),
                &tw.norm_f_weight,
                *st,
                false,
            );
        }
    } else {
        rmsnorm_rows_bwd(
            (&mut d_temporal, &mut grads.norm_f_weight),
            d_out,
            &norm_f_input,
            (&tw.norm_f_weight, &norm_f_stats),
        );
    }

    let mut mlp_grads: Vec<GatedMlpLayerWeights> = mlp
        .iter()
        .map(|m| GatedMlpLayerWeights {
            norm_weight: vec![0.0; dm],
            norm_bias: vec![0.0; m.norm_bias.len()],
            fc1_w: vec![0.0; m.fc1_w.len()],
            fc2_w: vec![0.0; m.fc2_w.len()],
        })
        .collect();
    let mut mlp_bwd = GatedMlpBackwardScratch::default();
    for l in (0..nl).rev() {
        if let Some(m) = mlp.get(l) {
            gated_mlp_rows_backward(
                &mut d_temporal,
                &mlp_in[l],
                bt,
                (m, &mut mlp_grads[l]),
                &mut mlp_bwd,
                eps,
            );
        }
        let mut g = TrainMamba3Weights::zeros(&one, input_dim);
        parallel_mamba3_backward(
            &mut d_temporal,
            &layer_acts[l],
            &single(l),
            &mut g,
            &one,
            scn.batch,
            input_dim,
        );
        grads.layers[l] = g.layers.remove(0);
    }

    let mut d_input = vec![0.0f32; bt * input_dim];
    sgemm_backward(
        &mut d_input,
        &mut grads.input_proj_w,
        Some(&mut grads.input_proj_b),
        &d_temporal,
        input,
        &tw.input_proj_w,
        (bt, input_dim, dm),
    );

    (out, grads, mlp_grads)
}

// ─── GPU path ───────────────────────────────────────────────────────────

/// GPU full-backbone forward + backward. Returns (forward output,
/// `GpuMamba3Grads.flat` downloaded to host).
fn run_gpu(
    scn: &Scenario,
    cpu_w: &Mamba3Weights,
    input: &[f32],
    d_out: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let dev = GpuDevice::new(0).expect("GpuDevice");
    let ctx = GpuCtx::new(&dev).expect("GpuCtx");
    // Full f32 SGEMM for a sharp comparison: the production TF32 math mode
    // adds ~1e-3 relative noise per GEMM which would force tolerances loose
    // enough to mask genuine small implementation bugs. TF32 numerics are
    // covered by the GPU-internal tests; this test isolates the math.
    ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
    let m3k = Mamba3Kernels::compile_with_state_cap(
        dev.context(),
        arch::arch0(),
        mamba_rs::gpu::kernels::state_capacity(scn.cfg.d_state).unwrap(),
    )
    .expect("Mamba3Kernels");

    let dims = scn.gpu_dims();
    let bt = scn.bt();
    let (nh, hd, ds) = (dims.nheads, dims.headdim, dims.d_state);
    let nl = dims.n_layers;
    let na = dims.n_angles.max(1);

    let gpu_w = GpuMamba3Weights::from_cpu(&ctx.stream, cpu_w, &scn.cfg, scn.input_dim())
        .expect("GpuMamba3Weights::from_cpu");
    let hidden = scn.cfg.mlp_hidden();
    let mut acts = GpuMamba3BackboneActs::new(&ctx.stream, &dims)
        .and_then(|a| a.with_mlp(&ctx.stream, &dims, hidden))
        .expect("gpu acts");
    let mut scratch = GpuMamba3Scratch::new(&ctx.stream, &dims)
        .and_then(|s| s.with_mlp(&ctx.stream, &dims, hidden))
        .expect("gpu scratch");

    let mut ssm = GpuBuffer::zeros(&ctx.stream, scn.batch * nl * nh * hd * ds).unwrap();
    let mut k_st = GpuBuffer::zeros(&ctx.stream, scn.batch * nl * nh * ds).unwrap();
    let mut v_st = GpuBuffer::zeros(&ctx.stream, scn.batch * nl * nh * hd).unwrap();
    let mut angle_st = GpuBuffer::zeros(&ctx.stream, scn.batch * nl * nh * na).unwrap();

    let mut temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    let mut input_buf = GpuBuffer::zeros(&ctx.stream, bt * dims.mamba_input_dim).unwrap();
    let mut d_temporal = GpuBuffer::zeros(&ctx.stream, bt * dims.d_model).unwrap();
    input_buf.upload(&ctx.stream, input).unwrap();
    d_temporal.upload(&ctx.stream, d_out).unwrap();

    let mut grads = GpuMamba3Grads::new(&ctx.stream, &scn.cfg, scn.input_dim()).unwrap();
    grads.zero(&ctx.stream).unwrap();

    let exec = M3Exec {
        ctx: &ctx,
        kernels: &m3k,
        dims: &dims,
    };
    gpu_forward_mamba3_backbone(
        &exec,
        &mut temporal,
        &mut acts,
        &gpu_w,
        &input_buf,
        GpuMamba3StateBufs {
            ssm: &mut ssm,
            k: &mut k_st,
            v: &mut v_st,
            angle: &mut angle_st,
        },
        &mut scratch,
    )
    .expect("gpu forward");
    ctx.stream.synchronize().unwrap();
    let out = temporal.to_cpu(&ctx.stream).expect("download forward out");

    gpu_backward_mamba3_backbone(&exec, &mut d_temporal, &acts, &gpu_w, &grads, &mut scratch)
        .expect("gpu backward");
    ctx.stream.synchronize().unwrap();
    let grads_flat = grads.flat.to_cpu(&ctx.stream).expect("download grads");

    (out, grads_flat)
}

// ─── Scenario driver ────────────────────────────────────────────────────

fn check_scenario(label: &str, scn: &Scenario) {
    let input_dim = scn.input_dim();
    let n_in = scn.bt() * input_dim;
    let n_out = scn.bt() * scn.cfg.d_model;

    let mut cpu_w = Mamba3Weights::init(&scn.cfg, input_dim, 0xC0FFEE);
    // A LayerNorm's biases start at zero; drawn ones put the bias add in
    // the forward too. An RMSNorm model has none.
    let biases = cpu_w
        .layers
        .iter_mut()
        .map(|lw| &mut lw.norm_bias)
        .chain(cpu_w.mlp.iter_mut().map(|m| &mut m.norm_bias))
        .chain(std::iter::once(&mut cpu_w.norm_f_bias));
    for (k, b) in biases.enumerate() {
        let d = det(b.len(), 0x1B + k as u32, 0.2);
        b.copy_from_slice(&d);
    }
    let tw = train_weights_from(&cpu_w);

    // Identical input + upstream gradient for both paths. Scales keep
    // activations well inside f32 range and gradients above denormals.
    let input = det(n_in, 0xAA, 1.0);
    let d_out = det(n_out, 0xBB, 0.1);

    let (cpu_out, cpu_grads, cpu_mlp_grads) = run_cpu(scn, (&tw, &cpu_w.mlp), &input, &d_out);
    let (gpu_out, gpu_flat) = run_gpu(scn, &cpu_w, &input, &d_out);

    // Forward parity first — a forward mismatch makes gradient diffs
    // uninterpretable.
    let fwd_segs = vec![("forward_output".to_string(), n_out)];
    assert_parity(label, &cpu_out, &gpu_out, &fwd_segs);

    let cpu_flat = flatten_cpu_grads(&cpu_grads, &cpu_mlp_grads);
    let segs = grad_segments(&scn.cfg, input_dim);
    assert_parity(label, &cpu_flat, &gpu_flat, &segs);

    let overall_cos = cos_sim(&cpu_flat, &gpu_flat);
    let overall_rl2 = rel_l2(&cpu_flat, &gpu_flat);
    eprintln!("{label} OVERALL: cos={overall_cos:.6} rel_l2={overall_rl2:.3e}");
    assert!(overall_cos >= COS_MIN, "{label}: overall cos {overall_cos}");
    assert!(
        overall_rl2 < REL_L2_MAX,
        "{label}: overall rel_l2 {overall_rl2:.3e}"
    );
}

/// d_model=64, expand=2, headdim=16 → nheads=8; d_state=16 → 4 RoPE angle
/// pairs at rope_fraction=0.5; reference a_floor=1e-4.
fn base_cfg(is_outproj_norm: bool) -> Mamba3Config {
    Mamba3Config {
        d_model: 64,
        d_state: 16,
        expand: 2,
        headdim: 16,
        ngroups: 1,
        n_layers: 2,
        rope_fraction: 0.5,
        a_floor: 1e-4,
        is_outproj_norm,
        ..Mamba3Config::default()
    }
}

/// Sequential GPU SSM (`m3_burnin_fwd` / `m3_backward_seq`) vs CPU,
/// SiLU output gate (is_outproj_norm = false).
#[test]
fn m3_cpu_gpu_train_parity_sequential_t64() {
    let scn = Scenario {
        cfg: base_cfg(false),
        batch: 2,
        seq_len: 64,
        use_parallel_scan: false,
    };
    check_scenario("seq_t64", &scn);
}

/// Chunked parallel GPU scan (chunk_size=64, T=256 → 4 chunks) vs CPU,
/// RMSNormGated output gate (is_outproj_norm = true).
#[test]
fn m3_cpu_gpu_train_parity_chunked_t256() {
    let scn = Scenario {
        cfg: base_cfg(true),
        batch: 2,
        seq_len: 256,
        use_parallel_scan: true,
    };
    check_scenario("chunked_t256", &scn);
}

/// Large state dimension (d_state = 128, past the historical 64-register
/// ceiling): both scan modes against the CPU oracle. The kernels are
/// compiled with the matching state capacity; the CPU side has no
/// ceiling, so this is a true independent reference.
#[test]
fn m3_cpu_gpu_train_parity_chunked_large_d_state() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_state: 128,
            headdim: 16,
            ..base_cfg(true)
        },
        batch: 1,
        seq_len: 128,
        use_parallel_scan: true,
    };
    check_scenario("chunked_ds128", &scn);
}

/// The largest state headdim 16 trains chunked with: the backward's tile
/// sits just under the 99 KB opt-in. The route used to admit d_state 143,
/// whose padded tile the backward then refused at the first step.
#[test]
fn m3_cpu_gpu_train_parity_chunked_tile_at_its_limit() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_state: 142,
            headdim: 16,
            ..base_cfg(true)
        },
        batch: 1,
        seq_len: 128,
        use_parallel_scan: true,
    };
    check_scenario("chunked_ds142_hd16", &scn);
}

/// Headdim 1, the narrowest warp-tier head, on the chunked route with the
/// SiLU gate. The gated output norm is left out on purpose: over a group of
/// one element it computes y / sqrt(y^2 + eps), whose backward subtracts two
/// nearly equal terms, w * dy * (1 - y_hat^2), so an f32 rounding in y comes
/// back multiplied by (y^2 + eps) / eps, up to about 1e5. With that norm the
/// CPU and the GPU both sit within the f32 band of the float64 gradient
/// (central differences, step 1e-7: rel_l2 6.7e-4 and 1.8e-3 at d_state 16)
/// and differ from each other by about 1e-3, past this file's tolerance.
#[test]
fn m3_cpu_gpu_train_parity_chunked_headdim_one() {
    let scn = Scenario {
        cfg: Mamba3Config {
            headdim: 1,
            ..base_cfg(false)
        },
        batch: 1,
        seq_len: 128,
        use_parallel_scan: true,
    };
    check_scenario("chunked_hd1", &scn);
}

#[test]
fn m3_cpu_gpu_train_parity_sequential_large_d_state() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_state: 128,
            headdim: 16,
            ..base_cfg(false)
        },
        batch: 1,
        seq_len: 64,
        use_parallel_scan: false,
    };
    check_scenario("seq_ds128", &scn);
}

/// The top of the supported range (d_state = 256 - the reference
/// implementations' own maximum) on the SEQUENTIAL path, which has no
/// shared-memory tiles - pure register/spill, vs CPU.
#[test]
fn m3_cpu_gpu_train_parity_sequential_d_state_256() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_state: 256,
            headdim: 16,
            // The explicit sequential mode skips the chunked-backward
            // shared-memory bound — that kernel never runs here.
            scan_mode: mamba_rs::config::ScanMode::Sequential,
            ..base_cfg(false)
        },
        batch: 1,
        seq_len: 64,
        use_parallel_scan: false,
    };
    check_scenario("seq_ds256", &scn);
}

/// d_state = 256, the top of the reference range, on the chunked path.
/// The warp kernel's tiles cannot hold it; the tiled backward can.
#[test]
fn m3_cpu_gpu_train_parity_chunked_d_state_256() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_state: 256,
            headdim: 16,
            ..base_cfg(true)
        },
        batch: 1,
        seq_len: 128,
        use_parallel_scan: true,
    };
    check_scenario("chunked_ds256", &scn);
}

/// The official checkpoints' head: headdim 64 with d_state 128, two heads,
/// with the gated output norm.
#[test]
fn m3_cpu_gpu_train_parity_chunked_official_head() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_state: 128,
            headdim: 64,
            ..base_cfg(true)
        },
        batch: 2,
        seq_len: 128,
        use_parallel_scan: true,
    };
    check_scenario("chunked_hd64_ds128", &scn);
}

/// A head and a state no tile width divides, over a ragged last chunk:
/// headdim 48 (wider than a warp, not a power of two) with d_state 17.
#[test]
fn m3_cpu_gpu_train_parity_chunked_ragged_head_and_state() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_model: 48,
            d_state: 17,
            headdim: 48,
            ..base_cfg(false)
        },
        batch: 2,
        seq_len: 100,
        use_parallel_scan: true,
    };
    check_scenario("chunked_hd48_ds17", &scn);
}

/// A wide head: headdim 100 with d_state 16.
#[test]
fn m3_cpu_gpu_train_parity_chunked_wide_head() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_model: 100,
            headdim: 100,
            ..base_cfg(false)
        },
        batch: 1,
        seq_len: 128,
        use_parallel_scan: true,
    };
    check_scenario("chunked_hd100", &scn);
}

/// A head wider than a warp on the sequential scan: the block-tier forward
/// and backward, headdim 64 with the gated output norm.
#[test]
fn m3_cpu_gpu_train_parity_sequential_wide_head() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_state: 32,
            headdim: 64,
            ..base_cfg(true)
        },
        batch: 2,
        seq_len: 64,
        use_parallel_scan: false,
    };
    check_scenario("seq_hd64_ds32", &scn);
}

/// A head no warp divides on the sequential scan: headdim 48 with d_state
/// 17, over a window that is no multiple of anything.
#[test]
fn m3_cpu_gpu_train_parity_sequential_ragged_head() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_model: 48,
            d_state: 17,
            headdim: 48,
            ..base_cfg(false)
        },
        batch: 2,
        seq_len: 37,
        use_parallel_scan: false,
    };
    check_scenario("seq_hd48_ds17", &scn);
}

/// The gated output norm past 1024 channels: 1280 channels in heads of 64.
#[test]
fn m3_cpu_gpu_train_parity_gated_norm_wide_inner() {
    let scn = Scenario {
        cfg: Mamba3Config {
            d_model: 640,
            d_state: 16,
            headdim: 64,
            ..base_cfg(true)
        },
        batch: 1,
        seq_len: 64,
        use_parallel_scan: true,
    };
    check_scenario("gated_di1280_hd64", &scn);
}

/// The gated output norm on heads that are no power of two, on both scans.
#[test]
fn m3_cpu_gpu_train_parity_gated_norm_ragged_head() {
    for parallel in [true, false] {
        let scn = Scenario {
            cfg: Mamba3Config {
                d_model: 48,
                d_state: 16,
                headdim: 48,
                ..base_cfg(true)
            },
            batch: 2,
            seq_len: 37,
            use_parallel_scan: parallel,
        };
        check_scenario(&format!("gated_hd48 parallel={parallel}"), &scn);
    }
}

/// The gated MLP after every mixer, the layout of the official checkpoints,
/// on both scans with both output gates. A d_intermediate of 160 rounds up
/// to a hidden width of 256.
#[test]
fn m3_cpu_gpu_train_parity_gated_mlp() {
    for (parallel, outproj_norm) in [(true, true), (false, false)] {
        let scn = Scenario {
            cfg: Mamba3Config {
                d_intermediate: 160,
                ..base_cfg(outproj_norm)
            },
            batch: 2,
            seq_len: if parallel { 128 } else { 64 },
            use_parallel_scan: parallel,
        };
        check_scenario(&format!("mlp parallel={parallel}"), &scn);
    }
}

/// LayerNorm block norms (`rms_norm = false`): the norm before every mixer,
/// the one before every gated MLP and the final norm, each with its bias,
/// on both scans with both output gates, with and without the MLP.
#[test]
fn m3_cpu_gpu_train_parity_layer_norm() {
    for (parallel, outproj_norm, d_intermediate) in
        [(true, true, 160), (false, false, 160), (true, false, 0)]
    {
        let scn = Scenario {
            cfg: Mamba3Config {
                block_norm: mamba_rs::BlockNorm::Layer,
                d_intermediate,
                ..base_cfg(outproj_norm)
            },
            batch: 2,
            seq_len: if parallel { 128 } else { 64 },
            use_parallel_scan: parallel,
        };
        check_scenario(
            &format!("layer norm parallel={parallel} mlp={d_intermediate}"),
            &scn,
        );
    }
}

/// Heads wider than a block on both scans: the sequential kernels run them
/// in slices, the chunked forward's fallback tiles take their elements in
/// turns, and at 1100 the gated output norm runs a group past 1024
/// channels.
#[test]
fn m3_cpu_gpu_train_parity_heads_wider_than_a_block() {
    for (d_model, headdim, parallel, outproj_norm) in [
        (320usize, 640usize, true, true),
        (320, 640, false, false),
        (550, 1100, true, false),
        (550, 1100, false, true),
    ] {
        let scn = Scenario {
            cfg: Mamba3Config {
                d_model,
                d_state: 16,
                headdim,
                ..base_cfg(outproj_norm)
            },
            batch: 1,
            seq_len: if parallel { 128 } else { 40 },
            use_parallel_scan: parallel,
        };
        check_scenario(&format!("hd{headdim} parallel={parallel}"), &scn);
    }
}
