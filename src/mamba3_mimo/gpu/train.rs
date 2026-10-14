//! Mamba-3 MIMO training on the GPU, in f32, tf32, bf16 or f16, by the
//! chunked path (`kernels/mamba3_mimo_chunked_bwd.cu`) or, with
//! `ScanMode::Sequential`, the sequential one.
//!
//! The forward runs the inference engine's launches, the chunked scan (its
//! per-token coefficients, angles, entering states and pre-gate outputs
//! kept) or the checkpointing recurrence (each head's state written every
//! `stride` tokens), and keeps per layer what the backward reads: the
//! layer's input rows and norm, the normed rows, the projection, the
//! normalised B and C, the gated rows, and for the MLP its input, norm,
//! normed rows, up projection and activation. The backward walks the
//! layers in reverse: the MLP, the out_proj GEMM, the scan backward (the
//! chunked one, or each segment replayed from its checkpoint, see
//! `kernels/mamba3_mimo_bwd.cu`), the B and C norms, the angle projection,
//! the in_proj GEMM and the block norm.
//!
//! The lanes are the Mamba-2 trainer's: tf32 keeps everything f32 with the
//! context's TF32 GEMM contract; bf16 and f16 keep the weights, gradients,
//! moments, residual stream, states and every reduction in f32, run the
//! GEMMs from half copies of the matrices refreshed after every step, and
//! store the rows those GEMMs read or write in the half type; f16 scales
//! the loss with the dynamic loss scaler.
//!
//! Windows carry the states from one call to the next, and the gradients
//! stop at the states a window starts from.

mod graph;

use crate::config::ModelFamily;
use crate::gpu::grad_clip::{GradRegionGeom, StepTail, norm_limit, read_tail, scale_grads};
use crate::gpu::grad_window::{FoldOverflow, GradWindow, check_backward_opts};
use crate::gpu::options::GpuOptions;
use crate::gpu::train::{BackwardMetrics, BackwardOpts, StepMetrics, TrainSessionCfg};
use crate::gpu::train_state::TrainerState;
use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::inference::{MimoChunkedPtrs, MimoExec, launch_chunked_scan, row_pieces};
use super::kernels::{
    CHUNK, ChunkedGeom, Mamba3MimoKernels, ScanBwdGlobalGeom, ScanGeom, chunked_geom,
    scan_bwd_global_geom, scan_bwd_record_floats, scan_bwd_smem_bytes, scan_bwd_threads, scan_geom,
};
use crate::config::ScanMode;
use crate::gpu::adamw::{
    AdamWBiasFactors, AdamWMultiPlan, AdamWPlanKey, AdamWStateBlob, AdamWTensorSpec, GpuAdamW,
    build_multi_plan, step_multi,
};
use crate::gpu::blas::{
    TypedPtr, gpu_gemm_bi_backward_dw_grad_typed, gpu_gemm_bi_backward_grad_raw,
    gpu_gemm_ex_backward_dx_typed, gpu_gemm_f32_forward_ptrs, gpu_gemm_typed_forward_raw,
};
use crate::gpu::block_norm::{self, NormParams};
use crate::gpu::buffers::{DtypedBuf, GpuBuffer, GpuByteBuffer, GradSlice};
use crate::gpu::context::{GemmRole, GpuCtx, gemm_mode_or_env};
use crate::gpu::device::GpuDevice;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::gemm_bi_inference::prepare_inference_arch_rung;
use crate::gpu::kernel_identity::PreparedGemmCaptureManifest;
use crate::gpu::launch::{grid_1d, grid_1d_cover, launch_cast, launch_elems, y_runs, yz_runs};
use crate::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, UnscaleFactor, check_inf_nan_gpu, scale_grads_skip_gpu,
};
use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::weights::Mamba3MimoWeights;
use crate::matrix::{Path, Request};

/// A tensor's place in the flat buffers, in f32 elements.
#[derive(Clone, Copy, Debug, Default)]
struct Span {
    off: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct LayerSpans {
    norm_weight: Span,
    norm_bias: Span,
    in_proj_w: Span,
    dt_bias: Span,
    b_norm_weight: Span,
    c_norm_weight: Span,
    b_bias: Span,
    c_bias: Span,
    d_param: Span,
    mimo_x: Span,
    mimo_z: Span,
    mimo_o: Span,
    out_proj_w: Span,
    /// The output norm's weight. Every model holds it, as the host
    /// trainer's state does, so the moments of both devices share one
    /// layout; without the norm (`is_outproj_norm` off) its gradient stays
    /// zero.
    norm_gate: Span,
}

#[derive(Clone, Copy, Debug, Default)]
struct MlpSpans {
    norm_weight: Span,
    norm_bias: Span,
    fc1_w: Span,
    fc2_w: Span,
}

/// The order of the flat buffers: the input projection, each layer's
/// mixer tensors, each layer's MLP, the final norm.
#[derive(Clone, Debug)]
struct Layout {
    input_proj_w: Span,
    input_proj_b: Span,
    layers: Vec<LayerSpans>,
    mlp: Vec<MlpSpans>,
    norm_f_weight: Span,
    norm_f_bias: Span,
    total: usize,
}

impl Layout {
    fn of(w: &Mamba3MimoWeights) -> Self {
        let mut at = 0usize;
        let mut take = |len: usize| {
            let span = Span { off: at, len };
            at += len;
            span
        };
        let input_proj_w = take(w.input_proj_w.len());
        let input_proj_b = take(w.input_proj_b.len());
        let layers = w
            .layers
            .iter()
            .map(|lw| LayerSpans {
                norm_weight: take(lw.norm_weight.len()),
                norm_bias: take(lw.norm_bias.len()),
                in_proj_w: take(lw.in_proj_w.len()),
                dt_bias: take(lw.dt_bias.len()),
                b_norm_weight: take(lw.b_norm_weight.len()),
                c_norm_weight: take(lw.c_norm_weight.len()),
                b_bias: take(lw.b_bias.len()),
                c_bias: take(lw.c_bias.len()),
                d_param: take(lw.d_param.len()),
                mimo_x: take(lw.mimo_x.len()),
                mimo_z: take(lw.mimo_z.len()),
                mimo_o: take(lw.mimo_o.len()),
                out_proj_w: take(lw.out_proj_w.len()),
                norm_gate: take(lw.norm_gate_weight.len()),
            })
            .collect();
        let mlp = w
            .mlp
            .iter()
            .map(|m| MlpSpans {
                norm_weight: take(m.norm_weight.len()),
                norm_bias: take(m.norm_bias.len()),
                fc1_w: take(m.fc1_w.len()),
                fc2_w: take(m.fc2_w.len()),
            })
            .collect();
        let norm_f_weight = take(w.norm_f_weight.len());
        let norm_f_bias = take(w.norm_f_bias.len());
        Self {
            input_proj_w,
            input_proj_b,
            layers,
            mlp,
            norm_f_weight,
            norm_f_bias,
            total: at,
        }
    }

    /// Every (tensor, span, decays) of `w` in layout order; the matrices
    /// decay under the reference's no-decay groups.
    fn pairs<'w>(&self, w: &'w Mamba3MimoWeights) -> Vec<(&'w [f32], Span, bool)> {
        let mut out = vec![
            (w.input_proj_w.as_slice(), self.input_proj_w, true),
            (w.input_proj_b.as_slice(), self.input_proj_b, false),
        ];
        for (lw, ls) in w.layers.iter().zip(&self.layers) {
            out.extend([
                (lw.norm_weight.as_slice(), ls.norm_weight, false),
                (lw.norm_bias.as_slice(), ls.norm_bias, false),
                (lw.in_proj_w.as_slice(), ls.in_proj_w, true),
                (lw.dt_bias.as_slice(), ls.dt_bias, false),
                (lw.b_norm_weight.as_slice(), ls.b_norm_weight, false),
                (lw.c_norm_weight.as_slice(), ls.c_norm_weight, false),
                (lw.b_bias.as_slice(), ls.b_bias, false),
                (lw.c_bias.as_slice(), ls.c_bias, false),
                (lw.d_param.as_slice(), ls.d_param, false),
                (lw.mimo_x.as_slice(), ls.mimo_x, true),
                (lw.mimo_z.as_slice(), ls.mimo_z, true),
                (lw.mimo_o.as_slice(), ls.mimo_o, true),
                (lw.out_proj_w.as_slice(), ls.out_proj_w, true),
                (lw.norm_gate_weight.as_slice(), ls.norm_gate, false),
            ]);
        }
        for (m, ms) in w.mlp.iter().zip(&self.mlp) {
            out.extend([
                (m.norm_weight.as_slice(), ms.norm_weight, false),
                (m.norm_bias.as_slice(), ms.norm_bias, false),
                (m.fc1_w.as_slice(), ms.fc1_w, true),
                (m.fc2_w.as_slice(), ms.fc2_w, true),
            ]);
        }
        out.push((w.norm_f_weight.as_slice(), self.norm_f_weight, false));
        out.push((w.norm_f_bias.as_slice(), self.norm_f_bias, false));
        out
    }

    fn flatten(&self, w: &Mamba3MimoWeights) -> Vec<f32> {
        let mut flat = vec![0f32; self.total];
        for (src, span, _) in self.pairs(w) {
            flat[span.off..span.off + span.len].copy_from_slice(src);
        }
        flat
    }

    fn unflatten(&self, flat: &[f32], template: &Mamba3MimoWeights) -> Mamba3MimoWeights {
        let mut w = template.clone();
        let read = |v: &mut Vec<f32>, span: Span| {
            v.copy_from_slice(&flat[span.off..span.off + span.len]);
        };
        read(&mut w.input_proj_w, self.input_proj_w);
        read(&mut w.input_proj_b, self.input_proj_b);
        for (lw, ls) in w.layers.iter_mut().zip(&self.layers) {
            read(&mut lw.norm_weight, ls.norm_weight);
            read(&mut lw.norm_bias, ls.norm_bias);
            read(&mut lw.in_proj_w, ls.in_proj_w);
            read(&mut lw.dt_bias, ls.dt_bias);
            read(&mut lw.b_norm_weight, ls.b_norm_weight);
            read(&mut lw.c_norm_weight, ls.c_norm_weight);
            read(&mut lw.b_bias, ls.b_bias);
            read(&mut lw.c_bias, ls.c_bias);
            read(&mut lw.d_param, ls.d_param);
            read(&mut lw.mimo_x, ls.mimo_x);
            read(&mut lw.mimo_z, ls.mimo_z);
            read(&mut lw.mimo_o, ls.mimo_o);
            read(&mut lw.out_proj_w, ls.out_proj_w);
            read(&mut lw.norm_gate_weight, ls.norm_gate);
        }
        for (m, ms) in w.mlp.iter_mut().zip(&self.mlp) {
            read(&mut m.norm_weight, ms.norm_weight);
            read(&mut m.norm_bias, ms.norm_bias);
            read(&mut m.fc1_w, ms.fc1_w);
            read(&mut m.fc2_w, ms.fc2_w);
        }
        read(&mut w.norm_f_weight, self.norm_f_weight);
        read(&mut w.norm_f_bias, self.norm_f_bias);
        w
    }
}

fn at(base: u64, span: Span) -> u64 {
    base + (span.off * 4) as u64
}

/// The control region of the gradient arena, as in SISO: the dd_dt, dd_A,
/// trap and angle columns of every layer's in_proj (the last
/// `3 * nheads + angles` of its split), then dt_bias. These are the
/// gradients of the step size and the decay, whose magnitude grows with
/// the window's length. The layers repeat with one stride, since the
/// layout holds the mixers one after another and the MLPs after them.
fn control_region_geom(layout: &Layout, cfg: &Mamba3MimoConfig) -> GradRegionGeom {
    let Some(l0) = layout.layers.first() else {
        return GradRegionGeom {
            n_layers: 0,
            layer_stride: 0,
            a_off: 0,
            a_rows: 0,
            a_cols: 0,
            a_row_stride: 0,
            b_off: 0,
            b_len: 0,
            c_off: 0,
            c_len: 0,
        };
    };
    let (dm, ip) = (cfg.d_model, cfg.in_proj_out_dim());
    let ctrl = 3 * cfg.nheads() + cfg.num_rope_angles();
    GradRegionGeom {
        n_layers: layout.layers.len() as i32,
        layer_stride: layout
            .layers
            .get(1)
            .map_or(0, |l1| (l1.norm_weight.off - l0.norm_weight.off) as i64),
        a_off: (l0.in_proj_w.off + ip - ctrl) as i64,
        a_rows: dm as i32,
        a_cols: ctrl as i32,
        a_row_stride: ip as i64,
        b_off: l0.dt_bias.off as i64,
        b_len: l0.dt_bias.len as i32,
        c_off: 0,
        c_len: 0,
    }
}

/// Rows in the activation dtype: f32 buffers in the f32 and tf32 lanes,
/// half buffers otherwise.
enum Rows {
    F32(GpuBuffer),
    Half(DtypedBuf),
}

impl Rows {
    fn zeros(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        n: usize,
        act: WeightDtype,
    ) -> Result<Self, String> {
        if act == WeightDtype::F32 {
            GpuBuffer::zeros(stream, n.max(1)).map(Rows::F32)
        } else {
            DtypedBuf::zeros(stream, n.max(1), act).map(Rows::Half)
        }
    }

    fn ptr(&self) -> u64 {
        match self {
            Rows::F32(b) => b.cached_ptr(),
            Rows::Half(b) => b.cached_ptr(),
        }
    }

    fn f32(&self) -> Result<&GpuBuffer, String> {
        match self {
            Rows::F32(b) => Ok(b),
            Rows::Half(_) => Err("f32 rows asked of a half lane".into()),
        }
    }

    fn f32_mut(&mut self) -> Result<&mut GpuBuffer, String> {
        match self {
            Rows::F32(b) => Ok(b),
            Rows::Half(_) => Err("f32 rows asked of a half lane".into()),
        }
    }
}

/// What one layer's forward keeps for its backward.
struct LayerActs {
    x_in: GpuBuffer,
    rms: GpuBuffer,
    post: Rows,
    proj: Rows,
    bn: GpuBuffer,
    cn: GpuBuffer,
    o: Rows,
    /// The sequential path's replay checkpoints.
    ck_ssm: GpuBuffer,
    ck_k: GpuBuffer,
    ck_v: GpuBuffer,
    ck_ang: GpuBuffer,
    /// The chunked path's step scalars, angles, first-token lead, entering
    /// states, readouts, and the carried keys and input the window started
    /// from.
    coef: GpuBuffer,
    sn: GpuBuffer,
    cs: GpuBuffer,
    e0: GpuBuffer,
    enter: GpuBuffer,
    y: GpuBuffer,
    k0: GpuBuffer,
    v0: GpuBuffer,
    mlp: Option<MlpActs>,
}

struct MlpActs {
    x_in: GpuBuffer,
    rms: GpuBuffer,
    post: Rows,
    up: Rows,
    act: Rows,
}

/// The backward's buffers, one layer's worth, reused across layers.
struct BwdScratch {
    d_rows: GpuBuffer,
    d_temp: GpuBuffer,
    d_temp_h: Option<DtypedBuf>,
    d_o: Rows,
    d_proj: Rows,
    d_post: Rows,
    d_pre: GpuBuffer,
    d_act: Rows,
    d_up: Rows,
    d_post2: Rows,
    d_bn_h: GpuBuffer,
    d_cn_h: GpuBuffer,
    d_tanh_h: GpuBuffer,
    rec: GpuBuffer,
    /// The global sequential backward's scratch: dS, the key gradients,
    /// the [P, R] tiles (each one element when in shared memory), the
    /// channels' own accumulators and the block sums' leaves.
    bwd_ds: GpuBuffer,
    bwd_kg: GpuBuffer,
    bwd_pr: GpuBuffer,
    bwd_ch: GpuBuffer,
    bwd_leaves: GpuBuffer,
    g_x: GpuBuffer,
    g_z: GpuBuffer,
    g_o: GpuBuffer,
    g_d: GpuBuffer,
    g_dtb: GpuBuffer,
    g_bb: GpuBuffer,
    g_cb: GpuBuffer,
    dw_b: GpuBuffer,
    dw_c: GpuBuffer,
    axis0: GpuBuffer,
    d_input: GpuBuffer,
    /// The chunked path's backward buffers.
    cb: ChunkedBwd,
}

/// The chunked backward's buffers (one element each on the sequential
/// path).
struct ChunkedBwd {
    dy: GpuBuffer,
    du: GpuBuffer,
    dq: GpuBuffer,
    dk: GpuBuffer,
    dsl: GpuBuffer,
    d_init: GpuBuffer,
    dot: GpuBuffer,
    inter: GpuBuffer,
    dcs: GpuBuffer,
    dwp: GpuBuffer,
    dsc3: GpuBuffer,
    dth: GpuBuffer,
    ddt_ang: GpuBuffer,
    de0: GpuBuffer,
    gpsi_init: GpuBuffer,
    gphi: GpuBuffer,
    gz: GpuBuffer,
    gpsi: GpuBuffer,
    gd: GpuBuffer,
    gdtb: GpuBuffer,
    tmp: GpuBuffer,
    /// The output norm's weight partials per row `[rows, d_inner]`.
    gw: GpuBuffer,
    /// The generic state terms' keys and inputs, the generic pair terms'
    /// staged rows and the lead's block-sum leaves, each in global memory
    /// only when the head needs it (one element otherwise).
    terms_glob: GpuBuffer,
    pairs_glob: GpuBuffer,
    lead_leaves: GpuBuffer,
}

/// Tokens per block of the generic state terms kernel.
const TERMS_GROUP: usize = 8;

/// A MIMO trainer's carried recurrence on the host: every layer's SSM
/// state, the K and V of the last step, and the accumulated angles.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba3MimoRecurrentStateBlob {
    pub ssm: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub angle: Vec<f32>,
}

/// The MIMO trainer: forward, backward and AdamW for windows of `seq_len`
/// tokens of `batch` sequences.
pub struct Mamba3MimoGpuTrainer {
    /// The captured training step; dropped before the buffers it runs on.
    graph: Option<graph::TrainGraph>,
    /// The GEMM inventory of the last eager step, which a capture takes.
    eager_gemm_manifest: Option<PreparedGemmCaptureManifest>,
    pub kernels: Mamba3MimoKernels,
    pub ctx: GpuCtx,
    cfg: Mamba3MimoConfig,
    batch: usize,
    seq_len: usize,
    input_dim: usize,
    /// Tokens between the sequential scan's checkpoints.
    stride: usize,
    /// The chunked path's geometry when it runs: every ScanMode but
    /// Sequential, when its kernels fit the device (Auto otherwise runs the
    /// recurrence).
    chunked: Option<ChunkedGeom>,
    /// The sequential forward's launch: the whole head per block, or tiles
    /// of its channels, and its counters (see the engine's).
    scan: ScanGeom,
    /// The sequential backward's global kernel and its launch, when the
    /// head does not fit the shared one.
    seq_bwd: Option<ScanBwdGlobalGeom>,
    scan_done: GpuByteBuffer,
    /// The chunked forward's chunk contributions, one layer's worth.
    contrib: GpuBuffer,
    dtype: WeightDtype,
    act: WeightDtype,
    template: Mamba3MimoWeights,
    layout: Layout,
    master: GpuBuffer,
    grads: GpuBuffer,
    adamw: GpuAdamW,
    state_ssm: GpuBuffer,
    state_k: GpuBuffer,
    state_v: GpuBuffer,
    state_ang: GpuBuffer,
    acts: Vec<LayerActs>,
    input: GpuBuffer,
    temporal: GpuBuffer,
    final_in: GpuBuffer,
    final_rms: GpuBuffer,
    final_out: GpuBuffer,
    out_rows: Rows,
    bwd: BwdScratch,
    mirror: Option<DtypedBuf>,
    mirror_pairs: Vec<(Span, Span)>,
    scaler: Option<DynamicLossScaler>,
    overflow: OverflowFlag,
    /// The skip flag a measured step tail writes (see [`StepTail`]).
    skip_flag: OverflowFlag,
    /// The options of the fused step, eager or captured.
    step_opts: BackwardOpts,
    /// The f16 loss scale on the device, which the backward multiplies the
    /// upstream gradient by.
    loss_scale: GpuBuffer,
    unscale: UnscaleFactor,
    overflowed: bool,
    /// Set by a completed forward, cleared when one starts.
    forwarded: bool,
    /// Set by a completed backward, consumed by the next optimizer step.
    fresh_grads: bool,
    /// The accumulation window and the clip of [`Self::backward_step`].
    window: GradWindow,
    /// The fused AdamW's chunk table, and the optimizer settings it bakes
    /// in: a weight decay, reference_no_decay or moment buffer changed
    /// through `adamw_mut` rebuilds it before the next step.
    multi_plan: Option<(AdamWPlanKey, AdamWMultiPlan)>,
    /// The AdamW bias corrections and learning rate, on the device.
    bias: AdamWBiasFactors,
}

/// Tokens between the sequential scan's checkpoints that [`Mamba3MimoGpuTrainer::new`]
/// takes.
pub const DEFAULT_CHECKPOINT_STRIDE: usize = 16;

impl Mamba3MimoGpuTrainer {
    /// Build a trainer for the session `session` names (shape and AdamW
    /// settings, see [`TrainSessionCfg::new`]) from `w`, on the device, in
    /// the storage precision (f32, tf32, bf16 or f16) and in the GEMM mode
    /// `opts` names, checkpointing the sequential scan every
    /// [`DEFAULT_CHECKPOINT_STRIDE`] tokens.
    pub fn new(
        w: &Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        session: TrainSessionCfg,
        opts: GpuOptions,
    ) -> Result<Self, String> {
        Self::new_with_stride(w, cfg, session, opts, DEFAULT_CHECKPOINT_STRIDE)
    }

    /// [`Self::new`] checkpointing the sequential scan every `stride`
    /// tokens: a shorter stride replays less and keeps more checkpoints.
    pub fn new_with_stride(
        w: &Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        session: TrainSessionCfg,
        opts: GpuOptions,
        stride: usize,
    ) -> Result<Self, String> {
        let mut cfg = cfg;
        cfg.scan_mode = opts.scan_mode_or(cfg.scan_mode);
        let TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr,
            weight_decay,
        } = session;
        let dtype = opts.dtype;
        w.validate(&cfg, input_dim)?;
        let request = Request {
            path: Path::Train,
            batch,
            seq_len,
            scan: cfg.scan_mode,
            precision: dtype,
            gemm: gemm_mode_or_env(opts.mode)?,
        };
        crate::mamba3_mimo::route::admit(&cfg, input_dim, &request)?;
        let device = GpuDevice::new(opts.device)?;
        let device = &device;
        let stride = stride.clamp(1, seq_len);
        let (hd, ds, rank, na) = (
            cfg.headdim,
            cfg.d_state,
            cfg.mimo_rank,
            cfg.num_rope_angles(),
        );
        let role = GemmRole::triad(dtype);
        let ctx = match opts.mode {
            Some(mode) => GpuCtx::new_with_state_cap_mode_and_role(device, 64, mode, role)?,
            None => GpuCtx::new_from_env_with_state_cap_and_role(device, 64, role)?,
        };
        let act = match dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => WeightDtype::F32,
            half => half,
        };
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = Mamba3MimoKernels::compile(device.context(), arch, cfg.mimo_rank)?;
        let optin = kernels.smem_optin;
        let scan = scan_geom((hd, ds, rank, na), optin, kernels.scan_threads).ok_or_else(|| {
            format!(
                "not one channel of a rank-{rank} scan with {ds} states fits this GPU's \
                 {optin} bytes of shared memory"
            )
        })?;
        let fitted = chunked_geom((hd, ds, rank), optin, true, kernels.chunk_threads);
        let chunked = if cfg.train_use_parallel_scan() {
            fitted
        } else {
            None
        };
        if chunked.is_none() && cfg.scan_mode == ScanMode::Parallel {
            return Err(format!(
                "the chunked training scan of a rank-{rank} head of {hd} channels and {ds} \
                 states does not fit this GPU's {optin} bytes of shared memory; \
                 ScanMode::Auto trains it by the recurrence"
            ));
        }
        // The sequential backward: the head in shared memory when it fits
        // there and in the kernel's block, else the global kernel.
        let seq_bwd = (scan_bwd_smem_bytes((hd, ds, rank, na)) > optin
            || scan_bwd_threads(hd) > kernels.scan_bwd_block)
            .then(|| {
                scan_bwd_global_geom((hd, ds, rank, na), kernels.scan_bwd_global_block, optin)
            });
        let layout = Layout::of(w);
        let s = &ctx.stream;
        let scan_done = GpuByteBuffer::zeros(s, (batch * cfg.nheads() * 4).max(4))?;
        let z = |n: usize| GpuBuffer::zeros(s, n.max(1));
        let typed = |n: usize| Rows::zeros(s, n, act);
        let c = &cfg;
        let rows = batch * seq_len;
        let (dm, ip, di, bc, nh) = (
            c.d_model,
            c.in_proj_out_dim(),
            c.d_inner(),
            c.bc_width(),
            c.nheads(),
        );
        let hidden = c.mlp_hidden();
        // Each block norm keeps one statistic per row for RMSNorm, two for
        // LayerNorm.
        let stats = c.block_norm.stats_per_row();
        let nseg = seq_len.div_ceil(stride);
        let nc = seq_len.div_ceil(CHUNK);
        // Each path's buffers; the other path's are left one element long.
        let seq_n = |n: usize| if chunked.is_some() { 1 } else { n };
        let ch_n = |n: usize| if chunked.is_some() { n } else { 1 };
        // The gate's backward buffers: the chunked path's, and the output
        // norm's on both paths.
        let gate_n = |n: usize| {
            if chunked.is_some() || cfg.is_outproj_norm {
                n
            } else {
                1
            }
        };
        let state_n = batch * nc * nh * hd * ds;
        let acts = (0..c.n_layers)
            .map(|l| -> Result<LayerActs, String> {
                Ok(LayerActs {
                    x_in: z(rows * dm)?,
                    rms: z(rows * stats)?,
                    post: typed(rows * dm)?,
                    proj: typed(rows * ip)?,
                    bn: z(rows * bc)?,
                    cn: z(rows * bc)?,
                    o: typed(rows * di)?,
                    ck_ssm: z(seq_n(batch * nseg * nh * hd * ds))?,
                    ck_k: z(seq_n(batch * nseg * nh * rank * ds))?,
                    ck_v: z(seq_n(batch * nseg * nh * hd))?,
                    ck_ang: z(seq_n(batch * nseg * nh * na))?,
                    coef: z(ch_n(rows * nh * 4))?,
                    sn: z(ch_n(rows * nh * na))?,
                    cs: z(ch_n(rows * nh * na))?,
                    e0: z(ch_n(batch * nh))?,
                    enter: z(ch_n(state_n))?,
                    y: z(if chunked.is_some() || cfg.is_outproj_norm {
                        rows * nh * rank * hd
                    } else {
                        1
                    })?,
                    k0: z(ch_n(batch * nh * rank * ds))?,
                    v0: z(ch_n(batch * nh * hd))?,
                    mlp: if l < w.mlp.len() {
                        Some(MlpActs {
                            x_in: z(rows * dm)?,
                            rms: z(rows * stats)?,
                            post: typed(rows * dm)?,
                            up: typed(rows * 2 * hidden)?,
                            act: typed(rows * hidden)?,
                        })
                    } else {
                        None
                    },
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let per_bh = batch * nh;
        // The global sequential backward's buffers: one element each unless
        // that kernel runs and keeps the group in global memory.
        let global = |keep: fn(&ScanBwdGlobalGeom) -> bool, n: usize| match seq_bwd {
            Some(g) if chunked.is_none() && keep(&g) => n,
            _ => 1,
        };
        let bwd = BwdScratch {
            d_rows: z(rows * dm)?,
            d_temp: z(rows * dm)?,
            d_temp_h: if act == WeightDtype::F32 {
                None
            } else {
                Some(DtypedBuf::zeros(s, rows * dm, act)?)
            },
            d_o: typed(rows * di)?,
            d_proj: typed(rows * ip)?,
            d_post: typed(rows * dm)?,
            d_pre: z(rows * dm)?,
            d_act: typed(rows * hidden)?,
            d_up: typed(rows * 2 * hidden)?,
            d_post2: typed(rows * dm)?,
            d_bn_h: z(rows * nh * rank * ds)?,
            d_cn_h: z(rows * nh * rank * ds)?,
            d_tanh_h: z(rows * nh * na)?,
            rec: z(seq_n(
                per_bh * stride * scan_bwd_record_floats(hd, ds, rank, na),
            ))?,
            bwd_ds: z(global(|g| !g.ds_shared, per_bh * hd * ds))?,
            bwd_kg: z(global(|g| !g.keys_shared, per_bh * 5 * rank * ds))?,
            bwd_pr: z(global(|g| !g.tiles_shared, per_bh * 3 * hd * rank))?,
            bwd_ch: z(global(|_| true, per_bh * 2 * hd))?,
            bwd_leaves: z(global(|g| g.wide, per_bh * 4 * hd.next_multiple_of(32)))?,
            g_x: z(per_bh * rank * hd)?,
            g_z: z(per_bh * rank * hd)?,
            g_o: z(per_bh * rank * hd)?,
            g_d: z(per_bh)?,
            g_dtb: z(per_bh)?,
            g_bb: z(per_bh * rank * ds)?,
            g_cb: z(per_bh * rank * ds)?,
            dw_b: z(rows * bc)?,
            dw_c: z(rows * bc)?,
            axis0: z(rows * dm)?,
            d_input: z(rows * input_dim)?,
            cb: ChunkedBwd {
                dy: z(gate_n(rows * nh * rank * hd))?,
                du: z(gate_n(rows * nh * rank * hd))?,
                dq: z(ch_n(rows * nh * rank * ds))?,
                dk: z(ch_n(rows * nh * rank * ds))?,
                dsl: z(ch_n(state_n))?,
                d_init: z(ch_n(batch * nh * hd * ds))?,
                dot: z(ch_n(batch * nc * nh))?,
                inter: z(ch_n(rows * nh))?,
                dcs: z(ch_n(rows * nh))?,
                dwp: z(ch_n(batch * nc * nh * CHUNK * (CHUNK + 1) / 2))?,
                dsc3: z(ch_n(rows * nh * 3))?,
                dth: z(ch_n(rows * nh * na))?,
                ddt_ang: z(ch_n(rows * nh))?,
                de0: z(ch_n(batch * nh))?,
                gpsi_init: z(ch_n(batch * nh * rank * hd))?,
                gphi: z(gate_n(rows * nh * rank * hd))?,
                gz: z(gate_n(rows * nh * rank * hd))?,
                gpsi: z(ch_n(rows * nh * rank * hd))?,
                gd: z(gate_n(rows * nh * hd))?,
                gdtb: z(ch_n(rows * nh))?,
                tmp: z(gate_n(nh * rank * hd))?,
                gw: z(if cfg.is_outproj_norm { rows * di } else { 1 })?,
                terms_glob: z(chunked.map_or(1, |_| {
                    if (rank * ds + rank * hd) * 4 > optin {
                        batch * nc * nh * CHUNK.div_ceil(TERMS_GROUP) * (rank * ds + rank * hd)
                    } else {
                        1
                    }
                }))?,
                pairs_glob: z(chunked.map_or(1, |g| {
                    if g.pairs_global {
                        batch * nc * nh * g.pairs.0 * rank * (3 * ds + 2 * hd + 2)
                    } else {
                        1
                    }
                }))?,
                lead_leaves: z(chunked.map_or(1, |_| {
                    if hd.next_multiple_of(32) > kernels.chunk_threads.generic {
                        batch * nh * hd.next_multiple_of(32)
                    } else {
                        1
                    }
                }))?,
            },
        };
        let mut mirror_pairs = Vec::new();
        let mut copy_at = 0usize;
        if act != WeightDtype::F32 {
            let mats = layout
                .layers
                .iter()
                .flat_map(|ls| [ls.in_proj_w, ls.out_proj_w])
                .chain(layout.mlp.iter().flat_map(|ms| [ms.fc1_w, ms.fc2_w]));
            for span in mats {
                mirror_pairs.push((
                    span,
                    Span {
                        off: copy_at,
                        len: span.len,
                    },
                ));
                copy_at += span.len;
            }
        }
        let mirror = if act == WeightDtype::F32 {
            None
        } else {
            Some(DtypedBuf::zeros(s, copy_at.max(1), act)?)
        };
        let per_layer = |n: usize| z(c.n_layers * batch * nh * n);
        let contrib = z(ch_n(state_n))?;
        let trainer = Self {
            graph: None,
            eager_gemm_manifest: None,
            forwarded: false,
            fresh_grads: false,
            window: GradWindow::new(s)?,
            multi_plan: None,
            bias: AdamWBiasFactors::new(s)?,
            master: GpuBuffer::from_cpu(s, &layout.flatten(w))?,
            grads: z(layout.total)?,
            adamw: GpuAdamW::new(s, layout.total)?
                .with_lr(lr)
                .with_weight_decay(weight_decay),
            state_ssm: per_layer(hd * ds)?,
            state_k: per_layer(rank * ds)?,
            state_v: per_layer(hd)?,
            state_ang: per_layer(na)?,
            acts,
            input: z(rows * input_dim)?,
            temporal: z(rows * dm)?,
            final_in: z(rows * dm)?,
            final_rms: z(rows * stats)?,
            final_out: z(rows * dm)?,
            out_rows: typed(rows * dm)?,
            bwd,
            mirror,
            mirror_pairs,
            scaler: (dtype == WeightDtype::F16).then(DynamicLossScaler::new),
            overflow: OverflowFlag::new(s)?,
            skip_flag: OverflowFlag::new(s)?,
            step_opts: BackwardOpts::default(),
            loss_scale: z(1)?,
            unscale: UnscaleFactor::new(s)?,
            overflowed: false,
            layout,
            template: w.clone(),
            kernels,
            ctx,
            cfg,
            batch,
            seq_len,
            input_dim,
            stride,
            chunked,
            scan,
            scan_done,
            seq_bwd,
            contrib,
            dtype,
            act,
        };
        trainer.refresh_mirror()?;
        Ok(trainer)
    }

    /// The model's configuration.
    pub fn config(&self) -> &Mamba3MimoConfig {
        &self.cfg
    }

    /// Sequences per window.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Tokens per window.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Input features per token.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// A host slice of a window must hold `width` values per token row.
    fn check_window_len(&self, what: &str, len: usize, width: usize) -> Result<(), String> {
        let want = self.batch * self.seq_len * width;
        if len == want {
            Ok(())
        } else {
            Err(format!(
                "{what} holds {len} values; this trainer's window takes {want}"
            ))
        }
    }

    /// The storage dtype this trainer runs.
    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    /// The optimizer, to set its learning rate, betas, epsilon and decay.
    pub fn adamw_mut(&mut self) -> &mut GpuAdamW {
        &mut self.adamw
    }

    /// Set the AdamW learning rate for the next steps.
    pub fn set_lr(&mut self, lr: f32) -> Result<(), String> {
        if !lr.is_finite() || lr <= 0.0 {
            return Err(format!("set_lr: invalid learning rate {lr}"));
        }
        self.adamw.lr = lr;
        Ok(())
    }

    /// The AdamW learning rate.
    pub fn lr(&self) -> f32 {
        self.adamw.lr
    }

    /// The CUDA context: the GEMM mode and route live here.
    pub fn ctx(&self) -> &GpuCtx {
        &self.ctx
    }

    /// Width of the model, the rows `forward` writes.
    pub fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    /// One training step on a window: [`Self::forward`] of `input`
    /// (`[batch, seq_len, input_dim]`) without the download, then
    /// [`Self::backward_step`] of `d_temporal` (`[batch, seq_len, d_model]`,
    /// d loss / d the final rows of every position) with the step options
    /// (see [`Self::set_step_opts`]).
    ///
    /// With a captured graph (see [`Self::capture_graph`]) the step
    /// replays it, the same options baked in. With an accumulation window
    /// open the step adds its micro-batch to the window and applies it, as
    /// [`Self::accumulate`] then [`Self::apply_window`] would.
    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        self.check_window_len("step input", input.len(), self.input_dim)?;
        self.check_window_len("step d_temporal", d_temporal.len(), self.cfg.d_model)?;
        // A step with a window open adds its micro-batch to the window and
        // applies it, as the split calls would close the window.
        if self.window.is_open() {
            self.accumulate(input, d_temporal)?;
            return self.apply_window();
        }
        if self.graph.is_some() {
            return self.step_graph(input, d_temporal);
        }
        if self.has_gemm_work() {
            prepare_inference_arch_rung(&self.ctx)?;
        }
        self.presize_gemm_scratch()?;
        let mut metrics = None;
        let manifest = GpuCtx::record_owner_eager_gemm_manifest(
            self,
            |t| &t.ctx,
            |t| {
                t.forwarded = false;
                t.forward_unchecked(input, None)?;
                t.forwarded = true;
                let opts = t.step_opts;
                metrics = Some(t.backward_step(d_temporal, opts)?);
                Ok(())
            },
        )?;
        if self.graph.is_none() {
            self.eager_gemm_manifest = Some(manifest);
        }
        let Some(m) = metrics else {
            return Err("the eager step returned no metrics".into());
        };
        Ok(StepMetrics::from_backward(m, false))
    }

    /// One micro-batch of an accumulation window in one call: the forward
    /// of `input` and the backward of `d_temporal`, as [`Self::step`] takes
    /// them, the gradient added to the window, which the first call opens;
    /// AdamW waits for [`Self::apply_window`]. In f16 the loss scale holds
    /// for the window, and an overflowed micro-batch marks it, so the apply
    /// skips the whole window. With captured graphs (see
    /// [`Self::capture_graph`]) the call replays one, with the eager bits.
    pub fn accumulate(
        &mut self,
        input: &[f32],
        d_temporal: &[f32],
    ) -> Result<BackwardMetrics, String> {
        self.check_window_len("accumulate input", input.len(), self.input_dim)?;
        self.check_window_len("accumulate d_temporal", d_temporal.len(), self.cfg.d_model)?;
        let loss_scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        self.window
            .open_for_device(&self.ctx, self.grads.len(), self.scaler.is_some())?;
        self.forwarded = false;
        self.fresh_grads = false;
        if self.graph.is_some() {
            self.accumulate_graph(input, d_temporal)?;
        } else {
            if self.has_gemm_work() {
                prepare_inference_arch_rung(&self.ctx)?;
            }
            self.presize_gemm_scratch()?;
            let manifest = GpuCtx::record_owner_eager_gemm_manifest(
                self,
                |t| &t.ctx,
                |t| {
                    t.forward_unchecked(input, None)?;
                    t.upload_d_out(d_temporal, loss_scale)?;
                    t.backward_device()?;
                    t.accumulate_device()
                },
            )?;
            self.eager_gemm_manifest = Some(manifest);
        }
        Ok(BackwardMetrics {
            step: self.adamw.step,
            optimizer_stepped: false,
            grad_norm: None,
            loss_scale,
            overflow_skipped: None,
        })
    }

    /// Apply the open accumulation window with the step options (see
    /// [`Self::set_step_opts`]): in f16 the overflow check over the
    /// window's gradient, then the control clip, the global clip, the
    /// spike skip and AdamW, as an applying [`Self::backward_step`] with
    /// those options closes a window, with its bits. With captured graphs
    /// the call replays one.
    pub fn apply_window(&mut self) -> Result<StepMetrics, String> {
        if !self.window.is_open() {
            return Err("apply_window: no accumulation window is open".into());
        }
        let scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        self.ensure_adamw_plan()?;
        // The step counts before the update, as the fused step counts it;
        // a step the tail or an overflow skipped gives it back.
        let (_, bias_c1, bias_c2) = self.adamw.advance();
        self.bias
            .write(&self.ctx.stream, bias_c1, bias_c2, self.adamw.lr)?;
        let replayed = self.graph.is_some();
        if replayed {
            self.apply_graph()?;
        } else {
            self.apply_device()?;
        }
        self.sync("apply_window")?;
        self.window.close();
        self.fresh_grads = false;
        self.overflowed = scale.is_some() && self.overflow.read(&self.ctx.stream)? != 0;
        let (skipped, grad_norm) = if self.tail_measures() {
            match read_tail(
                &self.ctx,
                &self.skip_flag,
                self.window.clip_coef(),
                self.overflowed,
            ) {
                Ok((skipped, norm)) => (skipped, Some(norm)),
                Err(e) => {
                    self.adamw.step -= 1;
                    return Err(e);
                }
            }
        } else {
            (self.overflowed, None)
        };
        if let Some(scaler) = self.scaler.as_mut() {
            scaler.update(self.overflowed);
        }
        if skipped {
            self.adamw.step -= 1;
        }
        Ok(StepMetrics {
            step: self.adamw.step,
            graph_replayed: replayed,
            loss_scale: scale,
            overflow_skipped: scale.map(|_| self.overflowed),
            optimizer_stepped: !skipped,
            grad_norm,
        })
    }

    /// The arena of the last backward added to the open window's sum,
    /// which an f16 overflow marks: the accumulating part of a captured
    /// micro-batch.
    pub(super) fn accumulate_device(&mut self) -> Result<(), String> {
        let overflow = if self.scaler.is_some() {
            self.overflow.stable_ptr(&self.ctx.stream)
        } else {
            0
        };
        self.window.add_device(&self.ctx, &self.grads, overflow)
    }

    /// The open window's sum into the arena, in f16 the overflow check over
    /// it, the step options' tail and AdamW: the applying part of a
    /// captured window.
    pub(super) fn apply_device(&mut self) -> Result<(), String> {
        self.window.copy_sum_into(&self.ctx, &mut self.grads)?;
        if self.scaler.is_some() {
            let stream = self.ctx.stream.clone();
            self.overflow.zero(&stream)?;
            check_inf_nan_gpu(
                &self.ctx,
                &self.ctx.kernels,
                &mut self.overflow,
                &self.grads,
            )?;
        }
        let skip = self.tail_device()?;
        self.adamw_device(skip)
    }

    /// The backward of the last [`Self::forward`] for `d_temporal`
    /// (`[batch, seq_len, d_model]`), then the optimizer tail `opts` asks
    /// for: the control-channel clip, the global-norm clip, the skip of a
    /// window whose norm spikes, and the AdamW step. With
    /// `opts.accumulate_only` the gradient joins the open accumulation
    /// window instead and the optimizer waits; the next applying call
    /// steps on the window's sum. Scale `d_temporal` by
    /// `1 / micro_batches` for a mean.
    ///
    /// The control channels are the dd_dt, dd_A, trap and angle columns
    /// of every layer's in_proj and dt_bias, as in SISO. A clip of either
    /// kind belongs on the applying call: on an accumulating one it is an
    /// error. In f16 the loss scale
    /// holds for the whole window, and an overflow in any of its
    /// micro-batches skips the whole window and backs the scale off.
    pub fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        self.backward_step_with(d_temporal, opts, None)
    }

    /// [`Self::backward_step`] that also writes d loss / d the window's
    /// input into `d_input` (`[batch, seq_len, input_dim]`), for a model
    /// that trains something upstream of this backbone. In f16 the
    /// gradient comes back unscaled.
    pub fn backward_step_input_grad(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: &mut [f32],
    ) -> Result<BackwardMetrics, String> {
        self.backward_step_with(d_temporal, opts, Some(d_input))
    }

    fn backward_step_with(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: Option<&mut [f32]>,
    ) -> Result<BackwardMetrics, String> {
        check_backward_opts(&opts, "MIMO trainer")?;
        let loss_scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        self.backward(d_temporal, d_input)?;
        if opts.accumulate_only {
            self.window
                .accumulate(&self.ctx, &self.grads, self.overflowed)?;
            self.fresh_grads = false;
            return Ok(BackwardMetrics {
                step: self.adamw.step,
                optimizer_stepped: false,
                grad_norm: None,
                loss_scale,
                overflow_skipped: None,
            });
        }
        let window_overflowed = match self.window.fold_into(&self.ctx, &mut self.grads)? {
            FoldOverflow::Clear => false,
            FoldOverflow::Overflowed => true,
            // A micro-batch the device added marks its overflow in the
            // window's values alone, which the fold carried into the arena.
            FoldOverflow::InValues => !self.overflowed && self.grads_overflow()?,
        };
        self.overflowed |= window_overflowed;
        self.apply_tail(
            (
                opts.clip_max_norm,
                opts.step_skip_above,
                opts.control_clip_max_norm,
            ),
            loss_scale,
        )
    }

    /// The gradient of the open accumulation window: the sum of its
    /// micro-batches so far, in the layout of the master weights. A
    /// data-parallel caller reduces it across ranks between the last
    /// accumulating call and [`Self::apply_step`]. An error when no window
    /// is open.
    pub fn grad_arena(&self) -> Result<&GpuBuffer, String> {
        self.window.sum()
    }

    /// The open window's gradient downloaded to the host.
    pub fn download_grad_arena(&self) -> Result<Vec<f32>, String> {
        self.window.sum()?.to_cpu(&self.ctx.stream)
    }

    /// Overwrite the open window's gradient with `values`, a reduced arena
    /// of the same length.
    pub fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String> {
        let sum = self.window.sum_mut()?;
        if values.len() != sum.len() {
            return Err(format!(
                "upload_grad_arena: {} values for an arena of {}",
                values.len(),
                sum.len()
            ));
        }
        sum.upload(&self.ctx.stream, values)
    }

    /// Run only the optimizer tail, with the global clip `clip_max_norm`,
    /// over the open window's gradient (see [`Self::apply_step_full`]).
    pub fn apply_step(&mut self, clip_max_norm: Option<f32>) -> Result<BackwardMetrics, String> {
        self.apply_step_full(clip_max_norm, None, None)
    }

    /// Set the options of the fused [`Self::step`]: the control-channel
    /// clip, the global clip and the spike skip, which the eager step and a
    /// captured graph both apply. Accumulation belongs to `backward_step`
    /// and is refused here; a captured graph baked the options in, so
    /// other options need `drop_graph` first.
    pub fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
        if opts.accumulate_only {
            return Err("the fused step applies its window: accumulate with backward_step".into());
        }
        if self.graph.is_some() && opts != self.step_opts {
            return Err(
                "MIMO set_step_opts: the captured graph baked the step options in; \
                 drop_graph first"
                    .into(),
            );
        }
        self.step_opts = opts;
        Ok(())
    }

    /// The options of the fused [`Self::step`].
    pub fn step_opts(&self) -> BackwardOpts {
        self.step_opts
    }

    /// Whether the step options' tail measures the norm.
    pub(super) fn tail_measures(&self) -> bool {
        self.step_opts.clip_max_norm.is_some() || self.step_opts.step_skip_above.is_some()
    }

    /// The step options' tail on the gradient as launches only (see
    /// [`StepTail`]), returning the flag the AdamW after it skips on.
    pub(super) fn tail_device(&mut self) -> Result<u64, String> {
        let tail = StepTail::of(
            &self.step_opts,
            control_region_geom(&self.layout, &self.cfg),
        );
        let stream = self.ctx.stream.clone();
        let overflow = if self.scaler.is_some() {
            self.overflow.stable_ptr(&stream)
        } else {
            0
        };
        let skip = self.skip_flag.stable_ptr(&stream);
        let scratch = self.window.clip_scratch();
        tail.enqueue(&self.ctx, &mut self.grads, scratch, (skip, overflow))
    }

    /// Run only the optimizer tail over the open window's gradient: the
    /// control-channel clip, the global clip, the skip of a spiking window
    /// and the AdamW step, then close the window. With accumulating
    /// `backward_step` calls before it, a reducer can run between the
    /// last backward and the update, and the pair gives the bits of an
    /// applying `backward_step`. In f16 the window's values decide the
    /// overflow, so every rank that holds the same reduced gradient takes
    /// the same skip.
    pub fn apply_step_full(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        let loss_scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        self.overflowed = self.window.take_into(&self.ctx, &mut self.grads)?;
        if self.scaler.is_some() {
            self.overflowed |= self.grads_overflow()?;
        }
        self.fresh_grads = true;
        self.apply_tail(
            (clip_max_norm, step_skip_above, control_clip_max_norm),
            loss_scale,
        )
    }

    /// Data-parallel applying backward: the local backward joins the
    /// window, the window's gradient is summed across the ranks and scaled
    /// to their mean, then the optimizer tail runs. An accumulating call,
    /// or a world of one, is a plain [`Self::backward_step`]. In f16 the
    /// ranks reduce gradients of one loss scale, so an overflow on any
    /// rank reaches every rank's window as inf or NaN, and every rank
    /// skips the same window and keeps the same scale.
    pub fn backward_step_dist(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        dist: &crate::dist::DistContext,
    ) -> Result<BackwardMetrics, String> {
        let world = dist.world_size();
        if world == 1 || opts.accumulate_only {
            return self.backward_step(d_temporal, opts);
        }
        check_backward_opts(&opts, "MIMO trainer")?;
        self.backward_step(
            d_temporal,
            BackwardOpts::default().with_accumulate_only(true),
        )?;
        let stream = self.ctx.stream.clone();
        dist.all_reduce_grad_sum(self.window.sum()?, &stream)?;
        // The mean is the sum times 1 / world: exact for a power-of-two
        // world, one rounding per element otherwise.
        scale_grads(&self.ctx, self.window.sum_mut()?, 1.0 / world as f32)?;
        self.apply_step_full(
            opts.clip_max_norm,
            opts.step_skip_above,
            opts.control_clip_max_norm,
        )
    }

    /// Whether the gradient arena holds an inf or a NaN.
    fn grads_overflow(&mut self) -> Result<bool, String> {
        let stream = self.ctx.stream.clone();
        self.overflow.zero(&stream)?;
        check_inf_nan_gpu(
            &self.ctx,
            &self.ctx.kernels,
            &mut self.overflow,
            &self.grads,
        )?;
        Ok(self.overflow.read(&stream)? != 0)
    }

    /// The optimizer tail over the window's whole gradient in `grads`:
    /// `(clip_max_norm, step_skip_above, control_clip_max_norm)` ask for
    /// the control-channel clip, the global clip and the skip of a
    /// spiking window, then the AdamW step, which an f16 overflow skips.
    fn apply_tail(
        &mut self,
        (clip_max_norm, step_skip_above, control_clip_max_norm): (
            Option<f32>,
            Option<f32>,
            Option<f32>,
        ),
        loss_scale: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        // The control channels first, so a window whose step-size and
        // decay gradients grow with its length cannot rescale the whole
        // arena through the global clip below.
        if let Some(c) = control_clip_max_norm
            && !self.overflowed
        {
            let geom = control_region_geom(&self.layout, &self.cfg);
            self.window
                .clip_region(&self.ctx, &mut self.grads, geom, c)?;
        }
        let grad_norm = match norm_limit(clip_max_norm, step_skip_above) {
            Some(c) if !self.overflowed => Some(self.window.clip(&self.ctx, &mut self.grads, c)?),
            _ => None,
        };
        // A window whose norm before the clip passes the threshold is
        // dropped whole: no AdamW step, and the loss scale counts it as a
        // clean step.
        if let (Some(norm), Some(limit)) = (grad_norm, step_skip_above)
            && norm > limit
        {
            self.fresh_grads = false;
            if let Some(scaler) = self.scaler.as_mut() {
                scaler.update(false);
            }
            return Ok(BackwardMetrics {
                step: self.adamw.step,
                optimizer_stepped: false,
                grad_norm,
                loss_scale,
                overflow_skipped: loss_scale.map(|_| false),
            });
        }
        let overflowed = self.overflowed;
        let stepped = self.apply_adamw()?;
        Ok(BackwardMetrics {
            step: self.adamw.step,
            optimizer_stepped: stepped,
            grad_norm,
            loss_scale,
            overflow_skipped: loss_scale.map(|_| overflowed),
        })
    }

    /// The f16 loss scale and its growth counter; `None` in the other lanes.
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        self.scaler.as_ref().map(DynamicLossScaler::state)
    }

    /// Restore the f16 loss scale and its growth counter (a checkpoint
    /// resume, or a scale chosen up front); no effect in the other lanes.
    /// A scale that is not finite and positive is an error in every lane.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
        DynamicLossScaler::check_scale(scale)?;
        if let Some(scaler) = self.scaler.as_mut() {
            scaler.load_state(scale, growth_tracker)?;
        }
        Ok(())
    }

    /// Whether the last backward's gradients overflowed the f16 loss scale
    /// (its AdamW step will be skipped).
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Download the full optimizer state (the Adam moments, the step
    /// counter and the update settings) for a bit-continuous resume. The
    /// learning rate belongs to the caller's schedule and is not part of
    /// it.
    pub fn optimizer_state(&self) -> Result<AdamWStateBlob, String> {
        self.adamw.export_state(&self.ctx.stream)
    }

    /// Upload a previously exported optimizer state, adopting its step
    /// counter and update settings; a blob of another parameterization is
    /// an error. A captured graph refuses its replay when the settings it
    /// baked in changed (see [`Self::capture_graph`]).
    pub fn load_optimizer_state(&mut self, blob: &AdamWStateBlob) -> Result<(), String> {
        self.adamw.import_state(&self.ctx.stream, blob)
    }

    /// Download the carried recurrence: every layer's SSM state, the K and
    /// V of the last step and the accumulated angles, which a resume
    /// uploads to continue the sequences.
    pub fn recurrent_state(&self) -> Result<Mamba3MimoRecurrentStateBlob, String> {
        let s = &self.ctx.stream;
        Ok(Mamba3MimoRecurrentStateBlob {
            ssm: self.state_ssm.to_cpu(s)?,
            k: self.state_k.to_cpu(s)?,
            v: self.state_v.to_cpu(s)?,
            angle: self.state_ang.to_cpu(s)?,
        })
    }

    /// Upload a previously exported recurrence; a blob of another shape is
    /// an error.
    pub fn load_recurrent_state(
        &mut self,
        blob: &Mamba3MimoRecurrentStateBlob,
    ) -> Result<(), String> {
        self.load_recurrent_parts([&blob.ssm, &blob.k, &blob.v, &blob.angle])
    }

    /// The (ssm, k, v, angle) recurrence from borrowed values.
    fn load_recurrent_parts(&mut self, [ssm, k, v, angle]: [&[f32]; 4]) -> Result<(), String> {
        let lens = [
            (ssm.len(), self.state_ssm.len()),
            (k.len(), self.state_k.len()),
            (v.len(), self.state_v.len()),
            (angle.len(), self.state_ang.len()),
        ];
        if lens.iter().any(|(b, t)| b != t) {
            return Err(format!(
                "recurrent state mismatch: the blob's (ssm, k, v, angle) lengths against \
                 this trainer's are {lens:?}"
            ));
        }
        let s = self.ctx.stream.clone();
        self.state_ssm.upload(&s, ssm)?;
        self.state_k.upload(&s, k)?;
        self.state_v.upload(&s, v)?;
        self.state_ang.upload(&s, angle)
    }

    /// Everything this trainer needs beside its weights to resume bit for
    /// bit: the optimizer state, the f16 loss scale and the carried
    /// recurrence. [`TrainerState::save`] writes it to a file.
    pub fn trainer_state(&self) -> Result<TrainerState, String> {
        let rec = self.recurrent_state()?;
        Ok(TrainerState {
            family: ModelFamily::Mamba3Mimo,
            optimizer: self.optimizer_state()?,
            scaler: self.scaler_state(),
            recurrent: vec![
                ("ssm".into(), rec.ssm),
                ("k".into(), rec.k),
                ("v".into(), rec.v),
                ("angle".into(), rec.angle),
            ],
        })
    }

    /// Load a state a trainer of this family gave with
    /// [`Self::trainer_state`]. The learning rate is not part of it.
    pub fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String> {
        state.check_family(ModelFamily::Mamba3Mimo)?;
        let parts = [
            state.recurrent_tensor("ssm")?,
            state.recurrent_tensor("k")?,
            state.recurrent_tensor("v")?,
            state.recurrent_tensor("angle")?,
        ];
        self.load_optimizer_state(&state.optimizer)?;
        if let Some((scale, tracker)) = state.scaler {
            self.load_scaler_state(scale, tracker)?;
        }
        self.load_recurrent_parts(parts)
    }

    /// Download the f32 master weights for a checkpoint or the CPU model.
    pub fn snapshot_master(&self) -> Result<Mamba3MimoWeights, String> {
        let flat = self.master.to_cpu(&self.ctx.stream)?;
        Ok(self.layout.unflatten(&flat, &self.template))
    }

    /// The gradients of the last backward, shaped like the weights.
    pub fn grads(&self) -> Result<Mamba3MimoWeights, String> {
        let flat = self.grads.to_cpu(&self.ctx.stream)?;
        // Fields the trainer does not train come out zero, not as weights.
        Ok(self.layout.unflatten(&flat, &self.template.zeros_like()))
    }

    /// Back to the empty states.
    pub fn reset_state(&mut self) -> Result<(), String> {
        let s = &self.ctx.stream;
        self.state_ssm.zero(s)?;
        self.state_k.zero(s)?;
        self.state_v.zero(s)?;
        self.state_ang.zero(s)
    }

    fn refresh_mirror(&self) -> Result<(), String> {
        let Some(mirror) = &self.mirror else {
            return Ok(());
        };
        let cast = match self.act {
            WeightDtype::Bf16 => &self.ctx.kernels.cast_f32_to_bf16,
            _ => &self.ctx.kernels.cast_f32_to_f16,
        };
        let (base, copy) = (self.master.cached_ptr(), mirror.cached_ptr());
        for &(src, dst) in &self.mirror_pairs {
            launch_cast(
                &self.ctx.stream,
                cast,
                (copy + (dst.off * 2) as u64, 2),
                (at(base, src), 4),
                src.len,
                format_args!("MIMO half weight copy"),
            )?;
        }
        Ok(())
    }

    /// A matrix the GEMMs read: its half copy, or the master itself in the
    /// f32 lanes.
    /// The matrix a GEMM reads: the f32 master, or its half copy in the
    /// half lanes. A span with no half copy is an error, never another
    /// matrix's copy.
    fn gemm_weight(&self, span: Span) -> Result<u64, String> {
        match &self.mirror {
            None => Ok(at(self.master.cached_ptr(), span)),
            Some(m) => {
                let dst = self
                    .mirror_pairs
                    .iter()
                    .find(|(src, _)| src.off == span.off)
                    .map(|(_, dst)| dst.off)
                    .ok_or("a GEMM matrix has no half copy")?;
                Ok(m.cached_ptr() + (dst * 2) as u64)
            }
        }
    }

    fn copy(&self, dst: u64, src: u64, n: usize, what: &str) -> Result<(), String> {
        let r = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, n * 4, self.ctx.stream.cu_stream())
        };
        if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("MIMO trainer copy {what}: {r:?}"))
        }
    }

    fn sync(&self, what: &str) -> Result<(), String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("MIMO trainer {what} sync: {e:?}"))
    }

    fn typed(&self, ptr: u64) -> TypedPtr {
        TypedPtr {
            ptr,
            dtype: self.act,
        }
    }

    /// A block norm of f32 rows into `y` (the activation dtype when
    /// `typed`); `stats` takes each row's statistics.
    fn block_norm(
        &self,
        (y, x, stats): (u64, u64, u64),
        norm: NormParams,
        (rows, typed): (usize, bool),
    ) -> Result<(), String> {
        let out = if typed { self.act } else { WeightDtype::F32 };
        let shape = (rows, self.cfg.d_model, self.cfg.rms_norm_eps);
        block_norm::forward(
            &self.ctx,
            out,
            (y, stats, x),
            norm,
            shape,
            "MIMO block norm",
        )
    }

    /// A block norm's parameters in the buffer at `base` laid out as the
    /// master: its scale, and the bias for LayerNorm (0 for RMSNorm).
    fn norm_params(base: u64, (w, b): (Span, Span)) -> NormParams {
        NormParams {
            w: at(base, w),
            b: if b.len > 0 { at(base, b) } else { 0 },
        }
    }

    /// `temporal += out` for `n` values, `out` in the activation dtype.
    fn residual_add(&self, temporal: u64, out: u64, n: usize) -> Result<(), String> {
        let k = &self.ctx.kernels;
        let (f, ptrs): (_, &[u64]) = if self.act == WeightDtype::F32 {
            (&k.vec_add_inplace, &[temporal, out])
        } else {
            (
                k.residual_add_f32_typed.get(self.act),
                &[temporal, temporal, out],
            )
        };
        launch_elems(&self.ctx.stream, f, ptrs, n, format_args!("MIMO residual"))
    }

    /// `y = x W` in the activation dtype.
    fn gemm(&self, (y, x, w): (u64, u64, u64), dims: (usize, usize, usize)) -> Result<(), String> {
        gpu_gemm_typed_forward_raw(
            &self.ctx,
            self.typed(y),
            self.typed(x),
            self.typed(w),
            None,
            dims,
        )
    }

    /// The training forward over one window: `input` is `[batch, seq_len,
    /// input_dim]`, `out` (`[batch, seq_len, d_model]`) receives every
    /// position's final-norm output. The states carry from the last call.
    pub fn forward(&mut self, input: &[f32], out: &mut [f32]) -> Result<(), String> {
        self.check_window_len("forward input", input.len(), self.input_dim)?;
        self.check_window_len("forward output", out.len(), self.cfg.d_model)?;
        self.forwarded = false;
        self.presize_gemm_scratch()?;
        self.forward_unchecked(input, Some(out))?;
        self.forwarded = true;
        Ok(())
    }

    /// Size the context's GEMM staging and upcast scratch for this
    /// trainer's widest GEMM before a window's first one. Every weight a
    /// GEMM reads is `d_model` wide on one side, so its length over
    /// `d_model` is the activation width on the other.
    fn presize_gemm_scratch(&self) -> Result<(), String> {
        let dm = self.cfg.d_model;
        let spans = std::iter::once(self.layout.input_proj_w)
            .chain(
                self.layout
                    .layers
                    .iter()
                    .flat_map(|l| [l.in_proj_w, l.out_proj_w]),
            )
            .chain(self.layout.mlp.iter().flat_map(|m| [m.fc1_w, m.fc2_w]));
        let (mut width, mut weight) = (dm, 0);
        for s in spans {
            width = width.max(s.len / dm.max(1));
            weight = weight.max(s.len);
        }
        self.ctx
            .presize_gemm_scratch(self.batch * self.seq_len, width, weight, self.dtype)
    }

    /// The forward kernels; `out`, when given, receives the final rows.
    fn forward_unchecked(&mut self, input: &[f32], out: Option<&mut [f32]>) -> Result<(), String> {
        let stream = self.ctx.stream.clone();
        self.input.upload(&stream, input)?;
        self.forward_device()?;
        let Some(out) = out else {
            return Ok(());
        };
        self.sync("forward")?;
        self.final_out.download(&stream, out)?;
        self.sync("forward")
    }

    /// The forward's kernels on the uploaded input, the final rows left in
    /// `final_out`: the forward part of a captured step.
    fn forward_device(&self) -> Result<(), String> {
        let c = self.cfg;
        let (b, t_len) = (self.batch, self.seq_len);
        let rows = b * t_len;
        let (dm, ip, di, bc) = (c.d_model, c.in_proj_out_dim(), c.d_inner(), c.bc_width());
        let (nh, hd, ds, rank, ng, na) = (
            c.nheads(),
            c.headdim,
            c.d_state,
            c.mimo_rank,
            c.ngroups,
            c.num_rope_angles(),
        );
        let stream = &self.ctx.stream;
        let base = self.master.cached_ptr();
        if self.layout.input_proj_w.len == 0 {
            self.copy(
                self.temporal.cached_ptr(),
                self.input.cached_ptr(),
                rows * dm,
                "input rows",
            )?;
        } else {
            let bias =
                (self.layout.input_proj_b.len > 0).then(|| at(base, self.layout.input_proj_b));
            unsafe {
                gpu_gemm_f32_forward_ptrs(
                    &self.ctx,
                    self.temporal.cached_ptr(),
                    self.input.cached_ptr(),
                    at(base, self.layout.input_proj_w),
                    bias,
                    (rows, self.input_dim, dm),
                )
            }?;
        }
        let temporal = self.temporal.cached_ptr();
        let per = |n: usize| (b * nh * n * 4) as u64;
        let ld_proj = ip as i64;
        let (b_col, c_col) = ((2 * di) as i32, (2 * di + bc) as i32);
        let rg = (rank * ng) as i32;
        let (t_i, nh_i, ng_i, hd_i, ds_i, r_i, na_i) = (
            t_len as i32,
            nh as i32,
            ng as i32,
            hd as i32,
            ds as i32,
            rank as i32,
            na as i32,
        );
        let stride_i = self.stride as i32;
        let (eps, a_floor) = (c.ssm_norm_eps, c.a_floor);
        let out_rows = self.out_rows.ptr();
        for (l, a) in self.acts.iter().enumerate() {
            let ls = self.layout.layers[l];
            self.copy(a.x_in.cached_ptr(), temporal, rows * dm, "layer input")?;
            self.block_norm(
                (a.post.ptr(), temporal, a.rms.cached_ptr()),
                Self::norm_params(base, (ls.norm_weight, ls.norm_bias)),
                (rows, true),
            )?;
            let proj = a.proj.ptr();
            self.gemm(
                (proj, a.post.ptr(), self.gemm_weight(ls.in_proj_w)?),
                (rows, dm, ip),
            )?;
            let (bn, cn) = (a.bn.cached_ptr(), a.cn.cached_ptr());
            let (wb, wc) = (at(base, ls.b_norm_weight), at(base, ls.c_norm_weight));
            for (first, n) in row_pieces(rows, rank * ng) {
                let normed = (first * rank * ng * ds * 4) as u64;
                let (bn_p, cn_p) = (bn + normed, cn + normed);
                let proj_p = proj + (first * ip * self.act.size_bytes()) as u64;
                let rows_p = n as i64;
                let mut bld = stream.launch_builder(self.kernels.bc_norm.get(self.act));
                bld.arg(&bn_p);
                bld.arg(&cn_p);
                bld.arg(&proj_p);
                bld.arg(&ld_proj);
                bld.arg(&b_col);
                bld.arg(&c_col);
                bld.arg(&wb);
                bld.arg(&wc);
                bld.arg(&rows_p);
                bld.arg(&rg);
                bld.arg(&ds_i);
                bld.arg(&eps);
                let cfg = LaunchConfig {
                    grid_dim: ((n * rank * ng) as u32, 2, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { bld.launch(cfg) }.map_err(|e| format!("MIMO B/C norm L{l}: {e:?}"))?;
            }
            if let Some(geom) = self.chunked {
                let (kst, vst) = (
                    self.state_k.cached_ptr() + l as u64 * per(rank * ds),
                    self.state_v.cached_ptr() + l as u64 * per(hd),
                );
                self.copy(a.k0.cached_ptr(), kst, b * nh * rank * ds, "carried keys")?;
                self.copy(a.v0.cached_ptr(), vst, b * nh * hd, "carried input")?;
                let ptrs = MimoChunkedPtrs {
                    proj,
                    bn,
                    cn,
                    out: a.o.ptr(),
                    y_out: a.y.cached_ptr(),
                    coef: a.coef.cached_ptr(),
                    sn: a.sn.cached_ptr(),
                    cs: a.cs.cached_ptr(),
                    e0: a.e0.cached_ptr(),
                    contrib: self.contrib.cached_ptr(),
                    enter: a.enter.cached_ptr(),
                    ssm: self.state_ssm.cached_ptr() + l as u64 * per(hd * ds),
                    kst,
                    vst,
                    ang: self.state_ang.cached_ptr() + l as u64 * per(na),
                    dt_bias: at(base, ls.dt_bias),
                    b_bias: at(base, ls.b_bias),
                    c_bias: at(base, ls.c_bias),
                    d: at(base, ls.d_param),
                    mimo_x: at(base, ls.mimo_x),
                    mimo_z: at(base, ls.mimo_z),
                    mimo_o: at(base, ls.mimo_o),
                };
                let x = MimoExec {
                    ctx: &self.ctx,
                    kernels: &self.kernels,
                    cfg: &self.cfg,
                    act: self.act,
                };
                launch_chunked_scan(x, &ptrs, &geom, (b, t_len), l)?;
            } else {
                let o = a.o.ptr();
                let ssm = self.state_ssm.cached_ptr() + l as u64 * per(hd * ds);
                let kst = self.state_k.cached_ptr() + l as u64 * per(rank * ds);
                let vst = self.state_v.cached_ptr() + l as u64 * per(hd);
                let ang = self.state_ang.cached_ptr() + l as u64 * per(na);
                let ptrs = [
                    at(base, ls.dt_bias),
                    at(base, ls.b_bias),
                    at(base, ls.c_bias),
                    at(base, ls.d_param),
                    at(base, ls.mimo_x),
                    at(base, ls.mimo_z),
                    at(base, ls.mimo_o),
                ];
                let cks = [
                    a.ck_ssm.cached_ptr(),
                    a.ck_k.cached_ptr(),
                    a.ck_v.cached_ptr(),
                    a.ck_ang.cached_ptr(),
                ];
                for (y_first, z_first, cfg) in yz_runs(self.scan.cfg(b, nh, hd)) {
                    let mut bld = stream.launch_builder(self.kernels.scan_ckpt.get(self.act));
                    bld.arg(&o);
                    bld.arg(&ssm);
                    bld.arg(&kst);
                    bld.arg(&vst);
                    bld.arg(&ang);
                    bld.arg(&proj);
                    bld.arg(&ld_proj);
                    bld.arg(&bn);
                    bld.arg(&cn);
                    for p in &ptrs {
                        bld.arg(p);
                    }
                    bld.arg(&t_i);
                    bld.arg(&nh_i);
                    bld.arg(&ng_i);
                    bld.arg(&hd_i);
                    bld.arg(&ds_i);
                    bld.arg(&r_i);
                    bld.arg(&na_i);
                    bld.arg(&a_floor);
                    for p in &cks {
                        bld.arg(p);
                    }
                    bld.arg(&stride_i);
                    let (tile, pipe) = (self.scan.tile as i32, i32::from(self.scan.pipe));
                    let done = self.scan_done.cached_ptr();
                    let y_out = if c.is_outproj_norm {
                        a.y.cached_ptr()
                    } else {
                        0u64
                    };
                    bld.arg(&tile);
                    bld.arg(&pipe);
                    bld.arg(&done);
                    bld.arg(&y_out);
                    bld.arg(&y_first);
                    bld.arg(&z_first);
                    unsafe { bld.launch(cfg) }.map_err(|e| format!("MIMO scan L{l}: {e:?}"))?;
                }
            }
            // The output norm gates the scans' readouts again, over the plain
            // gate's rows.
            if c.is_outproj_norm {
                let (o, y, nw) = (a.o.ptr(), a.y.cached_ptr(), at(base, ls.norm_gate));
                let (mz, mo) = (at(base, ls.mimo_z), at(base, ls.mimo_o));
                let cfg = LaunchConfig {
                    grid_dim: (rows as u32, nh as u32, 1),
                    block_dim: (hd.next_multiple_of(32).min(256) as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                for (y_first, cfg) in y_runs(cfg) {
                    let mut bld = stream.launch_builder(self.kernels.gate_normed.get(self.act));
                    bld.arg(&o);
                    bld.arg(&y);
                    bld.arg(&proj);
                    bld.arg(&ld_proj);
                    bld.arg(&nw);
                    bld.arg(&mz);
                    bld.arg(&mo);
                    bld.arg(&nh_i);
                    bld.arg(&hd_i);
                    bld.arg(&r_i);
                    bld.arg(&eps);
                    bld.arg(&y_first);
                    unsafe { bld.launch(cfg) }
                        .map_err(|e| format!("MIMO output norm L{l}: {e:?}"))?;
                }
            }
            self.gemm(
                (out_rows, a.o.ptr(), self.gemm_weight(ls.out_proj_w)?),
                (rows, di, dm),
            )?;
            self.residual_add(temporal, out_rows, rows * dm)?;
            if let Some(m) = &a.mlp {
                let ms = self.layout.mlp[l];
                let hidden = ms.fc2_w.len / dm;
                self.copy(m.x_in.cached_ptr(), temporal, rows * dm, "MLP input")?;
                self.block_norm(
                    (m.post.ptr(), temporal, m.rms.cached_ptr()),
                    Self::norm_params(base, (ms.norm_weight, ms.norm_bias)),
                    (rows, true),
                )?;
                self.gemm(
                    (m.up.ptr(), m.post.ptr(), self.gemm_weight(ms.fc1_w)?),
                    (rows, dm, 2 * hidden),
                )?;
                let (actp, upp) = (m.act.ptr(), m.up.ptr());
                let (rows_i, h_i) = (rows as i32, hidden as i32);
                let mut bld = stream.launch_builder(self.kernels.mlp_gate.get(self.act));
                bld.arg(&actp);
                bld.arg(&upp);
                bld.arg(&rows_i);
                bld.arg(&h_i);
                unsafe { bld.launch(grid_1d_cover(rows * hidden)?) }
                    .map_err(|e| format!("MIMO MLP gate L{l}: {e:?}"))?;
                self.gemm(
                    (out_rows, actp, self.gemm_weight(ms.fc2_w)?),
                    (rows, hidden, dm),
                )?;
                self.residual_add(temporal, out_rows, rows * dm)?;
            }
        }
        self.copy(
            self.final_in.cached_ptr(),
            temporal,
            rows * dm,
            "final rows",
        )?;
        self.block_norm(
            (
                self.final_out.cached_ptr(),
                temporal,
                self.final_rms.cached_ptr(),
            ),
            Self::norm_params(base, (self.layout.norm_f_weight, self.layout.norm_f_bias)),
            (rows, false),
        )
    }

    fn grad(&self, span: Span) -> GradSlice {
        GradSlice::from_raw(at(self.grads.cached_ptr(), span), span.len)
    }

    /// `out[j] += sum_r parts[r][j]`, a fixed-tree reduction over the rows.
    fn reduce_axis0(&self, out: u64, parts: u64, dims: (usize, usize)) -> Result<(), String> {
        self.reduce_rows(out, parts, dims, true)
    }

    /// `out[j] = sum_r parts[r][j]`, a fixed-tree reduction over the rows,
    /// over whatever `out` held.
    fn reduce_axis0_set(&self, out: u64, parts: u64, dims: (usize, usize)) -> Result<(), String> {
        self.reduce_rows(out, parts, dims, false)
    }

    /// The MIMO module's row reduction: the shared reduce_sum_axis0's
    /// order (so its bits) with 64-bit indices, since a training window's
    /// per-row partials can pass 2^31 elements. A block takes 32
    /// neighbouring columns when there are enough of them to fill the GPU,
    /// else 8, so narrow reductions still spread over blocks.
    fn reduce_rows(
        &self,
        out: u64,
        parts: u64,
        (rows, dim): (usize, usize),
        accumulate: bool,
    ) -> Result<(), String> {
        let width = rows.next_power_of_two().clamp(32, 256) as u32;
        let cw: u32 = if dim >= 4096 { 32 } else { 8 };
        let (rows_i, dim_i, acc) = (rows as i64, dim as i64, i32::from(accumulate));
        let (width_i, cw_i) = (width as i32, cw as i32);
        let mut bld = self.ctx.stream.launch_builder(&self.kernels.cb.reduce_rows);
        bld.arg(&out);
        bld.arg(&parts);
        bld.arg(&rows_i);
        bld.arg(&dim_i);
        bld.arg(&width_i);
        bld.arg(&cw_i);
        bld.arg(&acc);
        let cfg = LaunchConfig {
            grid_dim: ((dim as u32).div_ceil(cw), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: width * cw * 4,
        };
        unsafe { bld.launch(cfg) }
            .map(|_| ())
            .map_err(|e| format!("MIMO column reduction: {e:?}"))
    }

    /// A block norm's backward over `rows` rows into `dx` (f32): the
    /// gradients of the norm the `spans` (scale, bias) name accumulate in
    /// the arena; `dy` is in the activation dtype when `typed_dy`.
    fn norm_backward(
        &self,
        (dx, dy, x, stats): (u64, u64, u64, u64),
        (spans, typed_dy): ((Span, Span), bool),
        rows: usize,
    ) -> Result<(), String> {
        let dm = self.cfg.d_model;
        let axis0 = self.bwd.axis0.cached_ptr();
        let norm = Self::norm_params(self.master.cached_ptr(), spans);
        let g = Self::norm_params(self.grads.cached_ptr(), spans);
        let dy_dtype = if typed_dy { self.act } else { WeightDtype::F32 };
        block_norm::backward(
            &self.ctx,
            dy_dtype,
            (dx, axis0, dy, x, stats),
            (norm, g.b),
            (rows, dm),
            (false, 0),
            "MIMO block norm backward",
        )?;
        self.reduce_axis0(g.w, axis0, (rows, dm))
    }

    /// `d_temp += d_pre`.
    fn add_residual_grad(&self, rows: usize) -> Result<(), String> {
        launch_elems(
            &self.ctx.stream,
            &self.ctx.kernels.vec_add_inplace,
            &[self.bwd.d_temp.cached_ptr(), self.bwd.d_pre.cached_ptr()],
            rows * self.cfg.d_model,
            format_args!("MIMO backward residual"),
        )
    }

    /// The residual gradient in the half type (the half lanes' GEMM input).
    fn round_d_temp(&self, rows: usize) -> Result<u64, String> {
        let dth = self
            .bwd
            .d_temp_h
            .as_ref()
            .ok_or("half lane without its rounded residual gradient")?
            .cached_ptr();
        let cast = match self.act {
            WeightDtype::Bf16 => &self.ctx.kernels.cast_f32_to_bf16,
            _ => &self.ctx.kernels.cast_f32_to_f16,
        };
        launch_cast(
            &self.ctx.stream,
            cast,
            (dth, 2),
            (self.bwd.d_temp.cached_ptr(), 4),
            rows * self.cfg.d_model,
            format_args!("MIMO rounding rows"),
        )?;
        Ok(dth)
    }

    /// The backward of `y = x W` with no bias in the half lanes: dW
    /// accumulates from the typed rows, dx (typed) is written.
    fn half_linear_backward(
        &self,
        (dx, dy, x): (u64, u64, u64),
        (w, gw): (u64, &GradSlice),
        (rows, n_in, n_out): (usize, usize, usize),
    ) -> Result<(), String> {
        gpu_gemm_bi_backward_dw_grad_typed(
            &self.ctx,
            gw,
            self.typed(dy),
            self.typed(x),
            rows,
            n_in,
            n_out,
        )?;
        gpu_gemm_ex_backward_dx_typed(
            &self.ctx,
            self.typed(dx),
            self.typed(dy),
            self.typed(w),
            rows,
            n_in,
            n_out,
        )
    }

    /// The backward of the last [`Self::forward`]: `d_out` is d loss / d
    /// (its outputs), `[batch, seq_len, d_model]`. The gradients are
    /// overwritten (see [`Self::grads`]); `d_input`, when given, receives d
    /// loss / d (the inputs), `[batch, seq_len, input_dim]`.
    /// [`Self::backward_step`] is this backward with the optimizer step;
    /// this split with [`Self::apply_adamw`] is for a caller that trains
    /// something upstream on `d_input`.
    ///
    /// A window the forward did not complete is refused. After an f16
    /// overflow (see [`Self::overflowed`]) the gradients and `d_input` are
    /// not finite: the next [`Self::apply_adamw`] skips its step, and a
    /// caller training anything upstream must skip that step too.
    pub fn backward(&mut self, d_out: &[f32], d_input: Option<&mut [f32]>) -> Result<(), String> {
        if !self.forwarded {
            return Err("backward needs a completed forward of this window".into());
        }
        self.check_window_len("backward d_out", d_out.len(), self.cfg.d_model)?;
        if let Some(d) = d_input.as_deref() {
            self.check_window_len("backward d_input", d.len(), self.input_dim)?;
        }
        self.fresh_grads = false;
        self.backward_unchecked(d_out, d_input)?;
        self.fresh_grads = true;
        Ok(())
    }

    fn backward_unchecked(
        &mut self,
        d_out: &[f32],
        d_input: Option<&mut [f32]>,
    ) -> Result<(), String> {
        let stream = self.ctx.stream.clone();
        let scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        self.upload_d_out(d_out, scale)?;
        self.backward_device()?;
        self.overflowed = false;
        if scale.is_some() {
            self.sync("backward")?;
            self.overflowed = self.overflow.read(&stream)? != 0;
        }
        self.sync("backward")?;
        if let Some(d_input) = d_input {
            let src = if self.layout.input_proj_w.len == 0 {
                &self.bwd.d_temp
            } else {
                &self.bwd.d_input
            };
            src.download(&stream, d_input)?;
            self.sync("backward")?;
            if let Some(sc) = scale {
                let inv = 1.0 / sc;
                for v in d_input.iter_mut() {
                    *v *= inv;
                }
            }
        }
        Ok(())
    }

    /// d loss / d the final rows into the backward's input, and in f16 the
    /// loss scale the backward multiplies them by and the factor it
    /// unscales the gradients by.
    fn upload_d_out(&mut self, d_out: &[f32], scale: Option<f32>) -> Result<(), String> {
        let stream = self.ctx.stream.clone();
        self.bwd.d_rows.upload(&stream, d_out)?;
        if let Some(sc) = scale {
            self.loss_scale.upload(&stream, &[sc])?;
            self.unscale.write(&stream, 1.0 / sc)?;
        }
        Ok(())
    }

    /// The backward's kernels on the uploaded d_rows: the arena zeroed,
    /// every tensor's gradient, and in f16 the overflow check and the
    /// unscale, which leaves an overflowed arena as it is. The backward
    /// part of a captured step.
    fn backward_device(&mut self) -> Result<(), String> {
        let c = self.cfg;
        let rows = self.batch * self.seq_len;
        let dm = c.d_model;
        let stream = self.ctx.stream.clone();
        self.grads.zero(&stream)?;
        if self.scaler.is_some() {
            let (d_rows, scale) = (self.bwd.d_rows.cached_ptr(), self.loss_scale.cached_ptr());
            let n = self.bwd.d_rows.len() as i64;
            let mut bld = stream.launch_builder(&self.ctx.kernels.scale_grads_dev_f32);
            bld.arg(&d_rows);
            bld.arg(&scale);
            bld.arg(&n);
            unsafe { bld.launch(crate::gpu::launch::grid_1d_wide(self.bwd.d_rows.len())) }
                .map_err(|e| format!("loss scale of d_rows: {e:?}"))?;
        }
        let base = self.master.cached_ptr();
        self.norm_backward(
            (
                self.bwd.d_temp.cached_ptr(),
                self.bwd.d_rows.cached_ptr(),
                self.final_in.cached_ptr(),
                self.final_rms.cached_ptr(),
            ),
            ((self.layout.norm_f_weight, self.layout.norm_f_bias), false),
            rows,
        )?;
        for l in (0..c.n_layers).rev() {
            if self.acts[l].mlp.is_some() {
                self.mlp_backward(l)?;
            }
            self.layer_backward(l)?;
        }
        let identity_input = self.layout.input_proj_w.len == 0;
        if !identity_input {
            let gw = self.grad(self.layout.input_proj_w);
            let gb =
                (self.layout.input_proj_b.len > 0).then(|| self.grad(self.layout.input_proj_b));
            let w_ptr = at(base, self.layout.input_proj_w);
            gpu_gemm_bi_backward_grad_raw(
                &self.ctx,
                &mut self.bwd.d_input,
                (&gw, gb.as_ref()),
                &self.bwd.d_temp,
                &self.input,
                w_ptr,
                (rows, self.input_dim, dm),
            )?;
        }
        if self.scaler.is_some() {
            self.overflow.zero(&stream)?;
            check_inf_nan_gpu(
                &self.ctx,
                &self.ctx.kernels,
                &mut self.overflow,
                &self.grads,
            )?;
            scale_grads_skip_gpu(
                &self.ctx,
                &self.ctx.kernels,
                &mut self.overflow,
                &mut self.grads,
                &self.unscale,
            )?;
        }
        Ok(())
    }

    fn mlp_backward(&mut self, l: usize) -> Result<(), String> {
        let c = self.cfg;
        let rows = self.batch * self.seq_len;
        let dm = c.d_model;
        let ms = self.layout.mlp[l];
        let hidden = ms.fc2_w.len / dm;
        let gw2 = self.grad(ms.fc2_w);
        let w2 = self.gemm_weight(ms.fc2_w)?;
        if self.act == WeightDtype::F32 {
            let m = self.acts[l].mlp.as_ref().ok_or("the layer has no MLP")?;
            let bwd = &mut self.bwd;
            gpu_gemm_bi_backward_grad_raw(
                &self.ctx,
                bwd.d_act.f32_mut()?,
                (&gw2, None),
                &bwd.d_temp,
                m.act.f32()?,
                w2,
                (rows, hidden, dm),
            )?;
        } else {
            let dth = self.round_d_temp(rows)?;
            let act_rows = self.acts[l].mlp.as_ref().map_or(0, |m| m.act.ptr());
            self.half_linear_backward(
                (self.bwd.d_act.ptr(), dth, act_rows),
                (w2, &gw2),
                (rows, hidden, dm),
            )?;
        }
        {
            let (d_up, d_act) = (self.bwd.d_up.ptr(), self.bwd.d_act.ptr());
            let up = self.acts[l].mlp.as_ref().map_or(0, |m| m.up.ptr());
            let (rows_i, h_i) = (rows as i32, hidden as i32);
            let mut bld = self
                .ctx
                .stream
                .launch_builder(self.kernels.mlp_gate_bwd.get(self.act));
            bld.arg(&d_up);
            bld.arg(&d_act);
            bld.arg(&up);
            bld.arg(&rows_i);
            bld.arg(&h_i);
            unsafe { bld.launch(grid_1d_cover(rows * hidden)?) }
                .map_err(|e| format!("MIMO MLP gate backward L{l}: {e:?}"))?;
        }
        let gw1 = self.grad(ms.fc1_w);
        let w1 = self.gemm_weight(ms.fc1_w)?;
        if self.act == WeightDtype::F32 {
            let m = self.acts[l].mlp.as_ref().ok_or("the layer has no MLP")?;
            let bwd = &mut self.bwd;
            gpu_gemm_bi_backward_grad_raw(
                &self.ctx,
                bwd.d_post2.f32_mut()?,
                (&gw1, None),
                bwd.d_up.f32()?,
                m.post.f32()?,
                w1,
                (rows, dm, 2 * hidden),
            )?;
        } else {
            let post2 = self.acts[l].mlp.as_ref().map_or(0, |m| m.post.ptr());
            self.half_linear_backward(
                (self.bwd.d_post2.ptr(), self.bwd.d_up.ptr(), post2),
                (w1, &gw1),
                (rows, dm, 2 * hidden),
            )?;
        }
        let m = self.acts[l].mlp.as_ref().ok_or("the layer has no MLP")?;
        self.norm_backward(
            (
                self.bwd.d_pre.cached_ptr(),
                self.bwd.d_post2.ptr(),
                m.x_in.cached_ptr(),
                m.rms.cached_ptr(),
            ),
            ((ms.norm_weight, ms.norm_bias), true),
            rows,
        )?;
        self.add_residual_grad(rows)
    }

    fn layer_backward(&mut self, l: usize) -> Result<(), String> {
        let c = self.cfg;
        let (b, t_len) = (self.batch, self.seq_len);
        let rows = b * t_len;
        let (dm, ip, di, bc) = (c.d_model, c.in_proj_out_dim(), c.d_inner(), c.bc_width());
        let (nh, hd, ds, rank, ng, na) = (
            c.nheads(),
            c.headdim,
            c.d_state,
            c.mimo_rank,
            c.ngroups,
            c.num_rope_angles(),
        );
        let ls = self.layout.layers[l];
        let base = self.master.cached_ptr();
        let g_base = self.grads.cached_ptr();
        let stream = self.ctx.stream.clone();
        let act = self.act;

        // out_proj.
        let gwo = self.grad(ls.out_proj_w);
        let wo = self.gemm_weight(ls.out_proj_w)?;
        if act == WeightDtype::F32 {
            let bwd = &mut self.bwd;
            gpu_gemm_bi_backward_grad_raw(
                &self.ctx,
                bwd.d_o.f32_mut()?,
                (&gwo, None),
                &bwd.d_temp,
                self.acts[l].o.f32()?,
                wo,
                (rows, di, dm),
            )?;
        } else {
            let dth = self.round_d_temp(rows)?;
            self.half_linear_backward(
                (self.bwd.d_o.ptr(), dth, self.acts[l].o.ptr()),
                (wo, &gwo),
                (rows, di, dm),
            )?;
        }

        let k = &self.kernels;
        let a = &self.acts[l];
        let s = &self.bwd;
        let (ld_proj, rows_l) = (ip as i64, rows as i64);
        let (t_i, nh_i, ng_i, hd_i, ds_i, r_i, na_i) = (
            t_len as i32,
            nh as i32,
            ng as i32,
            hd as i32,
            ds as i32,
            rank as i32,
            na as i32,
        );
        let stride_i = self.stride as i32;
        let a_floor = c.a_floor;
        let (d_proj, proj) = (s.d_proj.ptr(), a.proj.ptr());
        let (d_bn_h, d_cn_h, d_tanh_h) = (
            s.d_bn_h.cached_ptr(),
            s.d_cn_h.cached_ptr(),
            s.d_tanh_h.cached_ptr(),
        );
        // The output norm's gate backward runs ahead of either path's scan
        // backward. It leaves the readouts' gradient, du, the z column and
        // the gate's weight partials where the plain gate would, and the norm
        // weight's partials in rows of their own.
        let normed = c.is_outproj_norm;
        if normed {
            let cbuf = &s.cb;
            let (dy, du, gphi, gz, gd, gw) = (
                cbuf.dy.cached_ptr(),
                cbuf.du.cached_ptr(),
                cbuf.gphi.cached_ptr(),
                cbuf.gz.cached_ptr(),
                cbuf.gd.cached_ptr(),
                cbuf.gw.cached_ptr(),
            );
            let (d_o, y) = (s.d_o.ptr(), a.y.cached_ptr());
            let (w_d, w_norm, w_x, w_z, w_o) = (
                at(base, ls.d_param),
                at(base, ls.norm_gate),
                at(base, ls.mimo_x),
                at(base, ls.mimo_z),
                at(base, ls.mimo_o),
            );
            let eps = c.ssm_norm_eps;
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, nh as u32, 1),
                block_dim: (hd.next_multiple_of(32).min(256) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            for (y_first, cfg) in y_runs(cfg) {
                let mut bld = stream.launch_builder(k.cb.gate_normed.get(act));
                bld.arg(&dy);
                bld.arg(&du);
                bld.arg(&d_proj);
                bld.arg(&ld_proj);
                bld.arg(&gphi);
                bld.arg(&gz);
                bld.arg(&gd);
                bld.arg(&gw);
                bld.arg(&d_o);
                bld.arg(&y);
                bld.arg(&proj);
                bld.arg(&w_d);
                bld.arg(&w_norm);
                bld.arg(&w_x);
                bld.arg(&w_z);
                bld.arg(&w_o);
                bld.arg(&nh_i);
                bld.arg(&hd_i);
                bld.arg(&r_i);
                bld.arg(&eps);
                bld.arg(&y_first);
                unsafe { bld.launch(cfg) }
                    .map_err(|e| format!("MIMO output norm backward L{l}: {e:?}"))?;
            }
            self.reduce_axis0(at(g_base, ls.norm_gate), gw, (rows, di))?;
        }
        if let Some(geom) = self.chunked {
            // The chunked path's backward, see kernels/mamba3_mimo_chunked_bwd.cu.
            let nc = t_len.div_ceil(CHUNK);
            let cbuf = &s.cb;
            let (dy, du, dq, dk, dsl, d_init) = (
                cbuf.dy.cached_ptr(),
                cbuf.du.cached_ptr(),
                cbuf.dq.cached_ptr(),
                cbuf.dk.cached_ptr(),
                cbuf.dsl.cached_ptr(),
                cbuf.d_init.cached_ptr(),
            );
            let (dot, inter, dcs, dwp, dsc3, dth) = (
                cbuf.dot.cached_ptr(),
                cbuf.inter.cached_ptr(),
                cbuf.dcs.cached_ptr(),
                cbuf.dwp.cached_ptr(),
                cbuf.dsc3.cached_ptr(),
                cbuf.dth.cached_ptr(),
            );
            let (ddt_ang, de0, gpsi_init, gphi, gz, gpsi) = (
                cbuf.ddt_ang.cached_ptr(),
                cbuf.de0.cached_ptr(),
                cbuf.gpsi_init.cached_ptr(),
                cbuf.gphi.cached_ptr(),
                cbuf.gz.cached_ptr(),
                cbuf.gpsi.cached_ptr(),
            );
            let (gd, gdtb, tmp) = (
                cbuf.gd.cached_ptr(),
                cbuf.gdtb.cached_ptr(),
                cbuf.tmp.cached_ptr(),
            );
            let (coef, sn, cs, e0, enter, y) = (
                a.coef.cached_ptr(),
                a.sn.cached_ptr(),
                a.cs.cached_ptr(),
                a.e0.cached_ptr(),
                a.enter.cached_ptr(),
                a.y.cached_ptr(),
            );
            let (k0, v0, bn, cn, d_o) = (
                a.k0.cached_ptr(),
                a.v0.cached_ptr(),
                a.bn.cached_ptr(),
                a.cn.cached_ptr(),
                s.d_o.ptr(),
            );
            let (w_d, w_x, w_z, w_o) = (
                at(base, ls.d_param),
                at(base, ls.mimo_x),
                at(base, ls.mimo_z),
                at(base, ls.mimo_o),
            );
            let (w_bb, w_cb, w_dtb) = (
                at(base, ls.b_bias),
                at(base, ls.c_bias),
                at(base, ls.dt_bias),
            );
            let heads_i = (2 * di + 2 * bc) as i32;
            let ang_col_i = (2 * di + 2 * bc + 3 * nh) as i32;
            let b_i = b as i32;
            // One launch: the kernel, its grid, a name for errors, its arguments.
            macro_rules! go {
                ($f:expr, $cfg:expr, $what:expr, [$($arg:expr),* $(,)?]) => {{
                    let mut bld = stream.launch_builder($f);
                    $(bld.arg(&$arg);)*
                    unsafe { bld.launch($cfg) }
                        .map_err(|e| format!("MIMO chunked backward {} L{l}: {e:?}", $what))?;
                }};
            }
            // One launch split into runs along its y axis, the run's first y
            // block after the arguments.
            macro_rules! go_y {
                ($f:expr, $cfg:expr, $what:expr, [$($arg:expr),* $(,)?]) => {{
                    for (y_first, cfg) in y_runs($cfg) {
                        let mut bld = stream.launch_builder($f);
                        $(bld.arg(&$arg);)*
                        bld.arg(&y_first);
                        unsafe { bld.launch(cfg) }
                            .map_err(|e| format!("MIMO chunked backward {} L{l}: {e:?}", $what))?;
                    }
                }};
            }
            // One launch split into runs along its y and z axes, the run's
            // first y and z blocks after the arguments.
            macro_rules! go_yz {
                ($f:expr, $cfg:expr, $what:expr, [$($arg:expr),* $(,)?]) => {{
                    for (y_first, z_first, cfg) in yz_runs($cfg) {
                        let mut bld = stream.launch_builder($f);
                        $(bld.arg(&$arg);)*
                        bld.arg(&y_first);
                        bld.arg(&z_first);
                        unsafe { bld.launch(cfg) }
                            .map_err(|e| format!("MIMO chunked backward {} L{l}: {e:?}", $what))?;
                    }
                }};
            }
            let cfg3 = |x: usize, y: usize, z: usize, block: usize, smem: usize| LaunchConfig {
                grid_dim: (x as u32, y as u32, z as u32),
                block_dim: (block as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            if !normed {
                go!(
                    k.cb.gate.get(act),
                    grid_1d_cover(rows * nh * hd)?,
                    "gate",
                    [
                        dy, du, d_proj, ld_proj, gphi, gz, gd, d_o, y, proj, w_d, w_x, w_z, w_o,
                        rows_l, nh_i, hd_i, r_i
                    ]
                );
            }
            if geom.inter_tiles.0 > 0 {
                let (tt, smem) = geom.inter_tiles;
                let tt_i = tt as i32;
                go_y!(
                    &k.cb.inter_tiles,
                    cfg3(
                        b * nc,
                        nh,
                        CHUNK.div_ceil(tt),
                        8 * (tt * rank).next_multiple_of(4),
                        smem
                    ),
                    "inter-chunk term",
                    [
                        dq, inter, dy, enter, cn, w_cb, coef, sn, cs, t_i, nh_i, ng_i, hd_i, ds_i,
                        r_i, na_i, tt_i
                    ]
                );
            } else {
                go_y!(
                    &k.cb.inter,
                    cfg3(b * nc, nh, CHUNK, 256, 0),
                    "inter-chunk term",
                    [
                        dq, inter, dy, enter, cn, w_cb, coef, sn, cs, t_i, nh_i, ng_i, hd_i, ds_i,
                        r_i, na_i
                    ]
                );
            }
            if geom.dsloc_tiles.0 > 0 {
                let (jc, sl, pg, smem) = geom.dsloc_tiles;
                let (jc_i, sl_i, pg_i) = (jc as i32, sl as i32, pg as i32);
                go_yz!(
                    &k.cb.dsloc_tiles,
                    cfg3(
                        b * nc,
                        nh,
                        ds.div_ceil(32 * jc) * hd.div_ceil(pg),
                        8 * pg,
                        smem
                    ),
                    "local state gradient",
                    [
                        dsl, dy, cn, w_cb, coef, sn, cs, t_i, nh_i, ng_i, hd_i, ds_i, r_i, na_i,
                        jc_i, sl_i, pg_i
                    ]
                );
            } else {
                let (nt, dsloc_smem) = geom.dsloc;
                let (nt_i, pt_i) = (nt as i32, geom.pt as i32);
                go_yz!(
                    &k.cb.dsloc,
                    cfg3(
                        b * nc,
                        nh,
                        ds.div_ceil(nt) * hd.div_ceil(geom.pt),
                        nt * geom.pt,
                        dsloc_smem
                    ),
                    "local state gradient",
                    [
                        dsl, dy, cn, w_cb, coef, sn, cs, t_i, nh_i, ng_i, hd_i, ds_i, r_i, na_i,
                        nt_i, pt_i
                    ]
                );
            }
            go_y!(
                &k.cb.pass,
                cfg3(b * nh, (hd * ds).div_ceil(256), 1, 256, 0),
                "state passing",
                [dsl, d_init, coef, b_i, t_i, nh_i, hd_i, ds_i]
            );
            let (count, pn_i) = ((b * nc * nh) as i64, (hd * ds) as i32);
            go!(
                &k.cb.chunk_dot,
                cfg3(b * nc * nh, 1, 1, 256, 0),
                "chunk dot",
                [dot, enter, dsl, count, pn_i]
            );
            // The state terms take the chunk's state gradient into shared
            // memory once per group of tokens when it fits, and their keys
            // and inputs into a global scratch when those do not.
            let (tg, keys_inputs) = (TERMS_GROUP, (rank * ds + rank * hd) * 4);
            let terms_glob = keys_inputs > k.smem_optin;
            let terms_base = if terms_glob { 0 } else { keys_inputs };
            let staged = terms_base + hd * (ds + 1) * 4;
            let (stage, terms_smem) = if staged <= k.smem_optin {
                (1i32, staged)
            } else {
                (0i32, terms_base)
            };
            let glob = if terms_glob {
                s.cb.terms_glob.cached_ptr()
            } else {
                0u64
            };
            let tg_i = tg as i32;
            if geom.terms_tiles.0 > 0 {
                let (tt, smem) = geom.terms_tiles;
                let tt_i = tt as i32;
                go_y!(
                    k.cb.state_terms_tiles.get(act),
                    cfg3(
                        b * nc,
                        nh,
                        CHUNK.div_ceil(tt),
                        8 * (tt * rank).next_multiple_of(4),
                        smem
                    ),
                    "state terms",
                    [
                        dk, du, dcs, dsl, proj, ld_proj, bn, w_bb, coef, sn, cs, w_x, t_i, nh_i,
                        ng_i, hd_i, ds_i, r_i, na_i, tt_i
                    ]
                );
            } else {
                go_y!(
                    k.cb.state_terms.get(act),
                    cfg3(b * nc, nh, CHUNK.div_ceil(tg), 256, terms_smem),
                    "state terms",
                    [
                        dk, du, dcs, dsl, proj, ld_proj, bn, w_bb, coef, sn, cs, w_x, glob, t_i,
                        nh_i, ng_i, hd_i, ds_i, r_i, na_i, tg_i, stage
                    ]
                );
            }
            let (tl, pairs_smem) = geom.pairs;
            let tl_i = tl as i32;
            if geom.pairs_tiles.0 > 0 {
                let (dq_smem, dk_smem) = geom.pairs_tiles;
                let threads = 8 * (tl * rank).next_multiple_of(4);
                let tiles = CHUNK.div_ceil(tl);
                go_y!(
                    k.cb.pairs_dq.get(act),
                    cfg3(b * nc, nh, tiles, threads, dq_smem),
                    "pairs dq",
                    [
                        dq, dy, proj, ld_proj, bn, w_bb, coef, sn, cs, w_x, t_i, nh_i, ng_i, hd_i,
                        ds_i, r_i, na_i, tl_i
                    ]
                );
                go_y!(
                    k.cb.pairs_dk.get(act),
                    cfg3(b * nc, nh, tiles, threads, dk_smem),
                    "pairs dk",
                    [
                        dk, du, dwp, dy, proj, ld_proj, bn, cn, w_bb, w_cb, coef, sn, cs, w_x, t_i,
                        nh_i, ng_i, hd_i, ds_i, r_i, na_i, tl_i
                    ]
                );
            } else {
                let pairs_glob = if geom.pairs_global {
                    s.cb.pairs_glob.cached_ptr()
                } else {
                    0u64
                };
                go_y!(
                    k.cb.pairs.get(act),
                    cfg3(b * nc, nh, 1, 256, pairs_smem),
                    "pairs",
                    [
                        dq, dk, du, dwp, dy, proj, ld_proj, bn, cn, w_bb, w_cb, coef, sn, cs, w_x,
                        pairs_glob, t_i, nh_i, ng_i, hd_i, ds_i, r_i, na_i, tl_i
                    ]
                );
            }
            // A thread per channel, or for a head wider than the block a
            // power of two of them taking the channels in turns.
            let lanes = hd.next_multiple_of(32);
            let lead_wide = lanes > k.chunk_threads.generic;
            let lead_threads = if lead_wide {
                1usize << k.chunk_threads.generic.ilog2()
            } else {
                lanes
            };
            let leaves = if lead_wide {
                s.cb.lead_leaves.cached_ptr()
            } else {
                0u64
            };
            go_y!(
                &k.cb.lead,
                cfg3(b, nh, 1, lead_threads, lanes.min(lead_threads) * 4),
                "entering lead",
                [
                    de0, gpsi_init, d_init, e0, k0, v0, w_x, leaves, nh_i, hd_i, ds_i, r_i
                ]
            );
            go_y!(
                &k.cb.combine,
                cfg3(b * nc, nh, 1, CHUNK, 0),
                "decay gradients",
                [dsc3, dwp, dcs, inter, dot, coef, t_i, nh_i]
            );
            go_y!(
                &k.cb.rot_bwd,
                cfg3(rows, nh, 1, 128, 0),
                "rotation",
                [
                    d_bn_h, d_cn_h, dth, dq, dk, bn, cn, w_bb, w_cb, sn, cs, nh_i, ng_i, ds_i, r_i,
                    na_i
                ]
            );
            if na > 0 {
                go_y!(
                    k.cb.angles.get(act),
                    cfg3(b, nh, 1, 32, na * 4),
                    "angles",
                    [
                        d_tanh_h, ddt_ang, dth, coef, proj, ld_proj, ang_col_i, t_i, nh_i, na_i
                    ]
                );
            }
            go!(
                k.cb.scalars.get(act),
                grid_1d_cover(rows * nh)?,
                "step scalars",
                [
                    d_proj, gdtb, dsc3, ddt_ang, de0, coef, proj, ld_proj, w_dtb, rows_l, t_i,
                    nh_i, heads_i, a_floor
                ]
            );
            go!(
                k.cb.u_bwd.get(act),
                grid_1d_cover(rows * nh * hd)?,
                "rank inputs",
                [
                    d_proj, gpsi, du, proj, ld_proj, w_x, rows_l, nh_i, hd_i, r_i
                ]
            );
            // The weight gradients from their per-row partials.
            let n_rp = nh * rank * hd;
            self.reduce_axis0(at(g_base, ls.mimo_o), gphi, (rows, n_rp))?;
            self.reduce_axis0(at(g_base, ls.mimo_z), gz, (rows, n_rp))?;
            self.reduce_axis0(at(g_base, ls.mimo_x), gpsi, (rows, n_rp))?;
            {
                let (gp, count_b, n_l) = (at(g_base, ls.mimo_x), b_i, n_rp as i64);
                go!(
                    &k.sum_rows,
                    grid_1d_cover(n_rp)?,
                    "entering Psi",
                    [gp, gpsi_init, count_b, n_l]
                );
            }
            self.reduce_axis0(at(g_base, ls.dt_bias), gdtb, (rows, nh))?;
            self.reduce_axis0(at(g_base, ls.b_bias), d_bn_h, (rows, nh * rank * ds))?;
            self.reduce_axis0(at(g_base, ls.c_bias), d_cn_h, (rows, nh * rank * ds))?;
            self.reduce_axis0_set(tmp, gd, (rows, nh * hd))?;
            {
                let (gp, n_i, inner) = (at(g_base, ls.d_param), nh_i, hd_i);
                go!(&k.cb.sum_inner, grid_1d(nh), "D", [gp, tmp, n_i, inner]);
            }
        } else {
            // The scan backward.
            {
                let parts = [
                    s.g_x.cached_ptr(),
                    s.g_z.cached_ptr(),
                    s.g_o.cached_ptr(),
                    s.g_d.cached_ptr(),
                    s.g_dtb.cached_ptr(),
                    s.g_bb.cached_ptr(),
                    s.g_cb.cached_ptr(),
                ];
                let rec = s.rec.cached_ptr();
                let d_o = s.d_o.ptr();
                let (bn, cn) = (a.bn.cached_ptr(), a.cn.cached_ptr());
                let cks = [
                    a.ck_ssm.cached_ptr(),
                    a.ck_k.cached_ptr(),
                    a.ck_v.cached_ptr(),
                    a.ck_ang.cached_ptr(),
                ];
                let weights = [
                    at(base, ls.dt_bias),
                    at(base, ls.b_bias),
                    at(base, ls.c_bias),
                    at(base, ls.d_param),
                    at(base, ls.mimo_x),
                    at(base, ls.mimo_z),
                    at(base, ls.mimo_o),
                ];
                let (kernel, threads, smem) = match self.seq_bwd {
                    None => (
                        &k.scan_bwd,
                        scan_bwd_threads(hd),
                        scan_bwd_smem_bytes((hd, ds, rank, na)),
                    ),
                    Some(g) => (&k.scan_bwd_global, g.threads, g.smem),
                };
                let cfg = LaunchConfig {
                    grid_dim: (b as u32, nh as u32, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: smem as u32,
                };
                // A null scratch keeps its group in shared memory.
                let scratch = self.seq_bwd.map(|g| {
                    let home = |global: bool, buf: &GpuBuffer| {
                        if global { buf.cached_ptr() } else { 0u64 }
                    };
                    [
                        home(!g.ds_shared, &s.bwd_ds),
                        home(!g.keys_shared, &s.bwd_kg),
                        home(!g.tiles_shared, &s.bwd_pr),
                        s.bwd_ch.cached_ptr(),
                        home(g.wide, &s.bwd_leaves),
                    ]
                });
                for (y_first, cfg) in y_runs(cfg) {
                    let mut bld = stream.launch_builder(kernel.get(act));
                    bld.arg(&d_proj);
                    bld.arg(&ld_proj);
                    bld.arg(&d_bn_h);
                    bld.arg(&d_cn_h);
                    bld.arg(&d_tanh_h);
                    for p in &parts {
                        bld.arg(p);
                    }
                    bld.arg(&rec);
                    if let Some(ptrs) = &scratch {
                        for p in ptrs {
                            bld.arg(p);
                        }
                    }
                    bld.arg(&d_o);
                    bld.arg(&proj);
                    bld.arg(&bn);
                    bld.arg(&cn);
                    for p in &cks {
                        bld.arg(p);
                    }
                    for p in &weights {
                        bld.arg(p);
                    }
                    bld.arg(&t_i);
                    bld.arg(&stride_i);
                    bld.arg(&nh_i);
                    bld.arg(&ng_i);
                    bld.arg(&hd_i);
                    bld.arg(&ds_i);
                    bld.arg(&r_i);
                    bld.arg(&na_i);
                    bld.arg(&a_floor);
                    let dy_in = if normed { s.cb.dy.cached_ptr() } else { 0u64 };
                    bld.arg(&dy_in);
                    bld.arg(&y_first);
                    unsafe { bld.launch(cfg) }
                        .map_err(|e| format!("MIMO scan backward L{l}: {e:?}"))?;
                }
            }
            // The per-sequence partials into the gradients. Under the output
            // norm the gate's three come from its pre-pass instead.
            for (g, part, n, gate) in [
                (ls.mimo_x, s.g_x.cached_ptr(), nh * rank * hd, false),
                (ls.mimo_z, s.g_z.cached_ptr(), nh * rank * hd, true),
                (ls.mimo_o, s.g_o.cached_ptr(), nh * rank * hd, true),
                (ls.d_param, s.g_d.cached_ptr(), nh, true),
                (ls.dt_bias, s.g_dtb.cached_ptr(), nh, false),
                (ls.b_bias, s.g_bb.cached_ptr(), nh * rank * ds, false),
                (ls.c_bias, s.g_cb.cached_ptr(), nh * rank * ds, false),
            ] {
                if gate && normed {
                    continue;
                }
                let (gp, count, n_l) = (at(g_base, g), b as i32, n as i64);
                let mut bld = stream.launch_builder(&k.sum_rows);
                bld.arg(&gp);
                bld.arg(&part);
                bld.arg(&count);
                bld.arg(&n_l);
                unsafe { bld.launch(grid_1d_cover(n)?) }
                    .map_err(|e| format!("MIMO weight partials L{l}: {e:?}"))?;
            }
            if normed {
                let n_rp = nh * rank * hd;
                let tmp = s.cb.tmp.cached_ptr();
                self.reduce_axis0(at(g_base, ls.mimo_o), s.cb.gphi.cached_ptr(), (rows, n_rp))?;
                self.reduce_axis0(at(g_base, ls.mimo_z), s.cb.gz.cached_ptr(), (rows, n_rp))?;
                self.reduce_axis0_set(tmp, s.cb.gd.cached_ptr(), (rows, nh * hd))?;
                let gp = at(g_base, ls.d_param);
                let mut bld = stream.launch_builder(&k.cb.sum_inner);
                bld.arg(&gp);
                bld.arg(&tmp);
                bld.arg(&nh_i);
                bld.arg(&hd_i);
                unsafe { bld.launch(grid_1d(nh)) }
                    .map_err(|e| format!("MIMO output norm D L{l}: {e:?}"))?;
            }
        }
        // The B and C norms, then their weights.
        {
            let (dw_b, dw_c) = (s.dw_b.cached_ptr(), s.dw_c.cached_ptr());
            let (b_col, c_col) = ((2 * di) as i32, (2 * di + bc) as i32);
            let (wb, wc) = (at(base, ls.b_norm_weight), at(base, ls.c_norm_weight));
            let eps = c.ssm_norm_eps;
            let esz = act.size_bytes();
            for (first, n) in row_pieces(rows, rank * ng) {
                let (rows_p, proj_at) = (n as i64, (first * ip * esz) as u64);
                let normed = (first * rank * ng * ds * 4) as u64;
                let heads = (first * nh * rank * ds * 4) as u64;
                let (d_proj_p, proj_p) = (d_proj + proj_at, proj + proj_at);
                let (dw_b_p, dw_c_p) = (dw_b + normed, dw_c + normed);
                let (d_bn_p, d_cn_p) = (d_bn_h + heads, d_cn_h + heads);
                let mut bld = stream.launch_builder(k.bc_norm_bwd.get(act));
                bld.arg(&d_proj_p);
                bld.arg(&dw_b_p);
                bld.arg(&dw_c_p);
                bld.arg(&d_bn_p);
                bld.arg(&d_cn_p);
                bld.arg(&proj_p);
                bld.arg(&ld_proj);
                bld.arg(&b_col);
                bld.arg(&c_col);
                bld.arg(&wb);
                bld.arg(&wc);
                bld.arg(&rows_p);
                bld.arg(&nh_i);
                bld.arg(&ng_i);
                bld.arg(&r_i);
                bld.arg(&ds_i);
                bld.arg(&eps);
                let cfg = LaunchConfig {
                    grid_dim: ((n * rank * ng) as u32, 2, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { bld.launch(cfg) }
                    .map_err(|e| format!("MIMO B/C norm backward L{l}: {e:?}"))?;
            }
            self.reduce_axis0(at(g_base, ls.b_norm_weight), dw_b, (rows * rank * ng, ds))?;
            self.reduce_axis0(at(g_base, ls.c_norm_weight), dw_c, (rows * rank * ng, ds))?;
        }
        // The angle projection.
        if na > 0 {
            let ang_col = (2 * di + 2 * bc + 3 * nh) as i32;
            let mut bld = stream.launch_builder(k.angle_bwd.get(act));
            bld.arg(&d_proj);
            bld.arg(&d_tanh_h);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&ang_col);
            bld.arg(&rows_l);
            bld.arg(&nh_i);
            bld.arg(&na_i);
            unsafe { bld.launch(grid_1d_cover(rows * na)?) }
                .map_err(|e| format!("MIMO angle backward L{l}: {e:?}"))?;
        }
        // in_proj, the block norm, the residual.
        let gwi = self.grad(ls.in_proj_w);
        let wi = self.gemm_weight(ls.in_proj_w)?;
        if act == WeightDtype::F32 {
            let bwd = &mut self.bwd;
            gpu_gemm_bi_backward_grad_raw(
                &self.ctx,
                bwd.d_post.f32_mut()?,
                (&gwi, None),
                bwd.d_proj.f32()?,
                self.acts[l].post.f32()?,
                wi,
                (rows, dm, ip),
            )?;
        } else {
            self.half_linear_backward(
                (
                    self.bwd.d_post.ptr(),
                    self.bwd.d_proj.ptr(),
                    self.acts[l].post.ptr(),
                ),
                (wi, &gwi),
                (rows, dm, ip),
            )?;
        }
        let a = &self.acts[l];
        self.norm_backward(
            (
                self.bwd.d_pre.cached_ptr(),
                self.bwd.d_post.ptr(),
                a.x_in.cached_ptr(),
                a.rms.cached_ptr(),
            ),
            ((ls.norm_weight, ls.norm_bias), true),
            rows,
        )?;
        self.add_residual_grad(rows)
    }

    /// One AdamW step on the gradients of the last backward (the matrices
    /// decay under the reference no-decay groups). Returns false when the
    /// f16 gradients overflowed: the step is skipped and the loss scale
    /// backs off.
    ///
    /// Each backward's gradients take one step: a second call before the
    /// next backward is refused rather than applying them twice.
    pub fn apply_adamw(&mut self) -> Result<bool, String> {
        if !self.fresh_grads {
            return Err("no new gradients since the last optimizer step".into());
        }
        self.fresh_grads = false;
        self.apply_adamw_unchecked()
    }

    /// Build the fused AdamW's chunk table over the layout, or rebuild it
    /// when the optimizer settings it bakes in changed since: under the
    /// reference's no-decay groups only the matrices decay.
    fn ensure_adamw_plan(&mut self) -> Result<(), String> {
        let key = AdamWPlanKey::of(&self.adamw);
        if matches!(&self.multi_plan, Some((k, _)) if *k == key) {
            return Ok(());
        }
        let (w_base, g_base) = (self.master.cached_ptr(), self.grads.cached_ptr());
        let specs: Vec<AdamWTensorSpec> = self
            .layout
            .pairs(&self.template)
            .into_iter()
            .map(|(_, span, decays)| AdamWTensorSpec {
                weight: at(w_base, span),
                grad: at(g_base, span),
                out: 0,
                out_elt_bytes: 0,
                out_is_f32: false,
                len: span.len,
                no_decay: !decays,
            })
            .collect();
        let plan = build_multi_plan(
            &self.ctx.stream,
            &self.adamw,
            g_base,
            &specs,
            self.adamw.reference_no_decay,
            self.adamw.weight_decay,
        )?;
        self.multi_plan = Some((key, plan));
        Ok(())
    }

    fn apply_adamw_unchecked(&mut self) -> Result<bool, String> {
        if let Some(scaler) = self.scaler.as_mut() {
            scaler.update(self.overflowed);
            if self.overflowed {
                return Ok(false);
            }
        }
        self.prepare_adamw()?;
        self.adamw_device(0)?;
        self.sync("AdamW")?;
        Ok(true)
    }

    /// The AdamW chunk table checked against the optimizer, the step
    /// counted, and its bias corrections and learning rate written for the
    /// device.
    fn prepare_adamw(&mut self) -> Result<(), String> {
        self.ensure_adamw_plan()?;
        let (_, bias_c1, bias_c2) = self.adamw.advance();
        self.bias
            .write(&self.ctx.stream, bias_c1, bias_c2, self.adamw.lr)
    }

    /// The fused AdamW over every tensor, then the half copies' refresh.
    /// `skip` is 0, or the f16 overflow flag, which leaves the weights and
    /// moments as they are when set: the optimizer part of a captured
    /// step.
    fn adamw_device(&self, skip: u64) -> Result<(), String> {
        let Some((_, plan)) = &self.multi_plan else {
            return Err("the AdamW chunk table is missing".into());
        };
        step_multi(
            &self.ctx,
            self.ctx.kernels.adamw_step_multi.get(WeightDtype::F32),
            plan,
            &self.adamw,
            self.bias.ptr(),
            skip,
        )?;
        self.refresh_mirror()
    }
}
