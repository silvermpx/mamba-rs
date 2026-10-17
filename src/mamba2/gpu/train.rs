//! Mamba-2 training on the GPU, in f32, tf32, bf16 or f16.
//!
//! The forward runs the prefill's launches (the shared `M2Exec` pieces)
//! and keeps, per layer, what the backward reads: the layer's input rows
//! and norm, the normed rows, the projection, the conv output and the conv
//! state the window started from, dt and its chunk cumsum, the C.B
//! products, the state entering every chunk, the scan output and the gated
//! rows, and for the gated MLP after the mixer its input rows and norm, the
//! normed rows, the up projection and the gate's rows. The backward walks
//! the layers in reverse through the kernels of
//! `kernels/mamba2_bwd.cu` and the Inference module's GEMMs and the Common module's norm and reduction
//! kernels. The weights, their gradients and the AdamW moments are three
//! flat buffers in one layout, so the optimizer runs per tensor with no
//! copies.
//!
//! In tf32 everything stays f32 and the GEMMs take the context's TF32
//! contract. In bf16 and f16 the weights, gradients and moments stay f32;
//! in_proj, out_proj and the MLP's two projections run from half copies
//! refreshed after every step, and the rows those GEMMs read or write (the
//! normed rows, the projection, the conv output, the gated rows, the MLP's
//! rows and their gradients) are stored half,
//! while the residual stream, the scan, the states and every reduction
//! stay f32. f16 scales the loss with the dynamic loss scaler: a step whose
//! gradients overflow is skipped and the scale backs off.
//!
//! Windows carry the conv and SSM states from one call to the next, and
//! the gradients stop at the states a window starts from.

mod graph;

use crate::bidi::{mlps_after, reads_reversed};
use crate::config::ModelFamily;
use crate::gpu::bidi::{WindowRows, reverse_rows, zero_bytes};
use crate::gpu::grad_clip::{GradRegionGeom, StepTail, norm_limit, read_tail, scale_grads};
use crate::gpu::grad_window::{FoldOverflow, GradWindow, check_backward_opts};
use crate::gpu::options::GpuOptions;
use crate::gpu::train::{
    BackwardMetrics, BackwardOpts, InputGrad, RowSource, StepMetrics, TrainSessionCfg,
};
use crate::gpu::train_state::TrainerState;
use crate::gpu::varlen::GpuVarlen;
use crate::varlen::Varlen;
use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::inference::check_device_envelope;
use super::inference::{LayerPtrs, M2Exec, Mamba2GpuState, MlpPtrs, WindowPtrs};
use super::kernels::{
    BC_TILES_SMEM_BYTES, CHUNK, DX_GROUP_CHANNELS, DX_GROUPS_SMEM_BYTES, Mamba2Kernels,
    PAIRS_TILES_SMEM_BYTES, SeqBwdRoute, chunk_state_geom, dx_tiles_geom,
    gated_norm_bwd_smem_bytes, seq_bwd_smem_bytes, seq_bwd_tiles_smem_bytes, seq_bwd_tiles_threads,
};
use crate::gpu::adamw::{
    AdamWBiasFactors, AdamWMultiPlan, AdamWPlanKey, AdamWStateBlob, AdamWTensorSpec, GpuAdamW,
    build_multi_plan, step_multi,
};
use crate::gpu::blas::{
    TypedPtr, gpu_gemm_bi_backward_dw_grad_typed, gpu_gemm_bi_backward_grad_raw,
    gpu_gemm_ex_backward_dx_typed, gpu_gemm_f32_forward_ptrs,
};
use crate::gpu::block_norm::{self, NormParams};
use crate::gpu::buffers::{DtypedBuf, GpuBuffer, GradSlice};
use crate::gpu::context::{GemmRole, GpuCtx, gemm_mode_or_env};
use crate::gpu::device::GpuDevice;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::gemm_bi_inference::prepare_inference_arch_rung;
use crate::gpu::kernel_identity::PreparedGemmCaptureManifest;
use crate::gpu::launch::{grid_1d, grid_colsum, grid_rows, grid_steps, launch_cast, y_runs};
use crate::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, UnscaleFactor, check_inf_nan_gpu, scale_grads_skip_gpu,
};
use crate::mamba2::config::Mamba2Config;
use crate::mamba2::weights::{Mamba2LayerWeights, Mamba2Weights};
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
    in_proj_b: Span,
    conv_w: Span,
    conv_b: Span,
    dt_bias: Span,
    a_log: Span,
    d_param: Span,
    gate_norm_weight: Span,
    out_proj_w: Span,
    out_proj_b: Span,
}

/// One layer's gated MLP in the flat buffers.
#[derive(Clone, Copy, Debug, Default)]
struct MlpSpans {
    norm_weight: Span,
    norm_bias: Span,
    fc1_w: Span,
    fc2_w: Span,
}

/// The order of the flat buffers: the input projection, then each layer's
/// tensors in [`Mamba2LayerWeights`] field order, then each layer's MLP,
/// then the final norm. A tensor the model does not have is an empty span.
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
    fn of(w: &Mamba2Weights) -> Self {
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
                in_proj_b: take(lw.in_proj_b.len()),
                conv_w: take(lw.conv_w.len()),
                conv_b: take(lw.conv_b.len()),
                dt_bias: take(lw.dt_bias.len()),
                a_log: take(lw.a_log.len()),
                d_param: take(lw.d_param.len()),
                gate_norm_weight: take(lw.gate_norm_weight.len()),
                out_proj_w: take(lw.out_proj_w.len()),
                out_proj_b: take(lw.out_proj_b.len()),
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

    /// Every (tensor, its span) pair of `w`, in layout order. The flag
    /// marks the matrices, the tensors AdamW decays under the reference's
    /// no-decay groups.
    fn pairs<'w>(&self, w: &'w Mamba2Weights) -> Vec<(&'w [f32], Span, bool)> {
        let mut out = vec![
            (w.input_proj_w.as_slice(), self.input_proj_w, true),
            (w.input_proj_b.as_slice(), self.input_proj_b, false),
        ];
        for (lw, ls) in w.layers.iter().zip(&self.layers) {
            out.extend([
                (lw.norm_weight.as_slice(), ls.norm_weight, false),
                (lw.norm_bias.as_slice(), ls.norm_bias, false),
                (lw.in_proj_w.as_slice(), ls.in_proj_w, true),
                (lw.in_proj_b.as_slice(), ls.in_proj_b, false),
                (lw.conv_w.as_slice(), ls.conv_w, true),
                (lw.conv_b.as_slice(), ls.conv_b, false),
                (lw.dt_bias.as_slice(), ls.dt_bias, false),
                (lw.a_log.as_slice(), ls.a_log, false),
                (lw.d_param.as_slice(), ls.d_param, false),
                (lw.gate_norm_weight.as_slice(), ls.gate_norm_weight, false),
                (lw.out_proj_w.as_slice(), ls.out_proj_w, true),
                (lw.out_proj_b.as_slice(), ls.out_proj_b, false),
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

    fn flatten(&self, w: &Mamba2Weights) -> Vec<f32> {
        let mut flat = vec![0f32; self.total];
        for (src, span, _) in self.pairs(w) {
            flat[span.off..span.off + span.len].copy_from_slice(src);
        }
        flat
    }

    /// `template` with every tensor read back from `flat`.
    fn unflatten(&self, flat: &[f32], template: &Mamba2Weights) -> Mamba2Weights {
        let mut w = template.clone();
        let read = |v: &mut Vec<f32>, span: Span| {
            v.copy_from_slice(&flat[span.off..span.off + span.len]);
        };
        read(&mut w.input_proj_w, self.input_proj_w);
        read(&mut w.input_proj_b, self.input_proj_b);
        for (lw, ls) in w.layers.iter_mut().zip(&self.layers) {
            let Mamba2LayerWeights {
                norm_weight,
                norm_bias,
                in_proj_w,
                in_proj_b,
                conv_w,
                conv_b,
                dt_bias,
                a_log,
                d_param,
                gate_norm_weight,
                out_proj_w,
                out_proj_b,
            } = lw;
            read(norm_weight, ls.norm_weight);
            read(norm_bias, ls.norm_bias);
            read(in_proj_w, ls.in_proj_w);
            read(in_proj_b, ls.in_proj_b);
            read(conv_w, ls.conv_w);
            read(conv_b, ls.conv_b);
            read(dt_bias, ls.dt_bias);
            read(a_log, ls.a_log);
            read(d_param, ls.d_param);
            read(gate_norm_weight, ls.gate_norm_weight);
            read(out_proj_w, ls.out_proj_w);
            read(out_proj_b, ls.out_proj_b);
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

/// The control region of the gradient arena: the dt columns of every
/// layer's in_proj (the last nheads of its `[z0 | x0 | z | xBC | dt]`
/// split), dt_bias and A_log, which lie one after the other, and the dt
/// entries of the in_proj bias where the model has the bias. These are the
/// gradients of the step size and the decay, whose magnitude grows with
/// the window's length. The layers repeat with one stride, since the
/// layout holds the mixers one after another and the MLPs after them.
fn control_region_geom(layout: &Layout, cfg: &Mamba2Config) -> GradRegionGeom {
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
    let (dm, ip, nh) = (cfg.d_model, cfg.in_proj_out_dim(), cfg.nheads());
    debug_assert_eq!(l0.a_log.off, l0.dt_bias.off + l0.dt_bias.len);
    let (c_off, c_len) = if l0.in_proj_b.len == 0 {
        (0, 0)
    } else {
        (l0.in_proj_b.off + ip - nh, nh)
    };
    GradRegionGeom {
        n_layers: layout.layers.len() as i32,
        layer_stride: layout
            .layers
            .get(1)
            .map_or(0, |l1| (l1.norm_weight.off - l0.norm_weight.off) as i64),
        a_off: (l0.in_proj_w.off + ip - nh) as i64,
        a_rows: dm as i32,
        a_cols: nh as i32,
        a_row_stride: ip as i64,
        b_off: l0.dt_bias.off as i64,
        b_len: (l0.dt_bias.len + l0.a_log.len) as i32,
        c_off: c_off as i64,
        c_len: c_len as i32,
    }
}

fn opt_at(base: u64, span: Span) -> Option<u64> {
    (span.len > 0).then(|| at(base, span))
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
    xbc: Rows,
    conv0: GpuBuffer,
    dt: GpuBuffer,
    acs: GpuBuffer,
    cb: GpuBuffer,
    states: GpuBuffer,
    /// The sequential path's replay checkpoints, [B, S, H, P, N].
    ckpt: GpuBuffer,
    y: GpuBuffer,
    gated: Rows,
    /// The gated MLP's own rows, for a layer with one.
    mlp: Option<MlpActs>,
}

/// What the gated MLP's forward keeps for its backward: its input rows and
/// norm, the normed rows, the up projection and the gate's rows.
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
    d_gated: Rows,
    dy: GpuBuffer,
    d_proj: Rows,
    dxbc: GpuBuffer,
    ds: GpuBuffer,
    /// The sequential path's replay records and per-head dB and dC.
    rec: GpuBuffer,
    /// The sequential backward's step scalars, [rows, H, 3].
    seq_scal: GpuBuffer,
    /// dS of the global sequential backward when it does not fit shared
    /// memory, [B, H, P, N].
    seq_ds: GpuBuffer,
    /// The global sequential backward's block-sum leaves for a head wider
    /// than its block, [B, H, 2, lanes].
    seq_leaves: GpuBuffer,
    db_h: GpuBuffer,
    dc_h: GpuBuffer,
    acc: GpuBuffer,
    ste: GpuBuffer,
    yi: GpuBuffer,
    dcbh: GpuBuffer,
    mm: GpuBuffer,
    ddtx: GpuBuffer,
    qv: GpuBuffer,
    inter: GpuBuffer,
    dot: GpuBuffer,
    da_parts: GpuBuffer,
    dbias_parts: GpuBuffer,
    d_parts: GpuBuffer,
    dw_parts: GpuBuffer,
    dv: GpuBuffer,
    d_post: Rows,
    d_pre: GpuBuffer,
    /// d loss / d (a bidirectional pair's output rows), which both of the
    /// pair's mixers take while the stream's gradient gathers theirs (one
    /// element without pairs).
    d_pair: GpuBuffer,
    /// The gradients of the MLP's up projection and gate rows.
    d_up: Rows,
    d_act: Rows,
    axis0: GpuBuffer,
    d_input: GpuBuffer,
    /// The residual gradient rounded to the half type, the out_proj
    /// backward's input in the half lanes.
    d_temp_h: Option<DtypedBuf>,
}

/// A Mamba-2 trainer's carried recurrence on the host, layer after layer
/// and within a layer sequence after sequence: the conv window `[conv_dim,
/// d_conv]` of the last inputs, newest last, and the SSM state `[nheads,
/// headdim, d_state]`. A window's first column, its oldest input, is read
/// by no later step and is zero here, so the CPU trainer, which keeps only
/// the inputs a step reads, exports the same state.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2RecurrentStateBlob {
    pub conv: Vec<f32>,
    pub ssm: Vec<f32>,
}

/// The Mamba-2 trainer: forward, backward and AdamW for windows of
/// `seq_len` tokens of `batch` sequences.
pub struct Mamba2GpuTrainer {
    /// The captured training step; dropped before the buffers it runs on.
    graph: Option<graph::TrainGraph>,
    /// The GEMM inventory of the last eager step, which a capture takes.
    eager_gemm_manifest: Option<PreparedGemmCaptureManifest>,
    pub kernels: Mamba2Kernels,
    pub ctx: GpuCtx,
    cfg: Mamba2Config,
    batch: usize,
    seq_len: usize,
    input_dim: usize,
    dtype: WeightDtype,
    act: WeightDtype,
    /// The sequential scan (ScanMode::Sequential) instead of the chunked
    /// one, and the tokens between its replay checkpoints.
    seq_mode: bool,
    stride: usize,
    /// The fused dx and token sums' (state indices per pass, shared bytes)
    /// on the chunked path when the head fits; else the two kernels run.
    dx_tiles: Option<(usize, usize)>,
    /// The sequential backward's kernel for the head (the chunked path
    /// never reads it).
    seq_bwd: SeqBwdRoute,
    template: Mamba2Weights,
    layout: Layout,
    master: GpuBuffer,
    grads: GpuBuffer,
    adamw: GpuAdamW,
    a_neg: GpuBuffer,
    conv_zero_b: GpuBuffer,
    state: Mamba2GpuState,
    acts: Vec<LayerActs>,
    input: GpuBuffer,
    temporal: GpuBuffer,
    final_in: GpuBuffer,
    final_rms: GpuBuffer,
    final_out: GpuBuffer,
    conv_next: GpuBuffer,
    out_rows: Rows,
    bwd: BwdScratch,
    /// The half copies of in_proj and out_proj (half lanes only), and the
    /// (master, copy) spans of each.
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
    /// The last backward's gradients overflowed (f16), so the next AdamW
    /// step is skipped.
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
    /// The sequences packed into the rows of the windows (see
    /// [`Self::set_varlen`]).
    varlen: GpuVarlen,
}

const PAIRS: usize = CHUNK * (CHUNK + 1) / 2;

/// Tokens between the sequential path's replay checkpoints: the replay
/// records one segment of states, so the stride trades that buffer against
/// the checkpoints' memory.
const SEQ_STRIDE: usize = 16;

impl Mamba2GpuTrainer {
    /// Build a trainer for the session `session` names (shape and AdamW
    /// settings, see [`TrainSessionCfg::new`]) from `w`, on the device, in
    /// the storage precision and in the GEMM mode `opts` names: f32, tf32,
    /// bf16 or f16 (see the module docs).
    pub fn new(
        w: &Mamba2Weights,
        cfg: Mamba2Config,
        session: TrainSessionCfg,
        opts: GpuOptions,
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
        crate::mamba2::route::admit(&cfg, input_dim, &request)?;
        let device = GpuDevice::new(opts.device)?;
        let device = &device;
        let seq_mode = !cfg.train_use_parallel_scan();
        let stride = SEQ_STRIDE.min(seq_len);
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
        let kernels = Mamba2Kernels::compile(device.context(), arch)?;
        check_device_envelope(&cfg, &kernels)?;
        let optin = kernels.smem_optin;
        let layout = Layout::of(w);
        let s = &ctx.stream;
        let z = |n: usize| GpuBuffer::zeros(s, n.max(1));
        let typed = |n: usize| Rows::zeros(s, n, act);
        let c = &cfg;
        let (rows, nc) = (batch * seq_len, seq_len.div_ceil(CHUNK));
        let (dm, ip, cd, d_ssm, di) = (
            c.d_model,
            c.in_proj_out_dim(),
            c.conv_dim(),
            c.ssm_dim(),
            c.d_inner(),
        );
        let (nh, hd, ds) = (c.nheads(), c.headdim, c.d_state);
        let hidden = c.mlp_hidden();
        // Each block norm keeps one statistic per row for RMSNorm, two for
        // LayerNorm.
        let stats = c.block_norm.stats_per_row();
        let per_chunk = batch * nc * nh;
        let nseg = seq_len.div_ceil(stride);
        // The chunked path's per-chunk buffers, or the sequential path's
        // checkpoints; the other path's are left one element long.
        let chunked = |n: usize| if seq_mode { 1 } else { n };
        let sequential = |n: usize| if seq_mode { n } else { 1 };
        // The fused dx keeps acc, ste and yi in shared memory.
        let dx_tiles = if seq_mode {
            None
        } else {
            dx_tiles_geom(hd, ds, optin, kernels.bwd.dx_tiles_threads)
        };
        let unfused = |n: usize| if dx_tiles.is_some() { 1 } else { chunked(n) };
        let seq_bwd = kernels.seq_bwd_route(hd, ds);
        let (seq_ds_len, seq_leaves_len) = match seq_bwd {
            SeqBwdRoute::Global(g) => (
                if g.shared_ds { 1 } else { batch * nh * hd * ds },
                if g.wide {
                    batch * nh * 2 * hd.next_multiple_of(32)
                } else {
                    1
                },
            ),
            _ => (1, 1),
        };
        let acts = (0..c.n_layers)
            .map(|l| -> Result<LayerActs, String> {
                Ok(LayerActs {
                    x_in: z(rows * dm)?,
                    rms: z(rows * stats)?,
                    post: typed(rows * dm)?,
                    proj: typed(rows * ip)?,
                    xbc: typed(rows * cd)?,
                    conv0: z(batch * cd * c.d_conv)?,
                    dt: z(chunked(per_chunk * CHUNK))?,
                    acs: z(chunked(per_chunk * CHUNK))?,
                    cb: z(chunked(batch * nc * c.ngroups * PAIRS))?,
                    states: z(chunked(per_chunk * hd * ds))?,
                    ckpt: z(sequential(batch * nseg * nh * hd * ds))?,
                    y: z(rows * d_ssm)?,
                    gated: typed(rows * di)?,
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
        let bwd = BwdScratch {
            d_rows: z(rows * dm)?,
            d_temp: z(rows * dm)?,
            d_gated: typed(rows * di)?,
            dy: z(rows * d_ssm)?,
            d_proj: typed(rows * ip)?,
            dxbc: z(rows * cd)?,
            ds: z(chunked(per_chunk * hd * ds))?,
            rec: z(sequential(batch * nh * stride * hd * ds))?,
            seq_scal: z(sequential(if seq_bwd == SeqBwdRoute::Shared {
                1
            } else {
                rows * nh * 3
            }))?,
            seq_ds: z(sequential(seq_ds_len))?,
            seq_leaves: z(sequential(seq_leaves_len))?,
            db_h: z(sequential(rows * nh * ds))?,
            dc_h: z(sequential(rows * nh * ds))?,
            acc: z(unfused(rows * d_ssm))?,
            ste: z(unfused(rows * d_ssm))?,
            yi: z(unfused(rows * d_ssm))?,
            dcbh: z(chunked(per_chunk * PAIRS))?,
            mm: z(chunked(per_chunk * PAIRS))?,
            ddtx: z(chunked(per_chunk * CHUNK))?,
            qv: z(chunked(per_chunk * CHUNK))?,
            inter: z(chunked(per_chunk * CHUNK))?,
            dot: z(chunked(per_chunk))?,
            da_parts: z(per_chunk)?,
            dbias_parts: z(per_chunk)?,
            d_parts: z(per_chunk * hd)?,
            dw_parts: z(rows * d_ssm)?,
            dv: z(rows * cd)?,
            d_post: typed(rows * dm)?,
            d_pre: z(rows * dm)?,
            d_pair: z(if c.bidirectional { rows * dm } else { 1 })?,
            d_up: typed(rows * 2 * hidden)?,
            d_act: typed(rows * hidden)?,
            axis0: z(rows * dm.max(d_ssm))?,
            d_input: z(rows * input_dim)?,
            d_temp_h: if act == WeightDtype::F32 {
                None
            } else {
                Some(DtypedBuf::zeros(s, rows * dm, act)?)
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
        let trainer = Self {
            graph: None,
            eager_gemm_manifest: None,
            forwarded: false,
            fresh_grads: false,
            window: GradWindow::new(s)?,
            multi_plan: None,
            bias: AdamWBiasFactors::new(s)?,
            varlen: GpuVarlen::default(),
            master: GpuBuffer::from_cpu(s, &layout.flatten(w))?,
            grads: z(layout.total)?,
            adamw: GpuAdamW::new(s, layout.total)?
                .with_lr(lr)
                .with_weight_decay(weight_decay),
            a_neg: z(c.n_layers * nh)?,
            conv_zero_b: z(cd)?,
            state: Mamba2GpuState {
                conv: z(c.n_layers * batch * cd * c.d_conv)?,
                ssm: z(c.n_layers * batch * nh * hd * ds)?,
            },
            acts,
            input: z(rows * input_dim)?,
            temporal: z(rows * dm)?,
            final_in: z(rows * dm)?,
            final_rms: z(rows * stats)?,
            final_out: z(rows * dm)?,
            conv_next: z(batch * cd * c.d_conv)?,
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
            dtype,
            act,
            seq_mode,
            stride,
            dx_tiles,
            seq_bwd,
        };
        trainer.refresh_mirror()?;
        Ok(trainer)
    }

    /// The model's configuration.
    pub fn config(&self) -> &Mamba2Config {
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

    /// Round the master in_proj, out_proj and MLP projection weights into
    /// their half copies.
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
                format_args!("Mamba-2 half weight copy"),
            )?;
        }
        Ok(())
    }

    /// The half copy of a master matrix, or the master itself in the f32
    /// lanes.
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

    fn exec(&self) -> M2Exec<'_> {
        M2Exec {
            ctx: &self.ctx,
            kernels: &self.kernels,
            cfg: &self.cfg,
            act: self.act,
        }
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
        self.step_rows(RowSource::Host(input), RowSource::Host(d_temporal))
    }

    /// [`Self::step`] on the rows the caller wrote on the device, on the
    /// trainer's stream: the input at [`Self::input_ptr`] and d loss / d
    /// the final rows at [`Self::d_temporal_ptr`]. No row crosses to the
    /// host, and a captured graph replays without an upload; the bits are
    /// the host step's.
    pub fn step_on_device(&mut self) -> Result<StepMetrics, String> {
        self.step_rows(RowSource::Device, RowSource::Device)
    }

    fn step_rows(
        &mut self,
        input: RowSource<'_>,
        d_temporal: RowSource<'_>,
    ) -> Result<StepMetrics, String> {
        // A step with a window open adds its micro-batch to the window and
        // applies it, as the split calls would close the window.
        if self.window.is_open() {
            self.accumulate_rows(input, d_temporal)?;
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
                metrics = Some(t.backward_step_with(d_temporal, opts, InputGrad::None)?);
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
        self.accumulate_rows(RowSource::Host(input), RowSource::Host(d_temporal))
    }

    /// [`Self::accumulate`] on the rows the caller wrote on the device
    /// (see [`Self::step_on_device`]).
    pub fn accumulate_on_device(&mut self) -> Result<BackwardMetrics, String> {
        self.accumulate_rows(RowSource::Device, RowSource::Device)
    }

    fn accumulate_rows(
        &mut self,
        input: RowSource<'_>,
        d_temporal: RowSource<'_>,
    ) -> Result<BackwardMetrics, String> {
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
    /// The control channels are the dt columns of every layer's in_proj
    /// (with their in_proj bias entries where the model has the bias),
    /// dt_bias and A_log. A clip of either kind belongs on the applying
    /// call: on an accumulating one it is an error. In f16 the loss scale
    /// holds for the whole window, and an overflow in any of its
    /// micro-batches skips the whole window and backs the scale off.
    pub fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        self.check_window_len("backward d_out", d_temporal.len(), self.cfg.d_model)?;
        self.backward_step_with(RowSource::Host(d_temporal), opts, InputGrad::None)
    }

    /// [`Self::backward_step`] of d loss / d the final rows the caller
    /// wrote at [`Self::d_temporal_ptr`] on the trainer's stream, after
    /// [`Self::forward_on_device`] or [`Self::forward`].
    pub fn backward_step_on_device(
        &mut self,
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        self.backward_step_with(RowSource::Device, opts, InputGrad::None)
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
        self.check_window_len("backward d_out", d_temporal.len(), self.cfg.d_model)?;
        self.check_window_len("backward d_input", d_input.len(), self.input_dim)?;
        self.backward_step_with(RowSource::Host(d_temporal), opts, InputGrad::Host(d_input))
    }

    /// [`Self::backward_step_on_device`] that also leaves d loss / d the
    /// window's input at [`Self::d_input_ptr`], unscaled in f16, for the
    /// caller's layer below on the device.
    pub fn backward_step_input_grad_on_device(
        &mut self,
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        self.backward_step_with(RowSource::Device, opts, InputGrad::Device)
    }

    fn backward_step_with(
        &mut self,
        d_temporal: RowSource<'_>,
        opts: BackwardOpts,
        d_input: InputGrad<'_>,
    ) -> Result<BackwardMetrics, String> {
        check_backward_opts(&opts, "Mamba-2 trainer")?;
        let loss_scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        self.backward_rows(d_temporal, d_input)?;
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
                "Mamba-2 set_step_opts: the captured graph baked the step options in; \
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
        check_backward_opts(&opts, "Mamba-2 trainer")?;
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

    /// Download the carried recurrence: every layer's conv window and SSM
    /// state, which a resume uploads to continue the sequences.
    pub fn recurrent_state(&self) -> Result<Mamba2RecurrentStateBlob, String> {
        let mut conv = self.state.conv.to_cpu(&self.ctx.stream)?;
        conv.iter_mut()
            .step_by(self.cfg.d_conv)
            .for_each(|oldest| *oldest = 0.0);
        Ok(Mamba2RecurrentStateBlob {
            conv,
            ssm: self.state.ssm.to_cpu(&self.ctx.stream)?,
        })
    }

    /// Upload a previously exported recurrence; a blob of another shape is
    /// an error.
    pub fn load_recurrent_state(&mut self, blob: &Mamba2RecurrentStateBlob) -> Result<(), String> {
        self.load_recurrent_parts(&blob.conv, &blob.ssm)
    }

    fn load_recurrent_parts(&mut self, conv: &[f32], ssm: &[f32]) -> Result<(), String> {
        let stream = self.ctx.stream.clone();
        if conv.len() != self.state.conv.len() || ssm.len() != self.state.ssm.len() {
            return Err(format!(
                "recurrent state mismatch: the blob holds {} conv and {} SSM values, this \
                 trainer {} and {}",
                conv.len(),
                ssm.len(),
                self.state.conv.len(),
                self.state.ssm.len()
            ));
        }
        self.state.conv.upload(&stream, conv)?;
        self.state.ssm.upload(&stream, ssm)
    }

    /// Everything this trainer needs beside its weights to resume bit for
    /// bit: the optimizer state, the f16 loss scale and the carried
    /// recurrence. [`TrainerState::save`] writes it to a file.
    pub fn trainer_state(&self) -> Result<TrainerState, String> {
        let rec = self.recurrent_state()?;
        Ok(TrainerState {
            family: ModelFamily::Mamba2,
            optimizer: self.optimizer_state()?,
            scaler: self.scaler_state(),
            recurrent: vec![("conv".into(), rec.conv), ("ssm".into(), rec.ssm)],
        })
    }

    /// Load a state a trainer of this family gave with
    /// [`Self::trainer_state`]. The learning rate is not part of it.
    pub fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String> {
        state.check_family(ModelFamily::Mamba2)?;
        let conv = state.recurrent_tensor("conv")?;
        let ssm = state.recurrent_tensor("ssm")?;
        self.load_optimizer_state(&state.optimizer)?;
        if let Some((scale, tracker)) = state.scaler {
            self.load_scaler_state(scale, tracker)?;
        }
        self.load_recurrent_parts(conv, ssm)
    }

    /// Download the f32 master weights for a checkpoint or the CPU model.
    pub fn snapshot_master(&self) -> Result<Mamba2Weights, String> {
        let flat = self.master.to_cpu(&self.ctx.stream)?;
        Ok(self.layout.unflatten(&flat, &self.template))
    }

    /// The gradients of the last backward, shaped like the weights.
    pub fn grads(&self) -> Result<Mamba2Weights, String> {
        let flat = self.grads.to_cpu(&self.ctx.stream)?;
        // Fields the trainer does not train come out zero, not as weights.
        Ok(self.layout.unflatten(&flat, &self.template.zeros_like()))
    }

    /// Back to the empty conv and SSM states.
    pub fn reset_state(&mut self) -> Result<(), String> {
        self.state.reset(&self.ctx)
    }

    /// Pack several sequences of different lengths into the rows of the
    /// next windows (see [`crate::varlen`]): no state, conv history or
    /// gradient crosses from one into the next. `None` runs each row as
    /// one sequence. The packing holds until replaced, its ids uploaded on
    /// the stream ahead of the windows that read them. A captured graph
    /// replays every packing set while it was on; turning it on or off
    /// needs `drop_graph` and a new capture.
    pub fn set_varlen(&mut self, packing: Option<&Varlen>) -> Result<(), String> {
        self.varlen.check_graph(self.graph.is_some(), packing)?;
        self.varlen
            .set(&self.ctx.stream, packing, (self.batch, self.seq_len))
    }

    fn layer_ptrs(&self, l: usize) -> Result<LayerPtrs, String> {
        let (base, ls) = (self.master.cached_ptr(), &self.layout.layers[l]);
        Ok(LayerPtrs {
            norm: NormParams {
                w: at(base, ls.norm_weight),
                b: opt_at(base, ls.norm_bias).unwrap_or(0),
            },
            in_w: self.gemm_weight(ls.in_proj_w)?,
            in_b: opt_at(base, ls.in_proj_b),
            conv_w: at(base, ls.conv_w),
            conv_b: opt_at(base, ls.conv_b).unwrap_or(self.conv_zero_b.cached_ptr()),
            dt_bias: at(base, ls.dt_bias),
            a_neg: self.a_neg.cached_ptr() + (l * self.cfg.nheads() * 4) as u64,
            d: at(base, ls.d_param),
            gate_w: opt_at(base, ls.gate_norm_weight),
            out_w: self.gemm_weight(ls.out_proj_w)?,
            out_b: opt_at(base, ls.out_proj_b),
        })
    }

    /// Layer `l`'s gated MLP as pointers into the master (its projections as
    /// the GEMMs read them), or None for a model without one.
    fn mlp_ptrs(&self, l: usize) -> Result<Option<MlpPtrs>, String> {
        let Some(ms) = self.layout.mlp.get(l) else {
            return Ok(None);
        };
        let base = self.master.cached_ptr();
        Ok(Some(MlpPtrs {
            norm: NormParams {
                w: at(base, ms.norm_weight),
                b: opt_at(base, ms.norm_bias).unwrap_or(0),
            },
            fc1_w: self.gemm_weight(ms.fc1_w)?,
            fc2_w: self.gemm_weight(ms.fc2_w)?,
            hidden: ms.fc2_w.len / self.cfg.d_model,
        }))
    }

    fn copy(&self, dst: u64, src: u64, n: usize, what: &str) -> Result<(), String> {
        let r = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, n * 4, self.ctx.stream.cu_stream())
        };
        if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("Mamba-2 trainer copy {what}: {r:?}"))
        }
    }

    fn sync(&self, what: &str) -> Result<(), String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("Mamba-2 trainer {what} sync: {e:?}"))
    }

    /// The training forward over one window: `input` is `[batch, seq_len,
    /// input_dim]`, `out` (`[batch, seq_len, d_model]`) receives every
    /// position's final-norm output. The states carry from the last call.
    pub fn forward(&mut self, input: &[f32], out: &mut [f32]) -> Result<(), String> {
        self.check_window_len("forward input", input.len(), self.input_dim)?;
        self.check_window_len("forward output", out.len(), self.cfg.d_model)?;
        self.forwarded = false;
        self.presize_gemm_scratch()?;
        self.forward_unchecked(RowSource::Host(input), Some(out))?;
        self.forwarded = true;
        Ok(())
    }

    /// [`Self::forward`] on the input rows the caller wrote at
    /// [`Self::input_ptr`] on the trainer's stream; the final rows stay
    /// at [`Self::temporal_ptr`]. The bits are the host forward's.
    pub fn forward_on_device(&mut self) -> Result<(), String> {
        self.forwarded = false;
        self.presize_gemm_scratch()?;
        self.forward_unchecked(RowSource::Device, None)?;
        self.forwarded = true;
        Ok(())
    }

    /// The window's input rows on the device, `[batch, seq_len,
    /// input_dim]` f32: the caller writes them on the trainer's stream
    /// before [`Self::forward_on_device`], [`Self::step_on_device`] or
    /// [`Self::accumulate_on_device`].
    pub fn input_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.input.cached_ptr()
    }

    /// The final rows of the last forward on the device, `[batch,
    /// seq_len, d_model]` f32 (the residual stream of a segment built
    /// with `final_norm = false`).
    pub fn temporal_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.final_out.cached_ptr()
    }

    /// d loss / d the final rows on the device, `[batch, seq_len,
    /// d_model]` f32: the caller writes it on the trainer's stream before
    /// a backward, step or micro-batch on the device. In f16 the backward
    /// scales it by the loss scale there, as it scales the host rows.
    pub fn d_temporal_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.bwd.d_rows.cached_ptr()
    }

    /// d loss / d the window's input on the device, `[batch, seq_len,
    /// input_dim]` f32, as [`Self::backward_step_input_grad_on_device`]
    /// leaves it.
    pub fn d_input_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.input_grad_buf().cached_ptr()
    }

    /// The buffer that holds d loss / d the input after a backward: the
    /// input projection's gradient, or the residual stream's for an
    /// identity input.
    fn input_grad_buf(&self) -> &GpuBuffer {
        if self.layout.input_proj_w.len == 0 {
            &self.bwd.d_temp
        } else {
            &self.bwd.d_input
        }
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
    fn forward_unchecked(
        &mut self,
        input: RowSource<'_>,
        out: Option<&mut [f32]>,
    ) -> Result<(), String> {
        let stream = self.ctx.stream.clone();
        if let Some(input) = input.host() {
            self.input.upload(&stream, input)?;
        }
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
        let (dm, nh) = (c.d_model, c.nheads());
        let stream = &self.ctx.stream;
        for (l, ls) in self.layout.layers.iter().enumerate() {
            let a_log = at(self.master.cached_ptr(), ls.a_log);
            let a_neg = self.a_neg.cached_ptr() + (l * nh * 4) as u64;
            let n_i = nh as i32;
            let mut bld = stream.launch_builder(&self.kernels.bwd.a_neg);
            bld.arg(&a_neg);
            bld.arg(&a_log);
            bld.arg(&n_i);
            unsafe { bld.launch(grid_1d(nh)) }.map_err(|e| format!("Mamba-2 A L{l}: {e:?}"))?;
        }
        let base = self.master.cached_ptr();
        if self.layout.input_proj_w.len == 0 {
            self.copy(
                self.temporal.cached_ptr(),
                self.input.cached_ptr(),
                rows * dm,
                "input rows",
            )?;
        } else {
            unsafe {
                gpu_gemm_f32_forward_ptrs(
                    &self.ctx,
                    self.temporal.cached_ptr(),
                    self.input.cached_ptr(),
                    at(base, self.layout.input_proj_w),
                    opt_at(base, self.layout.input_proj_b),
                    (rows, self.input_dim, dm),
                )
            }?;
        }
        let temporal = self.temporal.cached_ptr();
        let conv_per_layer = b * c.conv_dim() * c.d_conv;
        let ssm_per_layer = b * nh * c.headdim * c.d_state;
        let bidi = c.bidirectional;
        let seq_idx = self.varlen.ptr();
        // A bidirectional model reads whole sequences: every window starts
        // from the empty state.
        if bidi {
            zero_bytes(
                &self.ctx,
                self.state.conv.cached_ptr(),
                self.state.conv.len() * 4,
            )?;
            zero_bytes(
                &self.ctx,
                self.state.ssm.cached_ptr(),
                self.state.ssm.len() * 4,
            )?;
        }
        let x = self.exec();
        for (l, a) in self.acts.iter().enumerate() {
            let lp = self.layer_ptrs(l)?;
            // The second layer of a pair runs on the reversed window from the
            // pair's input, which the first layer saved, on: its record holds
            // the reversed rows and its output turns back before the add.
            let reversed = reads_reversed(bidi, l);
            let window = |ptr, elem_bytes| WindowRows {
                ptr,
                elem_bytes,
                batch: b,
                seq_len: t_len,
                width: dm,
            };
            let norm_in = if reversed {
                let pair_in = self.acts[l - 1].x_in.cached_ptr();
                self.copy(a.x_in.cached_ptr(), pair_in, rows * dm, "pair input")?;
                reverse_rows(&self.ctx, window(a.x_in.cached_ptr(), 4), seq_idx)?;
                a.x_in.cached_ptr()
            } else {
                self.copy(a.x_in.cached_ptr(), temporal, rows * dm, "layer input")?;
                temporal
            };
            x.norm_in_proj(
                &lp,
                (norm_in, a.post.ptr(), a.rms.cached_ptr(), a.proj.ptr()),
                rows,
            )?;
            let conv = self.state.conv.cached_ptr() + (l * conv_per_layer * 4) as u64;
            self.copy(a.conv0.cached_ptr(), conv, conv_per_layer, "conv state")?;
            let w = WindowPtrs {
                proj: a.proj.ptr(),
                xbc: a.xbc.ptr(),
                conv,
                conv_next: self.conv_next.cached_ptr(),
                ssm: self.state.ssm.cached_ptr() + (l * ssm_per_layer * 4) as u64,
                dt: a.dt.cached_ptr(),
                acs: a.acs.cached_ptr(),
                cb: a.cb.cached_ptr(),
                states: a.states.cached_ptr(),
                y: a.y.cached_ptr(),
                ckpt: if self.seq_mode {
                    a.ckpt.cached_ptr()
                } else {
                    0
                },
                stride: self.stride,
                seq_idx,
            };
            x.window(l, &lp, &w, (b, t_len, !self.seq_mode))?;
            let out = self.out_rows.ptr();
            x.gate_out(
                &lp,
                (a.proj.ptr(), a.y.cached_ptr(), a.gated.ptr(), out),
                rows,
            )?;
            if reversed {
                reverse_rows(&self.ctx, window(out, self.act.size_bytes()), seq_idx)?;
            }
            x.residual_add(temporal, out, rows)?;
            // A pair's MLPs follow both its mixers.
            for m_l in mlps_after(bidi, l) {
                let (Some(m), Some(mp)) = (&self.acts[m_l].mlp, self.mlp_ptrs(m_l)?) else {
                    continue;
                };
                self.copy(m.x_in.cached_ptr(), temporal, rows * dm, "MLP input")?;
                x.mlp_residual(
                    &mp,
                    (
                        temporal,
                        m.post.ptr(),
                        m.rms.cached_ptr(),
                        m.up.ptr(),
                        m.act.ptr(),
                        self.out_rows.ptr(),
                    ),
                    rows,
                )?;
            }
        }
        // The final norm's backward reads its input; without the norm the
        // rows pass through.
        if self.layout.norm_f_weight.len > 0 {
            self.copy(
                self.final_in.cached_ptr(),
                temporal,
                rows * dm,
                "final rows",
            )?;
        }
        x.block_norm(
            (
                self.final_out.cached_ptr(),
                temporal,
                self.final_rms.cached_ptr(),
            ),
            self.norm_f_params(base),
            (rows, false),
        )
    }

    fn grad(&self, span: Span) -> GradSlice {
        GradSlice::from_raw(at(self.grads.cached_ptr(), span), span.len)
    }

    fn grad_opt(&self, span: Span) -> Option<GradSlice> {
        (span.len > 0).then(|| self.grad(span))
    }

    /// The final norm's parameters in the master at `base`; none for a
    /// backbone built without it.
    fn norm_f_params(&self, base: u64) -> NormParams {
        NormParams {
            w: opt_at(base, self.layout.norm_f_weight).unwrap_or(0),
            b: opt_at(base, self.layout.norm_f_bias).unwrap_or(0),
        }
    }

    /// A block norm's backward over `rows` rows: `dx` = the input gradient,
    /// the scale's gradient accumulated into `g_scale` and a LayerNorm's
    /// bias gradient into `g_bias`; `dy` is in the half type when
    /// `typed_dy`, else f32.
    fn norm_backward(
        &self,
        (dx, dy, x, stats): (u64, u64, u64, u64),
        (norm, g_scale, g_bias): (NormParams, u64, u64),
        (rows, typed_dy): (usize, bool),
    ) -> Result<(), String> {
        let dm = self.cfg.d_model;
        let axis0 = self.bwd.axis0.cached_ptr();
        let dy_dtype = if typed_dy { self.act } else { WeightDtype::F32 };
        block_norm::backward(
            &self.ctx,
            dy_dtype,
            (dx, axis0, dy, x, stats),
            (norm, g_bias),
            (rows, dm),
            (false, 0),
            "Mamba-2 block norm backward",
        )?;
        // No norm, no scale to train.
        if norm.is_none() {
            return Ok(());
        }
        self.reduce_axis0(g_scale, axis0, (rows, dm))
    }

    /// `out[j] += sum_r parts[r][j]`, the shared fixed-tree reduction's
    /// order (so its bits) with a block taking 32 neighbouring columns when
    /// there are enough of them to fill the GPU, else 8.
    fn reduce_axis0(
        &self,
        out: u64,
        parts: u64,
        (rows, dim): (usize, usize),
    ) -> Result<(), String> {
        let width = (rows as u32).next_power_of_two().clamp(32, 256);
        let cw: u32 = if dim >= 4096 { 32 } else { 8 };
        let (rows_i, dim_i) = (rows as i64, dim as i64);
        let (width_i, cw_i) = (width as i32, cw as i32);
        let mut bld = self
            .ctx
            .stream
            .launch_builder(&self.kernels.bwd.reduce_rows);
        bld.arg(&out);
        bld.arg(&parts);
        bld.arg(&rows_i);
        bld.arg(&dim_i);
        bld.arg(&width_i);
        bld.arg(&cw_i);
        let cfg = LaunchConfig {
            grid_dim: ((dim as u32).div_ceil(cw), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: width * cw * 4,
        };
        unsafe { bld.launch(cfg) }
            .map(|_| ())
            .map_err(|e| format!("Mamba-2 column reduction: {e:?}"))
    }

    /// `dst[i] = round(src[i])` into the half type, n values.
    fn round_rows(&self, dst: u64, src: u64, n: usize) -> Result<(), String> {
        let cast = match self.act {
            WeightDtype::Bf16 => &self.ctx.kernels.cast_f32_to_bf16,
            _ => &self.ctx.kernels.cast_f32_to_f16,
        };
        launch_cast(
            &self.ctx.stream,
            cast,
            (dst, 2),
            (src, 4),
            n,
            format_args!("Mamba-2 rounding rows"),
        )
    }

    /// `g[j] += sum_r dy[r][j]` over f32 rows, the Inference column sum.
    fn column_sum_f32(&self, g: u64, dy: u64, (rows, n): (usize, usize)) -> Result<(), String> {
        let (rows_i, n_i) = (rows as i32, n as i32);
        let mut bld = self
            .ctx
            .stream
            .launch_builder(&self.ctx.kernels.colsum_accumulate);
        bld.arg(&g);
        bld.arg(&dy);
        bld.arg(&rows_i);
        bld.arg(&n_i);
        unsafe { bld.launch(grid_colsum(n)) }
            .map(|_| ())
            .map_err(|e| format!("Mamba-2 bias gradient: {e:?}"))
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
        self.check_window_len("backward d_out", d_out.len(), self.cfg.d_model)?;
        if let Some(d) = d_input.as_deref() {
            self.check_window_len("backward d_input", d.len(), self.input_dim)?;
        }
        let d_input = match d_input {
            Some(d) => InputGrad::Host(d),
            None => InputGrad::None,
        };
        self.backward_rows(RowSource::Host(d_out), d_input)
    }

    fn backward_rows(
        &mut self,
        d_out: RowSource<'_>,
        d_input: InputGrad<'_>,
    ) -> Result<(), String> {
        if !self.forwarded {
            return Err("backward needs a completed forward of this window".into());
        }
        self.fresh_grads = false;
        self.backward_unchecked(d_out, d_input)?;
        self.fresh_grads = true;
        Ok(())
    }

    fn backward_unchecked(
        &mut self,
        d_out: RowSource<'_>,
        d_input: InputGrad<'_>,
    ) -> Result<(), String> {
        let stream = self.ctx.stream.clone();
        let scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        self.upload_d_out(d_out, scale)?;
        self.backward_device()?;
        // The input gradient left on the device comes out unscaled there,
        // by the f32 product the host hand-out takes with the same factor.
        if let (InputGrad::Device, Some(_)) = (&d_input, scale) {
            let (grad, unscale) = (self.input_grad_buf().cached_ptr(), self.unscale.ptr());
            let n = self.input_grad_buf().len() as i64;
            let mut bld = stream.launch_builder(&self.ctx.kernels.scale_grads_dev_f32);
            bld.arg(&grad);
            bld.arg(&unscale);
            bld.arg(&n);
            unsafe { bld.launch(crate::gpu::launch::grid_1d_wide(n as usize)) }
                .map_err(|e| format!("unscale of the input gradient: {e:?}"))?;
        }
        self.overflowed = false;
        if scale.is_some() {
            self.sync("backward")?;
            self.overflowed = self.overflow.read(&stream)? != 0;
        }
        self.sync("backward")?;
        if let InputGrad::Host(d_input) = d_input {
            self.input_grad_buf().download(&stream, d_input)?;
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
    fn upload_d_out(&mut self, d_out: RowSource<'_>, scale: Option<f32>) -> Result<(), String> {
        let stream = self.ctx.stream.clone();
        if let Some(d_out) = d_out.host() {
            self.bwd.d_rows.upload(&stream, d_out)?;
        }
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
        let g_base = self.grads.cached_ptr();
        self.norm_backward(
            (
                self.bwd.d_temp.cached_ptr(),
                self.bwd.d_rows.cached_ptr(),
                self.final_in.cached_ptr(),
                self.final_rms.cached_ptr(),
            ),
            (
                self.norm_f_params(base),
                at(g_base, self.layout.norm_f_weight),
                opt_at(g_base, self.layout.norm_f_bias).unwrap_or(0),
            ),
            (rows, false),
        )?;
        if c.bidirectional {
            self.pairs_backward()?;
        } else {
            for l in (0..c.n_layers).rev() {
                if self.acts[l].mlp.is_some() {
                    self.mlp_backward(l)?;
                }
                self.layer_backward(l, self.bwd.d_temp.cached_ptr())?;
            }
        }
        let identity_input = self.layout.input_proj_w.len == 0;
        if !identity_input {
            let (gw, gb) = (
                self.grad(self.layout.input_proj_w),
                self.grad_opt(self.layout.input_proj_b),
            );
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

    /// Whether the last backward's gradients overflowed the f16 loss
    /// scale (its AdamW step will be skipped).
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// A bidirectional model's layers from the top down, a pair at a time:
    /// its MLPs, then both mixers on d loss / d (the pair's output). The
    /// second, which ran on the reversed window, takes that gradient
    /// reversed and adds its norm's input gradient turned back; the first
    /// takes it as it is, held aside in `d_pair`, and adds its own on top:
    /// the pair's input gradient is `(d_out + c_reversed) + c_forward`.
    fn pairs_backward(&mut self) -> Result<(), String> {
        let (b, t_len, dm) = (self.batch, self.seq_len, self.cfg.d_model);
        let rows = b * t_len;
        let seq_idx = self.varlen.ptr();
        let window = |ptr| WindowRows {
            ptr,
            elem_bytes: 4,
            batch: b,
            seq_len: t_len,
            width: dm,
        };
        for l in (0..self.cfg.n_layers).step_by(2).rev() {
            for m in [l + 1, l] {
                if self.acts[m].mlp.is_some() {
                    self.mlp_backward(m)?;
                }
            }
            let d_temp = self.bwd.d_temp.cached_ptr();
            self.copy(
                self.bwd.d_pair.cached_ptr(),
                d_temp,
                rows * dm,
                "pair gradient",
            )?;
            reverse_rows(&self.ctx, window(d_temp), seq_idx)?;
            self.layer_backward(l + 1, d_temp)?;
            reverse_rows(&self.ctx, window(d_temp), seq_idx)?;
            // The first layer reads the held gradient as its own and adds
            // into the gathered one; the two trade places back after it.
            std::mem::swap(&mut self.bwd.d_temp, &mut self.bwd.d_pair);
            let gathered = self.bwd.d_pair.cached_ptr();
            let first = self.layer_backward(l, gathered);
            std::mem::swap(&mut self.bwd.d_temp, &mut self.bwd.d_pair);
            first?;
        }
        Ok(())
    }

    /// One mixer's backward on d loss / d (its output rows) in `d_temp`,
    /// its norm's input gradient added into the rows at `residual`.
    fn layer_backward(&mut self, l: usize, residual: u64) -> Result<(), String> {
        let c = self.cfg;
        let (b, t_len) = (self.batch, self.seq_len);
        let rows = b * t_len;
        let nc = t_len.div_ceil(CHUNK);
        let (dm, ip, cd, di) = (c.d_model, c.in_proj_out_dim(), c.conv_dim(), c.d_inner());
        let (d_mlp, d_ssm, nh, hd, ds, ng) = (
            c.d_mlp(),
            c.ssm_dim(),
            c.nheads(),
            c.headdim,
            c.d_state,
            c.ngroups,
        );
        let ls = self.layout.layers[l];
        let lp = self.layer_ptrs(l)?;
        let g_base = self.grads.cached_ptr();
        let stream = self.ctx.stream.clone();
        let act = self.act;
        let k = &self.kernels.bwd;
        let a = &self.acts[l];
        let s = &self.bwd;
        let seq_idx = self.varlen.ptr();
        // One launch: the kernel, its grid, a name for errors, its arguments.
        macro_rules! go {
            ($f:expr, $cfg:expr, $what:expr, [$($arg:expr),* $(,)?]) => {{
                let mut bld = stream.launch_builder($f);
                $(bld.arg(&$arg);)*
                unsafe { bld.launch($cfg) }
                    .map_err(|e| format!("Mamba-2 backward {} L{l}: {e:?}", $what))?;
            }};
        }
        // One launch split into runs along its y axis, the run's first y
        // block after the arguments and before the tail ones.
        macro_rules! go_y {
            ($f:expr, $cfg:expr, $what:expr, [$($arg:expr),* $(,)?] $(, [$($tail:expr),* $(,)?])?) => {{
                for (y_first, cfg) in y_runs($cfg) {
                    let mut bld = stream.launch_builder($f);
                    $(bld.arg(&$arg);)*
                    bld.arg(&y_first);
                    $($(bld.arg(&$tail);)*)?
                    unsafe { bld.launch(cfg) }
                        .map_err(|e| format!("Mamba-2 backward {} L{l}: {e:?}", $what))?;
                }
            }};
        }
        let cfg3 = |x: usize, y: usize, z: usize, block: u32| LaunchConfig {
            grid_dim: (x as u32, y as u32, z as u32),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (b_i, t_i, nh_i, ng_i, hd_i, ds_i) = (
            b as i32,
            t_len as i32,
            nh as i32,
            ng as i32,
            hd as i32,
            ds as i32,
        );
        let (ld_proj, ld_xbc, ld_out) = (ip as i64, cd as i64, di as i64);
        let xbc_col = 2 * d_mlp + d_ssm;
        let (b_col, c_col) = (d_ssm as i32, (d_ssm + ng * ds) as i32);
        let (z_off, out_col, dt_off) = ((2 * d_mlp) as i32, d_mlp as i32, (xbc_col + cd) as i32);
        let rows_l = rows as i64;
        let per_chunk = b * nc * nh;
        let (dy, d_proj, dxbc, ds_buf) = (
            s.dy.cached_ptr(),
            s.d_proj.ptr(),
            s.dxbc.cached_ptr(),
            s.ds.cached_ptr(),
        );
        let (xbc, proj, y, states, cb, dt, acs) = (
            a.xbc.ptr(),
            a.proj.ptr(),
            a.y.cached_ptr(),
            a.states.cached_ptr(),
            a.cb.cached_ptr(),
            a.dt.cached_ptr(),
            a.acs.cached_ptr(),
        );

        // out_proj: d_gated, and its weight and bias gradients.
        {
            let (gw, gb) = (self.grad(ls.out_proj_w), self.grad_opt(ls.out_proj_b));
            if act == WeightDtype::F32 {
                let s = &mut self.bwd;
                gpu_gemm_bi_backward_grad_raw(
                    &self.ctx,
                    s.d_gated.f32_mut()?,
                    (&gw, gb.as_ref()),
                    &s.d_temp,
                    self.acts[l].gated.f32()?,
                    lp.out_w,
                    (rows, di, dm),
                )?;
            } else {
                let d_temp = self.bwd.d_temp.cached_ptr();
                let dth = self
                    .bwd
                    .d_temp_h
                    .as_ref()
                    .ok_or("half lane without its rounded residual gradient")?
                    .cached_ptr();
                self.round_rows(dth, d_temp, rows * dm)?;
                let t = |ptr| TypedPtr { ptr, dtype: act };
                gpu_gemm_bi_backward_dw_grad_typed(
                    &self.ctx,
                    &gw,
                    t(dth),
                    t(self.acts[l].gated.ptr()),
                    rows,
                    di,
                    dm,
                )?;
                gpu_gemm_ex_backward_dx_typed(
                    &self.ctx,
                    t(self.bwd.d_gated.ptr()),
                    t(dth),
                    t(lp.out_w),
                    rows,
                    di,
                    dm,
                )?;
                if let Some(gb) = gb {
                    self.column_sum_f32(gb.ptr(), d_temp, (rows, dm))?;
                }
            }
        }
        let s = &self.bwd;
        let d_gated = s.d_gated.ptr();

        // The gate (with or without the norm) and the inner MLP gate.
        if let Some(w) = lp.gate_w {
            let dw_parts = s.dw_parts.cached_ptr();
            let (eps, nbg) = (c.gated_norm_eps, i32::from(c.norm_before_gate));
            // The group staged in shared memory when it fits, else the
            // streamed kernel with the same bits.
            let staged = gated_norm_bwd_smem_bytes(d_ssm, ng, hd);
            let fits = staged <= self.kernels.smem_optin;
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, ng as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: if fits { staged } else { 128 * 4 } as u32,
            };
            let kernel = if fits {
                &k.gated_norm
            } else {
                &k.gated_norm_stream
            };
            let d_ssm_i = d_ssm as i32;
            go_y!(
                kernel.get(act),
                cfg,
                "gated norm",
                [
                    dy, d_proj, ld_proj, z_off, dw_parts, d_gated, ld_out, out_col, y, proj, w,
                    rows_l, d_ssm_i, ng_i, hd_i, eps, nbg
                ]
            );
            self.reduce_axis0(at(g_base, ls.gate_norm_weight), dw_parts, (rows, d_ssm))?;
        } else {
            let d_ssm_i = d_ssm as i32;
            go!(
                k.gate_only.get(act),
                grid_rows(rows, d_ssm),
                "gate",
                [
                    dy, d_proj, ld_proj, z_off, d_gated, ld_out, out_col, y, proj, rows_l, d_ssm_i
                ]
            );
        }
        if d_mlp > 0 {
            let d_mlp_i = d_mlp as i32;
            go!(
                k.mlp_gate.get(act),
                grid_rows(rows, d_mlp),
                "inner MLP",
                [d_proj, ld_proj, d_gated, ld_out, proj, rows_l, d_mlp_i]
            );
        }

        // D.
        {
            let parts = s.d_parts.cached_ptr();
            go_y!(
                k.d_parts.get(act),
                cfg3(b * nc, nh, 1, 128),
                "D parts",
                [parts, dy, xbc, ld_xbc, b_i, t_i, nh_i, hd_i]
            );
            let g_d = at(g_base, ls.d_param);
            let count = (b * nc) as i64;
            let per_channel = i32::from(c.d_has_hdim);
            let n = if c.d_has_hdim { nh * hd } else { nh };
            go!(
                &k.d_reduce,
                grid_1d(n),
                "D",
                [g_d, parts, count, nh_i, hd_i, per_channel]
            );
        }

        // The scan backward: the chunked path, or the sequential one.
        if !self.seq_mode {
            let pn = hd * ds;
            let (dh_cols, dh_pt, dh_smem) = chunk_state_geom(hd, ds, self.kernels.smem_optin);
            let dh_pt_i = dh_pt as i32;
            go_y!(
                k.dhloc.get(act),
                LaunchConfig {
                    grid_dim: ((b * nc) as u32, nh as u32, ds.div_ceil(dh_cols) as u32),
                    block_dim: ((32 * dh_pt / 4).min(256) as u32, 1, 1),
                    shared_mem_bytes: dh_smem as u32,
                },
                "dH",
                [
                    ds_buf, dy, xbc, ld_xbc, c_col, acs, b_i, t_i, nh_i, ng_i, hd_i, ds_i, dh_pt_i
                ],
                [seq_idx]
            );
            go_y!(
                &k.state_passing,
                cfg3(b * nh, pn.div_ceil(256), 1, 256),
                "state passing",
                [ds_buf, acs, b_i, t_i, nh_i, hd_i, ds_i],
                [seq_idx]
            );
            let dot = s.dot.cached_ptr();
            {
                let count = per_chunk as i64;
                let pn_i = pn as i32;
                go!(
                    &k.chunk_dot,
                    cfg3(per_chunk, 1, 1, 256),
                    "chunk dot",
                    [dot, states, ds_buf, count, pn_i, t_i, nh_i, seq_idx]
                );
            }
            let (ddtx, qv, inter) = (s.ddtx.cached_ptr(), s.qv.cached_ptr(), s.inter.cached_ptr());
            let d = lp.d;
            let per_channel = i32::from(c.d_has_hdim);
            if let Some((nk, smem)) = self.dx_tiles {
                let nk_i = nk as i32;
                go_y!(
                    k.dx_tiles.get(act),
                    LaunchConfig {
                        grid_dim: ((b * nc) as u32, nh as u32, 1),
                        block_dim: ((16 * hd.div_ceil(4)) as u32, 1, 1),
                        shared_mem_bytes: smem as u32,
                    },
                    "dx",
                    [
                        dxbc,
                        ddtx,
                        qv,
                        inter,
                        dy,
                        xbc,
                        ld_xbc,
                        b_col,
                        c_col,
                        states,
                        ds_buf,
                        cb,
                        dt,
                        acs,
                        d,
                        per_channel,
                        t_i,
                        nh_i,
                        ng_i,
                        hd_i,
                        ds_i,
                        nk_i
                    ],
                    [seq_idx]
                );
            } else {
                let (acc, ste, yi) = (s.acc.cached_ptr(), s.ste.cached_ptr(), s.yi.cached_ptr());
                go_y!(
                    k.dx_groups.get(act),
                    LaunchConfig {
                        grid_dim: (
                            (b * nc) as u32,
                            nh as u32,
                            hd.div_ceil(DX_GROUP_CHANNELS) as u32,
                        ),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: DX_GROUPS_SMEM_BYTES as u32,
                    },
                    "dx",
                    [
                        dxbc,
                        acc,
                        ste,
                        yi,
                        dy,
                        xbc,
                        ld_xbc,
                        b_col,
                        c_col,
                        states,
                        ds_buf,
                        cb,
                        dt,
                        acs,
                        d,
                        per_channel,
                        t_i,
                        nh_i,
                        ng_i,
                        hd_i,
                        ds_i
                    ],
                    [seq_idx]
                );
                go_y!(
                    k.rowsums.get(act),
                    cfg3(b * nc, nh, 1, CHUNK as u32),
                    "token sums",
                    [
                        ddtx, qv, inter, acc, ste, yi, dy, xbc, ld_xbc, dt, b_i, t_i, nh_i, hd_i
                    ],
                    [seq_idx]
                );
            }
            let (dcbh, mm) = (s.dcbh.cached_ptr(), s.mm.cached_ptr());
            go_y!(
                k.pairs.get(act),
                LaunchConfig {
                    grid_dim: ((b * nc) as u32, nh as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: PAIRS_TILES_SMEM_BYTES as u32,
                },
                "pairs",
                [
                    dcbh, mm, dy, xbc, ld_xbc, cb, dt, acs, b_i, t_i, nh_i, ng_i, hd_i
                ],
                [seq_idx]
            );
            let tiles = LaunchConfig {
                grid_dim: ((b * nc) as u32, ng as u32, ds.div_ceil(32) as u32),
                block_dim: (512, 1, 1),
                shared_mem_bytes: BC_TILES_SMEM_BYTES as u32,
            };
            go_y!(
                k.db_tiles.get(act),
                tiles,
                "dB",
                [
                    dxbc, dy, xbc, ld_xbc, b_col, c_col, ds_buf, dcbh, dt, acs, t_i, nh_i, ng_i,
                    hd_i, ds_i
                ],
                [seq_idx]
            );
            go_y!(
                k.dc_tiles.get(act),
                tiles,
                "dC",
                [
                    dxbc, dy, xbc, ld_xbc, b_col, c_col, states, dcbh, dt, acs, t_i, nh_i, ng_i,
                    hd_i, ds_i
                ],
                [seq_idx]
            );
            {
                let (da_parts, dbias_parts) = (s.da_parts.cached_ptr(), s.dbias_parts.cached_ptr());
                let (dt_bias, a_neg) = (lp.dt_bias, lp.a_neg);
                let (lo, hi) = c.dt_limit;
                let has_limit = i32::from(c.dt_limit != (0.0, f32::INFINITY));
                go_y!(
                    k.combine.get(act),
                    cfg3(b * nc, nh, 1, CHUNK as u32),
                    "decay gradients",
                    [
                        d_proj,
                        ld_proj,
                        dt_off,
                        da_parts,
                        dbias_parts,
                        mm,
                        qv,
                        inter,
                        ddtx,
                        dot,
                        dt,
                        acs,
                        proj,
                        dt_bias,
                        a_neg,
                        b_i,
                        t_i,
                        nh_i,
                        has_limit,
                        lo,
                        hi
                    ]
                );
                let (g_dt_bias, g_a_log) = (at(g_base, ls.dt_bias), at(g_base, ls.a_log));
                let count = (b * nc) as i64;
                go!(
                    &k.head_grads,
                    grid_1d(nh),
                    "head gradients",
                    [
                        g_dt_bias,
                        g_a_log,
                        da_parts,
                        dbias_parts,
                        a_neg,
                        count,
                        nh_i
                    ]
                );
            }
        } else {
            let (db_h, dc_h, rec, ckpt) = (
                s.db_h.cached_ptr(),
                s.dc_h.cached_ptr(),
                s.rec.cached_ptr(),
                a.ckpt.cached_ptr(),
            );
            let (da_parts, dbias_parts) = (s.da_parts.cached_ptr(), s.dbias_parts.cached_ptr());
            let (dt_bias, a_neg, d) = (lp.dt_bias, lp.a_neg, lp.d);
            let (lo, hi) = c.dt_limit;
            let limited = i32::from(c.dt_limit != (0.0, f32::INFINITY));
            let flags: i32 = i32::from(c.d_has_hdim) | (limited << 2);
            let (stride_i, cd_i) = (self.stride as i32, cd as i32);
            let scal = s.seq_scal.cached_ptr();
            if self.seq_bwd != SeqBwdRoute::Shared {
                go!(
                    k.seq_scalars.get(act),
                    grid_rows(rows, nh),
                    "sequential step scalars",
                    [
                        scal, xbc, proj, ld_proj, dt_off, dt_bias, a_neg, rows_l, nh_i, ng_i, hd_i,
                        ds_i, cd_i, flags, lo, hi
                    ]
                );
            }
            let block = |threads: usize, smem: usize| LaunchConfig {
                grid_dim: (b as u32, nh as u32, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            match self.seq_bwd {
                SeqBwdRoute::Tiles => {
                    go_y!(
                        k.seq_bwd_tiles.get(act),
                        block(seq_bwd_tiles_threads(hd), seq_bwd_tiles_smem_bytes(hd, ds)),
                        "sequential scan",
                        [
                            dxbc,
                            d_proj,
                            ld_proj,
                            dt_off,
                            db_h,
                            dc_h,
                            da_parts,
                            dbias_parts,
                            rec,
                            dy,
                            xbc,
                            proj,
                            ckpt,
                            scal,
                            dt_bias,
                            a_neg,
                            d,
                            t_i,
                            stride_i,
                            nh_i,
                            ng_i,
                            hd_i,
                            ds_i,
                            cd_i,
                            flags,
                            lo,
                            hi
                        ],
                        [seq_idx]
                    );
                }
                SeqBwdRoute::Shared => {
                    let threads = hd.next_multiple_of(32);
                    go_y!(
                        k.seq_bwd.get(act),
                        block(threads, seq_bwd_smem_bytes(hd, ds, threads)),
                        "sequential scan",
                        [
                            dxbc,
                            d_proj,
                            ld_proj,
                            dt_off,
                            db_h,
                            dc_h,
                            da_parts,
                            dbias_parts,
                            rec,
                            dy,
                            xbc,
                            proj,
                            ckpt,
                            dt_bias,
                            a_neg,
                            d,
                            t_i,
                            stride_i,
                            nh_i,
                            ng_i,
                            hd_i,
                            ds_i,
                            cd_i,
                            flags,
                            lo,
                            hi
                        ],
                        [seq_idx]
                    );
                }
                SeqBwdRoute::Global(geom) => {
                    // A null scratch tells the kernel dS sits in shared
                    // memory, or that the head fits the block.
                    let ds_glob = if geom.shared_ds {
                        0u64
                    } else {
                        s.seq_ds.cached_ptr()
                    };
                    let leaves = if geom.wide {
                        s.seq_leaves.cached_ptr()
                    } else {
                        0u64
                    };
                    go_y!(
                        k.seq_bwd_global.get(act),
                        block(geom.threads, geom.smem),
                        "sequential scan",
                        [
                            dxbc,
                            d_proj,
                            ld_proj,
                            dt_off,
                            db_h,
                            dc_h,
                            da_parts,
                            dbias_parts,
                            rec,
                            ds_glob,
                            leaves,
                            dy,
                            xbc,
                            proj,
                            ckpt,
                            scal,
                            dt_bias,
                            a_neg,
                            d,
                            t_i,
                            stride_i,
                            nh_i,
                            ng_i,
                            hd_i,
                            ds_i,
                            cd_i,
                            flags,
                            lo,
                            hi
                        ],
                        [seq_idx]
                    );
                }
            }
            let bc = grid_rows(rows, ds);
            go_y!(
                &k.seq_bc_groups,
                LaunchConfig {
                    grid_dim: (bc.grid_dim.0, 2 * ng as u32, 1),
                    ..bc
                },
                "B and C groups",
                [
                    dxbc, db_h, dc_h, rows_l, nh_i, ng_i, ds_i, ld_xbc, b_col, c_col
                ]
            );
            let (g_dt_bias, g_a_log) = (at(g_base, ls.dt_bias), at(g_base, ls.a_log));
            let count = b as i64;
            go!(
                &k.head_grads,
                grid_1d(nh),
                "head gradients",
                [
                    g_dt_bias,
                    g_a_log,
                    da_parts,
                    dbias_parts,
                    a_neg,
                    count,
                    nh_i
                ]
            );
        }

        // The conv.
        {
            let dv = s.dv.cached_ptr();
            let conv0 = a.conv0.cached_ptr();
            let (cw, cbias) = (lp.conv_w, lp.conv_b);
            let (xbc_col_i, cd_i, dconv_i) = (xbc_col as i32, cd as i32, c.d_conv as i32);
            let steps = grid_steps(b, t_len, cd)?;
            go_y!(
                k.conv_dv.get(act),
                steps,
                "conv dv",
                [
                    dv, dxbc, conv0, proj, ld_proj, xbc_col_i, cw, cbias, b_i, t_i, cd_i, dconv_i
                ],
                [seq_idx]
            );
            go_y!(
                k.conv_dx.get(act),
                steps,
                "conv dx",
                [d_proj, ld_proj, xbc_col_i, dv, cw, b_i, t_i, cd_i, dconv_i],
                [seq_idx]
            );
            let g_w = at(g_base, ls.conv_w);
            let g_b = opt_at(g_base, ls.conv_b).unwrap_or(0);
            // 32 channels a block with a warp per tap and one for the bias,
            // 128 rows of dv and the inputs (plus the d_conv - 1 rows before)
            // staged at a time. Past 31 taps the taps spread over blocks of
            // 32 warps, each staging the same rows.
            let warps = (c.d_conv + 1).min(32);
            let dw_cfg = LaunchConfig {
                grid_dim: (
                    cd.div_ceil(32) as u32,
                    (c.d_conv + 1).div_ceil(warps) as u32,
                    1,
                ),
                block_dim: (32, warps as u32, 1),
                shared_mem_bytes: ((128 + 128 + c.d_conv - 1) * 32 * 4) as u32,
            };
            go!(
                k.conv_dw.get(act),
                dw_cfg,
                "conv weights",
                [
                    g_w, g_b, dv, conv0, proj, ld_proj, xbc_col_i, b_i, t_i, cd_i, dconv_i, seq_idx
                ]
            );
        }

        // in_proj, then the block norm and the residual.
        {
            let (gw, gb) = (self.grad(ls.in_proj_w), self.grad_opt(ls.in_proj_b));
            if act == WeightDtype::F32 {
                let s = &mut self.bwd;
                let (d_post, d_proj) = (&mut s.d_post, &s.d_proj);
                gpu_gemm_bi_backward_grad_raw(
                    &self.ctx,
                    d_post.f32_mut()?,
                    (&gw, gb.as_ref()),
                    d_proj.f32()?,
                    self.acts[l].post.f32()?,
                    lp.in_w,
                    (rows, dm, ip),
                )?;
            } else {
                let t = |ptr| TypedPtr { ptr, dtype: act };
                let d_proj = self.bwd.d_proj.ptr();
                gpu_gemm_bi_backward_dw_grad_typed(
                    &self.ctx,
                    &gw,
                    t(d_proj),
                    t(self.acts[l].post.ptr()),
                    rows,
                    dm,
                    ip,
                )?;
                gpu_gemm_ex_backward_dx_typed(
                    &self.ctx,
                    t(self.bwd.d_post.ptr()),
                    t(d_proj),
                    t(lp.in_w),
                    rows,
                    dm,
                    ip,
                )?;
                if let Some(gb) = gb {
                    let (g, rows_l, n) = (gb.ptr(), rows as i64, ip as i32);
                    let mut bld = stream.launch_builder(self.kernels.bwd.bias_grad.get(act));
                    bld.arg(&g);
                    bld.arg(&d_proj);
                    bld.arg(&rows_l);
                    bld.arg(&n);
                    unsafe { bld.launch(grid_1d(ip)) }
                        .map_err(|e| format!("Mamba-2 in_proj bias gradient L{l}: {e:?}"))?;
                }
            }
        }
        let a = &self.acts[l];
        self.norm_backward(
            (
                self.bwd.d_pre.cached_ptr(),
                self.bwd.d_post.ptr(),
                a.x_in.cached_ptr(),
                a.rms.cached_ptr(),
            ),
            (
                lp.norm,
                at(g_base, ls.norm_weight),
                opt_at(g_base, ls.norm_bias).unwrap_or(0),
            ),
            (rows, act != WeightDtype::F32),
        )?;
        self.add_residual_grad(rows, l, residual)
    }

    /// `into += d_pre` over the window's rows: what a residual branch's
    /// input gradient adds to the stream's (`d_temp`, or a bidirectional
    /// pair's gathered gradient).
    fn add_residual_grad(&self, rows: usize, l: usize, into: u64) -> Result<(), String> {
        crate::gpu::launch::launch_elems(
            &self.ctx.stream,
            &self.ctx.kernels.vec_add_inplace,
            &[into, self.bwd.d_pre.cached_ptr()],
            rows * self.cfg.d_model,
            format_args!("Mamba-2 backward residual L{l}"),
        )
    }

    /// The gated MLP's backward in layer `l`: from the residual gradient,
    /// the gate rows' gradient and the down projection's weight gradient,
    /// the gate's, the normed rows' gradient and the up projection's weight
    /// gradient, then the MLP norm's, whose input gradient joins the
    /// residual gradient.
    fn mlp_backward(&mut self, l: usize) -> Result<(), String> {
        let rows = self.batch * self.seq_len;
        let dm = self.cfg.d_model;
        let act = self.act;
        let ms = self.layout.mlp[l];
        let hidden = ms.fc2_w.len / dm;
        let (base, g_base) = (self.master.cached_ptr(), self.grads.cached_ptr());
        let (w1, w2) = (self.gemm_weight(ms.fc1_w)?, self.gemm_weight(ms.fc2_w)?);
        let (gw1, gw2) = (self.grad(ms.fc1_w), self.grad(ms.fc2_w));
        let t = |ptr| TypedPtr { ptr, dtype: act };
        let no_mlp = "a layer without an MLP in the MLP backward";
        if act == WeightDtype::F32 {
            let m = self.acts[l].mlp.as_ref().ok_or(no_mlp)?;
            let s = &mut self.bwd;
            gpu_gemm_bi_backward_grad_raw(
                &self.ctx,
                s.d_act.f32_mut()?,
                (&gw2, None),
                &s.d_temp,
                m.act.f32()?,
                w2,
                (rows, hidden, dm),
            )?;
        } else {
            let d_temp = self.bwd.d_temp.cached_ptr();
            let dth = self
                .bwd
                .d_temp_h
                .as_ref()
                .ok_or("half lane without its rounded residual gradient")?
                .cached_ptr();
            self.round_rows(dth, d_temp, rows * dm)?;
            let act_rows = self.acts[l].mlp.as_ref().ok_or(no_mlp)?.act.ptr();
            gpu_gemm_bi_backward_dw_grad_typed(
                &self.ctx,
                &gw2,
                t(dth),
                t(act_rows),
                rows,
                hidden,
                dm,
            )?;
            gpu_gemm_ex_backward_dx_typed(
                &self.ctx,
                t(self.bwd.d_act.ptr()),
                t(dth),
                t(w2),
                rows,
                hidden,
                dm,
            )?;
        }
        let up = self.acts[l].mlp.as_ref().ok_or(no_mlp)?.up.ptr();
        let (d_up, d_act) = (self.bwd.d_up.ptr(), self.bwd.d_act.ptr());
        let (rows_i, hidden_i) = (rows as i32, hidden as i32);
        let mut bld = self
            .ctx
            .stream
            .launch_builder(self.kernels.bwd.gated_mlp.get(act));
        bld.arg(&d_up);
        bld.arg(&d_act);
        bld.arg(&up);
        bld.arg(&rows_i);
        bld.arg(&hidden_i);
        unsafe { bld.launch(grid_rows(rows, hidden)) }
            .map_err(|e| format!("Mamba-2 MLP gate backward L{l}: {e:?}"))?;
        if act == WeightDtype::F32 {
            let m = self.acts[l].mlp.as_ref().ok_or(no_mlp)?;
            let s = &mut self.bwd;
            gpu_gemm_bi_backward_grad_raw(
                &self.ctx,
                s.d_post.f32_mut()?,
                (&gw1, None),
                s.d_up.f32()?,
                m.post.f32()?,
                w1,
                (rows, dm, 2 * hidden),
            )?;
        } else {
            let post = self.acts[l].mlp.as_ref().ok_or(no_mlp)?.post.ptr();
            gpu_gemm_bi_backward_dw_grad_typed(
                &self.ctx,
                &gw1,
                t(d_up),
                t(post),
                rows,
                dm,
                2 * hidden,
            )?;
            gpu_gemm_ex_backward_dx_typed(
                &self.ctx,
                t(self.bwd.d_post.ptr()),
                t(d_up),
                t(w1),
                rows,
                dm,
                2 * hidden,
            )?;
        }
        let m = self.acts[l].mlp.as_ref().ok_or(no_mlp)?;
        self.norm_backward(
            (
                self.bwd.d_pre.cached_ptr(),
                self.bwd.d_post.ptr(),
                m.x_in.cached_ptr(),
                m.rms.cached_ptr(),
            ),
            (
                NormParams {
                    w: at(base, ms.norm_weight),
                    b: opt_at(base, ms.norm_bias).unwrap_or(0),
                },
                at(g_base, ms.norm_weight),
                opt_at(g_base, ms.norm_bias).unwrap_or(0),
            ),
            (rows, act != WeightDtype::F32),
        )?;
        self.add_residual_grad(rows, l, self.bwd.d_temp.cached_ptr())
    }

    /// One AdamW step on the gradients of the last backward. Under the
    /// optimizer's reference no-decay groups only the matrices (the input
    /// projection, in_proj, the conv and out_proj weights, the MLP's
    /// projections) decay. Returns
    /// false when the f16 gradients overflowed: the step is skipped and the
    /// loss scale backs off.
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
