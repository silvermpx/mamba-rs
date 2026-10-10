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

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::inference::{MimoChunkedPtrs, MimoExec, launch_chunked_scan};
use super::kernels::{
    CHUNK, ChunkedGeom, Mamba3MimoKernels, chunked_geom, scan_bwd_record_floats,
    scan_bwd_smem_bytes, scan_cfg,
};
use crate::config::ScanMode;
use crate::mamba_ssm::gpu::adamw::GpuAdamW;
use crate::mamba_ssm::gpu::blas::{
    TypedPtr, gpu_gemm_bi_backward_dw_grad_typed, gpu_gemm_bi_backward_grad_raw,
    gpu_gemm_ex_backward_dx_typed, gpu_gemm_f32_forward_ptrs, gpu_gemm_typed_forward_raw,
};
use crate::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer, GradSlice};
use crate::mamba_ssm::gpu::context::{GemmRole, GpuCtx};
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm, validate_rows_capacity};
use crate::mamba_ssm::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, UnscaleFactor, check_inf_nan_gpu, scale_grads_skip_gpu,
};
use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::weights::Mamba3MimoWeights;

/// A tensor's place in the flat buffers, in f32 elements.
#[derive(Clone, Copy, Debug, Default)]
struct Span {
    off: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct LayerSpans {
    norm_weight: Span,
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
}

#[derive(Clone, Copy, Debug, Default)]
struct MlpSpans {
    norm_weight: Span,
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
            })
            .collect();
        let mlp = w
            .mlp
            .iter()
            .map(|m| MlpSpans {
                norm_weight: take(m.norm_weight.len()),
                fc1_w: take(m.fc1_w.len()),
                fc2_w: take(m.fc2_w.len()),
            })
            .collect();
        let norm_f_weight = take(w.norm_f_weight.len());
        Self {
            input_proj_w,
            input_proj_b,
            layers,
            mlp,
            norm_f_weight,
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
            ]);
        }
        for (m, ms) in w.mlp.iter().zip(&self.mlp) {
            out.extend([
                (m.norm_weight.as_slice(), ms.norm_weight, false),
                (m.fc1_w.as_slice(), ms.fc1_w, true),
                (m.fc2_w.as_slice(), ms.fc2_w, true),
            ]);
        }
        out.push((w.norm_f_weight.as_slice(), self.norm_f_weight, false));
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
        }
        for (m, ms) in w.mlp.iter_mut().zip(&self.mlp) {
            read(&mut m.norm_weight, ms.norm_weight);
            read(&mut m.fc1_w, ms.fc1_w);
            read(&mut m.fc2_w, ms.fc2_w);
        }
        read(&mut w.norm_f_weight, self.norm_f_weight);
        w
    }
}

fn at(base: u64, span: Span) -> u64 {
    base + (span.off * 4) as u64
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
}

/// The MIMO trainer: forward, backward and AdamW for windows of `seq_len`
/// tokens of `batch` sequences.
pub struct Mamba3MimoGpuTrainer {
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
    unscale: UnscaleFactor,
    overflowed: bool,
    /// Set by a completed forward, cleared when one starts.
    forwarded: bool,
    /// Set by a completed backward, consumed by the next optimizer step.
    fresh_grads: bool,
}

impl Mamba3MimoGpuTrainer {
    /// A trainer in `dtype` (f32, tf32, bf16 or f16) for `batch` sequences
    /// of `seq_len` tokens starting from `w`, checkpointing the scan every
    /// `stride` tokens, with AdamW at its defaults (see
    /// [`Self::adamw_mut`]).
    pub fn new(
        device: &GpuDevice,
        w: &Mamba3MimoWeights,
        (cfg, input_dim, batch, seq_len): (Mamba3MimoConfig, usize, usize, usize),
        (dtype, stride): (WeightDtype, usize),
    ) -> Result<Self, String> {
        w.validate(&cfg, input_dim)?;
        if cfg.is_outproj_norm {
            return Err("the gated output norm does not train on the GPU yet".into());
        }
        if batch == 0 || seq_len == 0 {
            return Err("a training window needs at least one sequence and one token".into());
        }
        if cfg.d_model < 32 {
            return Err(format!(
                "the GPU row norm reduces whole warps; d_model {} is below 32",
                cfg.d_model
            ));
        }
        validate_rows_capacity(
            batch * seq_len,
            [
                input_dim,
                cfg.in_proj_out_dim(),
                cfg.mimo_rank * cfg.d_inner(),
                2 * cfg.mlp_hidden(),
                cfg.d_model,
            ]
            .into_iter()
            .max()
            .unwrap_or(0),
        )?;
        let stride = stride.clamp(1, seq_len);
        let (hd, ds, rank, na) = (
            cfg.headdim,
            cfg.d_state,
            cfg.mimo_rank,
            cfg.num_rope_angles(),
        );
        let threads = hd.next_multiple_of(32);
        if threads > 1024 || na > 256 {
            return Err(format!(
                "the GPU trainer serves heads of up to 1024 channels and 256 rotation angles, \
                 got {hd} and {na}"
            ));
        }
        let ctx = GpuCtx::new_from_env_with_state_cap_and_role(device, 64, GemmRole::triad(dtype))?;
        let act = match dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => WeightDtype::F32,
            half => half,
        };
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = Mamba3MimoKernels::compile(device.context(), arch)?;
        let optin = kernels.smem_optin;
        let fitted = chunked_geom((hd, ds, rank), optin, true);
        let chunked = if cfg.train_use_parallel_scan() {
            fitted
        } else {
            None
        };
        if chunked.is_none() {
            if cfg.scan_mode == ScanMode::Parallel {
                return Err(format!(
                    "the chunked training scan of a rank-{rank} head of {hd} channels and {ds} \
                     states does not fit this GPU's {optin} bytes of shared memory; \
                     ScanMode::Auto trains it by the recurrence"
                ));
            }
            let smem = scan_bwd_smem_bytes((hd, ds, rank, na), threads);
            if smem > optin {
                return Err(format!(
                    "a head of {hd} channels and {ds} states needs {smem} bytes of shared \
                     memory for the backward; this GPU serves {optin}"
                ));
            }
        }
        let layout = Layout::of(w);
        let s = &ctx.stream;
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
        let nseg = seq_len.div_ceil(stride);
        let nc = seq_len.div_ceil(CHUNK);
        // Each path's buffers; the other path's are left one element long.
        let seq_n = |n: usize| if chunked.is_some() { 1 } else { n };
        let ch_n = |n: usize| if chunked.is_some() { n } else { 1 };
        let state_n = batch * nc * nh * hd * ds;
        let acts = (0..c.n_layers)
            .map(|l| -> Result<LayerActs, String> {
                Ok(LayerActs {
                    x_in: z(rows * dm)?,
                    rms: z(rows)?,
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
                    y: z(ch_n(rows * nh * rank * hd))?,
                    k0: z(ch_n(batch * nh * rank * ds))?,
                    v0: z(ch_n(batch * nh * hd))?,
                    mlp: if l < w.mlp.len() {
                        Some(MlpActs {
                            x_in: z(rows * dm)?,
                            rms: z(rows)?,
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
                dy: z(ch_n(rows * nh * rank * hd))?,
                du: z(ch_n(rows * nh * rank * hd))?,
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
                gphi: z(ch_n(rows * nh * rank * hd))?,
                gz: z(ch_n(rows * nh * rank * hd))?,
                gpsi: z(ch_n(rows * nh * rank * hd))?,
                gd: z(ch_n(rows * nh * hd))?,
                gdtb: z(ch_n(rows * nh))?,
                tmp: z(ch_n(nh * rank * hd))?,
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
        let mut trainer = Self {
            forwarded: false,
            fresh_grads: false,
            master: GpuBuffer::from_cpu(s, &layout.flatten(w))?,
            grads: z(layout.total)?,
            adamw: GpuAdamW::new(s, layout.total)?,
            state_ssm: per_layer(hd * ds)?,
            state_k: per_layer(rank * ds)?,
            state_v: per_layer(hd)?,
            state_ang: per_layer(na)?,
            acts,
            input: z(rows * input_dim)?,
            temporal: z(rows * dm)?,
            final_in: z(rows * dm)?,
            final_rms: z(rows)?,
            final_out: z(rows * dm)?,
            out_rows: typed(rows * dm)?,
            bwd,
            mirror,
            mirror_pairs,
            scaler: (dtype == WeightDtype::F16).then(DynamicLossScaler::new),
            overflow: OverflowFlag::new(s)?,
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
            contrib,
            dtype,
            act,
        };
        trainer.refresh_mirror()?;
        Ok(trainer)
    }

    /// The storage dtype this trainer runs.
    /// The model's configuration.
    pub fn cfg(&self) -> &Mamba3MimoConfig {
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

    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    /// The optimizer, to set its learning rate, betas, epsilon and decay.
    pub fn adamw_mut(&mut self) -> &mut GpuAdamW {
        &mut self.adamw
    }

    /// The f16 loss scale and its growth counter; `None` in the other lanes.
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        self.scaler.as_ref().map(DynamicLossScaler::state)
    }

    /// Restore the f16 loss scale and its growth counter; no effect in the
    /// other lanes.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) {
        if let Some(scaler) = self.scaler.as_mut() {
            scaler.load_state(scale, growth_tracker);
        }
    }

    /// Whether the last backward's gradients overflowed the f16 loss scale
    /// (its AdamW step will be skipped).
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// The current weights.
    pub fn weights(&self) -> Result<Mamba3MimoWeights, String> {
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

    fn refresh_mirror(&mut self) -> Result<(), String> {
        let Some(mirror) = &self.mirror else {
            return Ok(());
        };
        let cast = match self.act {
            WeightDtype::Bf16 => &self.ctx.kernels.cast_f32_to_bf16,
            _ => &self.ctx.kernels.cast_f32_to_f16,
        };
        let (base, copy) = (self.master.cached_ptr(), mirror.cached_ptr());
        for &(src, dst) in &self.mirror_pairs {
            let (d, s_ptr, n) = (copy + (dst.off * 2) as u64, at(base, src), src.len as i32);
            let mut bld = self.ctx.stream.launch_builder(cast);
            bld.arg(&d);
            bld.arg(&s_ptr);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(src.len)) }
                .map_err(|e| format!("MIMO half weight copy: {e:?}"))?;
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

    /// RMSNorm of f32 rows into `y` (the activation dtype when `typed`).
    fn rmsnorm(
        &self,
        (y, x, w, rms): (u64, u64, u64, u64),
        (rows, typed): (usize, bool),
    ) -> Result<(), String> {
        let (rows_i, dm_i) = (rows as i32, self.cfg.d_model as i32);
        let eps = self.cfg.rms_norm_eps;
        let kernel = if typed && self.act != WeightDtype::F32 {
            self.ctx.kernels.rmsnorm_fwd_f32in_typed.get(self.act)
        } else {
            &self.ctx.kernels.rmsnorm_fwd
        };
        let mut bld = self.ctx.stream.launch_builder(kernel);
        bld.arg(&y);
        bld.arg(&rms);
        bld.arg(&x);
        bld.arg(&w);
        bld.arg(&rows_i);
        bld.arg(&dm_i);
        bld.arg(&eps);
        unsafe { bld.launch(grid_norm(rows, self.cfg.d_model)) }
            .map(|_| ())
            .map_err(|e| format!("MIMO rmsnorm: {e:?}"))
    }

    /// `temporal += out` for `n` values, `out` in the activation dtype.
    fn residual_add(&self, temporal: u64, out: u64, n: usize) -> Result<(), String> {
        let n_i = n as i32;
        let stream = &self.ctx.stream;
        let mut bld = if self.act == WeightDtype::F32 {
            let mut bld = stream.launch_builder(&self.ctx.kernels.vec_add_inplace);
            bld.arg(&temporal);
            bld
        } else {
            let mut bld =
                stream.launch_builder(self.ctx.kernels.residual_add_f32_typed.get(self.act));
            bld.arg(&temporal);
            bld.arg(&temporal);
            bld
        };
        bld.arg(&out);
        bld.arg(&n_i);
        unsafe { bld.launch(grid_1d(n)) }
            .map(|_| ())
            .map_err(|e| format!("MIMO residual: {e:?}"))
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
        self.forward_unchecked(input, out)?;
        self.forwarded = true;
        Ok(())
    }

    fn forward_unchecked(&mut self, input: &[f32], out: &mut [f32]) -> Result<(), String> {
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
        let stream = self.ctx.stream.clone();
        self.input.upload(&stream, input)?;
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
        let (rows_l, rg) = (rows as i64, (rank * ng) as i32);
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
            self.rmsnorm(
                (
                    a.post.ptr(),
                    temporal,
                    at(base, ls.norm_weight),
                    a.rms.cached_ptr(),
                ),
                (rows, true),
            )?;
            let proj = a.proj.ptr();
            self.gemm(
                (proj, a.post.ptr(), self.gemm_weight(ls.in_proj_w)?),
                (rows, dm, ip),
            )?;
            let (bn, cn) = (a.bn.cached_ptr(), a.cn.cached_ptr());
            {
                let (wb, wc) = (at(base, ls.b_norm_weight), at(base, ls.c_norm_weight));
                let mut bld = stream.launch_builder(self.kernels.bc_norm.get(self.act));
                bld.arg(&bn);
                bld.arg(&cn);
                bld.arg(&proj);
                bld.arg(&ld_proj);
                bld.arg(&b_col);
                bld.arg(&c_col);
                bld.arg(&wb);
                bld.arg(&wc);
                bld.arg(&rows_l);
                bld.arg(&rg);
                bld.arg(&ds_i);
                bld.arg(&eps);
                let cfg = LaunchConfig {
                    grid_dim: ((rows * rank * ng) as u32, 2, 1),
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
                unsafe { bld.launch(scan_cfg(b, nh, (hd, ds, rank, na))) }
                    .map_err(|e| format!("MIMO scan L{l}: {e:?}"))?;
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
                self.rmsnorm(
                    (
                        m.post.ptr(),
                        temporal,
                        at(base, ms.norm_weight),
                        m.rms.cached_ptr(),
                    ),
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
                unsafe { bld.launch(grid_1d(rows * hidden)) }
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
        self.rmsnorm(
            (
                self.final_out.cached_ptr(),
                temporal,
                at(base, self.layout.norm_f_weight),
                self.final_rms.cached_ptr(),
            ),
            (rows, false),
        )?;
        self.sync("forward")?;
        self.final_out.download(&stream, out)?;
        self.sync("forward")
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
    /// per-row partials can pass 2^31 elements.
    fn reduce_rows(
        &self,
        out: u64,
        parts: u64,
        (rows, dim): (usize, usize),
        accumulate: bool,
    ) -> Result<(), String> {
        let block = (rows as u32).next_power_of_two().clamp(32, 256);
        let (rows_i, dim_i, acc) = (rows as i64, dim as i64, i32::from(accumulate));
        let mut bld = self.ctx.stream.launch_builder(&self.kernels.cb.reduce_rows);
        bld.arg(&out);
        bld.arg(&parts);
        bld.arg(&rows_i);
        bld.arg(&dim_i);
        bld.arg(&acc);
        let cfg = LaunchConfig {
            grid_dim: (dim as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        unsafe { bld.launch(cfg) }
            .map(|_| ())
            .map_err(|e| format!("MIMO column reduction: {e:?}"))
    }

    /// RMSNorm backward of `rows` rows into `dx` (f32), the scale's
    /// gradient accumulated into `g_scale`; `dy` is in the activation dtype
    /// when `typed_dy`.
    fn rms_backward(
        &self,
        (dx, dy, x, scale, rms): (u64, u64, u64, u64, u64),
        (g_scale, typed_dy): (u64, bool),
        rows: usize,
    ) -> Result<(), String> {
        let dm = self.cfg.d_model;
        let (rows_i, dm_i) = (rows as i32, dm as i32);
        let axis0 = self.bwd.axis0.cached_ptr();
        let zero: i32 = 0;
        let null: u64 = 0;
        let kernel = if typed_dy && self.act != WeightDtype::F32 {
            self.ctx.kernels.rmsnorm_bwd_f32in_typed.get(self.act)
        } else {
            &self.ctx.kernels.rmsnorm_bwd
        };
        let mut bld = self.ctx.stream.launch_builder(kernel);
        bld.arg(&dx);
        bld.arg(&axis0);
        bld.arg(&dy);
        bld.arg(&x);
        bld.arg(&scale);
        bld.arg(&rms);
        bld.arg(&rows_i);
        bld.arg(&dm_i);
        bld.arg(&zero);
        bld.arg(&null);
        unsafe { bld.launch(grid_norm(rows, dm)) }
            .map_err(|e| format!("MIMO rmsnorm backward: {e:?}"))?;
        self.reduce_axis0(g_scale, axis0, (rows, dm))
    }

    /// `d_temp += d_pre`.
    fn add_residual_grad(&self, rows: usize) -> Result<(), String> {
        let n = (rows * self.cfg.d_model) as i32;
        let (d_temp, d_pre) = (self.bwd.d_temp.cached_ptr(), self.bwd.d_pre.cached_ptr());
        let mut bld = self
            .ctx
            .stream
            .launch_builder(&self.ctx.kernels.vec_add_inplace);
        bld.arg(&d_temp);
        bld.arg(&d_pre);
        bld.arg(&n);
        unsafe { bld.launch(grid_1d(rows * self.cfg.d_model)) }
            .map(|_| ())
            .map_err(|e| format!("MIMO backward residual: {e:?}"))
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
        let n = rows * self.cfg.d_model;
        let (src, n_i) = (self.bwd.d_temp.cached_ptr(), n as i32);
        let mut bld = self.ctx.stream.launch_builder(cast);
        bld.arg(&dth);
        bld.arg(&src);
        bld.arg(&n_i);
        unsafe { bld.launch(grid_1d(n)) }.map_err(|e| format!("MIMO rounding rows: {e:?}"))?;
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
        let c = self.cfg;
        let rows = self.batch * self.seq_len;
        let dm = c.d_model;
        let stream = self.ctx.stream.clone();
        self.grads.zero(&stream)?;
        let scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        match scale {
            Some(sc) => {
                let scaled: Vec<f32> = d_out.iter().map(|v| v * sc).collect();
                self.bwd.d_rows.upload(&stream, &scaled)?;
            }
            None => self.bwd.d_rows.upload(&stream, d_out)?,
        }
        let base = self.master.cached_ptr();
        let g_base = self.grads.cached_ptr();
        self.rms_backward(
            (
                self.bwd.d_temp.cached_ptr(),
                self.bwd.d_rows.cached_ptr(),
                self.final_in.cached_ptr(),
                at(base, self.layout.norm_f_weight),
                self.final_rms.cached_ptr(),
            ),
            (at(g_base, self.layout.norm_f_weight), false),
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
        self.overflowed = false;
        if let Some(sc) = scale {
            self.overflow.zero(&stream)?;
            check_inf_nan_gpu(
                &self.ctx,
                &self.ctx.kernels,
                &mut self.overflow,
                &self.grads,
            )?;
            self.unscale.write(&stream, 1.0 / sc)?;
            scale_grads_skip_gpu(
                &self.ctx,
                &self.ctx.kernels,
                &mut self.overflow,
                &mut self.grads,
                &self.unscale,
            )?;
            self.sync("backward")?;
            self.overflowed = self.overflow.read(&stream)? != 0;
        }
        self.sync("backward")?;
        if let Some(d_input) = d_input {
            let src = if identity_input {
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

    fn mlp_backward(&mut self, l: usize) -> Result<(), String> {
        let c = self.cfg;
        let rows = self.batch * self.seq_len;
        let dm = c.d_model;
        let ms = self.layout.mlp[l];
        let hidden = ms.fc2_w.len / dm;
        let g_base = self.grads.cached_ptr();
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
            unsafe { bld.launch(grid_1d(rows * hidden)) }
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
        self.rms_backward(
            (
                self.bwd.d_pre.cached_ptr(),
                self.bwd.d_post2.ptr(),
                m.x_in.cached_ptr(),
                at(self.master.cached_ptr(), ms.norm_weight),
                m.rms.cached_ptr(),
            ),
            (at(g_base, ms.norm_weight), true),
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
            let cfg3 = |x: usize, y: usize, z: usize, block: usize, smem: usize| LaunchConfig {
                grid_dim: (x as u32, y as u32, z as u32),
                block_dim: (block as u32, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            go!(
                k.cb.gate.get(act),
                grid_1d(rows * nh * hd),
                "gate",
                [
                    dy, du, d_proj, ld_proj, gphi, gz, gd, d_o, y, proj, w_d, w_x, w_z, w_o,
                    rows_l, nh_i, hd_i, r_i
                ]
            );
            go!(
                &k.cb.inter,
                cfg3(b * nc, nh, CHUNK, 256, 0),
                "inter-chunk term",
                [
                    dq, inter, dy, enter, cn, w_cb, coef, sn, cs, t_i, nh_i, ng_i, hd_i, ds_i, r_i,
                    na_i
                ]
            );
            let (nt, dsloc_smem) = geom.dsloc;
            let nt_i = nt as i32;
            go!(
                &k.cb.dsloc,
                cfg3(b * nc, nh, ds.div_ceil(nt), nt * hd, dsloc_smem),
                "local state gradient",
                [
                    dsl, dy, cn, w_cb, coef, sn, cs, t_i, nh_i, ng_i, hd_i, ds_i, r_i, na_i, nt_i
                ]
            );
            go!(
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
            go!(
                k.cb.state_terms.get(act),
                cfg3(b * nc, nh, CHUNK, 256, rank * ds * 4),
                "state terms",
                [
                    dk, du, dcs, dsl, proj, ld_proj, bn, w_bb, coef, sn, cs, w_x, t_i, nh_i, ng_i,
                    hd_i, ds_i, r_i, na_i
                ]
            );
            let (tl, pairs_smem) = geom.pairs;
            let tl_i = tl as i32;
            go!(
                k.cb.pairs.get(act),
                cfg3(b * nc, nh, 1, 256, pairs_smem),
                "pairs",
                [
                    dq, dk, du, dwp, dy, proj, ld_proj, bn, cn, w_bb, w_cb, coef, sn, cs, w_x, t_i,
                    nh_i, ng_i, hd_i, ds_i, r_i, na_i, tl_i
                ]
            );
            let lead_threads = hd.next_multiple_of(32);
            go!(
                &k.cb.lead,
                cfg3(b, nh, 1, lead_threads, lead_threads * 4),
                "entering lead",
                [
                    de0, gpsi_init, d_init, e0, k0, v0, w_x, nh_i, hd_i, ds_i, r_i
                ]
            );
            go!(
                &k.cb.combine,
                cfg3(b * nc, nh, 1, CHUNK, 0),
                "decay gradients",
                [dsc3, dwp, dcs, inter, dot, coef, t_i, nh_i]
            );
            go!(
                &k.cb.rot_bwd,
                cfg3(rows, nh, 1, 128, 0),
                "rotation",
                [
                    d_bn_h, d_cn_h, dth, dq, dk, bn, cn, w_bb, w_cb, sn, cs, nh_i, ng_i, ds_i, r_i,
                    na_i
                ]
            );
            if na > 0 {
                go!(
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
                grid_1d(rows * nh),
                "step scalars",
                [
                    d_proj, gdtb, dsc3, ddt_ang, de0, coef, proj, ld_proj, w_dtb, rows_l, t_i,
                    nh_i, heads_i, a_floor
                ]
            );
            go!(
                k.cb.u_bwd.get(act),
                grid_1d(rows * nh * hd),
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
                    grid_1d(n_rp),
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
                let threads = hd.next_multiple_of(32);
                let cfg = LaunchConfig {
                    grid_dim: (b as u32, nh as u32, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: scan_bwd_smem_bytes((hd, ds, rank, na), threads) as u32,
                };
                let mut bld = stream.launch_builder(k.scan_bwd.get(act));
                bld.arg(&d_proj);
                bld.arg(&ld_proj);
                bld.arg(&d_bn_h);
                bld.arg(&d_cn_h);
                bld.arg(&d_tanh_h);
                for p in &parts {
                    bld.arg(p);
                }
                bld.arg(&rec);
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
                unsafe { bld.launch(cfg) }
                    .map_err(|e| format!("MIMO scan backward L{l}: {e:?}"))?;
            }
            // The per-sequence partials into the gradients.
            for (g, part, n) in [
                (ls.mimo_x, s.g_x.cached_ptr(), nh * rank * hd),
                (ls.mimo_z, s.g_z.cached_ptr(), nh * rank * hd),
                (ls.mimo_o, s.g_o.cached_ptr(), nh * rank * hd),
                (ls.d_param, s.g_d.cached_ptr(), nh),
                (ls.dt_bias, s.g_dtb.cached_ptr(), nh),
                (ls.b_bias, s.g_bb.cached_ptr(), nh * rank * ds),
                (ls.c_bias, s.g_cb.cached_ptr(), nh * rank * ds),
            ] {
                let (gp, count, n_l) = (at(g_base, g), b as i32, n as i64);
                let mut bld = stream.launch_builder(&k.sum_rows);
                bld.arg(&gp);
                bld.arg(&part);
                bld.arg(&count);
                bld.arg(&n_l);
                unsafe { bld.launch(grid_1d(n)) }
                    .map_err(|e| format!("MIMO weight partials L{l}: {e:?}"))?;
            }
        }
        // The B and C norms, then their weights.
        {
            let (dw_b, dw_c) = (s.dw_b.cached_ptr(), s.dw_c.cached_ptr());
            let (b_col, c_col) = ((2 * di) as i32, (2 * di + bc) as i32);
            let (wb, wc) = (at(base, ls.b_norm_weight), at(base, ls.c_norm_weight));
            let eps = c.ssm_norm_eps;
            let mut bld = stream.launch_builder(k.bc_norm_bwd.get(act));
            bld.arg(&d_proj);
            bld.arg(&dw_b);
            bld.arg(&dw_c);
            bld.arg(&d_bn_h);
            bld.arg(&d_cn_h);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&b_col);
            bld.arg(&c_col);
            bld.arg(&wb);
            bld.arg(&wc);
            bld.arg(&rows_l);
            bld.arg(&nh_i);
            bld.arg(&ng_i);
            bld.arg(&r_i);
            bld.arg(&ds_i);
            bld.arg(&eps);
            let cfg = LaunchConfig {
                grid_dim: ((rows * rank * ng) as u32, 2, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { bld.launch(cfg) }
                .map_err(|e| format!("MIMO B/C norm backward L{l}: {e:?}"))?;
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
            unsafe { bld.launch(grid_1d(rows * na)) }
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
        self.rms_backward(
            (
                self.bwd.d_pre.cached_ptr(),
                self.bwd.d_post.ptr(),
                a.x_in.cached_ptr(),
                at(base, ls.norm_weight),
                a.rms.cached_ptr(),
            ),
            (at(g_base, ls.norm_weight), true),
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

    fn apply_adamw_unchecked(&mut self) -> Result<bool, String> {
        if let Some(scaler) = self.scaler.as_mut() {
            scaler.update(self.overflowed);
            if self.overflowed {
                return Ok(false);
            }
        }
        let (_, bias_c1, bias_c2) = self.adamw.advance();
        let (w_base, g_base) = (self.master.cached_ptr(), self.grads.cached_ptr());
        let (m_base, v_base) = (self.adamw.m.cached_ptr(), self.adamw.v.cached_ptr());
        let o = &self.adamw;
        for (_, span, decays) in self.layout.pairs(&self.template) {
            if span.len == 0 {
                continue;
            }
            let wd = if decays || !o.reference_no_decay {
                o.weight_decay
            } else {
                0.0
            };
            let (w, g, m, v) = (
                at(w_base, span),
                at(g_base, span),
                at(m_base, span),
                at(v_base, span),
            );
            let n = span.len as i32;
            let mut bld = self
                .ctx
                .stream
                .launch_builder(&self.ctx.kernels.adamw_step_f32);
            bld.arg(&w);
            bld.arg(&g);
            bld.arg(&m);
            bld.arg(&v);
            bld.arg(&o.lr);
            bld.arg(&o.beta1);
            bld.arg(&o.beta2);
            bld.arg(&o.eps);
            bld.arg(&wd);
            bld.arg(&bias_c1);
            bld.arg(&bias_c2);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(span.len)) }.map_err(|e| format!("MIMO AdamW: {e:?}"))?;
        }
        self.refresh_mirror()?;
        self.sync("AdamW")?;
        Ok(true)
    }
}
