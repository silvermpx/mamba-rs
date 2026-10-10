//! Mamba-2 training on the GPU, in f32, tf32, bf16 or f16.
//!
//! The forward runs the prefill's launches (the shared [`M2Exec`] pieces)
//! and keeps, per layer, what the backward reads: the layer's input rows
//! and norm, the normed rows, the projection, the conv output and the conv
//! state the window started from, dt and its chunk cumsum, the C.B
//! products, the state entering every chunk, the scan output and the gated
//! rows. The backward walks the layers in reverse through the kernels of
//! `kernels/mamba2_bwd.cu` and the Fixed module's GEMM, norm and reduction
//! kernels. The weights, their gradients and the AdamW moments are three
//! flat buffers in one layout, so the optimizer runs per tensor with no
//! copies.
//!
//! In tf32 everything stays f32 and the GEMMs take the context's TF32
//! contract. In bf16 and f16 the weights, gradients and moments stay f32;
//! in_proj and out_proj run from half copies refreshed after every step,
//! and the rows those GEMMs read or write (the normed rows, the projection,
//! the conv output, the gated rows and their gradients) are stored half,
//! while the residual stream, the scan, the states and every reduction
//! stay f32. f16 scales the loss with the dynamic loss scaler: a step whose
//! gradients overflow is skipped and the scale backs off.
//!
//! Windows carry the conv and SSM states from one call to the next, and
//! the gradients stop at the states a window starts from.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::inference::check_device_envelope;
use super::inference::{LayerPtrs, M2Exec, Mamba2GpuState, WindowPtrs};
use super::kernels::{
    CHUNK, Mamba2Kernels, bc_tiles_smem_bytes, dx_tiles_geom, gated_norm_bwd_smem_bytes,
    seq_bwd_smem_bytes, seq_bwd_tiles_smem_bytes, seq_bwd_tiles_threads,
};
use crate::mamba_ssm::gpu::adamw::GpuAdamW;
use crate::mamba_ssm::gpu::blas::{
    TypedPtr, gpu_gemm_bi_backward_dw_grad_typed, gpu_gemm_bi_backward_grad_raw,
    gpu_gemm_ex_backward_dx_typed, gpu_gemm_f32_forward_ptrs,
};
use crate::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer, GradSlice};
use crate::mamba_ssm::gpu::context::{GemmRole, GpuCtx};
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_colsum, grid_norm, validate_rows_capacity};
use crate::mamba_ssm::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, UnscaleFactor, check_inf_nan_gpu, scale_grads_skip_gpu,
};
use crate::mamba2::config::Mamba2Config;
use crate::mamba2::weights::{Mamba2LayerWeights, Mamba2Weights};

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

/// The order of the flat buffers: the input projection, then each layer's
/// tensors in [`Mamba2LayerWeights`] field order, then the final norm.
/// A tensor the model does not have is an empty span.
#[derive(Clone, Debug)]
struct Layout {
    input_proj_w: Span,
    input_proj_b: Span,
    layers: Vec<LayerSpans>,
    norm_f_weight: Span,
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
        let norm_f_weight = take(w.norm_f_weight.len());
        Self {
            input_proj_w,
            input_proj_b,
            layers,
            norm_f_weight,
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
        out.push((w.norm_f_weight.as_slice(), self.norm_f_weight, false));
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
        read(&mut w.norm_f_weight, self.norm_f_weight);
        w
    }
}

fn at(base: u64, span: Span) -> u64 {
    base + (span.off * 4) as u64
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
    axis0: GpuBuffer,
    d_input: GpuBuffer,
    /// The residual gradient rounded to the half type, the out_proj
    /// backward's input in the half lanes.
    d_temp_h: Option<DtypedBuf>,
}

/// The Mamba-2 trainer: forward, backward and AdamW for windows of
/// `seq_len` tokens of `batch` sequences.
pub struct Mamba2GpuTrainer {
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
    /// The sequential backward keeps the token's states in shared memory
    /// when the head fits; else the states come from global memory.
    seq_tiles: bool,
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
    unscale: UnscaleFactor,
    /// The last backward's gradients overflowed (f16), so the next AdamW
    /// step is skipped.
    overflowed: bool,
    /// Set by a completed forward, cleared when one starts.
    forwarded: bool,
    /// Set by a completed backward, consumed by the next optimizer step.
    fresh_grads: bool,
}

const PAIRS: usize = CHUNK * (CHUNK + 1) / 2;

/// Tokens between the sequential path's replay checkpoints: the replay
/// records one segment of states, so the stride trades that buffer against
/// the checkpoints' memory.
const SEQ_STRIDE: usize = 16;

impl Mamba2GpuTrainer {
    /// An f32 trainer for `batch` sequences of `seq_len` tokens starting
    /// from `w`, with AdamW at its defaults (see [`Self::adamw_mut`]).
    pub fn new(
        device: &GpuDevice,
        w: &Mamba2Weights,
        dims: (Mamba2Config, usize, usize, usize),
    ) -> Result<Self, String> {
        Self::new_with_dtype(device, w, dims, WeightDtype::F32)
    }

    /// A trainer in `dtype`: f32, tf32, bf16 or f16 (see the module docs).
    pub fn new_with_dtype(
        device: &GpuDevice,
        w: &Mamba2Weights,
        (cfg, input_dim, batch, seq_len): (Mamba2Config, usize, usize, usize),
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        w.validate(&cfg, input_dim)?;
        if !w.mlp.is_empty() {
            return Err(
                "the GPU Mamba-2 trainer does not run the MLP after the mixer yet; \
                 the CPU trains this model"
                    .into(),
            );
        }
        if cfg.d_conv > 8 {
            return Err(format!(
                "the GPU conv serves d_conv <= 8, got {}",
                cfg.d_conv
            ));
        }
        if batch == 0 || seq_len == 0 {
            return Err("a training window needs at least one sequence and one token".into());
        }
        let seq_mode = !cfg.train_use_parallel_scan();
        let stride = SEQ_STRIDE.min(seq_len);
        let ctx = GpuCtx::new_from_env_with_state_cap_and_role(device, 64, GemmRole::triad(dtype))?;
        let act = match dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => WeightDtype::F32,
            half => half,
        };
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = Mamba2Kernels::compile(device.context(), arch)?;
        check_device_envelope(&cfg, &kernels)?;
        let optin = kernels.smem_optin;
        if seq_mode {
            let threads = cfg.headdim.next_multiple_of(32);
            let smem = seq_bwd_smem_bytes(cfg.headdim, cfg.d_state, threads);
            if smem > optin || threads > 1024 {
                return Err(format!(
                    "the sequential backward of a {}-channel head with {} states needs {smem} \
                     bytes of shared memory; this GPU serves {optin}",
                    cfg.headdim, cfg.d_state
                ));
            }
        }
        if cfg.rmsnorm && gated_norm_bwd_smem_bytes(cfg.ssm_dim(), cfg.ngroups, cfg.headdim) > optin
        {
            return Err(format!(
                "the gated-norm backward of a {}-channel group needs {} bytes of shared memory; \
                 this GPU serves {optin}",
                cfg.ssm_dim() / cfg.ngroups,
                gated_norm_bwd_smem_bytes(cfg.ssm_dim(), cfg.ngroups, cfg.headdim)
            ));
        }
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
        validate_rows_capacity(
            rows,
            [input_dim, ip, cd, di, dm].into_iter().max().unwrap_or(0),
        )?;
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
            dx_tiles_geom(hd, ds, optin)
        };
        let unfused = |n: usize| if dx_tiles.is_some() { 1 } else { chunked(n) };
        let seq_tiles = seq_mode && seq_bwd_tiles_smem_bytes(hd, ds) <= optin;
        let acts = (0..c.n_layers)
            .map(|_| -> Result<LayerActs, String> {
                Ok(LayerActs {
                    x_in: z(rows * dm)?,
                    rms: z(rows)?,
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
            seq_scal: z(if seq_tiles { rows * nh * 3 } else { 1 })?,
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
            for ls in &layout.layers {
                for span in [ls.in_proj_w, ls.out_proj_w] {
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
        }
        let mirror = if act == WeightDtype::F32 {
            None
        } else {
            Some(DtypedBuf::zeros(s, copy_at.max(1), act)?)
        };
        let mut trainer = Self {
            forwarded: false,
            fresh_grads: false,
            master: GpuBuffer::from_cpu(s, &layout.flatten(w))?,
            grads: z(layout.total)?,
            adamw: GpuAdamW::new(s, layout.total)?,
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
            final_rms: z(rows)?,
            final_out: z(rows * dm)?,
            conv_next: z(batch * cd * c.d_conv)?,
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
            dtype,
            act,
            seq_mode,
            stride,
            dx_tiles,
            seq_tiles,
        };
        trainer.refresh_mirror()?;
        Ok(trainer)
    }

    /// The storage dtype this trainer runs.
    /// The model's configuration.
    pub fn cfg(&self) -> &Mamba2Config {
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

    /// The f16 loss scale and its growth counter; `None` in the other lanes.
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        self.scaler.as_ref().map(DynamicLossScaler::state)
    }

    /// Restore the f16 loss scale and its growth counter (a checkpoint
    /// resume, or a scale chosen up front); no effect in the other lanes.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) {
        if let Some(scaler) = self.scaler.as_mut() {
            scaler.load_state(scale, growth_tracker);
        }
    }

    /// Round the master in_proj and out_proj weights into their half copies.
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
                .map_err(|e| format!("Mamba-2 half weight copy: {e:?}"))?;
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

    /// The current weights.
    pub fn weights(&self) -> Result<Mamba2Weights, String> {
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

    fn layer_ptrs(&self, l: usize) -> Result<LayerPtrs, String> {
        let (base, ls) = (self.master.cached_ptr(), &self.layout.layers[l]);
        Ok(LayerPtrs {
            norm_w: at(base, ls.norm_weight),
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
        self.forward_unchecked(input, out)?;
        self.forwarded = true;
        Ok(())
    }

    fn forward_unchecked(&mut self, input: &[f32], out: &mut [f32]) -> Result<(), String> {
        let c = self.cfg;
        let (b, t_len) = (self.batch, self.seq_len);
        let rows = b * t_len;
        let (dm, nh) = (c.d_model, c.nheads());
        let stream = self.ctx.stream.clone();
        self.input.upload(&stream, input)?;
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
        let x = self.exec();
        for (l, a) in self.acts.iter().enumerate() {
            let lp = self.layer_ptrs(l)?;
            self.copy(a.x_in.cached_ptr(), temporal, rows * dm, "layer input")?;
            x.norm_in_proj(
                &lp,
                (temporal, a.post.ptr(), a.rms.cached_ptr(), a.proj.ptr()),
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
            };
            x.window(l, &lp, &w, (b, t_len, !self.seq_mode))?;
            x.gate_out_residual(
                &lp,
                (
                    temporal,
                    a.proj.ptr(),
                    a.y.cached_ptr(),
                    a.gated.ptr(),
                    self.out_rows.ptr(),
                ),
                rows,
            )?;
        }
        self.copy(
            self.final_in.cached_ptr(),
            temporal,
            rows * dm,
            "final rows",
        )?;
        x.rmsnorm(
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

    fn grad_opt(&self, span: Span) -> Option<GradSlice> {
        (span.len > 0).then(|| self.grad(span))
    }

    /// RMSNorm backward of `rows` rows: `dx` = the input gradient, the
    /// scale's gradient accumulated into `g_scale`; `dy` is in the half
    /// type when `typed_dy`, else f32.
    fn rms_backward(
        &self,
        (dx, dy, x, scale, rms): (u64, u64, u64, u64, u64),
        (g_scale, typed_dy): (u64, bool),
        rows: usize,
    ) -> Result<(), String> {
        let dm = self.cfg.d_model;
        let (rows_i, dm_i) = (rows as i32, dm as i32);
        let axis0 = self.bwd.axis0.cached_ptr();
        let stream = &self.ctx.stream;
        let zero: i32 = 0;
        let null: u64 = 0;
        let kernel = if typed_dy {
            self.ctx.kernels.rmsnorm_bwd_f32in_typed.get(self.act)
        } else {
            &self.ctx.kernels.rmsnorm_bwd
        };
        let mut bld = stream.launch_builder(kernel);
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
            .map_err(|e| format!("Mamba-2 rmsnorm backward: {e:?}"))?;
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
        let n_i = n as i32;
        let mut bld = self.ctx.stream.launch_builder(cast);
        bld.arg(&dst);
        bld.arg(&src);
        bld.arg(&n_i);
        unsafe { bld.launch(grid_1d(n)) }
            .map(|_| ())
            .map_err(|e| format!("Mamba-2 rounding rows: {e:?}"))
    }

    /// `g[j] += sum_r dy[r][j]` over f32 rows, the Fixed column sum.
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
            self.layer_backward(l)?;
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

    /// Whether the last backward's gradients overflowed the f16 loss
    /// scale (its AdamW step will be skipped).
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    fn layer_backward(&mut self, l: usize) -> Result<(), String> {
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
        // One launch: the kernel, its grid, a name for errors, its arguments.
        macro_rules! go {
            ($f:expr, $cfg:expr, $what:expr, [$($arg:expr),* $(,)?]) => {{
                let mut bld = stream.launch_builder($f);
                $(bld.arg(&$arg);)*
                unsafe { bld.launch($cfg) }
                    .map_err(|e| format!("Mamba-2 backward {} L{l}: {e:?}", $what))?;
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
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, ng as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: gated_norm_bwd_smem_bytes(d_ssm, ng, hd) as u32,
            };
            let d_ssm_i = d_ssm as i32;
            go!(
                k.gated_norm.get(act),
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
                grid_1d(rows * d_ssm),
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
                grid_1d(rows * d_mlp),
                "inner MLP",
                [d_proj, ld_proj, d_gated, ld_out, proj, rows_l, d_mlp_i]
            );
        }

        // D.
        {
            let parts = s.d_parts.cached_ptr();
            go!(
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
            go!(
                k.dhloc.get(act),
                cfg3(b * nc, nh, pn.div_ceil(256), 256),
                "dH",
                [
                    ds_buf, dy, xbc, ld_xbc, c_col, acs, b_i, t_i, nh_i, ng_i, hd_i, ds_i
                ]
            );
            go!(
                &k.state_passing,
                cfg3(b * nh, pn.div_ceil(256), 1, 256),
                "state passing",
                [ds_buf, acs, b_i, t_i, nh_i, hd_i, ds_i]
            );
            let dot = s.dot.cached_ptr();
            {
                let count = per_chunk as i64;
                let pn_i = pn as i32;
                go!(
                    &k.chunk_dot,
                    cfg3(per_chunk, 1, 1, 256),
                    "chunk dot",
                    [dot, states, ds_buf, count, pn_i]
                );
            }
            let (ddtx, qv, inter) = (s.ddtx.cached_ptr(), s.qv.cached_ptr(), s.inter.cached_ptr());
            let d = lp.d;
            let per_channel = i32::from(c.d_has_hdim);
            if let Some((nk, smem)) = self.dx_tiles {
                let nk_i = nk as i32;
                go!(
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
                    ]
                );
            } else {
                let (acc, ste, yi) = (s.acc.cached_ptr(), s.ste.cached_ptr(), s.yi.cached_ptr());
                go!(
                    k.dx.get(act),
                    cfg3(b * nc, nh, (CHUNK * hd).div_ceil(128), 128),
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
                        b_i,
                        t_i,
                        nh_i,
                        ng_i,
                        hd_i,
                        ds_i
                    ]
                );
                go!(
                    k.rowsums.get(act),
                    cfg3(b * nc, nh, 1, CHUNK as u32),
                    "token sums",
                    [
                        ddtx, qv, inter, acc, ste, yi, dy, xbc, ld_xbc, dt, b_i, t_i, nh_i, hd_i
                    ]
                );
            }
            let (dcbh, mm) = (s.dcbh.cached_ptr(), s.mm.cached_ptr());
            go!(
                k.pairs.get(act),
                cfg3(b * nc, nh, PAIRS.div_ceil(256), 256),
                "pairs",
                [
                    dcbh, mm, dy, xbc, ld_xbc, cb, dt, acs, b_i, t_i, nh_i, ng_i, hd_i
                ]
            );
            let bc_smem = bc_tiles_smem_bytes(hd);
            if bc_smem <= self.kernels.smem_optin {
                let tiles = LaunchConfig {
                    grid_dim: ((b * nc) as u32, ng as u32, ds.div_ceil(32) as u32),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: bc_smem as u32,
                };
                go!(
                    k.db_tiles.get(act),
                    tiles,
                    "dB",
                    [
                        dxbc, dy, xbc, ld_xbc, b_col, c_col, ds_buf, dcbh, dt, acs, t_i, nh_i,
                        ng_i, hd_i, ds_i
                    ]
                );
                go!(
                    k.dc_tiles.get(act),
                    tiles,
                    "dC",
                    [
                        dxbc, dy, xbc, ld_xbc, b_col, c_col, states, dcbh, dt, acs, t_i, nh_i,
                        ng_i, hd_i, ds_i
                    ]
                );
            } else {
                let bc_cfg = cfg3(b * nc, ng, (CHUNK * ds).div_ceil(128), 128);
                go!(
                    k.db.get(act),
                    bc_cfg,
                    "dB",
                    [
                        dxbc, xbc, ld_xbc, b_col, c_col, ds_buf, dcbh, dt, acs, b_i, t_i, nh_i,
                        ng_i, hd_i, ds_i
                    ]
                );
                go!(
                    k.dc.get(act),
                    bc_cfg,
                    "dC",
                    [
                        dxbc, dy, xbc, ld_xbc, b_col, c_col, states, dcbh, acs, b_i, t_i, nh_i,
                        ng_i, hd_i, ds_i
                    ]
                );
            }
            {
                let (da_parts, dbias_parts) = (s.da_parts.cached_ptr(), s.dbias_parts.cached_ptr());
                let (dt_bias, a_neg) = (lp.dt_bias, lp.a_neg);
                let (lo, hi) = c.dt_limit;
                let has_limit = i32::from(c.dt_limit != (0.0, f32::INFINITY));
                go!(
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
            let threads = hd.next_multiple_of(32);
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
            if self.seq_tiles {
                let scal = s.seq_scal.cached_ptr();
                go!(
                    k.seq_scalars.get(act),
                    grid_1d(rows * nh),
                    "sequential step scalars",
                    [
                        scal, xbc, proj, ld_proj, dt_off, dt_bias, a_neg, rows_l, nh_i, ng_i, hd_i,
                        ds_i, cd_i, flags, lo, hi
                    ]
                );
                let cfg = LaunchConfig {
                    grid_dim: (b as u32, nh as u32, 1),
                    block_dim: (seq_bwd_tiles_threads(hd) as u32, 1, 1),
                    shared_mem_bytes: seq_bwd_tiles_smem_bytes(hd, ds) as u32,
                };
                go!(
                    k.seq_bwd_tiles.get(act),
                    cfg,
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
                    ]
                );
            } else {
                let cfg = LaunchConfig {
                    grid_dim: (b as u32, nh as u32, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: seq_bwd_smem_bytes(hd, ds, threads) as u32,
                };
                go!(
                    k.seq_bwd.get(act),
                    cfg,
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
                    ]
                );
            }
            let n = rows * ng * ds * 2;
            go!(
                &k.seq_bc_groups,
                grid_1d(n),
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
            go!(
                k.conv_dv.get(act),
                grid_1d(rows * cd),
                "conv dv",
                [
                    dv, dxbc, conv0, proj, ld_proj, xbc_col_i, cw, cbias, b_i, t_i, cd_i, dconv_i
                ]
            );
            go!(
                k.conv_dx.get(act),
                grid_1d(rows * cd),
                "conv dx",
                [d_proj, ld_proj, xbc_col_i, dv, cw, b_i, t_i, cd_i, dconv_i]
            );
            let g_w = at(g_base, ls.conv_w);
            let g_b = opt_at(g_base, ls.conv_b).unwrap_or(0);
            // 32 channels a block with a warp per tap and one for the bias,
            // 128 rows of dv and the inputs (plus the d_conv - 1 rows before)
            // staged at a time.
            let dw_cfg = LaunchConfig {
                grid_dim: (cd.div_ceil(32) as u32, 1, 1),
                block_dim: (32, (c.d_conv + 1) as u32, 1),
                shared_mem_bytes: ((128 + 128 + c.d_conv - 1) * 32 * 4) as u32,
            };
            go!(
                k.conv_dw.get(act),
                dw_cfg,
                "conv weights",
                [
                    g_w, g_b, dv, conv0, proj, ld_proj, xbc_col_i, b_i, t_i, cd_i, dconv_i
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
        let (d_pre, d_temp) = (self.bwd.d_pre.cached_ptr(), self.bwd.d_temp.cached_ptr());
        self.rms_backward(
            (
                d_pre,
                self.bwd.d_post.ptr(),
                a.x_in.cached_ptr(),
                lp.norm_w,
                a.rms.cached_ptr(),
            ),
            (at(g_base, ls.norm_weight), act != WeightDtype::F32),
            rows,
        )?;
        let n = (rows * dm) as i32;
        let mut bld = stream.launch_builder(&self.ctx.kernels.vec_add_inplace);
        bld.arg(&d_temp);
        bld.arg(&d_pre);
        bld.arg(&n);
        unsafe { bld.launch(grid_1d(rows * dm)) }
            .map(|_| ())
            .map_err(|e| format!("Mamba-2 backward residual L{l}: {e:?}"))
    }

    /// One AdamW step on the gradients of the last backward. Under the
    /// optimizer's reference no-decay groups only the matrices (the input
    /// projection, in_proj, the conv and out_proj weights) decay. Returns
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
            unsafe { bld.launch(grid_1d(span.len)) }
                .map_err(|e| format!("Mamba-2 AdamW: {e:?}"))?;
        }
        self.refresh_mirror()?;
        self.sync("AdamW")?;
        Ok(true)
    }
}
