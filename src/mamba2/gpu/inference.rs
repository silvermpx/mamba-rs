//! Mamba-2 decode and prefill on the GPU, in f32, tf32, bf16 or f16
//! storage.
//!
//! Per layer: RMSNorm, the in_proj GEMM, the conv step with its SiLU
//! (reading the xBC columns of the projection in place), the Mamba-2 step
//! kernel, the gated norm (or the bare gate) and the inner MLP gate into
//! one row, the out_proj GEMM and the residual add; then the final norm.
//! The projections and the activation rows between them take the storage
//! dtype; the residual stream, the scan output, the states and every small
//! weight stay f32.

use cudarc::driver::PushKernelArg;

use cudarc::driver::LaunchConfig;

use super::kernels::{CHUNK, Mamba2Kernels, gated_norm_cfg, step_cfg};
use crate::mamba_ssm::gpu::blas::{
    TypedPtr, gpu_gemm_f32_forward_ptrs, gpu_gemm_typed_forward_raw,
};
use crate::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use crate::mamba_ssm::gpu::context::{GemmMode, GemmRole, GpuCtx};
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use crate::mamba2::config::Mamba2Config;
use crate::mamba2::weights::{Mamba2LayerWeights, Mamba2Weights};
use crate::ops::portable_math::exp_portable;

/// One layer's weights on the device.
struct GpuMamba2Layer {
    norm_weight: GpuBuffer,
    /// The projections, in the activation dtype.
    in_proj_w: DtypedBuf,
    in_proj_b: Option<GpuBuffer>,
    /// `[conv_dim, d_conv]`, the layout the conv step reads.
    conv_w: GpuBuffer,
    /// The conv bias, zeros for a model without one (the step kernel
    /// always adds a bias).
    conv_b: GpuBuffer,
    dt_bias: GpuBuffer,
    /// A = -exp(a_log), computed once on the host with the portable exp.
    a_neg: GpuBuffer,
    d_param: GpuBuffer,
    gate_norm_weight: Option<GpuBuffer>,
    out_proj_w: DtypedBuf,
    out_proj_b: Option<GpuBuffer>,
}

impl GpuMamba2Layer {
    fn upload(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        lw: &Mamba2LayerWeights,
        (cfg, act): (&Mamba2Config, WeightDtype),
    ) -> Result<Self, String> {
        let up = |v: &[f32]| GpuBuffer::from_cpu(stream, v);
        let typed = |v: &[f32]| -> Result<DtypedBuf, String> {
            let buf = DtypedBuf::zeros(stream, v.len(), act)?;
            buf.upload_f32(stream, v)?;
            Ok(buf)
        };
        let opt = |v: &[f32]| -> Result<Option<GpuBuffer>, String> {
            if v.is_empty() {
                Ok(None)
            } else {
                up(v).map(Some)
            }
        };
        let conv_b = if lw.conv_b.is_empty() {
            GpuBuffer::zeros(stream, cfg.conv_dim())?
        } else {
            up(&lw.conv_b)?
        };
        let a_neg: Vec<f32> = lw.a_log.iter().map(|&a| -exp_portable(a)).collect();
        Ok(Self {
            norm_weight: up(&lw.norm_weight)?,
            in_proj_w: typed(&lw.in_proj_w)?,
            in_proj_b: opt(&lw.in_proj_b)?,
            conv_w: up(&lw.conv_w)?,
            conv_b,
            dt_bias: up(&lw.dt_bias)?,
            a_neg: up(&a_neg)?,
            d_param: up(&lw.d_param)?,
            gate_norm_weight: opt(&lw.gate_norm_weight)?,
            out_proj_w: typed(&lw.out_proj_w)?,
            out_proj_b: opt(&lw.out_proj_b)?,
        })
    }

    fn ptrs(&self) -> LayerPtrs {
        LayerPtrs {
            norm_w: self.norm_weight.cached_ptr(),
            in_w: self.in_proj_w.cached_ptr(),
            in_b: self.in_proj_b.as_ref().map(GpuBuffer::cached_ptr),
            conv_w: self.conv_w.cached_ptr(),
            conv_b: self.conv_b.cached_ptr(),
            dt_bias: self.dt_bias.cached_ptr(),
            a_neg: self.a_neg.cached_ptr(),
            d: self.d_param.cached_ptr(),
            gate_w: self.gate_norm_weight.as_ref().map(GpuBuffer::cached_ptr),
            out_w: self.out_proj_w.cached_ptr(),
            out_b: self.out_proj_b.as_ref().map(GpuBuffer::cached_ptr),
        }
    }
}

/// One layer's weights as device pointers, the form the shared launches
/// take: the engine's buffers and the trainer's flat arena both give it.
#[derive(Clone, Copy)]
pub(crate) struct LayerPtrs {
    pub norm_w: u64,
    pub in_w: u64,
    pub in_b: Option<u64>,
    pub conv_w: u64,
    /// Zeros for a model without a conv bias.
    pub conv_b: u64,
    pub dt_bias: u64,
    pub a_neg: u64,
    pub d: u64,
    pub gate_w: Option<u64>,
    pub out_w: u64,
    pub out_b: Option<u64>,
}

/// One layer's window buffers: the projection rows it reads, what it
/// writes, and the conv and SSM states it carries (this layer's slices).
pub(crate) struct WindowPtrs {
    pub proj: u64,
    pub xbc: u64,
    pub conv: u64,
    pub conv_next: u64,
    pub ssm: u64,
    pub dt: u64,
    pub acs: u64,
    pub cb: u64,
    pub states: u64,
    pub y: u64,
    /// The sequential scan's replay checkpoints, [B, S, H, P, N] (0 for
    /// none), written every `stride` tokens.
    pub ckpt: u64,
    pub stride: usize,
}

/// What the engine and the trainer share to launch a layer: the context,
/// the Mamba-2 module, the shape and the activation dtype.
#[derive(Clone, Copy)]
pub(crate) struct M2Exec<'a> {
    pub ctx: &'a GpuCtx,
    pub kernels: &'a Mamba2Kernels,
    pub cfg: &'a Mamba2Config,
    pub act: WeightDtype,
}

impl M2Exec<'_> {
    /// RMSNorm of f32 rows into `y`: in the activation dtype when `typed`,
    /// else f32.
    pub(crate) fn rmsnorm(
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
            .map_err(|e| format!("Mamba-2 rmsnorm: {e:?}"))
    }

    /// Norm and in_proj of `rows` rows of `temporal` into `proj`.
    pub(crate) fn norm_in_proj(
        &self,
        lw: &LayerPtrs,
        (temporal, post, rms, proj): (u64, u64, u64, u64),
        rows: usize,
    ) -> Result<(), String> {
        self.rmsnorm((post, temporal, lw.norm_w, rms), (rows, true))?;
        let t = |ptr| TypedPtr {
            ptr,
            dtype: self.act,
        };
        gpu_gemm_typed_forward_raw(
            self.ctx,
            t(proj),
            t(post),
            t(lw.in_w),
            lw.in_b,
            (rows, self.cfg.d_model, self.cfg.in_proj_out_dim()),
        )
    }

    /// The gated rows `[silu(z0) x0 | gate(y, z)]`, the out_proj GEMM and
    /// the residual add into `temporal`, for `rows` rows.
    pub(crate) fn gate_out_residual(
        &self,
        lw: &LayerPtrs,
        (temporal, proj, y, gated, out): (u64, u64, u64, u64, u64),
        rows: usize,
    ) -> Result<(), String> {
        let c = &self.cfg;
        let stream = &self.ctx.stream;
        let (dm, di, ip) = (c.d_model, c.d_inner(), c.in_proj_out_dim());
        let (d_mlp, d_ssm, ng, hd) = (c.d_mlp(), c.ssm_dim(), c.ngroups, c.headdim);
        let ld_out = di as i64;
        let ld_proj = ip as i64;
        let out_col = d_mlp as i32;
        let z_off = (2 * d_mlp) as i32;
        let rows_l = rows as i64;
        let (d_ssm_i, ng_i, hd_i) = (d_ssm as i32, ng as i32, hd as i32);
        if let Some(w) = lw.gate_w {
            let eps = c.gated_norm_eps;
            let nbg = i32::from(c.norm_before_gate);
            let mut bld = stream.launch_builder(self.kernels.gated_norm_fwd.get(self.act));
            bld.arg(&gated);
            bld.arg(&ld_out);
            bld.arg(&out_col);
            bld.arg(&y);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&z_off);
            bld.arg(&w);
            bld.arg(&rows_l);
            bld.arg(&d_ssm_i);
            bld.arg(&ng_i);
            bld.arg(&hd_i);
            bld.arg(&eps);
            bld.arg(&nbg);
            unsafe { bld.launch(gated_norm_cfg(rows, d_ssm, ng, hd)) }
                .map_err(|e| format!("Mamba-2 gated norm: {e:?}"))?;
        } else {
            let mut bld = stream.launch_builder(self.kernels.gate_only_fwd.get(self.act));
            bld.arg(&gated);
            bld.arg(&ld_out);
            bld.arg(&out_col);
            bld.arg(&y);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&z_off);
            bld.arg(&rows_l);
            bld.arg(&d_ssm_i);
            unsafe { bld.launch(grid_1d(rows * d_ssm)) }
                .map_err(|e| format!("Mamba-2 gate: {e:?}"))?;
        }
        if d_mlp > 0 {
            let d_mlp_i = d_mlp as i32;
            let mut bld = stream.launch_builder(self.kernels.mlp_gate_fwd.get(self.act));
            bld.arg(&gated);
            bld.arg(&ld_out);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&rows_l);
            bld.arg(&d_mlp_i);
            unsafe { bld.launch(grid_1d(rows * d_mlp)) }
                .map_err(|e| format!("Mamba-2 inner MLP: {e:?}"))?;
        }
        let t = |ptr| TypedPtr {
            ptr,
            dtype: self.act,
        };
        gpu_gemm_typed_forward_raw(
            self.ctx,
            t(out),
            t(gated),
            t(lw.out_w),
            lw.out_b,
            (rows, di, dm),
        )?;
        let n = (rows * dm) as i32;
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
        bld.arg(&n);
        unsafe { bld.launch(grid_1d(rows * dm)) }
            .map(|_| ())
            .map_err(|e| format!("Mamba-2 residual: {e:?}"))
    }

    pub(crate) fn dt_flags(&self) -> (i32, f32, f32) {
        let (lo, hi) = self.cfg.dt_limit;
        let limited = self.cfg.dt_limit != (0.0, f32::INFINITY);
        (i32::from(limited), lo, hi)
    }

    /// One layer's window from its projection rows: the conv over the
    /// window and the conv state after it, then the scan into `w.y`: with
    /// `chunked`, dt and its chunk cumsum, the C.B products, the chunk
    /// states, the state passing and the chunk scan; else the recurrence
    /// token by token (the decode step's formulas, so its bits are those of
    /// as many decode steps; `dt`, `acs`, `cb` and `states` are not
    /// touched). The conv and SSM states carry in and out.
    pub(crate) fn window(
        &self,
        l: usize,
        lw: &LayerPtrs,
        w: &WindowPtrs,
        (b, t_len, chunked): (usize, usize, bool),
    ) -> Result<(), String> {
        let c = self.cfg;
        let rows = b * t_len;
        let nc = t_len.div_ceil(CHUNK);
        let (ip, cd) = (c.in_proj_out_dim(), c.conv_dim());
        let (d_mlp, d_ssm, nh, hd, ds, ng) = (
            c.d_mlp(),
            c.ssm_dim(),
            c.nheads(),
            c.headdim,
            c.d_state,
            c.ngroups,
        );
        let stream = &self.ctx.stream;
        let (proj, xbc, conv, conv_next, ssm) = (w.proj, w.xbc, w.conv, w.conv_next, w.ssm);
        let (dt, acs, cb, states, y) = (w.dt, w.acs, w.cb, w.states, w.y);
        let conv_per_layer = b * cd * c.d_conv;
        let xbc_col = 2 * d_mlp + d_ssm;
        let (b_i, t_i, cd_i, dconv_i) = (b as i32, t_len as i32, cd as i32, c.d_conv as i32);
        let (nh_i, ng_i, hd_i, ds_i) = (nh as i32, ng as i32, hd as i32, ds as i32);
        let ld_proj = ip as i64;
        let ld_xbc = cd as i64;
        let (xbc_col_i, dt_off_i) = (xbc_col as i32, (xbc_col + cd) as i32);
        let (b_col_i, c_col_i) = (d_ssm as i32, (d_ssm + ng * ds) as i32);
        let (limited, lo, hi) = self.dt_flags();
        let d_per_channel = i32::from(c.d_has_hdim);
        let launch =
            |cfg: LaunchConfig,
             what: &str,
             l: usize,
             run: &dyn Fn(LaunchConfig) -> Result<(), cudarc::driver::DriverError>| {
                run(cfg).map_err(|e| format!("Mamba-2 window {what} L{l}: {e:?}"))
            };
        let (cw, cbias) = (lw.conv_w, lw.conv_b);
        launch(grid_1d(rows * cd), "conv", l, &|g| unsafe {
            let mut bld = stream.launch_builder(self.kernels.conv_prefill.get(self.act));
            bld.arg(&xbc);
            bld.arg(&conv);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&xbc_col_i);
            bld.arg(&cw);
            bld.arg(&cbias);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&cd_i);
            bld.arg(&dconv_i);
            bld.launch(g).map(|_| ())
        })?;
        launch(grid_1d(b * cd), "conv state", l, &|g| unsafe {
            let mut bld = stream.launch_builder(self.kernels.conv_state.get(self.act));
            bld.arg(&conv_next);
            bld.arg(&conv);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&xbc_col_i);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&cd_i);
            bld.arg(&dconv_i);
            bld.launch(g).map(|_| ())
        })?;
        let copied = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                conv,
                conv_next,
                conv_per_layer * 4,
                stream.cu_stream(),
            )
        };
        if copied != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("Mamba-2 conv state copy L{l}: {copied:?}"));
        }
        let (dtb, an) = (lw.dt_bias, lw.a_neg);
        if !chunked {
            let dp = lw.d;
            let ip_i = ip as i32;
            let (z_off_i, dt_off_seq) = ((2 * d_mlp) as i32, (xbc_col + cd) as i32);
            let flags: i32 = d_per_channel | (limited << 2);
            let (ckpt, stride_i) = (w.ckpt, w.stride as i32);
            return launch(step_cfg(b, nh, hd, ds), "sequential scan", l, &|g| unsafe {
                let kernel = if ckpt == 0 {
                    &self.kernels.seq_fwd
                } else {
                    &self.kernels.seq_fwd_ckpt
                };
                let mut bld = stream.launch_builder(kernel.get(self.act));
                bld.arg(&y);
                bld.arg(&ssm);
                bld.arg(&xbc);
                bld.arg(&proj);
                bld.arg(&dtb);
                bld.arg(&an);
                bld.arg(&dp);
                bld.arg(&b_i);
                bld.arg(&t_i);
                bld.arg(&nh_i);
                bld.arg(&ng_i);
                bld.arg(&hd_i);
                bld.arg(&ds_i);
                bld.arg(&cd_i);
                bld.arg(&ip_i);
                bld.arg(&z_off_i);
                bld.arg(&dt_off_seq);
                bld.arg(&flags);
                bld.arg(&lo);
                bld.arg(&hi);
                if ckpt != 0 {
                    bld.arg(&ckpt);
                    bld.arg(&stride_i);
                }
                bld.launch(g).map(|_| ())
            });
        }
        let dt_cfg = LaunchConfig {
            grid_dim: ((b * nc) as u32, nh.div_ceil(32) as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch(dt_cfg, "dt", l, &|g| unsafe {
            let mut bld = stream.launch_builder(self.kernels.dt_cumsum.get(self.act));
            bld.arg(&dt);
            bld.arg(&acs);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&dt_off_i);
            bld.arg(&dtb);
            bld.arg(&an);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&nh_i);
            bld.arg(&limited);
            bld.arg(&lo);
            bld.arg(&hi);
            bld.launch(g).map(|_| ())
        })?;
        let bmm_cfg = LaunchConfig {
            grid_dim: ((b * nc) as u32, ng as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch(bmm_cfg, "bmm", l, &|g| unsafe {
            let mut bld = stream.launch_builder(self.kernels.bmm_chunk.get(self.act));
            bld.arg(&cb);
            bld.arg(&xbc);
            bld.arg(&ld_xbc);
            bld.arg(&b_col_i);
            bld.arg(&c_col_i);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&ng_i);
            bld.arg(&ds_i);
            bld.launch(g).map(|_| ())
        })?;
        let state_cfg = LaunchConfig {
            grid_dim: ((b * nc) as u32, nh as u32, (hd * ds).div_ceil(256) as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch(state_cfg, "chunk state", l, &|g| unsafe {
            let mut bld = stream.launch_builder(self.kernels.chunk_state.get(self.act));
            bld.arg(&states);
            bld.arg(&xbc);
            bld.arg(&ld_xbc);
            bld.arg(&b_col_i);
            bld.arg(&dt);
            bld.arg(&acs);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&nh_i);
            bld.arg(&ng_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.launch(g).map(|_| ())
        })?;
        let pass_cfg = LaunchConfig {
            grid_dim: ((b * nh) as u32, (hd * ds).div_ceil(256) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch(pass_cfg, "state passing", l, &|g| unsafe {
            let mut bld = stream.launch_builder(&self.kernels.state_passing);
            bld.arg(&states);
            bld.arg(&ssm);
            bld.arg(&acs);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&nh_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.launch(g).map(|_| ())
        })?;
        let dp = lw.d;
        let scan_cfg = LaunchConfig {
            grid_dim: (
                (b * nc) as u32,
                nh as u32,
                (CHUNK * hd).div_ceil(128) as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        launch(scan_cfg, "chunk scan", l, &|g| unsafe {
            let mut bld = stream.launch_builder(self.kernels.chunk_scan.get(self.act));
            bld.arg(&y);
            bld.arg(&states);
            bld.arg(&cb);
            bld.arg(&xbc);
            bld.arg(&ld_xbc);
            bld.arg(&c_col_i);
            bld.arg(&dt);
            bld.arg(&acs);
            bld.arg(&dp);
            bld.arg(&d_per_channel);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&nh_i);
            bld.arg(&ng_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.launch(g).map(|_| ())
        })?;
        Ok(())
    }
}

/// The recurrent state of every layer: the conv window `[L, B, conv_dim,
/// d_conv]` (the current input last) and the SSM state `[L, B, nheads,
/// headdim, d_state]`, both f32.
pub struct Mamba2GpuState {
    pub conv: GpuBuffer,
    pub ssm: GpuBuffer,
}

impl Mamba2GpuState {
    pub fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
        self.conv.zero(&ctx.stream)?;
        self.ssm.zero(&ctx.stream)
    }
}

/// Reusable device buffers of one engine's steps.
pub struct Mamba2GpuScratch {
    input: GpuBuffer,
    temporal: GpuBuffer,
    post_norm: DtypedBuf,
    rms: GpuBuffer,
    proj: DtypedBuf,
    xbc: DtypedBuf,
    y: GpuBuffer,
    gated: DtypedBuf,
    out: DtypedBuf,
    /// The final-norm rows, f32.
    final_out: GpuBuffer,
}

/// The Mamba-2 decode and prefill engine.
pub struct Mamba2GpuInferenceEngine {
    pub kernels: Mamba2Kernels,
    pub ctx: GpuCtx,
    pub cfg: Mamba2Config,
    pub batch: usize,
    pub input_dim: usize,
    /// The activation dtype: f32 for f32 and tf32 storage, else the half
    /// type.
    act: WeightDtype,
    layers: Vec<GpuMamba2Layer>,
    input_proj: Option<(GpuBuffer, GpuBuffer)>,
    norm_f_weight: GpuBuffer,
}

impl Mamba2GpuInferenceEngine {
    /// An f32 engine for `batch` sequences, in the GEMM mode
    /// `MAMBA_RS_GEMM_MODE` names (deterministic when unset).
    pub fn new(
        device: &GpuDevice,
        w: &Mamba2Weights,
        cfg: Mamba2Config,
        input_dim: usize,
        batch: usize,
    ) -> Result<Self, String> {
        Self::new_inner(device, w, (cfg, input_dim, batch), (WeightDtype::F32, None))
    }

    /// [`Self::new`] with an explicit GEMM mode.
    pub fn new_with_mode(
        device: &GpuDevice,
        w: &Mamba2Weights,
        (cfg, input_dim, batch): (Mamba2Config, usize, usize),
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::new_inner(
            device,
            w,
            (cfg, input_dim, batch),
            (WeightDtype::F32, Some(mode)),
        )
    }

    /// An engine storing its projections in `dtype`: f32, tf32 (f32 storage
    /// with the TF32 GEMM contract), bf16 or f16.
    pub fn new_with_dtype(
        device: &GpuDevice,
        w: &Mamba2Weights,
        (cfg, input_dim, batch): (Mamba2Config, usize, usize),
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::new_inner(device, w, (cfg, input_dim, batch), (dtype, None))
    }

    fn new_inner(
        device: &GpuDevice,
        w: &Mamba2Weights,
        (cfg, input_dim, batch): (Mamba2Config, usize, usize),
        (dtype, mode): (WeightDtype, Option<GemmMode>),
    ) -> Result<Self, String> {
        w.validate(&cfg, input_dim)?;
        if !w.mlp.is_empty() {
            return Err(
                "the GPU Mamba-2 engine does not run the MLP after the mixer yet; \
                 the CPU runs this model"
                    .into(),
            );
        }
        if cfg.d_conv > 8 {
            return Err(format!(
                "the GPU conv serves d_conv <= 8, got {}",
                cfg.d_conv
            ));
        }
        let act = match dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => WeightDtype::F32,
            half => half,
        };
        let role = GemmRole::inference(dtype);
        let ctx = match mode {
            Some(mode) => GpuCtx::new_with_state_cap_mode_and_role(device, 64, mode, role)?,
            None => GpuCtx::new_from_env_with_state_cap_and_role(device, 64, role)?,
        };
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = Mamba2Kernels::compile(device.context(), arch)?;
        let layers = w
            .layers
            .iter()
            .map(|lw| GpuMamba2Layer::upload(&ctx.stream, lw, (&cfg, act)))
            .collect::<Result<Vec<_>, _>>()?;
        let input_proj = if w.input_proj_w.is_empty() {
            None
        } else {
            Some((
                GpuBuffer::from_cpu(&ctx.stream, &w.input_proj_w)?,
                GpuBuffer::from_cpu(&ctx.stream, &w.input_proj_b)?,
            ))
        };
        let norm_f_weight = GpuBuffer::from_cpu(&ctx.stream, &w.norm_f_weight)?;
        Ok(Self {
            kernels,
            ctx,
            cfg,
            batch,
            input_dim,
            act,
            layers,
            input_proj,
            norm_f_weight,
        })
    }

    fn exec(&self) -> M2Exec<'_> {
        M2Exec {
            ctx: &self.ctx,
            kernels: &self.kernels,
            cfg: &self.cfg,
            act: self.act,
        }
    }

    pub fn ctx(&self) -> &GpuCtx {
        &self.ctx
    }

    pub fn alloc_state(&self) -> Result<Mamba2GpuState, String> {
        let (b, l, c) = (self.batch, self.cfg.n_layers, &self.cfg);
        Ok(Mamba2GpuState {
            conv: GpuBuffer::zeros(&self.ctx.stream, l * b * c.conv_dim() * c.d_conv)?,
            ssm: GpuBuffer::zeros(&self.ctx.stream, l * b * c.nheads() * c.headdim * c.d_state)?,
        })
    }

    pub fn alloc_scratch(&self) -> Result<Mamba2GpuScratch, String> {
        let (b, c) = (self.batch, &self.cfg);
        let s = &self.ctx.stream;
        let typed = |n: usize| DtypedBuf::zeros(s, n, self.act);
        Ok(Mamba2GpuScratch {
            input: GpuBuffer::zeros(s, b * self.input_dim)?,
            temporal: GpuBuffer::zeros(s, b * c.d_model)?,
            post_norm: typed(b * c.d_model)?,
            rms: GpuBuffer::zeros(s, b)?,
            proj: typed(b * c.in_proj_out_dim())?,
            xbc: typed(b * c.conv_dim())?,
            y: GpuBuffer::zeros(s, b * c.ssm_dim())?,
            gated: typed(b * c.d_inner())?,
            out: typed(b * c.d_model)?,
            final_out: GpuBuffer::zeros(s, b * c.d_model)?,
        })
    }

    /// One token for every sequence: `input` is `[batch, input_dim]`,
    /// `output` (`[batch, d_model]`) receives the final-norm rows.
    pub fn step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut Mamba2GpuState,
        scratch: &mut Mamba2GpuScratch,
    ) -> Result<(), String> {
        scratch.input.upload(&self.ctx.stream, input)?;
        self.step_kernels(state, scratch)?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("Mamba-2 step sync: {e:?}"))?;
        scratch.final_out.download(&self.ctx.stream, output)?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("Mamba-2 step sync: {e:?}"))
    }

    /// The input rows into `temporal`: a copy for the identity projection,
    /// else the input GEMM.
    fn project_input(
        &self,
        (temporal, input): (&mut GpuBuffer, &GpuBuffer),
        rows: usize,
    ) -> Result<(), String> {
        match &self.input_proj {
            None => temporal.copy_from_raw(input, &self.ctx.stream),
            Some((w, bias)) => unsafe {
                gpu_gemm_f32_forward_ptrs(
                    &self.ctx,
                    temporal.cached_ptr(),
                    input.cached_ptr(),
                    w.cached_ptr(),
                    Some(bias.cached_ptr()),
                    (rows, self.input_dim, self.cfg.d_model),
                )
            },
        }
    }

    /// The step's kernels on the stream, input already uploaded; the
    /// output is left in `scratch.temporal`.
    fn step_kernels(
        &self,
        state: &mut Mamba2GpuState,
        scratch: &mut Mamba2GpuScratch,
    ) -> Result<(), String> {
        let c = &self.cfg;
        let b = self.batch;
        let (ip, cd) = (c.in_proj_out_dim(), c.conv_dim());
        let (d_mlp, d_ssm, nh, hd, ds, ng) = (
            c.d_mlp(),
            c.ssm_dim(),
            c.nheads(),
            c.headdim,
            c.d_state,
            c.ngroups,
        );
        let stream = &self.ctx.stream;
        self.project_input((&mut scratch.temporal, &scratch.input), b)?;
        let temporal = scratch.temporal.cached_ptr();
        let (post, rms, proj) = (
            scratch.post_norm.cached_ptr(),
            scratch.rms.cached_ptr(),
            scratch.proj.cached_ptr(),
        );
        let conv_per_layer = b * cd * c.d_conv;
        let ssm_per_layer = b * nh * hd * ds;
        let xbc_col = 2 * d_mlp + d_ssm;
        let (b_i, cd_i, ip_i, dconv_i) = (b as i32, cd as i32, ip as i32, c.d_conv as i32);
        let (nh_i, ng_i, hd_i, ds_i) = (nh as i32, ng as i32, hd as i32, ds as i32);
        let (z_off_i, dt_off_i) = ((2 * d_mlp) as i32, (xbc_col + cd) as i32);
        let (limited, lo, hi) = self.exec().dt_flags();
        let flags: i32 = i32::from(c.d_has_hdim) | (limited << 2);
        for (l, lw) in self.layers.iter().enumerate() {
            self.exec()
                .norm_in_proj(&lw.ptrs(), (temporal, post, rms, proj), b)?;
            // The conv step reads the xBC columns of the projection in place.
            let xbc = scratch.xbc.cached_ptr();
            {
                let conv = state.conv.cached_ptr() + (l * conv_per_layer * 4) as u64;
                let (cw, cb) = (lw.conv_w.cached_ptr(), lw.conv_b.cached_ptr());
                let ld_proj = ip as i64;
                let xbc_col_i = xbc_col as i32;
                let mut bld = stream.launch_builder(self.kernels.conv_step.get(self.act));
                bld.arg(&xbc);
                bld.arg(&conv);
                bld.arg(&proj);
                bld.arg(&ld_proj);
                bld.arg(&xbc_col_i);
                bld.arg(&cw);
                bld.arg(&cb);
                bld.arg(&b_i);
                bld.arg(&cd_i);
                bld.arg(&dconv_i);
                unsafe { bld.launch(grid_1d(b * cd)) }
                    .map_err(|e| format!("Mamba-2 conv step L{l}: {e:?}"))?;
            }
            let y = scratch.y.cached_ptr();
            {
                let ssm = state.ssm.cached_ptr() + (l * ssm_per_layer * 4) as u64;
                let (dtb, an, dp) = (
                    lw.dt_bias.cached_ptr(),
                    lw.a_neg.cached_ptr(),
                    lw.d_param.cached_ptr(),
                );
                let mut bld = stream.launch_builder(self.kernels.step_fwd.get(self.act));
                bld.arg(&y);
                bld.arg(&ssm);
                bld.arg(&xbc);
                bld.arg(&proj);
                bld.arg(&dtb);
                bld.arg(&an);
                bld.arg(&dp);
                bld.arg(&b_i);
                bld.arg(&nh_i);
                bld.arg(&ng_i);
                bld.arg(&hd_i);
                bld.arg(&ds_i);
                bld.arg(&cd_i);
                bld.arg(&ip_i);
                bld.arg(&z_off_i);
                bld.arg(&dt_off_i);
                bld.arg(&flags);
                bld.arg(&lo);
                bld.arg(&hi);
                unsafe { bld.launch(step_cfg(b, nh, hd, ds)) }
                    .map_err(|e| format!("Mamba-2 step L{l}: {e:?}"))?;
            }
            self.exec().gate_out_residual(
                &lw.ptrs(),
                (
                    temporal,
                    proj,
                    y,
                    scratch.gated.cached_ptr(),
                    scratch.out.cached_ptr(),
                ),
                b,
            )?;
        }
        self.exec().rmsnorm(
            (
                scratch.final_out.cached_ptr(),
                temporal,
                self.norm_f_weight.cached_ptr(),
                rms,
            ),
            (b, false),
        )
    }

    /// Buffers for prompt windows of `seq_len` tokens.
    pub fn alloc_prefill(&self, seq_len: usize) -> Result<Mamba2GpuPrefill, String> {
        let c = &self.cfg;
        let rows = self.batch * seq_len;
        let nc = seq_len.div_ceil(CHUNK);
        let s = &self.ctx.stream;
        let z = |n: usize| GpuBuffer::zeros(s, n.max(1));
        let typed = |n: usize| DtypedBuf::zeros(s, n.max(1), self.act);
        Ok(Mamba2GpuPrefill {
            seq_len,
            input: z(rows * self.input_dim)?,
            temporal: z(rows * c.d_model)?,
            post: typed(rows * c.d_model)?,
            rms: z(rows)?,
            proj: typed(rows * c.in_proj_out_dim())?,
            xbc: typed(rows * c.conv_dim())?,
            conv_next: z(self.batch * c.conv_dim() * c.d_conv)?,
            dt: z(self.batch * nc * c.nheads() * CHUNK)?,
            acs: z(self.batch * nc * c.nheads() * CHUNK)?,
            cb: z(self.batch * nc * c.ngroups * CHUNK * (CHUNK + 1) / 2)?,
            states: z(self.batch * nc * c.nheads() * c.headdim * c.d_state)?,
            y: z(rows * c.ssm_dim())?,
            gated: typed(rows * c.d_inner())?,
            out: typed(rows * c.d_model)?,
            final_rows: z(rows * c.d_model)?,
        })
    }

    /// A prompt window through the backbone for every sequence: `input` is
    /// `[batch, seq_len, input_dim]`, `rows_out` (`[batch, seq_len,
    /// d_model]`) receives every position's final-norm output, and `state`
    /// carries in and out.
    pub fn prefill(
        &self,
        input: &[f32],
        rows_out: &mut [f32],
        state: &mut Mamba2GpuState,
        pf: &mut Mamba2GpuPrefill,
    ) -> Result<(), String> {
        pf.input.upload(&self.ctx.stream, input)?;
        self.prefill_kernels(state, pf)?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("Mamba-2 prefill sync: {e:?}"))?;
        pf.final_rows.download(&self.ctx.stream, rows_out)?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("Mamba-2 prefill sync: {e:?}"))
    }

    fn prefill_kernels(
        &self,
        state: &mut Mamba2GpuState,
        pf: &mut Mamba2GpuPrefill,
    ) -> Result<(), String> {
        let c = &self.cfg;
        let (b, t_len) = (self.batch, pf.seq_len);
        let rows = b * t_len;
        let conv_per_layer = b * c.conv_dim() * c.d_conv;
        let ssm_per_layer = b * c.nheads() * c.headdim * c.d_state;
        self.project_input((&mut pf.temporal, &pf.input), rows)?;
        let temporal = pf.temporal.cached_ptr();
        let (post, rms, proj, y) = (
            pf.post.cached_ptr(),
            pf.rms.cached_ptr(),
            pf.proj.cached_ptr(),
            pf.y.cached_ptr(),
        );
        let x = self.exec();
        for (l, lw) in self.layers.iter().enumerate() {
            let lp = lw.ptrs();
            x.norm_in_proj(&lp, (temporal, post, rms, proj), rows)?;
            let w = WindowPtrs {
                proj,
                xbc: pf.xbc.cached_ptr(),
                conv: state.conv.cached_ptr() + (l * conv_per_layer * 4) as u64,
                conv_next: pf.conv_next.cached_ptr(),
                ssm: state.ssm.cached_ptr() + (l * ssm_per_layer * 4) as u64,
                dt: pf.dt.cached_ptr(),
                acs: pf.acs.cached_ptr(),
                cb: pf.cb.cached_ptr(),
                states: pf.states.cached_ptr(),
                y,
                ckpt: 0,
                stride: 1,
            };
            x.window(l, &lp, &w, (b, t_len, c.prefill_parallel(t_len)))?;
            x.gate_out_residual(
                &lp,
                (
                    temporal,
                    proj,
                    y,
                    pf.gated.cached_ptr(),
                    pf.out.cached_ptr(),
                ),
                rows,
            )?;
        }
        x.rmsnorm(
            (
                pf.final_rows.cached_ptr(),
                temporal,
                self.norm_f_weight.cached_ptr(),
                rms,
            ),
            (rows, false),
        )
    }
}

/// Buffers of one prompt window length.
pub struct Mamba2GpuPrefill {
    seq_len: usize,
    input: GpuBuffer,
    temporal: GpuBuffer,
    post: DtypedBuf,
    rms: GpuBuffer,
    proj: DtypedBuf,
    xbc: DtypedBuf,
    conv_next: GpuBuffer,
    dt: GpuBuffer,
    acs: GpuBuffer,
    cb: GpuBuffer,
    states: GpuBuffer,
    y: GpuBuffer,
    gated: DtypedBuf,
    out: DtypedBuf,
    final_rows: GpuBuffer,
}
