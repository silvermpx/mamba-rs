//! Mamba-3 MIMO on the GPU, in f32, tf32, bf16 or f16 storage.
//!
//! One path serves decode and prefill: a decode step is a window of one
//! token. Per layer: RMSNorm, the in_proj GEMM, the B and C row norms, the
//! scan of every head (which writes the gated rows), the out_proj GEMM and
//! the residual add, then the gated MLP when the model has one; then the
//! final norm. The projections and the activation rows between them take
//! the storage dtype; the residual stream, the normalised B and C, the
//! states and every small weight stay f32, and the final norm writes f32.

use cudarc::driver::PushKernelArg;

use super::kernels::{Mamba3MimoKernels, SCAN_SMEM_MAX, scan_cfg, scan_smem_bytes};
use crate::mamba_ssm::gpu::blas::{
    TypedPtr, gpu_gemm_f32_forward_ptrs, gpu_gemm_typed_forward_raw,
};
use crate::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use crate::mamba_ssm::gpu::context::{GemmMode, GemmRole, GpuCtx};
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};

struct GpuMimoLayer {
    norm_weight: GpuBuffer,
    in_proj_w: DtypedBuf,
    dt_bias: GpuBuffer,
    b_norm_weight: GpuBuffer,
    c_norm_weight: GpuBuffer,
    b_bias: GpuBuffer,
    c_bias: GpuBuffer,
    d_param: GpuBuffer,
    mimo_x: GpuBuffer,
    mimo_z: GpuBuffer,
    mimo_o: GpuBuffer,
    out_proj_w: DtypedBuf,
    mlp: Option<GpuMlp>,
}

struct GpuMlp {
    norm_weight: GpuBuffer,
    fc1_w: DtypedBuf,
    fc2_w: DtypedBuf,
    hidden: usize,
}

/// The recurrent state of every layer, f32: ssm `[L, B, H, P, N]`, keys
/// `[L, B, H, R, N]`, the raw previous input `[L, B, H, P]`, the angles
/// `[L, B, H, na]`.
pub struct Mamba3MimoGpuState {
    pub ssm: GpuBuffer,
    pub k: GpuBuffer,
    pub v: GpuBuffer,
    pub angle: GpuBuffer,
}

/// Buffers for windows of one length (a decode step is a window of one).
pub struct Mamba3MimoGpuWindow {
    seq_len: usize,
    input: GpuBuffer,
    temporal: GpuBuffer,
    post: DtypedBuf,
    rms: GpuBuffer,
    proj: DtypedBuf,
    bn: GpuBuffer,
    cn: GpuBuffer,
    gated: DtypedBuf,
    out: DtypedBuf,
    mlp_up: DtypedBuf,
    mlp_act: DtypedBuf,
    final_rows: GpuBuffer,
}

/// The MIMO decode and prefill engine.
pub struct Mamba3MimoGpuEngine {
    pub kernels: Mamba3MimoKernels,
    pub ctx: GpuCtx,
    pub cfg: Mamba3MimoConfig,
    pub batch: usize,
    pub input_dim: usize,
    act: WeightDtype,
    layers: Vec<GpuMimoLayer>,
    input_proj: Option<(GpuBuffer, GpuBuffer)>,
    norm_f_weight: GpuBuffer,
}

impl Mamba3MimoGpuEngine {
    /// An engine for `batch` sequences storing its projections in `dtype`
    /// (f32, tf32, bf16 or f16), in the GEMM mode `MAMBA_RS_GEMM_MODE`
    /// names (deterministic when unset).
    pub fn new(
        device: &GpuDevice,
        w: &Mamba3MimoWeights,
        (cfg, input_dim, batch): (Mamba3MimoConfig, usize, usize),
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::new_inner(device, w, (cfg, input_dim, batch), (dtype, None))
    }

    /// [`Self::new`] with an explicit GEMM mode.
    pub fn new_with_mode(
        device: &GpuDevice,
        w: &Mamba3MimoWeights,
        (cfg, input_dim, batch): (Mamba3MimoConfig, usize, usize),
        (dtype, mode): (WeightDtype, GemmMode),
    ) -> Result<Self, String> {
        Self::new_inner(device, w, (cfg, input_dim, batch), (dtype, Some(mode)))
    }

    fn new_inner(
        device: &GpuDevice,
        w: &Mamba3MimoWeights,
        (cfg, input_dim, batch): (Mamba3MimoConfig, usize, usize),
        (dtype, mode): (WeightDtype, Option<GemmMode>),
    ) -> Result<Self, String> {
        w.validate(&cfg, input_dim)?;
        if cfg.is_outproj_norm {
            return Err("the gated output norm does not run on the GPU yet".into());
        }
        let smem = scan_smem_bytes(
            cfg.headdim,
            cfg.d_state,
            cfg.mimo_rank,
            cfg.num_rope_angles(),
        );
        if smem > SCAN_SMEM_MAX || cfg.headdim > 1024 {
            return Err(format!(
                "a head of {} channels and {} states needs {smem} bytes of shared memory; \
                 the GPU scan serves {SCAN_SMEM_MAX}",
                cfg.headdim, cfg.d_state
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
        let kernels = Mamba3MimoKernels::compile(device.context(), arch)?;
        let s = &ctx.stream;
        let up = |v: &[f32]| GpuBuffer::from_cpu(s, v);
        let typed = |v: &[f32]| -> Result<DtypedBuf, String> {
            let buf = DtypedBuf::zeros(s, v.len(), act)?;
            buf.upload_f32(s, v)?;
            Ok(buf)
        };
        let layer = |l: usize, lw: &Mamba3MimoLayerWeights| -> Result<GpuMimoLayer, String> {
            let mlp = match w.mlp.get(l) {
                None => None,
                Some(m) => Some(GpuMlp {
                    norm_weight: up(&m.norm_weight)?,
                    fc1_w: typed(&m.fc1_w)?,
                    fc2_w: typed(&m.fc2_w)?,
                    hidden: m.hidden(),
                }),
            };
            Ok(GpuMimoLayer {
                norm_weight: up(&lw.norm_weight)?,
                in_proj_w: typed(&lw.in_proj_w)?,
                dt_bias: up(&lw.dt_bias)?,
                b_norm_weight: up(&lw.b_norm_weight)?,
                c_norm_weight: up(&lw.c_norm_weight)?,
                b_bias: up(&lw.b_bias)?,
                c_bias: up(&lw.c_bias)?,
                d_param: up(&lw.d_param)?,
                mimo_x: up(&lw.mimo_x)?,
                mimo_z: up(&lw.mimo_z)?,
                mimo_o: up(&lw.mimo_o)?,
                out_proj_w: typed(&lw.out_proj_w)?,
                mlp,
            })
        };
        let layers = w
            .layers
            .iter()
            .enumerate()
            .map(|(l, lw)| layer(l, lw))
            .collect::<Result<Vec<_>, _>>()?;
        let input_proj = if w.input_proj_w.is_empty() {
            None
        } else {
            Some((up(&w.input_proj_w)?, up(&w.input_proj_b)?))
        };
        let norm_f_weight = up(&w.norm_f_weight)?;
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

    pub fn ctx(&self) -> &GpuCtx {
        &self.ctx
    }

    pub fn alloc_state(&self) -> Result<Mamba3MimoGpuState, String> {
        let c = &self.cfg;
        let (l, b, nh) = (c.n_layers, self.batch, c.nheads());
        let s = &self.ctx.stream;
        Ok(Mamba3MimoGpuState {
            ssm: GpuBuffer::zeros(s, l * b * nh * c.headdim * c.d_state)?,
            k: GpuBuffer::zeros(s, l * b * nh * c.mimo_rank * c.d_state)?,
            v: GpuBuffer::zeros(s, l * b * nh * c.headdim)?,
            angle: GpuBuffer::zeros(s, (l * b * nh * c.num_rope_angles()).max(1))?,
        })
    }

    /// Buffers for windows of `seq_len` tokens; `alloc_window(1)` serves
    /// decode steps.
    pub fn alloc_window(&self, seq_len: usize) -> Result<Mamba3MimoGpuWindow, String> {
        let c = &self.cfg;
        let rows = self.batch * seq_len;
        let s = &self.ctx.stream;
        let z = |n: usize| GpuBuffer::zeros(s, n.max(1));
        let typed = |n: usize| DtypedBuf::zeros(s, n.max(1), self.act);
        let hidden = c.mlp_hidden();
        Ok(Mamba3MimoGpuWindow {
            seq_len,
            input: z(rows * self.input_dim)?,
            temporal: z(rows * c.d_model)?,
            post: typed(rows * c.d_model)?,
            rms: z(rows)?,
            proj: typed(rows * c.in_proj_out_dim())?,
            bn: z(rows * c.bc_width())?,
            cn: z(rows * c.bc_width())?,
            gated: typed(rows * c.d_inner())?,
            out: typed(rows * c.d_model)?,
            mlp_up: typed(rows * 2 * hidden)?,
            mlp_act: typed(rows * hidden)?,
            final_rows: z(rows * c.d_model)?,
        })
    }

    /// A window through the backbone for every sequence: `input` is
    /// `[batch, seq_len, input_dim]`, `rows_out` (`[batch, seq_len,
    /// d_model]`) receives every position's final-norm output, and `state`
    /// carries in and out. A decode step is a window of one token.
    pub fn run(
        &self,
        input: &[f32],
        rows_out: &mut [f32],
        state: &mut Mamba3MimoGpuState,
        win: &mut Mamba3MimoGpuWindow,
    ) -> Result<(), String> {
        win.input.upload(&self.ctx.stream, input)?;
        self.kernels_on_stream(state, win)?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("MIMO sync: {e:?}"))?;
        win.final_rows.download(&self.ctx.stream, rows_out)?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("MIMO sync: {e:?}"))
    }

    fn typed_ptr(&self, ptr: u64) -> TypedPtr {
        TypedPtr {
            ptr,
            dtype: self.act,
        }
    }

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
        let stream = &self.ctx.stream;
        let n_i = n as i32;
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

    fn kernels_on_stream(
        &self,
        state: &mut Mamba3MimoGpuState,
        win: &mut Mamba3MimoGpuWindow,
    ) -> Result<(), String> {
        let c = &self.cfg;
        let (b, t_len) = (self.batch, win.seq_len);
        let rows = b * t_len;
        let (dm, di, ip, bc) = (c.d_model, c.d_inner(), c.in_proj_out_dim(), c.bc_width());
        let (nh, hd, ds, r, ng, na) = (
            c.nheads(),
            c.headdim,
            c.d_state,
            c.mimo_rank,
            c.ngroups,
            c.num_rope_angles(),
        );
        let stream = &self.ctx.stream;
        match &self.input_proj {
            None => win.temporal.copy_from_raw(&win.input, stream)?,
            Some((w, bias)) => unsafe {
                gpu_gemm_f32_forward_ptrs(
                    &self.ctx,
                    win.temporal.cached_ptr(),
                    win.input.cached_ptr(),
                    w.cached_ptr(),
                    Some(bias.cached_ptr()),
                    (rows, self.input_dim, dm),
                )
            }?,
        }
        let temporal = win.temporal.cached_ptr();
        let (post, rms, proj) = (
            win.post.cached_ptr(),
            win.rms.cached_ptr(),
            win.proj.cached_ptr(),
        );
        let (bn, cn, gated, out) = (
            win.bn.cached_ptr(),
            win.cn.cached_ptr(),
            win.gated.cached_ptr(),
            win.out.cached_ptr(),
        );
        let per = |n: usize| (b * nh * n * 4) as u64;
        let ld_proj = ip as i64;
        let (b_col, c_col) = ((2 * di) as i32, (2 * di + bc) as i32);
        let rows_l = rows as i64;
        let rg = (r * ng) as i32;
        let (seq_i, nh_i, ng_i, hd_i, ds_i, r_i, na_i) = (
            t_len as i32,
            nh as i32,
            ng as i32,
            hd as i32,
            ds as i32,
            r as i32,
            na as i32,
        );
        let eps = c.ssm_norm_eps;
        let a_floor = c.a_floor;
        for (l, lw) in self.layers.iter().enumerate() {
            self.rmsnorm(
                (post, temporal, lw.norm_weight.cached_ptr(), rms),
                (rows, true),
            )?;
            gpu_gemm_typed_forward_raw(
                &self.ctx,
                self.typed_ptr(proj),
                self.typed_ptr(post),
                self.typed_ptr(lw.in_proj_w.cached_ptr()),
                None,
                (rows, dm, ip),
            )?;
            {
                let (wb, wc) = (lw.b_norm_weight.cached_ptr(), lw.c_norm_weight.cached_ptr());
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
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((rows * r * ng) as u32, 2, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { bld.launch(cfg) }.map_err(|e| format!("MIMO B/C norm L{l}: {e:?}"))?;
            }
            {
                let ssm = state.ssm.cached_ptr() + l as u64 * per(hd * ds);
                let kst = state.k.cached_ptr() + l as u64 * per(r * ds);
                let vst = state.v.cached_ptr() + l as u64 * per(hd);
                let ang = state.angle.cached_ptr() + l as u64 * per(na);
                let ptrs = [
                    lw.dt_bias.cached_ptr(),
                    lw.b_bias.cached_ptr(),
                    lw.c_bias.cached_ptr(),
                    lw.d_param.cached_ptr(),
                    lw.mimo_x.cached_ptr(),
                    lw.mimo_z.cached_ptr(),
                    lw.mimo_o.cached_ptr(),
                ];
                let mut bld = stream.launch_builder(self.kernels.scan.get(self.act));
                bld.arg(&gated);
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
                bld.arg(&seq_i);
                bld.arg(&nh_i);
                bld.arg(&ng_i);
                bld.arg(&hd_i);
                bld.arg(&ds_i);
                bld.arg(&r_i);
                bld.arg(&na_i);
                bld.arg(&a_floor);
                unsafe { bld.launch(scan_cfg(b, nh, (hd, ds, r, na))) }
                    .map_err(|e| format!("MIMO scan L{l}: {e:?}"))?;
            }
            gpu_gemm_typed_forward_raw(
                &self.ctx,
                self.typed_ptr(out),
                self.typed_ptr(gated),
                self.typed_ptr(lw.out_proj_w.cached_ptr()),
                None,
                (rows, di, dm),
            )?;
            self.residual_add(temporal, out, rows * dm)?;
            if let Some(m) = &lw.mlp {
                let (upp, actp) = (win.mlp_up.cached_ptr(), win.mlp_act.cached_ptr());
                self.rmsnorm(
                    (post, temporal, m.norm_weight.cached_ptr(), rms),
                    (rows, true),
                )?;
                gpu_gemm_typed_forward_raw(
                    &self.ctx,
                    self.typed_ptr(upp),
                    self.typed_ptr(post),
                    self.typed_ptr(m.fc1_w.cached_ptr()),
                    None,
                    (rows, dm, 2 * m.hidden),
                )?;
                let (rows_i, h_i) = (rows as i32, m.hidden as i32);
                let mut bld = stream.launch_builder(self.kernels.mlp_gate.get(self.act));
                bld.arg(&actp);
                bld.arg(&upp);
                bld.arg(&rows_i);
                bld.arg(&h_i);
                unsafe { bld.launch(grid_1d(rows * m.hidden)) }
                    .map_err(|e| format!("MIMO MLP gate L{l}: {e:?}"))?;
                gpu_gemm_typed_forward_raw(
                    &self.ctx,
                    self.typed_ptr(out),
                    self.typed_ptr(actp),
                    self.typed_ptr(m.fc2_w.cached_ptr()),
                    None,
                    (rows, m.hidden, dm),
                )?;
                self.residual_add(temporal, out, rows * dm)?;
            }
        }
        self.rmsnorm(
            (
                win.final_rows.cached_ptr(),
                temporal,
                self.norm_f_weight.cached_ptr(),
                rms,
            ),
            (rows, false),
        )
    }
}
