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

use crate::bidi::{mlps_after, opens_pair, reads_reversed};
use crate::gpu::bidi::{WindowRows, reverse_rows, zero_bytes};

use super::kernels::{
    CHUNK, ChunkedGeom, Mamba3MimoKernels, ScanGeom, chunk_scan_tiles_cfg, chunked_geom, scan_geom,
};
use crate::config::ScanMode;
use crate::gpu::blas::{TypedPtr, gpu_gemm_f32_forward_ptrs, gpu_gemm_typed_forward_raw};
use crate::gpu::block_norm::{self, NormParams};
use crate::gpu::buffers::{DtypedBuf, GpuBuffer, GpuByteBuffer};
use crate::gpu::context::{GemmMode, GemmRole, GpuCtx, gemm_mode_or_env};
use crate::gpu::device::GpuDevice;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::gemm_bi_inference::prepare_inference_arch_rung;
use crate::gpu::graph_capture::{
    capture_into_graph_with_gemm_plan, require_deterministic_gemm_graph_plan,
    with_validated_gemm_graph_launch,
};
use crate::gpu::kernel_identity::{CapturedGemmGraphPlan, PreparedGemmCaptureManifest};
use crate::gpu::launch::{grid_1d_cover, grid_rows, launch_elems, y_runs, yz_runs};
use crate::gpu::train::RowSource;
use crate::mamba3_mimo::config::Mamba3MimoConfig;
use crate::mamba3_mimo::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use crate::matrix::{Path, Request};

/// Inference runs one sequence per row: the scans take no packing (see
/// [`crate::varlen`]).
const NO_PACKING: u64 = 0;

struct GpuMimoLayer {
    norm_weight: GpuBuffer,
    /// The block LayerNorm's bias, None for RMSNorm.
    norm_bias: Option<GpuBuffer>,
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
    /// The output norm's weight `[d_inner]`, for a model with
    /// `is_outproj_norm`.
    norm_gate: Option<GpuBuffer>,
    mlp: Option<GpuMlp>,
}

struct GpuMlp {
    norm_weight: GpuBuffer,
    /// The norm's LayerNorm bias, None for RMSNorm.
    norm_bias: Option<GpuBuffer>,
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

impl Mamba3MimoGpuState {
    /// Zero every layer's state, the start of a new sequence.
    pub fn reset(&mut self, ctx: &GpuCtx) -> Result<(), String> {
        self.ssm.zero(&ctx.stream)?;
        self.k.zero(&ctx.stream)?;
        self.v.zero(&ctx.stream)?;
        self.angle.zero(&ctx.stream)
    }
}

/// Buffers for windows of one length (a decode step is a window of one),
/// and the window's CUDA graph once [`Mamba3MimoGpuEngine::capture_graph`]
/// ran.
pub struct Mamba3MimoGpuWindow {
    // The graph drops first, before the buffers it reads.
    graph: Option<CapturedWindow>,
    /// The GEMM inventory of the last eager run of the window, which a
    /// capture takes.
    eager_gemm_manifest: Option<PreparedGemmCaptureManifest>,
    seq_len: usize,
    input: GpuBuffer,
    temporal: GpuBuffer,
    post: DtypedBuf,
    rms: GpuBuffer,
    proj: DtypedBuf,
    bn: GpuBuffer,
    cn: GpuBuffer,
    gated: DtypedBuf,
    /// The scans' readouts `[B, T, H, R, P]` the output norm gates (one
    /// element for a model without it).
    y: GpuBuffer,
    out: DtypedBuf,
    mlp_up: DtypedBuf,
    mlp_act: DtypedBuf,
    final_rows: GpuBuffer,
    /// The chunked path's per-token scalars `[B, T, H, 4]`, angle sines and
    /// cosines `[B, T, H, na]`, first-token lead `[B, H]`, chunk
    /// contributions and entering states `[B, nc, H, P, N]` (one element
    /// each when the window runs the recurrence).
    coef: GpuBuffer,
    sn: GpuBuffer,
    cs: GpuBuffer,
    e0: GpuBuffer,
    contrib: GpuBuffer,
    enter: GpuBuffer,
    /// A bidirectional pair's input rows, reversed for its second layer
    /// (one element without pairs).
    pair_in: GpuBuffer,
}

impl Mamba3MimoGpuWindow {
    /// The tokens a run of this window covers per sequence.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Whether runs of this window on the state it was captured with
    /// replay a graph.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Drop the window's graph, once a replay in flight finished: its runs
    /// go eagerly and may be captured again.
    pub fn clear_graph(&mut self) {
        self.graph = None;
    }

    /// Every position's final-norm row of the last run, f32
    /// `[batch, seq_len, d_model]`, on the device.
    pub fn final_rows(&self) -> &GpuBuffer {
        &self.final_rows
    }

    /// The window's input rows, f32 `[batch, seq_len, input_dim]`, on the
    /// device.
    pub(crate) fn input_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.input.cached_ptr()
    }
}

/// The MIMO decode and prefill engine.
pub struct Mamba3MimoGpuEngine {
    pub kernels: Mamba3MimoKernels,
    pub ctx: GpuCtx,
    cfg: Mamba3MimoConfig,
    batch: usize,
    input_dim: usize,
    act: WeightDtype,
    layers: Vec<GpuMimoLayer>,
    input_proj: Option<(GpuBuffer, GpuBuffer)>,
    norm_f_weight: GpuBuffer,
    /// The final LayerNorm's bias, None for RMSNorm.
    norm_f_bias: Option<GpuBuffer>,
    /// The chunked scan's launch geometry on this device, `None` when one
    /// of its kernels cannot fit (the windows then run the recurrence).
    chunked: Option<ChunkedGeom>,
    /// The recurrence's launch: the whole head per block, or tiles of its
    /// channels.
    scan: ScanGeom,
    /// One counter per (sequence, head) for a head in tiles, so the last
    /// tile to finish carries the keys and angles out.
    scan_done: GpuByteBuffer,
}

/// A captured window: the graph, the GEMM route and its resolved launches,
/// and every device address the replay reuses. It holds the kernels'
/// module and the context's resources, so they outlive the graph.
struct CapturedWindow {
    graph: cudarc::driver::CudaGraph,
    gemm_route: crate::gpu::context::GemmRoute,
    gemm_plan: Option<CapturedGemmGraphPlan>,
    /// The state's four buffers.
    state: [u64; 4],
    half_staging: u64,
    bi_upcast: [u64; 3],
    _modules: crate::gpu::kernels::CudaModuleAnchors,
    ctx: std::rc::Rc<crate::gpu::context::GpuCtxResources>,
}

impl Drop for CapturedWindow {
    fn drop(&mut self) {
        // A replay may still be in flight: the graph goes once it finished.
        let _ = self.ctx.stream.synchronize();
    }
}

fn state_ptrs(state: &Mamba3MimoGpuState) -> [u64; 4] {
    [
        state.ssm.cached_ptr(),
        state.k.cached_ptr(),
        state.v.cached_ptr(),
        state.angle.cached_ptr(),
    ]
}

impl Mamba3MimoGpuEngine {
    /// An engine for `batch` sequences storing its projections in `dtype`
    /// (f32, tf32, bf16 or f16), in the GEMM mode `MAMBA_RS_GEMM_MODE`
    /// names (deterministic when unset). The arguments are those of the
    /// Mamba-1 and SISO mixed engines.
    pub fn new(
        device: &GpuDevice,
        w: &Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        input_dim: usize,
        batch: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::new_inner(device, w, (cfg, input_dim, batch), (dtype, None))
    }

    /// [`Self::new`] with an explicit GEMM mode.
    pub fn new_with_mode(
        device: &GpuDevice,
        w: &Mamba3MimoWeights,
        cfg: Mamba3MimoConfig,
        input_dim: usize,
        batch: usize,
        dtype: WeightDtype,
        mode: GemmMode,
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
        let request = Request {
            path: Path::Decode,
            batch,
            seq_len: 1,
            scan: cfg.scan_mode,
            precision: dtype,
            gemm: gemm_mode_or_env(mode)?,
        };
        crate::mamba3_mimo::route::admit(&cfg, input_dim, &request)?;
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
        let kernels = Mamba3MimoKernels::compile(device.context(), arch, cfg.mimo_rank)?;
        let (hd, ds, rank) = (cfg.headdim, cfg.d_state, cfg.mimo_rank);
        let scan = scan_geom(
            (hd, ds, rank, cfg.num_rope_angles()),
            kernels.smem_optin,
            kernels.scan_threads,
        )
        .ok_or_else(|| {
            format!(
                "not one channel of a rank-{rank} scan with {ds} states fits this GPU's {} \
                 bytes of shared memory",
                kernels.smem_optin
            )
        })?;
        let chunked = chunked_geom(
            (hd, ds, rank),
            kernels.smem_optin,
            false,
            kernels.chunk_threads,
        );
        if cfg.scan_mode == ScanMode::Parallel && chunked.is_none() {
            return Err(format!(
                "the chunked scan of a rank-{rank} head of {hd} channels and {ds} states does \
                 not fit this GPU's {} bytes of shared memory; ScanMode::Auto runs it by the \
                 recurrence",
                kernels.smem_optin
            ));
        }
        let s = &ctx.stream;
        let up = |v: &[f32]| GpuBuffer::from_cpu(s, v);
        let opt = |v: &[f32]| -> Result<Option<GpuBuffer>, String> {
            if v.is_empty() {
                Ok(None)
            } else {
                up(v).map(Some)
            }
        };
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
                    norm_bias: opt(&m.norm_bias)?,
                    fc1_w: typed(&m.fc1_w)?,
                    fc2_w: typed(&m.fc2_w)?,
                    hidden: m.hidden(),
                }),
            };
            Ok(GpuMimoLayer {
                norm_weight: up(&lw.norm_weight)?,
                norm_bias: opt(&lw.norm_bias)?,
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
                norm_gate: if cfg.is_outproj_norm {
                    Some(up(&lw.norm_gate_weight)?)
                } else {
                    None
                },
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
        let norm_f_bias = opt(&w.norm_f_bias)?;
        let scan_done = GpuByteBuffer::zeros(s, (batch * cfg.nheads() * 4).max(4))?;
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
            norm_f_bias,
            chunked,
            scan,
            scan_done,
        })
    }

    /// Whether a window of `t_len` tokens runs the chunked scan: the
    /// config's choice, when the chunked kernels fit this device.
    fn use_chunked(&self, t_len: usize) -> bool {
        t_len > 1 && self.cfg.prefill_parallel(t_len) && self.chunked.is_some()
    }

    pub fn ctx(&self) -> &GpuCtx {
        &self.ctx
    }

    /// The model's configuration.
    pub fn config(&self) -> &Mamba3MimoConfig {
        &self.cfg
    }

    /// Sequences per call.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Input features per token.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// A host slice must hold `rows * width` values for this engine.
    fn check_host_len(what: &str, len: usize, want: usize) -> Result<(), String> {
        if len == want {
            Ok(())
        } else {
            Err(format!(
                "{what} holds {len} values; this engine takes {want}"
            ))
        }
    }

    /// A state of this engine's batch and model: another engine's would be
    /// read and written past its end.
    fn check_state(&self, st: &Mamba3MimoGpuState) -> Result<(), String> {
        let c = &self.cfg;
        let (l, b, nh) = (c.n_layers, self.batch, c.nheads());
        if st.ssm.len() != l * b * nh * c.headdim * c.d_state
            || st.k.len() != l * b * nh * c.mimo_rank * c.d_state
            || st.v.len() != l * b * nh * c.headdim
            || st.angle.len() != (l * b * nh * c.num_rope_angles()).max(1)
        {
            return Err("the state was not allocated by this engine".into());
        }
        Ok(())
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
        let request = Request {
            path: if seq_len == 1 {
                Path::Decode
            } else {
                Path::Prefill
            },
            batch: self.batch,
            seq_len,
            scan: c.scan_mode,
            precision: self.act,
            gemm: self.ctx.gemm_mode(),
        };
        crate::mamba3_mimo::route::admit(c, self.input_dim, &request)?;
        let rows = self.batch * seq_len;
        let s = &self.ctx.stream;
        let z = |n: usize| GpuBuffer::zeros(s, n.max(1));
        let typed = |n: usize| DtypedBuf::zeros(s, n.max(1), self.act);
        let hidden = c.mlp_hidden();
        // A decode step (one token) always runs the recurrence.
        let chunked = self.use_chunked(seq_len);
        let ch = |n: usize| if chunked { n } else { 1 };
        let (nh, na) = (c.nheads(), c.num_rope_angles());
        let nc = seq_len.div_ceil(CHUNK);
        Ok(Mamba3MimoGpuWindow {
            graph: None,
            eager_gemm_manifest: None,
            seq_len,
            input: z(rows * self.input_dim)?,
            temporal: z(rows * c.d_model)?,
            post: typed(rows * c.d_model)?,
            rms: z(rows * c.block_norm.stats_per_row())?,
            proj: typed(rows * c.in_proj_out_dim())?,
            bn: z(rows * c.bc_width())?,
            cn: z(rows * c.bc_width())?,
            gated: typed(rows * c.d_inner())?,
            y: z(if c.is_outproj_norm {
                rows * c.mimo_rank * c.d_inner()
            } else {
                1
            })?,
            out: typed(rows * c.d_model)?,
            mlp_up: typed(rows * 2 * hidden)?,
            mlp_act: typed(rows * hidden)?,
            final_rows: z(rows * c.d_model)?,
            pair_in: z(if c.bidirectional { rows * c.d_model } else { 1 })?,
            coef: z(ch(rows * nh * 4))?,
            sn: z(ch(rows * nh * na))?,
            cs: z(ch(rows * nh * na))?,
            e0: z(ch(self.batch * nh))?,
            contrib: z(ch(self.batch * nc * nh * c.headdim * c.d_state))?,
            enter: z(ch(self.batch * nc * nh * c.headdim * c.d_state))?,
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
        let rows = self.batch * win.seq_len;
        Self::check_host_len("output", rows_out.len(), rows * self.cfg.d_model)?;
        self.run_gpu_only(input, state, win)?;
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

    /// [`Self::run`] without the download: every position's final-norm
    /// row stays in [`Mamba3MimoGpuWindow::final_rows`] on the stream, for a
    /// consumer on the GPU such as a language-model head.
    pub fn run_gpu_only(
        &self,
        input: &[f32],
        state: &mut Mamba3MimoGpuState,
        win: &mut Mamba3MimoGpuWindow,
    ) -> Result<(), String> {
        self.run_from(RowSource::Host(input), state, win)
    }

    /// [`Self::run_gpu_only`] on rows from the host or already in the
    /// window's input buffer.
    pub(crate) fn run_from(
        &self,
        input: RowSource<'_>,
        state: &mut Mamba3MimoGpuState,
        win: &mut Mamba3MimoGpuWindow,
    ) -> Result<(), String> {
        let rows = self.batch * win.seq_len;
        if let Some(input) = input.host() {
            Self::check_host_len("input", input.len(), rows * self.input_dim)?;
        }
        Self::check_host_len(
            "window buffers",
            win.input.len(),
            (rows * self.input_dim).max(1),
        )?;
        self.check_state(state)?;
        if let Some(input) = input.host() {
            win.input.upload(&self.ctx.stream, input)?;
        }
        if let Some(g) = &win.graph
            && g.state == state_ptrs(state)
        {
            return self.replay(g);
        }
        if self.has_gemm_work() {
            prepare_inference_arch_rung(&self.ctx)?;
        }
        self.presize_gemm_scratch(self.batch * win.seq_len)?;
        win.eager_gemm_manifest = None;
        let manifest = self
            .ctx
            .record_eager_gemm_manifest(|| self.kernels_on_stream(state, win))?;
        win.eager_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// Copy each sequence's last row of `from`'s final rows into `to`'s
    /// first row slots: the prompt's last position of a prefill window,
    /// placed where a decode step's window leaves its output.
    pub fn gather_last_rows(
        &self,
        from: &Mamba3MimoGpuWindow,
        to: &mut Mamba3MimoGpuWindow,
    ) -> Result<(), String> {
        if from.seq_len == 0 || self.batch == 0 {
            return Ok(());
        }
        Self::check_host_len(
            "last-row window",
            to.final_rows.len(),
            self.batch * self.cfg.d_model,
        )?;
        let (b, t, dm) = (
            self.batch as i32,
            from.seq_len as i32,
            self.cfg.d_model as i32,
        );
        let mut bld = self
            .ctx
            .stream
            .launch_builder(&self.ctx.kernels.gather_last_timestep);
        bld.arg(to.final_rows.inner_mut());
        bld.arg(from.final_rows.inner());
        bld.arg(&b);
        bld.arg(&t);
        bld.arg(&dm);
        unsafe { bld.launch(grid_1d_cover(self.batch * self.cfg.d_model)?) }
            .map_err(|e| format!("MIMO last rows: {e:?}"))?;
        Ok(())
    }

    /// Whether a window runs any GEMM: the input projection or a layer.
    fn has_gemm_work(&self) -> bool {
        self.batch != 0 && (self.input_proj.is_some() || self.cfg.n_layers != 0)
    }

    /// Size the context's GEMM staging and upcast scratch for `rows` rows
    /// through every projection, before a window's first GEMM.
    fn presize_gemm_scratch(&self, rows: usize) -> Result<(), String> {
        let c = &self.cfg;
        let (dm, ip, di, h) = (c.d_model, c.in_proj_out_dim(), c.d_inner(), c.mlp_hidden());
        let width = dm.max(self.input_dim).max(ip).max(di).max(2 * h);
        let weight = (self.input_dim * dm)
            .max(dm * ip)
            .max(di * dm)
            .max(dm * 2 * h);
        self.ctx.presize_gemm_scratch(rows, width, weight, self.act)
    }

    /// Capture the kernels of `win` (a decode step, `alloc_window(1)`, or a
    /// prompt window) as one CUDA graph, held by the window; later
    /// [`Self::run`] calls of this window on this state replay it, while
    /// the window on another state, and every other window, run eagerly.
    ///
    /// Takes the GEMM inventory of the eager run of this window that must
    /// precede it, and replaces the window's earlier graph. The replay
    /// refuses a changed GEMM route or moved GEMM staging. After capture the
    /// context's GEMM staging can no longer grow: run and capture the
    /// longest window first.
    ///
    /// # Safety
    ///
    /// `state` must stay allocated, and the engine's weights, kernels and
    /// context unchanged, for as long as the graph replays.
    pub unsafe fn capture_graph(
        &self,
        state: &mut Mamba3MimoGpuState,
        win: &mut Mamba3MimoGpuWindow,
    ) -> Result<(), String> {
        self.check_state(state)?;
        let manifest = win.eager_gemm_manifest.take().ok_or_else(|| {
            "MIMO graph capture needs a successful eager run of the window first".to_string()
        })?;
        win.graph = None;
        self.ctx.presize_bi_scratch()?;
        let gemm_route = self.ctx.gemm_route();
        self.ctx.freeze_graph_scratch();
        let (graph, gemm_plan) = unsafe {
            capture_into_graph_with_gemm_plan(&self.ctx, manifest.route_capacity, &manifest, || {
                self.kernels_on_stream(state, win)
            })
        }?;
        require_deterministic_gemm_graph_plan(
            &self.ctx,
            self.has_gemm_work(),
            gemm_plan.as_ref(),
            "MIMO graph capture",
        )?;
        win.graph = Some(CapturedWindow {
            graph,
            gemm_route,
            gemm_plan,
            state: state_ptrs(state),
            half_staging: self.ctx.half_staging_ptr(),
            bi_upcast: self.ctx.bi_upcast_scratch_ptrs(),
            _modules: self.kernels.module_anchors(),
            ctx: self.ctx.resource_anchor(),
        });
        self.ctx.note_graph_capture();
        Ok(())
    }

    fn replay(&self, captured: &CapturedWindow) -> Result<(), String> {
        const LABEL: &str = "MIMO graph replay";
        if captured.gemm_route != self.ctx.gemm_route() {
            return Err(format!("{LABEL}: the GEMM route changed since capture"));
        }
        self.ctx
            .ensure_graph_scratch_ptrs(captured.half_staging, captured.bi_upcast, LABEL)?;
        with_validated_gemm_graph_launch(
            &self.ctx,
            self.has_gemm_work(),
            captured.gemm_plan.as_ref(),
            LABEL,
            || {
                captured
                    .graph
                    .launch()
                    .map_err(|e| format!("{LABEL}: {e:?}"))
            },
        )
    }

    fn typed_ptr(&self, ptr: u64) -> TypedPtr {
        TypedPtr {
            ptr,
            dtype: self.act,
        }
    }

    /// A block norm of f32 rows into `y`: in the activation dtype when
    /// `typed`, else f32; `stats` takes each row's statistics.
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

    /// One layer's scan over the window by the chunked path.
    fn chunked_scan(
        &self,
        lw: &GpuMimoLayer,
        (ssm, kst, vst, ang): (u64, u64, u64, u64),
        win: &Mamba3MimoGpuWindow,
        l: usize,
    ) -> Result<(), String> {
        let ptrs = MimoChunkedPtrs {
            proj: win.proj.cached_ptr(),
            bn: win.bn.cached_ptr(),
            cn: win.cn.cached_ptr(),
            out: win.gated.cached_ptr(),
            y_out: if lw.norm_gate.is_some() {
                win.y.cached_ptr()
            } else {
                0
            },
            coef: win.coef.cached_ptr(),
            sn: win.sn.cached_ptr(),
            cs: win.cs.cached_ptr(),
            e0: win.e0.cached_ptr(),
            contrib: win.contrib.cached_ptr(),
            enter: win.enter.cached_ptr(),
            ssm,
            kst,
            vst,
            ang,
            dt_bias: lw.dt_bias.cached_ptr(),
            b_bias: lw.b_bias.cached_ptr(),
            c_bias: lw.c_bias.cached_ptr(),
            d: lw.d_param.cached_ptr(),
            mimo_x: lw.mimo_x.cached_ptr(),
            mimo_z: lw.mimo_z.cached_ptr(),
            mimo_o: lw.mimo_o.cached_ptr(),
            seq_idx: NO_PACKING,
        };
        let x = MimoExec {
            ctx: &self.ctx,
            kernels: &self.kernels,
            cfg: &self.cfg,
            act: self.act,
        };
        let geom = self
            .chunked
            .ok_or("the chunked scan does not fit this GPU")?;
        launch_chunked_scan(x, &ptrs, &geom, (self.batch, win.seq_len), l)
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
        let bidi = c.bidirectional;
        // A bidirectional model's window is whole sequences from the empty
        // state: it carries nothing in or out.
        if bidi {
            for st in [&state.ssm, &state.k, &state.v, &state.angle] {
                zero_bytes(&self.ctx, st.cached_ptr(), st.len() * 4)?;
            }
        }
        let pair_in = win.pair_in.cached_ptr();
        let window = |ptr, elem_bytes| WindowRows {
            ptr,
            elem_bytes,
            batch: b,
            seq_len: t_len,
            width: dm,
        };
        for (l, lw) in self.layers.iter().enumerate() {
            // A pair's first layer keeps the pair's input; its second reads
            // it reversed, and its output turns back before the add.
            let reversed = reads_reversed(bidi, l);
            if opens_pair(bidi, l) {
                win.pair_in.copy_from_raw(&win.temporal, stream)?;
            }
            let norm_in = if reversed {
                reverse_rows(&self.ctx, window(pair_in, 4), 0)?;
                pair_in
            } else {
                temporal
            };
            self.block_norm(
                (post, norm_in, rms),
                norm_params(&lw.norm_weight, &lw.norm_bias),
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
            let (wb, wc) = (lw.b_norm_weight.cached_ptr(), lw.c_norm_weight.cached_ptr());
            for (first, n) in row_pieces(rows, r * ng) {
                let normed = (first * r * ng * ds * 4) as u64;
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
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((n * r * ng) as u32, 2, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { bld.launch(cfg) }.map_err(|e| format!("MIMO B/C norm L{l}: {e:?}"))?;
            }
            let states = (
                state.ssm.cached_ptr() + l as u64 * per(hd * ds),
                state.k.cached_ptr() + l as u64 * per(r * ds),
                state.v.cached_ptr() + l as u64 * per(hd),
                state.angle.cached_ptr() + l as u64 * per(na),
            );
            if self.use_chunked(t_len) {
                self.chunked_scan(lw, states, win, l)?;
            } else {
                let (ssm, kst, vst, ang) = states;
                let ptrs = [
                    lw.dt_bias.cached_ptr(),
                    lw.b_bias.cached_ptr(),
                    lw.c_bias.cached_ptr(),
                    lw.d_param.cached_ptr(),
                    lw.mimo_x.cached_ptr(),
                    lw.mimo_z.cached_ptr(),
                    lw.mimo_o.cached_ptr(),
                ];
                for (y_first, z_first, cfg) in yz_runs(self.scan.cfg(b, nh, hd)) {
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
                    let (tile, pipe) = (self.scan.tile as i32, i32::from(self.scan.pipe));
                    let done = self.scan_done.cached_ptr();
                    let y_out = if lw.norm_gate.is_some() {
                        win.y.cached_ptr()
                    } else {
                        0u64
                    };
                    bld.arg(&tile);
                    bld.arg(&pipe);
                    bld.arg(&done);
                    bld.arg(&y_out);
                    bld.arg(&y_first);
                    bld.arg(&z_first);
                    bld.arg(&NO_PACKING);
                    unsafe { bld.launch(cfg) }.map_err(|e| format!("MIMO scan L{l}: {e:?}"))?;
                }
            }
            // The output norm gates the scans' readouts again, over the
            // plain gate's rows.
            if let Some(nw) = &lw.norm_gate {
                let (y, nw, mz, mo) = (
                    win.y.cached_ptr(),
                    nw.cached_ptr(),
                    lw.mimo_z.cached_ptr(),
                    lw.mimo_o.cached_ptr(),
                );
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((b * t_len) as u32, nh as u32, 1),
                    block_dim: (hd.next_multiple_of(32).min(256) as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                for (y_first, cfg) in y_runs(cfg) {
                    let mut bld = stream.launch_builder(self.kernels.gate_normed.get(self.act));
                    bld.arg(&gated);
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
            gpu_gemm_typed_forward_raw(
                &self.ctx,
                self.typed_ptr(out),
                self.typed_ptr(gated),
                self.typed_ptr(lw.out_proj_w.cached_ptr()),
                None,
                (rows, di, dm),
            )?;
            if reversed {
                reverse_rows(&self.ctx, window(out, self.act.size_bytes()), 0)?;
            }
            self.residual_add(temporal, out, rows * dm)?;
            // A pair's MLPs follow both its mixers.
            for m_l in mlps_after(bidi, l) {
                let Some(m) = &self.layers[m_l].mlp else {
                    continue;
                };
                let (upp, actp) = (win.mlp_up.cached_ptr(), win.mlp_act.cached_ptr());
                self.block_norm(
                    (post, temporal, rms),
                    norm_params(&m.norm_weight, &m.norm_bias),
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
                unsafe { bld.launch(grid_rows(rows, m.hidden)) }
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
        self.block_norm(
            (win.final_rows.cached_ptr(), temporal, rms),
            norm_params(&self.norm_f_weight, &self.norm_f_bias),
            (rows, false),
        )
    }
}

/// A block norm's parameters on the device; an empty scale is no norm at
/// all, the final norm of a backbone built without it.
fn norm_params(w: &GpuBuffer, b: &Option<GpuBuffer>) -> NormParams {
    NormParams {
        w: if w.is_empty() { 0 } else { w.cached_ptr() },
        b: b.as_ref().map_or(0, GpuBuffer::cached_ptr),
    }
}

/// Pieces of whole rows, `(first row, rows)`, for a launch with
/// `blocks_per_row` blocks per row on the grid's x axis: at most its extent
/// of blocks each. The blocks of these launches are independent, so the
/// pieces write what one launch over every row would.
pub(crate) fn row_pieces(
    rows: usize,
    blocks_per_row: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let step = (i32::MAX as usize / blocks_per_row.max(1)).max(1);
    (0..rows)
        .step_by(step)
        .map(move |first| (first, step.min(rows - first)))
}

/// The context, module, shape and activation dtype a MIMO launch needs,
/// shared by the engine and the trainer.
#[derive(Clone, Copy)]
pub(crate) struct MimoExec<'a> {
    pub ctx: &'a GpuCtx,
    pub kernels: &'a Mamba3MimoKernels,
    pub cfg: &'a Mamba3MimoConfig,
    pub act: WeightDtype,
}

/// One layer's chunked scan, as device pointers: its rows, its scratch,
/// the states it carries (this layer's slices) and its weights. `y_out` (0
/// for none) receives the readouts the training backward reads.
#[derive(Clone, Copy)]
pub(crate) struct MimoChunkedPtrs {
    pub proj: u64,
    pub bn: u64,
    pub cn: u64,
    pub out: u64,
    pub y_out: u64,
    pub coef: u64,
    pub sn: u64,
    pub cs: u64,
    pub e0: u64,
    pub contrib: u64,
    pub enter: u64,
    pub ssm: u64,
    pub kst: u64,
    pub vst: u64,
    pub ang: u64,
    pub dt_bias: u64,
    pub b_bias: u64,
    pub c_bias: u64,
    pub d: u64,
    pub mimo_x: u64,
    pub mimo_z: u64,
    pub mimo_o: u64,
    /// The window's packing (see [`crate::varlen`]), 0 for none.
    pub seq_idx: u64,
}

/// One layer's scan over a window of `t_len` tokens of `b` sequences by the
/// chunked path: the step scalars and angles, each chunk's state, the
/// state passing (from the carried state with the first token's lead
/// term), the outputs and the gate into `ptrs.out`, the carried keys and
/// input.
pub(crate) fn launch_chunked_scan(
    x: MimoExec<'_>,
    ptrs: &MimoChunkedPtrs,
    geom: &ChunkedGeom,
    (b, t_len): (usize, usize),
    l: usize,
) -> Result<(), String> {
    let c = x.cfg;
    let nc = t_len.div_ceil(CHUNK);
    let (di, bc) = (c.d_inner(), c.bc_width());
    let (nh, hd, ds, r, ng, na) = (
        c.nheads(),
        c.headdim,
        c.d_state,
        c.mimo_rank,
        c.ngroups,
        c.num_rope_angles(),
    );
    let heads = 2 * di + 2 * bc;
    let ang_col = heads + 3 * nh;
    let ld_proj = c.in_proj_out_dim() as i64;
    let stream = &x.ctx.stream;
    let MimoChunkedPtrs {
        proj,
        bn,
        cn,
        out: gated,
        y_out,
        coef,
        sn,
        cs,
        e0,
        contrib,
        enter,
        ssm,
        kst,
        vst,
        ang,
        dt_bias,
        b_bias,
        c_bias,
        d,
        mimo_x,
        mimo_z,
        mimo_o,
        seq_idx,
    } = *ptrs;
    let (seq_i, nh_i, ng_i, hd_i, ds_i, r_i, na_i) = (
        t_len as i32,
        nh as i32,
        ng as i32,
        hd as i32,
        ds as i32,
        r as i32,
        na as i32,
    );
    let (heads_i, ang_col_i, b_i) = (heads as i32, ang_col as i32, b as i32);
    let a_floor = c.a_floor;
    let err = |what: &'static str| move |e| format!("MIMO chunked {what} L{l}: {e:?}");
    {
        let dtb = dt_bias;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (b as u32, nh as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        for (y_first, cfg) in y_runs(cfg) {
            let mut bld = stream.launch_builder(x.kernels.chunk_prep.get(x.act));
            bld.arg(&coef);
            bld.arg(&sn);
            bld.arg(&cs);
            bld.arg(&e0);
            bld.arg(&ang);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&dtb);
            bld.arg(&seq_i);
            bld.arg(&nh_i);
            bld.arg(&heads_i);
            bld.arg(&ang_col_i);
            bld.arg(&na_i);
            bld.arg(&a_floor);
            bld.arg(&y_first);
            bld.arg(&seq_idx);
            unsafe { bld.launch(cfg) }.map_err(err("prep"))?;
        }
    }
    if geom.state_tiles.0 > 0 {
        let (jc, sl, pg, smem) = geom.state_tiles;
        let (bb, px) = (b_bias, mimo_x);
        let (jc_i, sl_i, pg_i) = (jc as i32, sl as i32, pg as i32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                (b * nc) as u32,
                nh as u32,
                (ds.div_ceil(32 * jc) * hd.div_ceil(pg)) as u32,
            ),
            block_dim: ((8 * pg) as u32, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        for (y_first, z_first, cfg) in yz_runs(cfg) {
            let mut bld = stream.launch_builder(x.kernels.chunk_state_tiles.get(x.act));
            bld.arg(&contrib);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&bn);
            bld.arg(&coef);
            bld.arg(&sn);
            bld.arg(&cs);
            bld.arg(&bb);
            bld.arg(&px);
            bld.arg(&seq_i);
            bld.arg(&nh_i);
            bld.arg(&ng_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.arg(&r_i);
            bld.arg(&na_i);
            bld.arg(&jc_i);
            bld.arg(&sl_i);
            bld.arg(&pg_i);
            bld.arg(&y_first);
            bld.arg(&z_first);
            bld.arg(&seq_idx);
            unsafe { bld.launch(cfg) }.map_err(err("chunk state"))?;
        }
    } else {
        let (nt, sl, smem) = geom.state;
        let (bb, px, nt_i, sl_i) = (b_bias, mimo_x, nt as i32, sl as i32);
        let pt_i = geom.pt as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                (b * nc) as u32,
                nh as u32,
                (ds.div_ceil(nt) * hd.div_ceil(geom.pt)) as u32,
            ),
            block_dim: ((nt * geom.pt) as u32, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        for (y_first, z_first, cfg) in yz_runs(cfg) {
            let mut bld = stream.launch_builder(x.kernels.chunk_state.get(x.act));
            bld.arg(&contrib);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&bn);
            bld.arg(&coef);
            bld.arg(&sn);
            bld.arg(&cs);
            bld.arg(&bb);
            bld.arg(&px);
            bld.arg(&seq_i);
            bld.arg(&nh_i);
            bld.arg(&ng_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.arg(&r_i);
            bld.arg(&na_i);
            bld.arg(&nt_i);
            bld.arg(&sl_i);
            bld.arg(&pt_i);
            bld.arg(&y_first);
            bld.arg(&z_first);
            bld.arg(&seq_idx);
            unsafe { bld.launch(cfg) }.map_err(err("chunk state"))?;
        }
    }
    {
        let px = mimo_x;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((b * nh) as u32, (hd * ds).div_ceil(256) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        for (y_first, cfg) in y_runs(cfg) {
            let mut bld = stream.launch_builder(&x.kernels.state_pass);
            bld.arg(&enter);
            bld.arg(&ssm);
            bld.arg(&contrib);
            bld.arg(&coef);
            bld.arg(&e0);
            bld.arg(&kst);
            bld.arg(&vst);
            bld.arg(&px);
            bld.arg(&b_i);
            bld.arg(&seq_i);
            bld.arg(&nh_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.arg(&r_i);
            bld.arg(&y_first);
            bld.arg(&seq_idx);
            unsafe { bld.launch(cfg) }.map_err(err("state passing"))?;
        }
    }
    if geom.scan_tiles.0 > 0 {
        let (tq, kt, nk, smem) = geom.scan_tiles;
        let w = [b_bias, c_bias, d, mimo_x, mimo_z, mimo_o];
        let (tq_i, kt_i, nk_i) = (tq as i32, kt as i32, nk as i32);
        let cfg = chunk_scan_tiles_cfg((b, nc, nh), (hd, r), tq, smem);
        for (y_first, cfg) in y_runs(cfg) {
            let mut bld = stream.launch_builder(x.kernels.chunk_scan_tiles.get(x.act));
            bld.arg(&gated);
            bld.arg(&y_out);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&bn);
            bld.arg(&cn);
            bld.arg(&coef);
            bld.arg(&sn);
            bld.arg(&cs);
            bld.arg(&enter);
            for p in &w {
                bld.arg(p);
            }
            bld.arg(&seq_i);
            bld.arg(&nh_i);
            bld.arg(&ng_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.arg(&r_i);
            bld.arg(&na_i);
            bld.arg(&tq_i);
            bld.arg(&kt_i);
            bld.arg(&nk_i);
            bld.arg(&y_first);
            bld.arg(&seq_idx);
            unsafe { bld.launch(cfg) }.map_err(err("scan"))?;
        }
    } else {
        let (tt, st, smem) = geom.scan;
        let w = [b_bias, c_bias, d, mimo_x, mimo_z, mimo_o];
        let (tt_i, st_i, pt_i) = (tt as i32, st as i32, geom.pt as i32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                (b * nc) as u32,
                nh as u32,
                (CHUNK.div_ceil(tt) * hd.div_ceil(geom.pt)) as u32,
            ),
            // Whole warps only: the tile products leave a partial warp out,
            // so a block under 32 threads would leave G unwritten. Threads
            // past `tt * pt` own no output, and each product keeps the one
            // chain its owner runs, so the rounding moves no bits.
            block_dim: ((tt * geom.pt).next_multiple_of(32) as u32, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        for (y_first, z_first, cfg) in yz_runs(cfg) {
            let mut bld = stream.launch_builder(x.kernels.chunk_scan.get(x.act));
            bld.arg(&gated);
            bld.arg(&y_out);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&bn);
            bld.arg(&cn);
            bld.arg(&coef);
            bld.arg(&sn);
            bld.arg(&cs);
            bld.arg(&enter);
            for p in &w {
                bld.arg(p);
            }
            bld.arg(&seq_i);
            bld.arg(&nh_i);
            bld.arg(&ng_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.arg(&r_i);
            bld.arg(&na_i);
            bld.arg(&tt_i);
            bld.arg(&st_i);
            bld.arg(&pt_i);
            bld.arg(&y_first);
            bld.arg(&z_first);
            bld.arg(&seq_idx);
            unsafe { bld.launch(cfg) }.map_err(err("scan"))?;
        }
    }
    {
        let bb = b_bias;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (b as u32, nh as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        for (y_first, cfg) in y_runs(cfg) {
            let mut bld = stream.launch_builder(x.kernels.chunk_writeback.get(x.act));
            bld.arg(&kst);
            bld.arg(&vst);
            bld.arg(&proj);
            bld.arg(&ld_proj);
            bld.arg(&bn);
            bld.arg(&sn);
            bld.arg(&cs);
            bld.arg(&bb);
            bld.arg(&seq_i);
            bld.arg(&nh_i);
            bld.arg(&ng_i);
            bld.arg(&hd_i);
            bld.arg(&ds_i);
            bld.arg(&r_i);
            bld.arg(&na_i);
            bld.arg(&y_first);
            unsafe { bld.launch(cfg) }.map_err(err("writeback"))?;
        }
    }
    Ok(())
}
