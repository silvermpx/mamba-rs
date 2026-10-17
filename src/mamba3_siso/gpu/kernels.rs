//! Compile and register Mamba-3 SISO CUDA kernels.
//!
//! The Mamba-3 kernel registry, compiled via NVRTC at runtime and cached in
//! the same verified artifact format as the Mamba-1 registry.
//! Separate from Mamba SSM's `MambaKernels` — different pipeline, no conv1d.

use super::transport::{TransportAdmission, TransportKernels, transport_environment_admitted};
use crate::gpu::dtype::WeightDtype;
use crate::gpu::kernels::{ByIndexWidth, CudaModuleAnchors, HalfKernel, TypedKernel};
use crate::gpu::launch::{IndexWidth, opt_in_dynamic_smem, smem_budget};
use crate::mamba3_siso::route::{HeadTier, PLANNED_SMEM, head_tier};
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};
use std::sync::Arc;

/// The launch shape of the sequential scan's head kernels.
#[derive(Clone, Copy, Debug)]
pub struct SequentialShape {
    pub batch: usize,
    pub nheads: usize,
    pub headdim: usize,
    pub d_state: usize,
}

impl SequentialShape {
    /// One block per (sequence, head): the head's width in threads on the
    /// warp tier, rounded up to whole warps on the block tier, whose kernels
    /// take a wider head than their launch bound in slices.
    fn launch(&self, warp: bool, shared_mem_bytes: usize) -> LaunchConfig {
        let threads = if warp {
            self.headdim
        } else {
            crate::mamba3_siso::route::block_tier_threads(self.headdim)
        };
        LaunchConfig {
            grid_dim: (self.batch as u32, self.nheads as u32, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        }
    }
}

/// All compiled Mamba-3 SISO CUDA kernels.
pub struct Mamba3Kernels {
    _modules: CudaModuleAnchors,
    compiler_identity: crate::gpu::kernel_identity::CompilerIdentity,
    artifact_identity: crate::gpu::kernel_identity::ArtifactIdentity,
    /// Identity of the compiled M3 module (NVRTC cache key); pinned by
    /// the M3 graph guards.
    pub module_identity: String,

    /// State-dimension capacity the kernels were compiled with (the
    /// per-thread register-array size). The engine and trainer
    /// constructors derive it from the model config, so a mismatched
    /// launch cannot be built through the public constructors; the M1
    /// launch-path asserts additionally compare against it, and the
    /// kernels carry their own capacity guards.
    pub state_cap: usize,
    pub(super) transport: TransportKernels,

    // ── Sequential SSM (mamba3_siso.cu) ──
    pub m3_step_fwd: ByIndexWidth<CudaFunction>,
    pub m3_burnin_fwd: ByIndexWidth<CudaFunction>,
    /// The burn-in forward specialised on a state width, the twin of
    /// [`Self::m3_backward_seq_by_state`].
    pub m3_burnin_fwd_by_state: [(usize, ByIndexWidth<CudaFunction>); 4],
    pub m3_burnin_fwd_nosave: ByIndexWidth<CudaFunction>,
    pub m3_backward_seq: ByIndexWidth<CudaFunction>,
    /// The backward scan specialised on a state width, so its state loops
    /// carry a constant trip count. One per width the models here use; a
    /// width with no entry runs the general `m3_backward_seq` above.
    pub m3_backward_seq_by_state: [(usize, ByIndexWidth<CudaFunction>); 4],
    /// The sequential training forward for heads the warp kernels cannot
    /// hold: one CTA per head, the step's rows staged in shared memory.
    pub m3_burnin_fwd_block: TypedKernel,
    /// The target network's sequential forward on the block tier.
    pub m3_burnin_fwd_nosave_block: CudaFunction,
    /// The sequential backward on the block tier: the warp kernel's tree in
    /// every warp, the warps' sums added in order. It also serves every
    /// head of the bf16 and f16 sequential backward.
    pub m3_backward_seq_block: TypedKernel,
    pub m3_reduce_d_d: ByIndexWidth<CudaFunction>,

    // ── Shared ops (mamba3_ops.cu) ──
    pub m3_split: ByIndexWidth<CudaFunction>,
    pub m3_split_bwd: ByIndexWidth<CudaFunction>,
    pub bcnorm_bwd: ByIndexWidth<CudaFunction>,
    pub bc_bias_add: ByIndexWidth<CudaFunction>,
    pub bc_bias_add_bwd: ByIndexWidth<CudaFunction>,
    pub m3_angle_dt_fwd_seq: ByIndexWidth<CudaFunction>,
    /// Chunk-parallel angle accumulation pair — replaces the sequential
    /// kernel on multi-chunk windows (its serial fp64 chain dominates
    /// the prefill profile at production shapes).
    pub m3_angle_chunk_sums: ByIndexWidth<CudaFunction>,
    pub m3_angle_chunk_apply: ByIndexWidth<CudaFunction>,
    pub m3_angle_dt_bwd_seq: ByIndexWidth<CudaFunction>,
    pub rope_fwd: ByIndexWidth<CudaFunction>,
    /// Fused bc_bias_add (B + C) + rope_fwd - one launch replaces the
    /// F4c/F4d/F4ef triple; biased saves still materialize for backward.
    pub m3_bias_rope_fwd: ByIndexWidth<CudaFunction>,
    pub rope_bwd: ByIndexWidth<CudaFunction>,
    pub m3_compute_abg: ByIndexWidth<CudaFunction>,
    pub m3_abg_bwd: ByIndexWidth<CudaFunction>,
    pub silu_gate_fwd: ByIndexWidth<CudaFunction>,
    pub silu_gate_bwd: ByIndexWidth<CudaFunction>,
    pub rmsnorm_gated_fwd: ByIndexWidth<CudaFunction>,
    pub rmsnorm_gated_bwd: ByIndexWidth<CudaFunction>,

    // ── Shared kernels from elementwise.cu (used by training pipeline) ──
    pub colsum_accumulate: CudaFunction,
    /// Segmented column sum - the batched pooled route.
    pub colsum_segments: CudaFunction,
    /// Deterministic axis-0 reduction — finalises Rule-B per-sample partials
    /// into f32 master-grad slots. Replaces atomicAdd accumulators used in
    /// M3 backward (dD from m3_dqkv, d_angles_raw/d_dt_angle from
    /// m3_angle_dt_bwd_seq, and the block norms' scale partials).
    pub reduce_sum_axis0: CudaFunction,
    /// The angle gradients' axis-0 reduction once a partial row holds more
    /// columns than the 32-bit range: `reduce_sum_axis0`'s order per column,
    /// in launches from a column offset.
    pub reduce_sum_axis0_wide: CudaFunction,
    pub vec_add_inplace: CudaFunction,
    pub elementwise_mul: CudaFunction,
    /// 16-byte vectorized twin of `elementwise_mul` for f32 operands: the
    /// same multiply per element, four elements per thread. Launched when
    /// `vec8_ok` holds for the count and every operand pointer.
    pub elementwise_mul_v: CudaFunction,
    pub fill_scalar: CudaFunction,
    pub cast_f32_to_bf16: CudaFunction,
    pub cast_f32_to_f16: CudaFunction,
    pub cast_bf16_to_f32: CudaFunction,
    pub cast_f16_to_f32: CudaFunction,
    pub residual_add: CudaFunction,
    pub gather_last_timestep: CudaFunction,

    // ── AdamW optimizer (adamw.cu) ──
    pub adamw_step_f32: CudaFunction,
    /// CUDA-Graph-capturable variant: bias factors read from a device
    /// buffer instead of scalar args.
    pub adamw_step_f32_capturable: CudaFunction,
    pub adamw_step_multi: TypedKernel,

    // ── Chunked parallel scan (mamba3_chunked.cu) ──
    /// The intra-chunk scan with whole rows staged, one head per block of
    /// 256 threads, the triangle and the operands in dynamic shared memory,
    /// for the shapes whose tile fits [`COOP_SCAN_TILE_MAX`] (see
    /// [`chunk_scan_cfg`]).
    pub m3_chunk_scan_fwd_coop: ByIndexWidth<CudaFunction>,
    // m3_chunk_scan_bwd, m3_state_passing_bwd, m3_chunk_state_bwd,
    // m3_cumsum_bwd removed — dead code replaced by monolithic m3_dqkv +
    // m3_dqktheta path below.
    pub m3_extract_da_cs_sum: ByIndexWidth<CudaFunction>,
    pub m3_dqkv: ByIndexWidth<CudaFunction>,
    /// Per-chunk B terms of the reverse d_state recurrence (f32/typed
    /// activation reads; f32 output).
    pub m3_dqkv_state_terms_typed: ByIndexWidth<TypedKernel>,
    /// Serial per-(b,h,p,n) reverse fold producing each chunk's
    /// entering d_state, and the gradient of the state the window entered
    /// with.
    pub m3_dstate_passing_bwd: ByIndexWidth<CudaFunction>,
    /// The gradient of the trapezoidal fold a window adds to the state it
    /// enters with, one block of 256 per (sequence, head).
    pub m3_enter_fold_bwd: ByIndexWidth<CudaFunction>,
    /// The per-chunk terms of the reverse state pass for any head and state
    /// width, in 16 x 16 output tiles: the bits of the warp kernel wherever
    /// that one runs.
    pub m3_dqkv_state_terms_tiled: TypedKernel,
    /// The flat chain a chunk's saved state adds to dADT, one thread per
    /// (sequence, chunk, head).
    pub m3_dqkv_dm_chain: CudaFunction,
    /// The chunked backward's dQKV stage for any head and state width:
    /// `m3_dqkv`'s outputs in its order, through tiles of a fixed budget.
    pub m3_dqkv_tiled: TypedKernel,
    pub m3_dqktheta: ByIndexWidth<CudaFunction>,
    pub m3_ddt_dtrap: ByIndexWidth<CudaFunction>,
    pub m3_final_grads: ByIndexWidth<CudaFunction>,

    // ── Typed variants for end-to-end bf16/f16 inference ──
    /// 8-way split + fused softplus/sigmoid, bf16/f16 proj and activation outputs.
    /// Coefficient outputs (dt, a_val, trap, angles, raw saves) stay f32.
    pub m3_split_typed: ByIndexWidth<TypedKernel>,
    /// Fused B+C variant of bcnorm_fwd_typed (2× grid via blockIdx.y).
    pub bcnorm_fwd_bc_typed: ByIndexWidth<TypedKernel>,
    /// Fused B+C norm, f32 lane (the decode step merges its two bcnorm
    /// launches through it).
    pub bcnorm_fwd_bc_f32: ByIndexWidth<CudaFunction>,
    /// Fused typed twin of bias + rope (round-trip contract preserved).
    pub m3_bias_rope_fwd_typed: ByIndexWidth<TypedKernel>,
    /// Plain SiLU gate (no norm), half I/O.
    pub silu_gate_fwd_typed: ByIndexWidth<TypedKernel>,
    /// SiLU-gate backward (typed twins share the f32 kernel's argument
    /// order and factored d_silu form).
    pub silu_gate_bwd_typed: ByIndexWidth<TypedKernel>,
    /// RMSNorm-gated output (half I/O, f32 weight/rms_vals).
    pub rmsnorm_gated_fwd_typed: ByIndexWidth<TypedKernel>,
    /// M3 SSM step — shared with training, already templated in mamba3_siso.cu.
    pub m3_step_fwd_typed: ByIndexWidth<TypedKernel>,
    /// Block-tier decode step for heads wider than a warp (or of a width
    /// that is not a power of two): one CTA per (batch, head), the
    /// head-shared values staged in shared memory. Same arguments as
    /// `m3_step_fwd`; launched with `step_block_cfg`.
    pub m3_step_fwd_block_typed: ByIndexWidth<TypedKernel>,
    /// The gated MLP's activation `value * silu(gate)` over the up
    /// projection's rows (value columns, then gate columns).
    pub mlp_gate_fwd_typed: TypedKernel,
    /// The gate's backward into the up projection's row gradient.
    pub mlp_gate_bwd: TypedKernel,
    /// M3 burnin forward (training) — sequential T-loop SSM with activation
    /// saves. Typed x/k/q/y; f32 state + alpha/beta/gamma + D + saves.
    pub m3_burnin_fwd_typed_bf16: ByIndexWidth<CudaFunction>,
    pub m3_burnin_fwd_typed_f16: ByIndexWidth<CudaFunction>,
    burnin_fwd_typed_by_state: Option<ByIndexWidth<HalfKernel>>,

    // -- Typed M3 sequential backward kernels --
    /// RmsNorm over B/C groups, typed dy → typed d_B; f32 rms + weight +
    /// d_weight master-grad accumulator.
    pub bcnorm_bwd_typed: ByIndexWidth<TypedKernel>,
    /// Per-head bias reduction from typed d_B_biased → typed d_B_normed
    /// (expand groups backward). Bias grad handled by reduce_bias_typed.
    pub bc_bias_add_bwd_typed: ByIndexWidth<TypedKernel>,
    /// RoPE rotation backward, typed B/C grads + saved B/C, f32 angle saves.
    pub rope_bwd_typed: ByIndexWidth<TypedKernel>,
    /// RoPE rotation backward with f32 gradients against the bf16 or f16
    /// pre-RoPE B and C, for the sequential mixed backward.
    pub rope_bwd_mixed: ByIndexWidth<HalfKernel>,
    /// 8-way split backward: assemble typed d_proj from typed d_z/d_x/
    /// d_B_raw/d_C_raw plus f32 dd_dt/dd_a/trap/angles.
    pub m3_split_bwd_typed: ByIndexWidth<TypedKernel>,
    /// RMSNorm-gated backward (`out = RMSNorm(y) * weight * SiLU(z)`) —
    /// typed d_y/d_z/d_out/y/z; f32 weight + d_weight (per-sample,
    /// reduced later).
    pub rmsnorm_gated_bwd_typed: ByIndexWidth<TypedKernel>,
    // -- Typed M3 "final grad" kernels (HIGHEST RISK) --
    /// Huge dqkv kernel with smem tiles — typed Q_rot/K_scaled/V_in/dO;
    /// all 6 grad outputs stay f32 (atomicAdd on dD; master grads on others).
    pub m3_dqkv_typed: ByIndexWidth<TypedKernel>,
    /// Inverse-RoPE + bias backward — typed Q_raw/K_raw; 7 grad outputs f32.
    pub m3_dqktheta_typed: ByIndexWidth<TypedKernel>,

    // m3_chunk_scan_bwd_typed + m3_chunk_state_bwd_typed
    // removed (dead — the typed backward path, like the f32 path, now routes
    // through m3_dqkv_typed + m3_dqktheta_typed monolithic kernels).

    // -- Typed M3 chunked parallel forward kernels --
    /// The pre-state in one kernel with whole rows staged: each row's
    /// scale, gamma and Q.K dot, the prescaled K rows, the chunk's decay
    /// cumsum and its states. Typed K/Q/K_scaled/x; f32 DT/trap_sig/
    /// qk_dot/scale/gamma, and f32 states, as BPTT needs them (see
    /// [`chunk_pre_state_cfg`]).
    pub m3_chunk_pre_state_fused_typed: ByIndexWidth<TypedKernel>,
    /// The pre-state for any head and state width, the state staged in
    /// slices: the fused kernel's outputs, bit for bit, and the same
    /// arguments (see [`chunk_pre_state_cfg`]).
    pub m3_chunk_pre_state_sliced_typed: ByIndexWidth<TypedKernel>,
    /// The inter-chunk state pass of a window in one launch, one CTA per
    /// (b, h): the trapezoidal entering fold of a carried window, the
    /// exclusive prefix scan over the chunks, and the window's final SSM,
    /// K and V states, the folded seed and the final state in registers.
    /// Typed inputs k_flat/x_flat; every state f32. Launched with
    /// `state_pass_fused_cfg`.
    pub m3_state_pass_fused_typed: ByIndexWidth<TypedKernel>,
    /// The intra-chunk scan for any head and state width, the state staged
    /// in slices: typed y_out/x/Q/K_scaled, f32 qk_dot/dA_cumsum/
    /// prev_states/D, the cooperative scan's arguments and the static-tile
    /// kernel's arithmetic (see [`chunk_scan_cfg`]).
    pub m3_chunk_scan_fwd_sliced_typed: ByIndexWidth<TypedKernel>,
    /// The most threads one block of the block-tier decode step may run:
    /// the fewest any of its f32, bf16 and f16 kernels can launch with the
    /// registers its state capacity compiled to.
    pub step_block_threads: ByIndexWidth<usize>,
    /// The dynamic shared memory a block of the staged kernels plans its
    /// tiles with: what each of them opted in to on this device beside its
    /// own static stage, capped at [`PLANNED_SMEM`], or less when
    /// `MAMBA_RS_SMEM_BUDGET` asks for it.
    pub smem_optin: usize,
    /// Cooperative typed twin (see `m3_chunk_scan_fwd_coop`).
    pub m3_chunk_scan_fwd_coop_typed: ByIndexWidth<TypedKernel>,

    /// Shared from M1: f32 residual += half branch (stays f32).
    pub residual_add_f32_typed: HalfKernel,
    /// Shared from M1: gather last timestep of B×T×D into B×D, dtype-preserving.
    pub gather_last_timestep_typed: TypedKernel,
}

impl Mamba3Kernels {
    #[doc(hidden)]
    pub fn compiler_identity(&self) -> crate::gpu::kernel_identity::CompilerIdentity {
        self.compiler_identity
    }

    #[doc(hidden)]
    pub fn artifact_identity(&self) -> crate::gpu::kernel_identity::ArtifactIdentity {
        self.artifact_identity
    }

    /// Compile Mamba-3 kernels with the default state capacity of 64. Models with a
    /// larger `d_state` use [`Self::compile_with_state_cap`].
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        Self::compile_with_state_cap(ctx, arch, 64)
    }

    /// The backward scan to launch for a model of this state width, at the
    /// index width of the launch: the specialisation when one exists, the
    /// general entry otherwise. The two compute the same thing; the
    /// specialisation is faster because its loop bounds are constants the
    /// compiler can unroll.
    pub fn backward_seq_for_state(&self, d_state: usize, width: IndexWidth) -> &CudaFunction {
        self.m3_backward_seq_by_state
            .iter()
            .find(|(state, _)| *state == d_state)
            .map(|(_, function)| function)
            .unwrap_or(&self.m3_backward_seq)
            .get(width)
    }

    /// The burn-in forward to launch for a model of this state width.
    pub fn burnin_fwd_for_state(&self, d_state: usize, width: IndexWidth) -> &CudaFunction {
        self.m3_burnin_fwd_by_state
            .iter()
            .find(|(state, _)| *state == d_state)
            .map(|(_, function)| function)
            .unwrap_or(&self.m3_burnin_fwd)
            .get(width)
    }

    /// Select the sequential training forward kernel for an activation dtype
    /// and state width. Persistent state and backward saves remain FP32.
    ///
    /// BF16/F16 use constant-width loops on qualified Ada compiler targets;
    /// other devices, compilers and widths retain the general typed kernel.
    /// F32 uses the existing F32 state-width selector.
    pub fn burnin_fwd_typed_for_state(
        &self,
        dtype: WeightDtype,
        d_state: usize,
        width: IndexWidth,
    ) -> &CudaFunction {
        if dtype == WeightDtype::F32 {
            return self.burnin_fwd_for_state(d_state, width);
        }
        if matches!(d_state, 8 | 16 | 32 | 64)
            && d_state <= self.state_cap
            && let Some(kernels) = &self.burnin_fwd_typed_by_state
        {
            return kernels.get(width).get(dtype);
        }
        match dtype {
            WeightDtype::Bf16 => self.m3_burnin_fwd_typed_bf16.get(width),
            WeightDtype::F16 => self.m3_burnin_fwd_typed_f16.get(width),
            WeightDtype::F32 | WeightDtype::Tf32 => unreachable!("F32 uses its own selector"),
        }
    }

    /// The sequential training forward for a shape, with its launch. A head
    /// of one warp keeps the warp kernels; every other head takes the block
    /// tier, which gives the warp kernels' bits where both serve. The block
    /// tier indexes in 64 bits at every width.
    pub fn sequential_forward(
        &self,
        dtype: WeightDtype,
        shape: SequentialShape,
        width: IndexWidth,
    ) -> (&CudaFunction, LaunchConfig) {
        if head_tier(shape.headdim) == HeadTier::Warp {
            let kernel = match dtype {
                WeightDtype::F32 | WeightDtype::Tf32 => {
                    self.burnin_fwd_for_state(shape.d_state, width)
                }
                WeightDtype::Bf16 | WeightDtype::F16 => {
                    self.burnin_fwd_typed_for_state(dtype, shape.d_state, width)
                }
            };
            return (kernel, shape.launch(true, 0));
        }
        let smem = crate::mamba3_siso::route::sequential_forward_block_smem_bytes(shape.d_state);
        (
            self.m3_burnin_fwd_block.get(dtype),
            shape.launch(false, smem),
        )
    }

    /// The target network's sequential forward for a shape, with its launch.
    pub fn sequential_target_forward(
        &self,
        shape: SequentialShape,
        width: IndexWidth,
    ) -> (&CudaFunction, LaunchConfig) {
        if head_tier(shape.headdim) == HeadTier::Warp {
            return (self.m3_burnin_fwd_nosave.get(width), shape.launch(true, 0));
        }
        let smem = crate::mamba3_siso::route::sequential_forward_block_smem_bytes(shape.d_state);
        (&self.m3_burnin_fwd_nosave_block, shape.launch(false, smem))
    }

    /// The sequential backward for a shape, with its launch. The f32 warp
    /// kernels keep every head of one warp; the block tier serves every
    /// other head, and every head of the bf16 and f16 backward.
    pub fn sequential_backward(
        &self,
        dtype: WeightDtype,
        shape: SequentialShape,
        width: IndexWidth,
    ) -> (&CudaFunction, LaunchConfig) {
        let warp = head_tier(shape.headdim) == HeadTier::Warp;
        if warp && matches!(dtype, WeightDtype::F32 | WeightDtype::Tf32) {
            return (
                self.backward_seq_for_state(shape.d_state, width),
                shape.launch(true, 0),
            );
        }
        let smem = crate::mamba3_siso::route::sequential_backward_block_smem_bytes(
            shape.headdim,
            shape.d_state,
        );
        (
            self.m3_backward_seq_block.get(dtype),
            shape.launch(false, smem),
        )
    }

    /// Compile all Mamba-3 kernels. `state_cap` sizes the per-thread
    /// state register arrays (see
    /// [`crate::gpu::kernels::state_capacity`]); raising it
    /// past the register budget makes the compiler spill to local
    /// memory — correct, slower, accepted as a first-class capacity
    /// knob rather than a separate slow path.
    pub fn compile_with_state_cap(
        ctx: &Arc<CudaContext>,
        arch: &'static str,
        state_cap: usize,
    ) -> Result<Self, String> {
        let sources = [
            // Inline the prelude first so each source file's
            // `#include "_typed_prelude.cuh"` can be safely stripped below.
            include_str!("../../../kernels/_typed_prelude.cuh"),
            // The packed-row helpers (seq_idx) the kernels below read.
            include_str!("../../../kernels/_varlen.cuh"),
            include_str!("../../../kernels/mamba3_siso.cu"),
            include_str!("../../../kernels/mamba3_ops.cu"),
            include_str!("../../../kernels/mamba3_chunked.cu"),
            // Block-tier twins of the head kernels, for heads wider than a
            // warp; composed after the warp tier so its entries stay first.
            include_str!("../../../kernels/mamba3_siso_block.cu"),
            // The gated MLP's activation, for models trained with an MLP.
            include_str!("../../../kernels/gated_mlp.cu"),
            // Shared kernels needed by training pipeline; the block norms
            // run from the context's kernels.
            include_str!("../../../kernels/elementwise.cu"),
            include_str!("../../../kernels/gemm_bi_inference/casts.cu"),
            include_str!("../../../kernels/adamw.cu"),
        ];

        let combined_body: String = sources
            .iter()
            .map(|s| {
                s.lines()
                    .filter(|l| !l.trim().starts_with("#include \"_typed_prelude.cuh\""))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let combined = combined_body;
        let (nv_major, nv_minor) = crate::gpu::kernels::nvrtc_version();
        let mut option_strings = vec![
            "--fmad=true".to_string(),
            "--extra-device-vectorization".to_string(),
            // Mirrors the M1 compiler: strip device assert() trap
            // checks (none in the m3 sources today, but shared
            // headers may grow them); asserts compute no values.
            "-DNDEBUG".to_string(),
            format!("-DMAMBA_RS_STATE_CAP={state_cap}"),
        ];
        option_strings.extend(crate::gpu::kernel_identity::deterministic_nvrtc_options(
            (nv_major, nv_minor),
            "1295203121",
        ));
        let include_paths = crate::gpu::kernels::cuda_include_paths();
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: option_strings.clone(),
            include_paths: include_paths.clone(),
            ..Default::default()
        };

        let nvrtc_library_domain = crate::gpu::kernel_identity::nvrtc_library_domain();
        let header_manifest =
            crate::gpu::kernel_identity::header_manifest(combined.as_bytes(), &include_paths);
        let mut argv = vec![format!("--gpu-architecture={arch}").into_bytes()];
        argv.extend(option_strings.iter().map(|value| value.as_bytes().to_vec()));
        let key_material = crate::gpu::kernel_identity::CompileKeyMaterial {
            module_kind: crate::gpu::kernel_identity::ModuleKind::Mamba3Combined,
            source: combined.as_bytes().to_vec(),
            target: arch.as_bytes().to_vec(),
            argv,
            header_manifest: header_manifest.clone(),
            nvrtc_version: (nv_major, nv_minor),
            nvrtc_library_domain: nvrtc_library_domain.clone(),
            output_kind: crate::gpu::kernel_identity::ArtifactKind::Ptx,
            composer_revision: crate::gpu::kernel_identity::COMPOSER_REVISION,
            compiler_revision: crate::gpu::kernel_identity::COMPILER_REVISION,
            numeric_abi_revision: crate::gpu::kernel_identity::NUMERIC_ABI_REVISION,
            schedule_revision: crate::gpu::kernel_identity::SCHEDULE_REVISION,
        };
        let invocation_digest = key_material.invocation_digest();
        let cache_key = key_material.digest();
        let cache_path = cache_key.and_then(|key| {
            crate::gpu::kernels::kernel_cache_dir().map(|directory| {
                directory.join(format!(
                    "mamba3-kernels-v2-{}.bin",
                    crate::gpu::kernel_identity::digest_hex(&key)
                ))
            })
        });

        let mut loaded = None;
        if let (Some(path), Some(key)) = (&cache_path, cache_key)
            && let Some(hit) = crate::gpu::kernel_identity::read_cache(
                path,
                key,
                crate::gpu::kernel_identity::ArtifactKind::Ptx,
            )
            && let Ok(source) = crate::gpu::kernel_identity::canonical_ptx_from_cache(hit.payload)
            && let Ok(module) = ctx.load_module(cudarc::nvrtc::Ptx::from_src(source))
            && crate::gpu::kernel_identity::cache_hit_header_closure_is_current(
                combined.as_bytes(),
                &include_paths,
                &header_manifest,
            )
            && nvrtc_library_domain
                .as_deref()
                .is_some_and(crate::gpu::kernel_identity::nvrtc_library_domain_is_current)
        {
            loaded = Some((module, hit.artifact_digest));
        }
        let (module, artifact_digest) = match loaded {
            Some(value) => value,
            None => {
                let ptx = cudarc::nvrtc::compile_ptx_with_opts(&combined, opts).map_err(|e| {
                    format!(
                        "NVRTC M3 compile failed: {}",
                        format!("{e:?}").replace("\\n", "\n")
                    )
                })?;
                let ptx_image = ptx
                    .as_bytes()
                    .ok_or_else(|| "NVRTC returned M3 PTX without a raw image".to_string())?;
                let ptx_source = crate::gpu::kernel_identity::canonical_ptx_image(ptx_image)?;
                if !crate::gpu::kernel_identity::header_manifest_is_current(
                    combined.as_bytes(),
                    &include_paths,
                    &header_manifest,
                ) {
                    return Err(
                        "CUDA headers changed during M3 NVRTC compilation; retry initialization"
                            .into(),
                    );
                }
                if let Some(domain) = nvrtc_library_domain.as_deref()
                    && !crate::gpu::kernel_identity::nvrtc_library_domain_is_current(domain)
                {
                    return Err(
                        "NVRTC libraries changed during M3 compilation; retry initialization"
                            .into(),
                    );
                }
                let artifact_digest =
                    crate::gpu::kernel_identity::FramedSha256::bytes(ptx_source.as_bytes());
                if let (Some(path), Some(key)) = (&cache_path, cache_key) {
                    crate::gpu::kernel_identity::publish_cache(
                        path,
                        key,
                        crate::gpu::kernel_identity::ArtifactKind::Ptx,
                        ptx_source.as_bytes(),
                    );
                }
                let module = ctx
                    .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
                    .map_err(|e| format!("M3 module load failed: {e:?}"))?;
                (module, artifact_digest)
            }
        };
        let compiler_identity = crate::gpu::kernel_identity::CompilerIdentity {
            source_digest: crate::gpu::kernel_identity::FramedSha256::bytes(combined.as_bytes()),
            invocation_digest,
            header_manifest_digest: crate::gpu::kernel_identity::FramedSha256::new(
                b"cuda-header-manifest.v1",
            )
            .optional(b"manifest", header_manifest.as_deref())
            .finish(),
            target: crate::gpu::kernel_identity::CudaTarget::new(arch)?,
            nvrtc_version: (nv_major, nv_minor),
            nvrtc_library_domain: crate::gpu::kernel_identity::FramedSha256::new(
                b"nvrtc-library-set-identity.v2",
            )
            .optional(b"domain", nvrtc_library_domain.as_deref())
            .finish(),
            nvrtc_library_known: nvrtc_library_domain.is_some(),
            output_kind: crate::gpu::kernel_identity::ArtifactKind::Ptx,
            composer_revision: crate::gpu::kernel_identity::COMPOSER_REVISION,
            compiler_revision: crate::gpu::kernel_identity::COMPILER_REVISION,
            numeric_abi_revision: crate::gpu::kernel_identity::NUMERIC_ABI_REVISION,
            schedule_revision: crate::gpu::kernel_identity::SCHEDULE_REVISION,
        };
        let artifact_identity = crate::gpu::kernel_identity::ArtifactIdentity {
            module_kind: crate::gpu::kernel_identity::ModuleKind::Mamba3Combined,
            artifact_kind: crate::gpu::kernel_identity::ArtifactKind::Ptx,
            compile_key: invocation_digest,
            artifact_digest,
        };
        let module_identity = crate::gpu::kernel_identity::digest_hex(&invocation_digest);

        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("M3 kernel '{name}' not found: {e:?}"))
        };
        // Both index widths of a kernel built twice: `name` and `name_i64`.
        let pair = |name: &str| -> Result<ByIndexWidth<CudaFunction>, String> {
            Ok(ByIndexWidth {
                i32: get(name)?,
                i64: get(&format!("{name}_i64"))?,
            })
        };
        let typed_pair =
            |f32: &str, bf16: &str, f16: &str| -> Result<ByIndexWidth<TypedKernel>, String> {
                Ok(ByIndexWidth {
                    i32: TypedKernel {
                        f32: get(f32)?,
                        bf16: get(bf16)?,
                        f16: get(f16)?,
                    },
                    i64: TypedKernel {
                        f32: get(&format!("{f32}_i64"))?,
                        bf16: get(&format!("{bf16}_i64"))?,
                        f16: get(&format!("{f16}_i64"))?,
                    },
                })
            };
        // Each width's bound comes from its own builds, so the int builds
        // launch as they did before the 64-bit builds existed.
        let bound = |names: &[&str], suffix: &str, what: &str| -> Result<usize, String> {
            let kernels = names
                .iter()
                .map(|name| get(&format!("{name}{suffix}")))
                .collect::<Result<Vec<_>, _>>()?;
            block_threads_bound(&kernels.iter().collect::<Vec<_>>(), what)
        };
        let step_blocks = [
            "m3_step_fwd_block",
            "m3_step_fwd_block_bf16",
            "m3_step_fwd_block_f16",
        ];
        let step_block_threads = ByIndexWidth {
            i32: bound(&step_blocks, "", "block-tier decode step")?,
            i64: bound(&step_blocks, "_i64", "block-tier decode step")?,
        };

        // PTX targeting Ada can also run on newer devices. Admission checks
        // the actual device because preserving a compiler's FMA graph alone
        // does not establish compatibility with another device's released bits.
        let actual_device = ctx
            .compute_capability()
            .map_err(|error| format!("M3 device capability: {error:?}"))?;
        let transport_admission = TransportAdmission::new(
            transport_environment_admitted(actual_device, arch, (nv_major, nv_minor)),
            state_cap,
        );
        let transport = TransportKernels::load(transport_admission, get)?;

        let burnin_fwd_typed_by_state = if actual_device == (8, 9)
            && matches!(arch, "sm_89" | "compute_89")
            && matches!((nv_major, nv_minor), (12, 8) | (13, 0) | (13, 2))
            && matches!(state_cap, 16 | 32 | 64)
        {
            Some(ByIndexWidth {
                i32: HalfKernel {
                    bf16: get("m3_burnin_fwd_bf16_by_state")?,
                    f16: get("m3_burnin_fwd_f16_by_state")?,
                },
                i64: HalfKernel {
                    bf16: get("m3_burnin_fwd_bf16_by_state_i64")?,
                    f16: get("m3_burnin_fwd_f16_by_state_i64")?,
                },
            })
        } else {
            None
        };

        let mut kernels = Self {
            transport,
            burnin_fwd_typed_by_state,
            module_identity,
            state_cap,
            compiler_identity,
            artifact_identity,
            // Sequential SSM
            m3_step_fwd: pair("m3_step_fwd")?,
            m3_burnin_fwd: pair("m3_burnin_fwd")?,
            m3_burnin_fwd_by_state: [
                (8, pair("m3_burnin_fwd_ds8")?),
                (16, pair("m3_burnin_fwd_ds16")?),
                (32, pair("m3_burnin_fwd_ds32")?),
                (64, pair("m3_burnin_fwd_ds64")?),
            ],
            m3_burnin_fwd_nosave: pair("m3_burnin_fwd_nosave")?,
            m3_backward_seq: pair("m3_backward_seq")?,
            m3_burnin_fwd_block: TypedKernel {
                f32: get("m3_burnin_fwd_block")?,
                bf16: get("m3_burnin_fwd_block_bf16")?,
                f16: get("m3_burnin_fwd_block_f16")?,
            },
            m3_burnin_fwd_nosave_block: get("m3_burnin_fwd_nosave_block")?,
            m3_backward_seq_block: TypedKernel {
                f32: get("m3_backward_seq_block")?,
                bf16: get("m3_backward_seq_block_bf16")?,
                f16: get("m3_backward_seq_block_f16")?,
            },
            m3_backward_seq_by_state: [
                (8, pair("m3_backward_seq_ds8")?),
                (16, pair("m3_backward_seq_ds16")?),
                (32, pair("m3_backward_seq_ds32")?),
                (64, pair("m3_backward_seq_ds64")?),
            ],
            m3_reduce_d_d: pair("m3_reduce_d_D")?,

            // Shared ops
            m3_split: pair("m3_split")?,
            m3_split_bwd: pair("m3_split_bwd")?,
            bcnorm_bwd: pair("bcnorm_bwd")?,
            bc_bias_add: pair("bc_bias_add")?,
            bc_bias_add_bwd: pair("bc_bias_add_bwd")?,
            m3_angle_dt_fwd_seq: pair("m3_angle_dt_fwd_seq")?,
            m3_angle_chunk_sums: pair("m3_angle_chunk_sums")?,
            m3_angle_chunk_apply: pair("m3_angle_chunk_apply")?,
            m3_angle_dt_bwd_seq: pair("m3_angle_dt_bwd_seq")?,
            rope_fwd: pair("rope_fwd")?,
            m3_bias_rope_fwd: pair("m3_bias_rope_fwd")?,
            rope_bwd: pair("rope_bwd")?,
            m3_compute_abg: pair("m3_compute_abg")?,
            m3_abg_bwd: pair("m3_abg_bwd")?,
            silu_gate_fwd: pair("silu_gate_fwd")?,
            silu_gate_bwd: pair("silu_gate_bwd")?,
            rmsnorm_gated_fwd: pair("rmsnorm_gated_forward")?,
            rmsnorm_gated_bwd: pair("rmsnorm_gated_backward")?,

            // Shared (elementwise.cu)
            colsum_accumulate: get("colsum_accumulate")?,
            colsum_segments: get("colsum_segments")?,
            reduce_sum_axis0: get("reduce_sum_axis0")?,
            reduce_sum_axis0_wide: get("reduce_sum_axis0_wide")?,
            vec_add_inplace: get("vec_add_inplace")?,
            elementwise_mul: get("elementwise_mul")?,
            elementwise_mul_v: get("elementwise_mul_v_f32")?,
            fill_scalar: get("fill_scalar")?,
            cast_f32_to_bf16: get("cast_f32_to_bf16")?,
            cast_f32_to_f16: get("cast_f32_to_f16")?,
            cast_bf16_to_f32: get("cast_bf16_to_f32")?,
            cast_f16_to_f32: get("cast_f16_to_f32")?,
            residual_add: get("residual_add")?,
            gather_last_timestep: get("gather_last_timestep")?,

            // AdamW optimizer
            adamw_step_f32: get("adamw_step_f32")?,
            adamw_step_f32_capturable: get("adamw_step_f32_capturable")?,
            adamw_step_multi: TypedKernel {
                f32: get("adamw_step_multi_f32")?,
                bf16: get("adamw_step_multi_bf16")?,
                f16: get("adamw_step_multi_f16")?,
            },

            // Chunked parallel scan
            m3_chunk_scan_fwd_coop: pair("m3_chunk_scan_fwd_coop")?,
            m3_extract_da_cs_sum: pair("m3_extract_da_cs_sum")?,
            m3_dqkv: pair("m3_dqkv")?,
            m3_dqkv_state_terms_typed: typed_pair(
                "m3_dqkv_state_terms",
                "m3_dqkv_state_terms_bf16",
                "m3_dqkv_state_terms_f16",
            )?,
            m3_dstate_passing_bwd: pair("m3_dstate_passing_bwd")?,
            m3_enter_fold_bwd: pair("m3_enter_fold_bwd")?,
            m3_dqkv_state_terms_tiled: TypedKernel {
                f32: get("m3_dqkv_state_terms_tiled")?,
                bf16: get("m3_dqkv_state_terms_tiled_bf16")?,
                f16: get("m3_dqkv_state_terms_tiled_f16")?,
            },
            m3_dqkv_dm_chain: get("m3_dqkv_dm_chain")?,
            m3_dqkv_tiled: TypedKernel {
                f32: get("m3_dqkv_tiled")?,
                bf16: get("m3_dqkv_tiled_bf16")?,
                f16: get("m3_dqkv_tiled_f16")?,
            },
            m3_dqktheta: pair("m3_dqktheta")?,
            m3_ddt_dtrap: pair("m3_ddt_dtrap")?,
            m3_final_grads: pair("m3_final_grads")?,

            // Typed variants for mixed-dtype inference
            m3_split_typed: typed_pair("m3_split", "m3_split_bf16", "m3_split_f16")?,
            bcnorm_fwd_bc_typed: typed_pair(
                "bcnorm_fwd_bc_f32",
                "bcnorm_fwd_bc_bf16",
                "bcnorm_fwd_bc_f16",
            )?,
            bcnorm_fwd_bc_f32: pair("bcnorm_fwd_bc_f32")?,
            m3_bias_rope_fwd_typed: typed_pair(
                "m3_bias_rope_fwd",
                "m3_bias_rope_fwd_bf16",
                "m3_bias_rope_fwd_f16",
            )?,
            silu_gate_fwd_typed: typed_pair(
                "silu_gate_fwd",
                "silu_gate_fwd_bf16",
                "silu_gate_fwd_f16",
            )?,
            silu_gate_bwd_typed: typed_pair(
                "silu_gate_bwd",
                "silu_gate_bwd_bf16",
                "silu_gate_bwd_f16",
            )?,
            rmsnorm_gated_fwd_typed: typed_pair(
                "rmsnorm_gated_forward",
                "rmsnorm_gated_forward_bf16",
                "rmsnorm_gated_forward_f16",
            )?,
            m3_step_fwd_typed: typed_pair("m3_step_fwd", "m3_step_fwd_bf16", "m3_step_fwd_f16")?,
            m3_step_fwd_block_typed: typed_pair(
                "m3_step_fwd_block",
                "m3_step_fwd_block_bf16",
                "m3_step_fwd_block_f16",
            )?,
            mlp_gate_fwd_typed: TypedKernel {
                f32: get("mlp_gate_fwd")?,
                bf16: get("mlp_gate_fwd_bf16")?,
                f16: get("mlp_gate_fwd_f16")?,
            },
            mlp_gate_bwd: TypedKernel {
                f32: get("mlp_gate_bwd")?,
                bf16: get("mlp_gate_bwd_bf16")?,
                f16: get("mlp_gate_bwd_f16")?,
            },
            m3_burnin_fwd_typed_bf16: pair("m3_burnin_fwd_bf16")?,
            m3_burnin_fwd_typed_f16: pair("m3_burnin_fwd_f16")?,
            bcnorm_bwd_typed: typed_pair("bcnorm_bwd", "bcnorm_bwd_bf16", "bcnorm_bwd_f16")?,
            bc_bias_add_bwd_typed: typed_pair(
                "bc_bias_add_bwd",
                "bc_bias_add_bwd_bf16",
                "bc_bias_add_bwd_f16",
            )?,
            rope_bwd_typed: typed_pair("rope_bwd", "rope_bwd_bf16", "rope_bwd_f16")?,
            rope_bwd_mixed: ByIndexWidth {
                i32: HalfKernel {
                    bf16: get("rope_bwd_mixed_bf16")?,
                    f16: get("rope_bwd_mixed_f16")?,
                },
                i64: HalfKernel {
                    bf16: get("rope_bwd_mixed_bf16_i64")?,
                    f16: get("rope_bwd_mixed_f16_i64")?,
                },
            },
            m3_split_bwd_typed: typed_pair(
                "m3_split_bwd",
                "m3_split_bwd_bf16",
                "m3_split_bwd_f16",
            )?,
            rmsnorm_gated_bwd_typed: typed_pair(
                "rmsnorm_gated_backward",
                "rmsnorm_gated_backward_bf16",
                "rmsnorm_gated_backward_f16",
            )?,
            m3_dqkv_typed: typed_pair("m3_dqkv", "m3_dqkv_bf16", "m3_dqkv_f16")?,
            m3_dqktheta_typed: typed_pair("m3_dqktheta", "m3_dqktheta_bf16", "m3_dqktheta_f16")?,

            m3_chunk_pre_state_fused_typed: typed_pair(
                "m3_chunk_pre_state_fused",
                "m3_chunk_pre_state_fused_bf16",
                "m3_chunk_pre_state_fused_f16",
            )?,
            m3_chunk_pre_state_sliced_typed: typed_pair(
                "m3_chunk_pre_state_sliced",
                "m3_chunk_pre_state_sliced_bf16",
                "m3_chunk_pre_state_sliced_f16",
            )?,
            m3_state_pass_fused_typed: typed_pair(
                "m3_state_pass_fused_f32",
                "m3_state_pass_fused_bf16",
                "m3_state_pass_fused_f16",
            )?,
            m3_chunk_scan_fwd_coop_typed: typed_pair(
                "m3_chunk_scan_fwd_coop",
                "m3_chunk_scan_fwd_coop_bf16",
                "m3_chunk_scan_fwd_coop_f16",
            )?,
            m3_chunk_scan_fwd_sliced_typed: typed_pair(
                "m3_chunk_scan_fwd_sliced",
                "m3_chunk_scan_fwd_sliced_bf16",
                "m3_chunk_scan_fwd_sliced_f16",
            )?,

            residual_add_f32_typed: HalfKernel {
                bf16: get("residual_add_f32_bf16")?,
                f16: get("residual_add_f32_f16")?,
            },
            gather_last_timestep_typed: TypedKernel {
                f32: get("gather_last_timestep_f32")?,
                bf16: get("gather_last_timestep_bf16")?,
                f16: get("gather_last_timestep_f16")?,
            },

            step_block_threads,
            smem_optin: 0,
            _modules: CudaModuleAnchors::new(vec![module]),
        };

        // Every kernel that can stage more than the 48 KB default opts in to
        // what the device lets a block hold beside the kernel's own static
        // stage. Tiles are planned within PLANNED_SMEM, which every
        // supported part allows, so a shape takes the same kernels, and
        // gives the same bits, on every card.
        kernels.smem_optin = {
            let optin = ctx
                .attribute(
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                )
                .map_err(|e| format!("query opt-in shared memory: {e:?}"))?
                .max(0) as usize;
            let mut budget = PLANNED_SMEM;
            let k = &kernels;
            let by_width = [(&k.m3_dqkv, "m3_dqkv"), (&k.m3_dqktheta, "m3_dqktheta")];
            for (pair, name) in by_width {
                for f in [&pair.i32, &pair.i64] {
                    budget = budget.min(opt_in_dynamic_smem(f, optin, name)?);
                }
            }
            let typed_by_width = [
                (&k.m3_dqkv_typed, "m3_dqkv typed"),
                (
                    &k.m3_chunk_pre_state_fused_typed,
                    "m3_chunk_pre_state_fused",
                ),
                (&k.m3_dqktheta_typed, "m3_dqktheta typed"),
            ];
            for (pair, name) in typed_by_width {
                for t in [&pair.i32, &pair.i64] {
                    for f in [&t.f32, &t.bf16, &t.f16] {
                        budget = budget.min(opt_in_dynamic_smem(f, optin, name)?);
                    }
                }
            }
            let typed = [
                (&k.m3_dqkv_tiled, "m3_dqkv_tiled"),
                (&k.m3_backward_seq_block, "m3_backward_seq_block"),
            ];
            for (t, name) in typed {
                for f in [&t.f32, &t.bf16, &t.f16] {
                    budget = budget.min(opt_in_dynamic_smem(f, optin, name)?);
                }
            }
            if let Some(t) = &k.transport.dqkv {
                for f in [&t.f32, &t.bf16, &t.f16] {
                    budget = budget.min(opt_in_dynamic_smem(f, optin, "m3_dqkv transport")?);
                }
            }
            smem_budget(budget)
        };
        Ok(kernels)
    }

    pub(crate) fn module_anchors(&self) -> CudaModuleAnchors {
        self._modules.clone()
    }
}

/// Launch geometry for the fused B/C normalization kernels.
///
/// `rows` is batch-times-sequence-times-groups. Short state vectors share
/// full warps; larger vectors retain one row per block and the shared tree.
/// Use this configuration for every `bcnorm_fwd_bc_*` entry: its row mapping
/// depends on both block dimensions. Both arguments must be nonzero.
pub fn bcnorm_fwd_bc_cfg(rows: usize, d_state: usize) -> cudarc::driver::LaunchConfig {
    assert!(rows > 0 && d_state > 0);
    if d_state <= 32 {
        let width = d_state.next_power_of_two();
        let threads = if rows < 128 { 32 } else { 256 };
        let rows_per_block = threads / width;
        cudarc::driver::LaunchConfig {
            grid_dim: (rows.div_ceil(rows_per_block) as u32, 2, 1),
            block_dim: (width as u32, rows_per_block as u32, 1),
            shared_mem_bytes: 0,
        }
    } else {
        cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, 2, 1),
            block_dim: (d_state as u32, 1, 1),
            shared_mem_bytes: (d_state * 4) as u32,
        }
    }
}

/// Launch geometry for the gated output norm (`rmsnorm_gated_*`) over `rows`
/// rows of `d_inner` channels in groups of `group_size`: every group takes
/// its width rounded up to a power of two, a block holds as many groups as
/// fit in 256 threads (one when a group is wider, in at most 1024 threads
/// whose lanes a wider group's channels take in turns), and shared memory
/// holds one float per thread. The kernels derive their mapping from this
/// shape.
pub fn gated_norm_cfg(
    rows: usize,
    d_inner: usize,
    group_size: usize,
) -> cudarc::driver::LaunchConfig {
    assert!(rows > 0 && group_size > 0 && d_inner.is_multiple_of(group_size));
    let width = group_size.next_power_of_two();
    let threads = width.clamp(256, 1024);
    let per_block = (threads / width).max(1);
    cudarc::driver::LaunchConfig {
        grid_dim: (
            rows as u32,
            (d_inner / group_size).div_ceil(per_block) as u32,
            1,
        ),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: (threads * std::mem::size_of::<f32>()) as u32,
    }
}

/// The most threads one block of every kernel in `kernels` can launch with
/// the registers it compiled to, in whole warps.
fn block_threads_bound(kernels: &[&CudaFunction], what: &str) -> Result<usize, String> {
    use cudarc::driver::sys::CUfunction_attribute_enum as FnAttr;
    let mut threads = 1024usize;
    for f in kernels {
        let max = f
            .get_attribute(FnAttr::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map_err(|e| format!("threads per block of the {what}: {e:?}"))?;
        threads = threads.min(max.max(32) as usize);
    }
    Ok(threads / 32 * 32)
}

/// Launch geometry for the decode step of a head: the warp tier's one
/// warp of `headdim` lanes, or the block tier's CTA of `headdim` threads
/// rounded up to whole warps, at most `block_threads`
/// ([`Mamba3Kernels::step_block_threads`]) whose threads a wider head's
/// elements take in turns, with the step coefficients and the K and Q rows
/// staged in shared memory.
pub fn step_head_cfg(
    tier: crate::mamba3_siso::route::HeadTier,
    batch: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    block_threads: usize,
) -> cudarc::driver::LaunchConfig {
    use crate::mamba3_siso::route::HeadTier;
    match tier {
        HeadTier::Warp => cudarc::driver::LaunchConfig {
            grid_dim: (batch as u32, nh as u32, 1),
            block_dim: (hd as u32, 1, 1),
            shared_mem_bytes: 0,
        },
        HeadTier::Block => cudarc::driver::LaunchConfig {
            grid_dim: (batch as u32, nh as u32, 1),
            block_dim: (hd.next_multiple_of(32).min(block_threads) as u32, 1, 1),
            shared_mem_bytes: ((4 + 3 * ds) * std::mem::size_of::<f32>()) as u32,
        },
    }
}

/// Launch geometry for `m3_state_pass_fused`: one CTA per (b, h), its
/// threads over the head's hd*ds state elements, rounded up to whole warps
/// and capped at the 1024-thread block (an element loop covers the rest).
/// The thread count changes no bits: every element is one thread's chain.
pub fn state_pass_fused_cfg(
    batch: usize,
    nh: usize,
    hd: usize,
    ds: usize,
) -> cudarc::driver::LaunchConfig {
    let threads = (hd * ds).next_multiple_of(32).min(1024);
    cudarc::driver::LaunchConfig {
        grid_dim: (batch as u32, nh as u32, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Which kernel a stage of the chunked forward runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkArm {
    /// The kernel that stages whole state rows: the fused pre-state, the
    /// cooperative scan.
    Whole,
    /// The kernel that stages the state in slices, for any head and state
    /// width.
    Sliced,
}

/// The channels of a head a block of the sliced kernels takes (`M3PS_PB`
/// and `M3SS_PB` in `mamba3_chunked.cu`).
pub const SLICED_CHANNELS: usize = 64;

/// The floats of the sliced pre-state's tile, the kernel's `M3PS_FLOATS`:
/// a slice of the K rows and of the Q rows, the block's channels of the x
/// rows, and four per-row values.
pub const PRE_STATE_SLICED_FLOATS: usize = 2 * 64 * 20 + 64 * SLICED_CHANNELS + 4 * 64;

/// The floats of the sliced scan's tile, the kernel's `M3SS_FLOATS`: the
/// pair triangle, the slice arena (the Q, K and entering-state slices, or
/// the value rows after them) and five per-row values.
pub const SCAN_SLICED_FLOATS: usize = {
    let tri = (64 * 63 / 2 + 3) & !3;
    let slices = 2 * 64 * 20 + SLICED_CHANNELS * 20;
    let values = 64 * (SLICED_CHANNELS + 4);
    let arena = if slices > values { slices } else { values };
    tri + arena + 5 * 64
};

/// Launch geometry for the fused pre-state kernel, or None when its tile of
/// whole rows passes the `smem` bytes a block may stage
/// ([`Mamba3Kernels::smem_optin`]), its state width is not a whole number of
/// quads, or the chunk is wider than a block.
pub fn chunk_fused_cfg(
    batch: usize,
    n_chunks: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    chunk_size: usize,
    smem: usize,
) -> Option<cudarc::driver::LaunchConfig> {
    // K_scaled rows, x rows, each row's decay and the decay cumsum.
    let smem_bytes = (chunk_size * (ds + 4) + chunk_size * hd + 2 * chunk_size) * 4;
    if !ds.is_multiple_of(4) || chunk_size > 1024 || smem_bytes > smem {
        return None;
    }
    Some(cudarc::driver::LaunchConfig {
        grid_dim: ((batch * n_chunks) as u32, nh as u32, 1),
        block_dim: (chunk_size as u32, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    })
}

/// Launch geometry for the sliced pre-state kernel, which serves every head
/// and state width: a block of 256 threads per (sequence and chunk, head,
/// [`SLICED_CHANNELS`] channels of the head).
pub fn chunk_pre_state_sliced_cfg(
    batch: usize,
    n_chunks: usize,
    nh: usize,
    hd: usize,
) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: (
            (batch * n_chunks) as u32,
            nh as u32,
            hd.div_ceil(SLICED_CHANNELS) as u32,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (PRE_STATE_SLICED_FLOATS * 4) as u32,
    }
}

/// The pre-state kernel a shape takes, and its geometry: the fused kernel
/// where its tile fits the `smem` bytes a block may stage, the sliced
/// kernel everywhere else. Both take the same arguments and give the same
/// bits.
pub fn chunk_pre_state_cfg(
    batch: usize,
    n_chunks: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    chunk_size: usize,
    smem: usize,
) -> (ChunkArm, cudarc::driver::LaunchConfig) {
    match chunk_fused_cfg(batch, n_chunks, nh, hd, ds, chunk_size, smem) {
        Some(cfg) => (ChunkArm::Whole, cfg),
        None => (
            ChunkArm::Sliced,
            chunk_pre_state_sliced_cfg(batch, n_chunks, nh, hd),
        ),
    }
}

/// The tile of the cooperative scan the shapes it serves have always fit:
/// the 48 KB default window. The cooperative kernel and the static-tile
/// kernel the larger shapes ran on, whose arithmetic the sliced scan keeps,
/// left the last sums of an output to their compiler, which forms them
/// differently on some compilers and parts (on Blackwell under NVRTC 13.2
/// and later the static one folds the entering-state term into the diagonal
/// sum with one fused multiply-add, the cooperative one rounds it first), so
/// this bound is a numeric contract, not a budget: a shape keeps its kernel,
/// and its bits, whatever shared memory the device offers.
pub const COOP_SCAN_TILE_MAX: usize = 48 * 1024;

/// The largest staging `m3_dqktheta` takes. A thread of the stage holds its
/// whole row, so a bigger staged tile only lowers the blocks an SM keeps:
/// at the state width 48 in bf16 and at head 64 by state 64 the 5090
/// measured such tiles slower than the direct global reads, so they stay
/// direct. The staged and direct forms give the same bits.
pub const DQKTHETA_STAGING_MAX: usize = 48 * 1024;

/// Launch geometry for the cooperative scan, or None when its tile of whole
/// rows passes [`COOP_SCAN_TILE_MAX`] or the chunk is wider than 64.
pub fn chunk_scan_coop_cfg(
    batch: usize,
    n_chunks: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    chunk_size: usize,
) -> Option<cudarc::driver::LaunchConfig> {
    // q/k rows are padded to ds + 4 floats and the transposed entering
    // state's rows to hd + 4, to avoid bank conflicts. Keep this host
    // formula in lockstep with the kernel's layout.
    let smem_floats = chunk_size * (chunk_size - 1) / 2
        + 2 * chunk_size * (ds + 4)
        + chunk_size * hd
        + ds * (hd + 4)
        + 2 * chunk_size;
    let smem_bytes = smem_floats * std::mem::size_of::<f32>();
    (chunk_size <= 64 && smem_bytes <= COOP_SCAN_TILE_MAX).then(|| cudarc::driver::LaunchConfig {
        grid_dim: ((batch * n_chunks) as u32, nh as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    })
}

/// Launch geometry for the sliced scan, which serves every head and state
/// width: a block of 256 threads per (sequence and chunk, head,
/// [`SLICED_CHANNELS`] channels of the head).
pub fn chunk_scan_sliced_cfg(
    batch: usize,
    n_chunks: usize,
    nh: usize,
    hd: usize,
) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: (
            (batch * n_chunks) as u32,
            nh as u32,
            hd.div_ceil(SLICED_CHANNELS) as u32,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (SCAN_SLICED_FLOATS * 4) as u32,
    }
}

/// The intra-chunk scan a shape takes, and its geometry: the cooperative
/// kernel where its tile fits [`COOP_SCAN_TILE_MAX`], the sliced kernel,
/// with the static-tile kernel's arithmetic, everywhere else. Both take the
/// same arguments.
pub fn chunk_scan_cfg(
    batch: usize,
    n_chunks: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    chunk_size: usize,
) -> (ChunkArm, cudarc::driver::LaunchConfig) {
    match chunk_scan_coop_cfg(batch, n_chunks, nh, hd, ds, chunk_size) {
        Some(cfg) => (ChunkArm::Whole, cfg),
        None => (
            ChunkArm::Sliced,
            chunk_scan_sliced_cfg(batch, n_chunks, nh, hd),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host's tiles of the sliced kernels are the kernel's own: the
    /// source defines the same slice, channel block and chunk, and the
    /// floats each tile takes stay as counted.
    #[test]
    fn sliced_tiles_match_the_kernel_defines() {
        let source = include_str!("../../../kernels/mamba3_chunked.cu");
        for define in [
            "#define M3SS_NS 16",
            "#define M3SS_PB 64",
            "#define M3SS_CS 64",
            "#define M3PS_NS 16",
            "#define M3PS_PB 64",
            "#define M3PS_CS 64",
        ] {
            assert!(source.contains(define), "{define}");
        }
        assert_eq!(SLICED_CHANNELS, 64);
        assert_eq!(PRE_STATE_SLICED_FLOATS, 2560 + 4096 + 256);
        assert_eq!(SCAN_SLICED_FLOATS, 2016 + 4352 + 320);
    }
}
