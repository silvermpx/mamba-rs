//! Compile and register Mamba-3 SISO CUDA kernels.
//!
//! The Mamba-3 kernel registry, compiled via NVRTC at runtime and cached in
//! the same verified artifact format as the Mamba-1 registry.
//! Separate from Mamba SSM's `MambaKernels` — different pipeline, no conv1d.

use super::transport::{TransportAdmission, TransportKernels, transport_environment_admitted};
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::kernels::{CudaModuleAnchors, HalfKernel, TypedKernel};
use cudarc::driver::{CudaContext, CudaFunction};
use std::sync::Arc;

/// All compiled Mamba-3 SISO CUDA kernels.
pub struct Mamba3Kernels {
    _modules: CudaModuleAnchors,
    compiler_identity: crate::mamba_ssm::gpu::kernel_identity::CompilerIdentity,
    artifact_identity: crate::mamba_ssm::gpu::kernel_identity::ArtifactIdentity,
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
    pub m3_step_fwd: CudaFunction,
    pub m3_burnin_fwd: CudaFunction,
    /// The burn-in forward specialised on a state width, the twin of
    /// [`Self::m3_backward_seq_by_state`].
    pub m3_burnin_fwd_by_state: [(usize, CudaFunction); 4],
    pub m3_burnin_fwd_nosave: CudaFunction,
    pub m3_backward_seq: CudaFunction,
    /// The backward scan specialised on a state width, so its state loops
    /// carry a constant trip count. One per width the models here use; a
    /// width with no entry runs the general `m3_backward_seq` above.
    pub m3_backward_seq_by_state: [(usize, CudaFunction); 4],
    pub m3_reduce_d_d: CudaFunction,

    // ── Shared ops (mamba3_ops.cu) ──
    pub m3_split: CudaFunction,
    pub m3_split_bwd: CudaFunction,
    pub bcnorm_bwd: CudaFunction,
    pub bc_bias_add: CudaFunction,
    pub bc_bias_add_bwd: CudaFunction,
    pub m3_angle_dt_fwd_seq: CudaFunction,
    /// Chunk-parallel angle accumulation pair — replaces the sequential
    /// kernel on multi-chunk windows (its serial fp64 chain dominates
    /// the prefill profile at production shapes).
    pub m3_angle_chunk_sums: CudaFunction,
    pub m3_angle_chunk_apply: CudaFunction,
    pub m3_angle_dt_bwd_seq: CudaFunction,
    pub rope_fwd: CudaFunction,
    /// Fused bc_bias_add (B + C) + rope_fwd - one launch replaces the
    /// F4c/F4d/F4ef triple; biased saves still materialize for backward.
    pub m3_bias_rope_fwd: CudaFunction,
    pub rope_bwd: CudaFunction,
    pub m3_compute_abg: CudaFunction,
    pub m3_abg_bwd: CudaFunction,
    pub silu_gate_fwd: CudaFunction,
    pub silu_gate_bwd: CudaFunction,
    pub rmsnorm_gated_fwd: CudaFunction,
    pub rmsnorm_gated_bwd: CudaFunction,

    // ── Shared kernels from norms.cu + elementwise.cu (used by training pipeline) ──
    pub rmsnorm_fwd: CudaFunction,
    pub rmsnorm_bwd: CudaFunction,
    pub colsum_accumulate: CudaFunction,
    /// Segmented column sum - the batched pooled route.
    pub colsum_segments: CudaFunction,
    /// Deterministic axis-0 reduction — finalises Rule-B per-sample partials
    /// into f32 master-grad slots. Replaces atomicAdd accumulators used in
    /// M3 backward (dD from m3_dqkv, d_angles_raw/d_dt_angle from
    /// m3_angle_dt_bwd_seq, and d_scale from rmsnorm_bwd).
    pub reduce_sum_axis0: CudaFunction,
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
    pub m3_preprocess_chunks: CudaFunction,
    pub m3_da_cumsum: CudaFunction,
    pub m3_chunk_state_fwd: CudaFunction,
    pub m3_state_passing_fwd: CudaFunction,
    /// Trapezoidal boundary fold for state-carrying prefill: seeds the
    /// entering SSM state with `v_state (x) k_state * dt0 * (1 - trap0)`
    /// so the beta term's one-step reach-back across the window seam is
    /// honored (reference: mamba3_siso_fwd.py HAS_INITIAL_STATES).
    pub m3_chunk_entering_state: CudaFunction,
    pub m3_writeback_parallel_states: CudaFunction,
    pub m3_chunk_scan_fwd: CudaFunction,
    /// Cooperative-block twin: one head per 128-thread block, triangle
    /// tile + staged operands in dynamic smem. Routed when the smem total
    /// fits the 48 KB budget; bit-identical to the original.
    pub m3_chunk_scan_fwd_coop: CudaFunction,
    // m3_chunk_scan_bwd, m3_state_passing_bwd, m3_chunk_state_bwd,
    // m3_cumsum_bwd removed — dead code replaced by monolithic m3_dqkv +
    // m3_dqktheta path below.
    pub m3_extract_da_cs_sum: CudaFunction,
    pub m3_dqkv: CudaFunction,
    /// Per-chunk B terms of the reverse d_state recurrence (f32/typed
    /// activation reads; f32 output).
    pub m3_dqkv_state_terms_typed: TypedKernel,
    /// Serial per-(b,h,p,n) reverse fold producing each chunk's
    /// entering d_state.
    pub m3_dstate_passing_bwd: CudaFunction,
    pub m3_dqktheta: CudaFunction,
    pub m3_ddt_dtrap: CudaFunction,
    pub m3_final_grads: CudaFunction,

    // ── Typed variants for end-to-end bf16/f16 inference ──
    /// 8-way split + fused softplus/sigmoid, bf16/f16 proj and activation outputs.
    /// Coefficient outputs (dt, a_val, trap, angles, raw saves) stay f32.
    pub m3_split_typed: TypedKernel,
    /// Fused B+C variant of bcnorm_fwd_typed (2× grid via blockIdx.y).
    pub bcnorm_fwd_bc_typed: TypedKernel,
    /// Fused B+C norm, f32 lane (the decode step merges its two bcnorm
    /// launches through it).
    pub bcnorm_fwd_bc_f32: CudaFunction,
    /// Fused typed twin of bias + rope (round-trip contract preserved).
    pub m3_bias_rope_fwd_typed: TypedKernel,
    /// Plain SiLU gate (no norm), half I/O.
    pub silu_gate_fwd_typed: TypedKernel,
    /// SiLU-gate backward (typed twins share the f32 kernel's argument
    /// order and factored d_silu form).
    pub silu_gate_bwd_typed: TypedKernel,
    /// RMSNorm-gated output (half I/O, f32 weight/rms_vals).
    pub rmsnorm_gated_fwd_typed: TypedKernel,
    /// M3 SSM step — shared with training, already templated in mamba3_siso.cu.
    pub m3_step_fwd_typed: TypedKernel,
    /// M3 burnin forward (training) — sequential T-loop SSM with activation
    /// saves. Typed x/k/q/y; f32 state + alpha/beta/gamma + D + saves.
    pub m3_burnin_fwd_typed_bf16: CudaFunction,
    pub m3_burnin_fwd_typed_f16: CudaFunction,
    burnin_fwd_typed_by_state: Option<HalfKernel>,

    // -- Typed M3 sequential backward kernels --
    /// RmsNorm over B/C groups, typed dy → typed d_B; f32 rms + weight +
    /// d_weight master-grad accumulator.
    pub bcnorm_bwd_typed: TypedKernel,
    /// Per-head bias reduction from typed d_B_biased → typed d_B_normed
    /// (expand groups backward). Bias grad handled by reduce_bias_typed.
    pub bc_bias_add_bwd_typed: TypedKernel,
    /// RoPE rotation backward, typed B/C grads + saved B/C, f32 angle saves.
    pub rope_bwd_typed: TypedKernel,
    /// 8-way split backward: assemble typed d_proj from typed d_z/d_x/
    /// d_B_raw/d_C_raw plus f32 dd_dt/dd_a/trap/angles.
    pub m3_split_bwd_typed: TypedKernel,
    /// RMSNorm-gated backward (`out = RMSNorm(y) * weight * SiLU(z)`) —
    /// typed d_y/d_z/d_out/y/z; f32 weight + d_weight (per-sample,
    /// reduced later).
    pub rmsnorm_gated_bwd_typed: TypedKernel,
    // -- Typed M3 "final grad" kernels (HIGHEST RISK) --
    /// Huge dqkv kernel with smem tiles — typed Q_rot/K_scaled/V_in/dO;
    /// all 6 grad outputs stay f32 (atomicAdd on dD; master grads on others).
    pub m3_dqkv_typed: TypedKernel,
    /// Inverse-RoPE + bias backward — typed Q_raw/K_raw; 7 grad outputs f32.
    pub m3_dqktheta_typed: TypedKernel,

    // m3_chunk_scan_bwd_typed + m3_chunk_state_bwd_typed
    // removed (dead — the typed backward path, like the f32 path, now routes
    // through m3_dqkv_typed + m3_dqktheta_typed monolithic kernels).

    // -- Typed M3 chunked parallel forward kernels --
    /// Per-chunk gamma/scale + qk_dot + K prescale. Typed K/Q/K_scaled,
    /// f32 DT/trap_sig/qk_dot/scale/gamma (these are scalars computed in
    /// float and reused by the scan path).
    pub m3_preprocess_chunks_typed: TypedKernel,
    /// Per-chunk SSM state matmul (typed x, typed K_scaled, f32 dA_cumsum,
    /// f32 states_out — BPTT state MUST remain f32 per Tri Dao invariant).
    pub m3_chunk_state_fwd_typed: TypedKernel,
    /// Fused preprocess + chunk_state (3b): one kernel computes K_scaled/
    /// qk_dot/scale/gamma AND the chunk states, keeping K_scaled in smem
    /// across the seam instead of round-tripping it through L2. Launched
    /// only when `chunk_fused_cfg` returns Some; otherwise the pair runs.
    pub m3_chunk_pre_state_fused_typed: TypedKernel,
    /// Persist final states to ssm_state/k_state/v_state (all f32 persistent
    /// buffers). Typed inputs k_flat/x_flat.
    pub m3_writeback_parallel_states_typed: TypedKernel,
    /// Intra-chunk output: typed y_out/x/Q/K_scaled; f32 qk_dot/dA_cumsum/
    /// prev_states/D.
    pub m3_chunk_scan_fwd_typed: TypedKernel,
    /// Cooperative typed twin (see `m3_chunk_scan_fwd_coop`).
    pub m3_chunk_scan_fwd_coop_typed: TypedKernel,

    /// Shared from M1: f32 residual → half post-norm (identical kernel, reused).
    pub rmsnorm_fwd_f32in_typed: HalfKernel,
    /// Shared from M1: f32 residual += half branch (stays f32).
    pub residual_add_f32_typed: HalfKernel,
    /// Shared from M1: gather last timestep of B×T×D into B×D, dtype-preserving.
    pub gather_last_timestep_typed: TypedKernel,
}

impl Mamba3Kernels {
    #[doc(hidden)]
    pub fn compiler_identity(&self) -> crate::mamba_ssm::gpu::kernel_identity::CompilerIdentity {
        self.compiler_identity
    }

    #[doc(hidden)]
    pub fn artifact_identity(&self) -> crate::mamba_ssm::gpu::kernel_identity::ArtifactIdentity {
        self.artifact_identity
    }

    /// Compile Mamba-3 kernels with the default state capacity of 64. Models with a
    /// larger `d_state` use [`Self::compile_with_state_cap`].
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        Self::compile_with_state_cap(ctx, arch, 64)
    }

    /// The backward scan to launch for a model of this state width: the
    /// specialisation when one exists, the general entry otherwise. The
    /// two compute the same thing; the specialisation is faster because
    /// its loop bounds are constants the compiler can unroll.
    pub fn backward_seq_for_state(&self, d_state: usize) -> &CudaFunction {
        self.m3_backward_seq_by_state
            .iter()
            .find(|(width, _)| *width == d_state)
            .map(|(_, function)| function)
            .unwrap_or(&self.m3_backward_seq)
    }

    /// The burn-in forward to launch for a model of this state width.
    pub fn burnin_fwd_for_state(&self, d_state: usize) -> &CudaFunction {
        self.m3_burnin_fwd_by_state
            .iter()
            .find(|(width, _)| *width == d_state)
            .map(|(_, function)| function)
            .unwrap_or(&self.m3_burnin_fwd)
    }

    /// Select the sequential training forward kernel for an activation dtype
    /// and state width. Persistent state and backward saves remain FP32.
    ///
    /// BF16/F16 use constant-width loops on qualified Ada compiler targets;
    /// other devices, compilers and widths retain the general typed kernel.
    /// F32 uses the existing F32 state-width selector.
    pub fn burnin_fwd_typed_for_state(&self, dtype: WeightDtype, d_state: usize) -> &CudaFunction {
        if dtype == WeightDtype::F32 {
            return self.burnin_fwd_for_state(d_state);
        }
        if matches!(d_state, 8 | 16 | 32 | 64)
            && d_state <= self.state_cap
            && let Some(kernels) = &self.burnin_fwd_typed_by_state
        {
            return kernels.get(dtype);
        }
        match dtype {
            WeightDtype::Bf16 => &self.m3_burnin_fwd_typed_bf16,
            WeightDtype::F16 => &self.m3_burnin_fwd_typed_f16,
            WeightDtype::F32 => unreachable!("F32 uses its own selector"),
        }
    }

    /// Compile all Mamba-3 kernels. `state_cap` sizes the per-thread
    /// state register arrays (see
    /// [`crate::mamba_ssm::gpu::kernels::state_capacity`]); raising it
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
            include_str!("../../../kernels/mamba3_siso.cu"),
            include_str!("../../../kernels/mamba3_ops.cu"),
            include_str!("../../../kernels/mamba3_chunked.cu"),
            // Shared kernels needed by training pipeline
            include_str!("../../../kernels/norms.cu"),
            include_str!("../../../kernels/elementwise.cu"),
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
        let (nv_major, nv_minor) = crate::mamba_ssm::gpu::kernels::nvrtc_version();
        let mut option_strings = vec![
            "--fmad=true".to_string(),
            "--extra-device-vectorization".to_string(),
            // Mirrors the M1 compiler: strip device assert() trap
            // checks (none in the m3 sources today, but shared
            // headers may grow them); asserts compute no values.
            "-DNDEBUG".to_string(),
            format!("-DMAMBA_RS_STATE_CAP={state_cap}"),
        ];
        option_strings.extend(
            crate::mamba_ssm::gpu::kernel_identity::deterministic_nvrtc_options(
                (nv_major, nv_minor),
                "1295203121",
            ),
        );
        let include_paths = crate::mamba_ssm::gpu::kernels::cuda_include_paths();
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: option_strings.clone(),
            include_paths: include_paths.clone(),
            ..Default::default()
        };

        let nvrtc_library_domain = crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain();
        let header_manifest = crate::mamba_ssm::gpu::kernel_identity::header_manifest(
            combined.as_bytes(),
            &include_paths,
        );
        let mut argv = vec![format!("--gpu-architecture={arch}").into_bytes()];
        argv.extend(option_strings.iter().map(|value| value.as_bytes().to_vec()));
        let key_material = crate::mamba_ssm::gpu::kernel_identity::CompileKeyMaterial {
            module_kind: crate::mamba_ssm::gpu::kernel_identity::ModuleKind::Mamba3Combined,
            source: combined.as_bytes().to_vec(),
            target: arch.as_bytes().to_vec(),
            argv,
            header_manifest: header_manifest.clone(),
            nvrtc_version: (nv_major, nv_minor),
            nvrtc_library_domain: nvrtc_library_domain.clone(),
            output_kind: crate::mamba_ssm::gpu::kernel_identity::ArtifactKind::Ptx,
            composer_revision: crate::mamba_ssm::gpu::kernel_identity::COMPOSER_REVISION,
            compiler_revision: crate::mamba_ssm::gpu::kernel_identity::COMPILER_REVISION,
            numeric_abi_revision: crate::mamba_ssm::gpu::kernel_identity::NUMERIC_ABI_REVISION,
            schedule_revision: crate::mamba_ssm::gpu::kernel_identity::SCHEDULE_REVISION,
        };
        let invocation_digest = key_material.invocation_digest();
        let cache_key = key_material.digest();
        let cache_path = cache_key.and_then(|key| {
            crate::mamba_ssm::gpu::kernels::kernel_cache_dir().map(|directory| {
                directory.join(format!(
                    "mamba3-kernels-v2-{}.bin",
                    crate::mamba_ssm::gpu::kernel_identity::digest_hex(&key)
                ))
            })
        });

        let mut loaded = None;
        if let (Some(path), Some(key)) = (&cache_path, cache_key)
            && let Some(hit) = crate::mamba_ssm::gpu::kernel_identity::read_cache(
                path,
                key,
                crate::mamba_ssm::gpu::kernel_identity::ArtifactKind::Ptx,
            )
            && let Ok(source) =
                crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_from_cache(hit.payload)
            && let Ok(module) = ctx.load_module(cudarc::nvrtc::Ptx::from_src(source))
            && crate::mamba_ssm::gpu::kernel_identity::cache_hit_header_closure_is_current(
                combined.as_bytes(),
                &include_paths,
                &header_manifest,
            )
            && nvrtc_library_domain.as_deref().is_some_and(
                crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain_is_current,
            )
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
                let ptx_source =
                    crate::mamba_ssm::gpu::kernel_identity::canonical_ptx_image(ptx_image)?;
                if !crate::mamba_ssm::gpu::kernel_identity::header_manifest_is_current(
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
                    && !crate::mamba_ssm::gpu::kernel_identity::nvrtc_library_domain_is_current(
                        domain,
                    )
                {
                    return Err(
                        "NVRTC libraries changed during M3 compilation; retry initialization"
                            .into(),
                    );
                }
                let artifact_digest = crate::mamba_ssm::gpu::kernel_identity::FramedSha256::bytes(
                    ptx_source.as_bytes(),
                );
                if let (Some(path), Some(key)) = (&cache_path, cache_key) {
                    crate::mamba_ssm::gpu::kernel_identity::publish_cache(
                        path,
                        key,
                        crate::mamba_ssm::gpu::kernel_identity::ArtifactKind::Ptx,
                        ptx_source.as_bytes(),
                    );
                }
                let module = ctx
                    .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
                    .map_err(|e| format!("M3 module load failed: {e:?}"))?;
                (module, artifact_digest)
            }
        };
        let compiler_identity = crate::mamba_ssm::gpu::kernel_identity::CompilerIdentity {
            source_digest: crate::mamba_ssm::gpu::kernel_identity::FramedSha256::bytes(
                combined.as_bytes(),
            ),
            invocation_digest,
            header_manifest_digest: crate::mamba_ssm::gpu::kernel_identity::FramedSha256::new(
                b"cuda-header-manifest.v1",
            )
            .optional(b"manifest", header_manifest.as_deref())
            .finish(),
            target: crate::mamba_ssm::gpu::kernel_identity::CudaTarget::new(arch)?,
            nvrtc_version: (nv_major, nv_minor),
            nvrtc_library_domain: crate::mamba_ssm::gpu::kernel_identity::FramedSha256::new(
                b"nvrtc-library-set-identity.v2",
            )
            .optional(b"domain", nvrtc_library_domain.as_deref())
            .finish(),
            nvrtc_library_known: nvrtc_library_domain.is_some(),
            output_kind: crate::mamba_ssm::gpu::kernel_identity::ArtifactKind::Ptx,
            composer_revision: crate::mamba_ssm::gpu::kernel_identity::COMPOSER_REVISION,
            compiler_revision: crate::mamba_ssm::gpu::kernel_identity::COMPILER_REVISION,
            numeric_abi_revision: crate::mamba_ssm::gpu::kernel_identity::NUMERIC_ABI_REVISION,
            schedule_revision: crate::mamba_ssm::gpu::kernel_identity::SCHEDULE_REVISION,
        };
        let artifact_identity = crate::mamba_ssm::gpu::kernel_identity::ArtifactIdentity {
            module_kind: crate::mamba_ssm::gpu::kernel_identity::ModuleKind::Mamba3Combined,
            artifact_kind: crate::mamba_ssm::gpu::kernel_identity::ArtifactKind::Ptx,
            compile_key: invocation_digest,
            artifact_digest,
        };
        let module_identity =
            crate::mamba_ssm::gpu::kernel_identity::digest_hex(&invocation_digest);

        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("M3 kernel '{name}' not found: {e:?}"))
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
            Some(HalfKernel {
                bf16: get("m3_burnin_fwd_bf16_by_state")?,
                f16: get("m3_burnin_fwd_f16_by_state")?,
            })
        } else {
            None
        };

        let kernels = Self {
            transport,
            burnin_fwd_typed_by_state,
            module_identity,
            state_cap,
            compiler_identity,
            artifact_identity,
            // Sequential SSM
            m3_step_fwd: get("m3_step_fwd")?,
            m3_burnin_fwd: get("m3_burnin_fwd")?,
            m3_burnin_fwd_by_state: [
                (8, get("m3_burnin_fwd_ds8")?),
                (16, get("m3_burnin_fwd_ds16")?),
                (32, get("m3_burnin_fwd_ds32")?),
                (64, get("m3_burnin_fwd_ds64")?),
            ],
            m3_burnin_fwd_nosave: get("m3_burnin_fwd_nosave")?,
            m3_backward_seq: get("m3_backward_seq")?,
            m3_backward_seq_by_state: [
                (8, get("m3_backward_seq_ds8")?),
                (16, get("m3_backward_seq_ds16")?),
                (32, get("m3_backward_seq_ds32")?),
                (64, get("m3_backward_seq_ds64")?),
            ],
            m3_reduce_d_d: get("m3_reduce_d_D")?,

            // Shared ops
            m3_split: get("m3_split")?,
            m3_split_bwd: get("m3_split_bwd")?,
            bcnorm_bwd: get("bcnorm_bwd")?,
            bc_bias_add: get("bc_bias_add")?,
            bc_bias_add_bwd: get("bc_bias_add_bwd")?,
            m3_angle_dt_fwd_seq: get("m3_angle_dt_fwd_seq")?,
            m3_angle_chunk_sums: get("m3_angle_chunk_sums")?,
            m3_angle_chunk_apply: get("m3_angle_chunk_apply")?,
            m3_angle_dt_bwd_seq: get("m3_angle_dt_bwd_seq")?,
            rope_fwd: get("rope_fwd")?,
            m3_bias_rope_fwd: get("m3_bias_rope_fwd")?,
            rope_bwd: get("rope_bwd")?,
            m3_compute_abg: get("m3_compute_abg")?,
            m3_abg_bwd: get("m3_abg_bwd")?,
            silu_gate_fwd: get("silu_gate_fwd")?,
            silu_gate_bwd: get("silu_gate_bwd")?,
            rmsnorm_gated_fwd: get("rmsnorm_gated_forward")?,
            rmsnorm_gated_bwd: get("rmsnorm_gated_backward")?,

            // Shared (norms.cu + elementwise.cu)
            rmsnorm_fwd: get("rmsnorm_forward")?,
            rmsnorm_bwd: get("rmsnorm_backward")?,
            colsum_accumulate: get("colsum_accumulate")?,
            colsum_segments: get("colsum_segments")?,
            reduce_sum_axis0: get("reduce_sum_axis0")?,
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
            m3_preprocess_chunks: get("m3_preprocess_chunks")?,
            m3_da_cumsum: get("m3_dA_cumsum")?,
            m3_chunk_state_fwd: get("m3_chunk_state_fwd")?,
            m3_state_passing_fwd: get("m3_state_passing_fwd")?,
            m3_chunk_entering_state: get("m3_chunk_entering_state")?,
            m3_writeback_parallel_states: get("m3_writeback_parallel_states")?,
            m3_chunk_scan_fwd: get("m3_chunk_scan_fwd")?,
            m3_chunk_scan_fwd_coop: get("m3_chunk_scan_fwd_coop")?,
            m3_extract_da_cs_sum: get("m3_extract_da_cs_sum")?,
            m3_dqkv: get("m3_dqkv")?,
            m3_dqkv_state_terms_typed: TypedKernel {
                f32: get("m3_dqkv_state_terms")?,
                bf16: get("m3_dqkv_state_terms_bf16")?,
                f16: get("m3_dqkv_state_terms_f16")?,
            },
            m3_dstate_passing_bwd: get("m3_dstate_passing_bwd")?,
            m3_dqktheta: get("m3_dqktheta")?,
            m3_ddt_dtrap: get("m3_ddt_dtrap")?,
            m3_final_grads: get("m3_final_grads")?,

            // Typed variants for mixed-dtype inference
            m3_split_typed: TypedKernel {
                f32: get("m3_split")?,
                bf16: get("m3_split_bf16")?,
                f16: get("m3_split_f16")?,
            },
            bcnorm_fwd_bc_typed: TypedKernel {
                f32: get("bcnorm_fwd_bc_f32")?,
                bf16: get("bcnorm_fwd_bc_bf16")?,
                f16: get("bcnorm_fwd_bc_f16")?,
            },
            bcnorm_fwd_bc_f32: get("bcnorm_fwd_bc_f32")?,
            m3_bias_rope_fwd_typed: TypedKernel {
                f32: get("m3_bias_rope_fwd")?,
                bf16: get("m3_bias_rope_fwd_bf16")?,
                f16: get("m3_bias_rope_fwd_f16")?,
            },
            silu_gate_fwd_typed: TypedKernel {
                f32: get("silu_gate_fwd")?,
                bf16: get("silu_gate_fwd_bf16")?,
                f16: get("silu_gate_fwd_f16")?,
            },
            silu_gate_bwd_typed: TypedKernel {
                f32: get("silu_gate_bwd")?,
                bf16: get("silu_gate_bwd_bf16")?,
                f16: get("silu_gate_bwd_f16")?,
            },
            rmsnorm_gated_fwd_typed: TypedKernel {
                f32: get("rmsnorm_gated_forward")?,
                bf16: get("rmsnorm_gated_forward_bf16")?,
                f16: get("rmsnorm_gated_forward_f16")?,
            },
            m3_step_fwd_typed: TypedKernel {
                f32: get("m3_step_fwd")?,
                bf16: get("m3_step_fwd_bf16")?,
                f16: get("m3_step_fwd_f16")?,
            },
            m3_burnin_fwd_typed_bf16: get("m3_burnin_fwd_bf16")?,
            m3_burnin_fwd_typed_f16: get("m3_burnin_fwd_f16")?,
            bcnorm_bwd_typed: TypedKernel {
                f32: get("bcnorm_bwd")?,
                bf16: get("bcnorm_bwd_bf16")?,
                f16: get("bcnorm_bwd_f16")?,
            },
            bc_bias_add_bwd_typed: TypedKernel {
                f32: get("bc_bias_add_bwd")?,
                bf16: get("bc_bias_add_bwd_bf16")?,
                f16: get("bc_bias_add_bwd_f16")?,
            },
            rope_bwd_typed: TypedKernel {
                f32: get("rope_bwd")?,
                bf16: get("rope_bwd_bf16")?,
                f16: get("rope_bwd_f16")?,
            },
            m3_split_bwd_typed: TypedKernel {
                f32: get("m3_split_bwd")?,
                bf16: get("m3_split_bwd_bf16")?,
                f16: get("m3_split_bwd_f16")?,
            },
            rmsnorm_gated_bwd_typed: TypedKernel {
                f32: get("rmsnorm_gated_backward")?,
                bf16: get("rmsnorm_gated_backward_bf16")?,
                f16: get("rmsnorm_gated_backward_f16")?,
            },
            m3_dqkv_typed: TypedKernel {
                f32: get("m3_dqkv")?,
                bf16: get("m3_dqkv_bf16")?,
                f16: get("m3_dqkv_f16")?,
            },
            m3_dqktheta_typed: TypedKernel {
                f32: get("m3_dqktheta")?,
                bf16: get("m3_dqktheta_bf16")?,
                f16: get("m3_dqktheta_f16")?,
            },

            m3_preprocess_chunks_typed: TypedKernel {
                f32: get("m3_preprocess_chunks")?,
                bf16: get("m3_preprocess_chunks_bf16")?,
                f16: get("m3_preprocess_chunks_f16")?,
            },
            m3_chunk_state_fwd_typed: TypedKernel {
                f32: get("m3_chunk_state_fwd")?,
                bf16: get("m3_chunk_state_fwd_bf16")?,
                f16: get("m3_chunk_state_fwd_f16")?,
            },
            m3_chunk_pre_state_fused_typed: TypedKernel {
                f32: get("m3_chunk_pre_state_fused")?,
                bf16: get("m3_chunk_pre_state_fused_bf16")?,
                f16: get("m3_chunk_pre_state_fused_f16")?,
            },
            m3_writeback_parallel_states_typed: TypedKernel {
                f32: get("m3_writeback_parallel_states")?,
                bf16: get("m3_writeback_parallel_states_bf16")?,
                f16: get("m3_writeback_parallel_states_f16")?,
            },
            m3_chunk_scan_fwd_coop_typed: TypedKernel {
                f32: get("m3_chunk_scan_fwd_coop")?,
                bf16: get("m3_chunk_scan_fwd_coop_bf16")?,
                f16: get("m3_chunk_scan_fwd_coop_f16")?,
            },
            m3_chunk_scan_fwd_typed: TypedKernel {
                f32: get("m3_chunk_scan_fwd")?,
                bf16: get("m3_chunk_scan_fwd_bf16")?,
                f16: get("m3_chunk_scan_fwd_f16")?,
            },

            rmsnorm_fwd_f32in_typed: HalfKernel {
                bf16: get("rmsnorm_forward_f32in_bf16")?,
                f16: get("rmsnorm_forward_f32in_f16")?,
            },
            residual_add_f32_typed: HalfKernel {
                bf16: get("residual_add_f32_bf16")?,
                f16: get("residual_add_f32_f16")?,
            },
            gather_last_timestep_typed: TypedKernel {
                f32: get("gather_last_timestep_f32")?,
                bf16: get("gather_last_timestep_bf16")?,
                f16: get("gather_last_timestep_f16")?,
            },

            _modules: CudaModuleAnchors::new(vec![module]),
        };

        // The chunked-backward kernel's dynamic shared memory grows
        // linearly with d_state (two chunk-by-d_state operand tiles
        // dominate) and exceeds the 48 KB default past d_state ~ 70.
        // Opt these functions in to the device's extended budget so a
        // larger state runs on the same code path. Best effort: on a
        // device without the budget the attribute call fails here and
        // an oversized launch later fails loudly with its own error —
        // never silently.
        // Unconditional: the chunked-backward tiles exceed the 48 KB
        // default from ordinary shapes too (e.g. d_state 64 with headdim
        // 32 needs ~67 KB), not only at raised state capacities.
        {
            use cudarc::driver::sys::CUfunction_attribute_enum as FnAttr;
            // 99 KB: consumer parts (sm_89/sm_120) cap the per-block opt-in
            // near 99-100 KB; the triangle-packed pair matrices fit
            // comfortably at CS=64 tile sizes.
            let budget: i32 = 99 * 1024;
            for f in [
                &kernels.m3_dqkv,
                &kernels.m3_dqkv_typed.f32,
                &kernels.m3_dqkv_typed.bf16,
                &kernels.m3_dqkv_typed.f16,
            ] {
                let _ = f.set_attribute(
                    FnAttr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    budget,
                );
            }
            if let Some(transport) = &kernels.transport.dqkv {
                for f in [&transport.f32, &transport.bf16, &transport.f16] {
                    let _ = f.set_attribute(
                        FnAttr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        budget,
                    );
                }
            }
        }
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

/// Launch geometry for `m3_chunk_state_fwd`.
///
/// The kernel derives its thread layout from this block shape. When the
/// shared tile fits and `ds` is divisible by four, each thread owns four
/// ascending-timestep accumulation chains; other shapes use the scalar layout.
pub fn chunk_state_cfg(
    batch: usize,
    n_chunks: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    chunk_size: usize,
) -> cudarc::driver::LaunchConfig {
    // Quad layout: block (hd, ds/4, 2 heads) with the block's x/K/dA
    // chunk slices staged in dynamic smem (the kernel is L2-bound; the
    // staging collapses its redundant global reads). Legacy layout for
    // shapes that don't fit the smem budget or an odd ds.
    let heads = 2usize;
    let smem_bytes = heads * (chunk_size * hd + chunk_size * ds + chunk_size) * 4;
    if ds.is_multiple_of(4) && hd * (ds / 4) * heads <= 1024 && smem_bytes <= 48 * 1024 {
        cudarc::driver::LaunchConfig {
            grid_dim: ((batch * n_chunks) as u32, nh.div_ceil(heads) as u32, 1),
            block_dim: (hd as u32, (ds / 4) as u32, heads as u32),
            shared_mem_bytes: smem_bytes as u32,
        }
    } else {
        cudarc::driver::LaunchConfig {
            grid_dim: ((batch * n_chunks) as u32, nh.div_ceil(2) as u32, 1),
            block_dim: (hd as u32, 2, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Launch geometry for the FUSED preprocess+chunk_state kernel, or None
/// when the shape must run the unfused pair (odd ds - phase B is float4;
/// oversized smem; chunk_size beyond a block). Single source: the kernel
/// derives everything from blockDim/args, every call site MUST use this.
pub fn chunk_fused_cfg(
    batch: usize,
    n_chunks: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    chunk_size: usize,
) -> Option<cudarc::driver::LaunchConfig> {
    let smem_bytes = (chunk_size * (ds + 4) + chunk_size * hd + chunk_size) * 4;
    if !ds.is_multiple_of(4) || chunk_size > 1024 || smem_bytes > 48 * 1024 {
        return None;
    }
    Some(cudarc::driver::LaunchConfig {
        grid_dim: ((batch * n_chunks) as u32, nh as u32, 1),
        block_dim: (chunk_size as u32, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    })
}

/// Select the cooperative chunk scan when its shared tile fits the default
/// 48 KiB budget. Wider shapes keep the two-head static-tile kernel.
/// Returns `(use_coop, config)`; both kernels take the same arguments.
pub fn chunk_scan_cfg(
    batch: usize,
    n_chunks: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    chunk_size: usize,
) -> (bool, cudarc::driver::LaunchConfig) {
    // q/k/ps rows are padded to ds + 4 floats to avoid bank conflicts.
    // Keep this host formula in lockstep with the kernel's layout.
    let smem_floats = chunk_size * (chunk_size - 1) / 2
        + 2 * chunk_size * (ds + 4)
        + chunk_size * hd
        + hd * (ds + 4)
        + 2 * chunk_size;
    let smem_bytes = smem_floats * std::mem::size_of::<f32>();
    if chunk_size <= 64 && smem_bytes <= 48 * 1024 {
        (
            true,
            cudarc::driver::LaunchConfig {
                grid_dim: ((batch * n_chunks) as u32, nh as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: smem_bytes as u32,
            },
        )
    } else {
        (
            false,
            cudarc::driver::LaunchConfig {
                grid_dim: ((batch * n_chunks) as u32, nh.div_ceil(2) as u32, 1),
                block_dim: (hd as u32, 2, 1),
                shared_mem_bytes: 0,
            },
        )
    }
}
