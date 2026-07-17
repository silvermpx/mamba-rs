//! AdamW optimizer in f32 master precision.
//!
//! Mirrors `torch.optim.AdamW` (Loshchilov & Hutter, "Decoupled Weight Decay
//! Regularization", ICLR 2019). Designed for AMP/mixed-precision training:
//!
//! - **Master weights** stay in f32 (per-tensor `GpuBuffer`s in
//!   `GpuMambaTrainWeights` / `GpuMamba3Weights`).
//! - **Gradients** are accumulated in f32 (flat `GpuBuffer` in
//!   `GpuMambaGrads.flat` / `GpuMamba3Grads.flat`), which the loss-scaler
//!   has already unscaled before this step runs.
//! - **Optimizer state** (`m`, `v`) lives in f32 alongside grads. Storing
//!   Adam moments in bf16 empirically diverges within ~1k SSM steps;
//!   PyTorch / DeepSpeed / Apex / CleanRL all keep them in f32.
//!
//! ## Usage
//! ```ignore
//! let mut adam = GpuAdamW::new(&ctx.stream, grads.flat.len())?
//!     .with_lr(3e-4)
//!     .with_weight_decay(1e-2);
//!
//! for _ in 0..n_steps {
//!     grads.zero(&ctx.stream)?;
//!     forward_backward(...)?;
//!     // (optional) loss-scaler unscale + grad clip here
//!     adam.step_m1(&ctx, &kernels, &mut weights.master, &grads)?;
//!     weights.sync_master_to_compute(&ctx)?;
//! }
//! ```
//!
//! ## Precision rationale
//! Default `f32` for everything inside the optimizer matches:
//! - PyTorch `torch.optim.AdamW` (always f32 even under AMP)
//! - NVIDIA Apex `FusedAdam`
//! - DeepSpeed ZeRO stage 0/1/2
//! - Hugging Face `accelerate.optimizer.AcceleratedOptimizer`
//!
//! Bias-correction factors `1/(1-β1ᵗ)` and `1/(1-β2ᵗ)` are computed CPU-side
//! (one `powf` per step) so the kernel sees plain scalar multiplies — same
//! as PyTorch's `_single_tensor_adamw` non-capturable path.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, PushKernelArg};

use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GradSlice};
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::launch::grid_1d;

/// 2-element device buffer holding `[bias_c1, bias_c2]` for the
/// CUDA-Graph-capturable AdamW kernel. CPU writes the next-step values
/// here BEFORE each graph replay so the captured kernel reads fresh
/// bias-correction factors via a stable device pointer.
///
/// Mirrors the device-side state PyTorch's `AdamW(capturable=True)` keeps
/// per param group.
pub struct AdamWBiasFactors {
    pub buf: GpuBuffer,
}

impl AdamWBiasFactors {
    /// Allocate the 2-element device buffer. Initialises to `[1.0, 1.0]`
    /// — the neutral bias-correction value (`1 / (1 - β^1)` for step 1 is
    /// `~10` for β1=0.9 but the kernel tolerates any finite factor; 1.0
    /// produces an "Adam without bias correction" update if the buffer is
    /// ever read before `write()`). This guards against a silent-wrong
    /// update path where a zero-init buffer would make `m_hat = v_hat = 0`
    /// and the captured kernel would apply ONLY weight decay, no Adam
    /// step, on the first replay if `write()` was forgotten.
    pub fn new(stream: &Arc<CudaStream>) -> Result<Self, String> {
        let buf = GpuBuffer::zeros(stream, 2)?;
        let mut this = Self { buf };
        this.write(stream, 1.0, 1.0)?;
        Ok(this)
    }

    /// Write `(bc1, bc2)` for the upcoming step. Async H2D — the next
    /// graph replay will see these values via the device pointer.
    pub fn write(&mut self, stream: &Arc<CudaStream>, bc1: f32, bc2: f32) -> Result<(), String> {
        debug_assert!(
            bc1.is_finite() && bc2.is_finite() && bc1 > 0.0 && bc2 > 0.0,
            "AdamWBiasFactors::write got non-finite or non-positive values: bc1={bc1} bc2={bc2}"
        );
        self.buf.upload(stream, &[bc1, bc2])
    }

    pub fn ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.buf.cached_ptr()
    }
}

/// Device pointers for one AdamW tensor update: weight + grad + the
/// matching `m`/`v` slices (offset into the flat arenas by the same
/// amount as `grad` is into `grads.flat`).
#[derive(Clone, Copy, Debug)]
pub struct AdamWParamPtrs {
    pub weight: cudarc::driver::sys::CUdeviceptr,
    pub grad: cudarc::driver::sys::CUdeviceptr,
    pub m: cudarc::driver::sys::CUdeviceptr,
    pub v: cudarc::driver::sys::CUdeviceptr,
}

/// f32 fused AdamW optimizer (matches `torch.optim.AdamW`).
pub struct GpuAdamW {
    /// First moment (m) in f32, layout matches the flat grad arena.
    pub m: GpuBuffer,
    /// Second moment (v) in f32, layout matches the flat grad arena.
    pub v: GpuBuffer,
    /// Step counter (1-indexed at first call to [`Self::step_one`]).
    pub step: u64,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// Reference-faithful no-decay parameter groups. When true, the
    /// per-tensor step functions ([`step_m1_capturable`] /
    /// [`step_m3_capturable`]) pass `weight_decay = 0` for the parameters
    /// the reference implementation marks `_no_weight_decay` — M1: `a_log`,
    /// `d_param`, `dt_proj_b` and the RMSNorm scales; M3: `dt_bias`,
    /// `d_param` and every norm scale. Decaying `a_log` pulls all decay
    /// rates toward A = -1 over long runs. Default `false` preserves the
    /// historical decay-everything behavior bit-for-bit; costs nothing
    /// either way (the decay coefficient is a per-launch scalar).
    pub reference_no_decay: bool,
}

impl GpuAdamW {
    /// Allocate zero-initialized `m`, `v` of length `n_params` (= total
    /// number of f32 master weights = `grads.flat.len()`).
    ///
    /// Defaults match `torch.optim.AdamW(params)`: lr=1e-3, β1=0.9, β2=0.999,
    /// eps=1e-8, weight_decay=1e-2.
    pub fn new(stream: &Arc<CudaStream>, n_params: usize) -> Result<Self, String> {
        Ok(Self {
            m: GpuBuffer::zeros(stream, n_params)?,
            v: GpuBuffer::zeros(stream, n_params)?,
            step: 0,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-2,
            reference_no_decay: false,
        })
    }

    /// Toggle the reference-faithful no-decay parameter groups (see the
    /// field docs on [`Self::reference_no_decay`]).
    #[must_use]
    pub fn with_reference_no_decay(mut self, on: bool) -> Self {
        self.reference_no_decay = on;
        self
    }

    #[must_use]
    pub fn with_lr(mut self, lr: f32) -> Self {
        assert!(lr.is_finite() && lr >= 0.0, "lr must be finite and >= 0");
        self.lr = lr;
        self
    }

    #[must_use]
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&beta1) && (0.0..1.0).contains(&beta2),
            "betas must be in [0, 1), got beta1={beta1} beta2={beta2}"
        );
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        assert!(eps.is_finite() && eps > 0.0, "eps must be finite and > 0");
        self.eps = eps;
        self
    }

    #[must_use]
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        assert!(
            wd.is_finite() && wd >= 0.0,
            "weight_decay must be finite and >= 0"
        );
        self.weight_decay = wd;
        self
    }

    /// Reset `m`, `v`, and step counter — useful between training phases or
    /// after a checkpoint load that doesn't include optimizer state.
    pub fn zero_state(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
        self.m.zero(stream)?;
        self.v.zero(stream)?;
        self.step = 0;
        Ok(())
    }

    /// Save (step, lr) — not the f32 buffers. Use `m`/`v` field access for
    /// full state-dict (download with `to_cpu()`).
    pub fn state(&self) -> (u64, f32) {
        (self.step, self.lr)
    }

    /// Run one fused AdamW update on a single tensor. Caller supplies the
    /// device pointers (weight + grad + matching `m`/`v` slices) bundled
    /// in [`AdamWParamPtrs`].
    ///
    /// You normally want [`step_m1`] / [`step_m3`]; this is the low-level
    /// building block they use.
    pub fn step_one(
        &self,
        ctx: &GpuCtx,
        adamw_kernel: &CudaFunction,
        ptrs: AdamWParamPtrs,
        len: usize,
        bias_c1: f32,
        bias_c2: f32,
    ) -> Result<(), String> {
        if len == 0 {
            return Ok(());
        }
        let n = len as i32;
        let cfg = grid_1d(len);
        let mut bld = ctx.stream.launch_builder(adamw_kernel);
        bld.arg(&ptrs.weight);
        bld.arg(&ptrs.grad);
        bld.arg(&ptrs.m);
        bld.arg(&ptrs.v);
        bld.arg(&self.lr);
        bld.arg(&self.beta1);
        bld.arg(&self.beta2);
        bld.arg(&self.eps);
        bld.arg(&self.weight_decay);
        bld.arg(&bias_c1);
        bld.arg(&bias_c2);
        bld.arg(&n);
        unsafe { bld.launch(cfg) }.map_err(|e| format!("adamw_step_f32: {e:?}"))?;
        Ok(())
    }

    /// CUDA-Graph-capturable variant of [`Self::step_one`]. Bias factors
    /// are read from a 2-element device buffer (see [`AdamWBiasFactors`]),
    /// which the CPU updates BEFORE each graph replay.
    pub fn step_one_capturable(
        &self,
        ctx: &GpuCtx,
        adamw_kernel: &CudaFunction,
        ptrs: AdamWParamPtrs,
        bias_factors_ptr: cudarc::driver::sys::CUdeviceptr,
        len: usize,
    ) -> Result<(), String> {
        self.step_one_capturable_wd(
            ctx,
            adamw_kernel,
            ptrs,
            bias_factors_ptr,
            len,
            self.weight_decay,
        )
    }

    /// [`Self::step_one_capturable`] with an explicit per-tensor weight
    /// decay — the mechanism behind the reference no-decay groups (the
    /// decay coefficient is a per-launch scalar, so per-tensor decay costs
    /// nothing).
    pub fn step_one_capturable_wd(
        &self,
        ctx: &GpuCtx,
        adamw_kernel: &CudaFunction,
        ptrs: AdamWParamPtrs,
        bias_factors_ptr: cudarc::driver::sys::CUdeviceptr,
        len: usize,
        weight_decay: f32,
    ) -> Result<(), String> {
        if len == 0 {
            return Ok(());
        }
        let n = len as i32;
        let cfg = grid_1d(len);
        let mut bld = ctx.stream.launch_builder(adamw_kernel);
        bld.arg(&ptrs.weight);
        bld.arg(&ptrs.grad);
        bld.arg(&ptrs.m);
        bld.arg(&ptrs.v);
        bld.arg(&self.lr);
        bld.arg(&self.beta1);
        bld.arg(&self.beta2);
        bld.arg(&self.eps);
        bld.arg(&weight_decay);
        bld.arg(&bias_factors_ptr);
        bld.arg(&n);
        unsafe { bld.launch(cfg) }.map_err(|e| format!("adamw_step_f32_capturable: {e:?}"))?;
        Ok(())
    }

    /// Pre-compute `(bias_c1, bias_c2)` for the *next* step (i.e. after
    /// incrementing `self.step`). Returns the new step number and the two
    /// bias-correction multipliers used inside the kernel.
    pub fn advance(&mut self) -> (u64, f32, f32) {
        self.step += 1;
        // `powi` takes i32 for the exponent. Clamp to a step count beyond
        // which `β^t` is already below f64 round-off (≈ 1e-300 at step
        // ~3000 for β=0.9, step ~700k for β=0.999). 2^30 is ≈ 1.07B, well
        // inside i32 range and well past any realistic training horizon.
        // This avoids the silent overflow that cast `u64 as i32` produced
        // at step ≥ 2^31 (negative exponent → garbage bias factors).
        let t = self.step.min(1 << 30) as i32;
        let denom1 = 1.0 - (self.beta1 as f64).powi(t);
        let denom2 = 1.0 - (self.beta2 as f64).powi(t);
        let bias_c1 = (1.0 / denom1.max(1e-30)) as f32;
        let bias_c2 = (1.0 / denom2.max(1e-30)) as f32;
        (self.step, bias_c1, bias_c2)
    }
}

/// Iterate (weight_buffer, grad_slice) pairs in arena order and launch one
/// `adamw_step_f32` per tensor. The flat `m`/`v` are sliced at the same
/// offset that `grad_slice` has into `grads.flat`.
///
/// Caller must ensure the `(weight, grad)` pairs are fed in the SAME order
/// as `GpuMambaGrads::new` wrote them, so the m/v offsets align.
pub fn run_pairs(
    ctx: &GpuCtx,
    adamw_kernel: &CudaFunction,
    adam: &GpuAdamW,
    bias_c1: f32,
    bias_c2: f32,
    flat_grad_base: cudarc::driver::sys::CUdeviceptr,
    pairs: &[(&GpuBuffer, &GradSlice)],
) -> Result<(), String> {
    let m_base = adam.m.cached_ptr();
    let v_base = adam.v.cached_ptr();
    for (w, g) in pairs {
        // Skip empty master tensors. `MambaWeights` clears `input_proj_w`
        // to zero-length for HF Mamba's identity input projection — the
        // grad arena still reserves a slot for layout symmetry, but
        // there's nothing to update. Skipping leaves m/v at zero, which
        // is the correct AdamW state for an absent param.
        if w.is_empty() {
            continue;
        }
        if g.len() != w.len() {
            return Err(format!(
                "adamw: weight/grad len mismatch: w={} g={}",
                w.len(),
                g.len()
            ));
        }
        let g_ptr = g.ptr();
        let off_bytes = g_ptr - flat_grad_base;
        // Element offset (f32 = 4 bytes).
        let off_elems = off_bytes / 4;
        let m_ptr = m_base + off_bytes;
        let v_ptr = v_base + off_bytes;
        debug_assert!(
            off_elems as usize + g.len() <= adam.m.len(),
            "adamw m/v slice OOB: off_elems={off_elems} len={} m.len={}",
            g.len(),
            adam.m.len()
        );
        adam.step_one(
            ctx,
            adamw_kernel,
            AdamWParamPtrs {
                weight: w.cached_ptr(),
                grad: g_ptr,
                m: m_ptr,
                v: v_ptr,
            },
            g.len(),
            bias_c1,
            bias_c2,
        )?;
    }
    Ok(())
}

/// CUDA-Graph-capturable variant of [`run_pairs`]. Reads bias factors
/// from `bias_factors_ptr` (a 2-element device buffer) instead of taking
/// scalars. Caller is responsible for writing fresh `(bc1, bc2)` into that
/// buffer BEFORE each graph replay.
pub fn run_pairs_capturable(
    ctx: &GpuCtx,
    adamw_kernel: &CudaFunction,
    adam: &GpuAdamW,
    bias_factors_ptr: cudarc::driver::sys::CUdeviceptr,
    flat_grad_base: cudarc::driver::sys::CUdeviceptr,
    pairs: &[(&GpuBuffer, &GradSlice)],
) -> Result<(), String> {
    for pair in pairs {
        run_one_capturable_wd(
            ctx,
            adamw_kernel,
            adam,
            bias_factors_ptr,
            flat_grad_base,
            *pair,
            adam.weight_decay,
        )?;
    }
    Ok(())
}

/// One capturable AdamW launch for a (weight, grad-slice) pair with an
/// explicit weight decay. Shared body of [`run_pairs_capturable`] and the
/// per-tensor no-decay paths in [`step_m1_capturable`] /
/// [`step_m3_capturable`].
fn run_one_capturable_wd(
    ctx: &GpuCtx,
    adamw_kernel: &CudaFunction,
    adam: &GpuAdamW,
    bias_factors_ptr: cudarc::driver::sys::CUdeviceptr,
    flat_grad_base: cudarc::driver::sys::CUdeviceptr,
    (w, g): (&GpuBuffer, &GradSlice),
    weight_decay: f32,
) -> Result<(), String> {
    // Skip empty master tensors (HF Mamba identity input_proj). See
    // `run_pairs` for the rationale.
    if w.is_empty() {
        return Ok(());
    }
    if g.len() != w.len() {
        return Err(format!(
            "adamw: weight/grad len mismatch: w={} g={}",
            w.len(),
            g.len()
        ));
    }
    let g_ptr = g.ptr();
    let off_bytes = g_ptr - flat_grad_base;
    let off_elems = off_bytes / 4;
    let m_ptr = adam.m.cached_ptr() + off_bytes;
    let v_ptr = adam.v.cached_ptr() + off_bytes;
    debug_assert!(
        off_elems as usize + g.len() <= adam.m.len(),
        "adamw m/v slice OOB"
    );
    adam.step_one_capturable_wd(
        ctx,
        adamw_kernel,
        AdamWParamPtrs {
            weight: w.cached_ptr(),
            grad: g_ptr,
            m: m_ptr,
            v: v_ptr,
        },
        bias_factors_ptr,
        g.len(),
        weight_decay,
    )
}

/// Mamba SSM backbone AdamW step. Iterates the per-tensor master weights in
/// the SAME order as `GpuMambaGrads::new` wrote them into the flat arena,
/// so the m/v offsets line up element-for-element with grads.flat.
pub fn step_m1(
    ctx: &GpuCtx,
    adamw_kernel: &CudaFunction,
    adam: &mut GpuAdamW,
    weights: &mut crate::mamba_ssm::gpu::weights::GpuMambaTrainWeights,
    grads: &crate::mamba_ssm::gpu::weights::GpuMambaGrads,
) -> Result<(), String> {
    let (_, bc1, bc2) = adam.advance();
    let flat_base = grads.flat.cached_ptr();

    // Build paired iterator in the EXACT layout of `GpuMambaGrads::new`:
    // input_proj_w, input_proj_b, [layers...], norm_f_weight.
    let mut pairs: Vec<(&GpuBuffer, &GradSlice)> = Vec::new();
    pairs.push((&weights.input_proj_w, &grads.input_proj_w));
    pairs.push((&weights.input_proj_b, &grads.input_proj_b));
    for (lw, lg) in weights.layers.iter().zip(&grads.layers) {
        pairs.push((&lw.norm_weight, &lg.norm_weight));
        pairs.push((&lw.in_proj_w, &lg.in_proj_w));
        pairs.push((&lw.conv1d_weight, &lg.conv1d_weight));
        pairs.push((&lw.conv1d_bias, &lg.conv1d_bias));
        pairs.push((&lw.x_proj_w, &lg.x_proj_w));
        pairs.push((&lw.dt_proj_w, &lg.dt_proj_w));
        pairs.push((&lw.dt_proj_b, &lg.dt_proj_b));
        pairs.push((&lw.a_log, &lg.a_log));
        pairs.push((&lw.d_param, &lg.d_param));
        pairs.push((&lw.out_proj_w, &lg.out_proj_w));
    }
    pairs.push((&weights.norm_f_weight, &grads.norm_f_weight));

    run_pairs(ctx, adamw_kernel, adam, bc1, bc2, flat_base, &pairs)
}

/// CUDA-Graph-capturable variant of [`step_m1`]. Bias factors come from
/// the 2-element device buffer `bias_factors_ptr`, which the CPU rewrites
/// before each graph replay.
pub fn step_m1_capturable(
    ctx: &GpuCtx,
    adamw_kernel: &CudaFunction,
    adam: &GpuAdamW,
    bias_factors_ptr: cudarc::driver::sys::CUdeviceptr,
    weights: &mut crate::mamba_ssm::gpu::weights::GpuMambaTrainWeights,
    grads: &crate::mamba_ssm::gpu::weights::GpuMambaGrads,
) -> Result<(), String> {
    let flat_base = grads.flat.cached_ptr();
    // (weight, grad, reference-no-decay?). The no-decay group is the set the
    // reference implementation marks `_no_weight_decay`: a_log, D, dt bias,
    // plus every RMSNorm scale — active only when `adam.reference_no_decay`.
    let mut pairs: Vec<(&GpuBuffer, &GradSlice, bool)> = Vec::new();
    pairs.push((&weights.input_proj_w, &grads.input_proj_w, false));
    pairs.push((&weights.input_proj_b, &grads.input_proj_b, false));
    for (lw, lg) in weights.layers.iter().zip(&grads.layers) {
        pairs.push((&lw.norm_weight, &lg.norm_weight, true));
        pairs.push((&lw.in_proj_w, &lg.in_proj_w, false));
        pairs.push((&lw.conv1d_weight, &lg.conv1d_weight, false));
        pairs.push((&lw.conv1d_bias, &lg.conv1d_bias, false));
        pairs.push((&lw.x_proj_w, &lg.x_proj_w, false));
        pairs.push((&lw.dt_proj_w, &lg.dt_proj_w, false));
        pairs.push((&lw.dt_proj_b, &lg.dt_proj_b, true));
        pairs.push((&lw.a_log, &lg.a_log, true));
        pairs.push((&lw.d_param, &lg.d_param, true));
        pairs.push((&lw.out_proj_w, &lg.out_proj_w, false));
    }
    pairs.push((&weights.norm_f_weight, &grads.norm_f_weight, true));
    for (w, g, no_decay) in pairs {
        let wd = if no_decay && adam.reference_no_decay {
            0.0
        } else {
            adam.weight_decay
        };
        run_one_capturable_wd(
            ctx,
            adamw_kernel,
            adam,
            bias_factors_ptr,
            flat_base,
            (w, g),
            wd,
        )?;
    }
    Ok(())
}

/// CUDA-Graph-capturable variant of [`step_m3`].
pub fn step_m3_capturable(
    ctx: &GpuCtx,
    adamw_kernel: &CudaFunction,
    adam: &GpuAdamW,
    bias_factors_ptr: cudarc::driver::sys::CUdeviceptr,
    weights: &mut crate::mamba3_siso::gpu::weights::GpuMamba3Weights,
    grads: &crate::mamba3_siso::gpu::weights::GpuMamba3Grads,
) -> Result<(), String> {
    let flat_base = grads.flat.cached_ptr();
    // M3 no-decay group (mirror of the M1 set's principle): dt bias, D and
    // every norm scale. M3 has no fixed a_log (A is input-dependent); the
    // B/C biases stay decayed (only the reference-named analogues are
    // exempted). Active only when `adam.reference_no_decay`.
    let mut pairs: Vec<(&GpuBuffer, &GradSlice, bool)> = Vec::new();
    pairs.push((&weights.input_proj_w, &grads.input_proj_w, false));
    pairs.push((&weights.input_proj_b, &grads.input_proj_b, false));
    for (lw, lg) in weights.layers.iter().zip(&grads.layers) {
        pairs.push((&lw.norm_weight, &lg.norm_weight, true));
        pairs.push((&lw.in_proj_w, &lg.in_proj_w, false));
        pairs.push((&lw.dt_bias, &lg.dt_bias, true));
        pairs.push((&lw.b_norm_weight, &lg.b_norm_weight, true));
        pairs.push((&lw.c_norm_weight, &lg.c_norm_weight, true));
        pairs.push((&lw.b_bias, &lg.b_bias, false));
        pairs.push((&lw.c_bias, &lg.c_bias, false));
        pairs.push((&lw.d_param, &lg.d_param, true));
        pairs.push((&lw.norm_gate_weight, &lg.norm_gate_weight, true));
        pairs.push((&lw.out_proj_w, &lg.out_proj_w, false));
    }
    pairs.push((&weights.norm_f_weight, &grads.norm_f_weight, true));
    for (w, g, no_decay) in pairs {
        let wd = if no_decay && adam.reference_no_decay {
            0.0
        } else {
            adam.weight_decay
        };
        run_one_capturable_wd(
            ctx,
            adamw_kernel,
            adam,
            bias_factors_ptr,
            flat_base,
            (w, g),
            wd,
        )?;
    }
    Ok(())
}

/// Mamba-3 backbone AdamW step. Same idea as [`step_m1`] but for the M3
/// weight set (`GpuMamba3Weights` / `GpuMamba3Grads`).
pub fn step_m3(
    ctx: &GpuCtx,
    adamw_kernel: &CudaFunction,
    adam: &mut GpuAdamW,
    weights: &mut crate::mamba3_siso::gpu::weights::GpuMamba3Weights,
    grads: &crate::mamba3_siso::gpu::weights::GpuMamba3Grads,
) -> Result<(), String> {
    let (_, bc1, bc2) = adam.advance();
    let flat_base = grads.flat.cached_ptr();

    let mut pairs: Vec<(&GpuBuffer, &GradSlice)> = Vec::new();
    pairs.push((&weights.input_proj_w, &grads.input_proj_w));
    pairs.push((&weights.input_proj_b, &grads.input_proj_b));
    for (lw, lg) in weights.layers.iter().zip(&grads.layers) {
        pairs.push((&lw.norm_weight, &lg.norm_weight));
        pairs.push((&lw.in_proj_w, &lg.in_proj_w));
        pairs.push((&lw.dt_bias, &lg.dt_bias));
        pairs.push((&lw.b_norm_weight, &lg.b_norm_weight));
        pairs.push((&lw.c_norm_weight, &lg.c_norm_weight));
        pairs.push((&lw.b_bias, &lg.b_bias));
        pairs.push((&lw.c_bias, &lg.c_bias));
        pairs.push((&lw.d_param, &lg.d_param));
        pairs.push((&lw.norm_gate_weight, &lg.norm_gate_weight));
        pairs.push((&lw.out_proj_w, &lg.out_proj_w));
    }
    pairs.push((&weights.norm_f_weight, &grads.norm_f_weight));

    run_pairs(ctx, adamw_kernel, adam, bc1, bc2, flat_base, &pairs)
}

#[cfg(test)]
mod cpu_state_tests {
    #[test]
    fn bias_correction_step_one() {
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let bc1 = (1.0 / (1.0 - beta1.powi(1))) as f32;
        let bc2 = (1.0 / (1.0 - beta2.powi(1))) as f32;
        // At t=1: bc1 = 1/(1-0.9) = 10, bc2 = 1/(1-0.999) = 1000.
        assert!((bc1 - 10.0).abs() < 1e-3, "bc1={bc1}");
        assert!((bc2 - 1000.0).abs() < 1e-1, "bc2={bc2}");
    }

    #[test]
    fn bias_correction_step_large() {
        // At t=2000 (post-warmup), both bias factors → ~1.
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let bc1 = (1.0 / (1.0 - beta1.powi(2000))) as f32;
        let bc2 = (1.0 / (1.0 - beta2.powi(2000))) as f32;
        assert!(bc1 < 1.001, "bc1={bc1}");
        assert!(bc2 < 1.2, "bc2={bc2}");
    }
}
