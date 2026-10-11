//! The optimizer tail the Mamba-2 and MIMO trainers share: the gradient sum
//! of an accumulation window and the deterministic global-norm clip.
//!
//! Their backward writes the gradient arena whole, so a window of
//! micro-batches sums into a buffer of its own, in call order, and the
//! applying call adds that sum to its own gradient before the clip and the
//! optimizer step.

use std::sync::Arc;

use cudarc::driver::{CudaStream, PushKernelArg};

use super::buffers::{GpuBuffer, GpuByteBuffer};
use super::context::GpuCtx;
use super::dtype::WeightDtype;
use super::grad_clip::{alloc_partials, clip_grads_device};
use super::launch::grid_1d;
use crate::mamba1::gpu::trainer::BackwardOpts;

/// The gradient sum of an open accumulation window, and the scratch of
/// the global-norm clip.
pub(crate) struct GradWindow {
    /// The sum of the window's gradients so far, allocated by the first
    /// accumulating call.
    sum: Option<GpuBuffer>,
    /// Whether `sum` holds a window the optimizer has not stepped on yet.
    open: bool,
    partials: GpuByteBuffer,
    coef: GpuBuffer,
}

impl GradWindow {
    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Self, String> {
        Ok(Self {
            sum: None,
            open: false,
            partials: alloc_partials(stream)?,
            coef: GpuBuffer::zeros(stream, 2)?,
        })
    }

    /// Add the gradient of an accumulating call to the window.
    pub(crate) fn accumulate(&mut self, ctx: &GpuCtx, grads: &GpuBuffer) -> Result<(), String> {
        let sum = match &mut self.sum {
            Some(sum) => sum,
            None => self.sum.insert(GpuBuffer::zeros(&ctx.stream, grads.len())?),
        };
        if self.open {
            add_into(ctx, sum, grads)
        } else {
            sum.copy_from(grads, &ctx.stream)?;
            self.open = true;
            Ok(())
        }
    }

    /// Add the open window's sum to the gradient of the applying call and
    /// close the window. Nothing to add when no window is open.
    pub(crate) fn fold_into(&mut self, ctx: &GpuCtx, grads: &mut GpuBuffer) -> Result<(), String> {
        if !self.open {
            return Ok(());
        }
        self.open = false;
        let Some(sum) = &self.sum else {
            return Err("an accumulation window without its sum".into());
        };
        add_into(ctx, grads, sum)
    }

    /// Scale `grads` to the global L2 norm `max_norm` when it is larger,
    /// with the fixed-order f64 reduction, and return the norm before the
    /// clip. A non-finite norm is an error.
    pub(crate) fn clip(
        &mut self,
        ctx: &GpuCtx,
        grads: &mut GpuBuffer,
        max_norm: f32,
    ) -> Result<f32, String> {
        let norm = clip_grads_device(ctx, grads, &mut self.partials, &mut self.coef, max_norm)?;
        if !norm.is_finite() {
            return Err(format!(
                "clip_max_norm: non-finite global grad norm ({norm})"
            ));
        }
        Ok(norm)
    }
}

/// `dst += src`, element by element.
fn add_into(ctx: &GpuCtx, dst: &mut GpuBuffer, src: &GpuBuffer) -> Result<(), String> {
    let n = dst.len();
    if src.len() != n {
        return Err(format!(
            "gradient window: {} values against an arena of {n}",
            src.len()
        ));
    }
    let n_i = n as i32;
    let mut bld = ctx.stream.launch_builder(&ctx.kernels.vec_add_inplace);
    bld.arg(dst.inner_mut());
    bld.arg(src.inner());
    bld.arg(&n_i);
    unsafe { bld.launch(grid_1d(n)) }.map_err(|e| format!("gradient window add: {e:?}"))?;
    Ok(())
}

/// The backward options these trainers refuse: an accumulating call in
/// f16 (the loss scale has no defined state across a window), a clip on an
/// accumulating call (the norm is the whole window's), and a control-channel
/// clip on a family whose control columns it does not clip yet.
pub(crate) fn check_backward_opts(
    opts: &BackwardOpts,
    dtype: WeightDtype,
    family: &str,
    clips_control: bool,
) -> Result<(), String> {
    if opts.accumulate_only && dtype == WeightDtype::F16 {
        return Err(format!(
            "{family}: accumulate_only is not defined for f16, where the loss scale \
             owns each step"
        ));
    }
    if opts.accumulate_only && opts.clip_max_norm.is_some() {
        return Err(format!(
            "{family}: clip on the applying call; the norm is the whole window's"
        ));
    }
    if opts.control_clip_max_norm.is_some() && !clips_control {
        return Err(format!(
            "{family}: the control-channel clip is not implemented for this trainer"
        ));
    }
    Ok(())
}
