//! The optimizer tail the Mamba-2 and MIMO trainers share: the gradient sum
//! of an accumulation window and the deterministic global-norm clip.
//!
//! Their backward writes the gradient arena whole, so a window of
//! micro-batches sums into a buffer of its own, in call order, and the
//! applying call adds that sum to its own gradient before the clip and the
//! optimizer step. In f16 the window also remembers whether any of its
//! micro-batches overflowed the loss scale, which holds for the whole
//! window, so the applying call skips the whole window then. A micro-batch
//! added on the device marks its overflow in the sum's values alone, so a
//! window that took one has the applying call read the values.

use std::sync::Arc;

use cudarc::driver::CudaStream;

use super::buffers::{GpuBuffer, GpuByteBuffer};
use super::context::GpuCtx;
use super::grad_clip::{GradRegionGeom, alloc_partials, clip_grads_device, clip_region_device};
use super::launch::launch_elems;
use super::loss_scaler::mark_overflow;
use crate::mamba1::gpu::trainer::BackwardOpts;

/// The gradient sum of an open accumulation window, and the scratch of
/// the global-norm clip.
pub(crate) struct GradWindow {
    /// The sum of the window's gradients so far, allocated by the first
    /// accumulating call.
    sum: Option<GpuBuffer>,
    /// Whether `sum` holds a window the optimizer has not stepped on yet.
    open: bool,
    /// Whether a micro-batch of the open window overflowed the f16 loss
    /// scale, as far as the host knows.
    overflowed: bool,
    /// Whether a micro-batch joined the open window on the device under the
    /// f16 loss scale: its overflow shows only in the sum's values.
    scaled_on_device: bool,
    partials: GpuByteBuffer,
    coef: GpuBuffer,
}

impl GradWindow {
    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Self, String> {
        Ok(Self {
            sum: None,
            open: false,
            overflowed: false,
            scaled_on_device: false,
            partials: alloc_partials(stream)?,
            coef: GpuBuffer::zeros(stream, 2)?,
        })
    }

    /// Add the gradient of an accumulating call to the window, and whether
    /// it overflowed the f16 loss scale; an overflowed call also marks the
    /// sum with a NaN (see [`poison`]).
    pub(crate) fn accumulate(
        &mut self,
        ctx: &GpuCtx,
        grads: &GpuBuffer,
        overflowed: bool,
    ) -> Result<(), String> {
        self.open_zeroed(ctx, grads.len())?;
        self.overflowed |= overflowed;
        let Some(sum) = self.sum.as_mut() else {
            return Err("an accumulation window without its sum".into());
        };
        add_into(ctx, sum, grads)?;
        if overflowed {
            poison(ctx, sum)?;
        }
        Ok(())
    }

    /// The window's sum, allocated for `len` values when it has none yet,
    /// so a captured body can name a buffer that stays where it is.
    pub(crate) fn ensure_sum(&mut self, ctx: &GpuCtx, len: usize) -> Result<(), String> {
        if self.sum.as_ref().is_none_or(|s| s.len() != len) {
            self.sum = Some(GpuBuffer::zeros(&ctx.stream, len)?);
        }
        Ok(())
    }

    /// Open a window whose sum starts at zero; an open window stays as it
    /// is.
    fn open_zeroed(&mut self, ctx: &GpuCtx, len: usize) -> Result<(), String> {
        if self.open {
            return Ok(());
        }
        self.ensure_sum(ctx, len)?;
        if let Some(sum) = self.sum.as_mut() {
            sum.zero(&ctx.stream)?;
        }
        self.open = true;
        self.overflowed = false;
        self.scaled_on_device = false;
        Ok(())
    }

    /// Open the window for a micro-batch the device adds to its sum (see
    /// [`Self::add_device`]); an open window stays as it is. `scaled` says
    /// the micro-batch runs under the f16 loss scale, whose overflow marks
    /// only the sum's values, so the fold has the applying call read them.
    pub(crate) fn open_for_device(
        &mut self,
        ctx: &GpuCtx,
        len: usize,
        scaled: bool,
    ) -> Result<(), String> {
        self.open_zeroed(ctx, len)?;
        self.scaled_on_device |= scaled;
        Ok(())
    }

    /// Add `grads` to the open window's sum and, when `overflow` (the loss
    /// scaler's device flag, or 0 for none) is set, mark the sum with a NaN
    /// (see [`poison`]): launches only, for a captured body.
    pub(crate) fn add_device(
        &mut self,
        ctx: &GpuCtx,
        grads: &GpuBuffer,
        overflow: cudarc::driver::sys::CUdeviceptr,
    ) -> Result<(), String> {
        let Some(sum) = self.sum.as_mut() else {
            return Err("an accumulation window without its sum".into());
        };
        add_into(ctx, sum, grads)?;
        if overflow == 0 || sum.is_empty() {
            return Ok(());
        }
        mark_overflow(ctx, sum.cached_ptr(), overflow)
    }

    /// Copy the open window's sum into `grads` through the cached pointers
    /// (CUDA Graph safe), leaving the window open: the first part of an
    /// applying body.
    pub(crate) fn copy_sum_into(&self, ctx: &GpuCtx, grads: &mut GpuBuffer) -> Result<(), String> {
        let Some(sum) = self.sum.as_ref() else {
            return Err("an accumulation window without its sum".into());
        };
        if sum.len() != grads.len() {
            return Err(format!(
                "gradient window: {} values against an arena of {}",
                sum.len(),
                grads.len()
            ));
        }
        grads.copy_from_raw(sum, &ctx.stream)
    }

    /// Close the window after an applying body ran on its sum.
    pub(crate) fn close(&mut self) {
        self.open = false;
        self.overflowed = false;
        self.scaled_on_device = false;
    }

    /// Add the open window's sum to the gradient of the applying call,
    /// close the window, and return what is known of an f16 overflow in
    /// its micro-batches. Nothing to add when no window is open.
    pub(crate) fn fold_into(
        &mut self,
        ctx: &GpuCtx,
        grads: &mut GpuBuffer,
    ) -> Result<FoldOverflow, String> {
        if !self.open {
            return Ok(FoldOverflow::Clear);
        }
        self.open = false;
        let overflowed = std::mem::take(&mut self.overflowed);
        let in_values = std::mem::take(&mut self.scaled_on_device);
        let Some(sum) = &self.sum else {
            return Err("an accumulation window without its sum".into());
        };
        add_into(ctx, grads, sum)?;
        Ok(if overflowed {
            FoldOverflow::Overflowed
        } else if in_values {
            FoldOverflow::InValues
        } else {
            FoldOverflow::Clear
        })
    }

    /// Whether a window is open: an accumulating call came after the last
    /// applying one.
    pub(crate) fn is_open(&self) -> bool {
        self.open
    }

    /// The sum of the open window, which a data-parallel caller reduces
    /// across ranks before the applying call.
    pub(crate) fn sum(&self) -> Result<&GpuBuffer, String> {
        match &self.sum {
            Some(sum) if self.open => Ok(sum),
            _ => Err("no open accumulation window: accumulate a micro-batch first".into()),
        }
    }

    /// The sum of the open window, to overwrite or rescale in place.
    pub(crate) fn sum_mut(&mut self) -> Result<&mut GpuBuffer, String> {
        match &mut self.sum {
            Some(sum) if self.open => Ok(sum),
            _ => Err("no open accumulation window: accumulate a micro-batch first".into()),
        }
    }

    /// Copy the open window's sum into `grads`, close the window, and
    /// return whether the host knows of an f16 overflow in its
    /// micro-batches; one the device added shows only in the values, which
    /// the caller reads. An error when no window is open.
    pub(crate) fn take_into(
        &mut self,
        ctx: &GpuCtx,
        grads: &mut GpuBuffer,
    ) -> Result<bool, String> {
        let sum = self.sum()?;
        if sum.len() != grads.len() {
            return Err(format!(
                "gradient window: {} values against an arena of {}",
                sum.len(),
                grads.len()
            ));
        }
        grads.copy_from(sum, &ctx.stream)?;
        self.open = false;
        self.scaled_on_device = false;
        Ok(std::mem::take(&mut self.overflowed))
    }

    /// The clip's scratch, the partial sums and `[coef, norm]`, for a step
    /// tail that launches the clip itself.
    pub(crate) fn clip_scratch(&mut self) -> (&mut GpuByteBuffer, &mut GpuBuffer) {
        (&mut self.partials, &mut self.coef)
    }

    /// `[coef, norm]` of the last clip launched on this window's scratch.
    pub(crate) fn clip_coef(&self) -> &GpuBuffer {
        &self.coef
    }

    /// Scale the region `geom` of `grads` to its own L2 norm `max_norm`
    /// when it is larger, with the fixed-order f64 reduction the global
    /// clip uses, and return the region's norm before the clip.
    pub(crate) fn clip_region(
        &mut self,
        ctx: &GpuCtx,
        grads: &mut GpuBuffer,
        geom: GradRegionGeom,
        max_norm: f32,
    ) -> Result<f32, String> {
        clip_region_device(
            ctx,
            grads,
            &mut self.partials,
            &mut self.coef,
            geom,
            max_norm,
        )
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

/// What the fold of a window into the applying call's gradient knows of an
/// f16 overflow in the window's micro-batches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FoldOverflow {
    /// None of them overflowed.
    Clear,
    /// One of them overflowed.
    Overflowed,
    /// One of them joined on the device under the loss scale: only the
    /// folded gradient's values tell whether it overflowed.
    InValues,
}

/// Write a NaN into the first element of a window's sum. The unscale of an
/// overflowed micro-batch zeroes its gradients, so the sum alone would not
/// show the overflow; the NaN survives any sum it enters, so every rank of
/// a data-parallel world that reduces this window with its own sees the
/// overflow in the reduced arena and skips the same window.
fn poison(ctx: &GpuCtx, sum: &mut GpuBuffer) -> Result<(), String> {
    if sum.is_empty() {
        return Ok(());
    }
    let r = unsafe {
        cudarc::driver::sys::cuMemsetD32Async(
            sum.cached_ptr(),
            f32::NAN.to_bits(),
            1,
            ctx.stream.cu_stream(),
        )
    };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("gradient window overflow mark: {r:?}"));
    }
    Ok(())
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
    launch_elems(
        &ctx.stream,
        &ctx.kernels.vec_add_inplace,
        &[dst.cached_ptr(), src.cached_ptr()],
        n,
        format_args!("gradient window add"),
    )
}

/// The backward options every trainer refuses: a clip of either kind on
/// an accumulating call, since the norm is the whole window's.
pub(crate) fn check_backward_opts(opts: &BackwardOpts, family: &str) -> Result<(), String> {
    if opts.accumulate_only && opts.clip_max_norm.is_some() {
        return Err(format!(
            "{family}: clip on the applying call; the norm is the whole window's"
        ));
    }
    if opts.accumulate_only && opts.control_clip_max_norm.is_some() {
        return Err(format!(
            "{family}: control-channel clip on the applying call; the norm is the \
             whole window's"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

    /// The add covers every element once: the arena plus the window is the
    /// host sum.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn the_window_add_covers_every_element_once() {
        let device = GpuDevice::new(0).expect("open CUDA device");
        let ctx = GpuCtx::new(&device).expect("context");
        let n = 2500;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - i as f32).collect();
        let mut dst = GpuBuffer::zeros(&ctx.stream, n).expect("dst");
        let mut src = GpuBuffer::zeros(&ctx.stream, n).expect("src");
        dst.upload(&ctx.stream, &a).expect("upload a");
        src.upload(&ctx.stream, &b).expect("upload b");
        add_into(&ctx, &mut dst, &src).expect("window add");
        let mut got = vec![0.0f32; n];
        dst.download(&ctx.stream, &mut got).expect("download");
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(got, want);
    }
}
