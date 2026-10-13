// Loss-scaling helpers for f16/bf16 mixed-precision training (PyTorch
// GradScaler equivalent — see torch.cuda.amp.GradScaler / NVIDIA Apex AMP).
//
// Workflow:
//   1. CPU side: scaled_loss = loss * scale
//   2. backward(scaled_loss) → produces master grads scaled by `scale`
//   3. GPU: `check_inf_nan_f32` scans every grad buffer, atomicOr into a
//      single device int → 1 if any element is inf/nan
//   4. CPU reads the flag:
//        - overflow → skip optimizer.step(); scaler backs off (scale /= 2)
//        - clean   → `scale_grads_f32(grads, 1/scale)` to unscale, then step;
//                    after `growth_interval` clean steps, scale *= 2
//
// Why f32 only: master gradients are kept in f32 throughout the AMP path
// (bf16/f16 atomicAdd is not supported on ≤sm_89 and reduces precision).
// `scale_grads_f32` accepts a generic multiplier so it can also be used for
// pre-optimizer rescaling (e.g. grad clipping).

extern "C" __global__ void check_inf_nan_f32(
    int* __restrict__ found_overflow,  // [1] device int, atomicOr target
    const float* __restrict__ grads,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    int local = 0;
    for (long long i = idx; i < n; i += stride) {
        float v = grads[i];
        // Any non-finite triggers overflow. isfinite() returns false for both
        // ±inf and NaN, which is exactly the AMP semantics.
        if (!isfinite(v)) {
            local = 1;
            break;  // one overflow is enough — no need to keep scanning
        }
    }
    // Warp-collapse before HBM atomicOr: worst case
    // every lane has local=1 → 32 atomicOrs per warp → contention.
    // __any_sync collapses to a single ballot and one atomicOr per warp.
    if (__any_sync(0xFFFFFFFFu, local)) {
        if ((threadIdx.x & 31) == 0) {
            atomicOr(found_overflow, 1);
        }
    }
}

// Multiply every gradient in [grads, grads+n) by `scale`. Used both for
// unscaling (scale = 1 / loss_scale) and grad clipping (scale = clip_norm /
// grad_norm). NaN/inf in the input propagates — call check_inf_nan_f32 first
// if you need the AMP-safe path.
extern "C" __global__ void scale_grads_f32(
    float* __restrict__ grads,
    float scale,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) {
        grads[i] *= scale;
    }
}

// Device-scalar twin of `scale_grads_f32`: the factor is read from device
// memory (the clip coefficient the norm kernel just wrote) instead of
// being a launch argument, so the scaling pass can be enqueued without a
// host round trip. Same grid, same element-to-thread mapping, same
// multiply - bit-identical to passing the value by argument.
extern "C" __global__ void scale_grads_dev_f32(
    float* __restrict__ grads,
    const float* __restrict__ scale,
    long long n
) {
    float s = scale[0];
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) {
        grads[i] *= s;
    }
}

// CUDA-Graph-capturable variant of `scale_grads_f32` that conditionally
// zeros the gradient based on an overflow flag from `check_inf_nan_f32`.
//
//   grads[i] = (overflow_flag[0] != 0) ? 0.0 : grads[i] * unscale_factor
//
// This lets f16 AMP training capture the whole step into a CUDA graph: the
// optimizer kernel after it skips on the same flag, so an overflowed step
// leaves the master weights and both moments where they were, as
// `torch.cuda.amp.GradScaler` skips its step, and the eager step runs the
// same sequence with the same bits. The CPU reads the flag after the step
// to drive the scaler (backoff or growth).
//
// `unscale_factor` is read from a 1-element device buffer so the value
// can be updated between graph replays without re-capture. CPU writes
// `1/loss_scale` to `unscale_factor[0]` before each `cuGraphLaunch`.
extern "C" __global__ void scale_grads_skip_f32(
    float* __restrict__ grads,
    const int* __restrict__ overflow_flag,  // [1]
    const float* __restrict__ unscale_factor, // [1] = 1 / loss_scale
    long long n
) {
    // Branch on overflow and ASSIGN, not multiply. A multiply-by-zero would
    // propagate NaN: `Inf * 0 = NaN`, `NaN * 0 = NaN`. On overflow the raw
    // grads already contain ±Inf or NaN (that is *why* we overflowed), so
    // `grads[i] *= 0.0f` silently poisons master weights on the next AdamW
    // step. Assignment sanitises the buffer to a clean zero regardless of
    // the current contents, matching PyTorch GradScaler's behaviour of
    // skipping the optimizer step entirely on overflow.
    const int overflow = (overflow_flag[0] != 0);
    const float unscale = unscale_factor[0];
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) {
        grads[i] = overflow ? 0.0f : (grads[i] * unscale);
    }
}
