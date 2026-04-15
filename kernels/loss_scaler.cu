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
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int local = 0;
    for (int i = idx; i < n; i += stride) {
        float v = grads[i];
        // Any non-finite triggers overflow. isfinite() returns false for both
        // ±inf and NaN, which is exactly the AMP semantics.
        if (!isfinite(v)) {
            local = 1;
            break;  // one overflow is enough — no need to keep scanning
        }
    }
    if (local) {
        atomicOr(found_overflow, 1);
    }
}

// Multiply every gradient in [grads, grads+n) by `scale`. Used both for
// unscaling (scale = 1 / loss_scale) and grad clipping (scale = clip_norm /
// grad_norm). NaN/inf in the input propagates — call check_inf_nan_f32 first
// if you need the AMP-safe path.
extern "C" __global__ void scale_grads_f32(
    float* __restrict__ grads,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        grads[i] *= scale;
    }
}
