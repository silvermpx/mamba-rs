// Fused AdamW optimizer step in f32 master precision (PyTorch-AMP convention).
//
// Matches `torch.optim.AdamW` with `capturable=True` numerics and the
// decoupled weight-decay form (Loshchilov & Hutter, "Decoupled Weight Decay
// Regularization", ICLR 2019). Identical update rule to `torch.optim._functional.adamw`
// reference path (PyTorch 2.5, `torch/optim/_functional.py::adamw`):
//
//   m_t = β1·m_{t-1} + (1 - β1)·g_t
//   v_t = β2·v_{t-1} + (1 - β2)·g_t²
//   m̂   = m_t / (1 - β1^t)
//   v̂   = v_t / (1 - β2^t)
//   p_t = p_{t-1} · (1 - lr·wd) - lr · m̂ / (√v̂ + ε)
//
// Bias-correction factors `bias_c1 = 1/(1-β1^t)` and `bias_c2 = 1/(1-β2^t)`
// are computed on the CPU (once per optimizer.step()) and passed in as
// scalars. This avoids per-tensor powf evaluation on the device and is
// what PyTorch's `_single_tensor_adamw` does when `capturable=False`.
//
// Why f32-only: master weights live in f32 (AMP convention — see Micikevicius
// et al., "Mixed Precision Training", ICLR 2018). The optimizer state (m, v)
// must be in f32 to avoid precision collapse after many accumulations; bf16
// Adam accumulators empirically diverge within ~1k steps on SSM-class models.

extern "C" __global__ void adamw_step_f32(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_c1,   // 1 / (1 - beta1^t)
    float bias_c2,   // 1 / (1 - beta2^t)
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const float one_minus_b1 = 1.f - beta1;
    const float one_minus_b2 = 1.f - beta2;
    const float decay_factor = 1.f - lr * weight_decay;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        float p = param[i];
        float mi = m[i] * beta1 + one_minus_b1 * g;
        float vi = v[i] * beta2 + one_minus_b2 * g * g;
        m[i] = mi;
        v[i] = vi;
        float m_hat = mi * bias_c1;
        float v_hat = vi * bias_c2;
        // Decoupled weight decay: p *= (1 - lr·wd) first, THEN subtract the
        // adam term. This order is what torch.optim.AdamW does — critical
        // so that the effective LR on the regularizer is independent of the
        // adaptive denominator.
        param[i] = decay_factor * p - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}
