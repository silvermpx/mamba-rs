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

// CUDA-Graph-capturable variant: bias-correction factors are read from a
// 2-element device buffer instead of taken as scalar kernel args. The CPU
// writes [bc1, bc2] into that buffer (async H2D) BEFORE each graph replay,
// so the captured kernel sees the updated values via a stable device
// pointer. Mirrors PyTorch `torch.optim.AdamW(capturable=True)` semantics
// (PyTorch 2.5, `_multi_tensor_adamw` capturable branch).
extern "C" __global__ void adamw_step_f32_capturable(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    const float* __restrict__ bias_factors,  // [2] = {bc1, bc2}
    int n
) {
    const float bias_c1 = bias_factors[0];
    const float bias_c2 = bias_factors[1];
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
        param[i] = decay_factor * p - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

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


// ---------------------------------------------------------------------------
// Descriptor-table fused AdamW: ONE launch updates every master tensor.
//
// The per-tensor walk paid 243 kernel launches per optimizer step (plus a
// separate 4-cast-per-layer master->compute sync pass on the mixed lane).
// The chunk table is built ONCE at trainer construction (the flat-arena
// layout is static); each block processes one chunk of one tensor.
//
// Determinism: the update expression per element is IDENTICAL to
// adamw_step_f32_capturable (same operand order, same intrinsics); chunk
// scheduling only changes WHICH block computes an element, never the
// arithmetic — elementwise, no cross-element reduction. The optional
// fused typed write-out stores FROM_F(new_p) of the SAME f32 register
// the master store uses, which is exactly what the old separate cast
// pass produced from memory.
//
// bias_factors is a 3-element device buffer {bc1, bc2, lr}: lr moved
// device-side so a warmup/cosine schedule no longer forfeits the
// captured-graph lane (the old kernels baked lr by value at capture).
// ---------------------------------------------------------------------------

typedef struct {
    unsigned long long param;  // f32*, chunk-offset applied
    unsigned long long grad;   // const f32*
    unsigned long long m;      // f32*
    unsigned long long v;      // f32*
    unsigned long long out;    // typed shadow (0 = none), chunk-offset applied
    unsigned long long n_wd;   // low 32: n elems; high 32: f32 bits of weight_decay
} AdamWChunkDesc;

#define DEFINE_ADAMW_MULTI(SUFFIX, OUT_TY, FROM_F)                             \
extern "C" __global__ void adamw_step_multi_##SUFFIX(                          \
    const AdamWChunkDesc* __restrict__ chunks,                                 \
    float beta1,                                                               \
    float beta2,                                                               \
    float eps,                                                                 \
    const float* __restrict__ bias_factors  /* [3] = {bc1, bc2, lr} */         \
) {                                                                            \
    const AdamWChunkDesc c = chunks[blockIdx.x];                               \
    const float bias_c1 = bias_factors[0];                                     \
    const float bias_c2 = bias_factors[1];                                     \
    const float lr = bias_factors[2];                                          \
    float* param      = (float*)c.param;                                       \
    const float* grad = (const float*)c.grad;                                  \
    float* m          = (float*)c.m;                                           \
    float* v          = (float*)c.v;                                           \
    OUT_TY* out       = (OUT_TY*)c.out;                                        \
    /* Bit 31 of the packed length tags the shadow's dtype: f32-stays-f32 \
       tensors (norms, conv weights/bias, dt bias, D) get their compute   \
       copy written HERE instead of through a per-tensor D2D pass. */     \
    const int n = (int)(c.n_wd & 0x7FFFFFFFULL);                               \
    const bool out_is_f32 = ((c.n_wd >> 31) & 1ULL) != 0ULL;                   \
    const float weight_decay =                                                 \
        __uint_as_float((unsigned int)(c.n_wd >> 32));                         \
    const float one_minus_b1 = 1.f - beta1;                                    \
    const float one_minus_b2 = 1.f - beta2;                                    \
    const float decay_factor = 1.f - lr * weight_decay;                        \
    for (int i = threadIdx.x; i < n; i += blockDim.x) {                        \
        float g = grad[i];                                                     \
        float p = param[i];                                                    \
        float mi = m[i] * beta1 + one_minus_b1 * g;                            \
        float vi = v[i] * beta2 + one_minus_b2 * g * g;                        \
        m[i] = mi;                                                             \
        v[i] = vi;                                                             \
        float m_hat = mi * bias_c1;                                            \
        float v_hat = vi * bias_c2;                                            \
        float np = decay_factor * p - lr * m_hat / (sqrtf(v_hat) + eps);       \
        param[i] = np;                                                         \
        if (out) {                                                             \
            if (out_is_f32) { ((float*)out)[i] = np; }                         \
            else            { out[i] = FROM_F(np); }                           \
        }                                                                      \
    }                                                                          \
}

// FROM_F helpers come from the shared typed prelude (the kernels are
// compiled as one blob with _typed_prelude.cuh inlined first).
DEFINE_ADAMW_MULTI(f32,  float,         from_f_f32)
DEFINE_ADAMW_MULTI(bf16, __nv_bfloat16, from_f_bf16)
DEFINE_ADAMW_MULTI(f16,  __half,        from_f_f16)
