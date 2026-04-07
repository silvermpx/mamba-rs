// Mamba activation kernels: SiLU + softplus (forward + backward).
//
// All kernels operate on flat arrays [n] with 1D grid.
// Thread block: 256 threads. Grid: (n + 255) / 256 blocks.
//
// Uses exp2f(x * LOG2E) for single PTX instruction (Tri Dao optimization).

#define LOG2E 1.4426950408889634f

// SiLU (Swish) forward (in-place): x[i] = x[i] * sigmoid(x[i])
extern "C" __global__ void silu_forward(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v / (1.0f + exp2f(-v * LOG2E));
}

// Softplus forward (in-place): x[i] = x > 20 ? x : log(1 + exp(x))
// Threshold 20.0 matches Mamba convention (avoids overflow).
extern "C" __global__ void softplus_forward(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v > 20.0f ? v : logf(1.0f + exp2f(v * LOG2E));
}

// SiLU backward: dx[i] = dy[i] * sigma * (1 + x * (1 - sigma))
// Uses pre-SiLU input x (must be saved during forward).
extern "C" __global__ void silu_backward(
    float* dx, const float* x_saved, const float* dy, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x_saved[i];
    float sigma = 1.0f / (1.0f + exp2f(-xi * LOG2E));
    dx[i] = dy[i] * sigma * (1.0f + xi * (1.0f - sigma));
}

// Softplus backward: dx[i] = dy[i] * sigmoid(x[i])
extern "C" __global__ void softplus_backward(
    float* dx, const float* x_saved, const float* dy, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] = dy[i] / (1.0f + exp2f(-x_saved[i] * LOG2E));
}
