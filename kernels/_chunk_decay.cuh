// Log decay tape for a chunk of a causal recurrence, a*dt <= 0.
// A usual row stores its prefix. Once the carry from the chunk start is
// exactly zero in f32, a large prefix can hide all subsequent decays.
// Those rows instead store their own log step. The storage still exponentiates
// to zero for every consumer of the absolute prefix (including state passing).
#ifndef MAMBA_CHUNK_DECAY_CUH
#define MAMBA_CHUNK_DECAY_CUH

__device__ __forceinline__ float chunk_decay_prefix(float previous, float step, float& remainder, bool fresh) {
    if (fresh) { previous = 0.0f; remainder = 0.0f; }
    float sum = previous + step;
    // A visible incoming carry must keep its real prefix. Below exp(-104),
    // switch if subtraction already loses a material fraction of this step,
    // or before reaching the range reserved for local steps.
    if (previous <= -1024.0f || sum <= -1024.0f ||
        (sum <= -104.0f && step != 0.0f && fabsf(step) < fabsf(previous) * 0x1p-10f)) {
        if (isnan(step)) return step;
        // exp(-128) rounds to zero in f32. Scaling by an exact power of two
        // leaves every material step intact, even for sub-unit decays.
        // The sentinel represents a zero or a step smaller than 2^-108,
        // far too small to alter a float exponential over a chunk.
        return -fmaxf(1024.0f, fminf(-step, 128.0f) * 0x1p118f);
    }
    // Keep the correction when a small increment barely moves the live
    // prefix. Whole chunks of such steps can move a representable carry.
    if (remainder != 0.0f || (step != 0.0f && fabsf(step) < fabsf(previous) * 0x1p-10f)) {
        float increment = __fadd_rn(step, remainder);
        sum = __fadd_rn(previous, increment);
        remainder = __fsub_rn(increment, __fsub_rn(sum, previous));
    }
    return sum;
}

// Log decay across (left,right]. The usual subtraction remains the fast path.
// 'first' points to the left endpoint. Strides allow both scalar and
// interleaved coefficient tapes. The caller masks intervals across packed resets.
__device__ __forceinline__ float chunk_decay_interval(
    const float* first, long long stride, int distance, float left, float right
) {
    if (!(right <= -1024.0f)) return right - left;
    float tail = 0.0f;
    for (int k = distance; k > 0; --k) {
        float word = k == distance ? right : first[(long long)k * stride];
        if (!(word <= -1024.0f)) return left <= -1024.0f && !isnan(word) ? 0.0f : tail + (word - left);
        if (word != -1024.0f) tail += word * 0x1p-118f;
        if (tail <= -104.0f) return tail;
    }
    return tail;
}

// A transition whose own multiplier vanished separates two adjoint sums.
// The absolute prefix alone cannot distinguish it from many small decays.
__device__ __forceinline__ bool chunk_decay_cuts(const float* tape, int t, int first) {
    const float word = tape[t];
    if (!(word <= -104.0f)) return false;
    if (word <= -1024.0f) return word * 0x1p-118f <= -104.0f;
    return (t == first ? word : word - tape[t - 1]) <= -104.0f;
}

__device__ __forceinline__ int chunk_decay_span_end(const float* tape, int a, int e) {
    if (tape[e - 1] <= -104.0f) {
        for (int t = a + 1; t < e; ++t) if (chunk_decay_cuts(tape, t, a)) return t;
    }
    return e;
}

// When endpoint sums lose a small term to a much larger one, count just
// the pairs whose state actually crosses this token. The ordinary fold
// keeps its summation order without an extra triangle or shared buffer.
template<typename Pair, typename Entering>
__device__ __forceinline__ float chunk_direct_adjoint(int k, int a, int e, const float* exiting,
                                                     float passing, Pair pair, Entering entering) {
    double value = passing;
    for (int s = a; s < k; ++s) value += exiting[s];
    for (int t = k; t < e; ++t) {
        value += entering(t);
        for (int s = a; s < k; ++s) value += pair(s, t);
    }
    return (float)value;
}

#endif
