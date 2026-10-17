// Every sequence's rows in reverse order, in place: the second layer of a
// bidirectional pair reads its sequences backwards. Row t of a sequence
// spanning [first, end) of its batch row trades places with row
// first + end - 1 - t; without a packing (`seq_idx` null) a sequence is the
// whole batch row, and with one each sequence turns within its own span, so
// the ids stay where they are.
//
// The rows move as whole units: 16 bytes a thread when a row's bytes and
// the buffer allow it, else 4 or 2. One block row per pair of rows, owned
// by the pair's lower row; the middle row of an odd sequence stays.
//
// Grid: (B * T, ceil(units / 256)). Block: 256.

#include "_varlen.cuh"

#define DEFINE_ROWS_REVERSE(SUFFIX, UNIT)                                       \
extern "C" __global__ void rows_reverse_##SUFFIX(                               \
    UNIT* __restrict__ rows,             /* [B, T, units] */                    \
    int T, int units,                                                           \
    const int* __restrict__ seq_idx      /* [B, T] or nullptr */                \
) {                                                                             \
    long long row = blockIdx.x;                                                 \
    long long b = row / T;                                                      \
    int t = (int)(row - b * T);                                                 \
    VlSpan s = vl_span(seq_idx, b, T, 0, T, t);                                 \
    int mirror = s.first + s.end - 1 - t;                                       \
    if (mirror <= t) return;                                                    \
    int u = blockIdx.y * blockDim.x + threadIdx.x;                              \
    if (u >= units) return;                                                     \
    UNIT* a = rows + row * units;                                               \
    UNIT* c = rows + (b * T + mirror) * units;                                  \
    UNIT va = a[u];                                                             \
    UNIT vc = c[u];                                                             \
    a[u] = vc;                                                                  \
    c[u] = va;                                                                  \
}

DEFINE_ROWS_REVERSE(u128, uint4)
DEFINE_ROWS_REVERSE(u32, unsigned int)
DEFINE_ROWS_REVERSE(u16, unsigned short)
