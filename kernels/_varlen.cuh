// Several sequences packed into the rows of one window (varlen), as the
// reference trains Mamba-2 and Mamba-3 with `seq_idx`: `seq_idx` holds one
// id per position, `[B, T]`, non-decreasing along a row, or is null for
// one sequence per row. Id 0 is the sequence a row carries in from the
// window before; a position whose id differs from the one before it (from
// 0 at the first position) starts a sequence, from the empty state. No
// state, conv history, previous-step term, angle or gradient crosses from
// one sequence into the next.
//
// Composed after the typed prelude in every module whose kernels take a
// packing.

#ifndef _MAMBA_VARLEN_CUH
#define _MAMBA_VARLEN_CUH

// The id of row b's position t: 0 without a packing, and 0 before the
// window, the sequence carried into it.
__device__ __forceinline__ int vl_id(
    const int* __restrict__ seq_idx, long long b, int seq, int t
) {
    return (seq_idx != nullptr && t >= 0) ? seq_idx[b * seq + t] : 0;
}

// Whether row b's position t starts a sequence; never without a packing.
__device__ __forceinline__ bool vl_starts(
    const int* __restrict__ seq_idx, long long b, int seq, int t
) {
    return seq_idx != nullptr && vl_id(seq_idx, b, seq, t) != vl_id(seq_idx, b, seq, t - 1);
}

// The first position in [lo, hi) of row b whose id is at least `id`, hi
// when there is none: the ids of a row never decrease, so a binary search
// finds where a sequence begins in a span. Without a packing every id is 0
// and the answer is lo for id 0, hi for any later id.
__device__ __forceinline__ int vl_lower_bound(
    const int* __restrict__ seq_idx, long long b, int seq, int lo, int hi, int id
) {
    if (seq_idx == nullptr) return id <= 0 ? lo : hi;
    const int* row = seq_idx + b * seq;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (row[mid] < id) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// The first position of the sequence that holds row b's position t, within
// [lo, t]: lo when that sequence began before lo.
__device__ __forceinline__ int vl_seq_first(
    const int* __restrict__ seq_idx, long long b, int seq, int lo, int t
) {
    return vl_lower_bound(seq_idx, b, seq, lo, t + 1, vl_id(seq_idx, b, seq, t));
}

// Where the sequence of a span's position t lies in the span: the span is
// row b's positions start .. start + len - 1 (a chunk), and every position
// here is counted from its start.
struct VlSpan {
    int first;         // the sequence's first position in the span
    int end;           // one past its last position in the span
    bool from_before;  // it continues the sequence before the span
    bool to_end;       // it reaches the span's last position
};

__device__ __forceinline__ VlSpan vl_span(
    const int* __restrict__ seq_idx, long long b, int seq, int start, int len, int t
) {
    if (seq_idx == nullptr) return VlSpan{0, len, true, true};
    const int id = seq_idx[b * seq + start + t];
    VlSpan s;
    s.first = vl_lower_bound(seq_idx, b, seq, start, start + t + 1, id) - start;
    s.end = vl_lower_bound(seq_idx, b, seq, start + t, start + len, id + 1) - start;
    s.from_before = id == vl_id(seq_idx, b, seq, start - 1);
    s.to_end = s.end == len;
    return s;
}

// How many positions at the start of a span continue the sequence before
// it: all of them without a packing.
__device__ __forceinline__ int vl_cont_end(
    const int* __restrict__ seq_idx, long long b, int seq, int start, int len
) {
    return vl_lower_bound(seq_idx, b, seq, start, start + len,
                          vl_id(seq_idx, b, seq, start - 1) + 1) - start;
}

// Whether a span holds no start: the sequence before it runs through it.
__device__ __forceinline__ bool vl_passes(
    const int* __restrict__ seq_idx, long long b, int seq, int start, int len
) {
    return vl_id(seq_idx, b, seq, start + len - 1) == vl_id(seq_idx, b, seq, start - 1);
}

#endif  // _MAMBA_VARLEN_CUH
