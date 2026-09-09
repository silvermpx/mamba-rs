pub const SYMBOL: &str = "gemm_bi_tn_test_transpose_f32_32x8_v1";

const PRELUDE: &str = include_str!("../../kernels/_typed_prelude.cuh");

const SOURCE: &str = r#"
#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8
#define TRANSPOSE_THREADS 256
#define TRANSPOSE_SMEM_BYTES 4224

extern "C" __global__ __launch_bounds__(TRANSPOSE_THREADS, 4)
void __TRANSPOSE_SYMBOL__(
    float* __restrict__ destination,
    const float* __restrict__ source,
    int rows,
    int columns
) {
    __shared__ float tile[32][33];
    const int x = (int)blockIdx.x * TRANSPOSE_TILE_DIM + (int)threadIdx.x;
    const int y = (int)blockIdx.y * TRANSPOSE_TILE_DIM + (int)threadIdx.y;

#pragma unroll
    for (int offset = 0; offset < TRANSPOSE_TILE_DIM;
         offset += TRANSPOSE_BLOCK_ROWS) {
        const int row = y + offset;
        tile[(int)threadIdx.y + offset][(int)threadIdx.x] =
            row < rows && x < columns
                ? source[(long long)row * columns + x]
                : 0.0f;
    }
    __syncthreads();

    const int output_column =
        (int)blockIdx.y * TRANSPOSE_TILE_DIM + (int)threadIdx.x;
    const int output_row =
        (int)blockIdx.x * TRANSPOSE_TILE_DIM + (int)threadIdx.y;
#pragma unroll
    for (int offset = 0; offset < TRANSPOSE_TILE_DIM;
         offset += TRANSPOSE_BLOCK_ROWS) {
        const int row = output_row + offset;
        if (row < columns && output_column < rows) {
            destination[(long long)row * rows + output_column] =
                tile[(int)threadIdx.x][(int)threadIdx.y + offset];
        }
    }
}

#undef TRANSPOSE_SMEM_BYTES
#undef TRANSPOSE_THREADS
#undef TRANSPOSE_BLOCK_ROWS
#undef TRANSPOSE_TILE_DIM
"#;

pub fn compose_source() -> String {
    format!(
        "{PRELUDE}\n{}",
        SOURCE.replace("__TRANSPOSE_SYMBOL__", SYMBOL)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_is_canonical_32x8_padded_coalesced_transpose() {
        let source = compose_source();
        assert_eq!(source.matches(SYMBOL).count(), 1);
        for contract in [
            "TRANSPOSE_TILE_DIM 32",
            "TRANSPOSE_BLOCK_ROWS 8",
            "TRANSPOSE_THREADS 256",
            "TRANSPOSE_SMEM_BYTES 4224",
            "__launch_bounds__(TRANSPOSE_THREADS, 4)",
            "float tile[32][33]",
            "offset += TRANSPOSE_BLOCK_ROWS",
            "__syncthreads()",
            "destination[(long long)row * rows + output_column]",
            "source[(long long)row * columns + x]",
        ] {
            assert!(source.contains(contract), "missing {contract}");
        }
        assert_eq!(source.matches("offset += TRANSPOSE_BLOCK_ROWS").count(), 2);
        assert!(!source.contains("__fmaf_rn"));
        assert!(!source.contains("atomic"));
    }
}
