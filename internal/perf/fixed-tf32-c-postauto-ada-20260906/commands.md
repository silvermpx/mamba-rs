# Reproduction and cache provenance

The initial post-AUTO/prefix runs used this environment on the isolated Ada
source and binary:

```sh
export PATH=/root/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.2
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}
fixed_bench=/root/target-ada-dispatch-review-gxJWgz/release/deps/gemm_bi_fixed_performance-429f558dd590762a

"$fixed_bench" fixed_sm89_tf32_c_auto_prefix_special_bias_graph_bits \
  --ignored --exact --nocapture --test-threads=1

MAMBA_FIXED_ADA_VENDOR=1 MAMBA_FIXED_VENDOR_EXACT_CC=8.9 \
MAMBA_FIXED_ADA_ROWS=tf32 MAMBA_FIXED_ADA_CELLS=hot_c \
MAMBA_FIXED_ADA_BIAS=0,1 MAMBA_FIXED_VENDOR_TILES=Tf32M128S2 \
MAMBA_FIXED_VENDOR_PATHS=eager,graph MAMBA_FIXED_ADA_WINDOWS=101 \
  "$fixed_bench" fixed_ada_forced_rungs_paired_precision_cublas \
  --ignored --exact --nocapture --test-threads=1
```

`NVIDIA_TF32_OVERRIDE`, `CUDA_CACHE_DISABLE`, and `MAMBA_RS_KERNEL_CACHE`
were unset in those initial runs. They are default/warm-cache production
evidence, with actual loaded artifact identities embedded in every row;
they must not be described as cold/private-cache qualification. A follow-up
101-window run with Driver cache disabled and persistent kernel caching
disabled confirms all8own-win cohorts; see sibling
`../fixed-tf32-c-postauto-privatecache-ada-20260906/README.md` for the exact
cache-mode distinction. Preserve both records. An additional0700private-cache
run is supplementary, not a replacement of the uncached measurements.
