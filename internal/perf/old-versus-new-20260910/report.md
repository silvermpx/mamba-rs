# Old main (0.6.9, `d8f2efbe`) against the release tree (`c65c1c85`) on one board

Board: NVIDIA RTX 6000 Ada Generation (SM89, 142 SMs), driver 595.45.04,
CUDA 13.2. Both trees were built on the box from clean snapshots
(`/root/mamba-rs-main-d8f2efbe`, `/root/mamba-rs-new-c65c1c85`); the
adapters under `adapter/` are the only files added to either snapshot. Every
GEMM control variable is cleared by the runner; each process gets a private
driver and kernel cache; an idle check (three quiet samples) precedes every
process. Runs are mirrored blocks of separate processes (old, new, new, old).

The raw logs of each run are under `run-*/` next to this file; the runner
scripts and adapters under `adapter/`.

## Set A: inference step, exact f32, Triad family (common lane)

`GpuMambaBackbone` at `MambaConfig::default()` (d_model 128, 3 layers), seed
42, input 0.1, batches 1/4/16/64/128, twenty warmups, the original eager
and graph iteration counts of `tests/m1_gpu_benchmark.rs`. Both trees pin
`batch_invariant=true`, family Triad, tensor cores off, fast off, so both
run the exact scalar f32 route. Eight fixed steps are digested per batch
before timing; the digests of the two trees are identical for every batch,
step and path (80 of 80), eager repeats and graph replays are bit-identical
within each tree, and the CPU reference agrees (cos 1.000000, relative L2
about 1e-6).

Two blocks (`run-20260910T192454Z/A-*.log`), microseconds per step, median
of the four runs of each tree:

| batch | path | 0.6.9 | 0.7.0 | 0.6.9 / 0.7.0 |
|---:|---|---:|---:|---:|
| 1 | eager | 141.7 | 144.3 | 0.98× |
| 1 | graph | 102.8 | 150.3 | 0.68× |
| 4 | eager | 146.4 | 147.3 | 0.99× |
| 4 | graph | 106.9 | 154.4 | 0.69× |
| 16 | eager | 151.1 | 151.8 | 1.00× |
| 16 | graph | 112.1 | 157.3 | 0.71× |
| 64 | eager | 208.4 | 203.1 | 1.03× |
| 64 | graph | 165.3 | 229.8 | 0.72× |
| 128 | eager | 279.9 | 271.1 | 1.03× |
| 128 | graph | 237.2 | 295.3 | 0.80× |

Spread between the four runs of one tree: 0.1 to 3.2 percent. The eager
step is unchanged; the graph replay of this tiny f32 model is 20 to 30
percent slower on the release tree. This lane is not what a model context
runs by default in 0.7.0 (that is set B); it is the lane the two trees
share exactly, and the slowdown is a finding to explain before release.

## Set C: training step, `MambaTrainer`, all three GEMM settings

`benches/gemm_bi_trainer_step_bench.rs` on the release tree and the
equivalent `gemm_bi_determinism::bench_sgemm_bi_vs_tf32` on the old tree:
`MambaTrainer::step` with graph replay, three warmups and twenty timed
steps per arm, four shapes, f32/bf16/f16. The old bench compares f32
against cuBLAS with TF32 (the old default handle math) and half against
cuBLAS Pedantic; the release bench times cuBLAS Fast and cuBLAS Pedantic
for every dtype. One block (`run-20260910T192454Z/C-*.log`), milliseconds
per step, median of the two runs of each tree:

| model | dtype | cuBLAS Fast (0.6.9 / 0.7.0) | cuBLAS Pedantic (0.6.9 / 0.7.0) | deterministic scalar 0.6.9 | 0.7.0 | ratio | deterministic tensor cores 0.6.9 | 0.7.0 | ratio |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| d128, 2 layers, B=16, T=64 | f32 | 2.44 / 2.43 | – / 2.50 | 3.03 | 2.79 | 1.09× | – | – | – |
| d128, 2 layers, B=16, T=64 | bf16 | – / 2.02 | 2.52 / 2.51 | 2.84 | 2.58 | 1.10× | 2.35 | 2.10 | 1.12× |
| d128, 2 layers, B=16, T=64 | f16 | – / 2.04 | 2.16 / 2.13 | 2.88 | 2.62 | 1.10× | 2.39 | 2.14 | 1.12× |
| d256, 4 layers, B=16, T=128 | f32 | 9.96 / 9.87 | – / 10.18 | 12.52 | 12.08 | 1.04× | – | – | – |
| d256, 4 layers, B=16, T=128 | bf16 | – / 8.07 | 10.50 / 10.42 | 11.57 | 11.22 | 1.03× | 9.68 | 8.51 | 1.14× |
| d256, 4 layers, B=16, T=128 | f16 | – / 8.19 | 9.31 / 8.81 | 12.12 | 11.36 | 1.07× | 10.13 | 8.65 | 1.17× |
| d768, 4 layers, B=8, T=256 | f32 | 24.88 / 24.44 | – / 27.91 | 34.88 | 32.16 | 1.08× | – | – | – |
| d768, 4 layers, B=8, T=256 | bf16 | – / 20.11 | 26.94 / 26.72 | 31.83 | 30.36 | 1.05× | 22.32 | 20.30 | 1.10× |
| d768, 4 layers, B=8, T=256 | f16 | – / 20.27 | 24.62 / 24.39 | 32.09 | 30.57 | 1.05× | 22.55 | 20.54 | 1.10× |
| d1536, 2 layers, B=4, T=256 | f32 | 14.86 / 14.67 | – / 18.43 | 24.54 | 22.68 | 1.08× | – | – | – |
| d1536, 2 layers, B=4, T=256 | bf16 | – / 12.30 | 18.23 / 17.99 | 22.64 | 22.96 | 0.99× | 13.21 | 13.01 | 1.02× |
| d1536, 2 layers, B=4, T=256 | f16 | – / 12.86 | 16.59 / 16.49 | 23.12 | 23.25 | 0.99× | 13.57 | 13.29 | 1.02× |

The cuBLAS arms agree between the trees to within 1 to 2 percent (same
library, same board), which is the cross-tree control. The like-for-like
deterministic arms gain 2 to 17 percent per step; the step is dominated by
the scan and the other kernels this release did not change.

## Set B: inference step on the inference family, f32 and bf16

Same model, batches, warmups and iteration counts as set A; both trees pin
`batch_invariant=true`, the inference family (`Fixed` on 0.6.9, `Inference`
on the release), tensor cores off, fast off. bf16 uses an identity input
projection on both sides (the mixed engine requires it); the CPU reference
carries the identity as an explicit matrix. Digests: identical between the
trees for every batch, step and path in both storages (80 of 80 each);
eager repeats and graph replays bit-identical within each tree; CPU
reference cos 1.000000 (f32) and at least 0.99998 (bf16). Two blocks
(`run-20260910T200051Z/B-*.log`), microseconds per step, median of the
four runs of each tree:

| storage | batch | path | 0.6.9 (Fixed) µs/step | 0.7.0 (Inference) µs/step | 0.6.9 / 0.7.0 |
|---|---:|---|---:|---:|---:|
| f32 | 1 | eager | 238.5 | 238.6 | 1.00× |
| f32 | 1 | graph | 199.5 | 245.0 | 0.81× |
| f32 | 4 | eager | 246.0 | 246.8 | 1.00× |
| f32 | 4 | graph | 206.1 | 250.3 | 0.82× |
| f32 | 16 | eager | 246.7 | 253.4 | 0.97× |
| f32 | 16 | graph | 207.6 | 256.2 | 0.81× |
| f32 | 64 | eager | 270.6 | 262.7 | 1.03× |
| f32 | 64 | graph | 232.4 | 269.4 | 0.86× |
| f32 | 128 | eager | 296.1 | 287.0 | 1.03× |
| f32 | 128 | graph | 258.9 | 294.3 | 0.88× |
| bf16 | 1 | eager | 124.0 | 130.4 | 0.95× |
| bf16 | 1 | graph | 94.2 | 139.2 | 0.68× |
| bf16 | 4 | eager | 128.4 | 130.6 | 0.98× |
| bf16 | 4 | graph | 97.8 | 141.2 | 0.69× |
| bf16 | 16 | eager | 132.2 | 134.8 | 0.98× |
| bf16 | 16 | graph | 101.8 | 146.4 | 0.70× |
| bf16 | 64 | eager | 146.1 | 147.5 | 0.99× |
| bf16 | 64 | graph | 117.8 | 160.1 | 0.74× |
| bf16 | 128 | eager | 195.7 | 177.0 | 1.11× |
| bf16 | 128 | graph | 165.9 | 186.6 | 0.89× |

The eager step is unchanged. The graph replay is again 45 to 50
microseconds slower on the release tree at every batch and both storages,
the same constant as in set A; the cause is the per-replay launch-set
digest rebuild in `CapturedGemmGraphPlan::with_validated_launch`, fixed in
the working tree and to be re-measured.

## Set D: kernel level, every op and dtype

`adapter/gemm_kernel_old_vs_new.{old,new}.rs`, one copy per tree, installed
as `examples/gemm_kernel_old_vs_new.rs`. Per cell three arms in one process
(the tree's deterministic route with `batch_invariant` on and tensor cores
on for half; cuBLAS in the tree's fast setting; cuBLAS in the tree's
pedantic setting), one context per arm, operands uploaded per arm, a
numeric check of every arm against the pedantic arm (relative L2 below
5e-2 everywhere, below 1e-4 for exact f32), calibrated launch counts,
21 windows, arms rotating in mirrored order inside every window, CUDA
events. Two blocks (`run-20260910T201618Z/D-*.log`), median of the four
runs of each tree. The first launch of this set failed twice on the
adapter (the old tree has no `GpuDevice::identity`; the f16 conversion
rejected subnormal operands) and was restarted after each fix; the logs of
the aborted launches were discarded.

Geometric means of 0.6.9 / 0.7.0: Triad large shapes BF16 1.29, F16 1.25,
F32 exact 1.40; Triad small shapes 1.08 / 1.08 / 1.25; Inference BF16 1.33,
F16 1.33, F32 exact 1.39. Normalising each tree's deterministic time by its
own cuBLAS Fast time (identical calls for half precision) changes these by
at most 0.06 (half) and 0.14 (f32) in favour of the release tree: the
release process measured 0 to 12 percent slower on identical cuBLAS half
calls, run-to-run drift on the power-capped board. The f32 and Pedantic
vendor arms are not identical calls between the trees (`sgemm` under the
handle math mode against `GemmEx` with an explicit compute type).

| family | shape | M × K × N | op | precision | 0.6.9 det µs | 0.7.0 det µs | 0.6.9 / 0.7.0 | cuBLAS Fast µs | cuBLAS Pedantic µs | 0.7.0 vs Fast | 0.7.0 vs Pedantic |
|---|---|---|:--:|---|---:|---:|---:|---:|---:|---:|---:|
| triad | d128_out_proj | 1024 × 256 × 128 | NN | BF16 | 6.58 | 6.59 | 1.00× | 5.38 | 14.63 | 0.82× | 2.22× |
| triad | d128_out_proj | 1024 × 256 × 128 | NT | BF16 | 5.34 | 5.35 | 1.00× | 4.66 | 11.67 | 0.87× | 2.18× |
| triad | d128_out_proj | 1024 × 256 × 128 | TN | BF16 | 18.12 | 18.18 | 1.00× | 7.20 | 50.48 | 0.40× | 2.78× |
| triad | underfill | 256 × 512 × 384 | NN | BF16 | 10.31 | 6.62 | 1.56× | 7.35 | 25.94 | 1.11× | 3.92× |
| triad | underfill | 256 × 512 × 384 | NT | BF16 | 7.84 | 7.84 | 1.00× | 6.53 | 27.39 | 0.83× | 3.49× |
| triad | underfill | 256 × 512 × 384 | TN | BF16 | 8.86 | 8.92 | 0.99× | 5.53 | 16.97 | 0.62× | 1.90× |
| triad | d128_in_proj | 1024 × 128 × 512 | NN | BF16 | 7.39 | 5.76 | 1.28× | 4.81 | 10.44 | 0.84× | 1.81× |
| triad | d128_in_proj | 1024 × 128 × 512 | NT | BF16 | 9.31 | 9.32 | 1.00× | 7.64 | 35.15 | 0.82× | 3.77× |
| triad | d128_in_proj | 1024 × 128 × 512 | TN | BF16 | 18.17 | 18.35 | 0.99× | 7.91 | 50.46 | 0.43× | 2.75× |
| triad | d768_out_proj | 2048 × 1536 × 768 | NN | BF16 | 55.55 | 37.95 | 1.46× | 47.42 | 188.76 | 1.25× | 4.97× |
| triad | d768_out_proj | 2048 × 1536 × 768 | NT | BF16 | 60.30 | 41.48 | 1.45× | 39.62 | 206.62 | 0.96× | 4.98× |
| triad | d768_out_proj | 2048 × 1536 × 768 | TN | BF16 | 74.59 | 55.20 | 1.35× | 50.31 | 213.37 | 0.91× | 3.87× |
| triad | prism_in_proj | 4621 × 384 × 1928 | NN | BF16 | 83.63 | 67.09 | 1.25× | 76.54 | 289.41 | 1.14× | 4.31× |
| triad | prism_in_proj | 4621 × 384 × 1928 | NT | BF16 | 74.75 | 51.89 | 1.44× | 60.92 | 318.86 | 1.17× | 6.14× |
| triad | prism_in_proj | 4621 × 384 × 1928 | TN | BF16 | 81.36 | 77.67 | 1.05× | 54.77 | 334.08 | 0.71× | 4.30× |
| triad | d768_in_proj | 2048 × 768 × 3072 | NN | BF16 | 96.83 | 72.23 | 1.34× | 82.49 | 395.74 | 1.14× | 5.48× |
| triad | d768_in_proj | 2048 × 768 × 3072 | NT | BF16 | 104.20 | 71.36 | 1.46× | 77.02 | 361.20 | 1.08× | 5.06× |
| triad | d768_in_proj | 2048 × 768 × 3072 | TN | BF16 | 145.22 | 91.79 | 1.58× | 92.72 | 393.26 | 1.01× | 4.28× |
| triad | large_deep | 4096 × 3072 × 1536 | NN | BF16 | 385.14 | 378.78 | 1.02× | 264.67 | 1634.88 | 0.70× | 4.32× |
| triad | large_deep | 4096 × 3072 × 1536 | NT | BF16 | 419.33 | 408.34 | 1.03× | 268.32 | 1588.74 | 0.66× | 3.89× |
| triad | large_deep | 4096 × 3072 × 1536 | TN | BF16 | 466.08 | 392.93 | 1.19× | 329.65 | 1518.26 | 0.84× | 3.86× |
| triad | d128_out_proj | 1024 × 256 × 128 | NN | F16 | 6.54 | 6.63 | 0.99× | 4.79 | 9.39 | 0.72× | 1.42× |
| triad | d128_out_proj | 1024 × 256 × 128 | NT | F16 | 5.36 | 5.34 | 1.00× | 4.49 | 7.25 | 0.84× | 1.36× |
| triad | d128_out_proj | 1024 × 256 × 128 | TN | F16 | 18.05 | 18.23 | 0.99× | 7.39 | 9.89 | 0.41× | 0.54× |
| triad | underfill | 256 × 512 × 384 | NN | F16 | 10.30 | 6.61 | 1.56× | 5.73 | 11.56 | 0.87× | 1.75× |
| triad | underfill | 256 × 512 × 384 | NT | F16 | 7.82 | 7.83 | 1.00× | 5.30 | 11.33 | 0.68× | 1.45× |
| triad | underfill | 256 × 512 × 384 | TN | F16 | 8.88 | 8.87 | 1.00× | 5.57 | 10.69 | 0.63× | 1.20× |
| triad | d128_in_proj | 1024 × 128 × 512 | NN | F16 | 7.38 | 5.72 | 1.29× | 5.31 | 9.53 | 0.93× | 1.67× |
| triad | d128_in_proj | 1024 × 128 × 512 | NT | F16 | 9.40 | 9.40 | 1.00× | 5.85 | 12.64 | 0.62× | 1.34× |
| triad | d128_in_proj | 1024 × 128 × 512 | TN | F16 | 18.15 | 18.36 | 0.99× | 7.87 | 12.24 | 0.43× | 0.67× |
| triad | d768_out_proj | 2048 × 1536 × 768 | NN | F16 | 58.50 | 41.63 | 1.41× | 48.56 | 165.33 | 1.17× | 3.97× |
| triad | d768_out_proj | 2048 × 1536 × 768 | NT | F16 | 62.42 | 43.40 | 1.44× | 41.48 | 177.80 | 0.96× | 4.10× |
| triad | d768_out_proj | 2048 × 1536 × 768 | TN | F16 | 74.63 | 58.26 | 1.28× | 51.39 | 157.58 | 0.88× | 2.70× |
| triad | prism_in_proj | 4621 × 384 × 1928 | NN | F16 | 96.13 | 76.28 | 1.26× | 83.41 | 253.54 | 1.09× | 3.32× |
| triad | prism_in_proj | 4621 × 384 × 1928 | NT | F16 | 79.65 | 58.51 | 1.36× | 99.26 | 254.27 | 1.70× | 4.35× |
| triad | prism_in_proj | 4621 × 384 × 1928 | TN | F16 | 93.82 | 90.10 | 1.04× | 59.13 | 234.39 | 0.66× | 2.60× |
| triad | d768_in_proj | 2048 × 768 × 3072 | NN | F16 | 112.52 | 84.23 | 1.34× | 76.55 | 359.08 | 0.91× | 4.26× |
| triad | d768_in_proj | 2048 × 768 × 3072 | NT | F16 | 118.57 | 85.78 | 1.38× | 77.27 | 333.93 | 0.90× | 3.89× |
| triad | d768_in_proj | 2048 × 768 × 3072 | TN | F16 | 142.35 | 104.04 | 1.37× | 101.55 | 353.94 | 0.98× | 3.40× |
| triad | large_deep | 4096 × 3072 × 1536 | NN | F16 | 389.36 | 383.41 | 1.02× | 269.13 | 1236.03 | 0.70× | 3.22× |
| triad | large_deep | 4096 × 3072 × 1536 | NT | F16 | 466.09 | 446.35 | 1.04× | 296.18 | 1413.50 | 0.66× | 3.17× |
| triad | large_deep | 4096 × 3072 × 1536 | TN | F16 | 548.05 | 445.72 | 1.23× | 313.72 | 1395.52 | 0.70× | 3.13× |
| triad | d128_out_proj | 1024 × 256 × 128 | NN | F32 exact | 10.68 | 10.13 | 1.05× | 6.76 | 8.69 | 0.67× | 0.86× |
| triad | d128_out_proj | 1024 × 256 × 128 | NT | F32 exact | 13.27 | 12.84 | 1.03× | 5.02 | 8.17 | 0.39× | 0.64× |
| triad | d128_out_proj | 1024 × 256 × 128 | TN | F32 exact | 54.17 | 27.53 | 1.97× | 9.75 | 10.49 | 0.35× | 0.38× |
| triad | underfill | 256 × 512 × 384 | NN | F32 exact | 13.07 | 12.67 | 1.03× | 9.98 | 12.19 | 0.79× | 0.96× |
| triad | underfill | 256 × 512 × 384 | NT | F32 exact | 16.00 | 15.69 | 1.02× | 7.99 | 11.76 | 0.51× | 0.75× |
| triad | underfill | 256 × 512 × 384 | TN | F32 exact | 83.79 | 80.66 | 1.04× | 7.83 | 9.54 | 0.10× | 0.12× |
| triad | d128_in_proj | 1024 × 128 × 512 | NN | F32 exact | 16.46 | 15.80 | 1.04× | 5.66 | 9.95 | 0.36× | 0.63× |
| triad | d128_in_proj | 1024 × 128 × 512 | NT | F32 exact | 17.88 | 17.17 | 1.04× | 9.30 | 13.73 | 0.54× | 0.80× |
| triad | d128_in_proj | 1024 × 128 × 512 | TN | F32 exact | 105.63 | 35.41 | 2.98× | 9.98 | 12.69 | 0.28× | 0.36× |
| triad | d768_out_proj | 2048 × 1536 × 768 | NN | F32 exact | 227.55 | 166.72 | 1.36× | 77.20 | 173.85 | 0.46× | 1.04× |
| triad | d768_out_proj | 2048 × 1536 × 768 | NT | F32 exact | 369.26 | 181.79 | 2.03× | 86.03 | 189.65 | 0.47× | 1.04× |
| triad | d768_out_proj | 2048 × 1536 × 768 | TN | F32 exact | 303.73 | 200.17 | 1.52× | 86.13 | 143.73 | 0.43× | 0.72× |
| triad | prism_in_proj | 4621 × 384 × 1928 | NN | F32 exact | 320.72 | 248.44 | 1.29× | 132.70 | 267.06 | 0.53× | 1.07× |
| triad | prism_in_proj | 4621 × 384 × 1928 | NT | F32 exact | 499.64 | 349.18 | 1.43× | 89.32 | 283.07 | 0.26× | 0.81× |
| triad | prism_in_proj | 4621 × 384 × 1928 | TN | F32 exact | 400.47 | 287.24 | 1.39× | 122.05 | 233.84 | 0.42× | 0.81× |
| triad | d768_in_proj | 2048 × 768 × 3072 | NN | F32 exact | 350.68 | 300.03 | 1.17× | 123.26 | 357.63 | 0.41× | 1.19× |
| triad | d768_in_proj | 2048 × 768 × 3072 | NT | F32 exact | 826.43 | 365.30 | 2.26× | 158.01 | 368.56 | 0.43× | 1.01× |
| triad | d768_in_proj | 2048 × 768 × 3072 | TN | F32 exact | 530.94 | 326.97 | 1.62× | 137.93 | 277.40 | 0.42× | 0.85× |
| triad | large_deep | 4096 × 3072 × 1536 | NN | F32 exact | 1482.29 | 1644.94 | 0.90× | 427.66 | 1078.08 | 0.26× | 0.66× |
| triad | large_deep | 4096 × 3072 × 1536 | NT | F32 exact | 2859.45 | 2352.26 | 1.22× | 463.22 | 1223.87 | 0.20× | 0.52× |
| triad | large_deep | 4096 × 3072 × 1536 | TN | F32 exact | 1518.21 | 1424.00 | 1.07× | 498.81 | 974.51 | 0.35× | 0.68× |
| inference | hot_a | 4621 × 384 × 1928 | NN | BF16 | 87.24 | 68.01 | 1.28× | 75.98 | 278.97 | 1.12× | 4.10× |
| inference | hot_c | 4621 × 1928 × 384 | NN | BF16 | 69.62 | 54.67 | 1.27× | 76.35 | 298.13 | 1.40× | 5.45× |
| inference | hot_d | 2048 × 768 × 2304 | NN | BF16 | 89.23 | 65.97 | 1.35× | 60.26 | 278.22 | 0.91× | 4.22× |
| inference | hot_e | 2048 × 2304 × 768 | NN | BF16 | 78.56 | 60.65 | 1.30× | 67.80 | 280.68 | 1.12× | 4.63× |
| inference | hot_b | 4621 × 768 × 2304 | NN | BF16 | 189.76 | 132.93 | 1.43× | 109.89 | 762.18 | 0.83× | 5.73× |
| inference | hot_a | 4621 × 384 × 1928 | NN | F16 | 100.31 | 76.63 | 1.31× | 81.11 | 246.47 | 1.06× | 3.22× |
| inference | hot_c | 4621 × 1928 × 384 | NN | F16 | 78.52 | 61.94 | 1.27× | 80.76 | 235.03 | 1.30× | 3.79× |
| inference | hot_d | 2048 × 768 × 2304 | NN | F16 | 98.58 | 70.93 | 1.39× | 69.09 | 265.75 | 0.97× | 3.75× |
| inference | hot_e | 2048 × 2304 × 768 | NN | F16 | 85.94 | 67.88 | 1.27× | 58.99 | 236.83 | 0.87× | 3.49× |
| inference | hot_b | 4621 × 768 × 2304 | NN | F16 | 217.35 | 153.04 | 1.42× | 131.43 | 659.34 | 0.86× | 4.31× |
| inference | hot_a | 4621 × 384 × 1928 | NN | F32 exact | 346.05 | 240.49 | 1.44× | 131.81 | 267.54 | 0.55× | 1.11× |
| inference | hot_c | 4621 × 1928 × 384 | NN | F32 exact | 407.00 | 340.66 | 1.19× | 94.13 | 230.09 | 0.28× | 0.68× |
| inference | hot_d | 2048 × 768 × 2304 | NN | F32 exact | 303.10 | 216.95 | 1.40× | 103.72 | 253.20 | 0.48× | 1.17× |
| inference | hot_e | 2048 × 2304 × 768 | NN | F32 exact | 361.35 | 249.24 | 1.45× | 113.82 | 254.73 | 0.46× | 1.02× |
| inference | hot_b | 4621 × 768 × 2304 | NN | F32 exact | 888.70 | 604.26 | 1.47× | 275.58 | 626.88 | 0.46× | 1.04× |


## Sets A and B again, on the corrected release tree (`cce73716`)

`adapter/run-set-ab-fixed.sh` with `/root/mamba-rs-new-cce73716` (a
snapshot of the committed tree plus the two adapters) as the new endpoint;
logs in `run-abfixed-20260910T205149Z/`. Digests identical between the
trees in both sets and both storages, as before.

Set A (exact f32, Triad lane), median of four:

| batch | arm | old main µs/step (median of 4) | 0.7.0 µs/step (median of 4) | old / new |
|---:|---|---:|---:|---:|
| 1 | eager | 141.6 | 143.7 | 0.985× |
| 1 | graph | 102.8 | 104.0 | 0.988× |
| 4 | eager | 146.4 | 147.0 | 0.996× |
| 4 | graph | 106.7 | 107.5 | 0.992× |
| 16 | eager | 150.9 | 151.6 | 0.995× |
| 16 | graph | 112.0 | 112.4 | 0.997× |
| 64 | eager | 208.3 | 203.0 | 1.026× |
| 64 | graph | 164.9 | 159.8 | 1.032× |
| 128 | eager | 279.5 | 271.5 | 1.030× |
| 128 | graph | 237.2 | 229.4 | 1.034× |

Set B (inference family, f32 and bf16), median of four:

| storage | batch | path | 0.6.9 (Fixed) µs/step | 0.7.0 (Inference) µs/step | 0.6.9 / 0.7.0 |
|---|---:|---|---:|---:|---:|
| f32 | 1 | eager | 238.5 | 238.2 | 1.00× |
| f32 | 1 | graph | 199.6 | 196.9 | 1.01× |
| f32 | 4 | eager | 245.9 | 246.4 | 1.00× |
| f32 | 4 | graph | 206.1 | 204.3 | 1.01× |
| f32 | 16 | eager | 246.7 | 253.1 | 0.97× |
| f32 | 16 | graph | 207.8 | 212.1 | 0.98× |
| f32 | 64 | eager | 271.0 | 262.1 | 1.03× |
| f32 | 64 | graph | 232.6 | 222.8 | 1.04× |
| f32 | 128 | eager | 296.8 | 286.5 | 1.04× |
| f32 | 128 | graph | 259.4 | 250.0 | 1.04× |
| bf16 | 1 | eager | 124.1 | 127.2 | 0.98× |
| bf16 | 1 | graph | 93.9 | 96.6 | 0.97× |
| bf16 | 4 | eager | 128.8 | 129.6 | 0.99× |
| bf16 | 4 | graph | 97.8 | 100.3 | 0.98× |
| bf16 | 16 | eager | 132.5 | 134.0 | 0.99× |
| bf16 | 16 | graph | 102.1 | 103.2 | 0.99× |
| bf16 | 64 | eager | 146.4 | 147.4 | 0.99× |
| bf16 | 64 | graph | 117.8 | 119.6 | 0.98× |
| bf16 | 128 | eager | 197.5 | 176.3 | 1.12× |
| bf16 | 128 | graph | 167.5 | 146.6 | 1.14× |

The graph replay is back at the 0.6.9 level from batch 1 (104.0 against
102.8 µs) and ahead of it from batch 64; the per-replay launch-set digest
rebuild was the whole difference.
