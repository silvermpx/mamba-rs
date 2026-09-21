# Changelog

## 0.7.4 (2026-09-22)

### Fixed

- Mamba-3 CPU decoding with `d_state` above 64.
- Mamba-3 f16 eager training now skips the optimizer step when the
  gradients overflow.
- The f16 training path now honours `step_skip_above` in both models and
  `control_clip_max_norm` in Mamba-3.
- A possible race on the conv state in the tiled conv kernel for
  sequences longer than 128 tokens.
- Original-format checkpoints: `tie_embeddings` is read, and the
  vocabulary is padded to `pad_vocab_size_multiple`.

### Changed

- Mamba-1 initialization matches the reference: PyTorch default bounds,
  `out_proj` scaled by `1/sqrt(n_layers)`, random conv bias.
- AdamW no longer applies weight decay to `A_log`, `D`, the dt bias and
  the norm scales by default. `with_reference_no_decay(false)` restores
  the previous behavior.

## 0.7.3 (2026-09-18)

Two settings instead of seven knobs, plain names, and the measured kernels
on every SM80-tier card. Bits are unchanged on the exact f32, BF16 and F16
routes: every bit-ledger key matches 0.7.1 and 0.7.2, and each new route
reproduces the one it replaces word for word. One numeric change: a `Tf32`
context whose shape had no measured TF32 kernel used to run exact f32 and
now runs the portable TF32 tier (the neighbour band below), on the Ada and
everywhere else.

### The GEMM contract is two settings

A GPU context is `WeightDtype` plus `GemmMode`, nothing else. `WeightDtype`
gains `Tf32`: f32 storage, products on the deterministic TF32 kernels
where one is measured and on exact f32 elsewhere. It is the route 0.7.0
spelled `set_f32_triad_policy(AllowDeterministicTf32)` or
`MAMBA_RS_BI_F32_POLICY=tf32`; `generate --dtype tf32` selects it too.

Everything that tuned the deterministic mode from outside is gone and no
longer read: `set_batch_invariant`, `set_fast_gemm`, `disable_tf32`,
`set_bi_gemm_family`, `set_bi_tensor_cores`, `set_f32_triad_policy`,
`set_half_triad_policy`, their getters, `gemm_flags`, the public
`BiGemmFamily`, `F32TriadPolicy` and `HalfTriadPolicy`, and the variables
`MAMBA_RS_BATCH_INVARIANT`, `MAMBA_RS_FAST_GEMM`, `MAMBA_RS_BI_GEMM_FAMILY`,
`MAMBA_RS_BI_TENSOR_CORES`, `MAMBA_RS_BI_F32_POLICY`,
`MAMBA_RS_BI_HALF_POLICY` and `MAMBA_RS_ARCH_RUNG`. `MAMBA_RS_GEMM_MODE` is
the one variable left. Each of those knobs only made the kernels slower
or reproduced an older bit family. The context derives them itself: the
family from its role (Inference for a model, Triad for a trainer), tensor
cores wherever the shape and the board allow, stream-K for the deep
weight-gradient reductions, the architecture kernels behind their
self-check. The qualification tools reach the derived controls through
`GpuCtx::route_controls()`, which is hidden and not part of the API. One
route moves: a context built from the environment in a cuBLAS mode and
switched to `Deterministic` afterwards now runs its half-precision weight
gradients on the stream-K kernels like every other deterministic context.

| you had | you do now |
|---|---|
| `WeightDtype::F32` + `set_f32_triad_policy(AllowDeterministicTf32)` or `MAMBA_RS_BI_F32_POLICY=tf32` | `WeightDtype::Tf32` |
| `MAMBA_RS_BI_F32_POLICY=exact` (or unset) | nothing; `WeightDtype::F32` is exact |
| `set_batch_invariant(true)` / `MAMBA_RS_BATCH_INVARIANT=1` | nothing; it is the default |
| `set_batch_invariant(false)` / `MAMBA_RS_BATCH_INVARIANT=0` | `GemmMode::CublasPedantic` / `MAMBA_RS_GEMM_MODE=cublas-pedantic` |
| `set_fast_gemm(true)` / `MAMBA_RS_FAST_GEMM=1` | `GemmMode::CublasFast` / `MAMBA_RS_GEMM_MODE=cublas-fast` |
| `disable_tf32()` | `set_gemm_mode(GemmMode::CublasPedantic)` |
| `set_bi_gemm_family(...)` / `MAMBA_RS_BI_GEMM_FAMILY` | nothing; the role picks the family |
| `set_bi_tensor_cores(false)` / `MAMBA_RS_BI_TENSOR_CORES=0` | no replacement; the scalar half kernels are not selectable |
| `set_half_triad_policy(TiledParity)` / `MAMBA_RS_BI_HALF_POLICY=tiled` | no replacement; the stream-K weight gradient is the route |
| `MAMBA_RS_ARCH_RUNG=off` | no replacement; the rung's self-check decides |
| `gemm_flags()`, `batch_invariant()`, `fast_gemm()`, `tf32()`, `bi_gemm_family()`, `gemm_policy()` | `gemm_mode()`, the model's or trainer's `dtype()`, and `gemm_route()` for the complete identity |

### The measured kernels run on every SM80-tier card

The six specialized GEMM modules were built only for `("sm_89", (8, 9))`
with 142 SMs. Every kernel in them uses the SM80 instruction tier
(`mma.sync`, `cp.async`, `ldmatrix`, `cvt.rna.tf32`, scalar FMA), so the
restriction was a rule, not a hardware limit. Each module now compiles for
the device's own target on any SM80-tier board. The inference overlay
composes on every such target except CC 12.x, where the board's own
kernels serve and the module must stay byte-identical to its frozen
cohorts. The board table follows the CUDA 13.4 docs: CC 10.7 (`sm_107a`)
joins the SM100 family beside 10.0, 10.3 and 11.0.

How a route is admitted:

- A board with a frozen cohort for the route (the RTX 6000 Ada, on CUDA
  12.8, 13.0 and 13.2) takes it as before. The cohort pins the toolkit,
  the composed source, the compiled artifact and the board.
- Any other board proves the route at first use. The candidate and the
  reference of the same numeric contract run on the same operands into
  scratch outputs and every output word is compared. Equal words admit
  the candidate for the rest of the process; a difference declines it
  once, with the reason printed, and the reference serves from then on.
  A proof never runs inside a graph capture (the reference serves there),
  and the proof's own launches stay out of the eager route recording, so
  the manifest a trainer builds from its first eager step matches the
  capture that follows. The A100 gate lane caught the manifest carrying
  the proof launches (`tc_mixed_training_is_bit_identical_across_runs`).
- Before any proof the board counts waves. A candidate tile that takes
  strictly more waves on this board than the reference while computing a
  tile no smaller is declined outright; that is how a tile tuned for one
  SM count spills on another. Half tiles are held against the tiled 64x64
  kernel, the specialized TF32 tiles against the portable tile they
  replace.

A proof says nothing about speed. A route admitted on a board without a
cohort carries the speed evidence of the board it was measured on, and
the route identity records which kind of admission it holds.

`WeightDtype::Tf32` off the Ada used to fall back to exact f32 because the
TF32 policy had no measured route there. It now serves the portable
deterministic TF32 tier; exact f32 still serves any shape the tier does
not cover. The tier's bits are its own, as the precision documents.

The portable tier also reads the census by neighbourhood. A contiguous
shape within a factor of four of a measured shape on every dimension,
staged the same way (16-byte loads only when the leading dimension is a
multiple of four floats), takes the nearest measured shape's tile, and a
wide tile drops to 64x64 when the shape would not fill the board. Held
out one at a time, the measured cells reproduce their own tiles from
their neighbours 33 of 38 times at a factor of four, 12 of 12 at two and
37 of 54 at eight, so the band stops at four. Every route the band names
still passes the bit proof and the wave count. The dispatch epoch
`gemm_route()` reports is 46. The band serves the Ada too: a
shape with a cell keeps it, a shape without one takes its neighbour's
tile instead of the exact f32 kernel. On the Ada that is the Mamba-3
input projection, 10400 x 384 x 1716, whose three GEMMs ran scalar in
0.7.1 and run on the portable TF32 tiles now; the Mamba-1 projections all
have cells and do not move.

A board with its own module keeps its own table first. The RTX 5090
(SM120), and the SM90a and SM100 boards when they come, load the common
modules beside their own module; the loader used to refuse that pairing
and the artifact set had one slot for both. The dispatcher reads the
board's own cohort first: a shape it names runs the board's kernel, any
other shape takes the common tier through the proof. The SM90a WGMMA
module loads on CUDA 13.3 and newer (the CUDA 13.3 release notes on
`wgmma.wait_group`).

Fixed on the Ada: NVRTC 13.2 compiles one inference cell two ways from
the same source (the register allocation differs, the arithmetic does
not), and a process that got the other build lost the inference
copy-plan routes with it. The measured inference cells now compile as a
module of their own, so that variance no longer touches anything else.

### The fastest measured route in every family

Every route below reproduces the one it replaces bit for bit. Each
candidate was compared word for word against the incumbent on three
corpora (uniform full-mantissa values, log-uniform mixed exponents, and
Inf, NaN, -0 and denormals), eager and through a captured graph, before
and after timing. Graph-replay medians on an RTX 6000 Ada at CUDA 13.2,
candidate over incumbent.

| family | cell (m, k, n) | route | before | after | ratio |
|---|---|---|---:|---:|---:|
| triad exact f32 | d128 in_proj dW | `tn_sm89_f32_d128_in_m16n16_g8_s2_cg` | 35.15 us | 19.29 us | 0.549 |
| triad exact f32 | d128 out_proj dW | `tn_sm89_f32_d128_out_m16n16_g8_s2_cg` | 27.59 us | 13.01 us | 0.471 |
| triad half | d128 in_proj dW, bf16 | `tn_sm89_half_d128_in_m32n16_bk64_s4_cg` | 18.23 us | 6.83 us | 0.375 |
| triad half | d128 out_proj dW, bf16 | `tn_sm89_half_d128_out_m32n16_bk64_s4_cg` | 12.08 us | 5.82 us | 0.482 |
| triad half | d128 out_proj forward, bf16 | `nn_sm89_m16n64_bk64_s4` | 6.50 us | 4.42 us | 0.685 |
| triad half | d768 out_proj dW, bf16 | `tn_sm89_relay_m64n64_bk64_s3` | 55.05 us | 48.71 us | 0.885 |
| triad TF32 | prism in_proj dX (4621, 384, 1928) | `nt_sm89_tf32_rna_m144n96_w3x4_bk32_s2` | 150.46 us | 100.66 us | 0.669 |
| triad TF32 | 4096 x 3072 x 1536 dX | `nt_sm89_tf32_rowstage_m128n192_w2x4_bk32_s2` | 632.90 us | 507.95 us | 0.803 |
| triad TF32 | 3072 x 1536 x 4096 dW | `tn_sm89_tf32_m192n192_w3x4_bk32_s2` | 637.64 us | 486.32 us | 0.763 |
| triad TF32 | d768 in_proj dW | `tn_sm89_tf32_pre_rna_m96n192_w3x4_bk32_s2` | 158.71 us | 142.70 us | 0.899 |
| triad TF32 | d768 out_proj dW | `tn_sm89_tf32_pre_rna_m96n96_bk32_s3` | 93.18 us | 81.84 us | 0.878 |
| inference | 4621 x 1928 x 384 exact f32 | `nn_sm89_m112n128_bk32_s3_f32` | 299.75 us | 271.23 us | 0.905 |
| inference | 2048 x 768 x 2304 half to f32 | `nn_sm89_m128n144_bk32_s2_f32out_f16` | 99.38 us | 52.65 us | 0.530 |
| inference | 2048 x 768 x 2304 TF32 | `nn_sm89_m64n288_bk16_s2_tf32` | 130.58 us | 97.16 us | 0.744 |
| inference | 2048 x 2304 x 768 TF32, bias on | `nn_sm89_m64n96_bk32_s2_tf32` | 141.05 us | 125.13 us | 0.887 |
| inference | 2048 x 2304 x 768 half | `nn_sm89_m128n96_bk64_s2_vec_bf16` | 60.60 us | 57.51 us | 0.949 |
| inference | 2048 x 768 x 2304 half | `nn_sm89_m128n144_bk32_s2_vec_bf16` | 65.04 us | 62.50 us | 0.961 |

Two of them are schedules rather than tiles. The relay weight gradient
walks a persistent grid of (tile, slab) units; when a tile's chain crosses
a CTA boundary the earlier CTA hands its f32 accumulators to the next one
untouched, so nothing is folded, the result is the tiled result bit for
bit, and a tile count that is not a multiple of the resident CTAs no
longer pays a whole second wave. Like stream-K it is persistent, so it
serves where the request permits a schedule other than one owner CTA per
tile, which every deterministic context permits by default. The pre-RNA
TF32 tiles round the transposed operand once in their own pass, as the
routes they replace do.

### Changed

- Kernel symbols carry no `gemm_bi_` prefix, no `fixed` token and no
  `_v1` stamp any more; the route identity a program reads through
  `gemm_route()` and the kernel names in logs show the new names.
- The test suite runs on any CUDA board: a test written for one board
  skips on the others instead of failing.
- A CUDA 13.4 toolkit builds with `CUDARC_CUDA_VERSION=13030` (cudarc
  0.19.9 lists toolkits up to 13.3); the README says so.

### What each board gets in this release

| board | 0.7.3 |
|---|---|
| RTX 6000 Ada | all the measured kernels; the reference board |
| RTX 5090 | its own kernels plus the shared ones; faster than 0.7.2 on every training row |
| A100 | the shared kernels; TF32 now runs on the tensor cores instead of the exact f32 path; faster than 0.7.2 |
| H100 / H200 | the shared kernels; the Hopper WGMMA kernels come in a later release |
| B200 / B300 | builds and runs on the shared kernels; not measured yet |

### Measurements and verification

RTX 6000 Ada, CUDA 13.2, against the 0.7.1 ledger recorded on the same
board: all 263 keys identical (161 Mamba keys and 102 decode keys), none
moved, none missing. Every test target of the crate is green there (163
targets and the unit tests), the crate builds and lints clean with
`-D warnings` without `cuda` and with `cuda,hf,qualification`, and the
host gates also pass on the CUDA 13.4 toolkit, where the CC 10.7 target
compiles.

The kernel adapter that timed 0.7.1 on this board, run once on the
assembled tree against its own 0.7.1 record and against the 0.7.1 page,
is in [docs/gemm-benchmarks-0.7.3-ada.md](docs/gemm-benchmarks-0.7.3-ada.md).
Over its 21 Triad cells the geometric means of 0.7.1 time over new time
are 1.16x for exact f32, 1.17x for bf16 and 1.16x for f16; over its five
inference cells 1.04x, 1.00x and 0.98x, the last within run-to-run noise
of its one changed shape. TF32 has no adapter record; against the page,
whose cuBLAS arm ran 6 to 12 percent faster than this run's, the TF32
means are 0.99x for Triad and 0.94x for inference before that drift and
1.05x and 1.06x after it. The same run caught the bf16 packed-store tile
of hot_d asking the driver for its pipeline slabs alone while its
epilogue stages the whole f32 tile; the launch now allocates the tile, a
unit test pins every half cell to the larger of the two, and a GPU test
launches every Ada cell on its measured shape with and without a bias.

Whole training steps at the production shape (d_model 384, 24 layers,
B=8, T=1300), graph replay, one process per storage, against the 0.7.1
medians of the benchmark pages; `old/new` above 1 is faster:

| model | precision / policy | 0.7.1 ms/step | 0.7.3 ms/step | old/new |
|---|---|---:|---:|---:|
| Mamba-1 | BF16 | 110.93 | 111.87 | 0.992× |
| Mamba-1 | F16 | 113.16 | 114.01 | 0.993× |
| Mamba-1 | exact F32 | 206.15 | 204.00 | 1.011× |
| Mamba-1 | F32, TF32 permitted | 180.58 | 180.91 | 0.998× |
| Mamba-3 | BF16 | 132.46 | 132.84 | 0.997× |
| Mamba-3 | F16 | 132.92 | 133.35 | 0.997× |
| Mamba-3 | exact F32 | 174.18 | 173.67 | 1.003× |
| Mamba-3 | F32, TF32 permitted | 164.52 | 149.73 | 1.099× |

The Mamba-3 TF32 row is the neighbour band: under the CUDA profiler the
input projection's three GEMMs move from the scalar kernels (1972 ms of
the 0.7.2 process) to the portable TF32 tiles (1215 ms), everything else
the same, the process 9.3 percent shorter. Every other row is within 1
percent of 0.7.1 either way. The Mamba-1 BF16 and F16 rows are the
board's day, not the release: the 0.7.2 tree and this tree, run once more
the same afternoon under the profiler, gave the same graph step time
within 0.1 percent, the same peak memory to the mebibyte and the same
kernel census (the same kernels under their new names, the same launch
counts, per-kernel time within 3 percent, the sum within 0.2 percent).
Peak memory of every process is 82 MiB above the September 14 figures for
the same reason. Text generation from `state-spaces/mamba-130m-hf` (64
tokens, seed 42) is byte-identical between 0.7.2 and 0.7.3 on CPU f32,
GPU f32 and GPU bf16. Eager timings, peaks and fixtures are on the
[Mamba-1](docs/mamba1-benchmarks.md) and [Mamba-3](docs/mamba3-benchmarks.md)
pages; the other published instruments ran once each with their logs
kept.

Two rented boards, both on CUDA 13.2.78, graph replay. The RTX 5090
(CC 12.0) against the 0.7.2 numbers of its own page section; the A100
SXM4 40 GB (CC 8.0) had no saved numbers, so 0.7.2 ran once on the same
board first. On both the bit ledger against 0.7.2 holds on all 263 keys
and generation is byte-identical.

| board | model | precision / policy | 0.7.2 ms/step | 0.7.3 ms/step | old/new |
|---|---|---|---:|---:|---:|
| RTX 5090 | Mamba-1 | BF16 | 72.01 | 68.53 | 1.051× |
| RTX 5090 | Mamba-1 | F16 | 73.79 | 69.62 | 1.060× |
| RTX 5090 | Mamba-1 | exact F32 | 120.90 | 109.26 | 1.107× |
| RTX 5090 | Mamba-1 | F32, TF32 permitted | 116.15 | 111.17 | 1.045× |
| RTX 5090 | Mamba-3 | BF16 | 108.43 | 104.44 | 1.038× |
| RTX 5090 | Mamba-3 | F16 | 108.81 | 104.73 | 1.039× |
| RTX 5090 | Mamba-3 | exact F32 | 113.84 | 102.24 | 1.113× |
| RTX 5090 | Mamba-3 | F32, TF32 permitted | 113.72 | 102.82 | 1.106× |
| A100 | Mamba-1 | BF16 | 152.92 | 149.41 | 1.024× |
| A100 | Mamba-1 | F16 | 155.56 | 150.76 | 1.032× |
| A100 | Mamba-1 | exact F32 | 288.23 | 288.61 | 0.999× |
| A100 | Mamba-1 | F32, TF32 permitted | 288.02 | 229.07 | 1.257× |
| A100 | Mamba-3 | BF16 | 217.66 | 199.32 | 1.092× |
| A100 | Mamba-3 | F16 | 218.76 | 199.54 | 1.096× |
| A100 | Mamba-3 | exact F32 | 258.21 | 258.84 | 0.998× |
| A100 | Mamba-3 | F32, TF32 permitted | 258.83 | 188.51 | 1.373× |

The RTX 5090 keeps its own kernels on every shape they name; its gains
are the other shapes, which fell to the scalar or portable kernels before
and take the common cells now. The A100 had no kernels of its own; its
0.7.2 TF32 rows equal its F32 rows because TF32 fell back to exact f32
there. The first run of the release candidate on the RTX 5090 found two
defects, both fixed here: the loader refused to create a context on any
board with an architecture module, and the SM120 TF32 cohort had been
minted against the Ada box's NVRTC 13.2.51 while the rented boards run
13.2.78. The common tier also declines a route whose kernel the board's
portable module did not compose (CC 12.x leaves the extension kernels
out) instead of failing at launch.

The final tree was run again on all three boards. RTX 5090: contract lane
green on every target, acceptance capture 89 of 89 identical to 0.7.2,
bit ledger 263 of 263, identity trace and graph-capture gate green, TF32
rows 109.85 ms/step (Mamba-1) and 102.62 (Mamba-3); per-cell timings
against cuBLAS on the [RTX 5090 page](docs/gemm-benchmarks-0.7.3-rtx5090.md).
A100: the same set green, TF32 rows 229.06 and 187.74;
[A100 page](docs/gemm-benchmarks-0.7.3-a100.md). RTX 6000 Ada: ledger
against 0.7.1 at 263 of 263, contract lane green with its capture 97 of
97 identical to the release candidate's, identity trace green twice with
the kernel cache off, whole-step rows within noise (Mamba-1 TF32 180.98
vs 180.91 ms/step, BF16 111.95 vs 111.87, Mamba-3 TF32 149.11 vs 149.73).
The 5090 was also run with its own tier left out: eager, the common cells
beat its small-shape routes 2 to 6 times because every SM120 route there
sits on a 15 us per-launch floor; in the graph-replayed training step the
board's own tier wins at d_model 128 (33.70 vs 39.58 ms/step in BF16) as
at 384, so the tables keep the SM120 tier and the eager floor is on the
board's tuning list.

An H100 PCIe (CC 9.0, CUDA 13.2) ran the final tree last: routing
contracts, the graph-capture gate, the invariance matrix and the adapter
green on the first try, on the common tier since the WGMMA module waits
for CUDA 13.3. Its training rows (Mamba-1 BF16 128.07, TF32 181.56;
Mamba-3 BF16 154.42, TF32 160.85 ms/step) and per-cell timings are on the
benchmark pages and the [H100 page](docs/gemm-benchmarks-0.7.3-h100.md).
The release has booted on one board of every module family it claims
except SM100.

## 0.7.2 (2026-09-16)

Two verification instruments that ship with the crate were red on the
v0.7.1 tag, and the RTX 5090 got the same-board comparison 0.7.1 had left
to Ada. No kernel, route or numeric change: the bits of every ledger key,
the generation of a real checkpoint and the consumer build are the ones
0.7.1 shipped.

### Fixed

- The physical-trace ownership contract (`gemm_bi_tf32_contract`) now
  admits `gemm_bi_inference/runtime_bundle.rs`, the launch owner of the
  retained inference routes added in 0.7.1. Its single submission is
  audited under `launch_inference_bundle`, the same way every launcher in
  `gemm_bi_inference.rs` is; three of the contract's tests failed on the
  released tree because the owner was never registered. The same contract
  now audits the half-route observation and the prepared half graph node
  in the helpers the release moved them to,
  `resolve_half_gemm_observation_with_context` and
  `resolve_prepared_half_graph_node_with_context`, plus the node the
  retained small16 test resolves itself; its census error names the scope
  it was checking.
- The Mamba-1 fold qualification expected capacities 32 and 64 to compile
  the legacy Fixed source. Capacity 64 has carried the retained inference
  overlay since the route assembly, and the dispatcher's cohorts already
  bind that identity; the gate now expects the legacy digest at 32 and the
  qualified capacity-64 digest at 64. The raw byte-for-byte comparison of
  the new fold kernels at capacity 16 was never affected.

### Measurements and verification

RTX 6000 Ada, CUDA 13.2, v0.7.0 against v0.7.1, both trees recorded on the
same day: all 161 original Mamba ledger keys identical, none moved, none
missing, plus the 102 decode keys 0.7.1 added; the same three training
digests under CUDA 12.8 and 13.0; text generation from
`state-spaces/mamba-130m-hf` byte-identical between the two tags on CPU
f32, GPU f32 and GPU bf16. Every test target of the crate ran on that
board: 162 targets, 160 green, the two reds being the instruments fixed
above, and the compile gate green outside the runner's thirty-minute slot.
The ignored training instruments ran as well: the fifteen trainer
benchmarks, the Mamba-3 training steps at the multi-chunk and default
shapes, the serve prefill, both decode benchmarks and the GEMM
training-step benchmark. The crate checks without the `cuda` feature in
every feature set and on the 1.97 toolchain, and lints clean with
`-D warnings` in both feature lanes.

RTX 5090 (driver 595.58.03, CUDA 13.2), which 0.7.1 had not measured: the
same 161 ledger keys identical between the two tags, and the whole-step
training comparison of the 0.7.1 tables repeated on this board. 0.7.1 is
performance-neutral on the 5090, within noise on every row: its gains
came from routes retained for Ada, and the changelog of 0.7.1 said as
much. The tables are in [Mamba-1](docs/mamba1-benchmarks.md) and
[Mamba-3](docs/mamba3-benchmarks.md).

## 0.7.1 (2026-09-14)

Performance improvements to the Mamba kernels and deterministic GEMMs.
The GEMM modes and calling conventions stay the same: `Deterministic` is
the default, with `CublasFast` and `CublasPedantic` available explicitly.
The retained GEMM routes are integrated and bit-qualified.

### Performance

Full synthetic backbone training steps on RTX 6000 Ada, CUDA 13.2,
d_model 384, 24 layers, B=8, T=1300. CUDA Graph replay; median of two
process-average measurements per version. `old/new` above 1 means faster.

| model | precision / policy | 0.7.0 ms/step | 0.7.1 ms/step | old/new |
|---|---|---:|---:|---:|
| Mamba-1 | BF16 | 112.20 | 110.93 | 1.011× |
| Mamba-1 | F16 | 114.63 | 113.16 | 1.013× |
| Mamba-1 | exact F32 | 206.08 | 206.15 | 1.000× |
| Mamba-1 | F32, TF32 permitted | 180.86 | 180.58 | 1.002× |
| Mamba-3 | BF16 | 145.51 | 132.46 | 1.099× |
| Mamba-3 | F16 | 145.90 | 132.92 | 1.098× |
| Mamba-3 | exact F32 | 185.46 | 174.18 | 1.065× |
| Mamba-3 | F32, TF32 permitted | 175.90 | 164.52 | 1.069× |

Mamba-1 F32 and TF32 are effectively unchanged between releases. The
Inference half-to-F32 graph rows improve by 1.216× in geometric mean;
Triad BF16/F16 improve by about 1.027×, with the d128 output-projection
weight gradient improving by 1.487×/1.502×. These are measured shapes,
not a claim of that gain for every workload. See the full
[GEMM tables](docs/gemm-benchmarks-0.7.1-ada.md),
[Mamba-1](docs/mamba1-benchmarks.md) and
[Mamba-3](docs/mamba3-benchmarks.md) for eager timings, cuBLAS controls,
process ranges and fixture details. Prefill, decode and RTX 5090 were not
retimed for these tables; their earlier measurements remain historical.

### Changed

- Mamba-3 sequential training uses constant-width state loops and explicit
  launch bounds. Typed sequential burn-in also uses width-specialized
  entries on qualified Ada configurations, preserving the released
  recurrence's multiply/FMA rounding and persistent-state layout.
- Mamba-3 packs short B/C normalization rows and uses retained backward
  staging and reduction kernels on qualified shapes. The reduction order
  is unchanged; other configurations keep their existing entries.
- Mamba-1 uses retained BF16/F16 parallel-backward fold kernels on qualified
  Ada shapes. Selection binds the compiler, shape and available function.
- The Inference and Triad dispatchers select the retained Ada variants for
  14 shape, precision and bias configurations. The additions cover native
  half, half-to-F32, deterministic TF32 and exact F32 paths. Other
  architectures keep their existing routes; this pass does not assign
  Ada's measurements to RTX 5090.

### Added

- `m3_scan_micro_bench` records sequential prefill timing and hashes for
  the output and all three carried states.
- The decode ledger now includes native BF16/F16 eager and graph paths,
  per-step output and final persistent-state checks.

### Fixed

- Refresh the CLI's transitive `chacha20` lock entry to 0.10.2. The
  previous entry was yanked for an SSE2-backend soundness bug; this
  dependency is used by the model downloader's retry logic, not the
  Mamba or GEMM kernels.
- The Mamba-3 F16 training benchmark waits for loss-scale calibration
  before timing. Timed steps still report skipped optimizer updates;
  measurements with a skip are rejected. The production trainer and
  its loss-scaler settings are unchanged.

### Measurements and verification

The final assembled source `83079104` matches the released 0.7.0 source
on all 161 original normalized Mamba ledger keys, with no changed or
missing keys, on RTX 6000 Ada and CUDA 13.2. The separate decode inventory
matches all 48 cells, including the 22 cells added to the original 26.
These are different inventories, not a combined count.

Combined Inference/Triad acceptance checks 99 logical cases under CUDA
12.8, 13.0 and 13.2, at state capacities 16 and 64: 594 case/toolkit/capacity
combinations per version. Each configuration records finite and exceptional
inputs over six independent runs. All recorded output words match 0.7.0;
the launch census reaches 35 physical functions per toolkit/capacity pair.
This is same-board correctness evidence, not cross-architecture bit identity
or a performance measurement.

## 0.7.0 (2026-09-12)

**A big performance release.** The deterministic GEMM kernels that
mamba-rs uses for training and for serving were rewritten, and they are
now the default: every GPU context multiplies with the crate's own
kernels unless you ask for cuBLAS by name. Every Mamba kernel around them
(the scans, the convolution, the norms, the reductions, the decode steps)
then went through a pass of its own, and two routes moved to their faster
family. On an RTX 6000 Ada a training step of the release shapes is 1.13
to 1.63 times faster than 0.6.9, the Mamba-3 production step 1.42 to
1.59 times faster than the tree before the Mamba pass, and the decode
steps 1.2 to 1.3 times faster; the numbers are summarised below and given in full,
kernel by kernel, in [docs/determinism-benchmarks.md](docs/determinism-benchmarks.md).
Model weights, checkpoints and the CPU paths are unchanged. GPU results
change bit for bit compared with 0.6.9 because different kernels and, on
short sequences, a different scan route now compute them;
`GemmMode::CublasPedantic` together with `ScanMode::Sequential` reproduces
the 0.6.9 numbers.

### Highlights

- **Deterministic by default, with three modes.** A `GemmMode` on every GPU
  context chooses who multiplies: `Deterministic` (the default, the crate's
  kernels, never cuBLAS), `CublasFast` (cuBLAS with TF32 permitted) or
  `CublasPedantic` (cuBLAS with full-f32 accumulation, the 0.6.9 default).
  In the deterministic mode the same inputs give the same bits run after
  run, from an eager launch and from a captured graph, and for serving the
  same bits for a row at any batch size. See [docs/gemm-modes.md](docs/gemm-modes.md).
- **New training kernels.** The Triad family (`kernels/gemm_bi_triad/`, the
  forward, weight-gradient and input-gradient products) went from one
  6655-line file to 25 files of per-architecture kernels: deterministic
  TF32, which 0.6.9 did not have at all; measured Ada kernels for bf16 and
  f16, exact f32 and TF32; RTX 5090 kernels with 60 measured tiled entries,
  12 stream-K entries, TMA-fed exact-f32 kernels and a TF32 set qualified on
  CUDA 12.8, 13.0 and 13.2; and Hopper and datacenter-Blackwell kernels that
  bind behind a self-check but have not been timed on a board.
- **New inference kernels.** The forward-only family used by model and LM
  contexts, now called `Inference` (`kernels/gemm_bi_inference/`, formerly
  `Fixed`), grew from 8 files to 21 with 86 kernels: measured Ada kernels for
  half precision, exact f32 and deterministic TF32, and RTX 5090 TMA
  kernels. It is batch-invariant by construction and it is now the default
  family of every model context; in 0.6.9 it was an opt-in that no dispatch
  path reached.
- **The Mamba kernels, faster with the same bits.** After the GEMM work the
  scans, the convolution, the norms, the column reductions and the decode
  steps of both models went through a pass of their own, every change held
  to bit identity with 0.6.9 under digest suites run on both trees (161
  recorded hashes, all matching). The Mamba-1 fold backward, the largest
  kernel of the training step, runs 18 percent faster in bf16 and 26
  percent faster in f32; the Mamba-3 chunked backward's pair kernel 2.2
  times faster; the column sums over the batch 2.7 to 3.5 times faster;
  the Mamba-1 f32 decode step and the Mamba-3 decode step on fewer
  kernels per layer. The whole-step numbers are in the Performance
  section below.
- **Measured, not assumed.** Every automatic kernel selection on the two
  boards was timed cell by cell against cuBLAS Fast and cuBLAS Pedantic,
  with the winners frozen together with the board, the toolkit and the
  compiled artifact. A kernel outside its frozen identity is not selected;
  the portable kernel serves instead and says so once. The driver build is
  not part of that identity: a box on any driver that loads the artifact
  gets the same kernels, because the bits come from the compiler and the
  source, not from the driver.

### Performance

Speedups are cuBLAS time divided by mamba-rs time, or 0.6.9 time divided
by 0.7.0 time; above 1.0 the 0.7.0 kernel is faster. Exact f32 is compared
with cuBLAS Pedantic, which does the same arithmetic; bf16 and f16 with
cuBLAS Fast on the native half tensor-core kernels; deterministic TF32 with
cuBLAS Fast TF32.

**Inference family against cuBLAS** (five serving shapes, bias off and on,
geometric mean, eager path):

| input → output | compared with | RTX 6000 Ada | RTX 5090 |
|---|---|---:|---:|
| BF16 → BF16 | Fast | 1.19× | 1.24× |
| F16 → F16 | Fast | 1.17× | 1.23× |
| BF16 → F32 | Fast | 0.83× | 1.29× |
| F32 deterministic TF32 | Fast TF32 | 0.90× | 1.13× |
| F32 exact | Pedantic | 1.00× | 1.08× |

**Triad family against cuBLAS** (training products, geometric mean over the
large shapes, eager path; the small d128 shapes are launch-bound and
slower than cuBLAS on both boards):

| precision | compared with | RTX 6000 Ada | RTX 5090 |
|---|---|---:|---:|
| BF16 | Fast | 1.07× | 1.06× |
| F16 | Fast | 1.08× | 1.06× |
| F32 deterministic TF32 | Fast TF32 | 0.84× | 1.07× |
| F32 exact | Pedantic | 0.94× | 1.17× |

**The new kernels against the 0.6.9 kernels** (RTX 6000 Ada, CUDA 13.2,
the same program compiled against both trees, each tree's deterministic
kernel timed in the same process as its cuBLAS arms; geometric mean of the
speedup over the shapes of each class, and the range):

| family | precision | large shapes | small d128 shapes |
|---|---|---:|---:|
| Triad (training) | BF16 | 1.31× (1.01 to 1.63) | 1.08× |
| Triad (training) | F16 | 1.26× (1.02 to 1.42) | 1.08× |
| Triad (training) | F32 exact | 1.43× (1.05 to 2.29) | 1.25× (up to 2.98) |
| Inference (serving) | BF16 | 1.32× (1.26 to 1.44) | |
| Inference (serving) | F16 | 1.34× (1.26 to 1.46) | |
| Inference (serving) | F32 exact | 1.37× (1.20 to 1.46) | |

The largest single gains are the exact-f32 weight-gradient kernels on the
d128 shapes (2.0× and 3.0×) and the exact-f32 input-gradient kernel on the
d768 in_proj shape (2.3×); no cell is behind 0.6.9, the smallest gain on a
large shape being the deep 4096-row exact-f32 weight gradient at 1.05×. The
deterministic TF32 kernels have no 0.6.9 counterpart; against the only f32
answer 0.6.9 had, the exact kernels, they are 2.3 to 3.1 times faster on
the large shapes, at TF32 precision. The complete per-kernel tables, the
cuBLAS comparisons for both boards and the measurement protocol are in
[docs/determinism-benchmarks.md](docs/determinism-benchmarks.md).

**Whole training step, 0.6.9 against 0.7.0** (RTX 6000 Ada, CUDA 13.2,
`MambaTrainer` with graph replay, same shapes and settings in both trees,
median of four alternating runs, milliseconds per step):

| model | precision | 0.6.9 deterministic | 0.7.0 deterministic | speedup | cuBLAS Fast | cuBLAS Pedantic |
|---|---|---:|---:|---:|---:|---:|
| d128, 2 layers, B=16, T=64 | f32 | 3.03 | 2.47 | 1.22× | 2.12 | 2.19 |
| d128, 2 layers, B=16, T=64 | bf16 (tensor cores) | 2.35 | 1.94 | 1.21× | 1.85 | 2.35 |
| d256, 4 layers, B=16, T=128 | f32 | 12.45 | 10.99 | 1.13× | 8.86 | 9.38 |
| d256, 4 layers, B=16, T=128 | bf16 (tensor cores) | 9.54 | 7.66 | 1.25× | 7.21 | 9.43 |
| d768, 4 layers, B=8, T=256 | f32 | 34.81 | 24.57 | 1.42× | 16.93 | 20.52 |
| d768, 4 layers, B=8, T=256 | bf16 (tensor cores) | 22.17 | 13.60 | 1.63× | 13.37 | 19.99 |
| d1536, 2 layers, B=4, T=256 | f32 | 24.49 | 19.09 | 1.28× | 11.07 | 14.86 |
| d1536, 2 layers, B=4, T=256 | bf16 (tensor cores) | 13.07 | 9.45 | 1.38× | 9.04 | 14.49 |

The whole-step gain, 1.13 to 1.63×, is the GEMM kernels, the Mamba kernel
pass and the two route changes below together (the second table below
separates the pass). The cuBLAS columns moved too, since those arms share
every kernel but the products. The deterministic bf16 training step is
now within 2 percent of cuBLAS Fast on the d768 model and 1.5 times faster
than cuBLAS Pedantic there; f16 behaves like bf16. Opting the f32 step
into deterministic TF32 (`MAMBA_RS_BI_F32_POLICY=tf32`) keeps its gain on
the d768 model; on the other three shapes no measured TF32 kernel exists
yet and the exact kernels serve, so the time does not change.

**The Mamba kernels** (everything around the GEMMs) went through their own
pass after the GEMM work, at the production classifier shape (d384, 24
layers, B8, T1300) and the release shapes, on RTX 6000 Ada with CUDA 13.2.
Every change keeps the outputs bit-identical to the previous kernels: the
digest suites of both models (decode steps, scan forwards and backwards,
convolution, chunked backward, prefill, whole training runs) were recorded
on both trees and match line by line, 161 hashes. Per kernel, per launch:

| kernel | before | after | what changed |
|---|---:|---:|---|
| Mamba-1 fold backward, bf16 | 1.75 ms (2.08 mid-pass) | 1.68 ms | accumulators in registers, one block scan after the other, the warp hand-offs on the scans' own barriers |
| Mamba-1 fold backward, f32 | 3.76 ms (mid-pass) | 2.79 ms | three lanes staged in shared memory, two blocks per SM |
| Mamba-3 chunked backward, pair kernel | 3.41 ms | 1.58 ms | odd tile stride, flat triangle walk |
| column sums over B·T rows (16 / 384 / 768 columns) | 165 / 208 / 215 µs | 61 µs | a staged row tile, the owning thread keeps the chain |
| Mamba-1 conv backward | 286 µs | 245 µs | one kernel per tile, no pre-activation tape |
| Mamba-1 B/C reduction | 222 µs | 206 µs | |
| Mamba-1 conv burn-in (training forward) | 98 µs | 87 µs | |

The Mamba-1 f32 decode step runs on the same seven fused kernels per
layer as the half-precision step, and the Mamba-3 decode step on nine
kernels per layer instead of eleven.

**Two routes moved to the faster family** on the owner's word that the
new release need not reproduce 0.6.9's bits on the GPU. The half-precision
weight gradient takes the stream-K kernel on every reduction of 2048 rows
or more, with its persistent grid bounded by the CTAs the board keeps
resident (two per SM) instead of one per SM: tiled to stream-K on Ada, the
classifier page in_proj 79.9 → 65.5 µs and its out_proj 63.8 → 36.9, the
production training in_proj (10400 rows) 162 → 98, input_proj 135 → 48
and out_proj 135 → 62, the d768 out_proj 66.6 → 44.5; the d128 and thin
shapes, where a CTA would hold one or two units, lose and stay tiled. And
a Mamba-1 sequence of 65 to 256 steps trains on the parallel scan route:
the automatic threshold sat at 256 from before the parallel backward
existed, and every release shape of the training tables was on the
sequential kernel. Sequential against parallel, whole step with graph
replay: d768 B8 T256 18.8 → 13.6 ms in bf16 and 30.6 → 24.6 in f32, d1536
B4 T256 12.2 → 9.6 ms, d256 B16 T128 the same, d128 B16 T64 stays
sequential (1.89 against 2.04). Both routes are bit-identical run to run;
`tiled` and `ScanMode::Sequential` restore the previous families.

**Deterministic TF32 on Ada moved toward cuBLAS Fast TF32** in the last
pass of the release, since the next project's RL training will run on it.
The NT kernel of the joint Ada module was rebuilt: in an NT product both
operands are contiguous along the reduction, so B is loaded through
`ldmatrix` exactly as A is (one x4 for two atoms, one x2 for the third),
every operand word is rounded by the half-ulp add instead of `cvt.rna`,
the copy plan and the fragment addresses are computed once per thread, the
fragments are double-buffered with the copies spread over the four k8
issues, there is one barrier per K-tile, and the tile is stored through
shared memory in float4 rows. Its finite results are bit-identical to the
previous body (the same ascending k8 chain per element). Five cells moved
to kernels the tables had never measured them against. Against cuBLAS
Fast TF32, the same decisions on CUDA 12.8, 13.0 and 13.2: NT d768 in_proj
0.63 → 1.00×, NT d768 out_proj 0.71 → 1.05×, TN d128 out_proj 0.47 → 0.89×
(the fused split-K8 kernel, eight partials folded in a fixed order), TN
d768 in_proj 0.76 → 0.83× (the M64N96 tile), TN large_deep 0.53 → 0.77×,
NT large_deep 0.64 → 0.73×, TN d128 in_proj 0.49 → 0.58×. Over the 21 Ada
TF32 cells the geometric mean goes from 0.72× to 0.80×, on the twelve
large shapes from 0.74× to 0.84×. Three cells did not move: NN large_deep
(the joint kernels lose to the portable one there), TN d768 out_proj (a
96 × 96 tile is missing) and NT classifier page (the new body does not
win). The split-K8 cells carry a reduction order of their own; every
route stays deterministic run to run.

**Whole steps, 0.7.0 before and after the Mamba kernel pass** (RTX 6000
Ada, CUDA 13.2, the same programs, mirrored runs of separate processes,
median of four; the before tree is the one the tables above were taken
on):

| model, shape, path | before | after | speedup |
|---|---:|---:|---:|
| Mamba-1 training, d384 24 layers B8 T1300, bf16 tensor cores, graph | 118.13 ms | 112.38 ms | 1.05× |
| Mamba-1 training, d384 24 layers B8 T1300, f32, graph | 227.41 ms | 204.61 ms | 1.11× |
| Mamba-1 training, d768 4 layers B8 T256, bf16 tensor cores, graph | 20.43 ms | 13.60 ms | 1.50× |
| Mamba-1 training, d768 4 layers B8 T256, f32, graph | 32.10 ms | 24.56 ms | 1.31× |
| Mamba-1 training, d1536 2 layers B4 T256, bf16 tensor cores, graph | 12.93 ms | 9.48 ms | 1.36× |
| Mamba-1 training, d1536 2 layers B4 T256, f32, graph | 22.70 ms | 19.05 ms | 1.19× |
| Mamba-1 training, d256 4 layers B16 T128, bf16 tensor cores, graph | 8.57 ms | 7.64 ms | 1.12× |
| Mamba-1 training, d256 4 layers B16 T128, f32, graph | 12.09 ms | 10.97 ms | 1.10× |
| Mamba-1 training, d128 2 layers B16 T64, bf16 tensor cores, graph | 2.11 ms | 1.94 ms | 1.09× |
| Mamba-1 training, d128 2 layers B16 T64, f32, graph | 2.80 ms | 2.48 ms | 1.13× |
| Mamba-3 training, d384 24 layers B1 T256, bf16, graph | 17.25 ms | 14.68 ms | 1.18× |
| Mamba-3 training, d384 24 layers B1 T256, f32, graph | 18.31 ms | 15.66 ms | 1.17× |
| Mamba-3 training, d384 24 layers B8 T1300, bf16, graph | 231.21 ms | 145.47 ms | 1.59× |
| Mamba-3 training, d384 24 layers B8 T1300, f32, graph | 262.14 ms | 184.93 ms | 1.42× |
| Mamba-3 prefill, T4621 24 layers d384, f32 | 19.14 ms | 19.05 ms | 1.00× |
| Mamba-1 decode, d128 3 layers, f32 exact, Triad family, batch 1, graph | 103.98 µs | 79.02 µs | 1.32× |
| Mamba-1 decode, d128 3 layers, Inference family f32, batch 1, graph | 197.05 µs | 170.53 µs | 1.16× |
| Mamba-1 decode, d128 3 layers, Inference family bf16, batch 1, graph | 97.38 µs | 83.44 µs | 1.17× |
| Mamba-1 training forward, d128 3 layers, B1 T32, eager | 786 µs | 682 µs | 1.15× |
| Mamba-3 decode, d128 4 layers, batch 1, graph | 169.8 µs | 160.6 µs | 1.06× |

The Mamba-3 training step gains the most: its chunked backward's pair
kernel was a third of the step, the column sums a seventh, and its weight
gradients now run on the stream-K kernel. The Mamba-1 bf16 production
step gains five percent end to end: the first batch of the pass had cost
its fold backward a fifth (1.75 to 2.08 ms per launch, the vector row
loads), the register rewrite took it to 1.68, and the stream-K weight
gradient took the rest; the f32 step gains eleven percent. The release
shapes at T=256 gain a third to a half from the parallel scan route. The
per-kernel rows above take their before column from the tree the ledger
was recorded on, marked where that is the mid-pass tree.

**Whole inference step, 0.6.9 against 0.7.0** (RTX 6000 Ada, the
`MambaConfig::default()` model with d_model 128 and 3 layers, one decode
step, microseconds; median of four alternating runs of each tree). The
GEMM kernels alone left this step where 0.6.9 had it, within 3 percent
from batch 1 to batch 64 and 12 to 14 percent faster at batch 128 in
bf16: a model this small spends its step on kernel launches, not on
arithmetic. The Mamba kernel pass then cut the launches: the Mamba-1
decode step runs seven fused kernels per layer instead of eleven, and
against 0.6.9 it is 1.17× faster in f32 and 1.13× in bf16 at batch 1 with
the inference family (199.6 → 170.7 µs and 93.9 → 83.3 µs, graph), 1.30×
with the exact-f32 Triad family (102.8 → 79.0 µs), and 1.18 to 1.25× at
batch 128. The outputs of the trees are bit-identical at every batch and
step; the full tables are in
[docs/determinism-benchmarks.md](docs/determinism-benchmarks.md).

### The GEMM mode API

- `GemmMode { Deterministic, CublasFast, CublasPedantic }`, re-exported as
  `mamba_rs::mamba_ssm::gpu::GemmMode`. `Deterministic` is the default.
- `GpuCtx::new_with_mode`, `GpuCtx::new_with_state_cap_and_mode`,
  `GpuCtx::gemm_mode` and the fallible `GpuCtx::set_gemm_mode`, which is
  refused during a graph capture, restores the previous cuBLAS setting when
  the change fails, and marks the context unusable if a rollback cannot be
  verified.
- Explicit-mode constructors beside every environment-reading one:
  `GpuMambaBackbone::new_with_mode` and `new_with_dtype_and_mode`,
  `GpuMamba3Backbone::new_with_mode` and `new_with_dtype_and_mode`,
  `GpuMambaLM::from_hf_with_mode`, `from_hf_with_dtype_and_mode` and
  `from_hf_with_dtype_batch_and_mode`, `GpuMamba3LM::from_weights_with_mode`
  and `Mamba3LmBuild::build_with_mode`, `MambaTrainer::new_full_with_mode`
  and `Mamba3Trainer::new_full_with_mode`, and `new_with_mode` on the four
  inference engines. `TrainSessionCfg::new(input_dim, batch, seq_len)` gives
  the default optimizer settings of `new_with_dtype`, so the explicit-mode
  path of a trainer stays one call. The plain constructors read
  `MAMBA_RS_GEMM_MODE` (`deterministic`, `cublas-fast`, `cublas-pedantic`)
  and default to `Deterministic`; the explicit ones ignore the GEMM
  environment.
- Two policies inside the deterministic mode, both new: `F32TriadPolicy`
  (`exact`, the default, or `tf32`, which permits the measured deterministic
  TF32 kernels and stays exact elsewhere) through `set_f32_triad_policy` and
  `MAMBA_RS_BI_F32_POLICY`, and `HalfTriadPolicy` (`streamk`, the default
  with tensor cores on, which takes the stream-K weight-gradient kernel on
  every reduction deep enough to pay for it, or `tiled`, which reproduces
  the portable tensor-core kernels bit for bit) through
  `set_half_triad_policy` and `MAMBA_RS_BI_HALF_POLICY`.
- `GpuCtx::gemm_route` returns the complete numeric route: mode, family,
  policies, the selected kernels, and the compiler, artifact and device
  identity. Captured graphs record it and every replay checks it; a mode,
  family or policy change after a capture is refused at the next replay.
  `GpuCtx::gemm_flags` remains as the three-flag view for existing callers.
- Raw f32 products, the Mamba-3 projections and the tied LM heads follow the
  context's mode; before this release several of them called cuBLAS
  directly whatever the context said. Tied half-input heads keep f32 logits.

### Breaking changes

- The default GEMM path changed from cuBLAS to the deterministic kernels.
  GPU outputs differ bit for bit from 0.6.9. Construct with
  `GemmMode::CublasPedantic` to get the 0.6.9 numbers, or
  `GemmMode::CublasFast` for the fastest vendor path.
- Two routes inside the deterministic mode changed their numeric family
  where they were faster: the half-precision weight gradient takes the
  stream-K kernel on every reduction of 2048 rows or more (its fixed-order
  fold groups the sum differently from the tiled kernel; `tiled` restores
  the old bits), and a Mamba-1 sequence of 65 to 256 steps trains on the
  parallel scan route instead of the sequential kernel (`ScanMode::
  Sequential` restores it). Both stay bit-identical run to run.
- `BiGemmFamily::Fixed` is `BiGemmFamily::Inference`, the module
  `gemm_bi_fixed` is `gemm_bi_inference`, and `MAMBA_RS_BI_GEMM_FAMILY`
  accepts `triad` and `inference` only; `fixed` is rejected. Frozen kernel
  artifacts keep their recorded identities, so caches stay valid.
- Model and LM contexts now default to the Inference family; trainers and
  plain contexts default to Triad. `GpuCtx::new` no longer reads the
  environment; `GpuCtx::new_from_env` does.
- The older environment variables are resolved together: with neither set
  the mode is `Deterministic`; `MAMBA_RS_BATCH_INVARIANT=1` selects
  `Deterministic`; `MAMBA_RS_FAST_GEMM=1` selects `CublasFast`; an explicit
  `0` on either, with no positive selector, selects `CublasPedantic`. Both
  set to `1`, or either set together with `MAMBA_RS_GEMM_MODE`, is an error.
  A script that set `MAMBA_RS_BATCH_INVARIANT=0` under 0.6.9 therefore
  keeps its numbers; a script that set nothing moves to the deterministic
  kernels.
- The family, tensor-core and policy variables are rejected at construction
  when a cuBLAS mode is selected; they describe deterministic kernels only.
- Low-level graph capture: `graph_capture::capture_into_graph` and the
  `GpuCtx` capture methods that borrow external allocations are `unsafe`,
  because the caller must keep every captured resource alive and
  pointer-stable until the graph is destroyed. The safe trainer and
  inference owners handle this themselves. Public graph holders replay only
  on the context that captured them; raw training graph handles are no
  longer public.
- The `cuda` feature no longer enables cuBLASLt. The
  `cuda-cublaslt-qualification` feature adds it to the vendor-comparison
  harness only; production routing never used it.
- The low-level typed probes carry the `*_typed_native` suffix and may
  return `UNCOVERED`; the full-coverage entries are
  `blas::gemm_bi_forward_typed`, `gemm_bi_backward_dw_typed` and
  `gemm_bi_backward_dx_typed`.

### Deprecated

- `GpuCtx::set_batch_invariant`, `GpuCtx::set_fast_gemm` and
  `GpuCtx::disable_tf32` map onto `set_gemm_mode` and panic when the mode
  change is refused, because their signatures cannot return an error. Use
  `set_gemm_mode`. The read accessors `batch_invariant`, `fast_gemm` and
  `tf32` are not deprecated and now describe the mode.
- `MAMBA_RS_BATCH_INVARIANT` and `MAMBA_RS_FAST_GEMM` are still read but
  conflict with `MAMBA_RS_GEMM_MODE`; use the new variable.

### Fixed

- A captured graph re-hashed its whole launch set on every replay to check
  that nothing had changed since the capture. On a small decode step that
  host work cost 45 microseconds per replay, half again the step. The
  digest is now checked once when the plan is built; a replay checks the
  route identity and the policy of each recorded route, and refuses a
  changed route as before.
- Exact f32 kernels keep their `__fmaf_rn` contract on every route; the
  measured SM89 deep-K split-K cells that are faster on the scalar kernels
  stay on them instead of taking the tensor-core contract.
- Misaligned f32 subviews and odd output strides take alignment-safe scalar
  kernels without changing the reduction order.
- A dispatch that could not use a specialised kernel used to decline
  silently, which on any board other than the two measured ones meant the
  scalar kernels served without a word. Every decline now prints once,
  naming the identity field that did not match.
- Cached CUDA artifacts are trusted only with an exact compile identity
  (source, options, target, the literal header closure and the NVRTC
  library pair). Entries with wrong metadata, partial writes, stale
  identities or ambiguous preprocessor dependencies are rebuilt. The cache
  is Linux-only, requires an absolute private path, and is disabled when
  `__DATE__`, `__TIME__` or `__has_include` can reach the compiler. NVRTC
  12.9 and newer receives a stable per-module `--frandom-seed`, so two cold
  compiles produce byte-identical PTX.
- Graph capture: the stream, kernel registry, cuBLAS workspace and
  graph-visible scratch form one immutable resource core retained by
  captured graphs, and a panic inside a capture body ends the capture
  before unwinding. RTX 5090 kernels prepare their tensor maps eagerly and
  fail closed inside a capture instead of allocating there.

### Upgrading from 0.6.9

| you had | you do now |
|---|---|
| the default (cuBLAS) | nothing, to get the deterministic kernels; or pass `GemmMode::CublasPedantic` to keep the 0.6.9 numbers |
| `set_batch_invariant(true)` or `MAMBA_RS_BATCH_INVARIANT=1` | nothing; it is the default. Remove the call when convenient |
| `set_fast_gemm(true)` or `MAMBA_RS_FAST_GEMM=1` | `GemmMode::CublasFast` or `MAMBA_RS_GEMM_MODE=cublas-fast` |
| `MAMBA_RS_BI_GEMM_FAMILY=fixed` | `MAMBA_RS_BI_GEMM_FAMILY=inference`, or nothing for a model context |
| `BiGemmFamily::Fixed` in code | `BiGemmFamily::Inference` |
| a graph captured, then a flag changed | change the mode first, then capture |

### What comes next

0.7.0 is the first of a series. The kernels that surround the GEMMs in a
step, the scan, the convolution and the norms, are the next target and
carry most of a training step; the GEMM kernels keep moving toward cuBLAS
Fast in the 0.7.x releases; and the architectures that run the portable
deterministic kernels today (SM80, SM86, Hopper, the datacenter Blackwell
parts and CC 12.1) get their own measured kernels in later releases.

### Internals and tooling

- The test tree is declared explicitly: 108 regression targets under
  `tests/`, 15 benches under `benches/` with their own `main`, and 46
  hardware and toolkit instruments under `tools/qualification/` behind the
  non-default `qualification` feature. Kernel-candidate experiments left
  the crate for the maintainers' archive. `qual/lanes.toml` gives every
  target a lane and a host test keeps the manifest and the lanes in step.
- The qualification runs behind the numbers on this page, with their raw
  records, verification scripts and device identities, are kept in the
  maintainers' measurement archive, one packet per run.

## 0.6.9 (2026-08-26)

### Fixed

- Build and release-pipeline fixes. No functional change: every route
  returns bit-identical output to 0.6.8.

## 0.6.8 (2026-08-26)

Batched prefill pooling, kernel-file naming, and a correctness gate for
the fixed-tile family. No numeric change: every existing route returns
bit-identical output.

### Added

- `Mamba3PrefillOutputs::pooled_sum` accepts `batch > 1` and receives
  `[B * d_model]`. Each sample is summed over its own `seq_len` rows in
  the ascending-t f32 order the single-sample path already used, so a
  sample's pooled row is bit-identical whether it rode alone or inside a
  batch. Previously the pooled route refused any batch above one, which
  left a batched prefill with only the full-temporal download.
- `gemm_bi_fixed_correctness`: the fixed-tile family against a CPU
  reference across tile tails. The family had no direct test while it
  sat off every dispatch path.

### Changed

- Kernel files carry the names the API uses: `gemm_bi.cu` ->
  `gemm_bi_triad.cu`, `gemm_batch_invariant.cu` -> `gemm_bi_fixed.cu`,
  and the Rust module `gpu::gemm_bi` -> `gpu::gemm_bi_triad`. The
  `sgemm` prefix was BLAS notation for single precision and had not
  described the coverage since the typed and Tensor-Core sections
  landed.

### Fixed

- Documentation fixes.

## 0.6.7 (2026-08-26)

Makes the second batch-invariant GEMM family reachable and puts the
choice in the public API. No default behaviour change: the default
family is the one that already served, so forward and backward outputs
are bit-identical to 0.6.6 unless a caller selects otherwise.

### Added

- `GpuCtx::set_bi_gemm_family` / `bi_gemm_family` and
  `MAMBA_RS_BI_GEMM_FAMILY=triad|fixed`: which batch-invariant family
  serves the forward while `batch_invariant` is on. `Triad`
  (`kernels/gemm_bi_triad.cu`, the default) is the multi-tile dispatcher -
  it carries all three operand layouts, so it is the only family that
  can serve a backward, and its invariance holds across every M inside
  one dispatch bucket. `Fixed` (`kernels/gemm_bi_fixed.cu`) is
  one 64x64x32 tile with `SPLIT_K=1`, forward-only, batch-invariant by
  construction: the K-reduction for `C[i,j]` reads only `A[i,:]` and
  `B[:,j]`, so no bucket boundary exists to cross. The fixed-tile
  kernels were compiled and registered but had no dispatch entry -
  every flag combination reached the triad or cuBLAS.
- `GpuCtx::gemm_route`: the full numeric-route identity (the three tier
  flags plus the family). CUDA-graph capture guards in both backbones,
  both prefills and the split forward/backward cycle now compare this
  instead of `gemm_flags`, so a family flip after a capture is refused
  at replay exactly like a tier flip. `gemm_flags` keeps its shape and
  meaning for callers that assert on the tier flags alone.
- `blas::gemm_bi_forward_raw`: a direct entry to the fixed-tile family
  for benchmarks and for callers that select it explicitly.

### Fixed

- The README stated that the M3 engine stays on cuBLAS regardless of
  the batch-invariant flag. That has not been true since the M3 GEMMs
  moved onto the context-carrying dispatcher: the M3 prefill and
  training forward both pass `GpuCtx` and both follow the flag. The
  scope note now names what actually opts out - the tied LM heads and
  the no-context `*_blas` twins, which take no context.
- The `gemm_bi_triad.cu` header described the file as f32-first. The file
  has carried f32, bf16 and f16, on CUDA cores and Tensor Cores, since
  the typed and TC sections landed; the `S` in the name is historical
  BLAS notation and no longer describes the coverage.

### Measurements and verification

At a vision-classifier prefill shape (f32, M = 4621 rows per page,
RTX 6000 Ada) the two families run within a few percent of each other,
`fixed` ahead on the narrow-N projections and `triad` on the wide-N one;
both differ from cuBLAS by 1.0e-4 - 1.8e-4 and agree more closely with
each other than either does with cuBLAS. Reruns are bit-identical in
both. The full cuda suite passes.

## 0.6.6 (2026-08-25)

Training-stability release for small-batch Mamba-3 runs. No math or
checkpoint-format changes: forward and backward outputs are bit-identical
to 0.6.5; the additions are optimizer-tail policies, off by default.

### Added

- `BackwardOpts::step_skip_above`: on the applying call, a window whose
  pre-clip global gradient norm exceeds the threshold is discarded whole -
  no Adam advance, no weight update, the arena re-zeroes on the next
  backward. Small-batch Mamba-3 training meets rare inputs whose
  sequence-length-scaled dt-route gradients detonate (norms three orders
  of magnitude above baseline); a huge-batch recipe averages such windows
  away, a batch-2 recipe must be allowed to refuse them. Honored by the
  f32 and mixed lanes of both backbones and the single-world dist path;
  f16 keeps its own loss-scaler protocol.
- `BackwardOpts::control_clip_max_norm` plus `clip_region_device`: an
  optional separate clip for the Mamba-3 CONTROL channels (the
  dd_dt/dd_A/trap/angle columns of every layer's in_proj gradient and
  dt_bias) ahead of the global clip. Those columns carry the only
  gradients that scale with the sequence length; without their own bound
  one resonant input rescales the entire arena through the global clip
  and starves the representational columns. The region fold rides the
  same fixed-grid f64-partial ordered reduction as the global clip and
  is covered by a unit that pins the region norm, the scaled region and
  the untouched complement.
- `Mamba3Trainer::apply_step_full` exposes both policies on the split
  apply path used by gradient reducers.

### Measurements and verification

At the production classifier shape (d_model 384, 24 layers, bf16,
T 4621, batch 2 x accum 8) cold Mamba-3 window gradient norms measure
27-237 (median 78) against a Mamba-1 baseline near 1; a clip bound of
1.0 rescaled every step by that norm and froze learning at the class
prior, while a bound calibrated to the measured scale trains: val
soft-CE 2.48 -> 1.86 over a 20-epoch probe with zero discarded
windows. The full suite passes with the new region-clip unit; all
prior digests are unchanged (the policies default off).

## 0.6.5 (2026-08-24)

Correctness release for the Mamba-3 mixed-precision training lane.
Mamba-1 and the Mamba-3 f32 and serving lanes are unchanged bit for
bit; only bf16/f16 Mamba-3 training outputs change, and they change
because they were wrong.

### Fixed

- The Mamba-3 bf16/f16 training forward returned the last layer's raw
  residual stream instead of the post-norm_f output: the computed final
  RMSNorm went into an unread scratch buffer while the backward applied
  the norm's VJP unconditionally, so the forward and backward described
  two different networks and a classifier head consumed an unnormalized
  deep residual. Training on this lane could not converge (a from-cold
  classifier collapsed into constant-class predictions; the overfit
  probe could not reach the loss floor). The mixed lane now mirrors the
  f32 lane - the last layer lands in the pre-norm save and the f32
  rmsnorm writes the caller's output buffer. The mixed-vs-f32 parity
  test compared the pre-norm surfaces on both sides and could not see
  the defect; it now compares the post-norm output, and a new
  regression pins the per-row RMS of the mixed output at ~1.
- Mamba-3 weight init follows the shipped reference: Linear projections
  use the nn.Linear default bound 1/sqrt(fan_in) (the previous gain-1
  bound carried 3x the reference variance) and out_proj additionally
  divides by sqrt(n_layers), the GPT-2 prenorm residual rescale from
  the official mixer.
- The Mamba-3 reference no-decay group gains every bias (the input
  projection bias and the all-ones B/C biases), matching the reference
  parameter grouping; decayed B/C biases walk out of the positive
  regime the Mamba-3 ablation requires.

### Measurements and verification

The from-cold overfit probe (16 pages, d384x24 bf16) now drives the
loss to the entropy floor exactly (gap 0.0000; it previously stalled
with a broken-wire gap of 17.3), and the Mamba-3 parity suite passes
on the corrected post-norm surface. Mamba-1 digests and the Mamba-3
f32/serving lanes are bit-identical to 0.6.4.

## 0.6.4 (2026-08-24)

Inference performance release: a faster prefill serving chain, a
Mamba-3 serving surface, and a training-step pass over both
backbones. Checkpoint formats and the public API are unchanged;
training and inference outputs match 0.6.3 on every recorded
determinism baseline. Measurements are at the end of this entry.

### Added

- Mamba-3 serving surface: `Mamba3Prefill::run_full` emits the all-T
  post-`norm_f` temporal and an on-device pooled column sum
  (ascending-t f32 adds; divide by T on the host reproduces a CPU mean
  pool bit for bit while the per-page download drops to 1.5 KB), and
  `Mamba3PrefillPooledGraph` replays the whole pooled window - state
  reset included - as one CUDA graph over fixed buffers. A new parity
  test pins the surface bitwise against the trainer forward temporal.
- Bit gates for the inference lane: a 16-cell prefill serve hash suite
  (three GEMM tiers, cold and carried conv state), a decode run digest,
  and a per-kernel scan-backward hash set. Recorded once on 0.6.3;
  any kernel edit that moves an inference bit now fails in seconds at
  the exact kernel.
- A state-capacity invariance arm for the sequential/step kernel
  family: builds at `MAMBA_RS_STATE_CAP` 16/64/256 must produce
  identical bits for the same `d_state`.
- Mamba-3 kernels gained the PTX disk cache the Mamba-1 loader already
  had (key: source + arch + options + NVRTC version); the full NVRTC
  compile of seven sources per process boot is now a one-time cost.

### Performance

- The gradient clip computes its coefficient on device: a single-thread
  kernel folds the 512 f64 partials in the ascending order the host loop
  used, and the scaling pass reads the coefficient from device memory -
  the norm/scale pair now launches back to back instead of the GPU
  idling through a sync, a download and a host fold between them. All
  four trainer lanes use it.
- SiLU(gate) is no longer materialized: the split writes x and the raw
  gate, and the gating forward/backward recompute the activation from
  the saved pre-SiLU value in the split kernel's exact form - one
  [B*T*d_inner] activation freed per layer (383 MB at the production
  shape). The in_proj backward's concat pass is gone: the gating
  backward writes the gate half of d_proj and the conv dx pass writes
  the x half, both in place.
- Softplus is fused into the save-scan forwards (the kernels read the
  raw dt, apply softplus through the deleted copy pass's exact store
  rounding, and write the save the backward replays from) and its
  derivative into the fold backward's epilogue (round-first: the
  accumulator rounds exactly as the old store did, then the derivative
  divides the reloaded value) - two launches and one full activation
  read fewer per layer, per direction.
- The fold backward's dA tail writes one partial row per chunk instead
  of a global read-modify-write per (chunk, lane, state); a chunked
  reducer folds the rows in the same order the accumulator did.
- The mixed backward's per-layer f32-to-typed gradient cast survives
  only on the first layer processed: the norm backward mirrors its dx
  store into the typed buffer from the value already in a register.
- The hot elementwise kernels (gating, multiply, softplus) gained
  16-byte vectorized twins - one uint4 per operand per thread, same
  per-element arithmetic in the same order - selected only when the
  shape and every operand pointer allow it (the class measured
  instruction-bound: a uint4 copy of the same bytes runs 2.6x the
  scalar rate on sm_120).
- Every AdamW compute shadow (typed bulk and f32-stays-f32 alike) is
  written by the fused optimizer kernel in the same launch that updates
  its master; the per-tensor copy walk after each step is a no-op seam
  now. Chunks shrink 64k -> 8k elements, growing the grid from hundreds
  of long serial blocks into thousands that fill the machine.


- Prefill conv is T-tiled (the serial per-channel walk left the machine
  idle at B=1) and the `d_conv == 4` register fast path landed on the
  typed nosave and decode step conv kernels.
- The prefill chain drops `split_gate_silu` and the separate gating
  multiply: the conv reads the in_proj output strided and the scan
  fuses the gate into its y store, reproducing the replaced chain's
  per-store roundings exactly.
- The parallel-scan forward packs its shared-memory regions at the
  runtime `d_state` instead of the compile-time maximum; the B/C
  gathers for the scan stage through a padded shared-memory tile and
  write t-contiguous runs; the rmsnorm forwards keep their first
  strided elements in registers between the reduction and the write.
- Mamba-1 scan backward (the production fold): runtime-`d_state` slot
  stride (occupancy 2 -> 3 blocks/SM), packed 8-byte epilogue stores
  and stage-in loads, staged striped dB/dC stores, and fold-depth
  scratch sizing (-383 MB at the production shape).
- Mamba-3 chunk-scan forward runs one head per 128-thread cooperative
  block with a triangle-packed decayed tile and staged operands (the
  old 32-thread block sat behind 32 KB of static shared memory at 5-6%
  occupancy); wider `d_state` shapes keep the original kernel.
- The Mamba-3 forward writes each layer's residual into the next
  layer's slot directly (the per-layer temporal round trip is gone),
  bias-add and RoPE fused into one launch on every lane, the decode
  step merges its two BCNorm launches and drops its per-layer residual
  copy.

### Fixed

- The GEMM-tier environment flags are parsed strictly: an unrecognized
  value (`True`, `ON`, a stray space) now fails construction instead of
  silently meaning "off", and `MAMBA_RS_BI_TENSOR_CORES` without
  `MAMBA_RS_BATCH_INVARIANT` is rejected instead of being a silent
  no-op. `MAMBA_RS_SCAN_TAPE` and the test-world knobs reject typos the
  same way.
- The two cuBLAS determinism guards in the backward GEMM dispatch are
  hard asserts now - as `debug_assert!` they compiled out of every
  release build, exactly where the condition they name would happen
  silently. A mixed-dtype operand triple under the batch-invariant flag
  fails loudly instead of silently taking cuBLAS (no `matvec_bi`
  variant covers it).
- The batch-invariant GEMM scratch buffers are presized before any
  CUDA-graph capture; a first-use allocation on a capturing stream
  became a graph memory node and cached a graph-owned address that
  later eager launches would dereference.
- The PTX cache key includes the resolved CUDA include paths (a box
  with two toolkits could serve stale PTX), a failed cache publish no
  longer leaks its temp file, and NVRTC compile errors print with real
  newlines instead of one escaped blob.
- `nvrtc_arch` maps compute capability (8, 7) to `sm_87` (Jetson AGX
  Orin) instead of falling back to `sm_70`.
- The bf16-only training-graph capture returns an error on a wrong
  dtype instead of panicking inside a `Result`-returning function; the
  safetensors save path documents its length invariant instead of a
  bare unwrap.

### Measurements and verification

Classifier serve page (B=1, T=4621, d_model 384, 24 layers, f32,
cuBLAS+TF32): pooled prefill 29.3 -> 10.8 ms/page, full-temporal
30.1 -> 11.5 ms/page. Production training step (B=8, T=1300, d_model
384, 24 layers, bf16, graph lane): Mamba-1 131.5 -> 114.0 ms/step
(batch-invariant + tensor-core tier) and 169.0 -> 129.8 on cuBLAS,
peak memory down 766 MB; Mamba-3 179.4 -> 165.7 ms/step. Measured on
an RTX 5090; the last training-epilogue items landed after that box
retired and their bit gates re-ran on an RTX 6000 Ada, where the
scalar-tier digests and the decode digest match the 5090 recordings
bit for bit.

Verification: nine run digests (twice each), the 16-cell prefill
serve hash suite, the eleven Mamba-3 gradient hash arms, the decode
digest and the per-kernel scan-backward hashes all equal their
recorded baselines; full test suite 429 passed, 0 failed.

## 0.6.3 (2026-08-23)

Performance release. Production-shape training (d_model 384, 24 layers,
B=8, T=1300, bf16, batch-invariant + tensor-core tier, RTX 5090):
Mamba-1 441.4 -> 131.5 ms/step, Mamba-3 636 -> 179.4 ms/step; the
O(T) h tape is gone (-12.3 GB at that shape, a 4x larger micro-batch
fits). Run-to-run bit determinism holds throughout; reductions that
were deliberately regrouped form one bit family with baselines
recorded below - checkpoints from earlier versions do not resume into
0.6.3 training. Inference kernels and checkpoint formats are
unchanged.

### Fixed

- The parallel reverse-scan backward accumulates its per-chunk `d_a`
  partials into `d_a_log_local` with `+=`; the pre-launch zeroing
  removed in 0.6.2's dead-work pass is restored on the parallel route
  (the sequential kernel keeps its register-accumulator store and needs
  no memset). Stale scratch poisoned the `a_log` gradients with NaN;
  under f16 the loss scaler read the NaN as a permanent overflow,
  halved the scale to 1.0 and skipped every optimizer step.
- Graph-vs-eager parity tests drive the same fused optimizer kernel on
  both lanes and pass the adam lr into the device bias buffer.

### Changed

- Fused multi-tensor AdamW: one descriptor-table kernel
  (`adamw_step_multi_{f32,bf16,f16}`) replaces the per-tensor launch
  walk; the typed compute shadows for the bulk weights are written by
  the same kernel (`FROM_F(new_p)`, identical RNE bits to the old cast
  pass), and `sync_master_to_compute` covers only the f32-stays-f32
  tensors. Chunk plans rebuild on `set_reference_no_decay` /
  `load_optimizer_state`; empty identity-`input_proj` slots produce no
  chunk.
- The learning rate is read device-side from the widened
  `{bc1, bc2, lr}` bias buffer: `set_lr` now applies under a captured
  graph (a warmup/cosine schedule no longer forfeits the graph lane).
- Fused `ssm_reduce_d_BC_{f32,bf16,f16}` replaces the split dB/dC
  reducers with a full-domain `= (0.0f + sum)` store (callers drop
  their pre-zero memsets); `pack_xdbl_cols_{f32,bf16,f16}` assembles
  `d_xdbl` in one kernel from the three sources that tile the row,
  removing the zero + cast + scatter staging in both backward lanes.
  All stores keep the `0.0f + x` form and are bit-identical to the old
  chains (run digests equal on all three GEMM tiers).
- The f32 forward residual chain writes the next layer's residual slot
  (or `norm_f_input`) directly, mirroring the mixed lane — the
  per-layer `temporal`->residual copy and the pre-`norm_f` copy are
  gone (25 D2D memcpys per step).
- `m3_dqkv` (the dominant M3 backward kernel, both dtype copies):
  the chunk's K[a]·Q[b] and V[a]·dO[b] pair dots are staged in shared
  memory once per chunk as strict-upper-triangle matrices instead of
  being recomputed per lane (an hd-fold redundancy across four
  sections), and two heads pack into one block when `nh` is even so a
  16-lane head fills a full warp. Each element/lane keeps the same
  ascending-index arithmetic — bit-identical outputs. Configs whose
  matrix-inclusive tile would exceed the ~99 KB consumer smem opt-in
  (large d_state) fall back to the inline dots via a launch-time tier
  ladder, losing only the speedup, never the launch.
- `m3_dqkv` pair-mats tier also stages the decay triangle
  `exp2((dA[b]-dA[a])*LOG2E)` and the per-step `exp_fwd`/`exp_rev`
  lanes in shared memory next to the pair matrices — the five decay
  consumers and five exp consumers recomputed those transcendentals
  inline per lane (~7.4k `exp2f` per (b,h,chunk) lane), and the
  loop-invariant chunk-sum `exp2f` in the d_state update is hoisted.
  Every staged value uses the consumer-exact expression, so reads are
  bit-identical; the legacy tier keeps the inline forms. The in-kernel
  packed-head slice stride is now tier-conditional, matching the
  launcher's tile maths exactly on every tier (the old unconditional
  stride overlapped slot 0's tail arrays once the tile grew). M3
  production shape (B=8 T=1300 d_model 384, 24 layers, bf16 graph):
  381.0 -> 362.1 ms/step.
- The chunked M3 backward runs its chunks in parallel: the reverse
  d_state recurrence is decomposed into per-chunk terms
  (`m3_dqkv_state_terms`), a serial per-(b, h, p, n) fold producing
  each chunk's entering state (`m3_dstate_passing_bwd`), and `m3_dqkv`
  itself drops its serial chunk loop — chunks ride grid z, mirroring
  the forward's chunk_state/state_passing/chunk_scan structure. The
  zero-seeded per-chunk term sums group differently than the fused
  serial accumulate (same release-window family break; deterministic,
  shape-pure). dD partials go per (b, chunk, h) with the ascending-row
  reducer. Isolated m3_dqkv: 3.89 -> 1.94 ms/launch; M3 production step
  bf16 graph 219.6 -> 179.4 ms/step.
- `m3_dqktheta` hoists each angle's `cosf`/`sinf` into registers — the
  forward-RoPE, inverse-RoPE and dtheta loops each recomputed the same
  pair (3x SFU work per angle). Bit-identical (same functions, same
  inputs); production-neutral at d_state 16 (4 angles), the win scales
  with d_state.
- `m3_dqktheta` I/O is staged through six [CS][ds] shared-memory tiles
  (4 inputs, 2 outputs): the per-thread row loads/stores put adjacent
  threads nh*ds floats apart — one 32-byte sector per access. Threads
  now stream the tiles cooperatively and each reads/writes its own row
  in smem; identical values, bit-identical outputs (new
  `m3_dqktheta_output_hash` arm matches pre-change hashes). Configs
  whose tile exceeds the 48 KB no-opt-in dynamic-smem limit (large
  d_state) fall back to the direct-global path via a `use_staging`
  launch flag. Isolated: 0.612 -> 0.270 ms/launch; M3 production step bf16
  graph 226.0 -> 221.5 ms/step.
- M3-KILL-1, `m3_dqkv` t-split (both dtype copies): the kernel ran one
  32-thread warp per block behind an 88 KB two-head smem tile — one
  block per SM, ~2% occupancy, and 73% of the whole M3 training step
  by isolated measurement. Head packing is retired; blockDim becomes
  (hd, T_SPLIT=16) with one head per block (44 KB tile, up to 2
  blocks/SM), and every per-timestep loop strides its timesteps over
  the T_SPLIT lanes. Each output element keeps exactly one owning lane
  running the same inner-loop order, so outputs are bit-identical —
  proven by an FNV bit-hash of all six outputs at the production shape
  (new `m3_dqkv_output_hash` arm) matching the pre-change hashes
  exactly. The one order-sensitive scalar (dD) is resummed from the
  stored dQK lane in the historical t-ascending order on a single lane.
  The warp-reduce mask names only the hd-lane segment (t-split slices
  of one warp can run different trip counts — a whole-warp mask would
  be UB), and the typed copies drop their stale `__launch_bounds__(32)`
  pin. Isolated: 11.40 -> 4.02 ms/launch (f32), 10.97 -> 3.89 (bf16).
  New `m3_kernels_isolated_bench` arm keeps the M3 kernel ledger
  measurable without a profiler.
- The `gemm_bi_forward` scalar dispatcher gained a strided-X entry
  (`gemm_bi_forward_sub` with an explicit `lda`); the public wrapper
  delegates with `lda = K`, behavior unchanged.
- The workspace test harness runs single-threaded
  (`RUST_TEST_THREADS=1` via `.cargo/config.toml`): `cudaFree` from a
  sibling test's drop invalidates an in-flight stream capture
  (documented CUDA hazard; production runs one trainer per process).

### Added

- `tests/cublas_compute_probe.rs` (`--ignored`): bf16/f32 GemmEx
  accuracy probe across compute-type x math-mode cells against an
  on-device fp64 reference, with a findings table in
  `docs/determinism-benchmarks.md`. On CUDA 13 / sm_120 the
  `COMPUTE_32F_PEDANTIC` pin keeps its justification (the 32F accuracy
  gap persists and grows with K), `DISALLOW_REDUCED_PRECISION_REDUCTION`
  has no effect on bf16 GemmEx, and `COMPUTE_32F_EMULATED_16BFX9`
  matches true-fp32 accuracy at up to ~2x speed on f32 GEMMs.
- Benchmarks: env-driven production-shape arm
  (`bench_lm_train_production_shape`), split forward/backward attribution
  arm, parallel-scan T64 arms, an IEEE-f32 row
  (`MAMBA_RS_BENCH_IEEE_F32`), and env-shaped M3 train bench.
- M3 kernel instruments (`m3_final_grads_unit_parity`, `--ignored`):
  `m3_kernels_isolated_bench` times m3_dqkv/m3_dqktheta/colsum and the
  two forward chunk kernels standalone at the production shape (compiled
  at the production state cap), and `m3_dqkv_output_hash` FNV-hashes
  all six m3_dqkv outputs — the bit gate for lane-redistribution work
  on a kernel no run-digest instrument covers. T_SPLIT sweep recorded:
  8 -> 4.17, 16 -> 4.02, 32 -> 6.24 ms/launch (f32); 16 wins.

### Changed (production-shape program, second pass)

- conv1d dw/db goes T-tiled: each (b, d, tap, tile) lane keeps the
  descending-t order within its tile and the ascending-row reducer
  folds (b, tile) rows in a fixed order. The old single 1300-step
  serial walk per lane was 13.4 x24-layer ms; tiled it is 2.6. The
  new dW/db summation grouping applies on EVERY GEMM tier (the conv
  kernels are shared; the tier routes only GEMMs), so all nine run
  digests move — final 0.6.3 baselines: T300
  Cublas 66f066ef7631a79c / BI 9bc88814dedb2bd7 / TC 03afb9d1506c101d;
  T1300 a65976720310c7cb / 0ae205c65b6f08e8 / 637806799ed25e59;
  T2100 31aa43dff5dc6022 / 4b6c5288d4826b7c / a7c2a91cef94d002.
  Run-to-run determinism verified (two runs bit-equal on all arms);
  full park 421/421; the M3 output hashes are unchanged.
- The parallel-scan backward's per-n d_a reduce and the final d_D
  reduce finish their last five halvings with warp shuffles instead
  of smem+barrier rounds — identical pairing order, bit-identical
  sums, ~160 fewer barriers per block.
- Production shape (bi+tc): 155.4 -> 142.6 ms/step with the conv dw tiling.
- The parallel-scan backward folds dB/dC across d-groups in-kernel
  (`ssm_parallel_scan_bwd_fold_*`, group size 4): the ungrouped kernel
  materialized [B, ds, d_inner, T] locals whose stores alone were 57%
  of the kernel by ablation, and the reducer read them all back. Each
  block now owns four consecutive d lanes, folds their dB/dC terms in
  ascending-d order in registers, and writes one partial row per
  group - local-tensor traffic and the reducer's depth both drop 4x,
  and B/C reads are shared across the group. The group fold is a
  different dB/dC summation order on every tier (same release-window
  family break; the partition is a pure function of d_inner). The
  ungrouped kernel remains the path for d_inner not divisible by 4.
  Final 0.6.3 digest baselines: T300 aa7de66a1d3b9c56 /
  9e0dbf87666dcfc6 / 46ef33ac8c0a2616; T1300 31d84af8ee97c4a0 /
  3b8de71cdd677eb8 / 85c5c30051938604; T2100 53003ceadbf3219e /
  bde81eca3b8182c8 / ac8cd31a973be149 (Cublas / BI / BI-TC).
  Production shape (bi+tc): 142.6 -> 131.5 ms/step.

- BIT-FAMILY BREAK (batch-invariant lanes): the split-M TN partition
  drops its `n_in >= 128` floor, so small-K dW GEMMs against large
  batch reductions split across M instead of running a handful of
  CTAs (dt_proj dW: 6 -> 282 CTAs, 0.807 -> 0.064 ms/layer; the dt
  pair 21.3 -> 3.5 x24-layer ms). The split changes the dW summation
  order, so BatchInvariant/BatchInvariantTc run digests move to a new
  family; run-to-run determinism verified (two runs bit-equal), the
  partition stays a pure function of (batch, n_in, n_out), and the
  cuBLAS-lane digests are untouched. New digest baselines: T300 BI
  2b74dfbdb91ef0ec / TC 1feca6c2e341c873; T1300 BI 18f82bd55af98056 /
  TC a1127230fa767171; T2100 BI 3a124f6158a5922c / TC 3da29b372bc98d93
  (Cublas arms unchanged: d51f9412f5f09206 / 007082a9aa104f59 /
  ba13a79fc04a7faa).
- The production bench prints its resolved GEMM tier: the tier rides TWO
  env flags (`MAMBA_RS_BATCH_INVARIANT` plus `MAMBA_RS_BI_TENSOR_CORES`
  on top), and a reading taken with only the TC flag silently measures
  the cuBLAS lane — several same-day readings did exactly that. With
  the pair set, the batch-invariant tensor-core measurement is
  155.4 ms/step (cuBLAS lane: 169.0).

- T-major B/C for the parallel scan: the gather writes [b][n][t]
  (route-picked `gather_bc_cols_tmajor` twins) and all five parallel
  scan kernels read it, so each (d, n) lane walks contiguous t-runs.
  The old [b][t][n] layout paid one 32-byte sector per 2-byte element
  and owned 61% of the forward scan kernel by constant-load ablation;
  neither candidate chunk geometry (256x8, 128x16) moved anything,
  and the block scan and exp2f measured free. Pure permutation —
  identical values, all nine digest arms bit-equal. Isolated scan:
  fwd 2.646 -> 0.921 ms/layer, bwd 3.559 -> 2.227; production
  244.8 -> 169.3 ms/step (-31%). The scan launch geometry is also
  single-sourced now (SCAN_NTHREADS/SCAN_NITEMS in launch.rs mirror
  the kernel defines; resident-block pin scales with block size).

- S4, the h tape is gone on the parallel route: the forward saves only
  per-chunk (run_a, run_b, h_entry) rows — 3 floats per (b, d, n,
  chunk) instead of T+1 — and the backward replays h in-kernel with
  the same thread-local scan, the same block_inclusive_scan_ab and the
  same ((comp o run) applied to h_0) compose chain on the same inputs,
  so every replayed h is bit-identical to the value the old tape
  stored. The chunk-entry h (the backward's h_prev boundary read) is
  saved verbatim, and the replay reuses the backward's already-loaded
  delta/u/B and its da registers. All nine digest arms (three GEMM
  tiers x T300/T1300/T2100) are bit-equal to the pre-S4 baseline in
  BOTH modes; `MAMBA_RS_SCAN_TAPE=full` restores the full tape for one
  release. Production shape: 261.9 -> 247.4 ms/step and -511.6 MB/layer
  (-12.28 GB total) — a B=32 micro-batch now fits on the 32 GB card
  (963.8 ms/step; the full tape OOMs on its first 511.6 MB alloc).
- Tensor-core epilogues (NN forward, NT dX, TN dW, 128-tile family)
  store/accumulate the fragment's adjacent even-column pair with one
  packed access (32-bit typed store, float2 read-modify-write for the
  f32 dW accumulate) when the leading dimension is even; identical
  values, digest-clean. Production-neutral at d_model 384 — the epilogue
  was not a wall there; kept for the store-issue halving on
  wider-output shapes. Measured backward-GEMM ledger at the production
  shape (tensor-core tier, x24-layer ms): in_proj 4.9, x_proj 3.2,
  out_proj 3.7 — and dt_proj 21.3 on the scalar tier (its dW output is
  below the tensor-core gate), 64% of all backward-GEMM cost.
- The `bench_scan_kernels_isolated` forward arm pushed `h_saved` ninth
  instead of third, shifting every pointer after `y` by one slot — its
  earlier "scan fwd" readings measured a kernel reading the wrong
  buffers. Fixed with the S4 argument additions.

- conv1d forward (all dtypes): the sliding window lived in GLOBAL
  memory (~7 dependent accesses per timestep, 1300 deep, 24-block
  grid). It now lives in registers with one carry-in/carry-out —
  identical shifts and values. This single change was -28% of the
  production step.
- The conv pair is tiled over T (grid covers (b*d_inner) x T/128):
  `conv1d_burnin_forward_tiled_*` seeds tile windows from x_branch
  halo loads; `conv1d_bwd_dx_tiled_*` computes the anticausal 4-tap
  FIR with tile-boundary carries seeded in the serial association
  order. The dw/db pass is tap-split (one lane per (b, d, tap) plus a
  bias lane), each lane keeping its exact descending-t add order.
- LEG-4, the conv tape is gone: the forward saves only the carry-in
  window `[B*d_inner*d_conv]`; the backward reconstructs every window
  from the saved x_branch activation (now a layer act). Net -2.7 GB
  VRAM at the production shape.
- S2 tape layout: h_saved and the parallel backward's dB/dC locals go
  t-innermost on the parallel route, with `ssm_reduce_d_BC_tmajor_*`
  reducer twins (ascending-d sum and `0.0f + sum` store verbatim);
  scan smem staging for delta/u/dy/B/C replaced by direct loads
  (barrier diet). Both measured neutral at the production shape and
  kept for coalescing correctness and the chunk-tape groundwork.
- Multi-chunk digest arms (T=1300 / T=2100 across all three GEMM
  tiers) — the inter-chunk carry was previously outside every digest
  instrument — plus isolated per-kernel bench arms for the scan pair
  and the backward suspects.
- Digest-pin lessons recorded twice: an inlined product contracting
  to one FFMA (conv bias lane) and a fused accumulate store (rmsnorm)
  both move every digest; both are pinned with explicit __fmul_rn /
  __fadd_rn to the historical rounding shapes.

### Performance (RTX 5090)

- B2 T64 d768 L24 graph lane: LM f32 28.3 -> 23.9 ms/step; bf16
  43.9 -> 40.2; f16 49.1; parallel-scan T64 f32 18.4.
- Production shape (d384 L24 B8 T1300, batch-invariant + tensor-core
  tier — the classify trainer's stamped route): 441.4 -> 261.9
  ms/step (first pass), then 155.4 after the second (-65%
  total: slim tape, NDEBUG, T-major B/C, split-M dW). The cuBLAS
  default lane sits at 169.0 at the same shape. Remaining ledger
  (x24-layer ms): scan fwd 22.1 / bwd 53.4, conv dw 13.4,
  backward GEMMs ~15, reduce_d_BC 9.6. Split: fwd 134 -> 80, bwd+opt 309 -> 182. Isolated
  ledger after the first pass: scan bwd 80, scan fwd 62, conv dw 13.4,
  reduce_d_BC 9.6 (x24-layer ms). Note: earlier "BI+TC" production
  rows in this file's history measured plain BI — the tier flag is
  MAMBA_RS_BI_TENSOR_CORES.
- M3 production shape (bf16 graph): 636 -> 381 (dqkv pair matrices +
  two-head packing) -> 362.1 (decay/exp staging) -> 226.0 (t-split
  block widening) -> 221.5 ms/step (dqktheta coalesced staging).
  Isolated m3_dqkv: 11.0 -> 3.9 ms/launch; remaining M3 ledger
  (x24-layer ms): dqkv 93.4, chunk_scan_fwd 20.8, dqktheta 6.5,
  colsum pair 6.4, chunk_state_fwd 4.3.

## 0.6.2 (2026-08-22)

### Changed

- `DetReduceKernel::launch` is `unsafe fn` with a documented safety
  contract (it takes raw device pointers).
- Reducer scratch allocation binds its CUDA context before allocating.
- The rendezvous environment-contract constants (`ENV_RANK`,
  `ENV_WORLD`, `ENV_DEVICE`, `ENV_RENDEZVOUS_DIR`, `ENV_JOB_ID`,
  `ENV_SEED`) are re-exported from `dist`.

### Fixed

- The loopback harness synchronizes the stream on error paths before
  scratch is freed.
- A helper-thread panic under an init deadline is reported as a panic
  instead of a timeout.
- The reduce-contract match is exhaustive: adding a contract variant
  fails compilation instead of falling into the no-communicator error.
- `EmulatedWorld::reference_mean` validates arena count and lengths and
  returns `Result`.
- Bootstrap: a partially-set rendezvous environment override is
  rejected; a child whose status query fails is killed and reaped;
  builds without the `nccl` feature refuse to spawn a multi-process
  world; `DistConfig::validate` rejects duplicate device ordinals.
- The DDP mean scale rejects gradient arenas above the i32 kernel ABI
  limit on both trainers.

### Added

- Tests: launcher environment parsing (torchrun/SLURM/OpenMPI), job-id
  validation, `DistContext::shard`, `reference_mean` validation,
  fold-tree sensitivity, and empty-shard / zero-length reduction paths.

## 0.6.1 (2026-08-22)

### Added

- Transport-backed fixed-order reducer (`dist::reducer`): shard
  exchange over NCCL send/recv/broadcast as byte movement, with the
  per-element ascending-rank fold performed by the `det_sum_ranks`
  kernel. `ReduceContract::FixedOrder` now runs on live multi-process
  worlds. A single-GPU loopback harness pins the device fold against
  the host reference bit for bit, including uneven shard plans and
  delivery-order permutations; the live two-rank test covers both
  contracts.
- `all_reduce_host_f32` and `any()` run over the transport.
- CI type-checks the `cuda,nccl` feature combination.

### Changed

- Collective deadlines cover enqueue through completion: the watchdog
  spans the stream synchronize and aborts the communicator on expiry,
  so a peer failure surfaces as an error within `collective_timeout`.
- Watchdog fire/disarm is decided by a single compare-exchange; a
  disarm that loses the race reports the fire, and the park-based timer
  disarms immediately.
- Communicator teardown is single-shot: an aborted communicator is
  never re-aborted or destroyed, and `shutdown` no longer chains
  destroy after abort.
- The unique-id exchange and the NCCL init share one `init_timeout`.
- Supervisor-spawned ranks receive PDEATHSIG on Linux and exit with a
  dead supervisor.
- Bootstrap rejects a `logical_world` that disagrees with the
  launcher's world, validates job ids before path joins and purges, and
  honors the `MAMBA_RS_SEED` override at world size 1.
- f16 multi-GPU training is rejected at `backward_step_dist` (bf16 and
  f32 are supported).
- Reducer receive scratch is allocated uninitialized (every slot is
  written before the fold reads it), cached per arena length, and never
  freed while in-flight work may reference it.

## 0.6.0 (2026-08-22)

Deterministic data parallelism, Mamba-3 prompt prefill, first-class
large state dimensions, and a performance pass over the chunked
kernels.

### Added

- `dist` — deterministic data-parallel training: one process per GPU,
  one reduction per optimizer step over the flat f32 gradient arena.
  The default `FixedOrder` contract folds the per-element addends in
  ascending logical-rank order, making the reduced bits independent of
  transport, delivery order, topology, and library version; the
  `NcclSum` tier uses the library collective. Includes the seed law
  (all randomness derives from one master seed, never from a rank), a
  supervisor/attach bootstrap (self-spawn, torchrun, SLURM, OpenMPI),
  file rendezvous, and `EmulatedWorld`, a single-process oracle
  asserted bit-for-bit against the reference fold. Validated on two
  RTX 5090s: both ranks' final weights match the emulated oracle bit
  for bit. The reduction composes with every per-rank compute mode
  (GEMM tier, scan mode, dtype).
- `nccl` feature: a communicator over the pinned NCCL binding,
  unique-id exchange through the rendezvous, version preflight,
  fail-fast shutdown, and `backward_step_dist` on both trainer
  families (world size 1 is byte-identical to a plain step).
- Optimizer state export/import on both trainers: Adam moments, step
  counter, and hyperparameters travel with the checkpoint; a resumed
  run continues bit-for-bit. The carried recurrence exports alongside
  for TBPTT window handoff.
- `grad_arena` / `apply_step`: the applying backward splits into
  gradient accumulation and the optimizer tail, bit-identical to the
  fused call.
- Mamba-3 one-pass prompt prefill through the chunked pipeline, leaving
  all recurrent states positioned for decode; continued windows apply
  the trapezoidal boundary fold. Includes a CUDA-graph twin with
  bitwise replay and automatic prefill in the LM generate path for long
  prompts.
- Mamba-3 mixed-precision training closure: typed backward for the
  plain SiLU-gate output, mixed-precision training for non-identity
  input projections, and sentinel-sized sequential tapes on the chunked
  path.
- Large state dimensions on every kernel generation: per-thread state
  arrays are sized at JIT time from the model config, up to d_state
  256. The chunked-backward shared-memory requirement is validated by
  its exact formula.
- `rms_norm_eps` is a config value carried by checkpoints and applied
  by every norm kernel.

### Changed

- The f32 Mamba-3 trainer rejects an empty input projection at
  construction; pass an identity matrix for a pass-through (the
  empty-projection convention is mixed-precision only).
- The CPU-vs-GPU prefill oracle tolerance is 2e-3 to cover the float
  noise floor across GPU generations (sm_89 vs sm_120).
- cudarc floor raised to 0.19.9: upstream gates CudaSlice/SyncOnDrop
  teardown behind is_managing_stream_synchronization, which protects
  CUDA Graph capture from drop-time stream waits.
- Chunked-kernel performance pass: a shared-memory Q·K tile per
  (chunk, head), hoisted exponential and V load in the chunk-state
  kernel, two heads per block, chunk-parallel angle accumulation, and a
  warp-parallel decay-gradient section in the backward. The serial
  reverse-cumsum combine is unchanged.
- Trainer construction derives `a_neg` with the same device kernel the
  post-step refresh uses, fixing bit-continuous resume.
- Oversized shapes are rejected at construction: linear index ranges
  are validated against 32-bit kernel arithmetic, and sequential-tape
  sizing is an explicit constructor flag matched to the scan mode.

### Performance

Before/after ratios measured on an RTX 6000 Ada shared with other
load; release absolutes measured on an idle RTX 5090 (CUDA 13.0,
release build).

- Prompt prefill (T=4621, 24 layers, d_model=384): 384 ms before this
  cycle, 66.5 ms after (shared Ada); 23.65 ms (42.3 prefills/s) on the
  idle 5090.
- Multi-chunk training step (B=1, T=256, 24 layers, d_model=384):
  f32 424 -> 127 ms, bf16 395 -> 122 ms (shared Ada); f32 110.5 ms,
  bf16 112.0 ms on the idle 5090.
- Mamba-1 at the 130m-class shape (B=2, T=64, 24 layers, d_model=768),
  full step with AdamW, idle 5090: f32 28.5 ms, bf16 44.5 ms.
- Fused decode step across state capacities (Mamba-1, d_model=256,
  4 layers, Ada): 0.29 ms at d_state 64, 0.38 ms at 128, 1.50 ms
  at 256.

### Next

Multi-GPU inference for models larger than one device (pipeline
sharding) and the Mamba-2 generation are the focus of the next
releases.

## 0.5.3

Serving-performance release. No change to any number the 0.5.2 paths
produce; the new entries reproduce existing routes bit-for-bit (pinned
by tests).

### Added

- `gpu_forward_inference_prefill_pooled_sum_from_raw` — the raw-input
  prefill emitting the on-device column sum of the post-`norm_f`
  temporal. Mean-pool consumers download `d_model` floats and divide by
  T on the host instead of downloading the full `[T * d_model]`
  temporal (7.1 MB -> 1.5 KB at a 23M-classifier shape). Bit-identical
  to the host column sum by reduction-order construction (oracle in
  tests/gpu_pooled_prefill.rs).
- Disk cache for the NVRTC-emitted PTX, keyed by source blob + arch +
  compile options + NVRTC version (`MAMBA_RS_KERNEL_CACHE` overrides the
  location; `0`/`off` disables). A hit skips the NVRTC half of the boot
  tax; the emitted PTX is byte-identical by construction. Hit-path
  failures invalidate the entry and recompile; failures are never
  cached.
- `PinnedHostBuf` — cacheable page-locked host staging (cuMemHostAlloc
  flags=0; never the write-combined `alloc_pinned`, which regresses
  host-read consumers). Exposes plain slices for the existing
  upload/download paths, so the driver takes the true-DMA route with
  the same calls and the same bytes.
- `serialize::load_from_bytes` — for consumers that hash-verify the
  artifact and should not read the file twice.

### Added (serving kernels)

- `PrefillPooledGraph` — the whole per-page pooled prefill (state reset +
  projection + layer chain + norm_f + column sum) captured as one CUDA
  Graph over fixed buffers; a replay re-issues the exact captured
  kernels, bit-identical to the eager entry (100x-replay oracle). GEMM
  flags are snapshotted at capture and asserted at launch.

### Changed

- conv1d nosave burnin: `__restrict__` everywhere plus a d_conv==4
  register-window fast path — the shift register, taps and bias live in
  registers across the T loop instead of round-tripping global memory
  per timestep. Same FMA chain term for term; bitwise
  prefill-vs-training parity holds.
- Prefill residual handling ping-pongs buffers instead of copying the
  full activations every layer (was ~25 D2D copies of [T * d_model] per
  page). The one value-affecting change is the operand order of a
  commutative f32 addition — bit-equal, pinned by the parity suite.
- The gemm_bi split-K/split-M and transpose scratches (32 + 16 MB)
  allocate lazily on first batch-invariant use — inference-only
  consumers no longer hold 48 MB of dead VRAM.


## 0.5.2

Correctness and hardening release. No change to any number the 0.5.1
default paths produce.

### Fixed

- Mixed (bf16/f16) inference prefill now routes through the scan
  dispatcher: prompts above the parallel threshold run the typed parallel
  nosave kernel instead of unconditionally paying the O(T) sequential
  burnin. The sequential branch asserts the `d_state <= 64` register cap
  instead of silently returning; the f32 T=1 decode gets the same assert.
- `ScanMode::resolve` now takes `d_state` and delegates to `use_parallel`,
  removing a duplicated threshold that missed the `d_state > 64` override.
- M3: `m3_reduce_d_D` accumulates instead of overwriting (the D
  skip-connection gradient of prior micro-batches was discarded under
  `accumulate_only`); `Mamba3Trainer::new_full` validates the config on
  both dtype branches; the fused `step()` invalidates a pending split
  forward instead of letting a later `backward_step` run on overwritten
  activations.
- `seq_len == 0` / `batch == 0` are rejected with a clear error at the
  capacity gate instead of surfacing as an opaque CUDA error on a
  neighbouring launch.
- GEMM flags are guarded across graph capture/replay and the split
  forward/backward cycle: graphs snapshot the flags at capture and assert
  at replay; `backward_step` refuses a mid-cycle flip; the f16 graph gains
  the half-staging/upcast pointer asserts its bf16 twin already had;
  setters warn once a graph exists.
- `serialize` round-trips `scan_mode` and `rms_norm_eps` (load used to
  reset both to defaults, so a non-default-eps checkpoint silently served
  at 1e-5). Both fields are optional on read; pre-0.5.2 files load
  unchanged.
- Documentation now matches the code: the batch-invariant flag covers the
  training triads and the typed decode matvec (not the tied LM heads or
  the M3 engine); bit-reproducibility is a within-route contract, with
  tolerance across routes.

### Added

- `mamba_rs::VERSION` const for provenance stamping.
- `GpuCtx::gemm_flags()` — a snapshot of the active GEMM flags.
- Opt-in non-PEDANTIC typed-GEMM compute (`set_fast_gemm` /
  `MAMBA_RS_FAST_GEMM`), default off. Experimental.
- `_Pragma("unroll")` restored in the typed scan macros (a bare
  `#pragma` cannot survive macro expansion); bit-neutral on all three
  GEMM tiers.
- Tests: the parallel-vs-sequential parity suite now actually runs the
  parallel kernels (in-test dispatch assert, T=2048 case); a mixed-prefill
  parallel-route oracle; CPU-only scan-mode boundary pins; run-to-run bit
  determinism per GEMM tier with tier distinctness; a scan-mode step
  benchmark.
- CI: a `cargo check --features cuda` job.
- Docs: numeric-routes section in the architecture guide, three-tier
  determinism results, the M3 chunked statelessness contract.


## 0.5.1

Patch release: the GPU serving surface for pooled-output consumers. The
first production consumer (a 23M-param page classifier serving at
T=4621) scored through implementation details -- a pub scratch field for
the all-T temporal and a hand-rolled input projection; 0.5.1 makes both
official. No math changes; every existing bit-anchor is byte-untouched
(the new entries compose the exact body the existing prefill already ran).

### Added: official all-T + raw-input GPU inference prefill

- `gpu_forward_inference_prefill_full` -- the f32 prefill with a
  `PrefillOutputs` surface: the post-norm_f temporal for ALL T positions
  (`[B*T*d_model]`) as an official out-param alongside the last-timestep
  gather. Consumers that pool over the whole sequence (mean-pool
  classification heads) no longer read `scratch.out_flat` directly.
- `gpu_forward_inference_prefill_from_raw` -- one-call serving entry for
  pre-projection input (`PrefillRawInputs`, `[B*T*mamba_input_dim]`):
  applies `input_proj` internally with the SAME SGEMM call the training
  forward makes, then runs the shared prefill body -- the projected bits
  equal training's `ip_out` by construction. Rejects non-f32 input_proj
  weights loudly.
- All three f32 entries (legacy last-only, `_full`, `_from_raw`) now
  compose one private body -- drift between them is structurally
  impossible.
- The `_mixed` prefill deliberately has no all-T twin: its temporal is
  typed (bf16/f16) and no mixed consumer pools over the full sequence
  today; the entry lands when one exists.

### Added: GPU inference-prefill parity suite

`tests/gpu_inference_prefill_parity.rs` pins the serving claim: the
nosave inference prefill produces the SAME BITS as the GPU training
forward -- full temporal at every position, the last-position gather,
both scan kernels (Sequential small-T and Auto/parallel-scan above the
threshold), and end-to-end through `_from_raw` INCLUDING the internal
input projection. Before this suite the nosave prefill had zero test
coverage.

### Fixed

- `prefill_bench_classifier_shape` ran T=4617 (the bare patch grid); the
  production patchify appends 4 register tokens -- T=4621.

## 0.5.0

Feature release: the bring-your-own-loss training split, full-sequence
CPU prefill, and the supervised-training toolkit (grad clipping, gradient
accumulation, LR schedules, no-decay groups, trainable bf16/f16
input_proj). All additive; the fused `step()` and every captured-graph
path are byte-untouched (the split composes the exact eager phase bodies
the fused step already ran -- pinned by bit-identity tests).

MSRV 1.94 -> 1.97 (dev/CI toolchain 1.97.1 -- never 1.97.0, which carries
an LLVM miscompilation). cudarc 0.19.8, safetensors 0.8, hf-hub 1.0,
tokenizers 0.23.

### Added: forward/backward split on both trainers

`MambaTrainer::forward()` / `backward_step()` (and the `Mamba3Trainer`
mirrors): `forward` returns the full `batch * seq_len * d_model`
post-norm_f temporal output on the host (f32 on every dtype -- bf16/f16
upcast on device before download), so a caller-side loss can run between
the halves; `backward_step` backprops the caller's `d_temporal` through
the saved activations and runs AdamW. `BackwardOpts` carries:

- `clip_max_norm` -- `torch.nn.utils.clip_grad_norm_` semantics on a
  DETERMINISTIC fixed-order f64 partials reduction (one new kernel,
  `kernels/grad_clip.cu` -- the only new kernel in the release). For f16
  the norm is computed after the unscale, per the
  unscale-then-norm-then-clip ordering law.
- `accumulate_only` -- exact gradient accumulation: the arena is not
  zeroed, Adam does not advance, the optimizer tail is skipped. The
  fused `step()` refuses while a window is open (it would zero the
  arena and silently discard the accumulated gradients). Two accumulated
  micro-batches reproduce the one-big-batch weights to <=1e-5 under the
  batch-invariant flag.

f16 rides the GradScaler protocol through the split (overflow skip, step
rollback, scaler update); `accumulate_only` is rejected on f16 (the
loss-scale freeze window across micro-batches has no defined semantics).
`examples/custom_loss.rs` shows the full loop.

### Added: full-sequence CPU prefill (both architectures)

The CPU inference path was per-step matvec only -- a T-token prompt paid
T dispatches and re-streamed every weight matrix T times.
`forward_mamba_backbone_prefill(_mode)` and the M3 twin
`forward_mamba3_backbone_prefill(_mode)` run the training forward's
batched-SGEMM pipeline on inference types with no activation tape, write
the post-norm_f output at EVERY position, and carry the recurrent state
so the step path continues seamlessly (prefill-then-decode).
`PrefillMode::Parallel` parallelizes every phase (GEMMs via the gemm
crate's rayon parallelism, per-channel/per-head loops, transpose tiles)
with no cross-task reductions -- bit-equal to `Single` by construction,
and both bit-equal to the training forward (anchor tests pin all three
across the {default, gemm-blas, accelerate} backends). Batch helpers
`prefill_batch` / `prefill3_batch` parallelize per sample instead.
`examples/cpu_prefill.rs` demonstrates prefill-then-decode and pooling.

### Added: training-loop controls

- `set_lr` / `lr` / `drop_graph` on both trainers. `set_lr` errs while a
  captured graph exists -- the lr is baked BY VALUE into the captured
  AdamW kernel and a bare field write would be a silent no-op under
  replay; `drop_graph -> set_lr -> capture_graph` re-arms graph stepping.
- `set_reference_no_decay(true)` -- the reference `_no_weight_decay`
  parameter groups (M1: `a_log`, `D`, dt bias, every RMSNorm scale; M3:
  dt bias, `D`, every norm scale) get `weight_decay = 0`, matching
  upstream AdamW grouping. Default OFF preserves the historical
  decay-everything behavior bit-for-bit.

### Added: trainable input_proj in the mixed (bf16/f16) trainer

The mixed trainer previously rejected a non-identity `input_proj`
(HF-LM-shaped assumption). Both stubs are now filled -- typed forward
GEMM with f32 bias + upcast into the residual stream; backward computes
db (deterministic bias reduce) and dW (typed TN GEMM); dX below the
projection is intentionally not produced (nothing upstream of the
projection is trainable). Covered by the mixed parity walk (the
input_proj skip is REMOVED), a rectangular split==fused bit-identity
test, graph capture at a large patch dim, and batch-invariant probes.

### Fixed: matvec_bi faulted on arbitrary K

Pre-existing 0.4.x bug: `k_per_warp = ceil(K/8)` could be ODD (K=200 ->
25), producing a misaligned paired `LDS.U32` smem read
(`CUDA_ERROR_MISALIGNED_ADDRESS`); the vectorized `uint4` A-row load
also assumed 16-byte row alignment, and a `__builtin_assume(K % 8 == 0)`
was UB for most K. Historic HF d_models (768/1024/2048/2560) all
produced even spans, masking it. Fixed: even per-warp span (existing
shapes' reduction order untouched -- determinism suite bit-green),
alignment-guarded vector loads, assume removed. Fresh-process per-K
probes in `tests/matvec_probe.rs`.

### Fixed / changed, smaller

- Dead `da_exp` activation buffer removed from the GPU forward: kernels
  never wrote it and backward recomputes the value -- ~0.9 GB/layer at
  vision-class shapes (21.8 GB at d384x24 T=4617). Measured after
  removal (bf16, 32 GB 5090, T=4617, input_dim=1024): B=1 = 9.6 GiB,
  B=2 = 18.0 GiB, B=4 = OOM -> micro-batch 2 + accumulation.
- f16 graph capture pre-sizes the input_dim-aware batch-invariant upcast
  scratch (a lazy grow inside capture is illegal).
- Stale `step()` rustdoc claimed `d_temporal` is `batch * d_model`; the
  actual contract is `batch * seq_len * d_model` (fixed in all three
  sites plus the M3 heavy-tail A comment block).
- Slim GEMM geometry macros are `#undef`'d at the end of their section
  (the 0.4.0 geometry-leak class cannot recur by appending kernels).
- AdamW per-step pairs Vecs are pre-sized (no grow-reallocs per step).
- f64 shadow-forward gradient oracle for the CPU training path
  (`tests/grad_oracle.rs`) -- certifies every weight-tensor gradient
  against an independent f64 recomputation, replacing noisy
  finite-difference checks as the analytic reference.
- docs.rs metadata: renders the CPU API surface (`hf`, `gemm-blas`,
  `cli`); GPU doc-badge plumbing is a planned follow-up.

## 0.4.2

Performance update: the tensor-core deterministic tier gets a small-tile
kernel family and deeper K-staging. No API changes; all determinism and
batch-invariance contracts unchanged (outputs of the existing TC kernels
are bit-identical to 0.4.1 by golden-hash verification).

### Added: Tile64 TC family (64x64 output tiles)

Six new kernels (`gemm_bi_{nn,tn,nt}_tc64_{bf16,f16}`): CTA 128
threads / 4 warps, each warp owning a 32x32 quadrant. They cover the
64..127 output-dim band the 128-tile family gated out, and grids that
would underfill the GPU at 128x128 route to the 64-tile twins
(`tc_pick_tile`, threshold 72 CTAs). The two families are BIT-IDENTICAL
per output element — same ascending reduction slabs, same m16n8k16
chain, same tail zero-fill — so the shape-only routing can never change
output bits and the strict all-M forward invariance survives tile
switching (`tc64_and_tc128_bit_identical` asserts it directly).

Whole models that previously fell back to the scalar tier now ride
tensor cores (d128/d256), and the narrow projections of every model
size (x_proj N=80, dt_proj K<=96) come along.

### Changed: TC staging BK 32 -> 64, dynamic shared memory

Both tile families stage 64-deep K-slabs (half the barrier/wait_group
boundaries per CTA). The 128-tile family now exceeds the 48 KB static
cap (NN 71 680 / TN 69 632 / NT 73 728 B) and uses dynamic smem with a
75 776 B `MAX_DYNAMIC_SHARED_SIZE_BYTES` opt-in at load; the 64-tile
family stays static (36 864 B). No register spills on any of the 12 TC
functions.

### Performance (RTX 6000 Ada, CUDA 13.2, vs 0.4.1)

Trainer ms/step, bf16 TC tier vs cuBLAS-PEDANTIC of the same dtype:

| model | 0.4.1 | 0.4.2 | vs PEDANTIC |
|---|---:|---:|---:|
| d128 x2L  | 2.345 | 2.200 | 1.10x -> 1.04x |
| d256 x4L  | 9.561 | 9.105 | 1.05x -> 1.01x |
| d768 x4L  | 22.736 | 21.694 | 0.88x -> 0.85x |
| d1536 x2L | 13.599 | 12.504 | 0.76x -> 0.71x |

GEMM-level: big-shape TC forward 107 -> 116 TFLOPS (M2048 K768 N3072
bf16), 131 -> 145 TFLOPS at M4096 K1536 N3072; small-shape dW/dX
(d128-class) roughly 2x vs the 128-tile route. Full tables:
docs/determinism-benchmarks.md.


## 0.4.1

Bug-fix release for the deterministic GEMM engine. No API changes.

### Fixed: native typed Big kernels never actually ran

The bf16/f16 Big NN/TN/NT kernels introduced in 0.4.0 compiled with the
WRONG tile geometry: `kernels/gemm_bi_triad.cu` redefines `BM/BN/BK/NUM_THREADS`
for the Slim section partway through the file and leaves them redefined,
so the typed Big kernels (appended at the end) silently picked up Slim
constants — 128-thread launch bounds and 32-deep K tiles against a
dispatcher launching 256 threads with 16-deep-tile shared memory sizing.
Every launch failed with `CUDA_ERROR_INVALID_VALUE`.

This was invisible in 0.4.0 because the typed dispatch chain treated ANY
kernel error as "bucket not covered" and silently took the upcast
fallback. The fallback produces bit-identical results by contract, so all
parity and determinism tests stayed green — but the native kernels never
executed, and the Big-shape upcast scratch (~0.5 GB at 2.8b mixed) was
still being allocated and paid for.

- The typed Big section now carries its own `GEMM_BI_T_*` tile constants
  (256 threads, 128x128x16 tiles), immune to preprocessor state left by
  earlier sections. The kernels execute for real and remain bit-identical
  to the f32 reference on upcast inputs (typed parity + 60-shape
  gate-boundary sweep, all green).
- Found while extracting the engine into the standalone `sgemm-bi` crate,
  whose error handling does not mask launch failures.

### Fixed: dispatch fallbacks no longer swallow launch errors

`gemm_bi_forward_typed` / `_backward_dw_typed` / `_backward_dx_typed`
and the tensor-core try-first paths matched on `.is_ok()`, so a genuine
launch failure in a covered bucket was silently "recovered" by
recomputing through the fallback — hiding the root cause (this is exactly
how the dead-kernel bug above survived). Uncovered-shape errors now carry
an `UNCOVERED` marker; only those fall through. Real launch errors
propagate to the caller.

### Validation

Full `cuda,hf` suite + ignored (HF checkpoints 130m-2.8b) green on
RTX 6000 Ada, CUDA 13.2; performance tables in
`docs/determinism-benchmarks.md` re-measured with the native Big kernels
actually executing.

## 0.4.0

Deterministic GPU training for all three dtypes, plus an opt-in
tensor-core tier that makes deterministic bf16/f16 training FASTER than
cuBLAS on LLM-sized models. Validated on RTX 6000 Ada, CUDA 13.2:
299 tests + 62 ignored (HF checkpoints 130m–2.8b), zero failures.

### Batch-invariant training GEMM triad (f32 + bf16 + f16)

With `MAMBA_RS_BATCH_INVARIANT=1` / `ctx.set_batch_invariant(true)`,
every training GEMM (NN forward, TN dW, NT dX) routes through custom
deterministic kernels in `kernels/gemm_bi_triad.cu` — two runs with the same
seed/inputs produce bit-identical weights on every dtype.

- f32 triad: bucketed dispatch (GEMV, ultra-thin, narrow, split-K32,
  gap-fill, Big/Slim, split-M/N) under a zero-cuBLAS contract: an
  uncovered shape panics loudly, never falls back silently.
- bf16/f16: native typed buckets keep f32 smem + accumulation with the
  f32 twin's exact FMA chain and ONE RNE downcast at store; shapes
  without a native bucket run "upcast → f32 kernel → RNE downcast" —
  bit-identical to a native typed kernel by contract
  (tests/gemm_bi_typed_parity.rs, incl. a 60-shape gate-boundary sweep).
- Native typed Big NN/TN/NT kernels fire exactly where the f32 cascade
  picks Big (predicate-mirrored gates) and eliminate the Big-shape upcast
  scratch (~0.5 GB at 2.8b mixed).
- Forward dispatcher gap-fill extended past N=2048 / K=2048 (Mamba-1
  in_proj at micro-batch for d_model ≥ 576; d_model=2560 at M < 32).
- Invariance contract: training is per-bucket batch-invariant (same
  bucket → row 0 bit-identical across M); typed inference decode
  (M < 128) always routes the strict all-M `matvec_bi` path, preserving
  cross-batch logits parity (KL ≈ 1e-12).
- Cost per step vs cuBLAS: f32 1.28–1.53× (TF32 baseline), bf16
  1.11–1.20×, f16 1.20–1.37× (PEDANTIC baseline).

### Tensor-core deterministic tier (opt-in)

`MAMBA_RS_BI_TENSOR_CORES=1` / `ctx.set_bi_tensor_cores(true)` (on top
of the batch-invariant flag) swaps the typed training triad for
mma.sync.m16n8k16 kernels: 2-stage cp.async staging, ldmatrix
(.trans where the operand layout needs it) fragment loads, conflict-free
smem strides, f32 accumulators, no atomics or splits.

- SEPARATE numeric contract: tensor-core reduction order differs from
  the scalar FMA chain, so outputs do not bit-match the scalar tier —
  but runs are bit-identical to each other (incl. through CUDA Graph
  capture/replay) and the forward is STRICTLY batch-invariant across
  all M (each element's K-reduction lives in one warp).
- GEMM speedups vs the scalar deterministic tier (bf16, Ada): forward
  3.0–3.4×, dW 2.1–3.5×, dX 4.3–6.7×.
- End-to-end mixed training step: d768 0.88× of cuBLAS-PEDANTIC,
  d1536 0.77× — deterministic training faster than cuBLAS (and faster
  than f32 TF32 at d1536).

### Stream-ordering fixes (latent races)

`ctx.stream` is NON_BLOCKING and never orders against the legacy
stream; several host↔device copies used synchronous legacy-stream
`cuMemcpyHtoD_v2`/`DtoH_decoy`, whose tail DMA can race kernels launched
right after. All converted to stream-ordered async copies + sync:
`DtypedBuf::{upload,download}_f32`, `WeightSliceDyn` weight uploads
(M1 + M3 + gpu_lm/gpu_lm3 embeddings), `GradSlice` copies (bracketed
by device sync), and a one-time drain after kernel compilation for the
split-K scratch buffers.

### CUDA Graph guards

The typed-GEMM upcast scratch is presized BEFORE training-graph capture
(`presize_bi_upcast_scratch_for_train[_m3]`) and its pointers are
asserted at every replay, same discipline as `half_staging` — a lazy
regrow after capture can no longer leave a captured graph pointing at
freed memory.

### API

- `GpuCtx::set_bi_tensor_cores` / `bi_tensor_cores` (new flag,
  `MAMBA_RS_BI_TENSOR_CORES` env).
- `blas::gemm_bi_forward_typed` / `gemm_bi_backward_dw_typed` /
  `gemm_bi_backward_dx_typed` — full-coverage typed deterministic GEMM
  entries.
- `WeightSliceDyn::{download_to_f32, upload_from_cpu_f32,
  upload_raw_bytes}` now take the stream they order against (breaking
  for direct callers).


## 0.3.1

### `matvec_bi_*` perf

Decode throughput on RTX 6000 Ada (mamba-130m-hf, M=1, CUDA Graph):
923 → 974 tok/s bf16, 919 → 972 tok/s f16. f32 unchanged at 727 tok/s.

Changes:
- Vectorized 128-bit `cp.async` loads for the A-row staging path (sm_80+).
- Packed `pair_to_f2` smem reads, `__builtin_assume((k & 7) == 0)` to
  drop the tail loop.
- `__ldcs` streaming load on B-tile reads (bypasses L1).

KL parity unchanged (≈ 1e-11 between `b=1` and `b=N` per slot).

### Batch-invariant matvec is now opt-in

The `matvec_bi_*` kernel dispatch, introduced as the default in 0.3.0,
is now behind an opt-in flag — cuBLAS gemv is the default path.

Enable via either:
- `ctx.set_batch_invariant(true)` on the `GpuCtx`
- `MAMBA_RS_BATCH_INVARIANT=1` environment variable at process start

When enabled, output is bit-identical across batch sizes. When
disabled, cuBLAS's per-M algo selection may produce sub-ULP
differences between decode and prefill.

### Deterministic CPU parallel backward (M1 + M3)

`parallel_mamba_backward` and `parallel_mamba3_backward` now use
static sample-to-thread partitioning + pre-allocated per-thread
gradient slots + fixed balanced binary-tree reduce, replacing the
prior rayon work-stealing + `rayon::broadcast` collection path.

Result: bit-identical output across runs with the same input,
regardless of thread scheduling. No throughput change.

Removed: `BWD_EPOCH`, `BWD_GUARD`, `THREAD_GRADS`, `THREAD_GRADS_EPOCH`,
`ensure_thread_grads_zeroed` (M1) and the M3 equivalents.

## 0.3.0

GPU training and bf16/f16 inference for Mamba SSM and Mamba-3 SISO,
unified trainer API, custom batch-invariant matvec.

### Trainer API

`MambaTrainer` / `Mamba3Trainer` — one entry point per arch.
`step(input, d_temporal)` does forward + backward + AdamW + (for mixed)
master→compute sync. `WeightDtype` chosen at construction; the F32/Mixed
engine split is internal.

`capture_graph()` records the whole training step into a CUDA Graph and
asserts pointer stability on every replay (catches the silent-corruption
class if any backing buffer gets reallocated between capture and replay).

`scaler_state()` / `load_scaler_state()` round-trip the f16 dynamic
loss scaler across checkpoint resume so training picks up at the
last-known scale instead of repaying the ~2000 step discovery phase.

`GpuAdamW` matches PyTorch — decoupled WD, f64 bias correction,
capturable variant reads bias factors from a 2-element device buffer.
`DynamicLossScaler` matches `torch.cuda.amp.GradScaler`; eager skips
AdamW on overflow, captured-graph runs a device-side conditional
unscale that sanitises inf/NaN to zero.

### Inference

Native bf16/f16 activation pipeline for both archs. Activations stay
in weight dtype, residual stream stays f32 (matches HF
`residual_in_fp32=True`), SSM state stays f32. `step_kernels_mixed_native`
replaces the cast-staged path. Native parallel prefill — no f32 fallback.

`GpuMambaBackbone::new_with_dtype` / `GpuMamba3Backbone::new_with_dtype`,
plus the corresponding `GpuMambaLM` / `GpuMamba3LM` LM wrappers.

### Batch-invariant matvec (the headline change)

Custom CUDA kernel in `kernels/gemm_bi_fixed.cu`. Output is
bit-identical across batch sizes (KL ≈ 1e-11), against cuBLAS's
`cublasGemmEx` which drifts to KL ≈ 1e-3 between M=1 and M=N because
it picks a different split-K / tile / algo per M.

Layout: 8 warps × 32 threads per CTA, K split across warps, per-warp
partials reduced in smem with a fixed reduction tree
`(((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)))`. Grid is 2D
`(ceil(N/32), M)` — one CTA per `(m_row, col_chunk)`, each row
independent. The same kernel handles M=1 (decode) and M>1 (RL parallel
envs, prefill).

Parity guardrails (`tests/hf_batch_parity.rs`, `tests/extreme_edge_coverage.rs`):

- `bf16_batch_divergence_known` (adversarial prompt that hit
  KL ≈ 2.7 with cuBLAS): now **KL = 0.0000**.
- `bf16_multi_length_parity` lengths 3, 5, 32, 63, 64, 65, 128:
  all KL ≤ 3.5e-11.
- `inference_extreme_batch_parity_bf16_b{16,32}`: KL ≤ 2.2e-11.
- `hf_cpu_vs_gpu_inference_bf16`: 20/20 token match, KL = 2e-6.

### HuggingFace

`rms_norm_eps` and `layer_norm_epsilon` are read from `config.json`;
the loader warns when a checkpoint specifies a value other than the
kernel-hardcoded `1e-5` (e.g. FalconMamba uses 1e-6).

Untied `lm_head` stride bug — the GEMM wrote at `vocab_size` but the
CPU downloader sliced at `vocab_size_padded`, so batch slots beyond
the first got wrong logits on any vocab not already 64-aligned (e.g.
mamba-130m's 50 280). Padded on upload in both LM wrappers.

End-to-end bf16 inference verified on every cached
`state-spaces/mamba-*-hf` snapshot (130m / 370m / 1.4b / 2.8b):
15/15 greedy match vs f32, KL ≤ 1.6e-3 (best 4e-5 at 2.8b).

### Critical bugfixes

`a_neg` staleness in training — the per-layer `a_neg = -exp(a_log)`
buffer was computed once at trainer construction and never refreshed,
so optimizer updates to `a_log` never reached the SSM. Recomputed
after every AdamW step across the eager, captured-graph, and f16
paths. Pre-fix, gradient descent on the A-matrix was a silent no-op
for the entire training run.

f16 loss-scaler overflow corruption — the captured-graph unscale
kernel did `grads[i] *= 0.0f`, and `±Inf * 0 = NaN` poisoned the
next AdamW step. Switched to `grads[i] = overflow ? 0 : grads[i] * unscale`
so overflow is always a clean skip.

Mamba SSM mixed forward routed `WeightDtype::F32` through the legacy
`conv1d_burnin_forward`, whose argument order differs from the typed
variants — the persistent conv state and the post-conv activation
buffer got silently swapped every step. Routed through a typed-signature
f32 kernel.

RMSNorm finite-guard — on 48-layer bf16 models a transient activation
overflow produced one NaN that cascaded through every subsequent
RMSNorm via `1/NaN = NaN`. Catastrophic on mamba-1.4b-hf (0/15 vs f32
pre-fix). Added the `if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f`
guard to every RMSNorm / BCNorm / RMSNormGated variant.

M1 parallel scan at d_state > 64 used a too-restrictive smem load
filter; replaced with a strided loop so every `ds` entry loads regardless
of the `ds / hd` ratio.

### Other fixes

f16 eager overflow path now skips AdamW + sync + `a_neg` recompute on
scaler-detected overflow (matches `GradScaler`). Spurious skips on
`m1_trainer_f16_production_lr_stable`: 47/50 → 3/50.

`AdamWBiasFactors` defaults to `[1.0, 1.0]` instead of `[0.0, 0.0]`,
which would have produced an all-weight-decay update if a graph were
captured before the first bias-factor write.

AdamW step counter clamps the bias-correction exponent at `2^30` — the
prior `as i32` cast went negative past `i32::MAX`. Cosmetic, free fix.

`step_kernels_mixed_native` (M1 + M3) uses `cuMemcpyDtoDAsync` on
cached raw pointers instead of `GpuBuffer::copy_from`, which goes
through a `SyncOnDrop` slice view that invalidates graph capture.
M3 `angle_dt` launch fixed for the same reason.

M3 NVRTC inlines `_typed_prelude.cuh` so the bf16 / f16 helpers are
in scope before any `DEFINE_*` macro expansion.

`GpuMambaWeights` / `GpuMambaMixedWeights` layout formula uses the
actual CPU weight lengths instead of `d_model`-sized `input_proj` —
HF checkpoints frequently have an empty `input_proj` (identity).

### API + crate plumbing

Inner trainer engines (`MambaTrainerMixed`, `MambaTrainerF32`,
`Mamba3TrainerMixed`, `Mamba3TrainerF32`) are now `pub(crate)`. Public
surface is the wrappers only — the F32/Mixed dispatch lives behind a
private enum, mirroring the existing inference `BackboneEngine` pattern.

`cuda` feature pulls `half` + `bytemuck` directly so
`cargo build --features cuda` (no `hf`) works on its own.

`WeightDtype` is re-exported from the crate root. `pub mod gpu3`
mirrors the existing `pub mod gpu` so Mamba-3 types land one import deep.
`Mamba3Weights` / `Mamba3LayerWeights` are `#[derive(Clone)]`.

All remaining `#[allow(clippy::too_many_arguments)]` removed by
bundling args into named structs (`TypedPtr`, `TiedLmDims`,
`PrefillInputs`, `Mamba3States`, `Mamba3LmBuild`, `BiGemmArgs`).

README rewritten with a bf16/f16 quickstart and an HF LLM example.

### Tests

49 test files, 287 tests pass on Ada with `cuda hf` + 60 ignored
benches. Notable:

`hf_training_convergence` — 30-step real-checkpoint training on
mamba-130m-hf for all three dtypes, monotone weight progress + valid
post-training inference. `hf_full_cycle` — load → infer → train →
re-infer. `hf_batch_parity` — cross-batch logit parity on real weights
(CPU ↔ GPU f32 20/20 exact; bf16 20/20 KL ≤ 3e-6). `extreme_edge_coverage`
— batch=16/32, 1024-token generation stability, M3 training at
T=512/1024. `stability_stress` — CUDA Graph replay determinism, training
repeatability across independent trainer instances. `cpu_gpu_train_parity`
— M1 CPU vs GPU backward parity at f32. `coverage_gaps::a_log_actually_reaches_ssm_after_training`
— regression guard for the `a_neg` staleness fix.
`backward_mixed_parity::backbone_grad_parity_multi_layer_{bf16,f16}` and
`trainer_smoke::m{1,3}_trainer_multi_layer_bf16` — multi-layer
(`n_layers = 3`) parity coverage.

### Performance summary

`state-spaces/mamba-130m-hf` on RTX 6000 Ada with CUDA Graph:

| | f32 | bf16 | f16 |
|---|---:|---:|---:|
| Decode tok/s (B=1)                  | 725   | **898**   | 899   |
| Training step (B=1, T=32) µs        | 1 640 | 1 120     | 1 110 |
| 30-step real-checkpoint convergence | 3.8 s | 2.9 s     | 3.1 s |
| Cross-batch KL (B=1 vs B=32)        | —     | **2e-11** | 2e-11 |

Weight VRAM: bf16 / f16 = 0.50 × f32.

### Notes

No public Mamba-3 SISO HuggingFace checkpoint exists yet — the M3 LM
wrapper drives synthetic weights for now. When a real M3 checkpoint
lands, the HF loader becomes a key remapper on top of the existing
safetensors path, no pipeline changes.

`m3_compute_abg` and `m3_angle_dt_fwd_*` stay f32 by design; the
RoPE angle accumulator stays f64. Both match `state-spaces/mamba/mamba3.py`.

Pure-f32 inference and pure-f32 training are byte-unchanged from 0.2.x,
regression-guarded by `test_gpu_f32_backbone_unchanged_after_mixed_refactor`
on both archs.

## 0.2.1

### Fixed

- **Mamba-3 RoPE angle accumulation precision**: upcast angle accumulator to f64 for addition and modulo wrap, then back to f32 for sin/cos. Prevents drift over long inference sequences (390+ steps). Applied to CPU inference, CPU training forward, and all 3 GPU CUDA angle kernels (`angle_dt_fwd`, `m3_angle_dt_fwd_batch`, `m3_angle_dt_fwd_seq`). *(Provenance correction, 0.6: upstream `mamba3.py` is f32 throughout — the f64 accumulator is a deliberate mamba-rs DEVIATION for long-prefill accuracy, applied consistently CPU↔GPU; it does not mirror an upstream fix.)*

## 0.2.0

**Mamba-3 SISO** — full implementation with CPU + GPU inference/training, CUDA Graph, 47 kernels.

### Added

- `mamba3_siso` module: complete Mamba-3 SISO (Lahoti et al., ICLR 2026)
- CPU inference with BLAS matvec + SIMD SSM recurrence (pulp)
- CPU training: 7-phase forward (F1-F7) + 8-phase BPTT backward (B1-B8)
- GPU inference: `GpuMamba3Backbone` with CUDA Graph capture (~1.6x speedup)
- GPU training: `gpu_forward_mamba3_backbone` + `gpu_backward_mamba3_backbone`
- 47 CUDA kernels across 5 .cu files (SSM, chunked scan, ops, norms, elementwise)
- `Mamba3GpuInferenceEngine` with `disable_event_tracking()` for graph capture stability
- GPU weight upload: `GpuMamba3WeightsInf::from_cpu()` (flat buffer + WeightSlice)
- GPU training weights: `GpuMamba3Weights::from_cpu()` + `GpuMamba3Grads::new()`
- Parallel batch training via Rayon with thread-local scratch + epoch zeroing
- `Mamba3Config` with full validation (headdim, d_state, ngroups, RoPE, a_floor)
- 4 persistent states (SSM + K + V + angle) per layer
- Safetensors serialization (save/load)
- 25 integration tests (9 finite-diff gradient checks, correctness, stability)
- `pulp` dependency for SIMD vectorized SSM recurrence
- `ops/norms.rs`: shared RMSNorm, BCNorm, RMSNormGated

### Fixed

- `m3_dqkv` chunked backward: Part 2 read overwritten shared memory (ssm_sm held d_state instead of SSM_States)
- CPU inference/training parity: softplus uses std `f32::ln()`, sin_cos uses `f32::sin_cos()`
- CPU backward angle reconstruction: `angle_state_init` parameter for correct RoPE gradients with burn-in
- GPU inference: kernel arg order fixes (m3_step_fwd, m3_angle_dt, m3_split)
- GPU training: chunk_size scratch buffers use dims.chunk_size() (64) not hardcoded 16

### Architecture (Mamba-3 SISO vs Mamba-1)

| Feature | Mamba-1 | Mamba-3 SISO |
|---------|---------|-------------|
| Conv1d | Yes | No |
| A matrix | Fixed | Input-dependent per-head |
| Integration | Exponential | Trapezoidal |
| RoPE | No | Per-head angles [0, 2pi) |
| B/C | Single d_state | Multi-head + BCNorm |
| D | Per-channel | Per-head |
| Parallel scan | T>128 | T>64 (chunk_size=64) |

---

## 0.1.4

Extended GPU architecture support for all modern NVIDIA GPUs.

### Added

- SM 120 support: Blackwell consumer (RTX 5090, RTX 5080, RTX 5070)
- SM 61 support: Pascal consumer (GTX 1080, GTX 1070)
- SM 60 support: Pascal datacenter (P100)
- Future-proof fallback: GPUs with compute capability > 12.x automatically use sm_120

### Changed

- `nvrtc_arch()` now covers SM 60 through SM 120 (Pascal → Blackwell, 10 years of GPUs)
- Unknown future architectures (cc > 12) fall back to latest known (sm_120) instead of ancient sm_70

---

## 0.1.3

CPU performance, GPU architecture refactor, parallel training, bug fixes.

### Added

- Cephes degree-7 polynomial `fast_exp` with NEON (AArch64) and AVX2+FMA (x86_64) SIMD
- Pre-computed `a_neg = -exp(a_log)` at weight load time (eliminates 12K+ exp() per inference step)
- Batch `da_buf` + `fast_exp_inplace` in SSM inner loop for SIMD vectorization
- Apple Accelerate framework BLAS dispatch (`accelerate` feature, macOS AMX coprocessor)
- `gemm` crate BLAS dispatch (`gemm-blas` feature, AVX2/AVX-512/NEON microkernels)
- Rayon parallel batch inference (`mamba_step_batch`) with automatic threshold
- Parallel training forward + backward with thread-local gradient accumulation
- CPU training benchmark (B=1 sequential + parallel B=16/64/128)
- GPU parallel prefix scan for long sequences (T>256): warp shuffle scan, single-pass Y accumulation, chunked processing with inter-chunk carry
- `GpuMambaTrainWeights`: per-tensor GPU weight storage for training (industry standard)
- `GpuCtx::disable_tf32()` for exact f32 parity testing

### Changed

- GPU training weights: per-tensor `GpuBuffer` allocation (`GpuMambaTrainWeights`), matching PyTorch convention
- GPU inference weights: flat buffer + `WeightSlice` views (`GpuMambaWeights`), optimized for CUDA Graph capture
- GPU gradients: flat buffer + `GradSlice` views (`GpuMambaGrads`), single memset zeros all
- Activation kernels: scalar dispatch (safe for any buffer size)
- Backward parity test: PyTorch-style `allclose(atol + rtol * |expected|)` tolerance

### Fixed

- GPU backward gradient buffer synchronization
- NVRTC compile options aligned with production configuration
- Parallel backward gradient accumulation (epoch-based lazy zeroing)
- CPU backward scratch buffer size for `mamba_input_dim < d_model` configurations
- `gemm` crate: disabled unused f16 sub-crate (ARM Grace lacks `fullfp16` NEON)

### Performance (GH200 Grace, 72 cores)

**CPU Inference (T=1, B=1):**

| Config | Before | After | Speedup |
|--------|--------|-------|---------|
| small (64, 2L) | 61 us | **21 us** | 2.9x |
| default (128, 3L) | 377 us | **76 us** | 5.0x |
| medium (256, 4L) | 2.2 ms | **261 us** | 8.4x |
| large (512, 6L) | 13.6 ms | **1,226 us** | 11.1x |

**Parallel Training (default config, T=32, 72 cores):**

| Batch | Total | Samples/sec | vs sequential |
|-------|-------|-------------|---------------|
| B=16 | 17.6 ms | 910 | 12.6x |
| B=64 | 26.0 ms | 2,458 | 34.3x |
| B=128 | 40.2 ms | **3,183** | **44.3x** |

## 0.1.2

GPU inference engine, CUDA Graph support, comprehensive test suite.

### Added

- GPU inference engine (`GpuMambaBackbone`) with step/reset API
- CUDA Graph capture — 1.4x speedup on inference (115 us vs 155 us on H100)
- 3-level modular API: `mamba_layer_step` / `mamba_block_step` / `mamba_step`
- safetensors serialization (HuggingFace compatible, cross-framework)
- Batch CPU inference (B>1)
- Sequence forward (`forward_sequence`) for T>1
- Parallel prefix scan CUDA kernels (warp shuffle, for long sequences)
- Flat weight/gradient buffers with WeightSlice/GradSlice zero-cost views
- `GpuBuffer::copy_from_raw` — graph-safe D2D copy via `cuMemcpyDtoDAsync`
- Separate `gpu_input` buffer for input_dim != d_model
- 26 correctness tests covering CPU inference, GPU parity, training backward, serialization
- Full benchmark suite: GPU/CPU inference + training on GH200 and RTX 6000 Ada

### Fixed

- softplus backward kernel dispatch correctness
- CUDA Graph stream capture isolation (graph-safe kernel dispatch with cached device pointers)
- RmsNorm backward gradient validated against PyTorch source

### Performance (default config: d_model=128, 3 layers, 366K params)

- GPU inference B=1: 155 us (GH200), 124 us (RTX 6000 Ada)
- GPU inference + CUDA Graph B=1: 115 us (GH200), 79 us (RTX 6000 Ada)
- GPU training fwd+bwd T=32: 2.3 ms (GH200), 1.7 ms (Ada) — 5.5-8.6x vs CPU
- CPU inference B=1: 377 us (Grace ARM), 348 us (Xeon)

## 0.1.1

Minor fixes.

- Remove dead GPU BLAS functions (unused GpuBuffer variants)
- Remove unused multi-GPU infrastructure (GpuTopology, detect_topology, peer access)
- Remove no-op set_blas_threads and all call sites
- Remove unused Conv1dDims.batch field
- Clean up internal comments

## 0.1.0

Initial release.

- CPU inference: zero-allocation single-step recurrent forward pass
- CPU training: batched forward + backward with BPTT through SSM state
- Burn-in support for recurrent state warming
- CUDA GPU backend with custom kernels (SSM, conv1d, RMSNorm, fused ops)
- Flat contiguous weight/gradient buffers for optimizer fusion
- Rayon-parallel batch processing
- Mamba-specific weight initialization (A_log, dt_proj from paper Section 3.5)
