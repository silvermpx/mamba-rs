# Plan: LLM Support for mamba-rs (M1 + M3)

## Goal
Make mamba-rs usable for LLM inference (text generation) with HuggingFace pretrained Mamba checkpoints while keeping the RL backbone API untouched.

## Architecture

### Core: enum dispatch (not generic trait)
```rust
pub enum AnyBackbone {
    M1(MambaBackbone),
    M3(Mamba3Backbone),  // new wrapper
    // M2(...) — Phase 2
}

impl AnyBackbone {
    fn forward_step(&self, input: &[f32], output: &mut [f32], state: &mut AnyState, scratch: &mut AnyScratch);
    fn forward_sequence(&self, inputs: &[f32], outputs: &mut [f32], state: &mut AnyState, scratch: &mut AnyScratch, seq_len: usize);
    fn d_model(&self) -> usize;
    fn alloc_state(&self) -> AnyState;
    fn alloc_scratch(&self) -> AnyScratch;
}
```

Note: `&self` not `&mut self` — both M1 and M3 take state as a separate `&mut` argument.
The caller (MambaLM) owns `AnyState` + `AnyScratch` + a `temporal: Vec<f32>` working buffer.

For M3: `mamba3_step(temporal, input, scratch, weights, states, cfg)` uses `temporal` as both input and output (residual connections between layers). The M3 variant of `forward_step` maps `input → temporal` via input_proj, then delegates to `mamba3_layer_step` per layer. The caller's `temporal` buffer persists across tokens (same pattern as existing M3 RL inference).

```rust
pub enum AnyState {
    M1(MambaState),
    M3(Mamba3State),
}
impl AnyState {
    pub fn reset(&mut self);  // delegates to MambaState::reset() / Mamba3State::reset()
}
pub enum AnyScratch {
    M1(MambaStepScratch),
    M3(Mamba3StepScratch),
}
```

Why enum not trait: runtime model selection from config.json requires dispatch without generics. Associated types in trait block `dyn` dispatch. Enum is simpler and sufficient for 3 variants.

### MambaLM (language model wrapper)
```rust
pub struct MambaLM {
    backbone: AnyBackbone,
    state: AnyState,              // persistent recurrent state
    scratch: AnyScratch,          // reusable inference scratch buffers
    temporal: Vec<f32>,           // [d_model] working buffer (persists across tokens for M3 residual)
    embed: Vec<f32>,              // [vocab_size * d_model]
    lm_head: Option<Vec<f32>>,    // None = tied with embed
    logits: Vec<f32>,             // [vocab_size] reusable logits buffer
    vocab_size: usize,
    d_model: usize,
}

impl MambaLM {
    pub fn from_hf(dir: &Path) -> Result<Self, String>;
    pub fn generate(&mut self, prompt: &[u32], params: &SampleParams) -> Vec<u32>;
    pub fn generate_streaming(&mut self, prompt: &[u32], params: &SampleParams, cb: impl FnMut(u32, &str));
    pub fn save_state(&self) -> AnyState;     // clone current recurrent state (for multi-turn / prefix caching)
    pub fn restore_state(&mut self, s: AnyState); // restore saved state (skip prefill on shared prefix)
    pub fn reset(&mut self);                  // zero state for new conversation
}
```

Weight tying: when `lm_head` is None, `logits[v] = dot(embed[v*d_model..(v+1)*d_model], hidden)` for each v in 0..vocab_size. This is a standard gemv `[vocab_size × d_model] @ [d_model]` in row-major, NOT a transposed gemv. No memory duplication.

### Sampling
```rust
pub struct SampleParams {
    pub temperature: f32,       // <=0 = argmax
    pub top_k: usize,           // 0 = disabled
    pub top_p: f32,             // 1.0 = disabled
    pub min_p: f32,             // 0.0 = disabled; llama.cpp default 0.05. Filters tokens with prob < min_p * max_prob.
    pub repetition_penalty: f32, // 1.0 = disabled
    pub max_tokens: usize,
    pub eos_token_ids: Vec<u32>, // multiple EOS tokens (HF config: eos_token_id can be int or list)
    pub stop_strings: Vec<String>, // stop generation when decoded output contains any of these
    pub seed: u64,
}
```

Implementation (matches HuggingFace `transformers` order):
1. Repetition penalty (CTRL paper, Keskar et al. 2019): for each seen token, `if logit > 0 { logit /= penalty } else { logit *= penalty }`. Asymmetric — positive logits shrink, negative logits grow more negative. Both directions reduce probability of already-generated tokens.
2. Temperature: `logits /= T` (special-case T<=0 → argmax, skip steps 3-6)
3. TopK: `select_nth_unstable_by` O(n) partial sort, then sort k winners descending. Set logits outside top-k to `-f32::INFINITY`.
4. TopP: softmax the surviving logits (after topK filter), walk sorted probs descending, cumulative sum. Once cumsum > top_p, set remaining logits to `-f32::INFINITY`.
5. MinP: `threshold = min_p * max(probs)`, set all probs < threshold to 0. Applied after topP. (arXiv 2407.01082)
6. Re-normalize surviving probabilities (sum to 1.0).
7. Weighted random sample (xoshiro256++ — 4×u64 state, passes BigCrush, supports jump() for future batch generation. ~15 lines, no rand dep in lib)

Bounds check: `embed_lookup` must assert `token_id < vocab_size` with a descriptive error. Prompt tokens validated before prefill loop.

### Generation loop
```
// MambaLM owns: backbone, state, scratch, temporal, embed, logits

prefill:
    state.reset()
    for token_id in prompt:
        hidden = embed_lookup(embed, token_id)  // [d_model] — stack local
        backbone.forward_step(hidden, temporal, state, scratch)  // temporal updated in-place
    logits = lm_head(temporal)  // [vocab_size]

decode:
    decoded_text = String::new()  // for stop_strings matching
    loop:
        next_token = sample(logits, params)
        if params.eos_token_ids.contains(next_token) || count >= max_tokens: break
        piece = tokenizer.decode(next_token)  // TokenOutputStream buffers partial UTF-8
        decoded_text.push_str(piece)
        if params.stop_strings.any(|s| decoded_text.ends_with(s)): break
        cb(next_token, piece)  // streaming callback
        hidden = embed_lookup(embed, next_token)  // [d_model] — stack local
        backbone.forward_step(hidden, temporal, state, scratch)
        logits = lm_head(temporal)
```

For M1: `forward_step` does `input_proj(hidden) → temporal`, then layers read/write `temporal` via state.
For M3: `forward_step` does `input_proj(hidden) → temporal`, then `mamba3_layer_step(temporal, ...)` per layer (temporal carries residual between layers, same as existing M3 inference).
Both: after all layers, `norm_f(temporal) → temporal`. Output is always in `temporal`.

Note: `forward_sequence()` for efficient parallel prefill is Phase 1b (after basic step-by-step works).

---

## HF Loading

### Two config formats to support
1. **HF-native** (`-hf` models): `hidden_size`, `num_hidden_layers`, `state_size`, `conv_kernel`, `expand`, `vocab_size`, `time_step_rank`
2. **Original** (mamba_ssm): `d_model`, `n_layer`, `ssm_cfg.d_state`, `d_conv`, `expand`, `vocab_size`

Field mapping (HF → MambaConfig):
- `hidden_size` → `d_model`
- `num_hidden_layers` → `n_layers`
- `state_size` → `d_state`
- `conv_kernel` → `d_conv`
- `expand` → `expand` (same name, no default — must be parsed)
- `vocab_size` → used for embedding allocation
- `time_step_rank` → **assert** equals `d_model.div_ceil(16)`, return `Err` on mismatch (MambaConfig computes dt_rank from d_model; mismatch = corrupted/custom checkpoint)
- `tie_word_embeddings` → primary signal for weight tying (default `true` for Mamba). Key-absence check is secondary guard.
- `use_bias` → whether in_proj/out_proj have bias (default `false`). If `true`, load `.bias` keys too.
- `use_conv_bias` → conv1d bias (default `true`). Already handled by current remap table.
- `residual_in_fp32` → moot for f32 CPU path, but parse and document for future bf16 Phase 3.

Auto-detect: if `model_type` field exists → HF-native. If `d_model` field exists → original.

### ModelFamily enum
```rust
pub enum ModelFamily { Mamba1, Mamba2, Mamba3 }
```
Detected from: `model_type` = "mamba" → M1, "mamba2" → M2, "falcon_mamba" → M1.

### Key remapping (M1 HF → native)
```
backbone.layers.{i}.mixer.in_proj.weight  → layers.{i}.in_proj.weight
backbone.layers.{i}.mixer.conv1d.weight   → layers.{i}.conv1d.weight  (HF shape [d_inner, 1, d_conv] 3D → stored as [d_inner, d_conv] 2D, raw bytes identical)
backbone.layers.{i}.mixer.conv1d.bias     → layers.{i}.conv1d.bias
backbone.layers.{i}.mixer.x_proj.weight   → layers.{i}.x_proj.weight
backbone.layers.{i}.mixer.dt_proj.weight  → layers.{i}.dt_proj.weight
backbone.layers.{i}.mixer.dt_proj.bias    → layers.{i}.dt_proj.bias
backbone.layers.{i}.mixer.A_log           → layers.{i}.a_log  (case change!)
backbone.layers.{i}.mixer.D              → layers.{i}.d_param  (D in HF, d_param in Rust)
backbone.layers.{i}.mixer.out_proj.weight → layers.{i}.out_proj.weight
backbone.layers.{i}.norm.weight           → layers.{i}.norm.weight
backbone.norm_f.weight                    → norm_f.weight
backbone.embeddings.weight                → [embed matrix, not backbone] (plural "embeddings", not "embedding")
lm_head.weight                            → [lm_head, if present] (ABSENT from safetensors for tied-weight models — completely missing, not zero)
```

Note: `A_log` and `D` have **no** `.weight` suffix in the safetensors file — they are `nn.Parameter`, not `nn.Linear`.
Weight tying detection: if `lm_head.weight` key is absent from the safetensors file(s), weights are tied → `lm_head = None`, use `embed^T`.

### Config validation difference (M1 vs M3)
`MambaConfig::validate(&self)` returns `Result<(), String>` — call it at load time and propagate the `Err`. No `catch_unwind` needed.
`Mamba3Config::validate(&self)` **panics** on invalid config (returns `()`, uses `assert!`).
The HF loader must validate M3 constraints manually in `config_json.rs` (check headdim power-of-2, d_inner % headdim == 0, etc.) and return `Err` before ever calling `Mamba3Config::validate()`. This avoids panic-as-control-flow.

### bf16 handling
Upcast at load time via `half` crate's `HalfFloatSliceExt` trait:
```rust
use half::slice::HalfFloatSliceExt;
// raw_bytes: &[u8] from safetensors tensor data
let bf16_slice: &[bf16] = bytemuck::cast_slice(raw_bytes);
let mut f32_buf = vec![0.0f32; bf16_slice.len()];
bf16_slice.convert_to_f32_slice(&mut f32_buf);
```
`convert_to_f32_slice` is a trait method on `&[bf16]` (from `HalfFloatSliceExt`), NOT a standalone function. It uses SIMD when available.
Store as f32 internally. Compute in f32.
Add `half` v2 + `bytemuck` as optional deps under `hf` feature.

### Multi-shard
Parse `model.safetensors.index.json` → `weight_map: HashMap<String, String>` (key → shard_file).
Group by shard, mmap each once via `memmap2::Mmap`, call `SafeTensors::deserialize(&mmap)`.
For each tensor: validate shape against config before reading data (safetensors header has shapes — free pre-flight check).

### Pre-flight validation (before any weight allocation)
1. Parse safetensors header (JSON at file start) → get all tensor names, shapes, dtypes
2. Verify every required key exists for the given config (missing key → clear error: "missing weight layers.5.dt_proj.bias")
3. Compute total model size in bytes from shapes+dtypes → compare to available RAM
4. If insufficient: print `"Error: model requires ~{size}GB RAM. Available: ~{free}GB. Use a smaller model."` and return Err
5. Shape validation: for each weight, verify `tensor.len() == expected_elements * dtype_bytes` before materializing

### Vocab padding
Pad `vocab_size` to nearest multiple of 64: `vocab_size_padded = (vocab_size + 63) & !63`.
Allocate embed as `[vocab_size_padded * d_model]` with tail zeroed. Logits output clamped to `[..vocab_size]`.
Free 1-3% BLAS speedup (aligned dimensions for SIMD/microkernel tiling).

### input_proj bypass
HF models have no input_proj (embedding output = d_model directly).
Add `MambaBackbone::from_weights_no_proj(cfg, weights)` — dedicated constructor that:
- Sets `input_dim = cfg.d_model`
- Sets `identity_proj = true` (flag on MambaBackbone struct)
- **Skips `weights.validate()` entirely** — the existing `validate()` checks `input_proj_w.len() == input_dim * d_model`, which fails for empty vec. Instead, `from_weights_no_proj` does its own targeted validation: checks only layer weights (norm, in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj) and norm_f. No changes to `validate()` itself.
- **Calls `lw.compute_a_neg()` for every layer** — `a_neg` is a pre-computed cache of `-exp(a_log)`. The SSM recurrence reads `a_neg` directly, never `a_log` at inference time. Without this call, `a_neg` stays zero → all decay factors = exp(0) = 1.0 → broken SSM. This is the most dangerous silent-corruption gap.

The identity_proj branch must live inside `mamba_step()` itself (the free function in `mamba_ssm/cpu/inference.rs`), NOT in `MambaBackbone::forward_step`. Reason: `forward_step` delegates to `mamba_step` which unconditionally calls `matvec_with_bias(&mut output, input, &weights.input_proj_w, ...)`. A flag on `forward_step` alone does nothing.

Implementation: add `identity_proj: bool` parameter to `mamba_step()`:
```rust
if identity_proj {
    output[..d_model].copy_from_slice(&input[..d_model]);
} else {
    matvec_with_bias(&mut output[..d_model], input, &weights.input_proj_w, ...);
}
```

Same change needed in `mamba_step_batch()` — it also calls the matmul unconditionally. `forward_step_batch` must pass the flag through.

`forward_sequence` delegates to `forward_step` in a loop → inherits the fix automatically.

Original `from_weights(cfg, weights)` unchanged — RL API not affected. Existing callers pass `identity_proj=false`.
`MambaWeights.input_proj_w` and `input_proj_b` are empty vecs for HF path (never accessed).

---

## Real HF Model Compatibility Matrix

| Model | Family | Config | Dtype | Shards | Tied | Tokenizer |
|-------|--------|--------|-------|--------|------|-----------|
| mamba-130m-hf | M1 | HF | f32 | 1 | yes | GPTNeoX |
| mamba-370m-hf | M1 | HF | f32 | 1 | yes | GPTNeoX |
| mamba-790m-hf | M1 | HF | f32 | 2 | yes | GPTNeoX |
| mamba-1.4b-hf | M1 | HF | f32 | 2 | yes | GPTNeoX |
| mamba-2.8b-hf | M1 | HF | f32 | 3 | yes | GPTNeoX |
| falcon-mamba-7b | M1 | HF | bf16 | 3 | NO | Custom |
| Codestral-7B | M2 | HF | bf16 | 3 | NO | SentencePiece |
| mamba2-130m | M2 | orig | f32 | legacy .bin format | yes | GPTNeoX |

No M3 SISO models on Hub yet. Legacy `.bin` format is **out of scope** for v1 — only safetensors supported. Convert to safetensors offline if needed.

---

## Files

### New files (10):
```
src/hf/mod.rs           — re-exports, ModelFamily enum
src/hf/load.rs          — load_hf() top-level, shard discovery
src/hf/bf16.rs          — bf16_bytes_to_f32()
src/hf/config_json.rs   — parse both HF-native and original formats
src/hf/keys.rs          — remap_key() with per-family dispatch
src/hf/embed.rs         — HfEmbedWeights, embed_lookup, lm_head_logits
src/module/lm.rs        — MambaLM, AnyBackbone enum, generate()
src/module/sample.rs    — SampleParams, sampling logic
src/module/backbone3.rs — Mamba3Backbone wrapper (mirrors MambaBackbone for M3, owns weights + config, caller owns temporal buffer + state + scratch)
src/bin/generate.rs     — CLI binary
```

### Modified files (5):
```
Cargo.toml                     — features hf/cli (cli implies hf), deps half/bytemuck/memmap2/serde/serde_json/clap/tokenizers/hf-hub
src/lib.rs                     — #[cfg(feature = "hf")] pub mod hf
src/module/mod.rs              — pub mod lm, sample, backbone3
src/module/backbone.rs         — add identity_proj: bool field, from_weights_no_proj() (skips validate, calls compute_a_neg, sets identity_proj=true)
src/mamba_ssm/cpu/inference.rs — add identity_proj: bool param to mamba_step() and mamba_step_batch(), branch before input_proj matvec
```

### Dependencies:
| Dep | Feature | Purpose |
|-----|---------|---------|
| half | hf | bf16 → f32 |
| bytemuck | hf | zero-copy cast |
| memmap2 | hf | mmap safetensors (large LLM weights, library-level) |
| serde_json | hf | config.json parsing (full HF configs have optional/nested fields — hand-rolled parser is too fragile) |
| serde | hf | deserialize config structs |
| clap | cli | arg parsing |
| tokenizers | cli | BPE tokenization (requires C compiler for `onig_sys`; document in README) |
| hf-hub | cli | auto-download |

Feature wiring:
```toml
[features]
hf  = ["dep:half", "dep:bytemuck", "dep:memmap2", "dep:serde", "dep:serde_json"]
cli = ["hf", "dep:clap", "dep:tokenizers", "dep:hf-hub"]  # cli implies hf
```

Library (no features) stays zero new deps. RL users unaffected.
`tokenizers` crate pulls in `onig_sys` (C build). Document in README: `libclang-dev` or Xcode CLT required.

---

## Build Sequence

### Phase 1a — HF loading + MambaLM (1 week)
- [ ] `hf/` module: bf16, config parse, key remap, multi-shard, embed weights
- [ ] `backbone3.rs`: Mamba3Backbone wrapper
- [ ] `AnyBackbone` enum with forward_step
- [ ] `MambaLM::from_hf()` + `generate()` (step-by-step prefill)
- [ ] `sample.rs`: SampleParams + all sampling logic
- [ ] Integration test: load synthetic HF-format checkpoint, generate 10 tokens
- [ ] Test: verify weight tying (no lm_head.weight → use embed^T)

### Phase 1b — CLI binary (2-3 days)
- [ ] `bin/generate.rs` with clap, tokenizers, hf-hub
- [ ] `ApiBuilder::from_env().with_progress(true)` — download progress bar + HF_TOKEN for gated models
- [ ] Print cache path on first download: `"Downloading to ~/.cache/huggingface/hub"`
- [ ] Pre-flight RAM check: estimate model size from safetensors header, warn if insufficient
- [ ] TokenOutputStream: buffer partial UTF-8 bytes before flushing to stdout (non-ASCII fix)
- [ ] Streaming output: `print!("{piece}")` + `stdout().flush()` per token
- [ ] Metrics: `{prompt_tokens} prompt, prefill {ttft}ms | {n} tokens at {tok_s} tok/s | loaded in {load_s}s`
- [ ] Test: `cargo run --features cli --bin mamba-generate -- --model-id state-spaces/mamba-130m-hf "Hello"`

### Phase 1c — Parallel prefill (3 days)
- [ ] `forward_sequence()` on AnyBackbone — reuse existing `mamba_ssm/cpu/parallel.rs` parallel scan for M1
- [ ] This is a TRUE parallel associative scan (O(log T) depth via rayon), not just a loop of forward_step
- [ ] Critical: after parallel scan, extract final conv_state + ssm_state into AnyState for decode phase
- [ ] The conv_state (shift register) must be populated from last `d_conv-1` tokens of prompt — SSM state comes from scan naturally, conv does not
- [ ] Benchmark: 1000-token prompt prefill time (expect 5-40x vs step-by-step depending on T and core count)

### Phase 2 — Mamba-2 (2 weeks, separate milestone)
- [ ] Mamba2Backbone wrapper
- [ ] M2 key remap (dt_bias, no x_proj, mixer.norm)
- [ ] M2 config parsing
- [ ] Add M2 variant to AnyBackbone enum
- [ ] Test: load Codestral-7B

### Phase 3 — Optimizations (ongoing)
- [ ] Native bf16 compute (store bf16, compute f32 in hot loops)
- [ ] GPU generation path
- [ ] Batch generation
- [ ] Gradient checkpointing for fine-tuning

---

## v1 Scope Limits (explicit)
- CPU inference only (GPU deferred to Phase 3)
- Single sequence generation (batch deferred)
- M1 HF models only (M2/M3 Phase 2+)
- f32 compute (bf16 upcast at load, native bf16 Phase 3 — will 2x throughput for bf16 models)
- Safetensors only (legacy .bin format out of scope)
- No fine-tuning (inference only for v1)
- No chat templates (base models only in v1 — instruction-tuned template support Phase 2)
- No quantization (GGUF/GPTQ deferred to Phase 3+)

## Design requirements
- `#[derive(Clone)]` on `MambaState`, `Mamba3State`, `AnyState`, `AnyScratch` — enables state forking for future prefix caching / speculative decoding / beam search at zero v1 cost
- All state types must have `.reset()` method
- `MambaLM` exposes `save_state()` / `restore_state()` for multi-turn conversations without re-prefill

## Performance targets (f32 CPU, memory-bandwidth bound)

| Model | f32 size | M2 Pro (200 GB/s) | Desktop x86 (60 GB/s) |
|-------|----------|-------------------|----------------------|
| 130m  | 0.5 GB   | ~200 tok/s        | ~60 tok/s            |
| 370m  | 1.5 GB   | ~70 tok/s         | ~20 tok/s            |
| 1.4b  | 5.6 GB   | ~18 tok/s         | ~5 tok/s             |
| 2.8b  | 11.2 GB  | ~10 tok/s         | ~3 tok/s             |

These are 50-60% of theoretical bandwidth ceiling. Mamba has no KV cache overhead — should be faster per-param than equivalent Transformer.

---

## Test Plan

All tests run in `cargo test --features hf` with zero network access. No real model downloads in CI (same pattern as candle). Real model tests gated behind `#[ignore]`.

### Unit tests — sampling (`src/module/sample.rs`, ~10 tests)
All use synthetic 4-8 element logit vectors + seeded RNG:
```
test_greedy_argmax                          // T<=0 → always top logit
test_temperature_scaling                    // T=2.0 shifts distribution toward uniform
test_topk_k1_is_argmax                      // topK=1 always returns max
test_topk_k2_excludes_rest                  // topK=2, low-prob tokens never sampled (10K trials)
test_topp_filters_tail                      // top_p=0.5, verify tail excluded
test_minp_filters_by_fraction               // min_p=0.1, threshold = 0.1 * max_prob
test_repetition_penalty_positive_divides    // logit > 0: logit /= penalty
test_repetition_penalty_negative_multiplies // logit < 0: logit *= penalty
test_deterministic_with_seed                // same seed → same token sequence
test_different_seed_different_output         // seed=42 vs seed=99 → different sequence
```

### Unit tests — bf16 conversion (`src/hf/bf16.rs`, 4 tests)
```
test_bf16_zero_roundtrip          // [0.0] → f32 = 0.0
test_bf16_known_values            // [1.0, -1.0, 0.5] byte patterns → exact f32
test_bf16_nan_inf_passthrough     // NaN/Inf bf16 → correct f32 special values
test_bf16_simd_matches_scalar     // convert_to_f32_slice matches element-by-element loop
```

### Unit tests — config.json parsing (`src/hf/config_json.rs`, 7 tests)
Feed raw JSON strings, assert parsed struct fields:
```
test_parse_hf_native_config       // hidden_size, num_hidden_layers, state_size, conv_kernel, expand
test_parse_original_config        // d_model, n_layer, ssm_cfg.d_state, d_conv
test_auto_detect_format           // model_type="mamba" → HF; d_model present → original
test_model_family_dispatch        // "mamba"→M1, "mamba2"→M2, "falcon_mamba"→M1
test_time_step_rank_mismatch_err  // dt_rank != d_model.div_ceil(16) → Err
test_missing_field_returns_err    // drop required field → clear error message
test_tie_word_embeddings_default  // absent field → true (HF default)
```

### Unit tests — key remapping (`src/hf/keys.rs`, 6 tests)
```
test_remap_m1_hf_layer_key        // "backbone.layers.3.mixer.in_proj.weight" → "layers.3.in_proj_w"
test_remap_a_log_case             // "A_log" → "a_log"
test_remap_d_to_d_param           // "D" → "d_param"
test_remap_embeddings_plural      // "backbone.embeddings.weight" → embed matrix
test_remap_lm_head_absent_tied    // key absent → lm_head=None
test_remap_conv1d_shape_identity  // HF [d_inner, 1, d_conv] bytes == [d_inner, d_conv] bytes
```

### Integration test — synthetic checkpoint loading (`tests/hf_load.rs`, 3 tests)
Build a minimal safetensors file in tempdir, write config.json:
```
test_load_synthetic_m1_hf_checkpoint
```
Config: d_model=64, n_layers=2, d_state=16, d_conv=4, expand=2, vocab_size=256.
Write all required weight tensors with correct HF key names and shapes (random f32).
Call `MambaLM::from_hf(dir)` → assert Ok.
Verify: d_model==64, vocab_size==256, weight tying detected (no lm_head.weight).
Verify: `a_neg[0] < 0.0` on any layer (proves compute_a_neg was called — catches the silent corruption bug).

```
test_load_synthetic_bf16_checkpoint
```
Same but write bf16 tensors → verify f32 output matches within 1e-3.

```
test_load_synthetic_multishard
```
Write 2 shard files + model.safetensors.index.json → MambaLM::from_hf() succeeds.

### Integration test — generation (`tests/hf_generate.rs`, 3 tests)
Use synthetic checkpoint from above:
```
test_generate_deterministic
```
seed=42, T=0 (greedy), max_tokens=10. Run twice with fresh state → identical sequences.

```
test_generate_different_seed
```
seed=42 vs seed=99, T=0.7 → different sequences.

```
test_state_save_restore_continuity
```
Generate 5 tokens → save_state(). Generate 5 more. Restore → generate 5 more. Second and third 5-token blocks identical.

### Integration test — AnyBackbone dispatch (`tests/any_backbone.rs`, 2 tests)
```
test_m1_through_any_backbone
```
Build M1 MambaBackbone, wrap in AnyBackbone::M1, call forward_step → output matches direct MambaBackbone::forward_step.

```
test_identity_proj_matches_copy
```
Build M1 with identity_proj=true, input=[d_model]. forward_step output == forward_step with a real proj that is an identity matrix.

### Gated tests — real model (`#[ignore]`, manual only)
```
#[ignore = "requires MAMBA_TEST_MODEL_DIR"]
test_mamba_130m_known_tokens
```
Load real mamba-130m-hf, greedy generate from known 5-token prompt, assert first 10 output tokens match pre-recorded golden values from Python mamba_ssm.

### Test count summary
| Category | Tests | Network | Phase |
|----------|-------|---------|-------|
| Sampling unit | 10 | no | 1a |
| bf16 unit | 4 | no | 1a |
| Config parsing unit | 7 | no | 1a |
| Key remap unit | 6 | no | 1a |
| Checkpoint loading integration | 3 | no | 1a |
| Generation integration | 3 | no | 1a |
| AnyBackbone dispatch | 2 | no | 1a |
| Real model golden | 1 | yes | 1b |
| **Total** | **36** | | |

35 tests run in `cargo test --features hf` with zero network. 1 gated behind `#[ignore]`.