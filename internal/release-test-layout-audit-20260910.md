# Release test-layout audit (read-only recommendations)

Snapshot: `6c282b3c2363` on 2026-09-10. This audit did not run Cargo, a GPU,
or the network, and did not change source, manifests, or Git state. The worktree
had four production files modified by another owner and one untracked live WIP
target, `gemm_inference_route_inventory`; that WIP is deliberately outside the
tracked census and is not classified here.

## Census and decision rule

- `git ls-files` reports **253 tracked top-level `tests/*.rs` targets** (the
  earlier note's 253 included the then-uncommitted target; at this snapshot the
  count is still 253, but the live route-inventory target is untracked).
- There are **384 tracked files under `tests/`**, including **97 tracked Rust
  support modules under `tests/support/`** and six `tests/common/*.rs` helpers.
- Source inspection finds 64 targets with no `#[ignore]`, 189 with at least one
  `#[ignore]`; 23 are host-only, 111 have a crate-level CUDA gate, 15 have a
  crate-level HF+CUDA gate, and 104 mix host/source-contract code with CUDA
  arms. `#[ignore]` prevents execution, not compilation.
- The existing `qual/lanes.toml` is useful evidence (gate/contract/record/host),
  but it lists only 161 targets and predates 92 tracked targets. It cannot be
  the post-move source of truth without being generalized to all target roots.

Disposition codes below are based on the code's attributes, imports,
`include_str!`/`#[path]` dependencies, test function purpose, and live/manual
execution contract, not merely names:

| Code | Recommendation | Source evidence |
|---|---|---|
| R | retain under `tests/` | Tests public API behavior, parity, determinism, safety, source/dispatcher contracts, or shipping-kernel correctness. Host arms are CI-capable; CUDA/HF targets are feature-gated. |
| B | move to `benches/` | Primary payload is a small maintained timing instrument (`Instant`/CUDA events, normally ignored), with no release verdict mixed into the target. Convert to a stable `harness = false` bench entry point. |
| Q | move to `tools/qualification/` | Primary payload is explicit manual record/qualification: hardware census, NVRTC/ptxas/SASS, sanitizer, large model/sweep, external asset, or admission timing. Preserve assertions and reproducible invocation. |
| E | move to `internal/experiments/` and exclude from crate | Test-only candidate composition, tournament/scout/once timing, or focused diagnosis. Preserve source and receipts; it is discovery evidence rather than a shipping regression. |
| S | split before moving | Target mixes a public regression with timing/discovery/receipt code. Keep the named regression portion public and route the remainder to B/Q/E. |
| F | focused decision | Source is potentially redundant with a maintained benchmark but lacks enough intent/ownership evidence for blind retirement. Preserve until compared. |

## Per-target matrix (all 253 tracked targets exactly once)

### R — retain as user-facing regressions

These targets predominantly assert public/API behavior, exact/tolerance parity,
failure safety, production source shape, or selector/identity contracts. CUDA or
HF requirements are already explicit in crate-level `cfg` attributes where
needed; ignored live arms remain opt-in regressions rather than record tools.

```text
tests/a9_fused_softplus_repro.rs
tests/adamw_parity.rs
tests/arch_compile_gates.rs
tests/backward_mixed_parity.rs
tests/backward_mixed_smoke.rs
tests/coverage_gaps.rs
tests/cpu_gpu_train_parity.rs
tests/ddp_emulated_equivalence.rs
tests/ddp_fixed_order_reducer.rs
tests/ddp_nccl_two_ranks.rs
tests/ddp_reduction_tree.rs
tests/decode_digest.rs
tests/determinism_source_gates.rs
tests/extreme_edge_coverage.rs
tests/f32_training_graph_parity.rs
tests/fixed_epilogue_pins.rs
tests/gemm_bi_async_allocation.rs
tests/gemm_bi_fixed_bridge.rs
tests/gemm_bi_fixed_correctness.rs
tests/gemm_bi_fixed_sm120_exact_n64.rs
tests/gemm_bi_fixed_sm120_sliced.rs
tests/gemm_bi_fixed_sm89_exact_n64.rs
tests/gemm_bi_fixed_sm89_pipeline.rs
tests/gemm_bi_invariance_matrix.rs
tests/gemm_bi_sm100_contract.rs
tests/gemm_bi_sm120_contract.rs
tests/gemm_bi_sm89_exact_f32_d128_source_contract.rs
tests/gemm_bi_sm89_exact_f32_source_contract.rs
tests/gemm_bi_sm89_half_tn_source_contract.rs
tests/gemm_bi_sm90_contract.rs
tests/gemm_bi_tf32_joint_source_contract.rs
tests/gemm_bi_tf32_selector.rs
tests/gemm_context_routing.rs
tests/gemm_mode_api.rs
tests/gemm_mode_live.rs
tests/gemm_tied_f32_output.rs
tests/gpu_batch_test.rs
tests/gpu_bf16_parity.rs
tests/gpu_inference_prefill_parity.rs
tests/gpu_lm_bf16_test.rs
tests/gpu_lm_test.rs
tests/gpu_mamba3_lm_test.rs
tests/gpu_pooled_prefill.rs
tests/grad_clip.rs
tests/grad_oracle.rs
tests/grad_region_clip.rs
tests/graph_capture_panic.rs
tests/hf_batch_parity.rs
tests/hf_full_cycle.rs
tests/hf_integration.rs
tests/hf_training_convergence.rs
tests/hf_training_smoke.rs
tests/inference_f16_smoke.rs
tests/inference_graph_route.rs
tests/kernel_identity.rs
tests/kernel_identity_cuda.rs
tests/loss_scaler.rs
tests/m1_parallel_bwd_parity.rs
tests/m1_parallel_scan_typed_parity.rs
tests/m1_scan_bwd_hash.rs
tests/m1_test_correctness.rs
tests/m3_backward_mixed_parity.rs
tests/m3_bf16_parity.rs
tests/m3_chunked_fwd_unit_parity.rs
tests/m3_cpu_gpu_train_parity.rs
tests/m3_forward_mixed_parity.rs
tests/m3_gpu_prefill_parity.rs
tests/m3_kernel_identity_cuda.rs
tests/m3_mixed_train_smoke.rs
tests/m3_prefill_full_parity.rs
tests/m3_prefill_typed_parity.rs
tests/m3_test_correctness.rs
tests/m3_training_graph_parity.rs
tests/m3_training_graph_safety.rs
tests/m3_typed_bwd_unit_parity.rs
tests/mixed_prefill_parity.rs
tests/msrv_single_source.rs
tests/optimizer_resume_bitwise.rs
tests/parallel_run_determinism.rs
tests/qual_lane_census.rs
tests/rmsnorm_bf16_768.rs
tests/scan_mode_boundary.rs
tests/sm120_tf32_pipeline_source_gate.rs
tests/sm120_tf32_stack_source_gate.rs
tests/sm80_cp_async_exact_allocation_contract.rs
tests/sm80_tf32_cp_async_source_gate.rs
tests/sm80_typed_cp_async_source_gate.rs
tests/sm89_half_source_freeze.rs
tests/stability_stress.rs
tests/state_cap_invariance.rs
tests/trainer_smoke.rs
tests/trainer_split.rs
tests/trainer_split_m3.rs
tests/training_graph_parity.rs
tests/typed_bwd_parity.rs
```

### B — maintained benchmark set

Each has one or a few explicit timing arms and a bounded, reusable workload.
They should become stable `[[bench]]` targets with `harness = false` (or share a
small local bench runner), rather than continue masquerading as ignored tests.

```text
tests/classifier_gemm_tier_bench.rs
tests/epilogue_bench.rs
tests/m1_cpu_benchmark.rs
tests/m1_gpu_benchmark.rs
tests/m3_cpu_benchmark.rs
tests/m3_gpu_benchmark.rs
tests/m3_scan_micro_bench.rs
tests/thin_rung_decode_bench.rs
```

### Q — explicit reproducible qualification tools

These are predominantly ignored/manual, depend on CUDA/NVRTC/toolchain or
external models, inspect resources/occupancy/SASS, run admission matrices, or
emit measurement records. They remain valuable and public/reproducible; moving
them is isolation, not deletion.

```text
tests/bench_bf16_vs_f32.rs
tests/cublas_compute_probe.rs
tests/device_facts.rs
tests/gemm_bi_contract_census.rs
tests/gemm_bi_fixed_half_m64n128_qualification.rs
tests/gemm_bi_fixed_performance.rs
tests/gemm_bi_fixed_sm120_performance.rs
tests/gemm_bi_fixed_tf32_m64n128_qualification.rs
tests/gemm_bi_half_remaining_qualification_contract.rs
tests/gemm_bi_performance_matrix.rs
tests/gemm_bi_scalar_nn_m64n64_qualification.rs
tests/gemm_bi_scalar_nt_slim_qualification.rs
tests/gemm_bi_sm100_triad_census.rs
tests/gemm_bi_sm120_tf32_selector_qualification.rs
tests/gemm_bi_sm120_tf32_vendor.rs
tests/gemm_bi_sm120_triad_census.rs
tests/gemm_bi_sm89_exact_f32_d128_module_qualification.rs
tests/gemm_bi_sm89_exact_f32_d128_selector_qualification.rs
tests/gemm_bi_sm89_exact_f32_module_qualification.rs
tests/gemm_bi_sm89_exact_f32_tn_selector_qualification.rs
tests/gemm_bi_sm89_half_selector_qualification.rs
tests/gemm_bi_sm89_tf32_joint_module_qualification.rs
tests/gemm_bi_sm89_tf32_joint_selector_qualification.rs
tests/gemm_bi_sm90a_triad_census.rs
tests/gemm_bi_tf32_cohort_binding.rs
tests/gemm_bi_tf32_joint_cohort_binding.rs
tests/gemm_bi_tf32_m16n16_qualification.rs
tests/gemm_bi_tf32_nt_cublas.rs
tests/gemm_bi_tf32_selector_qualification.rs
tests/gemm_bi_tf32_splitk4_qualification.rs
tests/gemm_bi_tn_splitm_cp_async_qualification.rs
tests/gemm_denominator_probe.rs
tests/gemm_ladder_sweep.rs
tests/gpu_lm_profile.rs
tests/m3_prefill_bench.rs
tests/memory_footprint.rs
tests/rl_llm_bench.rs
tests/scan_mode_bench.rs
tests/sm100_tcgen05_census.rs
tests/sm80_cp_async_exact_allocation_sanitizer.rs
tests/sm90_wgmma_census.rs
tests/tcw64_census.rs
tests/trainer_benchmarks.rs
```

### E — preserved discovery/diagnostic evidence, excluded from crate

The source bodies define test-only candidates/adapters, compile-only gates,
resource assertions and scout/once timing policies, or narrowly diagnose an
already-understood failure. Preserve them with their CUDA companions and raw
receipts in `internal/experiments/`; do not auto-build or publish them.

```text
tests/bf16_130m_layer_bisect.rs
tests/bf16_130m_layer_diff.rs
tests/gemm_bi_bf16_nt_m64n192_s3_d768_out_discovery.rs
tests/gemm_bi_bf16_nt_m96n128_s3_d768_out_discovery.rs
tests/gemm_bi_fixed_f32_t256_discovery.rs
tests/gemm_bi_fixed_half_batch_discovery.rs
tests/gemm_bi_fixed_tf32_n96_discovery.rs
tests/gemm_bi_fixed_tf32_w4_discovery.rs
tests/gemm_bi_half_nn_n96_s3_d768_in_discovery.rs
tests/gemm_bi_half_nn_s3_bx4_d768_in_discovery.rs
tests/gemm_bi_half_nt_m64n128_s2_d768_out_discovery.rs
tests/gemm_bi_half_nt_m64n128_s3_groupm8_d768_out_discovery.rs
tests/gemm_bi_half_nt_m64n192_full_domain_d768_out_discovery.rs
tests/gemm_bi_half_nt_m64n192_issue2211_d768_out_discovery.rs
tests/gemm_bi_half_nt_m64n192_s3_d768_out_discovery.rs
tests/gemm_bi_half_nt_m96n128_s3_d768_out_discovery.rs
tests/gemm_bi_half_tn_8warp_16x32_discovery.rs
tests/gemm_bi_half_tn_bk32_s3_regpipe_vec2_bf16_d768_in_discovery.rs
tests/gemm_bi_half_tn_m64n96_regpipe_vec2_bf16_d768_out_discovery.rs
tests/gemm_bi_half_tn_m96n64_regpipe_vec2_bf16_d768_in_discovery.rs
tests/gemm_bi_half_tn_one_wave_atlas_bf16_d768_out_discovery.rs
tests/gemm_bi_half_tn_packed_raster_bf16_d768_in_discovery.rs
tests/gemm_bi_half_tn_regpipe_vec2_bx4_d768_in_discovery.rs
tests/gemm_bi_half_tn_regpipe_vec2_issue_d768_in_discovery.rs
tests/gemm_bi_half_tn_regpipe_vec2_sliced_d768_in_discovery.rs
tests/gemm_bi_half_tn_vec2_exact_entry_discovery.rs
tests/gemm_bi_half_tn_vec2_full_domain_discovery.rs
tests/gemm_bi_half_tn_vec2_full_tile_cg_discovery.rs
tests/gemm_bi_half_tn_vec2_full_tile_stage_discovery.rs
tests/gemm_bi_scalar_nn_m64n128_postproduction_experiment.rs
tests/gemm_bi_scalar_nn_m64n64_bcg_experiment.rs
tests/gemm_bi_scalar_nn_m64n64_exactshape_experiment.rs
tests/gemm_bi_scalar_nn_splitk_m32n128_experiment.rs
tests/gemm_bi_scalar_nn_splitk_m32n64_exactshape_tournament.rs
tests/gemm_bi_scalar_nn_splitk_m64n64_tournament.rs
tests/gemm_bi_scalar_nt_d128_out_m64n64_tournament.rs
tests/gemm_bi_scalar_nt_d768_out_m64n64_tournament.rs
tests/gemm_bi_scalar_nt_d768_transpose_tournament.rs
tests/gemm_bi_scalar_nt_large_deep_m64n64_tournament.rs
tests/gemm_bi_scalar_nt_prism_m64n64_tournament.rs
tests/gemm_bi_scalar_nt_prism_tournament.rs
tests/gemm_bi_scalar_nt_skinny_direct_experiment.rs
tests/gemm_bi_scalar_tn_copyplan_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_copyplan_dual_chunk_fused_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_copyplan_dual_chunk_m128n64_bk16_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_copyplan_dual_chunk_rolled_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_copyplan_fused_finalize_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_copyplan_fused_finalize_groupm12_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_copyplan_fused_finalize_n32_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_direct_d768_in_bk16_retained_dual_discovery.rs
tests/gemm_bi_scalar_tn_direct_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_direct_d768_out_bk16_discovery.rs
tests/gemm_bi_scalar_tn_direct_dual_chunk_fused_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_direct_prism_bk16_discovery.rs
tests/gemm_bi_scalar_tn_direct_prism_bk32_exact_tail_discovery.rs
tests/gemm_bi_scalar_tn_direct_prism_fused_n32_discovery.rs
tests/gemm_bi_scalar_tn_prism_vec4_reducer_discovery.rs
tests/gemm_bi_scalar_tn_splitm_copyplan_siblings_discovery.rs
tests/gemm_bi_scalar_tn_transpose_32x8_d768_in_discovery.rs
tests/gemm_bi_scalar_tn_underfill_direct_experiment.rs
tests/gemm_bi_scalar_wide_microtile_experiment.rs
tests/gemm_bi_tf32_nn_m96n96_s2_copy_lifetime_prism_discovery.rs
tests/gemm_bi_tf32_nn_m96n96_s2_direct_prism_discovery.rs
tests/gemm_bi_tf32_nn_m96n96_s2_single_fragment_prism_discovery.rs
tests/gemm_bi_tf32_nn_n96_b_ldmatrix_discovery.rs
tests/gemm_bi_tf32_nn_n96_cross_bk_prefetch_prism_discovery.rs
tests/gemm_bi_tf32_nn_n96_direct_epilogue_ca_prism_discovery.rs
tests/gemm_bi_tf32_nn_n96_direct_epilogue_discovery.rs
tests/gemm_bi_tf32_nn_n96_direct_epilogue_nobarrier_prism_discovery.rs
tests/gemm_bi_tf32_nn_n96_direct_grid_constant_prism_discovery.rs
tests/gemm_bi_tf32_nn_n96_full_domain_cpasync_prism_discovery.rs
tests/gemm_bi_tf32_nn_rect_wide_next_tournament.rs
tests/gemm_bi_tf32_nn_rect_wide_tournament.rs
tests/gemm_bi_tf32_nt_a_ldmatrix_prism_full_domain_discovery.rs
tests/gemm_bi_tf32_nt_a_ldmatrix_prism_k8_regpipe_discovery.rs
tests/gemm_bi_tf32_nt_compact_a_ldmatrix_sliced_d768_in_discovery.rs
tests/gemm_bi_tf32_nt_compact_a_ldmatrix_sliced_d768_out_discovery.rs
tests/gemm_bi_tf32_nt_compact_a_ldmatrix_sliced_prism_discovery.rs
tests/gemm_bi_tf32_nt_tournament.rs
tests/gemm_bi_tf32_tn_panel8_tournament.rs
tests/gemm_bi_tf32_tn_pre_rna_m64n96_s2_prism_discovery.rs
tests/gemm_bi_tf32_tn_splitm_tournament.rs
tests/gemm_bi_tf32_tn_streamk_tournament.rs
tests/gemm_bi_tf32_tn_transpose_ab_rna_n96_d768_out_discovery.rs
tests/gemm_bi_tf32_tn_transpose_rna_m64n128_s2_prism_discovery.rs
tests/gemm_bi_tf32_tn_transpose_rna_m64n64_d768_out_discovery.rs
tests/gemm_bi_tf32_tn_transpose_rna_m64n64_prism_discovery.rs
tests/gemm_bi_tf32_tn_transpose_rna_n96_d768_out_discovery.rs
tests/gemm_bi_tf32_tn_transpose_rna_n96_siblings_discovery.rs
tests/gemm_bi_tf32_tn_underfill_tournament.rs
tests/gemm_bi_tn_d128_fused_contract_tournament.rs
tests/gemm_bi_tn_narrow_partition_tournament.rs
tests/matvec_probe.rs
tests/sm120_tf32_pair_store_experiment.rs
```

### S — mandatory splits before relocation

```text
tests/gemm_bi_determinism.rs
tests/gemm_bi_scalar_nt_copyplan_siblings_discovery.rs
tests/gemm_bi_tc.rs
tests/gemm_bi_tf32_contract.rs
tests/gemm_bi_tf32_nt_compact_xor.rs
tests/gemm_bi_typed_parity.rs
tests/large_d_state.rs
tests/m3_final_grads_unit_parity.rs
tests/prefill.rs
tests/prefill_m3.rs
tests/prefill_serve_bench.rs
```

Split details:

| Target | Keep public | Move elsewhere |
|---|---|---|
| `gemm_bi_determinism` | five determinism/parity/invariance assertions | `bench_gemm_bi_vs_tf32` to B |
| `gemm_bi_scalar_nt_copyplan_siblings_discovery` | extract the three frozen receipt identity assertions to a small release fixture test | candidate adapters/scout timing to E; raw receipts stay archived and out of package |
| `gemm_bi_tc` | route, bit, numeric, tail and AUTO assertions | three timing/census arms to Q or B |
| `gemm_bi_tf32_contract` | source parser, ABI, export, guard, policy and production contract tests | NVRTC/ptxas/cuobjdump/SASS/hardware-only arms to Q; do not demote the host source gates |
| `gemm_bi_tf32_nt_compact_xor` | extract only the independent accepted-candidate composer/fixture needed by the library unit test | the 77-test candidate tournament and its unused adapters to E |
| `gemm_bi_typed_parity` | retained production route parity/guards | candidate tile screens and qualification planning to Q/E; this also shrinks its 19 support-module imports |
| `large_d_state` | two CPU/GPU parity tests and capacity-range regression | `m1_step_cost_across_state_capacities` to B |
| `m3_final_grads_unit_parity` | typed parity and absolute hash gates | `m3_kernels_isolated_bench` to B |
| `prefill` | five CPU prefill contracts | ignored classifier-shape timing arm to B |
| `prefill_m3` | four M3 prefill contracts | ignored M1-vs-M3 timing arm to B |
| `prefill_serve_bench` | absolute output hashes (a stated release gate) | headline production-shape timing to Q |

### F — resolve by focused duplicate comparison

```text
tests/gpu_bench_all.rs
tests/gpu_lm_bench.rs
```

Both are short HF+CUDA ignored timing targets with hard-coded local HF-cache
lookup and little intent documentation. Compare their metrics/configuration
against the maintained M1 GPU benchmark and the bf16-vs-f32 qualification. If
they add a distinct metric, merge it into B/Q; otherwise preserve the old source
in E. Do not silently delete them.

## `tests/support` dependency disposition (all 97 tracked Rust helpers)

### Release regression fixtures — keep packaged with their consumers

These are referenced by retained production source contracts, fixed correctness,
or the retained portion of typed parity. The sliced adapter called out below is
also a direct library unit-test fixture.

```text
tests/support/fixed_full_mantissa.rs
tests/support/triad_f32_tn_copyplan_dual_chunk_source.rs
tests/support/triad_f32_tn_direct_d768_out_bk16_source.rs
tests/support/triad_f32_tn_direct_prism_bk16_source.rs
tests/support/triad_half_nn_m128n64_source.rs
tests/support/triad_half_nn_n64_source.rs
tests/support/triad_half_nt_compact_source.rs
tests/support/triad_half_nt_direct_epilogue_source.rs
tests/support/triad_half_nt_fixed_s3_source.rs
tests/support/triad_half_nt_m64n128_s3_source.rs
tests/support/triad_half_nt_m96n128_s3_source.rs
tests/support/triad_half_nt_s3_source.rs
tests/support/triad_half_tile_screen.rs
tests/support/triad_half_tn_compact_source.rs
tests/support/triad_half_tn_fixed_s3_source.rs
tests/support/triad_half_tn_m64n128_regpipe_source.rs
tests/support/triad_half_tn_m64n128_source.rs
tests/support/triad_half_tn_microtile_source.rs
tests/support/triad_half_tn_regpipe_source.rs
tests/support/triad_half_tn_regpipe_vec2_sibling_plan.rs
tests/support/triad_half_tn_s4_source.rs
tests/support/triad_half_tn_vec2_epilogue_source.rs
tests/support/triad_half_tn_warpspecialized_source.rs
tests/support/triad_nn_n96_source.rs
tests/support/triad_tf32_nn_n96_direct_epilogue_source.rs
tests/support/triad_tf32_nt_a_ldmatrix_n96_s3_source.rs
tests/support/triad_tf32_nt_compact_a_ldmatrix_sliced_source.rs
tests/support/triad_tf32_tn_pre_rna_m64n96_s2_source.rs
tests/support/triad_tf32_tn_transpose_rna_m64n64_source.rs
tests/support/triad_tf32_tn_transpose_rna_n96_source.rs
tests/support/triad_tn_d128_direct_source.rs
tests/support/triad_tn_transpose_n96_source.rs
```

### Qualification helpers — move with Q and keep packaged if Q is public

```text
tests/support/fixed_sm120_exact_n64_admission.rs
tests/support/fixed_sm89_exact_n64_admission.rs
tests/support/fixed_sm89_toolkit_admission.rs
tests/support/sm89_exact_f32_tn_admission.rs
tests/support/triad_half_remaining_qualification.rs
```

### Discovery adapters — move with E and exclude from crate after extraction

```text
tests/support/triad_f32_tn_copyplan_dual_chunk_m128n64_bk16_source.rs
tests/support/triad_f32_tn_copyplan_dual_chunk_rolled_source.rs
tests/support/triad_f32_tn_copyplan_fused_finalize_groupm12_source.rs
tests/support/triad_f32_tn_copyplan_fused_finalize_n32_source.rs
tests/support/triad_f32_tn_copyplan_fused_finalize_source.rs
tests/support/triad_f32_tn_copyplan_raw_store_source.rs
tests/support/triad_f32_tn_direct_copyplan_source.rs
tests/support/triad_f32_tn_direct_d768_in_bk16_source.rs
tests/support/triad_f32_tn_direct_dual_chunk_source.rs
tests/support/triad_f32_tn_direct_prism_bk32_exact_tail_source.rs
tests/support/triad_f32_tn_direct_prism_fused_n32_source.rs
tests/support/triad_f32_tn_prism_vec4_reducer_source.rs
tests/support/triad_f32_transpose_32x8_source.rs
tests/support/triad_half_nn_n96_s3_source.rs
tests/support/triad_half_nn_s3_bx4_source.rs
tests/support/triad_half_nt_m64n128_s2_source.rs
tests/support/triad_half_nt_m64n128_s3_groupm8_source.rs
tests/support/triad_half_nt_m64n192_full_domain_source.rs
tests/support/triad_half_nt_m64n192_issue2211_source.rs
tests/support/triad_half_nt_m64n192_s3_bf16_source.rs
tests/support/triad_half_nt_m64n192_s3_source.rs
tests/support/triad_half_nt_m96n128_s3_bf16_source.rs
tests/support/triad_half_tn_8warp_16x32_source.rs
tests/support/triad_half_tn_bk32_s3_regpipe_vec2_source.rs
tests/support/triad_half_tn_m64n96_regpipe_vec2_source.rs
tests/support/triad_half_tn_m96n64_regpipe_vec2_source.rs
tests/support/triad_half_tn_one_wave_atlas_source.rs
tests/support/triad_half_tn_packed_raster_source.rs
tests/support/triad_half_tn_regpipe_vec2_bx4_source.rs
tests/support/triad_half_tn_regpipe_vec2_issue_source.rs
tests/support/triad_half_tn_regpipe_vec2_sliced_source.rs
tests/support/triad_half_tn_vec2_exact_entry_source.rs
tests/support/triad_half_tn_vec2_full_domain_source.rs
tests/support/triad_half_tn_vec2_full_tile_cg_source.rs
tests/support/triad_half_tn_vec2_full_tile_stage_source.rs
tests/support/triad_tf32_nn_m96n96_s2_copy_lifetime_source.rs
tests/support/triad_tf32_nn_m96n96_s2_direct_source.rs
tests/support/triad_tf32_nn_m96n96_s2_single_fragment_source.rs
tests/support/triad_tf32_nn_n96_b_ldmatrix_source.rs
tests/support/triad_tf32_nn_n96_cross_bk_prefetch_source.rs
tests/support/triad_tf32_nn_n96_direct_epilogue_ca_source.rs
tests/support/triad_tf32_nn_n96_direct_epilogue_nobarrier_source.rs
tests/support/triad_tf32_nn_n96_direct_grid_constant_source.rs
tests/support/triad_tf32_nn_n96_full_domain_cpasync_source.rs
tests/support/triad_tf32_nt_a_ldmatrix_prism_full_domain_source.rs
tests/support/triad_tf32_nt_a_ldmatrix_prism_k8_regpipe_source.rs
tests/support/triad_tf32_nt_compact_a_ldmatrix_sliced_d768_in_source.rs
tests/support/triad_tf32_nt_compact_a_ldmatrix_source.rs
tests/support/triad_tf32_nt_compact_ab_ldmatrix_source.rs
tests/support/triad_tf32_nt_shared_rna_source.rs
tests/support/triad_tf32_tn_small_regpipe_source.rs
tests/support/triad_tf32_tn_transpose_ab_rna_n96_source.rs
tests/support/triad_tf32_tn_transpose_rna_m64n128_s2_source.rs
tests/support/triad_tn_compact_source.rs
tests/support/triad_tn_dense_source.rs
tests/support/triad_tn_direct_n96_source.rs
tests/support/triad_tn_prepack_a_source.rs
tests/support/triad_tn_single_barrier_source.rs
```

### Currently unreferenced — investigate, then archive (do not delete blindly)

Repository-wide exact-path searches found no consumer for these two tracked
modules. One appears to be a layout helper and one an abandoned candidate
epilogue; confirm history before moving them to E.

```text
tests/support/fixed_half_batch_layout.rs
tests/support/triad_f32_tn_copyplan_epilogue_source.rs
```

## Compile-time dependencies that the move must not orphan

1. `sm89_finalist_source.rs` has a `#[cfg(test)]` module that always
   `include_str!`s the sliced adapter above. In non-CUDA library tests it also
   `include_str!`s the compact-XOR CUDA helper and `include!`s the entire
   compact-XOR Rust target to call its accepted candidate composer. That included
   stand declares eleven adapter modules. Before E is excluded, extract a small,
   independently frozen accepted-candidate fixture/composer and keep the current
   production-vs-frozen comparison plus missing/duplicate-anchor regressions.
   An expected string generated by the production composer is not independent.
2. The scalar NT copy-plan siblings target compile-time-includes three raw
   receipts from
   `internal/perf/ada-f32-nt-copyplan-siblings-batch-a-20260909/evidence/` and
   asserts compile/artifact/source/header/NVRTC-domain identities. Extract only
   the required normalized identity fixture into the release-test fixture tree,
   or keep the receipt regression with Q. Do not publish the raw `internal/perf`
   archive and do not leave a packaged target pointing outside the crate.
3. `tests/common/*.rs` is shared by many R/B/Q targets. Keep it as an ordinary
   packaged helper (or split it by lane only after a reference graph confirms no
   cross-lane include). The target matrix above does not count these modules as
   top-level Cargo targets.

## Cargo target strategy

Recommended boundary:

1. Set `autotests = false` and `autobenches = false` in `[package]`. Explicitly
   enumerate R targets with `[[test]]`, assigning `required-features` (`cuda`,
   `hf`, `nccl`, etc.) from their current crate-level gates. This prevents a new
   file under `tests/` from silently becoming a Cargo target.
2. Add a non-default feature such as `qualification = []`. Enumerate every Q
   source as an explicit `[[test]]` whose path may live under
   `tools/qualification/` and whose `required-features` includes
   `qualification` plus CUDA/HF as appropriate. Thus
   `cargo test --features cuda` does not build record tools, while a named
   `cargo test --features cuda,qualification --test <name>` remains reproducible.
   Public Q sources and their compact fixtures must be included in the package
   if the published manifest names them.
3. Convert B to explicit `[[bench]]` entries with `harness = false` and suitable
   `required-features`; expose deterministic CLI/env parameters and keep the set
   intentionally small. Do not keep `#[test] #[ignore]` merely for convenience.
4. Put E and its candidate-only CUDA/support files under `internal/experiments/`
   with no Cargo target declarations and exclude `internal/` from the published
   crate. Preserve useful source/history in Git.
5. Update the lane census to enumerate declared R/B/Q targets, not every `.rs`
   file under one directory. It should also fail if an explicit target path is
   missing or if E becomes reachable from Cargo/package includes.

An aggregator alternative (one top-level host test plus feature-specific module
aggregators) would reduce manifest volume, but explicit `[[test]]` targets are
safer here: they retain per-suite invocation, isolate compiler failures, and let
`required-features = ["qualification", ...]` prevent record targets from being
built during ordinary tests.

## Minimum release smoke/regression set

This is the minimum gate, not the whole retained R set:

- host/package invariants: `msrv_single_source`, `determinism_source_gates`,
  `qual_lane_census`, `fixed_epilogue_pins`, all SM80/SM120 host source gates;
- CPU public behavior: `m1_test_correctness`, `m3_test_correctness`, `grad_oracle`,
  `ddp_reduction_tree`, `hf_integration`, `scan_mode_boundary`, `prefill`,
  `prefill_m3` (after timing-arm extraction);
- CUDA API/wiring smoke: `gemm_mode_api`, `gemm_bi_tf32_selector`,
  `kernel_identity_cuda`, `gpu_inference_prefill_parity`,
  `inference_graph_route`, `graph_capture_panic`, `trainer_smoke`;
- promoted GEMM regressions: `arch_compile_gates`, architecture contract/source
  gates, `gemm_bi_fixed_correctness`, `gemm_bi_fixed_bridge`, retained portions
  of `gemm_bi_tf32_contract`, `gemm_bi_typed_parity`, `gemm_bi_tc`, plus the
  extracted compact-finalist independent-fixture regression;
- one end-to-end determinism/training parity arm for M1 and M3:
  `gemm_bi_determinism`, `training_graph_parity`, and
  `m3_training_graph_parity`.

The full release qualification remains Q on the applicable hardware/toolkits;
it should not be conflated with ordinary smoke.

## Extracted-package verification after cleanup

No command below was run in this audit. After the source/API owner closes and
the move is implemented:

1. Generate the crate (`cargo package --allow-dirty --no-verify` if validating
   local WIP), then extract the resulting `.crate` into a fresh temporary
   directory. Inspect that tree directly: R/B/Q target paths and their fixtures
   exist; `internal/experiments/`, raw `internal/perf/` receipts, and unrelated
   evidence do not. Do not rely only on `cargo package --list`.
2. In the extracted tree, run `cargo metadata --no-deps --format-version 1` and
   compare declared targets with the R/B/Q inventory. Confirm no E target and no
   missing explicit target path.
3. From the extracted tree run offline, locked host checks where dependency cache
   permits: `cargo test --no-default-features` and, critically,
   `cargo test --lib --no-default-features --no-run`. The latter exercises the
   non-CUDA compact-finalist fixture path that currently uses `include!`.
4. On a CUDA host with the selected toolkit, run
   `cargo test --lib --features cuda --no-run` and the minimum CUDA R set. This
   checks the other library-test configuration and the sliced-adapter fixture.
5. Build one representative explicit Q target with
   `--features cuda,qualification --test <name> --no-run`, then run the relevant
   supported-toolkit qualification commands on their proper hardware lanes.
6. Run one B target through `cargo bench --bench <name>` to verify the stable
   harness conversion. Finally run normal package verification from the same
   extracted contents (or compare it byte-for-file with what verification uses).

## Main concerns for the implementation pass

- Do not move/exclude the compact-XOR stand until its independent accepted-source
  fixture closure is extracted; otherwise non-CUDA library tests fail at compile
  time.
- Do not publish a target that still includes the three internal raw receipts.
- Do not treat the 92 targets absent from `qual/lanes.toml` as disposable; several
  are new production correctness/source-contract targets and are retained above.
- The WIP route-inventory test needs classification by its production owner after
  it stabilizes; this report intentionally makes no claim about its final content.
