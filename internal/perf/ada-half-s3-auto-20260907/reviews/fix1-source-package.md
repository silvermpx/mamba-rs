# Task6C source fix1 scoped review package

Original reviewed frozen source: ada-half-s3-auto-task6c-rust-review-v1.md. Fix requirements: ada-half-s3-auto-task6c-fix1-brief.md.
Root isolated full diff of three newly authorized fixture files and only performance-file hunks after original line9000; the original frozen harness edits are before line1000 and unchanged by fix1. Current performanceSHA7d61e9f1241c9b6b3e3018b48cc14d8c7ca29149c56a3a6fc3af2e6188dc40b2. Original other3 Rust hashes unchanged.
No commits yet; base4386e41a; original frozen performance3888a553. No production Triad/CUDA logic changes.
Reported covering tests: corrected CUDA12.8 full library645pass/0fail/46ignored,exit0; current direct-pair oracle1pass,exit0. ActualGPU smoke and allotherfinal qualification remain pending.

## Fix diff

diff --git a/src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs b/src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs
index 794c84cc..3ebfa54f 100644
--- a/src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs
+++ b/src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs
@@ -10111,26 +10111,26 @@ mod tf32_tests {
                     ResolvedGemmOp::Nt,
                     4621,
                     384,
                     1928,
                     Sm120ManifestRoute::Tiled(Tf32Sm120Tile::M64N64, Tf32Sm120Stages::S2),
                 ),
             ],
         );
     }
 
     #[test]
     fn tf32_tn_underfill_qualification_uses_current_tuning_revision() {
-        assert_eq!(TUNING_TABLE_REVISION, 42);
-        assert_eq!(F32_TF32_TUNING_REVISION, 42);
+        assert_eq!(TUNING_TABLE_REVISION, 43);
+        assert_eq!(F32_TF32_TUNING_REVISION, 43);
     }
 
     #[test]
     fn sm120_tf32_live_portable_tn_cell_is_exact_and_fail_closed() {
         let cohort = sm120_cohort((13, 2));
         let identity = cohort.identity;
         let request = normalized_request(ResolvedGemmOp::Tn, 128, 512, 1024);
         let operands = F32TriadOperands {
             output: 0x1000,
             a: 0x2000,
             b: 0x3000,
             bias: None,
diff --git a/src/mamba_ssm/gpu/gemm_bi_triad/launch.rs b/src/mamba_ssm/gpu/gemm_bi_triad/launch.rs
index ba977354..313758b3 100644
--- a/src/mamba_ssm/gpu/gemm_bi_triad/launch.rs
+++ b/src/mamba_ssm/gpu/gemm_bi_triad/launch.rs
@@ -13843,25 +13843,25 @@ mod prepared_f32_launch_tests {
         assert_eq!(resolved.shape, (256, 512, 384));
         assert_eq!(resolved.strides, (512, 384, 384));
         assert_eq!(resolved.tile, (16, 32));
         assert_eq!(resolved.bk, 32);
         assert_eq!(resolved.stages, 4);
         assert_eq!(resolved.threads, 128);
         assert_eq!(resolved.launch.grid_dim, config.grid_dim);
         assert_eq!(resolved.launch.block_dim, config.block_dim);
         assert_eq!(resolved.launch.shared_mem_bytes, config.shared_mem_bytes);
         assert_eq!(resolved.tensor_maps_digest, [0; 32]);
         assert_eq!(resolved.resources_digest, [2; 32]);
         assert_eq!(resolved.launch.arguments_digest, [3; 32]);
-        assert_eq!(resolved.tuning_table_revision, 42);
+        assert_eq!(resolved.tuning_table_revision, 43);
 
         let eager = build_resolved_gemm_launch_set(&[resolved]).unwrap();
         let graph = build_resolved_gemm_launch_set(&[resolved]).unwrap();
         assert_eq!(eager, graph);
         assert_ne!(eager.ordered_digest, [0; 32]);
         let mut mutated = resolved;
         mutated.launch.arguments_digest[0] ^= 1;
         assert_ne!(
             eager.ordered_digest,
             build_resolved_gemm_launch_set(&[mutated])
                 .unwrap()
                 .ordered_digest
@@ -13910,25 +13910,25 @@ mod prepared_f32_launch_tests {
         assert_eq!(resolved.shape, (512, 3_072, 768));
         assert_eq!(resolved.strides, (3_072, 768, 768));
         assert_eq!(resolved.tile, (80, 32));
         assert_eq!(resolved.bk, 64);
         assert_eq!(resolved.stages, 2);
         assert_eq!(resolved.threads, 160);
         assert_eq!(resolved.launch.grid_dim, config.grid_dim);
         assert_eq!(resolved.launch.block_dim, config.block_dim);
         assert_eq!(resolved.launch.shared_mem_bytes, config.shared_mem_bytes);
         assert_eq!(resolved.tensor_maps_digest, [1; 32]);
         assert_eq!(resolved.resources_digest, [2; 32]);
         assert_eq!(resolved.launch.arguments_digest, [3; 32]);
-        assert_eq!(resolved.tuning_table_revision, 42);
+        assert_eq!(resolved.tuning_table_revision, 43);
 
         let launch_set = build_resolved_gemm_launch_set(&[resolved]).unwrap();
         assert_eq!(launch_set.launch_count, 1);
         assert_ne!(launch_set.ordered_digest, [0; 32]);
         let mut mutated = resolved;
         mutated.launch.arguments_digest[0] ^= 1;
         let mutated_set = build_resolved_gemm_launch_set(&[mutated]).unwrap();
         assert_ne!(launch_set.ordered_digest, mutated_set.ordered_digest);
     }
 
     fn zero_request(op: ResolvedGemmOp) -> F32TriadRequest {
         let dims = match op {
diff --git a/tests/gemm_bi_tf32_cohort_binding.rs b/tests/gemm_bi_tf32_cohort_binding.rs
index 2a165b0e..056ec221 100644
--- a/tests/gemm_bi_tf32_cohort_binding.rs
+++ b/tests/gemm_bi_tf32_cohort_binding.rs
@@ -195,25 +195,25 @@ fn sm120_tf32_fresh_595_58_03_auto_symbols_graphs_and_bits() {
                 route,
                 PhysicalQualificationF32Epilogue::new(1.0, 0.0, false),
             );
             let mut qualified = qualify_physical_launch(&ctx, request)
                 .unwrap_or_else(|error| panic!("{dims:?}/{mode}: {error}"));
             qualified.validate_timed_request(&ctx, request).unwrap();
             let evidence = qualified.evidence();
             assert_eq!(evidence.launch_count(), 1);
             assert!(
                 evidence.eager_graph_equal(),
                 "actual captured/eager nodes differ"
             );
-            assert_eq!(evidence.route_identity().tuning_table_revision, 42);
+            assert_eq!(evidence.route_identity().tuning_table_revision, 43);
             let [node] = evidence.nodes() else {
                 panic!("one actual GEMM expected");
             };
             assert_eq!(node.symbol, symbol, "{dims:?}/{mode}: wrong actual symbol");
             assert_eq!(node.module_kind, ModuleKind::TriadSm120);
             assert_eq!(node.logical_op, ResolvedGemmOp::Nn);
             assert_eq!(node.shape, dims);
             assert_eq!(node.strides, (dims.1, dims.2, dims.2));
             assert_eq!(node.tile, Some(tile_shape));
             assert_eq!(
                 node.numeric_contract,
                 Some(ResolvedNumericContract::Sm120TmaMmaTf32RnaV1)
diff --git a/tests/gemm_bi_fixed_performance.rs b/tests/gemm_bi_fixed_performance.rs
index 547e19c9..53e2196e 100644
--- a/tests/gemm_bi_fixed_performance.rs
+++ b/tests/gemm_bi_fixed_performance.rs
@@ -9476,104 +9677,119 @@ const FIXED_AUTO_VENDOR_EXACT_CELLS: &[FixedAutoVendorCell] = &[
     },
     FixedAutoVendorCell {
         label: "hot_e",
         shape: FixedShape {
             m: 2048,
             k: 2304,
             n: 768,
         },
         expected: FixedTile::Legacy,
     },
 ];
 
-fn expected_ada_half_auto_v42(
+fn expected_ada_half_auto_v43(
     nvrtc: (i32, i32),
     dtype: WeightDtype,
     shape: FixedShape,
     has_bias: bool,
 ) -> Option<FixedTile> {
-    use FixedTile::{Tc128Sm89Pipeline as Pipeline, Tc128Sm89Swizzle as Swizzle};
+    use FixedTile::{
+        Tc128Sm89Pipeline as Pipeline, Tc128Sm89S3 as S3, Tc128Sm89Swizzle as Swizzle,
+    };
 
     match (nvrtc, dtype, (shape.m, shape.k, shape.n), has_bias) {
         ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 384, 1928), false) => Some(Pipeline),
         ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 384, 1928), true) => Some(Swizzle),
         ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 768, 2304), _) => Some(Swizzle),
         ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 1928, 384), _) => Some(Swizzle),
         ((12, 8) | (13, 0), WeightDtype::Bf16, (2048, 768, 2304), _) => Some(Swizzle),
         ((12, 8) | (13, 0), WeightDtype::Bf16, (2048, 2304, 768), _) => Some(Swizzle),
         ((12, 8) | (13, 0), WeightDtype::F16, (4621, 384, 1928), _) => Some(Pipeline),
         ((12, 8) | (13, 0), WeightDtype::F16, (4621, 768, 2304), _) => Some(Swizzle),
         ((12, 8) | (13, 0), WeightDtype::F16, (4621, 1928, 384), _) => Some(Swizzle),
         ((12, 8) | (13, 0), WeightDtype::F16, (2048, 768, 2304), _) => Some(Swizzle),
         ((12, 8) | (13, 0), WeightDtype::F16, (2048, 2304, 768), _) => Some(Swizzle),
         ((13, 2), WeightDtype::Bf16, (4621, 384, 1928), _) => Some(Pipeline),
-        ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), _) => Some(Swizzle),
+        ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), false) => Some(S3),
+        ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), true) => Some(Swizzle),
         ((13, 2), WeightDtype::Bf16, (4621, 1928, 384), _) => Some(Pipeline),
         ((13, 2), WeightDtype::Bf16, (2048, 768, 2304), _) => Some(Swizzle),
         ((13, 2), WeightDtype::Bf16, (2048, 2304, 768), _) => Some(Swizzle),
         ((13, 2), WeightDtype::F16, (4621, 384, 1928), _) => Some(Pipeline),
-        ((13, 2), WeightDtype::F16, (4621, 768, 2304), _) => Some(Swizzle),
+        ((13, 2), WeightDtype::F16, (4621, 768, 2304), false) => Some(S3),
+        ((13, 2), WeightDtype::F16, (4621, 768, 2304), true) => Some(Swizzle),
         ((13, 2), WeightDtype::F16, (4621, 1928, 384), _) => Some(Pipeline),
         ((13, 2), WeightDtype::F16, (2048, 768, 2304), _) => Some(Swizzle),
         ((13, 2), WeightDtype::F16, (2048, 2304, 768), _) => Some(Pipeline),
         _ => None,
     }
 }
 
 #[test]
-fn ada_half_auto_v42_harness_expectation_is_literal_and_fail_closed() {
+fn ada_half_auto_v43_harness_expectation_is_literal_and_fail_closed() {
     for &nvrtc in &[(12, 8), (13, 0), (13, 2)] {
         for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
             for cell in FIXED_AUTO_VENDOR_EXACT_CELLS {
                 for has_bias in [false, true] {
                     assert!(
-                        expected_ada_half_auto_v42(nvrtc, dtype, cell.shape, has_bias).is_some(),
+                        expected_ada_half_auto_v43(nvrtc, dtype, cell.shape, has_bias).is_some(),
                         "{nvrtc:?} {dtype:?} {} bias={has_bias}",
                         cell.label
                     );
                 }
             }
         }
     }
     let hot_a = FIXED_AUTO_VENDOR_EXACT_CELLS[0].shape;
     assert_eq!(
-        expected_ada_half_auto_v42((12, 8), WeightDtype::Bf16, hot_a, false),
+        expected_ada_half_auto_v43((12, 8), WeightDtype::Bf16, hot_a, false),
         Some(FixedTile::Tc128Sm89Pipeline)
     );
     assert_eq!(
-        expected_ada_half_auto_v42((12, 8), WeightDtype::Bf16, hot_a, true),
+        expected_ada_half_auto_v43((12, 8), WeightDtype::Bf16, hot_a, true),
         Some(FixedTile::Tc128Sm89Swizzle)
     );
     for nvrtc in [(12, 7), (13, 1), (13, 3), (14, 0)] {
         assert_eq!(
-            expected_ada_half_auto_v42(nvrtc, WeightDtype::Bf16, hot_a, false),
+            expected_ada_half_auto_v43(nvrtc, WeightDtype::Bf16, hot_a, false),
             None
         );
     }
     assert_eq!(
-        expected_ada_half_auto_v42((13, 2), WeightDtype::F32, hot_a, false),
+        expected_ada_half_auto_v43((13, 2), WeightDtype::F32, hot_a, false),
         None
     );
     assert_eq!(
-        expected_ada_half_auto_v42(
+        expected_ada_half_auto_v43(
             (13, 2),
             WeightDtype::F16,
             FixedShape {
                 m: hot_a.m - 1,
                 ..hot_a
             },
             false
         ),
         None
     );
+    let hot_b = FIXED_AUTO_VENDOR_EXACT_CELLS[1].shape;
+    for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
+        assert_eq!(
+            expected_ada_half_auto_v43((13, 2), dtype, hot_b, false),
+            Some(FixedTile::Tc128Sm89S3)
+        );
+        assert_eq!(
+            expected_ada_half_auto_v43((13, 2), dtype, hot_b, true),
+            Some(FixedTile::Tc128Sm89Swizzle)
+        );
+    }
 }
 
 fn fixed_auto_vendor_expected_exact_tile(
     cell: FixedAutoVendorCell,
     device_cc: (u32, u32),
     sm_count: u32,
     nvrtc_version: (i32, i32),
     nvrtc_library_known: bool,
     has_bias: bool,
     t256_loaded: bool,
 ) -> FixedTile {
     match device_cc {
@@ -14367,31 +14583,31 @@ fn fixed_ada_half_forced_direct_pair() {
                 .expect("direct-pair pipeline C");
             let swizzle = DtypedBuf::zeros(&ctx.stream, elements, output_dtype)
                 .expect("direct-pair swizzle C");
             let reference = DtypedBuf::zeros(&ctx.stream, elements, WeightDtype::F32)
                 .expect("direct-pair PEDANTIC reference C");
             let bias = DtypedBuf::zeros(&ctx.stream, shape.n, WeightDtype::F32)
                 .expect("direct-pair F32 bias");
             bias.upload_f32(&ctx.stream, &synth(shape.n, 0x0ada_b1a5))
                 .expect("direct-pair F32 bias upload");
 
             for &bias_index in &biases {
                 let has_bias = bias_index == 1;
-                let expected_auto = expected_ada_half_auto_v42(
+                let expected_auto = expected_ada_half_auto_v43(
                     compiler.nvrtc_version,
                     input_dtype,
                     shape,
                     has_bias,
                 )
-                .expect("literal revision-42 direct-pair AUTO expectation");
+                .expect("literal revision-43 direct-pair AUTO expectation");
                 let auto_ops = FixedFwdOperands {
                     c: typed(&auto, output_dtype),
                     x: typed(&a, input_dtype),
                     w: typed(&b, input_dtype),
                     bias_ptr: has_bias.then(|| bias.cached_ptr()),
                 };
                 let pipeline_ops = FixedFwdOperands {
                     c: typed(&pipeline, output_dtype),
                     ..auto_ops
                 };
                 let swizzle_ops = FixedFwdOperands {
                     c: typed(&swizzle, output_dtype),
