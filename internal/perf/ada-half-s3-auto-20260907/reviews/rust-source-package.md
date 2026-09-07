# Task6C frozen Rust source review package

Base: 4386e41a43f1b3b95baa829b1e7ab2b470eecf29. Head: uncommitted four-file source-stable snapshot; root alone will commit after review and runtime.
No intermediate implementation commits.

Source SHA256:
- gemm_bi_fixed.rs: 21127b055f53c6a8fb3968083dbdd043bac39eb01ddd26c978c37be9c724e477
- kernel_identity.rs: d3b821bab65bcdfb09cfdebf9a6be5e6260c611b244bf24754c47d43efcf966b
- gemm_bi_fixed_sm89_pipeline.rs: 51400bb8d9e2d6f357381b14bb02b0f133b3c1720a26b457fe16390c4e1a485c
- gemm_bi_fixed_performance.rs: 3888a55389044274bd74e9feb14c69133ca8ae4404889ecdfaad65644ec291f1

This review covers frozen Rust implementation/spec quality only. New Python utilities and final GPU evidence are a separate pending gate; do not imply full Task6C completion. Inspect unchanged helper context only for a concrete named risk.

## Diff

 src/mamba_ssm/gpu/gemm_bi_fixed.rs   | 174 ++++++++++----------
 src/mamba_ssm/gpu/kernel_identity.rs |  38 ++++-
 tests/gemm_bi_fixed_performance.rs   | 297 +++++++++++++++++++++++++++++------
 tests/gemm_bi_fixed_sm89_pipeline.rs |  18 ++-
 4 files changed, 384 insertions(+), 143 deletions(-)
diff --git a/src/mamba_ssm/gpu/gemm_bi_fixed.rs b/src/mamba_ssm/gpu/gemm_bi_fixed.rs
index d8217ed0..1313b986 100644
--- a/src/mamba_ssm/gpu/gemm_bi_fixed.rs
+++ b/src/mamba_ssm/gpu/gemm_bi_fixed.rs
@@ -473,50 +473,61 @@ struct FixedTileDevice {
 // Actual NVRTC CUDA12.8/13.0/13.2 qualification on the 142-SM Ada target.
 // The preferred route is the reviewed direct-pair winner; compatibility
 // fallback remains limited to independently qualified holders. Returning None
 // preserves the general architecture/portable ladder below this overlay.
 fn fixed_select_sm89_half_auto_tile(
     operands: FixedFwdOperands,
     shape: FixedShape,
     device: FixedTileDevice,
     nvrtc: (i32, i32),
     nvrtc_library_known: bool,
     pipeline_available: bool,
     swizzle_available: bool,
+    s3_available: bool,
 ) -> Option<FixedTile> {
-    use FixedTile::{Tc128Sm89Pipeline as Pipeline, Tc128Sm89Swizzle as Swizzle};
+    use FixedTile::{
+        Tc128Sm89Pipeline as Pipeline, Tc128Sm89S3 as S3, Tc128Sm89Swizzle as Swizzle,
+    };
 
     if !nvrtc_library_known
         || device.compute_capability != (8, 9)
         || device.multiprocessors != 142
         || !matches!(nvrtc, (12, 8) | (13, 0) | (13, 2))
         || operands.c.dtype == WeightDtype::F32
         || operands.c.dtype != operands.x.dtype
         || operands.x.dtype != operands.w.dtype
         || [operands.c.ptr, operands.x.ptr, operands.w.ptr]
             .into_iter()
             .any(|ptr| ptr == 0 || !ptr.is_multiple_of(16))
         || operands.bias_ptr.is_some_and(|ptr| !ptr.is_multiple_of(4))
         || !matches!(
             (shape.m, shape.k, shape.n),
             (4621, 384, 1928)
                 | (4621, 768, 2304)
                 | (4621, 1928, 384)
                 | (2048, 768, 2304)
                 | (2048, 2304, 768)
         )
     {
         return None;
     }
 
+    if s3_available
+        && nvrtc == (13, 2)
+        && (shape.m, shape.k, shape.n) == (4621, 768, 2304)
+        && operands.bias_ptr.is_none()
+    {
+        return Some(S3);
+    }
+
     let dims = (shape.m, shape.k, shape.n);
     let preferred = match (nvrtc, operands.c.dtype, dims, operands.bias_ptr.is_some()) {
         ((12, 8) | (13, 0), WeightDtype::Bf16, (4621, 384, 1928), false) => Pipeline,
         ((12, 8) | (13, 0), WeightDtype::F16, (4621, 384, 1928), _) => Pipeline,
         ((12, 8) | (13, 0), _, _, _) => Swizzle,
         ((13, 2), WeightDtype::Bf16, (4621, 384, 1928) | (4621, 1928, 384), _) => Pipeline,
         (
             (13, 2),
             WeightDtype::F16,
             (4621, 384, 1928) | (4621, 1928, 384) | (2048, 2304, 768),
             _,
         ) => Pipeline,
@@ -533,63 +544,86 @@ fn fixed_select_sm89_half_auto_tile(
         ((13, 2), Swizzle) if swizzle_available => Some(Swizzle),
         ((13, 2), Swizzle) if pipeline_available => Some(Pipeline),
         _ => None,
     }
 }
 
 #[cfg(test)]
 mod sm89_pipeline_auto_tests {
     use super::*;
 
     const P: FixedTile = FixedTile::Tc128Sm89Pipeline;
     const S: FixedTile = FixedTile::Tc128Sm89Swizzle;
+    const S3: FixedTile = FixedTile::Tc128Sm89S3;
 
     fn operands(dtype: WeightDtype, has_bias: bool) -> FixedFwdOperands {
         FixedFwdOperands {
             c: TypedPtr { ptr: 0x1000, dtype },
             x: TypedPtr { ptr: 0x2000, dtype },
             w: TypedPtr { ptr: 0x3000, dtype },
             bias_ptr: has_bias.then_some(0x4004),
         }
     }
 
     fn select(
         nvrtc: (i32, i32),
         dtype: WeightDtype,
         dims: (usize, usize, usize),
         has_bias: bool,
         known_library: bool,
         pipeline_available: bool,
         swizzle_available: bool,
+        s3_available: bool,
     ) -> Option<FixedTile> {
         fixed_select_sm89_half_auto_tile(
             operands(dtype, has_bias),
             FixedShape {
                 m: dims.0,
                 k: dims.1,
                 n: dims.2,
             },
             FixedTileDevice {
                 multiprocessors: 142,
                 compute_capability: (8, 9),
             },
             nvrtc,
             known_library,
             pipeline_available,
             swizzle_available,
+            s3_available,
         )
     }
 
     #[test]
-    fn sm89_half_auto_v42_matches_all_literal_preferred_cells() {
+    fn sm89_half_auto_v43_promotes_only_measured_cuda132_b0_no_bias_cells() {
+        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
+            assert_eq!(
+                select(
+                    (13, 2),
+                    dtype,
+                    (4621, 768, 2304),
+                    false,
+                    true,
+                    true,
+                    true,
+                    true,
+                ),
+                Some(S3),
+                "CUDA13.2 {dtype:?} B0/no-bias must select the measured S3 winner"
+            );
+        }
+    }
+
+    #[test]
+    fn sm89_half_auto_v43_matches_all_literal_cells_and_availability_states() {
         let cases = [
             ((12, 8), WeightDtype::Bf16, (4621, 384, 1928), false, P),
             ((12, 8), WeightDtype::Bf16, (4621, 384, 1928), true, S),
             ((12, 8), WeightDtype::Bf16, (4621, 768, 2304), false, S),
             ((12, 8), WeightDtype::Bf16, (4621, 768, 2304), true, S),
             ((12, 8), WeightDtype::Bf16, (4621, 1928, 384), false, S),
             ((12, 8), WeightDtype::Bf16, (4621, 1928, 384), true, S),
             ((12, 8), WeightDtype::Bf16, (2048, 768, 2304), false, S),
             ((12, 8), WeightDtype::Bf16, (2048, 768, 2304), true, S),
             ((12, 8), WeightDtype::Bf16, (2048, 2304, 768), false, S),
             ((12, 8), WeightDtype::Bf16, (2048, 2304, 768), true, S),
             ((12, 8), WeightDtype::F16, (4621, 384, 1928), false, P),
@@ -615,136 +649,111 @@ mod sm89_pipeline_auto_tests {
             ((13, 0), WeightDtype::F16, (4621, 384, 1928), false, P),
             ((13, 0), WeightDtype::F16, (4621, 384, 1928), true, P),
             ((13, 0), WeightDtype::F16, (4621, 768, 2304), false, S),
             ((13, 0), WeightDtype::F16, (4621, 768, 2304), true, S),
             ((13, 0), WeightDtype::F16, (4621, 1928, 384), false, S),
             ((13, 0), WeightDtype::F16, (4621, 1928, 384), true, S),
             ((13, 0), WeightDtype::F16, (2048, 768, 2304), false, S),
             ((13, 0), WeightDtype::F16, (2048, 768, 2304), true, S),
             ((13, 0), WeightDtype::F16, (2048, 2304, 768), false, S),
             ((13, 0), WeightDtype::F16, (2048, 2304, 768), true, S),
             ((13, 2), WeightDtype::Bf16, (4621, 384, 1928), false, P),
             ((13, 2), WeightDtype::Bf16, (4621, 384, 1928), true, P),
-            ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), false, S),
+            ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), false, S3),
             ((13, 2), WeightDtype::Bf16, (4621, 768, 2304), true, S),
             ((13, 2), WeightDtype::Bf16, (4621, 1928, 384), false, P),
             ((13, 2), WeightDtype::Bf16, (4621, 1928, 384), true, P),
             ((13, 2), WeightDtype::Bf16, (2048, 768, 2304), false, S),
             ((13, 2), WeightDtype::Bf16, (2048, 768, 2304), true, S),
             ((13, 2), WeightDtype::Bf16, (2048, 2304, 768), false, S),
             ((13, 2), WeightDtype::Bf16, (2048, 2304, 768), true, S),
             ((13, 2), WeightDtype::F16, (4621, 384, 1928), false, P),
             ((13, 2), WeightDtype::F16, (4621, 384, 1928), true, P),
-            ((13, 2), WeightDtype::F16, (4621, 768, 2304), false, S),
+            ((13, 2), WeightDtype::F16, (4621, 768, 2304), false, S3),
             ((13, 2), WeightDtype::F16, (4621, 768, 2304), true, S),
             ((13, 2), WeightDtype::F16, (4621, 1928, 384), false, P),
             ((13, 2), WeightDtype::F16, (4621, 1928, 384), true, P),
             ((13, 2), WeightDtype::F16, (2048, 768, 2304), false, S),
             ((13, 2), WeightDtype::F16, (2048, 768, 2304), true, S),
             ((13, 2), WeightDtype::F16, (2048, 2304, 768), false, P),
             ((13, 2), WeightDtype::F16, (2048, 2304, 768), true, P),
         ];
         assert_eq!(cases.len(), 60);
-        for (nvrtc, dtype, dims, has_bias, expected) in cases {
-            assert_eq!(
-                select(nvrtc, dtype, dims, has_bias, true, true, true),
-                Some(expected),
-                "{nvrtc:?} {dtype:?} {dims:?} bias={has_bias}"
-            );
-        }
-    }
-
-    #[test]
-    fn sm89_half_auto_v42_applies_independent_availability_fallbacks() {
-        let a = (4621, 384, 1928);
-        let b = (4621, 768, 2304);
-        let c = (4621, 1928, 384);
-        let e = (2048, 2304, 768);
-
-        for nvrtc in [(12, 8), (13, 0)] {
-            assert_eq!(
-                select(nvrtc, WeightDtype::Bf16, a, false, true, true, false),
-                Some(P)
-            );
-            assert_eq!(
-                select(nvrtc, WeightDtype::Bf16, a, false, true, false, true),
-                Some(S)
-            );
-            assert_eq!(
-                select(nvrtc, WeightDtype::Bf16, a, true, true, true, false),
-                Some(P)
-            );
-            assert_eq!(
-                select(nvrtc, WeightDtype::Bf16, a, true, true, false, true),
-                Some(S)
-            );
-            assert_eq!(
-                select(nvrtc, WeightDtype::Bf16, b, false, true, true, false),
-                Some(P)
-            );
-            assert_eq!(
-                select(nvrtc, WeightDtype::Bf16, b, false, true, false, false),
-                None
-            );
+        for (nvrtc, dtype, dims, has_bias, preferred) in cases {
+            for pipeline_available in [false, true] {
+                for swizzle_available in [false, true] {
+                    for s3_available in [false, true] {
+                        let old_preferred = if preferred == S3 { S } else { preferred };
+                        let expected = if preferred == S3 && s3_available {
+                            Some(S3)
+                        } else {
+                            match (nvrtc, old_preferred, pipeline_available, swizzle_available) {
+                                ((12, 8) | (13, 0), P, true, _) => Some(P),
+                                ((12, 8) | (13, 0), P, false, true) => Some(S),
+                                ((12, 8) | (13, 0), S, _, true) => Some(S),
+                                ((12, 8) | (13, 0), S, true, false) => Some(P),
+                                ((13, 2), P, true, _) => Some(P),
+                                ((13, 2), S, _, true) => Some(S),
+                                ((13, 2), S, true, false) => Some(P),
+                                _ => None,
+                            }
+                        };
+                        assert_eq!(
+                            select(
+                                nvrtc,
+                                dtype,
+                                dims,
+                                has_bias,
+                                true,
+                                pipeline_available,
+                                swizzle_available,
+                                s3_available,
+                            ),
+                            expected,
+                            "{nvrtc:?} {dtype:?} {dims:?} bias={has_bias} pipeline={pipeline_available} swizzle={swizzle_available} s3={s3_available}"
+                        );
+                    }
+                }
+            }
         }
-
-        assert_eq!(
-            select((13, 2), WeightDtype::Bf16, b, false, true, true, false),
-            Some(P)
-        );
-        assert_eq!(
-            select((13, 2), WeightDtype::Bf16, b, false, true, false, true),
-            Some(S)
-        );
-        assert_eq!(
-            select((13, 2), WeightDtype::Bf16, a, false, true, false, true),
-            None
-        );
-        assert_eq!(
-            select((13, 2), WeightDtype::Bf16, c, true, true, false, true),
-            None
-        );
-        assert_eq!(
-            select((13, 2), WeightDtype::F16, e, false, true, false, true),
-            None
-        );
-        assert_eq!(
-            select((13, 2), WeightDtype::F16, e, true, true, true, false),
-            Some(P)
-        );
-        assert_eq!(
-            select((13, 2), WeightDtype::F16, e, true, true, false, false),
-            None
-        );
     }
 
     #[test]
-    fn sm89_half_auto_v42_declines_every_common_gate_failure() {
+    fn sm89_half_auto_v43_declines_every_common_gate_failure() {
         let device = FixedTileDevice {
             multiprocessors: 142,
             compute_capability: (8, 9),
         };
         let shape = FixedShape {
             m: 4621,
-            k: 384,
-            n: 1928,
+            k: 768,
+            n: 2304,
         };
         let ops = operands(WeightDtype::Bf16, false);
         let choose = |o, s, d, v, known, pipeline, swizzle| {
-            fixed_select_sm89_half_auto_tile(o, s, d, v, known, pipeline, swizzle)
+            fixed_select_sm89_half_auto_tile(o, s, d, v, known, pipeline, swizzle, true)
         };
 
         assert_eq!(choose(ops, shape, device, (13, 2), false, true, true), None);
         assert_eq!(
-            choose(ops, shape, device, (13, 2), true, false, false),
+            fixed_select_sm89_half_auto_tile(
+                ops,
+                shape,
+                device,
+                (13, 2),
+                true,
+                false,
+                false,
+                false,
+            ),
             None
         );
         for version in [(12, 7), (12, 9), (13, 1), (13, 3), (14, 0)] {
             assert_eq!(choose(ops, shape, device, version, true, true, true), None);
         }
         for cc in [(8, 0), (8, 6), (8, 7), (9, 0), (10, 0), (12, 0), (12, 1)] {
             assert_eq!(
                 choose(
                     ops,
                     shape,
                     FixedTileDevice {
                         compute_capability: cc,
@@ -3687,34 +3696,39 @@ pub fn fixed_forward(
     if homogeneous_half {
         let selected = fixed_select_sm89_half_auto_tile(
             operands,
             shape,
             FixedTileDevice {
                 multiprocessors: ctx.kernels.multiprocessor_count(),
                 compute_capability: ctx.compute_capability(),
             },
             compiler.nvrtc_version,
             compiler.nvrtc_library_known,
             ctx.kernels.fixed_sm89_half_pipeline.is_some(),
             ctx.kernels.fixed_sm89_half_swizzle.is_some(),
+            ctx.kernels.fixed_sm89_half_s3.is_some(),
         );
         match selected {
             Some(FixedTile::Tc128Sm89Pipeline) => {
                 launch_sm89_half_pipeline(ctx, c.dtype, &args)?;
                 return Ok(FixedTile::Tc128Sm89Pipeline);
             }
             Some(FixedTile::Tc128Sm89Swizzle) => {
                 launch_sm89_half_swizzle(ctx, c.dtype, &args)?;
                 return Ok(FixedTile::Tc128Sm89Swizzle);
             }
+            Some(FixedTile::Tc128Sm89S3) => {
+                launch_sm89_half_s3(ctx, c.dtype, &args)?;
+                return Ok(FixedTile::Tc128Sm89S3);
+            }
             Some(_) => unreachable!("Ada half AUTO selector returned a foreign tile"),
             None => {}
         }
     }
     // Architecture rungs: on Hopper and datacenter Blackwell the arch's
     // own tensor path is the numeric family for every eligible shape
     // (per-arch families are the documented law; batch invariance holds
     // inside each). Unaligned operands fall back to the portable
     // ladder, whose kernels carry their own scalar staging.
     if homogeneous_half
         && fixed_sm120_half_eligible(ctx, &args)
         && let Some(tile) = fixed_pick_sm120_half_exact(
@@ -6313,25 +6327,25 @@ mod tests {
 
         let qualified_device = FixedSm120HalfExactDevice {
             compute_capability: (12, 0),
             multiprocessors: 170,
             nvrtc_version: (13, 2),
         };
         assert_eq!(
             (
                 NUMERIC_ABI_REVISION,
                 TUNING_TABLE_REVISION,
                 SCHEDULE_REVISION,
             ),
-            (5, 42, 8),
+            (5, 43, 8),
             "the release compiler identity must remain explicitly pinned"
         );
         let mut promoted = Vec::new();
         for (dims, incumbent) in [
             ((512, 1928, 2304), C),
             ((1536, 1928, 2304), C),
             ((1536, 1928, 1928), C),
             ((3072, 1928, 1928), C),
             ((4096, 520, 1536), A),
             ((2048, 1928, 1536), C),
             ((1024, 1928, 1928), FixedSm120HalfTile::M128N128Bk32S2),
             ((1024, 1928, 2304), FixedSm120HalfTile::M128N128Bk32S2),
diff --git a/src/mamba_ssm/gpu/kernel_identity.rs b/src/mamba_ssm/gpu/kernel_identity.rs
index 3dc5ed7e..1da2f11e 100644
--- a/src/mamba_ssm/gpu/kernel_identity.rs
+++ b/src/mamba_ssm/gpu/kernel_identity.rs
@@ -23,25 +23,25 @@ pub fn digest_hex(value: &Sha256Digest) -> String {
     output
 }
 
 const HASH_DOMAIN: &[u8] = b"mamba-rs.length-framed-sha256.v1";
 const CACHE_MAGIC: [u8; 16] = *b"MAMBA-PTX-CACHE\0";
 const CACHE_FORMAT_VERSION: u16 = 1;
 
 pub const COMPOSER_REVISION: u16 = 1;
 pub const COMPILER_REVISION: u16 = 3;
 pub const NUMERIC_ABI_REVISION: u16 = 5;
 // Host dispatch epoch: qualified Ada homogeneous-half AUTO; unchanged modules
 // and retained cohorts keep their original evidence, while all graphs re-capture.
-pub const TUNING_TABLE_REVISION: u16 = 42;
+pub const TUNING_TABLE_REVISION: u16 = 43;
 pub const SCHEDULE_REVISION: u16 = 8;
 
 const NUMERIC_CONTRACT_DOMAIN: &[u8] = b"mamba-rs.resolved-numeric-contract.v2";
 const ARTIFACT_DIGEST_DOMAIN: &[u8] = b"mamba-rs.artifact-digest.v1";
 const COMPILER_TARGET_DOMAIN: &[u8] = b"mamba-rs.compiler-target.v1";
 const DRIVER_BUILD_DIGEST_DOMAIN: &[u8] = b"mamba-rs.driver-build-digest.v1";
 const LAUNCH_COUNT_DOMAIN: &[u8] = b"mamba-rs.resolved-launch-count.v1";
 const ROUTE_INDEX_DOMAIN: &[u8] = b"mamba-rs.resolved-route-index.v1";
 const PHYSICAL_LAUNCH_INDEX_DOMAIN: &[u8] = b"mamba-rs.resolved-physical-launch-index.v1";
 
 pub(crate) fn deterministic_nvrtc_options(
     nvrtc_version: (i32, i32),
@@ -4952,47 +4952,47 @@ mod physical_launch_tests {
                 optin_shared_bytes: 101_376,
                 tensor_map_access: false,
             },
             tuning_table_revision: TUNING_TABLE_REVISION,
             schedule_set_revision: SCHEDULE_REVISION,
             state_capacity: 64,
         }
     }
 
     #[test]
     fn ada_rna_auto_epoch_rejects_revision39_graph_identity() {
         let current = physical_context();
-        assert_eq!(current.tuning_table_revision, 42);
+        assert_eq!(current.tuning_table_revision, 43);
         let mut captured = current;
         captured.tuning_table_revision = 39;
         assert_eq!(captured.compiler, current.compiler);
         assert_eq!(captured.artifacts, current.artifacts);
         assert_eq!(captured.numeric_contracts, current.numeric_contracts);
         assert_eq!(
             captured.schedule_set_revision,
             current.schedule_set_revision
         );
         let error = captured
             .ensure_current(current, "revision39 Fixed AUTO graph replay")
             .expect_err("RNA AUTO promotion reused the revision39 graph identity");
         assert!(error.contains("re-capture before replay"));
         current
             .ensure_current(current, "current Fixed AUTO graph replay")
             .unwrap();
     }
 
     #[test]
     fn ada_rna_toolkit_auto_epoch_rejects_revision40_graph_identity() {
         let current = physical_context();
-        assert_eq!(current.tuning_table_revision, 42);
+        assert_eq!(current.tuning_table_revision, 43);
         let mut captured = current;
         captured.tuning_table_revision = 40;
         assert_eq!(captured.compiler, current.compiler);
         assert_eq!(captured.artifacts, current.artifacts);
         assert_eq!(captured.numeric_contracts, current.numeric_contracts);
         assert_eq!(
             captured.schedule_set_revision,
             current.schedule_set_revision
         );
         let error = captured
             .ensure_current(current, "revision40 Fixed replay")
             .expect_err("toolkit AUTO reused an old captured route");
@@ -5014,56 +5014,78 @@ mod physical_launch_tests {
         let error = captured
             .ensure_current(current, "revision38 graph replay")
             .expect_err("host-only cohort promotion reused the revision38 graph identity");
         assert!(error.contains("re-capture before replay"));
         current
             .ensure_current(current, "current graph replay")
             .unwrap();
     }
 
     #[test]
     fn ada_half_auto_epoch_rejects_revision41_graph_identity() {
         let current = physical_context();
-        assert_eq!(current.tuning_table_revision, 42);
+        assert_eq!(current.tuning_table_revision, 43);
         let mut captured = current;
         captured.tuning_table_revision = 41;
         assert_eq!(captured.compiler, current.compiler);
         assert_eq!(captured.artifacts, current.artifacts);
         assert_eq!(captured.numeric_contracts, current.numeric_contracts);
         assert_eq!(
             captured.schedule_set_revision,
             current.schedule_set_revision
         );
         let error = captured
             .ensure_current(current, "revision41 Fixed replay")
             .expect_err("Ada half AUTO promotion reused a revision41 graph identity");
         assert!(error.contains("re-capture before replay"));
         current
-            .ensure_current(current, "current revision42 Fixed replay")
+            .ensure_current(current, "current revision43 Fixed replay")
             .unwrap();
     }
 
     #[test]
-    fn fixed_only_artifact_replacement_invalidates_graph_at_unchanged_epoch42() {
+    fn ada_half_s3_auto_epoch_rejects_revision42_graph_identity() {
+        let current = physical_context();
+        assert_eq!(current.tuning_table_revision, 43);
+        let mut captured = current;
+        captured.tuning_table_revision = 42;
+        assert_eq!(captured.compiler, current.compiler);
+        assert_eq!(captured.artifacts, current.artifacts);
+        assert_eq!(captured.numeric_contracts, current.numeric_contracts);
+        assert_eq!(
+            captured.schedule_set_revision,
+            current.schedule_set_revision
+        );
+        let error = captured
+            .ensure_current(current, "revision42 Fixed replay")
+            .expect_err("Ada half S3 AUTO promotion reused a revision42 graph identity");
+        assert!(error.contains("re-capture before replay"));
+        current
+            .ensure_current(current, "current revision43 Fixed replay")
+            .unwrap();
+    }
+
+    #[test]
+    fn fixed_only_artifact_replacement_invalidates_graph_at_unchanged_epoch43() {
         let captured = physical_context();
         let mut current = captured;
         let mut replaced_fixed = captured.artifacts.fixed;
         replaced_fixed.artifact_digest[0] ^= 1;
         current.artifacts = build_artifact_set(&[
             replaced_fixed,
             captured.artifacts.triad_scalar,
             captured.artifacts.triad_sm80,
         ])
         .unwrap();
-        assert_eq!(captured.tuning_table_revision, 42);
+        assert_eq!(captured.tuning_table_revision, 43);
         assert_eq!(
             current.tuning_table_revision,
             captured.tuning_table_revision
         );
         assert_eq!(current.compiler, captured.compiler);
         assert_eq!(
             current.artifacts.triad_scalar,
             captured.artifacts.triad_scalar
         );
         assert_eq!(current.artifacts.triad_sm80, captured.artifacts.triad_sm80);
         assert_eq!(
             current.schedule_set_revision,
@@ -5628,25 +5650,25 @@ mod physical_launch_tests {
 #[cfg(test)]
 mod cache_and_header_tests {
     use super::*;
     use crate::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
     use crate::mamba_ssm::gpu::device::GpuDevice;
     use std::cell::Cell;
     #[cfg(target_os = "linux")]
     use std::sync::{Arc, Barrier};
 
     #[test]
     fn only_the_tuning_table_revision_moved() {
         assert_eq!(NUMERIC_ABI_REVISION, 5);
-        assert_eq!(TUNING_TABLE_REVISION, 42);
+        assert_eq!(TUNING_TABLE_REVISION, 43);
     }
 
     #[test]
     fn prepared_capture_manifest_matches_an_exact_optional_ordered_launch_set() {
         let launch = ResolvedGemmLaunchSet {
             launch_count: 2,
             ordered_digest: [7; 32],
         };
         let reordered = ResolvedGemmLaunchSet {
             launch_count: 2,
             ordered_digest: [8; 32],
         };
diff --git a/tests/gemm_bi_fixed_performance.rs b/tests/gemm_bi_fixed_performance.rs
index 547e19c9..88e77dff 100644
--- a/tests/gemm_bi_fixed_performance.rs
+++ b/tests/gemm_bi_fixed_performance.rs
@@ -31,69 +31,230 @@ use mamba_rs::mamba_ssm::gpu::kernel_identity::{
     ModuleKind, ResolvedGemmOp, TUNING_TABLE_REVISION, digest_hex,
 };
 use sha2::{Digest, Sha256};
 
 const WARMUPS: usize = 10;
 const ITERS: usize = 200;
 
 // Task6B has a separate mirrored-bracket protocol; historical harnesses above
 // and below retain their original record meanings.
 mod ada_s3_pair {
     use super::*;
 
+    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
+    pub(super) enum Stage {
+        PrePromotion42,
+        PostAuto43,
+    }
+
+    impl Stage {
+        fn schema(self) -> &'static str {
+            match self {
+                Self::PrePromotion42 => "MambaBiFixedAdaS3PairedV1",
+                Self::PostAuto43 => "MambaBiFixedAdaS3PostAutoPairedV1",
+            }
+        }
+
+        fn revision(self) -> u16 {
+            match self {
+                Self::PrePromotion42 => 42,
+                Self::PostAuto43 => 43,
+            }
+        }
+
+        fn controls(self) -> [&'static str; 3] {
+            match self {
+                Self::PrePromotion42 => [
+                    "MAMBA_FIXED_ADA_S3_PAIR",
+                    "MAMBA_FIXED_ADA_S3_WINDOWS",
+                    "MAMBA_FIXED_ADA_S3_DTYPES",
+                ],
+                Self::PostAuto43 => [
+                    "MAMBA_FIXED_ADA_S3_POST_PAIR",
+                    "MAMBA_FIXED_ADA_S3_POST_WINDOWS",
+                    "MAMBA_FIXED_ADA_S3_POST_DTYPES",
+                ],
+            }
+        }
+
+        fn arms(self) -> [&'static str; 3] {
+            match self {
+                Self::PrePromotion42 => ["AUTO", "S3", "Fast"],
+                Self::PostAuto43 => ["Swizzle", "AUTO", "Fast"],
+            }
+        }
+
+        fn ratios(self) -> [&'static str; 3] {
+            match self {
+                Self::PrePromotion42 => ["S3/AUTO", "AUTO/Fast", "S3/Fast"],
+                Self::PostAuto43 => ["AUTO/Swizzle", "Swizzle/Fast", "AUTO/Fast"],
+            }
+        }
+    }
+
     fn schedule(window: usize, start: usize) -> Vec<(usize, usize, usize, usize)> {
         assert!(start < 2);
         let reverse = (window + start) % 2 == 1;
         let comparisons = if reverse { [2, 1, 0] } else { [0, 1, 2] };
         let mut observations = Vec::with_capacity(12);
         for (traversal, comparison) in comparisons.into_iter().enumerate() {
             let (a, b) = [(0, 1), (2, 0), (2, 1)][comparison];
             let arms = if reverse { [b, a, a, b] } else { [a, b, b, a] };
             for (position, arm) in arms.into_iter().enumerate() {
                 observations.push((traversal, comparison, position, arm));
             }
         }
         observations
     }
 
-    const ARMS: [&str; 3] = ["AUTO", "S3", "Fast"];
-    const RATIOS: [&str; 3] = ["S3/AUTO", "AUTO/Fast", "S3/Fast"];
     const GUARD: usize = 256;
 
-    fn config(windows: &str, dtypes: &str) -> Result<(usize, Vec<usize>), String> {
+    fn control_allowed(stage: Stage, key: &str) -> bool {
+        if stage.controls().contains(&key) {
+            return true;
+        }
+        !key.starts_with("MAMBA_FIXED_ADA_")
+            && !key.starts_with("MAMBA_FIXED_VENDOR_")
+            && !matches!(
+                key,
+                "MAMBA_FIXED_AUTO_VENDOR_ROW"
+                    | "MAMBA_FIXED_AUTO_VENDOR_CELL"
+                    | "MAMBA_FIXED_AUTO_VENDOR_BIAS"
+                    | "MAMBA_FIXED_HALF_TILE_CANDIDATE"
+                    | "NVIDIA_TF32_OVERRIDE"
+            )
+    }
+
+    fn validate_controls<'a>(
+        stage: Stage,
+        controls: impl IntoIterator<Item = &'a str>,
+    ) -> Result<(), String> {
+        for key in controls {
+            if !control_allowed(stage, key) {
+                return Err(format!(
+                    "stale control {key} forbidden in literal S3 experiment"
+                ));
+            }
+        }
+        Ok(())
+    }
+
+    fn config(stage: Stage, windows: &str, dtypes: &str) -> Result<(usize, Vec<usize>), String> {
         let windows = match windows {
             "1" => 1,
             "21" => 21,
             "101" => 101,
             _ => return Err("windows must be exactly 1, 21 or 101".into()),
         };
         let dtypes = fixed_ada_direct_pair_filter("S3 dtype", &["bf16", "f16"], Some(dtypes))?;
-        if windows != 101 && dtypes != [0, 1] {
-            return Err("smoke/screen requires explicit bf16,f16".into());
+        match stage {
+            Stage::PrePromotion42 if windows != 101 && dtypes != [0, 1] => {
+                return Err("historical smoke/screen requires explicit bf16,f16".into());
+            }
+            Stage::PostAuto43 if !matches!(windows, 1 | 101) || dtypes != [0, 1] => {
+                return Err("post-AUTO requires windows1/101 and explicit bf16,f16".into());
+            }
+            _ => {}
         }
         Ok((windows, dtypes))
     }
 
     #[test]
-    fn configuration_cannot_mute_screen_or_accept_implicit_windows() {
-        assert_eq!(config("21", "bf16,f16").unwrap(), (21, vec![0, 1]));
-        assert_eq!(config("101", "f16").unwrap(), (101, vec![1]));
+    fn pre42_and_post43_configuration_cannot_mute_screen_or_accept_implicit_windows() {
+        assert_eq!(
+            config(Stage::PrePromotion42, "21", "bf16,f16").unwrap(),
+            (21, vec![0, 1])
+        );
+        assert_eq!(
+            config(Stage::PrePromotion42, "101", "f16").unwrap(),
+            (101, vec![1])
+        );
+        assert_eq!(
+            config(Stage::PostAuto43, "1", "bf16,f16").unwrap(),
+            (1, vec![0, 1])
+        );
+        assert_eq!(
+            config(Stage::PostAuto43, "101", "bf16,f16").unwrap(),
+            (101, vec![0, 1])
+        );
+        assert!(config(Stage::PostAuto43, "21", "bf16,f16").is_err());
+        assert!(config(Stage::PostAuto43, "101", "bf16").is_err());
+        assert!(config(Stage::PostAuto43, "101", "f16").is_err());
         for w in ["", "0", "20", "021", " 21", "100", "102"] {
-            assert!(config(w, "bf16,f16").is_err());
+            assert!(config(Stage::PrePromotion42, w, "bf16,f16").is_err());
         }
         for d in ["", "f32", "bf16,bf16", "bf16,", "f16"] {
-            assert!(config("21", d).is_err());
+            assert!(config(Stage::PrePromotion42, "21", d).is_err());
+        }
+    }
+
+    #[test]
+    fn pre42_and_post43_stage_controls_and_directions_are_disjoint() {
+        let pre = Stage::PrePromotion42;
+        let post = Stage::PostAuto43;
+        assert_eq!(pre.schema(), "MambaBiFixedAdaS3PairedV1");
+        assert_eq!(post.schema(), "MambaBiFixedAdaS3PostAutoPairedV1");
+        assert_eq!(pre.revision(), 42);
+        assert_eq!(post.revision(), 43);
+        assert_eq!(pre.arms(), ["AUTO", "S3", "Fast"]);
+        assert_eq!(post.arms(), ["Swizzle", "AUTO", "Fast"]);
+        assert_eq!(pre.ratios(), ["S3/AUTO", "AUTO/Fast", "S3/Fast"]);
+        assert_eq!(post.ratios(), ["AUTO/Swizzle", "Swizzle/Fast", "AUTO/Fast"]);
+        assert!(validate_controls(pre, pre.controls()).is_ok());
+        assert!(validate_controls(post, post.controls()).is_ok());
+        assert!(validate_controls(pre, post.controls()).is_err());
+        assert!(validate_controls(post, pre.controls()).is_err());
+        for stale in [
+            "MAMBA_FIXED_AUTO_VENDOR_ROW",
+            "MAMBA_FIXED_AUTO_VENDOR_CELL",
+            "MAMBA_FIXED_AUTO_VENDOR_BIAS",
+            "MAMBA_FIXED_HALF_TILE_CANDIDATE",
+            "MAMBA_FIXED_VENDOR_LEGACY_CONTROL",
+            "NVIDIA_TF32_OVERRIDE",
+        ] {
+            assert!(
+                validate_controls(pre, [stale]).is_err(),
+                "pre accepted {stale}"
+            );
+            assert!(
+                validate_controls(post, [stale]).is_err(),
+                "post accepted {stale}"
+            );
         }
     }
 
+    fn unchanged_inputs(
+        saved_a: &[u8],
+        saved_b: &[u8],
+        observed_a: &[u8],
+        observed_b: &[u8],
+    ) -> Result<(), String> {
+        if observed_a != saved_a {
+            return Err("immutable A changed before timing".into());
+        }
+        if observed_b != saved_b {
+            return Err("immutable B changed before timing".into());
+        }
+        Ok(())
+    }
+
+    #[test]
+    fn pre_timing_input_gate_rejects_changed_a_or_b_bytes() {
+        let saved_a = [1, 2, 3, 4];
+        let saved_b = [5, 6, 7, 8];
+        assert!(unchanged_inputs(&saved_a, &saved_b, &saved_a, &saved_b).is_ok());
+        assert!(unchanged_inputs(&saved_a, &saved_b, &[1, 2, 0, 4], &saved_b).is_err());
+        assert!(unchanged_inputs(&saved_a, &saved_b, &saved_a, &[5, 6, 0, 8]).is_err());
+    }
+
     fn pair_ratio(samples: &[(usize, f64)], comparison: usize) -> Result<f64, String> {
         let (a, b) = [(0, 1), (2, 0), (2, 1)][comparison];
         if samples.len() != 4
             || samples.iter().any(|(_, t)| !t.is_finite() || *t <= 0.0)
             || samples.iter().filter(|(arm, _)| *arm == a).count() != 2
             || samples.iter().filter(|(arm, _)| *arm == b).count() != 2
         {
             return Err("invalid mirrored observations".into());
         }
         Ok(samples
             .iter()
             .filter(|(arm, _)| *arm == b)
@@ -341,88 +502,93 @@ mod ada_s3_pair {
                 )
                 .unwrap();
                 decoded = format!(
                     ",\"pointers\":{pointers:?},\"bundle\":{bundle:?},\"abi\":[[0,8],[8,8],[16,8],[24,8],[32,32]],\"sixth_rejected\":true"
                 );
             }
             entries.push(format!("{{\"symbol\":\"{}\",\"grid\":[{},{},{}],\"block\":[{},{},{}],\"shared_bytes\":{}{} }}",
                 fixed_sm120_tf32_bd_json_escape(symbol),p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ,p.sharedMemBytes,decoded));
         }
         (format!("[{}]", entries.join(",")), count)
     }
 
-    pub(super) fn run() {
+    pub(super) fn run(stage: Stage) {
+        let controls = stage.controls();
         assert_eq!(
-            std::env::var("MAMBA_FIXED_ADA_S3_PAIR").as_deref(),
+            std::env::var(controls[0]).as_deref(),
             Ok("1"),
-            "explicit S3 pair enable required"
+            "explicit stage-specific S3 pair enable required"
         );
         assert!(!cfg!(debug_assertions), "S3 pairing requires release");
-        for (key, _) in std::env::vars_os() {
-            let key = key.to_string_lossy();
-            if (key.starts_with("MAMBA_FIXED_ADA_")
-                && !matches!(
-                    key.as_ref(),
-                    "MAMBA_FIXED_ADA_S3_PAIR"
-                        | "MAMBA_FIXED_ADA_S3_WINDOWS"
-                        | "MAMBA_FIXED_ADA_S3_DTYPES"
-                ))
-                || key.starts_with("MAMBA_FIXED_VENDOR_")
-                || key == "NVIDIA_TF32_OVERRIDE"
-            {
-                panic!("stale control {key} forbidden in literal S3 experiment");
-            }
-        }
-        let dtype_text = std::env::var("MAMBA_FIXED_ADA_S3_DTYPES").expect("explicit dtypes");
+        let environment: Vec<_> = std::env::vars_os()
+            .map(|(key, _)| key.to_string_lossy().into_owned())
+            .collect();
+        validate_controls(stage, environment.iter().map(String::as_str)).unwrap();
+        let dtype_text = std::env::var(controls[2]).expect("explicit dtypes");
         let (windows, dtypes) = config(
-            &std::env::var("MAMBA_FIXED_ADA_S3_WINDOWS").expect("explicit windows"),
+            stage,
+            &std::env::var(controls[1]).expect("explicit windows"),
             &dtype_text,
         )
         .unwrap();
         fixed_sm120_tf32_bd_environment_preflight("Ada S3 mirrored paired").unwrap();
         let device = GpuDevice::new(0).expect("Ada device");
         assert_eq!(device.compute_capability, (8, 9));
         assert_eq!(device.multiprocessor_count(), 142);
-        assert_eq!(TUNING_TABLE_REVISION, 42);
+        assert_eq!(TUNING_TABLE_REVISION, stage.revision());
         let ctx = GpuCtx::new(&device).expect("Ada context");
         let compiler = ctx.kernels.compiler_identity();
         assert!(compiler.nvrtc_library_known);
         let toolkit = std::env::var("S3_TOOLKIT").expect("toolkit binding");
         assert_eq!(
             toolkit,
             format!("{}.{}", compiler.nvrtc_version.0, compiler.nvrtc_version.1)
         );
         assert!(matches!(
             compiler.nvrtc_version,
             (12, 8) | (13, 0) | (13, 2)
         ));
+        if stage == Stage::PostAuto43 {
+            assert_eq!(
+                compiler.nvrtc_version,
+                (13, 2),
+                "post-AUTO qualification is CUDA13.2-only"
+            );
+        }
         let artifact = ctx.kernels.artifact_set_identity().fixed;
         let source_hash = digest_hex(
             &Sha256::digest(std::fs::read("tests/gemm_bi_fixed_performance.rs").unwrap()).into(),
         );
         let binary_hash = digest_hex(
             &Sha256::digest(std::fs::read(std::env::current_exe().unwrap()).unwrap()).into(),
         );
         assert_eq!(
             source_hash,
             std::env::var("S3_SOURCE_SHA").expect("source binding")
         );
         assert_eq!(
             binary_hash,
             std::env::var("S3_BINARY_SHA").expect("binary binding")
         );
         let mode = modes(&ctx);
-        let metadata = format!(
-            "\"schema\":\"MambaBiFixedAdaS3PairedV1\",\"toolkit\":\"{toolkit}\",\"shape\":[4621,768,2304],\"bias\":false,\"alpha\":1,\"beta\":0,\"revision\":42"
-        );
+        let arms = stage.arms();
+        let directions = stage.ratios();
+        let metadata = match stage {
+            Stage::PrePromotion42 => format!(
+                "\"schema\":\"MambaBiFixedAdaS3PairedV1\",\"toolkit\":\"{toolkit}\",\"shape\":[4621,768,2304],\"bias\":false,\"alpha\":1,\"beta\":0,\"revision\":42"
+            ),
+            Stage::PostAuto43 => format!(
+                "\"schema\":\"MambaBiFixedAdaS3PostAutoPairedV1\",\"stage\":\"post_auto\",\"toolkit\":\"{toolkit}\",\"shape\":[4621,768,2304],\"bias\":false,\"alpha\":1,\"beta\":0,\"revision\":43"
+            ),
+        };
         println!(
             "{{{metadata},\"kind\":\"identity\",\"uuid\":\"GPU-d1edd7be-e88d-aed6-047d-622163306f0e\",\"cc\":\"8.9\",\"sm_count\":142,\"source_sha\":\"{source_hash}\",\"binary_sha\":\"{binary_hash}\",\"fixed_source_digest\":\"{}\",\"fixed_invocation_digest\":\"{}\",\"fixed_artifact_digest\":\"{}\",\"header_manifest_digest\":\"{}\",\"nvrtc_library_domain\":\"{}\",\"nvrtc_library_known\":true,\"dtypes\":\"{dtype_text}\",\"windows\":{windows},\"logical_ops\":20,\"warmup_eager\":128,\"percentile\":\"round((len-1)*fraction)\",{mode}}}",
             digest_hex(&compiler.source_digest),
             digest_hex(&compiler.invocation_digest),
             digest_hex(&artifact.artifact_digest),
             digest_hex(&compiler.header_manifest_digest),
             digest_hex(&compiler.nvrtc_library_domain)
         );
         let shape = FixedShape {
             m: 4621,
             k: 768,
             n: 2304,
@@ -456,43 +622,64 @@ mod ada_s3_pair {
                     c: TypedPtr {
                         ptr: owner.cached_ptr() + GUARD as u64,
                         dtype,
                     },
                     x: typed(&a, dtype),
                     w: typed(&b, dtype),
                     bias_ptr: None,
                 })
                 .collect();
             assert!(operands.iter().all(|ops| ops.c.ptr % 256 == 0));
             let launch = |arm: usize| match arm {
                 0 => {
-                    let actual = launch_fixed_auto_vendor_custom(&ctx, operands[0], shape);
-                    assert_eq!(
-                        actual,
-                        FixedTile::Tc128Sm89Swizzle,
-                        "actual AUTO must remain rev42 incumbent"
-                    );
+                    if stage == Stage::PrePromotion42 {
+                        let actual = launch_fixed_auto_vendor_custom(&ctx, operands[0], shape);
+                        assert_eq!(
+                            actual,
+                            FixedTile::Tc128Sm89Swizzle,
+                            "historical actual AUTO must remain rev42 incumbent"
+                        );
+                    } else {
+                        fixed_forward_with_tile(
+                            &ctx,
+                            operands[0],
+                            shape,
+                            FixedTile::Tc128Sm89Swizzle,
+                        )
+                        .expect("public forced Swizzle control");
+                    }
+                }
+                1 => {
+                    if stage == Stage::PrePromotion42 {
+                        fixed_forward_with_tile(&ctx, operands[1], shape, FixedTile::Tc128Sm89S3)
+                            .expect("public S3 force");
+                    } else {
+                        let actual = launch_fixed_auto_vendor_custom(&ctx, operands[1], shape);
+                        assert_eq!(
+                            actual,
+                            FixedTile::Tc128Sm89S3,
+                            "post-AUTO must use actual public S3 selection"
+                        );
+                    }
                 }
-                1 => fixed_forward_with_tile(&ctx, operands[1], shape, FixedTile::Tc128Sm89S3)
-                    .expect("public S3 force"),
                 2 => fast(&ctx, operands[2], shape),
                 _ => unreachable!(),
             };
             for arm in 0..3 {
                 launch(arm);
             }
             let expected: Vec<_> = owners.iter().map(|o| raw(&ctx, o)).collect();
             assert_eq!(
                 expected[0], expected[1],
-                "S3/AUTO exact homogeneous storage bits"
+                "S3 and Swizzle exact homogeneous storage bits"
             );
             let reference = DtypedBuf::zeros(&ctx.stream, elements, WeightDtype::F32).unwrap();
             fixed_ada_vendor_launch(
                 &ctx,
                 FixedFwdOperands {
                     c: typed(&reference, WeightDtype::F32),
                     ..operands[0]
                 },
                 shape,
                 cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
             );
             let reference_bits = f32_bits(&ctx, &reference, elements);
@@ -537,25 +724,25 @@ mod ada_s3_pair {
                 for graph in [&g, &g20] {
                     for _ in 0..2 {
                         assert!(
                             output_gate(&ctx, &owners[arm], &expected[arm], || graph
                                 .launch()
                                 .unwrap()),
                             "independent graph overwrite/bits gate"
                         );
                     }
                 }
                 println!(
                     "{{{metadata},\"kind\":\"physical\",\"dtype\":\"{dtype_name}\",\"arm\":\"{}\",\"pointers\":[{},{},{},0],\"allocation\":[{},{}],\"guard_bytes\":256,\"one\":{one_inventory},\"twenty\":{twenty_inventory},\"numerical_error\":{},\"tolerance\":{},\"reference\":\"PEDANTIC_F32\",\"eager_repeats\":2,\"graph_repeats\":2,\"poison_upload_verified\":true,\"repeat_bits\":true,\"guards\":true}}",
-                    ARMS[arm],
+                    arms[arm],
                     operands[arm].c.ptr,
                     operands[arm].x.ptr,
                     operands[arm].w.ptr,
                     owners[arm].cached_ptr(),
                     owners[arm].size_bytes(),
                     errors[arm],
                     row.custom_tolerance
                 );
                 one.push(g);
                 twenty.push(g20);
             }
             let noop = unsafe { capture_into_graph(&ctx.stream, || Ok(())) }
@@ -567,25 +754,33 @@ mod ada_s3_pair {
             for path in ["eager", "graph"] {
                 for start in 0..2 {
                     for arm in 0..3 {
                         assert!(output_gate(&ctx, &owners[arm], &expected[arm], || launch(
                             arm
                         )));
                         for graph in [&one[arm], &twenty[arm]] {
                             assert!(output_gate(&ctx, &owners[arm], &expected[arm], || graph
                                 .launch()
                                 .unwrap()));
                         }
                     }
-                    // Allocate all collection capacity before warmup. Nothing prints,
+                    unchanged_inputs(
+                        &a_raw,
+                        &b_raw,
+                        &fixed_explicit_vendor_raw_bytes(&ctx, &a),
+                        &fixed_explicit_vendor_raw_bytes(&ctx, &b),
+                    )
+                    .expect("pre-timing immutable input gate");
+                    // The required input readback gate is the final action before
+                    // timing preparation. After capacity is reserved, nothing prints,
                     // downloads, poisons, compiles or allocates device memory until
                     // all windows of this configuration have completed.
                     let schedule: Vec<_> = (0..windows)
                         .flat_map(|w| schedule(w, start).into_iter().map(move |s| (w, s)))
                         .collect();
                     let mut samples = Vec::with_capacity(12 * windows);
                     for _ in 0..128 {
                         for arm in 0..3 {
                             launch(arm);
                         }
                     }
                     for arm in 0..3 {
@@ -629,47 +824,47 @@ mod ada_s3_pair {
                     );
                     let key = format!(
                         "{metadata},\"dtype\":\"{dtype_name}\",\"path\":\"{path}\",\"start_parity\":{start}"
                     );
                     let mut ratios: [Vec<f64>; 3] =
                         std::array::from_fn(|_| Vec::with_capacity(windows));
                     for (chronology, &(w, traversal, comparison, position, arm, us)) in
                         samples.iter().enumerate()
                     {
                         let order = if (w + start) % 2 == 0 { "ABBA" } else { "BAAB" };
                         println!(
                             "{{{key},\"kind\":\"sample\",\"chronology\":{chronology},\"window\":{w},\"comparison\":{comparison},\"traversal\":{traversal},\"order\":\"{order}\",\"position\":{position},\"arm\":\"{}\",\"logical_ops\":20,\"us\":{us}}}",
-                            ARMS[arm]
+                            arms[arm]
                         );
                     }
                     for (bracket, observations) in samples.chunks_exact(4).enumerate() {
                         let (w, traversal, comparison, _, _, _) = observations[0];
                         let pair: Vec<_> = observations.iter().map(|o| (o.4, o.5)).collect();
                         let ratio = pair_ratio(&pair, comparison).unwrap();
                         ratios[comparison].push(ratio);
                         println!(
                             "{{{key},\"kind\":\"pair\",\"window\":{w},\"comparison\":{comparison},\"traversal\":{traversal},\"observations\":[{},{},{},{}],\"ratio\":{ratio}}}",
                             bracket * 4,
                             bracket * 4 + 1,
                             bracket * 4 + 2,
                             bracket * 4 + 3
                         );
                     }
                     for comparison in 0..3 {
                         ratios[comparison].sort_by(f64::total_cmp);
                         let p50 = percentile(&ratios[comparison], 0.5);
                         let p95 = percentile(&ratios[comparison], 0.95);
                         println!(
                             "{{{key},\"kind\":\"summary\",\"comparison\":{comparison},\"direction\":\"{}\",\"windows\":{windows},\"p50\":{p50},\"p95\":{p95}}}",
-                            RATIOS[comparison]
+                            directions[comparison]
                         );
                     }
                     println!(
                         "{{{key},\"kind\":\"configuration_complete\",\"samples\":{},\"pairs\":{},\"summaries\":3,\"pre_post_bits\":true,\"pre_post_graphs\":true,\"guards\":true,\"immutable_inputs\":true,\"noop_rejected\":true}}",
                         12 * windows,
                         3 * windows
                     );
                     configurations += 1;
                 }
             }
         }
         println!(
@@ -716,25 +911,31 @@ mod ada_s3_pair {
                 (2, 0, 3, 1)
             ]
         );
         assert_eq!(schedule(1, 0), schedule(0, 1));
         assert_eq!(schedule(1, 1), schedule(0, 0));
     }
 }
 
 // Separate protocol: do not reinterpret the historical triple comparator below.
 #[test]
 #[ignore = "requires explicit MAMBA_FIXED_ADA_S3_PAIR=1 and exclusive pinned Ada; production AUTO/S3/Fast mirrored brackets"]
 fn fixed_ada_half_s3_auto_fast_paired() {
-    ada_s3_pair::run();
+    ada_s3_pair::run(ada_s3_pair::Stage::PrePromotion42);
+}
+
+#[test]
+#[ignore = "requires explicit MAMBA_FIXED_ADA_S3_POST_PAIR=1 and exclusive pinned Ada CUDA13.2; forced Swizzle/actual AUTO43/Fast mirrored brackets"]
+fn fixed_ada_half_s3_post_auto_fast_paired() {
+    ada_s3_pair::run(ada_s3_pair::Stage::PostAuto43);
 }
 
 #[path = "support/fixed_sm89_exact_n64_admission.rs"]
 mod exact_n64_admission;
 
 #[test]
 #[ignore = "requires exclusive pinned Ada, actual NVRTC candidate admission and paired evidence output"]
 fn fixed_sm89_exact_n64_paired_admission() {
     exact_n64_admission::run();
 }
 
 fn typed(buffer: &DtypedBuf, dtype: WeightDtype) -> TypedPtr {
diff --git a/tests/gemm_bi_fixed_sm89_pipeline.rs b/tests/gemm_bi_fixed_sm89_pipeline.rs
index d9244aff..fffc158c 100644
--- a/tests/gemm_bi_fixed_sm89_pipeline.rs
+++ b/tests/gemm_bi_fixed_sm89_pipeline.rs
@@ -15,51 +15,55 @@ use mamba_rs::mamba_ssm::gpu::kernel_identity::TUNING_TABLE_REVISION;
 
 const CANDIDATE: FixedTile = FixedTile::Tc128Sm89Pipeline;
 const SWIZZLE_CANDIDATE: FixedTile = FixedTile::Tc128Sm89Swizzle;
 const S3_CANDIDATE: FixedTile = FixedTile::Tc128Sm89S3;
 const RUNGS: [FixedTile; 5] = [
     FixedTile::Tc16,
     FixedTile::Tc64,
     FixedTile::Tc128,
     FixedTile::TcW64,
     FixedTile::TcWn64,
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
 fn fixed_sm89_half_swizzle_has_a_distinct_forced_route() {
     assert_ne!(SWIZZLE_CANDIDATE, CANDIDATE);
     assert_eq!(format!("{SWIZZLE_CANDIDATE:?}"), "Tc128Sm89Swizzle");
 }
@@ -169,25 +173,25 @@ fn fixed_sm89_half_swizzle_forced_hot_a_e_prefix_view_graph_bits() {
 #[ignore = "requires exact Ada CC8.9 and the admitted forced half s3"]
 fn fixed_sm89_half_s3_forced_hot_a_e_prefix_view_graph_bits() {
     fixed_sm89_half_hot_cell_prefix_view_graph_bits(Some(S3_CANDIDATE));
 }
 
 fn fixed_sm89_half_hot_cell_prefix_view_graph_bits(forced: Option<FixedTile>) {
     let device = GpuDevice::new(0).expect("CUDA device");
     assert_eq!(device.compute_capability, (8, 9));
     assert_eq!(device.multiprocessor_count(), 142);
     let ctx = GpuCtx::new(&device).expect("NVRTC context");
     if forced.is_none() {
         let compiler = ctx.kernels.compiler_identity();
-        assert_eq!(TUNING_TABLE_REVISION, 42);
+        assert_eq!(TUNING_TABLE_REVISION, 43);
         assert!(compiler.nvrtc_library_known);
         assert!(matches!(
             compiler.nvrtc_version,
             (12, 8) | (13, 0) | (13, 2)
         ));
         assert!(ctx.kernels.fixed_sm89_half_pipeline.is_some());
         assert!(ctx.kernels.fixed_sm89_half_pipeline_rejection.is_none());
         assert!(ctx.kernels.fixed_sm89_half_swizzle.is_some());
         assert!(ctx.kernels.fixed_sm89_half_swizzle_rejection.is_none());
     }
     for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
         for (hot_m, k, n) in [
@@ -276,31 +280,31 @@ fn fixed_sm89_half_hot_cell_prefix_view_graph_bits(forced: Option<FixedTile>) {
                                         &ctx,
                                         view.c,
                                         view.x,
                                         view.w,
                                         view.bias_ptr,
                                         (m, k, n),
                                     )
                                 }
                             };
                             let picked = run().expect("hot-cell launch");
                             if forced.is_none() {
                                 if m == hot_m && output_offset == 8 {
-                                    let expected = expected_ada_half_auto_v42(
+                                    let expected = expected_ada_half_auto_v43(
                                         ctx.kernels.compiler_identity().nvrtc_version,
                                         dtype,
                                         FixedShape { m, k, n },
                                         has_bias,
                                     )
-                                    .expect("literal revision-42 hot-cell expectation");
+                                    .expect("literal revision-43 hot-cell expectation");
                                     assert_eq!(
                                         picked, expected,
                                         "AUTO promotion scope {dtype:?} M={m} K={k} N={n} row={row_offset} out={output_offset} bias={has_bias}"
                                     );
                                 } else {
                                     assert!(
                                         !matches!(
                                             picked,
                                             CANDIDATE | SWIZZLE_CANDIDATE | S3_CANDIDATE
                                         ),
                                         "AUTO candidate escaped scope {dtype:?} M={m} K={k} N={n} row={row_offset} out={output_offset} bias={has_bias}: {picked:?}"
                                     );

