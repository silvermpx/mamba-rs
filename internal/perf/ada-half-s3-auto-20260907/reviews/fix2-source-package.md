# Task6C host fix2 scoped review package

Requirements: ada-half-s3-auto-task6c-fix2-brief.md. Original findings: ada-half-s3-auto-task6c-host-source-review.md. Base utilities hashes9c73297e/a4e322bc/dc3fb73c; currenta18899f5/2d0634fa/844c22c3. Rust/performance source7d61e9f1 and all three binaries/build bindings unchanged.
Original source preserved in evidence/fix2-pre and root scratch copy with independently matching hashes. Only host I2 dependency pins and I3 adversarial coverage are under this review. Report and RED/GREEN files contain 11-test RED (1failure,2errors,exit1) then11passGREEN(exit0). No new GPU measurement required by this change.

## Fix diff

diff --git a/.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/ada-half-s3-auto-host-before-fix2/analyze.py b/internal/perf/ada-half-s3-auto-20260907/analyze.py
index a5198a3a..0ca8d89a 100644
--- a/.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/ada-half-s3-auto-host-before-fix2/analyze.py
+++ b/internal/perf/ada-half-s3-auto-20260907/analyze.py
@@ -4,40 +4,62 @@ import argparse
 import copy
 import hashlib
 import importlib.util
 import json
 from pathlib import Path
 import re
 import sys
 
 HERE = Path(__file__).resolve().parent
 PRE_PATH = HERE.parent / 'ada-half-s3-paired-20260907' / 'analyze.py'
-PRE_SPEC = importlib.util.spec_from_file_location('ada_s3_pair_frozen_analyze', PRE_PATH)
-PRE = importlib.util.module_from_spec(PRE_SPEC)
-PRE_SPEC.loader.exec_module(PRE)
+FROZEN_PRE_ANALYZER_SHA = (
+    '0c1391f47ec0253720b66733e950b8f9ad67ba112027b81e2d67eda9a1e63f30'
+)
+FROZEN_PRE_WRAPPER_SHA = (
+    'ce0de6882fa45835ba81b90d02a6cc085ed32226e2462b72260ce9cacd0fd713'
+)
+
+
+def load_pinned(name, path, expected_sha):
+    observed = hashlib.sha256(Path(path).read_bytes()).hexdigest()
+    if observed != expected_sha:
+        raise ValueError(f'frozen dependency digest mismatch: {path}')
+    spec = importlib.util.spec_from_file_location(name, path)
+    module = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(module)
+    return module
+
+
+PRE = load_pinned('ada_s3_pair_frozen_analyze', PRE_PATH,
+                  FROZEN_PRE_ANALYZER_SHA)
 
 SCHEMA = 'MambaBiFixedAdaS3PostAutoPairedV1'
 STAGE = 'post_auto'
 ARMS = ['Swizzle', 'AUTO', 'Fast']
 DIRECTIONS = ['AUTO/Swizzle', 'Swizzle/Fast', 'AUTO/Fast']
 BASE = {
     'schema': SCHEMA,
     'stage': STAGE,
     'toolkit': '13.2',
     'shape': [4621, 768, 2304],
     'bias': False,
     'alpha': 1,
     'beta': 0,
     'revision': 43,
 }
 
 
+def validate_frozen_binding(binding):
+    PRE.check(binding.get('frozen_pre_wrapper_sha') == FROZEN_PRE_WRAPPER_SHA,
+              'foreign bound frozen wrapper digest')
+
+
 def records(path):
     result = []
     marker = '{"schema":"' + SCHEMA + '"'
     for line in Path(path).read_text().splitlines():
         offset = line.find(marker)
         if offset >= 0:
             result.append(json.loads(line[offset:]))
     return result
 
 
@@ -50,20 +72,21 @@ def _pre_rows(rows):
         row['revision'] = 42
         row.pop('stage', None)
         if 'arm' in row:
             row['arm'] = arm_map[row['arm']]
         if 'direction' in row:
             row['direction'] = direction_map[row['direction']]
     return mapped
 
 
 def verify(rows, binding, qualification, windows, dtypes):
+    validate_frozen_binding(binding)
     PRE.check(binding.get('toolkit') == '13.2', 'post-AUTO binding must be CUDA13.2')
     PRE.check(windows in (1, 101), 'post-AUTO windows must be exactly1 or101')
     PRE.check(dtypes == ['bf16', 'f16'], 'post-AUTO requires exact two-dtype closure')
     PRE.check(qualification.get('tuning_table_revision') == 42,
               'Task6A compiled-artifact qualification must retain historical revision42')
     PRE.check(rows, 'missing post-AUTO records')
     for row in rows:
         PRE.equal_fields(row, BASE)
         if 'arm' in row:
             PRE.check(row['arm'] in ARMS, 'wrong post-AUTO arm')
@@ -82,25 +105,25 @@ def verify(rows, binding, qualification, windows, dtypes):
             summary['direction'] = direction_map.get(summary['direction'], summary['direction'])
     report['schema'] = SCHEMA
     report['stage'] = STAGE
     report['compiled_identity_revision'] = qualification['tuning_table_revision']
     report['routing_revision'] = 43
     return report
 
 
 def verify_run(directory, binding_path, qualification_path, ssh_log):
     run_path = HERE.parent / 'ada-half-s3-paired-20260907' / 'run.py'
-    run_spec = importlib.util.spec_from_file_location('ada_s3_pair_frozen_run', run_path)
-    run = importlib.util.module_from_spec(run_spec)
-    run_spec.loader.exec_module(run)
     directory = Path(directory)
     binding = json.loads(Path(binding_path).read_text())
+    validate_frozen_binding(binding)
+    run = load_pinned('ada_s3_pair_frozen_run', run_path,
+                      FROZEN_PRE_WRAPPER_SHA)
     qualification = json.loads(Path(qualification_path).read_text())
     result = json.loads((directory / 'result.json').read_text())
     PRE.equal_fields(result, {
         'toolkit': '13.2',
         'binary_sha': binding['binary_sha'],
         'source_sha': binding['inputs']['tests/gemm_bi_fixed_performance.rs'],
         'stage': STAGE,
     })
     PRE.check(
         hashlib.sha256((directory / 'test.log').read_bytes()).hexdigest()
@@ -138,21 +161,22 @@ def verify_run(directory, binding_path, qualification_path, ssh_log):
               'COMMAND_EXIT conflicts with saved test exit')
     run.closure(result['test_exit'], result['post_exit'],
                 int(wrapper_exit[1]), int(ssh_exit[1]))
     PRE.check('test result: ok. 1 passed; 0 failed;' in
               (directory / 'test.log').read_text(), 'test harness exit marker')
     report = verify(records(directory / 'test.log'), binding, qualification,
                     result['windows'], result['dtypes'].split(','))
     report['validation'] = {
         'revision': 'post-auto43-exact-same-attempt-adapter-v1',
         'analyzer_sha': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
-        'frozen_pre_analyzer_sha': hashlib.sha256(PRE_PATH.read_bytes()).hexdigest(),
+        'frozen_pre_analyzer_sha': FROZEN_PRE_ANALYZER_SHA,
+        'frozen_pre_wrapper_sha': FROZEN_PRE_WRAPPER_SHA,
         'ssh_log_sha': hashlib.sha256(Path(ssh_log).read_bytes()).hexdigest(),
         'binding_sha': hashlib.sha256(Path(binding_path).read_bytes()).hexdigest(),
         'qualification_sha': hashlib.sha256(Path(qualification_path).read_bytes()).hexdigest(),
     }
     return report
 
 
 if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('directory')
diff --git a/.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/ada-half-s3-auto-host-before-fix2/run.py b/internal/perf/ada-half-s3-auto-20260907/run.py
index d7f8da94..556a6a68 100644
--- a/.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/ada-half-s3-auto-host-before-fix2/run.py
+++ b/internal/perf/ada-half-s3-auto-20260907/run.py
@@ -1,39 +1,64 @@
 #!/usr/bin/env python3
 """Task6C isolated all-toolkit build and CUDA13.2 post-AUTO run wrapper."""
 import importlib.util
+import hashlib
 import json
 import os
 from pathlib import Path
 import re
 import stat
 import sys
 
 HERE = Path(__file__).resolve().parent
 FROZEN_PATH = HERE.parent / 'ada-half-s3-paired-20260907' / 'run.py'
-FROZEN_SPEC = importlib.util.spec_from_file_location('ada_s3_pair_frozen_run', FROZEN_PATH)
-FROZEN = importlib.util.module_from_spec(FROZEN_SPEC)
-FROZEN_SPEC.loader.exec_module(FROZEN)
+FROZEN_PRE_WRAPPER_SHA = (
+    'ce0de6882fa45835ba81b90d02a6cc085ed32226e2462b72260ce9cacd0fd713'
+)
+
+
+def sha(path):
+    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
+
+
+def load_pinned(name, path, expected_sha):
+    if sha(path) != expected_sha:
+        raise ValueError(f'frozen dependency digest mismatch: {path}')
+    spec = importlib.util.spec_from_file_location(name, path)
+    module = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(module)
+    return module
+
+
+FROZEN = load_pinned('ada_s3_pair_frozen_run', FROZEN_PATH,
+                     FROZEN_PRE_WRAPPER_SHA)
 
 ROOT = Path('/root/mamba-ada-half-s3-auto-fix1-20260907')
 EVIDENCE = Path('/root/evidence-ada-half-s3-auto-fix1-20260907')
 TOOLKITS = {'12.8': ('128', '12080'), '13.0': ('130', '13000'),
             '13.2': ('132', '13020')}
 POST_CONTROLS = {
     'MAMBA_FIXED_ADA_S3_POST_PAIR',
     'MAMBA_FIXED_ADA_S3_POST_WINDOWS',
     'MAMBA_FIXED_ADA_S3_POST_DTYPES',
 }
 
 FROZEN.ROOT = ROOT
 
 
+def validate_frozen_binding(binding):
+    FROZEN.require(sha(FROZEN_PATH) == FROZEN_PRE_WRAPPER_SHA,
+                   'frozen wrapper digest changed')
+    FROZEN.require(binding.get('frozen_pre_wrapper_sha') == FROZEN_PRE_WRAPPER_SHA,
+                   'foreign bound frozen wrapper digest')
+
+
 def forbidden_control(key):
     return ((key.startswith('MAMBA_FIXED_ADA_') and key not in POST_CONTROLS)
             or key.startswith('MAMBA_FIXED_VENDOR_')
             or key in {'MAMBA_FIXED_AUTO_VENDOR_ROW',
                        'MAMBA_FIXED_AUTO_VENDOR_CELL',
                        'MAMBA_FIXED_AUTO_VENDOR_BIAS',
                        'MAMBA_FIXED_HALF_TILE_CANDIDATE',
                        'NVIDIA_TF32_OVERRIDE'})
 
 
@@ -88,36 +113,37 @@ def main():
             'feature': 'cuda,cudarc/cuda-' + feature,
             'binary': str(binaries[0]),
             'binary_sha': FROZEN.sha(binaries[0]),
             'inputs': inputs,
             'cache': str(cache),
             'cache_mode': oct(stat.S_IMODE(cache.stat().st_mode)),
             'tools': {str(p): FROZEN.sha(p) for p in [cuda / 'bin/nvcc', cuda / 'bin/ptxas']},
             'libraries': {str(p): FROZEN.sha(p)
                           for pattern in ('libnvrtc.so.*', 'libcublas.so.*')
                           for p in (cuda / 'targets/x86_64-linux/lib').glob(pattern)},
-            'frozen_pre_wrapper_sha': FROZEN.sha(FROZEN_PATH),
+            'frozen_pre_wrapper_sha': FROZEN_PRE_WRAPPER_SHA,
         }
         with build_binding.open('x') as output:
             json.dump(binding, output, indent=2)
             output.write('\n')
         print('BUILD_COMPLETE ' + json.dumps({k: v for k, v in binding.items()
                                               if k != 'inputs'}), flush=True)
         return
 
     FROZEN.require(operation == 'run' and len(options) == 2,
                    'run requires windows dtypes')
     FROZEN.require(toolkit == '13.2', 'post-AUTO runtime is CUDA13.2-only')
     windows, dtypes = options
     FROZEN.require(windows in ('1', '101'), 'post-AUTO windows must be 1 or 101')
     FROZEN.require(dtypes == 'bf16,f16', 'post-AUTO requires both dtypes')
     binding = json.loads(build_binding.read_text())
+    validate_frozen_binding(binding)
     binary = Path(binding['binary'])
     inputs = FROZEN.source_inputs()
     FROZEN.validate_binding(binding, toolkit, inputs, FROZEN.sha(binary))
     FROZEN.require(stat.S_IMODE(cache.stat().st_mode) == 0o700,
                    'cache is not private0700')
     for path, digest in (binding['tools'] | binding['libraries']).items():
         FROZEN.require(FROZEN.sha(path) == digest, 'tool/library changed ' + path)
     env.update(MAMBA_FIXED_ADA_S3_POST_PAIR='1',
                MAMBA_FIXED_ADA_S3_POST_WINDOWS=windows,
                MAMBA_FIXED_ADA_S3_POST_DTYPES=dtypes,
@@ -125,20 +151,21 @@ def main():
                S3_SOURCE_SHA=binding['inputs']['tests/gemm_bi_fixed_performance.rs'],
                S3_BINARY_SHA=binding['binary_sha'])
     FROZEN.telemetry('PRE', directory, env)
     test_exit = FROZEN.command(
         [str(binary), '--ignored', '--exact',
          'fixed_ada_half_s3_post_auto_fast_paired', '--nocapture', '--test-threads=1'],
         directory / 'test.log', env, cwd=ROOT)
     post_exit = 0
     try:
         FROZEN.telemetry('POST', directory, env)
+        validate_frozen_binding(binding)
         FROZEN.validate_binding(binding, toolkit, FROZEN.source_inputs(), FROZEN.sha(binary))
     except Exception as error:
         print('POSTCHECK_FAILURE ' + str(error), flush=True)
         post_exit = 1
     result = {
         'test_exit': test_exit,
         'post_exit': post_exit,
         'toolkit': toolkit,
         'windows': int(windows),
         'dtypes': dtypes,
diff --git a/.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/ada-half-s3-auto-host-before-fix2/test_validation.py b/internal/perf/ada-half-s3-auto-20260907/test_validation.py
index 4391222c..0bd501aa 100644
--- a/.superpowers/sdd/handoff-codex-gemm-bi-triad-2026-09-05/ada-half-s3-auto-host-before-fix2/test_validation.py
+++ b/internal/perf/ada-half-s3-auto-20260907/test_validation.py
@@ -18,20 +18,23 @@ def load(name, path):
     spec.loader.exec_module(module)
     return module
 
 
 analyze = load('post_auto_analyze', HERE / 'analyze.py')
 run = load('post_auto_run', HERE / 'run.py')
 PERF = HERE.parent
 PRE = PERF / 'ada-half-s3-paired-20260907'
 FORCE = PERF / 'ada-half-s3-force-20260907'
 BINDING = json.loads((PRE / 'cuda132-binding-final.json').read_text())
+BINDING['frozen_pre_wrapper_sha'] = (
+    'ce0de6882fa45835ba81b90d02a6cc085ed32226e2462b72260ce9cacd0fd713'
+)
 QUALIFICATION = json.loads((FORCE / 'identity-cuda132.json').read_text())
 SEED = analyze.PRE.records(PRE / 'cuda132-smokefinal' / 'test.log')
 
 
 def transform_seed(row):
     row = copy.deepcopy(row)
     row['schema'] = analyze.SCHEMA
     row['stage'] = analyze.STAGE
     row['revision'] = 43
     if 'arm' in row:
@@ -178,22 +181,60 @@ class Protocol(unittest.TestCase):
         node['symbol'] = 'gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_bf16'
         with self.assertRaises(ValueError):
             validate(rows)
         for field in ('noop_rejected', 'pre_post_bits', 'pre_post_graphs',
                       'guards', 'immutable_inputs'):
             rows = fixture()
             next(row for row in rows if row['kind'] == 'configuration_complete')[field] = False
             with self.subTest(field=field), self.assertRaises(ValueError):
                 validate(rows)
 
+    def test_direct_chronology_physical_and_numerical_mutations_fail(self):
+        rows = fixture()
+        next(row for row in rows if row['kind'] == 'sample')['chronology'] = 1
+        with self.assertRaises(ValueError):
+            validate(rows)
+        for field in ('repeat_bits', 'poison_upload_verified', 'guards'):
+            rows = fixture()
+            next(row for row in rows if row['kind'] == 'physical')[field] = False
+            with self.subTest(field=field), self.assertRaises(ValueError):
+                validate(rows)
+        for numerical_error in (float('nan'), 0.0100001):
+            rows = fixture()
+            physical = next(row for row in rows if row['kind'] == 'physical')
+            physical['numerical_error'] = numerical_error
+            with self.subTest(numerical_error=numerical_error), self.assertRaises(ValueError):
+                validate(rows)
+
 
 class WrapperPolicy(unittest.TestCase):
+    def test_imported_frozen_modules_are_checked_before_execution(self):
+        with tempfile.TemporaryDirectory() as temporary:
+            marker = Path(temporary) / 'executed'
+            foreign = Path(temporary) / 'foreign.py'
+            foreign.write_text(
+                'from pathlib import Path\n'
+                f'Path({str(marker)!r}).write_text("executed")\n'
+            )
+            for module in (analyze, run):
+                with self.subTest(module=module.__name__), self.assertRaises(ValueError):
+                    module.load_pinned('foreign_dependency', foreign, '0' * 64)
+                self.assertFalse(marker.exists())
+
+    def test_foreign_bound_frozen_wrapper_digest_is_rejected(self):
+        binding = copy.deepcopy(BINDING)
+        binding['frozen_pre_wrapper_sha'] = '0' * 64
+        with self.assertRaises(ValueError):
+            validate(fixture(), binding=binding)
+        with self.assertRaises(ValueError):
+            run.validate_frozen_binding(binding)
+
     def test_stale_and_cross_stage_controls_are_rejected(self):
         run.reject_stale_controls(run.POST_CONTROLS)
         for key in [
             'MAMBA_FIXED_ADA_S3_PAIR',
             'MAMBA_FIXED_ADA_S3_WINDOWS',
             'MAMBA_FIXED_ADA_S3_DTYPES',
             'MAMBA_FIXED_AUTO_VENDOR_ROW',
             'MAMBA_FIXED_AUTO_VENDOR_CELL',
             'MAMBA_FIXED_AUTO_VENDOR_BIAS',
             'MAMBA_FIXED_HALF_TILE_CANDIDATE',
@@ -226,52 +267,54 @@ class WrapperPolicy(unittest.TestCase):
                 'toolkit': '13.2',
                 'windows': 1,
                 'dtypes': 'bf16,f16',
                 'stage': 'post_auto',
                 'binary_sha': BINDING['binary_sha'],
                 'source_sha': BINDING['inputs']['tests/gemm_bi_fixed_performance.rs'],
                 'test_log_sha': hashlib.sha256(test_text.encode()).hexdigest(),
                 'cache_files': {},
             }
             (directory / 'result.json').write_text(json.dumps(result))
+            binding_path = directory / 'binding.json'
+            binding_path.write_text(json.dumps(BINDING))
             pre = json.loads((directory / 'pre.json').read_text())
             post = json.loads((directory / 'post.json').read_text())
             lines = [
                 json.dumps(pre),
                 'COMMAND ' + json.dumps([
                     BINDING['binary'], '--ignored', '--exact',
                     'fixed_ada_half_s3_post_auto_fast_paired', '--nocapture',
                     '--test-threads=1']),
                 'COMMAND_EXIT test.log 0',
                 json.dumps(post),
                 'RUN_RESULT ' + json.dumps(result),
                 'WRAPPER_COMPLETE',
                 'WRAPPER_EXIT=0',
                 'OUTER_SSH_EXIT=0',
             ]
             transcript = directory / 'ssh.log'
             transcript.write_text('\n'.join(lines) + '\n')
-            report = analyze.verify_run(directory, PRE / 'cuda132-binding-final.json',
+            report = analyze.verify_run(directory, binding_path,
                                         FORCE / 'identity-cuda132.json', transcript)
             self.assertEqual((report['configurations'], report['samples'], report['pairs']),
                              (8, 96, 24))
             for index, replacement in [
                 (1, lines[1].replace(BINDING['binary'], BINDING['binary'] + '-foreign')),
                 (2, 'COMMAND_EXIT test.log 101'),
                 (5, 'WRAPPER_COMPLETEgarbage'),
                 (6, 'WRAPPER_EXIT=1'),
                 (7, 'OUTER_SSH_EXIT=255'),
             ]:
                 changed = lines.copy()
                 changed[index] = replacement
                 transcript.write_text('\n'.join(changed) + '\n')
                 with self.subTest(index=index), self.assertRaises(ValueError):
-                    analyze.verify_run(directory, PRE / 'cuda132-binding-final.json',
+                    analyze.verify_run(directory, binding_path,
                                        FORCE / 'identity-cuda132.json', transcript)
             transcript.write_text('\n'.join(lines + ['WRAPPER_COMPLETE']) + '\n')
             with self.assertRaises(ValueError):
-                analyze.verify_run(directory, PRE / 'cuda132-binding-final.json',
+                analyze.verify_run(directory, binding_path,
                                    FORCE / 'identity-cuda132.json', transcript)
 
 
 if __name__ == '__main__':
     unittest.main(verbosity=2)
