# Task6C frozen host utility source package

This is the first independent review package for three new Python utilities; frozen Rust/source-fix review is separate and complete. Do not reinterpret raw pre42 evidence. Post mapping is ephemeral adapter data after post43 BASE checks.

Hashes: analyze9c73297e354004b92ee7df7d16f8bf3d306900c1f608fdb7a324da09a67eec65; runa4e322bc3a0d3dba96b2d413669a218e4a7c5fd050a2cc272c538b7bca287fc6; testsdc3fb73c79ecdaee1fc60baafb44a8b4bbd2446c2c19174f58eda18407ce753d.
Root read all three sources and independently ran8hostgroupsPASS. Final actual rawlogs/closure are attached by separate evidence package. Frozen reused Task6B analyzer/run.py are unchanged; source-qualified under its I1 fix, exact eight-record same-attempt transcript contract. Inspect reused code only for concrete adapter-boundary risks.

## internal/perf/ada-half-s3-auto-20260907/analyze.py

```python
#!/usr/bin/env python3
"""Independent post-AUTO43 adapter over the frozen Task6B paired validator."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PRE_PATH = HERE.parent / 'ada-half-s3-paired-20260907' / 'analyze.py'
PRE_SPEC = importlib.util.spec_from_file_location('ada_s3_pair_frozen_analyze', PRE_PATH)
PRE = importlib.util.module_from_spec(PRE_SPEC)
PRE_SPEC.loader.exec_module(PRE)

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


def records(path):
    result = []
    marker = '{"schema":"' + SCHEMA + '"'
    for line in Path(path).read_text().splitlines():
        offset = line.find(marker)
        if offset >= 0:
            result.append(json.loads(line[offset:]))
    return result


def _pre_rows(rows):
    mapped = copy.deepcopy(rows)
    arm_map = {'Swizzle': 'AUTO', 'AUTO': 'S3', 'Fast': 'Fast'}
    direction_map = dict(zip(DIRECTIONS, PRE.DIRECTIONS))
    for row in mapped:
        row['schema'] = PRE.SCHEMA
        row['revision'] = 42
        row.pop('stage', None)
        if 'arm' in row:
            row['arm'] = arm_map[row['arm']]
        if 'direction' in row:
            row['direction'] = direction_map[row['direction']]
    return mapped


def verify(rows, binding, qualification, windows, dtypes):
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
        if row.get('kind') == 'summary':
            comparison = row.get('comparison')
            PRE.check(type(comparison) is int and 0 <= comparison < 3,
                      'wrong post-AUTO comparison')
            PRE.check(row.get('direction') == DIRECTIONS[comparison],
                      'wrong post-AUTO ratio direction')
    report = PRE.verify(_pre_rows(rows), binding, qualification, windows, dtypes)
    direction_map = dict(zip(PRE.DIRECTIONS, DIRECTIONS))
    for summary in report['summaries']:
        summary['direction'] = direction_map.get(summary['direction'], summary['direction'])
    for cell in report['cells']:
        for summary in cell['constituents']:
            summary['direction'] = direction_map.get(summary['direction'], summary['direction'])
    report['schema'] = SCHEMA
    report['stage'] = STAGE
    report['compiled_identity_revision'] = qualification['tuning_table_revision']
    report['routing_revision'] = 43
    return report


def verify_run(directory, binding_path, qualification_path, ssh_log):
    run_path = HERE.parent / 'ada-half-s3-paired-20260907' / 'run.py'
    run_spec = importlib.util.spec_from_file_location('ada_s3_pair_frozen_run', run_path)
    run = importlib.util.module_from_spec(run_spec)
    run_spec.loader.exec_module(run)
    directory = Path(directory)
    binding = json.loads(Path(binding_path).read_text())
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
        == result['test_log_sha'],
        'raw log hash mismatch',
    )
    lines = Path(ssh_log).read_text().splitlines()
    PRE.check(len(lines) == 8,
              'SSH transcript must contain exactly one complete eight-record attempt')
    pre = json.loads(lines[0])
    post = json.loads(lines[3])
    for phase, observed in [('PRE', pre), ('POST', post)]:
        saved = json.loads((directory / (phase.lower() + '.json')).read_text())
        PRE.check(observed == saved, f'{phase}: SSH/saved telemetry mismatch')
        PRE.equal_fields(observed, {'phase': phase})
        run.snapshot(phase, observed['gpu'], observed['apps'],
                     observed['gpu_exit'], observed['apps_exit'])
    PRE.check(lines[1].startswith('COMMAND '), 'missing/misordered COMMAND')
    command = json.loads(lines[1][len('COMMAND '):])
    PRE.check(command == [binding['binary'], '--ignored', '--exact',
                          'fixed_ada_half_s3_post_auto_fast_paired', '--nocapture',
                          '--test-threads=1'], 'wrong command/test/binary for this attempt')
    command_exit = re.fullmatch(r'COMMAND_EXIT test\.log (-?\d+)', lines[2])
    PRE.check(command_exit is not None, 'missing/malformed/misordered COMMAND_EXIT')
    PRE.check(lines[4].startswith('RUN_RESULT '), 'missing/misordered RUN_RESULT')
    PRE.check(json.loads(lines[4][len('RUN_RESULT '):]) == result,
              'SSH RUN_RESULT differs from saved result.json')
    PRE.check(lines[5] == 'WRAPPER_COMPLETE',
              'missing/malformed/misordered WRAPPER_COMPLETE')
    wrapper_exit = re.fullmatch(r'WRAPPER_EXIT=(-?\d+)', lines[6])
    ssh_exit = re.fullmatch(r'OUTER_SSH_EXIT=(-?\d+)', lines[7])
    PRE.check(wrapper_exit is not None and ssh_exit is not None,
              'missing/malformed/misordered wrapper/SSH exit')
    PRE.check(int(command_exit[1]) == result['test_exit'],
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
        'frozen_pre_analyzer_sha': hashlib.sha256(PRE_PATH.read_bytes()).hexdigest(),
        'ssh_log_sha': hashlib.sha256(Path(ssh_log).read_bytes()).hexdigest(),
        'binding_sha': hashlib.sha256(Path(binding_path).read_bytes()).hexdigest(),
        'qualification_sha': hashlib.sha256(Path(qualification_path).read_bytes()).hexdigest(),
    }
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('binding')
    parser.add_argument('qualification')
    parser.add_argument('ssh_log')
    args = parser.parse_args()
    try:
        print(json.dumps(verify_run(args.directory, args.binding, args.qualification,
                                    args.ssh_log), indent=2))
    except Exception as error:
        print('INVALID: ' + str(error), file=sys.stderr)
        sys.exit(1)
```

## internal/perf/ada-half-s3-auto-20260907/run.py

```python
#!/usr/bin/env python3
"""Task6C isolated all-toolkit build and CUDA13.2 post-AUTO run wrapper."""
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

HERE = Path(__file__).resolve().parent
FROZEN_PATH = HERE.parent / 'ada-half-s3-paired-20260907' / 'run.py'
FROZEN_SPEC = importlib.util.spec_from_file_location('ada_s3_pair_frozen_run', FROZEN_PATH)
FROZEN = importlib.util.module_from_spec(FROZEN_SPEC)
FROZEN_SPEC.loader.exec_module(FROZEN)

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


def forbidden_control(key):
    return ((key.startswith('MAMBA_FIXED_ADA_') and key not in POST_CONTROLS)
            or key.startswith('MAMBA_FIXED_VENDOR_')
            or key in {'MAMBA_FIXED_AUTO_VENDOR_ROW',
                       'MAMBA_FIXED_AUTO_VENDOR_CELL',
                       'MAMBA_FIXED_AUTO_VENDOR_BIAS',
                       'MAMBA_FIXED_HALF_TILE_CANDIDATE',
                       'NVIDIA_TF32_OVERRIDE'})


def reject_stale_controls(environment):
    stale = sorted(key for key in environment if forbidden_control(key))
    FROZEN.require(not stale, 'stale controls forbidden: ' + ','.join(stale))


def main():
    operation, toolkit, attempt, *options = sys.argv[1:]
    FROZEN.require(toolkit in TOOLKITS, 'unsupported toolkit')
    FROZEN.require(re.fullmatch(r'[a-z0-9-]+', attempt), 'invalid attempt name')
    tag, feature = TOOLKITS[toolkit]
    cuda = Path('/usr/local/cuda-' + toolkit)
    target = Path('/root/target-ada-half-s3-auto-fix1-cuda' + tag + '-20260907')
    cache = Path('/root/mamba-kcache-ada-half-s3-auto-fix1-cuda' + tag + '-20260907')
    env = dict(os.environ, CUDA_HOME=str(cuda), CUDA_PATH=str(cuda),
               PATH=str(cuda / 'bin') + ':/root/.cargo/bin:' + os.environ['PATH'],
               LD_LIBRARY_PATH=str(cuda / 'lib64'), CARGO_TARGET_DIR=str(target),
               MAMBA_RS_KERNEL_CACHE=str(cache))
    reject_stale_controls(env)
    directory = EVIDENCE / f'cuda{tag}-{attempt}'
    directory.mkdir()
    build_binding = EVIDENCE / f'cuda{tag}-binding.json'
    if operation == 'build':
        FROZEN.require(not options, 'build takes no extra arguments')
        cache.mkdir(mode=0o700, exist_ok=True)
        FROZEN.require(stat.S_IMODE(cache.stat().st_mode) == 0o700,
                       'cache is not private0700')
        for name, args in [
            ('nvcc', [str(cuda / 'bin/nvcc'), '--version']),
            ('rustc', ['rustc', '-Vv']),
            ('cargo', ['cargo', '-V']),
        ]:
            FROZEN.require(FROZEN.command(args, directory / (name + '.log'), env,
                                          cwd=ROOT) == 0,
                           name + ' failed')
        test = ['cargo', 'test', '--release', '--features',
                'cuda,cudarc/cuda-' + feature, '--test', 'gemm_bi_fixed_performance']
        FROZEN.require(FROZEN.command(test + ['ada_s3_pair::', '--', '--nocapture'],
                                     directory / 'focused.log', env, cwd=ROOT) == 0,
                       'focused tests failed')
        FROZEN.require(FROZEN.command(test + ['--', '--test-threads=1'],
                                     directory / 'nonignored.log', env, cwd=ROOT) == 0,
                       'full nonignored tests failed')
        binaries = [p for p in (target / 'release/deps').glob('gemm_bi_fixed_performance-*')
                    if p.is_file() and os.access(p, os.X_OK)]
        FROZEN.require(len(binaries) == 1, 'ambiguous binary inventory')
        inputs = FROZEN.source_inputs()
        binding = {
            'toolkit': toolkit,
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
            'frozen_pre_wrapper_sha': FROZEN.sha(FROZEN_PATH),
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
               S3_TOOLKIT=toolkit,
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
        'stage': 'post_auto',
        'binary_sha': FROZEN.sha(binary),
        'source_sha': binding['inputs']['tests/gemm_bi_fixed_performance.rs'],
        'test_log_sha': FROZEN.sha(directory / 'test.log'),
        'cache_files': {str(path): FROZEN.sha(path) for path in cache.glob('*.bin')},
    }
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('RUN_RESULT ' + json.dumps(result), flush=True)
    FROZEN.closure(test_exit, post_exit)
    print('WRAPPER_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('WRAPPER_FAILURE ' + repr(error), flush=True)
        sys.exit(1)
```

## internal/perf/ada-half-s3-auto-20260907/test_validation.py

```python
#!/usr/bin/env python3
"""Adversarial host tests for the post-AUTO43 adapter and wrapper policy."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = load('post_auto_analyze', HERE / 'analyze.py')
run = load('post_auto_run', HERE / 'run.py')
PERF = HERE.parent
PRE = PERF / 'ada-half-s3-paired-20260907'
FORCE = PERF / 'ada-half-s3-force-20260907'
BINDING = json.loads((PRE / 'cuda132-binding-final.json').read_text())
QUALIFICATION = json.loads((FORCE / 'identity-cuda132.json').read_text())
SEED = analyze.PRE.records(PRE / 'cuda132-smokefinal' / 'test.log')


def transform_seed(row):
    row = copy.deepcopy(row)
    row['schema'] = analyze.SCHEMA
    row['stage'] = analyze.STAGE
    row['revision'] = 43
    if 'arm' in row:
        row['arm'] = {'AUTO': 'Swizzle', 'S3': 'AUTO', 'Fast': 'Fast'}[row['arm']]
    if 'direction' in row:
        row['direction'] = dict(zip(analyze.PRE.DIRECTIONS,
                                    analyze.DIRECTIONS))[row['direction']]
    return row


def quantile(values, fraction):
    values = sorted(values)
    return values[math.floor((len(values) - 1) * fraction + 0.5)]


def fixture(profile='win', windows=101):
    identity = transform_seed(SEED[0])
    identity['windows'] = windows
    identity['dtypes'] = 'bf16,f16'
    rows = [identity] + [transform_seed(row) for row in SEED if row['kind'] == 'physical']
    base = {key: identity[key] for key in
            ('schema', 'stage', 'toolkit', 'shape', 'bias', 'alpha', 'beta', 'revision')}
    pair_arms = [('Swizzle', 'AUTO'), ('Fast', 'Swizzle'), ('Fast', 'AUTO')]
    for dtype in ('bf16', 'f16'):
        for path in ('eager', 'graph'):
            for start in (0, 1):
                key = dict(base, dtype=dtype, path=path, start_parity=start)
                samples = []
                pairs = []
                ratios = [[], [], []]
                for window in range(windows):
                    reverse = (window + start) % 2 == 1
                    comparisons = (2, 1, 0) if reverse else (0, 1, 2)
                    if profile == 'loss':
                        auto = 110.0
                    elif profile == 'mixed' and window >= 95:
                        auto = 120.0
                    else:
                        auto = 80.0
                    values = {'Swizzle': 100.0, 'AUTO': auto, 'Fast': 75.0}
                    for traversal, comparison in enumerate(comparisons):
                        a, b = pair_arms[comparison]
                        arms = (b, a, a, b) if reverse else (a, b, b, a)
                        offset = len(samples)
                        for position, arm in enumerate(arms):
                            samples.append(dict(
                                key, kind='sample', chronology=len(samples), window=window,
                                comparison=comparison, traversal=traversal,
                                order='BAAB' if reverse else 'ABBA', position=position,
                                arm=arm, logical_ops=20, us=values[arm]))
                        ratio = values[b] / values[a]
                        ratios[comparison].append(ratio)
                        pairs.append(dict(
                            key, kind='pair', window=window, comparison=comparison,
                            traversal=traversal,
                            observations=list(range(offset, offset + 4)), ratio=ratio))
                rows += samples + pairs
                for comparison, direction in enumerate(analyze.DIRECTIONS):
                    rows.append(dict(
                        key, kind='summary', comparison=comparison, direction=direction,
                        windows=windows, p50=quantile(ratios[comparison], 0.5),
                        p95=quantile(ratios[comparison], 0.95)))
                rows.append(dict(
                    key, kind='configuration_complete', samples=12 * windows,
                    pairs=3 * windows, summaries=3, pre_post_bits=True,
                    pre_post_graphs=True, guards=True, immutable_inputs=True,
                    noop_rejected=True))
    rows.append(dict(base, kind='complete', configurations=8,
                     samples=96 * windows, pairs=24 * windows, summaries=24,
                     rejected=0, passed=True))
    return rows


def validate(rows, windows=101, dtypes=None, binding=None):
    return analyze.verify(rows, binding or BINDING, QUALIFICATION, windows,
                          ['bf16', 'f16'] if dtypes is None else dtypes)


class Protocol(unittest.TestCase):
    def test_exact_post_smoke_adapter_closes_all_eight_configurations(self):
        rows = [transform_seed(row) for row in SEED]
        report = validate(rows, windows=1)
        self.assertEqual((report['configurations'], report['samples'], report['pairs']),
                         (8, 96, 24))
        self.assertEqual((report['routing_revision'], report['compiled_identity_revision']),
                         (43, 42))

    def test_post_rejects_window21_and_every_dtype_subset(self):
        rows = fixture()
        with self.assertRaises(ValueError):
            validate(rows, windows=21)
        for dtypes in (['bf16'], ['f16'], ['f16', 'bf16'], []):
            with self.subTest(dtypes=dtypes), self.assertRaises(ValueError):
                validate(rows, dtypes=dtypes)

    def test_genuine_own_win_loss_and_mixed_p95_are_preserved(self):
        for profile, p50, p95, admission in [
            ('win', 0.8, 0.8, True),
            ('loss', 1.1, 1.1, False),
            ('mixed', 0.8, 1.2, False),
        ]:
            with self.subTest(profile=profile):
                report = validate(fixture(profile))
                for cell in report['cells']:
                    self.assertEqual((cell['worst_own_p50'], cell['worst_own_p95']),
                                     (p50, p95))
                    self.assertEqual(cell['admission'], admission)

    def test_bad_revision_stage_auto_arm_direction_and_identity_fail(self):
        cases = [
            ('revision', 42),
            ('stage', 'pre_promotion'),
            ('toolkit', '13.0'),
            ('source_sha', '0' * 64),
            ('binary_sha', '0' * 64),
            ('fixed_artifact_digest', '0' * 64),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                rows = fixture()
                rows[0][key] = value
                with self.assertRaises(ValueError):
                    validate(rows)
        rows = fixture()
        next(row for row in rows if row['kind'] == 'sample' and row['arm'] == 'AUTO')['arm'] = 'S3'
        with self.assertRaises(ValueError):
            validate(rows)
        rows = fixture()
        next(row for row in rows if row['kind'] == 'summary')['direction'] = 'Swizzle/AUTO'
        with self.assertRaises(ValueError):
            validate(rows)

    def test_malformed_physical_args_symbol_poison_and_closure_flags_fail(self):
        for field in ('bundle', 'pointers', 'abi', 'sixth_rejected'):
            rows = fixture()
            node = next(row for row in rows
                        if row['kind'] == 'physical' and row['arm'] == 'AUTO')['one'][0]
            node[field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(rows)
        rows = fixture()
        node = next(row for row in rows
                    if row['kind'] == 'physical' and row['arm'] == 'AUTO')['one'][0]
        node['symbol'] = 'gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_bf16'
        with self.assertRaises(ValueError):
            validate(rows)
        for field in ('noop_rejected', 'pre_post_bits', 'pre_post_graphs',
                      'guards', 'immutable_inputs'):
            rows = fixture()
            next(row for row in rows if row['kind'] == 'configuration_complete')[field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(rows)


class WrapperPolicy(unittest.TestCase):
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
            'MAMBA_FIXED_VENDOR_OLD',
            'NVIDIA_TF32_OVERRIDE',
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.reject_stale_controls([key])

    def test_all_exit_layers_fail_closed(self):
        run.FROZEN.closure(0, 0, 0, 0)
        for exits in ((101, 0, 0, 0), (0, 1, 0, 0),
                      (0, 0, 1, 0), (0, 0, 0, 255)):
            with self.subTest(exits=exits), self.assertRaises(ValueError):
                run.FROZEN.closure(*exits)

    def test_exact_same_attempt_transcript_and_mutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = PRE / 'cuda132-smokefinal'
            for name in ('pre.json', 'post.json'):
                (directory / name).write_bytes((source / name).read_bytes())
            test_text = '\n'.join(json.dumps(row, separators=(',', ':'))
                                  for row in fixture('win', 1))
            test_text += '\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n'
            (directory / 'test.log').write_text(test_text)
            result = {
                'test_exit': 0,
                'post_exit': 0,
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
            report = analyze.verify_run(directory, PRE / 'cuda132-binding-final.json',
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
                    analyze.verify_run(directory, PRE / 'cuda132-binding-final.json',
                                       FORCE / 'identity-cuda132.json', transcript)
            transcript.write_text('\n'.join(lines + ['WRAPPER_COMPLETE']) + '\n')
            with self.assertRaises(ValueError):
                analyze.verify_run(directory, PRE / 'cuda132-binding-final.json',
                                   FORCE / 'identity-cuda132.json', transcript)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```
