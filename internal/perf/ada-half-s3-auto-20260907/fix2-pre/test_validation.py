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
