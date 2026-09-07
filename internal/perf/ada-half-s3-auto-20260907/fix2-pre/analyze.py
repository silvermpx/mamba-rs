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
