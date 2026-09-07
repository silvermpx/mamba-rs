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
