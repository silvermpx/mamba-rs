#!/usr/bin/env python3
"""Task6B isolated build/run wrapper. Logs and bindings are append-only attempts."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

UUID = 'GPU-d1edd7be-e88d-aed6-047d-622163306f0e'
ROOT = Path('/root/mamba-ada-half-s3-paired-20260907')
EVIDENCE = Path('/root/evidence-ada-half-s3-paired-20260907')
TOOLKITS = {'12.8': ('128', '12080'), '13.0': ('130', '13000'), '13.2': ('132', '13020')}

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def snapshot(phase, gpu, apps, gpu_exit=0, apps_exit=0):
    require(gpu_exit == 0 and apps_exit == 0, f'{phase}: telemetry query failed')
    fields = [v.strip() for v in gpu.strip().split(',')]
    require(len(fields) == 5 and fields[:3] == [UUID, 'NVIDIA RTX 6000 Ada Generation', '8.9'], f'{phase}: wrong device')
    require(not apps.strip(), f'{phase}: active compute apps')
    require(phase in ('PRE', 'POST', 'RELEASE'), 'bad telemetry phase')
    if phase in ('PRE', 'RELEASE'):
        require(fields[3:] == ['0 %', '0 %'], f'{phase}: not quiet')

def closure(test_exit, post_exit, wrapper_exit=0, ssh_exit=0):
    require((test_exit, post_exit, wrapper_exit, ssh_exit) == (0, 0, 0, 0), 'failed exit closure')

def source_inputs():
    paths = [ROOT / 'Cargo.toml', ROOT / 'Cargo.lock']
    for directory in ('src', 'kernels', 'tests'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and not p.name.startswith('._')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}

def validate_binding(binding, toolkit, inputs, binary_hash):
    require(binding['toolkit'] == toolkit, 'wrong bound toolkit')
    require(binding['inputs'] == inputs, 'source/build input mismatch')
    require(binding['binary_sha'] == binary_hash, 'binary mismatch')

def command(args, log, env, cwd=ROOT):
    print('COMMAND ' + json.dumps([str(a) for a in args]), flush=True)
    with log.open('x') as output:
        result = subprocess.run(args, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(f'COMMAND_EXIT {log.name} {result.returncode}', flush=True)
    return result.returncode

def telemetry(phase, directory, env):
    data = {'phase': phase, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    for kind, query in [('gpu', '--query-gpu=uuid,name,compute_cap,utilization.gpu,utilization.memory'),
                        ('apps', '--query-compute-apps=pid,gpu_uuid,process_name')]:
        r = subprocess.run(['/usr/bin/nvidia-smi', query, '--format=csv,noheader'], env=env, text=True, capture_output=True)
        data[kind] = r.stdout
        data[kind+'_stderr'] = r.stderr
        data[kind+'_exit'] = r.returncode
    (directory/(phase.lower()+'.json')).write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data), flush=True)
    snapshot(phase, data['gpu'], data['apps'], data['gpu_exit'], data['apps_exit'])

def main():
    operation, toolkit, attempt, *options = sys.argv[1:]
    require(toolkit in TOOLKITS, 'unsupported toolkit')
    require(re.fullmatch(r'[a-z0-9-]+', attempt), 'invalid attempt name')
    tag, feature = TOOLKITS[toolkit]
    cuda = Path('/usr/local/cuda-'+toolkit)
    target = Path('/root/target-ada-half-s3-paired-cuda'+tag+'-20260907')
    cache = Path('/root/mamba-kcache-ada-half-s3-paired-cuda'+tag+'-20260907')
    env = dict(os.environ, CUDA_HOME=str(cuda), CUDA_PATH=str(cuda),
               PATH=str(cuda/'bin')+':/root/.cargo/bin:'+os.environ['PATH'],
               LD_LIBRARY_PATH=str(cuda/'lib64'), CARGO_TARGET_DIR=str(target), MAMBA_RS_KERNEL_CACHE=str(cache))
    directory = EVIDENCE/f'cuda{tag}-{attempt}'
    directory.mkdir()
    build_binding = EVIDENCE/f'cuda{tag}-binding-final.json'
    if operation == 'build':
        require(not options, 'build takes no extra arguments')
        cache.mkdir(mode=0o700, exist_ok=True)
        require(stat.S_IMODE(cache.stat().st_mode)==0o700,'cache is not private0700')
        for name, args in [('nvcc', [str(cuda/'bin/nvcc'),'--version']), ('rustc',['rustc','-Vv']), ('cargo',['cargo','-V'])]:
            require(command(args,directory/(name+'.log'),env)==0, name+' failed')
        test = ['cargo','test','--release','--features','cuda,cudarc/cuda-'+feature,'--test','gemm_bi_fixed_performance']
        require(command(test+['ada_s3_pair::','--','--nocapture'],directory/'focused.log',env)==0,'focused tests failed')
        require(command(test+['--','--test-threads=1'],directory/'nonignored.log',env)==0,'full nonignored tests failed')
        binaries = [p for p in (target/'release/deps').glob('gemm_bi_fixed_performance-*') if p.is_file() and os.access(p,os.X_OK)]
        require(len(binaries)==1,'ambiguous binary inventory')
        binding = {'toolkit':toolkit,'feature':'cuda,cudarc/cuda-'+feature,'binary':str(binaries[0]),'binary_sha':sha(binaries[0]),
                   'inputs':source_inputs(),'cache':str(cache),'cache_mode':oct(stat.S_IMODE(cache.stat().st_mode)),
                   'tools':{str(p):sha(p) for p in [cuda/'bin/nvcc',cuda/'bin/ptxas']},
                   'libraries':{str(p):sha(p) for pattern in ('libnvrtc.so.*','libcublas.so.*') for p in (cuda/'targets/x86_64-linux/lib').glob(pattern)}}
        with build_binding.open('x') as out: json.dump(binding,out,indent=2); out.write('\n')
        print('BUILD_COMPLETE '+json.dumps({k:v for k,v in binding.items() if k!='inputs'}),flush=True)
        return
    require(operation == 'run' and len(options)==2,'run requires windows dtypes')
    windows,dtypes = options
    require(windows in ('1','21','101'),'invalid windows')
    require(dtypes in ('bf16,f16','bf16','f16'),'invalid dtype list')
    require(windows=='101' or dtypes=='bf16,f16','screen/smoke muted dtype')
    binding = json.loads(build_binding.read_text())
    binary = Path(binding['binary'])
    validate_binding(binding,toolkit,source_inputs(),sha(binary))
    require(stat.S_IMODE(cache.stat().st_mode)==0o700,'cache is not private0700')
    for p,h in (binding['tools']|binding['libraries']).items(): require(sha(p)==h,'tool/library changed '+p)
    env.update(MAMBA_FIXED_ADA_S3_PAIR='1',MAMBA_FIXED_ADA_S3_WINDOWS=windows,MAMBA_FIXED_ADA_S3_DTYPES=dtypes,
               S3_TOOLKIT=toolkit,S3_SOURCE_SHA=binding['inputs']['tests/gemm_bi_fixed_performance.rs'],S3_BINARY_SHA=binding['binary_sha'])
    telemetry('PRE',directory,env)
    test_exit = command([str(binary),'--ignored','--exact','fixed_ada_half_s3_auto_fast_paired','--nocapture','--test-threads=1'],directory/'test.log',env)
    post_exit = 0
    try:
        telemetry('POST',directory,env)
        validate_binding(binding,toolkit,source_inputs(),sha(binary))
    except Exception as error:
        print('POSTCHECK_FAILURE '+str(error),flush=True)
        post_exit = 1
    result = {'test_exit':test_exit,'post_exit':post_exit,'toolkit':toolkit,'windows':int(windows),'dtypes':dtypes,
              'binary_sha':sha(binary),'source_sha':binding['inputs']['tests/gemm_bi_fixed_performance.rs'],
              'test_log_sha':sha(directory/'test.log'),'cache_files':{str(p):sha(p) for p in cache.glob('*.bin')}}
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RUN_RESULT '+json.dumps(result),flush=True)
    closure(test_exit,post_exit)
    print('WRAPPER_COMPLETE',flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('WRAPPER_FAILURE '+repr(error),flush=True)
        sys.exit(1)
