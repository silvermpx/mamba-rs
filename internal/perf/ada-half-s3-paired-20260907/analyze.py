#!/usr/bin/env python3
"""Independent exact-key, chronological, physical and wrapper closure validator."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

SCHEMA = 'MambaBiFixedAdaS3PairedV1'
ARMS = ['AUTO','S3','Fast']
DIRECTIONS = ['S3/AUTO','AUTO/Fast','S3/Fast']
BASE = {'shape':[4621,768,2304],'bias':False,'alpha':1,'beta':0,'revision':42}
MODES = {'compute':'CUBLAS_COMPUTE_32F','algorithm':'CUBLAS_GEMM_DEFAULT_TENSOR_OP',
         'math':'CUBLAS_DEFAULT_MATH','pointer_mode':'CUBLAS_POINTER_MODE_HOST',
         'atomics':'CUBLAS_ATOMICS_NOT_ALLOWED','bias_broadcast':False}
IDENTITY_KEYS = ['fixed_source_digest','fixed_invocation_digest','fixed_artifact_digest',
                 'header_manifest_digest','nvrtc_library_domain','nvrtc_library_known']

def check(condition, reason):
    if not condition: raise ValueError(reason)

def equal_fields(record, expected):
    for key,value in expected.items():
        check(key in record and record[key] == value, f'{record.get("kind")}: wrong/missing {key}: {record.get(key)!r} != {value!r}')

def positive(value):
    return isinstance(value,(int,float)) and math.isfinite(value) and value > 0

def near(a,b):
    return positive(a) and positive(b) and math.isclose(a,b,rel_tol=2e-14,abs_tol=1e-14)

def quantile(values, fraction):
    return sorted(values)[math.floor((len(values)-1)*fraction+0.5)]

def records(path):
    result=[]
    for line in Path(path).read_text().splitlines():
        marker=line.find('{"schema":"'+SCHEMA+'"')
        if marker >= 0: result.append(json.loads(line[marker:]))
    return result

def verify(rows, binding, qualification, windows, dtypes):
    check(windows in (1,21,101),'bad window count')
    check(dtypes in (['bf16','f16'],['bf16'],['f16']),'bad requested dtypes')
    check(windows==101 or dtypes==['bf16','f16'],'muted screen/smoke')
    toolkit=binding['toolkit']
    check(toolkit in ('12.8','13.0','13.2'),'bad toolkit')
    check(qualification['nvrtc']==[int(v) for v in toolkit.split('.')],'wrong qualification toolkit')
    check(qualification['tuning_table_revision']==42,'wrong qualification revision')
    check(rows and rows[0]['kind']=='identity' and rows[-1]['kind']=='complete','identity/completion position')
    kinds={'identity','physical','sample','pair','summary','configuration_complete','complete'}
    for r in rows:
        check(r.get('kind') in kinds,'foreign record kind')
        equal_fields(r,dict(BASE,schema=SCHEMA,toolkit=toolkit))
        if 'dtype' in r: check(r['dtype'] in dtypes,'foreign dtype')
    identity=rows[0]
    equal_fields(identity,dict(MODES,uuid='GPU-d1edd7be-e88d-aed6-047d-622163306f0e',cc='8.9',sm_count=142,
        source_sha=binding['inputs']['tests/gemm_bi_fixed_performance.rs'],binary_sha=binding['binary_sha'],
        dtypes=','.join(dtypes),windows=windows,logical_ops=20,warmup_eager=128,percentile='round((len-1)*fraction)'))
    equal_fields(identity,{k:qualification[k] for k in IDENTITY_KEYS})
    check(sum(r['kind']=='identity' for r in rows)==1,'duplicate identity')
    physical=[r for r in rows if r['kind']=='physical']
    check([(r['dtype'],r['arm']) for r in physical]==[(d,a) for d in dtypes for a in ARMS],'physical exact key/order closure')
    for d in dtypes:
        byarm={r['arm']:r for r in physical if r['dtype']==d}
        pointers=[]
        for arm in ARMS:
            r=byarm[arm]
            equal_fields(r,dict(guard_bytes=256,reference='PEDANTIC_F32',eager_repeats=2,graph_repeats=2,
                poison_upload_verified=True,repeat_bits=True,guards=True,tolerance=0.01 if d=='bf16' else 0.0025))
            check(isinstance(r['numerical_error'],(int,float)) and math.isfinite(r['numerical_error']) and 0<=r['numerical_error']<=r['tolerance'],'invalid numerical gate')
            p=r['pointers']; allocation=r['allocation']
            check(len(p)==4 and all(type(v)==int for v in p) and all(v>0 and v%256==0 for v in p[:3]) and p[3]==0,'invalid physical pointers')
            check(allocation==[p[0]-256,4621*2304*2+512],'guarded interior allocation')
            pointers.append(p)
            one=r['one']; twenty=r['twenty']
            check(one and len(twenty)==len(one)*20,'20-op node inventory count')
            if arm!='Fast':
                check(len(one)==1,'one-op custom graph must have one kernel')
                for node in one+twenty:
                    equal_fields(node,dict(symbol=f'gemm_bi_nn_fixed_sm89_tc128_{"swizzle" if arm=="AUTO" else "s3"}_v1_{d}',
                        grid=[666,1,1],block=[256,1,1],shared_bytes=69632 if arm=='AUTO' else 98304,
                        pointers=p,bundle=[1065353216,0,4621,2304,768,768,2304,2304],
                        abi=[[0,8],[8,8],[16,8],[24,8],[32,32]],sixth_rejected=True))
            else:
                for node in one+twenty:
                    check(isinstance(node['symbol'],str) and node['symbol'] and 'gemm_bi_' not in node['symbol'] and 'bias' not in node['symbol'].lower(),'invalid native vendor graph symbol')
                    check(all(type(v)==int and v>0 for v in node['grid']+node['block']),'vendor grid/block')
                check(Counter(json.dumps(n,sort_keys=True) for n in twenty)==Counter({k:v*20 for k,v in Counter(json.dumps(n,sort_keys=True) for n in one).items()}),'vendor20 must repeat actual one-op node inventory')
        check(len({p[0] for p in pointers})==3 and len({tuple(p[1:]) for p in pointers})==1,'shared A/B and distinct guarded C arms')
    expected_configs=[(d,p,s) for d in dtypes for p in ('eager','graph') for s in (0,1)]
    complete=[r for r in rows if r['kind']=='configuration_complete']
    check([(r['dtype'],r['path'],r['start_parity']) for r in complete]==expected_configs,'configuration completion key/order closure')
    timing=[r for r in rows if r['kind'] in ('sample','pair','summary','configuration_complete')]
    for r in timing:
        check((r['dtype'],r['path'],r['start_parity']) in expected_configs,'foreign timing cohort')
    all_summaries=[]
    for dtype,path,start in expected_configs:
        cohort=[r for r in timing if (r['dtype'],r['path'],r['start_parity'])==(dtype,path,start)]
        expected_kinds=['sample']*(12*windows)+['pair']*(3*windows)+['summary']*3+['configuration_complete']
        check([r['kind'] for r in cohort]==expected_kinds,'cohort exact record-kind closure/order')
        samples=cohort[:12*windows]; pairs=cohort[12*windows:15*windows]; summaries=cohort[15*windows:15*windows+3]
        ratios=[[],[],[]]
        for w in range(windows):
            parity=(w+start)%2
            comparisons=[0,1,2] if parity==0 else [2,1,0]
            for traversal,comparison in enumerate(comparisons):
                a,b=[('AUTO','S3'),('Fast','AUTO'),('Fast','S3')][comparison]
                arms=[a,b,b,a] if parity==0 else [b,a,a,b]
                offset=w*12+traversal*4
                bracket=samples[offset:offset+4]
                for position,(r,arm) in enumerate(zip(bracket,arms)):
                    equal_fields(r,dict(chronology=offset+position,window=w,comparison=comparison,traversal=traversal,
                        order='ABBA' if parity==0 else 'BAAB',position=position,arm=arm,logical_ops=20))
                    check(positive(r['us']),'nonfinite/nonpositive raw timing')
                ratio=sum(r['us'] for r in bracket if r['arm']==b)/sum(r['us'] for r in bracket if r['arm']==a)
                pair=pairs[w*3+traversal]
                equal_fields(pair,dict(window=w,comparison=comparison,traversal=traversal,observations=list(range(offset,offset+4))))
                check(near(pair['ratio'],ratio),'forged pair ratio')
                ratios[comparison].append(ratio)
        for comparison,r in enumerate(summaries):
            equal_fields(r,dict(comparison=comparison,direction=DIRECTIONS[comparison],windows=windows))
            p50,p95=quantile(ratios[comparison],.5),quantile(ratios[comparison],.95)
            check(near(r['p50'],p50) and near(r['p95'],p95),'forged summary quantile')
            all_summaries.append({'dtype':dtype,'path':path,'start_parity':start,'comparison':comparison,'direction':DIRECTIONS[comparison],'p50':p50,'p95':p95})
        equal_fields(cohort[-1],dict(samples=12*windows,pairs=3*windows,summaries=3,pre_post_bits=True,
            pre_post_graphs=True,guards=True,immutable_inputs=True,noop_rejected=True))
    # Global timing chronology must also preserve entire cohort order.
    check([(r['dtype'],r['path'],r['start_parity']) for r in timing]==[key for key in expected_configs for _ in range(15*windows+4)],'global reordered chronology')
    equal_fields(rows[-1],dict(configurations=len(expected_configs),samples=len(expected_configs)*12*windows,
        pairs=len(expected_configs)*3*windows,summaries=len(expected_configs)*3,rejected=0,passed=True))
    check(sum(r['kind']=='complete' for r in rows)==1,'duplicate completion')
    cells=[]
    for dtype in dtypes:
        own=[r for r in all_summaries if r['dtype']==dtype and r['comparison']==0]
        passed=all(r['p50']<1 and r['p95']<1 for r in own)
        cells.append({'dtype':dtype,'toolkit':toolkit,'windows':windows,'valid':True,'own_win':passed,
            'admission':windows==101 and passed,'advance101':windows==21 and passed,
            'worst_own_p50':max(r['p50'] for r in own),'worst_own_p95':max(r['p95'] for r in own),
            'constituents':[r for r in all_summaries if r['dtype']==dtype]})
    return {'valid':True,'configurations':len(expected_configs),'samples':len(expected_configs)*12*windows,
            'pairs':len(expected_configs)*3*windows,'summaries':all_summaries,'cells':cells}

def verify_run(directory, binding_path, qualification_path, ssh_log):
    import run
    directory=Path(directory)
    binding=json.loads(Path(binding_path).read_text())
    qualification=json.loads(Path(qualification_path).read_text())
    result=json.loads((directory/'result.json').read_text())
    equal_fields(result,{'toolkit':binding['toolkit'],'binary_sha':binding['binary_sha'],
        'source_sha':binding['inputs']['tests/gemm_bi_fixed_performance.rs']})
    check(hashlib.sha256((directory/'test.log').read_bytes()).hexdigest()==result['test_log_sha'],'raw log hash mismatch')
    # The wrapper emits exactly these eight ordered records for one attempt.
    # Parsing the whole transcript also rejects extra failures, duplicates,
    # concatenated attempts, success substrings and conflicting later exits.
    lines=Path(ssh_log).read_text().splitlines()
    check(len(lines)==8,'SSH transcript must contain exactly one complete eight-record attempt')
    pre=json.loads(lines[0])
    post=json.loads(lines[3])
    for phase,observed in [('PRE',pre),('POST',post)]:
        saved=json.loads((directory/(phase.lower()+'.json')).read_text())
        check(observed==saved,f'{phase}: SSH/saved telemetry mismatch')
        equal_fields(observed,{'phase':phase})
        run.snapshot(phase,observed['gpu'],observed['apps'],observed['gpu_exit'],observed['apps_exit'])
    check(lines[1].startswith('COMMAND '),'missing/misordered COMMAND')
    command=json.loads(lines[1][len('COMMAND '):])
    check(command==[binding['binary'],'--ignored','--exact','fixed_ada_half_s3_auto_fast_paired',
        '--nocapture','--test-threads=1'],'wrong command/test/binary for this attempt')
    command_exit=re.fullmatch(r'COMMAND_EXIT test\.log (-?\d+)',lines[2])
    check(command_exit is not None,'missing/malformed/misordered COMMAND_EXIT')
    check(lines[4].startswith('RUN_RESULT '),'missing/misordered RUN_RESULT')
    check(json.loads(lines[4][len('RUN_RESULT '):])==result,'SSH RUN_RESULT differs from saved result.json')
    check(lines[5]=='WRAPPER_COMPLETE','missing/malformed/misordered WRAPPER_COMPLETE')
    wrapper_exit=re.fullmatch(r'WRAPPER_EXIT=(-?\d+)',lines[6])
    ssh_exit=re.fullmatch(r'OUTER_SSH_EXIT=(-?\d+)',lines[7])
    check(wrapper_exit is not None and ssh_exit is not None,'missing/malformed/misordered wrapper/SSH exit')
    check(int(command_exit[1])==result['test_exit'],'COMMAND_EXIT conflicts with saved test exit')
    run.closure(result['test_exit'],result['post_exit'],int(wrapper_exit[1]),int(ssh_exit[1]))
    check('test result: ok. 1 passed; 0 failed;' in (directory/'test.log').read_text(),'test harness exit marker')
    report=verify(records(directory/'test.log'),binding,qualification,result['windows'],result['dtypes'].split(','))
    report['validation']={'revision':'fix1-exact-same-attempt',
        'analyzer_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ssh_log_sha':hashlib.sha256(Path(ssh_log).read_bytes()).hexdigest(),
        'binding_sha':hashlib.sha256(Path(binding_path).read_bytes()).hexdigest(),
        'qualification_sha':hashlib.sha256(Path(qualification_path).read_bytes()).hexdigest()}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory');p.add_argument('binding');p.add_argument('qualification');p.add_argument('ssh_log')
    args=p.parse_args()
    try: print(json.dumps(verify_run(args.directory,args.binding,args.qualification,args.ssh_log),indent=2))
    except Exception as error:
        print('INVALID: '+str(error),file=sys.stderr);sys.exit(1)
