#!/usr/bin/env python3
"""Verify archived actual build inputs, binaries and production cache payloads."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()

def archive(path):
    with tarfile.open(path) as tar:
        return {m.name:sha(tar.extractfile(m).read()) for m in tar.getmembers() if m.isfile() and not Path(m.name).name.startswith('._')}

def main():
    source=archive(HERE/'source-final.tar.gz')
    baseline=archive(HERE.parent/'ada-half-s3-force-20260907/source-final-v2.tar.gz')
    assert source.keys()==baseline.keys(),'source input inventory differs from approved Task6A'
    changed=[p for p in source if source[p]!=baseline[p]]
    assert changed==['tests/gemm_bi_fixed_performance.rs'],f'unexpected Task6A source change {changed}'
    output=[]
    for tag in ('128','130','132'):
        binding=json.loads((HERE/f'cuda{tag}-binding-final.json').read_text())
        assert source==binding['inputs'],f'{tag} actual source/build inputs mismatch'
        qualification=json.loads((HERE.parent/'ada-half-s3-force-20260907'/f'identity-cuda{tag}.json').read_text())
        payloads=[]
        with tarfile.open(HERE/f'cuda{tag}-final-binary-cache.tar.gz') as tar:
            members=[m for m in tar.getmembers() if m.isfile()]
            binaries=[m for m in members if m.name==binding['binary'].lstrip('/')]
            assert len(binaries)==1 and sha(tar.extractfile(binaries[0]).read())==binding['binary_sha']
            caches=[m for m in members if m.name.endswith('.bin')]
            assert len(caches)==3,f'{tag} expected Fixed+two retained Triad artifacts'
            for m in caches:
                data=tar.extractfile(m).read()
                if b'gemm_bi_nn_fixed_sm89_tc128_s3_v1_bf16' in data:
                    assert qualification['fixed_invocation_digest'] in m.name
                    assert sha(data[91:])==qualification['fixed_artifact_digest']
                    payloads.append({'cache':m.name,'cache_sha':sha(data),'ptx_sha':sha(data[91:]),
                        'invocation':qualification['fixed_invocation_digest'],'source':qualification['fixed_source_digest']})
            assert len(payloads)==1
        output.append({'toolkit':binding['toolkit'],'input_files':len(source),'binary_sha':binding['binary_sha'],
            'cache_count':3,'fixed':payloads[0],'production_artifact_matches_approved_task6a':True})
    print(json.dumps({'valid':True,'source_files':len(source),'changed_from_task6a':changed,'toolkits':output},indent=2))

if __name__=='__main__':main()
