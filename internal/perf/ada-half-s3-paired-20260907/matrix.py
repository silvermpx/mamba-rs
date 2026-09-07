#!/usr/bin/env python3
"""Build the exact six-cell decision from validated screen and fresh confirm."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def own_failures(cell):
    return [r for r in cell['constituents'] if r['comparison']==0 and (r['p50']>=1 or r['p95']>=1)]

def main():
    matrix=[]
    for tag,toolkit in [('128','12.8'),('130','13.0'),('132','13.2')]:
        screen_path=HERE/f'cuda{tag}-screen21-analysis.json'
        screen=json.loads(screen_path.read_text())
        assert screen['valid'] and len(screen['cells'])==2
        eligible=[c['dtype'] for c in screen['cells'] if c['advance101']]
        confirm_path=HERE/f'cuda{tag}-confirm101-analysis.json'
        confirm=json.loads(confirm_path.read_text()) if eligible else None
        if confirm:
            assert confirm['valid']
            assert [c['dtype'] for c in confirm['cells']]==eligible
        for dtype in ('bf16','f16'):
            s=next(c for c in screen['cells'] if c['dtype']==dtype)
            c=next((c for c in confirm['cells'] if c['dtype']==dtype),None) if confirm else None
            assert s['toolkit']==toolkit and s['windows']==21
            if c:assert c['windows']==101 and c['toolkit']==toolkit
            admit=bool(c and c['admission'])
            final=c or s
            vendor={}
            for comparison,direction in [(1,'AUTO/Fast'),(2,'S3/Fast')]:
                values=[v for v in final['constituents'] if v['comparison']==comparison]
                vendor[direction]={'worst_p50':max(v['p50'] for v in values),'worst_p95':max(v['p95'] for v in values),
                    'all_path_parity_win':all(v['p50']<1 and v['p95']<1 for v in values)}
            matrix.append({'toolkit':toolkit,'dtype':dtype,'shape':[4621,768,2304],'bias':False,'revision':42,
                'candidate':'Tc128Sm89S3','observed_auto':'Tc128Sm89Swizzle','admission':admit,
                'outcome':'confirmed-own-win' if admit else 'confirm-rejected' if c else 'screen-rejected',
                'screen':s,'confirm':c,'failed_configurations':own_failures(final),'vendor_context':vendor,
                'screen_raw_sha':json.loads((HERE/f'cuda{tag}-screen21/result.json').read_text())['test_log_sha'],
                'confirm_raw_sha':json.loads((HERE/f'cuda{tag}-confirm101/result.json').read_text())['test_log_sha'] if c else None})
    assert len(matrix)==6
    print(json.dumps({'valid':True,'rule':'S3/AUTO p50<1 AND p95<1 in every eager/graph x start0/1 at21 then fresh101; vendor context independent',
        'percentile':'round((len-1)*fraction)','cells':matrix},indent=2))

if __name__=='__main__':main()
