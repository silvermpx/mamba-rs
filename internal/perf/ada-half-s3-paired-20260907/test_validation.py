#!/usr/bin/env python3
"""Adversarial protocol tests; seed physical identities come from smoke1."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import analyze
import run

HERE=Path(__file__).resolve().parent
SEED=analyze.records(HERE/'cuda128-smoke1/test.log')
BINDING=json.loads((HERE/'cuda128-binding.json').read_text())
QUAL=json.loads((HERE.parent/'ada-half-s3-force-20260907/identity-cuda128.json').read_text())

def fixture(profile='win', windows=21):
    # Independently constructed21-window input. Expected median and p95 below
    # are literal0.8/0.8,1.1/1.1,0.8/1.2, not produced by analyzer helpers.
    identity=copy.deepcopy(SEED[0]);identity['windows']=windows
    rows=[identity]+copy.deepcopy([r for r in SEED if r['kind']=='physical'])
    base={k:identity[k] for k in ('schema','toolkit','shape','bias','alpha','beta','revision')}
    for dtype in ('bf16','f16'):
        for path in ('eager','graph'):
            for start in (0,1):
                key=dict(base,dtype=dtype,path=path,start_parity=start)
                samples=[];pairs=[];ratios=[[],[],[]]
                for window in range(windows):
                    reverse=(window%2)!=start
                    for traversal,comparison in enumerate((2,1,0) if reverse else (0,1,2)):
                        aa,bb=(('AUTO','S3'),('Fast','AUTO'),('Fast','S3'))[comparison]
                        candidate=110.0 if profile=='loss' else 120.0 if profile=='mixed' and window>=(19 if windows==21 else 95) else 80.0
                        values={'AUTO':100.0,'S3':candidate,'Fast':75.0}
                        offset=len(samples)
                        for position,arm in enumerate((bb,aa,aa,bb) if reverse else (aa,bb,bb,aa)):
                            samples.append(dict(key,kind='sample',chronology=len(samples),window=window,comparison=comparison,
                                traversal=traversal,order='BAAB' if reverse else 'ABBA',position=position,arm=arm,logical_ops=20,us=values[arm]))
                        ratio=values[bb]/values[aa]
                        ratios[comparison].append(ratio)
                        pairs.append(dict(key,kind='pair',window=window,comparison=comparison,traversal=traversal,
                            observations=[offset,offset+1,offset+2,offset+3],ratio=ratio))
                rows+=samples+pairs
                for comparison,direction in enumerate(('S3/AUTO','AUTO/Fast','S3/Fast')):
                    ordered=sorted(ratios[comparison])
                    rows.append(dict(key,kind='summary',comparison=comparison,direction=direction,windows=windows,
                        p50=ordered[10 if windows==21 else 50],p95=ordered[19 if windows==21 else 95]))
                rows.append(dict(key,kind='configuration_complete',samples=12*windows,pairs=3*windows,summaries=3,pre_post_bits=True,
                    pre_post_graphs=True,guards=True,immutable_inputs=True,noop_rejected=True))
    rows.append(dict(base,kind='complete',configurations=8,samples=96*windows,pairs=24*windows,summaries=24,rejected=0,passed=True))
    return rows

def validate(rows,windows=21): return analyze.verify(rows,BINDING,QUAL,windows,['bf16','f16'])

class Protocol(unittest.TestCase):
    def test_actual_one_window_smoke_closes(self):
        report=analyze.verify_run(HERE/'cuda128-smoke1',HERE/'cuda128-binding.json',
            HERE.parent/'ada-half-s3-force-20260907/identity-cuda128.json',HERE/'cuda128-smoke1-ssh.log')
        self.assertEqual((report['configurations'],report['samples'],report['pairs']),(8,96,24))

    def test_genuine_recomputed_win_loss_and_mixed_p95_are_valid(self):
        for profile,p50,p95,advance in [('win',.8,.8,True),('loss',1.1,1.1,False),('mixed',.8,1.2,False)]:
            for windows in (21,101):
                with self.subTest(profile=profile,windows=windows):
                    report=validate(fixture(profile,windows),windows)
                    self.assertTrue(report['valid'])
                    for cell in report['cells']:
                        self.assertEqual((cell['worst_own_p50'],cell['worst_own_p95']),(p50,p95))
                        self.assertEqual(cell['advance101'],windows==21 and advance)
                        self.assertEqual(cell['admission'],windows==101 and advance)

    def test_missing_duplicate_foreign_sample_pair_and_cohort(self):
        for kind in ('sample','pair','summary','configuration_complete','physical','complete','identity'):
            for change in ('missing','duplicate'):
                with self.subTest(kind=kind,change=change):
                    rows=fixture();i=next(i for i,r in enumerate(rows) if r['kind']==kind)
                    if change=='missing': rows.pop(i)
                    else: rows.insert(i,copy.deepcopy(rows[i]))
                    with self.assertRaises(ValueError): validate(rows)
        for kind in ('sample','pair'):
            rows=fixture();next(r for r in rows if r['kind']==kind)['dtype']='f32'
            with self.assertRaises(ValueError):validate(rows)

    def test_chronology_arm_position_parity_traversal_and_finite_times(self):
        mutations={'arm':'Fast','position':3,'start_parity':1,'traversal':2,'order':'BAAB','chronology':1,
                   'window':1,'comparison':2,'logical_ops':19,'path':'foreign','dtype':'f32'}
        for key,value in mutations.items():
            with self.subTest(key=key):
                rows=fixture();next(r for r in rows if r['kind']=='sample')[key]=value
                with self.assertRaises(ValueError):validate(rows)
        for value in (0,-1,float('nan'),float('inf')):
            rows=fixture();next(r for r in rows if r['kind']=='sample')['us']=value
            with self.assertRaises(ValueError):validate(rows)
        rows=fixture();i=next(i for i,r in enumerate(rows) if r['kind']=='sample');rows[i],rows[i+1]=rows[i+1],rows[i]
        with self.assertRaises(ValueError):validate(rows)

    def test_literal_cell_source_binary_compiler_revision_and_modes(self):
        cases={'shape':[4621,256,2304],'dtype':'f32','toolkit':'13.2','revision':43,
            'source_sha':'0'*64,'binary_sha':'0'*64,'fixed_artifact_digest':'0'*64,
            'fixed_source_digest':'0'*64,'header_manifest_digest':'0'*64,'nvrtc_library_known':False,
            'math':'CUBLAS_PEDANTIC_MATH','compute':'CUBLAS_COMPUTE_32F_PEDANTIC',
            'pointer_mode':'CUBLAS_POINTER_MODE_DEVICE','atomics':'CUBLAS_ATOMICS_ALLOWED',
            'algorithm':'CUBLAS_GEMM_DEFAULT','bias_broadcast':True}
        for key,value in cases.items():
            with self.subTest(key=key):
                rows=fixture();rows[0][key]=value
                with self.assertRaises(ValueError):validate(rows)

    def test_stale_missing_captured_arguments_or_twenty_node_fail(self):
        for target in ('one','twenty'):
            for field in ('bundle','pointers','abi','sixth_rejected','symbol','shared_bytes','grid'):
                for change in ('missing','wrong'):
                    with self.subTest(target=target,field=field,change=change):
                        rows=fixture();r=next(r for r in rows if r['kind']=='physical' and r['arm']=='S3')
                        node=r[target][-1]
                        if change=='missing':node.pop(field)
                        else:node[field]=None
                        with self.assertRaises(ValueError):validate(rows)
        for field in ('noop_rejected','pre_post_bits','pre_post_graphs','guards','immutable_inputs'):
            rows=fixture();next(r for r in rows if r['kind']=='configuration_complete')[field]=False
            with self.assertRaises(ValueError):validate(rows)

    def test_forged_pair_summary_or_pair_membership_fails(self):
        for kind,field,value in [('pair','ratio',.01),('pair','observations',[4,5,6,7]),
                                 ('summary','p50',.01),('summary','p95',.01)]:
            rows=fixture();next(r for r in rows if r['kind']==kind)[field]=value
            with self.assertRaises(ValueError):validate(rows)

class Wrapper(unittest.TestCase):
    def test_verify_run_rejects_conflicting_or_foreign_attempt_transcripts(self):
        original=(HERE/'cuda128-smoke1-ssh.log').read_text()
        lines=original.splitlines()
        variants={
            'conflicting_nonzero': original+'WRAPPER_EXIT=1\nOUTER_SSH_EXIT=255\n',
            'nonzero_wrapper': original.replace('WRAPPER_EXIT=0','WRAPPER_EXIT=1'),
            'nonzero_ssh': original.replace('OUTER_SSH_EXIT=0','OUTER_SSH_EXIT=255'),
            'duplicate_complete': original+'WRAPPER_COMPLETE\n',
            'missing_complete': original.replace('WRAPPER_COMPLETE\n',''),
            'suffix_wrapper': original.replace('WRAPPER_EXIT=0','WRAPPER_EXIT=0garbage'),
            'suffix_complete': original.replace('WRAPPER_COMPLETE','WRAPPER_COMPLETEgarbage'),
            'foreign_attempt': (HERE/'cuda128-smokefinal-ssh.log').read_text(),
            'wrong_command_binary': original.replace(BINDING['binary'],BINDING['binary']+'-foreign'),
            'wrong_command_test': original.replace('fixed_ada_half_s3_auto_fast_paired','other_test'),
            'nonzero_command_exit': original.replace('COMMAND_EXIT test.log 0','COMMAND_EXIT test.log 101'),
            'command_exit_suffix': original.replace('COMMAND_EXIT test.log 0','COMMAND_EXIT test.log 0suffix'),
            'wrong_result': original.replace('"windows": 1','"windows": 21'),
            'wrong_pre_phase': original.replace('"phase": "PRE"','"phase": "POST"'),
            'wrong_post_phase': original.replace('"phase": "POST"','"phase": "PRE"'),
            'wrong_telemetry': original.replace('9 %, 1 %','8 %, 1 %'),
            'failure_marker': original+'WRAPPER_FAILURE forced\n',
        }
        swapped=lines.copy();swapped[0],swapped[3]=swapped[3],swapped[0]
        variants['swapped_telemetry']='\n'.join(swapped)+'\n'
        for label,transcript in variants.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'ssh.log';path.write_text(transcript)
                with self.assertRaises(ValueError):
                    analyze.verify_run(HERE/'cuda128-smoke1',HERE/'cuda128-binding.json',
                        HERE.parent/'ada-half-s3-force-20260907/identity-cuda128.json',path)

    def test_verify_run_binds_saved_telemetry_phase_and_result_to_transcript(self):
        for filename,field,value in [('pre.json','phase','POST'),('post.json','phase','PRE'),
            ('pre.json','utc','1900-01-01T00:00:00Z'),('post.json','utc','1900-01-01T00:00:00Z'),
            ('result.json','cache_files',{})]:
            with self.subTest(filename=filename,field=field), tempfile.TemporaryDirectory() as temp:
                directory=Path(temp)
                for name in ('pre.json','post.json','result.json','test.log'):
                    (directory/name).write_bytes((HERE/'cuda128-smoke1'/name).read_bytes())
                record=json.loads((directory/filename).read_text());record[field]=value
                (directory/filename).write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    analyze.verify_run(directory,HERE/'cuda128-binding.json',
                        HERE.parent/'ada-half-s3-force-20260907/identity-cuda128.json',HERE/'cuda128-smoke1-ssh.log')

    def test_preflight_rejects_wrong_device_busy_utilization_apps_and_query_failures(self):
        gpu=run.UUID+', NVIDIA RTX 6000 Ada Generation, 8.9, 0 %, 0 %\n'
        run.snapshot('PRE',gpu,'')
        for g,a,ge,ae in [(gpu.replace(run.UUID,'GPU-other'),'',0,0),(gpu,'123, job',0,0),
            (gpu.replace('0 %','1 %'),'',0,0),(gpu,'',1,0),(gpu,'',0,1)]:
            with self.assertRaises(ValueError):run.snapshot('PRE',g,a,ge,ae)
        run.snapshot('POST',gpu.replace('0 %','9 %'),'')
        with self.assertRaises(ValueError):run.snapshot('POST',gpu,'123, job')
        with self.assertRaises(ValueError):run.snapshot('POST',gpu.replace(run.UUID,'GPU-other'),'')

    def test_source_binary_toolkit_and_all_exit_layers_fail_closed(self):
        inputs={'test':'a'};binding={'toolkit':'12.8','inputs':inputs,'binary_sha':'b'}
        run.validate_binding(binding,'12.8',inputs,'b')
        for toolkit,source,binary in [('13.0',inputs,'b'),('12.8',{},'b'),('12.8',inputs,'c')]:
            with self.assertRaises(ValueError):run.validate_binding(binding,toolkit,source,binary)
        run.closure(0,0,0,0)
        for codes in [(101,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,255)]:
            with self.assertRaises(ValueError):run.closure(*codes)

if __name__=='__main__': unittest.main(verbosity=2)
