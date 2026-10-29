"""Pure admission test DEFINITIONS; never launch a Job or access real payloads.

The 108-call ledger, source plane, cached predictions, native receipts and
performance vectors below are all synthetic records in temporary directories.
These definitions do not constitute actual recovery QA or numerical release.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import memory_transport_execution_20261029 as execution
from experiments import memory_transport_study_20261029 as study


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def ledger_write(base, records):
    (base/'optimizer_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')


def call(category, run_id, step, cid):
    identity=dict(category=category,run_id=run_id,step=step,call_id=cid)
    return [dict(identity,event=e) for e in ('optimizer_step_attempt','optimizer_step_completed')]


class ExecutionGuards(unittest.TestCase):
    def test_UNKNOWN_refuses_all_entrances_before_source_reads_or_native_actions(self):
        from experiments import uncapped_windows_job_20261021 as job
        with patch.object(execution,'RELEASE_SHA','UNKNOWN'),patch.object(execution,'sha') as digest, \
                patch.object(execution,'read') as read,patch.object(job,'supervise') as native:
            for action in (execution.require_release,lambda:execution.freeze('a'*40),
                    lambda:execution.launch(execution.BASE,'a'*40,'prepare'),
                    lambda:execution.run(execution.BASE,'a'*40)):
                with self.assertRaisesRegex(ValueError,'UNKNOWN'): action()
            digest.assert_not_called();read.assert_not_called();native.assert_not_called()

    def test_release_requires_parent_review_authority_and_new1848_not_old_balance(self):
        value=dict(parent_remote_archival_verified=True,independent_implementation_review_passed=True,
            root_execution_authorized=True,parent_complete_sha256=execution.PARENT_COMPLETE_SHA,optimizer_upper_bound=1848)
        with patch.object(execution,'RELEASE_SHA','a'*64),patch.object(execution,'sha',return_value='a'*64), \
                patch.object(execution,'recovery_gate'):
            with patch.object(execution,'read',return_value=value):self.assertEqual(execution.require_release(),value)
            for key, bad in [('parent_remote_archival_verified',False),('independent_implementation_review_passed',False),
                    ('root_execution_authorized',False),('optimizer_upper_bound',976),('parent_complete_sha256','0'*64)]:
                altered=dict(value);altered[key]=bad
                with patch.object(execution,'read',return_value=altered):
                    with self.assertRaises(ValueError):execution.require_release()

    def test_fixed_actual_CPU34_CUDA3_records_are_byte_bound_not_replaced_by_flags(self):
        required={'results/memory_transport_20261029/model_cpu_qualification.json':'4d3068e30635c2cf06a7cbe4b35d47681833475001a09e8bfb755db498082c41',
            'results/memory_transport_20261029/model_cuda_qualification.json':'5f0f37f28f48749ea9bf196c0e4c8d1c827958bc25abe8ebf483aad40025ea9f'}
        for name,digest in required.items():self.assertEqual(execution.FIXED_RECORDS[name],digest)
        def digest(path):return execution.FIXED_RECORDS[Path(path).relative_to(execution.ROOT).as_posix()]
        with patch.object(execution,'sha',side_effect=digest),patch.object(execution,'read') as read:
            bound=execution.fixed_evidence()
            for name,value in required.items():self.assertEqual(bound[str(execution.ROOT/name)],value)
            read.assert_not_called()
        for target in required:
            def corrupt(path):
                name=Path(path).relative_to(execution.ROOT).as_posix()
                return '0'*64 if name==target else execution.FIXED_RECORDS[name]
            with patch.object(execution,'sha',side_effect=corrupt):
                with self.assertRaises(ValueError):execution.fixed_evidence()

    def test_source_plane_is_the_single_frozen_transaction_not_an_adaptive_replacement(self):
        from experiments import r_memory_execution_20261025 as io
        expected={study.SOURCE/name:digest for name,digest in study.SOURCE_SHA.items()}
        expected.update({study.PLANE:study.PLANE_SHA,study.PLANE_PUBLIC:study.PLANE_PUBLIC_SHA})
        plane=dict(status='COMPLETED',cutoff_pass=True,eigh_count=1,source_sha256=study.SOURCE_SHA['model.json'])
        self.assertEqual(study.PLANE_SHA,'3dee4225e77e5bf6ab6ad58a73ab4ad9c8a263e647d053fcb6842cc4a7281941')
        self.assertEqual(study.PLANE.parent.name,'source_plane01')
        with patch.object(io,'sha',side_effect=lambda p:expected.get(Path(p),'wrong')),patch.object(io,'read',return_value=plane) as read:
            self.assertEqual(study.source_identity()['plane_sha256'],study.PLANE_SHA)
            read.assert_called_once_with(study.PLANE)
            with patch.object(study,'PLANE',study.PLANE.parent/'replacement.json'):
                with self.assertRaises(ValueError):study.source_identity()
        for key,value in [('cutoff_pass',False),('eigh_count',2),('source_sha256','0'*64)]:
            bad=dict(plane);bad[key]=value
            with patch.object(io,'sha',side_effect=lambda p:expected[Path(p)]),patch.object(io,'read',return_value=bad):
                with self.assertRaises(ValueError):study.source_identity()

    def test_metadata_freeze_includes_plane_and_qualified_bytes_without_new_source_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'new_study';sgm=root/'synthetic_parent'
            profile=sgm/'study/A_28311/fit000096/state_gated_profile.json';write(profile,{})
            plane=root/'source_plane01/result.json';public=root/'source_plane_public.json'
            qualified={str(root/'CPU.json'):'c'*64,str(root/'CUDA.json'):'d'*64}
            parent=root/'closed_execution'
            records={parent/'binding.json':dict(files={str(profile):'f'*64}),
                sgm/'qualification/R_N_fit297/evidence.json':dict(prediction_files=[]),
                sgm/'study/A_28311/fit000096/evidence.json':dict(prediction_files=[])}
            lock_proof=dict(retained_r_memory_lock=dict(path=str(root/'retained_r.lock'),recovery_path=str(root/'retained_r_recovery.json')))
            with patch.multiple(execution,ROOT=root,BASE=base,PARENT=parent,OLD_BASE=root/'old_study01',DOCUMENTS=(),
                    RELEASE_RECORD=root/'release.json',RECOVERY_RECORD=root/'recovery_release.json'), \
                    patch.object(execution,'require_release'),patch.object(execution,'check_locks',return_value=lock_proof), \
                    patch.object(execution,'recovery_gate',return_value={'evidence':{}}), \
                    patch.object(execution,'original_prelaunch_evidence',return_value={'native_jobs_started':0,'optimizer_calls_spent':0}), \
                    patch.object(execution,'source_closure',return_value=[]),patch.object(execution,'sha',return_value='f'*64), \
                    patch.object(execution,'read',side_effect=lambda path:records[Path(path)]), \
                    patch.object(execution,'fixed_evidence',return_value=qualified), \
                    patch.object(execution,'parent_proof',return_value=({'files':[]},{'synthetic_closed':True})), \
                    patch.multiple(study,SOURCE=root/'source',SOURCE_SHA={'model.json':'b'*64},SGM=sgm,EVIDENCE_SHA={},PLANE=plane,PLANE_PUBLIC=public), \
                    patch.object(study,'source_identity') as identity:
                binding=execution.freeze('a'*40,base)
            self.assertEqual(binding['files'][str(plane)],study.PLANE_SHA)
            self.assertEqual(binding['files'][str(profile)],'f'*64)
            for name,digest in qualified.items():self.assertEqual(binding['files'][name],digest)
            self.assertEqual(binding['source_eigh_calls_this_loop'],0)
            self.assertEqual(binding['NNLS_calls'],0)
            self.assertFalse(binding['mathematical_suite_rerun'])
            identity.assert_called_once_with()

    def qa_case(self,folder,variants=('A','S','P')):
        base=Path(folder);rows=[]
        for variant in variants:
            for label in ('continuous','restored'):
                for step in range(1,19):rows.extend(call('qa',f'qa_{variant}_{label}',step,f'{variant}_{label}_{step}'))
        ledger_write(base,rows);path=base/'qa/optimizer_projection.json';write(path,rows)
        result=dict(eligible_variants=list(variants),models=[dict(variant=v,status='PASS',actual_optimizer_calls=36) for v in variants],
            qa_journal_projection=dict(path=str(path.resolve()),sha256=execution.sha(path)))
        return base,result,rows,path

    def test_QA_needs_six_real_unique18_step_trajectories_and_allows_later_formal(self):
        with tempfile.TemporaryDirectory() as folder:
            base,result,rows,path=self.qa_case(folder)
            self.assertTrue(execution.qa_evidence(base,result))
            self.assertEqual(study.counts(base)['categories']['qa'],dict(attempted=108,completed=108))
            ledger_write(base,rows+call('formal','A_30311',1,'later'))
            self.assertTrue(execution.qa_evidence(base,result))
            ledger_write(base,rows[:-1])
            with self.assertRaises(ValueError):execution.qa_evidence(base,result)
            ledger_write(base,rows)
            bad=copy.deepcopy(result);bad['models'][2]['variant']='A'
            with self.assertRaises(ValueError):execution.qa_evidence(base,bad)
            bad=copy.deepcopy(result);bad['models'][2]['actual_optimizer_calls']=35
            with self.assertRaises(ValueError):execution.qa_evidence(base,bad)

    def test_QA_rejects_early_formal_fake_projection_and_unregistered_run(self):
        with tempfile.TemporaryDirectory() as folder:
            base,result,rows,path=self.qa_case(folder)
            ledger_write(base,call('formal','A_30311',1,'early')+rows)
            with self.assertRaises(ValueError):execution.qa_evidence(base,result)
            ledger_write(base,rows);write(path,[]);result['qa_journal_projection']['sha256']=execution.sha(path)
            with self.assertRaises(ValueError):execution.qa_evidence(base,result)
            bad=copy.deepcopy(rows)
            for item in bad:
                if item['run_id']=='qa_P_restored':item['run_id']='qa_P_unregistered'
            ledger_write(base,bad);write(path,bad);result['qa_journal_projection']['sha256']=execution.sha(path)
            with self.assertRaises(ValueError):execution.qa_evidence(base,result)

    def test_partial_QA_can_retain_healthy_controls_without_inventing_failed_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            base,result,rows,path=self.qa_case(folder,('A','S'))
            pending=call('qa','qa_P_continuous',1,'P_pending')[0]
            ledger_write(base,rows+[pending]);write(path,rows+[pending]);result['qa_journal_projection']['sha256']=execution.sha(path)
            result['models'].append(dict(variant='P',status='NUMERICAL_UNRESOLVED',failure_is_physical_instability=False,quarantine={'synthetic':True}))
            self.assertTrue(execution.qa_evidence(base,result))
            self.assertEqual(study.counts(base)['pending'][0]['call_id'],'P_pending')
            result['eligible_variants'].append('P')
            with self.assertRaises(ValueError):execution.qa_evidence(base,result)

    def cross_case(self,folder):
        base=Path(folder);ids=[f'artificial_{i}' for i in range(297)];write(base/'prepare.json',dict(fit_ids=ids))
        pair=dict(finite_to_endpoint=True,los_pair_pass=True,coi_normalized_rmse=0.,common_normalized_rmse=0.)
        genuine=dict(available=True,finite=True,numerically_admissible=True,coi_normalized_rmse=0.,common_normalized_rmse=0.)
        intervals={}
        for scope,n in (('8s',800),('actual15',1000)):
            interval=dict(qualified=True,actual_output_steps=n,full_R_N_fine_finite=True,full_A_fine_finite=True)
            for direction in ('R_N_to_A','A_to_R_N'):
                interval[direction+'_pair']=dict(pair)
                interval[direction+'_scored_numerics']=[dict(pair,available=True,horizon='prefix')]
                if scope=='actual15':interval[direction+'_actual_interval']=dict(genuine)
            intervals[scope]=interval
        record=dict(qualified=True,record_count=297,event_ids=ids,original_schema_aliases=dict(R_N='frozen_A96',A='current_A0'),
            source_evidence_sha256=study.EVIDENCE_SHA['study/A_28311/fit000096/evidence.json'],source_A96_prediction_recomputed=False,
            scope='A96_A0_fit297_bidirectional_RK4_fine005_base01_8s_actual15',bidirectional=True,
            history_rows=11,initial_observation_index=10,fine_dt=.005,output_dt=.01,coarse_stride=2,
            rows=[dict(event_id=eid,actual_steps=1000,qualified=True,intervals=copy.deepcopy(intervals)) for eid in ids])
        path=base/'qualification/A_A96_compatibility.json';write(path,record)
        result=dict(A_A96_compatibility_qualified=True,A_A96_compatibility=dict(path=str(path.resolve()),sha256=execution.sha(path)))
        return base,result,record,path

    def test_cross_requires_exact_identity_scope_path_hash_not_qualified_boolean(self):
        with tempfile.TemporaryDirectory() as folder:
            base,result,record,path=self.cross_case(folder)
            self.assertTrue(execution.cross_model_evidence(base,result))
            for key,value in [('record_count',296),('source_A96_prediction_recomputed',True),('scope','8s_only'),('bidirectional',False)]:
                bad=copy.deepcopy(record);bad[key]=value;write(path,bad);result['A_A96_compatibility']['sha256']=execution.sha(path)
                with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)
            bad=copy.deepcopy(record);bad['rows'][1]['event_id']=bad['rows'][0]['event_id'];write(path,bad)
            result['A_A96_compatibility']['sha256']=execution.sha(path)
            with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)
            write(path,record);result['A_A96_compatibility']['sha256']='0'*64
            with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)
            result['A_A96_compatibility']['sha256']=execution.sha(path);result['A_A96_compatibility']['path']=str(path.parent/'wrong.json')
            with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)

    def test_cross_checks_original_bidirectional_genuine_interval_and_fine_substeps(self):
        with tempfile.TemporaryDirectory() as folder:
            base,result,record,path=self.cross_case(folder)
            for scope in ('8s','actual15'):
                for direction in ('R_N_to_A','A_to_R_N'):
                    bad=copy.deepcopy(record);bad['rows'][0]['intervals'][scope][direction+'_pair']['coi_normalized_rmse']=.002
                    write(path,bad);result['A_A96_compatibility']['sha256']=execution.sha(path)
                    with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)
                for field in ('full_R_N_fine_finite','full_A_fine_finite'):
                    bad=copy.deepcopy(record);bad['rows'][0]['intervals'][scope][field]=False
                    write(path,bad);result['A_A96_compatibility']['sha256']=execution.sha(path)
                    with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)
            for direction in ('R_N_to_A','A_to_R_N'):
                bad=copy.deepcopy(record);bad['rows'][0]['intervals']['actual15'][direction+'_actual_interval']['finite']=False
                write(path,bad);result['A_A96_compatibility']['sha256']=execution.sha(path)
                with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)

    def test_performance_flags_need_actual_ordered_complete_vectors_and_original_tolerance(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);ids=[f'artificial_{i}' for i in range(12)];write(base/'prepare.json',dict(cost12_ids=ids))
            target=base/'qualification/S_performance';rows=[]
            for name in ('uncached','cached'):
                row=dict(implementation=name,all_raw_gradients=True,J=1.,gradients=dict(kappa_raw=0.,lambda_raw=0.,a_bar=[0.]*45,s_raw=0.))
                rows.append(row);write(target/(name+'_attempt.json'),dict(variant='S',implementation=name,optimizer_calls=0));write(target/(name+'_result.json'),row)
            value=dict(status='PASS',equivalent=True,variant='S',event_ids=ids,horizon_s=8.,method='rk4',dtype='float64',optimizer_calls=0,atol=1e-10,rtol=1e-8,ordered_runs=rows)
            path=target/'summary.json';write(path,value);link=dict(performance_evidence=dict(path=str(path.resolve()),sha256=execution.sha(path)))
            self.assertTrue(execution.performance_evidence(base,'S',link))
            for edit in ('order','gradient','missing_coordinate','extra_attempt','range'):
                bad=copy.deepcopy(value)
                if edit=='order':bad['ordered_runs'].reverse()
                elif edit=='gradient':bad['ordered_runs'][1]['gradients']['s_raw']=1e-7
                elif edit=='missing_coordinate':del bad['ordered_runs'][1]['gradients']['s_raw']
                elif edit=='extra_attempt':write(target/'retry_attempt.json',{})
                else:bad['horizon_s']=5.
                for row in bad['ordered_runs']:write(target/(row['implementation']+'_result.json'),row)
                write(path,bad);link['performance_evidence']['sha256']=execution.sha(path)
                with self.assertRaises(ValueError):execution.performance_evidence(base,'S',link)
                if edit=='extra_attempt':(target/'retry_attempt.json').unlink()

    def test_prior_closed_identities_accept_reuse_only_after_original_native_guards(self):
        from experiments import weak_force_execution_20261029 as old
        from experiments.r_memory_execution_20261025 import LOGICAL_PYTHON
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder)
            for name,value in {'complete.json':{},'summary.json':{},'binding.json':dict(source_commit=execution.PARENT_COMMIT,runtime=dict(files=[]),files={}), 'prepare.json':{},'schedule.json':{}}.items():write(parent/name,value)
            receipts={}
            for i,phase in enumerate(old.PHASES):
                receipt=dict(source_commit=execution.PARENT_COMMIT,phase=phase,status='COMPLETED',cleanup_verified=True,
                    orphans_cleaned=False,actual_worker_state='GONE',job_active_processes=0,exit_code=0,worker_exit_code=0,launcher_exit_code=0,
                    source_before={'synthetic':'a'*64},source_after={'synthetic':'a'*64},
                    worker=dict(pid=100+i,creation_filetime_utc=1000+i,logical_executable=str(LOGICAL_PYTHON)))
                write(parent/'native'/phase/'receipt.json',receipt);receipts[phase]=receipt
            with patch.multiple(execution,PARENT=parent,PARENT_COMPLETE_SHA=execution.sha(parent/'complete.json'),PARENT_SUMMARY_SHA=execution.sha(parent/'summary.json')), \
                    patch.object(old,'completed_phase',return_value=({},{})) as completed:
                for state in ('GONE','REUSED_PID'):
                    _,proof=execution.parent_proof({},state=lambda identity:state)
                    self.assertEqual(len(proof['six_parent_receipt_hashes']),6)
                    self.assertTrue(all(r['current_identity_state']==state for r in proof['current_parent_worker_identities'].values()))
                self.assertEqual(completed.call_count,12)
                for state in ('RUNNING','UNKNOWN','ACCESS_DENIED'):
                    with self.assertRaises(ValueError):execution.parent_proof({},state=lambda identity:state)
                with self.assertRaises(OSError):execution.parent_proof({},state=Mock(side_effect=OSError('synthetic query failure')))
                first=old.PHASES[0];path=parent/'native'/first/'receipt.json'
                for key,value in [('cleanup_verified',False),('job_active_processes',1),('worker_exit_code',1),('source_after',{'synthetic':'b'*64})]:
                    bad=dict(receipts[first]);bad[key]=value;write(path,bad)
                    with self.assertRaises(ValueError):execution.parent_proof({},state=lambda identity:'REUSED_PID')
                write(path,receipts[first]);write(parent/'active.lock',{})
                with self.assertRaises(ValueError):execution.parent_proof({},state=lambda identity:'REUSED_PID')

    def test_phase_prefix_binds_QA_projection_but_not_append_only_live_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            for name in ('binding.json','schedule.json','prepare.json','qualification/summary.json','qa/optimizer_projection.json'):write(base/name,{})
            failure=dict(variant='P',status='NUMERICAL_UNRESOLVED',failure_is_physical_instability=False,quarantine={'synthetic':True})
            write(base/'qa/summary.json',dict(models=[failure]))
            ledger_write(base,[call('qa','qa_P_continuous',1,'pending')[0]])
            with patch.object(execution,'completed_phase',return_value=({},{})):
                files=execution.phase_inputs(base,'a'*40,'study')
                self.assertNotIn(str(base/'optimizer_calls.jsonl'),files)
                self.assertIn(str((base/'qa/optimizer_projection.json').resolve()),files)
                failure.pop('quarantine');write(base/'qa/summary.json',dict(models=[failure]))
                with self.assertRaises(ValueError):execution.phase_inputs(base,'a'*40,'study')

    def test_registered_loop_runs_five_phases_once_and_never_freezes_existing_binding(self):
        self.assertEqual(execution.PHASES,('prepare','qualify','qa','study','summary'))
        from experiments import uncapped_windows_job_20261021 as job
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);write(base/'binding.json',{})
            with patch.object(execution,'BASE',base),patch.object(execution,'require_release'), \
                    patch.object(execution,'launch') as launch,patch.object(execution,'freeze') as freeze,patch.object(job,'supervise') as native:
                execution.run(base,'a'*40)
            self.assertEqual([c.args[2] for c in launch.call_args_list],list(execution.PHASES))
            freeze.assert_not_called();native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
