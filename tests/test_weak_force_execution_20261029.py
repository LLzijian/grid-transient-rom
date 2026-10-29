"""Pure execution admission tests; no native launch or real payload access."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import weak_force_execution_20261029 as execution


class ExecutionAdmission(unittest.TestCase):
    def test_unknown_release_refuses_before_files_jobs_or_phases(self):
        with patch.object(execution,'RELEASE_SHA','UNKNOWN'),patch.object(execution,'sha') as digest:
            for action in (lambda:execution.require_release(),
                           lambda:execution.freeze('a'*40),
                           lambda:execution.launch(execution.BASE,'a'*40,'weak'),
                           lambda:execution.run(execution.BASE,'a'*40)):
                with self.assertRaisesRegex(ValueError,'UNKNOWN'):action()
            digest.assert_not_called()

    def test_release_requires_actual_parent_archive_and_review(self):
        required=dict(parent_remote_archival_verified=True,independent_implementation_review_passed=True,
            root_execution_authorized=True,parent_complete_sha256=execution.PARENT_COMPLETE_SHA,optimizer_upper_bound=976)
        with patch.object(execution,'RELEASE_SHA','a'*64),patch.object(execution,'sha',return_value='a'*64):
            for key in ('parent_remote_archival_verified','independent_implementation_review_passed','root_execution_authorized'):
                value=dict(required);value[key]=False
                with patch.object(execution,'read',return_value=value):
                    with self.assertRaises(ValueError):execution.require_release()
            with patch.object(execution,'read',return_value=required):
                self.assertEqual(execution.require_release(),required)

    def test_only_registered_directory_and_weak_once_in_loop(self):
        self.assertEqual(execution.checked_base(execution.BASE),execution.BASE.resolve())
        with self.assertRaises(ValueError):execution.checked_base(execution.BASE/'replacement')
        self.assertEqual(execution.PHASES,('prepare','weak','qualify','qa','study','summary'))
        self.assertEqual(execution.PHASES.count('weak'),1)

    def qa_case(self,folder):
        base=Path(folder);(base/'qa').mkdir()
        rows=[]
        for variant in ('D','W'):
            for label in ('continuous','restored'):
                for step in range(1,19):
                    identity=dict(category='qa',run_id=f'qa_{variant}_{label}',step=step,call_id=f'{variant}_{label}_{step}')
                    rows.extend([dict(identity,event='optimizer_step_attempt'),dict(identity,event='optimizer_step_completed')])
        (base/'optimizer_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        projection=base/'qa/optimizer_projection.json';projection.write_text(json.dumps(rows),encoding='utf-8')
        result=dict(eligible_variants=['D','W'],models=[dict(variant=v,status='PASS',actual_optimizer_calls=36) for v in ('D','W')],
                    qa_journal_projection=dict(path=str(projection.resolve()),sha256=execution.sha(projection)))
        return base,result,rows

    def test_QA_projects_actual_unique_72_calls_and_allows_later_formal_growth(self):
        with tempfile.TemporaryDirectory() as folder:
            base,result,rows=self.qa_case(folder)
            self.assertTrue(execution.qa_evidence(base,result))
            identity=dict(category='formal',run_id='D_29311',step=1,call_id='formal_after_QA')
            with (base/'optimizer_calls.jsonl').open('a',encoding='utf-8') as stream:
                for event in ('optimizer_step_attempt','optimizer_step_completed'):
                    stream.write(json.dumps(dict(identity,event=event))+'\n')
            self.assertTrue(execution.qa_evidence(base,result))
            bad=copy.deepcopy(result);bad['models'][1]['variant']='D'
            with self.assertRaises(ValueError):execution.qa_evidence(base,bad)
            # Literal PASS and 36+36 cannot rescue an incomplete actual ledger.
            (base/'optimizer_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]),encoding='utf-8')
            with self.assertRaises(ValueError):execution.qa_evidence(base,result)

    def test_QA_rejects_projection_tampering_and_premature_formal(self):
        with tempfile.TemporaryDirectory() as folder:
            base,result,rows=self.qa_case(folder)
            path=base/'qa/optimizer_projection.json';path.write_text('[]',encoding='utf-8')
            result['qa_journal_projection']['sha256']=execution.sha(path)
            with self.assertRaises(ValueError):execution.qa_evidence(base,result)
            path.write_text(json.dumps(rows),encoding='utf-8');result['qa_journal_projection']['sha256']=execution.sha(path)
            identity=dict(category='formal',run_id='D_29311',step=1,call_id='premature')
            earlier=[dict(identity,event=e) for e in ('optimizer_step_attempt','optimizer_step_completed')]+rows
            (base/'optimizer_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in earlier),encoding='utf-8')
            with self.assertRaises(ValueError):execution.qa_evidence(base,result)

    def cross_case(self,folder):
        from experiments import weak_force_study_20261029 as study
        base=Path(folder);(base/'qualification').mkdir();ids=[f'fixture_{i}' for i in range(297)]
        (base/'prepare.json').write_text(json.dumps(dict(fit_ids=ids)),encoding='utf-8')
        pair=dict(finite_to_endpoint=True,los_pair_pass=True,coi_normalized_rmse=0.,common_normalized_rmse=0.)
        actual=dict(available=True,finite=True,numerically_admissible=True,coi_normalized_rmse=0.,common_normalized_rmse=0.)
        intervals={}
        for scope,count in (('8s',800),('actual15',1000)):
            interval=dict(qualified=True,actual_output_steps=count,full_R_N_fine_finite=True,full_A_fine_finite=True)
            for direction in ('R_N_to_A','A_to_R_N'):
                interval[direction+'_pair']=copy.deepcopy(pair)
                interval[direction+'_scored_numerics']=[dict(pair,available=True,horizon='prefix')]
                if scope=='actual15':interval[direction+'_actual_interval']=copy.deepcopy(actual)
            intervals[scope]=interval
        record=dict(qualified=True,record_count=297,event_ids=ids,
            original_schema_aliases=dict(R_N='frozen_A96',A='current_D0'),
            source_evidence_sha256=study.EVIDENCE_SHA['study/A_28311/fit000096/evidence.json'],
            source_A96_prediction_recomputed=False,scope='A96_D0_fit297_bidirectional_RK4_fine005_base01_8s_actual15',
            bidirectional=True,history_rows=11,initial_observation_index=10,fine_dt=.005,output_dt=.01,coarse_stride=2,
            rows=[dict(event_id=eid,actual_steps=1000,qualified=True,intervals=copy.deepcopy(intervals)) for eid in ids])
        path=base/'qualification/D_A96_compatibility.json'
        path.write_text(json.dumps(record),encoding='utf-8')
        result=dict(D_A96_compatibility_qualified=True,D_A96_compatibility=dict(path=str(path.resolve()),sha256=execution.sha(path)))
        return base,result,record,path

    def test_cross_requires_hash_identity_and_original_both_direction_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            base,result,record,path=self.cross_case(folder)
            self.assertTrue(execution.cross_model_evidence(base,result))
            with self.assertRaises(ValueError):execution.cross_model_evidence(base,{})
            bad=copy.deepcopy(record);bad['rows'][1]['event_id']=bad['rows'][0]['event_id']
            path.write_text(json.dumps(bad),encoding='utf-8');result['D_A96_compatibility']['sha256']=execution.sha(path)
            with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)
            for scope in ('8s','actual15'):
                for direction in ('R_N_to_A','A_to_R_N'):
                    bad=copy.deepcopy(record)
                    bad['rows'][0]['intervals'][scope][direction+'_pair']['coi_normalized_rmse']=.002
                    path.write_text(json.dumps(bad),encoding='utf-8');result['D_A96_compatibility']['sha256']=execution.sha(path)
                    with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)
            bad=copy.deepcopy(record);bad['rows'][0]['intervals']['actual15']['A_to_R_N_actual_interval']['finite']=False
            path.write_text(json.dumps(bad),encoding='utf-8');result['D_A96_compatibility']['sha256']=execution.sha(path)
            with self.assertRaises(ValueError):execution.cross_model_evidence(base,result)

    def test_original_CPU_qualification_is_hash_bound_and_cannot_admit_training(self):
        valid=dict(shared_checks_qualified=True,per_variant_qualified=dict(A=True),training_admission=False)
        with patch.object(execution,'sha',return_value=execution.FIXED_CPU_QA[1]),patch.object(execution,'read',return_value=valid):
            self.assertEqual(list(execution.fixed_cpu_qualification().values()),[execution.FIXED_CPU_QA[1]])
        with patch.object(execution,'sha',return_value='0'*64):
            with self.assertRaises(ValueError):execution.fixed_cpu_qualification()
        invalid=dict(valid,training_admission=True)
        with patch.object(execution,'sha',return_value=execution.FIXED_CPU_QA[1]),patch.object(execution,'read',return_value=invalid):
            with self.assertRaises(ValueError):execution.fixed_cpu_qualification()


if __name__=='__main__':unittest.main()
