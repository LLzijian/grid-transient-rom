"""Unexecuted pure guards for one exact prelaunch engineering recovery.

All locks, recovery records, ownership records and call counts are synthetic.
No historical private file, process API, Job, model, event or optimizer is used.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import memory_transport_execution_20261029 as execution


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RetainedTrainingProof(unittest.TestCase):
    @contextmanager
    def case(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            lock=root/execution.RETAINED_TRAINING_LOCK
            recovery_path=root/execution.RETAINED_TRAINING_RECOVERY
            identities=dict(supervisor=(101,10001),worker=(102,10002))
            lock_record={name:dict(pid=value[0],creation_filetime_utc=value[1]) for name,value in identities.items()}
            write(lock,lock_record)
            recovery=dict(status='INTERRUPTED_RUNTIME',lock_preserved=True,lock_sha256=sha(lock),
                **{name:dict(pid=value[0],creation_filetime_utc=value[1],state='GONE') for name,value in identities.items()})
            write(recovery_path,recovery)
            with patch.multiple(execution,ROOT=root,RETAINED_TRAINING_LOCK_SHA=sha(lock),
                    RETAINED_TRAINING_RECOVERY_SHA=sha(recovery_path),RETAINED_TRAINING_IDENTITIES=identities):
                yield root,lock,recovery_path,lock_record,recovery

    def test_real_identity_state_is_retained_without_relabeling_or_process_action(self):
        self.assertEqual(execution.RETAINED_TRAINING_LOCK_SHA,'f9d72f0d5fccd8b5349907d69299aad543aa9b936f3e016971a7e24acd57c3e4')
        self.assertEqual(execution.RETAINED_TRAINING_RECOVERY_SHA,'e53d9b82c8a19d5a3899cbd04682a298f16874bce27215e18d2b297e654e709a')
        with self.case() as (root,lock,recovery_path,record,recovery):
            before={p:p.read_bytes() for p in (lock,recovery_path)}
            for states in (dict(supervisor='GONE',worker='GONE'),dict(supervisor='GONE',worker='REUSED_PID'),
                    dict(supervisor='REUSED_PID',worker='GONE')):
                by_pid={record[name]['pid']:state for name,state in states.items()}
                query=Mock(side_effect=lambda identity:by_pid[identity['pid']])
                proof=execution.retained_training_lock_proof(lock,root=root,state=query)
                self.assertEqual(proof['current_owner_states'],states)
                self.assertEqual(proof['historical_owner_states'],dict(supervisor='GONE',worker='GONE'))
                self.assertEqual(proof['recorded_owner_identities'],record)
                self.assertEqual(proof['lock_sha256'],sha(lock))
                self.assertEqual(proof['recovery_sha256'],sha(recovery_path))
                self.assertEqual(query.call_count,2)
                self.assertFalse(proof['reused_process_touched'])
                self.assertFalse(proof['measured_old_job_accounting_claimed'])
                self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))

    def test_only_GONE_or_REUSED_PID_are_accepted_and_query_errors_propagate(self):
        with self.case() as (root,lock,*_):
            for state in ('RUNNING','UNKNOWN','ACCESS_DENIED','ALIVE','LIVE',None):
                with self.subTest(state=state),self.assertRaises(ValueError):
                    execution.retained_training_lock_proof(lock,root=root,state=lambda identity:state)
            with self.assertRaises(OSError):
                execution.retained_training_lock_proof(lock,root=root,state=Mock(side_effect=OSError('synthetic query failure')))

    def test_exact_path_and_both_raw_hashes_are_required_before_query(self):
        with self.case() as (root,lock,recovery_path,record,recovery):
            alternate=lock.parent/'other.lock';write(alternate,record)
            query=Mock(return_value='GONE')
            with self.assertRaises(ValueError):execution.retained_training_lock_proof(alternate,root=root,state=query)
            query.assert_not_called()
            for target,value in ((lock,dict(record,extra='changed')),(recovery_path,dict(recovery,extra='changed'))):
                original=target.read_bytes();write(target,value)
                with self.assertRaises(ValueError):execution.retained_training_lock_proof(lock,root=root,state=query)
                target.write_bytes(original)
            query.assert_not_called()

    def test_recovery_semantics_cannot_be_rescued_by_an_updated_expected_hash(self):
        with self.case() as (root,lock,recovery_path,record,recovery):
            edits=[('status','COMPLETED'),('lock_preserved',False),('lock_sha256','0'*64)]
            for key,value in edits:
                bad=copy.deepcopy(recovery);bad[key]=value;write(recovery_path,bad)
                with patch.object(execution,'RETAINED_TRAINING_RECOVERY_SHA',sha(recovery_path)),self.assertRaises(ValueError):
                    execution.retained_training_lock_proof(lock,root=root,state=lambda identity:'GONE')
            for owner in ('supervisor','worker'):
                for key,value in (('state','REUSED_PID'),('pid',999),('creation_filetime_utc',999)):
                    bad=copy.deepcopy(recovery);bad[owner][key]=value;write(recovery_path,bad)
                    with patch.object(execution,'RETAINED_TRAINING_RECOVERY_SHA',sha(recovery_path)),self.assertRaises(ValueError):
                        execution.retained_training_lock_proof(lock,root=root,state=lambda identity:'GONE')

    def test_lock_typed_creation_identity_cannot_be_rescued_by_new_hashes(self):
        with self.case() as (root,lock,recovery_path,record,recovery):
            for owner in ('supervisor','worker'):
                for key,value in (('pid','101'),('pid',True),('pid',-1),('pid',999),
                        ('creation_filetime_utc','10001'),('creation_filetime_utc',0),('creation_filetime_utc',999)):
                    bad=copy.deepcopy(record);bad[owner][key]=value;write(lock,bad)
                    changed=copy.deepcopy(recovery);changed['lock_sha256']=sha(lock);write(recovery_path,changed)
                    with patch.multiple(execution,RETAINED_TRAINING_LOCK_SHA=sha(lock),RETAINED_TRAINING_RECOVERY_SHA=sha(recovery_path)),self.assertRaises(ValueError):
                        execution.retained_training_lock_proof(lock,root=root,state=lambda identity:'GONE')


class RecoveryAuthorityAndPreservation(unittest.TestCase):
    @contextmanager
    def original_case(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=root/'study01';retry=root/'study01_prelaunch_retry1'
            binding=dict(base=str(old),source_commit=execution.OLD_SOURCE_COMMIT,optimizer_call_upper_bound=1848)
            write(old/'binding.json',binding)
            scope={key:0 for key in ('performance_attempts','actual_source_eigh_calls','NNLS_calls','real_optimizer_attempts','real_optimizer_completed','formal_training_runs','new_FOM_calls')}
            scope.update({key:False for key in ('prepare_executed','qualify_executed','original_val_read','original_test_read')})
            attempt=dict(source_commit=execution.OLD_SOURCE_COMMIT,status='REJECTED_PRELAUNCH_LEGACY_IDENTITY',
                python_exit_code=1,shell_exit_code=1,execution_cost=dict(native_jobs_started=0),
                binding=dict(sha256=sha(old/'binding.json')),actual_scientific_scope=scope)
            failed=root/'failure.json';write(failed,attempt)
            with patch.multiple(execution,BASE=retry,OLD_BASE=old,OLD_BINDING_SHA=sha(old/'binding.json'),
                    FAILED_ATTEMPT_RECORD=failed,FAILED_ATTEMPT_SHA=sha(failed)):
                yield root,old,retry,failed,binding,attempt

    def test_old_binding_and_zero_work_failure_remain_byte_exact_and_budget_not_reset(self):
        from experiments import memory_transport_study_20261029 as study
        self.assertEqual(execution.BASE.name,'study01_prelaunch_retry1')
        self.assertEqual(execution.OLD_BASE.name,'study01')
        self.assertEqual(execution.RELEASE_SHA,'077ca8767b915242c09084cbddd8ab51e6d780acd278700514517c8f2342049a')
        self.assertEqual(execution.OLD_SOURCE_COMMIT,'c5707e4fb677eb0cb64ba378e59196c5b89b58db')
        with self.original_case() as (_,old,retry,failed,*_):
            before={path:path.read_bytes() for path in (old/'binding.json',failed)}
            proof=execution.original_prelaunch_evidence()
            self.assertTrue(proof['original_base_preserved'])
            self.assertEqual(proof['native_jobs_started'],0)
            self.assertEqual(proof['optimizer_calls_spent'],0)
            self.assertEqual(proof['same_scientific_optimizer_upper_bound'],1848)
            self.assertEqual(study.TOTAL,1848)
            self.assertEqual(study.LIMITS,dict(formal=1536,qa=108,repair=108,replay=96))
            self.assertTrue(all(path.read_bytes()==data for path,data in before.items()))
            self.assertFalse(retry.exists())

    def test_old_evidence_raw_hash_mismatch_and_new_work_are_hard_refusals(self):
        with self.original_case() as (_,old,retry,failed,binding,attempt):
            for target,value in ((old/'binding.json',dict(binding,extra=True)),(failed,dict(attempt,extra=True))):
                before=target.read_bytes();write(target,value)
                with self.assertRaises(ValueError):execution.original_prelaunch_evidence()
                target.write_bytes(before)
            for name in ('active.lock','optimizer_calls.jsonl','ownership','native','prepare.json','schedule.json','qualification','qa','study','summary.json','complete.json'):
                target=old/name;target.write_text('synthetic',encoding='utf-8')
                with self.subTest(name=name),self.assertRaises(ValueError):execution.original_prelaunch_evidence()
                target.unlink()

    def test_updated_failure_hash_cannot_hide_spent_calls_or_a_changed_original_source(self):
        with self.original_case() as (_,old,retry,failed,binding,attempt):
            for field in ('performance_attempts','actual_source_eigh_calls','NNLS_calls','real_optimizer_attempts','real_optimizer_completed','formal_training_runs','new_FOM_calls'):
                bad=copy.deepcopy(attempt);bad['actual_scientific_scope'][field]=1;write(failed,bad)
                with patch.object(execution,'FAILED_ATTEMPT_SHA',sha(failed)),self.assertRaises(ValueError):execution.original_prelaunch_evidence()
            for field in ('prepare_executed','qualify_executed','original_val_read','original_test_read'):
                bad=copy.deepcopy(attempt);bad['actual_scientific_scope'][field]=True;write(failed,bad)
                with patch.object(execution,'FAILED_ATTEMPT_SHA',sha(failed)),self.assertRaises(ValueError):execution.original_prelaunch_evidence()
            bad=copy.deepcopy(attempt);bad['source_commit']='0'*40;write(failed,bad)
            with patch.object(execution,'FAILED_ATTEMPT_SHA',sha(failed)),self.assertRaises(ValueError):execution.original_prelaunch_evidence()

    def recovery_record(self):
        return dict(root_recovery_authorized=True,independent_recovery_review_passed=True,old_base_preserved=True,one_prelaunch_retry=True,
            original_execution_release_sha256=execution.RELEASE_SHA,old_source_commit=execution.OLD_SOURCE_COMMIT,
            old_binding_sha256=execution.OLD_BINDING_SHA,failed_attempt_sha256=execution.FAILED_ATTEMPT_SHA,
            retry_base='study01_prelaunch_retry1',optimizer_upper_bound=1848,optimizer_calls_spent=0,
            source_plane_previously_consumed=1,source_eigh_calls_this_loop=0,automatic_retry=False,evidence={})

    def test_UNKNOWN_recovery_authority_refuses_before_recovery_source_reads(self):
        with patch.object(execution,'RECOVERY_SHA','UNKNOWN'),patch.object(execution,'sha') as digest,patch.object(execution,'read') as read:
            with self.assertRaisesRegex(ValueError,'UNKNOWN'):execution.recovery_gate()
            digest.assert_not_called();read.assert_not_called()
        from experiments import uncapped_windows_job_20261021 as job
        original=dict(parent_remote_archival_verified=True,independent_implementation_review_passed=True,
            root_execution_authorized=True,parent_complete_sha256=execution.PARENT_COMPLETE_SHA,optimizer_upper_bound=1848)
        with patch.multiple(execution,RELEASE_SHA='a'*64,RECOVERY_SHA='UNKNOWN'), \
                patch.object(execution,'sha',return_value='a'*64),patch.object(execution,'read',return_value=original), \
                patch.object(execution,'source_closure') as sources,patch.object(job,'supervise') as native:
            for action in (execution.require_release,lambda:execution.freeze('a'*40,execution.BASE),
                    lambda:execution.launch(execution.BASE,'a'*40,'prepare'),lambda:execution.run(execution.BASE,'a'*40)):
                with self.assertRaisesRegex(ValueError,'UNKNOWN'):action()
            sources.assert_not_called();native.assert_not_called()

    def test_recovery_needs_independent_root_authority_exact_scope_and_preserved_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'release.json';value=self.recovery_record();write(path,value)
            with patch.multiple(execution,RECOVERY_RECORD=path,RECOVERY_DOCUMENTS={}),patch.object(execution,'original_prelaunch_evidence') as original:
                with patch.object(execution,'RECOVERY_SHA',sha(path)):
                    self.assertEqual(execution.recovery_gate(),value)
                    original.assert_called_once_with()
                edits=[(key,False) for key in ('root_recovery_authorized','independent_recovery_review_passed','old_base_preserved','one_prelaunch_retry')]
                edits += [('optimizer_upper_bound',3696),('optimizer_calls_spent',1),('source_eigh_calls_this_loop',1),
                    ('source_plane_previously_consumed',2),('automatic_retry',True),('retry_base','study01'),
                    ('old_binding_sha256','0'*64),('original_execution_release_sha256','0'*64)]
                for key,bad_value in edits:
                    bad=dict(value);bad[key]=bad_value;write(path,bad);original.reset_mock()
                    with patch.object(execution,'RECOVERY_SHA',sha(path)),self.assertRaises(ValueError):execution.recovery_gate()
                    original.assert_not_called()

    def test_recovery_design_review_evidence_is_hash_bound_and_origin_failure_propagates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=root/'release.json';document=root/'review.json';write(document,dict(independent=True))
            value=self.recovery_record();value['evidence']={'review.json':sha(document)};write(record,value)
            with patch.multiple(execution,ROOT=root,RECOVERY_RECORD=record,RECOVERY_SHA=sha(record),RECOVERY_DOCUMENTS={'review.json':sha(document)}), \
                    patch.object(execution,'original_prelaunch_evidence',side_effect=ValueError('old evidence changed')):
                with self.assertRaisesRegex(ValueError,'old evidence changed'):execution.recovery_gate()
            write(document,dict(independent=False))
            with patch.multiple(execution,ROOT=root,RECOVERY_RECORD=record,RECOVERY_SHA=sha(record),RECOVERY_DOCUMENTS={'review.json':value['evidence']['review.json']}), \
                    patch.object(execution,'original_prelaunch_evidence') as original:
                with self.assertRaises(ValueError):execution.recovery_gate()
                original.assert_not_called()

    def test_old_base_is_rejected_and_retry_freeze_never_overwrites_an_existing_directory(self):
        with self.original_case() as (_,old,retry,failed,*_):
            before=(old/'binding.json').read_bytes()
            with self.assertRaises(ValueError):execution.checked_base(old)
            self.assertEqual(execution.checked_base(retry),retry.resolve())
            retry.mkdir()
            with patch.object(execution,'require_release'),patch.object(execution,'check_locks') as locks:
                with self.assertRaises(FileExistsError):execution.freeze('a'*40,retry)
                locks.assert_not_called()
            self.assertEqual((old/'binding.json').read_bytes(),before)


class FullLockScope(unittest.TestCase):
    @contextmanager
    def case(self):
        from experiments import weak_force_execution_20261029 as weak
        from experiments import state_gated_memory_execution_20261028 as sgm
        from experiments import force_graph_execution_20261028 as fg
        # Reuse only the synthetic fixture factory, never a test method.
        with RetainedTrainingProof.case(self) as (root,training,training_recovery,record,recovery):
            base=root/'study01_prelaunch_retry1';old_base=root/'failed_study01'
            parents=[root/name for name in ('closed_weak','closed_sgm','closed_fg')]
            retained=root/fg.OLD/'active.lock'
            retained_record={name:dict(pid=identity[0],creation_filetime_utc=identity[1]) for name,identity in fg.OLD_IDENTITIES.items()}
            write(retained,retained_record);retained_bytes=retained.read_bytes()
            retained_recovery=root/'results/r_memory_20261025/recovery_20261028.json'
            retained_proof=dict(status='INTERRUPTED_RUNTIME',lock_preserved=True,lock_sha256=fg.OLD_LOCK_SHA,
                owners={name:dict(pid_decimal=str(identity[0]),creation_filetime_utc_decimal=str(identity[1]),identity_state='GONE') for name,identity in fg.OLD_IDENTITIES.items()})
            write(retained_recovery,retained_proof)
            # Only the synthetic FG lock bytes map to its fixed public digest.
            # Old modules/constants stay unchanged. The new exact recovery hash
            # uses its synthetic fixture digest, never a real private record.
            def digest(path):
                path=Path(path)
                if path==retained and path.read_bytes()==retained_bytes:return fg.OLD_LOCK_SHA
                return sha(path)
            with patch.multiple(execution,BASE=base,OLD_BASE=old_base,PARENT=parents[0]), \
                    patch.object(execution,'RETAINED_R_MEMORY_RECOVERY_SHA',sha(retained_recovery)), \
                    patch.object(execution,'sha',side_effect=digest), \
                    patch.object(weak,'checked_base',return_value=parents[0]) as weak_base, \
                    patch.object(sgm,'checked_base',return_value=parents[1]) as sgm_base, \
                    patch.object(fg,'checked_base',return_value=parents[2]) as fg_base:
                yield dict(root=root,base=base,old_base=old_base,training=training,training_record=record,parents=parents,
                    retained=retained,retained_record=retained_record,retained_recovery=retained_recovery,retained_proof=retained_proof,
                    validators=(weak_base,sgm_base,fg_base))

    def test_full_scope_retains_exact_exception_and_all_parent_base_delegations(self):
        self.assertEqual(execution.RETAINED_R_MEMORY_RECOVERY_SHA,'6ebbe0897f23530709f8acf67899414285ad3c865f05f1f6da0e92b9eb7865f2')
        with self.case() as c:
            training_worker=c['training_record']['worker']['pid']
            query=Mock(side_effect=lambda identity:'REUSED_PID' if identity['pid']==training_worker else 'GONE')
            proof=execution.check_locks(c['base'],state=query)
            self.assertEqual(proof['checked_parent_bases'],[str(p) for p in c['parents']])
            self.assertEqual(proof['retained_training_lock']['current_owner_states']['worker'],'REUSED_PID')
            self.assertTrue(all(r['state']=='GONE' for r in proof['retained_r_memory_lock']['owners'].values()))
            self.assertIsNone(proof['retained_r_memory_lock']['old_JobCPU'])
            self.assertFalse(proof['new_active_lock_owned'])
            for validator in c['validators']:validator.assert_called_once()

    def test_unrelated_P1_locks_parent_current_locks_and_old_failed_lock_are_refused(self):
        with self.case() as c:
            p1=c['root']/'private_data/closure_mechanism_20261019/P1'
            paths=[p1/'unrelated/active.lock',p1/'rational_execution01/native_active.lock',
                *(parent/'active.lock' for parent in c['parents']),c['old_base']/'active.lock']
            for path in paths:
                write(path,dict(synthetic=True))
                with self.subTest(path=path.name),self.assertRaises(ValueError):
                    execution.check_locks(c['base'],state=lambda identity:'GONE')
                path.unlink()

    def test_parent_base_validator_refusal_is_not_bypassed(self):
        with self.case() as c:
            for validator in c['validators']:
                validator.side_effect=ValueError('synthetic parent base rejection')
                with self.assertRaisesRegex(ValueError,'parent base rejection'):
                    execution.check_locks(c['base'],state=lambda identity:'GONE')
                validator.side_effect=None

    def test_second_exact_lock_keeps_true_absent_states_and_refuses_every_other_state(self):
        with self.case() as c:
            supervisor=c['retained_record']['supervisor']['pid']
            expected=dict(supervisor='REUSED_PID',worker='GONE')
            proof=execution.check_locks(c['base'],state=lambda identity:'REUSED_PID' if identity['pid']==supervisor else 'GONE')['retained_r_memory_lock']
            self.assertEqual(proof['current_owner_states'],expected)
            self.assertEqual(proof['historical_owner_states'],dict(supervisor='GONE',worker='GONE'))
            self.assertEqual(proof['recorded_owner_identities'],c['retained_record'])
            self.assertEqual(proof['owners']['supervisor']['state'],'REUSED_PID')
            self.assertEqual(proof['owners']['supervisor']['historical_state'],'GONE')
            self.assertFalse(proof['reused_process_touched']);self.assertFalse(proof['measured_old_job_accounting_claimed'])
            for state in ('RUNNING','UNKNOWN','ACCESS_DENIED','LIVE',None):
                with self.subTest(state=state),self.assertRaises(ValueError):
                    execution.check_locks(c['base'],state=lambda identity:state if identity['pid']==supervisor else 'GONE')
            def query_error(identity):
                if identity['pid']==supervisor:raise OSError('synthetic second identity error')
                return 'GONE'
            with self.assertRaises(OSError):execution.check_locks(c['base'],state=query_error)
            saved=c['retained'].read_bytes();write(c['retained'],dict(synthetic_tampering=True))
            with self.assertRaises(ValueError):execution.check_locks(c['base'],state=lambda identity:'GONE')
            c['retained'].write_bytes(saved)
            recovery_bytes=c['retained_recovery'].read_bytes();write(c['retained_recovery'],dict(c['retained_proof'],extra='changed'))
            with self.assertRaises(ValueError):execution.check_locks(c['base'],state=lambda identity:'GONE')
            c['retained_recovery'].write_bytes(recovery_bytes)
            for field,value in (('status','COMPLETED'),('lock_preserved',False),('lock_sha256','0'*64)):
                bad=copy.deepcopy(c['retained_proof']);bad[field]=value;write(c['retained_recovery'],bad)
                with patch.object(execution,'RETAINED_R_MEMORY_RECOVERY_SHA',sha(c['retained_recovery'])),self.assertRaises(ValueError):
                    execution.check_locks(c['base'],state=lambda identity:'GONE')
            for owner in ('supervisor','worker'):
                for field,value in (('identity_state','REUSED_PID'),('pid_decimal','wrong'),('creation_filetime_utc_decimal','wrong')):
                    bad=copy.deepcopy(c['retained_proof']);bad['owners'][owner][field]=value;write(c['retained_recovery'],bad)
                    with patch.object(execution,'RETAINED_R_MEMORY_RECOVERY_SHA',sha(c['retained_recovery'])),self.assertRaises(ValueError):
                        execution.check_locks(c['base'],state=lambda identity:'GONE')
            c['retained_recovery'].write_bytes(recovery_bytes)
            for owner in ('supervisor','worker'):
                for field,value in (('pid','invalid'),('creation_filetime_utc',0)):
                    bad=copy.deepcopy(c['retained_record']);bad[owner][field]=value;write(c['retained'],bad)
                    with self.assertRaises(ValueError):execution.check_locks(c['base'],state=lambda identity:'GONE')
            c['retained'].write_bytes(saved)

    def test_missing_exact_training_lock_cannot_be_replaced_by_a_blanket_exception(self):
        with self.case() as c:
            c['training'].unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):
                execution.check_locks(c['base'],state=lambda identity:'GONE')

    def test_retry_own_lock_needs_exact_typed_claim_and_owned_worker_needs_a_lock(self):
        with self.case() as c:
            owner=dict(pid=301,creation_filetime_utc=30001);path=c['base']/'active.lock'
            with self.assertRaises(ValueError):execution.check_locks(c['base'],owned_identity=owner,state=lambda identity:'GONE')
            write(path,dict(worker=owner))
            with self.assertRaises(ValueError):execution.check_locks(c['base'],state=lambda identity:'GONE')
            self.assertTrue(execution.check_locks(c['base'],owned_identity=owner,state=lambda identity:'GONE')['new_active_lock_owned'])
            for key,value in (('pid',302),('creation_filetime_utc',30002)):
                wrong=dict(owner);wrong[key]=value
                with self.assertRaises(ValueError):execution.check_locks(c['base'],owned_identity=wrong,state=lambda identity:'GONE')
            write(path,dict(worker=dict(pid='301',creation_filetime_utc=30001)))
            with self.assertRaises(ValueError):execution.check_locks(c['base'],owned_identity=owner,state=lambda identity:'GONE')


class SingleRetryLoop(unittest.TestCase):
    def test_one_mock_retry_uses_existing_binding_and_exact_five_phases_without_new_budget(self):
        from experiments import uncapped_windows_job_20261021 as job
        from experiments import memory_transport_study_20261029 as study
        with RecoveryAuthorityAndPreservation.original_case(self) as (_,old,retry,failed,*_):
            before={path:path.read_bytes() for path in (old/'binding.json',failed)}
            write(retry/'binding.json',dict(optimizer_call_upper_bound=1848,source_plane_previously_consumed=1,source_eigh_calls_this_loop=0))
            with patch.object(execution,'require_release'),patch.object(execution,'launch') as launch, \
                    patch.object(execution,'freeze') as freeze,patch.object(job,'supervise') as native, \
                    patch.object(study,'source_identity') as source:
                execution.run(retry,'a'*40)
            self.assertEqual([item.args[2] for item in launch.call_args_list],['prepare','qualify','qa','study','summary'])
            self.assertTrue(all(item.args[0]==retry for item in launch.call_args_list))
            self.assertEqual(study.counts(retry)['attempted_total'],0)
            self.assertEqual(study.TOTAL,1848)
            freeze.assert_not_called();native.assert_not_called();source.assert_not_called()
            self.assertTrue(all(path.read_bytes()==data for path,data in before.items()))
            with patch.object(execution,'require_release'),patch.object(execution,'launch') as launch:
                with self.assertRaises(ValueError):execution.run(old,'a'*40)
                launch.assert_not_called()


if __name__=='__main__':
    unittest.main()
