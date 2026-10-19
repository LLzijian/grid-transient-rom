"""Synthetic new wiring only: no original scientific QA, FOM or weights."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import first_physics_pair_20261020 as w

class WiringContracts(unittest.TestCase):
    def test_fixed_path_refusal(self):
        with self.assertRaises(ValueError): w.inside(w.OLD_BASE/'campaign01/train')
        with self.assertRaises(ValueError): w.inside(w.BASE/'../active_fom_pair01/train')
        self.assertEqual(w.inside(w.CAMPAIGN/'train'),(w.CAMPAIGN/'train').resolve())

    def test_old_startup_history_strict(self):
        candidates=[{'id':str(i)} for i in range(48)]
        record={'qualified_count':0,'attempts':[{'condition':c,'steps':{s:{'status':'failed','error':"ModuleNotFoundError: No module named 'andes'",'diagnostics':{}} for s in ('0.005','0.0025')}} for c in candidates]}
        self.assertEqual(w.verify_startup_history({'candidates':candidates},record),96)
        changed=copy.deepcopy(record); changed['attempts'][0]['steps']['0.005']['raw_file']='anything.npz'
        with self.assertRaises(ValueError): w.verify_startup_history({'candidates':candidates},changed)

    def test_fit_campaign_restored(self):
        fit=SimpleNamespace(CAMPAIGN=w.OLD_BASE)
        with self.assertRaises(RuntimeError):
            with w.fit_campaign(fit):
                self.assertEqual(fit.CAMPAIGN,w.BASE.resolve()); raise RuntimeError('synthetic callback failure')
        self.assertEqual(fit.CAMPAIGN,w.OLD_BASE)
        with self.assertRaises(ValueError):
            with w.fit_campaign(SimpleNamespace(CAMPAIGN=w.ROOT)): pass

    def test_batch_environment_bypasses_old_exception_catch(self):
        from experiments import closure_short_history_20261019 as short
        calls=[]
        def fail(*args,**kwargs):
            calls.append(args[0]['id']); return {'status':'failed','error':"ModuleNotFoundError: No module named 'andes'"}
        records=[]
        with patch.object(short,'run_short_history',side_effect=fail):
            original=short.run_short_history
            with self.assertRaises(w.BatchEnvironmentStop):
                with w.batch_guard('short',lambda:None,records):
                    for i in range(48):
                        try: short.run_short_history({'id':str(i)},'synthetic',step=.005)
                        except Exception: pass  # Exact old loop boundary must not swallow sentinel.
            self.assertIs(short.run_short_history,original)
        self.assertEqual(calls,['0']); self.assertEqual(records[0]['status'],'environment_failure')

    def test_numerical_failure_remains_condition_local(self):
        from experiments import closure_short_history_20261019 as short
        calls=[]
        def fail(*args,**kwargs):
            calls.append(args[0]['id']); return {'status':'failed','error':'ValueError: numerical_TDS_failure'}
        with patch.object(short,'run_short_history',side_effect=fail):
            with w.batch_guard('short',lambda:None,[]):
                for i in range(3): short.run_short_history({'id':str(i)},'synthetic',step=.005)
        self.assertEqual(len(calls),3)

    def test_acquire_preserves_current_failure_and_remaining47(self):
        from experiments import active_fom_pair_20261020 as old
        from experiments import closure_short_history_20261019 as short
        from experiments import fom_execution_environment_20261020 as env
        candidates=[{'id':str(i)} for i in range(48)]
        calls=[]
        def original_condition(*args,**kwargs):
            calls.append(args[0]['id'])
            return {'status':'failed','error':"ModuleNotFoundError: No module named 'andes'"}
        def old_loop():
            for c in candidates:
                try: short.run_short_history(c,'synthetic',step=.005)
                except Exception: pass
            self.fail('old loop incorrectly swallowed batch stop')
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); campaign=base/'campaign01'; worker=base/'wrapper'
            campaign.mkdir(); worker.mkdir()
            (campaign/'design.json').write_text(json.dumps({'candidates':candidates}),encoding='utf-8')
            (campaign/'source_manifest.json').write_text(json.dumps({'source':{'synthetic.py':'fixture'}}),encoding='utf-8')
            args=SimpleNamespace(campaign=campaign,source_manifest=campaign/'source_manifest.json',
                                 output=base/'science_output',stage='short',source_commit='a'*40,max_seconds=100.)
            with patch.object(w,'BASE',base),patch.object(w,'CAMPAIGN',campaign),patch.object(w,'expected_environment',return_value={'fixture':'known'}),patch.object(env,'preflight',return_value={'fixture':'known'}),patch.object(w,'verify_source_files'),patch.object(old,'main',side_effect=old_loop),patch.object(short,'run_short_history',side_effect=original_condition):
                with self.assertRaises(w.BatchEnvironmentStop): w.acquire(args,worker)
            record=json.loads((worker/'batch_stop.json').read_text(encoding='utf-8'))
            self.assertEqual(record['denominator'],48)
            self.assertEqual(calls,['0'])
            self.assertEqual(record['attempts'][0]['condition']['id'],'0')
            self.assertEqual(record['attempts'][0]['status'],'environment_failure')
            self.assertEqual(len(record['not_attempted']),47)
            self.assertEqual([r['condition']['id'] for r in record['not_attempted']],[str(i) for i in range(1,48)])

    def test_preflight_guard_before_callback(self):
        from experiments import closure_short_history_20261019 as short
        records=[]
        with patch.object(short,'run_short_history') as callback:
            with self.assertRaises(w.BatchEnvironmentStop):
                with w.batch_guard('short',lambda:(_ for _ in ()).throw(ValueError('case identity changed')),records):
                    short.run_short_history({'id':'0'},'synthetic',step=.005)
            callback.assert_not_called()

    def test_expected_environment_uses_actual_accepted_qa(self):
        expected=w.expected_environment('gpu')
        from experiments.fom_execution_environment_20261020 import validate_identity
        validate_identity(expected,expected,'gpu')
        with patch.object(w,'read',return_value={'status':'FAIL'}):
            with self.assertRaises(ValueError): w.expected_environment('gpu')

    def test_train_val_events_path_gate(self):
        with patch.object(w,'read',return_value={'split':'train','old_train':True,'new_directories':[str(w.OLD_BASE/'campaign01/train')]}):
            with self.assertRaises(ValueError): w.validate_events_spec(w.BASE/'inputs/spec.json')
        with patch.object(w,'read',return_value={'split':'test','new_directories':[]}):
            with self.assertRaises(ValueError): w.validate_events_spec(w.BASE/'inputs/spec.json')

    def launcher(self):
        path=w.BASE/'launch_stage.py'; spec=importlib.util.spec_from_file_location('first_physics_launcher_qa',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_launcher_entry_and_worker_gate(self):
        launch=self.launcher(); commit='a'*40
        a=SimpleNamespace(category='qa',command=w.BASE/'commands/synthetic.json',prior=w.BASE/'inputs/synthetic_prior.json',source_commit=commit,
                          worker=w.BASE/'stages/synthetic_wrapper/wrapper_worker.json',cap=180.)
        names={Path(launch.__file__).relative_to(ROOT).as_posix(),a.command.relative_to(ROOT).as_posix(),a.prior.relative_to(ROOT).as_posix(),
            'experiments/native_process_receipt_20261020.ps1','experiments/fixed_edge_pair_20261019.py','experiments/linear_cache_cost_gate_20261019.py',
            'experiments/first_physics_pair_20261020.py','tests/test_first_physics_pair_20261020.py','experiments/fom_execution_environment_20261020.py',
            'experiments/active_fom_pair_20261020.py','experiments/active_fom_pair_fit_20261020.py','experiments/closure_discovery_data.py',
            'configs/private_dataset.json','reports/first_physics_pair_20261020_protocol.md','reports/astra6_first_physics_20261020.md'}
        args=['-m','experiments.first_physics_pair_20261020','qa','--source-commit',commit,'--wrapper-output',str(a.worker.parent),'--output',str(w.BASE/'stages/synthetic_out'),'--max-seconds','120']
        launch.command_gate(a,{'arguments':args},{'files':dict.fromkeys(names,'synthetic')})
        changed=args.copy(); changed[1]='experiments.active_fom_pair_20261020'
        with self.assertRaises(ValueError): launch.command_gate(a,{'arguments':changed},{'files':dict.fromkeys(names,'synthetic')})
        changed=args.copy(); changed[changed.index('--wrapper-output')+1]=str(w.OLD_BASE/'stages/wrong')
        with self.assertRaises(ValueError): launch.command_gate(a,{'arguments':changed},{'files':dict.fromkeys(names,'synthetic')})

    def test_launcher_qa_budget_refusal(self):
        launch=self.launcher()
        record={'category':'qa','status':'COMPLETED','outer':{'seconds':1.},'worker':{'argv':[]}}
        with patch.object(Path,'read_text',side_effect=[json.dumps([str(w.BASE/'native_synthetic/execution.json')]),json.dumps(record)]):
            a=SimpleNamespace(prior=w.BASE/'inputs/prior.json',category='qa',cap=100.)
            with self.assertRaises(ValueError): launch.prior_cost_gate(a,{'arguments':['qa']})

    def test_launcher_model_cumulative_cap(self):
        launch=self.launcher()
        args=['fit','adapt','--seed','6101','--structure','full8']
        record={'category':'adapt','status':'COMPLETED','outer':{'seconds':2900.},'worker_command':{'arguments':args}}
        with patch.object(Path,'read_text',side_effect=[json.dumps([str(w.BASE/'native_synthetic/execution.json')]),json.dumps(record)]):
            a=SimpleNamespace(prior=w.BASE/'inputs/prior.json',category='adapt',cap=101.)
            with self.assertRaises(ValueError): launch.prior_cost_gate(a,{'arguments':args})

    def test_launcher_wrong_role_executable_before_native(self):
        launch=self.launcher()
        with tempfile.TemporaryDirectory(dir=w.BASE) as tmp:
            base=Path(tmp); command=base/'command.json'; binding=base/'binding.json'
            command.write_text(json.dumps({'executable':str(w.ROOT/'.venv-gpu/Scripts/python.exe'),
                                          'arguments':[],'arguments_line':''}),encoding='utf-8')
            binding.write_text(json.dumps({'files':{'synthetic':'not-used'}}),encoding='utf-8')
            argv=['launch','--command',str(command),'--binding',str(binding),'--prior',str(base/'prior.json'),
                  '--output',str(base/'native'),'--worker',str(base/'wrapper/wrapper_worker.json'),
                  '--source-commit','a'*40,'--category','short','--cap','100']
            with patch.object(launch,'BASE',base),patch.object(launch,'verify_binding',return_value={}),patch.object(sys,'argv',argv):
                with self.assertRaises(ValueError): launch.main()
            self.assertFalse((base/'native').exists())

    def test_early_real_worker_and_source_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            with patch.object(w,'BASE',base): w.early_worker(base/'wrapper')
            worker=json.loads((base/'wrapper/wrapper_worker.json').read_text(encoding='utf-8'))
            import os
            self.assertEqual(worker['pid'],os.getpid()); self.assertIs(type(worker['creation_filetime_utc']),int)
            source=base/'synthetic.py'; source.write_text('x=1',encoding='utf-8')
            with patch.object(w,'ROOT',base):
                w.verify_source_files({'synthetic.py':w.sha(source)})
                with self.assertRaises(ValueError): w.verify_source_files({'synthetic.py':'0'*64})

    def test_argv_restoration_and_source_inventory_refusal(self):
        before=sys.argv
        with self.assertRaises(RuntimeError):
            with w.argv_for(['synthetic']): raise RuntimeError('synthetic exit')
        self.assertIs(sys.argv,before)
        with self.assertRaises(ValueError): w.validate_inventory(w.OLD_BASE/'campaign01/score/final.json')

def run_qa():
    stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WiringContracts))
    return {'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'log':stream.getvalue(),
            'scope':'new fixed-directory/guard wiring only; no FOM, model or old scientific QA'}
