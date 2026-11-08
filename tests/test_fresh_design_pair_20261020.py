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
from experiments import fresh_design_pair_20261020 as w

class WiringContracts(unittest.TestCase):
    def test_fixed_path_refusal(self):
        with self.assertRaises(ValueError): w.inside(w.OLD_BASE/'campaign01/train')
        with self.assertRaises(ValueError): w.inside(w.BASE/'../active_fom_pair01/train')
        self.assertEqual(w.inside(w.CAMPAIGN/'train'),(w.CAMPAIGN/'train').resolve())

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
        with tempfile.TemporaryDirectory(dir=w.BASE) as tmp:
            base=Path(tmp).resolve(); campaign=base/'campaign01'; worker=base/'wrapper'
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
        path=w.BASE/'launch_stage.py'; spec=importlib.util.spec_from_file_location('fresh_design_launcher_qa',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_launcher_entry_and_worker_gate(self):
        launch=self.launcher(); commit='a'*40
        a=SimpleNamespace(category='qa',command=w.BASE/'commands/synthetic.json',prior=w.BASE/'inputs/synthetic_prior.json',source_commit=commit,
                          worker=w.BASE/'stages/synthetic_wrapper/wrapper_worker.json',cap=180.)
        names={Path(launch.__file__).relative_to(ROOT).as_posix(),a.command.relative_to(ROOT).as_posix(),a.prior.relative_to(ROOT).as_posix(),
            'experiments/native_process_receipt_20261020.ps1','experiments/fixed_edge_pair_20261019.py','experiments/linear_cache_cost_gate_20261019.py',
            'experiments/fresh_design_pair_20261020.py','tests/test_fresh_design_pair_20261020.py','experiments/fom_execution_environment_20261020.py',
            'experiments/active_fom_pair_20261020.py','experiments/active_fom_pair_fit_20261020.py','experiments/closure_discovery_data.py',
            'configs/private_dataset.json','reports/fresh_design_pair_20261020_protocol.md','reports/astra6_after_first_physics_QA_20261020.md'}
        args=['-m','experiments.fresh_design_pair_20261020','qa','--source-commit',commit,'--wrapper-output',str(a.worker.parent),'--output',str(w.BASE/'stages/synthetic_out'),'--max-seconds','120']
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
            base=Path(tmp).resolve(); command=base/'command.json'; binding=base/'binding.json'
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
        with tempfile.TemporaryDirectory(dir=w.BASE) as tmp:
            base=Path(tmp).resolve()
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

    def test_actual_remaining384_metadata_pool(self):
        pool,excluded=w.build_pool()
        self.assertEqual(len(pool),384)
        self.assertFalse({w.key(r) for r in pool}&excluded)
        counts=[[sum(int(r['bus'])==b and w.stratum(r)==s for r in pool) for s in range(3)] for b in w.BUSES]
        self.assertEqual(counts,[list(r) for r in w.EXPECTED_COUNTS])
        self.assertEqual([sum(row[s] for row in counts) for s in range(3)],[104,176,104])

    def test_new_design_dp_on_synthetic384(self):
        import itertools,math
        pool=[]; times=((.075,.125),(.175,.225,.275),(.35,.5))
        for bus,counts in zip(w.BUSES,w.EXPECTED_COUNTS):
            for s,count in enumerate(counts):
                for duration,xf in list(itertools.product(times[s],(.001,math.sqrt(.001*.01),.01)))[:count]:
                    pool.append({'bus':bus,'bus_id':str(bus),'duration_s':duration,'xf':xf,
                                 'line':f'synthetic_{bus}','location':0.,'id':f'synthetic_{bus}_{duration}_{xf}'})
        design=w.make_design(pool)
        self.assertEqual(design['seeds'],[7301,7302,7303]); self.assertEqual(design['pool_size'],384)
        self.assertEqual(design['val_dp']['total_count'],4252621230)
        self.assertIn(design['candidate_dp']['total_count'],(1097450640,399072960))
        val=design['val_conditions']; candidates=design['candidates']
        self.assertEqual(len({w.key(r) for r in val+candidates}),72)
        self.assertFalse({w.key(r) for r in val}&{w.key(r) for r in candidates})
        for b in w.BUSES:
            self.assertEqual(sum(int(r['bus'])==b for r in val),1)
            cs=[r for r in candidates if int(r['bus'])==b]
            self.assertEqual(len(cs),2); self.assertEqual(len({r['stratum'] for r in cs}),2)
        for s in w.STRATA:
            self.assertEqual(sum(r['stratum']==s for r in val),8)
            self.assertEqual(sum(r['stratum']==s for r in candidates),16)
            self.assertEqual(sum(r['stratum']==s and r['id'] in design['random_ids'] for r in candidates),4)
        with self.assertRaises(ValueError): w.make_design(pool[:-1])

    def test_parsed_relative_paths_forward_as_absolute(self):
        import os
        from experiments import fom_execution_environment_20261020 as env
        with tempfile.TemporaryDirectory(dir=w.BASE) as tmp:
            base=Path(tmp).resolve(); output=base/'scientific'; worker=base/'wrapper'
            argv=['prepare-binding','--output',os.path.relpath(output,Path.cwd()),
                  '--wrapper-output',os.path.relpath(worker,Path.cwd()),'--max-seconds','30','--source-commit','a'*40]
            with patch.object(w,'BASE',base),patch.object(w,'expected_environment',return_value={'synthetic':'known'}),patch.object(env,'preflight',return_value={'synthetic':'known'}),patch.object(w,'prepare_binding',return_value={'synthetic_preparation':True}) as prepare:
                self.assertEqual(w.main(argv),0)
                prepare.assert_called_once()
            received=json.loads((worker/'wrapper_worker.json').read_text(encoding='utf-8'))
            self.assertTrue(Path(received['cwd']).is_absolute())
            self.assertEqual(Path(received['received_paths']['output']),output.resolve())
            self.assertTrue(all(Path(p).is_absolute() for p in received['received_paths'].values()))

def run_qa():
    stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WiringContracts))
    return {'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'log':stream.getvalue(),
            'scope':'new fixed-directory/guard wiring only; no FOM, model or old scientific QA'}
