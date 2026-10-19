"""Bounded real-train CUDA QA. Invoke through the cost-gate receipt supervisor."""
import copy
import json
from pathlib import Path
import sys
import time
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]


def check_owned_tree_cleanup():
    from edge_pair_cost_gate_20261019 import terminate_owned_tree, wait_owned
    calls = []
    class OwnedProcess:
        pid = 43210
        def wait(self, timeout):
            return 1
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='owned tree terminated', stderr='')
    result = terminate_owned_tree(OwnedProcess(), runner=runner, platform='nt')
    assert calls == [(['taskkill','/PID','43210','/T','/F'],
                      {'capture_output':True,'text':True,'timeout':2,'check':False})]
    assert result['returncode']==0 and result['launcher_returncode']==1
    class InterruptedProcess:
        pid = 54321
        def wait(self, timeout):
            raise KeyboardInterrupt('mock wait interruption')
    cleanup = {}
    with patch('edge_pair_cost_gate_20261019.terminate_owned_tree', return_value={'mock_terminated':True}) as terminate:
        process = InterruptedProcess()
        try:
            wait_owned(process,10,cleanup)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('must retain original interruption')
        terminate.assert_called_once_with(process)
    assert cleanup == {'mock_terminated':True}


def run_qa(output):
    import numpy as np
    import torch
    from edge_pair_cost_gate_20261019 import structures, TOLERANCES, dependency_hashes, COHORT_SHA
    from closure_mechanism_study import train_cohort, SOURCE
    from shared_closure_rom_0101 import load_model
    from symbolic_closure_rom import SymbolicClosureROM, term_key, _BASE_TERMS
    from closure_training_runtime import TensorEventStore, objective, export_cpu, resolve_device, synchronize
    from train_closure import make_optimizer, save_checkpoint, restore_checkpoint, write
    from train_local_memory_0098 import graph, histories_for
    from discover_closure import loss as reference_loss
    torch.set_num_threads(1)
    device = resolve_device('cuda')
    output = Path(output)
    record = {'status': 'RUNNING', 'tolerances': TOLERANCES, 'code': dependency_hashes(),
              'val_read': False, 'test_read': False, 'rows': []}
    write(output/'qa.json', record)  # tolerances recorded before any comparison
    def close(a, b, kind):
        atol, rtol = TOLERANCES[kind]
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    def layout(m, o):
        names = {id(p): n for n, p in m.named_parameters()}
        return [{'names': [names[id(p)] for p in g['params']], 'block': g['block'], 'lr': g['lr']}
                for g in o.param_groups]
    def compare_models(a, b):
        aa, bb = list(a.named_parameters()), list(b.named_parameters())
        assert [(n, p.requires_grad) for n,p in aa] == [(n,p.requires_grad) for n,p in bb]
        for (_,p),(_,q) in zip(aa,bb):
            close(p,q,'adam')
        assert dict(a.named_buffers()).keys() == dict(b.named_buffers()).keys()
        for n, p in a.named_buffers():
            assert torch.equal(p, dict(b.named_buffers())[n]), n
    def compare_adam(a, b):
        sa, sb = a.state_dict(), b.state_dict()
        assert sa['param_groups'] == sb['param_groups']
        assert sa['state'].keys() == sb['state'].keys()
        for k in sa['state']:
            assert sa['state'][k].keys() == sb['state'][k].keys()
            for name, p in sa['state'][k].items():
                q = sb['state'][k][name]
                if name == 'step':
                    assert torch.equal(p,q)
                else:
                    close(p,q,'adam')
    try:
        check_owned_tree_cleanup()
        worker = json.loads((output/'worker.json').read_text(encoding='utf-8'))
        import os
        assert worker['pid']==os.getpid() and worker['output']==str(output.resolve())
        record['owned_tree_cleanup_mock']='PASS; taskkill only mocked, no other process terminated'
        record['actual_worker_pid']=worker['pid']
        events, force, cohort = train_cohort()
        assert cohort['sha256'] == COHORT_SHA
        # Genuine fixed slices; local .2s target uses every actually available step.
        selected = [events[0], next(e for e in events if e['label']=='unstable')]
        slices = [{**e, 'x': e['x'][:14].copy(), 'torque': e['torque'][:14].copy()} for e in selected]
        samples = [(e,10) for e in slices]
        store = TensorEventStore(slices, device)
        record.update(cohort=cohort, real_slice_ids=[e['id'] for e in slices], real_slice_steps=[len(e['x']) for e in slices])
        for name, model in structures(device).items():
            # Independent original compiler, with expression actually absent for deletion.
            ref = SymbolicClosureROM.from_source(load_model(SOURCE)).with_terms(model.terms).to(device)
            assert len(ref.terms) == (8 if name=='full8' else 7)
            assert (term_key(_BASE_TERMS[2]) in map(term_key, ref.terms)) == (name=='full8')
            opt, ropt = make_optimizer(model), make_optimizer(ref)
            assert layout(model,opt) == layout(ref,ropt)
            compare_models(model, ref)
            history = store.histories(samples)
            z = torch.cat((history[:,-1], model.encode(history)), -1).detach()
            close(model.field(z), ref.field(z), 'field')
            internal = model.interface_to_state(z).detach().requires_grad_(True)
            for a,b in zip(model.energy_and_gradient(internal), ref.energy_and_gradient(internal)):
                close(a,b,'field')
            autodiff = torch.autograd.grad(model.energy(internal).sum(), internal)[0]
            close(model.energy_and_gradient(internal)[1], autodiff, 'field')
            with graph(model):
                value, terms = objective(model,store,samples,.03,force)
                value.backward()
            # Default-device context moves only newly constructed reference inputs to the
            # existing model's CUDA device; original equations and objective are unchanged.
            with torch.device(device), graph(ref):
                wanted, wanted_terms = reference_loss(ref,samples,histories_for(samples),.03,force)
                wanted.backward()
            close(value,wanted,'loss')
            assert terms.keys() == wanted_terms.keys()
            for key in terms:
                close(terms[key],wanted_terms[key],'loss')
            for (n,p),(rn,q) in zip(model.named_parameters(),ref.named_parameters()):
                assert n == rn and (p.grad is None) == (q.grad is None)
                if p.grad is not None:
                    close(p.grad,q.grad,'gradient')
            for m,o in ((model,opt),(ref,ropt)):
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],100.)
                o.step()
            compare_models(model,ref); compare_adam(opt,ropt)
            folder = output/name; folder.mkdir()
            exported = folder/'export.json'
            export_cpu(model,exported)
            recovered_export = SymbolicClosureROM.from_export(exported).to(device)
            close(model.field(z), recovered_export.field(z), 'export')
            close(model.rollout(history,3,.01),recovered_export.rollout(history,3,.01),'export')
            rng = np.random.default_rng(6101); coverage = {'cover_order': [1,0], 'cursor': 1}
            identity = {'qa': name, 'cohort': COHORT_SHA, 'code': record['code']}
            tick = time.perf_counter()
            save_checkpoint(folder,model,opt,rng,coverage,1,identity,[])
            synchronize(device); save_seconds = time.perf_counter()-tick
            numpy_state = copy.deepcopy(rng.bit_generator.state)
            cpu_state = torch.get_rng_state().clone(); gpu_state = torch.cuda.get_rng_state(device).clone()
            tick = time.perf_counter()
            restored, restored_opt, restored_rng, restored_coverage, meta, changed = restore_checkpoint(folder,device,identity)
            synchronize(device); restore_seconds = time.perf_counter()-tick
            assert not changed and meta['step']==1 and restored_coverage==coverage
            assert restored_rng.bit_generator.state == numpy_state
            assert torch.equal(torch.get_rng_state(),cpu_state)
            assert torch.equal(torch.cuda.get_rng_state(device),gpu_state)
            assert layout(model,opt) == layout(restored,restored_opt)
            compare_models(model,restored); compare_adam(opt,restored_opt)
            for m,o in ((model,opt),(restored,restored_opt)):
                o.zero_grad(set_to_none=True)
                with graph(m):
                    v,_=objective(m,store,samples,.03,force); v.backward()
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],100.)
                o.step()
            compare_models(model,restored); compare_adam(opt,restored_opt)
            record['rows'].append({'structure':name,'status':'PASS', 'optimizer_layout':layout(model,opt),
                                   'populated_checkpoint_seconds':save_seconds,'restore_seconds':restore_seconds})
            write(output/'qa.json',record)
        record['status']='PASS'
    except BaseException as exc:
        record.update(status='FAILED',error=repr(exc))
        raise
    finally:
        write(output/'qa.json',record)
