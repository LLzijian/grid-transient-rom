"""Integration contracts beyond isolated AST/helper unit tests."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import discover_closure as runner
from closure_discovery_benchmarks import toy_source
from symbolic_closure_rom import SymbolicClosureROM
from closure_discovery_support import propose_mutations


def test_two_parent_library_and_generated_actually_differ():
    source=toy_source()
    terms=[{'channel':'port','expression':{'op':'var','name':'q'}}]
    pools={mode:propose_mutations([terms,terms],mode=mode,max_candidates=64) for mode in ('library','generated')}
    assert {r['parent'] for r in pools['generated']}=={0,1}
    old={str(t) for t in source.terms}
    assert any(any(t['expression']['op'] in ('tanh','pow2','pow3','pow4') for t in r['terms']) for r in pools['generated'])
    assert not any(any(t['expression']['op']=='tanh' for t in r['terms']) for r in pools['library'])


def test_generated_expressions_depend_on_parent_subtrees():
    from closure_discovery_support import _candidate_terms,term_key
    p={'op':'var','name':'p'};q={'op':'var','name':'q'}
    parent={'channel':'node','expression':{'op':'mul','args':[{'op':'pow2','arg':p},q]}}
    target={'channel':'node','expression':{'op':'mul','args':[{'op':'pow2','arg':p},{'op':'pow2','arg':q}]}}
    assert term_key(target) in {term_key(t) for t in _candidate_terms([parent])}
    assert term_key(target) not in {term_key(t) for t in _candidate_terms([])}


def test_direction_change_is_whole_exchange_or_psd_group():
    source=toy_source();mask=source.template_mask[3:7].tolist();mask[3]=1.
    m=runner.proposal_model(source,{'terms':source.terms,'direction_mask':mask})
    J,R=m.matrices()
    assert torch.max(torch.abs(J+J.T))<1e-12
    assert torch.linalg.eigvalsh(R).min()>-1e-12
    assert source.template_mask[6]==0


def test_model_optimizer_groups_keep_old_rate_semantics():
    m=toy_source().with_terms([{'channel':'port','expression':{'op':'var','name':'q'}}])
    opt=runner.optimizer(m)
    named=dict(m.named_parameters())
    groups={id(p):g['block'] for g in opt.param_groups for p in g['params']}
    assert groups[id(named['coefficients'])]=='closure'
    assert groups[id(named['_template_scalars.3'])]=='closure'
    assert groups[id(named['_q_quadratic_raw'])]=='skeleton'


def test_checkpoint_resumes_optimizer_and_rng_without_data(tmp_path):
    m=toy_source();opt=runner.optimizer(m);rng=np.random.default_rng(1701)
    trace=[{'step':1,'loss':1.}]
    runner.checkpoint(tmp_path,m,opt,rng,1,trace)
    restored=SymbolicClosureROM.from_export(tmp_path/'model.json')
    x=torch.randn((2,40),dtype=torch.float64)*.02
    assert torch.allclose(m.field(x),restored.field(x),atol=1e-12,rtol=0)
    saved=torch.load(tmp_path/'training.pt',weights_only=False)
    check=np.random.default_rng();check.bit_generator.state=saved['numpy_rng']
    assert check.integers(100000)==rng.integers(100000)


def test_disabled_direction_remains_trainable_on_activation_and_optimizer_resume(tmp_path):
    old=toy_source()
    old._template_scalars[6].requires_grad_(False)
    # Exercise the registered-source conversion, where inactive amplitudes
    # historically carried requires_grad=False.
    from local_memory_rom_0098 import LocalMemoryROM
    legacy=LocalMemoryROM(old.structured,input_scale=old.input_scale,p_scale=old.p_scale,arm='L10-S')
    legacy._template_scalars[6].requires_grad_(False)
    model=SymbolicClosureROM.from_source(legacy)
    assert model._template_scalars[6].requires_grad
    mask=model.template_mask[3:7].tolist();mask[3]=1.
    candidate=runner.proposal_model(model,{'terms':model.terms,'direction_mask':mask})
    assert candidate._template_scalars[6].requires_grad
    candidate.export(tmp_path/'model.json');restored=SymbolicClosureROM.from_export(tmp_path/'model.json')
    assert {n for n,p in restored.named_parameters() if p.requires_grad}=={n for n,p in candidate.named_parameters() if p.requires_grad}
    runner.optimizer(restored).load_state_dict(runner.optimizer(candidate).state_dict())


def test_resume_ranks_saved_32_step_state_after_candidate_reaches_512(tmp_path,monkeypatch):
    model=toy_source().with_terms([{'channel':'port','expression':{'op':'var','name':'q'}}])
    with torch.no_grad():model.coefficients.fill_(.2)
    model.export(tmp_path/'model_at_step00032.json')
    with torch.no_grad():model.coefficients.fill_(99.)
    model.export(tmp_path/'model.json')
    (tmp_path/'checkpoint.json').write_text('{"step":512}')
    calls=[]
    def fake_score(m,events,**kwargs):
        value=float(m.coefficients[0].detach())
        calls.append(value);return {'risk':value,'events':[]}
    monkeypatch.setattr(runner,'score',fake_score)
    result=runner.score_at_level(tmp_path,[],32)
    assert result['risk']==pytest.approx(.2)
    assert result['adaptation_updates']==32
    assert runner.score_at_level(tmp_path,[],32)==result
    assert calls==[.2]
    (tmp_path/'model_at_step00032.json').unlink()
    with pytest.raises(ValueError,match='cannot relabel'):runner.score_at_level(tmp_path,[],32)


def test_primary_comparison_refuses_invalid_event():
    from closure_discovery_confirmation import paired_primary
    row={'id':'e','group':'g','label':'stable','complete_5s':True,'numerical_pass':False,'mse':None}
    assert paired_primary({'records':[row]},{'records':[row]})['status']=='unresolved'


def test_output_boolean_stays_boolean():
    assert runner.safe({'test_read':False})['test_read'] is False


def test_full_refit_schedule_exceeds_historical_800_cap():
    assert runner.horizon_for_update(100)==2.
    assert [runner.horizon_for_update(i) for i in range(101,104)]==[1.,2.,5.]
    assert runner.horizon_for_update(801) in (1.,2.,5.)
    assert runner.horizon_for_update(1024) in (1.,2.,5.)


def test_edge_cosine_is_centered_in_physical_angle_not_feature():
    from symbolic_closure_rom import numpy_field
    base=toy_source();m=base.with_terms([{'channel':'edge','expression':{'op':'var','name':'c'}}])
    with torch.no_grad():m.coefficients.fill_(1.)
    z=torch.sin(torch.arange(80,dtype=torch.float64)).reshape(2,40)*.1
    x=m.interface_to_state(z);eta=x[:,:10] @ m.base_D.T
    weight=m.base_coupling/m.base_coupling.mean()
    expected=(weight*(torch.cos(m.phi+eta)-torch.cos(m.phi)+torch.sin(m.phi)*eta)).sum(-1)
    assert expected.abs().max()>1e-6
    assert torch.allclose(m.energy(x)-base.energy(x),expected,atol=1e-12,rtol=0)
    exact=torch.autograd.functional.jacobian(lambda s:m.energy(s),x)
    x=x.detach().requires_grad_(True)
    gradient=torch.autograd.grad(m.energy(x).sum(),x)[0]
    assert torch.allclose(gradient,m.gradient(x),atol=1e-11,rtol=0)
    assert np.max(np.abs(m.field(z).detach().numpy()-numpy_field(m,z.numpy())))<1e-11


def test_machine_relabel_and_edge_orientation_invariance():
    import copy
    m=toy_source().with_terms([
      {'channel':'edge','expression':{'op':'mul','args':[{'op':'add','args':[{'op':'var','name':'qi'},{'op':'mul','args':[{'op':'const','value':-1},{'op':'var','name':'qj'}]}]},{'op':'var','name':'s'}]}},
      {'channel':'port','expression':{'op':'mul','args':[{'op':'var','name':'q'},{'op':'var','name':'w'}]}}])
    with torch.no_grad():m.coefficients.copy_(torch.tensor([.1,.2],dtype=torch.float64))
    perm=torch.tensor([2,0,7,4,1,9,5,8,6,3]);cols=torch.cat([perm+10*i for i in range(4)])
    n=copy.deepcopy(m)
    with torch.no_grad():
      for key in ('mass','p_scale','input_scale_delta','input_scale_omega','delta_star','rate','gain'):
        getattr(n,key).copy_(getattr(m,key)[perm])
      n.input_scale.copy_(torch.cat((n.input_scale_delta,n.input_scale_omega)))
      n.base_D.copy_(m.base_D[:,perm]);n.neighbor_A.copy_(m.neighbor_A[perm][:,perm])
      n.port_s_weight.copy_(m.port_s_weight[:,perm]);n.port_c_weight.copy_(m.port_c_weight[:,perm])
      n.structured.mass.copy_(m.structured.mass[perm]);n.structured.delta_star.copy_(m.structured.delta_star[perm])
      n.structured.gain.copy_(m.structured.gain[perm]);n.structured.D.copy_(m.structured.D[:,perm])
      for key in ('damping','jpi_p','rpp'):
        getattr(n.structured,key).copy_(getattr(m.structured,key)[perm])
        getattr(n.structured,key+'_mask').copy_(getattr(m.structured,key+'_mask')[perm])
      pq=torch.cat((perm,perm+10))
      n.structured.storage_base.copy_(m.structured.storage_base[pq]);n.structured.storage_base_mask.copy_(m.structured.storage_base_mask[pq])
      n.structured.scales.copy_(m.structured.scales[pq])
    z=torch.sin(torch.arange(80,dtype=torch.float64)).reshape(2,40)*.03
    assert torch.allclose(n.field(z[:,cols]),m.field(z)[:,cols],atol=1e-11,rtol=0)
    history=torch.sin(torch.arange(660,dtype=torch.float64)).reshape(2,11,30)*.03
    assert torch.allclose(n.encode(history[:,:,cols[:30]]),m.encode(history)[:,perm],atol=1e-12,rtol=0)
    reversed_model=copy.deepcopy(m)
    with torch.no_grad():
      reversed_model.base_D=-m.base_D.clone();reversed_model.phi=-m.phi.clone();reversed_model.port_s_weight=-m.port_s_weight.clone()
      reversed_model.structured.D=-m.structured.D.clone();reversed_model.structured.phi=-m.structured.phi.clone()
    assert torch.allclose(m.field(z),reversed_model.field(z),atol=1e-11,rtol=0)


def _condition_row(split='train', duration=.1):
    return {'id':'fault-1','bus':10,'bus_id':'10','line':'L1','location':.25,
            'duration_s':duration,'xf':.01,'split':split}


def _install_fake_generation(monkeypatch, tmp_path, *, status='complete', freeze=None):
    import json
    import closure_discovery_data as data

    calls=[]
    def normalize(conditions, split):
        rows=conditions.get('val_conditions' if split=='val' else 'train_selected') if isinstance(conditions,dict) else conditions
        return [{**row,'split':split} for row in rows]
    monkeypatch.setattr(data,'_conditions_for_split',normalize)
    if freeze is not None:
        monkeypatch.setattr(data,'_freeze_snapshot',lambda path:json.loads(Path(path).read_text(encoding='utf-8')))

    def generate(conditions, out, split, cap_seconds, *, freeze_path=None, simulator=None):
        calls.append((split,cap_seconds))
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        normalized=[data._attempt_record(row,split) for row in normalize(conditions,split)]
        attempts=[]
        for condition in normalized:
            steps={}
            for step in ('0.005','0.0025'):
                raw=out/'raw'/f"{condition['event_id']}_{step}.npz"
                raw.parent.mkdir(parents=True,exist_ok=True)
                raw.write_bytes(f"{condition['event_id']}:{step}".encode())
                raw.with_suffix('.json').write_text('{"metadata":{}}',encoding='utf-8')
                steps[step]={'status':'completed','raw_file':raw.relative_to(out).as_posix()}
            attempts.append({'condition':condition,'steps':steps,'pair_qualification':{'pass':True}})
        payload={'schema_version':1,'split':split,'conditions':normalized,'attempts':attempts,
                 'freeze':freeze if split=='val' else None,'status':status}
        (out/'generation.json').write_text(json.dumps(payload),encoding='utf-8')
        return payload
    monkeypatch.setattr(data,'generate_conditions',generate)
    return calls


def test_generate_or_reuse_reuses_only_matching_complete_raw_artifacts(tmp_path, monkeypatch):
    import closure_discovery_confirmation as confirmation

    calls=_install_fake_generation(monkeypatch,tmp_path)
    output=tmp_path/'generation'
    conditions=[_condition_row()]
    first=confirmation.generate_or_reuse(conditions,output,'train',60.)
    manifest_bytes=(output/'generation.json').read_bytes()
    assert first['_reused'] is False
    assert calls==[('train',60.)]

    second=confirmation.generate_or_reuse(conditions,output,'train',1.)
    assert second['_reused'] is True
    assert calls==[('train',60.)]
    assert (output/'generation.json').read_bytes()==manifest_bytes

    with pytest.raises(confirmation.RecoveryUnresolved,match='conditions changed'):
        confirmation.generate_or_reuse([_condition_row(duration=.2)],output,'train',60.)
    assert calls==[('train',60.)]

    raw_path=output/second['_raw_sha256'].keys().__iter__().__next__()
    raw_path.write_bytes(b'tampered')
    with pytest.raises(confirmation.RecoveryUnresolved,match='immutable artifact differs'):
        confirmation.generate_or_reuse(conditions,output,'train',60.)
    assert calls==[('train',60.)]


@pytest.mark.parametrize('status',['RUNNING','budget_exhausted'])
def test_generate_or_reuse_leaves_incomplete_generation_unresolved(tmp_path, monkeypatch, status):
    import json
    import closure_discovery_data as data
    import closure_discovery_confirmation as confirmation

    calls=_install_fake_generation(monkeypatch,tmp_path)
    output=tmp_path/'partial';output.mkdir()
    expected=[data._attempt_record(row,'train') for row in data._conditions_for_split([_condition_row()], 'train')]
    manifest={'split':'train','conditions':expected,'attempts':[],'freeze':None,'status':status}
    path=output/'generation.json';path.write_text(json.dumps(manifest),encoding='utf-8')
    before=path.read_bytes()
    with pytest.raises(confirmation.RecoveryUnresolved,match='unresolved'):
        confirmation.generate_or_reuse([_condition_row()],output,'train',60.)
    assert path.read_bytes()==before
    assert calls==[]


def test_val_generation_freeze_summary_change_is_rejected_without_regeneration(tmp_path, monkeypatch):
    import closure_discovery_data as data
    import closure_discovery_confirmation as confirmation

    freeze_path=tmp_path/'freeze.json'
    freeze_path.write_text('frozen',encoding='utf-8')
    state={'summary':{'path':str(freeze_path),'manifest_sha256':'a'*64,'models':[]}}
    monkeypatch.setattr(data,'_freeze_snapshot',lambda path:state['summary'])
    summary=state['summary']
    calls=_install_fake_generation(monkeypatch,tmp_path,freeze=summary)
    monkeypatch.setattr(data,'_freeze_snapshot',lambda path:state['summary'])
    output=tmp_path/'val'
    conditions=[_condition_row(split='val')]
    confirmation.generate_or_reuse(conditions,output,'val',60.,freeze_path=freeze_path)
    before=(output/'generation.json').read_bytes()
    state['summary']={'path':str(freeze_path),'manifest_sha256':'b'*64,'models':[]}
    with pytest.raises(confirmation.RecoveryUnresolved,match='freeze summary changed'):
        confirmation.generate_or_reuse(conditions,output,'val',60.,freeze_path=freeze_path)
    assert (output/'generation.json').read_bytes()==before
    assert len(calls)==1


def test_freeze_is_write_once_exact_and_rechecks_model_hash(tmp_path):
    import hashlib
    import closure_discovery_confirmation as confirmation

    model_paths=[];models=[]
    for name in ('main','fixed'):
        path=tmp_path/f'{name}.json';path.write_text(name,encoding='utf-8');model_paths.append(path)
        models.append({'name':name,'seed':2701,'path':str(path),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    conditions=[_condition_row('val')]
    source=tmp_path/'fault_manifest.json';source.write_text('{"val24":[]}',encoding='utf-8')
    path=tmp_path/'confirmation_freeze.json'
    first,reused=confirmation._freeze_or_reuse(path,models,conditions,source)
    assert reused is False
    frozen_bytes=path.read_bytes()
    second,reused=confirmation._freeze_or_reuse(path,None,conditions,source)
    assert reused is True and second==first
    assert path.read_bytes()==frozen_bytes
    with pytest.raises(confirmation.RecoveryUnresolved,match='cohort'):
        confirmation._freeze_or_reuse(path,None,[_condition_row('val',duration=.2)],source)
    model_paths[0].write_text('changed',encoding='utf-8')
    with pytest.raises(confirmation.RecoveryUnresolved,match='hash changed'):
        confirmation._freeze_or_reuse(path,None,conditions,source)
    assert path.read_bytes()==frozen_bytes


def test_evaluation_reuses_only_same_model_and_full_event_cohort(tmp_path, monkeypatch):
    import closure_discovery_confirmation as confirmation

    calls=[]
    def fake_evaluate(model,events,output,*,identity=None):
        calls.append(identity)
        result={'_identity':identity,'event_count':len(events),
                'records':[{'id':event['id'],'group':event['group'],'label':event['label'],
                            'numerical_pass':True,'complete_5s':True,'mse':0.,'metrics':[],
                            'components':{h:{} for h in ('1','2','5','8','prefix')},
                            'pair':{},'selected_pair':{},'scored_numerics':[]} for event in events],
                'primary_risk':0.,'primary_complete_stable_count':len(events),
                'all_numerical_pass':True,'classification_counts':{},'scope':'mock','test_read':False}
        confirmation._write_once_json(output,result)
        return result
    monkeypatch.setattr(confirmation,'evaluate_events',fake_evaluate)
    events=[{'id':'e1','group':'g1','label':'stable','new_split':'val',
             'x':np.asarray([[1.,2.]]),'torque':np.asarray([[3.]]),'t':np.asarray([0.])}]
    output=tmp_path/'evaluation.json'
    model=object()
    first=confirmation.evaluate_or_reuse(model,events,output,'model-hash-a')
    second=confirmation.evaluate_or_reuse(model,events,output,'model-hash-a')
    assert first==second and len(calls)==1
    with pytest.raises(confirmation.RecoveryUnresolved,match='identity changed'):
        confirmation.evaluate_or_reuse(model,events,output,'model-hash-b')
    changed=[{**events[0],'id':'different'}]
    with pytest.raises(confirmation.RecoveryUnresolved,match='identity changed'):
        confirmation.evaluate_or_reuse(model,changed,output,'model-hash-a')
    assert len(calls)==1


def test_train_prediction_disagreement_uses_structured_point_error(monkeypatch):
    import train_structured_rom
    import closure_discovery_confirmation as confirmation

    seen={}
    def point_error(pred,truth,model):
        seen.update(pred=pred,truth=truth,model=model)
        return torch.tensor([2.,6.],dtype=torch.float64)
    monkeypatch.setattr(train_structured_rom,'_point_error',point_error)
    main_prediction=torch.tensor([[1.]])
    alternative_prediction=torch.tensor([[4.]])
    model=object()
    assert confirmation.train_prediction_disagreement(main_prediction,alternative_prediction,model)==4.
    assert seen=={'pred':main_prediction,'truth':alternative_prediction,'model':model}


def test_run_confirmation_resumes_frozen_val_without_simulation_or_refit(tmp_path, monkeypatch):
    import hashlib
    import json
    import time
    import closure_discovery_data as data
    import closure_discovery_confirmation as confirmation
    from symbolic_closure_rom import SymbolicClosureROM

    out=tmp_path
    (out/'D').mkdir()
    condition=_condition_row('val')
    fault_manifest=out/'new_fault_manifest.json'
    fault_manifest.write_text(json.dumps({'val24':[condition],'train_pool':[]}),encoding='utf-8')
    source_models=[];endpoints=[]
    for name in ('main','fixed'):
        source_path=out/'C'/name/'2701'/'model.json'
        source_path.parent.mkdir(parents=True,exist_ok=True)
        source_path.write_text(f'c-{name}',encoding='utf-8')
        source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
        source={'name':name,'seed':2701,'path':str(source_path.parent),'sha256':source_sha}
        source_models.append(source)
        endpoint_path=out/'D'/'models'/name/'2701'/'model.json'
        endpoint_path.parent.mkdir(parents=True,exist_ok=True)
        endpoint_path.write_text(f'd-{name}',encoding='utf-8')
        endpoints.append({'name':name,'seed':2701,'source_path':str(source_path),
                          'source_sha256':source_sha,'path':str(endpoint_path),
                          'sha256':hashlib.sha256(endpoint_path.read_bytes()).hexdigest()})
    (out/'C').mkdir(exist_ok=True)
    (out/'C/frozen.json').write_text(json.dumps({'models':source_models}),encoding='utf-8')
    freeze_path=out/'D/confirmation_freeze.json'
    confirmation._freeze_or_reuse(freeze_path,endpoints,[condition],fault_manifest)
    frozen=json.loads(freeze_path.read_text(encoding='utf-8'))
    freeze_summary={'path':str(freeze_path.resolve()),
                    'manifest_sha256':hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                    'main_name':'main',
                    'models':[{'name':row['name'],'path':str(Path(row['path']).resolve()),
                               'sha256':row['sha256']} for row in frozen['models']]}
    calls=_install_fake_generation(monkeypatch,tmp_path,freeze=freeze_summary)
    monkeypatch.setattr(data,'_freeze_snapshot',lambda path:freeze_summary)
    val_out=out/'D/new_data/val'
    confirmation.generate_or_reuse([condition],val_out,'val',60.,freeze_path=freeze_path)

    # Reuse the already completed artifact; the simulator must not be called again.
    loaded={'id':'newval_fault-1','group':'10','label':'stable','new_split':'val',
            'trajectory_end_s':1.0,'x':np.zeros((4,30)),
            'torque':np.zeros((4,10)),'t':np.arange(4.)}
    monkeypatch.setattr(data,'load_new_events',lambda *a,**k:([loaded],[]))
    monkeypatch.setattr(confirmation,'evaluate_or_reuse',
                        lambda model,events,path,model_sha256,**kwargs:{'records':[{'id':loaded['id']}],
                                                               'model_sha256':model_sha256})
    monkeypatch.setattr(confirmation,'paired_primary',lambda main,control:{'status':'complete','difference':0.})
    monkeypatch.setattr(SymbolicClosureROM,'from_export',classmethod(lambda cls,path:object()))
    import discover_closure as runner_module
    monkeypatch.setattr(runner_module,'fit',lambda *a,**k:pytest.fail('fit must not rerun on frozen recovery'))
    monkeypatch.setattr(runner_module,'train_inputs',lambda *a,**k:pytest.fail('TRAIN must not rerun on frozen recovery'))

    result=confirmation.run_confirmation(out,time.perf_counter()+600.)
    assert result['generation_reused'] is True
    assert result['freeze_reused'] is True
    assert result['new_val']==1
    assert calls==[('val',60.)]


def test_d_fit_recovery_reuses_only_matching_verified_checkpoint(tmp_path):
    import hashlib
    import json
    import closure_discovery_confirmation as confirmation

    class Model:
        def __init__(self,terms=None,mask=None):
            self.terms=terms or [{'channel':'port','expression':{'op':'var','name':'q'}}]
            self.template_mask=torch.as_tensor(mask or [1.],dtype=torch.float64)
        @classmethod
        def from_export(cls,path):
            saved=json.loads(Path(path).read_text(encoding='utf-8'))
            return cls(saved['terms'],saved['template_mask'])

    calls=[]
    def fake_fit(model,events,scale,target,path,seed,*,deadline=None):
        calls.append((target,seed,deadline))
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        model_path=path/'model.json';training_path=path/'training.pt';checkpoint_path=path/'checkpoint.json'
        model_path.write_text(json.dumps({'terms':model.terms,
                                          'template_mask':model.template_mask.tolist()}),encoding='utf-8')
        training_path.write_bytes(b'optimizer-and-rng')
        checkpoint_path.write_text(json.dumps({'step':8,
            'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'training_sha256':hashlib.sha256(training_path.read_bytes()).hexdigest()}),encoding='utf-8')
        return model,{'updates':8}

    model=Model();events=[{'id':'train-1','group':'bus','label':'stable',
                           'x':np.zeros((2,3)),'torque':np.zeros((2,1)),'t':np.arange(2.)}]
    path=tmp_path/'models/main/2701'
    confirmation._fit_with_recovery(fake_fit,Model,model,events,2.,256,path,2701,100.,'source-hash')
    saved_training=(path/'training.pt').read_bytes()
    confirmation._fit_with_recovery(fake_fit,Model,model,events,2.,256,path,2701,100.,'source-hash')
    assert calls==[(256,2701,100.),(256,2701,100.)]
    assert (path/'training.pt').read_bytes()==saved_training
    with pytest.raises(confirmation.RecoveryUnresolved,match='immutable artifact differs'):
        confirmation._fit_with_recovery(fake_fit,Model,model,events+[{'id':'new'}],2.,256,
                                        path,2701,100.,'source-hash')
    assert len(calls)==2


def test_atomic_json_write_retries_transient_windows_reader(tmp_path,monkeypatch):
    import json
    import discover_closure as runner
    path=tmp_path/'progress.json'
    path.write_text('{"old": true}',encoding='utf-8')
    original_replace=runner.os.replace
    attempts=[]
    delays=[]
    def temporarily_locked(source,destination):
        attempts.append((source,destination))
        if len(attempts)<4:raise PermissionError('destination temporarily opened by a reader')
        original_replace(source,destination)
    monkeypatch.setattr(runner.os,'replace',temporarily_locked)
    monkeypatch.setattr(runner.time,'sleep',delays.append)
    runner.write(path,{'completed':32})
    assert json.loads(path.read_text())=={'completed':32}
    assert len(attempts)==4
    assert delays==[.01,.02,.04]
    assert not path.with_suffix('.json.tmp').exists()


def test_atomic_json_write_preserves_temporary_file_on_persistent_lock(tmp_path,monkeypatch):
    import json
    import discover_closure as runner
    path=tmp_path/'progress.json'
    path.write_text('{"old": true}',encoding='utf-8')
    def locked(*args):raise PermissionError('persistent destination lock')
    monkeypatch.setattr(runner.os,'replace',locked)
    monkeypatch.setattr(runner.time,'sleep',lambda delay:None)
    with pytest.raises(PermissionError):runner.write(path,{'completed':32})
    assert json.loads(path.read_text())=={'old':True}
    assert json.loads(path.with_suffix('.json.tmp').read_text())=={'completed':32}
