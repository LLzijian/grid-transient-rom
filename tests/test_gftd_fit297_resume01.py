"""Resume orchestration and I/O fault mocks; no scientific calls or real events."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import gftd_fit297_20261107_resume01 as run


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json.dumps(value).encode())


@pytest.fixture
def stored(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run.original,'ROOT',tmp_path)
    monkeypatch.setattr(run,'PARENT_OUTPUT',tmp_path/'results/parent');monkeypatch.setattr(run,'PARENT_PRIVATE',tmp_path/'private/parent')
    monkeypatch.setattr(run,'OUTPUT',tmp_path/'results/resume');monkeypatch.setattr(run,'PRIVATE',tmp_path/'private/resume')
    monkeypatch.setattr(run.fit_engine,'_CACHE',{});monkeypatch.setattr(run.fit_engine,'_PREPARATION_FAILURES',{});monkeypatch.setattr(run.fit_engine,'_LEDGERS',{})
    ids=[f'id{i:03d}' for i in range(297)]
    packets=[dict(theta_history=np.zeros((11,10)).tolist(),omega_history=np.zeros((11,10)).tolist(),
        history_times=np.linspace(-.1,0,11).tolist(),p0=[0.]*10,sigma=0.) for i in ids]
    mapping={'information_svd':(186,230),'conditional_projections':(135,180),'memory_svd':(594,0)}
    models={}
    for m,mode in enumerate(('static','modulated')):
        counts={k:v[m] for k,v in mapping.items()}
        ledger={k:dict(actual=v if mode=='modulated' else counts[k]) for k,v in run.EXPECTED_LEDGER.items()}
        models[mode]=SimpleNamespace(fit_state=dict(counts=counts,adaptation_indices=[0],analysis_budget=ledger,preparation_key='fixture'),
            inference_config={'preparation_key':'fixture','mode':mode,'fit_request':{}},functional=None,mass=np.ones(10),delta_star=np.zeros(10))
        dump(run.PARENT_PRIVATE/(mode+'_model.json'),{'fixture':mode})
        dump(run.PARENT_PRIVATE/(mode+'_fit_details.json'),{})
    counts=dict(fits_attempted=2,fits_completed=2,initializations_attempted=41,initializations_completed=8,
        future_pairs_attempted=0,future_pairs_persisted=0,future_pairs_completed=0,future_pairs_failed=0,
        rk4_attempted=0,rk4_completed=0,scoring_attempted=0,scoring_completed=0,
        fit={m:v.fit_state['counts'] for m,v in models.items()},initialization=run.EXPECTED_INIT.copy(),coverage={})
    latest=dict(counts=counts,artifacts=[],costs={mode:dict(fit_seconds=1.,initialization_seconds=2.,integration_seconds=0.) for mode in models})
    dump(run.PARENT_OUTPUT/'execution.partial.json.tmp',latest);dump(run.PARENT_OUTPUT/'execution.partial.json',latest)
    dump(run.PARENT_OUTPUT/'started.json',{'action_id':'parent'})
    dump(run.PARENT_PRIVATE/'inputs.json',dict(packets=packets,receipt=dict(legal_ids=ids,ordered_all_ids=ids,invalid=[],adaptation_indices=[0])))
    dump(run.PARENT_PRIVATE/'preparation.json',dict(fit_ids=ids))
    dump(run.PARENT_PRIVATE/'constants.json',dict(scales={'p_scale':[1.]*10}))
    for i,eid in enumerate(ids[:41]):
        artifact=None
        if i<8:
            path=run.PARENT_PRIVATE/'initializations'/f'static_{i:03d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,state=np.zeros(50,dtype=np.float64));artifact=dict(path=path.relative_to(tmp_path).as_posix(),sha256=run.original.digest(path))
        dump(run.PARENT_PRIVATE/'initializations'/f'static_{i:03d}.json',dict(event_id=eid,
            diagnostics=dict(status='completed' if i<8 else 'blocked_by_stage1',counts=run.EXPECTED_INIT.copy() if i==0 else {k:0 for k in run.EXPECTED_INIT}),state=artifact))
    monkeypatch.setattr(run.GFTDKuramotoModel,'load',lambda path:models[Path(path).name.split('_')[0]])
    for name in ('REVIEW','ADOPTION','HOST'):
        path=tmp_path/getattr(run,name+'_PATH');dump(path,{'fixture':name})
        monkeypatch.setattr(run,name+'_SHA256',run.original.digest(path))
    source_paths=['experiments/parent.py','experiments/gftd_fit297_20261107_resume01.py','tests/test_gftd_fit297_resume01.py']
    for name in source_paths:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture source')
    for name in (run.original.PROTOCOL,run.original.REVIEW):dump(tmp_path/name,{'parent_frozen':name})
    for name in ('design_adoption.json','parent_archive_verified.json'):dump(run.PARENT_OUTPUT/name,{'parent_frozen':name})
    dump(run.PARENT_PRIVATE/'reference_asset.json',{'asset':True})
    parent=dict(asset_sha256={(run.PARENT_PRIVATE/'reference_asset.json').relative_to(tmp_path).as_posix():run.original.digest(run.PARENT_PRIVATE/'reference_asset.json')},
        protocol_sha256=run.original.digest(tmp_path/run.original.PROTOCOL),design_review_sha256=run.original.digest(tmp_path/run.original.REVIEW),
        design_adoption_sha256=run.original.digest(run.PARENT_OUTPUT/'design_adoption.json'),parent_archive_sha256=run.original.digest(run.PARENT_OUTPUT/'parent_archive_verified.json'),
        released=True,action_id='parent',source_sha256={source_paths[0]:run.original.digest(tmp_path/source_paths[0])})
    dump(run.PARENT_OUTPUT/'execution_release.json',parent)
    assets={p.relative_to(tmp_path).as_posix():run.original.digest(p) for folder in (run.PARENT_OUTPUT,run.PARENT_PRIVATE) for p in folder.rglob('*') if p.is_file()}
    assets.update({name:run.original.digest(tmp_path/name) for name in (run.original.PROTOCOL,run.original.REVIEW)})
    release=dict(released=True,action_id='resume',parent_job_ended=True,environment={'fixture':True},
        parent_release_sha256=run.original.digest(run.PARENT_OUTPUT/'execution_release.json'),
        parent_artifact_sha256=assets,source_sha256={p:run.original.digest(tmp_path/p) for p in source_paths},
        design_review_sha256=run.REVIEW_SHA256,implementation_adoption_sha256=run.ADOPTION_SHA256,parent_host_receipt_sha256=run.HOST_SHA256)
    monkeypatch.setattr(run.original,'environment',lambda :{'fixture':True})
    dump(run.OUTPUT/'execution_release.json',release)
    return SimpleNamespace(models=models,release=release,ids=ids,packets=packets,latest=latest)


def forbid(*args,**kwargs):raise AssertionError('forbidden scientific repeat')


def test_restore_exact41_states_counts_and553_order_without_science(stored,monkeypatch):
    monkeypatch.setattr(run.fit_engine,'fit_shared',forbid);monkeypatch.setattr(run.fit_engine,'_prepare',forbid)
    monkeypatch.setattr(run.original.core,'prepare_weak_history',forbid);monkeypatch.setattr(np.linalg,'svd',forbid)
    assert run.verify_release(stored.release)['action_id']=='parent'
    saved=run.restore(stored.release)
    assert saved['remaining']==[('static',i) for i in range(41,297)]+[('modulated',i) for i in range(297)]
    assert len(saved['done'])==41 and sum(s is not None for s in saved['states']['static'])==8
    assert saved['states']['static'][8] is None
    assert saved['latest']['counts']['initialization']==run.EXPECTED_INIT
    assert run.fit_engine._LEDGERS['fixture']==run.EXPECTED_LEDGER and not run.fit_engine._CACHE
    # The real frozen helper's cache-unavailable path must not integrate or decompose.
    monkeypatch.setattr(run.fit_engine,'validate_config',lambda f,c:c);monkeypatch.setattr(run.fit_engine,'_key',lambda *args:'fixture')
    coverage=run.fit_engine.coverage_information(SimpleNamespace(n_nodes=10),[],[],{'fit_request':{},'preparation_key':'fixture'})
    assert coverage['status']=='unknown' and coverage['reason']=='matching preparation cache unavailable'
    assert not any(coverage['counts'].values())


@pytest.mark.parametrize('change',['source','artifact','review','release','missing','parent_asset_missing','parent_asset_hash','parent_protocol_missing'])
def test_release_hash_bindings_reject_change(stored,change):
    release=copy.deepcopy(stored.release)
    if change=='source':next(iter((run.ROOT/'experiments').glob('parent.py'))).write_bytes(b'changed')
    if change=='artifact':(run.PARENT_PRIVATE/'inputs.json').write_bytes(b'changed')
    if change=='review':release['design_review_sha256']='0'*64
    if change=='release':release['parent_release_sha256']='0'*64
    if change=='missing':release['parent_artifact_sha256'].pop(next(iter(release['parent_artifact_sha256'])))
    if change=='parent_asset_missing':release['parent_artifact_sha256'].pop((run.PARENT_PRIVATE/'reference_asset.json').relative_to(run.ROOT).as_posix())
    if change=='parent_asset_hash':release['parent_artifact_sha256'][(run.PARENT_PRIVATE/'reference_asset.json').relative_to(run.ROOT).as_posix()]='0'*64
    if change=='parent_protocol_missing':release['parent_artifact_sha256'].pop(run.original.PROTOCOL)
    with pytest.raises(ValueError):run.verify_release(release)


@pytest.mark.parametrize('change',['count','state','missing'])
def test_restore_rejects_nonexact_boundary(stored,change):
    if change=='count':
        path=run.PARENT_OUTPUT/'execution.partial.json.tmp';data=run.original.read_json(path);data['counts']['initialization']['actual_residual_calls']+=1;dump(path,data)
    if change=='state':
        path=run.PARENT_PRIVATE/'initializations/static_000.npz';np.savez(path,state=np.ones(50))
    if change=='missing':(run.PARENT_PRIVATE/'initializations/static_040.json').unlink()
    with pytest.raises(ValueError):run.restore(stored.release)


def test_replace_retries_only_identical_bytes_and_never_recomputes(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run.time,'sleep',lambda t:None)
    replace=run.os.replace;seen=[]
    def transient(src,dst):
        seen.append((str(src),Path(src).read_bytes()))
        if len(seen)<3:raise PermissionError('mock transient lock')
        replace(src,dst)
    monkeypatch.setattr(run.os,'replace',transient)
    run.atomic_bytes(tmp_path/'result.json',b'one already computed result')
    assert len(seen)==3 and len(set(seen))==1
    assert (tmp_path/'result.json').read_bytes()==seen[0][1]


def test_terminal_replace_is_bounded_and_preserves_pending_bytes(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'REPLACE_ATTEMPTS',3);monkeypatch.setattr(run.time,'sleep',lambda t:None)
    attempts=[]
    def locked(src,dst):attempts.append(src);raise PermissionError('mock permanent lock')
    monkeypatch.setattr(run.os,'replace',locked)
    with pytest.raises(PermissionError):run.atomic_bytes(tmp_path/'result.json',b'computed once')
    assert len(attempts)==3 and len(set(attempts))==1
    assert Path(attempts[0]).read_bytes()==b'computed once'


def test_duplicate_claim_rejected_before_restore(stored,monkeypatch):
    dump(run.OUTPUT/'started.json',{'unknown_existing_job':True})
    monkeypatch.setattr(run,'restore',forbid)
    with pytest.raises(FileExistsError):run.run()


def test_failure_after_initializer_keeps_cost_and_original_error_when_final_io_also_fails(stored,monkeypatch):
    calls=[]
    def initialize(packet):
        calls.append(True);model.last_initialization={'counts':{'optimization_tasks':1,'actual_residual_calls':2,'rk4_steps_attempted':80}}
        return np.zeros(50)
    model=stored.models['static'];model.initialize=initialize
    def bad_npz(*args,**kwargs):raise OSError('primary state output error')
    monkeypatch.setattr(run,'write_npz',bad_npz)
    original_write=run.write_json
    def bad_final(path,data):
        if data.get('status')=='INTERRUPTED_RUNTIME':raise PermissionError('secondary final receipt error')
        return original_write(path,data)
    monkeypatch.setattr(run,'write_json',bad_final)
    with pytest.raises(OSError,match='primary state output error') as exc:run.run()
    assert len(calls)==1
    emergency=list(run.OUTPUT.glob('emergency-*.json'));assert len(emergency)==1
    row=run.original.read_json(emergency[0])
    assert row['counts']['initialization']['actual_residual_calls']==9349
    assert row['counts']['initialization']['rk4_steps_attempted']==396100
    assert row['counts']['initializations_attempted']==42
    assert 'secondary final' in row['final_persistence_error']['message']
    assert any('final persistence failed' in note for note in exc.value.__notes__)


def test_checkpoint50_values_roundtrip_without_fit(tmp_path):
    from kiki.models.gftd_kuramoto import GFTDKuramotoModel
    model=GFTDKuramotoModel(n_nodes=10);path=tmp_path/'model.json';model.save(path)
    restored=model.load(path)
    state=np.linspace(-.01,.01,50)
    np.testing.assert_array_equal(restored.functional.rhs(state),model.functional.rhs(state))
    np.testing.assert_array_equal(restored.functional.pack_state(*restored.functional.split_state(state)),state)


def test_full_remaining553_sequence_freeze_and_inherited_budget_without_science(stored,monkeypatch):
    import closure_probe_20261030 as loader
    import gftd_fit297_analysis_20261107 as analysis
    sequence=[];frozen=[]
    for mode,model in stored.models.items():
        def initialize(packet,mode=mode):
            index=len([m for m,_ in sequence if m==mode])+(41 if mode=='static' else 0)
            sequence.append((mode,index))
            raise run.engine.InitializationFailure('fixture failure',{'status':'blocked_by_stage1',
                'counts':{'optimization_tasks':1,'actual_residual_calls':1,'rk4_steps_attempted':40,'rk4_steps_completed':40}})
        model.initialize=initialize
    monkeypatch.setattr(run.fit_engine,'fit_shared',forbid);monkeypatch.setattr(run.fit_engine,'_prepare',forbid)
    monkeypatch.setattr(run.original.core,'prepare_weak_history',forbid);monkeypatch.setattr(np.linalg,'svd',forbid)
    monkeypatch.setattr(run.fit_engine,'validate_config',lambda f,c:c);monkeypatch.setattr(run.fit_engine,'_key',lambda *args:'fixture')
    # Cache-unavailable returns before touching states, regardless of this stub's n_nodes.
    for model in stored.models.values():model.functional=SimpleNamespace(n_nodes=10)
    events=[dict(id=eid,label='stable' if i<267 else 'unstable',group='fixture') for i,eid in enumerate(stored.ids)]
    monkeypatch.setattr(loader,'load_frozen_events',lambda :(dict(fit_ids=stored.ids),events))
    def integrate(f,state,step,counts):
        assert len(sequence)==553 and frozen
        counts['rk4_attempted']+=round(15/step);counts['rk4_completed']+=round(15/step)
        return np.zeros((1,50)),None
    monkeypatch.setattr(run.original,'integrate',integrate)
    monkeypatch.setattr(run.original.l1,'score_event',lambda event,*args,**kwargs:dict(event_id=event['id'],label=event['label'],group='fixture',scopes={},failure=None))
    for name in ('l2_certificate','l2_point','rom_modes','kron_reference','fom_modes','match_modes','graph_correspondence'):
        monkeypatch.setattr(analysis,name,lambda *args,**kwargs:dict(status='fixture'))
    def fast_write(path,data):
        if Path(path).name=='all_initializations_frozen.json':frozen.append(copy.deepcopy(data))
        if Path(path).name!='execution.partial.json':dump(Path(path),run.original.safe(data))
        return dict(path=Path(path).relative_to(run.ROOT).as_posix(),sha256='fixture')
    monkeypatch.setattr(run,'write_json',fast_write)
    monkeypatch.setattr(run,'write_npz',lambda path,**arrays:dict(path=Path(path).relative_to(run.ROOT).as_posix(),sha256='fixture'))
    receipt=run.run()
    assert receipt['status']=='COMPLETED'
    assert receipt['combined_host_seconds'] is None and receipt['original_runner_wall_seconds'] is None
    assert receipt['costs']['static']['same_precision_conditions']==dict(device='cpu',dtype='float64',future_steps=[.01,.005],duration_s=15.,batch=1)
    assert receipt['coverage_information']['static']['status']=='unknown'
    assert sequence==[('static',i) for i in range(41,297)]+[('modulated',i) for i in range(297)]
    assert receipt['counts']['initializations_attempted']==594 and receipt['counts']['initializations_completed']==8
    assert receipt['counts']['initialization']['optimization_tasks']==49+553
    assert receipt['counts']['initialization']['actual_residual_calls']==9347+553
    assert receipt['counts']['initialization']['rk4_steps_attempted']==396020+553*40
    assert receipt['counts']['future_pairs_attempted']==8 and receipt['counts']['rk4_attempted']==8*4500
    assert receipt['remaining_work_budget']['new_tasks']==553
    assert receipt['analysis_budget']['information_svd']['actual']==416
    assert receipt['analysis_budget']['conditional_projections']['actual']==315
    assert receipt['analysis_budget']['memory_svd']['actual']==594
    assert sum(frozen[0]['success']['static'])==8 and not any(frozen[0]['success']['modulated'])
    coverage=run.original.read_json(run.PRIVATE/'static_coverage_information.json')
    assert coverage['status']=='unknown' and not any(coverage['counts'].values())
    assert not coverage['interruption_provenance']['cache_rebuilt']


def test_preclaim_defers_raw_fit_cache_bytes_until_scoring_claim(stored,monkeypatch):
    cache=run.ROOT/'private/raw/fit_events.npz';cache.parent.mkdir(parents=True);cache.write_bytes(b'raw event cache fixture')
    stored.release['parent_artifact_sha256'][cache.relative_to(run.ROOT).as_posix()]=hashlib.sha256(cache.read_bytes()).hexdigest()
    digest=run.original.digest
    def guarded(path):
        if Path(path)==cache:raise AssertionError('preclaim event cache read')
        return digest(path)
    monkeypatch.setattr(run.original,'digest',guarded)
    assert run.verify_release(stored.release)['action_id']=='parent'


@pytest.mark.parametrize('key,value',[('optimization_tasks',1107),('actual_residual_calls',995401)])
def test_unused_closed_opportunities_cannot_expand_repair_budget(stored,monkeypatch,key,value):
    calls=[];model=stored.models['static']
    def initialize(packet):
        calls.append(True);model.last_initialization={'counts':{key:value}}
        return np.zeros(50)
    model.initialize=initialize
    with pytest.raises(RuntimeError,match='remaining original opportunities'):run.run()
    receipt=run.original.read_json(run.OUTPUT/'execution.json')
    assert len(calls)==1
    assert receipt['counts']['initialization'][key]==run.EXPECTED_INIT.get(key,0)+value
