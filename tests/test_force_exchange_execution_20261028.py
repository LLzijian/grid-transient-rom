"""Data-free native controller tests. All receipts and ledger rows are MOCK_ONLY."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import force_exchange_execution_20261028 as c
COMMIT="1"*40

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding="utf-8")

def qualification(**changes):
    return {"status":"COMPLETED","training_admission":True,"training_method":"rk4",**dict.fromkeys(c.QUALIFICATION_GATES,True),**changes}

def mock_cross(base):
    prepared=base/'prepare.json'
    if not prepared.exists():write(prepared,{'fit_ids':[f'MOCK_fit{i}' for i in range(297)],'source':{'MOCK_ONLY':True}})
    p=c.read(prepared);path=(base/'qualification/F_C0_cross_model.json').resolve()
    rows=[{'event_id':eid,'actual_steps':100,'qualified':True,'intervals':{
        k:{'qualified':True,'actual_output_steps':800 if k=='8s' else 100,
           'full_F_fine_finite':True,'full_C0_fine_finite':True} for k in ('8s','actual15')}} for eid in p['fit_ids']]
    write(path,{'MOCK_ONLY':True,'status':'PASS','qualified':True,'record_count':297,'event_ids':p['fit_ids'],
        'source':p['source'],'scope':c.CROSS_MODEL_SCOPE,'bidirectional':True,'predictions_recomputed':False,
        'alignment':{'fine_dt':.005,'output_dt':.01,'stride':2,'coarse_stride':2,'history_rows':11,
          'history_dt':.01,'initial_observation_index':10,'eight_second_shapes':[1601,801]},
        'aggregates':{k:{'event_count':297,'qualified_count':297} for k in ('8s','actual15')},'rows':rows})
    return {'cross_model_F_C0_qualified':True,'cross_model_F_C0':{'path':str(path),'sha256':c.sha(path)}}

def phase(base,name,result,monkeypatch):
    monkeypatch.setattr(c,"clean_receipt",lambda r,commit,p:r)
    write(base/"binding.json",{"files":{"MOCK_SOURCE":"MOCK_SHA"}})
    if name=='prepare':result={**result,'fit_ids':[f'MOCK_fit{i}' for i in range(297)],'source':{'MOCK_ONLY':True}}
    if name=='qualify':result={**result,**mock_cross(base)}
    out=base/c.PHASE_OUTPUTS[name];write(out,result)
    admission=base/"ownership"/name/"admission.json"
    inputs={str(base/"binding.json"):c.sha(base/"binding.json")}
    write(admission,{"source_commit":COMMIT,"phase":name,"inputs":inputs})
    write(base/"native"/name/"receipt.json",{"MOCK_ONLY":True,"source_before":{"MOCK_SOURCE":"MOCK_SHA",**inputs,str(admission):c.sha(admission)}})
    write(base/"ownership"/name/"complete.json",{"status":"COMPLETED","source_commit":COMMIT,"phase":name,"scientific_output_sha256":c.sha(out)})

def journal(base,category,n):
    p=base/"optimizer_calls.jsonl"
    with p.open("a") as f:
        for i in range(n):
            cid=f"MOCK_{category}_{i}"
            for event in ("optimizer_step_attempt","optimizer_step_completed"):
                f.write(json.dumps({"MOCK_ONLY":True,"event":event,"category":category,"run_id":f"MOCK_{category}","step":i+1,"call_id":cid})+"\n")

def test_inert_import_finite_authority_and_no_calibration():
    assert "torch" not in sys.modules
    assert c.PHASES==("prepare","qualify","qa","study","summary")
    assert c.NORMAL_CATEGORY_LIMITS=={"qa":72,"formal":768,"repair":0,"replay":0}
    assert c.RESERVED_CATEGORY_LIMITS=={"repair":72,"replay":64}
    assert sum(c.NORMAL_CATEGORY_LIMITS.values())+sum(c.RESERVED_CATEGORY_LIMITS.values())==976
    assert c.commit_id(COMMIT)==COMMIT
    for v in (None,"main","a"*39,"Z"*40):
        with pytest.raises(ValueError):c.commit_id(v)

def test_only_fixed_new_base(tmp_path,monkeypatch):
    monkeypatch.setattr(c,"ROOT",tmp_path)
    base=tmp_path/"private_data/force_exchange_20261028/study01"
    assert c.checked_base(base)==base
    for p in (base.parent/"study02",tmp_path/"private_data/force_graph_20261028/study01"):
        with pytest.raises(ValueError):c.checked_base(p)

def test_historical_lock_checks_are_reused_and_new_owned_lock_is_exact(tmp_path,monkeypatch):
    monkeypatch.setattr(c,"ROOT",tmp_path)
    base=tmp_path/"private_data/force_exchange_20261028/study01"
    calls=[];monkeypatch.setattr(c.parent,"check_locks",lambda parent,**kwargs:calls.append((parent,kwargs)) or [{"MOCK_ONLY":True}])
    assert c.check_locks(base,state="MOCK_STATE")==[{"MOCK_ONLY":True}]
    assert calls==[(c.PARENT,{"state":"MOCK_STATE"})]
    own={"pid":123,"creation_filetime_utc":134000000000000001}
    write(base/"active.lock",{"worker":own});before=(base/"active.lock").read_bytes()
    with pytest.raises(ValueError):c.check_locks(base)
    assert c.check_locks(base,owned_identity=own)
    with pytest.raises(ValueError):c.check_locks(base,owned_identity={**own,"pid":124})
    assert (base/"active.lock").read_bytes()==before

@pytest.mark.parametrize("gate",c.QUALIFICATION_GATES)
def test_each_actual_scientific_qualification_gate_required(tmp_path,monkeypatch,gate):
    phase(tmp_path,"qualify",qualification(),monkeypatch)
    c._completed_phase(tmp_path,COMMIT,"qualify")
    write(tmp_path/c.PHASE_OUTPUTS['qualify'],qualification(**{gate:False}))
    complete=tmp_path/"ownership/qualify/complete.json";r=c.read(complete);r['scientific_output_sha256']=c.sha(tmp_path/c.PHASE_OUTPUTS['qualify']);write(complete,r)
    with pytest.raises(ValueError,match="gates"):c._completed_phase(tmp_path,COMMIT,"qualify")

@pytest.mark.parametrize("change",({"training_admission":False},{"training_method":"fast"},{"status":"UNRESOLVED"}))
def test_matched_cpu_alone_and_parent_fast_are_not_admission(tmp_path,monkeypatch,change):
    phase(tmp_path,"qualify",qualification(**change),monkeypatch)
    with pytest.raises(ValueError):c._completed_phase(tmp_path,COMMIT,"qualify")

def test_actual72_qa_and_closed_split_gate(tmp_path,monkeypatch):
    phase(tmp_path,"prepare",{"val_read":False,"test_read":False},monkeypatch)
    phase(tmp_path,"qualify",qualification(),monkeypatch)
    phase(tmp_path,"qa",{"status":"PASS","actual_optimizer_updates":72},monkeypatch)
    with pytest.raises(ValueError,match="actual72"):c.phase_inputs(tmp_path,COMMIT,"study")
    journal(tmp_path,"qa",72)
    assert str(tmp_path/"qa/summary.json") in c.phase_inputs(tmp_path,COMMIT,"study")
    phase(tmp_path,"prepare",{"val_read":True,"test_read":False},monkeypatch)
    with pytest.raises(ValueError,match="val/test"):c.phase_inputs(tmp_path,COMMIT,"study")

def test_source_and_output_evidence_cannot_be_relabelled(tmp_path,monkeypatch):
    phase(tmp_path,"prepare",{"val_read":False,"test_read":False},monkeypatch)
    c._completed_phase(tmp_path,COMMIT,"prepare")
    write(tmp_path/"binding.json",{"files":{"MOCK_SOURCE":"CHANGED"}})
    with pytest.raises(ValueError):c._completed_phase(tmp_path,COMMIT,"prepare")

def test_counters_require_call_identity_and_preserve_pending(tmp_path):
    journal(tmp_path,"qa",72);journal(tmp_path,"formal",768)
    r=c.call_counts(tmp_path);assert r['attempted_total']==r['completed_total']==840 and r['pending_attempts']==[]
    with (tmp_path/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps({'event':'optimizer_step_attempt','category':'repair','run_id':'MOCK','step':1,'call_id':'MOCK_repair'})+'\n')
    with pytest.raises(ValueError,match="disabled"):c.call_counts(tmp_path)

def test_pending_call_blocks_training_but_summary_can_report_it(tmp_path,monkeypatch):
    for p,r in (("prepare",{'val_read':False,'test_read':False}),("qualify",qualification()),("qa",{'status':'PASS','actual_optimizer_updates':72}),("study",{'status':'UNRESOLVED'})):
        phase(tmp_path,p,r,monkeypatch)
    journal(tmp_path,'qa',72)
    row={'event':'optimizer_step_attempt','category':'formal','run_id':'MOCK_C0','step':17,'call_id':'MOCK_pending'}
    with (tmp_path/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    assert c.call_counts(tmp_path)['pending_attempts']==[{'category':'formal','run_id':'MOCK_C0','step':17,'call_id':'MOCK_pending'}]
    with pytest.raises(ValueError,match='reconciliation'):c.phase_inputs(tmp_path,COMMIT,'study')
    assert c.phase_inputs(tmp_path,COMMIT,'summary')
    row.update(event='optimizer_step_completed',step=18)
    with (tmp_path/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='identity/order'):c.call_counts(tmp_path)

@pytest.mark.parametrize('phase',c.PHASES)
def test_exact_phase_api_without_retry(monkeypatch,phase):
    calls=[];m=SimpleNamespace(**{p:lambda *a,_p=p,**kw:calls.append((_p,a,kw)) for p in c.PHASES})
    monkeypatch.setattr(c.importlib,'import_module',lambda name:m)
    c._study_call(Path('MOCK'),phase)
    assert calls==[(phase,(Path('MOCK'),),{} if phase in ('prepare','summary') else {'device':'cuda'})]

def test_run_is_finite_sequential_and_never_resumes(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'checked_base',lambda value:tmp_path)
    calls=[];monkeypatch.setattr(c,'freeze',lambda commit,base:calls.append('freeze'))
    def launch(base,commit,phase):
        calls.append(phase)
        if phase=='qa':raise ValueError('MOCK actual QA unresolved')
    monkeypatch.setattr(c,'launch',launch)
    with pytest.raises(ValueError):c.run(tmp_path,COMMIT)
    assert calls==['freeze','prepare','qualify','qa'] and not (tmp_path/'complete.json').exists()

def test_native_receipt_forbids_all_process_quota_fields(monkeypatch):
    monkeypatch.setattr(c.parent,'clean_receipt',lambda r,*args:r)
    assert c.clean_receipt({'MOCK_ONLY':True},COMMIT,'prepare')
    for k in ('runtime_limits','cpu_limits','active_process_limit'):
        with pytest.raises(ValueError,match='quotas'):c.clean_receipt({'MOCK_ONLY':True,k:1},COMMIT,'prepare')

def test_missing_or_false_cross_prefix_blocks_next_training_phase(tmp_path,monkeypatch):
    phase(tmp_path,'prepare',{'val_read':False,'test_read':False},monkeypatch)
    phase(tmp_path,'qualify',qualification(),monkeypatch)
    path=tmp_path/c.PHASE_OUTPUTS['qualify'];original=c.read(path)
    for missing in ('cross_model_F_C0_qualified','cross_model_F_C0'):
        result=dict(original);result.pop(missing);write(path,result)
        cp=tmp_path/'ownership/qualify/complete.json';completion=c.read(cp);completion['scientific_output_sha256']=c.sha(path);write(cp,completion)
        with pytest.raises(ValueError,match='cross-model'):c.phase_inputs(tmp_path,COMMIT,'qa')
    result={**original,'cross_model_F_C0_qualified':False};write(path,result)
    completion['scientific_output_sha256']=c.sha(path);write(cp,completion)
    with pytest.raises(ValueError,match='cross-model'):c.phase_inputs(tmp_path,COMMIT,'qa')

def test_cross_prefix_hash_count_source_and_interval_scope_are_bound(tmp_path):
    link=mock_cross(tmp_path);path=Path(link['cross_model_F_C0']['path']);original=c.read(path)
    assert str(path) in c.cross_model_evidence(tmp_path,link)
    for field,value in (('record_count',296),('scope','MOCK_wrong_scope'),('event_ids',list(reversed(original['event_ids']))),
                        ('source',{'MOCK_changed':True}),('bidirectional',False)):
        record={**original,field:value};write(path,record)
        current={**link,'cross_model_F_C0':{'path':str(path),'sha256':c.sha(path)}}
        with pytest.raises(ValueError,match='cross-model'):c.cross_model_evidence(tmp_path,current)
    write(path,original)
    with pytest.raises(ValueError,match='hash'):c.cross_model_evidence(tmp_path,{**link,'cross_model_F_C0':{'path':str(path),'sha256':'0'*64}})
    record=c.read(path);record['rows'][0]['intervals']['actual15']['actual_output_steps']=1501;write(path,record)
    with pytest.raises(ValueError,match='interval'):c.cross_model_evidence(tmp_path,{**link,'cross_model_F_C0':{'path':str(path),'sha256':c.sha(path)}})
