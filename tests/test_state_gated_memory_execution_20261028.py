"""Metadata-only controller checks, no native Jobs or CUDA."""
import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import state_gated_memory_execution_20261028 as c

def test_finite_new_scope_and_source_separation():
 assert 'torch' not in sys.modules
 assert c.TOTAL_LIMIT==1464 and c.NORMAL_CATEGORY_LIMITS=={'qa':108,'formal':1152,'repair':0,'replay':0}
 assert c.RESERVED_CATEGORY_LIMITS=={'repair':108,'replay':96} and 'N6101_M_28102' in c.R_N_SOURCE
 assert c.PHASES==('prepare','qualify','qa','study','summary')
 assert 'A512_M' not in c.R_N_SOURCE

def test_new_base_not_alias(tmp_path,monkeypatch):
 monkeypatch.setattr(c,'ROOT',tmp_path);base=tmp_path/'private_data/state_gated_memory_20261028/study01';assert c.checked_base(base)==base
 with pytest.raises(ValueError):c.checked_base(tmp_path/'private_data/force_exchange_20261028/study01')

def test_pending_count_and_disabled_reserved(tmp_path):
 p=tmp_path/'optimizer_calls.jsonl';row={'call_id':'MOCK','event':'optimizer_step_attempt','category':'formal','run_id':'B_28311','step':1};p.write_text(json.dumps(row)+'\n')
 assert len(c.call_counts(tmp_path)['pending_attempts'])==1
 row['category']='repair';p.write_text(json.dumps(row)+'\n')
 with pytest.raises(ValueError):c.call_counts(tmp_path)

def test_pipeline_finite_and_no_retry(tmp_path,monkeypatch):
 monkeypatch.setattr(c,'checked_base',lambda b:tmp_path);calls=[]
 monkeypatch.setattr(c,'freeze',lambda *a:calls.append('freeze'))
 def launch(b,commit,phase):
  calls.append(phase)
  if phase=='qa':raise ValueError('MOCK unresolved actual call')
 monkeypatch.setattr(c,'launch',launch)
 with pytest.raises(ValueError):c.run(tmp_path,'1'*40)
 assert calls==['freeze','prepare','qualify','qa']


def mock_phase(tmp_path,name,result,monkeypatch):
 def write(p,r):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r))
 monkeypatch.setattr(c,'clean_receipt',lambda r,*a:r)
 write(tmp_path/'binding.json',{'files':{'MOCK_SOURCE':'MOCK_SHA'}})
 out=tmp_path/c.PHASE_OUTPUTS[name];write(out,result)
 admission=tmp_path/'ownership'/name/'admission.json';inputs={str(tmp_path/'binding.json'):c.sha(tmp_path/'binding.json')}
 write(admission,{'source_commit':'1'*40,'phase':name,'inputs':inputs})
 write(tmp_path/'native'/name/'receipt.json',{'source_before':{'MOCK_SOURCE':'MOCK_SHA',**inputs,str(admission):c.sha(admission)},'MOCK_ONLY':True})
 write(tmp_path/'ownership'/name/'complete.json',{'status':'COMPLETED','source_commit':'1'*40,'phase':name,'scientific_output_sha256':c.sha(out)})

def test_missing_A_or_B_does_not_globally_block_other_valid_qualifications(tmp_path,monkeypatch):
 variants={name:{'training_admission':name=='C','cpu_qualified':name=='C','cpu_cuda_pair_pass':name=='C','finite_12x8_backward':name=='C','fullfit297_qualified':name=='C'} for name in ('A','B','C')}
 result={'status':'COMPLETED','training_method':'rk4','eligible_variants':['C'],'per_variant':variants,'history_dt_main':.0025,'history_dt_reference':.00125}
 mock_phase(tmp_path,'qualify',result,monkeypatch);c._completed_phase(tmp_path,'1'*40,'qualify')
 result['per_variant']['C']['finite_12x8_backward']=False;mock_phase(tmp_path,'qualify',result,monkeypatch)
 with pytest.raises(ValueError,match='gate'):c._completed_phase(tmp_path,'1'*40,'qualify')

def test_partial_real_QA_isolation_never_fabricates108_calls(tmp_path,monkeypatch):
 result={'status':'COMPLETED','actual_optimizer_updates':36,'eligible_variants':['C'],'models':[{'variant':'B','status':'UNRESOLVED'},{'variant':'C','status':'PASS','actual_optimizer_calls':36}]}
 mock_phase(tmp_path,'qa',result,monkeypatch);c._completed_phase(tmp_path,'1'*40,'qa')
 result['models'][1]['actual_optimizer_calls']=35;mock_phase(tmp_path,'qa',result,monkeypatch)
 with pytest.raises(ValueError,match='actual36'):c._completed_phase(tmp_path,'1'*40,'qa')
