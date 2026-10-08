import importlib.util, json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location('extension_audit',ROOT/'experiments'/'check_structure_extension_0078.py'); audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
def _cp(p,step,s,o,m):
 p.mkdir(parents=True);torch.save(s,p/'model.pt');torch.save(o,p/'optimizer.pt');torch.save({'fit_numpy':{'x':1},'selection_numpy':{'x':2}},p/'rng_state.pt');(p/'checkpoint.json').write_text(json.dumps({**m,'step':step}))
def _fixture(t,arm='C-free'):
 m={'arm':arm,'requires_grad':{'train':True,'frozen':False},'optimizer_parameter_names':[['train']]};s={'train':torch.tensor([1.]),'frozen':torch.tensor([2.])};oi={'param_groups':[{'params':[1],'lr':1e-3}],'state':{1:{'step':torch.tensor(2)}}};of={'param_groups':[{'params':[1],'lr':1e-3}],'state':{1:{'step':torch.tensor(1802)}}};_cp(t/'initial',1200,s,oi,m);_cp(t/'resume',1200,s,oi,m);_cp(t/'final',3000,{'train':torch.tensor([3.]),'frozen':torch.tensor([2.])},of,m)
 tr=[{'step':x,'horizon_s':audit.extension_horizon(x),'loss':1.,'gradient_norm_before_cap':1.,'gradient_norm_clip_return':1.,'terms':{'complexity':0.,'x':1.},'checks':{'f0':0.,'probe_finite':True,'probe_nonzero':1.,'kinematic_max_abs':0.}} for x in range(1201,3001)]
 e={'arm':arm,'steps_from':1200,'steps_to':3000,'checkpoint':str((t/'initial').resolve()),'final_checkpoint':str((t/'final').resolve()),'traces':tr,'complexity_weight':0.,'hashes_before':{'a':'b'},'hashes_after':{'a':'b'},'monitor3000':{}};(t/'e.json').write_text(json.dumps(e));return m,e
def _audit(t):
 old=audit._monitor_ok;audit._monitor_ok=lambda _:True
 try:return audit.audit(t/'initial',t/'resume',t/'final',t/'e.json')
 finally:audit._monitor_ok=old
def test_complete_c_fixture_passes():
 from tempfile import TemporaryDirectory
 with TemporaryDirectory() as x:
  t=Path(x);_fixture(t)
  for p in ('initial','resume','final'):
   s=torch.load(t/p/'model.pt',weights_only=True);s['xi']=torch.ones(1);torch.save(s,t/p/'model.pt')
  assert _audit(t)['passed']
def test_g_xi_and_missing_trainable_optimizer_fail(tmp_path):
 _fixture(tmp_path,'G-auto');
 for p in ('initial','resume','final'):
  s=torch.load(tmp_path/p/'model.pt',weights_only=True);s['xi']=torch.ones(1);torch.save(s,tmp_path/p/'model.pt')
 assert not _audit(tmp_path)['checks']['no_xi']
 _fixture(tmp_path/'m');q=tmp_path/'m/final/checkpoint.json';d=json.loads(q.read_text());d['requires_grad']['missing']=True;q.write_text(json.dumps(d));assert not _audit(tmp_path/'m')['checks']['adam_trainable_advanced_1800']
def test_lr_frozen_resume_and_trace_fail_closed(tmp_path):
 _fixture(tmp_path);o=torch.load(tmp_path/'final/optimizer.pt',weights_only=True);o['param_groups'][0]['lr']=2e-3;torch.save(o,tmp_path/'final/optimizer.pt');assert not _audit(tmp_path)['checks']['optimizer_layout_unchanged']
 _fixture(tmp_path/'b');s=torch.load(tmp_path/'b/final/model.pt',weights_only=True);del s['frozen'];torch.save(s,tmp_path/'b/final/model.pt');assert not _audit(tmp_path/'b')['checks']['frozen_state_exact']
 _fixture(tmp_path/'r');q=tmp_path/'r/e.json';e=json.loads(q.read_text());e['traces'][0]['terms']['x']=float('inf');q.write_text(json.dumps(e));assert not _audit(tmp_path/'r')['checks']['trace_finite']

def test_resume_and_incomplete_trace_rejected(tmp_path):
 _fixture(tmp_path);s=torch.load(tmp_path/'resume/model.pt',weights_only=True);s['train']+=1;torch.save(s,tmp_path/'resume/model.pt');assert not _audit(tmp_path)['checks']['original_resume_model_exact']
 q=tmp_path/'e.json';e=json.loads(q.read_text());e['traces'].pop();q.write_text(json.dumps(e));assert not _audit(tmp_path)['checks']['trace_steps']

def test_trace_requires_real_finite_numbers(tmp_path):
 _,e=_fixture(tmp_path)
 e['traces'][0]['loss']=None
 assert not audit._trace_checks(e)['trace_finite']
 e['traces'][0]['loss']=float('nan')
 assert not audit._trace_checks(e)['trace_finite']
