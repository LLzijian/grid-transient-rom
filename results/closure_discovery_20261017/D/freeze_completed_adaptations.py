"""Root verification of all registered D adaptations, then immutable freeze; train only."""
from pathlib import Path
import hashlib,json,math,sys,time,datetime
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import torch
import discover_closure as dc
import closure_discovery_confirmation as conf
import closure_discovery_data as data
from closure_discovery_support import ProcessLedger
OUT=ROOT/'private_data/closure_discovery_20261017'; D=OUT/'D'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def main():
 torch.set_num_threads(1)
 ledger=ProcessLedger(OUT/'ledger')
 with ledger.run('D','root-verify-and-freeze-fourteen-adaptations'):
  assert dc.bind_execution_sources(OUT)=='b8d1f961bd5ae7700b8c656b28357b0ad537a77bd838fda5530c4c0c0f4694bc'
  jobs=[read(p) for p in D.glob('parallel_adapt_*_job.json')]
  assert len(jobs)==4 and all(j['status']=='completed' and not j.get('unresolved') for j in jobs)
  rows=[r for j in jobs for r in j['completed']]; by={(r['name'],r['seed']):r for r in rows}
  frozen=read(OUT/'C/frozen.json'); assert len(rows)==len(by)==14
  old,excluded_old,force_scale=dc.train_inputs(); new,excluded_new=data.load_new_events(D/'new_data/train','train')
  assert len(old)==346 and len(new)==20 and len(excluded_new)==4
  events=old+new; ids={str(e['id']) for e in events}; expected_cohort=conf._cohort_identity(events)
  endpoints=[]; checks=[]
  for source in frozen['models']:
   row=by[(source['name'],source['seed'])]; folder=conf._model_file(row).parent
   c=read(folder/'checkpoint.json'); identity=read(folder/'fit_identity.json'); payload=read(folder/'model.json')
   assert c['step']==256 and [v['step'] for v in c['trace']]==list(range(1,257))
   assert sha(folder/'model.json')==c['model_sha256']==row['sha256']==sha(folder/'model_at_step00256.json')
   assert sha(folder/'training.pt')==c['training_sha256']
   assert identity['seed']==source['seed'] and identity['target_updates']==256
   assert identity['source_model_sha256']==source['sha256'] and identity['train_cohort']==expected_cohort
   assert identity['force_scale']==force_scale and identity['terms_sha256']==conf._json_sha256(payload['terms'])
   source_payload=read(conf._model_file(source))
   assert payload['terms']==source_payload['terms']
   assert payload['state_dict']['template_mask']==source_payload['state_dict']['template_mask']==identity['direction_mask']
   for field in ('mass','D','delta_star','pm_star','p_scale','input_scale'):
    if field in payload['physical']: assert payload['physical'][field]==source_payload['physical'][field]
   assert payload['provenance']['implementation_identity_sha256']=='b8d1f961bd5ae7700b8c656b28357b0ad537a77bd838fda5530c4c0c0f4694bc'
   assert payload['provenance']['completed_updates_in_current_fit']==256
   used={str(x) for tr in c['trace'] for x in tr['sample_ids']}; assert used==ids
   assert all(math.isfinite(tr['loss']) and math.isfinite(tr['risk']) for tr in c['trace'])
   assert not (folder/'fit_failure.json').exists() and not (folder/'candidate_failure.json').exists()
   qa=read(folder/'endpoint_physics.json'); assert qa['pass'] is True
   state=torch.load(folder/'training.pt',map_location='cpu',weights_only=False)
   assert state['step']==256 and set(state['coverage']['cover_order'])==set(range(366))
   optsteps=[float(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
   assert optsteps and max(optsteps)<=256
   checks.append({'name':row['name'],'seed':row['seed'],'model_sha256':row['sha256'],'training_sha256':c['training_sha256'],'source_model_sha256':source['sha256'],'fit_identity_sha256':sha(folder/'fit_identity.json'),'endpoint_physics_sha256':sha(folder/'endpoint_physics.json'),'step':256,'covered_events':len(used),'optimizer_max_step':max(optsteps),'qa_pass':True,'counts':payload['counts']})
   endpoints.append(row)
  conf._verify_model_rows(endpoints,frozen['models'],OUT)
  manifest=read(OUT/'new_fault_manifest.json')
  freeze,reused=conf._freeze_or_reuse(D/'confirmation_freeze.json',endpoints,manifest['val24'],OUT/'new_fault_manifest.json',source_models=frozen['models'])
  result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_identity_sha256':sha(OUT/'implementation/identity.json'),'execution_protocol_sha256':sha(ROOT/'reports/closure_discovery_20261017_protocol.md'),'script_sha256':sha(__file__),'C_frozen_sha256':sha(OUT/'C/frozen.json'),'freeze_sha256':sha(D/'confirmation_freeze.json'),'freeze_reused':reused,'models':checks,'cohort':expected_cohort,'old_train':346,'qualified_new_train':20,'new_train_excluded':excluded_new,'main_library_same_seed_byte_identical':{str(seed):by[('main',seed)]['sha256']==by[('library',seed)]['sha256'] for seed in (2701,2702,2703)},'val_generated_before_freeze':(D/'new_data/val/generation.json').exists(),'test_read':False}
  assert not result['val_generated_before_freeze']
  dc.write(D/'adaptation_freeze_verification.json',result)
  print(json.dumps({'models':14,'freeze_sha256':result['freeze_sha256'],'same_seed_identity':result['main_library_same_seed_byte_identical']}))
if __name__=='__main__': main()
