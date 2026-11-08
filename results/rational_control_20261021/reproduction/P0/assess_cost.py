"""Root's standard-library assessment of the one complete fixed cost panel."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--generation-tool-seconds',type=float,required=True);a=ap.parse_args()
 rp=BASE/'native_cost01/execution.json';r=read(rp);q=read(BASE/'native_qa01/execution.json');w=read(BASE/'cost01_outer_wall.json')
 assert r['status']=='COMPLETED' and r['phase_gate_pass'] is True and r['source_before']==r['source_after']
 assert r['source_commit']==q['source_commit'] and r['immutable_science_files']==q['immutable_science_files']
 for x in (r,q):
  assert x['inner']['exit_code']==x['outer']['exit_code']==0 and x['inner']['cleanup_verified'] is True
  assert x['inner']['fixture_identity_after']==x['inner']['supervised_identity_after']=='GONE'
 c=r['complete']['result'];assert c['status']=='COMPLETED' and c['cost_panels_complete'] is True
 counts={'update':384,'g_train':12,'g_val':12,'v_complete':12,'save':48,'restore':12,'construct':24,'load':12,'raw_write':36,'json_write':256}
 assert c['counts']==counts
 for k,n in counts.items():
  assert len(c['panels'][k])==n and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in c['panels'][k])
 ticks=(int(w['monotonic_end_ticks'])-int(w['monotonic_start_ticks']))/int(w['monotonic_frequency'])
 assert abs(ticks-w['elapsed_seconds'])<1e-8 and ticks>=r['charged_seconds'] and w['launcher_exit_code']==0
 assert a.generation_tool_seconds>0 and math.isfinite(a.generation_tool_seconds)
 body=c['body_wall_seconds'];measured=sum(map(sum,c['panels'].values()))
 assert ticks>=body and abs(measured-c['body_measured_seconds'])<1e-6
 residual=max(0.,body-measured);assert abs(residual-c['body_unassigned_seconds'])<1e-6
 startup=a.generation_tool_seconds+ticks-body+residual
 mx={k:max(v) for k,v in c['panels'].items()}
 run=1536*mx['update']+3*mx['g_train']+12*mx['save']+3*mx['restore']+6*mx['construct']+3*mx['raw_write']+48*mx['json_write']+startup
 evaluation=12*mx['g_val']+12*mx['v_complete']+12*mx['load']+24*mx['raw_write']+64*mx['json_write']+startup
 total=4*run+evaluation
 gates={'each_run_25pct_within_2750':1.25*run<=2750,'evaluation_25pct_within_650':1.25*evaluation<=650,'complete_with_25pct_and_fixed_350_within_12000':1.25*total+350<=12000}
 result={'status':'PASS' if all(gates.values()) else 'REJECT','P1_complete_cap_seconds':12000,'numerical_source_commit':r['source_commit'],'immutable_science_files':r['immutable_science_files'],
  'counts':counts,'max_seconds':mx,'startup_per_process_seconds':startup,'startup_processes':5,'body_unassigned_seconds':residual,'outer_elapsed_seconds':ticks,'generation_tool_seconds':a.generation_tool_seconds,
  'T_each_run_seconds':run,'T_evaluate_seconds':evaluation,'T_total_seconds':total,'complete_25pct_plus350_seconds':1.25*total+350,'gates':gates,'cost_receipt_sha256':sha(rp),'outer_wall_sha256':sha(BASE/'cost01_outer_wall.json'),
  'scope':'Prospective full12 fixed512 paired synthetic causal study only; original900 remains closed; metadata50 includes additional scientific asset SHA/freeze; no promotion.'}
 write(ROOT/'results/rational_control_20261021/admission.json',result)
 if result['status']=='PASS':write(ROOT/'private_data/closure_mechanism_20261019/P1/rational_control01/admission.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ('immutable_science_files',)}))
if __name__=='__main__':main()
