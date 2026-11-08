"""Root's fixed arithmetic after the one complete measurement; no numerical imports."""
import argparse, hashlib, json, math
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation-seconds',required=True,type=float)
    parser.add_argument('--generation-tool-chunk',required=True)
    args=parser.parse_args()
    qp=BASE/'native_qa01/execution.json';cp=BASE/'native_cost01/execution.json'
    qa=read(qp);cost=read(cp);wall=read(BASE/'cost01_outer_wall.json')
    for record in (qa,cost):
        require(record['status']=='COMPLETED' and record['phase_gate_pass'] is True,'complete passing execution proof required')
        require(record['inner']['exit_code']==record['outer']['exit_code']==0,'both native exits required')
        require(record['inner']['cleanup_verified'] and record['inner']['fixture_identity_after']==record['inner']['supervised_identity_after']=='GONE','native cleanup required')
        require(record['inner']['native_receipt_complete'] and record['outer']['native_receipt_complete'],'complete native receipts required')
        require(record['source_before']==record['source_after'],'source changed during execution')
    require(qa['source_commit']==cost['source_commit'] and qa['immutable_science_files']==cost['immutable_science_files'],'same actually qualified source required')
    for name,digest in cost['source_after'].items():require(sha(ROOT/name)==digest,'bound source changed after cost: '+name)
    result=cost['complete']['result'];panels=result['panels']
    expected={'update':96,'g_train':3,'g_val':3,'v_complete':3,'save':12,'restore':3,'construct':6,'load':3,'raw_write':9,'json_write':64}
    require(result.get('cost_panels_complete') is True and result['counts']==expected,'complete fixed actual counts required')
    require(set(panels)==set(expected),'timing panel classes differ')
    for key,count in expected.items():
        require(len(panels[key])==count and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in panels[key]),'positive complete timing samples: '+key)
    measured=sum(sum(v) for v in panels.values());body=result['body_wall_seconds'];residual=result['body_unassigned_seconds']
    require(math.isclose(measured,result['body_measured_seconds'],rel_tol=1e-12,abs_tol=1e-8),'measured total differs')
    require(math.isfinite(body) and body>=measured-1e-8 and math.isclose(residual,max(0.,body-measured),rel_tol=1e-12,abs_tol=1e-8),'unmapped body residue differs')
    elapsed=(int(wall['monotonic_end_ticks'])-int(wall['monotonic_start_ticks']))/int(wall['monotonic_frequency'])
    require(wall['source_commit']==cost['source_commit'] and wall['phase']=='cost' and wall['launcher_exit_code']==0 and wall['error'] is None,'actual matching outside launch proof required')
    require(math.isclose(elapsed,wall['elapsed_seconds'],rel_tol=1e-12,abs_tol=1e-8) and elapsed>=cost['charged_seconds'] and elapsed>=body,'outside interval must cover complete actual native/body')
    require(math.isfinite(args.generation_seconds) and args.generation_seconds>0 and args.generation_tool_chunk,'actual generation tool observation required')
    startup=args.generation_seconds+elapsed-body+residual
    maxima={k:max(v) for k,v in panels.items()}
    run=1536*maxima['update']+3*maxima['g_train']+12*maxima['save']+3*maxima['restore']+6*maxima['construct']+3*maxima['raw_write']+32*maxima['json_write']+startup
    evaluate=3*maxima['g_val']+3*maxima['v_complete']+3*maxima['load']+6*maxima['raw_write']+32*maxima['json_write']+startup
    total=run+evaluate
    gates={'complete_with_margin':1.25*total+180<=900,'run_with_margin':1.25*run<=600,'evaluate_with_margin':1.25*evaluate<=100}
    out={'status':'PASS' if all(gates.values()) else 'REJECT','source_commit':cost['source_commit'],'P1_complete_cap_seconds':900,'support_seconds':180,'T_cap_seconds':576,'T_seconds':total,'run_T_seconds':run,'evaluate_T_seconds':evaluate,'with_margin_and_support_seconds':1.25*total+180,'run_with_margin_seconds':1.25*run,'evaluate_with_margin_seconds':1.25*evaluate,'gates':gates,'maxima_seconds':maxima,'all_samples_retained':True,'startup_per_process_seconds':startup,'body_unassigned_per_process_seconds':residual,'observed_outer_monotonic_seconds':elapsed,'generation_tool_wall_seconds':args.generation_seconds,'generation_tool_chunk':args.generation_tool_chunk,'actual_counts':expected,'native_qa_seconds':qa['charged_seconds'],'native_cost_seconds':cost['charged_seconds'],'qa_receipt_sha256':sha(qp),'cost_receipt_sha256':sha(cp),'outer_wall_sha256':sha(BASE/'cost01_outer_wall.json'),'assessment_code_sha256':sha(__file__),'scope':'Empirical complete estimate with fixed margin; not a guarantee of future runtime or scientific capability'}
    target=BASE/'admission.json'
    with target.open('x',encoding='utf-8') as stream:json.dump(out,stream,indent=2);stream.write('\n')
    print(json.dumps(out))

if __name__=='__main__':main()
