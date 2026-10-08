"""Loop0061 Stage-A frozen sparse long-system matrix audit."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

from audit_latent_order import (build_matrices, representative_B, fd_B_check, fd_Aq_check,
    pole_report, finite_gramians, _matrix_diagnostics, _pbh, similarity_check, _projector,
    _real_block_basis, _load_state)
from evaluate_long_latent import BASE, NETWORK, _construct
from train_joint_latent import read_development, tensor
from kiki.models.private_pm import MechanicalROM
from kiki.physics import coi_project


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def representatives(events):
    return [(event, index) for event in events for index in np.unique(np.rint(
        np.linspace(10, len(event['x'])-1, min(31, len(event['x'])-10))).astype(int))]

def physical_scales(events, mass):
    rows=[]
    for event in events:
        x=event['x'][10:]
        rows.append(np.mean(np.column_stack((coi_project(x[:,:10],mass),x[:,10:20],x[:,20:30]))**2,axis=0))
    return np.maximum(np.sqrt(np.mean(rows,axis=0)),.01)

def similarity_t5(A,B,C,Q):
    T=np.array([[2.,.4,-.2],[0.,.5,.3],[.1,0.,1.5]])
    _,_,hsv,_=finite_gramians(A,B,C,5.,Q)
    _,_,transformed,_=finite_gramians(T@A@np.linalg.inv(T),T@B,C@np.linalg.inv(T),5.,T@Q@T.T)
    error=float(np.max(np.abs(hsv-transformed)))
    assert error<1e-8, error
    return {'T5_hsv_max_abs_error':error}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1)
    base,network=MechanicalROM.load(BASE),dict(np.load(NETWORK,allow_pickle=False)); network['observed_buses']=np.unique(np.r_[network['load_buses'],network['generator_buses']]); network['load_positions']=np.searchsorted(network['observed_buses'],network['load_buses'])
    model=_construct(a.model_dir,base,network); state,norm=_load_state(a.model_dir/'model.pt'),dict(np.load(a.model_dir/'normalization.npz',allow_pickle=False))
    Aq,B,Cq,extra=build_matrices(state,norm,base,network)
    development,excluded,identities=read_development(base,network,full_train=True); events=development['train']
    lengths={e['id']:len(e['x']) for e in events}; assert len(events)==346 and min(lengths.values())>11 and max(lengths.values())>812
    reps=representatives(events); scales=physical_scales(events,base.mass); Bs=np.stack([representative_B(state,norm,base,e['x'][i])*scales[None,:] for e,i in reps])
    per_event=[]
    for event in events:
        block=Bs[[j for j,(e,_) in enumerate(reps) if e['id']==event['id']]]
        per_event.append(np.mean(block@np.swapaxes(block,1,2),axis=0))
    Q=np.mean(per_event,axis=0)
    poles,V,W,blocks,poleinfo=pole_report(Aq,B*np.asarray(scales)[None,:],Cq)
    poleinfo['damping_ratios']=[float(-pole.real/abs(pole)) if abs(pole) else None for pole in poles]
    O=np.vstack((Cq,Cq@Aq,Cq@Aq@Aq)); R=np.hstack((B,Aq@B,Aq@Aq@B)); Rs=np.hstack((B*scales[None,:],Aq@(B*scales[None,:]),Aq@Aq@(B*scales[None,:])))
    histories=tensor(np.stack([e['x'][:11] for e in events])); q0=model.encode(histories).detach().numpy()
    with torch.no_grad():
        xall=tensor(np.concatenate([e['x'][10:] for e in events])); zero=torch.zeros(1,33,dtype=torch.float64); fieldzero=model.field(zero)[0].numpy(); reference=fieldzero[30:]; actual=model.field(torch.cat((xall,torch.zeros(len(xall),3,dtype=torch.float64)),dim=1))[:,30:].numpy()-reference
    forcing=[]; cursor=0
    for event in events:
        n=len(event['x'])-10; forcing.append(actual[cursor:cursor+n]); cursor+=n
    modal=[]
    for block in blocks:
        P=_projector(block)
        forcing_rms=float(np.sqrt(np.mean([np.mean(np.abs((Cq@P@f.T).T)**2) for f in forcing])))
        initial_rms=float(np.sqrt(np.mean(np.abs((Cq@P@q0.T).T)**2)))
        modal.append({'kind':block['kind'],'poles':block['poles'],'forcing_rms':forcing_rms,'initial_output_rms':initial_rms})
    Bp=np.stack([representative_B(state,norm,base,e['x'][i]) for e,i in reps])
    rep_reports=[]
    for (event,index),physical,scaled in zip(reps,Bp,Bs,strict=True):
        residues=[{'kind':block['kind'],'poles':block['poles'],'physical_residue_norm':float(np.linalg.norm(Cq@_projector(block)@physical)),'train_scaled_residue_norm':float(np.linalg.norm(Cq@_projector(block)@scaled))} for block in blocks]
        rep_reports.append({'event_id':event['id'],'index':int(index),'R_physical':_matrix_diagnostics(np.hstack((physical,Aq@physical,Aq@Aq@physical))),'R_train_scaled':_matrix_diagnostics(np.hstack((scaled,Aq@scaled,Aq@Aq@scaled))),'PBH_reachability_physical':_pbh(Aq,physical,Cq,poles),'PBH_reachability_train_scaled':_pbh(Aq,scaled,Cq,poles),'modal_residues':residues})
    grams={}; arrays={'A':Aq,'Cq':Cq,'C_unit':extra['C_unit'],'Bphysical':B,'Bscaled':B*scales[None,:],'O':O,'Rphysical':R,'Rscaled':Rs,'input_scale':scales,'B_representative_physical':Bp,'B_representative_scaled':Bs,'representative_event_ids':np.asarray([e['id'] for e,_ in reps]),'representative_indices':np.asarray([i for _,i in reps]),'Qpooled':Q}
    for h in (1.,2.,5.,8.):
        wc,wo,hsv,balanced=finite_gramians(Aq,B*scales[None,:],Cq,h,Q); grams[str(int(h))]={'Wc':wc.tolist(),'Wo':wo.tolist(),'HSV':hsv.tolist(),'balanced':balanced}; arrays.update({f'Wc_T{h:g}':wc,f'Wo_T{h:g}':wo,f'HSV_T{h:g}':hsv})
        if h==5 and not balanced['singular']: arrays.update({'balanced_V_T5':np.asarray(balanced['V']),'balanced_W_T5':np.asarray(balanced['W'])})
    if V is not None:
        rv,rw,groups,kinds=_real_block_basis(poles,V,blocks); arrays.update({'modal_real_block_V':rv,'modal_real_block_W':rw,'modal_block_column_groups':groups,'modal_block_kinds':kinds})
    fd_points=[np.zeros(30),reps[0][0]['x'][reps[0][1]],reps[len(reps)//2][0]['x'][reps[len(reps)//2][1]],reps[-1][0]['x'][reps[-1][1]]]
    fd={'B':[fd_B_check(state,norm,base,x) for x in fd_points],'Aq':[fd_Aq_check(state,norm,base,x) for x in fd_points]}; assert max(fd['B']+fd['Aq'])<1e-7
    cqsv=np.linalg.svd(Cq,compute_uv=False); evpath=Path('private_data/task2_long_0060_5s_eval')/'evidence.json'; ev=json.loads(evpath.read_text()); assert ev['model_sha256']==digest(a.model_dir/'model.pt'); assert np.max(np.abs(cqsv-np.asarray(ev['Cq_singular_values'])))<1e-10
    out={'train_events':len(events),'train_lengths':lengths,'train_length_summary':{'min':min(lengths.values()),'max':max(lengths.values()),'max_last_index':max(lengths.values())-1},'validation_loaded_by_existing_reader_not_used':True,'excluded':excluded,'identities':identities,'A':Aq.tolist(),'Bphysical':B.tolist(),'Bscaled':(B*scales[None,:]).tolist(),'Cq':Cq.tolist(),'input_scale_event_balanced_fulltime_floor_0.01':scales.tolist(),'pooled_Q_eventweighted_representative':Q.tolist(),'poles':poleinfo,'observability':_matrix_diagnostics(O),'reachability_physical':_matrix_diagnostics(R),'reachability_train_scaled':_matrix_diagnostics(Rs),'PBH_reference_physical':_pbh(Aq,B,Cq,poles),'PBH_reference_train_scaled':_pbh(Aq,B*scales[None,:],Cq,poles),'regional_representatives':rep_reports,'modal_actual_forcing_and_initial_q':modal,'field_zero_reference_full':fieldzero.tolist(),'field_zero_reference_full_norm':float(np.linalg.norm(fieldzero)),'field_zero_reference_qdot':reference.tolist(),'analytic_fd_max_abs':fd,'Cq_singular_values':cqsv.tolist(),'gramians':grams,'similarity_check':similarity_check(Aq,B*scales[None,:],Cq,Q),'similarity_T5':similarity_t5(Aq,B*scales[None,:],Cq,Q),'source_sha256':{'model':digest(a.model_dir/'model.pt'),'normalization':digest(a.model_dir/'normalization.npz'),'base':digest(BASE),'network':digest(NETWORK),'audit_code':digest(__file__),'audit_helper_code':digest(Path(__file__).with_name('audit_latent_order.py')),'evaluator_code':digest(Path(__file__).with_name('evaluate_long_latent.py'))},'coordinate_warning':'Raw O/R spectra and conditions depend on coordinates and units; ranks, poles and transfer maps are the invariant quantities.','test_read':False}
    np.savez_compressed(a.output/'matrices.npz',**arrays); (a.output/'evidence.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__': main()
