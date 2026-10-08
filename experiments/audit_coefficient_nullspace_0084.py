"""Loop0084 model-only coefficient-Jacobian/nullspace diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"experiments"),str(ROOT/"src")]
from fit_coefficient_block_0082 import coefficient_fields, deterministic_test_z  # noqa: E402
from train_joint_latent import BASE, NETWORK  # noqa: E402
from train_structured_rom import SOURCE, load_model  # noqa: E402

BASELINE=ROOT/"private_data/task2_structure_0078_extend_G-fixed/step3000"
CANDIDATE=ROOT/"private_data/task2_coefficient_block_0082_attempt2/candidate"
FIT=ROOT/"private_data/task2_coefficient_block_0082_attempt2/fit_arrays.npz"
POLES=ROOT/"private_data/task2_coefficient_poles_0083/matrices.npz"
FD_STEPS=(1e-5,1e-6); WALL_SECONDS=300.
BASELINE_SHA="2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b"
CANDIDATE_SHA="72fce608287d9e13dd23957e860ad5d20abb2e3130cd883e0f493e1021348db5"


def sha(path: Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(baseline:Path,candidate:Path,fit:Path,poles:Path)->dict[str,str]:
    paths={"baseline_model":baseline/"model.pt","baseline_meta":baseline/"checkpoint.json","baseline_norm":baseline/"normalization.npz","baseline_export":baseline/"structured_numeric_export.json","candidate_model":candidate/"model.pt","candidate_meta":candidate/"checkpoint.json","candidate_norm":candidate/"normalization.npz","candidate_export":candidate/"structured_numeric_export.json","fit_arrays":fit,"poles_matrices":poles,"base":Path(BASE),"network":Path(NETWORK),"audit":Path(__file__),"fit_audit":ROOT/"experiments/fit_coefficient_block_0082.py","poles_audit":ROOT/"experiments/audit_coefficient_poles_0083.py","trainer":ROOT/"experiments/train_structured_rom.py","core":ROOT/"experiments/structured_rom.py","grammar":ROOT/"experiments/structure_grammar.py","joint_latent":ROOT/"experiments/train_joint_latent.py","joint_campaign":ROOT/"experiments/train_joint_campaign.py","force_audit":ROOT/"experiments/audit_force_initialization_0080.py","conditional_audit":ROOT/"experiments/audit_conditional_latent_0081.py"}
    for label,directory in (("baseline",baseline),("candidate",candidate)):
        source=Path(json.loads((directory/"checkpoint.json").read_text()).get("source") or SOURCE); paths[f"{label}_source_model"]=source/"model.pt"; paths[f"{label}_source_norm"]=source/"normalization.npz"
    return {key:sha(value) for key,value in paths.items()}


def require_time(deadline:float,stage:str="FD")->None:
    if time.perf_counter()>deadline:raise TimeoutError(f"0084 wall cap during {stage}")


def column_fd(field,h:float,deadline:float)->np.ndarray:
    """Central FD of all 24 coefficient fields, output [row, coefficient, input]."""
    out=np.empty((33,24,33))
    for index in range(33):
        require_time(deadline,"FD"); direction=np.zeros(33); direction[index]=h; plus=np.asarray(field(direction),float); minus=np.asarray(field(-direction),float)
        if plus.shape!=(33,24) or minus.shape!=(33,24) or not(np.isfinite(plus).all() and np.isfinite(minus).all()):raise FloatingPointError("nonfinite coefficient FD")
        out[:,:,index]=(plus-minus)/(2*h)
    return out


def spectrum(D:np.ndarray)->dict[str,Any]:
    matrix=np.asarray(D,float).transpose(0,2,1).reshape(1089,24); scales=np.linalg.norm(matrix,axis=0); threshold=1e-14*float(scales.max(initial=0.)); zero=scales<=threshold; active=np.flatnonzero(~zero); normalized=np.zeros_like(matrix); normalized[:,active]=matrix[:,active]/scales[active]
    if len(active):
        _, singular_active, vh_active=np.linalg.svd(normalized[:,active],full_matrices=False); embedded=np.zeros((len(active),24)); embedded[:,active]=vh_active
    else: singular_active=np.empty(0); embedded=np.empty((0,24))
    zero_axes=np.eye(24)[zero]; singular=np.r_[singular_active,np.zeros(zero.sum())]; vh=np.vstack((embedded,zero_axes)); ratios=singular/max(float(singular[0]) if len(singular) else 0.,1e-300)
    resolved=ratios>=1e-8; numerical_null=ratios<=1e-12; ambiguous=(ratios>1e-12)&(ratios<1e-8)
    return {"matrix":matrix,"scales":scales,"zero_columns":zero,"normalized":normalized,"singular_values":singular,"singular_ratios":ratios,"right_vectors":vh,"numerical_rank":int(resolved.sum()),"nullity":int(numerical_null.sum()),"condition":float(np.inf if len(singular)==0 or singular[-1]==0 else singular[0]/singular[-1]),"flags":{"all_resolved":bool(resolved.all() and not zero.any()),"numerical_null":bool(numerical_null.any()),"ambiguous":bool(ambiguous.any()),"clear_gap":bool(not ambiguous.any())}}


def directions_from_null(result:dict[str,Any])->np.ndarray:
    values=[]; scales=result["scales"]
    for index,ratio in enumerate(result["singular_ratios"]):
        if ratio<=1e-12:
            direction=result["right_vectors"][index]/np.where(scales>0,scales,1.); values.append(direction/np.linalg.norm(direction))
    return np.stack(values) if values else np.empty((0,24))


def analyze(D:np.ndarray,fd:dict[float,np.ndarray],theta_delta:np.ndarray,reference:np.ndarray,fields:np.ndarray)->tuple[dict[str,Any],dict[str,np.ndarray]]:
    D=np.asarray(D,float); result=spectrum(D); matrix=result["matrix"]; common=np.r_[np.ones(10),np.zeros(23)]
    fd_relative={str(h):float(np.linalg.norm((D-value).reshape(-1))/max(1e-300,np.linalg.norm(D.reshape(-1)))) for h,value in fd.items()}
    affine=np.einsum("rcs,c->rs",D,theta_delta); affine_relative=float(np.linalg.norm(affine-reference,"fro")/max(1.,np.linalg.norm(reference,"fro")))
    kinematic=float(np.max(np.abs(D[:10]))); common_max=float(np.max(np.abs(np.einsum("rcs,s->rc",D,common))))
    directions=directions_from_null(result); residuals=np.asarray([np.linalg.norm(matrix@d)/(max(1.,np.linalg.norm(matrix))*np.linalg.norm(d)) for d in directions]) if len(directions) else np.empty(0)
    nonlinear=np.einsum("zrc,nc->znr",fields,directions) if len(directions) else np.empty((len(fields),0,33)); field_norm=np.linalg.norm(fields.reshape(len(fields),-1),axis=1); nonlinear_norm=np.linalg.norm(nonlinear,axis=2) if len(directions) else np.empty((len(fields),0))
    flags={**result["flags"],"finite":bool(np.isfinite(D).all() and all(np.isfinite(value).all() for value in fd.values())),"kinematic":kinematic<=1e-10,"common_angle":common_max<=1e-10,"affine":affine_relative<=1e-10,"fd":all(value<=1e-7 for value in fd_relative.values()),"null_direction":bool(len(directions) and np.all(residuals<=1e-10)),"nonlinear_detected":bool(any(nonlinear_norm[i].max(initial=0.)>1e-8*max(1.,field_norm[i]) for i in range(len(fields))))}
    summary={"column_scales":result["scales"].tolist(),"zero_columns":np.flatnonzero(result["zero_columns"]).tolist(),"singular_values":result["singular_values"].tolist(),"singular_ratios":result["singular_ratios"].tolist(),"numerical_rank":result["numerical_rank"],"nullity":result["nullity"],"condition":result["condition"],"classification_flags":flags,"kinematic_max":kinematic,"common_angle_max":common_max,"affine_relative":affine_relative,"fd_relative":fd_relative,"null_residuals":residuals.tolist(),"nonlinear_norms":nonlinear_norm.tolist(),"field_norms":field_norm.tolist()}
    if not np.isfinite(summary["condition"]): summary["condition"]="infinite"
    arrays={"D":D,"FD_1e-5":fd[1e-5],"FD_1e-6":fd[1e-6],"normalized_D":result["normalized"],"column_scales":result["scales"],"right_vectors":result["right_vectors"],"null_directions":directions,"nonlinear_fields":nonlinear,"affine_difference":affine,"theta_delta":theta_delta,"reference_A_difference":reference,"fixed_fields":fields}
    return summary,arrays


def run(baseline:Path,candidate:Path,fit:Path,poles:Path,output:Path)->dict[str,Any]:
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1); started=time.perf_counter(); deadline=started+WALL_SECONDS; before=hashes(baseline,candidate,fit,poles); output.mkdir(); current={"stage":"load"}; arrays={}
    try:
        if before["baseline_model"]!=BASELINE_SHA or before["candidate_model"]!=CANDIDATE_SHA:raise AssertionError("preregistered model SHA mismatch")
        with np.load(fit,allow_pickle=False) as saved: theta_delta=np.asarray(saved["theta"])-np.asarray(saved["theta0"])
        with np.load(poles,allow_pickle=False) as saved: reference=np.asarray(saved["candidate_A33"])-np.asarray(saved["baseline_A33"])
        model=load_model(baseline); model.eval(); state={key:value.detach().clone() for key,value in model.state_dict().items()}; zero=torch.zeros(33,dtype=torch.float64)
        require_time(deadline,"loading")
        with torch.no_grad(): model.begin_graph_cache(); J,R,B=model._graph_cache
        def field(value:np.ndarray)->np.ndarray:
            z=torch.as_tensor(value,dtype=torch.float64)[None]
            with torch.no_grad():return coefficient_fields(model,z,J,R,B).squeeze(0).cpu().numpy()
        with torch.enable_grad():D=torch.autograd.functional.jacobian(lambda z:coefficient_fields(model,z[None],J,R,B).squeeze(0),zero).detach().cpu().numpy()
        require_time(deadline,"AD")
        if not np.isfinite(D).all():raise FloatingPointError("nonfinite AD derivative")
        current={"stage":"FD"}; fd={h:column_fd(field,h,deadline) for h in FD_STEPS}
        z=deterministic_test_z().detach().cpu().numpy(); fields=np.stack([field(value) for value in z])
        model.end_graph_cache()
        if any(not torch.equal(value,model.state_dict()[key]) for key,value in state.items()):raise AssertionError("model state changed")
        current={"stage":"analysis"}; summary,arrays=analyze(D,fd,theta_delta,reference,fields); after=hashes(baseline,candidate,fit,poles)
        require_time(deadline,"analysis")
        if before!=after:raise RuntimeError("source changed")
        np.savez_compressed(output/"derivatives.npz",**arrays); require_time(deadline,"final save"); result={"loop":"0084","test_read":False,"data_read":False,"hashes_before":before,"hashes_after":after,"seconds":time.perf_counter()-started,"summary":summary}
        (output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False));return result
    except Exception as error:
        if arrays:np.savez_compressed(output/"partial_derivatives.npz",**arrays)
        (output/"failure_evidence.json").write_text(json.dumps({"type":type(error).__name__,"message":str(error),"current":current,"test_read":False,"data_read":False,"hashes_before":before,"seconds":time.perf_counter()-started},indent=2));raise
    finally:
        if "model" in locals():model.end_graph_cache()


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--baseline",type=Path,default=BASELINE);parser.add_argument("--candidate",type=Path,default=CANDIDATE);parser.add_argument("--fit",type=Path,default=FIT);parser.add_argument("--poles",type=Path,default=POLES);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();run(args.baseline,args.candidate,args.fit,args.poles,args.output)
if __name__=="__main__":main()
