"""Loop0083 model-only local-pole audit; no development data are read."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
from train_joint_latent import BASE, NETWORK  # noqa: E402
from train_structured_rom import SOURCE, load_model  # noqa: E402

BASELINE = ROOT / "private_data/task2_structure_0078_extend_G-fixed/step3000"
CANDIDATE = ROOT / "private_data/task2_coefficient_block_0082_attempt2/candidate"
FD_STEPS = (1e-5, 1e-6)
WALL_SECONDS = 300.
BASELINE_SHA = "2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b"
CANDIDATE_SHA = "72fce608287d9e13dd23957e860ad5d20abb2e3130cd883e0f493e1021348db5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(baseline: Path, candidate: Path) -> dict[str, str]:
    paths = {"baseline_model": baseline / "model.pt", "baseline_meta": baseline / "checkpoint.json",
             "baseline_norm": baseline / "normalization.npz", "baseline_export": baseline / "structured_numeric_export.json",
             "candidate_model": candidate / "model.pt", "candidate_meta": candidate / "checkpoint.json",
             "candidate_norm": candidate / "normalization.npz", "candidate_export": candidate / "structured_numeric_export.json",
             "base": Path(BASE), "network": Path(NETWORK), "audit": Path(__file__),
             "trainer": ROOT / "experiments/train_structured_rom.py", "core": ROOT / "experiments/structured_rom.py",
             "grammar": ROOT / "experiments/structure_grammar.py", "joint_latent": ROOT / "experiments/train_joint_latent.py",
             "joint_campaign": ROOT / "experiments/train_joint_campaign.py"}
    for label, directory in (("baseline",baseline),("candidate",candidate)):
        source=Path(json.loads((directory/"checkpoint.json").read_text()).get("source") or SOURCE)
        paths[f"{label}_source_model"] = source / "model.pt"; paths[f"{label}_source_norm"] = source / "normalization.npz"
    return {key: sha(path) for key, path in paths.items()}


def require_time(deadline: float, stage: str) -> None:
    if time.perf_counter() > deadline: raise TimeoutError(f"0083 global wall cap during {stage}")


def quotient() -> np.ndarray:
    """Q removes exactly the common delta direction and retains every other coordinate."""
    _, _, vt = np.linalg.svd(np.ones((1, 10)), full_matrices=True)
    N = vt[1:].T
    Q = np.zeros((33, 32)); Q[:10, :9] = N; Q[10:, 9:] = np.eye(23)
    if np.max(np.abs(Q.T @ Q - np.eye(32))) > 1e-12: raise AssertionError("Q orthonormality")
    return Q


def central_fd(field, h: float, deadline: float) -> np.ndarray:
    zero = np.zeros(33); out = np.empty((33, 33))
    for col in range(33):
        if time.perf_counter() > deadline: raise TimeoutError("0083 FD wall cap")
        step = np.zeros(33); step[col] = h
        plus, minus = np.asarray(field(zero + step), float), np.asarray(field(zero - step), float)
        if plus.shape != (33,) or minus.shape != (33,) or not (np.isfinite(plus).all() and np.isfinite(minus).all()): raise FloatingPointError("nonfinite FD field")
        out[:, col] = (plus - minus) / (2 * h)
    return out


def eig_report(matrix: np.ndarray) -> dict[str, Any]:
    values, vectors = np.linalg.eig(matrix); vectors = vectors / np.linalg.norm(vectors,axis=0,keepdims=True); norm = np.linalg.norm(matrix, 2); residual=[]
    for j, value in enumerate(values):
        residual.append(float(np.linalg.norm(matrix @ vectors[:, j] - value * vectors[:, j]) / max(1e-300, (norm + abs(value)) * np.linalg.norm(vectors[:, j]))))
    abscissa=float(np.max(values.real))
    return {"values":values, "vectors":vectors, "spectral_abscissa":abscissa, "eigen_residual_max":max(residual,default=0.),
            "positive_count":int(np.sum(values.real > 1e-6)), "negative_count":int(np.sum(values.real < -1e-6)), "band_count":int(np.sum(np.abs(values.real) <= 1e-6)),
            "eigenvector_condition":float(np.linalg.cond(vectors))}


def analyze_matrix(A: np.ndarray, fd: dict[float, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    Q=quotient(); reduced=Q.T@A@Q; report=eig_report(reduced); fd_reduced={h:Q.T@value@Q for h,value in fd.items()}
    fd_eig={h:eig_report(value) for h,value in fd_reduced.items()}
    common=np.r_[np.ones(10),np.zeros(23)]
    fd_norms={h:float(np.linalg.norm(reduced-value,2)) for h,value in fd_reduced.items()}
    epsilon_floor=float(np.finfo(np.float64).eps*np.linalg.norm(reduced,2)); fd_max=max(fd_norms.values(),default=0.); radius=float(report["eigenvector_condition"]*max(fd_max,epsilon_floor)); sensitivity_limit=.1*abs(report["spectral_abscissa"])
    qa={"q_orthonormal_max":float(np.max(np.abs(Q.T@Q-np.eye(32))),),"eigen_residual_max":report["eigen_residual_max"],
        "fd_ad_relative":{h:float(np.linalg.norm(A-value,"fro")/max(1e-300,np.linalg.norm(A,"fro"))) for h,value in fd.items()},
        "fd_spectral_abscissa_difference":{h:abs(fd_eig[h]["spectral_abscissa"]-report["spectral_abscissa"]) for h in fd},
        "q_common_angle_max":float(np.max(np.abs(Q.T@common))),"sensitivity_radius":radius,"sensitivity_fd_2_norms":fd_norms,
        "sensitivity_fd_max_2_norm":fd_max,"sensitivity_epsilon_floor":epsilon_floor,"sensitivity_limit":sensitivity_limit,
        "sensitivity_pass":bool(radius<sensitivity_limit)}
    return {"spectral_abscissa":report["spectral_abscissa"],"positive_count":report["positive_count"],"negative_count":report["negative_count"],"band_count":report["band_count"],"eigenvector_condition":report["eigenvector_condition"],"qa":qa}, {"Q":Q,"A32":reduced,"eigenvalues":report["values"],"eigenvectors":report["vectors"],**{f"FD{h}_A33":value for h,value in fd.items()},**{f"FD{h}_A32":value for h,value in fd_reduced.items()},**{f"FD{h}_eigenvalues":fd_eig[h]["values"] for h in fd},**{f"FD{h}_eigenvectors":fd_eig[h]["vectors"] for h in fd}}


def _model_block(name: str, model, deadline: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state={key:value.detach().clone() for key,value in model.state_dict().items()}; zero=torch.zeros(33,dtype=torch.float64)
    def field_numpy(value: np.ndarray) -> np.ndarray:
        with torch.no_grad(): return model.field(torch.as_tensor(value,dtype=torch.float64)[None]).squeeze(0).cpu().numpy()
    require_time(deadline,"zero/gauge")
    f0=field_numpy(np.zeros(33)); shifted=np.zeros(33); shifted[:10]=.37; gauge=field_numpy(shifted)-f0
    with torch.enable_grad():
        ad=torch.autograd.functional.jacobian(lambda z:model.field(z[None]).squeeze(0),zero).detach().cpu().numpy()
    require_time(deadline,"AD")
    if not (np.isfinite(f0).all() and np.isfinite(gauge).all() and np.isfinite(ad).all()): raise FloatingPointError("nonfinite zero/gauge/AD matrix")
    fd={h:central_fd(field_numpy,h,deadline) for h in FD_STEPS}
    if any(not torch.equal(value,model.state_dict()[key]) for key,value in state.items()): raise AssertionError("model state changed")
    summary, arrays=analyze_matrix(ad,fd); common=np.r_[np.ones(10),np.zeros(23)]
    qa={"finite":bool(np.isfinite(ad).all() and all(np.isfinite(v).all() for v in fd.values())),"f0_max":float(np.max(np.abs(f0))),"gauge_field_max":float(np.max(np.abs(gauge))),"common_angle_max":float(np.max(np.abs(ad@common))),**summary.pop("qa")}
    qa["flags"]={"finite":qa["finite"],"origin":qa["f0_max"]<=1e-10,"gauge":qa["gauge_field_max"]<=1e-10,"common_angle":qa["common_angle_max"]<=1e-10,"Q":qa["q_orthonormal_max"]<=1e-12,"Q_common_angle":qa["q_common_angle_max"]<=1e-12,"eigen_residual":qa["eigen_residual_max"]<=1e-9,"fd_ad":all(value<=1e-7 for value in qa["fd_ad_relative"].values()),"fd_spectral":all(value<=1e-6 for value in qa["fd_spectral_abscissa_difference"].values()),"sensitivity":qa["sensitivity_pass"]}
    summary.update({"name":name,"qa":qa}); arrays.update({"A33":ad,"f0":f0,"gauge_difference":gauge,"common_angle":common})
    return summary, arrays


def run(baseline: Path, candidate: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1); started=time.perf_counter(); deadline=started+WALL_SECONDS; before=source_hashes(baseline,candidate); output.mkdir(); current={"stage":"load"}; arrays={}
    try:
        if before["baseline_model"] != BASELINE_SHA or before["candidate_model"] != CANDIDATE_SHA: raise AssertionError("preregistered model SHA mismatch")
        models={"baseline":load_model(baseline),"candidate":load_model(candidate)}
        require_time(deadline,"loading")
        for model in models.values(): model.eval()
        reports={}
        for name,model in models.items():
            current={"stage":"matrix","model":name}; report,block=_model_block(name,model,deadline); reports[name]=report; arrays.update({f"{name}_{key}":value for key,value in block.items()})
            require_time(deadline,f"{name} eigensystem")
        require_time(deadline,"matrix save")
        np.savez_compressed(output/"matrices.npz",**arrays)
        require_time(deadline,"final save")
        after=source_hashes(baseline,candidate)
        if before!=after: raise RuntimeError("source changed")
        result={"loop":"0083","test_read":False,"data_read":False,"hashes_before":before,"hashes_after":after,"wall_seconds":time.perf_counter()-started,"models":reports}
        (output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False)); return result
    except Exception as error:
        if arrays: np.savez_compressed(output/"partial_matrices.npz",**arrays)
        (output/"failure_evidence.json").write_text(json.dumps({"type":type(error).__name__,"message":str(error),"current":current,"test_read":False,"data_read":False,"hashes_before":before,"seconds":time.perf_counter()-started},indent=2)); raise


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--baseline",type=Path,default=BASELINE); parser.add_argument("--candidate",type=Path,default=CANDIDATE); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); run(args.baseline,args.candidate,args.output)


if __name__=="__main__": main()
