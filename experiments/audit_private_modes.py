"""Analytic frozen-M3 modes and unforced, descriptive full-order comparison."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

from kiki.models.private_pm import MechanicalROM


def full_jacobian(model):
    stiffness = (model.incidence*(model.coupling*np.cos(model.delta_star@model.incidence)))@model.incidence.T
    result = np.zeros((30,30))
    result[:10,10:20] = np.eye(10)
    result[10:20,:10] = -stiffness/model.mass[:,None]
    result[10:20,10:20] = -np.diag(model.damping/model.mass)
    result[10:20,20:] = np.diag(1/model.mass)
    result[20:,10:20] = -np.diag(model.gain)
    result[20:,20:] = -np.diag(model.rate)
    return result


def gauge_maps():
    project, embed = np.zeros((29,30)), np.zeros((30,29))
    project[:9,:9] = np.eye(9)
    project[:9,9] = -1
    project[9:,10:] = np.eye(20)
    embed[:9,:9] = np.eye(9)
    embed[10:,9:] = np.eye(20)
    return project, embed


def modal_records(matrix,names):
    values, right = np.linalg.eig(matrix)
    left = np.linalg.inv(right)
    participation = abs(right*left.T)
    participation /= participation.sum(axis=0)
    residual = float(np.linalg.norm(matrix@right-right*values)/max(1.,np.linalg.norm(matrix)*np.linalg.norm(right)))
    rows = []
    for j,value in enumerate(values):
        top = np.argsort(participation[:,j])[-5:][::-1]
        electromechanical = [i for i,n in enumerate(names) if "GENROU" in n and ("delta" in n or "omega" in n)]
        rows.append({"index":j,"real":float(value.real),"imag":float(value.imag),
                     "frequency_hz":float(abs(value.imag)/(2*np.pi)),
                     "damping_ratio":None if abs(value)<1e-10 else float(-value.real/abs(value)),
                     "electromechanical_participation":float(participation[electromechanical,j].sum()),
                     "top_states":{names[i]:float(participation[i,j]) for i in top}})
    return values,right,left,rows,residual


def descriptor_records(path):
    """Finite modes of the original DAE, avoiding index-two filter elimination."""
    data = np.load(path / "descriptor.npz")
    a = np.block([[data["fx"],data["fy"]],[data["gx"],data["gy"]]])
    time_constants = np.r_[data["time_constants"],np.zeros(len(data["gy"]))]
    b = np.diag(time_constants)
    homogeneous,left,right = eig(a,b,left=True,homogeneous_eigvals=True)
    finite = abs(homogeneous[1])>1e-10*np.sum(abs(homogeneous),axis=0)
    values = homogeneous[0,finite]/homogeneous[1,finite]
    right,left = right[:,finite],left[:,finite]
    residuals = np.linalg.norm(a@right-(b@right)*values,axis=0)/(
        (np.linalg.norm(a)+abs(values)*np.linalg.norm(b))*np.linalg.norm(right,axis=0))
    weights = left.conj()*time_constants[:,None]
    overlaps = np.sum(weights*right,axis=0)
    reliable = abs(overlaps)>=1e-12
    participation = abs(weights*right/np.where(reliable,overlaps,1.))
    participation /= participation.sum(axis=0)
    names = np.r_[data["state_names"],data["algebraic_names"]].tolist()
    em = [i for i,n in enumerate(names) if "GENROU" in n and ("delta" in n or "omega" in n)]
    rows = []
    for j,value in enumerate(values):
        top = np.argsort(participation[:,j])[-5:][::-1]
        rows.append({"index":j,"real":float(value.real),"imag":float(value.imag),
            "frequency_hz":float(abs(value.imag)/(2*np.pi)),
            "damping_ratio":None if abs(value)<1e-10 else float(-value.real/abs(value)),
            "participation_identifiable":bool(reliable[j]),
            "electromechanical_participation":float(participation[em,j].sum()) if reliable[j] else None,
            "top_states":{names[i]:float(participation[i,j]) for i in top} if reliable[j] else {}})
    row_scale = np.maximum(np.max(abs(a)+abs(b),axis=1),1e-10)
    scaled = eig(a/row_scale[:,None],b/row_scale[:,None],right=False,homogeneous_eigvals=True)
    other_finite = abs(scaled[1])>1e-10*np.sum(abs(scaled),axis=0)
    other = scaled[0,other_finite]/scaled[1,other_finite]
    assert len(other)==len(values)
    distance = abs(values[:,None]-other[None,:])/np.maximum(1.,abs(values[:,None]))
    ri,ci = linear_sum_assignment(distance)
    cross_error = float(np.max(distance[ri,ci]))
    checks = {"method":"original699-state descriptor QZ, no substituted algebraic constraints",
              "finite_modes":len(values),"infinite_modes":int(sum(~finite)),
              "indeterminate_pairs":int(sum(np.sum(abs(homogeneous),axis=0)<1e-10)),
              "max_eigen_residual":float(max(residuals)),"row_equilibration_eigen_error":cross_error,
              "min_left_E_right_overlap":float(min(abs(overlaps))),
              "nonidentifiable_participation_indices":np.flatnonzero(~reliable).tolist(),
              "positive_modes":int(sum(values.real>1e-6)),"zero_modes":int(sum(abs(values)<=1e-6))}
    assert checks["indeterminate_pairs"]==0 and cross_error<1e-6
    return rows,float(max(residuals)),checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullorder",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model = MechanicalROM.load("private_data/task2_m3_0020/model.npz")
    eq = np.r_[np.zeros(20),model.pm_star]
    jac = full_jacobian(model)
    errors = []
    for step in (1e-6,5e-7):
        fd = np.column_stack([(model.field(eq+step*d)-model.field(eq-step*d))/(2*step) for d in np.eye(30)])
        errors.append(float(np.max(abs(fd-jac)/np.maximum(1.,abs(jac)))))
    project,embed = gauge_maps()
    matrix = project@jac@embed
    names = [f"delta_{i+1}-delta_10" for i in range(9)]+[f"omega_{i+1}" for i in range(10)]+[f"p_{i+1}" for i in range(10)]
    values,right,left,rows,residual = modal_records(matrix,names)
    dominant = int(np.argmax(values.real))
    sensitivities = []
    for name in ("coupling","rate","gain","damping"):
        for i,parameter in enumerate(getattr(model,name)):
            if parameter <= 0:
                continue
            perturbed = MechanicalROM(**{k:v.copy() for k,v in vars(model).items()})
            getattr(perturbed,name)[i] += parameter
            derivative = project@(full_jacobian(perturbed)-jac)@embed/parameter
            predicted = left[dominant]@derivative@right[:,dominant]
            step = 1e-4*parameter
            pair = []
            for sign in (-1,1):
                getattr(perturbed,name)[i] = parameter+sign*step
                candidate = np.linalg.eigvals(project@full_jacobian(perturbed)@embed)
                pair.append(candidate[np.argmin(abs(candidate-values[dominant]))])
            observed = (pair[1]-pair[0])/(2*step)
            sensitivities.append({"parameter":name,"index":i,"value":float(parameter),
                "derivative_real":float(predicted.real),"derivative_imag":float(predicted.imag),
                "log_parameter_derivative_real":float(parameter*predicted.real),
                "finite_difference_relative_error":float(abs(observed-predicted)/max(1.,abs(predicted)))})
    frows,fr,descriptor_checks = descriptor_records(args.fullorder)
    full_evidence = json.loads((args.fullorder / "evidence.json").read_text())
    nonzero = [r for r in frows if abs(complex(r["real"],r["imag"]))>1e-6]
    fmodes = [r for r in frows if r["imag"]>1e-6 and r["electromechanical_participation"] is not None and r["electromechanical_participation"]>=.1]
    comparisons = []
    for row in rows:
        if row["imag"]<=1e-6 or not fmodes:
            continue
        nearest = min(fmodes,key=lambda r:abs(r["frequency_hz"]-row["frequency_hz"]))
        comparisons.append({"M3_mode":row["index"],"fullorder_mode":nearest["index"],
                            "frequency_ratio":row["frequency_hz"]/nearest["frequency_hz"],
                            "M3_damping":row["damping_ratio"],"fullorder_damping":nearest["damping_ratio"]})
    sensitivity_error = max(r["finite_difference_relative_error"] for r in sensitivities)
    passed = max(errors)<1e-6 and max(values.real)<-1e-6 and residual<1e-8 and fr<1e-8 and sensitivity_error<1e-4
    result = {"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "screen_pass":bool(passed),"equilibrium_residual":float(np.max(abs(model.field(eq)))),
              "jacobian_fd_errors":errors,"eigen_residual":residual,"fullorder_eigen_residual":fr,
              "local_spectral_margin":float(-max(values.real)),"dominant_mode":rows[dominant],
              "M3_modes":rows,"parameter_sensitivities":sensitivities,
              "fullorder_identity":full_evidence,"fullorder_dominant_nonzero":max(nonzero,key=lambda r:r["real"]),
              "descriptor_checks":descriptor_checks,
              "fullorder_electromechanical_modes":fmodes,"fullorder_all_modes":frows,
              "nearest_frequency_comparisons_descriptive":comparisons,
              "scope":"local autonomous reduced-ODE spectrum, not a FOM margin or nonlinear certificate"}
    assert result["equilibrium_residual"]<1e-10
    np.savez_compressed(args.output / "M3_linearization.npz",matrix=matrix,eigenvalues=values,state_names=np.array(names))
    (args.output / "evidence.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("M3_modes","parameter_sensitivities","fullorder_all_modes")},indent=2))


if __name__ == "__main__":
    main()
