"""Exact-rational conservative M3 barrier; finite-horizon probes are separate."""
import argparse
import json
import subprocess
from fractions import Fraction as F
from pathlib import Path

import h5py
import numpy as np
from run_private_m2 import development_data
from scipy.integrate import solve_ivp
from scipy.stats import rankdata, spearmanr

from kiki.data.private_dataset import los_index
from kiki.models.private_pm import MechanicalROM


def exact_bound(model):
    active = np.flatnonzero(model.coupling>0)
    edges = model.incidence[:,active].astype(int)
    star = [F(float(x)) for x in model.delta_star]
    stiffness = [[sum(F(float(model.coupling[e]))*int(model.incidence[i,e])*int(model.incidence[j,e]) for e in active) for j in range(9)] for i in range(9)]
    aug = [row.copy()+[F(int(i==j)) for j in range(9)] for i,row in enumerate(stiffness)]
    for i in range(9):
        pivot = aug[i][i]
        assert pivot>0
        aug[i] = [x/pivot for x in aug[i]]
        for j in range(9):
            if j!=i:
                value = aug[j][i]
                aug[j] = [x-value*y for x,y in zip(aug[j],aug[i],strict=True)]
    inverse = [row[9:] for row in aug]
    residual = max(abs(sum(stiffness[i][k]*inverse[k][j] for k in range(9))-int(i==j)) for i in range(9) for j in range(9))
    assert residual==0
    gamma = F(7,5)
    cosine_lower = 1-gamma**2/2+gamma**4/24-gamma**6/720
    values,records = [],[]
    for index,edge in zip(active,edges.T,strict=True):
        eta = sum(s*int(b) for s,b in zip(star,edge,strict=True))
        assert abs(eta)<gamma
        resistance = sum(int(edge[i])*inverse[i][j]*int(edge[j]) for i in range(9) for j in range(9))
        assert resistance>0
        value = cosine_lower*(gamma-abs(eta))**2/(2*resistance)
        values.append(value)
        records.append({"edge":int(index),"eta_star":float(eta),"resistance":float(resistance),"boundary_lower_bound":float(value)})
    bound = min(values)
    rounded = float(bound)
    if F(rounded)>bound:
        rounded = float(np.nextafter(rounded,-np.inf))
    assert F(rounded)<=bound and rounded>0 and cosine_lower>0
    float_inverse = np.array([[float(x) for x in row] for row in inverse])
    direct = np.linalg.inv(np.array([[float(x) for x in row] for row in stiffness]))
    error = float(np.max(abs(float_inverse-direct)))
    return rounded,{"gamma":"7/5","cosine_lower_exact":str(cosine_lower),
        "bound_exact_numerator":str(bound.numerator),"bound_exact_denominator":str(bound.denominator),
        "Hsafe_downward_float":rounded,"exact_inverse_residual":"0","float_inverse_error":error,
        "downward_rounding_verified":True,"facets":records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model = MechanicalROM.load("private_data/task2_m3_0020/model.npz")
    bound,certificate = exact_bound(model)
    active = model.coupling>0
    sets,metadata,excluded = development_data(Path("private_data/ieee39_postfault"))
    eq = np.r_[np.zeros(20),model.pm_star]
    records,initials = [],{}
    for split,events in sets.items():
        for event in events:
            x = np.r_[event["delta_sync"][0],event["omega_sync"][0],event["Pm_raw"][0]]
            initials[event["event_id"]] = x
            energy = float(model.energy(x))
            phase = float(np.max(abs((x[:10]+model.delta_star)@model.incidence[:,active])))
            row = metadata[event["event_id"]]
            records.append({"event_id":event["event_id"],"split":split,"group":row["fault_bus"],
                "label":row["label"],"duration":float(row["fault_duration_s"]),"H":energy,
                "margin":bound-energy,"normalized_margin":1-energy/bound,"max_active_phase":phase,
                "inside_cohesive":phase<1.4,"numerically_sufficient":bool(phase<1.4-1e-8 and energy<.99*bound)})
    stiffness = (model.incidence*(model.coupling*np.cos(model.delta_star@model.incidence)))@model.incidence.T
    qmetric = np.zeros((29,29))
    qmetric[:9,:9] = stiffness[:9,:9]
    qmetric[9:19,9:19] = np.diag(model.mass)
    qmetric[19:,19:] = np.diag(1/model.gain)
    directions,names = [],[]
    train = [r for r in records if r["split"]=="train"]
    for group in sorted({r["group"] for r in train}):
        chosen = max((r for r in train if r["group"]==group),key=lambda r:r["H"])
        x = initials[chosen["event_id"]]-eq
        directions.append(np.r_[x[:9]-x[9],x[10:]])
        names.append(chosen["event_id"])
    rng = np.random.default_rng(36)
    inverse_cholesky = np.linalg.inv(np.linalg.cholesky(qmetric).T)
    for i in range(17):
        directions.append(inverse_cholesky@rng.normal(size=29))
        names.append(f"random_{i}")
    grid,grid_labels = [],[]
    for name,direction in zip(names,directions,strict=True):
        direction = direction*np.sqrt(bound/(.5*direction@qmetric@direction))
        for amplitude in (.25,.5,1.,2.,4.,8.):
            state = eq+amplitude*np.r_[direction[:9],0.,direction[9:]]
            grid.append(state)
            grid_labels.append((name,amplitude))
    grid = np.array(grid)
    time = np.arange(6001)*.01
    print(json.dumps({"Hsafe":bound,"probe_n":len(grid),"development_sufficient":sum(r["numerically_sufficient"] for r in records)}),flush=True)
    predictions = model.rollout(grid,None,time,max_step=.005)
    with h5py.File("private_data/ieee39_postfault/dataset.h5","r") as file:
        config,omega_s = json.loads(file.attrs["config"]),float(file.attrs["omega_s"])
    probes = []
    for index,(pred,(name,amplitude)) in enumerate(zip(predictions,grid_labels,strict=True)):
        phase = np.max(abs((pred[:,:10]+model.delta_star)@model.incidence[:,active]),axis=1)
        energy = model.energy(pred)
        los = los_index(pred[:,:10]+model.delta_star,pred[:,10:20]/omega_s+1,model.mass,config)
        safe = bool(phase[0]<1.4-1e-8 and energy[0]<.99*bound)
        probes.append({"index":index,"direction":name,"amplitude":amplitude,"initial_H":float(energy[0]),
            "normalized_margin":float(1-energy[0]/bound),"initial_phase":float(phase[0]),
            "sufficient":safe,"cohesive_exit":bool(np.any(phase>1.4+1e-8)),
            "max_phase":float(max(phase)),"los_time":None if los is None else float(time[los]),
            "final_H":float(energy[-1]),"max_H_increase":float(max(np.diff(energy),default=0.))})
    selected = sorted(probes,key=lambda r:abs(r["initial_H"]/bound-1))[:3]
    numerical = []
    for row in selected:
        i = row["index"]
        solution = solve_ivp(lambda t,x:model.field(x),(0,60),grid[i],t_eval=time,method="DOP853",rtol=1e-10,atol=1e-11)
        assert solution.success
        error = float(np.max(abs(solution.y.T-predictions[i])/np.maximum(1.,abs(solution.y.T))))
        numerical.append({"index":i,"max_relative_state_error":error})
    strata = {}
    for split in sets:
        subset = [r for r in records if r["split"]==split]
        label = np.array([r["label"]=="unstable" for r in subset])
        score = np.array([r["H"] for r in subset])
        n1,n0 = sum(label),sum(~label)
        strata[split] = {"events":len(subset),"sufficient":sum(r["numerically_sufficient"] for r in subset),
            "sufficient_unstable":sum(r["numerically_sufficient"] and r["label"]=="unstable" for r in subset),
            "interior":sum(r["inside_cohesive"] for r in subset),
            "H_unstable_AUC":float((rankdata(score)[label].sum()-n1*(n1+1)/2)/(n1*n0)),
            "margin_duration_spearman":float(spearmanr(-score,[r["duration"] for r in subset]).statistic),
            "group_margin_duration_spearman":{g:float(spearmanr([r["margin"] for r in subset if r["group"]==g],[r["duration"] for r in subset if r["group"]==g]).statistic) for g in sorted({r["group"] for r in subset})}}
    violations = [r["index"] for r in probes if r["sufficient"] and r["cohesive_exit"]]
    passed = not violations and certificate["float_inverse_error"]<1e-10 and max(r["max_relative_state_error"] for r in numerical)<1e-4
    result = {"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "screen_pass":bool(passed),"certificate":certificate,"strata":strata,"records":records,
        "probe_records":probes,"probe_violations":violations,"integration_checks":numerical,
        "excluded_predecision":excluded,"scope":"analytic conservative exact-reduced-ODE bound; event energies and60s basin probes numerical only; no FOM certificate"}
    np.savez_compressed(args.output / "probe_initials.npz",states=grid)
    # Full output retained privately for independent quadrature / trajectory checks.
    np.savez_compressed(args.output / "probe_rollouts.npz",predictions=predictions,time=time)
    (args.output / "evidence.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records","probe_records","certificate")},indent=2))


if __name__ == "__main__":
    main()
