"""Fixed physical-amplitude ray probes; numerical LOS boundary, not full ROA."""
import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from run_private_m2 import development_data
from scipy.integrate import solve_ivp

from kiki.data.private_dataset import los_index
from kiki.models.private_pm import MechanicalROM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model = MechanicalROM.load("private_data/task2_m3_0020/model.npz")
    bound = json.loads(Path("private_data/task2_energy_boundary_0036/evidence.json").read_text())["certificate"]["Hsafe_downward_float"]
    sets,metadata,_ = development_data(Path("private_data/ieee39_postfault"))
    eq = np.r_[np.zeros(20),model.pm_star]
    states = {e["event_id"]:np.r_[e["delta_sync"][0],e["omega_sync"][0],e["Pm_raw"][0]] for e in sets["train"]}
    names,directions = [],[]
    for group in sorted({metadata[e]["fault_bus"] for e in states}):
        name = max((e for e in states if metadata[e]["fault_bus"]==group),key=lambda e:float(model.energy(states[e])))
        direction = states[name]-eq
        direction[:10] -= direction[9]
        names.append(name)
        directions.append(direction)
    directions = np.array(directions)
    with h5py.File("private_data/ieee39_postfault/dataset.h5","r") as file:
        config,omega_s = json.loads(file.attrs["config"]),float(file.attrs["omega_s"])
    time = np.arange(6001)*.01
    records,initials = [],[]
    def assess(pred,initial,direction_index,rho,stage):
        hit = los_index(pred[:,:10]+model.delta_star,pred[:,10:20]/omega_s+1,model.mass,config)
        h = model.energy(pred)
        phase = np.max(abs((pred[:,:10]+model.delta_star)@model.incidence[:,model.coupling>0]),axis=1)
        sufficient = bool(phase[0]<1.4-1e-8 and h[0]<.99*bound)
        row = {"direction":names[direction_index],"rho":float(rho),"stage":stage,
            "initial_H":float(h[0]),"final_H":float(h[-1]),"LOS":hit is not None,
            "los_time":None if hit is None else float(time[hit]),"initial_phase":float(phase[0]),
            "cohesive_exit":bool(max(phase)>1.4+1e-8),"sufficient":sufficient,
            "max_positive_H_step_relative":float(max(0.,max(np.diff(h)))/max(1.,abs(h[0])))}
        records.append(row)
        initials.append(initial.copy())
        return row
    grid = np.arange(17)*.125
    initial = np.array([eq+rho*d for d in directions for rho in grid])
    predictions = model.rollout(initial,None,time,max_step=.005)
    grid_rows = []
    for i in range(len(directions)):
        rows = []
        for j,rho in enumerate(grid):
            k=i*len(grid)+j
            rows.append(assess(predictions[k],initial[k],i,rho,"grid"))
        grid_rows.append(rows)
    del predictions
    brackets,nonmonotone = [],[]
    for i,rows in enumerate(grid_rows):
        flags = np.array([r["LOS"] for r in rows])
        if np.any(flags[:-1] & ~flags[1:]):
            nonmonotone.append(names[i])
        starts = np.flatnonzero(~flags[:-1] & flags[1:])
        if len(starts):
            j=starts[0]
            brackets.append({"direction_index":i,"lower":float(grid[j]),"upper":float(grid[j+1])})
    print(json.dumps({"grid_LOS":sum(r["LOS"] for r in records),"brackets":len(brackets),"nonmonotone":nonmonotone}),flush=True)
    for iteration in range(12):
        if not brackets:
            break
        middle = [(r["lower"]+r["upper"])/2 for r in brackets]
        initial = np.array([eq+rho*directions[r["direction_index"]] for r,rho in zip(brackets,middle,strict=True)])
        prediction = model.rollout(initial,None,time,max_step=.005)
        for k,(bracket,rho) in enumerate(zip(brackets,middle,strict=True)):
            row = assess(prediction[k],initial[k],bracket["direction_index"],rho,f"bisection_{iteration+1}")
            bracket["upper" if row["LOS"] else "lower"]=rho
        print(json.dumps({"bisection":iteration+1}),flush=True)
    checks = []
    for bracket in brackets[:3]:
        for side in ("lower","upper"):
            i = bracket["direction_index"]
            initial = eq+bracket[side]*directions[i]
            solution = solve_ivp(lambda t,x:model.field(x),(0,60),initial,t_eval=time,method="DOP853",rtol=1e-10,atol=1e-11)
            assert solution.success
            hit = los_index(solution.y.T[:,:10]+model.delta_star,solution.y.T[:,10:20]/omega_s+1,model.mass,config)
            checks.append({"direction":names[i],"side":side,"rho":bracket[side],"DOP853_LOS":hit is not None,"agrees":bool((hit is not None)==(side=="upper"))})
    for bracket in brackets:
        d = directions[bracket["direction_index"]]
        bracket.update({"direction":names[bracket["direction_index"]],
            "H_lower":float(model.energy(eq+bracket["lower"]*d)),
            "H_upper":float(model.energy(eq+bracket["upper"]*d)),
            "rho_width":bracket["upper"]-bracket["lower"]})
    violations = [i for i,r in enumerate(records) if r["sufficient"] and r["cohesive_exit"]]
    passed = bool(brackets) and not violations and all(r["agrees"] for r in checks) and max(r["max_positive_H_step_relative"] for r in records)<=1e-6
    result = {"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "screen_pass":passed,"Hsafe":bound,"brackets":brackets,"nonmonotone_rays":nonmonotone,
        "independent_checks":checks,"sufficient_region_violations":violations,"records":records,
        "scope":"60s numerical LOS first-transition intervals on fixed TRAIN-derived rays; not global/asymptotic ROA or FOM reachable synthetic states"}
    np.savez_compressed(args.output / "initials.npz",states=np.array(initials),directions=directions,event_ids=np.array(names))
    (args.output / "evidence.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))


if __name__ == "__main__":
    main()
