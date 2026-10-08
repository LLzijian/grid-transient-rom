"""TRAIN-only residual decomposition and hidden-state association diagnostics."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

from kiki.models.private_pm import MechanicalROM


def spectrum(moment):
    values,vectors=np.linalg.eigh(moment)
    order=np.argsort(values)[::-1]
    values,vectors=np.maximum(values[order],0),vectors[:,order]
    for j in range(vectors.shape[1]):
        if vectors[np.argmax(abs(vectors[:,j])),j]<0:
            vectors[:,j]*=-1
    return values/max(float(sum(values)),1e-30),vectors


def correlation(stats,direction):
    count,mr,mh,rr,hh,rh=stats
    mr,mh,rr,hh,rh=mr/count,mh/count,rr/count,hh/count,rh/count
    cov=direction@(rh-mr[:,None]*mh[None,:])
    variance=float(direction@(rr-np.outer(mr,mr))@direction)
    return cov/np.maximum(np.sqrt(np.maximum(variance,0)*np.maximum(hh-mh*mh,0)),1e-15)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model=MechanicalROM.load("private_data/task2_m3_0020/model.npz")
    bg=model.incidence[:,model.coupling>0]
    assert np.linalg.matrix_rank(bg)==9
    projector=bg@np.linalg.pinv(bg)
    groups,records,arrays={},[],{}
    moment=np.zeros((10,10))
    balanced_moment=np.zeros((10,10))
    strata={}
    hashes={}
    hidden_names=None
    for root in (Path("private_data/ieee39_postfault"),Path("private_data/ieee39_expansion")):
        split=json.loads((root/"splits.json").read_text())
        assert not set(split["train"])&(set(split["validation"])|set(split["test"]))
        with (root/"dataset_index.csv").open() as stream:
            metadata={r["event_id"]:r for r in csv.DictReader(stream)}
        hashes[str(root)]={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("dataset_index.csv","splits.json")}
        with h5py.File(root/"dataset.h5","r") as file:
            mapping=json.loads(file.attrs["state_mapping"])
            columns=[r[0] for r in mapping if not (r[2]=="GENROU" and r[1] in ("delta","omega"))]
            names=[f"{r[2]}/{r[3]}/{r[1]}" for r in mapping if r[0] in columns]
            if hidden_names is None:
                hidden_names=names
            assert names==hidden_names
            eq=file["equilibrium/full_state"][:][columns]
            eq_delta=file["equilibrium/delta"][:]
            omega_s=float(file.attrs["omega_s"])
            for eid in split["train"]:
                event=file["events"][eid]
                time=event["t"][:801]
                delta=event["delta"][:801]
                speed=event["omega"][:801]
                pm,te=event["Pm"][:801],event["te"][:801]
                hidden=event["full_state"][:801,columns]-eq
                omega=(speed[1:-1]-1)*omega_s
                electrical=((np.sin(delta[1:-1]@model.incidence)-np.sin(eq_delta@model.incidence))*model.coupling)@model.incidence.T
                residual=model.pm_star-te[1:-1]+electrical+model.damping*omega
                fd=model.mass*omega_s*(speed[2:]-speed[:-2])/.02
                forcing=pm[1:-1]-te[1:-1]
                discrepancy=float(np.sqrt(np.mean((fd-forcing)**2))/max(1.,np.sqrt(np.mean(forcing**2))))
                hidden=hidden[1:-1]
                assert len(residual)>0 and np.isfinite(residual).all() and np.isfinite(hidden).all()
                balanced=residual-residual.mean(axis=1,keepdims=True)
                rr=residual.T@residual/len(residual)
                br=balanced.T@balanced/len(residual)
                moment+=rr
                balanced_moment+=br
                row=metadata[eid]
                group=row["fault_bus"]
                for stratum in (root.name,row["label"]):
                    strata[stratum]=strata.get(stratum,np.zeros((10,10)))+rr
                if group not in groups:
                    groups[group]=[0,np.zeros(10),np.zeros(len(columns)),np.zeros((10,10)),np.zeros(len(columns)),np.zeros((10,len(columns)))]
                stats=groups[group]
                stats[0]+=1
                stats[1]+=residual.mean(0)
                stats[2]+=hidden.mean(0)
                stats[3]+=rr
                stats[4]+=np.mean(hidden**2,axis=0)
                stats[5]+=residual.T@hidden/len(residual)
                common=float(np.sum((residual-balanced)**2)/len(residual))
                total=float(np.trace(rr))
                records.append({"event_id":eid,"group":group,"shard":root.name,"label":row["label"],"n":len(residual),
                    "pre50ms_LOS":float(row["trajectory_end_s"])<=.05+1e-9,"fd_discrepancy":discrepancy,
                    "force_energy":total,"balanced_energy":float(np.trace(br)),"common_energy":common,
                    "decomposition_error":abs(total-float(np.trace(br))-common)/max(1.,total)})
                arrays[eid+"_residual"]=residual
                arrays[eid+"_t"]=time[1:-1]
    fractions,modes=spectrum(balanced_moment)
    full_fractions,_=spectrum(moment)
    edge=np.linalg.pinv(bg)@modes[:,:3]
    reconstruction=float(np.max(abs(bg@edge-modes[:,:3])))
    group_records={}
    all_stats=[sum(s[k] for s in groups.values()) for k in range(6)]
    associations=[]
    for j in range(3):
        corr=correlation(all_stats,modes[:,j])
        top=np.argsort(abs(corr))[-5:][::-1]
        associations.append([{"state":hidden_names[k],"correlation":float(corr[k]),
                              "group_correlations":{g:float(correlation(s,modes[:,j])[k]) for g,s in groups.items()}} for k in top])
    for group,stats in groups.items():
        group_balanced=projector@stats[3]@projector
        frac,local=spectrum(group_balanced)
        _,leave=spectrum(balanced_moment-group_balanced)
        group_records[group]={"events":stats[0],"balanced_spectrum":frac.tolist(),
            "leading_global_alignment":float(abs(local[:,0]@modes[:,0])),
            "leave_group_out_leading_alignment":float(abs(leave[:,0]@modes[:,0]))}
    for row in records:
        trace=arrays[row["event_id"]+"_residual"]@modes[:,:3]
        temporal=[]
        for j in range(3):
            y=trace[:,j]-trace[:,j].mean()
            norm=float(y@y)
            crossing=None
            if len(y)>=10 and norm>1e-15:
                for lag in range(1,min(101,len(y)//2+1)):
                    ac=float(y[:-lag]@y[lag:]/np.sqrt((y[:-lag]@y[:-lag])*(y[lag:]@y[lag:])+1e-30))
                    if ac<=np.exp(-1):
                        crossing=lag*.01
                        break
            frequency=None
            if len(y)>=201 and norm>1e-15:
                spectrum_power=abs(np.fft.rfft(y*np.hanning(len(y))))**2
                freq=np.fft.rfftfreq(len(y),.01)
                eligible=np.flatnonzero((freq>0)&(freq<=20))
                frequency=float(freq[eligible[np.argmax(spectrum_power[eligible])]])
            temporal.append({"decorrelation_lag_s":crossing,"dominant_frequency_hz":frequency})
        row["diagnostic_mode_temporal"]=temporal
    fdq=np.quantile([r["fd_discrepancy"] for r in records],[.5,.95,1])
    decomposition=max(r["decomposition_error"] for r in records)
    passed=fdq[0]<.01 and fdq[1]<.1 and decomposition<1e-10 and reconstruction<1e-10
    result={"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "screen_pass":bool(passed),"train_events":len(records),"pre50ms_LOS_events":sum(r["pre50ms_LOS"] for r in records),
        "sign":"r_e=M omega_dot-p+E0+Domega; electrical memory BgCq fits -r_mem",
        "model_sha256":hashlib.sha256(Path("private_data/task2_m3_0020/model.npz").read_bytes()).hexdigest(),"dataset_hashes":hashes,
        "full_force_energy_spectrum":full_fractions.tolist(),"balanced_force_energy_spectrum":fractions.tolist(),
        "topology_representable_energy_fraction":float(np.trace(balanced_moment)/np.trace(moment)),
        "fd_discrepancy_median_p95_max":fdq.tolist(),"decomposition_error":decomposition,"edge_reconstruction_error":reconstruction,
        "strata_force_energy_spectra":{k:spectrum(v)[0].tolist() for k,v in strata.items()},
        "groups":group_records,"hidden_state_associations":associations,"records":records,
        "scope":"TRAIN-only diagnostic force modes; not memory dimension selection or causal/rollout evidence"}
    np.savez_compressed(args.output/"residuals.npz",**arrays)
    np.savez_compressed(args.output/"moments.npz",full_moment=moment,balanced_moment=balanced_moment,force_modes=modes,edge_modes=edge,Bg=bg)
    (args.output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records","groups","hidden_state_associations")},indent=2))


if __name__=="__main__":
    main()
