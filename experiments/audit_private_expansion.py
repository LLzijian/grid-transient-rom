"""Cross-shard identity/coverage QA; never a model score or test feature fit."""
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import cdist

from kiki.data.private_dataset import canonical_arrays


def main():
    roots=[Path("private_data/ieee39_postfault"),Path("private_data/ieee39_expansion")]
    batches=[]
    for root in roots:
        with (root/"dataset_index.csv").open() as stream:
            rows={r["event_id"]:r for r in csv.DictReader(stream)}
        splits=json.loads((root/"splits.json").read_text())
        states={}
        with h5py.File(root/"dataset.h5","r") as file:
            eq={k:file["equilibrium"][k][:] for k in ("delta","omega","Pm","Pe")}
            identity={"signature":str(file.attrs["postfault_signature"]),"state_mapping":str(file.attrs["state_mapping"]),
                      "case_sha256":str(file.attrs["case_sha256"])}
            full_eq=file["equilibrium/full_state"][:]
            for split in ("train","validation"):
                values=[]
                for eid in splits[split]:
                    event=file["events"][eid]
                    raw={k:event[k][:1] for k in ("t","delta","omega","Pm","Pe")}
                    value=canonical_arrays(raw,eq,file["inertia"][:],file.attrs["omega_s"])
                    values.append(np.concatenate([value[k][0] for k in ("delta","omega","Pm")]))
                states[split]=np.array(values)
        stats={}
        for split,ids in splits.items():
            stats[split]={"n":len(ids),"stable":sum(rows[e]["label"]=="stable" for e in ids),
                          "unstable":sum(rows[e]["label"]=="unstable" for e in ids),
                          "groups":sorted({rows[e]["fault_bus"] for e in ids}),
                          "eligible_after50ms":sum(float(rows[e]["trajectory_end_s"])>.05+1e-9 for e in ids)}
        batches.append({"rows":rows,"splits":splits,"states":states,"identity":identity,"equilibrium":full_eq,"stats":stats})
    old,new=batches
    assert old["identity"]==new["identity"]
    assert np.array_equal(old["equilibrium"],new["equilibrium"])
    assert not set(old["rows"])&set(new["rows"])
    for key in ("fault_bus","fault_line"):
        assert not {r[key] for r in old["rows"].values()}&{r[key] for r in new["rows"].values()}
    center=old["states"]["train"].mean(0)
    scale=np.maximum(old["states"]["train"].std(0),1e-6)
    reference=(old["states"]["train"]-center)/scale
    distances=cdist(reference,reference)
    np.fill_diagonal(distances,np.inf)
    coverage={"pilot_train_leave_one_out_nn_quantiles":np.quantile(distances.min(1),[0,.5,.9,1]).tolist()}
    for split in ("train","validation"):
        values=(new["states"][split]-center)/scale
        near=cdist(values,reference).min(1)
        coverage["expansion_"+split+"_to_pilot_train_nn_quantiles"]=np.quantile(near,[0,.5,.9,1]).tolist()
    combined=np.r_[reference,(new["states"]["train"]-center)/scale]
    singular=np.linalg.svd(combined-combined.mean(0),compute_uv=False)
    coverage["combined_train_pca_fractions"]=(singular[:5]**2/sum(singular**2)).tolist()
    count=len(new["rows"])
    passed=count>=139 and len({r["fault_bus"] for r in new["rows"].values()})==15
    passed&=len(new["stats"]["train"]["groups"])>=8 and len(new["stats"]["validation"]["groups"])>=3
    passed&=any(r["label"]=="unstable" for r in new["rows"].values()) and any(r["label"]=="stable" for r in new["rows"].values())
    result={"cross_shard_pass":bool(passed),"combined_accepted":len(old["rows"])+count,"pilot":old["stats"],"expansion":new["stats"],
            "same_equilibrium_exact":True,"same_physical_signature":new["identity"]["signature"],"bus_and_line_sets_disjoint":True,
            "coverage_non_test_only":coverage,"hashes":{str(root/name):hashlib.sha256((root/name).read_bytes()).hexdigest() for root in roots for name in ("dataset_index.csv","splits.json")}}
    out=Path("private_data/expansion_0039_cross_audit.json")
    with out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
