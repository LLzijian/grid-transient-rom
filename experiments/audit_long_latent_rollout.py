"""Loop0061 Stage-B fixed sparse5 rollout interventions; development cache only."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"experiments"))
import audit_latent_order_rollout as core
from kiki.models.private_pm import MechanicalROM

DT=.01; SCALES=np.array([.47675314588683515,1.0033667249602924])
BASE=ROOT/"private_data/task2_m3_0020/model.npz"; MODEL=ROOT/"private_data/task2_long_0060_5s"; CACHE=ROOT/"private_data/task2_long_0060_5s_eval"; STEP0025=ROOT/"private_data/task2_long_0060_5s_step0025"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def coi(x,m): return x-(x*m).sum(axis=-1,keepdims=True)/m.sum()
def state(path):
 import torch
 return {k:np.asarray(v.detach().cpu().numpy(),dtype=float) for k,v in torch.load(path,map_location="cpu",weights_only=True).items()}
def params(model, matrices):
 s=state(model/"model.pt"); n=np.load(model/"normalization.npz",allow_pickle=False); b=MechanicalROM.load(BASE); active=b.coupling>0
 for k in ("mass","incidence","delta_star","coupling","damping","rate","gain"):
  if not np.array_equal(s[k],getattr(b,k)): raise AssertionError(k)
 p={k:np.asarray(getattr(b,k),float) for k in ("mass","incidence","delta_star","coupling","damping","rate","gain")}; p.update({k:s[k] for k in ("xi","support","alpha","alpha_support","c","amplitude","angle_incidence","ports")}); p["feature_scale"]=n["feature_scale"]
 if not np.array_equal(s["feature_scale"],n["feature_scale"]): raise AssertionError("feature_scale")
 if not np.array_equal(p["angle_incidence"],b.incidence[:,active]): raise AssertionError("angle_incidence")
 c=p["c"]/np.linalg.norm(p["c"],axis=0,keepdims=True); expected=p["ports"]@c*p["amplitude"][None,:]
 if not np.allclose(expected,matrices["Cq"],atol=1e-10,rtol=1e-10): raise AssertionError("Cq")
 p["Cq"]=matrices["Cq"]
 if p["Cq"].shape!=(10,3): raise ValueError("Cq")
 return p
def metric(pred,truth,m,h):
 k=len(truth)-1 if h=="prefix" else round(float(h)/DT)
 if k<=0 or (h!="prefix" and len(truth)<=k): return None
 e=pred[1:k+1,:20]-truth[1:k+1,:20]; d,w=coi(e[:,:10],m),coi(e[:,10:20],m); s=.5*((d*d).sum()/SCALES[0]**2+(w*w).sum()/SCALES[1]**2)
 return {"horizon":str(h),"n":k*10,"delta_sse":float((d*d).sum()),"omega_sse":float((w*w).sum()),"normalized_sse":float(s),"rmse":float(np.sqrt(s/(k*10)))}
def aggregate(rows):
 v=np.array([r["rmse"] for r in rows]); q=max(1,int(np.ceil(.1*len(v)))); return {"n_events":len(rows),"mean":float(v.mean()),"pooled":float(np.sqrt(sum(r["normalized_sse"] for r in rows)/sum(r["n"] for r in rows))),"cvar10":float(np.sort(v)[-q:].mean())}
def numerical(a,b,m,h):
 k=round(h/DT); e=a[:,:k+1,:20]-b[:,:k+1,:20]; d,w=coi(e[...,:10],m),coi(e[...,10:20],m); return np.sqrt(.5*np.mean((d/SCALES[0])**2+(w/SCALES[1])**2,axis=(1,2)) )
def finite_status(name,stage,array,ids):
 bad=np.flatnonzero(~np.isfinite(array).all(axis=(1,2)))
 return None if not len(bad) else {"arm":name,"stage":stage,"reason":"nonfinite","event_ids":[ids[i] for i in bad]}
def persist(output,ids,arrays): np.savez_compressed(output/"predictions_partial.npz",event_ids=np.asarray(ids),**arrays)
def scope_values(primary,companion,truth,ids,meta,m,h):
 fixed=numerical(primary,companion,m,h); complete=np.array([len(t)>round(h/DT) for t in truth]); unstable=np.array([meta[i]["label"]=="unstable" for i in ids])
 prefix=np.array([numerical(primary[i:i+1],companion[i:i+1],m,min(round(h/DT),len(t)-1)*DT)[0] for i,t in enumerate(truth)])
 def row(values,mask):
  if not mask.any(): return {"n":0,"max":None,"worst_event":None}
  values=values[mask]; pos=np.flatnonzero(mask); j=int(values.argmax())
  return {"n":int(mask.sum()),"max":float(values[j]),"worst_event":ids[pos[j]]}
 return {"all_fixed":row(fixed,np.ones(len(ids),bool)),"complete":row(fixed,complete),"available_prefix":row(prefix,np.ones(len(ids),bool)),"unstable_prefix":row(prefix,unstable),"per_event_fixed":[{"event_id":i,"value":float(v)} for i,v in zip(ids,fixed)],"per_event_available_prefix":[{"event_id":i,"value":float(v)} for i,v in zip(ids,prefix)],"claimed_valid_all":bool(fixed.max()<.001)}
def source_contract(audit,ids):
 from evaluate_long_latent import NETWORK
 actual={"model":sha(MODEL/"model.pt"),"normalization":sha(MODEL/"normalization.npz"),"base":sha(BASE),"network":sha(NETWORK)}
 stage=json.loads((audit/"evidence.json").read_text())["source_sha256"]
 if any(stage[k]!=actual[k] for k in actual): raise AssertionError("Stage-A source hash")
 cache=json.loads((CACHE/"evidence.json").read_text())
 if any(cache[f"{k}_sha256"]!=actual[k] for k in actual): raise AssertionError("cache source hash")
 step=json.loads((STEP0025/"evidence.json").read_text())
 if step["source_model_sha256"]!=actual["model"] or step["evaluation_model_sha256"]!=actual["model"]: raise AssertionError("step source hash")
 saved=np.load(STEP0025/"step_0025_predictions.npz",allow_pickle=False)
 if [str(x) for x in saved["event_ids"]]!=ids: raise AssertionError("step event ids")
 return actual
def screen(full,reduction,numerics):
 if full not in numerics or reduction not in numerics: return {"available":False,"reason":"invalid primary_or_companion"}
 ratios={k:numerics[reduction]["metrics"][k]/numerics[full]["metrics"][k] for k in ("mean","pooled","cvar10")}
 valid=all(numerics[n]["scopes"]["5"]["complete"]["max"]<.001 for n in (full,reduction))
 return {"available":True,"ratios":ratios,"thresholds":{"mean":1.02,"pooled":1.02,"cvar10":1.05},"primary_complete5_numerically_valid":bool(valid),"reported_booleans":{k:bool(ratios[k]<={"mean":1.02,"pooled":1.02,"cvar10":1.05}[k]) for k in ratios}}
def joint_equivalence(p, starts):
 """Compare the reused NumPy RHS/RK4 with the actual frozen JointROM once."""
 import torch
 from evaluate_long_latent import _construct, NETWORK
 network=dict(np.load(NETWORK,allow_pickle=False)); network["observed_buses"]=np.unique(np.r_[network["load_buses"],network["generator_buses"]]); network["load_positions"]=np.searchsorted(network["observed_buses"],network["load_buses"])
 model=_construct(MODEL,MechanicalROM.load(BASE),network,"challenger")
 x=starts[:3]
 with torch.no_grad(): actual=model.field(torch.as_tensor(x,dtype=torch.float64)).numpy(); actual_step=model.rk4(torch.as_tensor(x,dtype=torch.float64),.005).numpy()
 expected=core.rhs(x,p); expected_step=core.rk4(x,.005,lambda y:core.rhs(y,p))
 return float(np.abs(actual-expected).max()),float(np.abs(actual_step-expected_step).max())
def run(audit:Path,output:Path):
 if output.exists(): raise FileExistsError(output)
 z=np.load(audit/"matrices.npz",allow_pickle=False); matrices={k:z[k] for k in z.files}; required=("A","Bphysical","Cq","balanced_V_T5","balanced_W_T5","modal_real_block_V","modal_real_block_W","modal_block_column_groups","modal_block_kinds")
 if any(k not in matrices for k in required): raise KeyError("missing Stage-A matrix key")
 p=params(MODEL,matrices); cache=np.load(CACHE/"trajectories.npz",allow_pickle=False); ids=[str(x) for x in cache["event_ids"]]; starts=cache["fine_predictions"][:,0]; stored=cache["fine_predictions"][:,::2]; truth=[cache[f"{i}_truth"] for i in ids]; ev=json.loads((CACHE/"evidence.json").read_text()); meta={r["event_id"]:{"group":r["group"],"label":r["label"]} for r in ev["records"] if r["horizon"]=="prefix"}; m=p["mass"]
 if len(ids)!=107 or [sum(len(t)>round(h/DT) for t in truth) for h in (1,2,5,8)]!=[96,95,95,95] or sum(v["label"]=="unstable" for v in meta.values())!=12: raise AssertionError("cached cohort counts")
 hashes=source_contract(audit,ids)
 field_error,step_error=joint_equivalence(p,starts)
 if field_error>=1e-10 or step_error>=1e-10: raise AssertionError("JointROM field/step mismatch")
 output.mkdir(parents=True)
 same_dt=core.decoded_rollout(starts,1,p,dt=.005)
 replication={"same_dt_005_1s_physical_max_abs":float(np.abs(same_dt[:,:,:30]-stored[:,:101,:30]).max()),"full_0025_cache_physical_max_abs":None}
 (output/"replication_partial.json").write_text(json.dumps(replication,indent=2,allow_nan=False))
 if replication["same_dt_005_1s_physical_max_abs"]>=1e-8: raise AssertionError("full 005 1s replication")
 arms=[("full",{}),("memory_off",{"memory":False})]+[(f"output_component_{i+1}_off",{"memory_mask":np.ones(3)-np.eye(3)[i]}) for i in range(3)]
 V,W=matrices["balanced_V_T5"][:,:2],matrices["balanced_W_T5"][:2];
 if not np.allclose(W@V,np.eye(2),atol=1e-10): raise AssertionError("balanced W V")
 arms.append(("balanced_rank2_T5",{"V":V,"W":W}))
 for name,v,w in core.modal_arms(matrices): arms.append((name,{"V":v,"W":w}))
 arrays={}; rows=[]; invalid=[]; nums={}
 for name,kw in arms:
  fine=core.decoded_rollout(starts,8,p,dt=.0025,**kw); arrays[name]=fine; persist(output,ids,arrays)
  status=finite_status(name,"primary_0025",fine,ids)
  if status: invalid.append(status); continue
  if name=="full":
   old=np.load(STEP0025/"step_0025_predictions.npz",allow_pickle=False)["predictions"]
   replication["full_0025_cache_physical_max_abs"]=float(np.abs(fine[:,:,:30]-old[:,:,:30]).max())
   replication["full_0025_cache_physical_by_horizon"]={str(h):float(np.abs(fine[:,:round(h/DT)+1,:30]-old[:,:round(h/DT)+1,:30]).max()) for h in (1,2,5,8)}
   replication["full_0025_cache_physical_available_prefix_max_abs"]=float(max(np.abs(fine[i,:len(t),:30]-old[i,:len(t),:30]).max() for i,t in enumerate(truth)))
   (output/"replication_partial.json").write_text(json.dumps(replication,indent=2,allow_nan=False))
   # Preregistered absolute replication gate is 1s, not an added all-event
   # 8s absolute gate. Retain all 8s discrepancies and their original scopes.
   replication["full_0025_cache_normalized_scopes"]={str(h):scope_values(fine,old,truth,ids,meta,m,h) for h in (1,2,5,8)}
  for i,(eid,t) in enumerate(zip(ids,truth,strict=True)):
   for h in (1,2,5,8,"prefix"):
    r=metric(fine[i],t,m,h)
    if r is not None: rows.append({"arm":name,"event_id":eid,**meta[eid],"scope":"available_prefix" if h=="prefix" else "complete","complete":h!="prefix",**r})
 for name,kw in arms:
  if name in {item["arm"] for item in invalid}: continue
  coarse=core.decoded_rollout(starts,8,p,dt=.005,**kw); arrays[name+"_005"]=coarse; persist(output,ids,arrays)
  status=finite_status(name,"companion_005",coarse,ids)
  if status: invalid.append(status); continue
  nums[name]={"scopes":{str(h):scope_values(arrays[name],coarse,truth,ids,meta,m,h) for h in (1,2,5,8)}}
  if any(v["all_fixed"]["max"]>=.001 for v in nums[name]["scopes"].values()):
   diag=core.decoded_rollout(starts,8,p,dt=.00125,**kw); arrays[name+"_00125"]=diag; persist(output,ids,arrays)
   status=finite_status(name,"finer_00125",diag,ids)
   if status: invalid.append(status); nums[name]["finer_00125"]={"available":False,"reason":"nonfinite"}; continue
   nums[name]["finer_00125"]={"available":True,"scopes":{str(h):scope_values(arrays[name],diag,truth,ids,meta,m,h) for h in (1,2,5,8)}}
 names=[n for n,_ in arms if n in nums and n not in {v["arm"] for v in invalid}]; descriptive=[n for n,_ in arms if any(r["arm"]==n for r in rows)]; complete={n:{str(h):aggregate([r for r in rows if r["arm"]==n and r["horizon"]==str(h)]) for h in (1,2,5,8)} for n in descriptive}; prefix={n:aggregate([r for r in rows if r["arm"]==n and r["horizon"]=="prefix"]) for n in descriptive}; unstable={n:aggregate([r for r in rows if r["arm"]==n and r["horizon"]=="prefix" and r["label"]=="unstable"]) for n in descriptive}
 for n in names: nums[n]["metrics"]=complete[n]["5"]
 eligible_nums={n:nums[n] for n in names}
 rank2={"balanced_rank2_T5":screen("full","balanced_rank2_T5",eligible_nums),"real_modal_rank2":{}}
 for n,kw in arms:
  if kw.get("V") is not None and kw["V"].shape[1]==2 and n!="balanced_rank2_T5": rank2["real_modal_rank2"][n]=screen("full",n,eligible_nums)
 np.savez_compressed(output/"predictions.npz",event_ids=np.array(ids),**arrays); report={"test_read":False,"field_equivalence_max_abs":field_error,"one_step_equivalence_max_abs":step_error,"replication":replication,"invalid_arms":invalid,"complete":complete,"prefix":prefix,"unstable_prefix":unstable,"rows":rows,"numerical_0025_vs_005":nums,"rank2_T5_screen":rank2,"source_sha256":{**hashes,"cache":sha(CACHE/"trajectories.npz"),"matrices":sha(audit/"matrices.npz"),"code":sha(__file__)}}
 (output/"evidence.json").write_text(json.dumps(report,indent=2,allow_nan=False)); return report
def main():
 q=argparse.ArgumentParser(); q.add_argument("--audit",type=Path,required=True); q.add_argument("--output",type=Path,required=True); a=q.parse_args(); run(a.audit,a.output)
if __name__=="__main__": main()
