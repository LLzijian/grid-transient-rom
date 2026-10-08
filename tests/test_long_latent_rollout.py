import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"experiments"))
spec=importlib.util.spec_from_file_location("long",ROOT/"experiments"/"audit_long_latent_rollout.py"); x=importlib.util.module_from_spec(spec); spec.loader.exec_module(x)
def test_variable_truth_and_no_initial_error():
 t=np.zeros((15,30)); p=np.zeros((801,33)); p[0,0]=99
 assert x.metric(p,t,np.ones(10),1) is None
 assert x.metric(p,t,np.ones(10),"prefix")["rmse"]==0
def test_rank2_projection_and_complex_pair_contract():
 V=np.array([[1.,0.],[0.,1.],[.2,.3]]); W=np.linalg.inv(V.T@V)@V.T
 assert np.allclose(W@V,np.eye(2))
 import audit_latent_order_rollout as old
 c={"modal_real_block_V":np.eye(3),"modal_real_block_W":np.eye(3),"modal_block_column_groups":np.array([[0,2],[2,3]]),"modal_block_kinds":np.array(["conjugate_block","simple_real"])}
 arms=old.modal_arms(c); assert arms[0][1].shape[1]==1 and arms[1][1].shape[1]==2
def test_nonfinite_persist_and_true_prefix_ignores_tail(tmp_path):
 ids=["a","b"]; arr=np.zeros((2,3,33)); arr[1,1]=np.nan
 stages=[x.finite_status("arm",stage,arr,ids) for stage in ("primary_0025","companion_005","finer_00125")]
 assert [s["stage"] for s in stages]==["primary_0025","companion_005","finer_00125"]
 assert all(s["event_ids"]==["b"] for s in stages)
 x.persist(tmp_path,ids,{"arm":arr}); assert (tmp_path/"predictions_partial.npz").exists()
 a=np.zeros((2,801,33)); b=a.copy(); a[:,101:,0]=1e9
 scoped=x.scope_values(a,b,[np.zeros((101,30)),np.zeros((11,30))],ids,{"a":{"label":"stable"},"b":{"label":"unstable"}},np.ones(10),2)
 assert scoped["available_prefix"]["max"]==0 and scoped["unstable_prefix"]["max"]==0
 assert scoped["all_fixed"]["max"]>1e6
 assert x.scope_values(a,b,[np.zeros((2,30)),np.zeros((2,30))],ids,{"a":{"label":"stable"},"b":{"label":"stable"}},np.ones(10),1)["unstable_prefix"]["max"] is None
def test_rank1_not_selected_for_rank2_screen():
 arms=[("rank1",{"V":np.ones((3,1))}),("rank2",{"V":np.ones((3,2))})]
 assert [name for name,kw in arms if kw.get("V") is not None and kw["V"].shape[1]==2]==["rank2"]
