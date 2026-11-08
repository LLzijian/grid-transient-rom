"""Pure algebra and mocked schedule checks; registered graph cases remain unrun."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
import gftd_sparse_topology_20261107 as run


def counts():
    return dict(generation_attempts=0,generated_grids=0,observation_packages=0,A_grid_attempts=0,A_grid_checks=0,
        reconstruction_attempts=0,reconstructions=0,settings_attempted=0,settings_completed=0,
        algebra_terms_planned=3120,algebra_terms_actual=0,algebra_terms_skipped=0,
        nnls_attempts=0,nnls_completed=0,nnls_qualified=0)


def algebra(k=None,scale=.01):
    k=np.array([.4,0.,.8,0.,0.,.6]) if k is None else np.asarray(k)
    A=np.zeros((16,6))
    A[4:10]=np.eye(6)*scale
    Z=np.zeros((16,4))
    Z[:4]=np.eye(4)
    b=A@k+Z@np.array([.1,-.2,.3,-.4])
    return A,Z,b


def packet(n=4):
    return dict(theta_history=np.zeros((11,n)),omega_history=np.zeros((11,n)),
                history_times=np.linspace(0.,.1,11),p_end=np.zeros(n),sigma=0.)


def test_k4_edge_order_and_node_generic_dictionary_port_formula():
    f=run.functional()
    assert f.edges==((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    system=run.build_system(packet(),.0025)
    assert system["A"].shape==(16,6) and system["Z"].shape==(16,4)
    np.testing.assert_array_equal(system["A"],np.zeros((16,6)))
    np.testing.assert_array_equal(system["b"],np.zeros(16))
    t=np.linspace(0.,.1,41)
    omega=np.full((41,4),.027)
    p_end=np.array([.013,-.017,.019,.023])
    p,rf=run.reconstruct_ports(t,omega,p_end,f)
    np.testing.assert_allclose(p,np.exp(.3*(.1-t))[:,None]*p_end+np.expm1(.3*(.1-t))[:,None]*.027/.3,atol=2e-14)
    np.testing.assert_allclose(rf,.7/.3*(-np.expm1(-.3*t))[:,None]*np.full((1,4),.027),atol=2e-14)
    np.testing.assert_array_equal(p[-1],p_end)


@pytest.mark.parametrize("field",["k","support","graph","true_r","c","excitation","future"])
def test_illegal_truth_fields_rejected(field):
    obs=packet()
    obs[field]=0
    with pytest.raises(ValueError,match="only theta"):
        run.build_system(obs,.0025)


def test_data_relative_kkt_rejects_small_scale_wrong_zero_and_nnls_solves_same_target():
    X=np.array([[2e-6]])
    y=np.array([2e-6])
    bad=run.kkt_audit(X,y,np.zeros(1))
    assert bad["violation"]<1e-9
    assert bad["relative_violation"]==1.
    assert not bad["qualified"]
    full=np.zeros((1,6))
    full[:,0]=X[:,0]
    tally=counts()
    solved=run.solve_subset(full,y,[0],tally)
    assert solved["audit"]["qualified"]
    np.testing.assert_allclose(solved["k"][0],1.,atol=1e-13)
    assert tally["nnls_attempts"]==tally["nnls_completed"]==1
    zero=run.kkt_audit(np.zeros((2,1)),np.zeros(2),np.zeros(1))
    assert zero["trivial_optimum"] and zero["relative_violation"] is None


def test_complete_subset_enumeration_and_reported_support_are_separate_from_raw():
    A,Z,b=algebra()
    tally=counts()
    first=run.infer_setting(A,Z,b,0.,tally)
    second=copy.deepcopy(first)
    pair=run.qualify_pair(first,second)
    assert pair["support_qualified"]
    assert first["support"]==[0,2,5]
    np.testing.assert_allclose(first["raw_k"],[.4,0.,.8,0.,0.,.6],atol=1e-13)
    assert len(first["enumeration"])==64
    assert tally["algebra_terms_actual"]==65
    assert tally["nnls_attempts"]==64 # 63 nonempty candidates and one nonempty debias.
    assert tally["nnls_qualified"]==64


def test_excited_zero_graph_is_empty_but_no_information_is_null_with_skipped_terms():
    A,Z,b=algebra(np.zeros(6))
    first=run.infer_setting(A,Z,b,0.,counts())
    run.qualify_pair(first,copy.deepcopy(first))
    assert first["support"]==[] and first["support_qualified"]
    tally=counts()
    null=run.infer_setting(np.zeros_like(A),Z,np.zeros_like(b),0.,tally)
    assert null["support"] is None and null["raw_support"] is None
    assert all(not row["actual"] for row in null["enumeration"])
    assert not null["debias"]["actual"]
    assert tally["algebra_terms_skipped"]==65 and tally["nnls_attempts"]==0


def test_unselected_unexcited_edge_still_blocks_qualified_support():
    A,Z,b=algebra(np.zeros(6))
    A[:,5]=0.
    first=run.infer_setting(A,Z,b,0.,counts())
    pair=run.qualify_pair(first,copy.deepcopy(first))
    assert first["raw_support"]==[]
    assert first["support"] is None and not pair["support_qualified"]
    assert "coarse.all_edges_excited" in pair["reasons"]


def test_confounded_ties_keep_raw_representative_without_qualified_discovery():
    A,Z,_=algebra()
    A[:,1]=A[:,0]
    b=A[:,0]*.6
    first=run.infer_setting(A,Z,b,0.,counts())
    pair=run.qualify_pair(first,copy.deepcopy(first))
    assert len(first["tied_supports"])>=2
    assert first["representative_subset"]==[0]
    assert first["information"]["confounded"]
    assert first["support"] is None and not pair["support_qualified"]


def test_one_unresolved_subset_propagates_even_when_available_winner_is_empty(monkeypatch):
    A,Z,b=algebra(np.zeros(6))
    real=run.nnls
    calls=[]
    def faulty(X,y,**kwargs):
        calls.append(1)
        if len(calls)==1:
            raise RuntimeError("mock one-subset failure")
        return real(X,y,**kwargs)
    monkeypatch.setattr(run,"nnls",faulty)
    first=run.infer_setting(A,Z,b,0.,counts())
    pair=run.qualify_pair(first,copy.deepcopy(first))
    assert first["raw_support"]==[]
    assert first["enumeration_unresolved"] and not first["algorithm_unique"]
    assert not pair["support_qualified"]


def test_rank_sensitivity_on_both_grids_is_not_an_approval():
    A,Z,b=algebra(np.zeros(6))
    first=run.infer_setting(A,Z,b,0.,counts())
    first["information"]["rank_threshold_sensitive"]=True
    pair=run.qualify_pair(first,copy.deepcopy(first))
    assert pair["numerically_qualified"]
    assert not pair["support_qualified"]
    assert "coarse.rank_insensitive" in pair["reasons"]


def test_debias_threshold_drop_is_retained_without_iterated_support_changes(monkeypatch):
    A,Z,b=algebra()
    calls=[]
    def stub(X,y,subset,tally):
        calls.append(list(subset))
        k=np.zeros(6)
        if subset==[0]:
            k[0]=.04 if len(calls)==65 else .06
        loss=0. if subset==[0] else 1.
        return dict(subset=list(subset),actual=True,k=k,loss=loss,status="qualified",audit=dict(qualified=True))
    monkeypatch.setattr(run,"solve_subset",stub)
    first=run.infer_setting(A,Z,b,0.,counts())
    run.qualify_pair(first,copy.deepcopy(first))
    assert len(calls)==65
    assert first["raw_support"]==[0] and first["raw_k"][0]==.04
    assert first["support_threshold_inconsistent"] and first["support"] is None


def test_mock_generation_checks_150000_seed_order_and_noise_pairing(monkeypatch,tmp_path):
    monkeypatch.setattr(run,"conditions",lambda:[(i//4,i%4,np.zeros(6),np.zeros(20)) for i in range(16)])
    monkeypatch.setattr(run,"rk4",lambda f,state,step:state.copy())
    seeds=[]
    monkeypatch.setattr(run.np.random,"PCG64",lambda seed:seed)
    def generator(seed):
        seeds.append(seed)
        return SimpleNamespace(standard_normal=lambda shape:np.full(shape,seed/1e6))
    monkeypatch.setattr(run.np.random,"Generator",generator)
    tally=counts()
    _,observed_path,plan_path=run.generate(tmp_path,tally)
    observed=json.loads(observed_path.read_text())
    plan=json.loads(plan_path.read_text())
    assert seeds==list(range(150000,150016))
    assert len(observed)==48 and tally["generated_grids"]==32
    for entry in plan:
        obs=observed[entry["input_id"]]
        actual=np.column_stack((obs["theta_history"],obs["omega_history"]))
        np.testing.assert_allclose(actual,np.full((11,8),.1*entry["sigma"]*entry["seed"]/1e6),rtol=1e-15)


def test_fit_B_keeps_48_settings_and_96_reconstructions_under_mock(monkeypatch,tmp_path):
    A,Z,b=algebra()
    monkeypatch.setattr(run,"build_system",lambda packet,step:dict(A=A,Z=Z,b=b))
    def infer(A,Z,b,sigma,tally):
        tally["algebra_terms_skipped"]+=65
        return dict(raw_support=None,raw_k=None)
    monkeypatch.setattr(run,"infer_setting",infer)
    monkeypatch.setattr(run,"qualify_pair",lambda *args:dict(support_qualified=False))
    observed={}
    plan=[]
    for batch in range(4):
        for excitation in range(4):
            for sigma in run.SIGMAS:
                key=f"obs_{len(observed):03d}"
                observed[key]=packet()
                plan.append(dict(input_id=key,batch=batch,excitation=excitation,sigma=sigma))
    tally=counts()
    result=run.fit_B(observed,plan,tmp_path,tally)
    assert len(result["results"])==48 and tally["settings_completed"]==48
    assert tally["reconstructions"]==96
    assert tally["algebra_terms_skipped"]==3120 and tally["nnls_attempts"]==0


def test_estimates_hash_mismatch_precedes_truth_load(tmp_path):
    path=tmp_path/"estimates.json"
    path.write_text("{}")
    with pytest.raises(ValueError,match="changed"):
        run.score_frozen(path,"wrong",tmp_path/"missingtruth",tmp_path/"missingplan")
