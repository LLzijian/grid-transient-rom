import numpy as np
import torch
from task4.synthetic import fixture
from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder,PredictiveHead
from task4.objectives import AnchorTargets,ShootingBank,batch_objective,one_event_loss
from task4.symbolic import PolynomialFunction,numpy_value_gradient,fit_function
from task4.evaluation import aggregate

def test_anchor_censor_mask():
    es,p=fixture(steps=16);a=AnchorTargets(p,[.02,.05,.1,.5]);target,mask=a.get(es[0],10)
    assert mask.tolist()==[True,True,False,False]
    assert target.shape==(3,4,3)
    assert np.max(np.abs(target[:,2:]))==0

def test_polynomial_numeric_gradient():
    pw=[[0,0],[1,0],[0,1],[2,0],[1,1]];co=[[1.],[2.],[3.],[4.],[5.]]
    m=PolynomialFunction(pw,co);x=torch.tensor([[.2,-.3]],dtype=torch.float64,requires_grad=True)
    y=m(x);g=torch.autograd.grad(y.sum(),x)[0]
    yn,gn=numpy_value_gradient(m.export(),x.detach().numpy())
    np.testing.assert_allclose(yn,y.detach().numpy(),atol=1e-12)
    np.testing.assert_allclose(gn[:,0],g.numpy(),atol=1e-12)

def test_scalar_compression_engine():
    m=PolynomialFunction([[0,0],[1,0],[0,2]],[[1.],[.5],[.2]])
    x=torch.randn(100,2,dtype=torch.float64)*.2
    s,facts=fit_function(m,x,max_terms=4,degree=2,ridge=1e-10,steps=0)
    assert facts['value_mse']<1e-10

def test_multi_shooting_backprop():
    es,p=fixture(steps=51);m=PowerGraph(p,[.2,.3,.1],hidden=8);e=HistoryEncoder(m,8);bank=ShootingBank()
    opt=torch.optim.Adam(list(m.parameters())+list(e.parameters()),lr=1e-3)
    a,c=one_event_loss(m,e,es[0],.4,.01,True,.25,bank,opt)
    (a+c).backward()
    assert len(bank.states)==1
    assert all(v.grad is not None for v in bank.states.values())

def test_numerical_failures_not_dropped():
    rows=[{'id':'a','qualified':True,'metrics':{'rmse':1.,'normalized_sse':1.,'n':1}},
          {'id':'b','qualified':False,'metrics':{'rmse':100.,'normalized_sse':10000.,'n':1}}]
    a=aggregate(rows);assert a['pooled'] is None and a['count']==2 and a['unresolved_ids']==['b']

def test_segment_wider_than_target_matches_autonomous_batch():
    es,p=fixture(events=2,steps=21);torch.manual_seed(12)
    m=PowerGraph(p,[.2,.3,.1],hidden=8);e=HistoryEncoder(m,8);head=PredictiveHead(m,[.02,.05],8)
    cfg={'integration_dt':.01,'continuity_weight':1.,'arm':'A','anchor_weight':0.,'latent_weight':0.,'torque_weight':0.,'complexity_weight':0.}
    opt=torch.optim.Adam(list(m.parameters())+list(e.parameters()),lr=1e-3)
    expected=[]
    for event in es:expected.append(one_event_loss(m,e,event,.04,.01,True)[0])
    from task4.objectives import tail_risk
    ref=tail_risk(torch.stack(expected))
    got,terms=batch_objective(m,e,head,es,.04,cfg,AnchorTargets(p,[.02,.05]),np.random.default_rng(3),ShootingBank(),opt,.25,0.)
    torch.testing.assert_close(got,ref,atol=1e-13,rtol=1e-10)
    assert terms['continuity']==0
