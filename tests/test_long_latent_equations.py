import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'experiments'))
from audit_latent_order_rollout import decoded_rollout, rhs


def parameters():
    incidence=np.zeros((10,1)); incidence[0,0]=1.; incidence[1,0]=-1.
    return {'mass':np.arange(1.,11.),'incidence':incidence,'angle_incidence':incidence.copy(),
            'delta_star':np.zeros(10),'coupling':np.array([2.]),'damping':np.ones(10),
            'rate':np.ones(10),'gain':np.ones(10),'feature_scale':np.ones(22),
            'alpha':np.zeros((2,3)),'alpha_support':np.ones((2,3)),
            'Cq':np.eye(10,3),'xi':np.zeros((26,3)),'support':np.ones((26,3))}


def test_rhs_direct_physical_alpha_memory_signs_and_masked_qdot():
    p=parameters(); p['alpha'][0,0]=1.; p['xi'][1,0]=2.; p['xi'][4,1]=3.
    y=np.zeros((1,33)); y[0,0]=.2; y[0,10]=.5; y[0,20]=4.; y[0,30]=2.
    out=rhs(y,p)[0]; af=np.sin(.2); e0=2*np.sin(.2)
    # action is memory q plus alpha(sin), and it is subtracted from swing force.
    assert np.isclose(out[0],.5)
    assert np.isclose(out[10],(4-e0-.5-(2+af))/1.)
    assert np.isclose(out[20],-4-.5)
    assert np.isclose(out[30],4.) and np.isclose(out[31],1.5)
    no_memory=rhs(y,p,memory=False)[0]
    assert np.isclose(no_memory[10],(4-e0-.5-af))


def test_projected_rhs_and_decoding_match_full_state_on_subspace():
    p=parameters(); p['coupling'][:]=0.; p['damping'][:]=0.; p['rate'][:]=0.; p['gain'][:]=0.
    p['xi'][1:4]=np.array([[1.,2.,0.],[0.,1.,0.],[0.,0.,3.]])
    V=np.array([[1.,0.],[0.,1.],[0.,0.]]); W=np.array([[1.,0.,0.],[0.,1.,0.]])
    x=np.zeros(30); x[10]=.4; z=np.array([.3,-.2]); full=np.r_[x,V@z]; reduced=np.r_[x,z]
    ffull=rhs(full,p); fred=rhs(reduced,p,V=V,W=W)
    assert np.allclose(fred[:30],ffull[:30]) and np.allclose(fred[30:],ffull[30:]@W.T)
    decoded=decoded_rollout(full[None],.02,p,V=V,W=W,dt=.005)
    assert np.allclose(decoded[:,0],full[None])


def test_decoded_rollout_grid_and_constant_omega_at_all_registered_steps():
    p=parameters(); p['coupling'][:]=0.; p['damping'][:]=0.; p['rate'][:]=0.; p['gain'][:]=0.
    start=np.zeros((1,33)); start[0,10]=.7
    for dt in (.0025,.005,.00125):
        values=decoded_rollout(start,.02,p,dt=dt)
        assert values.shape==(1,3,33)
        assert np.allclose(values[0,:,10],.7)
        assert np.allclose(values[0,:,0],[0.,.007,.014])
