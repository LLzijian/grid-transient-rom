import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'experiments'))
from audit_long_latent_matrices import representatives, physical_scales
from audit_latent_order import _modal_blocks, fd_Aq_check, fd_B_check, _synthetic_base


def test_representatives_cover_genuine_full_trajectory_endpoints():
    event={'id':'e','x':np.zeros((1501,30))}
    points=[i for _,i in representatives([event])]
    assert len(points)==31 and points[0]==10 and points[-1]==1500 and len(set(points))==31


def test_event_balanced_fulltime_scales_apply_floor():
    first={'x':np.zeros((12,30))}; second={'x':np.zeros((14,30))}
    second['x'][10:,10]=4.
    scales=physical_scales([first,second],np.ones(10))
    assert scales.shape==(30,) and scales[0]==.01 and np.isclose(scales[10],np.sqrt(8.))


def test_complex_blocks_and_feature_derivatives_reuse_verified_contracts():
    _,_,_,blocks,_=_modal_blocks(np.array([[-1.,-2.,0.],[2.,-1.,0.],[0.,0.,-3.]]))
    assert sorted(block['kind'] for block in blocks)==['conjugate_block','simple_real']
    rng=np.random.default_rng(1); base=_synthetic_base(); state={'xi':rng.normal(size=(28,3)),'support':np.ones((28,3))}
    norm={'feature_scale':np.ones(24)}; point=rng.normal(size=30)
    assert fd_B_check(state,norm,base,point)<1e-7 and fd_Aq_check(state,norm,base,point)<1e-7
