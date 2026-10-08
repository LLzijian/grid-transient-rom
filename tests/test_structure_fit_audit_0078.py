import sys, json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from check_structure_fit_0078 import audit, FROZEN_COMMON, _horizon


def test_independent_audit_rejects_extra_R_duplicate_of_base_or_previous_group():
    from check_structure_fit_0078 import _novel_auto_dissipation
    state={'structured.damping_mask':torch.ones(10),'structured.rpp_mask':torch.ones(10)}
    def direction(i):
        v=torch.zeros(23); v[i]=1.; return v
    state['structured.dissipation_a.0']=direction(19)
    assert not _novel_auto_dissipation(state)
    state['structured.dissipation_a.0']=direction(20)
    assert _novel_auto_dissipation(state)
    state['structured.dissipation_a.1']=direction(20)
    assert not _novel_auto_dissipation(state)

def _checkpoint(path,arm,step,state):
    path.mkdir(); torch.save(state,path/'model.pt'); req={'head.bias':False}; req.update({k:False for k in state if 'exchange_a' in k or 'exchange_b' in k or 'dissipation_a' in k}); (path/'checkpoint.json').write_text(json.dumps({'arm':arm,'step':step,'requires_grad':req,'counts':{'active_groups':{'ports':2}}}))

def _monitors():
    row={'per_event':[{'event_id':str(i),'horizon_s':float(h),'error':1.} for h in (1,2,5) for i in range(346)],'horizons':{str(h):{'n_events':346,'mean':1.,'tail35':1.,'risk':1.} for h in (1,2,5)},'average_risk':1.}
    return {'0':row,'800_before_growth':row,'1200':row}

def test_synthetic_g_audit_accepts_frozen_state_and_registered_trace(tmp_path):
    state={k:torch.ones(1) for k in FROZEN_COMMON}; state.update({'head.bias':torch.zeros(1),'structured.mass':torch.ones(2),'structured.D':torch.ones(1,2),'structured.delta_star':torch.zeros(2),'structured.phi':torch.zeros(1),'structured.scales':torch.ones(3),'structured.gain':torch.ones(2),'structured.coupling_mask':torch.ones(1),'structured.damping_mask':torch.ones(2),'structured.jpi_p_mask':torch.ones(2),'structured.rpp_mask':torch.ones(2),'structured.storage_base_mask':torch.ones(3),'structured.exchange_a.0':torch.tensor([1.]),'structured.exchange_b.0':torch.tensor([0.])})
    _checkpoint(tmp_path/'zero','G-auto',0,state); _checkpoint(tmp_path/'step1200','G-auto',1200,state)
    trace=[{'step':i,'horizon_s':_horizon(i),'loss':1.,'gradient_norm_before_cap':1.} for i in range(1,1201)]+[{'step':400,'growth':{'proposals':[{}]*8}},{'step':800,'growth':{'proposals':[{}]*8}},{'step':1000,'pruning':{}}]
    (tmp_path/'e.json').write_text(json.dumps({'arm':'G-auto','final_checkpoint':str(tmp_path/'step1200'),'traces':trace,'monitors':_monitors(),'hashes':{'model':'synthetic'},'hashes_after':{'model':'synthetic'}}))
    assert audit(tmp_path/'zero',tmp_path/'step1200',tmp_path/'e.json')['passed']

def test_synthetic_c_support_growth_is_rejected(tmp_path):
    buffers={k:torch.ones(1) for k in FROZEN_COMMON}; buffers.update({k:torch.ones(1) for k in ('core.mass','core.incidence','core.delta_star','core.coupling','core.damping','core.rate','core.gain','core.input_scale','core.z_scale','core.voltage_star','core.load_positions','core.load_g','core.load_star','core.load_scale')}); s0={**buffers,'head.bias':torch.zeros(1),'core.support':torch.tensor([1.,0.]),'core.alpha_support':torch.tensor([1.])}; sf={**buffers,'head.bias':torch.zeros(1),'core.support':torch.tensor([1.,1.]),'core.alpha_support':torch.tensor([1.])}
    _checkpoint(tmp_path/'zero','C-free',0,s0); _checkpoint(tmp_path/'step1200','C-free',1200,sf)
    trace=[{'step':i,'horizon_s':_horizon(i),'loss':1.,'gradient_norm_before_cap':1.} for i in range(1,1201)]+[{'step':1000,'pruning':{}}]
    (tmp_path/'e.json').write_text(json.dumps({'arm':'C-free','final_checkpoint':'step1200','traces':trace,'monitors':_monitors()}))
    assert not audit(tmp_path/'zero',tmp_path/'step1200',tmp_path/'e.json')['checks']['core.support_subset_only']
