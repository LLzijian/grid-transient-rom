"""Same-state additive vector-field diagnostics; no refitting or ablation."""
from collections import defaultdict

import numpy as np
import torch


def fields(model, state):
    n = model.n
    theta, omega, p, a, e = state.split(n, -1)
    parts = model._parts(state)
    z = torch.zeros_like(omega)
    def put(value, block):
        chunks = [z] * 5
        chunks[block] = value
        return torch.cat(chunks, -1)
    def acceleration(value):
        return put(value @ model.inverse_mass.T, 1)
    result = {'kinematics': put(omega, 0),
              'mechanical_output': acceleration(parts.ports.tau_m),
              'damping': acceleration(-omega @ model.damping.T),
              'mechanical_dynamics': put((torch.stack((p, omega, p.tanh(), (omega/model.omega_scale).tanh()), -2)*model.mechanical_coeff).sum(-2), 2),
              'amplitude_dynamics': put(model._amplitude_derivative(parts, omega/model.omega_scale), 3)}
    left, right = model.edge_index
    c = model.c.clamp_min(0)
    beta = torch.minimum(model.beta.clamp_min(0), c)
    for j in range(c.numel()):
        b = model.incidence[:, j]
        base = -c[j]*(parts.phase[..., j].sin()-model.sin_ref[j])
        modulation = -beta[j]*(a[..., left[j]]+a[..., right[j]]).tanh().expm1()*parts.phase[..., j].sin()
        result[f'c_edge_{j}'] = acceleration(base[..., None]*b)
        result[f'beta_edge_{j}'] = acceleration(modulation[..., None]*b)
    result['environment_e'] = acceleration(-model.Se*e)
    result['environment_a'] = acceleration(-model.environment_coeff[0]*a.tanh())
    result['environment_a2'] = acceleration(-model.environment_coeff[1]*a.tanh().square())
    cos_delta = parts.cos_phase-model.cos_ref
    def edge_node(feature, index):
        values = feature[..., :, None]*model.edge_coeff[..., index]
        return z.clone().index_add(-1, model.edge_nodes, values.flatten(-2))
    result['environment_edge_cos'] = acceleration(-edge_node(cos_delta, 4))
    features = (e, a, omega/model.omega_scale, e.tanh(), a.tanh())
    for i, feature in enumerate(features):
        result[f'memory_node_{i}'] = put(feature*model.memory_coeff[i], 4)
    result['memory_edge_sin'] = put(edge_node(parts.phase.sin()-model.sin_ref, 2), 4)
    result['memory_edge_cos'] = put(edge_node(cos_delta, 3), 4)
    return result


def energies(model, trajectory, output_step=.01):
    """Per-event time mean, including true endpoint by trapezoidal integration."""
    scale = torch.cat((torch.ones_like(model.omega_scale), model.omega_scale, model.init_bounds))
    components = fields(model, trajectory)
    full = model.rhs(None, trajectory)
    if not torch.allclose(sum(components.values()), full, atol=1e-11, rtol=1e-10):
        raise ValueError('additive RHS decomposition differs')
    components['coupling_c'] = sum(v for k,v in components.items() if k.startswith('c_edge_'))
    components['coupling_beta'] = sum(v for k,v in components.items() if k.startswith('beta_edge_'))
    components['full_rhs'] = full
    for j in range(model.c.numel()):
        components[f'combined_edge_{j}'] = components[f'c_edge_{j}']+components[f'beta_edge_{j}']
    duration = (trajectory.shape[-2]-1)*output_step
    if duration <= 0:
        raise ValueError('positive actual trajectory duration required')
    return {key: torch.trapezoid((value/scale).square().mean(-1), dx=output_step, dim=-1)/duration
            for key, value in components.items()}


def distribution(value):
    values = np.asarray(value, dtype=float).reshape(-1)
    keys = ('min', 'q05', 'median', 'q95', 'max')
    if not values.size:
        return dict(count=0, mean=None, quantiles=None)
    return dict(count=values.size, mean=float(values.mean()),
                quantiles=dict(zip(keys, np.quantile(values, (0,.05,.5,.95,1)).tolist(), strict=True)))


def parameter_statistics(model):
    raw = {name: value.detach() for name, value in model.named_parameters() if not name.startswith('encoder.')}

    def stats(value):
        x = value.abs()
        maximum = float(x.max())
        return dict(count=x.numel(), exact_zero=int((x==0).sum()), relative_small=int((x<=.001*maximum).sum()),
                    abs_max=maximum, signed=distribution(value), absolute=distribution(x))

    families = {name: stats(value) for name, value in raw.items()}
    environment_subfamilies = {
        **{key: stats(raw['environment_coeff'][i]) for i,key in enumerate(('a_readout','a_squared_readout'))},
        **{key: stats(raw['memory_coeff'][i]) for i,key in enumerate(('e_linear','a_linear','omega_scaled','e_tanh','a_tanh'))},
        **{key: stats(raw['edge_coeff'][...,i]) for i,key in enumerate(('memory_edge_sin','memory_edge_cos','environment_edge_cos'), 2)},
    }
    c=model.c.detach().clamp_min(0)
    beta=torch.minimum(model.beta.detach().clamp_min(0), c)
    ratio=beta[c>0]/c[c>0]
    return dict(families=families, environment_subfamilies=environment_subfamilies,
                total_parameters=sum(x['count'] for x in families.values()),
                modulated_edges=int((beta>0).sum()), beta_over_c=distribution(ratio))


@torch.no_grad()
def measure(model, events, work, local):
    from experiments.osgkd_v2_p2a_equivariant_graph_20261124.run import baseline
    from experiments.osgkd_v2_fast_zero_sum_20261124.scoring import integrate
    if len(events)!=385:
        raise ValueError('complete train385 required')
    groups=defaultdict(list)
    for event in events:
        if event['history'] is None:
            raise ValueError('missing train history')
        groups[min(800,len(event['t'])-1)].append(event)
    totals=defaultdict(list)
    diagnostic_rhs = 0
    for steps, group in sorted(groups.items()):
        for start in range(0,len(group),16):
            batch=group[start:start+16]
            expected=8*steps*len(batch)
            if work.rhs['forward']+expected>2464000:
                raise ValueError('train forward cap exceeded')
            histories=[torch.as_tensor(np.stack([e['history'][key] for e in batch]),dtype=torch.float64) for key in ('theta','omega')]
            state=model.initialize(*histories); work.encoder_events+=len(batch)
            before=work.rhs['forward']
            trajectory=integrate(model,state,steps,internal_step=.005,work=work)
            if work.rhs['forward']-before!=expected:
                raise ValueError('train work differs')
            values=energies(model,trajectory)
            diagnostic_rhs += (steps + 1) * len(batch)
            for key,value in values.items():
                totals[key].extend(value.tolist())
    if any(len(value) != 385 for value in totals.values()):
        raise ValueError('functional contribution coverage differs')
    means={key:sum(value)/385 for key,value in totals.items()}
    denominator=means['full_rhs']
    report={key:dict(E_g=value,C_g=value/denominator if denominator else None) for key,value in means.items()}
    edge_rows=[]
    for j,(left,right) in enumerate(model.edge_index.T.tolist()):
        edge_rows.append(dict(edge_index=j,left=left,right=right,c=float(model.c[j]),beta=float(model.beta[j]),
                              baseline=report[f'c_edge_{j}'], modulation=report[f'beta_edge_{j}'],
                              exchange=report[f'combined_edge_{j}']))
    anonymous = {}
    for family in ('baseline', 'modulation', 'exchange'):
        ordered = sorted(edge_rows, key=lambda row: -row[family]['E_g'])
        for rank, row in enumerate(ordered, 1):
            row[family]['rank'] = rank
        anonymous[family] = dict(distribution=distribution([row[family]['C_g'] for row in ordered if row[family]['C_g'] is not None]),
                                 top_five=[row[family] for row in ordered[:5]])
    edge_rows.sort(key=lambda row: row['exchange']['rank'])
    ref=baseline.atomic_write(local/'edge_contributions.json',edge_rows)
    params=baseline.atomic_write(local/'physical_parameters.json',{name:value.detach().tolist() for name,value in model.named_parameters() if not name.startswith('encoder.')})
    public={key:value for key,value in report.items() if '_edge_' not in key or key in ('environment_edge_cos','memory_edge_sin','memory_edge_cos')}
    return dict(events=385,groups=public,parameter_statistics=parameter_statistics(model),local_edge_ref=ref,
                local_parameter_ref=params,anonymous_edge_rankings=anonymous,diagnostic_rhs=diagnostic_rhs,
                interpretation='nonorthogonal same-state instantaneous vector fields; no refit; event time means then equal event weights',
                normalization='S=(1,omega_scale,init_bounds); reference time 1s; squared norm divided by 5N')


def self_check():
    from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import fit
    constants={key:np.ones(10) for key in ('mass','damping','Sp','Se','omega_scale','S_m')}
    constants.update(F=np.eye(10),theta_ref=np.linspace(-.2,.2,10),candidate_graph='undirected_complete_K10',s_coi=2.)
    model=fit.new_model(constants)
    generator=torch.Generator().manual_seed(913)
    with torch.no_grad():
        for name,value in model.named_parameters():
            if not name.startswith('encoder.'):
                value.copy_(torch.randn(value.shape,generator=generator,dtype=torch.float64)*.03)
        model.c.abs_(); model.beta.copy_(model.c*.4)
    state=torch.randn((2,5,50),generator=generator,dtype=torch.float64)*.1
    pieces=fields(model,state)
    assert torch.allclose(sum(pieces.values()),model.rhs(None,state),atol=1e-12,rtol=1e-11)
    values=energies(model,state)
    scale=torch.cat((torch.ones(10),model.omega_scale,model.init_bounds))
    expected=torch.trapezoid((model.rhs(None,state)/scale).square().mean(-1),dx=.01,dim=-1)/.04
    assert torch.allclose(values['full_rhs'],expected)
    assert parameter_statistics(model)['total_parameters']==680
    constant=state[:,:1].expand(-1,5,-1)
    actual=energies(model,constant)['full_rhs']
    assert torch.allclose(actual,(model.rhs(None,state[:,0])/scale).square().mean(-1))
    return dict(additive_rhs=True,normalization=True,trapezoid=True,physical_parameters=680,scientific_payloads_opened=False)
