"""Package an explicit S checkpoint as an independent NumPy ODE artifact."""
import _bootstrap
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from task4.paths import ROOT, project_path, write_json
from task4.training import check_authorized, load_checkpoint
from task4.symbolic_runtime import export_symbolic, SymbolicPowerGraph


SIMULATOR = '''"""Standalone NumPy RK4. Only the first 100 ms history initializes the ODE."""
import argparse
from pathlib import Path
import numpy as np
from runtime import SymbolicPowerGraph

def simulate(model, history, seconds=5., dt=.005):
    if seconds <= 0 or not 0 < dt <= .01:
        raise ValueError("positive duration and 0 < dt <= .01 required")
    h = np.asarray(history, dtype=np.float64)
    if h.ndim == 3: h = h[None]
    x = model.initial(h)
    count = int(np.ceil(seconds/dt)); step = seconds/count
    states = [x.copy()]
    for _ in range(count):
        a = model.field(x)
        b = model.field(x+step*a/2)
        c = model.field(x+step*b/2)
        d = model.field(x+step*c)
        x = x+step*(a+2*b+2*c+d)/6
        if not all(np.isfinite(v).all() for v in (a,b,c,d,x)):
            raise FloatingPointError("Nonfinite integration; no fabricated tail returned")
        states.append(x.copy())
    return .1+np.arange(count+1)*step, np.stack(states, axis=1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(__file__).with_name("model.json")))
    p.add_argument("--history", required=True, help="NPY [11,N,3] or [B,11,N,3], physical delta/omega/p")
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=5.)
    p.add_argument("--dt", type=float, default=.005)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    model = SymbolicPowerGraph.from_json(a.model)
    times, states = simulate(model, np.load(a.history, allow_pickle=False), a.seconds, a.dt)
    np.savez_compressed(output, time=times, state=states, physical=model.physical(states))
'''


def equations(spec):
    lines = ['# Explicit TASK4 power-structure ODE', '',
             'This artifact is an executable formula candidate. Accuracy and stability require the separate endpoint reports.', '',
             'The GRU initializes q once from eleven samples. The subsequent ODE contains only explicit scalar formulas, fixed graph operations and coefficients.', '',
             '## Coordinates and units', '', '```json',
             json.dumps(spec['coordinates'], ensure_ascii=False, indent=2), '```', '',
             '## Outer equations', '']
    lines.extend(f'- **{key}**: {value}' for key,value in spec['structure'].items())
    lines.extend(['', 'All fixed graph, physical and initializer coefficients are in model.json. Below x0, x1, ... follow the declared function input order. Output numbers are zero-based; edge exchange outputs are row-major matrix entries.', '',
                  'All printed coefficients use 17 significant digits; no magnitude threshold deletes a nonzero coefficient.', ''])
    for name, fn in spec['functions'].items():
        lines.extend(['## '+name, '', '```text'])
        pw=np.asarray(fn['powers'],int); co=np.asarray(fn['coefficients'])
        for out in range(fn['outputs']):
            terms=[]
            for powers, coefficient in zip(pw, co[:,out]):
                if coefficient==0:continue
                monomial='*'.join('x'+str(i)+(f'^{int(power)}' if power!=1 else '') for i,power in enumerate(powers) if power)
                terms.append(format(float(coefficient),'.17g')+('*'+monomial if monomial else ''))
            lines.append(f'F[{out}] = '+(' + '.join(terms) if terms else '0'))
        lines.extend(['```',''])
    return '\n'.join(lines)


def package(checkpoint, output):
    check_authorized(False)
    deadline=float(os.environ.get('TASK4_SAVE_DEADLINE_UNIX','nan'))
    if not np.isfinite(deadline) or time.time()+10>=deadline:raise TimeoutError('Managed deadline required')
    source=project_path(checkpoint); destination=project_path(output)
    if destination.exists():raise FileExistsError(destination)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    model,encoder,_,obj=load_checkpoint(source)
    if source_hash!=hashlib.sha256(source.read_bytes()).hexdigest():raise RuntimeError('Source changed during load')
    if obj['synthetic'] or obj.get('kind')!='T4-S':raise ValueError('Requires real all-symbolic checkpoint')
    torch.set_num_threads(1)
    destination.mkdir(parents=True)
    write_json(destination/'package_validation.json',{'complete':False,'source_sha256':source_hash,'status':'packaging_in_progress'})
    spec=export_symbolic(model,encoder,destination/'model.json',omega_s=obj['physics']['omega_s'],
        metadata={'checkpoint_sha256':source_hash,'checkpoint':str(source),'step':obj['step'],
                  'autonomous_only_training':bool(obj['config'].get('autonomous_only')),
                  'final_steps_complete':obj['step']==obj['config']['steps'],
                  'prediction_accuracy_or_stability_claim':False})
    runtime=SymbolicPowerGraph(spec)
    # Fixed synthetic probes test implementation agreement, never predictive accuracy.
    rng=np.random.default_rng(8041)
    histories=rng.normal(size=(3,11,model.n,3))*np.asarray(spec['scales'])[None,None,None,:]*.03
    z0=runtime.initial(histories)
    torch_z0=model.initial(torch.tensor(histories,dtype=torch.float64),encoder).detach()
    scale=np.concatenate([np.full(model.n,spec['scales'][0]),model.mass.numpy()*spec['scales'][1],np.full(model.n,spec['scales'][2]),np.ones(model.n*model.rank)])
    states=np.concatenate([np.zeros((1,model.dim)),z0,rng.normal(size=(32,model.dim))*scale*.02])
    field,details=runtime.field(states,details=True)
    tf,td=model.field(torch.tensor(states,dtype=torch.float64),create_graph=False,details=True)
    errors={
        'initial_max_abs':float(np.max(np.abs(z0-torch_z0.numpy()))),
        'field_max_abs':float(np.max(np.abs(field-tf.numpy()))),
        'gradient_max_abs':float(np.max(np.abs(details['gradient']-td['gradient'].numpy()))),
        'energy_max_abs':float(np.max(np.abs(details['H']-td['H'].numpy()))),
        'power_balance_max_abs':float(np.max(np.abs(details['Hdot']-details['expected_Hdot']))),
    }
    np.testing.assert_allclose(z0,torch_z0.numpy(),rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(field,tf.numpy(),rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(details['gradient'],td['gradient'].numpy(),rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(details['H'],td['H'].numpy(),rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(details['Hdot'],details['expected_Hdot'],rtol=1e-9,atol=1e-10)
    shutil.copyfile(ROOT/'src/task4/symbolic_runtime.py',destination/'runtime.py')
    (destination/'simulate.py').write_text(SIMULATOR,encoding='utf-8')
    (destination/'equations.md').write_text(equations(spec),encoding='utf-8')
    (destination/'README.md').write_text(
        '# TASK4 explicit ODE candidate\n\nRequires Python and NumPy only. Run `python simulate.py --history history.npy --output forecast.npz`. '
        'The supplied history must use the original canonical physical units and contain 11 samples at 10 ms spacing; '
        'p at initialization is assumed known. Only delta and omega histories enter the GRU. Forecast time starts at 0.1 s.\n\n'
        'model.json contains explicit formula coefficients and one-time GRU weights. equations.md renders every nonzero scalar coefficient. '
        'runtime.py computes the autonomous field without Torch, automatic differentiation or a teacher. '
        'No accuracy, speed or stability certificate follows from packaging; consult the separately frozen evaluation.\n',encoding='utf-8')
    if source_hash!=hashlib.sha256(source.read_bytes()).hexdigest():raise RuntimeError('Source checkpoint changed')
    result={'complete':True,'source_sha256':source_hash,'test_payload_read':False,'event_payload_read':False,
            'probe_kind':'fixed synthetic implementation checks only','errors':errors,
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file() and p.name!='package_validation.json'}}
    write_json(destination/'package_validation.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    print(json.dumps(package(**vars(parser.parse_args())),indent=2))
