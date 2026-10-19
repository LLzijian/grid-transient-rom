"""Disposable P0 engineering inputs and fixed four-mode timing; no heldout data."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src'), str(ROOT / 'tests')]

# This module's supervisor uses only stdlib before the measured child starts.
def supervised(args, script):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if not 0 < args.max_seconds <= 1800 or not 1 <= args.attempt <= 2:
        raise ValueError('finite remaining execution cap <=1800; attempts 1 or 2 only')
    command = [sys.executable, str(script), '--child', '--output', str(output),
               '--max-seconds', str(args.max_seconds), '--attempt', str(args.attempt), '--device', args.device]
    start = time.perf_counter()
    status = 'failed'
    try:
        with (output/'child.log').open('w', encoding='utf-8') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.max_seconds)
        status = 'passed' if proc.returncode == 0 else 'failed'
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        status, returncode = 'timeout', 124
    finally:
        receipt = {'native_parent_wall_seconds': time.perf_counter()-start,
                   'scope': 'child startup, imports, initialization, checks/updates and persistence counted once',
                   'attempt': args.attempt, 'cap_seconds': args.max_seconds, 'status': status,
                   'command': command}
        (output/'wall_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt), flush=True)
    return returncode


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--attempt', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--child', action='store_true')
    return p


def engineering_inputs(device):
    import numpy as np
    import torch
    from closure_discovery_benchmarks import toy_source
    from closure_causal_recovery_20261019 import copy_runtime, trajectory
    from tanh_port_runtime_20261019 import TERM
    source = toy_source()
    reference = copy_runtime(source, [TERM], device)
    rng = np.random.default_rng(7401)
    initial = rng.normal(size=(12, 40)) * np.where(np.arange(12)[:, None] < 6, .08, .3)
    initial[:, 30:] = rng.normal(size=(12, 10)) * np.where(np.arange(12)[:, None] < 6, .2, 1.2)
    initial[6, 30], initial[7, 31] = 3., -3.
    z = torch.tensor(initial, dtype=torch.float64, device=device)
    with torch.no_grad():
        reference.coefficients.fill_(.4)
        observed = trajectory(reference, z, 60).detach()
    scales = observed.std((0, 1), correction=0).clamp_min(1e-6)
    # Shared start independent of the truth coordinates; hidden q never enters loss.
    from closure_causal_recovery_20261019 import runtime_model
    start = runtime_model(source, [TERM], 7401, device)
    torch.manual_seed(7402)
    with torch.no_grad():
        for parameter in start.parameters():
            if parameter.requires_grad:
                parameter.add_(.005 * torch.randn_like(parameter))
    sources = [ROOT/'experiments'/n for n in (
        'tanh_port_runtime_20261019.py', 'benchmark_tanh_port_20261019.py',
        'closure_training_runtime.py', 'symbolic_closure_rom.py', 'structured_rom.py',
        'local_memory_rom_0098.py', 'closure_causal_recovery_20261019.py',
        'closure_causal_history_residual_20261019.py', 'closure_discovery_benchmarks.py')]
    sources.append(ROOT/'tests/test_tanh_port_runtime_20261019.py')
    meta = {'seed':7401, 'perturbation_seed':7402, 'train_events':12, 'observed_points':61,
            'future_steps':50, 'q_supervision':False, 'heldout_generated':False,
            'initial_sha256':hashlib.sha256(initial.tobytes()).hexdigest(),
            'observations_sha256':hashlib.sha256(observed.cpu().numpy().tobytes()).hexdigest(),
            'scales_sha256':hashlib.sha256(scales.cpu().numpy().tobytes()).hexdigest(),
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'torch':torch.__version__, 'scope':'disposable engineering only; discard all updated parameters'}
    return start, z, observed, scales, meta


def modes(start, observed):
    from tanh_port_runtime_20261019 import select_runtime
    from closure_causal_history_residual_20261019 import CausalHistoryResidual, history_features
    features = history_features(observed[:, :11], start.mass.detach())
    scales = features.square().mean((0,1)).sqrt().clamp_min(1e-6).detach()
    return [(name, CausalHistoryResidual(select_runtime(start, fast=fast), scales) if history else select_runtime(start, fast=fast))
            for name, fast, history in [('reference_gru',False,False),('fast_gru',True,False),
                                       ('reference_history',False,True),('fast_history',True,True)]]


def update(model, opt, observed, scales, deadline=None):
    import torch
    from closure_causal_history_residual_20261019 import loss
    opt.zero_grad(set_to_none=True)
    value = loss(model, observed, scales, deadline)
    if not bool(torch.isfinite(value)):
        raise FloatingPointError('nonfinite loss')
    value.backward()
    if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
        raise FloatingPointError('nonfinite gradient')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
    opt.step()
    return value.detach()


def benchmark(args):
    import numpy as np
    import torch
    from closure_training_runtime import resolve_device, synchronize
    from closure_causal_history_residual_20261019 import make_optimizer
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    if device.type != 'cuda':
        raise ValueError('registered timing requires GPU')
    deadline = time.perf_counter()+args.max_seconds
    start,z,observed,scales,meta = engineering_inputs(device)
    output = Path(args.output)
    (output/'identity.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    batches = [torch.tensor(np.random.default_rng(7401+i).permutation(12),device=device) for i in range(4)]
    results = {}
    for name, model in modes(start,observed):
        opt = make_optimizer(model)
        times=[]
        for ix in batches:
            synchronize(device)
            before=time.perf_counter()
            update(model,opt,observed[ix],scales,deadline)
            synchronize(device)
            times.append(time.perf_counter()-before)
        results[name]={'warmup_seconds':times[0], 'update_seconds':times[1:], 'max_seconds':max(times[1:])}
        (output/'timings.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    accepted=True
    for kind in ('gru','history'):
        ratios=np.array(results['fast_'+kind]['update_seconds']) / np.array(results['reference_'+kind]['update_seconds'])
        results[kind+'_comparison']={'ratios':ratios.tolist(),'median_ratio':float(np.median(ratios)),
                                    'faster_count':int((ratios<1).sum())}
        accepted=accepted and np.median(ratios)<1 and (ratios<1).sum()>=2
    tau=max(results['fast_'+k]['max_seconds'] for k in ('gru','history'))
    results['engineering_timing_passed']=bool(accepted)
    results['C_seconds']=1.25*(512*tau+120)
    results['tau_max_seconds']=tau
    results['C_excludes']='child startup, imports, fixture generation and persistence; charged in wall_receipt'
    (output/'timings.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__ == '__main__':
    args=parser().parse_args()
    if args.child:
        benchmark(args)
    else:
        raise SystemExit(supervised(args,Path(__file__)))
