"""One preregistered TRAIN-only 1200->3000 continuation for Loop0078.

The command is intentionally inert until explicit completed evidence for all
three screen arms and a selected step1200 checkpoint are supplied.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import screen_structure_0078 as screen
import train_structured_rom as train
from kiki.models.private_pm import MechanicalROM
from train_joint_latent import NETWORK, read_development


def horizon_for_extension(step: int) -> float:
    if not 1201 <= step <= 3000: raise ValueError("extension step must be 1201..3000")
    return (1., 2., 5.)[(step - 301) % 3]


def _terminal(path: Path) -> tuple[str,dict[str,Any]]:
    try:
        data=json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error: raise ValueError(f'missing or invalid arm evidence: {path}') from error
    if data.get('exception_type') or data.get('type') or data.get('failure') is True: return 'failed',data
    if data.get("smoke") is False and data.get("steps") == 1200 and isinstance(data.get("monitors"),dict): return 'completed',data
    raise ValueError(f'nonterminal arm evidence: {path}')


def select_extension(arm: str, evidence_paths: dict[str,Path]) -> tuple[dict[str,Any],list[str]]:
    expected={'C-free','G-fixed','G-auto'}
    if set(evidence_paths)!=expected: raise ValueError('explicit evidence is required for exactly C-free, G-fixed, G-auto')
    terminal={name:_terminal(path) for name,path in evidence_paths.items()}
    for name,(_,data) in terminal.items():
        if data.get('arm') != name: raise ValueError(f'arm identity mismatch in {evidence_paths[name]}')
    completed={name:data for name,(state,data) in terminal.items() if state=='completed'}
    selected=screen.select_extensions({name:value["monitors"] for name,value in completed.items()})
    if arm not in selected: raise ValueError(f"{arm} is not one of the preregistered TRAIN-risk extensions: {selected}")
    return completed[arm],selected


def restore_optimizer(model: torch.nn.Module, checkpoint: Path) -> torch.optim.Adam:
    """Recreate exact named Adam groups, including frozen historical tensors."""
    meta=json.loads((checkpoint/'checkpoint.json').read_text(encoding='utf-8'))
    saved=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True)
    names=dict(model.named_parameters()); saved_groups=saved['param_groups']; named_groups=meta.get('optimizer_parameter_names')
    if not isinstance(named_groups,list) or len(named_groups)!=len(saved_groups): raise ValueError('optimizer group-name metadata mismatch')
    groups=[]
    for old,named in zip(saved_groups,named_groups,strict=True):
        if not all(name in names for name in named): raise ValueError('checkpoint optimizer parameter absent from reconstructed model')
        options={key:value for key,value in old.items() if key!='params'}
        groups.append({"params":[names[name] for name in named],**options})
    optimizer=torch.optim.Adam(groups); optimizer.load_state_dict(saved); return optimizer


def restore_rng(checkpoint: Path) -> tuple[np.random.Generator,np.random.Generator]:
    state=torch.load(checkpoint/'rng_state.pt',map_location='cpu',weights_only=True)
    fit=np.random.default_rng(); select=np.random.default_rng()
    fit.bit_generator.state=state['fit_numpy']; select.bit_generator.state=state['selection_numpy']; torch.set_rng_state(state['torch'])
    return fit,select


def frozen_state(model: torch.nn.Module) -> dict[str,torch.Tensor]:
    parameters=dict(model.named_parameters())
    return {name:value.detach().clone() for name,value in model.state_dict().items()
            if name not in parameters or not parameters[name].requires_grad}


def assert_frozen(model: torch.nn.Module, before: dict[str,torch.Tensor]) -> None:
    now=model.state_dict()
    if set(now) < set(before) or any(not torch.equal(now[name],value) for name,value in before.items()):
        raise AssertionError('support/frozen parameter changed during extension')


def _failure(output: Path, model: torch.nn.Module, arm: str, step: int, error: Exception,
             hashes: dict[str,str], traces: list[dict[str,Any]]) -> None:
    torch.save(model.state_dict(),output/'failure_model.pt')
    (output/'failure_evidence.json').write_text(json.dumps({"arm":arm,"step":step,"type":type(error).__name__,
        "reason":str(error),"hashes_before":hashes,"hashes_after":train._source_hashes(Path(hashes['_source_path'])),
        "traces":traces,"test_read":False},indent=2),encoding='utf-8')


def _hashes(source: Path, checkpoint: Path, evidence_paths: dict[str,Path]) -> dict[str,str]:
    out={**train._source_hashes(source), 'extension_code':train.digest(Path(__file__))}
    for name,path in evidence_paths.items(): out[f'evidence_{name}']=train.digest(path)
    for name in ('model.pt','checkpoint.json','optimizer.pt','rng_state.pt','normalization.npz'):
        out[f'checkpoint_{name}']=train.digest(checkpoint/name)
    if (checkpoint/'structured_numeric_export.json').exists():
        out['checkpoint_numeric_export']=train.digest(checkpoint/'structured_numeric_export.json')
    return out


def _match_training_provenance(evidence: dict[str,Any], checkpoint: Path, hashes: dict[str,str]) -> None:
    final=Path(evidence.get('final_checkpoint','')).resolve()
    if checkpoint.resolve()!=final: raise ValueError('extension checkpoint is not the exact selected evidence final_checkpoint')
    old=evidence.get('hashes',evidence.get('source_hashes_before'))
    if not isinstance(old,dict): raise ValueError('selected training evidence lacks source hashes')
    common=set(old)&set(hashes)
    if not common or any(old[key]!=hashes[key] for key in common): raise ValueError('selected checkpoint/source hash mismatch')


def _finite_monitor(monitor: dict[str,Any]) -> None:
    rows=monitor.get('horizons',{})
    if set(rows) != {'1','2','5'} or not math.isfinite(float(monitor.get('average_risk',float('nan')))) or any(not math.isfinite(float(row.get('risk',float('nan')))) for row in rows.values()):
        raise FloatingPointError('nonfinite 3000 TRAIN monitor')


def run(arm: str, checkpoint: Path, output: Path, evidence_paths: dict[str,Path]) -> dict[str,Any]:
    if output.exists(): raise FileExistsError(output)
    selected_evidence,selected=select_extension(arm,evidence_paths)
    meta=json.loads((checkpoint/'checkpoint.json').read_text(encoding='utf-8'))
    if meta.get('arm')!=arm or meta.get('step')!=1200: raise ValueError('checkpoint must be the selected arm step1200 directory')
    output.mkdir(parents=True); started=time.perf_counter(); git=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=train.ROOT,text=True).strip(); torch.set_num_threads(1); source=Path(meta['source'])
    hashes={**_hashes(source,checkpoint,evidence_paths),'_source_path':str(source)}; _match_training_provenance(selected_evidence,checkpoint,hashes); step=1200; traces=[]
    base=MechanicalROM.load(train.BASE); network=dict(np.load(NETWORK,allow_pickle=False)); network['observed_buses']=np.unique(np.r_[network['load_buses'],network['generator_buses']]); network['load_positions']=np.searchsorted(network['observed_buses'],network['load_buses'])
    model=train.load_model(checkpoint,base); model.train(); optimizer=restore_optimizer(model,checkpoint); fit_rng,selection_rng=restore_rng(checkpoint); frozen=frozen_state(model)
    try:
        train._invariants(model); train._checkpoint(output,'step1200_resume',model,optimizer,fit_rng,selection_rng,source,arm,step=1200)
        events,excluded,identities=read_development(base,network,full_train=True); train_events=events['train']
        if len(train_events)!=346: raise AssertionError('unexpected TRAIN cardinality')
        for step in range(1201,3001):
            optimizer.zero_grad(set_to_none=True); samples,histories=train.sample_batch(train_events,fit_rng,train.BATCH)
            if isinstance(model,train.GROM): model.begin_graph_cache()
            try: loss,terms=train.objective(model,samples,histories,horizon_for_extension(step),complexity_weight=0.)
            finally:
                if isinstance(model,train.GROM): model.end_graph_cache()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite extension loss')
            loss.backward(); grads=[]
            for name,p in model.named_parameters():
                if p.requires_grad:
                    if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError(f'extension gradient: {name}')
                    grads.append(p.grad.reshape(-1))
            raw_norm=float(torch.linalg.vector_norm(torch.cat(grads))); clipped=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],100.)); optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad): raise FloatingPointError('nonfinite extension parameter')
            assert_frozen(model,frozen); checks=train._invariants(model)
            traces.append({"step":step,"horizon_s":horizon_for_extension(step),"loss":float(loss.detach()),"gradient_norm_before_cap":raw_norm,"gradient_norm_clip_return":clipped,"terms":{k:float(v.detach()) for k,v in terms.items()},"checks":checks,"wall_seconds":time.perf_counter()-started})
            if step%25==0: print(json.dumps({"step":step,"loss":traces[-1]['loss'],"seconds":traces[-1]['wall_seconds']}),flush=True)
        print(json.dumps({"event":"train_monitor_start","step":3000}),flush=True); monitor=train._train_monitor(model,train_events); (output/'monitor_step3000.json').write_text(json.dumps(monitor,indent=2),encoding='utf-8'); _finite_monitor(monitor); print(json.dumps({"event":"train_monitor_finish","step":3000}),flush=True)
        final=train._checkpoint(output,'step3000',model,optimizer,fit_rng,selection_rng,source,arm,step=3000)
        after={**_hashes(source,checkpoint,evidence_paths),'_source_path':str(source)}
        if after!=hashes: raise AssertionError('source artifact changed during extension')
        evidence={"arm":arm,"checkpoint":str(checkpoint),"selected_extensions":selected,"selection_evidence":{name:str(path) for name,path in evidence_paths.items()},"selected_arm_evidence":selected_evidence,"steps_from":1200,"steps_to":3000,"complexity_weight":0.,"test_read":False,"validation_used":False,"git_commit":git,"hashes_before":hashes,"hashes_after":after,"excluded":excluded,"identities":identities,"traces":traces,"monitor3000":monitor,"final_checkpoint":str(final),"seconds":time.perf_counter()-started}
        (output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False),encoding='utf-8'); return evidence
    except Exception as error:
        _failure(output,model,arm,step,error,hashes,traces); raise


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--arm',choices=('C-free','G-fixed','G-auto'),required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--train-evidence-c-free',type=Path,required=True); p.add_argument('--train-evidence-g-fixed',type=Path,required=True); p.add_argument('--train-evidence-g-auto',type=Path,required=True)
    a=p.parse_args(); run(a.arm,a.checkpoint,a.output,{'C-free':a.train_evidence_c_free,'G-fixed':a.train_evidence_g_fixed,'G-auto':a.train_evidence_g_auto})


if __name__=='__main__': main()
