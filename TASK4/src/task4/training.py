"""Runnable initial joint trainer for P/A/U. Root agent plans larger studies.
This implementation is an engineering starting point, not a validated TASK4 result.
"""
import json, os, random, time
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT,project_path,write_json,read_json
from .data import load_events,train_scales,Physics
from .dynamics import PowerGraph,UnstructuredGraph
from .encoder import HistoryEncoder,PredictiveHead
from .objectives import AnchorTargets,ShootingBank,batch_objective,one_event_loss
from .synthetic import fixture
from .workflow import Workflow
from .symbolic import PolynomialFunction

QUADRATIC_FUNCTION_MODES=('off','energy_only','all_poly')

def quadratic_function_mode(value):
    if type(value) is not str or value not in QUADRATIC_FUNCTION_MODES:
        raise ValueError('quadratic_functions must be off, energy_only, or all_poly')
    return value

def configure_quadratic_functions(model,mode):
    """Select an existing executor before tracing; never change formula state."""
    mode=quadratic_function_mode(mode)
    functions=model.function_modules()
    selected=[] if mode=='off' else ['node_energy','edge_energy'] if mode=='energy_only' else list(functions)
    if mode!='off' and type(model) is not PowerGraph:
        raise ValueError('Quadratic function execution supports only the exact PowerGraph')
    # Validate all requested modules before enabling any of them. This uses
    # the existing layout qualification, not a new polynomial representation.
    for name in selected:
        fn=functions[name]
        if type(fn) is not PolynomialFunction:
            raise ValueError(f'Quadratic function execution requires explicit PolynomialFunction: {name}')
        if fn._quadratic_rejection is not None:
            raise ValueError(f'{name}: {fn._quadratic_rejection}')
    for name,fn in functions.items():
        if type(fn) is PolynomialFunction:
            fn.set_quadratic_execution(name in selected)
    return {'quadratic_functions':mode,'enabled_function_modules':selected,
            'scientific_definition_changed_by_executor':False,'algebraically_equivalent_not_bitwise_claim':True,
            'source_preflight_role':'not performed' if mode=='off' else 'CPU executor eligibility only; not model selection'}

def preflight_quadratic_source(cfg,source):
    """Reject unsupported opt-ins before event I/O, preserving caller RNG.

    A CPU source reconstruction is discarded. The normal warm/resume loading
    order below is retained, so its random allocations and optimizer restore
    keep their existing sequence. Generic loading itself remains unchanged.
    """
    mode=quadratic_function_mode(cfg.get('quadratic_functions','off'))
    if mode=='off':return
    if cfg['arm']=='U':raise ValueError('Quadratic function execution is unavailable for U control')
    if source is None:raise ValueError('Quadratic function execution requires a polynomial warm_start/resume source; fresh functions are neural')
    py_state,np_state=random.getstate(),np.random.get_state()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            old_model,_,_,old=load_checkpoint(project_path(source),'cpu')
            quadratic_function_mode(old['config'].get('quadratic_functions','off'))
            if bool(old['synthetic'])!=bool(cfg.get('synthetic',False)):
                raise ValueError('Quadratic executor source synthetic/real mode mismatch')
            for key in ('hidden','latent_per_machine'):
                if old['config'][key]!=cfg[key]:raise ValueError('Quadratic executor source architecture mismatch')
            configure_quadratic_functions(old_model,mode)
    finally:
        random.setstate(py_state);np.random.set_state(np_state)

def optimizer_groups(model,encoder,head,cfg):
    """Registered blocks share the same Adam rule; scales default to one."""
    blocks={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        key=('coupling' if name=='log_coupling' else 'energy' if 'energy' in name else
             'exchange' if 'exchange' in name else 'dissipation' if 'dissipation' in name else
             'port' if name.startswith('port.') else 'vector_control')
        blocks.setdefault(key,[]).append(p)
    blocks['encoder']=[p for p in encoder.parameters() if p.requires_grad]
    blocks['head']=[p for p in head.parameters() if p.requires_grad]
    scales=cfg.get('block_lr_scales',{})
    return [{'params':params,'lr':cfg['lr']*float(scales.get(name,1.)),'name':name}
            for name,params in blocks.items() if params]

def capture_updates(optim):
    before=[];stats={}
    for group in optim.param_groups:
        name=group.get('name','shooting' if len(group['params'])==1 else 'joint_flat')
        row=stats.setdefault(name,{'gradient_squared':0.,'parameter_squared':0.,'update_squared':0.,'lr':group['lr']})
        for param in group['params']:
            if param.grad is None:continue
            row['gradient_squared']+=float(param.grad.detach().square().sum())
            row['parameter_squared']+=float(param.detach().square().sum())
            before.append((name,param,param.detach().clone()))
    return before,stats

def summarize_updates(before,stats):
    for name,p,old in before:stats[name]['update_squared']+=float((p.detach()-old).square().sum())
    return {name:{'lr':v['lr'],'gradient_l2':v['gradient_squared']**.5,
                  'parameter_l2':v['parameter_squared']**.5,'update_l2':v['update_squared']**.5,
                  'relative_update':(v['update_squared']/max(v['parameter_squared'],1e-24))**.5}
            for name,v in stats.items()}

def check_authorized(synthetic):
    if synthetic:return
    rid=os.environ.get('TASK4_MANAGED_RUN_ID');s=Workflow().load()
    if not rid or rid not in s['active_jobs'] or s['status']!='RUNNING' or s.get('pending_direction'):
        raise PermissionError('Real training must run under scripts/project.py run after explicit user start')

def checkpoint(path,model,encoder,head,bank,optim,config,physics,step,rng,records,boundary_complete=True):
    path=project_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    obj={'format':1,'kind':'T4-N' if config['arm']!='U' else 'U-control','config':config,'physics':physics.as_dict(),
         'scales':model.scales.detach().cpu().tolist(),'step':step,'model':model.state_dict(),'encoder':encoder.state_dict(),
         'head':head.state_dict(),'shooting':bank.state_dict(),'shooting_keys':bank.keys,'optimizer':optim.state_dict(),
         'event_ids':config.get('_event_ids',[]),'rng_numpy':rng.bit_generator.state,'rng_torch':torch.get_rng_state(),'records':records,'synthetic':config['synthetic']}
    obj['exact_resume_eligible']=boundary_complete
    symbolic={name:fn.export() for name,fn in model.function_modules().items() if isinstance(fn,PolynomialFunction)}
    if symbolic:obj['symbolic_functions']=symbolic;obj['kind']='T4-S' if len(symbolic)==7 else 'T4-partially-compressed'
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);tmp.replace(path)

def load_checkpoint(path,device='cpu'):
    obj=torch.load(path,map_location=device,weights_only=True)
    cfg=obj['config'];physics=Physics.from_dict(obj['physics'])
    cls=UnstructuredGraph if cfg['arm']=='U' else PowerGraph
    extra={'u_hidden':cfg.get('u_hidden',cfg['hidden'])} if cfg['arm']=='U' else {}
    m=cls(physics,obj['scales'],cfg['latent_per_machine'],cfg['hidden'],**extra).to(device)
    m.strength=cfg.get('model_strength',.01)
    for name,spec in obj.get('symbolic_functions',{}).items():
        setattr(m,name,PolynomialFunction(spec['powers'],spec['coefficients']).to(device))
    e=HistoryEncoder(m,cfg['hidden']).to(device);h=PredictiveHead(m,cfg['anchor_horizons'],cfg['hidden']).to(device)
    m.load_state_dict(obj['model']);e.load_state_dict(obj['encoder']);h.load_state_dict(obj['head'])
    return m,e,h,obj

def fit(config,project_config,output,resume=None,warm_start=None):
    
    if resume and warm_start:raise ValueError('Choose resume OR warm_start')
    cfg=dict(config);check_authorized(cfg.get('synthetic',False))
    cfg['quadratic_functions']=quadratic_function_mode(cfg.get('quadratic_functions','off'))
    preflight_quadratic_source(cfg,resume or warm_start)
    output=project_path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('New runs require a new empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(1)
    device=cfg['device']
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if cfg['synthetic']:events,physics=fixture();facts={'synthetic':True,'event_count':len(events),'test_payload_read':False}
    else:events,physics,facts=load_events(project_config,'train')
    write_json(output/'data_facts.json',facts)
    cfg['_event_ids']=[e.id for e in events]
    # Event-balanced TRAIN scale in the actual te-pm_star target units.
    cfg['torque_scale']=max(1e-5,float(np.sqrt(np.mean([np.mean(e.torque[10:]**2) for e in events if e.torque is not None]))))
    scales=train_scales(events,physics);cls=UnstructuredGraph if cfg['arm']=='U' else PowerGraph
    extra={'u_hidden':cfg.get('u_hidden',cfg['hidden'])} if cfg['arm']=='U' else {}
    model=cls(physics,scales,cfg['latent_per_machine'],cfg['hidden'],**extra).to(device)
    encoder=HistoryEncoder(model,cfg['hidden']).to(device);head=PredictiveHead(model,cfg['anchor_horizons'],cfg['hidden']).to(device)
    bank=ShootingBank().to(device);rng=np.random.default_rng(cfg['seed']);start=0;records=[]
    if cfg.get('initialization_checkpoint') and not (resume or warm_start):
        _,init_encoder,init_head,initial=load_checkpoint(project_path(cfg['initialization_checkpoint']),device)
        if initial['config']['latent_per_machine']!=cfg['latent_per_machine'] or initial['config']['hidden']!=cfg['hidden']:
            raise ValueError('Initialization-only source architecture mismatch')
        if initial['event_ids']!=cfg['_event_ids'] or not np.allclose(initial['scales'],scales):
            raise ValueError('Initialization-only source TRAIN/scales mismatch')
        for key,value in physics.as_dict().items():
            if not np.allclose(np.asarray(value),np.asarray(initial['physics'][key])):raise ValueError('Initialization-only physics mismatch')
        encoder.load_state_dict(init_encoder.state_dict());head.load_state_dict(init_head.state_dict())
    if warm_start:
        old_model,old_encoder,old_head,old=load_checkpoint(project_path(warm_start),device)
        if isinstance(old_model,UnstructuredGraph)!=isinstance(model,UnstructuredGraph):raise ValueError('Cannot silently remap U and power-structured models')
        if old['config']['hidden']!=cfg['hidden'] or old['config']['latent_per_machine']!=cfg['latent_per_machine']:raise ValueError('Warm-start architecture mismatch')
        if not np.allclose(old['scales'],scales):raise ValueError('Warm-start source scales differ; reconcile before reuse')
        for key,value in physics.as_dict().items():
            if not np.allclose(np.asarray(value),np.asarray(old['physics'][key])):raise ValueError(f'Warm-start physics mismatch: {key}')
        for name,spec in old.get('symbolic_functions',{}).items():
            setattr(model,name,PolynomialFunction(spec['powers'],spec['coefficients']).to(device))
        model.load_state_dict(old_model.state_dict());encoder.load_state_dict(old_encoder.state_dict());head.load_state_dict(old_head.state_dict())
    if cfg.get('autonomous_only'):head.requires_grad_(False)
    params=[p for module in [model,encoder,head] for p in module.parameters() if p.requires_grad]
    if not resume:cfg['_optimizer_blocks']=True
    optim=torch.optim.Adam(optimizer_groups(model,encoder,head,cfg) if cfg.get('_optimizer_blocks') else params,lr=cfg['lr'])
    if resume:
        model,encoder,head,obj=load_checkpoint(project_path(resume),device)
        if not obj.get('exact_resume_eligible',True):raise ValueError('Interrupted-step diagnostic is not an exact-resume checkpoint; use last.pt')
        if obj['config'].get('_optimizer_blocks'):cfg['_optimizer_blocks']=True
        runtime_keys={'device','checkpoint_steps','trace_field','reuse_origin_terms','quadratic_functions'}
        for key in set(obj['config'])|set(cfg):
            if key not in runtime_keys and cfg.get(key)!=obj['config'].get(key):raise ValueError(f'Exact resume config changed: {key}; use a new protocol and --warm-start instead')
        if obj['config']['arm']!=cfg['arm'] or obj['config']['latent_per_machine']!=cfg['latent_per_machine']:raise ValueError('Resume architecture changed')
        if [e.id for e in events]!=obj.get('event_ids',[e.id for e in events]):raise ValueError('Resume dataset changed')
        if cfg.get('autonomous_only'):head.requires_grad_(False)
        params=[p for module in [model,encoder,head] for p in module.parameters() if p.requires_grad]
        optim=torch.optim.Adam(optimizer_groups(model,encoder,head,cfg) if cfg.get('_optimizer_blocks') else params,lr=cfg['lr'])
        for k in obj['shooting_keys']:
            bank.states[k]=torch.nn.Parameter(obj['shooting']['states.'+k].to(device));bank.keys.append(k)
            optim.add_param_group({'params':[bank.states[k]],'lr':cfg['lr']*cfg.get('block_lr_scales',{}).get('shooting',1.),'name':'shooting'})
        optim.load_state_dict(obj['optimizer']);rng.bit_generator.state=obj['rng_numpy'];torch.set_rng_state(obj['rng_torch'].cpu())
        start=obj['step'];records=obj.get('records',[])
    optim.task4_shooting_lr=cfg['lr']*cfg.get('block_lr_scales',{}).get('shooting',1.)
    target=AnchorTargets(physics,cfg['anchor_horizons']);deadline=float(os.environ.get('TASK4_SAVE_DEADLINE_UNIX','inf'))
    model.strength=cfg.get('model_strength',.01)
    model.checkpoint_steps=cfg.get('checkpoint_steps',10)
    model.reuse_origin_terms=cfg.get('reuse_origin_terms',False)
    model.save_deadline=deadline-5
    quadratic_execution=configure_quadratic_functions(model,cfg['quadratic_functions'])
    if cfg.get('trace_field',False):
        from .accelerate import trace_field
        trace_field(model)
    teacher=None
    if cfg.get('teacher_checkpoint'):
        teacher,_,_,teacher_obj=load_checkpoint(project_path(cfg['teacher_checkpoint']),device)
        if teacher_obj['synthetic']!=cfg['synthetic']:raise ValueError('Teacher source mode mismatch')
        for param in teacher.parameters():param.requires_grad_(False)
    phase_final=1-float(cfg['final_autonomous_fraction']);t0=time.monotonic();last=start
    monitor_ids=list(range(len(events)))
    if cfg.get('monitor_count'):
        # Fixed TRAIN panel, independent of losses and of validation data.
        bygroup={}
        for i,e in enumerate(events):bygroup.setdefault((e.group,e.label),i)
        monitor_ids=list(bygroup.values())[:cfg['monitor_count']]
    write_json(output/'run_config.json',cfg)
    checkpoint(output/'last.pt',model,encoder,head,bank,optim,cfg,physics,last,rng,records)
    failure=None;step_cost=0.;stop_reason='steps_completed';seen=set()
    # Reporting only: optimizer membership does not imply head evaluation or
    # a gradient-backed optimizer step (notably during compression adaptation).
    head_parameter_ids={id(p) for p in head.parameters()}
    head_in_optimizer=any(id(p) in head_parameter_ids for group in optim.param_groups for p in group['params'])
    head_used=False;head_optimized=False
    try:
      with (output/'steps.jsonl').open('w',encoding='utf-8') as log:
        for step in range(start+1,cfg['steps']+1):
            add=step-start
            if time.time()+max(5.,step_cost*1.5)>=deadline:
                stop_reason='save_deadline';break
            step_started=time.monotonic()
            progress=step/cfg['steps'];n=len(events)
            # Deterministic coverage plus stratified draws; no event is permanently excluded.
            ids=[((step-1)*max(1,cfg['batch_size']//2)+j)%n for j in range(max(1,cfg['batch_size']//2))]
            for label in ['unstable','stable']:
                pool=[j for j,e in enumerate(events) if e.label==label]
                if pool and len(ids)<cfg['batch_size']:ids.append(int(rng.choice(pool)))
            while len(ids)<cfg['batch_size']:ids.append(int(rng.integers(n)))
            horizon=cfg['horizons'][min(len(cfg['horizons'])-1,int(progress*len(cfg['horizons'])))]
            if progress>=phase_final:horizon=max(cfg['horizons'])
            segment=None
            if progress<min(phase_final,cfg['shooting_fraction']) and not cfg['synthetic']:
                j=min(len(cfg['segment_schedule'])-1,int(progress/max(cfg['shooting_fraction'],1e-9)*len(cfg['segment_schedule'])))
                segment=cfg['segment_schedule'][j]
            anchor_factor=max(0.,1-progress/max(phase_final,1e-9))
            optim.zero_grad(set_to_none=True)
            loss,terms=batch_objective(model,encoder,head,[events[j] for j in ids],horizon,cfg,target,rng,bank,optim,segment,anchor_factor,teacher)
            # Mirror the head.forward branch in batch_objective, including its
            # scheduled factor; a user metadata flag is not execution evidence.
            head_used=head_used or (not cfg.get('autonomous_only') and cfg['arm']!='A' and cfg['anchor_weight']>0 and anchor_factor>0)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            before,block_stats=capture_updates(optim)
            active=[p for g in optim.param_groups for p in g['params']]
            norm=torch.nn.utils.clip_grad_norm_(active,cfg['grad_clip'],error_if_nonfinite=True)
            optim.step();last=step
            head_optimized=head_optimized or any(id(param) in head_parameter_ids for _,param,_ in before)
            blocks=summarize_updates(before,block_stats)
            del before
            seen.update(events[j].id for j in ids)
            step_cost=time.monotonic()-step_started
            row={'step':step,'additional_step':add,'ids':[events[j].id for j in ids],'horizon':horizon,'segment_seconds':segment,
                 'loss':float(loss.detach()),'gradient_norm':float(norm),'elapsed_seconds':time.monotonic()-t0,'step_seconds':step_cost,'blocks':blocks,
                 **{k:float(v.detach()) for k,v in terms.items()}}
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if add%cfg.get('save_every',10)==0:
                checkpoint(output/'last.pt',model,encoder,head,bank,optim,cfg,physics,last,rng,records)
            if step in cfg['milestones'] or step==cfg['steps']:
                checkpoint(output/f'step_{step:06d}.pt',model,encoder,head,bank,optim,cfg,physics,step,rng,records)
                # Pure single-start monitor; do not select model by developer/TEST feedback.
                vals=[]
                for i in monitor_ids:
                    if time.time()>=deadline:break
                    val,_=one_event_loss(model,encoder,events[i],min(horizon,5.),cfg['integration_dt'],False)
                    vals.append(float(val.detach()))
                rec={'step':step,'single_start_mean':float(np.mean(vals)) if vals else None,'monitored':len(vals),
                     'required':len(monitor_ids),'complete':len(vals)==len(monitor_ids),'scope':'fixed_TRAIN_monitor_panel','numerically_verified':False}
                records.append(rec);write_json(output/'monitor.json',records)
            if add%max(1,min(100,cfg['steps']))==0:print(f'step={step} loss={row["loss"]:.6g}',flush=True)
    except TimeoutError:
        stop_reason='save_deadline_during_step'
    except BaseException as exc:
        failure=exc;stop_reason=type(exc).__name__+': '+str(exc)
    finally:
        # Preserve a previous complete-update RNG/bank boundary for exact resume.
        complete=failure is None and stop_reason!='save_deadline_during_step'
        checkpoint(output/('last.pt' if complete else 'interrupted.pt'),model,encoder,head,bank,optim,cfg,physics,last,rng,records,boundary_complete=complete)
    report={'steps_completed':last-start,'requested_total_steps':cfg['steps'],'synthetic':cfg['synthetic'],
            'final_single_start_phase_completed':last==cfg['steps'],
            'stage':('U vector control; no power-structure or symbolic claim' if cfg['arm']=='U' else 'T4-S complete autonomous truth fine-tuning' if cfg.get('autonomous_only') and all(isinstance(fn,PolynomialFunction) for fn in model.function_modules().values()) else 'T4-N joint learning only; not symbolic completion'),'test_payload_read':False,
            'future_input_at_deployment':False,'acceptance':'requires independent evaluation and root decision'}
    report.update(stop_reason=stop_reason,unique_events_optimized_this_run=len(seen),required_train_events=len(events),wall_seconds=time.monotonic()-t0)
    report['declared_final_autonomous_horizon_seconds']=max(cfg['horizons'])
    report['autonomous_only']=bool(cfg.get('autonomous_only'))
    report['prediction_head_in_optimizer']=head_in_optimizer
    report['prediction_head_used']=bool(head_used)
    report['prediction_head_deployed']=False
    report['prediction_head_used_or_deployed']=bool(head_used)
    report['prediction_head_optimized']=bool(head_optimized)
    report['prediction_head_usage_scope']='this fit invocation: completed objective calls and completed optimizer steps'
    report['prediction_head_optimized_semantics']='at least one completed optimizer step with a head parameter gradient; not a nonzero-update claim'
    # Keep the historical key readable without equating membership with use.
    report['prediction_head_used_or_optimized']=head_in_optimizer
    report['prediction_head_used_or_optimized_semantics']='legacy_optimizer_membership_label'
    report['function_execution']=quadratic_execution
    write_json(output/'training_summary.json',report)
    if failure is not None:raise failure
    return report
