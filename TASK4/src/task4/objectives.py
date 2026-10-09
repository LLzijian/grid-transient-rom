"""Losses use real available prefixes only. No target availability mask enters GRU."""
import numpy as np
import torch
from .integrators import rollout

def physical_error(pred,truth,mass,scales):
    err=pred-truth
    de=err[...,0];we=err[...,1]
    dc=de-(de*mass).sum(-1,keepdim=True)/mass.sum()
    common=(we*mass).sum(-1)/mass.sum()
    wc=we-common[...,None]
    return .5*((dc/scales[0]).square().mean(-1)+(wc/scales[1]).square().mean(-1))+(common/scales[1]).square()+((err[...,2])/scales[2]).square().mean(-1)

def tail_risk(values):
    k=max(1,int(np.ceil(.1*len(values))))
    return .8*values.mean()+.2*values.topk(k).values.mean()

def m3_numpy(physics,y0,steps,dt=.01):
    """Frozen physical reference, canonical delta/omega/p coordinates."""
    def f(y):
        eta=(y[:,0]+physics.delta_star)@physics.incidence
        electrical=((np.sin(eta)-np.sin(physics.delta_star@physics.incidence))*physics.coupling)@physics.incidence.T
        return np.stack([y[:,1],(y[:,2]-electrical-physics.damping*y[:,1])/physics.mass,
                         -physics.rate*y[:,2]-physics.gain*y[:,1]],-1)
    out=[np.array(y0,copy=True)];y=out[0]
    for _ in range(steps):
        # Two .005 substeps for a .01 sample by default.
        for __ in range(2):
            h=dt/2;a=f(y);b=f(y+h*a/2);c=f(y+h*b/2);d=f(y+h*c)
            y=y+h*(a+2*b+2*c+d)/6
        out.append(y.copy())
    return np.stack(out)

class AnchorTargets:
    def __init__(self,physics,horizons,dt=.01,max_cache=4096):
        self.physics=physics;self.horizons=list(horizons);self.dt=dt;self.cache={};self.max_cache=max_cache
    def get(self,event,index):
        key=(event.id,index)
        if key not in self.cache:
            offsets=[round(t/self.dt) for t in self.horizons]
            available=len(event.y)-index-1
            count=min(max(offsets),available)
            baseline=m3_numpy(self.physics,event.y[index],count,self.dt)
            target=np.zeros((self.physics.n,len(offsets),3));mask=np.zeros(len(offsets),dtype=bool)
            for j,k in enumerate(offsets):
                if k<=available:target[:,j]=event.y[index+k]-baseline[k];mask[j]=True
            if len(self.cache)>=self.max_cache:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=(target,mask)
        return self.cache[key]

class ShootingBank(torch.nn.Module):
    """TRAIN-only latent segment starts; never part of a deployment export."""
    def __init__(self):super().__init__();self.states=torch.nn.ParameterDict();self.keys=[]
    def get(self,e,index,encoder,optim,device):
        key=f'{e.id.replace(".","_").replace("/","_")}__{index}'
        if key not in self.states:
            h=torch.as_tensor(e.y[index-10:index+1],dtype=torch.float64,device=device)[None]
            value=encoder(h).detach()
            self.states[key]=torch.nn.Parameter(value);self.keys.append(key)
            optim.add_param_group({'params':[self.states[key]],
                                   'lr':getattr(optim,'task4_shooting_lr',optim.defaults['lr']),
                                   'name':'shooting'})
        return self.states[key]

def one_event_loss(model,encoder,event,horizon,dt,training=True,segment_seconds=None,bank=None,optim=None,continuity_weight=1.):
    dev=model.mass.device;steps=min(round(horizon/.01),len(event.y)-11)
    if steps<1:raise ValueError('Event has no forecast after the 100 ms history')
    hist=torch.as_tensor(event.y[:11],dtype=torch.float64,device=dev)[None]
    z0=model.initial(hist,encoder)
    truth=torch.as_tensor(event.y[10:11+steps],dtype=torch.float64,device=dev)[None]
    if segment_seconds is None or round(segment_seconds/.01)>=steps:
        pred=rollout(model,z0,steps,max_step=dt,training=training)
        return physical_error(model.physical(pred[:,1:]),truth[:,1:],model.mass,model.scales).mean(),z0.new_zeros(())
    width=max(1,round(segment_seconds/.01));total=z0.new_zeros(());cont=z0.new_zeros(());count=0
    start=0;current=z0
    while start<steps:
        stop=min(steps,start+width)
        pred=rollout(model,current,stop-start,max_step=dt,training=training)
        total=total+physical_error(model.physical(pred[:,1:]),truth[:,start+1:stop+1],model.mass,model.scales).sum()
        count+=stop-start
        if stop<steps:
            q_next=bank.get(event,10+stop,encoder,optim,dev)
            next_z=model.from_physical(truth[:,stop],q_next)
            # Full-state continuity, with fixed physical and latent scales.
            err=physical_error(model.physical(pred[:,-1]),truth[:,stop],model.mass,model.scales).mean()
            eq=(model.split(pred[:,-1])[-1]-q_next).square().mean()
            cont=cont+err+eq
            current=next_z
        start=stop
    return total/count,continuity_weight*cont/max(1,int(np.ceil(steps/width))-1)

def batch_objective(model,encoder,head,events,horizon,config,anchor_targets,rng,bank,optim,segment_seconds=None,anchor_factor=1.,teacher=None):
    autonomous_only=config.get('autonomous_only',False)
    if autonomous_only:
        if segment_seconds is not None:
            raise ValueError('autonomous_only requires segment_seconds=None')
        forbidden_weights=['anchor_weight','latent_weight','torque_weight','teacher_field_weight','teacher_gradient_weight']
        active=[name for name in forbidden_weights if config.get(name,0.)!=0.]
        if active:
            raise ValueError('autonomous_only requires zero auxiliary weights: '+', '.join(active))
        if teacher is not None:
            raise ValueError('autonomous_only requires teacher=None')
        if not events:
            raise ValueError('autonomous_only requires at least one eligible event')
    losses=[];continuity=[];aux=[];lat=[];tor=[];fd=[];gd=[];device=model.mass.device
    batched=None
    lengths=[min(round(horizon/.01),len(e.y)-11) for e in events]
    if autonomous_only and min(lengths)<1:
        raise ValueError('autonomous_only requires a real forecast after the 100 ms history')
    if segment_seconds is None or round(segment_seconds/.01)>=max(lengths):
        # Autonomous predictions may exceed a short true prefix; only existing
        # samples contribute to its loss. No padded/extrapolated truth is made.
        batched={}
        for count in sorted(set(lengths)):
            group=[i for i,n in enumerate(lengths) if n==count]
            histories=torch.as_tensor(np.stack([events[i].y[:11] for i in group]),dtype=torch.float64,device=device)
            pred=model.physical(rollout(model,model.initial(histories,encoder),count,
                                       max_step=config['integration_dt'],training=True))
            for j,i in enumerate(group):batched[i]=pred[j]
    for ei,e in enumerate(events):
        if batched is None:
            loss,c=one_event_loss(model,encoder,e,horizon,config['integration_dt'],True,segment_seconds,bank,optim,config['continuity_weight'])
        else:
            count=lengths[ei]
            truth=torch.as_tensor(e.y[11:11+count],dtype=torch.float64,device=device)
            loss=physical_error(batched[ei][1:count+1],truth,model.mass,model.scales).mean()
            c=loss.new_zeros(())
        losses.append(loss);continuity.append(c)
        # Final D: each event was initialized exactly once from its first 11
        # samples above. Future truth contributes only to physical_error. Never
        # enter the random-history, head, teacher, torque or latent-reset paths.
        if autonomous_only:continue
        index=int(rng.integers(10,max(11,len(e.y)-1)))
        h=torch.as_tensor(e.y[index-10:index+1],dtype=torch.float64,device=device)[None]
        q=encoder(h);z=model.from_physical(h[:,-1],q)
        if teacher is not None:
            scale=torch.cat([model.scales[0].expand(model.n),model.mass*model.scales[1],model.scales[2].expand(model.n),z.new_ones(model.n*model.rank)])
            sf=model.field(z,create_graph=True);tf=teacher.field(z,create_graph=False)
            sg=model.energy_gradient(z);tg=teacher.energy_gradient(z).detach()
            fd.append(((sf-tf)/scale).square().mean())
            gd.append(((sg-tg)*scale).square().mean())
        if config['arm']!='A' and config['anchor_weight']>0 and anchor_factor>0:
            target,mask=anchor_targets.get(e,index)
            target=torch.as_tensor(target,dtype=torch.float64,device=device)[None]/model.scales
            m=torch.as_tensor(mask,device=device)
            per=((head(q)-target).square().mean(dim=(0,1,3)))
            if m.any():aux.append((per,m))
        if index+1<len(e.y):
            adjacent=torch.as_tensor(e.y[index-9:index+2],dtype=torch.float64,device=device)[None]
            zp=rollout(model,z,1,max_step=config['integration_dt'],training=True)[:,-1]
            lat.append((model.split(zp)[-1]-encoder(adjacent)).square().mean())
        if e.torque is not None:
            dz=model.field(z,create_graph=True)
            pred_torque=h[:,-1,:,2]-dz[:,model.n:2*model.n]
            target_torque=torch.as_tensor(e.torque[index],dtype=torch.float64,device=device)[None]
            tor.append(((pred_torque-target_torque)/config.get('torque_scale',model.scales[2])).square().mean())
    zero=losses[0].new_zeros(())
    mean=lambda seq:torch.stack(seq).mean() if seq else zero
    complexity=sum(p.square().mean() for p in model.parameters() if p.requires_grad)
    anchor_mean=zero
    if aux:
        av=torch.stack([a for a,m in aux]);am=torch.stack([m for a,m in aux]);counts=am.sum(0);legal=counts>0
        anchor_mean=((av*am).sum(0)[legal]/counts[legal]).mean()
    terms={'rollout':tail_risk(torch.stack(losses)),'continuity':mean(continuity),'anchor':anchor_mean,'latent':mean(lat),'torque':mean(tor),'regularization':complexity,
           'teacher_field':mean(fd),'teacher_gradient':mean(gd)}
    if autonomous_only:
        return terms['rollout']+config.get('complexity_weight',0.)*complexity,terms
    total=terms['rollout']+terms['continuity']+config['anchor_weight']*anchor_factor*terms['anchor']+config['latent_weight']*terms['latent']+config['torque_weight']*terms['torque']+config['complexity_weight']*complexity
    total=total+config.get('teacher_field_weight',0.)*terms['teacher_field']+config.get('teacher_gradient_weight',0.)*terms['teacher_gradient']
    return total,terms
