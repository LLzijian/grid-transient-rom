"""State-dependent force memory, fixed 40D coordinates and 47 raw parameters.

History is an observed-driven LTV problem, not exp(integral Q). No event reader
or optimizer is present. Legacy force/exchange qualifications are unchanged.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from force_graph_model_20261028 import ForceGraphROM, RIDGE

FORMAT="state_gated_memory_model_20261028"
GATE_MODES=("A","B","C")


class ControlDegenerate(RuntimeError):
    """Weighted control has no valid ordinary training derivative; preserve state."""


class StateGatedMemoryROM(ForceGraphROM):
    def __init__(self,source,gate_mode="A",*,history_dt=.0025,source_identity=None):
        if gate_mode not in GATE_MODES: raise ValueError("gate mode must be A/B/C")
        self._history_dt(history_dt)
        super().__init__(source,"M",source_identity=source_identity)
        self.gate_mode=gate_mode
        self.history_dt=float(history_dt)

    @staticmethod
    def _history_dt(dt):
        if isinstance(dt,bool) or dt not in (.005,.0025,.00125):
            raise ValueError("history dt must be legacy .005, primary .0025 or reference .00125")

    def _stamp(self):
        return super()._stamp()+(tuple(b._version for b in self.buffers()),tuple(p.requires_grad for p in self.parameters()),
                                 self.gate_mode,self.history_dt)

    def _new_context(self):
        return dict(kappa=self.kappa,lambda_=self.lambda_,coupling=self.reference_coupling45)

    def gate_values(self,delta):
        return (1+torch.cos(self._tensor(delta)@self.B))/2

    def effective_weights(self,delta):
        g=self.gate_values(delta);a=self.lambda_reference*self.a_bar
        if self.gate_mode=="A": return a.expand_as(g)
        if self.gate_mode=="C": return a*g
        # Scale both numerator and denominator equally. The scale remains in
        # autograd: this is algebraic normalization, not a frozen gate weight.
        scale=a.abs().max()
        if not torch.isfinite(a).all() or (a<0).any(): raise ControlDegenerate("invalid physical B weights")
        if scale.item()==0:
            if torch.is_grad_enabled() and (self.a_bar.requires_grad or self._tensor(delta).requires_grad):
                raise ControlDegenerate("B total a=0 has no unique ordinary training derivative")
            return torch.zeros_like(g)
        normalized=a/scale
        mean=(g*normalized).sum(-1,keepdim=True)/normalized.sum()
        weights=a*mean
        if not torch.isfinite(weights).all(): raise ControlDegenerate("B weighted ratio nonfinite")
        return weights

    def memory_matrix(self,delta):
        context=self._context()
        weights=self.effective_weights(delta)
        return context["lambda_"]*self.W+(self.B*weights[...,None,:])@self.B.T

    def memory_action(self,delta,r,context=None):
        context=self._context() if context is None else context
        weights=self.effective_weights(delta)
        return context["lambda_"]*r+((r@self.B)*weights)@self.B.T

    def field(self,state,u=None):
        if u is not None: raise ValueError("no external ports")
        z=self._tensor(state)
        if z.shape[-1]!=40: raise ValueError("interface state width must be40")
        d,w,p,r=z.split(10,dim=-1);c=self._context()
        return torch.cat((w,(self.base_momentum_field(z[...,:30],c)-r)/self.mass,
            -self.jpi_p*w-self.rpp*self.storage_base*p,c["kappa"]*w-self.memory_action(d,r,c)),-1)

    def matrices(self,state):
        z=self._tensor(state)
        if z.shape[-1]!=40: raise ValueError("explicit state required for R(delta)")
        c=self._context();Q=self.memory_matrix(z[...,:10]);shape=Q.shape[:-2]+(40,40)
        J=self.mass.new_zeros(shape);R=J.clone()
        J[...,:10,10:20],J[...,10:20,:10]=self.W,-self.W
        G=torch.diag(self.jpi_p)
        J[...,10:20,20:30],J[...,20:30,10:20]=G,-G
        J[...,10:20,30:],J[...,30:,10:20]=-c["kappa"]*self.W,c["kappa"]*self.W
        R[...,10:20,10:20]=torch.diag(self.damping)
        R[...,20:30,20:30]=torch.diag(self.rpp)
        R[...,30:,30:]=c["kappa"]*Q
        return J,R

    def fast_step(self,*args,**kwargs): raise ValueError("only true RK4 is registered for state-dependent Q")

    def encode(self,history,*,history_dt=None):
        return self.history_system(history,history_dt=history_dt)["r0"]

    def initialize(self,history,*,history_dt=None):
        h=self._tensor(history)
        return torch.cat((h[...,-1,:30],self.encode(h,history_dt=history_dt)),-1)

    def rollout(self,history,steps,dt,*,method="rk4",history_dt=None):
        if method!="rk4": raise ValueError("only true RK4 is registered")
        if type(steps) is not int or steps<0: raise ValueError("finite nonnegative rollout step count required")
        self._dt(dt)
        owns_cache=self._cache is None
        if owns_cache: self.begin_graph_cache()
        try:
            state=self.initialize(history,history_dt=history_dt);out=[state]
            for _ in range(steps):
                state=self.rk4(state,dt);out.append(state)
            return torch.stack(out,-2)
        finally:
            if owns_cache: self.end_graph_cache()

    def history_system(self,history,*,history_dt=None):
        dt=self.history_dt if history_dt is None else history_dt;self._history_dt(dt)
        h=self._tensor(history)
        if h.ndim not in (2,3) or h.shape[-2:]!=(11,30): raise ValueError("11 causal30-column rows required")
        if not torch.isfinite(h[...,:20]).all() or not torch.isfinite(h[...,-1,20:]).all():
            raise ValueError("finite delta/omega and terminal p required")
        c=self._context();omega=h[...,10:20]
        # Gamma is absent: p has an exact independent affine flow and Zp=0.
        slope=(omega[...,1:,:]-omega[...,:-1,:])/.01
        p=[h[...,-1,20:]]
        for i in range(9,-1,-1):
            p.append(torch.cat((p[-1],omega[...,i+1,:],slope[...,i,:]),-1)@self.p_backward.T)
        p=torch.stack(list(reversed(p)),-2)
        # Affine columns [particular, ten terminal-force basis vectors].
        affine=self.mass.new_zeros(h.shape[:-2]+(10,11))
        affine[...,1:]=self.W
        states=[affine];substeps=round(.01/dt)
        def rhs(value,d,w):
            weights=self.effective_weights(d)
            transposed=value.transpose(-1,-2)
            out=-(c["lambda_"]*transposed+((transposed@self.B)*weights[...,None,:])@self.B.T).transpose(-1,-2)
            forcing=torch.cat((c["kappa"]*w[...,None],torch.zeros_like(value[...,1:])), -1)
            return out+forcing
        for i in range(9,-1,-1):
            right,left=h[...,i+1,:20],h[...,i,:20]
            for j in range(substeps):
                x0=right+(left-right)*(j/substeps)
                xm=right+(left-right)*((j+.5)/substeps)
                x1=right+(left-right)*((j+1)/substeps)
                a=rhs(affine,x0[...,:10],x0[...,10:])
                b=rhs(affine-dt*a/2,xm[...,:10],xm[...,10:])
                cc=rhs(affine-dt*b/2,xm[...,:10],xm[...,10:])
                d=rhs(affine-dt*cc,x1[...,:10],x1[...,10:])
                affine=affine-dt/6*(a+2*b+2*cc+d)
            states.append(affine)
        states=torch.stack(list(reversed(states)),-3)
        rbar,Zr=states[...,0],states[...,1:]
        C=(.02/6)*(Zr[...,0:10:2,:,:]+4*Zr[...,1:10:2,:,:]+Zr[...,2:11:2,:,:])
        design=(C*self.rstar[None,:]/(.02*self.rstar[:,None])).reshape(h.shape[:-2]+(50,10))
        fbase=self.base_momentum_field(torch.cat((h[...,:20],p),-1),c)
        f=fbase-rbar
        b=(.02/6)*(f[...,0:10:2,:]+4*f[...,1:10:2,:]+f[...,2:11:2,:])-self.mass*(omega[...,2:11:2,:]-omega[...,0:10:2,:])
        beta=(b/(.02*self.rstar)).flatten(-2)
        F=Zr[...,0,:,:]*self.rstar[None,:]/self.rstar[:,None];q=rbar[...,0,:]/self.rstar
        augmented=torch.cat((design,math.sqrt(RIDGE)*F),-2)
        target=torch.cat((beta,-math.sqrt(RIDGE)*q),-1)
        qr,tri=torch.linalg.qr(augmented,mode="reduced")
        s=torch.linalg.solve_triangular(tri,(qr.transpose(-1,-2)@target[...,None]),upper=True).squeeze(-1)
        r0=self.rstar*s;r=rbar+(Zr@r0[...,None,:,None]).squeeze(-1)
        return dict(r0=r0,u=r[...,0,:],s=s,p_history=p,r_history=r,rbar=rbar,Zr=Zr,
            C=C,A=design,F=F,q=q,b=b,bscaled=beta,fbase=fbase,design=augmented,normal=augmented.transpose(-1,-2)@augmented,
            history_dt=dt)

    def initialization_diagnostics(self,history):
        s=self.history_system(history);res=(s["A"]@s["s"][...,None]).squeeze(-1)-s["bscaled"]
        prior=(s["F"]@s["s"][...,None]).squeeze(-1)+s["q"]
        return dict(history_dt=s["history_dt"],ridge=RIDGE,scaled_residual_l2=res.norm(dim=-1).detach().cpu().tolist(),
            ridge_penalty=(RIDGE*prior.square().sum(-1)).detach().cpu().tolist(),
            augmented_design_condition_number=torch.linalg.cond(s["design"]).detach().cpu().tolist(),
            claim_scope="causal ten terminal-force unknowns; no parameter identifiability claim")

    @torch.no_grad()
    def project_runtime_(self):
        super().project_nonnegative_()
        if self.gate_mode=="B" and self.a_bar.sum().item()==0:
            raise ControlDegenerate("projected B graph allzero: save completed update/Adam/RNG and stop control")
        return self

    def export(self,path):
        if self._cache is not None: raise RuntimeError("close graph cache before export")
        if self.history_dt not in (.005,.0025):
            raise ValueError("reference history step is qualification only; cannot publish a training checkpoint")
        payload=dict(format=FORMAT,mode="M",gate_mode=self.gate_mode,history_dt=self.history_dt,ridge=RIDGE,
          config=copy.deepcopy(self._source_config),source_identity=copy.deepcopy(self.source_identity),
          state_dict={k:v.detach().cpu().tolist() for k,v in self.state_dict().items()},
          requires_grad={k:p.requires_grad for k,p in self.named_parameters()},gamma="absent; fixed0",integration="true RK4")
        Path(path).write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        return payload

    @classmethod
    def from_force_graph_export(cls,path,gate_mode="A",*,history_dt=.0025):
        path=Path(path);parent=ForceGraphROM.from_export(path)
        if parent.mode!="M": raise ValueError("state gated source must have M physical parameterization")
        identity=dict(source_path=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parent=parent.source_identity)
        model=cls(parent._source_config,gate_mode,history_dt=history_dt,source_identity=identity)
        model.load_state_dict(parent.state_dict());return model

    @classmethod
    def from_export(cls,path):
        p=json.loads(Path(path).read_text(encoding="utf-8"))
        if p.get("format")!=FORMAT or p.get("ridge")!=RIDGE or p.get("mode")!="M": raise ValueError("wrong gated export")
        model=cls(p["config"],p["gate_mode"],history_dt=p["history_dt"],source_identity=p["source_identity"])
        template=model.state_dict()
        if set(template)!=set(p["state_dict"]): raise ValueError("export keys differ")
        values={k:torch.as_tensor(p["state_dict"][k],dtype=v.dtype) for k,v in template.items()}
        if any(v.shape!=template[k].shape or not torch.isfinite(v).all() for k,v in values.items()): raise ValueError("invalid export")
        model.load_state_dict(values)
        if (model.a_bar<0).any() or set(p["requires_grad"])!=set(dict(model.named_parameters())): raise ValueError("invalid weights/layout")
        for n,param in model.named_parameters(): param.requires_grad_(p["requires_grad"][n])
        return model
