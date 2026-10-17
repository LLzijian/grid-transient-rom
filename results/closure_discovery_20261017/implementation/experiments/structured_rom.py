"""Small explicit shared-gradient port-Hamiltonian ROM for Loop 0076."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from structure_grammar import as_specs, primitive_value_and_gradient

def _t(x): return torch.as_tensor(x,dtype=torch.float64)
def _spinv(x):
    x=_t(x); return torch.where(x>20,x,torch.log(torch.expm1(torch.clamp(x,min=1e-14))))
def _unit(x): return x/torch.linalg.vector_norm(x).clamp_min(1e-15)
@dataclass(frozen=True)
class ExchangeSpec: a: tuple; b: tuple; coefficient: float; name: str=""
@dataclass(frozen=True)
class DissipationSpec: a: tuple; coefficient: float; name: str=""

class AnchoredGRUEncoder(nn.Module):
    """Optional 11-step, 20->8->3 encoder with differentiable origin anchor."""
    def __init__(self, input_size=20, hidden_size=8, output_size=3):
        super().__init__(); self.gru=nn.GRU(input_size,hidden_size,batch_first=True,dtype=torch.float64); self.head=nn.Linear(hidden_size,output_size,bias=False,dtype=torch.float64)
    def _raw(self, history): return self.head(self.gru(history)[0][:,-1])
    def forward(self, history):
        if history.shape[-2:] != (11, self.gru.input_size): raise ValueError('expected [batch, 11, 20] input history')
        return self._raw(history)-self._raw(torch.zeros_like(history))

def numpy_field(payload, x, u=None):
    """Independent NumPy evaluator for an exported numeric field.

    It intentionally does not construct a torch module or use autograd.  It is
    provided for export/reload QA rather than training.
    """
    x=np.asarray(x,dtype=np.float64); scalar=x.ndim==1; x=np.atleast_2d(x)
    m=np.asarray(payload['mass']); D=np.asarray(payload['D']); n=len(m); np_=int(payload.get('n_p',n)); nq=int(payload['n_q']); ds=np.asarray(payload['delta_star']); phi=D@ds
    d,pi,p,q=x[:,:n],x[:,n:2*n],x[:,2*n:2*n+np_],x[:,2*n+np_:]; eta=d@D.T; s=np.c_[p,q]; gd=np.zeros_like(d); gs=s*np.asarray(payload.get('storage_base',np.r_[1/np.asarray(payload['gain']),np.ones(nq)]))
    if D.shape[0]: gd=((np.sin(phi+eta)-np.sin(phi))*np.asarray(payload['coupling']))@D
    for sp in payload.get('storage',[]):
        kind=sp['kind'].replace('-','_'); ind=sp.get('indices',()); c=sp['coefficient']; scale=sp.get('scale',()); ph=sp.get('phi',0.)
        if kind in ('sin','sin_eta','centered_sin'):
            j=ind[0]; gd+=((c*(np.cos(ph+eta[:,j])-np.cos(ph)))[:,None]*D[j])
        elif kind in ('cos','cos_eta','centered_cos'):
            j=ind[0]; gd+=((c*(-np.sin(ph+eta[:,j])+np.sin(ph)))[:,None]*D[j])
        elif kind in ('quadratic','state_quadratic','sq'):
            i=ind[0]; sc=scale[0] if scale else 1.; gs[:,i]+=c*s[:,i]/sc**2
        elif kind in ('product','state_product','cross'):
            i,j=ind; si,sj=scale or (1.,1.); gs[:,i]+=c*s[:,j]/(si*sj);gs[:,j]+=c*s[:,i]/(si*sj)
        elif kind in ('state_sin','state_sin_difference','sin_state'):
            i,j=ind;sc=scale[0] if scale else 1.; gd+=(c*s[:,i]/sc*np.cos(ph+eta[:,j]))[:,None]*D[j];gs[:,i]+=c*(np.sin(ph+eta[:,j])-np.sin(ph))/sc
        elif kind in ('state_cos','state_cos_difference','cos_state'):
            i,j=ind;sc=scale[0] if scale else 1.; gd+=(-c*s[:,i]/sc*np.sin(ph+eta[:,j]))[:,None]*D[j];gs[:,i]+=c*(np.cos(ph+eta[:,j])-np.cos(ph))/sc
    e=np.c_[gd,pi/m,gs]; J=np.asarray(payload['J']);R=np.asarray(payload['R']); f=e@(J-R).T
    B=np.asarray(payload.get('B',[]));
    if B.size:
        vals=[];vs=np.c_[pi,p,q]
        for port in payload.get('ports',[]):
            vv=np.zeros(len(x))
            for term in port.get('terms',[]):
                kind=term.get('kind','linear');i=term.get('index',term.get('indices',(0,))[0]);c=term.get('coefficient',1.);sc=term.get('scale',1.)
                vv+=c*(vs[:,i]/sc if kind=='linear' else (np.sin(phi[i]+eta[:,i])-np.sin(phi[i]) if kind in ('sin','sin_difference') else np.cos(phi[i]+eta[:,i])-np.cos(phi[i])))
            vals.append(vv)
        pv=np.stack(vals,-1) if u is None else np.asarray(u); f+=pv@B.T
    return f[0] if scalar else f

class StructuredROM(nn.Module):
    """Internal state is (delta, pi, p, q); interface state is (delta,omega,p,q)."""
    def __init__(self,mass,D=None,coupling=None,delta_star=None,damping=None,rate=None,
                 gain=None,n_q=0,scales=None,storage=None,exchange=None,dissipation=None,
                 ports=None,trainable=True,n_p=None,jpi_p=None,rpp=None,storage_base=None,**unused):
        super().__init__(); mass=_t(mass).reshape(-1); self.n=mass.numel(); self.n_p=int(self.n if n_p is None else n_p); self.n_q=int(n_q)
        if self.n_p != self.n: raise ValueError('0076 canonical M3 template requires n_p == n')
        if not torch.isfinite(mass).all() or torch.any(mass <= 0): raise ValueError('mass must be finite and strictly positive')
        D=_t(D if D is not None else np.zeros((0,self.n))).reshape(-1,self.n)
        if not torch.isfinite(D).all() or (D.numel() and torch.max(torch.abs(D.sum(-1))) > 1e-12): raise ValueError('D must be finite with gauge-invariant zero row sums')
        self.register_buffer('mass',mass); self.register_buffer('D',D)
        self.register_buffer('delta_star',_t(delta_star if delta_star is not None else np.zeros(self.n)))
        if not torch.isfinite(self.delta_star).all(): raise ValueError('delta_star must be finite')
        self.register_buffer('phi',D@self.delta_star)
        self.register_buffer('scales',_t(scales if scales is not None else np.ones(self.n_p+self.n_q)))
        if self.scales.numel()!=self.n_p+self.n_q: raise ValueError('scales must cover p,q')
        if not torch.isfinite(self.scales).all() or torch.any(self.scales <= 0): raise ValueError('scales must be finite and strictly positive')
        self.storage_specs=as_specs(storage); self.exchange_specs=tuple(x if isinstance(x,ExchangeSpec) else ExchangeSpec(**dict(x)) for x in (exchange or ())); self.dissipation_specs=tuple(x if isinstance(x,DissipationSpec) else DissipationSpec(**dict(x)) for x in (dissipation or ())); self.port_specs=tuple(dict(x) for x in (ports or ()))
        for spec in self.storage_specs:
            if not np.isfinite(spec.coefficient) or not np.isfinite(spec.phi) or any(not np.isfinite(a) or a <= 0 for a in spec.scale): raise ValueError('storage primitive coefficients, phases, and scales must be finite; scales positive')
        def positive(name,value,size):
            v=_t(value if value is not None else np.zeros(size)).reshape(-1)
            if v.numel()==1:v=v.repeat(size)
            if not torch.isfinite(v).all() or torch.any(v < 0): raise ValueError(f'{name} must be finite and nonnegative')
            self.register_buffer(name+'_mask',(v!=0).to(torch.float64)); setattr(self,name,nn.Parameter(_spinv(v.abs().clamp_min(1e-14)),requires_grad=trainable))
        ne=D.shape[0]; positive('coupling',coupling if coupling is not None else np.ones(ne),ne); positive('damping',damping,self.n)
        gv=_t(gain if gain is not None else np.ones(self.n_p));
        if gv.numel()==1:gv=gv.repeat(self.n_p)
        if not torch.isfinite(gv).all() or torch.any(gv <= 0): raise ValueError('gain must be finite and strictly positive')
        self.register_buffer('gain',gv) # fixed empirical scale: H storage is independently refittable
        positive('jpi_p',gv if jpi_p is None else jpi_p,self.n_p)
        rv=( _t(rate if rate is not None else np.zeros(self.n_p))*gv if rpp is None else rpp ); positive('rpp',rv,self.n_p)
        positive('storage_base',torch.cat((1/gv,torch.ones(self.n_q))) if storage_base is None else storage_base,self.n_p+self.n_q)
        self.storage_gamma=nn.ParameterList([nn.Parameter(_t(s.coefficient),requires_grad=trainable) for s in self.storage_specs])
        if any(not np.isfinite(s.coefficient) for s in self.exchange_specs): raise ValueError('exchange kappa must be finite')
        self.exchange_kappa=nn.ParameterList([nn.Parameter(_t(s.coefficient),requires_grad=trainable) for s in self.exchange_specs]); self.exchange_a=nn.ParameterList([nn.Parameter(_t(s.a),requires_grad=trainable) for s in self.exchange_specs]); self.exchange_b=nn.ParameterList([nn.Parameter(_t(s.b),requires_grad=trainable) for s in self.exchange_specs])
        if any(not np.isfinite(s.coefficient) or s.coefficient < 0 for s in self.dissipation_specs): raise ValueError('dissipation rho must be finite and nonnegative')
        self.register_buffer('dissipation_mask',torch.tensor([s.coefficient != 0 for s in self.dissipation_specs],dtype=torch.float64)); self.dissipation_rho=nn.ParameterList([nn.Parameter(_spinv(s.coefficient if s.coefficient else 1e-14),requires_grad=trainable) for s in self.dissipation_specs]); self.dissipation_a=nn.ParameterList([nn.Parameter(_t(s.a),requires_grad=trainable) for s in self.dissipation_specs])
        if any(not torch.isfinite(a).all() or torch.linalg.vector_norm(a).item() == 0 for a in self.exchange_a) or any(not torch.isfinite(b).all() or torch.linalg.vector_norm(b).item() == 0 for b in self.exchange_b) or any(not torch.isfinite(a).all() or torch.linalg.vector_norm(a).item() == 0 for a in self.dissipation_a): raise ValueError('exchange/dissipation directions must be finite and nonzero')
        if any(a.numel()!=self.n+self.n_p+self.n_q for a in list(self.exchange_a)+list(self.exchange_b)+list(self.dissipation_a)): raise ValueError('extra directions act only on complete (pi,p,q) coordinates')
        self.port_direction=nn.ParameterList(); self.port_coefficient=nn.ParameterList()
        if len(self.port_specs)>2: raise ValueError('at most two ports')
        for p in self.port_specs:
            d=p.get('direction',p.get('b')); terms=p.get('terms',())
            if d is None or len(terms)>8 or _t(d).numel()!=self.n+self.n_p+self.n_q or not torch.isfinite(_t(d)).all() or torch.linalg.vector_norm(_t(d)).item()==0: raise ValueError('ports require finite nonzero (pi,p,q) direction and <=8 terms')
            for term in terms:
                if not np.isfinite(term.get('coefficient',1.)) or not np.isfinite(term.get('scale',1.)) or term.get('scale',1.) <= 0: raise ValueError('port coefficients must be finite and scales positive')
            self.port_direction.append(nn.Parameter(_t(d),requires_grad=trainable)); self.port_coefficient.append(nn.Parameter(_t([t.get('coefficient',1.) for t in terms]),requires_grad=trainable))
    @property
    def dim(self):return 2*self.n+self.n_p+self.n_q
    @property
    def interface_dim(self):return self.dim
    def _pos(self,name):return torch.nn.functional.softplus(getattr(self,name))*getattr(self,name+'_mask')
    def _split(self,x):return x[...,:self.n],x[...,self.n:2*self.n],x[...,2*self.n:2*self.n+self.n_p],x[...,2*self.n+self.n_p:]
    def _eta_s(self,x):d,_,p,q=self._split(x); return d@self.D.T,torch.cat((p,q),-1)
    def energy_and_gradient(self,x):
        x=x if torch.is_tensor(x) else _t(x); d,pi,p,q=self._split(x); eta,s=self._eta_s(x); base=self._pos('storage_base')
        H=.5*(pi*pi/self.mass).sum(-1)+.5*(base*s*s).sum(-1); gd=torch.zeros_like(d); gs=base*s
        if self.D.shape[0]:
            k=self._pos('coupling'); z=self.phi+eta; H=H+(k*(torch.cos(self.phi)-torch.cos(z)-torch.sin(self.phi)*eta)).sum(-1); gd=((torch.sin(z)-torch.sin(self.phi))*k)@self.D
        for gamma,spec in zip(self.storage_gamma,self.storage_specs):
            v,ge,gi=primitive_value_and_gradient(spec,eta,s); H=H+gamma*v; gd=gd+(gamma*ge)@self.D; gs=gs+gamma*gi
        return H,torch.cat((gd,pi/self.mass,gs[...,:self.n_p],gs[...,self.n_p:]),-1)
    def energy(self,x):return self.energy_and_gradient(x)[0]
    def gradient(self,x,create_graph=True):return self.energy_and_gradient(x)[1]
    def matrices(self):
        z=torch.zeros((self.dim,self.dim),dtype=self.mass.dtype,device=self.mass.device); J=z.clone(); R=z.clone(); n=self.n; eye=torch.eye(n,dtype=z.dtype,device=z.device)
        J[:n,n:2*n]=eye; J[n:2*n,:n]=-eye; G=torch.diag(self._pos('jpi_p')); J[n:2*n,2*n:2*n+self.n_p]=G; J[2*n:2*n+self.n_p,n:2*n]=-G
        R[n:2*n,n:2*n]=torch.diag(self._pos('damping')); R[2*n:2*n+self.n_p,2*n:2*n+self.n_p]=torch.diag(self._pos('rpp'))
        for k,a,b in zip(self.exchange_kappa,self.exchange_a,self.exchange_b):
            aa=torch.zeros(self.dim,dtype=z.dtype,device=z.device);bb=aa.clone();aa[n:]=_unit(a);bb[n:]=_unit(b);J=J+k*(torch.outer(aa,bb)-torch.outer(bb,aa))
        for i,(rho,a) in enumerate(zip(self.dissipation_rho,self.dissipation_a)):
            aa=torch.zeros(self.dim,dtype=z.dtype,device=z.device);aa[n:]=_unit(a);R=R+(torch.nn.functional.softplus(rho)*self.dissipation_mask[i])*torch.outer(aa,aa)
        return J,R
    def port_values(self,x):
        x=x if torch.is_tensor(x) else _t(x);eta,s=self._eta_s(x);_,pi,p,q=self._split(x); vstate=torch.cat((pi,p,q),-1);out=[]
        for p,cs in zip(self.port_specs,self.port_coefficient):
            val=eta.sum(-1)*0.
            for c,t in zip(cs,p.get('terms',())):
                kind=t.get('kind','linear').lower();i=int(t.get('index',t.get('indices',(0,))[0]));sc=t.get('scale',1.)
                if kind=='linear':b=vstate[...,i]/sc
                elif kind in ('sin','sin_difference'):b=torch.sin(self.phi[i]+eta[...,i])-torch.sin(self.phi[i])
                elif kind in ('cos','cos_difference'):b=torch.cos(self.phi[i]+eta[...,i])-torch.cos(self.phi[i])
                else:raise ValueError('port basis must be linear, sin, or cos')
                val=val+c*b
            out.append(val)
        return torch.stack(out,-1) if out else eta.new_zeros(eta.shape[:-1]+(0,))
    def port_matrix(self):
        B=torch.zeros((self.dim,len(self.port_specs)),dtype=self.mass.dtype,device=self.mass.device)
        for j,d in enumerate(self.port_direction):B[self.n:,j]=_unit(d)
        return B
    def field(self,x,u=None,return_parts=False):
        x=x if torch.is_tensor(x) else _t(x);_,e=self.energy_and_gradient(x);J,R=self.matrices();pv=self.port_values(x) if u is None else (u if torch.is_tensor(u) else _t(u));f=e@(J-R).T
        if pv.shape[-1]:f=f+pv@self.port_matrix().T
        return (f,e,pv,J,R) if return_parts else f
    def interface_to_state(self,z):z=z if torch.is_tensor(z) else _t(z);d,w,p,q=self._split(z);return torch.cat((d,w*self.mass,p,q),-1)
    def state_to_interface(self,x):d,pi,p,q=self._split(x);return torch.cat((d,pi/self.mass,p,q),-1)
    def interface_field(self,z,u=None):return self.state_to_interface(self.field(self.interface_to_state(z),u=u))
    def export(self,path):
        J,R=self.matrices(); B=self.port_matrix()
        raw=sum(p.numel() for p in self.parameters() if p.requires_grad)
        masked=sum(int((getattr(self,n+'_mask') == 0).sum()) for n in ('coupling','damping','jpi_p','rpp','storage_base') if getattr(self,n).requires_grad)
        masked+=sum(int(self.dissipation_mask[i] == 0) for i,p in enumerate(self.dissipation_rho) if p.requires_grad)
        directions=list(self.exchange_a)+list(self.exchange_b)+list(self.dissipation_a)+list(self.port_direction)
        radial=sum(int(p.requires_grad) for p in directions)
        # Gradient atoms are counted before algebraic collection.  A centered
        # trig expression is one atom, even though it contains subtractive
        # centering terms; product primitives contribute one atom to each of
        # their two state-gradient coordinates.
        atoms=torch.zeros(self.dim,dtype=torch.float64); atoms[self.n:2*self.n]=1
        for edge in range(self.D.shape[0]):
            if self._pos('coupling')[edge] != 0: atoms[:self.n] += (self.D[edge] != 0)
        base=self._pos('storage_base')
        atoms[2*self.n:] += (base != 0)
        for gamma,spec in zip(self.storage_gamma,self.storage_specs):
            if gamma.detach() == 0: continue
            kind=spec.kind.lower().replace('-','_'); ind=spec.indices
            if kind in ('sin','sin_eta','centered_sin','cos','cos_eta','centered_cos'):
                atoms[:self.n] += (self.D[ind[0]] != 0)
            elif kind in ('quadratic','state_quadratic','sq'):
                atoms[2*self.n+ind[0]] += 1
            elif kind in ('product','state_product','cross'):
                atoms[2*self.n+ind[0]] += 1; atoms[2*self.n+ind[1]] += 1
            elif kind in ('state_sin','state_sin_difference','sin_state','state_cos','state_cos_difference','cos_state'):
                atoms[2*self.n+ind[0]] += 1; atoms[:self.n] += (self.D[ind[1]] != 0)
        Jbase=torch.zeros_like(J); Rbase=torch.zeros_like(R); n=self.n; eye=torch.eye(n,dtype=J.dtype)
        Jbase[:n,n:2*n]=eye; Jbase[n:2*n,:n]=-eye; G=torch.diag(self._pos('jpi_p')); Jbase[n:2*n,2*n:2*n+self.n_p]=G; Jbase[2*n:2*n+self.n_p,n:2*n]=-G
        Rbase[n:2*n,n:2*n]=torch.diag(self._pos('damping')); Rbase[2*n:2*n+self.n_p,2*n:2*n+self.n_p]=torch.diag(self._pos('rpp'))
        weighted=lambda A: int((((A != 0).sum(dim=0).to(atoms.dtype))*atoms).sum().item())
        base_nnz=weighted(Jbase-Rbase); ex_nnz=0; dis_nnz=0
        for k,a,b in zip(self.exchange_kappa,self.exchange_a,self.exchange_b):
            if k.detach() != 0:
                aa=torch.zeros(self.dim,dtype=J.dtype); bb=aa.clone(); aa[n:]=_unit(a); bb[n:]=_unit(b); ex_nnz += weighted(k*(torch.outer(aa,bb)-torch.outer(bb,aa)))
        for i,(rho,a) in enumerate(zip(self.dissipation_rho,self.dissipation_a)):
            if self.dissipation_mask[i] != 0:
                aa=torch.zeros(self.dim,dtype=J.dtype); aa[n:]=_unit(a); dis_nnz += weighted(torch.nn.functional.softplus(rho)*torch.outer(aa,aa))
        port_nnz=sum(int(torch.count_nonzero(B[:,j]))*sum(int(c.detach()!=0) for c in cs) for j,cs in enumerate(self.port_coefficient))
        counts={'active_groups':{'storage':len(self.storage_specs),'exchange':len(self.exchange_specs),'dissipation':len(self.dissipation_specs),'ports':len(self.port_specs)},'raw_trainable_scalar_count':raw,'masked_inactive_scalar_coordinates':masked,'normalized_direction_radial_redundancies':radial,'effective_parameter_coordinate_upper_bound':raw-masked-radial,'effective_parameter_coordinate_note':'upper bound only; normalized exchange a/b directions can have further non-identifiability','expanded_rhs_uncollected_scalar_contributions':{'base_J_R':base_nnz,'exchange_J':ex_nnz,'extra_R':dis_nnz,'B_times_active_port_terms':port_nnz,'total':base_nnz+ex_nnz+dis_nnz+port_nnz,'convention':'for every RHS row, sum each nonzero matrix coefficient once per analytic gradient atom in its column; centered trig is one atom and state-angle primitive has state plus angle atoms; add each nonzero B entry once per active port term, before collection'}}
        payload={'mass':self.mass.tolist(),'D':self.D.tolist(),'delta_star':self.delta_star.tolist(),'coupling':self._pos('coupling').detach().tolist(),'damping':self._pos('damping').detach().tolist(),'gain':self.gain.tolist(),'jpi_p':self._pos('jpi_p').detach().tolist(),'rpp':self._pos('rpp').detach().tolist(),'storage_base':self._pos('storage_base').detach().tolist(),'n_p':self.n_p,'n_q':self.n_q,'scales':self.scales.tolist(),'storage':[{**vars(s),'coefficient':float(c.detach())} for s,c in zip(self.storage_specs,self.storage_gamma)],'exchange':[{**vars(s),'a':a.detach().tolist(),'b':b.detach().tolist(),'coefficient':float(c.detach())} for s,c,a,b in zip(self.exchange_specs,self.exchange_kappa,self.exchange_a,self.exchange_b)],'dissipation':[{**vars(s),'a':a.detach().tolist(),'coefficient':float(torch.nn.functional.softplus(c).detach()*self.dissipation_mask[i])} for i,(s,c,a) in enumerate(zip(self.dissipation_specs,self.dissipation_rho,self.dissipation_a))],'ports':[{**p,'direction':d.detach().tolist(),'terms':[{**t,'coefficient':float(c)} for t,c in zip(p.get('terms',()),cs.detach())]} for p,d,cs in zip(self.port_specs,self.port_direction,self.port_coefficient)],'J':J.detach().tolist(),'R':R.detach().tolist(),'B':B.detach().tolist(),'counts':counts}
        Path(path).write_text(json.dumps(payload,indent=2));return payload
    @classmethod
    def from_export(cls,path):return cls(**json.loads(Path(path).read_text()))
