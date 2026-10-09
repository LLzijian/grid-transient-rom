"""Executable T4-N reference: scalar energies, skew exchanges, PSD dissipation,
limited state-driven ports. State-dependent skew does NOT imply Jacobi/Poisson.
Internal layout: delta[N], pi[N], p[N], q[N*rank] (machine-major).
"""
from itertools import combinations
import numpy as np
import torch
from torch import nn
from .data import Physics

class FunctionNet(nn.Module):
    def __init__(self,inputs,outputs,hidden=32):
        super().__init__();self.inputs=inputs;self.outputs=outputs
        self.net=nn.Sequential(nn.Linear(inputs,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,outputs))
        nn.init.normal_(self.net[-1].weight,std=0.01);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)
    def value_gradient(self,x):
        """Exact analytic input Jacobian, itself differentiable in x and weights."""
        a,b,c=self.net[0],self.net[2],self.net[4]
        h1=torch.tanh(a(x));h2=torch.tanh(b(h1));value=c(h2)
        jac=(c.weight*(1-h2.square()).unsqueeze(-2))@b.weight
        jac=(jac*(1-h1.square()).unsqueeze(-2))@a.weight
        return value,jac

class PowerGraph(nn.Module):
    def __init__(self,physics:Physics,scales,rank=2,hidden=32,strength=0.01):
        super().__init__();physics.check();self.n=physics.n;self.rank=rank;self.hidden=hidden;self.strength=float(strength)
        self.reuse_origin_terms=False  # Optional executor, not serialized scientific state.
        self.dim=(3+rank)*self.n;self.m=2+rank;self.sdim=1+rank
        for key in ['mass','incidence','delta_star','pm_star','coupling','damping','rate','gain']:
            self.register_buffer(key,torch.as_tensor(getattr(physics,key),dtype=torch.float64))
        self.register_buffer('scales',torch.as_tensor(scales,dtype=torch.float64))
        self.register_buffer('mu',torch.log(torch.stack([self.mass,self.damping.clamp_min(1e-8),self.rate,self.gain],-1)))
        ends=[]
        for col in physics.incidence.T:ends.append([int(np.flatnonzero(col>0)[0]),int(np.flatnonzero(col<0)[0])])
        self.register_buffer('edges',torch.tensor(ends,dtype=torch.long))
        adj=np.abs(physics.incidence)@np.diag(np.maximum(physics.coupling,0))@np.abs(physics.incidence).T
        np.fill_diagonal(adj,0);adj=adj/np.maximum(adj.sum(-1,keepdims=True),1e-12)
        self.register_buffer('adjacency',torch.as_tensor(adj,dtype=torch.float64))
        self.register_buffer('phase',self.delta_star@self.incidence)
        self.pairs=list(combinations(range(self.m),2))
        self.log_coupling=nn.Parameter(torch.zeros_like(self.coupling))
        self.node_energy=FunctionNet(self.sdim+4,1,hidden)
        self.edge_energy=FunctionNet(2+2*self.sdim+8,1,hidden)
        self.node_exchange=FunctionNet(self.m+4,len(self.pairs),hidden)
        self.node_dissipation=FunctionNet(self.m+4,self.m,hidden)
        self.edge_exchange=FunctionNet(2+2*self.m+8,self.m*self.m,hidden)
        self.edge_dissipation=FunctionNet(2+2*self.m+8,self.m,hidden)
        self.port=FunctionNet(self.m+4,2,hidden)  # Two fixed directions per machine: pi and p.
        self.double()
    def split(self,x):
        n=self.n
        return x[...,:n],x[...,n:2*n],x[...,2*n:3*n],x[...,3*n:].reshape(*x.shape[:-1],n,self.rank)
    def assemble(self,d,pi,p,q):return torch.cat([d,pi,p,q.flatten(-2)],-1)
    def physical(self,x):
        d,pi,p,_=self.split(x);return torch.stack([d,pi/self.mass,p],-1)
    def from_physical(self,y,q):return self.assemble(y[...,0],y[...,1]*self.mass,y[...,2],q)
    def initial(self,history,encoder):return self.from_physical(history[:,-1],encoder(history))
    def _mu(self,t):return self.mu.expand(*t.shape[:-2],-1,-1)
    def _scaled_s(self,p,q):return torch.cat([p.unsqueeze(-1)/self.scales[2],q],-1)
    def _local(self,w,p,q):return torch.cat([w.unsqueeze(-1)/self.scales[1],p.unsqueeze(-1)/self.scales[2],q],-1)
    def _node_input(self,v):return torch.cat([v,self._mu(v)],-1)
    def _edge_input(self,eta,v,swap=False):
        i,j=self.edges[:,0],self.edges[:,1]
        si,sj=v[...,i,:],v[...,j,:];mi,mj=self._mu(v)[...,i,:],self._mu(v)[...,j,:]
        s=torch.sin(self.phase+eta)-torch.sin(self.phase);c=torch.cos(self.phase+eta)-torch.cos(self.phase)
        if swap:return torch.cat([-s.unsqueeze(-1),c.unsqueeze(-1),sj,si,mj,mi],-1)
        return torch.cat([s.unsqueeze(-1),c.unsqueeze(-1),si,sj,mi,mj],-1)
    def _edge_energy_from_v(self,v):
        eta=v[...,0];si=v[...,1:1+self.sdim];sj=v[...,1+self.sdim:]
        shape=v.shape[:-2];i,j=self.edges[:,0],self.edges[:,1]
        mi=self.mu[i].expand(*shape,-1,-1);mj=self.mu[j].expand(*shape,-1,-1)
        s=torch.sin(self.phase+eta)-torch.sin(self.phase);c=torch.cos(self.phase+eta)-torch.cos(self.phase)
        a=torch.cat([s[...,None],c[...,None],si,sj,mi,mj],-1)
        b=torch.cat([-s[...,None],c[...,None],sj,si,mj,mi],-1)
        return .5*(self.edge_energy(a).squeeze(-1)+self.edge_energy(b).squeeze(-1))
    @staticmethod
    def _centered(fn,v):
        v0=torch.zeros_like(v,requires_grad=True);h0=fn(v0)
        g0=torch.autograd.grad(h0.sum(),v0,create_graph=True)[0]
        return fn(v)-h0-(g0*v).sum(-1)
    def energy(self,x):
        d,pi,p,q=self.split(x);eta=d@self.incidence;phase=self.phase
        k=self.coupling*torch.exp(self.log_coupling)
        kinetic=.5*(pi.square()/self.mass).sum(-1)
        potential=(k*(torch.cos(phase)-torch.cos(phase+eta)-torch.sin(phase)*eta)).sum(-1)
        base=.5*(p.square()/self.gain).sum(-1)+.5*q.square().sum((-1,-2))
        sn=self._scaled_s(p,q)
        hn=self._centered(lambda a:self.node_energy(self._node_input(a)).squeeze(-1),sn).sum(-1)
        si,sj=sn[...,self.edges[:,0],:],sn[...,self.edges[:,1],:]
        ve=torch.cat([eta.unsqueeze(-1),si,sj],-1)
        he=self._centered(self._edge_energy_from_v,ve).sum(-1)
        return kinetic+potential+base+self.strength*(hn+he)
    def _edge_energy_gradient(self,eta,s):
        fa=self._edge_input(eta,s);fb=self._edge_input(eta,s,True)
        _,ga=self.edge_energy.value_gradient(fa);_,gb=self.edge_energy.value_gradient(fb)
        ga=ga.squeeze(-2);gb=gb.squeeze(-2)
        angle=.5*((ga[...,0]-gb[...,0])*torch.cos(self.phase+eta)
                   -(ga[...,1]+gb[...,1])*torch.sin(self.phase+eta))
        left=.5*(ga[...,2:2+self.sdim]+gb[...,2+self.sdim:2+2*self.sdim])
        right=.5*(ga[...,2+self.sdim:2+2*self.sdim]+gb[...,2:2+self.sdim])
        return angle,left,right
    def origin_terms(self,x):
        """Batch-shaped differentiable origins for one rollout/parameter version.

        The caller owns this tuple. Do not detach it, put it on the model, or
        retain it across backward/optimizer updates. Physical mu stays intact.
        """
        if type(self) is not PowerGraph:raise ValueError('Origin reuse supports only the exact PowerGraph')
        d,pi,p,q=self.split(x)
        sn0=torch.zeros_like(self._scaled_s(p,q));eta0=torch.zeros_like(d@self.incidence)
        _,gn0=self.node_energy.value_gradient(self._node_input(sn0))
        eg0,el0,er0=self._edge_energy_gradient(eta0,sn0)
        local0=torch.zeros_like(self._local(pi/self.mass,p,q))
        port0=self.port(self._node_input(local0))
        return gn0,eg0,el0,er0,port0
    def energy_gradient(self,x,origins=None):
        """Analytic chain rule; avoids nested autograd graphs at every RK stage.

        energy() remains an independently differentiated reference in tests.
        """
        d,pi,p,q=self.split(x);eta=d@self.incidence;sn=self._scaled_s(p,q)
        _,gn=self.node_energy.value_gradient(self._node_input(sn))
        if origins is None:
            _,gn0=self.node_energy.value_gradient(self._node_input(torch.zeros_like(sn)))
        else:gn0=origins[0]
        node=(gn-gn0).squeeze(-2)[...,:self.sdim]
        ii,jj=self.edges[:,0],self.edges[:,1]
        eg,el,er=self._edge_energy_gradient(eta,sn)
        if origins is None:eg0,el0,er0=self._edge_energy_gradient(torch.zeros_like(eta),torch.zeros_like(sn))
        else:eg0,el0,er0=origins[1:4]
        internal=node.index_add(-2,ii,el-el0).index_add(-2,jj,er-er0)
        k=self.coupling*torch.exp(self.log_coupling)
        gd=((torch.sin(self.phase+eta)-torch.sin(self.phase))*k+self.strength*(eg-eg0))@self.incidence.T
        gp=p/self.gain+self.strength*internal[...,0]/self.scales[2]
        gq=q+self.strength*internal[...,1:]
        return self.assemble(gd,pi/self.mass,gp,gq)
    def _force(self,x,e,origins=None):
        d,pi,p,q=self.split(x);ed,ew,ep,eq=self.split(e)
        w=pi/self.mass;eta=d@self.incidence
        local=self._local(w,p,q);feat=self._node_input(local)
        effort=torch.cat([ew[...,None],ep[...,None],eq],-1)
        k=self.strength*self.node_exchange(feat)
        # Nonzero pi-q initialization keeps the encoder connected to the dynamics.
        offsets=torch.zeros(len(self.pairs),dtype=x.dtype,device=x.device)
        offsets[self.pairs.index((0,2))]=self.strength
        k=k+offsets
        force=torch.zeros_like(effort)
        for a,(i,j) in enumerate(self.pairs):
            u=torch.nn.functional.one_hot(torch.tensor(i,device=x.device),self.m).to(x.dtype)
            v=torch.nn.functional.one_hot(torch.tensor(j,device=x.device),self.m).to(x.dtype)
            force=force+k[...,a,None]*(effort[...,j,None]*u-effort[...,i,None]*v)
        rho0=torch.cat([torch.sqrt(self.damping)[...,None],torch.sqrt(self.rate*self.gain)[...,None],
                        torch.full((self.n,self.rank),.5,dtype=x.dtype,device=x.device)],-1)
        ell=rho0+self.strength*self.node_dissipation(feat)
        force=force-ell.square()*effort
        diss=(ell.square()*effort.square()).sum((-1,-2))
        fi=self._edge_input(eta,local);fj=self._edge_input(eta,local,True)
        K1=self.edge_exchange(fi).reshape(*fi.shape[:-1],self.m,self.m)
        K2=self.edge_exchange(fj).reshape(*fi.shape[:-1],self.m,self.m)
        K=.5*self.strength*(K1-K2.transpose(-1,-2))
        ii,jj=self.edges[:,0],self.edges[:,1];ei,ej=effort[...,ii,:],effort[...,jj,:]
        di=torch.einsum('...eab,...eb->...ea',K,ej)
        dj=-torch.einsum('...eba,...eb->...ea',K,ei)
        le=.5*self.strength*(self.edge_dissipation(fi)+self.edge_dissipation(fj))
        diff=ei-ej;di=di-.5*le.square()*diff;dj=dj+.5*le.square()*diff
        force=force.index_add(-2,ii,di).index_add(-2,jj,dj)
        diss=diss+.5*(le.square()*diff.square()).sum((-1,-2))
        # Center only the learned part of state-driven supply; mu is not set to zero.
        port=self.port(feat)
        port0=self.port(self._node_input(torch.zeros_like(local))) if origins is None else origins[4]
        ports=self.strength*(port-port0)
        up=ports[...,1]-self.gain*w
        ud=ports[...,0]
        dpi=-ed+force[...,0]+p+ud
        dp=force[...,1]+up;dq=force[...,2:]
        f=self.assemble(w,dpi,dp,dq)
        supply=(ew*(p+ud)+ep*up).sum(-1)
        return f,{'dissipation':diss,'supply':supply,'expected_Hdot':supply-diss,'port_pi':ud,'port_p':up}
    def field(self,x,create_graph=True,details=False,origins=None):
        if not details and getattr(self,'_compiled_field',None) is not None:
            with torch.set_grad_enabled(create_graph):
                if getattr(self,'_compiled_uses_origins',False):
                    current=self.origin_terms(x) if origins is None else origins
                    return self._compiled_field(x,*current)
                if origins is None:return self._compiled_field(x)
                # Explicit origins can still use eager execution when an older
                # x-only trace exists; rollout rejects that mixed executor mode.
        with torch.set_grad_enabled(create_graph):
            xx=x
            e=self.energy_gradient(xx,origins)
            f,parts=self._force(xx,e,origins)
            if details:
                with torch.enable_grad():h=self.energy(xx)
                parts.update(H=h,Hdot=(e*f).sum(-1),gradient=e)
        if not create_graph:
            f=f.detach();parts={k:v.detach() for k,v in parts.items()}
        return (f,parts) if details else f
    def function_modules(self):
        return {name:getattr(self,name) for name in ['node_energy','edge_energy','node_exchange','node_dissipation','edge_exchange','edge_dissipation','port']}

class UnstructuredGraph(PowerGraph):
    """U control keeps kinematics and graph sharing, but has no H/J/R claim."""
    def __init__(self,*args,u_hidden=None,**kwargs):
        super().__init__(*args,**kwargs)
        width=self.hidden if u_hidden is None else int(u_hidden)
        self.node_rhs=FunctionNet(self.m+4,self.m,width).double()
        self.edge_rhs=FunctionNet(2+2*self.m+8,self.m,width).double()
        # Remove unused structured trainables from optimization.
        for mod in self.function_modules().values():
            for p in mod.parameters():p.requires_grad_(False)
    def field(self,x,create_graph=True,details=False):
        if self.reuse_origin_terms:raise ValueError('Origin reuse is unsupported for UnstructuredGraph')
        if not details and getattr(self,'_compiled_field',None) is not None:
            with torch.set_grad_enabled(create_graph):return self._compiled_field(x)
        d,pi,p,q=self.split(x);w=pi/self.mass;eta=d@self.incidence
        phase=self.phase;k=self.coupling*torch.exp(self.log_coupling)
        electric=((torch.sin(phase+eta)-torch.sin(phase))*k)@self.incidence.T
        loc=self._local(w,p,q);nf=self._node_input(loc);n0=self._node_input(torch.zeros_like(loc))
        force=self.strength*(self.node_rhs(nf)-self.node_rhs(n0))
        for rev,ends in [(False,self.edges[:,0]),(True,self.edges[:,1])]:
            ef=self._edge_input(eta,loc,rev);ez=self._edge_input(torch.zeros_like(eta),torch.zeros_like(loc),rev)
            force=force.index_add(-2,ends,self.strength*(self.edge_rhs(ef)-self.edge_rhs(ez)))
        dpi=p-electric-self.damping*w+force[...,0]
        dp=-self.rate*p-self.gain*w+force[...,1]
        dq=-.5*q+force[...,2:]
        f=self.assemble(w,dpi,dp,dq)
        if not create_graph:f=f.detach()
        return (f,{'power_structure':'not_applicable'}) if details else f
