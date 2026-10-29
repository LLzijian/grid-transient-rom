"""Fixed-plane conservative/dissipative memory transport, external [d,w,p,r].

Physical coordinate count stays 40; variant A/S/P is distinct from legacy mode M.
No event loader, optimizer, plane scan, numerical qualification or run authorization.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from force_graph_model_20261028 import RIDGE
from state_gated_memory_model_20261028 import StateGatedMemoryROM

FORMAT='memory_transport_model_20261029'
VARIANTS=('A','S','P')


def helmert_basis():
    """Deterministic ten-by-nine Euclidean orthonormal basis of 1-perp."""
    basis=torch.zeros(10,9,dtype=torch.float64)
    for j in range(9):
        denominator=math.sqrt((j+1)*(j+2))
        basis[:j+1,j]=1/denominator;basis[j+1,j]=-(j+1)/denominator
    return basis


def prepare_plane(source_export,*,expected_sha256,output_directory):
    """One exclusive output transaction; no implicit retry or scientific release.

    The controller must bind this sole directory before calling. Existing output
    is refused before reading the source or computing a decomposition. Failures
    preserve an immutable result and are re-raised, never retried here.
    """
    output=Path(output_directory);output.mkdir(parents=True,exist_ok=False)
    def record(name,value):
        with (output/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush()
    record('attempt.json',dict(expected_source_sha256=expected_sha256,method='one Helmert9D eigh',authorization='external_root_required'))
    try:
        result=_prepare_plane_values(source_export,expected_sha256=expected_sha256)
    except BaseException as exc:
        record('result.json',dict(status='PLANE_PREPARATION_FAILED',expected_source_sha256=expected_sha256,
                                 error_type=type(exc).__name__,message=str(exc)))
        raise
    record('result.json',result);return result


def _prepare_plane_values(source_export,*,expected_sha256):
    """Definition only: root authorizes ONE frozen-source decomposition externally.

    The sole eigh is nine-dimensional. The full symmetric Q norm equals its
    largest eigenvalue, also max(lambda, largest restricted eigenvalue), so no
    second eigensolve/SVD is used for the prescribed cutoff rule.
    """
    path=Path(source_export);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=expected_sha256:raise ValueError('frozen A96 source hash differs')
    model=StateGatedMemoryROM.from_export(path).cpu().double()
    if model.gate_mode!='A' or model.mode!='M':raise ValueError('plane source must be the frozen A closure')
    with torch.no_grad():
        if not torch.isfinite(model.kappa) or not torch.isfinite(model.lambda_) or model.kappa<=0 or model.lambda_<=0:
            raise ValueError('strict positive kappa/lambda source required; no boundary rescue')
        model_values=model._new_context()
        Q=model_values['lambda_']*model.W+(model.B*(model.lambda_reference*model.a_bar))@model.B.T
        basis=helmert_basis();eigenvalues,eigenvectors=torch.linalg.eigh(basis.T@Q@basis)
        norm=float(torch.maximum(model.lambda_.abs(),eigenvalues.abs().max()))
        gap=float(eigenvalues[2]-eigenvalues[1]);threshold=100*torch.finfo(torch.float64).eps*max(1.,norm)
        record=dict(source_sha256=digest,source_identity=copy.deepcopy(model.source_identity),
            basis_method='fixed Helmert ten-by-nine; Euclidean 1-perp',eigh_count=1,
            restricted_eigenvalues=eigenvalues.tolist(),cutoff_gap=gap,cutoff_threshold=threshold,
            cutoff_pass=gap>threshold,claim_scope='numerical fixed-plane selection; no physical-mode identification')
        if not record['cutoff_pass']:
            return dict(record,status='PLANE_UNRESOLVED',u=None,v=None)
        plane=basis@eigenvectors[:,:2]
        for column in range(2):
            index=int(plane[:,column].abs().argmax())
            if plane[index,column]<0:plane[:,column].neg_()
        return dict(record,status='COMPLETED',u=plane[:,0].tolist(),v=plane[:,1].tolist())


class MemoryTransportROM(StateGatedMemoryROM):
    def __init__(self,source,variant,*,u,v,history_dt=.0025,source_identity=None,plane_identity=None):
        if variant not in VARIANTS:raise ValueError('registered variants A/S/P only')
        super().__init__(source,'A',history_dt=history_dt,source_identity=source_identity)
        self.variant=variant;self.plane_identity=copy.deepcopy(plane_identity or {})
        for name,value in (('u',u),('v',v)):
            vector=torch.as_tensor(value,dtype=torch.float64).detach().cpu().clone()
            if vector.shape!=(10,) or not torch.isfinite(vector).all():raise ValueError('finite ten-vector plane required')
            self.register_buffer(name,vector)
        self._validate_plane()
        if variant=='S':self.s_raw=nn.Parameter(torch.tensor(0.,dtype=torch.float64))
        if variant=='P':self.d_raw=nn.Parameter(torch.tensor(0.,dtype=torch.float64))

    @property
    def mode(self):return 'M'

    @mode.setter
    def mode(self,value):
        if value!='M':raise ValueError('transport mode stays legacy M; use variant A/S/P')

    @property
    def gate_mode(self):return 'A'

    @gate_mode.setter
    def gate_mode(self,value):
        if value!='A':raise ValueError('transport uses the constant A graph, not a state gate')

    @property
    def variant(self):return self._variant

    @variant.setter
    def variant(self,value):
        if value not in VARIANTS or hasattr(self,'_variant') and value!=self._variant:
            raise ValueError('variant is immutable; construct a distinct registered model')
        self._variant=value

    @staticmethod
    def _history_dt(dt):
        if isinstance(dt,bool) or dt not in (.0025,.00125):
            raise ValueError('registered history steps .0025/.00125 only')

    def _validate_plane(self):
        frame=torch.stack((self.u,self.v),dim=1)
        if (not torch.isfinite(frame).all() or not torch.allclose(frame.T@frame,torch.eye(2,dtype=frame.dtype,device=frame.device),atol=1e-12,rtol=0)
                or not torch.allclose(frame.sum(0),torch.zeros(2,dtype=frame.dtype,device=frame.device),atol=1e-12,rtol=0)):
            raise ValueError('plane must be unit orthogonal and lie in Euclidean 1-perp')

    @property
    def s(self):
        return self.s_raw*self.lambda_reference if self.variant=='S' else self.mass.new_zeros(())

    @property
    def d(self):
        return self.d_raw*self.lambda_reference if self.variant=='P' else self.mass.new_zeros(())

    def _stamp(self):
        return super()._stamp()+(self.variant,)

    def _new_context(self):
        kappa,lam=self.kappa,self.lambda_
        Q=lam*self.W+(self.B*(self.lambda_reference*self.a_bar))@self.B.T
        K=self.u[:,None]*self.v[None,:]-self.v[:,None]*self.u[None,:]
        P2=self.u[:,None]*self.u[None,:]+self.v[:,None]*self.v[None,:]
        symmetric=Q+self.d*P2
        skew=self.s*K
        return dict(kappa=kappa,lambda_=lam,coupling=self.reference_coupling45,
                    Q=Q,K=K,P2=P2,symmetric=symmetric,skew=skew,generator=symmetric+skew)

    def memory_matrix(self,delta=None):
        A=self._context()['generator']
        return A if delta is None else A.expand(self._tensor(delta).shape[:-1]+(10,10))

    def gate_values(self,delta):
        return self.mass.new_ones(self._tensor(delta).shape[:-1]+(45,))

    def effective_weights(self,delta):
        """Original graph contribution only; full operator is memory_matrix."""
        return (self.lambda_reference*self.a_bar).expand(self._tensor(delta).shape[:-1]+(45,))

    def memory_action(self,delta,r,context=None):
        c=self._context() if context is None else context
        return self._tensor(r)@c['generator'].T

    def field_reference(self,state,u=None):
        """Uncached equivalence branch; no numerical-method or physics change."""
        if u is not None:raise ValueError('no external ports')
        z=self._tensor(state)
        if z.shape[-1]!=40:raise ValueError('interface state width must be40')
        _,w,p,r=z.split(10,dim=-1);c=self._new_context()
        action=r@c['Q'].T
        if self.variant=='S':action=action+self.s*((r@self.v)[...,None]*self.u-(r@self.u)[...,None]*self.v)
        if self.variant=='P':action=action+self.d*((r@self.u)[...,None]*self.u+(r@self.v)[...,None]*self.v)
        return torch.cat((w,(self.base_momentum_field(z[...,:30],c)-r)/self.mass,
                          -self.jpi_p*w-self.rpp*self.storage_base*p,c['kappa']*w-action),-1)

    def matrices(self,state=None):
        """J/R are in [delta,pi=M*omega,p,r], never raw omega coordinates."""
        c=self._context();shape=(40,40) if state is None else self._tensor(state).shape[:-1]+(40,40)
        if state is not None and self._tensor(state).shape[-1]!=40:raise ValueError('state width must be40')
        J=self.mass.new_zeros(shape);R=J.clone();G=torch.diag(self.jpi_p)
        J[...,:10,10:20],J[...,10:20,:10]=self.W,-self.W
        J[...,10:20,20:30],J[...,20:30,10:20]=G,-G
        J[...,10:20,30:],J[...,30:,10:20]=-c['kappa']*self.W,c['kappa']*self.W
        J[...,30:,30:]=-c['kappa']*c['skew']
        R[...,10:20,10:20]=torch.diag(self.damping)
        R[...,20:30,20:30]=torch.diag(self.rpp)
        R[...,30:,30:]=c['kappa']*c['symmetric']
        return J,R

    def _linear_generator(self,context):
        # Parent assembly is in pi coordinates; replace its r block explicitly.
        result=super()._linear_generator(context)
        result[30:,30:]=-context['generator']
        return result

    def _history_context(self,context):
        raise ValueError('legacy symmetric force-graph history helper is disabled; use full-generator history_system')

    def fast_step(self,*args,**kwargs):
        raise ValueError('only registered true RK4 future integration')

    def history_system(self,history,*,history_dt=None,reference=False):
        dt=self.history_dt if history_dt is None else history_dt;self._history_dt(dt)
        h=self._tensor(history)
        if h.ndim not in (2,3) or h.shape[-2:]!=(11,30):raise ValueError('11 causal30-column rows required')
        if not torch.isfinite(h[...,:20]).all() or not torch.isfinite(h[...,-1,20:]).all():
            raise ValueError('finite delta/omega and terminal p required')
        c=self._new_context() if reference else self._context();omega=h[...,10:20]
        # Original exact p reverse block for piecewise-linear observed omega.
        slope=(omega[...,1:,:]-omega[...,:-1,:])/.01
        p=[h[...,-1,20:]]
        for i in range(9,-1,-1):
            p.append(torch.cat((p[-1],omega[...,i+1,:],slope[...,i,:]),-1)@self.p_backward.T)
        p=torch.stack(list(reversed(p)),-2)
        affine=self.mass.new_zeros(h.shape[:-2]+(10,11));affine[...,1:]=self.W
        states=[affine];substeps=round(.01/dt)
        def rhs(value,w):
            forcing=torch.cat((c['kappa']*w[...,None],torch.zeros_like(value[...,1:])), -1)
            return -(c['generator']@value)+forcing
        for i in range(9,-1,-1):
            right,left=omega[...,i+1,:],omega[...,i,:]
            for j in range(substeps):
                w0=right+(left-right)*(j/substeps);wm=right+(left-right)*((j+.5)/substeps)
                w1=right+(left-right)*((j+1)/substeps)
                a=rhs(affine,w0);b=rhs(affine-dt*a/2,wm)
                cc=rhs(affine-dt*b/2,wm);d=rhs(affine-dt*cc,w1)
                affine=affine-dt/6*(a+2*b+2*cc+d)
            states.append(affine)
        states=torch.stack(list(reversed(states)),-3);rbar,Zr=states[...,0],states[...,1:]
        C=(.02/6)*(Zr[...,0:10:2,:,:]+4*Zr[...,1:10:2,:,:]+Zr[...,2:11:2,:,:])
        design=(C*self.rstar[None,:]/(.02*self.rstar[:,None])).reshape(h.shape[:-2]+(50,10))
        fbase=self.base_momentum_field(torch.cat((h[...,:20],p),-1),c);f=fbase-rbar
        b=(.02/6)*(f[...,0:10:2,:]+4*f[...,1:10:2,:]+f[...,2:11:2,:])-self.mass*(omega[...,2:11:2,:]-omega[...,0:10:2,:])
        beta=(b/(.02*self.rstar)).flatten(-2)
        F=Zr[...,0,:,:]*self.rstar[None,:]/self.rstar[:,None];q=rbar[...,0,:]/self.rstar
        augmented=torch.cat((design,math.sqrt(RIDGE)*F),-2)
        target=torch.cat((beta,-math.sqrt(RIDGE)*q),-1)
        qr,tri=torch.linalg.qr(augmented,mode='reduced')
        coordinates=torch.linalg.solve_triangular(tri,(qr.transpose(-1,-2)@target[...,None]),upper=True).squeeze(-1)
        r0=self.rstar*coordinates;r=rbar+(Zr@r0[...,None,:,None]).squeeze(-1)
        return dict(r0=r0,u=r[...,0,:],s=coordinates,p_history=p,r_history=r,rbar=rbar,Zr=Zr,
            C=C,A=design,F=F,q=q,b=b,bscaled=beta,fbase=fbase,design=augmented,
            normal=augmented.transpose(-1,-2)@augmented,history_dt=dt)

    @torch.no_grad()
    def project_runtime_(self):
        self.project_nonnegative_()
        return self

    @torch.no_grad()
    def project_nonnegative_(self):
        super().project_nonnegative_()
        if self.variant=='P':self.d_raw.clamp_min_(0.)
        self.clear_cache();return self

    def complexity(self,state=None):
        return self.mass.new_tensor(float(sum(p.numel() for p in self.parameters())))

    def complexity_counts(self):
        return dict(super().complexity_counts(),raw_trainable_scalar_count=sum(p.numel() for p in self.parameters()),
            fixed_plane_stored_scalars=20,fixed_plane_used=self.variant!='A',
            variant=self.variant,plane_scope='fixed source plane; P may be redundant inside original graph cone')

    def export(self,path):
        if self._cache is not None:raise RuntimeError('close graph cache before export')
        self._validate_plane()
        if not torch.isfinite(self.kappa) or not torch.isfinite(self.lambda_) or self.kappa<=0 or self.lambda_<=0 or (self.a_bar<0).any() or self.variant=='P' and self.d_raw<0:
            raise ValueError('strict physical parameter domain required')
        payload=dict(format=FORMAT,mode='M',gate_mode='A',variant=self.variant,history_dt=self.history_dt,ridge=RIDGE,
            config=copy.deepcopy(self._source_config),source_identity=copy.deepcopy(self.source_identity),
            plane_identity=copy.deepcopy(self.plane_identity),plane=dict(u=self.u.detach().cpu().tolist(),v=self.v.detach().cpu().tolist()),
            state_dict={k:v.detach().cpu().tolist() for k,v in self.state_dict().items()},
            requires_grad={k:p.requires_grad for k,p in self.named_parameters()},counts=self.complexity_counts(),
            T_ref_seconds=1.,gamma='absent; fixed0',integration='true RK4',coordinate_interface='delta,omega,p,r')
        Path(path).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8');return payload

    @classmethod
    def from_export(cls,path):
        payload=json.loads(Path(path).read_text(encoding='utf-8'))
        if (payload.get('format')!=FORMAT or payload.get('mode')!='M' or payload.get('gate_mode')!='A'
                or payload.get('ridge')!=RIDGE or payload.get('T_ref_seconds')!=1.):
            raise ValueError('wrong transport model format')
        m=cls(payload['config'],payload['variant'],u=payload['plane']['u'],v=payload['plane']['v'],
              history_dt=payload['history_dt'],source_identity=payload['source_identity'],plane_identity=payload['plane_identity'])
        template=m.state_dict();supplied=payload['state_dict']
        if set(template)!=set(supplied):raise ValueError('transport export keys differ')
        values={k:torch.as_tensor(supplied[k],dtype=v.dtype) for k,v in template.items()}
        if any(values[k].shape!=v.shape or not torch.isfinite(values[k]).all() for k,v in template.items()):
            raise ValueError('invalid exported shape/value')
        buffers=dict(m.named_buffers())
        if any(not torch.equal(values[k],template[k]) for k in buffers):raise ValueError('fixed export buffers/config/plane differ')
        m.load_state_dict(values);m._validate_plane()
        if (m.a_bar<0).any() or m.variant=='P' and m.d_raw<0 or not torch.isfinite(m.kappa) or not torch.isfinite(m.lambda_) or m.kappa<=0 or m.lambda_<=0:
            raise ValueError('invalid restored parameter domain')
        params=dict(m.named_parameters())
        if set(payload['requires_grad'])!=set(params):raise ValueError('transport gradient layout differs')
        for name,p in params.items():
            if type(payload['requires_grad'][name]) is not bool:raise ValueError('requires_grad must be boolean')
            p.requires_grad_(payload['requires_grad'][name])
        return m

    @classmethod
    def from_a_export(cls,path,variant,*,u,v,expected_sha256,history_dt=.0025,plane_identity=None):
        path=Path(path);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('frozen A96 source changed')
        parent=StateGatedMemoryROM.from_export(path)
        if parent.gate_mode!='A' or parent.mode!='M':raise ValueError('source must be frozen A')
        identity=dict(source_sha256=expected_sha256,parent=copy.deepcopy(parent.source_identity),ancestry='N_M192 -> SGM A96')
        m=cls(parent._source_config,variant,u=u,v=v,history_dt=history_dt,source_identity=identity,plane_identity=plane_identity)
        values=m.state_dict();parent_values=parent.state_dict()
        if set(parent_values)!=set(values)-{'u','v','s_raw','d_raw'}:raise ValueError('source A parameter/buffer layout differs')
        values.update(parent_values);m.load_state_dict(values);return m

    @classmethod
    def from_force_graph_export(cls,*args,**kwargs):
        raise ValueError('use from_a_export with frozen A96 identity and plane')

    @classmethod
    def from_source_export(cls,*args,**kwargs):
        raise ValueError('use from_a_export with frozen A96 identity and plane')
