"""Initial scalar-function compression engine (not full formula-search automation).
Polynomial students have analytic gradients and JSON exports. Replacing each scalar
module preserves the outer H/J/R construction, but MUST be followed by truth-rollout
fine-tuning and numerical evaluation. No compression success is asserted by fitting H alone.
"""
from itertools import combinations_with_replacement
import numpy as np
import torch
from torch import nn

def powers_library(inputs,degree=3,max_candidates=256):
    powers=[np.zeros(inputs,dtype=int)]
    for degree_now in range(1,degree+1):
        combos=list(combinations_with_replacement(range(inputs),degree_now))
        # Keep all lower-degree atoms first; bounded diagonal cubic terms are next.
        if degree_now>=3:combos.sort(key=lambda c:(len(set(c)),c))
        for comb in combos:
            p=np.zeros(inputs,dtype=int)
            for i in comb:p[i]+=1
            powers.append(p)
            if len(powers)>=max_candidates:return np.stack(powers)
    return np.stack(powers)

class PolynomialFunction(nn.Module):
    def __init__(self,powers,coefficients):
        super().__init__();self.register_buffer('powers',torch.as_tensor(powers,dtype=torch.long))
        self.coefficients=nn.Parameter(torch.as_tensor(coefficients,dtype=torch.float64))
        self.register_buffer('support',self.coefficients.detach()!=0)
        self.inputs=self.powers.shape[1];self.outputs=self.coefficients.shape[1]
        self.quadratic_execution=False  # Runtime only; never stored in state_dict.
        self._quadratic_trace_mode=None
        self._compile_factors()
        self.register_load_state_dict_post_hook(self._recompile_after_load)
    def _compile_factors(self):
        """Compile only nonzero exponents; transient buffers follow cpu/cuda moves."""
        powers=self.powers.detach().cpu().numpy()
        if powers.ndim!=2 or np.any(powers<0):raise ValueError('Expected nonnegative polynomial powers')
        def factors(pw):
            degrees=pw.sum(-1);width=max(1,int(degrees.max()) if len(degrees) else 0)
            indices=np.zeros((len(pw),width),dtype=np.int64)
            mask=np.zeros((len(pw),width),dtype=bool)
            for row,p in enumerate(pw):
                variables=np.repeat(np.arange(self.inputs,dtype=np.int64),p)
                indices[row,:len(variables)]=variables;mask[row,:len(variables)]=True
            return indices,mask
        def buffer(name,value):
            self.register_buffer(name,torch.as_tensor(value,device=self.powers.device),persistent=False)
        indices,mask=factors(powers)
        buffer('_factor_indices',indices);buffer('_factor_mask',mask)
        for d in range(self.inputs):
            rows=np.flatnonzero(powers[:,d]>0)
            derivative=powers[rows].copy();multipliers=derivative[:,d].copy()
            derivative[:,d]-=1
            indices,mask=factors(derivative)
            buffer(f'_derivative_rows_{d}',rows)
            buffer(f'_derivative_indices_{d}',indices)
            buffer(f'_derivative_mask_{d}',mask)
            buffer(f'_derivative_multipliers_{d}',multipliers)
        self._compile_quadratic_layout(powers)
    def _compile_quadratic_layout(self,powers):
        """Integer placement only; no floating coefficient aggregation/cache."""
        self._quadratic_rejection=None
        if np.any(powers.sum(-1)>2):
            self._quadratic_rejection='Quadratic execution requires total degree <=2'
        elif len(np.unique(powers,axis=0))!=len(powers):
            self._quadratic_rejection='Quadratic execution requires unique monomial powers'
        slots=[]
        if self._quadratic_rejection is None:
            for power in powers:
                variables=np.repeat(np.arange(self.inputs),power)
                if len(variables)==0:slot=0
                elif len(variables)==1:slot=1+int(variables[0])
                else:slot=1+self.inputs+int(variables[0])*self.inputs+int(variables[1])
                slots.append(slot)
        self.register_buffer('_quadratic_slots',torch.as_tensor(slots,dtype=torch.long,
                                                               device=self.powers.device),persistent=False)
    def set_quadratic_execution(self,enabled=True):
        """Opt in before trace_field(); use a fresh model to change a traced mode.

        Upper-triangular Q keeps each mixed coefficient intact (no division by
        two, including subnormals). This changes floating-point reduction and
        multiplication order, so it promises algebraic equivalence, not bitwise
        equality, identical overflow/underflow boundaries or deployment speed.
        """
        enabled=bool(enabled)
        if enabled and self._quadratic_rejection is not None:
            raise ValueError(self._quadratic_rejection)
        if self._quadratic_trace_mode is not None and enabled!=self._quadratic_trace_mode:
            raise ValueError('Choose quadratic execution before tracing; use a fresh model to change mode')
        self.quadratic_execution=enabled
        return self
    def _use_quadratic(self):
        mode=bool(self.quadratic_execution)
        if mode and self._quadratic_rejection is not None:
            raise ValueError(self._quadratic_rejection)
        if torch.jit.is_tracing():self._quadratic_trace_mode=mode
        return mode
    def _quadratic_value_gradient(self,x,gradient):
        # Unique integer slots make index_copy a differentiable rearrangement.
        # Rebuild from current masked coefficients at every call/optimizer step;
        # the original Parameter object and its data_ptr remain untouched.
        coefficients=self.coefficients*self.support
        size=1+self.inputs+self.inputs*self.inputs
        packed=coefficients.new_zeros((size,self.outputs)).index_copy(0,self._quadratic_slots,coefficients)
        constant=packed[0];linear=packed[1:1+self.inputs]
        quadratic=packed[1+self.inputs:].reshape(self.inputs,self.inputs,self.outputs)
        qx=torch.einsum('...j,ijo->...io',x,quadratic)
        value=constant+x@linear+(x.unsqueeze(-1)*qx).sum(-2)
        if not gradient:return value,None
        qtx=torch.einsum('...i,ijo->...jo',x,quadratic)
        return value,(qx+qtx).transpose(-1,-2)+linear.T
    @staticmethod
    def _recompile_after_load(module,incompatible_keys):
        # Same-shaped powers can contain different monomials in a checkpoint.
        # Rebuild after copying persistent buffers, including assign=True loads.
        module._compile_factors()
    @staticmethod
    def _factor_product(x,indices,mask):
        factors=torch.where(mask,x[...,indices],1.0)
        atoms=factors[...,0]
        # Degree is normally <=3. Explicit products retain exact derivatives at
        # zero, without division by a variable or zero-sensitive prod backward.
        for position in range(1,factors.shape[-1]):atoms=atoms*factors[...,position]
        return atoms
    def forward(self,x):
        if self._use_quadratic():return self._quadratic_value_gradient(x,False)[0]
        atoms=self._factor_product(x,self._factor_indices,self._factor_mask)
        return atoms@(self.coefficients*self.support)
    def value_gradient(self,x):
        if self._use_quadratic():return self._quadratic_value_gradient(x,True)
        coefficients=self.coefficients*self.support
        value=self._factor_product(x,self._factor_indices,self._factor_mask)@coefficients
        grad=[]
        for d in range(self.inputs):
            rows=getattr(self,f'_derivative_rows_{d}')
            if rows.numel()==0:
                grad.append(value*0.0);continue
            atoms=self._factor_product(x,getattr(self,f'_derivative_indices_{d}'),
                                         getattr(self,f'_derivative_mask_{d}'))
            atoms=atoms*getattr(self,f'_derivative_multipliers_{d}')
            grad.append(atoms@coefficients[rows])
        return value,torch.stack(grad,-1)
    def export(self):
        return {'type':'polynomial','powers':self.powers.cpu().tolist(),'coefficients':(self.coefficients*self.support).detach().cpu().tolist(),
                'inputs':self.inputs,'outputs':self.outputs,'nonzero_per_output':((self.coefficients.detach()*self.support).abs()>0).sum(0).cpu().tolist()}

def numpy_value_gradient(spec,x):
    if spec['type']!='polynomial':raise ValueError('Unsupported expression node')
    x=np.asarray(x,dtype=float);powers=np.asarray(spec['powers'],int);coef=np.asarray(spec['coefficients'],float)
    atoms=np.prod(x[...,None,:]**powers,axis=-1);value=atoms@coef
    grad=[]
    for d in range(powers.shape[1]):
        pd=powers.copy();pd[:,d]=np.maximum(0,pd[:,d]-1)
        terms=powers[:,d]*np.prod(x[...,None,:]**pd,axis=-1)
        grad.append(terms@coef)
    return value,np.stack(grad,axis=-1)

def fit_function(teacher,x,max_terms=16,degree=3,ridge=1e-6,gradient_weight=.1,steps=200):
    """TRAIN-only samples are caller's responsibility; no full-RHS distillation here."""
    x=x.detach();teacher.eval();device=x.device
    pw=powers_library(x.shape[-1],degree);powers=torch.as_tensor(pw,device=device)
    with torch.no_grad():
        y=teacher(x);A=(x[:,None,:]**powers).prod(-1)
        scale=torch.linalg.vector_norm(A,dim=0).clamp_min(1e-8)
        An=A/scale
        coeff=torch.linalg.solve(An.T@An+ridge*torch.eye(len(pw),dtype=x.dtype,device=device),An.T@y)/scale[:,None]
        contribution=coeff.abs()*scale[:,None];mask=torch.zeros_like(coeff)
        for o in range(y.shape[-1]):mask[contribution[:,o].topk(min(max_terms,len(pw))).indices,o]=1
    student=PolynomialFunction(pw,(coeff*mask).cpu()).to(device);optim=torch.optim.Adam(student.parameters(),lr=1e-3)
    teacher_x=x.clone().requires_grad_(True);ty=teacher(teacher_x)
    tg=torch.stack([torch.autograd.grad(ty[:,o].sum(),teacher_x,retain_graph=True)[0].detach() for o in range(ty.shape[-1])],-2)
    ty=ty.detach()
    for _ in range(steps):
        xx=x.clone().requires_grad_(True);sy=student(xx)
        sg=torch.stack([torch.autograd.grad(sy[:,o].sum(),xx,create_graph=True,retain_graph=True)[0] for o in range(sy.shape[-1])],-2)
        loss=(sy-ty).square().mean()+gradient_weight*(sg-tg).square().mean()
        optim.zero_grad();loss.backward();student.coefficients.grad.mul_(mask);optim.step()
        with torch.no_grad():student.coefficients.mul_(mask)
    return student,{'value_mse':float((student(x).detach()-ty).square().mean()),'note':'Function-level fit only; no autonomous acceptance or full symbolic completion'}

def fit_sparse_hermite(teacher,x,max_terms=16,degree=3,ridge=1e-6,gradient_weight=.1):
    """Finite library, value+gradient regression with support selection on TRAIN.

    The last fifth of caller-provided TRAIN samples is an inner fitting check,
    never development data. Fixed support is refitted by a linear solve and
    persisted in PolynomialFunction for subsequent joint truth fine-tuning.
    """
    xn=x.detach().cpu().numpy();pw=powers_library(x.shape[-1],degree)
    with torch.no_grad():ty,tg=teacher.value_gradient(x)
    yn=ty.detach().cpu().numpy();gn=tg.detach().cpu().numpy()
    n,d=xn.shape;cut=max(1,int(.8*n));outputs=yn.shape[1]
    atoms=np.prod(xn[:,None,:]**pw,axis=-1)
    derivatives=[]
    for dim in range(d):
        pp=pw.copy();pp[:,dim]=np.maximum(0,pp[:,dim]-1)
        derivatives.append(pw[:,dim]*np.prod(xn[:,None,:]**pp,axis=-1))
    factor=np.sqrt(gradient_weight/max(1,d))
    aa=np.concatenate([atoms[:cut]]+[factor*g[:cut] for g in derivatives])
    yy=np.concatenate([yn[:cut]]+[factor*gn[:cut,:,j] for j in range(d)])
    scales=np.maximum(np.linalg.norm(aa,axis=0),1e-12);an=aa/scales
    full=np.linalg.solve(an.T@an+ridge*np.eye(an.shape[1]),an.T@yy)
    coef=np.zeros_like(full)
    for out in range(outputs):
        selected=np.argsort(np.abs(full[:,out]))[-min(max_terms,len(pw)):]
        narrow=an[:,selected]
        values=np.linalg.solve(narrow.T@narrow+ridge*np.eye(len(selected)),narrow.T@yy[:,out])
        coef[selected,out]=values/scales[selected]
    use=np.any(coef!=0,axis=1);student=PolynomialFunction(pw[use],coef[use]).to(x.device)
    with torch.no_grad():sy,sg=student.value_gradient(x)
    vy=(sy.detach().cpu().numpy()-yn)**2;vg=(sg.detach().cpu().numpy()-gn)**2
    return student,{'degree':degree,'library_atoms':len(pw),'selected_union_atoms':int(use.sum()),
        'nonzero_per_output':student.export()['nonzero_per_output'],'inner_train_samples':cut,'inner_check_samples':n-cut,
        'train_value_mse':float(vy[:cut].mean()),'train_gradient_mse':float(vg[:cut].mean()),
        'inner_check_value_mse':float(vy[cut:].mean()) if cut<n else None,
        'inner_check_gradient_mse':float(vg[cut:].mean()) if cut<n else None,
        'gradient_weight':gradient_weight,'ridge':ridge,'source':'caller-supplied TRAIN teacher samples; no FOM truth invented'}
