"""Explicit exact edge-polynomial backend; historical runtime remains unchanged."""
from __future__ import annotations
import copy
from fractions import Fraction
import torch
from closure_training_runtime import TrainingClosureROM

VARIABLES=('pi','pj','qi','qj','s','c')
LIMIT=256


def _add(a,b):
    out=dict(a)
    for k,v in b.items():out[k]=out.get(k,Fraction(0))+v
    return {k:v for k,v in out.items() if v}
def _mul(a,b):
    out={}
    for x,u in a.items():
        for y,v in b.items():
            k=tuple(i+j for i,j in zip(x,y));out[k]=out.get(k,Fraction(0))+u*v
            if len(out)>LIMIT:raise ValueError('polynomial expansion exceeds limit')
    return {k:v for k,v in out.items() if v}
def _expand(ast):
    op=ast['op'];zero=(0,)*6
    if op=='var':return {tuple(int(i==VARIABLES.index(ast['name'])) for i in range(6)):Fraction(1)}
    if op=='const':return {zero:Fraction(ast['value'])}
    if op in ('add','mul'):
        out={} if op=='add' else {zero:Fraction(1)}
        for child in ast['args']:
            out=(_add if op=='add' else _mul)(out,_expand(child))
            if len(out)>LIMIT:raise ValueError('polynomial expansion exceeds limit')
        return out
    if op not in ('pow2','pow3','pow4'):raise ValueError('unsupported polynomial operator')
    out={zero:Fraction(1)};base=_expand(ast['arg'])
    for _ in range(int(op[-1])):out=_mul(out,base)
    return out
def compile_polynomial(term):
    """Exact endpoint symmetry and feature centering; physical c tangent separate."""
    if term['channel']!='edge':raise ValueError('whole-field fallback for mixed channels')
    raw=_expand(term['expression']);reverse={}
    for k,v in raw.items():
        r=(k[1],k[0],k[3],k[2],k[4],k[5])
        reverse[r]=reverse.get(r,Fraction(0))+v*(-1 if k[4]%2 else 1)
    symmetric={k:v/2 for k,v in _add(raw,reverse).items()}
    slope=symmetric.get((0,0,0,0,0,1),Fraction(0))
    centered={k:v for k,v in symmetric.items() if sum(k)>1}
    derivatives=[]
    for axis in range(6):
        d={}
        for k,v in centered.items():
            if k[axis]:
                lowered=list(k);lowered[axis]-=1;d[tuple(lowered)]=v*k[axis]
        derivatives.append(d)
    return centered,derivatives,slope


class EdgePolynomialROM(TrainingClosureROM):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._compile_tables()

    def _compile_tables(self):
        self._poly_enabled=False;self._poly_reason=None;self._poly_device=None;self._poly_graph_weights=None
        try:
            if self.structured.storage_specs or not self.terms:raise ValueError('unsupported storage or empty closure')
            specs=[compile_polynomial(t) for t in self.terms]
            basis=sorted(set().union(*(set(v)|set().union(*(set(d) for d in ds)) for v,ds,_ in specs)))
            if len(basis)>LIMIT:raise ValueError('shared derivative basis exceeds256')
            # Constant-zero centered mechanisms still need an explicit zero column.
            if not basis:basis=[(0,)*6]
            lookup={k:i for i,k in enumerate(basis)}
            table=[[[0.]*7 for _ in specs] for _ in basis]
            for j,(values,derivatives,_) in enumerate(specs):
                for axis,mapping in enumerate([values]+derivatives):
                    for k,v in mapping.items():table[lookup[k]][j][axis]=float(v)
            self._poly_basis=tuple(basis);self._poly_table=table
            self._poly_used_axes=tuple(i for i in range(6) if any(k[i] for k in basis))
            self._poly_slopes=tuple(float(s[2]) for s in specs)
            self._poly_enabled=True
        except (ValueError,KeyError,TypeError,OverflowError) as error:self._poly_reason=str(error)
        self._clear_runtime()

    @property
    def polynomial_enabled(self):return self._poly_enabled

    def _clear_runtime(self):
        super()._clear_runtime();self._poly_graph_weights=None;self._poly_graph_directions=None

    def _apply(self,fn,recurse=True):
        self._poly_device=None
        return super()._apply(fn,recurse=recurse)

    def load_state_dict(self,*args,**kwargs):
        self._poly_device=None
        return super().load_state_dict(*args,**kwargs)

    def _tables(self):
        if self._poly_device is None:
            table=self.mass.new_tensor(self._poly_table)
            slopes=self.mass.new_tensor(self._poly_slopes)
            self._poly_device=(table,slopes)
        return self._poly_device

    def _weights(self):
        table,slopes=self._tables()
        return torch.einsum('ktv,t->kv',table[...,1:],self.coefficients),slopes@self.coefficients

    def begin_graph_cache(self):
        super().begin_graph_cache()
        if self._poly_enabled:
            self._poly_graph_weights=self._weights()
            self._poly_graph_directions=(self.base_D.clamp_min(0),(-self.base_D).clamp_min(0))

    def _monomials(self,features):
        # Powers use products rather than value/feature or tensor pow at zero.
        # Each feature's powers are shared by every value/derivative monomial.
        columns=[]
        for axis in self._poly_used_axes:
            x=features[...,axis];powers=[torch.ones_like(x),x]
            for degree in range(2,max(k[axis] for k in self._poly_basis)+1):powers.append(powers[-1]*x)
            columns.append(torch.stack([powers[k[axis]] for k in self._poly_basis],-1))
        if not columns:return features.new_ones(features.shape[:-1]+(len(self._poly_basis),))
        result=columns[0]
        for column in columns[1:]:result=result*column
        return result

    def feature_value_gradient(self,features):
        """Centered feature polynomial, before the separate physical-c correction."""
        if not self._poly_enabled:raise ValueError('unsupported polynomial backend')
        table,_=self._tables()
        values=torch.einsum('...k,ktv->...tv',self._monomials(features),table)
        return values[...,0],values[...,1:]

    def field(self,z,u=None):
        if not self._poly_enabled:return super().field(z,u)
        z=torch.as_tensor(z,dtype=self.mass.dtype,device=self.mass.device)
        cache=getattr(self,'_runtime_cache',None)
        A,coupling,storage,quadratic,sin0,cos0,absD,weights=cache if cache is not None else self._constants()
        d,w,p,q=z.split(10,-1);eta=d@self.base_D.T;angle=self.phi+eta
        sin,cos=angle.sin(),angle.cos();s,c=sin-sin0,cos-cos0
        positive,negative=(self._poly_graph_directions if self._poly_graph_directions is not None
                           else (self.base_D.clamp_min(0),(-self.base_D).clamp_min(0)))
        pn=p/self.p_scale
        features=torch.stack((pn@positive.T,pn@negative.T,q@positive.T,q@negative.T,s,c),-1)
        derivative_weights,slope=self._poly_graph_weights if self._poly_graph_weights is not None else self._weights()
        g=(self._monomials(features)@derivative_weights)*weights[...,None]
        gd=((g[...,4]*cos-g[...,5]*sin)+weights*slope*(-sin+sin0))@self.base_D
        gd=gd+(s*coupling)@self.base_D
        gp=storage[:10]*p+(g[...,0]@positive+g[...,1]@negative)/self.p_scale
        gq=(storage[10:]+quadratic)*q+g[...,2]@positive+g[...,3]@negative
        gradient=torch.cat((gd,w,gp,gq),-1)
        pv=torch.zeros_like(p) if u is None else torch.as_tensor(u,dtype=z.dtype,device=z.device)
        if pv.shape[-1]!=10:raise ValueError('external port input must have width10')
        f=gradient@A
        return torch.cat((f[...,:10],(f[...,10:20]+pv)/self.mass,f[...,20:]),-1)

    def with_terms(self,terms):
        if getattr(self,'_runtime_cache',None) is not None:raise RuntimeError('close graph before changing terms')
        result=super().with_terms(terms);result._compile_tables();return result

    def _export_payload(self):
        if getattr(self,'_runtime_cache',None) is not None:raise RuntimeError('close graph before export')
        self._clear_runtime()
        return super()._export_payload()


def adapt_base(reference):
    """Independent parameter objects; convert before constructing an optimizer."""
    if getattr(reference,'_runtime_cache',None) is not None:raise RuntimeError('close graph before conversion')
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        # Historical complexity metadata uses CPU-only NumPy access. Export an
        # independent CPU snapshot; never migrate the caller's model/optimizer.
        snapshot=copy.deepcopy(reference).cpu()
        payload=snapshot._export_payload()
        result=EdgePolynomialROM.from_config(payload['config'],terms=payload['terms'])
        result.load_state_dict(snapshot.state_dict())
        flags={n:p.requires_grad for n,p in reference.named_parameters()}
        for n,p in result.named_parameters():p.requires_grad_(flags[n])
        result.provenance=copy.deepcopy(reference.provenance)
        return result.to(device=reference.mass.device,dtype=reference.mass.dtype)
    finally:
        torch.set_rng_state(cpu)
        if cuda:torch.cuda.set_rng_state_all(cuda)


def adapt_wrapper(reference):
    result=copy.deepcopy(reference);result.base=adapt_base(reference.base);return result


def make_reference(term,seed,obs):
    from closure_causal_recovery_20261019 import runtime_model
    from closure_discovery_benchmarks import toy_source
    from closure_causal_history_residual_20261019 import CausalHistoryResidual,make_optimizer
    model=CausalHistoryResidual.from_train(runtime_model(toy_source(),[term],seed,'cuda'),torch.as_tensor(obs[:,:11],device='cuda'))
    return model,make_optimizer(model)


def make(term,seed,obs):
    from closure_causal_history_residual_20261019 import CausalHistoryResidual,make_optimizer
    from closure_causal_recovery_20261019 import runtime_model
    from closure_discovery_benchmarks import toy_source
    reference=CausalHistoryResidual.from_train(runtime_model(toy_source(),[term],seed,'cuda'),torch.as_tensor(obs[:,:11],device='cuda'))
    model=adapt_wrapper(reference)
    return model,make_optimizer(model)
