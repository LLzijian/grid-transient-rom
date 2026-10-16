"""Exact identities for the small closure grammar, before numerical fitting.

Polynomial expansion is used only for identity checks. Runtime formulas retain
their bounded trees. Port tanh subexpressions remain opaque atoms.
"""
from fractions import Fraction
import functools
import json


def compact_factors(args):
    """Give repeated factors one stable power representation."""
    groups={}
    for child in args:
        base=child['arg'] if child['op'] in ('pow2','pow3','pow4') else child
        count=int(child['op'][-1]) if child['op'] in ('pow2','pow3','pow4') else 1
        key=json.dumps(base,sort_keys=True,separators=(',',':'))
        if key not in groups:groups[key]=[base,0,[]]
        groups[key][1]+=count;groups[key][2].append(child)
    result=[]
    for base,count,originals in groups.values():
        result.extend([{'op':f'pow{count}','arg':base}] if 2<=count<=4 else [base] if count==1 else originals)
    return result


def _add(a,b):
    result=dict(a)
    for key,value in b.items():result[key]=result.get(key,Fraction(0))+value
    return {k:v for k,v in result.items() if v}


def _mul(a,b):
    result={}
    for x,u in a.items():
        for y,v in b.items():
            key=tuple(sorted(x+y));result[key]=result.get(key,Fraction(0))+u*v
    return {k:v for k,v in result.items() if v}


def _polynomial(ast):
    op=ast['op']
    if op=='var':return {(ast['name'],):Fraction(1)}
    if op=='const':return {():Fraction(ast['value'])}
    if op=='tanh':return {('tanh:'+json.dumps(ast,sort_keys=True,separators=(',',':')),):Fraction(1)}
    if op in ('add','mul'):
        value={} if op=='add' else {():Fraction(1)}
        for child in ast['args']:value=(_add if op=='add' else _mul)(value,_polynomial(child))
        return value
    result={():Fraction(1)};base=_polynomial(ast['arg'])
    for _ in range(int(op[-1])):result=_mul(result,base)
    return result


@functools.lru_cache(maxsize=4096)
def _signature(serialized):
    term=json.loads(serialized);channel=term['channel'];p=_polynomial(term['expression'])
    if channel=='edge':
        reverse={};swap={'pi':'pj','pj':'pi','qi':'qj','qj':'qi'}
        for monomial,value in p.items():
            key=tuple(sorted(swap.get(v,v) for v in monomial))
            reverse[key]=reverse.get(key,Fraction(0))+value*(-1 if monomial.count('s')%2 else 1)
        p={k:v/2 for k,v in _add(p,reverse).items()}
    # c is nonlinear in physical angle: its physical tangent subtraction
    # preserves a nonzero curvature. Node/edge p and q linear terms vanish.
    p={k:v for k,v in p.items() if k and (channel=='port' or len(k)>1 or k==('c',))}
    if not p:return None
    rows=sorted(p.items());scale=rows[0][1]
    return channel,tuple((k,str(v/scale)) for k,v in rows)


def mechanism_signature(term):
    """Centered, edge-symmetrized identity up to a nonzero scalar coefficient."""
    return _signature(json.dumps(term,sort_keys=True,separators=(',',':')))


def distinct_mechanisms(terms):
    signatures=[mechanism_signature(t) for t in terms]
    return None not in signatures and len(set(signatures))==len(signatures)
