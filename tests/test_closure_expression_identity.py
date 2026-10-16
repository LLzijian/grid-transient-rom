import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
from experiments.closure_expression_identity import mechanism_signature,distinct_mechanisms,compact_factors


def v(name):return {'op':'var','name':name}
def mul(*args):return {'op':'mul','args':list(args)}
def term(channel,expression):return {'channel':channel,'expression':expression}


def test_polynomial_and_edge_symmetry_identities_are_not_new_mechanisms():
    a=term('node',mul(v('p'),v('q'),v('q')))
    b=term('node',mul(v('p'),{'op':'pow2','arg':v('q')}))
    assert mechanism_signature(a)==mechanism_signature(b)
    edge1=term('edge',mul(v('qi'),v('s')))
    edge2=term('edge',mul({'op':'add','args':[v('qi'),mul({'op':'const','value':-1},v('qj'))]},v('s')))
    assert mechanism_signature(edge1)==mechanism_signature(edge2)
    assert not distinct_mechanisms([edge1,edge2])


def test_physical_angle_cosine_survives_centering_but_linear_storage_vanishes():
    assert mechanism_signature(term('edge',v('c'))) is not None
    assert mechanism_signature(term('edge',v('s'))) is None
    assert mechanism_signature(term('node',v('p'))) is None
    assert mechanism_signature(term('edge',v('qi'))) is None


def test_factor_compaction_agrees_in_model_and_search_canonicalization():
    from experiments.symbolic_closure_rom import expression_key
    from experiments.closure_discovery_support import expression_key as search_key
    a=mul(v('p'),v('q'),v('q'));b=mul(v('p'),{'op':'pow2','arg':v('q')})
    assert expression_key(a)==expression_key(b)==search_key(a)==search_key(b)
