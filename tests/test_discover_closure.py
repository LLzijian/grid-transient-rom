"""Integration contracts beyond isolated AST/helper unit tests."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import discover_closure as runner
from closure_discovery_benchmarks import toy_source
from symbolic_closure_rom import SymbolicClosureROM
from closure_discovery_support import propose_mutations


def test_two_parent_library_and_generated_actually_differ():
    source=toy_source()
    terms=[{'channel':'port','expression':{'op':'var','name':'q'}}]
    pools={mode:propose_mutations([terms,terms],mode=mode,max_candidates=64) for mode in ('library','generated')}
    assert {r['parent'] for r in pools['generated']}=={0,1}
    old={str(t) for t in source.terms}
    assert any(any(t['expression']['op'] in ('tanh','pow2','pow3','pow4') for t in r['terms']) for r in pools['generated'])
    assert not any(any(t['expression']['op']=='tanh' for t in r['terms']) for r in pools['library'])


def test_generated_expressions_depend_on_parent_subtrees():
    from closure_discovery_support import _candidate_terms,term_key
    p={'op':'var','name':'p'};q={'op':'var','name':'q'}
    parent={'channel':'node','expression':{'op':'mul','args':[{'op':'pow2','arg':p},q]}}
    target={'channel':'node','expression':{'op':'mul','args':[{'op':'pow2','arg':p},{'op':'pow2','arg':q}]}}
    assert term_key(target) in {term_key(t) for t in _candidate_terms([parent])}
    assert term_key(target) not in {term_key(t) for t in _candidate_terms([])}


def test_direction_change_is_whole_exchange_or_psd_group():
    source=toy_source();mask=source.template_mask[3:7].tolist();mask[3]=1.
    m=runner.proposal_model(source,{'terms':source.terms,'direction_mask':mask})
    J,R=m.matrices()
    assert torch.max(torch.abs(J+J.T))<1e-12
    assert torch.linalg.eigvalsh(R).min()>-1e-12
    assert source.template_mask[6]==0


def test_model_optimizer_groups_keep_old_rate_semantics():
    m=toy_source().with_terms([{'channel':'port','expression':{'op':'var','name':'q'}}])
    opt=runner.optimizer(m)
    named=dict(m.named_parameters())
    groups={id(p):g['block'] for g in opt.param_groups for p in g['params']}
    assert groups[id(named['coefficients'])]=='closure'
    assert groups[id(named['_template_scalars.3'])]=='closure'
    assert groups[id(named['_q_quadratic_raw'])]=='skeleton'


def test_checkpoint_resumes_optimizer_and_rng_without_data(tmp_path):
    m=toy_source();opt=runner.optimizer(m);rng=np.random.default_rng(1701)
    trace=[{'step':1,'loss':1.}]
    runner.checkpoint(tmp_path,m,opt,rng,1,trace)
    restored=SymbolicClosureROM.from_export(tmp_path/'model.json')
    x=torch.randn((2,40),dtype=torch.float64)*.02
    assert torch.allclose(m.field(x),restored.field(x),atol=1e-12,rtol=0)
    saved=torch.load(tmp_path/'training.pt',weights_only=False)
    check=np.random.default_rng();check.bit_generator.state=saved['numpy_rng']
    assert check.integers(100000)==rng.integers(100000)


def test_primary_comparison_refuses_invalid_event():
    from closure_discovery_confirmation import paired_primary
    row={'id':'e','group':'g','label':'stable','complete_5s':True,'numerical_pass':False,'mse':None}
    assert paired_primary({'records':[row]},{'records':[row]})['status']=='unresolved'


def test_output_boolean_stays_boolean():
    assert runner.safe({'test_read':False})['test_read'] is False


def test_full_refit_schedule_exceeds_historical_800_cap():
    assert runner.horizon_for_update(100)==2.
    assert [runner.horizon_for_update(i) for i in range(101,104)]==[1.,2.,5.]
    assert runner.horizon_for_update(801) in (1.,2.,5.)
    assert runner.horizon_for_update(1024) in (1.,2.,5.)
