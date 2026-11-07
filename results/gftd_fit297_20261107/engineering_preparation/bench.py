"""Deterministic RHS engineering benchmark; no trajectory, optimizer or data loading."""
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from kiki.physics.kuramoto_functional import (GeneratingFunctionalKuramoto, StructuralGates,
    bregman_kuramoto_potential,bregman_kuramoto_force,latent_coupling_modulation)

OUT=Path(__file__).resolve().parent


class VectorGradient(GeneratingFunctionalKuramoto):
    def gradients(self,state):
        theta,omega,p,a,r=self.split_state(state)
        phi=self.incidence.T@theta
        psi=bregman_kuramoto_potential(phi,self.eta_edge)
        mod,_,_=latent_coupling_modulation(a,self.edges,self.gamma_amp,self.gates)
        active=self.gates.s_edge*self.k_edge
        with np.errstate(over='raise',under='raise',invalid='raise'):
            exponential=np.exp(np.abs(self.incidence).T@(self.gamma_amp*a))
        ga=self.lambda_amp*a+self.gamma_amp*(np.abs(self.incidence)@(active*self.gates.s_mod*exponential*psi))
        return (self.incidence@(active*mod*bregman_kuramoto_force(phi,self.eta_edge)),
                self.mass*omega,self.storage_p*p,ga,self.lambda_mem*r)


class ConstantGate(GeneratingFunctionalKuramoto):
    def dissipative_operators(self,state):
        if self.gate_weight_amp is None and self.gate_weight_mem is None:
            self.split_state(state)
            return 1./self.tau_amp,np.diag(1./self.tau_mem_sync)
        return super().dissipative_operators(state)


class Combined(VectorGradient,ConstantGate):
    pass


def fixture(n,mode,cls=GeneratingFunctionalKuramoto):
    e=n*(n-1)//2
    return cls(mass=np.linspace(1.,2.,n),damping=np.full(n,.2),k_edge=np.linspace(.3,.9,e),
        eta_edge=np.linspace(-.05,.05,e),gamma_amp=np.full(n,.5),lambda_amp=np.linspace(.8,1.2,n),
        lambda_mem=np.linspace(.7,1.3,n),tau_amp=np.full(n,5.),tau_mem_sync=np.full(n,2.),
        tau_mem_coi=2.,j_omega_mem=np.full(n,.7),storage_p=np.ones(n),gain_p=np.ones(n),rate_p=np.full(n,.3),
        gates=StructuralGates(np.ones(e),np.full(e,.5 if mode=='modulated' else 0.),np.ones(n)))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(f,state):
    for _ in range(10):f.rhs(state)
    begin=time.perf_counter()
    for _ in range(1000):f.rhs(state)
    seconds=time.perf_counter()-begin
    return dict(warmup_rhs_calls=10,timed_rhs_calls=1000,seconds=seconds,seconds_per_rhs=seconds/1000)


def main():
    release_path=ROOT/'results/gftd_modulation_20261107/execution_release.json'
    release=json.loads(release_path.read_text(encoding='utf-8-sig'))
    before={path:digest(ROOT/path) for path in release['source_sha256']}
    assert all(value==release['source_sha256'][path].lower() for path,value in before.items())
    rows=[]
    begin=time.perf_counter()
    for n in (4,10):
        for mode in ('static','modulated'):
            baseline=fixture(n,mode)
            cases=[np.linspace(-.2,.3,5*n),np.zeros(5*n)]
            bound=np.linspace(-.1,.1,5*n)
            bound[3*n:4*n]=.2*(-1.)**np.arange(n)
            bound[4*n:]=.2*(-1.)**np.arange(n)
            cases.append(bound)
            for name,cls in [('baseline',GeneratingFunctionalKuramoto),('vector_gradient',VectorGradient),
                             ('constant_gate',ConstantGate),('combined',Combined)]:
                f=fixture(n,mode,cls)
                records=[]
                for state in cases:
                    old,new=baseline.rhs(state),f.rhs(state)
                    oldgrad=np.concatenate(baseline.gradients(state))
                    newgrad=np.concatenate(f.gradients(state))
                    delta=float(np.max(np.abs(new-old)))
                    scale=float(np.max(np.abs(old)))
                    gradient_delta=float(np.max(np.abs(newgrad-oldgrad)))
                    power=f.power_balance_audit(state)
                    fd=[]
                    for coordinate in range(5*n):
                        shift=np.zeros(5*n);shift[coordinate]=1e-6
                        fd.append((f.hamiltonian(state+shift)-f.hamiltonian(state-shift))/(2e-6))
                    finite_difference_max=float(np.max(np.abs(np.array(fd)-newgrad)))
                    assert delta<=1e-12 and gradient_delta<=1e-12
                    assert power['strictly_dissipative'] and power['power_leakage_abs']<=1e-12
                    assert finite_difference_max<=1e-7
                    shifted=state.copy();shifted[:n]+=2.
                    assert np.max(np.abs(f.rhs(shifted)-f.rhs(state)))<=1e-12
                    records.append(dict(rhs_abs_max=delta,rhs_relative_max=delta/scale if scale else 0.,
                        gradient_abs_max=gradient_delta,power_leakage_abs=power['power_leakage_abs'],
                        finite_difference_gradient_abs_max=finite_difference_max))
                for invalid in (np.zeros(5*n-1),np.full(5*n,np.nan),np.full(5*n,np.inf)):
                    try:f.rhs(invalid)
                    except ValueError:pass
                    else:raise AssertionError('invalid state accepted')
                # State-dependent gates retain the original branch, including biases.
                f.gate_weight_amp=np.full((n,3),.1);f.gate_weight_mem=np.full((n,3),-.1)
                baseline.gate_weight_amp=f.gate_weight_amp.copy();baseline.gate_weight_mem=f.gate_weight_mem.copy()
                dynamic_delta=float(np.max(np.abs(f.rhs(cases[0])-baseline.rhs(cases[0]))))
                assert dynamic_delta<=1e-12
                f.gate_weight_amp=f.gate_weight_mem=None
                baseline.gate_weight_amp=baseline.gate_weight_mem=None
                timing=measure(f,cases[0])
                rows.append(dict(n_nodes=n,n_edges=f.n_edges,mode=mode,implementation=name,checks=records,
                    dynamic_gate_fallback_rhs_abs_max=dynamic_delta,boundary_rejections=3,**timing))
    after={path:digest(ROOT/path) for path in release['source_sha256']}
    assert after==before
    result=dict(scope='Deterministic array RHS engineering benchmark; no RK4, trajectory, optimizer, fitted parameters or private data.',
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        processor=platform.processor(),processor_identifier=os.environ.get('PROCESSOR_IDENTIFIER'),
        logical_cpus=os.cpu_count(),threadpools=None,thread_env={key:os.environ.get(key) for key in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},dtype='float64',
        source_sha256=before,source_unchanged=True,release_sha256=digest(release_path),
        benchmark_sha256=digest(__file__),rows=rows,timing_trials_per_cell=1,
        timed_rhs_calls_total=16000,warmup_rhs_calls_total=160,qualification_rhs_calls=None,
        qualification_note='RHS/gradient/power/finite-difference checks are untimed; no residual integral was executed.',
        wall_seconds=time.perf_counter()-begin,history_rhs_calls_per_residual=160,
        n10_jacobian=dict(simple_dimensions=10,smooth_static_dimensions_sigma0=10,
            smooth_modulated_dimensions_sigma0=20,smooth_static_dimensions_positive_sigma=29,
            smooth_modulated_dimensions_positive_sigma=39,
            central_difference_residual_calls=dict(simple=21,smooth_static_sigma0=21,
                smooth_modulated_sigma0=41,smooth_static_positive_sigma=59,smooth_modulated_positive_sigma=79),
            note='1 base residual plus 2 per variable, no optimizer invocation; boundary one-sided formulas may differ.'),
        scientific_execution=False,formal_conditions_or_seeds_used=False,private_data_reads=False,new_fom=0)
    (OUT/'benchmark.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(result_path=str(OUT/'benchmark.json'),rows=len(rows),timed_rhs_calls=16000,
        warmup_rhs_calls=160,source_unchanged=True,wall_seconds=result['wall_seconds'])))
    for row in rows:print(row['n_nodes'],row['mode'],row['implementation'],round(row['seconds_per_rhs']*1e6,3))


if __name__=='__main__':main()
