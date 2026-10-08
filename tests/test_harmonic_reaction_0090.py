"""Synthetic physical assembly tests; no private inputs."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import audit_harmonic_reaction_0090 as m


def oscillator():
    G=np.zeros((33,33)); G[:10,10:20]=np.eye(10)
    G[10:20,:10]=-2*np.eye(10); G[10:20,10:20]=-.3*np.eye(10)
    G[20:,20:]=-np.eye(13)
    mass=np.linspace(.4,1.3,10); E=np.ones(33); E[10:20]=mass*m.OMEGA_S
    scale=np.ones(33); scale[10:20]=1/m.OMEGA_S
    F=E[:,None]*scale[:,None]*G/scale[None,:]
    return G,F,E,mass


def test_known_reaction_omega_mass_conversion_and_zero_rhs():
    G,F,E,mass=oscillator(); s=1.2j
    reactions=[]
    for A,diag,omega,isfom in ((F,E,m.OMEGA_S,True),(G,np.ones(33),1.,False)):
        H,B=m.harmonic_system(A,diag,s,omega)
        solved=m.solve_scaled(H,B,time.perf_counter()+10)
        assert solved['qa_pass']
        T,R,res=m.reconstruct(A,diag,mass,s,solved['X'],omega,isfom)
        assert res<1e-13
        assert np.allclose(R,mass[:,None]*(s*s+2+.3*s)*np.eye(10))
        reactions.append(R)
    assert np.allclose(*reactions)


def test_algebraic_feedthrough_reconstruction():
    G,F,E,mass=oscillator(); E[20]=0.; F[20]=0.; F[20,20]=-2.; F[20,0]=3.
    F[10,20]=.8; s=.7j
    H,B=m.harmonic_system(F,E,s,m.OMEGA_S)
    solved=m.solve_scaled(H,B,time.perf_counter()+10)
    assert solved['qa_pass'] and abs(solved['X'][0,0]-1.5)<1e-13
    T,R,res=m.reconstruct(F,E,mass,s,solved['X'],m.OMEGA_S,True)
    assert res<1e-13
    assert abs(T[0,0]-(mass[0]*(-2-.3*s)+1.2))<1e-13


def test_normalized_metrics_keep_common_output():
    mass=np.arange(1.,11.); R=np.tile(np.eye(10),(13,1,1)).astype(complex)
    Kf,Kg,rows,aggregate,torque=m.metric_report(R,R,R,1.1*R,mass,np.ones(13))
    assert abs(aggregate-.1)<1e-13 and torque==0.
    assert all(abs(row['relative_reaction_error']-.1)<1e-13 and row['common_error_absolute']>0 for row in rows)


def test_main_preserves_first_solve_failure_without_retry(tmp_path,monkeypatch):
    source=tmp_path/'source.txt'; source.write_text('synthetic')
    monkeypatch.setattr(m,'source_paths',lambda:{'synthetic':source})
    G,F,E,mass=oscillator()
    monkeypatch.setattr(m,'assemble_inputs',lambda paths:(F,E,G,mass,np.ones(13),{}))
    calls=[]
    def failed(H,B,deadline):
        calls.append(1)
        return {'qa_pass':False,'error':'synthetic_singular','metrics':{},'solve_attempts':1,'svd_attempts':1,'X':None,'Y':None,'row_scale':None,'column_scale':None,'singular_values':None}
    monkeypatch.setattr(m,'solve_scaled',failed)
    output=tmp_path/'fresh'
    with pytest.raises(ArithmeticError):m.run(output)
    evidence=json.loads((output/'failure_evidence.json').read_text())
    assert evidence['solve_attempts']==1 and len(calls)==1
    assert (output/'arrays.npz').exists() and (output/'failed_solve.npz').exists()
