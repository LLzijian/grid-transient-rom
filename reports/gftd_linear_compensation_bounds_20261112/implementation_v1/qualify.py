"""Released hand-written qualification of exact bounds and the BVLS guard."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import run

np = run.np


def rejects(callback, error=ValueError):
    try:
        callback()
    except error:
        return
    raise AssertionError("expected rejection")


def cert(A, b, lo, hi, z):
    work = {}
    v = run.arrays(dict(A=A,b=b,lower=lo,upper=hi,v=z),(len(A),len(z)))
    result = run.certificate(*(v[k] for k in ("A","b","lower","upper","v")),work)
    return result,work


def zero():
    (lower,upper,public,private),work = cert([[1]],[-1],[0],[2],[1])
    assert lower == upper == 0 and private["t"] == [0,1] and public["decision"] == "reachable"
    assert work["matrix_products_logical"] == work["matrix_products_actual"] == 2


def positive():
    (lower,upper,public,_),_ = cert([[0]],[2],[-1],[1],[0])
    assert lower == upper == 2 and public["decision"] == "unreachable_in_fixed_linear_box"


def crossing():
    (lower,upper,public,private),_ = cert([[1]],[1],[-2],[2],[0])
    assert lower == 0 and upper == F(1,2) and public["decision"] == "unresolved"
    assert private["h"] == [-1,1] and private["t"] == [0,1]


def ties():
    t = run.THETA
    assert run.classify(t,t) == "reachable"
    assert run.classify(t,2*t) == "unresolved"
    assert run.classify(2*t,2*t) == "unreachable_in_fixed_linear_box"
    rejects(lambda: run.classify(2*t,t))


def box_signs():
    (lower,upper,_,private),_ = cert([[1,-1]],[4],[-1,-2],[2,1],[-1,1])
    assert lower == upper == F(2)
    assert private["h"] == [4,1] and private["t"] == [1,1]


def dyadic():
    subnormal = np.nextafter(0.,1.)
    (lower,upper,_,private),work = cert([[1e16,subnormal,-1e16]],[0],[0]*3,[2]*3,[1]*3)
    exact = F(1,2**1074)
    assert private["residual"] == [run.ratio(exact)]
    assert upper == exact**2/2 and 0 <= lower <= upper
    assert work["matrix_products_actual"] == 6


def bad_inputs():
    for z in ([float("nan")],[2]):
        rejects(lambda: run.arrays(dict(A=[[1]],b=[0],lower=[0],upper=[1],v=z),(1,1)))
    rejects(lambda: run.check_hashes({"A":np.zeros((1,1))},{"A":"incorrect"}))


def real_solver(modules, ledger):
    A,b,lo,hi = np.eye(2),np.array([-2.,-.25]),np.zeros(2),np.ones(2)
    z,mask = run.bvls(A,b,lo,hi,modules,ledger,max_iter=4,limit=16)
    assert run.feasible(z,lo,hi) and np.allclose(z,[1.,.25]) and mask == [1.,0.]
    assert {row["entry"] for row in ledger["lstsq"]} == {"unbounded","bvls"}
    assert all(row["status"] == "returned" for row in ledger["lstsq"])
    rejects(lambda: run.bvls(A,b,lo,hi,modules,ledger,max_iter=4,limit=16))


def mock_solver(fail):
    A,b,lo,hi = np.eye(2),np.array([-2.,-.25]),np.zeros(2),np.ones(2)
    def decompose(matrix,rhs,*args,**kwargs):
        if fail and rhs[0] == 1:
            raise np.linalg.LinAlgError("fixture")
        return np.zeros(matrix.shape[1]),np.zeros(0),0,np.zeros(matrix.shape[1])
    first = SimpleNamespace(np=SimpleNamespace(linalg=SimpleNamespace(lstsq=decompose)))
    second = SimpleNamespace(lstsq=decompose)
    def exact(*_args,**_kwargs):
        first.np.linalg.lstsq(A,b)
        if fail:
            second.lstsq(A,np.ones(2))
        else:
            for _ in range(16):
                second.lstsq(A,b)
        raise AssertionError("guard did not stop")
    first.lsq_linear = exact
    ledger = dict(problems=0,lstsq=[])
    assert run.bvls(A,b,lo,hi,(first,second),ledger,max_iter=4,limit=16) == (None,None)
    assert first.np.linalg.lstsq is second.lstsq is decompose
    assert len(ledger["lstsq"]) == (2 if fail else 16)
    if fail:
        assert ledger["lstsq"][-1]["status"] == "failed"
        assert ledger["lstsq"][-1]["ended_utc"] >= ledger["lstsq"][-1]["started_utc"]
    else:
        assert ledger["stop_reason"] == "DecompositionLimit"
    rejects(lambda: run.bvls(A,b,lo,hi,(first,second),ledger,max_iter=4,limit=16))


def privacy():
    (_,_,public,private),_ = cert([[1]],[2],[-1],[1],[0])
    assert "residual" not in public and "z_f64_le_hex" not in public and "active_mask" not in public
    assert "residual" in private and "z_f64_le_hex" in private
    run.json_bytes(public)


def main():
    run.require(__debug__, "qualification assertions must be enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=Path,required=True)
    args = parser.parse_args()
    release,identity,modules = run.release_context(args.release,"qualify")
    attempt,previous = release["attempt"],release["previous"]
    run.require(type(attempt) is int and 1 <= attempt <= 2 and len(previous) == attempt-1,
                "qualification attempt differs")
    for index,ref in enumerate(previous,1):
        old = json.loads(run.read_ref(ref,run.RESULTS/f"qualification_{index}.json"))
        run.require(old["run"] == run.RUN and old["attempt"] == index and old["status"] == "failed"
                    and old["protocol_sha256"] == run.PROTOCOL and old["real_science_calls"] == 0
                    and old["private_reads"] == 0 and 0 <= old["cases"] <= 12
                    and 0 <= old["dense_lstsq"] <= 16 and old["bvls"]["problems"] <= 1,
                    "prior qualification ledger differs")
    identity.update(attempt=attempt,previous=previous)
    claim = run.save_json(run.RESULTS/f"qualification_claim_{attempt}.json",identity)
    ledger = dict(problems=0,lstsq=[])
    cases = (zero,positive,crossing,ties,box_signs,dyadic,bad_inputs,
             lambda: real_solver(modules,ledger),lambda: mock_solver(False),lambda: mock_solver(True),privacy)
    completed, attempted = [],0
    receipt = dict(**identity,claim=claim,status="failed",real_science_calls=0,private_reads=0)
    wall,cpu = time.perf_counter(),time.process_time()
    try:
        run.require(len(cases) <= 12, "qualification case budget")
        for case,name in zip(cases,("zero","positive","crossing","ties","box_signs","dyadic","bad_inputs",
                                   "real_bvls","dual_entry_limit","failure_ledger","privacy"),strict=True):
            attempted += 1
            case()
            completed.append(name)
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        receipt.update(cases=attempted,completed=completed,bvls=ledger,dense_lstsq=len(ledger["lstsq"]),
                       function_wall_seconds=time.perf_counter()-wall,function_cpu_seconds=time.process_time()-cpu)
        run.save_json(run.RESULTS/f"qualification_{attempt}.json",receipt)


if __name__ == "__main__":
    main()
