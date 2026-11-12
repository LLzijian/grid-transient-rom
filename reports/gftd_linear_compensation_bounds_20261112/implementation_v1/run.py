"""Released exact bounds for the saved linear compensation box."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from decimal import Decimal, localcontext
from fractions import Fraction
import gzip
import hashlib
import importlib
from importlib.metadata import distribution
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
if __name__ == "__main__":
    for variable in THREADS:
        os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_linear_compensation_bounds_20261112"
PARENT = "gftd_joint_distinguishability_20261112"
RESULTS, PRIVATE = Path("results")/RUN, Path("artifacts")/RUN
ADOPTION = "ba6a975f28c4e47dd60a00948f18050bfa9d572bb4040f247e00fd19286c05d6"
PROTOCOL = "16aef2bc134436c44208098f79b31d81b78bcbabfaa32cd81fdfbb11a1c5eded"
THETA = Fraction(2048, 10**12)
PARENT_FILES = {
    "adoption": (Path("results")/PARENT/"final_adoption.json", "23f3dca451c5ab6083522bcc328fe4add1247e2a992712577df5935b044699ac"),
    "execution": (Path("results")/PARENT/"science/execution.json", "27d3ad39306c52e3fa581d48c68c54db8af2dd1da0d0fa8974c03932637a2dca"),
    "audit": (Path("results")/PARENT/"result_audit_v2.json", "ebc7380be6a62bebaec884d3562d026ec8d8b1901ba2b80e20b20a244d3eedd7"),
    "evidence": (Path("artifacts")/PARENT/"science/evidence.json.gz", "81a64b608f7330e48eae65c19ee3cf9e9a4b2337401aaae444341a4915c14422"),
}
INPUT_SHA = {"A": "77ec1bd31bedd68c7b468920c988e52579a781fcf900591afa19c31c54f36a44",
             "b": "8a791621faf208a8e4d714abf853869b3e4e6a6ac59d9bfe9bb17726b551fa5f"}
LIBRARIES = {
    "scipy/optimize/_lsq/lsq_linear.py": "9a6e3a1e3f965510ec49c193d86543f897e6d7a89f2962aec4eb9e0b73af3b9d",
    "scipy/optimize/_lsq/bvls.py": "87d6d88776acb4846f32d306821b441a91b97c3b21ef848c45b9a01771cdf183",
    "numpy/linalg/_linalg.py": "76ee5b752078b400b2ee9a946a7bc50b16d98034cb8a5f4a43d6d94af00f6d22",
    "numpy/linalg/_umath_linalg.cp312-win_amd64.pyd": "26315b00a500ebfa5723776853ba651e3cae05c22144f41305ba0cf3b29b9542",
    "numpy.libs/libscipy_openblas64_-327b2e0bcffce2882e0dc04cdeb4eaa6.dll": "327b2e0bcffce2882e0dc04cdeb4eaa6382ed4dba29871b6176fa157782f484a",
    "numpy.libs/msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll": "a4c2229bdc2a2a630acdc095b4d86008e5c3e3bc7773174354f3da4f5beb9cde",
}
SCRIPT_PATHS = tuple(Path("experiments")/RUN/f"{name}.py" for name in ("run", "qualify"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local(path):
    target = (ROOT/path).resolve()
    require(target.is_relative_to(ROOT) and target != ROOT, "invalid workspace path")
    return target


def reference(path, raw):
    return dict(path=path.as_posix(), bytes=len(raw), sha256=sha(raw))


def read_bound(path, digest, size=None):
    raw = local(path).read_bytes()
    require(sha(raw) == digest and (size is None or len(raw) == size), f"frozen file differs: {path}")
    return raw


def read_ref(ref, path):
    require(ref["path"] == path.as_posix(), f"unexpected frozen path: {path}")
    return read_bound(path, ref["sha256"], ref["bytes"])


def json_bytes(payload):
    return (json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))+"\n").encode()


def save(path, raw):
    target = local(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return reference(path, raw)


def save_json(path, payload):
    return save(path, json_bytes(payload))


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release_context(path, phase):
    raw = local(path).read_bytes()
    release = json.loads(raw)
    adoption = json.loads(read_bound(RESULTS/"design_adoption.json", ADOPTION))
    require(adoption["status"] == "design_accepted_for_minimal_implementation"
            and adoption["references"]["protocol"]["sha256"] == PROTOCOL, "adoption identity differs")
    for ref in adoption["references"].values():
        read_ref(ref, Path(ref["path"]))
    sources = {p.as_posix(): file_sha(local(p)) for p in SCRIPT_PATHS}
    runtime = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=distribution("scipy").version,
                   python_executable_sha256=file_sha(Path(sys.executable)),
                   python_base_executable_sha256=file_sha(Path(sys._base_executable)),
                   threads={name: os.environ.get(name) for name in THREADS})
    require((runtime["python"], runtime["numpy"], runtime["scipy"]) == ("3.12.14", "2.5.2", "1.18.0")
            and set(runtime["threads"].values()) == {"1"}, "numerical runtime differs")
    require(release.get("run") == RUN and release.get("phase") == phase and release.get("released") is True
            and isinstance(release.get("action_id"), str) and bool(release["action_id"])
            and release.get("protocol_sha256") == PROTOCOL and release.get("design_adoption_sha256") == ADOPTION
            and release.get("source_sha256") == sources and release.get("runtime") == runtime
            and release.get("library_sha256") == LIBRARIES, "explicit frozen release differs")
    installed = {}
    for name, digest in LIBRARIES.items():
        package = "scipy" if name.startswith("scipy/") else "numpy"
        installed[name] = Path(distribution(package).locate_file(name)).resolve()
        require(file_sha(installed[name]) == digest, f"numerical library differs: {name}")
    modules = tuple(importlib.import_module(name) for name in
                    ("scipy.optimize._lsq.lsq_linear", "scipy.optimize._lsq.bvls"))
    for module, name in zip(modules, tuple(LIBRARIES)[:2], strict=True):
        require(Path(module.__file__).resolve() == installed[name], "SciPy import path differs")
    require(Path(importlib.import_module("numpy.linalg._linalg").__file__).resolve()
            == installed["numpy/linalg/_linalg.py"]
            and Path(importlib.import_module("numpy.linalg._umath_linalg").__file__).resolve()
            == installed["numpy/linalg/_umath_linalg.cp312-win_amd64.pyd"]
            and modules[0].np is np and modules[1].lstsq is np.linalg.lstsq, "lstsq identity differs")
    identity = {key: release[key] for key in ("run", "phase", "action_id", "protocol_sha256",
                "design_adoption_sha256", "source_sha256", "runtime", "library_sha256")}
    identity["release_sha256"] = sha(raw)
    return release, identity, modules


def arrays(raw, shape):
    m, n = shape
    shapes = dict(A=(m,n), b=(m,), lower=(n,), upper=(n,), v=(n,))
    values = {name: np.asarray(raw[name], dtype="<f8", order="C") for name in shapes}
    require(all(v.shape == shapes[k] and np.all(np.isfinite(v)) for k,v in values.items()),
            "nonfinite input or invalid shape")
    require(np.all(values["lower"] < values["upper"]) and
            feasible(values["v"], values["lower"], values["upper"]), "invalid box or saved point")
    return values


def feasible(z, lo, hi):
    return bool(np.shape(z) == lo.shape and np.all(np.isfinite(z)) and np.all((lo <= z) & (z <= hi)))


def binary_sha(array):
    return sha(np.asarray(array, dtype="<f8", order="C").tobytes())


def check_hashes(values, expected):
    require(all(binary_sha(values[key]) == digest for key, digest in expected.items()), "binary64 input hash differs")


def read_input(evidence_ref):
    path, digest = PARENT_FILES["evidence"]
    require(evidence_ref["sha256"] == digest, "private input binding differs")
    raw = read_ref(evidence_ref, path)
    decoded = json.loads(gzip.decompress(raw))
    values = arrays(decoded, (5280,525))
    check_hashes(values, INPUT_SHA)
    return values


def rational(x):
    return Fraction(*float(x).as_integer_ratio())


def ratio(x):
    return [x.numerator, x.denominator]


def number(x):
    with localcontext() as context:
        context.prec = 18
        decimal = str(Decimal(x.numerator)/Decimal(x.denominator))
    return dict(numerator=x.numerator, denominator=x.denominator, decimal=decimal)


def classify(lower, upper, theta=THETA):
    require(0 <= lower <= upper, "invalid certificate interval")
    if upper <= theta:
        return "reachable"
    if lower > theta:
        return "unreachable_in_fixed_linear_box"
    return "unresolved"


def certificate(A, b, lo, hi, z, work):
    require(feasible(z, lo, hi), "certificate point outside original box")
    m,n = A.shape
    work.update(matrix_products_logical=0, matrix_products_actual=0, squares=0,
                b_times_r=0, box_endpoint_products=0)
    zz, bb = list(map(rational,z)), list(map(rational,b))
    low, high = list(map(rational,lo)), list(map(rational,hi))
    q, residual = [Fraction(0)]*n, []
    s = h = Fraction(0)
    for i,row in enumerate(A):
        entries = [(j,rational(row[j])) for j in np.flatnonzero(row)]
        r = bb[i]
        work["matrix_products_logical"] += n
        for j,a in entries:
            r += a*zz[j]
            work["matrix_products_actual"] += 1
        residual.append(r)
        s += r*r
        h += bb[i]*r
        work["squares"] += 1
        work["b_times_r"] += 1
        work["matrix_products_logical"] += n
        for j,a in entries:
            q[j] += a*r
            work["matrix_products_actual"] += 1
    for l,u,qi in zip(low,high,q,strict=True):
        h += min(l*qi,u*qi)
        work["box_endpoint_products"] += 2
    t = max(Fraction(0),h)/s if s else Fraction(0)
    U = s/2
    L = t*h-t*t*s/2 if s else Fraction(0)
    decision = classify(L,U)
    witness = dict(z_f64_le_hex=np.asarray(z,dtype="<f8").tobytes().hex(),
                   residual=list(map(ratio,residual)), q_sha256=sha(json_bytes(list(map(ratio,q)))),
                   s=ratio(s), h=ratio(h), t=ratio(t), L=ratio(L), U=ratio(U))
    public = dict(lower=number(L), upper=number(U), gap=number(U-L), feasible=True,
                  decision=decision, z_sha256=binary_sha(z), q_sha256=witness["q_sha256"])
    return L,U,public,witness


class DecompositionLimit(RuntimeError):
    """No dense call was made after the registered limit."""


def finite_number(value):
    val = float(value)
    return val if math.isfinite(val) else None


def bvls(A, b, lo, hi, modules, ledger, *, max_iter=100, limit=1051):
    require(ledger["problems"] == 0 and (max_iter,limit) in ((100,1051),(4,16)), "BVLS call already used or unregistered")
    max_shape = (5280,525) if max_iter == 100 else (4,3)
    require(A.ndim == 2 and A.shape == (b.size,lo.size) and lo.shape == hi.shape
            and all(v <= cap for v,cap in zip(A.shape,max_shape,strict=True)), "invalid BVLS shape")
    ledger["problems"] += 1
    calls = ledger["lstsq"]
    first, active = modules

    def guard(alias, original):
        def invoke(matrix, rhs, *args, **kwargs):
            if len(calls) >= limit:
                raise DecompositionLimit("dense lstsq allowance exhausted")
            require(matrix.ndim == 2 and rhs.shape == (matrix.shape[0],)
                    and all(v <= cap for v,cap in zip(matrix.shape,max_shape,strict=True)), "invalid internal lstsq shape")
            wall,cpu = time.perf_counter(),time.process_time()
            row = dict(number=len(calls)+1, entry=alias, shape=list(matrix.shape),
                       started_utc=time.time(), status="interrupted")
            calls.append(row)
            try:
                returned = original(matrix,rhs,*args,**kwargs)
                _,_,rank,singular = returned
                row.update(status="returned",rank=int(rank),singular_values=dict(count=int(singular.size),
                           min=finite_number(np.min(singular)) if singular.size else None,
                           max=finite_number(np.max(singular)) if singular.size else None,
                           finite=bool(np.all(np.isfinite(singular)))))
                return returned
            except Exception as error:
                row.update(status="failed",error_type=type(error).__name__)
                raise
            finally:
                row.update(ended_utc=time.time(),wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
        return invoke

    wall,cpu = time.perf_counter(),time.process_time()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(first.np.linalg,"lstsq",guard("unbounded",first.np.linalg.lstsq)))
            stack.enter_context(patch.object(active,"lstsq",guard("bvls",active.lstsq)))
            result = first.lsq_linear(A,-b,bounds=(lo,hi),method="bvls",lsq_solver="exact",
                                      tol=1e-12,max_iter=max_iter,verbose=0)
        z = np.asarray(result.x,dtype="<f8")
        valid = feasible(z,lo,hi)
        ledger["return"] = dict(status=int(result.status),nit=int(result.nit),message=str(result.message),
                                cost=finite_number(result.cost),optimality=finite_number(result.optimality),
                                active_mask_counts={str(i):int(np.count_nonzero(result.active_mask == i))
                                                    for i in (-1,0,1)}, feasible=valid)
        return (z,np.asarray(result.active_mask).tolist()) if valid else (None,None)
    except (DecompositionLimit,np.linalg.LinAlgError,FloatingPointError) as error:
        ledger["stop_reason"] = type(error).__name__
        return None,None
    finally:
        ledger.update(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)


def verify_public_inputs():
    refs, records = {}, {}
    for key in ("adoption","execution","audit"):
        path,digest = PARENT_FILES[key]
        raw = read_bound(path,digest)
        refs[key],records[key] = reference(path,raw),json.loads(raw)
    execution = records["execution"]
    adopted = records["adoption"]["references"]
    require(execution["status"] == "completed" and adopted["execution"] == refs["execution"]
            and adopted["result_audit"] == refs["audit"], "parent decision references differ")
    evidence = execution["evidence"]
    path,digest = PARENT_FILES["evidence"]
    require(evidence["path"] == path.as_posix() and evidence["sha256"] == digest, "private reference differs")
    refs["evidence"] = evidence
    return refs


def science(identity, modules):
    path = RESULTS/"science"
    claim = save_json(path/"claim.json",identity)
    ledger = dict(certificates=[], bvls=dict(problems=0,lstsq=[]))
    public = dict(**identity,claim=claim,status="interrupted",decision=None,theta=number(THETA),
                  costs=ledger,model_rhs=0,future_reads=0,original_data_reads=0,lsmr_calls=0,independent_svd_calls=0)
    private = []
    wall,cpu = time.perf_counter(),time.process_time()
    try:
        public["parent_refs"] = verify_public_inputs()
        values = read_input(public["parent_refs"]["evidence"])
        public["input_binary64_sha256"] = {k:binary_sha(v) for k,v in values.items()}
        A,b,lo,hi,v = (values[key] for key in ("A","b","lower","upper","v"))

        def certify(label,z,mask=None):
            detail = dict(label=label,status="interrupted")
            ledger["certificates"].append(detail)
            start,clock = time.perf_counter(),time.process_time()
            try:
                low,upper,summary,witness = certificate(A,b,lo,hi,z,detail)
                private.append(dict(label=label,active_mask=mask,**witness))
                detail.update(status="completed",**summary)
                return low,upper
            finally:
                detail.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-clock)

        lower,upper = certify("saved",v)
        if classify(lower,upper) == "unresolved":
            new,mask = bvls(A,b,lo,hi,modules,ledger["bvls"])
            if new is not None:
                new_lower,new_upper = certify("bvls",new,mask)
                lower,upper = max(lower,new_lower),min(upper,new_upper)
        public.update(status="completed",decision=classify(lower,upper),
                      lower=number(lower),upper=number(upper),gap=number(upper-lower))
    except Exception as error:
        public.update(status="invalid",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        public.update(function_wall_seconds=time.perf_counter()-wall,function_cpu_seconds=time.process_time()-cpu)
        start = time.perf_counter()
        try:
            if private:
                public["witnesses"] = save(PRIVATE/"science/witnesses.json.gz",
                                            gzip.compress(json_bytes(private),mtime=0))
        except Exception as error:
            public.update(status="interrupted",decision=None,serialization_error_type=type(error).__name__)
            raise
        finally:
            public["serialization_wall_seconds"] = time.perf_counter()-start
            save_json(path/"execution.json",public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=Path,required=True)
    args = parser.parse_args()
    release,identity,modules = release_context(args.release,"science")
    qualification = release["qualification"]
    attempt = qualification["attempt"]
    require(type(attempt) is int and 1 <= attempt <= 2, "invalid qualification attempt")
    receipt = json.loads(read_ref(qualification["receipt"],RESULTS/f"qualification_{attempt}.json"))
    require(receipt["status"] == "passed" and receipt["attempt"] == attempt and receipt["run"] == RUN
            and all(receipt[k] == identity[k] for k in ("protocol_sha256","design_adoption_sha256",
                                                       "source_sha256","runtime","library_sha256"))
            and receipt["real_science_calls"] == 0 and receipt["private_reads"] == 0
            and receipt["cases"] <= 12 and receipt["dense_lstsq"] <= 16, "qualification binding differs")
    science(dict(**identity,qualification=qualification),modules)


if __name__ == "__main__":
    main()
