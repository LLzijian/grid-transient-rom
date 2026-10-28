"""One isolated CPU prototype/verification; never adopted by the live driver.

Imports only byte-for-byte local clones, not the running experiment modules.
No optimizer/CUDA/event payload is accessed. Timings cover fields only.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import traceback
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from baseline_state_gated_memory_model_20261028 import StateGatedMemoryROM,ControlDegenerate

torch.set_num_threads(1)
SOURCE=ROOT/"private_data/force_graph_20261028/study01/study/N6101_M_28102/checkpoints/step000192/model.json"
SOURCE_SHA="937f00b740875913923487b8d772f40d2a8145a88eba2cdaf61f2d889715bfc9"
PROTECTED={
 "experiments/state_gated_memory_model_20261028.py":"ad788b1c7533ee8a803b15805f4b06af2341d7ecad4df55b541a9de0618c697b",
 "experiments/force_graph_model_20261028.py":"44e5fe2e1403bc7e072d2fb87100eaa694b15da5b4694169de141cc6059089af"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FixedContextPrototype(StateGatedMemoryROM):
    """Only field/effective-weight algebra changes; C and history solver intact."""
    def _fixed(self,c):
        if "perf_fixed" in c: return c["perf_fixed"]
        a=self.lambda_reference*self.a_bar
        data=dict(a=a,L=(self.B*a)@self.B.T)
        if self.gate_mode=="B":
            scale=a.abs().max()
            if not torch.isfinite(a).all() or (a<0).any():
                raise ControlDegenerate("invalid physical B weights")
            zero=scale.item()==0
            data["zero"]=zero
            if not zero:
                data["normalized"]=a/scale
                data["normalizer"]=data["normalized"].sum()
        c["perf_fixed"]=data
        return data

    def _mean_and_checked_weights(self,delta,c):
        # Keep the state-dependent gate computation and actual per-state
        # weight finite check, including the original zero-graph semantics.
        g=self.gate_values(delta);data=self._fixed(c)
        if data["zero"]:
            if torch.is_grad_enabled() and (self.a_bar.requires_grad or self._tensor(delta).requires_grad):
                raise ControlDegenerate("B total a=0 has no unique ordinary training derivative")
            return None,torch.zeros_like(g)
        mean=(g*data["normalized"]).sum(-1,keepdim=True)/data["normalizer"]
        weights=data["a"]*mean
        if not torch.isfinite(weights).all(): raise ControlDegenerate("B weighted ratio nonfinite")
        return mean,weights

    def effective_weights(self,delta):
        if self.gate_mode=="C": return super().effective_weights(delta)
        c=self._context()
        if self.gate_mode=="A":
            d=self._tensor(delta)
            return self._fixed(c)["a"].expand(d.shape[:-1]+(45,))
        return self._mean_and_checked_weights(delta,c)[1]

    def memory_action(self,delta,r,context=None):
        if self.gate_mode=="C": return super().memory_action(delta,r,context)
        c=self._context() if context is None else context
        if self.gate_mode=="A": return c["lambda_"]*r+r@self._fixed(c)["L"].T
        mean,weights=self._mean_and_checked_weights(delta,c)
        if mean is None:
            # Preserve even the baseline's IEEE nonfinite propagation at the
            # zero-graph boundary; this exceptional branch is not accelerated.
            return c["lambda_"]*r+((r@self.B)*weights)@self.B.T
        return c["lambda_"]*r+mean*(r@self._fixed(c)["L"].T)


def pair(mode):
    return [cls.from_force_graph_export(SOURCE,mode) for cls in (StateGatedMemoryROM,FixedContextPrototype)]


GEN=torch.Generator().manual_seed(20261029)
STATES=[.35*torch.randn(12,40,dtype=torch.float64,generator=GEN) for _ in range(4)]
WEIGHT=torch.linspace(-.4,.9,40,dtype=torch.float64)


def compare(mode,scale=1.):
    models=pair(mode)
    for m in models:
        with torch.no_grad(): m.a_bar.mul_(scale)
    results=[]
    for m in models:
        z=[s.detach().clone().requires_grad_() for s in STATES]
        m.begin_graph_cache()
        try:
            fields=[m.field(x) for x in z]
            # Repeated use of the same cached parameter graph then one backward.
            loss=sum(((f*WEIGHT).square().sum()) for f in fields)
            loss.backward()
            results.append(dict(field=[f.detach() for f in fields],state=[x.grad.detach() for x in z],
               raw={n:p.grad.detach().clone() for n,p in m.named_parameters()}))
        finally: m.end_graph_cache()
    errors={}
    for kind in ("field","state"):
        for i,(a,b) in enumerate(zip(results[0][kind],results[1][kind])):
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            errors[f"{kind}{i}"]=float((a-b).abs().max())
    for name in results[0]["raw"]:
        a,b=results[0]["raw"][name],results[1]["raw"][name]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        errors[name]=float((a-b).abs().max())
    # Version change inside a live context must invalidate all fixed tensors.
    for m in models:
        m.zero_grad(set_to_none=True);m.begin_graph_cache();m.field(STATES[0])
        with torch.no_grad(): m.a_bar.add_(.0007);m.kappa_raw.add_(.0001);m.lambda_raw.sub_(.0002)
    for m in models:
        output=m.field(STATES[1]);assert m._cache is None
        output.square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    torch.testing.assert_close(models[0].field(STATES[1]),models[1].field(STATES[1]),atol=1e-12,rtol=1e-12)
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-12,rtol=1e-12)
    # A second fresh graph after end_graph_cache and a grad-mode transition.
    for m in models:
        with torch.no_grad(): m.begin_graph_cache();m.field(STATES[0])
        m.zero_grad(set_to_none=True);m.field(STATES[0]).square().sum().backward()
        assert m._cache is None
        m.begin_graph_cache();m.field(STATES[0]).sum().backward();m.end_graph_cache()
        assert m._cache is None
    return dict(mode=mode,a_scale=scale,maximum_absolute_errors=errors)


def exceptions():
    rows=[]
    for condition in ("zero_no_grad","zero_grad","negative","nan_weight","infinite_weight","nan_state","infinite_state"):
        observed=[]
        for m in pair("B"):
            with torch.no_grad():
                if condition.startswith("zero"): m.a_bar.zero_()
                elif condition=="negative": m.a_bar[0]=-.1
                elif condition=="nan_weight": m.a_bar[0]=float("nan")
                elif condition=="infinite_weight": m.a_bar[0]=float("inf")
            z=STATES[0].clone()
            if condition=="nan_state": z[0,0]=float("nan")
            if condition=="infinite_state": z[0,0]=float("inf")
            try:
                if condition=="zero_no_grad":
                    with torch.no_grad():
                        out=m.field(z)
                        expected=m.kappa*z[:,10:20]-m.lambda_*z[:,30:]
                        torch.testing.assert_close(out[:,30:],expected,atol=1e-12,rtol=1e-12)
                else: m.field(z)
                observed.append("legal_forward")
            except ControlDegenerate as exc: observed.append(type(exc).__name__+":"+str(exc))
        assert observed[0]==observed[1],(condition,observed)
        assert (observed[0]=="legal_forward")== (condition=="zero_no_grad")
        rows.append(dict(condition=condition,baseline=observed[0],prototype=observed[1]))
    # State-dependent failures cannot be skipped after fixed parameters cached.
    for m in pair("B"):
        m.begin_graph_cache();m.field(STATES[0]);z=STATES[0].clone();z[0,0]=float("nan")
        try: m.field(z)
        except ControlDegenerate: pass
        else: raise AssertionError("per-state finite check was skipped")
        finally: m.end_graph_cache()
    return rows


def timings():
    output=[]
    for mode in ("A","B"):
        row=dict(mode=mode,scope="CPU one thread float64; 128 forward field calls on fixed12-state sequence in one no_grad context; not initializer/rollout/CUDA/training")
        for name,m in zip(("baseline","prototype"),pair(mode)):
            with torch.no_grad():
                m.begin_graph_cache();wall=time.perf_counter();cpu=time.process_time()
                checksum=0.
                for i in range(128): checksum+=float(m.field(STATES[i%4]).sum())
                row[name]=dict(wall_seconds=time.perf_counter()-wall,process_cpu_seconds=time.process_time()-cpu,checksum=checksum)
                m.end_graph_cache()
        output.append(row)
    return output


if __name__=="__main__":
    result=dict(status="UNRESOLVED_CPU_EQUIVALENCE",prototype_count=1,verification_attempts=1,optimizer_calls=0,cuda_calls=0,
        real_event_arrays_read=False,training_admission=False,adopted=False,added_to_native_job_ledger=False,
        isolated_clones=True,command="PowerShell __PYVENV_LAUNCHER__=.venv-gpu/Scripts/python.exe; bundled Python private_data/state_gated_memory_20261029_perf_check/probe.py",
        tolerances=dict(atol=1e-12,rtol=1e-12),checks=[],timings=[])
    start=time.perf_counter();cpu=time.process_time()
    try:
        assert sha(SOURCE)==SOURCE_SHA
        assert all(sha(ROOT/name)==digest for name,digest in PROTECTED.items())
        assert sha(HERE/"baseline_state_gated_memory_model_20261028.py")==PROTECTED["experiments/state_gated_memory_model_20261028.py"]
        assert sha(HERE/"force_graph_model_20261028.py")==PROTECTED["experiments/force_graph_model_20261028.py"]
        for mode,scale in (("A",1.),("B",1.),("B",1e-12),("C",1.)):
            result["checks"].append(compare(mode,scale))
        result["exception_checks"]=exceptions()
        result["timings"]=timings()
        result["status"]="PASS_CPU_FIELD_EQUIVALENCE_ONLY"
    except Exception as exc:
        result["failure"]=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        result["wall_seconds"]=time.perf_counter()-start;result["process_cpu_seconds"]=time.process_time()-cpu
        result["protected_unchanged"]=all(sha(ROOT/name)==digest for name,digest in PROTECTED.items())
        result["source_model_json_sha256"]=sha(SOURCE)
        result["protected_source_hashes"]=PROTECTED
        result["prototype_sha256"]=sha(__file__)
        result["not_verified"]=["GPU speedup","full training objective equivalence","history initialization cost","integration into runtime","scientific or training admission"]
        output=ROOT/"results/state_gated_memory_20261029_perf_check.json"
        if output.exists(): raise FileExistsError("one verification only; prior result must be preserved")
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result["status"]=="PASS_CPU_FIELD_EQUIVALENCE_ONLY" else 1)
