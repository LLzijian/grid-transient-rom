"""Read-only checkpoint gradient decomposition; no optimizer or weight output."""
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch
from experiments.osgkd_v2_sprint2a_20261125 import run
from kiki.models.osgkd_spectral import projected_jacobian, reference_geometry, spectral_penalty

def audit():
    torch.set_num_threads(1)
    out = {'official_updates': 0, 'split': 'train', 'attempt': 897, 'internal_step': .01, 'horizon_s': 8, 'jobs': {}}
    for job in ('graph_24111', 'flat_24111'):
        started = time.perf_counter()
        release_path = ROOT/f'results/{run.RUN}/{job}_release.json'
        release = json.loads(release_path.read_bytes())
        inputs = run.validate(release, job)
        raw, constants = run.load_train(inputs)
        sequence = run.fit.plan(raw)
        local = ROOT/f'artifacts/{run.RUN}/{job}'
        cp_path = local/'attempt_896.pt'
        cp = torch.load(run.verified(run.identity(cp_path)), map_location='cpu', weights_only=False)
        assert cp['attempt'] == 896 and cp['release'] == release
        assert json.loads(run.verified(cp['plan_ref']).read_bytes()) == sequence
        original = json.loads((local/'attempt_0897.json').read_bytes())
        ids = sequence['batches'][896]
        assert ids == original['event_ids']
        lookup = {e['id']: e for e in run.tensor_events(raw, 'cpu', max_steps=800)}
        batch = [lookup[i] for i in ids]
        cls = run.OSGKDGraph if job.startswith('graph') else run.fit.OSGKDMatchedFlat
        model = cls.from_checkpoint(cp['model'])
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        params = list(model.parameters())
        def flat(gs):
            return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,gs)])
        work = run.IntegrationWork()
        data = run.fit.objective(run.fit.components(model,batch,constants,work,.01,800))
        penalty, alpha = spectral_penalty(model,reference_geometry(model))
        combined = data + penalty
        gd = flat(torch.autograd.grad(data,params,allow_unused=True,retain_graph=True))
        gs = flat(torch.autograd.grad(penalty,params,allow_unused=True,retain_graph=True))
        gc = flat(torch.autograd.grad(combined,params,allow_unused=True))
        values = dict(data_loss=float(data.detach()),spec=float(penalty.detach()),loss=float(combined.detach()),alpha=float(alpha.detach()),gradient_norm=float(gc.norm()))
        matches = {k: bool(np.isclose(v,original[k],rtol=1e-10,atol=1e-12)) for k,v in values.items()}
        A = projected_jacobian(model,reference_geometry(model)).detach().numpy()
        eig,V = np.linalg.eig(A)
        inverse = np.linalg.inv(V)
        order = np.argsort(-eig.real)
        top = int(order[0]); lam = eig[top]
        distances = abs(eig-lam); distances[top] = np.inf
        conjugates = abs(eig-lam.conjugate()) < 1e-10
        excluded = distances.copy()
        excluded[conjugates] = np.inf
        distinct_real = eig.real < lam.real-1e-10
        real_gap = float(lam.real-eig.real[distinct_real].max()) if distinct_real.any() else None
        directions = [('spectral_gradient', gs/gs.norm())]
        generator = torch.Generator().manual_seed(24111)
        random_direction = torch.randn(gs.shape,generator=generator,dtype=gs.dtype)
        random_direction /= random_direction.norm()
        directions.append(('fixed_random',random_direction))
        base = [p.detach().clone() for p in params]
        checks=[]
        for name,direction in directions:
            ad=float(torch.dot(gs,direction))
            for step in (1e-4,2e-5):
                vals=[]; alphas=[]
                try:
                    for sign in (1,-1):
                        offset=0
                        with torch.no_grad():
                            for p,b in zip(params,base):
                                count=p.numel(); p.copy_(b+sign*step*direction[offset:offset+count].reshape_as(p)); offset+=count
                        val,al=spectral_penalty(model,reference_geometry(model)); vals.append(float(val.detach())); alphas.append(float(al.detach()))
                finally:
                    with torch.no_grad():
                        for p,b in zip(params,base): p.copy_(b)
                fd=(vals[0]-vals[1])/(2*step)
                checks.append(dict(direction=name,step=step,ad=ad,fd=fd,absolute_error=abs(fd-ad),relative_error=abs(fd-ad)/max(abs(ad),1e-30),hinge_active_both=all(a>-.05 for a in alphas),passed=abs(fd-ad)<=1e-7+1e-3*abs(ad)))
        unchanged=all(torch.equal(v,saved[k]) for k,v in model.state_dict().items())
        finite=all(bool(torch.isfinite(g).all()) for g in (gd,gs,gc))
        ratio=float(gs.norm()/gc.norm())
        row=dict(checkpoint_ref=run.identity(cp_path),original_attempt_ref=run.identity(local/'attempt_0897.json'),parameter_count=sum(p.numel() for p in params),batch_size=len(batch),batch_target_steps=[min(800,len(e['targets']['theta'])-1) for e in batch],values=values,original_log_matches=matches,data_norm=float(gd.norm()),spectral_norm=float(gs.norm()),combined_norm=float(gc.norm()),spectral_to_combined_norm=ratio,data_spectral_cosine=float(torch.dot(gd,gs)/(gd.norm()*gs.norm())),gradient_additivity_error=float((gd+gs-gc).norm()),finite=finite,top_eigenvalues=[dict(real=float(eig[i].real),imag=float(eig[i].imag)) for i in order[:5]],top_real_gap_excluding_equal_real=real_gap,top_equal_real_count=int(np.sum(abs(eig.real-lam.real)<1e-10)),top_complex_distance=float(distances.min()),top_complex_distance_excluding_conjugate=float(excluded.min()),top_eigenvalue_condition=float(np.linalg.norm(V[:,top])*np.linalg.norm(inverse[top,:])),top_relative_residual=float(np.linalg.norm(A@V[:,top]-lam*V[:,top])/(np.linalg.norm(A)*np.linalg.norm(V[:,top]))),fd_checks=checks,model_state_unchanged=unchanged,work=work.snapshot(),wall_s=time.perf_counter()-started)
        row['passed']=bool(finite and unchanged and all(matches.values()) and all(c['passed'] and c['hinge_active_both'] for c in checks) and ratio<.5)
        out['jobs'][job]=row
        print(json.dumps({'job':job,**{k:row[k] for k in ('passed','data_norm','spectral_norm','combined_norm','spectral_to_combined_norm','wall_s')}}),flush=True)
    out['passed']=all(r['passed'] for r in out['jobs'].values())
    out['source_ref']=run.identity(Path(__file__))
    target=ROOT/f'results/{run.RUN}/gradient_audit.json'
    target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output_ref':run.identity(target),'passed':out['passed']}),flush=True)

if __name__=='__main__': audit()
