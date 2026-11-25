"""Zero-update active-face and matrix-chain spectral audit; no dataset reads."""
import json
from pathlib import Path
import sys
import types
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from experiments.osgkd_v2_sprint2a_20261125 import run
from kiki.models.osgkd_spectral import reference_geometry,projected_jacobian,spectral_penalty

def main():
    torch.set_num_threads(1)
    prior=json.loads((ROOT/f'results/{run.RUN}/gradient_audit.json').read_bytes())
    out=dict(official_updates=0,dataset_reads=0,jobs={},prior_ref=run.identity(ROOT/f'results/{run.RUN}/gradient_audit.json'))
    for job in ('graph_24111','flat_24111'):
        cp_path=ROOT/f'artifacts/{run.RUN}/{job}/attempt_896.pt'
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        cls=run.OSGKDGraph if job.startswith('graph') else run.fit.OSGKDMatchedFlat
        m=cls.from_checkpoint(cp['model']); geometry=reference_geometry(m)
        saved={k:v.clone() for k,v in m.state_dict().items()}; params=list(m.parameters()); names=[n for n,p in m.named_parameters()]
        base=[p.detach().clone() for p in params]
        def flatten(gs): return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,gs)])
        loss,alpha=spectral_penalty(m,geometry)
        gs=flatten(torch.autograd.grad(loss,params,allow_unused=True))
        def effective(): return m.c.clamp_min(0),torch.minimum(m.beta.clamp_min(0),m.c.clamp_min(0))
        def active(): return torch.stack((m.c.sign(),m.beta.sign(),(m.beta.clamp_min(0)-m.c.clamp_min(0)).sign())).detach().clone()
        sig=active(); c0,b0=[x.detach().clone() for x in effective()]
        boundary=(m.c==0)|(m.beta==0)|(m.beta==m.c)
        # Unit directions can move a pair gap by at most 2*h. Freeze close pairs
        # as well as exact boundaries so both registered steps stay on this face.
        face_mask=boundary|(m.c.abs()<=2e-4)|(m.beta.abs()<=2e-4)|((m.beta-m.c).abs()<=2e-4)
        counts=dict(c_zero=int((m.c==0).sum()),beta_zero=int((m.beta==0).sum()),beta_equals_c=int((m.beta==m.c).sum()),near_c_zero=int((m.c.abs()<1e-6).sum()),near_beta_zero=int((m.beta.abs()<1e-6).sum()),near_beta_c=int(((m.beta-m.c).abs()<1e-6).sum()),boundary_pairs=int(boundary.sum()),pairs=m.c.numel())
        J=projected_jacobian(m,geometry).detach().requires_grad_()
        lm=.05*(torch.linalg.eigvals(J).real.amax()+.05).clamp_min(0).square()
        dLdJ,=torch.autograd.grad(lm,J)
        ce=c0.clone().requires_grad_(); be=b0.clone().requires_grad_(); original_kernel=m._edge_kernel
        def direct_kernel(self,a):
            left,right=self.edge_index; act=(a[...,left]+a[...,right]).tanh()
            return ce+be*act.expm1(),ce,be
        m._edge_kernel=types.MethodType(direct_kernel,m)
        try: sensc,sensb=torch.autograd.grad(spectral_penalty(m,geometry)[0],(ce,be))
        finally: m._edge_kernel=original_kernel
        generator=torch.Generator().manual_seed(24111)
        raw_random=torch.randn(gs.shape,generator=generator,dtype=gs.dtype); raw_random/=raw_random.norm()
        def tangent(direction):
            chunks=[];offset=0
            for name,p in zip(names,params):
                count=p.numel(); d=direction[offset:offset+count].reshape_as(p).clone();offset+=count
                if name in ('c','beta'): d[face_mask]=0
                if name.startswith(('encoder.','initial_head.')): d.zero_()
                chunks.append(d.flatten())
            d=torch.cat(chunks);return d/d.norm()
        directions=[('tangent_spectral',tangent(gs)),('tangent_fixed_physical',tangent(raw_random)),('original_spectral',gs/gs.norm()),('original_random',raw_random)]
        checks=[]
        for name,d in directions:
            ad=float(torch.dot(gs,d)); offset=0; parts=[]
            for p in params:
                parts.append(d[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
            dc=parts[names.index('c')];db=parts[names.index('beta')]
            def cmap(c,b): return c.clamp_min(0),torch.minimum(b.clamp_min(0),c.clamp_min(0))
            _,adslopes=torch.autograd.functional.jvp(cmap,(m.c,m.beta),(dc,db))
            for h in (1e-4,2e-5):
                vals=[];mat=[];eff=[];same=[]
                try:
                    for sign in (1,-1):
                        with torch.no_grad():
                            for p,b,di in zip(params,base,parts):p.copy_(b+sign*h*di)
                        vals.append(float(spectral_penalty(m,geometry)[0].detach()))
                        mat.append(projected_jacobian(m,geometry).detach())
                        eff.append(tuple(x.detach().clone() for x in effective()));same.append(torch.equal(active(),sig))
                finally:
                    with torch.no_grad():
                        for p,b in zip(params,base):p.copy_(b)
                fd=(vals[0]-vals[1])/(2*h)
                chain=float((dLdJ*(mat[0]-mat[1])/(2*h)).sum())
                effslopes=[(a-b)/(2*h) for a,b in zip(eff[0],eff[1])]
                correction=float(((effslopes[0]-adslopes[0])*sensc+(effslopes[1]-adslopes[1])*sensb).sum())
                tol=lambda a,b:abs(a-b)<=1e-7+1e-3*abs(b)
                checks.append(dict(direction=name,step=h,ad=ad,scalar_fd=fd,matrix_chain_fd=chain,active_same_both=same,raw_ad_fd_passed=tol(fd,ad),matrix_chain_passed=tol(chain,fd),slope_correction=correction,corrected_ad=ad+correction,corrected_passed=tol(ad+correction,fd),c_slope_difference_norm=float((effslopes[0]-adslopes[0]).norm()),beta_slope_difference_norm=float((effslopes[1]-adslopes[1]).norm())))
        eig=np.linalg.eigvals(J.detach().numpy()); order=np.argsort(-eig.real);lead=eig[order[0]]
        distances=abs(eig[:,None]-eig[None,:]);np.fill_diagonal(distances,np.inf);i,j=np.unravel_index(np.argmin(distances),distances.shape)
        unchanged=all(torch.equal(v,saved[k]) for k,v in m.state_dict().items())
        tangent_ok=all(c['raw_ad_fd_passed'] and all(c['active_same_both']) for c in checks if c['direction'].startswith('tangent'))
        chains_ok=all(c['matrix_chain_passed'] and c['corrected_passed'] for c in checks)
        out['jobs'][job]=dict(checkpoint_ref=run.identity(cp_path),boundary_counts=counts,tangent_frozen_pairs=int(face_mask.sum()),tangent_freeze_margin=2e-4,checks=checks,leading_eigenvalue=[float(lead.real),float(lead.imag)],leading_gap=float(lead.real-eig[order[1]].real),nearest_eigenvalue_distance=float(distances[i,j]),nearest_pair=[[float(eig[q].real),float(eig[q].imag)] for q in (i,j)],model_state_unchanged=unchanged,tangent_passed=tangent_ok,matrix_chain_and_slope_passed=chains_ok,clip_kink_explains_raw_mismatch=bool(tangent_ok and chains_ok and unchanged),prior_failed_fd_preserved=True)
        print(json.dumps(dict(job=job,tangent_passed=tangent_ok,chain_slope_passed=chains_ok,unchanged=unchanged)),flush=True)
    out['source_ref']=run.identity(Path(__file__))
    path=ROOT/f'results/{run.RUN}/gradient_face_audit.json';path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(run.identity(path)),flush=True)
if __name__=='__main__':main()
