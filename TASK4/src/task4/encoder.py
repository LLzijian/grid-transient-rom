"""One shared GRU per node; prediction head is TRAIN-only and cannot bypass q."""
import torch
from torch import nn

class HistoryEncoder(nn.Module):
    def __init__(self,model,hidden=32):
        super().__init__();self.n=model.n;self.rank=model.rank
        for name in ['mass','adjacency','scales','mu']:
            self.register_buffer(name,getattr(model,name).detach().clone())
        # relative angle, omega, neighbor angle difference, neighbor omega difference,
        # common omega, and 4 known physical parameter features. No p or event IDs.
        self.gru=nn.GRU(9,hidden,batch_first=True).double();self.head=nn.Linear(hidden,self.rank).double()
    def _features(self,h):
        d,w=h[...,0],h[...,1]
        dc=d-(d*self.mass).sum(-1,keepdim=True)/self.mass.sum()
        common=(w*self.mass).sum(-1,keepdim=True)/self.mass.sum()
        nd=d@self.adjacency.T-d;nw=w@self.adjacency.T-w
        f=torch.stack([dc/self.scales[0],w/self.scales[1],nd/self.scales[0],nw/self.scales[1],common.expand_as(w)/self.scales[1]],-1)
        mu=self.mu.expand(*f.shape[:-2],-1,-1)
        return torch.cat([f,mu],-1)
    def _raw(self,h):
        f=self._features(h);b,t,n,k=f.shape
        seq=f.permute(0,2,1,3).reshape(b*n,t,k)
        out=self.head(self.gru(seq)[0][:,-1])
        return out.reshape(b,n,self.rank)
    def forward(self,h):
        if h.shape[1]!=11 or h.shape[2:]!=(self.n,3):raise ValueError('Expected [batch,11,N,3]')
        # The third physical channel is deliberately ignored by _features.
        return self._raw(h)-self._raw(torch.zeros_like(h))

class PredictiveHead(nn.Module):
    def __init__(self,model,horizons,hidden=32):
        super().__init__();self.horizons=list(horizons)
        self.register_buffer('adjacency',model.adjacency.detach().clone())
        self.register_buffer('mu',model.mu.detach().clone())
        self.net=nn.Sequential(nn.Linear(2*model.rank+4,hidden),nn.Tanh(),nn.Linear(hidden,3*len(horizons))).double()
    def forward(self,q):
        neighbor=torch.einsum('ij,bjr->bir',self.adjacency,q)
        inp=torch.cat([q,neighbor,self.mu.expand(q.shape[0],-1,-1)],-1)
        return self.net(inp).reshape(q.shape[0],q.shape[1],len(self.horizons),3)
