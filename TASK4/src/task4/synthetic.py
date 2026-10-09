"""Synthetic engineering fixtures, explicitly NOT IEEE39 or performance evidence."""
import numpy as np
from .data import Physics,Event
from .objectives import m3_numpy

def fixture(n=3,events=4,steps=31):
    d=np.zeros((n,n))
    for i in range(n):d[i,i]=1;d[(i+1)%n,i]=-1
    p=Physics(np.linspace(.8,1.2,n),d,np.linspace(-.1,.1,n),np.ones(n),np.ones(n)*.5,
              np.ones(n)*.2,np.ones(n)*.6,np.ones(n)*.3,omega_s=2*np.pi*60)
    rng=np.random.default_rng(17);out=[]
    for k in range(events):
        y0=rng.normal(size=(n,3))*np.array([.08,.04,.02])
        y=m3_numpy(p,y0,steps-1)
        torque=[]
        for yy in y:
            el=((np.sin((yy[:,0]+p.delta_star)@d)-np.sin(p.delta_star@d))*p.coupling)@d.T
            torque.append(el+p.damping*yy[:,1])
        out.append(Event(f'synthetic_{k}',str(k),'stable',np.arange(steps)*.01,y,np.asarray(torque),'synthetic_M3_fixture'))
    return out,p
