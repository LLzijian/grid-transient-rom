"""Render the frozen known-state torque conditioning grid with editable text."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def render(source, destination):
    sections = json.loads(source.read_text(encoding='utf-8'))['negative_controls']
    rows = sections['condition_map']['rows']
    excitations = sorted({r['excitation'] for r in rows})
    overlaps = sorted({r['environmental_waveform_overlap'] for r in rows})
    assert len(rows) == len(excitations)*len(overlaps) == 20
    indexed = {(r['excitation'], r['environmental_waveform_overlap']):r for r in rows}
    values = np.asarray([[100*indexed[e,o]['recovery_curve'][1]['parameter_relative_l2']
                          for o in overlaps] for e in excitations])
    assert np.isfinite(values).all() and values.min()>0
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
                         'svg.hashsalt':'osgkd-torque-identifiability',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(7.2047, 3.9370), layout='constrained')  # 183 × 100 mm
    coords = np.arange(len(overlaps)+1)-.5, np.arange(len(excitations)+1)-.5
    tiles = ax.pcolormesh(*coords,values,cmap='Blues',norm=LogNorm(vmin=.5,vmax=100),
                         edgecolors='white',linewidth=.65)
    for y, eps in enumerate(excitations):
        for x, rho in enumerate(overlaps):
            row = indexed[eps,rho]
            svd = row['svd']
            rank = svd['rank']
            minimum = svd['singular_values'][-1] if rank==57 else 0.
            c = 'white' if values[y,x]>=18 else '#14243a'
            ax.text(x,y+.12,f'Rank {rank}/57',ha='center',va='center',fontweight='bold',color=c)
            ax.text(x,y-.13,f'min σ {minimum:.3g}',ha='center',va='center',color=c)
    ax.set(xticks=range(len(overlaps)),xticklabels=[f'{o:g}' for o in overlaps],
           yticks=range(len(excitations)),yticklabels=[f'{e:g}' for e in excitations],
           xlabel='Environmental waveform overlap',ylabel='Angle excitation amplitude',
           title='Known-state torque identifiability')
    ax.tick_params(length=0,pad=4)
    ax.set_xlim(-.5,len(overlaps)-.5)
    ax.set_ylim(-.5,len(excitations)-.5)
    bar = fig.colorbar(tiles, ax=ax, fraction=.05, pad=.035, ticks=[.5,1,5,10,50,100])
    bar.ax.set_yticklabels(['0.5','1','5','10','50','100'])
    bar.set_label('Relative parameter error (%) at noise SD 0.01')
    destination.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.pdf','.png'):
        fig.savefig(destination.with_suffix(suffix), dpi=300)
    svg = destination.with_suffix('.svg')
    fig.savefig(svg, metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
                   encoding='utf-8', newline='\n')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    arguments=parser.parse_args()
    render(arguments.source,arguments.destination)
