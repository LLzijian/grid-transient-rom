"""Show both sides of the registered paired edge-count comparison."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def render(source, destination, audit_dir):
    result = json.loads(source.read_text(encoding='utf-8'))
    scores = result['val']
    seeds = (24111, 24121, 24131)
    methods = ('top20','top30','static')
    palettes = ('#28649b','#a65220','#715590')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'osgkd-wp1a',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(7.2047,2.6772))
    fig.subplots_adjust(left=.085,right=.98,top=.90,bottom=.28,wspace=.30)
    for seed,color,marker in zip(seeds,palettes,('o','^','D'),strict=True):
        assert result['full_guards'][str(seed)]['passed']
        points = [scores[f'{method}_{seed}'] for method in methods]
        ref = scores[f'full_{seed}']
        assert all(p['qualified_events']==107 for p in [ref,*points])
        x = [p['complexity']['active_edge_paths'] for p in points]
        assert x==[20,30,45]
        for ax,key in zip(axes,('stable','recall'),strict=True):
            if key=='stable':
                y = [p['scopes']['stable_8s']['risk'] for p in points]
                full = ref['scopes']['stable_8s']['risk']
            else:
                y = [p['autonomous']['unstable_recall_8s']['autonomous_trigger'] for p in points]
                full = ref['autonomous']['unstable_recall_8s']['autonomous_trigger']
            ax.plot(x,y,marker=marker,markersize=4,color=color,lw=1.3,label=str(seed))
            ax.plot([45],[full],marker='s',markersize=6.5,mfc='none',mec=color,ls='none')
    axes[0].set(ylabel='Stable 8 s motion risk',ylim=(.25,1.3))
    axes[1].set(ylabel='8 s instability recall (hits / 12)',ylim=(0,12),yticks=(0,3,6,9,12))
    for tag,ax in zip(('a','b'),axes,strict=True):
        ax.set_title(tag,loc='left',fontweight='bold')
        ax.set(xlim=(18,47),xticks=(20,30,45),xlabel='Active edge paths in the ODE')
        ax.grid(axis='y',color='#d6dde5',lw=.45)
        ax.set_axisbelow(True)
    handles = axes[0].lines[::2]
    handles.append(Line2D([],[],marker='s',mfc='none',mec='#333333',ls='none',label='Full refit'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.012),ncol=4,frameon=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,json_out=audit_dir/'panel_alignment.json',strict=True)
    pdf = destination.with_suffix('.pdf')
    fig.savefig(pdf)
    fig.savefig(destination.with_suffix('.png'),dpi=300)
    svg = destination.with_suffix('.svg')
    fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines())+'\n',
                   encoding='utf-8',newline='\n')
    plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','destination','audit_dir'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    args.audit_dir.mkdir(parents=True,exist_ok=True)
    render(args.source,args.destination,args.audit_dir)
