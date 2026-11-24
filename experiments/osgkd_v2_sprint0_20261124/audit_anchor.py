"""Freeze the amended metrics of the completed flat baseline, without a new rollout."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import torch

from experiments.osgkd_v2_train385_risk_alignment_20261123 import analyze as earlier
from experiments.osgkd_v2_train385_baseline_20261123 import run as baseline
from kiki.evaluation.osgkd_autonomous import summarize_autonomous_8s

RUN = 'osgkd_v2_sprint0_20261124'
SOURCE = 'results/osgkd_v2_fast_zero_sum_20261124/pair01/summary.json'
SOURCE_SHA = 'f0355d3036612e0b5d55e56d29a334296f7f9600a1d971c4fe072a30227d689e'
ANCHOR_SHA = 'bab042b13f2a2619398e9061232966fdd0e6529ae92836ba0add0738e0934688'
AMENDMENT = 'campaigns/osgkd_v2/evaluation_amendment_20261124.md'


def main():
    source_ref = earlier.identity(ROOT / SOURCE)
    earlier.require(source_ref['sha256'] == SOURCE_SHA, 'source result identity differs')
    source = earlier.read(source_ref)
    result = source['result']
    earlier.require(source['status'] == 'completed' and result['fixed_endpoint'] == 2260 and result['valid_endpoint'],
                    'source endpoint differs')
    control = result['treatments']['control']
    checkpoint = control['checkpoint']
    earlier.require(checkpoint['sha256'] == ANCHOR_SHA and '/control/' in checkpoint['path'], 'wrong flat anchor')
    from experiments.osgkd_v2_early_closure_20261123.run import verified
    saved = torch.load(verified(checkpoint), map_location='cpu', weights_only=True)
    from kiki.models.osgkd import OSGKD
    model = OSGKD.from_checkpoint(saved['model'])
    encoder_parameters = sum(p.numel() for group in (model.encoder, model.initial_head) for p in group.parameters())
    earlier.require(saved['attempt'] == 2260 and model.n == 10 and encoder_parameters == 6174, 'wrong flat encoder')

    evaluated = {}
    for split, expected in (('train', (342, 43, 43, 30)), ('val', (95, 22, 12, 7))):
        ref = control['evaluation'][split]
        rows, metrics = earlier.read_score(ref, split)
        earlier.require(earlier.qualification(rows, split), 'anchor numerical qualification differs')
        amended = summarize_autonomous_8s(rows)
        stable, unstable = (amended[key] for key in ('stable_false_alarm_8s', 'unstable_recall_8s'))
        earlier.require((stable['raw_denominator'], unstable['raw_denominator'], unstable['input_eligible_denominator'],
                         unstable['autonomous_trigger']) == expected and stable['autonomous_trigger'] == 0,
                        'anchor autonomous population differs')
        evaluated[split] = {**amended,
            'stable_8s_motion_risk': metrics['splits'][split]['stable_8s']['official']['motion_risk_v1'],
            'data_identity': ref['data_identity'], 'rows_ref': ref['rows_ref'], 'metrics_ref': ref['metrics_ref']}
    output = dict(run_id=RUN, status='completed', anchor=checkpoint, anchor_kind='flat_50d_v1_reference',
        source=source_ref, source_code=earlier.identity(Path(__file__).resolve()), amendment=earlier.identity(ROOT / AMENDMENT),
        evaluator=earlier.identity(ROOT / 'src/kiki/evaluation/osgkd_autonomous.py'),
        encoder_parameters=encoder_parameters, total_parameters=control['parameters'],
        evaluation=evaluated, training_updates=0, rollout_rhs=0, new_fom=0)
    dest = ROOT / 'results' / RUN / 'analysis.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(baseline.atomic_write(dest, output, public=True)))


if __name__ == '__main__':
    main()
