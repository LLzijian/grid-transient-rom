"""Registered .0025-step convergence diagnostic for saved Loop0060 forecasts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_long_latent import BASE, NETWORK, SCALES, _construct
from kiki.models.private_pm import MechanicalROM


HORIZONS = (1, 2, 5, 8)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coi(values, mass):
    return values-(values*mass).sum(axis=-1, keepdims=True)/mass.sum()


def normalized_rmse(old, new, mass):
    difference = old[..., :20]-new[..., :20]
    delta, omega = coi(difference[..., :10], mass), coi(difference[..., 10:20], mass)
    value = np.sqrt(.5*np.mean((delta/SCALES[0])**2+(omega/SCALES[1])**2))
    return float(value) if np.isfinite(value) else None


def summary(rows):
    values = [row['normalized_rmse'] for row in rows if row['normalized_rmse'] is not None]
    return {'n_events': len(rows), 'n_finite': len(values),
            'mean_normalized_rmse': float(np.mean(values)) if values else None,
            'max_normalized_rmse': float(np.max(values)) if values else None,
            'nonfinite': len(values) != len(rows)}


def rk4_0025(model, initial, steps=3200):
    """Integrate cached causal q(0) in one event batch and retain .01 samples."""
    current = torch.as_tensor(initial, dtype=torch.float64)
    saved = [current.detach().cpu().numpy()]
    failure_step = None
    with torch.no_grad():
        for step in range(1, steps+1):
            current = model.rk4(current, .0025)
            if not torch.isfinite(current).all() and failure_step is None:
                failure_step = step
            if step % 4 == 0:
                saved.append(current.detach().cpu().numpy())
    return np.stack(saved, axis=1), failure_step


def labels_and_groups(rows):
    return {row['event_id']: {'label': row['label'], 'group': row['group']} for row in rows}


def comparisons(old, new, ids, truths, metadata, mass):
    result = {'per_event': [], 'summary': {}}
    requested = [(str(h), int(round(h/.01))) for h in HORIZONS]
    requested.append(('prefix', None))
    for name, fixed_steps in requested:
        all_rows, complete_rows = [], []
        for index, (event_id, truth) in enumerate(zip(ids, truths, strict=True)):
            steps = min(len(truth)-1, 800) if fixed_steps is None else fixed_steps
            entry = {'event_id': event_id, **metadata[event_id], 'horizon': name,
                     'steps': steps, 'complete': fixed_steps is None or len(truth) >= fixed_steps+1,
                     'finite_old': bool(np.isfinite(old[index, :steps+1]).all()),
                     'finite_new': bool(np.isfinite(new[index, :steps+1]).all())}
            entry['normalized_rmse'] = normalized_rmse(old[index, :steps+1], new[index, :steps+1], mass)
            all_rows.append(entry)
            if entry['complete']:
                complete_rows.append(entry)
        result['per_event'].extend(all_rows)
        result['summary'][name] = {'all_fixed': summary(all_rows), 'complete': summary(complete_rows)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('challenger', 'teacher'), default='challenger')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    evidence = json.loads((args.evaluation/'evidence.json').read_text(encoding='utf-8'))
    assert evidence['model_sha256'] == digest(args.model_dir/'model.pt'), 'evaluation/source model hash mismatch'
    arrays = np.load(args.evaluation/'trajectories.npz', allow_pickle=False)
    ids = [str(value) for value in arrays['event_ids']]
    old_fine = arrays['fine_predictions'][:, ::2]
    initial = arrays['fine_predictions'][:, 0]
    truths = [arrays[f'{event_id}_truth'] for event_id in ids]
    metadata = labels_and_groups(json.loads((args.evaluation/'per_event_rows.json').read_text(encoding='utf-8')))
    assert set(ids) == set(metadata)
    base, network = MechanicalROM.load(BASE), dict(np.load(NETWORK, allow_pickle=False))
    network['observed_buses'] = np.unique(np.r_[network['load_buses'], network['generator_buses']])
    network['load_positions'] = np.searchsorted(network['observed_buses'], network['load_buses'])
    model = _construct(args.model_dir, base, network, model_name=args.model)
    new_fine, failure_step = rk4_0025(model, initial)
    assert new_fine.shape == old_fine.shape
    compared = comparisons(old_fine, new_fine, ids, truths, metadata, base.mass)
    np.savez_compressed(args.output/'step_0025_predictions.npz', event_ids=np.array(ids), predictions=new_fine)
    finite = bool(np.isfinite(new_fine).all())
    result = {'model_dir': str(args.model_dir), 'evaluation': str(args.evaluation), 'model_kind': args.model,
              'source_model_sha256': digest(args.model_dir/'model.pt'), 'evaluation_model_sha256': evidence['model_sha256'],
              'code_sha256': digest(__file__), 'dt_old': .005, 'dt_new': .0025, 'output_dt': .01,
              'initial_source': 'evaluation fine_predictions[:,0] (cached causal state)',
              'event_count': len(ids), 'finite_predictions': finite, 'nonfinite_failure_step': failure_step,
              'comparison': compared, 'test_read': False}
    (args.output/'evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'finite_predictions': finite, 'nonfinite_failure_step': failure_step,
                      'horizons': compared['summary']}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
