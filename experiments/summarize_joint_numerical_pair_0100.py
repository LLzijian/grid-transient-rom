"""Read existing 0100 JSON artifacts; no model loading, integration or TEST access.

Incomplete live files are reported as unavailable. This extractor checks recorded
identities and preserves numerical results without selecting a scientific endpoint.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from summarize_joint_adequacy_0099 import _monitor_summary_from_dict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'private_data/task2_joint_numerical_pair_0100'
INITIAL = ROOT/'private_data/task2_joint_adequacy_0099/formal_L10-S/monitor_step0600/evidence.json'
FIELDS = ('step', 'new_step', 'sample_ids', 'endpoints', 'horizon', 'coverage_before',
          'coverage_after', 'numpy_rng_before_sha256', 'numpy_rng_after_sha256')


class Reader:
    def __init__(self):
        self.inputs = {}

    def read(self, path, *, lines=False):
        path = Path(path)
        info = {'path': str(path), 'available': False}
        self.inputs[str(path)] = info
        try:
            raw = path.read_bytes()
            info.update(available=True, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            text = raw.decode('utf-8-sig')
            if not lines:
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError('expected JSON object')
                return value
            result, errors = [], []
            for n, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError('expected object')
                    result.append(item)
                except (ValueError, TypeError):
                    errors.append(n)
            info['parse_error_lines'] = errors
            info['parsed_rows'] = len(result)
            return result
        except (OSError, UnicodeError, ValueError) as exc:
            info.update(available=False, reason=type(exc).__name__, detail=str(exc))
            return [] if lines else None


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def statistics(values):
    finite_values = [v for v in values if finite(v)]
    complete = bool(values) and len(values) == len(finite_values)
    return {'rows': len(values), 'finite_rows': len(finite_values),
            'mean': sum(values)/len(values) if complete else None,
            'minimum': min(values) if complete else None, 'maximum': max(values) if complete else None,
            'finite_diagnostic_mean': sum(finite_values)/len(finite_values) if finite_values else None}


def row_index(rows):
    counts = Counter(r.get('step') for r in rows)
    return ({r['step']: r for r in rows if isinstance(r.get('step'), int)},
            [s for s,n in counts.items() if n > 1])


def compare_rows(a, b):
    return [k for k in FIELDS if k not in a or k not in b or a[k] != b[k]]


def paired_identity(arms, expected_steps=100):
    expected = list(range(1168,1168+expected_steps))
    problems, maps = [], {}
    for arm, data in arms.items():
        maps[arm] = {}
        for kind in ('samples','trace'):
            rows = data[kind]
            idx, duplicate = row_index(rows)
            maps[arm][kind] = idx
            if duplicate:
                problems.append({'arm':arm,'kind':kind,'duplicate_steps':duplicate})
            sequence = [r.get('step') for r in rows]
            if sequence != sorted(idx):
                problems.append({'arm':arm,'kind':kind,'reason':'nonunique or out-of-order sequence'})
            for step, row in idx.items():
                absent = [k for k in FIELDS if k not in row]
                if absent:
                    problems.append({'arm':arm,'kind':kind,'step':step,'missing_fields':absent})
                invalid = []
                if len(row.get('sample_ids') or []) != 12 or len(row.get('endpoints') or []) != 12:
                    invalid.append('12 sample IDs/endpoints')
                for name in ('coverage_before','coverage_after'):
                    coverage = row.get(name) or {}
                    if (sorted(coverage.get('cover_order') or []) != list(range(346)) or
                            not isinstance(coverage.get('cursor'),int) or not 0 <= coverage.get('cursor',-1) <= 346):
                        invalid.append(name)
                for name in ('numpy_rng_before_sha256','numpy_rng_after_sha256'):
                    value = row.get(name)
                    if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                        invalid.append(name)
                if invalid:
                    problems.append({'arm':arm,'kind':kind,'step':step,'invalid_identity_fields':invalid})
                if step not in expected or row.get('new_step') != step-1167:
                    problems.append({'arm':arm,'kind':kind,'step':step,'reason':'outside registered sequence'})
                if row.get('horizon') != (1,2,5)[(step-301)%3]:
                    problems.append({'arm':arm,'kind':kind,'step':step,'reason':'absolute horizon mismatch'})
                if row.get('training_dt') != {'dt10':.01,'dt5':.005}[arm]:
                    problems.append({'arm':arm,'kind':kind,'step':step,'reason':'training grid mismatch'})
                prev = idx.get(step-1)
                if prev and (prev.get('coverage_after') != row.get('coverage_before') or
                             prev.get('numpy_rng_after_sha256') != row.get('numpy_rng_before_sha256')):
                    problems.append({'arm':arm,'kind':kind,'step':step,'reason':'coverage/RNG chain mismatch'})
        for step in sorted(maps[arm]['trace']):
            left = maps[arm]['samples'].get(step)
            differences = compare_rows(left or {}, maps[arm]['trace'][step])
            if differences:
                problems.append({'arm':arm,'step':step,'reason':'samples-versus-committed-trace','fields':differences})
    checked = {}
    for kind in ('samples','trace'):
        left, right = maps['dt10'][kind], maps['dt5'][kind]
        common = sorted(left.keys() & right.keys())
        for step in common:
            fields = compare_rows(left[step],right[step])
            if fields:
                problems.append({'kind':kind,'step':step,'reason':'paired identity mismatch','fields':fields})
        checked[kind] = {'paired_rows':len(common),'paired_steps':common,
                         'missing_dt10':[s for s in expected if s not in left],
                         'missing_dt5':[s for s in expected if s not in right]}
    complete = all(not c['missing_dt10'] and not c['missing_dt5'] for c in checked.values())
    parse_errors = any(data.get('parse_errors') for data in arms.values())
    return {'expected_steps':expected_steps,'absolute_first':1168,'absolute_last':1167+expected_steps,
            'fields_compared':list(FIELDS),'status':'mismatch' if problems or parse_errors else ('complete_match' if complete else 'partial_match'),
            'complete_match':complete and not problems and not parse_errors,'checked':checked,'problems':problems,
            'parse_errors_present':parse_errors,
            'rng_scope':'recorded NumPy Generator hashes; no per-step Torch RNG fingerprint was recorded'}


def monitor_summary(row, path, step):
    if row is None:
        return {'available':False,'step':step,'path':str(path)}
    result = _monitor_summary_from_dict(row, Path(path), label=f'new_step{step}')
    result['step'] = step
    result['numeric_failures'] = {}
    for grid in ('initial_grid_qa','final_grid_qa'):
        qa = row.get(grid) or {}
        result['numeric_failures'][grid] = {k:[r for r in v if not r.get('pass')]
            for k,v in qa.items() if isinstance(v,list) and all(isinstance(r,dict) for r in v)}
        result['numeric_failures'][grid]['other_gate_fields'] = {k:v for k,v in qa.items() if not isinstance(v,list)}
        terms = qa.get('term_pass') or []
        names = qa.get('term_names') or []
        differences = qa.get('term_absolute_differences') or []
        result['numeric_failures'][grid]['objective_term_failures'] = [
            {'batch_index':bi,'horizon':(1,2,5)[hi], 'term':names[ti] if ti<len(names) else ti,
             'absolute_difference':differences[bi][hi][ti] if differences else None}
            for bi,batch in enumerate(terms) for hi,h in enumerate(batch)
            for ti,passed in enumerate(h) if not passed]
    result['event_summary'] = row.get('event_summary')
    result['legacy_component_loop'] = row.get('loop')
    return result


def evaluation(reader, data_root, arm):
    base = data_root/f'eval_{arm}'
    paths = {'wrapper':base/'evidence.json', 'validation':base/'validation/trajectories/evidence.json',
             'modal':base/'validation/modal/evidence.json','power':base/'validation/evidence.json',
             'train':base/'train/evidence.json', 'train_per_event':base/'train/per_event_rows.json'}
    values = {k:reader.read(p) for k,p in paths.items()}
    val, train = values['validation'] or {}, values['train'] or {}
    modal, power = values['modal'] or {}, values['power'] or {}
    # Keep all stored metrics, including denominators, severe rows, all modal pairs,
    # power QA and adverse classifications. Do not reconstruct missing evaluator results.
    return {'path':str(base),'available':any(v is not None for v in values.values()),
            'missing_components':[k for k,v in values.items() if v is None],
            'component_paths':{k:str(p) for k,p in paths.items()},
            'metrics':{'validation_complete':val.get('summary'),
                'validation_severe_prefix':val.get('unstable_available_prefix'),
                'validation_common_frequency':val.get('common_frequency'),
                'validation_numerical_gate_pass':val.get('model_numerical_gate_pass'),
                'l1_numerically_gated_counts':val.get('l1_numerically_gated_counts'),
                'train_complete':train.get('summary'),'train_denominators':train.get('denominators'),
                'train_complete_counts':train.get('complete_counts'),
                'train_all_unstable_prefix_rows':train.get('unstable_prefix_event_rows'),
                'modal_matches':modal.get('matched_count'),'modal_qa':modal.get('qa'),
                'local_alpha_per_s':modal.get('spectral_alpha'),'power_qa':power.get('energy_power_qa')},
            'stored_evidence':values,
            'charged_seconds':(values['wrapper'] or {}).get('seconds'),
            'timing_note':'Only outer wrapper duration is charged; nested durations overlap.'}


def build_report(data_root=DATA, expected_steps=100):
    started = time.perf_counter()
    reader = Reader()
    data_root = Path(data_root)
    card = ROOT/'reports/task2_joint_numerical_pair_0100.md'
    # Text preregistration is hashed independently; it is not a JSON input.
    card_hash = hashlib.sha256(card.read_bytes()).hexdigest()
    arms, raw = {}, {}
    for arm in ('dt10','dt5'):
        base = data_root/f'formal_{arm}'
        formal = reader.read(base/'evidence.json')
        sample_rows = reader.read(base/'samples.jsonl',lines=True)
        trace = reader.read(base/'trace.jsonl',lines=True)
        parse = {kind:reader.inputs[str(base/f'{kind}.jsonl')].get('parse_error_lines',[])
                 for kind in ('samples','trace')}
        raw[arm] = {'samples':sample_rows,'trace':trace,'parse_errors':any(parse.values())}
        restore = reader.read(data_root/f'qa_restore_{arm}/evidence.json')
        cached = (formal or restore or {}).get('cache_audit') or {}
        monitors = {}
        for step in (0,100,200):
            row = ((formal or {}).get('monitors') or {}).get(str(step))
            path = base/'evidence.json'
            if row is None:
                path = base/f'monitor_step{step:04d}/evidence.json'
                row = reader.read(path)
            if row is None and step == 0 and cached.get('accepted'):
                # Only the explicitly pinned old initial monitor can be reused.
                path = INITIAL
                row = reader.read(path)
                if row is not None and cached.get('sha256') != reader.inputs[str(path)]['sha256']:
                    row = None
            monitors[str(step)] = monitor_summary(row,path,step)
        checkpoints = {name:reader.read(base/name/'checkpoint.json')
                       for name in ('step0000','step1267','step1367')}
        arms[arm] = {'formal_evidence':formal,'formal_evidence_available':formal is not None,
            'state_note':'No formal evidence yet: may be running or interrupted; not inferred as failure.' if formal is None else None,
            'trace':{'rows':len(trace),'first_step':trace[0].get('step') if trace else None,
                     'last_step':trace[-1].get('step') if trace else None,'parse_errors':parse,
                     'loss':statistics([r.get('loss') for r in trace]),
                     'gradient_norm':statistics([r.get('gradient_norm') for r in trace]),
                     'recorded_clipped_count':sum(r.get('gradient_clipped') is True for r in trace),
                     'step_seconds':statistics([r.get('seconds') for r in trace]),
                     'step_seconds_by_horizon':{str(h):statistics([r.get('seconds') for r in trace if r.get('horizon')==h]) for h in (1,2,5)},
                     'summed_step_seconds':sum(r['seconds'] for r in trace) if all(finite(r.get('seconds')) for r in trace) else None,
                     'latest_process_seconds':trace[-1].get('elapsed_seconds') if trace else None},
            'monitors':monitors,'milestone_metadata':checkpoints,
            'evaluation':evaluation(reader,data_root,arm)}
    result = {'loop':'0100','scope':'Factual read-only extraction; no route decision, promotion or endpoint selection.',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'data_root':str(data_root),
        'preregistration':{'path':str(card),'sha256':card_hash},
        'pair_identity':paired_identity(raw,expected_steps),'arms':arms,
        'execution_costs':reader.read(data_root/'execution_costs.json'),
        'formal_launch':reader.read(data_root/'formal_launch.json'),
        'input_snapshots':list(reader.inputs.values()),'extractor_seconds':time.perf_counter()-started,
        'raw_data_read':False,'test_read':False,'model_loaded':False,'integration_run':False,
        'snapshot_note':'Sequential byte snapshots of live artifacts; differing row counts can reflect concurrent progress.'}
    return result


def write_report(data_root=DATA, output=None, expected_steps=100):
    data_root = Path(data_root).resolve()
    output = Path(output or data_root/'derived'/f'capability_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.json').resolve()
    allowed = DATA.resolve()/'derived'
    if not output.is_relative_to(allowed):
        raise ValueError('output must be a new artifact under 0100/derived')
    if output.exists():
        raise FileExistsError(output)
    report = build_report(data_root,expected_steps)
    report['extractor_code'] = {'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return output,report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--expected-steps',type=int,choices=(100,200),default=100)
    args = parser.parse_args()
    output,report = write_report(**vars(args))
    print(json.dumps({'output':str(output),'pair_status':report['pair_identity']['status'],
                      'trace_rows':{k:v['trace']['rows'] for k,v in report['arms'].items()},
                      'extractor_seconds':report['extractor_seconds']}))
