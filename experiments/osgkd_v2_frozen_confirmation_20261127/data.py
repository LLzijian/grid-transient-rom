"""Frozen confirmation adapters; this module neither authorizes nor opens test on import."""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from kiki.data import osgkd as d
from kiki.data import osgkd_train_val as current


SHARDS = ('ieee39_expansion', 'ieee39_holdout_supplement')
TEST_COUNTS = {'ieee39_expansion': 45, 'ieee39_holdout_supplement': 41}
SOURCE_METADATA = {
    'ieee39_expansion': current.SOURCE_METADATA['ieee39_expansion'],
    'ieee39_holdout_supplement': (
        'd1f3df87964e048501a8a257eda57f2584f19631747fc194ce41b8c659a60a7a',
        '6edcddcf90cca620d7acb0b2629d179a066ce3dde7eda3a24947aae816b1ac58',
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _metadata_file(root, shard, name, expected):
    relative = Path('private_data')/shard/name
    path = (root/relative).resolve()
    require(path.is_relative_to(root), 'metadata path escapes supplied root')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == expected, f'frozen {shard}/{name} identity differs')
    return raw, dict(path=relative.as_posix(), bytes=len(raw), sha256=digest)


def _scenario(row):
    return (int(row['fault_bus']), float(row['fault_duration_s']), float(row['fault_xf']))


def read_source_metadata(root):
    """Read ONLY the two frozen splits.json and dataset_index.csv files.

    No H5/NPZ path is opened, hashed or inspected. The returned private whitelist
    contains every submitted test ID, without exposing train/val trajectories.
    """
    root = Path(root).resolve()
    partitions = {key: set() for key in ('train', 'validation', 'test')}
    all_ids, scenarios, rows, sources = set(), set(), {}, {}
    for shard in SHARDS:
        split_sha, index_sha = SOURCE_METADATA[shard]
        split_raw, split_ref = _metadata_file(root, shard, 'splits.json', split_sha)
        index_raw, index_ref = _metadata_file(root, shard, 'dataset_index.csv', index_sha)
        split = json.loads(split_raw)
        require(set(partitions) <= set(split), 'official partition keys differ')
        local_ids = set()
        for name in partitions:
            ids = split[name]
            require(isinstance(ids, list) and all(isinstance(eid, str) and eid for eid in ids), 'invalid official IDs')
            require(len(set(ids)) == len(ids) and not local_ids.intersection(ids), 'within-source split overlap')
            local_ids.update(ids)
            partitions[name].update(ids)
        require(not all_ids.intersection(local_ids), 'cross-source ID overlap')
        all_ids.update(local_ids)
        require(len(split['test']) == TEST_COUNTS[shard], 'frozen source test denominator differs')
        if shard == 'ieee39_holdout_supplement':
            require(not split['train'] and not split['validation'], 'supplement must be test-only')
        descriptions = list(csv.DictReader(io.StringIO(index_raw.decode('utf-8-sig'), newline='')))
        indexed = {row['event_id']: row for row in descriptions}
        require(len(indexed) == len(descriptions) and set(indexed) == local_ids, 'source index/split coverage differs')
        local_scenarios = {_scenario(row) for row in descriptions}
        require(len(local_scenarios) == len(descriptions) and not scenarios.intersection(local_scenarios),
                'duplicate physical event identity')
        scenarios.update(local_scenarios)
        for eid in split['test']:
            row = indexed[eid]
            require(row['label'] in ('stable', 'unstable'), 'invalid test label')
            rows[eid] = {**row, 'source_root': shard}
        sources[shard] = dict(split_ref=split_ref, index_ref=index_ref, test_ids=sorted(split['test']),
                              test_count=len(split['test']))
    require(all(not partitions[a].intersection(partitions[b]) for a, b in
                (('train', 'validation'), ('train', 'test'), ('validation', 'test'))), 'global split overlap')
    require(set(rows) == partitions['test'] and len(rows) == 86, 'complete submitted test whitelist required')
    return dict(sources=sources, rows=rows, event_ids=sorted(rows), counts=TEST_COUNTS.copy(),
                total=86, id_and_scenario_disjoint=True, trajectory_files_opened=0)


def _system_reference(file, constants, previous):
    """The same system/reference/controls gates as current.load_val."""
    config = json.loads(file.attrs['config'])
    for source, key in (('omega_s', 'omega_s'), ('mva_base', 'mva_base'),
                        ('case_sha256', 'source_case_sha256'), ('andes_version', 'source_andes_version'),
                        ('raw_units', 'raw_units'), ('postfault_signature', 'postfault_signature')):
        require(file.attrs[source] == constants[key], f'confirmation {source} differs')
    require(config['sample_interval_s'] == d.DT and config['integration_step_s'] == .005
            and config['fault_type'] == current.FAULT
            and config['disable_builtin_timed_events'] == constants['common_controls']['disable_builtin_timed_events']
            and config['disable_pss'] == constants['common_controls']['disable_pss'], 'confirmation controls differ')
    mass = np.asarray(constants['mass'])
    reference = {key: file['equilibrium'][key][...] for key in ('delta', 'omega', 'Pm')}
    require(all(value.shape == (10,) and np.isfinite(value).all() for value in reference.values()),
            'invalid confirmation equilibrium')
    require(hashlib.sha256(np.asarray(reference['Pm'], dtype='<f8').tobytes(order='C')).hexdigest()
            == current.PM_REFERENCE_SHA256, 'confirmation/train common torque reference differs')
    if previous is not None:
        require(all(np.array_equal(value, previous[key]) for key, value in reference.items()),
                'confirmation shards have different fixed references')
    delta = reference['delta']
    mapping = json.loads(file.attrs['state_mapping'])
    generator_states = [[state, 'GENROU', generator]
                        for state in ('delta', 'omega') for generator in constants['generator_order']]
    require([entry[1:] for entry in mapping[:20]] == generator_states, 'confirmation generator order differs')
    require(np.allclose(file['inertia'][...], mass*constants['omega_s'], rtol=1e-12, atol=1e-10)
            and np.allclose(delta-delta@(mass/mass.sum()), constants['theta_ref'], rtol=0, atol=1e-12)
            and np.allclose(reference['omega'], 1., rtol=0, atol=1e-12), 'confirmation common reference differs')
    return reference


def _convert(eid, row, raw, reference, constants):
    t = raw['t']
    require(t.ndim == 1 and len(t) == int(row['valid_length']) and len(t) > 0
            and np.isfinite(t).all() and abs(t[0]) < 1e-9
            and np.allclose(np.diff(t), d.DT, atol=1e-9, rtol=0)
            and abs(t[-1]-float(row['trajectory_end_s'])) < 1e-8, 'confirmation timing differs')
    require(all(raw[key].shape == (len(t), 10) and np.isfinite(raw[key]).all()
                for key in ('delta', 'omega', 'Pm', 'te')), 'invalid confirmation arrays')
    require(row['label'] in ('stable', 'unstable') and row['fault_type'] == current.FAULT
            and row['postfault_signature'] == constants['postfault_signature'], 'confirmation source labels differ')
    meta = current._event_meta(row, t[-1])
    require(row['label'] != 'unstable' or meta['true_event_time_record_s'] is not None
            and abs(meta['true_event_time_record_s']-t[-1]) < 1e-8, 'confirmation event time differs')
    group = f"{current.FAULT}|bus:{int(row['fault_bus'])}"
    if len(t) < 11:
        return dict(id=eid, group=group, label=row['label'], history=None, metadata=meta,
                    missing_reason='fewer_than_11_original_history_points', original_points=len(t))
    return d.event_record(eid, group, row['label'], t, raw['delta'],
                          (raw['omega']-1)*constants['omega_s'], raw['Pm']-reference['Pm'],
                          raw['Pm']-reference['Pm'], raw['te']-reference['Pm'], meta, constants['mass'])


def load_allowed_events(root, metadata, constants, on_read):
    """AUTHORIZED MAIN ONLY: open each submitted H5 once and cache all events.

    This adapter supplies no approval, lock, retry, resume or authorization flow.
    A mandatory callback receives pre-open/pre-read records as well as completed
    reads, so the caller can seal an honest private receipt even after a failure.
    """
    import h5py
    require(callable(on_read), 'persistent read-accounting callback required')
    root = Path(root).resolve()
    require(read_source_metadata(root) == metadata, 'submitted metadata whitelist differs')
    events, reference = [], None
    for shard in SHARDS:
        relative = Path('private_data')/shard/'dataset.h5'
        path = (root/relative).resolve()
        require(path.is_relative_to(root), 'trajectory path escapes supplied root')
        on_read(dict(phase='before_open_shard', shard=shard, path=relative.as_posix()))
        with h5py.File(path, 'r') as file:
            on_read(dict(phase='opened_shard', shard=shard, path=relative.as_posix()))
            reference = _system_reference(file, constants, reference)
            for eid in metadata['sources'][shard]['test_ids']:
                row = metadata['rows'][eid]
                require(row['source_root'] == shard, 'event source differs from whitelist')
                on_read(dict(phase='before_read_event', shard=shard, event_id=eid))
                group = file['events'][eid]
                raw = {key: group[key][...] for key in ('t', 'delta', 'omega', 'Pm', 'te')}
                on_read(dict(phase='read_event', shard=shard, event_id=eid))
                events.append(_convert(eid, row, raw, reference, constants))
    events.sort(key=lambda event: event['id'])
    require([event['id'] for event in events] == metadata['event_ids'], 'read event coverage differs from whitelist')
    return events
