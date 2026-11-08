"""Raw electrical load/loss balance explains the generator-only net residual."""
import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import andes
import h5py
import numpy as np
from andes.shared import matrix as dense_matrix
from generate_private_dataset import physical_parameters

from kiki.models.private_pm import MechanicalROM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.getLogger('andes').setLevel(logging.ERROR)
    config = json.loads(Path('configs/private_dataset.json').read_text())
    system = andes.load(andes.get_case(config['case']), setup=False, no_output=True, default_config=True)
    for model in system.models.values():
        if model.group == 'TimedEvent' and model.n:
            model.u.v[:] = [0]*model.n
    system.setup()
    assert system.PFlow.run()
    system.TDS.config.no_tqdm = 1
    system.TDS.init()
    assert system.TDS.test_ok
    pq_config = {k: float(getattr(system.PQ.config, k)) for k in ('p2p', 'p2i', 'p2z', 'q2q', 'q2i', 'q2z')}
    assert pq_config == {'p2p': 0., 'p2i': 0., 'p2z': 1., 'q2q': 0., 'q2i': 0., 'q2z': 1.}
    with h5py.File('private_data/ieee39_postfault/dataset.h5', 'r') as file:
        assert physical_parameters(system) == file.attrs['postfault_signature']
        assert hashlib.sha256(Path(andes.get_case(config['case'])).read_bytes()).hexdigest() == file.attrs['case_sha256']
        eq_errors = [float(np.max(abs(system.dae.x-file['equilibrium/full_state'][:]))),
                     float(np.max(abs(system.dae.y-file['equilibrium/algebraic_state'][:])))]
    assert max(eq_errors) < 1e-8
    bus_ids = list(system.Bus.idx.v)
    generator_buses = [bus_ids.index(bus) for bus in system.GENROU.bus.v]
    load_buses = [bus_ids.index(bus) for bus in system.PQ.bus.v]
    load_g = system.PQ.u.v*system.PQ.Req.v
    shunt_buses = [bus_ids.index(bus) for bus in system.Shunt.bus.v]
    shunt_g = system.Shunt.u.v*system.Shunt.g.v
    yline = np.asarray(dense_matrix(system.Line.build_y()))
    incidence = np.zeros((len(bus_ids), system.Line.n))
    for k, (a, b, status) in enumerate(zip(system.Line.bus1.v, system.Line.bus2.v, system.Line.u.v, strict=True)):
        if status:
            incidence[bus_ids.index(a), k] = 1.
            incidence[bus_ids.index(b), k] = -1.
    ports = incidence[generator_buses]
    base = MechanicalROM.load('private_data/task2_m3_0020/model.npz')
    def components(voltage, angle, current_d, current_q):
        phasor = voltage*np.exp(1j*angle)
        line_loss = np.sum((phasor*np.conj(phasor@yline.T)).real, axis=-1)
        load = voltage[..., load_buses]**2@load_g
        shunt = voltage[..., shunt_buses]**2@shunt_g
        copper = (current_d**2+current_q**2)@system.GENROU.ra.v
        return np.stack((load, line_loss, shunt, copper), axis=-1)
    eq_components = components(system.Bus.v.v, system.Bus.a.v, system.GENROU.Id.v, system.GENROU.Iq.v)
    rows, arrays, moments = [], {}, np.zeros((5, 5))
    atlas = json.loads(Path('private_data/task2_residual_atlas_0040/evidence.json').read_text())
    for root in (Path('private_data/ieee39_postfault'), Path('private_data/ieee39_expansion')):
        split = json.loads((root/'splits.json').read_text())
        with h5py.File(root/'dataset.h5', 'r') as file:
            assert list(json.loads(file.attrs['algebraic_names'])) == system.dae.y_name
            for eid in split['train']:
                raw = file['events'][eid]
                algebraic = raw['algebraic_state'][:801]
                current_d, current_q = algebraic[:, system.GENROU.Id.a], algebraic[:, system.GENROU.Iq.a]
                physical = components(raw['bus_v'][:801], raw['bus_a'][:801], current_d, current_q)
                pe, te = raw['Pe'][:801], raw['te'][:801]
                torque_error = np.sqrt(np.mean((te-pe-system.GENROU.ra.v*(current_d**2+current_q**2))**2))/max(1., np.sqrt(np.mean(te**2)))
                power_error = np.sqrt(np.mean((pe.sum(axis=1)-physical[:, :3].sum(axis=1))**2))/max(1., np.sqrt(np.mean(pe.sum(axis=1)**2)))
                omega = (raw['omega'][:801]-1)*float(file.attrs['omega_s'])
                damping = omega@base.damping
                channels = np.column_stack((-(physical-eq_components), damping))
                net = base.pm_star.sum()-te.sum(axis=1)+damping
                net_error = np.sqrt(np.mean((net-channels.sum(axis=1))**2))/max(1., np.sqrt(np.mean(te.sum(axis=1)**2)))
                assert np.isfinite(channels).all()
                moments += channels.T@channels/len(channels)
                load_error = float(np.mean((net-channels[:, 0])**2))
                rows.append({'event_id': eid, 'group': str(raw.attrs['fault_bus']), 'label': raw.attrs['label'],
                    'n': len(channels), 'torque_error': float(torque_error), 'power_error': float(power_error),
                    'net_error': float(net_error), 'net_energy': float(np.mean(net**2)), 'load_only_error': load_error})
                arrays[eid+'_channels'] = channels
                arrays[eid+'_net'] = net
                arrays[eid+'_t'] = raw['t'][:801]
    assert len(rows) == 369
    errors = {key: np.quantile([r[key] for r in rows], [.5, .95, 1]).tolist() for key in ('torque_error', 'power_error', 'net_error')}
    passed = all(value[0] < 1e-4 and value[1] < .005 for value in errors.values())
    result = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dataset_hashes': atlas['dataset_hashes'], 'baseline_sha256': atlas['model_sha256'], 'andes': andes.__version__,
        'equilibrium_errors': eq_errors, 'PQ_config': pq_config, 'train_events': len(rows), 'diagnostic_pass': passed,
        'normalized_error_median_p95_max': errors, 'port_incidence_rank': int(np.linalg.matrix_rank(ports)),
        'generator_terminal_buses': list(map(int, system.GENROU.bus.v)), 'actual_branch_count': system.Line.n,
        'equilibrium_load_line_shunt_copper': eq_components.tolist(),
        'channel_names': ['minus_load_deviation', 'minus_line_loss_deviation', 'minus_shunt_deviation', 'minus_copper_deviation', 'ROM_damping'],
        'channel_second_moment': (moments/len(rows)).tolist(),
        'load_only_net_energy_explained': 1-sum(r['load_only_error'] for r in rows)/sum(r['net_energy'] for r in rows),
        'records': rows, 'scope': 'TRAIN-only physical power balance. Port incidence is not a fitted memory model or Kron reduction.'}
    np.savez_compressed(args.output/'physical_network.npz', bus_ids=bus_ids, generator_buses=generator_buses, load_buses=load_buses,
        load_g=load_g, shunt_buses=shunt_buses, shunt_g=shunt_g, yline=yline, stator_ra=system.GENROU.ra.v,
        incidence=incidence, port_incidence=ports, line_ids=np.array(system.Line.idx.v), equilibrium_components=eq_components)
    np.savez_compressed(args.output/'components.npz', **arrays)
    (args.output/'evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'channel_second_moment')}, indent=2))


if __name__ == '__main__':
    main()
