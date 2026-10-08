"""Loop0060 Phase-2 fixed-anchor nonlinear latent-RHS teacher."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kiki.models.private_pm import MechanicalROM
from train_joint_latent import BASE, DT, NETWORK, JointROM, read_development, tensor
from train_long_latent import campaign_loss, sample_batch


class TeacherROM(JointROM):
    """Sparse anchor plus the preregistered zero-at-origin latent correction."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.rank == 3, 'registered teacher input is fixed at q3 + 72 physical features'
        # Parent parameters are immediately overwritten by the anchor; reset
        # here so the teacher's default Linear initialization is seed-61 exact.
        torch.manual_seed(61)
        self.W1 = nn.Linear(75, 16).double()
        self.W2 = nn.Linear(16, self.rank, bias=False).double()
        nn.init.zeros_(self.W2.weight)

    def field(self, state):
        anchor = super().field(state)
        u = self.features(state)[..., 1:]
        correction = self.W2(torch.tanh(self.W1(u))-torch.tanh(self.W1.bias))
        return torch.cat((anchor[..., :30], anchor[..., 30:]+correction), dim=-1)

    def latent_jacobian_at_reference(self):
        masked_xi = self.xi*self.support
        derivative = 1-torch.tanh(self.W1.bias).square()
        return masked_xi[1:4].T + self.W2.weight @ torch.diag(derivative) @ self.W1.weight[:, :3]

    aq_reference = latent_jacobian_at_reference


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def construct_teacher(anchor, base, network):
    """Construct from immutable anchor normalization and verify saved buffers."""
    norm = np.load(anchor/'normalization.npz', allow_pickle=False)
    electrical = (norm['z_scale'], norm['voltage_star'], norm['load_positions'],
                  network['load_g'], network['equilibrium_components'][0])
    model = TeacherROM(base, network['port_incidence'][:, norm['port_ids'].astype(int)],
                       norm['input_scale'], norm['feature_scale'], float(norm['force_scale']),
                       float(norm['load_scale']), electrical=electrical, latent_dim=3)
    state = torch.load(anchor/'model.pt', map_location='cpu', weights_only=True)
    buffers = ('mass', 'incidence', 'delta_star', 'coupling', 'damping', 'rate', 'gain',
               'angle_incidence', 'ports', 'input_scale', 'feature_scale', 'force_scale',
               'load_scale', 'metric_scale', 'z_scale', 'voltage_star', 'load_positions',
               'load_g', 'load_star')
    for name in buffers:
        assert torch.equal(model.state_dict()[name], state[name]), name
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert set(missing) == {'W1.weight', 'W1.bias', 'W2.weight'} and not unexpected
    for name, value in state.items():
        assert torch.equal(model.state_dict()[name], value), name
    return model, state


def freeze_teacher_only(model, anchor_state):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('W1.') or name.startswith('W2.'))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()
              if name in anchor_state}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1264
    return before


def assert_anchor_unchanged(model, before):
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value), name


def write_failure(output, args, step, reason, losses):
    payload = {'start_model': str(args.start_model), 'source_anchor_sha256': digest(args.start_model/'model.pt'),
               'code_sha256': digest(__file__), 'step': step, 'reason': reason, 'losses': losses,
               'test_read': False}
    (output/'failure_evidence.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-model', type=Path, required=True)
    parser.add_argument('--horizon', type=float, choices=(2, 5), required=True)
    parser.add_argument('--resume-teacher', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(61); rng = np.random.default_rng(61)
    started = time.perf_counter()
    base, network = MechanicalROM.load(BASE), dict(np.load(NETWORK, allow_pickle=False))
    network['observed_buses'] = np.unique(np.r_[network['load_buses'], network['generator_buses']])
    network['load_positions'] = np.searchsorted(network['observed_buses'], network['load_buses'])
    events, excluded, identities = read_development(base, network, full_train=True)
    model, anchor_state = construct_teacher(args.start_model, base, network)
    if args.resume_teacher is not None:
        resumed = torch.load(args.resume_teacher/'model.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(resumed, strict=True)
        for name, value in anchor_state.items():
            assert torch.equal(model.state_dict()[name], value), name
    before = freeze_teacher_only(model, anchor_state)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0003)
    losses = []
    sampled_endpoints = []
    for step in range(700):
        optimizer.zero_grad(set_to_none=True)
        samples, histories = sample_batch(events['train'], rng)
        sampled_endpoints.extend(endpoint for _, endpoint in samples)
        loss, terms = campaign_loss(model, samples, histories, args.horizon)
        if not torch.isfinite(loss):
            write_failure(args.output, args, step+1, 'nonfinite loss', losses)
            raise FloatingPointError(f'nonfinite loss at step {step+1}')
        loss.backward()
        gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
        if not torch.isfinite(gradient).all():
            write_failure(args.output, args, step+1, 'nonfinite teacher gradient', losses)
            raise FloatingPointError(f'nonfinite gradient at step {step+1}')
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
            write_failure(args.output, args, step+1, 'nonfinite teacher parameter', losses)
            raise FloatingPointError(f'nonfinite teacher parameter at step {step+1}')
        if step % 50 == 0 or step == 699:
            row = {'step': step+1, 'loss': float(loss.detach()), 'teacher_grad_norm': float(torch.linalg.vector_norm(gradient)),
                   **{k: float(v.detach()) for k, v in terms.items()}}
            losses.append(row); print(json.dumps(row), flush=True)
    assert_anchor_unchanged(model, before)
    torch.save(model.state_dict(), args.output/'model.pt')
    shutil.copyfile(args.start_model/'normalization.npz', args.output/'normalization.npz')
    resume_hash = digest(args.resume_teacher/'model.pt') if args.resume_teacher else None
    evidence = {'start_model': str(args.start_model), 'resume_teacher': str(args.resume_teacher) if args.resume_teacher else None,
                'source_anchor_sha256': digest(args.start_model/'model.pt'), 'source_normalization_sha256': digest(args.start_model/'normalization.npz'),
                'resume_model_sha256': resume_hash, 'loss_implementation_sha256': digest(Path(__file__).with_name('train_long_latent.py')),
                'model_sha256': digest(args.output/'model.pt'), 'normalization_sha256': digest(args.output/'normalization.npz'),
                'code_sha256': digest(__file__), 'seed': 61, 'lr': .0003, 'steps': 700, 'horizon_s': args.horizon,
                'trainable_parameters': 1264, 'frozen_anchor_equal': True, 'Aq_reference': model.latent_jacobian_at_reference().detach().tolist(),
                'losses': losses, 'excluded': excluded, 'identities': identities,
                'train_lengths': {e['id']: len(e['x']) for e in events['train']},
                'sampled_endpoint_coverage': {'min': min(sampled_endpoints), 'max': max(sampled_endpoints),
                                              'count': len(sampled_endpoints), 'count_gt_700': sum(i > 700 for i in sampled_endpoints),
                                              'count_gt_810': sum(i > 810 for i in sampled_endpoints)},
                'seconds': time.perf_counter()-started,
                'nonfinite': False, 'test_read': False}
    (args.output/'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
