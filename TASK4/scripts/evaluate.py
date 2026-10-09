import _bootstrap
import argparse
import json
import torch

from task4.paths import ROOT, read_json, write_json, project_path, source_root, source_file
from task4.data import load_events, train_scales, digest
from task4.training import load_checkpoint, check_authorized
from task4.evaluation import evaluate, FrozenM3
from task4.accelerate import trace_field
from task4.dynamics import PowerGraph
from task4.synthetic import fixture


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--baseline-m3', action='store_true')
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--output', required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--endpoint', choices=['P-N', 'P-S', 'A', 'U', 'M3'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--fine-dt', type=float, default=.0025)
    parser.add_argument('--coarse-dt', type=float, default=.005)
    parser.add_argument('--frozen-manifest')
    parser.add_argument('--trace-field', action='store_true',
                        help='Use the existing exact TorchScript field executor for a checkpoint endpoint')
    parser.add_argument('--reuse-origin-terms', action='store_true',
                        help='With --trace-field, reuse batch-shaped PowerGraph origins once per fine/coarse rollout')
    args = parser.parse_args()
    check_authorized(args.synthetic)
    if args.reuse_origin_terms:
        if not args.trace_field:
            raise ValueError('--reuse-origin-terms requires --trace-field')
        if args.baseline_m3 or args.endpoint in ('M3', 'U'):
            raise ValueError('--reuse-origin-terms is limited to structured PowerGraph N/S/A endpoints')
    if args.baseline_m3:
        if args.trace_field:
            raise ValueError('--trace-field is for checkpoint endpoints; fixed M3 uses the eager executor')
        if args.endpoint not in (None, 'M3'):
            raise ValueError('--baseline-m3 can only use the fixed M3 endpoint')
        args.endpoint = 'M3'
    elif args.endpoint == 'M3':
        raise ValueError('M3 endpoint requires --baseline-m3')
    if not args.synthetic and args.endpoint is None:
        raise ValueError('Real evaluation requires a registered --endpoint P-N/P-S/A/U/M3')
    torch.set_num_threads(1)
    project = None if args.synthetic else read_json(ROOT / 'configs/project.json')
    guarded_sources = {}
    manifest = None
    if args.frozen_manifest:
        if args.synthetic or args.endpoint is None:
            raise ValueError('Frozen endpoint manifest is for registered real endpoints')
        manifest_path = project_path(args.frozen_manifest)
        guarded_sources[str(manifest_path)] = digest(manifest_path)
        manifest = read_json(manifest_path)
        if (manifest.get('complete') is not True or manifest.get('protocol') not in ('T4_0001', 'T4_0002')
                or manifest.get('fine_dt') != args.fine_dt or manifest.get('coarse_dt') != args.coarse_dt):
            raise ValueError('Incomplete or incompatible frozen endpoint manifest')
    if project is not None:
        baseline_path = source_file(source_root(project), project['baseline'])
        guarded_sources[str(baseline_path)] = digest(baseline_path)
        if manifest is not None and guarded_sources[str(baseline_path)] != manifest['endpoints']['M3']['source_sha256']:
            raise ValueError('Fixed M3 source differs from frozen identity before development read')

    def source_guard():
        try:
            unchanged = all(digest(path) == value for path, value in guarded_sources.items())
        except OSError as exc:
            raise ValueError('Frozen source became inaccessible during evaluation') from exc
        if not unchanged:
            raise ValueError('Frozen checkpoint/manifest/physical source changed during evaluation')

    checkpoint = None
    if args.checkpoint:
        checkpoint_path = project_path(args.checkpoint)
        guarded_sources[str(checkpoint_path)] = digest(checkpoint_path)
        if manifest is not None:
            expected = manifest['endpoints'][args.endpoint]
            if (str(checkpoint_path) != expected['checkpoint']
                    or guarded_sources[str(checkpoint_path)] != expected['checkpoint_sha256']):
                raise ValueError('Checkpoint differs from frozen endpoint before development read')
        model, encoder, _, checkpoint = load_checkpoint(checkpoint_path)
        source_guard()
        if bool(checkpoint['synthetic']) != args.synthetic:
            raise ValueError('Synthetic/real checkpoint mode mismatch')
        model.evaluation_identity = {'checkpoint': str(checkpoint_path),
                                     'checkpoint_sha256': guarded_sources[str(checkpoint_path)],
                                     'kind': checkpoint.get('kind'), 'step': checkpoint.get('step')}
        if manifest is not None and (checkpoint['kind'] != expected['kind'] or checkpoint['step'] != expected['step']
                                     or checkpoint['physics'] != manifest['reference_physics']
                                     or checkpoint['scales'] != manifest['training_scales']):
            raise ValueError('Loaded checkpoint disagrees with frozen metadata')
    if args.endpoint is not None and checkpoint is not None:
        arm = checkpoint['config']['arm']
        expected_arm = 'P' if args.endpoint in ('P-N', 'P-S') else args.endpoint
        if arm != expected_arm:
            raise ValueError('Registered endpoint and checkpoint arm disagree')
        if args.endpoint == 'P-S' and checkpoint.get('kind') != 'T4-S':
            raise ValueError('P-S endpoint requires all seven explicit symbolic functions')
        if args.endpoint == 'P-N' and checkpoint.get('kind') != 'T4-N':
            raise ValueError('P-N endpoint requires an uncompressed T4-N checkpoint')
    if args.reuse_origin_terms:
        if type(model) is not PowerGraph:
            raise ValueError('Evaluation origin reuse supports only the exact PowerGraph')
        model.reuse_origin_terms = True
    if args.trace_field:
        if not args.reuse_origin_terms and bool(getattr(model, 'reuse_origin_terms', False)):
            raise ValueError('Evaluation tracing does not enable origin-term reuse')
        trace_field(model)
    source_guard()
    if args.synthetic:
        events, physics = fixture()
        facts = {'split': 'synthetic', 'raw_count': len(events), 'eligible_count': len(events),
                 'excluded': [], 'test_payload_read': False}
    else:
        events, physics, facts = load_events(project, args.split)
        if manifest is not None and physics.as_dict() != manifest['reference_physics']:
            raise ValueError('Current evaluation physical parameters disagree with frozen manifest')
    if args.baseline_m3:
        if args.synthetic or args.split == 'train':
            train_events, train_physics = events, physics
        else:
            train_events, train_physics, _ = load_events(project, 'train')
        scales = train_scales(train_events, train_physics)
        model, encoder = FrozenM3(physics, scales), None
        model.evaluation_identity = {'kind': 'fixed_M3', 'fitted_on_development': False,
                                     'scale_source': 'synthetic_fixture' if args.synthetic else 'TRAIN',
                                     'scale_event_ids': [event.id for event in train_events]}
        if not args.synthetic:
            model.evaluation_identity.update(baseline_source=str(baseline_path),
                                              baseline_source_sha256=guarded_sources[str(baseline_path)])
            if manifest is not None and (physics.as_dict() != manifest['reference_physics']
                                        or list(scales) != manifest['training_scales']):
                raise ValueError('M3 physical parameters/TRAIN scales disagree with frozen manifest')
    if manifest is not None:
        model.evaluation_identity['frozen_manifest'] = str(manifest_path)
        model.evaluation_identity['frozen_manifest_sha256'] = guarded_sources[str(manifest_path)]
    source_guard()
    result = evaluate(model, encoder, events, args.output,
                      horizons=(.05, .1) if args.synthetic else (1., 2., 5., 8.),
                      physics=physics, facts=facts, batch_size=args.batch_size, endpoint=args.endpoint,
                      fine_dt=args.fine_dt, coarse_dt=args.coarse_dt)
    result['execution'] = {'field_executor': 'torchscript_trace' if args.trace_field else 'eager',
                           'trace_field_requested': args.trace_field,
                           'trace_build_seconds': float(getattr(model, 'trace_build_seconds', 0.)) if args.trace_field else 0.,
                           'origin_terms_reused': args.reuse_origin_terms}
    if args.reuse_origin_terms:
        result['execution']['origin_build_scope'] = 'once per fine/coarse rollout for the full initial batch; active/singleton slices'
    try:
        source_guard()
    except ValueError:
        result['source_identity_unchanged'] = False
        result['evidence_validity'] = 'invalid_source_changed'
        write_json(project_path(args.output), result)
        raise
    result['source_identity_unchanged'] = True
    write_json(project_path(args.output), result)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
