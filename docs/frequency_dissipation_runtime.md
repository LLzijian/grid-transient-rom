# Frequency-dependent dissipative port runtime

The optional W8 runtime keeps the existing seven-term energy and adds one shared port coefficient:

`u = u_N + b_W * (omega / input_scale_omega)**3`, with `b_W <= 0`.

The existing positive frequency scales and inertia are retained. The added interface frequency derivative is `b_W * (omega / scale)**3 / mass`. External `u` replaces the entire autonomous port, including this added term. The added energy rate is `b_W * sum(omega**4 / scale**3) <= 0` in model H-units per second. This identity concerns the added port; the full model retains its other energy inputs.

Use `build_W` with an exact N7 source, or restore an exported W8 model using `WRuntime.from_export`. The existing `encode`, `field`, `rollout` and export format remain available. `coefficient_indices` identifies `b_W` by expression identity. Cached mappings retain all coefficient and skeleton gradients and are rebuilt for each optimizer graph. Exports require a finite model, a nonpositive added coefficient and no active graph.

`one_update` uses the registered full objective, clips the full gradient at100, applies Adam, checks finite parameters/gradients/moments, then projects only `b_W` to `min(b_W,0)`. Adam moments are retained. The trace records raw and clipped gradients, Adam and projected increments, and projection events. The intermediate positive Adam proposal is permitted for projection; it is not an admissible saved model. `extra_port_power` reports the added H-rate; `extra_port_field` returns ten added frequency derivatives in rad/s².

Validation used a separately retained, hash-locked synthetic fixture. The bounded QA completed with48 passing checks:22 runtime,16 runner and10 native. Runtime checks include generic/NumPy parity, nonunit inertia and nonuniform scales, complete5s objective gradients and Adam/projection, invariances, causal rollout and nonempty checkpoint/RNG recovery. The new clean production-module subprocess also completed six synthetic baseline contracts with `PYTHONPATH` unset. The subsequent unique real-data cost run completed the actual source/RNG contracts and12 discarded updates and admitted the registered three full fits. These are correctness and execution-admission results; they do not establish predictive improvement or model adoption.

Reproduction requires the existing project modules and the five frozen P0 synthetic assets under `private_data/closure_mechanism_20261019/P0/fused_runtime01/prepare01`. Fixture manifest SHA256: `0a1f2463693b0a5227c8f39e725a4e8891af0300dfccdfde705d66e346ee9816`. Set `FUSED_CLOSURE_FIXTURE` explicitly to that directory. The public clone does not contain those private assets and is not a self-contained CI fixture. Historical runners and their outcomes remain unchanged; the current W8 research runner and native helpers are retained on the archival branch.
