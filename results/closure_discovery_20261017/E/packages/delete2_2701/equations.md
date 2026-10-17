# Symbolic closure ROM equations

This package evaluates the serialized L10-S model in its formal model-storage units; these values are not SI units. The serialized scales are used as-is, without conversion.

## State and field

The interface state is `z = (delta, omega, p, q)` with 10 coordinates per block. The energy-gradient coordinates use `x = (delta, pi, p, q)` where `pi = mass * omega`. The exported NumPy field evaluates the serialized `J`, `R`, and `B` matrices as `f = (J - R) grad(H) + B u`, with the model's port closure supplying `u`. It converts the second block back by dividing by mass; the displacement kinematics are `delta_dot = omega`.

The physical phase is `phi = D delta_star`; edge phase changes are `eta = D delta`, with `s = sin(phi + eta) - sin(phi)` and `c = cos(phi + eta) - cos(phi)`. The fixed M3 energy is represented by the serialized coupling and storage arrays. The retained shared quadratic contributes `0.5 * q_quadratic * sum(q**2)`.

## Feature scaling and AST terms

Node features are `(p / p_scale, q)`; `q` remains in the serialized raw state scale. Edge features are `(p_i, p_j, q_i, q_j, s, c)`, where the endpoint powers are normalized by `p_scale` and use the positive and negative parts of `D`. The AST names `pi` and `pj` denote these endpoint powers; momentum is the separate state block `pi = mass * omega`. With `eta = delta @ D.T`, the angle features are `s = sin(phi + eta) - sin(phi)` and `c = cos(phi + eta) - cos(phi)`. Edge expressions are averaged with their endpoint-reversed form (`i <-> j`, `s -> -s`, `c` unchanged), then centered in physical coordinates `(p_i, p_j, q_i, q_j, eta)` by subtracting the symmetrized equilibrium value and first derivative. After feature-origin centering, the compiler restores `g_c0 * (c + sin(phi) * eta)`, where `g_c0` is the symmetrized expression's derivative with respect to `c` at the feature origin. This retains the angle curvature of a standalone cosine expression. Node energy expressions subtract their feature-origin value and tangent. Port features are `(omega / input_scale_omega, p / p_scale, q, es, ec, nq)`; port expressions subtract their value at the feature origin. `es` and `ec` use the serialized fixed port weights, and `nq = q @ neighbor_A.T - q`.

For each node AST `F`, its energy contribution is `c * sum_i [F(x_i) - F(0) - grad(F)(0) dot x_i]`. For each edge AST, let `z_e = (p_i / p_scale, p_j / p_scale, q_i, q_j, eta)` and let `G_e(z_e)` be the endpoint-symmetrized expression composed with the angle feature map above. Its contribution is `c * sum_e [k_e / mean(k) * (G_e(z_e) - G_e(0) - grad_z(G_e)(0) dot z_e)]`. A port AST contributes `u_i = c * [F(y_i) - F(0)]`. The JSON terms below are the exact ASTs, and each listed coefficient is the serialized numeric value used by the simulator.

The encoder uses the original seven local features per machine: `coi = delta - sum(mass * delta) / sum(mass)`, `dnorm = coi / input_scale_delta`, `wnorm = w / input_scale_omega`, `nei_d = (coi @ neighbor_A.T - coi) / input_scale_delta`, `nei_w = (w @ neighbor_A.T) / input_scale_omega`, `mass / mean(mass)`, `rate / mean(rate)`, and `gain / mean(gain)`. These seven features feed the serialized one-layer GRU and linear head; `p` does not enter the encoder. Its latent state is `0.1 * (raw(history) - raw(zero_history))`. The history endpoint's 30 physical coordinates, including `p`, are concatenated with this latent state to initialize rollout.

### Serialized symbolic terms

1. Channel `node`, coefficient `0.035432881373743647`

```json
{
  "args": [
    {
      "name": "p",
      "op": "var"
    },
    {
      "name": "q",
      "op": "var"
    }
  ],
  "op": "mul"
}
```

2. Channel `edge`, coefficient `0.6302978331205743`

```json
{
  "args": [
    {
      "args": [
        {
          "args": [
            {
              "name": "qj",
              "op": "var"
            },
            {
              "op": "const",
              "value": -1
            }
          ],
          "op": "mul"
        },
        {
          "name": "qi",
          "op": "var"
        }
      ],
      "op": "add"
    },
    {
      "name": "s",
      "op": "var"
    }
  ],
  "op": "mul"
}
```

3. Channel `port`, coefficient `-0.30277646875410624`

```json
{
  "name": "w",
  "op": "var"
}
```

4. Channel `port`, coefficient `-0.36391481529089148`

```json
{
  "name": "p",
  "op": "var"
}
```

5. Channel `port`, coefficient `-0.41011303688844486`

```json
{
  "name": "q",
  "op": "var"
}
```

6. Channel `port`, coefficient `-0.96589542793067518`

```json
{
  "name": "ec",
  "op": "var"
}
```

7. Channel `port`, coefficient `0.71913074840361058`

```json
{
  "name": "nq",
  "op": "var"
}
```

The symbolic complexity metadata records operator counts and limits. The inherited expanded-RHS cost field is intentionally retained as descriptive text; this package does not infer or invent an expanded contribution count.

## Numerical arrays

The model has 10 machines and 26 serialized edge rows. `mass`, `D`, `delta_star`, `phi`, graph coupling, M3 coupling, storage, scales, fixed port maps, and `J`/`R`/`B` are stored in `model.json`.
