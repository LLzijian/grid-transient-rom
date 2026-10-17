# NumPy symbolic closure package

This directory contains `model.json`, a standalone `simulate.py`, and `equations.md`.
The simulator imports only NumPy and Python standard-library modules; it does not
depend on the project source tree or PyTorch.

Run a fixed-step rollout from an 11-sample history shaped `[batch, 11, 30]`:

```text
python simulate.py model.json history.npy --steps 20 --dt 0.001 --output trajectory.npy
```

The output includes the initialized state followed by each RK4 state, with shape
`[batch, steps + 1, 40]`. See `equations.md` for feature scaling, AST terms, and
the serialized energy/field conventions.
