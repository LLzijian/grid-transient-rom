# TASK4 explicit ODE candidate

Requires Python and NumPy only. Run `python simulate.py --history history.npy --output forecast.npz`. The supplied history must use the original canonical physical units and contain 11 samples at 10 ms spacing; p at initialization is assumed known. Only delta and omega histories enter the GRU. Forecast time starts at 0.1 s.

model.json contains explicit formula coefficients and one-time GRU weights. equations.md renders every nonzero scalar coefficient. runtime.py computes the autonomous field without Torch, automatic differentiation or a teacher. No accuracy, speed or stability certificate follows from packaging; consult the separately frozen evaluation.
