"""Loop0078 R-growth must not duplicate learnable base diagonal directions."""
import sys
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_structured_rom import screen, source_fixture


def _r_choice(model):
    x=torch.full((4,33),.1,dtype=torch.float64)
    internal=model.structured.interface_to_state(x)
    _, energy=model.structured.energy_and_gradient(internal)
    J,R=model.structured.matrices(); scale=screen._selection_scale(model,internal)
    # Make already-covered pi/p directions overwhelmingly attractive.  A
    # legal q direction must still be selected instead.
    sensitivity=torch.zeros_like(internal)
    sensitivity[:,10:30]=1e6
    sensitivity[:,30:]=1.
    return screen._best_candidate(model,internal,energy.detach(),sensitivity,J.detach(),R.detach(),scale,'R')


def test_r_growth_excludes_active_base_pi_and_p_directions():
    base, source=source_fixture(); model=screen.GROM(base,source,'G-auto')
    choice=_r_choice(model)
    assert choice is not None
    assert choice[3] in (20,21,22)


def test_r_slots_remain_missing_after_all_three_q_directions_exist():
    base, source=source_fixture(); model=screen.GROM(base,source,'G-auto')
    for coordinate in (20,21,22): model.add_dissipation(coordinate,1e-3)
    assert _r_choice(model) is None
