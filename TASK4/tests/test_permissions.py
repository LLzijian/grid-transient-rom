import pytest
from task4.training import check_authorized

def test_real_training_not_started_by_packaging(monkeypatch):
    monkeypatch.delenv('TASK4_MANAGED_RUN_ID',raising=False)
    with pytest.raises(PermissionError):check_authorized(False)

def test_synthetic_smoke_is_allowed_without_scientific_authority():
    check_authorized(True)
