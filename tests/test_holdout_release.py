from __future__ import annotations

import pytest

from kiki.evaluation.release import SinglePassHoldout


class _Store:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def load_event(self, event_id: str) -> dict[str, str]:
        self.calls.append(event_id)
        if self.fail:
            raise OSError("simulated uncertain payload read")
        return {"event_id": event_id}


def test_contaminated_dataset_cannot_be_rearmed() -> None:
    guard = SinglePassHoldout(frozenset({"a"}), frozenset({"used-build"}))
    with pytest.raises(RuntimeError, match="already consumed"):
        guard.authorize(release_status="armed", dataset_build_id="used-build")


def test_holdout_payload_is_loaded_at_most_once() -> None:
    store = _Store()
    guard = SinglePassHoldout(frozenset({"a"}), frozenset())
    guard.authorize(release_status="armed", dataset_build_id="fresh-build")
    assert guard.load_event_once(store, "a") == {"event_id": "a"}
    with pytest.raises(RuntimeError, match="more than once"):
        guard.load_event_once(store, "a")
    assert store.calls == ["a"]


def test_claim_is_persisted_before_payload_read_and_failed_read_cannot_retry() -> None:
    order: list[tuple[str, str]] = []

    class OrderedStore(_Store):
        def load_event(self, event_id: str) -> dict[str, str]:
            order.append(("load", event_id))
            return super().load_event(event_id)

    def persist(audit: dict[str, object]) -> None:
        claimed = audit["claimed_event_ids"]
        assert isinstance(claimed, list)
        order.append(("claim", str(claimed[-1])))

    store = OrderedStore(fail=True)
    guard = SinglePassHoldout(
        frozenset({"a"}), frozenset(), claim_callback=persist
    )
    guard.authorize(release_status="armed", dataset_build_id="fresh-build")
    with pytest.raises(OSError, match="uncertain payload read"):
        guard.load_event_once(store, "a")
    with pytest.raises(RuntimeError, match="more than once"):
        guard.load_event_once(store, "a")
    assert order == [("claim", "a"), ("load", "a")]
    assert store.calls == ["a"]


def test_holdout_cannot_load_before_authorization() -> None:
    store = _Store()
    guard = SinglePassHoldout(frozenset({"a"}), frozenset())
    with pytest.raises(RuntimeError, match="not been authorized"):
        guard.load_event_once(store, "a")
    assert store.calls == []


def test_holdout_finalize_requires_exact_population() -> None:
    store = _Store()
    guard = SinglePassHoldout(frozenset({"a", "b"}), frozenset())
    guard.authorize(release_status="armed", dataset_build_id="fresh-build")
    guard.load_event_once(store, "a")
    with pytest.raises(RuntimeError, match=r"missing=\['b'\]"):
        guard.finalize()
    guard.load_event_once(store, "b")
    assert guard.finalize()["single_pass_complete"] is True
