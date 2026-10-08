"""Guards for a preregistered, single-pass confirmatory holdout release."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SinglePassHoldout:
    """Enforce one claimed payload read per expected holdout event.

    A claim is recorded before the backing store is called.  This deliberately
    burns an event when a load fails: permitting a retry after an uncertain read
    would violate the at-most-once release contract.
    """

    expected_event_ids: frozenset[str]
    contaminated_build_ids: frozenset[str]
    claim_callback: Callable[[dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    claimed_event_ids: set[str] = field(default_factory=set, init=False)
    authorized: bool = field(default=False, init=False)

    def authorize(self, *, release_status: str, dataset_build_id: str) -> None:
        if release_status != "armed":
            raise RuntimeError("holdout release is not armed by its preregistration")
        if dataset_build_id in self.contaminated_build_ids:
            raise RuntimeError(
                "dataset build was already consumed by exploratory/legacy analyses"
            )
        if not self.expected_event_ids:
            raise ValueError("holdout release requires expected event IDs")
        self.authorized = True

    def audit_snapshot(self) -> dict[str, Any]:
        return {
            "authorized": self.authorized,
            "expected_events": len(self.expected_event_ids),
            "claimed_events": len(self.claimed_event_ids),
            "claimed_event_ids": sorted(self.claimed_event_ids),
            "maximum_payload_read_attempts_per_event": 1,
        }

    def load_event_once(self, store: Any, event_id: str) -> Any:
        if not self.authorized:
            raise RuntimeError("holdout release has not been authorized")
        if event_id not in self.expected_event_ids:
            raise KeyError(f"event is outside the preregistered holdout: {event_id}")
        if event_id in self.claimed_event_ids:
            raise RuntimeError(f"holdout event payload requested more than once: {event_id}")
        self.claimed_event_ids.add(event_id)
        if self.claim_callback is not None:
            self.claim_callback(self.audit_snapshot())
        return store.load_event(event_id)

    def finalize(self) -> dict[str, Any]:
        if not self.authorized:
            raise RuntimeError("holdout release has not been authorized")
        missing = sorted(self.expected_event_ids - self.claimed_event_ids)
        unexpected = sorted(self.claimed_event_ids - self.expected_event_ids)
        if missing or unexpected:
            raise RuntimeError(
                f"holdout consumption incomplete: missing={missing}, unexpected={unexpected}"
            )
        return {
            **self.audit_snapshot(),
            "single_pass_complete": True,
        }
