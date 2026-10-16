"""Pure, bounded helpers for closure-discovery orchestration.

This module does not load project data or train models.  Callers provide all timing,
candidate scores, allowed TRAIN metadata, and paired event-level errors explicitly.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import itertools
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Iterator


PHASE_CAPS_SECONDS: dict[str, float] = {
    "A": 43_200.0,
    "B": 86_400.0,
    "C": 57_600.0,
    "D": 43_200.0,
    "E": 14_400.0,
    "reserve": 14_400.0,
}
PARENT_CAP_SECONDS = 259_200.0

FIXED_LIBRARY_TERMS: tuple[dict[str, Any], ...] = (
    {"channel": "node", "expression": {"op": "mul", "args": [{"op": "var", "name": "p"}, {"op": "var", "name": "q"}]}},
    {"channel": "edge", "expression": {"op": "mul", "args": [{"op": "add", "args": [{"op": "var", "name": "qi"}, {"op": "mul", "args": [{"op": "const", "value": -1}, {"op": "var", "name": "qj"}]}]}, {"op": "var", "name": "s"}]}},
    {"channel": "edge", "expression": {"op": "mul", "args": [{"op": "add", "args": [{"op": "var", "name": "qi"}, {"op": "var", "name": "qj"}]}, {"op": "var", "name": "c"}]}},
    *(
        {"channel": "port", "expression": {"op": "var", "name": name}}
        for name in ("w", "p", "q", "es", "ec", "nq")
    ),
)

FAULT_DURATIONS_SECONDS: tuple[float, ...] = (0.075, 0.125, 0.175, 0.225, 0.275, 0.35, 0.5)
FAULT_XF_VALUES: tuple[float, ...] = (0.001, math.sqrt(0.001 * 0.01), 0.01)


class InsufficientBudget(RuntimeError):
    """Raised before work starts when its stage has insufficient remaining budget."""

    def __init__(self, stage: str, needed_seconds: float, remaining_seconds: float):
        self.stage = stage
        self.needed_seconds = needed_seconds
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"stage {stage} needs {needed_seconds:.3f}s; "
            f"only {remaining_seconds:.3f}s remain (no cross-stage transfer)"
        )


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_expr(expr: Mapping[str, Any]) -> Any:
    op = expr.get("op")
    if op == "var":
        return {"op": "var", "name": str(expr["name"])}
    if op == "const":
        value = expr["value"]
        if value not in (-1, 1) or isinstance(value, bool):
            raise ValueError("AST constants support exactly -1 or 1")
        return {"op": "const", "value": int(value)}
    if op in {"add", "mul"}:
        args = []
        for child in expr["args"]:
            canonical = _canonical_expr(child)
            args.extend(canonical["args"] if canonical.get("op") == op else [canonical])
        args.sort(key=_json_dump)
        if op == "mul":
            sign = 1
            rest = []
            for child in args:
                if child.get("op") == "const":
                    sign *= child["value"]
                else:
                    rest.append(child)
            if not rest:
                return {"op": "const", "value": sign}
            if sign < 0:
                rest.append({"op": "const", "value": -1})
            args = sorted(rest, key=_json_dump)
        return args[0] if len(args) == 1 else {"op": op, "args": args}
    if op in {"pow2", "pow3", "pow4", "tanh"}:
        result = {"op": op, "arg": _canonical_expr(expr["arg"])}
        if op == "tanh":
            scale = float(expr.get("scale", 1))
            if scale not in (0.5, 1, 2):
                raise ValueError("tanh scale must be 0.5, 1, or 2")
            result["scale"] = scale
        return result
    raise ValueError(f"unsupported expression operation: {op!r}")


def expression_key(expr: Mapping[str, Any]) -> str:
    """Return a stable canonical key for a supported expression tree."""
    return _json_dump(_canonical_expr(expr))


def term_key(term: Mapping[str, Any]) -> str:
    """Return a stable key including the term's closure channel."""
    channel = term.get("channel")
    if channel not in {"node", "edge", "port"}:
        raise ValueError(f"unknown term channel: {channel!r}")
    return _json_dump({"channel": channel, "expression": _canonical_expr(term["expression"])})


def fixed_library() -> list[dict[str, Any]]:
    """Return a fresh JSON-compatible copy of the registered nine-term library."""
    return copy.deepcopy(list(FIXED_LIBRARY_TERMS))


def _stage_spend_from_receipts(receipts: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    spend = {stage: 0.0 for stage in PHASE_CAPS_SECONDS}
    for receipt in receipts:
        if receipt.get("counted", True):
            stage = receipt["stage"]
            if stage not in spend:
                raise ValueError(f"receipt has unknown stage {stage!r}")
            spend[stage] += float(receipt["elapsed_seconds"])
    return spend


class ProcessLedger:
    """Small append-only JSON receipt ledger with an atomically refreshed snapshot.

    Receipts are authoritative.  The state file is a recoverable convenience snapshot.
    Nested work gets its own receipt with ``counted=False``; only its outermost timed
    scope contributes wall time, preventing subprocess time from being charged twice.
    """

    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.receipts_path = self.directory / "receipts.jsonl"
        self.state_path = self.directory / "state.json"
        self.lock_path = self.directory / ".ledger.lock"
        self._local_lock = contextvars.ContextVar(f"ledger_lock_{id(self)}", default=None)
        self._scope = contextvars.ContextVar(f"ledger_scope_{id(self)}", default=None)
        self._initialize_state()

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        # Locking is process-safe on Windows and POSIX, while the context-local RLock
        # also protects multiple threads in this process.
        import threading

        lock = self._local_lock.get()
        if lock is None:
            lock = threading.RLock()
            self._local_lock.set(lock)
        with lock:
            with self.lock_path.open("a+b") as handle:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    if os.fstat(handle.fileno()).st_size == 0:
                        handle.write(b"0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    try:
                        yield
                    finally:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_receipts(self) -> list[dict[str, Any]]:
        if not self.receipts_path.exists():
            return []
        out = []
        with self.receipts_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid receipt at line {line_number}") from exc
                out.append(item)
        return out

    def _write_state(self, receipts: Sequence[Mapping[str, Any]]) -> None:
        state = {
            "schema_version": 1,
            "caps_seconds": PHASE_CAPS_SECONDS,
            "parent_cap_seconds": PARENT_CAP_SECONDS,
            "stage_spent_seconds": _stage_spend_from_receipts(receipts),
            "receipt_count": len(receipts),
            "updated_utc_epoch": time.time(),
        }
        fd, temporary = tempfile.mkstemp(prefix="state.", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _initialize_state(self) -> None:
        with self._locked():
            self._write_state(self._read_receipts())

    @property
    def stage_spend_seconds(self) -> dict[str, float]:
        return _stage_spend_from_receipts(self._read_receipts())

    def remaining_seconds(self, stage: str) -> float:
        if stage not in PHASE_CAPS_SECONDS:
            raise ValueError(f"unknown stage {stage!r}")
        return max(0.0, PHASE_CAPS_SECONDS[stage] - self.stage_spend_seconds[stage])

    def spent(self, stage: str | None = None) -> float:
        """Return counted wall time for one stage or cumulatively across spendable stages."""
        spend = self.stage_spend_seconds
        if stage is None:
            return sum(spend.values())
        if stage not in spend:
            raise ValueError(f"unknown stage {stage!r}")
        return spend[stage]

    def remaining(self, stage: str) -> float:
        """Compatibility spelling for :meth:`remaining_seconds`."""
        return self.remaining_seconds(stage)

    def require_budget(self, stage: str, estimated_seconds: float) -> None:
        """Fail explicitly before launch if a stage's own cap cannot cover the estimate."""
        if stage not in PHASE_CAPS_SECONDS:
            raise ValueError(f"unknown stage {stage!r}")
        estimate = float(estimated_seconds)
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError("estimated_seconds must be finite and nonnegative")
        remaining = self.remaining_seconds(stage)
        if estimate > remaining:
            raise InsufficientBudget(stage, estimate, remaining)

    def record_external(
        self,
        stage: str,
        category: str,
        elapsed_seconds: float,
        status: str = "completed",
        *,
        receipt_id: str | None = None,
        parent_receipt_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        counted: bool = True,
    ) -> str:
        """Append a completed/failed externally timed attempt and refresh atomic state."""
        elapsed = float(elapsed_seconds)
        if stage not in PHASE_CAPS_SECONDS:
            raise ValueError(f"unknown stage {stage!r}")
        if stage == "reserve" and not category.startswith("repair:"):
            raise ValueError("reserve receipts require an explicit repair: category")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if not category or not status:
            raise ValueError("category and status must be nonempty")
        if details is not None:
            _json_dump(details)
        receipt = {
            "receipt_id": receipt_id or str(uuid.uuid4()),
            "parent_receipt_id": parent_receipt_id,
            "stage": stage,
            "category": category,
            "elapsed_seconds": elapsed,
            "status": status,
            "counted": bool(counted),
            "recorded_utc_epoch": time.time(),
            "details": dict(details or {}),
        }
        with self._locked():
            receipts = self._read_receipts()
            if any(item["receipt_id"] == receipt["receipt_id"] for item in receipts):
                raise ValueError(f"duplicate receipt id {receipt['receipt_id']!r}")
            # Preserve actual overrun evidence. require_budget is the preflight gate;
            # an unforeseen overrun is appended and visible instead of discarded.
            receipt["over_budget_after_receipt"] = bool(
                counted
                and _stage_spend_from_receipts(receipts)[stage] + elapsed > PHASE_CAPS_SECONDS[stage]
            )
            with self.receipts_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(_json_dump(receipt) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            receipts.append(receipt)
            self._write_state(receipts)
        return receipt["receipt_id"]

    @contextlib.contextmanager
    def receipt(
        self,
        stage: str,
        category: str,
        *,
        receipt_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> Iterator[str]:
        """Time one attempt, append it even on failure, and retain nested timing once."""
        if stage not in PHASE_CAPS_SECONDS:
            raise ValueError(f"unknown stage {stage!r}")
        if stage == "reserve" and not category.startswith("repair:"):
            raise ValueError("reserve receipts require an explicit repair: category")
        parent_id = self._scope.get()
        current_id = receipt_id or str(uuid.uuid4())
        token = self._scope.set(current_id)
        start = time.monotonic()
        status = "completed"
        try:
            yield current_id
        except BaseException:
            status = "failed"
            raise
        finally:
            elapsed = time.monotonic() - start
            self._scope.reset(token)
            self.record_external(
                stage,
                category,
                elapsed,
                status=status,
                receipt_id=current_id,
                parent_receipt_id=parent_id,
                details=details,
                counted=parent_id is None,
            )

    def run(self, stage: str, label: str) -> contextlib.AbstractContextManager[str]:
        """Runner-facing timed scope alias; yields the persistent receipt ID."""
        return self.receipt(stage, label)

    def run_subprocess(
        self,
        stage: str,
        category: str,
        argv: Sequence[str | os.PathLike[str]],
        **run_kwargs: Any,
    ) -> tuple[subprocess.CompletedProcess[Any], str]:
        """Run one subprocess as a receipt; nested callers are recorded without rebilling."""
        receipt_id = str(uuid.uuid4())
        with self.receipt(stage, category, receipt_id=receipt_id, details={"argv": [str(a) for a in argv]}):
            completed = subprocess.run(argv, **run_kwargs)
        return completed, receipt_id


def _timing_costs(timings: Mapping[str, Any]) -> dict[str, float]:
    """Validate the explicit timing schema used for the B/C/D depth gate."""
    step_key = next((key for key in ("max_measured_step_s", "maxstepseconds", "max_step_seconds") if key in timings), None)
    required = ("fom_cost_s", "known_eval_cost_s")
    missing = [key for key in required if key not in timings]
    if step_key is None:
        missing.insert(0, "max_measured_step_s")
    if missing:
        raise KeyError(f"missing timing fields: {', '.join(missing)}")
    known_eval = timings["known_eval_cost_s"]
    if not isinstance(known_eval, Mapping):
        raise TypeError("known_eval_cost_s must map stage names to separate B/C/D costs")
    out = {
        "max_measured_step_s": float(timings[step_key]),
        "fom_cost_s": float(timings["fom_cost_s"]),
    }
    out.update({f"eval_{stage}": float(known_eval.get(stage, 0.0)) for stage in ("B", "C", "D")})
    if any(not math.isfinite(value) or value < 0 for value in out.values()):
        raise ValueError("timings must be finite and nonnegative")
    return out


def estimate_stage_costs(depth: int, timings: Mapping[str, Any]) -> dict[str, float]:
    """Estimate independent B/C/D spend for one selected library+AST depth."""
    if depth < 1:
        raise ValueError("depth must be positive")
    t = _timing_costs(timings)
    b_updates = 6 * (6 * 32 + 2 * 96 + 2 * 384) * depth  # six runs at the requested depth
    c_updates = 14 * 1024
    d_updates = 14 * 256
    step = t["max_measured_step_s"]
    if 'seconds_by_horizon' in timings:
        horizon_costs={float(k):float(v) for k,v in timings['seconds_by_horizon'].items()}
        if set(horizon_costs)!={1.,2.,5.} or any(not math.isfinite(v) or v<=0 for v in horizon_costs.values()):
            raise ValueError('three positive measured horizon costs required')
        def cost(n):
            return sum(horizon_costs[2. if i<=100 else (1.,2.,5.)[(i-101)%3]] for i in range(1,n+1))
        b_training=6*depth*(6*cost(32)+2*(cost(128)-cost(32))+2*(cost(512)-cost(128)))
        c_training=14*cost(1024);d_training=14*cost(256)
    else:
        b_training=b_updates*step;c_training=c_updates*step;d_training=d_updates*step
    return {
        "B": 1.25 * (b_training + t["eval_B"]),
        "C": 1.25 * (c_training + t["eval_C"]),
        "D": 1.25 * (d_training + 48 * 2 * t["fom_cost_s"] + t["eval_D"]),
    }


def select_depth(timings: Mapping[str, Any], *, stage_caps: Mapping[str, float] = PHASE_CAPS_SECONDS) -> dict[str, Any]:
    """Choose depth 3, 2, or 1 if each B/C/D estimate fits its own phase cap."""
    costs = {depth: estimate_stage_costs(depth, timings) for depth in (3, 2, 1)}
    chosen = next(
        (
            depth
            for depth in (3, 2, 1)
            if all(costs[depth][stage] <= float(stage_caps[stage]) for stage in ("B", "C", "D"))
        ),
        None,
    )
    return {"depth": chosen, "stage_costs_seconds": costs.get(chosen) if chosen else None, "estimates_by_depth": costs}


def _expr(op: str, *args: Any, **extra: Any) -> dict[str, Any]:
    if op == "var":
        return {"op": "var", "name": args[0]}
    if op == "const":
        return {"op": "const", "value": args[0]}
    if op in {"pow2", "pow3", "pow4", "tanh"}:
        result = {"op": op, "arg": args[0]}
        result.update(extra)
        return result
    return {"op": op, "args": list(args)}


def _term(channel: str, expression: Mapping[str, Any]) -> dict[str, Any]:
    return {"channel": channel, "expression": copy.deepcopy(dict(expression))}


def _tanh_count(expression: Mapping[str, Any]) -> int:
    op = expression.get("op")
    if op == "tanh":
        return 1 + _tanh_count(expression["arg"])
    if op in {"add", "mul"}:
        return sum(_tanh_count(child) for child in expression["args"])
    if op in {"pow2", "pow3", "pow4"}:
        return _tanh_count(expression["arg"])
    return 0


def _subtree_edits(expression: Mapping[str, Any], names: Sequence[str], *, port: bool):
    """Construct new trees by editing the actual parent's subexpressions."""
    expr=copy.deepcopy(dict(expression))
    for name in names:
        atom=_expr('var',name)
        yield atom
        yield _expr('mul',expr,atom)
        yield _expr('add',expr,atom)
    for op in ('pow2','pow3','pow4'):
        yield _expr(op,expr)
    if port and _tanh_count(expr)==0:
        for scale in (.5,1.,2.):yield _expr('tanh',expr,scale=scale)
    op=expr['op']
    if op in ('add','mul'):
        for index,child in enumerate(expr['args']):
            for changed in _subtree_edits(child,names,port=port):
                args=copy.deepcopy(expr['args']);args[index]=changed
                yield _expr(op,*args)
    elif op in ('pow2','pow3','pow4','tanh'):
        yield expr['arg']
        for changed in _subtree_edits(expr['arg'],names,port=port):
            yield {**expr,'arg':changed}


def _candidate_terms(parent_terms=()) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def add(channel: str, expr: Mapping[str, Any]) -> None:
        from symbolic_closure_rom import validate_expression
        try:expr=validate_expression(expr,channel)
        except ValueError:return
        candidate = _term(channel, expr)
        out[term_key(candidate)] = candidate

    channel_vars = {
        "node": ("p", "q"),
        "edge": ("pi", "pj", "qi", "qj", "s", "c"),
        "port": ("w", "p", "q", "es", "ec", "nq"),
    }
    for channel, names in channel_vars.items():
        variables = [_expr("var", name) for name in names]
        for var in variables:
            add(channel, var)
            powers = ("pow2", "pow3", "pow4") if channel != "port" else ("pow2", "pow3")
            for power in powers:
                add(channel, _expr(power, var))
            if channel == "port":
                for scale in (0.5, 1, 2):
                    add(channel, _expr("tanh", var, scale=scale))
        for left, right in itertools.combinations(variables, 2):
            add(channel, _expr("mul", left, right))

    p, q = _expr("var", "p"), _expr("var", "q")
    qi, qj, s, c = (_expr("var", name) for name in ("qi", "qj", "s", "c"))
    neg_qj = _expr("mul", _expr("const", -1), qj)
    add("node", _expr("mul", p, _expr("pow2", q)))
    add("node", _expr("mul", _expr("pow2", p), q))
    add("node", _expr("mul", q, _expr("pow2", q)))
    add("edge", _expr("mul", _expr("add", qi, neg_qj), s))
    add("edge", _expr("mul", _expr("add", qi, qj), c))
    add("edge", _expr("mul", _expr("add", qi, neg_qj), c))
    add("edge", _expr("mul", _expr("add", qi, qj), s))
    port_vars = {name: _expr("var", name) for name in channel_vars["port"]}
    for left, right in (("w", "p"), ("w", "q"), ("p", "q"), ("q", "es"), ("p", "es"), ("q", "ec"), ("p", "ec"), ("w", "es"), ("w", "ec")):
        add("port", _expr("mul", port_vars[left], port_vars[right]))
    # These expressions depend on the selected parent. Repeated rounds can
    # compose mechanisms absent from the initial atom/pair basis, e.g. p^2*q^2.
    for term in parent_terms:
        channel=term['channel']
        for expr in _subtree_edits(term['expression'],channel_vars[channel],port=channel=='port'):
            add(channel,expr)
    return [out[key] for key in sorted(out)]


def _stable_hash(value: Any, seed: int = 0) -> str:
    payload = f"{seed}:" + _json_dump(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_mutation_pool(
    parent_terms: Sequence[Mapping[str, Any]],
    *,
    mode: str = "generated",
    pool_size: int = 32,
    seed: int = 0,
    max_terms: int = 12,
) -> list[dict[str, Any]]:
    """Enumerate bounded single edits from the fixed library or generated grammar."""
    if pool_size < 1:
        return []
    if mode not in {"library", "generated"}:
        raise ValueError("mode must be 'library' or 'generated'")
    parent = [copy.deepcopy(dict(term)) for term in parent_terms]
    if not parent:
        raise ValueError("parent_terms must not be empty")
    for term in parent:
        term_key(term)
    parent_keys = {term_key(term) for term in parent}
    available = fixed_library() if mode == "library" else _candidate_terms(parent)
    by_kind: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ("add", "delete", "replace")}

    def proposal(kind: str, candidate_terms: list[dict[str, Any]], removed=None, added=None) -> None:
        if sum(_tanh_count(term["expression"]) for term in candidate_terms) > 1:
            return
        key_payload = [kind, sorted(term_key(t) for t in candidate_terms)]
        candidate_key = _stable_hash(key_payload)
        if any(item["candidate_key"] == candidate_key for item in by_kind[kind]):
            return
        by_kind[kind].append(
            {
                "mutation": kind,
                "terms": candidate_terms,
                "removed": copy.deepcopy(removed),
                "added": copy.deepcopy(added),
                "candidate_key": candidate_key,
            }
        )

    if len(parent) < max_terms:
        for term in available:
            key = term_key(term)
            if key in parent_keys:
                continue
            proposal("add", parent + [copy.deepcopy(term)], added=term)
    if len(parent) > 1:
        for index, term in enumerate(parent):
            proposal("delete", parent[:index] + parent[index + 1 :], removed=term)
    for index, old in enumerate(parent):
        for new in available:
            if term_key(new) in parent_keys or new["channel"] != old["channel"]:
                continue
            changed = parent.copy()
            changed[index] = copy.deepcopy(new)
            proposal("replace", changed, removed=old, added=new)

    # Take equal category quotas first; if a category is exhausted, redistribute slots.
    ranked = {
        kind: sorted(items, key=lambda item: _stable_hash(item["candidate_key"], seed))
        for kind, items in by_kind.items()
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < min(pool_size, sum(map(len, ranked.values()))):
        progressed = False
        for kind in ("add", "delete", "replace"):
            if ranked[kind] and len(selected) < pool_size:
                selected.append(ranked[kind].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def propose_mutations(
    beams_terms: Sequence[Any],
    *,
    mode: str = "library",
    seed: int = 0,
    max_candidates: int = 6,
) -> list[dict[str, Any]]:
    """Create mutations for beam parents and return a balanced global candidate pool.

    Every parent receives its own pool of at most 32 proposals.  The returned
    ``parent`` field is the zero-based index into ``beams_terms`` so a runner can
    attach the proposal to the right model state after its own TRAIN screening.
    """
    if mode not in {"library", "generated"}:
        raise ValueError("mode must be 'library' or 'generated'")
    if max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative")
    parents = list(beams_terms)
    if parents and all(isinstance(term, Mapping) and "channel" in term for term in parents):
        parents = [parents]
    per_parent = [
        generate_mutation_pool(parent, mode=mode, pool_size=32, seed=seed + parent_index)
        for parent_index, parent in enumerate(parents)
    ]
    result = []
    offsets = [0] * len(per_parent)
    while len(result) < max_candidates:
        progressed = False
        for parent_index, proposals in enumerate(per_parent):
            if offsets[parent_index] < len(proposals) and len(result) < max_candidates:
                proposal = copy.deepcopy(proposals[offsets[parent_index]])
                offsets[parent_index] += 1
                proposal["parent"] = parent_index
                proposal["mode"] = mode
                result.append(proposal)
                progressed = True
        if not progressed:
            break
    return result


def select_mutation_shortlist(
    pool: Sequence[Mapping[str, Any]], train_scores: Mapping[str, float], *, max_total: int = 6, seed: int = 0
) -> list[dict[str, Any]]:
    """Keep up to six lowest TRAIN-score proposals with balanced edit-type coverage."""
    if max_total < 0:
        raise ValueError("max_total must be nonnegative")
    grouped: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ("add", "delete", "replace")}
    for item in pool:
        kind, key = item["mutation"], item["candidate_key"]
        if kind not in grouped or key not in train_scores:
            continue
        score = float(train_scores[key])
        if not math.isfinite(score):
            continue
        grouped[kind].append({**copy.deepcopy(dict(item)), "train_score": score})
    for kind in grouped:
        grouped[kind].sort(key=lambda item: (item["train_score"], _stable_hash(item["candidate_key"], seed)))
    result = []
    while len(result) < min(max_total, sum(len(v) for v in grouped.values())):
        advanced = False
        for kind in ("add", "delete", "replace"):
            if grouped[kind] and len(result) < max_total:
                result.append(grouped[kind].pop(0))
                advanced = True
        if not advanced:
            break
    return result


def nondominated_front(
    candidates: Sequence[Mapping[str, Any]], *, error_key: str = "error", complexity_key: str = "complexity"
) -> list[dict[str, Any]]:
    """Return candidates not dominated on lower-is-better error and complexity."""
    rows = [copy.deepcopy(dict(row)) for row in candidates]
    for row in rows:
        if not (math.isfinite(float(row[error_key])) and math.isfinite(float(row[complexity_key]))):
            raise ValueError("candidate objectives must be finite")
    front = []
    for index, row in enumerate(rows):
        error, complexity = float(row[error_key]), float(row[complexity_key])
        dominated = any(
            other_index != index
            and float(other[error_key]) <= error
            and float(other[complexity_key]) <= complexity
            and (float(other[error_key]) < error or float(other[complexity_key]) < complexity)
            for other_index, other in enumerate(rows)
        )
        if not dominated:
            front.append(row)
    return front


def simplest_within_one_se(
    candidates: Sequence[Mapping[str, Any]], *, error_key: str = "error", se_key: str = "se", complexity_key: str = "complexity"
) -> dict[str, Any]:
    """Choose the lowest-complexity candidate within one SE of the best mean error."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    rows = [copy.deepcopy(dict(row)) for row in candidates]
    best = min(rows, key=lambda row: float(row[error_key]))
    threshold = float(best[error_key]) + float(best[se_key])
    eligible = [row for row in rows if float(row[error_key]) <= threshold]
    selected = min(eligible, key=lambda row: (float(row[complexity_key]), float(row[error_key])))
    selected["one_se_threshold"] = threshold
    return selected


def paired_group_bootstrap_ci(
    event_differences: Mapping[str | int, float | Sequence[float]], *, n_resamples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Bootstrap the mean paired difference by resampling event groups.

    Each mapping value is either an event-level paired difference or multiple paired
    differences within that event; within-event values are averaged before resampling.
    The sign convention is caller-defined (for example candidate error minus baseline).
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not event_differences:
        raise ValueError("event_differences must not be empty")
    per_event = []
    for event_id in sorted(event_differences, key=str):
        raw = event_differences[event_id]
        values = [float(raw)] if isinstance(raw, (int, float)) else [float(v) for v in raw]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError(f"event {event_id!r} has empty or nonfinite paired differences")
        per_event.append(sum(values) / len(values))
    if any(not math.isfinite(value) for value in per_event):
        raise ValueError("paired differences must be finite")
    rng = __import__("random").Random(seed)
    count = len(per_event)
    estimates = sorted(sum(per_event[rng.randrange(count)] for _ in range(count)) / count for _ in range(n_resamples))
    low_index = max(0, int(math.floor(0.025 * (n_resamples - 1))))
    high_index = min(n_resamples - 1, int(math.ceil(0.975 * (n_resamples - 1))))
    return {
        "mean_difference": sum(per_event) / count,
        "ci95": [estimates[low_index], estimates[high_index]],
        "event_count": count,
        "resamples": n_resamples,
        "seed": seed,
    }


def _read_bus_id(row: Any) -> str:
    if isinstance(row, Mapping):
        for key in ("bus_id", "train_bus", "bus", "bus_id_train"):
            if key in row:
                return str(row[key])
        raise ValueError("allowed TRAIN row must provide bus_id, train_bus, bus, or bus_id_train")
    return str(row)


def _pair_key(duration: Any, xf: Any) -> tuple[float, float]:
    return round(float(duration), 12), round(float(xf), 12)


def _existing_faults(existing_pairs: Iterable[Any]) -> tuple[set[tuple[float, float]], set[tuple[str, float, float]]]:
    global_pairs: set[tuple[float, float]] = set()
    bus_pairs: set[tuple[str, float, float]] = set()
    for item in existing_pairs:
        if isinstance(item, Mapping):
            duration = item.get("duration_s", item.get("duration"))
            xf = item.get("xf", item.get("x_f"))
            if duration is None or xf is None:
                raise ValueError("existing fault rows need duration_s/duration and xf/x_f")
            pair = _pair_key(duration, xf)
            bus = item.get("bus_id", item.get("bus", item.get("train_bus")))
            if bus is None:
                global_pairs.add(pair)
            else:
                bus_pairs.add((str(bus), *pair))
        else:
            values = tuple(item)
            if len(values) == 2:
                global_pairs.add(_pair_key(values[0], values[1]))
            elif len(values) == 3:
                bus_pairs.add((str(values[0]), *_pair_key(values[1], values[2])))
            else:
                raise ValueError("existing pairs must be (duration, xf) or (bus, duration, xf)")
    return global_pairs, bus_pairs


def _condition_key(row: Mapping[str, Any]) -> tuple[str, float, float]:
    return (str(row["bus_id"]), *_pair_key(row["duration_s"], row["xf"]))


def fault_condition_key(row: Mapping[str, Any]) -> str:
    """Return the deterministic key used to attach a TRAIN-only discrepancy score."""
    return _stable_hash(_condition_key(row))


def _stratum(duration: float) -> str:
    if duration < 0.15:
        return "short"
    if duration <= 0.3:
        return "medium"
    return "long"


def _make_val_conditions(buses: Sequence[str], blocked_global, blocked_bus) -> list[dict[str, Any]]:
    result = []
    used = set()
    strata = ("short", "medium", "long")
    for stratum_index, stratum in enumerate(strata):
        choices = [
            (bus, duration, xf)
            for duration in FAULT_DURATIONS_SECONDS
            if _stratum(duration) == stratum
            for xf in FAULT_XF_VALUES
            for bus in buses
            if _pair_key(duration, xf) not in blocked_global
            and (bus, *_pair_key(duration, xf)) not in blocked_bus
        ]
        # Interleave buses so each reserved stratum cycles through the allowed TRAIN buses.
        choices.sort(key=lambda x: (x[1], x[2], buses.index(x[0])))
        choices = sorted(choices, key=lambda x: ((buses.index(x[0]) - stratum_index) % len(buses), x[1], x[2]))
        count_by_bus = {bus: 0 for bus in buses}
        for bus, duration, xf in choices:
            key = (bus, *_pair_key(duration, xf))
            if key in used:
                continue
            # Prefer broad bus coverage before assigning a second condition to a bus.
            if count_by_bus[bus] >= math.ceil(8 / len(buses)) and any(count_by_bus[b] < math.floor(8 / len(buses)) for b in buses):
                continue
            result.append({"bus_id": bus, "duration_s": duration, "xf": xf, "split": "val", "stratum": stratum})
            used.add(key)
            count_by_bus[bus] += 1
            if sum(1 for row in result if row["stratum"] == stratum) == 8:
                break
        if sum(1 for row in result if row["stratum"] == stratum) != 8:
            raise ValueError(f"not enough unused allowed TRAIN conditions to reserve 8 {stratum} val rows")
    return result


def select_discrepant_train_conditions(
    candidate_pool: Sequence[Mapping[str, Any]], discrepancy_scores: Mapping[str, float], *, count: int = 24
) -> list[dict[str, Any]]:
    """Select the highest-scored TRAIN candidates; scores must be supplied by the caller."""
    ranked = []
    for row in candidate_pool:
        key = fault_condition_key(row)
        if key not in discrepancy_scores:
            continue
        score = float(discrepancy_scores[key])
        if not math.isfinite(score):
            continue
        ranked.append((score, key, copy.deepcopy(dict(row))))
    if len(ranked) < count:
        raise ValueError(f"need scores for {count} distinct candidate conditions; received {len(ranked)}")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    seen = set()
    for score, key, row in ranked:
        condition_key = _condition_key(row)
        if condition_key in seen:
            continue
        seen.add(condition_key)
        row["split"] = "train"
        row["discrepancy_score"] = score
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"candidate pool contains only {len(selected)} distinct conditions")
    return selected


def build_new_fault_manifest(
    allowed_train_rows: Sequence[Any],
    existing_pairs: Iterable[Any],
    *,
    frozen_val_conditions: Sequence[Mapping[str, Any]] | None = None,
    discrepancy_scores: Mapping[str, float] | None = None,
    train_count: int = 24,
    pool_limit: int = 96,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a fault-condition manifest from caller-provided allowed TRAIN metadata.

    The function performs no filesystem reads and accepts no labels or outcomes.  Existing
    fault rows carrying a bus exclude only that bus-condition triple; legacy 2-tuples exclude
    the duration/xf pair globally.  A caller-supplied frozen validation manifest is preserved
    byte-for-byte at the JSON-value level.
    """
    buses = list(dict.fromkeys(_read_bus_id(row) for row in allowed_train_rows))
    if not buses:
        raise ValueError("allowed_train_rows must contain at least one allowed TRAIN bus")
    global_pairs, bus_pairs = _existing_faults(existing_pairs)
    if frozen_val_conditions is None:
        val = _make_val_conditions(buses, global_pairs, bus_pairs)
    else:
        val = copy.deepcopy([dict(row) for row in frozen_val_conditions])
        if len(val) != 24:
            raise ValueError("frozen validation manifest must contain exactly 24 conditions")
        if any(_read_bus_id(row) not in buses for row in val):
            raise ValueError("frozen validation manifest contains a bus outside allowed TRAIN rows")
        counts = {group: sum(_stratum(float(row["duration_s"])) == group for row in val) for group in ("short", "medium", "long")}
        if counts != {"short": 8, "medium": 8, "long": 8}:
            raise ValueError("frozen validation manifest must remain stratified 8/8/8")
    val_keys = {_condition_key(row) for row in val}
    all_candidates = []
    for bus, duration, xf in itertools.product(buses, FAULT_DURATIONS_SECONDS, FAULT_XF_VALUES):
        key = (bus, *_pair_key(duration, xf))
        if key in val_keys or key in bus_pairs or _pair_key(duration, xf) in global_pairs:
            continue
        all_candidates.append({"bus_id": bus, "duration_s": duration, "xf": xf, "split": "train_candidate"})
    # Stable seeded order is only for pool truncation; candidate values never use outcomes.
    all_candidates.sort(key=lambda row: _stable_hash(_condition_key(row), seed))
    candidate_pool = all_candidates[:max(0, pool_limit)]
    selected = None
    if discrepancy_scores is not None:
        selected = select_discrepant_train_conditions(candidate_pool, discrepancy_scores, count=train_count)
    if selected is None and len(candidate_pool) < train_count:
        raise ValueError(f"need at least {train_count} disjoint TRAIN candidates; only {len(candidate_pool)} available")
    return {
        "schema_version": 1,
        "val_conditions": val,
        "train_candidate_pool": candidate_pool,
        "train_selected": selected,
        "train_selection_rule": "highest caller-provided discrepancy score; deterministic hash tie-break",
        "seed": seed,
    }


def generate_new_fault_manifest(
    allowedtrainrows: Sequence[Any],
    seed: int = 3701,
    *,
    existing_pairs: Iterable[Any] | None = None,
    frozen_val_conditions: Sequence[Mapping[str, Any]] | None = None,
    discrepancy_scores: Mapping[str, float] | None = None,
    train_count: int = 24,
    pool_limit: int = 96,
) -> dict[str, Any]:
    """Runner-compatible alias returning both canonical and concise manifest keys.

    Existing condition triples are taken only from caller-supplied row metadata.  This
    wrapper never opens paths or looks up row outcomes.
    """
    rows = [copy.deepcopy(row) if isinstance(row, Mapping) else row for row in allowedtrainrows]
    original_conditions = []
    inferred_existing = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if any(key in row for key in ("duration_s", "duration")) and any(key in row for key in ("xf", "x_f")):
            duration = row.get("duration_s", row.get("duration"))
            xf = row.get("xf", row.get("x_f"))
            bus = row.get("bus_id", row.get("train_bus", row.get("bus", row.get("bus_id_train"))))
            condition = {"duration_s": float(duration), "xf": float(xf)}
            if bus is not None:
                condition["bus_id"] = str(bus)
            original_conditions.append(copy.deepcopy(condition))
            inferred_existing.append(copy.deepcopy(condition))
    manifest = build_new_fault_manifest(
        rows,
        inferred_existing if existing_pairs is None else existing_pairs,
        frozen_val_conditions=frozen_val_conditions,
        discrepancy_scores=discrepancy_scores,
        train_count=train_count,
        pool_limit=pool_limit,
        seed=seed,
    )
    manifest["val24"] = copy.deepcopy(manifest["val_conditions"])
    manifest["train_pool"] = copy.deepcopy(manifest["train_candidate_pool"])
    manifest["original_conditions"] = original_conditions
    return manifest
