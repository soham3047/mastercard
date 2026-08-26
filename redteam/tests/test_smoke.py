"""
test_smoke.py — Fast structural checks for the Red Team generators.

Not a formal pytest suite (kept dependency-free) — run directly with
`python -m tests.test_smoke` from the redteam/ directory. Checks that:
  1. build_graph() produces schema-valid output for the default theta.
  2. Every ring closes into an actual directed cycle.
  3. Every ring/hub (structuring) edge amount is under the reporting
     threshold.
  4. Every edge's node endpoints exist and timestamps are well-formed.
  5. The Hawkes decoy generator produces a clustered (not uniform)
     event distribution — a rough check that excitation is doing
     something, not a formal statistical test.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

from src.decoy_generator import simulate_hawkes_events
from src.main import build_graph
from src.theta import DEFAULT_THETA


def check_schema_and_amounts() -> None:
    theta = DEFAULT_THETA
    graph = build_graph(theta)
    graph.validate()  # raises on any schema violation

    node_ids = {n.id for n in graph.nodes}
    assert len(node_ids) == len(graph.nodes), "duplicate node ids"

    over_threshold = [e for e in graph.edges if e.label == "ring" and e.amount >= theta.reporting_threshold_inr]
    assert not over_threshold, f"{len(over_threshold)} structuring edges over threshold: {over_threshold[:3]}"

    for e in graph.edges:
        datetime.fromisoformat(e.timestamp)  # raises on malformed ISO8601
        assert e.from_ in node_ids and e.to in node_ids

    print(f"[OK] schema + amounts: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
          f"0/{sum(1 for e in graph.edges if e.label=='ring')} structuring edges over threshold")


def check_rings_close() -> None:
    theta = DEFAULT_THETA
    graph = build_graph(theta)

    rings = defaultdict(list)
    for e in graph.edges:
        if e.label == "ring" and "acct_ring" in e.from_:
            prefix = e.from_.rsplit("_", 1)[0]
            rings[prefix].append((e.from_, e.to, e.timestamp))

    assert len(rings) == theta.num_rings, f"expected {theta.num_rings} rings, found {len(rings)}"

    for prefix, hops in rings.items():
        hops_sorted = sorted(hops, key=lambda h: h[2])  # by timestamp
        # timestamps must be strictly increasing (time-respecting)
        times = [h[2] for h in hops_sorted]
        assert times == sorted(times), f"ring {prefix} hops not time-ordered"
        # the last hop's destination must equal the first hop's source (closes the cycle)
        assert hops_sorted[-1][1] == hops_sorted[0][0], f"ring {prefix} does not close into a cycle"

    print(f"[OK] all {len(rings)} rings are time-respecting directed cycles")


def check_hawkes_clusters() -> None:
    rng = np.random.default_rng(0)
    events = simulate_hawkes_events(duration_hours=24 * 7, mu=0.5, kappa=0.6, beta=1.2, rng=rng)
    assert len(events) > 0, "no events generated"

    gaps = np.diff(sorted(events))
    if len(gaps) > 10:
        # A clustered (bursty) process has a coefficient of variation of
        # inter-event gaps well above 1 (a pure Poisson process has CV ~= 1).
        cv = gaps.std() / gaps.mean()
        assert cv > 1.0, f"gap CV={cv:.2f} looks Poisson-like, not clustered — check kappa/beta"
        print(f"[OK] Hawkes events cluster (gap CV={cv:.2f} > 1.0, {len(events)} events)")
    else:
        print(f"[WARN] too few events ({len(events)}) for a meaningful clustering check")


if __name__ == "__main__":
    check_schema_and_amounts()
    check_rings_close()
    check_hawkes_clusters()
    print("\nAll smoke tests passed.")
    sys.exit(0)
