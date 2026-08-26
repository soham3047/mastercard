"""
redteam_adapter.py — Stage C2 adapter around A's REAL generator.

Wraps redteam.src.main.build_graph + redteam.src.theta.Theta so the
adversarial loop keeps working in plain theta DICTS (as it has since
C0/C1) while A's real functions work in Theta OBJECTS and
TransactionGraph objects.

FALLBACK (C4 follow-up): mirrors blueteam_adapter.py's existing pattern
exactly — if redteam.src.theta/main aren't importable, build_real_graph
falls back to a flagged synthetic graph generator instead of crashing at
import time, so run_real_rounds.py (and test_c2_smoke.py) can run end to
end before A's real files exist. BUILD_GRAPH_IS_REAL tells you which
path fired, same convention as blueteam_adapter's SCORE_BATCH_IS_REAL —
check it before trusting output as A's real generator.
"""
from __future__ import annotations
import random
from typing import Any, Dict, List

BUILD_GRAPH_IS_REAL = False
Theta = None
DEFAULT_THETA = None
_build_graph = None

try:
    from redteam.src.theta import Theta, DEFAULT_THETA  # noqa: F811
    from redteam.src.main import build_graph as _build_graph  # noqa: F811
    BUILD_GRAPH_IS_REAL = True
except ImportError:
    BUILD_GRAPH_IS_REAL = False

# See bo_real_space.py's docstring for the full reasoning behind this
# exact 4-field subset.
TUNED_KEYS = ["ring_size_lambda", "decoy_density", "hop_timing_lambda", "fan_out_ratio"]


def theta_dict_to_real_theta(theta_dict: Dict[str, Any], seed: int) -> "Theta":
    """Merges a tuned-subset theta dict (from bo_real_space's GP) onto
    DEFAULT_THETA's other 12 fields to build a real Theta object.
    Only called on the real path — see build_real_graph."""
    base = DEFAULT_THETA.as_dict()
    base.update({k: theta_dict[k] for k in TUNED_KEYS if k in theta_dict})
    base["seed"] = seed
    return Theta.from_dict(base)


def _fallback_build_graph(theta_dict: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """
    Used ONLY if redteam.src.theta/main aren't importable. Produces a
    synthetic TransactionGraph matching the root README's schema exactly
    (nodes: id/type/created_at; edges: from/to/timestamp/amount/label),
    loosely driven by the tuned theta_dict so the GP isn't optimizing
    against pure noise. NOT a substitute for A's real generator — no
    ring-closure structure, no decoy realism, no Hawkes timing. Exists
    only so downstream (blueteam_adapter, bottleneck) has something
    schema-shaped to consume; must never be mistaken for A's real
    attack generator. Mirrors blueteam_adapter.py's
    _fallback_score_batch in spirit and in flagging.
    """
    rng = random.Random(seed)
    ring_size = max(3, int(round(theta_dict.get("ring_size_lambda", 5.0))))
    fan_out = theta_dict.get("fan_out_ratio", 0.5)
    n_decoys = int(round(ring_size * theta_dict.get("decoy_density", 0.5) * 3))

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    for i in range(ring_size):
        nodes.append({"id": f"ring_{i:03d}", "type": "ring", "created_at": "2026-01-01T00:00:00Z"})
    for i in range(n_decoys):
        nodes.append({"id": f"decoy_{i:03d}", "type": "decoy", "created_at": "2026-01-01T00:00:00Z"})

    for i in range(ring_size):
        j = (i + 1) % ring_size
        edges.append({
            "from": f"ring_{i:03d}", "to": f"ring_{j:03d}",
            "timestamp": "2026-01-01T00:00:00Z",
            "amount": round(rng.uniform(500, 9800), 2),
            "label": "ring",
        })
    for i in range(n_decoys):
        src = f"ring_{rng.randrange(ring_size):03d}" if rng.random() < fan_out else f"decoy_{i:03d}"
        edges.append({
            "from": src, "to": f"decoy_{i:03d}",
            "timestamp": "2026-01-01T00:00:00Z",
            "amount": round(rng.uniform(50, 5000), 2),
            "label": "decoy",
        })

    return {"nodes": nodes, "edges": edges}


def build_real_graph(theta_dict: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """generate_fn's real counterpart, called from inside score_fn (see
    run_real_rounds.py): tuned theta dict -> real Theta -> A's real
    TransactionGraph -> plain JSON dict, ready for B's score_batch.
    Falls back to a flagged synthetic graph if A's real modules aren't
    importable — check BUILD_GRAPH_IS_REAL before trusting output as
    A's real generator.

    Theta.validate() is called internally by A's build_graph() (per
    main.py) — not duplicated here, on the real path."""
    if not BUILD_GRAPH_IS_REAL:
        return _fallback_build_graph(theta_dict, seed=seed)
    theta = theta_dict_to_real_theta(theta_dict, seed=seed)
    graph = _build_graph(theta)
    return graph.to_dict()
