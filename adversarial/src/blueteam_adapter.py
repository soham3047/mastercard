"""
blueteam_adapter.py — Stage C2 adapter around B's REAL score_batch.

FLAGGED, NOT GUESSED: the actual file `blueteam/src/b7_score_batch.py`
was never received by this track — only PROGRESS_B.md's prose synopsis
of its signature and output shape. This adapter attempts a real import
and makes the outcome loud and checkable (`SCORE_BATCH_IS_REAL`) rather
than silently succeeding-or-failing.

IMPORT PATH UNCERTAINTY: PROGRESS_B.md's own "Commands to run" for B7
show `cd blueteam/src && python b7_score_batch.py ring_1.json` — a flat
script invocation, not `python -m src.b7_score_batch` the way A's repo
is structured (A's uses `-m src.main` with relative imports throughout).
That's a real signal blueteam/ may NOT be set up as an importable package
the same way redteam/ is. This adapter tries the package-style import
first, then falls back to a sys.path insertion + flat import. Whichever
one actually succeeds should be confirmed with B — don't assume the
first branch is the one that fired without checking SCORE_BATCH_IS_REAL
and the printed warning in run_real_rounds.py.
"""
from __future__ import annotations
import os
import sys
from typing import Any, Dict

SCORE_BATCH_IS_REAL = False
_score_batch = None

try:
    from blueteam.src.b7_score_batch import score_batch as _score_batch  # package-style
    SCORE_BATCH_IS_REAL = True
except ImportError:
    try:
        _blueteam_src = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "blueteam", "src")
        )
        if _blueteam_src not in sys.path:
            sys.path.insert(0, _blueteam_src)
        from b7_score_batch import score_batch as _score_batch  # flat-script style
        SCORE_BATCH_IS_REAL = True
    except ImportError:
        SCORE_BATCH_IS_REAL = False


def _fallback_score_batch(graph_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Used ONLY if both import attempts above fail. Unlike C0/C1's
    stub_scorer (pure random noise, theta-independent), this fallback is
    at least graph-size-dependent, so the GP isn't fitting to literal
    noise if this path is ever silently hit — but it is NOT a substitute
    for B's real detector and must never be the path used in a demo.
    Deliberately has NO "diagram" key, so loop.py's bottleneck_fn hook
    (C2) cleanly falls back to its own random-stub bottleneck rather than
    computing a bottleneck distance against fabricated diagrams.
    """
    import random
    edges = graph_json.get("edges", [])
    n_edges = len(edges)
    n_ring_edges = sum(1 for e in edges if e.get("label") == "ring")
    rng = random.Random(n_edges if n_edges else 1)
    recall = round(min(0.95, 0.3 + 0.4 * (n_ring_edges / max(n_edges, 1))), 4)
    precision = round(rng.uniform(0.2, 0.9), 4)
    return {
        "ring_persistence_score": 0.0,
        "detected_ring_nodes": [],
        "explanation": (
            "FALLBACK STUB (blueteam_adapter.py) — b7_score_batch.py "
            "import failed; this is NOT B's real detector output."
        ),
        "precision": precision,
        "recall": recall,
        "f1": 0.0,
        "auc": 0.0,
        "per_account_fraud_prob": {},
    }


def score_batch(graph_json: Dict[str, Any]) -> Dict[str, Any]:
    """Real score_batch if importable, else the flagged fallback above.
    Check SCORE_BATCH_IS_REAL before trusting output as B's real numbers."""
    if SCORE_BATCH_IS_REAL:
        # --- Pickle-load compatibility shim (PROGRESS_C.md Addendum 3) ---
        # b7_model.pkl was pickled while b7_score_batch.py ran as __main__
        # (B trained it via `python b7_score_batch.py --train ...`), so the
        # class reference baked into the pickle is "__main__.FusedModel",
        # not "b7_score_batch.FusedModel". Whenever this adapter imports
        # b7_score_batch normally instead of running it directly, __main__
        # is whatever process actually launched this run (test_c2_smoke.py,
        # run_real_rounds.py, D's backend, ...) -- which has no FusedModel
        # attribute -- so FusedModel.load()'s pickle.load(f) fails with:
        #   AttributeError: module '__main__' has no attribute 'FusedModel'
        #
        # Fix: pull FusedModel out of whichever module _score_batch is
        # actually bound to (works whether the package-style or flat-script
        # import above is the one that fired, per this file's own
        # IMPORT PATH UNCERTAINTY note) and alias it onto
        # sys.modules["__main__"] before every call, so the unpickler's
        # lookup resolves no matter what launched this process. Cheap and
        # idempotent -- safe to run on every call, not just once.
        _fused_model_cls = getattr(_score_batch, "__globals__", {}).get("FusedModel")
        if _fused_model_cls is not None:
            sys.modules["__main__"].FusedModel = _fused_model_cls
        return _score_batch(graph_json)
    return _fallback_score_batch(graph_json)
