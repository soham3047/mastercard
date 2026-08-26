"""
bottleneck.py — Stage C2 bottleneck distance between consecutive rounds'
persistence diagrams.

DEVIATION FROM THE MASTER PROMPT, FLAGGED NOT SILENT: the master prompt
says to use B's `update_diagram(prev_diagram, new_edges)` for incremental
updates between rounds instead of full recompute. That function's actual
module was never provided to this track (only referenced in
PROGRESS_B.md's B3 section — no file). More importantly, its contract
(prev_diagram + a NEW_EDGES delta) doesn't map cleanly onto this loop's
actual round structure: each round calls A's real build_graph(theta),
which samples an entirely NEW synthetic graph from theta.seed each time —
it does not append edges onto the previous round's graph. There is no
natural "new_edges since last round" to hand update_diagram here; the two
rounds' graphs are independent samples, not one accumulating graph.

Given that, this file computes bottleneck distance directly from the two
full diagrams B7's score_batch already returns each round. Note this
doesn't cost anything update_diagram would have saved: since score_batch
already recomputes the diagram from scratch every call under this round
design, the "avoid full recompute" benefit update_diagram exists for was
never available here in the first place — the missing efficiency isn't
being left on the table by this file, it's a property of how rounds are
structured upstream.

OPEN QUESTION FOR THE TEAM: if genuine incremental diagram updates are
wanted, the round loop's design needs to change so each round mutates the
PREVIOUS round's graph (e.g. via small theta deltas that add/remove
edges) rather than resampling from scratch each time — that's a decision
above this file, flagged in PROGRESS_C.md, not resolved here.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from persim import bottleneck as _persim_bottleneck
    _HAVE_PERSIM = True
    _PERSIM_IMPORT_ERROR: Optional[str] = None
except ImportError as e:
    _HAVE_PERSIM = False
    _PERSIM_IMPORT_ERROR = str(e)


def _diagram_to_array(diagram: List[Dict[str, Any]], dimension: int = 1) -> np.ndarray:
    """
    Filters B7's diagram schema down to one homology dimension and
    converts to the Nx2 (birth, death) array persim expects.

    B7's schema: {"birth": float, "death": float|null, "dimension": int,
    "filtration": str}. `death: null` (an essential/infinite class) is
    mapped to `birth + 1e6` rather than `np.inf` — a documented, finite
    sentinel approximation, since persim's bottleneck implementation
    isn't guaranteed to handle literal infinities identically across
    versions. Flagged here rather than silently risking a NaN/inf
    propagation bug of exactly the kind B7's own synopsis found and fixed
    in the fusion model.
    """
    pts = []
    for pt in diagram:
        if pt.get("dimension") != dimension:
            continue
        birth = float(pt["birth"])
        death = pt.get("death")
        death = float(death) if death is not None else birth + 1e6
        pts.append([birth, death])
    if not pts:
        return np.empty((0, 2))
    return np.array(pts)


def compute_bottleneck_distance(
    prev_diagram: Optional[List[Dict[str, Any]]],
    curr_diagram: List[Dict[str, Any]],
    dimension: int = 1,
) -> float:
    """
    Bottleneck distance between two rounds' persistence diagrams,
    restricted to homology dimension `dimension` (default 1 = the H1
    ring feature B0-B3 build the whole detection pipeline around — see
    PROGRESS_B.md).

    Returns 0.0 if prev_diagram is None (round 1, no prior round — same
    convention as C0's original stub).

    Returns -1.0 (a loud, documented sentinel — NOT a plausible-looking
    random number like C0/C1's stub) if persim isn't installed, so a
    missing dependency is diagnosable rather than silently producing fake
    data.
    """
    if prev_diagram is None:
        return 0.0
    if not _HAVE_PERSIM:
        import warnings
        warnings.warn(
            f"compute_bottleneck_distance: persim import failed "
            f"({_PERSIM_IMPORT_ERROR}); returning -1.0 sentinel instead of a "
            f"real distance. Fix the underlying import (check `pip show persim` "
            f"and its transitive deps, e.g. hopcroftkarp) before trusting any "
            f"bottleneck-distance number downstream.",
            RuntimeWarning,
            stacklevel=2,
        )
        return -1.0  # sentinel: persim not installed — see this file's docstring
    a = _diagram_to_array(prev_diagram, dimension=dimension)
    b = _diagram_to_array(curr_diagram, dimension=dimension)
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(_persim_bottleneck(a, b))
