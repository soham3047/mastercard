"""
VINEYARD — Blue Team, Stage B7: Final scoring interface.

IMPLEMENTATION NOTE — read before presenting this stage:

This stage doesn't add new math. Its only job is to wire B0-B6 into one
callable, score_batch(graph_json), and lock its output shape — because
Team C (Adversarial Loop) and Team D (Dashboard) build against this
interface directly and won't be reading blue team's internals.

This version is written against the REAL, confirmed source of
b3_vineyards.py, b4_account_features.py, b5_fusion.py, and
b6_explainability.py (previously this file was written against guessed
interfaces, logic-tested only against mocks — that guessing produced three
real, confirmed bugs, described below; this pass fixes all three against
actual source rather than documented behavior).

BUGS FOUND AND FIXED THIS PASS (confirmed against real source + terminal
output, not guessed):

  1. One-shot diagram computation was wired through b3_vineyards.
     PersistentGraph.add_edges(), which internally does `t - t0` on raw
     numeric timestamps (confirmed: b3_vineyards.py has NO iso-parsing
     anywhere — its own __main__ test feeds it plain ints). Real edges
     carry ISO8601 strings per the root README schema, so this crashed
     with TypeError on the first call, in both score_batch() AND
     train_final_model() (--train was never reaching completion either).

     FIX: one-shot scoring (score_batch, train_final_model) now goes
     through b5_fusion.graph_to_diagram_once(graph_json) instead —
     confirmed via b6_explainability.py's own __main__ block, which
     imports and calls it exactly this way, and whose return shape
     ({"diagram":..., "ring_persistence_score":...}) is exactly what B5's
     already-confirmed table (PROGRESS_B.md: ring_1=1.6583, ring_2=1.85,
     fanout/clean=0.0) and B6's own test rely on. b3_vineyards.
     PersistentGraph is reserved for what it's actually built for: genuine
     ROUND-BY-ROUND incremental state in update_diagram() below, which is
     the one place the ISO->numeric bridge is unavoidable.

     STILL AN ASSUMPTION, not fully probed: graph_to_diagram_once()'s own
     source hasn't been read directly, only its usage site in b6's
     __main__ and its name in b5_fusion's function list. High confidence
     given that usage site, but worth one quick sanity check before the
     demo:
         python -c "
         import sys; sys.path.insert(0,'src')
         import b5_fusion
         d = b5_fusion.build_synthetic_dataset()
         name, kind, g, labels = d[0]  # ring_1
         r = b5_fusion.graph_to_diagram_once(g)
         print(name, r['ring_persistence_score'])  # expect 1.6583
         "
     If that doesn't print 1.6583, tell me what it prints instead and
     I'll adjust _diagram_for_graph() below.

  2. _account_features() called b4_account_features's function as
     fn(nodes, edges) -- two positional args. Real signature (confirmed):
     extract_account_features(graph_json, burst_window_days=1.0), ONE
     merged dict. The old call silently passed `edges` as
     `burst_window_days` and crashed one line later on
     `graph_json.get(...)` since `nodes` (a list) has no `.get()`.

     FIX: merge nodes+edges into one graph_json dict before calling, and
     call the real, confirmed function name directly instead of guessing
     across a list of possible names.

  3. _explain_graph() had the identical bug against b6_explainability:
     called fn(nodes, edges) against a real signature of
     explain_graph(graph_json, diagram=None, ring_persistence_score=None,
     threshold=...). This is the exact TypeError("list indices must be
     integers or slices, not str") from the earlier probe -- b6 tried to
     do graph_json["nodes"] on what was actually the `nodes` list itself.

     FIX: merge into graph_json (keeping the ORIGINAL, unconverted ISO
     edges -- b6_explainability.ring_stats() calls datetime.fromisoformat()
     directly on e["timestamp"], so it needs real ISO strings, not the
     numeric-days form B3 wants). Also now passes the already-computed
     diagram/ring_persistence_score through, and threshold=b3_vineyards.
     DEFAULT_THRESHOLD, so B6's REPORTING_THRESHOLD and B3's
     DEFAULT_THRESHOLD stop being two independent copies of the same
     number -- directly resolves the open item PROGRESS_B.md (B6) flagged.

  4. _DiagramHandle silently corrupted data instead of crashing --
     the worst kind of bug. b3_vineyards.update_diagram() returns
     (diagram_dict, state) where diagram_dict is a DICT: {"diagram":[...],
     "ring_persistence_score":..., "round":..., "total_edges":...} (all
     confirmed from real source). The old _DiagramHandle(list) did
     list.__new__(cls, diagram_dict) -- constructing a list from a dict
     iterates its KEYS, so the "diagram" handed back to the adversarial
     loop silently became ['diagram', 'ring_persistence_score', 'round',
     'total_edges'] instead of the actual persistence points. No
     exception anywhere -- Team C would have gotten garbage that looks
     structurally plausible.

     FIX: _DiagramHandle now correctly unpacks diagram_dict["diagram"] as
     its list contents, and keeps ring_persistence_score/round/total_edges
     as extra readable attributes alongside the existing hidden ._state.
     update_diagram() (the incremental, README-contract one) also now
     converts new_edges' ISO timestamps to numeric days before handing
     them to b3_vineyards -- reusing b4_account_features._parse_timestamp
     (already-confirmed, UTC-safe) rather than writing a second copy of
     that logic, in the same spirit B6's own docstring uses to justify not
     re-deriving B2/B3's bifiltration formula.

TWO OPEN ITEMS FROM THE ORIGINAL PASS, STILL RELEVANT:

  (a) [RESOLVED differently than originally planned -- see bug #1 above.]
      update_diagram() (incremental, adversarial-loop contract) still
      needs prev_state to travel across rounds; _DiagramHandle still
      carries it exactly as before, just fixed to unwrap the right shape.

  (b) The root README describes a per-GRAPH output contract (diagram /
      ring_persistence_score / detected_ring_nodes / explanation) AND a
      score_batch() call that returns per-ACCOUNT eval metrics. Resolution
      unchanged from the original pass: graph_json may contain one graph
      or several named graphs (same shape B5/B6 already use). score_batch
      returns BOTH: one contract-shaped object per graph under "graphs",
      plus aggregate precision/recall/f1/auc and per_account_fraud_prob
      pooled across the batch. CONFIRM THIS WITH C/D before they start
      integrating.

RESOLVED (from before, unchanged): _infer_fraud_label() is the fallback
label guesser for training graphs without their own label map, and the
label source for live/real batches inside score_batch() (which have no
ground-truth field per the README's schema). train_final_model() prefers
b5_fusion's real per-graph {account_id: 0/1} label maps (confirmed 4-tuple
dataset shape) whenever present. The score_batch() usage is still an
assumption to verify against Team A's real generator before trusting eval
metrics computed on real data.
"""

from __future__ import annotations
import math
import json
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import b3_vineyards
import b4_account_features

try:
    import b6_explainability
except ImportError:
    b6_explainability = None

try:
    import b5_fusion
except ImportError:
    b5_fusion = None


# ---------------------------------------------------------------------------
# ADAPTER ASSUMPTIONS — the one place to edit if your real B3/B4/B5/B6 names
# differ from what's used here. Everything below this block is written only
# against these names, all confirmed against real source this pass.
# ---------------------------------------------------------------------------

def _diagram_for_graph(nodes: list[dict], edges: list[dict]) -> tuple[list[dict], float]:
    """
    One-shot diagram computation for a whole graph at once (used by
    score_batch and train_final_model — NOT the incremental adversarial-
    loop path, that's update_diagram() near the bottom of this file).

    CONFIRMED via b6_explainability.py's own __main__ block: b5_fusion.
    graph_to_diagram_once(graph_json) -> {"diagram": [...],
    "ring_persistence_score": float, ...}. Self-contained — handles
    whatever timestamp parsing it needs internally, so no ISO/numeric
    bridging required here (unlike the incremental path below, which
    genuinely needs one because b3_vineyards.PersistentGraph has no
    ISO-parsing of its own).
    """
    if b5_fusion is None or not hasattr(b5_fusion, "graph_to_diagram_once"):
        raise AttributeError(
            "Expected b5_fusion.graph_to_diagram_once(graph_json) -> "
            "{'diagram':..., 'ring_persistence_score':...}. "
            f"b5_fusion top-level names: "
            f"{[n for n in dir(b5_fusion) if not n.startswith('_')] if b5_fusion else 'b5_fusion not importable'}. "
            "Update _diagram_for_graph() in b7 to match your real B5 API."
        )
    graph_json = {"nodes": nodes, "edges": edges}
    result = b5_fusion.graph_to_diagram_once(graph_json)
    missing = {"diagram", "ring_persistence_score"} - set(result.keys())
    if missing:
        raise KeyError(
            f"b5_fusion.graph_to_diagram_once() result is missing {missing}. "
            f"Got keys: {list(result.keys())}. Update _diagram_for_graph() "
            "in b7 to match the real key names."
        )
    return result["diagram"], result["ring_persistence_score"]


def _account_features(nodes: list[dict], edges: list[dict]) -> dict[str, dict]:
    """
    CONFIRMED real signature: b4_account_features.extract_account_features
    (graph_json, burst_window_days=1.0) -> {account_id: {feature: value}}.
    Takes ONE merged graph dict, not separate nodes/edges positionally.
    """
    if not hasattr(b4_account_features, "extract_account_features"):
        raise AttributeError(
            "Expected b4_account_features.extract_account_features(graph_json). "
            f"Top-level names: {[n for n in dir(b4_account_features) if not n.startswith('_')]}. "
            "Update _account_features() in b7 to match your real B4 API."
        )
    graph_json = {"nodes": nodes, "edges": edges}
    return b4_account_features.extract_account_features(graph_json)


def _explain_graph(
    nodes: list[dict],
    edges: list[dict],
    diagram: list[dict] | None = None,
    ring_persistence_score: float | None = None,
) -> dict:
    """
    CONFIRMED real signature: b6_explainability.explain_graph(graph_json,
    diagram=None, ring_persistence_score=None, threshold=...) ->
    {"diagram":..., "ring_persistence_score":..., "detected_ring_nodes":
    [...], "explanation": "..."}.

    IMPORTANT: pass the ORIGINAL edges here (real ISO8601 timestamp
    strings), not any numeric-days-converted copy — b6's ring_stats()
    calls datetime.fromisoformat() directly on e["timestamp"] and will
    raise if handed a plain number.

    threshold is passed as b3_vineyards.DEFAULT_THRESHOLD so B3's and B6's
    reporting-threshold constants stop being two independent numbers that
    could drift apart — resolves the open item PROGRESS_B.md (B6) flagged.
    """
    if b6_explainability is None:
        raise ImportError(
            "b6_explainability isn't importable. score_batch needs it for "
            "detected_ring_nodes/explanation — make sure b6_explainability.py "
            "is next to this file (same as B5/B6's setup)."
        )
    if not hasattr(b6_explainability, "explain_graph"):
        raise AttributeError(
            "Expected b6_explainability.explain_graph(graph_json, diagram=, "
            "ring_persistence_score=, threshold=). Top-level names: "
            f"{[n for n in dir(b6_explainability) if not n.startswith('_')]}."
        )
    graph_json = {"nodes": nodes, "edges": edges}
    result = b6_explainability.explain_graph(
        graph_json,
        diagram=diagram,
        ring_persistence_score=ring_persistence_score,
        threshold=getattr(b3_vineyards, "DEFAULT_THRESHOLD", 10000.0),
    )
    missing = {"detected_ring_nodes", "explanation"} - set(result.keys())
    if missing:
        raise KeyError(f"b6_explainability.explain_graph() result missing {missing}.")
    return result


def _infer_fraud_label(node: dict) -> int:
    """
    Fallback label guesser — used only when a training graph doesn't carry
    its own ground-truth label map, and used inside score_batch() for
    live/real batches (which follow the README's node schema and have no
    explicit ground-truth field). VERIFY AGAINST TEAM A'S ACTUAL GENERATOR
    before trusting precision/recall/f1/auc that come from THIS path.
    Prefers an explicit label if the node has one; otherwise treats any
    type other than decoy/clean/normal/legit as fraud.
    """
    for key in ("is_fraud", "fraud_label", "fraud"):
        if key in node:
            return int(bool(node[key]))
    return int(node.get("type", "").lower() not in {"decoy", "clean", "normal", "legit"})

def _safe_feature_value(feats: dict, key: str, default: float = 0.0) -> float:
    """feats.get(key, default), but also catches a PRESENT-but-NaN/inf
    value, not just a missing key. b4_account_features can produce NaN for
    degenerate single-transaction accounts (e.g. sample variance with
    ddof=1 divides by n-1=0 when n=1) — dict.get() alone doesn't guard
    against that, it only guards a missing key. This is the actual source
    of the 'Input X contains NaN' crash in LogisticRegression.fit()."""
    v = feats.get(key, default)
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return default
    return float(v)

def _edges_with_numeric_days(edges: list[dict]) -> list[dict]:
    """
    Converts a copy of `edges` so "timestamp" is numeric days, for feeding
    into b3_vineyards.PersistentGraph — the ONLY consumer in this file that
    genuinely needs that (its _time_norm does `t - t0` on raw numbers with
    no ISO-parsing of its own, confirmed from real source).

    Reuses b4_account_features._parse_timestamp rather than writing a
    second copy of the ISO/UTC handling logic — it already: (1) passes
    numeric input through unchanged, so this is safe to call even if a
    caller hands in already-numeric timestamps, and (2) forces naive
    (no-offset) ISO strings to UTC instead of the machine's local timezone,
    which is exactly the cross-machine consistency gotcha B4's own
    docstring flags.
    """
    converted = []
    for e in edges:
        e2 = dict(e)
        e2["timestamp"] = b4_account_features._parse_timestamp(e["timestamp"])
        converted.append(e2)
    return converted


# ---------------------------------------------------------------------------
# (a) update_diagram — hides B3's (diagram_dict, state) shape behind the
# bare new_diagram signature the root README promises. This is the ONLY
# function in this file that talks to b3_vineyards.PersistentGraph — it's
# built for genuine round-by-round incremental use, which is exactly what
# the adversarial loop needs here.
# ---------------------------------------------------------------------------

class _DiagramHandle(list):
    """
    Looks and serializes like the plain diagram list the root README
    documents (iterate it, json.dumps it, whatever) — but also carries the
    live PersistentGraph B3 needs for the *next* incremental update, plus
    the extra fields b3_vineyards.update_diagram()'s real dict return
    includes, riding in attributes nothing else needs to know about.

    b3_vineyards.update_diagram() returns (diagram_dict, state) where
    diagram_dict = {"diagram": [...], "ring_persistence_score": float,
    "round": int, "total_edges": int} — CONFIRMED from real source. The
    list contents here are diagram_dict["diagram"] specifically, not the
    whole dict (constructing a list from a dict would silently iterate its
    keys instead — this was the actual, confirmed bug in the previous pass).
    """

    def __new__(cls, diagram_dict, state):
        return super().__new__(cls, diagram_dict["diagram"])

    def __init__(self, diagram_dict, state):
        super().__init__(diagram_dict["diagram"])
        self._state = state
        self.ring_persistence_score = diagram_dict.get("ring_persistence_score")
        self.round = diagram_dict.get("round")
        self.total_edges = diagram_dict.get("total_edges")


def update_diagram(prev_diagram, new_edges):
    """
    Public contract per root README: update_diagram(prev_diagram, new_edges)
    -> new_diagram.

    prev_diagram: None on the first call, or whatever this function
    returned last time. new_diagram: same shape, safe to hand straight
    back in on the next call.

    new_edges: list of {"from", "to", "timestamp", "amount"} with ISO8601
    "timestamp" strings, per the root README schema — converted to numeric
    days here before reaching b3_vineyards, which has no ISO-parsing of
    its own (confirmed from real source).
    """
    if not hasattr(b3_vineyards, "update_diagram"):
        raise AttributeError(
            "Expected b3_vineyards.update_diagram(state, new_edges) -> "
            "(diagram_dict, state). Top-level names found: "
            f"{[n for n in dir(b3_vineyards) if not n.startswith('_')]}."
        )
    prev_state = getattr(prev_diagram, "_state", None)
    numeric_edges = _edges_with_numeric_days(new_edges)
    diagram_dict, state = b3_vineyards.update_diagram(prev_state, numeric_edges)
    return _DiagramHandle(diagram_dict, state)


# ---------------------------------------------------------------------------
# Fused model — B7 fits ONE final model on all available labeled graphs
# (not leave-one-out; that CV was B5's job, for measuring generalization).
# This is what score_batch actually calls at inference time.
# ---------------------------------------------------------------------------

@dataclass
class FusedModel:
    scaler: StandardScaler
    clf: LogisticRegression
    feature_names: list[str]  # feature_names[0] is always "ring_persistence_score"

    # FusedModel._vector:
    def _vector(self, account_feats: dict, ring_score: float) -> np.ndarray:
        row = [ring_score] + [_safe_feature_value(account_feats, k) for k in self.feature_names[1:]]
        return np.array(row, dtype=float).reshape(1, -1)

    def predict_proba(self, account_feats: dict, ring_score: float) -> float:
        x_scaled = self.scaler.transform(self._vector(account_feats, ring_score))
        return float(self.clf.predict_proba(x_scaled)[0, 1])

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FusedModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def _normalize_training_graphs(training_graphs) -> dict[str, dict]:
    """
    Accepts whatever shape b5_fusion's dataset actually comes back as and
    normalizes it to {graph_name: {"nodes": [...], "edges": [...],
    "labels": {account_id: 0/1}}} — "labels" is only present in the
    returned dict when the source item actually carried one.

    Handles:
      - already-correct dict: {graph_name: {"nodes":..., "edges":...}}
      - CONFIRMED real b5_fusion.py shape (via runtime probe): list of
        (name, kind, graph_dict, label_dict) 4-tuples, e.g.
        ("ring_1", "ring", {"nodes":..., "edges":...},
         {"r1_r000": 1, "r1_r001": 1, ..., "r1_d0a": 0, ...})
        label_dict maps account_id -> 0/1 ground-truth fraud label,
        straight from B5's own dataset construction.
      - list of (name, kind, graph_dict) 3-tuples (no label dict attached)
      - list of (name, graph_dict) 2-tuples
      - list of dicts, each carrying its own name under "name"/"graph_name"/
        "id"/"kind" (falls back to "graph_0", "graph_1", ... if none present,
        de-duplicated if a name repeats, e.g. two graphs both labeled "ring")
    """
    if isinstance(training_graphs, dict):
        return training_graphs

    if not isinstance(training_graphs, list):
        raise TypeError(
            f"training_graphs must be a dict or list, got {type(training_graphs)}. "
            "Check what your b5_fusion dataset function actually returns."
        )

    normalized: dict[str, dict] = {}
    for i, item in enumerate(training_graphs):
        labels = None
        if isinstance(item, tuple) and len(item) == 4:
            # confirmed real shape: (name, kind, graph_dict, label_dict)
            name, _kind, g, labels = item
            name = str(name)
        elif isinstance(item, tuple) and len(item) == 3:
            name, _kind, g = item
            name = str(name)
        elif isinstance(item, tuple) and len(item) == 2:
            name, g = item
            name = str(name)
        elif isinstance(item, dict):
            g = item
            name = None
            for key in ("name", "graph_name", "id", "kind"):
                if key in item:
                    name = str(item[key])
                    break
            name = name or f"graph_{i}"
        else:
            raise TypeError(
                f"Don't know how to read training_graphs[{i}] of type {type(item)}. "
                "Expected a dict with nodes/edges (optionally with a name/id/kind "
                "field), or a (name, graph_dict) / (name, kind, graph_dict) / "
                "(name, kind, graph_dict, label_dict) tuple. Update "
                "_normalize_training_graphs() in b7 to match your real shape."
            )
        base_name, suffix = name, 1
        while name in normalized:
            suffix += 1
            name = f"{base_name}_{suffix}"
        if "nodes" not in g or "edges" not in g:
            raise KeyError(
                f"training_graphs[{i}] (name={name!r}) is missing 'nodes' or "
                f"'edges'. Keys found: {list(g.keys())}."
            )
        entry: dict[str, Any] = {"nodes": g["nodes"], "edges": g["edges"]}
        if labels is not None:
            if not isinstance(labels, dict):
                raise TypeError(
                    f"training_graphs[{i}] (name={name!r}) has a 4th tuple "
                    f"element that isn't a dict (got {type(labels)}). Expected "
                    "{account_id: 0/1}. Update _normalize_training_graphs() if "
                    "b5_fusion's 4th element means something else."
                )
            entry["labels"] = labels
        normalized[name] = entry
    return normalized


def train_final_model(training_graphs, save_path: str = "b7_model.pkl") -> FusedModel:
    """
    training_graphs: {graph_name: {"nodes": [...], "edges": [...]}}, or a
    list in any of the shapes _normalize_training_graphs() handles above —
    normalized automatically either way.

    Fits one final logistic-regression model on ALL given graphs combined.
    Uses each graph's own ground-truth label map when one is present
    (b5_fusion's real 4-tuple shape provides this); falls back to
    _infer_fraud_label()'s guess only for graphs that don't carry one.
    """
    training_graphs = _normalize_training_graphs(training_graphs)

    rows: list[list[float]] = []
    labels: list[int] = []
    feature_names: list[str] | None = None
    used_ground_truth = False
    used_guess_fallback = False

    for gname, g in training_graphs.items():
        nodes, edges = g["nodes"], g["edges"]
        label_map = g.get("labels")  # None if this graph came in without one
        _, ring_score = _diagram_for_graph(nodes, edges)
        acct_feats = _account_features(nodes, edges)

        for acct_id, f in acct_feats.items():
            bad = [k for k, v in f.items() if isinstance(v, float) and not math.isfinite(v)]
            if bad:
                print(f"[diagnostic] {gname}/{acct_id}: non-finite feature(s) {bad}")

        for node in nodes:
            acct_id = node["id"]
            feats = acct_feats.get(acct_id, {})
            
        for node in nodes:
            acct_id = node["id"]
            feats = acct_feats.get(acct_id, {})
            if feature_names is None:
                feature_names = ["ring_persistence_score"] + sorted(feats.keys())
            row = [ring_score] + [_safe_feature_value(feats, k) for k in feature_names[1:]]
            rows.append(row)

            if label_map is not None and acct_id in label_map:
                labels.append(int(label_map[acct_id]))
                used_ground_truth = True
            else:
                labels.append(_infer_fraud_label(node))
                used_guess_fallback = True

    if not rows:
        raise ValueError("training_graphs produced zero accounts — nothing to fit.")

    nan_counts: dict[str, int] = {}
    for row in rows:
        for name, val in zip(feature_names, row):
            if not math.isfinite(val):
                nan_counts[name] = nan_counts.get(name, 0) + 1
    # (rows have already been sanitized by _safe_feature_value at this point,
    # so this loop won't actually find anything post-fix — see the debug
    # block below instead, which checks BEFORE sanitizing)
    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)
    model = FusedModel(scaler=scaler, clf=clf, feature_names=feature_names)
    model.save(save_path)

    if used_ground_truth and used_guess_fallback:
        label_source = (
            "a MIX of b5_fusion ground-truth labels and _infer_fraud_label() "
            "guesses — check which graphs lacked a label map, that's "
            "unexpected if every graph came from the same dataset"
        )
    elif used_ground_truth:
        label_source = "b5_fusion's ground-truth labels"
    else:
        label_source = "_infer_fraud_label() guesses (no ground-truth label map found on any graph)"

    print(
        f"Trained final fused model on {len(y)} accounts across "
        f"{len(training_graphs)} graphs ({int(y.sum())} fraud / "
        f"{int(len(y) - y.sum())} normal) using {label_source}. "
        f"Saved to {save_path}."
    )
    if used_ground_truth and not used_guess_fallback:
        print(
            "Sanity check: PROGRESS_B.md (B5) reports 22 fraud / 17 normal "
            "across 39 accounts on this same 6-graph set. If the counts "
            "above don't match that, something changed upstream (dataset "
            "regenerated, graphs added/removed, etc.) — worth a quick look "
            "before trusting the trained model."
        )
    return model


# ---------------------------------------------------------------------------
# score_batch — the interface C and D build against.
# ---------------------------------------------------------------------------

def score_batch(
    graph_json: dict,
    model_path: str = "b7_model.pkl",
    fraud_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    graph_json: either
      (a) a single graph  {"nodes": [...], "edges": [...]}, or
      (b) a batch of named graphs {"graph_name": {"nodes":..., "edges":...}, ...}
    (b) matches the shape B5/B6 already used (ring_1, fanout_1, ...); (a)
    is the plain single-graph shape the root README's input-schema example
    shows. Auto-detected by whether "nodes" is a top-level key.

    Returns:
      {
        "graphs": {
            graph_name: {diagram, ring_persistence_score,
                          detected_ring_nodes, explanation}, ...
        },
        "precision": float | None, "recall": float | None,
        "f1": float | None, "auc": float | None,
        "per_account_fraud_prob": {account_id: prob, ...}
      }

    precision/recall/f1/auc are None if ground-truth fraud labels aren't
    inferable from the input (e.g. a genuinely unlabeled batch, or a batch
    that's all-fraud/all-normal) — per_account_fraud_prob is always fully
    populated regardless. NOTE: this path uses _infer_fraud_label()'s guess
    (real batches follow the README's node schema, not b5_fusion's dataset
    shape) — still worth verifying against Team A's real generator.
    """
    graphs = {"graph": graph_json} if "nodes" in graph_json else graph_json

    try:
        model = FusedModel.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No trained model at {model_path!r}. Call train_final_model(...) "
            "once (e.g. `python b7_score_batch.py --train`) before calling score_batch()."
        )

    per_graph_out: dict[str, dict] = {}
    per_account_prob: dict[str, float] = {}
    all_true: list[int] = []
    all_pred_prob: list[float] = []

    for gname, g in graphs.items():
        nodes, edges = g["nodes"], g["edges"]
        diagram, ring_score = _diagram_for_graph(nodes, edges)
        acct_feats = _account_features(nodes, edges)
        explain = _explain_graph(nodes, edges, diagram=diagram, ring_persistence_score=ring_score)

        per_graph_out[gname] = {
            "diagram": diagram,
            "ring_persistence_score": ring_score,
            "detected_ring_nodes": explain["detected_ring_nodes"],
            "explanation": explain["explanation"],
        }

        for node in nodes:
            acct_id = node["id"]
            feats = acct_feats.get(acct_id, {})
            prob = model.predict_proba(feats, ring_score)
            per_account_prob[acct_id] = prob
            all_pred_prob.append(prob)
            all_true.append(_infer_fraud_label(node))

    metrics: dict[str, float | None] = {
        "precision": None,
        "recall": None,
        "f1": None,
        "auc": None,
    }
    if all_true and len(set(all_true)) > 1:
        pred_label = [1 if p >= fraud_threshold else 0 for p in all_pred_prob]
        metrics["precision"] = precision_score(all_true, pred_label, zero_division=0)
        metrics["recall"] = recall_score(all_true, pred_label, zero_division=0)
        metrics["f1"] = f1_score(all_true, pred_label, zero_division=0)
        try:
            metrics["auc"] = roc_auc_score(all_true, all_pred_prob)
        except ValueError:
            metrics["auc"] = None  # only one class present in this batch

    return {
        "graphs": per_graph_out,
        **metrics,
        "per_account_fraud_prob": per_account_prob,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--train" in sys.argv:
        if b5_fusion is None:
            print("b5_fusion isn't importable — make sure b5_fusion.py is next "
                  "to this file, or pass a training set built some other way "
                  "directly to train_final_model(training_graphs).")
            sys.exit(1)

        training_graphs = None
        for name in ("SYNTHETIC_GRAPHS", "build_synthetic_dataset", "make_dataset"):
            attr = getattr(b5_fusion, name, None)
            if attr is None:
                continue
            training_graphs = attr() if callable(attr) else attr
            break

        if training_graphs is None:
            print("Couldn't find the synthetic dataset in b5_fusion.py automatically.")
            print(f"Top-level names: {[n for n in dir(b5_fusion) if not n.startswith('_')]}")
            print("Call train_final_model(your_training_graphs_dict) directly instead.")
            sys.exit(1)

        try:
            train_final_model(training_graphs)
        except (TypeError, KeyError, AttributeError) as e:
            print(f"train_final_model couldn't read the dataset shape: {e}\n")
            print("Raw structure b5_fusion actually returned:")
            print(f"  type: {type(training_graphs)}")
            if isinstance(training_graphs, list) and training_graphs:
                first = training_graphs[0]
                print(f"  length: {len(training_graphs)}")
                print(f"  first item type: {type(first)}")
                if isinstance(first, (list, tuple)):
                    print(f"  first item tuple/list length: {len(first)}")
                    print(f"  first item element types, in order: {[type(x).__name__ for x in first]}")
                    for idx, el in enumerate(first):
                        if isinstance(el, dict):
                            print(f"    element[{idx}] is a dict with keys: {list(el.keys())}")
                        else:
                            print(f"    element[{idx}] ({type(el).__name__}): {repr(el)[:200]}")
                else:
                    print(f"  first item (repr, truncated): {repr(first)[:1500]}")
            elif isinstance(training_graphs, dict) and training_graphs:
                first_key = next(iter(training_graphs))
                print(f"  first key: {first_key!r}")
                print(f"  first value (repr, truncated): {repr(training_graphs[first_key])[:1500]}")
            print(
                "\nPaste that printed structure back to me and I'll fix "
                "_normalize_training_graphs()/train_final_model() to match it exactly."
            )
            sys.exit(1)
    else:
        if len(sys.argv) < 2:
            print("Usage: python b7_score_batch.py --train")
            print("       python b7_score_batch.py <graph_json_path>")
            sys.exit(1)
        with open(sys.argv[1]) as f:
            loaded_graph_json = json.load(f)
        result = score_batch(loaded_graph_json)
        print(json.dumps(result, indent=2, default=str))