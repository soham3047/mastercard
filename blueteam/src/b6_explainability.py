"""
VINEYARD — Blue Team, Stage B6: Explainability layer.

IMPLEMENTATION NOTE — read before presenting this stage:
B0-B3 established THAT a ring exists in a graph (a persistent H1 feature).
B5 explicitly flagged that WHICH accounts are the ring members, as opposed
to innocent bystanders sharing the same graph, was left unsolved -- that's
this stage's job.

DESIGN DECISION: node-level ring attribution here does NOT re-derive B2/B3's
combined (time, amount-proximity) bifiltration order. It doesn't need to --
finding WHICH nodes form a cycle is a plain graph-structure question
(directed-cycle detection), independent of WHEN/HOW persistent that cycle
scores. Reusing or reimplementing the bifiltration formula here would risk
exactly the "third copy drifting out of sync" problem B5's docstring
already warned about, for no benefit -- so this stage deliberately doesn't
touch it. It only consumes: (1) the raw graph_json Red Team hands over, and
(2) B0-B3/B5's already-computed diagram / ring_persistence_score, to build
the final output record. One source of truth for scoring (B0-B3), one
source of truth for attribution + narrative (this file).

A GENUINE LIMITATION worth stating up front, not burying: cycle detection
here uses networkx's simple_cycles on the directed transaction graph, which
is exact but combinatorially expensive on dense graphs with many
overlapping cycles. Fine at this project's scale (small per-batch
subgraphs); would need a smarter approach (e.g. restricting the search to
the specific time window / edges that produced the H1 feature B3 already
found) if graphs get much larger. Flagging now, not treating it as solved.

DEPENDENCY: this stage introduces networkx, not used by B0-B5. Install with
`pip install networkx` (or `--break-system-packages` on this project's
Python setup) if not already in requirements.txt.
"""
from datetime import datetime

import networkx as nx

REPORTING_THRESHOLD = 10000  # ₹ -- placeholder constant. If B1/B2 already
# define a shared reporting-threshold constant, import and reuse that one
# instead of this local copy, to avoid two numbers drifting out of sync.


def _build_digraph(graph_json):
    g = nx.DiGraph()
    for n in graph_json["nodes"]:
        g.add_node(n["id"])
    for e in graph_json["edges"]:
        g.add_edge(e["from"], e["to"], timestamp=e["timestamp"], amount=e["amount"])
    return g


def find_ring_cycles(graph_json):
    """Returns a list of directed cycles (each a list of node ids, in
    traversal order) found in the transaction graph. A ring/layering attack
    IS a directed cycle: money flows around and returns to its origin.
    Fan-out/star patterns (hub with one-way spokes) are DAGs and never
    produce a directed cycle, so this naturally returns [] for them with no
    special-casing needed."""
    g = _build_digraph(graph_json)
    return [list(c) for c in nx.simple_cycles(g)]


def _cycle_edges(graph_json, cycle):
    """Pulls the actual edge records (timestamp, amount) for consecutive
    node pairs around a detected cycle, wrapping the last pair back to the
    first node."""
    edge_lookup = {(e["from"], e["to"]): e for e in graph_json["edges"]}
    n = len(cycle)
    edges = []
    for i in range(n):
        frm, to = cycle[i], cycle[(i + 1) % n]
        e = edge_lookup.get((frm, to))
        if e is not None:
            edges.append(e)
    return edges


def _parse_iso(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def ring_stats(graph_json, cycle):
    """Duration (earliest-to-latest edge in the cycle) and average transfer
    amount -- both computable directly from the cycle's own edges,
    independent of which specific edge B2/B3's bifiltration would call the
    'birth' edge."""
    edges = _cycle_edges(graph_json, cycle)
    if not edges:
        return None
    timestamps = [_parse_iso(e["timestamp"]) for e in edges]
    amounts = [e["amount"] for e in edges]
    duration_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
    return {
        "duration_days": round(duration_days, 2),
        "avg_amount": round(sum(amounts) / len(amounts), 2),
        "n_edges": len(edges),
    }


def _format_explanation(cycle, stats, threshold=REPORTING_THRESHOLD):
    path = " -> ".join(cycle) + f" -> {cycle[0]}"
    under_threshold = stats["avg_amount"] < threshold
    threshold_clause = (
        f"consistently just under the ₹{threshold:,.0f} reporting threshold"
        if under_threshold
        else f"averaging ₹{stats['avg_amount']:,.0f} per transfer"
    )
    duration_str = (
        f"{stats['duration_days']:.0f} days" if stats["duration_days"] >= 1
        else f"{stats['duration_days'] * 24:.1f} hours"
    )
    return (
        f"Ring closed via {path} over {duration_str}, "
        f"avg transfer ₹{stats['avg_amount']:,.0f}, {threshold_clause}."
    )


def explain_graph(graph_json, diagram=None, ring_persistence_score=None,
                   threshold=REPORTING_THRESHOLD):
    """Top-level entry point for this stage. Returns the README output
    schema fields (diagram, ring_persistence_score passed through
    unchanged from B0-B5; detected_ring_nodes and explanation newly
    produced here).

    Returns an empty detected_ring_nodes + a plain no-ring explanation for
    fan-out/clean graphs -- matching B0-B3's H1=0 result for those. This is
    a second, independent confirmation of the same "no cycle here"
    conclusion, from plain graph structure rather than persistent
    homology -- worth mentioning as a cross-check if asked.
    """
    cycles = find_ring_cycles(graph_json)

    if not cycles:
        return {
            "diagram": diagram,
            "ring_persistence_score": ring_persistence_score,
            "detected_ring_nodes": [],
            "explanation": "No ring structure detected in this batch.",
        }

    rings = []
    for cycle in cycles:
        stats = ring_stats(graph_json, cycle)
        if stats is None:
            continue
        rings.append({"nodes": cycle, "explanation": _format_explanation(cycle, stats, threshold)})

    # README schema shows singular detected_ring_nodes/explanation fields.
    # Current dataset (B0-B5) never produces more than one ring per graph,
    # so the common case below is exact. If a later stage's graphs carry
    # multiple independent rings, this still degrades sensibly: nodes get
    # unioned and explanations get joined, rather than silently dropping
    # all but the first one.
    all_nodes = [n for r in rings for n in r["nodes"]]
    combined_explanation = " ".join(r["explanation"] for r in rings)

    return {
        "diagram": diagram,
        "ring_persistence_score": ring_persistence_score,
        "detected_ring_nodes": all_nodes,
        "explanation": combined_explanation,
    }


if __name__ == "__main__":
    from b5_fusion import build_synthetic_dataset, graph_to_diagram_once

    dataset = build_synthetic_dataset()
    print("=== B6 EXPLAINABILITY — per graph ===\n")
    for graph_id, kind, graph_json, labels in dataset:
        diagram_result = graph_to_diagram_once(graph_json)
        result = explain_graph(
            graph_json,
            diagram=diagram_result["diagram"],
            ring_persistence_score=diagram_result["ring_persistence_score"],
        )
        ring_members = set(result["detected_ring_nodes"])
        true_ring_members = {acct for acct, lbl in labels.items() if lbl == 1}
        match = "MATCH" if ring_members.issubset(true_ring_members) or kind != "ring" else "MISMATCH"
        print(f"{graph_id} ({kind}):")
        print(f"  detected_ring_nodes = {result['detected_ring_nodes']}")
        print(f"  explanation = {result['explanation']}")
        print(f"  [{match} vs known fraud labels]\n")
