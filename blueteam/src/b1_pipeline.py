"""
VINEYARD — Blue Team, Stage B1: Filtration + persistence pipeline.

Generalizes B0 into a real pipeline: takes any transaction graph matching
the shared schema (root README, contract 1) and returns a persistence
diagram plus a ring_persistence_score (contract 2 — partial; the fields
`detected_ring_nodes` and `explanation` are added in B4/B6).
"""
from datetime import datetime
import gudhi


def _parse_time(ts):
    """ISO8601 string -> float. Accepts a raw float/int too (for test data)."""
    if isinstance(ts, (int, float)):
        return float(ts)
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def graph_to_diagram(graph_json, filtration="time"):
    """
    graph_json: {"nodes": [{"id": ..., ...}], "edges": [{"from": ..., "to": ..., "timestamp": ..., "amount": ..., ...}]}
    filtration: "time" (B1 — only mode implemented; "time_amount_bifiltration" arrives in B2)

    Returns dict matching contract 2 (minus detected_ring_nodes/explanation, added later):
        {"diagram": [...], "ring_persistence_score": float}
    """
    if filtration != "time":
        raise NotImplementedError("Only 'time' filtration is implemented until B2.")

    edges = graph_json["edges"]
    if not edges:
        return {"diagram": [], "ring_persistence_score": 0.0}

    node_ids = sorted({n["id"] for n in graph_json["nodes"]})
    node_index = {n: i for i, n in enumerate(node_ids)}

    times = [_parse_time(e["timestamp"]) for e in edges]
    t_min = min(times)
    # normalize so filtration starts at 0 — keeps values readable/comparable across graphs
    norm_times = [t - t_min for t in times]
    t_max = max(norm_times)

    st = gudhi.SimplexTree()
    for n in node_ids:
        st.insert([node_index[n]], filtration=0.0)
    for e, t in zip(edges, norm_times):
        st.insert([node_index[e["from"]], node_index[e["to"]]], filtration=t)

    raw_diagram = st.persistence(persistence_dim_max=True, min_persistence=-1)

    # Cap infinite deaths at (t_max + 1) so the score/diagram are always finite
    # and JSON-safe. Essential classes are expected here since B1 doesn't
    # expand to a flag complex yet — revisit if B2's bifiltration changes this.
    diagram = []
    h1_lifetimes = []
    for dim, (birth, death) in raw_diagram:
        death_out = (t_max + 1.0) if death == float("inf") else death
        diagram.append({"birth": birth, "death": death_out, "dimension": dim, "filtration": "time"})
        if dim == 1:
            h1_lifetimes.append(death_out - birth)

    ring_persistence_score = max(h1_lifetimes) if h1_lifetimes else 0.0

    return {"diagram": diagram, "ring_persistence_score": round(ring_persistence_score, 4)}


if __name__ == "__main__":
    # Synthetic test: one embedded 6-node ring + decoy noise, NOT hand-tuned
    # like B0 — this is meant to stress the generalized pipeline a bit harder.
    test_graph = {
        "nodes": [{"id": f"acct_{i:03d}"} for i in range(1, 15)],
        "edges": [
            # 6-node ring: 001->002->...->006->001
            {"from": "acct_001", "to": "acct_002", "timestamp": 0.0},
            {"from": "acct_002", "to": "acct_003", "timestamp": 1.0},
            {"from": "acct_003", "to": "acct_004", "timestamp": 2.0},
            {"from": "acct_004", "to": "acct_005", "timestamp": 3.0},
            {"from": "acct_005", "to": "acct_006", "timestamp": 4.0},
            {"from": "acct_006", "to": "acct_001", "timestamp": 5.0},
            # decoy fan-out (star, no cycle)
            {"from": "acct_007", "to": "acct_008", "timestamp": 0.5},
            {"from": "acct_007", "to": "acct_009", "timestamp": 1.2},
            {"from": "acct_007", "to": "acct_010", "timestamp": 2.1},
            # decoy chain
            {"from": "acct_011", "to": "acct_012", "timestamp": 0.8},
            {"from": "acct_012", "to": "acct_013", "timestamp": 1.9},
            {"from": "acct_013", "to": "acct_014", "timestamp": 3.3},
        ],
    }

    result = graph_to_diagram(test_graph)
    print("=== B1 PIPELINE TEST ===")
    print(f"ring_persistence_score: {result['ring_persistence_score']}")
    h1 = [p for p in result["diagram"] if p["dimension"] == 1]
    print(f"H1 features: {len(h1)}")
    for p in h1:
        print(f"  birth={p['birth']}, death={p['death']}")

    if len(h1) == 1 and result["ring_persistence_score"] > 0:
        print("\nPASS: pipeline correctly isolates the single embedded ring "
              "from fan-out and chain decoys on a graph it wasn't hand-built for.")
    else:
        print("\nFAIL: unexpected H1 count or zero score.")