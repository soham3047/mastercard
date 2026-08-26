"""
VINEYARD — Blue Team, Stage B2: Bifiltration (time + amount-proximity-to-threshold).

IMPLEMENTATION NOTE — read before presenting this stage to judges or the team:
True multiparameter persistent homology (a genuine 2-parameter persistence
module, via e.g. the `multipers` library) was tried first. `multipers`
installs cleanly but segfaulted on a trivial 3-edge test in this
environment, and correctly wiring module approximation / signed measures is
itself a research-grade task, not a same-day one. Given the deadline, this
stage instead fuses the two axes into ONE scalar filtration per edge:
    combined = alpha * time_norm + beta * amount_suspicion_penalty_norm
This is an engineering approximation, not a full 2-parameter persistence
module decomposition — say so plainly if asked. It still demonstrably does
the actual job: ranking a structuring ring as MORE urgent than pure time
would, and an ordinary-amount ring as LESS urgent, using the same weights.

Both axes are normalized on FIXED absolute scales (not per-graph min-max).
Per-graph min-max was tried first and is a real bug to avoid: it made an
ORDINARY ring score higher than a STRUCTURING ring, because normalizing
penalties within a single graph erases how close to the threshold the
amounts actually are in absolute terms. Verified below with actual numbers,
not assumed.
"""
import gudhi

DEFAULT_THRESHOLD = 10000.0
TIME_SCALE_DAYS = 180.0   # fixed reference window; a ring spanning this long is "maximally slow"
ESSENTIAL_SCORE_BASE = 2.0  # score baseline for a hole that never closes (see build_and_score)


def _time_norm(t, t0):
    return max(0.0, (t - t0)) / TIME_SCALE_DAYS


def _amount_penalty_norm(amount, threshold=DEFAULT_THRESHOLD, band=0.15):
    """0 = right at the threshold (max suspicion, enters complex early).
    1 = outside the suspicious band entirely (enters complex late/never)."""
    if amount is None or amount <= 0:
        return 1.0
    distance = threshold - amount
    if 0 <= distance <= threshold * band:
        return distance / (threshold * band)
    return 1.0


def _build_and_score(node_ids, node_index, edges, values):
    st = gudhi.SimplexTree()
    for n in node_ids:
        st.insert([node_index[n]], filtration=0.0)
    for e, v in zip(edges, values):
        st.insert([node_index[e["from"]], node_index[e["to"]]], filtration=v)
    raw = st.persistence(persistence_dim_max=True, min_persistence=-1)
    h1 = [(b, d) for dim, (b, d) in raw if dim == 1]
    if not h1:
        return {"birth": None, "death": None, "score": 0.0}
    birth, death = h1[0]
    # A hole that never closes (death=inf) has no defined lifetime — score it
    # by how early it formed instead: lower birth = more urgent = higher score.
    score = (ESSENTIAL_SCORE_BASE - birth) if death == float("inf") else (death - birth)
    return {"birth": birth, "death": death, "score": round(score, 4)}


def graph_to_diagram(graph_json, threshold=DEFAULT_THRESHOLD, alpha=0.5, beta=0.5):
    """
    Returns contract-2-shaped output plus a time-only score for comparison:
        {"diagram": [...], "ring_persistence_score": float,
         "time_only_ring_persistence_score": float}
    """
    edges = graph_json["edges"]
    if not edges:
        return {"diagram": [], "ring_persistence_score": 0.0, "time_only_ring_persistence_score": 0.0}

    node_ids = sorted({n["id"] for n in graph_json["nodes"]})
    node_index = {n: i for i, n in enumerate(node_ids)}
    t0 = min(e["timestamp"] for e in edges)

    time_vals = [_time_norm(e["timestamp"], t0) for e in edges]
    penalty_vals = [_amount_penalty_norm(e.get("amount"), threshold) for e in edges]
    combined_vals = [alpha * t + beta * p for t, p in zip(time_vals, penalty_vals)]

    time_result = _build_and_score(node_ids, node_index, edges, time_vals)
    combined_result = _build_and_score(node_ids, node_index, edges, combined_vals)

    diagram = []
    if combined_result["birth"] is not None:
        diagram.append({
            "birth": combined_result["birth"],
            "death": combined_result["death"] if combined_result["death"] != float("inf") else None,
            "dimension": 1,
            "filtration": "time_amount_bifiltration",
        })

    return {
        "diagram": diagram,
        "ring_persistence_score": combined_result["score"],
        "time_only_ring_persistence_score": time_result["score"],
    }


if __name__ == "__main__":
    # Structuring ring: 6-node cycle spread over 120 days (looks slow/low-urgency
    # under time alone), but every transfer sits just under a ₹10,000 threshold.
    structuring_ring = {
        "nodes": [{"id": f"acct_{i:03d}"} for i in range(1, 7)],
        "edges": [
            {"from": "acct_001", "to": "acct_002", "timestamp": 0,   "amount": 9850},
            {"from": "acct_002", "to": "acct_003", "timestamp": 20,  "amount": 9700},
            {"from": "acct_003", "to": "acct_004", "timestamp": 45,  "amount": 9900},
            {"from": "acct_004", "to": "acct_005", "timestamp": 70,  "amount": 9600},
            {"from": "acct_005", "to": "acct_006", "timestamp": 95,  "amount": 9800},
            {"from": "acct_006", "to": "acct_001", "timestamp": 120, "amount": 9750},
        ],
    }
    # Control: identical topology and timing, ordinary (non-suspicious) amounts.
    ordinary_ring = {
        "nodes": [{"id": f"acct_{i:03d}"} for i in range(1, 7)],
        "edges": [
            {"from": "acct_001", "to": "acct_002", "timestamp": 0,   "amount": 500},
            {"from": "acct_002", "to": "acct_003", "timestamp": 20,  "amount": 1200},
            {"from": "acct_003", "to": "acct_004", "timestamp": 45,  "amount": 300},
            {"from": "acct_004", "to": "acct_005", "timestamp": 70,  "amount": 2000},
            {"from": "acct_005", "to": "acct_006", "timestamp": 95,  "amount": 750},
            {"from": "acct_006", "to": "acct_001", "timestamp": 120, "amount": 1100},
        ],
    }

    print("=== B2 BIFILTRATION TEST ===")
    r_struct = graph_to_diagram(structuring_ring)
    r_ord = graph_to_diagram(ordinary_ring)

    print(f"Structuring ring — time-only: {r_struct['time_only_ring_persistence_score']}, "
          f"combined: {r_struct['ring_persistence_score']}")
    print(f"Ordinary ring    — time-only: {r_ord['time_only_ring_persistence_score']}, "
          f"combined: {r_ord['ring_persistence_score']}")

    struct_uplift = r_struct["ring_persistence_score"] - r_struct["time_only_ring_persistence_score"]
    ord_uplift = r_ord["ring_persistence_score"] - r_ord["time_only_ring_persistence_score"]
    print(f"\nStructuring ring uplift: {round(struct_uplift, 4)}")
    print(f"Ordinary ring uplift:    {round(ord_uplift, 4)}")

    if struct_uplift > 0 > ord_uplift:
        print("\nPASS: the amount axis correctly boosts urgency for the structuring ring "
              "and correctly suppresses it for the ordinary ring, despite identical topology and timing.")
    else:
        print("\nFAIL: amount axis isn't distinguishing structuring from ordinary as expected.")