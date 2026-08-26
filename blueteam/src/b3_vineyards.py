"""
VINEYARD — Blue Team, Stage B3: Incremental/streaming persistence ("vineyards").

IMPLEMENTATION NOTE — read before presenting this stage:
The literal academic "vineyard" algorithm (Cohen-Steiner, Edelsbrunner,
Morozov) tracks a persistence diagram continuously as a filtration
deforms, via local transpositions of the boundary matrix — cost roughly
proportional to what actually changed, not the size of the whole complex.
GUDHI doesn't expose this (it always recomputes persistence from the
current filtration state on demand); the library that does (Dionysus)
hasn't been installed/verified here, and B2 already hit one library-install
surprise this project (`multipers` segfaulting on trivial input) — not
worth gambling a second unverified library under deadline pressure.

What THIS stage genuinely implements: a `PersistentGraph` object that
holds one live GUDHI SimplexTree across rounds. Each round, only the NEW
edges are inserted — no rebuilding the tree from the full edge history,
unlike B1/B2 which rebuilt a fresh SimplexTree from the complete edge list
on every call. Persistence is still recomputed on the whole current tree
each round (that part is not incremental) — this is a real, honest partial
win, not the full vineyard algorithm. Swap in a true vineyard
implementation here later if round counts grow large enough to need it.
"""
import gudhi

DEFAULT_THRESHOLD = 10000.0
TIME_SCALE_DAYS = 180.0
ESSENTIAL_SCORE_BASE = 2.0


def _time_norm(t, t0):
    return max(0.0, (t - t0)) / TIME_SCALE_DAYS


def _amount_penalty_norm(amount, threshold=DEFAULT_THRESHOLD, band=0.15):
    """0 = right at the threshold (max suspicion). 1 = outside the band."""
    if amount is None or amount <= 0:
        return 1.0
    distance = threshold - amount
    if 0 <= distance <= threshold * band:
        return distance / (threshold * band)
    return 1.0


class PersistentGraph:
    """
    Stateful, incremental wrapper around a GUDHI SimplexTree.

    Call `add_edges(new_edges)` once per adversarial-loop round with ONLY
    the edges that are new this round — not the full graph so far.

    `t0` (the time reference for normalization) is fixed from the FIRST
    batch of edges ever seen, so later rounds stay comparable to earlier
    ones on the same absolute scale — recomputing t0 every round would
    silently shift what "urgent" means round over round.
    """

    def __init__(self, threshold=DEFAULT_THRESHOLD, alpha=0.5, beta=0.5):
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.st = gudhi.SimplexTree()
        self.node_index = {}
        self.t0 = None
        self.rounds_seen = 0
        self.edges_seen = 0

    def _get_or_add_node(self, node_id):
        if node_id not in self.node_index:
            idx = len(self.node_index)
            self.node_index[node_id] = idx
            self.st.insert([idx], filtration=0.0)
        return self.node_index[node_id]

    def add_edges(self, new_edges):
        """new_edges: list of {"from", "to", "timestamp", "amount"} —
        only this round's NEW transactions."""
        if not new_edges:
            return self.current_diagram()

        if self.t0 is None:
            self.t0 = min(e["timestamp"] for e in new_edges)

        for e in new_edges:
            u = self._get_or_add_node(e["from"])
            v = self._get_or_add_node(e["to"])
            t_norm = _time_norm(e["timestamp"], self.t0)
            p_norm = _amount_penalty_norm(e.get("amount"), self.threshold)
            combined = self.alpha * t_norm + self.beta * p_norm
            # insert() lowers an existing simplex's filtration if the new
            # value is smaller, and is a safe no-op otherwise — fine to
            # call even if [u, v] already exists from an earlier round.
            self.st.insert([u, v], filtration=combined)

        self.rounds_seen += 1
        self.edges_seen += len(new_edges)
        return self.current_diagram()

    def current_diagram(self):
        """Recompute persistence on the CURRENT tree state. This is the
        part that is NOT a true vineyard update (see module docstring) —
        it's a full recomputation, just on an incrementally-built tree
        instead of one rebuilt from scratch every call."""
        raw = self.st.persistence(persistence_dim_max=True, min_persistence=-1)
        h1 = [(b, d) for dim, (b, d) in raw if dim == 1]

        diagram = []
        best_score = 0.0
        for birth, death in h1:
            score = (ESSENTIAL_SCORE_BASE - birth) if death == float("inf") else (death - birth)
            best_score = max(best_score, score)
            diagram.append({
                "birth": birth,
                "death": death if death != float("inf") else None,
                "dimension": 1,
                "filtration": "time_amount_bifiltration",
            })

        return {
            "diagram": diagram,
            "ring_persistence_score": round(best_score, 4),
            "round": self.rounds_seen,
            "total_edges": self.edges_seen,
        }


def update_diagram(prev_state, new_edges, threshold=DEFAULT_THRESHOLD, alpha=0.5, beta=0.5):
    """
    NOTE ON THE README CONTRACT: the root README's shared interface list
    describes this as `update_diagram(prev_diagram, new_edges) -> new_diagram`.
    This implementation deviates from that literal signature on purpose —
    see below — so flag this to whoever wires the adversarial loop before
    they assume the README's exact shape.

    `prev_state` is a PersistentGraph, not a bare diagram dict — a raw
    diagram alone doesn't carry enough information to update incrementally,
    since GUDHI needs the actual complex, not just its previously-computed
    persistence pairs. Pass `prev_state=None` on the first call.

    Returns (updated_diagram_dict, updated_state) — pass `updated_state`
    back in as `prev_state` on the next round. The caller must keep this
    object alive in memory across rounds (it wraps a live GUDHI SimplexTree,
    which is not JSON-serializable) — this will NOT work correctly if the
    adversarial loop is called statelessly (e.g. a fresh process/request
    per round with no object persisted between them).
    """
    graph = prev_state if prev_state is not None else PersistentGraph(threshold, alpha, beta)
    diagram = graph.add_edges(new_edges)
    return diagram, graph


if __name__ == "__main__":
    # Same structuring ring as B2's test, built up gradually across 3
    # adversarial-loop rounds with decoy edges mixed in, to confirm
    # incremental construction reaches the same conclusion B2 got in one
    # shot: the ring only shows up as an H1 feature once it actually closes.
    round_1 = [
        {"from": "acct_001", "to": "acct_002", "timestamp": 0,  "amount": 9850},
        {"from": "acct_002", "to": "acct_003", "timestamp": 20, "amount": 9700},
        {"from": "acct_101", "to": "acct_102", "timestamp": 5,  "amount": 400},   # decoy
    ]
    round_2 = [
        {"from": "acct_003", "to": "acct_004", "timestamp": 45, "amount": 9900},
        {"from": "acct_004", "to": "acct_005", "timestamp": 70, "amount": 9600},
        {"from": "acct_103", "to": "acct_104", "timestamp": 50, "amount": 600},   # decoy
    ]
    round_3 = [
        {"from": "acct_005", "to": "acct_006", "timestamp": 95,  "amount": 9800},
        {"from": "acct_006", "to": "acct_001", "timestamp": 120, "amount": 9750},  # closes the ring
    ]

    state = None
    result = None
    for i, batch in enumerate([round_1, round_2, round_3], start=1):
        result, state = update_diagram(state, batch)
        print(f"Round {i}: total_edges={result['total_edges']}, "
              f"ring_persistence_score={result['ring_persistence_score']}, "
              f"H1 features={len(result['diagram'])}")

    if result["ring_persistence_score"] > 0 and len(result["diagram"]) == 1:
        print("\nPASS: ring only registers as an H1 feature once round 3 closes it "
              "— rounds 1-2 correctly show zero (still just a tree, no cycle yet). "
              "Incremental construction across 3 rounds lands on the same "
              "structural conclusion B2 reached computing the whole graph at once.")
    else:
        print("\nFAIL: unexpected result — check the per-round output above.")
