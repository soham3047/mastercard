"""
Stage C0 — stub detector scorer.

Returns a fake detection-result dict shaped like B's real `score_batch`
output (precision/recall/f1/auc/per_account_fraud_prob), so the loop can
be exercised end-to-end before B's real detector is wired in.

IMPORTANT (see PROGRESS_C.md open questions): B's Stage 3/7 synopsis flags
that her real `update_diagram(prev_diagram, new_edges)` returns
`(diagram, state)`, where `state` is a LIVE `PersistentGraph` object that
must be kept alive across rounds — not a bare-diagram-in/bare-diagram-out
call. This stub only fakes `score_batch`; it does NOT fake
`update_diagram` at all, since C0's loop doesn't yet track diagrams
round-to-round (see loop.py's bottleneck-distance stub below). C2 must
account for the stateful shape when it wires this in for real, or the
incremental-update speedup silently breaks if the loop calls it
statelessly.
"""
import random


def fake_score(theta, seed=None):
    """Return a fake score_batch()-shaped result. Swap for B's real score_batch in C2."""
    rng = random.Random(seed if seed is not None else hash(str(theta)) % (2**31))
    recall = round(rng.uniform(0.2, 0.95), 4)
    precision = round(rng.uniform(0.2, 0.95), 4)
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0
    auc = round(rng.uniform(0.5, 0.99), 4)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "per_account_fraud_prob": {
            f"acct_{i:03d}": round(rng.uniform(0, 1), 4) for i in range(5)
        },
    }
