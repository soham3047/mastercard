"""
Stage C0 — round-loop skeleton. (Modified in C2 — see note below.)

Proves the shape of the adversarial loop (generate -> score -> log round
result) using stub functions only. There is NO real Bayesian optimization
yet — `generate_fn` here is just "call the generator again," not a GP
surrogate + Expected Improvement acquisition choosing the next theta.
That's C1.

Output schema per round matches the root README's "Round Result" contract
exactly:
    {"round": int, "theta_used": dict, "recall": float,
     "bottleneck_distance_from_prev_round": float}

C2 CHANGE (backward-compatible): added an optional `bottleneck_fn`
parameter. When None (the default — C0/C1's existing behavior and tests
are unaffected), `bottleneck_distance_from_prev_round` stays the C0
random-placeholder stub. When provided, the loop instead calls
`bottleneck_fn(prev_diagram, scored["diagram"])` each round and tracks
`prev_diagram` across iterations itself — this only activates when the
score_fn's return dict actually contains a "diagram" key (B's real
score_batch does; stub_scorer.fake_score does not), so C0/C1's stub path
is untouched either way.
"""
import json
import os
import random
from pathlib import Path

from .stub_generator import fake_generate_theta
from .stub_scorer import fake_score


def run_loop(n_rounds=5, out_dir="output/stub_rounds", seed=42,
             generate_fn=None, score_fn=None, bottleneck_fn=None):
    """
    Run n_rounds of generate -> score -> log, using stub functions by
    default. generate_fn/score_fn are swappable so C2 can pass in real
    A/B calls without touching this loop's structure.

    generate_fn(seed, round_num) -> theta dict
    score_fn(theta, seed) -> {precision, recall, f1, auc, per_account_fraud_prob, ...}
        (C2: B's real score_batch also includes "diagram",
        "ring_persistence_score", "detected_ring_nodes", "explanation" —
        this loop only reads "recall" and, if bottleneck_fn is set,
        "diagram"; it ignores the rest.)
    bottleneck_fn(prev_diagram, curr_diagram) -> float
        Optional (C2). If None, falls back to C0's random stub. If set,
        called every round after round 1, with prev_diagram=None on
        round 1 handled by the caller's own convention (this loop passes
        whatever bottleneck_fn returns for round 1 through unchanged —
        see bottleneck.py, which returns 0.0 for a None prev_diagram).

    Returns the list of round-result dicts (also written to disk).
    """
    generate_fn = generate_fn or fake_generate_theta
    score_fn = score_fn or fake_score

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    results = []
    prev_diagram = None

    for round_num in range(1, n_rounds + 1):
        theta = generate_fn(seed=seed + round_num, round_num=round_num)
        scored = score_fn(theta, seed=seed + round_num)

        if bottleneck_fn is not None and "diagram" in scored:
            bottleneck = bottleneck_fn(prev_diagram, scored["diagram"])
            prev_diagram = scored["diagram"]
        else:
            # C0 stub path, unchanged: round 1 has no prior round to
            # compare against, so it's fixed at 0.0. Later rounds get a
            # random stand-in until a real bottleneck_fn is wired in.
            bottleneck = 0.0 if round_num == 1 else round(rng.uniform(0.0, 1.0), 4)

        result = {
            "round": round_num,
            "theta_used": theta,
            "recall": scored["recall"],
            "bottleneck_distance_from_prev_round": bottleneck,
        }
        results.append(result)

        with open(os.path.join(out_dir, f"round_{round_num}.json"), "w") as f:
            json.dump(result, f, indent=2)

    with open(os.path.join(out_dir, "all_rounds.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    for r in run_loop():
        print(
            f"Round {r['round']}: recall={r['recall']}, "
            f"bottleneck={r['bottleneck_distance_from_prev_round']}, "
            f"theta={r['theta_used']}"
        )
