"""
run_real_rounds.py — Stage C2 entry point: the round loop wired for
real, per the master prompt's spec:

    generate (calls A) -> score (calls B) -> optimize (this) -> repeat

Runs 3-5 real rounds using:
  - A's real build_graph, via redteam_adapter.py (import verified against
    the actual main.py/theta.py/schema.py files received this stage)
  - B's real score_batch, via blueteam_adapter.py IF that import succeeds
    — prints a loud warning below and tags results if it silently fell
    back to the non-real stub; see blueteam_adapter.py's docstring
  - This track's own BayesOptimizer (bo.py, C1), reconfigured with the
    real search space (bo_real_space.py) instead of the C1 stub space
  - Bottleneck distance computed directly between consecutive rounds'
    real diagrams (bottleneck.py) rather than via B's update_diagram —
    see bottleneck.py's docstring for why that's a flagged deviation
    from the master prompt, not a silent one

Writes round-by-round results to disk via loop.run_loop, in the exact
Round Result schema the root README specifies.
"""
from __future__ import annotations
import sys

from .loop import run_loop
from .bo import BayesOptimizer
from .bo_real_space import REAL_SEARCH_SPACE, real_vector_to_theta_dict
from .redteam_adapter import build_real_graph
from .blueteam_adapter import score_batch, SCORE_BATCH_IS_REAL
from .bottleneck import compute_bottleneck_distance


def make_real_generate_and_score_fns(seed: int = 42):
    bo = BayesOptimizer(
        seed=seed,
        # Only 2 random-exploration rounds before the GP model kicks in
        # (down from C1's 3) — see PROGRESS_C.md's carried-forward open
        # question: with only 3-5 total real rounds, most/all of a run
        # could otherwise be pure random exploration. Still a guess, not
        # a resolution of that question — flagged again below.
        n_initial_points=2,
        search_space=REAL_SEARCH_SPACE,
        vector_to_theta=real_vector_to_theta_dict,
    )

    def generate_fn(seed=None, round_num=None):
        return bo.ask(seed=seed, round_num=round_num)

    def score_fn(theta_dict, seed=None):
        graph_json = build_real_graph(theta_dict, seed=seed)
        scored = score_batch(graph_json)
        if scored.get("recall") is None:
            # B's real score_batch can return recall=None when this
            # round's graph has no computable ground-truth positives
            # (recall undefined -- 0/0 -- not the same thing as "detector
            # scored 0"). Substituting 0.0 so the loop/GP can continue
            # rather than crashing inside bo.tell() (see bo.py's matching
            # guard). NOT a resolved decision -- 0.0 treats "nothing to
            # detect" the same as "detector missed everything," which
            # pulls the GP in a specific direction and collides with the
            # still-open objective-sign question from C1/C2. Flagged
            # loudly here and in PROGRESS_C.md rather than silently
            # patched, since a wrong guess here biases every later round,
            # not just this one.
            print(
                f"[run_real_rounds] WARNING round theta={theta_dict}: "
                "B's real score_batch returned recall=None (most likely "
                "this round's graph had no computable ground-truth ring "
                "nodes). Substituting recall=0.0 to keep the loop running "
                "-- this round's recall is NOT a real detector score, flag "
                "it before reporting round-over-round numbers.",
                file=sys.stderr,
            )
            scored = dict(scored)  # don't mutate B's returned dict in place
            scored["recall"] = 0.0
        bo.tell(scored["recall"])
        return scored

    return generate_fn, score_fn, bo


if __name__ == "__main__":
    if not SCORE_BATCH_IS_REAL:
        print(
            "WARNING: blueteam.src.b7_score_batch could not be imported "
            "(both package-style and flat-script import attempts failed) "
            "— falling back to blueteam_adapter.py's non-real stub. "
            "Results below are NOT B's real detector output, and "
            "bottleneck distances will show the C0 random-stub fallback "
            "(no 'diagram' key from the fallback scorer). Confirm the "
            "real b7_score_batch.py's location/package layout with B.",
            file=sys.stderr,
        )

    generate_fn, score_fn, bo = make_real_generate_and_score_fns(seed=42)
    results = run_loop(
        n_rounds=5,
        out_dir="output/c2_real_rounds",
        seed=42,
        generate_fn=generate_fn,
        score_fn=score_fn,
        bottleneck_fn=compute_bottleneck_distance,
    )
    for r in results:
        print(
            f"Round {r['round']}: recall={r['recall']}, "
            f"bottleneck={r['bottleneck_distance_from_prev_round']}, "
            f"theta={r['theta_used']}"
        )
