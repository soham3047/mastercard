"""Stage C1 smoke test: confirms the GP-driven loop still matches the
README's Round Result contract, and that the GP is actually accumulating
observations round over round (not just re-running the stub randomly)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loop import run_loop
from src.bo import make_bo_generate_and_score_fns, BayesOptimizer


def check_ask_shape():
    bo = BayesOptimizer(seed=7)
    theta = bo.ask()
    expected_keys = {"ring_size", "decoy_density", "hop_timing_params", "fan_out_ratio"}
    assert set(theta.keys()) == expected_keys, f"unexpected theta keys {theta.keys()}"
    assert isinstance(theta["hop_timing_params"], dict) and "lambda" in theta["hop_timing_params"]
    print("PASS: BayesOptimizer.ask() output matches stub_generator's theta shape")


def check_gp_accumulates_observations():
    bo = BayesOptimizer(seed=7, n_initial_points=2)
    for i, recall in enumerate([0.5, 0.6, 0.4, 0.7, 0.55], start=1):
        bo.ask()
        bo.tell(recall)
    assert len(bo.opt.Xi) == 5, f"expected 5 observations recorded, got {len(bo.opt.Xi)}"
    assert len(bo.opt.yi) == 5, f"expected 5 objective values recorded, got {len(bo.opt.yi)}"
    print("PASS: BayesOptimizer accumulates (theta, objective) pairs across ask/tell cycles")


def check_loop_shape_with_bo():
    generate_fn, score_fn, _ = make_bo_generate_and_score_fns(seed=3)
    results = run_loop(
        n_rounds=4,
        out_dir="output/test_bo_rounds",
        seed=3,
        generate_fn=generate_fn,
        score_fn=score_fn,
    )
    assert len(results) == 4, f"expected 4 rounds, got {len(results)}"
    expected_keys = {"round", "theta_used", "recall", "bottleneck_distance_from_prev_round"}
    for i, r in enumerate(results, start=1):
        assert r["round"] == i
        assert set(r.keys()) == expected_keys, f"round {i}: unexpected keys {r.keys()}"
        assert 0.0 <= r["recall"] <= 1.0
    print("PASS: GP-driven loop output still matches the README's Round Result contract exactly")


if __name__ == "__main__":
    check_ask_shape()
    check_gp_accumulates_observations()
    check_loop_shape_with_bo()
