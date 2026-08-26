"""Stage C0 smoke test: confirms the stub loop produces correctly-shaped output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loop import run_loop


def check_loop_shape():
    results = run_loop(n_rounds=3, out_dir="output/test_stub_rounds", seed=1)

    assert len(results) == 3, f"expected 3 rounds, got {len(results)}"

    expected_keys = {"round", "theta_used", "recall", "bottleneck_distance_from_prev_round"}
    for i, r in enumerate(results, start=1):
        assert r["round"] == i, f"round {i}: wrong round number {r['round']}"
        assert set(r.keys()) == expected_keys, f"round {i}: unexpected keys {r.keys()}"
        assert 0.0 <= r["recall"] <= 1.0, f"round {i}: recall out of range"
        assert isinstance(r["theta_used"], dict), f"round {i}: theta_used not a dict"

    assert results[0]["bottleneck_distance_from_prev_round"] == 0.0, \
        "round 1 bottleneck distance should be fixed at 0.0 (no prior round)"

    print("PASS: stub loop produces 3 correctly-shaped rounds matching the README contract")


if __name__ == "__main__":
    check_loop_shape()
