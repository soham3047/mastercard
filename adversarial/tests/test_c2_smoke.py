"""
Stage C2 smoke test.

Checks:
  1. redteam_adapter.build_real_graph produces a valid graph JSON dict
     (nodes/edges keys present) from a tuned theta dict.
  2. blueteam_adapter.score_batch returns a dict with a "recall" key
     regardless of whether the real import succeeded or the fallback
     fired (SCORE_BATCH_IS_REAL tells you which).
  3. bottleneck.compute_bottleneck_distance returns 0.0 when
     prev_diagram is None, and a float (or the -1.0 sentinel if persim
     isn't installed) otherwise.
  4. The full run_real_rounds pipeline produces n_rounds results with
     round-result schema matching the root README exactly.

This is a SMOKE test, not a correctness test of B's actual detection
quality or A's actual generator realism — those are B's and A's tracks.
Per project ground rules, this file is written but not run — see the
"Commands to run" in PROGRESS_C.md's C2 synopsis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.redteam_adapter import build_real_graph, TUNED_KEYS
from src.blueteam_adapter import score_batch, SCORE_BATCH_IS_REAL
from src.bottleneck import compute_bottleneck_distance
from src.run_real_rounds import make_real_generate_and_score_fns
from src.loop import run_loop


def check_build_real_graph():
    theta_dict = {
        "ring_size_lambda": 6.0,
        "decoy_density": 0.7,
        "hop_timing_lambda": 4.0,
        "fan_out_ratio": 0.4,
    }
    graph_json = build_real_graph(theta_dict, seed=1)
    assert "nodes" in graph_json and "edges" in graph_json, "graph JSON missing nodes/edges"
    assert len(graph_json["nodes"]) > 0, "expected at least one node"
    print(f"PASS check_build_real_graph: {len(graph_json['nodes'])} nodes, "
          f"{len(graph_json['edges'])} edges")


def check_score_batch(graph_json):
    scored = score_batch(graph_json)
    assert "recall" in scored, "score_batch result missing 'recall'"
    print(f"PASS check_score_batch: recall={scored['recall']} "
          f"(SCORE_BATCH_IS_REAL={SCORE_BATCH_IS_REAL})")
    return scored


def check_bottleneck(scored):
    d0 = compute_bottleneck_distance(None, scored.get("diagram", []))
    assert d0 == 0.0, "bottleneck distance vs. None prev_diagram should be 0.0"
    d1 = compute_bottleneck_distance(scored.get("diagram", []), scored.get("diagram", []))
    assert isinstance(d1, float), "bottleneck distance should be a float (or -1.0 sentinel)"
    print(f"PASS check_bottleneck: round1={d0}, self-distance={d1}")


def check_full_loop():
    generate_fn, score_fn, bo = make_real_generate_and_score_fns(seed=1)
    results = run_loop(
        n_rounds=3,
        out_dir="output/c2_smoke_test",
        seed=1,
        generate_fn=generate_fn,
        score_fn=score_fn,
        bottleneck_fn=compute_bottleneck_distance,
    )
    assert len(results) == 3, "expected 3 round results"
    for r in results:
        for key in ("round", "theta_used", "recall", "bottleneck_distance_from_prev_round"):
            assert key in r, f"round result missing '{key}'"
    assert results[0]["bottleneck_distance_from_prev_round"] == 0.0, \
        "round 1 bottleneck should be 0.0"
    print(f"PASS check_full_loop: {len(results)} rounds, "
          f"schema matches root README's Round Result contract")


if __name__ == "__main__":
    check_build_real_graph()
    theta_dict = {
        "ring_size_lambda": 6.0,
        "decoy_density": 0.7,
        "hop_timing_lambda": 4.0,
        "fan_out_ratio": 0.4,
    }
    graph_json = build_real_graph(theta_dict, seed=1)
    scored = check_score_batch(graph_json)
    check_bottleneck(scored)
    check_full_loop()
    print("ALL C2 SMOKE TESTS PASSED")
    sys.exit(0)
