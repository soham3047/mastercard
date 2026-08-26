"""
Stage C4 smoke test: confirms the ACTUAL on-disk output files (not the
code that produced them) match the root README's documented schemas
exactly. This is what D reads directly, so it's checked as data on disk,
not by re-importing C's Python and trusting the objects in memory.
"""
import json
import os

ROUND_RESULT_KEYS = {"round", "theta_used", "recall", "bottleneck_distance_from_prev_round"}
CROSS_BANK_PAIR_KEYS = {"bank_a", "bank_b", "overlap_estimate", "simulated"}


def test_round_result_files_match_contract():
    out_dir = "output/bo_rounds"
    assert os.path.isdir(out_dir), f"{out_dir} missing — run `python -m src.bo` first"
    all_rounds = json.load(open(os.path.join(out_dir, "all_rounds.json")))
    assert len(all_rounds) > 0
    for r in all_rounds:
        assert set(r.keys()) == ROUND_RESULT_KEYS, f"round {r.get('round')}: keys {r.keys()} != README contract"
        assert isinstance(r["theta_used"], dict)
        assert isinstance(r["recall"], float)
        assert isinstance(r["bottleneck_distance_from_prev_round"], float)
    print(f"PASS test_round_result_files_match_contract: {len(all_rounds)} rounds on disk, all match README's Round Result schema")


def test_cross_bank_file_matches_contract():
    path = "output/cross_bank/bank_pairs.json"
    assert os.path.isfile(path), f"{path} missing — run `python -m src.run_cross_bank` first"
    result = json.load(open(path))
    assert set(result.keys()) == {"bank_pairs"}, f"top-level keys {result.keys()} != README contract"
    assert len(result["bank_pairs"]) > 0
    for pair in result["bank_pairs"]:
        assert set(pair.keys()) == CROSS_BANK_PAIR_KEYS, f"pair {pair} keys != README contract"
        assert pair["simulated"] is True
    print(f"PASS test_cross_bank_file_matches_contract: {len(result['bank_pairs'])} pairs on disk, all match README's Cross-Bank Verification schema")


if __name__ == "__main__":
    test_round_result_files_match_contract()
    test_cross_bank_file_matches_contract()
    print("ALL C4 CONTRACT TESTS PASSED")
