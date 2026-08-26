"""
Stage C3 smoke test.

Per project ground rules, this file is written but not run here — see
"Commands to run" in PROGRESS_C.md's C3 synopsis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crossbank.commitment import commit_set, overlap_estimate
from src.crossbank.simulate import simulate_cross_bank_verification


def test_schema_exact_match():
    result = simulate_cross_bank_verification(bank_names=["Bank1", "Bank2"], seed=1)
    assert "bank_pairs" in result
    pair = result["bank_pairs"][0]
    for key in ("bank_a", "bank_b", "overlap_estimate", "simulated"):
        assert key in pair, f"missing key {key}"
    assert pair["simulated"] is True
    print(f"PASS test_schema_exact_match: {pair}")


def test_three_banks_gives_three_pairs():
    result = simulate_cross_bank_verification(bank_names=["Bank1", "Bank2", "Bank3"], seed=2)
    assert len(result["bank_pairs"]) == 3, "3 banks should produce C(3,2)=3 pairs"
    print(f"PASS test_three_banks_gives_three_pairs: {len(result['bank_pairs'])} pairs")


def test_known_overlap_roughly_recovered():
    result = simulate_cross_bank_verification(
        bank_names=["Bank1", "Bank2"],
        n_flagged_per_bank=50,
        known_overlap_fraction=0.4,
        seed=7,
    )
    est = result["bank_pairs"][0]["overlap_estimate"]
    # Jaccard, not containment, so exact recovery of known_overlap_fraction
    # isn't expected -- just a sanity range check that it's in the right
    # ballpark and not 0 or 1.
    assert 0.15 < est < 0.7, f"overlap_estimate {est} outside sane range for a 40%-by-construction overlap"
    print(f"PASS test_known_overlap_roughly_recovered: overlap_estimate={est}")


def test_independent_salts_never_match():
    """
    Demonstrates, rather than just asserts in a comment, the failure mode
    documented in commitment.py: per-bank-private salts make naive
    commitment intersection ALWAYS empty, even for identical raw
    fingerprints. This is why simulate.py uses one shared salt instead —
    and why that sharing is exactly the non-cryptographic simplification
    this whole layer is flagged "simulated" for.
    """
    same_fingerprints = ["acct_A", "acct_B", "acct_C"]
    commitments_bank1 = commit_set(same_fingerprints, shared_salt="bank1-private-salt")
    commitments_bank2 = commit_set(same_fingerprints, shared_salt="bank2-private-salt")
    est = overlap_estimate(commitments_bank1, commitments_bank2)
    assert est == 0.0, "expected zero overlap with independent per-bank salts, even for identical raw data"
    print(f"PASS test_independent_salts_never_match: overlap_estimate={est} (proves the mechanism, not just asserted)")


def test_shared_salt_matches_identical_fingerprints():
    """Sanity counterpart to the above: same fingerprints + same shared
    salt DOES match, confirming the mechanism works when the (simulated,
    non-real) precondition is met."""
    same_fingerprints = ["acct_A", "acct_B", "acct_C"]
    commitments_bank1 = commit_set(same_fingerprints, shared_salt="shared")
    commitments_bank2 = commit_set(same_fingerprints, shared_salt="shared")
    est = overlap_estimate(commitments_bank1, commitments_bank2)
    assert est == 1.0, "identical fingerprints + shared salt should give overlap_estimate 1.0"
    print(f"PASS test_shared_salt_matches_identical_fingerprints: overlap_estimate={est}")


if __name__ == "__main__":
    test_schema_exact_match()
    test_three_banks_gives_three_pairs()
    test_known_overlap_roughly_recovered()
    test_independent_salts_never_match()
    test_shared_salt_matches_identical_fingerprints()
    print("ALL C3 SMOKE TESTS PASSED")
