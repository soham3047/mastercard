"""
commitment.py — Stage C3 mocked cross-bank commitment primitives.

*** EXPLICITLY SIMULATED — NOT REAL ZK/PSI CRYPTOGRAPHY. ***
Read this docstring in full before reusing any of this outside a
hackathon demo context.

What real privacy-preserving set intersection (PSI) needs, and what this
file deliberately does NOT do:

  - Real PSI protocols (e.g. Diffie-Hellman-based PSI, OPRF-based PSI)
    let two parties learn the intersection of their sets WITHOUT either
    party revealing anything about non-matching elements, and WITHOUT
    requiring a shared secret known to both sides in the clear.

  - This file uses a single SHARED, PUBLICLY-KNOWN salt across all
    "banks" so that H(fingerprint || shared_salt) matches when two banks
    hold the same fingerprint. That is the only way naive hashing can
    produce matching commitments across parties at all — but it is a
    real-world privacy break: if the fingerprint space is small or
    guessable (structured account IDs, phone numbers, PAN-like
    identifiers), anyone who knows the shared salt can precompute
    H(candidate || salt) for every candidate and match it against a
    bank's published commitment SET, learning exact membership. This is
    a classic dictionary/rainbow-table attack against salted hashing
    used as if it were a PSI commitment scheme, which it is not.

  - We deliberately do NOT use per-bank-private salts either, because
    that fails a different way: with independent salts,
    H(x || salt_A) != H(x || salt_B) for the same raw x, so naive
    intersection of two banks' commitment sets would ALWAYS be empty,
    no matter how much real overlap exists. See
    test_crossbank_smoke.py::test_independent_salts_never_match, which
    DEMONSTRATES this failure mode directly rather than just asserting
    it here in a comment.

  - Real deployments solve this with structured cryptographic operations
    that are commutative or oblivious (DH-PSI, OPRF-PSI, or a real
    trusted-setup MPC/ZK protocol) specifically so no shared secret ever
    needs to exist in the clear. Building one of those is explicitly out
    of scope for this hackathon track — the master prompt calls for a
    MOCKED layer, and every output record here carries `"simulated":
    true` precisely so nobody downstream mistakes this for production
    privacy tech.

What this file DOES demonstrate: the mechanical shape of a commitment-
based overlap estimate (hash, publish, intersect, ratio) that a real PSI
protocol would eventually replace piece-for-piece — useful for the
cross-bank UI/data-flow demo, not for demoing real privacy guarantees.
"""
from __future__ import annotations
import hashlib
from typing import Iterable, Set


def commit(fingerprint: str, shared_salt: str) -> str:
    """H(fingerprint || shared_salt). shared_salt is PUBLIC/SHARED across
    all banks in this simulation — see module docstring for why that's
    the non-cryptographic simplification, not an accident."""
    return hashlib.sha256(f"{fingerprint}{shared_salt}".encode("utf-8")).hexdigest()


def commit_set(fingerprints: Iterable[str], shared_salt: str) -> Set[str]:
    return {commit(fp, shared_salt) for fp in fingerprints}


def overlap_estimate(commitments_a: Set[str], commitments_b: Set[str], method: str = "jaccard") -> float:
    """
    Estimate raw-set overlap from two commitment sets, without either
    side ever seeing the other's raw fingerprints — only hashed
    commitments computed with the same shared_salt (see module docstring
    for why the shared salt is the simulated/non-real part).

    method="jaccard" (default): |A ∩ B| / |A ∪ B|
    method="containment": |A ∩ B| / min(|A|, |B|) — use instead if the
        two banks' flagged-account counts are expected to differ a lot,
        since Jaccard's union-in-the-denominator otherwise under-reports
        overlap relative to the smaller bank's own coverage.
    """
    if not commitments_a and not commitments_b:
        return 0.0
    intersection = commitments_a & commitments_b
    if method == "containment":
        denom = min(len(commitments_a), len(commitments_b))
    else:
        denom = len(commitments_a | commitments_b)
    if denom == 0:
        return 0.0
    return round(len(intersection) / denom, 4)
