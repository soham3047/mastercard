"""
ring_generator.py — Stage A1: directed cyclic ring (layering pattern).

acct_0 -> acct_1 -> ... -> acct_{k-1} -> acct_0, time-respecting (each
hop's timestamp strictly after the previous one), amounts drawn from the
structuring LogNormal (calibration.py) so each hop individually sits
just under the reporting threshold. This is the shape Person B's
persistent-homology filtration (B0-B3) is built to catch as a long-lived
H1 feature.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .calibration import REPORTING_THRESHOLD_INR, sample_structuring_amount
from .schema import Edge, Node, TransactionGraph


def generate_ring(
    ring_id: int,
    k_lambda: float,
    hop_timing_lambda: float,
    amount_mu: float,
    amount_sigma: float,
    start_time: datetime,
    rng: np.random.Generator,
    reporting_threshold: float = REPORTING_THRESHOLD_INR,
) -> TransactionGraph:
    """Generate one directed cyclic ring.

    ring_id: namespaces account ids across multiple rings in one graph.
    k_lambda: Poisson lambda for ring size k (clamped to >= 3 — fewer
        than 3 accounts can't form a meaningful directed cycle).
    hop_timing_lambda: Exponential lambda (events/hour) for inter-hop gaps.
    amount_mu, amount_sigma: LogNormal params for structuring amounts.
    reporting_threshold: amounts are rejection-sampled to stay under this
        — see calibration.sample_structuring_amount.
    """
    k = max(3, int(rng.poisson(k_lambda)))

    graph = TransactionGraph()
    account_ids = [f"acct_ring{ring_id:03d}_{i:02d}" for i in range(k)]
    for aid in account_ids:
        graph.nodes.append(Node(id=aid, type="ring", created_at=start_time.isoformat()))

    t = start_time
    for i in range(k):
        src = account_ids[i]
        dst = account_ids[(i + 1) % k]  # wraps around to close the cycle
        gap_hours = rng.exponential(1.0 / hop_timing_lambda)
        t = t + timedelta(hours=float(gap_hours))
        amount = sample_structuring_amount(rng, amount_mu, amount_sigma, reporting_threshold)
        graph.edges.append(Edge(from_=src, to=dst, timestamp=t.isoformat(), amount=amount, label="ring"))

    return graph


def generate_rings(
    num_rings: int,
    k_lambda: float,
    hop_timing_lambda: float,
    amount_mu: float,
    amount_sigma: float,
    start_time: datetime,
    rng: np.random.Generator,
    reporting_threshold: float = REPORTING_THRESHOLD_INR,
) -> TransactionGraph:
    graph = TransactionGraph()
    for i in range(num_rings):
        # Stagger ring start times so rings don't all begin simultaneously
        # (that itself would be an unrealistic tell).
        ring_start = start_time + timedelta(hours=float(rng.uniform(0, 48)))
        graph.merge(
            generate_ring(i, k_lambda, hop_timing_lambda, amount_mu, amount_sigma, ring_start, rng, reporting_threshold)
        )
    return graph
