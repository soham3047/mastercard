"""
fanout_generator.py — Stage A2: hub-and-spoke (fan-out / mule network).

One hub account fans money out to (or gathers money in from) many spoke
accounts in a compressed time window. Topologically this is a tree, not
a cycle — invisible to persistent-homology H1 detection (Person B's
B1-B3), which is exactly why B4's graph/account feature extractor
(in-degree bursts, fan-out ratio, time-compression) exists to catch it
separately. This generator exists specifically to make sure that
feature-fusion half of Person B's work has something real to catch.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .calibration import REPORTING_THRESHOLD_INR, sample_structuring_amount
from .schema import Edge, Node, TransactionGraph


def generate_fanout(
    hub_id: int,
    out_degree_lambda: float,
    fan_out_ratio: float,
    hop_timing_lambda: float,
    amount_mu: float,
    amount_sigma: float,
    start_time: datetime,
    rng: np.random.Generator,
    reporting_threshold: float = REPORTING_THRESHOLD_INR,
) -> TransactionGraph:
    """Generate one hub-and-spoke fan-out network.

    out_degree_lambda: Poisson lambda for the hub's spoke count.
    fan_out_ratio: 0-1, scales spoke count further so C can dial "how
        aggressive" the fan-out is without touching out_degree_lambda.
    """
    num_spokes = max(2, int(rng.poisson(out_degree_lambda) * (0.5 + fan_out_ratio)))

    graph = TransactionGraph()
    hub = f"acct_hub{hub_id:03d}"
    graph.nodes.append(Node(id=hub, type="hub", created_at=start_time.isoformat()))

    spokes = [f"acct_hub{hub_id:03d}_spoke{i:02d}" for i in range(num_spokes)]
    for s in spokes:
        # Spokes are fraud participants (type "ring") even though the
        # *shape* isn't a cycle — schema's NodeType only distinguishes
        # ring/decoy/hub, and "ring" here means "fraud-participant node".
        graph.nodes.append(Node(id=s, type="ring", created_at=start_time.isoformat()))

    t = start_time
    for s in spokes:
        # Real mule fan-outs happen in a short compressed burst — this is
        # the "time-compression" signal B4's feature extractor targets —
        # so spoke hops fire ~3x tighter than ring hops, not spread evenly.
        gap_hours = rng.exponential(1.0 / (hop_timing_lambda * 3))
        t = t + timedelta(hours=float(gap_hours))
        amount = sample_structuring_amount(rng, amount_mu, amount_sigma, reporting_threshold)
        # Direction: ~50/50 fan-out (hub -> spokes, distributing funds) vs
        # fan-in (spokes -> hub, collecting funds) — both are real mule
        # patterns and neither should be structurally favored.
        if rng.random() < 0.5:
            graph.edges.append(Edge(from_=hub, to=s, timestamp=t.isoformat(), amount=amount, label="ring"))
        else:
            graph.edges.append(Edge(from_=s, to=hub, timestamp=t.isoformat(), amount=amount, label="ring"))

    return graph


def generate_fanouts(
    num_hubs: int,
    out_degree_lambda: float,
    fan_out_ratio: float,
    hop_timing_lambda: float,
    amount_mu: float,
    amount_sigma: float,
    start_time: datetime,
    rng: np.random.Generator,
    reporting_threshold: float = REPORTING_THRESHOLD_INR,
) -> TransactionGraph:
    graph = TransactionGraph()
    for i in range(num_hubs):
        hub_start = start_time + timedelta(hours=float(rng.uniform(0, 48)))
        graph.merge(
            generate_fanout(
                i, out_degree_lambda, fan_out_ratio, hop_timing_lambda,
                amount_mu, amount_sigma, hub_start, rng, reporting_threshold,
            )
        )
    return graph
