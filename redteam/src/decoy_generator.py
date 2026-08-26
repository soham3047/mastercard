"""
decoy_generator.py — Stage A3: background non-fraud transactions + realism layer.

Same distribution families as the ring/fan-out generators (so amount
alone isn't a tell), but with NO ring/star graph structure — just random
account pairs.

Timing uses a self-exciting Hawkes process (Ogata's thinning algorithm)
instead of a flat Poisson process, so decoy activity clusters in time
the same way real transaction traffic does. That matters because it
means Person B can't separate ring/decoy bursts by timing density
alone — the topological signal (B1-B3) has to be doing the real
detection work, not just "is this account unusually busy right now".
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import numpy as np

from .schema import Edge, Node, TransactionGraph


def simulate_hawkes_events(
    duration_hours: float,
    mu: float,
    kappa: float,
    beta: float,
    rng: np.random.Generator,
    max_events: int = 20000,
) -> List[float]:
    """Ogata's thinning algorithm for a 1-D Hawkes process.

    Intensity: lambda(t) = mu + kappa * sum_{t_i < t} beta * exp(-beta * (t - t_i))
      mu:    baseline event rate (events/hour)
      kappa: branching ratio / excitation strength — MUST be < 1, or the
             process is supercritical and can explode into an infinite
             cascade of self-triggered events.
      beta:  decay rate of excitation (higher = shorter-lived bursts)

    Returns event times in hours from 0.
    """
    if kappa >= 1.0:
        raise ValueError("kappa must be < 1 for a stable (non-exploding) Hawkes process")

    events: List[float] = []
    t = 0.0

    def intensity(t_now: float) -> float:
        if not events:
            return mu
        excitation = sum(beta * np.exp(-beta * (t_now - te)) for te in events if te < t_now)
        return mu + kappa * excitation

    while t < duration_hours and len(events) < max_events:
        lam_upper = intensity(t) + kappa * beta  # local upper bound just past t
        if lam_upper <= 0:
            break
        w = rng.exponential(1.0 / lam_upper)
        t_candidate = t + w
        if t_candidate >= duration_hours:
            break
        if rng.random() <= intensity(t_candidate) / lam_upper:
            events.append(t_candidate)
        t = t_candidate

    return events


def generate_decoys(
    num_accounts: int,
    duration_hours: float,
    hawkes_mu: float,
    hawkes_kappa: float,
    hawkes_beta: float,
    amount_mu: float,
    amount_sigma: float,
    start_time: datetime,
    rng: np.random.Generator,
) -> TransactionGraph:
    graph = TransactionGraph()
    accounts = [f"acct_decoy{i:03d}" for i in range(num_accounts)]
    for a in accounts:
        graph.nodes.append(Node(id=a, type="decoy", created_at=start_time.isoformat()))

    event_times = simulate_hawkes_events(duration_hours, hawkes_mu, hawkes_kappa, hawkes_beta, rng)

    accounts_arr = np.array(accounts)
    for et in event_times:
        src, dst = rng.choice(accounts_arr, size=2, replace=False)
        ts = start_time + timedelta(hours=float(et))
        amount = float(rng.lognormal(amount_mu, amount_sigma))
        graph.edges.append(Edge(from_=str(src), to=str(dst), timestamp=ts.isoformat(), amount=amount, label="decoy"))

    return graph
