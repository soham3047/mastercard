"""
main.py — Red Team orchestrator.

Builds the full synthetic transaction graph (rings + fan-outs + decoy
noise) from a single theta config, generates identities, and writes the
interface-contract JSON. This is the function the adversarial loop
(Person C) calls programmatically each round with a mutated theta; it's
also runnable standalone from the command line for local testing / demo.

Usage:
    python -m src.main --out output/round0
    python -m src.main --out output/round0 --seed 7
    python -m src.main --out output/round0 --theta my_theta.json
    python -m src.main --out output/smoketest --skip-llm   # fast, no LLM calls
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from .decoy_generator import generate_decoys
from .fanout_generator import generate_fanouts
from .identity_llm import generate_identities
from .ring_generator import generate_rings
from .schema import TransactionGraph
from .theta import DEFAULT_THETA, Theta
from .writer import write_output

DECOY_WINDOW_HOURS = 24 * 7  # one week of background traffic per run


def build_graph(theta: Theta) -> TransactionGraph:
    theta.validate()
    rng = np.random.default_rng(theta.seed)
    start_time = datetime.now(timezone.utc)

    graph = TransactionGraph()

    graph.merge(generate_rings(
        num_rings=theta.num_rings,
        k_lambda=theta.ring_size_lambda,
        hop_timing_lambda=theta.hop_timing_lambda,
        amount_mu=theta.ring_amount_mu,
        amount_sigma=theta.ring_amount_sigma,
        start_time=start_time,
        rng=rng,
        reporting_threshold=theta.reporting_threshold_inr,
    ))

    graph.merge(generate_fanouts(
        num_hubs=theta.num_hubs,
        out_degree_lambda=theta.hub_out_degree_lambda,
        fan_out_ratio=theta.fan_out_ratio,
        hop_timing_lambda=theta.hop_timing_lambda,
        amount_mu=theta.ring_amount_mu,
        amount_sigma=theta.ring_amount_sigma,
        start_time=start_time,
        rng=rng,
        reporting_threshold=theta.reporting_threshold_inr,
    ))

    graph.merge(generate_decoys(
        num_accounts=theta.num_decoy_accounts,
        duration_hours=DECOY_WINDOW_HOURS,
        hawkes_mu=theta.hawkes_baseline_mu,
        hawkes_kappa=theta.hawkes_kappa,
        hawkes_beta=theta.hawkes_beta,
        amount_mu=theta.decoy_amount_mu,
        amount_sigma=theta.decoy_amount_sigma,
        start_time=start_time,
        rng=rng,
    ))

    # decoy_density knob: trim decoy edges down to a target ratio of
    # decoy:fraud edges, so C can dial noise up/down per round without
    # regenerating the Hawkes process from scratch. (Only trims, never
    # pads — raise hawkes_baseline_mu/kappa in theta to generate more
    # decoy events in the first place if you need a higher ceiling.)
    fraud_edge_count = sum(1 for e in graph.edges if e.label == "ring")
    target_decoy_count = int(fraud_edge_count * theta.decoy_density)
    decoy_edges = [e for e in graph.edges if e.label == "decoy"]
    if len(decoy_edges) > target_decoy_count:
        rng.shuffle(decoy_edges)
        keep_ids = {id(e) for e in decoy_edges[:target_decoy_count]}
        graph.edges = [e for e in graph.edges if e.label != "decoy" or id(e) in keep_ids]

    graph.validate()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="VINEYARD Red Team -- generate a synthetic transaction graph")
    parser.add_argument("--out", default="output/run0", help="output directory")
    parser.add_argument("--theta", default=None, help="path to a JSON file overriding theta defaults")
    parser.add_argument("--seed", type=int, default=None, help="override theta.seed")
    parser.add_argument("--skip-llm", action="store_true", help="skip identity generation entirely (fast smoke test)")
    args = parser.parse_args()

    theta = DEFAULT_THETA
    if args.theta:
        with open(args.theta) as f:
            overrides = json.load(f)
        theta = Theta.from_dict({**theta.as_dict(), **overrides})
    if args.seed is not None:
        theta.seed = args.seed

    graph = build_graph(theta)

    if args.skip_llm:
        identities = {}
    else:
        accounts = [{"account_id": n.id, "node_type": n.type} for n in graph.nodes]
        identities = generate_identities(accounts, seed=theta.seed)

    write_output(graph, identities, args.out)

    with open(f"{args.out}/theta_used.json", "w") as f:
        json.dump(theta.as_dict(), f, indent=2)
    print(f"Wrote theta -> {args.out}/theta_used.json")


if __name__ == "__main__":
    main()
