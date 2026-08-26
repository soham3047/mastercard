"""
Stage C0 — stub theta generator.

Returns a fake attack-parameter config dict shaped like the root README's
rough sketch of A's theta (`ring_size`, `decoy_density`, `hop_timing_params`,
`fan_out_ratio`). This exists ONLY to prove the round-loop's structure
before wiring in A's real generator.

IMPORTANT (see PROGRESS_C.md open questions): A's Stage 6 synopsis shows
the REAL interface is `redteam.src.theta.Theta` / `DEFAULT_THETA`, with
different exact key names (`ring_size_lambda`, `num_rings`,
`hub_out_degree_lambda`, `hop_timing_lambda`, `hawkes_baseline_mu`,
`hawkes_kappa`, `hawkes_beta`, `ring_amount_mu/sigma`,
`decoy_amount_mu/sigma`, `reporting_threshold_inr`, `structuring_margin`,
`decoy_density`, `num_decoy_accounts`, `seed`, plus `fan_out_ratio` which
IS shared). This stub's keys are deliberately the README's rough sketch,
not A's real keys — do not assume they line up. C1/C2 should import
`Theta`/`DEFAULT_THETA` directly per A's own note in her synopsis, rather
than hardcoding either this stub's keys or guessing at A's.
"""
import random


def fake_generate_theta(seed=None, round_num=0):
    """Return a fake theta dict. Swap for A's build_graph(theta)/Theta in C2."""
    rng = random.Random(seed if seed is not None else round_num)
    return {
        "ring_size": rng.randint(3, 8),
        "decoy_density": round(rng.uniform(0.1, 0.9), 3),
        "hop_timing_params": {"lambda": round(rng.uniform(0.5, 5.0), 3)},
        "fan_out_ratio": round(rng.uniform(0.0, 1.0), 3),
    }
