"""
bo_real_space.py — Stage C2 real search space, re-keyed against A's
actual Theta (redteam/src/theta.py) instead of C1's stub-shaped
SEARCH_SPACE in bo.py.

Which fields are tuned (decision, not a silent guess):
Of Theta's 16 fields, this optimizes exactly 4 — the closest real-key
mapping onto the master prompt's rough sketch (`ring_size`,
`decoy_density`, `hop_timing_params`, `fan_out_ratio`) now that A's real
Theta is available:
    ring_size    -> ring_size_lambda   (Poisson lambda, not a fixed size —
                                         closest real analog)
    hop_timing_params.lambda -> hop_timing_lambda (exact analog)
    decoy_density   -> decoy_density   (identical field name, shared as-is)
    fan_out_ratio   -> fan_out_ratio   (identical field name, shared as-is)
All other Theta fields (num_rings, num_hubs, Hawkes params, the
calibrated amount/threshold fields from A's calibration.py, seed, etc.)
are held at DEFAULT_THETA's values — not part of the search space. This
keeps the loop real and runnable now; widening the search space is a
team call, not something to guess wider or narrower silently. See
PROGRESS_C.md open questions.

Bounds: ring_size_lambda and hop_timing_lambda keep C1's original
stub-era numeric ranges (never calibrated either way in A's
calibration.py, so nothing informative is lost re-using them).
decoy_density and fan_out_ratio use Theta.validate()'s own hard bounds
([0,1] for fan_out_ratio; decoy_density only enforces >=0 in
validate(), so an upper bound of 1.0 here is this file's own choice, not
one enforced by Theta itself — flagged for team review since a value
above 1.0 wouldn't actually be rejected by Theta.validate()).
"""
from skopt.space import Real

REAL_SEARCH_SPACE = [
    Real(3.0, 8.0, name="ring_size_lambda"),
    Real(0.0, 1.0, name="decoy_density"),   # NOTE: Theta.validate() only checks >=0;
                                             # the 1.0 upper bound here is this file's
                                             # choice, not Theta's own hard limit.
    Real(0.5, 5.0, name="hop_timing_lambda"),
    Real(0.0, 1.0, name="fan_out_ratio"),   # matches Theta.validate()'s [0,1] check
]


def real_vector_to_theta_dict(x):
    """skopt gives a flat vector in REAL_SEARCH_SPACE's declared order;
    this re-nests it into the flat dict shape redteam_adapter.py expects
    (itself just the 4 tuned Theta field names — Theta has no nesting,
    unlike the old stub's hop_timing_params.lambda)."""
    ring_size_lambda, decoy_density, hop_timing_lambda, fan_out_ratio = x
    return {
        "ring_size_lambda": float(ring_size_lambda),
        "decoy_density": float(decoy_density),
        "hop_timing_lambda": float(hop_timing_lambda),
        "fan_out_ratio": float(fan_out_ratio),
    }
