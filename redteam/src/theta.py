"""
theta.py — Stage A6: flat config object theta exposed to the
adversarial-loop teammate (Person C).

Every knob the Red Team generators read is collected here as ONE flat
dataclass/dict so C's Bayesian-optimization loop can mutate it between
rounds without knowing anything about how the generators are implemented
internally. This is the final handoff artifact for the Red Team track —
C's synopsis should reference these exact key names.

Field-by-field provenance:
  - ring_amount_mu / ring_amount_sigma / decoy_amount_mu /
    decoy_amount_sigma / reporting_threshold_inr : calibrated against
    public aggregate statistics — see calibration.py for sources and
    derivation.
  - Everything else (topology sizes, timing rates, Hawkes params): not
    derivable from public aggregate stats (timing distributions for
    individual account pairs aren't published) — principled modeling
    choices, tunable by C.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from .calibration import (
    DECOY_AMOUNT_MU,
    DECOY_AMOUNT_SIGMA,
    DEFAULT_STRUCTURING_MARGIN,
    REPORTING_THRESHOLD_INR,
    RING_AMOUNT_MU,
    RING_AMOUNT_SIGMA,
)


@dataclass
class Theta:
    # --- topology (not calibrated — C's primary optimization surface) ---
    ring_size_lambda: float = 6.0        # Poisson lambda for ring size k (min 3 enforced)
    num_rings: int = 3
    fan_out_ratio: float = 0.4           # 0-1, scales hub spoke-count up/down
    hub_out_degree_lambda: float = 8.0   # Poisson lambda for hub out-degree
    num_hubs: int = 2

    # --- timing (not calibrated — see calibration.py docstring) ---
    hop_timing_lambda: float = 4.0       # Exponential lambda for inter-hop gap (events/hour)
    hawkes_baseline_mu: float = 0.5      # background decoy event rate (events/hour)
    hawkes_kappa: float = 0.6            # excitation strength; MUST be < 1 (stability)
    hawkes_beta: float = 1.2             # decay rate of excitation (higher = shorter bursts)

    # --- amounts (calibrated — see calibration.py) ---
    ring_amount_mu: float = RING_AMOUNT_MU
    ring_amount_sigma: float = RING_AMOUNT_SIGMA
    decoy_amount_mu: float = DECOY_AMOUNT_MU
    decoy_amount_sigma: float = DECOY_AMOUNT_SIGMA
    reporting_threshold_inr: float = REPORTING_THRESHOLD_INR
    structuring_margin: float = DEFAULT_STRUCTURING_MARGIN

    # --- noise / realism ---
    decoy_density: float = 0.7           # ratio of decoy : fraud edges
    num_decoy_accounts: int = 40

    # --- reproducibility ---
    seed: int = 42

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Theta":
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def validate(self) -> None:
        if self.hawkes_kappa >= 1.0:
            raise ValueError("hawkes_kappa must be < 1 (Hawkes process would be unstable/explosive)")
        if not (0.0 <= self.fan_out_ratio <= 1.0):
            raise ValueError("fan_out_ratio must be in [0, 1]")
        if not (0.0 <= self.decoy_density):
            raise ValueError("decoy_density must be >= 0")
        if self.num_rings < 0 or self.num_hubs < 0:
            raise ValueError("num_rings / num_hubs must be >= 0")


DEFAULT_THETA = Theta()
