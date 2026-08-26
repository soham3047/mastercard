"""
calibration.py — Stage A4: public aggregate statistics used to calibrate
Red Team distribution parameters.

IMPORTANT: these are PUBLIC AGGREGATE statistics about India's digital
payments ecosystem (UPI volumes/values, regulatory thresholds) — NOT
labeled fraud data, per the team's ground rule for this stage. They're
used to make the *background* traffic statistically realistic and to
ground the *structuring threshold* in a real regulation, so the
"fidelity of simulation" claim in the writeup is defensible rather than
an arbitrary guess.

Sources (accessed August 2026):
  [1] NPCI UPI monthly statistics, FY2025-26 — average UPI ticket size
      approx. Rs 1,300, reflecting heavy use for low-value merchant
      payments. https://coinlaw.io/upi-statistics/
  [2] Worldline India Digital Payments Report, H2 2024 (via Business
      Standard) — P2P UPI average ticket size Rs 2,666; P2M average
      ticket size Rs 627.
      https://www.business-standard.com/amp/finance/news/small-transactions-drive-7-8-drop-in-upi-payments-average-ticket-size-125040201160_1.html
  [3] NPCI UPI statistics roundup — approx. 86% of UPI transactions fall
      in the Rs 0-500 band (heavy right skew: many small transfers, a
      long tail of larger ones pulling the mean well above the median).
      https://meetanshi.com/blog/upi-statistics/
  [4] FIU-IND / PMLA Rule 3 — Cash Transaction Report (CTR) threshold:
      any transaction over Rs 10,00,000 (10 lakh) must be reported to
      the Financial Intelligence Unit - India. This is the real Indian
      analogue of "structuring just under the reporting threshold" (the
      shared interface doc's Rs 10,000 example was illustrative only,
      not this project's actual regulatory number).
      https://fiuindia.gov.in/files/FAQs/faqs.html

What is, and isn't, derived from these sources:
  - decoy_amount_mu / decoy_amount_sigma ARE derived below from [1] + [3]
    via the documented method.
  - REPORTING_THRESHOLD_INR IS the real PMLA CTR figure [4].
  - Hop-timing (Exponential lambda) and Hawkes-process burst parameters
    are NOT derivable from these aggregate stats — NPCI publishes
    volume/value, not inter-transaction timing distributions for
    individual account pairs. They're principled modeling choices,
    documented as such in theta.py, not claimed as calibrated.
"""
from __future__ import annotations

import math

# --- Background (decoy) amount distribution --------------------------------
# Person-to-person transfers are the closest real-world analogue to the
# inter-account "hops" this project simulates (as opposed to P2M retail
# checkout payments). We fit a LogNormal(mu, sigma) to two public data
# points rather than guessing mu/sigma directly:
#
#   1. ~86% of transactions fall under Rs 500 [3]. Treating this as an
#      upper bound on the *median* gives a conservative interior estimate
#      of median ~= Rs 300 (86% is well past the 50th percentile, so the
#      true median is very likely lower than 500; 300 is a reasonable
#      interior guess documented here as an assumption, not a citation).
#   2. Overall mean UPI ticket size ~= Rs 1,300 [1].
#
# For LogNormal: median = exp(mu), mean = exp(mu + sigma^2 / 2).
# Solving for sigma given both anchors:
MEDIAN_ESTIMATE_INR = 300.0   # assumption, bounded by [3]
MEAN_ESTIMATE_INR = 1300.0    # from [1]

DECOY_AMOUNT_MU = math.log(MEDIAN_ESTIMATE_INR)
DECOY_AMOUNT_SIGMA = math.sqrt(2 * (math.log(MEAN_ESTIMATE_INR) - DECOY_AMOUNT_MU))


def implied_lognormal_mean(mu: float, sigma: float) -> float:
    return math.exp(mu + sigma ** 2 / 2)


# --- Structuring / reporting threshold --------------------------------------
# Real PMLA Cash Transaction Report threshold [4].
REPORTING_THRESHOLD_INR = 1_000_000.0  # Rs 10,00,000

# Ring/structuring edges are generated to land just under this threshold,
# at a configurable margin (theta.structuring_margin) — the classic
# "smurfing" pattern: split a large transfer into several transactions
# each individually below the reporting limit.
DEFAULT_STRUCTURING_MARGIN = 0.06  # amounts land within ~6% below threshold


def structuring_lognormal_params(
    threshold: float = REPORTING_THRESHOLD_INR,
    margin: float = DEFAULT_STRUCTURING_MARGIN,
) -> tuple[float, float]:
    """LogNormal params for ring/structuring amounts.

    Median is placed at threshold * (1 - margin/2) so the bulk of the
    distribution sits just under the threshold. Sigma is kept SMALL
    (tight cluster) since real structuring amounts are deliberately
    similar to each other by design, AND because for a LogNormal, sigma
    is a *multiplicative* spread — sigma=0.18 (an early draft value)
    put ~40% of the mass over threshold, which defeats the entire point
    of structuring. sigma=0.02 keeps ~94% of draws under threshold on
    the first sample; sample_structuring_amount() below rejection-samples
    the rest so a structuring edge is (for all practical purposes) never
    generated above the threshold. This sigma is a modeling choice, not
    derived from public data (see module docstring).
    """
    target_median = threshold * (1 - margin / 2)
    mu = math.log(target_median)
    sigma = 0.02
    return mu, sigma


RING_AMOUNT_MU, RING_AMOUNT_SIGMA = structuring_lognormal_params()


def sample_structuring_amount(
    rng,
    mu: float = RING_AMOUNT_MU,
    sigma: float = RING_AMOUNT_SIGMA,
    threshold: float = REPORTING_THRESHOLD_INR,
    max_tries: int = 10,
) -> float:
    """Draw a structuring amount, guaranteed (for practical purposes) to
    land under `threshold` — an amount that trips the reporting limit
    isn't structuring anymore, it's just a large transfer, so this isn't
    an optional nicety.

    Rejection-samples up to max_tries; falls back to a clipped value just
    under threshold in the astronomically unlikely case all tries fail
    (with sigma=0.02 each draw already lands under threshold ~94% of the
    time, so max_tries=10 fails with probability ~6e-13).
    """
    for _ in range(max_tries):
        amt = float(rng.lognormal(mu, sigma))
        if amt < threshold:
            return amt
    return threshold * 0.99


if __name__ == "__main__":
    print(f"Decoy LogNormal(mu={DECOY_AMOUNT_MU:.3f}, sigma={DECOY_AMOUNT_SIGMA:.3f})")
    print(f"  implied mean   ~= Rs {implied_lognormal_mean(DECOY_AMOUNT_MU, DECOY_AMOUNT_SIGMA):.0f}"
          f" (target Rs {MEAN_ESTIMATE_INR:.0f}, from NPCI FY25-26 [1])")
    print(f"  implied median ~= Rs {math.exp(DECOY_AMOUNT_MU):.0f}"
          f" (assumption bounded by 'under Rs 500' [3])")
    print()
    print(f"Ring LogNormal(mu={RING_AMOUNT_MU:.3f}, sigma={RING_AMOUNT_SIGMA:.3f})")
    print(f"  implied median ~= Rs {math.exp(RING_AMOUNT_MU):.0f}"
          f" (threshold Rs {REPORTING_THRESHOLD_INR:.0f}, PMLA CTR [4])")
