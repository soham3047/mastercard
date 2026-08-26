# adversarial/ — Adversarial Loop + Cross-Bank Layer (Person C)

Bayesian-optimization loop that tunes Red Team's attack parameters each
round against whatever Blue Team's detector missed, plus a mocked
cross-bank layer simulating fraud-ring overlap verification across
institutions without sharing raw data. Part of **VINEYARD**, built for the
Mastercard Innovation Challenge @ GFF 2026 — see the root `README.md` for
the full project concept and the shared interface contracts every track
builds against.

## Status

**Stage C1 done** (Bayesian optimization core). See `PROGRESS_C.md` for
the full synopsis, decisions, and open questions. Stages C2–C4 not
started yet.

| Stage | Name | Status |
|---|---|---|
| C0 | Scaffold + stub loop | ✅ done |
| C1 | Bayesian optimization core (GP + Expected Improvement) | ✅ done |
| C2 | Round loop wired for real (calls A + B) | not started |
| C3 | Mocked cross-bank commitment layer | not started |
| C4 | Finalize output interfaces | not started |

## What's here right now (C0 + C1)

- `src/stub_generator.py` — fake theta generator standing in for A's real
  `build_graph(theta)` / `Theta`.
- `src/stub_scorer.py` — fake detector scorer standing in for B's real
  `score_batch(graph_json)`.
- `src/loop.py` — the round-loop skeleton itself: `run_loop()` calls
  generate → score → log, N times, with `generate_fn`/`score_fn` as
  swappable parameters so later stages can drop in real A/B calls without
  touching the loop's control flow.
- `src/bo.py` — **(C1, new)** the GP surrogate + Expected Improvement
  acquisition core. `BayesOptimizer` proposes each round's theta instead
  of drawing one randomly; `make_bo_generate_and_score_fns()` wires it
  into a `(generate_fn, score_fn)` pair that plugs straight into
  `loop.run_loop()` unchanged.
- `tests/test_smoke.py` — confirms C0's stub loop output matches the root
  README's Round Result JSON shape exactly.
- `tests/test_bo_smoke.py` — **(C1, new)** confirms `BayesOptimizer`'s
  output shape, that it actually accumulates observations across
  ask/tell cycles, and that the GP-driven loop still matches the Round
  Result contract.

**Not real yet:** the scorer is still `stub_scorer.fake_score` — it
ignores theta's actual values, so the GP in C1 is fitting a surrogate to
random noise. The optimizer's mechanics (ask/tell, search space,
acquisition function) are real and tested; the optimization itself only
becomes meaningful once C2 swaps in B's real, theta-dependent
`score_batch`. Bottleneck distances are still stubbed/random too — real
starting C2.

## Known integration gaps (read before starting C2)

These come from reading A's and B's progress files, and from building
C1, not guesswork:

- **A's real theta keys differ from this stub's.** A's Stage 6 synopsis
  defines `Theta` with keys like `ring_size_lambda`, `num_rings`,
  `hub_out_degree_lambda`, `hop_timing_lambda`,
  `hawkes_baseline_mu/kappa/beta`, `ring_amount_mu/sigma`,
  `decoy_amount_mu/sigma`, `reporting_threshold_inr`,
  `structuring_margin`, `decoy_density`, `num_decoy_accounts`, `seed`.
  Only `fan_out_ratio` and `decoy_density` are actually shared with this
  stub's rough-sketch keys. Import `Theta`/`DEFAULT_THETA` directly from
  `redteam.src.theta` in C2 — don't hardcode either set of names. This
  also means `bo.py`'s `SEARCH_SPACE` needs re-keying against A's real
  bounds, not just the stub's.
- **B's `update_diagram` is stateful.** Her Stage 3 synopsis flags that
  `update_diagram(prev_diagram, new_edges)` actually returns
  `(diagram, state)`, where `state` is a live `PersistentGraph` object
  that must be kept alive across rounds for the incremental-update
  speedup to work. A stateless per-request call silently breaks it. The
  loop doesn't track diagram state at all yet — C2 needs to own that
  state object across rounds or get a stateless wrapper from B.
- **Objective sign is unresolved (new, from C1).** `bo.py` implements
  `J(theta) = 1 - recall` exactly as the master prompt specifies, but
  minimizing that (skopt's default) pushes the GP towards *higher*
  recall — which reads as backwards from "tune attack params against
  what the detector missed" (an evading attacker wants recall to go
  *down*). Needs a team decision before C2 wires this against a real,
  theta-dependent scorer — see `PROGRESS_C.md` Stage C1 open questions.

## Interfaces

**Consumes:**
- From A: `build_graph(theta) -> TransactionGraph` (stubbed via
  `stub_generator.fake_generate_theta`; C1's `BayesOptimizer.ask()` also
  produces theta in this same stub shape)
- From B: `score_batch(graph_json) -> {precision, recall, f1, auc,
  per_account_fraud_prob}` and `update_diagram(prev_diagram, new_edges) ->
  (new_diagram, state)` (stubbed via `stub_scorer.fake_score`;
  `update_diagram` not yet touched — see gaps above)

**Exposes** (per the root README's shared contracts — real starting C2,
shape-only through C1):
```json
{
  "round": 3,
  "theta_used": {"ring_size": 6, "decoy_density": 0.7, "fan_out_ratio": 0.4},
  "recall": 0.61,
  "bottleneck_distance_from_prev_round": 0.44
}
```
```json
{
  "bank_pairs": [{"bank_a": "Bank1", "bank_b": "Bank2", "overlap_estimate": 0.73, "simulated": true}]
}
```
(cross-bank shape above is C3 — not built yet.)

**New in C1:**
```python
from src.bo import BayesOptimizer, make_bo_generate_and_score_fns

# Low-level: ask/tell directly
bo = BayesOptimizer(seed=42)
theta = bo.ask()          # -> theta dict, same shape as stub_generator's
bo.tell(recall=0.61)      # feeds (theta, 1 - recall) back into the GP

# High-level: drop straight into loop.run_loop, no other changes needed
generate_fn, score_fn, bo = make_bo_generate_and_score_fns(seed=42)
run_loop(n_rounds=8, generate_fn=generate_fn, score_fn=score_fn)
```

## Getting started

```bash
cd adversarial
pip install -r requirements.txt
python -m src.loop            # C0: runs 5 stub rounds (random theta), writes output/stub_rounds/
python -m src.bo               # C1: runs 8 GP-driven rounds, writes output/bo_rounds/
python tests/test_smoke.py    # confirms C0 output shape
python tests/test_bo_smoke.py # confirms C1 output shape + GP accumulates observations
```

## Stack

Python, scikit-optimize (Gaussian Process + Expected Improvement, C1 —
now in use via `src/bo.py`), GUDHI/Ripser or `persim` for bottleneck
distance (C2; can also call B's own implementation instead of a separate
library).
