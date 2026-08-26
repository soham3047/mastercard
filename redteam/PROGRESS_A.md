# PROGRESS_A.md — Red Team / Attack Generation

## SYNOPSIS — Person A Stage 0: Scaffold
- Files created/changed: `redteam/requirements.txt`, `redteam/src/__init__.py`, `redteam/src/schema.py`, `redteam/src/writer.py`
- What it does: Sets up the `redteam/` package and defines the shared `Node`/`Edge`/`TransactionGraph` dataclasses that every later stage imports from, plus a `stub_output()` that writes placeholder data matching the exact interface-contract shape so B and D can build against the real file format immediately.
- Key decisions/assumptions made:
  - Used dataclasses + a `validate()` method (checks for duplicate node ids, dangling edge references, non-positive amounts, self-loops) rather than raw dicts, so every later stage gets schema errors caught immediately instead of silently writing malformed JSON.
  - `Edge.from_` (Python keyword collision) serializes to `"from"` on the wire via `to_dict()` — internal attribute name differs from the JSON key by design.
- Interface it EXPOSES: `{"nodes": [{"id","type","created_at"}], "edges": [{"from","to","timestamp","amount","label"}]}` exactly as specified in the shared contract.
- Interface it CONSUMES: none — self-contained.
- Commands to run: `python -m src.writer` (writes `output/stub/`)
- Open questions for the team: none

---

## SYNOPSIS — Person A Stage 1: Ring generator
- Files created/changed: `redteam/src/ring_generator.py`
- What it does: Generates directed cyclic rings — `acct_0 -> acct_1 -> ... -> acct_{k-1} -> acct_0` — with Poisson-distributed ring size (min 3), Exponential inter-hop timing gaps, and LogNormal structuring amounts. This is the shape Person B's persistent-homology filtration is meant to catch as a long-lived H1 feature.
- Key decisions/assumptions made:
  - Ring size `k = max(3, Poisson(λ))` — clamped to 3 since a "ring" under 3 nodes isn't a meaningful directed cycle.
  - Multiple rings in one graph are time-staggered (random start offset up to 48h) rather than all starting at t=0, since simultaneous starts would be an unrealistic tell.
  - Amounts drawn via `calibration.sample_structuring_amount()`, not raw `rng.lognormal()` — see the Stage A4 bugfix note below.
- Interface it EXPOSES: `generate_ring(...)`, `generate_rings(...)` -> `TransactionGraph`
- Interface it CONSUMES: `calibration.sample_structuring_amount`, `schema.{Node,Edge,TransactionGraph}`
- Commands to run: exercised via `python -m src.main`; unit-checked in `tests/test_smoke.py::check_rings_close`
- Open questions for the team: none

---

## SYNOPSIS — Person A Stage 2: Fan-out/star generator
- Files created/changed: `redteam/src/fanout_generator.py`
- What it does: Generates hub-and-spoke mule networks — one hub, several spokes, ~50/50 split between fan-out (hub→spokes) and fan-in (spokes→hub) edges, fired in a compressed burst (3x tighter timing than ring hops). Topologically a tree, not a cycle — invisible to B's persistent homology by design, which is why B4's graph-feature extractor exists.
- Key decisions/assumptions made:
  - Spoke count = `max(2, Poisson(λ) * (0.5 + fan_out_ratio))` — `fan_out_ratio ∈ [0,1]` scales aggressiveness without needing to retune λ directly; this is the primary knob C will likely mutate.
  - Spoke nodes are typed `"ring"` (fraud-participant), not `"decoy"` — the schema's `NodeType` only distinguishes ring/decoy/hub, and spokes are fraud participants even though the shape isn't a cycle.
- Interface it EXPOSES: `generate_fanout(...)`, `generate_fanouts(...)` -> `TransactionGraph`
- Interface it CONSUMES: `calibration.sample_structuring_amount`, `schema.{Node,Edge,TransactionGraph}`
- Commands to run: exercised via `python -m src.main`
- Open questions for the team: B4's feature extractor should confirm `time-compression` (my 3x-tighter timing) and `fan_out_ratio` are actually the features it's computing over — I picked names to match, worth a 1-line confirm from B.

---

## SYNOPSIS — Person A Stage 3: Decoy noise + realism layer
- Files created/changed: `redteam/src/decoy_generator.py`
- What it does: Background non-fraud transactions between random account pairs, same LogNormal amount family as everything else, but timed via a self-exciting Hawkes process (Ogata's thinning algorithm) instead of flat Poisson — so decoy activity clusters in bursts the same way real traffic does, and B can't separate ring/decoy bursts by timing density alone.
- Key decisions/assumptions made:
  - Hawkes intensity `λ(t) = μ + κ·Σ β·exp(-β(t-tᵢ))`; enforced `κ < 1` (stability — a supercritical process can explode into an infinite cascade).
  - One-week (`24*7` hour) decoy window per run, independent of the ring/fan-out time span.
  - `decoy_density` (in theta) trims decoy *edges* post-generation to hit a target decoy:fraud ratio — it doesn't feed back into the Hawkes parameters themselves. Raise `hawkes_baseline_mu`/`hawkes_kappa` if you need a higher decoy-count ceiling than trimming can reach.
- Interface it EXPOSES: `simulate_hawkes_events(...)`, `generate_decoys(...)` -> `TransactionGraph`
- Interface it CONSUMES: `schema.{Node,Edge,TransactionGraph}`
- Commands to run: `tests/test_smoke.py::check_hawkes_clusters` (checks inter-event gap coefficient-of-variation > 1, confirming clustering vs. a Poisson baseline)
- Open questions for the team: none

---

## SYNOPSIS — Person A Stage 4: Calibrate distributions to real-world statistics
- Files created/changed: `redteam/src/calibration.py`
- What it does: Grounds two things in public aggregate statistics instead of arbitrary guesses — (1) the background/decoy amount distribution, fit to NPCI's published FY2025-26 average UPI ticket size (~₹1,300) and the published ~86%-under-₹500 mass statistic; (2) the structuring/reporting threshold, set to the real PMLA Cash Transaction Report limit (₹10,00,000), replacing the shared doc's illustrative ₹10,000 example.
- Key decisions/assumptions made:
  - **Sources** (accessed August 2026): NPCI UPI monthly stats via coinlaw.io (mean ticket size); Worldline India Digital Payments Report H2 2024 via Business Standard (P2P vs P2M ticket size split); NPCI stats via meetanshi.com (86%-under-₹500 figure); FIU-IND/PMLA Rule 3 (CTR threshold). Full citations in `calibration.py` docstring.
  - Fit a 2-parameter LogNormal to the mean (₹1,300, from published stats) and a conservative median estimate (₹300, bounded by the 86%-under-₹500 stat but not equal to it — a single LogNormal can't simultaneously match a mean, a median, and a specific quantile mass if the real distribution is closer to a mixture; verified at scale (n=200k) this gets mean=₹1,306 and median=₹300 almost exactly, but only 61.6% under ₹500 vs. the real 86% — **documented limitation, not hidden**, worth a line in the writeup rather than an overclaim.
  - Hop-timing and Hawkes burst parameters are explicitly **not** calibrated from these sources — NPCI publishes volume/value, not inter-transaction timing distributions for individual account pairs — flagged as principled modeling choices instead of "calibrated," to avoid overclaiming.
  - **Bug found and fixed during testing:** an initial `sigma=0.18` for the structuring LogNormal put ~40% of generated ring amounts *over* the ₹10 lakh threshold (sigma is a multiplicative spread for LogNormal, so 0.18 is far wider than it sounds). Fixed by dropping to `sigma=0.02` and adding `sample_structuring_amount()`, which rejection-samples (up to 10 tries, then clips) to guarantee structuring edges land under threshold — verified across 5 seeds / 155 structuring edges with 0 violations.
- Interface it EXPOSES: `DECOY_AMOUNT_MU/SIGMA`, `RING_AMOUNT_MU/SIGMA`, `REPORTING_THRESHOLD_INR`, `sample_structuring_amount(...)`
- Interface it CONSUMES: none — self-contained.
- Commands to run: `python -m src.calibration` (prints the full derivation + sanity numbers)
- Open questions for the team: the 61.6%-vs-86%-under-₹500 gap above is worth mentioning honestly in the "fidelity of simulation" section of the writeup rather than glossing over — happy to discuss whether a mixture distribution is worth the extra complexity given remaining time.

---

## SYNOPSIS — Person A Stage 5: LLM identity + transaction content
- Files created/changed: `redteam/src/identity_llm.py`
- What it does: Generates a synthetic KYC-style identity (name, occupation, income bracket, address-plausibility note) and a one-sentence transaction narrative per account via an LLM call — this is what makes the output "GenAI-powered," not just parametric sampling. Falls back to an offline templated generator (no network needed) whenever `ANTHROPIC_API_KEY` isn't set, the SDK isn't installed, or any individual API call fails.
- Key decisions/assumptions made:
  - Batches accounts (default 25/call) rather than one call per account, to keep this fast for graphs with 50-100+ accounts.
  - Fallback is per-batch, not all-or-nothing — a single failed batch degrades to offline generation for just that batch instead of killing the whole run.
  - System prompt explicitly frames all output as synthetic test fixtures for a fraud-detection research benchmark, not real people/data.
- Interface it EXPOSES: `generate_identities(accounts, seed, batch_size) -> {account_id: identity_dict}`
- Interface it CONSUMES: Anthropic API (optional — graceful offline fallback)
- Commands to run: `python -m src.main --out output/run0` (omit `--skip-llm`); set `ANTHROPIC_API_KEY` for real GenAI output, otherwise runs offline automatically
- Open questions for the team: D should decide whether `identities.json` gets its own dashboard panel or gets merged into the existing "explanation" panel that consumes B's output — I kept it as a separate file rather than guessing.

---

## SYNOPSIS — Person A Stage 6: Parameterize for adversarial loop
- Files created/changed: `redteam/src/theta.py`, `redteam/src/main.py`
- What it does: Collects every generator knob into one flat `Theta` dataclass (`ring_size_lambda`, `num_rings`, `fan_out_ratio`, `hub_out_degree_lambda`, `num_hubs`, `hop_timing_lambda`, `hawkes_baseline_mu/kappa/beta`, `ring_amount_mu/sigma`, `decoy_amount_mu/sigma`, `reporting_threshold_inr`, `structuring_margin`, `decoy_density`, `num_decoy_accounts`, `seed`) with calibrated defaults from Stage A4. `main.build_graph(theta)` is the single function Person C's Bayesian-optimization loop should call each round.
- Key decisions/assumptions made:
  - Exact key names differ from the shared doc's rough sketch (`ring_size`, `hop_timing_params`) — this file is the source of truth per the doc's own note ("exact keys confirmed by A's synopsis"). C should import `Theta`/`DEFAULT_THETA` directly rather than hardcoding key names.
  - `Theta.validate()` catches the one truly unsafe input (`hawkes_kappa >= 1.0`, which would make the Hawkes process explode) plus basic range checks — called automatically inside `build_graph()`.
  - `main.py` is both a CLI entrypoint (`python -m src.main`) and an importable function (`from src.main import build_graph`) — C should use the import path for speed across many rounds, not shell out per round.
- Interface it EXPOSES: `Theta` dataclass + `DEFAULT_THETA`; `build_graph(theta) -> TransactionGraph`
- Interface it CONSUMES: all prior Red Team stages (A1-A5)
- Commands to run: `python -m src.main --out output/round0 --theta overrides.json` for file-based overrides, or import `build_graph` directly for in-process use
- Open questions for the team: none — this is the final Red Team handoff; C/D should treat `src/theta.py` and `src/schema.py` as the two files to read first.
