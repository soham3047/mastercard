# VINEYARD — Topological Synthetic-Identity Fraud Detection
### Seeing fraud rings as shapes: persistent homology in a closed Red Team / Blue Team adversarial loop

Built for the **Mastercard Innovation Challenge @ GFF 2026** (Identify → Generate → Defend).

VINEYARD detects synthetic-identity fraud rings by treating a bank's transaction
network as a *shape*, not a table of rows. Ring/layering attacks show up as
long-lived cycles under persistent homology; fan-out mule networks don't, and are
caught with graph features instead. A Bayesian-optimization loop then plays
attacker against detector round-over-round, a mocked cross-bank layer simulates
ring-overlap verification without banks sharing raw data, and a federated-learning
extension explores the privacy/utility tradeoff of pooling fraud signal across
banks under differential privacy. Everything is wired end-to-end behind a FastAPI
backend and a live React dashboard — no stubs remain in the core pipeline.

> **Status:** all four tracks (A/B/C/D below) have shipped real, working code and
> are wired together for real by the backend. The system runs; the honest caveats
> and open questions the team flagged along the way are collected in
> [Known limitations & open items](#known-limitations--open-items) rather than
> swept under the rug.

## Concept

Synthetic identity fraud rings look clean per-transaction but form a
detectable *shape* at the network level:

- **Ring / layering attacks** → cycles → detected via persistent homology
  (a long-lived H1 feature in a time+amount bifiltration).
- **Fan-out / mule networks** → stars → topologically trivial, caught instead
  via graph/account features (in-degree bursts, fan-out ratio, time-compression).

A Bayesian-optimization adversarial loop then tunes the attack parameters
round-over-round against whatever the detector missed, a mocked cross-bank
layer simulates ring-overlap verification across institutions without
sharing raw data, and a federated-learning layer explores pooling fraud
signal across banks under differential privacy.

## What's actually in here now

The system grew past the original three-endpoint plan. Beyond the core
generate → detect → optimize loop, it now also includes:

- **Calibrated attack generation** — background transaction amounts and the
  structuring/reporting threshold are grounded in published NPCI/Worldline
  UPI statistics and the real PMLA CTR limit, not arbitrary numbers (see
  `redteam/src/calibration.py`).
- **LLM-generated synthetic identities** — optional GenAI-authored KYC-style
  identity + narrative content per account, with an offline templated
  fallback when no API key is set.
- **Incremental "vineyards" persistence** — streaming diagram updates instead
  of a full recompute every round (with one caveat — see below).
- **Explainability** — plain-language, cycle-based attribution of exactly
  which accounts closed a ring and why.
- **Time-to-detect benchmarking** — how many transactions elapse before a
  structuring ring closes and gets caught, at realistic decoy-noise levels.
- **Diagram-shape matching** — classifies a persistence diagram against a
  small library of canonical attack shapes (small ring / large ring /
  multi-ring layering / fan-out star / decoy-only noise) via bottleneck
  distance.
- **Synthetic-identity-at-onboarding detection** — a second, separate attack
  family (account-opening fraud, not transaction rings).
- **Mocked cross-bank commitment layer** — salted-hash overlap estimation
  between banks' flagged-account lists, explicitly labeled `"simulated": true`
  everywhere, with the real-crypto gap (DH-PSI/OPRF-PSI vs. a shared salt)
  documented rather than glossed over.
- **Federated learning + differential privacy** — a small FedAvg simulation
  across banks, plus a privacy/utility tradeoff curve over varying ε.

## Team & tracks

| Person | Track | Folder | Owns |
|---|---|---|---|
| A | Red Team / Attack Generation | `redteam/` | Ring + fan-out generators, decoy realism (Hawkes-process timing), calibration, LLM identity content, `Theta` parameterization |
| B | Blue Team / Detection Core | `blueteam/` | Persistent homology, bifiltration, incremental "vineyards" updates, feature fusion, explainability, final `score_batch` interface |
| C | Adversarial Loop + Cross-Bank | `adversarial/` | Bayesian-optimization round loop, mocked cross-bank commitment layer, federated learning + differential privacy |
| D | Backend + Dashboard | `backend/`, `vineyard-frontend/` | FastAPI API, real end-to-end integration of A+B+C, benchmarking/shape-library analysis endpoints, React dashboard |

Each person works in their own folder, in their own Claude session, using
their individual master prompt. Work happens in stages — finish a stage,
write its synopsis, stop, wait for "continue."

## Progress tracking

Each folder has its own `PROGRESS_[A-D].md`. Append one synopsis block per
completed stage:

```
## SYNOPSIS — Person [X] Stage N: [name]
- Files created/changed: [paths]
- What it does: [2-4 sentences]
- Key decisions/assumptions made: [bullets]
- Interface it EXPOSES: [schema/function signature]
- Interface it CONSUMES: [bullets, or "none"]
- Commands to run: [exact commands]
- Open questions for the team: [bullets, or "none"]
```

Merge into root `PROGRESS.md` at team sync points. Every track has kept this
up — the individual files are the most detailed record of what was built,
what broke, and what's still open; this README summarizes them.

## Live API (`backend/`)

The FastAPI backend wires all three tracks in for real (no stubs). Run it
with `uvicorn app.main:app --reload` from `backend/`, then:

| Endpoint | What it does |
|---|---|
| `GET /health` | Per-track live/unavailable status, fails loud instead of silently degrading |
| `GET /api/theta/default` | A's `DEFAULT_THETA`, for slider defaults/ranges |
| `GET /api/graph` | A's real generator — `?decoy_density=&seed=` |
| `POST /api/detect` | B's real `score_batch` on a supplied transaction graph |
| `GET /api/scan` | Convenience: generate (A) + detect (B) in one call |
| `GET /api/rounds` | C's real Bayesian-optimization round loop over A+B, with real bottleneck distance between rounds — `?n_rounds=&seed=&refresh=` |
| `GET /api/cross-bank` | C's simulated cross-bank commitment layer, federated-learning summary merged in |
| `GET /api/federated/privacy-tradeoff` | Differential-privacy/utility curve — `?epsilons=0.2,0.5,1,3,10&seed_base=` |
| `GET /api/federated/flags` | Federated flagging output — `?secure=true` |
| `GET /api/benchmarks/time-to-detect` | Transactions-to-closure benchmark — `?n_runs=&seed_start=&decoy_density=&refresh=` |
| `GET /api/shape-library` | Catalog of canonical attack-shape templates |
| `POST /api/shape-match` | Classify a diagram against the shape library |
| `GET /api/synthetic-identity-scan` | D's second attack family: account-opening synthetic identity fraud |
| `POST /api/identities` | On-demand LLM identity generation for a set of accounts (slower — not part of the poll loop) |

## Shared interface contracts

**1. Transaction Graph** — Red Team → Detection, Red Team → Dashboard
```json
{
  "nodes": [{"id": "acct_001", "type": "ring|decoy|hub", "created_at": "ISO8601"}],
  "edges": [{"from": "acct_001", "to": "acct_002", "timestamp": "ISO8601", "amount": 452.30, "label": "ring|decoy"}]
}
```

**2. Persistence Diagram + RPS + Explanation** — Detection → Adversarial Loop, Detection → Dashboard
```json
{
  "diagram": [{"birth": 0.2, "death": 4.7, "dimension": 1, "filtration": "time|time_amount_bifiltration"}],
  "ring_persistence_score": 3.81,
  "detected_ring_nodes": ["acct_003", "acct_004"],
  "explanation": "Ring closed via acct_003 -> acct_004 -> acct_007 over 6 days, avg transfer ₹9,800, consistently just under the ₹10,000 reporting threshold."
}
```

**3. Detector Scoring Call** — Detection → Adversarial Loop
```
score_batch(graph_json) -> {
  "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0,
  "per_account_fraud_prob": {"acct_001": 0.92}
}
```

**4. Incremental Update Call** — Detection internal, Detection → Adversarial Loop
```
update_diagram(prev_diagram, new_edges) -> (new_diagram, state)   # avoids full recompute per round
```
> `state` is a live `PersistentGraph` object that must be kept alive across
> rounds — not a bare-diagram-in/bare-diagram-out call. The real adversarial
> loop (`adversarial/`) doesn't currently call this at all: each round
> resamples a fresh graph rather than accumulating one, so there's no
> "new edges since last round" to hand it. Bottleneck distance is instead
> computed directly from consecutive rounds' full diagrams. See
> `adversarial/PROGRESS_C.md` Stage 2 for the full reasoning — flagged as a
> deliberate, documented deviation, not an oversight.

**5. Round Result** — Adversarial Loop → Dashboard
```json
{
  "round": 3,
  "theta_used": {"ring_size_lambda": 6, "decoy_density": 0.7, "fan_out_ratio": 0.4, "hop_timing_params": {"lambda": 2.1}},
  "recall": 0.61,
  "bottleneck_distance_from_prev_round": 0.44
}
```
> `theta_used`'s real keys follow A's `Theta` dataclass, not the illustrative
> 3-key sketch above — `hop_timing_params` is a real 4th key. See
> `redteam/src/theta.py` for the full 16-field set (the adversarial loop
> currently tunes a 4-field subset; the rest sit at `DEFAULT_THETA`).

**6. Cross-Bank Verification** — Adversarial Loop → Dashboard
```json
{
  "bank_pairs": [{"bank_a": "Bank1", "bank_b": "Bank2", "overlap_estimate": 0.73, "simulated": true}]
}
```

## Known limitations & open items

Carried forward from the individual tracks' progress logs, because the
project's own convention is to flag these rather than hide them:

- **Adversarial objective sign is still undecided.** The loop minimizes
  `J(theta) = 1 - recall`, which pushes the GP toward *higher* recall each
  round — but "tune attacks against what the detector missed" reads like the
  attacker should be evading detection (recall going *down*). Round-over-round
  trends should currently be read as "the GP explored the search space," not
  yet as "attacker got better." (`adversarial/PROGRESS_C.md`, C1/C2)
- **Detector precision/recall/f1/auc of 1.0 is a training-set number, not a
  validated one.** `score_batch` was evaluated on the same synthetic set the
  model trained on; held-out evaluation is still needed before these numbers
  go in a writeup. (`blueteam/PROGRESS_B.md`, B7)
- **Cross-bank verification is explicitly simulated**, not real
  privacy-preserving cryptography — it uses a shared, publicly-known salt so
  matching works at all, which is itself a real privacy weakness (dictionary
  attack over a guessable fingerprint space). A real system would need
  DH-PSI/OPRF-PSI instead. Every output record is tagged `"simulated": true`
  for exactly this reason. (`adversarial/src/crossbank/commitment.py`)
- **Calibration gap:** the fitted LogNormal amount distribution matches the
  published mean (₹1,300) and median (₹300) closely but only gets ~61.6% of
  mass under ₹500 vs. the real ~86% — a single LogNormal can't match a mean,
  median, and quantile mass simultaneously if the real distribution is closer
  to a mixture. Documented, not hidden. (`redteam/src/calibration.py`)
- **Shape-matching confuses ring *sizes*.** Bottleneck distance cleanly
  separates "has a ring vs. doesn't" and "one ring vs. many," but small-ring
  vs. large-ring margins were razor-thin (~0.001) in testing — would need a
  size-aware feature alongside the diagram match to fix.
- **Frontend/backend contract mismatch.** The dashboard's `src/api/client.js`
  was built against an earlier, illustrative endpoint shape
  (`GET /api/graph?round=n`, `GET /api/persistence`, `GET /api/rps`, etc.)
  and falls back to mock data when a route isn't live. The real backend's
  endpoints (see [Live API](#live-api-backend) above) use different paths
  and query params (`?decoy_density=&seed=`, no `/api/persistence` or
  `/api/rps` split). Reconciling these is the next integration step — only
  `client.js` and `mockData.js` should need to change.
- **`redteam_adapter.py` has no fallback** the way `blueteam_adapter.py`
  does — if `redteam/` isn't importable, the adversarial loop hard-crashes
  at import time instead of degrading loudly-but-gracefully like the B side.
- **External-dataset validation (PaySim/SAML-D) was deliberately not
  started** — flagged as a parallel-track decision, not folded into any
  single track unilaterally.

## Getting started (each person)

```bash
git clone <repo-url>
cd vineyard/<your-folder>
pip install -r requirements.txt      # backend/redteam/blueteam/adversarial — use --break-system-packages if needed
# or, for frontend:
npm install
```

Then paste your track's master prompt into a fresh Claude chat, point it at
your folder, and start with your Stage 0.

### Running the full stack

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# First request trains B's model in-process once and caches it to
# backend/app/_trained_model.pkl (gitignored) — restarts after that are instant.

# Frontend (separate terminal)
cd vineyard-frontend
npm install
npm run dev
```

Quick check everything's alive:
```bash
curl localhost:8000/health
curl "localhost:8000/api/scan?seed=1"
curl "localhost:8000/api/rounds?n_rounds=3&seed=1"
curl localhost:8000/api/cross-bank
curl "localhost:8000/api/benchmarks/time-to-detect?n_runs=15&decoy_density=12"
curl localhost:8000/api/shape-library
```

If `/health` shows `"unavailable"` for any track, the same response's
`ab_error` / `adv_error` field names the cause — almost always a missing
package from `requirements.txt`.

## Stack

- **Red Team** (`redteam/`): Python, numpy/scipy, networkx, Anthropic API
  client (optional, offline fallback), python-dotenv
- **Blue Team** (`blueteam/`): Python, GUDHI, networkx, scikit-learn, scipy
  (Mann-Whitney U / Kruskal-Wallis significance testing), persim
- **Adversarial Loop** (`adversarial/`): Python, scikit-optimize (GP +
  Expected Improvement), GUDHI, persim (bottleneck distance), scikit-learn
- **Backend** (`backend/`): Python, FastAPI, uvicorn, pydantic — imports all
  three tracks' code in-process
- **Frontend** (`vineyard-frontend/`): React + Vite, Recharts
#   M a s t e r c a r d  
 