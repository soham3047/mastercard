# VINEYARD

Synthetic identity fraud ring detection via persistent homology, in a closed
Red Team / Blue Team loop with adversarial refinement.
Built for the **Mastercard Innovation Challenge @ GFF 2026** (Identify → Generate → Defend).

## Concept

Synthetic identity fraud rings look clean per-transaction but form a
detectable *shape* at the network level:

- **Ring / layering attacks** → cycles → detected via persistent homology
  (a long-lived H1 feature in a time+amount filtration).
- **Fan-out / mule networks** → stars → topologically trivial, caught instead
  via graph/account features (in-degree bursts, fan-out ratio, time-compression).

A Bayesian-optimization adversarial loop then tunes the attack parameters
round-over-round against whatever the detector missed, and a mocked
cross-bank layer simulates ring-overlap verification across institutions
without sharing raw data.

## Team & tracks

| Person | Track | Folder | Owns |
|---|---|---|---|
| A | Red Team / Attack Generation | `redteam/` | Ring + fan-out generators, decoy realism, LLM identity content |
| **B** | **Blue Team / Detection Core** | `blueteam/` | **Persistent homology, bifiltration, incremental "vineyards" updates, feature fusion, explainability** |
| C | Adversarial Loop + Cross-Bank | `adversarial/` | Bayesian optimization loop, mocked cross-bank commitment layer |
| D | Backend + Dashboard | `backend/`, `frontend/` | FastAPI API, React dashboard, end-to-end integration |

Each person works in their own folder, in their own Claude session, using
their individual master prompt (the per-person sections in the original
`vineyard.md` planning doc). Work happens in stages — finish a stage, write
its synopsis, stop, wait for "continue."

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

Merge into root `PROGRESS.md` at team sync points.

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
update_diagram(prev_diagram, new_edges) -> new_diagram   # avoids full recompute per round
```

**5. Round Result** — Adversarial Loop → Dashboard
```json
{
  "round": 3,
  "theta_used": {"ring_size": 6, "decoy_density": 0.7, "fan_out_ratio": 0.4},
  "recall": 0.61,
  "bottleneck_distance_from_prev_round": 0.44
}
```

**6. Cross-Bank Verification** — Adversarial Loop → Dashboard
```json
{
  "bank_pairs": [{"bank_a": "Bank1", "bank_b": "Bank2", "overlap_estimate": 0.73, "simulated": true}]
}
```

## Getting started (each person)

```bash
git clone <repo-url>
cd vineyard/<your-folder>
pip install -r requirements.txt      # backend/redteam/blueteam/adversarial
# or, for frontend:
npm install
```

Then paste your track's master prompt into a fresh Claude chat, point it at
your folder, and start with your Stage 0.

## Stack

- Red Team: Python, numpy/scipy, LLM API client, NetworkX-compatible JSON
- Blue Team: Python, GUDHI/Ripser, NetworkX, scikit-learn, scipy.stats
- Adversarial Loop: Python, scikit-optimize/GPy/botorch, GUDHI/Ripser or `persim`
- Backend: Python/FastAPI
- Frontend: React + Vite, Recharts or D3
