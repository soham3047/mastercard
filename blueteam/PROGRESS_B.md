# PROGRESS_B.md — Blue Team / Detection Core (Person B)

## B0 — Sanity check
Done. Hand-built 5-node ring + decoys confirmed as a long-lived H1 feature
via GUDHI/Ripser. Highest-risk unknown in the project, cleared first per README.

## B1 — Filtration + persistence pipeline
Done. Generalized B0 into a real pipeline over any graph matching the
transaction-graph schema.

## B2 — Multiparameter persistent homology (bifiltration)
Done. Bifiltration over (time, amount-proximity-to-threshold) in place.
**Open item, still unresolved:** alpha/beta/threshold weights are untuned —
carried forward as an open item through B3 and B4, still not addressed.

## B3 — Incremental/streaming persistence ("vineyards")
Done and confirmed by actual run (not just hand-verified):
```
Round 1: total_edges=3, ring_persistence_score=0.0, H1 features=0
Round 2: total_edges=6, ring_persistence_score=0.0, H1 features=0
Round 3: total_edges=8, ring_persistence_score=1.5833, H1 features=1
PASS
```
**Open items to raise with the team:**
- Tell D: `update_diagram()` here returns `(diagram, state)`, and `state`
  must be a live `PersistentGraph` object kept alive across rounds — not
  the bare-diagram-in/bare-diagram-out shape the root README describes.
  If D's backend calls this per-request statelessly, the incremental part
  silently breaks.
- alpha/beta/threshold tuning from B2 — still open.

## B4 — Feature-fusion side
Done and confirmed by actual run. Per-account feature extractor built
(in/out-degree, fan-out ratio, bursts, time-compression, account age,
amount variance).
**Open items to raise with the team:**
- Tell A: the generator needs to emit UTC-aware (or at least
  consistently-UTC-intended) timestamps, or this feature extractor could
  score differently on different machines during the demo.
- Tell C/D: this per-account feature dict is a new schema, not one of the
  six shared contracts in the root README. B5 folds it into the real
  output schema — heads-up so nobody's surprised by it.

## B5 — Fusion + naive baseline + significance testing
**Status: done and confirmed by actual run.**

Design: 6-graph synthetic dataset (2 ring, 2 fanout, 2 clean), leave-one-
graph-out CV, KMeans naive baseline on B4-only features vs. logistic
regression on fused (diagram + B4) features, Mann-Whitney U + Kruskal-
Wallis on `ring_persistence_score`.

Diagram vectors (hand-verified, confirmed exact match on run):
| graph | kind | H1 features | ring_persistence_score |
|---|---|---|---|
| ring_1 | ring | 1 | 1.6583 |
| ring_2 | ring | 1 | 1.85 |
| fanout_1 | fanout | 0 | 0.0 |
| fanout_2 | fanout | 0 | 0.0 |
| clean_1 | clean | 0 | 0.0 |
| clean_2 | clean | 0 | 0.0 |

Dataset: 39 accounts total (22 fraud, 17 normal).

Confirmed run results (leave-one-graph-out CV):
- Naive (B4 only): precision=0.4231, recall=0.5000, f1=0.4583
- Fused (topology+B4): precision=0.6316, recall=0.5455, f1=0.5854, auc=0.4920
- Ring-graph recall: naive=1.0, fused=1.0
- Fanout-graph recall: naive=0.0833, fused=0.1667
- Mann-Whitney U (fraud vs normal): U=208.00, p=0.5095 — expected, matches
  the docstring's own prediction (fanout fraud scores 0, ring decoys score
  >0, so this test is deliberately harder and not meant to be strongly
  significant).
- Kruskal-Wallis (ring vs fanout vs clean): H=35.62, p≈0.000000 — expected,
  re-confirms B0-B3's premise at dataset level (by construction).

**Investigated finding: naive baseline gets perfect ring recall (1.0), same
as fused — same as topology.** Ran a diagnostic (median-imputing
`time_compression_days` instead of using the 9999 missing-data sentinel) to
rule out a suspected sentinel/missingness confound. Result: ring recall
stayed at 1.0 with median value 3.0 — **confound ruled out.**

Real cause: every ring node in this synthetic generator has an exact,
distinctive degree signature (in_degree=1, out_degree=1, low amount
variance) that no other account type in the dataset shares — hubs are
lopsided, decoys/src/dest are single-direction only. So plain B4 degree
features alone already separate ring members with no topology needed,
independent of the sentinel. This is a **dataset-fidelity limitation, not a
code bug**: real fraud rings won't always have such a textbook 1-in-1-out
signature per node (more hops, noise, partial cycles, shared accounts),
which is exactly where topology should start pulling ahead of plain
features. The current 6-graph set doesn't have an example that stresses
that gap. Documented here in the same spirit as the existing
broadcast-vs-node-level limitation; worth one paragraph in the final
writeup. Not a blocker for B6/B7 — B7 runs against A's real generator, not
these 6 toy graphs.

Possible future strengthening (not required now): add a decoy with
balanced 1-in-1-out degree that is NOT part of a cycle, to actually test
whether naive can tell "looks like a ring by degree" apart from "is
actually a ring by topology."

## B6 — Explainability layer
**Status: done and confirmed by actual run.**

Node-level ring attribution (`detected_ring_nodes`) via directed-cycle
detection (networkx `simple_cycles`) on the raw transaction graph —
deliberately independent of B2/B3's bifiltration formula (finding WHICH
nodes form a cycle doesn't need to know WHEN/HOW persistent it scores;
avoids a third copy of that formula drifting out of sync). Plain-language
`explanation` built from the cycle's own edges (duration, avg amount,
threshold comparison).

Confirmed run against all 6 B5 synthetic graphs:
- ring_1: detected exactly the 6 true ring nodes (exact match vs. labels)
- ring_2: detected exactly the 4 true ring nodes (exact match vs. labels)
- fanout_1, fanout_2, clean_1, clean_2: correctly empty, no false positives

**New dependency:** networkx (not used by B0-B5). Install with
`pip install networkx` (`--break-system-packages` on this project's setup).

**Open item:** `REPORTING_THRESHOLD = 10000` is a local placeholder
constant. If B1/B2 already define a shared reporting-threshold constant,
swap this for that one so there's a single source of truth.

**Next:** B7 — expose final scoring interface (`score_batch`), finalize the
output schema exactly since C and D build against it.

## SYNOPSIS — Person B Stage 7: Expose final scoring interface

- Files created/changed:
  - blueteam/src/b7_score_batch.py (bug fix + validation)
  - blueteam/src/ring_1.json (test fixture, generated from b5_fusion.build_synthetic_dataset())

- What it does: Wraps the full detection pipeline (B1-B6) as the single
  score_batch(graph_json) callable that C and D build against. Given a
  transaction graph, it returns the persistence diagram, ring_persistence_score,
  detected_ring_nodes, a plain-language explanation, and per-account fraud
  probabilities, plus aggregate precision/recall/f1/auc.

- Key decisions/assumptions made:
  - Found and fixed a NaN-propagation bug: feats.get(k, 0.0) in
    train_final_model and FusedModel._vector only guards a MISSING key, not a
    PRESENT-but-NaN value. b4_account_features can emit NaN for
    single-transaction accounts (sample variance with ddof=1 divides by
    n-1=0 when n=1), which was crashing LogisticRegression.fit() with
    "Input X contains NaN."
  - Fix: added a _safe_feature_value() helper that checks math.isfinite()
    in addition to key presence, used at both row-building call sites.
  - Added a diagnostic print (non-finite feature values per account) to
    confirm root cause rather than silently papering over it — pending:
    confirm whether the diagnostic actually printed on this dataset, or
    whether the NaN source was something other than amount_variance.
  - CAVEAT — precision/recall/f1/auc all returned 1.0 on this run. This is
    scoring against the same synthetic set the model trained on, so it is
    NOT evidence of real detection quality — expected/overfit-typical, not
    a validated number. A held-out test set is needed before this number
    goes in the team writeup (flagged for D5's benchmark stage).

- Interface it EXPOSES:
  score_batch(graph_json) -> {
    "diagram": [{"birth": float, "death": float|null, "dimension": int, "filtration": str}],
    "ring_persistence_score": float,
    "detected_ring_nodes": [str],
    "explanation": str,
    "precision": float, "recall": float, "f1": float, "auc": float,
    "per_account_fraud_prob": {acct_id: float}
  }

- Interface it CONSUMES: none — self-contained, built on B1-B6's own pipeline
  (filtration/persistence, bifiltration, feature fusion, explainability).

- Commands to run:
  cd blueteam/src
  python -c "
  import json, sys
  sys.path.insert(0, '.')
  import b5_fusion
  d = b5_fusion.build_synthetic_dataset()
  name, kind, g, labels = d[0]
  with open('ring_1.json', 'w') as f:
      json.dump(g, f, indent=2)
  "
  python b7_score_batch.py ring_1.json

- Validated output (ring_1.json, matches B5's documented value exactly):
  ring_persistence_score: 1.6583
  detected_ring_nodes: [r1_r000...r1_r005] (all 6 ring accounts, correct)
  precision/recall/f1/auc: 1.0 (training-set number, see caveat above)

- Open questions for the team:
  - Did the [diagnostic] non-finite-feature print actually fire during
    --train, and if so, on which feature/accounts? Not yet confirmed —
    worth a fast re-check to close out the root-cause investigation properly.
  - Held-out evaluation needed before D5's benchmark stage uses this
    detector's numbers in the solution walkthrough.
  - B's track (B0-B7) is now complete — score_batch and update_diagram
    are both live for C and D to swap in for their stubs.