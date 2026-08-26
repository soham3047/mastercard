# Blue Team / Detection Core — Person B

The mathematical heart of the project. Fraud rings show up as topological
structure in a transaction graph: a ring/layering attack creates a cycle,
detectable via persistent homology as a long-lived H1 feature in a
time-based filtration. A fan-out/star attack is topologically trivial —
caught separately via graph/account features. Both halves are fused into
one classifier, benchmarked against a naive clustering baseline.

## Stages

- **B0 — Sanity check (highest-risk unknown in the project — do first, hardcoded).**
  Hand-build one tiny 5-node ring + a few decoys, run through GUDHI/Ripser,
  confirm it shows up as a long-lived H1 feature.
- **B1 — Filtration + persistence pipeline.** Generalize B0 into a real
  pipeline over any input graph matching the transaction-graph schema.
- **B2 — Multiparameter persistent homology (bifiltration).** Extend to a
  bifiltration over (time, amount-proximity-to-threshold) to catch
  structuring/smurfing that a pure time-filtration misses.
- **B3 — Incremental/streaming persistence ("vineyards").** Implement
  `update_diagram(prev_diagram, new_edges)` — the technique the project is
  named after (Cohen-Steiner, Edelsbrunner, Morozov).
- **B4 — Feature-fusion side.** Graph/account feature extractor (in/out-degree
  bursts, fan-out ratio, time-compression, account age, amount variance) —
  catches what B1–B3 structurally cannot see.
- **B5 — Fusion + naive baseline + significance testing.** Vectorize
  diagrams, concatenate with B4 features, train a classifier; build the
  naive clustering baseline; Mann-Whitney U / Kruskal-Wallis significance test.
- **B6 — Explainability layer.** Translate detected-ring data into the
  plain-language `explanation` field.
- **B7 — Expose final scoring interface.** Finalize `score_batch` and the
  output schema exactly, since C and D build against it.

## Input schema you consume (from Red Team)

```json
{
  "nodes": [{"id": "acct_001", "type": "ring|decoy|hub", "created_at": "ISO8601"}],
  "edges": [{"from": "acct_001", "to": "acct_002", "timestamp": "ISO8601", "amount": 452.30, "label": "ring|decoy"}]
}
```

## Output schema (interface contract — Adversarial Loop and Dashboard depend on this exactly)

```json
{
  "diagram": [{"birth": 0.2, "death": 4.7, "dimension": 1, "filtration": "time|time_amount_bifiltration"}],
  "ring_persistence_score": 3.81,
  "detected_ring_nodes": ["acct_003", "acct_004"],
  "explanation": "Ring closed via acct_003 -> acct_004 -> acct_007 over 6 days, avg transfer ₹9,800, consistently just under the ₹10,000 reporting threshold."
}
```

Plus callables: `score_batch(graph_json) -> {precision, recall, f1, auc, per_account_fraud_prob}`
and `update_diagram(prev_diagram, new_edges) -> new_diagram`.

Log progress in `PROGRESS_B.md` in this folder.
