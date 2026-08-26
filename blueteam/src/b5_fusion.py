"""
VINEYARD — Blue Team, Stage B5: Fusion + naive baseline + significance testing.

IMPLEMENTATION NOTE — read before presenting this stage:
B0-B3 gave us a per-GRAPH topological score. B4 gave us per-ACCOUNT graph
features. This stage fuses them into one feature vector per account and
trains a real classifier on it, then compares that against a naive
clustering baseline that only sees B4's features (no topology at all), and
runs the significance tests the README calls for.

A GENUINE LIMITATION worth stating up front, not burying: the topological
half of the fused vector is a per-GRAPH score BROADCAST to every account in
that graph. It does not yet know WHICH specific accounts are the ring
members versus innocent bystanders sharing the same batch — that
node-level attribution is explicitly B6's job (`detected_ring_nodes`). So
even the fused classifier here is expected to correctly flag "this graph
contains a ring" more often than the naive baseline, but its PRECISION on
telling a true ring member apart from an unrelated decoy in the same graph
is a fundamentally harder problem this stage does not attempt to solve.

A SECOND LIMITATION found during review, also worth stating up front:
MISSING_COMPRESSION_SENTINEL below is only hit by accounts with strictly
one-directional flow (in-only or out-only). In this synthetic dataset that
happens to correlate heavily with fraud label (ring/hub nodes are almost
exactly the bidirectional ones), which means a distance-based method like
KMeans could partly be clustering on "has bidirectional flow" rather than
genuine topology-blind B4 signal -- weakening the naive-vs-fused contrast
this stage exists to demonstrate. Worth checking cluster centroids /
column-wise variance contribution once real results are in, and/or
re-running the naive baseline with this column median-imputed or replaced
by a binary flag as a robustness check.

ALSO WORTH STATING: unlike B0-B4, this stage fits real ML models (KMeans,
logistic regression) and runs nonparametric significance tests. Those don't
have closed-form answers that can be verified by hand the way persistent
homology birth/death values can — the exact numbers below are the actual
run's to report, not something pre-computed here.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import mannwhitneyu, kruskal

from b3_vineyards import PersistentGraph, ESSENTIAL_SCORE_BASE
from b4_account_features import extract_account_features, _parse_timestamp

# Missing-data sentinels for time_compression_days / account_age_days.
# Both features are SUSPICIOUS when SMALL (rapid pass-through; freshly
# created and immediately used). Silently defaulting missing data to 0.0 --
# as a lazier implementation might -- would quietly bias the classifier
# toward flagging accounts we simply lack data on, which is backwards. Large
# sentinels push AWAY from the suspicious direction when data is missing,
# instead of accidentally assuming the worst.
#
# CAVEAT (see module docstring): in THIS synthetic dataset, missingness on
# time_compression_days is strongly correlated with fraud label because of
# how the generator assigns single- vs bidirectional edges. Sanity-check
# real results against this before trusting naive-baseline recall numbers
# at face value.
MISSING_COMPRESSION_SENTINEL = 9999.0
MISSING_AGE_SENTINEL = 9999.0

DIAGRAM_VECTOR_NAMES = ["num_h1_features", "ring_persistence_score", "mean_feature_score", "min_birth", "max_birth"]
B4_VECTOR_NAMES = [
    "in_degree", "out_degree", "fan_out_ratio", "in_burst_count", "out_burst_count",
    "time_compression_days", "account_age_days", "amount_mean_in", "amount_mean_out",
    "amount_variance_in", "amount_variance_out",
]


# ---------- topological side: one-shot diagram + vectorization ----------

def graph_to_diagram_once(graph_json):
    """One-shot diagram for a whole graph, reusing B3's incremental engine
    as a plain non-streaming calculator (construct one PersistentGraph, feed
    it the entire edge list as a single round). Same bifiltration math as
    B2/B3 -- deliberately NOT re-derived here, to keep one source of truth
    for the formulas instead of a third copy drifting out of sync.

    NOTE: assumes PersistentGraph.add_edges() returns a dict with at least
    "diagram" and "ring_persistence_score" keys directly (not the
    (diagram, state) tuple shape update_diagram() uses). Confirm with
    print(graph_to_diagram_once(...).keys()) once against your actual
    b3_vineyards.py before trusting this end to end.
    """
    converted_edges = [
        {**e, "timestamp": _parse_timestamp(e["timestamp"])} for e in graph_json["edges"]
    ]
    pg = PersistentGraph()
    return pg.add_edges(converted_edges)


def diagram_to_vector(diagram_result):
    """Fixed 5-number summary: [num_h1_features, ring_persistence_score,
    mean_feature_score, min_birth, max_birth].

    Chose plain summary statistics over a full persistence-image/landscape
    representation (persim, already in requirements.txt) -- deliberately.
    This project's diagrams are graph-only complexes with typically 0-2 H1
    features per graph (see B0-B3), so an image/landscape would mostly
    encode a near-empty grid: more machinery (resolution, kernel bandwidth,
    birth-persistence coordinates) for a representation with almost nothing
    in it, and harder to explain to a judge than five plain numbers. Worth
    revisiting if a later stage's graphs get topologically richer (multiple
    independent rings per graph).
    """
    feats = diagram_result["diagram"]
    if not feats:
        return [0, diagram_result["ring_persistence_score"], 0.0, 0.0, 0.0]
    scores = [(ESSENTIAL_SCORE_BASE - f["birth"]) if f["death"] is None else (f["death"] - f["birth"]) for f in feats]
    births = [f["birth"] for f in feats]
    return [
        len(feats),
        diagram_result["ring_persistence_score"],
        round(sum(scores) / len(scores), 4),
        round(min(births), 4),
        round(max(births), 4),
    ]


def b4_features_to_vector(f):
    """Fixed-order numeric vector from one extract_account_features() entry.
    See MISSING_*_SENTINEL comments above for why None isn't just replaced
    with 0.0."""
    return [
        f["in_degree"], f["out_degree"], f["fan_out_ratio"],
        f["in_burst_count"], f["out_burst_count"],
        f["time_compression_days"] if f["time_compression_days"] is not None else MISSING_COMPRESSION_SENTINEL,
        f["account_age_days"] if f["account_age_days"] is not None else MISSING_AGE_SENTINEL,
        f["amount_mean_in"] if f["amount_mean_in"] is not None else 0.0,
        f["amount_mean_out"] if f["amount_mean_out"] is not None else 0.0,
        f["amount_variance_in"], f["amount_variance_out"],
    ]


# ---------- synthetic dataset ----------

def _add_days(iso_date, n):
    dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt + timedelta(days=n)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_ring_graph(prefix, n_ring, base_amount, amount_step, day_step, base_date, n_decoys=2):
    """Builds one closed n_ring-cycle plus n_decoys unrelated single-edge
    decoy pairs. Amounts step DOWN by `amount_step` (negative) each edge, so
    distance-from-threshold grows monotonically with edge index, and
    timestamps step FORWARD by `day_step` days each edge -- both increasing
    together on purpose, so the edge that closes the cycle always has the
    highest combined filtration value and is always the cycle's birth. This
    is a deliberate property of this test-data generator for easy hand
    verification, not a general truth about real transaction graphs.
    """
    nodes, edges, labels = [], [], {}
    created = _add_days(base_date, -1)

    def acct(i):
        return f"{prefix}r{i:03d}"

    for i in range(n_ring):
        nodes.append({"id": acct(i), "type": "ring", "created_at": created})
        labels[acct(i)] = 1

    for i in range(n_ring):
        frm, to = acct(i), acct((i + 1) % n_ring)
        edges.append({
            "from": frm, "to": to,
            "timestamp": _add_days(base_date, i * day_step),
            "amount": base_amount + i * amount_step,
            "label": "ring",
        })

    for d in range(n_decoys):
        a, b = f"{prefix}d{d}a", f"{prefix}d{d}b"
        nodes += [
            {"id": a, "type": "decoy", "created_at": created},
            {"id": b, "type": "decoy", "created_at": created},
        ]
        edges.append({
            "from": a, "to": b,
            "timestamp": _add_days(base_date, d + 1),
            "amount": 400 + d * 100,
            "label": "decoy",
        })
        labels[a] = 0
        labels[b] = 0

    return {"nodes": nodes, "edges": edges}, labels


def build_synthetic_dataset():
    """Six small hand-built graphs -- 2 ring, 2 fan-out (one distributor,
    one collector), 2 clean -- with per-account fraud labels. Kept to two
    of each 'kind' on purpose: leave-one-graph-out CV (below) needs every
    fold's TRAINING set to still contain at least one example of each
    pattern, or the classifier can never learn a pattern it's never seen.
    Still synthetic and hand-built, not A's real generator -- same caveat
    every B-stage so far has carried.
    """
    dataset = []  # list of (graph_id, kind, graph_json, labels)

    g, l = _make_ring_graph("r1_", n_ring=6, base_amount=9850, amount_step=-150,
                             day_step=3, base_date="2026-03-01T00:00:00Z", n_decoys=2)
    dataset.append(("ring_1", "ring", g, l))

    g, l = _make_ring_graph("r2_", n_ring=4, base_amount=9900, amount_step=-100,
                             day_step=2, base_date="2026-05-01T00:00:00Z", n_decoys=1)
    dataset.append(("ring_2", "ring", g, l))

    # fanout_1: distributor -- one lump sum in, fanned out to 5 mules.
    fanout_1_nodes = [
        {"id": "f1_H", "type": "hub", "created_at": "2025-12-30T00:00:00Z"},
        {"id": "f1_src", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f1_r1", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f1_r2", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f1_r3", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f1_r4", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f1_r5", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f1_c1", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f1_c2", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
    ]
    fanout_1_edges = [
        {"from": "f1_src", "to": "f1_H", "timestamp": "2026-01-01T00:00:00Z", "amount": 50000, "label": "ring"},
        {"from": "f1_H", "to": "f1_r1", "timestamp": "2026-01-01T02:00:00Z", "amount": 9800, "label": "ring"},
        {"from": "f1_H", "to": "f1_r2", "timestamp": "2026-01-01T05:00:00Z", "amount": 9820, "label": "ring"},
        {"from": "f1_H", "to": "f1_r3", "timestamp": "2026-01-01T09:00:00Z", "amount": 9780, "label": "ring"},
        {"from": "f1_H", "to": "f1_r4", "timestamp": "2026-01-01T14:00:00Z", "amount": 9760, "label": "ring"},
        {"from": "f1_H", "to": "f1_r5", "timestamp": "2026-01-01T20:00:00Z", "amount": 9790, "label": "ring"},
        {"from": "f1_c1", "to": "f1_c2", "timestamp": "2026-01-10T00:00:00Z", "amount": 2000, "label": "decoy"},
    ]
    fanout_1_labels = {"f1_H": 1, "f1_r1": 1, "f1_r2": 1, "f1_r3": 1, "f1_r4": 1, "f1_r5": 1,
                        "f1_src": 0, "f1_c1": 0, "f1_c2": 0}
    dataset.append(("fanout_1", "fanout", {"nodes": fanout_1_nodes, "edges": fanout_1_edges}, fanout_1_labels))

    # fanout_2: collector -- 5 structured deposits in, one lump sum out.
    fanout_2_nodes = [
        {"id": "f2_H", "type": "hub", "created_at": "2026-03-30T00:00:00Z"},
        {"id": "f2_s1", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f2_s2", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f2_s3", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f2_s4", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f2_s5", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "f2_dest", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
    ]
    fanout_2_edges = [
        {"from": "f2_s1", "to": "f2_H", "timestamp": "2026-04-01T02:00:00Z", "amount": 9800, "label": "ring"},
        {"from": "f2_s2", "to": "f2_H", "timestamp": "2026-04-01T05:00:00Z", "amount": 9800, "label": "ring"},
        {"from": "f2_s3", "to": "f2_H", "timestamp": "2026-04-01T09:00:00Z", "amount": 9800, "label": "ring"},
        {"from": "f2_s4", "to": "f2_H", "timestamp": "2026-04-01T14:00:00Z", "amount": 9800, "label": "ring"},
        {"from": "f2_s5", "to": "f2_H", "timestamp": "2026-04-01T20:00:00Z", "amount": 9800, "label": "ring"},
        {"from": "f2_H", "to": "f2_dest", "timestamp": "2026-04-01T22:00:00Z", "amount": 49000, "label": "ring"},
    ]
    fanout_2_labels = {"f2_H": 1, "f2_s1": 1, "f2_s2": 1, "f2_s3": 1, "f2_s4": 1, "f2_s5": 1, "f2_dest": 0}
    dataset.append(("fanout_2", "fanout", {"nodes": fanout_2_nodes, "edges": fanout_2_edges}, fanout_2_labels))

    # clean_1 / clean_2: unremarkable accounts, no ring, no hub concentration.
    clean_1_nodes = [{"id": f"c1_{x}", "type": "decoy", "created_at": "2023-06-01T00:00:00Z"} for x in "ABCD"]
    clean_1_edges = [
        {"from": "c1_A", "to": "c1_B", "timestamp": "2026-02-01T00:00:00Z", "amount": 2000, "label": "decoy"},
        {"from": "c1_C", "to": "c1_D", "timestamp": "2026-02-11T00:00:00Z", "amount": 3500, "label": "decoy"},
    ]
    clean_1_labels = {f"c1_{x}": 0 for x in "ABCD"}
    dataset.append(("clean_1", "clean", {"nodes": clean_1_nodes, "edges": clean_1_edges}, clean_1_labels))

    clean_2_nodes = [{"id": f"c2_{x}", "type": "decoy", "created_at": "2023-08-01T00:00:00Z"} for x in "ABC"]
    clean_2_edges = [
        {"from": "c2_A", "to": "c2_B", "timestamp": "2026-02-01T00:00:00Z", "amount": 1500, "label": "decoy"},
        {"from": "c2_B", "to": "c2_C", "timestamp": "2026-03-03T00:00:00Z", "amount": 800, "label": "decoy"},
    ]
    clean_2_labels = {f"c2_{x}": 0 for x in "ABC"}
    dataset.append(("clean_2", "clean", {"nodes": clean_2_nodes, "edges": clean_2_edges}, clean_2_labels))

    return dataset


# ---------- build the fused feature table ----------

def build_feature_table(dataset):
    """One row per account: fused vector (diagram + B4), B4-only vector,
    label, graph_id, kind."""
    rows = []
    for graph_id, kind, graph_json, labels in dataset:
        diagram_result = graph_to_diagram_once(graph_json)
        dvec = diagram_to_vector(diagram_result)
        acct_features = extract_account_features(graph_json)
        for acct_id, label in labels.items():
            b4vec = b4_features_to_vector(acct_features[acct_id])
            rows.append({
                "account_id": acct_id,
                "graph_id": graph_id,
                "kind": kind,
                "label": label,
                "fused_vector": dvec + b4vec,
                "b4_vector": b4vec,
                "ring_persistence_score": diagram_result["ring_persistence_score"],
            })
    return rows


# ---------- leave-one-graph-out evaluation ----------

def _cluster_then_label(X_train, y_train, X_test):
    """Naive baseline: unsupervised KMeans(k=2) on B4-only features, then
    label each resulting cluster by the MAJORITY true label among its own
    TRAINING points (a legitimate, standard way to evaluate a clustering
    baseline out-of-sample -- the clustering itself never sees labels, only
    the after-the-fact interpretation does)."""
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    train_clusters = km.fit_predict(X_train)
    cluster_label = {}
    for c in set(train_clusters):
        members = [y_train[i] for i in range(len(y_train)) if train_clusters[i] == c]
        if members:
            cluster_label[c] = Counter(members).most_common(1)[0][0]
    # Fallback for the (essentially unreachable with n_init=10 on this data
    # size) case where predict() returns a cluster id with zero training
    # members -- majority overall label beats crashing.
    overall_majority = Counter(y_train).most_common(1)[0][0]
    test_clusters = km.predict(X_test)
    return [cluster_label.get(c, overall_majority) for c in test_clusters]


def evaluate(rows):
    """Leave-one-GRAPH-out CV (not leave-one-account-out): holding out a
    whole graph is the honest test, since accounts from the same graph
    share a diagram vector and are not independent of each other."""
    graph_ids = sorted(set(r["graph_id"] for r in rows))
    out = {"y_true": [], "y_pred_naive": [], "y_pred_fused": [], "y_prob_fused": [],
           "kind": [], "graph_id": [], "account_id": []}

    for test_graph in graph_ids:
        train_rows = [r for r in rows if r["graph_id"] != test_graph]
        test_rows = [r for r in rows if r["graph_id"] == test_graph]

        y_train = [r["label"] for r in train_rows]
        Xb4_train = [r["b4_vector"] for r in train_rows]
        Xb4_test = [r["b4_vector"] for r in test_rows]
        Xf_train = [r["fused_vector"] for r in train_rows]
        Xf_test = [r["fused_vector"] for r in test_rows]

        # Scalers fit on TRAIN only -- fitting on the full set first would
        # leak the held-out graph's distribution into training, a classic
        # and easy mistake to make with cross-validation + preprocessing.
        scaler_b4 = StandardScaler().fit(Xb4_train)
        Xb4_train_s, Xb4_test_s = scaler_b4.transform(Xb4_train), scaler_b4.transform(Xb4_test)

        scaler_f = StandardScaler().fit(Xf_train)
        Xf_train_s, Xf_test_s = scaler_f.transform(Xf_train), scaler_f.transform(Xf_test)

        naive_preds = _cluster_then_label(Xb4_train_s, y_train, Xb4_test_s)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xf_train_s, y_train)
        fused_preds = clf.predict(Xf_test_s)
        fused_probs = clf.predict_proba(Xf_test_s)[:, 1]

        for i, r in enumerate(test_rows):
            out["y_true"].append(r["label"])
            out["y_pred_naive"].append(naive_preds[i])
            out["y_pred_fused"].append(fused_preds[i])
            out["y_prob_fused"].append(fused_probs[i])
            out["kind"].append(r["kind"])
            out["graph_id"].append(r["graph_id"])
            out["account_id"].append(r["account_id"])

    return out


def _recall_subset(y_true, y_pred, mask):
    yt = [y for y, m in zip(y_true, mask) if m]
    yp = [y for y, m in zip(y_pred, mask) if m]
    if sum(yt) == 0:
        return None
    return round(recall_score(yt, yp, zero_division=0), 4)


if __name__ == "__main__":
    dataset = build_synthetic_dataset()
    rows = build_feature_table(dataset)

    print("=== B5 GRAPH-LEVEL DIAGRAM VECTORS ===")
    seen = set()
    for r in rows:
        if r["graph_id"] not in seen:
            seen.add(r["graph_id"])
            print(f"  {r['graph_id']:10s} ({r['kind']:7s}) ring_persistence_score={r['ring_persistence_score']}")

    print(f"\nTotal accounts: {len(rows)}  |  fraud={sum(r['label'] for r in rows)}  "
          f"normal={sum(1 - r['label'] for r in rows)}")

    print("\n=== LEAVE-ONE-GRAPH-OUT EVALUATION ===")
    results = evaluate(rows)
    yt = results["y_true"]
    yp_naive, yp_fused, yprob = results["y_pred_naive"], results["y_pred_fused"], results["y_prob_fused"]

    print("\nNaive clustering baseline (B4 features only, no topology):")
    print(f"  precision={precision_score(yt, yp_naive, zero_division=0):.4f}  "
          f"recall={recall_score(yt, yp_naive, zero_division=0):.4f}  "
          f"f1={f1_score(yt, yp_naive, zero_division=0):.4f}  "
          f"(no AUC -- hard clustering has no natural probability score)")

    print("\nFused classifier (topology + B4 features):")
    print(f"  precision={precision_score(yt, yp_fused, zero_division=0):.4f}  "
          f"recall={recall_score(yt, yp_fused, zero_division=0):.4f}  "
          f"f1={f1_score(yt, yp_fused, zero_division=0):.4f}  "
          f"auc={roc_auc_score(yt, yprob):.4f}")

    ring_mask = [k == "ring" for k in results["kind"]]
    fanout_mask = [k == "fanout" for k in results["kind"]]

    print("\nRecall specifically on RING-graph fraud accounts (the pattern only topology sees):")
    print(f"  naive={_recall_subset(yt, yp_naive, ring_mask)}   fused={_recall_subset(yt, yp_fused, ring_mask)}")
    print("Recall specifically on FANOUT-graph fraud accounts (the pattern B4 features alone can see):")
    print(f"  naive={_recall_subset(yt, yp_naive, fanout_mask)}   fused={_recall_subset(yt, yp_fused, fanout_mask)}")

    print("\n=== SIGNIFICANCE TESTS on ring_persistence_score ===")
    fraud_scores = [r["ring_persistence_score"] for r in rows if r["label"] == 1]
    normal_scores = [r["ring_persistence_score"] for r in rows if r["label"] == 0]
    u_stat, u_p = mannwhitneyu(fraud_scores, normal_scores, alternative="two-sided")
    print(f"Mann-Whitney U (fraud vs normal, all {len(rows)} accounts): U={u_stat:.2f}, p={u_p:.4f}")

    ring_scores = [r["ring_persistence_score"] for r in rows if r["kind"] == "ring"]
    fanout_scores = [r["ring_persistence_score"] for r in rows if r["kind"] == "fanout"]
    clean_scores = [r["ring_persistence_score"] for r in rows if r["kind"] == "clean"]
    h_stat, h_p = kruskal(ring_scores, fanout_scores, clean_scores)
    print(f"Kruskal-Wallis (ring vs fanout vs clean graphs): H={h_stat:.2f}, p={h_p:.6f}")

    print("\n=== HOW TO READ THIS ===")
    print("- Kruskal-Wallis SHOULD be extremely significant: every ring-graph account carries a")
    print("  nonzero ring_persistence_score and every fanout/clean-graph account carries exactly 0,")
    print("  by construction -- this just re-confirms B0-B3's premise at the dataset level.")
    print("- Mann-Whitney (fraud vs normal) is a harder test: fanout fraud accounts score 0 (topology")
    print("  is blind to them) and ring-graph DECOY accounts score >0 (the broadcast graph-level score")
    print("  doesn't know which specific nodes are the ring) -- don't be surprised if this one is")
    print("  weaker or not conventionally significant. That's the fusion argument, not a bug.")
    print("- Expect fused to beat naive on ring-graph recall (only fused has any topological signal),")
    print("  and both to do comparably on fanout-graph recall (B4 features alone already see it).")
    print("- CHECK: naive-baseline recall may be partly riding the time_compression_days sentinel")
    print("  confound (see module docstring) rather than genuinely topology-blind B4 signal -- sanity")
    print("  check cluster centroids / per-feature contribution before citing this comparison as-is.")
