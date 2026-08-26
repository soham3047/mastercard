"""
VINEYARD — Blue Team, Stage B4: Graph/account feature extraction (feature-fusion side).

IMPLEMENTATION NOTE — read before presenting this stage:
B0-B3 built the TOPOLOGICAL half of detection: ring/layering attacks show up
as a long-lived H1 feature because they're literal cycles. Fan-out / mule
networks are the other named attack pattern in the project (see root
README), and they are topologically TRIVIAL — a star graph has no cycles,
so persistent homology sees nothing there BY DESIGN, not by bug. This stage
is the other half: plain graph/account features that catch exactly what
B0-B3 structurally cannot see.

This module deliberately does NOT combine these into a single risk score —
that fusion (concatenate with the topological features, train a classifier,
benchmark against a naive baseline) is explicitly B5's job per the README.
B4's only job is to expose clean, per-account numbers.
"""
from collections import defaultdict
from datetime import datetime, timezone
import statistics

DEFAULT_BURST_WINDOW_DAYS = 1.0


def _parse_timestamp(t):
    """Accepts either a raw numeric timestamp (what B0-B3's synthetic tests
    used) or an ISO8601 string (the contract's actual format, and the first
    time this project's code parses `created_at`/`timestamp` for real). All
    values, of either input form, resolve to a float in DAYS — consistent
    units for burst windows, time-compression, and account age.

    Gotcha worth documenting the same way B0 flagged its GUDHI gotcha: a
    naive (no offset/'Z') ISO8601 string handed to `.timestamp()` is
    silently interpreted in the MACHINE'S LOCAL timezone, not UTC — meaning
    the exact same input graph could score differently on different judges'
    laptops during the live demo. Naive strings are explicitly forced to
    UTC here instead of trusting the system default.
    """
    if isinstance(t, (int, float)):
        return float(t)
    dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0


def _safe_div(a, b, default=0.0):
    return a / b if b else default


def _max_events_in_window(times, window_days):
    """Max number of events falling inside ANY sliding window of length
    `window_days`, via a standard two-pointer scan over sorted timestamps.
    O(n) after the sort. Returns 0 for an empty list."""
    times = sorted(times)
    max_count = 0
    left = 0
    for right in range(len(times)):
        while times[right] - times[left] > window_days:
            left += 1
        max_count = max(max_count, right - left + 1)
    return max_count


def _avg_gap_days(times):
    """Average gap between consecutive events, in days. None if there are
    fewer than 2 events (no gap is defined)."""
    times = sorted(times)
    if len(times) < 2:
        return None
    gaps = [b - a for a, b in zip(times, times[1:])]
    return sum(gaps) / len(gaps)


def extract_account_features(graph_json, burst_window_days=DEFAULT_BURST_WINDOW_DAYS):
    """
    Takes a graph matching the shared transaction-graph schema and returns
    one feature dict per account:

    {
      "acct_001": {
        "in_degree": int, "out_degree": int,
        "fan_out_ratio": float,        # out_degree / in_degree -- >>1 = distributor
                                        # (fans money OUT to many), <<1 = collector
                                        # (funnels money IN from many). Both are the
                                        # "star" shapes B0-B3 can't see.
        "in_burst_count": int,         # max incoming edges in any `burst_window_days` window
        "out_burst_count": int,        # same, outgoing
        "time_compression_days": float | None,  # avg gap between this account's
                                                  # own transactions (both directions).
                                                  # Small = rapid pass-through mule
                                                  # behavior. None if <2 transactions.
        "account_age_days": float | None,  # (this account's OWN first transaction) -
                                            # created_at -- i.e. how long it sat dormant
                                            # before it was first used, NOT its age as
                                            # of "today". Small/negative = synthetic
                                            # identity created and used almost
                                            # immediately. None if no created_at or no
                                            # transactions at all.
        "amount_mean_in": float | None, "amount_mean_out": float | None,
        "amount_variance_in": float, "amount_variance_out": float,
                                        # population variance, split by direction on
                                        # purpose -- a mule that collects one lump sum
                                        # and redistributes it in near-identical
                                        # threshold-hugging pieces has LOW out-variance
                                        # and unrelated in-variance; blending the two
                                        # directions into one number would wash that
                                        # signal out.
      },
      ...
    }

    Every account that appears anywhere (in `nodes`, or as an edge endpoint
    even if missing from `nodes`) gets an entry; fields that need a
    `created_at` we don't have simply come back as None rather than raising.
    """
    nodes = graph_json.get("nodes", [])
    edges = graph_json.get("edges", [])

    node_created = {n["id"]: n.get("created_at") for n in nodes}
    all_ids = set(node_created) | {e["from"] for e in edges} | {e["to"] for e in edges}

    incoming = defaultdict(list)  # node_id -> [(t_days, amount), ...]
    outgoing = defaultdict(list)
    for e in edges:
        t = _parse_timestamp(e["timestamp"])
        amt = e.get("amount")
        outgoing[e["from"]].append((t, amt))
        incoming[e["to"]].append((t, amt))

    features = {}
    for node_id in all_ids:
        in_edges = incoming.get(node_id, [])
        out_edges = outgoing.get(node_id, [])
        in_degree, out_degree = len(in_edges), len(out_edges)

        in_amounts = [a for _, a in in_edges if a is not None]
        out_amounts = [a for _, a in out_edges if a is not None]

        fan_out_ratio = _safe_div(out_degree, in_degree, default=float(out_degree))

        in_burst_count = _max_events_in_window([t for t, _ in in_edges], burst_window_days)
        out_burst_count = _max_events_in_window([t for t, _ in out_edges], burst_window_days)

        own_times = [t for t, _ in in_edges] + [t for t, _ in out_edges]
        time_compression_days = _avg_gap_days(own_times)

        created_at_raw = node_created.get(node_id)
        if created_at_raw and own_times:
            account_age_days = min(own_times) - _parse_timestamp(created_at_raw)
        else:
            account_age_days = None

        features[node_id] = {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "fan_out_ratio": round(fan_out_ratio, 4),
            "in_burst_count": in_burst_count,
            "out_burst_count": out_burst_count,
            "time_compression_days": round(time_compression_days, 4) if time_compression_days is not None else None,
            "account_age_days": round(account_age_days, 4) if account_age_days is not None else None,
            "amount_mean_in": round(statistics.mean(in_amounts), 2) if in_amounts else None,
            "amount_mean_out": round(statistics.mean(out_amounts), 2) if out_amounts else None,
            "amount_variance_in": round(statistics.pvariance(in_amounts), 4) if in_amounts else 0.0,
            "amount_variance_out": round(statistics.pvariance(out_amounts), 4) if out_amounts else 0.0,
        }

    return features


if __name__ == "__main__":
    # acct_H: classic fan-out placement -- ONE lump sum in, then rapidly
    # split into 5 near-identical, just-under-threshold transfers out, from
    # an account created only 2 days before it first activated.
    # acct_N: a normal account -- few transactions, spread over months,
    # varied amounts, long-established before it ever transacted.
    graph = {
        "nodes": [
            {"id": "acct_H",   "type": "hub",   "created_at": "2025-12-30T00:00:00Z"},
            {"id": "acct_099", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "acct_201", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "acct_202", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "acct_203", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "acct_204", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "acct_205", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "acct_N",   "type": "decoy", "created_at": "2024-11-01T00:00:00Z"},
            {"id": "acct_301", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "acct_302", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "acct_303", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
        ],
        "edges": [
            {"from": "acct_099", "to": "acct_H",   "timestamp": "2026-01-01T00:00:00Z", "amount": 50000, "label": "ring"},
            {"from": "acct_H",   "to": "acct_201", "timestamp": "2026-01-01T02:00:00Z", "amount": 9800,  "label": "ring"},
            {"from": "acct_H",   "to": "acct_202", "timestamp": "2026-01-01T05:00:00Z", "amount": 9820,  "label": "ring"},
            {"from": "acct_H",   "to": "acct_203", "timestamp": "2026-01-01T09:00:00Z", "amount": 9780,  "label": "ring"},
            {"from": "acct_H",   "to": "acct_204", "timestamp": "2026-01-01T14:00:00Z", "amount": 9760,  "label": "ring"},
            {"from": "acct_H",   "to": "acct_205", "timestamp": "2026-01-01T20:00:00Z", "amount": 9790,  "label": "ring"},
            {"from": "acct_301", "to": "acct_N",   "timestamp": "2026-01-01T00:00:00Z", "amount": 5000,  "label": "decoy"},
            {"from": "acct_N",   "to": "acct_302", "timestamp": "2026-01-25T00:00:00Z", "amount": 300,   "label": "decoy"},
            {"from": "acct_303", "to": "acct_N",   "timestamp": "2026-02-20T00:00:00Z", "amount": 1200,  "label": "decoy"},
        ],
    }

    features = extract_account_features(graph)
    h, n = features["acct_H"], features["acct_N"]

    print("=== B4 ACCOUNT FEATURES TEST ===")
    for label, f in [("acct_H (fan-out hub)", h), ("acct_N (normal)", n)]:
        print(f"\n{label}:")
        for k, v in f.items():
            print(f"  {k}: {v}")

    checks = {
        "fan_out_ratio (H > N)": h["fan_out_ratio"] > n["fan_out_ratio"],
        "out_burst_count (H >= 5)": h["out_burst_count"] >= 5,
        "time_compression_days (H < N)": h["time_compression_days"] < n["time_compression_days"],
        "account_age_days (H < N)": h["account_age_days"] < n["account_age_days"],
        "amount_variance_out (H << N's in-variance)": h["amount_variance_out"] < n["amount_variance_in"],
    }

    print("\n=== CHECKS ===")
    all_pass = True
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_pass = all_pass and ok

    if all_pass:
        print("\nPASS: acct_H is correctly flagged as the more suspicious account on every "
              "axis B0-B3 structurally cannot see -- despite being part of a pure star/fan-out "
              "shape with zero cycles, so it would score a flat 0 on ring_persistence_score.")
    else:
        print("\nFAIL: check the printed numbers above against the expected values.")
