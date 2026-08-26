"""
One-shot diagnostic probe for B7 integration.
Run this from the same place you'd run b7_score_batch.py:
    python probe_b7.py
Paste the FULL output back.
"""
import inspect
import json
import sys

sys.path.insert(0, "src")  # adjust if your modules aren't under src/

def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# Tiny fake graph matching the README's input schema exactly.
nodes = [
    {"id": "a", "type": "ring", "created_at": "2024-01-01T00:00:00Z"},
    {"id": "b", "type": "ring", "created_at": "2024-01-01T00:00:00Z"},
    {"id": "c", "type": "ring", "created_at": "2024-01-01T00:00:00Z"},
    {"id": "d", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
]
edges = [
    {"from": "a", "to": "b", "timestamp": "2024-01-01T00:00:00Z", "amount": 9800.0, "label": "ring"},
    {"from": "b", "to": "c", "timestamp": "2024-01-02T00:00:00Z", "amount": 9700.0, "label": "ring"},
    {"from": "c", "to": "a", "timestamp": "2024-01-03T00:00:00Z", "amount": 9600.0, "label": "ring"},
    {"from": "d", "to": "a", "timestamp": "2024-01-01T00:00:00Z", "amount": 50.0, "label": "decoy"},
]

# ---------------------------------------------------------------- B3 ----
hr("B3: b3_vineyards")
import b3_vineyards as b3

print("DEFAULT_THRESHOLD:", getattr(b3, "DEFAULT_THRESHOLD", "MISSING"))
print("ESSENTIAL_SCORE_BASE:", getattr(b3, "ESSENTIAL_SCORE_BASE", "MISSING"))
print("TIME_SCALE_DAYS:", getattr(b3, "TIME_SCALE_DAYS", "MISSING"))

print("\nPersistentGraph.__init__ signature:", inspect.signature(b3.PersistentGraph.__init__))
print("PersistentGraph.add_edges signature:", inspect.signature(b3.PersistentGraph.add_edges))
print("update_diagram signature:", inspect.signature(b3.update_diagram))

pg = b3.PersistentGraph()
print("\nFresh PersistentGraph attrs before add_edges:")
for attr in ("alpha", "beta", "threshold", "t0", "rounds_seen", "edges_seen", "current_diagram"):
    print(f"  {attr} =", repr(getattr(pg, attr, "MISSING")))

result = pg.add_edges(edges)
print("\nadd_edges(edges) return type:", type(result))
if isinstance(result, dict):
    print("add_edges(edges) return keys:", list(result.keys()))
    print("add_edges(edges) full return (repr, truncated):", repr(result)[:1500])
else:
    print("add_edges(edges) full return (repr, truncated):", repr(result)[:1500])

print("\nPersistentGraph attrs AFTER add_edges:")
for attr in ("alpha", "beta", "threshold", "t0", "rounds_seen", "edges_seen", "current_diagram"):
    print(f"  {attr} =", repr(getattr(pg, attr, "MISSING"))[:500])

# try calling update_diagram fresh (state=None case)
try:
    ud_result = b3.update_diagram(None, edges)
    print("\nupdate_diagram(None, edges) return type:", type(ud_result))
    print("update_diagram(None, edges) repr (truncated):", repr(ud_result)[:1000])
except Exception as e:
    print("\nupdate_diagram(None, edges) RAISED:", repr(e))

# ---------------------------------------------------------------- B4 ----
hr("B4: b4_account_features")
import b4_account_features as b4

print("extract_account_features signature:", inspect.signature(b4.extract_account_features))
feats = b4.extract_account_features(nodes, edges)
print("Return type:", type(feats))
print("Return (repr, truncated):", repr(feats)[:1500])

# ---------------------------------------------------------------- B6 ----
hr("B6: b6_explainability")
import b6_explainability as b6

print("REPORTING_THRESHOLD:", getattr(b6, "REPORTING_THRESHOLD", "MISSING"))
print("explain_graph signature:", inspect.signature(b6.explain_graph))
expl = b6.explain_graph(nodes, edges)
print("Return type:", type(expl))
print("Return (repr, truncated):", repr(expl)[:1500])

hr("DONE — paste everything above back")