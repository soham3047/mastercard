"""
Probe v2 — works around B3's numeric-timestamp requirement so we can
actually reach B4 and B6 and see their real return shapes.
"""
import inspect
import sys

sys.path.insert(0, "src")

def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ---- Two edge sets: one numeric (for B3), one ISO8601 (for B4/B6/README shape) ----
nodes = [
    {"id": "a", "type": "ring", "created_at": "2024-01-01T00:00:00Z"},
    {"id": "b", "type": "ring", "created_at": "2024-01-01T00:00:00Z"},
    {"id": "c", "type": "ring", "created_at": "2024-01-01T00:00:00Z"},
    {"id": "d", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
]

edges_numeric = [
    {"from": "a", "to": "b", "timestamp": 0,  "amount": 9800.0, "label": "ring"},
    {"from": "b", "to": "c", "timestamp": 1,  "amount": 9700.0, "label": "ring"},
    {"from": "c", "to": "a", "timestamp": 2,  "amount": 9600.0, "label": "ring"},
    {"from": "d", "to": "a", "timestamp": 0,  "amount": 50.0,   "label": "decoy"},
]

edges_iso = [
    {"from": "a", "to": "b", "timestamp": "2024-01-01T00:00:00Z", "amount": 9800.0, "label": "ring"},
    {"from": "b", "to": "c", "timestamp": "2024-01-02T00:00:00Z", "amount": 9700.0, "label": "ring"},
    {"from": "c", "to": "a", "timestamp": "2024-01-03T00:00:00Z", "amount": 9600.0, "label": "ring"},
    {"from": "d", "to": "a", "timestamp": "2024-01-01T00:00:00Z", "amount": 50.0,   "label": "decoy"},
]

# ---------------------------------------------------------------- B3 ----
hr("B3: add_edges with NUMERIC timestamps (workaround)")
import b3_vineyards as b3

pg = b3.PersistentGraph()
result = pg.add_edges(edges_numeric)
print("add_edges() return type:", type(result))
print("add_edges() return:", result)

hr("B3: update_diagram(None, edges_numeric) return shape")
ud_result = b3.update_diagram(None, edges_numeric)
print("type:", type(ud_result))
print("len:", len(ud_result) if hasattr(ud_result, "__len__") else "n/a")
print("element 0 type:", type(ud_result[0]))
print("element 0 value:", ud_result[0])
print("element 1 type:", type(ud_result[1]))

# ---------------------------------------------------------------- B4 ----
hr("B4: extract_account_features")
import b4_account_features as b4

print("signature:", inspect.signature(b4.extract_account_features))
feats = b4.extract_account_features(nodes, edges_iso)
print("Return type:", type(feats))
print("Return (repr, truncated):", repr(feats)[:2000])

# also try with numeric timestamps in case B4 also wants numeric
try:
    feats2 = b4.extract_account_features(nodes, edges_numeric)
    print("\nAlso works with numeric timestamps:", repr(feats2)[:500])
except Exception as e:
    print("\nFAILS with numeric timestamps (expected if it needs ISO):", repr(e))

# ---------------------------------------------------------------- B6 ----
hr("B6: explain_graph")
import b6_explainability as b6

print("REPORTING_THRESHOLD:", b6.REPORTING_THRESHOLD)
print("signature:", inspect.signature(b6.explain_graph))
expl = b6.explain_graph(nodes, edges_iso)
print("Return type:", type(expl))
print("Return (repr, truncated):", repr(expl)[:2000])

try:
    expl2 = b6.explain_graph(nodes, edges_numeric)
    print("\nAlso works with numeric timestamps:", repr(expl2)[:500])
except Exception as e:
    print("\nFAILS with numeric timestamps (expected if it needs ISO):", repr(e))

hr("DONE — paste everything above back")