import inspect
import sys

sys.path.insert(0, "src")

def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

nodes = [
    {"id": "a", "type": "ring", "created_at": "2024-01-01T00:00:00Z"},
    {"id": "b", "type": "ring", "created_at": "2024-01-01T00:00:00Z"},
    {"id": "c", "type": "ring", "created_at": "2024-01-01T00:00:00Z"},
    {"id": "d", "type": "decoy", "created_at": "2024-01-01T00:00:00Z"},
]

edges_numeric = [
    {"from": "a", "to": "b", "timestamp": 0, "amount": 9800.0, "label": "ring"},
    {"from": "b", "to": "c", "timestamp": 1, "amount": 9700.0, "label": "ring"},
    {"from": "c", "to": "a", "timestamp": 2, "amount": 9600.0, "label": "ring"},
    {"from": "d", "to": "a", "timestamp": 0, "amount": 50.0,   "label": "decoy"},
]

edges_iso = [
    {"from": "a", "to": "b", "timestamp": "2024-01-01T00:00:00Z", "amount": 9800.0, "label": "ring"},
    {"from": "b", "to": "c", "timestamp": "2024-01-02T00:00:00Z", "amount": 9700.0, "label": "ring"},
    {"from": "c", "to": "a", "timestamp": "2024-01-03T00:00:00Z", "amount": 9600.0, "label": "ring"},
    {"from": "d", "to": "a", "timestamp": "2024-01-01T00:00:00Z", "amount": 50.0,   "label": "decoy"},
]

graph_json_iso = {"nodes": nodes, "edges": edges_iso}
graph_json_numeric = {"nodes": nodes, "edges": edges_numeric}

# ---------------------------------------------------------------- B4 ----
hr("B4: extract_account_features(graph_json)")
import b4_account_features as b4

feats_iso = b4.extract_account_features(graph_json_iso)
print("With ISO timestamps -- type:", type(feats_iso))
print("Return (repr, truncated):", repr(feats_iso)[:2000])

try:
    feats_num = b4.extract_account_features(graph_json_numeric)
    print("\nAlso works with numeric timestamps:", repr(feats_num)[:500])
except Exception as e:
    print("\nFAILS with numeric timestamps:", repr(e))

# ---------------------------------------------------------------- B6 ----
hr("B6: explain_graph signature + call")
import b6_explainability as b6

print("REPORTING_THRESHOLD:", b6.REPORTING_THRESHOLD)
sig = inspect.signature(b6.explain_graph)
print("signature:", sig)

# try both calling conventions based on signature param count
params = list(sig.parameters.keys())
print("param names:", params)

try:
    if len(params) == 1:
        expl = b6.explain_graph(graph_json_iso)
    else:
        expl = b6.explain_graph(nodes, edges_iso)
    print("Return type:", type(expl))
    print("Return (repr, truncated):", repr(expl)[:2000])
except Exception as e:
    print("Call with ISO failed:", repr(e))
    try:
        if len(params) == 1:
            expl = b6.explain_graph(graph_json_numeric)
        else:
            expl = b6.explain_graph(nodes, edges_numeric)
        print("Numeric version worked. Return:", repr(expl)[:2000])
    except Exception as e2:
        print("Numeric version also failed:", repr(e2))

hr("DONE — paste everything above back")