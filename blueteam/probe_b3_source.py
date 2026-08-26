import sys, inspect
sys.path.insert(0, "src")
import b3_vineyards as b3

def hr(t):
    print("\n" + "=" * 70); print(t); print("=" * 70)

hr("source: _time_norm")
print(inspect.getsource(b3._time_norm))

hr("source: PersistentGraph.add_edges")
print(inspect.getsource(b3.PersistentGraph.add_edges))

hr("source: PersistentGraph.__init__")
print(inspect.getsource(b3.PersistentGraph.__init__))

hr("full module source (fallback, in case above missed helpers)")
print(inspect.getsource(b3))
