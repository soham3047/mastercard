"""
VINEYARD — Blue Team, Stage B0: Sanity check.

Hand-built (hardcoded) 5-node ring + a few non-cyclic decoy transactions.
Goal: confirm the ring shows up as a long-lived H1 (1-dimensional hole) in
a time-based filtration, and that the decoys (no closed cycle) do NOT
produce an H1 feature. Does NOT consume Person A's generator — per the
B0 ground rule, this stage is deliberately hardcoded and self-contained.
"""
import gudhi

# --- Hardcoded input: matches the shared transaction-graph schema ---
# Ring: acct_001 -> acct_002 -> acct_003 -> acct_004 -> acct_005 -> acct_001
#       time-respecting hops, closes the cycle at t=4.0
ring_edges = [
    ("acct_001", "acct_002", 0.0),
    ("acct_002", "acct_003", 1.0),
    ("acct_003", "acct_004", 2.0),
    ("acct_004", "acct_005", 3.0),
    ("acct_005", "acct_001", 4.0),  # closing edge — this is where the H1 feature is born
]

# Decoys: two separate transaction chains, no closed cycle -> topologically trivial
decoy_edges = [
    ("acct_006", "acct_007", 0.5),
    ("acct_007", "acct_008", 1.5),
    ("acct_009", "acct_010", 2.5),
]

all_edges = ring_edges + decoy_edges
node_ids = sorted({n for e in all_edges for n in (e[0], e[1])})
node_index = {n: i for i, n in enumerate(node_ids)}

# --- Build filtered simplicial complex: vertices at t=0, edges at their timestamp ---
st = gudhi.SimplexTree()
for n in node_ids:
    st.insert([node_index[n]], filtration=0.0)
for u, v, t in all_edges:
    st.insert([node_index[u], node_index[v]], filtration=t)

# persistence_dim_max=True is required here: our complex's top dimension is 1
# (edges only, no filled triangles), so H1 IS the top-dimension homology, and
# GUDHI suppresses top-dimension classes by default unless this is passed
# directly to persistence() (compute_persistence()'s own flag doesn't carry
# over — this is a real GUDHI gotcha, not a modeling choice).
# min_persistence=-1 keeps infinite-death (essential) classes in the output.
diagram = st.persistence(persistence_dim_max=True, min_persistence=-1)

h1_features = [(birth, death) for (dim, (birth, death)) in diagram if dim == 1]

print("=== B0 SANITY CHECK ===")
print(f"Nodes: {len(node_ids)}  Edges: {len(all_edges)}")
print(f"H1 features found: {len(h1_features)}")
for birth, death in h1_features:
    lifetime = "inf" if death == float("inf") else round(death - birth, 3)
    print(f"  birth={birth}, death={death}, lifetime={lifetime}")

if len(h1_features) == 1:
    birth, death = h1_features[0]
    print(f"\nPASS: exactly one H1 feature, born at t={birth} "
          f"(the ring's closing edge acct_005->acct_001), "
          f"lifetime={'infinite' if death == float('inf') else death - birth} "
          f"(no chord ever closes it — the decoy chains contribute zero H1 features).")
else:
    print("\nFAIL: expected exactly one H1 feature from the ring, got a different count.")
