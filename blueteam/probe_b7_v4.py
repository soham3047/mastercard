"""
Fixed B6 probe v5 -- auto-discovers module names instead of guessing.
Run from D:\mastercard_hack\vineyard\blueteam
    python probe_b7_v5.py
Paste the ENTIRE output back.
"""
import sys, os, importlib, inspect, glob

sys.path.insert(0, 'src')

print("="*70)
print("Files in src/")
print("="*70)
src_files = sorted(glob.glob(os.path.join('src', '*.py')))
for f in src_files:
    print(" ", f)

# ---- load b5_fusion (confirmed working already) ----
import b5_fusion
d = b5_fusion.SYNTHETIC_GRAPHS() if callable(getattr(b5_fusion, 'SYNTHETIC_GRAPHS', None)) else getattr(b5_fusion, 'SYNTHETIC_GRAPHS', None)
d = d or (b5_fusion.build_synthetic_dataset() if hasattr(b5_fusion, 'build_synthetic_dataset') else None)
name, kind, graph_json, labels = d[0]  # ring_1
print("\nUsing synthetic graph:", name, kind)

# ---- try to find and import every b*.py module, report what's inside ----
print("\n" + "="*70)
print("Scanning every src/b*.py module for relevant functions")
print("="*70)

modules = {}
for f in src_files:
    modname = os.path.splitext(os.path.basename(f))[0]
    if modname in ('__init__',):
        continue
    try:
        mod = importlib.import_module(modname)
        modules[modname] = mod
        funcs = [n for n, obj in inspect.getmembers(mod) if inspect.isfunction(obj) and obj.__module__ == modname]
        classes = [n for n, obj in inspect.getmembers(mod) if inspect.isclass(obj) and obj.__module__ == modname]
        print(f"\n[{modname}]")
        print("  functions:", funcs)
        print("  classes:  ", classes)
    except Exception as e:
        print(f"\n[{modname}] FAILED TO IMPORT: {e!r}")

# ---- try to find the explain-style function specifically ----
print("\n" + "="*70)
print("Looking for an 'explain' style function across all modules")
print("="*70)
explain_fn = None
explain_owner = None
for modname, mod in modules.items():
    for n, obj in inspect.getmembers(mod):
        if inspect.isfunction(obj) and obj.__module__ == modname and 'explain' in n.lower():
            print(f"  candidate: {modname}.{n}  sig={inspect.signature(obj)}")
            explain_fn = obj
            explain_owner = f"{modname}.{n}"

if explain_fn:
    print(f"\nTrying {explain_owner}(graph_json) ...")
    try:
        out = explain_fn(graph_json)
        print("SUCCESS. Return type:", type(out))
        print("Return (truncated):", repr(out)[:1200])
    except Exception as e:
        print("FAILED with graph_json alone:", repr(e))
else:
    print("  No function with 'explain' in its name found in any module.")

print("\n" + "="*70)
print("DONE -- paste everything above back")
print("="*70)