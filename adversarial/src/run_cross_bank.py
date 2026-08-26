"""
run_cross_bank.py — Stage C3 entry point.

Runs the mocked 3-bank verification with default settings and writes the
result to disk in the exact Cross-Bank Verification schema, so D can read
it directly for the dashboard without needing to import Python.
"""
import json
import os

from .crossbank.simulate import simulate_cross_bank_verification

if __name__ == "__main__":
    result = simulate_cross_bank_verification()
    out_dir = "output/cross_bank"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bank_pairs.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2))
