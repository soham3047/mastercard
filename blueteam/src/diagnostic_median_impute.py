# diagnostic_median_impute.py — temporary, not part of the pipeline
import statistics
from b5_fusion import (build_synthetic_dataset, build_feature_table, evaluate,
                        _recall_subset, B4_VECTOR_NAMES)

dataset = build_synthetic_dataset()
rows = build_feature_table(dataset)

# index 5 in b4_vector is time_compression_days (see B4_VECTOR_NAMES order)
COMPRESSION_IDX = B4_VECTOR_NAMES.index("time_compression_days")
real_vals = [r["b4_vector"][COMPRESSION_IDX] for r in rows if r["b4_vector"][COMPRESSION_IDX] != 9999.0]
median_val = statistics.median(real_vals)

for r in rows:
    if r["b4_vector"][COMPRESSION_IDX] == 9999.0:
        r["b4_vector"][COMPRESSION_IDX] = median_val
        r["fused_vector"][B4_VECTOR_NAMES.index("time_compression_days") + 5] = median_val
        # +5 because fused_vector = 5 diagram features + 11 b4 features, in that order

results = evaluate(rows)
yt, yp_naive = results["y_true"], results["y_pred_naive"]
ring_mask = [k == "ring" for k in results["kind"]]
print(f"Median (real) value used: {median_val}")
print(f"Naive ring recall WITH median-impute: {_recall_subset(yt, yp_naive, ring_mask)}")
print("Compare to 1.0 from the sentinel version — if this drops, the confound was real.")