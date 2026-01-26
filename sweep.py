# sweep.py
import pandas as pd
import numpy as np

# ----------------------------
# Load + merge
# ----------------------------
BASE_PARQUET = "data/processed/toy_noisy_rows.parquet"
POST_CSV     = "data/outputs/sieve_results_toy_noisy_rows.csv"
OUT_SUMMARY  = "data/outputs/sweep_summary.csv"

base = pd.read_parquet(BASE_PARQUET)[["row_id", "noise_label"]].rename(
    columns={"noise_label": "noise_label_gt"}
)
post = pd.read_csv(POST_CSV)

m = post.merge(base, on="row_id", how="left")
m["dirty"] = m["noise_label_gt"].ne("clean")

# ----------------------------
# Safety / type coercion
# ----------------------------
REQUIRED = ["sim_score", "gap", "category", "row_id", "noise_label_gt", "dirty"]
missing = [c for c in REQUIRED if c not in m.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

m["sim_score"] = pd.to_numeric(m["sim_score"], errors="coerce")
m["gap"] = pd.to_numeric(m["gap"], errors="coerce")
m["category"] = m["category"].fillna("UNKNOWN")

# Drop rows where we can't score (optional; comment out if you want to keep)
m = m.dropna(subset=["sim_score", "gap"]).reset_index(drop=True)

# ----------------------------
# Metrics helpers
# ----------------------------
def pr_metrics(flagged: pd.Series, dirty: pd.Series):
    flagged = flagged.fillna(False)
    dirty = dirty.fillna(False)

    tp = int((flagged &  dirty).sum())
    fp = int((flagged & ~dirty).sum())
    fn = int((~flagged &  dirty).sum())

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return tp, fp, fn, prec, rec, f1

def print_row(tag: str, tp, fp, fn, prec, rec, f1):
    print(f"{tag:42s}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  TP={tp} FP={fp} FN={fn}")

def add_result(rows, name: str, flagged: pd.Series):
    tp, fp, fn, prec, rec, f1 = pr_metrics(flagged, m["dirty"])
    rows.append({"rule": name, "TP": tp, "FP": fp, "FN": fn, "prec": prec, "rec": rec, "f1": f1})
    return tp, fp, fn, prec, rec, f1

rows = []

# ----------------------------
# A) Global threshold sweep: OR vs AND
# ----------------------------
sim_grid = [0.25, 0.30, 0.35, 0.40]
gap_grid = [0.005, 0.01, 0.02, 0.03, 0.05]

print("\n=== GLOBAL THRESHOLDS (raw) ===")
for sim_t in sim_grid:
    for gap_t in gap_grid:
        flag_or  = (m["sim_score"] < sim_t) | (m["gap"] < gap_t)
        flag_and = (m["sim_score"] < sim_t) & (m["gap"] < gap_t)

        tp, fp, fn, prec, rec, f1 = add_result(rows, f"RAW_OR  sim<{sim_t:.2f} gap<{gap_t:.3f}", flag_or)
        print_row(f"OR  sim<{sim_t:.2f}  gap<{gap_t:.3f}", tp, fp, fn, prec, rec, f1)

        tp, fp, fn, prec, rec, f1 = add_result(rows, f"RAW_AND sim<{sim_t:.2f} gap<{gap_t:.3f}", flag_and)
        print_row(f"AND sim<{sim_t:.2f}  gap<{gap_t:.3f}", tp, fp, fn, prec, rec, f1)

# ----------------------------
# B) Per-category normalization (percentile ranks)
# ----------------------------
print("\n=== PER-CATEGORY (percentile) ===")
m["sim_pct"] = m.groupby("category")["sim_score"].rank(pct=True, method="average")
m["gap_pct"] = m.groupby("category")["gap"].rank(pct=True, method="average")

pct_grid = [0.01, 0.02, 0.05, 0.10]

for sim_p in pct_grid:
    for gap_p in pct_grid:
        flag_or  = (m["sim_pct"] < sim_p) | (m["gap_pct"] < gap_p)
        flag_and = (m["sim_pct"] < sim_p) & (m["gap_pct"] < gap_p)

        tp, fp, fn, prec, rec, f1 = add_result(rows, f"PCT_OR  sim<{sim_p:.2f} gap<{gap_p:.2f}", flag_or)
        print_row(f"OR  sim_pct<{sim_p:.2f}  gap_pct<{gap_p:.2f}", tp, fp, fn, prec, rec, f1)

        tp, fp, fn, prec, rec, f1 = add_result(rows, f"PCT_AND sim<{sim_p:.2f} gap<{gap_p:.2f}", flag_and)
        print_row(f"AND sim_pct<{sim_p:.2f}  gap_pct<{gap_p:.2f}", tp, fp, fn, prec, rec, f1)

# ----------------------------
# C) Budget-aware Top-K flagging
# ----------------------------
print("\n=== TOP-K (budgeted review) ===")
m["risk"] = (1 - m["sim_pct"]) + (1 - m["gap_pct"])

for K in [25, 50, 75, 100, 150, 200]:
    flagged = m["risk"].rank(ascending=False, method="first") <= K
    tp, fp, fn, prec, rec, f1 = add_result(rows, f"TOPK risk K={K}", flagged)
    print_row(f"Top-K risk  K={K}", tp, fp, fn, prec, rec, f1)

# ----------------------------
# Best by F1 + save all results
# ----------------------------
res = pd.DataFrame(rows).sort_values("f1", ascending=False)
print("\n=== BEST BY F1 (top 15) ===")
print(res.head(15).to_string(index=False))

res.to_csv(OUT_SUMMARY, index=False)
print(f"\nSaved: {OUT_SUMMARY}")

# ----------------------------
# OPTIONAL: quick FN/FP breakdown for a chosen rule
# (Edit thresholds below)
# ----------------------------
CHOSEN_SIM = 0.30
CHOSEN_GAP = 0.05
chosen_flagged = (m["sim_score"] < CHOSEN_SIM) & (m["gap"] < CHOSEN_GAP)

fn = m[(~chosen_flagged) & (m["dirty"])]
fp = m[( chosen_flagged) & (~m["dirty"])]

print("\n=== DEBUG (chosen rule) ===")
print(f"Chosen: sim<{CHOSEN_SIM} AND gap<{CHOSEN_GAP}")
print("FN by category:\n", fn["category"].value_counts().head(10).to_string())
print("FP by category:\n", fp["category"].value_counts().head(10).to_string())

print("\nFN samples:")
print(fn[["row_id","category","noise_label_gt","sim_score","gap"]].head(15).to_string(index=False))
