import pandas as pd
import numpy as np

BASE_PARQUET = "data/processed/toy_noisy_rows.parquet"
POST_CSV     = "data/outputs/sieve_results_toy_noisy_rows.csv"
OUT_THRESH   = "data/outputs/per_category_thresholds_constrained.csv"

# ----------------------------
# Load + merge
# ----------------------------
base = (
    pd.read_parquet(BASE_PARQUET)[["row_id", "noise_label"]]
    .rename(columns={"noise_label": "noise_label_gt"})
)
post = pd.read_csv(POST_CSV)

m = post.merge(base, on="row_id", how="left")
m["dirty"] = m["noise_label_gt"].ne("clean")

for col in ["sim_score", "gap", "category"]:
    if col not in m.columns:
        raise KeyError(f"Missing required column: {col}")

m["sim_score"] = pd.to_numeric(m["sim_score"], errors="coerce")
m["gap"] = pd.to_numeric(m["gap"], errors="coerce")
m["category"] = m["category"].fillna("UNKNOWN")
m = m.dropna(subset=["sim_score", "gap"]).reset_index(drop=True)

# Treat negative gap as "very suspicious"
m["gap_fixed"] = m["gap"].clip(lower=0)

# ----------------------------
# Metrics
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

# ----------------------------
# Search space
# ----------------------------
sim_grid = [0.25, 0.27, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]
gap_grid = [0.005, 0.01, 0.02, 0.03, 0.05]

# ----------------------------
# Choose your objective here
# ----------------------------
MODE = "max_recall_with_min_precision"   # or: "max_f1", "min_fp_with_min_recall"
MIN_PREC = 0.80
MIN_REC  = 0.80

rows = []

for cat, mc in m.groupby("category"):
    best = None

    for sim_t in sim_grid:
        for gap_t in gap_grid:
            flagged = (mc["sim_score"] < sim_t) & (mc["gap_fixed"] < gap_t)
            tp, fp, fn, prec, rec, f1 = pr_metrics(flagged, mc["dirty"])

            cand = {
                "category": cat,
                "sim_t": sim_t,
                "gap_t": gap_t,
                "TP": tp, "FP": fp, "FN": fn,
                "prec": prec, "rec": rec, "f1": f1,
                "n_rows": len(mc),
                "n_dirty": int(mc["dirty"].sum()),
            }

            if MODE == "max_f1":
                key = (f1, prec)  # tie-break on precision
            elif MODE == "max_recall_with_min_precision":
                if prec < MIN_PREC:
                    continue
                key = (rec, f1)   # maximize recall, tie-break on f1
            elif MODE == "min_fp_with_min_recall":
                if rec < MIN_REC:
                    continue
                key = (-fp, f1)   # minimize FP, tie-break on f1
            else:
                raise ValueError("Unknown MODE")

            if best is None or key > best[0]:
                best = (key, cand)

    # fallback if constraint impossible
    if best is None:
        # fallback to max_f1
        best2 = None
        for sim_t in sim_grid:
            for gap_t in gap_grid:
                flagged = (mc["sim_score"] < sim_t) & (mc["gap_fixed"] < gap_t)
                tp, fp, fn, prec, rec, f1 = pr_metrics(flagged, mc["dirty"])
                key = (f1, prec)
                cand = {
                    "category": cat, "sim_t": sim_t, "gap_t": gap_t,
                    "TP": tp, "FP": fp, "FN": fn, "prec": prec, "rec": rec, "f1": f1,
                    "n_rows": len(mc), "n_dirty": int(mc["dirty"].sum()),
                }
                if best2 is None or key > best2[0]:
                    best2 = (key, cand)
        rows.append(best2[1])
    else:
        rows.append(best[1])

best_df = pd.DataFrame(rows).sort_values("category")
best_df.to_csv(OUT_THRESH, index=False)

print("\n=== PER-CATEGORY THRESHOLDS (constrained) ===")
print(best_df.to_string(index=False))
print(f"\nSaved: {OUT_THRESH}")

# ----------------------------
# Apply thresholds and compute global
# ----------------------------
th_map = best_df.set_index("category")[["sim_t", "gap_t"]].to_dict("index")

def apply_rowwise(row):
    t = th_map.get(row["category"], {"sim_t": 0.30, "gap_t": 0.05})
    return (row["sim_score"] < t["sim_t"]) and (row["gap_fixed"] < t["gap_t"])

flagged_all = m.apply(apply_rowwise, axis=1)
tp, fp, fn, prec, rec, f1 = pr_metrics(flagged_all, m["dirty"])

print("\n=== GLOBAL METRICS (constrained per-category) ===")
print(f"prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  TP={tp} FP={fp} FN={fn}")

fn_df = m[(~flagged_all) & (m["dirty"])]
fp_df = m[( flagged_all) & (~m["dirty"])]

print("\nFN by category:\n", fn_df["category"].value_counts().to_string())
print("\nFP by category:\n", fp_df["category"].value_counts().to_string())
