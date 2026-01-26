import pandas as pd
import numpy as np

BASE_PARQUET = "data/processed/toy_noisy_rows.parquet"
POST_CSV     = "data/outputs/sieve_results_toy_noisy_rows.csv"

# ------------------------------------------------------------
# TWO-TIER SETTINGS (edit THESE)
# ------------------------------------------------------------
# 1) Flag-for-review (high recall, ok to have some FP)
SUSPECT = dict(
    NAME="suspect",
    OUT_THRESH="data/outputs/per_category_thresholds_suspect.csv",
    MODE="max_recall_with_min_precision",   # "max_f1" | "max_recall_with_min_precision" | "min_fp_with_min_recall"
    MIN_PREC=0.65,                          # lower => more recall (more FP)
    MIN_REC=0.75,                           # used only if MODE="min_fp_with_min_recall"
    RULES=("AND", "OR"),                    # allow OR to boost recall
    DEFAULT_RULE="AND",
    DEFAULT_SIM=0.30,
    DEFAULT_GAP=0.05,
)

# 2) Auto-fixable subset (very high precision; only these can be auto-edited)
AUTOFIX = dict(
    NAME="autofix",
    OUT_THRESH="data/outputs/per_category_thresholds_autofix.csv",
    MODE="max_recall_with_min_precision",
    MIN_PREC=0.95,                          # higher => safer (lower recall)
    MIN_REC=0.75,
    RULES=("AND",),                         # keep strict; avoid OR for auto-fix
    DEFAULT_RULE="AND",
    DEFAULT_SIM=0.28,
    DEFAULT_GAP=0.05,
)

# Wider grids (you can expand further if needed)
SIM_GRID = [0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45]
GAP_GRID = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]

# Optional ladder print (helps you see best recall at min precision)
PRINT_LADDER = True
PREC_LADDER = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

# Optional row-level output
OUT_FLAGS_CSV = "data/outputs/sieve_results_with_flags.csv"

# ------------------------------------------------------------
# Load + merge
# ------------------------------------------------------------
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
m["gap"]       = pd.to_numeric(m["gap"], errors="coerce")
m["category"]  = m["category"].fillna("UNKNOWN")
m = m.dropna(subset=["sim_score", "gap"]).reset_index(drop=True)

# Treat negative gaps as "very suspicious"
# We keep both raw and a clipped version.
m["gap_fixed"] = m["gap"].clip(lower=0)
m["gap_neg"]   = m["gap"] < 0  # if True, you can force-flag in OR modes (see make_flag)

# ------------------------------------------------------------
# Metrics + rules
# ------------------------------------------------------------
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

def make_flag(df, sim_t, gap_t, rule):
    # Base conditions
    sim_bad = (df["sim_score"] < sim_t)
    gap_bad = (df["gap_fixed"] < gap_t)

    # Optional: always consider negative gap suspicious
    # This helps catch weird cases even if gap_fixed got clipped.
    gap_bad = gap_bad | df["gap_neg"]

    if rule == "AND":
        return sim_bad & gap_bad
    if rule == "OR":
        return sim_bad | gap_bad
    raise ValueError("rule must be AND or OR")

def select_thresholds(profile: dict, df_all: pd.DataFrame) -> pd.DataFrame:
    MODE = profile["MODE"]
    MIN_PREC = float(profile["MIN_PREC"])
    MIN_REC  = float(profile["MIN_REC"])
    RULES = profile["RULES"]

    if PRINT_LADDER:
        print(f"\n=== PRECISION–RECALL LADDER ({profile['NAME']}) ===")
        for cat, dfc in df_all.groupby("category"):
            print(f"\n[{cat}]")
            all_cands = []
            for rule in RULES:
                for sim_t in SIM_GRID:
                    for gap_t in GAP_GRID:
                        flagged = make_flag(dfc, sim_t, gap_t, rule)
                        tp, fp, fn, prec, rec, f1 = pr_metrics(flagged, dfc["dirty"])
                        all_cands.append((rule, sim_t, gap_t, tp, fp, fn, prec, rec, f1))
            cand_df = pd.DataFrame(all_cands, columns=["rule","sim_t","gap_t","TP","FP","FN","prec","rec","f1"])

            for p in PREC_LADDER:
                feasible = cand_df[cand_df["prec"] >= p]
                if len(feasible) == 0:
                    print(f"  min_prec>={p:.2f}: no solution in grid")
                    continue
                best = feasible.sort_values(["rec","f1"], ascending=False).head(1).iloc[0]
                print(
                    f"  min_prec>={p:.2f}: best={best.rule} sim<{best.sim_t:.2f} gap<{best.gap_t:.3f}  "
                    f"prec={best.prec:.3f} rec={best.rec:.3f} f1={best.f1:.3f}  "
                    f"TP={int(best.TP)} FP={int(best.FP)} FN={int(best.FN)}"
                )

    # Pick best per category under MODE
    rows = []
    for cat, dfc in df_all.groupby("category"):
        best = None

        for rule in RULES:
            for sim_t in SIM_GRID:
                for gap_t in GAP_GRID:
                    flagged = make_flag(dfc, sim_t, gap_t, rule)
                    tp, fp, fn, prec, rec, f1 = pr_metrics(flagged, dfc["dirty"])

                    cand = {
                        "category": cat,
                        "rule": rule,
                        "sim_t": float(sim_t),
                        "gap_t": float(gap_t),
                        "TP": tp, "FP": fp, "FN": fn,
                        "prec": prec, "rec": rec, "f1": f1,
                        "n_rows": len(dfc),
                        "n_dirty": int(dfc["dirty"].sum()),
                    }

                    if MODE == "max_f1":
                        key = (f1, prec, rec)
                    elif MODE == "max_recall_with_min_precision":
                        if prec < MIN_PREC:
                            continue
                        key = (rec, f1, -fp)
                    elif MODE == "min_fp_with_min_recall":
                        if rec < MIN_REC:
                            continue
                        key = (-fp, f1, rec)
                    else:
                        raise ValueError(f"Unknown MODE: {MODE}")

                    if best is None or key > best[0]:
                        best = (key, cand)

        # fallback: if constraints impossible, revert to max_f1 within RULES
        if best is None:
            best2 = None
            for rule in RULES:
                for sim_t in SIM_GRID:
                    for gap_t in GAP_GRID:
                        flagged = make_flag(dfc, sim_t, gap_t, rule)
                        tp, fp, fn, prec, rec, f1 = pr_metrics(flagged, dfc["dirty"])
                        key = (f1, prec, rec)
                        cand = {
                            "category": cat, "rule": rule, "sim_t": float(sim_t), "gap_t": float(gap_t),
                            "TP": tp, "FP": fp, "FN": fn, "prec": prec, "rec": rec, "f1": f1,
                            "n_rows": len(dfc), "n_dirty": int(dfc["dirty"].sum()),
                        }
                        if best2 is None or key > best2[0]:
                            best2 = (key, cand)
            rows.append(best2[1])
        else:
            rows.append(best[1])

    best_df = pd.DataFrame(rows).sort_values("category")
    best_df.to_csv(profile["OUT_THRESH"], index=False)

    print(f"\n=== CHOSEN PER-CATEGORY RULES ({profile['NAME']}) ===")
    print(best_df.to_string(index=False))
    print(f"Saved: {profile['OUT_THRESH']}")

    # Global metrics for this profile
    th_map = best_df.set_index("category")[["rule","sim_t","gap_t"]].to_dict("index")

    def apply_row(row):
        t = th_map.get(row["category"], {
            "rule": profile["DEFAULT_RULE"],
            "sim_t": profile["DEFAULT_SIM"],
            "gap_t": profile["DEFAULT_GAP"],
        })
        if t["rule"] == "AND":
            return (row["sim_score"] < t["sim_t"]) and ((row["gap_fixed"] < t["gap_t"]) or row["gap_neg"])
        else:
            return (row["sim_score"] < t["sim_t"]) or  ((row["gap_fixed"] < t["gap_t"]) or row["gap_neg"])

    flagged_all = df_all.apply(apply_row, axis=1)
    tp, fp, fn, prec, rec, f1 = pr_metrics(flagged_all, df_all["dirty"])

    print(f"\n=== GLOBAL METRICS ({profile['NAME']}) ===")
    print(f"MODE={profile['MODE']}  MIN_PREC={profile['MIN_PREC']}  MIN_REC={profile['MIN_REC']}")
    print(f"prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  TP={tp} FP={fp} FN={fn}")

    fn_df = df_all[(~flagged_all) & (df_all["dirty"])]
    fp_df = df_all[( flagged_all) & (~df_all["dirty"])]

    print("\nFN by category:\n", fn_df["category"].value_counts().to_string())
    print("\nFP by category:\n", fp_df["category"].value_counts().to_string())

    # return thresholds + flags
    best_df = best_df.copy()
    best_df["profile"] = profile["NAME"]
    return best_df, flagged_all

# ------------------------------------------------------------
# Run both profiles
# ------------------------------------------------------------
sus_df, suspect_flags = select_thresholds(SUSPECT, m)
aut_df, autofix_flags = select_thresholds(AUTOFIX, m)

# Force nesting: autofix ⊆ suspect
suspect_flags = (suspect_flags | autofix_flags)

out = m.copy()
out["flag_suspect"] = suspect_flags.values
out["flag_autofix"] = autofix_flags.values

# Some useful breakdowns
print("\n=== TWO-TIER SUMMARY ===")
print("flag_suspect count:", int(out["flag_suspect"].sum()))
print("flag_autofix  count:", int(out["flag_autofix"].sum()))
print("autofix subset of suspect? ->", bool((out["flag_autofix"] & ~out["flag_suspect"]).sum() == 0))

out.to_csv(OUT_FLAGS_CSV, index=False)
print(f"Saved row-level flags: {OUT_FLAGS_CSV}")
