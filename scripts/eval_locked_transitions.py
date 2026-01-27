from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def load_thresholds(csv_path: Path) -> pd.DataFrame:
    th = pd.read_csv(csv_path)
    # Expect at least: category, rule, sim_t, gap_t
    for c in ["category", "rule", "sim_t", "gap_t"]:
        if c not in th.columns:
            raise KeyError(f"Threshold file missing column: {c}")
    th["category"] = th["category"].astype(str)
    th["rule"] = th["rule"].astype(str).str.upper()
    th["sim_t"] = pd.to_numeric(th["sim_t"], errors="coerce")
    th["gap_t"] = pd.to_numeric(th["gap_t"], errors="coerce")
    return th[["category", "rule", "sim_t", "gap_t"]]


def compute_scores_from_npz(npz_path: Path) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    row_ids = z["row_id"].astype(str)
    text_emb = z["text_emb"].astype(np.float32)
    image_emb = z["image_emb"].astype(np.float32)

    text_emb = l2_normalize(text_emb)
    image_emb = l2_normalize(image_emb)

    # Cosine(sim) for each row's own (text_i, image_i)
    # If rows are aligned, this is correct.
    sims = text_emb @ image_emb.T  # (N, N)

    own_sim = np.diag(sims).copy()

    # gap = own_sim - best_other_text_for_this_image
    # best_other_text_for_image_i = max_j!=i sims[j, i]
    sims_no_diag = sims.copy()
    np.fill_diagonal(sims_no_diag, -1e9)
    best_other = sims_no_diag.max(axis=0)

    gap = own_sim - best_other

    return pd.DataFrame(
        {
            "row_id": row_ids,
            "sim_score": own_sim.astype(float),
            "gap": gap.astype(float),
        }
    )


def apply_locked_thresholds(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    th: pd.DataFrame,
    default_rule: str = "AND",
    default_sim_t: float = 0.30,
    default_gap_t: float = 0.05,
) -> pd.DataFrame:
    df = scores.merge(meta[["row_id", "category"]], on="row_id", how="left")
    df["category"] = df["category"].fillna("UNKNOWN").astype(str)

    # Treat negative gap as "very suspicious" by clipping to 0 (so it's always below small thresholds)
    df["gap_fixed"] = df["gap"].clip(lower=0)

    th2 = th.copy()
    th2["category"] = th2["category"].astype(str)

    df = df.merge(th2, on="category", how="left")
    df["rule"] = df["rule"].fillna(default_rule).astype(str).str.upper()
    df["sim_t"] = df["sim_t"].fillna(default_sim_t).astype(float)
    df["gap_t"] = df["gap_t"].fillna(default_gap_t).astype(float)

    # Vectorized flag
    cond_sim = df["sim_score"] < df["sim_t"]
    cond_gap = df["gap_fixed"] < df["gap_t"]

    df["flagged_locked"] = np.where(df["rule"] == "OR", (cond_sim | cond_gap), (cond_sim & cond_gap))
    df["flagged_locked"] = df["flagged_locked"].astype(bool)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_parquet", required=True, help="e.g., data/processed/toy_noisy_rows.parquet")
    ap.add_argument("--after_parquet", required=True, help="e.g., data/processed/repaired_toy_noisy_rows.parquet")
    ap.add_argument("--before_npz", required=True, help="CLIP embeddings for BEFORE dataset")
    ap.add_argument("--after_npz", required=True, help="CLIP embeddings for AFTER dataset")
    ap.add_argument("--suspect_thresh", required=True, help="per_category_thresholds_suspect.csv (locked)")
    ap.add_argument("--autofix_thresh", required=True, help="per_category_thresholds_autofix.csv (locked)")
    ap.add_argument("--out_csv", default="", help="output transitions csv path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    before_parquet = (root / args.before_parquet).resolve() if not Path(args.before_parquet).is_absolute() else Path(args.before_parquet)
    after_parquet  = (root / args.after_parquet).resolve()  if not Path(args.after_parquet).is_absolute()  else Path(args.after_parquet)
    before_npz     = (root / args.before_npz).resolve()     if not Path(args.before_npz).is_absolute()     else Path(args.before_npz)
    after_npz      = (root / args.after_npz).resolve()      if not Path(args.after_npz).is_absolute()      else Path(args.after_npz)

    suspect_thresh = (root / args.suspect_thresh).resolve() if not Path(args.suspect_thresh).is_absolute() else Path(args.suspect_thresh)
    autofix_thresh = (root / args.autofix_thresh).resolve() if not Path(args.autofix_thresh).is_absolute() else Path(args.autofix_thresh)

    for p in [before_parquet, after_parquet, before_npz, after_npz, suspect_thresh, autofix_thresh]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # Load row meta
    bef = pd.read_parquet(before_parquet)
    aft = pd.read_parquet(after_parquet)

    for df in [bef, aft]:
        if "row_id" not in df.columns:
            raise KeyError("parquet missing row_id")
        if "category" not in df.columns:
            df["category"] = "UNKNOWN"
        df["row_id"] = df["row_id"].astype(str)
        df["category"] = df["category"].astype(str)

    # Repair metadata (only exists on after usually)
    for c in ["repaired", "repair_action", "repair_notes", "repair_source_row_id"]:
        if c not in aft.columns:
            aft[c] = "" if c != "repaired" else False

    # Scores from embeddings
    scores_before = compute_scores_from_npz(before_npz)
    scores_after  = compute_scores_from_npz(after_npz)

    # Thresholds
    th_sus = load_thresholds(suspect_thresh)
    th_fix = load_thresholds(autofix_thresh)

    # Apply locked flags
    bef_sus = apply_locked_thresholds(scores_before, bef, th_sus).rename(columns={"flagged_locked": "flag_before_suspect"})
    aft_sus = apply_locked_thresholds(scores_after,  aft, th_sus).rename(columns={"flagged_locked": "flag_after_suspect"})

    bef_fix = apply_locked_thresholds(scores_before, bef, th_fix).rename(columns={"flagged_locked": "flag_before_autofix"})
    aft_fix = apply_locked_thresholds(scores_after,  aft, th_fix).rename(columns={"flagged_locked": "flag_after_autofix"})

    # Merge transitions by row_id
    keep_cols = ["row_id", "category", "sim_score", "gap", "gap_fixed"]
    before_merge = bef_sus[keep_cols + ["flag_before_suspect"]].merge(
        bef_fix[["row_id", "flag_before_autofix"]],
        on="row_id",
        how="left",
    )
    after_merge = aft_sus[keep_cols + ["flag_after_suspect"]].merge(
        aft_fix[["row_id", "flag_after_autofix"]],
        on="row_id",
        how="left",
    )

    merged = before_merge.merge(
        after_merge,
        on=["row_id"],
        suffixes=("_before", "_after"),
        how="inner",
    )

    # Bring after-side repair metadata
    merged = merged.merge(
        aft[["row_id", "repaired", "repair_action", "repair_notes", "repair_source_row_id"]],
        on="row_id",
        how="left",
    )

    # Rename sim/gap columns so you can compare improvements
    merged = merged.rename(
        columns={
            "category_before": "category",
            "sim_score_before": "sim_before",
            "gap_before": "gap_before",
            "gap_fixed_before": "gap_fixed_before",
            "sim_score_after": "sim_after",
            "gap_after": "gap_after",
            "gap_fixed_after": "gap_fixed_after",
        }
    )

    # Transition labels
    def transition(a: bool, b: bool) -> str:
        if a and not b: return "fixed(True->False)"
        if a and b:     return "still_flagged(True->True)"
        if (not a) and b: return "regressed(False->True)"
        return "still_clean(False->False)"

    merged["suspect_transition"] = [
        transition(a, b) for a, b in zip(merged["flag_before_suspect"], merged["flag_after_suspect"])
    ]
    merged["autofix_transition"] = [
        transition(a, b) for a, b in zip(merged["flag_before_autofix"], merged["flag_after_autofix"])
    ]

    # Output
    out_csv = Path(args.out_csv) if args.out_csv else (root / "data/outputs/locked_eval_transitions_toy.csv")
    out_csv = out_csv if out_csv.is_absolute() else (root / out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    # ---- Print summary ----
    def print_block(name: str, before_col: str, after_col: str, trans_col: str):
        nb = int(merged[before_col].sum())
        na = int(merged[after_col].sum())
        print(f"\n=== {name} (LOCKED) ===")
        print(f"flagged_before: {nb}   flagged_after: {na}   delta: {na - nb}")
        print("\nTransitions:")
        print(merged[trans_col].value_counts().to_string())

        # By repair_action
        if "repair_action" in merged.columns:
            print("\nBy repair_action (fixed rate = True->False):")
            tmp = merged.copy()
            tmp["fixed"] = (tmp[before_col] == True) & (tmp[after_col] == False)
            g = tmp.groupby("repair_action")["fixed"].agg(["sum", "count"])
            g["fixed_rate"] = (g["sum"] / (g["count"].clip(lower=1))).round(3)
            print(g.sort_values("count", ascending=False).to_string())

    print("✅ Locked-threshold before→after evaluation")
    print("Rows compared:", len(merged))
    print("Output:", out_csv)

    print_block("SUSPECT", "flag_before_suspect", "flag_after_suspect", "suspect_transition")
    print_block("AUTOFIX",  "flag_before_autofix",  "flag_after_autofix",  "autofix_transition")

    # Focus: color patches
    color_df = merged[merged["repair_action"].astype(str) == "v2t_color_patch"].copy()
    if len(color_df) > 0:
        color_df["sim_delta"] = color_df["sim_after"] - color_df["sim_before"]
        print("\n=== Color patch focus (v2t_color_patch) ===")
        print("count:", len(color_df))
        print("median sim_delta:", float(color_df["sim_delta"].median()))
        show = color_df.sort_values("sim_delta", ascending=False).head(10)[
            ["row_id", "category", "sim_before", "sim_after", "sim_delta",
             "flag_before_suspect", "flag_after_suspect",
             "flag_before_autofix", "flag_after_autofix"]
        ]
        print("\nTop +sim rows:")
        print(show.to_string(index=False))
    else:
        print("\n(no v2t_color_patch rows found in repaired parquet metadata)")

    print("\nDone.")


if __name__ == "__main__":
    main()
