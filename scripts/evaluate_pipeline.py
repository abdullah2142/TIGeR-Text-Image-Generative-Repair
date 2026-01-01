from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def prf(tp, fp, fn):
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(prec), float(rec), float(f1)


def resolve_path(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()


def coalesce_noise_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    After merges, noise_label may become:
      - noise_label
      - noise_label_x / noise_label_y
      - noise_label_lbl (our label copy)
      - noise_label_label (rare)
    Ensure df['noise_label'] exists and is filled as best as possible.
    """
    # If we already have noise_label, still try filling from lbl if available
    if "noise_label" in df.columns:
        if "noise_label_lbl" in df.columns:
            df["noise_label"] = (
                df["noise_label"].astype("string").fillna(df["noise_label_lbl"].astype("string"))
            )
        return df

    # Otherwise, coalesce from candidates
    candidates = [c for c in ["noise_label_x", "noise_label_y", "noise_label_lbl", "noise_label_label"] if c in df.columns]
    if not candidates:
        raise KeyError("noise_label not found after merges (no noise_label / noise_label_x/y / noise_label_lbl).")

    s = df[candidates[0]].astype("string")
    for c in candidates[1:]:
        s = s.fillna(df[c].astype("string"))
    df["noise_label"] = s
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy_parquet", required=True, help="e.g., data/processed/toy_noisy_rows.parquet")
    ap.add_argument("--sieve_noisy_csv", required=True, help="e.g., data/outputs/sieve_results_toy_noisy_rows.csv")
    ap.add_argument("--sieve_repaired_csv", required=True, help="e.g., data/outputs/sieve_results_repaired_toy_noisy_rows.csv")
    ap.add_argument("--out_prefix", default="data/outputs/evaluation_report", help="Output prefix (no extension)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    noisy_parquet = resolve_path(root, args.noisy_parquet)
    sieve_noisy_csv = resolve_path(root, args.sieve_noisy_csv)
    sieve_repaired_csv = resolve_path(root, args.sieve_repaired_csv)
    out_prefix = resolve_path(root, args.out_prefix)

    if not noisy_parquet.exists():
        raise FileNotFoundError(noisy_parquet)
    if not sieve_noisy_csv.exists():
        raise FileNotFoundError(sieve_noisy_csv)
    if not sieve_repaired_csv.exists():
        raise FileNotFoundError(sieve_repaired_csv)

    # ----------------------------
    # Load ground-truth labels
    # ----------------------------
    labels = pd.read_parquet(noisy_parquet)
    if "noise_label" not in labels.columns:
        raise ValueError("noisy_parquet must contain a 'noise_label' column.")
    labels = labels[["row_id", "noise_label"]].copy()
    labels["row_id"] = labels["row_id"].astype(str)
    labels = labels.rename(columns={"noise_label": "noise_label_lbl"})

    # ----------------------------
    # Load Sieve outputs (before/after)
    # ----------------------------
    noisy = pd.read_csv(sieve_noisy_csv)
    repaired = pd.read_csv(sieve_repaired_csv)

    noisy["row_id"] = noisy["row_id"].astype(str)
    repaired["row_id"] = repaired["row_id"].astype(str)

    # Safety: make sure the before file has a threshold column (we need it for locked evaluation)
    if "sieve_threshold" not in noisy.columns:
        raise ValueError(
            "sieve_noisy_csv must contain 'sieve_threshold'. "
            "Your run_sieve.py should output it."
        )

    # ----------------------------
    # Merge: noisy sieve + labels + repaired sieve
    # Overlapping cols like 'flagged' and 'sim_score' become *_before / *_after
    # ----------------------------
    df = noisy.merge(labels, on="row_id", how="left")
    df = df.merge(
        repaired[["row_id", "flagged", "sim_score"]],
        on="row_id",
        how="left",
        suffixes=("_before", "_after"),
    )

    # Ensure we have df["noise_label"]
    df = coalesce_noise_label(df)

    # ----------------------------
    # Ground truth: dirty if noise_label != clean
    # ----------------------------
    df["is_dirty"] = df["noise_label"].fillna("unknown").ne("clean").astype(int)

    # ----------------------------
    # Predictions (Before)
    # ----------------------------
    df["flagged_before"] = df["flagged_before"].fillna(False).astype(bool)
    df["pred_dirty_before"] = df["flagged_before"].astype(int)

    # ---- Phase 1: Detection metrics (Before) ----
    y_true = df["is_dirty"].to_numpy()
    y_pred_before = df["pred_dirty_before"].to_numpy()
    tp, fp, tn, fn = confusion_counts(y_true, y_pred_before)
    prec, rec, f1 = prf(tp, fp, fn)

    # ----------------------------
    # Repair impact (LOCKED THRESHOLD)
    #
    # We do NOT trust df["flagged_after"] as "repair success" because the second run
    # can recompute thresholds and cause drift.
    #
    # Fair comparison: use the BEFORE threshold as a constant:
    #   pred_dirty_after_locked = (sim_score_after < sieve_threshold_before)
    # ----------------------------
    df["sim_score_before"] = pd.to_numeric(df.get("sim_score_before"), errors="coerce")
    df["sim_score_after"] = pd.to_numeric(df.get("sim_score_after"), errors="coerce")

    # "before threshold" is the one from the BEFORE run
    df["sieve_threshold_before"] = pd.to_numeric(df["sieve_threshold"], errors="coerce")

    df["pred_dirty_after_locked"] = (
        (df["sim_score_after"].notna())
        & (df["sieve_threshold_before"].notna())
        & (df["sim_score_after"] < df["sieve_threshold_before"])
    ).astype(int)

    flagged_before = df["flagged_before"] == True

    # Fixed means: was flagged before AND is NOT dirty anymore under locked threshold
    fixed_locked = flagged_before & (df["pred_dirty_after_locked"] == 0)
    fix_rate_locked = float(fixed_locked.sum() / (flagged_before.sum() + 1e-12))

    # Also report any “newly flagged” under locked threshold (should be 0)
    newly_flagged_locked = (~flagged_before) & (df["pred_dirty_after_locked"] == 1)

    # similarity delta (always useful)
    df["sim_delta"] = df["sim_score_after"] - df["sim_score_before"]
    avg_delta_all = float(df["sim_delta"].dropna().mean())
    avg_delta_fixed = float(df.loc[fixed_locked, "sim_delta"].dropna().mean()) if fixed_locked.any() else 0.0

    # Breakdown by noise_label
    by_noise = (
        df.groupby("noise_label", dropna=False)
        .agg(
            n=("row_id", "count"),
            dirty=("is_dirty", "sum"),
            flagged_before=("flagged_before", "sum"),
            fixed_locked=("row_id", lambda x: int((fixed_locked.loc[x.index]).sum())),
            newly_flagged_locked=("row_id", lambda x: int((newly_flagged_locked.loc[x.index]).sum())),
            avg_sim_before=("sim_score_before", "mean"),
            avg_sim_after=("sim_score_after", "mean"),
            avg_sim_delta=("sim_delta", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )

    # ----------------------------
    # Print report
    # ----------------------------
    print("\n===== PHASE 1: DETECTION METRICS (Sieve BEFORE) =====")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    print("\n===== REPAIR IMPACT (LOCKED THRESHOLD) =====")
    print(f"Flagged before: {int(flagged_before.sum())}")
    print(f"Fixed (flagged->ok, locked): {int(fixed_locked.sum())}")
    print(f"Fix rate (locked): {fix_rate_locked:.3f}")
    print(f"Newly flagged (ok->flagged, locked): {int(newly_flagged_locked.sum())}")
    print(f"Avg sim delta (all rows): {avg_delta_all:.5f}")
    print(f"Avg sim delta (fixed rows): {avg_delta_fixed:.5f}")

    print("\n===== BREAKDOWN BY noise_label =====")
    print(by_noise.to_string(index=False))

    # ----------------------------
    # Save
    # ----------------------------
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_detail = out_prefix.with_name(out_prefix.name + "_detail.csv")
    out_breakdown = out_prefix.with_name(out_prefix.name + "_breakdown.csv")

    df.to_csv(out_detail, index=False)
    by_noise.to_csv(out_breakdown, index=False)

    print("\n✅ Saved:")
    print("Detail:", out_detail)
    print("Breakdown:", out_breakdown)


if __name__ == "__main__":
    main()
