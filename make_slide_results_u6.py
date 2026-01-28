# make_slide_results_u6.py
# Creates slide-ready metrics for Results 1–4 (no case studies).
# Outputs to: results/u6/metrics/
#
# Inputs (current repo paths):
# - data/processed/toy_noisy_rows.parquet
# - data/outputs/sieve_results_toy_noisy_rows.csv
# - data/outputs/sieve_results_repaired_toy_noisy_rows_v6.csv
# - data/outputs/upgrade6_vlm_log_v6.csv (optional)

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


ROOT = Path.cwd()

NOISY_PARQUET = ROOT / "data" / "processed" / "toy_noisy_rows.parquet"
SIEVE_BEFORE = ROOT / "data" / "outputs" / "sieve_results_toy_noisy_rows.csv"
SIEVE_AFTER = ROOT / "data" / "outputs" / "sieve_results_repaired_toy_noisy_rows_v6.csv"
VLM_LOG = ROOT / "data" / "outputs" / "upgrade6_vlm_log_v6.csv"

OUT_DIR = ROOT / "results" / "u6" / "metrics"


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Pick a column from df whose lowercased name matches one of candidates."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    """Normalize typical boolean-ish columns to True/False."""
    v = s.astype(str).str.strip().str.lower()
    return v.isin(["1", "true", "yes", "y", "t"])


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Sieve results ---
    ensure_exists(SIEVE_BEFORE, "Sieve BEFORE CSV")
    ensure_exists(SIEVE_AFTER, "Sieve AFTER CSV")

    b = pd.read_csv(SIEVE_BEFORE)
    a = pd.read_csv(SIEVE_AFTER)

    # Find common ID column
    id_candidates = ["row_id", "id", "idx", "index", "item_id"]
    id_b = pick_col(b, id_candidates)
    id_a = pick_col(a, id_candidates)

    if id_b is None or id_a is None:
        raise SystemExit(
            "Could not find a row id column in sieve CSVs.\n"
            f"Before columns: {b.columns.tolist()}\n"
            f"After columns:  {a.columns.tolist()}\n"
            "Expected one of: row_id / id / idx / index / item_id"
        )

    # Find flagged columns
    flag_col_b = pick_col(b, ["flagged", "is_flagged", "suspect", "is_suspect"])
    flag_col_a = pick_col(a, ["flagged", "is_flagged", "suspect", "is_suspect"])

    if flag_col_b is None or flag_col_a is None:
        raise SystemExit(
            "Could not find flagged/is_flagged/suspect column in sieve CSVs.\n"
            f"Before columns: {b.columns.tolist()}\n"
            f"After columns:  {a.columns.tolist()}\n"
        )

    # Normalize IDs as strings
    b[id_b] = b[id_b].astype(str)
    a[id_a] = a[id_a].astype(str)

    # Normalize flagged
    b_flag = to_bool_series(b[flag_col_b])
    a_flag = to_bool_series(a[flag_col_a])

    # --- Result 2: transitions_simple.csv ---
    bm = b[[id_b]].copy()
    bm["flag_before"] = b_flag.values

    am = a[[id_a]].copy()
    am["flag_after"] = a_flag.values
    am = am.rename(columns={id_a: id_b})

    m = bm.merge(am, on=id_b, how="inner")

    fixed = int(((m.flag_before == True) & (m.flag_after == False)).sum())
    remaining = int(((m.flag_before == True) & (m.flag_after == True)).sum())
    regress = int(((m.flag_before == False) & (m.flag_after == True)).sum())
    stable_ok = int(((m.flag_before == False) & (m.flag_after == False)).sum())
    flagged_before = int((m.flag_before == True).sum())
    ok_before = int(len(m) - flagged_before)

    trans = pd.DataFrame(
        [
            {
                "flagged_before": flagged_before,
                "fixed_flagged_to_ok": fixed,
                "remaining_flagged": remaining,
                "regress_ok_to_flagged": regress,
                "stable_ok": stable_ok,
                "fix_rate_over_flagged_before": (fixed / flagged_before) if flagged_before else 0.0,
                "regression_rate_over_ok_before": (regress / ok_before) if ok_before else 0.0,
                "n_rows_compared": int(len(m)),
            }
        ]
    )
    trans_path = OUT_DIR / "transitions_simple.csv"
    trans.to_csv(trans_path, index=False)

    # --- Result 4: reasons_before/after.csv ---
    reason_col_b = pick_col(b, ["flag_reason", "reason", "reasons", "mismatch_reason"])
    reason_col_a = pick_col(a, ["flag_reason", "reason", "reasons", "mismatch_reason"])

    def reason_counts(df: pd.DataFrame, flag_col: str, reason_col: str | None) -> pd.DataFrame:
        if reason_col is None:
            return pd.DataFrame([{"reason": "(no_reason_column_found)", "count": 0}])
        flagged = to_bool_series(df[flag_col])
        c = df.loc[flagged, reason_col].astype(str).value_counts().reset_index()
        c.columns = ["reason", "count"]
        return c

    rb = reason_counts(b, flag_col_b, reason_col_b)
    ra = reason_counts(a, flag_col_a, reason_col_a)

    rb.to_csv(OUT_DIR / "reasons_before.csv", index=False)
    ra.to_csv(OUT_DIR / "reasons_after.csv", index=False)

    # --- Result 1: detection_noisy.csv (Precision/Recall/F1 + confusion matrix) ---
    if NOISY_PARQUET.exists():
        try:
            p = pd.read_parquet(NOISY_PARQUET)
        except Exception as e:
            print(f"WARN: could not read noisy parquet for GT metrics: {e}")
            p = None

        if p is not None:
            gt_col = pick_col(p, ["noise_label", "is_noisy", "label"])
            if gt_col is None:
                print("WARN: no noise_label/is_noisy/label column in parquet. Skipping detection metrics.")
            else:
                # Build GT boolean
                if gt_col.lower() == "is_noisy":
                    gt = to_bool_series(p[gt_col])
                else:
                    v = p[gt_col].astype(str).str.strip().str.lower()
                    gt = (~v.isin(["", "clean", "none", "nan"])) & (v.notna())

                # Find id in parquet (prefer same as sieve id)
                pid = pick_col(p, [id_b] + ["row_id", "id", "idx", "index", "item_id"])
                if pid is None:
                    # fallback: use row index as id
                    p = p.reset_index().rename(columns={"index": id_b})
                    pid = id_b

                p[pid] = p[pid].astype(str)

                pred = b[[id_b]].copy()
                pred["pred_flagged"] = b_flag.values
                pred = pred.rename(columns={id_b: pid})

                pm = p[[pid]].copy()
                pm["gt_noisy"] = gt.values

                mm = pm.merge(pred, on=pid, how="inner")

                TP = int(((mm.gt_noisy == True) & (mm.pred_flagged == True)).sum())
                FP = int(((mm.gt_noisy == False) & (mm.pred_flagged == True)).sum())
                TN = int(((mm.gt_noisy == False) & (mm.pred_flagged == False)).sum())
                FN = int(((mm.gt_noisy == True) & (mm.pred_flagged == False)).sum())

                prec = TP / (TP + FP) if (TP + FP) else 0.0
                rec = TP / (TP + FN) if (TP + FN) else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

                det = pd.DataFrame(
                    [
                        {
                            "TP": TP,
                            "FP": FP,
                            "TN": TN,
                            "FN": FN,
                            "precision": prec,
                            "recall": rec,
                            "f1": f1,
                            "n_merged": int(len(mm)),
                        }
                    ]
                )
                det.to_csv(OUT_DIR / "detection_noisy.csv", index=False)
    else:
        print(f"WARN: No noisy parquet found at {NOISY_PARQUET}. Skipping detection metrics.")

    # --- Result 3: VLM status counts ---
    if VLM_LOG.exists():
        vdf = pd.read_csv(VLM_LOG)
        status_col = pick_col(vdf, ["status", "decision", "result", "outcome", "action", "applied", "is_applied"])
        if status_col is None:
            # still write something so slides pipeline doesn't break
            v_sum = pd.DataFrame([{"status": "(no_status_column_found)", "count": 0}])
        else:
            v_sum = vdf[status_col].astype(str).value_counts().reset_index()
            v_sum.columns = ["status", "count"]
        v_sum.to_csv(OUT_DIR / "vlm_status_counts.csv", index=False)
    else:
        # not fatal
        pd.DataFrame([{"status": "(vlm_log_missing)", "count": 0}]).to_csv(OUT_DIR / "vlm_status_counts.csv", index=False)

    print("\nDONE ✅ Slide metrics written to:", OUT_DIR)
    for f in ["detection_noisy.csv", "transitions_simple.csv", "reasons_before.csv", "reasons_after.csv", "vlm_status_counts.csv"]:
        fp = OUT_DIR / f
        print(" -", fp, "(exists)" if fp.exists() else "(missing)")


if __name__ == "__main__":
    main()
