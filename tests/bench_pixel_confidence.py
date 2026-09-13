"""Does `pixel_color_confidence` rank correctness? (B4). Not a pytest module.

Run:  python tests/bench_pixel_confidence.py

Reads the committed B0 estimator-attribution report from the corrected ABO run
(`paper_assets/results/abo/v2t_estimator_diagnostics.csv`) and asks the one
question B4 poses: the solver gates the pixel estimator on a raw pixel share at
0.55 -- does that share actually predict whether the value written was right?

Rows recur across ablation configs with the same estimator output, so the
population is deduplicated on (row_id, pixel_value, pixel_conf).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "paper_assets/results/abo/v2t_estimator_diagnostics.csv"
BINS = [0.0, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.01]


def main() -> None:
    df = pd.read_csv(CSV)
    u = df.drop_duplicates(subset=["row_id", "pixel_value", "pixel_conf"])
    px = u.dropna(subset=["pixel_conf", "pixel_correct"])

    print(f"{len(df)} report rows -> {len(u)} unique estimator outcomes, "
          f"{len(px)} with a pixel estimate and a ground-truth colour\n")

    print("1. Would the pixel value have been correct, by its own share?")
    b = pd.cut(px.pixel_conf, BINS, right=False)
    t = px.groupby(b, observed=True).pixel_correct.agg(["count", "mean"])
    for iv, row in t.iterrows():
        print(f"   share {str(iv):14s} n={int(row['count']):4d}  correct {row['mean']:.1%}")
    lo, hi = px[px.pixel_conf < 0.55], px[px.pixel_conf >= 0.55]
    print(f"   the 0.55 gate: below n={len(lo)} {lo.pixel_correct.mean():.1%} | "
          f"above n={len(hi)} {hi.pixel_correct.mean():.1%}")
    print(f"   corr(share, correct) = {np.corrcoef(px.pixel_conf, px.pixel_correct)[0, 1]:.3f}\n")

    print("2. What the gate can still change, after B6 escalates disagreements:")
    has_px = df.pixel_value.notna() & (df.pixel_value != "")
    has_pr = df.probe_value.notna() & (df.probe_value != "")
    print(f"   both estimators produced a value: {int((has_px & has_pr).sum())} rows")
    print(f"   pixel only (the gate decides alone): {int((has_px & ~has_pr).sum())} rows")
    print(f"   probe only (no pixel to gate):       {int((~has_px & has_pr).sum())} rows")
    agree = df.groupby(df.estimators_agree.astype(str)).agg(
        n=("row_id", "size"), written=("written_color", "count"))
    print(f"   by agreement:\n{agree.to_string()}\n")

    print("3. On the rows the gate does not decide -- both agree, value committed:")
    a = u[u.estimators_agree == True].dropna(subset=["correct"])  # noqa: E712
    print(f"   n={len(a)}  written value correct {a.correct.mean():.1%}")
    ab = pd.cut(a.pixel_conf, [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01], right=False)
    for iv, row in a.groupby(ab, observed=True).correct.agg(["count", "mean"]).iterrows():
        print(f"   share {str(iv):14s} n={int(row['count']):3d}  correct {row['mean']:.1%}")

    print("\n4. Where the pixel path's errors actually live (by value returned):")
    for val, row in px.groupby("pixel_value").pixel_correct.agg(
            ["count", "mean"]).sort_values("count", ascending=False).head(6).iterrows():
        print(f"   {val:12s} n={int(row['count']):3d}  correct {row['mean']:.1%}")


if __name__ == "__main__":
    main()
