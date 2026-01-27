# scripts/verify_repair_effect.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    return z


def idx_of(z, row_id: str) -> int:
    ids = z["row_id"]
    matches = np.where(ids == row_id)[0]
    if len(matches) == 0:
        raise KeyError(f"row_id not found: {row_id}")
    return int(matches[0])


def global_stats(z):
    sim = z["sim_score"].astype(float)
    sim = sim[sim >= 0]  # missing stored as -1
    flagged = z["flagged"].astype(int)
    return {
        "n": int(len(flagged)),
        "flagged": int(flagged.sum()),
        "sim_mean": float(sim.mean()) if sim.size else float("nan"),
        "sim_min": float(sim.min()) if sim.size else float("nan"),
        "sim_max": float(sim.max()) if sim.size else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="before npz")
    ap.add_argument("--b", required=True, help="after npz")
    ap.add_argument("--row_id", default="", help="optional row_id to diff")
    args = ap.parse_args()

    a = load_npz(Path(args.a))
    b = load_npz(Path(args.b))

    # Basic sanity
    if "row_id" not in a.files or "row_id" not in b.files:
        raise ValueError("Both npz must contain row_id")

    same_order = (a["row_id"] == b["row_id"]).all()
    if not same_order:
        raise AssertionError("row_id order differs between A and B (bad to compare index-wise).")
    print("✅ row_id order identical")

    # Global summary
    sa = global_stats(a)
    sb = global_stats(b)

    print("\n=== Global ===")
    print(f"rows           : {sa['n']} -> {sb['n']}")
    print(f"flagged        : {sa['flagged']} -> {sb['flagged']}  (Δ {sb['flagged']-sa['flagged']:+d})")
    print(f"sim_score mean : {sa['sim_mean']:.6f} -> {sb['sim_mean']:.6f}  (Δ {sb['sim_mean']-sa['sim_mean']:+.6f})")
    print(f"sim_score min  : {sa['sim_min']:.6f} -> {sb['sim_min']:.6f}")
    print(f"sim_score max  : {sa['sim_max']:.6f} -> {sb['sim_max']:.6f}")

    # Per-row summary
    if args.row_id:
        rid = args.row_id
        i = idx_of(a, rid)
        j = idx_of(b, rid)
        assert i == j

        keys = [
            "sim_score",
            "sim_title",
            "sim_attrs",
            "gap",
            "sieve_threshold",
            "attrs_threshold",
            "gap_threshold",
            "flagged",
            "color_mismatch",
        ]

        print(f"\n=== Row: {rid} (idx {i}) ===")
        for k in keys:
            if k not in a.files or k not in b.files:
                continue
            va = a[k][i]
            vb = b[k][j]
            try:
                va_f = float(va)
                vb_f = float(vb)
                print(f"{k:15s} {va_f: .6f} -> {vb_f: .6f}  (Δ {vb_f-va_f:+.6f})")
            except Exception:
                print(f"{k:15s} {va} -> {vb}")

    print("\nDone.")


if __name__ == "__main__":
    main()
