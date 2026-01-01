from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


def run(cmd: list[str], cwd: Path) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def flagged_count(sieve_csv: Path) -> int:
    df = pd.read_csv(sieve_csv)
    return int(df["flagged"].sum())


def main():
    root = Path(__file__).resolve().parents[1]  # repo root (../)

    # -------- Defaults (no-arg friendly) --------
    default_in_parquet = os.getenv(
        "TIGER_IN_PARQUET",
        "data/processed/toy_noisy_rows.parquet",
    )
    default_thresholds_json = os.getenv(
        "TIGER_THRESHOLDS_JSON",
        "data/outputs/baseline_thresholds.json",
    )
    default_max_iters = int(os.getenv("TIGER_MAX_ITERS", "5"))
    # -------------------------------------------

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", default=default_in_parquet)
    ap.add_argument("--thresholds_json", default=default_thresholds_json)
    ap.add_argument("--max_iters", type=int, default=default_max_iters)
    args = ap.parse_args()

    # Resolve paths relative to repo root
    cur_parquet = Path(args.in_parquet)
    if not cur_parquet.is_absolute():
        cur_parquet = (root / cur_parquet).resolve()

    thresholds_json = Path(args.thresholds_json)
    if not thresholds_json.is_absolute():
        thresholds_json = (root / thresholds_json).resolve()

    if not cur_parquet.exists():
        raise FileNotFoundError(f"Missing --in_parquet: {cur_parquet}")
    if not thresholds_json.exists():
        raise FileNotFoundError(f"Missing --thresholds_json: {thresholds_json}")

    prev_flagged = None

    for it in range(1, args.max_iters + 1):
        # 1) run sieve
        run(
            [
                "python",
                "scripts/run_sieve.py",
                "--input_parquet",
                str(cur_parquet),
                "--thresholds_json",
                str(thresholds_json),
            ],
            cwd=root,
        )

        tag = cur_parquet.stem  # run_sieve uses stem-based naming already
        sieve_csv = (root / "data/outputs" / f"sieve_results_{tag}.csv").resolve()
        emb_npz = (root / "data/cache_embeddings" / f"clip_embeddings_{tag}.npz").resolve()

        if not sieve_csv.exists():
            raise FileNotFoundError(f"Expected sieve output missing: {sieve_csv}")
        if not emb_npz.exists():
            raise FileNotFoundError(f"Expected embedding cache missing: {emb_npz}")

        f = flagged_count(sieve_csv)
        print(f"[iter {it}] flagged = {f}")

        if prev_flagged is not None and f > prev_flagged:
            raise SystemExit(f"REGRESSION: flagged increased {prev_flagged} -> {f}")

        if f == 0:
            print("✅ Done: flagged=0")
            return

        prev_flagged = f

        # 2) mismatch analyzer
        run(
            [
                "python",
                "scripts/mismatch_analyzer.py",
                "--sieve_csv",
                str(sieve_csv),
                "--emb_npz",
                str(emb_npz),
            ],
            cwd=root,
        )

        queue_csv = (root / "data/outputs" / f"arbiter_queue_{tag}.csv").resolve()
        if not queue_csv.exists():
            raise FileNotFoundError(f"Expected queue csv missing: {queue_csv}")

        # 3) arbiter
        run(
            [
                "python",
                "scripts/run_arbiter.py",
                "--queue_csv",
                str(queue_csv),
            ],
            cwd=root,
        )

        plan_csv = (root / "data/outputs" / f"arbiter_plan_{tag}.csv").resolve()
        if not plan_csv.exists():
            raise FileNotFoundError(f"Expected plan csv missing: {plan_csv}")

        # 4) apply repairs → next parquet
        next_parquet = (root / "data/processed" / f"{tag}_r{it}.parquet").resolve()
        run(
            [
                "python",
                "scripts/apply_repairs.py",
                "--in_parquet",
                str(cur_parquet),
                "--plan_csv",
                str(plan_csv),
                "--out_parquet",
                str(next_parquet),
            ],
            cwd=root,
        )

        if not next_parquet.exists():
            raise FileNotFoundError(f"Expected repaired parquet missing: {next_parquet}")

        cur_parquet = next_parquet

    raise SystemExit("Reached max_iters without flagged=0")


if __name__ == "__main__":
    main()
