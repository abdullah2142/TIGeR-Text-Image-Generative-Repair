from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    # .../scripts/run_repair_cycle.py -> project root = parents[1]
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def flagged_count(sieve_csv: Path) -> int:
    if not sieve_csv.exists():
        raise FileNotFoundError(f"Expected sieve results not found: {sieve_csv}")
    df = pd.read_csv(sieve_csv)
    if "flagged" not in df.columns:
        raise ValueError(f"'flagged' column missing in {sieve_csv}")
    return int(df["flagged"].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_parquet",
        required=True,
        help="Input parquet (e.g., data/processed/toy_noisy_rows.parquet)",
    )
    ap.add_argument(
        "--thresholds_json",
        required=True,
        help="Thresholds json (e.g., data/outputs/baseline_thresholds.json)",
    )
    ap.add_argument("--max_iters", type=int, default=5)
    ap.add_argument(
        "--out_parquet_final",
        default="",
        help="Optional: where to copy the final parquet when flagged=0",
    )
    args = ap.parse_args()

    root = project_root()
    py = sys.executable  # ensures we use the current venv python

    cur_parquet = Path(args.in_parquet)
    cur_parquet = cur_parquet if cur_parquet.is_absolute() else (root / cur_parquet)
    cur_parquet = cur_parquet.resolve()

    thresholds_json = Path(args.thresholds_json)
    thresholds_json = thresholds_json if thresholds_json.is_absolute() else (root / thresholds_json)
    thresholds_json = thresholds_json.resolve()

    if not cur_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {cur_parquet}")
    if not thresholds_json.exists():
        raise FileNotFoundError(f"Missing thresholds json: {thresholds_json}")

    prev_flagged = None

    for it in range(1, args.max_iters + 1):
        tag = cur_parquet.stem  # run_sieve uses stem-based naming

        # 1) run sieve
        run(
            [py, "scripts/run_sieve.py", "--input_parquet", str(cur_parquet), "--thresholds_json", str(thresholds_json)],
            cwd=root,
        )

        sieve_csv = (root / "data/outputs" / f"sieve_results_{tag}.csv").resolve()
        emb_npz = (root / "data/cache_embeddings" / f"clip_embeddings_{tag}.npz").resolve()

        f = flagged_count(sieve_csv)
        print(f"[iter {it}] flagged = {f}")

        if prev_flagged is not None and f > prev_flagged:
            raise SystemExit(f"REGRESSION: flagged increased {prev_flagged} -> {f}")

        if f == 0:
            print("✅ Done: flagged=0")
            if args.out_parquet_final:
                out_final = Path(args.out_parquet_final)
                out_final = out_final if out_final.is_absolute() else (root / out_final)
                out_final.parent.mkdir(parents=True, exist_ok=True)
                out_final.write_bytes(cur_parquet.read_bytes())
                print("Final parquet copied to:", out_final.resolve())
            else:
                print("Final parquet:", cur_parquet)
            return

        prev_flagged = f

        # 2) mismatch analyzer
        run(
            [py, "scripts/mismatch_analyzer.py", "--sieve_csv", str(sieve_csv), "--emb_npz", str(emb_npz)],
            cwd=root,
        )
        queue_csv = (root / "data/outputs" / f"arbiter_queue_{tag}.csv").resolve()
        if not queue_csv.exists():
            raise FileNotFoundError(f"Expected arbiter queue not found: {queue_csv}")

        # 3) arbiter
        run([py, "scripts/run_arbiter.py", "--queue_csv", str(queue_csv)], cwd=root)
        plan_csv = (root / "data/outputs" / f"arbiter_plan_{tag}.csv").resolve()
        if not plan_csv.exists():
            raise FileNotFoundError(f"Expected arbiter plan not found: {plan_csv}")

        # 4) apply repairs → next parquet
        next_parquet = (root / "data/processed" / f"{tag}_r{it}.parquet").resolve()
        run(
            [py, "scripts/apply_repairs.py", "--in_parquet", str(cur_parquet), "--plan_csv", str(plan_csv), "--out_parquet", str(next_parquet)],
            cwd=root,
        )

        cur_parquet = next_parquet

    raise SystemExit("Reached max_iters without flagged=0")


if __name__ == "__main__":
    main()

#python scripts/run_repair_cycle.py `
  #--in_parquet data/processed/toy_noisy_rows.parquet `
  #--thresholds_json data/outputs/baseline_thresholds.json `
  #--max_iters 5
