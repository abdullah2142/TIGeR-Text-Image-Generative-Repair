"""Command-line entry points for the tiger pipeline.

Usage (from repo root, venv active):

  python -m tiger.cli synthgen                     # build bundled sample catalogue
  python -m tiger.cli noise [--seed N]             # inject errors into the report split
  python -m tiger.cli calibrate                    # fit locked thresholds on clean calibration split
  python -m tiger.cli detect [--seed N]            # sieve the noisy report split with locked thresholds
  python -m tiger.cli evaluate [--seed N]          # detection metrics with product-level CIs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from tiger import sieve as sieve_mod
from tiger.data import noise as noise_mod
from tiger.data import synthgen
from tiger.encoders import ClipEncoder
from tiger.eval import detection as det_eval
from tiger.schema import load_schema

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: str = "configs/tiger.yaml") -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _paths(cfg: dict) -> dict[str, Path]:
    d = cfg["data"]
    return {
        "sample": ROOT / d["sample_dir"],
        "processed": ROOT / d["processed_dir"],
        "cache": ROOT / d["cache_dir"],
        "outputs": ROOT / d["outputs_dir"],
    }


def _encoder(cfg: dict) -> ClipEncoder:
    m = cfg["models"]
    return ClipEncoder(m["clip_model_name"], device=m.get("device", "cpu"),
                       batch_size=int(m.get("batch_size", 32)),
                       cache_dir=_paths(cfg)["cache"])


def cmd_synthgen(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    g = cfg.get("synthgen", {})
    df = synthgen.generate(
        ROOT, schema,
        seed=int(g.get("seed", 20260717)),
        products_per_category=int(g.get("products_per_category", 30)),
        image_size=int(g.get("image_size", 224)),
        calibration_fraction=float(g.get("calibration_fraction", 0.5)),
        out_dir=cfg["data"]["sample_dir"],
    )
    print(f"generated {len(df)} products -> {cfg['data']['sample_dir']}/products.parquet")
    print(df.groupby(["category", "split"]).size().to_string())


def cmd_noise(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    df = pd.read_parquet(p["sample"] / "products.parquet")
    report = df[df["split"] == "report"].reset_index(drop=True)

    ncfg = dict(cfg.get("noise", {}))
    if args.seed is not None:
        ncfg["seed"] = int(args.seed)
    res = noise_mod.inject(report, schema, ncfg)

    p["processed"].mkdir(parents=True, exist_ok=True)
    tag = f"seed{ncfg.get('seed', 7)}"
    out = p["processed"] / f"noisy_report_{tag}.parquet"
    res.df.to_parquet(out, index=False)
    res.audit.to_csv(p["processed"] / f"noise_audit_{tag}.csv", index=False)
    (p["processed"] / f"noisy_report_{tag}.meta.json").write_text(
        json.dumps(res.meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(res.meta, indent=2))


def cmd_calibrate(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    df = pd.read_parquet(p["sample"] / "products.parquet")
    cal = df[df["split"] == "calibration"].reset_index(drop=True)

    enc = _encoder(cfg)
    sig, _arrays = sieve_mod.compute_signals(cal, enc, schema, cfg, ROOT)
    thr = sieve_mod.calibrate(sig, cfg, schema)

    out = ROOT / "data/thresholds/tiger_locked_thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(thr.to_json(), encoding="utf-8")
    print(f"locked thresholds -> {out}")
    print(thr.to_json())


def cmd_detect(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    tag = f"seed{args.seed if args.seed is not None else cfg.get('noise', {}).get('seed', 7)}"
    noisy = pd.read_parquet(p["processed"] / f"noisy_report_{tag}.parquet")

    thr_path = ROOT / "data/thresholds/tiger_locked_thresholds.json"
    thr = sieve_mod.SieveThresholds.from_json(thr_path.read_text(encoding="utf-8"))

    enc = _encoder(cfg)
    sig, arrays = sieve_mod.compute_signals(noisy, enc, schema, cfg, ROOT)
    flagged = sieve_mod.apply_thresholds(sig, thr)

    import numpy as np
    np.savez_compressed(p["outputs"] / f"sieve_{tag}_arrays.npz", **arrays)

    p["outputs"].mkdir(parents=True, exist_ok=True)
    out = p["outputs"] / f"sieve_{tag}.parquet"
    flagged.to_parquet(out, index=False)
    print(f"wrote {out}")
    print(f"flagged {int(flagged['flagged'].sum())}/{len(flagged)} rows")
    print(flagged["flag_reason"].value_counts().to_string())


def cmd_evaluate(cfg: dict, args) -> None:
    p = _paths(cfg)
    tag = f"seed{args.seed if args.seed is not None else cfg.get('noise', {}).get('seed', 7)}"
    df = pd.read_parquet(p["outputs"] / f"sieve_{tag}.parquet")
    res = det_eval.evaluate(df)
    out = p["outputs"] / f"detection_metrics_{tag}.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(det_eval.format_report(res))
    print(f"\nwrote {out}")


def cmd_sweep(cfg: dict, args) -> None:
    """Multi-seed evaluation (roadmap 5.5): noise -> detect -> evaluate per seed,
    then aggregate mean +/- std and pooled per-error-type recall."""
    import numpy as np

    seeds = [int(s) for s in args.seeds.split(",")]
    p = _paths(cfg)
    all_res, pooled = [], []
    for s in seeds:
        ns = argparse.Namespace(seed=s)
        cmd_noise(cfg, ns)
        cmd_detect(cfg, ns)
        df = pd.read_parquet(p["outputs"] / f"sieve_seed{s}.parquet")
        pooled.append(df.assign(seed=s))
        res = det_eval.evaluate(df, n_bootstrap=0)
        all_res.append(res)
        print(f"seed {s}: P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f}")

    P = [r["precision"] for r in all_res]
    R = [r["recall"] for r in all_res]
    F = [r["f1"] for r in all_res]
    print(f"\n=== {len(seeds)} seeds ===")
    print(f"precision {np.mean(P):.3f} +/- {np.std(P, ddof=1):.3f}")
    print(f"recall    {np.mean(R):.3f} +/- {np.std(R, ddof=1):.3f}")
    print(f"f1        {np.mean(F):.3f} +/- {np.std(F, ddof=1):.3f}")

    big = pd.concat(pooled, ignore_index=True)
    # product-level bootstrap on the pooled runs (product resampled across seeds)
    res_pooled = det_eval.evaluate(big, n_bootstrap=2000)
    print("\n=== pooled across seeds ===")
    print(det_eval.format_report(res_pooled))
    out = p["outputs"] / "detection_metrics_sweep.json"
    out.write_text(json.dumps({"seeds": seeds, "per_seed": all_res, "pooled": res_pooled},
                              indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="tiger")
    ap.add_argument("command", choices=["synthgen", "noise", "calibrate", "detect", "evaluate", "sweep"])
    ap.add_argument("--config", default="configs/tiger.yaml")
    ap.add_argument("--seed", type=int, default=None, help="noise seed override")
    ap.add_argument("--seeds", default="7,8,9,10,11", help="sweep: comma-separated noise seeds")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    {
        "synthgen": cmd_synthgen,
        "noise": cmd_noise,
        "calibrate": cmd_calibrate,
        "detect": cmd_detect,
        "evaluate": cmd_evaluate,
        "sweep": cmd_sweep,
    }[args.command](cfg, args)


if __name__ == "__main__":
    main()
