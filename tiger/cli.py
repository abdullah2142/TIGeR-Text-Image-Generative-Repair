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


def _tag(split: str, seed: int) -> str:
    return f"seed{seed}" if split == "report" else f"cal_seed{seed}"


def cmd_noise(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    df = pd.read_parquet(p["sample"] / "products.parquet")
    split = getattr(args, "split", "report")
    report = df[df["split"] == split].reset_index(drop=True)

    ncfg = dict(cfg.get("noise", {}))
    if args.seed is not None:
        ncfg["seed"] = int(args.seed)
    res = noise_mod.inject(report, schema, ncfg)

    p["processed"].mkdir(parents=True, exist_ok=True)
    tag = _tag(split, ncfg.get("seed", 7))
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
    sig, arrays = sieve_mod.compute_signals(cal, enc, schema, cfg, ROOT)
    thr = sieve_mod.calibrate(sig, cfg, schema)

    out = ROOT / "data/thresholds/tiger_locked_thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(thr.to_json(), encoding="utf-8")
    print(f"locked thresholds -> {out}")

    # Eq. 18 normalisation: clean-split leave-one-out delta stats (Phase 2.1)
    from tiger import analyzer as analyzer_mod
    loo_stats = analyzer_mod.calibrate_loo(sig, arrays, enc, schema, schema.checkable_fields())
    loo_out = ROOT / "data/thresholds/tiger_loo_calibration.json"
    loo_out.write_text(json.dumps(loo_stats, indent=2), encoding="utf-8")
    print(f"LOO calibration -> {loo_out}")


def cmd_detect(cfg: dict, args) -> None:
    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    tag = _tag(getattr(args, "split", "report"),
               args.seed if args.seed is not None else cfg.get("noise", {}).get("seed", 7))
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


def cmd_analyze(cfg: dict, args) -> None:
    """Phase 2.1/2.2: compute Eq. 18 + Eq. 19 evidence for flagged rows."""
    import numpy as np

    schema = load_schema(ROOT / cfg["data"]["schema"])
    p = _paths(cfg)
    tag = _tag(getattr(args, "split", "report"),
               args.seed if args.seed is not None else cfg.get("noise", {}).get("seed", 7))

    df = pd.read_parquet(p["outputs"] / f"sieve_{tag}.parquet")
    z = np.load(p["outputs"] / f"sieve_{tag}_arrays.npz")
    arrays = {k: z[k] for k in z.files}

    thr = sieve_mod.SieveThresholds.from_json(
        (ROOT / "data/thresholds/tiger_locked_thresholds.json").read_text(encoding="utf-8"))
    loo_stats = json.loads(
        (ROOT / "data/thresholds/tiger_loo_calibration.json").read_text(encoding="utf-8"))

    from tiger import analyzer as analyzer_mod
    enc = _encoder(cfg)
    evidences = analyzer_mod.analyze(df, arrays, enc, schema, thr, loo_stats, cfg, ROOT)
    out = p["outputs"] / f"evidence_{tag}.jsonl"
    analyzer_mod.save_evidence(evidences, out)
    print(f"wrote {out} ({len(evidences)} evidence records)")


def _load_evidence_with_labels(p: dict, tag: str) -> tuple[list[dict], list[str]]:
    ev = [json.loads(l) for l in (p["outputs"] / f"evidence_{tag}.jsonl").open(encoding="utf-8")]
    truth = pd.read_parquet(p["outputs"] / f"sieve_{tag}.parquet") \
        .set_index("row_id")["noise_label"].astype(str).to_dict()
    labels = [truth.get(e["row_id"], "clean") for e in ev]
    return ev, labels


def cmd_train_arbiter(cfg: dict, args) -> None:
    """Phase 2.3: fit p(E1..E4) router on CALIBRATION-split noise runs."""
    from tiger import arbiter as arbiter_mod

    p = _paths(cfg)
    seeds = [int(s) for s in cfg.get("arbiter", {}).get("train_seeds", [1007, 1008, 1009, 1010])]
    train_seeds, holdout_seed = seeds[:-1], seeds[-1]

    for s in seeds:
        ns = argparse.Namespace(seed=s, split="calibration")
        cmd_noise(cfg, ns)
        cmd_detect(cfg, ns)
        cmd_analyze(cfg, ns)

    train_ev, train_lab = [], []
    for s in train_seeds:
        ev, lab = _load_evidence_with_labels(p, _tag("calibration", s))
        # missing-modality rows are routed by rule, not by the model
        keep = [(e, l) for e, l in zip(ev, lab) if not e.get("image_missing") and not e.get("text_missing")]
        train_ev += [e for e, _ in keep]
        train_lab += [l for _, l in keep]

    model = arbiter_mod.train(train_ev, train_lab,
                              meta={"train_seeds": train_seeds, "holdout_seed": holdout_seed,
                                    "split": "calibration"})

    hold_ev, hold_lab = _load_evidence_with_labels(p, _tag("calibration", holdout_seed))
    keep = [(e, l) for e, l in zip(hold_ev, hold_lab) if not e.get("image_missing") and not e.get("text_missing")]
    hold_ev, hold_lab = [e for e, _ in keep], [l for _, l in keep]
    rel = arbiter_mod.reliability_table(model, hold_ev, hold_lab)
    model.training_meta["reliability_holdout"] = rel

    import numpy as np
    correct = 0
    for e, l in zip(hold_ev, hold_lab):
        pr = model.predict_proba(arbiter_mod.featurize(e))
        correct += (max(pr, key=pr.get) == arbiter_mod.LABEL_TO_CLASS.get(l, "CLEAN"))
    model.training_meta["holdout_accuracy"] = correct / max(1, len(hold_ev))

    out = ROOT / "data/thresholds/tiger_arbiter_model.json"
    out.write_text(model.to_json(), encoding="utf-8")
    print(f"trained on {model.training_meta['n_train']} rows "
          f"(classes {model.training_meta['class_counts']})")
    print(f"holdout (seed {holdout_seed}) accuracy: {model.training_meta['holdout_accuracy']:.3f}")
    print("reliability (holdout):")
    for b in rel:
        print(f"  conf~{b['mean_confidence']:.2f} -> acc {b['empirical_accuracy']:.2f} (n={b['n']})")
    print(f"wrote {out}")


def cmd_route(cfg: dict, args) -> None:
    """Apply the trained Arbiter to a report-split evidence file -> routing plan."""
    from tiger import arbiter as arbiter_mod

    p = _paths(cfg)
    tag = _tag("report", args.seed if args.seed is not None else cfg.get("noise", {}).get("seed", 7))
    model = arbiter_mod.ArbiterModel.from_json(
        (ROOT / "data/thresholds/tiger_arbiter_model.json").read_text(encoding="utf-8"))

    ev, labels = _load_evidence_with_labels(p, tag)
    routes = [arbiter_mod.route(e, model, cfg) for e in ev]

    rows = []
    for r, e, l in zip(routes, ev, labels):
        d = r.to_dict()
        d["probs"] = json.dumps({k: round(v, 3) for k, v in r.probs.items()})
        d["truth"] = arbiter_mod.LABEL_TO_CLASS.get(l, l.upper() if l.startswith("missing") else "CLEAN")
        d["truth_raw"] = l
        rows.append(d)
    plan = pd.DataFrame(rows)
    out = p["outputs"] / f"route_plan_{tag}.csv"
    plan.to_csv(out, index=False)

    print(plan.groupby(["truth_raw", "error_type"]).size().unstack(fill_value=0).to_string())
    print(f"\nactions:\n{plan['action'].value_counts().to_string()}")
    print(f"wrote {out}")


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
    ap.add_argument("command", choices=["synthgen", "noise", "calibrate", "detect", "evaluate",
                                        "analyze", "train-arbiter", "route", "sweep"])
    ap.add_argument("--config", default="configs/tiger.yaml")
    ap.add_argument("--seed", type=int, default=None, help="noise seed override")
    ap.add_argument("--seeds", default="7,8,9,10,11", help="sweep: comma-separated noise seeds")
    ap.add_argument("--split", default="report", choices=["report", "calibration"])
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    {
        "synthgen": cmd_synthgen,
        "noise": cmd_noise,
        "calibrate": cmd_calibrate,
        "detect": cmd_detect,
        "evaluate": cmd_evaluate,
        "analyze": cmd_analyze,
        "train-arbiter": cmd_train_arbiter,
        "route": cmd_route,
        "sweep": cmd_sweep,
    }[args.command](cfg, args)


if __name__ == "__main__":
    main()
