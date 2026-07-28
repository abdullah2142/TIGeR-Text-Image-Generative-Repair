"""Detection evaluation: per-error-type recall, product-level resampling (A3.1, 5.5).

- Per-error-type recall is a first-class metric (aggregate F1 hid the dead
  mutate_text class).
- Confidence intervals: Wilson interval on product-level counts, plus a
  product-level bootstrap for precision/recall/F1 (rows of the same product
  never split across resamples -- F4).
- Per-signal precision breakdown supports roadmap 3.4/3.6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tiger.sieve import SIGNAL_FLAGS


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def evaluate(df: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """df needs: noise_label, noise_subtype, flagged, product_id, per-signal flags."""
    d = df.copy()
    d["dirty"] = (d["noise_label"] != "clean").astype(int)
    d["pred"] = d["flagged"].astype(int)

    tp = int(((d.dirty == 1) & (d.pred == 1)).sum())
    fp = int(((d.dirty == 0) & (d.pred == 1)).sum())
    tn = int(((d.dirty == 0) & (d.pred == 0)).sum())
    fn = int(((d.dirty == 1) & (d.pred == 0)).sum())
    prec, rec, f1 = _prf(tp, fp, fn)

    out: dict = {
        "n_rows": int(len(d)),
        "n_products": int(d["product_id"].nunique()) if "product_id" in d else int(len(d)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": prec, "recall": rec, "f1": f1,
        "precision_wilson95": wilson_interval(tp, tp + fp),
        "recall_wilson95": wilson_interval(tp, tp + fn),
    }

    # ---------- per-error-type recall (first-class) ----------
    per_type = {}
    for label, grp in d[d.dirty == 1].groupby("noise_label"):
        k, n = int(grp.pred.sum()), int(len(grp))
        per_type[str(label)] = {"recall": k / n if n else float("nan"), "caught": k, "total": n,
                                "wilson95": wilson_interval(k, n)}
    out["recall_by_label"] = per_type

    per_sub = {}
    for sub, grp in d[d.dirty == 1].groupby("noise_subtype"):
        k, n = int(grp.pred.sum()), int(len(grp))
        per_sub[str(sub)] = {"recall": k / n if n else float("nan"), "caught": k, "total": n}
    out["recall_by_subtype"] = per_sub

    # ---------- per-signal precision (3.4/3.6) ----------
    per_signal = {}
    for sig in SIGNAL_FLAGS:
        if sig not in d.columns:
            continue
        fired = d[d[sig].astype(bool)]
        if len(fired) == 0:
            per_signal[sig] = {"fired": 0, "precision": None}
            continue
        per_signal[sig] = {
            "fired": int(len(fired)),
            "precision": float(fired.dirty.mean()),
            "hits_by_label": fired[fired.dirty == 1]["noise_label"].value_counts().to_dict(),
        }
    out["signal_precision"] = per_signal

    # ---------- product-level bootstrap (F4) ----------
    if "product_id" in d.columns and n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        products = d["product_id"].unique()
        groups = {p: g[["dirty", "pred"]].values for p, g in d.groupby("product_id")}
        stats = np.empty((n_bootstrap, 3))
        for b in range(n_bootstrap):
            sample = rng.choice(products, size=len(products), replace=True)
            arr = np.concatenate([groups[p] for p in sample])
            dirty, pred = arr[:, 0], arr[:, 1]
            btp = int(((dirty == 1) & (pred == 1)).sum())
            bfp = int(((dirty == 0) & (pred == 1)).sum())
            bfn = int(((dirty == 1) & (pred == 0)).sum())
            stats[b] = _prf(btp, bfp, bfn)
        lo, hi = np.nanpercentile(stats, [2.5, 97.5], axis=0)
        out["bootstrap95_product_level"] = {
            "precision": [float(lo[0]), float(hi[0])],
            "recall": [float(lo[1]), float(hi[1])],
            "f1": [float(lo[2]), float(hi[2])],
            "n_bootstrap": n_bootstrap,
        }
    return out


def format_report(res: dict) -> str:
    lines = []
    c = res["confusion"]
    
    total_products = res['n_products']
    flagged = c['tp'] + c['fp']
    missed = c['fn']
    false_alarms = c['fp']
    
    lines.append("\n\n📊 Detection Performance Report")
    lines.append("-" * 50)
    lines.append(f"Out of {total_products} products, we flagged {flagged} suspicious items.")
    lines.append(f"(We successfully caught {c['tp']} real errors, missed {missed}, and accidentally flagged {false_alarms} clean products).")
    
    lines.append("\nOverall Accuracy:")
    lines.append(f"- Precision: {res['precision']*100:.1f}% (When we flag an error, we are correct {res['precision']*100:.1f}% of the time)")
    lines.append(f"- Recall:    {res['recall']*100:.1f}% (We successfully caught {res['recall']*100:.1f}% of all real errors)")
    
    lines.append("\nHow well did we catch specific errors?")
    
    # Map raw subtype names to readable strings
    subtype_map = {
        "swap_image": "Wrong Image Entirely",
        "swap_image_same_category": "Wrong Image (Same Category)",
        "color_flip": "Wrong Color",
        "near_color_flip": "Slightly Wrong Color",
        "material_flip": "Wrong Material",
        "attribute_drop": "Missing Attribute",
        "title_contradiction": "Title/Attribute Mismatch",
        "missing_image": "Missing Image",
        "mixed": "Mixed (Multiple Errors)"
    }
    
    for sub, v in sorted(res.get("recall_by_subtype", {}).items()):
        if sub == "clean":
            continue
        readable_name = subtype_map.get(sub, sub)
        lines.append(f"- {readable_name:<28}: Caught {v['caught']}/{v['total']} ({v['recall']*100:.1f}%)")
        
    return "\n".join(lines)
