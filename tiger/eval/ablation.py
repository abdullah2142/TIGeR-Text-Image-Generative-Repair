"""Baselines and ablations (roadmap 5.4).

Every configuration is scored on the SAME signal-bearing report-split frames, so
differences are purely the detection rule. Per-error-type recall is reported for
each (A3.1: aggregate F1 hides dead classes). Configurations:

  random@budget   -- flag k rows at random, k = #flagged by the full system
  global_only     -- CLIP low-similarity gate alone (F2 shows its ceiling)
  probes_only     -- per-field contrastive probes alone
  text_only       -- schema/contradiction checks alone (no CLIP)
  full            -- all signals OR-fused (default apply_thresholds)
  full_fusion     -- full + precision-floor-calibrated per-signal margins (3.4)

Bundled with the arbiter/verify story this is the ablation set reviewers ask for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tiger import sieve as sieve_mod
from tiger.eval.detection import evaluate

CONFIGS = {
    "global_only": ["flag_low_sim", "flag_missing_image", "flag_missing_text"],
    "probes_only": ["flag_probe_color", "flag_probe_material", "flag_probe_pattern"],
    "text_only": ["flag_text_out_of_domain", "flag_title_contradiction"],
    "full": None,  # all SIGNAL_FLAGS
}


def _apply_config(signals_frames: list[pd.DataFrame], thr, name: str, fusion=None) -> pd.DataFrame:
    out = []
    for df in signals_frames:
        if name == "full_fusion":
            flagged = sieve_mod.apply_thresholds(df, thr, fusion=fusion)
        else:
            flagged = sieve_mod.apply_thresholds(df, thr, enabled_signals=CONFIGS.get(name))
        out.append(flagged)
    return pd.concat(out, ignore_index=True)


def run_ablations(signals_frames: list[pd.DataFrame], thr, fusion=None, seed: int = 0) -> dict:
    """signals_frames: per-seed report-split frames carrying signal + noise columns."""
    results = {}
    rng = np.random.default_rng(seed)

    names = list(CONFIGS)
    if fusion is not None:
        names.append("full_fusion")
    for name in names:
        pooled = _apply_config(signals_frames, thr, name, fusion=fusion)
        results[name] = evaluate(pooled, n_bootstrap=0)

    # random @ matched budget (matched to the full system's flag count, per seed)
    rand_frames = []
    for df in signals_frames:
        flagged = sieve_mod.apply_thresholds(df, thr)
        k = int(flagged["flagged"].sum())
        d = df.copy()
        idx = rng.choice(len(d), size=min(k, len(d)), replace=False)
        d["flagged"] = False
        d.loc[d.index[idx], "flagged"] = True
        rand_frames.append(d)
    results["random@budget"] = evaluate(pd.concat(rand_frames, ignore_index=True), n_bootstrap=0)

    return results


def format_ablations(results: dict) -> str:
    order = ["random@budget", "text_only", "global_only", "probes_only", "full", "full_fusion"]
    labels = sorted(set().union(*[r["recall_by_label"].keys() for r in results.values()]))
    header = f"{'config':16s} {'P':>6s} {'R':>6s} {'F1':>6s}  " + "  ".join(f"{l[:10]:>10s}" for l in labels)
    lines = [header, "-" * len(header)]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        rec = "  ".join(f"{r['recall_by_label'].get(l, {}).get('recall', float('nan')):>10.2f}" for l in labels)
        lines.append(f"{name:16s} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}  {rec}")
    return "\n".join(lines)
