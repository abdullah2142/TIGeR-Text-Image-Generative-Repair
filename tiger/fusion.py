"""Decision fusion calibrated to a precision floor (roadmap 3.4 + 3.6 guardrails).

The sieve OR-combines its signals. That maximises recall but lets a weak signal
(e.g. the material probe at ~0.73 precision) erode the precision story. This
module tunes each probe signal's z-margin on a LABELLED calibration set so that
every retained signal individually meets a precision floor, and QUARANTINES any
signal that cannot reach the floor even at its tightest setting.

Non-VLM half of the precision guardrails (3.6). The VLM audit of a random sample
of new flags is deferred until an API key is available; the quarantine mechanism
and the per-signal precision report exist now.

Output: a FusionConfig (per-signal z-margin + quarantine flag) consumed by
tiger.sieve.apply_thresholds. Calibrated on the calibration split only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FusionConfig:
    per_signal: dict = field(default_factory=dict)  # signal -> {z_margin, quarantined, precision, fired}
    precision_floor: float = 0.85
    meta: dict = field(default_factory=dict)

    def z_margin(self, field_name: str, default: float) -> float:
        return float(self.per_signal.get(field_name, {}).get("z_margin", default))

    def is_quarantined(self, signal: str) -> bool:
        return bool(self.per_signal.get(signal, {}).get("quarantined", False))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "FusionConfig":
        return cls(**json.loads(s))


def calibrate_fusion(labeled: pd.DataFrame, probe_fields: list[str],
                     precision_floor: float = 0.85,
                     z_grid: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0),
                     zcol_fmt: str = "probe_{}_z") -> FusionConfig:
    """Tune each probe's z-margin to the smallest value meeting the precision floor.

    `labeled` must carry `noise_label` and per-probe z columns. A row counts as a
    true positive for a probe if it is dirty (noise_label != clean). Non-probe
    signals (low_sim, text checks, missing) are measured and reported but not
    swept here -- low_sim precision is governed by the locked tau, and the text
    checks are near-deterministic.
    """
    dirty = (labeled["noise_label"].astype(str) != "clean")
    per_signal: dict[str, dict] = {}

    for fld in probe_fields:
        zcol = zcol_fmt.format(fld)
        if zcol not in labeled.columns:
            continue
        z = labeled[zcol]
        chosen = None
        for zt in z_grid:
            fired = (~z.isna()) & (z <= -zt)
            n = int(fired.sum())
            if n == 0:
                continue
            prec = float(dirty[fired].mean())
            if prec >= precision_floor:
                chosen = {"z_margin": float(zt), "quarantined": False,
                          "precision": round(prec, 3), "fired": n}
                break
        if chosen is None:
            # even the tightest margin misses the floor -> quarantine
            zt = z_grid[-1]
            fired = (~z.isna()) & (z <= -zt)
            n = int(fired.sum())
            prec = float(dirty[fired].mean()) if n else float("nan")
            chosen = {"z_margin": float(zt), "quarantined": True,
                      "precision": None if n == 0 else round(prec, 3), "fired": n}
        per_signal[f"flag_probe_{fld}"] = chosen

    # report-only precision for the non-swept signals
    for sig in ["flag_low_sim", "flag_text_out_of_domain", "flag_title_contradiction"]:
        if sig in labeled.columns:
            fired = labeled[sig].astype(bool)
            n = int(fired.sum())
            per_signal.setdefault(sig, {})
            per_signal[sig].update({"precision": round(float(dirty[fired].mean()), 3) if n else None,
                                    "fired": n, "quarantined": False, "z_margin": None})

    return FusionConfig(
        per_signal=per_signal, precision_floor=precision_floor,
        meta={"z_grid": list(z_grid), "n_calibration_rows": int(len(labeled)),
              "n_dirty": int(dirty.sum())},
    )
