"""Verify: per-repair acceptance gates Eq. 27-29 (roadmap 2.5, findings F7/F14).

The legacy verify_repair_effect.py compared aggregate flag counts before/after
and reported a mean similarity delta -- circular (repairs chosen to maximise
CLIP similarity, accepted by CLIP similarity) and never per-repair.

This module makes acceptance a per-repair decision with three gates:

  Eq. 27  schema validity:      A' |= C      (patched record satisfies Omega_j + C)
  Eq. 28  threshold:            c' >= tau_hat (LOCKED per-category threshold)
  Eq. 29  improvement margin:   c' - c >= eps (eps = caption-rewording noise floor)

Rejected repairs are rolled back by the caller and re-routed through the
Arbiter; the loop is capped at two passes (tiger.repair).

epsilon is not an arbitrary constant. It is the noise floor of image-text
similarity under a meaning-preserving caption rewording, measured on clean
calibration rows: if a repair moves c by less than a mere paraphrase would, the
improvement is wording wobble, not correction. Reported per run.

Structural note (F7): CLIP is both the repair objective and, here, the
acceptance signal, so these gates confirm the objective moved -- they do not by
themselves prove correctness. The independent-verifier ensemble (roadmap 6.4)
is the structural cure and is tracked separately; the VLM judge slots in via
`independent_ok` when an API key is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tiger import text_views
from tiger.encoders import ClipEncoder
from tiger.schema import Schema


@dataclass
class VerifyCalibration:
    epsilon: float
    epsilon_by_category: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def eps_for(self, category: str) -> float:
        return float(self.epsilon_by_category.get(category, self.epsilon))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "VerifyCalibration":
        return cls(**json.loads(s))


def calibrate_epsilon(df_clean_signals: pd.DataFrame, arrays: dict, encoder: ClipEncoder,
                      quantile: float = 0.95) -> VerifyCalibration:
    """eps = `quantile` of |c(paraphrase) - c(caption)| over clean rows."""
    df = df_clean_signals.reset_index(drop=True)
    image_emb = arrays["image_emb"]
    ok = np.asarray(arrays["image_ok"], dtype=bool)

    deltas, cat_deltas = [], {}
    for i in df.index:
        if not ok[i] or pd.isna(df.at[i, "sim_full"]):
            continue
        attrs = text_views.parse_attrs(df.at[i, "attributes"])
        cat = str(df.at[i, "category"])
        para = text_views.full_caption_paraphrase(cat, attrs)
        e = encoder.encode_texts([para])[0]
        d = abs(float(image_emb[i] @ e) - float(df.at[i, "sim_full"]))
        deltas.append(d)
        cat_deltas.setdefault(cat, []).append(d)
    encoder.save_cache()

    eps = float(np.quantile(deltas, quantile)) if deltas else 0.0
    by_cat = {c: float(np.quantile(v, quantile)) for c, v in cat_deltas.items() if len(v) >= 8}
    return VerifyCalibration(
        epsilon=eps, epsilon_by_category=by_cat,
        meta={"quantile": quantile, "n_rows": len(deltas),
              "epsilon_mean": float(np.mean(deltas)) if deltas else 0.0,
              "epsilon_max": float(np.max(deltas)) if deltas else 0.0},
    )


@dataclass
class Verdict:
    row_id: str
    accepted: bool
    c_before: float
    c_after: float
    delta: float
    tau: float
    epsilon: float
    schema_ok: bool
    threshold_ok: bool
    margin_ok: bool
    independent_ok: bool | None
    reason: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def verify_repair(row_id: str, category: str, attrs_after: dict, c_before: float,
                  c_after: float, tau: float, epsilon: float, schema: Schema,
                  independent_ok: bool | None = None) -> Verdict:
    """Apply Eq. 27-29 (and optional independent verifier) to one repair."""
    schema_ok = schema.is_valid(category, attrs_after)                 # Eq. 27
    threshold_ok = (c_after >= tau) if not np.isnan(tau) else True     # Eq. 28
    delta = c_after - c_before
    margin_ok = delta >= epsilon                                       # Eq. 29
    indep_ok = True if independent_ok is None else bool(independent_ok)

    accepted = schema_ok and threshold_ok and margin_ok and indep_ok
    if accepted:
        reason = "accepted"
    else:
        fails = []
        if not schema_ok:
            fails.append("schema(Eq27)")
        if not threshold_ok:
            fails.append(f"c'={c_after:.3f}<tau={tau:.3f}(Eq28)")
        if not margin_ok:
            fails.append(f"delta={delta:.3f}<eps={epsilon:.3f}(Eq29)")
        if independent_ok is False:
            fails.append("independent_verifier")
        reason = "rejected: " + ", ".join(fails)

    return Verdict(row_id, accepted, c_before, c_after, delta, tau, epsilon,
                   schema_ok, threshold_ok, margin_ok,
                   None if independent_ok is None else indep_ok, reason)


def load_calibration(path: str | Path) -> VerifyCalibration:
    return VerifyCalibration.from_json(Path(path).read_text(encoding="utf-8"))
