"""Analyzer: emits EVIDENCE only (review A1.2).

The legacy mismatch_analyzer.py chose the repair direction and even built the
patch payload inside diagnosis; both moved out (direction -> Arbiter, payload
-> Solver). This module computes, per flagged row:

  - Eq. 18 leave-one-out field attribution: s_i^(j) = sim(I, caption \\ field j)
    - c_i, z-normalised per field per category against clean-calibration stats.
    Positive delta => removing the field helps => field j is suspect.
  - Eq. 19 TopKNN retrieval evidence over image embeddings, excluding the row's
    own product (F4): neighbour attribute agreement + neighbour self-consistency.
    High self-consistency with low agreement is attribute-side (E1) evidence; a
    visual outlier among attribute-matched rows is image-side (E2) evidence.
  - swap evidence with a per-category STANDARDISED margin (F5; replaces the raw
    0.01): best other-product text beats own caption by >= z std-devs.
  - deterministic pixel evidence: HSV dominant colour vs declared colour (1.7).
  - missing-modality routing constraints (F11): a missing image can never
    support V2T (there is no visual evidence); missing text can never support
    T2V direction inference.

The neighbour queries go through NeighborIndex, a minimal interface that can be
swapped for an ANN index at catalogue scale (F5 O(n^2) note, roadmap 4.4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tiger import text_views
from tiger.colors import estimate_dominant_color
from tiger.encoders import ClipEncoder
from tiger.schema import Schema


# ---------------------------------------------------------------------------
# neighbour interface (ANN-swappable)
# ---------------------------------------------------------------------------

class NeighborIndex:
    """Exact cosine top-k. Same interface an ANN backend would implement."""

    def __init__(self, emb: np.ndarray, product_ids: list[str], valid: np.ndarray):
        self.emb = emb
        self.product_ids = np.asarray(product_ids, dtype=object)
        self.valid = valid.astype(bool)

    def topk(self, query: np.ndarray, k: int, exclude_product: str | None = None) -> list[tuple[int, float]]:
        sims = self.emb @ query
        mask = self.valid.copy()
        if exclude_product is not None:
            mask &= self.product_ids != exclude_product
        sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order if np.isfinite(sims[i])]


# ---------------------------------------------------------------------------
# evidence records
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    row_id: str
    product_id: str
    category: str
    flag_reason: str
    # global signal
    sim_full: float | None = None
    sim_z: float | None = None
    # Eq. 18
    loo: dict = field(default_factory=dict)          # field -> {"delta","z"}
    loo_top_field: str = ""
    loo_top_z: float | None = None
    # probes (copied from sieve columns)
    probes: dict = field(default_factory=dict)       # field -> {"margin","z","pred"}
    # Eq. 19
    knn_self_consistency: float | None = None        # mean pairwise cosine among neighbours
    knn_agreement: dict = field(default_factory=dict)  # field -> fraction of neighbours agreeing
    knn_category_agreement: float | None = None
    attr_group_outlier_z: float | None = None        # visual outlier among attribute-matched rows
    # swap evidence (z-scored, F5)
    swap_best_other_sim: float | None = None
    swap_own_sim: float | None = None
    swap_z: float | None = None
    swap_best_other_product: str = ""
    # deterministic pixel evidence
    pixel_color: str = ""
    pixel_color_confidence: float | None = None
    pixel_agrees_declared: bool | None = None
    # text-only checks
    text_out_of_domain: bool = False
    title_contradiction: bool = False
    # missing modalities (F11 routing constraints)
    image_missing: bool = False
    text_missing: bool = False
    allowed_directions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


# ---------------------------------------------------------------------------
# calibration of LOO deltas on the clean split
# ---------------------------------------------------------------------------

def loo_caption(category: str, attrs: dict, drop_field: str) -> str:
    a = {k: v for k, v in attrs.items() if k != drop_field}
    return text_views.full_caption(category, a)


def calibrate_loo(df_clean_signals: pd.DataFrame, arrays: dict, encoder: ClipEncoder,
                  schema: Schema, fields: list[str]) -> dict:
    """Per-field per-category mean/std of clean LOO deltas (normalisation for Eq. 18)."""
    df = df_clean_signals
    image_emb = arrays["image_emb"]
    ok = arrays["image_ok"]
    out: dict[str, dict] = {f: {} for f in fields}

    deltas: dict[str, dict[str, list[float]]] = {f: {} for f in fields}
    for i in df.index:
        if not ok[i] or pd.isna(df.at[i, "sim_full"]):
            continue
        attrs = text_views.parse_attrs(df.at[i, "attributes"])
        cat = str(df.at[i, "category"])
        for f in fields:
            if f not in attrs:
                continue
            cap = loo_caption(cat, attrs, f)
            e = encoder.encode_texts([cap])[0]
            delta = float(image_emb[i] @ e) - float(df.at[i, "sim_full"])
            deltas[f].setdefault(cat, []).append(delta)

    for f in fields:
        for cat, vals in deltas[f].items():
            v = np.asarray(vals, dtype=float)
            out[f][cat] = {"mean": float(v.mean()),
                           "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                           "n": int(len(v))}
    encoder.save_cache()
    return out


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _zscore(x: float, stats: dict | None) -> float | None:
    if not stats or not stats.get("std"):
        return None
    return (x - stats["mean"]) / stats["std"]


def allowed_directions(image_missing: bool, text_missing: bool) -> list[str]:
    """F11 routing constraints.

    A missing image can NEVER support V2T (there is no visual evidence to trust);
    it routes to T2V acquisition (re-download / candidate pool) or human.
    Missing text legitimately routes to V2T. Both missing -> human only.
    """
    if image_missing and text_missing:
        return ["HUMAN"]
    if image_missing:
        return ["T2V_ACQUIRE", "HUMAN"]
    if text_missing:
        return ["V2T", "HUMAN"]
    return ["V2T", "T2V", "HUMAN"]


def analyze(df_signals: pd.DataFrame, arrays: dict, encoder: ClipEncoder, schema: Schema,
            thresholds, loo_stats: dict, cfg: dict, root: Path) -> list[Evidence]:
    """Compute evidence for every flagged row in df_signals."""
    acfg = cfg.get("analyzer", {})
    k = int(acfg.get("knn_k", 8))
    probe_fields = thresholds.config.get("probe_fields", [])

    df = df_signals.reset_index(drop=True)
    image_emb = arrays["image_emb"]
    cap_emb = arrays["caption_emb"]
    ok = np.asarray(arrays["image_ok"], dtype=bool)
    product_ids = df["product_id"].astype(str).tolist()
    categories = df["category"].astype(str).tolist()
    attrs_all = [text_views.parse_attrs(a) for a in df["attributes"]]

    index = NeighborIndex(image_emb, product_ids, ok)

    # per-category image-embedding centroids of attribute-matched groups
    # (used for the "visual outlier among attribute-matched rows" E2 evidence)
    evidences: list[Evidence] = []

    for i in df.index[df["flagged"].astype(bool)]:
        row = df.loc[i]
        cat = categories[i]
        attrs = attrs_all[i]
        ev = Evidence(
            row_id=str(row["row_id"]), product_id=product_ids[i], category=cat,
            flag_reason=str(row.get("flag_reason", "")),
            image_missing=bool(row.get("is_image_missing", False)),
            text_missing=bool(row.get("is_text_missing", False)),
            text_out_of_domain=bool(row.get("flag_text_out_of_domain", False)),
            title_contradiction=bool(row.get("flag_title_contradiction", False)),
        )

        # ----- F11 routing constraints -----
        ev.allowed_directions = allowed_directions(ev.image_missing, ev.text_missing)

        if not pd.isna(row.get("sim_full", np.nan)):
            ev.sim_full = float(row["sim_full"])
            ev.sim_z = float(row["sim_z"]) if not pd.isna(row.get("sim_z", np.nan)) else None

        for f in probe_fields:
            mcol, zcol, pcol = f"probe_{f}_margin", f"probe_{f}_z", f"probe_{f}_pred"
            if mcol in df.columns and not pd.isna(row.get(mcol, np.nan)):
                ev.probes[f] = {
                    "margin": float(row[mcol]),
                    "z": float(row[zcol]) if zcol in df.columns and not pd.isna(row.get(zcol, np.nan)) else None,
                    "pred": str(row.get(pcol, "")),
                }

        if ev.image_missing:
            evidences.append(ev)
            continue

        # ----- Eq. 18 leave-one-out -----
        if ev.sim_full is not None:
            best_f, best_z = "", None
            for f in schema.checkable_fields():
                if f not in attrs:
                    continue
                cap = loo_caption(cat, attrs, f)
                e = encoder.encode_texts([cap])[0]
                delta = float(image_emb[i] @ e) - ev.sim_full
                z = _zscore(delta, loo_stats.get(f, {}).get(cat))
                ev.loo[f] = {"delta": round(delta, 5), "z": None if z is None else round(z, 3)}
                if z is not None and (best_z is None or z > best_z):
                    best_f, best_z = f, z
            ev.loo_top_field, ev.loo_top_z = best_f, best_z

        # ----- Eq. 19 kNN evidence -----
        nbrs = index.topk(image_emb[i], k, exclude_product=product_ids[i])
        if nbrs:
            n_idx = [j for j, _ in nbrs]
            n_emb = image_emb[n_idx]
            if len(n_idx) > 1:
                pair = n_emb @ n_emb.T
                iu = np.triu_indices(len(n_idx), 1)
                ev.knn_self_consistency = float(pair[iu].mean())
            for f in schema.checkable_fields():
                declared = schema.normalize(f, attrs.get(f, "")) if attrs.get(f) else ""
                if not declared:
                    continue
                agree = [1.0 if schema.normalize(f, attrs_all[j].get(f, "")) == declared else 0.0
                         for j in n_idx]
                ev.knn_agreement[f] = float(np.mean(agree))
            ev.knn_category_agreement = float(np.mean([1.0 if categories[j] == cat else 0.0
                                                       for j in n_idx]))

        # visual outlier among attribute-matched rows (same category + colour)
        declared_color = schema.normalize("color", attrs.get("color", "")) if attrs.get("color") else ""
        if declared_color:
            group = [j for j in df.index
                     if ok[j] and j != i and categories[j] == cat
                     and schema.normalize("color", attrs_all[j].get("color", "")) == declared_color
                     and product_ids[j] != product_ids[i]]
            if len(group) >= 3:
                g_emb = image_emb[group]
                centroid = g_emb.mean(axis=0)
                centroid /= (np.linalg.norm(centroid) + 1e-12)
                own = float(image_emb[i] @ centroid)
                members = g_emb @ centroid
                std = float(members.std(ddof=1)) or 1e-6
                ev.attr_group_outlier_z = (own - float(members.mean())) / std

        # ----- swap evidence, z-scored (F5/1.6) -----
        sims_text = cap_emb @ image_emb[i]
        other = np.array([p != product_ids[i] for p in product_ids])
        if other.any():
            masked = np.where(other, sims_text, -np.inf)
            j = int(np.argmax(masked))
            ev.swap_best_other_sim = float(masked[j])
            ev.swap_own_sim = float(sims_text[i])
            ev.swap_best_other_product = product_ids[j]
            cat_stats = thresholds.sim.get(cat, thresholds.sim_global)
            std = cat_stats.get("std") or None
            if std:
                ev.swap_z = (ev.swap_best_other_sim - ev.swap_own_sim) / std

        # ----- deterministic pixel evidence (1.7) -----
        img_rel = str(row.get("image_path", "")).strip()
        img_path = (root / img_rel).resolve() if img_rel else None
        if img_path and img_path.exists():
            est = estimate_dominant_color(img_path)
            ev.pixel_color = est.top
            ev.pixel_color_confidence = est.confidence
            if declared_color:
                ev.pixel_agrees_declared = (est.top == declared_color)

        evidences.append(ev)

    encoder.save_cache()
    return evidences


def save_evidence(evidences: list[Evidence], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in evidences:
            f.write(json.dumps(ev.to_dict(), ensure_ascii=False, default=float) + "\n")
