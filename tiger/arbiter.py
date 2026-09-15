"""Arbiter: calibrated error typing p(E1..E4), gamma gate, policy-gated routing.

Implements roadmap 2.3 (+2.6), fixing F8: the legacy run_arbiter.py was a
five-row label->action lookup with no probabilities, no confidence gate and no
policy object.

Design:
  - a transparent multinomial logistic model maps evidence features (Eq. 18
    LOO z, Eq. 19 kNN features, z-scored swap margin, probe z-margins, pixel
    agreement, text checks) to p(E1), p(E2), p(E3), p(CLEAN);
  - trained on flagged rows of the CALIBRATION split under separate noise
    seeds, so the router never sees the reporting split;
  - stored as plain JSON (feature names, standardisation stats, coefficients)
    -- inspectable and free of pickle;
  - Eq. 22 gate: if max p < gamma the row becomes E4 (explicit ambiguity
    state) and escalates to human review;
  - direction selection respects the Analyzer's allowed_directions (F11) and
    a T2V policy object (allowed categories / cost ceiling, roadmap 2.6);
  - CLEAN with high confidence is dismissed (detected as a sieve false
    positive); CLEAN with middling confidence escalates instead.

Error-type conventions: E1 = text wrong (V2T), E2 = image wrong (T2V),
E3 = mixed (both), E4 = ambiguous (escalate). Labels map from noise ground
truth: mutate_text->E1, swap_image->E2, mixed->E3, clean->CLEAN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASSES = ["E1", "E2", "E3", "CLEAN"]

LABEL_TO_CLASS = {
    "mutate_text": "E1",
    "swap_image": "E2",
    "mixed": "E3",
    "clean": "CLEAN",
}

FEATURES = [
    "sim_z",
    "swap_z",
    "loo_top_z",
    "loo_top_is_color",
    "grp_outlier_z",
    "knn_self_consistency",
    "knn_agree_color",
    "knn_category_agreement",
    "pixel_agrees",          # 1 agree / -1 disagree / 0 unknown
    "probe_color_z",
    "probe_material_z",
    "probe_pattern_z",
    "title_contradiction",
    "text_out_of_domain",
]


def featurize(ev: dict) -> np.ndarray:
    """Evidence record (analyzer JSONL dict) -> fixed-length feature vector.

    Missing values are encoded as 0 after standardisation-time centring is
    trained with the same convention, keeping train/serve symmetric.
    """
    probes = ev.get("probes") or {}

    def pz(f):
        v = (probes.get(f) or {}).get("z")
        return float(v) if v is not None else 0.0

    pixel = ev.get("pixel_agrees_declared")
    vals = {
        "sim_z": float(ev["sim_z"]) if ev.get("sim_z") is not None else 0.0,
        "swap_z": float(ev["swap_z"]) if ev.get("swap_z") is not None else 0.0,
        "loo_top_z": float(ev["loo_top_z"]) if ev.get("loo_top_z") is not None else 0.0,
        "loo_top_is_color": 1.0 if ev.get("loo_top_field") == "color" else 0.0,
        "grp_outlier_z": float(ev["attr_group_outlier_z"]) if ev.get("attr_group_outlier_z") is not None else 0.0,
        "knn_self_consistency": float(ev["knn_self_consistency"]) if ev.get("knn_self_consistency") is not None else 0.0,
        "knn_agree_color": float((ev.get("knn_agreement") or {}).get("color", 0.0) or 0.0),
        "knn_category_agreement": float(ev["knn_category_agreement"]) if ev.get("knn_category_agreement") is not None else 0.0,
        "pixel_agrees": 0.0 if pixel is None else (1.0 if pixel else -1.0),
        "probe_color_z": pz("color"),
        "probe_material_z": pz("material"),
        "probe_pattern_z": pz("pattern"),
        "title_contradiction": 1.0 if ev.get("title_contradiction") else 0.0,
        "text_out_of_domain": 1.0 if ev.get("text_out_of_domain") else 0.0,
    }
    return np.array([vals[f] for f in FEATURES], dtype=np.float64)


@dataclass
class ArbiterModel:
    feature_names: list
    classes: list
    mean: list
    scale: list
    coef: list          # (n_classes, n_features)
    intercept: list     # (n_classes,)
    training_meta: dict = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        xs = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        logits = np.asarray(self.coef) @ xs + np.asarray(self.intercept)
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return {c: float(v) for c, v in zip(self.classes, p)}

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ArbiterModel":
        return cls(**json.loads(s))


def train(evidence_records: list[dict], labels: list[str], meta: dict | None = None) -> ArbiterModel:
    """Fit the multinomial logistic router on labelled evidence records."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.stack([featurize(e) for e in evidence_records])
    y = np.array([LABEL_TO_CLASS.get(l, "CLEAN") for l in labels])

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xs, y)

    # align coefficient rows to the CLASSES order (sklearn sorts classes)
    coef_rows, intercepts = [], []
    for c in CLASSES:
        if c in clf.classes_:
            i = list(clf.classes_).index(c)
            coef_rows.append(clf.coef_[i].tolist())
            intercepts.append(float(clf.intercept_[i]))
        else:  # class absent from training data: impossible under the softmax
            coef_rows.append([0.0] * X.shape[1])
            intercepts.append(-1e3)

    return ArbiterModel(
        feature_names=FEATURES,
        classes=CLASSES,
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=coef_rows,
        intercept=intercepts,
        training_meta={**(meta or {}),
                       "n_train": int(len(y)),
                       "class_counts": {c: int((y == c).sum()) for c in CLASSES}},
    )


def reliability_table(model: ArbiterModel, evidence_records: list[dict], labels: list[str],
                      n_bins: int = 5) -> list[dict]:
    """Coarse reliability curve: predicted max-p vs empirical accuracy per bin."""
    rows = []
    for ev, lab in zip(evidence_records, labels):
        p = model.predict_proba(featurize(ev))
        pred = max(p, key=p.get)
        rows.append((max(p.values()), pred == LABEL_TO_CLASS.get(lab, "CLEAN")))
    rows.sort()
    out = []
    edges = np.linspace(0, len(rows), n_bins + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = rows[a:b]
        if not chunk:
            continue
        out.append({"mean_confidence": float(np.mean([c for c, _ in chunk])),
                    "empirical_accuracy": float(np.mean([ok for _, ok in chunk])),
                    "n": len(chunk)})
    return out


def calibration_report(model: ArbiterModel, evidence_records: list[dict], labels: list[str],
                       gamma: float, n_bins: int = 10) -> dict:
    """How far the router's stated confidence is from its empirical accuracy.

    The gamma gate (Eq. 22) escalates on ``max p < gamma`` and the dismiss rule
    fires on ``max p >= dismiss_threshold``, so both read ``predict_proba`` as a
    probability. `class_weight="balanced"` reweights the training objective,
    which is a standard way to *lose* calibration -- D1. This measures whether
    that happened rather than assuming either way, and reports the error where
    the decisions are actually taken, not only in aggregate.

    Returns expected calibration error over equal-count bins, plus the signed
    gap (empirical accuracy - stated confidence) in a window around `gamma`.
    A negative gap means the router is under-confident there, so the gate
    escalates rows it would in fact have routed correctly.
    """
    conf, hit, pred = [], [], []
    for ev, lab in zip(evidence_records, labels):
        p = model.predict_proba(featurize(ev))
        top = max(p, key=p.get)
        conf.append(p[top])
        pred.append(top)
        hit.append(float(top == LABEL_TO_CLASS.get(lab, "CLEAN")))
    conf, hit, pred = np.asarray(conf), np.asarray(hit), np.asarray(pred)
    if conf.size == 0:
        return {}

    # Equal-WIDTH bins, the textbook definition. Equal-count bins would split a
    # run of identical confidences into pure-outcome bins and report calibration
    # error that is an artefact of the binning.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    ece = sum(abs(hit[m].mean() - conf[m].mean()) * int(m.sum())
              for a, b in zip(edges[:-1], edges[1:])
              if (m := (conf >= a) & (conf < b)).any())
    near = (conf >= gamma - 0.05) & (conf < gamma + 0.05)
    above = conf >= gamma

    # Per-class, because the aggregate hides the failure that matters. The
    # dismiss rule reads p_top as "probability this row is CLEAN" and acts on
    # it; on the ABO run that quantity was flat -- rows called CLEAN were
    # actually clean 68-83% of the time whether the router stated 0.55 or 0.92
    # -- while overall ECE sat at a healthy 0.027. A decision keyed to one
    # class needs that class measured (D19).
    per_class = {}
    for c in CLASSES:
        m = pred == c
        if not m.any():
            continue
        cc, hh = conf[m], hit[m]
        e = sum(abs(hh[b].mean() - cc[b].mean()) * int(b.sum())
                for a2, b2 in zip(edges[:-1], edges[1:])
                if (b := (cc >= a2) & (cc < b2)).any())
        per_class[c] = {"n": int(m.sum()), "ece": float(e / m.sum()),
                        "accuracy": float(hh.mean()),
                        "mean_confidence": float(cc.mean())}

    return {
        "n": int(conf.size),
        "ece": float(ece / conf.size),
        "gamma": float(gamma),
        "gap_at_gamma": float(hit[near].mean() - conf[near].mean()) if near.any() else None,
        "n_near_gamma": int(near.sum()),
        "share_above_gamma": float(above.mean()),
        "accuracy_above_gamma": float(hit[above].mean()) if above.any() else None,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# routing (Eq. 22 gate + policy)
# ---------------------------------------------------------------------------

@dataclass
class Route:
    row_id: str
    probs: dict
    error_type: str          # E1..E4 or CLEAN
    direction: str           # V2T | T2V | BOTH | NONE | HUMAN
    action: str
    tier: int
    reason: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def route(ev: dict, model: ArbiterModel, cfg: dict) -> Route:
    acfg = cfg.get("arbiter", {})
    gamma = float(acfg.get("gamma", 0.60))
    dismiss_thr = float(acfg.get("dismiss_threshold", 0.80))
    policy = acfg.get("t2v_policy", {}) or {}
    allowed_dirs = ev.get("allowed_directions") or ["V2T", "T2V", "HUMAN"]
    row_id = str(ev.get("row_id", ""))

    # missing-modality rows bypass the model (their visual features are void)
    if ev.get("image_missing"):
        return Route(row_id, {}, "E4", "T2V_ACQUIRE", "acquire_image", 3,
                     "missing image: acquisition or human (never V2T, F11)")
    if ev.get("text_missing"):
        return Route(row_id, {}, "E1", "V2T", "generate_text", 2,
                     "missing text: V2T generation tier")

    probs = model.predict_proba(featurize(ev))
    top = max(probs, key=probs.get)
    p_top = probs[top]

    # Eq. 22 gamma gate -> explicit E4 ambiguity state
    if p_top < gamma:
        return Route(row_id, probs, "E4", "HUMAN", "human_review", 3,
                     f"max p={p_top:.2f} < gamma={gamma}: ambiguous (E4)")

    if top == "CLEAN":
        # Safety guard: never dismiss while a strong contrary signal is live
        # (a confident-but-wrong CLEAN would silently drop a dirty row).
        #
        # D14: the threshold matters more than it looks. The Sieve flags a row
        # when a probe z-margin falls to `sieve.probes.z_margin` (2.0, i.e.
        # z <= -2.0); if this guard uses the same cut, every probe-flagged row
        # arrives with the guard already tripped and can never be dismissed --
        # the guard cancels the path it guards. Configurable so the trade can
        # be swept; the default reproduces the original behaviour exactly.
        # Which signals may veto is configurable, because a signal only belongs
        # here if it is *strong* evidence. `title_contradiction` was in this set
        # and measures 0.510 precision on ABO -- a coin flip, which blocks clean
        # and dirty rows in equal proportion and so contributes no discrimination
        # while suppressing the path entirely. Swept over 10,554 flagged rows
        # (D19): removing it takes clearances from 57 to 331 (1.8% -> 10.6% of
        # clean rows) with precision unchanged, 0.722 -> 0.720. It remains an
        # Arbiter *feature*, where the model can weigh it against everything
        # else rather than wielding an absolute veto.
        contrary_z = float(acfg.get("dismiss_contrary_z", -2.0))
        vetoes = acfg.get("dismiss_contrary_signals",
                          ["probe", "text_out_of_domain", "title_contradiction"])
        probes = ev.get("probes") or {}
        strong_probe = any((probes.get(f) or {}).get("z") is not None
                           and float(probes[f]["z"]) <= contrary_z for f in probes)
        contrary = ((strong_probe and "probe" in vetoes)
                    or (ev.get("title_contradiction") and "title_contradiction" in vetoes)
                    or (ev.get("text_out_of_domain") and "text_out_of_domain" in vetoes))
        if p_top >= dismiss_thr and not contrary:
            return Route(row_id, probs, "CLEAN", "NONE", "dismiss", 0,
                         f"routed clean with p={p_top:.2f}: sieve false positive")
        why = "contrary signal live" if contrary else f"p={p_top:.2f} < {dismiss_thr}"
        return Route(row_id, probs, "E4", "HUMAN", "human_review", 3,
                     f"clean but not dismissable ({why})")

    if top == "E1":
        if "V2T" not in allowed_dirs:
            return Route(row_id, probs, "E1", "HUMAN", "human_review", 3,
                         "E1 but V2T not allowed by modality constraints")
        return Route(row_id, probs, "E1", "V2T", "v2t_patch", 1, f"E1 p={p_top:.2f}")

    # E2 / E3 need T2V: consult the policy object (roadmap 2.6)
    cat_ok = ev.get("category") in (policy.get("allowed_categories") or [ev.get("category")])
    if "T2V" not in allowed_dirs or not cat_ok:
        return Route(row_id, probs, top, "HUMAN", "human_review", 3,
                     f"{top} but T2V blocked by policy/modality")
    if top == "E2":
        return Route(row_id, probs, "E2", "T2V", "t2v_replace_image", 1, f"E2 p={p_top:.2f}")
    return Route(row_id, probs, "E3", "BOTH", "t2v_replace_image_then_v2t", 2,
                 f"E3 p={p_top:.2f}: image first, then re-diagnose text")
