"""Arbiter unit tests: gamma gate, dismiss guard, policy gate, model round-trip."""

import numpy as np

from tiger import arbiter as A

CFG = {"arbiter": {"gamma": 0.60, "dismiss_threshold": 0.80,
                   "t2v_policy": {"allowed_categories": ["shirts"]}}}


def constant_model(target: str, conf: float = 0.95) -> A.ArbiterModel:
    """Model whose softmax always favours `target` with ~conf probability."""
    n = len(A.FEATURES)
    coef = [[0.0] * n for _ in A.CLASSES]
    k = len(A.CLASSES)
    other = np.log((1 - conf) / (k - 1))
    intercept = [np.log(conf) if c == target else float(other) for c in A.CLASSES]
    return A.ArbiterModel(A.FEATURES, A.CLASSES, [0.0] * n, [1.0] * n, coef, intercept)


BASE_EV = {"row_id": "r1", "category": "shirts", "allowed_directions": ["V2T", "T2V", "HUMAN"],
           "probes": {}, "image_missing": False, "text_missing": False}


def test_gamma_gate_escalates_to_e4():
    m = constant_model("E1", conf=0.40)  # below gamma
    r = A.route(dict(BASE_EV), m, CFG)
    assert r.error_type == "E4"
    assert r.action == "human_review"


def test_e1_routes_v2t():
    r = A.route(dict(BASE_EV), constant_model("E1"), CFG)
    assert (r.error_type, r.direction, r.action) == ("E1", "V2T", "v2t_patch")


def test_e2_routes_t2v():
    r = A.route(dict(BASE_EV), constant_model("E2"), CFG)
    assert (r.direction, r.action) == ("T2V", "t2v_replace_image")


def test_e3_routes_image_first_then_text():
    r = A.route(dict(BASE_EV), constant_model("E3"), CFG)
    assert r.direction == "BOTH"
    assert r.action == "t2v_replace_image_then_v2t"


def test_policy_gate_blocks_t2v_for_disallowed_category():
    ev = dict(BASE_EV, category="shoes")  # policy only allows shirts
    r = A.route(ev, constant_model("E2"), CFG)
    assert r.action == "human_review"


def test_confident_clean_dismissed():
    r = A.route(dict(BASE_EV), constant_model("CLEAN", 0.95), CFG)
    assert r.action == "dismiss"


def test_dismiss_guard_on_contrary_probe():
    ev = dict(BASE_EV, probes={"color": {"z": -3.5, "margin": -0.05, "pred": "blue"}})
    r = A.route(ev, constant_model("CLEAN", 0.95), CFG)
    assert r.action == "human_review"  # strong contrary signal blocks dismissal


def test_missing_image_never_v2t():
    ev = dict(BASE_EV, image_missing=True, allowed_directions=["T2V_ACQUIRE", "HUMAN"])
    r = A.route(ev, constant_model("E1"), CFG)
    assert r.direction == "T2V_ACQUIRE"
    assert r.action == "acquire_image"


def test_model_json_roundtrip():
    m = constant_model("E2")
    m2 = A.ArbiterModel.from_json(m.to_json())
    x = np.zeros(len(A.FEATURES))
    assert m.predict_proba(x) == m2.predict_proba(x)


def test_probs_sum_to_one():
    m = constant_model("E1")
    p = m.predict_proba(np.random.default_rng(0).normal(size=len(A.FEATURES)))
    assert abs(sum(p.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# D1 · the gamma gate reads predict_proba as a probability -- measure that
# ---------------------------------------------------------------------------

def test_calibration_report_is_zero_for_a_perfectly_calibrated_router():
    """A model that says 0.80 and is right 80% of the time has no calibration
    error, whatever its accuracy."""
    m = constant_model("E1", conf=0.80)
    ev = [dict(BASE_EV) for _ in range(100)]
    labels = ["mutate_text"] * 80 + ["clean"] * 20   # mutate_text -> E1
    rep = A.calibration_report(m, ev, labels, gamma=0.60)
    assert rep["n"] == 100
    assert rep["ece"] < 0.01
    assert rep["accuracy_above_gamma"] == 0.80


def test_calibration_report_catches_an_overconfident_router():
    m = constant_model("E1", conf=0.95)
    ev = [dict(BASE_EV) for _ in range(100)]
    labels = ["mutate_text"] * 50 + ["clean"] * 50
    rep = A.calibration_report(m, ev, labels, gamma=0.60)
    assert rep["ece"] > 0.40          # says 0.95, is right half the time
    assert rep["share_above_gamma"] == 1.0


def test_calibration_report_reports_the_gap_where_the_gate_sits():
    """The aggregate can look fine while the router is wrong exactly at gamma."""
    m = constant_model("E1", conf=0.62)   # inside the +/-0.05 window around 0.60
    ev = [dict(BASE_EV) for _ in range(100)]
    labels = ["mutate_text"] * 30 + ["clean"] * 70
    rep = A.calibration_report(m, ev, labels, gamma=0.60)
    assert rep["n_near_gamma"] == 100
    assert rep["gap_at_gamma"] < -0.30    # states 0.62, delivers 0.30


# ---------------------------------------------------------------------------
# D19 · only strong evidence may veto a dismissal
# ---------------------------------------------------------------------------

def _cfg(**over):
    a = {"gamma": 0.60, "dismiss_threshold": 0.80, "dismiss_contrary_z": -3.0,
         "t2v_policy": {"allowed_categories": ["shirts"]}}
    a.update(over)
    return {"arbiter": a}


def test_title_contradiction_does_not_veto_a_dismissal_by_default():
    """It measures 0.510 precision on ABO. A coin flip blocks clean and dirty
    rows in equal proportion, so as a veto it suppresses the path without
    discriminating -- 57 clearances against 331 with it removed, same precision."""
    ev = dict(BASE_EV, title_contradiction=True)
    r = A.route(ev, constant_model("CLEAN", 0.95),
                _cfg(dismiss_contrary_signals=["probe", "text_out_of_domain"]))
    assert r.action == "dismiss"


def test_it_still_vetoes_when_explicitly_configured_to():
    ev = dict(BASE_EV, title_contradiction=True)
    r = A.route(ev, constant_model("CLEAN", 0.95),
                _cfg(dismiss_contrary_signals=["probe", "title_contradiction"]))
    assert r.action == "human_review"


def test_a_strong_probe_still_vetoes():
    ev = dict(BASE_EV, probes={"color": {"z": -4.0, "margin": -0.05, "pred": "blue"}})
    r = A.route(ev, constant_model("CLEAN", 0.95),
                _cfg(dismiss_contrary_signals=["probe", "text_out_of_domain"]))
    assert r.action == "human_review"


def test_out_of_domain_still_vetoes():
    ev = dict(BASE_EV, text_out_of_domain=True)
    r = A.route(ev, constant_model("CLEAN", 0.95),
                _cfg(dismiss_contrary_signals=["probe", "text_out_of_domain"]))
    assert r.action == "human_review"


def test_the_shipped_config_excludes_the_title_signal():
    from tiger import cli
    v = cli.load_cfg()["arbiter"]["dismiss_contrary_signals"]
    assert "title_contradiction" not in v
    assert "probe" in v and "text_out_of_domain" in v


def test_calibration_is_reported_per_class():
    """D19: the dismiss rule keys on p_top for ONE class, so that class needs
    its own calibration. An aggregate ECE of 0.027 hid a CLEAN head that was
    flat -- 68-83% actually clean whether it stated 0.55 or 0.92."""
    m = constant_model("E1", conf=0.95)
    ev = [dict(BASE_EV) for _ in range(100)]
    labels = ["mutate_text"] * 50 + ["clean"] * 50
    rep = A.calibration_report(m, ev, labels, gamma=0.60)
    assert "per_class" in rep
    assert rep["per_class"]["E1"]["n"] == 100
    assert rep["per_class"]["E1"]["accuracy"] == 0.5
    assert rep["per_class"]["E1"]["ece"] > 0.40
