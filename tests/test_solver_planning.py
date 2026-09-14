"""Repair-planning tests: cost-minimal V2T patch, F14-safe T2V candidate pool."""

from pathlib import Path

import numpy as np
import pytest

from tiger.schema import load_schema
from tiger.solver import CandidatePool, plan_repair

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schema():
    return load_schema(ROOT / "configs/schema.yaml")


class Route:
    def __init__(self, direction):
        self.direction = direction


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_v2t_plan_single_field_from_evidence(schema):
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "blue", "pixel_color_confidence": 0.9,
          "pixel_color_region": "foreground",
          "probes": {"color": {"z": -3.0, "pred": "blue"}}}
    row = {"attributes": '{"color": "red", "material": "cotton", "pattern": "solid", "size": "M"}',
           "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, pool=None, cat_ids=None,
                       caption_emb=None, schema=schema)
    assert plan.plannable
    assert plan.patch == {"color": "blue"}
    assert plan.cost == 1.0


def test_v2t_escalates_on_estimator_disagreement(schema):
    """B6: pixel estimator and CLIP probe disagree -> escalate, don't coin-flip.

    Before B6, _corrected_value's if/else silently committed the pixel value
    (confidence >= 0.55) without ever checking whether the probe agreed.
    """
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "blue", "pixel_color_confidence": 0.9,
          "pixel_color_region": "foreground",
          "probes": {"color": {"z": -3.0, "pred": "red"}}}
    row = {"attributes": '{"color": "green", "material": "cotton", "pattern": "solid", "size": "M"}',
           "title": "Green Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, pool=None, cat_ids=None,
                       caption_emb=None, schema=schema)
    assert not plan.plannable
    assert "disagree" in plan.notes
    # Diagnostics must survive the escalation for downstream attribution (B0/B6).
    assert plan.pixel_value == "blue"
    assert plan.probe_value == "red"
    assert plan.estimators_agree is False


def test_v2t_plans_normally_when_only_one_estimator_applies(schema):
    """A non-colour field has no pixel estimator at all -- that's not a
    disagreement (nothing to disagree with), so it must still plan normally."""
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "material", "probes": {"material": {"z": -3.0, "pred": "denim"}}}
    row = {"attributes": '{"color": "blue", "material": "cotton", "pattern": "solid", "size": "M"}',
           "title": "Blue Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, pool=None, cat_ids=None,
                       caption_emb=None, schema=schema)
    assert plan.plannable
    assert plan.patch == {"material": "denim"}


def test_v2t_unplannable_when_no_valid_value(schema):
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "probes": {"color": {"z": -3.0, "pred": ""}}}
    row = {"attributes": '{"color": "red"}', "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, None, None, None, schema)
    assert not plan.plannable


def test_v2t_skips_noop_patch(schema):
    # image already agrees with the declared value -> no repair needed
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "red", "pixel_color_confidence": 0.9,
          "pixel_color_region": "foreground",
          "probes": {"color": {"z": -3.0, "pred": "red"}}}
    row = {"attributes": '{"color": "red"}', "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, None, None, None, schema)
    assert not plan.plannable


def test_candidate_pool_excludes_own_product():
    emb = np.stack([_unit(np.array([1.0, 0.0])),   # p1 (own)
                    _unit(np.array([0.98, 0.1])),  # p2
                    _unit(np.array([0.0, 1.0]))])  # p3
    pool = CandidatePool(emb, ["p1", "p2", "p3"],
                         ["a.jpg", "b.jpg", "c.jpg"], np.array([True, True, True]))
    # caption points at p1's direction; own product excluded -> must pick p2
    res = pool.best_for_text(emb[0], exclude_product="p1")
    assert res is not None
    j, _ = res
    assert j == 1


def test_t2v_plan_uses_pool(schema):
    emb = np.stack([_unit(np.array([1.0, 0.0])),
                    _unit(np.array([0.9, 0.2])),
                    _unit(np.array([0.0, 1.0]))])
    pool = CandidatePool(emb, ["r1", "r2", "r3"],
                         ["own.jpg", "cand.jpg", "other.jpg"], np.array([True, True, True]))
    cat_ids = np.array(["shirts", "shirts", "shirts"])
    ev = {"row_id": "r1", "product_id": "r1", "category": "shirts"}
    plan = plan_repair(ev, Route("T2V"), {"attributes": "{}"}, pool, cat_ids,
                       caption_emb=emb[0], schema=schema)
    assert plan.plannable
    assert plan.candidate_product_id == "r2"      # own product held out (F14)
    assert plan.candidate_image_path == "cand.jpg"


def test_t2v_unplannable_when_pool_empty(schema):
    emb = np.stack([_unit(np.array([1.0, 0.0]))])
    pool = CandidatePool(emb, ["r1"], ["own.jpg"], np.array([True]))
    ev = {"row_id": "r1", "product_id": "r1", "category": "shirts"}
    plan = plan_repair(ev, Route("T2V"), {"attributes": "{}"}, pool,
                       np.array(["shirts"]), emb[0], schema)
    assert not plan.plannable


def test_pixel_value_used_when_the_product_was_localised(schema):
    """B4: with no probe value, the pixel estimator is the only candidate --
    and it is allowed to supply one when it actually found the product."""
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "blue", "pixel_color_confidence": 0.9,
          "pixel_color_region": "foreground", "probes": {"color": {"z": -3.0, "pred": ""}}}
    row = {"attributes": '{"color": "red"}', "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, None, None, None, schema)
    assert plan.plannable
    assert plan.patch == {"color": "blue"}
    assert plan.value_source == "pixel"
    assert plan.pixel_region == "foreground"


def test_pixel_value_refused_when_the_product_was_not_localised(schema):
    """B4: `center_box` means the estimator could not find a studio ground and
    measured a fixed box instead, so its 0.9 share is 0.9 of an unknown region.
    A high share is not a reason to write that value into the catalogue."""
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "blue", "pixel_color_confidence": 0.9,
          "pixel_color_region": "center_box", "probes": {"color": {"z": -3.0, "pred": ""}}}
    row = {"attributes": '{"color": "red"}', "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, None, None, None, schema)
    assert not plan.plannable


def test_declined_pixel_estimate_is_not_a_disagreement(schema):
    """B9: "multicolour" is the pixel estimator reporting that no single colour
    dominates -- it is declining, not answering. Treating it as a value made the
    row look like a two-estimator conflict and escalate under B6, when only the
    probe ever spoke. A colour row where the estimator declined is the same
    situation as material or pattern, which have no pixel estimator at all."""
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "multicolour",
          "pixel_color_confidence": 0.4, "pixel_color_region": "foreground",
          "probes": {"color": {"z": -3.0, "pred": "blue"}}}
    row = {"attributes": '{"color": "red"}', "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, None, None, None, schema)
    assert plan.plannable
    assert plan.patch == {"color": "blue"}      # the probe decides alone
    assert plan.value_source == "probe"
    assert plan.pixel_declined is True
    assert plan.pixel_value == ""               # no opinion, not a conflicting one
    assert plan.estimators_agree is False       # nothing to agree with


def test_unknown_pixel_estimate_is_also_a_decline(schema):
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "unknown",
          "pixel_color_confidence": 0.0, "pixel_color_region": "foreground",
          "probes": {"color": {"z": -3.0, "pred": "blue"}}}
    row = {"attributes": '{"color": "red"}', "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, None, None, None, schema)
    assert plan.plannable and plan.pixel_declined is True


def test_a_real_disagreement_still_escalates(schema):
    """The B6 escalation must survive: two actual values that differ."""
    ev = {"row_id": "r1", "category": "shirts", "product_id": "r1",
          "loo_top_field": "color", "pixel_color": "green",
          "pixel_color_confidence": 0.9, "pixel_color_region": "foreground",
          "probes": {"color": {"z": -3.0, "pred": "blue"}}}
    row = {"attributes": '{"color": "red"}', "title": "Red Shirt", "category": "shirts"}
    plan = plan_repair(ev, Route("V2T"), row, None, None, None, schema)
    assert not plan.plannable
    assert "disagree" in plan.notes
    assert plan.pixel_declined is False
