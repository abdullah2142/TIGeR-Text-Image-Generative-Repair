"""B6 emptied the B0 report's own disagreement bucket -- fix it back.

_evaluate_run only ever scanned final_status == "repaired" outcomes. Since B6
(escalate on estimator disagreement) landed, every disagreement case is
final_status == "escalated" and never reached that scan -- so
format_v2t_diagnostics's "disagree" bucket, the one comparison the whole
report exists to make (does pixel or probe tend to be right when they
differ?), was silently always empty. Cases with correct=None represent an
escalated disagreement: nothing was written, but pixel_correct/probe_correct
(what *would* have been written by each side) are still scored.
"""

from tiger.eval.repair_ablation import format_v2t_diagnostics

CASES = [
    # committed, estimators agreed
    {"config": "full", "row_id": "r1", "true_color": "blue", "written_color": "blue",
     "correct": 1, "value_source": "pixel", "pixel_value": "blue", "pixel_conf": 0.9,
     "probe_value": "blue", "estimators_agree": True, "pixel_correct": 1, "probe_correct": 1},
    # escalated, estimators disagreed -- nothing written
    {"config": "full", "row_id": "r2", "true_color": "red", "written_color": None,
     "correct": None, "value_source": "", "pixel_value": "blue", "pixel_conf": 0.8,
     "probe_value": "red", "estimators_agree": False, "pixel_correct": 0, "probe_correct": 1},
    {"config": "full", "row_id": "r3", "true_color": "green", "written_color": None,
     "correct": None, "value_source": "", "pixel_value": "green", "pixel_conf": 0.7,
     "probe_value": "yellow", "estimators_agree": False, "pixel_correct": 1, "probe_correct": 0},
]


def test_escalated_cases_do_not_crash_the_report():
    out = format_v2t_diagnostics({"_v2t_cases": CASES}, config="full")
    assert "no scored V2T cases" not in out


def test_committed_count_excludes_escalated_rows():
    out = format_v2t_diagnostics({"_v2t_cases": CASES}, config="full")
    assert "accuracy as shipped (committed only) : 100.0%  (1/1)" in out


def test_disagree_bucket_is_populated_not_empty():
    """The exact bug: before the fix, disagree was always [] post-B6."""
    out = format_v2t_diagnostics({"_v2t_cases": CASES}, config="full")
    assert "disagree :   2 rows" in out
    assert "escalated, nothing committed" in out
    # pixel right once, probe right once, of 2 disagreement rows -> 50% each
    assert "pixel right 50.0%, probe right 50.0%" in out


def test_counterfactual_spans_committed_and_escalated_rows():
    """The estimator counterfactual should score every attempt (3 scorable),
    not just the 1 committed row -- that's the point of including escalations."""
    out = format_v2t_diagnostics({"_v2t_cases": CASES}, config="full")
    assert "always pixel  : 66.7%  (2/3 scorable)" in out
    assert "always probe  : 66.7%  (2/3 scorable)" in out
    assert "either right  : 100.0%  (3/3)" in out
