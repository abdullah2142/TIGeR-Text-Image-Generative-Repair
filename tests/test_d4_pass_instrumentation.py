"""D4 hard-proof instrumentation: did a row really get a second pass?

passes_used alone is ambiguous (a row can reach pass 2 for unrelated reasons),
so the real signal is both_directions_rows -- a row whose log contains both a
T2V and a V2T entry got both repair types across passes, which is exactly
what D4 made possible and could never happen before it.
"""

from tiger.eval.repair_ablation import format_repair_ablations

BASE_ROW = {"repaired": 1, "escalated": 0, "attr_accuracy": 1.0, "attr_total": 1,
           "t2v_accuracy": 1.0, "t2v_total": 1, "color_accuracy": 1.0, "v2t_total": 1}


def test_reports_multi_pass_and_both_direction_counts():
    results = {"full": {**BASE_ROW, "passes_used_distribution": {1: 8, 2: 2},
                        "multi_pass_rows": 2, "both_directions_rows": 2}}
    out = format_repair_ablations(results)
    assert "rows that reached pass 2+: 2" in out
    assert "the direct signature of D4 working): 2" in out
    assert "fired on real rows" in out


def test_zero_case_is_explained_not_alarming():
    results = {"full": {**BASE_ROW, "passes_used_distribution": {1: 10},
                        "multi_pass_rows": 0, "both_directions_rows": 0}}
    out = format_repair_ablations(results)
    assert "does not mean D4 is broken" in out


def test_missing_instrumentation_is_silently_skipped():
    """Older repair_ablations.json without this key must not crash the report."""
    results = {"full": BASE_ROW}
    out = format_repair_ablations(results)
    assert "D4 check" not in out
