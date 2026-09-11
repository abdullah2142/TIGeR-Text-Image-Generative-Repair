"""D4: an accepted repair must stay 'pending' and re-enter the next pass.

Before D4, run_repair_cycle set final_status="repaired" the instant a V2T or
T2V patch was accepted, immediately excluding the row from active_mask in
every later pass -- so a row needing BOTH an image swap and a text fix (E3)
only ever got the image swap; the text-fix half of run_repair_cycle's own
documented "the cycle re-diagnoses so a row needing two fixes gets two"
behaviour had never actually run.

_promote_clean_pending is the fix's core: it decides whether a still-"pending"
row has earned a terminal "repaired" status yet, purely from whether the
sieve's fresh re-run still flags it. Testing it directly (rather than driving
the full CLIP/encoder pipeline) keeps this fast and dependency-free, matching
how the rest of this suite treats deep integration points (see
test_fusion_wiring.py).
"""

import pandas as pd

from tiger.repair import RepairOutcome, _promote_clean_pending


def _flagged_df(flagged_ids: set[str], all_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"row_id": all_ids,
                         "flagged": [rid in flagged_ids for rid in all_ids]})


def test_pending_row_promoted_once_no_longer_flagged():
    """One fix was enough (E1/E2 case): the row drops out of the flagged set
    on the very next pass and must be closed out as repaired, not left pending."""
    outcomes = {"r1": RepairOutcome("r1", "pending")}
    _promote_clean_pending(outcomes, _flagged_df(flagged_ids=set(), all_ids=["r1", "r2"]))
    assert outcomes["r1"].final_status == "repaired"


def test_pending_row_stays_pending_while_still_flagged():
    """The two-fix case (E3/BOTH): pass 1 fixed the image, but the row is
    still flagged (bad text against the new image) -- it must stay pending
    so the main loop's active_mask picks it up again next pass, not get
    closed out early with the text half never addressed."""
    outcomes = {"r1": RepairOutcome("r1", "pending")}
    _promote_clean_pending(outcomes, _flagged_df(flagged_ids={"r1"}, all_ids=["r1", "r2"]))
    assert outcomes["r1"].final_status == "pending"


def test_terminal_rows_are_never_touched():
    """An already-escalated/dismissed row must not be reconsidered just
    because it happens to still appear in the flagged/unflagged set."""
    outcomes = {"r1": RepairOutcome("r1", "escalated"), "r2": RepairOutcome("r2", "dismissed")}
    _promote_clean_pending(outcomes, _flagged_df(flagged_ids=set(), all_ids=["r1", "r2"]))
    assert outcomes["r1"].final_status == "escalated"
    assert outcomes["r2"].final_status == "dismissed"


def test_row_absent_from_this_pass_frame_is_left_pending():
    """A row missing from the current flagged frame (e.g. filtered upstream)
    must not be silently promoted -- absence is not evidence of cleanliness."""
    outcomes = {"r1": RepairOutcome("r1", "pending")}
    _promote_clean_pending(outcomes, _flagged_df(flagged_ids=set(), all_ids=["r2", "r3"]))
    assert outcomes["r1"].final_status == "pending"
