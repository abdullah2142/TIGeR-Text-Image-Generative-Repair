from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def safe_json_load(s: str) -> dict:
    if s is None:
        return {}
    s = str(s).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue_csv", required=True, help="arbiter_queue_*.csv from mismatch_analyzer")
    ap.add_argument("--out_csv", default="", help="Optional output path for arbiter plan csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    queue_csv = Path(args.queue_csv)
    queue_csv = queue_csv if queue_csv.is_absolute() else (root / queue_csv)
    queue_csv = queue_csv.resolve()

    if not queue_csv.exists():
        raise FileNotFoundError(f"Missing queue CSV: {queue_csv}")

    dfq = pd.read_csv(queue_csv)

    plan_rows = []
    for _, r in dfq.iterrows():
        row_id = str(r.get("row_id", "")).strip()
        suggested = str(r.get("suggested_direction", "")).strip().upper()
        error_type = str(r.get("error_type", "")).strip()
        mismatch_aspects = str(r.get("mismatch_aspects", "")).strip()

        proposed_fix = safe_json_load(str(r.get("proposed_fix", "")))

        # Default plan
        action = "human_review"
        tier = 0
        cost_units = 0.0
        notes = ""

        if suggested == "T2V":
            # MVP T2V = replace image using candidate row's image
            cand = proposed_fix.get("image_replacement_candidate_row_id", "")
            if cand:
                action = "replace_image_from_row"
                tier = 1
                cost_units = 1.0
                notes = f"Replace image using candidate row {cand}"
            else:
                action = "t2v_generate_image_unimplemented"
                tier = 3
                cost_units = 10.0
                notes = "Would require generative T2V editing; not implemented in MVP."

        elif suggested == "V2T":
            # MVP V2T = patch text / attributes
            patch = proposed_fix.get("text_patch", {})
            if patch:
                action = "apply_text_patch"
                tier = 1
                cost_units = 0.5
                notes = "Apply text_patch to row fields/attributes."
            else:
                action = "v2t_generate_text_unimplemented"
                tier = 2
                cost_units = 4.0
                notes = "Would require VLM-based text generation; not implemented in MVP."

        else:
            action = "human_review"
            tier = 0
            cost_units = 0.0
            notes = "Low confidence or missing modalities."

        plan_rows.append(
            {
                "row_id": row_id,
                "error_type": error_type,
                "mismatch_aspects": mismatch_aspects,
                "suggested_direction": suggested,
                "action": action,
                "tier": tier,
                "cost_units": cost_units,
                "proposed_fix": json.dumps(proposed_fix, ensure_ascii=False),
                "notes": notes,
            }
        )

    plan = pd.DataFrame(plan_rows)

    stem = queue_csv.stem.replace("arbiter_queue_", "")
    out_csv = Path(args.out_csv) if args.out_csv else (root / "data/outputs" / f"arbiter_plan_{stem}.csv")
    out_csv = out_csv if out_csv.is_absolute() else (root / out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    plan.to_csv(out_csv, index=False)

    print("✅ Arbiter complete")
    print("Input queue:", queue_csv)
    print("Output plan:", out_csv)
    print("\nPlan preview:")
    print(plan.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
