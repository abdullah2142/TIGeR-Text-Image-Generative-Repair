# scripts/upgrade6_vlm_repair.py
from __future__ import annotations

import argparse
import json
import pandas as pd
from vlm_client import vlm_suggest

# -----------------------------
# Upgrade 6 safety controls
# -----------------------------
VLM_ALLOWED_FIELDS = {"color"}
VLM_ALLOWED_FLAG_REASONS = {"color_mismatch"}

# -----------------------------
# Utilities
# -----------------------------
def parse_attrs(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}

def _to_bool(x):
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    if isinstance(x, (int, float)):
        return bool(int(x))
    return str(x).strip().lower() in ("true", "1", "yes", "y", "t")

def validate_suggestion(sug, attrs, conf_thr=0.75):
    # required keys
    for k in ["field", "old_value", "new_value", "confidence", "evidence"]:
        if k not in sug:
            return False, f"missing_key:{k}"

    field = str(sug["field"])

    # Step 2 — strict field allowlist
    if field not in VLM_ALLOWED_FIELDS:
        return False, "field_not_allowed"

    if field not in attrs:
        return False, "field_not_in_attrs"

    # confidence
    try:
        c = float(sug["confidence"])
    except Exception:
        return False, "bad_confidence_type"

    if not (0.0 <= c <= 1.0):
        return False, "confidence_out_of_range"
    if c < conf_thr:
        return False, "confidence_below_threshold"

    newv = str(sug["new_value"]).strip()
    if newv == "":
        return False, "empty_new_value"

    # prevent no-op
    if newv.lower() == str(attrs.get(field, "")).strip().lower():
        return False, "no_change"

    # sanity: old value should match current
    cur = str(attrs.get(field, "")).strip().lower()
    oldv = str(sug["old_value"]).strip().lower()
    if cur and oldv and cur != oldv:
        return False, "old_value_mismatch"

    return True, "ok"

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--sieve_csv", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--out_log_csv", default="data/outputs/upgrade6_vlm_log.csv")
    ap.add_argument("--conf_threshold", type=float, default=0.75)
    ap.add_argument("--max_rows", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)
    sieve = pd.read_csv(args.sieve_csv)

    if "flagged" not in sieve.columns or "flag_autofix" not in sieve.columns:
        raise RuntimeError("sieve_csv must contain flagged and flag_autofix columns")

    sieve = sieve.copy()
    sieve["flagged_bool"] = sieve["flagged"].apply(_to_bool)
    sieve["flag_autofix_bool"] = sieve["flag_autofix"].apply(_to_bool)

    hard = sieve[
        (sieve["flagged_bool"] == True) &
        (sieve["flag_autofix_bool"] == False)
    ].head(args.max_rows)

    if "row_id" not in df.columns:
        raise RuntimeError("input_parquet must contain row_id column")

    df_idx = df.set_index("row_id", drop=False).copy()

    LOG_COLS = [
        "row_id", "status", "reject_reason", "applied",
        "field", "old_value", "new_value", "confidence",
        "evidence", "flag_reason", "image_path",
        "sim_score", "sim_attrs", "gap",
        "suggestion_json", "error",
    ]
    logs = []

    def log_row(**kw):
        row = {c: None for c in LOG_COLS}
        row.update(kw)
        logs.append(row)

    for _, r in hard.iterrows():
        row_id = str(r.get("row_id", "")).strip()
        flag_reason = str(r.get("flag_reason", "")).strip()

        # Step 3 — only target specific reasons
        if flag_reason not in VLM_ALLOWED_FLAG_REASONS:
            log_row(row_id=row_id, status="skip", reject_reason="not_vlm_target", applied=False)
            continue

        if row_id not in df_idx.index:
            log_row(row_id=row_id, status="skip", reject_reason="row_id_not_found", applied=False)
            continue

        cur = df_idx.loc[row_id]
        attrs = parse_attrs(cur.get("attributes"))
        image_path = cur.get("image_path", r.get("image_path", ""))

        payload = {
            "row_id": row_id,
            "image_path": image_path,
            "attributes": attrs,
            "flag_reason": flag_reason,
            "diagnostics": {
                "sim_score": r.get("sim_score"),
                "sim_attrs": r.get("sim_attrs"),
                "gap": r.get("gap"),
            },
            "editable_fields": list(attrs.keys()),
        }

        try:
            sug = vlm_suggest(payload)
        except Exception as e:
            log_row(
                row_id=row_id,
                status="error",
                reject_reason="vlm_call_failed",
                applied=False,
                error=str(e),
            )
            continue

        ok, reason = validate_suggestion(sug, attrs, args.conf_threshold)

        if ok:
            field = sug["field"]
            oldv = attrs.get(field)
            attrs[field] = sug["new_value"]

            # keep schema stable
            df_idx.at[row_id, "attributes"] = json.dumps(attrs, ensure_ascii=False)

            log_row(
                row_id=row_id,
                status="applied",
                reject_reason=None,
                applied=True,
                field=field,
                old_value=oldv,
                new_value=sug["new_value"],
                confidence=float(sug["confidence"]),
                evidence=sug["evidence"],
                flag_reason=flag_reason,
                image_path=image_path,
                sim_score=r.get("sim_score"),
                sim_attrs=r.get("sim_attrs"),
                gap=r.get("gap"),
                suggestion_json=json.dumps(sug, ensure_ascii=False),
                error=None,
            )
        else:
            log_row(
                row_id=row_id,
                status="rejected",
                reject_reason=reason,
                applied=False,
                field=sug.get("field"),
                old_value=sug.get("old_value"),
                new_value=sug.get("new_value"),
                confidence=sug.get("confidence"),
                evidence=sug.get("evidence"),
                flag_reason=flag_reason,
                image_path=image_path,
                sim_score=r.get("sim_score"),
                sim_attrs=r.get("sim_attrs"),
                gap=r.get("gap"),
                suggestion_json=json.dumps(sug, ensure_ascii=False),
                error=None,
            )

    out_df = df_idx.reset_index(drop=True)
    out_df.to_parquet(args.out_parquet, index=False)
    pd.DataFrame(logs).to_csv(args.out_log_csv, index=False)

    print("DONE")
    print("Output parquet:", args.out_parquet)
    print("Log csv:", args.out_log_csv)
    print("Hard cases processed:", len(hard))

if __name__ == "__main__":
    main()
