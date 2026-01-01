from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def safe_json_load(s) -> dict:
    if s is None:
        return {}
    if isinstance(s, dict):
        return s
    if isinstance(s, float) and pd.isna(s):
        return {}
    s = str(s).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def rebuild_canonical_text(title: str, category: str, attrs: dict) -> str:
    attrs_for_text = ", ".join([f"{k}={v}" for k, v in attrs.items()])
    return f"{title}. Category: {category}. Attributes: {attrs_for_text}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True, help="Input rows parquet (e.g., toy_noisy_rows.parquet)")
    ap.add_argument("--plan_csv", required=True, help="arbiter_plan_*.csv")
    ap.add_argument("--out_parquet", default="", help="Output repaired parquet path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    in_parquet = Path(args.in_parquet)
    in_parquet = in_parquet if in_parquet.is_absolute() else (root / in_parquet)
    in_parquet = in_parquet.resolve()

    plan_csv = Path(args.plan_csv)
    plan_csv = plan_csv if plan_csv.is_absolute() else (root / plan_csv)
    plan_csv = plan_csv.resolve()

    if not in_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {in_parquet}")
    if not plan_csv.exists():
        raise FileNotFoundError(f"Missing plan csv: {plan_csv}")

    df = pd.read_parquet(in_parquet)
    plan = pd.read_csv(plan_csv)

    # -------------------------------
    # Upgrade 1B: cycle-safe snapshot
    # -------------------------------
    # If plans contain swap-cycles (A<-B, B<-A) or chains, applying row-by-row can overwrite donors.
    # So we snapshot each row's original image file path BEFORE any modifications.
    donor_image_snapshot: dict[str, Path | None] = {}

    row_ids = df["row_id"].astype(str).values if "row_id" in df.columns else []
    img_paths = df["image_path"].astype(str).values if "image_path" in df.columns else []

    for rid, img_rel in zip(row_ids, img_paths):
        img_rel = str(img_rel).strip()
        if not img_rel or img_rel.lower() == "nan":
            donor_image_snapshot[rid] = None
            continue

        # normalize windows backslashes -> forward slashes
        img_rel_norm = img_rel.replace("\\", "/")
        p = Path(img_rel_norm)

        # make absolute against repo root if needed
        abs_p = p if p.is_absolute() else (root / p)
        donor_image_snapshot[rid] = abs_p.resolve()
        
    # Ensure columns exist
    for col in ["row_id", "title", "category", "attributes", "canonical_text", "image_path"]:
        if col not in df.columns:
            df[col] = ""

    # ✅ Repair metadata columns with correct types
    if "repaired" not in df.columns:
        df["repaired"] = False
    else:
        df["repaired"] = df["repaired"].fillna(False).astype(bool)

    for col in ["repair_action", "repair_notes", "repair_source_row_id"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    # Index for fast lookup
    idx_by_rowid = {str(rid): i for i, rid in enumerate(df["row_id"].astype(str).values)}

    repaired_images_dir = (root / "data/processed/repaired_images").resolve()
    repaired_images_dir.mkdir(parents=True, exist_ok=True)

    log_rows = []

    for _, p in plan.iterrows():
        row_id = str(p.get("row_id", "")).strip()
        action = str(p.get("action", "")).strip()
        proposed_fix = safe_json_load(p.get("proposed_fix", "{}"))

        if row_id not in idx_by_rowid:
            log_rows.append({"row_id": row_id, "status": "fail", "reason": "row_id_not_found", "action": action})
            continue

        i = idx_by_rowid[row_id]
        status = "ok"
        reason = ""

        try:
            if action == "replace_image_from_row":
                cand = str(proposed_fix.get("image_replacement_candidate_row_id", "")).strip()
                if not cand or cand not in idx_by_rowid:
                    status = "fail"
                    reason = "candidate_missing_or_not_found"
                else:
                    # Upgrade 1B: use snapshot instead of df.at[j,"image_path"]
                    src_path = donor_image_snapshot.get(cand)

                    if src_path is None:
                        status = "fail"
                        reason = "candidate_has_no_image_path_snapshot"
                    elif not src_path.exists():
                        status = "fail"
                        reason = f"candidate_image_file_missing_snapshot:{src_path.as_posix()}"
                    else:
                        dst_path = repaired_images_dir / f"{row_id}.jpg"
                        shutil.copy2(src_path, dst_path)

                        # Store as relative POSIX path
                        df.at[i, "image_path"] = dst_path.relative_to(root).as_posix()
                        df.at[i, "repaired"] = True
                        df.at[i, "repair_action"] = "replace_image_from_row"
                        df.at[i, "repair_source_row_id"] = cand
                        df.at[i, "repair_notes"] = str(p.get("notes", ""))

            elif action == "apply_text_patch":
                patch = proposed_fix.get("text_patch", {})
                if not isinstance(patch, dict) or not patch:
                    status = "fail"
                    reason = "missing_text_patch"
                else:
                    attrs = safe_json_load(df.at[i, "attributes"])
                    title = str(df.at[i, "title"])
                    category = str(df.at[i, "category"])

                    for k, v in patch.items():
                        k = str(k)
                        if k == "title":
                            title = str(v)
                        elif k.startswith("attributes."):
                            sub = k.split(".", 1)[1]
                            attrs[sub] = v
                        else:
                            if k in df.columns:
                                df.at[i, k] = v

                    df.at[i, "title"] = title
                    df.at[i, "attributes"] = json.dumps(attrs, ensure_ascii=False)
                    df.at[i, "canonical_text"] = rebuild_canonical_text(title, category, attrs)

                    df.at[i, "repaired"] = True
                    df.at[i, "repair_action"] = "apply_text_patch"
                    df.at[i, "repair_source_row_id"] = ""
                    df.at[i, "repair_notes"] = str(p.get("notes", ""))

            elif action == "human_review":
                df.at[i, "repaired"] = False
                df.at[i, "repair_action"] = "human_review"
                df.at[i, "repair_notes"] = str(p.get("notes", ""))

            else:
                status = "skip"
                reason = f"action_not_supported:{action}"

        except Exception as e:
            status = "fail"
            reason = f"exception:{type(e).__name__}:{e}"

        log_rows.append({"row_id": row_id, "status": status, "reason": reason, "action": action})

    # ✅ Final dtype safety before saving
    df["repaired"] = df["repaired"].fillna(False).astype(bool)
    df["repair_action"] = df["repair_action"].fillna("").astype(str)
    df["repair_notes"] = df["repair_notes"].fillna("").astype(str)
    df["repair_source_row_id"] = df["repair_source_row_id"].fillna("").astype(str)

    # Output paths
    stem = in_parquet.stem
    out_parquet = Path(args.out_parquet) if args.out_parquet else (root / "data/processed" / f"repaired_{stem}.parquet")
    out_parquet = out_parquet if out_parquet.is_absolute() else (root / out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_parquet, index=False)

    log_df = pd.DataFrame(log_rows)
    log_csv = (root / "data/outputs" / f"repair_log_{stem}.csv").resolve()
    log_df.to_csv(log_csv, index=False)

    print("✅ Repairs applied")
    print("Input parquet:", in_parquet)
    print("Plan:", plan_csv)
    print("Output parquet:", out_parquet)
    print("Repair log:", log_csv)
    print("\nLog summary:")
    print(log_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
