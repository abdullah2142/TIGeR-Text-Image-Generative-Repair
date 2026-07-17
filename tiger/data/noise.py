"""Error injection with self-verifying ground truth (roadmap 1.1, findings F12/F4).

Fixes over the legacy make_toy_noisy.py:

  - a row could be LABELLED swap_image while the swap silently did not happen
    (donor index == own index, or donor was a duplicate of the same product
    carrying the identical image). Here donors must come from a DIFFERENT
    product and the image path must actually change; failures re-draw.
  - replacement values are always drawn from Omega_j minus the current value.
  - every injected error logs intended vs applied change (old/new value, donor
    row/product) so ground truth is self-verifying; a post-pass asserts that
    every dirty row differs from its clean original and every clean row does
    not. No-op noise is a hard failure, not a silent mislabel.
  - error types are configurable per-rate and include the E3 mixed generator
    and "subtle" variants (same-category swap, near-colour flip) the review
    asked for (A3.3).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tiger import text_views
from tiger.schema import Schema

# perceptually adjacent colours for the subtle near-colour generator
NEAR_COLORS = {
    "red": ["orange", "pink"],
    "orange": ["red", "yellow", "brown"],
    "yellow": ["orange"],
    "green": ["blue"],
    "blue": ["purple", "green"],
    "purple": ["blue", "pink"],
    "pink": ["red", "purple"],
    "brown": ["orange", "red"],
    "black": ["gray"],
    "gray": ["black", "white"],
    "white": ["gray"],
}

# error type -> (label, is_text_error, is_image_error)
ERROR_TYPES = {
    "swap_image": ("swap_image", False, True),
    "swap_image_same_category": ("swap_image", False, True),
    "color_flip": ("mutate_text", True, False),
    "near_color_flip": ("mutate_text", True, False),
    "material_flip": ("mutate_text", True, False),
    "attribute_drop": ("mutate_text", True, False),
    "title_contradiction": ("mutate_text", True, False),
    "mixed_swap_color": ("mixed", True, True),
    "missing_image": ("missing_image", False, True),
}


@dataclass
class InjectionRecord:
    row_id: str
    product_id: str
    subtype: str
    label: str
    field: str = ""
    old_value: str = ""
    new_value: str = ""
    donor_row_id: str = ""
    donor_product_id: str = ""
    verified_changed: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class NoiseResult:
    df: pd.DataFrame
    audit: pd.DataFrame
    meta: dict = field(default_factory=dict)


_replace_color_word = text_views.replace_color_word_safe
_title_color_word = text_views.find_color_word


class NoiseInjector:
    def __init__(self, schema: Schema, cfg: dict, rng: random.Random):
        self.schema = schema
        self.cfg = cfg
        self.rng = rng
        self.colors = [c for c in schema.domain("color") if c != "multicolour"]
        self.materials = schema.domain("material")

    # ---------- individual generators; each returns an InjectionRecord ----------

    def _flip_enum(self, row: pd.Series, fld: str, pool: list[str], subtype: str) -> InjectionRecord | None:
        attrs = text_views.parse_attrs(row["attributes"])
        old = str(attrs.get(fld, "")).lower()
        choices = [c for c in pool if c != old]
        if not choices:
            return None
        new = self.rng.choice(choices)
        attrs[fld] = new

        title = str(row["title"])
        brands = {str(text_views.parse_attrs(row["attributes"]).get("brand", ""))}
        if fld == "color" and old:
            title = _replace_color_word(title, old, new, brands)

        row["attributes"] = json.dumps(attrs, ensure_ascii=False)
        row["title"] = title
        row["canonical_text"] = text_views.canonical_text(title, str(row["category"]), attrs)
        return InjectionRecord(str(row["row_id"]), str(row["product_id"]), subtype,
                               ERROR_TYPES[subtype][0], field=fld, old_value=old, new_value=new)

    def color_flip(self, row: pd.Series) -> InjectionRecord | None:
        return self._flip_enum(row, "color", self.colors, "color_flip")

    def near_color_flip(self, row: pd.Series) -> InjectionRecord | None:
        attrs = text_views.parse_attrs(row["attributes"])
        old = str(attrs.get("color", "")).lower()
        pool = NEAR_COLORS.get(old, self.colors)
        return self._flip_enum(row, "color", pool, "near_color_flip")

    def material_flip(self, row: pd.Series) -> InjectionRecord | None:
        return self._flip_enum(row, "material", self.materials, "material_flip")

    def attribute_drop(self, row: pd.Series) -> InjectionRecord | None:
        attrs = text_views.parse_attrs(row["attributes"])
        old = str(attrs.pop("color", "") or "")
        if not old:
            return None
        row["attributes"] = json.dumps(attrs, ensure_ascii=False)
        row["canonical_text"] = text_views.canonical_text(str(row["title"]), str(row["category"]), attrs)
        return InjectionRecord(str(row["row_id"]), str(row["product_id"]), "attribute_drop",
                               "mutate_text", field="color", old_value=old, new_value="")

    def title_contradiction(self, row: pd.Series) -> InjectionRecord | None:
        """Title colour word changed; attributes stay correct (title-attr contradiction)."""
        attrs = text_views.parse_attrs(row["attributes"])
        brands = {str(attrs.get("brand", ""))}
        title = str(row["title"])
        present = _title_color_word(title, self.colors, brands)
        true_color = str(attrs.get("color", "")).lower()
        wrong = self.rng.choice([c for c in self.colors if c not in (present, true_color)])
        if present:
            new_title = _replace_color_word(title, present, wrong, brands)
        else:
            new_title = f"{title} in {wrong.capitalize()}"
        row["title"] = new_title
        row["canonical_text"] = text_views.canonical_text(new_title, str(row["category"]), attrs)
        return InjectionRecord(str(row["row_id"]), str(row["product_id"]), "title_contradiction",
                               "mutate_text", field="title", old_value=present or "", new_value=wrong)

    def swap_image(self, df: pd.DataFrame, idx: int, same_category: bool, subtype: str) -> InjectionRecord | None:
        row = df.loc[idx]
        own_product = str(row["product_id"])
        own_image = str(row["image_path"])

        mask = (df["product_id"].astype(str) != own_product) & (df["image_path"].astype(str) != own_image) \
               & (~df["is_image_missing"].astype(bool))
        if same_category:
            mask &= df["category"].astype(str) == str(row["category"])
        candidates = df.index[mask].tolist()
        if not candidates:
            return None
        donor_idx = self.rng.choice(candidates)
        donor = df.loc[donor_idx]

        df.at[idx, "image_path"] = str(donor["image_path"])
        return InjectionRecord(str(row["row_id"]), own_product, subtype, "swap_image",
                               field="image_path", old_value=own_image,
                               new_value=str(donor["image_path"]),
                               donor_row_id=str(donor["row_id"]),
                               donor_product_id=str(donor["product_id"]))

    def missing_image(self, df: pd.DataFrame, idx: int) -> InjectionRecord | None:
        row = df.loc[idx]
        old = str(row["image_path"])
        if not old:
            return None
        df.at[idx, "image_path"] = ""
        df.at[idx, "is_image_missing"] = True
        return InjectionRecord(str(row["row_id"]), str(row["product_id"]), "missing_image",
                               "missing_image", field="image_path", old_value=old, new_value="")


def inject(df_clean: pd.DataFrame, schema: Schema, noise_cfg: dict) -> NoiseResult:
    """Apply configured error generators; return noisy df + self-verifying audit."""
    rng = random.Random(int(noise_cfg.get("seed", 7)))
    copies = int(noise_cfg.get("copies_per_row", 1))
    rates: dict[str, float] = dict(noise_cfg.get("rates", {}))
    max_redraws = int(noise_cfg.get("max_redraws", 20))

    rows = []
    for _, r in df_clean.iterrows():
        for k in range(copies):
            rr = r.copy()
            rr["row_id"] = f"{r['row_id']}_{k}" if copies > 1 else str(r["row_id"])
            rows.append(rr)
    df = pd.DataFrame(rows).reset_index(drop=True)
    originals = df.copy(deep=True)

    for col, default in [("noise_label", "clean"), ("noise_subtype", "clean"),
                         ("noise_field", ""), ("noise_old_value", ""), ("noise_new_value", ""),
                         ("noise_donor_row_id", ""), ("noise_donor_product_id", "")]:
        df[col] = default

    n = len(df)
    order = list(df.index)
    rng.shuffle(order)

    # deterministic assignment: consecutive slices of the shuffled index per type
    injector = NoiseInjector(schema, noise_cfg, rng)
    audit_records: list[InjectionRecord] = []
    cursor = 0
    for subtype, rate in rates.items():
        if subtype not in ERROR_TYPES:
            raise ValueError(f"unknown noise type in config: {subtype}")
        count = int(round(n * float(rate)))
        targets = order[cursor: cursor + count]
        cursor += count

        for idx in targets:
            rec = None
            for _ in range(max_redraws):
                trial = df.loc[idx].copy()
                if subtype in ("swap_image", "swap_image_same_category"):
                    rec = injector.swap_image(df, idx, subtype.endswith("same_category"), subtype)
                elif subtype == "missing_image":
                    rec = injector.missing_image(df, idx)
                elif subtype == "mixed_swap_color":
                    rec_s = injector.swap_image(df, idx, False, "mixed_swap_color")
                    row = df.loc[idx].copy()
                    rec_c = injector.color_flip(row)
                    if rec_s and rec_c:
                        df.loc[idx, ["attributes", "title", "canonical_text"]] = \
                            row[["attributes", "title", "canonical_text"]]
                        rec = InjectionRecord(rec_s.row_id, rec_s.product_id, "mixed_swap_color",
                                              "mixed", field="image_path+color",
                                              old_value=f"{rec_s.old_value}|{rec_c.old_value}",
                                              new_value=f"{rec_s.new_value}|{rec_c.new_value}",
                                              donor_row_id=rec_s.donor_row_id,
                                              donor_product_id=rec_s.donor_product_id)
                    else:
                        rec = None
                else:
                    row = df.loc[idx].copy()
                    rec = getattr(injector, subtype)(row)
                    if rec is not None:
                        df.loc[idx, ["attributes", "title", "canonical_text"]] = \
                            row[["attributes", "title", "canonical_text"]]
                        if "is_text_missing" in row.index:
                            df.at[idx, "is_text_missing"] = row["is_text_missing"]
                if rec is not None:
                    break
                df.loc[idx] = trial  # restore before re-draw
            if rec is None:
                continue  # could not inject after redraws; row stays clean

            df.at[idx, "noise_label"] = rec.label
            df.at[idx, "noise_subtype"] = rec.subtype
            df.at[idx, "noise_field"] = rec.field
            df.at[idx, "noise_old_value"] = rec.old_value
            df.at[idx, "noise_new_value"] = rec.new_value
            df.at[idx, "noise_donor_row_id"] = rec.donor_row_id
            df.at[idx, "noise_donor_product_id"] = rec.donor_product_id
            audit_records.append(rec)

    # ---------- self-verification pass: no no-op noise, no corrupted cleans ----------
    check_cols = ["title", "attributes", "canonical_text", "image_path", "is_image_missing"]
    problems = []
    for idx in df.index:
        changed = any(str(df.at[idx, c]) != str(originals.at[idx, c]) for c in check_cols)
        label = df.at[idx, "noise_label"]
        if label == "clean" and changed:
            problems.append((str(df.at[idx, "row_id"]), "clean_row_was_modified"))
        if label != "clean" and not changed:
            problems.append((str(df.at[idx, "row_id"]), f"no_op_noise:{df.at[idx, 'noise_subtype']}"))
    if problems:
        raise AssertionError(f"noise self-verification failed for {len(problems)} rows: {problems[:10]}")

    for rec in audit_records:
        rec.verified_changed = True

    audit = pd.DataFrame([r.to_dict() for r in audit_records])
    meta = {
        "seed": int(noise_cfg.get("seed", 7)),
        "copies_per_row": copies,
        "rates": rates,
        "rows_total": int(len(df)),
        "rows_noisy": int((df["noise_label"] != "clean").sum()),
        "by_label": df["noise_label"].value_counts().to_dict(),
        "by_subtype": df.loc[df["noise_label"] != "clean", "noise_subtype"].value_counts().to_dict(),
        "self_verified": True,
    }
    return NoiseResult(df=df, audit=audit, meta=meta)
