"""Every schema category must be generatable by synthgen, not just the 4 fashion ones.

Phase 1 added 15 ABO categories to configs/schema.yaml, but MATERIALS, SIZES and
TITLE_NOUNS in tiger/data/synthgen.py were never extended to match -- generate()
crashed with KeyError('chair') on the very first non-fashion category, before
writing a single row (found via the Kaggle Phase-2 dry run, 2026-09-11).
"""

import tempfile
from pathlib import Path

from tiger.data import synthgen
from tiger.schema import load_schema


def test_every_schema_category_generates():
    schema = load_schema("configs/schema.yaml")
    out_dir = Path(tempfile.mkdtemp())
    df = synthgen.generate(root=Path("."), schema=schema, products_per_category=1,
                           out_dir=str(out_dir))
    generated = set(df["category"].unique())
    missing = set(schema.categories) - generated
    assert not missing, f"synthgen produced no rows for: {sorted(missing)}"


def test_non_fashion_category_gets_no_size():
    """schema.yaml: 'the ABO furnishing and accessory verticals carry no size enum'.

    This asserted `size == ""` and so pinned the defect it was meant to prevent:
    an empty string is a *present* key whose value is outside the size enum, so
    every such row failed schema validation. The key must be absent.
    """
    schema = load_schema("configs/schema.yaml")
    out_dir = Path(tempfile.mkdtemp())
    df = synthgen.generate(root=Path("."), schema=schema, products_per_category=1,
                           out_dir=str(out_dir))
    import json
    chair = df[df["category"] == "chair"].iloc[0]
    assert "size" not in json.loads(chair["attributes"])


def test_clean_rows_pass_schema_validation():
    """A clean catalogue must not flag itself.

    `size` was written as "" for the 15 categories with no size enum, so schema
    validation failed on every one of those rows and `flag_text_out_of_domain`
    fired: ~79% of a clean catalogue reported as dirty by construction, and the
    single largest false-positive source in the detection table.
    """
    import json

    schema = load_schema("configs/schema.yaml")
    out_dir = Path(tempfile.mkdtemp())
    df = synthgen.generate(root=Path("."), schema=schema, products_per_category=2,
                           out_dir=str(out_dir))

    offenders = []
    for _, r in df.iterrows():
        if str(r["row_id"]).startswith("forced_gen"):
            continue          # deliberately planted, see honest_limitations.md §5
        v = schema.validate_attrs(str(r["category"]), json.loads(r["attributes"]))
        if v:
            offenders.append((r["row_id"], [f"{x.rule}:{x.field}" for x in v]))
    assert not offenders, f"clean rows failing validation: {offenders[:5]}"


def test_sized_categories_still_get_a_valid_size():
    import json

    schema = load_schema("configs/schema.yaml")
    out_dir = Path(tempfile.mkdtemp())
    df = synthgen.generate(root=Path("."), schema=schema, products_per_category=2,
                           out_dir=str(out_dir))
    shirts = df[df["category"] == "shirts"]
    assert len(shirts)
    for _, r in shirts.iterrows():
        assert json.loads(r["attributes"])["size"] in ("XS", "S", "M", "L", "XL")
