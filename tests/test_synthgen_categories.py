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
    """schema.yaml: 'the ABO furnishing and accessory verticals carry no size enum'."""
    schema = load_schema("configs/schema.yaml")
    out_dir = Path(tempfile.mkdtemp())
    df = synthgen.generate(root=Path("."), schema=schema, products_per_category=1,
                           out_dir=str(out_dir))
    import json
    chair = df[df["category"] == "chair"].iloc[0]
    assert json.loads(chair["attributes"])["size"] == ""
