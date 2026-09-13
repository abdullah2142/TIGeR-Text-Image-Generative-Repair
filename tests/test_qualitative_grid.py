"""D10: the qualitative grid is built from a run's artifacts, not by hand.

Before this, no code in the repository produced `qualitative_grid_final.png`.
The committed figure was carried along in `paper_figures/` and round-tripped
through the Kaggle output zip unchanged, so it was evidence about a pipeline
that no longer existed.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from tiger import viz


def _img(path: Path, rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), rgb).save(path)


@pytest.fixture
def run(tmp_path):
    """A minimal completed run: two repaired rows, one of them T2V-generated."""
    root = tmp_path
    rows = []
    for i, (rid, colour) in enumerate([("r1", (200, 35, 35)), ("r2", (45, 75, 200))]):
        _img(root / f"data/sample/images/{rid}.jpg", colour)
        rows.append({"row_id": rid, "product_id": rid, "category": "shirts",
                     "title": f"{rid} shirt", "attributes": json.dumps({"color": "red"}),
                     "image_path": f"data/sample/images/{rid}.jpg", "split": "report"})
    clean = pd.DataFrame(rows)
    (root / "data/sample").mkdir(parents=True, exist_ok=True)
    clean.to_parquet(root / "data/sample/products.parquet", index=False)

    noisy = clean.copy()
    noisy["noise_subtype"] = ["color_flip", "swap_image"]
    noisy.loc[0, "attributes"] = json.dumps({"color": "blue"})
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    noisy.to_parquet(root / "data/processed/noisy_report_seed7.parquet", index=False)

    repaired = clean.copy()
    _img(root / "data/sample/images/generated/r2.jpg", (20, 160, 60))
    repaired.loc[1, "image_path"] = "data/sample/images/generated/r2.jpg"
    repaired.to_parquet(root / "data/processed/repaired_report_seed7.parquet", index=False)

    report = {"outcomes": {
        "r1": {"final_status": "repaired",
               "log": [{"direction": "V2T", "patch": {"color": "red"}}]},
        "r2": {"final_status": "repaired",
               "log": [{"direction": "T2V", "candidate_product": "GENERATED"}]},
    }}
    (root / "data/outputs").mkdir(parents=True, exist_ok=True)
    (root / "data/outputs/repair_report_seed7.json").write_text(json.dumps(report))
    return root


def test_grid_is_written_from_run_artifacts(run):
    out = viz.build_qualitative_grid(run, seed=7)
    assert out is not None and out.exists()
    assert out.name == "qualitative_grid_final.png"
    with Image.open(out) as im:
        assert im.width > 3 * 224          # three stages side by side
        assert im.height > 2 * 224         # two rows


def test_generated_row_is_shown_first(run):
    report = json.loads((run / "data/outputs/repair_report_seed7.json").read_text())
    noisy = pd.read_parquet(run / "data/processed/noisy_report_seed7.parquet")
    assert viz.select_rows(report, noisy, max_rows=6)[0] == "r2"


def test_selection_is_deterministic(run):
    report = json.loads((run / "data/outputs/repair_report_seed7.json").read_text())
    noisy = pd.read_parquet(run / "data/processed/noisy_report_seed7.parquet")
    assert viz.select_rows(report, noisy) == viz.select_rows(report, noisy)


def test_max_rows_is_respected(run):
    report = json.loads((run / "data/outputs/repair_report_seed7.json").read_text())
    noisy = pd.read_parquet(run / "data/processed/noisy_report_seed7.parquet")
    assert len(viz.select_rows(report, noisy, max_rows=1)) == 1


def test_run_with_nothing_repaired_returns_none(run):
    report = {"outcomes": {"r1": {"final_status": "escalated", "log": []}}}
    (run / "data/outputs/repair_report_seed7.json").write_text(json.dumps(report))
    assert viz.build_qualitative_grid(run, seed=7) is None


def test_missing_image_is_drawn_not_raised(run):
    """An E4 row has no image and a T2V replacement may not have been written;
    neither is a reason for the figure to fail."""
    rep = pd.read_parquet(run / "data/processed/repaired_report_seed7.parquet")
    rep.loc[1, "image_path"] = "data/sample/images/generated/does_not_exist.jpg"
    rep.to_parquet(run / "data/processed/repaired_report_seed7.parquet", index=False)
    assert viz.build_qualitative_grid(run, seed=7) is not None
