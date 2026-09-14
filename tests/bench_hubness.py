"""Does hubness correction improve T2V retrieval? (D16). Not a pytest module.

Run:  python tests/bench_hubness.py [path/to/run] [--k 10]

A few images sit close to many captions in CLIP space and get retrieved over
and over -- the hubness problem in cross-modal retrieval. On the reported run,
304 image installs drew on 114 distinct donors and one image was installed on
18 rows; repairs using a donor installed 9+ times were correct 25.5% of the
time against ~46% for donors used 3-8 times.

CSLS (Conneau et al., ICLR 2018) is the standard correction: penalise each
candidate by how well it matches the *neighbourhood* of captions, so an image
that matches everything loses its advantage.

    score(t, i) = 2 * cos(t, i) - r(i),   r(i) = mean cos(i, its k nearest captions)

This replays retrieval on the calibration seeds, which ship their embedding
arrays, and scores both rules the way the ablation does: a swap is correct when
the installed image depicts a product matching the row's true category+colour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tiger import text_views

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014]


def hubness(image_emb: np.ndarray, caption_emb: np.ndarray, k: int,
            chunk: int = 512) -> np.ndarray:
    """r(i): mean similarity of each image to its k nearest captions."""
    out = np.zeros(len(image_emb), dtype=np.float32)
    k = max(1, min(k, len(caption_emb)))
    for a in range(0, len(image_emb), chunk):
        sims = image_emb[a:a + chunk] @ caption_emb.T
        part = np.partition(sims, -k, axis=1)[:, -k:]
        out[a:a + chunk] = part.mean(axis=1)
    return out


def run_seed(run: Path, seed: int, k: int) -> pd.DataFrame:
    df = pd.read_parquet(run / "outputs" / f"sieve_cal_seed{seed}.parquet")
    z = np.load(run / "outputs" / f"sieve_cal_seed{seed}_arrays.npz")
    img, cap, ok = z["image_emb"], z["caption_emb"], z["image_ok"].astype(bool)
    audit = pd.read_csv(run / "processed" / f"noise_audit_cal_seed{seed}.csv")

    # ground truth: what each image actually depicts, and each row's true colour
    true_color, true_image = {}, {}
    for _, r in audit.iterrows():
        rid = str(r["row_id"])
        if r.get("field") == "color" and pd.notna(r.get("old_value")):
            true_color[rid] = str(r["old_value"])
        if str(r.get("subtype", "")).startswith("swap_image") and pd.notna(r.get("old_value")):
            true_image[rid] = str(r["old_value"])

    depicts = {}
    for _, r in df.iterrows():
        rid = str(r["row_id"])
        own = true_image.get(rid) or str(r["image_path"])
        attrs = text_views.parse_attrs(r["attributes"])
        depicts[own] = (str(r["category"]),
                        true_color.get(rid) or str(attrs.get("color", "")))

    # the pool is what the pipeline builds it from: flagged rows with an image
    pool = df["flagged"].astype(bool).to_numpy() & ok
    cats = df["category"].astype(str).to_numpy()
    pids = df["product_id"].astype(str).to_numpy()
    paths = df["image_path"].astype(str).to_numpy()
    r_i = hubness(img, cap, k)

    rows = []
    targets = np.flatnonzero((df["noise_label"].astype(str) == "swap_image").to_numpy() & pool)
    for i in targets:
        mask = pool & (pids != pids[i]) & (cats == cats[i])
        if not mask.any():
            continue
        sims = img @ cap[i]
        want = (cats[i], true_color.get(str(df.at[i, "row_id"]),
                                        str(text_views.parse_attrs(df.at[i, "attributes"]).get("color", ""))))
        picks = {}
        for name, score in (("cosine", sims), ("csls", 2.0 * sims - r_i)):
            j = int(np.argmax(np.where(mask, score, -np.inf)))
            picks[name] = int(depicts.get(paths[j], (None, None)) == want)
            picks[name + "_donor"] = pids[j]
        rows.append(picks)
    return pd.DataFrame(rows)


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else ROOT / "data"
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 10

    frames = [run_seed(run, s, k) for s in SEEDS
              if (run / "outputs" / f"sieve_cal_seed{s}_arrays.npz").exists()]
    if not frames:
        raise SystemExit(f"no calibration arrays under {run}/outputs")
    d = pd.concat(frames, ignore_index=True)

    print(f"{len(d)} swap_image rows replayed across {len(frames)} seeds, CSLS k={k}\n")
    print(f"{'rule':>8s} {'correct donor':>15s} {'distinct donors':>17s} {'max installs':>13s}")
    for name in ("cosine", "csls"):
        donors = d[name + "_donor"].value_counts()
        print(f"{name:>8s} {d[name].mean():>14.1%} {len(donors):>17d} {donors.iloc[0]:>13d}")
    both = d[(d.cosine == 1) | (d.csls == 1)]
    print(f"\nrows where they disagree: {(d.cosine != d.csls).sum()} "
          f"(cosine right {(d.cosine > d.csls).sum()}, csls right {(d.csls > d.cosine).sum()})")


if __name__ == "__main__":
    main()
