"""Old vs new dominant-colour estimator (B2/B3/B5/B8). Not a pytest module.

Run:  python tests/bench_color_estimator.py

The behaviour tests in ``test_colors.py`` pin the mechanisms; this reports the
magnitude. It needs a corpus, and `data/raw/abo/` is not on this machine any
more (the Kaggle notebook deletes the raw release after each run), so the
corpus is the repo's own synthetic renderer plus one describable perturbation.
Neither number estimates accuracy on real ABO photography -- for that the
estimator has to be re-run on Kaggle. What they establish is direction and
mechanism, with the pre-fix estimator reproduced below so the comparison stays
runnable after the fix has landed.
"""
from __future__ import annotations

import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from tiger.colors import (ACHROMATIC_SAT_MAX, BLACK_V_MAX, MULTI_DOMINANCE_MIN,
                          WHITE_V_MIN, _hue_to_name, estimate_dominant_color)
from tiger.data.synthgen import COLOR_RGB, render_product_image

CATS = ["shirts", "shoes", "bags", "hats"]
COLORS = [c for c in COLOR_RGB if c != "magenta"]
N = 480
SEED = 20260913


def legacy_estimate(image_path: Path, size: int = 96) -> str:
    """The estimator as it stood before B2/B3/B5/B8: square resize, fixed
    central 70% box, saturation-first split, global 85% white discount."""
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    lo, hi = int(size * 0.15), int(size * 0.85)
    arr = arr[lo:hi, lo:hi, :].reshape(-1, 3)

    mx, mn = arr.max(axis=1), arr.min(axis=1)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    delta = np.maximum(mx - mn, 1e-9)
    h = np.zeros_like(v)
    m_r = (mx == r) & (delta > 1e-6)
    m_g = (mx == g) & ~m_r & (delta > 1e-6)
    m_b = (mx == b) & ~m_r & ~m_g & (delta > 1e-6)
    h[m_r] = (60.0 * ((g[m_r] - b[m_r]) / delta[m_r])) % 360.0
    h[m_g] = 60.0 * ((b[m_g] - r[m_g]) / delta[m_g]) + 120.0
    h[m_b] = 60.0 * ((r[m_b] - g[m_b]) / delta[m_b]) + 240.0

    achro = s < ACHROMATIC_SAT_MAX
    names = np.empty(v.shape, dtype=object)
    names[achro & (v <= BLACK_V_MAX)] = "black"
    names[achro & (v >= WHITE_V_MIN)] = "white"
    names[achro & (v > BLACK_V_MAX) & (v < WHITE_V_MIN)] = "gray"
    names[~achro] = _hue_to_name(h[~achro], s[~achro], v[~achro])

    counted = names[names != ""]
    if counted.size == 0:
        return "unknown"
    vals, counts = np.unique(counted, return_counts=True)
    ranked = [(str(vals[i]), float(counts[i]) / counted.size) for i in np.argsort(-counts)]
    if ranked[0][0] == "white" and len(ranked) > 1 and ranked[0][1] < 0.85:
        rest = [(n, p) for n, p in ranked if n != "white"]
        tot = sum(p for _, p in rest)
        if tot > 0.05:
            ranked = [(n, p / tot) for n, p in rest]
    top, top_p = ranked[0]
    if top_p < MULTI_DOMINANCE_MIN and len(ranked) > 1 and ranked[1][1] > 0.25:
        return "multicolour"
    return top


def perturb(src: Path, dst: Path, rng: random.Random) -> None:
    """Crop the product out of a synthgen render and paste it somewhere else.

    One transform, applied to the same renders: the product keeps its colour and
    loses the guarantee that it is centred, frame-filling and square.
    """
    im = Image.open(src).convert("RGB")
    a = np.asarray(im, dtype=np.int16)
    bg = a[2, 2]
    fg = np.abs(a - bg).sum(axis=2) > 24
    rows, cols = np.flatnonzero(fg.any(axis=1)), np.flatnonzero(fg.any(axis=0))
    prod = im.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))

    W = rng.choice([160, 224, 320, 400])
    H = int(W * rng.uniform(0.5, 2.0))
    scale = rng.uniform(0.20, 0.55) * min(W, H) / max(prod.size)
    prod = prod.resize((max(4, int(prod.width * scale)), max(4, int(prod.height * scale))))
    canvas = Image.new("RGB", (W, H), tuple(int(c) for c in bg))
    canvas.paste(prod, (rng.randint(0, W - prod.width), rng.randint(0, H - prod.height)))
    canvas.save(dst, format="JPEG", quality=92)


def main() -> None:
    rng = random.Random(SEED)
    td = Path(tempfile.mkdtemp())
    rows = []
    for i in range(N):
        cat, col = CATS[i % len(CATS)], COLORS[i % len(COLORS)]
        plain, pert = td / f"p{i}.jpg", td / f"q{i}.jpg"
        render_product_image(plain, cat, col, "solid", rng, size=224)
        perturb(plain, pert, rng)
        new_plain, new_pert = estimate_dominant_color(plain), estimate_dominant_color(pert)
        rows.append((col, legacy_estimate(plain), new_plain.top,
                     legacy_estimate(pert), new_pert.top, new_pert.region))

    print(f"n = {N} solid-colour renders, {len(set(COLORS))} colours, "
          f"{len(CATS)} categories, seed {SEED}\n")
    for label, io, inew in (("synthgen as rendered ", 1, 2),
                            ("geometry-perturbed   ", 3, 4)):
        o = sum(r[io] == r[0] for r in rows) / len(rows)
        n = sum(r[inew] == r[0] for r in rows) / len(rows)
        print(f"{label}  old {o:6.1%}   new {n:6.1%}   delta {(n - o) * 100:+.1f} pts")

    print(f"\nregion chosen (perturbed): {dict(Counter(r[5] for r in rows))}")
    print(f"\n{'colour':10s} {'n':>4s} {'old':>6s} {'new':>6s}   (perturbed)")
    per = defaultdict(list)
    for r in rows:
        per[r[0]].append(r)
    for col, rs in sorted(per.items()):
        o = sum(r[3] == col for r in rs) / len(rs)
        n = sum(r[4] == col for r in rs) / len(rs)
        print(f"{col:10s} {len(rs):4d} {o:6.0%} {n:6.0%}")


if __name__ == "__main__":
    main()
