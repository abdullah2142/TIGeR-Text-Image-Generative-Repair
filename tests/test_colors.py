from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tiger.colors import estimate_dominant_color


def draw(tmp_path: Path, name: str, paint) -> Path:
    im = Image.new("RGB", (128, 128), (245, 245, 245))
    d = ImageDraw.Draw(im)
    paint(d)
    p = tmp_path / f"{name}.png"
    im.save(p)
    return p


def test_solid_red(tmp_path):
    p = draw(tmp_path, "red", lambda d: d.rectangle([20, 20, 108, 108], fill=(200, 35, 35)))
    assert estimate_dominant_color(p).top == "red"


def test_striped_red_white_not_pink(tmp_path):
    """F6 regression: mean-RGB called red/white stripes pink; mode must not."""
    def paint(d):
        for x in range(20, 108, 16):
            d.rectangle([x, 20, x + 8, 108], fill=(200, 35, 35))
            d.rectangle([x + 8, 20, x + 16, 108], fill=(250, 250, 250))
    p = draw(tmp_path, "stripes", paint)
    est = estimate_dominant_color(p)
    assert est.top in ("red", "multicolour")
    assert est.top != "pink"


def test_white_background_does_not_dominate(tmp_path):
    # small blue object on a big white studio background
    p = draw(tmp_path, "smallblue", lambda d: d.rectangle([44, 44, 84, 84], fill=(45, 75, 200)))
    assert estimate_dominant_color(p).top == "blue"


def test_black_product(tmp_path):
    p = draw(tmp_path, "black", lambda d: d.rectangle([20, 20, 108, 108], fill=(20, 20, 22)))
    assert estimate_dominant_color(p).top == "black"


def test_white_product(tmp_path):
    p = draw(tmp_path, "white", lambda d: d.rectangle([10, 10, 118, 118], fill=(250, 250, 250)))
    assert estimate_dominant_color(p).top == "white"


def test_multicolour(tmp_path):
    def paint(d):
        d.rectangle([20, 20, 64, 108], fill=(200, 35, 35))
        d.rectangle([64, 20, 108, 108], fill=(45, 75, 200))
    p = draw(tmp_path, "redblue", paint)
    est = estimate_dominant_color(p)
    assert est.top == "multicolour"
    names = {n for n, _ in est.top2}
    assert names == {"red", "blue"}


# ---------------------------------------------------------------------------
# B2 · product localisation (the fixed central 70% box measured the wrong object)
# ---------------------------------------------------------------------------

def test_offcentre_product_is_localised(tmp_path):
    """B2: a product in the corner is invisible to the central box."""
    p = draw(tmp_path, "corner", lambda d: d.rectangle([6, 6, 46, 46], fill=(200, 35, 35)))
    est = estimate_dominant_color(p)
    assert est.top == "red"
    assert est.region == "foreground"


def test_bottom_band_product_is_localised(tmp_path):
    """B2: shoes/rugs sit at the bottom of the frame, not in the middle."""
    p = draw(tmp_path, "band", lambda d: d.rectangle([10, 96, 118, 122], fill=(45, 75, 200)))
    est = estimate_dominant_color(p)
    assert est.top == "blue"


def test_lifestyle_shot_falls_back_to_centre_box(tmp_path):
    """B2 guard: no uniform ground means no trustworthy flood -- do not run one."""
    def paint(d):
        for i, fill in enumerate([(200, 35, 35), (45, 75, 200), (35, 160, 60), (230, 200, 40)]):
            d.rectangle([0, i * 32, 128, i * 32 + 32], fill=fill)
    p = draw(tmp_path, "lifestyle", paint)
    assert estimate_dominant_color(p).region == "center_box"


def test_localisation_ignores_background_gradient(tmp_path):
    """A graduated studio ground is still one ground, and must not become the answer."""
    im = Image.new("RGB", (128, 128))
    d = ImageDraw.Draw(im)
    for y in range(128):
        g = 210 + int(35 * y / 127)
        d.line([(0, y), (128, y)], fill=(g, g, g))
    d.rectangle([2, 2, 44, 16], fill=(35, 160, 60))   # corner: outside the old centre box
    p = tmp_path / "gradient.png"
    im.save(p)
    assert estimate_dominant_color(p).top == "green"


# ---------------------------------------------------------------------------
# B3 · white products vs white grounds, decided spatially rather than by share
# ---------------------------------------------------------------------------

def test_white_product_survives_the_shadow_that_used_to_outvote_it(tmp_path):
    """B3: a genuinely white product under the 85% share rule returned its shadow."""
    im = Image.new("RGB", (128, 128), (200, 200, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([14, 14, 114, 114], fill=(160, 160, 160))   # shadow skirt
    d.rectangle([14, 14, 114, 89], fill=(252, 252, 252))    # the product itself
    p = tmp_path / "whiteish.png"
    im.save(p)
    est = estimate_dominant_color(p)
    assert est.top == "white"
    assert est.region == "foreground"


def test_white_product_on_white_ground_is_still_white(tmp_path):
    """When the flood cannot separate product from ground they are the same colour."""
    p = draw(tmp_path, "whiteonwhite", lambda d: d.rectangle([10, 10, 118, 118], fill=(250, 250, 250)))
    est = estimate_dominant_color(p)
    assert est.top == "white"
    assert est.region == "flooded"


# ---------------------------------------------------------------------------
# B5 · aspect ratio destroyed before cropping
# ---------------------------------------------------------------------------

def test_resize_preserves_aspect_ratio():
    from tiger.colors import _load_rgb
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tall.png"
        Image.new("RGB", (160, 480), (245, 245, 245)).save(p)
        arr = _load_rgb(p, 96)
    assert arr.shape[:2] == (96, 32)


def test_tall_product_keeps_its_shape_through_the_estimator(tmp_path):
    """B5: the square resize made a 1:3 product square before it was localised."""
    from tiger.colors import _load_rgb, select_region
    im = Image.new("RGB", (160, 480), (245, 245, 245))
    ImageDraw.Draw(im).rectangle([60, 120, 99, 239], fill=(45, 75, 200))
    p = tmp_path / "tallbar.png"
    im.save(p)
    mask, region, _ = select_region(_load_rgb(p, 96))
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    aspect = (rows[-1] - rows[0] + 1) / (cols[-1] - cols[0] + 1)
    assert region == "foreground"
    assert 2.5 < aspect < 3.5      # the drawn bar is 40 x 120 -> 1:3


# ---------------------------------------------------------------------------
# B8 · saturation is unreliable at the dark end
# ---------------------------------------------------------------------------

def test_noisy_black_product_is_black_not_blue(tmp_path):
    """B8: s = (max-min)/max explodes on near-black pixels, so a few levels of
    sensor/JPEG noise on a black product used to be hue-binned as blue."""
    rng = np.random.default_rng(7)
    im = Image.new("RGB", (128, 128), (245, 245, 245))
    px = im.load()
    for y in range(20, 108):
        for x in range(20, 108):
            px[x, y] = tuple(int(np.clip(c + rng.integers(-18, 19), 0, 255)) for c in (28, 28, 30))
    p = tmp_path / "noisyblack.png"
    im.save(p)
    assert estimate_dominant_color(p).top == "black"


def test_dark_pixels_do_not_become_hues(tmp_path):
    """The same rule stated directly: value decides below BLACK_V_MAX."""
    from tiger.colors import BLACK_V_MAX
    im = Image.new("RGB", (128, 128), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 108, 108], fill=(8, 12, 52))   # v = 0.204, s = 0.85
    p = tmp_path / "darknavy.png"
    im.save(p)
    assert 52 / 255 <= BLACK_V_MAX
    assert estimate_dominant_color(p).top == "black"
