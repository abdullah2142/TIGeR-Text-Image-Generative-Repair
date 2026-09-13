"""HSV dominant-colour estimation (roadmap 1.7, finding F6).

Replaces the mean-RGB / nearest-prototype estimator, which was systematically
wrong (mean of red+white stripes is pink; studio-white backgrounds dominate the
mean; RGB distance is not perceptual).

Method:
  1. downscale *preserving aspect ratio* (B5: a square resize squashes the 63.8%
     of ABO images that are not square, moving the product out of the sampled
     region before it is ever measured)
  2. localise the product (B2): flood the studio ground inward from the border
     and keep what the flood cannot reach. The previous fixed central 70% box
     assumes the product is centred and frame-filling, which is true of the
     synthetic catalogue and false of real product photography -- a rug, a lamp
     and a wall hanging occupy very different parts of the frame. Falls back to
     the central box when the border is not a uniform ground (lifestyle shots).
  3. split achromatic pixels (low saturation / extreme value) from chromatic ones
  4. histogram mode over hue bins for chromatic pixels; white/gray/black decided
     by value for achromatic ones
  5. white counts as background only where the flood actually reached it (B3),
     not because it fell under a global 85% proportion threshold; the proportion
     rule survives only as the fallback for images with no usable mask
  6. return top-2 colour names with pixel proportions and a confidence;
     "multicolour" is a legal output when no single colour dominates

Known limitation of step 2: a white region of the product that touches a white
ground is absorbed by the flood along with the ground, because nothing separates
them. This is inherent to background-connectivity localisation, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Hue ranges on the 0-360 wheel per colour name. Red wraps around 0.
HUE_RANGES = {
    "red": [(0.0, 14.0), (346.0, 360.0)],
    "orange": [(14.0, 40.0)],
    "yellow": [(40.0, 70.0)],
    "green": [(70.0, 165.0)],
    "blue": [(165.0, 255.0)],
    "purple": [(255.0, 290.0)],
    "pink": [(290.0, 346.0)],
}

# Brown is dark orange/red: handled after hue binning via value check.
ACHROMATIC_SAT_MAX = 0.18
BLACK_V_MAX = 0.22
WHITE_V_MIN = 0.82
MULTI_DOMINANCE_MIN = 0.55  # top colour must own >= this share of counted pixels

# --- background localisation (B2/B3) --------------------------------------
BORDER_FRAC = 0.04        # frame width sampled to characterise the ground
BG_TOL_MIN = 0.04         # floor on the growth tolerance (flat studio ground)
BG_TOL_MAX = 0.18         # ceiling (strongly graduated ground)
BG_TOL_MAD_K = 3.0        # growth tolerance = k x the border's own spread
BORDER_UNIFORM_MIN = 0.60  # share of border pixels the ground must explain
MIN_FOREGROUND = 0.02     # below this the flood ate the product: use everything

# The central box the estimator falls back to when no ground can be identified.
CENTER_LO, CENTER_HI = 0.15, 0.85


@dataclass
class ColorEstimate:
    top: str                      # best single answer, may be "multicolour"
    top2: list[tuple[str, float]]  # [(name, proportion)] for the two largest masses
    confidence: float             # proportion of the winning colour
    n_pixels: int
    # B2 attribution: which region the histogram was actually taken over, and
    # how much of the frame survived the background flood.
    region: str = "center_box"
    foreground_share: float = 1.0

    def to_dict(self) -> dict:
        return {
            "top": self.top,
            "top2": [[n, round(p, 4)] for n, p in self.top2],
            "confidence": round(self.confidence, 4),
            "n_pixels": self.n_pixels,
            "region": self.region,
            "foreground_share": round(self.foreground_share, 4),
        }


def _hue_to_name(h_deg: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised hue -> colour-name index mapping. Returns array of names."""
    names = np.empty(h_deg.shape, dtype=object)
    names[:] = ""
    for name, ranges in HUE_RANGES.items():
        m = np.zeros(h_deg.shape, dtype=bool)
        for lo, hi in ranges:
            m |= (h_deg >= lo) & (h_deg < hi)
        names[m] = name
    # brown: dark, moderately saturated orange/red hues
    brown = ((h_deg >= 0) & (h_deg < 50) | (h_deg >= 346)) & (v < 0.6) & (s > 0.2)
    names[brown] = "brown"
    return names


def _load_rgb(image_path: str | Path, size: int) -> np.ndarray:
    """Decode to RGB and scale the long edge to ``size``, keeping aspect (B5)."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = size / float(max(w, h))
        target = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        im = im.resize(target)
        return np.asarray(im, dtype=np.float32) / 255.0


def _grow(seed: np.ndarray, allowed: np.ndarray, max_steps: int) -> np.ndarray:
    """4-connected morphological reconstruction of ``seed`` inside ``allowed``."""
    cur = seed & allowed
    for _ in range(max_steps):
        grown = cur.copy()
        grown[1:, :] |= cur[:-1, :]
        grown[:-1, :] |= cur[1:, :]
        grown[:, 1:] |= cur[:, :-1]
        grown[:, :-1] |= cur[:, 1:]
        grown &= allowed
        if grown.sum() == cur.sum():
            break
        cur = grown
    return cur


def background_mask(rgb: np.ndarray) -> np.ndarray | None:
    """Pixels belonging to the studio ground, or ``None`` if there is no ground.

    The ground is defined by connectivity, not by colour alone: a pixel is
    background only if a path of similarly-coloured pixels links it to the
    image border. The growth tolerance is derived from the border's own spread
    so that a flat ground stays tight (and will not cross a low-contrast
    product edge) while a graduated ground is still absorbed whole.
    """
    h, w, _ = rgb.shape
    if h < 4 or w < 4:
        return None
    b = max(1, int(round(BORDER_FRAC * min(h, w))))
    frame = np.zeros((h, w), dtype=bool)
    frame[:b, :] = True
    frame[-b:, :] = True
    frame[:, :b] = True
    frame[:, -b:] = True

    ref = np.median(rgb[frame], axis=0)
    dist = np.sqrt(((rgb - ref) ** 2).sum(axis=2))
    spread = float(np.median(dist[frame]))
    tol = float(np.clip(BG_TOL_MAD_K * spread, BG_TOL_MIN, BG_TOL_MAX))

    # A border the reference colour cannot explain is a lifestyle/in-context
    # shot, not a studio ground: refuse rather than flood into the scene.
    if float((dist[frame] <= tol).mean()) < BORDER_UNIFORM_MIN:
        return None

    seed = frame & (dist <= tol)
    if not seed.any():
        return None
    return _grow(seed, dist <= tol, max_steps=2 * (h + w))


def _center_box(h: int, w: int) -> np.ndarray:
    sel = np.zeros((h, w), dtype=bool)
    sel[int(h * CENTER_LO):int(h * CENTER_HI), int(w * CENTER_LO):int(w * CENTER_HI)] = True
    return sel


def select_region(rgb: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Choose the pixels to histogram. Returns ``(mask, region_name, fg_share)``."""
    h, w, _ = rgb.shape
    bg = background_mask(rgb)
    if bg is None:
        return _center_box(h, w), "center_box", 1.0
    fg = ~bg
    share = float(fg.mean())
    if share < MIN_FOREGROUND:
        # The flood reached everything: the product is the same colour as the
        # ground (a white product on white). Measuring the whole frame returns
        # that colour, which is the right answer; the central box would return
        # whatever shadow noise it happened to contain.
        return np.ones((h, w), dtype=bool), "flooded", share
    return fg, "foreground", share


def estimate_dominant_color(image_path: str | Path, size: int = 96) -> ColorEstimate:
    rgb = _load_rgb(image_path, size)
    sel, region, fg_share = select_region(rgb)
    arr = rgb[sel]

    mx = arr.max(axis=1)
    mn = arr.min(axis=1)
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

    # B8: saturation is (max-min)/max, so it explodes at the dark end -- an
    # RGB spread of a few levels on a near-black pixel reads as s > 0.4 and the
    # pixel gets hue-binned as blue/green/pink. Value decides first below
    # BLACK_V_MAX; only above it is saturation trustworthy enough to split
    # achromatic from chromatic.
    dark = v <= BLACK_V_MAX
    achromatic = (s < ACHROMATIC_SAT_MAX) & ~dark
    names = np.empty(v.shape, dtype=object)
    names[dark] = "black"
    names[achromatic & (v >= WHITE_V_MIN)] = "white"
    names[achromatic & (v < WHITE_V_MIN)] = "gray"

    chrom = ~achromatic & ~dark
    names[chrom] = _hue_to_name(h[chrom], s[chrom], v[chrom])

    counted = names[names != ""]
    n = counted.size
    if n == 0:
        return ColorEstimate("unknown", [], 0.0, 0, region, fg_share)

    vals, counts = np.unique(counted, return_counts=True)
    order = np.argsort(-counts)
    ranked = [(str(vals[i]), float(counts[i]) / n) for i in order]

    # B3: when the flood identified the ground, background white is already
    # gone and any white left is on the product -- discounting it again would
    # throw away the answer for genuinely white products. The global 85%
    # proportion rule applies only to the unlocalised fallback, where white
    # background does still survive into the sample.
    if region == "center_box" and ranked[0][0] == "white" and len(ranked) > 1 and ranked[0][1] < 0.85:
        rest = [(nme, p) for nme, p in ranked if nme != "white"]
        tot = sum(p for _, p in rest)
        if tot > 0.05:
            ranked = [(nme, p / tot) for nme, p in rest]

    top2 = ranked[:2]
    top_name, top_p = top2[0]
    if top_p < MULTI_DOMINANCE_MIN and len(top2) > 1 and top2[1][1] > 0.25:
        return ColorEstimate("multicolour", top2, float(top_p), n, region, fg_share)
    return ColorEstimate(top_name, top2, float(top_p), n, region, fg_share)
