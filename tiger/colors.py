"""HSV dominant-colour estimation (roadmap 1.7, finding F6).

Replaces the mean-RGB / nearest-prototype estimator, which was systematically
wrong (mean of red+white stripes is pink; studio-white backgrounds dominate the
mean; RGB distance is not perceptual).

Method:
  1. downscale, convert to HSV
  2. centre-weighted foreground mask (drops the studio-background border)
  3. split achromatic pixels (low saturation / extreme value) from chromatic ones
  4. histogram mode over hue bins for chromatic pixels; white/gray/black decided
     by value for achromatic ones
  5. return top-2 colour names with pixel proportions and a confidence;
     "multicolour" is a legal output when no single colour dominates
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


@dataclass
class ColorEstimate:
    top: str                      # best single answer, may be "multicolour"
    top2: list[tuple[str, float]]  # [(name, proportion)] for the two largest masses
    confidence: float             # proportion of the winning colour
    n_pixels: int

    def to_dict(self) -> dict:
        return {
            "top": self.top,
            "top2": [[n, round(p, 4)] for n, p in self.top2],
            "confidence": round(self.confidence, 4),
            "n_pixels": self.n_pixels,
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


def estimate_dominant_color(image_path: str | Path, size: int = 96) -> ColorEstimate:
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize((size, size))
        arr = np.asarray(im, dtype=np.float32) / 255.0

    # centre-weighted foreground mask: keep the central 70% box
    lo, hi = int(size * 0.15), int(size * 0.85)
    arr = arr[lo:hi, lo:hi, :].reshape(-1, 3)

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

    achromatic = s < ACHROMATIC_SAT_MAX
    names = np.empty(v.shape, dtype=object)
    names[achromatic & (v <= BLACK_V_MAX)] = "black"
    names[achromatic & (v >= WHITE_V_MIN)] = "white"
    names[achromatic & (v > BLACK_V_MAX) & (v < WHITE_V_MIN)] = "gray"

    chrom = ~achromatic
    names[chrom] = _hue_to_name(h[chrom], s[chrom], v[chrom])

    # Studio shots: white background survives the centre crop; discount white
    # unless it is overwhelmingly dominant (a genuinely white product).
    counted = names[names != ""]
    n = counted.size
    if n == 0:
        return ColorEstimate("unknown", [], 0.0, 0)

    vals, counts = np.unique(counted, return_counts=True)
    order = np.argsort(-counts)
    ranked = [(str(vals[i]), float(counts[i]) / n) for i in order]

    if ranked[0][0] == "white" and len(ranked) > 1 and ranked[0][1] < 0.85:
        # treat white as background; renormalise over the rest
        rest = [(nme, p) for nme, p in ranked if nme != "white"]
        tot = sum(p for _, p in rest)
        if tot > 0.05:
            ranked = [(nme, p / tot) for nme, p in rest]

    top2 = ranked[:2]
    top_name, top_p = top2[0]
    if top_p < MULTI_DOMINANCE_MIN and len(top2) > 1 and top2[1][1] > 0.25:
        return ColorEstimate("multicolour", top2, float(top_p), n)
    return ColorEstimate(top_name, top2, float(top_p), n)
