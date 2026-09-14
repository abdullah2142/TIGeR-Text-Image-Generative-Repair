"""B7: the per-field probes may run on a different encoder from the similarity
path, so the attribute-binding signal can be upgraded without moving the
reported CLIP baseline (sim_full, swap, LOO)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tiger import sieve
from tiger.schema import Schema

ROOT_DIR = Path(__file__).resolve().parents[1]

SCHEMA = Schema(
    attributes={"color": {"type": "enum", "values": ["red", "blue"]}},
    categories=["shirts"],
    constraints=[],
)
CFG = {"sieve": {"probes": {"fields": ["color"]}}}


class _Encoder:
    """Two-dimensional encoder that votes for one colour word.

    Images embed as [1, 0]; a caption naming `votes_for` embeds as [1, 0] and
    every other caption as [0, 1], so the probe argmax is `votes_for`.
    """

    def __init__(self, votes_for: str):
        self.votes_for = votes_for
        self.images_encoded = 0

    def encode_images(self, paths):
        self.images_encoded += len(paths)
        return np.tile([1.0, 0.0], (len(paths), 1)), np.ones(len(paths), dtype=bool)

    def encode_texts(self, texts):
        hit = [self.votes_for in t.lower() for t in texts]
        return np.array([[1.0, 0.0] if h else [0.0, 1.0] for h in hit], dtype=float)

    def save_cache(self):
        pass


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    """compute_signals asserts the CLIP token budget, which needs transformers.
    These tests are about which encoder answers, not about caption length."""
    from tiger import text_views
    monkeypatch.setattr(text_views, "assert_token_budget", lambda *a, **k: [])


@pytest.fixture
def catalogue(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (16, 16), (200, 35, 35)).save(img)
    df = pd.DataFrame([{
        "row_id": "r1", "product_id": "r1", "category": "shirts",
        "title": "A Shirt", "attributes": json.dumps({"color": "red"}),
        "image_path": "a.png",
    }])
    return df, tmp_path


def test_probes_follow_the_primary_encoder_by_default(catalogue):
    df, root = catalogue
    sig, _ = sieve.compute_signals(df, _Encoder("red"), SCHEMA, CFG, root)
    assert sig.at[0, "probe_color_pred"] == "red"


def test_probe_encoder_overrides_only_the_probes(catalogue):
    """The probe encoder decides `probe_color_pred`; the primary encoder still
    decides sim_full, which is the number the paper reports."""
    df, root = catalogue
    primary, probe = _Encoder("red"), _Encoder("blue")
    sig, arrays = sieve.compute_signals(df, primary, SCHEMA, CFG, root, probe_encoder=probe)

    assert sig.at[0, "probe_color_pred"] == "blue"     # probe encoder's answer
    assert primary.images_encoded == 1                  # both saw the image
    assert probe.images_encoded == 1
    # sim_full comes from the primary encoder's caption embedding: its caption
    # names "red", so it scores 1.0 against the image.
    assert sig.at[0, "sim_full"] == pytest.approx(1.0)


def test_probe_encoder_that_cannot_read_the_image_drops_the_row_from_probes(catalogue):
    df, root = catalogue

    class _Blind(_Encoder):
        def encode_images(self, paths):
            emb, _ = super().encode_images(paths)
            return emb, np.zeros(len(paths), dtype=bool)

    sig, _ = sieve.compute_signals(df, _Encoder("red"), SCHEMA, CFG, root,
                                   probe_encoder=_Blind("blue"))
    assert sig.at[0, "probe_color_pred"] == ""          # no probe verdict
    assert not bool(sig.at[0, "is_image_missing"])      # but the row is not missing


# ---------------------------------------------------------------------------
# resolving the probe encoder from config, in one place
# ---------------------------------------------------------------------------

def test_probe_encoder_is_none_unless_configured(tmp_path):
    from tiger.encoders import probe_encoder_from_cfg
    base = {"data": {"cache_dir": "data/cache_embeddings"},
            "models": {"clip_model_name": "openai/clip-vit-base-patch32"}}
    assert probe_encoder_from_cfg(base, tmp_path) is None
    same = {**base, "models": {**base["models"],
                               "probe_model_name": "openai/clip-vit-base-patch32"}}
    assert probe_encoder_from_cfg(same, tmp_path) is None      # same model: share it
    blank = {**base, "models": {**base["models"], "probe_model_name": "  "}}
    assert probe_encoder_from_cfg(blank, tmp_path) is None


def test_probe_encoder_is_memoised(tmp_path):
    """The sieve, the repair cycle's re-diagnosis passes and the ablations all
    resolve it independently; loading the model once matters."""
    from tiger.encoders import probe_encoder_from_cfg
    cfg = {"data": {"cache_dir": "data/cache_embeddings"},
           "models": {"clip_model_name": "openai/clip-vit-base-patch32",
                      "probe_model_name": "google/siglip-base-patch16-224"}}
    a = probe_encoder_from_cfg(cfg, tmp_path)
    b = probe_encoder_from_cfg(cfg, tmp_path)
    assert a is not None and a is b
    assert a.model_name == "google/siglip-base-patch16-224"


# ---------------------------------------------------------------------------
# the shipped configuration (B7, decided by measurement 2026-09-14)
# ---------------------------------------------------------------------------

def test_shipped_config_probes_on_siglip_and_scores_on_clip():
    """The probes move, the reported baseline does not. If these ever collapse
    to one model the detection numbers stop being comparable to the published
    CLIP baseline, and nothing else would notice."""
    from tiger import cli
    from tiger.encoders import probe_encoder_from_cfg

    cfg = cli.load_cfg()
    assert cfg["models"]["clip_model_name"] == "openai/clip-vit-base-patch32"
    probe = probe_encoder_from_cfg(cfg, cli.ROOT)
    assert probe is not None, "probe_model_name is unset; B7's measured swap is not live"
    assert probe.model_name == "google/siglip-base-patch16-224"
    assert probe.model_name != cfg["models"]["clip_model_name"]


def test_probe_prompts_fit_siglips_shorter_context():
    """SigLIP pads to a fixed 64-token context where CLIP truncates at 77. The
    probe encoder only ever sees field-caption templates -- never the full
    captions or titles, which stay on CLIP -- so the budget is not tight, but
    an unbounded category noun would silently truncate (cf. F2)."""
    from tiger import text_views
    from tiger.schema import load_schema

    schema = load_schema(ROOT_DIR / "configs/schema.yaml")
    longest = 0
    for cat in schema.categories:
        for fld in ("color", "material", "pattern"):
            for v in schema.domain(fld):
                for t in text_views.field_caption_templates(cat, fld, v):
                    longest = max(longest, len(t.split()))
    assert longest <= 20, f"longest probe prompt is {longest} words; SigLIP pads to 64 tokens"
