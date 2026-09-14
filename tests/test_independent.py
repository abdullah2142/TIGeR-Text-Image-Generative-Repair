"""IndependentVerifier veto behaviour (roadmap 6.4 partial), no real model.

A FakeEncoder maps colour words (in captions) and colour-named image paths to
one-hot vectors over the colour domain, so the SigLIP-style probe is fully
deterministic and we can assert the verifier both confirms and vetoes.
"""

from pathlib import Path

import numpy as np
import pytest

from tiger.schema import load_schema
from tiger.verify import IndependentVerifier

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schema():
    return load_schema(ROOT / "configs/schema.yaml")


class FakeEncoder:
    """Deterministic stand-in: colour word in a caption / path -> one-hot."""

    def __init__(self, colors):
        self.colors = colors
        self.index = {c: i for i, c in enumerate(colors)}

    def _onehot(self, color):
        v = np.zeros(len(self.colors), dtype=np.float32)
        if color in self.index:
            v[self.index[color]] = 1.0
        return v

    def _color_in(self, text):
        for c in self.colors:
            if c in text.lower():
                return c
        return None

    def encode_texts(self, texts):
        return np.stack([self._onehot(self._color_in(t)) for t in texts])

    def encode_images(self, paths):
        embs, ok = [], []
        for p in paths:
            c = self._color_in(str(p))
            embs.append(self._onehot(c))
            ok.append(c is not None)
        return np.stack(embs), np.array(ok)

    def save_cache(self):
        pass


def test_v2t_confirms_matching_patch(schema):
    colors = schema.domain("color")
    iv = IndependentVerifier(FakeEncoder(colors), schema)
    # image is blue, patch sets colour to blue -> independent encoder agrees
    assert iv.check_v2t("/x/blue_shirt.jpg", "shirts", "color", "blue") is True


def test_v2t_vetoes_mismatching_patch(schema):
    colors = schema.domain("color")
    iv = IndependentVerifier(FakeEncoder(colors), schema)
    # image is red, patch claims green -> independent encoder disagrees
    assert iv.check_v2t("/x/red_shirt.jpg", "shirts", "color", "green") is False


def test_t2v_confirms_better_candidate(schema):
    colors = schema.domain("color")
    iv = IndependentVerifier(FakeEncoder(colors), schema)
    # caption is blue; new image blue beats old image red
    assert iv.check_t2v("/x/red.jpg", "/x/blue.jpg", "a photo of a blue shirt") is True


def test_t2v_vetoes_worse_candidate(schema):
    colors = schema.domain("color")
    iv = IndependentVerifier(FakeEncoder(colors), schema)
    # caption is blue; new image green does not beat old image blue
    assert iv.check_t2v("/x/blue.jpg", "/x/green.jpg", "a photo of a blue shirt") is False


def test_unreadable_image_not_vetoed_on_v2t(schema):
    colors = schema.domain("color")
    iv = IndependentVerifier(FakeEncoder(colors), schema)
    # path has no colour -> FakeEncoder marks image not-ok -> do not veto
    assert iv.check_v2t("/x/unnamed.jpg", "shirts", "color", "blue") is True


# ---------------------------------------------------------------------------
# D17 · the T2V check must be able to reject
# ---------------------------------------------------------------------------

class _RankEncoder:
    """Independent encoder with its own opinion about which image fits.

    Images embed as a one-hot over `order`, the caption as a weight vector, so
    the encoder's preference between candidates is whatever `order` says --
    independent of whichever candidate the primary encoder chose.
    """

    def __init__(self, preference: dict[str, float]):
        self.preference = preference

    def encode_images(self, paths):
        import numpy as np
        return (np.array([[self.preference.get(str(p), 0.0)] for p in paths], dtype=float),
                np.ones(len(paths), dtype=bool))

    def encode_texts(self, texts):
        import numpy as np
        return np.array([[1.0]] * len(texts), dtype=float)

    def save_cache(self):
        pass


def _verifier(preference, schema):
    return IndependentVerifier(_RankEncoder(preference), schema)


def test_t2v_accepts_when_the_independent_encoder_agrees_on_the_choice(schema):
    v = _verifier({"old.jpg": 0.1, "new.jpg": 0.9, "runner.jpg": 0.5}, schema)
    assert v.check_t2v("old.jpg", "new.jpg", "a red chair", runner_up_image_path="runner.jpg")


def test_t2v_rejects_when_the_independent_encoder_prefers_the_runner_up(schema):
    """The case that made this check vacuous: the new image beats the OLD one --
    it was selected to -- but the second encoder would have picked differently.
    Under the old direction-only test this returned True, always."""
    v = _verifier({"old.jpg": 0.1, "new.jpg": 0.5, "runner.jpg": 0.9}, schema)
    assert not v.check_t2v("old.jpg", "new.jpg", "a red chair", runner_up_image_path="runner.jpg")


def test_t2v_still_rejects_a_swap_that_is_worse_than_the_original(schema):
    v = _verifier({"old.jpg": 0.9, "new.jpg": 0.2, "runner.jpg": 0.1}, schema)
    assert not v.check_t2v("old.jpg", "new.jpg", "a red chair", runner_up_image_path="runner.jpg")


def test_t2v_falls_back_to_the_direction_test_with_no_runner_up(schema):
    """Single-candidate case: there is no second choice to agree about."""
    v = _verifier({"old.jpg": 0.1, "new.jpg": 0.5}, schema)
    assert v.check_t2v("old.jpg", "new.jpg", "a red chair")
    assert not v.check_t2v("new.jpg", "old.jpg", "a red chair")
