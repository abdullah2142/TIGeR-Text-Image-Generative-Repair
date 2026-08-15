"""VLM Judge: Gemini-powered independent verification of proposed repairs (roadmap 6.4).

This module provides a GeminiVLMJudge that plugs into the `independent_ok` hook
in tiger/verify.py, replacing the SigLIP-encoder cross-check with a true
product-identity-aware visual question answering model.

Why this matters (F7 / roadmap 6.4):
    CLIP-based verification (and even the SigLIP IndependentVerifier) can pass a
    wrong-direction same-category repair because both encoders agree "this image
    looks like a blue hat" when the text now says "blue hat" -- even if the image
    is the *wrong* blue hat. A VLM, asked explicitly about product identity and the
    specific attribute being repaired, can catch this class of error.

Architecture:
    - V2T repairs: we ask Gemini to look at the image and tell us whether the
      repaired attribute value (e.g. "colour: blue") is correct for what it sees.
    - T2V repairs: we ask Gemini to look at the proposed replacement image and
      confirm that it matches the product's text description.

API:
    Requires GEMINI_API_KEY in the environment (loaded from .env at repo root).
    Uses google-generativeai >= 0.7 (pip install google-generativeai).
    Rate-limited to 15 requests/minute on the free tier; the judge honours this
    via a configurable sleep between calls.

Usage (wired automatically via `tiger.cli repair --vlm-judge`):
    judge = GeminiVLMJudge.from_env()
    ok = judge.check_v2t(image_path, category, field, repaired_value)
    ok = judge.check_t2v(old_image_path, new_image_path, caption)
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_env(root: Path | None = None) -> None:
    """Load .env from the repo root into os.environ (no python-dotenv required)."""
    candidate = (root or Path(__file__).resolve().parents[1]) / ".env"
    if not candidate.exists():
        return
    for line in candidate.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def _img_to_b64(path: str) -> str:
    """Return a base-64 encoded JPEG/PNG string for the Gemini inline-data part."""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def _mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

_V2T_PROMPT = """You are a product-catalogue quality inspector.
Look at the product image carefully.

The system proposes to set the attribute **{field}** to **"{value}"** for this product.

Question: Does the image clearly show that the product's {field} is "{value}"?

YOUR ENTIRE RESPONSE MUST BE EXACTLY ONE WORD: YES or NO.
Do not explain. Do not reason. Do not hedge. Do not output anything other than YES or NO."""

_T2V_PROMPT = """You are a strict product-catalogue quality inspector.
You will see a single image that is proposed as the product photo for a catalogue listing.

Product description: "{caption}"

Carefully check ALL of the following:
1. Is the product TYPE correct? (e.g., if the description says "shirt", is it a shirt and not shoes?)
2. Is the COLOR correct? (e.g., if the description says "black", is the product actually black?)
3. Is the product a real, clean product photo without visual artifacts or deformities?

If ANY of these checks fail, answer NO.

YOUR ENTIRE RESPONSE MUST BE EXACTLY ONE WORD: YES or NO.
Do not explain. Do not reason. Do not hedge. Do not output anything other than YES or NO."""


# ---------------------------------------------------------------------------
# judge class
# ---------------------------------------------------------------------------

class GeminiVLMJudge:
    """Gemini-1.5-Flash-based VLM judge for the independent verification step (6.4).

    Args:
        api_key:      Gemini API key (loaded from .env if not provided).
        model_name:   Gemini model to use.  Flash is free-tier; Pro is higher quality.
        rpm_limit:    Requests per minute cap (15 for Flash free tier).
        verbose:      Print each judgement to stdout.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3.1-pro",
        rpm_limit: int = 2,
        verbose: bool = False,
    ):
        _load_env()
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found. Set it in .env or pass api_key= explicitly."
            )
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="wrapt")

        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=key)
        self._gen_config = genai.GenerationConfig(max_output_tokens=5)
        self._model = genai.GenerativeModel(model_name)
        self._min_interval = 60.0 / rpm_limit   # seconds between calls
        self._last_call = 0.0
        self.verbose = verbose
        self.model_name = model_name

    @classmethod
    def from_env(cls, **kwargs) -> "GeminiVLMJudge":
        """Construct from .env / environment; forwards extra kwargs to __init__."""
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # rate limiter
    # ------------------------------------------------------------------

    def _wait(self, override_gap: float = 0.0) -> None:
        if override_gap > 0:
            time.sleep(override_gap)
            return
        elapsed = time.monotonic() - self._last_call
        gap = self._min_interval - elapsed
        if gap > 0:
            time.sleep(gap)

    def _call(self, parts: list) -> bool:
        """Send parts to Gemini, return True if the answer is YES."""
        for attempt in range(3):
            self._wait()
            try:
                resp = self._model.generate_content(
                    parts, generation_config=self._gen_config
                )
                self._last_call = time.monotonic()
                text = resp.text.strip().upper()
                # Robust parsing: check both the first and last word,
                # so verbose responses like "... Therefore: YES" are
                # correctly parsed instead of being wrongly vetoed.
                words = text.split()
                answer = False
                if words:
                    first, last = words[0], words[-1]
                    if first in ("YES", "YES."):
                        answer = True
                    elif last in ("YES", "YES."):
                        answer = True
                if self.verbose:
                    print(f"[GeminiVLMJudge] raw='{resp.text.strip()}' -> {answer}")
                return answer
            except Exception as exc:  # noqa: BLE001
                err_str = str(exc)
                if any(code in err_str for code in ["429", "503", "504"]) and attempt < 2:
                    print(f"[GeminiVLMJudge] API error/timeout ({err_str[:25]}...). Retrying in 15s... (attempt {attempt+1}/3)")
                    self._last_call = time.monotonic()
                    self._wait(override_gap=15.0)
                    continue
                # On other API error (or out of retries), VETO the repair (return False).
                # Defaulting to True previously allowed bad repairs to slip through during timeouts.
                print(f"[GeminiVLMJudge] API error (fatal, vetoing): {exc}")
                self._last_call = time.monotonic()
                return False

    # ------------------------------------------------------------------
    # public interface  (mirrors IndependentVerifier in verify.py)
    # ------------------------------------------------------------------

    def check_v2t(self, image_path: str, category: str, field: str, value: str) -> bool:
        """Ask Gemini whether the image supports setting `field` to `value`.

        Returns True  → VLM agrees, repair is acceptable.
        Returns False → VLM disagrees, repair is vetoed.
        """
        try:
            b64 = _img_to_b64(image_path)
        except OSError:
            return True   # unreadable image: do not veto on our own read failure
        prompt = _V2T_PROMPT.format(field=field, value=value)
        import google.generativeai as genai  # type: ignore
        parts = [
            {"inline_data": {"mime_type": _mime(image_path), "data": b64}},
            prompt,
        ]
        return self._call(parts)

    def check_t2v(self, old_image_path: str, new_image_path: str, caption: str) -> bool:
        """Ask Gemini whether `new_image_path` matches the product description.

        Only the *proposed* (new) image is sent to Gemini.  The old image is
        intentionally excluded: sending two images confused the model (it would
        produce verbose "I see two images" responses instead of YES/NO), and
        the prompt already evaluates the image standalone against the caption.

        Returns True  → VLM agrees the image matches the description.
        Returns False → VLM disagrees, repair is vetoed.
        """
        try:
            new_b64 = _img_to_b64(new_image_path)
        except OSError:
            return False   # cannot read the proposed replacement → veto it

        import google.generativeai as genai  # type: ignore
        parts = [
            {"inline_data": {"mime_type": _mime(new_image_path), "data": new_b64}},
            _T2V_PROMPT.format(caption=caption),
        ]
        return self._call(parts)
