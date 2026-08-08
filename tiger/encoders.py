"""CLIP encoder wrapper with a content-hash embedding cache.

Cache keys are sha1 hashes of the text string or the image file bytes, so a
repaired row (new image bytes / new caption) can never silently reuse a stale
embedding (review A1.4). The cache is a single NPZ per model living in
data/cache_embeddings/.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _as_tensor(out):
    """transformers <5 returns a tensor; >=5 returns a model output whose
    pooler_output holds the projected features."""
    if hasattr(out, "pooler_output"):
        return out.pooler_output
    return out


class ClipEncoder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32,
                 cache_dir: str | Path | None = None):
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self._model = None
        self._processor = None

        self._cache_path = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_dirty = False
        if cache_dir is not None:
            safe = model_name.replace("/", "_")
            self._cache_path = Path(cache_dir) / f"embcache_{safe}.npz"
            if self._cache_path.exists():
                z = np.load(self._cache_path)
                self._cache = {k: z[k] for k in z.files}

    @property
    def is_siglip(self) -> bool:
        return "siglip" in self.model_name.lower()

    # ---------- lazy model (family-agnostic: CLIP, SigLIP, ...) ----------

    def _ensure_model(self):
        if self._model is None:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoProcessor
            import transformers
            transformers.logging.set_verbosity_error()

            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval().to(self.device)
            try:
                self._processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
            except TypeError:
                self._processor = AutoProcessor.from_pretrained(self.model_name)

    def _text_inputs(self, texts: list[str]):
        # SigLIP was trained with a fixed 64-token padded context; CLIP pads to
        # the longest in-batch sequence and truncates at 77.
        if self.is_siglip:
            return self._processor(text=texts, return_tensors="pt",
                                   padding="max_length", truncation=True)
        return self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)

    @property
    def dim(self) -> int:
        self._ensure_model()
        for attr in ("projection_dim",):
            v = getattr(self._model.config, attr, None)
            if v:
                return int(v)
        tc = getattr(self._model.config, "text_config", None)
        if tc is not None and getattr(tc, "hidden_size", None):
            return int(tc.hidden_size)
        return 512

    # ---------- cache ----------

    @staticmethod
    def text_key(text: str) -> str:
        return "t_" + hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def image_key(path: str | Path) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return "i_" + h.hexdigest()

    def save_cache(self):
        if self._cache_path is not None and self._cache_dirty:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self._cache_path, **self._cache)
            self._cache_dirty = False

    # ---------- encoding ----------

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """L2-normalised text embeddings, cached by content hash."""
        keys = [self.text_key(t) for t in texts]
        missing = [(i, t) for i, (k, t) in enumerate(zip(keys, texts)) if k not in self._cache]
        if missing:
            self._ensure_model()
            import torch

            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start: start + self.batch_size]
                inputs = self._text_inputs([t for _, t in chunk])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    emb = _as_tensor(self._model.get_text_features(**inputs))
                emb = emb.detach().cpu().numpy().astype(np.float32)
                emb = _l2(emb)
                for (i, _), e in zip(chunk, emb):
                    self._cache[keys[i]] = e
                    self._cache_dirty = True
        return np.stack([self._cache[k] for k in keys])

    def encode_images(self, paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        """L2-normalised image embeddings, cached by file-content hash.

        Returns (embeddings, ok_mask); rows whose image failed to load get a
        zero vector and ok=False.
        """
        from PIL import Image

        n = len(paths)
        out = np.zeros((n, self.dim if self._cache_path is None or not self._cache else
                        len(next(iter(self._cache.values())))), dtype=np.float32) \
            if self._cache else None
        # simpler: resolve dim lazily below
        keys: list[str | None] = []
        ok = np.zeros(n, dtype=bool)
        for p in paths:
            try:
                keys.append(self.image_key(p))
            except OSError:
                keys.append(None)

        missing: list[tuple[int, "Image.Image"]] = []
        for i, (p, k) in enumerate(zip(paths, keys)):
            if k is None:
                continue
            if k in self._cache:
                ok[i] = True
                continue
            try:
                with Image.open(p) as im:
                    missing.append((i, im.convert("RGB")))
                ok[i] = True
            except Exception:
                keys[i] = None

        if missing:
            self._ensure_model()
            import torch

            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start: start + self.batch_size]
                inputs = self._processor(images=[im for _, im in chunk], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    emb = _as_tensor(self._model.get_image_features(**inputs))
                emb = emb.detach().cpu().numpy().astype(np.float32)
                emb = _l2(emb)
                for (i, _), e in zip(chunk, emb):
                    self._cache[keys[i]] = e
                    self._cache_dirty = True

        dim = len(next(iter(self._cache.values()))) if self._cache else self.dim
        out = np.zeros((n, dim), dtype=np.float32)
        for i, k in enumerate(keys):
            if k is not None and k in self._cache:
                out[i] = self._cache[k]
                ok[i] = True
            else:
                ok[i] = False
        return out, ok
