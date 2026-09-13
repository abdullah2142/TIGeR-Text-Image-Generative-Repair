"""Stable Diffusion image generation fallback for missing catalogue items."""

from pathlib import Path

from tiger import text_views


def build_prompt(caption: str, category: str = "", attrs: dict | None = None) -> tuple[str, str]:
    """Text-to-image prompt for a repaired row. Returns ``(prompt, subject)``.

    D10: colour, material *and* pattern all reach the prompt. Only colour and
    category used to, while `honest_limitations.md` explained the loss of
    "striped"/"printed" as diffusion models struggling with fine-grained
    pattern adherence -- an attribution the code could not support, because the
    pattern was discarded before generation.

    D8/D9: the category noun comes from `text_views.singular`, the same table
    the probe and LOO captions use. A local `removesuffix("s")` here meant the
    generator asked SDXL for a "wall_art" and a "light_fixture" -- underscore
    tokens, and not nouns -- for exactly the categories D8 added real nouns for.

    Pure and free of the diffusers import, so what the model is actually asked
    for is testable without a GPU.
    """
    attrs = attrs or {}
    color = str(attrs.get("color", "") or "")
    material = str(attrs.get("material", "") or "")
    pattern = str(attrs.get("pattern", "") or "")
    cat_singular = text_views.singular(category) if category else ""

    descriptors = " ".join(d for d in (color, material) if d)
    subject = f"{descriptors} {cat_singular}" if descriptors and cat_singular else caption
    if pattern and pattern != "solid" and cat_singular:
        subject = f"{subject} with a {pattern} pattern"

    prompt = (f"Professional studio product photo of a single {subject}, "
              f"perfectly centered on a pure bright white background, studio lighting")
    return prompt, subject


class StableDiffusionGenerator:
    def __init__(self, device: str = "cuda", model_id: str = "stabilityai/sdxl-turbo"):
        try:
            import os
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*Flax classes.*")
            
            from diffusers import AutoPipelineForText2Image
            import diffusers
            import transformers
            diffusers.logging.set_verbosity_error()
            transformers.logging.set_verbosity_error()
            
            import logging as _logging
            _logging.getLogger("diffusers").setLevel(_logging.ERROR)
            _logging.getLogger("transformers").setLevel(_logging.ERROR)
            _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
            
            import torch
        except ImportError:
            raise ImportError("Please install diffusers to use generative fallback: pip install -e '.[gen]'")
            
        self.device = device
        dtype = torch.float16 if "cuda" in device else torch.float32
        variant = "fp16" if dtype == torch.float16 else None
        
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            variant=variant,
            local_files_only=False
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe = self.pipe.to(device)
        self.seed = 42

    def generate(self, caption: str, out_path: Path, category: str = "", attrs: dict = None) -> Path:
        """Generate a product image matching the caption and attributes, and save to out_path."""
        import torch

        prompt, subject = build_prompt(caption, category, attrs)
        print(f"[Generative Fallback] Synthesizing (SDXL-Turbo): '{subject}'")
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.seed += 1
        
        # SDXL-Turbo generates in 1-4 steps with guidance_scale 0.0
        image = self.pipe(
            prompt=prompt, 
            generator=generator, 
            num_inference_steps=4, 
            guidance_scale=0.0
        ).images[0]
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="JPEG", quality=92)
        return out_path
