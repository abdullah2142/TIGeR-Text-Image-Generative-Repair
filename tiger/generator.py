"""Stable Diffusion image generation fallback for missing catalogue items."""

from pathlib import Path

class StableDiffusionGenerator:
    def __init__(self, device: str = "cuda", model_id: str = "runwayml/stable-diffusion-v1-5"):
        try:
            import os
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*Flax classes.*")
            
            from diffusers import StableDiffusionPipeline
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
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=False
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe = self.pipe.to(device)
        self.seed = 42

    def generate(self, caption: str, out_path: Path, category: str = "", attrs: dict = None) -> Path:
        """Generate a product image matching the caption and attributes, and save to out_path."""
        import torch
        attrs = attrs or {}
        
        # Build an attribute-heavy prompt to force Stable Diffusion adherence
        color = attrs.get("color", "")
        # Remove trailing 's' from category (e.g., 'shirts' -> 'shirt') for better image generation
        cat_singular = category.rstrip("s") if category else ""
        
        # Give a 30% boost to color and 20% to category to force SD1.5 to obey them
        subject = f"({color}:1.3) ({cat_singular}:1.2)" if color and cat_singular else caption
        
        prompt = f"Professional studio product photo of a single {subject}, high resolution, 8k, sharp focus, perfectly centered on a pure bright white background, studio lighting"
        negative_prompt = "blurry, cropped, distorted, text, watermark, low quality, bad anatomy, deformed, background details, multiple items, noisy, messy, person, model, human"
        print(f"[Generative Fallback] Synthesizing: '{color} {cat_singular}' (from '{caption}')")
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.seed += 1
        
        # Increased steps and added negative prompt for much better quality
        image = self.pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            generator=generator, 
            num_inference_steps=35, 
            guidance_scale=7.5
        ).images[0]
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="JPEG", quality=92)
        return out_path
