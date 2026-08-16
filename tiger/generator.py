"""Stable Diffusion image generation fallback for missing catalogue items."""

from pathlib import Path

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
        attrs = attrs or {}
        
        color = attrs.get("color", "")
        cat_singular = category.rstrip("s") if category else ""
        
        # SDXL understands natural language much better, so we just construct a clear sentence
        subject = f"{color} {cat_singular}" if color and cat_singular else caption
        
        prompt = f"Professional studio product photo of a single {subject}, perfectly centered on a pure bright white background, studio lighting"
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
