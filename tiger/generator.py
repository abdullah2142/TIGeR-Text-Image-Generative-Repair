"""Stable Diffusion image generation fallback for missing catalogue items."""

from pathlib import Path

class StableDiffusionGenerator:
    def __init__(self, device: str = "cuda", model_id: str = "runwayml/stable-diffusion-v1-5"):
        try:
            import os
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            from diffusers import StableDiffusionPipeline
            import diffusers
            import transformers
            diffusers.logging.set_verbosity_error()
            transformers.logging.set_verbosity_error()
            
            import torch
        except ImportError:
            raise ImportError("Please install diffusers to use generative fallback: pip install -e '.[gen]'")
            
        print(f"Loading Generative Fallback Model: {model_id} (quiet mode)...")
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

    def generate(self, caption: str, out_path: Path) -> Path:
        """Generate a product image matching the caption and save to out_path."""
        import torch
        prompt = f"A simple product photo of a {caption} on a white background"
        print(f"[Generative Fallback] Synthesizing: '{prompt}'")
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.seed += 1
        
        # 20 steps is a good balance for SD v1.5 speed/quality
        image = self.pipe(prompt, generator=generator, num_inference_steps=20).images[0]
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="JPEG", quality=92)
        return out_path
