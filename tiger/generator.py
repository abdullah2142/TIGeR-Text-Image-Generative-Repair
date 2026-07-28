"""Stable Diffusion image generation fallback for missing catalogue items."""

from pathlib import Path

class StableDiffusionGenerator:
    def __init__(self, device: str = "cuda", model_id: str = "runwayml/stable-diffusion-v1-5"):
        try:
            from diffusers import StableDiffusionPipeline
            import torch
        except ImportError:
            raise ImportError("Please install diffusers to use generative fallback: pip install -e '.[gen]'")
            
        print(f"Loading Generative Fallback Model: {model_id}...")
        self.device = device
        dtype = torch.float16 if "cuda" in device else torch.float32
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            safety_checker=None
        )
        self.pipe = self.pipe.to(device)
        self.seed = 42

    def generate(self, caption: str, out_path: Path) -> Path:
        """Generate a product image matching the caption and save to out_path."""
        import torch
        prompt = f"A studio product photograph of a {caption}, isolated on a pure white background, highly detailed, 4k"
        print(f"[Generative Fallback] Synthesizing: '{prompt}'")
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.seed += 1
        
        # 20 steps is a good balance for SD v1.5 speed/quality
        image = self.pipe(prompt, generator=generator, num_inference_steps=20).images[0]
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="JPEG", quality=92)
        return out_path
