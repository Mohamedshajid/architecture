from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class SmallStableDiffusionRuntime:
    """CPU Diffusers runtime for OFA-Sys/small-stable-diffusion-v0."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return

        from diffusers import StableDiffusionPipeline
        import torch

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Small Stable Diffusion model path does not exist: {self.model_path}"
            )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float32,
            safety_checker=None,
        )

        self.pipeline = self.pipeline.to("cpu")
        self.pipeline.set_progress_bar_config(disable=True)

        self.loaded = True

    def unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        self.loaded = False

        try:
            import gc
            gc.collect()

            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 10,
        guidance_scale: float = 7.5,
        cancellation: threading.Event | None = None,
    ) -> dict[str, Any]:

        if not self.loaded or self.pipeline is None:
            raise RuntimeError("Small Stable Diffusion runtime is not loaded")

        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("Image generation requires a prompt")

        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("image generation cancelled")

        width = max(256, min(int(width), 768))
        height = max(256, min(int(height), 768))
        num_inference_steps = max(1, min(int(num_inference_steps), 30))
        guidance_scale = max(1.0, min(float(guidance_scale), 15.0))

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("image generation cancelled")

        image = result.images[0]

        output_dir = Path("data/images")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"sd_{__import__('uuid').uuid4().hex}.png"
        image.save(str(output_path), format="PNG")

        return {
            "image_path": str(output_path.resolve()),
            "width": image.width,
            "height": image.height,
            "prompt": prompt,
            "runtime": "small-stable-diffusion",
            "steps": num_inference_steps,
        }
