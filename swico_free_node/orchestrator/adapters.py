from __future__ import annotations
import re

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Any

from .models import Capability, ModelLifecycleState, ModelProfile


class ModelAdapter(ABC):
    def __init__(self, profile: ModelProfile):
        self.profile = profile
        self.state = ModelLifecycleState.READY.value if profile.integration_status in ("READY", "MOCK") else ModelLifecycleState.INSTALLED.value
        self.last_load_time: float | None = None
        self.last_unload_time: float | None = None

    async def load(self) -> None:
        import time
        if self.state == ModelLifecycleState.LOADED.value:
            return
        self.state = ModelLifecycleState.LOADING.value
        try:
            self.state = ModelLifecycleState.LOADED.value
            self.last_load_time = time.time()
        except Exception:
            self.state = ModelLifecycleState.LOAD_FAILED.value
            raise

    async def unload(self) -> None:
        import time
        if self.state in (ModelLifecycleState.INSTALLED.value, ModelLifecycleState.READY.value, ModelLifecycleState.EVICTED.value):
            return
        self.state = ModelLifecycleState.UNLOADING.value
        self.state = ModelLifecycleState.EVICTED.value
        self.last_unload_time = time.time()

    async def health(self) -> str:
        return "ok"

    def estimate_resources(self, _input: Any) -> dict[str, int]:
        return {"ram_mb": self.profile.ram_requirement_mb, "cpu": self.profile.cpu_requirement}

    @abstractmethod
    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        raise NotImplementedError

    async def cancel(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {"model_id": self.profile.model_id, "capability": self.profile.capability.value, "runtime": self.profile.runtime, "state": self.state, "integration_status": self.profile.integration_status, "loaded": self.state in (ModelLifecycleState.LOADED.value, ModelLifecycleState.EXECUTING.value), "health": "ok" if self.state != ModelLifecycleState.UNHEALTHY.value else "unhealthy", "last_load_time": self.last_load_time, "last_unload_time": self.last_unload_time, "cancellation_supported": self.profile.cancellation_supported}


class MockAdapter(ModelAdapter):
    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        await asyncio.sleep(0)
        if cancellation.is_set():
            raise asyncio.CancelledError()
        text = input_data if isinstance(input_data, str) else str(input_data)
        return {"text": f"[{self.profile.capability.value} adapter] {text}", "mock": True}


class QwenTextAdapter(ModelAdapter):
    """Adapter that keeps orchestration above the existing protected Qwen runtime."""
    def __init__(self, profile: ModelProfile, runtime: Any, max_tokens: int = 256):
        super().__init__(profile)
        self.runtime = runtime
        self.max_tokens = max_tokens

    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        max_tokens = self.max_tokens
        if isinstance(input_data, dict):
            messages = input_data.get("messages", [])
            requested_max_tokens = input_data.get("max_output_tokens")
            if requested_max_tokens is not None:
                max_tokens = min(int(requested_max_tokens), self.max_tokens)
        else:
            messages = input_data if isinstance(input_data, list) else [{"role": "user", "content": str(input_data)}]

        text, usage = await asyncio.to_thread(self.runtime.generate, messages, max_tokens, cancellation)
        return {"text": text, "usage": usage, "runtime": "qwen"}


class ChatModelAdapter(QwenTextAdapter):
    """Chat-specific contract over the shared text runtime."""
    async def stream(self, input_data: Any, cancellation: threading.Event):
        result = await self.execute(input_data, cancellation)
        yield result.get("text", ""), result.get("usage", {})


class CodingModelAdapter(QwenTextAdapter):
    """Coding-specific contract over a separately supplied Qwen runtime."""
    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        result = await super().execute(input_data, cancellation)
        result["runtime"] = "qwen-coding"
        return result


class WhisperModelAdapter(ModelAdapter):
    """Speech-to-text adapter over a separately supplied local Whisper runtime."""
    def __init__(self, profile: ModelProfile, runtime: Any):
        super().__init__(profile)
        self.runtime = runtime

    async def load(self) -> None:
        await super().load()
        if hasattr(self.runtime, "load"):
            try:
                await asyncio.to_thread(self.runtime.load)
            except Exception:
                self.state = "LOAD_FAILED"
                raise

    async def unload(self) -> None:
        if hasattr(self.runtime, "unload"):
            await asyncio.to_thread(self.runtime.unload)
        await super().unload()

    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        if isinstance(input_data, dict):
            audio_path = input_data.get("audio_path")
            language = input_data.get("language")
        else:
            audio_path, language = input_data, None
        if not audio_path:
            raise ValueError("STT input requires audio_path")
        return await asyncio.to_thread(
            self.runtime.transcribe, __import__("pathlib").Path(str(audio_path)), cancellation, language,
        )


class MockChatAdapter(MockAdapter):
    async def stream(self, input_data: Any, cancellation: threading.Event):
        result = await self.execute(input_data, cancellation)
        yield result.get("text", ""), {"mock": True}


class ModelUnavailableError(RuntimeError):
    pass


class SpecializedUnavailableAdapter(ModelAdapter):
    """Contract adapter for a selected model whose runtime/weights are not installed."""
    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        raise ModelUnavailableError(f"{self.profile.model_id} is not installed; configure its runtime and model path")


class ImageModelAdapter(ModelAdapter):
    """Image generation adapter for the local Small Stable Diffusion runtime."""

    def __init__(self, profile: ModelProfile, runtime: Any):
        super().__init__(profile)
        self.runtime = runtime

    async def load(self) -> None:
        if self.state == ModelLifecycleState.LOADED.value:
            return

        self.state = ModelLifecycleState.LOADING.value

        try:
            await asyncio.to_thread(self.runtime.load)
            self.state = ModelLifecycleState.LOADED.value
        except Exception:
            self.state = ModelLifecycleState.LOAD_FAILED.value
            raise

    async def unload(self) -> None:
        if hasattr(self.runtime, "unload"):
            await asyncio.to_thread(self.runtime.unload)

        self.state = ModelLifecycleState.EVICTED.value

    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        if isinstance(input_data, dict):
            prompt = input_data.get("prompt") or input_data.get("text")
            negative_prompt = input_data.get("negative_prompt")
            width = input_data.get("width", 512)
            height = input_data.get("height", 512)
            num_inference_steps = input_data.get("num_inference_steps", 10)
            guidance_scale = input_data.get("guidance_scale", 7.5)
        else:
            prompt = str(input_data)
            negative_prompt = None
            width = 512
            height = 512
            num_inference_steps = 10
            guidance_scale = 7.5

        if not prompt:
            raise ValueError("Image generation requires prompt")

        if cancellation.is_set():
            raise asyncio.CancelledError()

        return await asyncio.to_thread(
            self.runtime.generate,
            str(prompt),
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            cancellation=cancellation,
        )


class VideoModelAdapter(SpecializedUnavailableAdapter):
    async def progress(self, _task_id: str) -> dict[str, Any]:
        return {"status": "unavailable", "progress": 0}


class STTModelAdapter(SpecializedUnavailableAdapter):
    pass


class TTSModelAdapter(SpecializedUnavailableAdapter):
    pass


class KokoroModelAdapter(ModelAdapter):
    """Adapter boundary for a supplied Kokoro runtime; no implicit fallback."""
    def __init__(self, profile: ModelProfile, runtime: Any):
        super().__init__(profile)
        self.runtime = runtime

    async def load(self) -> None:
        await asyncio.to_thread(self.runtime.load)
        self.state = ModelLifecycleState.LOADED.value

    async def unload(self) -> None:
        await asyncio.to_thread(self.runtime.unload)
        self.state = ModelLifecycleState.EVICTED.value

    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        request = input_data if isinstance(input_data, dict) else {"text": str(input_data)}
        text = str(request.get("text", "")).strip()
        if not text:
            raise ValueError("TTS input requires text")
        return await asyncio.to_thread(self.runtime.synthesize, text, request.get("voice", "af_heart"), request.get("speed", 1.0), cancellation)


class E5ModelAdapter(ModelAdapter):
    """Health/lifecycle boundary for the one E5 runtime used by retrieval."""
    def __init__(self, profile: ModelProfile, runtime: Any):
        super().__init__(profile)
        self.runtime = runtime
        if getattr(runtime, "loaded", False):
            self.state = "LOADED"

    async def execute(self, input_data: Any, cancellation: threading.Event) -> Any:
        if cancellation.is_set():
            raise asyncio.CancelledError()
        texts = input_data.get("texts", []) if isinstance(input_data, dict) else list(input_data)
        modes = input_data.get("modes", ["query"] * len(texts)) if isinstance(input_data, dict) else ["query"] * len(texts)
        return await asyncio.to_thread(self.runtime.embed, texts, modes)


class DocumentAnalysisModelAdapter(ModelAdapter):
    """CPU SmolDocling-256M adapter for PDF/image document analysis."""

    def __init__(self, profile: ModelProfile):
        super().__init__(profile)
        self.processor = None
        self.model = None

    async def load(self) -> None:
        if self.state == ModelLifecycleState.LOADED.value:
            return

        self.state = ModelLifecycleState.LOADING.value

        try:
            import torch
            from transformers import AutoProcessor, Idefics3ForConditionalGeneration

            model_path = self.profile.model_path
            if not model_path:
                raise RuntimeError("SmolDocling model path is not configured.")

            self.processor = await asyncio.to_thread(
                AutoProcessor.from_pretrained,
                model_path,
            )

            self.model = await asyncio.to_thread(
                Idefics3ForConditionalGeneration.from_pretrained,
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
            )

            self.model.eval()
            self.state = ModelLifecycleState.LOADED.value

        except Exception:
            self.processor = None
            self.model = None
            self.state = ModelLifecycleState.LOAD_FAILED.value
            raise

    async def execute(self, input_data: Any, cancellation: threading.Event | None = None) -> Any:
        if self.state != ModelLifecycleState.LOADED.value:
            await self.load()

        if not isinstance(input_data, dict):
            raise ValueError("Document analysis input must contain prompt and document_path.")

        document_path = input_data.get("document_path")
        prompt = str(
            input_data.get(
                "prompt",
                "Analyze this document and summarize its main content.",
            )
        )

        if not document_path:
            raise ValueError("document_path is required for document analysis.")

        document_path = str(document_path)

        if not document_path.startswith("data/documents/"):
            raise ValueError("Document path must be inside data/documents/.")

        from pathlib import Path

        path = Path(document_path).resolve()
        allowed_dir = Path("data/documents").resolve()

        if allowed_dir not in path.parents:
            raise ValueError("Document path is outside the allowed documents directory.")

        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {document_path}")

        if cancellation and cancellation.is_set():
            raise asyncio.CancelledError()

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            import pymupdf
            from PIL import Image
            import io

            pdf = await asyncio.to_thread(pymupdf.open, str(path))

            try:
                page_count = min(len(pdf), 8)
                results = []

                for page_index in range(page_count):
                    if cancellation and cancellation.is_set():
                        raise asyncio.CancelledError()

                    page = pdf[page_index]
                    pix = await asyncio.to_thread(
                        page.get_pixmap,
                        matrix=pymupdf.Matrix(1.5, 1.5),
                        alpha=False,
                    )

                    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

                    result = await self._analyze_image(
                        image,
                        prompt,
                        cancellation,
                    )

                    if result:
                        results.append(
                            f"Page {page_index + 1}: {result}"
                        )

                return {
                    "text": "\n\n".join(results),
                    "pages": page_count,
                    "runtime": "smoldocling",
                    "document_path": str(path),
                }

            finally:
                pdf.close()

        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            from PIL import Image

            image = await asyncio.to_thread(Image.open, str(path))
            image = image.convert("RGB")

            result = await self._analyze_image(
                image,
                prompt,
                cancellation,
            )

            return {
                "text": result,
                "pages": 1,
                "runtime": "smoldocling",
                "document_path": str(path),
            }

        raise ValueError(
            f"Unsupported document type: {suffix}. "
            "Supported types: PDF, PNG, JPG, JPEG, WEBP."
        )

    async def _analyze_image(
        self,
        image: Any,
        prompt: str,
        cancellation: threading.Event | None = None,
    ) -> str:
        if cancellation and cancellation.is_set():
            raise asyncio.CancelledError()

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = await asyncio.to_thread(
            self.processor,
            text=chat_prompt,
            images=[image],
            return_tensors="pt",
        )

        inputs = {
            key: value.to("cpu") if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        if cancellation and cancellation.is_set():
            raise asyncio.CancelledError()

        with torch.no_grad():
            generated_ids = await asyncio.to_thread(
                self.model.generate,
                **inputs,
                max_new_tokens=48,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return self._clean_output(generated_text)

    @staticmethod
    def _clean_output(text: str) -> str:
        text = str(text).strip()

        if "Assistant:" in text:
            text = text.split("Assistant:", 1)[1].strip()

        text = re.sub(r"<ocr>|</ocr>", " ", text)
        text = re.sub(r"\\s+", " ", text).strip()

        sentences = re.split(r"(?<=[.!?])\\s+", text)

        cleaned = []
        seen = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            normalized = re.sub(r"[^a-z0-9 ]+", " ", sentence.lower())
            normalized = re.sub(r"\\s+", " ", normalized).strip()

            if not normalized:
                continue

            # Remove exact duplicates.
            if normalized in seen:
                continue

            # Remove highly repetitive sentences that mostly repeat
            # information already present in an earlier sentence.
            current_words = set(normalized.split())

            is_redundant = False
            for previous in cleaned:
                previous_normalized = re.sub(
                    r"[^a-z0-9 ]+", " ", previous.lower()
                )
                previous_normalized = re.sub(
                    r"\\s+", " ", previous_normalized
                ).strip()

                previous_words = set(previous_normalized.split())

                if not current_words or not previous_words:
                    continue

                overlap = len(current_words & previous_words) / min(
                    len(current_words),
                    len(previous_words),
                )

                if overlap >= 0.85:
                    is_redundant = True
                    break

            if not is_redundant:
                cleaned.append(sentence)
                seen.add(normalized)

        return " ".join(cleaned).strip()

    async def unload(self) -> None:
        self.model = None
        self.processor = None

        try:
            import gc
            gc.collect()
        except Exception:
            pass

        await super().unload()


class DocumentCreationModelAdapter(ModelAdapter):
    """SmolLM2-360M adapter for generating structured document content."""

    def __init__(self, profile: ModelProfile):
        super().__init__(profile)
        self.tokenizer = None
        self.model = None

    async def load(self) -> None:
        if self.state == ModelLifecycleState.LOADED.value:
            return

        self.state = ModelLifecycleState.LOADING.value

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_path = self.profile.model_path
            if not model_path:
                raise RuntimeError("SmolLM2 model path is not configured")

            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                model_path,
            )

            self.model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained,
                model_path,
                dtype=torch.float32,
                device_map="cpu",
            )

            self.state = ModelLifecycleState.LOADED.value

        except Exception:
            self.state = ModelLifecycleState.LOAD_FAILED.value
            self.tokenizer = None
            self.model = None
            raise

    async def unload(self) -> None:
        if self.model is None and self.tokenizer is None:
            self.state = ModelLifecycleState.EVICTED.value
            return

        self.state = ModelLifecycleState.UNLOADING.value

        self.model = None
        self.tokenizer = None

        import gc
        gc.collect()

        self.state = ModelLifecycleState.EVICTED.value

    async def execute(
        self,
        input_data: Any,
        cancellation: threading.Event,
    ) -> Any:
        if cancellation.is_set():
            raise asyncio.CancelledError()

        if self.model is None or self.tokenizer is None:
            await self.load()

        request = (
            input_data
            if isinstance(input_data, dict)
            else {"prompt": str(input_data)}
        )

        prompt = str(request.get("prompt", "")).strip()

        if not prompt:
            raise ValueError("Document creation requires a prompt")

        requested_tokens = int(request.get("max_new_tokens", 256))

        # Short document requests should finish quickly on CPU.
        # Longer documents can explicitly request more tokens, but
        # never exceed the adapter safety ceiling.
        short_request = bool(
            re.search(
                r"\b(short|brief|small|concise|3[- ]line|few lines)\b",
                prompt.lower(),
            )
        )

        if short_request:
            max_new_tokens = min(requested_tokens, 256)
        else:
            max_new_tokens = min(requested_tokens, 256)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional document creation assistant. "
                    "Generate clean, structured, accurate document content. "
                    "Return only the document content."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        def generate():
            import torch

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            new_tokens = output[0][inputs["input_ids"].shape[1]:]

            return self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

        self.state = ModelLifecycleState.EXECUTING.value

        try:
            text = await asyncio.to_thread(generate)

            if cancellation.is_set():
                raise asyncio.CancelledError()

            return {
                "text": text,
                "model": self.profile.model_id,
                "runtime": "transformers",
                "format": request.get("format", "txt"),
            }
        finally:
            self.state = ModelLifecycleState.LOADED.value

    async def render(self, spec: dict[str, Any], output_format: str) -> Any:
        """Generate document content and render it into the requested file format."""
        from pathlib import Path
        import uuid

        result = await self.execute(
            {
                **spec,
                "format": output_format,
            },
            threading.Event(),
        )

        text = str(result.get("text", "")).strip()
        if not text:
            raise ValueError("Document creation produced empty content")

        format_name = str(output_format or "txt").lower().lstrip(".")
        output_dir = Path("data/documents")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"document_{uuid.uuid4().hex}.{format_name}"

        if format_name == "docx":
            from .renderers import DOCXRenderer
            renderer = DOCXRenderer()
        elif format_name == "pdf":
            from .renderers import PDFRenderer
            renderer = PDFRenderer()
        elif format_name == "pptx":
            from .renderers import PPTXRenderer
            renderer = PPTXRenderer()
        elif format_name == "xlsx":
            from .renderers import XLSXRenderer
            renderer = XLSXRenderer()
        else:
            renderer = None

        if renderer is None:
            if format_name in {"txt", "text"}:
                output_path.write_text(text, encoding="utf-8")
            elif format_name in {"md", "markdown"}:
                output_path.write_text(text, encoding="utf-8")
            else:
                raise ValueError(f"Unsupported document format: {format_name}")
        else:
            renderer.render(
                {
                    "text": text,
                    "content": text,
                },
                str(output_path),
            )

        result["format"] = format_name
        result["file_path"] = str(output_path)
        result["file_name"] = output_path.name

        return result
