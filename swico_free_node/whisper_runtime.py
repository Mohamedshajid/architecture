from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import time


class WhisperRuntime:
    """CPU-local Whisper runtime for an exact Transformers checkpoint."""

    def __init__(self, path: Path, *, threads: int = 2) -> None:
        if not path.is_dir():
            raise RuntimeError("The configured Whisper artifact directory does not exist")
        if not (path / "model.safetensors").is_file():
            raise RuntimeError("The configured Whisper artifact is missing model.safetensors")
        try:
            import torch
            from transformers import AutoProcessor, WhisperForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for Whisper") from exc
        torch.set_num_threads(threads)
        self.model_path = path
        self.threads = threads
        self.sample_rate = 16_000
        self._torch = torch
        self._AutoProcessor = AutoProcessor
        self._WhisperForConditionalGeneration = WhisperForConditionalGeneration
        self._processor = None
        self._model = None
        self.last_load_seconds: float | None = None
        self.last_unload_seconds: float | None = None
        self.last_transcribe_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.loaded:
            return
        started = time.perf_counter()
        self._processor = self._AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self._model = self._WhisperForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=self._torch.float32,
        ).to("cpu").eval()
        self.last_load_seconds = time.perf_counter() - started

    def unload(self) -> None:
        if not self.loaded:
            return
        started = time.perf_counter()
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        import gc
        gc.collect()
        self.last_unload_seconds = time.perf_counter() - started

    def transcribe(
        self,
        audio_path: Path,
        cancellation: Event | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Whisper runtime is not loaded")
        started = time.perf_counter()
        if cancellation is not None and cancellation.is_set():
            return {"cancelled": True, "transcript": "", "runtime": "whisper"}
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for Whisper audio input") from exc
        if not audio_path.is_file():
            raise RuntimeError("The configured audio file does not exist")
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.sample_rate:
            source_x = np.linspace(0, 1, num=len(audio), endpoint=False)
            target_length = max(1, round(len(audio) * self.sample_rate / sample_rate))
            target_x = np.linspace(0, 1, num=target_length, endpoint=False)
            audio = np.interp(target_x, source_x, audio).astype(np.float32)
            sample_rate = self.sample_rate
        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        if cancellation is not None and cancellation.is_set():
            return {"cancelled": True, "transcript": "", "runtime": "whisper"}
        with self._torch.no_grad():
            generated = self._model.generate(
                inputs.input_features,
                max_new_tokens=256,
                language=language,
                task="transcribe",
            )
        if cancellation is not None and cancellation.is_set():
            return {"cancelled": True, "transcript": "", "runtime": "whisper"}
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        result = {
            "transcript": text,
            "language": language,
            "segments": [],
            "timestamps": None,
            "confidence": None,
            "sample_rate": sample_rate,
            "audio_samples": int(len(audio)),
            "audio_duration_seconds": len(audio) / sample_rate if sample_rate else 0.0,
            "runtime": "whisper",
        }
        self.last_transcribe_seconds = time.perf_counter() - started
        return result
