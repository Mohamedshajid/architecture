from pathlib import Path
import threading
import torch
import soundfile as sf
from kokoro import KModel, KPipeline


class KokoroRuntime:
    def __init__(self, model_path: Path | None = None, threads: int = 2):
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.threads = threads
        self.model = None
        self.pipeline = None
        self.model_loaded = False

    def load(self) -> None:
        torch.set_num_threads(self.threads)

        if self.model_path and self.model_path.exists():
            self.model = KModel(model=str(self.model_path))
        else:
            self.model = KModel()

        self.pipeline = KPipeline(
            "a",
            model=self.model,
            device="cpu",
        )
        self.model_loaded = True

    def synthesize(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        cancellation: threading.Event | None = None,
    ) -> dict:
        if not self.model_loaded or self.pipeline is None:
            raise RuntimeError("Kokoro runtime is not loaded")

        text = str(text).strip()
        if not text:
            raise ValueError("TTS text cannot be empty")

        if cancellation and cancellation.is_set():
            raise RuntimeError("TTS cancelled")

        outputs = []

        for result in self.pipeline(
            text,
            voice=voice,
            speed=float(speed),
        ):
            if cancellation and cancellation.is_set():
                raise RuntimeError("TTS cancelled")

            if result.output is not None:
                outputs.append(result.output.audio)

        if not outputs:
            raise RuntimeError("Kokoro produced no audio")

        audio = torch.cat(outputs).numpy()

        output_dir = Path("data/tts")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"kokoro_{__import__('uuid').uuid4().hex}.wav"
        sf.write(str(output_path), audio, 24000)

        return {
            "audio_path": str(output_path.resolve()),
            "sample_rate": 24000,
            "audio_samples": int(len(audio)),
            "duration_seconds": round(len(audio) / 24000, 3),
            "voice": voice,
            "speed": float(speed),
            "runtime": "kokoro",
        }

    def unload(self) -> None:
        self.pipeline = None
        self.model = None
        self.model_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
