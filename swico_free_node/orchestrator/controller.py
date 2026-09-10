from __future__ import annotations

import re

from .models import Capability


def detect_capabilities(prompt: str) -> list[Capability]:
    lower = prompt.lower()
    rules = [
        (Capability.CODING, r"\b(code|coding|debug|python|javascript|typescript|function|implement|bug|program|script)\b"),
        (Capability.IMAGE_GENERATION, r"\b(image|picture|illustration|logo|draw|generate an image)\b"),
        (Capability.STT, r"\b(transcribe|speech to text|audio)\b"),
        (Capability.TTS, r"\b(text to speech|read aloud|voice)\b"),
        (Capability.VIDEO_GENERATION, r"\b(video|animation|clip)\b"),
        (Capability.DOCUMENT_ANALYSIS, r"\b(analy[sz]e|summari[sz]e|extract|review).*(document|pdf|file|table)\b"),
        (Capability.DOCUMENT_CREATION, r"\b(create|make|write|export).*(document|pdf|docx|presentation|pptx|spreadsheet)\b"),
        (Capability.CHAT, r"\b(what is|what are|explain|how does|how do|why is|why does|hello|hi|help me understand|tell me about|compare)\b"),
    ]
    return [capability for capability, pattern in rules if re.search(pattern, lower)]


class MainController:
    """Deterministic request understanding kept independent from model inference."""
    def understand(self, prompt: str, capability: Capability | str | None = None, *, priority: int | None = None, deadline_seconds: float | None = None, verify: bool = False) -> dict:
        complexity = "high" if len(prompt) > 800 or len(re.findall(r"\band\b|\bthen\b", prompt.lower())) >= 2 else "low"
        if capability is not None:
            selected = capability if isinstance(capability, Capability) else Capability(capability)
            capabilities = [selected]
        else:
            capabilities = detect_capabilities(prompt)
        return {"intent": "multimodal_task" if complexity == "high" else "single_task", "complexity": complexity, "capabilities": [c.value for c in capabilities], "priority": 50 if priority is None else priority, "deadline_seconds": deadline_seconds, "verification_requested": verify}
