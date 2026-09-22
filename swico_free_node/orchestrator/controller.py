from __future__ import annotations

import re

from .models import Capability


def detect_capabilities(prompt: str) -> list[Capability]:
    lower = prompt.lower()

    capabilities: list[Capability] = []

    # Explicit document-generation requests take precedence over
    # generic "create/write + programming language" coding matches.
    document_creation_pattern = (
        r"\b(create|make|write|export|generate)\b.*"
        r"\b(document|pdf|docx|presentation|pptx|spreadsheet|report)\b"
    )

    project_document_pattern = (
        r"\b(create|make|write|export|generate)\b.*"
        r"\b(project summary|professional summary|project report|"
        r"application summary|technical summary)\b"
    )

    is_document_creation = bool(
        re.search(document_creation_pattern, lower)
        or re.search(project_document_pattern, lower)
    )

    if is_document_creation:
        capabilities.append(Capability.DOCUMENT_CREATION)

    # Coding should require an actual programming/coding intent.
    coding_patterns = (
        r"\b(write|create|implement|debug|fix|modify|refactor|optimize|generate)\b.*"
        r"\b(code|function|class|script|program|algorithm)\b",
        r"\b(write|create|implement|debug|fix|modify|refactor|optimize|generate)\b.*"
        r"\b(python|javascript|typescript|java|c\+\+|c#|sql)\b",
        r"\b(python|javascript|typescript|java|c\+\+|c#|sql)\b.*"
        r"\b(function|class|code|script|program|implement|debug|write)\b",
        r"\b(code|coding|debug|function|class|script|program|algorithm|syntax|compile|exception|error)\b",
    )

    if (
        not is_document_creation
        and any(re.search(pattern, lower) for pattern in coding_patterns)
    ):
        capabilities.append(Capability.CODING)

    rules = [
        (
            Capability.IMAGE_GENERATION,
            r"\b(image|picture|illustration|logo|draw|render|visualize|"
            r"generate an image|create an image|make an image|generate a picture|"
            r"create a picture)\b",
        ),
        (Capability.STT, r"\b(transcribe|speech to text|audio transcription)\b"),
        (Capability.TTS, r"\b(text to speech|read aloud|voice synthesis)\b"),
        (Capability.VIDEO_GENERATION, r"\b(video|animation|clip)\b"),
        (
            Capability.DOCUMENT_ANALYSIS,
            r"\b(analy[sz]e|summari[sz]e|extract|review).*(document|pdf|file|table)\b",
        ),
        # Document creation is handled before coding detection above.
    ]

    for capability, pattern in rules:
        if re.search(pattern, lower):
            capabilities.append(capability)

    # Natural image prompts may describe a visual scene without saying
    # "generate an image". Detect common visual-scene language.
    natural_image_pattern = (
        r"\b(sitting|standing|walking|running|portrait|landscape|sunset|"
        r"sunrise|garden|beach|mountain|forest|sky|background|scene|"
        r"cinematic|realistic|cute|beautiful|colorful)\b"
    )

    if not capabilities and re.search(natural_image_pattern, lower):
        capabilities.append(Capability.IMAGE_GENERATION)

    # General explanatory/conversational requests go to Chat.
    chat_pattern = (
        r"\\b(what is|what are|explain|how does|how do|why is|why does|"
        r"hello|hi|help me understand|tell me about|compare)\\b"
    )

    if re.search(chat_pattern, lower):
        capabilities.append(Capability.CHAT)

    # If nothing specialized was detected, default to Chat.
    if not capabilities:
        capabilities.append(Capability.CHAT)

    return capabilities



class MainController:
    """Deterministic request understanding kept independent from model inference."""
    def understand(self, prompt: str, capability: Capability | str | None = None, *, priority: int | None = None, deadline_seconds: float | None = None, verify: bool = False) -> dict:
        lower_prompt = prompt.lower()
        detailed_markers = (
            "detailed", "deep", "step by step", "architecture",
            "compare", "comparison", "explain how", "implementation",
            "design", "advantages", "disadvantages", "example architecture"
        )
        complexity = (
            "high"
            if (
                len(prompt) > 300
                or len(re.findall(r"\band\b|\bthen\b", lower_prompt)) >= 2
                or any(marker in lower_prompt for marker in detailed_markers)
            )
            else "low"
        )
        if capability is not None:
            selected = capability if isinstance(capability, Capability) else Capability(capability)
            capabilities = [selected]
        else:
            capabilities = detect_capabilities(prompt)
        return {"intent": "multimodal_task" if complexity == "high" else "single_task", "complexity": complexity, "capabilities": [c.value for c in capabilities], "priority": 50 if priority is None else priority, "deadline_seconds": deadline_seconds, "verification_requested": verify}
