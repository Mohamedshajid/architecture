from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DocumentRenderer(ABC):
    format: str

    @abstractmethod
    def render(self, specification: dict[str, Any], output_path: str) -> str:
        """Render validated structured content deterministically."""
        raise NotImplementedError


class PDFRenderer(DocumentRenderer):
    format = "pdf"


class DOCXRenderer(DocumentRenderer):
    format = "docx"


class PPTXRenderer(DocumentRenderer):
    format = "pptx"


class XLSXRenderer(DocumentRenderer):
    format = "xlsx"
