from __future__ import annotations

import re

from .models import Evidence, TaskResult, VerificationStatus


def _claims(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?\n]+", text) if part.strip()]


class TriRAG:
    """Deterministic retrieve/validate/rank/resolve/synthesize/verify pipeline."""

    def __init__(self, retriever=None):
        self.retriever = retriever

    def retrieve_from_index(self, query: str, top_k: int = 5) -> list[Evidence]:
        if self.retriever is None:
            return []
        return [Evidence(item.chunk.source, item.chunk.text, item.similarity, dict(item.chunk.metadata), item.chunk.document_id, item.chunk.chunk_id, item.similarity) for item in self.retriever.search(query, top_k)]

    def retrieve(self, query: str, evidence: list[Evidence]) -> list[Evidence]:
        words = {word.lower() for word in query.split() if len(word) > 3}
        return sorted(self.validate(evidence), key=lambda e: (len(words & set(e.content.lower().split())) / max(1, len(words)), e.score), reverse=True)

    def validate(self, evidence: list[Evidence]) -> list[Evidence]:
        valid: dict[tuple[str, str | None, str | None, str], Evidence] = {}
        for item in evidence:
            if not (item.content.strip() and item.source_id.strip() and -1 <= item.score <= 1 and (item.document_id is None or item.document_id.strip()) and (item.chunk_id is None or item.chunk_id.strip())):
                continue
            key = (item.source_id, item.document_id, item.chunk_id, item.content.strip())
            if key not in valid or item.score > valid[key].score:
                valid[key] = item
        return list(valid.values())

    def rank(self, query: str, evidence: list[Evidence]) -> list[Evidence]:
        return self.retrieve(query, evidence)

    def resolve_conflicts(self, evidence: list[Evidence]) -> list[dict]:
        variants: dict[tuple[str, ...], list[Evidence]] = {}
        for item in evidence:
            numbers = tuple(re.findall(r"\b\d+(?:\.\d+)?\b", item.content))
            if numbers:
                variants.setdefault(numbers, []).append(item)
        if len(variants) <= 1:
            return []
        return [{"type": "CONFLICT_DETECTED", "variants": [{"numbers": list(key), "sources": [item.source_id for item in items]} for key, items in variants.items()]}]

    def resolve(self, results: list[TaskResult], evidence: list[Evidence]) -> dict:
        ranked = self.rank(" ".join(str(r.output) for r in results), evidence)
        return {"results": results, "evidence": ranked, "conflicts": self.resolve_conflicts(ranked)}

    def synthesize_package(self, query: str, evidence: list[Evidence]) -> dict:
        ranked = self.rank(query, evidence)
        return {"query": query, "evidence": [{"document_id": e.document_id, "chunk_id": e.chunk_id, "text": e.content, "similarity": e.similarity if e.similarity is not None else e.score, "source": e.source_id} for e in ranked], "conflicts": self.resolve_conflicts(ranked), "grounding_status": "SUPPORTED" if ranked else "INSUFFICIENT_EVIDENCE"}

    def synthesize(self, results: list[TaskResult]) -> str:
        return "\n\n".join(str(r.output.get("text", r.output)) for r in results)

    def grounded_synthesis(self, query: str, outputs: list[str], evidence: list[Evidence]) -> dict:
        ranked = self.rank(query, evidence)
        verification = self.verify_claims(outputs, ranked)

        if verification["verification_status"] == VerificationStatus.VERIFIED.value and ranked:
            text = ranked[0].content
            grounding_status = "SUPPORTED"
        else:
            text = "\n\n".join(outputs)
            grounding_status = verification["grounding_status"]

        return {
            "text": text,
            "grounding_status": grounding_status,
            "verification_status": verification["verification_status"],
        }

    def verify_claims(self, outputs: list[str], evidence: list[Evidence]) -> dict:
        valid = self.rank(" ".join(outputs), evidence)
        claims = [claim for output in outputs for claim in _claims(output)]
        support, contradictions, uncertain = [], [], []
        for claim in claims:
            numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
            matches = [item for item in valid if set(claim.lower().split()) & set(item.content.lower().split())]
            if not matches:
                uncertain.append(claim)
            elif numbers and any(numbers.isdisjoint(set(re.findall(r"\b\d+(?:\.\d+)?\b", item.content))) for item in matches):
                contradictions.append({"claim": claim, "evidence": [item.content for item in matches]})
            else:
                support.append({"claim": claim, "evidence": [item.content for item in matches]})
        if contradictions:
            status = VerificationStatus.PARTIALLY_VERIFIED if support else VerificationStatus.UNCERTAIN
        elif support and not uncertain:
            status = VerificationStatus.VERIFIED
        elif support:
            status = VerificationStatus.PARTIALLY_VERIFIED
        else:
            status = VerificationStatus.UNCERTAIN
        return {"claims": claims, "support": support, "contradictions": contradictions, "uncertain": uncertain, "agreement": len(outputs) < 2 or len(set(outputs)) == 1, "conflicts": self.resolve_conflicts(valid), "verification_status": status.value, "grounding_status": "SUPPORTED" if support else "INSUFFICIENT_EVIDENCE", "evidence": valid}

    def verify(self, query: str, results: list[TaskResult], evidence: list[Evidence]) -> tuple[VerificationStatus, list[Evidence]]:
        ranked = self.rank(query, evidence)
        if not results:
            return VerificationStatus.FAILED, ranked
        return VerificationStatus(self.verify_claims([self.synthesize(results)], ranked)["verification_status"]), ranked
