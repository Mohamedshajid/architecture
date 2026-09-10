from __future__ import annotations

from typing import Any

from .models import Evidence, TaskResult


class ResultAggregator:
    def combine(self, results: list[TaskResult], tasks: list[Any] | None = None) -> dict[str, Any]:
        task_records = []
        for task in tasks or results:
            if isinstance(task, TaskResult):
                task_records.append({"task_id": task.task_id, "capability": task.capability.value, "model_id": task.model_id, "status": "completed" if task.error is None else "failed", "error": task.error})
            else:
                task_records.append({"task_id": task.task_id, "capability": task.capability.value, "model_id": task.model_id, "status": task.status.value, "error": task.error})
        return {
            "outputs": [r.output for r in results],
            "provenance": [{"task_id": r.task_id, "request_id": r.request_id, "model_id": r.model_id, "capability": r.capability.value, "latency_ms": round(r.latency_ms, 2), **r.provenance} for r in results],
            "model_count": len(results),
            "tasks": task_records,
            "failed_tasks": [record for record in task_records if record["status"] == "failed"],
        }


class EvidenceStore:
    def __init__(self):
        self._items: dict[str, list[Evidence]] = {}

    def add(self, request_id: str, evidence: list[Evidence]) -> None:
        self._items.setdefault(request_id, []).extend(evidence)

    def get(self, request_id: str) -> list[Evidence]:
        return list(self._items.get(request_id, []))
