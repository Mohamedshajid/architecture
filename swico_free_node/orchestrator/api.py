from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .models import Capability, Evidence
from .orchestrator import Orchestrator


class OrchestrateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=24_000)
    capability: Capability | None = None
    priority: int | None = Field(default=None, ge=0, le=100)
    request_id: str | None = None
    verify: bool = False
    deadline_seconds: float = Field(default=45, ge=1, le=300)
    evidence: list[dict[str, Any]] = Field(default_factory=list, max_length=64)
    background: bool = False


def create_router(orchestrator: Orchestrator) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.post("/orchestrate")
    async def orchestrate(payload: OrchestrateRequest) -> dict[str, Any]:
        evidence = [Evidence(source_id=str(item.get("source_id", "unknown")), content=str(item.get("content", "")), score=float(item.get("score", 0))) for item in payload.evidence]
        if payload.background:
            request_id = payload.request_id or __import__('uuid').uuid4().__str__()
            task = __import__('asyncio').create_task(orchestrator.run(payload.prompt, request_id=request_id, verify=payload.verify, evidence=evidence, deadline_seconds=payload.deadline_seconds, capability=payload.capability, priority=payload.priority))
            orchestrator.jobs.setdefault(request_id, {"graph": None, "started_at": __import__('time').time(), "cancelled": False, "future": task})
            return {"request_id": request_id, "status": "accepted"}
        try:
            return await orchestrator.run(payload.prompt, request_id=payload.request_id, verify=payload.verify, evidence=evidence, deadline_seconds=payload.deadline_seconds, capability=payload.capability, priority=payload.priority)
        except (TimeoutError, RuntimeError) as exc:
            raise HTTPException(503, {"code": "orchestration_unavailable", "message": str(exc)}) from exc

    @router.get("/models")
    async def models() -> dict[str, Any]:
        return {"models": orchestrator.registry.as_dicts()}

    @router.get("/models/health")
    async def model_health() -> dict[str, Any]:
        return {"models": orchestrator.registry.health(orchestrator.adapters, orchestrator.scheduler.status())}

    @router.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {"capabilities": orchestrator.capabilities.as_dict()}

    @router.get("/tasks/{request_id}")
    async def task_status(request_id: str) -> dict[str, Any]:
        job = orchestrator.jobs.get(request_id)
        if job is None:
            raise HTTPException(404, {"code": "task_not_found"})
        return {"request_id": request_id, "cancelled": job["cancelled"], "tasks": [{"task_id": t.task_id, "capability": t.capability.value, "status": t.status.value, "model_id": t.model_id, "error": t.error} for t in job["graph"].tasks.values()]}

    @router.post("/tasks/{request_id}/cancel")
    async def cancel(request_id: str) -> dict[str, Any]:
        if not orchestrator.cancel(request_id):
            raise HTTPException(404, {"code": "task_not_found"})
        return {"request_id": request_id, "cancelled": True}

    return router
