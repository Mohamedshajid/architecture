from __future__ import annotations

from .controller import detect_capabilities
from .models import Capability, Task, TaskStatus


class TaskGraph:
    def __init__(self, tasks: list[Task]):
        task_list = list(tasks)
        task_ids = [task.task_id for task in task_list]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task graph contains duplicate task IDs")
        self.tasks = {task.task_id: task for task in task_list}
        self.validate()

    def validate(self) -> None:
        for task in self.tasks.values():
            if task.task_id in task.dependencies:
                raise ValueError(f"task {task.task_id} depends on itself")
            missing = [dependency for dependency in task.dependencies if dependency not in self.tasks]
            if missing:
                raise ValueError(f"task {task.task_id} has unknown dependencies: {missing}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task graph contains a dependency cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in self.tasks[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in self.tasks:
            visit(task_id)

    def ready(self) -> list[Task]:
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and task.dependencies and not all(self.tasks[d].status == TaskStatus.COMPLETED for d in task.dependencies):
                task.status = TaskStatus.WAITING_FOR_DEPENDENCY
            elif task.status == TaskStatus.WAITING_FOR_DEPENDENCY and all(self.tasks[d].status == TaskStatus.COMPLETED for d in task.dependencies):
                task.status = TaskStatus.READY
        return sorted((t for t in self.tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.READY)), key=lambda t: (-t.priority, t.deadline))

    def cancel_dependents(self, task_id: str) -> None:
        for task in self.tasks.values():
            if task_id in task.dependencies and task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.status = TaskStatus.CANCELLED
                self.cancel_dependents(task.task_id)


class TaskPlanner:
    def plan(self, request_id: str, prompt: str, verify: bool = False, deadline_seconds: float = 45, capability: Capability | str | None = None, priority: int | None = None) -> TaskGraph:
        if capability is not None:
            selected = capability if isinstance(capability, Capability) else Capability(capability)
            capabilities: list[Capability] = [selected]
        else:
            capabilities = detect_capabilities(prompt)
        if not capabilities:
            capabilities = [Capability.CHAT]
        elif Capability.CODING not in capabilities and not any(c in capabilities for c in (Capability.IMAGE_GENERATION, Capability.STT, Capability.TTS, Capability.VIDEO_GENERATION, Capability.DOCUMENT_CREATION, Capability.DOCUMENT_ANALYSIS)):
            capabilities.insert(0, Capability.CHAT)
        if Capability.DOCUMENT_ANALYSIS in capabilities and Capability.CHAT not in capabilities:
            capabilities.insert(capabilities.index(Capability.DOCUMENT_ANALYSIS) + 1, Capability.CHAT)
        if Capability.DOCUMENT_ANALYSIS in capabilities and Capability.DOCUMENT_CREATION in capabilities:
            capabilities.remove(Capability.DOCUMENT_ANALYSIS)
            capabilities.remove(Capability.DOCUMENT_CREATION)
            capabilities.extend((Capability.DOCUMENT_ANALYSIS, Capability.DOCUMENT_CREATION))
        tasks: list[Task] = []
        previous_analysis: str | None = None
        for capability in capabilities:
            dependent = capability in (Capability.CHAT, Capability.DOCUMENT_CREATION) and previous_analysis is not None
            task = Task(request_id=request_id, task_type=capability.value, capability=capability,
                        input=prompt, dependencies=[previous_analysis] if dependent else [],
                        priority=priority if priority is not None else (80 if capability == Capability.VIDEO_GENERATION else 60),
                        verification_required=verify, deadline=__import__('time').time() + deadline_seconds)
            tasks.append(task)
            previous_analysis = task.task_id if capability == Capability.DOCUMENT_ANALYSIS else previous_analysis
        return TaskGraph(tasks)
