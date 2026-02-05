from __future__ import annotations

from typing import Optional

from core.constraints import Constraints


def _constraints_text(constraints: Optional[Constraints]) -> str:
    if constraints is None:
        return ""
    pieces = []
    if constraints.domain_tags:
        pieces.append(f"Domain tags: {', '.join(constraints.domain_tags)}")
    if constraints.role_tags:
        pieces.append(f"Role tags: {', '.join(constraints.role_tags)}")
    if constraints.tool_tags:
        pieces.append(f"Tool tags: {', '.join(constraints.tool_tags)}")
    if constraints.modalities:
        pieces.append(f"Modalities: {', '.join(constraints.modalities)}")
    if constraints.output_formats:
        pieces.append(f"Output formats: {', '.join(constraints.output_formats)}")
    if constraints.permissions:
        pieces.append(f"Permissions: {', '.join(constraints.permissions)}")
    if constraints.cost_tier:
        pieces.append(f"Cost tier: {constraints.cost_tier}")
    if constraints.latency_tier:
        pieces.append(f"Latency tier: {constraints.latency_tier}")
    if constraints.reliability_prior is not None:
        pieces.append(f"Reliability prior >= {constraints.reliability_prior}")
    return " | ".join(pieces)


def build_role_query(task_text: str, role: str, constraints: Optional[Constraints] = None) -> str:
    role_key = role.strip().lower()
    if constraints is not None and not isinstance(constraints, Constraints):
        constraints = Constraints(**constraints)
    constraints_text = _constraints_text(constraints)
    task_text = task_text.strip()

    if role_key in {"planner", "code-planner"}:
        template = (
            "You are a code planner. Break down the task into milestones, dependencies, and checkpoints. "
            "Focus on sequencing and feasibility. Task: {task}. {constraints}"
        )
    elif role_key == "researcher":
        template = (
            "You are a researcher. Gather evidence, compare sources, and note open questions. "
            "Focus on coverage and uncertainty. Task: {task}. {constraints}"
        )
    elif role_key in {"builder", "code-generation"}:
        template = (
            "You are a code generator. Generate the solution implementation in code. "
            "Focus on correct logic, clear structure, and runnable output. Task: {task}. {constraints}"
        )
    elif role_key in {"tester", "checker", "code-testing"}:
        template = (
            "You are a code tester. Validate outputs, test assumptions, and look for edge cases or failures. "
            "Focus on correctness, coverage, and risks. Task: {task}. {constraints}"
        )
    elif role_key in {"refractor", "refactor", "code-refactoring"}:
        template = (
            "You are a code refactoring expert. Improve and refactor the solution. "
            "Fix errors, polish style, and restructure code for clarity and maintainability. "
            "Focus on minimal, correct changes. Task: {task}. {constraints}"
        )
    elif role_key == "manager":
        template = (
            "You are a manager. Coordinate roles, delegate tasks, and synthesize progress. "
            "Focus on sequencing and decision-making. Task: {task}. {constraints}"
        )
    else:
        template = "Role: {role}. Task: {task}. {constraints}"

    return template.format(task=task_text, role=role, constraints=constraints_text).strip()
