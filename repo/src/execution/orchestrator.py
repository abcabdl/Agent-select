from __future__ import annotations

import json
import os
import re
import sys
import math
import hashlib
import sqlite3
import subprocess
import tempfile
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.contracts import validate_output
from core.failure_router import route_failure
from core.logging import write_event
from core.query_builder import build_role_query
from core.topology import TopologyConfig, TopologyType
from execution.probe_commit import probe_commit
from execution.tool_executor import ToolExecutor
from generation.llm_client import LLMClient
from retrieval.search_service import get_candidates
from routing.bandit_store import BanditStore
from routing.reranker_model import TfidfLinearReranker


class MockLLMClient:
    """Mock LLM client that returns structured outputs per role."""

    def generate(self, role: str, task_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        agent = context.get("agent", {})
        agent_name = agent.get("name") or agent.get("id") or "agent"
        agent_domain = ",".join(agent.get("domain_tags") or [])
        agent_tools = ",".join(agent.get("tool_tags") or [])
        role_key = _normalize_role_name(role)
        if role_key == "planner":
            return {
                "steps": [
                    f"{agent_name} clarify scope for: {task_text}",
                    "Break down milestones",
                    "Sequence tasks with dependencies",
                ],
                "acceptance_criteria": [
                    "Plan covers core tasks",
                    f"Dependencies explicit for {agent_domain or 'general'}",
                ],
            }
        if role_key == "researcher":
            return {
                "search_queries": [f"{task_text} overview", f"{task_text} {agent_domain}"],
                "sources": ["internal_knowledge_base", "public_docs"],
                "evidence_points": [f"Findings summarized with {agent_tools or 'tools'}"],
            }
        if role_key == "builder":
            return {
                "runnable_plan": ["Implement core logic", "Wire dependencies", "Run tests"],
                "code_or_commands": f"run build; use {agent_tools or 'default tools'}",
                "self_test": ["unit tests pass", "smoke test pass"],
            }
        if role_key == "refactor":
            return {
                "runnable_plan": ["Refactor implementation", "Preserve behavior", "Re-run regression tests"],
                "code_or_commands": f"run refactor; use {agent_tools or 'default tools'}",
                "self_test": ["regression tests pass", "smoke test pass"],
            }
        if role_key == "checker":
            return {
                "test_cases": ["happy path", "edge case"],
                "verdicts": ["pass", "needs review"],
                "failure_localization": f"review edge case for {agent_name}",
            }
        if role_key == "manager":
            roles = context.get("available_roles") or []
            next_role = roles[0] if roles else "builder"
            return {
                "status": "delegate",
                "next_role": next_role,
                "instruction": f"Handle: {task_text}",
                "summary": f"Delegated to {next_role}",
                "notes": ["Use upstream context for continuity."],
            }
        return {}

    def rewrite_format(self, role: str, output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return self.generate(role, "format_fix", context)

    def clarify(self, role: str, task_text: str) -> Dict[str, Any]:
        return {"clarifier": f"Provide missing details for {role} about {task_text}"}

    def plan_tool(
        self,
        role: str,
        task_text: str,
        context: Dict[str, Any],
        tools: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if context.get("tool_only") and tools:
            return {
                "action": "tool",
                "tool_id": tools[0].get("id"),
                "tool_input": {},
                "reason": "tool_only_mock",
            }
        return {"action": "final", "reason": "mock"}


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_blob(text: str) -> Optional[str]:
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _safe_load_json(text: str, fallback: Any) -> Any:
    blob = _extract_json_blob(text) or text
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return fallback


class RealLLMClient:
    """LLM-backed client that returns structured outputs per role."""

    _ROLE_SCHEMAS = {
        "planner": {
            "fields": {
                "steps": "list[str]",
                "acceptance_criteria": "list[str]",
                "ready_to_handoff": "bool (optional, true when planning is sufficient)",
                "next_role": "str (optional, suggested next role when handing off)",
            },
            "guidance": (
                "Provide a concise task plan and acceptance criteria. "
                "Set ready_to_handoff=true only when the plan is sufficient for execution."
            ),
        },
        "researcher": {
            "fields": {"search_queries": "list[str]", "sources": "list[str]", "evidence_points": "list[str]"},
            "guidance": "Propose concrete search queries, sources, and evidence points.",
        },
        "builder": {
            "fields": {"runnable_plan": "list[str]", "code_or_commands": "str", "self_test": "list[str]"},
            "guidance": "Outline implementation steps and provide runnable code/commands.",
        },
        "refactor": {
            "fields": {"runnable_plan": "list[str]", "code_or_commands": "str", "self_test": "list[str]"},
            "guidance": "Refactor code with minimal behavior-preserving edits and include updated code.",
        },
        "checker": {
            "fields": {"test_cases": "list[str]", "verdicts": "list[str]", "failure_localization": "str"},
            "guidance": "Define tests, verdicts, and failure localization notes.",
        },
        "manager": {
            "fields": {
                "status": "str (delegate|finish)",
                "next_role": "str or null",
                "instruction": "str or null",
                "summary": "str or null",
                "notes": "list[str]",
            },
            "guidance": "Decide next role or finish. Provide instruction and a short summary.",
        },
    }

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def _build_messages(self, role: str, task_text: str, context: Dict[str, Any], fix: bool) -> List[Dict[str, str]]:
        role_key = _normalize_role_name(role)
        schema = self._ROLE_SCHEMAS.get(role_key) or {
            "fields": {"summary": "str", "key_points": "list[str]", "next_hint": "str"},
            "guidance": "Summarize the work and suggest the next action.",
        }

        agent = context.get("agent", {})
        constraints = context.get("constraints") or {}
        upstream = context.get("upstream") or {}
        tool_history = context.get("tool_history") or []
        available_tools = context.get("available_tools") or []
        is_decentralized = context.get("is_decentralized", False)
        available_roles = context.get("available_roles") or []

        agent_desc = _truncate_text(agent.get("description"), max_chars=600)
        agent_embed = _truncate_text(agent.get("embedding_text"), max_chars=600)
        tool_history_text = _truncate_text(
            json.dumps(tool_history, ensure_ascii=True, default=str), max_chars=1200
        )
        tools_text = _truncate_text(json.dumps(available_tools, ensure_ascii=True, default=str), max_chars=1200)
        diagnostics = context.get("diagnostics") or {}
        diagnostics_text = _truncate_text(
            json.dumps(diagnostics, ensure_ascii=True, default=str), max_chars=1200
        )

        # In decentralized topology, agents can specify next_role
        if is_decentralized and role_key != "manager":
            schema = dict(schema)  # Copy to avoid modifying original
            schema["fields"] = dict(schema["fields"])
            schema["fields"]["next_role"] = "str (optional: specify next agent role, or 'finish' to end)"
            schema["guidance"] += (
                f" In decentralized topology, you can optionally specify 'next_role' to choose which agent should act next "
                f"from available roles: {available_roles}, or set it to 'finish' to end the workflow."
            )

        schema_lines = [f"- {key}: {value}" for key, value in schema["fields"].items()]
        schema_text = "\n".join(schema_lines)

        system_msg = (
            f"You are the {role_key} role in a multi-agent system. "
            "Return ONLY a JSON object that matches the required schema exactly. "
            "Do not include markdown or extra keys."
        )
        user_msg = (
            f"Task: {task_text}\n"
            f"Agent name: {agent.get('name') or agent.get('id')}\n"
            f"Agent domain_tags: {agent.get('domain_tags')}\n"
            f"Agent role_tags: {agent.get('role_tags')}\n"
            f"Agent tool_tags: {agent.get('tool_tags')}\n"
            f"Agent description: {agent_desc}\n"
            f"Agent embedding_text: {agent_embed}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=True)}\n"
            f"Upstream outputs: {json.dumps(upstream, ensure_ascii=True)}\n"
            f"Available roles: {json.dumps(context.get('available_roles') or [], ensure_ascii=True)}\n"
            f"Available tools: {tools_text}\n"
            f"Tool history: {tool_history_text}\n"
            f"Diagnostics: {diagnostics_text}\n"
            f"Schema:\n{schema_text}\n"
            f"Guidance: {schema.get('guidance','')}\n"
        )
        if fix:
            user_msg += (
                "Fix the previous output to match the schema.\n"
                f"Previous output: {json.dumps(context.get('bad_output'), ensure_ascii=True)}\n"
            )
        return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

    def _call(self, role: str, task_text: str, context: Dict[str, Any], fix: bool) -> Dict[str, Any]:
        messages = self._build_messages(role, task_text, context, fix)
        response = self.llm.chat(
            messages,
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        data = _safe_load_json(str(response), {})
        return data if isinstance(data, dict) else {}

    def generate(self, role: str, task_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._call(role, task_text, context, fix=False)

    def rewrite_format(self, role: str, output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        fix_context = dict(context)
        fix_context["bad_output"] = output
        return self._call(role, "format_fix", fix_context, fix=True)

    def clarify(self, role: str, task_text: str) -> Dict[str, Any]:
        return {"clarifier": f"Provide missing details for {role} about {task_text}"}

    def plan_tool(
        self,
        role: str,
        task_text: str,
        context: Dict[str, Any],
        tools: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        agent = context.get("agent", {})
        constraints = context.get("constraints") or {}
        upstream = context.get("upstream") or {}
        tool_only = bool(context.get("tool_only"))
        tool_history_text = _truncate_text(
            json.dumps(tool_history, ensure_ascii=True, default=str), max_chars=1200
        )
        tools_text = _truncate_text(json.dumps(tools, ensure_ascii=True, default=str), max_chars=1200)
        diagnostics = context.get("diagnostics") or {}
        diagnostics_text = _truncate_text(
            json.dumps(diagnostics, ensure_ascii=True, default=str), max_chars=1200
        )

        system_msg = (
            "You are an intelligent tool-using controller. Analyze the task and tool history to select the BEST tool for the job. "
            "If previous attempts failed, choose a DIFFERENT tool that can fix the specific error. "
            "Return ONLY JSON with keys: action ('tool' or 'final'), tool_id, tool_input, reason."
        )
        
        # Analyze tool history for failures and suggest better alternatives
        failure_analysis = ""
        if tool_history:
            failed_tools = []
            last_error = None
            for entry in tool_history:
                tid = entry.get("tool_id", "")
                result = entry.get("result", {})
                if isinstance(result, dict):
                    if result.get("ok") is False or result.get("error"):
                        failed_tools.append(tid)
                        if result.get("error"):
                            error_obj = result["error"]
                            if isinstance(error_obj, dict):
                                last_error = f"{error_obj.get('code', '')}: {error_obj.get('message', '')}"
            
            if failed_tools:
                failure_analysis = (
                    f"\n\nIMPORTANT - Previous failures:\n"
                    f"Failed tools: {', '.join(failed_tools)}\n"
                )
                if last_error:
                    failure_analysis += f"Last error: {last_error}\n"
                failure_analysis += (
                    "You have two strategies:\n"
                    "1. If the error can be fixed with better input/refinement, you MAY retry the same tool with improved instructions\n"
                    "2. If the tool is fundamentally unsuitable, choose a DIFFERENT tool from available tools\n"
                    "Consider whether refinement or switching is more appropriate for fixing the error.\n"
                )
        
        user_msg = (
            f"Role: {role}\n"
            f"Task: {task_text}\n"
            f"Agent name: {agent.get('name') or agent.get('id')}\n"
            f"Agent domain_tags: {agent.get('domain_tags')}\n"
            f"Agent role_tags: {agent.get('role_tags')}\n"
            f"Agent tool_tags: {agent.get('tool_tags')}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=True)}\n"
            f"Upstream outputs: {json.dumps(upstream, ensure_ascii=True)}\n"
            f"Available tools: {tools_text}\n"
            f"Tool history: {tool_history_text}\n"
            f"Diagnostics: {diagnostics_text}\n"
            f"{failure_analysis}"
            "If no tool is needed, choose action='final'."
        )
        if tool_only:
            user_msg += (
                "\nTool-only mode is enabled. You MUST choose action='tool' and "
                "set tool_id to one of the available tools.\n"
            )
            tool_ids = [tool.get("id") for tool in tools if tool.get("id")]
            if tool_ids:
                user_msg += f"Allowed tool_ids: {tool_ids}\n"
        response = self.llm.chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        data = _safe_load_json(str(response), {})
        return data if isinstance(data, dict) else {"action": "final"}

def _build_candidate_texts(registry, candidates: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for candidate in candidates:
        card_id = candidate.get("card_id")
        if not card_id:
            texts.append("")
            continue
        card = registry.get(card_id)
        if card is None:
            texts.append("")
            continue
        texts.append(card.embedding_text or card.description or card.name)
    return texts


def _format_router_candidates(candidates: List[dict]) -> str:
    """Format candidates with name interpretation for better router decisions.
    
    Agent names follow the format: {OriginalName}-{CreativityLevel}-{ModelCapability}
    - Creativity levels: VeryPrecise, HighlyPrecise, Precise, Moderate, Balanced, Creative, HighlyCreative, VeryCreative
    - Model capabilities: Pro (highest quality), Adv (balanced), Std (standard)
    """
    parts: List[str] = []
    for cand in candidates:
        agent_name = cand.get('name', '')
        name_interpretation = _interpret_agent_name(agent_name)
        role_score = cand.get("role_score")
        team_score = cand.get("team_score")
        cost_penalty = cand.get("cost_penalty")
        latency_penalty = cand.get("latency_penalty")

        score_lines: List[str] = []
        if role_score is not None:
            score_lines.append(f"  role_score: {float(role_score):.4f}")
        if team_score is not None:
            score_lines.append(f"  team_score: {float(team_score):.4f}")
        if cost_penalty is not None:
            score_lines.append(f"  cost_penalty: {float(cost_penalty):.4f}")
        if latency_penalty is not None:
            score_lines.append(f"  latency_penalty: {float(latency_penalty):.4f}")
        parts.append(
            "\n".join(
                [
                    f"- id: {cand.get('id')}",
                    f"  name: {agent_name}",
                    f"  name_interpretation: {name_interpretation}",
                    f"  domain_tags: {cand.get('domain_tags')}",
                    f"  role_tags: {cand.get('role_tags')}",
                    f"  tool_tags: {cand.get('tool_tags')}",
                    f"  description: {cand.get('description')}",
                    *score_lines,
                ]
            )
        )
    return "\n".join(parts)


def _interpret_agent_name(name: str) -> str:
    """Interpret agent name to provide context about its configuration.
    
    Examples:
    - AlgorithmArchitect-HighlyPrecise-Pro -> Highly precise, deterministic output (temp ~0.10-0.20), top-quality models
    - StringMaster-Moderate-Std -> Moderate creativity (temp ~0.30-0.40), standard models
    - Researcher-VeryCreative-Adv -> Very creative, diverse exploration (temp ≥0.70), advanced models
    """
    if not name or '-' not in name:
        return "Standard configuration"
    
    parts = name.split('-')
    if len(parts) < 3:
        return "Standard configuration"
    
    # Extract creativity and model from the last two parts
    model_capability = parts[-1]  # Pro, Adv, or Std
    creativity_level = parts[-2]  # VeryPrecise, HighlyPrecise, etc.
    
    # Map creativity to temperature range and description
    creativity_map = {
        'VeryPrecise': ('temp <0.10', 'extremely deterministic, minimal variation'),
        'HighlyPrecise': ('temp 0.10-0.20', 'highly deterministic, very consistent'),
        'Precise': ('temp 0.20-0.30', 'precise, consistent output'),
        'Moderate': ('temp 0.30-0.40', 'moderate creativity, balanced approach'),
        'Balanced': ('temp 0.40-0.50', 'balanced creativity and consistency'),
        'Creative': ('temp 0.50-0.60', 'creative, diverse output'),
        'HighlyCreative': ('temp 0.60-0.70', 'highly creative, exploratory'),
        'VeryCreative': ('temp ≥0.70', 'very creative, maximum diversity'),
    }
    
    # Map model to capability description
    model_map = {
        'Pro': 'highest quality models (GPT-4o dominant)',
        'Adv': 'balanced model mix (advanced capabilities)',
        'Std': 'standard models (cost-efficient)',
    }
    
    creativity_desc = creativity_map.get(creativity_level, ('unknown temp', 'unknown creativity'))
    model_desc = model_map.get(model_capability, 'unknown model capability')
    
    return f"{creativity_desc[1]} ({creativity_desc[0]}), {model_desc}"


def _select_with_router_llm(
    llm: LLMClient,
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    registry,
) -> Optional[str]:
    candidate_cards: List[dict] = []
    for cand in candidates:
        card_id = cand.get("card_id")
        if not card_id:
            continue
        card = registry.get(card_id)
        if card is None:
            continue
        candidate_cards.append(
            {
                "id": card.id,
                "name": card.name,
                "description": card.description,
                "domain_tags": card.domain_tags,
                "role_tags": card.role_tags,
                "tool_tags": card.tool_tags,
                "role_score": cand.get("role_score", cand.get("score")),
                "team_score": cand.get("team_score"),
                "cost_penalty": cand.get("cost_penalty"),
                "latency_penalty": cand.get("latency_penalty"),
            }
        )
    if not candidate_cards:
        return None

    candidates_text = _format_router_candidates(candidate_cards)
    system_msg = (
        "You are a routing model. Select the single best agent id from the candidate list. "
        "Each agent's name includes configuration info: {OriginalName}-{CreativityLevel}-{ModelCapability}. "
        "The 'name_interpretation' field explains the agent's creativity (temperature) and model quality. "
        "Consider both the agent's capabilities (description, tags) AND its configuration (creativity/model) when selecting. "
        "For tasks requiring precision/consistency, prefer Precise/HighlyPrecise agents. "
        "For tasks requiring exploration/diversity, prefer Creative/HighlyCreative/VeryCreative agents. "
        "Return ONLY JSON: {\"selected_id\": \"...\"}."
    )
    user_msg = (
        f"Role: {role}\n"
        f"Task: {task_text}\n"
        f"Constraints: {json.dumps(constraints or {}, ensure_ascii=True)}\n"
        f"Candidates:\n{candidates_text}\n"
    )
    response = llm.chat(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    data = _safe_load_json(str(response), {})
    selected_id = data.get("selected_id")
    valid_ids = {c["id"] for c in candidate_cards}
    return selected_id if selected_id in valid_ids else None


def _get_agent_context(registry, agent_id: Optional[str]) -> Dict[str, Any]:
    if not agent_id:
        print(f"[ERROR] _get_agent_context called with empty agent_id!", file=sys.stderr)
        return {"id": None, "name": None, "domain_tags": [], "tool_tags": []}
    card = registry.get(agent_id)
    if card is None:
        print(f"[ERROR] Agent '{agent_id}' not found in registry!", file=sys.stderr)
        return {"id": agent_id, "name": None, "domain_tags": [], "tool_tags": []}
    return {
        "id": card.id,
        "name": card.name,
        "domain_tags": card.domain_tags,
        "role_tags": card.role_tags,
        "tool_tags": card.tool_tags,
        "description": card.description,
        "embedding_text": card.embedding_text,
        "available_tool_ids": getattr(card, "available_tool_ids", []),
    }


def _summarize_output(output: Any, max_chars: int = 400) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        summary = output
    else:
        try:
            summary = json.dumps(output, ensure_ascii=True)
        except TypeError:
            summary = str(output)
    if len(summary) > max_chars:
        return summary[: max_chars - 3] + "..."
    return summary


def _truncate_text(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _extract_assertion_hints(message: Optional[str], max_items: int = 2) -> List[str]:
    if not message:
        return []
    hints: List[str] = []
    for line in str(message).splitlines():
        text = line.strip()
        if not text:
            continue
        if "assert" in text and "candidate" in text:
            hints.append(text)
            if len(hints) >= max_items:
                break
    return hints


def _summarize_failure(error_message: Optional[str], test_error: Optional[str], max_chars: int = 700) -> str:
    parts: List[str] = []
    if test_error:
        parts.append(f"test_error={_truncate_text(str(test_error), max_chars=max_chars)}")
    if error_message and error_message != test_error:
        parts.append(f"runtime_error={_truncate_text(str(error_message), max_chars=max_chars)}")
    return " | ".join(parts)


def _is_likely_error_text(code: str) -> bool:
    text = str(code or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if text.startswith("# Error") or text.startswith("Error:"):
        return True
    if "traceback (most recent call last)" in lowered:
        return True
    if "error calling llm" in lowered:
        return True
    if "timed out" in lowered or "timeout" in lowered:
        return True
    if text.startswith("Exception:"):
        return True
    return False


def _append_diagnostics_to_task(task_text: str, role_context: Dict[str, Any]) -> str:
    diagnostics = role_context.get("diagnostics") or {}
    if not diagnostics:
        return task_text
    if "Diagnostics:" in task_text:
        return task_text

    error_message = diagnostics.get("error_message") or ""
    test_error = diagnostics.get("test_error") or ""
    original_test_error = diagnostics.get("original_test_error") or ""
    latest_test_error = diagnostics.get("latest_test_error") or ""
    postprocess_error = diagnostics.get("postprocess_error") or ""
    failure_summary = diagnostics.get("failure_summary") or ""
    failed_code = diagnostics.get("failed_code") or ""

    lines = ["Diagnostics:"]
    if failure_summary:
        lines.append(f"Failure summary: {_truncate_text(str(failure_summary), max_chars=800)}")
    if error_message:
        lines.append(f"Error: {_truncate_text(error_message, max_chars=500)}")
    if test_error and test_error != error_message:
        lines.append(f"Test error: {_truncate_text(test_error, max_chars=500)}")
    if original_test_error and original_test_error not in {test_error, error_message}:
        lines.append(f"Original test error: {_truncate_text(original_test_error, max_chars=500)}")
    if latest_test_error and latest_test_error not in {test_error, error_message, original_test_error}:
        lines.append(f"Latest test error: {_truncate_text(latest_test_error, max_chars=500)}")
    if postprocess_error and postprocess_error not in {test_error, error_message, latest_test_error}:
        lines.append(f"Postprocess error: {_truncate_text(postprocess_error, max_chars=500)}")
    assertion_hints = _extract_assertion_hints(original_test_error or test_error)
    if assertion_hints:
        lines.append("Assertion hints:")
        for hint in assertion_hints:
            lines.append(f"- {_truncate_text(hint, max_chars=500)}")
    if failed_code:
        lines.append("Failed code:")
        lines.append(_truncate_text(str(failed_code), max_chars=1200))

    return f"{task_text}\n\n" + "\n".join(lines)


def _ensure_role_constraints(constraints: Optional[Dict[str, Any]], role: str) -> Dict[str, Any]:
    merged = dict(constraints) if constraints else {}
    role_tags = list(merged.get("role_tags") or [])
    if not role_tags:
        role_key = role.strip().lower()
        tags: List[str] = [role_key] if role_key else []
        if role_key == "checker":
            tags.append("tester")
            tags.append("code-testing")
        elif role_key == "tester":
            tags.append("checker")
            tags.append("code-testing")
        elif role_key in {"refractor", "refactor"}:
            tags.append("refactor")
            tags.append("refractor")
            tags.append("code-refactoring")
        elif role_key == "code-generation":
            tags.append("builder")
            tags.append("code-generation")
        elif role_key == "code-planner":
            tags.append("planner")
            tags.append("code-planner")
        elif role_key == "code-testing":
            tags.append("tester")
            tags.append("checker")
            tags.append("code-testing")
        elif role_key == "code-refactoring":
            tags.append("refactor")
            tags.append("refractor")
            tags.append("code-refactoring")
        if tags:
            merged["role_tags"] = tags
    return merged


def _build_selection_cache_key(*, role: str, task_text: str, step_idx: int, topology: str) -> str:
    digest = hashlib.sha256((task_text or "").encode("utf-8")).hexdigest()[:12]
    topo = (topology or "unknown").strip().lower()
    return f"{topo}|{role}|s{step_idx}|q{digest}"


def _jaccard_overlap(a: List[str], b: List[str]) -> float:
    set_a = {str(item).strip().lower() for item in (a or []) if str(item).strip()}
    set_b = {str(item).strip().lower() for item in (b or []) if str(item).strip()}
    if not set_a and not set_b:
        return 0.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return float(len(set_a.intersection(set_b)) / len(union))


def _tier_penalty(value: Optional[str]) -> float:
    if not value:
        return 0.25
    normalized = str(value).strip().lower()
    if normalized in {"low", "cheap", "economy", "fast"}:
        return 0.0
    if normalized in {"medium", "mid", "normal", "balanced"}:
        return 0.5
    if normalized in {"high", "expensive", "slow", "premium"}:
        return 1.0
    return 0.25


_ROLE_ALIAS = {
    "code-generation": "builder",
    "code-planner": "planner",
    "code-testing": "checker",
    "code-refactoring": "refactor",
    "tester": "checker",
    "refractor": "refactor",
}

_ROLE_PAIR_WEIGHTS = {
    ("researcher", "planner"): 1.15,
    ("researcher", "builder"): 1.10,
    ("planner", "builder"): 1.25,
    ("builder", "refactor"): 1.20,
    ("refactor", "checker"): 1.20,
    ("checker", "refactor"): 1.10,
    ("planner", "checker"): 1.05,
    ("builder", "checker"): 1.35,
    ("checker", "builder"): 1.10,
}


def _normalize_role_name(role: Optional[str]) -> str:
    key = str(role or "").strip().lower()
    if not key:
        return ""
    return _ROLE_ALIAS.get(key, key)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y", "ready", "done"}
    return False


def _planner_ready_to_handoff(output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    if "ready_to_handoff" in output:
        return _to_bool(output.get("ready_to_handoff"))
    if str(output.get("next_role") or "").strip():
        return True
    status = str(output.get("status") or "").strip().lower()
    if status in {"delegate", "handoff", "ready_to_handoff", "plan_ready"}:
        return True
    steps = output.get("steps")
    if isinstance(steps, list) and len(steps) > 0:
        return True
    return False


def _planner_next_role(output: Any) -> Optional[str]:
    if not isinstance(output, dict):
        return None
    candidate = str(output.get("next_role") or "").strip()
    if not candidate:
        return None
    normalized = _normalize_role_name(candidate)
    if normalized in {"finish", "none", "null"}:
        return None
    return normalized


def _role_pair_weight(prev_role: Optional[str], curr_role: Optional[str]) -> float:
    prev = _normalize_role_name(prev_role)
    curr = _normalize_role_name(curr_role)
    if not prev or not curr:
        return 1.0
    if prev == curr:
        return 0.8
    return float(_ROLE_PAIR_WEIGHTS.get((prev, curr), 1.0))


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / float(len(values)))


def _io_chain_score(prev_card: Any, curr_card: Any) -> float:
    if prev_card is None or curr_card is None:
        return 0.0

    format_overlap = _jaccard_overlap(
        list(getattr(prev_card, "output_formats", []) or []),
        list(getattr(curr_card, "output_formats", []) or []),
    )
    modality_overlap = _jaccard_overlap(
        list(getattr(prev_card, "modalities", []) or []),
        list(getattr(curr_card, "modalities", []) or []),
    )
    tool_overlap = _jaccard_overlap(
        list(getattr(prev_card, "tool_tags", []) or []),
        list(getattr(curr_card, "tool_tags", []) or []),
    )

    io_match = (0.5 * format_overlap) + (0.3 * modality_overlap) + (0.2 * tool_overlap)
    conversion_penalty = 1.0 - format_overlap

    prev_tool_tags = {str(tag).strip().lower() for tag in (getattr(prev_card, "tool_tags", []) or []) if str(tag).strip()}
    curr_tool_tags = {str(tag).strip().lower() for tag in (getattr(curr_card, "tool_tags", []) or []) if str(tag).strip()}
    cleaner_tokens = ("clean", "normalize", "format", "parse", "sanitize")
    prev_is_cleaner = any(any(token in tag for token in cleaner_tokens) for tag in prev_tool_tags)
    curr_is_cleaner = any(any(token in tag for token in cleaner_tokens) for tag in curr_tool_tags)
    repeated_clean_penalty = 0.2 if (prev_is_cleaner and curr_is_cleaner) else 0.0

    score = io_match - (0.3 * conversion_penalty) - repeated_clean_penalty
    return float(max(-1.0, min(1.0, score)))


def _ensure_pair_bandit_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pair_bandits (
            workflow_version TEXT NOT NULL,
            prev_role TEXT NOT NULL,
            prev_card_id TEXT NOT NULL,
            role TEXT NOT NULL,
            card_id TEXT NOT NULL,
            alpha REAL NOT NULL,
            beta REAL NOT NULL,
            PRIMARY KEY (workflow_version, prev_role, prev_card_id, role, card_id)
        )
        """
    )
    conn.commit()


def _pair_success_key(
    prev_role: Optional[str],
    prev_card_id: Optional[str],
    role: Optional[str],
    card_id: Optional[str],
) -> Tuple[str, str, str, str]:
    return (
        _normalize_role_name(prev_role),
        str(prev_card_id or "").strip(),
        _normalize_role_name(role),
        str(card_id or "").strip(),
    )


def _load_pair_success_priors(
    db_path: Optional[str],
    workflow_version: str,
) -> Dict[Tuple[str, str, str, str], float]:
    priors: Dict[Tuple[str, str, str, str], float] = {}
    if not db_path or not os.path.exists(db_path):
        return priors
    conn = sqlite3.connect(db_path)
    try:
        _ensure_pair_bandit_schema(conn)
        cursor = conn.execute(
            """
            SELECT prev_role, prev_card_id, role, card_id, alpha, beta
            FROM pair_bandits
            WHERE workflow_version = ?
            """,
            (workflow_version,),
        )
        for row in cursor.fetchall():
            key = _pair_success_key(row[0], row[1], row[2], row[3])
            alpha = float(row[4] or 1.0)
            beta = float(row[5] or 1.0)
            denom = alpha + beta
            priors[key] = float(alpha / denom) if denom > 0 else 0.5
    except Exception:
        return {}
    finally:
        conn.close()
    return priors


def _pair_success_rate(
    priors: Optional[Dict[Tuple[str, str, str, str], float]],
    *,
    prev_role: Optional[str],
    prev_card_id: Optional[str],
    role: Optional[str],
    card_id: Optional[str],
) -> float:
    if not priors:
        return 0.5
    key = _pair_success_key(prev_role, prev_card_id, role, card_id)
    return float(priors.get(key, 0.5))


def _update_pair_success_stats(
    db_path: Optional[str],
    *,
    workflow_version: str,
    selections: List[Dict[str, Any]],
    reward: float,
    confidence: float = 1.0,
) -> None:
    if not db_path or not selections:
        return

    reward = max(0.0, min(1.0, float(reward)))
    confidence = max(0.0, min(1.0, float(confidence)))
    if confidence <= 0.0:
        return

    conn = sqlite3.connect(db_path)
    try:
        _ensure_pair_bandit_schema(conn)
        for idx in range(1, len(selections)):
            prev = selections[idx - 1] or {}
            curr = selections[idx] or {}

            prev_role = _normalize_role_name(prev.get("role"))
            prev_card_id = str(prev.get("selected_main") or "").strip()
            role = _normalize_role_name(curr.get("role"))
            card_id = str(curr.get("selected_main") or "").strip()
            if not prev_role or not prev_card_id or not role or not card_id:
                continue

            cursor = conn.execute(
                """
                SELECT alpha, beta FROM pair_bandits
                WHERE workflow_version = ? AND prev_role = ? AND prev_card_id = ? AND role = ? AND card_id = ?
                """,
                (workflow_version, prev_role, prev_card_id, role, card_id),
            )
            row = cursor.fetchone()
            alpha = float(row[0]) if row else 1.0
            beta = float(row[1]) if row else 1.0
            alpha += reward * confidence
            beta += (1.0 - reward) * confidence
            conn.execute(
                """
                INSERT OR REPLACE INTO pair_bandits
                (workflow_version, prev_role, prev_card_id, role, card_id, alpha, beta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (workflow_version, prev_role, prev_card_id, role, card_id, alpha, beta),
            )
        conn.commit()
    finally:
        conn.close()


def _apply_team_level_scoring(
    *,
    candidates: List[Dict[str, Any]],
    registry,
    selected_agent_ids: List[str],
    lambda_compat: float,
    mu_cost: float,
    nu_latency: float,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    _ = lambda_compat  # kept for backward-compatible signature.
    _ = selected_agent_ids  # keep signature stable; similarity-based team overlap is intentionally disabled.

    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        enriched = dict(candidate)
        card_id = candidate.get("card_id")
        card = registry.get(card_id) if card_id else None

        role_score = float(candidate.get("score", 0.0) or 0.0)
        cost_penalty = _tier_penalty(getattr(card, "cost_tier", None)) if card is not None else 0.25
        latency_penalty = _tier_penalty(getattr(card, "latency_tier", None)) if card is not None else 0.25
        team_score = role_score - (mu_cost * cost_penalty) - (nu_latency * latency_penalty)

        enriched["role_score"] = role_score
        enriched["cost_penalty"] = cost_penalty
        enriched["latency_penalty"] = latency_penalty
        enriched["team_score"] = team_score
        scored.append(enriched)

    scored.sort(key=lambda item: float(item.get("team_score", item.get("score", 0.0)) or 0.0), reverse=True)
    return scored


def _unary_score(candidate: Dict[str, Any], card: Any, mu_cost: float, nu_latency: float) -> float:
    base = float(candidate.get("score", 0.0) or 0.0)
    cost_penalty = _tier_penalty(getattr(card, "cost_tier", None)) if card is not None else 0.25
    latency_penalty = _tier_penalty(getattr(card, "latency_tier", None)) if card is not None else 0.25
    return float(base - (mu_cost * cost_penalty) - (nu_latency * latency_penalty))


def _prepare_role_candidates(
    *,
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    registry,
    index,
    embedder,
    reranker: TfidfLinearReranker,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    router_no_rerank: bool,
    selected_agent_ids: Optional[List[str]],
    team_lambda_compat: float,
    team_mu_cost: float,
    team_nu_latency: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = get_candidates(
        task_text=task_text,
        role=role,
        constraints=constraints,
        kind="agent",
        top_n=top_n,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        registry=registry,
        index=index,
        embedder=embedder,
    )

    if candidates:
        if router_no_rerank:
            reranked = list(candidates)
        else:
            candidate_texts = _build_candidate_texts(registry, candidates)
            query_text = build_role_query(task_text, role, constraints)
            rerank_indices, _ = reranker.rank(query_text, candidate_texts, top_m=min(rerank_top_m, len(candidates)))
            reranked = [candidates[i] for i in rerank_indices]
    else:
        reranked = []

    base_ranked = reranked if reranked else candidates
    scored = _apply_team_level_scoring(
        candidates=base_ranked,
        registry=registry,
        selected_agent_ids=list(selected_agent_ids or []),
        lambda_compat=team_lambda_compat,
        mu_cost=team_mu_cost,
        nu_latency=team_nu_latency,
    )
    return candidates, (scored if scored else base_ranked)


class _MCTSNode:
    def __init__(
        self,
        *,
        role: Optional[str],
        depth: int,
        selected_ids: Tuple[str, ...],
        selected_roles: Tuple[str, ...],
        parent: Optional["_MCTSNode"] = None,
        action_card_id: Optional[str] = None,
        action_reward: float = 0.0,
    ) -> None:
        self.role = role
        self.depth = depth
        self.selected_ids = selected_ids
        self.selected_roles = selected_roles
        self.parent = parent
        self.action_card_id = action_card_id
        self.action_reward = action_reward
        self.children: Dict[str, _MCTSNode] = {}
        self.untried_actions: List[str] = []
        self.visits: int = 0
        self.total_reward: float = 0.0

    @property
    def value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.total_reward / float(self.visits)


def _select_agent_with_mcts_dynamic(
    *,
    task_text: str,
    role: str,
    step_idx: int,
    config: TopologyConfig,
    constraints_per_role: Dict[str, Dict[str, Any]],
    selected_agent_ids: List[str],
    selected_agent_roles: List[str],
    registry,
    index,
    embedder,
    reranker: TfidfLinearReranker,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    router_no_rerank: bool,
    team_lambda_compat: float,
    team_mu_cost: float,
    team_nu_latency: float,
    mcts_iterations: int,
    mcts_rollout_depth: int,
    mcts_exploration: float,
    mcts_discount: float,
    mcts_max_candidates: int,
    pair_success_priors: Optional[Dict[Tuple[str, str, str, str], float]],
) -> Optional[Dict[str, Any]]:
    available_roles = list(config.roles or [role])
    if role not in available_roles:
        available_roles.insert(0, role)

    role_cache: Dict[str, List[Dict[str, Any]]] = {}
    card_cache: Dict[str, Any] = {}
    role_candidate_map: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _get_candidates_for_role(target_role: str) -> List[Dict[str, Any]]:
        if target_role in role_cache:
            return role_cache[target_role]
        role_constraints = _ensure_role_constraints(constraints_per_role.get(target_role), target_role)
        _, ranked = _prepare_role_candidates(
            task_text=task_text,
            role=target_role,
            constraints=role_constraints,
            registry=registry,
            index=index,
            embedder=embedder,
            reranker=reranker,
            top_n=top_n,
            top_k=top_k,
            rerank_top_m=rerank_top_m,
            mmr_lambda=mmr_lambda,
            router_no_rerank=router_no_rerank,
            selected_agent_ids=selected_agent_ids,
            team_lambda_compat=team_lambda_compat,
            team_mu_cost=team_mu_cost,
            team_nu_latency=team_nu_latency,
        )
        limited = list(ranked[: max(1, mcts_max_candidates)])
        role_cache[target_role] = limited
        role_candidate_map[target_role] = {}
        for item in limited:
            cid = str(item.get("card_id") or "")
            if not cid:
                continue
            role_candidate_map[target_role][cid] = item
            if cid not in card_cache:
                card_cache[cid] = registry.get(cid)
        return limited

    team_beta_bottleneck = max(0.0, 0.75 * float(team_lambda_compat))
    team_eta_io = max(0.0, 0.5 * float(team_lambda_compat))
    team_zeta_success = max(0.0, 0.8 * float(team_lambda_compat))

    def _action_reward(
        prev_ids: Tuple[str, ...],
        prev_roles: Tuple[str, ...],
        target_role: str,
        card_id: str,
    ) -> float:
        cand = role_candidate_map.get(target_role, {}).get(card_id)
        card = card_cache.get(card_id)
        reward = _unary_score(cand or {"score": 0.0}, card, team_mu_cost, team_nu_latency)
        if not prev_ids:
            return float(reward)

        success_weighted_sum = 0.0
        success_weight_sum = 0.0
        success_values: List[float] = []

        for idx, prev_id in enumerate(prev_ids):
            prev_card = card_cache.get(prev_id) or registry.get(prev_id)
            if prev_card is not None and prev_id not in card_cache:
                card_cache[prev_id] = prev_card
            prev_role = prev_roles[idx] if idx < len(prev_roles) else ""
            pair_weight = _role_pair_weight(prev_role, target_role)

            success_rate = _pair_success_rate(
                pair_success_priors,
                prev_role=prev_role,
                prev_card_id=prev_id,
                role=target_role,
                card_id=card_id,
            )
            success_weighted_sum += pair_weight * success_rate
            success_weight_sum += pair_weight
            success_values.append(success_rate)

        mean_success = (success_weighted_sum / success_weight_sum) if success_weight_sum > 0 else 0.5
        min_success = min(success_values) if success_values else 0.5
        bottleneck_penalty = 1.0 - min_success

        prev_last_id = prev_ids[-1]
        prev_last_card = card_cache.get(prev_last_id) or registry.get(prev_last_id)
        if prev_last_card is not None and prev_last_id not in card_cache:
            card_cache[prev_last_id] = prev_last_card
        io_chain = _io_chain_score(prev_last_card, card)

        centered_success = mean_success - 0.5

        reward -= float(team_beta_bottleneck) * bottleneck_penalty
        reward += float(team_eta_io) * io_chain
        reward += float(team_zeta_success) * centered_success
        return float(reward)

    def _predict_next_role(current_role: Optional[str], depth: int) -> Optional[str]:
        if current_role is None:
            return None
        if depth >= max(1, mcts_rollout_depth):
            return None
        neighbors = _neighbors_for_role(current_role, config)
        candidates = neighbors if neighbors else available_roles
        if not candidates:
            return None
        avoid_role = current_role if len(candidates) > 1 else None
        return _select_next_role(
            task_text=task_text,
            available_roles=candidates,
            history=[],
            llm=None,
            avoid_role=avoid_role,
        )

    root_candidates = _get_candidates_for_role(role)
    if not root_candidates:
        return None

    normalized_role_history = tuple(_normalize_role_name(item) for item in (selected_agent_roles or []))
    root = _MCTSNode(
        role=role,
        depth=0,
        selected_ids=tuple(selected_agent_ids),
        selected_roles=normalized_role_history,
    )
    root.untried_actions = [str(item.get("card_id")) for item in root_candidates if item.get("card_id")]

    iterations = max(1, int(mcts_iterations))
    depth_limit = max(1, int(mcts_rollout_depth))
    exploration = max(0.0, float(mcts_exploration))
    discount = min(1.0, max(0.0, float(mcts_discount)))

    def _is_terminal(node: _MCTSNode) -> bool:
        return node.role is None or node.depth >= depth_limit

    for _ in range(iterations):
        node = root
        path = [root]
        total_reward = 0.0

        while not _is_terminal(node) and not node.untried_actions and node.children:
            best_child = None
            best_ucb = None
            parent_visits = max(1, node.visits)
            for child in node.children.values():
                if child.visits <= 0:
                    ucb = float("inf")
                else:
                    exploit = child.value
                    explore = exploration * ((2.0 * max(1e-9, float(math.log(parent_visits)))) / child.visits) ** 0.5
                    ucb = exploit + explore
                if best_ucb is None or ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            if best_child is None:
                break
            node = best_child
            path.append(node)
            total_reward += (discount ** max(0, node.depth - 1)) * node.action_reward

        if not _is_terminal(node) and node.untried_actions:
            action_id = node.untried_actions.pop(0)
            reward = _action_reward(node.selected_ids, node.selected_roles, node.role or "", action_id)
            next_role = _predict_next_role(node.role, node.depth + 1)
            child = _MCTSNode(
                role=next_role,
                depth=node.depth + 1,
                selected_ids=node.selected_ids + (action_id,),
                selected_roles=node.selected_roles + (_normalize_role_name(node.role),),
                parent=node,
                action_card_id=action_id,
                action_reward=reward,
            )
            if child.role:
                child.untried_actions = [
                    str(item.get("card_id"))
                    for item in _get_candidates_for_role(child.role)
                    if item.get("card_id")
                ]
            node.children[action_id] = child
            node = child
            path.append(node)
            total_reward += (discount ** max(0, node.depth - 1)) * reward

        sim_role = node.role
        sim_depth = node.depth
        sim_selected = list(node.selected_ids)
        sim_selected_roles = list(node.selected_roles)
        while sim_role is not None and sim_depth < depth_limit:
            rollout_candidates = _get_candidates_for_role(sim_role)
            if not rollout_candidates:
                break
            best_rollout = None
            best_reward = None
            prev_ids_tuple = tuple(sim_selected)
            prev_roles_tuple = tuple(sim_selected_roles)
            for cand in rollout_candidates:
                cand_id = str(cand.get("card_id") or "")
                if not cand_id:
                    continue
                reward = _action_reward(prev_ids_tuple, prev_roles_tuple, sim_role, cand_id)
                if best_reward is None or reward > best_reward:
                    best_reward = reward
                    best_rollout = cand_id
            if best_rollout is None or best_reward is None:
                break
            total_reward += (discount ** sim_depth) * float(best_reward)
            sim_selected.append(best_rollout)
            sim_selected_roles.append(_normalize_role_name(sim_role))
            sim_depth += 1
            sim_role = _predict_next_role(sim_role, sim_depth)

        for path_node in path:
            path_node.visits += 1
            path_node.total_reward += float(total_reward)

    if not root.children:
        return None

    best_main = None
    best_visit = None
    for card_id, child in root.children.items():
        if best_visit is None or child.visits > best_visit:
            best_visit = child.visits
            best_main = card_id
    if not best_main:
        return None

    shadow_sorted = sorted(
        [(cid, child.visits, child.value) for cid, child in root.children.items() if cid != best_main],
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    selected_shadows = [item[0] for item in shadow_sorted[:2]]
    selected_shadow = selected_shadows[0] if selected_shadows else None

    reranked = _get_candidates_for_role(role)
    selection = {
        "selected_main": best_main,
        "selected_shadows": selected_shadows,
        "selected_shadow": selected_shadow,
        "candidates": list(reranked),
        "reranked": list(reranked),
        "mcts": {
            "iterations": iterations,
            "depth_limit": depth_limit,
            "children": {
                cid: {"visits": child.visits, "avg_reward": child.value}
                for cid, child in root.children.items()
            },
        },
    }
    return selection


def _append_role_result(results: Dict[str, Any], role: str, output: Any) -> None:
    if role not in results:
        results[role] = output
        return
    existing = results[role]
    if isinstance(existing, list):
        existing.append(output)
        return
    results[role] = [existing, output]


def _append_tool_exec(tool_exec: Dict[str, Any], role: str, tool_results: Any) -> None:
    if role not in tool_exec:
        tool_exec[role] = tool_results
        return
    existing = tool_exec[role]
    if isinstance(existing, list) and isinstance(tool_results, list):
        existing.extend(tool_results)
        return
    tool_exec[role] = [existing, tool_results]


def _describe_tools(registry, tool_ids: List[str]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for tool_id in tool_ids:
        card = registry.get(tool_id)
        if card is None:
            continue
        tools.append(
            {
                "id": card.id,
                "name": card.name,
                "description": card.description,
            }
        )
    return tools


def _normalize_tool_input(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = _safe_load_json(value, None)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _resolve_tool_id(
    decision: Dict[str, Any], tool_ids: List[str], tools_info: List[Dict[str, Any]]
) -> Optional[str]:
    raw = (
        decision.get("tool_id")
        or decision.get("tool")
        or decision.get("tool_name")
        or decision.get("name")
        or decision.get("id")
    )
    if isinstance(raw, dict):
        raw = raw.get("id") or raw.get("tool_id") or raw.get("name")
    if isinstance(raw, int):
        idx = raw
        if 0 <= idx < len(tools_info):
            return tools_info[idx].get("id")
        if 1 <= idx <= len(tools_info):
            return tools_info[idx - 1].get("id")
        return None
    if isinstance(raw, str):
        candidate = raw.strip()
        if candidate in tool_ids:
            return candidate
        if ":" in candidate:
            tail = candidate.split(":")[-1].strip()
            if tail in tool_ids:
                return tail
            candidate = tail
        if candidate.isdigit():
            idx = int(candidate)
            if 0 <= idx < len(tools_info):
                return tools_info[idx].get("id")
            if 1 <= idx <= len(tools_info):
                return tools_info[idx - 1].get("id")
        lower = candidate.lower()
        for tool in tools_info:
            name = str(tool.get("name") or "").strip().lower()
            tool_id = str(tool.get("id") or "").strip().lower()
            if lower == name or lower == tool_id:
                return tool.get("id")
    return None


def _plan_and_run_tools(
    *,
    llm: Any,
    role: str,
    task_text: str,
    role_context: Dict[str, Any],
    agent_context: Dict[str, Any],
    registry,
    tool_executor: Optional[ToolExecutor],
    results: Dict[str, Any],
    tool_max_rounds: int,
    tool_only: bool,
    task_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    tool_history: List[Dict[str, Any]] = []
    # Ensure no stale success flag leaks into this run
    role_context.pop("test_passed", None)
    if tool_executor is None:
        return tool_history
    tool_ids = list(agent_context.get("available_tool_ids") or [])
    if not tool_ids:
        tool_tags = list(agent_context.get("tool_tags") or [])
        if tool_tags:
            try:
                tool_cards = registry.list({"kind": "tool", "tool_tags": tool_tags})
                tool_ids = [card.id for card in tool_cards]
            except Exception:
                tool_ids = []
    if not tool_ids:
        agent_name = agent_context.get("name") or agent_context.get("id") or "unknown"
        agent_tags = agent_context.get("tool_tags") or []
        print(f"[WARNING] Agent '{agent_name}' has NO tools! tool_tags={agent_tags}, role={role}", file=sys.stderr)
        return tool_history
    
    agent_name = agent_context.get("name") or agent_context.get("id") or "unknown"
    print(f"[DEBUG] Agent '{agent_name}' has {len(tool_ids)} tools for role '{role}': {tool_ids[:3]}...", file=sys.stderr)
    
    tools_info = _describe_tools(registry, tool_ids)
    role_context["available_tools"] = tools_info
    if tool_only:
        role_context["tool_only"] = True
    
    # Initialize tracking variables
    used_tool_ids = set()  # Track which tools have been tried
    tool_failure_counts = {}  # Track failures per tool for switching strategy
    tool_category_counts = {}  # Track failures per tool category (e.g., 'assemblesnippets', 'generatemath')
    successful_code = None  # Track if we got non-placeholder code
    test_passed = False  # Track if code passed actual test
    has_errors = False  # Track if last tool had errors
    failed_code = None  # Track the last failed code for refinement
    last_error = None  # Track the last error message
    last_test_error = None  # Track the last test failure message
    original_test_error = None  # First test failure in current refinement chain
    latest_test_error = None  # Most recent test/postprocess failure
    postprocess_error = None  # Most recent postprocess-specific failure
    consecutive_same_category_failures = 0  # Track consecutive failures in same category
    consecutive_failures = 0  # Track consecutive failures across tools
    
    # Early stopping detection
    last_code_length = None
    same_code_length_count = 0
    tried_categories = set()  # Track which categories we've tried
    
    last_tool_category = None  # Track previous tool category
    
    for round_idx in range(max(0, tool_max_rounds)):
        # Early termination: if code passed test, stop immediately
        if test_passed and not tool_only:
            break
            
        decision = llm.plan_tool(role, task_text, role_context, tools_info, tool_history)
        action = str(decision.get("action") or "final").strip().lower()
        tool_id = _resolve_tool_id(decision, tool_ids, tools_info)
        forced = False
        
        # Check if LLM wants to stop
        if action != "tool":
            if tool_only:
                # In tool_only mode, must use tools
                action = "tool"
                forced = True
                # If LLM chose final but we still have errors, suggest an unused tool
                if has_errors and used_tool_ids:
                    # Try to find an unused tool for LLM to consider
                    unused_tools = [tid for tid in tool_ids if tid not in used_tool_ids]
                    if unused_tools:
                        # Don't force a specific tool, let LLM choose in next iteration
                        pass
            else:
                # Not tool_only mode
                # If we have errors and haven't tried all tools, give LLM another chance
                if has_errors and len(used_tool_ids) < len(tool_ids):
                    # Suggest to LLM that there are more tools available
                    unused_tools = [tid for tid in tool_ids if tid not in used_tool_ids]
                    if unused_tools:
                        # Add hint to context for next iteration
                        role_context["suggested_tools"] = unused_tools
                        # Let LLM reconsider
                        continue
                # Otherwise, stop
                break
        
        # CRITICAL: If we forced tool diversity, override LLM's choice
        if role_context.get("force_different_category") or role_context.get("force_different_tool"):
            suggested = role_context.get("suggested_tools", [])
            if suggested:
                # Check if LLM chose a tool from the suggested list
                if tool_id not in suggested:
                    # LLM ignored our suggestion, force the first suggested tool
                    old_tool = tool_id
                    tool_id = suggested[0]
                    print(f"[DEBUG] ⚠️ FORCING tool switch: {old_tool} → {tool_id} (LLM ignored suggestion)", file=sys.stderr)
                    forced = True
                else:
                    print(f"[DEBUG] ✓ LLM accepted suggestion: {tool_id}", file=sys.stderr)
        
        if tool_id not in tool_ids:
            if tool_only and tool_ids:
                # Tool ID is invalid, select first unused tool or fallback to first tool
                for tid in tool_ids:
                    if tid not in used_tool_ids:
                        tool_id = tid
                        break
                else:
                    tool_id = tool_ids[0]
                forced = True
            else:
                tool_history.append(
                    {
                        "tool_id": tool_id,
                        "ok": False,
                        "error": {"code": "tool_not_allowed", "message": "Tool not in agent tool list"},
                    }
                )
                break
        
        used_tool_ids.add(tool_id)
        
        tool_input = _normalize_tool_input(decision.get("tool_input"))
        exec_inputs = {
            "query": task_text,
            "task": task_text,
            "role": role,
            "agent": agent_context,
            "upstream": results,
        }
        # Add task context if available (e.g., entry_point for HumanEval)
        if task_context:
            exec_inputs["task_context"] = task_context
        
        # Pass failed code and error to next tool for refinement
        if last_error:
            if failed_code:
                exec_inputs["failed_code"] = failed_code
            exec_inputs["error_message"] = last_error
            
            # Build detailed diagnostic message
            diagnostic_hints = []
            
            if last_test_error:
                exec_inputs["test_error"] = last_test_error
                if original_test_error:
                    exec_inputs["original_test_error"] = original_test_error
                if latest_test_error:
                    exec_inputs["latest_test_error"] = latest_test_error
                if postprocess_error:
                    exec_inputs["postprocess_error"] = postprocess_error
                failure_summary = _summarize_failure(
                    last_error,
                    original_test_error or last_test_error,
                )
                if failure_summary:
                    exec_inputs["failure_summary"] = failure_summary
                
                # Analyze specific error patterns and provide targeted hints
                focus_error_text = original_test_error or last_test_error
                error_lower = focus_error_text.lower()
                
                # Type checking issues - be very aggressive about isinstance(x, int) bug
                if failed_code and "isinstance" in failed_code and "int" in failed_code:
                    if "5.0" in focus_error_text or "float" in error_lower or "assertionerror" in error_lower:
                        diagnostic_hints.append(
                            "❌ TYPE CHECK ERROR: Using isinstance(x, int) filters out floats like 5.0 that are mathematically integers.\n"
                            "✅ CRITICAL FIX: Replace isinstance(x, int) with: isinstance(x, (int, float)) and x == int(x)\n"
                            "   This accepts both 5 and 5.0 as valid integers."
                        )
                
                # Assertion errors with specific values
                if "assertionerror" in error_lower:
                    # Extract expected vs actual if present
                    if "==" in focus_error_text:
                        diagnostic_hints.append(
                            f"❌ ASSERTION FAILED: The test expectation was not met.\n"
                            f"✅ HINT: Check the logic carefully - the output doesn't match expected result."
                        )
                
                # Index/Key errors
                if "indexerror" in error_lower or "keyerror" in error_lower:
                    diagnostic_hints.append(
                        "❌ ACCESS ERROR: Trying to access invalid index/key.\n"
                        "✅ FIX: Add boundary checks or validate indices before access."
                    )
                
                # Name errors
                if "nameerror" in error_lower:
                    diagnostic_hints.append(
                        "❌ NAME ERROR: Variable not defined.\n"
                        "✅ FIX: Check variable names and ensure they're defined before use."
                    )
                
                # Type errors (wrong argument types)
                if "typeerror" in error_lower and "argument" in error_lower:
                    diagnostic_hints.append(
                        "❌ ARGUMENT ERROR: Wrong type passed to function.\n"
                        "✅ FIX: Check parameter types and convert if needed."
                    )
                
                # Build refinement request with diagnostics
                refinement_parts = [
                    "🔍 PREVIOUS ATTEMPT FAILED - DETAILED ANALYSIS:",
                    f"\n📋 Primary Test Error: {focus_error_text[:300]}",
                ]
                if postprocess_error and postprocess_error != focus_error_text:
                    refinement_parts.append(f"\n📋 Latest Postprocess Error: {postprocess_error[:300]}")
                
                if diagnostic_hints:
                    refinement_parts.append("\n💡 SPECIFIC ISSUES IDENTIFIED:")
                    refinement_parts.extend(f"\n{hint}" for hint in diagnostic_hints)
                
                failure_code_text = failed_code or "(no code)"
                refinement_parts.append(
                    f"\n\n❌ Failed Code:\n{failure_code_text}\n"
                    f"\n⚠️ CRITICAL: Do NOT repeat the same logic error. "
                    f"Analyze the error carefully and implement a DIFFERENT approach to fix it."
                )
                
                exec_inputs["refinement_request"] = "\n".join(refinement_parts)
            else:
                failure_code_text = failed_code or "(no code)"
                exec_inputs["refinement_request"] = (
                    f"🔍 PREVIOUS ATTEMPT FAILED:\n"
                    f"Error: {last_error}\n\n"
                    f"❌ Failed Code:\n{failure_code_text}\n\n"
                    f"⚠️ Please fix the error above and generate corrected code."
                )
            
            refinement_request = exec_inputs.get("refinement_request")
            if refinement_request:
                exec_inputs["task"] = f"{task_text}\n\n{refinement_request}"
                exec_inputs["query"] = exec_inputs["task"]

            # Force tool diversity after consecutive failures
            # NOTE: Only switches within current agent's available tools (tool_ids is from agent_context)
            if consecutive_same_category_failures >= 3:
                # Get tools from DIFFERENT categories within this agent's tool list
                different_category_tools = []
                for tid in tool_ids:  # tool_ids = current agent's available tools only
                    parts = tid.split('-')
                    if len(parts) >= 3:
                        tool_family = parts[2]
                        import re
                        tid_category = re.sub(r'\d+$', '', tool_family)
                    else:
                        tid_category = tid.rsplit('-', 1)[0] if '-' in tid else tid
                    
                    if tid_category != tool_category:
                        different_category_tools.append(tid)
                
                if different_category_tools:
                    # Only suggest tools from current agent's pool, just different category
                    role_context["suggested_tools"] = different_category_tools
                    role_context["force_different_category"] = True
                    role_context["failed_category"] = tool_category
                    # Extract categories from different_category_tools
                    alt_categories = list(set([re.sub(r'\d+$', '', t.split('-')[2]) if len(t.split('-')) >= 3 else t for t in different_category_tools[:5]]))
                    role_context["reason"] = (
                        f"Tool category '{tool_category}' failed {consecutive_same_category_failures} times consecutively. "
                        f"MUST switch to a DIFFERENT tool category (within current agent's tools) from: {alt_categories}"
                    )
                    print(f"[DEBUG] Forcing tool category switch from '{tool_category}' after {consecutive_same_category_failures} failures", file=sys.stderr)
                    print(f"[DEBUG] Available different-category tools in current agent: {different_category_tools[:3]}", file=sys.stderr)
                else:
                    # No different category tools available - force switch to DIFFERENT tool within same category
                    same_category_different_tools = [tid for tid in tool_ids if tid != tool_id]
                    if same_category_different_tools:
                        role_context["suggested_tools"] = same_category_different_tools
                        role_context["force_different_tool"] = True
                        role_context["failed_tool"] = tool_id
                        role_context["reason"] = (
                            f"Tool '{tool_id}' failed {consecutive_same_category_failures} times consecutively. "
                            f"Current agent only has '{tool_category}' category tools. "
                            f"MUST switch to a DIFFERENT tool from: {same_category_different_tools[:5]}"
                        )
                        print(f"[DEBUG] Agent only has '{tool_category}' tools - forcing switch from '{tool_id}' to different tool", file=sys.stderr)
                        print(f"[DEBUG] Available alternative tools: {same_category_different_tools[:3]}", file=sys.stderr)
                    else:
                        print(f"[DEBUG] Cannot switch: agent only has one tool '{tool_id}'", file=sys.stderr)
            
            # If same specific tool failed 3+ times, suggest alternatives
            elif tool_failure_counts.get(tool_id, 0) >= 3:
                unused_tools = [tid for tid in tool_ids if tid not in used_tool_ids or tool_failure_counts.get(tid, 0) < 3]
                if unused_tools:
                    role_context["suggested_tools"] = unused_tools
                    role_context["reason"] = f"Tool {tool_id} failed {tool_failure_counts[tool_id]} times, consider alternative"
        
        exec_inputs.update(tool_input)
        
        # Debug output: verify refinement_request is being passed
        if "refinement_request" in exec_inputs:
            print(f"\n[DEBUG] Round {round_idx}: Passing refinement_request to tool {tool_id}", file=sys.stderr)
            print(f"[DEBUG] Refinement request (first 300 chars):", file=sys.stderr)
            print(f"[DEBUG] {exec_inputs['refinement_request'][:300]}...", file=sys.stderr)
        if "failed_code" in exec_inputs:
            print(f"[DEBUG] Passing failed_code (length: {len(exec_inputs['failed_code'])})", file=sys.stderr)
        if "test_error" in exec_inputs:
            print(f"[DEBUG] Passing test_error: {exec_inputs['test_error'][:100]}...", file=sys.stderr)
        
        tool_result = tool_executor.run_tool(tool_id, exec_inputs)
        
        # Extract tool category for diversity tracking (use the specific tool family, not the broad category)
        # Tool IDs are like: code-generation-assemblesnippets, code-generation-generatemath6
        # We want to extract: assemblesnippets, generatemath (without the number suffix)
        parts = tool_id.split('-')
        if len(parts) >= 3:
            # Extract the tool family name (e.g., 'assemblesnippets', 'generatemath')
            tool_family = parts[2]
            # Remove trailing numbers (e.g., 'generatemath6' -> 'generatemath')
            import re
            tool_category = re.sub(r'\d+$', '', tool_family)
        else:
            tool_category = tool_id.rsplit('-', 1)[0] if '-' in tool_id else tool_id
        
        # Note: We track category change but DON'T reset consecutive_same_category_failures here
        # The counter should only reset on SUCCESS, not just category change
        # This allows the force-switch logic (at 3 failures) to work properly
        if last_tool_category and last_tool_category != tool_category:
            print(f"[DEBUG] Tool category changed from '{last_tool_category}' to '{tool_category}'", file=sys.stderr)
        last_tool_category = tool_category
        
        # Check if this tool produced useful code or had errors
        has_errors = False
        if isinstance(tool_result, dict):
            # Check for execution failure (ok=False or has error)
            if tool_result.get("ok") is False:
                has_errors = True
                # Capture the failed code and error for next tool
                if tool_result.get("error"):
                    error_obj = tool_result["error"]
                    if isinstance(error_obj, dict):
                        last_error = f"{error_obj.get('code', 'Error')}: {error_obj.get('message', 'Unknown error')}"
                    else:
                        last_error = str(error_obj)
            elif tool_result.get("error"):
                # Has error even with ok=True
                has_errors = True
                error_obj = tool_result["error"]
                if isinstance(error_obj, dict):
                    last_error = f"{error_obj.get('code', 'Error')}: {error_obj.get('message', 'Unknown error')}"
                else:
                    last_error = str(error_obj)
            elif tool_result.get("ok") and "output" in tool_result:
                # Tool executed successfully, check if code is useful
                output = tool_result["output"]
                if isinstance(output, dict):
                    # Handle nested output structure: output.output.code
                    inner_output = None
                    success_flag = True  # Default to true if not specified
                    
                    if "output" in output and isinstance(output["output"], dict):
                        inner_output = output["output"]
                        code = inner_output.get("code_or_commands") or inner_output.get("code") or inner_output.get("solution")
                        # Check success flag in nested structure
                        if "success" in inner_output:
                            success_flag = bool(inner_output.get("success"))
                    else:
                        code = output.get("code_or_commands") or output.get("code") or output.get("solution")
                        # Check success flag in flat structure
                        if "success" in output:
                            success_flag = bool(output.get("success"))
                    
                    # Validate code: must exist, be string, not be error message, and have success=true
                    if code and isinstance(code, str) and success_flag:
                        code_length = len(code)
                        print(f"[DEBUG] Round {round_idx}: Extracted code (len={code_length}), success_flag={success_flag}, is_placeholder={_is_placeholder_code(code)}", file=sys.stderr)
                        
                        # Early stopping detection: check if code length hasn't changed
                        if code_length == last_code_length:
                            same_code_length_count += 1
                            if same_code_length_count >= 3:
                                print(f"[DEBUG] ⚠️ Early stop: Same code length ({code_length}) for {same_code_length_count} rounds, agent stuck!", file=sys.stderr)
                                has_errors = True  # Mark as having errors to stop loop
                                break
                        else:
                            same_code_length_count = 0
                            last_code_length = code_length
                        
                        # Check if code is actually an error message
                        if _is_likely_error_text(code):
                            has_errors = True
                            failed_code = None
                            last_error = code.strip()
                        elif not _is_placeholder_code(code):
                            # Got valid code, now test it if test context available
                            if task_context and task_context.get("test"):
                                print(f"[DEBUG] Running test for tool {tool_id}, round {round_idx}", file=sys.stderr)
                                
                                test_ok, test_error = _test_code(code, task_context)
                                
                                print(f"[DEBUG] Test result: ok={test_ok}, error={test_error[:100] if test_error else 'None'}", file=sys.stderr)
                                
                                if test_ok:
                                    # Test passed! Set success and stop immediately
                                    successful_code = code
                                    test_passed = True
                                    has_errors = False
                                    failed_code = None
                                    last_error = None
                                    last_test_error = None
                                    original_test_error = None
                                    latest_test_error = None
                                    postprocess_error = None
                                    role_context["test_passed"] = True
                                    role_context.pop("diagnostics", None)
                                    
                                    print(f"[DEBUG] Test PASSED! Stopping immediately at round {round_idx}", file=sys.stderr)
                                    
                                    # Record successful result and break immediately
                                    tool_history.append(
                                        {
                                            "tool_id": tool_id,
                                            "input": tool_input,
                                            "result": tool_result,
                                            "reason": decision.get("reason") or ("tool_only_forced" if forced else None),
                                        }
                                    )
                                    if tool_history:
                                        role_context["tool_history"] = tool_history
                                    return tool_history  # Stop immediately on test pass
                                else:
                                    # Test failed, treat as error and retry
                                    has_errors = True
                                    failed_code = code
                                    last_error = f"Test failed: {test_error}"
                                    last_test_error = test_error
                                    latest_test_error = test_error
                                    postprocess_error = None
                                    if original_test_error is None:
                                        original_test_error = test_error
                                    failure_summary = _summarize_failure(last_error, test_error)
                                    role_context["diagnostics"] = {
                                        "test_error": test_error,
                                        "error_message": last_error,
                                        "failed_code": code,
                                        "original_test_error": original_test_error,
                                        "latest_test_error": latest_test_error,
                                        "failure_summary": failure_summary,
                                    }
                                    # Track failure count for this tool and category
                                    tool_failure_counts[tool_id] = tool_failure_counts.get(tool_id, 0) + 1
                                    tool_category_counts[tool_category] = tool_category_counts.get(tool_category, 0) + 1
                                    consecutive_same_category_failures += 1
                                    
                                    print(f"[DEBUG] Test FAILED: {test_error[:200]}, retrying...", file=sys.stderr)
                                    print(f"[DEBUG] Tool category '{tool_category}' failures: {tool_category_counts[tool_category]}, consecutive: {consecutive_same_category_failures}", file=sys.stderr)
                                    
                                    repair_llm = _resolve_repair_llm(llm, tool_executor)
                                    post_ok, post_error, post_code = _postprocess_tool_failure(
                                        code,
                                        task_context,
                                        test_error,
                                        repair_llm,
                                        timeout_s=5.0,
                                    )
                                    if post_ok:
                                        successful_code = post_code
                                        test_passed = True
                                        has_errors = False
                                        failed_code = None
                                        last_error = None
                                        last_test_error = None
                                        original_test_error = None
                                        latest_test_error = None
                                        postprocess_error = None
                                        role_context["test_passed"] = True
                                        role_context.pop("diagnostics", None)
                                        
                                        print(
                                            f"[DEBUG] Postprocess+LLM repair fixed tool {tool_id} at round {round_idx}",
                                            file=sys.stderr,
                                        )
                                        
                                        tool_history.append(
                                            {
                                                "tool_id": tool_id,
                                                "input": tool_input,
                                                "result": tool_result,
                                                "reason": decision.get("reason") or ("tool_only_forced" if forced else None),
                                            }
                                        )
                                        if tool_history:
                                            role_context["tool_history"] = tool_history
                                        return tool_history
                                    else:
                                        failed_code = post_code or failed_code
                                        last_error = f"Postprocess failed: {post_error}"
                                        latest_test_error = post_error
                                        postprocess_error = post_error
                                        failure_summary = _summarize_failure(last_error, original_test_error or last_test_error)
                                        role_context["diagnostics"] = {
                                            "test_error": original_test_error or last_test_error or post_error,
                                            "error_message": last_error,
                                            "failed_code": failed_code,
                                            "original_test_error": original_test_error,
                                            "latest_test_error": latest_test_error,
                                            "postprocess_error": postprocess_error,
                                            "failure_summary": failure_summary,
                                        }
                            else:
                                # No test available, accept code based on success flag
                                # Mark as successful but continue to allow LLM to decide when to stop
                                print(f"[DEBUG] No test context available (task_context={bool(task_context)}, has_test={bool(task_context.get('test') if task_context else False)})", file=sys.stderr)
                                
                                successful_code = code
                                has_errors = False
                                failed_code = None
                                last_error = None
                                original_test_error = None
                                latest_test_error = None
                                postprocess_error = None
                                consecutive_same_category_failures = 0  # Reset on success
                        else:
                            # Placeholder code counts as "no useful result"
                            has_errors = True
                            failed_code = code  # Save for refinement
                            last_error = "Generated placeholder code (e.g., 'return None')"
                    else:
                        # No valid code or success=false
                        has_errors = True
                        if not success_flag and inner_output:
                            last_error = inner_output.get("error") or "Tool returned success=false"

        if has_errors:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        if has_errors and (last_error or failed_code or last_test_error):
            failure_summary = _summarize_failure(
                last_error,
                original_test_error or last_test_error,
            )
            diagnostics_payload = {
                "error_message": last_error,
                "test_error": original_test_error or last_test_error,
                "failed_code": failed_code,
                "original_test_error": original_test_error,
                "latest_test_error": latest_test_error,
                "postprocess_error": postprocess_error,
                "failure_summary": failure_summary,
            }
            role_context["diagnostics"] = {k: v for k, v in diagnostics_payload.items() if v}

        if tool_only and role.strip().lower() == "builder" and consecutive_failures >= 3:
            role_context["allow_llm_once"] = True
            role_context["allow_llm_reason"] = "consecutive tool failures"
            role_context["stop_tool_loop"] = True

        # Track last tool category for next iteration
        last_tool_category = tool_category
        
        # Track which categories we've tried
        if tool_category:
            tried_categories.add(tool_category)
        
        # Early stopping: if we've tried multiple categories and all failed
        if len(tried_categories) >= 3 and consecutive_same_category_failures >= 5:
            print(f"[DEBUG] ⚠️ Early stop: Tried {len(tried_categories)} categories, all failing. Agent exhausted!", file=sys.stderr)
            break
        
        # Clear force flags after each round (they should only affect one round)
        role_context.pop("force_different_category", None)
        role_context.pop("force_different_tool", None)
        role_context.pop("suggested_tools", None)
        role_context.pop("reason", None)
        role_context.pop("failed_category", None)
        role_context.pop("failed_tool", None)
        
        # Only append to history if we didn't already return (test didn't pass)
        if not (test_passed and successful_code):
            tool_history.append(
                {
                    "tool_id": tool_id,
                    "input": tool_input,
                    "result": tool_result,
                    "reason": decision.get("reason") or ("tool_only_forced" if forced else None),
                }
            )

        if role_context.get("stop_tool_loop"):
            break

        # Stop conditions:
        # 1. Got successful code AND no errors AND not forced to use all tools
        # 2. OR reached max rounds (removed the "tried all tools" condition to allow refinement)
        if successful_code and not has_errors and not tool_only:
            break
        # Note: Removed "if len(used_tool_ids) >= len(tool_ids): break" to allow tools
        # to be retried for refinement. The loop will continue up to tool_max_rounds.
    if tool_history:
        role_context["tool_history"] = tool_history
    return tool_history


def _extract_tool_output(tool_history: List[Dict[str, Any]]) -> Optional[Any]:
    for entry in reversed(tool_history):
        result = entry.get("result")
        if isinstance(result, dict) and result.get("ok") and "output" in result:
            return result.get("output")
    return None


def _is_placeholder_code(code: str) -> bool:
    """Check if the code is a placeholder like 'return None' or empty."""
    if not code or not isinstance(code, str):
        return True
    normalized = code.strip().lower()
    # Empty or trivial code
    if not normalized or normalized == "pass":
        return True
    # Common placeholder patterns
    placeholder_patterns = [
        "return none",
        "return null",
        "# todo",
        "# placeholder",
        "not implemented",
        "raise notimplementederror",
    ]
    for pattern in placeholder_patterns:
        if pattern in normalized:
            return True
    # Very short code that's likely just a placeholder
    if len(normalized.split('\n')) == 1 and len(normalized) < 20:
        if normalized.startswith('return') or normalized == '...':
            return True
    return False


def _apply_stop_tokens(text: str, stop_tokens: List[str]) -> str:
    if not text or not stop_tokens:
        return text
    stop_pos = None
    for token in stop_tokens:
        if not token:
            continue
        idx = text.find(token)
        if idx == -1:
            continue
        if stop_pos is None or idx < stop_pos:
            stop_pos = idx
    if stop_pos is None:
        return text
    return text[:stop_pos]


def _unwrap_json_code(text: str) -> str:
    if not text or "{" not in text:
        return text
    try:
        start_idx = text.find("{")
        if start_idx == -1:
            return text
        json_text = text[start_idx:]
        data = json.loads(json_text)
        if isinstance(data, dict):
            for key in ["code_or_commands", "code", "solution", "output"]:
                if key in data:
                    value = data[key]
                    if isinstance(value, str) and value.strip():
                        return value
                    if isinstance(value, dict):
                        for nested_key in ["code_or_commands", "code", "solution"]:
                            if nested_key in value and isinstance(value[nested_key], str):
                                return value[nested_key]
    except Exception:
        return text
    return text


def _extract_code_block(text: str) -> str:
    if not text:
        return text
    text = _unwrap_json_code(text)
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) < 3:
        return text
    code = parts[1]
    lines = code.splitlines()
    if lines and lines[0].strip().lower() in {"python", "py", "json"}:
        lines = lines[1:]
    return "\n".join(lines).strip() or text


def _strip_redundant_def(text: str, entry_point: Optional[str]) -> str:
    if not entry_point or not text:
        return text
    target = f"def {entry_point}"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(target):
            body_lines = lines[idx + 1 :]
            if not body_lines:
                return text
            return "\n".join(body_lines).lstrip("\n")
    return text


def _normalize_completion(text: str, entry_point: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\\n", "\n")
    if not entry_point:
        return text
    lines = text.lstrip("\n").splitlines()
    if not lines:
        return text
    if lines[0].lstrip().startswith(("def ", "class ", "@")):
        return "\n".join(lines)
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    min_pos_indent = min((i for i in indents if i > 0), default=0)
    if min_pos_indent > 0:
        new_lines = []
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
            leading = len(line) - len(line.lstrip())
            trim = min(leading, min_pos_indent)
            new_lines.append(line[trim:])
        lines = new_lines
    lines = [("    " + line) if line.strip() else line for line in lines]
    return "\n".join(lines)


def _build_program(
    prompt: str, completion: str, test_code: str, entry_point: Optional[str] = None
) -> str:
    prompt_text = prompt or ""
    completion_text = completion or ""
    completion_text = _extract_code_block(completion_text)
    completion_text = _strip_redundant_def(completion_text, entry_point)
    completion_text = _normalize_completion(completion_text, entry_point)
    if not completion_text.endswith("\n"):
        completion_text += "\n"
    return f"{prompt_text}{completion_text}\n{test_code}\n"


def _looks_like_syntax_error(message: str) -> bool:
    if not message:
        return False
    return any(token in message for token in ("SyntaxError", "IndentationError", "TabError"))


def _repair_completion(completion: str, entry_point: Optional[str]) -> str:
    text = str(completion or "")
    text = text.replace("\r\n", "\n")
    text = _extract_code_block(text)
    text = text.expandtabs(4)
    text = _strip_redundant_def(text, entry_point)
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    text = "\n".join(lines)
    text = textwrap.dedent(text).strip("\n")
    if entry_point:
        lines = text.splitlines()
        lines = [("    " + line) if line.strip() else line for line in lines]
        text = "\n".join(lines)
    return text


def _run_python(code: str, timeout_s: float) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "eval_task.py")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return False, f"File write error: {str(e)}"
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "Test execution timeout"
        except Exception as e:
            return False, f"Test execution error: {str(e)}"
    if result.returncode == 0:
        return True, ""
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    message = stderr or stdout or f"exit_code={result.returncode}"
    return False, message


def _run_with_repair(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> Tuple[bool, str]:
    code = _build_program(prompt, str(completion), test_code, entry_point)
    ok, message = _run_python(code, timeout_s=timeout_s)
    if ok or not _looks_like_syntax_error(message):
        return ok, message
    repaired = _repair_completion(str(completion), entry_point)
    if not repaired or repaired == str(completion):
        return ok, message
    repaired_code = _build_program(prompt, repaired, test_code, entry_point)
    ok2, message2 = _run_python(repaired_code, timeout_s=timeout_s)
    return ok2, ("" if ok2 else message2)


def _test_code(code: str, task_context: Optional[Dict[str, Any]], timeout_s: float = 5.0) -> Tuple[bool, str]:
    """Test generated code if task_context contains test information.
    
    Args:
        code: Generated code to test
        task_context: Context containing prompt, test, entry_point
        timeout_s: Timeout for test execution
        
    Returns:
        (test_passed, error_message)
    """
    if not task_context:
        return True, ""  # No test available, assume pass
    
    prompt = task_context.get("prompt", "")
    test_code = task_context.get("test", "")
    entry_point = task_context.get("entry_point")
    stop_tokens = task_context.get("stop_tokens") or []
    
    if not test_code:
        return True, ""  # No test available
    
    completion = str(code)
    if stop_tokens:
        completion = _apply_stop_tokens(_extract_code_block(completion), stop_tokens)
    ok, message = _run_with_repair(prompt, completion, test_code, entry_point, timeout_s=timeout_s)
    return ok, message


def _resolve_repair_llm(llm: Any, tool_executor: Optional[ToolExecutor]) -> Optional[Any]:
    if tool_executor is not None:
        tool_llm = getattr(tool_executor, "llm", None)
        if tool_llm is not None:
            return tool_llm
    if hasattr(llm, "llm"):
        inner = getattr(llm, "llm", None)
        if inner is not None:
            return inner
    if hasattr(llm, "chat"):
        return llm
    return None


def _postprocess_tool_failure(
    code: str,
    task_context: Optional[Dict[str, Any]],
    error_message: str,
    llm_client: Optional[Any],
    timeout_s: float = 5.0,
) -> Tuple[bool, str, str]:
    if not task_context:
        return False, error_message, code

    prompt = task_context.get("prompt", "")
    test_code = task_context.get("test", "")
    entry_point = task_context.get("entry_point")
    stop_tokens = task_context.get("stop_tokens") or []

    if not test_code:
        return False, error_message, code

    candidate = str(code or "")
    try:
        from evaluation import humaneval_postprocess as _hp

        param_names = _hp._extract_param_names_from_prompt(prompt, entry_point)
        repaired_text, _ = _hp._run_postprocess_tool_agent(
            completion=candidate,
            prompt=prompt,
            entry_point=entry_point,
            stop_tokens=stop_tokens,
            use_stop_tokens=bool(stop_tokens),
            param_names=param_names,
            enable_syntax_repair=True,
            error_message=error_message,
        )
        if repaired_text and str(repaired_text).strip():
            candidate = str(repaired_text)
        candidate = _hp._normalize_completion(
            _hp._strip_redundant_def(_hp._extract_code_block(candidate), entry_point),
            entry_point,
        )
        candidate = _hp._fix_missing_indents(candidate)
        if llm_client is not None:
            repaired_llm = _hp._repair_assertion_completion_with_llm(
                llm_client=llm_client,
                prompt=prompt,
                completion=candidate,
                test_code=test_code,
                entry_point=entry_point,
                error_message=error_message,
            )
            if repaired_llm and str(repaired_llm).strip():
                candidate = repaired_llm
    except Exception:
        pass

    ok, message = _run_with_repair(prompt, candidate, test_code, entry_point, timeout_s=timeout_s)
    return ok, message, candidate


def _summarize_tool_history(tool_history: List[Dict[str, Any]]) -> List[str]:
    summaries: List[str] = []
    for entry in tool_history:
        tool_id = entry.get("tool_id") or "tool"
        reason = entry.get("reason")
        result = entry.get("result")
        ok = result.get("ok") if isinstance(result, dict) else None
        if reason:
            summaries.append(f"{tool_id} ok={ok} reason={reason}")
        else:
            summaries.append(f"{tool_id} ok={ok}")
    return summaries


def _tool_history_to_output(role: str, tool_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not tool_history:
        return {"error": "no_tool_used"}
    payload = _extract_tool_output(tool_history)
    if payload is None:
        payload_text = ""
    elif isinstance(payload, str):
        payload_text = payload
    elif isinstance(payload, dict):
        payload_text = ""
        for key in ["code_or_commands", "code", "solution", "output", "completion"]:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                payload_text = value
                break
        if not payload_text:
            payload_text = json.dumps(payload, ensure_ascii=True, default=str)
    else:
        payload_text = json.dumps(payload, ensure_ascii=True, default=str)
    summary = _summarize_tool_history(tool_history)
    role_key = _normalize_role_name(role)
    if role_key == "builder":
        return {"runnable_plan": summary, "code_or_commands": payload_text, "self_test": []}
    if role_key == "refactor":
        return {"runnable_plan": summary, "code_or_commands": payload_text, "self_test": []}
    if role_key == "planner":
        return {"steps": summary, "acceptance_criteria": [], "code_or_commands": payload_text}
    if role_key == "researcher":
        return {"search_queries": [], "sources": [], "evidence_points": summary, "code_or_commands": payload_text}
    if role_key == "checker":
        return {"test_cases": [], "verdicts": [], "failure_localization": payload_text or "tool_only"}
    if role_key == "manager":
        return {
            "status": "finish",
            "next_role": None,
            "instruction": None,
            "summary": payload_text or "tool_only",
            "notes": summary,
            "steps": summary if isinstance(summary, list) else [summary] if summary else [],
            "code_or_commands": payload_text,  # Add code_or_commands for extraction
        }
    return {"output": payload_text}


def _select_agent_for_role(
    *,
    task_text: str,
    role: str,
    constraints: Optional[Dict[str, Any]],
    registry,
    index,
    embedder,
    reranker: TfidfLinearReranker,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    router_llm_client: Optional[LLMClient],
    router_top_m: int,
    router_no_rerank: bool,
    reuse_cache: bool,
    cache: Dict[str, Dict[str, Any]],
    cache_key: Optional[str],
    selected_agent_ids: Optional[List[str]],
    team_lambda_compat: float,
    team_mu_cost: float,
    team_nu_latency: float,
) -> Dict[str, Any]:
    constraints = _ensure_role_constraints(constraints, role)
    effective_cache_key = cache_key or role
    if reuse_cache and effective_cache_key in cache:
        cached = cache[effective_cache_key]
        return dict(cached)

    candidates, router_candidates = _prepare_role_candidates(
        task_text=task_text,
        role=role,
        constraints=constraints,
        registry=registry,
        index=index,
        embedder=embedder,
        reranker=reranker,
        top_n=top_n,
        top_k=top_k,
        rerank_top_m=rerank_top_m,
        mmr_lambda=mmr_lambda,
        router_no_rerank=router_no_rerank,
        selected_agent_ids=selected_agent_ids,
        team_lambda_compat=team_lambda_compat,
        team_mu_cost=team_mu_cost,
        team_nu_latency=team_nu_latency,
    )

    selected_main = None
    selected_shadows: List[str] = []
    selected_shadow = None

    if router_llm_client and router_candidates:
        trimmed = router_candidates[: max(1, router_top_m)]
        selected_main = _select_with_router_llm(
            router_llm_client,
            task_text,
            role,
            constraints,
            trimmed,
            registry,
        )
        if selected_main:
            for cand in trimmed:
                cand_id = cand.get("card_id")
                if not cand_id or cand_id == selected_main:
                    continue
                selected_shadows.append(cand_id)
                if len(selected_shadows) >= 2:
                    break
            selected_shadow = selected_shadows[0] if selected_shadows else None

    if selected_main is None:
        probe_result = probe_commit(
            task_text=task_text,
            role=role,
            constraints=constraints,
            candidates=router_candidates,
            top_probe=min(5, len(router_candidates)),
            max_shadows=2,
        )
        selected_main = probe_result.get("selected_main")
        selected_shadows = probe_result.get("selected_shadows", [])
        if selected_main is None and router_candidates:
            selected_main = router_candidates[0]["card_id"]
        selected_shadow = selected_shadows[0] if selected_shadows else None

    selection = {
        "selected_main": selected_main,
        "selected_shadows": selected_shadows,
        "selected_shadow": selected_shadow,
        "candidates": candidates,
        "reranked": router_candidates,
        "base_reranked": list(router_candidates),
    }
    cache[effective_cache_key] = dict(selection)
    return selection


def _execute_role_with_selection(
    *,
    role: str,
    task_text: str,
    selection: Dict[str, Any],
    registry,
    llm: Any,
    tool_executor: Optional[ToolExecutor],
    max_attempts: int,
    results: Dict[str, Any],
    constraints: Optional[Dict[str, Any]],
    allow_unknown_roles: bool,
    runs_dir: str,
    workflow_version: str,
    task_id: str,
    meta: Optional[Dict[str, Any]] = None,
    extra_context: Optional[Dict[str, Any]] = None,
    tool_max_rounds: int = 10,
    tool_only: bool = False,
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected_main = selection.get("selected_main")
    selected_shadow = selection.get("selected_shadow")
    reranked = selection.get("reranked") or []
    
    if not selected_main:
        print(f"[ERROR] No agent selected for role '{role}'! selection={selection}", file=sys.stderr)

    attempt = 0
    role_key = role.strip().lower()
    role_context: Dict[str, Any] = {"upstream": results, "constraints": constraints}
    agent_context = _get_agent_context(registry, selected_main)
    role_context["agent"] = agent_context
    if extra_context:
        role_context.update(extra_context)

    tool_history = _plan_and_run_tools(
        llm=llm,
        role=role,
        task_text=task_text,
        role_context=role_context,
        agent_context=agent_context,
        registry=registry,
        tool_executor=tool_executor,
        results=results,
        tool_max_rounds=tool_max_rounds,
        tool_only=tool_only,
        task_context=task_context,
    )

    test_passed = bool(role_context.get("test_passed"))

    if test_passed:
        tool_output = _tool_history_to_output(role, tool_history)
        if isinstance(tool_output, dict) and tool_output.get("error"):
            validation = {"ok": False, "errors": [str(tool_output.get("error"))]}
            stored_output = tool_output
        else:
            valid, errors, model = validate_output(role, tool_output, allow_unknown=allow_unknown_roles)
            validation = {"ok": valid, "errors": errors}
            if valid and model is not None:
                stored_output = model.model_dump()
            else:
                stored_output = tool_output
        if isinstance(stored_output, dict):
            stored_output["test_passed"] = True
        log_path = write_event(
            task_id=task_id,
            workflow_version=workflow_version,
            role=role,
            selected_main=selected_main,
            selected_shadow=selected_shadow,
            candidates_topk=reranked,
            output=tool_output,
            validation=validation,
            executor_result=tool_history or None,
            failure_type=None,
            action="tool_test_passed",
            meta=meta,
            runs_dir=runs_dir,
        )
        return {
            "output": stored_output,
            "raw_output": tool_output,
            "executor_result": tool_history or None,
            "log_path": log_path,
            "selected_main": selected_main,
            "selected_shadows": selection.get("selected_shadows", []),
        }

    if tool_only:
        if role_key == "builder" and role_context.get("allow_llm_once"):
            role_context.pop("allow_llm_once", None)
            llm_task_text = _append_diagnostics_to_task(task_text, role_context)
            output = llm.generate(role, llm_task_text, role_context)
            valid, errors, model = validate_output(role, output, allow_unknown=allow_unknown_roles)
            if valid:
                if model is not None:
                    stored_output = model.model_dump()
                elif isinstance(output, dict):
                    stored_output = output
                else:
                    stored_output = {"raw": output}
                if task_context:
                    code_text = stored_output.get("code_or_commands") if isinstance(stored_output, dict) else ""
                    if isinstance(code_text, str) and code_text.strip():
                        test_ok, test_error = _test_code(code_text, task_context)
                        if test_ok:
                            stored_output["test_passed"] = True
                            log_path = write_event(
                                task_id=task_id,
                                workflow_version=workflow_version,
                                role=role,
                                selected_main=selected_main,
                                selected_shadow=selected_shadow,
                                candidates_topk=reranked,
                                output=output,
                                validation={"ok": True, "errors": []},
                                executor_result=tool_history or None,
                                failure_type=None,
                                action="tool_only_llm_fallback",
                                meta=meta,
                                runs_dir=runs_dir,
                            )
                            return {
                                "output": stored_output,
                                "raw_output": output,
                                "executor_result": tool_history or None,
                                "log_path": log_path,
                                "selected_main": selected_main,
                                "selected_shadows": selection.get("selected_shadows", []),
                            }
                        role_context["diagnostics"] = {
                            "test_error": test_error,
                            "error_message": f"Test failed: {test_error}",
                            "failed_code": code_text,
                        }
                else:
                    code_text = stored_output.get("code_or_commands") if isinstance(stored_output, dict) else ""
                    if isinstance(code_text, str) and code_text.strip():
                        log_path = write_event(
                            task_id=task_id,
                            workflow_version=workflow_version,
                            role=role,
                            selected_main=selected_main,
                            selected_shadow=selected_shadow,
                            candidates_topk=reranked,
                            output=output,
                            validation={"ok": True, "errors": []},
                            executor_result=tool_history or None,
                            failure_type=None,
                            action="tool_only_llm_fallback",
                            meta=meta,
                            runs_dir=runs_dir,
                        )
                        return {
                            "output": stored_output,
                            "raw_output": output,
                            "executor_result": tool_history or None,
                            "log_path": log_path,
                            "selected_main": selected_main,
                            "selected_shadows": selection.get("selected_shadows", []),
                        }
        tool_output = _tool_history_to_output(role, tool_history)
        if isinstance(tool_output, dict) and tool_output.get("error"):
            validation = {"ok": False, "errors": [str(tool_output.get("error"))]}
            stored_output = tool_output
        else:
            valid, errors, model = validate_output(role, tool_output, allow_unknown=allow_unknown_roles)
            validation = {"ok": valid, "errors": errors}
            if valid and model is not None:
                stored_output = model.model_dump()
            else:
                stored_output = tool_output
        log_path = write_event(
            task_id=task_id,
            workflow_version=workflow_version,
            role=role,
            selected_main=selected_main,
            selected_shadow=selected_shadow,
            candidates_topk=reranked,
            output=tool_output,
            validation=validation,
            executor_result=tool_history or None,
            failure_type="tool_only",
            action="tool_only",
            meta=meta,
            runs_dir=runs_dir,
        )
        return {
            "output": stored_output,
            "raw_output": tool_output,
            "executor_result": tool_history or None,
            "log_path": log_path,
            "selected_main": selected_main,
            "selected_shadows": selection.get("selected_shadows", []),
        }

    while attempt < max_attempts:
        llm_task_text = _append_diagnostics_to_task(task_text, role_context)
        output = llm.generate(role, llm_task_text, role_context)
        valid, errors, model = validate_output(role, output, allow_unknown=allow_unknown_roles)
        validation = {"ok": valid, "errors": errors}
        executor_result = tool_history or None
        failure_type = None
        action = None

        if valid:
            if model is not None:
                stored_output = model.model_dump()
            elif isinstance(output, dict):
                stored_output = output
            else:
                stored_output = {"raw": output}

            log_path = write_event(
                task_id=task_id,
                workflow_version=workflow_version,
                role=role,
                selected_main=selected_main,
                selected_shadow=selected_shadow,
                candidates_topk=reranked,
                output=output,
                validation=validation,
                executor_result=executor_result,
                failure_type=failure_type,
                action=action,
                meta=meta,
                runs_dir=runs_dir,
            )
            return {
                "output": stored_output,
                "raw_output": output,
                "executor_result": executor_result,
                "log_path": log_path,
                "selected_main": selected_main,
                "selected_shadows": selection.get("selected_shadows", []),
            }

        route = route_failure(role, output, errors, executor_result)
        failure_type = route["failure_type"]
        action = route["action"]

        if failure_type == "A_contract":
            repaired = llm.rewrite_format(role, output, role_context)
            valid_fix, errors_fix, model_fix = validate_output(role, repaired, allow_unknown=allow_unknown_roles)
            validation = {"ok": valid_fix, "errors": errors_fix}
            if valid_fix:
                stored_output = model_fix.model_dump() if model_fix is not None else repaired
                log_path = write_event(
                    task_id=task_id,
                    workflow_version=workflow_version,
                    role=role,
                    selected_main=selected_main,
                    selected_shadow=selected_shadow,
                    candidates_topk=reranked,
                    output=repaired,
                    validation=validation,
                    executor_result=executor_result,
                    failure_type=failure_type,
                    action=action,
                    meta=meta,
                    runs_dir=runs_dir,
                )
                return {
                    "output": stored_output,
                    "raw_output": repaired,
                    "executor_result": executor_result,
                    "log_path": log_path,
                    "selected_main": selected_main,
                    "selected_shadows": selection.get("selected_shadows", []),
                }
            log_path = write_event(
                task_id=task_id,
                workflow_version=workflow_version,
                role=role,
                selected_main=selected_main,
                selected_shadow=selected_shadow,
                candidates_topk=reranked,
                output=repaired,
                validation=validation,
                executor_result=executor_result,
                failure_type=failure_type,
                action=action,
                meta=meta,
                runs_dir=runs_dir,
            )
            attempt += 1
            continue

        if failure_type == "B_missing_info":
            role_context.update(llm.clarify(role, task_text))
            log_path = write_event(
                task_id=task_id,
                workflow_version=workflow_version,
                role=role,
                selected_main=selected_main,
                selected_shadow=selected_shadow,
                candidates_topk=reranked,
                output=output,
                validation=validation,
                executor_result=executor_result,
                failure_type=failure_type,
                action=action,
                meta=meta,
                runs_dir=runs_dir,
            )
            attempt += 1
            continue

        if failure_type == "C_capability":
            log_path = write_event(
                task_id=task_id,
                workflow_version=workflow_version,
                role=role,
                selected_main=selected_main,
                selected_shadow=selected_shadow,
                candidates_topk=reranked,
                output=output,
                validation=validation,
                executor_result=executor_result,
                failure_type=failure_type,
                action=action,
                meta=meta,
                runs_dir=runs_dir,
            )
            if selected_shadow and selected_shadow != selected_main:
                selected_main = selected_shadow
                selected_shadow = None
                agent_context = _get_agent_context(registry, selected_main)
                role_context["agent"] = agent_context
                attempt += 1
                continue
            break

        attempt += 1

    return {"output": {"error": "no_valid_output"}, "raw_output": None, "executor_result": None, "log_path": None}


def _heuristic_topology(task_text: str) -> TopologyType:
    text = task_text.lower()
    if any(token in text for token in ["brainstorm", "multi-angle", "debate", "discussion", "对比", "多角度", "讨论"]):
        return TopologyType.DECENTRALIZED
    if any(token in text for token in ["then", "first", "next", "after", "并且", "然后", "最后", "步骤", "流程"]):
        return TopologyType.CENTRALIZED
    # Check for code generation tasks
    if any(token in text for token in ["def ", "function", "implement", "write code", "算法", "函数", "实现", "代码"]):
        return TopologyType.CENTRALIZED
    return TopologyType.SINGLE


def _plan_topology(
    *,
    task_text: str,
    roles: List[str],
    topology: Optional[str],
    topology_config: Optional[Dict[str, Any]],
    meta_router_llm_client: Optional[LLMClient],
    max_steps: int,
    strict_roles: bool = False,
) -> TopologyConfig:
    if topology_config:
        return TopologyConfig(**topology_config).normalized(default_roles=roles)

    topology_value = (topology or "").strip().lower()
    if topology_value and topology_value not in {"auto", "dynamic"}:
        try:
            topo = TopologyType(topology_value)
        except ValueError:
            topo = _heuristic_topology(task_text)
        return TopologyConfig(topology=topo, roles=roles, max_steps=max_steps).normalized(default_roles=roles)

    if meta_router_llm_client:
        system_msg = (
            "You are a meta-router. Decide the best agent topology for the task. "
            "Return ONLY JSON with keys: topology, roles, manager_role, entry_role, max_steps, flow_type."
        )
        user_msg = (
            f"Task: {task_text}\n"
            f"Available roles: {json.dumps(roles, ensure_ascii=True)}\n"
            "Topology options: single, centralized, decentralized, chain.\n"
            "IMPORTANT: For code generation tasks, prefer 'centralized' topology to leverage multiple specialized roles (planner, builder, checker, refactor) for better code quality.\n"
            "Use 'single' only for extremely simple tasks that clearly need just one role.\n"
        )
        response = meta_router_llm_client.chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        try:
            os.makedirs("runs", exist_ok=True)
            debug_path = os.path.join("runs", "meta_router_raw.jsonl")
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "task_text": task_text,
                            "available_roles": roles,
                            "raw_response": response,
                        },
                        ensure_ascii=True,
                        default=str,
                    )
                    + "\n"
                )
        except Exception:
            pass
        data = _safe_load_json(str(response), {})
        if isinstance(data, dict) and data:
            # Ensure max_steps is a valid integer (not None or missing)
            if "max_steps" not in data or data["max_steps"] is None:
                data["max_steps"] = max_steps
            else:
                try:
                    llm_steps = int(data["max_steps"])
                except Exception:
                    llm_steps = max_steps
                # Never allow meta-router to exceed CLI max_steps budget.
                llm_steps = max(1, min(llm_steps, int(max_steps)))
                data["max_steps"] = llm_steps
            
            # UPDATED: Trust LoRA's role selection, but validate roles are in available list
            # If LoRA returns roles, validate them against available roles list
            if "roles" in data and isinstance(data["roles"], list):
                llm_roles = [r.lower().strip() for r in data["roles"] if r]
                # Filter to only roles that exist in our available roles
                valid_roles = [r for r in llm_roles if r in roles]
                # If LoRA returned valid roles, use them; otherwise fallback to user-specified
                if valid_roles:
                    data["roles"] = valid_roles
                else:
                    data["roles"] = roles  # Fallback if LoRA returned invalid roles
            else:
                data["roles"] = roles  # Use user-specified if LoRA didn't return roles
            
            # If strict_roles=True and LLM set entry_role to something not in our roles list, clear it
            if strict_roles and "entry_role" in data and data["entry_role"] not in data["roles"]:
                del data["entry_role"]  # Let normalized() pick from roles list
            return TopologyConfig(**data).normalized(default_roles=roles)

    return TopologyConfig(topology=_heuristic_topology(task_text), roles=roles, max_steps=max_steps).normalized(
        default_roles=roles
    )


def _select_next_role(
    *,
    task_text: str,
    available_roles: List[str],
    history: List[Dict[str, Any]],
    llm: Optional[LLMClient],
    avoid_role: Optional[str] = None,
    force_role: Optional[str] = None,
) -> Optional[str]:
    if not available_roles:
        return None
    # If force_role is specified, skip LLM and directly return it
    if force_role:
        if force_role in available_roles:
            print(f"[DEBUG] Force using role: {force_role}", file=sys.stderr)
            return force_role
        else:
            print(f"[WARNING] force_role '{force_role}' not in available_roles {available_roles}, falling back", file=sys.stderr)
    if history and any(item.get("test_passed") or item.get("status") == "success" for item in history):
        print("[DEBUG] Task already completed, stopping role selection", file=sys.stderr)
        return None
    if llm:
        recent = history[-3:] if history else []
        recent_summary = []
        for item in recent:
            if not item.get("summary"):
                continue
            row: Dict[str, Any] = {"role": item.get("role"), "summary": item.get("summary")}
            if "planner_ready_to_handoff" in item:
                row["planner_ready_to_handoff"] = bool(item.get("planner_ready_to_handoff"))
            if item.get("planner_next_role"):
                row["planner_next_role"] = item.get("planner_next_role")
            recent_summary.append(row)
        # Check if task is already completed (test passed in recent history)
        task_completed = any(
            item.get("test_passed") or item.get("status") == "success" 
            for item in recent
        )
        system_msg = (
            "You are a dynamic router. Choose the next role from the list to continue working on the task. "
            "ONLY respond with 'finish' if the task is already completed successfully. "
            "If the task is not completed, you MUST choose a role from the available list to try a different approach. "
            "Return ONLY JSON: {\"next_role\": \"...\"}."
        )
        user_msg = (
            f"Task: {task_text}\n"
            f"Task completed: {task_completed}\n"
            f"Available roles: {json.dumps(available_roles, ensure_ascii=True)}\n"
            f"Recent history: {json.dumps(recent_summary, ensure_ascii=True)}\n"
        )
        response = llm.chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        data = _safe_load_json(str(response), {})
        candidate = data.get("next_role")
        print(f"[DEBUG] LLM returned next_role='{candidate}' (available: {available_roles})", file=sys.stderr)
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate.lower() in ("finish", "none"):
                return None
            if candidate in available_roles:
                if avoid_role and candidate == avoid_role and len(available_roles) > 1:
                    for role in available_roles:
                        if role != avoid_role:
                            return role
                return candidate
            else:
                # LLM returned invalid role, warn and fall back
                print(f"[WARNING] LLM returned invalid role '{candidate}', not in {available_roles}. Using fallback.", file=sys.stderr)

    text = task_text.lower()
    role_priority = [
        ("researcher", ["research", "search", "find", "查", "搜索", "调研", "资料"]),
        ("planner", ["plan", "planning", "拆解", "规划", "路线"]),
        ("builder", ["build", "implement", "code", "开发", "实现", "写代码", "生成"]),
        ("refactor", ["refactor", "refractor", "polish", "cleanup", "重构", "优化代码"]),
        ("checker", ["test", "verify", "check", "验证", "检查", "评估"]),
    ]
    for role_name, keywords in role_priority:
        if role_name in available_roles and any(token in text for token in keywords):
            if avoid_role and role_name == avoid_role and len(available_roles) > 1:
                continue
            return role_name
    for role in available_roles:
        if role != avoid_role:
            return role
    return available_roles[0]


def _build_route_query(
    *,
    task_text: str,
    history: List[Dict[str, Any]],
    last_output: Optional[Any],
    llm: Optional[LLMClient],
    fallback_max_chars: int,
) -> str:
    summary = _summarize_output(last_output, max_chars=fallback_max_chars)
    if llm:
        recent = history[-3:] if history else []
        system_msg = (
            "You are a router planner. Produce a short capability-oriented query for the next agent. "
            "Return ONLY JSON: {\"route_query\": \"...\"}."
        )
        user_msg = (
            f"Task: {task_text}\n"
            f"Last output summary: {summary}\n"
            f"Recent history: {json.dumps(recent, ensure_ascii=True)}\n"
        )
        response = llm.chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        data = _safe_load_json(str(response), {})
        candidate = data.get("route_query")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return summary or task_text


def _neighbors_for_role(role: str, config: TopologyConfig) -> List[str]:
    if role in config.routing_table:
        return list(config.routing_table.get(role) or [])
    if config.edges:
        return [dst for src, dst in config.edges if src == role]
    return []


def _checker_ok(results: Dict[str, Any]) -> bool:
    checker_output = results.get("checker")
    if checker_output is None:
        return True
    if isinstance(checker_output, list):
        return all(not (isinstance(item, dict) and item.get("error")) for item in checker_output)
    if isinstance(checker_output, dict) and checker_output.get("error"):
        return False
    return True


def _run_dynamic_workflow(
    *,
    task_text: str,
    role_list: List[str],
    constraints_per_role: Dict[str, Dict[str, Any]],
    workflow_version: str,
    registry,
    index,
    embedder,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    max_attempts: int,
    reranker: TfidfLinearReranker,
    execute_tools: bool,
    tool_timeout_s: float,
    auto_fill_tool_inputs: bool,
    tool_llm: Optional[Any],
    auto_install_common_libs: bool,
    auto_install_timeout_s: float,
    auto_install_user: bool,
    llm: Any,
    router_llm_client: Optional[LLMClient],
    router_top_m: int,
    router_no_rerank: bool,
    runs_dir: str,
    task_id: str,
    topology: Optional[str],
    topology_config: Optional[Dict[str, Any]],
    meta_router_llm_client: Optional[LLMClient],
    next_role_llm_client: Optional[LLMClient],
    route_query_llm_client: Optional[LLMClient],
    max_steps: int,
    force_final_builder_extra_step: bool,
    allow_unknown_roles: bool,
    reuse_role_selection: bool,
    reuse_same_role_agent_once: bool,
    summary_max_chars: int,
    soft_connection: bool,
    tool_only: bool,
    team_lambda_compat: float,
    team_mu_cost: float,
    team_nu_latency: float,
    mcts_dynamic_optimization: bool,
    mcts_iterations: int,
    mcts_rollout_depth: int,
    mcts_exploration: float,
    mcts_discount: float,
    mcts_max_candidates: int,
    pair_success_priors: Optional[Dict[Tuple[str, str, str, str], float]],
    task_context: Optional[Dict[str, Any]] = None,
    force_role: Optional[str] = None,
    strict_roles: bool = False,
) -> Dict[str, Any]:
    config = _plan_topology(
        task_text=task_text,
        roles=role_list,
        topology=topology,
        topology_config=topology_config,
        meta_router_llm_client=meta_router_llm_client,
        max_steps=max_steps,
        strict_roles=strict_roles,
    )

    results: Dict[str, Any] = {}
    tool_exec: Dict[str, Any] = {}
    log_path: Optional[str] = None
    selection_log: List[Dict[str, Any]] = []
    selection_cache: Dict[str, Dict[str, Any]] = {}
    role_once_selection_cache: Dict[str, Dict[str, Any]] = {}
    history: List[Dict[str, Any]] = []
    selected_agent_ids: List[str] = []
    selected_agent_roles: List[str] = []
    workflow_succeeded = False
    forced_final_builder_executed = False

    tool_executor = (
        ToolExecutor(
            registry,
            timeout_s=tool_timeout_s,
            auto_fill=auto_fill_tool_inputs,
            llm=tool_llm,
            auto_install_common_libs=auto_install_common_libs,
            auto_install_timeout_s=auto_install_timeout_s,
            auto_install_user=auto_install_user,
        )
        if execute_tools
        else None
    )

    def _execute_role(role: str, task_for_role: str, step_idx: int, reason: str, override_tool_only: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        nonlocal log_path
        role_constraints = _ensure_role_constraints(constraints_per_role.get(role), role)
        effective_cache_key = _build_selection_cache_key(
            role=role,
            task_text=task_for_role,
            step_idx=step_idx,
            topology=config.topology.value,
        )
        role_once_key = _normalize_role_name(role)
        reused_same_role_agent_once = False

        if reuse_same_role_agent_once and role_once_key in role_once_selection_cache:
            selection = dict(role_once_selection_cache[role_once_key])
            reused_same_role_agent_once = True
        elif reuse_role_selection and effective_cache_key in selection_cache:
            selection = dict(selection_cache[effective_cache_key])
        else:
            selection = None
            if mcts_dynamic_optimization:
                selection = _select_agent_with_mcts_dynamic(
                    task_text=task_for_role,
                    role=role,
                    step_idx=step_idx,
                    config=config,
                    constraints_per_role=constraints_per_role,
                    selected_agent_ids=selected_agent_ids,
                    selected_agent_roles=selected_agent_roles,
                    registry=registry,
                    index=index,
                    embedder=embedder,
                    reranker=reranker,
                    top_n=top_n,
                    top_k=top_k,
                    rerank_top_m=rerank_top_m,
                    mmr_lambda=mmr_lambda,
                    router_no_rerank=router_no_rerank,
                    team_lambda_compat=team_lambda_compat,
                    team_mu_cost=team_mu_cost,
                    team_nu_latency=team_nu_latency,
                    mcts_iterations=mcts_iterations,
                    mcts_rollout_depth=mcts_rollout_depth,
                    mcts_exploration=mcts_exploration,
                    mcts_discount=mcts_discount,
                    mcts_max_candidates=mcts_max_candidates,
                    pair_success_priors=pair_success_priors,
                )

            if selection is None:
                selection = _select_agent_for_role(
                    task_text=task_for_role,
                    role=role,
                    constraints=role_constraints,
                    registry=registry,
                    index=index,
                    embedder=embedder,
                    reranker=reranker,
                    top_n=top_n,
                    top_k=top_k,
                    rerank_top_m=rerank_top_m,
                    mmr_lambda=mmr_lambda,
                    router_llm_client=router_llm_client,
                    router_top_m=router_top_m,
                    router_no_rerank=router_no_rerank,
                    reuse_cache=False,
                    cache=selection_cache,
                    cache_key=effective_cache_key,
                    selected_agent_ids=selected_agent_ids,
                    team_lambda_compat=team_lambda_compat,
                    team_mu_cost=team_mu_cost,
                    team_nu_latency=team_nu_latency,
                )

            if reuse_role_selection:
                selection_cache[effective_cache_key] = dict(selection)
            if reuse_same_role_agent_once:
                role_once_selection_cache[role_once_key] = dict(selection)
        selected_main = selection.get("selected_main")
        if selected_main:
            selected_agent_ids.append(str(selected_main))
            selected_agent_roles.append(_normalize_role_name(role))
        selection_item = {
            "role": role,
            "selected_main": selected_main,
            "selected_shadows": selection.get("selected_shadows"),
        }
        if reused_same_role_agent_once:
            selection_item["reused_same_role_agent_once"] = True
        if mcts_dynamic_optimization and selection.get("mcts") is not None:
            selection_item["mcts"] = selection.get("mcts")
        selection_log.append(selection_item)
        meta = {
            "step": step_idx,
            "topology": config.topology.value,
            "flow_type": config.flow_type,
            "reason": reason,
        }
        extra_context = {
            "history": history,
            "available_roles": config.roles,
            "topology": config.model_dump(),
            "is_decentralized": (config.topology == TopologyType.DECENTRALIZED or config.topology == TopologyType.CHAIN),
        }
        
        # Add previous agent's failure context if available (for multi-agent collaboration)
        if history and len(history) > 0:
            last_history = history[-1]
            # If previous agent failed (no test_passed or status != success), pass failure info
            if not last_history.get("test_passed") and last_history.get("status") != "success":
                extra_context["previous_agent_failed"] = True
                extra_context["previous_agent_role"] = last_history.get("role")
                extra_context["previous_attempt_summary"] = last_history.get("summary", "")[:500]
        
        # Use override if provided, otherwise use the global tool_only setting
        use_tool_only = override_tool_only if override_tool_only is not None else tool_only
        exec_result = _execute_role_with_selection(
            role=role,
            task_text=task_for_role,
            selection=selection,
            registry=registry,
            llm=llm,
            tool_executor=tool_executor,
            max_attempts=max_attempts,
            results=results,
            constraints=role_constraints,
            allow_unknown_roles=allow_unknown_roles,
            runs_dir=runs_dir,
            workflow_version=workflow_version,
            task_id=task_id,
            meta=meta,
            extra_context=extra_context,
            tool_only=use_tool_only,
            task_context=task_context,
        )
        log_path_local = exec_result.get("log_path")
        if log_path_local:
            log_path = log_path_local
        output = exec_result.get("output")
        _append_role_result(results, role, output)
        if exec_result.get("executor_result") is not None:
            _append_tool_exec(tool_exec, role, exec_result.get("executor_result"))
        
        # Record history with test status for next agent to see
        history_entry = {
            "step": step_idx,
            "role": role,
            "summary": _summarize_output(output, max_chars=summary_max_chars),
        }
        
        # Add test status if available (for next agent to know if task completed)
        if isinstance(output, dict):
            if "test_passed" in output:
                history_entry["test_passed"] = output["test_passed"]
            elif "status" in output and output["status"] == "success":
                history_entry["status"] = "success"
            if _normalize_role_name(role) == "planner":
                history_entry["planner_ready_to_handoff"] = _planner_ready_to_handoff(output)
                planner_hint = _planner_next_role(output)
                if planner_hint:
                    history_entry["planner_next_role"] = planner_hint

        history.append(history_entry)
        
        return output

    roles_available = config.roles or role_list
    if not roles_available:
        raise ValueError("No roles available for dynamic workflow")

    if config.topology == TopologyType.SINGLE:
        single_output = _execute_role(config.entry_role or roles_available[0], task_text, 0, "single")
        if isinstance(single_output, dict) and (
            single_output.get("test_passed") or single_output.get("status") == "success"
        ):
            workflow_succeeded = True
    elif config.topology == TopologyType.CENTRALIZED:
        manager = config.manager_role or (roles_available[0] if roles_available else "planner")
        # Manager/planner should use LLM for planning, not tools for code generation
        # Disable tool_only for manager execution so it can generate planning steps
        manager_output = _execute_role(manager, task_text, 0, "manager", override_tool_only=False)
        manager_is_planner = _normalize_role_name(manager) == "planner"
        planner_ready = _planner_ready_to_handoff(manager_output) if manager_is_planner else True
        
        # Extract steps from manager output (for logging/planning purposes)
        steps = []
        if isinstance(manager_output, dict):
            steps = manager_output.get("steps") or manager_output.get("plan") or []
        print(f"[DEBUG] Manager output keys: {list(manager_output.keys()) if isinstance(manager_output, dict) else 'not a dict'}", file=sys.stderr)
        print(f"[DEBUG] Manager generated {len(steps)} steps: {steps[:3] if len(steps) > 3 else steps}", file=sys.stderr)
        
        # In CENTRALIZED mode, workers collaborate on the same task (not separate steps)
        # Keep trying different workers until task is solved or no more workers available
        worker_roles = [role for role in roles_available if role != manager] or roles_available
        last_worker = None
        for worker_idx in range(config.max_steps):
            candidate_roles = list(worker_roles)
            avoid_role = last_worker
            if manager_is_planner and not planner_ready and not force_role:
                # Planner has not handed off yet: include manager as a routable option.
                # The concrete next role is still chosen by the centralized routing LLM.
                candidate_roles = [manager] + [role for role in worker_roles if role != manager]
                avoid_role = None
                print("[DEBUG] Centralized planner_ready=false, allowing manager replanning in role routing", file=sys.stderr)
            next_role = _select_next_role(
                task_text=task_text,
                available_roles=candidate_roles,
                history=history,
                llm=next_role_llm_client,
                force_role=force_role,
                avoid_role=avoid_role,  # Try to avoid repeating the same worker unless planner isn't ready
            )
            if not next_role:
                print(f"[DEBUG] No more workers to try, stopping CENTRALIZED workflow", file=sys.stderr)
                break
            if next_role == manager:
                manager_output = _execute_role(
                    manager,
                    task_text,
                    worker_idx + 1,
                    "centralized_manager_replan",
                    override_tool_only=False,
                )
                if isinstance(manager_output, dict) and (
                    manager_output.get("test_passed") or manager_output.get("status") == "success"
                ):
                    print("[DEBUG] Manager reported success, stopping CENTRALIZED workflow", file=sys.stderr)
                    workflow_succeeded = True
                    break
                planner_ready = _planner_ready_to_handoff(manager_output) if manager_is_planner else True
                continue
            worker_output = _execute_role(next_role, task_text, worker_idx + 1, "centralized")
            last_worker = next_role
            if isinstance(worker_output, dict) and (worker_output.get("test_passed") or worker_output.get("status") == "success"):
                print("[DEBUG] Test passed, stopping CENTRALIZED workflow", file=sys.stderr)
                workflow_succeeded = True
                break
    else:
        current_role = config.entry_role or roles_available[0]
        visited: Dict[str, int] = {}
        last_output: Optional[Any] = None
        route_query = task_text
        for step_idx in range(config.max_steps):
            task_for_role = task_text
            if soft_connection:
                task_for_role = f"{task_text}\nRouting focus: {route_query}"
            last_output = _execute_role(current_role, task_for_role, step_idx, "dynamic_soft" if soft_connection else "dynamic")
            if isinstance(last_output, dict) and (last_output.get("test_passed") or last_output.get("status") == "success"):
                print("[DEBUG] Test passed, stopping dynamic workflow", file=sys.stderr)
                workflow_succeeded = True
                break
            visited[current_role] = visited.get(current_role, 0) + 1
            neighbors = _neighbors_for_role(current_role, config)
            candidates = neighbors if neighbors else roles_available
            avoid_role = current_role if visited.get(current_role, 0) > 1 else None

            # Planner controls handoff timing in non-centralized flows.
            # If the planner says it's not ready, keep planner for next step.
            current_role_key = _normalize_role_name(current_role)
            if current_role_key == "planner" and not force_role:
                planner_ready = _planner_ready_to_handoff(last_output)
                if not planner_ready:
                    print("[DEBUG] Planner not ready_to_handoff, continuing planner", file=sys.stderr)
                    continue

                planner_suggested = _planner_next_role(last_output)
                resolved_planner_suggested = None
                if planner_suggested:
                    resolved_planner_suggested = next(
                        (candidate for candidate in candidates if _normalize_role_name(candidate) == planner_suggested),
                        None,
                    )
                if resolved_planner_suggested:
                    print(
                        f"[DEBUG] Planner ready_to_handoff, suggested next_role={resolved_planner_suggested}",
                        file=sys.stderr,
                    )
                    next_role = resolved_planner_suggested
                    current_role = next_role
                    continue

                builder_fallback = next(
                    (candidate for candidate in candidates if _normalize_role_name(candidate) == "builder"),
                    None,
                )
                if builder_fallback:
                    print(
                        "[DEBUG] Planner ready_to_handoff, no valid suggestion; fallback next_role=builder",
                        file=sys.stderr,
                    )
                    current_role = builder_fallback
                    continue
            
            # Check if agent specified next_role in output (agent-driven routing for decentralized)
            agent_suggested_role = None
            if isinstance(last_output, dict):
                agent_suggested_role = last_output.get("next_role")
                if agent_suggested_role:
                    agent_suggested_role = str(agent_suggested_role).strip().lower()
                    # Validate that suggested role is available
                    if agent_suggested_role in candidates:
                        print(f"[DEBUG] Agent {current_role} suggested next_role: {agent_suggested_role}", file=sys.stderr)
                        next_role = agent_suggested_role
                    elif agent_suggested_role == "finish":
                        print(f"[DEBUG] Agent {current_role} requested finish", file=sys.stderr)
                        next_role = None
                    else:
                        print(f"[WARNING] Agent suggested invalid next_role '{agent_suggested_role}', not in {candidates}. Using router.", file=sys.stderr)
                        agent_suggested_role = None
            
            # If agent didn't specify next_role, use the router
            if agent_suggested_role is None or agent_suggested_role not in candidates:
                if soft_connection:
                    route_query = _build_route_query(
                        task_text=task_text,
                        history=history,
                        last_output=last_output,
                        llm=route_query_llm_client,
                        fallback_max_chars=summary_max_chars,
                    )
                next_role = _select_next_role(
                    task_text=route_query if soft_connection else task_text,
                    available_roles=candidates,
                    history=history,
                    llm=next_role_llm_client,
                    avoid_role=avoid_role,
                    force_role=force_role,
                )
            
            if not next_role:
                break
            current_role = next_role

    last_history = history[-1] if history else {}
    last_role = _normalize_role_name(last_history.get("role"))
    last_step_failed = not (
        bool(last_history.get("test_passed")) or str(last_history.get("status") or "") == "success"
    )
    should_force_final_builder = (
        force_final_builder_extra_step
        and not workflow_succeeded
        and last_role == "checker"
        and last_step_failed
    )
    if should_force_final_builder:
        builder_role = next(
            (role for role in roles_available if _normalize_role_name(role) == "builder"),
            None,
        )
        if builder_role:
            _execute_role(
                builder_role,
                task_text,
                config.max_steps + 1,
                "forced_final_builder_extra",
            )
            forced_final_builder_executed = True

    checker_ok = _checker_ok(results)
    return {
        "results": results,
        "tool_exec": tool_exec,
        "log_path": log_path,
        "selections": selection_log,
        "checker_ok": checker_ok,
        "topology": config.model_dump(),
        "forced_final_builder_executed": forced_final_builder_executed,
    }


def run_workflow(
    task_text: str,
    roles: Optional[List[str]] = None,
    constraints_per_role: Optional[Dict[str, Dict[str, Any]]] = None,
    workflow_version: str = "v1",
    registry=None,
    index=None,
    embedder=None,
    top_n: int = 20,
    top_k: int = 5,
    rerank_top_m: int = 3,
    mmr_lambda: float = 0.5,
    max_attempts: int = 3,
    bandit_db_path: Optional[str] = None,
    reranker_model_path: Optional[str] = None,
    execute_tools: bool = True,
    tool_timeout_s: float = 1.0,
    auto_fill_tool_inputs: bool = False,
    tool_llm: Optional[Any] = None,
    auto_install_common_libs: bool = False,
    auto_install_timeout_s: float = 300.0,
    auto_install_user: bool = False,
    llm_client: Optional[LLMClient] = None,
    router_llm_client: Optional[LLMClient] = None,
    router_top_m: int = 5,
    router_no_rerank: bool = False,
    runs_dir: str = "runs",
    task_context: Optional[Dict[str, Any]] = None,
    dynamic_topology: bool = False,
    topology: Optional[str] = None,
    topology_config: Optional[Dict[str, Any]] = None,
    meta_router_llm_client: Optional[LLMClient] = None,
    next_role_llm_client: Optional[LLMClient] = None,
    max_steps: int = 6,
    force_final_builder_extra_step: bool = True,
    allow_unknown_roles: bool = False,
    reuse_role_selection: bool = True,
    reuse_same_role_agent_once: bool = False,
    summary_max_chars: int = 400,
    soft_connection: bool = False,
    tool_only: bool = False,
    team_lambda_compat: float = 0.2,
    team_mu_cost: float = 0.05,
    team_nu_latency: float = 0.05,
    mcts_dynamic_optimization: bool = False,
    mcts_iterations: int = 64,
    mcts_rollout_depth: int = 4,
    mcts_exploration: float = 1.414,
    mcts_discount: float = 0.95,
    mcts_max_candidates: int = 8,
    force_role: Optional[str] = None,
    strict_roles: bool = False,
) -> Dict[str, Any]:
    if registry is None or index is None or embedder is None:
        raise ValueError("registry, index, embedder are required")

    role_list = roles or ["planner", "builder", "checker", "refactor"]
    constraints_per_role = constraints_per_role or {}
    task_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    llm = RealLLMClient(llm_client) if llm_client else MockLLMClient()
    if reranker_model_path and os.path.exists(reranker_model_path):
        reranker = TfidfLinearReranker.load(reranker_model_path)
    else:
        reranker = TfidfLinearReranker()
    results: Dict[str, Any] = {}
    tool_exec: Dict[str, Any] = {}
    log_path: Optional[str] = None
    selections: Dict[str, Dict[str, Any]] = {}
    tool_executor = (
        ToolExecutor(
            registry,
            timeout_s=tool_timeout_s,
            auto_fill=auto_fill_tool_inputs,
            llm=tool_llm,
            auto_install_common_libs=auto_install_common_libs,
            auto_install_timeout_s=auto_install_timeout_s,
            auto_install_user=auto_install_user,
        )
        if execute_tools
        else None
    )

    topology_value = (topology or "").strip().lower()
    dynamic_mode = dynamic_topology or topology_config is not None or (
        topology_value and topology_value not in {"linear", "fixed"}
    )
    if dynamic_mode:
        pair_success_priors = _load_pair_success_priors(bandit_db_path, workflow_version)
        dynamic_result = _run_dynamic_workflow(
            task_text=task_text,
            role_list=role_list,
            constraints_per_role=constraints_per_role,
            workflow_version=workflow_version,
            registry=registry,
            index=index,
            embedder=embedder,
            top_n=top_n,
            top_k=top_k,
            rerank_top_m=rerank_top_m,
            mmr_lambda=mmr_lambda,
            max_attempts=max_attempts,
            reranker=reranker,
            execute_tools=execute_tools,
            tool_timeout_s=tool_timeout_s,
            auto_fill_tool_inputs=auto_fill_tool_inputs,
            tool_llm=tool_llm,
            auto_install_common_libs=auto_install_common_libs,
            auto_install_timeout_s=auto_install_timeout_s,
            auto_install_user=auto_install_user,
            llm=llm,
            router_llm_client=router_llm_client,
            router_top_m=router_top_m,
            router_no_rerank=router_no_rerank,
            runs_dir=runs_dir,
            task_id=task_id,
            topology=topology,
            topology_config=topology_config,
            meta_router_llm_client=meta_router_llm_client or router_llm_client,
            next_role_llm_client=next_role_llm_client or router_llm_client,
            route_query_llm_client=next_role_llm_client or router_llm_client,
            max_steps=max_steps,
            force_final_builder_extra_step=force_final_builder_extra_step,
            allow_unknown_roles=allow_unknown_roles or dynamic_mode,
            force_role=force_role,
            reuse_role_selection=reuse_role_selection,
            reuse_same_role_agent_once=reuse_same_role_agent_once,
            summary_max_chars=summary_max_chars,
            soft_connection=soft_connection,
            tool_only=tool_only,
            team_lambda_compat=team_lambda_compat,
            team_mu_cost=team_mu_cost,
            team_nu_latency=team_nu_latency,
            mcts_dynamic_optimization=mcts_dynamic_optimization,
            mcts_iterations=mcts_iterations,
            mcts_rollout_depth=mcts_rollout_depth,
            mcts_exploration=mcts_exploration,
            mcts_discount=mcts_discount,
            mcts_max_candidates=mcts_max_candidates,
            pair_success_priors=pair_success_priors,
            task_context=task_context,
            strict_roles=strict_roles,
        )

        if bandit_db_path:
            checker_ok = dynamic_result.get("checker_ok", True)
            selections_list = dynamic_result.get("selections") or []
            with BanditStore(bandit_db_path) as store:
                for selection in selections_list:
                    role = selection.get("role", "")
                    main_id = selection.get("selected_main")
                    shadow_ids = selection.get("selected_shadows", [])
                    if main_id:
                        store.update(
                            workflow_version,
                            role,
                            main_id,
                            reward=1.0 if checker_ok else 0.0,
                            confidence=1.0,
                        )
                    for shadow_id in shadow_ids:
                        store.update(
                            workflow_version,
                            role,
                            shadow_id,
                            reward=1.0 if checker_ok else 0.0,
                            confidence=0.5,
                        )
            _update_pair_success_stats(
                bandit_db_path,
                workflow_version=workflow_version,
                selections=selections_list,
                reward=1.0 if checker_ok else 0.0,
                confidence=1.0,
            )

        results = dynamic_result.get("results", {})
        answer_lines = [f"[{role}] {results.get(role)}" for role in results.keys()]
        answer = "\n".join(answer_lines)
        return {
            "task_id": task_id,
            "answer": answer,
            "log_path": dynamic_result.get("log_path"),
            "results": results,
            "tool_exec": dynamic_result.get("tool_exec"),
            "topology": dynamic_result.get("topology"),
            "forced_final_builder_executed": bool(dynamic_result.get("forced_final_builder_executed")),
        }

    selected_agent_ids: List[str] = []
    selection_cache: Dict[str, Dict[str, Any]] = {}
    for role in role_list:
        role_key = _normalize_role_name(role)
        constraints = _ensure_role_constraints(constraints_per_role.get(role), role)
        selection = _select_agent_for_role(
            task_text=task_text,
            role=role,
            constraints=constraints,
            registry=registry,
            index=index,
            embedder=embedder,
            reranker=reranker,
            top_n=top_n,
            top_k=top_k,
            rerank_top_m=rerank_top_m,
            mmr_lambda=mmr_lambda,
            router_llm_client=router_llm_client,
            router_top_m=router_top_m,
            router_no_rerank=router_no_rerank,
            reuse_cache=reuse_role_selection,
            cache=selection_cache,
            cache_key=(
                _normalize_role_name(role)
                if reuse_same_role_agent_once
                else _build_selection_cache_key(
                    role=role,
                    task_text=task_text,
                    step_idx=0,
                    topology="linear",
                )
            ),
            selected_agent_ids=selected_agent_ids,
            team_lambda_compat=team_lambda_compat,
            team_mu_cost=team_mu_cost,
            team_nu_latency=team_nu_latency,
        )

        selected_main = selection.get("selected_main")
        selected_shadows = list(selection.get("selected_shadows") or [])
        selected_shadow = selection.get("selected_shadow")
        reranked = list(selection.get("reranked") or [])

        if selected_main:
            selected_agent_ids.append(str(selected_main))

        selections[role] = {
            "main": selected_main,
            "shadows": selected_shadows,
        }

        attempt = 0
        role_context: Dict[str, Any] = {"upstream": results, "constraints": constraints, "available_roles": role_list}
        agent_context = _get_agent_context(registry, selected_main)
        role_context["agent"] = agent_context
        tool_history = _plan_and_run_tools(
            llm=llm,
            role=role,
            task_text=task_text,
            role_context=role_context,
            agent_context=agent_context,
            registry=registry,
            tool_executor=tool_executor,
            results=results,
            tool_max_rounds=5,  # Reduced from 10 to allow faster agent switching
            tool_only=tool_only,
            task_context=task_context,
        )
        if tool_only:
            tool_output = _tool_history_to_output(role, tool_history)
            if isinstance(tool_output, dict) and tool_output.get("error"):
                validation = {"ok": False, "errors": [str(tool_output.get("error"))]}
                stored_output = tool_output
            else:
                valid, errors, model = validate_output(role, tool_output, allow_unknown=allow_unknown_roles)
                validation = {"ok": valid, "errors": errors}
                if valid and model is not None:
                    stored_output = model.model_dump()
                else:
                    stored_output = tool_output
            results[role] = stored_output
            if tool_history:
                tool_exec[role] = tool_history
            log_path = write_event(
                task_id=task_id,
                workflow_version=workflow_version,
                role=role,
                selected_main=selected_main,
                selected_shadow=selected_shadow,
                candidates_topk=reranked,
                output=tool_output,
                validation=validation,
                executor_result=tool_history or None,
                failure_type="tool_only",
                action="tool_only",
                runs_dir=runs_dir,
            )
            continue
        while attempt < max_attempts:
            llm_task_text = _append_diagnostics_to_task(task_text, role_context)
            output = llm.generate(role, llm_task_text, role_context)
            valid, errors, model = validate_output(role, output, allow_unknown=allow_unknown_roles)
            validation = {"ok": valid, "errors": errors}
            executor_result = tool_history or None
            failure_type = None
            action = None

            if valid:
                if model is not None:
                    stored_output = model.model_dump()
                elif isinstance(output, dict):
                    stored_output = output
                else:
                    stored_output = {"raw": output}
                if role_key == "builder" and task_context:
                    code_text = stored_output.get("code_or_commands") if isinstance(stored_output, dict) else ""
                    if isinstance(code_text, str) and code_text.strip():
                        test_ok, test_error = _test_code(code_text, task_context)
                        if not test_ok:
                            role_context["diagnostics"] = {
                                "test_error": test_error,
                                "error_message": f"Test failed: {test_error}",
                                "failed_code": code_text,
                            }
                            attempt += 1
                            continue
                results[role] = stored_output
                if tool_history:
                    tool_exec[role] = tool_history
                log_path = write_event(
                    task_id=task_id,
                    workflow_version=workflow_version,
                    role=role,
                    selected_main=selected_main,
                    selected_shadow=selected_shadow,
                    candidates_topk=reranked,
                    output=output,
                    validation=validation,
                    executor_result=executor_result,
                    failure_type=failure_type,
                    action=action,
                    runs_dir=runs_dir,
                )
                break

            route = route_failure(role, output, errors, executor_result)
            failure_type = route["failure_type"]
            action = route["action"]

            if failure_type == "A_contract":
                repaired = llm.rewrite_format(role, output, role_context)
                valid_fix, errors_fix, model_fix = validate_output(role, repaired, allow_unknown=allow_unknown_roles)
                validation = {"ok": valid_fix, "errors": errors_fix}
                if valid_fix:
                    if model_fix is not None:
                        stored_output = model_fix.model_dump()
                    elif isinstance(repaired, dict):
                        stored_output = repaired
                    else:
                        stored_output = {"raw": repaired}
                    results[role] = stored_output
                    log_path = write_event(
                        task_id=task_id,
                        workflow_version=workflow_version,
                        role=role,
                        selected_main=selected_main,
                        selected_shadow=selected_shadow,
                        candidates_topk=reranked,
                        output=repaired,
                        validation=validation,
                        executor_result=executor_result,
                        failure_type=failure_type,
                        action=action,
                        runs_dir=runs_dir,
                    )
                    break
                log_path = write_event(
                    task_id=task_id,
                    workflow_version=workflow_version,
                    role=role,
                    selected_main=selected_main,
                    selected_shadow=selected_shadow,
                    candidates_topk=reranked,
                    output=repaired,
                    validation=validation,
                    executor_result=executor_result,
                    failure_type=failure_type,
                    action=action,
                    runs_dir=runs_dir,
                )
                attempt += 1
                continue

            if failure_type == "B_missing_info":
                role_context.update(llm.clarify(role, task_text))
                log_path = write_event(
                    task_id=task_id,
                    workflow_version=workflow_version,
                    role=role,
                    selected_main=selected_main,
                    selected_shadow=selected_shadow,
                    candidates_topk=reranked,
                    output=output,
                    validation=validation,
                    executor_result=executor_result,
                    failure_type=failure_type,
                    action=action,
                    runs_dir=runs_dir,
                )
                attempt += 1
                continue

            if failure_type == "C_capability":
                log_path = write_event(
                    task_id=task_id,
                    workflow_version=workflow_version,
                    role=role,
                    selected_main=selected_main,
                    selected_shadow=selected_shadow,
                    candidates_topk=reranked,
                    output=output,
                    validation=validation,
                    executor_result=executor_result,
                    failure_type=failure_type,
                    action=action,
                    runs_dir=runs_dir,
                )
                if selected_shadow and selected_shadow != selected_main:
                    selected_main = selected_shadow
                    selected_shadow = None
                    agent_context = _get_agent_context(registry, selected_main)
                    role_context["agent"] = agent_context
                    attempt += 1
                    continue
                break

            attempt += 1

        if role not in results:
            results[role] = {"error": "no_valid_output"}

    checker_ok = True
    checker_output = results.get("checker")
    if isinstance(checker_output, dict) and checker_output.get("error"):
        checker_ok = False

    if bandit_db_path:
        with BanditStore(bandit_db_path) as store:
            for role, selection in selections.items():
                main_id = selection.get("main")
                shadow_ids = selection.get("shadows", [])
                if main_id:
                    store.update(workflow_version, role, main_id, reward=1.0 if checker_ok else 0.0, confidence=1.0)
                for shadow_id in shadow_ids:
                    store.update(workflow_version, role, shadow_id, reward=1.0 if checker_ok else 0.0, confidence=0.5)
        static_selections = [
            {"role": role, "selected_main": (selections.get(role) or {}).get("main")}
            for role in role_list
            if (selections.get(role) or {}).get("main")
        ]
        _update_pair_success_stats(
            bandit_db_path,
            workflow_version=workflow_version,
            selections=static_selections,
            reward=1.0 if checker_ok else 0.0,
            confidence=1.0,
        )

    answer_lines = []
    for role in role_list:
        answer_lines.append(f"[{role}] {results.get(role)}")
    answer = "\n".join(answer_lines)

    return {
        "task_id": task_id,
        "answer": answer,
        "log_path": log_path,
        "results": results,
        "tool_exec": tool_exec,
    }
