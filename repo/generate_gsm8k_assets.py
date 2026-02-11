from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


STAGES = ["analyze", "model", "solve", "verify", "format"]

FAMILY_SETTINGS = {
    "solver": {
        "role": "builder",
        "style": "HighlyPrecise",
        "family_prompt": "Prioritize robust equation building and exact numerical solving.",
        "stage_cfg": {
            "analyze": ("gpt-4o-mini", 0.05, 700),
            "model": ("gpt-4o", 0.18, 900),
            "solve": ("gpt-4o", 0.22, 1100),
            "verify": ("gpt-4o-mini", 0.03, 700),
            "format": ("gpt-4o-mini", 0.00, 280),
        },
    },
    "verifier": {
        "role": "checker",
        "style": "AuditStrict",
        "family_prompt": "Prioritize error detection, consistency checks, and independent recomputation.",
        "stage_cfg": {
            "analyze": ("gpt-4o-mini", 0.03, 650),
            "model": ("gpt-4o-mini", 0.08, 800),
            "solve": ("gpt-4o", 0.12, 900),
            "verify": ("gpt-4o", 0.02, 1000),
            "format": ("gpt-4o-mini", 0.00, 260),
        },
    },
    "formatter": {
        "role": "refractor",
        "style": "Deterministic",
        "family_prompt": "Prioritize deterministic final-answer extraction and strict output normalization.",
        "stage_cfg": {
            "analyze": ("gpt-4o-mini", 0.02, 500),
            "model": ("gpt-4o-mini", 0.05, 650),
            "solve": ("gpt-4o-mini", 0.08, 700),
            "verify": ("gpt-4o-mini", 0.02, 600),
            "format": ("gpt-4o-mini", 0.00, 220),
        },
    },
}


SOLVER_FOCUSES = [
    "ratio_chain",
    "unit_rate",
    "percent_change",
    "fraction_conversion",
    "mixture_balance",
    "work_rate",
    "time_distance",
    "profit_margin",
    "discount_tax",
    "sequence_count",
    "age_equation",
    "set_overlap",
    "piecewise_pricing",
    "geometry_area",
    "geometry_volume",
    "probability_expected",
    "average_weighted",
    "clock_calendar",
    "travel_schedule",
    "inventory_flow",
    "salary_overtime",
    "loan_interest",
    "currency_exchange",
    "measurement_conversion",
    "digit_number_theory",
    "combinatoric_selection",
    "resource_allocation",
    "production_planning",
    "water_tank",
    "comparison_scaling",
    "multi_step_budget",
    "distance_roundtrip",
    "surplus_deficit",
    "equal_partition",
]

VERIFIER_FOCUSES = [
    "unit_consistency",
    "arithmetic_recompute",
    "equation_balance",
    "boundary_checks",
    "constraint_satisfaction",
    "sign_error_guard",
    "rounding_audit",
    "percentage_sanity",
    "rate_time_crosscheck",
    "counting_doublecheck",
    "currency_integrity",
    "timeline_consistency",
    "logic_gap_detection",
    "invariant_tracking",
    "alternative_method_check",
    "dimensional_analysis",
    "assumption_audit",
    "off_by_one_guard",
    "order_of_ops_audit",
    "fraction_decimal_match",
    "integer_feasibility",
    "negative_value_guard",
    "extreme_case_probe",
    "duplicate_count_guard",
    "set_relation_check",
    "ratio_normalization",
    "total_vs_parts_check",
    "resource_conservation",
    "symmetry_check",
    "parity_check",
    "reasoning_trace_audit",
    "answer_magnitude_check",
    "independence_check",
]

FORMATTER_FOCUSES = [
    "final_integer",
    "final_decimal",
    "final_currency",
    "final_percentage",
    "final_fraction",
    "final_time_unit",
    "final_distance_unit",
    "final_count_unit",
    "final_volume_unit",
    "final_area_unit",
    "final_mass_unit",
    "final_rate_unit",
    "hashline_strict",
    "comma_cleanup",
    "trailing_zero_trim",
    "scientific_to_plain",
    "negative_sign_normalize",
    "mixed_number_normalize",
    "unit_suffix_standardize",
    "currency_symbol_strip",
    "percent_symbol_append",
    "answer_only_mode",
    "step_summary_short",
    "explanation_compact",
    "json_answer_pack",
    "markdown_free_output",
    "locale_number_normalize",
    "approximation_tagging",
    "exact_value_tagging",
    "box_answer_style",
    "double_hash_style",
    "newline_safe_output",
    "dedupe_whitespace",
]


@dataclass
class ToolDef:
    tool_id: str
    name: str
    family: str
    focus: str
    stage: str
    model: str
    temperature: float
    max_tokens: int
    code: str


@dataclass
class AgentDef:
    agent_id: str
    name: str
    family: str
    role: str
    focus: str
    tool_ids: List[str]


def _slug(value: str) -> str:
    cleaned = "".join(ch for ch in value.lower().replace(" ", "-") if ch.isalnum() or ch == "-")
    return cleaned.strip("-") or "item"


def _title_from_slug(value: str) -> str:
    return "".join(part.title() for part in value.split("_"))


def _model_tag(model: str) -> str:
    return "4omini" if "mini" in model else "4o"


def _stage_directive(stage: str) -> str:
    mapping = {
        "analyze": "Extract known values, unknown target, units, and latent constraints.",
        "model": "Build symbolic relations and compact equation plans before arithmetic.",
        "solve": "Execute calculations carefully and keep intermediate results explicit.",
        "verify": "Recompute independently and detect inconsistency, sign, and unit mistakes.",
        "format": "Return only normalized final answer in strict GSM8K style.",
    }
    return mapping[stage]


def _temp_jitter(index: int, stage_idx: int, family: str, base: float) -> float:
    if base <= 0.0:
        return 0.0
    patterns = {
        "solver": [0.00, 0.02, -0.01, 0.01, -0.02],
        "verifier": [0.00, -0.01, 0.01, -0.02, 0.00],
        "formatter": [0.00, 0.01, -0.01, 0.00, 0.00],
    }
    delta = patterns[family][(index + stage_idx) % len(patterns[family])]
    return max(0.0, min(0.35, round(base + delta, 3)))


def _build_tool_code(
    tool_id: str,
    family: str,
    focus: str,
    stage: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    family_prompt = FAMILY_SETTINGS[family]["family_prompt"]
    stage_prompt = _stage_directive(stage)
    focus_label = focus.replace("_", " ")
    return f'''"""
{tool_id}: GSM8K {family} tool ({stage}) with focus on {focus_label}.
"""
import json
import os
import re
import httpx


TOOL_ID = "{tool_id}"
FAMILY = "{family}"
FOCUS = "{focus_label}"
STAGE = "{stage}"
MODEL = "{model}"
TEMPERATURE = {temperature}
MAX_TOKENS = {max_tokens}


def _pick_task(inputs):
    if not isinstance(inputs, dict):
        return str(inputs or "")
    return (
        inputs.get("question")
        or inputs.get("task_text")
        or inputs.get("prompt")
        or inputs.get("task")
        or inputs.get("query")
        or ""
    )


def _extract_json_blob(text):
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\\s*(.*?)\\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    if "{{" in text and "}}" in text:
        start = text.find("{{")
        end = text.rfind("}}")
        if end > start:
            return text[start : end + 1]
    return text.strip()


def _safe_json(text):
    blob = _extract_json_blob(text)
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _collect_numbers(text):
    if not text:
        return []
    return re.findall(r"-?\\d+(?:\\.\\d+)?", text)


def _fallback(inputs, task):
    numbers = _collect_numbers(task)
    candidate = ""
    if isinstance(inputs, dict):
        candidate = str(inputs.get("final_answer") or inputs.get("answer") or "").strip()
    if not candidate and numbers:
        candidate = numbers[-1]
    if STAGE == "analyze":
        return {{
            "known_numbers": numbers,
            "target": "unknown",
            "notes": f"fallback_{{STAGE}}",
        }}
    if STAGE == "model":
        return {{
            "equation_plan": "derive relation from quantities and solve for target",
            "focus": FOCUS,
            "notes": f"fallback_{{STAGE}}",
        }}
    if STAGE == "solve":
        return {{
            "steps": ["fallback solve path"],
            "numeric_answer": candidate,
            "notes": f"fallback_{{STAGE}}",
        }}
    if STAGE == "verify":
        return {{
            "is_consistent": bool(candidate),
            "recomputed_answer": candidate,
            "issues": [] if candidate else ["missing_candidate_answer"],
            "notes": f"fallback_{{STAGE}}",
        }}
    formatted = f"#### {{candidate}}" if candidate else "#### 0"
    return {{
        "final_answer": candidate or "0",
        "formatted": formatted,
        "notes": f"fallback_{{STAGE}}",
    }}


def _call_llm(prompt):
    api_key = os.getenv("LLM_API_KEY", "")
    base = (os.getenv("LLM_API_BASE") or "").rstrip("/")
    if not api_key or not base:
        return None
    url = base if base.endswith("/chat/completions") else f"{{base}}/chat/completions"
    system = (
        "You are a GSM8K specialist tool. "
        "Always return ONLY a JSON object with keys: stage, focus, result, confidence, checks. "
        "No markdown fences, no prose outside JSON."
    )
    payload = {{
        "model": MODEL,
        "messages": [
            {{"role": "system", "content": system}},
            {{"role": "user", "content": prompt}},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {{"type": "json_object"}},
    }}
    headers = {{
        "Authorization": f"Bearer {{api_key}}",
        "Content-Type": "application/json",
    }}
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{{}}])[0].get("message", {{}}).get("content", "")
    except Exception:
        return None


def run(inputs):
    task = _pick_task(inputs)
    if not task:
        return {{"output": {{"tool_id": TOOL_ID, "error": "missing_task"}}}}
    prompt = (
        "Solve this GSM8K word problem component.\\n"
        + "family=" + FAMILY + ", stage=" + STAGE + ", focus=" + FOCUS + "\\n"
        + "family_policy: {family_prompt}\\n"
        + "stage_policy: {stage_prompt}\\n"
        + f"problem: {{task}}\\n"
        + "Return JSON keys: stage, focus, result, confidence, checks."
    )
    raw = _call_llm(prompt)
    parsed = _safe_json(raw) if raw else None
    if parsed is None:
        payload = _fallback(inputs if isinstance(inputs, dict) else {{}}, task)
    else:
        payload = parsed
    if STAGE == "format":
        if isinstance(payload, dict):
            formatted = str(payload.get("formatted") or "").strip()
            if not formatted:
                candidate = str(payload.get("final_answer") or "").strip() or "0"
                payload["formatted"] = f"#### {{candidate}}"
    return {{
        "output": {{
            "tool_id": TOOL_ID,
            "family": FAMILY,
            "focus": FOCUS,
            "stage": STAGE,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "payload": payload,
        }}
    }}
'''


def _build_profiles() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pairs.extend([("solver", focus) for focus in SOLVER_FOCUSES])
    pairs.extend([("verifier", focus) for focus in VERIFIER_FOCUSES])
    pairs.extend([("formatter", focus) for focus in FORMATTER_FOCUSES])
    return pairs


def _build_agents_and_tools() -> Tuple[List[AgentDef], List[ToolDef]]:
    profiles = _build_profiles()
    agents: List[AgentDef] = []
    tools: List[ToolDef] = []
    family_counter: Dict[str, int] = {"solver": 0, "verifier": 0, "formatter": 0}

    for family, focus in profiles:
        family_counter[family] += 1
        local_idx = family_counter[family]
        role = FAMILY_SETTINGS[family]["role"]
        style = FAMILY_SETTINGS[family]["style"]
        focus_title = _title_from_slug(focus)
        family_title = family.title()
        agent_id = f"gsm8k-agent-{family}-{local_idx:03d}"
        agent_name = f"{family_title}-{focus_title}{local_idx:02d}-{style}-GSM8K"
        tool_ids: List[str] = []

        for stage_idx, stage in enumerate(STAGES):
            base_model, base_temp, max_tokens = FAMILY_SETTINGS[family]["stage_cfg"][stage]
            temp = _temp_jitter(local_idx, stage_idx, family, base_temp)
            model_tag = _model_tag(base_model)
            temp_tag = f"t{int(round(temp * 100)):03d}"
            tool_id = f"gsm8k-{family}-{stage}-{_slug(focus)}-{model_tag}-{temp_tag}-v{local_idx:03d}"
            tool_name = f"GSM8K-{family_title}-{stage.title()}-{focus_title}-V{local_idx:03d}"
            tool_code = _build_tool_code(
                tool_id=tool_id,
                family=family,
                focus=focus,
                stage=stage,
                model=base_model,
                temperature=temp,
                max_tokens=max_tokens,
            )
            tools.append(
                ToolDef(
                    tool_id=tool_id,
                    name=tool_name,
                    family=family,
                    focus=focus,
                    stage=stage,
                    model=base_model,
                    temperature=temp,
                    max_tokens=max_tokens,
                    code=tool_code,
                )
            )
            tool_ids.append(tool_id)

        agents.append(
            AgentDef(
                agent_id=agent_id,
                name=agent_name,
                family=family,
                role=role,
                focus=focus,
                tool_ids=tool_ids,
            )
        )

    return agents, tools


def _write_tools(tools: List[ToolDef], tools_dir: Path) -> None:
    tools_dir.mkdir(parents=True, exist_ok=True)
    for tool in tools:
        path = tools_dir / f"{tool.tool_id}.py"
        path.write_text(tool.code, encoding="utf-8")


def _build_design_payload(agents: List[AgentDef]) -> Dict[str, Dict[str, object]]:
    payload: Dict[str, Dict[str, object]] = {}
    for agent in agents:
        domain = f"gsm8k-{agent.family}"
        payload[agent.name] = {
            "description": (
                f"GSM8K {agent.family} specialist focused on {agent.focus.replace('_', ' ')}. "
                f"Role={agent.role}. Uses 5 dedicated non-overlapping tools."
            ),
            "role_tags": [agent.role],
            "domain_tags": ["gsm8k-math-word-problems", domain],
            "tools": agent.tool_ids,
        }
    return payload


def _write_manifest(
    manifest_path: Path,
    agents: List[AgentDef],
    tools: List[ToolDef],
    design_path: Path,
    tools_dir: Path,
) -> None:
    by_family: Dict[str, int] = {"solver": 0, "verifier": 0, "formatter": 0}
    for agent in agents:
        by_family[agent.family] += 1
    tool_usage: Dict[str, int] = {}
    for agent in agents:
        for tid in agent.tool_ids:
            tool_usage[tid] = tool_usage.get(tid, 0) + 1
    duplicate_tool_refs = [tid for tid, cnt in tool_usage.items() if cnt > 1]
    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "agent_count": len(agents),
        "tool_count": len(tools),
        "distribution": by_family,
        "tools_per_agent": 5,
        "all_tools_unique_across_agents": not duplicate_tool_refs,
        "duplicate_tool_ids": duplicate_tool_refs,
        "design_path": str(design_path),
        "tools_dir": str(tools_dir),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def _validate(agents: List[AgentDef], tools: List[ToolDef]) -> None:
    if len(agents) != 100:
        raise ValueError(f"Expected 100 agents, got {len(agents)}")
    if len(tools) != 500:
        raise ValueError(f"Expected 500 tools, got {len(tools)}")
    all_tool_ids = [tool.tool_id for tool in tools]
    if len(all_tool_ids) != len(set(all_tool_ids)):
        raise ValueError("Duplicate tool IDs found in generated tools.")
    used_tool_ids: List[str] = []
    for agent in agents:
        if len(agent.tool_ids) != 5:
            raise ValueError(f"Agent {agent.agent_id} does not have exactly 5 tools.")
        used_tool_ids.extend(agent.tool_ids)
    if len(used_tool_ids) != len(set(used_tool_ids)):
        raise ValueError("Found repeated tool assignments across agents.")


def _register_cards(
    db_path: str,
    agents: List[AgentDef],
    tools: List[ToolDef],
) -> None:
    from core.cards import AgentCard, ToolCard
    from core.registry import SQLiteRegistry

    now = datetime.utcnow()
    with SQLiteRegistry(db_path) as registry:
        for tool in tools:
            tool_card = ToolCard(
                id=tool.tool_id,
                name=tool.name,
                kind="tool",
                version="1.0",
                updated_at=now,
                domain_tags=["gsm8k-math-word-problems", f"gsm8k-{tool.family}"],
                role_tags=[FAMILY_SETTINGS[tool.family]["role"]],
                tool_tags=[tool.tool_id, tool.stage, tool.focus, tool.family],
                modalities=["text"],
                output_formats=["json"],
                permissions=["read"],
                cost_tier="medium" if tool.model == "gpt-4o" else "low",
                latency_tier="medium",
                reliability_prior=0.8,
                description=(
                    f"GSM8K {tool.family} tool for stage={tool.stage}, "
                    f"focus={tool.focus.replace('_', ' ')}, model={tool.model}, temp={tool.temperature}."
                ),
                examples=[f"Apply {tool.stage} on GSM8K problem focused on {tool.focus.replace('_', ' ')}."],
                embedding_text=f"gsm8k {tool.family} {tool.stage} {tool.focus}",
            )
            registry.update(tool_card)
            registry.register_tool_code(tool.tool_id, tool.code, updated_at=now)

        for agent in agents:
            agent_card = AgentCard(
                id=agent.agent_id,
                name=agent.name,
                kind="agent",
                version="1.0",
                updated_at=now,
                domain_tags=["gsm8k-math-word-problems", f"gsm8k-{agent.family}"],
                role_tags=[agent.role],
                tool_tags=list(agent.tool_ids),
                modalities=["text"],
                output_formats=["json"],
                permissions=["read"],
                cost_tier="medium",
                latency_tier="medium",
                reliability_prior=0.82,
                description=(
                    f"GSM8K {agent.family} specialist for {agent.focus.replace('_', ' ')}. "
                    "Uses five non-overlapping stage tools."
                ),
                examples=[f"Solve/validate GSM8K tasks around {agent.focus.replace('_', ' ')}."],
                embedding_text=f"gsm8k {agent.family} {agent.focus} role {agent.role}",
                available_tool_ids=list(agent.tool_ids),
            )
            registry.update(agent_card)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 100 GSM8K agents and 500 unique GSM8K tools.")
    parser.add_argument("--tools_dir", default="generated_tools", help="directory to write generated tool files")
    parser.add_argument(
        "--design_path",
        default="agent_tool_design_gsm8k_v1.json",
        help="output JSON mapping from agent name to metadata/tools",
    )
    parser.add_argument(
        "--manifest_path",
        default="gsm8k_agent_manifest.json",
        help="output generation summary JSON",
    )
    parser.add_argument("--register_db", action="store_true", help="also register generated cards into SQLite db")
    parser.add_argument("--db", default="demo_registry.sqlite", help="SQLite registry path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tools_dir = ROOT / args.tools_dir
    design_path = ROOT / args.design_path
    manifest_path = ROOT / args.manifest_path

    agents, tools = _build_agents_and_tools()
    _validate(agents, tools)
    _write_tools(tools, tools_dir)

    design_payload = _build_design_payload(agents)
    design_path.write_text(json.dumps(design_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    _write_manifest(manifest_path, agents, tools, design_path, tools_dir)

    if args.register_db:
        _register_cards(db_path=str(ROOT / args.db), agents=agents, tools=tools)

    print(
        json.dumps(
            {
                "agents": len(agents),
                "tools": len(tools),
                "design_path": str(design_path),
                "tools_dir": str(tools_dir),
                "registered_to_db": bool(args.register_db),
                "db_path": str(ROOT / args.db) if args.register_db else "",
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
