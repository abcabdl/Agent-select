from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.cards import AgentCard, ToolCard  # noqa: E402
from core.registry import SQLiteRegistry  # noqa: E402


TOOLS = [
    {
        "id": "postprocess-fallback-guard",
        "name": "PostprocessFallbackGuard",
        "file": "postprocess-fallback-guard.py",
        "description": "Detect fallback/refusal/placeholder code and mark completion invalid.",
    },
    {
        "id": "postprocess-extract-cleanup",
        "name": "PostprocessExtractCleanup",
        "file": "postprocess-extract-cleanup.py",
        "description": "Extract raw code from JSON/markdown wrappers and strip redundant entry def.",
    },
    {
        "id": "postprocess-stop-token-trim",
        "name": "PostprocessStopTokenTrim",
        "file": "postprocess-stop-token-trim.py",
        "description": "Trim completion using earliest matched stop token.",
    },
    {
        "id": "postprocess-normalize-body",
        "name": "PostprocessNormalizeBody",
        "file": "postprocess-normalize-body.py",
        "description": "Normalize function-body indentation for HumanEval body-style completions.",
    },
    {
        "id": "postprocess-param-consistency",
        "name": "PostprocessParamConsistency",
        "file": "postprocess-param-consistency.py",
        "description": "Repair mismatched parameter aliases to exact signature parameter names.",
    },
    {
        "id": "postprocess-builtin-shadowing",
        "name": "PostprocessBuiltinShadowing",
        "file": "postprocess-builtin-shadowing.py",
        "description": "Repair isinstance/type checks when params shadow builtin type names.",
    },
    {
        "id": "postprocess-syntax-indent-repair",
        "name": "PostprocessSyntaxIndentRepair",
        "file": "postprocess-syntax-indent-repair.py",
        "description": "Attempt syntax/indent body repair only when parse failure is detected.",
    },
    {
        "id": "postprocess-name-scope-repair",
        "name": "PostprocessNameScopeRepair",
        "file": "postprocess-name-scope-repair.py",
        "description": "Repair common NameError/UnboundLocalError missing symbols via lightweight imports/helpers.",
    },
    {
        "id": "postprocess-timeout-guard",
        "name": "PostprocessTimeoutGuard",
        "file": "postprocess-timeout-guard.py",
        "description": "Repair likely timeout-prone loop patterns with bounded guards.",
    },
]


AGENT_ID = "special-postprocess-agent"
AGENT_NAME = "PostprocessToolAgent-Special"


def _read_tool_code(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Tool file not found: {path}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    db_path = ROOT / "demo_registry.sqlite"
    tools_dir = ROOT / "generated_tools"
    now = datetime.utcnow()

    registered_tools = []
    with SQLiteRegistry(str(db_path)) as registry:
        for tool in TOOLS:
            tool_path = tools_dir / tool["file"]
            code = _read_tool_code(tool_path)
            tool_card = ToolCard(
                id=tool["id"],
                name=tool["name"],
                kind="tool",
                version="1.0",
                updated_at=now,
                domain_tags=["humaneval", "code-postprocess"],
                role_tags=["postprocess"],
                tool_tags=["postprocess", tool["id"]],
                modalities=["text"],
                output_formats=["json"],
                permissions=["read"],
                cost_tier="low",
                latency_tier="low",
                reliability_prior=0.9,
                description=tool["description"],
                examples=["Postprocess one completion after generation."],
                embedding_text=f"humaneval postprocess {tool['id']}",
            )
            registry.update(tool_card)
            registry.register_tool_code(tool["id"], code, updated_at=now)
            registered_tools.append(tool["id"])

        agent_card = AgentCard(
            id=AGENT_ID,
            name=AGENT_NAME,
            kind="agent",
            version="1.0",
            updated_at=now,
            domain_tags=["humaneval", "code-postprocess"],
            role_tags=["postprocess"],
            tool_tags=list(registered_tools),
            modalities=["text"],
            output_formats=["json"],
            permissions=["read"],
            cost_tier="low",
            latency_tier="low",
            reliability_prior=0.92,
            description=(
                "Special postprocess agent for HumanEval completions. "
                "Runs fallback guard + extraction + normalization + targeted repairs."
            ),
            examples=["Apply postprocess tools after each code generation."],
            embedding_text="special humaneval postprocess agent",
            available_tool_ids=list(registered_tools),
        )
        registry.update(agent_card)

    print(
        json.dumps(
            {
                "db_path": str(db_path),
                "agent_id": AGENT_ID,
                "tool_count": len(registered_tools),
                "tools": registered_tools,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
