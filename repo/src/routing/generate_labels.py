import json
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure we can import from src
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generation.llm_client import LLMClient


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_system_prompt(strategy: str = "single_preference") -> str:
    """
    Generate system prompt based on agent topology strategy.
    
    Args:
        strategy: One of:
            - "single_preference" (default): Strongly encourages single agent (current behavior)
            - "balanced": Neutral stance, chooses based on task complexity
            - "multi_preference": Moderately encourages multi-agent for complex tasks
            - "multi_aggressive": Strongly encourages multi-agent collaboration
    """
    
    base_prompt = """You are a meta-router for code development. Decide the best topology.
Return JSON: {"topology": "...", "roles": [...], "manager_role": "...", "entry_role": "...", "max_steps": N}

Roles: builder, planner, checker, researcher
"""

    if strategy == "single_preference":
        # Current behavior - strongly prefer single agent
        topology_guide = """
**Topology:**
1. "single" (DEFAULT): Simple funcs, basic algos, <15 lines
2. "centralized": Complex algos (DP/graphs), multi-phase, edge cases
3. "decentralized" (RARE): Parallel tasks, agents set next_role

**Roles:** single=["builder"], centralized=["planner","builder"](+checker), decentralized=2-3 no mgr
**Steps:** single=1, centralized=2-3, decentralized=2-4
**Rules:** single: entry_role="builder" mgr=null | centralized: mgr+entry="planner" | decentralized: mgr=null entry=1st. Default single.
"""
    
    elif strategy == "balanced":
        # Neutral approach - decide based on actual complexity, aim for ~50-50 split
        topology_guide = """
**Topology:**
1. "single": Simple funcs, straightforward algos (<15 lines), no edge cases
2. "centralized": Algos with planning needs (DP/graphs/recursion), conditions, edge cases, multi-step
3. "decentralized": Parallel tasks, agents set next_role

**Roles:** single=["builder"], centralized=["planner","builder"](+checker), decentralized=2-3 no mgr
**Steps:** single=1, centralized=2-4, decentralized=2-4
**Rules:** single: entry="builder" mgr=null | centralized: mgr+entry="planner" | decentralized: mgr=null entry=1st.
**Decision:** Choose based on complexity. Use single for straightforward implementations, centralized when planning/testing adds value.
"""
    
    elif strategy == "multi_preference":
        # Moderately prefer multi-agent, aim for ~70-80% centralized
        topology_guide = """
**Topology:**
1. "single": Only trivial funcs (<5 lines), no logic, hardcoded returns
2. "centralized" (RECOMMENDED): Standard for most algos, conditions, transforms, edge cases
3. "decentralized": Parallel components, agents set next_role

**Roles:** single=["builder"], centralized=["planner","builder"](+checker), decentralized=2-3 no mgr
**Steps:** single=1, centralized=2-4, decentralized=2-4
**Rules:** single: entry="builder" mgr=null | centralized: mgr+entry="planner" | decentralized: mgr=null entry=1st.
**Decision:** Prefer centralized for better code quality and planning. Reserve single for truly trivial implementations.
"""
    
    elif strategy == "multi_aggressive":
        # Strongly prefer multi-agent for better quality, aim for >90% centralized
        topology_guide = """
**Topology:**
1. "single" (ALMOST NEVER): Only for hardcoded constants or trivial returns
2. "centralized" (MANDATORY): All algorithmic tasks, all logic, all transformations
3. "decentralized": Parallel independent components, agents set next_role

**Roles:** single=["builder"](avoid), centralized=ALWAYS["planner","builder","checker"], decentralized=3-4 parallel
**Steps:** single=1(rare), centralized=3-6, decentralized=3-6
**Rules:** single: entry="builder" mgr=null | centralized: mgr+entry="planner" | decentralized: mgr=null entry=1st.
**Decision:** Always use centralized for quality and robustness. Single only if implementation is completely trivial (e.g., return constant).
"""
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of: single_preference, balanced, multi_preference, multi_aggressive")
    
    return base_prompt + "\n" + topology_guide


def _normalize_role(value: Any) -> str:
    """Normalize role to lowercase string."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_topology_label(label: Dict[str, Any], available_roles: List[str]) -> Dict[str, Any]:
    if not isinstance(label, dict):
        label = {}

    topology = str(label.get("topology") or "").lower()
    if topology not in {"single", "centralized", "decentralized", "chain"}:
        topology = "single"

    roles = label.get("roles")
    manager_role = _normalize_role(label.get("manager_role"))
    entry_role = _normalize_role(label.get("entry_role"))
    max_steps = label.get("max_steps")

    # Backward-compat: {"roles": {"manager": "...", "workers": [...]}}
    if isinstance(roles, dict):
        mgr = _normalize_role(roles.get("manager"))
        workers = roles.get("workers") or []
        roles = []
        if mgr:
            roles.append(mgr)
        if isinstance(workers, list):
            roles.extend([_normalize_role(worker) for worker in workers])
        if manager_role is None:
            manager_role = mgr

    # Backward-compat: {"agent_id": "..."}
    if not roles:
        agent_id = _normalize_role(label.get("agent_id"))
        if agent_id:
            roles = [agent_id]

    if isinstance(roles, str):
        roles = [_normalize_role(roles)]
    if not isinstance(roles, list):
        roles = []

    roles = [_normalize_role(role) for role in roles]
    roles = [role for role in roles if role in available_roles]
    if not roles:
        roles = [available_roles[0]] if available_roles else ["builder"]

    if topology == "single":
        entry_role = entry_role or roles[0]
        if entry_role not in roles:
            entry_role = roles[0]
        roles = [entry_role]
        manager_role = None
        if max_steps is None:
            max_steps = 1
    elif topology == "decentralized":
        # Decentralized: no manager, agents work in parallel/sequentially with self-coordination
        manager_role = None
        entry_role = entry_role or roles[0]
        if entry_role not in roles:
            entry_role = roles[0]
        if max_steps is None:
            max_steps = min(6, max(2, len(roles)))
    elif topology == "centralized":
        # Centralized: manager coordinates workers
        if manager_role and manager_role not in roles:
            manager_role = None
        if not manager_role:
            manager_role = roles[0] if roles else None
        entry_role = entry_role or manager_role
        if max_steps is None:
            worker_count = max(1, len([r for r in roles if r != manager_role]))
            max_steps = min(3, worker_count + 1)
    else:
        # Other topologies (fallback)
        entry_role = entry_role or roles[0]
        if max_steps is None:
            max_steps = min(6, len(roles))

    return {
        "topology": topology,
        "roles": roles,
        "manager_role": manager_role,
        "entry_role": entry_role,
        "max_steps": int(max_steps) if max_steps else 1,
    }

def load_humaneval_problems(file_paths: List[str]) -> List[Dict[str, Any]]:
    problems = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    # Normalize fields
                    prompt_text = data.get("prompt", "")
                    task_id = data.get("task_id", "") or data.get("name", "")
                    entry_point = data.get("entry_point", "")
                    
                    if prompt_text:
                        problems.append({
                            "task_id": task_id,
                            "prompt": prompt_text,
                            "entry_point": entry_point,
                            "original_data": data
                        })
                except json.JSONDecodeError:
                    continue
    return problems

def generate_labels(
    problems: List[Dict[str, Any]],
    output_file: str,
    model: str,
    available_roles: List[str],
    strategy: str = "single_preference",
):
    # Initialize Client (Start your stronger model here, e.g. GPT-4)
    # Ensure LLM_API_KEY and LLM_BASE_URL are set in environment
    client = LLMClient(model=model)
    
    # Get system prompt based on strategy
    system_prompt = get_system_prompt(strategy)
    
    print(f"DEBUG: Using LLM API Base URL: {client.base_url}")
    print(f"Starting label generation for {len(problems)} problems using {model}...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, prob in enumerate(problems):
            task_text = f"Task: {prob['prompt']}"
            if prob['entry_point']:
                task_text += f"\nEntry point: {prob['entry_point']}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_text}
            ]

            try:
                print(f"[{i+1}/{len(problems)}] Processing {prob['task_id']}...", end="", flush=True)
                
                # Call Teacher Model
                # We use JSON mode if supported by the provider, otherwise normal
                response_text = client.chat(
                    messages=messages, 
                    temperature=0.1,
                    # response_format={"type": "json_object"} # Uncomment if using OpenAI GPT-4-1106+
                )
                
                # Parse JSON to ensure valid
                # Sometimes models wrap JSON in markdown ```json ... ```
                cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
                label_json = json.loads(cleaned_response)
                label_json = _normalize_topology_label(label_json, available_roles)

                # Construct Training Sample
                training_sample = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task_text},
                        {"role": "assistant", "content": json.dumps(label_json, ensure_ascii=False)}
                    ],
                    "sample_type": "humaneval_teacher_distilled",
                    "origin_task_id": prob['task_id']
                }

                f_out.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
                f_out.flush()
                print(" Done.")

            except Exception as e:
                print(f" Failed: {e}")

    print(f"Label generation complete. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Router Labels using a Teacher LLM")
    parser.add_argument("--data", type=str, required=True, help="Path to HumanEval file (e.g. Humaneval.jsonl)")
    parser.add_argument("--output", type=str, default="humaneval_router_labels.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Teacher model name")
    parser.add_argument(
        "--roles",
        type=str,
        default="builder,planner,checker,researcher",
        help="comma-separated available roles to constrain the label output",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="single_preference",
        choices=["single_preference", "balanced", "multi_preference", "multi_aggressive"],
        help="Agent topology preference strategy: "
             "single_preference (default, strongly prefer single agent), "
             "balanced (neutral, choose based on complexity), "
             "multi_preference (moderately prefer multi-agent), "
             "multi_aggressive (strongly prefer multi-agent collaboration)",
    )
    
    args = parser.parse_args()
    
    probs = load_humaneval_problems([args.data])
    roles = [role.lower() for role in _parse_list(args.roles)]
    generate_labels(probs, args.output, args.model, roles, args.strategy)
