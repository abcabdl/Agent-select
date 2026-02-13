from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional progress bar.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Set up logger
logger = logging.getLogger(__name__)

# Ensure `generation` is importable when running as a module or script.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.humaneval_postprocess import (
    _apply_stop_tokens,
    _build_program,
    _extract_code_block,
    _extract_code_from_results,
    _extract_function_signature,
    _extract_assertion_hint,
    _looks_like_assertion_error,
    _normalize_completion,
    _strip_redundant_def,
    _repair_assertion_completion_with_llm,
    _evaluate_with_postprocess_check,
    _run_python,
)

def _load_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def _load_solutions(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    records = _load_jsonl(path)
    solutions: Dict[str, str] = {}
    for record in records:
        name = record.get("name") or record.get("task_id")
        completion = record.get("completion") or record.get("solution") or record.get("code")
        if name and completion:
            solutions[str(name)] = str(completion)
    return solutions

def _build_prompt(prompt: str, entry_point: Optional[str], strict: bool, strict_text: Optional[str]) -> str:
    base = prompt or ""
    if not strict:
        return base
    if strict_text:
        suffix = strict_text.strip()
    else:
        entry_hint = f"(function name: {entry_point})" if entry_point else ""
        suffix = (
            "Output only the function body code with correct indentation. "
            "Do not repeat function definition, explanation, or markdown fences. "
            f"{entry_hint}\n"
            "Important constraints:\n"
            "1. MUST use exact parameter names from function signature (e.g., use lst, not data/items/input_string).\n"
            "2. MUST handle edge cases: empty input ([], '', None), single element, and boundary values (0, 1).\n"
            "3. Generate only direct solution code; no unrelated parsing/template boilerplate.\n"
            "4. Ensure logic is correct and testable.\n"
        )
    return f"{base}\n\n{suffix}\n"

def _auto_generate_solutions(
    tasks: List[dict],
    out_path: str,
    local_model_dir: str,
    local_lora_dir: Optional[str],
    local_device: Optional[str],
    local_dtype: str,
    local_use_4bit: bool,
    local_use_8bit: bool,
    max_new_tokens: int,
    temperature: float,
    strict_prompt: bool,
    strict_prompt_text: Optional[str],
    use_stop_tokens: bool,
) -> Dict[str, str]:
    from generation.local_llm import LocalLLMClient

    client = LocalLLMClient(
        model_path=local_model_dir,
        lora_path=local_lora_dir,
        device=local_device,
        dtype=local_dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_4bit=local_use_4bit,
        use_8bit=local_use_8bit,
    )

    solutions: Dict[str, str] = {}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for task in _iter_progress(tasks, desc="generate"):
            name = task.get("name") or task.get("task_id") or ""
            prompt = task.get("prompt") or ""
            entry_point = task.get("entry_point")
            stop_tokens = task.get("stop_tokens") or []
            prompt = _build_prompt(prompt, entry_point, strict_prompt, strict_prompt_text)
            messages = [{"role": "user", "content": prompt}]
            completion = client.chat(messages, temperature=temperature, max_tokens=max_new_tokens)
            completion = _extract_code_block(completion)
            completion = _strip_redundant_def(completion, entry_point)
            if use_stop_tokens:
                completion = _apply_stop_tokens(completion, stop_tokens)
            completion = _normalize_completion(completion, entry_point)
            solutions[str(name)] = completion
            f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
    return solutions

def _load_selections(log_path: Optional[str], registry) -> Dict[str, str]:
    if not log_path or not os.path.exists(log_path):
        return {}
    selections: Dict[str, str] = {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                role = record.get("role")
                selected = record.get("selected_main")
                if role and selected:
                    card = registry.get(selected)
                    name = getattr(card, "name", None) if card is not None else None
                    if name:
                        selections[role] = name
    except Exception:
        return selections
    return selections

def _auto_generate_solutions_orchestrator(
    tasks: List[dict],
    out_path: str,
    *,
    db_path: str,
    index_dir: str,
    dim: int,
    seed: int,
    roles: List[str],
    constraints: Dict[str, Dict[str, Any]],
    workflow_version: str,
    reranker_model: str,
    bandit_db: str,
    top_n: int,
    top_k: int,
    rerank_top_m: int,
    mmr_lambda: float,
    router_top_m: int,
    router_no_rerank: bool,
    dynamic_topology: bool,
    topology: str,
    topology_config: Optional[Dict[str, Any]],
    soft_connection: bool,
    max_steps: int,
    allow_unknown_roles: bool,
    reuse_role_selection: bool,
    reuse_same_role_agent_once: bool,
    local_model_dir: str,
    local_lora_dir: Optional[str],
    local_device: Optional[str],
    local_dtype: str,
    local_use_4bit: bool,
    local_use_8bit: bool,
    max_new_tokens: int,
    temperature: float,
    strict_prompt: bool,
    strict_prompt_text: Optional[str],
    use_stop_tokens: bool,
    code_model: Optional[str],
    code_base_url: Optional[str],
    code_api_key: Optional[str],
    code_timeout: float,
    code_retries: int,
    next_role_model: Optional[str],
    next_role_base_url: Optional[str],
    next_role_api_key: Optional[str],
    next_role_timeout: float,
    next_role_retries: int,
    embedder_kind: str,
    embedder_model: str,
    embedder_device: Optional[str],
    embedder_normalize: bool,
    include_tool_trace: bool,
    tool_only: bool,
    tool_timeout_s: float,
    mcts_dynamic_optimization: bool,
    mcts_iterations: int,
    mcts_rollout_depth: int,
    mcts_exploration: float,
    mcts_discount: float,
    mcts_max_candidates: int,
    max_attempts: int,
    baseline_fixed_bcb_mcts: bool,
    baseline_fixed_bcb_router_gpt4o: bool,
    dynamic_workflow_router_gpt4o: bool,
    force_role: Optional[str] = None,
    strict_roles: bool = True,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    from core.registry import SQLiteRegistry
    from execution.orchestrator import run_workflow
    from generation.llm_client import LLMClient
    from generation.local_llm import LocalLLMClient
    from retrieval.embedder import build_embedder
    from retrieval.faiss_index import HNSWIndex

    local_client = LocalLLMClient(
        model_path=local_model_dir,
        lora_path=local_lora_dir,
        device=local_device,
        dtype=local_dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_4bit=local_use_4bit,
        use_8bit=local_use_8bit,
    )
    
    # Propagate CLI args to environment variables for tools that rely on os.getenv
    if code_base_url:
        os.environ["LLM_API_BASE"] = code_base_url
    if code_api_key:
        os.environ["LLM_API_KEY"] = code_api_key

    code_client: Optional[LLMClient] = None
    if code_model or code_base_url or code_api_key:
        code_client = LLMClient(
            api_key=code_api_key or None,
            base_url=code_base_url or None,
            model=code_model or None,
            timeout_s=code_timeout,
            max_retries=code_retries,
        )
    
    next_role_client: Optional[LLMClient] = None
    if next_role_model or next_role_base_url or next_role_api_key:
        next_role_client = LLMClient(
            api_key=next_role_api_key or None,
            base_url=next_role_base_url or None,
            model=next_role_model or None,
            timeout_s=next_role_timeout,
            max_retries=next_role_retries,
        )

    if baseline_fixed_bcb_mcts and baseline_fixed_bcb_router_gpt4o:
        raise SystemExit(
            "Only one baseline switch can be enabled at a time: "
            "--baseline_fixed_bcb_mcts or --baseline_fixed_bcb_router_gpt4o"
        )
    if dynamic_workflow_router_gpt4o and (
        baseline_fixed_bcb_mcts or baseline_fixed_bcb_router_gpt4o
    ):
        raise SystemExit(
            "--dynamic_workflow_router_gpt4o cannot be combined with "
            "--baseline_fixed_bcb_mcts or --baseline_fixed_bcb_router_gpt4o"
        )

    baseline_fixed_topology_config: Dict[str, Any] = {
        "topology": "chain",
        "roles": ["builder", "checker", "builder"],
        "entry_role": "builder",
        "max_steps": 3,
        "flow_type": "sequential",
    }

    index = HNSWIndex.load(os.path.join(index_dir, "faiss.index"))
    embedder = build_embedder(
        kind=embedder_kind,
        dim=dim,
        seed=seed,
        model_name=embedder_model,
        device=embedder_device,
        normalize=embedder_normalize,
    )
    if embedder.dim != index.dim:
        raise ValueError(
            f"Embedder dim {embedder.dim} != index dim {index.dim}. "
            "Rebuild index with matching embedder/model."
        )

    solutions: Dict[str, str] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with SQLiteRegistry(db_path) as registry, open(out_path, "w", encoding="utf-8") as f:
        for task in _iter_progress(tasks, desc="generate"):
            task_started = time.perf_counter()
            name = task.get("name") or task.get("task_id") or ""
            original_prompt = task.get("prompt") or ""  # Save original prompt without suffix
            entry_point = task.get("entry_point")
            stop_tokens = task.get("stop_tokens") or []
            test = task.get("test") or ""  # Extract test code from task
            
            # Extract function signature for better context
            func_sig = _extract_function_signature(original_prompt, entry_point)
            
            prompt = _build_prompt(original_prompt, entry_point, strict_prompt, strict_prompt_text)
            
            # Add explicit parameter names to task text
            param_hint = ""
            if func_sig and func_sig.get("parameters"):
                params = func_sig["parameters"]
                param_hint = f"\nParameter names: {', '.join(params)}. Use these exact names in code.\n"

            task_text = (
                f"{prompt}\n\n"
                "Put the implementation in JSON field `code_or_commands` as runnable function-body code only.\n"
                "Requirements:\n"
                "1. Use exact parameter names from the function signature (no generic names like data/items/input_string).\n"
                "2. Handle edge cases: empty input ([], '', None), single element, and boundary values (0, 1).\n"
                "3. Generate direct solution code only; avoid unrelated parsing/template boilerplate.\n"
                "4. Ensure the logic is correct.\n"
                f"{param_hint}"
            )

            # Build task context with function signature and test information
            # IMPORTANT: Use original_prompt (without Chinese suffix) for testing!
            task_context = {
                "entry_point": entry_point,
                "function_signature": func_sig,
                "prompt": original_prompt,  # Use original prompt for test execution
                "test": test,
                "stop_tokens": stop_tokens,
            } if entry_point else None

            effective_dynamic_topology = dynamic_topology
            effective_topology = topology
            effective_topology_config = topology_config
            effective_max_steps = max_steps
            effective_mcts_dynamic_optimization = mcts_dynamic_optimization
            effective_reuse_same_role_agent_once = reuse_same_role_agent_once
            effective_router_llm_client: Optional[LLMClient] = local_client
            effective_next_role_llm_client: Optional[LLMClient] = next_role_client or local_client
            effective_meta_router_llm_client: Optional[LLMClient] = local_client

            if baseline_fixed_bcb_mcts or baseline_fixed_bcb_router_gpt4o:
                effective_dynamic_topology = True
                effective_topology = "chain"
                effective_topology_config = dict(baseline_fixed_topology_config)
                effective_max_steps = 3
                effective_reuse_same_role_agent_once = True
                effective_next_role_llm_client = None
                effective_meta_router_llm_client = None

            if baseline_fixed_bcb_mcts:
                effective_mcts_dynamic_optimization = True
                # router llm is not used on MCTS path; keep local as harmless default
                effective_router_llm_client = local_client
            elif baseline_fixed_bcb_router_gpt4o:
                # old embedding retrieval + router LLM final choice
                effective_mcts_dynamic_optimization = False
                router_remote_client = next_role_client or code_client
                if router_remote_client is None:
                    raise SystemExit(
                        "--baseline_fixed_bcb_router_gpt4o requires a remote OpenAI-compatible client. "
                        "Please set --next_role_model/--next_role_base_url/--next_role_api_key "
                        "or --code_model/--code_base_url/--code_api_key."
                    )
                effective_router_llm_client = router_remote_client
            elif dynamic_workflow_router_gpt4o:
                # dynamic workflow + embedding retrieval + remote router LLM (e.g. GPT-4o) final choice
                effective_dynamic_topology = True
                effective_mcts_dynamic_optimization = False
                router_remote_client = next_role_client or code_client
                if router_remote_client is None:
                    raise SystemExit(
                        "--dynamic_workflow_router_gpt4o requires a remote OpenAI-compatible client. "
                        "Please set --next_role_model/--next_role_base_url/--next_role_api_key "
                        "or --code_model/--code_base_url/--code_api_key."
                    )
                effective_router_llm_client = router_remote_client

            workflow_started = time.perf_counter()
            result = run_workflow(
                task_text=task_text,
                roles=roles,
                constraints_per_role=constraints,
                workflow_version=workflow_version,
                registry=registry,
                index=index,
                embedder=embedder,
                top_n=top_n,
                top_k=top_k,
                rerank_top_m=rerank_top_m,
                mmr_lambda=mmr_lambda,
                reranker_model_path=reranker_model,
                bandit_db_path=bandit_db,
                max_attempts=max_attempts,
                llm_client=code_client or local_client,
                router_llm_client=effective_router_llm_client,
                router_top_m=router_top_m,
                task_context=task_context,
                router_no_rerank=router_no_rerank,
                dynamic_topology=effective_dynamic_topology,
                topology=effective_topology,
                topology_config=effective_topology_config,
                meta_router_llm_client=effective_meta_router_llm_client,
                next_role_llm_client=effective_next_role_llm_client,
                soft_connection=soft_connection,
                max_steps=effective_max_steps,
                allow_unknown_roles=allow_unknown_roles,
                reuse_role_selection=reuse_role_selection,
                reuse_same_role_agent_once=effective_reuse_same_role_agent_once,
                tool_only=tool_only,
                tool_timeout_s=tool_timeout_s,
                mcts_dynamic_optimization=effective_mcts_dynamic_optimization,
                mcts_iterations=mcts_iterations,
                mcts_rollout_depth=mcts_rollout_depth,
                mcts_exploration=mcts_exploration,
                mcts_discount=mcts_discount,
                mcts_max_candidates=mcts_max_candidates,
                force_role=force_role,
                strict_roles=strict_roles,
            )
            workflow_elapsed = time.perf_counter() - workflow_started

            results = result.get("results") or {}
            tool_exec = result.get("tool_exec")
            tool_trace = tool_exec if include_tool_trace else None
            
            # Extract topology and agent selection info FIRST (before checking completion)
            topology_info = result.get("topology") or {"topology": "linear", "roles": roles}
            selections = _load_selections(result.get("log_path"), registry)
            task_meta = {"topology": topology_info, "selected_agents": selections}
            if include_tool_trace and tool_trace is not None:
                task_meta["tool_trace"] = json.loads(json.dumps(tool_trace, ensure_ascii=True, default=str))
            task_meta["workflow_time_s"] = round(workflow_elapsed, 3)
            
            # Include tool_exec in extraction so we can recover code even if role outputs
            # were validated into schemas that drop code_or_commands (e.g., planner).
            extraction_payload = dict(results)
            if tool_exec:
                extraction_payload["tool_exec"] = tool_exec
            completion = _extract_code_from_results(
                extraction_payload,
                entry_point,
                param_names=(func_sig.get("parameters") if isinstance(func_sig, dict) else None),
            )
            if completion is None or not completion.strip():
                # Record meta even if no code was generated (helps debugging)
                task_meta["extraction_failed"] = True
                task_meta["results_summary"] = {k: type(v).__name__ for k, v in results.items()}
                # Try to extract from tool_trace as last resort
                tool_trace = result.get("tool_exec")
                if tool_trace and isinstance(tool_trace, dict):
                    # Reuse the main extractor on tool_exec for robust nested output handling.
                    completion = _extract_code_from_results(
                        {"tool_exec": tool_trace},
                        entry_point,
                        param_names=(func_sig.get("parameters") if isinstance(func_sig, dict) else None),
                    )
                    if completion:
                        task_meta["extracted_from"] = "tool_exec"
                
                # Final fallback: Use LLM to generate code directly if all tools failed
                if not completion or not completion.strip():
                    logger.warning(f"All tools failed for {name}, using LLM fallback")
                    task_meta["used_llm_fallback"] = True
                    try:
                        # Simple fallback with enhanced prompt (no failure context to avoid confusion)
                        llm_prompt = (
                            f"{prompt}\n\n"
                            "You MUST follow these output rules exactly:\n"
                            "1. Output ONLY raw function body code with 4-space indentation\n"
                            "2. Do NOT include labels (python/json), markdown fences, titles, explanations, or comments\n"
                            "3. CRITICAL: Use ONLY the actual parameter names from the function signature (e.g., if parameter is 'lst', use 'lst' NOT 'data' or 'items')\n"
                            "4. Do NOT use generic variable names like: data, items, input_string, inputs, arr (unless they are the actual parameter names)\n"
                            "5. MUST handle edge cases: empty inputs ([], '', None), single elements, boundary values (0, 1)\n"
                            "6. Generate ONLY the function body that directly solves the problem, NOT generic parsing/processing templates\n"
                            "7. If asked for a function body, do NOT include def/class/import/main/docstrings\n"
                            "8. Empty/whitespace output is invalid\n"
                        )
                        llm_response = (code_client or local_client).chat(
                            [{"role": "user", "content": llm_prompt}],
                            temperature=0.2,
                            max_tokens=600,
                        )
                        completion = _extract_code_block(str(llm_response))
                        if completion and completion.strip():
                            task_meta["extracted_from"] = "llm_fallback"
                    except Exception as e:
                        logger.error(f"LLM fallback failed: {e}")
                
                if not completion or not completion.strip():
                    # Use empty completion to allow test to run and show error
                    completion = ""
            
            if use_stop_tokens:
                completion = _apply_stop_tokens(completion, stop_tokens)
            completion = _normalize_completion(completion, entry_point)

            # Assertion-only logic repair is enabled only for dynamic workflow + MCTS mode
            # (explicitly excluding fixed-baseline modes).
            enable_assertion_logic_repair = (
                effective_dynamic_topology
                and effective_mcts_dynamic_optimization
                and not baseline_fixed_bcb_mcts
                and not baseline_fixed_bcb_router_gpt4o
            )
            if (
                enable_assertion_logic_repair
                and completion
                and completion.strip()
                and entry_point
                and test
            ):
                repair_client = code_client or local_client
                if repair_client is not None:
                    eval_timeout_s = max(5.0, float(tool_timeout_s))
                    current_completion = completion
                    repair_logs: List[Dict[str, Any]] = []
                    for attempt in range(2):
                        candidate_program = _build_program(
                            original_prompt, current_completion, test, entry_point
                        )
                        ok_pre, msg_pre = _run_python(candidate_program, timeout_s=eval_timeout_s)
                        if ok_pre or (not _looks_like_assertion_error(msg_pre)):
                            break

                        repaired_completion = _repair_assertion_completion_with_llm(
                            llm_client=repair_client,
                            prompt=original_prompt,
                            completion=current_completion,
                            test_code=test,
                            entry_point=entry_point,
                            error_message=msg_pre,
                            failure_context=(
                                f"assertion_hint={_extract_assertion_hint(msg_pre)}\n"
                                f"error={msg_pre}"
                            ),
                        )
                        if not repaired_completion or repaired_completion == current_completion:
                            break

                        repaired_program = _build_program(
                            original_prompt, repaired_completion, test, entry_point
                        )
                        ok_repaired, msg_repaired = _run_python(
                            repaired_program, timeout_s=eval_timeout_s
                        )
                        repair_logs.append(
                            {
                                "attempt": attempt + 1,
                                "repair_type": "assertion_logic",
                                "success": ok_repaired,
                                "original_error": msg_pre,
                                "final_error": "" if ok_repaired else msg_repaired,
                            }
                        )
                        if ok_repaired:
                            completion = repaired_completion
                            break
                        current_completion = repaired_completion
                    if repair_logs:
                        task_meta["assertion_logic_repairs"] = repair_logs
            solutions[str(name)] = completion
            f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
            task_meta["generation_time_s"] = round(time.perf_counter() - task_started, 3)
            meta[str(name)] = task_meta

    return solutions, meta

def evaluate(
    tasks_path: str,
    solutions_path: Optional[str],
    timeout_s: float,
    postprocess_rounds: int = 1,
) -> Dict[str, object]:
    tasks = _load_jsonl(tasks_path)
    solutions = _load_solutions(solutions_path)
    passed = 0
    results: List[dict] = []
    for task in tasks:
        eval_started = time.perf_counter()
        name = task.get("name") or task.get("task_id") or ""
        prompt = task.get("prompt") or ""
        test_code = task.get("test") or ""
        entry_point = task.get("entry_point")
        completion = task.get("completion")
        if completion is None:
            completion = solutions.get(str(name), "")
        ok, message, postprocess_info = _evaluate_with_postprocess_check(
            prompt,
            str(completion),
            test_code,
            entry_point,
            timeout_s=timeout_s,
            max_postprocess_rounds=postprocess_rounds,
        )
        if ok:
            passed += 1
        record = {"name": name, "ok": ok, "error": message}
        record["eval_time_s"] = round(time.perf_counter() - eval_started, 3)
        if postprocess_info:
            record["postprocess_check"] = postprocess_info
        results.append(record)
    total = len(results)
    pass_rate = passed / total if total else 0.0
    return {"total": total, "passed": passed, "pass_rate": pass_rate, "results": results}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HumanEval-style tasks")
    parser.add_argument("--tasks", required=True, type=str, help="JSONL tasks file (prompt/test)")
    parser.add_argument("--solutions", default="", type=str, help="JSONL solutions file (name+completion)")
    parser.add_argument("--timeout", default=5.0, type=float, help="timeout seconds per task")
    # Backward-compatible alias for older scripts/commands.
    parser.add_argument("--time_out", dest="timeout", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--out", default="", type=str, help="optional output JSON path")
    parser.add_argument("--auto_generate", action="store_true", help="generate solutions if missing")
    parser.add_argument("--local_model_dir", type=str, default=None, help="local base model path")
    parser.add_argument("--local_lora_dir", type=str, default=None, help="local LoRA adapter path")
    parser.add_argument("--local_device", type=str, default=None)
    parser.add_argument("--local_dtype", type=str, default="auto")
    parser.add_argument("--local_use_4bit", action="store_true")
    parser.add_argument("--local_use_8bit", action="store_true")
    parser.add_argument("--code_model", type=str, default="", help="remote code model (OpenAI-compatible)")
    parser.add_argument("--code_base_url", type=str, default="", help="remote code API base URL")
    parser.add_argument("--code_api_key", type=str, default="", help="remote code API key (optional; uses env)")
    parser.add_argument("--code_timeout", type=float, default=60.0, help="remote code API timeout seconds")
    parser.add_argument("--code_retries", type=int, default=2, help="remote code API retry count")
    parser.add_argument("--next_role_model", type=str, default="", help="remote next-role model (OpenAI-compatible)")
    parser.add_argument("--next_role_base_url", type=str, default="", help="remote next-role API base URL")
    parser.add_argument("--next_role_api_key", type=str, default="", help="remote next-role API key (optional; uses env)")
    parser.add_argument("--next_role_timeout", type=float, default=60.0, help="remote next-role API timeout seconds")
    parser.add_argument("--next_role_retries", type=int, default=2, help="remote next-role API retry count")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no_strict_prompt", action="store_true", help="disable strict prompt suffix")
    parser.add_argument("--strict_prompt_text", type=str, default="", help="custom strict prompt suffix")
    parser.add_argument("--use_stop_tokens", action="store_true", help="apply stop_tokens during generation")
    parser.add_argument("--use_orchestrator", action="store_true", help="generate solutions via orchestrator")
    parser.add_argument("--db", default="demo_registry.sqlite", type=str)
    parser.add_argument("--index_dir", default="index", type=str)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--embedder", default="sentence-transformer", type=str, help="dummy|sentence-transformer")
    parser.add_argument(
        "--embedder_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        type=str,
        help="SentenceTransformer model name",
    )
    parser.add_argument("--embedder_device", default=None, type=str)
    parser.add_argument("--embedder_normalize", action="store_true")
    parser.add_argument("--include_tool_trace", action="store_true")
    parser.add_argument("--tool_only", action="store_true", help="use tools only; no LLM generation")
    parser.add_argument("--tool_timeout", type=float, default=5.0, help="tool execution timeout seconds")
    parser.add_argument(
        "--postprocess_rounds",
        type=int,
        default=1,
        help="postprocess repair rounds after initial failure",
    )
    parser.add_argument("--mcts_dynamic_optimization", action="store_true")
    parser.add_argument("--mcts_iterations", type=int, default=64)
    parser.add_argument("--mcts_rollout_depth", type=int, default=4)
    parser.add_argument("--mcts_exploration", type=float, default=1.414)
    parser.add_argument("--mcts_discount", type=float, default=0.95)
    parser.add_argument("--mcts_max_candidates", type=int, default=8)
    parser.add_argument(
        "--baseline_fixed_bcb_mcts",
        action="store_true",
        help="baseline: fixed workflow builder->checker->builder with MCTS agent selection",
    )
    parser.add_argument(
        "--baseline_fixed_bcb_router_gpt4o",
        action="store_true",
        help="baseline: fixed workflow builder->checker->builder with embedding retrieval + GPT-4o router final choice",
    )
    parser.add_argument(
        "--dynamic_workflow_router_gpt4o",
        action="store_true",
        help=(
            "one-shot switch: enable dynamic workflow and use embedding retrieval + remote router LLM "
            "(e.g. GPT-4o) for final agent selection"
        ),
    )
    parser.add_argument("--roles", default="code-generation,code-planner,code-testing,code-refactoring", type=str)
    parser.add_argument("--constraints", default="", type=str, help="JSON string per role")
    parser.add_argument("--workflow_version", default="v1", type=str)
    parser.add_argument("--reranker_model", default="models/reranker.json", type=str)
    parser.add_argument("--bandit_db", default="models/bandit.sqlite", type=str)
    parser.add_argument("--top_n", default=20, type=int)
    parser.add_argument("--top_k", default=5, type=int)
    parser.add_argument("--rerank_top_m", default=3, type=int)
    parser.add_argument("--mmr_lambda", default=0.5, type=float)
    parser.add_argument("--router_top_m", default=5, type=int)
    parser.add_argument("--router_no_rerank", action="store_true")
    parser.add_argument("--dynamic_topology", action="store_true")
    parser.add_argument("--topology", default="linear", type=str)
    parser.add_argument("--topology_config", default="", type=str)
    parser.add_argument("--soft_connection", action="store_true")
    parser.add_argument("--max_steps", default=6, type=int)
    parser.add_argument("--max_attempts", default=3, type=int, help="max retries per role/tool execution")
    parser.add_argument("--allow_unknown_roles", action="store_true")
    parser.add_argument("--no_reuse_role_selection", action="store_true")
    parser.add_argument(
        "--reuse_same_role_agent_once",
        action="store_true",
        help="within one query, search/select each role once and reuse the same agent for repeated role steps",
    )
    parser.add_argument("--force_role", default="", type=str, help="force all queries to use this role (skip Router)")
    return parser.parse_args()

def _iter_progress(items: List[dict], desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, total=len(items))

def main() -> None:
    args = parse_args()
    if args.dynamic_workflow_router_gpt4o:
        args.use_orchestrator = True
        args.dynamic_topology = True
        args.mcts_dynamic_optimization = False
        args.baseline_fixed_bcb_mcts = False
        args.baseline_fixed_bcb_router_gpt4o = False

    tasks = _load_jsonl(args.tasks)
    effective_postprocess_rounds = args.postprocess_rounds
    if (args.mcts_dynamic_optimization or args.baseline_fixed_bcb_mcts) and effective_postprocess_rounds < 2:
        effective_postprocess_rounds = 2
    solutions_path = args.solutions or None
    if not solutions_path and args.out:
        out_path = Path(args.out)
        solutions_path = str(out_path.with_suffix(".solutions.jsonl"))
    solutions: Dict[str, str] = {}
    meta_by_task: Dict[str, Dict[str, Any]] = {}
    topology_config = None
    if args.topology_config:
        if os.path.exists(args.topology_config):
            with open(args.topology_config, "r", encoding="utf-8") as f:
                topology_config = json.load(f)
        else:
            topology_config = json.loads(args.topology_config)

    if solutions_path and os.path.exists(solutions_path):
        solutions = _load_solutions(solutions_path)
    else:
        should_generate = args.auto_generate or bool(args.local_model_dir)
        if should_generate:
            if not args.local_model_dir:
                raise SystemExit("--auto_generate requires --local_model_dir")
            if not solutions_path:
                solutions_path = args.tasks + ".solutions.jsonl"
            if args.use_orchestrator:
                roles = [item.strip() for item in args.roles.split(",") if item.strip()]
                constraints = json.loads(args.constraints) if args.constraints else {}
                solutions, meta_by_task = _auto_generate_solutions_orchestrator(
                    tasks=tasks,
                    out_path=solutions_path,
                    db_path=args.db,
                    index_dir=args.index_dir,
                    dim=args.dim,
                    seed=args.seed,
                    roles=roles or ["builder"],
                    constraints=constraints,
                    workflow_version=args.workflow_version,
                    reranker_model=args.reranker_model,
                    bandit_db=args.bandit_db,
                    top_n=args.top_n,
                    top_k=args.top_k,
                    rerank_top_m=args.rerank_top_m,
                    mmr_lambda=args.mmr_lambda,
                    router_top_m=args.router_top_m,
                    router_no_rerank=args.router_no_rerank,
                    dynamic_topology=args.dynamic_topology,
                    topology=args.topology,
                    topology_config=topology_config,
                    soft_connection=args.soft_connection,
                    max_steps=args.max_steps,
                    max_attempts=args.max_attempts,
                    allow_unknown_roles=args.allow_unknown_roles,
                    reuse_role_selection=not args.no_reuse_role_selection,
                    reuse_same_role_agent_once=args.reuse_same_role_agent_once,
                    local_model_dir=args.local_model_dir,
                    local_lora_dir=args.local_lora_dir,
                    local_device=args.local_device,
                    local_dtype=args.local_dtype,
                    local_use_4bit=args.local_use_4bit,
                    local_use_8bit=args.local_use_8bit,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    strict_prompt=not args.no_strict_prompt,
                    strict_prompt_text=args.strict_prompt_text or None,
                    use_stop_tokens=args.use_stop_tokens,
                    code_model=args.code_model or None,
                    code_base_url=args.code_base_url or None,
                    code_api_key=args.code_api_key or None,
                    code_timeout=args.code_timeout,
                    code_retries=args.code_retries,
                    next_role_model=args.next_role_model or None,
                    next_role_base_url=args.next_role_base_url or None,
                    next_role_api_key=args.next_role_api_key or None,
                    next_role_timeout=args.next_role_timeout,
                    next_role_retries=args.next_role_retries,
                    embedder_kind=args.embedder,
                    embedder_model=args.embedder_model,
                    embedder_device=args.embedder_device,
                    embedder_normalize=args.embedder_normalize,
                    include_tool_trace=args.include_tool_trace,
                    tool_only=args.tool_only,
                    tool_timeout_s=args.tool_timeout,
                    mcts_dynamic_optimization=args.mcts_dynamic_optimization,
                    mcts_iterations=args.mcts_iterations,
                    mcts_rollout_depth=args.mcts_rollout_depth,
                    mcts_exploration=args.mcts_exploration,
                    mcts_discount=args.mcts_discount,
                    mcts_max_candidates=args.mcts_max_candidates,
                    baseline_fixed_bcb_mcts=args.baseline_fixed_bcb_mcts,
                    baseline_fixed_bcb_router_gpt4o=args.baseline_fixed_bcb_router_gpt4o,
                    dynamic_workflow_router_gpt4o=args.dynamic_workflow_router_gpt4o,
                    force_role=args.force_role or None,
                )
            else:
                solutions = _auto_generate_solutions(
                    tasks=tasks,
                    out_path=solutions_path,
                    local_model_dir=args.local_model_dir,
                    local_lora_dir=args.local_lora_dir,
                    local_device=args.local_device,
                    local_dtype=args.local_dtype,
                    local_use_4bit=args.local_use_4bit,
                    local_use_8bit=args.local_use_8bit,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    strict_prompt=not args.no_strict_prompt,
                    strict_prompt_text=args.strict_prompt_text or None,
                    use_stop_tokens=args.use_stop_tokens,
                )
        elif solutions_path:
            raise SystemExit(f"Solutions file not found: {solutions_path}")

    passed = 0
    results: List[dict] = []
    for task in _iter_progress(tasks, desc="evaluate"):
        eval_started = time.perf_counter()
        name = task.get("name") or task.get("task_id") or ""
        prompt = task.get("prompt") or ""
        test_code = task.get("test") or ""
        entry_point = task.get("entry_point")
        completion = task.get("completion")
        if completion is None:
            completion = solutions.get(str(name), "")
        ok, message, postprocess_info = _evaluate_with_postprocess_check(
            prompt,
            str(completion),
            test_code,
            entry_point,
            timeout_s=args.timeout,
            max_postprocess_rounds=effective_postprocess_rounds,
        )
        if ok:
            passed += 1
        record = {"name": name, "ok": ok, "error": message}
        record["eval_time_s"] = round(time.perf_counter() - eval_started, 3)
        meta = meta_by_task.get(str(name))
        if meta:
            record.update(meta)
        if postprocess_info:
            record["postprocess_check"] = postprocess_info
        results.append(record)
    total = len(results)
    pass_rate = passed / total if total else 0.0
    report = {"total": total, "passed": passed, "pass_rate": pass_rate, "results": results}
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=True))
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)

if __name__ == "__main__":
    main()
