from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
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
    """Unwrap code from JSON string format like '{"code_or_commands": "..."}'."""
    if not text or "{" not in text:
        return text
    try:
        # Try to find and parse JSON structure
        start_idx = text.find("{")
        if start_idx == -1:
            return text
        # Extract potential JSON
        json_text = text[start_idx:]
        # Try to parse as JSON
        data = json.loads(json_text)
        if isinstance(data, dict):
            # Look for code in common keys
            for key in ["code_or_commands", "code", "solution", "output"]:
                if key in data:
                    value = data[key]
                    if isinstance(value, str) and value.strip():
                        return value
                    elif isinstance(value, dict):
                        # Nested structure
                        for nested_key in ["code_or_commands", "code", "solution"]:
                            if nested_key in value and isinstance(value[nested_key], str):
                                return value[nested_key]
    except (json.JSONDecodeError, ValueError):
        pass
    return text


def _extract_code_block(text: str) -> str:
    if not text:
        return text
    
    # First try to unwrap JSON-formatted code
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
    if lines and lines[0].lstrip().startswith(("def ", "class ", "@")):
        return "\n".join(lines)

    # Compute smallest positive indent (ignoring entirely blank lines); use it to dedent uneven bodies.
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


def _extract_function_signature(prompt: str, entry_point: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract function signature (name and parameters) from prompt."""
    if not prompt or not entry_point:
        return None
    
    import re
    # Match: def function_name(param1, param2, ...): including multiline signatures
    # Use DOTALL to handle signatures that span multiple lines
    pattern = rf"def\s+{re.escape(entry_point)}\s*\((.*?)\)\s*(?:->.*?)?:"
    match = re.search(pattern, prompt, re.DOTALL)
    if not match:
        return None
    
    params_str = match.group(1).strip()
    if not params_str:
        return {"function_name": entry_point, "parameters": []}
    
    # Parse parameters (simple parsing, doesn't handle complex annotations)
    params = []
    # Remove newlines and extra spaces
    params_str = " ".join(params_str.split())
    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue
        # Remove type annotations like ": List[int]" or "= default_value"
        param_name = param.split(":")[0].split("=")[0].strip()
        if param_name and param_name not in ("*", "**"):
            params.append(param_name)
    
    return {"function_name": entry_point, "parameters": params}


def _build_prompt(prompt: str, entry_point: Optional[str], strict: bool, strict_text: Optional[str]) -> str:
    base = prompt or ""
    if not strict:
        return base
    if strict_text:
        suffix = strict_text.strip()
    else:
        entry_hint = f"（函数名 {entry_point}）" if entry_point else ""
        suffix = (
            "请只输出函数体代码，保持缩进，不要重复函数定义，不要解释，不要 markdown/代码块。"
            f"{entry_hint}\n"
            "重要约束:\n"
            "1. 必须使用函数签名中的实际参数名(如果参数是 lst 就用 lst,不要用 data/items/input_string 等通用名)\n"
            "2. 必须处理边界情况: 空输入([], '', None), 单元素, 边界值(0, 1)\n"
            "3. 只生成直接解决问题的函数体,不要生成无关的解析/处理模板代码\n"
            "4. 确保代码逻辑正确,能通过所有测试用例"
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


def _is_tool_error(error_obj: Any) -> bool:
    """Check if error indicates tool execution failure (not code logic error)."""
    if not error_obj:
        return False
    if isinstance(error_obj, dict):
        error_code = error_obj.get("code", "")
        error_msg = error_obj.get("message", "")
        # Tool-level errors that should disqualify the result
        fatal_errors = [
            "ImportError",
            "SyntaxError", 
            "IndentationError",
            "TabError",
            "NameError",
        ]
        if error_code in fatal_errors:
            return True
        # Also check message content
        if isinstance(error_msg, str):
            if "Import blocked" in error_msg:
                return True
            if "unterminated string literal" in error_msg:
                return True
    return False


def _is_fallback(text: str) -> bool:
    """Check if text is a fallback/placeholder code (like 'return None')."""
    if not text:
        return True
    
    # Check raw text for common indented patterns first (before stripping)
    # Handle "return None" with various indentation levels
    raw_stripped_lines = [line for line in text.split('\n') if line.strip()]
    if len(raw_stripped_lines) == 1:
        line_content = raw_stripped_lines[0].strip().lower()
        if line_content in ("none", "return none", "return null", "null"):
            return True
    
    # Strip and check for empty
    stripped = text.strip()
    if not stripped:
        return True
    
    # Normalize for case-insensitive comparison
    normalized = stripped.lower()
    
    # Check if it's exactly "return None" or just "None"
    if normalized in ("none", "return none", "return null", "null"):
        return True
    
    # Check if the ENTIRE text is just a single return statement with None
    # This handles multi-line strings that only contain "return None"
    lines = [line.strip() for line in stripped.split('\n') if line.strip()]
    if len(lines) == 1 and lines[0].lower() in ("return none", "return null", "none", "null"):
        return True
    
    # Also check for lines with only whitespace + return None
    # e.g., "    return None" or "\treturn None"
    if all(line.strip().lower() in ("return none", "return null", "none", "null", "") for line in text.split('\n')):
        # Make sure at least one line has the return statement
        if any(line.strip().lower() in ("return none", "return null") for line in text.split('\n')):
            return True
    
    # Check for AI refusal patterns (these appear at START of text)
    refusal_patterns = [
        "i'm sorry",
        "i need more",
        "could you please",
        "please provide",
        "more specific",
        "more information",
        "i apologize",
        "i cannot",
    ]
    for pattern in refusal_patterns:
        if normalized.startswith(pattern):
            return True
    
    # Check for stdin template code (these are very specific patterns)
    if "sys.stdin.read" in text or "input(" in text:
        # Only flag if it's clearly a template (has both import and read)
        if "import sys" in text and "sys.stdin" in text:
            return True
    
    # Check for completely unrelated helper functions
    # These are signs of wrong code extraction
    unrelated_patterns = [
        "def is_sorted_recursive",  # Wrong function extracted
        "def helper",
        "def check_",
    ]
    for pattern in unrelated_patterns:
        if pattern in text:
            return True
    
    return False


def _extract_code_from_results(results: Dict[str, Any], entry_point: Optional[str]) -> Optional[str]:
    def _is_irrelevant(text: str) -> bool:
        if not entry_point:
            return False
        # If another function is being defined and it's not the target entry point, skip it.
        if "def " in text and entry_point not in text:
            return True
        return False

    def _normalize(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            # Try to unwrap JSON-formatted strings
            unwrapped = _unwrap_json_code(value)
            result = unwrapped if unwrapped != value else value
            # Skip if it's a fallback pattern
            if result and not _is_fallback(result):
                return result
            return None
        if isinstance(value, dict):
            # CRITICAL: Skip if ok=False or has error
            if value.get("ok") is False:
                return None
            # Check for tool-level fatal errors first
            if "error" in value and _is_tool_error(value["error"]):
                return None
            if "error" in value and value["error"]:
                return None
            
            # Only extract from successful tool executions with ok=True
            if "ok" in value and value["ok"] is True:
                # Look for code in output first
                if "output" in value and isinstance(value["output"], dict):
                    # Support nested output.output.code
                    if "output" in value["output"] and isinstance(value["output"]["output"], dict):
                        inner = value["output"]["output"]
                        for key in ["code_or_commands", "code", "solution"]:
                            if key in inner and isinstance(inner[key], str):
                                code = inner[key]
                                if code and code.strip() and not _is_fallback(code):
                                    return _unwrap_json_code(code)
                    for key in ["code_or_commands", "code", "solution"]:
                        if key in value["output"] and isinstance(value["output"][key], str):
                            code = value["output"][key]
                            if code and code.strip() and not _is_fallback(code):
                                return _unwrap_json_code(code)
            
            # Then check direct keys (for backward compatibility)
            for key in ["code_or_commands", "code", "solution"]:
                if key in value and isinstance(value[key], str):
                    code = value[key]
                    if code and code.strip() and not _is_fallback(code):
                        return _unwrap_json_code(code)
            
            # Recurse into nested result
            if "result" in value:
                nested = _normalize(value.get("result"))
                if nested:
                    return nested
        return None

    def _gather_code_strings(obj: Any) -> List[str]:
        collected: List[str] = []
        if obj is None:
            return collected
        if isinstance(obj, list):
            for item in obj:
                collected.extend(_gather_code_strings(item))
            return collected
        if isinstance(obj, dict):
            # Check if this is a tool execution result
            has_error = "error" in obj and obj["error"]
            has_ok = "ok" in obj
            
            # If it's a tool result with ok=True, prioritize it
            if has_ok and obj.get("ok") is True and not has_error:
                normalized = _normalize(obj)
                if normalized and normalized.strip():
                    collected.append(normalized)
            # If it has error but also has output, try to extract
            elif has_error and "output" in obj:
                normalized = _normalize(obj)
                if normalized and normalized.strip():
                    collected.append(normalized)
            # Otherwise, if no explicit error marker
            elif not has_error and not has_ok:
                normalized = _normalize(obj)
                if normalized and normalized.strip():
                    collected.append(normalized)
            
            # Recurse to find nested successes
            for key, val in obj.items():
                # Skip error fields themselves
                if key != "error":
                    collected.extend(_gather_code_strings(val))
            return collected
        # Ignore bare strings that are not under code-bearing keys to avoid picking reasons/log lines.
        return collected

    deep_strings = _gather_code_strings(results)

    def _best_from_list(items: List[Any]) -> Optional[str]:
        """
        遍历items列表，收集所有非fallback(非return None)的有效代码。
        返回最后一个有效代码。完全忽略所有的return None输出和失败的工具执行。
        """
        valid_codes: List[str] = []
        
        for item_idx, item in enumerate(items):
            # If item is a dict with ok=True, process it
            if isinstance(item, dict):
                # Skip if tool had fatal errors
                if "error" in item and _is_tool_error(item["error"]):
                    continue
                # MUST have ok=True and no error
                if item.get("ok") is True and not item.get("error"):
                    # This is a successful tool output
                    normalized = _normalize(item)
                    if normalized and normalized.strip():
                        # Skip fallback and irrelevant code
                        if not _is_fallback(normalized) and not _is_irrelevant(normalized):
                            valid_codes.append(normalized)
                # Skip anything that's not explicitly ok=True
                continue
            
            # Only try to extract from nested structures if not a dict with status
            # (to avoid processing failed tool results)
            if not isinstance(item, dict) or "ok" not in item:
                texts = _gather_code_strings(item)
                for text in texts:
                    if text.strip():
                        # Skip fallback and irrelevant code
                        if not _is_fallback(text) and not _is_irrelevant(text):
                            valid_codes.append(text)
        
        # Return the last valid code, or None if no valid code found
        return valid_codes[-1] if valid_codes else None

    candidates: List[str] = []
    
    # First, check if results directly contains tool_trace (common structure)
    if "tool_trace" in results or "tool_exec" in results:
        trace_key = "tool_trace" if "tool_trace" in results else "tool_exec"
        tool_trace = results[trace_key]
        if isinstance(tool_trace, dict):
            # Prioritize builder role for code generation
            for role in ["builder", "researcher", "planner", "tester", "refractor"]:
                traces = tool_trace.get(role)
                if isinstance(traces, list) and traces:
                    # Extract all result objects from the trace list
                    result_objs = []
                    for trace in traces:
                        if isinstance(trace, dict):
                            result_obj = trace.get("result", {})
                            if result_obj:
                                result_objs.append(result_obj)
                    
                    # Use _best_from_list to find the best code from all results
                    if result_objs:
                        best_code = _best_from_list(result_objs)
                        if best_code and best_code.strip():
                            candidates.append(best_code)
                    
                    # If we found a candidate from builder, prefer it
                    if candidates and role == "builder":
                        break
    
    # Then try to extract from role-based results
    if not candidates:
        for role in ["builder", "planner", "researcher", "checker", "tester", "refractor"]:
            payload = results.get(role)
            if isinstance(payload, list) and payload:
                text = _best_from_list(payload)
            else:
                text = _normalize(payload)
            if text and text.strip() and not _is_fallback(text) and not _is_irrelevant(text):
                candidates.append(text)

    # If no candidates yet, search all results values (but skip metadata keys)
    if not candidates:
        for key, payload in results.items():
            # Skip known non-code keys
            if key in ["log_path", "topology", "tool_exec", "tool_trace", "selected_agents"]:
                continue
            if isinstance(payload, list) and payload:
                text = _best_from_list(payload)
            else:
                text = _normalize(payload)
            if text and text.strip() and not _is_fallback(text) and not _is_irrelevant(text):
                candidates.append(text)

    code = ""
    if candidates:
        for candidate in reversed(candidates):
            if not _is_fallback(candidate) and not _is_irrelevant(candidate):
                code = candidate
                break
        if not code:
            code = ""

    # If we still landed on a fallback (e.g., only "return None" present), search deeper across all values.
    if not code or _is_fallback(code):
        for candidate in reversed(deep_strings):
            if not _is_fallback(candidate) and not _is_irrelevant(candidate):
                code = candidate
                break
        if not code:
            code = ""
    
    if not code:
        return None
    
    # Clean up the code: handle escaped newlines and JSON artifacts
    code = code.replace("\\n", "\n").replace("\\t", "\t")
    code = _extract_code_block(code)
    code = _strip_redundant_def(code, entry_point)
    code = _normalize_completion(code, entry_point)
    return code.strip("\n")


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
    baseline_fixed_bcb_mcts: bool,
    baseline_fixed_bcb_router_gpt4o: bool,
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
                param_hint = f"\n注意: 函数参数名是: {', '.join(params)}. 必须在代码中使用这些确切的参数名!\n"
            
            task_text = (
                f"{prompt}\n\n"
                "请在 JSON 输出中把实现写入 code_or_commands 字段，仅包含可执行的函数体代码。\n"
                "关键要求:\n"
                "1. 必须使用函数签名中的实际参数名,不要使用data/items/input_string等通用名\n"
                "2. 必须处理边界情况: 空输入([], '', None), 单元素, 特殊值(0, 1)\n"
                "3. 只生成直接解决问题的代码,不要生成无关的解析/处理模板\n"
                "4. 确保算法逻辑正确"
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

            results = result.get("results") or {}
            tool_exec = result.get("tool_exec")
            tool_trace = tool_exec if include_tool_trace else None
            
            # Extract topology and agent selection info FIRST (before checking completion)
            topology_info = result.get("topology") or {"topology": "linear", "roles": roles}
            selections = _load_selections(result.get("log_path"), registry)
            task_meta = {"topology": topology_info, "selected_agents": selections}
            if include_tool_trace and tool_trace is not None:
                task_meta["tool_trace"] = json.loads(json.dumps(tool_trace, ensure_ascii=True, default=str))
            
            # Include tool_exec in extraction so we can recover code even if role outputs
            # were validated into schemas that drop code_or_commands (e.g., planner).
            extraction_payload = dict(results)
            if tool_exec:
                extraction_payload["tool_exec"] = tool_exec
            completion = _extract_code_from_results(extraction_payload, entry_point)
            if completion is None or not completion.strip():
                # Record meta even if no code was generated (helps debugging)
                task_meta["extraction_failed"] = True
                task_meta["results_summary"] = {k: type(v).__name__ for k, v in results.items()}
                # Try to extract from tool_trace as last resort
                tool_trace = result.get("tool_exec")
                if tool_trace and isinstance(tool_trace, dict):
                    # Reuse the main extractor on tool_exec for robust nested output handling.
                    completion = _extract_code_from_results({"tool_exec": tool_trace}, entry_point)
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
            solutions[str(name)] = completion
            f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
            meta[str(name)] = task_meta

    return solutions, meta


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


def _run_with_repair(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    code = _build_program(prompt, str(completion), test_code, entry_point)
    ok, message = _run_python(code, timeout_s=timeout_s)
    if ok or not _looks_like_syntax_error(message):
        return ok, message, None

    repaired = _repair_completion(str(completion), entry_point)
    if not repaired or repaired == str(completion):
        return ok, message, None

    repaired_code = _build_program(prompt, repaired, test_code, entry_point)
    ok2, message2 = _run_python(repaired_code, timeout_s=timeout_s)
    repair_info = {"attempted": True, "success": ok2, "original_error": message}
    if not ok2:
        repair_info["final_error"] = message2
    return ok2, ("" if ok2 else message2), repair_info


def _run_python(code: str, timeout_s: float) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "eval_task.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if result.returncode == 0:
        return True, ""
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    message = stderr or stdout or f"exit_code={result.returncode}"
    return False, message


def evaluate(tasks_path: str, solutions_path: Optional[str], timeout_s: float) -> Dict[str, object]:
    tasks = _load_jsonl(tasks_path)
    solutions = _load_solutions(solutions_path)
    passed = 0
    results: List[dict] = []
    for task in tasks:
        name = task.get("name") or task.get("task_id") or ""
        prompt = task.get("prompt") or ""
        test_code = task.get("test") or ""
        entry_point = task.get("entry_point")
        completion = task.get("completion")
        if completion is None:
            completion = solutions.get(str(name), "")
        ok, message, repair_info = _run_with_repair(
            prompt,
            str(completion),
            test_code,
            entry_point,
            timeout_s=timeout_s,
        )
        if ok:
            passed += 1
        record = {"name": name, "ok": ok, "error": message}
        if repair_info:
            record["repair"] = repair_info
        results.append(record)
    total = len(results)
    pass_rate = passed / total if total else 0.0
    return {"total": total, "passed": passed, "pass_rate": pass_rate, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HumanEval-style tasks")
    parser.add_argument("--tasks", required=True, type=str, help="JSONL tasks file (prompt/test)")
    parser.add_argument("--solutions", default="", type=str, help="JSONL solutions file (name+completion)")
    parser.add_argument("--timeout", default=5.0, type=float, help="timeout seconds per task")
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
    tasks = _load_jsonl(args.tasks)
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
        name = task.get("name") or task.get("task_id") or ""
        prompt = task.get("prompt") or ""
        test_code = task.get("test") or ""
        entry_point = task.get("entry_point")
        completion = task.get("completion")
        if completion is None:
            completion = solutions.get(str(name), "")
        ok, message, repair_info = _run_with_repair(
            prompt,
            str(completion),
            test_code,
            entry_point,
            timeout_s=args.timeout,
        )
        if ok:
            passed += 1
        record = {"name": name, "ok": ok, "error": message}
        meta = meta_by_task.get(str(name))
        if meta:
            record.update(meta)
        if repair_info:
            record["repair"] = repair_info
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
