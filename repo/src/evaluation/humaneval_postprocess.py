from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_POSTPROCESS_AGENT_NAME = "postprocess_tool_agent"
_POSTPROCESS_AGENT_ID = "special-postprocess-agent"
_POSTPROCESS_TOOL_IDS = [
    "postprocess-fallback-guard",
    "postprocess-extract-cleanup",
    "postprocess-stop-token-trim",
    "postprocess-normalize-body",
    "postprocess-param-consistency",
    "postprocess-builtin-shadowing",
    "postprocess-syntax-indent-repair",
    "postprocess-name-scope-repair",
    "postprocess-timeout-guard",
    "postprocess-fallback-guard",
]
_POSTPROCESS_REGISTRY = None
_POSTPROCESS_EXECUTOR = None
_POSTPROCESS_DB_PATH: Optional[str] = None

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
    target_pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if target_pattern.match(line):
            body_lines = lines[idx + 1 :]
            if not body_lines:
                return text
            return "\n".join(body_lines).lstrip("\n")
    return text


def _normalize_body_indentation(lines: List[str]) -> List[str]:
    """
    Normalize body indentation levels for extracted snippets.
    This repairs cases like:
        # comment
            if ...
    where an `unexpected indent` would occur in a function body.
    """
    block_openers = re.compile(
        r"^(?:if|elif|else|for|while|try|except|finally|with|def|class|match|case)\b.*:\s*$"
    )
    normalized: List[str] = []
    indent_stack: List[int] = []

    for raw in lines:
        if not raw.strip():
            normalized.append("")
            continue

        stripped = raw.lstrip()
        lead = len(raw) - len(stripped)

        while indent_stack and lead < indent_stack[-1]:
            indent_stack.pop()

        max_allowed = (indent_stack[-1] + 4) if indent_stack else 0
        if lead > max_allowed:
            lead = max_allowed

        rebuilt = (" " * lead) + stripped
        normalized.append(rebuilt)

        if block_openers.match(stripped):
            indent_stack.append(lead + 4)

    return normalized


def _normalize_completion(text: str, entry_point: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\\n", "\n")
    if not entry_point:
        return text
    lines = text.lstrip("\n").splitlines()
    if not lines:
        return text
    candidate_module = "\n".join(lines)
    # Keep as full code only if it explicitly defines the target entry point at top-level.
    # Otherwise treat as function-body snippet to avoid indentation breakage when helper defs appear.
    try:
        tree = ast.parse(candidate_module)
        top_level_defs = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if entry_point in top_level_defs:
            return candidate_module
    except SyntaxError:
        pass

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

    lines = _normalize_body_indentation(lines)
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


def _resolve_postprocess_db_path(db_path: Optional[str]) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    if db_path:
        candidate = Path(db_path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
    else:
        candidate = repo_root / "demo_registry.sqlite"
    return str(candidate)


def _get_postprocess_executor(db_path: Optional[str] = None):
    global _POSTPROCESS_REGISTRY, _POSTPROCESS_EXECUTOR, _POSTPROCESS_DB_PATH

    resolved = _resolve_postprocess_db_path(db_path)
    if (
        _POSTPROCESS_EXECUTOR is not None
        and _POSTPROCESS_REGISTRY is not None
        and _POSTPROCESS_DB_PATH == resolved
    ):
        return _POSTPROCESS_EXECUTOR

    from core.registry import SQLiteRegistry
    from execution.tool_executor import ToolExecutor

    if _POSTPROCESS_REGISTRY is not None:
        try:
            _POSTPROCESS_REGISTRY.close()
        except Exception:
            pass

    _POSTPROCESS_REGISTRY = SQLiteRegistry(resolved)
    _POSTPROCESS_EXECUTOR = ToolExecutor(_POSTPROCESS_REGISTRY, timeout_s=2.0)
    _POSTPROCESS_DB_PATH = resolved
    return _POSTPROCESS_EXECUTOR


def _extract_postprocess_payload(raw_output: Any) -> Dict[str, Any]:
    if not isinstance(raw_output, dict):
        return {}
    nested = raw_output.get("output")
    if isinstance(nested, dict) and ("completion" in nested or "changed" in nested or "is_fallback" in nested):
        return nested
    return raw_output


def _run_postprocess_tool_direct(tool_id: str, inputs: Dict[str, Any], error_hint: Optional[str] = None) -> Dict[str, Any]:
    if _POSTPROCESS_REGISTRY is None:
        return {
            "ok": False,
            "output": None,
            "error": {"code": "registry_missing", "message": "postprocess registry not initialized"},
        }
    code = _POSTPROCESS_REGISTRY.get_tool_code(tool_id)
    if not code:
        return {
            "ok": False,
            "output": None,
            "error": {"code": "tool_code_missing", "message": f"missing tool code: {tool_id}"},
        }
    try:
        scope: Dict[str, Any] = {}
        exec(code, scope)
        run_fn = scope.get("run")
        if not callable(run_fn):
            return {
                "ok": False,
                "output": None,
                "error": {"code": "invalid_tool", "message": f"{tool_id} has no callable run(inputs)"},
            }
        output = run_fn(inputs)
        if not isinstance(output, dict):
            return {
                "ok": False,
                "output": None,
                "error": {"code": "invalid_output", "message": f"{tool_id} run(inputs) must return dict"},
            }
        return {"ok": True, "output": output, "error": None}
    except Exception as exc:
        message = str(exc)
        if error_hint:
            message = f"{error_hint}; direct_exec_error={message}"
        return {"ok": False, "output": None, "error": {"code": type(exc).__name__, "message": message}}


def _run_postprocess_tool_agent(
    completion: str,
    *,
    prompt: str,
    entry_point: Optional[str],
    stop_tokens: Optional[List[str]] = None,
    use_stop_tokens: bool = False,
    param_names: Optional[List[str]] = None,
    enable_syntax_repair: bool = True,
    error_message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    current = str(completion or "")
    resolved_params = [p for p in (param_names or _extract_param_names_from_prompt(prompt, entry_point)) if p]
    meta: Dict[str, Any] = {
        "agent": _POSTPROCESS_AGENT_NAME,
        "agent_id": _POSTPROCESS_AGENT_ID,
        "applied_tools": [],
        "fallback_detected": False,
        "tool_errors": {},
    }

    if not current.strip():
        meta["fallback_detected"] = True
        return "", meta

    try:
        executor = _get_postprocess_executor(db_path)
    except Exception as exc:
        meta["tool_errors"]["bootstrap"] = str(exc)
        return current.strip("\n"), meta

    base_inputs = {
        "prompt": prompt or "",
        "entry_point": entry_point,
        "stop_tokens": [str(t) for t in (stop_tokens or []) if str(t)],
        "use_stop_tokens": bool(use_stop_tokens),
        "param_names": resolved_params,
        "enable_syntax_repair": bool(enable_syntax_repair),
        "error_message": str(error_message or ""),
    }

    for tool_id in _POSTPROCESS_TOOL_IDS:
        tool_inputs = dict(base_inputs)
        tool_inputs["completion"] = current
        try:
            result = executor.run_tool(tool_id, tool_inputs)
        except Exception as exc:
            result = _run_postprocess_tool_direct(tool_id, tool_inputs, error_hint=f"sandbox_exec_error={exc}")
        if not result.get("ok"):
            meta["tool_errors"][tool_id] = result.get("error") or {"code": "unknown", "message": "tool failed"}
            continue
        payload = _extract_postprocess_payload(result.get("output"))
        if payload.get("is_fallback"):
            meta["fallback_detected"] = True
            meta["applied_tools"].append(tool_id)
            return "", meta
        next_completion = payload.get("completion")
        if not isinstance(next_completion, str):
            continue
        changed = bool(payload.get("changed")) or (next_completion != current)
        if changed:
            current = next_completion
            meta["applied_tools"].append(tool_id)
        if not current.strip():
            meta["fallback_detected"] = True
            return "", meta

    current = current.strip("\n")
    if not current.strip():
        meta["fallback_detected"] = True
        return "", meta
    if not meta["tool_errors"]:
        meta.pop("tool_errors", None)
    return current, meta


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

    # Tool timeout/error text accidentally extracted as code.
    if "error calling llm" in normalized:
        return True
    
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


def _extract_param_names_from_prompt(prompt: str, entry_point: Optional[str]) -> List[str]:
    sig = _extract_function_signature(prompt or "", entry_point)
    if not sig:
        return []
    params = sig.get("parameters")
    if not isinstance(params, list):
        return []
    return [str(p).strip() for p in params if str(p).strip()]


def _has_irrelevant_top_level_def(text: str, entry_point: Optional[str]) -> bool:
    """
    Check only module top-level function definitions via AST.
    Nested defs in function bodies are allowed.
    """
    if not text or not entry_point:
        return False

    candidate = _extract_code_block(str(text))
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        # Most function-body snippets are not parseable as a standalone module.
        # In body mode we should not enforce entry_point naming.
        return False

    top_level_defs = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not top_level_defs:
        return False
    return entry_point not in top_level_defs


def _extract_code_from_results(
    results: Dict[str, Any], entry_point: Optional[str], param_names: Optional[List[str]] = None
) -> Optional[str]:
    role_priority = [
        "builder",
        "code-generation",
        "planner",
        "code-planner",
        "researcher",
        "checker",
        "tester",
        "code-testing",
        "refractor",
        "refactor",
        "code-refactoring",
    ]

    def _is_irrelevant(text: str) -> bool:
        return _has_irrelevant_top_level_def(text, entry_point)

    def _normalize(value: Any, *, allow_string: bool = False) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            if not allow_string:
                return None
            unwrapped = _unwrap_json_code(value)
            result = unwrapped if unwrapped != value else value
            if result and result.strip() and not _is_fallback(result) and not _is_irrelevant(result):
                return result
            return None
        if isinstance(value, dict):
            if value.get("ok") is False:
                return None
            if "error" in value and _is_tool_error(value["error"]):
                return None
            if "error" in value and value["error"]:
                return None

            for key in ["code_or_commands", "code", "solution"]:
                if key in value:
                    nested = _normalize(value.get(key), allow_string=True)
                    if nested:
                        return nested

            if "output" in value:
                nested = _normalize(value.get("output"), allow_string=False)
                if nested:
                    return nested
            if "result" in value:
                nested = _normalize(value.get("result"), allow_string=False)
                if nested:
                    return nested
        return None

    def _first_candidate(obj: Any) -> Optional[str]:
        text = _normalize(obj, allow_string=False)
        if text:
            return text

        if isinstance(obj, list):
            for item in obj:
                hit = _first_candidate(item)
                if hit:
                    return hit
            return None

        if isinstance(obj, dict):
            for key in ("code_or_commands", "code", "solution"):
                if key in obj:
                    hit = _normalize(obj.get(key), allow_string=True)
                    if hit:
                        return hit
            for key in ("output", "result"):
                if key in obj:
                    hit = _first_candidate(obj.get(key))
                    if hit:
                        return hit
        return None

    code: Optional[str] = None

    trace_key = "tool_trace" if "tool_trace" in results else ("tool_exec" if "tool_exec" in results else None)
    reversed_roles = list(reversed(role_priority))

    if trace_key:
        tool_trace = results.get(trace_key)
        if isinstance(tool_trace, dict):
            for role in reversed_roles:
                traces = tool_trace.get(role)
                if isinstance(traces, list):
                    for trace in reversed(traces):
                        result_obj = trace.get("result") if isinstance(trace, dict) else trace
                        hit = _first_candidate(result_obj)
                        if hit:
                            code = hit
                            break
                if code:
                    break

    if not code:
        for role in reversed_roles:
            hit = _first_candidate(results.get(role))
            if hit:
                code = hit
                break

    if not code or _is_fallback(code):
        return None

    code = _normalize_completion(_strip_redundant_def(_extract_code_block(code), entry_point), entry_point)
    return code.strip("\n") if code.strip() else None


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


def _looks_like_assertion_error(message: str) -> bool:
    if not message:
        return False
    return "AssertionError" in message


def _extract_assertion_hint(message: str) -> str:
    if not message:
        return ""
    for line in message.splitlines():
        if "assert candidate" in line:
            return line.strip()
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _repair_assertion_completion_with_llm(
    *,
    llm_client: Any,
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: Optional[str],
    error_message: str,
    max_tokens: int = 700,
) -> Optional[str]:
    if llm_client is None:
        return None

    assertion_hint = _extract_assertion_hint(error_message)
    tests_preview = "\n".join((test_code or "").splitlines()[:80])
    messages = [
        {
            "role": "user",
            "content": (
                f"{prompt or ''}\n\n"
                f"Entry point: {entry_point or ''}\n\n"
                "You are fixing a failing implementation.\n"
                "Return ONLY raw function body code (no markdown, no explanation, no function signature).\n"
                "Keep correct Python indentation for function body lines.\n"
                "Do not output import/main/test code.\n\n"
                f"Current implementation body:\n{completion or ''}\n\n"
                f"Failing assertion/error hint:\n{assertion_hint}\n\n"
                f"Reference tests (truncated):\n{tests_preview}\n"
            ),
        }
    ]
    try:
        repaired = llm_client.chat(messages, temperature=0.1, max_tokens=max_tokens)
    except Exception:
        return None

    repaired = _extract_code_block(str(repaired))
    repaired = _strip_redundant_def(repaired, entry_point)
    repaired = _normalize_completion(repaired, entry_point)
    repaired = repaired.strip("\n")
    if not repaired or _is_fallback(repaired):
        return None
    return repaired


def _evaluate_with_postprocess_check(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
    max_postprocess_rounds: int = 1,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    completion_text = str(completion or "")
    code = _build_program(prompt, completion_text, test_code, entry_point)
    ok, message = _run_python(code, timeout_s=timeout_s)
    if ok:
        return True, "", None

    rounds = max(1, int(max_postprocess_rounds))
    original_error = message
    attempts: List[Dict[str, Any]] = []
    current = completion_text

    for round_idx in range(rounds):
        # Invoke postprocess agent after initial failure.
        param_names = _extract_param_names_from_prompt(prompt, entry_point)
        repaired_text, post_meta = _run_postprocess_tool_agent(
            completion=current,
            prompt=prompt,
            entry_point=entry_point,
            stop_tokens=[],
            use_stop_tokens=False,
            param_names=param_names,
            enable_syntax_repair=True,
            error_message=message,
        )

        attempt: Dict[str, Any] = {
            "round": round_idx + 1,
            "tools": post_meta.get("applied_tools", []),
            "original_error": message,
        }
        if not repaired_text or repaired_text == current:
            attempt["no_change"] = True
            attempts.append(attempt)
            break

        repaired_code = _build_program(prompt, repaired_text, test_code, entry_point)
        ok2, message2 = _run_python(repaired_code, timeout_s=timeout_s)
        attempt["success"] = ok2
        attempt["final_error"] = "" if ok2 else message2
        attempts.append(attempt)

        if ok2:
            postprocess_info: Dict[str, Any] = {
                "attempted": True,
                "success": True,
                "check_type": "postprocess_agent_check",
                "original_error": original_error,
                "attempts": attempts,
            }
            return True, "", postprocess_info

        current = repaired_text
        message = message2

    postprocess_info = {
        "attempted": True,
        "success": False,
        "check_type": "postprocess_agent_check",
        "original_error": original_error,
        "attempts": attempts,
    }
    if attempts and attempts[-1].get("no_change"):
        postprocess_info["no_change"] = True
    return False, message, postprocess_info


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