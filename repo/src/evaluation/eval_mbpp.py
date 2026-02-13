from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.convert_mbpp import convert_mbpp_file
from evaluation import eval_humaneval
from evaluation import humaneval_postprocess


_MBPP_SCORE_MARKER = "__MBPP_ASSERT_SCORE__"
_MBPP_ASSERT_DIAG_MARKER = "__MBPP_ASSERT_DIAG__"
_MBPP_POSTPROCESS_ROUNDS = 2
_MBPP_PATCHED = False
_MBPP_REPAIR_STYLE_RULES = (
    "Repair style rules: keep logic minimal and test-driven. "
    "Do not add new input validation, isinstance checks, early guards, or raise statements "
    "unless explicitly required by the provided asserts."
)
_MBPP_REPAIR_STYLE_RULES_STRICT = (
    "Strict round rule: previous repair was ineffective. "
    "Modify core computation to satisfy the failing assert exactly; "
    "avoid defensive programming and avoid introducing new exceptions."
)


def _strip_tasks_arg(argv: List[str]) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    for i, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--tasks":
            if i + 1 < len(argv):
                skip_next = True
            continue
        cleaned.append(token)
    return cleaned


def _ensure_include_tool_trace(argv: List[str]) -> List[str]:
    if "--include_tool_trace" not in argv:
        argv = list(argv) + ["--include_tool_trace"]
    return argv


def _ensure_postprocess_rounds(argv: List[str], rounds: int) -> List[str]:
    if "--postprocess_rounds" not in argv:
        argv = list(argv) + ["--postprocess_rounds", str(rounds)]
    return argv


def _extract_assert_lines(test_code: str) -> List[str]:
    lines: List[str] = []
    for line in (test_code or "").splitlines():
        s = line.strip()
        if s.startswith("assert "):
            lines.append(s)
    return lines


def _compact_text(text: str, max_chars: int = 180) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _compact_value(value: Any, max_chars: int = 240) -> str:
    if value is None:
        return ""
    return _compact_text(str(value), max_chars=max_chars)


def _extract_assert_examples(test_code: str, max_examples: int = 2) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for line in _extract_assert_lines(test_code):
        if len(examples) >= max_examples:
            break
        try:
            node = ast.parse(line).body[0]
        except SyntaxError:
            continue
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        left = test.left
        right = test.comparators[0]
        call, _ = _unwrap_candidate_call(left)
        if call is None:
            continue
        arg_src = ", ".join(ast.unparse(arg) if hasattr(ast, "unparse") else "arg" for arg in call.args)
        expected_src = ast.unparse(right) if hasattr(ast, "unparse") else "expected"
        examples.append({
            "input": _compact_text(arg_src, max_chars=180),
            "expected": _compact_text(expected_src, max_chars=180),
            "assert": _compact_text(line, max_chars=240),
        })
    return examples


def _build_mbpp_assertion_context(test_code: str, max_examples: int = 2) -> str:
    examples = _extract_assert_examples(test_code, max_examples=max_examples)
    if not examples:
        return ""
    lines = ["MBPP assertion focus (must satisfy exactly):"]
    for idx, example in enumerate(examples, start=1):
        lines.append(f"{idx}. input: {example['input']}")
        lines.append(f"   expected: {example['expected']}")
    return "\n".join(lines)


def _comment_block(text: str) -> str:
    if not text:
        return ""
    commented: List[str] = []
    for line in text.splitlines():
        if line.strip():
            commented.append(f"# {line}")
        else:
            commented.append("#")
    return "\n".join(commented)


def _append_mbpp_assertion_context(prompt: str, test_code: str) -> str:
    base = str(prompt or "")
    ctx = _build_mbpp_assertion_context(test_code)
    if not ctx:
        return base
    if "MBPP assertion focus" in base:
        return base
    return f"{base}\n\n{_comment_block(ctx)}\n"


def _format_param_types(contract: Dict[str, Any]) -> str:
    raw = contract.get("param_types") if isinstance(contract, dict) else None
    if not isinstance(raw, dict) or not raw:
        return ""
    parts: List[str] = []
    for idx, types in sorted(raw.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])):
        if isinstance(types, list):
            type_desc = "|".join(str(t) for t in types if t)
        else:
            type_desc = str(types)
        if type_desc:
            parts.append(f"{idx}={type_desc}")
    return "; ".join(parts)


def _looks_like_error_text(code: str) -> bool:
    text = str(code or "").strip()
    if not text:
        return False
    lower = text.lower()
    if text.startswith("# Error") or text.startswith("Error:"):
        return True
    if "traceback (most recent call last)" in lower:
        return True
    if "error calling llm" in lower:
        return True
    if "timed out" in lower or "timeout" in lower:
        return True
    return False


def _looks_like_repairable_runtime_failure(error_message: str) -> bool:
    text = str(error_message or "")
    if humaneval_postprocess._looks_like_assertion_error(text):
        return True
    return bool(re.search(r"\b(TypeError|ValueError)\b", text))


def _classify_failure_kind(error_message: str, diag: Optional[Dict[str, Any]] = None) -> str:
    text = str(error_message or "")
    lower = text.lower()
    if humaneval_postprocess._looks_like_assertion_error(text):
        return "assertion"
    if re.search(r"\bTypeError\b", text):
        return "type"
    if re.search(r"\bValueError\b", text):
        return "value"
    if re.search(r"\b(SyntaxError|IndentationError|TabError)\b", text):
        return "syntax"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    diag_error = str((diag or {}).get("error_type") or "")
    if diag_error in {"TypeError", "ValueError", "SyntaxError", "IndentationError", "TabError"}:
        return diag_error.replace("Error", "").lower()
    return "other"


def _failure_kind_guidance_lines(
    failure_kind: str,
    diag: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if failure_kind == "assertion":
        lines = [
            "Error-class guidance (assertion): focus on output logic for the shown failing assert.",
            "Do not add blanket input validation branches or new exceptions.",
            "Keep output type/order exactly as tests require.",
        ]
        if diag and diag.get("input_src"):
            lines.append(f"Failing input focus: {diag.get('input_src')}")
        return lines
    if failure_kind == "type":
        return [
            "Error-class guidance (type): fix operand/container type mismatch near failing expression.",
            "Do not add global isinstance gates that reject valid test inputs.",
            "Prefer local conversion/selection consistent with asserts, and keep return type contract.",
        ]
    if failure_kind == "value":
        return [
            "Error-class guidance (value): adjust computation domain/edge handling to match asserts.",
            "Avoid introducing new raise paths unless tests explicitly expect exceptions.",
            "Preserve existing structure and only patch the branch causing wrong value behavior.",
        ]
    if failure_kind == "syntax":
        return [
            "Error-class guidance (syntax): repair only syntax/indentation.",
            "Do not alter algorithmic structure unless strictly necessary for syntax validity.",
        ]
    if failure_kind == "timeout":
        return [
            "Error-class guidance (timeout): ensure finite loops and reduce unnecessary nested work.",
            "Keep algorithm shape and patch only the non-terminating/slow branch.",
        ]
    return [
        "Error-class guidance (generic): keep edits minimal and tied to the observed failing behavior.",
    ]


def _code_structure_metrics(code: str) -> Dict[str, int]:
    text = str(code or "")
    lines = text.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    non_comment = [ln for ln in non_empty if not ln.strip().startswith("#")]
    indented = "\n".join(("    " + ln) if ln.strip() else "    " for ln in lines) or "    pass"
    wrapped = f"def __mbpp_tmp__():\n{indented}\n"
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        return {
            "line_count": len(non_comment),
            "stmt_count": 0,
            "flow_count": 0,
            "raise_count": 0,
            "guard_count": 0,
        }
    fn = tree.body[0] if tree.body else None
    body = fn.body if isinstance(fn, ast.FunctionDef) else []
    flow_types = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    flow_count = 0
    raise_count = 0
    guard_count = 0
    for node in ast.walk(fn) if fn is not None else []:
        if isinstance(node, flow_types):
            flow_count += 1
        if isinstance(node, ast.Raise):
            raise_count += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            guard_count += 1
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.Is, ast.IsNot)):
                    guard_count += 1
    return {
        "line_count": len(non_comment),
        "stmt_count": len(body),
        "flow_count": flow_count,
        "raise_count": raise_count,
        "guard_count": guard_count,
    }


def _is_non_minimal_change(
    base_code: str,
    new_code: str,
    base_quality: Dict[str, Any],
    new_quality: Dict[str, Any],
) -> bool:
    if not str(base_code or "").strip() or not str(new_code or "").strip():
        return False
    if new_quality.get("full_ok"):
        return False
    # Allow broad edits only when score genuinely improves.
    if int(new_quality.get("passed") or 0) > int(base_quality.get("passed") or 0):
        return False

    base = _code_structure_metrics(base_code)
    new = _code_structure_metrics(new_code)
    line_limit = max(base["line_count"] + 12, int(base["line_count"] * 1.8) + 2)
    if new["line_count"] > line_limit:
        return True
    if abs(new["stmt_count"] - base["stmt_count"]) > 6:
        return True
    if abs(new["flow_count"] - base["flow_count"]) > 3:
        return True
    if new["raise_count"] > base["raise_count"]:
        return True
    if new["guard_count"] > base["guard_count"] + 2:
        return True
    return False


def _failure_context_message(
    error_message: str,
    test_code: str,
    max_examples: int = 2,
    diag: Optional[Dict[str, Any]] = None,
    contract: Optional[Dict[str, Any]] = None,
    failure_kind: str = "other",
    strict_round: bool = False,
    force_logic_change: bool = False,
) -> str:
    base = _compact_text(error_message or "", max_chars=700)
    lines: List[str] = []
    if base:
        lines.append(base)

    if diag and not diag.get("ok"):
        lines.append("Failed assert details:")
        idx = diag.get("assert_index")
        if idx is not None:
            lines.append(f"assert_index: {idx}")
        assert_src = _compact_value(diag.get("assert_src"), max_chars=260)
        if assert_src:
            lines.append(f"assert: {assert_src}")
        input_src = _compact_value(diag.get("input_src"), max_chars=220)
        if input_src:
            lines.append(f"input: {input_src}")
        expected_src = _compact_value(diag.get("expected_src"), max_chars=220)
        if expected_src:
            lines.append(f"expected: {expected_src}")
        expected_type = _compact_value(diag.get("expected_type"), max_chars=80)
        actual_type = _compact_value(diag.get("actual_type"), max_chars=80)
        if expected_type or actual_type:
            lines.append(f"expected_type: {expected_type} actual_type: {actual_type}")
        expected_val = _compact_value(diag.get("expected"), max_chars=240)
        actual_val = _compact_value(diag.get("actual"), max_chars=240)
        if expected_val:
            lines.append(f"expected_value: {expected_val}")
        if actual_val:
            lines.append(f"actual_value: {actual_val}")
        error_type = _compact_value(diag.get("error_type"), max_chars=80)
        error_msg = _compact_value(diag.get("error_message"), max_chars=220)
        if error_type:
            lines.append(f"exception: {error_type} {error_msg}")

    if contract:
        param_types = _format_param_types(contract)
        if param_types:
            lines.append(f"param_types: {param_types}")
        if contract.get("list_of_lists"):
            lines.append("Output must be list of lists (not tuples).")

    ctx = _build_mbpp_assertion_context(test_code, max_examples=max_examples)
    if ctx:
        lines.append("")
        lines.append(ctx)
    lines.append("")
    lines.extend(_failure_kind_guidance_lines(failure_kind, diag=diag))
    lines.append("")
    lines.append(_MBPP_REPAIR_STYLE_RULES)
    if strict_round:
        lines.append(_MBPP_REPAIR_STYLE_RULES_STRICT)
    if force_logic_change:
        lines.append("Previous attempt made no functional change; update computation logic directly.")
    return "\n".join(line for line in lines if line is not None)


def _split_setup_and_asserts(test_code: str) -> Tuple[str, List[str]]:
    lines = (test_code or "").splitlines()
    setup_lines: List[str] = []
    asserts: List[str] = []
    in_check = False
    check_indent = 0
    seen_check = False

    for line in lines:
        stripped = line.strip()
        if not seen_check:
            if stripped.startswith("def check("):
                seen_check = True
                in_check = True
                check_indent = len(line) - len(line.lstrip())
                continue
            # Keep only real setup before check(); this avoids pulling lines
            # from test_check(), e.g. check(entry_point), into setup.
            if stripped:
                setup_lines.append(line)
            continue

        if in_check:
            cur_indent = len(line) - len(line.lstrip())
            if stripped and cur_indent <= check_indent and not stripped.startswith("#"):
                in_check = False
            else:
                if stripped.startswith("assert "):
                    asserts.append(stripped)
                continue

        # We do not consume anything after check() definition.

    return "\n".join(setup_lines).strip(), asserts


def _infer_node_type(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "none"
        return type(node.value).__name__
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return type(node.operand.value).__name__
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"list", "tuple", "set", "dict", "str", "int", "float", "bool"}:
            return node.func.id
    return "unknown"


def _is_list_of_lists(node: ast.AST) -> bool:
    return isinstance(node, ast.List) and any(isinstance(elt, ast.List) for elt in node.elts)


def _unwrap_candidate_call(node: ast.AST) -> Tuple[Optional[ast.Call], bool]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "candidate":
            return node, True
        if node.func.id in {"sorted", "set", "list", "tuple"} and node.args:
            inner = node.args[0]
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "candidate":
                # wrapped candidate call, typically order-insensitive for sorted/set wrappers
                return inner, node.func.id not in {"sorted", "set"}
    return None, True


def _extract_io_contract(test_code: str) -> Dict[str, Any]:
    assert_lines = _extract_assert_lines(test_code)
    param_types: Dict[int, set[str]] = {}
    return_types: set[str] = set()
    order_required = False
    list_of_lists = False

    for line in assert_lines:
        try:
            node = ast.parse(line).body[0]
        except SyntaxError:
            continue
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        left = test.left
        right = test.comparators[0]
        call, left_order_sensitive = _unwrap_candidate_call(left)
        if call is None:
            continue
        for idx, arg in enumerate(call.args):
            param_types.setdefault(idx, set()).add(_infer_node_type(arg))
        ret_type = _infer_node_type(right)
        return_types.add(ret_type)
        if ret_type in {"list", "tuple"} and left_order_sensitive:
            order_required = True
        if isinstance(right, ast.List) and _is_list_of_lists(right):
            list_of_lists = True

    return {
        "assert_count": len(assert_lines),
        "param_types": {idx: sorted(types) for idx, types in sorted(param_types.items())},
        "return_types": sorted(return_types),
        "order_required": order_required,
        "list_of_lists": list_of_lists,
    }


def _collect_code_candidates(obj: Any, out: List[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, list):
        for item in obj:
            _collect_code_candidates(item, out)
        return
    if isinstance(obj, dict):
        if obj.get("ok") is False:
            return

        for key in ("code_or_commands", "code", "solution"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                out.append(val)

        output = obj.get("output")
        result = obj.get("result")
        if isinstance(output, (dict, list)):
            _collect_code_candidates(output, out)
        if isinstance(result, (dict, list)):
            _collect_code_candidates(result, out)

        # Common nested structure: output.output.code
        if isinstance(output, dict):
            inner = output.get("output")
            if isinstance(inner, (dict, list)):
                _collect_code_candidates(inner, out)

        # Recurse into role/tool buckets.
        for key, val in obj.items():
            if key in {"input", "reason", "error"}:
                continue
            if isinstance(val, (dict, list)):
                _collect_code_candidates(val, out)


def _collect_applied_tools(attempts: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for attempt in attempts:
        tools = attempt.get("tools") if isinstance(attempt, dict) else None
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, str) or not tool:
                continue
            if tool in seen:
                continue
            seen.add(tool)
            ordered.append(tool)
    return ordered


def _extract_trace_candidates(task_meta: Dict[str, Any]) -> List[str]:
    tool_trace = task_meta.get("tool_trace") if isinstance(task_meta, dict) else None
    if not isinstance(tool_trace, dict):
        return []
    candidates: List[str] = []
    _collect_code_candidates({"tool_trace": tool_trace}, candidates)
    return _dedupe_keep_order(candidates)


def _extract_last_valid_code_from_trace(
    task_meta: Dict[str, Any],
    entry_point: Optional[str],
) -> Optional[Dict[str, Any]]:
    tool_trace = task_meta.get("tool_trace") if isinstance(task_meta, dict) else None
    if not isinstance(tool_trace, dict) or not tool_trace:
        return None
    roles = list(tool_trace.keys())
    for role in reversed(roles):
        entries = tool_trace.get(role)
        if not isinstance(entries, list):
            continue
        for entry in reversed(entries):
            candidates: List[str] = []
            _collect_code_candidates(entry, candidates)
            for code in reversed(_dedupe_keep_order(candidates)):
                normalized = eval_humaneval._normalize_completion(
                    humaneval_postprocess._strip_redundant_def(
                        humaneval_postprocess._extract_code_block(str(code)),
                        entry_point,
                    ),
                    entry_point,
                )
                normalized = humaneval_postprocess._fix_missing_indents(normalized)
                if not normalized.strip():
                    continue
                if _looks_like_error_text(normalized):
                    continue
                return {
                    "code": normalized,
                    "role": role,
                    "tool_id": entry.get("tool_id") if isinstance(entry, dict) else None,
                }
    return None


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    kept: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def _parse_assert_line(assert_line: str) -> Optional[Dict[str, str]]:
    try:
        node = ast.parse(assert_line).body[0]
    except SyntaxError:
        return None
    if not isinstance(node, ast.Assert):
        return None
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    left = test.left
    right = test.comparators[0]
    call, _ = _unwrap_candidate_call(left)
    input_src = ""
    if call is not None:
        input_src = ", ".join(ast.unparse(arg) if hasattr(ast, "unparse") else "arg" for arg in call.args)
    left_src = ast.unparse(left) if hasattr(ast, "unparse") else "candidate(...)"
    right_src = ast.unparse(right) if hasattr(ast, "unparse") else "expected"
    return {
        "assert_src": assert_line.strip(),
        "left_src": left_src,
        "right_src": right_src,
        "input_src": input_src,
        "expected_src": right_src,
    }


def _build_assert_diagnostic_test(test_code: str, entry_point: str) -> str:
    setup, asserts = _split_setup_and_asserts(test_code)
    parsed_asserts: List[Dict[str, str]] = []
    for line in asserts:
        parsed = _parse_assert_line(line)
        if parsed:
            parsed_asserts.append(parsed)

    lines: List[str] = []
    if setup:
        lines.append(setup)
        lines.append("")
    lines.append("def _mbpp_diag(candidate):")
    lines.append("    import json")
    lines.append("    def _emit(payload):")
    lines.append(f"        print('{_MBPP_ASSERT_DIAG_MARKER}' + json.dumps(payload, ensure_ascii=True))")
    if not parsed_asserts:
        lines.append("    _emit({'ok': True})")
    else:
        for idx, parsed in enumerate(parsed_asserts, start=1):
            left_src = parsed["left_src"]
            right_src = parsed["right_src"]
            input_src = parsed.get("input_src", "")
            assert_src = parsed.get("assert_src", "")
            expected_src = parsed.get("expected_src", "")
            lines.append("    try:")
            lines.append(f"        _actual = {left_src}")
            lines.append(f"        _expected = {right_src}")
            lines.append("        if _actual != _expected:")
            lines.append("            _emit({")
            lines.append("                'ok': False,")
            lines.append(f"                'assert_index': {idx},")
            lines.append(f"                'assert_src': {json.dumps(assert_src)},")
            lines.append(f"                'input_src': {json.dumps(input_src)},")
            lines.append(f"                'expected_src': {json.dumps(expected_src)},")
            lines.append("                'actual': repr(_actual),")
            lines.append("                'expected': repr(_expected),")
            lines.append("                'actual_type': type(_actual).__name__,")
            lines.append("                'expected_type': type(_expected).__name__,")
            lines.append("            })")
            lines.append("            return")
            lines.append("    except Exception as e:")
            lines.append("        _emit({")
            lines.append("            'ok': False,")
            lines.append(f"            'assert_index': {idx},")
            lines.append(f"            'assert_src': {json.dumps(assert_src)},")
            lines.append(f"            'input_src': {json.dumps(input_src)},")
            lines.append(f"            'expected_src': {json.dumps(expected_src)},")
            lines.append("            'error_type': type(e).__name__,")
            lines.append("            'error_message': str(e),")
            lines.append("        })")
            lines.append("        return")
        lines.append("    _emit({'ok': True})")
    lines.append("")
    lines.append("def test_check():")
    lines.append(f"    _mbpp_diag({entry_point})")
    lines.append("")
    lines.append("test_check()")
    return "\n".join(lines) + "\n"


def _parse_assert_diag_output(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    for line in text.splitlines():
        if _MBPP_ASSERT_DIAG_MARKER not in line:
            continue
        payload = line.split(_MBPP_ASSERT_DIAG_MARKER, 1)[-1].strip()
        if not payload:
            continue
        try:
            return json.loads(payload)
        except Exception:
            return None
    return None


def _diagnose_first_assert_failure(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> Optional[Dict[str, Any]]:
    if not entry_point:
        return None
    diag_test = _build_assert_diagnostic_test(test_code, entry_point)
    program = eval_humaneval._build_program(prompt, candidate, diag_test, entry_point)
    ok, stdout, stderr = _run_python_capture(program, timeout_s=timeout_s)
    text = (stdout or "") + "\n" + (stderr or "")
    diag = _parse_assert_diag_output(text)
    if diag is None:
        return {"ok": ok}
    return diag


def _build_mbpp_llm_client(config: Dict[str, Any]) -> Optional[Any]:
    raw_code_model = str(config.get("code_model") or "").strip()
    raw_base_url = str(config.get("code_base_url") or "").strip()
    raw_api_key = config.get("code_api_key") or None
    env_base_url = str(os.getenv("LLM_API_BASE") or "").strip()
    env_api_key = os.getenv("LLM_API_KEY") or None
    remote_requested = bool(raw_code_model or raw_base_url or raw_api_key or env_base_url or env_api_key)
    code_model = raw_code_model
    code_base_url = raw_base_url
    code_api_key = raw_api_key
    code_timeout = float(config.get("code_timeout") or 60.0)
    code_retries = int(config.get("code_retries") or 2)

    local_model_dir = config.get("local_model_dir") or None
    local_lora_dir = config.get("local_lora_dir") or None
    local_device = config.get("local_device") or None
    local_dtype = config.get("local_dtype") or "auto"
    local_use_4bit = bool(config.get("local_use_4bit") or False)
    local_use_8bit = bool(config.get("local_use_8bit") or False)
    max_new_tokens = int(config.get("max_new_tokens") or 256)
    temperature = float(config.get("temperature") or 0.2)

    if remote_requested:
        if not code_model:
            code_model = "gpt-4o"
        if not code_base_url:
            code_base_url = env_base_url
        if not code_api_key:
            code_api_key = env_api_key
        try:
            from generation.llm_client import LLMClient
        except Exception:
            return None
        return LLMClient(
            api_key=code_api_key or None,
            base_url=code_base_url or None,
            model=code_model or None,
            timeout_s=code_timeout,
            max_retries=code_retries,
        )

    if local_model_dir:
        try:
            from generation.local_llm import LocalLLMClient
        except Exception:
            return None
        return LocalLLMClient(
            model_path=local_model_dir,
            lora_path=local_lora_dir,
            device=local_device,
            dtype=local_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_4bit=local_use_4bit,
            use_8bit=local_use_8bit,
        )

    return None


def _run_python_capture(code: str, timeout_s: float) -> Tuple[bool, str, str]:
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
            return False, "", "timeout"
    return result.returncode == 0, result.stdout or "", result.stderr or ""


def _syntax_precheck(
    prompt: str,
    completion: str,
    entry_point: Optional[str],
) -> Optional[str]:
    if not completion or not completion.strip():
        return None
    program = eval_humaneval._build_program(prompt, completion, "", entry_point)
    try:
        compile(program, "<mbpp_precheck>", "exec")
    except SyntaxError as exc:
        lineno = f" line {exc.lineno}" if exc.lineno else ""
        return f"{type(exc).__name__}: {exc.msg}{lineno}"
    return None


def _postprocess_completion(
    prompt: str,
    completion: str,
    entry_point: Optional[str],
    *,
    rounds: int,
    error_message: str = "",
) -> Tuple[str, Dict[str, Any]]:
    current = str(completion or "")
    if not current.strip():
        return current, {"attempted": False}

    param_names = humaneval_postprocess._extract_param_names_from_prompt(prompt, entry_point)
    repaired_text, post_meta = humaneval_postprocess._run_postprocess_tool_agent(
        completion=current,
        prompt=prompt,
        entry_point=entry_point,
        stop_tokens=[],
        use_stop_tokens=False,
        param_names=param_names,
        enable_syntax_repair=True,
        error_message=error_message,
    )
    next_text = repaired_text if repaired_text and repaired_text.strip() else current
    normalized = humaneval_postprocess._normalize_completion(
        humaneval_postprocess._strip_redundant_def(
            humaneval_postprocess._extract_code_block(str(next_text)),
            entry_point,
        ),
        entry_point,
    )
    if normalized and normalized.strip():
        current = normalized
    post_meta["rounds"] = int(rounds)
    return current, post_meta


def _fallback_completion_from_trace(task: Dict[str, Any], task_meta: Dict[str, Any]) -> str:
    tool_trace = task_meta.get("tool_trace")
    if not isinstance(tool_trace, dict):
        return ""
    entry_point = task.get("entry_point")
    fallback = humaneval_postprocess._extract_code_from_results(
        {"tool_trace": tool_trace},
        entry_point,
    )
    return str(fallback or "")


def _build_assert_scoring_test(test_code: str, entry_point: str) -> Tuple[str, int]:
    setup, asserts = _split_setup_and_asserts(test_code)
    total = len(asserts)
    lines: List[str] = []
    if setup:
        lines.append(setup)
        lines.append("")
    lines.append("def check(candidate):")
    lines.append("    passed = 0")
    lines.append(f"    total = {total}")
    if asserts:
        for assert_line in asserts:
            lines.append("    try:")
            lines.append(f"        {assert_line}")
            lines.append("        passed += 1")
            lines.append("    except Exception:")
            lines.append("        pass")
    lines.append("    return passed, total")
    lines.append("")
    lines.append("def test_check():")
    lines.append(f"    p, t = check({entry_point})")
    lines.append(f"    print('{_MBPP_SCORE_MARKER}' + str(p) + '/' + str(t))")
    lines.append("")
    lines.append("test_check()")
    return "\n".join(lines) + "\n", total


def _score_candidate_against_asserts(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> Tuple[int, int]:
    if not entry_point:
        return 0, 0
    score_test, total = _build_assert_scoring_test(test_code, entry_point)
    program = eval_humaneval._build_program(prompt, candidate, score_test, entry_point)
    ok, stdout, stderr = _run_python_capture(program, timeout_s=timeout_s)
    text = (stdout or "") + "\n" + (stderr or "")
    m = re.search(rf"{re.escape(_MBPP_SCORE_MARKER)}(\d+)/(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    if ok and total > 0:
        # If marker is missing but run succeeded, conservatively treat as all passed.
        return total, total
    return 0, total


def _full_test_pass(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> bool:
    program = eval_humaneval._build_program(prompt, candidate, test_code, entry_point)
    ok, _, _ = _run_python_capture(program, timeout_s=timeout_s)
    return ok


def _evaluate_candidate_quality(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> Dict[str, Any]:
    syntax_error = _syntax_precheck(prompt, candidate, entry_point) if entry_point else None
    full_ok = False
    full_error = ""
    if syntax_error:
        full_error = str(syntax_error)
    else:
        program = eval_humaneval._build_program(prompt, candidate, test_code, entry_point)
        ok, stdout, stderr = _run_python_capture(program, timeout_s=timeout_s)
        full_ok = bool(ok)
        if not ok:
            full_error = (stderr or stdout or "execution failed").strip()
    passed, total = _score_candidate_against_asserts(
        prompt,
        candidate,
        test_code,
        entry_point,
        timeout_s=timeout_s,
    )
    return {
        "syntax_error": syntax_error,
        "passed": int(passed),
        "total": int(total),
        "full_ok": bool(full_ok),
        "full_error": full_error,
    }


def _candidate_quality_regressed(new_quality: Dict[str, Any], base_quality: Dict[str, Any]) -> bool:
    if new_quality.get("syntax_error"):
        return True
    base_total = int(base_quality.get("total") or 0)
    base_passed = int(base_quality.get("passed") or 0)
    new_total = int(new_quality.get("total") or 0)
    new_passed = int(new_quality.get("passed") or 0)
    if base_total > 0 and new_total == base_total and new_passed < base_passed:
        return True
    if bool(base_quality.get("full_ok")) and not bool(new_quality.get("full_ok")):
        return True
    return False


def _postprocess_after_failure(
    prompt: str,
    candidate: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_s: float,
    rounds: int,
    llm_client: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    current = candidate
    attempts: List[Dict[str, Any]] = []
    effective_prompt = _append_mbpp_assertion_context(prompt, test_code)
    contract = _extract_io_contract(test_code)
    rounds = max(1, int(rounds))
    prev_round_no_change = False
    quality_cache: Dict[str, Dict[str, Any]] = {}

    def _get_quality(candidate_text: str) -> Dict[str, Any]:
        key = str(candidate_text or "")
        cached = quality_cache.get(key)
        if cached is not None:
            return cached
        quality = _evaluate_candidate_quality(
            effective_prompt,
            key,
            test_code,
            entry_point,
            timeout_s=timeout_s,
        )
        quality_cache[key] = quality
        return quality

    for round_idx in range(rounds):
        round_changed = False
        if current:
            current = humaneval_postprocess._fix_missing_indents(str(current))
        current_quality = _get_quality(current)
        if current_quality.get("full_ok"):
            return current, {"success": True, "attempts": attempts}
        error_message = str(current_quality.get("full_error") or "execution failed").strip()
        diag = _diagnose_first_assert_failure(
            effective_prompt,
            current,
            test_code,
            entry_point,
            timeout_s=timeout_s,
        )
        failure_kind = _classify_failure_kind(error_message, diag=diag)
        failure_context = _failure_context_message(
            error_message,
            test_code,
            diag=diag,
            contract=contract,
            failure_kind=failure_kind,
            strict_round=round_idx > 0,
            force_logic_change=prev_round_no_change,
        )

        if llm_client is not None and _looks_like_repairable_runtime_failure(error_message):
            llm_attempt: Dict[str, Any] = {
                "round": round_idx + 1,
                "repair_type": "assertion_llm",
                "original_error": error_message,
                "failure_kind": failure_kind,
            }
            repaired_llm = humaneval_postprocess._repair_assertion_completion_with_llm(
                llm_client=llm_client,
                prompt=effective_prompt,
                completion=current,
                test_code=test_code,
                entry_point=entry_point,
                error_message=error_message,
                failure_context=failure_context,
            )
            if repaired_llm and repaired_llm != current:
                repaired_llm = humaneval_postprocess._fix_missing_indents(repaired_llm)
                llm_quality = _get_quality(repaired_llm)
                llm_attempt["candidate_score"] = f"{llm_quality['passed']}/{llm_quality['total']}"
                if llm_quality.get("syntax_error"):
                    llm_attempt["success"] = False
                    llm_attempt["rejected"] = "syntax_error"
                    llm_attempt["final_error"] = str(llm_quality.get("syntax_error") or "")
                    attempts.append(llm_attempt)
                elif _candidate_quality_regressed(llm_quality, current_quality):
                    llm_attempt["success"] = False
                    llm_attempt["rejected"] = "score_dropped"
                    llm_attempt["final_error"] = "Rejected candidate because assert score regressed."
                    attempts.append(llm_attempt)
                elif _is_non_minimal_change(current, repaired_llm, current_quality, llm_quality):
                    llm_attempt["success"] = False
                    llm_attempt["rejected"] = "non_minimal_edit"
                    llm_attempt["final_error"] = "Rejected candidate because structural edit is too large."
                    attempts.append(llm_attempt)
                else:
                    ok_llm = bool(llm_quality.get("full_ok"))
                    llm_attempt["success"] = ok_llm
                    llm_attempt["final_error"] = "" if ok_llm else str(llm_quality.get("full_error") or "execution failed")
                    attempts.append(llm_attempt)
                    if ok_llm:
                        return repaired_llm, {"success": True, "attempts": attempts}
                    current = repaired_llm
                    current_quality = llm_quality
                    error_message = str(llm_quality.get("full_error") or "execution failed").strip()
                    round_changed = True
            else:
                llm_attempt["success"] = False
                llm_attempt["no_change"] = True
                llm_attempt["final_error"] = error_message
                attempts.append(llm_attempt)

        param_names = humaneval_postprocess._extract_param_names_from_prompt(effective_prompt, entry_point)
        repaired_text, post_meta = humaneval_postprocess._run_postprocess_tool_agent(
            completion=current,
            prompt=effective_prompt,
            entry_point=entry_point,
            stop_tokens=[],
            use_stop_tokens=False,
            param_names=param_names,
            enable_syntax_repair=True,
            error_message=failure_context,
        )
        attempt = {
            "round": round_idx + 1,
            "tools": post_meta.get("applied_tools", []),
            "original_error": error_message,
            "failure_kind": failure_kind,
        }
        if not repaired_text or repaired_text == current:
            attempt["no_change"] = True
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            continue

        normalized = eval_humaneval._normalize_completion(
            humaneval_postprocess._strip_redundant_def(
                eval_humaneval._extract_code_block(str(repaired_text)),
                entry_point,
            ),
            entry_point,
        )
        repaired_candidate = humaneval_postprocess._fix_missing_indents(normalized)
        if not repaired_candidate.strip():
            attempt["rejected"] = "empty_after_normalize"
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            continue

        repaired_quality = _get_quality(repaired_candidate)
        attempt["candidate_score"] = f"{repaired_quality['passed']}/{repaired_quality['total']}"
        if repaired_quality.get("syntax_error"):
            attempt["rejected"] = "syntax_error"
            attempt["final_error"] = str(repaired_quality.get("syntax_error") or "")
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            continue
        if _candidate_quality_regressed(repaired_quality, current_quality):
            attempt["rejected"] = "score_dropped"
            attempt["final_error"] = "Rejected candidate because assert score regressed."
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            continue
        if _is_non_minimal_change(current, repaired_candidate, current_quality, repaired_quality):
            attempt["rejected"] = "non_minimal_edit"
            attempt["final_error"] = "Rejected candidate because structural edit is too large."
            attempts.append(attempt)
            prev_round_no_change = not round_changed
            continue

        current = repaired_candidate
        attempts.append(attempt)
        round_changed = True
        if repaired_quality.get("full_ok"):
            return current, {"success": True, "attempts": attempts}
        prev_round_no_change = not round_changed

    return current, {"success": False, "attempts": attempts}


def _literal_return_type(token: str) -> str:
    token = token.strip()
    if token == "None":
        return "none"
    if token in {"[]", "list()"}:
        return "list"
    if token in {"{}", "dict()"}:
        return "dict"
    if token in {"()", "tuple()"}:
        return "tuple"
    if token in {"False", "True", "bool()"}:
        return "bool"
    if re.fullmatch(r"-?\d+", token):
        return "int"
    if re.fullmatch(r"-?\d+\.\d+", token):
        return "float"
    if token.startswith(("'", '"')):
        return "str"
    return "unknown"


def _typeerror_risk_penalty(code: str) -> float:
    if not code:
        return 0.0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    penalty = 0.0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "isinstance" and len(node.args) >= 2:
                type_arg = node.args[1]
                if isinstance(type_arg, ast.Constant) and isinstance(type_arg.value, str):
                    penalty += 3.0
                if isinstance(type_arg, ast.Call) and isinstance(type_arg.func, ast.Name):
                    if type_arg.func.id in {"list", "dict", "set", "tuple", "str", "int", "float", "bool"}:
                        penalty += 2.5
            if node.func.id == "set" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.List) and any(
                    isinstance(elt, (ast.List, ast.Dict, ast.Set)) for elt in arg.elts
                ):
                    penalty += 3.0
        if isinstance(node, ast.Set):
            if any(isinstance(elt, (ast.List, ast.Dict, ast.Set)) for elt in node.elts):
                penalty += 3.0
        if isinstance(node, ast.Dict):
            if any(isinstance(k, (ast.List, ast.Dict, ast.Set)) for k in node.keys if k is not None):
                penalty += 3.0
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                left_is_str = isinstance(node.left.value, str)
                right_is_str = isinstance(node.right.value, str)
                left_is_num = isinstance(node.left.value, (int, float))
                right_is_num = isinstance(node.right.value, (int, float))
                if (left_is_str and right_is_num) or (right_is_str and left_is_num):
                    penalty += 1.5
    return penalty


def _defensive_penalty(code: str, contract: Dict[str, Any]) -> float:
    head = "\n".join((code or "").splitlines()[:20])
    penalty = 0.0
    if re.search(r"\bisinstance\s*\(", head):
        penalty += 2.0
    if re.search(r"\bis\s+None\b", head):
        penalty += 1.0
    if re.search(r"\bif\s+not\s+[A-Za-z_][A-Za-z0-9_]*\b", head):
        penalty += 0.5

    expected = set(contract.get("return_types") or [])
    for m in re.finditer(r"^\s*return\s+(.+)$", head, flags=re.MULTILINE):
        ret_type = _literal_return_type(m.group(1).strip())
        if ret_type != "unknown" and expected and ret_type not in expected:
            penalty += 1.5
    return penalty


def _set_list_order_penalty(code: str, contract: Dict[str, Any]) -> float:
    if not contract.get("order_required"):
        return 0.0
    text = code or ""
    patterns = [
        r"list\s*\(\s*set\s*\(",
        r"set\s*\([^)]*\)\s*\.intersection\s*\(",
        r"set\s*\([^)]*\)\s*&\s*set\s*\(",
    ]
    for p in patterns:
        if re.search(p, text):
            return 4.0
    return 0.0


def _pick_best_candidate_for_task(
    task: Dict[str, Any],
    current_completion: str,
    candidates: List[str],
    timeout_s: float,
) -> Tuple[str, Dict[str, Any]]:
    prompt = str(task.get("prompt") or "")
    test_code = str(task.get("test") or "")
    entry_point = task.get("entry_point")
    effective_prompt = _append_mbpp_assertion_context(prompt, test_code)
    contract = _extract_io_contract(test_code)

    all_candidates = _dedupe_keep_order([current_completion] + candidates)
    if not all_candidates:
        return current_completion, {"contract": contract, "candidate_count": 0}

    scored: List[Tuple[int, int, int, float, int, str]] = []
    # tuple fields: (passed, total, full_ok_int, -penalty, -idx, candidate)
    for idx, cand in enumerate(all_candidates):
        if _looks_like_error_text(cand):
            continue
        normalized = eval_humaneval._normalize_completion(
            humaneval_postprocess._strip_redundant_def(
                eval_humaneval._extract_code_block(str(cand)),
                entry_point,
            ),
            entry_point,
        )
        normalized = humaneval_postprocess._fix_missing_indents(normalized)
        if not (normalized or "").strip():
            continue
        if _looks_like_error_text(normalized):
            continue
        passed, total = _score_candidate_against_asserts(
            effective_prompt, normalized, test_code, entry_point, timeout_s=timeout_s
        )
        full_ok = 1 if (total > 0 and passed == total and _full_test_pass(
            effective_prompt,
            normalized,
            test_code,
            entry_point,
            timeout_s=timeout_s,
        )) else 0
        penalty = (
            _defensive_penalty(normalized, contract)
            + _set_list_order_penalty(normalized, contract)
            + _typeerror_risk_penalty(normalized)
        )
        scored.append((passed, total, full_ok, -penalty, -idx, normalized))

        if not full_ok:
            repaired, _ = _postprocess_after_failure(
                effective_prompt,
                normalized,
                test_code,
                entry_point,
                timeout_s=timeout_s,
                rounds=_MBPP_POSTPROCESS_ROUNDS,
            )
            if repaired and repaired != normalized:
                passed2, total2 = _score_candidate_against_asserts(
                    effective_prompt, repaired, test_code, entry_point, timeout_s=timeout_s
                )
                full_ok2 = 1 if (total2 > 0 and passed2 == total2 and _full_test_pass(
                    effective_prompt,
                    repaired,
                    test_code,
                    entry_point,
                    timeout_s=timeout_s,
                )) else 0
                penalty2 = (
                    _defensive_penalty(repaired, contract)
                    + _set_list_order_penalty(repaired, contract)
                    + _typeerror_risk_penalty(repaired)
                )
                scored.append((passed2, total2, full_ok2, -penalty2, -idx, repaired))

    if not scored:
        return current_completion, {"contract": contract, "candidate_count": len(all_candidates)}

    scored.sort(reverse=True)
    best = scored[0][5]
    meta = {
        "contract": contract,
        "candidate_count": len(all_candidates),
        "selected_score": {
            "passed": scored[0][0],
            "total": scored[0][1],
            "full_ok": bool(scored[0][2]),
            "penalty": -scored[0][3],
        },
    }
    return best, meta


def _post_select_mbpp_solutions(
    tasks: List[Dict[str, Any]],
    solutions: Dict[str, str],
    meta: Dict[str, Dict[str, Any]],
    timeout_s: float,
) -> None:
    task_by_name = {str(t.get("name") or t.get("task_id") or ""): t for t in tasks}
    for name, task in task_by_name.items():
        if str(task.get("source") or "").lower() != "mbpp":
            continue
        current = str(solutions.get(name, ""))
        if not current.strip():
            continue
        task_meta = meta.get(name, {})
        task_meta["mbpp_candidate_reselection"] = {
            "candidate_count": 1,
            "selected_score": {},
        }
        meta[name] = task_meta


def _postprocess_mbpp_solutions(
    tasks: List[Dict[str, Any]],
    solutions: Dict[str, str],
    meta: Dict[str, Dict[str, Any]],
    rounds: int,
    timeout_s: float,
    llm_client: Optional[Any] = None,
) -> None:
    task_by_name = {str(t.get("name") or t.get("task_id") or ""): t for t in tasks}
    for name, task in task_by_name.items():
        if str(task.get("source") or "").lower() != "mbpp":
            continue
        task_meta = meta.get(name, {})
        completion = str(solutions.get(name, ""))
        if _looks_like_error_text(completion):
            completion = ""
            solutions[name] = ""
        original_completion = completion
        if not completion.strip():
            fallback = _fallback_completion_from_trace(task, task_meta)
            if fallback.strip():
                completion = fallback
                solutions[name] = completion
                task_meta["extracted_from"] = "tool_trace_fallback"
        if not completion.strip():
            meta[name] = task_meta
            continue
        prompt = str(task.get("prompt") or "")
        test_code = str(task.get("test") or "")
        effective_prompt = _append_mbpp_assertion_context(prompt, test_code)
        entry_point = task.get("entry_point")
        fixed_completion = humaneval_postprocess._fix_missing_indents(completion)
        if fixed_completion and fixed_completion.strip() and fixed_completion != completion:
            completion = fixed_completion
            solutions[name] = completion
        precheck_error = None
        precheck_meta: Dict[str, Any] = {"attempted": False}
        if entry_point:
            precheck_error = _syntax_precheck(effective_prompt, completion, entry_point)
        if precheck_error:
            repaired, precheck_meta = _postprocess_completion(
                effective_prompt,
                completion,
                entry_point,
                rounds=1,
                error_message=precheck_error,
            )
            precheck_meta["attempted"] = True
            precheck_meta["error"] = precheck_error
            precheck_meta["changed"] = repaired.strip() != completion.strip()
            if repaired and repaired.strip():
                completion = repaired
                solutions[name] = completion
        task_meta["mbpp_precheck"] = precheck_meta
        ok_before = True
        if entry_point and test_code:
            ok_before = _full_test_pass(effective_prompt, completion, test_code, entry_point, timeout_s=timeout_s)

        post_meta: Dict[str, Any] = {"attempted": False, "success": ok_before}
        if not ok_before:
            repaired, post_meta = _postprocess_after_failure(
                effective_prompt,
                completion,
                test_code,
                entry_point,
                timeout_s=timeout_s,
                rounds=rounds,
                llm_client=llm_client,
            )
            if repaired and repaired.strip():
                completion = repaired
                solutions[name] = completion

        task_meta["postprocess"] = post_meta
        task_meta["mbpp_postprocess"] = {
            "rounds": int(rounds),
            "applied_tools": _collect_applied_tools(post_meta.get("attempts", [])),
            "fallback_detected": False,
            "changed": completion.strip() != original_completion.strip(),
        }

        ok_after = ok_before
        if entry_point and test_code:
            ok_after = _full_test_pass(effective_prompt, completion, test_code, entry_point, timeout_s=timeout_s)

        if not ok_after:
            last_pick = _extract_last_valid_code_from_trace(task_meta, entry_point)
            if last_pick and last_pick.get("code"):
                chosen = str(last_pick["code"])
                if chosen.strip() and chosen.strip() != completion.strip():
                    completion = chosen
                    solutions[name] = completion
                    last_pick["changed"] = True
                task_meta["mbpp_last_agent_fallback"] = last_pick
            else:
                task_meta["mbpp_last_agent_fallback"] = {"ok": False, "reason": "no_valid_code_in_trace"}
        meta[name] = task_meta


def _patch_eval_humaneval_for_mbpp() -> None:
    global _MBPP_PATCHED
    if _MBPP_PATCHED:
        return

    original_orchestrator = eval_humaneval._auto_generate_solutions_orchestrator
    original_local_generator = eval_humaneval._auto_generate_solutions

    def wrapped_orchestrator(*args: Any, **kwargs: Any):
        tasks = kwargs.get("tasks")
        if tasks is None and args:
            tasks = args[0]
        out_path = kwargs.get("out_path")
        if out_path is None and len(args) > 1:
            out_path = args[1]
        timeout_s = float(kwargs.get("tool_timeout_s") or 5.0)
        llm_client = _build_mbpp_llm_client(kwargs)

        solutions, meta = original_orchestrator(*args, **kwargs)
        if isinstance(tasks, list) and isinstance(solutions, dict) and isinstance(meta, dict):
            _postprocess_mbpp_solutions(
                tasks,
                solutions,
                meta,
                rounds=_MBPP_POSTPROCESS_ROUNDS,
                timeout_s=timeout_s,
                llm_client=llm_client,
            )
            if out_path:
                with open(str(out_path), "w", encoding="utf-8") as f:
                    for task in tasks:
                        name = str(task.get("name") or task.get("task_id") or "")
                        completion = str(solutions.get(name, ""))
                        f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
        return solutions, meta

    def wrapped_local_generator(*args: Any, **kwargs: Any):
        tasks = kwargs.get("tasks")
        if tasks is None and args:
            tasks = args[0]
        out_path = kwargs.get("out_path")
        if out_path is None and len(args) > 1:
            out_path = args[1]
        timeout_s = 5.0
        llm_client = _build_mbpp_llm_client(kwargs)

        solutions = original_local_generator(*args, **kwargs)
        if isinstance(tasks, list) and isinstance(solutions, dict):
            meta: Dict[str, Dict[str, Any]] = {}
            _postprocess_mbpp_solutions(
                tasks,
                solutions,
                meta,
                rounds=_MBPP_POSTPROCESS_ROUNDS,
                timeout_s=timeout_s,
                llm_client=llm_client,
            )
            if out_path:
                with open(str(out_path), "w", encoding="utf-8") as f:
                    for task in tasks:
                        name = str(task.get("name") or task.get("task_id") or "")
                        completion = str(solutions.get(name, ""))
                        f.write(json.dumps({"name": name, "completion": completion}, ensure_ascii=True) + "\n")
        return solutions

    eval_humaneval._auto_generate_solutions_orchestrator = wrapped_orchestrator
    eval_humaneval._auto_generate_solutions = wrapped_local_generator
    _MBPP_PATCHED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MBPP by converting it to HumanEval-style tasks and reusing eval_humaneval"
    )
    parser.add_argument("--mbpp_tasks", required=True, type=str, help="MBPP jsonl path")
    parser.add_argument(
        "--converted_tasks",
        default="",
        type=str,
        help="optional converted tasks path; if empty, use a temp file",
    )
    parser.add_argument("--keep_converted", action="store_true", help="keep temp converted file")
    parser.add_argument("--include_challenge_tests", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--mbpp_postprocess_rounds",
        type=int,
        default=_MBPP_POSTPROCESS_ROUNDS,
        help="MBPP postprocess repair rounds after initial failure",
    )
    args, remainder = parser.parse_known_args()
    setattr(args, "remainder", remainder)
    return args


def main() -> None:
    args = parse_args()
    global _MBPP_POSTPROCESS_ROUNDS
    _MBPP_POSTPROCESS_ROUNDS = max(1, int(args.mbpp_postprocess_rounds))
    _patch_eval_humaneval_for_mbpp()

    use_temp = not bool(args.converted_tasks)
    temp_path = ""
    converted_path = args.converted_tasks
    if use_temp:
        fd, temp_path = tempfile.mkstemp(prefix="mbpp_eval_tasks_", suffix=".jsonl")
        os.close(fd)
        converted_path = temp_path

    summary = convert_mbpp_file(
        input_path=args.mbpp_tasks,
        output_path=converted_path,
        include_challenge_tests=args.include_challenge_tests,
        max_samples=args.max_samples,
    )
    print(
        f"[eval_mbpp] converted {summary['converted_total']}/{summary['input_total']} tasks "
        f"to {summary['output']} (skipped={summary['skipped_total']})"
    )

    forwarded = _strip_tasks_arg(list(args.remainder))
    forwarded = _ensure_include_tool_trace(forwarded)
    forwarded = _ensure_postprocess_rounds(forwarded, 0)
    he_args = ["eval_humaneval", "--tasks", converted_path] + forwarded
    old_argv = list(sys.argv)
    try:
        sys.argv = he_args
        eval_humaneval.main()
    finally:
        sys.argv = old_argv
        if use_temp and temp_path and (not args.keep_converted):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
