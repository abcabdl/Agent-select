import json
import os
import re
import httpx

ROLE = 'builder'
FOCUS = 'robustness'
STRATEGY = 'edge_case_first'
MODEL = 'gpt-4o-mini'
TEMPERATURE = 0.18

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\\s*(.*?)```", re.DOTALL)

def _extract_code(text):
    if not text:
        return ""
    match = _CODE_BLOCK_RE.search(str(text))
    if match:
        return match.group(1).strip()
    return str(text).strip()

def _call_llm(system_prompt, user_prompt, response_format=None, max_tokens=800):
    api_key = os.getenv("LLM_API_KEY")
    base = (os.getenv("LLM_API_BASE") or "").rstrip("/")
    if not api_key or not base:
        return ""
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return ""

def run(inputs):
    task = str(inputs.get("task") or inputs.get("query") or "").strip()
    code = str(inputs.get("code") or inputs.get("completion") or inputs.get("text") or "").strip()
    error_hint = str(inputs.get("error_message") or "").strip()
    profile = {"role": ROLE, "focus": FOCUS, "strategy": STRATEGY, "model": MODEL, "temperature": TEMPERATURE}
    if ROLE == "planner":
        steps = [
            "Clarify problem constraints",
            "Decompose into milestones",
            "Prepare handoff checklist",
        ]
        criteria = ["plan_complete", "handoff_ready"]
        if task:
            user_prompt = (
                "Task: " + task + "\n"
                "Focus: " + FOCUS + "\n"
                "Strategy: " + STRATEGY + "\n"
                "Return ONLY JSON with keys steps(list[str]) and acceptance_criteria(list[str])."
            )
            raw = _call_llm(
                "You are a planning tool. Return strict JSON only.",
                user_prompt,
                response_format={"type": "json_object"},
                max_tokens=400,
            )
            try:
                data = json.loads(raw) if raw else {}
                if isinstance(data.get("steps"), list) and data.get("steps"):
                    steps = [str(item) for item in data["steps"] if str(item).strip()] or steps
                if isinstance(data.get("acceptance_criteria"), list) and data.get("acceptance_criteria"):
                    criteria = [str(item) for item in data["acceptance_criteria"] if str(item).strip()] or criteria
            except Exception:
                pass
        return {"output": {"steps": steps, "acceptance_criteria": criteria, "profile": profile}}
    if ROLE == "checker":
        tests = ["happy_path", "edge_case", "invalid_input"]
        verdicts = ["pending", "pending", "pending"]
        localization = "inspect branching logic"
        if task:
            user_prompt = (
                "Task: " + task + "\n"
                "Focus: " + FOCUS + "\n"
                "Strategy: " + STRATEGY + "\n"
                "Return ONLY JSON with keys test_cases(list[str]), verdicts(list[str]), failure_localization(str)."
            )
            raw = _call_llm(
                "You are a testing tool. Return strict JSON only.",
                user_prompt,
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            try:
                data = json.loads(raw) if raw else {}
                if isinstance(data.get("test_cases"), list) and data.get("test_cases"):
                    tests = [str(item) for item in data["test_cases"] if str(item).strip()] or tests
                if isinstance(data.get("verdicts"), list) and data.get("verdicts"):
                    verdicts = [str(item) for item in data["verdicts"] if str(item).strip()] or verdicts
                if isinstance(data.get("failure_localization"), str) and data.get("failure_localization").strip():
                    localization = data.get("failure_localization").strip()
            except Exception:
                pass
        return {"output": {"test_cases": tests, "verdicts": verdicts, "failure_localization": localization, "profile": profile}}
    if ROLE in {"builder", "refractor"}:
        prompt_task = task or "refine the current function body"
        input_code = code or "    return None"
        if ROLE == "builder":
            system_prompt = (
                "You generate Python function body code. "
                "Output ONLY raw function body code with correct indentation. "
                "No markdown, no labels, no def/imports/comments."
            )
            user_prompt = (
                "Task: " + prompt_task + "\n"
                "Focus: " + FOCUS + "\n"
                "Strategy: " + STRATEGY + "\n"
                "Return ONLY the function body code."
            )
        else:
            system_prompt = (
                "You perform minimal Python refactoring on an existing function body. "
                "Preserve behavior and signature assumptions. "
                "Output ONLY the refactored function body code."
            )
            user_prompt = (
                "Task: " + prompt_task + "\n"
                "Focus: " + FOCUS + "\n"
                "Strategy: " + STRATEGY + "\n"
                "Error hint: " + error_hint + "\n"
                "Current code:\n" + input_code + "\n"
                "Return ONLY the refactored function body code."
            )
        raw = _call_llm(system_prompt, user_prompt, max_tokens=900)
        patched = _extract_code(raw)
        if not patched.strip():
            patched = input_code
        return {"code_or_commands": patched, "output": {"profile": profile}}
    return {"output": {"profile": profile}}
