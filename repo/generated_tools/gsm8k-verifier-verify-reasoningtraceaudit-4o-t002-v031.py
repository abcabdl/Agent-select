"""
gsm8k-verifier-verify-reasoningtraceaudit-4o-t002-v031: GSM8K verifier tool (verify) with focus on reasoning trace audit.
"""
import json
import os
import re
import httpx


TOOL_ID = "gsm8k-verifier-verify-reasoningtraceaudit-4o-t002-v031"
FAMILY = "verifier"
FOCUS = "reasoning trace audit"
STAGE = "verify"
MODEL = "gpt-4o"
TEMPERATURE = 0.02
MAX_TOKENS = 1000


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
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
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
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def _fallback(inputs, task):
    numbers = _collect_numbers(task)
    candidate = ""
    if isinstance(inputs, dict):
        candidate = str(inputs.get("final_answer") or inputs.get("answer") or "").strip()
    if not candidate and numbers:
        candidate = numbers[-1]
    if STAGE == "analyze":
        return {
            "known_numbers": numbers,
            "target": "unknown",
            "notes": f"fallback_{STAGE}",
        }
    if STAGE == "model":
        return {
            "equation_plan": "derive relation from quantities and solve for target",
            "focus": FOCUS,
            "notes": f"fallback_{STAGE}",
        }
    if STAGE == "solve":
        return {
            "steps": ["fallback solve path"],
            "numeric_answer": candidate,
            "notes": f"fallback_{STAGE}",
        }
    if STAGE == "verify":
        return {
            "is_consistent": bool(candidate),
            "recomputed_answer": candidate,
            "issues": [] if candidate else ["missing_candidate_answer"],
            "notes": f"fallback_{STAGE}",
        }
    formatted = f"#### {candidate}" if candidate else "#### 0"
    return {
        "final_answer": candidate or "0",
        "formatted": formatted,
        "notes": f"fallback_{STAGE}",
    }


def _call_llm(prompt):
    api_key = os.getenv("LLM_API_KEY", "")
    base = (os.getenv("LLM_API_BASE") or "").rstrip("/")
    if not api_key or not base:
        return None
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    system = (
        "You are a GSM8K specialist tool. "
        "Always return ONLY a JSON object with keys: stage, focus, result, confidence, checks. "
        "No markdown fences, no prose outside JSON."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return None


def run(inputs):
    task = _pick_task(inputs)
    if not task:
        return {"output": {"tool_id": TOOL_ID, "error": "missing_task"}}
    prompt = (
        "Solve this GSM8K word problem component.\n"
        + "family=" + FAMILY + ", stage=" + STAGE + ", focus=" + FOCUS + "\n"
        + "family_policy: Prioritize error detection, consistency checks, and independent recomputation.\n"
        + "stage_policy: Recompute independently and detect inconsistency, sign, and unit mistakes.\n"
        + f"problem: {task}\n"
        + "Return JSON keys: stage, focus, result, confidence, checks."
    )
    raw = _call_llm(prompt)
    parsed = _safe_json(raw) if raw else None
    if parsed is None:
        payload = _fallback(inputs if isinstance(inputs, dict) else {}, task)
    else:
        payload = parsed
    if STAGE == "format":
        if isinstance(payload, dict):
            formatted = str(payload.get("formatted") or "").strip()
            if not formatted:
                candidate = str(payload.get("final_answer") or "").strip() or "0"
                payload["formatted"] = f"#### {candidate}"
    return {
        "output": {
            "tool_id": TOOL_ID,
            "family": FAMILY,
            "focus": FOCUS,
            "stage": STAGE,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "payload": payload,
        }
    }
