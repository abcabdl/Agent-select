import os
import re
import httpx
from collections import deque
import heapq

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)

def _extract_code(text):
    """提取代码块中的代码"""
    if not text:
        return ""
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return str(text).strip()

def _call_llm(prompt):
    """调用LLM生成代码"""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise Exception("LLM_API_KEY is not set")
    base = (os.getenv("LLM_API_BASE") or "").rstrip("/")
    if not base:
        raise Exception("LLM_API_BASE is not set")
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    
    system = """You are a data structure designer. Specialize in class design and data organization.
CRITICAL RULES:
1. Design proper class structure with __init__, methods, and properties
2. Output ONLY class definition or data structure code with 4-space indentation
3. Include magic methods (__str__, __repr__, __len__) when appropriate
4. Use appropriate data structures (list, dict, set, deque, heap)
5. No markdown fences, no explanations
6. Ensure O(1) or O(log n) operations where possible"""
    
    user = """Design data structure: {prompt}. Include efficient methods. Return complete class definition."""
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user.format(prompt=prompt)}
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _extract_code(text)

def run(inputs):
    """主函数：生成代码"""
    prompt = inputs.get("prompt", "") or inputs.get("query", "") or inputs.get("task", "")
    if not prompt:
        return {"output": {"code": "# Error: No prompt provided"}}
    
    try:
        code = _call_llm(prompt)
        return {
            "output": {
                "code": code,
                "success": bool(code and code.strip())
            }
        }
    except Exception as e:
        return {
            "output": {
                "code": f"# Error: {str(e)}",
                "success": False,
                "error": str(e)
            }
        }
