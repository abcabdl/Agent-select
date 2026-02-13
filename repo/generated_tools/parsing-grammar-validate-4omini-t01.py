import os
import re
import httpx
import json

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)

def _extract_code(text):
    """提取代码块中的代码"""
    if not text:
        return ""
    
    code_block_match = re.search(r"```(?:python|py|json)?\s*(.*?)```", text, re.DOTALL)
    if code_block_match:
        extracted = code_block_match.group(1).strip()
        if extracted.startswith("{") or extracted.startswith("["):
            try:
                data = json.loads(extracted)
                if isinstance(data, dict):
                    for key in ["code_or_commands", "code", "commands", "result"]:
                        if key in data:
                            code = data[key]
                            if isinstance(code, list):
                                return _normalize_indentation("\n".join(code))
                            return _normalize_indentation(str(code).strip())
            except (json.JSONDecodeError, ValueError):
                pass
        return _normalize_indentation(extracted)
    
    try:
        clean_text = text.strip()
        if clean_text.lower().startswith("json"):
            clean_text = re.sub(r'^json\s*', '', clean_text, flags=re.IGNORECASE)
        
        data = json.loads(clean_text)
        if isinstance(data, dict):
            for key in ["code_or_commands", "code", "commands", "result"]:
                if key in data:
                    code = data[key]
                    if isinstance(code, list):
                        return _normalize_indentation("\n".join(code))
                    return _normalize_indentation(str(code).strip())
    except (json.JSONDecodeError, ValueError):
        pass
    
    return _normalize_indentation(str(text).strip())

def _normalize_indentation(code):
    """规范化代码缩进"""
    if not code:
        return ""
    
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return ""
    
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    if min_indent == 0:
        result_lines = []
        for line in lines:
            if line.strip():
                result_lines.append("    " + line)
            else:
                result_lines.append("")
        return "\n".join(result_lines)
    
    return code

def _post_process_parsing(code):
    """
    parsing特定的后处理逻辑
    """
    if not code or not code.strip():
        return code
    
    # 通用代码检查
    if not code.strip().startswith("    "):
        # 确保缩进
        lines = code.split("\n")
        code = "\n".join("    " + line if line.strip() else line for line in lines)
    
    return code


def _format_and_validate(code):
    """调用LLM进行最终格式验证和修正"""
    if not code or not code.strip():
        return code
    
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return _local_format_fix(code)
    
    base = (os.getenv("LLM_API_BASE") or "").rstrip("/")
    if not base:
        return _local_format_fix(code)
    
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    
    system_prompt = """You are a Python code formatter. Your ONLY task is to ensure correct indentation.

CRITICAL RULES:
1. Input is a Python function BODY (without 'def' line)
2. ALL lines must start with exactly 4 spaces as base indentation
3. Preserve internal indentation structure (if/for/while blocks need +4 spaces)
4. Output ONLY the corrected code, no explanations, no markdown
5. Keep all logic unchanged, only fix indentation

Example Input:
if not lst:
return 0
return sum(x)

Example Output:
    if not lst:
        return 0
    return sum(x)"""
    
    user_prompt = f"""Fix indentation for this Python function body. Ensure 4-space base indentation and correct nested indentation:

{code}

Output only the corrected code with proper indentation."""
    
    try:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "max_tokens": 2000,
        }
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        formatted = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # 移除可能的markdown包裹
        formatted = re.sub(r'^```(?:python|py)?\s*', '', formatted.strip(), flags=re.MULTILINE)
        formatted = re.sub(r'```\s*$', '', formatted.strip(), flags=re.MULTILINE)
        
        # 验证格式化后的代码
        if formatted and formatted.strip():
            lines = formatted.split("\n")
            has_indent = any(line.startswith("    ") for line in lines if line.strip())
            if has_indent:
                return formatted
        
        return _local_format_fix(code)
    
    except Exception:
        return _local_format_fix(code)

def _local_format_fix(code):
    """本地格式修正(备用方案)"""
    if not code:
        return ""
    
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return ""
    
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    if min_indent == 0:
        result = []
        for line in lines:
            if line.strip():
                result.append("    " + line)
            else:
                result.append("")
        return "\n".join(result)
    
    return code


def _call_llm(prompt):
    """调用LLM生成代码"""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise Exception("LLM_API_KEY is not set")
    base = (os.getenv("LLM_API_BASE") or "").rstrip("/")
    if not base:
        raise Exception("LLM_API_BASE is not set")
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    
    system = """You are a parsing specialist. Build ASTs and tokenize expressions.
CRITICAL RULES:
1. MUST use exact parameter names from the function signature
2. Output ONLY function body code with 4-space indentation
3. Handle edge cases: empty inputs ([], '', None), single elements
4. No markdown fences, no explanations, no placeholder code
5. Write clean, correct code"""
    
    user = """Task: {prompt}. Use exact parameter names from signature. Return function body only."""
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user.format(prompt=prompt)}
        ],
        "temperature": 0.01,
        "max_tokens": 1000,
    }
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _extract_code(text)

def run(inputs):
    """主函数:生成代码"""
    prompt = inputs.get("prompt", "") or inputs.get("query", "") or inputs.get("task", "")
    if not prompt:
        return {"output": {"code": "# Error: No prompt provided"}}
    
    refinement_request = inputs.get("refinement_request", "")
    failed_code = inputs.get("failed_code", "")
    test_error = inputs.get("test_error", "")
    
    if refinement_request or failed_code:
        enhanced_prompt = prompt
        if refinement_request:
            enhanced_prompt += f"\n\n{refinement_request}"
        elif failed_code and test_error:
            enhanced_prompt += (
                f"\n\n⚠️ PREVIOUS ATTEMPT FAILED:\n"
                f"Failed Code:\n{failed_code}\n\n"
                f"Error: {test_error}\n\n"
                f"Please analyze the error and generate CORRECTED code."
            )
        prompt = enhanced_prompt
    
    try:
        code = _call_llm(prompt)
        code = _post_process_parsing(code)
        return {
            "output": {
                "code": code,
                "success": bool(code and code.strip() and ("# Error" not in str(code)) and ("Error calling LLM" not in str(code)))
            }
        }
    except Exception as e:
        return {
            "output": {
                "code": f"# Error: {str(e)}",
                "success": False
            }
        }
