import os
import re
import httpx
import json


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)

def _extract_code(text):
    """提取代码块中的代码,支持markdown和JSON格式,并规范化缩进"""
    if not text:
        return ""
    
    # 1. 尝试提取markdown代码块
    code_block_match = re.search(r"```(?:python|py|json)?\s*(.*?)```", text, re.DOTALL)
    if code_block_match:
        extracted = code_block_match.group(1).strip()
        # 如果markdown中包含JSON,继续解析
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
    
    # 2. 尝试解析JSON格式 {"code_or_commands": "..."}
    try:
        # 移除可能的"json"前缀和换行
        clean_text = text.strip()
        # 移除开头的 "json" 标记(可能后面跟换行或空格)
        if clean_text.lower().startswith("json"):
            clean_text = re.sub(r'^json\s*', '', clean_text, flags=re.IGNORECASE)
        
        data = json.loads(clean_text)
        if isinstance(data, dict):
            # 尝试常见的键名
            for key in ["code_or_commands", "code", "commands", "result"]:
                if key in data:
                    code = data[key]
                    # 如果是列表,join成字符串
                    if isinstance(code, list):
                        return _normalize_indentation("\n".join(code))
                    return _normalize_indentation(str(code).strip())
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 3. 直接返回文本
    return _normalize_indentation(str(text).strip())

def _normalize_indentation(code):
    """规范化代码缩进,确保所有行都有适当的4空格缩进"""

    if not code:
        return ""
    
    lines = code.split("\n")
    if not lines:
        return ""
    
    # 移除完全空白的行(开头和结尾)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return ""
    
    # 找到最小缩进级别(非空行)
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # 只看非空行
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    # 如果最小缩进是0,说明有代码顶格写,需要统一添加4空格
    if min_indent == 0:
        result_lines = []
        for line in lines:
            if line.strip():  # 非空行直接加4空格前缀
                result_lines.append("    " + line)
            else:  # 空行保持空
                result_lines.append("")
        return "\n".join(result_lines)
    
    # 如果已经有缩进,保持原样
    return code

def _format_and_validate(code):
    """调用LLM进行最终格式验证和修正"""
    if not code or not code.strip():
        return code
    
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        # 如果没有API密钥,尝试本地修正
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
        
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        formatted = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # 移除可能的markdown包裹
        formatted = re.sub(r'^```(?:python|py)?\s*', '', formatted.strip(), flags=re.MULTILINE)
        formatted = re.sub(r'```\s*$', '', formatted.strip(), flags=re.MULTILINE)
        
        # 验证格式化后的代码
        if formatted and formatted.strip():
            # 检查是否有基础缩进
            lines = formatted.split("\n")
            has_indent = any(line.startswith("    ") for line in lines if line.strip())
            if has_indent:
                return formatted
        
        # 如果格式化失败,使用本地修正
        return _local_format_fix(code)
    
    except Exception:
        # API调用失败,使用本地修正
        return _local_format_fix(code)

def _local_format_fix(code):
    """本地格式修正(备用方案)"""
    if not code:
        return ""
    
    lines = code.split("\n")
    # 移除首尾空行
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return ""
    
    # 找最小缩进
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    # 如果最小缩进为0,统一添加4空格
    if min_indent == 0:
        result = []
        for line in lines:
            if line.strip():
                result.append("    " + line)
            else:
                result.append("")
        return "\n".join(result)
    
    return code



def run(inputs):
    task = inputs.get("task") or inputs.get("query") or ""
    scope = inputs.get("scope") or "general"
    constraints = inputs.get("constraints") or []
    if not isinstance(task, str):
        task = str(task)
    if isinstance(constraints, str):
        constraints = [c.strip() for c in constraints.split(",") if c.strip()]
    steps = []
    mode = "organize2"
    if task.strip():
        steps.append(f"Define goal for {{task}} ({{scope}})")
        steps.extend(['"Organize2 step 1"', '"Organize2 step 2"', '"Review and finalize"'])
    return {"output": {"mode": mode, "steps": steps, "constraints": constraints}}
