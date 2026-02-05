"""
添加二次格式化检查:在返回前调用LLM验证并修正缩进
"""
import os
import glob
import re

def create_format_check_function():
    """创建格式检查和修正函数"""
    return r'''
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
        formatted = re.sub(r'^```(?:python|py)?\\s*', '', formatted.strip(), flags=re.MULTILINE)
        formatted = re.sub(r'```\\s*$', '', formatted.strip(), flags=re.MULTILINE)
        
        # 验证格式化后的代码
        if formatted and formatted.strip():
            # 检查是否有基础缩进
            lines = formatted.split("\\n")
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
    
    lines = code.split("\\n")
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
        return "\\n".join(result)
    
    return code
'''

def update_all_tools():
    """更新所有工具,添加格式检查"""
    tools_dir = r"c:\Users\zrz20\Desktop\vscode\Agent\Agent-router\repo\generated_tools"
    tool_files = glob.glob(os.path.join(tools_dir, "code-generation-*.py"))
    
    print(f"找到 {len(tool_files)} 个工具文件")
    
    format_check_func = create_format_check_function()
    
    for tool_file in tool_files:
        try:
            with open(tool_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 1. 在_normalize_indentation后添加_format_and_validate和_local_format_fix函数
            pattern1 = r'(def _normalize_indentation\(code\):.*?return code\n)'
            
            if not re.search(r'def _format_and_validate\(code\):', content):
                content = re.sub(
                    pattern1,
                    r'\1' + format_check_func + '\n',
                    content,
                    flags=re.DOTALL
                )
            
            # 2. 修改run()函数,在返回前调用_format_and_validate
            # 查找 code = _call_llm(prompt) 后面的返回语句
            pattern2 = r'(code = _call_llm\(prompt\)\s+)(return {)'
            
            replacement2 = r'\1# 最终格式验证和修正\n        code = _format_and_validate(code)\n        \2'
            
            if re.search(pattern2, content):
                content = re.sub(pattern2, replacement2, content)
            else:
                # 备用模式: 查找 return {"output": {"code": code,
                pattern3 = r'(code = _call_llm\(prompt\)\s+)(return\s+{[^}]*"code":\s*code)'
                if re.search(pattern3, content):
                    replacement3 = r'\1# 最终格式验证和修正\n        code = _format_and_validate(code)\n        \2'
                    content = re.sub(pattern3, replacement3, content)
            
            with open(tool_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ {os.path.basename(tool_file)}")
        
        except Exception as e:
            print(f"✗ {os.path.basename(tool_file)} - 错误: {e}")
    
    print("\n修复完成!")

if __name__ == "__main__":
    update_all_tools()
