"""
Fix the _extract_code function in all generated tools to properly handle JSON-wrapped output.
"""
import os
import re
from pathlib import Path

# New improved _extract_code function
NEW_EXTRACT_CODE = r'''def _extract_code(text):
    """提取代码块中的代码，支持markdown和JSON格式"""
    if not text:
        return ""
    
    # 1. 尝试提取markdown代码块
    code_block_match = re.search(r"```(?:python|py|json)?\s*(.*?)```", text, re.DOTALL)
    if code_block_match:
        extracted = code_block_match.group(1).strip()
        # 如果markdown中包含JSON，继续解析
        if extracted.startswith("{") or extracted.startswith("["):
            try:
                data = json.loads(extracted)
                if isinstance(data, dict):
                    for key in ["code_or_commands", "code", "commands", "result"]:
                        if key in data:
                            code = data[key]
                            if isinstance(code, list):
                                return "\n".join(code)
                            return str(code).strip()
            except (json.JSONDecodeError, ValueError):
                pass
        return extracted
    
    # 2. 尝试解析JSON格式 {"code_or_commands": "..."}
    try:
        # 移除可能的"json"前缀和换行
        clean_text = text.strip()
        # 移除开头的 "json" 标记（可能后面跟换行或空格）
        if clean_text.lower().startswith("json"):
            clean_text = re.sub(r'^json\s*', '', clean_text, flags=re.IGNORECASE)
        
        data = json.loads(clean_text)
        if isinstance(data, dict):
            # 尝试常见的键名
            for key in ["code_or_commands", "code", "commands", "result"]:
                if key in data:
                    code = data[key]
                    # 如果是列表，join成字符串
                    if isinstance(code, list):
                        return "\n".join(code)
                    return str(code).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 3. 直接返回文本
    return str(text).strip()'''

def fix_tool_file(file_path):
    """修复单个工具文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换 _extract_code 函数
    pattern = r'def _extract_code\(text\):.*?(?=\ndef [^_]|\nclass |\Z)'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # 手动替换，避免re.sub的转义问题
        before = content[:match.start()]
        after = content[match.end():]
        new_content = before + NEW_EXTRACT_CODE + after
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    """主函数：修复所有工具文件"""
    tools_dir = Path(__file__).parent / "generated_tools"
    
    if not tools_dir.exists():
        print(f"Error: {tools_dir} does not exist")
        return
    
    fixed_count = 0
    total_count = 0
    
    # 遍历所有Python文件
    for file_path in tools_dir.glob("*.py"):
        total_count += 1
        if fix_tool_file(file_path):
            fixed_count += 1
            print(f"Fixed: {file_path.name}")
    
    print(f"\nSummary: Fixed {fixed_count}/{total_count} files")

if __name__ == "__main__":
    main()
