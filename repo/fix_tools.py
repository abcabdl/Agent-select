#!/usr/bin/env python3
"""批量修改所有代码生成工具,添加更强的约束"""
import os
import re
from pathlib import Path

TOOLS_DIR = Path(__file__).parent / "generated_tools"

# 增强的system prompt
ENHANCED_SYSTEM = '''You MUST follow these output rules exactly:
1. Output ONLY raw function body code with 4-space indentation
2. Do NOT include labels (python/json), markdown fences, titles, explanations, or comments
3. CRITICAL: Use ONLY the actual parameter names from the function signature (e.g., if parameter is 'lst', use 'lst' NOT 'data' or 'items')
4. Do NOT use generic variable names like: data, items, input_string, inputs, arr (unless they are the actual parameter names)
5. MUST handle edge cases: empty inputs ([], '', None), single elements, boundary values (0, 1)
6. Generate ONLY the function body that directly solves the problem, NOT generic parsing/processing templates
7. If asked for a function body, do NOT include def/class/import/main/docstrings
8. Empty/whitespace output is invalid'''

def fix_system_prompt(content):
    """替换system prompt为增强版本"""
    # 匹配 system = ( ... ) 这种模式
    pattern = r'(system\s*=\s*\()\s*"[^"]*?"(?:\s+"[^"]*?")*\s*(\))'
    
    def replacer(match):
        return f'{match.group(1)}\n        """{ENHANCED_SYSTEM}"""\n    {match.group(2)}'
    
    # 如果找到旧的system定义,替换它
    if re.search(pattern, content):
        content = re.sub(pattern, replacer, content)
    else:
        # 如果没找到,尝试另一种模式
        pattern2 = r'system\s*=\s*\(\s*"[^"]*"'
        if re.search(pattern2, content):
            content = re.sub(
                pattern2,
                f'system = (\n        """{ENHANCED_SYSTEM}"""',
                content
            )
    
    return content

def enhance_prompt_generation(content):
    """增强生成函数体的prompt"""
    replacements = [
        # 在prompt中强调使用实际参数名
        (
            r'("Task: Generate a Python function body[^"]*")',
            r'\1 + " CRITICAL: Use the exact parameter names from the function signature, NOT generic names like data/items/input_string."'
        ),
        # 在prompt中强调处理边界情况
        (
            r'("Task: Generate a Python function body[^"]*)(\.?")',
            r'\1. Handle edge cases (empty inputs, single elements, boundary values)\2'
        ),
        # 强调不要生成无关模板
        (
            r'("Task: Generate[^"]*function body[^"]*")',
            r'\1 + " Do NOT generate generic parsing/processing boilerplate."'
        ),
    ]
    
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, count=1)
    
    return content

def process_file(filepath):
    """处理单个工具文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # 应用修复
        content = fix_system_prompt(content)
        content = enhance_prompt_generation(content)
        
        # 只有内容改变时才写入
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    if not TOOLS_DIR.exists():
        print(f"Error: {TOOLS_DIR} does not exist")
        return
    
    files = list(TOOLS_DIR.glob("code-generation-*.py"))
    if not files:
        print(f"No code-generation tools found in {TOOLS_DIR}")
        return
    
    print(f"Found {len(files)} tool files")
    modified = 0
    
    for filepath in files:
        if process_file(filepath):
            modified += 1
            print(f"✓ Modified: {filepath.name}")
    
    print(f"\nCompleted: {modified}/{len(files)} files modified")

if __name__ == "__main__":
    main()
