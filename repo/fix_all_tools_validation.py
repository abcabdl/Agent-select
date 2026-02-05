#!/usr/bin/env python3
"""
为所有工具添加统一的代码验证逻辑
"""

import os
import re
import ast
from pathlib import Path

# 统一的验证函数定义
VALIDATION_FUNCTION = '''
def _is_valid(body):
    """验证生成的代码是否有效（语法正确且有实际内容）"""
    if not body or not body.strip():
        return False
    try:
        # Ensure proper indentation for function body
        lines = body.split('\\n')
        indented_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                # Add 4-space indent if not already indented
                if not line.startswith(' ') and not line.startswith('\\t'):
                    indented_lines.append('    ' + line)
                else:
                    indented_lines.append(line)
            else:
                indented_lines.append(line)
        indented_body = '\\n'.join(indented_lines)
        ast.parse("def _tmp():\\n" + indented_body)
        
        # Check if code is meaningful (not just "return None")
        stripped = body.strip().lower()
        if stripped in ("return none", "none", "pass"):
            return False
        
        return True
    except Exception:
        return False
'''

def add_validation_to_tool(file_path: str) -> tuple[bool, str]:
    """为工具添加或改进验证逻辑"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    changes = []
    
    # 1. 如果已经有 _is_valid 函数，确保使用改进版本
    if 'def _is_valid(body):' in content:
        # 替换旧的实现
        old_simple_pattern = r'def _is_valid\(body\):[\s\S]*?(?=\ndef |\Z)'
        if re.search(old_simple_pattern, content):
            # 找到函数的位置
            match = re.search(r'(def _is_valid\(body\):[\s\S]*?)(?=\ndef |\Z)', content)
            if match:
                old_func = match.group(1)
                # 只替换如果不是新版本
                if 'Ensure proper indentation' not in old_func:
                    content = content.replace(old_func, VALIDATION_FUNCTION + '\n')
                    modified = True
                    changes.append("更新_is_valid函数")
    
    # 2. 如果没有 _is_valid 函数，添加它
    elif 'def _is_valid' not in content:
        # 找到 _call_llm 函数后面插入
        if 'def _call_llm' in content:
            # 在 _call_llm 函数后插入
            pattern = r'(def _call_llm\([^)]*\):[\s\S]*?return [^\n]+\n)'
            match = re.search(pattern, content)
            if match:
                insert_pos = match.end()
                content = content[:insert_pos] + '\n' + VALIDATION_FUNCTION + '\n' + content[insert_pos:]
                modified = True
                changes.append("添加_is_valid函数")
    
    # 3. 确保在使用 body 前进行验证
    # 模式1: body = _call_llm(...) 后面直接 if not body:
    pattern1 = r'(    body(?:_text)? = _call_llm\([^)]+\))\n(    (?:body = _normalize_body\(body_text\)\n    )?if not body:\n        (?:import sys\n        print.*?\n        )?body = "    return None")'
    
    replacement1 = r'''\1
    if not _is_valid(body):
        # Try to fix invalid code
        fix_prompt = (
            "Task: Fix this Python function body to be syntactically valid. "
            "Return ONLY the corrected function body with proper indentation. "
            "Body to fix: " + str(body)
        )
        try:
            body = _call_llm(fix_prompt)
        except Exception:
            pass
    
    if not _is_valid(body):
        import sys
        print(f"WARNING: Code validation failed in {__file__}: body={str(body)[:100]}", file=sys.stderr)
        body = "    return None"'''
    
    new_content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
    if new_content != content:
        content = new_content
        modified = True
        changes.append("添加验证和修复逻辑")
    
    # 模式2: 处理 body = _normalize_body(body_text) 的情况
    pattern2 = r'(    body_text = _call_llm\([^)]+\))\n(    body = _normalize_body\(body_text\))\n(    if not body:\n        (?:import sys\n        print.*?\n        )?body = "    return None")'
    
    replacement2 = r'''\1
\2
    if not _is_valid(body):
        # Try to fix invalid code
        fix_prompt = (
            "Task: Fix this Python function body to be syntactically valid. "
            "Return ONLY the corrected function body with proper indentation. "
            "Body to fix: " + str(body)
        )
        try:
            fixed_text = _call_llm(fix_prompt)
            body = _normalize_body(fixed_text)
        except Exception:
            pass
    
    if not _is_valid(body):
        import sys
        print(f"WARNING: Code validation failed in {__file__}: body={str(body)[:100]}", file=sys.stderr)
        body = "    return None"'''
    
    new_content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE)
    if new_content != content:
        content = new_content
        modified = True
        changes.append("添加验证和修复逻辑(normalize版本)")
    
    # 4. 对于已经有 _is_valid 检查的，确保有日志
    pattern2 = r'(    if not _is_valid\(body\):\n)(        body = "    return None")'
    replacement2 = r'''\1        import sys
        print(f"WARNING: Code validation failed in {__file__}: body={str(body)[:100]}", file=sys.stderr)
\2'''
    
    new_content = re.sub(pattern2, replacement2, content)
    if new_content != content:
        content = new_content
        modified = True
        changes.append("添加验证失败日志")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, ", ".join(changes)
    return False, ""

def main():
    tools_dir = Path(__file__).parent / 'generated_tools'
    
    fixed_count = 0
    failed_count = 0
    
    print("开始为所有工具添加统一验证逻辑...")
    print()
    
    for tool_file in sorted(tools_dir.glob('code-generation-*.py')):
        try:
            modified, changes = add_validation_to_tool(str(tool_file))
            if modified:
                fixed_count += 1
                print(f"✓ {tool_file.name}: {changes}")
        except Exception as e:
            failed_count += 1
            print(f"✗ {tool_file.name}: {e}")
    
    print()
    print(f"{'='*70}")
    print(f"完成！处理了 {fixed_count} 个工具，{failed_count} 个失败")
    print(f"{'='*70}")
    
    if fixed_count > 0:
        print()
        print("改进内容：")
        print("1. ✅ 所有工具现在都有 _is_valid() 验证函数")
        print("2. ✅ 验证函数会自动处理缩进问题")
        print("3. ✅ 验证失败时会尝试让LLM修复代码")
        print("4. ✅ 最终验证失败会输出详细日志到stderr")
        print("5. ✅ 过滤掉无意义的代码（如只有return None）")
        print()
        print("建议：重新运行评估测试验证改进效果")

if __name__ == '__main__':
    main()
