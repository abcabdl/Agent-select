#!/usr/bin/env python3
"""
修复工具中的 _is_valid 函数和 fallback 逻辑，使其能够正确处理缩进问题
"""

import os
import re
from pathlib import Path

def fix_is_valid_function(file_path: str) -> tuple[bool, str]:
    """修复单个文件中的 _is_valid 函数和 fallback 逻辑"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes = []
    modified = False
    
    # 1. 修复 _is_valid 函数
    if 'def _is_valid(body):' in content:
        old_pattern = r'''def _is_valid\(body\):
    if not body:
        return False
    try:
        ast\.parse\("def _tmp\(\):\\n" \+ body\)
        return True
    except Exception:
        return False'''
        
        new_implementation = '''def _is_valid(body):
    if not body:
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
        return True
    except Exception:
        return False'''
        
        new_content = re.sub(old_pattern, new_implementation, content)
        if new_content != content:
            content = new_content
            modified = True
            changes.append("_is_valid函数")
    
    # 2. 添加日志记录到 fallback 位置（帮助调试）
    # 查找并修复: if not _is_valid(body): body = "    return None"
    fallback_pattern1 = r'''(    if not _is_valid\(body\):
        body = "    return None")'''
    
    fallback_replacement1 = r'''    if not _is_valid(body):
        # Debug: Log validation failure
        import sys
        print(f"WARNING: Code validation failed in {__file__}: {body[:100]}", file=sys.stderr)
        body = "    return None"'''
    
    new_content = re.sub(fallback_pattern1, fallback_replacement1, content)
    if new_content != content:
        content = new_content
        modified = True
        changes.append("添加验证失败日志")
    
    # 3. 修复直接设置 fallback 的情况（没有 _is_valid 检查）
    # 查找: if not body: body = "    return None"
    simple_fallback_pattern = r'(    if not body:\n        body = "    return None")'
    simple_fallback_replacement = r'''    if not body:
        import sys
        print(f"WARNING: Empty code generated in {__file__}", file=sys.stderr)
        body = "    return None"'''
    
    new_content = re.sub(simple_fallback_pattern, simple_fallback_replacement, content)
    if new_content != content:
        content = new_content
        modified = True
        changes.append("添加空代码日志")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, ", ".join(changes)
    return False, ""

def main():
    # 查找所有问题工具
    tools_dir = Path(__file__).parent / 'generated_tools'
    
    # 实际上应该修复所有使用这个模式的工具
    fixed_count = 0
    failed_count = 0
    
    print("开始修复工具中的验证逻辑...")
    print()
    
    for tool_file in tools_dir.glob('code-generation-*.py'):
        try:
            modified, changes = fix_is_valid_function(str(tool_file))
            if modified:
                fixed_count += 1
                print(f"✓ 修复: {tool_file.name} ({changes})")
        except Exception as e:
            failed_count += 1
            print(f"✗ 失败: {tool_file.name} - {e}")
    
    print()
    print(f"完成！修复了 {fixed_count} 个工具，{failed_count} 个失败")
    print()
    
    if fixed_count > 0:
        print("=" * 60)
        print("重要提示：")
        print("1. 工具验证逻辑已优化，现在能处理未缩进的代码")
        print("2. 添加了调试日志，失败时会输出到 stderr")
        print("3. 建议重新运行评估以验证修复效果")
        print("=" * 60)

if __name__ == '__main__':
    main()
