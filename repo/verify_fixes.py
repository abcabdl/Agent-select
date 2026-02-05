#!/usr/bin/env python3
"""验证工具修改是否有效"""
import re
from pathlib import Path

TOOLS_DIR = Path(__file__).parent / "generated_tools"

def check_tool(filepath):
    """检查工具是否包含增强的约束"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "有system prompt": 'system = (' in content or 'system=' in content,
        "提到parameter names": 'parameter names' in content.lower() or 'parameter is' in content.lower(),
        "提到edge cases": 'edge case' in content.lower(),
        "禁止generic names": 'generic' in content.lower() or 'data, items' in content.lower(),
        "禁止parsing templates": 'parsing' in content.lower() or 'template' in content.lower(),
    }
    
    return checks

def main():
    files = list(TOOLS_DIR.glob("code-generation-generate*.py"))[:5]  # 检查前5个
    
    print(f"验证前5个工具文件的修改...\n")
    
    for filepath in files:
        print(f"📄 {filepath.name}")
        checks = check_tool(filepath)
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        print()

if __name__ == "__main__":
    main()
