"""Fix broken syntax in tool files."""
from pathlib import Path
import re

TOOLS_DIR = Path(__file__).parent / "generated_tools"

def fix_broken_syntax(filepath):
    """Fix 'Task:f"...' -> f"Task:...'"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern 1: "Task:f"\n...
    if 'Task:f"' in content:
        content = content.replace('Task:f"', 'f"Task:')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "fixed Task:f"
    
    return False, "no issue"

def main():
    tools = sorted(TOOLS_DIR.glob("code-generation-*.py"))
    fixed = 0
    
    for tool in tools:
        ok, msg = fix_broken_syntax(tool)
        if ok:
            print(f"[FIXED] {tool.name}")
            fixed += 1
    
    print(f"\n[Summary] Fixed: {fixed}")
    if fixed > 0:
        print("[Next] Run: python auto_sync_tools.py")

if __name__ == "__main__":
    main()
