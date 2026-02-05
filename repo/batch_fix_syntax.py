"""Batch fix syntax errors in all tools."""
from pathlib import Path

TOOLS_DIR = Path(__file__).parent / "generated_tools"

fixes = [
    # Fix pattern 1: "Task:f"... -> f"Task:...
    ('"Task:f"\\nFunction signature: {func_signature or \\\'unknown\\\'}\\n" + "Task:', 
     'f"Function signature: {func_signature or \'unknown\'}\\n\\nTask:'),
    
    # Fix pattern 2: f"Task:f"... -> f"Task:...  
    ('f"Task:f"\\nFunction signature: {func_signature or \\\'unknown\\\'}\\n" + "Task:',
     'f"Function signature: {func_signature or \'unknown\'}\\n\\nTask:'),
]

def main():
    tools = list(TOOLS_DIR.glob("code-generation-*.py"))
    fixed = 0
    
    for tool in tools:
        content = tool.read_text(encoding='utf-8')
        modified = False
        
        for old, new in fixes:
            if old in content:
                content = content.replace(old, new)
                modified = True
        
        if modified:
            tool.write_text(content, encoding='utf-8')
            print(f"[OK] {tool.name}")
            fixed += 1
    
    print(f"\n[Done] Fixed {fixed}/{len(tools)} files")
    if fixed > 0:
        print("[Next] python auto_sync_tools.py")

if __name__ == "__main__":
    main()
