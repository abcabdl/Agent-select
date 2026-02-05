"""
CRITICAL FIX: Make tools extract and use function signature from task.
This tells LLM the actual parameter names!
"""
import re
from pathlib import Path

TOOLS_DIR = Path(__file__).parent / "generated_tools"

def fix_tool(filepath):
    """Add function signature extraction and pass to LLM."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip if already has func_signature
    if 'func_signature' in content:
        return False, "has func_signature"
    
    # Must have entry_point extraction
    if 'entry_point = task_context.get("entry_point")' not in content:
        return False, "no entry_point"
    
    # Replace entry_point extraction with enhanced version
    old = '''    # Extract entry_point from task_context if available
    task_context = inputs.get("task_context", {})
    entry_point = task_context.get("entry_point") if isinstance(task_context, dict) else None'''
    
    new = '''    # Extract entry_point and function signature
    task_context = inputs.get("task_context", {})
    entry_point = task_context.get("entry_point") if isinstance(task_context, dict) else None
    
    # Extract function signature from task (CRITICAL for knowing parameter names!)
    task_text = inputs.get("task", "") or inputs.get("query", "")
    func_signature = None
    if task_text:
        import re as _re
        match = _re.search(r'def\\s+(\\w+)\\s*\\([^)]*\\)\\s*:', task_text, _re.MULTILINE)
        if match:
            func_signature = match.group(0)'''
    
    if old not in content:
        return False, "pattern not found"
    
    content = content.replace(old, new)
    
    # Add signature to prompt - find where prompt is constructed
    if 'prompt = (' in content or 'prompt = f"' in content or 'prompt = "' in content:
        # Add before first prompt construction
        patterns = [
            # Match: prompt = (f"Task: or prompt = ("Task:
            (r'(prompt\s*=\s*\(\s*)(f?)"Task:', r'\1f"Function signature: {func_signature or \'unknown\'}\\n\\nTask:'),
            # Match: prompt = f"Task:
            (r'(prompt\s*=\s*)(f)"Task:', r'\1f"Function signature: {func_signature or \'unknown\'}\\n\\nTask:'),
            # Match: prompt = "Task:
            (r'(prompt\s*=\s*)"Task:', r'\1f"Function signature: {func_signature or \'unknown\'}\\n\\nTask:'),
        ]
        
        for pat, repl in patterns:
            if re.search(pat, content):
                content = re.sub(pat, repl, content, count=1)
                break
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True, "fixed"

def main():
    tools = sorted(TOOLS_DIR.glob("code-generation-*.py"))
    print(f"Fixing {len(tools)} tools...\n")
    
    fixed, skipped = 0, 0
    reasons = {}
    
    for tool in tools:
        ok, reason = fix_tool(tool)
        if ok:
            print(f"[OK] {tool.name}")
            fixed += 1
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
            skipped += 1
    
    print(f"\n[Summary] Fixed: {fixed}, Skipped: {skipped}")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")
    
    if fixed > 0:
        print(f"\n[Next] Run: python auto_sync_tools.py")

if __name__ == "__main__":
    main()
