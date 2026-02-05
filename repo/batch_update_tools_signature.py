"""
Batch update all code generation tools to properly extract and use function signatures.
"""
import re
from pathlib import Path

REPO_DIR = Path(__file__).parent
TOOLS_DIR = REPO_DIR / "generated_tools"

# Pattern to find the old task_context extraction code
OLD_PATTERN = re.compile(
    r'(def run\(inputs\):.*?)'
    r'task_context = inputs\.get\("task_context", \{\}\)\s*'
    r'entry_point = task_context\.get\("entry_point"\) if isinstance\(task_context, dict\) else None\s*'
    r'(\s*# Extract function signature from task)',
    re.DOTALL
)

# Also match the simpler pattern without the comment
OLD_PATTERN_SIMPLE = re.compile(
    r'(task_context = inputs\.get\("task_context", \{\}\)\s*)'
    r'(entry_point = task_context\.get\("entry_point"\) if isinstance\(task_context, dict\) else None)',
    re.MULTILINE
)

# Patterns to match
OLD_CODE_PATTERN = '''task_context = inputs.get("task_context", {})
    entry_point = task_context.get("entry_point") if isinstance(task_context, dict) else None'''

# New code to replace with
NEW_CODE = '''task_context = inputs.get("task_context", {})
    entry_point = None
    func_signature = None
    param_names = []
    
    if isinstance(task_context, dict):
        entry_point = task_context.get("entry_point")
        func_sig_info = task_context.get("function_signature")
        if func_sig_info and isinstance(func_sig_info, dict):
            param_names = func_sig_info.get("parameters", [])
            if entry_point and param_names:
                func_signature = f"def {entry_point}({', '.join(param_names)}):"
    
    # Fallback: Extract function signature from task text
    task_text = inputs.get("task", "") or inputs.get("query", "")
    if not func_signature and task_text:
        import re as _re
        match = _re.search(r'def\\s+(\\w+)\\s*\\(([^)]*)\\)\\s*:', task_text, _re.MULTILINE)
        if match:
            func_signature = match.group(0)
            params_str = match.group(2).strip()
            if params_str and not param_names:
                for param in params_str.split(","):
                    param = param.strip()
                    if param:
                        param_name = param.split(":")[0].split("=")[0].strip()
                        if param_name and param_name not in ("*", "**"):
                            param_names.append(param_name)'''

def update_tool_file(tool_path: Path) -> bool:
    """Update a single tool file."""
    try:
        content = tool_path.read_text(encoding='utf-8')
        original = content
        
        # Skip if already updated (check for "function_signature")
        if 'function_signature' in content and 'param_names = []' in content:
            return False
        
        # Simple string replacement
        if OLD_CODE_PATTERN in content:
            content = content.replace(OLD_CODE_PATTERN, NEW_CODE, 1)
            
            if content != original:
                tool_path.write_text(content, encoding='utf-8')
                return True
        
        return False
    except Exception as e:
        print(f"Error updating {tool_path.name}: {e}")
        return False

def add_param_warning_to_prompts(tool_path: Path) -> bool:
    """Add parameter warning to LLM prompts in tool files."""
    try:
        content = tool_path.read_text(encoding='utf-8')
        original = content
        
        # Skip if already has param_warning
        if 'param_warning = ""' in content or 'param_warning =' in content:
            return False
        
        # Find where prompts are constructed (look for _call_llm or similar)
        # This is more complex, so we'll add a template section before any prompt construction
        
        # Look for common prompt patterns and add warning logic before them
        patterns = [
            (r'(mode = "[^"]*"\s*\n)', r'\1    param_warning = ""\n    if param_names:\n        param_warning = f"\\n\\nCRITICAL: The function parameters are: {\', \'.join(param_names)}. You MUST use these exact names in your code, NOT generic names like \'data\', \'items\', \'input_string\', etc."\n    \n'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content, count=1)
                break
        
        if content != original:
            tool_path.write_text(content, encoding='utf-8')
            return True
        
        return False
    except Exception as e:
        print(f"Error adding warning to {tool_path.name}: {e}")
        return False

def main():
    """Update all code generation tools."""
    tool_files = sorted(TOOLS_DIR.glob("code-generation-*.py"))
    
    print(f"Found {len(tool_files)} code generation tools")
    print("=" * 60)
    
    updated_sig = 0
    skipped = 0
    failed = 0
    
    for tool_file in tool_files:
        result = update_tool_file(tool_file)
        
        if result:
            print(f"✅ Updated: {tool_file.name}")
            updated_sig += 1
        elif 'function_signature' in tool_file.read_text(encoding='utf-8'):
            skipped += 1
        else:
            failed += 1
            print(f"⚠️  Could not update: {tool_file.name}")
    
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Updated: {updated_sig}")
    print(f"  Already up-to-date: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(tool_files)}")
    
    if updated_sig > 0:
        print(f"\n✅ Successfully updated {updated_sig} tools!")
        print("   Run 'python sync_tools_to_db.py' to sync changes to database.")
    else:
        print("\n✓ All tools are already up-to-date!")

if __name__ == "__main__":
    main()
