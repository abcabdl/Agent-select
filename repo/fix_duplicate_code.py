"""
Fix duplicate code in tools that have both old and new extraction logic.
"""
from pathlib import Path
import re

REPO_DIR = Path(__file__).parent
TOOLS_DIR = REPO_DIR / "generated_tools"

def fix_tool_file(tool_path: Path) -> bool:
    """Fix a single tool file by removing duplicate extraction code."""
    try:
        content = tool_path.read_text(encoding='utf-8')
        original = content
        
        # Check if file has the problem: both new code (with param_names) and old code
        has_new_code = 'param_names = []' in content and 'func_sig_info = task_context.get("function_signature")' in content
        has_old_duplicate = content.count('# Extract function signature from task') > 1 or \
                           content.count('# Fallback: Extract function signature from task') > 0 and \
                           content.count('task_text = inputs.get("task"') > 2
        
        if not (has_new_code and has_old_duplicate):
            return False
        
        # Find where the new code ends (look for the line after param_names.append)
        # Then remove any duplicate extraction code that follows
        
        # Pattern: Look for duplicate "Extract function signature from task" comment after the new code
        pattern = r'(param_names\.append\(param_name\)\s*\n\s*# Extract function signature from task.*?func_signature = match\.group\(0\))'
        
        if re.search(pattern, content, re.DOTALL):
            # Remove the duplicate extraction block
            content = re.sub(
                r'\n\s*# Extract function signature from task \(CRITICAL for knowing parameter names!\)\s*\n\s*task_text = inputs\.get\("task", ""\) or inputs\.get\("query", ""\)\s*\n\s*func_signature = None\s*\n\s*if task_text:\s*\n\s*import re as _re\s*\n\s*match = _re\.search\(r\'def\\s\+\(\\w\+\)\\s\*\\\(\[^\)\]\*\\\)\\s\*:\', task_text, _re\.MULTILINE\)\s*\n\s*if match:\s*\n\s*func_signature = match\.group\(0\)',
                '',
                content
            )
        
        # Also add param_warning if missing
        if 'param_warning = ""' not in content and 'param_names' in content:
            # Add param_warning construction before any mode= line
            content = re.sub(
                r'(\n\s*)(mode = "[^"]*")',
                r'\1# Build parameter warning for LLM\n\1param_warning = ""\n\1if param_names:\n\1    param_warning = f"\\n\\nCRITICAL: The function parameters are: {\', \'.join(param_names)}. You MUST use these exact names in your code, NOT generic names like \'data\', \'items\', \'input_string\', etc."\n\1\n\1\2',
                content,
                count=1
            )
        
        # Update prompts to include param_warning
        if 'param_warning' in content:
            # Update outline_prompt
            content = re.sub(
                r'outline_prompt = \(\s*f"Function signature: \{func_signature or \'unknown\'\}\\n\\n',
                r'outline_prompt = (\n        f"Function signature: {func_signature or \'unknown\'}{param_warning}\\n\\n',
                content
            )
            
            # Update main prompt
            content = re.sub(
                r'prompt = \(\s*f"Task: Generate',
                r'prompt = (\n        f"Function signature: {func_signature or \'unknown\'}{param_warning}\\n\\n"\n        f"Task: Generate',
                content
            )
        
        if content != original:
            tool_path.write_text(content, encoding='utf-8')
            return True
        
        return False
    except Exception as e:
        print(f"Error fixing {tool_path.name}: {e}")
        return False

def main():
    """Fix all code generation tools."""
    tool_files = sorted(TOOLS_DIR.glob("code-generation-*.py"))
    
    print(f"Checking {len(tool_files)} code generation tools for duplicate code...")
    print("=" * 60)
    
    fixed = 0
    
    for tool_file in tool_files:
        result = fix_tool_file(tool_file)
        
        if result:
            print(f"✅ Fixed: {tool_file.name}")
            fixed += 1
    
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Fixed: {fixed}")
    print(f"  Total: {len(tool_files)}")
    
    if fixed > 0:
        print(f"\n✅ Successfully fixed {fixed} tools!")
        print("   Run 'python sync_tools_to_db.py' to sync changes to database.")
    else:
        print("\n✓ No duplicate code found!")

if __name__ == "__main__":
    main()
