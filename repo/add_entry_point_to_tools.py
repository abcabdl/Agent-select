"""
Add entry_point extraction from task_context to all generated tool files.
"""
import os
import re
from pathlib import Path

TOOLS_DIR = Path(__file__).parent / "generated_tools"

# Pattern to find the run function definition
RUN_FUNC_PATTERN = re.compile(r'^def run\(inputs\):', re.MULTILINE)

# Code to insert at the beginning of run function
ENTRY_POINT_EXTRACTION = '''    # Extract entry_point from task_context if available
    task_context = inputs.get("task_context", {})
    entry_point = task_context.get("entry_point") if isinstance(task_context, dict) else None
'''

def add_entry_point_extraction(filepath):
    """Add entry_point extraction to run() function."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already has task_context extraction
    if 'task_context = inputs.get("task_context"' in content:
        return False
    
    # Find def run(inputs): and insert extraction code
    match = RUN_FUNC_PATTERN.search(content)
    if not match:
        print(f"⚠️  No 'def run(inputs):' found in {filepath.name}")
        return False
    
    # Insert after the def line
    insert_pos = match.end()
    new_content = content[:insert_pos] + '\n' + ENTRY_POINT_EXTRACTION + content[insert_pos:]
    
    # Now find _call_llm calls and add entry_point to prompt if available
    # Look for patterns like: _call_llm(prompt) or _call_llm(some_prompt, ...)
    
    def enhance_prompt(match_obj):
        """Add entry_point hint to prompt construction."""
        # This is complex - for now just add a check before _call_llm
        return match_obj.group(0)
    
    # For simplicity, add a generic hint before first _call_llm
    # Find first _call_llm after our insertion
    llm_call_pattern = r'(\s+)(prompt\s*=\s*["\'])'
    
    def add_entry_hint(match_obj):
        indent = match_obj.group(1)
        rest = match_obj.group(2)
        # Add code before prompt assignment
        hint_code = (
            f'\n{indent}# Add entry_point hint if available\n'
            f'{indent}if entry_point:\n'
            f'{indent}    pass  # TODO: Incorporate entry_point into prompt\n'
            f'{indent}'
        )
        return hint_code + rest
    
    # Actually, let's keep it simpler - just add the extraction
    # Tools can manually use entry_point in their prompts
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def main():
    if not TOOLS_DIR.exists():
        print(f"❌ Tools directory not found: {TOOLS_DIR}")
        return
    
    tool_files = list(TOOLS_DIR.glob("code-generation-*.py"))
    print(f"Found {len(tool_files)} tool files")
    
    modified = 0
    skipped = 0
    
    for filepath in sorted(tool_files):
        if add_entry_point_extraction(filepath):
            print(f"✅ Modified: {filepath.name}")
            modified += 1
        else:
            print(f"⏭️  Skipped: {filepath.name}")
            skipped += 1
    
    print(f"\n📊 Summary:")
    print(f"   Modified: {modified}")
    print(f"   Skipped: {skipped}")
    print(f"   Total: {len(tool_files)}")

if __name__ == "__main__":
    main()
