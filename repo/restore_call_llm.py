"""
Restore the missing _call_llm function in all affected tool files.
"""
import re
from pathlib import Path

REPO_DIR = Path(__file__).parent
TOOLS_DIR = REPO_DIR / "generated_tools"

def fix_tool_file(file_path):
    """Restore _call_llm function if missing"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if _call_llm function is missing
    if 'def _call_llm' in content:
        return False  # Already has the function
    
    # Check if the file has the run() function that calls _call_llm
    if 'code = _call_llm(prompt)' not in content:
        return False  # Doesn't use _call_llm
    
    # Find where to insert _call_llm (after _extract_code function)
    pattern = r'(def _extract_code\(text\):.*?return str\(text\)\.strip\(\))\n'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print(f"  WARNING: Could not find insertion point in {file_path.name}")
        return False
    
    # Insert _call_llm function after _extract_code
    insert_pos = match.end()
    
    # Determine the LLM system prompt based on tool name
    tool_name = file_path.stem
    
    # Default system prompt
    system_prompt = '''"""You are a code generator. Create clean, efficient implementations.
CRITICAL RULES:
1. Output ONLY function body code with 4-space indentation
2. Use exact parameter names from signature
3. Handle edge cases: None, empty, boundary values
4. No markdown fences, no explanations
5. Direct implementation without unnecessary abstractions"""'''
    
    user_prompt = '''"""Generate code: {prompt}. Use given parameters. Return body only."""'''
    
    # Customize based on tool type
    if 'assemblesnippets' in tool_name:
        system_prompt = '''"""You are a code snippet assembler. Your specialty is combining code fragments correctly.
CRITICAL RULES:
1. MUST use exact parameter names from the function signature
2. Output ONLY function body code with 4-space indentation
3. Handle edge cases: empty inputs ([], '', None), single elements
4. No markdown fences, no explanations, no placeholder code
5. Assemble snippets logically to solve the specific problem"""'''
        user_prompt = '''"""Task: {prompt}. Use EXACT parameter names from signature. Return function body only."""'''
    
    elif 'generatefunctionbody' in tool_name:
        system_prompt = '''"""You are a function body generator. Create clean, direct implementations.
CRITICAL RULES:
1. Generate ONLY function body without 'def' line
2. Output with 4-space indentation
3. Use exact parameter names from signature
4. Include return statement (never 'return None' or 'pass')
5. No markdown fences, no explanations
6. Direct implementation without unnecessary abstractions"""'''
        user_prompt = '''"""Generate function body: {prompt}. Use given parameters. Return body only."""'''
    
    elif 'generateedgecase' in tool_name:
        system_prompt = '''"""You are an edge case handler. Specialize in robust input validation and boundary conditions.
CRITICAL RULES:
1. Handle ALL edge cases: None, empty, single element, negative, zero, large numbers
2. Add explicit checks with meaningful error messages or default returns
3. Output ONLY function body code with 4-space indentation
4. Test boundary values: [], [1], [1,1], very large/small inputs
5. Use try-except for potential errors
6. No markdown fences, comprehensive edge case coverage"""'''
        user_prompt = '''"""Implement with edge cases: {prompt}. Handle None, empty, boundary values. Return function body."""'''
    
    elif 'generatedatastructure' in tool_name:
        system_prompt = '''"""You are a data structure designer. Specialize in class design and data organization.
CRITICAL RULES:
1. Design proper class structure with __init__, methods, and properties
2. Output ONLY class definition or data structure code with 4-space indentation
3. Include magic methods (__str__, __repr__, __len__) when appropriate
4. Use appropriate data structures (list, dict, set, deque, heap)
5. No markdown fences, no explanations
6. Ensure O(1) or O(log n) operations where possible"""'''
        user_prompt = '''"""Design data structure: {prompt}. Include efficient methods. Return complete class definition."""'''
    
    elif 'generatealgorithm' in tool_name:
        system_prompt = '''"""You are an algorithm designer. Specialize in efficient algorithmic solutions.
CRITICAL RULES:
1. Design optimal algorithm with clear time/space complexity
2. Output ONLY function body code with 4-space indentation
3. Use appropriate algorithms (sorting, searching, DP, greedy, etc.)
4. Optimize for performance
5. No markdown fences, no explanations
6. Include edge case handling"""'''
        user_prompt = '''"""Design algorithm: {prompt}. Use optimal approach. Return function body only."""'''
    
    elif 'generategreedy' in tool_name:
        system_prompt = '''"""You are a greedy algorithm expert. Make optimal local choices.
CRITICAL RULES:
1. Use greedy approach for optimal solutions
2. Output ONLY function body code with 4-space indentation
3. Explain greedy choice in comments
4. No markdown fences, no explanations outside code
5. Handle edge cases"""'''
        user_prompt = '''"""Implement greedy: {prompt}. Make optimal local choices. Return function body."""'''
    
    # Build _call_llm function
    call_llm_function = f'''
def _call_llm(prompt):
    """调用LLM生成代码"""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise Exception("LLM_API_KEY is not set")
    base = (os.getenv("LLM_API_BASE") or "").rstrip("/")
    if not base:
        raise Exception("LLM_API_BASE is not set")
    url = base if base.endswith("/chat/completions") else f"{{base}}/chat/completions"
    
    system = {system_prompt}
    
    user = {user_prompt}
    
    payload = {{
        "model": "gpt-4o",
        "messages": [
            {{"role": "system", "content": system}},
            {{"role": "user", "content": user.format(prompt=prompt)}}
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }}
    
    headers = {{"Authorization": f"Bearer {{api_key}}", "Content-Type": "application/json"}}
    resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices", [{{}}])[0].get("message", {{}}).get("content", "")
    return _extract_code(text)
'''
    
    # Insert the function
    new_content = content[:insert_pos] + call_llm_function + content[insert_pos:]
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def main():
    """Fix all affected tool files"""
    if not TOOLS_DIR.exists():
        print(f"Error: {TOOLS_DIR} does not exist")
        return
    
    fixed_count = 0
    skipped_count = 0
    
    # Process all tool files
    for file_path in sorted(TOOLS_DIR.glob("code-generation-*.py")):
        if fix_tool_file(file_path):
            fixed_count += 1
            print(f"Fixed: {file_path.name}")
        else:
            skipped_count += 1
    
    print(f"\nSummary:")
    print(f"  Fixed: {fixed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {fixed_count + skipped_count}")

if __name__ == "__main__":
    main()
