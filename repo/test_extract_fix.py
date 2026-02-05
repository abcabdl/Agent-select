"""
Test the fixed _extract_code function
"""
import sys
import os
import re
import json

# Load the function directly
with open('generated_tools/code-generation-assemblesnippets.py', 'r', encoding='utf-8') as f:
    exec(f.read(), globals())

# Test case 1: JSON with "json" prefix (the problematic case)
test1 = '''json
{
  "code_or_commands": "    if string is None:\\n        return 0\\n    return len(string)"
}'''

print("Test 1 - JSON with 'json' prefix:")
result1 = _extract_code(test1)
print(repr(result1))
print()

# Test case 2: Plain JSON without prefix
test2 = '''{
  "code_or_commands": "    if not lst:\\n        return 0\\n    return sum(x**2 for x in lst)"
}'''

print("Test 2 - Plain JSON:")
result2 = _extract_code(test2)
print(repr(result2))
print()

# Test case 3: Markdown code block
test3 = '''```python
if not text:
    return ""
return text.strip()
```'''

print("Test 3 - Markdown code block:")
result3 = _extract_code(test3)
print(repr(result3))
print()

# Test case 4: List in JSON
test4 = '''{
  "code_or_commands": ["if values is None:", "    return []", "return [v for v in values if isinstance(v, int)]"]
}'''

print("Test 4 - List in JSON:")
result4 = _extract_code(test4)
print(repr(result4))
