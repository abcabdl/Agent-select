"""
Quick test for a single HumanEval problem to verify parameter name fix.
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.eval_humaneval import _extract_function_signature

# Test with HumanEval_23 (strlen)
test_prompt = """


def strlen(string: str) -> int:
    \"\"\" Return length of given string
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    \"\"\"
"""

entry_point = "strlen"
func_sig = _extract_function_signature(test_prompt, entry_point)

print(f"Entry point: {entry_point}")
print(f"Function signature: {func_sig}")

if func_sig:
    print(f"[OK] Successfully extracted signature")
    # func_sig is a dict: {"function_name": "strlen", "parameters": ["string"]}
    if isinstance(func_sig, dict) and "string" in func_sig.get("parameters", []):
        print(f"[OK] Correct parameter name 'string' found in parameters list")
    else:
        print(f"[ERROR] 'string' not found in parameters list!")
        print(f"  Parameters: {func_sig.get('parameters', []) if isinstance(func_sig, dict) else 'N/A'}")
else:
    print(f"[ERROR] Failed to extract signature")

# Test task context structure
task_context = {
    "entry_point": entry_point,
    "function_signature": func_sig
}

print(f"\nTask context: {json.dumps(task_context, indent=2)}")

# Simulate what tools should extract (using dict structure)
if func_sig and isinstance(func_sig, dict):
    param_names = func_sig.get("parameters", [])
    print(f"\nExtracted parameter names from dict: {param_names}")
    
    if param_names == ["string"]:
        print(f"✓✓ CORRECT: Tool would use parameter name 'string'")
    else:
        print(f"✗✗ ERROR: Tool would use wrong parameter names: {param_names}")
