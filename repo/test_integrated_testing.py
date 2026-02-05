"""
Quick test to verify the integrated testing logic works.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test the _test_code function
from execution.orchestrator import _test_code

# Test case 1: Valid code that should pass
task_context_1 = {
    "prompt": "def strlen(string: str) -> int:\n    \"\"\" Return length of given string\n    >>> strlen('')\n    0\n    >>> strlen('abc')\n    3\n    \"\"\"",
    "test": "def check(candidate):\n    assert candidate('') == 0\n    assert candidate('abc') == 3\n    assert candidate('hello') == 5",
    "entry_point": "strlen"
}

code_1 = "    return len(string)"

print("Test 1: Valid code")
print("=" * 60)
test_ok, test_error = _test_code(code_1, task_context_1)
print(f"Result: {'PASS' if test_ok else 'FAIL'}")
if not test_ok:
    print(f"Error: {test_error}")
print()

# Test case 2: Invalid code that should fail
code_2 = "    return 0"

print("Test 2: Invalid code (always returns 0)")
print("=" * 60)
test_ok, test_error = _test_code(code_2, task_context_1)
print(f"Result: {'PASS' if test_ok else 'FAIL'}")
if not test_ok:
    print(f"Error: {test_error[:200]}...")  # Truncate long error
print()

# Test case 3: No test context (should assume pass)
print("Test 3: No test context")
print("=" * 60)
test_ok, test_error = _test_code(code_1, None)
print(f"Result: {'PASS' if test_ok else 'FAIL'}")
print(f"(Expected: PASS - no test available, assume pass)")
print()

print("=" * 60)
print("Integration test complete!")
print()
print("If all 3 tests show expected results, the new logic is active.")
print("Expected: Test 1 PASS, Test 2 FAIL, Test 3 PASS")
