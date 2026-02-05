"""
Test function signature extraction from HumanEval prompts
"""

import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluation.eval_humaneval import _extract_function_signature

# Test cases
test_cases = [
    {
        "name": "HumanEval_95_check_dict_case",
        "prompt": '''
def check_dict_case(dict):
    """
    Given a dictionary, return True if all keys are strings in lower 
    case or all keys are strings in upper case, else return False.
    """
''',
        "entry_point": "check_dict_case",
        "expected_params": ["dict"]
    },
    {
        "name": "HumanEval_23_strlen",
        "prompt": '''
def strlen(string: str) -> int:
    """ Return length of given string
    """
''',
        "entry_point": "strlen",
        "expected_params": ["string"]
    },
    {
        "name": "HumanEval_22_filter_integers",
        "prompt": '''
def filter_integers(values: List[Any]) -> List[int]:
    """ Filter given list of any python values only for integers
    """
''',
        "entry_point": "filter_integers",
        "expected_params": ["values"]
    },
    {
        "name": "HumanEval_151_double_the_difference",
        "prompt": '''
def double_the_difference(lst):
    """
    Given a list of numbers, return the sum of squares of the numbers
    in the list that are odd. Ignore numbers that are negative or not integers.
    """
''',
        "entry_point": "double_the_difference",
        "expected_params": ["lst"]
    }
]

print("Testing function signature extraction...\n")
all_passed = True

for test in test_cases:
    result = _extract_function_signature(test["prompt"], test["entry_point"])
    
    if result is None:
        print(f"❌ {test['name']}: Failed to extract signature")
        all_passed = False
        continue
    
    params = result.get("parameters", [])
    expected = test["expected_params"]
    
    if params == expected:
        print(f"✅ {test['name']}: {result['function_name']}({', '.join(params)})")
    else:
        print(f"❌ {test['name']}: Expected {expected}, got {params}")
        all_passed = False

print("\n" + ("="*50))
if all_passed:
    print("✅ All tests passed!")
else:
    print("❌ Some tests failed!")
