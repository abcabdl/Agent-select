#!/usr/bin/env python3
"""
Test the first 3 HumanEval problems with updated tools.
"""
import sys
from pathlib import Path
import json
import sqlite3

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from execution.orchestrator import run_workflow
from retrieval.embedder import get_embedder
from retrieval.index_service import FaissIndex
from registry.registry import ToolRegistry

# Test data - first 20 from HumanEval
test_problems = [
    {
        "name": "HumanEval_23_strlen",
        "prompt": '''def strlen(string: str) -> int:
    """ Return length of given string
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    """
''',
        "test": '''
def check(candidate):
    assert candidate('') == 0
    assert candidate('abc') == 3
check(strlen)
'''
    },
    {
        "name": "HumanEval_151_double_the_difference",
        "prompt": '''def double_the_difference(lst):
    """
    Given a list of numbers, return the sum of squares of the numbers
    in the list that are odd. Ignore numbers that are negative or not integers.
    
    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0  
   
    If the input list is empty, return 0.
    """
''',
        "test": '''
def check(candidate):
    assert candidate([1, 3, 2, 0]) == 10
    assert candidate([-1, -2, 0]) == 0
    assert candidate([9, -2]) == 81
    assert candidate([0]) == 0
    assert candidate([5, 4]) == 25
    assert candidate([0.1, 0.2, 0.3]) == 0
    assert candidate([-10, -20, -30]) == 0
    assert candidate([-1, -2, 8]) == 0
    assert candidate([0.2, 3, 5]) == 34
    
    import random
    random.seed(42)
    assert candidate([5.0, 4.0]) == 25
    assert candidate([0.1, 0.2, 0.3, 0.4, 0.5]) == 0
    assert candidate([]) == 0
check(double_the_difference)
'''
    },
    {
        "name": "HumanEval_160_do_algebra",
        "prompt": '''def do_algebra(operator, operand):
    """
    Given two lists operator, and operand. The first list has basic algebra operations, and 
    the second list is a list of integers. Use the two given lists to build the algebric 
    expression and return the evaluation of this expression.

    The basic algebra operations:
    Addition ( + ) 
    Subtraction ( - ) 
    Multiplication ( * ) 
    Floor division ( // ) 
    Exponentiation ( ** ) 

    Example:
    operator['+', '*', '-']
    operand = [2, 3, 4, 5]
    result = 2 + 3 * 4 - 5
    => result = 9

    Note:
        The length of operator list is equal to the length of operand list minus one.
        Operand is a list of of non-negative integers.
        Operator list has at least one operator, and operand list has at least two operands.

    """
''',
        "test": '''
def check(candidate):
    assert candidate(['**', '*', '+'], [2, 3, 4, 5]) == 37
    assert candidate(['+', '*', '-'], [2, 3, 4, 5]) == 9
    assert candidate(['//', '*'], [7, 3, 4]) == 8
check(do_algebra)
'''
    }
]

def run_single_test(registry, index, embedder, problem):
    """Run a single HumanEval problem."""
    print(f"\n{'='*80}")
    print(f"Testing: {problem['name']}")
    print(f"{'='*80}")
    
    # Get the function name from prompt
    func_name = problem['prompt'].split('(')[0].replace('def ', '').strip()
    
    # Create task description
    task = f"Generate Python code:\n\n{problem['prompt']}\n\nComplete this function with correct implementation."
    
    try:
        # Run workflow
        result = run_workflow(
            task_text=task,
            roles=["builder"],
            registry=registry,
            index=index,
            embedder=embedder,
            max_attempts=10,
            tool_only=True,
            tool_timeout_s=5.0,
        )
        
        # Extract code from result
        generated_code = None
        if "tool_results" in result:
            for tool_result in result.get("tool_results", []):
                output = tool_result.get("output", {})
                if isinstance(output, dict):
                    code = output.get("code") or (output.get("output", {}) or {}).get("code")
                    if code and "def " in code:
                        generated_code = code
                        break
        
        if not generated_code:
            print(f"❌ FAILED - No valid code generated")
            return {"name": problem["name"], "ok": False, "reason": "no_code"}
        
        # Test the generated code
        test_code = f"{generated_code}\n\n{problem['test']}"
        
        try:
            exec(test_code, {})
            print(f"✅ PASSED")
            return {"name": problem["name"], "ok": True}
        except Exception as e:
            print(f"❌ FAILED - Test error: {type(e).__name__}: {str(e)[:100]}")
            return {"name": problem["name"], "ok": False, "reason": str(e)[:200]}
            
    except Exception as e:
        print(f"❌ FAILED - Workflow error: {type(e).__name__}: {str(e)[:100]}")
        return {"name": problem["name"], "ok": False, "reason": str(e)[:200]}


def main():
    print("Initializing registry, index, and embedder...")
    
    # Initialize components
    db_path = "demo_registry.sqlite"
    index_dir = "index"
    dim = 64
    
    registry = ToolRegistry(db_path)
    
    embedder = get_embedder(
        embedder_type="sentence-transformer",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
        normalize=False
    )
    
    index = FaissIndex(dim=dim, index_dir=index_dir)
    index.load()
    
    print(f"\nRunning {len(test_problems)} HumanEval tests...")
    
    results = []
    for i, problem in enumerate(test_problems, 1):
        print(f"\n[{i}/{len(test_problems)}]")
        result = run_single_test(registry, index, embedder, problem)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    pass_rate = passed / total if total > 0 else 0
    
    print(f"Passed: {passed}/{total} ({pass_rate:.1%})")
    print("\nFailed tests:")
    for r in results:
        if not r["ok"]:
            reason = r.get("reason", "unknown")
            print(f"  - {r['name']}: {reason[:100]}")
    
    # Save results
    output = {
        "total": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "results": results
    }
    
    output_file = "test_20_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
