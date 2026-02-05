"""测试 orchestrator.py 中的代码提取逻辑修复"""

def test_code_extraction():
    """模拟 orchestrator.py 第 706-745 行的逻辑"""
    
    test_cases = [
        {
            "name": "成功 - 嵌套output",
            "tool_result": {
                "ok": True,
                "output": {
                    "output": {
                        "code": "if string is None:\n    return 0\nreturn len(string)",
                        "success": True
                    }
                }
            },
            "expected_success": True
        },
        {
            "name": "失败 - 超时错误",
            "tool_result": {
                "ok": True,
                "output": {
                    "output": {
                        "code": "# Error: The read operation timed out",
                        "success": False,
                        "error": "The read operation timed out"
                    }
                }
            },
            "expected_success": False
        },
        {
            "name": "成功 - 扁平结构",
            "tool_result": {
                "ok": True,
                "output": {
                    "code": "return x + y",
                    "success": True
                }
            },
            "expected_success": True
        },
        {
            "name": "失败 - success=False",
            "tool_result": {
                "ok": True,
                "output": {
                    "output": {
                        "code": "some code",
                        "success": False
                    }
                }
            },
            "expected_success": False
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test['name']}")
        tool_result = test["tool_result"]
        
        # 模拟 orchestrator.py 的提取逻辑
        successful_code = None
        has_errors = False
        
        if tool_result.get("ok") and "output" in tool_result:
            output = tool_result["output"]
            if isinstance(output, dict):
                inner_output = None
                success_flag = True
                
                if "output" in output and isinstance(output["output"], dict):
                    inner_output = output["output"]
                    code = inner_output.get("code_or_commands") or inner_output.get("code") or inner_output.get("solution")
                    if "success" in inner_output:
                        success_flag = bool(inner_output.get("success"))
                else:
                    code = output.get("code_or_commands") or output.get("code") or output.get("solution")
                    if "success" in output:
                        success_flag = bool(output.get("success"))
                
                if code and isinstance(code, str) and success_flag:
                    if code.strip().startswith("# Error:") or "timed out" in code.lower():
                        has_errors = True
                    else:
                        successful_code = code
                        has_errors = False
                else:
                    has_errors = True
        
        # 验证结果
        actual_success = (successful_code is not None and not has_errors)
        expected = test["expected_success"]
        
        if actual_success == expected:
            print(f"  ✅ PASS - successful_code={'设置' if successful_code else 'None'}, has_errors={has_errors}")
        else:
            print(f"  ❌ FAIL - 期望={expected}, 实际={actual_success}")
            print(f"     successful_code={successful_code}, has_errors={has_errors}")

if __name__ == "__main__":
    test_code_extraction()
    print("\n" + "="*60)
    print("所有测试完成！")
