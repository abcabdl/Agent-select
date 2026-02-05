"""Test the code extraction logic."""

# Simulate the structure from your test case
test_case_1 = {
    "tool_trace": {
        "builder": [
            {
                "tool_id": "code-generation-generategreedy56",
                "result": {
                    "ok": True,
                    "output": {
                        "code_or_commands": "if string is None:\n        return 0\n    length = 0\n    for _ in string:\n        length += 1\n    return length"
                    },
                    "error": None
                }
            },
            {
                "tool_id": "code-generation-generateparsing55",
                "result": {
                    "ok": True,
                    "output": {
                        "code_or_commands": "json\n{\n  \"code_or_commands\": \"    if string is None:\\n        return 0\\n    return len(string)\"\n}"
                    },
                    "error": None
                }
            },
            {
                "tool_id": "code-generation-generateparsing55",
                "result": {
                    "ok": True,
                    "output": {
                        "code_or_commands": "if string is None:\n        return 0\n    return len(string)"
                    },
                    "error": None
                }
            }
        ]
    }
}

# Case with return None
test_case_2 = {
    "tool_trace": {
        "builder": [
            {
                "tool_id": "code-generation-generategreedy44",
                "result": {
                    "ok": True,
                    "output": {
                        "code_or_commands": "    return None"
                    },
                    "error": None
                }
            },
            {
                "tool_id": "code-generation-generategreedy44",
                "result": {
                    "ok": True,
                    "output": {
                        "code_or_commands": "    return None"
                    },
                    "error": None
                }
            },
            {
                "tool_id": "code-generation-generategreedy44",
                "result": {
                    "ok": True,
                    "output": {
                        "code_or_commands": "    return None"
                    },
                    "error": None
                }
            }
        ]
    }
}

# Case with valid code then return None
test_case_3 = {
    "tool_trace": {
        "builder": [
            {
                "tool_id": "tool1",
                "result": {
                    "ok": True,
                    "output": {
                        "code_or_commands": "if n <= 0:\n        return []\n    return list(range(n))"
                    },
                    "error": None
                }
            },
            {
                "tool_id": "tool2",
                "result": {
                    "ok": True,
                    "output": {
                        "code_or_commands": "return None"
                    },
                    "error": None
                }
            },
            {
                "tool_id": "tool3",
                "result": {
                    "ok": True,
                    "output": {
                        "code_or_commands": "# some other code\nreturn 42"
                    },
                    "error": None
                }
            }
        ]
    }
}

print("Test cases prepared. Import eval_humaneval and test _extract_code_from_results.")
print("\nTest case 1: Should return the last valid code")
print("Test case 2: Should return None (all are fallbacks)")
print("Test case 3: Should return the first valid code (before 'return None')")
