"""测试 _extract_code 函数处理数组格式的JSON"""
import sys
import os
import re
import json

# 加载修复后的 _extract_code 函数
with open('generated_tools/code-generation-assemblesnippets.py', 'r', encoding='utf-8') as f:
    exec(f.read(), globals())

# 测试用例：HumanEval_17_parse_music 的返回格式
test_input = '''json
{
  "code_or_commands": [
    "if not music_string:",
    "    return []",
    "beats = {'o': 4, 'o|': 2, '.|': 1}",
    "notes = music_string.split()",
    "return [beats[note] for note in notes]"
  ]
}'''

print("测试输入:")
print(test_input)
print("\n" + "="*70)
print("提取结果:")
result = _extract_code(test_input)
print(result)
print("\n" + "="*70)

# 验证结果
expected = """if not music_string:
    return []
beats = {'o': 4, 'o|': 2, '.|': 1}
notes = music_string.split()
return [beats[note] for note in notes]"""

if result == expected:
    print("✅ 测试通过！代码正确提取并格式化")
else:
    print("❌ 测试失败！")
    print("\n期望的输出:")
    print(repr(expected))
    print("\n实际的输出:")
    print(repr(result))
