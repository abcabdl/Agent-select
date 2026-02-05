#!/usr/bin/env python3
"""Analyze the 20 test results to check if the improved prompts are working."""

import json

results_json = """
{
  "total": 20,
  "passed": 17,
  "pass_rate": 0.85,
  "results": [
    {
      "name": "HumanEval_23_strlen",
      "ok": true
    },
    {
      "name": "HumanEval_89_encrypt",
      "ok": true
    },
    {
      "name": "HumanEval_95_check_dict_case",
      "ok": true
    },
    {
      "name": "HumanEval_85_add",
      "ok": true
    },
    {
      "name": "HumanEval_140_fix_spaces",
      "ok": true
    },
    {
      "name": "HumanEval_63_fibfib",
      "ok": true
    },
    {
      "name": "HumanEval_151_double_the_difference",
      "ok": false,
      "error": "AssertionError: candidate([5.0, 4.0]) == 25",
      "tool_trace": {
        "builder": [
          {"tool_id": "code-generation-assemblesnippets"},
          {"tool_id": "code-generation-assemblesnippets"},
          {"tool_id": "code-generation-assemblesnippets12"},
          {"tool_id": "code-generation-assemblesnippets"},
          {"tool_id": "code-generation-assemblesnippets"},
          {"tool_id": "code-generation-assemblesnippets"},
          {"tool_id": "code-generation-assemblesnippets"},
          {"tool_id": "code-generation-assemblesnippets"},
          {"tool_id": "code-generation-assemblesnippets"},
          {"tool_id": "code-generation-assemblesnippets"}
        ]
      }
    },
    {
      "name": "HumanEval_22_filter_integers",
      "ok": true
    },
    {
      "name": "HumanEval_41_car_race_collision",
      "ok": true
    },
    {
      "name": "HumanEval_17_parse_music",
      "ok": true
    },
    {
      "name": "HumanEval_79_decimal_to_binary",
      "ok": true
    },
    {
      "name": "HumanEval_14_all_prefixes",
      "ok": true
    },
    {
      "name": "HumanEval_53_add",
      "ok": true
    },
    {
      "name": "HumanEval_159_eat",
      "ok": true
    },
    {
      "name": "HumanEval_115_max_fill",
      "ok": true
    },
    {
      "name": "HumanEval_160_do_algebra",
      "ok": false,
      "error": "AssertionError: candidate(['+', '*', '-'], [2, 3, 4, 5]) == 9",
      "tool_trace": {
        "builder": [
          {"tool_id": "code-generation-generatemath6", "error": "Import blocked: decimal"},
          {"tool_id": "code-generation-generatemath18", "error": "Import blocked: decimal"},
          {"tool_id": "code-generation-generatemath30", "error": "Import blocked: decimal"},
          {"tool_id": "code-generation-generatemath42", "error": "Import blocked: decimal"},
          {"tool_id": "code-generation-generatemath54", "error": "Import blocked: decimal"}
        ]
      }
    },
    {
      "name": "HumanEval_27_flip_case",
      "ok": true
    },
    {
      "name": "HumanEval_105_by_length",
      "ok": false,
      "error": "NameError: name 'lst' is not defined. Did you mean: 'list'?",
      "tool_trace": {
        "planner": [
          {"tool_id": "code-planner-planrisk13"},
          {"tool_id": "code-planner-plantimebox14"},
          {"tool_id": "code-planner-planarchitecture15"}
        ]
      }
    },
    {
      "name": "HumanEval_25_factorize",
      "ok": true
    },
    {
      "name": "HumanEval_96_count_up_to",
      "ok": true
    }
  ]
}
"""

data = json.loads(results_json)

print("=" * 80)
print("20个测试结果分析")
print("=" * 80)
print(f"\n总测试数: {data['total']}")
print(f"通过数: {data['passed']}")
print(f"通过率: {data['pass_rate'] * 100:.1f}%")
print(f"失败数: {data['total'] - data['passed']}")

# 对比历史数据
print("\n" + "=" * 80)
print("与之前161个测试的对比")
print("=" * 80)
historical_pass_rate = 77.02
current_pass_rate = data['pass_rate'] * 100
improvement = current_pass_rate - historical_pass_rate

print(f"之前通过率 (161 tests): {historical_pass_rate:.2f}%")
print(f"当前通过率 (20 tests): {current_pass_rate:.2f}%")
print(f"变化: {improvement:+.2f}%")

if improvement > 0:
    print("\n✅ 通过率有提升！")
else:
    print("\n⚠️  通过率下降或持平")

# 分析失败案例
print("\n" + "=" * 80)
print("失败案例分析")
print("=" * 80)

failed_tests = [r for r in data['results'] if not r['ok']]

for i, test in enumerate(failed_tests, 1):
    print(f"\n{i}. {test['name']}")
    print(f"   错误: {test['error']}")
    
    if 'tool_trace' in test:
        tools = []
        if 'builder' in test['tool_trace']:
            tools = test['tool_trace']['builder']
        elif 'planner' in test['tool_trace']:
            tools = test['tool_trace']['planner']
        
        tool_ids = [t.get('tool_id', '') for t in tools if isinstance(t, dict)]
        unique_tools = set(tool_ids)
        
        print(f"   工具调用次数: {len(tool_ids)}")
        print(f"   使用的不同工具: {len(unique_tools)}")
        
        # Check if tool repeated without learning
        if len(tool_ids) > 5 and len(unique_tools) < 3:
            print(f"   ⚠️  工具重复使用问题: {len(tool_ids)}次调用但只用了{len(unique_tools)}个不同工具")

# 重点关注 HumanEval_151
print("\n" + "=" * 80)
print("HumanEval_151 分析 (isinstance 类型判断问题)")
print("=" * 80)

he151 = next((r for r in data['results'] if r['name'] == 'HumanEval_151_double_the_difference'), None)
if he151:
    print(f"状态: {'❌ 失败' if not he151['ok'] else '✅ 通过'}")
    print(f"错误: {he151.get('error', 'N/A')}")
    
    if 'tool_trace' in he151 and 'builder' in he151['tool_trace']:
        tools = he151['tool_trace']['builder']
        tool_names = [t.get('tool_id', '') for t in tools]
        print(f"工具调用: {len(tool_names)} 次")
        print(f"不同工具: {len(set(tool_names))} 个")
        
        if len(tool_names) == 10 and len(set(tool_names)) <= 2:
            print("\n⚠️  问题依旧: 10轮重试仅使用1-2个工具，生成相同错误代码")
            print("说明: 改进的提示词可能还未生效或需要进一步优化")
        else:
            print("\n✅ 工具多样性有改善")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

if current_pass_rate > historical_pass_rate + 5:
    print("✅ 明显改善: 通过率提升超过5%，提示词改进有效")
elif current_pass_rate > historical_pass_rate:
    print("✔️  轻微改善: 通过率有提升但不明显，可能需要更多测试验证")
elif abs(current_pass_rate - historical_pass_rate) < 3:
    print("⚠️  持平: 20个样本可能不足以判断，需要完整161个测试")
else:
    print("❌ 退化: 通过率下降，需要检查改进是否引入新问题")

print("\n建议:")
print("1. 运行完整161个测试以获得统计显著性")
print("2. 检查HumanEval_151是否仍然重复相同错误代码")
print("3. 分析失败案例的refinement_request是否包含了详细诊断信息")
