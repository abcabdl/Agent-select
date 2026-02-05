#!/usr/bin/env python3
"""Check pass rate of first 20 tests in historical 161 test report."""

import json

# Read the historical report
with open('data/Humaneval/humaneval_report.json', 'r', encoding='utf-8') as f:
    historical_data = json.load(f)

print("=" * 80)
print("历史161个测试的前20个结果分析")
print("=" * 80)

# Get first 20 results
first_20 = historical_data['results'][:20]

passed_count = sum(1 for test in first_20 if test['ok'])
failed_count = 20 - passed_count

print(f"\n前20个测试:")
print(f"  总数: 20")
print(f"  通过: {passed_count}")
print(f"  失败: {failed_count}")
print(f"  通过率: {passed_count / 20 * 100:.2f}%")

print(f"\n历史161个测试总体:")
print(f"  总数: {historical_data['total']}")
print(f"  通过: {historical_data['passed']}")
print(f"  失败: {historical_data['total'] - historical_data['passed']}")
print(f"  通过率: {historical_data['pass_rate'] * 100:.2f}%")

print("\n前20个测试详情:")
print("-" * 80)
for i, test in enumerate(first_20, 1):
    status = "✅" if test['ok'] else "❌"
    print(f"{i:2d}. {status} {test['name']}")
    if not test['ok']:
        error = test.get('error', 'No error message')
        if len(error) > 60:
            error = error[:60] + "..."
        print(f"     错误: {error}")

print("\n" + "=" * 80)
print("对比分析")
print("=" * 80)

current_pass_rate = 85.0  # From the 20 new tests
historical_first_20_rate = passed_count / 20 * 100

print(f"历史前20个通过率: {historical_first_20_rate:.2f}%")
print(f"当前20个通过率: {current_pass_rate:.2f}%")
print(f"提升: {current_pass_rate - historical_first_20_rate:+.2f}%")

if current_pass_rate > historical_first_20_rate:
    print("\n✅ 改进后的通过率更高！")
elif current_pass_rate == historical_first_20_rate:
    print("\n⚠️  通过率持平")
else:
    print("\n❌ 通过率下降")
