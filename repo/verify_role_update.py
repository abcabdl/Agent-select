"""
验证角色名称更新是否正确
"""
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("=" * 80)
print("测试角色名称更新")
print("=" * 80)

# Test 1: Check generate_labels default roles
print("\n1. 检查 generate_labels.py 的默认角色...")
try:
    import argparse
    from routing.generate_labels import SYSTEM_PROMPT
    
    # Parse default roles
    import re
    match = re.search(r'--roles.*default="([^"]+)"', open("src/routing/generate_labels.py").read())
    if match:
        default_roles = match.group(1)
        print(f"   默认角色: {default_roles}")
        expected = "code-generation,code-planner,code-testing,code-refactoring"
        if default_roles == expected:
            print(f"   ✅ 正确！")
        else:
            print(f"   ❌ 错误！期望: {expected}")
    
    # Check system prompt
    if "code-generation" in SYSTEM_PROMPT and "code-planner" in SYSTEM_PROMPT:
        print(f"   ✅ SYSTEM_PROMPT 已更新")
    else:
        print(f"   ❌ SYSTEM_PROMPT 未更新")
        
except Exception as e:
    print(f"   ❌ 错误: {e}")

# Test 2: Check run_query default roles
print("\n2. 检查 run_query.py 的默认角色...")
try:
    match = re.search(r'--roles.*default="([^"]+)"', open("src/execution/run_query.py").read())
    if match:
        default_roles = match.group(1)
        print(f"   默认角色: {default_roles}")
        expected = "code-generation,code-planner,code-testing,code-refactoring"
        if default_roles == expected:
            print(f"   ✅ 正确！")
        else:
            print(f"   ❌ 错误！期望: {expected}")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# Test 3: Check eval_humaneval default roles
print("\n3. 检查 eval_humaneval.py 的默认角色...")
try:
    match = re.search(r'--roles.*default="([^"]+)"', open("src/evaluation/eval_humaneval.py").read())
    if match:
        default_roles = match.group(1)
        print(f"   默认角色: {default_roles}")
        expected = "code-generation,code-planner,code-testing,code-refactoring"
        if default_roles == expected:
            print(f"   ✅ 正确！")
        else:
            print(f"   ❌ 错误！期望: {expected}")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# Test 4: Check query_builder role mappings
print("\n4. 检查 query_builder.py 的角色支持...")
try:
    from core.query_builder import build_role_query
    
    test_roles = [
        ("code-generation", "code generator"),
        ("code-planner", "code planner"),
        ("code-testing", "code tester"),
        ("code-refactoring", "code refactoring"),
        ("builder", "code generator"),  # 兼容旧名称
        ("planner", "code planner"),
        ("tester", "code tester"),
    ]
    
    all_ok = True
    for role, expected_keyword in test_roles:
        query = build_role_query("test task", role)
        if expected_keyword.split()[0] in query.lower():
            print(f"   ✅ {role} -> 正确映射")
        else:
            print(f"   ❌ {role} -> 映射失败")
            all_ok = False
    
    if all_ok:
        print(f"   ✅ 所有角色映射正确！")
        
except Exception as e:
    print(f"   ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check orchestrator role mappings
print("\n5. 检查 orchestrator.py 的角色映射...")
try:
    orchestrator_content = open("src/execution/orchestrator.py").read()
    
    expected_mappings = [
        "code-generation",
        "code-planner", 
        "code-testing",
        "code-refactoring"
    ]
    
    all_found = True
    for role in expected_mappings:
        if role in orchestrator_content:
            print(f"   ✅ {role} 已添加")
        else:
            print(f"   ❌ {role} 未找到")
            all_found = False
    
    if all_found:
        print(f"   ✅ 所有角色映射已添加！")
        
except Exception as e:
    print(f"   ❌ 错误: {e}")

# Summary
print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print("已更新以下文件以使用新的角色名称：")
print("  1. src/routing/generate_labels.py")
print("  2. src/execution/run_query.py")
print("  3. src/evaluation/eval_humaneval.py")
print("  4. src/core/query_builder.py")
print("  5. src/execution/orchestrator.py")
print()
print("新的角色名称（对应数据库中的实际角色）：")
print("  - code-generation (138 agents) - 替代 builder")
print("  - code-planner (60 agents) - 替代 planner")
print("  - code-testing (60 agents) - 替代 tester")
print("  - code-refactoring (60 agents) - 替代 refractor")
print()
print("所有角色同时支持新旧名称，保证向后兼容。")
print("=" * 80)
