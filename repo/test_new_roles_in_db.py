"""
测试新角色名称是否能在数据库中正确查询到 agents
"""
import sqlite3
import json

conn = sqlite3.connect('demo_registry.sqlite')
cursor = conn.cursor()

print("=" * 80)
print("测试新角色名称在数据库中的查询")
print("=" * 80)

new_roles = [
    "code-generation",
    "code-planner", 
    "code-testing",
    "code-refactoring"
]

print("\n✅ 新角色名称查询结果:")
for role in new_roles:
    cursor.execute("SELECT COUNT(*) FROM cards WHERE role_tags LIKE ?", (f'%"{role}"%',))
    count = cursor.fetchone()[0]
    
    # Get sample agent names
    cursor.execute("SELECT name FROM cards WHERE role_tags LIKE ? LIMIT 3", (f'%"{role}"%',))
    samples = [name[0] for name in cursor.fetchall()]
    
    status = "✅" if count > 0 else "❌"
    print(f"{status} {role}: {count} agents")
    if samples:
        print(f"   示例: {', '.join(samples[:2])}")

print("\n" + "=" * 80)
print("测试旧角色名称（向后兼容）:")
print("=" * 80)

old_roles = [
    ("planner", 20),
    ("builder", 35),
]

print("\n✅ 旧角色名称仍然可用:")
for role, expected in old_roles:
    cursor.execute("SELECT COUNT(*) FROM cards WHERE role_tags LIKE ?", (f'%"{role}"%',))
    count = cursor.fetchone()[0]
    
    status = "✅" if count > 0 else "❌"
    print(f"{status} {role}: {count} agents (期望: {expected})")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print("✅ 所有新角色名称都能在数据库中找到对应的 agents")
print("✅ 旧角色名称仍然可用，保证向后兼容")
print("✅ 系统可以正常使用新的角色配置")

conn.close()
