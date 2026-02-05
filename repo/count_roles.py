import sqlite3

conn = sqlite3.connect('demo_registry.sqlite')
cursor = conn.cursor()

roles = ['code-testing', 'code-refactoring', 'checker', 'code-generation', 'researcher', 'code-planner']

print("=" * 80)
print("Role counts in database:")
print("=" * 80)
for role in roles:
    cursor.execute("SELECT COUNT(*) FROM cards WHERE role_tags LIKE ?", (f'%"{role}"%',))
    count = cursor.fetchone()[0]
    print(f"  {role}: {count} agents")

print("\n" + "=" * 80)
print("Complete role statistics:")
print("=" * 80)

# Get all roles with counts
import json
cursor.execute("SELECT role_tags FROM cards WHERE role_tags IS NOT NULL")
all_role_tags = cursor.fetchall()

role_counts = {}
for (role_tags_str,) in all_role_tags:
    try:
        if role_tags_str:
            roles_array = json.loads(role_tags_str)
            if isinstance(roles_array, list):
                for role in roles_array:
                    role_counts[role] = role_counts.get(role, 0) + 1
    except:
        pass

for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {role}: {count}")

conn.close()
