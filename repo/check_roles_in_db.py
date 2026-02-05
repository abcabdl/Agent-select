import sqlite3
import json

# Connect to database
conn = sqlite3.connect('demo_registry.sqlite')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("=" * 80)
print("Tables in database:")
print("=" * 80)
for table in tables:
    print(f"  - {table[0]}")

# Get schema of cards table
print("\n" + "=" * 80)
print("Schema of cards:")
print("=" * 80)
cursor.execute("PRAGMA table_info(cards)")
schema = cursor.fetchall()
for col in schema:
    print(f"  {col[1]} ({col[2]})")

# Get sample data
print("\n" + "=" * 80)
print("Sample cards (first 5):")
print("=" * 80)
cursor.execute("SELECT id, name, role_tags, description FROM cards LIMIT 5")
samples = cursor.fetchall()
for card_id, name, role_tags, desc in samples:
    desc_short = desc[:60] if desc else "No description"
    print(f"\n  ID: {card_id}")
    print(f"  Name: {name}")
    print(f"  Role Tags: {role_tags}")
    print(f"  Description: {desc_short}...")

# Get all unique role_tags (they might be JSON arrays)
print("\n" + "=" * 80)
print("Analyzing role_tags:")
print("=" * 80)
cursor.execute("SELECT DISTINCT role_tags FROM cards WHERE role_tags IS NOT NULL AND role_tags != ''")
role_tags_list = cursor.fetchall()
print(f"Found {len(role_tags_list)} unique role_tags values")

# Parse role_tags and extract individual roles
import json
all_roles = set()
for (role_tags_str,) in role_tags_list[:20]:  # Check first 20
    try:
        if role_tags_str:
            roles_array = json.loads(role_tags_str)
            if isinstance(roles_array, list):
                all_roles.update(roles_array)
    except:
        pass

print(f"\nExtracted individual roles:")
for role in sorted(all_roles):
    print(f"  - {role}")

# Check if router training roles exist
print("\n" + "=" * 80)
print("Checking for router training roles:")
print("=" * 80)
training_roles = ["planner", "builder", "tester", "refractor"]
for role in training_roles:
    # Check if role appears in any role_tags
    cursor.execute("SELECT COUNT(*) FROM cards WHERE role_tags LIKE ?", (f'%"{role}"%',))
    count = cursor.fetchone()[0]
    status = "✓" if count > 0 else "✗"
    print(f"  {status} {role}: {count} agents")
    
    # Show examples if found
    if count > 0 and count <= 3:
        cursor.execute("SELECT name FROM cards WHERE role_tags LIKE ? LIMIT 3", (f'%"{role}"%',))
        examples = cursor.fetchall()
        for (example_name,) in examples:
            print(f"      Example: {example_name}")

conn.close()
