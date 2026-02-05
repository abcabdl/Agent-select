import sqlite3

conn = sqlite3.connect('demo_registry.bak.20260202T163937')
c = conn.cursor()
c.execute('SELECT code FROM tool_code WHERE id=?', ('code-generation-generatedatastructure50',))
result = c.fetchone()
if result:
    with open('original_tool.py', 'w', encoding='utf-8') as f:
        f.write(result[0])
    print("Original code saved to original_tool.py")
else:
    print("Tool not found in database")
conn.close()
