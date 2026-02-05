import sqlite3
conn = sqlite3.connect('demo_registry.sqlite')
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
print('\n'.join([row[0] for row in cursor.fetchall()]))
