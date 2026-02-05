"""
Check which tools are missing from database.
"""
import sqlite3
from pathlib import Path

REPO_DIR = Path(__file__).parent
TOOLS_DIR = REPO_DIR / "generated_tools"
DB_PATH = REPO_DIR / "demo_registry.sqlite"

def main():
    # Get all tool files
    tool_files = sorted(TOOLS_DIR.glob("code-generation-*.py"))
    file_ids = {f.stem for f in tool_files}
    print(f"Tool files on disk: {len(file_ids)}")
    
    # Get all tools in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM tool_code")
    total_db = cursor.fetchone()[0]
    print(f"Total tools in database: {total_db}")
    
    cursor.execute("SELECT id FROM tool_code WHERE id LIKE 'code-generation-%'")
    db_ids = {row[0] for row in cursor.fetchall()}
    print(f"Code-generation tools in DB: {len(db_ids)}")
    
    # Find missing
    missing = file_ids - db_ids
    print(f"\nMissing from database: {len(missing)}")
    
    if missing:
        print("\nMissing tools (by pattern):")
        patterns = {}
        for tool_id in sorted(missing):
            # Extract pattern: code-generation-PATTERN123
            parts = tool_id.replace("code-generation-", "").rstrip("0123456789")
            patterns[parts] = patterns.get(parts, 0) + 1
        
        for pattern, count in sorted(patterns.items()):
            print(f"  {pattern}: {count} tools")
        
        print(f"\nFirst 10 missing:")
        for tool_id in sorted(missing)[:10]:
            print(f"  - {tool_id}")
    
    # Show what's in DB
    print(f"\nTools in database (by pattern):")
    db_patterns = {}
    for tool_id in sorted(db_ids):
        parts = tool_id.replace("code-generation-", "").rstrip("0123456789")
        db_patterns[parts] = db_patterns.get(parts, 0) + 1
    
    for pattern, count in sorted(db_patterns.items()):
        print(f"  {pattern}: {count} tools")
    
    conn.close()

if __name__ == "__main__":
    main()
