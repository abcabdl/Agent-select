"""
Add missing code-generation tools to database.
"""
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent
TOOLS_DIR = REPO_DIR / "generated_tools"
DB_PATH = REPO_DIR / "demo_registry.sqlite"

def backup_database():
    """Create backup before modification."""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = DB_PATH.parent / f"{DB_PATH.stem}.bak.{timestamp}"
    shutil.copy2(DB_PATH, backup_path)
    print(f"[Backup] Created: {backup_path.name}")
    return backup_path

def add_missing_tools():
    """Add tools that exist on disk but not in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get existing tools
    cursor.execute("SELECT id FROM tool_code WHERE id LIKE 'code-generation-%'")
    existing = {row[0] for row in cursor.fetchall()}
    
    # Get file tools
    file_tools = {f.stem: f for f in TOOLS_DIR.glob("code-generation-*.py")}
    
    # Find missing
    missing = set(file_tools.keys()) - existing
    
    print(f"\n[Analysis]")
    print(f"  Existing in DB: {len(existing)}")
    print(f"  Files on disk: {len(file_tools)}")
    print(f"  Missing in DB: {len(missing)}")
    
    if not missing:
        print("\n[Info] All tools already in database!")
        conn.close()
        return
    
    print(f"\n[Action] Adding {len(missing)} missing tools...")
    
    added = 0
    timestamp = datetime.now().isoformat()
    
    for tool_id in sorted(missing):
        tool_file = file_tools[tool_id]
        
        # Read tool code
        with open(tool_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Insert into tool_code table
        try:
            cursor.execute("""
                INSERT INTO tool_code (id, code, updated_at)
                VALUES (?, ?, ?)
            """, (tool_id, code, timestamp))
            print(f"  [+] {tool_id}")
            added += 1
        except sqlite3.IntegrityError:
            print(f"  [SKIP] {tool_id} (already exists)")
    
    conn.commit()
    conn.close()
    
    print(f"\n[Summary]")
    print(f"  Added: {added}")
    print(f"  Total now: {len(existing) + added}")

def main():
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        return
    
    backup_database()
    
    try:
        add_missing_tools()
        print(f"\n[Done] Database updated: {DB_PATH}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
