"""
Automatically sync modified tool files to demo_registry.sqlite.
Run this after batch modifying tool files to update database.
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

def sync_tool_to_database(tool_file: Path, conn: sqlite3.Connection):
    """Sync single tool file to database."""
    tool_id = tool_file.stem  # e.g., "code-generation-assemblesnippets3"
    
    # Read tool code
    with open(tool_file, 'r', encoding='utf-8') as f:
        tool_code = f.read()
    
    # Update tool_code table (schema: id, code, updated_at)
    cursor = conn.cursor()
    
    # Check if tool exists in database
    cursor.execute("SELECT id FROM tool_code WHERE id = ?", (tool_id,))
    exists = cursor.fetchone()
    
    if exists:
        # Update existing tool
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            UPDATE tool_code 
            SET code = ?, updated_at = ?
            WHERE id = ?
        """, (tool_code, timestamp, tool_id))
        return "updated"
    else:
        # Tool doesn't exist in database - skip or warn
        return "not_found"

def main():
    """Sync all modified tools to database."""
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        return
    
    if not TOOLS_DIR.exists():
        print(f"[ERROR] Tools directory not found: {TOOLS_DIR}")
        return
    
    # Backup database
    backup_path = backup_database()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    try:
        tool_files = sorted(TOOLS_DIR.glob("code-generation-*.py"))
        print(f"\n[Sync] Syncing {len(tool_files)} tool files to database...")
        
        updated = 0
        not_found = 0
        
        for tool_file in tool_files:
            result = sync_tool_to_database(tool_file, conn)
            
            if result == "updated":
                print(f"  [OK] {tool_file.name}")
                updated += 1
            elif result == "not_found":
                print(f"  [SKIP] {tool_file.name} (not in database)")
                not_found += 1
        
        # Commit changes
        conn.commit()
        
        print(f"\n[Summary]")
        print(f"   Updated: {updated}")
        print(f"   Not found in DB: {not_found}")
        print(f"   Total files: {len(tool_files)}")
        print(f"\n[Done] Database updated: {DB_PATH}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        conn.rollback()
        print(f"   Restoring backup...")
        shutil.copy2(backup_path, DB_PATH)
        print(f"   [OK] Database restored from backup")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
