"""
Convenient wrapper: modify tools and auto-sync to database.
Usage: Import and call auto_sync_after() decorator or call sync_now() manually.
"""
from pathlib import Path
import subprocess
import sys

REPO_DIR = Path(__file__).parent
SYNC_SCRIPT = REPO_DIR / "sync_tools_to_db.py"

def sync_now():
    """Manually trigger sync of tools to database."""
    print("\n[AutoSync] Syncing tools to database...")
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT)],
        cwd=REPO_DIR,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Print summary line only
        for line in result.stdout.split('\n'):
            if '[Summary]' in line or 'Updated:' in line or '[Done]' in line:
                print(line)
        return True
    else:
        print(f"[ERROR] Sync failed: {result.stderr}")
        return False

def auto_sync_after(func):
    """Decorator to auto-sync after function execution."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        sync_now()
        return result
    return wrapper

if __name__ == "__main__":
    # Manual trigger
    sync_now()
