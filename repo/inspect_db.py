#!/usr/bin/env python3
"""检查并更新数据库"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "demo_registry.sqlite"

def inspect_and_update():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 检查所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"数据库表: {tables}\n")
    
    # 检查cards表结构
    if 'cards' in tables:
        cursor.execute("PRAGMA table_info(cards)")
        columns = cursor.fetchall()
        print("cards 表结构:")
        for col in columns:
            print(f"  {col[1]}: {col[2]}")
        
        # 查看示例数据
        cursor.execute("SELECT * FROM cards LIMIT 3")
        rows = cursor.fetchall()
        print(f"\ncards 表示例数据 (前3行):")
        col_names = [desc[0] for desc in cursor.description]
        for row in rows:
            print(f"\n  记录:")
            for i, val in enumerate(row):
                val_str = str(val)[:100] if val else "NULL"
                print(f"    {col_names[i]}: {val_str}")
    
    # 检查tool_code表结构
    if 'tool_code' in tables:
        print("\n" + "="*60)
        cursor.execute("PRAGMA table_info(tool_code)")
        columns = cursor.fetchall()
        print("\ntool_code 表结构:")
        for col in columns:
            print(f"  {col[1]}: {col[2]}")
        
        # 查看示例数据
        cursor.execute("SELECT * FROM tool_code WHERE tool_id LIKE 'code-generation-%' LIMIT 2")
        rows = cursor.fetchall()
        print(f"\ntool_code 表示例数据 (代码生成工具,前2行):")
        col_names = [desc[0] for desc in cursor.description]
        for row in rows:
            print(f"\n  工具:")
            for i, val in enumerate(row):
                val_str = str(val)[:150] if val else "NULL"
                print(f"    {col_names[i]}: {val_str}")
    
    # 统计代码生成工具
    if 'tool_code' in tables:
        cursor.execute("SELECT COUNT(*) FROM tool_code WHERE tool_id LIKE 'code-generation-%'")
        count = cursor.fetchone()[0]
        print(f"\n代码生成工具总数: {count}")
    
    conn.close()

def main():
    if not DB_PATH.exists():
        print(f"错误: 数据库不存在: {DB_PATH}")
        sys.exit(1)
    
    print(f"检查数据库: {DB_PATH}\n")
    inspect_and_update()

if __name__ == "__main__":
    main()
