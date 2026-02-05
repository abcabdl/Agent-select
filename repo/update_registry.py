#!/usr/bin/env python3
"""更新数据库中的工具定义以反映新的约束"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "demo_registry.sqlite"

def update_registry():
    """重新扫描所有工具文件并更新数据库"""
    # 备份数据库
    import shutil
    from datetime import datetime
    backup_path = f"{DB_PATH}.bak.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    shutil.copy(DB_PATH, backup_path)
    print(f"✓ 数据库已备份到: {backup_path}")
    
    # 连接数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 检查表结构
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n数据库表: {tables}")
    
    if 'agents' not in tables:
        print("错误: 数据库中没有 agents 表")
        conn.close()
        return False
    
    # 获取当前代码生成工具数量
    cursor.execute("SELECT COUNT(*) FROM agents WHERE tool_id LIKE 'code-generation-%'")
    count = cursor.fetchone()[0]
    print(f"\n当前代码生成工具数量: {count}")
    
    # 获取工具列表
    cursor.execute("""
        SELECT tool_id, name, description, path 
        FROM agents 
        WHERE tool_id LIKE 'code-generation-%'
        ORDER BY tool_id
        LIMIT 5
    """)
    sample_tools = cursor.fetchall()
    print(f"\n示例工具 (前5个):")
    for tool_id, name, desc, path in sample_tools:
        print(f"  - {tool_id}: {name}")
        print(f"    路径: {path}")
        print(f"    描述: {desc[:100]}..." if desc and len(desc) > 100 else f"    描述: {desc}")
    
    # 更新描述以反映新的约束
    enhanced_suffix = """

增强约束 (2026-02-01更新):
- 必须使用函数签名中的实际参数名,不使用data/items/input_string等通用名
- 必须处理边界情况: 空输入([], '', None), 单元素, 边界值(0, 1)
- 只生成直接解决问题的代码,不生成无关的解析/处理模板
- 确保算法逻辑正确"""
    
    # 检查是否需要更新
    cursor.execute("""
        SELECT COUNT(*) FROM agents 
        WHERE tool_id LIKE 'code-generation-%' 
        AND (description IS NULL OR description NOT LIKE '%增强约束%')
    """)
    needs_update = cursor.fetchone()[0]
    
    if needs_update == 0:
        print(f"\n✓ 所有工具描述已是最新,无需更新")
        conn.close()
        return True
    
    print(f"\n需要更新 {needs_update} 个工具的描述...")
    
    # 更新描述
    cursor.execute("""
        UPDATE agents 
        SET description = COALESCE(description, '') || ?
        WHERE tool_id LIKE 'code-generation-%' 
        AND (description IS NULL OR description NOT LIKE '%增强约束%')
    """, (enhanced_suffix,))
    
    updated_count = cursor.rowcount
    conn.commit()
    
    print(f"✓ 已更新 {updated_count} 个工具的描述")
    
    # 验证更新
    cursor.execute("""
        SELECT COUNT(*) FROM agents 
        WHERE tool_id LIKE 'code-generation-%' 
        AND description LIKE '%增强约束%'
    """)
    verified = cursor.fetchone()[0]
    print(f"✓ 验证: {verified} 个工具包含增强约束说明")
    
    conn.close()
    return True

def main():
    if not DB_PATH.exists():
        print(f"错误: 数据库不存在: {DB_PATH}")
        sys.exit(1)
    
    print(f"正在更新数据库: {DB_PATH}\n")
    success = update_registry()
    
    if success:
        print("\n✅ 数据库更新完成!")
    else:
        print("\n❌ 数据库更新失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
