#!/usr/bin/env python3
"""更新数据库中代码生成工具的描述"""
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "demo_registry.sqlite"

def update_database():
    # 备份数据库
    import shutil
    backup_path = f"{DB_PATH}.bak.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    shutil.copy(DB_PATH, backup_path)
    print(f"✓ 数据库已备份到: {backup_path}\n")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 获取所有代码生成工具
    cursor.execute("""
        SELECT id, name, description 
        FROM cards 
        WHERE id LIKE 'code-generation-%'
        ORDER BY id
    """)
    tools = cursor.fetchall()
    
    if not tools:
        print("警告: 没有找到代码生成工具")
        conn.close()
        return False
    
    print(f"找到 {len(tools)} 个代码生成工具\n")
    
    # 增强约束说明
    enhanced_note = """

【2026-02-01 增强约束更新】
此工具已增强以下约束:
• 必须使用函数签名中的实际参数名(如参数是lst就用lst,禁用data/items/input_string等通用名)
• 必须处理边界情况: 空输入([], '', None), 单元素, 边界值(0, 1)  
• 只生成直接解决问题的函数体代码,禁止生成无关的解析/处理模板
• 确保算法逻辑正确,能通过所有测试用例
• System prompt已更新为8条严格规则"""
    
    # 检查哪些需要更新
    needs_update = []
    already_updated = []
    
    for tool_id, name, desc in tools:
        if desc and "2026-02-01 增强约束更新" in desc:
            already_updated.append(tool_id)
        else:
            needs_update.append((tool_id, name, desc))
    
    print(f"已更新: {len(already_updated)} 个")
    print(f"需更新: {len(needs_update)} 个\n")
    
    if not needs_update:
        print("✓ 所有工具描述已是最新!")
        conn.close()
        return True
    
    # 显示前5个需要更新的
    print("需要更新的工具 (前5个):")
    for tool_id, name, desc in needs_update[:5]:
        print(f"  • {tool_id}: {name}")
    print()
    
    # 更新描述
    updated_count = 0
    for tool_id, name, old_desc in needs_update:
        new_desc = (old_desc or "") + enhanced_note
        cursor.execute("""
            UPDATE cards 
            SET description = ?,
                updated_at = ?
            WHERE id = ?
        """, (new_desc, datetime.now().isoformat(), tool_id))
        updated_count += 1
    
    conn.commit()
    print(f"✓ 已更新 {updated_count} 个工具的描述")
    
    # 验证更新
    cursor.execute("""
        SELECT COUNT(*) FROM cards 
        WHERE id LIKE 'code-generation-%' 
        AND description LIKE '%2026-02-01 增强约束更新%'
    """)
    verified = cursor.fetchone()[0]
    print(f"✓ 验证: {verified}/{len(tools)} 个工具包含增强约束说明")
    
    # 显示更新后的示例
    cursor.execute("""
        SELECT id, name, description 
        FROM cards 
        WHERE id LIKE 'code-generation-generate%' 
        LIMIT 1
    """)
    sample = cursor.fetchone()
    if sample:
        print(f"\n更新后的示例:")
        print(f"  工具ID: {sample[0]}")
        print(f"  名称: {sample[1]}")
        print(f"  描述: {sample[2][-300:]}")  # 显示最后300字符
    
    conn.close()
    return True

def main():
    if not DB_PATH.exists():
        print(f"错误: 数据库不存在: {DB_PATH}")
        sys.exit(1)
    
    print(f"正在更新数据库: {DB_PATH}\n")
    print("="*60 + "\n")
    
    success = update_database()
    
    print("\n" + "="*60)
    if success:
        print("\n✅ 数据库更新完成!")
        print("\n下一步: 重新运行HumanEval评估以验证改进效果")
    else:
        print("\n❌ 数据库更新失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
