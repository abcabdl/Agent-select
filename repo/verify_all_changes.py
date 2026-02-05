#!/usr/bin/env python3
"""验证所有优化修改是否成功应用"""
import sqlite3
from pathlib import Path
import re

REPO_DIR = Path(__file__).parent
TOOLS_DIR = REPO_DIR / "generated_tools"
DB_PATH = REPO_DIR / "demo_registry.sqlite"
EVAL_FILE = REPO_DIR / "src" / "evaluation" / "eval_humaneval.py"

def check_tools():
    """检查工具文件修改"""
    print("1️⃣  检查工具文件修改")
    print("-" * 60)
    
    files = list(TOOLS_DIR.glob("code-generation-*.py"))
    enhanced_count = 0
    
    for filepath in files[:10]:  # 检查前10个
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'parameter names' in content.lower() and 'edge case' in content.lower():
            enhanced_count += 1
    
    print(f"✓ 已检查 10 个工具文件")
    print(f"✓ {enhanced_count}/10 包含增强约束")
    
    if enhanced_count >= 9:
        print("✅ 工具文件修改成功\n")
        return True
    else:
        print("⚠️  部分工具文件可能未正确修改\n")
        return False

def check_eval_code():
    """检查评估代码修改"""
    print("2️⃣  检查评估代码修改")
    print("-" * 60)
    
    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "_build_prompt 包含约束": "必须使用函数签名中的实际参数名" in content,
        "task_text 包含要求": "不要使用data/items/input_string等通用名" in content,
        "_build_prompt 处理边界情况": "必须处理边界情况" in content,
        "task_text 禁止模板": "不要生成无关的解析/处理模板" in content,
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    all_passed = all(checks.values())
    if all_passed:
        print("✅ 评估代码修改成功\n")
    else:
        print("⚠️  部分评估代码修改可能缺失\n")
    
    return all_passed

def check_database():
    """检查数据库更新"""
    print("3️⃣  检查数据库更新")
    print("-" * 60)
    
    if not DB_PATH.exists():
        print("✗ 数据库文件不存在\n")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 检查更新数量
    cursor.execute("""
        SELECT COUNT(*) FROM cards 
        WHERE id LIKE 'code-generation-%'
    """)
    total = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM cards 
        WHERE id LIKE 'code-generation-%' 
        AND description LIKE '%2026-02-01 增强约束更新%'
    """)
    updated = cursor.fetchone()[0]
    
    # 检查备份
    backups = list(REPO_DIR.glob("demo_registry.sqlite.bak.*"))
    latest_backup = max(backups, key=lambda p: p.name) if backups else None
    
    conn.close()
    
    print(f"✓ 总工具数: {total}")
    print(f"✓ 已更新: {updated}/{total}")
    if latest_backup:
        print(f"✓ 最新备份: {latest_backup.name}")
    
    if updated == total:
        print("✅ 数据库更新成功\n")
        return True
    else:
        print(f"⚠️  只有 {updated}/{total} 个工具已更新\n")
        return False

def main():
    print("="*60)
    print("HumanEval 优化修改验证")
    print("="*60 + "\n")
    
    results = {
        "工具文件": check_tools(),
        "评估代码": check_eval_code(),
        "数据库": check_database(),
    }
    
    print("="*60)
    print("总结")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅" if passed else "⚠️"
        print(f"{status} {name}: {'成功' if passed else '部分完成'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有修改已成功应用!")
        print("\n下一步:")
        print("1. 重新运行HumanEval评估")
        print("2. 对比result1.json和result2.json")
        print("3. 分析通过率提升情况")
    else:
        print("\n⚠️  部分修改未完全应用,请检查上述警告")

if __name__ == "__main__":
    main()
