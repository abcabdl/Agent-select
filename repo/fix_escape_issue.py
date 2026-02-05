"""修复字符串转义问题"""
import os
import glob

def fix_string_escapes():
    tools_dir = r"c:\Users\zrz20\Desktop\vscode\Agent\Agent-router\repo\generated_tools"
    tool_files = glob.glob(os.path.join(tools_dir, "code-generation-*.py"))
    
    print(f"找到 {len(tool_files)} 个工具文件")
    
    fixed_count = 0
    for tool_file in tool_files:
        try:
            with open(tool_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否有问题
            if 'code.split("' in content and '\n")' in content:
                # 这表明有未正确转义的换行符
                # 但这个检测可能不准确,直接修复特定模式
                
                # 修复 split("\n") 中的实际换行
                original = content
                # 替换在_normalize_indentation和_local_format_fix中的split
                content = content.replace('code.split("\n', 'code.split("\\n')
                content = content.replace('"\n".join', '"\\n".join')
                
                if content != original:
                    with open(tool_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed_count += 1
                    print(f"✓ {os.path.basename(tool_file)}")
            
        except Exception as e:
            print(f"✗ {os.path.basename(tool_file)} - 错误: {e}")
    
    print(f"\n修复完成! 共修复 {fixed_count} 个文件")

if __name__ == "__main__":
    fix_string_escapes()
