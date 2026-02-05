"""修复缩进处理逻辑"""
import os
import glob
import re

def create_fixed_normalize():
    return '''def _normalize_indentation(code):
    """规范化代码缩进,确保所有行都有适当的4空格缩进"""
    if not code:
        return ""
    
    lines = code.split("\\n")
    if not lines:
        return ""
    
    # 移除完全空白的行(开头和结尾)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return ""
    
    # 找到最小缩进级别(非空行)
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # 只看非空行
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    # 如果最小缩进是0,说明有代码顶格写,需要统一添加4空格
    if min_indent == 0:
        result_lines = []
        for line in lines:
            if line.strip():  # 非空行直接加4空格前缀
                result_lines.append("    " + line)
            else:  # 空行保持空
                result_lines.append("")
        return "\\n".join(result_lines)
    
    # 如果已经有缩进,保持原样
    return code
'''

def fix_all_tools():
    tools_dir = r"c:\Users\zrz20\Desktop\vscode\Agent\Agent-router\repo\generated_tools"
    tool_files = glob.glob(os.path.join(tools_dir, "code-generation-*.py"))
    
    print(f"找到 {len(tool_files)} 个工具文件")
    
    fixed_func = create_fixed_normalize()
    
    for tool_file in tool_files:
        try:
            with open(tool_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换_normalize_indentation函数
            pattern = r'def _normalize_indentation\(code\):.*?(?=\ndef _call_llm\(prompt\):)'
            
            if re.search(pattern, content, re.DOTALL):
                new_content = re.sub(pattern, fixed_func + '\n', content, flags=re.DOTALL)
                
                with open(tool_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✓ {os.path.basename(tool_file)}")
            else:
                print(f"✗ {os.path.basename(tool_file)} - 未找到函数")
        
        except Exception as e:
            print(f"✗ {os.path.basename(tool_file)} - 错误: {e}")
    
    print("\n修复完成!")

if __name__ == "__main__":
    fix_all_tools()
