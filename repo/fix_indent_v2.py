"""
修复_extract_code函数,添加缩进规范化逻辑
"""
import os
import glob
import re

def fix_all_tools():
    """修复所有工具文件"""
    tools_dir = r"c:\Users\zrz20\Desktop\vscode\Agent\Agent-router\repo\generated_tools"
    tool_files = glob.glob(os.path.join(tools_dir, "code-generation-*.py"))
    
    print(f"找到 {len(tool_files)} 个工具文件")
    
    for tool_file in tool_files:
        try:
            with open(tool_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找旧的_extract_code函数并替换
            # 查找从def _extract_code开始到def _call_llm之前的内容
            pattern = r'(def _extract_code\(text\):.*?)(def _call_llm\(prompt\):)'
            
            match = re.search(pattern, content, re.DOTALL)
            if match:
                # 生成新的代码
                new_extract = '''def _extract_code(text):
    """提取代码块中的代码,支持markdown和JSON格式,并规范化缩进"""
    if not text:
        return ""
    
    # 1. 尝试提取markdown代码块
    code_block_match = re.search(r"```(?:python|py|json)?\\s*(.*?)```", text, re.DOTALL)
    if code_block_match:
        extracted = code_block_match.group(1).strip()
        # 如果markdown中包含JSON,继续解析
        if extracted.startswith("{") or extracted.startswith("["):
            try:
                data = json.loads(extracted)
                if isinstance(data, dict):
                    for key in ["code_or_commands", "code", "commands", "result"]:
                        if key in data:
                            code = data[key]
                            if isinstance(code, list):
                                return _normalize_indentation("\\n".join(code))
                            return _normalize_indentation(str(code).strip())
            except (json.JSONDecodeError, ValueError):
                pass
        return _normalize_indentation(extracted)
    
    # 2. 尝试解析JSON格式 {"code_or_commands": "..."}
    try:
        # 移除可能的"json"前缀和换行
        clean_text = text.strip()
        # 移除开头的 "json" 标记(可能后面跟换行或空格)
        if clean_text.lower().startswith("json"):
            clean_text = re.sub(r'^json\\s*', '', clean_text, flags=re.IGNORECASE)
        
        data = json.loads(clean_text)
        if isinstance(data, dict):
            # 尝试常见的键名
            for key in ["code_or_commands", "code", "commands", "result"]:
                if key in data:
                    code = data[key]
                    # 如果是列表,join成字符串
                    if isinstance(code, list):
                        return _normalize_indentation("\\n".join(code))
                    return _normalize_indentation(str(code).strip())
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 3. 直接返回文本
    return _normalize_indentation(str(text).strip())

def _normalize_indentation(code):
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
    
    # 找到第一行的缩进级别
    first_line = lines[0]
    first_indent = len(first_line) - len(first_line.lstrip())
    
    # 如果第一行没有缩进,给所有非空行添加4空格缩进
    if first_indent == 0:
        result_lines = []
        for line in lines:
            if line.strip():  # 非空行
                # 保留行内的相对缩进
                content = line.lstrip()
                original_indent = len(line) - len(content)
                # 添加基础4空格 + 原有缩进
                result_lines.append("    " + " " * original_indent + content)
            else:
                result_lines.append("")
        return "\\n".join(result_lines)
    
    # 如果已经有缩进,保持原样
    return code

'''
                
                new_content = content[:match.start(1)] + new_extract + match.group(2) + content[match.end(2):]
                
                with open(tool_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✓ {os.path.basename(tool_file)}")
            else:
                print(f"✗ {os.path.basename(tool_file)} - 未找到匹配模式")
        
        except Exception as e:
            print(f"✗ {os.path.basename(tool_file)} - 错误: {e}")
    
    print("\n修复完成!")

if __name__ == "__main__":
    fix_all_tools()
