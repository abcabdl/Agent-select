"""
批量更新所有code-generation工具，让它们使用refinement_request参数
"""
import os
import re

TOOLS_DIR = "generated_tools"

# 要修改的run函数新版本
NEW_RUN_FUNCTION = '''def run(inputs):
    """主函数：生成代码"""
    # 获取基础任务描述
    prompt = inputs.get("prompt", "") or inputs.get("query", "") or inputs.get("task", "")
    if not prompt:
        return {"output": {"code": "# Error: No prompt provided"}}
    
    # 检查是否有错误修复请求（重试场景）
    refinement_request = inputs.get("refinement_request", "")
    failed_code = inputs.get("failed_code", "")
    test_error = inputs.get("test_error", "")
    
    # 如果有refinement_request，说明这是重试，需要包含错误信息
    if refinement_request or failed_code:
        enhanced_prompt = prompt
        
        if refinement_request:
            # 使用详细的错误分析和修复提示
            enhanced_prompt += f"\\n\\n{refinement_request}"
        elif failed_code and test_error:
            # 如果只有failed_code但没有refinement_request，构建基本提示
            enhanced_prompt += (
                f"\\n\\n⚠️ PREVIOUS ATTEMPT FAILED:\\n"
                f"Failed Code:\\n{failed_code}\\n\\n"
                f"Error: {test_error}\\n\\n"
                f"Please analyze the error and generate CORRECTED code."
            )
        
        prompt = enhanced_prompt
    
    try:
        code = _call_llm(prompt)
        # 最终格式验证和修正
        code = _format_and_validate(code)
        return {
            "output": {
                "code": code,
                "success": bool(code and code.strip())
            }
        }
    except Exception as e:
        return {
            "output": {
                "code": f"# Error: {str(e)}",
                "success": False,
                "error": str(e)
            }
        }
'''

def update_tool_file(filepath):
    """更新单个工具文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找run函数的位置
    # 匹配 def run(inputs): 开始到下一个顶层def或文件结尾
    pattern = r'def run\(inputs\):.*?(?=\ndef \w+|$)'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"  ⚠️ 未找到run函数: {filepath}")
        return False
    
    # 替换run函数
    new_content = content[:match.start()] + NEW_RUN_FUNCTION.strip() + '\n' + content[match.end():]
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def main():
    """主函数"""
    updated_count = 0
    skipped_count = 0
    
    # 遍历所有code-generation工具文件
    for filename in os.listdir(TOOLS_DIR):
        if filename.startswith('code-generation-') and filename.endswith('.py'):
            filepath = os.path.join(TOOLS_DIR, filename)
            print(f"处理: {filename}")
            
            if update_tool_file(filepath):
                updated_count += 1
                print(f"  ✅ 已更新")
            else:
                skipped_count += 1
    
    print(f"\n完成！更新了 {updated_count} 个文件，跳过了 {skipped_count} 个文件")

if __name__ == "__main__":
    main()
