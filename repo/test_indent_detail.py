"""详细测试缩进处理"""

# 当前的实现
def current_impl(code):
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return ""
    
    first_line = lines[0]
    first_indent = len(first_line) - len(first_line.lstrip())
    
    if first_indent == 0:
        result_lines = []
        for line in lines:
            if line.strip():
                content = line.lstrip()
                original_indent = len(line) - len(content)
                print(f"  处理: {repr(line)}")
                print(f"    content={repr(content)}, original_indent={original_indent}")
                new_line = "    " + " " * original_indent + content
                print(f"    结果: {repr(new_line)}")
                result_lines.append(new_line)
            else:
                result_lines.append("")
        return "\n".join(result_lines)
    
    return code

# 测试有嵌套的情况
test = """if not lst:
    return 0
return sum(x**2 for x in lst)"""

print("原始代码:")
print(test)
print("\n处理过程:")
result = current_impl(test)
print("\n最终结果:")
print(result)
print("\n结果的repr:")
print(repr(result))
