"""验证新缩进逻辑"""

def new_normalize(code):
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return ""
    
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    if min_indent == 0:
        result_lines = []
        for line in lines:
            if line.strip():
                result_lines.append("    " + line)
            else:
                result_lines.append("")
        return "\n".join(result_lines)
    
    return code

# 测试1: 完全无缩进
test1 = """if not lst:
return 0"""
print("测试1: 完全无缩进")
print("输入:", repr(test1))
print("输出:", repr(new_normalize(test1)))
print("预期: '    if not lst:\\n    return 0'")
print()

# 测试2: 有嵌套缩进
test2 = """if not lst:
    return 0
return sum(...)"""
print("测试2: 有嵌套缩进")
print("输入:", repr(test2))
result2 = new_normalize(test2)
print("输出:", repr(result2))
print("可视化:")
print(result2)
print()

# 测试3: 已有正确缩进
test3 = """    if not lst:
        return 0
    return sum(...)"""
print("测试3: 已有正确缩进")
print("输入:", repr(test3))
print("输出:", repr(new_normalize(test3)))
print("预期: 不变")
