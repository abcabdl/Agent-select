"""测试当前的缩进逻辑是否正确"""

def _normalize_indentation_current(code):
    """当前的实现"""
    if not code:
        return ""
    
    lines = code.split("\n")
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
        return "\n".join(result_lines)
    
    # 如果已经有缩进,保持原样
    return code


def _normalize_indentation_correct(code):
    """正确的实现"""
    if not code:
        return ""
    
    lines = code.split("\n")
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
            if line.strip():  # 非空行
                result_lines.append("    " + line)  # 直接添加4空格,保留原有结构
            else:
                result_lines.append("")  # 空行保持空
        return "\n".join(result_lines)
    
    # 如果已经有缩进,保持原样
    return code


# 测试用例1: 完全没有缩进的代码
test1 = """if not lst:
return 0
return sum(x**2 for x in lst)"""

print("测试1: 完全无缩进")
print("原始:")
print(repr(test1))
print("\n当前实现:")
print(repr(_normalize_indentation_current(test1)))
print("\n正确实现:")
print(repr(_normalize_indentation_correct(test1)))
print("\n" + "="*60 + "\n")

# 测试用例2: 有嵌套缩进的代码
test2 = """if not lst:
    return 0
return sum(x**2 for x in lst)"""

print("测试2: 有嵌套缩进")
print("原始:")
print(repr(test2))
print("\n当前实现:")
print(repr(_normalize_indentation_current(test2)))
print("\n正确实现:")
print(repr(_normalize_indentation_correct(test2)))
print("\n" + "="*60 + "\n")

# 测试用例3: 已经有正确4空格缩进
test3 = """    if not lst:
        return 0
    return sum(x**2 for x in lst)"""

print("测试3: 已有正确缩进")
print("原始:")
print(repr(test3))
print("\n当前实现:")
print(repr(_normalize_indentation_current(test3)))
print("\n正确实现:")
print(repr(_normalize_indentation_correct(test3)))
