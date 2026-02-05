# HumanEval 评估修复报告

## 问题分析

根据您提供的评估结果，发现以下核心问题：

### 1. **函数参数名错误**
- **症状**：生成的代码使用了通用变量名（如 `input_data`, `data`, `items`），而不是实际的参数名
- **例子**：`HumanEval_95_check_dict_case` 的参数是 `dict`，但生成代码使用了 `input_data`，导致 `NameError: name 'input_data' is not defined`

### 2. **上下文传递不完整**
- **症状**：`task_context` 只包含 `entry_point`，没有完整的函数签名信息
- **结果**：工具无法知道实际的参数名称，只能猜测或使用通用名

### 3. **大多数任务使用单一拓扑**
- **症状**：大部分任务被分配为 `single` topology，只使用一个 builder
- **可能原因**：任务描述不够完整，动态拓扑判断不够准确

## 修复方案

### 修复 1: 提取函数签名信息

**文件**: `src/evaluation/eval_humaneval.py`

新增函数 `_extract_function_signature()` 来从 HumanEval prompt 中提取完整的函数签名：

```python
def _extract_function_signature(prompt: str, entry_point: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract function signature (name and parameters) from prompt."""
    # 使用正则表达式提取函数名和参数列表
    # 支持类型注解 (如 string: str, values: List[Any])
    # 返回: {"function_name": "strlen", "parameters": ["string"]}
```

**关键特性**:
- 支持类型注解（`: str`, `: List[int]`）
- 支持返回类型注解（`-> int`）
- 支持多行函数签名
- 正确解析参数名，忽略类型和默认值

### 修复 2: 传递函数签名到 task_context

**文件**: `src/evaluation/eval_humaneval.py` - `_auto_generate_solutions_orchestrator()`

```python
# 提取函数签名
func_sig = _extract_function_signature(prompt, entry_point)

# 在 task_text 中明确指出参数名
param_hint = ""
if func_sig and func_sig.get("parameters"):
    params = func_sig["parameters"]
    param_hint = f"\n注意: 函数参数名是: {', '.join(params)}. 必须在代码中使用这些确切的参数名!\n"

# 构建 task_context，包含完整签名信息
task_context = {
    "entry_point": entry_point,
    "function_signature": func_sig  # 新增！
} if entry_point else None
```

### 修复 3: 更新代码生成工具

**文件**: 
- `generated_tools/code-generation-generateparsing31.py`
- `generated_tools/code-generation-generategreedy32.py`
- `generated_tools/code-generation-generaterecursion33.py`

**改进内容**:

1. **从 task_context 提取函数签名**:
```python
task_context = inputs.get("task_context", {})
func_sig_info = task_context.get("function_signature")
if func_sig_info:
    param_names = func_sig_info.get("parameters", [])
    func_signature = f"def {entry_point}({', '.join(param_names)}):"
```

2. **添加明确的参数名警告**:
```python
param_warning = ""
if param_names:
    param_warning = f"\n\nCRITICAL: The function parameters are: {', '.join(param_names)}. You MUST use these exact names in your code, NOT generic names like 'data', 'items', 'input_string', etc."
```

3. **在所有 LLM 调用中包含函数签名和警告**:
```python
prompt = (
    f"Function signature: {func_signature or 'unknown'}{param_warning}\n\n"
    f"Task: Generate a Python function body..."
)
```

### 修复 4: 已有的系统提示改进

工具的 `_call_llm()` 系统提示已经包含了相关警告：

```python
system = """You MUST follow these output rules exactly:
3. CRITICAL: Use ONLY the actual parameter names from the function signature 
   (e.g., if parameter is 'lst', use 'lst' NOT 'data' or 'items')
4. Do NOT use generic variable names like: data, items, input_string, inputs, arr 
   (unless they are the actual parameter names)
"""
```

但这还不够，需要在每个请求中明确提供参数名。

## 测试验证

创建了测试脚本 `test_signature_extraction.py` 来验证函数签名提取：

```
✅ HumanEval_95_check_dict_case: check_dict_case(dict)
✅ HumanEval_23_strlen: strlen(string)
✅ HumanEval_22_filter_integers: filter_integers(values)
✅ HumanEval_151_double_the_difference: double_the_difference(lst)
✅ All tests passed!
```

## 数据库同步

所有修改的工具已同步到数据库：
```bash
python sync_tools_to_db.py
# [Summary] Updated: 123, Not found in DB: 0
```

## 预期改进

修复后，工具将能够：

1. ✅ **正确使用参数名**: 从 task_context 获取参数名，而不是猜测
2. ✅ **减少 NameError**: 不再使用未定义的变量名
3. ✅ **提高代码质量**: 生成的代码更符合函数签名要求
4. ✅ **更好的上下文理解**: LLM 收到明确的参数名警告

## 下一步测试

建议重新运行 HumanEval 评估来验证修复效果：

```bash
python -m src.evaluation.eval_humaneval \
    --tasks data/humaneval_tasks.jsonl \
    --auto_generate \
    --use_orchestrator \
    --out data/humaneval_solutions_fixed.jsonl \
    --db demo_registry.sqlite \
    --index_dir index \
    # ... 其他参数
```

## 关于 Topology 问题

如果大多数任务仍然使用 `single` topology，可能需要：

1. 检查任务描述是否足够详细
2. 调整 `_plan_topology()` 的逻辑
3. 考虑为 HumanEval 类型的任务强制使用特定的 topology（如 `linear` 或 `centralized`）

但这应该是次要问题，主要问题（参数名错误）已经修复。
