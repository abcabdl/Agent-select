# 工具修复总结 - 使用错误提示进行重试

## 2026-02-03

## 问题诊断

### 发现的核心问题

**orchestrator.py 已经正确传递了错误信息，但工具完全没有使用！**

1. ✅ **orchestrator.py 正常工作**：
   - 第761行：构建详细的 `refinement_request`（包含错误分析、修复提示）
   - 第693-695行：传递 `failed_code`, `error_message`, `test_error`
   - 第820行：通过 `tool_executor.run_tool(tool_id, exec_inputs)` 传递给工具

2. ❌ **工具实现有致命缺陷**：
   ```python
   def run(inputs):
       prompt = inputs.get("prompt", "") or inputs.get("query", "") or inputs.get("task", "")
       # 完全没有读取 refinement_request, failed_code, test_error !!!
       code = _call_llm(prompt)
   ```

### 测试结果证据

**HumanEval_151_double_the_difference - 10轮重试全部失败**：
```
[DEBUG] Round 1: Passing refinement_request to tool code-generation-assemblesnippets
[DEBUG] Refinement request (first 300 chars):
[DEBUG] 🔍 PREVIOUS ATTEMPT FAILED - DETAILED ANALYSIS:
      ❌ TYPE CHECK ERROR: Using isinstance(x, int) filters out floats like 5.0...
[DEBUG] Passing failed_code (length: 109)
[DEBUG] Passing test_error: ...
```

但是工具每次都生成**完全相同的错误代码**：
```python
isinstance(x, int)  # 仍然过滤掉 5.0
```

**原因**：工具收到了 `refinement_request`，但 `run` 函数完全忽略了这个参数！

## 实施的修复

### 修改内容

更新了所有 123 个 `code-generation-*` 工具的 `run` 函数：

**修改前**：
```python
def run(inputs):
    prompt = inputs.get("prompt", "") or inputs.get("query", "") or inputs.get("task", "")
    if not prompt:
        return {"output": {"code": "# Error: No prompt provided"}}
    
    try:
        code = _call_llm(prompt)
        # ...
```

**修改后**：
```python
def run(inputs):
    """主函数：生成代码"""
    # 获取基础任务描述
    prompt = inputs.get("prompt", "") or inputs.get("query", "") or inputs.get("task", "")
    if not prompt:
        return {"output": {"code": "# Error: No prompt provided"}}
    
    # 🔥 新增：检查是否有错误修复请求（重试场景）
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
        # ...
```

### 更新的工具列表

**批量更新了 123 个工具文件**，包括：
- assemblesnippets 系列（21个）
- generatemath 系列（5个）
- generateedgecase 系列（5个）
- generatefunctionbody 系列（19个）
- generategreedy 系列（5个）
- generatealgorithm, generatedatastructure, generatedp 系列
- generaterecursion, generateparsing, generatestring 系列
- 等等...

## 预期效果

### HumanEval_151 修复后的执行流程

**Round 0**:
- prompt: "实现 double_the_difference 函数..."
- 工具生成错误代码: `isinstance(x, int)` 过滤掉 5.0
- 测试失败

**Round 1** (🔥 **现在会使用错误信息！**):
- prompt: "实现 double_the_difference 函数..."
- **+ refinement_request**: 
  ```
  🔍 PREVIOUS ATTEMPT FAILED - DETAILED ANALYSIS:
  
  📋 Test Error: AssertionError: candidate([5.0, 4.0]) == 25
  
  💡 SPECIFIC ISSUES IDENTIFIED:
  ❌ TYPE CHECK ERROR: Using isinstance(x, int) filters out floats like 5.0
  ✅ CRITICAL FIX: Replace isinstance(x, int) with: isinstance(x, (int, float)) and x == int(x)
  
  ❌ Failed Code:
  isinstance(x, int) and x > 0 and x % 2 != 0
  
  ⚠️ CRITICAL: Do NOT repeat the same logic error.
  ```
- **工具现在会看到完整的错误分析和修复建议**
- 应该生成修复后的代码: `isinstance(x, (int, float)) and x == int(x)`

### HumanEval_160 修复后的执行流程

**Round 0-2**: generatemath 工具尝试导入 decimal，ImportError

**Round 3** (🔥 **工具多样性机制触发 + 错误信息传递**):
- 强制切换到 assemblesnippets 类别
- prompt 包含:
  ```
  Previous 3 attempts all failed with ImportError: Import blocked: decimal
  Do NOT use decimal module!
  ```
- assemblesnippets 工具不会尝试导入 decimal
- 应该生成不依赖 decimal 的代码

## 验证方法

### 1. 快速验证工具是否读取参数

运行单个测试并检查 debug 输出：
```bash
python -m src.evaluation.eval_humaneval --tasks data/humaneval/humaneval-py.jsonl --out test.json --use_orchestrator --max_tests 1
```

查看日志中是否有：
```
[DEBUG] Round 1: Passing refinement_request to tool...
[DEBUG] Refinement request: 🔍 PREVIOUS ATTEMPT FAILED...
```

### 2. 验证工具生成不同的代码

查看 tool_trace，检查同一个工具在不同round是否生成了不同的代码（说明使用了错误信息）

### 3. 完整测试

```bash
python -m src.evaluation.eval_humaneval --tasks data/humaneval/humaneval-py.jsonl --out improved_results.json --use_orchestrator --max_tests 20
```

**预期改进**：
- HumanEval_151: 应该在Round 1-2就修复isinstance错误
- HumanEval_160: 切换到assemblesnippets后应该能避免ImportError
- 整体通过率: 从80%提升到85-90%

## 技术细节

### 错误信息传递链路

1. **orchestrator.py 第928行**: 测试失败 → `failed_code = code`, `last_test_error = test_error`
2. **orchestrator.py 第693-761行**: 构建 `refinement_request`（包含智能诊断提示）
3. **orchestrator.py 第809行**: `exec_inputs.update(tool_input)` 保留所有参数
4. **orchestrator.py 第820行**: `tool_executor.run_tool(tool_id, exec_inputs)` 传递给工具
5. **🔥 工具 run 函数（修复后）**: 读取 `refinement_request` 并追加到 prompt
6. **工具 _call_llm**: 完整的 prompt（包含错误信息）发送给 LLM

### 智能诊断提示示例

orchestrator.py 会根据错误模式生成针对性提示：

**isinstance 错误**:
```
❌ TYPE CHECK ERROR: Using isinstance(x, int) filters out floats like 5.0
✅ CRITICAL FIX: Replace isinstance(x, int) with: isinstance(x, (int, float)) and x == int(x)
```

**AssertionError**:
```
❌ ASSERTION FAILED: The test expectation was not met.
✅ HINT: Check the logic carefully - the output doesn't match expected result.
```

**NameError**:
```
❌ NAME ERROR: Variable not defined.
✅ FIX: Check variable names and ensure they're defined before use.
```

## 下一步

1. ✅ 运行20个测试验证修复效果
2. ✅ 观察 HumanEval_151 和 HumanEval_160 是否修复
3. ✅ 如果通过率提升，运行完整161个测试
4. 如果仍有问题，可能需要：
   - 调整 LLM prompt 的格式（让错误提示更明显）
   - 增加更多错误模式的智能诊断
   - 优化工具多样性触发条件

## 文件清单

**修改的文件**:
- `orchestrator.py` - 已有错误传递逻辑（无需修改）
- `generated_tools/code-generation-*.py` - 123个工具（批量更新）

**新增文件**:
- `batch_update_tool_refinement.py` - 批量更新脚本
- `TOOL_REFINEMENT_FIX.md` - 本文档
