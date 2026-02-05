# 已实现的改进 (Implemented Improvements)

## 日期: 2026-02-03

## 改进内容

### 1. ✅ 智能错误诊断提示词

**位置**: `orchestrator.py` lines 690-765

**功能**:
- 自动检测常见错误模式并给出针对性修复建议
- 特别针对 `isinstance(x, int)` 类型判断问题
- 包含 AssertionError, NameError, IndexError, TypeError 等常见错误的诊断

**示例诊断信息**:
```
❌ TYPE CHECK ERROR: Using isinstance(x, int) filters out floats like 5.0.
✅ FIX: Use 'x == int(x)' or 'isinstance(x, (int, float)) and x == int(x)' to accept mathematical integers.
```

### 2. ✅ 调试输出验证错误信息传递

**位置**: `orchestrator.py` lines 797-806

**功能**:
- 在每轮工具调用时输出调试信息
- 验证 `refinement_request` 是否被传递给工具
- 显示 `failed_code`, `test_error` 等关键信息
- 帮助诊断错误信息是否到达工具层

**调试输出示例**:
```
[DEBUG] Round 3: Passing refinement_request to tool code-generation-assemblesnippets
[DEBUG] Refinement request (first 300 chars):
[DEBUG] 🔍 PREVIOUS ATTEMPT FAILED - DETAILED ANALYSIS:
📋 Test Error: AssertionError: candidate([5.0, 4.0]) == 25
...
[DEBUG] Passing failed_code (length: 85)
[DEBUG] Passing test_error: AssertionError: candidate([5.0, 4.0]) == 25...
```

### 3. ✅ 强制工具类别多样性机制

**位置**: `orchestrator.py` lines 768-787

**功能**:
- 跟踪工具类别失败次数 (`tool_category_counts`)
- 检测连续使用同类别工具失败 (`consecutive_same_category_failures`)
- **失败3次后强制切换到不同类别的工具**
- 自动建议不同类别的工具列表

**工具类别分类**:
- `assemblesnippets` 类: assemblesnippets, assemblesnippets12, assemblesnippets15 等
- `generatemath` 类: generatemath6, generatemath18, generatemath30 等
- `generateedgecase` 类: generateedgecase12 等
- `generatefunctionbody` 类: generatefunctionbody 等

**切换逻辑**:
```python
if consecutive_same_category_failures >= 3:
    # 强制切换到不同类别的工具
    different_category_tools = [tid for tid in tool_ids 
                                if get_category(tid) != current_category]
    role_context["suggested_tools"] = different_category_tools
    role_context["force_different_category"] = True
```

**调试输出示例**:
```
[DEBUG] Tool category 'assemblesnippets' failures: 3, consecutive: 3
[DEBUG] Forcing tool category switch from 'assemblesnippets' after 3 failures
[DEBUG] Suggesting different category tools: ['code-generation-generateedgecase12', 'code-generation-generatefunctionbody']
```

## 改进效果预期

### 针对 HumanEval_151 (isinstance 错误)
- ✅ 详细诊断信息会明确指出类型判断问题
- ✅ 3次失败后会强制切换工具类别
- ⚠️ 需要工具本身读取并使用 `refinement_request`

### 针对 HumanEval_160 (ImportError)
- ✅ 3次 generatemath 工具失败后会切换到 assemblesnippets 等其他类别
- ✅ 避免重复尝试同类工具

### 针对 HumanEval_105 (参数名错误)
- ✅ 约束条件已在prompt中强调
- ⚠️ 仍可能需要角色路由改进

## 验证方法

### 1. 查看调试输出
运行测试时，stderr 会显示:
- 是否传递了 refinement_request
- 工具类别失败统计
- 何时触发强制切换

### 2. 检查工具选择
观察 tool_trace 中:
- 是否在3次失败后切换了工具类别
- 不同类别工具是否被调用

### 3. 对比测试结果
- 运行相同的20个测试
- 对比通过率变化
- 检查 HumanEval_151 是否有改善

## 下一步优化建议

### P0 - 工具实现检查
1. ✅ 确认 `assemblesnippets` 工具是否读取 `refinement_request`
2. ✅ 确认工具是否使用 `failed_code` 和 `error_message`
3. 如果工具忽略这些参数，需要修改工具实现

### P1 - LLM 提示词优化
4. 在 `plan_tool()` 中强调工具多样性
5. 添加 ImportError 特殊处理逻辑
6. 改进 failure_analysis 的提示

### P2 - 角色路由优化
7. 确保 builder 任务不路由到 planner
8. 改进 llm_fallback 的代码提取

## 监控指标

运行测试时关注:
1. **工具多样性**: 失败案例是否使用了3+不同类别的工具
2. **调试信息**: refinement_request 是否每次都传递
3. **通过率**: 整体通过率是否提升
4. **HumanEval_151**: 是否仍重复相同错误

## 已知限制

1. **工具必须支持**: 工具需要实际读取并使用 `refinement_request` 参数
2. **LLM 仍可能忽略**: 即使建议不同工具，LLM 仍可能选择已失败的工具
3. **诊断仅在 refinement_request 中**: 如果工具不读取此参数，诊断无效

## 测试命令

```bash
# 运行前20个测试（快速验证）
python eval_humaneval.py --max_tests 20 --output test_results_improved.json

# 查看调试输出
python eval_humaneval.py --max_tests 20 2> debug_output.log

# 对比结果
python analyze_20_results.py
```
