# 标签与现有工具对应关系分析

## 问题分析

查看您提供的训练标签和数据库中的实际角色，发现**存在不匹配**：

### 训练标签中使用的角色
```json
["planner", "builder", "tester", "refractor"]
```

### 数据库中实际存在的角色
```
✓ planner: 20 agents
✓ builder: 35 agents
✗ tester: 0 agents          ← 问题：不存在
✗ refractor: 0 agents       ← 问题：不存在（拼写可能错误？）
```

### 数据库中的所有角色
```
- algorithms
- builder               ✓ 匹配
- checker
- code-generation
- code-planner
- code-refactoring      ← 可能是 "refractor" 的正确名称？
- code-testing          ← 可能是 "tester" 的正确名称？
- data-structures
- dynamic-programming
- edge-cases
- graph-algorithms
- greedy
- io-operations
- mathematics
- parsing
- planner               ✓ 匹配
- recursion
- researcher
- robustness
- string-operations
```

---

## 核心问题

### 1. **角色名称不匹配**
训练标签使用的 `tester` 和 `refractor` 在数据库中不存在。

可能的对应关系：
- `tester` → `code-testing` (0 agents in DB)
- `refractor` → `code-refactoring` (应该有 agents，但拼写错误？)

### 2. **训练标签不合理**
查看示例标签：

```json
// 示例 1: strlen - 简单函数
{
  "topology": "single",
  "roles": ["builder"],      ← 合理
  "manager_role": null,
  "entry_role": "builder",
  "max_steps": 1
}

// 示例 2: encrypt - 字符串加密
{
  "topology": "single",
  "roles": ["builder"],      ← 合理
  "manager_role": null,
  "entry_role": "builder",
  "max_steps": 1
}

// 示例 4: add - 复杂逻辑（奇数索引的偶数元素求和）
{
  "topology": "centralized",
  "roles": ["planner", "builder", "tester"],  ← 使用了不存在的 tester
  "manager_role": "planner",
  "entry_role": "builder",   ← ⚠️ entry_role 应该是 planner？
  "max_steps": 5
}
```

---

## 具体问题详解

### 问题 1: `tester` 角色不存在
**影响**: 
- 训练时模型学习使用不存在的角色
- 推理时无法找到对应的 agent
- 导致路由失败

**解决方案**:
1. **方案 A**: 修改训练角色列表
   ```python
   # 修改 generate_labels.py 中的默认角色
   --roles "planner,builder,code-testing,code-refactoring"
   ```

2. **方案 B**: 创建 `tester` 和 `refractor` agents
   ```python
   # 添加这些角色的 agents 到数据库
   ```

3. **方案 C**: 使用数据库中实际存在的角色映射
   ```python
   # 在推理时映射角色名称
   tester -> code-testing
   refractor -> code-refactoring
   ```

### 问题 2: 拓扑结构可能不合理
示例 4 中的 `entry_role` 设置为 `builder`，但在 centralized 模式下通常应该是 `planner`（manager）。

### 问题 3: 角色粒度不一致
- 训练标签: 粗粒度 (`planner`, `builder`, `tester`)
- 数据库: 细粒度 (`code-planner`, `code-testing`, `code-refactoring`)

---

## 推荐解决方案

### 方案 1: 修改训练标签生成（推荐）

修改 `generate_labels.py` 使用数据库中实际存在的角色：

```python
python -m src.routing.generate_labels \
  --data data/Humaneval/humaneval-py.jsonl \
  --output data/router_labels_corrected.jsonl \
  --model gpt-4o \
  --roles "planner,builder,code-testing,code-refactoring,checker"
```

### 方案 2: 创建角色映射层

在路由系统中添加角色映射：

```python
ROLE_MAPPING = {
    "tester": "code-testing",
    "refractor": "code-refactoring",
    "planner": "planner",
    "builder": "builder"
}
```

### 方案 3: 统一角色命名

选择一种命名规范并统一：
- 要么全部用简短名称 (`planner`, `builder`, `tester`)
- 要么全部用带前缀的名称 (`code-planner`, `code-builder`, `code-tester`)

---

## 数据库中的角色统计

```
planner: 20 agents
builder: 35 agents
code-testing: ??? (需要查询)
code-refactoring: ??? (需要查询)
```

让我查询一下其他角色的数量...

---

## 立即行动建议

### 1. 重新生成标签（使用正确的角色）
```powershell
python -m src.routing.generate_labels `
  --data data/Humaneval/humaneval-py.jsonl `
  --output data/router_labels_v2.jsonl `
  --model gpt-4o `
  --roles "planner,builder,checker,code-generation,code-refactoring"
```

### 2. 验证数据库中的角色
```powershell
python check_roles_in_db.py
```

### 3. 调整标签标准化逻辑
修改 `generate_labels.py` 中的 `_normalize_topology_label` 函数，添加角色映射：

```python
def _normalize_role(value: Any) -> str:
    if value is None:
        return ""
    role = str(value).strip().lower()
    
    # 角色映射
    ROLE_ALIAS = {
        "tester": "checker",  # 或 "code-testing"
        "refractor": "code-refactoring",
        "test": "checker",
        "refactor": "code-refactoring"
    }
    
    return ROLE_ALIAS.get(role, role)
```

---

## 总结

**当前状态**: ❌ **不对应**
- 训练标签使用的 `tester` 和 `refractor` 在数据库中不存在
- 这会导致路由失败，无法找到对应的 agents

**建议**: 
1. ✅ 重新生成标签，使用数据库中实际存在的角色
2. ✅ 添加角色映射逻辑
3. ✅ 统一命名规范

**优先级**: 🔴 **高** - 这会直接影响系统功能
