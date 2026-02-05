# 工具代码自动同步到数据库

## 使用说明

### 方法1: 手动触发同步

修改工具代码后,运行:

```bash
python auto_sync_tools.py
```

或直接运行完整同步脚本(带详细日志):

```bash
python sync_tools_to_db.py
```

### 方法2: 在Python脚本中自动同步

```python
from auto_sync_tools import sync_now

# 修改工具代码...
# ... your tool modification code ...

# 自动同步到数据库
sync_now()
```

### 方法3: 使用装饰器

```python
from auto_sync_tools import auto_sync_after

@auto_sync_after
def modify_tools():
    # 批量修改工具代码
    pass

# 函数执行完后自动同步
modify_tools()
```

## 工作流程

1. **备份**: 每次同步前自动创建数据库备份 (demo_registry.bak.TIMESTAMP)
2. **同步**: 读取 `generated_tools/code-generation-*.py` 文件
3. **更新**: 更新 `tool_code` 表中对应工具的 code 字段
4. **提交**: 成功后提交,失败自动回滚并恢复备份

## 注意事项

- 只同步文件名与数据库id匹配的工具 (约60个)
- 数据库schema: `tool_code(id, code, updated_at)`
- 每次同步会更新 `updated_at` 时间戳
- 失败会自动恢复备份,数据安全有保障

## 当前修改

已完成的entry_point传递修改:

1. ✅ orchestrator.py - 添加task_context参数传递链
2. ✅ eval_humaneval.py - 构造并传递entry_point
3. ✅ 101个工具文件 - 添加entry_point提取代码
4. ✅ 60个工具 - 已同步到数据库

工具现在可以通过 `inputs.get("task_context", {}).get("entry_point")` 获取函数名,并在LLM prompt中使用正确的参数名。
