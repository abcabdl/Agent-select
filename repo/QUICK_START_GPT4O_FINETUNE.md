# GPT-4o 微调快速开始指南

## 一、环境准备（一次性配置）

### 1.1 设置 API 密钥
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
$env:LLM_API_BASE = "https://az.gptplus5.com/v1"
```

### 1.2 安装训练依赖
```powershell
cd c:\Users\zrz20\Desktop\vscode\Agent\Agent-router\repo
pip install -r requirements-train.txt
```

### 1.3 验证环境（可选）
```powershell
python test_gpt4o_finetune_flow.py
```

---

## 二、完整微调流程

### 方案 A: 使用 GPT-4o 生成高质量标签（推荐）

#### 第 1 步: 生成训练标签
```powershell
python -m src.routing.generate_labels `
  --data data/Humaneval/humaneval-py.jsonl `
  --output data/router_labels_gpt4o.jsonl `
  --model gpt-4o `
  --roles "planner,builder,tester,refractor"
```

**预计时间**: 约 10-20 分钟（161 个样本）  
**费用**: 约 $0.50-1.00（取决于提示长度）

#### 第 2 步: LoRA 微调
```powershell
python -m src.routing.train_router_lora `
  --data data/router_labels_gpt4o.jsonl `
  --model Qwen/Qwen3-8B-Instruct `
  --output_dir models/router_lora_gpt4o `
  --epochs 3 `
  --batch_size 1 `
  --grad_accum 8 `
  --lr 1e-4 `
  --use_4bit `
  --bf16 `
  --gradient_checkpointing `
  --save_steps 50
```

**预计时间**: 约 30-60 分钟（取决于 GPU）  
**GPU 需求**: 12GB+ VRAM（使用 4-bit 量化）

---

### 方案 B: 使用启发式标签（快速测试）

直接微调，使用内置的启发式标签生成：

```powershell
python -m src.routing.train_router_lora `
  --data data/Humaneval/humaneval-py.jsonl `
  --model Qwen/Qwen3-8B-Instruct `
  --output_dir models/router_lora_heuristic `
  --convert_humaneval `
  --drop_no_output `
  --epochs 1 `
  --batch_size 1 `
  --use_4bit
```

**优点**: 快速，无需 API 调用  
**缺点**: 标签质量较低，可能影响模型性能

---

## 三、快速测试（小规模验证）

### 3.1 生成 10 个样本标签
```powershell
# 先创建测试数据
Get-Content data/Humaneval/humaneval-py.jsonl | Select-Object -First 10 > data/test_10.jsonl

# 生成标签
python -m src.routing.generate_labels `
  --data data/test_10.jsonl `
  --output data/test_labels.jsonl `
  --model gpt-4o
```

### 3.2 快速微调测试
```powershell
python -m src.routing.train_router_lora `
  --data data/test_labels.jsonl `
  --model Qwen/Qwen3-8B-Instruct `
  --output_dir models/router_test `
  --epochs 1 `
  --batch_size 1 `
  --max_samples 10 `
  --use_4bit
```

**预计时间**: 5-10 分钟  
**用途**: 验证流程是否正常工作

---

## 四、高级配置选项

### 4.1 标签生成选项

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--model` | 教师模型 | gpt-4o | gpt-4o（高质量）或 gpt-4o-mini（低成本） |
| `--roles` | 可用角色 | planner,builder,tester,refractor | 根据需求调整 |
| `--data` | 输入数据路径 | 必填 | - |
| `--output` | 输出路径 | humaneval_router_labels.jsonl | - |

### 4.2 微调训练选项

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--model` | 基座模型 | Qwen/Qwen3-8B-Instruct | Qwen/Qwen3-8B-Instruct 或 Qwen2.5-7B-Instruct |
| `--epochs` | 训练轮数 | 1 | 1-3（过多可能过拟合） |
| `--batch_size` | 批次大小 | 1 | 1（内存小）或 2-4（内存大） |
| `--grad_accum` | 梯度累积 | 8 | 8-16 |
| `--lr` | 学习率 | 1e-4 | 1e-4 或 2e-4 |
| `--use_4bit` | 4位量化 | False | True（节省内存） |
| `--bf16` | BF16精度 | False | True（A100/H100）|
| `--gradient_checkpointing` | 梯度检查点 | False | True（节省内存） |
| `--lora_r` | LoRA 秩 | 8 | 8-16 |
| `--lora_alpha` | LoRA alpha | 16 | 16-32 |

### 4.3 数据过滤选项

```powershell
# 只训练特定类型的样本
--sample_types "humaneval_teacher_distilled,custom_type"

# 删除空输出样本
--drop_no_output

# 限制样本数量（快速测试）
--max_samples 100

# 自动转换 HumanEval 格式
--convert_humaneval
```

---

## 五、常见问题排查

### 问题 1: API 连接失败
```
RuntimeError: LLM_API_KEY/OPENAI_API_KEY is not set
```

**解决**: 
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

### 问题 2: CUDA 内存不足
```
RuntimeError: CUDA out of memory
```

**解决**: 
```powershell
# 方案 1: 启用 4-bit 量化
--use_4bit

# 方案 2: 减小批次大小
--batch_size 1 --grad_accum 16

# 方案 3: 启用梯度检查点
--gradient_checkpointing
```

### 问题 3: 训练依赖缺失
```
ModuleNotFoundError: No module named 'peft'
```

**解决**:
```powershell
pip install datasets peft accelerate bitsandbytes
```

### 问题 4: 模型下载慢
```powershell
# 预下载模型
--download_model --model_dir models/hf_cache --hf_token YOUR_TOKEN
```

---

## 六、验证训练效果

### 6.1 检查输出文件

训练后会生成：
- `models/router_lora/adapter_config.json` - LoRA 配置
- `models/router_lora/adapter_model.bin` - LoRA 权重
- `models/router_lora/router_lora_meta.json` - 元数据
- `models/router_lora/tokenizer_config.json` - Tokenizer 配置

### 6.2 测试推理（Python）

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "models/router_lora")
tokenizer = AutoTokenizer.from_pretrained("models/router_lora")

# 测试
messages = [
    {"role": "system", "content": "You are a meta-router."},
    {"role": "user", "content": "Task: Sort a list of integers"}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

### 6.3 评估指标

- **训练 loss**: 应该逐步下降
- **输出格式**: 检查 JSON 格式是否正确
- **拓扑选择**: 验证选择是否合理

---

## 七、性能优化技巧

### 7.1 提升标签质量
1. 使用 GPT-4o 而非 GPT-4o-mini
2. 降低 temperature（0.1-0.2）
3. 添加更多示例到提示中

### 7.2 加速训练
1. 使用更大的 `--grad_accum`（有效 batch size = batch_size × grad_accum）
2. 启用 `--bf16`（A100/H100）
3. 减少 `--save_steps`（减少 I/O）

### 7.3 节省内存
1. `--use_4bit`（最大节省）
2. `--gradient_checkpointing`（中等节省）
3. 减小 `--max_length`（少量节省）
4. 减小 `--lora_r`（少量节省）

### 7.4 防止过拟合
1. 使用较小的 `--epochs`（1-3）
2. 添加 `--weight_decay`（0.01）
3. 使用更多训练数据
4. 使用数据增强

---

## 八、完整示例命令

### 示例 1: 标准流程（推荐）
```powershell
# 步骤 1: 生成标签
python -m src.routing.generate_labels `
  --data data/Humaneval/humaneval-py.jsonl `
  --output data/router_labels.jsonl `
  --model gpt-4o

# 步骤 2: 微调
python -m src.routing.train_router_lora `
  --data data/router_labels.jsonl `
  --model Qwen/Qwen3-8B-Instruct `
  --output_dir models/router_lora `
  --epochs 2 `
  --batch_size 2 `
  --grad_accum 8 `
  --lr 1e-4 `
  --use_4bit `
  --bf16 `
  --gradient_checkpointing
```

### 示例 2: 快速测试
```powershell
# 生成 10 个样本
Get-Content data/Humaneval/humaneval-py.jsonl | Select-Object -First 10 > data/test.jsonl

# 标签生成
python -m src.routing.generate_labels --data data/test.jsonl --output data/test_labels.jsonl --model gpt-4o

# 快速微调
python -m src.routing.train_router_lora --data data/test_labels.jsonl --model Qwen/Qwen3-8B-Instruct --output_dir models/test --epochs 1 --use_4bit
```

### 示例 3: 低内存配置（<12GB VRAM）
```powershell
python -m src.routing.train_router_lora `
  --data data/router_labels.jsonl `
  --model Qwen/Qwen3-8B-Instruct `
  --output_dir models/router_lora_lowmem `
  --batch_size 1 `
  --grad_accum 16 `
  --use_4bit `
  --gradient_checkpointing `
  --max_length 1024 `
  --lora_r 4
```

---

## 九、检查清单

微调前确认：
- [ ] API 密钥已设置
- [ ] 训练依赖已安装
- [ ] GPU 可用（或使用 CPU）
- [ ] 有足够的磁盘空间（~10GB 用于模型）
- [ ] 训练数据存在且格式正确

微调后验证：
- [ ] 输出目录包含 LoRA 权重
- [ ] 训练 loss 正常下降
- [ ] 模型可以加载
- [ ] 推理输出格式正确
- [ ] 拓扑选择合理

---

## 十、下一步

微调完成后，可以：
1. 在完整 HumanEval 数据集上评估性能
2. 集成到路由系统中
3. 在实际任务上测试效果
4. 迭代改进（更多数据、更好的提示、更大的模型）

---

**注意**: 首次运行会下载模型（~15GB），请确保网络连接稳定。
