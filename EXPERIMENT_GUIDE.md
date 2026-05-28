# 古诗字符级 GPT 实验完整指南

## 目录
1. [快速开始](#快速开始)
2. [必做部分](#必做部分)
3. [加分项](#加分项)
4. [报告整理](#报告整理)

---

## 快速开始

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

# 生成数据（如果未生成）
python prepare_data.py
```

### 快速验证（确保model.py补全正确）
```bash
# 尝试一个小规模训练（只用CPU跑1个epoch快速验证）
python train.py --epochs 1 --batch_size 4 --max_train_batches 10 --log_interval 2
```

---

## 必做部分

### 1. 补全 `model.py`（已完成 ✓）

三处需要补全的代码：

#### 1.1 `CausalSelfAttention.forward` （第33行）
- **实现多头因果自注意力**
- 包括：Q/K/V投影、多头拆分、缩放点积、因果掩码、softmax、dropout、合并头、输出投影
- 关键点：使用上三角掩码禁止看到未来位置

#### 1.2 `FeedForward.__init__` （第64行）
- **注册前馈网络层**
- 结构：d_model → d_ff（两倍扩展） → GELU → d_model
- 必须使用 `nn.Sequential` 或在 `__init__` 中注册各层

#### 1.3 `FeedForward.forward` （第72行）
- **简单调用已注册的网络**
- 输入输出形状保持 (B, T, d_model)

### 2. 完整训练流程

```bash
# 标准训练（推荐配置）
python train.py --epochs 3 --batch_size 32 --log_interval 50

# 或自定义参数
python train.py \
  --epochs 5 \
  --d_model 256 \
  --n_layer 4 \
  --lr 3e-4 \
  --batch_size 32 \
  --train_num_samples 50000

# 参数说明：
# --epochs: 训练轮数
# --d_model: 嵌入维度（默认256）
# --n_layer: Transformer层数（默认4）
# --lr: 学习率（默认3e-4）
# --batch_size: 批大小（默认32）
# --train_num_samples: 每轮训练样本数（默认50000，0=无限制）
```

**预期结果：**
- 生成 `ckpt_best.pt`：最优模型权重
- 生成 `loss_curve.png`：Loss曲线图
- 终端输出：每轮的 train_loss 和 val_loss

### 3. 生成诗文

```bash
# 交互式生成
python predict.py --temperature 0.9

# 一次性生成（非交互）
python predict.py --once --prompt "春" --temperature 0.9 --max_new 200
```

---

## 加分项

### 加分项1：Temperature 对比实验 ⭐⭐

目标：在不同温度下观察生成文本的差异。

#### 运行方式

```bash
python temperature_experiment.py \
  --temperatures 0.4 0.7 1.0 1.3 \
  --prompt "春" \
  --max_new 100 \
  --num_samples 3
```

#### 参数说明
- `--temperatures`：要测试的温度列表（默认 0.4 0.7 1.0 1.3）
- `--prompt`：起笔字（默认"春"）
- `--max_new`：新增长度（默认100）
- `--num_samples`：每个温度下的生成样本数（默认3）
- `--output_txt`：结果输出文本文件（默认 temperature_results.txt）
- `--output_csv`：结果输出CSV文件（默认 temperature_results.csv）

#### 预期输出

1. **文本文件** (`temperature_results.txt`)：
   - 每个温度下的生成文本
   - 便于复制到报告中

2. **CSV文件** (`temperature_results.csv`)：
   - 结构化数据，便于分析

#### 报告建议

在报告中包含：

| Temperature | 样本1 | 样本2 | 样本3 |
|---|---|---|---|
| 0.4 | [文本] | [文本] | [文本] |
| 0.7 | [文本] | [文本] | [文本] |
| 1.0 | [文本] | [文本] | [文本] |
| 1.3 | [文本] | [文本] | [文本] |

**观察（2-4句）：**
- 温度较低（0.4）：通常更**连贯和重复性**（如"春...春..."）
- 温度中等（0.7-1.0）：**平衡**的多样性和连贯性
- 温度较高（1.3）：更**随机和多样**（有时可能不太符合逻辑）

---

### 加分项2：超参对比实验 ⭐⭐⭐

目标：在不同超参配置下进行训练，观察性能差异。

#### 运行方式

```bash
python hyperparameter_experiment.py
```

> **重要**：此脚本会进行多次训练，耗时较长（建议在GPU上运行）。
> 如果运行缓慢，可在脚本中降低 `epochs` 或 `train_num_samples`。

#### 脚本内预置的4个配置

| 配置 | 特点 | d_model | n_layer | lr | dropout |
|------|------|---------|---------|-----|---------|
| 配置1-基础 | 基准配置 | 256 | 4 | 3e-4 | 0.1 |
| 配置2-较大模型 | 增加模型容量 | 384 | 6 | 3e-4 | 0.1 |
| 配置3-高学习率 | 提高学习速度 | 256 | 4 | 5e-4 | 0.1 |
| 配置4-高dropout | 增加正则化 | 256 | 4 | 3e-4 | 0.2 |

> **如需自定义配置**，编辑 `hyperparameter_experiment.py` 中的 `configurations` 字典。

#### 预期输出

1. **文本文件** (`hyperparameter_results.txt`)：
   - 汇总表：各配置的Loss对比
   - 详细配置：每个配置的超参设置
   - 分析：最优配置和性能差异

2. **CSV文件** (`hyperparameter_results.csv`)：
   - 结构化数据

#### 报告建议

在报告中包含：

**表：超参对比实验结果**

| 配置 | 最终验证Loss | 参数量 | 主要区别 |
|------|--------|--------|---------|
| 配置1-基础 | 2.8543 | 4.2M | 基准 |
| 配置2-较大模型 | 2.6821 | 8.9M | 增加d_model和n_layer |
| 配置3-高学习率 | 2.8102 | 4.2M | 提高学习率 |
| 配置4-高dropout | 2.9234 | 4.2M | 增加正则化 |

**分析（1-2句）：**
- 配置2因模型容量更大，验证Loss最低，但参数量翻倍
- 配置4虽增加了正则化，但没有改善效果，说明原始dropout足够

---

## 报告整理

### 报告应包含的部分

#### 1. 数据说明
- 使用的数据：古诗数据集，约10万首诗
- 词表大小：8319个字符
- 训练/验证集划分

#### 2. 模型说明
- Transformer 架构（Decoder-only）
- 使用的超参（d_model, n_head, n_layer等）
- 参数量统计

#### 3. 必做实验结果
- **Loss 曲线图**（截图 `loss_curve.png`）
- **最终验证 Loss**
- **生成示例**（至少1条，标注temperature和起笔字）

#### 4. 加分项1：Temperature 对比

包含：
- 上述表格
- 每个温度各1条代表生成文本
- 观察（2-4句）

#### 5. 加分项2：超参对比

包含：
- 上述表格
- 各配置的详细参数
- 性能分析（1-2句）

#### 6. 理论问题简答（2-4句/题）

```
Q1. 因果掩码（Causal Mask）的作用是什么？
A: [你的答案]

Q2. 为什么要在注意力中除以 sqrt(d_k)？
A: [你的答案]

Q3. FFN 在 Transformer 块中起什么作用？
A: [你的答案]
```

---

## 常见问题排查

### Q：model.py 补全后运行报错"张量形状不匹配"
**A：** 检查以下几点：
- CausalSelfAttention.forward 中，q/k/v 的 view 和 transpose 是否正确
- 最后的合并（y.transpose(...).view(...)）是否恢复为 (B, T, C)
- FeedForward 中线性层的维度是否正确

### Q：温度实验生成缓慢
**A：** 可以：
- 降低 `--max_new`（新增长度）
- 降低 `--num_samples`（样本数）
- 使用GPU（确保cuda可用）

### Q：超参实验出现 CUDA OOM 错误
**A：** 编辑 `hyperparameter_experiment.py` 中的 `configurations`，降低：
- `batch_size`（降至16或8）
- `d_model` 和 `d_ff`（使用较小的维度）
- `epochs`（只跑1个epoch快速测试）

### Q：生成的诗文质量不好
**A：** 这是正常的，原因可能是：
- 模型太小（尝试加大 d_model 或 n_layer）
- 训练不足（增加 epochs）
- Temperature 设置不当（尝试 0.7-0.8 范围）

---

## 文件清单

最终应有以下文件：

```
transformer/
├── model.py                      ✓ (已补全)
├── train.py                      (原文件，已可用)
├── predict.py                    (原文件，已可用)
├── temperature_experiment.py     ✓ (新增)
├── hyperparameter_experiment.py  ✓ (新增)
├── EXPERIMENT_GUIDE.md           ✓ (本文件)
│
├── ckpt_best.pt                  (训练生成)
├── loss_curve.png                (训练生成)
├── temperature_results.txt       (加分1生成)
├── temperature_results.csv       (加分1生成)
├── hyperparameter_results.txt    (加分2生成)
├── hyperparameter_results.csv    (加分2生成)
│
└── [其他原文件...]
```

---

## 典型完整实验流程

```bash
# 第1步：快速验证补全
python train.py --epochs 1 --batch_size 4 --max_train_batches 10

# 第2步：完整训练
python train.py --epochs 3 --batch_size 32

# 第3步：生成示例诗文
python predict.py --once --prompt "春" --temperature 0.9 --max_new 150

# 第4步：Temperature 对比实验
python temperature_experiment.py --temperatures 0.4 0.7 1.0 1.3 --num_samples 3

# 第5步：超参对比实验（可选，耗时长）
python hyperparameter_experiment.py

# 第6步：收集所有结果、截图、整理报告
# 复制 loss_curve.png、temperature_results.txt、hyperparameter_results.txt 等到报告中
```

---

## 补充说明

- **时间估计**：
  - 必做部分（3个epochs）：~30-60分钟（GPU）/ ~2-4小时（CPU）
  - 加分项1（Temperature）：~5-10分钟
  - 加分项2（超参）：~2-3小时（GPU）/ ~10-20小时（CPU）

- **显存需求**：
  - 最小：4GB（降低 batch_size）
  - 推荐：8GB+
  - 如果显存不足，可在训练脚本中调整 `BATCH_SIZE` 和模型大小

---

祝实验顺利！ 🎉
