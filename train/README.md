# Genie 训练脚本

针对不同序列长度的训练脚本，使用多种优化。

## 可用脚本

### 1. `train_long.py` - 长序列

使用 Stage 3 优化进行长序列训练。

**特性：**
- 因子化配对特征 (Stage 1)
- 因子化三角形运算 (Stage 2)
- 渐进式训练课程
- 混合精度 (FP16)
- 分块损失计算

**使用方法：**

```bash
# 使用 Stage 3 进行基础训练
python train/train_long.py --config configs/train_stage3.config

# 多 GPU 训练
python train/train_long.py --config configs/train_stage3.config --gpus 0,1

# 从检查点恢复训练
python train/train_long.py --config configs/train_stage3.config --resume path/to/checkpoint.ckpt

# 调试模式
python train/train_long.py --config configs/train_stage3.config --debug
```

**推荐用于：**
- 长序列
- 单 GPU (12-24GB 显存)
- 高效训练且保证质量

---

### 2. `train_ultralong.py` - 超长序列

使用所有优化阶段进行超长序列训练。

**特性：**

- 全部 Stage 1-3 优化
- 稀疏 k-NN 配对 (Stage 3 v2)
- 轴向注意力 (Stage 4)
- 梯度检查点和模型压缩
- 分布式训练支持 (Stage 5)

**使用方法：**

```bash
# Stage 3 V2: 超长序列 L=4096
python train/train_ultralong.py --config configs/train_stage3v2.config

# Stage 4: 很长序列 L=8192
python train/train_ultralong.py --config configs/train_stage4.config --gpus 0

# Stage 5: 分布式训练 L=16384+
torchrun --nproc_per_node=8 train/train_ultralong.py --config configs/train_stage5.config

# 多节点分布式训练
torchrun --nproc_per_node=8 --nnodes=2 train/train_ultralong.py \
    --config configs/train_stage5.config --nodes 2

# 启用性能分析
python train/train_ultralong.py --config configs/train_stage4.config --profile
```

**推荐用于：**
- 超长序列
- 单 GPU 或多 GPU 设置
- 最大内存效率

---

## 配置文件

配置文件位于 `configs/` 目录：

| 配置文件 | 阶段 | 用途 |
|----------|------|------|
| `train_stage1.config` | 1 | 因子化配对 |
| `train_stage2.config` | 1+2 | + 三角形运算 |
| `train_stage3.config` | 1+2+3 | + 训练优化 (FP16) |
| `train_stage3v2.config` | 1+2+3+V2 | + 稀疏 k-NN |
| `train_stage4.config` | 1+2+3+V2+4 | + 轴向注意力 |
| `train_stage5.config` | 全部 | + 分布式训练 |

---

## 命令行参数

### 通用参数 (两个脚本)

- `--config`: 配置文件路径 **(必填)**
- `--gpus`: 要使用的 GPU ID (例如 "0,1,2,3")
- `--resume`: 要恢复的检查点路径
- `--debug`: 启用调试模式并输出详细日志

### 额外参数 (train_ultralong.py)

- `--nodes`: 分布式训练的节点数
- `--profile`: 启用性能分析模式

---

## 示例

### 示例 1: 使用 Stage 3 训练 

```bash
python train/train_long.py --config configs/train_stage3.config
```

### 示例 2: 多 GPU 训练 

```bash
python train/train_long.py --config configs/train_stage3.config --gpus 0,1,2,3
```

### 示例 3: 超长序列使用稀疏 k-NN 

```bash
python train/train_ultralong.py --config configs/train_stage3v2.config
```

### 示例 4: 分布式训练

```bash
# 单节点 8 个 GPU
torchrun --nproc_per_node=8 train/train_ultralong.py \
    --config configs/train_stage5.config

# 2 节点共 16 个 GPU
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr="192.168.1.1" --master_port=29500 \
    train/train_ultralong.py --config configs/train_stage5.config --nodes 2
```

---

## 技巧

1. **从小开始:** 先使用 Stage 3 和较短序列验证配置
2. **监控内存:** 使用 `nvidia-smi` 检查 GPU 显存使用情况
3. **调整批量大小:** 如果出现 OOM，在配置中减小 `batchSize`
4. **使用梯度累积:** 模拟更大的批次而不消耗更多内存
5. **渐进式训练:** 有助于长序列的收敛

---

## 可能故障排除

### 内存不足 (OOM)

1. 减小配置文件中的批量大小
2. 启用梯度检查点 (Stage 4+)
3. 使用分块损失计算
4. 尝试较小的最大序列长度

### 训练缓慢

1. 启用混合精度 (FP16)
2. 使用梯度累积实现有效的大批次
3. 启用轴向注意力 (Stage 4)
4. 使用分布式训练 (Stage 5)

### NaN 损失

1. 减小学习率
2. 启用梯度裁剪
3. 在混合精度下使用梯度缩放器
4. 检查数据问题

---

## 另请参阅

- [训练配置文档](../configs/README.md) - 详细配置说明
- [采样脚本](../sample/README.md) - 如何生成样本
- [长序列指南](../docs/LONG_SEQUENCE_README.md) - 技术细节
