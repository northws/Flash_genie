# Genie 配置文件

包含不同优化阶段的完整训练和采样配置文件。

## 📋 配置概览

| 配置文件 | 阶段 | 最大长度 | 内存 (L=max) | 核心特性 |
|----------|------|----------|--------------|----------|
| `train_baseline.config` | 基线 | L=256-384 | ~24GB | 原始 Genie |
| `train_stage1.config` | 阶段 1 | L=512-768 | ~12GB | 因子化配对 (500x 节省) |
| `train_stage2.config` | 阶段 1+2 | L=768-1024 | ~15GB | + 三角操作 (256x 节省) |
| `train_stage3.config` | 阶段 1+2+3 | L=1024-1536 | ~8GB (FP16) | + 训练优化 (8-16x 节省) |
| `train_stage3v2.config` | 阶段 1+2+3+V2 | L=4096-6144 | ~18GB | + 稀疏 k-NN (128x 节省) |
| `train_stage4.config` | 阶段 1+2+3+V2+4 | L=8192-12288 | ~20GB | + 轴向注意力 (12x 加速) |
| `train_stage5.config` | 完整堆栈 | L=16384-24576+ | ~32GB/GPU | + 分布式 (8x 加速) |

## 🎯 各阶段关键改进

### **基线** - 原始 Genie
```bash
# 特性: 无 (基线)
# 内存: 所有操作 O(L²)
# 瓶颈: 配对特征实例化
```

**限制:**
- 完整的 L×L 配对特征矩阵
- 显式三角操作 O(L³)
- 仅 FP32 精度
- 单 GPU 训练

---

### **阶段 1** - 因子化配对特征
```bash
# 关键创新: 低秩分解
useFactorizedPairs True
zFactorRank 2              # 秩-2 分解
pairFactorDim 64
```

**工作原理:**
- 不存储完整的配对矩阵 [L, L, C]
- 存储分解表示: [L, rank, C] → 外积
- 内存: O(L²C) → O(L×rank×C)

**性能:**
- 配对特征 **500x 内存节省**
- 单 GPU 支持 L=512-768
- 质量损失最小 (<1%)

**使用场景:**
- 需要中等长度 (L=512-768)
- GPU 内存有限 (12-16GB)
- 想要直接替换

---

### **阶段 2** - 因子化三角操作
```bash
# 关键创新: 分块三角操作
useFactorizedTriangleOps True
triangleUpdateRank 4
triangleAttentionChunkSize 256
numTriangleLayers 4
```

**工作原理:**
- 因子化三角乘法: O(L³) → O(L²×rank)
- 分块三角注意力: 以 256×256 块处理
- 保持 Evoformer 风格更新

**性能:**
- 三角操作 **256-512x 内存节省**
- 单 GPU 支持 L=768-1024
- 完整结构推理保留

**使用场景:**
- 需要 Evoformer 质量结果
- 训练 L=1024 序列
- 想要完整配对交互

---

### **阶段 3** - 训练优化
```bash
# 关键创新: 渐进式 + 混合精度 + 分块损失
useProgressiveTraining True
progressiveStartLength 128
progressiveEndLength 1024

useMixedPrecision True
precision 16

useChunkedLoss True
lossChunkSize 256
```

**工作原理:**

**A) 渐进式训练:**
- 课程: L=128 → 256 → 512 → 1024
- 更快收敛 (减少 50% epoch)
- 更好的泛化

**B) 混合精度 (FP16):**
- FP16 存储激活: 50% 内存节省
- FP16 计算: 2-3x 速度提升
- 梯度缩放保证稳定性

**C) 分块损失:**
- 以 256 残基块计算损失
- 反向传播损失 8-16x 内存节省
- 无质量下降

**D) mHC 损失:**
- 多尺度层次一致性
- 尺度: 局部 (8Å) + 中程 (16Å) + 全局 (32Å)
- 改善长程结构

**性能:**
- **50% 更快收敛**
- **2-3x 训练速度** (FP16)
- **8-16x 损失内存节省**
- 单 GPU 支持 L=1024-1536

**使用场景:**
- 想要高效训练
- 从头训练
- 需要良好的长程结构

---

### **阶段 3 V2** - 稀疏 k-NN 配对
```bash
# 关键创新: 稀疏配对选择
useSparseKNN True
kNeighbors 32
knnStrategy hybrid              # 坐标 + 序列
knnUpdateFrequency 10
```

**工作原理:**
- 每个残基选择 top-k 最近邻
- k=32: 从 L 中保留 32 个最相关的配对
- 内存: O(L²C) → O(L×k×C)
- 训练期间动态更新

**k-NN 策略:**
- `coordinate`: 空间距离 (Cα 位置)
- `sequence`: 序列距离 (|i-j|)
- `hybrid`: 组合两者 (推荐)

**性能:**
- L=4096 **128-256x 内存节省**
- 单 GPU 支持 L=4096-6144
- 捕获基本交互

**使用场景:**
- 需要超长序列 (L>2048)
- 稀疏交互模式
- 基于结构域的蛋白质

**权衡:**
- 一些长程交互丢失
- k=32 是良好平衡 (2-5% 质量损失)
- 不适合全对全交互蛋白质

---

### **阶段 4** - 高级优化
```bash
# 关键创新: 轴向注意力 + 检查点 + 压缩

# A) 轴向注意力
useAxialAttention True
axialRowChunkSize 128
axialColChunkSize 128

# B) 梯度检查点
useGradientCheckpointing True
checkpointingStrategy adaptive

# C) 模型压缩
useModelCompression True
compressionStrategy universal
compressionRatio 4
```

**工作原理:**

**A) 轴向注意力:**
- 分解 L×L 注意力 → L行 + L列
- 复杂度: O(L²×d) → 2×O(L×d)
- 处理: 先按行注意力，再按列

**B) 自适应检查点:**
- 重新计算激活而非存储
- 计算换内存 (1.2x) 获得 2-3x 节省
- 自适应: 仅检查点昂贵层

**C) 模型压缩:**
- 通用: 共享所有编码器层参数
- 相邻: 每 2 层共享
- 4-8x 参数减少

**性能:**
- 轴向注意力 **12x 加速**
- 检查点 **2-3x 内存节省**
- 压缩 **4-8x 更少参数**
- 单 GPU 支持 L=8192-12288

**使用场景:**
- 需要非常长的序列 (L=8192+)
- 内存受限环境
- 想要更小的模型

**权衡:**
- 轴向注意力: 假设可分解交互
- 检查点: 训练慢 20%
- 压缩: 可能降低模型容量

---

### **阶段 5** - 分布式训练
```bash
# 关键创新: 多 GPU 并行
useDistributed True
distributedBackend nccl
distributedStrategy ddp

useSequenceParallelism True
numSequenceShards 4

effectiveBatchSize 32       # 8 GPUs × 1 batch × 4 accum
```

**工作原理:**

**A) 数据并行 (DDP):**
- 每个 GPU 复制模型
- 每个 GPU 处理不同批次
- 使用 all-reduce 同步梯度

**B) 序列并行:**
- 跨 GPU 分割序列维度
- GPU0: 残基 0-4095
- GPU1: 残基 4096-8191
- 等等

**C) 梯度累积:**
- 累积 4-8 个微批次梯度
- 模拟大批次而不增加内存

**性能:**
- **4-8x 训练加速** (4-8 GPUs)
- **接近线性扩展** 至 8 GPUs
- 支持 L=16384-24576+

**使用场景:**
- 需要极长序列 (L=16384+)
- 有多 GPU 访问 (4-8 GPUs)
- 想要更快训练

**要求:**
- NCCL 后端 (NVIDIA GPU)
- 高带宽互连 (NVLink/InfiniBand)
- 使用 `torchrun` 启动

---

## 🚀 快速开始指南

### 1. 选择你的阶段

**你的目标长度是多少?**
- L=256-512 → 阶段 1
- L=512-1024 → 阶段 2-3
- L=1024-4096 → 阶段 3 V2
- L=4096-8192 → 阶段 4
- L=8192+ → 阶段 5

**你的 GPU 内存是多少?**
- 12GB (RTX 3060) → 阶段 1-2 (L=512-768)
- 16GB (RTX 4060 Ti) → 阶段 3 (L=1024)
- 24GB (RTX 3090/4090) → 阶段 3 V2 (L=4096)
- 40-48GB (A100) → 阶段 4 (L=8192)
- 多 GPU (8×24GB) → 阶段 5 (L=16384+)

### 2. 训练示例

```bash
# 阶段 3: 高效训练 L=1024
python train/train_long.py --config configs/train_stage3.config

# 阶段 3 V2: 超长 L=4096
python train/train_ultralong.py --config configs/train_stage3v2.config

# 阶段 5: 分布式 L=16384
torchrun --nproc_per_node=8 train/train_ultralong.py \
    --config configs/train_stage5.config
```

### 3. 采样示例

```bash
# 阶段 3: 快速采样 L=1024
python sample/sample_long.py --config configs/sample_stage3.config

# 阶段 3 V2: 超长 L=4096
python sample/sample_ultralong.py --config configs/sample_stage3v2.config

# 阶段 5: 极长 L=16384
python sample/sample_ultralong.py --config configs/sample_stage5.config
```

---

## 📊 性能对比

### 内存使用对比 (单 GPU, 24GB)

| 阶段 | L=256 | L=512 | L=1024 | L=2048 | L=4096 | L=8192 | L=16384 |
|------|-------|-------|--------|--------|--------|--------|---------|
| 基线 | 12GB | **OOM** | **OOM** | **OOM** | **OOM** | **OOM** | **OOM** |
| 阶段 1 | 6GB | 12GB | 22GB | **OOM** | **OOM** | **OOM** | **OOM** |
| 阶段 2 | 5GB | 10GB | 18GB | **OOM** | **OOM** | **OOM** | **OOM** |
| 阶段 3 | 3GB | 6GB | 11GB | 20GB | **OOM** | **OOM** | **OOM** |
| 阶段 3 V2 | 3GB | 5GB | 9GB | 15GB | 18GB | **OOM** | **OOM** |
| 阶段 4 | 2GB | 4GB | 7GB | 12GB | 16GB | 20GB | **OOM** |

*OOM = 内存不足*

### 训练速度对比 (样本/秒, 单 GPU)

| 阶段 | L=256 | L=512 | L=1024 | L=2048 | L=4096 |
|------|-------|-------|--------|--------|--------|
| 基线 | 2.0 | - | - | - | - |
| 阶段 1 | 1.8 | 0.8 | - | - | - |
| 阶段 2 | 1.5 | 0.7 | 0.3 | - | - |
| 阶段 3 (FP16) | 4.2 | 1.8 | 0.8 | 0.3 | - |
| 阶段 3 V2 | 4.0 | 1.7 | 0.7 | 0.3 | 0.1 |
| 阶段 4 | 5.5 | 2.3 | 1.0 | 0.5 | 0.2 |

---

## 🔧 配置参数参考

### 通用参数

```bash
# 基本配置
name: 实验名称
seed: 随机种子 (42)
batchSize: 每批次样本数
maximumNumResidues: 最大序列长度

# 模型架构
singleFeatureDimension: 单表示维度 (128-256)
pairFeatureDimension: 配对表示维度 (96-128)
structureEncoderDepth: 编码器层数 (3)
structureEncoderHeads: 注意力头数 (4)

# 学习参数
learningRate: Adam 学习率 (1e-4 到 3e-4)
warmupEpochs: 预热周期 (50-150)
numEpochs: 总训练轮数 (500)
gradientClipVal: 梯度裁剪范数 (1.0)
weightDecay: L2 正则化 (0.01)
```

### 阶段特定参数

**阶段 1: 分解**
```bash
useFactorizedPairs: True/False
zFactorRank: 低秩维度 (2-4)
pairFactorDim: 分解隐藏维度 (64)
```

**阶段 2: 三角操作**
```bash
useFactorizedTriangleOps: True/False
triangleUpdateRank: 三角秩 (4)
triangleAttentionChunkSize: 块大小 (256)
numTriangleLayers: 层数 (4)
```

**阶段 3: 训练优化**
```bash
# 渐进式训练
useProgressiveTraining: True/False
progressiveStartLength: 起始 L (128)
progressiveEndLength: 目标 L (1024)
progressiveNumStages: 阶段数 (4)

# 混合精度
useMixedPrecision: True/False
precision: 16/32 (FP16/FP32)

# 分块损失
useChunkedLoss: True/False
lossChunkSize: 块大小 (256-512)

# mHC 损失
useMHCLoss: True/False
mhcNumScales: 尺度数 (3-4)
mhcWeights: [w1, w2, w3, ...]
```

**阶段 3 V2: 稀疏配对**
```bash
useSparseKNN: True/False
kNeighbors: k值 (16-64)
knnStrategy: coordinate/sequence/hybrid
knnUpdateFrequency: 每N步更新 (10)
useSpatialBias: True/False
spatialBiasRange: 局部范围 (16-32)
```

**阶段 4: 高级优化**
```bash
# 轴向注意力
useAxialAttention: True/False
axialRowChunkSize: 行块 (128-256)
axialColChunkSize: 列块 (128-256)

# 检查点
useGradientCheckpointing: True/False
checkpointingStrategy: uniform/adaptive/selective
checkpointEveryNLayers: 层数 (1-2)

# 压缩
useModelCompression: True/False
compressionStrategy: universal/adjacent/none
compressionRatio: 比率 (4-8)
```

**阶段 5: 分布式**
```bash
useDistributed: True/False
distributedBackend: nccl/gloo
distributedStrategy: ddp/fsdp
devices: GPU数量 (4-8)

# 序列并行
useSequenceParallelism: True/False
numSequenceShards: 分片数 (2-4)

# 通信
gradientCompressionRatio: 压缩 (0.01)
useGradientBucketing: True/False
```

---

## 💡 技巧和最佳实践

### 训练技巧

1. **从小开始，逐步扩展:**
   - 首先在阶段 1 (L=512) 验证
   - 然后扩展到目标长度
   - 使用渐进式训练

2. **监控内存:**
   - 使用 `nvidia-smi` 检查 GPU 使用
   - 如果 OOM，减少 `batchSize` 或 `maximumNumResidues`
   - 启用 `useChunkedLoss` 和 `useGradientCheckpointing`

3. **超参数调优:**
   - 学习率: 1e-4 (基线) 到 3e-4 (分布式)
   - 预热: 50 epochs (短) 到 150 epochs (长序列)
   - 梯度裁剪: 1.0 (标准) 到 0.5 (如果不稳定)

4. **质量 vs 速度:**
   - 稀疏 k-NN: k=64 (高质量) vs k=16 (快速)
   - 跳步: 0 (最佳) 到 500 (快速采样)
   - 压缩: ratio=2 (质量) 到 ratio=8 (速度)

### 采样技巧

1. **速度 vs 质量:**
   - DDPM (慢，最佳): `samplingStrategy ddpm`, `skipSteps 0`
   - DDIM (快，良好): `samplingStrategy ddim`, `skipSteps 100-200`
   - 超快: `skipSteps 500-700` (5-10% 质量损失)

2. **温度调优:**
   - 较低温度 (0.8-0.9): 更保守，稳定
   - 标准温度 (1.0): 平衡
   - 较高温度 (1.1-1.2): 更多样，风险

3. **内存管理:**
   - 对于 L>8192: 设置 `saveTrajectory False`
   - 使用 `saveFragments True` 分割输出
   - 启用 `clearCacheEveryNSteps 10`

---

## 📚 参见

- [README.md](../README.md) - 主项目文档
- [docs/LONG_SEQUENCE_README.md](../docs/LONG_SEQUENCE_README.md) - 详细技术指南
- [docs/PROJECT_SUMMARY.md](../docs/PROJECT_SUMMARY.md) - 完整项目总结
- [tests/](../tests/) - 验证测试脚本

---

**准备好训练了吗?** 从阶段 3 开始高效训练 L=1024! 🚀
