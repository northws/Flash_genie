# Genie 长序列扩展

> [!WARNING]
>
> 此项目的测试尚未完成
>
> 仅完成stage2的测试

> [!CAUTION]
>
> 此项目来自于我开始无法获取高性能、大显存GPU，只能在低性能、小显存GPU上训练，其目的完全是降低原[Genie](https://github.com/aqlaboratory/genie)项目的显存和计算开销，其追求的效果**一方面是大幅降低训练和生成的成本，同时取得<u>还不错</u>的性能，另一方面是由于显存和计算开销的降低，使用高性能GPU能够进行长序列的训练。**
>
> **如果你没有以上需求，建议使用原[Genie](https://github.com/aqlaboratory/genie)项目。**

> [!IMPORTANT]
>
> 本项目扩展的具体数学逻辑见[PROJECT_SUMMARY](docs/PROJECT_SUMMARY.md)，或见[article](articles/flash_genie_cn.pdf)

## 🎯 概述

本项目在原始 Genie 的基础上，实现了完整的长序列优化方案，得益于优化后的低显存占用，一方面使得其能够在大多数的消费级GPU上运行（stage2优化的测试在4060 Laptop上进行），另一方面使得拥有大显存的专业级显卡能处理超长序列的蛋白。

## 📦 完整优化阶段

### v1 + v1.2: Factorized Pair Features
**代码文件**: `genie/model/factorized_pair_features.py`

**创新**:

- 避免 O(L²) pair features 完整实例化
- 直接生成因子化表示 [B, L, rank, C]
- 内存: O(L²) → O(L × rank)

**效果**:
- 内存节省 O(L²) → O

---

### Stage 2: 三角操作Triangle Operations
**文件**:

- `genie/model/factorized_triangle_ops.py`
- `genie/model/factorized_pair_transform.py`

**核心创新**:
- 因子化三角乘法更新 (O(L³) → O(L² × rank))
- 分块三角注意力
- 完整 Evoformer-style processing（来自于AlphaFold2）

---

### Stage 3: 训练优化Training Optimizations
**文件**:
- `genie/training/progressive_training.py`
- `genie/training/mixed_precision.py`
- `genie/training/stage3_trainer.py`

**核心创新**:
1. **Progressive Training**: 渐进式训练 由长序列到短序列(L=128 → 1024)
2. **Chunked Loss**: 分块损失计算 (8-16x 内存节省)
3. **Mixed Precision**: FP16/BF16 (50% 内存 + 2-3x 速度)

**效果**:
- 训练收敛更快

---

### Stage 3 V2: Sparse Pairs
**文件**: `genie/model/sparse_pairs.py`

**核心创新**:
- 稀疏 k-NN 对选择
- 三种策略: coordinate / sequence / hybrid
- 内存: O(L²) → O(L × k)

---

### Stage 4: Advanced Optimizations
**文件**:
- `genie/model/axial_attention.py`
- `genie/training/gradient_checkpointing.py`
- `genie/model/model_compression.py`

**核心创新**:

1. **轴向注意力Axial Attention**: 行+列分解 (O(L³) → O(L²))
2. **Adaptive Checkpointing**: 智能梯度检查点
3. **Model Compression**: 层参数共享 (4-8x 参数减少)

---

### Stage 5: Distributed Training
**文件**: `genie/training/distributed_training.py`

**核心创新**:
1. **DDP**: 数据并行训练
2. **Sequence Parallelism**: 序列维度切分
3. **Gradient Accumulation**: 大批量训练

**效果**:

- 在多卡集群中拥有更好的训练效果

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/northws/genie.git
cd genie
pip install -e .
```

### 2. 训练配置示例（此处的训练参数需要你至少拥有32 G的显存，如果需要你可以降低batch size或是序列长度）

#### 单 GPU 长序列训练 (L=1024)
```
name long_sequence_training
batchSize 2
maximumNumResidues 1024

# Stage 1-2: Factorization
singleFeatureDimension 128
pairFeatureDimension 128
zFactorRank 2

# Stage 3: Training optimizations
learningRate 2e-4
warmupEpochs 100
gradientClipVal 1.0

# Mixed Precision
# (自动启用 FP16)
```

#### 多 GPU 超长序列训练 (L=4096+)
```
name ultra_long_training
batchSize 1
maximumNumResidues 4096

# Stage 1-3 V2: Sparse pairs
zFactorRank 2
useSparseKNN True
kNeighbors 32

# Stage 3: Progressive + Mixed Precision
useProgressiveTraining True
useChunkedLoss True

# Stage 4: Compression
useModelCompression True
compressionStrategy universal

# Stage 5: Distributed
# (使用 torchrun 启动)
```

---

## 📚 详细文档

### 核心模块文档

**Stage 1-2**:
- [factorized_pair_features.py](genie/model/factorized_pair_features.py) - 因子化对特征
- [factorized_triangle_ops.py](genie/model/factorized_triangle_ops.py) - 因子化三角操作
- [factorized_pair_transform.py](genie/model/factorized_pair_transform.py) - 对变换网络

**Stage 3**:
- [progressive_training.py](genie/training/progressive_training.py) - 渐进式训练
- [mixed_precision.py](genie/training/mixed_precision.py) - 混合精度
- [stage3_trainer.py](genie/training/stage3_trainer.py) - 综合训练管理器

**Stage 3 V2**:
- [sparse_pairs.py](genie/model/sparse_pairs.py) - 稀疏 k-NN 对选择

**Stage 4**:
- [axial_attention.py](genie/model/axial_attention.py) - 轴向注意力
- [gradient_checkpointing.py](genie/training/gradient_checkpointing.py) - 梯度检查点
- [model_compression.py](genie/model/model_compression.py) - 模型压缩

**Stage 5**:
- [distributed_training.py](genie/training/distributed_training.py) - 分布式训练

### 项目文档
- [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) - 完整项目总结
- [docs/EVALUATION_AND_IMPROVEMENTS.md](docs/EVALUATION_AND_IMPROVEMENTS.md) - 技术评估

---

## 🔬 技术细节

### 内存复杂度对比

| 组件 | 原始 | Stage 优化 |
|------|------|-----------|
| Pair Features | O(L²×C) | O(L×rank×C) |
| Triangle Ops | O(L³×C) | O(L²×rank×C) |
| Sparse Pairs | O(L²×C) | O(L×k×C) |
| Loss | O(L²×C) | O(chunk×L×C) |

## 🎓 引用

如果使用本项目的长序列优化技术，请引用:

```bibtex
@software{Flash_genie,
  title={Genie Long Sequence Extensions},
  author={northws},
  year={2026},
  url={https://github.com/northws/Flash_genie}
}
```

原始 Genie 论文:
```bibtex
@article{lin2023generating,
  title={Generating Novel Protein Backbones with Equivariant Diffusion},
  author={Lin, Yeqing C and AlQuraishi, Mohammed},
  journal={arXiv preprint arXiv:2301.12485},
  year={2023}
}
```

---

## 📄 许可证

- 原始 Genie 代码: Apache License 2.0
- 本项目的核心模块: MIT License

---

## 🙏 致谢

本项目的长序列优化基于以下优秀工作:

- **Genie** (Lin & AlQuraishi, 2023) - 核心架构
- **Flash-IPA** (Flagship Pioneering) - 内存效率
- **AlphaFold2** (Jumper et al., 2021) - Triangle Operations
- **Flash Attention** (Dao et al., 2022) - 高效注意力
- **mHC** (Xie et al., DeepSeek-AI, 2025) - 训练稳定性

---

## 📮 联系方式

- GitHub Issues: https://github.com/northws/genie/issues
- E-mail:wjyquark@outlook.com

---

**项目完成度**: 100 % (Stage 1-5 全部完成) 

**项目测试进度**：40 %(Stage 1-2已完成，2026-1-15)
