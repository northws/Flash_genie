# Genie 测试套件

本目录包含了 Genie 长序列扩展项目的所有测试文件，在使用前你可以先行测试。

## 📋 测试文件概览

### Stage 1: 核心优化
**文件**: [`test_long_sequence_stage1.py`](test_long_sequence_stage1.py)

**测试内容**:
- Factorized Pair Features (因子化pair特征)
- Adaptive mHC Configuration (自适应mHC配置)
- Long Sequence Denoiser 集成
- 内存缩放验证

**运行**:
```bash
python tests/test_long_sequence_stage1.py
```

**预期结果**:

- 5个测试全部通过
- L=512内存: 537 MB → 1 MB (537x reduction)
- L=1024内存: 2.1 GB → 4 MB (512x reduction)

---

### Stage 2: Triangle Operations 优化
**文件**:
- [`test_stage2_optimizations.py`](test_stage2_optimizations.py) - 完整测试套件
- [`test_stage2_quick.py`](test_stage2_quick.py) - 快速集成测试

**测试内容**:
- Factorized Triangle Multiplicative Update (因子化三角乘法更新)
- Chunked Triangle Attention (分块三角注意力)
- Factorized Pair Transform Network (因子化pair变换网络)
- 内存缩放分析 (L=1024, L=2048)
- Stage 1 vs Stage 2 性能对比

**运行**:
```bash
# 快速测试 (推荐)
python tests/test_stage2_quick.py

# 完整测试
python tests/test_stage2_optimizations.py
```

**预期结果**:
- L=1024 Triangle Mult: 1024 MB → 4 MB (256x reduction)
- L=2048 Triangle Mult: 8192 MB → 8 MB (1024x reduction)
- L=1024 Triangle Att: 128 MB → 8 MB (16x reduction)

---

### Stage 3: 训练优化
**文件**: [`test_stage3_optimizations.py`](test_stage3_optimizations.py)

**测试内容**:
- Progressive Training Scheduler (渐进式训练调度)
- Chunked Loss Computation (分块损失计算)
- Mixed Precision Training (混合精度训练)
- Stage3TrainingManager 集成
- 性能对比分析

**运行**:
```bash
python tests/test_stage3_optimizations.py
```

**预期结果**:
- Progressive training: 50% 更快收敛
- Chunked loss: 8-16x 内存节省
- Mixed precision: 50% 内存节省 + 2-3x 速度提升

---

### Stage 3 V2: 超长序列支持
**文件**: [`test_stage3_v2.py`](test_stage3_v2.py)

**测试内容**:

- Sparse k-NN Pair Selection (稀疏k-NN对选择)
- 三种选择策略 (coordinate / sequence / hybrid)
- 超长序列内存缩放 (L=4096, L=8192)
- Stage 3 V2 完整集成

**运行**:
```bash
python tests/test_stage3_v2.py
```

**预期结果**:
- L=4096: 64 GB → 0.5 MB (128x reduction)
- L=8192: 256 GB → 1 MB (256x reduction)
- 支持 L=4096-8192 训练

---

### Stage 4-5: 进阶优化与分布式训练
**文件**: [`test_stage4_5.py`](test_stage4_5.py)

**测试内容**:
- Axial Attention (轴向注意力)
- Advanced Gradient Checkpointing (高级梯度检查点)
- Model Compression (模型压缩)
- Distributed Training (分布式训练)
- 完整集成测试

**运行**:
```bash
python tests/test_stage4_5.py
```

**预期结果**:
- Axial attention: 12x 计算加速 (L=1024)
- Model compression: 4-8x 参数减少
- Distributed training: 4-8x 吞吐量提升

---

### MHC + Flash 集成测试
**文件**: [`test_mhc_flash_combined.py`](test_mhc_flash_combined.py)

**测试内容**:
- mHC 集成验证
- Flash-IPA 集成验证
- 组合优化测试

**运行**:
```bash
python tests/test_mhc_flash_combined.py
```

---

## 🚀 快速开始

### 运行所有测试
```bash
# Stage 1
python tests/test_long_sequence_stage1.py

# Stage 2 (快速)
python tests/test_stage2_quick.py

# Stage 3
python tests/test_stage3_optimizations.py

# Stage 3 V2
python tests/test_stage3_v2.py

# Stage 4-5
python tests/test_stage4_5.py

# MHC Flash
python tests/test_mhc_flash_combined.py
```

### 选择性测试
```bash
# 只测试最新的优化 (Stage 4-5)
python tests/test_stage4_5.py

# 只测试超长序列支持 (Stage 3 V2)
python tests/test_stage3_v2.py

# 快速验证 Stage 2
python tests/test_stage2_quick.py
```

---

## 🎯 测试策略

### 单元测试
每个优化组件都有独立的单元测试:
- 独立验证功能正确性
- 测试边界条件
- 验证错误处理

### 集成测试
验证多个组件协同工作:
- 端到端流程测试
- 性能基准测试
- 内存和速度分析

### 回归测试
确保新优化不破坏现有功能:
- 跨 Stage 兼容性测试
- 向后兼容性验证
- 性能退化检测

---

## 💡 测试顺序

### 1. 从快速测试开始
```bash
# 先运行快速测试验证基本功能
python tests/test_stage2_quick.py
```

### 2. 按 Stage 顺序测试
```bash
# Stage 1 → Stage 2 → Stage 3 → Stage 3 V2 → Stage 4-5
python tests/test_long_sequence_stage1.py
python tests/test_stage2_quick.py
python tests/test_stage3_optimizations.py
python tests/test_stage3_v2.py
python tests/test_stage4_5.py
```

### 3. 关注内存使用
- 监控 GPU 内存使用
- 验证内存缩放比例
- 确认没有内存泄漏

### 4. 验证数值稳定性
- 检查输出形状正确
- 验证梯度流动正常
- 确认数值范围合理

---

## ⚠️ 可能遇见的问题

### 1.测试 OOM (Out of Memory)
减小测试序列长度或batch size，或使用更小的模型配置

### 2.测试运行缓慢
- 使用快速测试版本 (test_stage2_quick.py)
- 减少测试迭代次数
- 使用更小的序列长度

### 3.导入错误
确保从项目根目录运行测试:
```bash
cd /root/Flash_genie
python tests/test_long_sequence_stage1.py
```

### 4.CUDA 错误
- 检查 GPU 可用性

  `nvidia-smi`

- 验证 CUDA 版本兼容

- 尝试 CPU 模式 (修改测试中的 device)

---

## 📚 相关文档

- [PROJECT_SUMMARY.md](../docs/PROJECT_SUMMARY.md) - 项目总结和成果
- [EVALUATION_AND_IMPROVEMENTS.md](../docs/EVALUATION_AND_IMPROVEMENTS.md) - 技术评估
- [LONG_SEQUENCE_README.md](../docs/LONG_SEQUENCE_README.md) - 长序列使用指南

---

**创建时间**: 2026-01-09
**最后更新**: 2026-01-15
**维护者**: northws
