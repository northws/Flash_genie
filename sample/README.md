# Genie 采样脚本

用于生成不同序列长度蛋白质结构的采样脚本。

## 可用脚本

### 1. `sample_long.py` - 长序列

使用 Stage 3 优化进行长序列快速采样。

**特性：**
- 因子化配对特征 (Stage 1)
- 因子化三角形运算 (Stage 2)
- 混合精度 (FP16) 加速
- 批量采样支持
- DDIM/DDPM 采样策略

**使用方法：**

```bash
# 使用 DDIM 进行基础采样
python sample/sample_long.py --config configs/sample_stage3.config

# 高质量 DDPM 采样
python sample/sample_long.py --config configs/sample_stage3.config --ddpm --skip_steps 0

# 自定义参数的批量采样
python sample/sample_long.py \
    --config configs/sample_stage3.config \
    --num_samples 100 \
    --length 1024 \
    --batch_size 4

# 保存 PDB 文件
python sample/sample_long.py --config configs/sample_stage3.config --save_pdb

# 自定义 GPU 和随机种子
python sample/sample_long.py --config configs/sample_stage3.config --gpu 1 --seed 12345
```

**推荐用于：**
- 长序列
- 快速生成 (FP16 加速 2-3 倍)
- 批量采样

---

### 2. `sample_ultralong.py` - 超长序列

内存高效的超长序列采样。

**特性：**
- 全部 Stage 1-3 优化
- 稀疏 k-NN 配对 (Stage 3 V2)
- 轴向注意力 (Stage 4)
- Flash attention 支持
- 内存管理和缓存清理
- 轨迹片段保存

**使用方法：**

```bash
# Stage 3 V2: 超长序列 L=4096
python sample/sample_ultralong.py --config configs/sample_stage3v2.config

# Stage 5: 极长序列 L=16384 闪存模式
python sample/sample_ultralong.py \
    --config configs/sample_stage5.config \
    --flash_mode \
    --save_fragments

# 内存高效模式，自定义分块大小
python sample/sample_ultralong.py \
    --config configs/sample_stage5.config \
    --chunk_size 2048 \
    --clear_cache_steps 5

# 保存轨迹片段而非完整轨迹
python sample/sample_ultralong.py \
    --config configs/sample_stage3v2.config \
    --save_fragments \
    --save_pdb
```

**推荐用于：**
- 序列长度 L=4096+
- 内存受限环境
- 极长序列生成 (L>10k)

---

## 配置文件

采样配置文件位于 `configs/` 目录：

| 配置文件 | 阶段 | 特性 |
|----------|------|------|
| `sample_stage1.config` | 1 | 因子化配对 |
| `sample_stage3.config` | 1+2+3 | + 混合精度 |
| `sample_stage3v2.config` | 1+2+3+V2 | + 稀疏 k-NN |
| `sample_stage5.config` | 全部 | 完整优化 |

---

## 命令行参数

### 通用参数 (两个脚本)

- `--config`: 配置文件路径 **(必填)**
- `--checkpoint`: 模型检查点路径 (覆盖配置)
- `--output_dir`: 输出目录 (覆盖配置)
- `--num_samples`: 要生成的样本数量
- `--length`: 目标序列长度
- `--gpu`: GPU 设备 ID (默认: 0)
- `--seed`: 随机种子 (用于复现)
- `--ddpm`: 使用 DDPM 而非 DDIM (较慢，质量更高)
- `--skip_steps`: 要跳过的扩散步数
- `--temperature`: 采样温度 (默认: 1.0)
- `--save_pdb`: 将结构保存为 PDB 文件
- `--save_trajectory`: 保存完整轨迹 (内存密集)

### 额外参数 (sample_ultralong.py)

- `--chunk_size`: 内存高效采样的分块大小
- `--flash_mode`: 启用 flash attention
- `--save_fragments`: 保存轨迹片段 (节省内存)
- `--clear_cache_steps`: 每 N 步清空 CUDA 缓存 (默认: 10)

---

## 采样策略

### DDIM (快速，默认)

```bash
python sample/sample_long.py --config configs/sample_stage3.config
```

- **速度:** 快 (比 DDPM 快 10-50 倍)
- **质量:** 良好 (5-10% 质量损失)
- **使用 skip_steps:** 100-200 获得良好的速度/质量平衡

### DDPM (高质量)

```bash
python sample/sample_long.py --config configs/sample_stage3.config --ddpm --skip_steps 0
```

- **速度:** 慢 (完整扩散过程)
- **质量:** 最佳
- **使用 skip_steps:** 0 获得最高质量

---

## 示例

### 示例 1: 快速批量采样 (L=1024)

```bash
python sample/sample_long.py \
    --config configs/sample_stage3.config \
    --num_samples 50 \
    --length 1024 \
    --batch_size 4 \
    --skip_steps 100
```

### 示例 2: 高质量单样本 (L=1024)

```bash
python sample/sample_long.py \
    --config configs/sample_stage3.config \
    --num_samples 1 \
    --length 1024 \
    --ddpm \
    --skip_steps 0 \
    --save_pdb
```

### 示例 3: 超长序列采样 (L=4096)

```bash
python sample/sample_ultralong.py \
    --config configs/sample_stage3v2.config \
    --num_samples 10 \
    --length 4096 \
    --save_pdb
```

### 示例 4: 极长序列闪存模式 (L=16384)

```bash
python sample/sample_ultralong.py \
    --config configs/sample_stage5.config \
    --length 16384 \
    --flash_mode \
    --save_fragments \
    --clear_cache_steps 5
```

### 示例 5: 温度扫描

```bash
for temp in 0.8 0.9 1.0 1.1 1.2; do
    python sample/sample_long.py \
        --config configs/sample_stage3.config \
        --temperature $temp \
        --output_dir samples/temp_$temp
done
```

---

## 输出文件

### 标准输出 (`.npy`)

所有样本保存为 NumPy 数组：
```
samples/
├── sample_00000.npy
├── sample_00001.npy
└── ...
```

### PDB 输出 (使用 `--save_pdb`)

```
samples/
├── sample_00000.npy
├── sample_00000.pdb
└── ...
```

### 轨迹输出 (使用 `--save_trajectory`)

```
samples/
├── sample_00000.npy
├── trajectory_00000.npy  # 完整轨迹
└── ...
```

### 片段输出 (使用 `--save_fragments`)

```
samples/
├── sample_00000.npy
└── fragments/
    ├── sample_00000_step_0000.npy
    ├── sample_00000_step_0001.npy
    └── ...
```

---

## 性能优化TIPs

### 速度优化

1. **使用 DDIM:** 比 DDPM 快 10-50 倍
2. **增加 skip_steps:** 100-200 获得良好平衡
3. **启用混合精度:** 2-3 倍加速
4. **批量采样:** 并行处理多个样本
5. **降低温度:** 稍微加快收敛

### 质量优化

1. **使用 DDPM:** 最高质量
2. **设置 skip_steps=0:** 使用所有扩散步数
3. **降低温度:** 更加保守 (0.8-0.9)
4. **标准温度:** 平衡 (1.0)
5. **提高温度:** 更加多样 (1.1-1.2)

### 内存优化

1. **使用 flash_mode:** 适用于序列 L>8192
2. **启用 save_fragments:** 而非完整轨迹
3. **更频繁清理缓存:** --clear_cache_steps 5-10
4. **禁用轨迹保存:** 不使用 --save_trajectory
5. **一次采样一个:** 适用于超长序列

---

## 可能的故障排除

### 内存不足 (OOM)

1. 对超长序列使用 `--flash_mode`
2. 启用 `--save_fragments` 而非 `--save_trajectory`
3. 减小 `--clear_cache_steps` 更频繁清理缓存
4. 一次采样一个 (不使用批量)
5. 减小序列长度

### 采样缓慢

1. 使用 DDIM 而非 DDPM
2. 增加 `--skip_steps` (尝试 100-200)
3. 在配置中启用混合精度
4. 对多个样本使用批量采样
5. 降低分辨率/质量要求

### 质量差

1. 使用 DDPM 且 `--skip_steps 0`
2. 调整温度 (尝试 0.9-1.0)
3. 检查模型检查点质量
4. 确保模型在相似长度上训练过
5. 尝试不同的随机种子

### 多样性低

1. 提高温度 (1.1-1.2)
2. 使用不同的随机种子
3. 检查模型训练收敛情况
4. 尝试 DDPM 而非 DDIM

---

## 另请参阅

- [采样配置文档](../configs/README.md) - 详细配置说明
- [训练脚本](../train/README.md) - 如何训练模型
- [长序列指南](../docs/LONG_SEQUENCE_README.md) - 技术细节
