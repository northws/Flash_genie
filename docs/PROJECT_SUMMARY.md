# 🎯 Genie 长序列扩展项目总结

> [!CAUTION]
>
> GitHub网页上**部分数学公式无法正常显示**，如需查看只能到本地markdown编辑器中查看

## 📊 项目结果

### 当前实现

| 维度 | 说明 |
|------|------|
| **Flash-IPA 集成(Stage 1)** | ✅ 已实现 |
| **mHC 集成(Stage 1)** | ✅ 已实现 |
| **Pair Features(Stage 1)** | ✅ 已实现， V2修复数学问题，保留几何信息 |
| **Triangle Ops (Stage 2)** | ✅ 因子化三角操作 |
| **Training (Stage 3)** | ✅ 渐进式训练+混合精度 |
| **Sparse Pairs (Stage 3 V2)** | ✅ k-NN稀疏对 |
| **Axial Attention (Stage 4)** | ✅ 计算效率优化 |
| **Model Compression (Stage 4)** | ✅ 参数共享 |
| **Distributed Training (Stage 5)** | ✅ 多GPU支持 |
| **训练稳定性** | ✅ mHC + Progressive Training |
| **文档完整性** | ✅ 完整的文档和测试 |

## v1 + v1.2: Factorized Pair Features

 `factorized_pair_features.py` 模块核心功能的数学逻辑描述如下

该模块的核心思想是将 $O(L^2)$ 的配对张量 $\mathbf{P}$ 分解为两个秩为 $R$ 的低秩因子 $\mathbf{F}_L$ 和 $\mathbf{F}_R$，从而将内存复杂度从 $O(L^2 \cdot C)$ 降低到 $O(L \cdot R \cdot C)$。

### 1. 核心分解公式 (Core Factorization)

完整的配对特征 $\mathbf{p}_{ij} \in \mathbb{R}^C$ 被近似为两个因子张量的收缩。

设：

- $L$ 为序列长度
- $R$ 为分解的秩 (Rank)
- $C$ 为特征维度

重构公式（对应代码中的 `reconstruct_pair` 和 `forward` 的逻辑）：

$$\mathbf{p}_{ij} = \sum_{r=1}^{R} \left( \mathbf{f}_{L, i, r} \odot \mathbf{f}_{R, j, r} \right)$$

其中：

- $\mathbf{f}_{L, i, r} \in \mathbb{R}^C$ 是左因子在位置 $i$、秩 $r$ 的特征。
- $\mathbf{f}_{R, j, r} \in \mathbb{R}^C$ 是右因子在位置 $j$、秩 $r$ 的特征。
- $\odot$ 表示沿特征维度 $C$ 的逐元素乘法 (Hadamard product)。

------

### 2. 因子生成过程 (Factor Generation)

因子 $\mathbf{F}_L$ 和 $\mathbf{F}_R$ 由三部分组成：单序列投影、相对位置编码和模板特征。

$$\mathbf{f}_{L, i, r} = \mathbf{f}_{L, i, r}^{(s)} + \mathbf{f}_{L, i, r}^{(\text{rel})} + \mathbf{f}_{L, i, r}^{(\text{tmpl})}$$

$$\mathbf{f}_{R, j, r} = \mathbf{f}_{R, j, r}^{(s)} + \mathbf{f}_{R, j, r}^{(\text{rel})} + \mathbf{f}_{R, j, r}^{(\text{tmpl})}$$

#### 2.1 单序列特征投影 (Single Feature Projection)

给定单序列特征 $\mathbf{s}_i \in \mathbb{R}^{C_s}$：

$$\mathbf{f}_{L, i, r}^{(s)} = \mathbf{W}_{L}^{(s)} \mathbf{s}_i, \quad \mathbf{f}_{R, j, r}^{(s)} = \mathbf{W}_{R}^{(s)} \mathbf{s}_j$$

#### 2.2 因子化反对称相对位置编码 (Factorized Antisymmetric RelPos)

代码类：`FactorizedRelPos`

为了在因子化形式中保留相对位置 $i-j$ 的反对称性质（因为 $i-j \neq j-i$），引入了反对称偏置 $\mathbf{b}_{rel}$。

1. **位置嵌入组合**:

   $$\mathbf{h}_i = \left[ \text{Emb}_{\text{abs}}(i) \, ; \, \text{Emb}_{\text{bin}}\left( \text{clamp}(i - \frac{L}{2}) \right) \right]$$

2. **生成因子**:

   $$\mathbf{f}_{L, i, r}^{(\text{rel})} = \mathbf{W}_{L}^{(\text{pos})} \mathbf{h}_i + \mathbf{b}_{\text{rel}, r}$$

   $$\mathbf{f}_{R, j, r}^{(\text{rel})} = \mathbf{W}_{R}^{(\text{pos})} \mathbf{h}_j - \mathbf{b}_{\text{rel}, r}$$

   *注：当计算交叉项时，$(+b)(\dots) + (\dots)(-b)$ 的结构有助于打破反对称性。*

#### 2.3 因子化模板特征 (Factorized Template - SVD Style)

代码类：`FactorizedTemplate`

给定模板张量 $\mathbf{T} \in \mathbb{R}^{L \times L \times C_t}$，使用注意力池化聚合行和列信息。

1. **特征提取**:

   - 对角线: $\mathbf{v}_{\text{diag}, i} = \mathbf{T}_{ii}$
   - 行聚合: $\mathbf{v}_{\text{row}, i} = \sum_k \text{Softmax}(\mathbf{w}_q^T \mathbf{T}_{ik}) \cdot \mathbf{T}_{ik}$
   - 列聚合: $\mathbf{v}_{\text{col}, i} = \sum_k \text{Softmax}(\mathbf{w}_q^T \mathbf{T}_{ki}) \cdot \mathbf{T}_{ki}$

2. **非对称组合 (Asymmetric Combination)**:

   $$\mathbf{u}_{L, i} = \text{LayerNorm}([\mathbf{v}_{\text{diag}, i}, \mathbf{v}_{\text{row}, i}, \mathbf{v}_{\text{col}, i}])$$

   $$\mathbf{u}_{R, j} = \text{LayerNorm}([\mathbf{v}_{\text{diag}, j}, \mathbf{v}_{\text{col}, j}, \mathbf{v}_{\text{row}, j}]) \quad (\text{注意行列顺序交换})$$

3. **SVD风格投影**:

   $$\mathbf{f}_{L, i, r}^{(\text{tmpl})} = (\mathbf{W}_U \mathbf{u}_{L, i}) \cdot \sigma_r$$

   $$\mathbf{f}_{R, j, r}^{(\text{tmpl})} = \mathbf{W}_V \mathbf{u}_{R, j}$$

   其中 $\sigma_r$ 是可学习的奇异值标量，用于加权不同秩的重要性。

------

### 3. 因子化配对精炼 (Factorized Pair Refinement)

代码类：`FactorizedPairRefinementLayer`

该模块模拟三角更新（Triangular Update），但不构建 $L^2$ 矩阵，而是通过因子间的交叉注意力实现。

对于每一层 $l$ 和每一个秩 $r$：

$$\mathbf{\hat{f}}_{L, i, r} = \text{LayerNorm}(\mathbf{f}_{L, i, r})$$

$$\mathbf{\hat{f}}_{R, j, r} = \text{LayerNorm}(\mathbf{f}_{R, j, r})$$

**交叉注意力 (Cross-Factor Interaction):**

$$\mathbf{z}_{L, i, r} = \text{Attention}(Q=\mathbf{\hat{f}}_{L, \cdot, r}, K=\mathbf{\hat{f}}_{R, \cdot, r}, V=\mathbf{\hat{f}}_{R, \cdot, r})_i$$

$$\mathbf{z}_{R, j, r} = \text{Attention}(Q=\mathbf{\hat{f}}_{R, \cdot, r}, K=\mathbf{\hat{f}}_{L, \cdot, r}, V=\mathbf{\hat{f}}_{L, \cdot, r})_j$$

**门控更新 (Gated Update):**

$$\mathbf{f}_{L, i, r} \leftarrow \mathbf{f}_{L, i, r} + \sigma(\mathbf{W}_{g1}[\mathbf{f}_{L, i, r}; \mathbf{z}_{L, i, r}]) \odot \mathbf{z}_{L, i, r}$$

最后，通过秩混合层 (Rank Mixing) 允许不同秩之间交换信息：

$$\mathbf{F}_L \leftarrow \text{Linear}_{\text{rank}}(\mathbf{F}_L), \quad \mathbf{F}_R \leftarrow \text{Linear}_{\text{rank}}(\mathbf{F}_R)$$

## Stage 2 更新 (2026-01-11)

### Triangle Operations 优化

Stage 2 实现了类似AlphaFold2 Evoformer 的 Triangle Operations，但此处通过因子化，避免 O(L³) 内存开销。

#### **问题**: 原始三角乘法更新需要 O(L³) 内存，三角注意力需要 O(L²) 注意力矩阵

 `factorized_triangle_ops.py` 模块实现的因子化三角操作的数学逻辑描述如下，该模块主要针对 AlphaFold2 中内存消耗巨大的“三角更新”和“三角注意力”进行了低秩因子化（Factorized）和分块（Chunked）优化。

### 符号定义

- $L$: 序列长度 (Sequence Length)
- $R$: 秩 (Rank)
- $C$: 通道数 (Channels/Feature dimension)
- $H$: 注意力头数 (Number of Heads)
- $d_h$: 每个头的维度 ($C_{hidden} / H$)
- $\sigma$: Sigmoid 激活函数
- $\text{LN}$: Layer Normalization

------

### 1. 因子化三角乘法更新 (Factorized Triangle Multiplicative Update)

该模块对应类 `FactorizedTriangleMultiplicativeUpdate`。它将原本 $O(L^3)$ 的复杂度降低为 $O(L^2 \times R)$。

核心思想：

原始的三角乘法更新计算公式为：

$$z_{ij} \leftarrow \sum_{k} a_{ik} \odot b_{kj}$$

代码中通过维护低秩因子 $Z_{left}, Z_{right} \in \mathbb{R}^{L \times R \times C}$ 来避免构建完整的 $L \times L$ 张量。

#### 1.1 预处理与投影

首先对输入因子进行归一化和门控线性投影：

$$\begin{aligned} Z'_{left} &= \text{LN}(Z_{left}) \\ Z'_{right} &= \text{LN}(Z_{right}) \end{aligned}$$

生成中间变量 $A$ 和 $B$（包含门控机制）：

$$\begin{aligned} A_{left} &= \text{Linear}_{a}(Z'_{left}) \odot \sigma(\text{Linear}_{g\_a}(Z'_{left})) \\ B_{right} &= \text{Linear}_{b}(Z'_{right}) \odot \sigma(\text{Linear}_{g\_b}(Z'_{right})) \end{aligned}$$

#### 1.2 跨秩混合 (Rank Mixing)

为了允许不同秩之间的信息交互，对秩维度 $R$ 进行线性变换：

$$\begin{aligned} \tilde{A} &= A_{left} W_{mix\_a}^T \\ \tilde{B} &= B_{right} W_{mix\_b}^T \end{aligned}$$

其中 $W_{mix} \in \mathbb{R}^{R \times R}$。

#### 1.3 因子化聚合 (Factorized Aggregation)

这是优化的核心。以 **出边 (Outgoing)** 为例，代码近似计算了所有中间节点 $k$ 的聚合信息：

$$\bar{B} = \frac{1}{L} \sum_{k=1}^{L} \tilde{B}_{k}$$

这里 $\bar{B} \in \mathbb{R}^{1 \times R \times C_{hidden}}$ 是对序列维度的均值聚合。

更新左因子的公式为：

$$U_{left} = \tilde{A} \odot \bar{B}$$

(注：对于入边 Incoming，操作是对称的：聚合 $\tilde{A}$ 并更新 $\tilde{B}$)。

#### 1.4 输出投影

最终输出经过归一化、投影、门控和 Dropout：

$$\begin{aligned} O_{left} &= \text{Linear}_{out}(\text{LN}(U_{left})) \\ Gate_{left} &= \sigma(\text{Linear}_{gate}(Z_{left})) \\ Z_{out} &= \text{Dropout}(O_{left} \odot Gate_{left}) \end{aligned}$$

------

### 2. 分块三角注意力 (Chunked Triangle Attention)

该模块对应类 `ChunkedTriangleAttention`。它解决了标准注意力机制中 $O(L^2)$ 的显存占用问题。

#### 2.1 特征聚合 (Feature Aggregation)

由于输入是因子化的形式，首先将其聚合为伪配对特征（Pseudo-pair features）以计算 Query, Key, Value：

$$Z_{pair} = \text{LN}\left(\sum_{r=1}^{R} Z_{left, r} + \sum_{r=1}^{R} Z_{right, r}\right)$$

其中 $Z_{pair} \in \mathbb{R}^{L \times C}$。

#### 2.2 Q, K, V 投影

$$\begin{aligned} Q &= Z_{pair} W_Q \\ K &= Z_{pair} W_K \\ V &= Z_{pair} W_V \end{aligned}$$

这三个矩阵被重塑为 $\mathbb{R}^{L \times H \times d_h}$。

#### 2.3 偏置计算 (Bias Calculation)

三角注意力通常包含从配对表示中导出的偏置项 $b_{ij}$。代码中通过左因子导出：

$$\text{BiasFeatures} = \sum_{r=1}^{R} Z_{left, r}$$

$$B_{bias} = \text{Linear}_{bias}(\text{BiasFeatures})$$

$B_{bias}$ 广播后形状为 $\mathbb{R}^{H \times L \times L}$。

#### 2.4 分块注意力机制 (Chunked Attention Mechanism)

为了节省内存，注意力分数不一次性计算完整的 $L \times L$ 矩阵，而是沿 Query 的序列维度切分为大小为 $S$ ($chunk\_size$) 的块。

对于第 $m$ 个块（索引范围 $[i, i+S]$）：

$$\begin{aligned} \text{Scores}^{(m)} &= \frac{Q_{[i:i+S]} K^T}{\sqrt{d_h}} + B_{bias[i:i+S]} \\ \text{Attn}^{(m)} &= \text{Softmax}(\text{Scores}^{(m)}) \\ O^{(m)} &= \text{Attn}^{(m)} V \end{aligned}$$

最终将所有块的输出拼接：

$$O = \text{Concat}(O^{(1)}, O^{(2)}, \dots)$$

#### 2.5 输出分发

注意力输出经过投影和门控后，重新分配回因子化表示（这里简化为均分给秩维度）：



$$\begin{aligned} Y &= \text{Linear}_{out}(O) \odot \sigma(\text{Linear}_{gate}(Z_{pair})) \\ Z_{left}^{new} &= \frac{1}{R} \cdot Y \quad (\text{Broadcast over rank}) \end{aligned}$$


### 2. Factorized Pair Transform Network 

 `factorized_pair_transform.py` 实现了一个**因子化配对变换网络 (Factorized Pair Transform Network)** ，以下是**因子化配对变换网络 (Factorized Pair Transform Network)** 的数学逻辑。

该模块的主要功能是将 AlphaFold2 中的 `PairTransformNet`（包含三角更新和三角注意力）适配为因子化形式。核心设计模式是**“计算因子更新 $\to$ 聚合为全局更新 $\to$ 均匀分配回因子”**，这种机制在保持低秩结构的同时，促进了不同秩之间的信息交换。

------

### 1. 总体架构 (General Architecture)

网络由 $N$ 个堆叠的 **因子化配对变换层 (FactorizedPairTransformLayer)** 组成。

**输入/输出状态**:

- $\mathbf{F}_L^{(l)} \in \mathbb{R}^{L \times R \times C}$: 第 $l$ 层的左因子张量。
- $\mathbf{F}_R^{(l)} \in \mathbb{R}^{L \times R \times C}$: 第 $l$ 层的右因子张量。
- $R$: 因子化秩 (Rank)。

每一层的处理流程包含 5 个顺序子模块：

1. 因子化三角乘法更新 (出边)
2. 因子化三角乘法更新 (入边)
3. 分块三角注意力 (起始节点)
4. 分块三角注意力 (结束节点)
5. 配对过渡 (Pair Transition)

------

### 2. 通用更新机制 (The Update-Aggregate-Distribute Mechanism)

代码中所有子模块（除了 Pair Transition）都遵循相同的残差连接模式。为了数学上的简洁，定义一个通用算子 $\text{ApplyBlock}$：

**设子模块操作为 $\Phi(\mathbf{F}_L, \mathbf{F}_R)$，返回更新量 $(\Delta \mathbf{F}_L, \Delta \mathbf{F}_R)$。**

**聚合与分配步骤**:

1. **执行操作**:

   $$(\Delta \mathbf{F}_L, \Delta \mathbf{F}_R) = \Phi(\mathbf{F}_L, \mathbf{F}_R)$$

2. 秩聚合 (Rank Aggregation):

   将左、右因子的更新量沿秩维度求和，得到全局序列更新量 $\Delta \mathbf{z} \in \mathbb{R}^{L \times C}$：

   $$\Delta \mathbf{z} = \sum_{r=1}^{R} \Delta \mathbf{F}_{L, \cdot, r, \cdot} + \sum_{r=1}^{R} \Delta \mathbf{F}_{R, \cdot, r, \cdot}$$

3. **Dropout**:

   $$\Delta \mathbf{z}' = \text{Dropout}(\Delta \mathbf{z})$$

4. 均匀分配残差 (Distribute Residual):

   将聚合后的更新量均匀分配回所有秩，实现信息同步：

   $$\mathbf{F}_{L, \cdot, r, \cdot} \leftarrow \mathbf{F}_{L, \cdot, r, \cdot} + \frac{1}{R} \Delta \mathbf{z}'$$

   $$\mathbf{F}_{R, \cdot, r, \cdot} \leftarrow \mathbf{F}_{R, \cdot, r, \cdot} + \frac{1}{R} \Delta \mathbf{z}'$$

------

### 3. 子模块逻辑详解

#### 3.1 因子化三角乘法更新 (Factorized Triangle Multiplicative Update)

利用前一个文件定义的 `FactorizedTriangleMultiplication` 模块。

**出边更新 (Outgoing):**

$$\Phi_{\text{mul\_out}}(\mathbf{F}_L, \mathbf{F}_R) \implies \text{ApplyBlock}$$

- 注：在 `FactorizedTriangleMultiplicationOutgoing` 中，通常 $\Delta \mathbf{F}_R = \mathbf{0}$，因此聚合步骤主要收集 $\Delta \mathbf{F}_L$ 的信息。

**入边更新 (Incoming):**

$$\Phi_{\text{mul\_in}}(\mathbf{F}_L, \mathbf{F}_R) \implies \text{ApplyBlock}$$

#### 3.2 分块三角注意力 (Chunked Triangle Attention)

利用前一个文件定义的 `ChunkedTriangleAttention` 模块。

**起始节点注意力 (Starting Node):**

$$\Phi_{\text{att\_start}}(\mathbf{F}_L, \mathbf{F}_R) \implies \text{ApplyBlock}$$

- 此处使用 **Row-wise Dropout**。

**结束节点注意力 (Ending Node):**

$$\Phi_{\text{att\_end}}(\mathbf{F}_L, \mathbf{F}_R) \implies \text{ApplyBlock}$$

- 此处使用 **Column-wise Dropout**。

#### 3.3 因子化配对过渡 (Factorized Pair Transition)

标准的 Pair Transition 是一个 MLP，作用于 $L \times L \times C$ 张量的每个元素。为了节省内存，此实现对聚合后的 $L \times C$ 特征进行操作。

1. **特征聚合**:

   $$\mathbf{z}_{\text{agg}} = \sum_{r=1}^R \mathbf{F}_{L, \cdot, r, \cdot} + \sum_{r=1}^R \mathbf{F}_{R, \cdot, r, \cdot}$$

2. 过渡层 (Transition):

   应用两层 MLP (Linear $\to$ ReLU $\to$ Linear)：

   $$\mathbf{z}_{\text{trans}} = \text{Linear}_2(\text{ReLU}(\text{Linear}_1(\text{LayerNorm}(\mathbf{z}_{\text{agg}}))))$$

   *注：代码复用了标准的 `PairTransition` 模块，但输入被 reshape 为模拟形状以适应接口，实际计算是逐位置 (position-wise) 的。*

3. **残差分配**:

   $$\mathbf{F}_{L, \cdot, r, \cdot} \leftarrow \mathbf{F}_{L, \cdot, r, \cdot} + \frac{1}{R} \mathbf{z}_{\text{trans}}$$

   $$\mathbf{F}_{R, \cdot, r, \cdot} \leftarrow \mathbf{F}_{R, \cdot, r, \cdot} + \frac{1}{R} \mathbf{z}_{\text{trans}}$$

### 4. 掩码处理 (Masking)

在每一层的最后，应用序列掩码以确保无效位置保持为 0：

$$\mathbf{F}_{L, i, r} \leftarrow \mathbf{F}_{L, i, r} \cdot m_i$$

$$\mathbf{F}_{R, i, r} \leftarrow \mathbf{F}_{R, i, r} \cdot m_i$$

其中 $m_i \in \{0, 1\}$ 是序列掩码。

### 总结：内存复杂度对比

对于层数 $N$，序列长度 $L$，通道数 $C$，秩 $R$：

- **标准 PairTransform**: $O(N \cdot L^2 \cdot C)$
- **因子化 PairTransform**: $O(N \cdot L \cdot R \cdot C)$

## Stage 3 更新 (2026-01-12)

### 训练效率优化

Stage 3 实现了完整的训练优化流程，大幅提升训练速度和稳定性。

 `progressive_training.py`模块主要实现了两个关键的训练优化机制：**渐进式训练调度（Curriculum Learning），分块损失计算（Chunked Loss Computation）**。

以下是各个模块对应的数学公式说明。

### 符号定义

- $t$: 当前训练步数 (Current Step)
- $L_{curr}$: 当前训练使用的序列长度
- $L_{min}, L_{max}$: 最小和最大序列长度
- $T_{warmup}$: 预热步数
- $T_{growth}$: 增长阶段步数
- $\mathbf{x}_i, \hat{\mathbf{x}}_i \in \mathbb{R}^3$: 第 $i$ 个残基的真实坐标和预测坐标
- $M_{ij} \in \{0, 1\}$: 掩码矩阵，当残基 $i$ 和 $j$ 都存在时为 1

------

### 1. 渐进式训练调度器 (Progressive Training Scheduler)

该模块通过课程学习（Curriculum Learning）动态调整序列裁剪长度，使模型先学习短序列的局部特征，再逐步过渡到长序列的全局特征。

#### 1.1 进度计算 (Progress Calculation)

定义归一化的增长进度 $p \in [0, 1]$：

$$p = \text{clamp}\left(\frac{t - T_{warmup}}{T_{growth}}, 0, 1\right)$$

#### 1.2 增长策略 (Growth Schedule)

对应 `growth_schedule` 参数，插值系数 $\alpha$ 的计算方式如下：

- 线性 (Linear):

  $$\alpha = p$$

- 余弦 (Cosine):

  $$\alpha = \frac{1 - \cos(\pi \cdot p)}{2}$$

- 指数 (Exponential):

  $$\alpha = p^2$$

#### 1.3 当前序列长度 (Current Max Length)

$$L_{curr}(t) = \begin{cases} L_{min} & \text{if } t < T_{warmup} \\ \lfloor L_{min} + \alpha \cdot (L_{max} - L_{min}) \rfloor & \text{if } T_{warmup} \le t < T_{total} \\ L_{max} & \text{otherwise} \end{cases}$$

------

### 2. 分块损失计算 (Chunked Loss Computation)

为了避免构建形状为 $B \times L \times L$ 的完整距离矩阵（显存消耗 $O(L^2)$），代码将 $L$ 维度切分为大小为 $S$ 的块（Chunk），将空间复杂度降低为 $O(S \cdot L)$。

#### 2.1 距离计算基础

对于任意两点 $i, j$，其欧几里得距离为：

$$d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|_2, \quad \hat{d}_{ij} = \|\hat{\mathbf{x}}_i - \hat{\mathbf{x}}_j\|_2$$

在分块模式下，外层循环遍历块索引 $k$，内层计算块内残基 $i \in [kS, (k+1)S]$ 与全序列残基 $j \in [1, L]$ 的距离。

#### 2.2 损失 

> [!WARNING]
>
> *虽然代码函数名为 `compute_fape_loss_chunked`，但其实我现在的逻辑仍然是基于**标量距离矩阵的误差**（Distance Deviation），而非 AlphaFold2 原文中基于局部坐标系的 Frame Aligned Point Error（**可能未来会考虑引入**）。以下公式对应代码的实际逻辑。*

对于每个块，计算距离误差并进行截断（Clamping）：

$$E_{ij} = \min\left( \left| \hat{d}_{ij} - d_{ij} \right|, \tau \right)$$

其中 $\tau$ 是截断阈值 (clamp_distance，默认 10.0)。

总损失为加权平均：

$$\mathcal{L}_{dist} = \frac{\sum_{k} \sum_{i \in \text{chunk}_k} \sum_{j=1}^L M_{ij} E_{ij}}{\sum_{i,j} M_{ij} + \epsilon}$$

#### 2.3 dRMSD 损失 (分块版)

距离均方根偏差（Distance RMSD）用于衡量内部几何结构的相似性，不依赖于全局叠加。

$$SquaredError_{ij} = (\hat{d}_{ij} - d_{ij})^2$$

分块累积计算后：

$$\mathcal{L}_{dRMSD} = \sqrt{ \frac{\sum_{k} \sum_{i \in \text{chunk}_k} \sum_{j=1}^L M_{ij} (\hat{d}_{ij} - d_{ij})^2}{\sum_{i,j} M_{ij} + \epsilon} }$$

### 3. 内存与效率分析

通过分块，距离矩阵的显存占用从二次方降低为线性：

$$Memory_{standard} \propto B \cdot L^2$$

$$Memory_{chunked} \propto B \cdot S \cdot L$$

当 $L=1024, S=64$ 时，内存占用减少约 **16 倍**，这允许在有限显存的 GPU 上训练更长的蛋白质序列。

## Stage 3 V2 更新 (2026-01-13)

Stage 3 V2 通过Sparse k-NN Pairs，支持更长序列

 `sparse_pairs.py`模块实现了一种**稀疏 $k$-最近邻 ($k$-NN) 配对选择机制**。其核心目的是将长序列蛋白质建模中的配对特征从稠密的 $O(L^2)$ 降低到稀疏的 $O(L \cdot k)$。

以下是各个选择策略的数学逻辑：

### 符号定义

- $L$: 序列长度
- $k$: 每个残基选择的邻居数量
- $\mathbf{x}_i \in \mathbb{R}^3$: 第 $i$ 个残基的空间坐标 (通常是 $C_\alpha$ 或 $C_\beta$)
- $\mathcal{N}_i$: 第 $i$ 个残基的邻居索引集合
- $w$: 局部窗口大小 (`local_window`)

------

### 1. 基于坐标的选择 (Coordinate-based Selection)

这是通过计算 3D 空间中的欧几里得距离来寻找最近的邻居。

#### 1.1 距离矩阵计算

首先计算所有残基对之间的欧几里得距离 $D \in \mathbb{R}^{L \times L}$：

$$d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|_2 = \sqrt{\sum_{d=1}^3 (x_{i,d} - x_{j,d})^2}$$

#### 1.2 掩码处理 (Masking)

为了处理 padding 或缺失的残基，应用掩码 $M \in \{0, 1\}^L$：

$$d'_{ij} = \begin{cases} d_{ij} & \text{if } M_i \cdot M_j = 1 \\ \infty & \text{if } M_i \cdot M_j = 0 \end{cases}$$

#### 1.3 Top-k 选择

对于每个残基 $i$，选择距离最小的 $k$ 个索引：

$$\mathcal{N}_i^{coord} = \underset{j}{\text{argtopk}}(d'_{i, :}, k, \text{largest=False})$$

这对应代码中的 `torch.topk(dist, k, largest=False)`。

------

### 2. 基于序列的选择 (Sequence-based Selection)

这种方法仅基于残基在氨基酸序列中的索引距离 $|i-j|$ 来选择邻居，捕捉一级序列上的局部性。

对于残基 $i$，选择其前 $k/2$ 个和后 $k/2$ 个残基：

$$\mathcal{N}_i^{seq} = \{j \mid \max(0, i - \lfloor \frac{k}{2} \rfloor) \le j \le \min(L-1, i + \lfloor \frac{k}{2} \rfloor), j \neq i \}$$

如果边界处的邻居数量不足 $k$，代码逻辑会用 $i$ 自身进行填充以保持张量形状固定。

------

### 3. 混合策略 (Hybrid Strategy)

混合策略结合了空间几何信息和序列局部信息。它将 $k$ 分配为两部分：$k_{coord} = \lfloor k/2 \rfloor$ 和 $k_{seq} = k - k_{coord}$。

$$\mathcal{N}_i^{hybrid} = \mathcal{N}_i^{coord}(k_{coord}) \cup \mathcal{N}_i^{seq}(k_{seq})$$

最终的邻居张量是两者的拼接（Concatenation）：

$$\text{Indices}_{hybrid} = \text{Concat}(\text{Indices}_{coord}, \text{Indices}_{seq})$$

------

### 4. 强制局部配对 (Mandatory Local Pairs)

为了保证模型始终能看到局部的二级结构信息，代码提供了一个选项 `include_all_local`，强制包含窗口 $w$ 内的所有残基。

局部索引集合：

$$\mathcal{N}_i^{local} = \{j \mid |i - j| \le w \}$$

最终集合为 $k$-NN 集合与局部集合的并集：

$$\mathcal{N}_i^{final} = \mathcal{N}_i^{knn} \cup \mathcal{N}_i^{local}$$

由于并集操作会导致每个残基的邻居数量不一致，代码实现中通常会取最大长度并进行 Padding。

------

### 5. 复杂度与内存分析

通过这种稀疏化，内存消耗从序列长度的平方级降低到线性级：

- 密集配对 (Dense):

  $$\text{Memory} \propto O(L^2 \cdot C)$$

  对于 $L=4096$，这是不可行的。

- 稀疏配对 (Sparse):

  $$\text{Memory} \propto O(L \cdot k \cdot C)$$

  对于 $L=4096, k=32$，内存减少约 120倍。

## Stage 4 更新 (2026-01-14)

Stage 4 实现了计算效率优化和模型压缩技术。

### 1. Axial Attention 

 `axial_attention.py`，模块实现了**轴向注意力 (Axial Attention)**，旨在将 AlphaFold2 中计算复杂度为 $O(L^3)$ 的三角注意力降低为 $O(L^2)$。此外，还实现了一个结合了低秩因子化的版本。

以下是核心模块的数学公式说明。

#### 符号定义

- $X \in \mathbb{R}^{B \times L \times L \times C}$: 输入的配对张量 (Batch, Row, Column, Channel)
- $L$: 序列长度
- $C$: 通道维度
- $H$: 注意力头数
- $\mathcal{A}$: 注意力函数 (Attention)
- $\sigma$: Sigmoid 激活函数

------

#### 1. 轴向注意力 (Axial Attention)

轴向注意力将全二维注意力分解为两个顺序执行的一维注意力：**行注意力 (Row Attention)** 和 **列注意力 (Column Attention)**。

##### 1.1 注意力通用形式

对于输入序列 $Y$（形状为 $N \times S \times C$），多头注意力计算如下：

$$\begin{aligned} Q &= Y W_Q, \quad K = Y W_K, \quad V = Y W_V \\ \text{Scores} &= \frac{Q K^T}{\sqrt{d_k}} \\ \text{Attn} &= \text{Softmax}(\text{Scores} + M) \\ O &= \text{Attn} \cdot V \end{aligned}$$

最终输出经过线性投影和门控：

$$Y_{out} = (O W_O) \odot \sigma(Y W_G)$$

##### 1.2 行注意力 (Row-wise Attention)

对每一行 $i$，在所有列 $j \in [1, L]$ 上进行注意力计算。

将 $X$ 视为 $B \cdot L$ 个长度为 $L$ 的序列：

$$X_{row\_view} \in \mathbb{R}^{(B \cdot L) \times L \times C}$$

计算：

$$X^{(1)} = X + \text{Attention}_{row}(X_{row\_view})$$

其中注意力发生在第 3 维度（列索引 $j$）上。

##### 1.3 列注意力 (Column-wise Attention)

对每一列 $j$，在所有行 $i \in [1, L]$ 上进行注意力计算。

首先转置输入，交换行和列维度：

$$X_{transposed} = (X^{(1)})^T \in \mathbb{R}^{B \times L \times L \times C} \quad (\text{dim } 1 \leftrightarrow 2)$$

将转置后的张量视为 $B \cdot L$ 个长度为 $L$ 的序列：

$$X^{(2)} = X^{(1)} + \left( \text{Attention}_{col}(X_{transposed}) \right)^T$$

其中注意力发生在第 2 维度（行索引 $i$）上。

------

#### 2. 因子化轴向注意力 (Factorized Axial Attention)

该模块尝试在因子化表示 $Z_{left}, Z_{right} \in \mathbb{R}^{L \times R \times C}$ 上应用轴向注意力，通过构建“伪配对”来计算交互，然后投影回因子。

对于每个秩 $r \in [1, R]$：

##### 2.1 伪配对构建 (Pseudo-Pair Construction)

通过广播加法构建临时的 $L \times L$ 特征（代码中显式扩展了维度，实际应用中通常使用分块以节省内存，此处展示逻辑公式）：

$$P^{(r)}_{ij} = Z_{left, i, r} + Z_{right, j, r}$$

其中 $P^{(r)} \in \mathbb{R}^{L \times L \times C}$。

##### 2.2 轴向注意力应用

代码中仅应用了行注意力（Row Attention）作为演示或特定变体：

$$U^{(r)} = \text{Attention}_{row}(P^{(r)})$$

$U^{(r)}$ 包含了更新后的配对信息。

##### 2.3 投影回因子 (Projection Back to Factors)

通过平均池化将 $L \times L$ 信息压缩回 $L$ 维度：

更新左因子 (对列求均值):

$$Z'_{left, i, r} = \frac{1}{L} \sum_{j=1}^L U^{(r)}_{ij}$$

更新右因子 (对行求均值):

$$Z'_{right, j, r} = \frac{1}{L} \sum_{i=1}^L U^{(r)}_{ij}$$

##### 2.4 最终合并

$$Z_{left}^{new} = \text{Concat}([Z'_{left, \cdot, 1}, \dots, Z'_{left, \cdot, R}], \text{dim}=rank)$$

$$Z_{right}^{new} = \text{Concat}([Z'_{right, \cdot, 1}, \dots, Z'_{right, \cdot, R}], \text{dim}=rank)$$

------

#### 3. 复杂度对比

| **方法**             | **显存复杂度**             | **计算复杂度**                |
| -------------------- | -------------------------- | ----------------------------- |
| **标准三角注意力**   | $O(L^2)$                   | $O(L^3)$                      |
| **轴向注意力**       | $O(L^2)$ (分块可达 $O(L)$) | $O(L^2)$                      |
| **因子化轴向注意力** | $O(L \cdot R)$             | $O(L^2 \cdot R)$ (含重建过程) |

对于 $L=2048$，轴向注意力将计算量减少了约 24 倍。

### 2. Advanced Gradient Checkpointing 

自适应梯度检查点策略
```python
class AdaptiveCheckpointManager:
    """
    根据序列长度和可用内存动态调整检查点策略

    短序列 + 充足内存: 不检查点 (速度优先)
    中等序列: 选择性检查点 (平衡)
    长序列 + 紧张内存: 全检查点 (内存优先)
    """
    def get_adaptive_config(seq_len, available_memory_gb):
        if seq_len < 256 and available_memory_gb > 10:
            return CheckpointConfig(enabled=False)  # 无需检查点
        elif seq_len < 512:
            return CheckpointConfig(
                checkpoint_triangles=True,  # 只检查点三角操作
            )
        else:
            return CheckpointConfig(
                checkpoint_structure=True,  # 检查点所有
                checkpoint_pairs=True,
                checkpoint_triangles=True,
            )
```

### 3. Model Compression 

层参数共享 (Universal Transformer)风格

 `model_compression.py`模块实现了一系列**参数高效的模型压缩技术 (Parameter-Efficient Model Compression)**。其核心思想是通过**层共享 (Layer Sharing)** 和 **瓶颈结构 (Bottleneck Architectures)** 来在减少参数量的同时保持网络的深度和表达能力。

以下是各个核心组件的数学逻辑。

#### 符号定义

- $x^{(l)}$: 第 $l$ 层的输入张量。
- $f_\theta(\cdot)$: 参数为 $\theta$ 的神经网络层（通常包含 Attention, MLP, Norm 等）。
- $L$: 网络的总层数。
- $C$: 输入特征维度 (Channels)。
- $C_{bot}$: 瓶颈层维度 ($C / r$)。

------

##### 1. 通用层共享 (Universal Layer Sharing)

该策略对应类 `SharedLayerModule`。这是 **Universal Transformer** 的核心机制。网络在深度方向上应用递归（Recurrence），即每一层复用完全相同的参数 $\theta$。

对于 $l = 1, \dots, L$：

$$x^{(l)} = f_{\theta}(x^{(l-1)})$$

- **参数复杂度**: $O(1 \times |\theta|)$，与网络总深度 $L$ 无关。
- **物理意义**: 可以看作是在时间步上展开的 RNN，但在空间（深度）维度上操作，旨在寻找不动点或逐步细化表示。

------

##### 2. 交替层共享 (Alternating Layer Sharing)

该策略对应类 `AlternatingSharedLayers`。这是 **ALBERT (A Lite BERT)** 中使用的一种变体。它将参数分为两组：奇数层参数 $\theta_{odd}$ 和偶数层参数 $\theta_{even}$。

对于 $l = 0, \dots, L-1$：

$$x^{(l+1)} = \begin{cases} f_{\theta_{even}}(x^{(l)}) & \text{if } l \equiv 0 \pmod 2 \\ f_{\theta_{odd}}(x^{(l)}) & \text{if } l \equiv 1 \pmod 2 \end{cases}$$

- **参数复杂度**: $O(2 \times |\theta|)$。
- **压缩率**: 相比标准网络，压缩率约为 $L/2$。

------

##### 3. 分块层共享 (Block-wise Layer Sharing)

该策略对应类 `BlockSharedLayers`。网络被划分为 $K$ 个块 (Block)，每个块包含 $M$ 个子层 ($L = K \times M$)。块与块之间参数不同，但块内的 $M$ 次迭代共享参数。

令 $\theta_k$ 为第 $k$ 个块的参数。对于第 $k$ 个块中的第 $m$ 次迭代：

$$x^{(k, m+1)} = f_{\theta_k}(x^{(k, m)}), \quad m \in [0, M-1]$$

- **参数复杂度**: $O(K \times |\theta|)$，其中 $K \ll L$。
- 允许网络在不同阶段（如浅层特征 vs 深层语义）拥有不同的处理逻辑，同时在局部保持参数效率。

------

##### 4. 瓶颈层 (Bottleneck Layer)

该策略对应类 `BottleneckLayer`。为了减少计算量和参数量，在执行昂贵的操作（如全连接或注意力）之前，先将维度投影到低维空间。

设输入 $x \in \mathbb{R}^{d_{in}}$，瓶颈比率为 $r$。

1. 降维 (Project Down):

   $$x_{bot} = x W_{down} + b_{down}, \quad W_{down} \in \mathbb{R}^{d_{in} \times (d_{in}/r)}$$

2. 核心操作 (Operation):

   $$h = \text{Operation}(x_{bot})$$

   注意：这里的 Operation 是在低维空间 $d_{in}/r$ 进行的，参数量大幅减少。

3. 升维 (Project Up):

   $$y = h W_{up} + b_{up}, \quad W_{up} \in \mathbb{R}^{(d_{in}/r) \times d_{in}}$$

4. 残差与归一化 (Residual & Norm):

   $$Output = \text{LayerNorm}(x + y)$$

------

##### 5. 深度可分离层 (Depthwise Separable Layer)

该策略对应类 `DepthwiseSeparableLayer`。它将标准卷积分解为两个步骤，常用于轻量级网络（如 MobileNet）。

假设输入 $X \in \mathbb{R}^{C \times L}$，卷积核大小为 $K$。

1. 深度卷积 (Depthwise Conv): 对每个通道独立进行空间滤波。

   $$Y_{c, i} = \sum_{k} W_{c, k}^{depth} \cdot X_{c, i+k}$$

   参数量: $C \times K$。

2. 逐点卷积 (Pointwise Conv): 使用 $1 \times 1$ 卷积混合通道信息。

   $$Z_{j, i} = \sum_{c} W_{j, c}^{point} \cdot Y_{c, i}$$

   参数量: $C_{out} \times C_{in} \times 1$。

**总参数量对比**:

- 标准卷积: $C_{out} \times C_{in} \times K$
- 深度可分离: $C_{in} \times K + C_{out} \times C_{in}$
- 当 $K$ 较大时，参数减少显著。

------

##### 6. 参数压缩率分析

代码中的 `get_compression_ratio` 计算如下：

$$Ratio = \frac{\text{Baseline Parameters}}{\text{Compressed Parameters}} = \frac{L \times |\theta|_{base}}{|\Theta|_{shared}}$$

例如，对于 $L=12$ 层的网络：

- **Universal**: $|\Theta| = 1 \times |\theta| \Rightarrow Ratio = 12\times$
- **Alternating**: $|\Theta| = 2 \times |\theta| \Rightarrow Ratio = 6\times$
- **Block (Size=4)**: $|\Theta| = 3 \times |\theta| \Rightarrow Ratio = 4\times$

这些技术使得在显存受限的情况下（如 `model_compression.py` 注释中提到的 Stage 4），可以训练比常规网络深得多的模型。

## Stage 5 更新 (2026-01-15)

### 系统级优化: 分布式训练

Stage 5 实现了分布式训练支持，允许跨多GPU扩展。

#### 1. Distributed Data Parallel (DDP) 

**功能**: 标准数据并行训练
```python
class DistributedModelWrapper:
```

#### 2. Sequence Tensor Parallelism 

**创新**: 在序列维度上切分
```python
class SequenceTensorParallel:
```

#### 3. Gradient Accumulation 

**功能**: 大批量训练支持
```python
class GradientAccumulator:
```

**效果**:

- 大批量训练成为可能
- 更稳定的梯度

#### 4. Stage 5 综合效果

**分布式训练**:
- 多GPU DDP
- Gradient Accumulation: 大批量训练

---

## 📦 文件清单

### Stage 4-5 新增文件 (4 个核心模块) ** Stage 4-5**

18. **`genie/model/axial_attention.py`** (600+ 行) ** Stage 4**
    - `AxialAttention`: 轴向注意力 (行+列分解)
    - `FactorizedAxialAttention`: 因子化轴向注意力
19. **`genie/training/gradient_checkpointing.py`** (400+ 行) **Stage 4**
    - `CheckpointConfig`: 检查点配置
    - `AdaptiveCheckpointManager`: 自适应检查点管理
    - `CheckpointedSequential`: 检查点序列模块
20. **`genie/model/model_compression.py`** (500+ 行) **Stage 4**
    - `CompressedStructureNet`: 压缩结构网络
    - `SharedLayerModule`: 共享层模块
    - `AlternatingSharedLayers`: 交替共享层
21. **`genie/training/distributed_training.py`** (500+ 行) **Stage 5**
    - `DistributedModelWrapper`: DDP封装
    - `SequenceTensorParallel`: 序列张量并行
    - `GradientAccumulator`: 梯度累积
22. **`test_stage4_5.py`** (500+ 行) **Stage 4-5**
    - 6 个综合测试
    - Stage 4-5 集成测试

### Stage 3 V2 新增文件 (2 个核心模块) **Stage 3 V2**

16. **`genie/model/sparse_pairs.py`** (500+ 行) **Stage 3 V2**
    - `SparseKNNPairSelector`: 稀疏 k-NN 对选择器
    - 三种选择策略: coordinate / sequence / hybrid
    - 支持超长序列 
17. **`test_stage3_v2.py`** (400+ 行) **Stage 3 V2**
    - 4 个综合测试
    - Ultra-long memory scaling

### Stage 3 新增文件 (4 个核心模块) **Stage 3**

12. **`genie/training/progressive_training.py`** (400+ 行) **Stage 3**
    - `ProgressiveTrainingScheduler`: 渐进式训练调度器
    - `ChunkedLossComputation`: 分块损失计算
    - 支持 linear/cosine/exponential 增长曲线
    - FAPE 和 dRMSD 损失支持
13. **`genie/training/mixed_precision.py`** (300+ 行) **Stage 3**
    - `MixedPrecisionTrainer`: 混合精度训练管理器
    - `SelectiveMixedPrecision`: 选择性精度控制
    - FP16/BF16 支持 + 动态损失缩放
    - **收益**: 50% 内存节省 + 2-3x 训练加速
14. **`genie/training/stage3_trainer.py`** (400+ 行) **Stage 3**
    - `Stage3TrainingManager`: 综合训练管理器
    - 集成所有 Stage 3 优化
    - 统一训练接口
    - Checkpoint 支持
15. **`test_stage3_optimizations.py`** (400+ 行) **Stage 3**
    - 5 个综合测试
    - Performance comparison

### Stage 2 新增文件 (3 个核心模块)

9. **`genie/model/factorized_triangle_ops.py`** (500+ 行) **Stage 2**
   - `FactorizedTriangleMultiplicativeUpdate`: 因子化三角乘法更新
   - `FactorizedTriangleMultiplicationOutgoing`: Outgoing 变体
   - `FactorizedTriangleMultiplicationIncoming`: Incoming 变体
   - `ChunkedTriangleAttention`: 分块三角注意力
   - `ChunkedTriangleAttentionStartingNode`: 行方向注意力
   - `ChunkedTriangleAttentionEndingNode`: 列方向注意力
10. **`genie/model/factorized_pair_transform.py`** (300+ 行) **Stage 2**
    - `FactorizedPairTransformLayer`: 单层 pair 转换
    - `FactorizedPairTransformNet`: 多层 pair 转换网络
    - 完整的 Evoformer-style processing
    - 所有操作都在因子化表示上进行
11. **`test_stage2_optimizations.py`** (400+ 行) **Stage 2**
    - 5 个综合测试
    - 内存缩放分析
    - Stage 1 vs Stage 2 对比
    - **`test_stage2_quick.py`**: 快速集成测试

### 核心实现 (Stage 1 + V2 文件)

1. **`genie/model/factorized_pair_features.py`** (560 行)
   - `FactorizedPairFeatureNet`: 主要类
   - `FactorizedRelPos`: 因子化位置编码
   - `FactorizedTemplate`: 因子化模板特征
   - `AdaptiveFactorizationRank`: 动态 rank 调整

2. **`genie/model/long_sequence_denoiser.py`** (400+ 行)
   - `LongSequenceDenoiser`: 集成所有优化
   - 自动配置和内存估算
   - 完整的测试函数

3. **`genie/utils/adaptive_config.py`** (500+ 行)
   - `AdaptiveMHCConfig`: mHC 配置
   - `DynamicBatchSize`: 批次大小计算
   - `AdaptiveFactorizationRank`: Rank 计算
   - `MemoryEstimator`: 内存估算工具

### 文档 (4 个文件)

4. **`EVALUATION_AND_IMPROVEMENTS.md`** (2000+ 行)
   - 完整的评估报告
   - 5 阶段优化路线图
   - 详细的技术分析
   - 代码示例和基准测试
6. **`mhc_code_review_fixes.md`** (之前创建)
   - Bug 修复总结
   - Skip Connection 详细分析
   - Sinkhorn 优化说明

## 🎓 创新点

### 1. End-to-End Factorization
**创新**: 完全避免 pair tensor 实例化
```
传统: s → p[L²] → factorize → factors[L×rank]
     (需要 537 MB)          (需要 1 MB)

创新: s → factors[L×rank] (直接生成)
     (仅需 1 MB, 节省 537x!)
```

### 2. Adaptive Architecture
**创新**: 序列长度感知的模型配置
- 短序列: 高容量 (质量优先)
- 长序列: 低容量 (效率优先)
- 动态平衡: 自动调整

### 3. Memory-First Design
**创新**: 以内存为第一约束（最开始是没有高性能gpu，所以只能改）

- 每个优化都有内存估算
- 配置自动检查和警告
- 提供详细的内存分析工具

## 💡 关键

### 技术互补
- Genie: 核心架构
- Flash-IPA: 内存效率
- AlphaFold2 Evoformer: 因子化三角操作 (Stage 2)
- Curriculum Learning: 渐进式训练 (Stage 3)
- Mixed Precision: 训练加速 (Stage 3)
- Sparse Attention: k-NN 稀疏对选择 (Stage 3 V2)
- Axial Attention: 计算效率优化 (Stage 4) 
- Universal Transformer: 模型压缩 (Stage 4) 
- Distributed Training: 多GPU扩展 (Stage 5) 

---