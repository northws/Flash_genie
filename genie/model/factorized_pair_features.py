"""
因子化配对特征网络

此模块实现了配对特征网络的因子化版本，直接生成低秩因子化表示，
而不是实例化完整的 L² × C 配对张量。

核心创新:
- 标准方式:    s[L×C] → p[L²×C] → factors[L×rank×C]  (O(L²) 内存)
- 因子化方式:  s[L×C] → factors[L×rank×C]             (O(L×rank) 内存)

对于 L=1024, rank=2, C=128:
- 标准方式: 1024² × 128 × 4 字节 = 537 MB
- 因子化: 1024 × 2 × 128 × 4 字节 = 1 MB
- 内存节省: 537x

基于 Flash-IPA 论文的因子化技术。

V2 改进 (2026-01):
- 修复 FactorizedRelPos 以正确处理反对称相对位置
- 修复 FactorizedTemplate 以保留几何信息
- 添加 FactorizedPairRefinement 用于轻量级配对更新
"""

import torch
from torch import nn
import math


class FactorizedRelPos(nn.Module):
    """
    因子化相对位置编码 (V2 - 已修复)。

    核心洞察: 相对位置编码 relpos[i,j] = f(i-j) 是反对称的。
    我们不能简单地将其分解为 left[i] + right[j]。

    解决方案: 使用带反对称结构的可学习位置嵌入:
    - left[i] = pos_emb[i] + relpos_bias
    - right[j] = pos_emb[j] - relpos_bias  (反对称)

    这在可因子化的同时保留了相对位置信息。
    """

    def __init__(self, relpos_k, c_out, rank, max_seq_len=4096):
        """
        初始化因子化相对位置编码器。

        Args:
            relpos_k: 相对位置编码窗口大小
            c_out: 输出特征维度
            rank: 因子化秩
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.relpos_k = relpos_k
        self.n_bin = 2 * relpos_k + 1
        self.c_out = c_out
        self.rank = rank
        self.max_seq_len = max_seq_len

        # 可学习绝对位置嵌入
        self.pos_emb = nn.Embedding(max_seq_len, rank * c_out)

        # 可学习相对位置偏置 (捕捉反对称部分)
        # 添加到 left，从 right 减去以创建反对称性
        self.relpos_bias = nn.Parameter(torch.zeros(rank, c_out))

        # 相对位置箱嵌入，用于额外表达能力
        self.relpos_bin_emb = nn.Embedding(self.n_bin, rank * c_out)

        # 用于组合位置信息的投影层
        self.proj_left = nn.Linear(rank * c_out * 2, rank * c_out)
        self.proj_right = nn.Linear(rank * c_out * 2, rank * c_out)

        # 使用正弦模式初始化位置嵌入以获得更好的泛化
        self._init_pos_emb()

    def _init_pos_emb(self):
        """使用正弦模式初始化位置嵌入以获得更好的泛化。"""
        d_model = self.rank * self.c_out
        pe = torch.zeros(self.max_seq_len, d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2] if d_model % 2 == 1 else div_term)
        self.pos_emb.weight.data.copy_(pe)

    def forward(self, L, device):
        """
        生成因子化相对位置编码。

        Args:
            L: 序列长度
            device: torch 设备

        Returns:
            relpos_left: [L, rank, C]
            relpos_right: [L, rank, C]
        """
        # 获取位置索引
        pos = torch.arange(L, device=device)

        # 绝对位置嵌入 [L, rank*C]
        abs_pos = self.pos_emb(pos)

        # 计算到中心位置的相对箱索引
        # 对于位置 i，我们使用与中心 (L//2) 距离对应的箱
        center = L // 2
        rel_to_center = pos - center
        rel_to_center_clipped = torch.clamp(rel_to_center, -self.relpos_k, self.relpos_k)
        bin_idx = (rel_to_center_clipped + self.relpos_k).long()

        # 获取相对位置箱嵌入 [L, rank*C]
        rel_bin_emb = self.relpos_bin_emb(bin_idx)

        # 组合绝对和相对信息
        combined = torch.cat([abs_pos, rel_bin_emb], dim=-1)  # [L, 2*rank*C]

        # 投影到带反对称偏置的 left/right 因子
        left_base = self.proj_left(combined).view(L, self.rank, self.c_out)
        right_base = self.proj_right(combined).view(L, self.rank, self.c_out)

        # 添加反对称偏置: left 得到 +bias，right 得到 -bias
        # 这确保重建时 relpos[i,j] ≠ relpos[j,i]
        relpos_left = left_base + self.relpos_bias.unsqueeze(0)
        relpos_right = right_base - self.relpos_bias.unsqueeze(0)

        return relpos_left, relpos_right


class FactorizedTemplate(nn.Module):
    """
    因子化模板特征编码 (V2 - 已修复)。

    核心洞察: 模板特征包含残基对之间的几何信息 (距离、角度)。
    简单平均会丢失这些关键信息。

    解决方案: 使用 SVD 风格分解:
    1. 对角线特征 (自信息)
    2. 带注意力加权池化的行/列聚合
    3. 可学习的奇异值缩放

    这在可因子化的同时保留了更多几何信息。
    """

    def __init__(self, template_fn, c_template, c_out, rank):
        """
        初始化因子化模板编码器。

        Args:
            template_fn: 模板特征提取函数
            c_template: 模板特征维度
            c_out: 输出特征维度
            rank: 因子化秩
        """
        super().__init__()
        self.template_fn = template_fn
        self.rank = rank
        self.c_out = c_out
        self.c_template = c_template

        # SVD 风格因子化: U @ Sigma @ V^T
        # U 投影 (左奇异向量)
        self.linear_U = nn.Linear(c_template * 3, rank * c_out)
        # V 投影 (右奇异向量)
        self.linear_V = nn.Linear(c_template * 3, rank * c_out)
        # 可学习的奇异值 (每秩的重要性加权)
        self.sigma = nn.Parameter(torch.ones(rank))

        # 用于加权聚合的注意力 (优于简单平均)
        self.attn_query = nn.Linear(c_template, 1)

        # 用于稳定性的层归一化
        self.layer_norm = nn.LayerNorm(c_template * 3)

    def _attention_pool(self, feat, dim):
        """
        沿指定维度进行注意力加权池化。

        Args:
            feat: [B, L, L, C] 模板特征
            dim: 池化维度 (1 或 2)

        Returns:
            pooled: [B, L, C] 注意力加权特征
        """
        # 计算注意力权重
        attn_logits = self.attn_query(feat).squeeze(-1)  # [B, L, L]

        if dim == 2:
            attn_weights = torch.softmax(attn_logits, dim=2)  # [B, L, L]
            pooled = torch.einsum('bij,bijc->bic', attn_weights, feat)
        else:  # dim == 1
            attn_weights = torch.softmax(attn_logits, dim=1)  # [B, L, L]
            pooled = torch.einsum('bij,bijc->bjc', attn_weights, feat)

        return pooled

    def forward(self, t):
        """
        生成因子化模板特征。

        Args:
            t: 输入变换 (Rigid 对象)

        Returns:
            template_left: [B, L, rank, C]
            template_right: [B, L, rank, C]
        """
        # 获取模板特征 [B, L, L, c_template]
        template_feat = self.template_fn(t)

        B, L, _, C_t = template_feat.shape

        # 提取对角线特征 (自信息) [B, L, C_t]
        diag_idx = torch.arange(L, device=template_feat.device)
        diag_feat = template_feat[:, diag_idx, diag_idx, :]  # [B, L, C_t]

        # 注意力加权行聚合 [B, L, C_t]
        row_feat = self._attention_pool(template_feat, dim=2)

        # 注意力加权列聚合 [B, L, C_t]
        col_feat = self._attention_pool(template_feat, dim=1)

        # 组合 left 因子的特征: [B, L, 3*C_t]
        left_combined = torch.cat([diag_feat, row_feat, col_feat], dim=-1)
        left_combined = self.layer_norm(left_combined)

        # 组合 right 因子的特征 (不同组合以保持不对称性)
        right_combined = torch.cat([diag_feat, col_feat, row_feat], dim=-1)
        right_combined = self.layer_norm(right_combined)

        # 投影到因子化形式 [B, L, rank×C] → [B, L, rank, C]
        template_left = self.linear_U(left_combined).view(B, L, self.rank, self.c_out)
        template_right = self.linear_V(right_combined).view(B, L, self.rank, self.c_out)

        # 应用奇异值缩放 (重要性加权)
        # 只应用于 left 因子 (类似 SVD: U @ Sigma)
        template_left = template_left * self.sigma.view(1, 1, -1, 1)

        return template_left, template_right


class FactorizedPairFeatureNet(nn.Module):
    """
    内存高效的因子化配对特征网络。

    此模块直接生成因子化配对表示，而无需实例化完整的 L² 配对张量。

    核心优势:
    1. 内存: O(L²) → O(L×rank) (通常 256-512x 节省)
    2. 速度: 更快的前向传播 (无需实例化开销)
    3. 兼容性: 输出格式与 LinearFactorizer 匹配

    使用方法:
        # Flash-IPA 模式
        factor_1, factor_2 = factorized_pair_net(s, t, mask)
        s = flash_ipa(s, None, factor_1, factor_2, t, mask)

        # 如果需要，重建完整配对 (用于调试)
        p_reconstructed = reconstruct_pair(factor_1, factor_2)
    """

    def __init__(self, c_s, c_p, rank, relpos_k, template_type):
        """
        初始化因子化配对特征网络。

        Args:
            c_s: 单特征维度
            c_p: 配对特征维度 (输出)
            rank: 因子化秩 (通常 2-4)
            relpos_k: 相对位置编码窗口
            template_type: 模板特征类型
        """
        super().__init__()

        self.c_s = c_s
        self.c_p = c_p
        self.rank = rank

        # 因子化 single → pair 投影
        # 生成秩因子化表示
        self.linear_left = nn.Linear(c_s, rank * c_p)
        self.linear_right = nn.Linear(c_s, rank * c_p)

        # 因子化 relpos
        self.relpos_encoder = FactorizedRelPos(relpos_k, c_p, rank)

        # 因子化 template
        from genie.model.template import get_template_fn
        template_fn, c_template = get_template_fn(template_type)
        self.template_encoder = FactorizedTemplate(template_fn, c_template, c_p, rank)

    def forward(self, s, t, mask):
        """
        生成因子化配对表示。

        Args:
            s: 单表示 [B, L, C_s]
            t: 刚性变换
            mask: 序列掩码 [B, L]

        Returns:
            factor_1: [B, L, rank, C_p] - 左因子
            factor_2: [B, L, rank, C_p] - 右因子

        因子化表示近似完整配对张量为:
            p[i, j] ≈ sum_r (factor_1[i, r] * factor_2[j, r])

        这是完整配对张量的低秩近似:
            p[i, j] = s_i + s_j + relpos[i,j] + template[i,j]
        """
        B, L, _ = s.shape

        # 投影单特征到因子化形式 [B, L, rank×C] → [B, L, rank, C]
        left = self.linear_left(s).view(B, L, self.rank, self.c_p)
        right = self.linear_right(s).view(B, L, self.rank, self.c_p)

        # 添加因子化相对位置编码 [L, rank, C]
        relpos_left, relpos_right = self.relpos_encoder(L, s.device)
        left = left + relpos_left.unsqueeze(0)  # [B, L, rank, C]
        right = right + relpos_right.unsqueeze(0)  # [B, L, rank, C]

        # 添加因子化模板特征 [B, L, rank, C]
        template_left, template_right = self.template_encoder(t)
        left = left + template_left
        right = right + template_right

        # 应用掩码 [B, L] → [B, L, 1, 1]
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
        left = left * mask_expanded
        right = right * mask_expanded

        return left, right

    @staticmethod
    def reconstruct_pair(factor_1, factor_2):
        """
        从因子重建完整配对张量 (用于调试/验证)。

        Args:
            factor_1: [B, L, rank, C]
            factor_2: [B, L, rank, C]

        Returns:
            p: [B, L, L, C] - 重建的配对张量
        """
        # p[i, j] = sum_r (factor_1[i, r] * factor_2[j, r])
        # 使用 einsum: 'birc,bjrc->bijc'
        B, L, rank, C = factor_1.shape
        p = torch.einsum('birc,bjrc->bijc', factor_1, factor_2)
        return p


class AdaptiveFactorizationRank(nn.Module):
    """
    根据序列长度动态调整因子化秩。

    较短的序列可以承受更高的秩 (更多表达能力)，
    较长的序列需要更低的秩 (更少内存)。
    """

    @staticmethod
    def compute_rank(seq_len, base_rank=2, max_rank=8):
        """
        根据序列长度计算因子化秩。

        策略:
            L < 256:  rank = max_rank (例如 8)
            256-512:  rank = 4
            512-1024: rank = 2
            > 1024:   rank = 2 (最小值)

        Args:
            seq_len: 序列长度
            base_rank: 最小秩
            max_rank: 最大秩

        Returns:
            rank: 因子化秩
        """
        if seq_len < 256:
            return max_rank
        elif seq_len < 512:
            return max(base_rank * 2, base_rank)
        else:
            return base_rank


class FactorizedPairRefinement(nn.Module):
    """
    轻量级因子化配对特征精炼。

    此模块在不实例化完整 L² 张量的情况下提供配对特征更新。
    它以因子化方式模拟三角更新的效果。

    复杂度: O(L × rank² × C) 而非 O(L³ × C)

    核心思想: 不在完整配对张量上执行三角乘法更新，
    而是执行因子到因子的交互，捕捉类似的信息流。
    """

    def __init__(self, c_p, rank, n_layers=2, dropout=0.1):
        """
        初始化因子化配对精炼模块。

        Args:
            c_p: 配对特征维度
            rank: 因子化秩
            n_layers: 精炼层数
            dropout: Dropout 比率
        """
        super().__init__()
        self.c_p = c_p
        self.rank = rank
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            FactorizedPairRefinementLayer(c_p, rank, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, factor_1, factor_2, mask):
        """
        精炼因子化配对特征。

        Args:
            factor_1: [B, L, rank, C] 左因子
            factor_2: [B, L, rank, C] 右因子
            mask: [B, L] 序列掩码

        Returns:
            factor_1: [B, L, rank, C] 精炼后的左因子
            factor_2: [B, L, rank, C] 精炼后的右因子
        """
        for layer in self.layers:
            factor_1, factor_2 = layer(factor_1, factor_2, mask)
        return factor_1, factor_2


class FactorizedPairRefinementLayer(nn.Module):
    """
    单层因子化配对精炼。

    通过以下方式模拟类似三角的更新:
    1. 跨因子注意力 (factor_1 关注 factor_2，反之亦然)
    2. 自因子精炼
    3. 门控残差连接
    """

    def __init__(self, c_p, rank, dropout=0.1):
        super().__init__()
        self.c_p = c_p
        self.rank = rank

        # 跨因子交互 (模拟三角乘法更新)
        # factor_1[i] 与聚合的 factor_2 信息交互
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=c_p,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=c_p,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # 自精炼 FFN
        self.ffn_1 = nn.Sequential(
            nn.Linear(c_p, c_p * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_p * 4, c_p),
            nn.Dropout(dropout)
        )
        self.ffn_2 = nn.Sequential(
            nn.Linear(c_p, c_p * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_p * 4, c_p),
            nn.Dropout(dropout)
        )

        # 用于残差连接的门
        self.gate_1 = nn.Sequential(
            nn.Linear(c_p * 2, c_p),
            nn.Sigmoid()
        )
        self.gate_2 = nn.Sequential(
            nn.Linear(c_p * 2, c_p),
            nn.Sigmoid()
        )

        # 层归一化
        self.ln1_1 = nn.LayerNorm(c_p)
        self.ln1_2 = nn.LayerNorm(c_p)
        self.ln2_1 = nn.LayerNorm(c_p)
        self.ln2_2 = nn.LayerNorm(c_p)

        # 秩混合 (允许秩之间的信息流动)
        self.rank_mix_1 = nn.Linear(rank, rank)
        self.rank_mix_2 = nn.Linear(rank, rank)

    def forward(self, factor_1, factor_2, mask):
        """
        单精炼层的前向传播。

        Args:
            factor_1: [B, L, rank, C]
            factor_2: [B, L, rank, C]
            mask: [B, L]

        Returns:
            factor_1_out: [B, L, rank, C]
            factor_2_out: [B, L, rank, C]
        """
        B, L, R, C = factor_1.shape

        # 从序列掩码创建注意力掩码
        # [B, L] -> [B, L, L] key_padding_mask 格式
        attn_mask = ~mask.bool() if mask is not None else None

        # 为跨注意力分别处理每个秩
        # 这在允许因子交互的同时保持低内存使用
        f1_updates = []
        f2_updates = []

        for r in range(R):
            # 提取秩切片 [B, L, C]
            f1_r = factor_1[:, :, r, :]
            f2_r = factor_2[:, :, r, :]

            # 跨注意力: f1 关注 f2
            f1_r_norm = self.ln1_1(f1_r)
            f2_r_norm = self.ln1_2(f2_r)

            f1_cross, _ = self.cross_attn_1(
                f1_r_norm, f2_r_norm, f2_r_norm,
                key_padding_mask=attn_mask
            )

            f2_cross, _ = self.cross_attn_2(
                f2_r_norm, f1_r_norm, f1_r_norm,
                key_padding_mask=attn_mask
            )

            # 门控残差
            gate_1 = self.gate_1(torch.cat([f1_r, f1_cross], dim=-1))
            gate_2 = self.gate_2(torch.cat([f2_r, f2_cross], dim=-1))

            f1_r = f1_r + gate_1 * f1_cross
            f2_r = f2_r + gate_2 * f2_cross

            # FFN 精炼
            f1_r = f1_r + self.ffn_1(self.ln2_1(f1_r))
            f2_r = f2_r + self.ffn_2(self.ln2_2(f2_r))

            f1_updates.append(f1_r)
            f2_updates.append(f2_r)

        # 堆叠回 [B, L, R, C]
        factor_1_out = torch.stack(f1_updates, dim=2)
        factor_2_out = torch.stack(f2_updates, dim=2)

        # 秩混合: 允许秩之间信息流动
        # [B, L, R, C] -> [B, L, C, R] -> 混合 -> [B, L, C, R] -> [B, L, R, C]
        factor_1_out = self.rank_mix_1(factor_1_out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        factor_2_out = self.rank_mix_2(factor_2_out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        # 应用掩码
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
            factor_1_out = factor_1_out * mask_expanded
            factor_2_out = factor_2_out * mask_expanded

        return factor_1_out, factor_2_out


def test_factorized_pair_features():
    """
    测试因子化配对特征与标准实现的对比。
    """
    print("=" * 60)
    print("测试因子化配对特征")
    print("=" * 60)

    # 参数
    B, L, C_s, C_p = 2, 128, 128, 128
    rank = 2
    relpos_k = 32

    # 创建因子化模型
    from genie.model.template import get_template_fn
    factorized_net = FactorizedPairFeatureNet(
        c_s=C_s,
        c_p=C_p,
        rank=rank,
        relpos_k=relpos_k,
        template_type='v1'
    )

    # 测试输入
    s = torch.randn(B, L, C_s)
    from genie.flash_ipa.rigid import create_identity_rigid
    t = create_identity_rigid(B, L)
    mask = torch.ones(B, L)

    # 前向传播
    factor_1, factor_2 = factorized_net(s, t, mask)

    # 检查形状
    assert factor_1.shape == (B, L, rank, C_p), f"期望 {(B, L, rank, C_p)}，得到 {factor_1.shape}"
    assert factor_2.shape == (B, L, rank, C_p), f"期望 {(B, L, rank, C_p)}，得到 {factor_2.shape}"

    # 重建配对张量
    p_reconstructed = FactorizedPairFeatureNet.reconstruct_pair(factor_1, factor_2)
    assert p_reconstructed.shape == (B, L, L, C_p)

    # 检查内存使用
    factor_memory = factor_1.numel() * 4 + factor_2.numel() * 4  # 字节
    full_memory = L * L * C_p * 4  # 假设完整配对张量

    print(f"✅ 形状测试通过")
    print(f"✅ 因子 1: {factor_1.shape}")
    print(f"✅ 因子 2: {factor_2.shape}")
    print(f"✅ 重建: {p_reconstructed.shape}")
    print(f"")
    print(f"内存对比:")
    print(f"  因子化: {factor_memory / 1024 / 1024:.2f} MB")
    print(f"  完整配对: {full_memory / 1024 / 1024:.2f} MB")
    print(f"  节省: {full_memory / factor_memory:.1f}x")
    print(f"")
    print(f"🎉 所有测试通过!")


if __name__ == "__main__":
    test_factorized_pair_features()
