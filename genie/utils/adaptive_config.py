"""
自适应配置工具 (V2)

此模块提供根据序列长度动态调整模型配置的工具，
以实现高效的长序列训练。

核心特性:
1. 自适应 mHC 扩展率 (降低长序列的扩展)
2. 动态批次大小 (保持恒定内存占用)
3. 自适应因子化秩 (长序列使用低秩)
4. 内存估算工具
5. 配对精炼配置 (V2 新增)

V2 改进 (2026-01):
- 添加配对精炼配置
- GPU 内存感知配置
- 更细粒度的序列长度阈值
"""

import torch
import math


class AdaptiveMHCConfig:
    """
    根据序列长度动态调整 mHC 配置。

    策略:
        - 短序列 (<256): 对结构使用完整 mHC (expansion=4)，配对也使用
        - 中等序列 (256-512): 减少配对扩展，保持结构扩展
        - 长序列 (512-1024): 减少结构扩展，禁用配对 mHC
        - 非常长的序列 (>1024): 为内存效率使用最小扩展
    """

    @staticmethod
    def get_config(seq_len, gpu_memory_gb=24):
        """
        获取给定序列长度的自适应 mHC 配置。

        Args:
            seq_len: 序列长度
            gpu_memory_gb: 可用 GPU 内存 (GB) (用于内存感知配置)

        Returns:
            包含以下键的字典:
                - structure_expansion: 结构模块的扩展率
                - pair_expansion: 配对模块的扩展率 (1 = 禁用)
                - sinkhorn_iters: Sinkhorn-Knopp 迭代次数
                - use_pair_mhc: 是否在配对特征上使用 mHC
                - z_factor_rank: 配对特征的因子化秩
                - use_pair_refinement: 是否使用因子化配对精炼
                - pair_refinement_layers: 配对精炼层数
        """
        # 估算可用内存因子
        mem_factor = gpu_memory_gb / 24.0  # 归一化到 24GB 基线

        if seq_len < 256:
            return {
                'structure_expansion': 4,
                'pair_expansion': 2,  # 从 4 减少以节省内存
                'sinkhorn_iters': 20,
                'sinkhorn_iters_inference': 5,
                'use_pair_mhc': True,
                'z_factor_rank': min(8, int(8 * mem_factor)),
                'use_pair_refinement': True,
                'pair_refinement_layers': 2,
            }
        elif seq_len < 384:
            return {
                'structure_expansion': 4,
                'pair_expansion': 1,  # 禁用
                'sinkhorn_iters': 15,
                'sinkhorn_iters_inference': 4,
                'use_pair_mhc': False,
                'z_factor_rank': min(6, int(6 * mem_factor)),
                'use_pair_refinement': True,
                'pair_refinement_layers': 2,
            }
        elif seq_len < 512:
            return {
                'structure_expansion': 4,
                'pair_expansion': 1,  # 禁用
                'sinkhorn_iters': 15,
                'sinkhorn_iters_inference': 3,
                'use_pair_mhc': False,
                'z_factor_rank': min(4, int(4 * mem_factor)),
                'use_pair_refinement': True,
                'pair_refinement_layers': 2,
            }
        elif seq_len < 768:
            return {
                'structure_expansion': 3,  # 减少
                'pair_expansion': 1,
                'sinkhorn_iters': 12,
                'sinkhorn_iters_inference': 3,
                'use_pair_mhc': False,
                'z_factor_rank': max(2, int(3 * mem_factor)),
                'use_pair_refinement': True,
                'pair_refinement_layers': 1,
            }
        elif seq_len < 1024:
            return {
                'structure_expansion': 2,  # 减少
                'pair_expansion': 1,
                'sinkhorn_iters': 10,
                'sinkhorn_iters_inference': 2,
                'use_pair_mhc': False,
                'z_factor_rank': 2,
                'use_pair_refinement': True,
                'pair_refinement_layers': 1,
            }
        elif seq_len < 1536:
            return {
                'structure_expansion': 2,  # 最小
                'pair_expansion': 1,
                'sinkhorn_iters': 10,
                'sinkhorn_iters_inference': 2,
                'use_pair_mhc': False,
                'z_factor_rank': 2,
                'use_pair_refinement': mem_factor >= 1.0,  # 仅当内存足够时
                'pair_refinement_layers': 1 if mem_factor >= 1.0 else 0,
            }
        else:  # >= 1536
            return {
                'structure_expansion': 2,  # 最小
                'pair_expansion': 1,
                'sinkhorn_iters': 8,
                'sinkhorn_iters_inference': 2,
                'use_pair_mhc': False,
                'z_factor_rank': 2,
                'use_pair_refinement': False,  # 对非常长的序列禁用
                'pair_refinement_layers': 0,
            }

    @staticmethod
    def print_config(seq_len, gpu_memory_gb=24):
        """打印给定序列长度的自适应配置。"""
        config = AdaptiveMHCConfig.get_config(seq_len, gpu_memory_gb)
        print(f"自适应 mHC 配置 (L={seq_len}, GPU: {gpu_memory_gb}GB):")
        print(f"  结构扩展: {config['structure_expansion']}x")
        print(f"  配对扩展: {config['pair_expansion']}x {'(禁用)' if config['pair_expansion'] == 1 else ''}")
        print(f"  Sinkhorn 迭代: {config['sinkhorn_iters']} (训练) / {config['sinkhorn_iters_inference']} (推理)")
        print(f"  使用配对 mHC: {config['use_pair_mhc']}")
        print(f"  Z 因子秩: {config['z_factor_rank']}")
        print(f"  使用配对精炼: {config['use_pair_refinement']}")
        if config['use_pair_refinement']:
            print(f"  配对精炼层数: {config['pair_refinement_layers']}")


class DynamicBatchSize:
    """
    根据序列长度动态调整批次大小以保持恒定内存使用。

    策略:
        - 内存使用 ∝ batch_size × L²
        - 保持: batch_size × L² ≈ 常数
        - 因此: batch_size = base_batch × (base_len / L)²
    """

    @staticmethod
    def compute_batch_size(seq_len, base_batch=32, base_len=128, min_batch=1, max_batch=64):
        """
        计算给定序列长度的批次大小。

        Args:
            seq_len: 当前序列长度
            base_batch: base_len 的批次大小
            base_len: 基础序列长度
            min_batch: 最小批次大小
            max_batch: 最大批次大小

        Returns:
            batch_size: 调整后的批次大小

        示例:
            L=128:  batch=32 (基础)
            L=256:  batch=8  (基础的 1/4，因为 L² 是 4 倍)
            L=512:  batch=2
            L=1024: batch=1  (最小值，使用梯度累积)
        """
        ratio = (base_len / seq_len) ** 2
        batch = int(base_batch * ratio)
        batch = max(min_batch, min(max_batch, batch))
        return batch

    @staticmethod
    def compute_accumulation_steps(seq_len, base_batch=32, base_len=128, effective_batch=32):
        """
        计算梯度累积步数以保持有效批次大小。

        Args:
            seq_len: 当前序列长度
            base_batch: 基础批次大小
            base_len: 基础序列长度
            effective_batch: 目标有效批次大小

        Returns:
            accumulation_steps: 累积步数

        示例:
            L=1024, batch=1, target_effective=32 → 累积 32 步
        """
        actual_batch = DynamicBatchSize.compute_batch_size(seq_len, base_batch, base_len)
        steps = max(1, effective_batch // actual_batch)
        return steps

    @staticmethod
    def print_batch_config(seq_len, base_batch=32, base_len=128):
        """打印给定序列长度的批次配置。"""
        batch = DynamicBatchSize.compute_batch_size(seq_len, base_batch, base_len)
        accum_steps = DynamicBatchSize.compute_accumulation_steps(seq_len, base_batch, base_len, base_batch)
        effective_batch = batch * accum_steps

        print(f"批次配置 (L={seq_len}):")
        print(f"  批次大小: {batch}")
        print(f"  累积步数: {accum_steps}")
        print(f"  有效批次: {effective_batch}")


class AdaptiveFactorizationRank:
    """
    根据序列长度和内存约束动态调整配对特征的因子化秩。

    策略:
        - 短序列: 更高秩以获得更好的质量
        - 长序列: 更低秩以获得内存效率
    """

    @staticmethod
    def compute_rank(seq_len, base_rank=2, max_rank=8):
        """
        根据序列长度计算因子化秩。

        Args:
            seq_len: 序列长度
            base_rank: 最小秩 (默认: 2)
            max_rank: 最大秩 (默认: 8)

        Returns:
            rank: 因子化秩

        策略:
            L < 256:  rank = 8 (高质量)
            256-512:  rank = 4
            512-1024: rank = 2 (内存高效)
            > 1024:   rank = 2 (最小值)
        """
        if seq_len < 256:
            return max_rank
        elif seq_len < 512:
            return max(base_rank * 2, base_rank)
        else:
            return base_rank

    @staticmethod
    def print_rank(seq_len):
        """打印给定序列长度的因子化秩。"""
        rank = AdaptiveFactorizationRank.compute_rank(seq_len)
        print(f"因子化秩 (L={seq_len}): {rank}")


class MemoryEstimator:
    """
    估算不同序列长度和配置的内存使用。
    """

    @staticmethod
    def estimate_pair_memory(seq_len, c_p=128, use_factorization=False, rank=2, dtype_bytes=4):
        """
        估算配对特征内存使用。

        Args:
            seq_len: 序列长度
            c_p: 配对特征维度
            use_factorization: 是否使用因子化配对
            rank: 因子化秩
            dtype_bytes: 每个元素的字节数 (FP32 为 4, FP16/BF16 为 2)

        Returns:
            memory_mb: 内存使用 (MB)
        """
        if use_factorization:
            # 因子化: [2, L, rank, C]
            memory = 2 * seq_len * rank * c_p * dtype_bytes
        else:
            # 完整: [L, L, C]
            memory = seq_len * seq_len * c_p * dtype_bytes

        return memory / (1024 ** 2)  # 转换为 MB

    @staticmethod
    def estimate_total_memory(seq_len, batch_size, c_s=128, c_p=128, use_factorization=False, use_mhc=False, mhc_expansion=4):
        """
        估算总 GPU 内存使用。

        Args:
            seq_len: 序列长度
            batch_size: 批次大小
            c_s: 单特征维度
            c_p: 配对特征维度
            use_factorization: 使用因子化配对
            use_mhc: 使用 mHC
            mhc_expansion: mHC 扩展率

        Returns:
            包含内存细分的字典
        """
        dtype_bytes = 4  # FP32

        # 单特征: [B, L, C]
        single_mem = batch_size * seq_len * c_s * dtype_bytes / (1024 ** 2)

        # 带 mHC 的单特征: [B, L, n, C]
        if use_mhc:
            single_mem *= mhc_expansion

        # 配对特征
        pair_mem = batch_size * MemoryEstimator.estimate_pair_memory(
            seq_len, c_p, use_factorization, rank=2, dtype_bytes=dtype_bytes
        )

        # 结构模块 (IPA 激活)
        # 近似: 约 3 倍单特征大小
        structure_mem = single_mem * 3

        # 梯度 (约 2 倍参数 + 激活)
        activation_mem = single_mem + pair_mem + structure_mem
        gradient_mem = activation_mem * 2

        total_mem = activation_mem + gradient_mem

        return {
            'single': single_mem,
            'pair': pair_mem,
            'structure': structure_mem,
            'activations': activation_mem,
            'gradients': gradient_mem,
            'total': total_mem,
        }

    @staticmethod
    def print_memory_comparison(seq_lengths=[128, 256, 512, 1024], batch_size=32, base_len=128):
        """
        打印不同配置的内存对比。
        """
        print("=" * 80)
        print("内存使用对比 (MB 每批次)")
        print("=" * 80)
        print(f"{'长度':<10} {'批次':<10} {'标准':<15} {'因子化':<15} {'减少':<15}")
        print("-" * 80)

        for seq_len in seq_lengths:
            # 计算动态批次大小
            batch = DynamicBatchSize.compute_batch_size(seq_len, batch_size, base_len)

            # 标准 (完整配对)
            mem_standard = MemoryEstimator.estimate_total_memory(
                seq_len, batch, use_factorization=False, use_mhc=False
            )['total']

            # 因子化配对
            mem_factorized = MemoryEstimator.estimate_total_memory(
                seq_len, batch, use_factorization=True, use_mhc=False
            )['total']

            reduction = mem_standard / mem_factorized if mem_factorized > 0 else 0

            print(f"{seq_len:<10} {batch:<10} {mem_standard:<15.1f} {mem_factorized:<15.1f} {reduction:<15.1f}x")

        print("=" * 80)


def print_adaptive_configs():
    """
    打印不同序列长度的所有自适应配置。
    """
    print("=" * 80)
    print("长序列的自适应配置")
    print("=" * 80)
    print()

    lengths = [128, 256, 384, 512, 768, 1024, 1536, 2048]

    for seq_len in lengths:
        print(f"{'=' * 80}")
        print(f"序列长度: {seq_len}")
        print(f"{'=' * 80}")

        # mHC 配置
        AdaptiveMHCConfig.print_config(seq_len)
        print()

        # 批次配置
        DynamicBatchSize.print_batch_config(seq_len, base_batch=32, base_len=128)
        print()

        # 因子化秩
        AdaptiveFactorizationRank.print_rank(seq_len)
        print()

        # 内存估算
        batch = DynamicBatchSize.compute_batch_size(seq_len, base_batch=32, base_len=128)
        mem = MemoryEstimator.estimate_total_memory(
            seq_len, batch, use_factorization=True, use_mhc=True, mhc_expansion=4
        )
        print(f"内存估算:")
        print(f"  激活: {mem['activations']:.1f} MB")
        print(f"  总计 (含梯度): {mem['total']:.1f} MB")
        print()


if __name__ == "__main__":
    print_adaptive_configs()
    print()
    print()
    MemoryEstimator.print_memory_comparison()
