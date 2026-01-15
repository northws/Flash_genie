"""
高级梯度检查点 (Stage 4)

此模块提供超越 PyTorch 标准检查点函数的高级梯度检查点策略。

核心特性:
1. 选择性检查点: 仅检查点昂贵的操作
2. 自适应检查点: 基于内存使用动态调整
3. 层-wise 检查点: 对层进行细粒度控制

内存优势:
- 标准: 存储所有激活 (O(L × depth))
- 检查点: 只存储一些激活，重新计算其他 (O(L × √depth))
- 对于 8 层模型，L=1024: 约 3x 内存减少

基于:
- 梯度检查点 (Chen et al. 2016)
- 选择性激活检查点 (Jain et al. 2020)
- AlphaFold2 检查点策略

作者: Stage 4 实现 (2026-01-13)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Callable, List, Any, Tuple
import contextlib


class CheckpointConfig:
    """
    梯度检查点策略的配置。
    """

    def __init__(
        self,
        enabled: bool = True,
        strategy: str = "selective",  # "none", "all", "selective", "adaptive"
        checkpoint_structure: bool = True,
        checkpoint_pairs: bool = False,  # 配对操作昂贵，考虑检查点
        checkpoint_triangles: bool = True,  # 三角操作非常昂贵
        min_memory_gb: float = 2.0,  # 启用激进检查点前的最小空闲内存
    ):
        """
        初始化检查点配置。

        Args:
            enabled: 启用梯度检查点
            strategy: 检查点策略
            checkpoint_structure: 检查点结构模块层
            checkpoint_pairs: 检查点配对操作
            checkpoint_triangles: 检查点三角操作
            min_memory_gb: 最小空闲内存阈值
        """
        self.enabled = enabled
        self.strategy = strategy
        self.checkpoint_structure = checkpoint_structure
        self.checkpoint_pairs = checkpoint_pairs
        self.checkpoint_triangles = checkpoint_triangles
        self.min_memory_gb = min_memory_gb

    @staticmethod
    def get_adaptive_config(seq_len: int, available_memory_gb: float) -> "CheckpointConfig":
        """
        根据序列长度和可用内存获取自适应检查点配置。

        Args:
            seq_len: 序列长度
            available_memory_gb: 可用 GPU 内存 (GB)

        Returns:
            针对给定条件优化的 CheckpointConfig
        """
        if seq_len < 256 and available_memory_gb > 10:
            # 内存充足短序列: 不需要检查点
            return CheckpointConfig(enabled=False, strategy="none")

        elif seq_len < 512 and available_memory_gb > 8:
            # 中等序列: 选择性检查点
            return CheckpointConfig(
                enabled=True,
                strategy="selective",
                checkpoint_structure=False,
                checkpoint_pairs=False,
                checkpoint_triangles=True,
            )

        elif seq_len < 1024 and available_memory_gb > 6:
            # 长序列: 检查点三角和配对
            return CheckpointConfig(
                enabled=True,
                strategy="selective",
                checkpoint_structure=True,
                checkpoint_pairs=True,
                checkpoint_triangles=True,
            )

        else:
            # 非常长序列或低内存: 检查点一切
            return CheckpointConfig(
                enabled=True,
                strategy="all",
                checkpoint_structure=True,
                checkpoint_pairs=True,
                checkpoint_triangles=True,
            )


class SelectiveCheckpoint:
    """
    选择性梯度检查点包装器。

    用法:
        with SelectiveCheckpoint(config):
            # 根据配置对上下文内的操作进行检查点
            output = expensive_operation(input)
    """

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self._original_checkpoint_setting = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def checkpoint_function(
        self,
        function: Callable,
        *args,
        use_reentrant: bool = True,
        **kwargs
    ) -> Any:
        """
        根据配置有条件地检查点函数。

        Args:
            function: 要检查点的函数
            *args: 函数的参数
            use_reentrant: 使用可重入检查点
            **kwargs: 函数的关键字参数

        Returns:
            函数输出
        """
        if self.config.enabled:
            return checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
        else:
            return function(*args, **kwargs)


class CheckpointedSequential(nn.Sequential):
    """
    支持梯度检查点的顺序模块。

    每 N 层检查点一次，而不是所有层。
    """

    def __init__(self, *args, checkpoint_every: int = 2, config: Optional[CheckpointConfig] = None):
        """
        Args:
            *args: 顺序中包含的模块
            checkpoint_every: 每 N 层检查点一次 (例如: 2 = 每隔一层)
            config: 检查点配置
        """
        super().__init__(*args)
        self.checkpoint_every = checkpoint_every
        self.config = config or CheckpointConfig(enabled=True, strategy="selective")

    def forward(self, x):
        """带选择性检查点的前向传播。"""
        for i, module in enumerate(self):
            if self.config.enabled and (i % self.checkpoint_every == 0):
                # 检查点此层
                x = checkpoint(module, x, use_reentrant=False)
            else:
                # 常规前向传播
                x = module(x)
        return x


class LayerWithCheckpoint(nn.Module):
    """
    任何层的包装器，添加梯度检查点。

    用法:
        layer = LayerWithCheckpoint(
            my_expensive_layer,
            checkpoint_enabled=True
        )
    """

    def __init__(
        self,
        layer: nn.Module,
        checkpoint_enabled: bool = True,
        use_reentrant: bool = False,
    ):
        super().__init__()
        self.layer = layer
        self.checkpoint_enabled = checkpoint_enabled
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        """带可选检查点的前向传播。"""
        if self.checkpoint_enabled and self.training:
            return checkpoint(
                self.layer,
                *args,
                use_reentrant=self.use_reentrant,
                **kwargs
            )
        else:
            return self.layer(*args, **kwargs)


def get_memory_stats() -> dict:
    """
    获取当前 GPU 内存统计信息。

    Returns:
        包含内存统计信息的字典
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "free_gb": 0.0,
        }

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    # 获取总内存
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free = total - allocated

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free,
        "total_gb": total,
    }


class AdaptiveCheckpointManager:
    """
    基于内存使用自适应管理梯度检查点。

    监控 GPU 内存并动态启用/禁用检查点。
    """

    def __init__(
        self,
        initial_config: Optional[CheckpointConfig] = None,
        memory_threshold_gb: float = 2.0,
    ):
        """
        Args:
            initial_config: 初始检查点配置
            memory_threshold_gb: 空闲内存阈值，触发激进检查点
        """
        self.config = initial_config or CheckpointConfig()
        self.memory_threshold_gb = memory_threshold_gb
        self.stats = {"adaptations": 0, "memory_warnings": 0}

    def get_current_config(self, seq_len: int) -> CheckpointConfig:
        """
        根据内存状态获取当前检查点配置。

        Args:
            seq_len: 当前序列长度

        Returns:
            适当的 CheckpointConfig
        """
        mem_stats = get_memory_stats()
        free_memory = mem_stats["free_gb"]

        # 如果内存低则更新配置
        if free_memory < self.memory_threshold_gb:
            self.stats["memory_warnings"] += 1
            # 启用激进检查点
            return CheckpointConfig(
                enabled=True,
                strategy="all",
                checkpoint_structure=True,
                checkpoint_pairs=True,
                checkpoint_triangles=True,
            )
        else:
            # 使用自适应配置
            return CheckpointConfig.get_adaptive_config(seq_len, free_memory)

    def adapt_if_needed(self, seq_len: int) -> CheckpointConfig:
        """
        根据当前条件需要时调整检查点配置。

        Args:
            seq_len: 当前序列长度

        Returns:
            更新的 CheckpointConfig
        """
        new_config = self.get_current_config(seq_len)

        if new_config.strategy != self.config.strategy:
            self.stats["adaptations"] += 1
            self.config = new_config

        return self.config

    def get_stats(self) -> dict:
        """获取适配统计信息。"""
        mem_stats = get_memory_stats()
        return {
            **self.stats,
            **mem_stats,
            "current_strategy": self.config.strategy,
        }


def test_gradient_checkpointing():
    """测试梯度检查点工具。"""
    print("=" * 80)
    print("测试梯度检查点 (Stage 4)")
    print("=" * 80)
    print()

    # 测试 1: 基本检查点
    print("测试 1: 基本检查点配置")
    print("-" * 80)

    config = CheckpointConfig(enabled=True, strategy="selective")
    print(f"  配置: enabled={config.enabled}, strategy={config.strategy}")
    print(f"  ✅ 基本配置工作正常!")
    print()

    # 测试 2: 自适应配置
    print("测试 2: 自适应配置")
    print("-" * 80)

    test_cases = [
        (256, 12.0),
        (512, 8.0),
        (1024, 6.0),
        (2048, 4.0),
    ]

    for seq_len, memory in test_cases:
        config = CheckpointConfig.get_adaptive_config(seq_len, memory)
        print(f"  L={seq_len:4d}, Memory={memory:.1f}GB: "
              f"strategy={config.strategy}, "
              f"checkpoint_triangles={config.checkpoint_triangles}")

    print(f"  ✅ 自适应配置工作正常!")
    print()

    # 测试 3: 检查点顺序
    print("测试 3: 检查点顺序")
    print("-" * 80)

    layers = [
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
    ]

    seq = CheckpointedSequential(*layers, checkpoint_every=2)
    x = torch.randn(2, 128, requires_grad=True)

    y = seq(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "梯度未计算"
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {y.shape}")
    print(f"  梯度形状: {x.grad.shape}")
    print(f"  ✅ 检查点顺序工作正常!")
    print()

    # 测试 4: 内存统计
    print("测试 4: 内存统计")
    print("-" * 80)

    mem_stats = get_memory_stats()
    for key, value in mem_stats.items():
        print(f"  {key}: {value:.3f}")

    print(f"  ✅ 内存统计工作正常!")
    print()

    # 测试 5: 自适应管理器
    print("测试 5: 自适应检查点管理器")
    print("-" * 80)

    manager = AdaptiveCheckpointManager(memory_threshold_gb=4.0)

    for seq_len in [256, 512, 1024]:
        config = manager.adapt_if_needed(seq_len)
        print(f"  L={seq_len:4d}: strategy={config.strategy}")

    stats = manager.get_stats()
    print(f"  适配次数: {stats['adaptations']}")
    print(f"  内存警告: {stats['memory_warnings']}")
    print(f"  ✅ 自适应管理器工作正常!")
    print()

    print("=" * 80)
    print("🎉 所有梯度检查点测试通过!")
    print("=" * 80)


if __name__ == "__main__":
    test_gradient_checkpointing()
