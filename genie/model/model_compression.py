"""
Model Compression via Layer Sharing (Stage 4)

This module implements parameter-efficient model compression techniques:
1. Universal Transformer style layer sharing
2. Alternating layer sharing
3. Bottleneck layers

Key Benefits:
- 4-8x parameter reduction
- Enables deeper networks with same memory
- Better generalization (parameter regularization effect)

Based on:
- Universal Transformers (Dehghani et al. 2019)
- ALBERT (Lan et al. 2020)
- Parameter Sharing in Deep Networks

Author: Stage 4 Implementation (2026-01-13)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable
import copy


class SharedLayerModule(nn.Module):
    """
    Base module that can be shared across multiple positions in the network.

    This allows Universal Transformer-style parameter sharing where the same
    layer is applied multiple times.
    """

    def __init__(self, base_layer: nn.Module, num_iterations: int = 1):
        """
        Args:
            base_layer: The layer to share
            num_iterations: Number of times to apply the layer
        """
        super().__init__()
        self.layer = base_layer
        self.num_iterations = num_iterations

    def forward(self, x, *args, **kwargs):
        """
        Apply the shared layer multiple times.

        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments for the layer

        Returns:
            Output after num_iterations applications
        """
        for _ in range(self.num_iterations):
            x = self.layer(x, *args, **kwargs)
        return x


class AlternatingSharedLayers(nn.Module):
    """
    Alternating layer sharing: Share parameters between alternating layers.

    Example with 6 layers:
        Layer 0, 2, 4 share parameters (odd positions)
        Layer 1, 3, 5 share parameters (even positions)

    Reduces parameters by ~2x while maintaining expressivity.
    """

    def __init__(
        self,
        layer_odd: nn.Module,
        layer_even: nn.Module,
        num_layers: int,
    ):
        """
        Args:
            layer_odd: Shared layer for odd positions (0, 2, 4, ...)
            layer_even: Shared layer for even positions (1, 3, 5, ...)
            num_layers: Total number of layers
        """
        super().__init__()
        self.layer_odd = layer_odd
        self.layer_even = layer_even
        self.num_layers = num_layers

    def forward(self, x, *args, **kwargs):
        """
        Apply layers in alternating pattern.

        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments for layers

        Returns:
            Output after all layers
        """
        for i in range(self.num_layers):
            if i % 2 == 0:
                x = self.layer_odd(x, *args, **kwargs)
            else:
                x = self.layer_even(x, *args, **kwargs)
        return x


class BlockSharedLayers(nn.Module):
    """
    Block-wise layer sharing: Share parameters within blocks.

    Example with 12 layers, block_size=4:
        Layers 0-3: Share first block parameters
        Layers 4-7: Share second block parameters
        Layers 8-11: Share third block parameters

    Reduces parameters by block_size while maintaining per-block specialization.
    """

    def __init__(
        self,
        block_layers: List[nn.Module],
        iterations_per_block: int = 4,
    ):
        """
        Args:
            block_layers: List of unique block layers
            iterations_per_block: How many times to apply each block
        """
        super().__init__()
        self.blocks = nn.ModuleList(block_layers)
        self.iterations_per_block = iterations_per_block

    def forward(self, x, *args, **kwargs):
        """
        Apply each block multiple times.

        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments for layers

        Returns:
            Output after all blocks
        """
        for block in self.blocks:
            for _ in range(self.iterations_per_block):
                x = block(x, *args, **kwargs)
        return x


class BottleneckLayer(nn.Module):
    """
    Bottleneck layer: Reduce dimensionality before expensive operations.

    Structure:
        Input (C) -> Project down (C//r) -> Operation -> Project up (C)

    Where r is the bottleneck ratio (e.g., r=4 for 4x compression).
    """

    def __init__(
        self,
        operation: nn.Module,
        c_in: int,
        bottleneck_ratio: int = 4,
    ):
        """
        Args:
            operation: The operation to apply in bottleneck space
            c_in: Input channel dimension
            bottleneck_ratio: Compression ratio
        """
        super().__init__()

        c_bottleneck = max(c_in // bottleneck_ratio, 32)

        self.project_down = nn.Linear(c_in, c_bottleneck)
        self.operation = operation
        self.project_up = nn.Linear(c_bottleneck, c_in)
        self.norm = nn.LayerNorm(c_in)

    def forward(self, x):
        """
        Apply bottleneck transformation.

        Args:
            x: Input tensor [..., C]

        Returns:
            Output tensor [..., C]
        """
        identity = x

        # Project down
        x = self.project_down(x)

        # Apply operation in bottleneck space
        x = self.operation(x)

        # Project up
        x = self.project_up(x)

        # Residual connection
        x = x + identity
        x = self.norm(x)

        return x


class CompressedStructureNet(nn.Module):
    """
    Compressed structure network with layer sharing.

    Combines multiple compression techniques:
    1. Universal layer sharing
    2. Bottleneck projections
    3. Block-wise specialization
    """

    def __init__(
        self,
        c_s: int,
        c_hidden: int,
        num_layers: int = 8,
        sharing_strategy: str = "universal",  # "universal", "alternating", "block"
        block_size: int = 4,
        use_bottleneck: bool = False,
        bottleneck_ratio: int = 4,
    ):
        """
        Args:
            c_s: Single representation dimension
            c_hidden: Hidden dimension
            num_layers: Total number of layers
            sharing_strategy: How to share parameters
            block_size: Block size for block sharing
            use_bottleneck: Use bottleneck layers
            bottleneck_ratio: Bottleneck compression ratio
        """
        super().__init__()

        self.c_s = c_s
        self.num_layers = num_layers
        self.sharing_strategy = sharing_strategy

        # Create base layer
        base_layer = nn.Sequential(
            nn.Linear(c_s, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_s),
            nn.LayerNorm(c_s),
        )

        if use_bottleneck:
            base_layer = BottleneckLayer(base_layer, c_s, bottleneck_ratio)

        # Configure sharing strategy
        if sharing_strategy == "universal":
            # All layers share the same parameters
            self.layers = SharedLayerModule(base_layer, num_iterations=num_layers)
            self.param_count = self._count_parameters(base_layer)

        elif sharing_strategy == "alternating":
            # Two sets of shared layers
            layer_even = copy.deepcopy(base_layer)
            layer_odd = copy.deepcopy(base_layer)
            self.layers = AlternatingSharedLayers(layer_odd, layer_even, num_layers)
            self.param_count = self._count_parameters(layer_odd) + self._count_parameters(layer_even)

        elif sharing_strategy == "block":
            # Block-wise sharing
            num_blocks = max(num_layers // block_size, 1)
            iterations_per_block = block_size
            block_layers = [copy.deepcopy(base_layer) for _ in range(num_blocks)]
            self.layers = BlockSharedLayers(block_layers, iterations_per_block)
            self.param_count = sum(self._count_parameters(bl) for bl in block_layers)

        else:
            raise ValueError(f"Unknown sharing strategy: {sharing_strategy}")

        # Standard (non-shared) baseline for comparison
        self.baseline_param_count = self._count_parameters(base_layer) * num_layers

    def forward(self, s):
        """
        Forward pass through compressed network.

        Args:
            s: Single representation [B, L, c_s]

        Returns:
            Updated representation [B, L, c_s]
        """
        return self.layers(s)

    def _count_parameters(self, module: nn.Module) -> int:
        """Count parameters in a module."""
        return sum(p.numel() for p in module.parameters())

    def get_compression_ratio(self) -> float:
        """Get parameter compression ratio vs baseline."""
        return self.baseline_param_count / self.param_count

    def print_stats(self):
        """Print compression statistics."""
        compression = self.get_compression_ratio()
        print(f"  Strategy: {self.sharing_strategy}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Baseline params: {self.baseline_param_count:,}")
        print(f"  Compressed params: {self.param_count:,}")
        print(f"  Compression ratio: {compression:.2f}x")


class DepthwiseSeparableLayer(nn.Module):
    """
    Depthwise separable layer for further parameter reduction.

    Separates spatial (depthwise) and channel (pointwise) operations.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()

        # Depthwise: Operate on each channel separately
        self.depthwise = nn.Conv1d(
            c_in, c_in, kernel_size, padding=kernel_size // 2, groups=c_in
        )

        # Pointwise: Mix channels
        self.pointwise = nn.Conv1d(c_in, c_out, 1)

        self.norm = nn.LayerNorm(c_out)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input [B, L, C]

        Returns:
            Output [B, L, C']
        """
        # Transpose for conv1d: [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)

        # Depthwise + pointwise
        x = self.depthwise(x)
        x = self.pointwise(x)

        # Transpose back: [B, C', L] -> [B, L, C']
        x = x.transpose(1, 2)

        x = self.norm(x)
        x = self.activation(x)

        return x


def test_model_compression():
    """Test model compression techniques."""
    print("=" * 80)
    print("Testing Model Compression (Stage 4)")
    print("=" * 80)
    print()

    B, L, C = 2, 256, 128
    num_layers = 8

    # Test 1: Universal sharing
    print("Test 1: Universal Transformer Style Sharing")
    print("-" * 80)

    model_universal = CompressedStructureNet(
        c_s=C,
        c_hidden=C * 2,
        num_layers=num_layers,
        sharing_strategy="universal",
    )

    x = torch.randn(B, L, C)
    y = model_universal(x)

    assert y.shape == (B, L, C), f"Shape mismatch: {y.shape}"
    assert not torch.isnan(y).any(), "NaN in output"

    model_universal.print_stats()
    print(f"  ✅ Universal sharing works!")
    print()

    # Test 2: Alternating sharing
    print("Test 2: Alternating Layer Sharing")
    print("-" * 80)

    model_alternating = CompressedStructureNet(
        c_s=C,
        c_hidden=C * 2,
        num_layers=num_layers,
        sharing_strategy="alternating",
    )

    y = model_alternating(x)
    assert y.shape == (B, L, C), f"Shape mismatch: {y.shape}"

    model_alternating.print_stats()
    print(f"  ✅ Alternating sharing works!")
    print()

    # Test 3: Block sharing
    print("Test 3: Block-wise Sharing")
    print("-" * 80)

    model_block = CompressedStructureNet(
        c_s=C,
        c_hidden=C * 2,
        num_layers=num_layers,
        sharing_strategy="block",
        block_size=4,
    )

    y = model_block(x)
    assert y.shape == (B, L, C), f"Shape mismatch: {y.shape}"

    model_block.print_stats()
    print(f"  ✅ Block sharing works!")
    print()

    # Test 4: Bottleneck layers
    print("Test 4: Bottleneck Compression")
    print("-" * 80)

    model_bottleneck = CompressedStructureNet(
        c_s=C,
        c_hidden=C * 2,
        num_layers=num_layers,
        sharing_strategy="universal",
        use_bottleneck=True,
        bottleneck_ratio=4,
    )

    y = model_bottleneck(x)
    assert y.shape == (B, L, C), f"Shape mismatch: {y.shape}"

    model_bottleneck.print_stats()
    print(f"  ✅ Bottleneck compression works!")
    print()

    # Test 5: Comparison
    print("Test 5: Compression Comparison")
    print("-" * 80)

    models = {
        "Baseline (no sharing)": (num_layers, 1.0),
        "Universal": (1, model_universal.get_compression_ratio()),
        "Alternating": (2, model_alternating.get_compression_ratio()),
        "Block (4)": (num_layers // 4, model_block.get_compression_ratio()),
    }

    print(f"{'Strategy':<25} {'Unique Layers':<15} {'Compression':<15}")
    print("-" * 80)
    for name, (unique, compression) in models.items():
        print(f"{name:<25} {unique:<15} {compression:<15.2f}x")

    print()
    print(f"  ✅ Compression comparison complete!")
    print()

    print("=" * 80)
    print("🎉 All model compression tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_model_compression()
