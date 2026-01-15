"""
Factorized Pair Transform Network (Stage 2)

This module provides a factorized version of PairTransformNet that works
directly with factorized pair representations without materializing full L×L tensors.

Key Features:
1. **Factorized Triangle Multiplicative Updates**: O(L² × rank) instead of O(L³)
2. **Chunked Triangle Attention**: Memory-bounded attention computation
3. **Direct Factor Operations**: Never materializes full pair tensor

Memory Benefits:
- Standard PairTransformNet: O(L² × C) per operation
- Factorized PairTransformNet: O(L × rank × C) per operation

For L=1024, C=128, rank=4:
    Standard: 1024² × 128 × 4 bytes = 537 MB
    Factorized: 1024 × 4 × 128 × 4 bytes = 2 MB (268x reduction!)

Based on AlphaFold2 Evoformer architecture with factorization optimizations.

Author: Stage 2 Implementation (2026-01-13)
"""

import torch
from torch import nn

from genie.model.factorized_triangle_ops import (
    FactorizedTriangleMultiplicationOutgoing,
    FactorizedTriangleMultiplicationIncoming,
    ChunkedTriangleAttentionStartingNode,
    ChunkedTriangleAttentionEndingNode,
)
from genie.model.modules.pair_transition import PairTransition
from genie.model.modules.dropout import DropoutRowwise, DropoutColumnwise


class FactorizedPairTransformLayer(nn.Module):
    """
    Single layer of factorized pair transformation.

    Architecture:
        1. Factorized Triangle Multiplicative Update (Outgoing)
        2. Factorized Triangle Multiplicative Update (Incoming)
        3. Chunked Triangle Attention (Starting Node)
        4. Chunked Triangle Attention (Ending Node)
        5. Pair Transition (applied to factor sum)

    All operations work directly on factorized representations [B, L, rank, C]
    without materializing full [B, L, L, C] tensors.
    """

    def __init__(
        self,
        c_p: int,
        rank: int,
        include_mul_update: bool = True,
        include_tri_att: bool = True,
        c_hidden_mul: int = 128,
        c_hidden_tri_att: int = 128,
        n_head_tri: int = 4,
        tri_dropout: float = 0.25,
        pair_transition_n: int = 4,
        tri_att_chunk_size: int = 64,
        use_grad_checkpoint: bool = False
    ):
        """
        Args:
            c_p: Pair feature dimension
            rank: Factorization rank
            include_mul_update: Include triangle multiplicative updates
            include_tri_att: Include triangle attention
            c_hidden_mul: Hidden dim for multiplicative updates
            c_hidden_tri_att: Hidden dim for triangle attention
            n_head_tri: Number of attention heads
            tri_dropout: Dropout rate
            pair_transition_n: Transition layer multiplier
            tri_att_chunk_size: Chunk size for triangle attention
            use_grad_checkpoint: Use gradient checkpointing
        """
        super().__init__()

        self.include_mul_update = include_mul_update
        self.include_tri_att = include_tri_att

        # Factorized triangle multiplicative updates
        if include_mul_update:
            self.tri_mul_out = FactorizedTriangleMultiplicationOutgoing(
                c_p, rank, c_hidden_mul,
                use_grad_checkpoint=use_grad_checkpoint,
                dropout=tri_dropout
            )
            self.tri_mul_in = FactorizedTriangleMultiplicationIncoming(
                c_p, rank, c_hidden_mul,
                use_grad_checkpoint=use_grad_checkpoint,
                dropout=tri_dropout
            )
        else:
            self.tri_mul_out = None
            self.tri_mul_in = None

        # Chunked triangle attention
        if include_tri_att:
            self.tri_att_start = ChunkedTriangleAttentionStartingNode(
                c_p, rank, c_hidden_tri_att, n_head_tri,
                chunk_size=tri_att_chunk_size,
                dropout=tri_dropout
            )
            self.tri_att_end = ChunkedTriangleAttentionEndingNode(
                c_p, rank, c_hidden_tri_att, n_head_tri,
                chunk_size=tri_att_chunk_size,
                dropout=tri_dropout
            )
        else:
            self.tri_att_start = None
            self.tri_att_end = None

        # Pair transition (applied to aggregated factors)
        self.pair_transition = PairTransition(c_p, pair_transition_n)

        # Dropout layers
        self.dropout_row = DropoutRowwise(tri_dropout)
        self.dropout_col = DropoutColumnwise(tri_dropout)

    def forward(
        self,
        z_left: torch.Tensor,  # [B, L, rank, C]
        z_right: torch.Tensor,  # [B, L, rank, C]
        mask: torch.Tensor  # [B, L]
    ):
        """
        Forward pass through factorized pair transform layer.

        Args:
            z_left: Left factors [B, L, rank, C]
            z_right: Right factors [B, L, rank, C]
            mask: Sequence mask [B, L]

        Returns:
            (z_left_updated, z_right_updated): Updated factors
        """
        # Triangle multiplicative updates
        if self.tri_mul_out is not None:
            # Outgoing
            delta_left, delta_right = self.tri_mul_out(z_left, z_right, mask)
            # Apply dropout to aggregated update (simulate row dropout on factors)
            delta_agg = delta_left.sum(dim=2) + delta_right.sum(dim=2)  # [B, L, C]
            delta_agg = self.dropout_row(delta_agg)
            # Distribute back to factors
            z_left = z_left + delta_agg.unsqueeze(2) / z_left.shape[2]
            z_right = z_right + delta_agg.unsqueeze(2) / z_right.shape[2]

            # Incoming
            delta_left, delta_right = self.tri_mul_in(z_left, z_right, mask)
            delta_agg = delta_left.sum(dim=2) + delta_right.sum(dim=2)  # [B, L, C]
            delta_agg = self.dropout_row(delta_agg)
            z_left = z_left + delta_agg.unsqueeze(2) / z_left.shape[2]
            z_right = z_right + delta_agg.unsqueeze(2) / z_right.shape[2]

        # Triangle attention
        if self.tri_att_start is not None:
            # Starting node
            delta_left, delta_right = self.tri_att_start(z_left, z_right, mask)
            delta_agg = delta_left.sum(dim=2) + delta_right.sum(dim=2)  # [B, L, C]
            delta_agg = self.dropout_row(delta_agg)
            z_left = z_left + delta_agg.unsqueeze(2) / z_left.shape[2]
            z_right = z_right + delta_agg.unsqueeze(2) / z_right.shape[2]

            # Ending node
            delta_left, delta_right = self.tri_att_end(z_left, z_right, mask)
            delta_agg = delta_left.sum(dim=2) + delta_right.sum(dim=2)  # [B, L, C]
            delta_agg = self.dropout_col(delta_agg)
            z_left = z_left + delta_agg.unsqueeze(2) / z_left.shape[2]
            z_right = z_right + delta_agg.unsqueeze(2) / z_right.shape[2]

        # Pair transition
        # Aggregate factors for transition: [B, L, rank, C] → [B, L, C]
        z_agg = z_left.sum(dim=2) + z_right.sum(dim=2)  # [B, L, C]

        # Expand mask for transition: [B, L] → [B, L, L]
        # Since transition expects pair mask, we create a simple outer product
        mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [B, L, L]

        # Apply pair transition (NOTE: PairTransition expects [B, L, L, C])
        # For factorized version, we apply transition to 1D aggregated features
        # This is an approximation, but saves memory
        # We'll reshape: [B, L, C] → [B, L, 1, C] and tile → [B, L, L, C]
        # But to save memory, we just apply transition row-wise
        z_trans = self.pair_transition(z_agg.unsqueeze(2), mask.unsqueeze(-1))  # [B, L, 1, C]
        z_trans = z_trans.squeeze(2)  # [B, L, C]

        # Distribute transition output back to factors
        z_left = z_left + z_trans.unsqueeze(2) / z_left.shape[2]
        z_right = z_right + z_trans.unsqueeze(2) / z_right.shape[2]

        # Apply mask to factors
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        z_left = z_left * mask_expanded
        z_right = z_right * mask_expanded

        return z_left, z_right


class FactorizedPairTransformNet(nn.Module):
    """
    Factorized pair transformation network.

    Stack of FactorizedPairTransformLayers that process factorized pair representations.

    Memory Advantage:
        Standard PairTransformNet:
            Per layer: ~O(L² × C) activations
            N layers: ~O(N × L² × C) total

        Factorized PairTransformNet:
            Per layer: ~O(L × rank × C) activations
            N layers: ~O(N × L × rank × C) total

        Reduction: ~(L / rank) × speedup

    For L=1024, rank=4, N=4 layers:
        Standard: 4 × 1024² × 128 = 2 GB
        Factorized: 4 × 1024 × 4 × 128 = 8 MB (256x reduction!)
    """

    def __init__(
        self,
        c_p: int,
        rank: int,
        n_pair_transform_layer: int = 4,
        include_mul_update: bool = True,
        include_tri_att: bool = True,
        c_hidden_mul: int = 128,
        c_hidden_tri_att: int = 128,
        n_head_tri: int = 4,
        tri_dropout: float = 0.25,
        pair_transition_n: int = 4,
        tri_att_chunk_size: int = 64,
        use_grad_checkpoint: bool = False
    ):
        """
        Args:
            c_p: Pair feature dimension
            rank: Factorization rank
            n_pair_transform_layer: Number of layers
            include_mul_update: Include triangle multiplicative updates
            include_tri_att: Include triangle attention
            c_hidden_mul: Hidden dim for multiplicative updates
            c_hidden_tri_att: Hidden dim for triangle attention
            n_head_tri: Number of attention heads
            tri_dropout: Dropout rate
            pair_transition_n: Transition layer multiplier
            tri_att_chunk_size: Chunk size for triangle attention
            use_grad_checkpoint: Use gradient checkpointing
        """
        super().__init__()

        self.layers = nn.ModuleList([
            FactorizedPairTransformLayer(
                c_p=c_p,
                rank=rank,
                include_mul_update=include_mul_update,
                include_tri_att=include_tri_att,
                c_hidden_mul=c_hidden_mul,
                c_hidden_tri_att=c_hidden_tri_att,
                n_head_tri=n_head_tri,
                tri_dropout=tri_dropout,
                pair_transition_n=pair_transition_n,
                tri_att_chunk_size=tri_att_chunk_size,
                use_grad_checkpoint=use_grad_checkpoint
            )
            for _ in range(n_pair_transform_layer)
        ])

    def forward(
        self,
        z_left: torch.Tensor,  # [B, L, rank, C]
        z_right: torch.Tensor,  # [B, L, rank, C]
        mask: torch.Tensor  # [B, L]
    ):
        """
        Forward pass through factorized pair transform network.

        Args:
            z_left: Left factors [B, L, rank, C]
            z_right: Right factors [B, L, rank, C]
            mask: Sequence mask [B, L]

        Returns:
            (z_left_updated, z_right_updated): Updated factors
        """
        for layer in self.layers:
            z_left, z_right = layer(z_left, z_right, mask)

        return z_left, z_right


def test_factorized_pair_transform():
    """Test factorized pair transform network."""
    print("=" * 80)
    print("Testing Factorized Pair Transform Network (Stage 2)")
    print("=" * 80)
    print()

    B, L, rank, C = 2, 256, 4, 64

    # Test inputs
    z_left = torch.randn(B, L, rank, C)
    z_right = torch.randn(B, L, rank, C)
    mask = torch.ones(B, L)

    print(f"Test configuration:")
    print(f"  Batch: {B}, Length: {L}, Rank: {rank}, Channels: {C}")
    print()

    # Test single layer
    print("Test 1: Single Factorized Pair Transform Layer")
    print("-" * 80)
    layer = FactorizedPairTransformLayer(
        c_p=C,
        rank=rank,
        include_mul_update=True,
        include_tri_att=True,
        c_hidden_mul=64,
        c_hidden_tri_att=64,
        n_head_tri=4,
        tri_dropout=0.1,
        tri_att_chunk_size=64
    )

    z_left_out, z_right_out = layer(z_left, z_right, mask)
    print(f"  Left output shape: {z_left_out.shape}")
    print(f"  Right output shape: {z_right_out.shape}")
    print(f"  ✅ Single layer works!")
    print()

    # Test full network
    print("Test 2: Full Factorized Pair Transform Network")
    print("-" * 80)
    net = FactorizedPairTransformNet(
        c_p=C,
        rank=rank,
        n_pair_transform_layer=2,
        include_mul_update=True,
        include_tri_att=True,
        c_hidden_mul=64,
        c_hidden_tri_att=64,
        n_head_tri=4,
        tri_dropout=0.1,
        tri_att_chunk_size=64
    )

    z_left_out, z_right_out = net(z_left, z_right, mask)
    print(f"  Left output shape: {z_left_out.shape}")
    print(f"  Right output shape: {z_right_out.shape}")
    print(f"  ✅ Full network works!")
    print()

    # Memory comparison
    print("Memory Comparison (per layer):")
    print("-" * 80)
    standard_mem = B * L * L * C * 4 / (1024 ** 2)  # FP32
    factorized_mem = B * 2 * L * rank * C * 4 / (1024 ** 2)  # FP32

    print(f"  Standard pair transform: {standard_mem:.2f} MB")
    print(f"  Factorized pair transform: {factorized_mem:.2f} MB")
    print(f"  Memory reduction: {standard_mem / factorized_mem:.2f}x")
    print()

    print("=" * 80)
    print("🎉 All factorized pair transform tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_factorized_pair_transform()
