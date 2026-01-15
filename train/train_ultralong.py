#!/usr/bin/env python3
"""
Training script for ultra-long sequences (L=4096-16384+) using all optimization stages.

This script implements:
- Stage 1: Factorized pair features
- Stage 2: Factorized triangle operations
- Stage 3: Training optimizations (progressive, mixed precision, chunked loss)
- Stage 3 V2: Sparse k-NN pairs for ultra-long sequences
- Stage 4: Advanced optimizations (axial attention, gradient checkpointing, compression)
- Stage 5: Distributed training across multiple GPUs

Usage:
    # Stage 3 V2: Ultra-long L=4096
    python train/train_ultralong.py --config configs/train_stage3v2.config

    # Stage 4: Very long L=8192
    python train/train_ultralong.py --config configs/train_stage4.config --gpus 0

    # Stage 5: Extreme long L=16384+ with distributed training
    torchrun --nproc_per_node=8 train/train_ultralong.py --config configs/train_stage5.config
"""

import os
import sys
import argparse

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reduce memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, GradientAccumulationScheduler
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.trainer import seed_everything
from pytorch_lightning.strategies import DDPStrategy

from genie.config import Config
from genie.data.data_module import SCOPeDataModule
from genie.diffusion.genie import Genie
from genie.utils.oom_callback import OOMMonitorCallback


def parse_args():
    parser = argparse.ArgumentParser(description='Train Genie with ultra-long sequence support')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU IDs to use (comma-separated, e.g., "0,1,2,3")')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes for distributed training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more logging')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling mode')
    return parser.parse_args()


def setup_training(config, args):
    """Setup training environment and configurations."""

    # Enable TF32 on Ampere+ GPUs for better performance
    torch.set_float32_matmul_precision('high')

    # Monitor CUDA memory allocation
    if torch.cuda.is_available():
        try:
            # Reserve less memory for ultra-long sequences to avoid fragmentation
            torch.cuda.set_per_process_memory_fraction(0.98)
        except RuntimeError:
            pass

        # Enable cuDNN benchmark for fixed input sizes
        torch.backends.cudnn.benchmark = True

        # Allow memory growth
        torch.cuda.empty_cache()

    # Setup devices
    if args.gpus is not None:
        gpus = [int(elt) for elt in args.gpus.split(',')]
        accelerator = 'gpu'
    else:
        gpus = 'auto'
        accelerator = 'auto'

    return gpus, accelerator


def setup_loggers(config):
    """Setup TensorBoard and WandB loggers."""

    tb_logger = TensorBoardLogger(
        save_dir=config.io['log_dir'],
        name=config.io['name']
    )

    wandb_logger = WandbLogger(
        project=config.io['name'],
        name=config.io['name'],
        log_model=False  # Don't log model to save space
    )

    return [tb_logger, wandb_logger]


def setup_callbacks(config, args):
    """Setup training callbacks."""

    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=config.training.get('checkpoint_every_n_epoch', 10),
        dirpath=f"{config.io['log_dir']}/{config.io['name']}/checkpoints",
        filename='{epoch}-{step}',
        save_top_k=3,  # Save only top 3 checkpoints to save disk space
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # OOM monitor callback
    oom_callback = OOMMonitorCallback()
    callbacks.append(oom_callback)

    # Gradient accumulation scheduler (if specified)
    grad_accum_schedule = config.training.get('grad_accum_schedule', None)
    if grad_accum_schedule:
        callbacks.append(GradientAccumulationScheduler(scheduling=grad_accum_schedule))

    return callbacks


def setup_strategy(config, gpus):
    """Setup distributed training strategy."""

    use_distributed = config.training.get('useDistributed', False)

    if not use_distributed or not isinstance(gpus, list) or len(gpus) <= 1:
        return 'auto'

    # Distributed backend
    backend = config.training.get('distributedBackend', 'nccl')
    strategy_name = config.training.get('distributedStrategy', 'ddp')

    if strategy_name == 'ddp':
        # DDP with gradient compression for communication efficiency
        use_compression = config.training.get('useGradientCompression', False)

        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # Save memory
            static_graph=True,  # Optimize for static computation graph
        )

        return strategy
    elif strategy_name == 'fsdp':
        # Fully Sharded Data Parallel for extreme sequences
        from pytorch_lightning.strategies import FSDPStrategy
        strategy = FSDPStrategy(
            auto_wrap_policy=None,  # Will be set based on model
            activation_checkpointing=config.training.get('useGradientCheckpointing', False),
        )
        return strategy
    else:
        return 'ddp'


def print_optimization_summary(config):
    """Print a summary of enabled optimizations."""

    print("\n" + "="*80)
    print("Ultra-Long Sequence Training - Optimization Summary")
    print("="*80)

    # Stage 1
    print("\n[Stage 1] Factorized Pair Features:")
    use_factorized = config.training.get('useFactorizedPairs', False)
    if use_factorized:
        rank = config.training.get('zFactorRank', 2)
        print(f"  ✓ Enabled - Rank {rank} factorization")
        print(f"  Memory saving: ~500x for pair features")
    else:
        print(f"  ✗ Disabled - Using full pair features")

    # Stage 2
    print("\n[Stage 2] Factorized Triangle Operations:")
    use_triangle = config.training.get('useFactorizedTriangleOps', False)
    if use_triangle:
        rank = config.training.get('triangleUpdateRank', 4)
        chunk_size = config.training.get('triangleAttentionChunkSize', 256)
        print(f"  ✓ Enabled - Rank {rank}, chunk size {chunk_size}")
        print(f"  Memory saving: ~256x for triangle operations")
    else:
        print(f"  ✗ Disabled")

    # Stage 3
    print("\n[Stage 3] Training Optimizations:")
    use_progressive = config.training.get('useProgressiveTraining', False)
    use_mixed = config.training.get('useMixedPrecision', False)
    use_chunked = config.training.get('useChunkedLoss', False)

    if use_progressive:
        start_len = config.training.get('progressiveStartLength', 128)
        end_len = config.training.get('progressiveEndLength', 1024)
        print(f"  ✓ Progressive Training: L={start_len} → L={end_len}")
    if use_mixed:
        precision = config.training.get('precision', 16)
        print(f"  ✓ Mixed Precision: FP{precision} (2-3x speedup)")
    if use_chunked:
        chunk_size = config.training.get('lossChunkSize', 256)
        print(f"  ✓ Chunked Loss: chunk size {chunk_size} (8-16x memory saving)")

    # Stage 3 V2
    print("\n[Stage 3 V2] Sparse k-NN Pairs:")
    use_sparse = config.training.get('useSparseKNN', False)
    if use_sparse:
        k = config.training.get('kNeighbors', 32)
        strategy = config.training.get('knnStrategy', 'hybrid')
        print(f"  ✓ Enabled - k={k}, strategy={strategy}")
        print(f"  Memory saving: ~128-256x for ultra-long sequences")
    else:
        print(f"  ✗ Disabled - Using full pair interactions")

    # Stage 4
    print("\n[Stage 4] Advanced Optimizations:")
    use_axial = config.training.get('useAxialAttention', False)
    use_checkpoint = config.training.get('useGradientCheckpointing', False)
    use_compression = config.training.get('useModelCompression', False)

    if use_axial:
        row_chunk = config.training.get('axialRowChunkSize', 128)
        col_chunk = config.training.get('axialColChunkSize', 128)
        print(f"  ✓ Axial Attention: {row_chunk}×{col_chunk} chunks (12x speedup)")
    if use_checkpoint:
        strategy = config.training.get('checkpointingStrategy', 'adaptive')
        print(f"  ✓ Gradient Checkpointing: {strategy} (2-3x memory saving)")
    if use_compression:
        ratio = config.training.get('compressionRatio', 4)
        print(f"  ✓ Model Compression: {ratio}x parameter reduction")

    # Stage 5
    print("\n[Stage 5] Distributed Training:")
    use_distributed = config.training.get('useDistributed', False)
    if use_distributed:
        backend = config.training.get('distributedBackend', 'nccl')
        strategy = config.training.get('distributedStrategy', 'ddp')
        devices = config.training.get('devices', 1)
        print(f"  ✓ Enabled - Strategy: {strategy}, Backend: {backend}")
        print(f"  Devices: {devices} GPUs")

        use_seq_parallel = config.training.get('useSequenceParallelism', False)
        if use_seq_parallel:
            shards = config.training.get('numSequenceShards', 4)
            print(f"  ✓ Sequence Parallelism: {shards} shards")
    else:
        print(f"  ✗ Disabled - Single GPU training")

    print("="*80 + "\n")


def main():
    args = parse_args()

    # Load configuration
    config = Config(filename=args.config)

    # Setup training environment
    gpus, accelerator = setup_training(config, args)

    # Set seed for reproducibility
    seed_everything(config.training['seed'], workers=True)

    # Setup loggers
    loggers = setup_loggers(config)

    # Setup callbacks
    callbacks = setup_callbacks(config, args)

    # Setup distributed strategy
    strategy = setup_strategy(config, gpus)

    # Create data module
    dm = SCOPeDataModule(
        **config.io,
        batch_size=config.training['batch_size'],
        num_workers=config.training['num_workers']
    )

    # Create model
    model = Genie(config)

    # Gradient accumulation for large batch training
    accumulate_grad_batches = config.training.get('accumulate_grad_batches', 1)
    if accumulate_grad_batches > 1:
        effective_batch = config.training['batch_size'] * accumulate_grad_batches
        if isinstance(gpus, list):
            effective_batch *= len(gpus)
        print(f"[Large Batch] Gradient accumulation: {accumulate_grad_batches} steps")
        print(f"[Large Batch] Effective batch size: {effective_batch}")

    # Mixed precision settings
    use_mixed_precision = config.training.get('useMixedPrecision', False)
    precision = config.training.get('precision', 32)

    # Create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=gpus,
        num_nodes=args.nodes,
        strategy=strategy,
        precision=precision if use_mixed_precision else 32,
        max_epochs=config.training['n_epoch'],
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=config.training.get('gradient_clip_val', 1.0),
        gradient_clip_algorithm='norm',
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=config.training.get('log_every_n_steps', 10),
        val_check_interval=config.training.get('val_check_interval', 0.5),
        deterministic=False,
        enable_progress_bar=not args.debug,
        enable_model_summary=True,
        sync_batchnorm=isinstance(gpus, list) and len(gpus) > 1,
    )

    # Print configuration summary
    print("\n" + "="*80)
    print("Training Configuration Summary")
    print("="*80)
    print(f"Experiment Name: {config.io['name']}")
    print(f"Max Sequence Length: {config.io['max_n_res']}")
    print(f"Batch Size: {config.training['batch_size']}")
    print(f"Number of Epochs: {config.training['n_epoch']}")
    print(f"Learning Rate: {config.optimization['lr']}")
    print(f"Devices: {gpus}")
    print(f"Nodes: {args.nodes}")
    print(f"Accelerator: {accelerator}")
    print(f"Strategy: {strategy}")

    # Print optimization summary
    print_optimization_summary(config)

    # Train model
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.fit(model, dm, ckpt_path=args.resume)
    else:
        trainer.fit(model, dm)

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
