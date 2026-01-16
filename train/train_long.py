#!/usr/bin/env python3
"""
Training script for long sequences (L=1024-1536) using Stage 3 optimizations.

This script implements:
- Factorized pair features (Stage 1)
- Factorized triangle operations (Stage 2)
- Training optimizations (Stage 3): progressive training, mixed precision, chunked loss

Usage:
    python train/train_long.py --config configs/train_stage3.config
    python train/train_long.py --config configs/train_stage3.config --gpus 0,1
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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.trainer import seed_everything

from genie.config import Config
from genie.data.data_module import SCOPeDataModule
from genie.diffusion.genie import Genie
from genie.utils.oom_callback import OOMMonitorCallback


def parse_args():
    parser = argparse.ArgumentParser(description='Train Genie with long sequence support')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU IDs to use (comma-separated, e.g., "0,1")')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more logging')
    return parser.parse_args()


def setup_training(config, args):
    """Setup training environment and configurations."""

    # Enable TF32 on Ampere+ GPUs for better performance
    torch.set_float32_matmul_precision('medium')

    # Monitor CUDA memory allocation
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
        except RuntimeError:
            pass

        # Enable cuDNN benchmark for fixed input sizes
        torch.backends.cudnn.benchmark = True

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
        name=config.io['name']
    )

    return [tb_logger, wandb_logger]


def setup_callbacks(config):
    """Setup training callbacks."""

    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=config.training['checkpoint_every_n_epoch'],
        dirpath=f"{config.io['log_dir']}/{config.io['name']}/checkpoints",
        filename='{epoch}-{step}',
        save_top_k=-1,
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # OOM monitor callback
    oom_callback = OOMMonitorCallback()
    callbacks.append(oom_callback)

    return callbacks


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
    callbacks = setup_callbacks(config)

    # Create data module
    dm = SCOPeDataModule(
        **config.io,
        batch_size=config.training['batch_size'],
        num_workers=config.training['num_workers']
    )

    # Create model
    model = Genie(config)

    # Determine DDP strategy
    use_flash_mode = config.training.get('use_flash_mode', False)
    if use_flash_mode:
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'ddp' if isinstance(gpus, list) and len(gpus) > 1 else 'auto'

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
    if use_mixed_precision:
        print(f"[Mixed Precision] Training with FP{precision}")

    # Progressive training info
    use_progressive = config.training.get('useProgressiveTraining', False)
    if use_progressive:
        start_len = config.training.get('progressiveStartLength', 128)
        end_len = config.training.get('progressiveEndLength', 1024)
        print(f"[Progressive Training] Curriculum: L={start_len} → L={end_len}")

    # Create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=gpus,
        strategy=strategy,
        precision=precision if use_mixed_precision else 32,
        max_epochs=config.training['n_epoch'],
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=config.training.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=config.training.get('log_every_n_steps', 50),
        val_check_interval=config.training.get('val_check_interval', 1.0),
        deterministic=False,  # Allow cuDNN benchmark
        enable_progress_bar=not args.debug,
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
    print(f"Accelerator: {accelerator}")
    print(f"Strategy: {strategy}")

    # Feature flags
    print("\nOptimization Features:")
    print(f"  - Factorized Pairs: {config.training.get('useFactorizedPairs', False)}")
    print(f"  - Factorized Triangle Ops: {config.training.get('useFactorizedTriangleOps', False)}")
    print(f"  - Progressive Training: {use_progressive}")
    print(f"  - Mixed Precision: {use_mixed_precision} (FP{precision})")
    print(f"  - Chunked Loss: {config.training.get('useChunkedLoss', False)}")
    print("="*80 + "\n")

    # Train model
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.fit(model, dm, ckpt_path=args.resume)
    else:
        trainer.fit(model, dm)

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
