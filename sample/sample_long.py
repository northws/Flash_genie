#!/usr/bin/env python3
"""
Sampling script for long sequences (L=1024-1536) using Stage 3 optimizations.

This script implements:
- Factorized pair features (Stage 1)
- Factorized triangle operations (Stage 2)
- Mixed precision sampling for speed (Stage 3)

Usage:
    # Fast sampling with DDIM
    python sample/sample_long.py --config configs/sample_stage3.config

    # High quality sampling with DDPM
    python sample/sample_long.py --config configs/sample_stage3.config --ddpm --skip_steps 0

    # Batch sampling
    python sample/sample_long.py --config configs/sample_stage3.config --num_samples 100
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast

from genie.config import Config
from genie.utils.model_io import load_model
from genie.utils.structure_io import save_pdb


def parse_args():
    parser = argparse.ArgumentParser(description='Sample long protein sequences with Genie')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to sampling configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to generate (overrides config)')
    parser.add_argument('--length', type=int, default=None,
                        help='Target sequence length (overrides config)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for parallel sampling')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--ddpm', action='store_true',
                        help='Use DDPM instead of DDIM (slower but higher quality)')
    parser.add_argument('--skip_steps', type=int, default=None,
                        help='Number of steps to skip (overrides config)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--save_pdb', action='store_true',
                        help='Save PDB files (may use significant disk space)')
    parser.add_argument('--save_trajectory', action='store_true',
                        help='Save full trajectory (requires significant memory)')
    return parser.parse_args()


def load_config_and_model(args):
    """Load configuration and model."""

    # Load configuration
    config = Config(filename=args.config)

    # Override config with command line arguments
    if args.checkpoint:
        config.sampling['checkpointPath'] = args.checkpoint
    if args.output_dir:
        config.sampling['outputDir'] = args.output_dir
    if args.num_samples:
        config.sampling['numSamples'] = args.num_samples
    if args.length:
        config.sampling['targetLength'] = args.length
    if args.ddpm:
        config.sampling['samplingStrategy'] = 'ddpm'
    if args.skip_steps is not None:
        config.sampling['skipSteps'] = args.skip_steps
    if args.temperature:
        config.sampling['temperature'] = args.temperature
    if args.seed:
        config.sampling['seed'] = args.seed

    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Set random seed
    seed = config.sampling['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model
    print(f"Loading model from: {config.sampling['checkpointPath']}")
    model = load_model(config.sampling['checkpointPath']).to(device)
    model.eval()

    return config, model, device


def create_output_directory(config):
    """Create output directory for samples."""

    output_dir = Path(config.sampling['outputDir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def print_sampling_config(config, args):
    """Print sampling configuration."""

    print("\n" + "="*80)
    print("Sampling Configuration")
    print("="*80)
    print(f"Target Length: {config.sampling['targetLength']}")
    print(f"Number of Samples: {config.sampling['numSamples']}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sampling Strategy: {config.sampling['samplingStrategy']}")
    print(f"Diffusion Steps: {config.sampling['numDiffusionSteps']}")
    print(f"Skip Steps: {config.sampling.get('skipSteps', 0)}")
    print(f"Temperature: {config.sampling['temperature']}")
    print(f"Random Seed: {config.sampling['seed']}")

    # Optimization features
    print("\nOptimization Features:")
    print(f"  - Factorized Pairs: {config.sampling.get('useFactorizedPairs', False)}")
    print(f"  - Factorized Triangle Ops: {config.sampling.get('useFactorizedTriangleOps', False)}")
    print(f"  - Mixed Precision: {config.sampling.get('useMixedPrecision', False)}")

    print(f"\nOutput Directory: {config.sampling['outputDir']}")
    print("="*80 + "\n")


def sample_batch(model, config, batch_size, target_length, device):
    """Sample a batch of structures."""

    # Create mask for target length
    max_n_res = model.config.io['max_n_res']
    mask = torch.zeros(batch_size, max_n_res).to(device)
    mask[:, :target_length] = 1.0

    # Get sampling parameters
    noise_scale = config.sampling.get('noiseScale', 1.0)
    use_mixed_precision = config.sampling.get('useMixedPrecision', False)

    # Sample with mixed precision if enabled
    if use_mixed_precision and torch.cuda.is_available():
        with autocast('cuda', dtype=torch.float16):
            with torch.no_grad():
                samples = model.p_sample_loop(mask, noise_scale, verbose=True)
    else:
        with torch.no_grad():
            samples = model.p_sample_loop(mask, noise_scale, verbose=True)

    return samples


def save_samples(samples, output_dir, start_idx, save_pdb=False):
    """Save generated samples."""

    for i, sample in enumerate(samples):
        sample_idx = start_idx + i

        # Save as numpy array
        np_file = output_dir / f"sample_{sample_idx:05d}.npy"
        np.save(np_file, sample.cpu().numpy())

        # Optionally save as PDB
        if save_pdb:
            pdb_file = output_dir / f"sample_{sample_idx:05d}.pdb"
            save_pdb(sample, pdb_file)


def main():
    args = parse_args()

    # Load configuration and model
    config, model, device = load_config_and_model(args)

    # Create output directory
    output_dir = create_output_directory(config)

    # Print configuration
    print_sampling_config(config, args)

    # Get sampling parameters
    num_samples = config.sampling['numSamples']
    target_length = config.sampling['targetLength']
    batch_size = args.batch_size

    # Calculate number of batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    total_time = 0
    samples_generated = 0

    print(f"Starting sampling: {num_samples} samples in {num_batches} batches\n")

    # Sample in batches
    for batch_idx in tqdm(range(num_batches), desc="Batching"):
        # Calculate batch size for this batch (last batch may be smaller)
        current_batch_size = min(batch_size, num_samples - samples_generated)

        # Sample
        start_time = time.time()
        samples = sample_batch(model, config, current_batch_size, target_length, device)
        batch_time = time.time() - start_time
        total_time += batch_time

        # Save samples
        save_samples(
            samples,
            output_dir,
            samples_generated,
            save_pdb=args.save_pdb
        )

        samples_generated += current_batch_size

        # Print batch statistics
        time_per_sample = batch_time / current_batch_size
        print(f"\nBatch {batch_idx + 1}/{num_batches}:")
        print(f"  - Samples: {current_batch_size}")
        print(f"  - Time: {batch_time:.2f}s")
        print(f"  - Time per sample: {time_per_sample:.2f}s")

    # Print final statistics
    avg_time_per_sample = total_time / samples_generated
    print("\n" + "="*80)
    print("Sampling Completed")
    print("="*80)
    print(f"Total samples generated: {samples_generated}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {avg_time_per_sample:.2f}s")
    print(f"Throughput: {1.0/avg_time_per_sample:.2f} samples/s")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
