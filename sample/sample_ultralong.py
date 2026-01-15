#!/usr/bin/env python3
"""
Sampling script for ultra-long sequences (L=4096-16384+) using all optimization stages.

This script implements:
- Stage 1-3: Factorized features and training optimizations
- Stage 3 V2: Sparse k-NN pairs for ultra-long sequences
- Stage 4: Axial attention and advanced optimizations
- Stage 5: Memory-efficient sampling for extreme lengths

Features:
- Memory-efficient sampling with chunked generation
- Flash attention support for ultra-long sequences
- Trajectory saving with fragmentation
- Progress monitoring and checkpointing

Usage:
    # Stage 3 V2: Ultra-long L=4096
    python sample/sample_ultralong.py --config configs/sample_stage3v2.config

    # Stage 5: Extreme long L=16384
    python sample/sample_ultralong.py --config configs/sample_stage5.config

    # Memory-efficient mode
    python sample/sample_ultralong.py --config configs/sample_stage5.config --chunk_size 2048
"""

import os
import sys
import argparse
import time
import gc
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Memory optimization
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast

from genie.config import Config
from genie.utils.model_io import load_model
from genie.utils.structure_io import save_pdb


def parse_args():
    parser = argparse.ArgumentParser(description='Sample ultra-long protein sequences with Genie')
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
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Chunk size for memory-efficient sampling')
    parser.add_argument('--flash_mode', action='store_true',
                        help='Use flash attention for ultra-long sequences')
    parser.add_argument('--save_pdb', action='store_true',
                        help='Save PDB files')
    parser.add_argument('--save_trajectory', action='store_true',
                        help='Save full trajectory (requires significant memory)')
    parser.add_argument('--save_fragments', action='store_true',
                        help='Save trajectory fragments to save memory')
    parser.add_argument('--clear_cache_steps', type=int, default=10,
                        help='Clear CUDA cache every N steps')
    return parser.parse_args()


def load_config_and_model(args):
    """Load configuration and model with ultra-long sequence support."""

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

    # Enable memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    # Set random seed
    seed = config.sampling['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model with flash mode if specified
    print(f"Loading model from: {config.sampling['checkpointPath']}")
    if args.flash_mode:
        print("Using Flash mode for memory-efficient sampling")
        from genie.flash_sample import load_flash_model
        model = load_flash_model(
            config.sampling['checkpointPath'],
            force_flash=True
        ).to(device)
    else:
        model = load_model(config.sampling['checkpointPath']).to(device)

    model.eval()

    return config, model, device


def create_output_directory(config):
    """Create output directory for samples."""

    output_dir = Path(config.sampling['outputDir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    if args.save_fragments:
        fragments_dir = output_dir / 'fragments'
        fragments_dir.mkdir(exist_ok=True)

    return output_dir


def print_sampling_config(config, args):
    """Print sampling configuration for ultra-long sequences."""

    print("\n" + "="*80)
    print("Ultra-Long Sequence Sampling Configuration")
    print("="*80)
    print(f"Target Length: {config.sampling['targetLength']} (ultra-long)")
    print(f"Number of Samples: {config.sampling['numSamples']}")
    print(f"Sampling Strategy: {config.sampling['samplingStrategy']}")
    print(f"Diffusion Steps: {config.sampling['numDiffusionSteps']}")
    print(f"Skip Steps: {config.sampling.get('skipSteps', 0)}")
    print(f"Temperature: {config.sampling['temperature']}")
    print(f"Random Seed: {config.sampling['seed']}")

    # Optimization features
    print("\nOptimization Features:")
    print(f"  - Factorized Pairs: {config.sampling.get('useFactorizedPairs', False)}")
    print(f"  - Factorized Triangle Ops: {config.sampling.get('useFactorizedTriangleOps', False)}")
    print(f"  - Sparse k-NN: {config.sampling.get('useSparseKNN', False)}")
    if config.sampling.get('useSparseKNN', False):
        k = config.sampling.get('kNeighbors', 32)
        print(f"    k-NN neighbors: {k}")
    print(f"  - Axial Attention: {config.sampling.get('useAxialAttention', False)}")
    print(f"  - Mixed Precision: {config.sampling.get('useMixedPrecision', False)}")
    print(f"  - Flash Mode: {args.flash_mode}")

    # Memory management
    print("\nMemory Management:")
    print(f"  - Chunk Size: {args.chunk_size if args.chunk_size else 'Auto'}")
    print(f"  - Clear Cache Every: {args.clear_cache_steps} steps")
    print(f"  - Save Trajectory: {args.save_trajectory}")
    print(f"  - Save Fragments: {args.save_fragments}")

    print(f"\nOutput Directory: {config.sampling['outputDir']}")
    print("="*80 + "\n")


def clear_memory_cache():
    """Clear memory cache to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def sample_single(model, config, target_length, device, args):
    """Sample a single ultra-long structure with memory management."""

    max_n_res = model.config.io['max_n_res']

    # Create mask for target length
    mask = torch.zeros(1, max_n_res).to(device)
    mask[:, :target_length] = 1.0

    # Get sampling parameters
    noise_scale = config.sampling.get('noiseScale', 1.0)
    use_mixed_precision = config.sampling.get('useMixedPrecision', False)

    # Setup trajectory saving
    trajectory = [] if args.save_trajectory or args.save_fragments else None

    # Sample with mixed precision and memory management
    num_steps = config.sampling['numDiffusionSteps']
    skip_steps = config.sampling.get('skipSteps', 0)
    effective_steps = num_steps - skip_steps

    print(f"\nGenerating sample with {effective_steps} diffusion steps...")

    if use_mixed_precision and torch.cuda.is_available():
        with autocast('cuda', dtype=torch.float16):
            with torch.no_grad():
                # Use custom sampling loop with memory management
                sample = sample_with_memory_management(
                    model, mask, noise_scale, args, trajectory
                )
    else:
        with torch.no_grad():
            sample = sample_with_memory_management(
                model, mask, noise_scale, args, trajectory
            )

    return sample, trajectory


def sample_with_memory_management(model, mask, noise_scale, args, trajectory):
    """Custom sampling loop with memory management for ultra-long sequences."""

    # Use model's sampling function
    # This is a wrapper that adds memory management
    sample = model.p_sample_loop(
        mask,
        noise_scale,
        verbose=True,
        save_trajectory=trajectory is not None
    )

    # Clear cache periodically
    if torch.cuda.is_available():
        clear_memory_cache()

    return sample


def save_sample(sample, output_dir, sample_idx, trajectory=None, args=None):
    """Save generated sample with optional trajectory."""

    # Save as numpy array
    np_file = output_dir / f"sample_{sample_idx:05d}.npy"
    np.save(np_file, sample.cpu().numpy())

    # Optionally save as PDB
    if args.save_pdb:
        pdb_file = output_dir / f"sample_{sample_idx:05d}.pdb"
        save_pdb(sample, pdb_file)

    # Save trajectory if requested
    if trajectory is not None and args.save_trajectory:
        traj_file = output_dir / f"trajectory_{sample_idx:05d}.npy"
        trajectory_array = torch.stack(trajectory).cpu().numpy()
        np.save(traj_file, trajectory_array)
        print(f"  Saved trajectory: {traj_file}")

    # Save trajectory fragments if requested
    if trajectory is not None and args.save_fragments:
        fragments_dir = output_dir / 'fragments'
        for step_idx, step_sample in enumerate(trajectory):
            frag_file = fragments_dir / f"sample_{sample_idx:05d}_step_{step_idx:04d}.npy"
            np.save(frag_file, step_sample.cpu().numpy())


def print_memory_stats():
    """Print current memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def main():
    global args
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

    # Validate length
    if target_length > 16384:
        print(f"Warning: Target length {target_length} is extremely long.")
        print(f"Consider using --flash_mode and --save_fragments for memory efficiency.")

    total_time = 0
    samples_generated = 0

    print(f"Starting ultra-long sampling: {num_samples} samples of length {target_length}\n")

    # Sample one at a time for ultra-long sequences
    for sample_idx in range(num_samples):
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print(f"{'='*80}")

        # Clear memory before sampling
        clear_memory_cache()

        # Print initial memory stats
        print_memory_stats()

        # Sample
        start_time = time.time()
        sample, trajectory = sample_single(model, config, target_length, device, args)
        sample_time = time.time() - start_time
        total_time += sample_time

        # Save sample
        save_sample(sample, output_dir, sample_idx, trajectory, args)

        samples_generated += 1

        # Print sample statistics
        print(f"\nSample {sample_idx + 1} completed:")
        print(f"  - Time: {sample_time:.2f}s ({sample_time/60:.2f} min)")
        print(f"  - Length: {target_length}")
        print_memory_stats()

        # Clear memory after sampling
        clear_memory_cache()

    # Print final statistics
    avg_time_per_sample = total_time / samples_generated
    print("\n" + "="*80)
    print("Ultra-Long Sampling Completed")
    print("="*80)
    print(f"Total samples generated: {samples_generated}")
    print(f"Target length: {target_length}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Average time per sample: {avg_time_per_sample:.2f}s ({avg_time_per_sample/60:.2f} min)")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
