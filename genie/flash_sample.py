"""
Flash Sample - Memory-Efficient Sampling for Long Sequences

This module provides a memory-efficient sampling script that uses Flash IPA
for generating protein backbone structures from long sequences.

Key features:
- Supports both standard and Flash mode models
- Memory-efficient for sequences > 512 residues
- Compatible with existing model checkpoints
- Parallel sampling for different sequence lengths
"""

import os
import sys

# Add the project root to sys.path to enable imports from the 'genie' package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm, trange
from torch.amp import autocast, GradScaler

from genie.config import Config
from genie.diffusion.genie import Genie
from genie.utils.model_io import get_versions, get_epochs


def load_flash_model(rootdir, name, version=None, epoch=None, force_flash=False):
    """
    Load a Genie model with optional Flash mode override.
    
    Args:
        rootdir: Root directory containing model checkpoints
        name: Model name
        version: Model version (None for latest)
        epoch: Model epoch (None for latest)
        force_flash: If True, force Flash mode even if not specified in config
    
    Returns:
        Genie model instance
    """
    import glob
    
    # Load configuration
    basedir = os.path.join(rootdir, name)
    config_filepath = os.path.join(basedir, 'configuration')
    config = Config(config_filepath)
    
    # Override Flash mode if requested
    if force_flash:
        print("Force enabling Flash mode for sampling...")
        config.training['use_flash_mode'] = True
        # Ensure Flash-specific parameters are set
        if 'z_factor_rank' not in config.model:
            config.model['z_factor_rank'] = 2
        if 'k_neighbors' not in config.model:
            config.model['k_neighbors'] = 10
    
    # Check for latest version if needed
    available_versions = get_versions(rootdir, name)
    if version is None:
        if len(available_versions) == 0:
            print('No checkpoint available (version)')
            sys.exit(0)
        version = np.max(available_versions)
    else:
        if version not in available_versions:
            print('Missing checkpoint version: {}'.format(version))
            sys.exit(0)
    
    # Check for latest epoch if needed
    available_epochs = get_epochs(rootdir, name, version)
    if epoch is None:
        if len(available_epochs) == 0:
            print('No checkpoint available (epoch)')
            sys.exit(0)
        epoch = np.max(available_epochs)
    else:
        if epoch not in available_epochs:
            print('Missing checkpoint epoch: {}'.format(epoch))
            print('Available epochs: {}'.format(available_epochs))
            sys.exit(0)
    
    # Find checkpoint file - handle special filename format epoch=epoch=XXX
    ckpt_filepath = None
    
    # Search in multiple locations
    possible_base_paths = [
        os.path.join(basedir, 'version_{}'.format(version), 'checkpoints'),
        os.path.join(basedir, 'version_{}'.format(version)),
        os.path.join(basedir, 'checkpoints'),
        basedir,
    ]
    
    # Try both possible filename patterns
    for base_path in possible_base_paths:
        # Pattern 1: epoch=epoch=XXXX (special format)
        pattern1 = os.path.join(base_path, 'epoch=epoch={}*.ckpt'.format(str(epoch).zfill(4)))
        found = glob.glob(pattern1)
        if found:
            ckpt_filepath = found[0]
            break
        
        # Pattern 2: epoch=XXXX (standard format)
        pattern2 = os.path.join(base_path, 'epoch={}*.ckpt'.format(epoch))
        found = glob.glob(pattern2)
        if found:
            ckpt_filepath = found[0]
            break
    
    if ckpt_filepath is None:
        print(f"Could not find checkpoint file for epoch {epoch}")
        print(f"Searched in: {possible_base_paths}")
        sys.exit(1)
    
    print(f"Loading checkpoint from: {ckpt_filepath}")
    
    # Load model
    # Note: When force_flash=True and the checkpoint was trained with standard mode,
    # we need to handle weight mapping carefully
    if force_flash and not config.training.get('use_flash_mode', False):
        print("Warning: Loading standard model weights into Flash model architecture.")
        print("         Some weights (PairTransformNet) will be randomly initialized.")
        print("         For best results, train a model with useFlashMode=True")
    
    # 修复：强制将权重加载到当前可用GPU或CPU，避免因设备不匹配报错
    import torch
    map_location = None
    if torch.cuda.is_available():
        # 默认加载到cuda:0
        map_location = lambda storage, loc: storage.cuda(0)
    else:
        map_location = 'cpu'
    diffusion = Genie.load_from_checkpoint(ckpt_filepath, config=config, strict=False, map_location=map_location)
    
    # Save checkpoint information
    diffusion.rootdir = rootdir
    diffusion.name = name
    diffusion.version = version
    diffusion.epoch = epoch
    diffusion.checkpoint = ckpt_filepath
    
    return diffusion


def sample_with_flash(model, mask, noise_scale=0.4, verbose=True):
    """
    Sample using the model with memory-efficient settings.
    
    This function wraps p_sample_loop with additional memory optimizations
    suitable for long sequences.
    
    Args:
        model: Genie model instance
        mask: Sequence mask [B, L]
        noise_scale: Sampling noise scale
        verbose: Show progress bar
    
    Returns:
        List of T objects representing the diffusion trajectory
    """
    # Enable inference mode optimizations
    with torch.inference_mode():
        # Clear any cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run sampling
        ts_seq = model.p_sample_loop(mask, noise_scale, verbose=verbose)
        
    return ts_seq


def main(args):
    """Main sampling function with Flash mode support and parallel sampling."""
    
    # Device setup
    if args.gpu is not None:
        device = f'cuda:{args.gpu}'
        # Set memory allocation strategy for long sequences
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)
            except RuntimeError:
                pass
    else:
        device = 'cpu'
    
    print("="*60)
    print("Flash Sample - Memory-Efficient Protein Generation")
    print("="*60)
    
    # Load model
    model = load_flash_model(
        args.rootdir, 
        args.model_name, 
        args.model_version, 
        args.model_epoch,
        force_flash=args.flash_mode
    ).to(device)
    
    # Check if Flash mode is active
    is_flash_mode = hasattr(model.model, '__class__') and 'Flash' in model.model.__class__.__name__
    print(f"Model type: {model.model.__class__.__name__}")
    print(f"Flash mode: {'Enabled' if is_flash_mode else 'Disabled'}")
    
    # Output directory
    outdir = os.path.join(model.rootdir, model.name, f'version_{model.version}', 'samples')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir = os.path.join(outdir, f'epoch_{model.epoch}')
    if os.path.exists(outdir):
        print(f'Output directory exists: {outdir}')
    else:
        os.makedirs(outdir)
    
    # Length validation
    min_length = args.min_length
    max_length = args.max_length
    max_n_res = model.config.io['max_n_res']
    
    if max_length > max_n_res:
        print(f"Warning: max_length ({max_length}) > max_n_res ({max_n_res})")
        print(f"         Clamping to {max_n_res}")
        max_length = max_n_res
    
    # Parallel sampling parameters
    n_seq = args.n_seq  # Number of samples per sequence length
    batch_size = args.batch_size  # Total batch size (should be n_seq * n_lengths_per_batch)
    
    # Calculate how many lengths to process in parallel
    if batch_size % n_seq != 0:
        print(f"Warning: batch_size ({batch_size}) is not divisible by n_seq ({n_seq})")
        print(f"Adjusting batch_size to {batch_size - (batch_size % n_seq)}")
        batch_size = batch_size - (batch_size % n_seq)
    
    n_lengths_per_batch = batch_size // n_seq
    total_lengths = max_length - min_length + 1
    n_batches = (total_lengths + n_lengths_per_batch - 1) // n_lengths_per_batch

    print(f"\nParallel sampling configuration:")
    print(f"  - Samples per length (n_seq): {n_seq}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Lengths per batch: {n_lengths_per_batch}")
    print(f"  - Total batches: {n_batches}")
    print(f"  - Total lengths: {total_lengths}")
    print(f"Generating lengths: {min_length} to {max_length}")
    print(f"Noise scale: {args.noise_scale}")
    print("="*60)
    
    # Memory estimation for Flash mode
    if is_flash_mode and torch.cuda.is_available():
        # Estimate memory usage
        estimated_mem = batch_size * max_length * 128 * 4 / (1024**3)  # Rough estimate in GB
        available_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"Estimated peak memory: ~{estimated_mem:.1f} GB")
        print(f"Available GPU memory: {available_mem:.1f} GB")
    
    # Mixed precision scaler
    scaler = GradScaler('cuda', enabled=True) if torch.cuda.is_available() else None
    
    # Sample in parallel for different lengths
    model.eval()
    total_samples = 0
    batch_idx = 0
    
    for batch_start in trange(0, total_lengths, n_lengths_per_batch, desc="Batch"):
        batch_end = min(batch_start + n_lengths_per_batch, total_lengths)
        lengths_in_batch = list(range(min_length + batch_start, min_length + batch_end))
        
        # Create combined mask for all lengths in this batch
        # Layout: [length_0_seq_0, length_0_seq_1, ..., length_1_seq_0, ...]
        batch_masks = []
        for length in lengths_in_batch:
            for seq_idx in range(n_seq):
                mask = torch.cat([
                    torch.ones((1, length)),
                    torch.zeros((1, max_n_res - length))
                ], dim=1)
                batch_masks.append(mask)
        
        mask = torch.cat(batch_masks, dim=0).to(device)
        # mask shape: (batch_size, max_n_res)
        
        # Run sampling loop with mixed precision
        # Show progress bar with timestep information
        show_progress = args.show_progress and batch_idx == 0  # Only show for first batch
        if torch.cuda.is_available() and scaler is not None:
            with autocast('cuda', dtype=torch.float16):
                with torch.inference_mode():
                    ts_seq = model.p_sample_loop(mask, args.noise_scale, verbose=show_progress)
        else:
            with torch.inference_mode():
                ts_seq = model.p_sample_loop(mask, args.noise_scale, verbose=show_progress)
        
        ts = ts_seq[-1]
        
        # Save samples for each length
        seq_offset = 0
        for length_idx, length in enumerate(lengths_in_batch):
            for seq_idx in range(n_seq):
                sample_idx = batch_idx * n_seq + seq_idx
                ts_idx = seq_offset + seq_idx
                
                coords = ts[ts_idx].trans.detach().cpu().numpy()
                coords = coords[:length]
                
                # Save coordinates
                np.savetxt(
                    os.path.join(outdir, f'{length}_{sample_idx}.npy'), 
                    coords, fmt='%.3f', delimiter=','
                )
                
                # Save trajectory if requested
                if args.save_trajectory:
                    traj_coords = []
                    for step_ts in ts_seq:
                        step_coords = step_ts[ts_idx].trans.detach().cpu().numpy()
                        step_coords = step_coords[:length]
                        step_coords = step_coords - step_coords.mean(axis=0)
                        traj_coords.append(step_coords)
                    traj_coords = np.array(traj_coords)
                    np.save(
                        os.path.join(outdir, f'{length}_{sample_idx}_traj.npy'), 
                        traj_coords
                    )
                
                total_samples += 1
            
            seq_offset += n_seq
        
        batch_idx += 1
        
        # Clear cache periodically for long runs
        if torch.cuda.is_available() and batch_idx % 5 == 0:
            torch.cuda.empty_cache()
    
    # Calculate total samples: (max_length - min_length + 1) * n_seq
    total_samples = total_lengths * n_seq
    print("="*60)
    print(f"Sampling complete! Generated {total_samples} samples")
    print(f"Output directory: {outdir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Flash Sample - Memory-efficient protein backbone generation"
    )
    
    # Model arguments
    parser.add_argument('-g', '--gpu', type=str, nargs='?', const='0',
                        help='GPU device to use (default: None for CPU)')
    parser.add_argument('-r', '--rootdir', type=str, default='runs',
                        help='Root directory containing models')
    parser.add_argument('-n', '--model_name', type=str, required=True,
                        help='Name of Genie model')
    parser.add_argument('-v', '--model_version', type=int,
                        help='Version of Genie model (default: latest)')
    parser.add_argument('-e', '--model_epoch', type=int,
                        help='Epoch of checkpoint (default: latest)')
    
    # Sampling arguments
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Total batch size for parallel sampling')
    parser.add_argument('--n_seq', type=int, default=5,
                        help='Number of samples per sequence length')
    parser.add_argument('--noise_scale', type=float, default=0.6,
                        help='Sampling noise scale (lower = more deterministic)')
    parser.add_argument('--min_length', type=int, default=50,
                        help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    
    # Flash mode arguments
    parser.add_argument('--flash_mode', action='store_true',
                        help='Force Flash mode for memory-efficient sampling')
    
    # Output arguments
    parser.add_argument('--save_trajectory', action='store_true',
                        help='Save all timesteps for visualization')
    parser.add_argument('--show_progress', action='store_true', default=True,
                        help='Show sampling progress with timestep bar (default: True)')
    
    args = parser.parse_args()
    
    # Run with error handling
    try:
        main(args)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('\n' + '='*60)
            print('CRITICAL ERROR: CUDA Out of Memory (OOM) during sampling.')
            print('='*60)
            if torch.cuda.is_available():
                print(f'Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
                print(f'Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB')
            print('\nSuggestions:')
            print('1. Reduce --batch_size')
            print('2. Reduce --n_seq')
            print('3. Reduce --max_length')
            print('4. Enable --flash_mode for memory-efficient sampling')
            print('='*60 + '\n')
            sys.exit(1)
        else:
            raise e
