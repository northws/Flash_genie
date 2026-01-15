import os
import sys

# Add the project root to sys.path to enable imports from the 'genie' package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm, trange
from torch.amp import autocast, GradScaler

from genie.utils.model_io import load_model


def main(args):

	# device
	device = 'cuda:{}'.format(args.gpu) if args.gpu is not None else 'cpu'

	# model - use Flash mode if specified
	if args.flash_mode:
		# Use Flash mode for memory-efficient sampling
		from genie.flash_sample import load_flash_model
		model = load_flash_model(
			args.rootdir, args.model_name, args.model_version, args.model_epoch,
			force_flash=True
		).to(device)
		print("Flash mode enabled for memory-efficient sampling")
	else:
		model = load_model(args.rootdir, args.model_name, args.model_version, args.model_epoch).to(device)

	# output directory
	outdir = os.path.join(model.rootdir, model.name, 'version_{}'.format(model.version), 'samples')
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	outdir = os.path.join(outdir, 'epoch_{}'.format(model.epoch))
	if os.path.exists(outdir):
		print('Samples existed!')
	else:
		os.mkdir(outdir)

	# sanity check
	min_length = args.min_length
	max_length = args.max_length
	max_n_res = model.config.io['max_n_res']
	assert max_length <= max_n_res

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

	print(f"Parallel sampling configuration:")
	print(f"  - Samples per length (n_seq): {n_seq}")
	print(f"  - Batch size: {batch_size}")
	print(f"  - Lengths per batch: {n_lengths_per_batch}")
	print(f"  - Total batches: {n_batches}")
	print(f"  - Total lengths: {total_lengths}")

	# Mixed precision scaler
	scaler = GradScaler('cuda', enabled=True) if torch.cuda.is_available() else None

	# Sample in parallel for different lengths
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
				ts_seq = model.p_sample_loop(mask, args.noise_scale, verbose=show_progress)
		else:
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
				np.savetxt(os.path.join(outdir, f'{length}_{sample_idx}.npy'), coords, fmt='%.3f', delimiter=',')
				
				if args.save_trajectory:
					# Save trajectory for this sample
					traj_coords = []
					for step_ts in ts_seq:
						step_coords = step_ts[ts_idx].trans.detach().cpu().numpy()
						step_coords = step_coords[:length]
						step_coords = step_coords - step_coords.mean(axis=0)
						traj_coords.append(step_coords)
					traj_coords = np.array(traj_coords)
					np.save(os.path.join(outdir, f'{length}_{sample_idx}_traj.npy'), traj_coords)
			
			seq_offset += n_seq
		
		batch_idx += 1
		
		# Clear cache periodically
		if torch.cuda.is_available() and batch_idx % 5 == 0:
			torch.cuda.empty_cache()

	# Calculate total samples: (max_length - min_length + 1) * n_seq
	total_samples = total_lengths * n_seq
	print(f"Sampling complete! Total samples: {total_samples}")


if __name__ == '__main__':

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', type=str, nargs='?', const='0', help='GPU device to use')
	parser.add_argument('-r', '--rootdir', type=str, help='Root directory (default to runs)', default='runs')
	parser.add_argument('-n', '--model_name', type=str, help='Name of Genie model', required=True)
	parser.add_argument('-v', '--model_version', type=int, help='Version of Genie model')
	parser.add_argument('-e', '--model_epoch', type=int, help='Epoch Genie model checkpointed')
	parser.add_argument('--batch_size', type=int, help='Total batch size for parallel sampling', default=20)
	parser.add_argument('--n_seq', type=int, help='Number of samples per sequence length', default=5)
	parser.add_argument('--num_batches', type=int, help='Number of batches (deprecated, use batch_size and n_seq)', default=2)
	parser.add_argument('--noise_scale', type=float, help='Sampling noise scale', default=0.6)
	parser.add_argument('--min_length', type=int, help='Minimum length', default=50)
	parser.add_argument('--max_length', type=int, help='Maximum length', default=128)
	parser.add_argument('--save_trajectory', action='store_true', help='Save all timesteps for visualization')
	parser.add_argument('--flash_mode', action='store_true', 
						help='Enable Flash IPA for memory-efficient sampling (recommended for long sequences)')
	parser.add_argument('--show_progress', action='store_true', default=True,
						help='Show sampling progress with timestep bar (default: True)')
	args = parser.parse_args()

	# run
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
			print('Suggestions:')
			print('1. Reduce --batch_size')
			print('2. Reduce --n_seq')
			print('3. Reduce --max_length')
			print('4. Enable --flash_mode for memory-efficient sampling')
			print('='*60 + '\n')
			sys.exit(1)
		else:
			raise e
