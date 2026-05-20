# Configuration and CLI

## Goal

Provide reproducible command-line controls for training, inference, validation, and packaging. The smallest useful outcome is that each executable script can parse paths, seeds, device choices, and core hyperparameters with safe defaults.

## Inputs

- `doc/proposal.md`: Expose CLI arguments for image paths, output paths, checkpoints, sample counts, training steps, batch size, learning rate, seed, and save/eval frequency.
- `doc/detailed-design.md`: Own argument parsing, default values, reproducibility settings, device selection, and resolved runtime paths.

## Tasks

- [ ] Add `argparse` options to the training script for dataset path, output directory, epochs or max steps, batch size, gradient accumulation, learning rate, seed, checkpoint frequency, preview frequency, and U-Net capacity.
- [ ] Add `argparse` options to the inference script for checkpoint directory, output directory, sample count, batch size, inference steps, seed, and device.
- [ ] Add validation and packaging CLI arguments for result directory, reference directory, scorer path, output directory, and student ID.
- [ ] Implement shared seed setup for Python, NumPy, and PyTorch where scripts need deterministic smoke tests.
- [ ] Validate non-positive numeric arguments and fail early when required input directories are missing.
- [ ] Run each executable with `--help` and a smoke-test argument set to confirm paths and defaults parse correctly.

## Done When

- [ ] Training, inference, validation, and packaging entrypoints expose the required arguments with documented defaults.
- [ ] Invalid paths or non-positive numeric settings fail before expensive model or data loading begins.
- [ ] A fixed seed can reproduce the first sampled latent/noise tensor in a tiny local check.
