# Preview Sampler

## Goal

Generate small qualitative image previews during training without touching final submission results. The smallest useful outcome is a preview function that samples from the current in-memory U-Net, decodes through the frozen VAE, and writes valid PNG previews.

## Inputs

- `doc/proposal.md`: Implement periodic sampling to catch collapsed or invalid training before final generation.
- `doc/detailed-design.md`: Use the current U-Net, frozen VAE, DDPM reverse timesteps, deterministic preview seeds, and separate `samples/` output.

## Tasks

- [ ] Implement preview sampling under `torch.no_grad()` with U-Net and VAE in eval mode.
- [ ] Initialize random latents with the verified latent shape and run scheduler reverse steps.
- [ ] Decode final latents with the VAE adapter and save PNG previews or a preview grid under the run `samples/` directory.
- [ ] Detect NaNs in generated tensors and report the training step if preview generation fails.
- [ ] Restore the U-Net to train mode after preview generation.
- [ ] Run a random-weight preview smoke test for 1 to 4 images and verify saved files are 256x256 PNGs.

## Done When

- [ ] Preview generation writes valid PNG files outside final `results/`.
- [ ] Preview generation does not leave the U-Net stuck in eval mode during training.
- [ ] A deterministic preview seed can reproduce the same preview latents for checkpoint comparison.
