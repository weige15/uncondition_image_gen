# U-Net Factory

## Goal

Build the from-scratch latent denoiser used by training. The smallest useful outcome is a configurable `UNet2DModel` initialized from config, compatible with VAE latents, and save/load compatible with Diffusers.

## Inputs

- `doc/proposal.md`: Define a legal from-scratch U-Net with `in_channels=4`, `out_channels=4`, attention in deeper blocks, and configurable capacity.
- `doc/detailed-design.md`: Use `UNet2DModel` initialized from config, verify latent sample size, and save checkpoints with `save_pretrained()`.

## Tasks

- [ ] Implement a U-Net construction function that accepts sample size, block channels, layers per block, and down/up block types.
- [ ] Set baseline latent defaults to `sample_size=32`, `in_channels=4`, and `out_channels=4`, with validation against the verified VAE latent shape.
- [ ] Ensure the factory never loads pretrained U-Net weights during training-mode construction.
- [ ] Add a synthetic forward-pass smoke test with latents shaped `[2, 4, 32, 32]` and random timesteps.
- [ ] Save the model to a temporary checkpoint directory with `save_pretrained()` and reload it with `UNet2DModel.from_pretrained()`.

## Done When

- [ ] U-Net output shape exactly matches the input latent tensor shape.
- [ ] Training construction starts from random initialized weights only.
- [ ] Saved checkpoints include enough config for inference to reload the architecture.
