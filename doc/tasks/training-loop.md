# Training Loop

## Goal

Train the from-scratch U-Net to predict Gaussian noise added to scaled VAE latents. The smallest useful outcome is an end-to-end smoke training run that updates weights, logs finite loss, saves a checkpoint, and can trigger preview sampling.

## Inputs

- `doc/proposal.md`: Complete latent DDPM training in `train.py`, optimize MSE between predicted and sampled noise, and save compatible checkpoints.
- `doc/detailed-design.md`: Coordinate Dataset Loader, VAE Adapter, U-Net Factory, `DDPMScheduler`, optimizer, checkpointing, logging, and preview triggers.

## Tasks

- [ ] Create the DDPM scheduler with `num_train_timesteps=1000`, scaled-linear beta range `0.00085` to `0.012`, and `clip_sample=False`.
- [ ] Implement the training step: encode pixels, sample Gaussian noise, sample random timesteps, add scheduler noise, predict noise, and compute MSE loss.
- [ ] Add optimizer setup and gradient accumulation, dividing loss during accumulation and stepping at accumulation boundaries.
- [ ] Stop with a clear error if the loss becomes NaN or infinite.
- [ ] Write checkpoints under an explicit checkpoint directory using `unet.save_pretrained()`.
- [ ] Keep preview output separate from final `results/` and trigger preview generation only at configured intervals.
- [ ] Run a tiny data-backed smoke test with a small subset and one or a few max training steps.

## Done When

- [ ] A smoke run completes a U-Net update with finite loss.
- [ ] A checkpoint is written and can be loaded by `UNet2DModel.from_pretrained()`.
- [ ] Training outputs are separated into checkpoints, logs, and preview samples.
