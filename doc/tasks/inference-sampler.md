# Inference Sampler

## Goal

Generate final or test image sets from a saved U-Net checkpoint. The smallest useful outcome is an inference script that can produce sequential RGB 256x256 PNG files from random latent noise.

## Inputs

- `doc/proposal.md`: Complete reverse diffusion sampling in `inference.py` and generate `0000.png` through `2999.png` for final submission.
- `doc/detailed-design.md`: Load `UNet2DModel.from_pretrained()`, use the same scheduler family and VAE adapter, enforce output count and directory safety.

## Tasks

- [ ] Load the trained U-Net checkpoint and frozen VAE from CLI-provided paths.
- [ ] Create the inference scheduler with training-compatible beta settings and call `set_timesteps(num_inference_steps)`.
- [ ] Generate latents batch by batch until `num_samples` images are produced.
- [ ] Run reverse diffusion, decode final latents, and save deterministic sequential PNG filenames.
- [ ] Fail if the output directory already contains PNG files unless an explicit overwrite workflow is implemented.
- [ ] Run a small inference smoke test with `--num_samples 2` from a smoke checkpoint.
- [ ] Verify generated filenames are unique, sequential, RGB, and 256x256.

## Done When

- [ ] Inference can generate a small valid PNG set from a saved checkpoint.
- [ ] Final-generation mode defaults to 3,000 samples and warns or fails when configured otherwise.
- [ ] Existing output PNGs are not silently overwritten or mixed with new outputs.
