# VAE Adapter

## Goal

Encapsulate all legal use of the allowed `stabilityai/sd-vae-ft-mse` VAE. The smallest useful outcome is frozen encode/decode helpers that apply the correct latent scaling and produce decodable 256x256 RGB outputs.

## Inputs

- `doc/proposal.md`: Use the pretrained VAE only for pixel-to-latent and latent-to-pixel conversion.
- `doc/detailed-design.md`: Provide `load_vae`, `encode_images`, and `decode_latents` helpers with eval mode, no gradients, and `vae.config.scaling_factor` handling.

## Tasks

- [ ] Implement `load_vae(device)` using `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")`.
- [ ] Freeze all VAE parameters, set eval mode, and keep encode/decode calls inside no-grad regions.
- [ ] Implement `encode_images` to sample the latent distribution and multiply by `vae.config.scaling_factor`.
- [ ] Implement `decode_latents` to divide by `vae.config.scaling_factor`, decode, clamp, and map images to `[0, 1]`.
- [ ] Add a smoke check that encodes and decodes one dataset batch and verifies latent channel count is 4.
- [ ] Verify decoded tensors can be saved or converted as RGB 256x256 PNG images.

## Done When

- [ ] All VAE parameters have `requires_grad == False`.
- [ ] Encoding a 256x256 batch produces scaled latents with 4 channels and the expected spatial size.
- [ ] Decoding a latent batch produces valid image tensors suitable for PNG output.
