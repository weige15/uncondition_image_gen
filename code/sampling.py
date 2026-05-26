from pathlib import Path

import torch
from tqdm.auto import tqdm

from common import LATENT_CHANNELS, LATENT_SIZE, assert_finite_tensor, decode_latents, save_image_batch


@torch.no_grad()
def sample_to_pngs(
    unet,
    vae,
    scheduler,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_inference_steps: int,
    device: torch.device,
    seed: int,
    start_index: int = 0,
    class_labels: torch.Tensor | None = None,
    show_progress: bool = True,
) -> int:
    unet_was_training = unet.training
    unet.eval()
    vae.eval()
    scheduler.set_timesteps(num_inference_steps)
    if class_labels is not None and class_labels.shape[0] != num_samples:
        raise ValueError(f"class_labels must have {num_samples} entries, got {class_labels.shape[0]}")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    written = 0
    total_batches = (num_samples + batch_size - 1) // batch_size
    iterator = range(total_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Sampling")

    for _ in iterator:
        current_batch = min(batch_size, num_samples - written)
        batch_class_labels = None
        if class_labels is not None:
            batch_class_labels = class_labels[written : written + current_batch].to(device)
        latents = torch.randn(
            (current_batch, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
            generator=generator,
            device=device,
        )
        for timestep in scheduler.timesteps:
            noise_pred = unet(latents, timestep, class_labels=batch_class_labels).sample
            assert_finite_tensor(noise_pred, "predicted noise")
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample
            assert_finite_tensor(latents, "sampled latents")

        images = decode_latents(vae, latents)
        assert_finite_tensor(images, "decoded images")
        save_image_batch(images, output_dir, start_index + written)
        written += current_batch

    if unet_was_training:
        unet.train()
    return written
