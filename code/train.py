import argparse
from contextlib import contextmanager
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from common import (
    DEFAULT_BLOCK_CHANNELS,
    DEFAULT_DOWN_BLOCKS,
    DEFAULT_UP_BLOCKS,
    IMAGE_SIZE,
    build_unet,
    create_scheduler,
    encode_images,
    load_vae,
    make_dataloader,
    non_negative_int,
    parse_int_tuple,
    parse_str_tuple,
    positive_float,
    positive_int,
    resolve_device,
    set_seed,
)
from sampling import sample_to_pngs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch latent DDPM U-Net for HW5.")
    parser.add_argument("--image_dir", type=Path, default=Path("public_data/images"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or a torch device such as 'cuda:0'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--max_train_steps", type=positive_int, default=None, help="Maximum optimizer steps.")
    parser.add_argument("--batch_size", type=positive_int, default=8)
    parser.add_argument("--num_workers", type=non_negative_int, default=4)
    parser.add_argument("--subset_size", type=positive_int, default=None, help="Optional smoke-test subset size.")
    parser.add_argument("--gradient_accumulation_steps", type=positive_int, default=1)
    parser.add_argument("--learning_rate", type=positive_float, default=1e-4)
    parser.add_argument("--ema_decay", type=positive_float, default=0.9999, help="EMA decay for saved sampling weights.")
    parser.add_argument("--disable_ema", action="store_true", help="Disable EMA checkpointing.")
    parser.add_argument("--save_freq", type=non_negative_int, default=1000, help="Checkpoint every N optimizer steps; 0 disables periodic saves.")
    parser.add_argument("--eval_freq", type=non_negative_int, default=1000, help="Preview every N optimizer steps; 0 disables previews.")
    parser.add_argument("--preview_num_samples", type=positive_int, default=4)
    parser.add_argument("--preview_batch_size", type=positive_int, default=4)
    parser.add_argument("--preview_inference_steps", type=positive_int, default=50)
    parser.add_argument("--sample_size", type=positive_int, default=32)
    parser.add_argument("--layers_per_block", type=positive_int, default=2)
    parser.add_argument(
        "--block_out_channels",
        type=parse_int_tuple,
        default=DEFAULT_BLOCK_CHANNELS,
        help="Comma-separated U-Net block widths, e.g. 64,128,256,256.",
    )
    parser.add_argument(
        "--down_block_types",
        type=parse_str_tuple,
        default=DEFAULT_DOWN_BLOCKS,
        help="Comma-separated diffusers down block classes.",
    )
    parser.add_argument(
        "--up_block_types",
        type=parse_str_tuple,
        default=DEFAULT_UP_BLOCKS,
        help="Comma-separated diffusers up block classes.",
    )
    return parser.parse_args()


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("ema_decay must be in the open interval (0, 1)")
        self.decay = decay
        self.shadow_params = [param.detach().clone() for param in self._tracked_params(model)]
        for param in self.shadow_params:
            param.requires_grad_(False)
        self.backup_params: list[torch.Tensor] | None = None

    @staticmethod
    def _tracked_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
        return [param for param in model.parameters() if param.requires_grad]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow_param, param in zip(self.shadow_params, self._tracked_params(model), strict=True):
            shadow_param.lerp_(param.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        self.backup_params = [param.detach().clone() for param in self._tracked_params(model)]

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for shadow_param, param in zip(self.shadow_params, self._tracked_params(model), strict=True):
            param.copy_(shadow_param.to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self.backup_params is None:
            raise RuntimeError("EMA restore called before store")
        for backup_param, param in zip(self.backup_params, self._tracked_params(model), strict=True):
            param.copy_(backup_param.to(device=param.device, dtype=param.dtype))
        self.backup_params = None


@contextmanager
def use_ema_weights(ema: EMAModel, model: torch.nn.Module):
    ema.store(model)
    ema.copy_to(model)
    try:
        yield
    finally:
        ema.restore(model)


def save_checkpoint(unet, checkpoint_dir: Path, step: int | None = None, name: str | None = None) -> Path:
    path = checkpoint_dir / (name if name is not None else f"unet_{step}")
    path.mkdir(parents=True, exist_ok=False)
    unet.save_pretrained(str(path))
    return path


def save_ema_checkpoint(
    ema: EMAModel,
    unet,
    checkpoint_dir: Path,
    step: int | None = None,
    name: str | None = None,
) -> Path:
    with use_ema_weights(ema, unet):
        return save_checkpoint(unet, checkpoint_dir, step=step, name=name)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    run_dir = args.output_dir / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataloader = make_dataloader(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    print(f"Loaded {len(dataloader.dataset)} images at {IMAGE_SIZE}x{IMAGE_SIZE}")

    vae = load_vae(device)
    unet = build_unet(
        sample_size=args.sample_size,
        block_out_channels=args.block_out_channels,
        layers_per_block=args.layers_per_block,
        down_block_types=args.down_block_types,
        up_block_types=args.up_block_types,
    ).to(device)
    unet.train()
    ema = None if args.disable_ema else EMAModel(unet, decay=args.ema_decay)
    if ema is not None:
        print(f"EMA enabled with decay {args.ema_decay}")

    scheduler = create_scheduler()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_updates = args.max_train_steps or (args.epochs * updates_per_epoch)
    global_step = 0
    accumulation_count = 0
    running_loss = 0.0

    progress = tqdm(total=total_updates, desc="Training")
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(dataloader):
            if global_step >= total_updates:
                break

            pixel_values = batch.to(device, non_blocking=True)
            latents = encode_images(vae, pixel_values)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {global_step}: {loss.item()}")

            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            accumulation_count += 1
            running_loss += loss.item()

            is_last_batch = batch_index == len(dataloader) - 1
            should_step = accumulation_count == args.gradient_accumulation_steps or is_last_batch
            if should_step:
                optimizer.step()
                if ema is not None:
                    ema.update(unet)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{running_loss / accumulation_count:.4f}")
                accumulation_count = 0
                running_loss = 0.0

                if args.save_freq and global_step % args.save_freq == 0:
                    path = save_checkpoint(unet, checkpoint_dir, global_step)
                    print(f"Saved checkpoint: {path}")
                    if ema is not None:
                        ema_path = save_ema_checkpoint(ema, unet, checkpoint_dir, name=f"unet_{global_step}_ema")
                        print(f"Saved EMA checkpoint: {ema_path}")
                if args.eval_freq and global_step % args.eval_freq == 0:
                    preview_dir = sample_dir / f"step_{global_step:06d}"
                    if ema is None:
                        sample_to_pngs(
                            unet=unet,
                            vae=vae,
                            scheduler=create_scheduler(),
                            output_dir=preview_dir,
                            num_samples=args.preview_num_samples,
                            batch_size=args.preview_batch_size,
                            num_inference_steps=args.preview_inference_steps,
                            device=device,
                            seed=args.seed + global_step,
                            show_progress=False,
                        )
                    else:
                        with use_ema_weights(ema, unet):
                            sample_to_pngs(
                                unet=unet,
                                vae=vae,
                                scheduler=create_scheduler(),
                                output_dir=preview_dir,
                                num_samples=args.preview_num_samples,
                                batch_size=args.preview_batch_size,
                                num_inference_steps=args.preview_inference_steps,
                                device=device,
                                seed=args.seed + global_step,
                                show_progress=False,
                            )
                    unet.train()

                if global_step >= total_updates:
                    break
        if global_step >= total_updates:
            break

    progress.close()
    final_path = save_checkpoint(unet, checkpoint_dir, name="unet_final")
    print(f"Saved final checkpoint: {final_path}")
    if ema is not None:
        ema_final_path = save_ema_checkpoint(ema, unet, checkpoint_dir, name="unet_final_ema")
        print(f"Saved final EMA checkpoint: {ema_final_path}")


if __name__ == "__main__":
    main()
