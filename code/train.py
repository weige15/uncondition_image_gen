import argparse
from contextlib import contextmanager
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from common import (
    CLASS_NAMES,
    DEFAULT_BLOCK_CHANNELS,
    DEFAULT_DOWN_BLOCKS,
    DEFAULT_UP_BLOCKS,
    IMAGE_SIZE,
    MULTI_ATTN_DOWN_BLOCKS,
    MULTI_ATTN_UP_BLOCKS,
    build_unet,
    create_scheduler,
    encode_images,
    load_vae,
    make_class_label_sequence,
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
    parser.add_argument(
        "--lr_scheduler",
        choices=("constant", "cosine"),
        default="constant",
        help="Optimizer-step learning-rate schedule. Cosine is intended for longer runs.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=non_negative_int,
        default=0,
        help="Linearly warm up the learning rate over this many optimizer steps.",
    )
    parser.add_argument(
        "--lr_min_factor",
        type=positive_float,
        default=0.1,
        help="Final LR as a fraction of --learning_rate for cosine scheduling.",
    )
    parser.add_argument(
        "--latent_mode",
        choices=("sample", "mode"),
        default="sample",
        help="Use sampled VAE latents or deterministic posterior modes during training.",
    )
    parser.add_argument(
        "--class_conditioning",
        action="store_true",
        help="Condition the U-Net on labels inferred from filename prefixes: ntu, nccu, nycu.",
    )
    parser.add_argument("--loss_weighting", choices=("none", "min_snr"), default="none")
    parser.add_argument("--min_snr_gamma", type=positive_float, default=5.0)
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
        "--architecture_preset",
        choices=("custom", "default", "multi_attn"),
        default="custom",
        help="Use 'multi_attn' for attention at both 16x16 and 8x8 latent resolutions.",
    )
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


def compute_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    loss_weighting: str,
    min_snr_gamma: float,
) -> torch.Tensor:
    if loss_weighting == "none":
        return F.mse_loss(noise_pred, noise)

    per_sample_loss = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2, 3))
    alphas_cumprod = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    alpha = alphas_cumprod[timesteps]
    snr = alpha / (1.0 - alpha).clamp_min(1e-8)
    weights = torch.minimum(snr, torch.full_like(snr, min_snr_gamma)) / snr.clamp_min(1e-8)
    return (per_sample_loss * weights).mean()


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    total_updates: int,
    warmup_steps: int,
    min_factor: float,
):
    if not 0.0 < min_factor <= 1.0:
        raise ValueError("--lr_min_factor must be in the interval (0, 1]")
    if warmup_steps >= total_updates:
        raise ValueError("--lr_warmup_steps must be smaller than the total optimizer-step count")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        if scheduler_type == "constant":
            return 1.0

        decay_steps = max(1, total_updates - warmup_steps)
        decay_progress = min(max(step - warmup_steps, 0), decay_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
        return_labels=args.class_conditioning,
    )
    print(f"Loaded {len(dataloader.dataset)} images at {IMAGE_SIZE}x{IMAGE_SIZE}")

    vae = load_vae(device)
    down_block_types = args.down_block_types
    up_block_types = args.up_block_types
    if args.architecture_preset == "default":
        down_block_types = DEFAULT_DOWN_BLOCKS
        up_block_types = DEFAULT_UP_BLOCKS
    elif args.architecture_preset == "multi_attn":
        down_block_types = MULTI_ATTN_DOWN_BLOCKS
        up_block_types = MULTI_ATTN_UP_BLOCKS

    unet = build_unet(
        sample_size=args.sample_size,
        block_out_channels=args.block_out_channels,
        layers_per_block=args.layers_per_block,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        num_class_embeds=len(CLASS_NAMES) if args.class_conditioning else None,
    ).to(device)
    unet.train()
    ema = None if args.disable_ema else EMAModel(unet, decay=args.ema_decay)
    if ema is not None:
        print(f"EMA enabled with decay {args.ema_decay}")
    if args.loss_weighting == "min_snr":
        print(f"Min-SNR loss weighting enabled with gamma {args.min_snr_gamma}")
    if args.latent_mode == "mode":
        print("Training on deterministic VAE posterior modes")
    if args.class_conditioning:
        print(f"Class conditioning enabled for labels: {', '.join(CLASS_NAMES)}")
    if args.architecture_preset != "custom":
        print(f"Architecture preset: {args.architecture_preset}")

    scheduler = create_scheduler()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_updates = args.max_train_steps or (args.epochs * updates_per_epoch)
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=args.lr_scheduler,
        total_updates=total_updates,
        warmup_steps=args.lr_warmup_steps,
        min_factor=args.lr_min_factor,
    )
    if args.lr_scheduler == "cosine" or args.lr_warmup_steps:
        print(
            "LR schedule: "
            f"{args.lr_scheduler}, warmup_steps={args.lr_warmup_steps}, min_factor={args.lr_min_factor}"
        )
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    accumulation_count = 0
    running_loss = 0.0

    progress = tqdm(total=total_updates, desc="Training")
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(dataloader):
            if global_step >= total_updates:
                break

            if args.class_conditioning:
                pixel_values, class_labels = batch
                class_labels = class_labels.to(device, non_blocking=True)
            else:
                pixel_values = batch
                class_labels = None
            pixel_values = pixel_values.to(device, non_blocking=True)
            latents = encode_images(vae, pixel_values, latent_mode=args.latent_mode)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, class_labels=class_labels).sample
            loss = compute_loss(
                noise_pred=noise_pred,
                noise=noise,
                timesteps=timesteps,
                scheduler=scheduler,
                loss_weighting=args.loss_weighting,
                min_snr_gamma=args.min_snr_gamma,
            )
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
                lr_scheduler.step()
                if ema is not None:
                    ema.update(unet)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                current_lr = optimizer.param_groups[0]["lr"]
                progress.set_postfix(loss=f"{running_loss / accumulation_count:.4f}", lr=f"{current_lr:.2e}")
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
                    preview_class_labels = None
                    if args.class_conditioning:
                        preview_class_labels = make_class_label_sequence(
                            num_samples=args.preview_num_samples,
                            mode="train_prior",
                            seed=args.seed + global_step,
                            device=device,
                        )
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
                            class_labels=preview_class_labels,
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
                                class_labels=preview_class_labels,
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
