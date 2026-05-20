import argparse
import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


IMAGE_SIZE = 256
LATENT_CHANNELS = 4
LATENT_SIZE = 32
FINAL_SAMPLE_COUNT = 3000
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
SCHEDULER_KWARGS = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "clip_sample": False,
}
DEFAULT_BLOCK_CHANNELS = (64, 128, 256, 256)
DEFAULT_DOWN_BLOCKS = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
DEFAULT_UP_BLOCKS = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all tuple values must be positive integers")
    return parsed


def parse_str_tuple(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected comma-separated names")
    return parsed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def require_directory(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def list_pngs(path: Path) -> list[Path]:
    return sorted(Path(path).glob("*.png"))


def fail_if_output_pngs_exist(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        existing = list_pngs(output_dir)
        if existing:
            raise FileExistsError(
                f"{output_dir} already contains {len(existing)} PNG files. "
                "Choose a clean directory or pass an explicit overwrite option."
            )


class ProfessorFaceDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = IMAGE_SIZE):
        self.image_dir = require_directory(Path(image_dir), "image_dir")
        self.image_size = image_size
        self.image_paths = list_pngs(self.image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG files found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
                array = np.asarray(image, dtype=np.float32) / 255.0
        except Exception as exc:
            raise RuntimeError(f"Failed to read image {path}") from exc
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor.mul(2.0).sub(1.0)


def make_dataloader(
    image_dir: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    subset_size: int | None = None,
    seed: int | None = None,
) -> DataLoader:
    dataset: Dataset = ProfessorFaceDataset(image_dir)
    if subset_size is not None:
        if subset_size <= 0:
            raise ValueError("subset_size must be positive when provided")
        dataset = Subset(dataset, range(min(subset_size, len(dataset))))
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def load_vae(device: torch.device):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


@torch.no_grad()
def encode_images(vae, pixel_values: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    if latents.ndim != 4 or latents.shape[1] != LATENT_CHANNELS:
        raise RuntimeError(f"Unexpected VAE latent shape: {tuple(latents.shape)}")
    return latents


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return decoded.clamp(-1, 1).add(1.0).div(2.0)


def create_scheduler():
    from diffusers import DDPMScheduler

    return DDPMScheduler(**SCHEDULER_KWARGS)


def build_unet(
    sample_size: int = LATENT_SIZE,
    block_out_channels: Sequence[int] = DEFAULT_BLOCK_CHANNELS,
    layers_per_block: int = 2,
    down_block_types: Sequence[str] = DEFAULT_DOWN_BLOCKS,
    up_block_types: Sequence[str] = DEFAULT_UP_BLOCKS,
):
    from diffusers import UNet2DModel

    if len(block_out_channels) != len(down_block_types) or len(block_out_channels) != len(up_block_types):
        raise ValueError("block_out_channels, down_block_types, and up_block_types must have equal lengths")
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=LATENT_CHANNELS,
        out_channels=LATENT_CHANNELS,
        layers_per_block=layers_per_block,
        block_out_channels=tuple(block_out_channels),
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
    )


def load_unet_checkpoint(checkpoint_dir: Path, device: torch.device):
    from diffusers import UNet2DModel

    require_directory(checkpoint_dir, "checkpoint_dir")
    unet = UNet2DModel.from_pretrained(str(checkpoint_dir)).to(device)
    unet.requires_grad_(False)
    unet.eval()
    return unet


def assert_finite_tensor(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{label} contains NaN or infinite values")


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    array = image.mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def save_image_batch(images: torch.Tensor, output_dir: Path, start_index: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(images):
        tensor_to_pil(image).save(output_dir / f"{start_index + offset:04d}.png")


def validate_png_directory(
    results_dir: Path,
    expected_count: int = FINAL_SAMPLE_COUNT,
    image_size: int = IMAGE_SIZE,
) -> list[Path]:
    results_dir = require_directory(results_dir, "results_dir")
    pngs = list_pngs(results_dir)
    if len(pngs) != expected_count:
        raise ValueError(f"Expected {expected_count} PNG files in {results_dir}, found {len(pngs)}")
    for path in pngs:
        try:
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError(f"{path} has mode {image.mode}, expected RGB")
                if image.size != (image_size, image_size):
                    raise ValueError(f"{path} has size {image.size}, expected {(image_size, image_size)}")
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError(f"Failed to validate image {path}") from exc
    return pngs


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_files(src_dir: Path, dst_dir: Path, patterns: Iterable[str]) -> None:
    src_dir = require_directory(src_dir, "source directory")
    dst_dir.mkdir(parents=True, exist_ok=False)
    for pattern in patterns:
        for src in sorted(src_dir.rglob(pattern)):
            if src.is_file():
                relative = src.relative_to(src_dir)
                copy_file(src, dst_dir / relative)


def stage_validation_input(results_dir: Path, ref_dir: Path, validation_input_dir: Path) -> None:
    if validation_input_dir.exists():
        raise FileExistsError(f"validation input directory already exists: {validation_input_dir}")
    ref_src = require_directory(ref_dir, "ref_dir")
    res_src = require_directory(results_dir, "results_dir")
    ref_dst = validation_input_dir / "ref"
    res_dst = validation_input_dir / "res"
    ref_dst.mkdir(parents=True)
    res_dst.mkdir()
    for name in ("config.json", "test_mu.npy", "test_sigma.npy"):
        copy_file(require_file(ref_src / name, name), ref_dst / name)
    for path in list_pngs(res_src):
        copy_file(path, res_dst / path.name)


def run_scorer(scorer_path: Path, input_dir: Path, output_dir: Path, config_name: str = "config.json") -> Path:
    require_file(scorer_path, "scorer_path")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(scorer_path),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--config",
        config_name,
    ]
    subprocess.run(command, check=True)
    scores_path = output_dir / "scores.json"
    require_file(scores_path, "scores.json")
    return scores_path


def read_scores(scores_path: Path) -> dict:
    with open(scores_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def make_zip(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        raise FileExistsError(f"zip file already exists: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir.parent))
