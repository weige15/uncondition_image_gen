import argparse
from pathlib import Path

from common import (
    CLASS_NAMES,
    FINAL_SAMPLE_COUNT,
    create_scheduler,
    fail_if_output_pngs_exist,
    load_unet_checkpoint,
    load_vae,
    make_class_label_sequence,
    positive_int,
    resolve_device,
    set_seed,
    validate_png_directory,
)
from sampling import sample_to_pngs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HW5 images from a trained U-Net checkpoint.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--num_samples", type=positive_int, default=FINAL_SAMPLE_COUNT)
    parser.add_argument("--batch_size", type=positive_int, default=32)
    parser.add_argument("--num_inference_steps", type=positive_int, default=1000)
    parser.add_argument("--scheduler", choices=("ddpm", "ddim", "dpm_solver"), default="ddpm")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--class_sampling",
        choices=("none", "train_prior", "uniform"),
        default="train_prior",
        help="Class-label schedule for class-conditional checkpoints; ignored by unconditional checkpoints.",
    )
    parser.add_argument(
        "--class_label",
        choices=CLASS_NAMES,
        default=None,
        help="Generate only one class for class-conditional checkpoints.",
    )
    parser.add_argument(
        "--allow_smoke_count",
        action="store_true",
        help="Allow --num_samples to differ from 3000 for local smoke tests.",
    )
    parser.add_argument(
        "--overwrite_existing_pngs",
        action="store_true",
        help="Delete existing PNG files in output_dir one by one before generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples != FINAL_SAMPLE_COUNT and not args.allow_smoke_count:
        raise ValueError(
            f"--num_samples defaults to {FINAL_SAMPLE_COUNT}. "
            "Pass --allow_smoke_count for small smoke-test generations."
        )
    if args.scheduler == "dpm_solver" and args.num_inference_steps > 300:
        raise ValueError(
            "DPM-Solver should be run with a low step count for this epsilon-prediction checkpoint. "
            "Use one --num_inference_steps flag, typically 50, 100, or 250."
        )

    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.overwrite_existing_pngs and args.output_dir.exists():
        for png_path in sorted(args.output_dir.glob("*.png")):
            png_path.unlink()
    fail_if_output_pngs_exist(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)
    unet = load_unet_checkpoint(args.checkpoint_dir, device)
    scheduler = create_scheduler(args.scheduler)
    num_class_embeds = getattr(unet.config, "num_class_embeds", None)
    class_labels = None
    if num_class_embeds:
        class_labels = make_class_label_sequence(
            num_samples=args.num_samples,
            mode=args.class_sampling,
            seed=args.seed,
            device=device,
            class_name=args.class_label,
        )
        if class_labels is None:
            raise ValueError("This checkpoint is class-conditional; use --class_sampling train_prior/uniform or --class_label.")
    elif args.class_label is not None or args.class_sampling != "train_prior":
        print("Ignoring class sampling options because the checkpoint is unconditional.")

    written = sample_to_pngs(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        device=device,
        seed=args.seed,
        class_labels=class_labels,
    )
    if args.num_samples == FINAL_SAMPLE_COUNT:
        validate_png_directory(args.output_dir)
    print(f"Saved {written} PNG images to {args.output_dir}")


if __name__ == "__main__":
    main()
