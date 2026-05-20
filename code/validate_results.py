import argparse
from pathlib import Path

import torch

from common import (
    FINAL_SAMPLE_COUNT,
    IMAGE_SIZE,
    positive_int,
    read_scores,
    run_scorer,
    stage_validation_input,
    validate_png_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated HW5 PNGs and optionally run local FID.")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--ref_dir", type=Path, default=Path("ref"))
    parser.add_argument("--scorer_path", type=Path, default=Path("scoring_program/score.py"))
    parser.add_argument("--validation_input_dir", type=Path, default=Path("validation_input"))
    parser.add_argument("--validation_output_dir", type=Path, default=Path("validation_output"))
    parser.add_argument("--expected_count", type=positive_int, default=FINAL_SAMPLE_COUNT)
    parser.add_argument("--image_size", type=positive_int, default=IMAGE_SIZE)
    parser.add_argument("--skip_fid", action="store_true", help="Only run PNG count/mode/size preflight.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pngs = validate_png_directory(
        results_dir=args.results_dir,
        expected_count=args.expected_count,
        image_size=args.image_size,
    )
    print(f"Preflight passed: {len(pngs)} RGB PNG files at {args.image_size}x{args.image_size}")

    if args.skip_fid:
        return
    if args.expected_count != FINAL_SAMPLE_COUNT:
        raise ValueError("FID scoring should be run against the final 3000-image output set.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The provided scoring_program/score.py hardcodes torch.device('cuda:0'), "
            "but CUDA is not available. Preflight passed; run FID on a CUDA machine."
        )

    stage_validation_input(args.results_dir, args.ref_dir, args.validation_input_dir)
    scores_path = run_scorer(args.scorer_path, args.validation_input_dir, args.validation_output_dir)
    scores = read_scores(scores_path)
    print(f"Scores written to {scores_path}: {scores}")


if __name__ == "__main__":
    main()
