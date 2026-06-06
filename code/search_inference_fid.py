import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import CLASS_NAMES, FINAL_SAMPLE_COUNT, positive_float, positive_int


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def parse_class_count_grid(value: str) -> tuple[tuple[int, ...], ...]:
    grids = []
    for group in value.split(";"):
        group = group.strip()
        if not group:
            continue
        try:
            counts = tuple(int(item.strip()) for item in group.split(",") if item.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError("expected semicolon-separated count triplets") from exc
        if len(counts) != len(CLASS_NAMES):
            raise argparse.ArgumentTypeError(f"each count group must have {len(CLASS_NAMES)} values")
        if any(count < 0 for count in counts):
            raise argparse.ArgumentTypeError("class counts must be non-negative")
        grids.append(counts)
    if not grids:
        raise argparse.ArgumentTypeError("expected at least one class-count group")
    return tuple(grids)


def fmt_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an inference/FID grid search for HW5 DDPM checkpoints.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, default=Path("fid_search"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scheduler", choices=("ddpm", "ddim", "dpm_solver"), default="ddpm")
    parser.add_argument("--num_samples", type=positive_int, default=FINAL_SAMPLE_COUNT)
    parser.add_argument("--batch_size", type=positive_int, default=16)
    parser.add_argument("--num_inference_steps", type=positive_int, default=1000)
    parser.add_argument("--seeds", type=parse_int_list, default=(1234, 42, 2025, 777, 3407))
    parser.add_argument("--initial_noise_scales", type=parse_float_list, default=(1.0,))
    parser.add_argument("--guidance_scales", type=parse_float_list, default=(1.0,))
    parser.add_argument(
        "--class_count_grid",
        type=parse_class_count_grid,
        default=((1701, 695, 604), (1680, 710, 610), (1720, 680, 600), (1701, 710, 589)),
        help="Semicolon-separated ntu,nccu,nycu count triplets.",
    )
    parser.add_argument("--target_fid", type=positive_float, default=20.0)
    parser.add_argument("--ref_dir", type=Path, default=Path("ref"))
    parser.add_argument("--scorer_path", type=Path, default=Path("scoring_program/score.py"))
    parser.add_argument("--reuse_existing", action="store_true", help="Score existing candidate outputs when present.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.num_samples != FINAL_SAMPLE_COUNT:
        raise ValueError(f"FID search requires the official {FINAL_SAMPLE_COUNT}-image output set")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.work_dir / "summary.jsonl"
    best: tuple[float, str] | None = None

    for seed in args.seeds:
        for counts in args.class_count_grid:
            if sum(counts) != args.num_samples:
                raise ValueError(f"class counts {counts} sum to {sum(counts)}, expected {args.num_samples}")
            count_label = "-".join(str(count) for count in counts)
            for noise_scale in args.initial_noise_scales:
                for guidance_scale in args.guidance_scales:
                    name = (
                        f"seed{seed}_counts{count_label}_noise{fmt_float(noise_scale)}"
                        f"_cfg{fmt_float(guidance_scale)}"
                    )
                    results_dir = args.work_dir / name / "results"
                    validation_input = args.work_dir / name / "validation_input"
                    validation_output = args.work_dir / name / "validation_output"

                    if not args.reuse_existing or not results_dir.exists():
                        inference_cmd = [
                            sys.executable,
                            "code/inference.py",
                            "--checkpoint_dir",
                            str(args.checkpoint_dir),
                            "--output_dir",
                            str(results_dir),
                            "--device",
                            args.device,
                            "--scheduler",
                            args.scheduler,
                            "--num_samples",
                            str(args.num_samples),
                            "--batch_size",
                            str(args.batch_size),
                            "--num_inference_steps",
                            str(args.num_inference_steps),
                            "--seed",
                            str(seed),
                            "--class_counts",
                            ",".join(str(count) for count in counts),
                            "--initial_noise_scale",
                            str(noise_scale),
                            "--guidance_scale",
                            str(guidance_scale),
                        ]
                        run_command(inference_cmd)

                    validation_cmd = [
                        sys.executable,
                        "code/validate_results.py",
                        "--results_dir",
                        str(results_dir),
                        "--expected_count",
                        str(args.num_samples),
                        "--ref_dir",
                        str(args.ref_dir),
                        "--scorer_path",
                        str(args.scorer_path),
                        "--validation_input_dir",
                        str(validation_input),
                        "--validation_output_dir",
                        str(validation_output),
                    ]
                    run_command(validation_cmd)
                    with open(validation_output / "scores.json", "r", encoding="utf-8") as handle:
                        fid = float(json.load(handle)["FID"])
                    record = {
                        "name": name,
                        "fid": fid,
                        "seed": seed,
                        "class_counts": counts,
                        "initial_noise_scale": noise_scale,
                        "guidance_scale": guidance_scale,
                        "results_dir": str(results_dir),
                    }
                    with open(summary_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    if best is None or fid < best[0]:
                        best = (fid, name)
                    print(f"{name}: FID={fid:.4f}; best={best[0]:.4f} ({best[1]})", flush=True)
                    if fid < args.target_fid:
                        print(f"Target reached: {fid:.4f} < {args.target_fid}", flush=True)
                        return

    if best is not None:
        print(f"Best candidate: {best[1]} with FID={best[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
