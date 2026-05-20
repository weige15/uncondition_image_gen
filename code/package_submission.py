import argparse
from pathlib import Path

from common import (
    FINAL_SAMPLE_COUNT,
    IMAGE_SIZE,
    copy_file,
    copy_tree_files,
    list_pngs,
    make_zip,
    require_directory,
    require_file,
    validate_png_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the HW5 E3 submission folder.")
    parser.add_argument("--student_id", required=True, help="Student ID used in hw5_{student_id}.")
    parser.add_argument("--code_dir", type=Path, default=Path("code"))
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--model_note", type=Path, default=None, help="Text file containing an accepted model-link note.")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_parent", type=Path, default=Path("."))
    parser.add_argument("--create_zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_dir is None and args.model_note is None:
        raise ValueError("Provide either --checkpoint_dir or --model_note.")
    if args.checkpoint_dir is not None and args.model_note is not None:
        raise ValueError("Provide only one of --checkpoint_dir or --model_note.")

    package_root = args.output_parent / f"hw5_{args.student_id}"
    if package_root.exists():
        raise FileExistsError(f"Package root already exists: {package_root}")

    source_results = require_directory(args.results_dir, "results_dir").resolve()
    training_images = Path("public_data/images").resolve()
    if source_results == training_images:
        raise ValueError("Refusing to package public_data/images as generated results.")
    validate_png_directory(args.results_dir, expected_count=FINAL_SAMPLE_COUNT, image_size=IMAGE_SIZE)

    code_dst = package_root / "code"
    model_dst = package_root / "model"
    results_dst = package_root / "results"
    package_root.mkdir(parents=True)

    copy_tree_files(args.code_dir, code_dst, patterns=("*.py", "requirements.txt"))
    if args.checkpoint_dir is not None:
        copy_tree_files(args.checkpoint_dir, model_dst, patterns=("*",))
    else:
        model_dst.mkdir()
        copy_file(require_file(args.model_note, "model_note"), model_dst / args.model_note.name)

    results_dst.mkdir()
    for path in list_pngs(args.results_dir):
        copy_file(path, results_dst / path.name)

    validate_png_directory(results_dst, expected_count=FINAL_SAMPLE_COUNT, image_size=IMAGE_SIZE)
    expected_dirs = {"code", "model", "results"}
    actual_dirs = {path.name for path in package_root.iterdir() if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise RuntimeError(f"Unexpected package directories: {sorted(actual_dirs)}")

    if args.create_zip:
        zip_path = args.output_parent / f"hw5_{args.student_id}.zip"
        make_zip(package_root, zip_path)
        print(f"Created {zip_path}")
    print(f"Package assembled at {package_root}")


if __name__ == "__main__":
    main()
