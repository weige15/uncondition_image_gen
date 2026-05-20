# GenAI HW5

Code and notes for the GenAI HW5 latent diffusion assignment.

## Layout

- `code/`: training, inference, validation, and packaging scripts
- `doc/`: design notes and task planning documents
- `literature/`: paper notes used during implementation
- `ref/`: reference statistics used by the local scorer
- `scoring_program/`: provided scoring script
- `public_data/sample_code/`: provided sample implementation files

Generated samples, checkpoints, validation scratch directories, and dataset images are ignored by Git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python code/train.py --image_dir public_data/images --output_dir outputs --run_name baseline
```

For a quick smoke test:

```bash
python code/train.py --subset_size 64 --max_train_steps 5 --eval_freq 0 --save_freq 0
```

## Generate Results

```bash
python code/inference.py --checkpoint_dir outputs/baseline/checkpoints/unet_final --output_dir results
```

## Validate

```bash
python code/validate_results.py --results_dir results --skip_fid
```

Run without `--skip_fid` on a CUDA machine to invoke the provided FID scorer.

## Package Submission

```bash
python code/package_submission.py --student_id YOUR_STUDENT_ID --checkpoint_dir outputs/baseline/checkpoints/unet_final --create_zip
```

