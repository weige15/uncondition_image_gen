# Vibe Coding Implementation Prompt

## Objective

Implement the HW5 unconditional professor face generation solution end to end. Build a legal latent diffusion pipeline that trains a from-scratch denoising U-Net in the latent space of the allowed `stabilityai/sd-vae-ft-mse` VAE, generates exactly 3,000 RGB PNG images at 256x256 resolution, validates them with the provided local FID scorer, and assembles the E3 submission layout.

The implementation must obey the assignment constraints: the pretrained VAE may be used only for pixel-to-latent and latent-to-pixel conversion, the U-Net denoiser must be initialized from scratch, no other pretrained generative model or pretrained U-Net may be used, and generated `results/` must not contain copied or transformed training images.

## Inputs

Read these planning artifacts first:

- `doc/proposal.md`
- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/tasks/progress.md`
- `doc/tasks/configuration-and-cli.md`
- `doc/tasks/dataset-loader.md`
- `doc/tasks/vae-adapter.md`
- `doc/tasks/u-net-factory.md`
- `doc/tasks/training-loop.md`
- `doc/tasks/preview-sampler.md`
- `doc/tasks/inference-sampler.md`
- `doc/tasks/validation-runner.md`
- `doc/tasks/packaging-assembler.md`

Inspect these current code and data areas before editing:

- `public_data/sample_code/train.py`
- `public_data/sample_code/inference.py`
- `public_data/sample_code/requirements.txt`
- `public_data/images/`
- `scoring_program/score.py`
- `ref/config.json`
- `ref/test_mu.npy`
- `ref/test_sigma.npy`
- `hw5.pdf` if any assignment wording is unclear

## Current Implementation

The repository currently contains planning docs, public data, reference FID statistics, a scoring program, and sample scaffold scripts. There is no completed final implementation under `code/` yet, and there is no existing test suite under `tests/`.

Important observed facts:

- `public_data/images/` contains 5,500 professor-face PNG files.
- `public_data/sample_code/train.py` defines an `ImageDataset`, loads `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")`, creates a placeholder `UNet2DModel()`, creates a `DDPMScheduler`, encodes images into scaled latents, and leaves preview sampling plus the DDPM training loss unimplemented.
- `public_data/sample_code/inference.py` loads a saved U-Net with `UNet2DModel.from_pretrained()`, loads the allowed VAE, creates the same scheduler family, and leaves the reverse diffusion loop unimplemented.
- `public_data/sample_code/requirements.txt` lists `torch`, `torchvision`, `numpy`, `tqdm`, `transformers`, `scipy`, and `tensorboard`. The sample scripts and design also require `diffusers`, `Pillow`, and the scorer imports `open_clip`; update requirements if needed.
- `scoring_program/score.py` expects an input directory with `ref/` and `res/`, writes `scores.json`, validates PNG count and image size when configured, and hardcodes `torch.device("cuda:0")`.
- `ref/config.json` sets `image_size` to `256`, `num_images` to `3000`, `scores` to `["fid"]`, `ref_mu` to `test_mu.npy`, `ref_sigma` to `test_sigma.npy`, `batch_size` to `32`, `num_workers` to `4`, and `verbose` to `true`.
- The intended final implementation should live under `code/` for packaging. Treat `public_data/sample_code/` as reference scaffolding, not the final source location.

Preserve the local operating constraint: do not use recursive or batch deletion commands such as `rm -rf`, `rmdir /s`, `rd /s`, `del /s`, or `Remove-Item -Recurse`. If cleanup is needed, delete only one clear file path at a time, or stop and ask the user to delete manually.

## Execution Model

Work autonomously until the project is implemented, verified, and ready for a training run. The main agent owns overall progress tracking, keeps `doc/tasks/progress.md` current, decomposes work into independent modules, spawns worker subagents for disjoint write scopes where useful, integrates their changes, and completes the implementation without human-in-the-loop checkpoints unless genuinely blocked.

When spawning worker agents, tell each worker they are not alone in the codebase, must not revert edits made by others, and must adapt their implementation to concurrent changes. Keep worker write scopes disjoint. The main agent should review and integrate all worker outputs before running final quality gates.

Make conservative assumptions when details are open. Do not ask the user about GPU budget, target FID, optional scraped data, or visual checkpoint selection before implementing the baseline; those are experiment decisions after the pipeline works. Ask concise questions only if implementation is blocked by information that cannot be inferred from the repo.

## Module Plan

### Workstream 1: Project Layout, CLI, and Shared Utilities

Own files under `code/` that define shared constants, argument parsing helpers, seed setup, path checks, and lightweight validation utilities. Create the final implementation in `code/`, using the sample scripts as scaffolds.

Required behavior:

- Provide executable training, inference, validation, and packaging entrypoints under `code/`.
- Expose CLI controls for paths, seed, device, batch size, training steps or epochs, gradient accumulation, learning rate, checkpoint cadence, preview cadence, U-Net capacity, inference sample count, and inference steps.
- Validate non-positive numeric arguments and missing required input directories before expensive model loading.
- Set seeds for Python, NumPy, and PyTorch where deterministic smoke tests require it.
- Keep defaults runnable from the project root.

Suggested files:

- `code/train.py`
- `code/inference.py`
- `code/validate_results.py`
- `code/package_submission.py`
- `code/common.py` or similarly small shared module if useful
- `code/requirements.txt`

### Workstream 2: Dataset Loader and VAE Adapter

Own image loading and legal VAE conversion code. Do not edit training loop internals beyond integrating the public API.

Required behavior:

- Discover PNGs in a deterministic sorted list from `public_data/images/` by default.
- Load each image with PIL, convert to RGB, resize to 256x256, convert to tensor, and normalize to `[-1, 1]`.
- Build a configurable `DataLoader` with batch size, shuffle, worker count, and optional subset size for smoke tests.
- Fail clearly if no PNG files are found or an image cannot be read, including the problematic path.
- Implement `load_vae(device)`, `encode_images(vae, pixel_values)`, and `decode_latents(vae, latents)` helpers.
- Load only `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")`, set it to eval mode, freeze all parameters, and apply `vae.config.scaling_factor` correctly.
- Verify that 256x256 inputs produce 4-channel latents, expected to be spatial size 32x32.

Suggested tests:

- Dataset length against `public_data/images/` is nonzero.
- One item has shape `[3, 256, 256]`, dtype `float32`, and values compatible with `[-1, 1]`.
- A one-batch loader works with `batch_size=2` and `num_workers=0`.
- VAE parameters all have `requires_grad == False`.
- Encoding and decoding one batch yields valid tensors suitable for 256x256 RGB PNG output.

### Workstream 3: U-Net Factory, Training Loop, and Preview Sampler

Own model construction, DDPM training, checkpointing, logs, and preview generation.

Required behavior:

- Build a legal from-scratch `diffusers.UNet2DModel` from config, never pretrained U-Net weights.
- Use defaults compatible with VAE latents: `sample_size=32`, `in_channels=4`, `out_channels=4`.
- Keep block channels, layers per block, and block types configurable so small local smoke tests and larger A100 runs are both possible.
- Use this baseline scheduler unless an experiment explicitly changes both training and inference consistently:

```python
DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
)
```

- Implement the DDPM noise-prediction training step:

```text
pixel_values -> scaled VAE latents
noise = torch.randn_like(latents)
timesteps = torch.randint(0, scheduler.num_train_timesteps, [batch])
noisy_latents = scheduler.add_noise(latents, noise, timesteps)
noise_pred = unet(noisy_latents, timesteps).sample
loss = mse_loss(noise_pred, noise)
```

- Implement optimizer stepping with gradient accumulation; divide loss during accumulation and step at accumulation boundaries or at the final partial accumulation.
- Stop with a clear error if loss becomes NaN or infinite.
- Save checkpoints with `unet.save_pretrained()` under explicit checkpoint directories, for example `outputs/<run_name>/checkpoints/unet_<step>/`.
- Keep preview outputs separate from final `results/`, for example `outputs/<run_name>/samples/`.
- Implement preview sampling under `torch.no_grad()` with U-Net and VAE eval mode, reverse diffusion over configured timesteps, VAE decode, PNG/grid saving, NaN detection, and restoration of `unet.train()`.

Suggested tests:

- Synthetic U-Net forward pass with latents `[2, 4, 32, 32]` and random timesteps returns the same shape.
- Save and reload a U-Net checkpoint with `UNet2DModel.from_pretrained()`.
- Run one or a few training steps on a small subset; verify finite loss, weight update, checkpoint write, and reload.
- Generate 1 to 4 random-weight preview images and verify PNG validity and 256x256 size.

### Workstream 4: Inference Sampler

Own final and smoke-test image generation from a saved checkpoint. Do not own training, scoring, or packaging.

Required behavior:

- Load the trained U-Net checkpoint with `UNet2DModel.from_pretrained(checkpoint_dir)`.
- Load the frozen allowed VAE.
- Create the training-compatible scheduler and call `scheduler.set_timesteps(num_inference_steps)`.
- Generate latents batch by batch with shape `[current_batch, 4, 32, 32]`.
- Run reverse diffusion using `scheduler.step(noise_pred, t, latents).prev_sample`.
- Decode final latents through the VAE adapter.
- Save deterministic sequential PNG filenames such as `0000.png` through `2999.png`.
- Default `--num_samples` to `3000`. For final-generation mode, warn or fail if it differs from 3,000; allow smaller counts for explicit smoke tests.
- Fail if the output directory already contains PNG files unless an explicit safe overwrite mode is implemented. Do not silently mix old and new images.

Suggested tests:

- Generate `--num_samples 2` from a smoke-test checkpoint.
- Verify filenames are unique and sequential.
- Open outputs with PIL and assert mode `RGB` and size `(256, 256)`.

### Workstream 5: Validation Runner and Packaging Assembler

Own output validation, local FID invocation, and E3 package assembly. Do not mutate generated images.

Required validation behavior:

- Implement a preflight check that counts PNG files and verifies each one is RGB and 256x256.
- Stage or document a non-destructive scorer layout:

```text
validation_input/
  ref/
  res/
validation_output/
  scores.json
```

- Invoke the provided scorer in this form after staging:

```bash
python scoring_program/score.py --input_dir <validation_input> --output_dir <validation_output> --config config.json
```

- Ensure `<validation_input>/ref/config.json`, `test_mu.npy`, and `test_sigma.npy` are present. The `--config config.json` path is relative to `<validation_input>/ref`.
- Stop before FID or packaging if count, mode, or size validation fails.
- Note that `scoring_program/score.py` currently hardcodes CUDA via `torch.device("cuda:0")`; if CUDA is unavailable, report that blocker or implement a clearly documented device option without changing the scoring semantics.

Required packaging behavior:

- Require a student ID and create `hw5_{student_id}/`.
- Assemble exactly:

```text
hw5_{student_id}/
  code/
  model/
  results/
```

- Copy only final source files and requirements into `code/`.
- Copy the selected U-Net checkpoint artifacts or an accepted model-link note into `model/`.
- Copy only validated generated PNG files into `results/`; never copy from `public_data/images/`.
- Fail if the package root already exists with stale contents. Do not recursively delete it.
- Verify the package tree shape and confirm packaged `results/` still contains exactly 3,000 RGB 256x256 PNGs.
- Optionally create `hw5_{student_id}.zip` after package validation passes.

## Testing and Quality Gates

Add focused tests because the repository currently has no test suite. Prefer `pytest` tests under `tests/` for pure logic and smoke-testable modules. Use small counts and CPU-friendly tests where possible; VAE/network-dependent tests may be marked or kept as explicit smoke commands if model weights are not locally available.

Required quality gates before completion:

```bash
python -m compileall code tests
python -m pytest
python code/train.py --help
python code/inference.py --help
python code/validate_results.py --help
python code/package_submission.py --help
```

Required smoke gates before declaring the implementation ready:

```bash
python code/train.py --image_dir public_data/images --output_dir outputs/smoke --max_train_steps 1 --batch_size 1 --num_workers 0 --preview_freq 1 --save_freq 1
python code/inference.py --checkpoint_dir outputs/smoke/checkpoints/<actual_smoke_checkpoint> --output_dir outputs/smoke_results --num_samples 2 --batch_size 1 --num_inference_steps <small_test_step_count>
python code/validate_results.py --results_dir outputs/smoke_results --expected_count 2 --image_size 256 --preflight_only
```

Run full scoring only after final inference creates exactly 3,000 images:

```bash
python code/validate_results.py --results_dir <final_results_dir> --expected_count 3000 --image_size 256 --ref_dir ref --scorer_path scoring_program/score.py --validation_input <validation_input> --validation_output <validation_output>
```

If formatting or linting tools are added to the repo, run their configured checks before finishing. If no formatter, linter, or type checker is configured, do not invent a heavyweight toolchain; document that no such gate exists and rely on `compileall`, `pytest`, CLI smoke tests, and scorer/preflight validation.

## Acceptance Criteria

The implementation is complete when all of the following are true:

- `code/` contains final training, inference, validation, packaging, and any shared helper files needed by the E3 package.
- `code/requirements.txt` or equivalent dependency documentation includes all packages actually required by the final scripts.
- The dataset loader reads `public_data/images/` deterministically and returns normalized RGB 256x256 tensors.
- The VAE adapter uses only `stabilityai/sd-vae-ft-mse`, freezes it, and applies latent scaling correctly.
- The U-Net factory constructs a from-scratch latent denoiser compatible with `[B, 4, 32, 32]` latents.
- The training loop completes at least a one-step data-backed smoke run, writes a reloadable `save_pretrained()` checkpoint, and keeps checkpoints/logs/previews separated.
- Preview sampling writes valid PNGs outside final `results/`.
- Inference from a saved checkpoint writes sequential RGB 256x256 PNGs and protects against output-directory contamination.
- Validation preflight catches wrong counts, modes, or sizes and can invoke `scoring_program/score.py` with the repo's `ref/config.json` layout when CUDA/scorer dependencies are available.
- Packaging creates the required `hw5_{student_id}/code`, `hw5_{student_id}/model`, and `hw5_{student_id}/results` structure without recursive deletion and without training-image leakage.
- `doc/tasks/progress.md` reflects completed module tasks.
- The repository passes the quality gates listed above, or any skipped gate is explicitly explained with the blocker.

## Uncertainty Protocol

Known open questions that should not block the baseline implementation:

- The exact student ID is needed only for final packaging; implement `--student_id` and fail clearly if it is missing.
- GPU memory and training-time budget are unknown; keep model capacity, batch size, accumulation, training steps, and inference steps configurable.
- The target FID tier is not specified; implement correctness first, then allow experiments toward FID below 90 and ideally below 50.
- Additional scraped data is allowed by the assignment but intentionally excluded from the first implementation pass.
- Checkpoint selection by lowest local FID versus manual visual quality is an experiment decision after the pipeline works.

When docs and implementation conflict, prefer the assignment constraints and detailed design, then update comments or documentation to make the resolution explicit. If a blocker remains after reasonable conservative assumptions, ask one concise question and stop only the blocked part of the work.
