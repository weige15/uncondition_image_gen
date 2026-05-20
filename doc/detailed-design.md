# Detailed Design

## Purpose

Design the HW5 unconditional professor face generation solution described in `doc/proposal.md` and constrained by `hw5.pdf`. The solution trains a from-scratch denoising U-Net in the latent space of the allowed `stabilityai/sd-vae-ft-mse` VAE, then generates exactly 3,000 RGB PNG images at 256x256 resolution for Codabench FID evaluation and E3 submission.

This document preserves the module boundaries from the proposal and high-level design. It is a design artifact only; it does not implement or run training.

## Source Proposal Summary

The proposal targets latent diffusion for professor face generation using the provided `public_data/images/` dataset. The pretrained VAE is allowed only for image-to-latent and latent-to-image conversion. The U-Net denoiser must be initialized and trained from scratch, with no other pretrained generative model, pretrained U-Net, CLIP generator, or copied training image in the generated results.

The implementation will complete the missing training and sampling logic in the provided sample-code style, expose reproducible command-line controls, save U-Net checkpoints with `save_pretrained()`, generate preview images during training, run final inference from a saved checkpoint, validate output count and size with the provided scoring program, and package the E3 submission as `hw5_{student_id}/code`, `hw5_{student_id}/model`, and `hw5_{student_id}/results`.

## Design Goals

- Use only the allowed `stabilityai/sd-vae-ft-mse` VAE as a frozen pixel/latent converter.
- Train the denoising U-Net from scratch on the provided professor-face images.
- Keep training, inference, validation, and packaging responsibilities separated.
- Make hyperparameters and paths configurable so the same code supports local smoke tests and longer Colab A100 runs.
- Save checkpoints in a format loadable by `UNet2DModel.from_pretrained()`.
- Generate exactly 3,000 final PNG files, each RGB and 256x256.
- Validate final outputs with the local scoring program and reference statistics before submission.
- Avoid any batch deletion workflow while experimenting or packaging.

## Non-Goals

- Do not use additional scraped data in the first implementation pass.
- Do not use a pretrained U-Net, pretrained diffusion model, CLIP-based generator, or any pretrained model other than the allowed VAE conversion model.
- Do not introduce a new training framework beyond the package family implied by the sample code.
- Do not automate checkpoint selection with hidden feedback loops from Codabench.
- Do not copy, transform, or include training images in `results/`.

## Architecture Overview

The system is a compact latent diffusion pipeline with explicit executable stages:

1. Training reads professor face PNGs, normalizes them to `[-1, 1]`, encodes them with the frozen VAE, adds scheduler noise at random timesteps, trains the from-scratch U-Net to predict the sampled noise, and writes checkpoints plus preview images.
2. Inference loads one saved U-Net checkpoint, starts from Gaussian latent noise, applies the reverse diffusion scheduler, decodes final latents through the frozen VAE, and writes sequential PNG files.
3. Validation points the provided scorer at a Codabench-style `input/ref` and `input/res` layout, checking the 3,000-image and 256x256 contracts before computing FID.
4. Packaging assembles code, model artifacts, and generated results into the E3 folder shape.

The core data contract is:

```text
RGB PNG image, 256x256
  -> normalized tensor in [-1, 1], shape [B, 3, 256, 256]
  -> VAE latent tensor scaled by vae.config.scaling_factor, expected shape [B, 4, 32, 32]
  -> noised latent tensor at timestep t
  -> U-Net predicted noise tensor, same shape as latent
  -> reverse-sampled latent tensor
  -> decoded RGB PNG image, 256x256
```

The expected latent shape follows the behavior of the allowed VAE for 256x256 inputs and must be verified by a smoke test before full training.

## Module Designs

### Configuration and CLI

#### Responsibility

Own command-line parsing, default values, reproducibility settings, device selection, and resolved runtime paths. This module does not own training logic, model construction, image loading, scoring, or filesystem cleanup.

#### Inputs and Outputs

Inputs:

- Training CLI arguments such as `--image_dir`, `--output_dir`, `--epochs`, `--max_train_steps`, `--batch_size`, `--gradient_accumulation_steps`, `--learning_rate`, `--seed`, `--save_freq`, `--eval_freq`, and U-Net capacity options.
- Inference CLI arguments such as `--checkpoint_dir`, `--output_dir`, `--num_samples`, `--batch_size`, `--num_inference_steps`, `--seed`, and `--device`.
- Validation and packaging arguments such as result directory, reference directory, scorer path, and student ID.

Outputs:

- A runtime config namespace or dataclass passed to the executable module.
- Logged configuration for reproducibility.

#### Internal Design

Use `argparse` in each executable script to keep the command line simple and compatible with the provided sample code. Defaults should support project-root execution, while all important paths must remain overrideable for Colab or packaging.

Shared defaults can be duplicated in scripts at first, but if implementation grows, a small `config.py` may centralize constants such as image size, latent channels, VAE model ID, scheduler beta values, and default sample count.

#### Dependencies

- Python standard library: `argparse`, `os`, `pathlib`, `random`.
- PyTorch and NumPy seed APIs for reproducibility.

#### Failure Handling

- Reject non-positive batch sizes, sample counts, epochs, or inference steps.
- Fail early if `--image_dir` or `--checkpoint_dir` does not exist.
- Warn or fail if `--num_samples` differs from 3,000 for final-generation mode.
- Log the selected device; if CUDA is requested but unavailable, fail rather than silently changing final-generation behavior.

#### Independent Test Plan

- Run each script with `--help` and confirm expected options appear.
- Instantiate config with smoke-test arguments and confirm all paths resolve relative to the project root.
- Verify setting the same seed produces the same first latent noise tensor in a tiny isolated script.

#### Open Questions

- The final `hw5_{student_id}` package name requires the exact student ID.
- The preferred default compute profile is unresolved because the available A100 training budget is not specified.

### Dataset Loader

#### Responsibility

Own discovery and loading of training images from the configured image directory. It converts each image to RGB, resizes it to 256x256, and normalizes pixels to the VAE input range. It does not encode images, apply diffusion noise, augment data, or write outputs.

#### Inputs and Outputs

Inputs:

- `image_dir`, defaulting to `public_data/images/` or the assignment-provided image folder.
- `image_size`, fixed to 256 for the assignment output contract.
- Loader controls: batch size, shuffle flag, number of workers, and optional smoke-test subset size.

Outputs:

- Batches of tensors with shape `[B, 3, 256, 256]`, dtype `float32`, normalized to approximately `[-1, 1]`.
- A deterministic sorted list of source PNG paths before shuffling.

#### Internal Design

Use a `torch.utils.data.Dataset` similar to the provided `ImageDataset`. The transform should convert all images with `PIL.Image.open(path).convert("RGB")`, resize to 256x256, convert to tensor, and normalize with channel-wise mean and standard deviation `[0.5, 0.5, 0.5]`.

The proposal does not require augmentation. Augmentations such as mild horizontal flip can be considered an experiment only after the baseline contract is working; they should be CLI-gated if added.

#### Dependencies

- `PIL.Image`.
- `torch.utils.data.Dataset` and `DataLoader`.
- `torchvision.transforms`.

#### Failure Handling

- If no PNG files are found, fail before training starts.
- If an image cannot be read, surface the file path in the exception.
- Keep training images read-only. No preprocessing step should overwrite source images.

#### Independent Test Plan

- Instantiate the dataset against `public_data/images/` and verify length is nonzero.
- Fetch one item and assert shape `[3, 256, 256]`, dtype `float32`, and value range compatible with `[-1, 1]`.
- Run a one-batch `DataLoader` smoke test with `batch_size=2` and `num_workers=0`.

#### Open Questions

- Whether to add data augmentation is unresolved and should be treated as an experiment, not a baseline design requirement.

### VAE Adapter

#### Responsibility

Own all interactions with the allowed pretrained VAE, including loading, freezing, encoding, decoding, eval mode, and latent scaling. This module does not train any VAE weights or decide diffusion timesteps.

#### Inputs and Outputs

Inputs:

- Normalized pixel tensor `[B, 3, 256, 256]` in `[-1, 1]` for encoding.
- Scaled latent tensor `[B, 4, 32, 32]` for decoding.

Outputs:

- Scaled latent tensor for diffusion training.
- Decoded image tensor in the VAE output range, clamped and converted by caller for PNG saving.

Public interface:

- `load_vae(device) -> AutoencoderKL`
- `encode_images(vae, pixel_values) -> latents`
- `decode_latents(vae, latents) -> images`

#### Internal Design

Load `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")`, move it to the selected device, call `eval()`, and set `requires_grad_(False)`.

Encoding samples from `vae.encode(pixel_values).latent_dist.sample()` and multiplies by `vae.config.scaling_factor`.

Decoding divides latents by `vae.config.scaling_factor`, calls `vae.decode(..., return_dict=False)[0]`, clamps to `[-1, 1]`, and maps to `[0, 1]` before conversion to PIL images.

#### Dependencies

- `diffusers.AutoencoderKL`.
- PyTorch no-grad/autocast utilities if mixed precision is later added.

#### Failure Handling

- Fail clearly if the VAE cannot be loaded because dependencies or weights are unavailable.
- Assert or log latent shape during smoke tests; unexpected latent dimensions must stop architecture assumptions from continuing.
- Keep the VAE in no-grad regions to avoid accidental memory growth.

#### Independent Test Plan

- Encode and decode one dataset batch without a U-Net.
- Assert the encoded latent channel count is 4.
- Assert decoded tensors can be converted to RGB 256x256 PNG files.
- Confirm all VAE parameters have `requires_grad == False`.

#### Open Questions

- None for the assignment baseline. The VAE model ID is specified by the sample code and proposal.

### U-Net Factory

#### Responsibility

Own construction of the legal from-scratch latent denoiser. It does not load pretrained U-Net weights, train the model, or run sampling.

#### Inputs and Outputs

Inputs:

- Architecture configuration: latent sample size, input channels, output channels, block channels, layers per block, down/up block types, and optional attention block positions.

Outputs:

- A randomly initialized `UNet2DModel` compatible with scaled VAE latents and scheduler timesteps.

#### Internal Design

Use `diffusers.UNet2DModel` initialized from config, not from pretrained weights. For 256x256 images encoded by the VAE, the expected diffusion sample size is 32 with 4 latent channels. The model must therefore use `in_channels=4`, `out_channels=4`, and a `sample_size` matching the verified latent spatial size.

The exact width/depth remains configurable because GPU memory and training budget are open questions. The implementation should start with a conservative baseline that can pass smoke tests, then scale capacity for longer runs.

Checkpoints must be saved using `unet.save_pretrained(checkpoint_dir)` so inference can load them with `UNet2DModel.from_pretrained(checkpoint_dir)`.

#### Dependencies

- `diffusers.UNet2DModel`.
- PyTorch for initialization and device movement.

#### Failure Handling

- Validate `in_channels == out_channels == latent_channels`.
- Validate the model output tensor shape equals the input latent tensor shape in a forward-pass smoke test.
- Do not load a checkpoint inside the factory unless the caller explicitly enters inference mode.

#### Independent Test Plan

- Build the U-Net on CPU or GPU and run a forward pass with synthetic latents shaped `[2, 4, 32, 32]` and random timesteps.
- Assert `model(sample, timestep).sample.shape == sample.shape`.
- Save to a temporary checkpoint directory and reload with `UNet2DModel.from_pretrained()`.

#### Open Questions

- Final architecture width, attention placement, and layers per block depend on available memory and training-time budget.

### Training Loop

#### Responsibility

Own optimization of the from-scratch U-Net. It coordinates data loading, VAE encoding, scheduler noise addition, U-Net noise prediction, loss computation, optimizer stepping, logging, checkpointing, and preview sampling triggers.

It does not own final 3,000-image inference, FID scoring, or E3 packaging.

#### Inputs and Outputs

Inputs:

- Runtime config.
- DataLoader batches of normalized images.
- Frozen VAE.
- Randomly initialized U-Net.
- `DDPMScheduler` configured for training.

Outputs:

- U-Net checkpoints.
- Training logs.
- Optional preview PNGs in a preview-only directory.

Training step contract:

```text
pixel_values -> latents
noise = randn_like(latents)
timesteps = randint(0, scheduler.num_train_timesteps, [batch])
noisy_latents = scheduler.add_noise(latents, noise, timesteps)
noise_pred = unet(noisy_latents, timesteps).sample
loss = mse_loss(noise_pred, noise)
```

#### Internal Design

Use the DDPM training objective from the proposal: predict the Gaussian noise added to scaled VAE latents. The scheduler should match the sample code baseline:

```python
DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
)
```

The loop should support gradient accumulation. Loss should be divided by `gradient_accumulation_steps` before backpropagation when accumulation is enabled, and optimizer updates should occur only at accumulation boundaries or the end of an epoch/step cap.

Checkpoint directories should be separate from preview image directories, for example:

```text
outputs/<run_name>/checkpoints/unet_<step>/
outputs/<run_name>/samples/
outputs/<run_name>/logs/
```

#### Dependencies

- Dataset Loader.
- VAE Adapter.
- U-Net Factory.
- `diffusers.DDPMScheduler`.
- `torch.optim.Adam` or `AdamW`.
- `torch.nn.functional.mse_loss`.
- `tqdm`.
- Optional TensorBoard logging, consistent with `requirements.txt`.

#### Failure Handling

- Stop early if loss becomes NaN or infinite.
- Save checkpoints only to explicit checkpoint directories.
- Keep preview outputs separate from final `results/`.
- If out-of-memory occurs, the operator should reduce batch size, model width, or enable more gradient accumulation through CLI settings.

#### Independent Test Plan

- Run one training step on synthetic latents without the Dataset Loader or VAE to verify loss/backprop/update mechanics.
- Run a tiny data-backed smoke test with a small subset and `max_train_steps=1` or equivalent to verify VAE encoding, scheduler noise, U-Net forward, optimizer step, and checkpoint write.
- Reload the checkpoint with `UNet2DModel.from_pretrained()` and run a forward pass.

#### Open Questions

- The exact maximum training steps, number of epochs, and target checkpoint cadence depend on available GPU time.
- The target selection policy between best local FID and manual visual-quality gating remains unresolved.

### Preview Sampler

#### Responsibility

Generate small qualitative previews during training from the current in-memory U-Net. It detects gross failures early, such as blank images, severe artifacts, or non-face outputs. It does not generate the final 3,000-image submission.

#### Inputs and Outputs

Inputs:

- Current U-Net.
- Frozen VAE.
- Sampling scheduler.
- Preview sample count and batch size.
- Current step or epoch for filename naming.

Outputs:

- Preview PNGs or grids under the training run's `samples/` directory.

#### Internal Design

Set U-Net and VAE to eval mode inside `torch.no_grad()`. Initialize random latents with the verified latent shape, run scheduler reverse timesteps, divide by `vae.config.scaling_factor`, decode, clamp, and save preview images.

The reverse process should use the same scheduler family and beta configuration as training unless an experiment explicitly changes inference steps. After preview generation, the Training Loop should restore `unet.train()`.

#### Dependencies

- VAE Adapter.
- `DDPMScheduler`.
- `torchvision.transforms.ToPILImage` or `torchvision.utils.save_image`.

#### Failure Handling

- Save previews outside final `results/` to avoid accidental submission.
- If generated tensors contain NaNs, fail the preview step and report the training step.
- Use deterministic preview seeds when comparing checkpoints visually.

#### Independent Test Plan

- With a randomly initialized U-Net, run preview generation for 1 to 4 images and confirm files are valid PNGs at 256x256.
- Verify preview generation leaves the U-Net able to return to train mode.

#### Open Questions

- Preview frequency and preview count should be tuned to training speed and storage budget.

### Inference Sampler

#### Responsibility

Generate final or test image sets from a saved U-Net checkpoint. It owns the final output count, image naming, batch-wise sampling, decoding, and PNG writing. It does not train, score, or package.

#### Inputs and Outputs

Inputs:

- `--checkpoint_dir` containing a U-Net saved with `save_pretrained()`.
- `--output_dir`.
- `--num_samples`, default 3,000.
- `--batch_size`.
- `--num_inference_steps`.
- `--seed`.
- `--device`.

Outputs:

- Sequential PNG files, preferably `0000.png` through `2999.png`, in the configured output directory.

#### Internal Design

Load the frozen VAE and the trained U-Net checkpoint. Create a scheduler with the same base training configuration, call `scheduler.set_timesteps(num_inference_steps)` for inference, and process batches until the requested sample count is reached.

For each batch:

1. Draw Gaussian latents with shape `[current_batch, 4, 32, 32]`.
2. Iterate over scheduler timesteps.
3. Predict noise with the U-Net.
4. Update latents with `scheduler.step(noise_pred, t, latents).prev_sample`.
5. Decode final latents with the VAE Adapter.
6. Save RGB PNGs with deterministic sequential filenames.

The output directory should be intentionally selected by the user. The sampler should not batch-delete old files; if the directory already contains PNGs, it should either fail or write into a new run-specific directory.

#### Dependencies

- VAE Adapter.
- `UNet2DModel.from_pretrained()`.
- `DDPMScheduler`.
- PyTorch and PIL/torchvision image saving.

#### Failure Handling

- Fail if checkpoint config cannot be loaded.
- Fail if output directory contains existing PNG files and overwrite mode is not explicit.
- Verify each saved image is RGB and 256x256 during small runs or final validation.
- Report how many images were written.

#### Independent Test Plan

- Run inference with `--num_samples 2` from a smoke-test checkpoint and assert two PNGs exist.
- Verify filenames are sequential and unique.
- Open each image with PIL and assert mode `RGB` and size `(256, 256)`.

#### Open Questions

- Final `num_inference_steps` should be selected by quality/runtime tradeoff after baseline training works.

### Validation Runner

#### Responsibility

Validate generated outputs against the assignment scoring contract and compute local FID using the provided scoring program. It does not modify generated images or select checkpoints by itself.

#### Inputs and Outputs

Inputs:

- Generated result directory containing PNG images.
- Reference files under `ref/`: `test_mu.npy`, `test_sigma.npy`, and `config.json`.
- `scoring_program/score.py`.

Outputs:

- Validation errors, if output count or image size is wrong.
- `scores.json` containing local FID when the scoring program succeeds.

#### Internal Design

The scorer expects an input directory containing `ref/` and `res/`. The validation workflow should create or use a non-destructive staging layout such as:

```text
validation_input/
  ref/
  res/
validation_output/
  scores.json
```

Then run:

```bash
python scoring_program/score.py --input_dir validation_input --output_dir validation_output --config config.json
```

The local `ref/config.json` specifies `image_size: 256`, `num_images: 3000`, `scores: ["fid"]`, reference statistics filenames, batch size, worker count, and verbose output.

#### Dependencies

- Provided `scoring_program/score.py`.
- `ref/config.json`, `ref/test_mu.npy`, `ref/test_sigma.npy`.
- PyTorch, torchvision, SciPy, NumPy, PIL, and the scorer's installed dependencies.

#### Failure Handling

- If the scorer reports an incorrect image count or size, stop before packaging.
- If CUDA is unavailable for the scorer, the current scorer code uses `cuda:0`; adapting it for CPU would be a separate implementation choice and should be documented if needed.
- Do not rewrite generated images during validation.

#### Independent Test Plan

- Run a preflight script that only counts PNGs and checks PIL mode/size before invoking FID.
- Run the provided scorer against a tiny invalid folder to confirm count validation catches the expected failure.
- Run full validation only after final inference creates 3,000 images.

#### Open Questions

- The exact validation staging directory is an implementation detail; it should be chosen to avoid overwriting experiment outputs.

### Packaging Assembler

#### Responsibility

Arrange final deliverables into the E3-required layout. It does not train, infer, or score. It should never copy training images into `results/`.

#### Inputs and Outputs

Inputs:

- Final code files.
- Selected trained U-Net checkpoint or a cloud-link note if the model artifact is too large.
- Final validated 3,000-image result directory.
- Student ID.

Outputs:

```text
hw5_{student_id}/
  code/
  model/
  results/
```

The final zip should be named `hw5_{student_id}.zip`.

#### Internal Design

Copy only the needed source files into `code/`, the selected checkpoint artifacts into `model/`, and the generated PNG files into `results/`. Include any requirements or run instructions needed for the TA to reproduce generation from the provided model.

Because the local instruction forbids batch deletion, the assembler should not clean existing package directories by recursively deleting them. If a stale package exists, create a fresh package directory name or ask the user to manually remove the stale directory.

#### Dependencies

- Python standard library or shell copy commands.
- Final artifacts produced by Training Loop and Inference Sampler.

#### Failure Handling

- Fail if student ID is missing.
- Fail if `results/` does not contain exactly 3,000 PNG files.
- Fail if any result file matches a known training path or if training data is accidentally selected as a source.
- Avoid overwriting existing package contents unless each overwritten file is explicitly targeted.

#### Independent Test Plan

- Dry-run package assembly by printing source and destination file counts.
- Verify package tree shape.
- Verify `results/` count and image size after package assembly.
- Optionally unzip the final archive into a temporary directory and verify the root folder name.

#### Open Questions

- The exact student ID is needed before final packaging.
- If the trained checkpoint is too large, the cloud-link format accepted by the TA should be confirmed before submission.

## Cross-Module Contracts

- Image size is 256 throughout dataset loading, decoding, inference outputs, validation, and packaging.
- Pixel tensors entering the VAE are normalized to `[-1, 1]`.
- Latents used by diffusion are multiplied by `vae.config.scaling_factor` after encoding and divided by that factor before decoding.
- The baseline latent shape for 256x256 images is expected to be `[B, 4, 32, 32]`; implementation must verify this before full training.
- The U-Net predicts noise, not denoised latents or velocity, unless a future experiment changes both training target and scheduler consistently.
- Training and inference should use compatible scheduler configuration.
- Checkpoints are saved with `UNet2DModel.save_pretrained()` and loaded with `UNet2DModel.from_pretrained()`.
- Preview outputs and final `results/` are separate directories.
- Final result images are sequential PNGs, RGB, 256x256, and exactly 3,000 files.
- Validation consumes generated results but does not mutate them.
- Packaging consumes only validated generated results and selected model/code artifacts.

## Test Strategy

Use layered validation so each module can fail close to the source:

1. CLI tests: run `--help` and parse smoke-test arguments.
2. Dataset tests: load one image and one batch; verify shape, dtype, and value range.
3. VAE tests: encode/decode one batch; verify frozen parameters and latent shape.
4. U-Net tests: forward synthetic latents and timesteps; verify output shape; save and reload a checkpoint.
5. Training smoke test: run one or a few update steps on a tiny subset; verify loss is finite and checkpoint files are written.
6. Preview sampler test: generate a few images from the smoke-test checkpoint and verify PNG validity.
7. Inference sampler test: generate a small count such as 2 or 8 images from a checkpoint and verify filenames, mode, and size.
8. Final inference validation: generate exactly 3,000 images, preflight count/size, then run the provided scorer with `ref/config.json`.
9. Packaging validation: verify `hw5_{student_id}/code`, `model`, and `results` exist and `results` preserves the validated output contract.

Known relevant scorer command:

```bash
python scoring_program/score.py --input_dir <validation_input> --output_dir <validation_output> --config config.json
```

The implementation test commands for training and inference will depend on the final script paths under `code/`; those paths are not yet implemented in this design step.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| U-Net is too small | Blurry or unrealistic samples, high FID. | Keep architecture configurable and scale width/depth after smoke tests pass. |
| U-Net is too large | Out-of-memory failures or slow iteration. | Start with conservative settings; use batch size and gradient accumulation controls. |
| Incorrect VAE latent scaling | Broken training, washed-out images, or unusable samples. | Centralize scaling in VAE Adapter and test encode/decode paths. |
| Scheduler mismatch | Inference samples differ from the trained denoising process. | Keep scheduler config shared and document any experimental changes. |
| Output directory contamination | Wrong image count or accidental submission of previews/old images. | Fail if output directory already has PNGs unless an explicit overwrite plan exists. |
| Training-image leakage | Assignment score can become zero. | Generate only from random latents; keep training and result directories separate. |
| Scorer environment mismatch | Local FID validation may fail to run. | First run count/size preflight; document scorer dependency or CUDA blockers if encountered. |
| Missing packaging metadata | E3 submission penalty. | Require student ID and verify exact folder shape before zipping. |

## Open Questions

- What student ID should be used for `hw5_{student_id}`?
- What GPU and training-time budget should define the main training profile?
- Should checkpoint selection use the lowest local FID only, or require manual visual-quality approval before final generation?
- Should any data augmentation be included after the baseline works?
- If the model checkpoint is too large for E3, what cloud-link format should be used in `model/`?
