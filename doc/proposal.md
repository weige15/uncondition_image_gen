# Proposal: HW5 Latent Diffusion Professor Face Generation

## Objective
Build a complete HW5 solution for unconditional professor face generation. The implementation will train a diffusion model in VAE latent space using the provided professor-face images, then generate 3,000 PNG images at 256x256 resolution for Codabench FID evaluation and E3 submission.

The solution must follow the assignment rules: use the allowed pretrained `stabilityai/sd-vae-ft-mse` VAE only for pixel-to-latent and latent-to-pixel conversion, train the denoising U-Net from scratch, and avoid any other pretrained generative model or pretrained U-Net.

## Current Project State
The project contains the HW5 assignment PDF, public training images, sample training and inference code, reference FID statistics, and a scoring program.

Observed files and constraints:

- `hw5.pdf` defines the task as unconditional generation of professor face images from about 5,500 cropped and aligned images from NTU, NYCU, and NCCU scholar platforms.
- `public_data/sample_code/train.py` provides a scaffold that loads the allowed VAE, creates an empty `UNet2DModel`, encodes images into latent space, and leaves the diffusion training loss and evaluation sampler unimplemented.
- `public_data/sample_code/inference.py` provides a scaffold that loads a saved trained U-Net checkpoint and the allowed VAE, but leaves the reverse diffusion sampling loop unimplemented.
- `public_data/sample_code/requirements.txt` lists the expected package family: `torch`, `torchvision`, `numpy`, `tqdm`, `transformers`, `scipy`, and `tensorboard`.
- `scoring_program/score.py` computes FID using reference mean and covariance files under `ref/`, and requires the generated submission to contain exactly the expected number of PNG images at the expected image size.
- The E3 submission should unzip into `hw5_{student_id}/` containing `code/`, `model/`, and `results/`.

## Assumptions
The implementation will be based on the provided sample code rather than a new training framework.

The training data will initially be the provided `public_data/images/` folder. Additional scraped data is allowed by the assignment, but will be treated as optional because distribution mismatch can hurt FID.

The U-Net architecture can be changed, but it must be initialized from scratch. The proposal assumes a compact latent-space U-Net sized for 256x256 images encoded by the VAE into lower-resolution latents.

The output contract is 3,000 generated PNG files, each 256x256 RGB, with no training images copied into the result folder.

## Proposed Approach
Implement the missing latent diffusion training path in `train.py`.

The dataset loader will resize images to 256x256, convert them to RGB tensors, and normalize them to `[-1, 1]` for the VAE. During training, each image batch will be encoded with the frozen VAE, scaled by `vae.config.scaling_factor`, and noised using `DDPMScheduler.add_noise`. For each sample, the code will draw Gaussian noise and a random timestep, ask the U-Net to predict the added noise, and optimize mean squared error between predicted and true noise.

Define a legal from-scratch `UNet2DModel` architecture.

The model should match the latent tensor shape produced by the VAE and use reasonable capacity for the small face dataset. A starting point is a latent U-Net with `in_channels=4`, `out_channels=4`, multiple down/up blocks, attention in deeper blocks, and block widths chosen to fit available GPU memory. Architecture and hyperparameters should be configurable so experiments can compare FID against training cost.

Implement periodic sampling for qualitative checks.

The `generate_and_save_images` helper in `train.py` will start from random latent noise, run the scheduler timesteps in reverse, decode final latents through the VAE, clamp images to valid range, and save preview PNGs. This makes it possible to catch collapsed or invalid training before generating the full 3,000-image submission.

Implement the final inference pipeline in `inference.py`.

The inference script will load the trained U-Net checkpoint with `UNet2DModel.from_pretrained`, create the same scheduler configuration used in training, sample random latents in batches, run the reverse diffusion loop, decode with the frozen VAE, and save sequential PNGs such as `0000.png` through `2999.png`.

Add experiment controls and reproducibility.

Training and inference should expose CLI arguments for image directory, output directory, checkpoint directory, number of epochs or steps, batch size, learning rate, gradient accumulation, seed, save frequency, evaluation frequency, and sampling steps where applicable. Checkpoints should be saved with `save_pretrained()` so the sample inference loading path remains compatible.

Validate locally before Codabench upload.

After generation, use the provided scoring program to verify that the result folder contains exactly 3,000 PNG images at 256x256 and to estimate FID using `ref/test_mu.npy` and `ref/test_sigma.npy`. Also manually inspect a grid of generated images to confirm they are face-like, diverse, and not memorized training samples.

Package the final submission.

Prepare the required E3 layout with `code/` containing training and inference code, `model/` containing the trained U-Net checkpoint or a cloud link if too large, and `results/` containing the generated 3,000 PNG images. The packaging step should avoid copying any training images into `results/`.

## Milestones
1. Complete the baseline latent DDPM training loop and from-scratch U-Net configuration in `train.py`.
2. Complete reverse diffusion sampling in both preview generation and `inference.py`.
3. Run a short smoke test to confirm training, checkpoint saving, checkpoint loading, and image generation work end to end.
4. Train the model to a useful checkpoint, generate preview samples periodically, and choose the best checkpoint by local FID and visual quality.
5. Generate the final 3,000-image result set, validate PNG count and 256x256 resolution, then run the local FID scorer.
6. Assemble the E3 submission folder with `code/`, `model/`, and `results/`.

## Open Questions
The available GPU memory and training time budget are not specified. These will affect U-Net width, batch size, gradient accumulation, number of training steps, and whether faster schedulers or fewer inference steps should be used for final generation.

The target performance tier is not specified. A practical first goal is to beat the weak baseline FID threshold below 90, then iterate toward the strong baseline threshold below 50 if compute allows.

Whether to use additional scraped face data is unresolved. It is allowed by the assignment, but should only be added after the provided-data baseline is working and measurable.

## Validation Plan
Run a training smoke test on a small subset or a few batches to verify that the loss computation works, gradients update the U-Net, and checkpoints are written with `save_pretrained()`.

Run inference from a saved checkpoint with a small sample count first, then verify that every output file is PNG, RGB, and exactly 256x256.

Run the provided scoring program against a generated folder to check the image count, image size contract, and local FID estimate.

Inspect generated image grids during training and final inference to screen for blank outputs, severe artifacts, duplicated samples, and accidental inclusion of training images.

Before submission, confirm the final folder structure matches `hw5_{student_id}/code`, `hw5_{student_id}/model`, and `hw5_{student_id}/results`.
