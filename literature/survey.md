# Literature Survey: HW5 Professor Face Latent Diffusion

## Scope

The assignment is unconditional 256x256 professor-face generation with a frozen `stabilityai/sd-vae-ft-mse` VAE used only for pixel/latent conversion, and a denoising U-Net trained from scratch. The useful literature is therefore latent diffusion, unconditional DDPM training, face-generation diffusion practice, and legal training/sampling improvements that do not import pretrained generative weights.

## Current Baseline Diagnosis

The implemented baseline is a legal latent DDPM:

- frozen SD VAE encoder/decoder,
- from-scratch `UNet2DModel`,
- epsilon/noise prediction with MSE,
- DDPM scaled-linear beta schedule,
- 1000-step DDPM sampling.

Observed local result: final 3,000-image validation passes, but FID is about `100.82`. The training loss around `0.09-0.12` is not enough evidence of good sample quality; FID often depends strongly on checkpoint selection, EMA weights, U-Net capacity, data augmentation, and timestep/loss weighting.

## Most Relevant Sources

1. `ddpm-ho-2020.md`
   - Base unconditional diffusion recipe.
   - Confirms epsilon/noise prediction is a standard objective.

2. `improved-ddpm-nichol-dhariwal-2021.md`
   - Practical improvements to DDPM training and sampling.
   - Relevant for noise schedules and variance/sampling design.

3. `adm-diffusion-beats-gans-2021.md`
   - Shows architecture details and U-Net scale matter substantially for FID.
   - Supports trying a larger U-Net before assuming the method is wrong.

4. `latent-diffusion-rombach-2022.md`
   - Closest conceptual match: diffusion in autoencoder latent space.
   - Supports the assignment's VAE-latent training design.

5. `edm-karras-2022.md`
   - Shows sampling/noise design can dominate quality and efficiency.
   - More invasive than EMA, but useful if baseline improvements plateau.

6. `min-snr-hang-2023.md`
   - Loss reweighting by timestep/SNR can improve diffusion convergence.
   - Legal and relatively contained to implement.

7. `ddpm-celeba-hq-ema-project.md`
   - Face-generation project/checkpoint using EMA.
   - Do not use the weights, but it supports EMA as standard practice for face DDPMs.

## Recommended Integration Order

### 1. EMA Checkpointing

Highest-priority next change. Keep an exponential moving average copy of U-Net weights and save EMA checkpoints. Generate and score from EMA, not only raw training weights.

Why: DDPM and face-generation projects commonly sample from EMA-smoothed weights. It is legal because it is derived only from this run's scratch-trained weights.

### 2. Checkpoint FID Sweep

Evaluate multiple checkpoints, for example:

- `unet_40000`
- `unet_50000`
- `unet_60000`
- `unet_68000`
- `unet_final`

Why: final loss is not a reliable proxy for best FID. Best perceptual quality can occur before the last training step.

### 3. Larger U-Net

Try:

```bash
--block_out_channels 128,256,512,512
--layers_per_block 2
```

If this exceeds VRAM, keep the larger model and reduce physical batch size with gradient accumulation.

### 4. Horizontal Flip Augmentation

Add optional `--random_horizontal_flip`. Faces remain semantically valid under horizontal flip, and the training set is small enough that simple augmentation should help.

### 5. Min-SNR Loss Weighting

Add an option such as `--loss_weighting min_snr --min_snr_gamma 5.0`.

Why: the paper argues timestep objectives conflict and Min-SNR balances them. This is legal because it changes only the loss scalar weights.

## Lower Priority / Riskier Ideas

- Switching to EDM or a Karras scheduler: promising but requires more careful training/sampling consistency changes.
- v-prediction: useful in modern diffusion, but this code currently uses `DDPMScheduler` epsilon prediction; changing target parameterization needs careful scheduler support.
- Learned variance: likely too invasive for the remaining assignment timeline.
- Additional scraped data: allowed only if assignment permits, but distribution mismatch can hurt FID and raises packaging/provenance risk.

## Explicitly Not Legal / Not Recommended

- Loading pretrained Stable Diffusion U-Net weights.
- Loading CelebA-HQ or FFHQ pretrained diffusion checkpoints.
- Using CLIP, face-recognition, or classifier guidance at generation time unless assignment explicitly allows those pretrained models.
- Copying, transforming, or nearest-neighbor selecting training images into `results/`.

## Practical Next Experiment

Implement EMA first and run:

1. Resume/retrain baseline with EMA.
2. Save both raw and EMA U-Net checkpoints.
3. Generate 3,000 images from EMA checkpoints at several steps.
4. Validate with fresh staging directories.
5. Compare FID trajectory and preview grids.

Expected result: EMA should improve sample stability and usually lowers FID more reliably than simply training the current raw-weight baseline longer.
