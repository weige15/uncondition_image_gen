# DDPM CelebA-HQ EMA Project

- Project: `fusing/ddpm-celeba-hq-ema`
- URL: https://huggingface.co/fusing/ddpm-celeba-hq-ema/tree/main

## Key Ideas

This is a face-generation DDPM checkpoint using EMA weights. It is useful as a project reference showing that EMA is standard practice for face diffusion models.

## Relevance to HW5

The dataset domain is similar: aligned face generation. However, the checkpoint itself is not legal to use for HW5 because the assignment requires training the denoising U-Net from scratch and forbids pretrained generative U-Net weights.

## Integration Takeaway

Do not use the model weights. Do implement EMA locally and save EMA checkpoints derived only from the HW5 training run.
