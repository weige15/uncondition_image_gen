# Denoising Diffusion Probabilistic Models

- Authors: Jonathan Ho, Ajay Jain, Pieter Abbeel
- Year: 2020
- URL: https://arxiv.org/abs/2006.11239

## Key Ideas

DDPM trains a neural network to reverse a gradual Gaussian noising process. The practical training objective predicts the sampled Gaussian noise at randomly sampled timesteps, usually with an MSE loss.

## Relevance to HW5

This is the base method used by the current implementation:

- sample noise,
- sample timestep,
- add scheduler noise,
- predict noise with U-Net,
- optimize MSE.

The paper reports strong unconditional generation results, showing the method is viable, but good FID depends on architecture, training details, and sampling choices.

## Integration Takeaway

The current epsilon-prediction training target is appropriate. Poor FID should first be treated as an implementation/training-quality issue rather than evidence that DDPM is unsuitable.
