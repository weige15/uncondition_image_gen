# Improved Denoising Diffusion Probabilistic Models

- Authors: Alexander Quinn Nichol, Prafulla Dhariwal
- Year: 2021
- URL: https://proceedings.mlr.press/v139/nichol21a.html

## Key Ideas

This work improves DDPMs with better likelihood/sample-quality tradeoffs, improved noise schedules, and variance-related changes. It is a major practical follow-up to the original DDPM.

## Relevance to HW5

The assignment code currently uses a fixed DDPM scheduler and epsilon prediction. Improved DDPM suggests that schedule and sampling design can materially affect sample quality.

## Integration Takeaway

Potential future changes:

- expose scheduler variants,
- try cosine schedule if training and inference are changed consistently,
- consider improved variance only if there is enough implementation time.

For the immediate HW5 path, EMA and checkpoint selection are simpler and lower-risk.
