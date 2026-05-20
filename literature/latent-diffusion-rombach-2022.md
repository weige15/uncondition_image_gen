# High-Resolution Image Synthesis with Latent Diffusion Models

- Authors: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer
- Year: 2022
- URL: https://arxiv.org/abs/2112.10752
- CVF page: https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper

## Key Ideas

Latent Diffusion Models train diffusion in a compressed autoencoder latent space rather than directly in pixel space. This reduces compute while preserving much of the perceptual structure needed for high-resolution image synthesis.

## Relevance to HW5

This is the closest conceptual match to the assignment. HW5 explicitly allows the pretrained `stabilityai/sd-vae-ft-mse` VAE only for pixel-to-latent and latent-to-pixel conversion, while requiring a from-scratch denoising U-Net.

## Integration Takeaway

The project architecture is aligned with LDM practice:

- encode 256x256 images to 4x32x32 latents,
- train the denoiser in latent space,
- decode final latents to RGB PNGs.

The main improvements should be training/sampling refinements, not a pipeline rewrite.
