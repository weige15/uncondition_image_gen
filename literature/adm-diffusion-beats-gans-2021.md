# Diffusion Models Beat GANs on Image Synthesis

- Authors: Prafulla Dhariwal, Alexander Quinn Nichol
- Year: 2021
- URL: https://arxiv.org/abs/2105.05233

## Key Ideas

The paper shows that diffusion models can outperform GANs on image synthesis when paired with stronger U-Net architecture choices and careful ablations.

## Relevance to HW5

The current default U-Net uses `64,128,256,256` block channels. FID around `100.82` after a full baseline run may indicate insufficient model capacity, weak checkpoint selection, or missing training stabilizers such as EMA.

## Integration Takeaway

Try a larger from-scratch U-Net if VRAM allows:

```bash
--block_out_channels 128,256,512,512
--layers_per_block 2
```

This remains legal because the U-Net is initialized from scratch.
