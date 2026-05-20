# Efficient Diffusion Training via Min-SNR Weighting Strategy

- Authors: Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, Baining Guo
- Year: 2023
- URL: https://arxiv.org/abs/2303.09556
- Paper page: https://huggingface.co/papers/2303.09556

## Key Ideas

The paper frames diffusion timesteps as conflicting training tasks and proposes Min-SNR-gamma weighting to balance timestep losses. The reported motivation is faster convergence and improved generation performance.

## Relevance to HW5

The baseline samples timesteps uniformly and gives each per-sample MSE equal weight. If FID remains high after EMA and model-capacity improvements, timestep loss imbalance may be part of the problem.

## Integration Takeaway

Add an optional loss weighting mode:

```bash
--loss_weighting min_snr --min_snr_gamma 5.0
```

Implementation sketch for epsilon prediction:

```python
alpha = scheduler.alphas_cumprod[timesteps]
snr = alpha / (1 - alpha)
weights = torch.minimum(snr, gamma * torch.ones_like(snr)) / snr
loss = (per_sample_mse * weights).mean()
```

This is legal because it uses no external pretrained assets.
