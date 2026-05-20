# Elucidating the Design Space of Diffusion-Based Generative Models

- Authors: Tero Karras, Miika Aittala, Timo Aila, Samuli Laine
- Year: 2022
- URL: https://arxiv.org/abs/2206.00364
- NeurIPS page: https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html

## Key Ideas

EDM separates and studies diffusion design choices such as noise parameterization, preconditioning, training distribution, and sampler design. It shows that these choices can strongly improve FID and efficiency.

## Relevance to HW5

The current code uses a conventional DDPM scheduler. EDM-style changes could improve sample quality, but they are more invasive than EMA or Min-SNR because training and inference must be changed consistently.

## Integration Takeaway

Keep EDM as a second-stage direction if simpler changes plateau. For the immediate assignment, prioritize EMA and larger U-Net first.
