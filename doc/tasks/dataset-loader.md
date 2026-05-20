# Dataset Loader

## Goal

Load professor-face PNGs from the configured image directory and return VAE-ready tensors. The smallest useful outcome is a deterministic dataset and DataLoader that produce RGB 256x256 tensors normalized to `[-1, 1]`.

## Inputs

- `doc/proposal.md`: Use `public_data/images/` initially and avoid copying or modifying training images.
- `doc/detailed-design.md`: Convert images to RGB, resize to 256x256, normalize with mean/std `0.5`, and expose batch-size, shuffle, worker, and optional subset controls.

## Tasks

- [ ] Implement a dataset that discovers PNG files with a deterministic sorted path list.
- [ ] Load each image with PIL, convert to RGB, resize to 256x256, convert to tensor, and normalize to `[-1, 1]`.
- [ ] Add DataLoader construction with configurable batch size, shuffle, worker count, and optional smoke-test subset size.
- [ ] Fail clearly if no PNG files are found or an image cannot be read, including the source path.
- [ ] Run a one-item dataset check for shape `[3, 256, 256]`, dtype `float32`, and expected value range.
- [ ] Run a one-batch DataLoader smoke test with `batch_size=2` and `num_workers=0`.

## Done When

- [ ] Dataset length is nonzero against `public_data/images/`.
- [ ] Loaded batches match `[B, 3, 256, 256]` and are normalized for the VAE.
- [ ] Source training images remain read-only and no preprocessing output is written into the image directory.
