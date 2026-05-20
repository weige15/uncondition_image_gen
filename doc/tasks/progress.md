# Task Progress

- [x] Configuration and CLI (`doc/tasks/configuration-and-cli.md`)
- [x] Dataset Loader (`doc/tasks/dataset-loader.md`)
- [x] VAE Adapter (`doc/tasks/vae-adapter.md`)
- [x] U-Net Factory (`doc/tasks/u-net-factory.md`)
- [x] Training Loop (`doc/tasks/training-loop.md`)
- [x] Preview Sampler (`doc/tasks/preview-sampler.md`)
- [x] Inference Sampler (`doc/tasks/inference-sampler.md`)
- [x] Validation Runner (`doc/tasks/validation-runner.md`)
- [x] Packaging Assembler (`doc/tasks/packaging-assembler.md`)

## Verification Notes

- `python -m py_compile code/*.py` passes.
- `python code/train.py --help`, `python code/inference.py --help`, `python code/validate_results.py --help`, and `python code/package_submission.py --help` pass.
- Dataset smoke check against `public_data/images/` finds 5,500 PNGs and returns `[B, 3, 256, 256]` float32 tensors in `[-1, 1]`.
- PNG validation preflight passes on a small synthetic RGB 256x256 smoke set.
- VAE smoke check loads `stabilityai/sd-vae-ft-mse`, confirms all parameters are frozen, encodes one dataset item to `[1, 4, 32, 32]`, and decodes to `[1, 3, 256, 256]`.
- U-Net smoke check builds a from-scratch tiny `UNet2DModel`, verifies output shape `[2, 4, 32, 32]`, saves it, and reloads it with `UNet2DModel.from_pretrained()`.
- One-step CPU training smoke test completes with finite loss and writes `/tmp/hw5_train_smoke_codex/checkpoints/unet_final`.
- Two-image CPU inference smoke test from that checkpoint writes sequential PNGs to `/tmp/hw5_infer_smoke_codex`, and validation preflight confirms RGB 256x256 outputs.
- Local FID was not run because the provided scorer hardcodes `cuda:0`; run `code/validate_results.py` without `--skip_fid` on a CUDA machine after generating the final 3,000 images.
