# Packaging Assembler

## Goal

Assemble the final E3 submission folder from selected code, model artifacts, and validated generated results. The smallest useful outcome is a package tree shaped as `hw5_{student_id}/code`, `hw5_{student_id}/model`, and `hw5_{student_id}/results`.

## Inputs

- `doc/proposal.md`: Prepare the E3 layout with code, model, and 3,000 generated PNG results; avoid copying training images.
- `doc/detailed-design.md`: Require student ID, use validated results only, avoid recursive cleanup, and verify package shape before zipping.

## Tasks

- [ ] Require a student ID and construct the package root name as `hw5_{student_id}`.
- [ ] Copy only final source files and requirements into `code/`.
- [ ] Copy the selected U-Net checkpoint artifacts or an accepted model-link note into `model/`.
- [ ] Copy only validated generated PNG files into `results/`, never files from `public_data/images/`.
- [ ] Fail if the package root already exists with stale contents rather than recursively deleting it.
- [ ] Verify the package tree shape and confirm `results/` contains exactly 3,000 RGB 256x256 PNGs.
- [ ] Optionally create the final zip after package validation passes.

## Done When

- [ ] The package root contains exactly the required `code/`, `model/`, and `results/` directories.
- [ ] Packaged results preserve the validated output contract.
- [ ] No batch deletion or recursive cleanup is used during assembly.
