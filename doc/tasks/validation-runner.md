# Validation Runner

## Goal

Validate generated outputs against the assignment contract and run the provided local FID scorer. The smallest useful outcome is a non-mutating validation command that checks count, mode, size, and produces scorer output when dependencies allow.

## Inputs

- `doc/proposal.md`: Use `scoring_program/score.py`, `ref/test_mu.npy`, `ref/test_sigma.npy`, and `ref/config.json` to validate generated results before upload.
- `doc/detailed-design.md`: Stage or consume a Codabench-style `ref/` and `res/` layout without rewriting generated images.

## Tasks

- [ ] Implement a preflight check that counts PNG files and verifies each image is RGB and 256x256.
- [ ] Build or document a non-destructive validation staging layout with `ref/` and `res/` inputs for the provided scorer.
- [ ] Invoke `python scoring_program/score.py --input_dir <validation_input> --output_dir <validation_output> --config config.json`.
- [ ] Stop before FID or packaging if image count or size validation fails.
- [ ] Preserve generated images exactly as inference wrote them.
- [ ] Record or print the generated `scores.json` path and local FID when scoring succeeds.

## Done When

- [ ] A generated result directory with the wrong count or image size fails preflight clearly.
- [ ] A valid 3,000-image directory can be passed to the provided scorer layout.
- [ ] Validation does not mutate or rename generated PNG files.
