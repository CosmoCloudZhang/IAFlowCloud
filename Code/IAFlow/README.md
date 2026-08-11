# IAFlow

IAFlow prepares the NLA `A_theta(z, k)` surfaces, trains an unconditional
Conv1D autoencoder, evaluates selected checkpoints, and exports learned
latents for a later normalizing-flow stage. The baseline input is always
`(batch, 31, 101)` and no physical parameters are passed to the encoder or
decoder.

## Workflow

From the repository root, activate `MLConda`, install the local packages, and
run:

```bash
pip install -e .
python -m IAFlow.Scripts.PrepareData --config Config/NLA/AutoEncoderConv1D.yml
python -m IAFlow.Scripts.TrainAutoEncoder --config Config/NLA/AutoEncoderConv1D.yml
```

The cache is one source-ordered `Surfaces.npy` array. The authoritative HDF5
split indices select training, validation, and test rows without duplicating
the surfaces. Normalization is estimated exclusively from training rows.

For a short smoke run:

```bash
python -m IAFlow.Scripts.TrainAutoEncoder \
  --config Config/NLA/AutoEncoderConv1D.yml \
  --epochs 2 \
  --maximum-train-samples 1024 \
  --maximum-validation-samples 256 \
  --run-directory Runs/NLA/AutoEncoder/Conv1D/Smoke
```

After model selection on validation data, evaluate exactly once on the test
split and export latents:

```bash
python -m IAFlow.Scripts.EvaluateAutoEncoder \
  --config Config/NLA/AutoEncoderConv1D.yml \
  --checkpoint Runs/NLA/AutoEncoder/Conv1D/<run>/Best.pt --split test

python -m IAFlow.Scripts.ExportLatents \
  --config Config/NLA/AutoEncoderConv1D.yml \
  --checkpoint Runs/NLA/AutoEncoder/Conv1D/<run>/Best.pt
```

The current compressor’s latent summarizes all spectral variation. If future
datasets jointly sample cosmological and nuisance parameters, retain the
cosmology vector unchanged beside the exported spectral latent. A
cosmology-conditioned nuisance-only latent is a future extension, not a
property of this unconditional baseline.
