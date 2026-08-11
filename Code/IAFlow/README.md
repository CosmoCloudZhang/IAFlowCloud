# IAFlow Conv1D autoencoder

The package implements a leakage-safe nonlinear compressor for the positive
`A_theta(z, k)` surface. The 31 redshift positions are represented as Conv1D
channels and convolution runs along the 101-point wavenumber axis. The default
model has a two-value linear bottleneck and no skip connections.

## Local workflow

From the repository root:

```bash
conda activate MLConda
python Code/PrepareData.py
python Code/TrainAutoEncoder.py
```

For a short integration check before a long run:

```bash
python Code/TrainAutoEncoder.py \
  --epochs 2 \
  --maximum-train-samples 1024 \
  --maximum-validation-samples 256 \
  --run-directory Runs/AutoEncoder/Smoke
```

Architecture and optimization settings live in
`Config/AutoEncoderConv1D.yml`. CLI overrides are deliberately limited to
common operational choices; create a second YAML file for a scientific
experiment so its full configuration is retained with the checkpoint.

The baseline disables per-sample activation normalization. GroupNorm is
available for controlled ablations, but it can suppress the global
amplitude-like variation that dominates the PCA spectrum.

Training writes `Best.pt`, `Last.pt`, periodic checkpoints, `History.json`,
`Summary.json`, the resolved configuration, an architecture summary, and
TensorBoard logs. The test split is never touched by training.

Resume an interrupted local or Colab run in the same directory. Here `--epochs`
is the desired total, not the number of additional epochs:

```bash
python Code/TrainAutoEncoder.py \
  --resume Runs/AutoEncoder/<run>/Last.pt \
  --epochs 300
```

After freezing the model using validation results:

```bash
python Code/EvaluateAutoEncoder.py \
  --checkpoint Runs/AutoEncoder/<run>/Best.pt \
  --split test

python Code/ExportLatents.py \
  --checkpoint Runs/AutoEncoder/<run>/Best.pt
```

`ExportLatents.py` preserves source row indices for every split and creates the
HDF5 input for the later normalizing-flow stage.

## Colab workflow

Clone or mount the repository, select a GPU runtime, install the pinned pip
packages from `MLConda.yml`, and run the same scripts from the repository root.
Conda does not need to be recreated inside Colab: its managed runtime already
provides Python, and a small `pip` setup cell is more robust there. Store the
prepared cache and run directory on persistent Drive storage if a session may
disconnect. Copying the active cache to `/content` before training is faster
than memory-mapping it directly from Drive; copy checkpoints back after each
run.

## Scientific acceptance rule

Use validation `variance_recovered = 1 - SSE/SST`, where SST is measured around
the training mean surface in `log10(A_theta)`. Report log10 RMSE and physical
relative-error tails as well. A normalizing flow may reshape the latent
distribution later, but it cannot improve autoencoder reconstruction.
`Summary.json` records whether the configured 0.999 validation target was met.
