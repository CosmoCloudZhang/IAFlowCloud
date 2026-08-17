# IAFlowCloud

IAFlowCloud generates intrinsic-alignment (IA) spectra and learns compact
representations of them. The active physical model is NLA. Its positive
`A_theta(z, k)` surface is sampled on 31 redshifts and 101 wavenumbers, then
Conv1D and Conv2D autoencoders compress each `(31, 101)` surface to 2, 4, 6, 8,
or 10 latent variables. A normalizing flow will later model the selected latent
distribution.

## Project structure

- `Code/ia_models/utilities` contains coordinate, HDF5, and data-split utilities.
- `Code/ia_models/nla` contains the NLA equations, prior, sampling, validation,
  and atomic HDF5 generation.
- `Code/iaflow/architectures` contains the shared autoencoder interface and the
  Conv1D and Conv2D implementations.
- `Code/iaflow/commands` contains descriptive installed commands for preparation,
  configuration validation, training, evaluation, diagnostics, and latent export.
- `Config/NLA/Surface/Standard.yaml` and `Config/NLA/Training/Standard.yaml`
  contain the shared surface-data and optimization policies.
- `Config/NLA/AE/<architecture>/DepthXX.yaml` contains the six reusable direct-AE
  architecture-depth templates.
- `Config/NLA/PCA_AE` is intentionally only a placeholder until the frozen-PCA
  plus MLP implementation is added.
- `Scripts/NLA/Run_AE.sh` runs one architecture-depth sweep sequentially.
- `Notebooks/NLA` contains the scientific derivation, sampling, PCA, AE,
  PCA_AE, and model-selection notebooks without another directory layer.
- `Reference` contains educational material outside the active workflow.

The package names remain intentionally distinct: `ia_models` is the collection
of physical IA model families, while `iaflow` is the learning workflow.

## Python source style

`Code/ia_models/nla/model.py` is the formatting reference for active Python
modules. Module, class, function, and method docstrings put their opening and
closing triple quotes on separate lines. Function docstrings begin with a
direct summary and use the following sections when applicable:

```text
Arguments:
    name (type):
        Meaning, units, shape, and constraints.

Returns:
    result (type):
        Meaning and shape of the returned value.
```

Module-level multiline signatures list one argument per line and retain a
trailing comma. Every blank separator line deliberately retains the indentation
of the innermost statement suite it separates. `.editorconfig` therefore
disables automatic trailing-whitespace removal for Python files. Do not run
Black, `ruff format`, or Ruff rule `W293`. Validate with:

```bash
python3 ~/.codex/skills/python-style/scripts/check_python_style.py Code Notebooks
ruff check Code
```

## Data and run organization

```text
Data/
├── Cosmology/Planck.json
└── NLA/
    ├── Samples/NLA.hdf5
    ├── Cache/
    │   ├── Surface/{Surfaces.npy,Normalization.npz,Metadata.json}
    │   └── PCA_AE/
    ├── PCA/{PCAValidationMetrics.json,pca_log10_A_theta_25_components.joblib}
    └── Latents/
        ├── AE/{Conv1D,Conv2D}
        └── PCA_AE/
```

`Data/NLA/Samples/NLA.hdf5` is authoritative. It stores coordinates, 13 shape
parameters, factorized components, diagnostics, and disjoint
train/validation/test indices. The surface cache is prepared once and reused by
every direct-AE run. Training uses only the training split, architecture choices
use only validation data, and the complete test split is read once after model
selection is frozen.

Candidate latent exports stay inside their run directories. Only a promoted,
selected latent artifact belongs under `Data/NLA/Latents`.

New runs use this canonical layout:

```text
Runs/NLA/AE/
├── Conv1D/
│   ├── Depth03/LatentXX/<run>
│   ├── Depth04/LatentXX/<run>
│   └── Depth05/LatentXX/<run>
└── Conv2D/
    ├── Depth03/LatentXX/<run>
    ├── Depth04/LatentXX/<run>
    └── Depth05/LatentXX/<run>
```

Historical 500-epoch Conv1D runs remain under
`Runs/NLA/AE/Conv1D/LatentXX`. Their checkpoints and embedded provenance are
not rewritten; the AE notebook can still discover them as the legacy baseline.

## Configuration resolution

Reusable templates intentionally contain neither `latent_dim` nor `run_name`.
`latent_dim` is a scientific runtime axis supplied by the command or sweep.
`run_name` is not part of the schema: the run manager generates a timestamped
directory or accepts an explicit one.

```text
Depth03.yaml + latent_dim=6 + concrete run directory
    -> strict ExperimentConfig
    -> ResolvedConfig.json
```

The resolved artifact persists the latent dimension, exact run directory,
shared-file hashes, architecture, training policy, and any smoke-run limits.
Evaluation, diagnostics, and latent export accept `--run-directory` and load
that resolved artifact, which prevents configuration/checkpoint mismatches.

## Environments

Create or update the lightweight ML environment:

```bash
conda env create -f MLConda.yaml
# If MLConda already exists:
conda env update -n MLConda -f MLConda.yaml --prune

conda activate MLConda
python -m pip install -e .
python -m ipykernel install --user --name mlconda --display-name "Python (MLConda)"
```

`MLConda.yaml` pins Python 3.12, NumPy 2.0, PyTorch 2.10, and Ruff 0.16.2.
The physical-model and PCA notebooks use the separate PyCCL environment:

```bash
conda env create -f CosmoConda.yaml
# If CosmoConda already exists:
conda env update -n CosmoConda -f CosmoConda.yaml --prune

conda activate CosmoConda
python -m pip install -e .
python -m ipykernel install --user --name cosmoconda --display-name "Python (CosmoConda)"
```

## Notebook order

Run notebooks from a fresh kernel in this order:

1. `Notebooks/NLA/Formula.ipynb`
2. `Notebooks/NLA/Sampling.ipynb`
3. `Notebooks/NLA/Power.ipynb`
4. `Notebooks/NLA/PCA.ipynb`
5. `Notebooks/NLA/AE.ipynb`
6. `Notebooks/NLA/ModelSelection.ipynb`

`Notebooks/NLA/PCA_AE.ipynb` currently checks only the reserved structure. It
becomes active after the frozen-PCA plus MLP implementation and templates exist.

## Validate and run AE experiments

All direct-AE templates share a 1000-epoch maximum, early stopping, common
prepared data, optimizer, batch sizes, seed, and 50-epoch archival checkpoint
interval. Conv1D capacity increases materially from Depth03 to Depth05. Conv2D
uses one internal input channel and convolves jointly across redshift and
log-wavenumber while preserving the public `(batch, 31, 101)` interface.

Validate all six templates across all five latent dimensions:

```bash
iaflow-validate-configs --config Config/NLA/AE --latent-dims 2 4 6 8 10
```

Prepare or check the shared cache once:

```bash
iaflow-prepare-data --config Config/NLA/AE/Conv1D/Depth03.yaml
```

Run one model directly:

```bash
iaflow-train-autoencoder \
  --config Config/NLA/AE/Conv1D/Depth03.yaml \
  --latent-dim 6
```

For a validation-only smoke run:

```bash
iaflow-train-autoencoder \
  --config Config/NLA/AE/Conv2D/Depth03.yaml \
  --latent-dim 2 \
  --epochs 2 \
  --maximum-train-samples 1024 \
  --maximum-validation-samples 256 \
  --run-directory Runs/NLA/AE/Conv2D/Depth03/Latent02/Smoke
```

Resume from `Last.pt`; `--epochs` is the new total:

```bash
iaflow-train-autoencoder \
  --config Config/NLA/AE/Conv2D/Depth03.yaml \
  --latent-dim 2 \
  --run-directory Runs/NLA/AE/Conv2D/Depth03/Latent02/Smoke \
  --resume Runs/NLA/AE/Conv2D/Depth03/Latent02/Smoke/Last.pt \
  --epochs 3
```

The sequential runner validates configuration shapes, checks the cache once,
then trains, evaluates against matched-rank PCA, generates validation-tail
diagnostics, and exports train/validation latents for every dimension:

```bash
caffeinate -i bash Scripts/NLA/Run_AE.sh Conv1D Depth03
caffeinate -i bash Scripts/NLA/Run_AE.sh Conv2D Depth03
```

The runner is fail-fast and stage-aware. It continues the latest incomplete run
or skips artifacts already completed. Set `IAFLOW_FORCE_NEW_RUN=1` to request a
new run even when a completed candidate exists. The noninteractive runner gives
its Python children a valid standard input internally, including when launched
detached. MPS jobs must run sequentially.

## Evaluation, diagnostics, and latent export

Compare one completed run with PCA on the complete validation split:

```bash
iaflow-evaluate-autoencoder \
  --run-directory Runs/NLA/AE/Conv1D/Depth03/Latent06/<run> \
  --split validation \
  --pca-metrics Data/NLA/PCA/PCAValidationMetrics.json
```

Generate reusable error maps and worst surfaces:

```bash
iaflow-diagnose-autoencoder \
  --run-directory Runs/NLA/AE/Conv1D/Depth03/Latent06/<run>
```

Export ordered train and validation latents into the run:

```bash
iaflow-export-latents \
  --run-directory Runs/NLA/AE/Conv1D/Depth03/Latent06/<run>
```

After model selection is permanently frozen, evaluate the complete test split
once and then explicitly include test latents:

```bash
iaflow-evaluate-autoencoder \
  --run-directory Runs/NLA/AE/<architecture>/<depth>/LatentXX/<final-run> \
  --split test \
  --confirm-final-test

iaflow-export-latents \
  --run-directory Runs/NLA/AE/<architecture>/<depth>/LatentXX/<final-run> \
  --include-test
```

Select the smallest latent dimension that reaches at least `0.999` validation
variance recovery and outperforms PCA at the same dimension in both validation
variance recovery and log10 MSE. Physical-space tail errors are always reported
but have no invented pass threshold.

PCA and autoencoder evaluation use the same reconstruction-metric contract.
Physical relative error is `abs(10**(prediction_log10 - target_log10) - 1)`;
global maxima and per-surface tail percentiles are compared but remain
diagnostics rather than additional model-selection thresholds.

## Future PCA_AE experiments

PCA_AE will use a frozen rank-25 PCA transform followed by an MLP coefficient
autoencoder. PCA coefficients are one-dimensional features, so no Conv1D folder
is retained for that family. The future direct layout will be
`Config/NLA/PCA_AE/DepthXX.yaml` and
`Runs/NLA/PCA_AE/DepthXX/LatentXX/<run>`. Coefficients may be standardized for
conditioning, but the objective must unscale them before evaluating the
surface-space reconstruction loss. No PCA_AE scientific implementation or
configuration is claimed yet.
