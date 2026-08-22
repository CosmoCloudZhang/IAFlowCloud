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
- `Code/iaflow/autoencoder` contains one direct-AE model module for the explicit
  Conv1D and Conv2D paths, together with configuration, checkpoints, training,
  inference, and family-local commands.
- `Code/iaflow/pca_autoencoder` contains the independent frozen-PCA transform,
  coefficient cache, MLP model, weighted loss, checkpoints, training, and its
  own family-local commands.
- `Code/iaflow/core` contains the family-independent configuration, surface-data,
  metric, evaluation, diagnostic, run, runtime, serialization, and workflow
  services used by both model families. It never imports either model family.
- `Code/iaflow/comparison.py` is the single cross-family result-discovery and
  PCA-benchmarking layer. There are no duplicate top-level compatibility modules
  or shared command directory.
- `Config/NLA/Surface/Standard.yaml` and `Config/NLA/Training/Standard.yaml`
  contain the shared surface-data and optimization policies.
- `Config/NLA/AE/<architecture>/DepthXX.yaml` contains the six reusable direct-AE
  architecture-depth templates.
- `Config/NLA/PCA_AE/DepthXX.yaml` contains the additive frozen-PCA plus MLP
  templates; it uses separate runtime modules and commands from direct AE.
- `Scripts/NLA/Launch_AE.sh` and `Scripts/NLA/Launch_PCA_AE.sh` safely detach one
  architecture-depth sweep and keep its aggregate and stage logs together.
- `Scripts/NLA/Run_AE.sh` and `Scripts/NLA/Run_PCA_AE.sh` are the foreground
  workers used by those launchers and by explicit sequential queues.
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
    ├── PCA/{PCAValidationMetrics.json,PCATransformMetadata.json,
    │       pca_log10_A_theta_30_components.{joblib,npz}}
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

Historical direct-AE runs remain under their depth and latent directories. Their
checkpoints and embedded provenance are not rewritten; current model-selection
code distinguishes them from revised candidates through `ResolvedConfig.json`.

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
6. `Notebooks/NLA/PCA_AE.ipynb`
7. `Notebooks/NLA/ModelSelection.ipynb`

`Notebooks/NLA/PCA_AE.ipynb` validates the portable rank-30 basis, coefficient
cache, all architecture/latent combinations, and completed PCA-AE validation
runs. It does not read the test split.

## Validate and run AE experiments

All direct-AE templates share a 1000-epoch maximum, early stopping, common
prepared data, optimizer, batch size 512, evaluation batch size 512, seed, and
50-epoch archival checkpoint interval. Depth03, Depth04, and Depth05 are total
capacity tiers: each tier increases both convolutional depth and the dense path
between the flattened convolutional representation `F` and latent dimension `L`.
The decoder mirrors the configured encoder widths automatically.

| Capacity tier | Encoder dense path | Decoder dense path |
|---|---|---|
| Depth03 | `F -> 256 -> 64 -> 16 -> L` | `L -> 16 -> 64 -> 256 -> F` |
| Depth04 | `F -> 512 -> 256 -> 64 -> 16 -> L` | `L -> 16 -> 64 -> 256 -> 512 -> F` |
| Depth05 | `F -> 768 -> 512 -> 256 -> 64 -> 16 -> L` | `L -> 16 -> 64 -> 256 -> 512 -> 768 -> F` |

At latent dimension 6, the revised parameter counts are:

| Architecture | Depth03 | Depth04 | Depth05 |
|---|---:|---:|---:|
| Conv1D | 2,041,989 | 5,058,693 | 9,255,301 |
| Conv2D | 7,074,919 | 15,793,639 | 29,429,479 |

Conv2D uses one internal input channel and convolves jointly across redshift and
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

The detached launcher validates configuration shapes, checks the cache once,
then trains, evaluates against matched-rank PCA, generates validation-tail
diagnostics, and exports train/validation latents for every dimension:

```bash
bash Scripts/NLA/Launch_AE.sh Conv1D Depth03
bash Scripts/NLA/Launch_AE.sh Conv2D Depth03
```

The launcher returns immediately and prints the aggregate `Sweep.log` and
`Sweep.pid` paths. These files share the timestamped `SweepLogs` directory with
the detailed validation, preparation, training, evaluation, diagnostic, and
export logs, so a separate detached-log directory is unnecessary. Wait for the
`AE sweep completed` marker before launching another MPS sweep.

The worker is fail-fast and stage-aware. It continues or skips a candidate only
when its complete resolved data, model, training, and output policies match the
requested template and latent dimension. Historical and revised architectures
can therefore coexist below the same depth and latent parent directories without
resuming an incompatible checkpoint. Model-selection code retains revised
direct-AE candidates only when their resolved dense schedule matches the capacity
tier. The noninteractive worker gives its Python children a valid standard input
internally, including when launched detached. Run the worker directly only when
foreground execution is wanted:

```bash
caffeinate -i bash Scripts/NLA/Run_AE.sh Conv1D Depth03
```

```bash
bash Scripts/NLA/Launch_AE.sh --fresh Conv1D Depth03
```

`--fresh` requests a new, configuration-identical replicate. Without it, the
worker resumes a compatible incomplete run and skips completed stages. A fresh
run is not required to protect revised architectures from legacy runs.

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

Each complete `ValidationMetrics.json` stores a versioned matched-PCA
comparison. For every error metric, the reported fractional error reduction is
`1 - model_error / matched_PCA_error`; positive values mean improvement and
negative values mean degradation. Variance recovery is reported separately as
`100 * (model_variance_recovered - PCA_variance_recovered)` percentage points.
The fractional table includes log10 MSE, RMSE, and MAE, mean and maximum
physical relative errors, and the p95/p99 per-surface RMSE and maximum-error
summaries. Normalized MSE is omitted from that derived table because its
fractional reduction is identical to log10 MSE under the shared global RMS.
The legacy signed `autoencoder_minus_pca` fields remain in the artifact for
compatibility.

Both depth runners validate the current comparison schema before skipping the
evaluation stage. A historical result containing only absolute differences is
therefore re-evaluated from its existing best checkpoint; training, diagnostics,
and latent export remain independently resumable. A ratio of stored p95 or p99
summaries is not presented as a percentile of paired surface-by-surface ratios.

## PCA-AE experiments

PCA-AE uses the frozen rank-30 PCA transform followed by a symmetric MLP
coefficient autoencoder. PCA coefficients are one-dimensional features, so no
Conv1D folder is used for this family. Its independent layout is
`Config/NLA/PCA_AE/DepthXX.yaml` and
`Runs/NLA/PCA_AE/DepthXX/LatentXX/<run>`. Its detailed implementation lives in
`iaflow.pca_autoencoder`, while direct AE lives in `iaflow.autoencoder`. Both
reuse only the family-independent services in `iaflow.core`. Cross-family model
selection uses `iaflow.comparison`. The installed commands, configuration schemas, run
paths, checkpoint contents, and direct-AE behavior are unchanged.

Rank 28 is the smallest evaluated basis below the 5% validation global
maximum-relative-error ceiling, while rank 30 supplies the frozen front end.
Raw coefficients are standardized using the complete training split only. The
fixed weighted coefficient loss is exactly the trainable contribution to
normalized log-surface MSE; validation adds the cached rank-30 projection
residual before reporting the shared surface metric schema.

Prepare and validate the PCA-AE-only products:

```bash
iaflow-prepare-pca-ae-data --config Config/NLA/PCA_AE/Depth03.yaml
iaflow-validate-pca-ae-configs \
  --config Config/NLA/PCA_AE \
  --latent-dims 2 4 6 8 10
```

Run one model or launch a safely detached complete depth sweep:

```bash
iaflow-train-pca-autoencoder \
  --config Config/NLA/PCA_AE/Depth03.yaml \
  --latent-dim 6

bash Scripts/NLA/Launch_PCA_AE.sh Depth03
```

The PCA-AE runner has the same fail-fast, resume, validation, diagnostic, and
latent-export stages as the direct-AE runner but invokes only PCA-AE-specific
commands. Use `bash Scripts/NLA/Launch_PCA_AE.sh --fresh Depth03` to force new
configuration-identical PCA-AE candidates.
