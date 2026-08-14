# IAFlowCloud

IAFlowCloud generates intrinsic-alignment (IA) spectra and learns compact
representations of them. The active physical model is NLA. Its positive
`A_theta(z, k)` surface is sampled on 31 redshifts and 101 wavenumbers, then a
Conv1D autoencoder compresses each `(31, 101)` surface to two latent variables.
A normalizing flow will later map that two-dimensional latent distribution to a
standard normal distribution.

## Project structure

- `Code/ia_models/general` contains coordinate, HDF5, and data-split utilities.
- `Code/ia_models/nla` contains the NLA equations, prior, sampling, validation,
  and atomic HDF5 generation.
- `Code/iaflow` contains configuration, cache preparation, architectures, training,
  evaluation, checkpointing, inference, and latent export.
- `Config/NLA` contains the authoritative Conv1D experiment configuration.
- `Notebooks` contains the scientific derivation, sampling, PCA, and ML
  dashboards. Reusable implementation stays in Python modules.
- `Data`, `Figure`, and `Runs` contain generated products and are ignored by
  Git except for the fixed Planck input.

TATT, halo, and hybrid implementations will be added only when working code is
available; the repository does not maintain empty model placeholders.

## Python source style

`Code/ia_models/nla/model.py` is the formatting reference for active Python
modules. Module, class, function, and method docstrings put their opening and
closing triple quotes on separate lines. Function docstrings begin with a
direct summary, add context only where it clarifies the scientific or runtime
contract, and use the following sections when applicable:

```text
Arguments:
    name (type):
        Meaning, units, shape, and constraints.

Returns:
    result (type):
        Meaning and shape of the returned value.
```

Module-level multiline signatures list one argument per line and retain a
trailing comma. Class methods may remain compact when their signatures are
short. Every blank separator line deliberately retains the indentation of the
innermost statement suite it separates, including nested `if`, `for`, `while`,
`try`, `with`, and `match` blocks. A separator after a nested block dedents to
the surrounding suite. `.editorconfig` therefore disables automatic
trailing-whitespace removal for Python files. Do not run Black, `ruff format`,
or Ruff rule `W293`, because they erase this project-specific visual structure.
Run `python3 ~/.codex/skills/python-style/scripts/check_python_style.py Code
Notebooks` for the project-specific style audit and `ruff check Code` for
static linting.

## Data flow and scientific split policy

```text
NLA parameters
    -> Data/NLA/Samples/NLA.hdf5
    -> source-ordered log10 cache
    -> Conv1D autoencoder
    -> two-dimensional latent
    -> later normalizing flow
```

`Data/NLA/Samples/NLA.hdf5` is authoritative. It stores coordinates,
13 shape parameters, factorized components, diagnostics, and disjoint
train/validation/test indices. The HDF5 structure is intentionally not
version-labelled.

The ML cache contains exactly:

- `Surfaces.npy`: one source-ordered, memory-mappable `float32` array;
- `Normalization.npz`: training-only mean surface and one global RMS;
- `Metadata.json`: compact provenance and split sizes, but no copied indices.

Training uses only the training split. Architecture and hyperparameters are
selected only with validation data. Test target rows are read once, after all
choices are frozen. Smoke runs must never use the test split.

## Local environments

Create or update the lightweight ML environment:

```bash
conda env create -f MLConda.yml
# If MLConda already exists:
conda env update -n MLConda -f MLConda.yml --prune

conda activate MLConda
python -m pip install -e .
python -m ipykernel install --user --name MLConda --display-name "Python (MLConda)"
python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
```

`MLConda.yml` pins Python 3.12, NumPy 2.0, PyTorch 2.10, and Ruff 0.16.2.
The numerical versions match the Colab 2026.04 runtime. The environment also
installs TensorBoard, `torchinfo`, and `zuko`. Check the importable Python code
from the repository root with:

```bash
ruff check Code
```

The physical-model and PCA notebooks use a separate PyCCL environment:

```bash
conda env create -f CosmoConda.yml
# If CosmoConda already exists:
conda env update -n CosmoConda -f CosmoConda.yml --prune

conda activate CosmoConda
python -m pip install -e .
python -m ipykernel install --user --name CosmoConda --display-name "Python (CosmoConda)"
```

## Notebook order

Run notebooks from a fresh kernel in this order:

1. `Notebooks/General/CCL.ipynb`
2. `Notebooks/NLA/Formula.ipynb`
3. `Notebooks/NLA/Sampling.ipynb`
4. `Notebooks/NLA/Power.ipynb`
5. `Notebooks/NLA/PCA.ipynb`
6. `Notebooks/IAFlow/NLA/AutoEncoder.ipynb`

Sampling and PCA regenerate their data products, fitted model, diagnostics, and
figures whenever they run. Each notebook ends with consistency checks; these
and the runtime checks in the Python modules replace a separate `Tests` package.

## Prepare, train, and resume

Run commands from the repository root with `MLConda` active:

```bash
python -m iaflow.scripts.prepare_data \
  --config Config/NLA/AutoEncoderConv1D.yml

python -m iaflow.scripts.train_autoencoder \
  --config Config/NLA/AutoEncoderConv1D.yml
```

For a validation-only smoke run:

```bash
python -m iaflow.scripts.train_autoencoder \
  --config Config/NLA/AutoEncoderConv1D.yml \
  --epochs 2 \
  --maximum-train-samples 1024 \
  --maximum-validation-samples 256 \
  --run-directory Runs/NLA/AutoEncoder/Conv1D/Smoke
```

Resume from `Last.pt`; `--epochs` is the new total epoch count:

```bash
python -m iaflow.scripts.train_autoencoder \
  --config Config/NLA/AutoEncoderConv1D.yml \
  --resume Runs/NLA/AutoEncoder/Conv1D/Smoke/Last.pt \
  --epochs 3 \
  --maximum-train-samples 1024 \
  --maximum-validation-samples 256
```

Resume restores optimizer, scheduler, early-stopping, random-number, and
data-loader state. All settings except total epochs and device must match the
original run.

Each run records the original resolved configuration, compact data provenance,
architecture, environment, history, checkpoints, validation metrics, and any
resume events. Stored paths are repository-relative so runs can move between
local and cloud machines.

## Evaluation and latent export

Validation evaluation is unrestricted:

```bash
python -m iaflow.scripts.evaluate_autoencoder \
  --config Config/NLA/AutoEncoderConv1D.yml \
  --checkpoint Runs/NLA/AutoEncoder/Conv1D/<run>/Best.pt \
  --split validation
```

After model selection is permanently frozen, evaluate the complete test split
once. The explicit confirmation is required, partial test evaluation is
forbidden, and an existing final result is never overwritten:

```bash
python -m iaflow.scripts.evaluate_autoencoder \
  --config Config/NLA/AutoEncoderConv1D.yml \
  --checkpoint Runs/NLA/AutoEncoder/Conv1D/<final-run>/Best.pt \
  --split test \
  --confirm-final-test
```

Export ordered train/validation/test latents after final checkpoint selection:

```bash
python -m iaflow.scripts.export_latents \
  --config Config/NLA/AutoEncoderConv1D.yml \
  --checkpoint Runs/NLA/AutoEncoder/Conv1D/<final-run>/Best.pt \
  --include-test
```

Without `--include-test`, latent export contains only training and validation
groups and is safe for exploratory normalizing-flow development.

The primary acceptance criterion is at least `0.999` validation variance
recovered with a two-dimensional latent. Physical-space tail errors are always
reported but have no invented pass threshold.

For the final robustness study, run the frozen configuration with several
explicit seeds (for example `--seed 41`, `--seed 42`, and `--seed 43`) in
separate run directories. Compare validation results only; do not evaluate each
candidate on the test split.

## Google Colab

Use a hosted Colab runtime as a managed Python environment; do not install
Conda or replace its bundled PyTorch. Clone the repository, install the local
package and the two small missing ML packages, mount Drive for persistence,
then copy the large HDF5/cache to `/content` before training so repeated random
access does not run over Drive:

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone <repository-url> /content/IAFlowCloud
%cd /content/IAFlowCloud
!python -m pip install -e . torchinfo==1.8.0 zuko==1.6.0
```

Copy completed run directories back to Drive before ending the runtime. Free
hosted Colab is notebook-driven and should not be treated as a remote SSH
server. Colab's supported “local runtime” does the reverse: its web frontend
connects to Jupyter on hardware you control, optionally through SSH port
forwarding to your own remote machine.

- [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html)
- [Colab local runtimes](https://research.google.com/colaboratory/local-runtimes.html)
- [Colab usage restrictions](https://research.google.com/colaboratory/intl/en-GB/faq.html)

## Future cosmology-dependent datasets

The current NLA target is nuisance-only and uses fixed cosmology. When a future
dataset samples both cosmological and nuisance parameters, the intended joint
representation is

```text
[sampled cosmological parameters unchanged, compressed IA latent]
```

Calling the latent nuisance-only will then require a cosmology-conditioned
compressor; an unconditional spectral latent may otherwise mix both sources of
variation.
