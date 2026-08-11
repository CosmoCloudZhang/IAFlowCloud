# Notebooks

Run the active notebooks in this order:

1. `General/CCL.ipynb`
2. `NLA/Formula.ipynb`
3. `NLA/Sampling.ipynb`
4. `NLA/Power.ipynb`
5. `NLA/PCA.ipynb`
6. `IAFlow/NLA/AutoEncoder.ipynb`

Each notebook ends with a **Final consistency checks** section. Restart its
kernel, run all cells, confirm those assertions, and save the current outputs
before treating a generated artifact as valid.

The NLA scientific notebooks require PyCCL. Use `CosmoConda` (or another
PyCCL-enabled environment) for them; use `MLConda` for the autoencoder stage.
