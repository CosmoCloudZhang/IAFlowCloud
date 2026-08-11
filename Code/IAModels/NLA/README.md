# NLA

`Model.py` implements the nonlinear-alignment amplitude factors. `Sampling.py`
defines the frozen nuisance prior, evaluates and validates sampled models, and
writes the authoritative NLA HDF5 dataset.

The primary learned target is the positive `components/A_theta` surface with
shape `(sample, 31, 101)`.

The physical model depends on PyCCL, so run generation and science notebooks
in the PyCCL-enabled cosmology environment.
