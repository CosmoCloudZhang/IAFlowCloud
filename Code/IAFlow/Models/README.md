# IAFlow models

`Conv1D.py` implements the current compressor. It treats the 31 redshift
locations as channels and convolves along the 101 wavenumber positions.

Other architectures will be added as separate implementations when they are
ready; shared data preparation, training, evaluation, artifacts, and inference
remain outside this directory.
