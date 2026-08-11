# Data

`General/Planck.json` supplies the fixed cosmology used by the current NLA
workflow. `NLA/Samples/NLA.hdf5` is the authoritative generated dataset;
`NLA/PCA`, `NLA/Cache`, and `NLA/Latents` contain reproducible derived
products and are not committed to Git.

The NLA HDF5 file keeps `coordinates`, `parameters`, `components`,
`diagnostics`, and `splits`. Diagnostics use `minimum`, `maximum`,
`log_dynamic_range`, and `extreme_flag`; `diagnostics.attrs["component"]`
identifies their target.

The present NLA file has fixed cosmology. For future joint cosmology and
nuisance sampling, retain the sampled cosmology vector losslessly alongside
the learned spectral latent; do not claim that an unconditional spectral latent
is nuisance-only.
