# IAModels

Physical-model code is organized by implemented intrinsic-alignment family.
`General` holds coordinate, HDF5-schema, and split utilities that are not
specific to one model. `NLA` contains the nonlinear-alignment model and its
nuisance-prior sampling and HDF5 writer.

The HDF5 writer stores generic diagnostic paths and labels the diagnosed
component with `diagnostics.attrs["component"]`.
