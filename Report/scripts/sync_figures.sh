#!/usr/bin/env bash

set -euo pipefail

script_directory="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
report_directory="$(dirname "$script_directory")"
default_project_root="$(dirname "$report_directory")"
project_root="${IAFLOW_PROJECT_ROOT:-$default_project_root}"
destination="$report_directory/figures"

mkdir -p "$destination"

copy_figure() {
  local source="$1"
  local target="$2"

  if [[ ! -f "$project_root/$source" ]]; then
    echo "Missing source figure: $project_root/$source" >&2
    return 1
  fi

  install -m 0644 "$project_root/$source" "$destination/$target"
  echo "Updated $target"
}

copy_figure "Figure/NLA/Formula/Scale_Redshift_Surface.pdf" \
  "ia_scale_redshift_surface.pdf"
copy_figure "Figure/NLA/Sampling/Final_Prior_Amplitude_Ensemble.pdf" \
  "sampling_amplitude_ensemble.pdf"
copy_figure "Figure/NLA/PCA/Explained_Variance.pdf" \
  "pca_explained_variance.pdf"
copy_figure "Figure/NLA/PCA/Reconstruction_Error_by_Rank.pdf" \
  "pca_reconstruction_error_by_rank.pdf"

echo "Figure synchronization complete: $destination"
