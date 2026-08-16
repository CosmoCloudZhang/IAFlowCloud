#!/usr/bin/env bash

set -euo pipefail

script_directory="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
project_root="$(dirname "$script_directory")"
conda_root="${IAFLOW_CONDA_ROOT:-/opt/homebrew/anaconda3}"
conda_environment="${IAFLOW_CONDA_ENVIRONMENT:-MLConda}"
conda_setup="$conda_root/etc/profile.d/conda.sh"
latent_dimensions=(02 04 06 08 10)

if [[ ! -f "$conda_setup" ]]; then
  echo "Conda shell setup was not found: $conda_setup" >&2
  echo "Set IAFLOW_CONDA_ROOT to the directory containing etc/profile.d/conda.sh." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$conda_setup"
conda activate "$conda_environment"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$conda_environment" ]]; then
  echo "Failed to activate Conda environment: $conda_environment" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
cd "$project_root"

for latent_dimension in "${latent_dimensions[@]}"; do
  config_path="Config/NLA/AutoEncoderConv1D/Latent${latent_dimension}.yml"
  if [[ ! -f "$config_path" ]]; then
    echo "Missing latent-dimension configuration: $config_path" >&2
    exit 1
  fi
done

sweep_timestamp="$(date '+%Y%m%d-%H%M%S')"
log_directory="Runs/NLA/AutoEncoder/Conv1D/SweepLogs/$sweep_timestamp"
mkdir -p "$log_directory"

run_logged_stage() {
  local stage_name="$1"
  shift
  local log_path="$log_directory/${stage_name}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $stage_name"
  "$@" 2>&1 | tee "$log_path"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed $stage_name"
}

echo "Project root: $project_root"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python executable: $(command -v python)"
echo "Sweep logs: $log_directory"

for latent_dimension in "${latent_dimensions[@]}"; do
  run_logged_stage \
  "PrepareData" \
  iaflow-prepare-data \
  --config "Config/NLA/AutoEncoderConv1D/Latent${latent_dimension}.yml" \
  --overwrite

  run_logged_stage \
    "Latent${latent_dimension}" \
    iaflow-train-autoencoder \
    --config "Config/NLA/AutoEncoderConv1D/Latent${latent_dimension}.yml" \
    --no-progress
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Latent-dimension sweep completed"
