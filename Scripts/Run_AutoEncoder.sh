#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIRECTORY")"
CONFIGURATION_DIRECTORY="$PROJECT_ROOT/Config/NLA/AutoEncoderConv1D"
CONDA_ROOT="${IAFLOW_CONDA_ROOT:-/opt/homebrew/anaconda3}"
CONDA_ENVIRONMENT="${IAFLOW_CONDA_ENVIRONMENT:-MLConda}"
CONDA_SETUP="$CONDA_ROOT/etc/profile.d/conda.sh"
LATENT_DIMENSIONS=(02 04 06 08 10)


activate_conda_environment() {
    if [[ ! -f "$CONDA_SETUP" ]]; then
        echo "Conda shell setup was not found: $CONDA_SETUP" >&2
        echo "Set IAFLOW_CONDA_ROOT to the directory containing " \
            "etc/profile.d/conda.sh." >&2
        exit 1
    fi

    # shellcheck source=/dev/null
    source "$CONDA_SETUP"
    conda activate "$CONDA_ENVIRONMENT"

    if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENVIRONMENT" ]]; then
        echo "Failed to activate Conda environment: $CONDA_ENVIRONMENT" >&2
        exit 1
    fi
}


configuration_path() {
    local latent_dimension="$1"

    echo "$CONFIGURATION_DIRECTORY/Latent${latent_dimension}.yml"
}


check_sweep_inputs() {
    local command_name
    local configuration_file
    local latent_dimension

    for command_name in iaflow-prepare-data iaflow-train-autoencoder; do
        if ! command -v "$command_name" >/dev/null 2>&1; then
            echo "Required command is unavailable in " \
                "$CONDA_ENVIRONMENT: $command_name" >&2
            exit 1
        fi
    done

    for latent_dimension in "${LATENT_DIMENSIONS[@]}"; do
        configuration_file="$(configuration_path "$latent_dimension")"
        if [[ ! -f "$configuration_file" ]]; then
            echo "Missing latent-dimension configuration: $configuration_file" >&2
            exit 1
        fi
    done
}


initialize_log_directory() {
    local sweep_timestamp

    sweep_timestamp="$(date '+%Y%m%d-%H%M%S')"
    LOG_DIRECTORY="$PROJECT_ROOT/Runs/NLA/AutoEncoder/Conv1D/SweepLogs/$sweep_timestamp"
    mkdir -p "$LOG_DIRECTORY"
}


run_logged_command() {
    local stage_name="$1"
    shift
    local log_path="$LOG_DIRECTORY/${stage_name}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $stage_name"
    "$@" 2>&1 | tee "$log_path"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed $stage_name"
}


run_latent_experiment() {
    local latent_dimension="$1"
    local configuration_file

    configuration_file="$(configuration_path "$latent_dimension")"

    # Rebuild the common cache immediately before each dimension is trained.
    run_logged_command \
        "Latent${latent_dimension}_PrepareData" \
        iaflow-prepare-data \
        --config "$configuration_file" \
        --overwrite

    run_logged_command \
        "Latent${latent_dimension}_Train" \
        iaflow-train-autoencoder \
        --config "$configuration_file" \
        --no-progress
}


main() {
    local latent_dimension

    activate_conda_environment
    export PYTHONUNBUFFERED=1
    cd "$PROJECT_ROOT"

    check_sweep_inputs
    initialize_log_directory

    echo "Project root: $PROJECT_ROOT"
    echo "Conda environment: $CONDA_DEFAULT_ENV"
    echo "Python executable: $(command -v python)"
    echo "Sweep logs: $LOG_DIRECTORY"

    for latent_dimension in "${LATENT_DIMENSIONS[@]}"; do
        run_latent_experiment "$latent_dimension"
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Latent-dimension sweep completed"
}


main "$@"
