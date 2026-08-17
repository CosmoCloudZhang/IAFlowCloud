#!/usr/bin/env bash

set -euo pipefail

# The sweep is noninteractive. Give every Python child a valid standard input
# even when the parent terminal or detached launcher closes its descriptor.
exec 0</dev/null

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)"
ARCHITECTURE="${1:-Conv1D}"
DEPTH="${2:-Depth03}"
CONFIGURATION_FILE="$PROJECT_ROOT/Config/NLA/AE/$ARCHITECTURE/$DEPTH.yaml"
RUN_ROOT="$PROJECT_ROOT/Runs/NLA/AE/$ARCHITECTURE/$DEPTH"
PCA_METRICS="$PROJECT_ROOT/Data/NLA/PCA/PCAValidationMetrics.json"
CONDA_ROOT="${IAFLOW_CONDA_ROOT:-/opt/homebrew/anaconda3}"
CONDA_ENVIRONMENT="${IAFLOW_CONDA_ENVIRONMENT:-MLConda}"
CONDA_SETUP="$CONDA_ROOT/etc/profile.d/conda.sh"
FORCE_NEW_RUN="${IAFLOW_FORCE_NEW_RUN:-0}"
LATENT_DIMENSIONS=(02 04 06 08 10)


activate_conda_environment() {
    if [[ ! -f "$CONDA_SETUP" ]]; then
        echo "Conda shell setup was not found: $CONDA_SETUP" >&2
        echo "Set IAFLOW_CONDA_ROOT to the directory containing " \
            "etc/profile.d/conda.sh." >&2
        exit 1
    fi

    # Conda activation and deactivation hooks may probe optional backup
    # variables. Suspend nounset only while those environment hooks run.
    set +u
    # shellcheck source=/dev/null
    source "$CONDA_SETUP"
    conda activate "$CONDA_ENVIRONMENT"
    set -u

    if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENVIRONMENT" ]]; then
        echo "Failed to activate Conda environment: $CONDA_ENVIRONMENT" >&2
        exit 1
    fi
}


check_sweep_inputs() {
    local command_name

    if [[ ! -f "$CONFIGURATION_FILE" ]]; then
        echo "Missing architecture-depth template: $CONFIGURATION_FILE" >&2
        exit 1
    fi

    if [[ ! -f "$PCA_METRICS" ]]; then
        echo "Missing PCA validation benchmark: $PCA_METRICS" >&2
        echo "Run Notebooks/NLA/PCA.ipynb before starting the sweep." >&2
        exit 1
    fi

    for command_name in \
        iaflow-validate-configs \
        iaflow-prepare-data \
        iaflow-train-autoencoder \
        iaflow-evaluate-autoencoder \
        iaflow-diagnose-autoencoder \
        iaflow-export-latents; do
        if ! command -v "$command_name" >/dev/null 2>&1; then
            echo "Required command is unavailable in " \
                "$CONDA_ENVIRONMENT: $command_name" >&2
            exit 1
        fi
    done
}


initialize_sweep() {
    SWEEP_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
    LOG_DIRECTORY="$RUN_ROOT/SweepLogs/$SWEEP_TIMESTAMP"
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


latest_candidate() {
    local latent_root="$1"
    local pointer="$latent_root/LatestRun.txt"
    local candidate=""

    if [[ -f "$pointer" ]]; then
        candidate="$(<"$pointer")"
        if [[ "$candidate" != /* ]]; then
            candidate="$PROJECT_ROOT/$candidate"
        fi
        if [[ -d "$candidate" ]]; then
            echo "$candidate"
            return
        fi
        candidate="$latent_root/$(basename -- "$candidate")"
        if [[ -d "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    fi

    find "$latent_root" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -print 2>/dev/null | sort | tail -n 1
}


select_run_directory() {
    local latent_dimension="$1"
    local latent_root="$RUN_ROOT/Latent${latent_dimension}"
    local candidate=""

    mkdir -p "$latent_root"
    if [[ "$FORCE_NEW_RUN" != "1" ]]; then
        candidate="$(latest_candidate "$latent_root")"
    fi
    if [[ -n "$candidate" \
        && ( -f "$candidate/Summary.json" || -f "$candidate/Last.pt" ) ]]; then
        echo "$candidate"
    else
        echo "$latent_root/$SWEEP_TIMESTAMP"
    fi
}


train_if_needed() {
    local latent_dimension="$1"
    local run_directory="$2"
    local arguments=(
        --config "$CONFIGURATION_FILE"
        --latent-dim "$((10#$latent_dimension))"
        --run-directory "$run_directory"
        --no-progress
    )

    if [[ -f "$run_directory/Summary.json" && -f "$run_directory/Best.pt" ]]; then
        echo "Training already complete: $run_directory"
        return
    fi
    if [[ -f "$run_directory/Last.pt" ]]; then
        arguments+=(--resume "$run_directory/Last.pt")
    fi
    run_logged_command \
        "Latent${latent_dimension}_Train" \
        iaflow-train-autoencoder \
        "${arguments[@]}"
}


evaluate_if_needed() {
    local latent_dimension="$1"
    local run_directory="$2"

    if [[ -f "$run_directory/ValidationMetrics.json" ]] \
        && grep -q 'pca_comparison' "$run_directory/ValidationMetrics.json"; then
        echo "PCA-matched validation already complete: $run_directory"
        return
    fi
    run_logged_command \
        "Latent${latent_dimension}_Evaluate" \
        iaflow-evaluate-autoencoder \
        --run-directory "$run_directory" \
        --split validation \
        --pca-metrics "$PCA_METRICS" \
        --no-progress
}


diagnose_if_needed() {
    local latent_dimension="$1"
    local run_directory="$2"

    if [[ -f "$run_directory/ValidationDiagnostics.json" \
        && -f "$run_directory/ValidationDiagnostics.npz" ]]; then
        echo "Validation diagnostics already complete: $run_directory"
        return
    fi
    run_logged_command \
        "Latent${latent_dimension}_Diagnose" \
        iaflow-diagnose-autoencoder \
        --run-directory "$run_directory" \
        --no-progress
}


export_if_needed() {
    local latent_dimension="$1"
    local run_directory="$2"

    if [[ -f "$run_directory/Latents.hdf5" ]]; then
        echo "Train and validation latents already exported: $run_directory"
        return
    fi
    run_logged_command \
        "Latent${latent_dimension}_ExportLatents" \
        iaflow-export-latents \
        --run-directory "$run_directory"
}


run_latent_experiment() {
    local latent_dimension="$1"
    local run_directory

    run_directory="$(select_run_directory "$latent_dimension")"
    echo "Latent ${latent_dimension} run directory: $run_directory"
    train_if_needed "$latent_dimension" "$run_directory"
    evaluate_if_needed "$latent_dimension" "$run_directory"
    diagnose_if_needed "$latent_dimension" "$run_directory"
    export_if_needed "$latent_dimension" "$run_directory"
}


main() {
    local latent_dimension

    activate_conda_environment
    export PYTHONUNBUFFERED=1
    cd "$PROJECT_ROOT"

    check_sweep_inputs
    initialize_sweep

    echo "Project root: $PROJECT_ROOT"
    echo "Conda environment: $CONDA_DEFAULT_ENV"
    echo "Python executable: $(command -v python)"
    echo "Experiment template: $CONFIGURATION_FILE"
    echo "Sweep logs: $LOG_DIRECTORY"

    run_logged_command \
        "ValidateConfigurations" \
        iaflow-validate-configs \
        --config "$CONFIGURATION_FILE" \
        --latent-dims 2 4 6 8 10

    run_logged_command \
        "PrepareData" \
        iaflow-prepare-data \
        --config "$CONFIGURATION_FILE"

    for latent_dimension in "${LATENT_DIMENSIONS[@]}"; do
        run_latent_experiment "$latent_dimension"
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] AE sweep completed"
}


main "$@"
