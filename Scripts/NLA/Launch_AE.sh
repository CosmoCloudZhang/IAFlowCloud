#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)"
RUN_SCRIPT="$SCRIPT_DIRECTORY/Run_AE.sh"
FORCE_NEW_RUN=0


usage() {
    echo "Usage: bash Scripts/NLA/Launch_AE.sh [--fresh] <architecture> <depth>"
    echo ""
    echo "Architecture: Conv1D or Conv2D"
    echo "Depth: Depth03, Depth04, or Depth05"
    echo "The detached sweep processes latent dimensions 02, 04, 06, 08, and 10."
}


check_command() {
    local command_name="$1"

    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required launch command is unavailable: $command_name" >&2
        exit 1
    fi
}


main() {
    local architecture
    local depth
    local log_directory
    local pid_file
    local run_root
    local sweep_log
    local sweep_pid
    local sweep_timestamp

    case "${1:-}" in
        --help|-h)
            usage
            return
            ;;
        --fresh)
            FORCE_NEW_RUN=1
            shift
            ;;
    esac

    if [[ "$#" -ne 2 ]]; then
        usage >&2
        exit 2
    fi

    architecture="$1"
    depth="$2"

    case "$architecture" in
        Conv1D|Conv2D) ;;
        *)
            echo "Expected architecture: Conv1D or Conv2D" >&2
            exit 2
            ;;
    esac

    case "$depth" in
        Depth03|Depth04|Depth05) ;;
        *)
            echo "Expected depth: Depth03, Depth04, or Depth05" >&2
            exit 2
            ;;
    esac

    if [[ "$FORCE_NEW_RUN" != "0" && "$FORCE_NEW_RUN" != "1" ]]; then
        echo "IAFLOW_FORCE_NEW_RUN must be 0 or 1." >&2
        exit 2
    fi

    check_command bash
    check_command caffeinate
    check_command nohup

    sweep_timestamp="$(date '+%Y%m%d-%H%M%S')"
    run_root="$PROJECT_ROOT/Runs/NLA/AE/$architecture/$depth"
    log_directory="$run_root/SweepLogs/$sweep_timestamp"
    sweep_log="$log_directory/Sweep.log"
    pid_file="$log_directory/Sweep.pid"

    if [[ -e "$log_directory" ]]; then
        echo "Sweep log directory already exists: $log_directory" >&2
        exit 1
    fi
    mkdir -p "$log_directory"

    nohup env \
        IAFLOW_FORCE_NEW_RUN="$FORCE_NEW_RUN" \
        IAFLOW_SWEEP_TIMESTAMP="$sweep_timestamp" \
        caffeinate -i \
        bash "$RUN_SCRIPT" "$architecture" "$depth" \
        >"$sweep_log" 2>&1 < /dev/null &

    sweep_pid=$!
    printf '%s\n' "$sweep_pid" >"$pid_file"

    echo "$architecture $depth sweep launched safely in the background."
    echo "PID: $sweep_pid"
    echo "Log: $sweep_log"
    echo "PID file: $pid_file"
    echo "Follow: tail -f \"$sweep_log\""
    echo "Wait for 'AE sweep completed' before launching another MPS sweep."
}


main "$@"
