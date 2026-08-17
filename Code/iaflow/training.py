"""
Device-portable training and evaluation procedures.
"""

from __future__ import annotations

import copy
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .architectures import AutoEncoder, build_autoencoder
from .artifacts import (
    CHECKPOINT_FORMAT_VERSION,
    build_checkpoint,
    check_checkpoint_data_compatibility,
    portable_path,
    save_checkpoint,
    save_json,
)
from .config import ExperimentConfig, config_to_dict
from .data import (
    CachedSurfaceDataset,
    NormalizationStats,
    build_dataloader,
    check_surface_cache,
    prepare_surface_cache,
)
from .evaluation import evaluate_autoencoder, evaluate_reconstruction_objective
from .losses import build_reconstruction_loss
from .runs import write_latest_run

__all__ = [
    "fit_autoencoder",
    "resolve_device",
    "set_reproducibility",
]


_Scheduler = (
    torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
)


@dataclass(slots=True)
class _TrainingProgress:
    """
    Store the mutable optimization and early-stopping state of one run.
    """
    
    start_epoch: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)
    best_mse: float = math.inf
    best_epoch: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)
    epochs_without_improvement: int = 0
    stopped_early: bool = False


def set_reproducibility(
    seed: int,
    deterministic: bool = False,
) -> None:
    """
    Seed Python, NumPy, and PyTorch without silently forcing slow kernels.
    
    Arguments:
        seed (int):
            Shared random seed.
        deterministic (bool):
            Whether PyTorch should prefer deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _capture_rng_state(
    train_generator: torch.Generator,
) -> dict[str, Any]:
    """
    Capture every random-number state needed for trajectory-preserving resume.
    
    Arguments:
        train_generator (torch.Generator):
            Generator controlling training-loader permutations.
    
    Returns:
        state (dict[str, Any]):
            Python, NumPy, PyTorch, loader, and optional CUDA states.
    """
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(np.array(numpy_state[1], copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "train_loader": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(
    state: dict[str, Any],
    train_generator: torch.Generator,
) -> None:
    """
    Restore every random-number state captured in a checkpoint.
    
    Arguments:
        state (dict[str, Any]):
            Stored reproducibility-state mapping.
        train_generator (torch.Generator):
            Generator controlling training-loader permutations.
    """
    if not isinstance(state, dict):
        raise ValueError("Resume checkpoint is missing reproducibility state.")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    train_generator.set_state(state["train_loader"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _strict_resume_config_matches(
    stored: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    """
    Allow only the documented total-epoch and device changes on resume.
    
    Arguments:
        stored (dict[str, Any]):
            Original resolved experiment configuration.
        current (dict[str, Any]):
            Configuration requested for the resumed process.
    
    Returns:
        matches (bool):
            Whether all immutable settings match exactly.
    """
    stored_copy = copy.deepcopy(stored)
    current_copy = copy.deepcopy(current)
    for values in (stored_copy, current_copy):
        values["training"].pop("epochs", None)
        values["training"].pop("device", None)
    return stored_copy == current_copy


def resolve_device(
    requested: str = "auto",
) -> torch.device:
    """
    Resolve an explicit or automatic PyTorch compute device.
    
    Arguments:
        requested (str):
            One of auto, cpu, mps, or cuda.
    
    Returns:
        device (torch.device):
            Available device selected for the run.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    
    if requested == "mps" and not (
        torch.backends.mps.is_built() and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Apple MPS was requested but is unavailable.")
    return device


def _optimizer(
    model: nn.Module,
    config: Any,
) -> torch.optim.Optimizer:
    """
    Construct the configured optimizer for all model parameters.
    
    Arguments:
        model (torch.nn.Module):
            Model whose parameters will be optimized.
        config (object):
            Checked training-configuration section.
    
    Returns:
        optimizer (torch.optim.Optimizer):
            Configured Adam or AdamW optimizer.
    """
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[config.optimizer]
    return optimizer_class(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: Any,
) -> _Scheduler | None:
    """
    Construct the configured learning-rate scheduler.
    
    Arguments:
        optimizer (torch.optim.Optimizer):
            Optimizer whose learning rate will be scheduled.
        config (object):
            Checked training-configuration section.
    
    Returns:
        scheduler (object or None):
            Configured scheduler, or None when scheduling is disabled.
    """
    if config.scheduler == "none":
        return None
    
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.minimum_learning_rate,
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )


def _limited_dataset(
    dataset: Dataset,
    maximum: int | None,
    seed: int,
) -> Dataset:
    """
    Select a reproducible subset for smoke training or validation.
    
    Arguments:
        dataset (torch.utils.data.Dataset):
            Complete stored split.
        maximum (int or None):
            Maximum number of rows to retain.
        seed (int):
            Seed controlling subset selection.
    
    Returns:
        selected_dataset (torch.utils.data.Dataset):
            Original dataset or reproducible subset.
    """
    if maximum is None or maximum >= len(dataset):
        return dataset
    
    if maximum <= 0:
        raise ValueError("Sample limits must be positive.")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=maximum, replace=False)).tolist()
    return Subset(dataset, indices)


def _run_directory(
    config: ExperimentConfig,
    explicit: str | Path | None,
    *,
    allow_existing: bool = False,
) -> Path:
    """
    Resolve and safely initialize the output directory for one run.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        explicit (str or pathlib.Path or None):
            Optional explicit run directory.
        allow_existing (bool):
            Whether an existing non-empty directory may be reused for resume.
    
    Returns:
        path (pathlib.Path):
            Absolute initialized run directory.
    """
    configured = config.resolve_path(config.output.run_directory)
    path = config.resolve_path(explicit) if explicit is not None else configured
    if path != configured:
        raise ValueError("run_directory must match the resolved experiment configuration.")
    
    if not allow_existing and path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _train_epoch(
    model: AutoEncoder,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None,
    scaler: torch.amp.GradScaler | None,
    show_progress: bool,
) -> float:
    """
    Optimize the model for one complete training epoch.
    
    Arguments:
        model (AutoEncoder):
            Autoencoder to optimize.
        loader (torch.utils.data.DataLoader):
            Reproducibly shuffled training loader.
        loss_function (torch.nn.Module):
            Configured reconstruction objective.
        optimizer (torch.optim.Optimizer):
            Optimizer used for parameter updates.
        device (torch.device):
            Device used for training.
        gradient_clip_norm (float or None):
            Optional maximum gradient norm.
        scaler (torch.amp.GradScaler or None):
            Optional CUDA mixed-precision gradient scaler.
        show_progress (bool):
            Whether to display the batch progress bar.
    
    Returns:
        mean_loss (float):
            Element-weighted mean training loss.
    """
    model.train()
    total_loss = 0.0
    total_values = 0
    progress = tqdm(loader, desc="train", leave=False, disable=not show_progress)
    for target in progress:
        target = target.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(target)
                loss = loss_function(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            prediction = model(target)
            loss = loss_function(prediction, target)
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        
        number_of_values = target.numel()
        total_loss += float(loss.detach().item()) * number_of_values
        total_values += number_of_values
        progress.set_postfix(loss=f"{total_loss / total_values:.3e}")
    return total_loss / total_values


def _resolve_resume_checkpoint(
    config: ExperimentConfig,
    resume_checkpoint: str | Path | None,
    run_directory: str | Path | None,
) -> tuple[Path | None, str | Path | None]:
    """
    Resolve an optional resume checkpoint and its required run directory.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        resume_checkpoint (str or pathlib.Path or None):
            Optional checkpoint requested for trajectory-aware resume.
        run_directory (str or pathlib.Path or None):
            Optional explicit run output directory.
    
    Returns:
        result (tuple[pathlib.Path or None, str or pathlib.Path or None]):
            Resolved checkpoint and compatible output-directory request.
    """
    if resume_checkpoint is None:
        return None, run_directory
    
    resolved_resume = config.resolve_path(resume_checkpoint)
    if not resolved_resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resolved_resume}")
    inferred_run_directory = resolved_resume.parent
    if run_directory is not None:
        requested_run_directory = config.resolve_path(run_directory)
        if requested_run_directory != inferred_run_directory:
            raise ValueError("A resumed run must write back to the checkpoint directory.")
    return resolved_resume, inferred_run_directory


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Move every tensor in restored optimizer state to the training device.
    
    Arguments:
        optimizer (torch.optim.Optimizer):
            Optimizer whose state was loaded on CPU.
        device (torch.device):
            Device used for resumed parameter updates.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _record_resume_event(
    config: ExperimentConfig,
    output_directory: Path,
    resume_checkpoint: Path,
    start_epoch: int,
    device: torch.device,
) -> None:
    """
    Append one portable resume event to the run history.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        output_directory (pathlib.Path):
            Existing resumed run directory.
        resume_checkpoint (pathlib.Path):
            Checkpoint used to continue the run.
        start_epoch (int):
            First epoch that will execute after restoration.
        device (torch.device):
            Device used for resumed training.
    """
    resume_history_path = output_directory / "ResumeHistory.json"
    resume_events: list[dict[str, Any]] = []
    if resume_history_path.is_file():
        with resume_history_path.open("r", encoding="utf-8") as stream:
            stored_events = json.load(stream)
        if not isinstance(stored_events, list):
            raise ValueError("ResumeHistory.json must contain a list of events.")
        resume_events = stored_events
    resume_events.append(
        {
            "checkpoint": portable_path(resume_checkpoint, config.project_root),
            "from_epoch": start_epoch - 1,
            "new_total_epochs": config.training.epochs,
            "device": str(device),
            "timestamp": datetime.now().astimezone().isoformat(),
        }
    )
    save_json(resume_events, resume_history_path)


def _check_resume_identity(
    config: ExperimentConfig,
    resumed: dict[str, Any],
    resolved_config: dict[str, Any],
    normalization: NormalizationStats,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Check checkpoint format, configuration, model, and prepared-data identity.
    
    Arguments:
        config (ExperimentConfig):
            Current checked experiment configuration.
        resumed (dict[str, Any]):
            Loaded resume checkpoint payload.
        resolved_config (dict[str, Any]):
            Current JSON-safe experiment configuration.
        normalization (NormalizationStats):
            Training-only normalization loaded from the current cache.
        metadata (dict[str, Any]):
            Checked current cache manifest.
    
    Returns:
        stored_config (dict[str, Any]):
            Original resolved experiment configuration stored in the checkpoint.
    """
    if resumed.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Resume checkpoint format is unsupported or stale.")
    stored_config = resumed.get("experiment_config")
    if not isinstance(stored_config, dict) or not _strict_resume_config_matches(
        stored_config,
        resolved_config,
    ):
        raise ValueError(
            "Resume configuration differs from the original run outside epochs/device."
        )
    if resumed.get("model_config") != dict(config.model):
        raise ValueError("Resume checkpoint model configuration does not match.")
    
    if tuple(resumed.get("input_shape", ())) != config.data.input_shape:
        raise ValueError("Resume checkpoint input shape does not match.")
    check_checkpoint_data_compatibility(resumed, normalization, metadata)
    return stored_config


def _restore_optimization_state(
    resumed: dict[str, Any],
    model: AutoEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: _Scheduler | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> None:
    """
    Restore model, optimizer, scheduler, and optional scaler state.
    
    Arguments:
        resumed (dict[str, Any]):
            Loaded resume checkpoint payload.
        model (AutoEncoder):
            Newly constructed model receiving stored parameters.
        optimizer (torch.optim.Optimizer):
            Newly constructed optimizer receiving stored state.
        scheduler (object or None):
            Optional scheduler receiving stored state.
        scaler (torch.amp.GradScaler or None):
            Optional CUDA gradient scaler receiving stored state.
        device (torch.device):
            Device used for resumed training.
    """
    model.load_state_dict(resumed["model_state_dict"], strict=True)
    optimizer.load_state_dict(resumed["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    if scheduler is not None:
        if "scheduler_state_dict" not in resumed:
            raise ValueError("Resume checkpoint is missing scheduler state.")
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
    
    if scaler is not None and "scaler_state_dict" in resumed:
        scaler.load_state_dict(resumed["scaler_state_dict"])


def _load_training_progress(
    config: ExperimentConfig,
    resumed: dict[str, Any],
    output_directory: Path,
) -> _TrainingProgress:
    """
    Restore epoch history and best-model state from a resumed run.
    
    Arguments:
        config (ExperimentConfig):
            Current checked experiment configuration.
        resumed (dict[str, Any]):
            Loaded resume checkpoint payload.
        output_directory (pathlib.Path):
            Existing resumed run directory.
    
    Returns:
        progress (_TrainingProgress):
            Restored history, best result, and early-stopping state.
    """
    progress = _TrainingProgress(start_epoch=int(resumed["epoch"]) + 1)
    if progress.start_epoch > config.training.epochs:
        raise ValueError(
            "The resume checkpoint has already reached the configured total epochs."
        )
    
    history_path = output_directory / "History.json"
    if not history_path.is_file():
        raise FileNotFoundError("A resumed run requires its existing History.json.")
    with history_path.open("r", encoding="utf-8") as stream:
        stored_history = json.load(stream)
    if not isinstance(stored_history, list):
        raise ValueError("History.json must contain a list of epoch records.")
    progress.history = stored_history
    if not progress.history or int(progress.history[-1]["epoch"]) != progress.start_epoch - 1:
        raise ValueError("History.json does not end at the resume checkpoint epoch.")
    training_state = resumed.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("Resume checkpoint is missing early-stopping state.")
    progress.best_epoch = int(training_state["best_epoch"])
    progress.best_mse = float(training_state["best_mse"])
    progress.best_metrics = dict(training_state["best_metrics"])
    progress.epochs_without_improvement = int(
        training_state["epochs_without_improvement"]
    )
    if progress.epochs_without_improvement >= config.training.early_stopping_patience:
        raise ValueError("The original run had already reached its early-stopping condition.")
    return progress


def _check_resolved_config_artifact(
    output_directory: Path,
    stored_config: dict[str, Any],
) -> None:
    """
    Check the persisted resolved configuration against the resume checkpoint.
    
    Arguments:
        output_directory (pathlib.Path):
            Existing resumed run directory.
        stored_config (dict[str, Any]):
            Original configuration stored in the resume checkpoint.
    """
    resolved_config_path = output_directory / "ResolvedConfig.json"
    if not resolved_config_path.is_file():
        raise FileNotFoundError("A resumed run requires its original ResolvedConfig.json.")
    with resolved_config_path.open("r", encoding="utf-8") as stream:
        if json.load(stream) != stored_config:
            raise ValueError("ResolvedConfig.json differs from the resume checkpoint.")


def _restore_training_run(
    config: ExperimentConfig,
    *,
    resume_checkpoint: Path,
    output_directory: Path,
    resolved_config: dict[str, Any],
    model: AutoEncoder,
    normalization: NormalizationStats,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: _Scheduler | None,
    scaler: torch.amp.GradScaler | None,
    train_generator: torch.Generator,
    device: torch.device,
) -> tuple[dict[str, Any], _TrainingProgress]:
    """
    Check and restore every state required for trajectory-aware resume.
    
    Arguments:
        config (ExperimentConfig):
            Current checked experiment configuration.
        resume_checkpoint (pathlib.Path):
            Existing checkpoint from the resumed run.
        output_directory (pathlib.Path):
            Existing resumed run directory.
        resolved_config (dict[str, Any]):
            Current JSON-safe experiment configuration.
        model (AutoEncoder):
            Newly constructed model receiving stored parameters.
        normalization (NormalizationStats):
            Training-only normalization loaded from the current cache.
        metadata (dict[str, Any]):
            Checked current cache manifest.
        optimizer (torch.optim.Optimizer):
            Newly constructed optimizer receiving stored state.
        scheduler (object or None):
            Optional scheduler receiving stored state.
        scaler (torch.amp.GradScaler or None):
            Optional CUDA gradient scaler receiving stored state.
        train_generator (torch.Generator):
            Training-loader generator receiving stored state.
        device (torch.device):
            Device used for resumed training.
    
    Returns:
        result (tuple[dict[str, Any], _TrainingProgress]):
            Original resolved configuration and restored training progress.
    """
    resumed = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(resumed, dict):
        raise ValueError("Resume checkpoint payload must be a mapping.")
    stored_config = _check_resume_identity(
        config,
        resumed,
        resolved_config,
        normalization,
        metadata,
    )
    _restore_optimization_state(
        resumed,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
    )
    progress = _load_training_progress(config, resumed, output_directory)
    _restore_rng_state(resumed.get("rng_state"), train_generator)
    _check_resolved_config_artifact(output_directory, stored_config)
    _record_resume_event(
        config,
        output_directory,
        resume_checkpoint,
        progress.start_epoch,
        device,
    )
    return stored_config, progress


def _write_initial_run_artifacts(
    output_directory: Path,
    resolved_config: dict[str, Any],
    model: AutoEncoder,
    metadata: dict[str, Any],
    device: torch.device,
) -> None:
    """
    Write configuration, architecture, data, and environment provenance.
    
    Arguments:
        output_directory (pathlib.Path):
            Newly initialized run directory.
        resolved_config (dict[str, Any]):
            JSON-safe experiment configuration and runtime limits.
        model (AutoEncoder):
            Newly initialized autoencoder.
        metadata (dict[str, Any]):
            Checked cache manifest.
        device (torch.device):
            Device used for training.
    """
    save_json(resolved_config, output_directory / "ResolvedConfig.json")
    save_json(model.architecture_summary(), output_directory / "Architecture.json")
    save_json(metadata, output_directory / "DataMetadata.json")
    save_json(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                torch.backends.mps.is_built() and torch.backends.mps.is_available()
            ),
        },
        output_directory / "Environment.json",
    )


def _step_scheduler(
    scheduler: _Scheduler | None,
    scheduler_name: str,
    validation_mse: float,
) -> None:
    """
    Advance the configured scheduler after one validation measurement.
    
    Arguments:
        scheduler (object or None):
            Configured scheduler, or None when scheduling is disabled.
        scheduler_name (str):
            Configured scheduler mode.
        validation_mse (float):
            Current normalized validation mean-squared error.
    """
    if scheduler is None:
        return
    
    if scheduler_name == "plateau":
        scheduler.step(validation_mse)
    else:
        scheduler.step()


def _save_epoch_checkpoints(
    checkpoint: dict[str, Any],
    output_directory: Path,
    epoch: int,
    save_every_epochs: int,
    improved: bool,
) -> None:
    """
    Save latest, periodic, and newly best checkpoint variants.
    
    Arguments:
        checkpoint (dict[str, Any]):
            Complete checkpoint payload for the current epoch.
        output_directory (pathlib.Path):
            Run directory receiving checkpoint files.
        epoch (int):
            Completed one-based training epoch.
        save_every_epochs (int):
            Interval between archival epoch checkpoints.
        improved (bool):
            Whether this checkpoint establishes a new best validation MSE.
    """
    save_checkpoint(checkpoint, output_directory / "Last.pt")
    if epoch % save_every_epochs == 0:
        save_checkpoint(checkpoint, output_directory / f"Epoch{epoch:04d}.pt")
    
    if improved:
        save_checkpoint(checkpoint, output_directory / "Best.pt")


def _run_training_epochs(
    config: ExperimentConfig,
    *,
    model: AutoEncoder,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _Scheduler | None,
    scaler: torch.amp.GradScaler | None,
    normalization: NormalizationStats,
    device: torch.device,
    progress: _TrainingProgress,
    resolved_config: dict[str, Any],
    metadata: dict[str, Any],
    output_directory: Path,
    train_generator: torch.Generator,
    writer: SummaryWriter,
    show_progress: bool,
) -> _TrainingProgress:
    """
    Execute, record, and checkpoint all remaining training epochs.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        model (AutoEncoder):
            Autoencoder to optimize and evaluate.
        train_loader (torch.utils.data.DataLoader):
            Reproducibly shuffled training loader.
        validation_loader (torch.utils.data.DataLoader):
            Ordered validation loader.
        loss_function (torch.nn.Module):
            Configured reconstruction objective.
        optimizer (torch.optim.Optimizer):
            Optimizer used for parameter updates.
        scheduler (object or None):
            Optional learning-rate scheduler.
        scaler (torch.amp.GradScaler or None):
            Optional CUDA mixed-precision gradient scaler.
        normalization (NormalizationStats):
            Training-only normalization paired with the data.
        device (torch.device):
            Device used for training and validation.
        progress (_TrainingProgress):
            Current history, best result, and early-stopping state.
        resolved_config (dict[str, Any]):
            Original resolved configuration stored in checkpoints.
        metadata (dict[str, Any]):
            Checked cache provenance stored in checkpoints.
        output_directory (pathlib.Path):
            Run directory receiving history and checkpoints.
        train_generator (torch.Generator):
            Training-loader generator captured in checkpoints.
        writer (SummaryWriter):
            TensorBoard writer for scalar diagnostics.
        show_progress (bool):
            Whether to display batch progress bars.
    
    Returns:
        progress (_TrainingProgress):
            Updated training and early-stopping state.
    """
    for epoch in range(progress.start_epoch, config.training.epochs + 1):
        epoch_start = time.perf_counter()
        training_start = time.perf_counter()
        train_loss = _train_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
            config.training.gradient_clip_norm,
            scaler,
            show_progress,
        )
        training_seconds = time.perf_counter() - training_start
        validation_start = time.perf_counter()
        validation_metrics = evaluate_reconstruction_objective(
            model,
            validation_loader,
            normalization,
            device,
            show_progress=show_progress,
        )
        validation_seconds = time.perf_counter() - validation_start
        validation_mse = validation_metrics["normalized_mse"]
        _step_scheduler(scheduler, config.training.scheduler, validation_mse)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improvement = progress.best_mse - validation_mse
        improved = improvement > config.training.minimum_improvement
        if improved:
            progress.best_mse = validation_mse
            progress.best_epoch = epoch
            progress.best_metrics = validation_metrics
            progress.epochs_without_improvement = 0
        else:
            progress.epochs_without_improvement += 1
        
        checkpoint_start = time.perf_counter()
        checkpoint = build_checkpoint(
            model,
            normalization,
            epoch=epoch,
            validation_metrics=validation_metrics,
            experiment_config=resolved_config,
            data_provenance=metadata,
            training_state={
                "best_epoch": progress.best_epoch,
                "best_mse": progress.best_mse,
                "best_metrics": progress.best_metrics,
                "epochs_without_improvement": progress.epochs_without_improvement,
            },
            rng_state=_capture_rng_state(train_generator),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        _save_epoch_checkpoints(
            checkpoint,
            output_directory,
            epoch,
            config.output.save_every_epochs,
            improved,
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_start
        
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation_mse", validation_mse, epoch)
        writer.add_scalar(
            "metrics/validation_variance_recovered",
            validation_metrics["variance_recovered"],
            epoch,
        )
        writer.add_scalar("optimization/learning_rate", learning_rate, epoch)
        writer.add_scalar("timing/training_seconds", training_seconds, epoch)
        writer.add_scalar("timing/validation_seconds", validation_seconds, epoch)
        writer.add_scalar("timing/checkpoint_seconds", checkpoint_seconds, epoch)
        
        duration_seconds = time.perf_counter() - epoch_start
        writer.add_scalar("timing/duration_seconds", duration_seconds, epoch)
        progress.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "duration_seconds": duration_seconds,
                "validation": validation_metrics,
            }
        )
        save_json(progress.history, output_directory / "History.json")
        
        print(
            f"Epoch {epoch:04d} | train={train_loss:.4e} | "
            f"val={validation_mse:.4e} | "
            f"variance={validation_metrics['variance_recovered']:.6f} | "
            f"lr={learning_rate:.2e} | time={duration_seconds:.1f}s"
        )
        if progress.epochs_without_improvement >= config.training.early_stopping_patience:
            progress.stopped_early = True
            break
    return progress


def _evaluate_best_checkpoint(
    config: ExperimentConfig,
    output_directory: Path,
    model: AutoEncoder,
    validation_loader: DataLoader,
    normalization: NormalizationStats,
    device: torch.device,
    *,
    show_progress: bool,
) -> tuple[int, dict[str, float]]:
    """
    Evaluate the selected checkpoint with the complete diagnostic metrics.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        output_directory (pathlib.Path):
            Completed run directory containing Best.pt.
        model (AutoEncoder):
            Autoencoder receiving the selected model state.
        validation_loader (torch.utils.data.DataLoader):
            Ordered validation loader used for detailed evaluation.
        normalization (NormalizationStats):
            Training-only normalization paired with the model.
        device (torch.device):
            Device used for detailed evaluation.
        show_progress (bool):
            Whether to display the final evaluation progress bar.
    
    Returns:
        result (tuple[int, dict[str, float]]):
            Selected epoch and complete validation metrics.
    """
    checkpoint_path = output_directory / "Best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Best checkpoint payload must be a mapping.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    metrics = evaluate_autoencoder(
        model,
        validation_loader,
        normalization,
        device,
        show_progress=show_progress,
    )
    checkpoint["validation_metrics"] = metrics
    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("Best checkpoint is missing training state.")
    training_state["best_metrics"] = metrics
    save_checkpoint(checkpoint, checkpoint_path)
    selected_epoch = int(checkpoint["epoch"])
    save_json(
        {
            "checkpoint": portable_path(checkpoint_path, config.project_root),
            "checkpoint_epoch": selected_epoch,
            "split": "validation",
            "final_test": False,
            "device": str(device),
            "metrics": metrics,
        },
        output_directory / "ValidationMetrics.json",
    )
    return selected_epoch, metrics


def _finalize_training_run(
    config: ExperimentConfig,
    output_directory: Path,
    device: torch.device,
    model: AutoEncoder,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    validation_loader: DataLoader,
    normalization: NormalizationStats,
    progress: _TrainingProgress,
    resume_checkpoint: Path | None,
    show_progress: bool,
) -> dict[str, Any]:
    """
    Write and return the final portable training summary.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        output_directory (pathlib.Path):
            Completed run directory.
        device (torch.device):
            Device used for training.
        model (AutoEncoder):
            Trained autoencoder.
        train_dataset (torch.utils.data.Dataset):
            Complete or smoke-limited training dataset.
        validation_dataset (torch.utils.data.Dataset):
            Complete or smoke-limited validation dataset.
        validation_loader (torch.utils.data.DataLoader):
            Ordered validation loader used for detailed final metrics.
        normalization (NormalizationStats):
            Training-only normalization paired with the model.
        progress (_TrainingProgress):
            Completed history, best result, and stopping state.
        resume_checkpoint (pathlib.Path or None):
            Checkpoint used to resume this process, if any.
        show_progress (bool):
            Whether to display the final evaluation progress bar.
    
    Returns:
        summary (dict[str, Any]):
            Best validation result, sample counts, device, and stopping state.
    """
    best_epoch, best_metrics = _evaluate_best_checkpoint(
        config,
        output_directory,
        model,
        validation_loader,
        normalization,
        device,
        show_progress=show_progress,
    )
    if best_epoch != progress.best_epoch:
        raise RuntimeError("Best checkpoint epoch disagrees with training progress.")
    progress.best_metrics = best_metrics
    summary: dict[str, Any] = {
        "run_directory": portable_path(output_directory, config.project_root),
        "device": str(device),
        "number_of_parameters": model.number_of_parameters,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_epoch": progress.best_epoch,
        "best_validation_metrics": progress.best_metrics,
        "target_variance_recovered": config.training.target_variance_recovered,
        "target_met": bool(
            progress.best_metrics.get("variance_recovered", -math.inf)
            >= config.training.target_variance_recovered
        ),
        "epochs_completed": len(progress.history),
        "resumed_from": (
            portable_path(resume_checkpoint, config.project_root)
            if resume_checkpoint is not None
            else None
        ),
        "stopped_early": progress.stopped_early,
        "test_split_used_during_training": False,
    }
    save_json(summary, output_directory / "Summary.json")
    write_latest_run(config, output_directory)
    return summary


def fit_autoencoder(
    config: ExperimentConfig,
    *,
    run_directory: str | Path | None = None,
    prepare_data: bool = False,
    overwrite_cache: bool = False,
    maximum_train_samples: int | None = None,
    maximum_validation_samples: int | None = None,
    resume_checkpoint: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Train on the stored training split and select only on validation MSE.
    
    The procedure writes portable provenance, supports strict trajectory-aware
    resume, and never reads the test split.
    
    Arguments:
        config (ExperimentConfig):
            Fully checked experiment configuration.
        run_directory (str or pathlib.Path or None):
            Optional explicit run output directory.
        prepare_data (bool):
            Whether to prepare or check the surface cache before training.
        overwrite_cache (bool):
            Whether cache preparation may replace existing artifacts.
        maximum_train_samples (int or None):
            Optional reproducible training-split limit for smoke runs.
        maximum_validation_samples (int or None):
            Optional reproducible validation-split limit for smoke runs.
        resume_checkpoint (str or pathlib.Path or None):
            Optional Last.pt checkpoint from the same run directory.
        show_progress (bool):
            Whether to display training and validation progress bars.
    
    Returns:
        summary (dict[str, Any]):
            Best validation result, sample counts, device, and stopping state.
    """
    set_reproducibility(config.training.seed, config.training.deterministic)
    if prepare_data:
        prepare_surface_cache(config, overwrite=overwrite_cache)
    metadata = check_surface_cache(config)
    cache_directory = config.resolve_path(config.data.cache_directory)
    normalization = NormalizationStats.load(
        cache_directory / metadata["normalization_file"]
    )
    device = resolve_device(config.training.device)
    
    train_dataset = _limited_dataset(
        CachedSurfaceDataset(config, "train"),
        maximum_train_samples,
        config.training.seed,
    )
    validation_dataset = _limited_dataset(
        CachedSurfaceDataset(config, "validation"),
        maximum_validation_samples,
        config.training.seed + 1,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(config.training.seed)
    train_loader = build_dataloader(
        train_dataset,
        config.data,
        training=True,
        seed=config.training.seed,
        device=device,
        generator=train_generator,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        config.data,
        training=False,
        seed=config.training.seed,
        device=device,
    )
    
    config.runtime.update(
        {
            "maximum_train_samples": maximum_train_samples,
            "maximum_validation_samples": maximum_validation_samples,
        }
    )
    resolved_config = config_to_dict(config)
    resolved_resume, run_directory = _resolve_resume_checkpoint(
        config,
        resume_checkpoint,
        run_directory,
    )
    output_directory = _run_directory(
        config,
        run_directory,
        allow_existing=resolved_resume is not None,
    )
    
    model = build_autoencoder(config.model, config.data.input_shape).to(device)
    loss_function = build_reconstruction_loss(config.training.loss)
    optimizer = _optimizer(model, config.training)
    scheduler = _scheduler(optimizer, config.training)
    use_scaler = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    progress = _TrainingProgress()
    if resolved_resume is not None:
        resolved_config, progress = _restore_training_run(
            config,
            resume_checkpoint=resolved_resume,
            output_directory=output_directory,
            resolved_config=resolved_config,
            model=model,
            normalization=normalization,
            metadata=metadata,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_generator=train_generator,
            device=device,
        )
    
    writer = SummaryWriter(
        log_dir=output_directory / "TensorBoard",
        purge_step=progress.start_epoch if resolved_resume is not None else None,
    )
    if resolved_resume is None:
        _write_initial_run_artifacts(
            output_directory,
            resolved_config,
            model,
            metadata,
            device,
        )
    
    try:
        progress = _run_training_epochs(
            config,
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            normalization=normalization,
            device=device,
            progress=progress,
            resolved_config=resolved_config,
            metadata=metadata,
            output_directory=output_directory,
            train_generator=train_generator,
            writer=writer,
            show_progress=show_progress,
        )
    finally:
        writer.close()
    return _finalize_training_run(
        config,
        output_directory,
        device,
        model,
        train_dataset,
        validation_dataset,
        validation_loader,
        normalization,
        progress,
        resolved_resume,
        show_progress,
    )
