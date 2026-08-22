"""
Independent training and validation workflow for the additive PCA-AE model.
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

from .model import PCAAutoEncoder
from ..core.artifacts import portable_path, save_checkpoint, save_json
from ..core.config import config_to_dict
from ..core.data import CachedSurfaceDataset, build_dataloader
from ..core.evaluation import evaluate_autoencoder
from ..core.runs import write_latest_run
from .artifacts import (
    PCA_AE_CHECKPOINT_FORMAT_VERSION,
    build_pca_ae_checkpoint,
    build_pca_autoencoder,
)
from .config import PCAAEExperimentConfig
from .data import (
    CachedPCACoefficientDataset,
    prepare_pca_ae_cache,
)
from .loss import SurfaceEquivalentCoefficientMSE
from ..core.runtime import resolve_device, set_reproducibility

__all__ = ["fit_pca_autoencoder"]


_Scheduler = (
    torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
)


@dataclass(slots=True)
class _TrainingProgress:
    """
    Store mutable PCA-AE optimization and early-stopping state.
    """
    
    start_epoch: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)
    best_mse: float = math.inf
    best_epoch: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)
    epochs_without_improvement: int = 0
    stopped_early: bool = False


def _limited_dataset(
    dataset: Dataset,
    maximum: int | None,
    seed: int,
) -> Dataset:
    """
    Select a reproducible subset for PCA-AE smoke runs.
    
    Arguments:
        dataset (torch.utils.data.Dataset):
            Complete stored split.
        maximum (int or None):
            Optional maximum row count.
        seed (int):
            Seed controlling subset selection.
    
    Returns:
        selected (torch.utils.data.Dataset):
            Complete dataset or reproducible subset.
    """
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum <= 0:
        raise ValueError("Sample limits must be positive.")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=maximum, replace=False)).tolist()
    return Subset(dataset, indices)


def _optimizer(
    model: nn.Module,
    config: Any,
) -> torch.optim.Optimizer:
    """
    Construct the configured PCA-AE optimizer.
    
    Arguments:
        model (torch.nn.Module):
            PCA-AE model whose trainable parameters are optimized.
        config (object):
            Checked training configuration.
    
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
    Construct the configured PCA-AE learning-rate scheduler.
    
    Arguments:
        optimizer (torch.optim.Optimizer):
            Optimizer controlled by the scheduler.
        config (object):
            Checked training configuration.
    
    Returns:
        scheduler (object or None):
            Configured scheduler or None.
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


def _capture_rng_state(
    train_generator: torch.Generator,
) -> dict[str, Any]:
    """
    Capture every random-number state required for exact resume.
    
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
    Restore every PCA-AE random-number state.
    
    Arguments:
        state (dict[str, Any]):
            Stored reproducibility-state mapping.
        train_generator (torch.Generator):
            Training-loader generator receiving its prior state.
    """
    if not isinstance(state, dict):
        raise ValueError("PCA-AE resume checkpoint is missing RNG state.")
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
    Permit only total-epoch and device changes on PCA-AE resume.
    
    Arguments:
        stored (dict[str, Any]):
            Original resolved PCA-AE configuration.
        current (dict[str, Any]):
            Requested resumed configuration.
    
    Returns:
        matches (bool):
            Whether all immutable settings agree exactly.
    """
    stored_copy = copy.deepcopy(stored)
    current_copy = copy.deepcopy(current)
    for values in (stored_copy, current_copy):
        values["training"].pop("epochs", None)
        values["training"].pop("device", None)
    return stored_copy == current_copy


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Move restored optimizer tensors to the PCA-AE training device.
    
    Arguments:
        optimizer (torch.optim.Optimizer):
            Optimizer whose state was restored on CPU.
        device (torch.device):
            Active training device.
    """
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def _run_directory(
    config: PCAAEExperimentConfig,
    explicit: str | Path | None,
    *,
    allow_existing: bool,
) -> Path:
    """
    Resolve and initialize one PCA-AE output directory.
    
    Arguments:
        config (PCAAEExperimentConfig):
            Checked resolved PCA-AE configuration.
        explicit (str or pathlib.Path or None):
            Optional explicit run directory.
        allow_existing (bool):
            Whether an existing directory may be resumed.
    
    Returns:
        directory (pathlib.Path):
            Initialized PCA-AE run directory.
    """
    configured = config.resolve_path(config.output.run_directory)
    directory = config.resolve_path(explicit) if explicit is not None else configured
    if directory != configured:
        raise ValueError("run_directory must match the resolved PCA-AE config.")
    if not allow_existing and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"PCA-AE run directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _train_epoch(
    model: PCAAutoEncoder,
    loader: DataLoader,
    loss_function: SurfaceEquivalentCoefficientMSE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None,
    scaler: torch.amp.GradScaler | None,
    show_progress: bool,
) -> float:
    """
    Optimize one complete PCA-AE coefficient epoch.
    
    Arguments:
        model (PCAAutoEncoder):
            PCA-AE model to optimize.
        loader (torch.utils.data.DataLoader):
            Shuffled coefficient training loader.
        loss_function (SurfaceEquivalentCoefficientMSE):
            Fixed surface-equivalent coefficient objective.
        optimizer (torch.optim.Optimizer):
            Optimizer used for updates.
        device (torch.device):
            Training device.
        gradient_clip_norm (float or None):
            Optional maximum gradient norm.
        scaler (torch.amp.GradScaler or None):
            Optional CUDA mixed-precision scaler.
        show_progress (bool):
            Whether to display batch progress.
    
    Returns:
        mean_loss (float):
            Sample-weighted trainable normalized-surface-MSE contribution.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc="train", leave=False, disable=not show_progress)
    for coefficients, _projection_residual, _baseline_norm in progress:
        coefficients = coefficients.to(
            device,
            non_blocking=device.type == "cuda",
        )
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model.reconstruct_coefficients(coefficients)
                loss = loss_function(prediction, coefficients)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            prediction = model.reconstruct_coefficients(coefficients)
            loss = loss_function(prediction, coefficients)
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        batch_size = len(coefficients)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{total_loss / total_samples:.3e}")
    return total_loss / total_samples


@torch.inference_mode()
def _evaluate_coefficient_objective(
    model: PCAAutoEncoder,
    loader: DataLoader,
    device: torch.device,
    *,
    show_progress: bool,
) -> dict[str, float]:
    """
    Evaluate exact aggregate surface MSE without inverse PCA per epoch.
    
    Arguments:
        model (PCAAutoEncoder):
            PCA-AE model to evaluate.
        loader (torch.utils.data.DataLoader):
            Ordered coefficient validation loader.
        device (torch.device):
            Inference device.
        show_progress (bool):
            Whether to display validation progress.
    
    Returns:
        metrics (dict[str, float]):
            Exact MSE, variance recovery, and PCA-floor decomposition.
    """
    model.eval()
    coefficient_squared_error = 0.0
    projection_squared_error = 0.0
    baseline_normalized_squared_error = 0.0
    number_of_surfaces = 0
    for coefficients, projection_residual, baseline_norm in tqdm(
        loader,
        desc="validate",
        leave=False,
        disable=not show_progress,
    ):
        coefficients = coefficients.to(
            device,
            non_blocking=device.type == "cuda",
        )
        prediction = model.reconstruct_coefficients(coefficients)
        raw_residual = (
            prediction - coefficients
        ) * model.coefficient_scale
        coefficient_squared_error += float(
            torch.sum(torch.square(raw_residual)).item()
        )
        projection_squared_error += float(torch.sum(projection_residual).item())
        baseline_normalized_squared_error += float(torch.sum(baseline_norm).item())
        number_of_surfaces += len(coefficients)
    if number_of_surfaces == 0 or baseline_normalized_squared_error <= 0.0:
        raise ValueError("PCA-AE validation loader is empty or degenerate.")
    number_of_values = number_of_surfaces * int(np.prod(model.input_shape))
    total_log10_squared_error = (
        coefficient_squared_error + projection_squared_error
    )
    surface_scale = float(model.surface_scale.item())
    normalized_squared_error = total_log10_squared_error / surface_scale**2
    normalized_mse = normalized_squared_error / number_of_values
    log10_mse = total_log10_squared_error / number_of_values
    return {
        "normalized_mse": normalized_mse,
        "log10_mse": log10_mse,
        "log10_rmse": math.sqrt(log10_mse),
        "variance_recovered": 1.0
        - normalized_squared_error / baseline_normalized_squared_error,
        "coefficient_normalized_mse": (
            coefficient_squared_error / surface_scale**2 / number_of_values
        ),
        "pca_floor_normalized_mse": (
            projection_squared_error / surface_scale**2 / number_of_values
        ),
    }


def _step_scheduler(
    scheduler: _Scheduler | None,
    scheduler_name: str,
    validation_mse: float,
) -> None:
    """
    Advance the PCA-AE scheduler after validation.
    
    Arguments:
        scheduler (object or None):
            Configured scheduler.
        scheduler_name (str):
            Plateau, cosine, or none.
        validation_mse (float):
            Current complete normalized validation MSE.
    """
    if scheduler is None:
        return
    if scheduler_name == "plateau":
        scheduler.step(validation_mse)
    else:
        scheduler.step()


def _write_initial_artifacts(
    output_directory: Path,
    resolved_config: dict[str, Any],
    model: PCAAutoEncoder,
    metadata: dict[str, Any],
    device: torch.device,
) -> None:
    """
    Write initial PCA-AE configuration and provenance artifacts.
    
    Arguments:
        output_directory (pathlib.Path):
            New PCA-AE run directory.
        resolved_config (dict[str, Any]):
            JSON-safe resolved PCA-AE configuration.
        model (PCAAutoEncoder):
            Newly initialized PCA-AE model.
        metadata (dict[str, Any]):
            Checked coefficient-cache manifest.
        device (torch.device):
            Selected training device.
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


def _record_resume_event(
    config: PCAAEExperimentConfig,
    output_directory: Path,
    checkpoint_path: Path,
    start_epoch: int,
    device: torch.device,
) -> None:
    """
    Append one portable PCA-AE resume event.
    
    Arguments:
        config (PCAAEExperimentConfig):
            Checked resolved PCA-AE configuration.
        output_directory (pathlib.Path):
            Resumed run directory.
        checkpoint_path (pathlib.Path):
            Checkpoint used for continuation.
        start_epoch (int):
            First epoch after restoration.
        device (torch.device):
            Resumed training device.
    """
    path = output_directory / "ResumeHistory.json"
    events: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            stored = json.load(stream)
        if not isinstance(stored, list):
            raise ValueError("PCA-AE ResumeHistory.json must contain a list.")
        events = stored
    events.append(
        {
            "checkpoint": portable_path(checkpoint_path, config.project_root),
            "from_epoch": start_epoch - 1,
            "new_total_epochs": config.training.epochs,
            "device": str(device),
            "timestamp": datetime.now().astimezone().isoformat(),
        }
    )
    save_json(events, path)


def _restore_run(
    config: PCAAEExperimentConfig,
    checkpoint_path: Path,
    output_directory: Path,
    resolved_config: dict[str, Any],
    model: PCAAutoEncoder,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: _Scheduler | None,
    scaler: torch.amp.GradScaler | None,
    train_generator: torch.Generator,
    device: torch.device,
) -> tuple[dict[str, Any], _TrainingProgress]:
    """
    Validate and restore one interrupted PCA-AE training trajectory.
    
    Arguments:
        config (PCAAEExperimentConfig):
            Current resolved PCA-AE configuration.
        checkpoint_path (pathlib.Path):
            Last.pt checkpoint used for resume.
        output_directory (pathlib.Path):
            Existing run directory.
        resolved_config (dict[str, Any]):
            Current JSON-safe configuration.
        model (PCAAutoEncoder):
            Newly initialized model receiving saved state.
        metadata (dict[str, Any]):
            Current coefficient-cache manifest.
        optimizer (torch.optim.Optimizer):
            Optimizer receiving saved state.
        scheduler (object or None):
            Scheduler receiving saved state.
        scaler (torch.amp.GradScaler or None):
            Optional mixed-precision scaler receiving saved state.
        train_generator (torch.Generator):
            Loader generator receiving saved state.
        device (torch.device):
            Active training device.
    
    Returns:
        result (tuple[dict[str, Any], _TrainingProgress]):
            Original configuration and restored progress.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("PCA-AE resume checkpoint must contain a mapping.")
    if checkpoint.get("checkpoint_format_version") != PCA_AE_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("PCA-AE resume checkpoint format is unsupported.")
    stored_config = checkpoint.get("experiment_config")
    if not isinstance(stored_config, dict) or not _strict_resume_config_matches(
        stored_config,
        resolved_config,
    ):
        raise ValueError("PCA-AE resume configuration differs outside epochs/device.")
    if checkpoint.get("model_config") != dict(config.model):
        raise ValueError("PCA-AE resume model configuration does not match.")
    if checkpoint.get("coefficient_provenance") != metadata:
        raise ValueError("PCA-AE resume data provenance does not match.")
    if checkpoint.get("pca_transform_sha256") != model.transform_sha256:
        raise ValueError("PCA-AE resume uses a different PCA transform.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    _restore_rng_state(checkpoint["rng_state"], train_generator)
    start_epoch = int(checkpoint["epoch"]) + 1
    if start_epoch > config.training.epochs:
        raise ValueError("PCA-AE checkpoint already reached the requested epochs.")
    history_path = output_directory / "History.json"
    with history_path.open("r", encoding="utf-8") as stream:
        history = json.load(stream)
    if not isinstance(history, list) or not history:
        raise ValueError("PCA-AE resume requires non-empty History.json.")
    if int(history[-1]["epoch"]) != start_epoch - 1:
        raise ValueError("PCA-AE history does not end at the resume epoch.")
    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("PCA-AE resume checkpoint lacks training state.")
    progress = _TrainingProgress(
        start_epoch=start_epoch,
        history=history,
        best_mse=float(training_state["best_mse"]),
        best_epoch=int(training_state["best_epoch"]),
        best_metrics=dict(training_state["best_metrics"]),
        epochs_without_improvement=int(
            training_state["epochs_without_improvement"]
        ),
    )
    if progress.epochs_without_improvement >= config.training.early_stopping_patience:
        raise ValueError("The original PCA-AE run had already stopped early.")
    _record_resume_event(
        config,
        output_directory,
        checkpoint_path,
        start_epoch,
        device,
    )
    return stored_config, progress


def _save_epoch_checkpoints(
    checkpoint: dict[str, Any],
    output_directory: Path,
    epoch: int,
    save_every_epochs: int,
    improved: bool,
) -> None:
    """
    Save latest, periodic, and best PCA-AE checkpoint variants.
    
    Arguments:
        checkpoint (dict[str, Any]):
            Complete checkpoint payload.
        output_directory (pathlib.Path):
            Run directory receiving checkpoint files.
        epoch (int):
            Completed one-based epoch.
        save_every_epochs (int):
            Periodic archival interval.
        improved (bool):
            Whether this checkpoint is a new validation best.
    """
    save_checkpoint(checkpoint, output_directory / "Last.pt")
    if epoch % save_every_epochs == 0:
        save_checkpoint(checkpoint, output_directory / f"Epoch{epoch:04d}.pt")
    if improved:
        save_checkpoint(checkpoint, output_directory / "Best.pt")


def fit_pca_autoencoder(
    config: PCAAEExperimentConfig,
    *,
    prepare_data: bool = False,
    overwrite_cache: bool = False,
    maximum_train_samples: int | None = None,
    maximum_validation_samples: int | None = None,
    resume_checkpoint: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Train PCA-AE on coefficients and select using exact validation surface MSE.
    
    The procedure is separate from direct-AE training and never reads the test
    split. Complete surface metrics are evaluated only for the selected
    validation checkpoint.
    
    Arguments:
        config (PCAAEExperimentConfig):
            Fully resolved PCA-AE configuration.
        prepare_data (bool):
            Whether to prepare or check the coefficient cache before training.
        overwrite_cache (bool):
            Whether preparation may regenerate existing cache products.
        maximum_train_samples (int or None):
            Optional reproducible training-split smoke limit.
        maximum_validation_samples (int or None):
            Optional reproducible validation-split smoke limit.
        resume_checkpoint (str or pathlib.Path or None):
            Optional Last.pt checkpoint from the same run.
        show_progress (bool):
            Whether to show batch progress bars.
    
    Returns:
        summary (dict[str, Any]):
            Final validation-selected PCA-AE run summary.
    """
    set_reproducibility(config.training.seed, config.training.deterministic)
    if prepare_data:
        prepare_pca_ae_cache(config, overwrite=overwrite_cache)
    (
        model,
        surface_normalization,
        coefficient_normalization,
        metadata,
    ) = build_pca_autoencoder(config)
    device = resolve_device(config.training.device)
    model.to(device)
    train_dataset = _limited_dataset(
        CachedPCACoefficientDataset(config, "train"),
        maximum_train_samples,
        config.training.seed,
    )
    validation_dataset = _limited_dataset(
        CachedPCACoefficientDataset(config, "validation"),
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
    resume_path = (
        config.resolve_path(resume_checkpoint)
        if resume_checkpoint is not None
        else None
    )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"PCA-AE resume checkpoint not found: {resume_path}")
    output_directory = _run_directory(
        config,
        config.output.run_directory,
        allow_existing=resume_path is not None,
    )
    loss_function = SurfaceEquivalentCoefficientMSE(
        model.coefficient_scale.detach().cpu(),
        int(np.prod(model.input_shape)),
        surface_normalization.scale,
    ).to(device)
    optimizer = _optimizer(model, config.training)
    scheduler = _scheduler(optimizer, config.training)
    use_scaler = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    progress = _TrainingProgress()
    if resume_path is not None:
        resolved_config, progress = _restore_run(
            config,
            resume_path,
            output_directory,
            resolved_config,
            model,
            metadata,
            optimizer,
            scheduler,
            scaler,
            train_generator,
            device,
        )
    else:
        _write_initial_artifacts(
            output_directory,
            resolved_config,
            model,
            metadata,
            device,
        )
    writer = SummaryWriter(
        log_dir=output_directory / "TensorBoard",
        purge_step=progress.start_epoch if resume_path is not None else None,
    )
    try:
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
            validation_metrics = _evaluate_coefficient_objective(
                model,
                validation_loader,
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
            checkpoint = build_pca_ae_checkpoint(
                model,
                surface_normalization,
                coefficient_normalization,
                epoch=epoch,
                validation_metrics=validation_metrics,
                experiment_config=resolved_config,
                coefficient_provenance=metadata,
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
            duration_seconds = time.perf_counter() - epoch_start
            writer.add_scalar("loss/train_weighted_coefficients", train_loss, epoch)
            writer.add_scalar("loss/validation_normalized_mse", validation_mse, epoch)
            writer.add_scalar(
                "loss/validation_pca_floor",
                validation_metrics["pca_floor_normalized_mse"],
                epoch,
            )
            writer.add_scalar(
                "metrics/validation_variance_recovered",
                validation_metrics["variance_recovered"],
                epoch,
            )
            writer.add_scalar("optimization/learning_rate", learning_rate, epoch)
            writer.add_scalar("timing/training_seconds", training_seconds, epoch)
            writer.add_scalar("timing/validation_seconds", validation_seconds, epoch)
            writer.add_scalar("timing/checkpoint_seconds", checkpoint_seconds, epoch)
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
            if (
                progress.epochs_without_improvement
                >= config.training.early_stopping_patience
            ):
                progress.stopped_early = True
                break
    finally:
        writer.close()
    best_path = output_directory / "Best.pt"
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model.to(device)
    complete_validation_dataset = CachedSurfaceDataset(config, "validation")
    if maximum_validation_samples is None:
        surface_validation_dataset: Dataset = complete_validation_dataset
    else:
        coefficient_subset = validation_dataset
        if not isinstance(coefficient_subset, Subset):
            raise RuntimeError("Expected a smoke-limited PCA-AE validation subset.")
        surface_validation_dataset = Subset(
            complete_validation_dataset,
            coefficient_subset.indices,
        )
    surface_validation_loader = build_dataloader(
        surface_validation_dataset,
        config.data,
        training=False,
        seed=config.training.seed,
        device=device,
    )
    complete_metrics = evaluate_autoencoder(
        model,
        surface_validation_loader,
        surface_normalization,
        device,
        show_progress=show_progress,
    )
    best_checkpoint["validation_metrics"] = complete_metrics
    training_state = best_checkpoint["training_state"]
    training_state["best_metrics"] = complete_metrics
    save_checkpoint(best_checkpoint, best_path)
    save_json(
        {
            "checkpoint": portable_path(best_path, config.project_root),
            "checkpoint_epoch": int(best_checkpoint["epoch"]),
            "split": "validation",
            "final_test": False,
            "device": str(device),
            "metrics": complete_metrics,
        },
        output_directory / "ValidationMetrics.json",
    )
    progress.best_metrics = complete_metrics
    summary: dict[str, Any] = {
        "run_directory": portable_path(output_directory, config.project_root),
        "model_family": "PCA_AE",
        "device": str(device),
        "pca_rank": int(config.model.pca_rank),
        "pca_transform_sha256": model.transform_sha256,
        "number_of_parameters": model.number_of_parameters,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_epoch": progress.best_epoch,
        "best_validation_metrics": complete_metrics,
        "target_variance_recovered": config.training.target_variance_recovered,
        "target_met": bool(
            complete_metrics["variance_recovered"]
            >= config.training.target_variance_recovered
        ),
        "epochs_completed": len(progress.history),
        "resumed_from": (
            portable_path(resume_path, config.project_root)
            if resume_path is not None
            else None
        ),
        "stopped_early": progress.stopped_early,
        "test_split_used_during_training": False,
    }
    save_json(summary, output_directory / "Summary.json")
    write_latest_run(config, output_directory)
    return summary
