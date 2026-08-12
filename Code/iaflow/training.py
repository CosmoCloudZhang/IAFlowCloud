"""Device-portable training and evaluation procedures."""

from __future__ import annotations

import math
import platform
import random
import sys
import time
import json
import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .artifacts import (
    CHECKPOINT_FORMAT_VERSION,
    build_checkpoint,
    portable_path,
    save_checkpoint,
    save_json,
)
from .config import ExperimentConfig, config_to_dict
from .data import (
    CachedSurfaceDataset,
    NormalizationStats,
    build_dataloader,
    prepare_surface_cache,
    validate_surface_cache,
)
from .evaluation import evaluate_autoencoder
from .losses import build_reconstruction_loss
from .architectures import Conv1dAutoEncoder, build_autoencoder

__all__ = [
    "fit_autoencoder",
    "resolve_device",
    "set_reproducibility",
]


def set_reproducibility(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without silently forcing slow kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _capture_rng_state(train_generator: torch.Generator) -> dict[str, Any]:
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


def _restore_rng_state(state: dict[str, Any], train_generator: torch.Generator) -> None:
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
    stored: dict[str, Any], current: dict[str, Any]
) -> bool:
    """Allow only the documented total-epoch and device changes on resume."""
    stored_copy = copy.deepcopy(stored)
    current_copy = copy.deepcopy(current)
    for values in (stored_copy, current_copy):
        values["training"].pop("epochs", None)
        values["training"].pop("device", None)
    return stored_copy == current_copy


def resolve_device(requested: str = "auto") -> torch.device:
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


def _optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[config.optimizer]
    return optimizer_class(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _scheduler(optimizer: torch.optim.Optimizer, config):
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


def _limited_dataset(dataset: Dataset, maximum: int | None, seed: int) -> Dataset:
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
    if explicit is not None:
        path = Path(explicit).expanduser()
        path = path.resolve() if path.is_absolute() else (config.project_root / path).resolve()
    else:
        run_name = config.output.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        path = config.resolve_path(config.output.root_directory) / run_name
    if not allow_existing and path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _train_epoch(
    model: Conv1dAutoEncoder,
    loader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None,
    scaler: torch.amp.GradScaler | None,
    show_progress: bool,
) -> float:
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
    """Train on the stored training split and select only on validation MSE."""
    set_reproducibility(config.training.seed, config.training.deterministic)
    if prepare_data:
        prepare_surface_cache(config, overwrite=overwrite_cache)
    metadata = validate_surface_cache(config)
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

    resolved_config = config_to_dict(config)
    resolved_config["runtime"] = {
        "maximum_train_samples": maximum_train_samples,
        "maximum_validation_samples": maximum_validation_samples,
    }

    resolved_resume: Path | None = None
    if resume_checkpoint is not None:
        resolved_resume = Path(resume_checkpoint).expanduser()
        if not resolved_resume.is_absolute():
            resolved_resume = (config.project_root / resolved_resume).resolve()
        else:
            resolved_resume = resolved_resume.resolve()
        if not resolved_resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resolved_resume}")
        inferred_run_directory = resolved_resume.parent
        if run_directory is not None:
            requested_run_directory = Path(run_directory).expanduser()
            if not requested_run_directory.is_absolute():
                requested_run_directory = config.project_root / requested_run_directory
            if requested_run_directory.resolve() != inferred_run_directory:
                raise ValueError("A resumed run must write back to the checkpoint directory.")
        run_directory = inferred_run_directory

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
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_mse = math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    resume_events: list[dict[str, Any]] = []

    if resolved_resume is not None:
        resumed = torch.load(resolved_resume, map_location="cpu", weights_only=True)
        if resumed.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Resume checkpoint format is unsupported or stale.")
        stored_config = resumed.get("experiment_config")
        if not isinstance(stored_config, dict) or not _strict_resume_config_matches(
            stored_config, resolved_config
        ):
            raise ValueError(
                "Resume configuration differs from the original run outside epochs/device."
            )
        resolved_config = stored_config
        if resumed.get("model_config") != dict(config.model):
            raise ValueError("Resume checkpoint model configuration does not match.")
        if tuple(resumed.get("input_shape", ())) != config.data.input_shape:
            raise ValueError("Resume checkpoint input shape does not match.")
        resumed_normalization = resumed["normalization"]
        if not np.allclose(resumed_normalization["mean"].numpy(), normalization.mean) or not np.isclose(
            float(resumed_normalization["scale"]), normalization.scale
        ):
            raise ValueError("Resume checkpoint normalization does not match the data cache.")
        provenance_keys = {
            "source_structure_sha256",
            "source_size",
            "target_dataset",
            "transform",
            "normalization",
            "input_shape",
        }
        stored_provenance = resumed.get("data_provenance", {})
        if any(stored_provenance.get(key) != metadata.get(key) for key in provenance_keys):
            raise ValueError("Resume checkpoint data provenance does not match the cache.")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        if scheduler is not None and "scheduler_state_dict" in resumed:
            scheduler.load_state_dict(resumed["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in resumed:
            scaler.load_state_dict(resumed["scaler_state_dict"])
        start_epoch = int(resumed["epoch"]) + 1
        if start_epoch > config.training.epochs:
            raise ValueError(
                "The resume checkpoint has already reached the configured total epochs."
            )

        history_path = output_directory / "History.json"
        if not history_path.is_file():
            raise FileNotFoundError("A resumed run requires its existing History.json.")
        with history_path.open("r", encoding="utf-8") as stream:
            history = json.load(stream)
        if not history or int(history[-1]["epoch"]) != start_epoch - 1:
            raise ValueError("History.json does not end at the resume checkpoint epoch.")
        training_state = resumed.get("training_state")
        if not isinstance(training_state, dict):
            raise ValueError("Resume checkpoint is missing early-stopping state.")
        best_epoch = int(training_state["best_epoch"])
        best_mse = float(training_state["best_mse"])
        best_metrics = dict(training_state["best_metrics"])
        epochs_without_improvement = int(training_state["epochs_without_improvement"])
        if epochs_without_improvement >= config.training.early_stopping_patience:
            raise ValueError("The original run had already reached its early-stopping condition.")
        _restore_rng_state(resumed.get("rng_state"), train_generator)

        resolved_config_path = output_directory / "ResolvedConfig.json"
        if not resolved_config_path.is_file():
            raise FileNotFoundError("A resumed run requires its original ResolvedConfig.json.")
        with resolved_config_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != stored_config:
                raise ValueError("ResolvedConfig.json differs from the resume checkpoint.")
        resume_history_path = output_directory / "ResumeHistory.json"
        if resume_history_path.is_file():
            with resume_history_path.open("r", encoding="utf-8") as stream:
                resume_events = json.load(stream)
        resume_events.append(
            {
                "checkpoint": portable_path(resolved_resume, config.project_root),
                "from_epoch": start_epoch - 1,
                "new_total_epochs": config.training.epochs,
                "device": str(device),
                "timestamp": datetime.now().astimezone().isoformat(),
            }
        )
        save_json(resume_events, resume_history_path)

    writer = SummaryWriter(
        log_dir=output_directory / "TensorBoard",
        purge_step=start_epoch if resolved_resume is not None else None,
    )

    if resolved_resume is None:
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

    stopped_early = False

    try:
        for epoch in range(start_epoch, config.training.epochs + 1):
            start_time = time.perf_counter()
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
            validation_metrics = evaluate_autoencoder(
                model,
                validation_loader,
                normalization,
                device,
                show_progress=show_progress,
            )
            validation_mse = validation_metrics["normalized_mse"]
            if scheduler is not None:
                if config.training.scheduler == "plateau":
                    scheduler.step(validation_mse)
                else:
                    scheduler.step()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "duration_seconds": time.perf_counter() - start_time,
                "validation": validation_metrics,
            }
            history.append(epoch_record)
            save_json(history, output_directory / "History.json")

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/validation_mse", validation_mse, epoch)
            writer.add_scalar(
                "metrics/validation_variance_recovered",
                validation_metrics["variance_recovered"],
                epoch,
            )
            writer.add_scalar("optimization/learning_rate", learning_rate, epoch)

            improvement = best_mse - validation_mse
            improved = improvement > config.training.minimum_improvement
            if improved:
                best_mse = validation_mse
                best_epoch = epoch
                best_metrics = validation_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            checkpoint = build_checkpoint(
                model,
                normalization,
                epoch=epoch,
                validation_metrics=validation_metrics,
                experiment_config=resolved_config,
                data_provenance=metadata,
                training_state={
                    "best_epoch": best_epoch,
                    "best_mse": best_mse,
                    "best_metrics": best_metrics,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                rng_state=_capture_rng_state(train_generator),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            save_checkpoint(checkpoint, output_directory / "Last.pt")
            if epoch % config.output.save_every_epochs == 0:
                save_checkpoint(checkpoint, output_directory / f"Epoch{epoch:04d}.pt")
            if improved:
                save_checkpoint(checkpoint, output_directory / "Best.pt")

            print(
                f"Epoch {epoch:04d} | train={train_loss:.4e} | "
                f"val={validation_mse:.4e} | "
                f"variance={validation_metrics['variance_recovered']:.6f} | "
                f"lr={learning_rate:.2e}"
            )
            if epochs_without_improvement >= config.training.early_stopping_patience:
                stopped_early = True
                break
    finally:
        writer.close()

    summary: dict[str, Any] = {
        "run_directory": portable_path(output_directory, config.project_root),
        "device": str(device),
        "number_of_parameters": model.number_of_parameters,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "target_variance_recovered": config.training.target_variance_recovered,
        "target_met": bool(
            best_metrics.get("variance_recovered", -math.inf)
            >= config.training.target_variance_recovered
        ),
        "epochs_completed": len(history),
        "resumed_from": (
            portable_path(resolved_resume, config.project_root)
            if resolved_resume is not None
            else None
        ),
        "stopped_early": stopped_early,
        "test_split_used_during_training": False,
    }
    save_json(summary, output_directory / "Summary.json")
    latest_file = config.resolve_path(config.output.root_directory) / "LatestRun.txt"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(
        f"{portable_path(output_directory, config.project_root)}\n", encoding="utf-8"
    )
    return summary
