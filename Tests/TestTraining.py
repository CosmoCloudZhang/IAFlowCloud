from pathlib import Path

import torch

from IAFlow.Artifacts import load_autoencoder_checkpoint
from IAFlow.Config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from IAFlow.Data import prepare_surface_cache
from IAFlow.Train import fit_autoencoder
from TestData import write_synthetic_source


def test_end_to_end_cpu_training_writes_reloadable_artifacts(tmp_path: Path):
    write_synthetic_source(tmp_path / "Source.hdf5", shape=(3, 9), number_of_rows=20)
    config = ExperimentConfig(
        data=DataConfig(
            source_path="Source.hdf5",
            cache_directory="Cache",
            input_shape=(3, 9),
            preparation_block_size=5,
            batch_size=4,
            evaluation_batch_size=4,
        ),
        model=ModelConfig(
            latent_dim=2,
            encoder_channels=[4, 8],
            kernel_sizes=[3, 3],
            strides=[2, 2],
            dense_hidden=[8],
            group_count=4,
        ),
        training=TrainingConfig(
            epochs=1,
            device="cpu",
            learning_rate=1.0e-3,
            scheduler="none",
            mixed_precision=False,
            early_stopping_patience=3,
        ),
        output=OutputConfig(root_directory="Runs", save_every_epochs=1),
        project_root=tmp_path,
    )
    prepare_surface_cache(config)
    run_directory = tmp_path / "Run"
    first_summary = fit_autoencoder(
        config,
        run_directory=run_directory,
        show_progress=False,
    )
    assert first_summary["epochs_completed"] == 1

    config.training.epochs = 2
    summary = fit_autoencoder(
        config,
        resume_checkpoint=run_directory / "Last.pt",
        show_progress=False,
    )

    assert summary["epochs_completed"] == 2
    assert summary["resumed_from"] == str((run_directory / "Last.pt").resolve())
    assert summary["test_split_used"] is False
    assert summary["target_variance_recovered"] == 0.999
    assert isinstance(summary["target_met"], bool)
    assert (run_directory / "Best.pt").is_file()
    assert (run_directory / "Last.pt").is_file()
    assert (run_directory / "History.json").is_file()
    assert (run_directory / "Environment.json").is_file()

    model, normalization, checkpoint = load_autoencoder_checkpoint(
        run_directory / "Best.pt"
    )
    assert checkpoint["epoch"] in {1, 2}
    assert normalization.mean.shape == (3, 9)
    assert model(torch.zeros(2, 3, 9)).shape == (2, 3, 9)
