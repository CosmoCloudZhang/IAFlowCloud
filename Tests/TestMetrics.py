import pytest
import torch

from IAFlow.Metrics import ReconstructionMetrics


def test_identity_reconstruction_has_perfect_metrics():
    target = torch.tensor(
        [[[1.0, -1.0], [0.5, -0.5]], [[0.2, -0.3], [0.7, -0.8]]]
    )
    metrics = ReconstructionMetrics(normalization_scale=0.25)
    metrics.update(target, target)
    result = metrics.compute()

    assert result["normalized_mse"] == pytest.approx(0.0)
    assert result["log10_rmse"] == pytest.approx(0.0)
    assert result["variance_recovered"] == pytest.approx(1.0)
    assert result["maximum_relative_error"] == pytest.approx(0.0)


def test_training_mean_prediction_recovers_zero_variance():
    target = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]])
    prediction = torch.zeros_like(target)
    metrics = ReconstructionMetrics(normalization_scale=0.5)
    metrics.update(target, prediction)
    result = metrics.compute()

    assert result["normalized_mse"] == pytest.approx(7.5)
    assert result["log10_rmse"] == pytest.approx(0.5 * (7.5**0.5))
    assert result["variance_recovered"] == pytest.approx(0.0)
