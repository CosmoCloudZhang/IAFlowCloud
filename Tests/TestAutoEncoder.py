import pytest
import torch

from IAFlow.AutoEncoder import Conv1dAutoEncoder
from IAFlow.Config import ModelConfig


@pytest.mark.parametrize(
    ("input_shape", "channels", "kernels", "strides"),
    [
        ((31, 101), [64, 128, 256], [5, 5, 3], [2, 2, 2]),
        ((3, 17), [5, 7], [3, 5], [2, 3]),
        ((4, 16), [8], [3], [1]),
    ],
)
def test_autoencoder_round_trip_shape(input_shape, channels, kernels, strides):
    config = ModelConfig(
        latent_dim=2,
        encoder_channels=channels,
        kernel_sizes=kernels,
        strides=strides,
        dense_hidden=[11],
        group_count=4,
    )
    model = Conv1dAutoEncoder(config, input_shape)
    inputs = torch.randn(4, *input_shape)
    latent = model.encode(inputs)
    reconstruction = model.decode(latent)

    assert latent.shape == (4, 2)
    assert reconstruction.shape == inputs.shape
    reconstruction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_autoencoder_rejects_wrong_input_shape():
    model = Conv1dAutoEncoder(ModelConfig(), (31, 101))
    with pytest.raises(ValueError, match="surfaces must have shape"):
        model(torch.randn(2, 1, 31, 101))


def test_model_configuration_rejects_even_kernels():
    with pytest.raises(ValueError, match="Odd kernel"):
        ModelConfig(
            encoder_channels=[8],
            kernel_sizes=[4],
            strides=[2],
        )
