"""Configurable Conv1D autoencoder for redshift-channel IA surfaces."""

from __future__ import annotations

from math import gcd

import torch
from torch import nn

from .Config import ModelConfig

__all__ = ["Conv1dAutoEncoder"]


def _activation(name: str) -> nn.Module:
    return {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }[name]()


def _normalization(name: str, channels: int, maximum_groups: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    groups = gcd(channels, maximum_groups)
    return nn.GroupNorm(num_groups=max(groups, 1), num_channels=channels)


def _convolution_length(length: int, kernel: int, stride: int) -> int:
    padding = kernel // 2
    return (length + 2 * padding - (kernel - 1) - 1) // stride + 1


class _ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        config: ModelConfig,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            _normalization(config.normalization, out_channels, config.group_count),
            _activation(config.activation),
        ]
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        super().__init__(*layers)


class _DenseBlock(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, config: ModelConfig) -> None:
        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        if config.normalization == "group":
            layers.append(nn.LayerNorm(out_features))
        layers.append(_activation(config.activation))
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        super().__init__(*layers)


class Conv1dAutoEncoder(nn.Module):
    """Symmetric Conv1D autoencoder with an unconstrained linear bottleneck.

    Inputs have shape ``(batch, redshift_channels, wavenumber_samples)``. The
    convolution slides only along wavenumber while mixing all redshift channels.
    No skip connections are used, so every reconstruction passes through the
    requested latent dimension.
    """

    def __init__(self, config: ModelConfig, input_shape: tuple[int, int]) -> None:
        super().__init__()
        self.config = config
        self.input_shape = tuple(int(value) for value in input_shape)
        input_channels, input_length = self.input_shape

        channel_path = [input_channels, *config.encoder_channels]
        lengths = [input_length]
        encoder_blocks: list[nn.Module] = []
        for index, (kernel, stride) in enumerate(zip(config.kernel_sizes, config.strides)):
            encoder_blocks.append(
                _ConvBlock(
                    channel_path[index],
                    channel_path[index + 1],
                    kernel,
                    stride,
                    config,
                )
            )
            lengths.append(_convolution_length(lengths[-1], kernel, stride))
        self.encoder_convolution = nn.Sequential(*encoder_blocks)
        self.encoder_lengths = tuple(lengths)

        flattened_features = config.encoder_channels[-1] * lengths[-1]
        encoder_dense: list[nn.Module] = []
        previous_width = flattened_features
        for width in config.dense_hidden:
            encoder_dense.append(_DenseBlock(previous_width, width, config))
            previous_width = width
        encoder_dense.append(nn.Linear(previous_width, config.latent_dim))
        self.encoder_dense = nn.Sequential(*encoder_dense)

        decoder_dense: list[nn.Module] = []
        previous_width = config.latent_dim
        for width in reversed(config.dense_hidden):
            decoder_dense.append(_DenseBlock(previous_width, width, config))
            previous_width = width
        decoder_dense.append(nn.Linear(previous_width, flattened_features))
        self.decoder_dense = nn.Sequential(*decoder_dense)

        decoder_blocks: list[nn.Module] = []
        reversed_layer_indices = list(reversed(range(len(config.encoder_channels))))
        for decoder_index, encoder_index in enumerate(reversed_layer_indices):
            in_channels = channel_path[encoder_index + 1]
            out_channels = channel_path[encoder_index]
            kernel = config.kernel_sizes[encoder_index]
            stride = config.strides[encoder_index]
            source_length = lengths[encoder_index + 1]
            target_length = lengths[encoder_index]
            padding = kernel // 2
            base_length = (
                (source_length - 1) * stride - 2 * padding + (kernel - 1) + 1
            )
            output_padding = target_length - base_length
            if output_padding < 0 or output_padding >= stride:
                raise ValueError(
                    "The configured convolution stack cannot be inverted to the exact "
                    f"input length at layer {encoder_index}."
                )
            decoder_blocks.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            is_output_layer = decoder_index == len(reversed_layer_indices) - 1
            if not is_output_layer:
                decoder_blocks.extend(
                    [
                        _normalization(
                            config.normalization,
                            out_channels,
                            config.group_count,
                        ),
                        _activation(config.activation),
                    ]
                )
                if config.dropout > 0.0:
                    decoder_blocks.append(nn.Dropout(config.dropout))
        self.decoder_convolution = nn.Sequential(*decoder_blocks)
        self.encoded_shape = (config.encoder_channels[-1], lengths[-1])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Begin close to the training-mean predictor (zero in normalized space)
        # while retaining a small gradient path through the full decoder.
        output_layer = self.decoder_convolution[-1]
        if not isinstance(output_layer, nn.ConvTranspose1d):
            raise RuntimeError("The decoder output layer must be ConvTranspose1d.")
        nn.init.normal_(output_layer.weight, mean=0.0, std=1.0e-3)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    def encode(self, surfaces: torch.Tensor) -> torch.Tensor:
        self._check_input(surfaces)
        features = self.encoder_convolution(surfaces)
        return self.encoder_dense(features.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"latent must have shape (batch, {self.config.latent_dim}), "
                f"found {tuple(latent.shape)}."
            )
        features = self.decoder_dense(latent)
        features = features.reshape(latent.shape[0], *self.encoded_shape)
        reconstructed = self.decoder_convolution(features)
        if tuple(reconstructed.shape[1:]) != self.input_shape:
            raise RuntimeError(
                f"Decoder produced {tuple(reconstructed.shape[1:])}, "
                f"expected {self.input_shape}."
            )
        return reconstructed

    def forward(self, surfaces: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(surfaces))

    def _check_input(self, surfaces: torch.Tensor) -> None:
        if surfaces.ndim != 3 or tuple(surfaces.shape[1:]) != self.input_shape:
            raise ValueError(
                f"surfaces must have shape (batch, {self.input_shape[0]}, "
                f"{self.input_shape[1]}), found {tuple(surfaces.shape)}."
            )

    @property
    def number_of_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def architecture_summary(self) -> dict[str, object]:
        return {
            "input_shape": list(self.input_shape),
            "encoded_shape": list(self.encoded_shape),
            "encoder_lengths": list(self.encoder_lengths),
            "latent_dim": self.config.latent_dim,
            "number_of_parameters": self.number_of_parameters,
            "skip_connections": False,
        }
