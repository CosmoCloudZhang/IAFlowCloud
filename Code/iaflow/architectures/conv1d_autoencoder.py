"""
Configurable Conv1D autoencoder for redshift-channel IA surfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd

import torch
from torch import nn

from .base import AutoEncoder

__all__ = ["Conv1dAutoEncoder"]


def _activation(
    name: str,
) -> nn.Module:
    """
    Construct an activation layer from its configuration name.
    
    Arguments:
        name (str):
            Supported activation name.
    
    Returns:
        activation (torch.nn.Module):
            Newly constructed stateless activation layer.
    """
    return {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }[name]()


def _normalization(
    name: str,
    channels: int,
    maximum_groups: int,
) -> nn.Module:
    """
    Construct an identity or channel-compatible GroupNorm layer.
    
    Arguments:
        name (str):
            Normalization choice from the model configuration.
        channels (int):
            Number of feature channels to normalize.
        maximum_groups (int):
            Maximum requested number of normalization groups.
    
    Returns:
        normalization (torch.nn.Module):
            Configured normalization layer.
    """
    if name == "none":
        return nn.Identity()
    groups = gcd(channels, maximum_groups)
    return nn.GroupNorm(num_groups=max(groups, 1), num_channels=channels)


def _convolution_length(
    length: int,
    kernel: int,
    stride: int,
) -> int:
    """
    Calculate the output length of one symmetrically padded convolution.
    
    Arguments:
        length (int):
            Input sequence length.
        kernel (int):
            Convolution kernel size.
        stride (int):
            Convolution stride.
    
    Returns:
        output_length (int):
            Sequence length produced by the convolution.
    """
    padding = kernel // 2
    return (length + 2 * padding - (kernel - 1) - 1) // stride + 1


class _ConvBlock(nn.Sequential):
    """
    Combine convolution, optional normalization, activation, and dropout.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        config: object,
    ) -> None:
        """
        Construct one configurable encoder convolution block.
        
        Arguments:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            kernel_size (int):
                Convolution kernel size.
            stride (int):
                Convolution stride.
            config (object):
                Checked Conv1D architecture configuration.
        """
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
    """
    Combine a linear map, optional normalization, activation, and dropout.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: object,
    ) -> None:
        """
        Construct one configurable dense block.
        
        Arguments:
            in_features (int):
                Number of input features.
            out_features (int):
                Number of output features.
            config (object):
                Checked Conv1D architecture configuration.
        """
        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        if config.normalization == "group":
            layers.append(nn.LayerNorm(out_features))
        layers.append(_activation(config.activation))
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        super().__init__(*layers)


def _build_encoder_convolution(
    config: object,
    input_channels: int,
    input_length: int,
) -> tuple[nn.Sequential, tuple[int, ...]]:
    """
    Build the encoder convolution stack and record every sequence length.
    
    Arguments:
        config (object):
            Checked Conv1D architecture configuration.
        input_channels (int):
            Number of redshift channels in one input surface.
        input_length (int):
            Number of wavenumber samples in one input channel.
    
    Returns:
        result (tuple[torch.nn.Sequential, tuple[int, ...]]):
            Encoder layers and the input/output length at every level.
    """
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
    return nn.Sequential(*encoder_blocks), tuple(lengths)


def _build_dense_stack(
    input_features: int,
    hidden_features: Sequence[int],
    output_features: int,
    config: object,
) -> nn.Sequential:
    """
    Build one configurable dense stack ending in a linear output map.
    
    Arguments:
        input_features (int):
            Width entering the first dense layer.
        hidden_features (collections.abc.Sequence[int]):
            Ordered hidden-layer widths.
        output_features (int):
            Width produced by the final linear layer.
        config (object):
            Checked Conv1D architecture configuration.
    
    Returns:
        layers (torch.nn.Sequential):
            Dense hidden blocks followed by the linear output map.
    """
    layers: list[nn.Module] = []
    previous_width = input_features
    for width in hidden_features:
        layers.append(_DenseBlock(previous_width, width, config))
        previous_width = width
    layers.append(nn.Linear(previous_width, output_features))
    return nn.Sequential(*layers)


def _build_decoder_convolution(
    config: object,
    input_channels: int,
    encoder_lengths: tuple[int, ...],
) -> nn.Sequential:
    """
    Build the transpose-convolution stack that exactly restores the input length.
    
    Arguments:
        config (object):
            Checked Conv1D architecture configuration.
        input_channels (int):
            Number of redshift channels in the reconstructed surface.
        encoder_lengths (tuple[int, ...]):
            Sequence lengths before and after every encoder convolution.
    
    Returns:
        decoder (torch.nn.Sequential):
            Exactly invertible decoder convolution stack.
    """
    channel_path = [input_channels, *config.encoder_channels]
    decoder_blocks: list[nn.Module] = []
    reversed_layer_indices = list(reversed(range(len(config.encoder_channels))))
    for decoder_index, encoder_index in enumerate(reversed_layer_indices):
        in_channels = channel_path[encoder_index + 1]
        out_channels = channel_path[encoder_index]
        kernel = config.kernel_sizes[encoder_index]
        stride = config.strides[encoder_index]
        source_length = encoder_lengths[encoder_index + 1]
        target_length = encoder_lengths[encoder_index]
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
    return nn.Sequential(*decoder_blocks)


class Conv1dAutoEncoder(AutoEncoder):
    """
    Symmetric Conv1D autoencoder with an unconstrained linear bottleneck.
    
    Inputs have shape ``(batch, redshift_channels, wavenumber_samples)``. The
    convolution slides only along wavenumber while mixing all redshift channels.
    No skip connections are used, so every reconstruction passes through the
    requested latent dimension.
    """
    
    def __init__(self, config: object, input_shape: tuple[int, int]) -> None:
        """
        Construct the encoder, bottleneck, and exactly invertible decoder stack.
        
        Arguments:
            config (object):
                Checked Conv1D architecture configuration.
            input_shape (tuple[int, int]):
                Redshift-channel count and wavenumber-sample count.
        """
        super().__init__(config, input_shape)
        input_channels, input_length = self.input_shape
        
        self.encoder_convolution, self.encoder_lengths = _build_encoder_convolution(
            config,
            input_channels,
            input_length,
        )
        
        flattened_features = config.encoder_channels[-1] * self.encoder_lengths[-1]
        self.encoder_dense = _build_dense_stack(
            flattened_features,
            config.dense_hidden,
            config.latent_dim,
            config,
        )
        
        self.decoder_dense = _build_dense_stack(
            config.latent_dim,
            tuple(reversed(config.dense_hidden)),
            flattened_features,
            config,
        )
        
        self.decoder_convolution = _build_decoder_convolution(
            config,
            input_channels,
            self.encoder_lengths,
        )
        self.encoded_shape = (
            config.encoder_channels[-1],
            self.encoder_lengths[-1],
        )
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """
        Initialize trainable layers and a near-zero decoder output map.
        """
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
        """
        Encode normalized input surfaces into latent coordinates.
        
        Arguments:
            surfaces (torch.Tensor):
                Normalized surfaces with shape (batch, channels, length).
        
        Returns:
            latent (torch.Tensor):
                Latent coordinates with shape (batch, latent_dim).
        """
        self._check_input(surfaces)
        features = self.encoder_convolution(surfaces)
        return self.encoder_dense(features.flatten(start_dim=1))
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent coordinates into normalized surface reconstructions.
        
        Arguments:
            latent (torch.Tensor):
                Coordinates with shape (batch, latent_dim).
        
        Returns:
            reconstructed (torch.Tensor):
                Reconstructed surfaces with the configured input shape.
        """
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
        """
        Reconstruct normalized surfaces through the latent bottleneck.
        
        Arguments:
            surfaces (torch.Tensor):
                Normalized input surfaces.
        
        Returns:
            reconstructed (torch.Tensor):
                Normalized reconstructed surfaces.
        """
        return self.decode(self.encode(surfaces))
    
    def _check_input(self, surfaces: torch.Tensor) -> None:
        """
        Check a batch against the configured channel and length dimensions.
        
        Arguments:
            surfaces (torch.Tensor):
                Candidate input batch.
        """
        if surfaces.ndim != 3 or tuple(surfaces.shape[1:]) != self.input_shape:
            raise ValueError(
                f"surfaces must have shape (batch, {self.input_shape[0]}, "
                f"{self.input_shape[1]}), found {tuple(surfaces.shape)}."
            )
    
    @property
    def number_of_parameters(self) -> int:
        """
        Return the total number of trainable and non-trainable parameters.
        
        Returns:
            count (int):
                Number of scalar model parameters.
        """
        return sum(parameter.numel() for parameter in self.parameters())
    
    def architecture_summary(self) -> dict[str, object]:
        """
        Return a JSON-safe summary of the resolved architecture.
        
        Returns:
            summary (dict[str, object]):
                Input, encoded, bottleneck, and parameter-count metadata.
        """
        return {
            "name": "Conv1D",
            "input_shape": list(self.input_shape),
            "encoded_shape": list(self.encoded_shape),
            "encoder_lengths": list(self.encoder_lengths),
            "latent_dim": self.config.latent_dim,
            "number_of_parameters": self.number_of_parameters,
            "skip_connections": False,
        }
