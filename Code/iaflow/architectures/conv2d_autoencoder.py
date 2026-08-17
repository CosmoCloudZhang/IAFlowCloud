"""
Configurable Conv2D autoencoder for joint redshift-wavenumber IA surfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd

import torch
from torch import nn

from .base import AutoEncoder

__all__ = ["Conv2dAutoEncoder"]


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


def _convolution_size(
    size: tuple[int, int],
    kernel: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[int, int]:
    """
    Calculate the output size of one symmetrically padded convolution.
    
    Arguments:
        size (tuple[int, int]):
            Input redshift and wavenumber sizes.
        kernel (tuple[int, int]):
            Two-dimensional convolution kernel.
        stride (tuple[int, int]):
            Two-dimensional convolution stride.
    
    Returns:
        output_size (tuple[int, int]):
            Redshift and wavenumber sizes produced by the convolution.
    """
    return tuple(
        (length + 2 * (width // 2) - (width - 1) - 1) // step + 1
        for length, width, step in zip(size, kernel, stride)
    )


class _ConvBlock(nn.Sequential):
    """
    Combine Conv2D, optional normalization, activation, and dropout.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        config: object,
    ) -> None:
        """
        Construct one configurable encoder convolution block.
        
        Arguments:
            in_channels (int):
                Number of input feature channels.
            out_channels (int):
                Number of output feature channels.
            kernel_size (tuple[int, int]):
                Redshift and wavenumber kernel dimensions.
            stride (tuple[int, int]):
                Redshift and wavenumber strides.
            config (object):
                Checked Conv2D architecture configuration.
        """
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=tuple(width // 2 for width in kernel_size),
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
                Checked Conv2D architecture configuration.
        """
        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        if config.normalization == "group":
            layers.append(nn.LayerNorm(out_features))
        layers.append(_activation(config.activation))
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        super().__init__(*layers)


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
            Checked Conv2D architecture configuration.
    
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


def _build_encoder_convolution(
    config: object,
    input_size: tuple[int, int],
) -> tuple[nn.Sequential, tuple[tuple[int, int], ...]]:
    """
    Build the encoder stack and record every joint-surface size.
    
    Arguments:
        config (object):
            Checked Conv2D architecture configuration.
        input_size (tuple[int, int]):
            Redshift and wavenumber sizes of one surface.
    
    Returns:
        result (tuple[torch.nn.Sequential, tuple[tuple[int, int], ...]]):
            Encoder layers and the surface size at every level.
    """
    channel_path = [1, *config.encoder_channels]
    sizes = [input_size]
    blocks: list[nn.Module] = []
    for index, (kernel, stride) in enumerate(zip(config.kernel_sizes, config.strides)):
        blocks.append(
            _ConvBlock(
                channel_path[index],
                channel_path[index + 1],
                kernel,
                stride,
                config,
            )
        )
        sizes.append(_convolution_size(sizes[-1], kernel, stride))
    return nn.Sequential(*blocks), tuple(sizes)


def _build_decoder_convolution(
    config: object,
    encoder_sizes: tuple[tuple[int, int], ...],
) -> nn.Sequential:
    """
    Build the transpose-convolution stack that exactly restores both axes.
    
    Arguments:
        config (object):
            Checked Conv2D architecture configuration.
        encoder_sizes (tuple[tuple[int, int], ...]):
            Surface sizes before and after every encoder convolution.
    
    Returns:
        decoder (torch.nn.Sequential):
            Exactly shape-inverting decoder stack.
    """
    channel_path = [1, *config.encoder_channels]
    blocks: list[nn.Module] = []
    indices = list(reversed(range(len(config.encoder_channels))))
    for decoder_index, encoder_index in enumerate(indices):
        kernel = config.kernel_sizes[encoder_index]
        stride = config.strides[encoder_index]
        source_size = encoder_sizes[encoder_index + 1]
        target_size = encoder_sizes[encoder_index]
        padding = tuple(width // 2 for width in kernel)
        base_size = tuple(
            (source - 1) * step - 2 * pad + (width - 1) + 1
            for source, step, pad, width in zip(
                source_size,
                stride,
                padding,
                kernel,
            )
        )
        output_padding = tuple(
            target - base for target, base in zip(target_size, base_size)
        )
        if any(
            value < 0 or value >= step
            for value, step in zip(output_padding, stride)
        ):
            raise ValueError(
                "The configured Conv2D stack cannot exactly invert layer "
                f"{encoder_index}."
            )
        blocks.append(
            nn.ConvTranspose2d(
                channel_path[encoder_index + 1],
                channel_path[encoder_index],
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        )
        is_output_layer = decoder_index == len(indices) - 1
        if not is_output_layer:
            blocks.extend(
                [
                    _normalization(
                        config.normalization,
                        channel_path[encoder_index],
                        config.group_count,
                    ),
                    _activation(config.activation),
                ]
            )
            if config.dropout > 0.0:
                blocks.append(nn.Dropout(config.dropout))
    return nn.Sequential(*blocks)


class Conv2dAutoEncoder(AutoEncoder):
    """
    Symmetric Conv2D autoencoder with one internal image channel.
    
    Input surfaces retain the public shape ``(batch, N_z, N_k)``. The model
    inserts one feature-channel axis internally so convolutions learn local
    structure jointly across the uniform redshift and log-wavenumber grids.
    """
    
    def __init__(self, config: object, input_shape: tuple[int, int]) -> None:
        """
        Construct the encoder, bottleneck, and exactly invertible decoder.
        
        Arguments:
            config (object):
                Checked Conv2D architecture configuration.
            input_shape (tuple[int, int]):
                Redshift and wavenumber dimensions of one surface.
        """
        super().__init__(config, input_shape)
        self.encoder_convolution, self.encoder_sizes = _build_encoder_convolution(
            config,
            self.input_shape,
        )
        final_size = self.encoder_sizes[-1]
        flattened_features = (
            config.encoder_channels[-1] * final_size[0] * final_size[1]
        )
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
            self.encoder_sizes,
        )
        self.encoded_shape = (
            config.encoder_channels[-1],
            *final_size,
        )
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """
        Initialize trainable layers and a near-zero decoder output map.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        output_layer = self.decoder_convolution[-1]
        if not isinstance(output_layer, nn.ConvTranspose2d):
            raise RuntimeError("The decoder output layer must be ConvTranspose2d.")
        nn.init.normal_(output_layer.weight, mean=0.0, std=1.0e-3)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)
    
    def encode(self, surfaces: torch.Tensor) -> torch.Tensor:
        """
        Encode normalized surfaces into latent coordinates.
        
        Arguments:
            surfaces (torch.Tensor):
                Normalized surfaces with shape (batch, N_z, N_k).
        
        Returns:
            latent (torch.Tensor):
                Latent coordinates with shape (batch, latent_dim).
        """
        self._check_input(surfaces)
        features = self.encoder_convolution(surfaces.unsqueeze(1))
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
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must have shape (batch, {self.latent_dim}), "
                f"found {tuple(latent.shape)}."
            )
        features = self.decoder_dense(latent)
        features = features.reshape(latent.shape[0], *self.encoded_shape)
        reconstructed = self.decoder_convolution(features).squeeze(1)
        if tuple(reconstructed.shape[1:]) != self.input_shape:
            raise RuntimeError(
                f"Decoder produced {tuple(reconstructed.shape[1:])}, "
                f"expected {self.input_shape}."
            )
        return reconstructed
    
    def _check_input(self, surfaces: torch.Tensor) -> None:
        """
        Check a public surface batch against the configured shape.
        
        Arguments:
            surfaces (torch.Tensor):
                Candidate input batch.
        """
        if surfaces.ndim != 3 or tuple(surfaces.shape[1:]) != self.input_shape:
            raise ValueError(
                f"surfaces must have shape (batch, {self.input_shape[0]}, "
                f"{self.input_shape[1]}), found {tuple(surfaces.shape)}."
            )
    
    def architecture_summary(self) -> dict[str, object]:
        """
        Return a JSON-safe summary of the resolved architecture.
        
        Returns:
            summary (dict[str, object]):
                Surface sizes, bottleneck, and parameter-count metadata.
        """
        return {
            "name": "Conv2D",
            "input_shape": list(self.input_shape),
            "internal_input_shape": [1, *self.input_shape],
            "encoded_shape": list(self.encoded_shape),
            "encoder_sizes": [list(size) for size in self.encoder_sizes],
            "latent_dim": self.latent_dim,
            "number_of_parameters": self.number_of_parameters,
            "skip_connections": False,
        }
