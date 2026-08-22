"""
Shared interface for IA surface compressors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

__all__ = ["AutoEncoder"]


class AutoEncoder(nn.Module, ABC):
    """
    Define the model contract used by training, evaluation, and export.
    """
    
    def __init__(self, config: object, input_shape: tuple[int, int]) -> None:
        """
        Store the resolved architecture and two-dimensional surface shape.
        
        Arguments:
            config (object):
                Checked architecture configuration containing latent_dim.
            input_shape (tuple[int, int]):
                Redshift and wavenumber dimensions of one surface.
        """
        super().__init__()
        self.config = config
        self.input_shape = tuple(int(value) for value in input_shape)
        if len(self.input_shape) != 2 or min(self.input_shape) <= 0:
            raise ValueError("input_shape must contain two positive dimensions.")
    
    @abstractmethod
    def encode(self, surfaces: torch.Tensor) -> torch.Tensor:
        """
        Encode normalized surfaces into latent coordinates.
        
        Arguments:
            surfaces (torch.Tensor):
                Normalized batch with shape (batch, *input_shape).
        
        Returns:
            latent (torch.Tensor):
                Coordinates with shape (batch, latent_dim).
        """
    
    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent coordinates into normalized surfaces.
        
        Arguments:
            latent (torch.Tensor):
                Coordinates with shape (batch, latent_dim).
        
        Returns:
            surfaces (torch.Tensor):
                Reconstructions with shape (batch, *input_shape).
        """
    
    def forward(self, surfaces: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct normalized surfaces through the latent bottleneck.
        
        Arguments:
            surfaces (torch.Tensor):
                Normalized batch with shape (batch, *input_shape).
        
        Returns:
            reconstructed (torch.Tensor):
                Normalized reconstruction with the same shape.
        """
        return self.decode(self.encode(surfaces))
    
    @property
    def latent_dim(self) -> int:
        """
        Return the resolved bottleneck dimension.
        
        Returns:
            dimension (int):
                Number of latent coordinates per surface.
        """
        return int(self.config.latent_dim)
    
    @property
    def number_of_parameters(self) -> int:
        """
        Return the total number of scalar model parameters.
        
        Returns:
            count (int):
                Number of trainable and non-trainable scalar parameters.
        """
        return sum(parameter.numel() for parameter in self.parameters())
    
    @abstractmethod
    def architecture_summary(self) -> dict[str, object]:
        """
        Return a JSON-safe summary of the resolved architecture.
        
        Returns:
            summary (dict[str, object]):
                Shape, bottleneck, and parameter-count metadata.
        """
