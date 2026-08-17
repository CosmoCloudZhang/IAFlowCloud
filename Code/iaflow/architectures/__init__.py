"""
Neural-network implementations used by IAFlow training machinery.
"""

from .base import AutoEncoder
from .conv1d_autoencoder import Conv1dAutoEncoder
from .conv2d_autoencoder import Conv2dAutoEncoder

__all__ = [
    "AutoEncoder",
    "Conv1dAutoEncoder",
    "Conv2dAutoEncoder",
    "build_autoencoder",
]


def build_autoencoder(
    config: object,
    input_shape: tuple[int, int],
) -> AutoEncoder:
    """
    Construct the configured surface autoencoder.
    
    Arguments:
        config (object):
            Checked architecture configuration.
        input_shape (tuple[int, int]):
            Redshift and wavenumber dimensions of one input surface.
    
    Returns:
        model (AutoEncoder):
            Autoencoder selected by config.name.
    """
    architectures = {
        "Conv1D": Conv1dAutoEncoder,
        "Conv2D": Conv2dAutoEncoder,
    }
    name = getattr(config, "name", None)
    try:
        architecture = architectures[name]
    except KeyError as error:
        raise ValueError(f"Unsupported model architecture: {name!r}.") from error
    return architecture(config, input_shape)
