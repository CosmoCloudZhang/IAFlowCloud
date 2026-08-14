"""
Neural-network implementations used by IAFlow training machinery.
"""

from .conv1d_autoencoder import Conv1dAutoEncoder

__all__ = ["Conv1dAutoEncoder", "build_autoencoder"]


def build_autoencoder(
    config: object,
    input_shape: tuple[int, int],
) -> Conv1dAutoEncoder:
    """
    Construct the configured autoencoder through one extension point.
    
    Arguments:
        config (object):
            Checked architecture configuration.
        input_shape (tuple[int, int]):
            Channel and data dimensions of one input surface.
    
    Returns:
        model (Conv1dAutoEncoder):
            Autoencoder selected by config.name.
    """
    builders = {"Conv1D": Conv1dAutoEncoder}
    try:
        builder = builders[config.name]
    except (AttributeError, KeyError) as error:
        name = getattr(config, "name", None)
        raise ValueError(f"Unsupported model architecture: {name!r}.") from error
    return builder(config, input_shape)
