"""
Neural-network implementations used by IAFlow training machinery.
"""

from .conv1d_autoencoder import Conv1dAutoEncoder, validate_conv1d_config

__all__ = ["Conv1dAutoEncoder", "build_autoencoder", "validate_model_config"]


def validate_model_config(
    config: object,
) -> None:
    """
    Dispatch configuration validation to the selected architecture.
    
    Arguments:
        config (object):
            Mutable model-configuration section containing an architecture name.
    """
    validators = {"Conv1D": validate_conv1d_config}
    try:
        validator = validators[config.name]
    except (AttributeError, KeyError) as error:
        name = getattr(config, "name", None)
        raise ValueError(f"Unsupported model architecture: {name!r}.") from error
    validator(config)


def build_autoencoder(
    config: object,
    input_shape: tuple[int, int],
) -> Conv1dAutoEncoder:
    """
    Construct the configured autoencoder through one extension point.
    
    Arguments:
        config (object):
            Validated architecture configuration.
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
