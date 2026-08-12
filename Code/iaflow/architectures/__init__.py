"""Neural-network implementations used by IAFlow training machinery."""

from .conv1d_autoencoder import Conv1dAutoEncoder, validate_conv1d_config

__all__ = ["Conv1dAutoEncoder", "build_autoencoder", "validate_model_config"]


def validate_model_config(config: object) -> None:
    """Dispatch architecture-owned configuration validation."""
    validators = {"Conv1D": validate_conv1d_config}
    try:
        validator = validators[config.name]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"Unsupported model architecture: {getattr(config, 'name', None)!r}.") from error
    validator(config)


def build_autoencoder(config: object, input_shape: tuple[int, int]) -> Conv1dAutoEncoder:
    """Construct the configured autoencoder through one extension point."""
    builders = {"Conv1D": Conv1dAutoEncoder}
    try:
        builder = builders[config.name]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"Unsupported model architecture: {getattr(config, 'name', None)!r}.") from error
    return builder(config, input_shape)
