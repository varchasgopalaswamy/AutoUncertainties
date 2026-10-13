"""Validated runtime configuration, initialized from AUTO_UNCERTAINTIES_* variables."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "configure", "get_settings"]


class Settings(BaseSettings):
    """
    Configure library behavior from keyword arguments and environment variables.

    Environment variables use the ``AUTO_UNCERTAINTIES_`` prefix and are read
    when the settings are instantiated. Explicit arguments take precedence.

    :param round_on_display: Enable PDG rounding when displaying uncertainties.
    :param error_on_downcast: Raise an error instead of warning when a conversion
        discards uncertainties.
    :param error_on_equality: Raise an error instead of warning when equal nominal
        values have different errors.
    :param compare_rtol: Nonnegative relative tolerance for scalar equality comparisons.
    :param error_on_numpy_kwargs: Compatibility setting; argument validation is always
        enabled.
    :param warn_on_numpy_kwargs: Compatibility setting; unsupported arguments are
        rejected.
    :param nonfinite_error_policy: Use ``"zero"`` to replace nonfinite errors with zero,
        or ``"propagate"`` to preserve infinite errors and propagate NaNs.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUTO_UNCERTAINTIES_", extra="forbid", validate_assignment=True
    )

    round_on_display: bool = True
    error_on_downcast: bool = False
    error_on_equality: bool = False
    compare_rtol: float = Field(default=1e-9, ge=0, allow_inf_nan=False)
    error_on_numpy_kwargs: bool = True
    warn_on_numpy_kwargs: bool = True
    nonfinite_error_policy: Literal["zero", "propagate"] = "zero"


_settings = Settings()


def get_settings() -> Settings:
    """
    Return the shared runtime configuration.

    :return: The settings used by subsequent library operations.
    """
    return _settings


def configure(**changes: Any) -> None:
    """
    Validate all changes before updating the shared settings.

    Changes affect subsequent operations; existing uncertainty objects are not
    rewritten. Later environment changes do not affect the shared settings.

    :param changes: Setting names and their replacement values.
    :return: ``None``.
    :raises ValueError: A setting name is unknown or a value fails validation.
    """
    unknown = changes.keys() - Settings.model_fields.keys()
    if unknown:
        msg = f"Unknown settings: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    candidate = Settings(**(_settings.model_dump() | changes))
    for name in changes:
        setattr(_settings, name, getattr(candidate, name))
