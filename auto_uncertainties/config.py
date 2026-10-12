"""Validated runtime configuration, initialized from AUTO_UNCERTAINTIES_* variables."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "configure", "get_settings"]


class Settings(BaseSettings):
    """Library settings. Environment variables are read when instantiated."""

    model_config = SettingsConfigDict(
        env_prefix="AUTO_UNCERTAINTIES_", extra="forbid", validate_assignment=True
    )

    round_on_display: bool = False
    error_on_downcast: bool = False
    error_on_equality: bool = False
    compare_rtol: float = Field(default=1e-9, ge=0, allow_inf_nan=False)
    error_on_numpy_kwargs: bool = True
    warn_on_numpy_kwargs: bool = True
    nonfinite_error_policy: Literal["zero", "propagate"] = "zero"


_settings = Settings()


def get_settings() -> Settings:
    """Return the shared settings instance."""
    return _settings


def configure(**changes: Any) -> None:
    """Validate all changes before updating the shared settings.

    Changes affect subsequent operations; existing uncertainty objects are not
    rewritten. Later environment changes do not affect the shared settings.
    """
    unknown = changes.keys() - Settings.model_fields.keys()
    if unknown:
        msg = f"Unknown settings: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    candidate = Settings(**(_settings.model_dump() | changes))
    for name in changes:
        setattr(_settings, name, getattr(candidate, name))
