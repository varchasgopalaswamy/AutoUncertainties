from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
from pydantic import ValidationError
import pytest

from auto_uncertainties import (
    EqualityWarning,
    Settings,
    Uncertainty,
    configure,
    get_settings,
)


def test_environment_initialization():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AUTO_UNCERTAINTIES_")
    }
    env.update(
        AUTO_UNCERTAINTIES_NONFINITE_ERROR_POLICY="propagate",
        AUTO_UNCERTAINTIES_ROUND_ON_DISPLAY="false",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from auto_uncertainties import get_settings; s = get_settings(); print(s.nonfinite_error_policy, s.round_on_display)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "propagate False"


def test_display_rounding_defaults_to_pdg():
    assert get_settings().round_on_display is True
    value = Uncertainty(1.234567, 0.0123)
    assert str(value) == "1.235 +/- 0.012"
    configure(round_on_display=False)
    assert str(value) == "1.23457 +/- 0.0123"
    assert value.value == 1.234567
    assert value.error == 0.0123


def test_environment_validation(monkeypatch):
    monkeypatch.setenv("AUTO_UNCERTAINTIES_NONFINITE_ERROR_POLICY", "invalid")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    (
        ("round_on_display", "false", False),
        ("error_on_downcast", "1", True),
        ("error_on_equality", "true", True),
        ("compare_rtol", "0.01", 0.01),
        ("nonfinite_error_policy", "propagate", "propagate"),
    ),
)
def test_environment_fields(monkeypatch, field, value, expected):
    monkeypatch.setenv(f"AUTO_UNCERTAINTIES_{field.upper()}", value)
    assert getattr(Settings(), field) == expected


@pytest.mark.parametrize(
    "changes",
    (
        {"compare_rtol": -1},
        {"compare_rtol": np.nan},
        {"compare_rtol": np.inf},
        {"nonfinite_error_policy": "invalid"},
        {"unknown": True},
        {"_env_file": ".env"},
        {"error_on_downcast": "invalid"},
    ),
)
def test_invalid_updates_are_atomic(changes):
    original = get_settings().model_dump()
    with pytest.raises(ValueError, match=r"Unknown settings:|validation error"):
        configure(round_on_display=False, **changes)
    assert get_settings().model_dump() == original


def test_runtime_configuration_does_not_reload_environment(monkeypatch):
    settings = get_settings()
    monkeypatch.setenv("AUTO_UNCERTAINTIES_COMPARE_RTOL", "0.5")
    configure(round_on_display=False)
    assert get_settings() is settings
    assert settings.compare_rtol == 1e-9
    with pytest.raises(ValidationError):
        settings.compare_rtol = -1


def test_comparison_tolerance():
    u = Uncertainty(1.0)
    assert (u == 1.01) is False
    configure(compare_rtol=0.02)
    with pytest.warns(EqualityWarning):
        assert u == 1.01


@pytest.mark.parametrize("name", ("error_on_numpy_kwargs", "warn_on_numpy_kwargs"))
def test_removed_numpy_keyword_settings_are_rejected(name):
    with pytest.raises(ValueError, match="Unknown settings"):
        configure(**{name: True})
