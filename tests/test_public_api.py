from __future__ import annotations

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("module_name", "exports"),
    (
        (
            "auto_uncertainties",
            {
                "DowncastError",
                "DowncastWarning",
                "EqualityError",
                "EqualityWarning",
                "NegativeStdDevError",
                "Settings",
                "UType",
                "Uncertainty",
                "configure",
                "get_settings",
                "nominal_values",
                "propagate_uncertainties",
                "std_devs",
            },
        ),
        (
            "auto_uncertainties.uncertainty",
            {"UType", "Uncertainty", "nominal_values", "std_devs"},
        ),
        ("auto_uncertainties.numpy", set()),
    ),
)
def test_public_exports(module_name, exports):
    module = import_module(module_name)
    assert set(module.__all__) == exports
    for name in exports:
        assert getattr(module, name) is not None
