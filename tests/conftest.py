from __future__ import annotations

import pytest

from auto_uncertainties import Settings, configure, get_settings


@pytest.fixture(autouse=True)
def restore_settings():
    original = get_settings().model_dump()
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    configure(**defaults)
    yield
    configure(**original)
