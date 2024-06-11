from __future__ import annotations

import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = ["HANDLED_FUNCTIONS", "HANDLED_UFUNCS", "numpy_wrappers", "wrap_numpy"]
