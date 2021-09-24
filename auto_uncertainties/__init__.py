# -*- coding: utf-8 -*-
class NegativeStdDevError(Exception):
    """An exception for when the standard deviation is negative"""

    pass


class NumpyDowncastWarning(RuntimeWarning):
    """An exception for when an uncertainties array is downcast to a numpy array"""

    pass


from .uncertainty import Uncertainty

try:
    from .pandas_compat import UncertaintyArray
except ImportError:
    pass
