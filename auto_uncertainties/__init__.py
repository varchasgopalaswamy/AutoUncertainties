# -*- coding: utf-8 -*-
class NegativeStdDevError(Exception):
    """An exception for when the standard deviation is negative"""

    pass


from .uncertainty import Uncertainty

try:
    from .pandas_compat import UncertaintyArray
except ImportError:
    pass
