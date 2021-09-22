# -*- coding: utf-8 -*-
class NegativeStdDevError(Exception):
    """An exception for when the standard deviation is negative"""

    pass


from .uncertainty import Uncertainty
