# -*- coding: utf-8 -*-
import warnings
from functools import wraps
from . import NumpyDowncastWarning


def ignore_runtime_warnings(f):
    """
    A decorator to ignore runtime warnings
    Parameters
    ----------
    f: function
        The wrapped function

    Returns
    -------
    wrapped_function: function
        The wrapped function
    """

    @wraps(f)
    def runtime_warn_inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=RuntimeWarning)
            response = f(*args, **kwargs)
        return response

    return runtime_warn_inner


def ignore_numpy_downcast_warnings(f):
    """
    A decorator to ignore NumpyDowncastWarning
    Parameters
    ----------
    f: function
        The wrapped function

    Returns
    -------
    wrapped_function: function
        The wrapped function
    """

    @wraps(f)
    def user_warn_inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=NumpyDowncastWarning)
            response = f(*args, **kwargs)
        return response

    return user_warn_inner


def is_iterable(y):
    try:
        iter(y)
    except TypeError:
        return False
    return True


def has_length(y):
    try:
        len(y)
    except TypeError:
        return False
    return True


def is_np_duck_array(cls):
    """Check if object is a numpy array-like, but not a Uncertainty

    Parameters
    ----------
    cls : class

    Returns
    -------
    bool
    """
    try:
        import numpy as np
    except ImportError:
        return False

    return issubclass(cls, np.ndarray) or (
        not hasattr(cls, "_nom")
        and not hasattr(cls, "_err")
        and hasattr(cls, "__array_function__")
        and hasattr(cls, "ndim")
        and hasattr(cls, "dtype")
    )


class Display(object):
    default_format: str = ""

    def _repr_html_(self):
        if hasattr(self._nom, "units"):
            val_ = self._nom.m
            err_ = self._err.m
            u = self._nom.units
        else:
            val_ = self._nom
            err_ = self._err
            u = None
        if is_np_duck_array(type(self._nom)):
            header = "<table><tbody>"
            footer = "</tbody></table>"
            val = f"<tr><th>Magnitude</th><td style='text-align:left;'><pre>{val_}</pre></td></tr>"
            err = f"<tr><th>Error</th><td style='text-align:left;'><pre>{err_}</pre></td></tr>"
            if u is None:
                units = ""
            else:
                units = (
                    f"<tr><th>Units</th><td style='text-align:left;'>{u._repr_html_()}</td></tr>"
                )
            return header + val + err + units + footer
        else:
            val = f"{val_}"
            err = f"{err_}"
            if u is None:
                units = ""
            else:
                units = f"{u}"
            return f"{val} {chr(0x00B1)} {err} {units}"

    def _repr_latex_(self):
        if hasattr(self._nom, "units"):
            val_ = self._nom.m
            err_ = self._err.m
            u = self._nom.units
        else:
            val_ = self._nom
            err_ = self._err
            u = None
        if is_np_duck_array(type(self._nom)):
            s = ", ".join([f"{v} \\pm {e}" for v, e in zip(val_.ravel(), err_.ravel())]) + "~"
            header = "$"
            footer = "$"
            if u is None:
                units = ""
            else:
                units = u._repr_latex_()
            return header + s + units + footer
        else:
            val = f"{val_}"
            err = f"{err_}"
            if u is None:
                units = ""
            else:
                units = u._repr_latex_()
            return f"${val} \\pm {err} {units}$"

    def __str__(self) -> str:
        if hasattr(self._nom, "units"):
            val_ = self._nom.m
            err_ = self._err.m
            u = self._nom.units
        else:
            val_ = self._nom
            err_ = self._err
            u = None

        if u is None:
            units = ""
        else:
            units = f" {u}"

        if self._nom is not None:
            if self._err is not None:
                if is_np_duck_array(type(self._nom)):
                    return (
                        "["
                        + ", ".join([f"{v} +/- {e}" for v, e in zip(val_.ravel(), err_.ravel())])
                        + "]"
                        + units
                    )
                else:
                    return f"{self._nom} +/- {self._err}" + units
            else:
                return f"{self._nom}" + units

    def __format__(self, fmt):
        return f"{self.value:{fmt}} +/- {self.error:{fmt}}"

    def __repr__(self) -> str:
        return str(self)
