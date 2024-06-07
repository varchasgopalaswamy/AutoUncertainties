from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from pandas.api.extensions import register_extension_dtype
from pandas.core.dtypes.base import ExtensionDtype

from auto_uncertainties.uncertainty import ScalarUncertainty, Uncertainty

if TYPE_CHECKING:
    from pandas._typing import type_t

    from .unc_array import UncertaintyArray

__all__ = ["UncertaintyDtype"]


@register_extension_dtype
class UncertaintyDtype(ExtensionDtype):
    type = Uncertainty
    name = "Uncertainty"
    _match = re.compile(r"^[U|u]ncertainty(\[([A-Za-z0-9]+)\])?$")
    _metadata = {}  # noqa: RUF012

    def __init__(self, dtype: np.dtype | str):
        self.value_dtype = np.dtype(dtype).name

    @property
    def na_value(self):
        return ScalarUncertainty(np.nan, 0)

    def __repr__(self) -> str:
        return f"Uncertainty[{self.value_dtype}]"

    @classmethod
    def construct_array_type(cls) -> type_t[UncertaintyArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from .unc_array import UncertaintyArray

        return UncertaintyArray

    @classmethod
    def construct_from_string(cls, string: str):
        r"""
        Construct this type from a string.

        This is useful mainly for data types that accept parameters.
        For example, a period dtype accepts a frequency parameter that
        can be set as ``period[H]`` (where H means hourly frequency).

        By default, in the abstract class, just the name of the type is
        expected. But subclasses can overwrite this method to accept
        parameters.

        Parameters
        ----------
        string : str
            The name of the type, for example ``category``.

        Returns
        -------
        ExtensionDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a class cannot be constructed from this 'string'.

        Examples
        --------
        For extension dtypes with arguments the following may be an
        adequate implementation.

        >>> import re
        >>> @classmethod
        ... def construct_from_string(cls, string):
        ...     pattern = re.compile(r"^my_type\[(?P<arg_name>.+)\]$")
        ...     match = pattern.match(string)
        ...     if match:
        ...         return cls(**match.groupdict())
        ...     else:
        ...         raise TypeError(
        ...             f"Cannot construct a '{cls.__name__}' from '{string}'"
        ...         )
        """
        if not isinstance(string, str):
            msg = f"'construct_from_string' expects a string, got {type(string)}"
            raise TypeError(msg)
        # error: Non-overlapping equality check (left operand type: "str", right
        #  operand type: "Callable[[ExtensionDtype], str]")  [comparison-overlap]
        assert isinstance(cls.name, str), (cls, type(cls.name))

        match = cls._match.match(string)
        if match is None:
            msg = f"Cannot construct a '{UncertaintyDtype.__name__}' from '{string}'"
            raise TypeError(msg)
        if match.group(1) is None:
            return cls("float64")
        return cls(match.group(1))

    @property
    def _is_numeric(self) -> bool:
        return True

    def __eq__(self, other) -> bool:
        """
        Check whether 'other' is equal to self.

        By default, 'other' is considered equal if either

        * it's a string matching 'self.name'.
        * it's an instance of this type and all of the attributes
          in ``self._metadata`` are equal between `self` and `other`.

        Parameters
        ----------
        other : Any

        Returns
        -------
        bool
        """
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False
        if isinstance(other, type(self)):
            return all(
                getattr(self, attr) == getattr(other, attr) for attr in self._metadata
            )
        return False

    def __hash__(self):
        # make myself hashable
        return hash(str(self))
