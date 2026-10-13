from __future__ import annotations

import decimal
import math
from typing import Generic, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt

from auto_uncertainties.config import get_settings

__all__ = [
    "ScalarDisplay",
    "UncertaintyDisplay",
    "VectorDisplay",
]


_DisplayT = TypeVar("_DisplayT", float, np.floating, npt.NDArray[np.floating])


class UncertaintyDisplay(Generic[_DisplayT]):
    """Provide text, HTML, and LaTeX representations of nominal values and errors."""

    default_format: str = ""
    _nom: _DisplayT
    _err: _DisplayT

    def _repr_html_(self) -> str:
        """
        Format the uncertainty for HTML display.

        :return: The formatted value and error, using a table for arrays.
        """
        val_ = self._nom
        err_ = self._err

        # Vector uncertainty
        if isinstance(val_, np.ndarray) and isinstance(err_, np.ndarray):
            header = "<table><tbody>"
            footer = "</tbody></table>"
            vformatted = []
            eformatted = []
            for v, e in zip(val_.ravel(), err_.ravel(), strict=False):
                vformat, eformat = pdg_round(v, e, return_zero=True)
                vformatted.append(vformat)
                eformatted.append(eformat)
            val = f"<tr><th>Magnitude</th><td style='text-align:left;'><pre>{', '.join(vformatted)}</pre></td></tr>"
            err = f"<tr><th>Error</th><td style='text-align:left;'><pre>{', '.join(eformatted)}</pre></td></tr>"
            return header + val + err + footer

        # Scalar uncertainty
        vformat, eformat = pdg_round(
            cast("float | np.floating", val_), cast("float | np.floating", err_)
        )
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} {chr(0x00B1)} {eformat}"

    def _repr_latex_(self) -> str:
        """
        Format the uncertainty for LaTeX display.

        :return: The formatted value and error with LaTeX uncertainty notation.
        """
        val_ = self._nom
        err_ = self._err

        # Vector uncertainty
        if isinstance(val_, np.ndarray) and isinstance(err_, np.ndarray):
            s = []
            for v, e in zip(val_.ravel(), err_.ravel(), strict=False):
                vformat, eformat = pdg_round(v, e, return_zero=True)
                s.append(f"{vformat} \\pm {eformat}")
            s = ", ".join(s) + "~"
            header = "$"
            footer = "$"
            return header + s + footer

        # Scalar uncertainty
        vformat, eformat = pdg_round(
            cast("float | np.floating", val_), cast("float | np.floating", err_)
        )
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} \\pm {eformat}"

    def __str__(self) -> str:
        """
        Format the uncertainty for text display.

        :return: The nominal value and error formatted using the current rounding
            settings.
        """
        val_ = self._nom
        err_ = self._err

        # Vector uncertainty
        if isinstance(val_, np.ndarray) and isinstance(err_, np.ndarray):
            s = []
            for v, e in zip(val_.ravel(), err_.ravel(), strict=False):
                vformat, eformat = pdg_round(v, e, return_zero=True)
                s.append(f"{vformat} +/- {eformat}")
            return "[" + ", ".join(s) + "]"

        # Scalar uncertainty
        vformat, eformat = pdg_round(
            cast("float | np.floating", val_), cast("float | np.floating", err_)
        )
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} +/- {eformat}"

    def __format__(self, fmt: str) -> str:
        """
        Format nominal values and errors with a shared format specification.

        :param fmt: Python numeric format specification applied after any configured PDG
            rounding.
        :return: The formatted uncertainty, including a zero error when applicable.
        """
        val_ = self._nom
        err_ = self._err

        if isinstance(val_, np.ndarray) and isinstance(err_, np.ndarray):
            s = []
            for v, e in zip(val_.ravel(), err_.ravel(), strict=False):
                vformat, eformat = pdg_round(v, e, format_spec=fmt, return_zero=True)
                s.append(f"{vformat} +/- {eformat}")

            return "[" + ", ".join(s) + "]"

        # Scalar uncertainty
        vformat, eformat = pdg_round(
            cast("float | np.floating", val_),
            cast("float | np.floating", err_),
            format_spec=fmt,
            return_zero=True,
        )
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} +/- {eformat}"

    def __repr__(self) -> str:
        """
        Return the text representation of the uncertainty.

        :return: The same display text as :meth:`__str__`.
        """
        return str(self)


# Kept for compatibility.
ScalarDisplay = UncertaintyDisplay
VectorDisplay = UncertaintyDisplay


# From https://github.com/lmfit/uncertainties/blob/master/uncertainties/core.py
def first_digit(value: float | np.floating) -> int:
    """
    Find the position of the first significant digit.

    Position zero is immediately before the decimal point; positions to its
    right are negative.

    :param value: Value whose magnitude determines the digit position.
    :return: The first significant digit position, or zero when the value is zero.
    """
    try:
        return math.floor(math.log10(abs(value)))
    except ValueError:  # Case of value == 0
        return 0


# From https://github.com/lmfit/uncertainties/blob/master/uncertainties/core.py
@overload
def PDG_precision(std_dev: decimal.Decimal) -> tuple[int, decimal.Decimal]: ...
@overload
def PDG_precision(std_dev: float | np.floating) -> tuple[int, float | np.floating]: ...
def PDG_precision(
    std_dev: decimal.Decimal | float | np.floating,
) -> tuple[int, decimal.Decimal | float | np.floating]:
    """
    Choose display precision using the Particle Data Group rounding rules.

    The rules are described in the `Particle Data Group review (2010)
    <http://pdg.lbl.gov/2010/reviews/rpp2010-rev-rpp-intro.pdf>`_.
    Errors with leading digits from 950 through 999 carry to the next decade.

    :param std_dev: Positive, finite standard deviation to format.
    :return: The number of significant digits and the effective standard deviation to
        display.
    :raises ValueError: The standard deviation is nonpositive or nonfinite.
    """
    uncertainty = decimal.Decimal(str(std_dev))
    if not uncertainty.is_finite() or uncertainty <= 0:
        msg = "Standard deviation must be positive and finite"
        raise ValueError(msg)

    exponent = uncertainty.adjusted()
    leading = uncertainty.as_tuple().digits[:3]
    digits = int("".join(str(digit) for digit in leading).ljust(3, "0"))
    if digits <= 354:
        return 2, std_dev
    if digits <= 949:
        return 1, std_dev

    # Carry to the next decade, retaining two significant digits.
    effective = decimal.Decimal((0, (1,), exponent + 1))
    return 2, effective if isinstance(std_dev, decimal.Decimal) else float(effective)


def pdg_round(
    value: float | np.floating,
    uncertainty: float | np.floating | None,
    format_spec: str = "g",
    *,
    return_zero: bool = False,
) -> tuple[str, str]:
    """
    Format a value and its uncertainty using the current display settings.

    PDG rounding is enabled by default through ``round_on_display``. When enabled,
    both components are rounded to the decimal place determined by the error.
    Nonfinite components are formatted directly without PDG rounding.

    :param value: Central value to display.
    :param uncertainty: Standard deviation; ``None`` omits the error when PDG rounding
        is enabled.
    :param format_spec: Python numeric format specification applied to both components.
    :param return_zero: Display ``"0"`` instead of an empty error for missing or
        nonpositive errors when PDG rounding is enabled.
    :return: The formatted nominal value and formatted error, in that order.
    """
    if not np.isfinite(value) or (uncertainty is not None and not np.isfinite(uncertainty)):
        return (
            f"{value:{format_spec}}",
            f"{uncertainty:{format_spec}}" if uncertainty is not None else "",
        )

    if get_settings().round_on_display:
        if uncertainty is not None and uncertainty > 0:
            # String conversion supports NumPy scalars and avoids introducing
            # binary float artifacts at PDG thresholds and rounding ties.
            value_dec = decimal.Decimal(str(value))
            uncertainty_dec = decimal.Decimal(str(uncertainty))
            significant_digits, pdg_unc = PDG_precision(uncertainty_dec)
            place = pdg_unc.adjusted() - significant_digits + 1
            quantum = decimal.Decimal((0, (1,), place))

            # Isolate display rounding from the caller's precision, exponent
            # limits and traps. Both components end at the same decimal place.
            context = decimal.Context(
                prec=max(28, value_dec.adjusted() - place + 2),
                rounding=decimal.ROUND_HALF_EVEN,
                Emin=decimal.MIN_EMIN,
                Emax=decimal.MAX_EMAX,
                traps=[
                    decimal.InvalidOperation,
                    decimal.DivisionByZero,
                    decimal.Overflow,
                ],
            )
            with decimal.localcontext(context):
                rounded_value = value_dec.quantize(quantum)
                rounded_uncertainty = pdg_unc.quantize(quantum)

                return (
                    f"{rounded_value:{format_spec}}",
                    f"{rounded_uncertainty:{format_spec}}",
                )

        else:
            return f"{value:{format_spec}}", "0" if return_zero else ""
    else:
        return f"{value:{format_spec}}", f"{uncertainty:{format_spec}}"
