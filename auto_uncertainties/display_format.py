from __future__ import annotations

import decimal
import math

import numpy as np
import numpy.typing as npt

from auto_uncertainties.config import get_settings

__all__ = [
    "ScalarDisplay",
    "UncertaintyDisplay",
    "VectorDisplay",
]


class UncertaintyDisplay:
    default_format: str = ""
    _nom: npt.NDArray[np.floating] | np.floating | float
    _err: npt.NDArray[np.floating] | np.floating | float

    def _repr_html_(self):
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
        vformat, eformat = pdg_round(val_, err_)
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} {chr(0x00B1)} {eformat}"

    def _repr_latex_(self):
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
        vformat, eformat = pdg_round(val_, err_)
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} \\pm {eformat}"

    def __str__(self) -> str:
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
        vformat, eformat = pdg_round(val_, err_)
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} +/- {eformat}"

    def __format__(self, fmt):
        val_ = self._nom
        err_ = self._err

        if isinstance(val_, np.ndarray) and isinstance(err_, np.ndarray):
            s = []
            for v, e in zip(val_.ravel(), err_.ravel(), strict=False):
                vformat, eformat = pdg_round(v, e, format_spec=fmt, return_zero=True)
                s.append(f"{vformat} +/- {eformat}")

            return "[" + ", ".join(s) + "]"

        # Scalar uncertainty
        vformat, eformat = pdg_round(val_, err_, format_spec=fmt, return_zero=True)
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} +/- {eformat}"

    def __repr__(self) -> str:
        return str(self)


# Kept for compatibility.
ScalarDisplay = UncertaintyDisplay
VectorDisplay = UncertaintyDisplay


# From https://github.com/lmfit/uncertainties/blob/master/uncertainties/core.py
def first_digit(value):
    """
    Return the first digit position of the given value, as an integer.

    0 is the digit just before the decimal point. Digits to the right
    of the decimal point have a negative position.

    Return 0 for a null value.
    """
    try:
        return math.floor(math.log10(abs(value)))
    except ValueError:  # Case of value == 0
        return 0


# From https://github.com/lmfit/uncertainties/blob/master/uncertainties/core.py
def PDG_precision(std_dev):
    """
    Return the number of significant digits to be used for the given
    standard deviation, according to the rounding rules of the
    Particle Data Group (2010)
    (http://pdg.lbl.gov/2010/reviews/rpp2010-rev-rpp-intro.pdf).

    Also returns the effective standard deviation to be used for
    display.
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
    value, uncertainty, format_spec="g", *, return_zero: bool = False
) -> tuple[str, str]:
    """
    Format a value with uncertainty according to PDG rounding rules.

    :param value: The central value
    :param uncertainty: The uncertainty of the value
    :param format_spec:
    :param return_zero:

    :return: The formatted value with uncertainty.
    """

    if not np.isfinite(value) or (
        uncertainty is not None and not np.isfinite(uncertainty)
    ):
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

            # Both components end at the same decimal place. A larger context
            # also supports finite values with very large or small uncertainties.
            with decimal.localcontext() as context:
                context.prec = max(28, value_dec.adjusted() - place + 2)
                context.rounding = decimal.ROUND_HALF_EVEN
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
