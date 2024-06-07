from __future__ import annotations

import decimal
import math

from numpy.typing import NDArray

ROUND_ON_DISPLAY = False

__all__ = ["set_display_rounding", "VectorDisplay", "ScalarDisplay"]


def set_display_rounding(val: bool):
    """Set the rounding on display to PDG recommendations."""
    global ROUND_ON_DISPLAY
    ROUND_ON_DISPLAY = val


class VectorDisplay:
    default_format: str = ""
    _nom: NDArray
    _err: NDArray

    def _repr_html_(self):
        val_ = self._nom
        err_ = self._err
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

    def _repr_latex_(self):
        val_ = self._nom
        err_ = self._err
        s = []
        for v, e in zip(val_.ravel(), err_.ravel(), strict=False):
            vformat, eformat = pdg_round(v, e, return_zero=True)
            s.append(f"{vformat} \\pm {eformat}")
        s = ", ".join(s) + "~"
        header = "$"
        footer = "$"
        return header + s + footer

    def __str__(self) -> str:
        val_ = self._nom
        err_ = self._err

        s = []
        for v, e in zip(val_.ravel(), err_.ravel(), strict=False):
            vformat, eformat = pdg_round(v, e, return_zero=True)
            s.append(f"{vformat} +/- {eformat}")
        return "[" + ", ".join(s) + "]"

    def __format__(self, fmt):
        val_ = self._nom
        err_ = self._err
        s = []
        for v, e in zip(val_.ravel(), err_.ravel(), strict=False):
            vformat, eformat = pdg_round(v, e, format_spec=fmt, return_zero=True)
            s.append(f"{vformat} +/- {eformat}")

        return "[" + ", ".join(s) + "]"

    def __repr__(self) -> str:
        return str(self)


class ScalarDisplay:
    default_format: str = ""
    _nom: float
    _err: float

    def _repr_html_(self):
        val_ = self._nom
        err_ = self._err
        vformat, eformat = pdg_round(val_, err_)
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} {chr(0x00B1)} {eformat}"

    def _repr_latex_(self):
        val_ = self._nom
        err_ = self._err
        vformat, eformat = pdg_round(val_, err_)
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} \\pm {eformat}"

    def __str__(self) -> str:
        val_ = self._nom
        err_ = self._err

        vformat, eformat = pdg_round(val_, err_)
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} +/- {eformat}"

    def __format__(self, fmt):
        val_ = self._nom
        err_ = self._err

        vformat, eformat = pdg_round(val_, err_)
        if eformat == "":
            return f"{vformat}"
        else:
            return f"{vformat} +/- {eformat}"

    def __repr__(self) -> str:
        return str(self)


# From https://github.com/lmfit/uncertainties/blob/master/uncertainties/core.py
def first_digit(value):
    """
    Return the first digit position of the given value, as an integer.

    0 is the digit just before the decimal point. Digits to the right
    of the decimal point have a negative position.

    Return 0 for a null value.
    """
    try:
        return int(math.floor(math.log10(abs(value))))
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

    exponent = first_digit(std_dev)

    # The first three digits are what matters: we get them as an
    # integer number in [100; 999).
    #
    # In order to prevent underflow or overflow when calculating
    # 10**exponent, the exponent is slightly modified first and a
    # factor to be applied after "removing" the new exponent is
    # defined.
    #
    # Furthermore, 10**(-exponent) is not used because the exponent
    # range for very small and very big floats is generally different.
    if exponent >= 0:
        # The -2 here means "take two additional digits":
        (exponent, factor) = (exponent - 2, 1)
    else:
        (exponent, factor) = (exponent + 1, 1000)
    digits = int(std_dev / 10.0**exponent * factor)  # int rounds towards zero

    # Rules:
    if digits <= 354:
        return (2, std_dev)
    elif digits <= 949:
        return (1, std_dev)
    else:
        # The parentheses matter, for very small or very large
        # std_dev:
        return (2, 10.0**exponent * (1000 / factor))


def pdg_round(
    value, uncertainty, format_spec="g", *, return_zero: bool = False
) -> tuple[str, str]:
    """
    Format a value with uncertainty according to PDG rounding rules.

    Args:
        value (float): The central value.
        uncertainty (float): The uncertainty of the value.

    Returns:
        str: The formatted value with uncertainty.
    """
    if ROUND_ON_DISPLAY:
        if uncertainty is not None and uncertainty > 0:
            _, pdg_unc = PDG_precision(uncertainty)
            # Determine the order of magnitude of the uncertainty
            order_of_magnitude = 10 ** (int(math.floor(math.log10(pdg_unc))) - 1)

            # Round the uncertainty based on how many digits we want to keep
            rounded_uncertainty = (
                round(pdg_unc / order_of_magnitude) * order_of_magnitude
            )
            # Round the central value according to the rounded uncertainty
            unc_impled_digits_to_keep = -int(
                math.floor(math.log10(rounded_uncertainty))
            )
            if value != 0:
                # Keep at least two digits for the central value, even if the uncertainty is much larger
                digits = max(
                    unc_impled_digits_to_keep,
                    -int(math.floor(math.log10(abs(value)))) + 1,
                )
            else:
                digits = unc_impled_digits_to_keep

            # Use decimal to keep trailing zeros
            rounded_value_dec = round(decimal.Decimal(value), digits)
            rounded_unc_dec = round(
                decimal.Decimal(rounded_uncertainty),
                unc_impled_digits_to_keep + 1,
            )
            return (
                f"{rounded_value_dec:{format_spec}}",
                f"{rounded_unc_dec:{format_spec}}",
            )

        else:
            return f"{value:{format_spec}}", "0" if return_zero else ""
    else:
        return f"{value:{format_spec}}", f"{uncertainty:{format_spec}}"
