# Based heavily on the implementation of pint's numpy array function wrapping

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import wraps
import inspect
from typing import Any, Literal, TypeGuard, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from auto_uncertainties._propagation import _UncertaintyLike, propagate
from auto_uncertainties.util import has_length, is_iterable
lazy from auto_uncertainties.uncertainty.uncertainty_containers import (
    Uncertainty,
    _power_uncertainty,
)

_F = TypeVar("_F", bound=Callable[..., Any])
HANDLED_UFUNCS: dict[str, Callable[..., Any]] = {}
HANDLED_FUNCTIONS: dict[str, Callable[..., Any]] = {}

__all__ = ["HANDLED_FUNCTIONS", "HANDLED_UFUNCS", "wrap_numpy"]


def _is_uncertainty(obj: object) -> TypeGuard[_UncertaintyLike]:
    """
    Check for uncertainty attributes without importing the container class.

    Attribute checks avoid a circular import with the uncertainty container.

    :param obj: Object to inspect.
    :return: Whether the object exposes both ``_nom`` and ``_err``.
    """
    return hasattr(obj, "_nom") and hasattr(obj, "_err")


def _is_sequence_with_uncertainty_elements(obj: Any) -> bool:
    """
    Check for an iterable with a length and at least one uncertainty element.

    :param obj: Object to inspect; strings and non-object arrays are excluded.
    :return: Whether the object contains an uncertainty among its immediate elements.
    """
    if isinstance(obj, np.ndarray) and obj.dtype.kind != "O":
        return False
    return (
        is_iterable(obj)
        and has_length(obj)
        and not isinstance(obj, str)
        and any(_is_uncertainty(item) for item in obj)
    )


def convert_arg(arg: Any, attr: Literal["_nom", "_err"] = "_nom") -> Any:
    """
    Extract nominal values or errors from an argument.

    Dictionaries, lists, and tuples retain their structure. Other sequences
    containing uncertainties become nominal arrays when values are requested;
    error extraction from those sequences returns ``None``. Exact leaves are
    unchanged for nominal extraction and become ``None`` for error extraction.

    :param arg: Argument that may contain uncertainties.
    :param attr: Use ``"_nom"`` for nominal values or ``"_err"`` for errors.
    :return: The extracted values or errors, following the input structure where
        supported.
    """
    if _is_uncertainty(arg):
        return getattr(arg, attr)
    elif isinstance(arg, dict):
        return {key: convert_arg(value, attr) for key, value in arg.items()}
    elif isinstance(arg, (tuple, list)):
        values = [convert_arg(value, attr) for value in arg]
        return tuple(values) if isinstance(arg, tuple) else values
    elif _is_sequence_with_uncertainty_elements(arg):
        if attr != "_nom":
            return None
        else:
            return np.array([convert_arg(item, attr) for item in arg])
    else:
        if attr != "_nom":
            return None
        else:
            return arg


def classify_and_split_args_and_kwargs(
    *args: Any, **kwargs: Any
) -> tuple[tuple[int, ...], tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
    """
    Separate uncertainty data from call arguments.

    This compatibility helper strips keyword arguments to their nominal values.
    Propagation itself uses the shared engine, which supports uncertain keywords.

    :param args: Positional arguments to classify and extract.
    :param kwargs: Keyword arguments from which to extract nominal values.
    :return: Four items: positions of uncertain positional arguments, nominal positional
        arguments, positional errors, and nominal keyword arguments. Exact positional
        arguments receive zero errors.
    """
    uncert_argnums = tuple(idx for idx, arg in enumerate(args) if _is_uncertainty(arg))
    uncert_arg_nom = tuple(convert_arg(arg, "_nom") for arg in args)
    uncert_arg_err = []
    for aidx, arg in enumerate(args):
        carg = convert_arg(arg, "_err")
        if carg is not None:
            uncert_arg_err.append(carg)
        else:
            uncert_arg_err.append(np.zeros_like(uncert_arg_nom[aidx]))
    uncert_arg_err = tuple(uncert_arg_err)
    uncert_kwarg_nom = {key: convert_arg(arg, "_nom") for key, arg in kwargs.items()}
    return uncert_argnums, uncert_arg_nom, uncert_arg_err, uncert_kwarg_nom


def _nominal(arg: Any) -> Any:
    """
    Extract nominal values from an argument or argument tree.

    :param arg: Argument that may contain uncertainties.
    :return: The nominal argument with supported container structure preserved.
    """
    return convert_arg(arg, "_nom")


def _error(arg: Any) -> npt.NDArray[Any] | np.number | float:
    """
    Extract the error of an operand.

    :param arg: Uncertain or exact operand.
    :return: The operand error, or zeros matching an exact operand.
    """
    return arg._err if _is_uncertainty(arg) else np.zeros_like(arg)


def _concatenate_signature(
    arrays: Iterable[npt.ArrayLike],
    axis: int | None = 0,
    out: npt.NDArray[Any] | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
    casting: str = "same_kind",
) -> None:
    """
    Provide an inspectable signature for :func:`numpy.concatenate`.

    This placeholder is used only for argument binding and performs no operation.

    :param arrays: Arrays to concatenate.
    :param axis: Axis to join along, or ``None`` to flatten the inputs.
    :param out: Optional destination for the result.
    :param dtype: Requested result dtype.
    :param casting: NumPy casting rule for the result.
    :return: ``None``.
    """


def _signature(func: Callable[..., Any]) -> inspect.Signature:
    """
    Find a NumPy callable signature, including known builtin fallbacks.

    :param func: Callable whose parameters are needed for argument binding.
    :return: The inspectable call signature.
    :raises ValueError: The callable has no inspectable signature or supported fallback.
    """
    try:
        return inspect.signature(func)
    except ValueError:
        if func is np.concatenate:
            return inspect.signature(_concatenate_signature)
        raise


def _write_out(
    result: Any,
    out: Any,
    *,
    where: npt.ArrayLike = True,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe", "same_value"] = "same_kind",
) -> Any:
    """
    Copy a result into an output buffer after validating the update.

    For uncertain outputs, both components are staged before either is changed.

    :param result: Computed result to copy.
    :param out: Destination matching the result shape and uncertainty status.
    :param where: Mask selecting entries to update.
    :param casting: NumPy casting rule applied to the copies.
    :return: The original output object after the update.
    :raises TypeError: The destination has incompatible uncertainty status or violates
        the casting rule.
    :raises ValueError: Shapes, write permissions, or copy arguments are incompatible.
    """
    if _is_uncertainty(result):
        if not _is_uncertainty(out):
            msg = "An uncertain result requires an Uncertainty out argument"
            raise TypeError(msg)
        if np.shape(out._nom) != np.shape(result._nom):
            msg = "out must have the same shape as the result"
            raise ValueError(msg)
        if isinstance(out._nom, np.ndarray) and not (
            out._nom.flags.writeable and out._err.flags.writeable
        ):
            msg = "Both nominal and error out buffers must be writable"
            raise ValueError(msg)
        value = np.array(out._nom, copy=True)
        error = np.array(out._err, copy=True)
        np.copyto(value, result._nom, where=where, casting=casting)
        np.copyto(error, result._err, where=where, casting=casting)
        if isinstance(out._nom, np.ndarray):
            out._nom[...] = value
            out._err[...] = error
        else:
            out._nom, out._err = value.item(), error.item()
        return out
    if _is_uncertainty(out):
        msg = "An exact result requires an ndarray out argument"
        raise TypeError(msg)
    value = np.array(out, copy=True)
    np.copyto(value, result, where=where, casting=casting)
    out[...] = value
    return out


def _dispatch_with_out(
    func: Callable[..., Any],
    numpy_func: Callable[..., Any],
    func_type: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Bind NumPy arguments and handle output buffers for a registered operation.

    :param func: Uncertainty-aware implementation to invoke.
    :param numpy_func: Original NumPy operation whose signature governs the call.
    :param func_type: Registry category: ``"function"`` or ``"ufunc"``.
    :param args: Positional arguments supplied by the caller.
    :param kwargs: Keyword arguments supplied by the caller.
    :return: The computed result, or the supplied output object after updating it.
    :raises TypeError: Arguments cannot be bound or the output specification is invalid.
    :raises ValueError: A ufunc mask excludes entries without an output buffer to
        preserve them.
    """
    kwargs = dict(kwargs)
    where = True
    if func_type == "ufunc" or isinstance(numpy_func, np.ufunc):
        assert isinstance(numpy_func, np.ufunc)
        out = kwargs.pop("out", None)
        if len(args) > numpy_func.nin:
            if out is not None or len(args) != numpy_func.nin + 1:
                msg = "Invalid out arguments"
                raise TypeError(msg)
            args, out = args[: numpy_func.nin], args[-1]
        if isinstance(out, tuple):
            if len(out) != 1:
                msg = "Only one output is supported"
                raise TypeError(msg)
            out = out[0]
        where = kwargs.pop("where", True)
        if out is None and not np.all(where):
            msg = "where requires an out argument to preserve unselected values"
            raise ValueError(msg)
    else:
        bound = _signature(numpy_func).bind(*args, **kwargs)
        out = bound.arguments.pop("out", None)
        if (
            numpy_func is np.concatenate
            and out is not None
            and bound.arguments.get("dtype") is not None
        ):
            msg = "concatenate cannot accept both out and dtype"
            raise TypeError(msg)
        args, kwargs = bound.args, bound.kwargs
    result = func(*args, **kwargs)
    if out is None:
        return result
    return _write_out(result, out, where=where, casting=kwargs.get("casting", "same_kind"))


def implements(numpy_func_string: str, func_type: str) -> Callable[[_F], _F]:
    """
    Register an uncertainty-aware NumPy function or ufunc implementation.

    :param numpy_func_string: Name of the NumPy operation, including any submodule path.
    :param func_type: Registry category: ``"function"`` or ``"ufunc"``.
    :return: A decorator that registers an output-aware dispatcher and returns the
        implementation.
    :raises ValueError: The registry category is invalid when applying the decorator.
    """

    def decorator(func: _F) -> _F:
        numpy_func = get_func_from_package(numpy_func_string, np)
        assert numpy_func is not None

        @wraps(func)
        def dispatch(*args: Any, **kwargs: Any) -> Any:
            return _dispatch_with_out(func, numpy_func, func_type, args, kwargs)

        if func_type == "function":
            HANDLED_FUNCTIONS[numpy_func_string] = dispatch
        elif func_type == "ufunc":
            HANDLED_UFUNCS[numpy_func_string] = dispatch
        else:
            msg = f"Invalid func_type {func_type}"
            raise ValueError(msg)
        return func

    return decorator


def get_func_from_package(func_str: str, namespace: Any) -> Callable[..., Any] | None:
    # Handle functions in submodules
    """
    Resolve a dotted operation name within a namespace.

    :param func_str: Operation name relative to the namespace.
    :param namespace: Module or object containing the operation.
    :return: The resolved operation, or ``None`` when the first name is unavailable.
    :raises AttributeError: A nested component of an available name does not exist.
    """
    func_str_split = func_str.split(".")
    func = getattr(namespace, func_str_split[0], None)
    # If the function is not available, do not attempt to implement it
    if func is None:
        return None
    for func_str_piece in func_str_split[1:]:
        func = getattr(func, func_str_piece)

    return func


def _select_extreme(
    func: Callable[..., Any],
    selection_operator: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Uncertainty[Any]:
    """
    Select the error belonging to the nominal minimum or maximum.

    :param func: NumPy minimum or maximum reduction.
    :param selection_operator: Use ``"argmin"`` or ``"argmax"`` to locate the selected
        sample.
    :param args: Positional reduction arguments.
    :param kwargs: Keyword reduction arguments, including masks and initial values.
    :return: The nominal extreme with the error of its selected input.
    """
    bound = inspect.signature(func).bind(*args, **kwargs)
    data = bound.arguments["a"]
    value = func(*_nominal(args), **_nominal(kwargs))
    nominal = np.asarray(_nominal(data))
    error = np.asarray(_error(data))
    axis = bound.arguments.get("axis")
    axes = (
        tuple(range(nominal.ndim))
        if axis is None
        else (axis if isinstance(axis, tuple) else (axis,))
    )
    axes = tuple(a % nominal.ndim for a in axes) if nominal.ndim else ()
    remaining = tuple(a for a in range(nominal.ndim) if a not in axes)
    outer_shape = tuple(nominal.shape[a] for a in remaining)
    reduced_size = int(np.prod([nominal.shape[a] for a in axes], dtype=int))
    order = remaining + axes
    samples = np.transpose(nominal, order).reshape((*outer_shape, reduced_size))
    errors = np.transpose(error, order).reshape((*outer_shape, reduced_size))
    where = np.broadcast_to(_nominal(bound.arguments.get("where", True)), nominal.shape)
    where = np.transpose(where, order).reshape(samples.shape)
    is_max = selection_operator == "argmax"
    masked = np.where(where, samples, -np.inf if is_max else np.inf)
    initial = bound.arguments.get("initial", np._NoValue)
    if reduced_size:
        indices = getattr(np, selection_operator)(masked, axis=-1)[..., None]
        err = np.take_along_axis(errors, indices, axis=-1)[..., 0]
        chosen = np.take_along_axis(masked, indices, axis=-1)[..., 0]
        valid = np.take_along_axis(where, indices, axis=-1)[..., 0]
        if initial is not np._NoValue and initial is not None:
            initial_value = _nominal(initial)
            wins = initial_value >= chosen if is_max else initial_value <= chosen
            wins = wins | ~valid | np.isnan(initial_value)
            err = np.where(wins, _error(initial), err)
    else:
        err = np.broadcast_to(_error(initial), outer_shape)
    return Uncertainty(value, np.reshape(err, np.shape(value)))


def implement_func(
    func_type: str,
    func_str: str,
    implement_mode: str,
    selection_operator: str | None = None,
    custom_jax_dispatch: Callable[..., Any] | None = None,
) -> None:
    """
    Register a NumPy operation using a standard propagation strategy.

    Operations unavailable in NumPy are skipped. Strategies requiring derivatives
    must have a JAX implementation.

    :param func_type: Registry category: ``"function"`` or ``"ufunc"``.
    :param func_str: Operation name relative to NumPy and JAX NumPy.
    :param implement_mode: Propagation strategy: ``"same_shape"``,
        ``"same_shape_bool"``, ``"nograd"``, ``"selection_operator"``, ``"selection"``,
        ``"apply_to_both"``, ``"reduction_binary"``, or ``"reduction_unary"``.
    :param selection_operator: Name of the index operation used by the ``"selection"``
        strategy.
    :param custom_jax_dispatch: Optional JAX callable overriding the operation resolved
        by name.
    :return: ``None``.
    :raises ValueError: The category or strategy is invalid, or a required JAX operation
        is missing.
    """
    if custom_jax_dispatch is not None:
        func = custom_jax_dispatch
    else:
        func = get_func_from_package(func_str, jnp)
    # Skip the JAX overhead if you dont need gradient info
    func_np = get_func_from_package(func_str, np)
    if func_np is None:
        return
    if func is None and implement_mode in {
        "same_shape",
        "reduction_binary",
        "reduction_unary",
    }:
        msg = f"No JAX implementation is available for {func_str}"
        raise ValueError(msg)

    @implements(func_str, func_type)
    def implementation(*args: Any, **kwargs: Any) -> Any:

        if implement_mode == "same_shape":

            def jax_func(*operands: npt.ArrayLike, **options: Any) -> jax.Array:
                assert func is not None
                dtype = options.pop("dtype", None)
                if dtype is not None:
                    if np.dtype(dtype).kind not in "fc":
                        msg_0 = "Uncertainty propagation requires a floating-point dtype"
                        raise TypeError(msg_0)
                    operands = tuple(jnp.asarray(arg, dtype=dtype) for arg in operands)
                for key in ("casting", "order", "subok"):
                    options.pop(key, None)
                return func(*operands, **options)

            return propagate(jax_func, args, kwargs, nominal_func=func_np, elementwise=True)
        elif (
            implement_mode == "same_shape_bool"
            or implement_mode == "nograd"
            or implement_mode == "selection_operator"
        ):
            return func_np(*_nominal(args), **_nominal(kwargs))
        elif implement_mode == "selection":
            assert selection_operator is not None
            return _select_extreme(func_np, selection_operator, args, kwargs)
        elif implement_mode == "apply_to_both":
            if func_type == "ufunc":
                val = func_np(*_nominal(args), **_nominal(kwargs))
                err = _error(args[0])
            else:
                bound = inspect.signature(func_np).bind(*args, **kwargs)
                data = bound.arguments["a"]
                nominal = inspect.signature(func_np).bind(*_nominal(args), **_nominal(kwargs))
                val = func_np(*nominal.args, **nominal.kwargs)
                nominal.arguments["a"] = _error(data)
                err = func_np(*nominal.args, **nominal.kwargs)
            return Uncertainty(val, err)
        elif implement_mode in ["reduction_binary", "reduction_unary"]:
            assert func is not None
            if func_str == "median":
                bound = _signature(func_np).bind(*args, **kwargs)
                bound.arguments["overwrite_input"] = False
                args, kwargs = bound.args, bound.kwargs
            return propagate(func, args, kwargs, nominal_func=func_np)
        else:
            msg = f"Invalid implement_mode {implement_mode}"
            raise ValueError(msg)


# Returns a bool array of the same shape (i.e. elementwise conditionals)
unary_bcast_same_shape_bool_ufuncs = [
    "isnan",
    "isinf",
    "isfinite",
    "signbit",
]
binary_bcast_same_shape_bool_ufuncs = [
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
]
bcast_same_shape_bool_ufuncs = (
    binary_bcast_same_shape_bool_ufuncs + unary_bcast_same_shape_bool_ufuncs
)
for ufunc in bcast_same_shape_bool_ufuncs:
    implement_func("ufunc", ufunc, implement_mode="same_shape_bool")

# Applies the ufunc to the value and discards the error
bcast_nograd_ufuncs = [
    "sign",
    "floor_divide",
    "fmod",
    "mod",
    "remainder",
    "copysign",
    "nextafter",
    "trunc",
    "spacing",
]
for ufunc in bcast_nograd_ufuncs:
    implement_func("ufunc", ufunc, implement_mode="nograd")
# Returns an Uncertainty array of the same shape using the gradient (i.e. elementwise math operations)
unary_bcast_same_shape_ufuncs = [
    "sqrt",
    "cbrt",
    "square",
    "reciprocal",
    "arccos",
    "arcsin",
    "arctan",
    "arccosh",
    "arcsinh",
    "arctanh",
    "exp",
    "expm1",
    "exp2",
    "log",
    "log10",
    "log1p",
    "log2",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "radians",
    "degrees",
    "deg2rad",
    "rad2deg",
    "absolute",
]
binary_bcast_same_shape_ufuncs = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "logaddexp",
    "logaddexp2",
    "divide",
    "arctan2",
    "hypot",
]
bcast_same_shape_ufuncs = binary_bcast_same_shape_ufuncs + unary_bcast_same_shape_ufuncs
for ufunc in bcast_same_shape_ufuncs:
    implement_func("ufunc", ufunc, implement_mode="same_shape")

# Returns the indices of the Uncertainty array by some criteria
bcast_selection_operator_funcs = ["argmax", "argmin"]

for ufunc in bcast_selection_operator_funcs:
    implement_func("function", ufunc, implement_mode="selection_operator")


# This can only be done correctly via Monte-Carlo estimation
def _monte_carlo_reduction(
    a: Uncertainty[npt.NDArray[np.floating]],
    axis: int | tuple[int, ...] | None = None,
    **kwargs: Any,
) -> Uncertainty[Any]:
    """
    Estimate a reduction and its error using independent normal samples.

    :param a: Input uncertainty array to sample.
    :param axis: Reduction axes, or ``None`` to reduce all input dimensions.
    :param kwargs: Reduction options; ``op`` supplies the NumPy reduction name.
    :return: The sample mean and standard deviation wrapped as an uncertainty.
    """
    N = 10000
    samples = np.random.normal(size=(*a._nom.shape, N))  # noqa: NPY002
    samples = a._nom[..., None] + samples * a._err[..., None]

    if axis is None:
        axis = tuple(range(a._nom.ndim))

    operation = getattr(np, kwargs.pop("op"))
    result = operation(samples, axis=axis, **kwargs)

    mean_value = np.mean(result, axis=-1)
    std_value = np.std(result, axis=-1)

    return Uncertainty(mean_value, std_value)


# apply_via_monte_carlo = ["max", "min", "amax", "amin", "median"]
# for ufunc in apply_via_monte_carlo:
#     implements(ufunc, "function")(partial(_monte_carlo_reduction, op=ufunc))

# Selects a sub-section of or reshapes the Uncertainty array by some criteria
bcast_selection_funcs = {
    "max": "argmax",
    "min": "argmin",
    "amax": "argmax",
    "amin": "argmin",
}

for ufunc, sel_op in bcast_selection_funcs.items():
    implement_func(
        "function",
        ufunc,
        implement_mode="selection",
        selection_operator=sel_op,
    )

# Applies ufunc or func to both the value and error
bcast_apply_to_both_funcs = [
    "compress",
    "diagonal",
    "ravel",
    "repeat",
    "reshape",
    "squeeze",
    "swapaxes",
    "take",
    "transpose",
]
bcast_apply_to_both_ufuncs = [
    "conj",
    "conjugate",
    "negative",
    "positive",
    "fabs",
    "ceil",
    "floor",
    "rint",
]
for ufunc in bcast_apply_to_both_ufuncs:
    implement_func("ufunc", ufunc, implement_mode="apply_to_both")
for ufunc in bcast_apply_to_both_funcs:
    implement_func("function", ufunc, implement_mode="apply_to_both")

# Applies a reduction
implement_func(
    "function",
    "trapezoid",
    implement_mode="reduction_binary",
    custom_jax_dispatch=jax.scipy.integrate.trapezoid,
)

bcast_reduction_unary = ["std", "sum", "var", "mean", "ptp", "median"]
for ufunc in bcast_reduction_unary:
    implement_func("function", ufunc, implement_mode="reduction_unary")


def _power(
    x1: Uncertainty[Any] | npt.ArrayLike,
    x2: Uncertainty[Any] | npt.ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> Uncertainty[Any]:
    """
    Raise a base to an exponent and propagate independent operand errors.

    :param x1: Base values, possibly uncertain.
    :param x2: Exponent values, possibly uncertain.
    :param args: Additional positional arguments forwarded to :func:`numpy.power`.
    :param kwargs: NumPy power options; ufunc signature overrides are unsupported.
    :return: The power result with propagated standard deviations.
    :raises TypeError: A requested dtype cannot represent propagated errors or a
        signature override is supplied.
    """
    if "signature" in kwargs or "sig" in kwargs:
        msg = "Uncertain power supports dtype, but not a ufunc signature override"
        raise TypeError(msg)
    dtype = kwargs.get("dtype")
    if dtype is not None and np.dtype(dtype).kind not in "fc":
        msg = "Uncertainty propagation requires a floating-point dtype"
        raise TypeError(msg)

    if _is_uncertainty(x1):
        sA = x1._err
        A = x1._nom
    else:
        sA = 0
        A = x1
    if _is_uncertainty(x2):
        sB = x2._err
        B = x2._nom
    else:
        sB = 0
        B = x2

    new_mag = np.power(A, B, *args, **kwargs)
    if dtype is not None:
        A, sA, B, sB = (np.asarray(arg, dtype=dtype) for arg in (A, sA, B, sB))
    _, new_err = _power_uncertainty(A, sA, B, sB)

    return Uncertainty(new_mag, new_err)


implements("power", "function")(_power)
implements("power", "ufunc")(_power)


def _searchsort(
    x1: Uncertainty[Any] | npt.ArrayLike,
    x2: Uncertainty[Any] | npt.ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> np.intp | npt.NDArray[np.intp]:
    """
    Find insertion positions using nominal values.

    :param x1: Sorted search array, possibly uncertain.
    :param x2: Values to insert, possibly uncertain.
    :param args: Additional positional arguments for :func:`numpy.searchsorted`.
    :param kwargs: Keyword options for :func:`numpy.searchsorted`.
    :return: The insertion indices in the nominal search array.
    """
    A = x1._nom if _is_uncertainty(x1) else x1
    B = x2._nom if _is_uncertainty(x2) else x2

    return np.searchsorted(A, B, *args, **kwargs)


implements("searchsorted", "function")(_searchsort)


@implements("unique", "function")
def _unique(*args: Any, **kwargs: Any) -> Uncertainty[Any] | tuple[Any, ...]:
    """
    Find unique nominal values while retaining the selected input errors.

    :param args: Positional arguments for :func:`numpy.unique`.
    :param kwargs: Keyword options for :func:`numpy.unique`, including metadata flags.
    :return: Unique uncertain values, optionally followed by the requested indices,
        inverse indices, and counts.
    """
    bound = inspect.signature(np.unique).bind(*args, **kwargs)
    ar = bound.arguments["ar"]
    return_index = bound.arguments.get("return_index", False)
    return_inverse = bound.arguments.get("return_inverse", False)
    return_counts = bound.arguments.get("return_counts", False)
    axis = bound.arguments.get("axis")
    bound.arguments["ar"] = _nominal(ar)
    bound.arguments["return_index"] = True
    values, indices, *metadata = np.unique(*bound.args, **bound.kwargs)
    errors = np.take(_error(ar), indices, axis=axis)
    result = Uncertainty(values, errors)
    if not (return_index or return_inverse or return_counts):
        return result
    return (result, *((indices,) if return_index else ()), *metadata)


@implements("round", "function")
def _round(a: Uncertainty[Any], *args: Any, **kwargs: Any) -> Uncertainty[Any]:
    """
    Round nominal values while preserving their errors.

    :param a: Input uncertainty array.
    :param args: Additional positional arguments for :func:`numpy.round`.
    :param kwargs: Keyword options for :func:`numpy.round`.
    :return: The rounded nominal values with their original errors.
    """
    val = np.round(a._nom, *args, **kwargs)
    return a.__class__(val, a._err)


@implements("sort", "function")
def _sort(a: Uncertainty[Any], *args: Any, **kwargs: Any) -> Uncertainty[Any]:
    """
    Sort nominal values and reorder errors with them.

    :param a: Input uncertainty array.
    :param args: Additional positional arguments for :func:`numpy.sort`.
    :param kwargs: Keyword options for :func:`numpy.sort`.
    :return: The sorted uncertainty array.
    """
    ind = np.argsort(a._nom, *args, **kwargs)
    bound = inspect.signature(np.sort).bind(a, *args, **kwargs)
    axis = bound.arguments.get("axis", -1)
    return _take_along_axis(a, ind, axis=axis)


@implements("take_along_axis", "function")
def _take_along_axis(a: Uncertainty[Any], *args: Any, **kwargs: Any) -> Uncertainty[Any]:
    """
    Select matching nominal values and errors along an axis.

    :param a: Input uncertainty array.
    :param args: Additional positional arguments for :func:`numpy.take_along_axis`.
    :param kwargs: Keyword options for :func:`numpy.take_along_axis`.
    :return: The selected uncertainty values.
    """
    val = np.take_along_axis(a._nom, *args, **kwargs)
    err = np.take_along_axis(a._err, *args, **kwargs)
    return a.__class__(val, err)


@implements("concatenate", "function")
def _concatenate(
    concat: Iterable[Uncertainty[Any] | npt.ArrayLike], *args: Any, **kwargs: Any
) -> Uncertainty[npt.NDArray[np.floating]]:
    """
    Join nominal values and errors along an axis.

    :param concat: Input arrays to join.
    :param args: Additional positional arguments for :func:`numpy.concatenate`.
    :param kwargs: Keyword options for :func:`numpy.concatenate`.
    :return: The concatenated uncertainty array, with zero errors for exact inputs.
    """
    concat = tuple(concat)
    val = np.concatenate([_nominal(a) for a in concat], *args, **kwargs)
    error_kwargs = {key: value for key, value in kwargs.items() if key != "dtype"}
    err = np.concatenate([_error(a) for a in concat], *args, **error_kwargs)
    return Uncertainty(val, err)


@implements("ndim", "function")
def _ndim(a: Uncertainty[Any], *args: Any, **kwargs: Any) -> int:
    """
    Count the dimensions of an uncertainty value.

    :param a: Input uncertainty.
    :param args: Reserved positional arguments; the dispatcher rejects extra arguments.
    :param kwargs: Reserved keyword arguments; the dispatcher rejects extra arguments.
    :return: The number of nominal array dimensions.
    """
    return np.ndim(a._nom)


def wrap_numpy(
    func_type: str, func: str | Callable[..., Any], args: Sequence[Any], kwargs: dict[str, Any]
) -> Any:
    """
    Dispatch a NumPy operation to its registered uncertainty implementation.

    :param func_type: Registry category: ``"function"`` or ``"ufunc"``.
    :param func: NumPy callable or its registered name.
    :param args: Positional arguments forwarded to the implementation.
    :param kwargs: Keyword arguments forwarded to the implementation.
    :return: The operation result, or ``NotImplemented`` if it is not registered.
    :raises ValueError: The registry category is invalid.
    """
    if func_type == "function":
        handled = HANDLED_FUNCTIONS
        # Need to handle functions in submodules
        if isinstance(func, str):
            name = func
        else:
            name = ".".join([*func.__module__.split(".")[1:], func.__name__])
    elif func_type == "ufunc":
        handled = HANDLED_UFUNCS
        # ufuncs do not have func.__module__
        name = func if isinstance(func, str) else func.__name__
    else:
        msg = f"Invalid func_type {func_type}"
        raise ValueError(msg)

    if name not in handled:
        return NotImplemented
    return handled[name](*args, **kwargs)
