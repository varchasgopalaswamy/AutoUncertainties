# Based heavily on the implementation of pint's numpy array function wrapping

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial, wraps
import inspect
from typing import Any, Literal, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from auto_uncertainties.arithmetic import binary
from auto_uncertainties.propagation import is_uncertainty, propagate
from auto_uncertainties.util import has_length, is_iterable
lazy from auto_uncertainties.uncertainty.uncertainty_containers import Uncertainty

F = TypeVar("F", bound=Callable[..., Any])
HANDLED_UFUNCS: dict[str, Callable[..., Any]] = {}
HANDLED_FUNCTIONS: dict[str, Callable[..., Any]] = {}

__all__ = ()


def is_sequence_with_uncertainty_elements(obj: Any) -> bool:
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
        and any(is_uncertainty(item) for item in obj)
    )


def extract_nominal(arg: Any) -> Any:
    """
    Extract nominal values while preserving an argument tree's call structure.

    :param arg: Argument containing uncertain leaves or exact metadata.
    :return: Nominal components with exact leaves and container structure preserved.
    """
    if is_uncertainty(arg):
        return arg.nom
    if isinstance(arg, dict):
        return {key: extract_nominal(value) for key, value in arg.items()}
    if isinstance(arg, tuple | list):
        values = [extract_nominal(value) for value in arg]
        return tuple(values) if isinstance(arg, tuple) else values
    if is_sequence_with_uncertainty_elements(arg):
        return np.array([extract_nominal(item) for item in arg])
    return arg


def extract_error(arg: Any) -> npt.NDArray[Any] | np.number | float:
    """
    Extract the error of an operand.

    :param arg: Uncertain or exact operand.
    :return: The operand error, or zeros matching an exact operand.
    """
    return arg.err if is_uncertainty(arg) else np.zeros_like(arg)


def concatenate_signature(
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


def signature(func: Callable[..., Any]) -> inspect.Signature:
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
            return inspect.signature(concatenate_signature)
        raise


def write_out(
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
    if is_uncertainty(result):
        if not is_uncertainty(out):
            msg = "An uncertain result requires an Uncertainty out argument"
            raise TypeError(msg)
        if np.shape(out.nom) != np.shape(result.nom):
            msg = "out must have the same shape as the result"
            raise ValueError(msg)
        if isinstance(out.nom, np.ndarray) and not (
            out.nom.flags.writeable and out.err.flags.writeable
        ):
            msg = "Both nominal and error out buffers must be writable"
            raise ValueError(msg)
        if np.shares_memory(out.nom, out.err):
            msg = "Nominal and error out buffers must not overlap"
            raise ValueError(msg)
        value = np.array(out.nom, copy=True)
        error = np.array(out.err, copy=True)
        np.copyto(value, result.nom, where=where, casting=casting)
        np.copyto(error, result.err, where=where, casting=casting)
        if isinstance(out.nom, np.ndarray):
            out.nom[...] = value
            out.err[...] = error
        else:
            out.nom, out.err = value.item(), error.item()
        return out
    if is_uncertainty(out):
        msg = "An exact result requires an ndarray out argument"
        raise TypeError(msg)
    value = np.array(out, copy=True)
    np.copyto(value, result, where=where, casting=casting)
    out[...] = value
    return out


def dispatch_with_out(
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
        bound = signature(numpy_func).bind(*args, **kwargs)
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
    return write_out(result, out, where=where, casting=kwargs.get("casting", "same_kind"))


def implements(numpy_func_string: str, func_type: str) -> Callable[[F], F]:
    """
    Register an uncertainty-aware NumPy function or ufunc implementation.

    :param numpy_func_string: Name of the NumPy operation, including any submodule path.
    :param func_type: Registry category: ``"function"`` or ``"ufunc"``.
    :return: A decorator that registers an output-aware dispatcher and returns the
        implementation.
    :raises ValueError: The registry category is invalid when applying the decorator.
    """

    def decorator(func: F) -> F:
        numpy_func = get_func_from_package(numpy_func_string, np)
        assert numpy_func is not None

        @wraps(func)
        def dispatch(*args: Any, **kwargs: Any) -> Any:
            return dispatch_with_out(func, numpy_func, func_type, args, kwargs)

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


def select_extreme(
    func: Callable[..., Any],
    selection_operator: str,
    *args: Any,
    **kwargs: Any,
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
    value = func(*extract_nominal(args), **extract_nominal(kwargs))
    nominal = np.asarray(extract_nominal(data))
    error = np.asarray(extract_error(data))
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
    where = np.broadcast_to(extract_nominal(bound.arguments.get("where", True)), nominal.shape)
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
            initial_value = extract_nominal(initial)
            wins = initial_value >= chosen if is_max else initial_value <= chosen
            wins = wins | ~valid | np.isnan(initial_value)
            err = np.where(wins, extract_error(initial), err)
    else:
        err = np.broadcast_to(extract_error(initial), outer_shape)
    return Uncertainty(value, np.reshape(err, np.shape(value)))


def exact(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Apply an operation to nominal values only.

    :param func: NumPy operation.
    :param args: Positional arguments.
    :param kwargs: Keyword arguments.
    :return: The exact nominal result.
    """
    return func(*extract_nominal(args), **extract_nominal(kwargs))


def propagate_elementwise(func: np.ufunc, *args: Any, **kwargs: Any) -> Any:
    """
    Differentiate an elementwise operation after validating its NumPy call.

    :param func: NumPy ufunc with a JAX implementation.
    :param args: Positional operands.
    :param kwargs: Ufunc options after output handling.
    :return: The propagated result.
    """
    if "signature" in kwargs or "sig" in kwargs:
        msg = "Uncertainty propagation does not support ufunc signature overrides"
        raise TypeError(msg)
    nominal = func(*extract_nominal(args), **extract_nominal(kwargs))
    jax_func = getattr(jnp, func.__name__)

    def evaluate(*operands: Any, **options: Any) -> Any:
        """
        Evaluate derivatives using options supported by JAX.

        :param operands: Promoted nominal operands.
        :param options: Options already validated by NumPy.
        :return: Differentiable nominal result.
        """
        options.pop("dtype", None)  # The propagation engine has already cast operands.
        for key in ("casting", "order", "subok"):
            options.pop(key, None)
        return jax_func(*operands, **options)

    result = propagate(evaluate, args, kwargs, elementwise=True)
    return Uncertainty(nominal, result.error)


def arithmetic(func: np.ufunc, left: Any, right: Any, **kwargs: Any) -> Any:
    """
    Apply shared analytic arithmetic to two operands.

    :param func: Smooth binary NumPy ufunc.
    :param left: Left operand, possibly uncertain.
    :param right: Right operand, possibly uncertain.
    :param kwargs: NumPy ufunc options after output handling.
    :return: The nominal result with propagated errors.
    """
    return Uncertainty(
        *binary(
            func,
            extract_nominal(left),
            left.err if is_uncertainty(left) else 0,
            extract_nominal(right),
            right.err if is_uncertainty(right) else 0,
            **kwargs,
        )
    )


def structural(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Apply the same array transformation to values and errors.

    :param func: NumPy structural operation with an ``a`` parameter.
    :param args: Positional arguments.
    :param kwargs: Keyword arguments.
    :return: The transformed uncertainty.
    """
    bound = signature(func).bind(*args, **kwargs)
    data = bound.arguments["a"]
    nominal = signature(func).bind(*extract_nominal(args), **extract_nominal(kwargs))
    value = func(*nominal.args, **nominal.kwargs)
    nominal.arguments["a"] = extract_error(data)
    return Uncertainty(value, func(*nominal.args, **nominal.kwargs))


def preserve_error(func: np.ufunc, value: Any, **kwargs: Any) -> Any:
    """
    Transform nominal values while preserving their errors.

    :param func: Unary NumPy ufunc.
    :param value: Operand, possibly uncertain.
    :param kwargs: NumPy ufunc options after output handling.
    :return: The transformed nominal values with unchanged errors.
    """
    return Uncertainty(func(extract_nominal(value), **kwargs), extract_error(value))


def reduction(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Propagate a reduction using the general JAX engine.

    :param func: NumPy reduction with a JAX implementation.
    :param args: Positional arguments.
    :param kwargs: Keyword arguments.
    :return: The propagated reduction result.
    """
    if func is np.median:
        bound = signature(func).bind(*args, **kwargs)
        bound.arguments["overwrite_input"] = False
        args, kwargs = bound.args, bound.kwargs
    jax_func = (
        jax.scipy.integrate.trapezoid if func is np.trapezoid else getattr(jnp, func.__name__)
    )
    return propagate(jax_func, args, kwargs, nominal_func=func)


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
    implements(ufunc, "ufunc")(partial(exact, getattr(np, ufunc)))

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
    implements(ufunc, "ufunc")(partial(exact, getattr(np, ufunc)))
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
    "arctan2",
    "hypot",
]
bcast_same_shape_ufuncs = binary_bcast_same_shape_ufuncs + unary_bcast_same_shape_ufuncs
for ufunc in bcast_same_shape_ufuncs:
    handler = (
        arithmetic
        if ufunc in {"add", "subtract", "multiply", "divide", "true_divide"}
        else propagate_elementwise
    )
    implements(ufunc, "ufunc")(partial(handler, getattr(np, ufunc)))

# Returns the indices of the Uncertainty array by some criteria
bcast_selection_operator_funcs = ["argmax", "argmin"]

for ufunc in bcast_selection_operator_funcs:
    implements(ufunc, "function")(partial(exact, getattr(np, ufunc)))


# Selects a sub-section of or reshapes the Uncertainty array by some criteria
bcast_selection_funcs = {
    "max": "argmax",
    "min": "argmin",
    "amax": "argmax",
    "amin": "argmin",
}

for ufunc, sel_op in bcast_selection_funcs.items():
    implements(ufunc, "function")(partial(select_extreme, getattr(np, ufunc), sel_op))

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
    implements(ufunc, "ufunc")(partial(preserve_error, getattr(np, ufunc)))
for ufunc in bcast_apply_to_both_funcs:
    implements(ufunc, "function")(partial(structural, getattr(np, ufunc)))

bcast_reduction_unary = ["std", "sum", "var", "mean", "ptp", "median"]
for ufunc in [*bcast_reduction_unary, "trapezoid"]:
    implements(ufunc, "function")(partial(reduction, getattr(np, ufunc)))

implements("power", "ufunc")(partial(arithmetic, np.power))


def searchsort(
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
    A = x1.nom if is_uncertainty(x1) else x1
    B = x2.nom if is_uncertainty(x2) else x2

    return np.searchsorted(A, B, *args, **kwargs)


implements("searchsorted", "function")(searchsort)


@implements("unique", "function")
def unique(*args: Any, **kwargs: Any) -> Uncertainty[Any] | tuple[Any, ...]:
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
    bound.arguments["ar"] = extract_nominal(ar)
    bound.arguments["return_index"] = True
    values, indices, *metadata = np.unique(*bound.args, **bound.kwargs)
    errors = np.take(extract_error(ar), indices, axis=axis)
    result = Uncertainty(values, errors)
    if not (return_index or return_inverse or return_counts):
        return result
    return (result, *((indices,) if return_index else ()), *metadata)


@implements("round", "function")
def round_values(a: Uncertainty[Any], *args: Any, **kwargs: Any) -> Uncertainty[Any]:
    """
    Round nominal values while preserving their errors.

    :param a: Input uncertainty array.
    :param args: Additional positional arguments for :func:`numpy.round`.
    :param kwargs: Keyword options for :func:`numpy.round`.
    :return: The rounded nominal values with their original errors.
    """
    val = np.round(a.nom, *args, **kwargs)
    return a.__class__(val, a.err)


@implements("clip", "function")
def clip(a: Uncertainty[Any], *args: Any, **kwargs: Any) -> Uncertainty[Any]:
    """
    Clip nominal values while preserving their existing errors.

    :param a: Input uncertainty array.
    :param args: Positional clipping bounds.
    :param kwargs: NumPy clipping options after output handling.
    :return: The clipped nominal values with unchanged errors.
    """
    return Uncertainty(np.clip(a.nom, *extract_nominal(args), **extract_nominal(kwargs)), a.err)


@implements("sort", "function")
def sort(a: Uncertainty[Any], *args: Any, **kwargs: Any) -> Uncertainty[Any]:
    """
    Sort nominal values and reorder errors with them.

    :param a: Input uncertainty array.
    :param args: Additional positional arguments for :func:`numpy.sort`.
    :param kwargs: Keyword options for :func:`numpy.sort`.
    :return: The sorted uncertainty array.
    """
    ind = np.argsort(a.nom, *args, **kwargs)
    bound = inspect.signature(np.sort).bind(a, *args, **kwargs)
    axis = bound.arguments.get("axis", -1)
    return take_along_axis(a, ind, axis=axis)


@implements("take_along_axis", "function")
def take_along_axis(a: Uncertainty[Any], *args: Any, **kwargs: Any) -> Uncertainty[Any]:
    """
    Select matching nominal values and errors along an axis.

    :param a: Input uncertainty array.
    :param args: Additional positional arguments for :func:`numpy.take_along_axis`.
    :param kwargs: Keyword options for :func:`numpy.take_along_axis`.
    :return: The selected uncertainty values.
    """
    val = np.take_along_axis(a.nom, *args, **kwargs)
    err = np.take_along_axis(a.err, *args, **kwargs)
    return a.__class__(val, err)


@implements("concatenate", "function")
def concatenate(
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
    val = np.concatenate([extract_nominal(a) for a in concat], *args, **kwargs)
    error_kwargs = {key: value for key, value in kwargs.items() if key != "dtype"}
    err = np.concatenate([extract_error(a) for a in concat], *args, **error_kwargs)
    return Uncertainty(val, err)


@implements("ndim", "function")
def ndim(a: Uncertainty[Any], *args: Any, **kwargs: Any) -> int:
    """
    Count the dimensions of an uncertainty value.

    :param a: Input uncertainty.
    :param args: Reserved positional arguments; the dispatcher rejects extra arguments.
    :param kwargs: Reserved keyword arguments; the dispatcher rejects extra arguments.
    :return: The number of nominal array dimensions.
    """
    return np.ndim(a.nom)


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
