# -*- coding: utf-8 -*-
# Based heavily on the implementation of pint's numpy array function wrapping

from __future__ import annotations

try:
    import numpy as np
except ImportError:
    np = None
try:
    import jax
    import jax.numpy as jnp

except ImportError:
    jax = None
    jnp = None

from .util import has_length, is_iterable, ndarray_to_scalar

HANDLED_UFUNCS = {}
HANDLED_FUNCTIONS = {}


def _is_uncertainty(obj):
    """Test for _nom and _err attrs.

    This is done in place of isinstance(Uncertainty, arg), which would cause a circular import.

    Parameters
    ----------
    obj : Object


    Returns
    -------
    bool
    """
    return hasattr(obj, "_nom") and hasattr(obj, "_err")


def _is_sequence_with_uncertainty_elements(obj):
    """Test for sequences of uncertainties.

    Parameters
    ----------
    obj : object


    Returns
    -------
    True if obj is a sequence and at least one element is an Uncertainty; False otherwise
    """
    return (
        is_iterable(obj)
        and has_length(obj)
        and not isinstance(obj, str)
        and any(_is_uncertainty(item) for item in obj)
    )


def convert_arg(arg, attr: str = None):
    """Convert uncertainties and sequences of uncertainties to nominal values or errors

    This function has a different behavior if the nominal value or errors are requested.
    If the nominal value is requested, the outputs are:
        copies if arg is not an uncertainty (or sequence of)
        nominal values (or sequence of) if arg is an uncertainty (or sequence of)
    If the errors are requested, the outputs are:
        None if arg is not an uncertainty (or sequence of)
        errors (or sequence of) if arg is an uncertainty (or sequence of)
    """
    if _is_uncertainty(arg):
        return getattr(arg, attr)
    elif _is_sequence_with_uncertainty_elements(arg):
        if attr != "_nom":
            return None
        else:
            return [convert_arg(item, attr) for item in arg]
    else:
        if attr != "_nom":
            return None
        else:
            return arg


def classify_and_split_args_and_kwargs(*args, **kwargs):
    """Checks the args and kwargs to see if they contain uncertanty info, and prepares them for use by JAX.

    JAX does not support differentiating with respect to kwargs so uncertainty info there is just discarded.
    Returns:
        uncert_argnums: list of int
            A list of the positional arguments with respect to which derivatives need to be taken
        uncert_arg_nom:
            A list of arguments to the function, without errors
        uncert_arg_err:
            A list of errors for the positional arguments which will have derivatives
        uncert_kwarg_nom:
            A dict of keyword args that will be passed to the function
        uncert_instance:
            Returns an instance of an Uncertainty object for the class constructor
    """

    uncert_argnums = tuple(
        idx
        for idx, arg in enumerate(args)
        if convert_arg(arg, "_nom") is not None
    )
    uncert_arg_nom = tuple(convert_arg(arg, "_nom") for arg in args)
    uncert_arg_err = []
    for aidx, arg in enumerate(args):
        carg = convert_arg(arg, "_err")
        if convert_arg(arg, "_err") is not None:
            uncert_arg_err.append(carg)
        else:
            uncert_arg_err.append(jnp.zeros_like(uncert_arg_nom[aidx]))
    uncert_arg_err = tuple(uncert_arg_err)
    uncert_kwarg_nom = {
        key: convert_arg(arg, "_nom") for key, arg in kwargs.items()
    }
    return uncert_argnums, uncert_arg_nom, uncert_arg_err, uncert_kwarg_nom


def implements(numpy_func_string, func_type):
    """Register an __array_function__/__array_ufunc__ implementation for Uncertainty
    objects.

    """

    def decorator(func):
        if func_type == "function":
            HANDLED_FUNCTIONS[numpy_func_string] = func
        elif func_type == "ufunc":
            HANDLED_UFUNCS[numpy_func_string] = func
        else:
            raise ValueError("Invalid func_type {}".format(func_type))
        return func

    return decorator


def get_func_from_package(func_str, namespace):
    # Handle functions in submodules
    func_str_split = func_str.split(".")
    func = getattr(namespace, func_str_split[0], None)
    # If the function is not available, do not attempt to implement it
    if func is None:
        return
    for func_str_piece in func_str_split[1:]:
        func = getattr(func, func_str_piece)

    return func


def elementwise_grad(g):
    def wrapped(*args, **kwargs):
        y, g_vjp = jax.vjp(lambda *a: g(*a, **kwargs), *args)
        x_bar = g_vjp(np.ones_like(y))
        return x_bar

    return wrapped


def get_mappable_dims(*args):
    # Check that all the args have the same dimension
    assert all([a.ndim == args[0].ndim for a in args])
    # Check that the size of each dimension is either the same as the maximum, or 1
    mappable = [None for a in args]
    max_dim_sizes = []
    for i, dim in enumerate(range(args[0].ndim)):
        sz = [a.shape[dim] for a in args]
        max_sz = max(sz)
        max_dim_sizes.append(max_sz)
        assert all([s == max_sz or s == 1 for s in sz])
    for i, a in enumerate(args):
        map_axes = []
        for j, dim in enumerate(range(args[0].ndim)):
            if a.shape[dim] == max_dim_sizes[j]:
                map_axes.append(j)
        if len(map_axes) > 1:
            mappable[i] = tuple(map_axes)
        elif len(map_axes) == 1:
            mappable[i] = map_axes[0]
    return mappable, max_dim_sizes


def implement_func(
    func_type,
    func_str,
    implement_mode,
    grad_argnum_override=None,
    selection_operator=None,
    output_rank=0,
):
    """Add default-behavior NumPy function/ufunc to the handled list.

    Parameters
    ----------
    func_type : str
        "function" for NumPy functions, "ufunc" for NumPy ufuncs
    func_str : str
        String representing the name of the NumPy function/ufunc to add
    implement_mode: str
        Instructs on the implementation type
    grad_argnum_override: list of int
        Positions of arguments that should be differentiated, if necessary to enforce
    selection_operator: func
        An operator that provides selection indices that correspond to the action of the function that needs to be implemented
        E.g. np.argmax for np.max
    output_rank: int
        The rank of the output. If it's greater than rank 0 and derivatives are needed, jacfwd needs to be used instead of grad.
    """
    # If Jax+NumPy is not available, do not attempt implement that which does not exist
    if jnp is None:
        return

    func = get_func_from_package(func_str, jnp)
    # Skip the JAX overhead if you dont need gradient info
    func_np = get_func_from_package(func_str, np)

    @implements(func_str, func_type)
    def implementation(*args, **kwargs):
        from auto_uncertainties import Uncertainty

        (
            uncert_argnums,
            uncert_arg_nom,
            uncert_arg_err,
            uncert_kwarg_nom,
        ) = classify_and_split_args_and_kwargs(*args, **kwargs)

        # Determine result through base numpy function on stripped arguments
        if implement_mode == "same_shape":
            bcast_args_nom = np.broadcast_arrays(*uncert_arg_nom)
            bcast_args_err = np.broadcast_arrays(*uncert_arg_err)
            value = func_np(*bcast_args_nom, **uncert_kwarg_nom)
            grads = elementwise_grad(func)(*bcast_args_nom, **uncert_kwarg_nom)
            error_dot_grad_sqr = [
                (e * g) ** 2 for e, g in zip(bcast_args_err, grads)
            ]
            error = np.sum(error_dot_grad_sqr, axis=0) ** 0.5
            return Uncertainty(value, error)
        #            return uncert_instance.__class__(val, err)
        elif implement_mode == "same_shape_bool":
            return func_np(*uncert_arg_nom, **uncert_kwarg_nom)
        elif implement_mode == "nograd":
            return func_np(*uncert_arg_nom, **uncert_kwarg_nom)
        elif implement_mode == "selection_operator":
            return func_np(*uncert_arg_nom, **uncert_kwarg_nom)
        elif implement_mode == "selection":
            sel_func_np = get_func_from_package(selection_operator, np)
            axis = uncert_kwarg_nom.pop("axis", None)
            if axis is None:
                idx = sel_func_np(*uncert_arg_nom, **uncert_kwarg_nom)
                return np.ravel(uncert_arg_nom[0])[idx]
            else:
                idxs = np.expand_dims(
                    sel_func_np(
                        *uncert_arg_nom, axis=axis, **uncert_kwarg_nom
                    ),
                    axis=axis,
                )
                return np.take_along_axis(uncert_arg_nom[0], idxs, axis=axis)
        elif implement_mode in ["apply_to_both"]:
            val = func_np(*uncert_arg_nom, **uncert_kwarg_nom)
            err = np.abs(func_np(*uncert_arg_err, **uncert_kwarg_nom))
            return Uncertainty(val, err)
        elif implement_mode in ["reduction_binary", "reduction_unary"]:
            axis = uncert_kwarg_nom.get("axis", None)

            bcast_args_nom = np.broadcast_arrays(*uncert_arg_nom)
            bcast_args_err = np.broadcast_arrays(*uncert_arg_err)

            val = func_np(*bcast_args_nom, **uncert_kwarg_nom)
            grads = elementwise_grad(func)(*bcast_args_nom, **uncert_kwarg_nom)
            if axis is not None:
                axis = tuple(axis)
                error_dot_grad_sqr = [
                    np.sum((e * g) ** 2, axis=axis)
                    for e, g in zip(bcast_args_err, grads)
                ]
            else:
                error_dot_grad_sqr = [
                    np.sum((e * g) ** 2) for e, g in zip(bcast_args_err, grads)
                ]

            err = np.sum(error_dot_grad_sqr, axis=0) ** 0.5
            return Uncertainty(ndarray_to_scalar(val), ndarray_to_scalar(err))


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
bcast_same_shape_ufuncs = (
    binary_bcast_same_shape_ufuncs + unary_bcast_same_shape_ufuncs
)
for ufunc in bcast_same_shape_ufuncs:
    implement_func("ufunc", ufunc, implement_mode="same_shape")

# Returns the indices of the Uncertainty array by some criteria
bcast_selection_operator_funcs = ["argmax", "argmin"]

for ufunc in bcast_selection_operator_funcs:
    implement_func("function", ufunc, implement_mode="selection_operator")

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
    "round",
    "ceil",
    "floor",
    "rint",
]
for ufunc in bcast_apply_to_both_ufuncs:
    implement_func("ufunc", ufunc, implement_mode="apply_to_both")
for ufunc in bcast_apply_to_both_funcs:
    implement_func("function", ufunc, implement_mode="apply_to_both")

# Applies a reduction
implement_func("function", "trapz", implement_mode="reduction_binary")

bcast_reduction_unary = ["std", "sum", "var", "mean", "ptp", "median"]
for ufunc in bcast_reduction_unary:
    implement_func("function", ufunc, implement_mode="reduction_unary")


@implements("round", "function")
def _round(a, *args, **kwargs):
    val = np.round(a._nom, *args, **kwargs).squeeze()
    err = a._err + 0.5 * np.sign(a._err)
    return a.__class__(val, err)


@implements("take_along_axis", "function")
def _take_along_axis(a, *args, **kwargs):
    val = np.take_along_axis(a._nom, *args, **kwargs).squeeze()
    err = np.take_along_axis(a._err, *args, **kwargs).squeeze()
    return a.__class__(val, err)


@implements("concatenate", "function")
def _concatenate(concat, *args, **kwargs):
    for a in concat:
        if not _is_uncertainty(a):
            raise ValueError
    val = np.concatenate([a._nom for a in concat], **kwargs)
    err = np.concatenate([a._err for a in concat], **kwargs)
    return a.__class__(val, err)


@implements("ndim", "function")
def _ndim(a, *args, **kwargs):
    return np.ndim(a._nom)


def wrap_numpy(func_type, func, args, kwargs):
    """Return the result from a JAX+NumPy function/ufunc as wrapped by uncert."""

    if func_type == "function":
        handled = HANDLED_FUNCTIONS
        # Need to handle functions in submodules
        if isinstance(func, str):
            name = func
        else:
            name = ".".join(func.__module__.split(".")[1:] + [func.__name__])
    elif func_type == "ufunc":
        handled = HANDLED_UFUNCS
        # ufuncs do not have func.__module__
        if isinstance(func, str):
            name = func
        else:
            name = func.__name__
    else:
        raise ValueError("Invalid func_type {}".format(func_type))

    if name not in handled:
        return NotImplemented
    return handled[name](*args, **kwargs)
