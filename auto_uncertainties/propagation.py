"""Shared first-order propagation, preserving call structure and JAX dtypes."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Protocol, TypeGuard
lazy from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

lazy from auto_uncertainties.uncertainty.uncertainty_containers import Uncertainty

__all__ = ()


class UncertaintyLike(Protocol):
    """Describe the nominal values and errors used by propagation helpers."""

    nom: Any
    err: Any


def is_uncertainty(value: object) -> TypeGuard[UncertaintyLike]:
    """
    Check for the nominal-value and error attributes of an uncertainty.

    :param value: Object to inspect.
    :return: Whether the object exposes both ``nom`` and ``err``.
    """
    return hasattr(value, "nom") and hasattr(value, "err")


@contextmanager
def precision_context(*trees: Any) -> Iterator[None]:
    """
    Enable x64 locally when JAX's promoted input dtype needs it.

    Probe promotion with x64 available so that an ambient ``jax_enable_x64=False``
    cannot canonicalize float64 inputs to float32 before their dtype is inspected.
    Integer and boolean metadata and nonnumeric objects do not determine precision.
    The previous JAX precision setting is restored when the context exits.

    :param trees: Argument pytrees whose nominal values, errors, and dtype
        specifications determine precision.
    :yields: ``None`` while the required precision is enabled.
    """
    leaves = jax.tree_util.tree_leaves(trees, is_leaf=is_uncertainty)
    numeric = []
    for leaf in leaves:
        candidates = (leaf.nom, leaf.err) if is_uncertainty(leaf) else (leaf,)
        for value in candidates:
            if isinstance(value, np.dtype | type):
                try:
                    dtype = np.dtype(value)
                except TypeError:
                    continue
                if dtype.kind in "fc":
                    numeric.append(dtype)
                continue
            if isinstance(value, str):
                try:
                    dtype = np.dtype(value)
                except TypeError:
                    continue
                if dtype.kind in "fc":
                    numeric.append(dtype)
                continue
            if value is None or isinstance(value, bytes):
                continue
            try:
                dtype = np.asarray(value).dtype
            except TypeError, ValueError:
                continue
            if dtype.kind in "fc":
                numeric.append(value)
    with jax.enable_x64(new_val=True):
        dtype = jnp.result_type(*numeric) if numeric else np.dtype("float32")
        needs_x64 = dtype in (np.dtype("float64"), np.dtype("complex128"))
    with jax.enable_x64(jax.config.read("jax_enable_x64") or needs_x64):
        yield


def weighted_derivative(derivative: npt.ArrayLike, error: npt.ArrayLike) -> npt.NDArray[Any]:
    """
    Scale a derivative by its input error without introducing spurious NaNs.

    :param derivative: Derivative values, including JAX zero tangents.
    :param error: Input standard deviations broadcastable to the derivative.
    :return: Weighted derivatives, with zero contributions wherever the error or
        derivative is zero.
    """
    derivative, error = np.asarray(derivative), np.asarray(error)
    if derivative.dtype == jax.dtypes.float0:
        return np.zeros(derivative.shape, dtype=error.dtype)
    # An exact operand must not contribute NaN through 0 * inf at a boundary.
    result = np.zeros(
        np.broadcast_shapes(derivative.shape, error.shape),
        dtype=np.result_type(derivative, error),
    )
    with np.errstate(invalid="ignore", over="ignore"):
        np.multiply(derivative, error, out=result, where=(error != 0) & (derivative != 0))
    return result


def wrap_result(
    value: npt.ArrayLike, error: npt.ArrayLike
) -> Uncertainty[Any] | np.bool_ | npt.NDArray[np.bool_]:
    """
    Attach propagated errors to a numeric result.

    :param value: Nominal output values.
    :param error: Output standard deviations.
    :return: An uncertainty object, or the unchanged nominal values for boolean outputs.
    """
    value = np.asarray(value)
    if value.dtype.kind == "b":
        return value[()] if value.ndim == 0 else value
    return Uncertainty(value, np.asarray(error))


def propagate_elementwise(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    nominal_func: Callable[..., Any] | None,
) -> Uncertainty[Any] | np.bool_ | npt.NDArray[np.bool_]:
    """
    Propagate errors through an elementwise operation using a diagonal Jacobian.

    :param func: JAX operation acting independently on each broadcast element.
    :param args: Positional operands, possibly containing uncertainties.
    :param kwargs: Keyword options passed to the operation.
    :param nominal_func: Optional operation used to compute the nominal result.
    :return: The result with propagated errors, or exact boolean values.
    :raises TypeError: Operand promotion does not produce an inexact dtype.
    """
    nominal = tuple(
        arg.nom
        if is_uncertainty(arg)
        else np.asarray(arg)
        if isinstance(arg, tuple | list)
        else arg
        for arg in args
    )
    dtype = jnp.result_type(*nominal)
    if kwargs.get("dtype") is not None:
        dtype = jnp.result_type(np.dtype(kwargs["dtype"]))
    if not jnp.issubdtype(dtype, jnp.inexact):
        msg = f"Uncertainty propagation requires floating-point promotion, got {dtype}"
        raise TypeError(msg)
    promoted = jnp.broadcast_arrays(*(jnp.asarray(arg, dtype=dtype) for arg in nominal))
    positions = tuple(i for i, arg in enumerate(args) if is_uncertainty(arg))

    def evaluate(*values: jax.Array) -> jax.Array:
        operands = list(promoted)
        for position, value in zip(positions, values, strict=True):
            operands[position] = value
        return func(*operands, **kwargs)

    if positions:
        value, pullback = jax.vjp(evaluate, *(promoted[i] for i in positions))
        gradients = pullback(jnp.ones_like(value))
        error = np.zeros(np.shape(value), dtype=np.asarray(value).dtype)
        for position, gradient in zip(positions, gradients, strict=True):
            error = np.hypot(error, weighted_derivative(gradient, args[position].err))
    else:
        value = func(*promoted, **kwargs)
        error = np.zeros(np.shape(value))
    if nominal_func is not None:
        value = nominal_func(*nominal, **kwargs)
    return wrap_result(value, error)


def propagate(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    nominal_func: Callable[..., Any] | None = None,
    elementwise: bool = False,
) -> Any:
    """
    Propagate independent uncertainties through a real-valued JAX callable.

    Uncertainty leaves can occur anywhere in positional and keyword argument
    pytrees. All other leaves are captured unchanged, including integer indices,
    shapes, and strings. General propagation contracts each output's Jacobian
    with input errors before combining independent contributions in quadrature.
    The elementwise fast path is reserved for scalar-valued ufunc operations.

    :param func: JAX-compatible function to evaluate and differentiate.
    :param args: Positional arguments for the function.
    :param kwargs: Keyword arguments for the function.
    :param nominal_func: Optional function used for nominal results while ``func``
        supplies derivatives.
    :param elementwise: Use the diagonal-Jacobian fast path for elementwise operations.
    :return: The output pytree with numeric leaves wrapped as uncertainties and boolean
        leaves kept exact.
    :raises TypeError: Uncertain inputs do not have a supported floating-point dtype.
    """
    with precision_context(args, kwargs):
        if elementwise:
            return propagate_elementwise(func, args, kwargs, nominal_func)

        leaves, structure = jax.tree_util.tree_flatten((args, kwargs), is_leaf=is_uncertainty)
        positions = tuple(i for i, leaf in enumerate(leaves) if is_uncertainty(leaf))
        nominal_leaves = [leaf.nom if is_uncertainty(leaf) else leaf for leaf in leaves]
        values = tuple(jnp.asarray(nominal_leaves[i]) for i in positions)
        errors = tuple(np.asarray(leaves[i].err) for i in positions)
        for value in values:
            if not jnp.issubdtype(value.dtype, jnp.floating):
                msg = f"Uncertain inputs must have real floating-point values, got {value.dtype}"
                raise TypeError(msg)

        def evaluate(*dynamic: jax.Array) -> Any:
            call_leaves = list(nominal_leaves)
            for position, value in zip(positions, dynamic, strict=True):
                call_leaves[position] = value
            call_args, call_kwargs = jax.tree_util.tree_unflatten(structure, call_leaves)
            return func(*call_args, **call_kwargs)

        if not positions:
            value = (nominal_func or func)(*args, **kwargs)
            return jax.tree_util.tree_map(lambda val: wrap_result(val, np.zeros_like(val)), value)

        def value_with_aux(*dynamic: jax.Array) -> tuple[Any, Any]:
            value = evaluate(*dynamic)
            # Boolean/index outputs have zero derivatives. Casting only the
            # differentiation target avoids JAX float0 Jacobian limitations,
            # while the auxiliary value preserves the callable's output dtype.
            differentiable = jax.tree_util.tree_map(
                lambda leaf: (
                    leaf
                    if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
                    else jnp.asarray(leaf, dtype=jnp.float32)
                ),
                value,
            )
            return differentiable, value

        sample = evaluate(*values)
        samples = jax.tree_util.tree_leaves(sample)
        if not samples:
            return sample
        reverse_mode = all(
            jnp.issubdtype(np.asarray(output).dtype, jnp.floating) for output in samples
        ) and sum(np.size(output) for output in samples) <= sum(value.size for value in values)
        differentiate = jax.jacrev if reverse_mode else jax.jacfwd
        jacobians, value = differentiate(
            value_with_aux, argnums=tuple(range(len(values))), has_aux=True
        )(*values)
        # At singular derivatives, one AD direction can create spurious
        # off-diagonal 0 * inf NaNs. Use the other direction only to recover
        # entries it can determine; genuinely undefined derivatives stay NaN.
        if any(np.isnan(np.asarray(leaf)).any() for leaf in jax.tree_util.tree_leaves(jacobians)):
            alternate = jax.jacfwd if reverse_mode else jax.jacrev
            other, _ = alternate(value_with_aux, argnums=tuple(range(len(values))), has_aux=True)(
                *values
            )
            jacobians = jax.tree_util.tree_map(
                lambda original, alternative: np.where(np.isnan(original), alternative, original),
                jacobians,
                other,
            )
        if nominal_func is not None:
            nominal_args, nominal_kwargs = jax.tree_util.tree_unflatten(structure, nominal_leaves)
            value = nominal_func(*nominal_args, **nominal_kwargs)
        outputs, output_structure = jax.tree_util.tree_flatten(value)
        output_jacobians = output_structure.flatten_up_to(jacobians)
        results = []
        for output_value, derivatives in zip(outputs, output_jacobians, strict=True):
            output = np.asarray(output_value)
            error = np.zeros(output.shape, dtype=np.result_type(output, *errors))
            for derivative, input_error in zip(derivatives, errors, strict=True):
                weighted = weighted_derivative(derivative, input_error)
                weighted = weighted.reshape((*output.shape, input_error.size))
                contribution = np.hypot.reduce(np.abs(weighted), axis=-1, initial=0)
                error = np.hypot(error, contribution)
            results.append(wrap_result(output, error))
        return jax.tree_util.tree_unflatten(output_structure, results)
