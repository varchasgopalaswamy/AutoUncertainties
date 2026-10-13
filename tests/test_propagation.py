from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auto_uncertainties import Uncertainty, propagate_uncertainties
from auto_uncertainties._propagation import precision_context


def test_permutations_and_coupled_outputs():
    u = Uncertainty([1, 2], [0.1, 0.2])
    reverse = propagate_uncertainties(lambda x: x[::-1])(u)
    np.testing.assert_allclose(reverse.error, [0.2, 0.1])
    coupled = propagate_uncertainties(lambda x: jnp.array([x[0] + x[1], x[0] - x[1]]))(u)
    np.testing.assert_allclose(coupled.value, [3, -1])
    np.testing.assert_allclose(coupled.error, np.hypot(0.1, 0.2))
    centered = propagate_uncertainties(lambda x: x - jnp.mean(x))(u)
    np.testing.assert_allclose(centered.error, np.hypot(0.1, 0.2) / 2)


def test_reused_input_and_scalar_broadcast_preserve_internal_dependence():
    cancel = propagate_uncertainties(lambda x: x - x)(Uncertainty(2, 0.1))
    assert cancel.error == 0
    summed = propagate_uncertainties(lambda x: jnp.sum(x * jnp.ones(10)))(Uncertainty(5, 0.5))
    assert summed.value == 50
    assert summed.error == 5


def test_different_input_shapes_and_keyword_uncertainty():
    @propagate_uncertainties()
    def scaled(x, *, gain):
        return {"scaled": x * gain, "total": jnp.sum(x * gain)}

    result = scaled(Uncertainty([1, 2], [0.1, 0.2]), gain=Uncertainty(2, 0.05))
    np.testing.assert_allclose(result["scaled"].error, np.hypot([0.2, 0.4], [0.05, 0.1]))
    assert result["total"].value == 6
    assert result["total"].error == pytest.approx(np.hypot(np.hypot(0.2, 0.4), 0.15))


def test_nested_inputs_and_exact_metadata():
    @propagate_uncertainties
    def operate(data, indices, *, config):
        assert config["mode"] == "sum"
        return jnp.sum(data["x"][indices]) * config["gain"] + data["offsets"][0]

    result = operate(
        {"x": Uncertainty([1, 2, 3], [0.1, 0.2, 0.3]), "offsets": [Uncertainty(1, 0.5)]},
        np.array([0, 2]),
        config={"gain": 2, "mode": "sum"},
    )
    assert result.value == 9
    assert result.error == pytest.approx(np.linalg.norm([0.2, 0.6, 0.5]))


def test_matrix_product_and_nested_output():
    @propagate_uncertainties
    def transform(x, matrix):
        return (matrix @ x, {"norm_squared": jnp.sum(x * x)})

    matrix = np.array([[2.0, 3.0], [-1.0, 4.0], [0.0, 5.0]])
    result, stats = transform(Uncertainty([1, 2], [0.1, 0.2]), matrix)
    np.testing.assert_allclose(result.value, [8, 7, 10])
    np.testing.assert_allclose(result.error, np.sqrt((matrix**2) @ np.array([0.01, 0.04])))
    assert stats["norm_squared"].error == pytest.approx(np.hypot(0.2, 0.8))


@pytest.mark.parametrize("axis", (None, 0, 1, (0,), (0, 1), ()))
@pytest.mark.parametrize("keepdims", (False, True))
def test_reduction_axes(axis, keepdims):
    u = Uncertainty(np.arange(6.0).reshape(2, 3), np.arange(1.0, 7.0).reshape(2, 3) / 10)
    result = propagate_uncertainties(jnp.sum)(u, axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(result.value, np.sum(u.value, axis=axis, keepdims=keepdims))
    expected = np.sqrt(np.sum(u.error**2, axis=axis, keepdims=keepdims))
    np.testing.assert_allclose(result.error, expected)
    assert np.shape(result.value) == np.shape(expected)


@pytest.mark.parametrize(
    ("value", "error", "operation", "expected"),
    (
        (50.0, 0.1, jnp.exp, np.exp(50.0) * 0.1),
        (1e30, 1.0, jnp.log, 1e-30),
        (1e40, 1e20, jnp.sqrt, 0.5),
    ),
)
def test_float64_and_large_dynamic_range(value, error, operation, expected):
    with jax.enable_x64(new_val=False):
        result = propagate_uncertainties(operation)(Uncertainty(value, error))
        assert result.error == pytest.approx(expected, rel=1e-12, abs=0)
        assert not jax.config.x64_enabled


@pytest.mark.parametrize(("dtype", "enabled"), ((np.float32, False), (np.float64, True)))
def test_precision_context_uses_jax_promotion_and_restores(dtype, enabled):
    u = Uncertainty(np.array([1], dtype=dtype), np.array([0.1], dtype=dtype))
    with jax.enable_x64(new_val=False):
        with precision_context(u, 2):
            assert jax.config.x64_enabled is enabled
        assert not jax.config.x64_enabled
        result = propagate_uncertainties(jnp.sin)(u)
        assert result.value.dtype == dtype
        assert not jax.config.x64_enabled
        msg = "test"
        with pytest.raises(RuntimeError), precision_context(u):
            raise RuntimeError(msg)
        assert not jax.config.x64_enabled


def test_exact_positional_integer_is_not_a_differentiation_input():
    result = propagate_uncertainties(lambda x, count: jnp.sum(jnp.repeat(x, count)))(
        Uncertainty(2, 0.1), 3
    )
    assert result.value == 6
    assert result.error == pytest.approx(0.3)


def test_no_uncertain_inputs_and_output_metadata():
    fn = propagate_uncertainties(lambda x: (jnp.sum(x), jnp.argmax(x)))
    result, index = fn(np.array([1.0, 2.0]))
    assert result.value == 3 and result.error == 0
    assert index.value == 1 and index.error == 0
    result, index = fn(Uncertainty([1, 2], [0.1, 0.2]))
    assert result.error == pytest.approx(np.hypot(0.1, 0.2))
    assert index.value == 1 and index.error == 0


def test_legacy_modes_use_general_propagation_and_preserve_metadata():
    def reverse(x):
        """Original documentation."""
        return x[::-1]

    fn = propagate_uncertainties(reverse, "same_shape")
    assert fn.__name__ == "reverse"
    assert fn.__doc__ == reverse.__doc__
    np.testing.assert_allclose(fn(Uncertainty([1, 2], [0.1, 0.2])).error, [0.2, 0.1])
    with pytest.raises(ValueError, match="Invalid implement_mode"):
        propagate_uncertainties(reverse, "invalid")


def test_empty_input_and_output_arrays():
    u = Uncertainty(np.empty(0), np.empty(0))
    empty = propagate_uncertainties(lambda x: x)(u)
    assert empty.value.shape == empty.error.shape == (0,)
    total = propagate_uncertainties(jnp.sum)(u)
    assert total.value == 0 and total.error == 0


def test_unused_nonfinite_errors_do_not_contaminate_outputs():
    from auto_uncertainties import configure

    configure(nonfinite_error_policy="propagate")
    for invalid in (np.inf, np.nan):
        result = propagate_uncertainties(lambda x: x[0])(Uncertainty([1, 2], [0.1, invalid]))
        assert result.value == 1 and result.error == pytest.approx(0.1)
    result = propagate_uncertainties(lambda x: x - x)(Uncertainty(2, np.inf))
    assert result.value == 0 and result.error == 0


def test_singular_derivatives_do_not_contaminate_other_outputs():
    from auto_uncertainties import configure

    configure(nonfinite_error_policy="propagate")
    source = Uncertainty([0, 1], [0.1, 0.2])
    result = propagate_uncertainties(jnp.sqrt)(source)
    np.testing.assert_array_equal(result.value, [0, 1])
    np.testing.assert_allclose(result.error, [np.inf, 0.1])
    selected = propagate_uncertainties(lambda x: jnp.sqrt(x)[1])(source)
    assert selected.value == 1 and selected.error == pytest.approx(0.1)


def test_boolean_and_empty_output_pytrees():
    result = propagate_uncertainties(lambda x: (x > 0, x * 2))(Uncertainty(1, 0.1))
    assert bool(result[0])
    assert result[1].error == pytest.approx(0.2)
    for output in ((), {}, None):
        assert (
            propagate_uncertainties(lambda _x, output=output: output)(Uncertainty(1, 0.1))
            == output
        )


@pytest.mark.parametrize("dtype", (np.float64, "float64", np.dtype("float64")))
def test_explicit_output_dtype_controls_differentiation_precision(dtype):
    source = Uncertainty(np.array([100], dtype=np.float32), np.array([0.01], dtype=np.float32))
    with jax.enable_x64(new_val=False):
        result = np.exp(source, dtype=dtype)
        np.testing.assert_allclose(result.error, np.exp(100.0) * source.error.astype(np.float64))
        assert not jax.config.x64_enabled


def test_jax_strict_promotion_is_respected():
    source = Uncertainty(np.array([1], dtype=np.float32), np.array([0.1], dtype=np.float32))
    with jax.numpy_dtype_promotion("strict"):
        np.testing.assert_allclose(np.add(source, 2).error, source.error)
        with pytest.raises(jax.dtypes.TypePromotionError, match="promotion"):
            np.add(source, np.array([2], dtype=np.int32))
