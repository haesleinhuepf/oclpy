import numpy as np
import pytest
import oclpy as xp


def test_asarray_has_array_api_metadata():
    x = xp.asarray([1, 2, 3], dtype=xp.float32)
    assert x.__array_namespace__() is xp
    assert x.shape == (3,)
    assert x.ndim == 1
    assert x.size == 3
    assert x.dtype == xp.float32
    np.testing.assert_array_equal(xp.asnumpy(x), [1, 2, 3])


@pytest.mark.parametrize("factory,value", [(xp.zeros, 0), (xp.ones, 1)])
def test_creation_factories(factory, value):
    result = factory((2, 3), dtype=xp.float32)
    np.testing.assert_array_equal(xp.asnumpy(result), np.full((2, 3), value))


def test_full_and_arange():
    np.testing.assert_array_equal(xp.asnumpy(xp.full((2,), 7, dtype=xp.int32)), [7, 7])
    np.testing.assert_array_equal(xp.asnumpy(xp.arange(1, 5, dtype=xp.int32)), [1, 2, 3, 4])


@pytest.mark.parametrize("dtype", [xp.bool, xp.int64, xp.uint64, xp.float64])
def test_backend_rejects_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match="not supported"):
        xp.asarray([1], dtype=dtype)


def test_asarray_round_trips_four_dimensional_data():
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    result = xp.asarray(data)
    assert result.shape == data.shape
    np.testing.assert_array_equal(xp.asnumpy(result), data)


def test_creation_factories_support_four_dimensions():
    shape = (2, 3, 4, 5)
    np.testing.assert_array_equal(xp.asnumpy(xp.zeros(shape, dtype=xp.float32)), np.zeros(shape, dtype=np.float32))
    np.testing.assert_array_equal(xp.asnumpy(xp.ones(shape, dtype=xp.float32)), np.ones(shape, dtype=np.float32))
