import numpy as np
import oclpy as xp


def test_transpose_and_reshape():
    x = xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=xp.float32)
    np.testing.assert_array_equal(xp.asnumpy(x.T), [[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal(xp.asnumpy(xp.reshape(x, (3, 2))), [[1, 2], [3, 4], [5, 6]])


def test_matmul():
    x = xp.asarray([[1, 2], [3, 4]], dtype=xp.float32)
    np.testing.assert_array_equal(xp.asnumpy(xp.matmul(x, x)), [[7, 10], [15, 22]])


def test_scalar_reductions_return_zero_dimensional_arrays():
    x = xp.asarray([1, 2, 3], dtype=xp.float32)
    assert xp.sum(x).shape == ()
    assert xp.asnumpy(xp.sum(x)) == 6
