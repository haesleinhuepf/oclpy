import numpy as np
import oclpy as xp


def test_transpose_and_reshape():
    x = xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=xp.float32)
    np.testing.assert_array_equal(xp.asnumpy(x.T), [[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal(xp.asnumpy(xp.reshape(x, (3, 2))), [[1, 2], [3, 4], [5, 6]])


def test_reshape_and_permute_dims_work_for_four_dimensional_arrays():
    x = xp.asarray(np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5))
    reshaped = xp.reshape(x, (6, 20))
    assert reshaped.shape == (6, 20)
    np.testing.assert_array_equal(xp.asnumpy(reshaped), np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(6, 20))
    transposed = xp.permute_dims(x, (0, 2, 3, 1))
    assert transposed.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(xp.asnumpy(transposed), np.transpose(np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5), (0, 2, 3, 1)))


def test_matmul():
    x = xp.asarray([[1, 2], [3, 4]], dtype=xp.float32)
    np.testing.assert_array_equal(xp.asnumpy(xp.matmul(x, x)), [[7, 10], [15, 22]])


def test_batched_matmul():
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    y = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    np.testing.assert_array_equal(xp.asnumpy(xp.matmul(xp.asarray(x), xp.asarray(y))), np.matmul(x, y))


def test_scalar_reductions_return_zero_dimensional_arrays():
    x = xp.asarray([1, 2, 3], dtype=xp.float32)
    assert xp.sum(x).shape == ()
    assert xp.asnumpy(xp.sum(x)) == 6
