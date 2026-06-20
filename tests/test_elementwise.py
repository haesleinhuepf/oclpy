import numpy as np
import pytest
import oclpy as xp


@pytest.mark.parametrize(
    "operation,expected",
    [(xp.add, [4, 6]), (xp.subtract, [-2, -2]), (xp.multiply, [3, 8]), (xp.divide, [1/3, 1/2])],
)
def test_arithmetic(operation, expected):
    x = xp.asarray([1, 2], dtype=xp.float32)
    y = xp.asarray([3, 4], dtype=xp.float32)
    np.testing.assert_allclose(xp.asnumpy(operation(x, y)), expected)


def test_operators_and_broadcast_scalar():
    x = xp.asarray([1, 2], dtype=xp.float32)
    np.testing.assert_array_equal(xp.asnumpy((x + 1) * 2), [4, 6])


@pytest.mark.parametrize("operation,expected", [(xp.less, [True, False]), (xp.equal, [False, True])])
def test_comparisons(operation, expected):
    x = xp.asarray([1, 2], dtype=xp.float32)
    y = xp.asarray([2, 2], dtype=xp.float32)
    np.testing.assert_array_equal(xp.asnumpy(operation(x, y)), expected)
