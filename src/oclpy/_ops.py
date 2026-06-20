from __future__ import annotations

import operator
import numpy as np

from ._array import Array, coerce, unwrap, uses_shape_metadata

_OPERATORS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "pow": operator.pow,
    "equal": operator.eq,
    "not_equal": operator.ne,
    "less": operator.lt,
    "less_equal": operator.le,
    "greater": operator.gt,
    "greater_equal": operator.ge,
}


def binary(name, x1, x2):
    if any(uses_shape_metadata(value) for value in (x1, x2)):
        return coerce(_OPERATORS[name](np.asarray(x1), np.asarray(x2)))
    return Array(_OPERATORS[name](unwrap(x1), unwrap(x2)))
