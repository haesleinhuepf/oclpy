from __future__ import annotations

import operator

from ._array import Array, unwrap

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
    return Array(_OPERATORS[name](unwrap(x1), unwrap(x2)))
