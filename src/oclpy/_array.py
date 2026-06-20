from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from ._backend import get_backend
from ._dtypes import require_supported


class Array:
    """Array API object owning a pyclesperanto device array."""

    __array_priority__ = 1000

    def __init__(self, data: Any):
        self._data = data

    @property
    def shape(self):
        return tuple(self._data.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        result = 1
        for item in self.shape:
            result *= item
        return result

    @property
    def dtype(self):
        return np.dtype(self._data.dtype)

    @property
    def device(self):
        return getattr(self._data, "device", None)

    @property
    def T(self):
        from . import permute_dims

        if self.ndim != 2:
            raise ValueError("x.T is only defined for two-dimensional arrays")
        return permute_dims(self, (1, 0))

    def __array_namespace__(self, *, api_version=None):
        if api_version not in (None, "2023.12", "2024.12"):
            raise ValueError(f"unsupported Array API version: {api_version}")
        import oclpy

        return oclpy

    def __array__(self, dtype=None, copy=None):
        result = get_backend().pull(self._data)
        if dtype is not None:
            result = np.asarray(result, dtype=dtype)
        if copy is True:
            result = result.copy()
        return result

    def __bool__(self):
        if self.size != 1:
            raise ValueError("the truth value of an array with more than one element is ambiguous")
        return bool(np.asarray(self).item())

    def __iter__(self) -> Iterator["Array"]:
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        return (Array(item) for item in self._data)

    def __getitem__(self, key):
        result = self._data[key]
        return Array(result) if hasattr(result, "shape") else result

    def __repr__(self):
        return f"oclpy.Array({np.asarray(self)!r}, device={self.device!r})"

    def astype(self, dtype, *, copy=True, device=None):
        from . import astype

        return astype(self, dtype, copy=copy, device=device)

    def _binary(self, other, name):
        from ._ops import binary

        return binary(name, self, other)

    __add__ = lambda self, other: self._binary(other, "add")
    __radd__ = __add__
    __sub__ = lambda self, other: self._binary(other, "subtract")
    __rsub__ = lambda self, other: Array._coerce(other)._binary(self, "subtract")
    __mul__ = lambda self, other: self._binary(other, "multiply")
    __rmul__ = __mul__
    __truediv__ = lambda self, other: self._binary(other, "divide")
    __rtruediv__ = lambda self, other: Array._coerce(other)._binary(self, "divide")
    __pow__ = lambda self, other: self._binary(other, "pow")
    __eq__ = lambda self, other: self._binary(other, "equal")
    __ne__ = lambda self, other: self._binary(other, "not_equal")
    __lt__ = lambda self, other: self._binary(other, "less")
    __le__ = lambda self, other: self._binary(other, "less_equal")
    __gt__ = lambda self, other: self._binary(other, "greater")
    __ge__ = lambda self, other: self._binary(other, "greater_equal")
    __neg__ = lambda self: Array(-self._data)
    __pos__ = lambda self: self

    @staticmethod
    def _coerce(value):
        from . import asarray

        return value if isinstance(value, Array) else asarray(value)


def wrap(data):
    return data if isinstance(data, Array) else Array(data)


def unwrap(data):
    return data._data if isinstance(data, Array) else data
