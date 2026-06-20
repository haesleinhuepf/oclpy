"""Python Array API namespace backed by pyclesperanto."""

from __future__ import annotations

import numpy as _np

from ._array import Array, unwrap, uses_shape_metadata
from ._backend import get_backend, set_backend
from ._dtypes import *
from ._dtypes import normalize_dtype, require_supported

__version__ = "0.1.0"
__array_api_version__ = "2023.12"


def _shape_tuple(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)


def _buffer_shape(shape):
    shape = _shape_tuple(shape)
    if len(shape) <= 3:
        return shape
    size = 1
    for length in shape:
        size *= length
    return (size,)


def _from_host(host, *, dtype, device=None):
    shape = host.shape
    buffer = host.reshape(_buffer_shape(shape))
    return Array(get_backend().push(buffer, dtype=dtype, device=device), shape=shape)


def _logical_shape(shape, size):
    shape = _shape_tuple(shape)
    unknown = None
    known_size = 1
    for index, length in enumerate(shape):
        if length == -1:
            if unknown is not None:
                raise ValueError("can only specify one unknown dimension")
            unknown = index
            continue
        if length < 0:
            raise ValueError("negative dimensions are not allowed")
        known_size *= length
    if unknown is None:
        if known_size != size:
            raise ValueError(f"cannot reshape array of size {size} into shape {shape}")
        return shape
    if known_size == 0 or size % known_size != 0:
        raise ValueError(f"cannot reshape array of size {size} into shape {shape}")
    return shape[:unknown] + (size // known_size,) + shape[unknown + 1:]


def asarray(obj, /, *, dtype=None, device=None, copy=None):
    if isinstance(obj, Array) and dtype is None and copy is not True:
        return obj
    if isinstance(obj, Array):
        obj = asnumpy(obj)
    host = _np.asarray(obj, dtype=normalize_dtype(dtype))
    dtype = require_supported(host.dtype)
    if copy is True:
        host = host.copy()
    return _from_host(host, dtype=dtype, device=device)


def asnumpy(x, /):
    return _np.asarray(x)


def empty(shape, *, dtype=float32, device=None):
    shape = _shape_tuple(shape)
    return Array(get_backend().empty(_buffer_shape(shape), dtype=require_supported(dtype), device=device), shape=shape)


def zeros(shape, *, dtype=float32, device=None):
    shape = _shape_tuple(shape)
    return Array(get_backend().zeros(_buffer_shape(shape), dtype=require_supported(dtype), device=device), shape=shape)


def ones(shape, *, dtype=float32, device=None):
    shape = _shape_tuple(shape)
    return Array(get_backend().ones(_buffer_shape(shape), dtype=require_supported(dtype), device=device), shape=shape)


def full(shape, fill_value, *, dtype=None, device=None):
    dtype = require_supported(dtype or _np.asarray(fill_value).dtype)
    out = empty(shape, dtype=dtype, device=device)
    out._data.fill(fill_value)
    return out


def empty_like(x, *, dtype=None, device=None):
    return empty(x.shape, dtype=dtype or x.dtype, device=device or x.device)


def zeros_like(x, *, dtype=None, device=None):
    return zeros(x.shape, dtype=dtype or x.dtype, device=device or x.device)


def ones_like(x, *, dtype=None, device=None):
    return ones(x.shape, dtype=dtype or x.dtype, device=device or x.device)


def full_like(x, fill_value, *, dtype=None, device=None):
    return full(x.shape, fill_value, dtype=dtype or x.dtype, device=device or x.device)


def arange(start, stop=None, step=1, *, dtype=None, device=None):
    if stop is None:
        start, stop = 0, start
    dtype = require_supported(dtype or _np.float32)
    # pyclesperanto has no range constructor; upload is the only transfer here.
    return asarray(_np.arange(start, stop, step, dtype=dtype), device=device)


def astype(x, dtype, /, *, copy=True, device=None):
    dtype = require_supported(dtype)
    if not copy and dtype == x.dtype and (device is None or device == x.device):
        return x
    return Array(unwrap(x).astype(dtype), shape=x.shape)


def _binary(name):
    return lambda x1, x2, /: getattr(Array._coerce(x1), f"__{name}__")(x2)


add = _binary("add")
subtract = _binary("sub")
multiply = _binary("mul")
divide = _binary("truediv")
pow = _binary("pow")
equal = _binary("eq")
not_equal = _binary("ne")
less = _binary("lt")
less_equal = _binary("le")
greater = _binary("gt")
greater_equal = _binary("ge")


def negative(x, /):
    return -x


def positive(x, /):
    return +x


def matmul(x1, x2, /):
    if x1.ndim != 2 or x2.ndim != 2:
        return asarray(_np.matmul(asnumpy(x1), asnumpy(x2)))
    backend = get_backend()
    function = getattr(backend, "multiply_matrix", None)
    if function is None:
        raise NotImplementedError("pyclesperanto does not expose multiply_matrix")
    return Array(function(unwrap(x1), unwrap(x2)))


def permute_dims(x, axes, /):
    axes = tuple(axes)
    if x.ndim == 2 and axes == (1, 0) and not uses_shape_metadata(x):
        return Array(get_backend().transpose_xy(unwrap(x)))
    return asarray(_np.transpose(asnumpy(x), axes))


def reshape(x, shape, /, *, copy=None):
    shape = _logical_shape(shape, x.size)
    method = getattr(unwrap(x), "reshape", None)
    if method is not None and len(shape) <= 3 and not uses_shape_metadata(x):
        return Array(method(shape))
    return Array(unwrap(x), shape=shape)


def _reduction(method):
    def apply(x, /, *, axis=None, dtype=None, keepdims=False):
        if axis is not None or keepdims:
            raise NotImplementedError("axis reductions are not exposed by pyclesperanto")
        result = getattr(unwrap(x), method)()
        return asarray(result, dtype=dtype or x.dtype)
    return apply


sum = _reduction("sum")
max = _reduction("max")
min = _reduction("min")


def result_type(*arrays_and_dtypes):
    values = [value.dtype if isinstance(value, Array) else value for value in arrays_and_dtypes]
    return _np.result_type(*values)


def can_cast(from_, to, /):
    source = from_.dtype if isinstance(from_, Array) else from_
    return _np.can_cast(source, to)


def finfo(type, /):
    return _np.finfo(type)


def iinfo(type, /):
    return _np.iinfo(type)
