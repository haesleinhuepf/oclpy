from __future__ import annotations

import warnings

import numpy as np

bool = np.dtype("bool")
int8 = np.dtype("int8")
int16 = np.dtype("int16")
int32 = np.dtype("int32")
uint8 = np.dtype("uint8")
uint16 = np.dtype("uint16")
uint32 = np.dtype("uint32")
float32 = np.dtype("float32")

# 64-bit and boolean dtypes are not natively supported by pyclesperanto.
# They are exposed for Array API namespace compatibility and automatically
# downcast to their nearest supported equivalents with a UserWarning.
int64 = np.dtype("int64")
uint64 = np.dtype("uint64")
float64 = np.dtype("float64")

SUPPORTED_DTYPES = {int8, int16, int32, uint8, uint16, uint32, float32}

# Maps unsupported dtypes to their nearest supported fallback.
DTYPE_FALLBACKS = {
    np.dtype("bool"): np.dtype("int8"),
    np.dtype("int64"): np.dtype("int32"),
    np.dtype("uint64"): np.dtype("uint32"),
    np.dtype("float64"): np.dtype("float32"),
}


def normalize_dtype(dtype):
    return None if dtype is None else np.dtype(dtype)


def require_supported(dtype):
    dtype = np.dtype(dtype)
    if dtype in SUPPORTED_DTYPES:
        return dtype
    fallback = DTYPE_FALLBACKS.get(dtype)
    if fallback is not None:
        warnings.warn(
            f"dtype {dtype} is not supported by pyclesperanto; falling back to {fallback}",
            UserWarning,
            stacklevel=3,
        )
        return fallback
    raise TypeError(f"dtype {dtype} is not supported by pyclesperanto")
