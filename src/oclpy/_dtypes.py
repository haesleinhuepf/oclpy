from __future__ import annotations

import numpy as np

bool = np.dtype("bool")
int8 = np.dtype("int8")
int16 = np.dtype("int16")
int32 = np.dtype("int32")
uint8 = np.dtype("uint8")
uint16 = np.dtype("uint16")
uint32 = np.dtype("uint32")
float32 = np.dtype("float32")

# Exposed for namespace compatibility, but rejected before device allocation
# because pyclesperanto does not currently support them.
int64 = np.dtype("int64")
uint64 = np.dtype("uint64")
float64 = np.dtype("float64")

SUPPORTED_DTYPES = {int8, int16, int32, uint8, uint16, uint32, float32}


def normalize_dtype(dtype):
    return None if dtype is None else np.dtype(dtype)


def require_supported(dtype):
    dtype = np.dtype(dtype)
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"dtype {dtype} is not supported by pyclesperanto")
    return dtype
