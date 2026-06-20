import numpy as np
import pytest


def test_real_pyclesperanto_backend_smoke():
    cle = pytest.importorskip("pyclesperanto")
    try:
        cle.select_device()
    except Exception as error:
        pytest.skip(f"no pyclesperanto device available: {error}")
    import oclpy as xp
    xp.set_backend(cle)
    x = xp.asarray([1, 2, 3], dtype=xp.float32)
    np.testing.assert_array_equal(xp.asnumpy(x + 1), [2, 3, 4])
