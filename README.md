# oclpy

`oclpy` is an experimental, GPU-first implementation of the
[Python Array API standard](https://data-apis.org/array-api/latest/) using
[`pyclesperanto`](https://github.com/clEsperanto/pyclesperanto) as its device
backend.

```bash
pip install oclpy
```

```python
import oclpy as xp

x = xp.asarray([[1, 2], [3, 4]], dtype=xp.float32)
y = xp.ones((2, 2), dtype=xp.float32)
print(xp.add(x, y))
```

The initial release implements creation, elementwise arithmetic and
comparisons, dtype handling, reductions, matrix multiplication, and basic
manipulation. Operations stay on the selected pyclesperanto device; conversion
to NumPy is explicit through `xp.asnumpy()`.

## Development

```bash
pip install -e ".[test]"
pytest
```

Tests use a small in-memory pyclesperanto-compatible backend, so the core API
can be validated on machines without a GPU. `tests/test_pyclesperanto_smoke.py`
is the real-device integration test and skips when no backend/device is
available.
