import numpy as np
import pytest


class FakeDeviceArray:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.device = "fake:0"
    shape = property(lambda self: self.value.shape)
    dtype = property(lambda self: self.value.dtype)
    def fill(self, value): self.value.fill(value)
    def astype(self, dtype): return type(self)(self.value.astype(dtype))
    def reshape(self, shape): return type(self)(self.value.reshape(shape))
    def sum(self): return self.value.sum()
    def min(self): return self.value.min()
    def max(self): return self.value.max()
    def __getitem__(self, key): return type(self)(self.value[key])
    def _op(self, other, op):
        other = other.value if isinstance(other, FakeDeviceArray) else other
        return type(self)(op(self.value, other))
    __add__ = lambda s, o: s._op(o, np.add)
    __sub__ = lambda s, o: s._op(o, np.subtract)
    __mul__ = lambda s, o: s._op(o, np.multiply)
    __truediv__ = lambda s, o: s._op(o, np.divide)
    __pow__ = lambda s, o: s._op(o, np.power)
    __eq__ = lambda s, o: s._op(o, np.equal)
    __ne__ = lambda s, o: s._op(o, np.not_equal)
    __lt__ = lambda s, o: s._op(o, np.less)
    __le__ = lambda s, o: s._op(o, np.less_equal)
    __gt__ = lambda s, o: s._op(o, np.greater)
    __ge__ = lambda s, o: s._op(o, np.greater_equal)
    __neg__ = lambda s: type(s)(-s.value)


class FakeBackend:
    push = staticmethod(lambda value, **kwargs: FakeDeviceArray(value))
    pull = staticmethod(lambda value: value.value.copy())
    empty = staticmethod(lambda shape, dtype, **kwargs: FakeDeviceArray(np.empty(shape, dtype=dtype)))
    zeros = staticmethod(lambda shape, dtype, **kwargs: FakeDeviceArray(np.zeros(shape, dtype=dtype)))
    ones = staticmethod(lambda shape, dtype, **kwargs: FakeDeviceArray(np.ones(shape, dtype=dtype)))
    transpose_xy = staticmethod(lambda value: FakeDeviceArray(value.value.T))
    multiply_matrix = staticmethod(lambda x, y: FakeDeviceArray(x.value @ y.value))


@pytest.fixture(autouse=True)
def fake_backend():
    import oclpy
    oclpy.set_backend(FakeBackend())
