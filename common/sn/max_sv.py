import chainer
from chainer import cuda
from chainer.functions.math import sum
from chainer.functions.connection import linear
from chainer.functions.array import transpose


def _l2normalize(v, eps=1e-12):
    return v / (((v ** 2).sum()) ** 0.5 + eps)


def max_singular_value(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter
    """
    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(xp.dot(_u, W.data), eps=1e-12)
        _u = _l2normalize(xp.dot(_v, W.data.transpose()), eps=1e-12)
    sigma = sum.sum(linear.linear(_u, transpose.transpose(W)) * _v)
    return sigma, _u, _v
