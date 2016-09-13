import theano
import numpy as np

floatX = theano.config.floatX

def shared_x(x, dtype=floatX, name=None):
    return theano.shared(np.asarray(x, dtype=dtype), name=name)

def shared_zeros(shape, dtype=floatX, name=None):
    return shared_x(np.zeros(shape), dtype=dtype, name=name)

def chunker(x, y, size):
    assert len(x) == len(y)
    return ((x[pos:pos + size], y[pos:pos + size]) for pos in xrange(0, len(x), size))
