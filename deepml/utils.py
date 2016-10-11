import theano
import numpy as np

floatX = theano.config.floatX

def shared_x(x, dtype=floatX, name=None, target=None):
    if target:
        return theano.shared(
            np.asarray(x, dtype=dtype), name=name, target=target)
    return theano.shared(np.asarray(x, dtype=dtype), name=name)

def shared_zeros(shape, dtype=floatX, name=None, target=None):
    if target:
        return shared_x(
            np.zeros(shape), dtype=dtype, name=name, target=target)
    return shared_x(np.zeros(shape), dtype=dtype, name=name)

def chunker(x, y, size):
    assert len(x) == len(y)
    return ((x[pos:pos + size], y[pos:pos + size]) for pos in xrange(0, len(x), size))

def copy_params(src, dest):
    """ Set source parameters to destination list. 
    The variables can be placed on different GPUs. 

    dest:       list of shared variables 
    targets:    list of shared variables

    """
    if len(dest) != len(src):
        raise ValueError(
            "mismatch: got %d parameters to set %d parameters" %
            (len(targets), len(dest)))

    for d, s in zip(dest, src):
        d_shp, s_shp = (d.get_value(borrow=True).shape,
            s.get_value(borrow=True).shape)

        if  d_shp != s_shp:
            raise ValueError("mismatch: got parameter with shape %r"
                "to set parameter with shape %r" % (
                t_shp, d_shp))

        d.set_value(s.get_value())


