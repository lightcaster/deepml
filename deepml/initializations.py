import numpy as np

def orthogonal(shape, scale=1.1):
    ''' From Lasagne
    '''

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)

    return scale * q[:shape[0], :shape[1]]

def ortho(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(np.float32)


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def uniform(shape, scale=0.1):
    return np.random.uniform(low=-scale, high=scale, size=shape).astype(np.float32)

def glorot_uniform(shape):
    ''' From Keras
    '''

    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))

    return uniform(shape, s)

