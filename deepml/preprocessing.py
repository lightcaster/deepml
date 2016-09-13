from skimage import transform
from skimage.filter import gaussian_filter
import numpy as np
import pylab as pl
from deepml.layers import whiten_2d

import theano
from theano import tensor as T

def gaussian_blur(x, sigma=0.5, mode='reflect'):
    '''
    Assume:
        bc01 for 'x'

    '''
    bx = np.zeros_like(x)

    for i in range(len(bx)):
        blured = gaussian_filter(x[i].transpose(1,2,0), sigma=sigma, mode=mode)
        bx[i] = blured.transpose(2,0,1).astype(x.dtype)
    return bx

def rand_flip_xy(x, y):
    '''
    Randomly flip axes for both items and labels.

    Assume:
        bc01 for 'x'
        b01  for 'y'

    '''
    rx = np.zeros_like(x)
    ry = np.zeros_like(y)

    flips = np.random.choice((-1,1),size=(rx.shape[0],2))

    for i in xrange(len(x)):
        fx, fy = flips[i]
        rx[i] = x[i][:,::fx,::fy]
        ry[i] = y[i][::fx,::fy]

    return rx, ry

def rand_flip_x(x):
    '''
    Randomly flip axes for both items and labels.

    Assume:
        bc01 for 'x'
        b01  for 'y'

    '''
    rx = np.zeros_like(x)

    flips = np.random.choice((-1,1),size=(rx.shape[0],2))

    for i in xrange(len(x)):
        fx, fy = flips[i]
        rx[i] = x[i][:,::fx,::fy]

    return rx



def rand_rotate_xy(x, y, max_angle=360, resize=False, mode='reflect'):
    '''
    Rotate both items and labels.

    Assume:
        bc01 for 'x'
        b01  for 'y'

    '''
    assert len(x.shape) == 4
    assert len(y.shape) == 3

    rx = np.zeros_like(x)
    ry = np.zeros_like(y)

    for i in xrange(len(x)):
        # choose a random angle
        angle = np.random.randint(0, max_angle)

        rimg = transform.rotate(
            x[i].transpose(1,2,0), angle, resize=resize, mode=mode)
        rx[i] = rimg.transpose(2,0,1).astype(x.dtype)

        ry[i] = transform.rotate(
            y[i].astype(np.float32), angle, resize=resize, mode=mode)
        ry[i] = (ry[i] > 0).astype(y.dtype)

        '''
        f, (a0, a1) = pl.subplots(1,2)

        a0.imshow(rx[i].transpose(1,2,0))
        a1.imshow(ry[i])
        pl.show()
        '''

    return rx, ry

def whitening_transform(x, W, m):
    x_ = T.ftensor4('x')
    whiten = theano.function(inputs=[x_], outputs=whiten_2d(x_,W,m))

    xw = []
    for i in range(len(x)):
        xw.append(whiten(x[i:i+1])[0])

    return np.array(xw, dtype=x.dtype)

def shuffle_xy(x, y):
    ''' Shuffle X,Y (no in-place) '''
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]
