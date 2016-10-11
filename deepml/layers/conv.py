import numpy as np
import theano
import theano.tensor as T
#from theano.tensor.signal import downsample
from deepml.initializations import glorot_uniform

floatX = theano.config.floatX

class Conv2D(object):

    def __init__(self, n_filters, stack_size, n_rows, n_cols, target=None):

        W = glorot_uniform(
            shape=(n_filters, stack_size, n_rows, n_cols)).astype(floatX)
        b = np.zeros((n_filters,)).astype(floatX)

        if target:
            self.W = theano.shared(W, target=target)
            self.b = theano.shared(b, target=target)
        else:
            self.W = theano.shared(W)
            self.b = theano.shared(b)

        self.params = [self.W, self.b]

    def apply(self, x, border_mode='valid', subsample=(1,1)):

        output = T.nnet.conv2d(
            x, 
            self.W, 
            filter_shape=self.W.get_value(borrow=True).shape,
            border_mode=border_mode, 
            #subsample=subsample
        ) + self.b.dimshuffle('x',0,'x','x')

        return output

def whiten_2d(x, W, m):

    W = theano.shared(W)
    m = theano.shared(m)

    # TODO: make a reflection border mode
    mean =  T.dot(m.flatten(), W.flatten(2).T)
    out = T.nnet.conv2d(x, W, border_mode='full') - mean.dimshuffle('x', 0, 'x', 'x')

    # crop the edges
    mid = W.shape[2] // 2
    out = out[:,:,mid:-mid,mid:-mid]

    return out

def batch_norm(x, eps=1e-3):
    m = T.mean(x, axis=0)
    v = T.var(x, axis=0)
    
    xn = (x - m) / T.sqrt(v + eps)

    return xn 

