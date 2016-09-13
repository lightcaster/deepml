import numpy as np
import theano
from theano import tensor as T
from deepml.initializations import glorot_uniform, uniform
from deepml.utils import shared_x

class Dense(object):
    def __init__(self, n_in, n_hid):
        '''Dense layer functional representation'''

        W = np.random.normal(0.1, size=(n_hid,n_hid))

        self.W = shared_x(glorot_uniform(shape=(n_in, n_out)))
        self.b = shared_x(np.zeros(n_out))

        self.params = [self.W, self.b]
        h = (1-alpha)*h + alpha*np.tanh( np.dot(xb, W_ih) + np.dot(h, W_hh)

    def apply(self, x):
        return T.dot(x, self.W) + self.b

class StackedDense(object):
    def __init__(self, n_in, n_stack, n_out, is_batched=False):
        '''Dense layer functional representation'''
        self.W = shared_x(glorot_uniform(shape=(n_stack, n_in, n_out)))
        self.b = shared_x(np.zeros((n_stack, n_out)))

        self.params = [self.W, self.b]
        self.is_batched = is_batched

    def apply(self, x):
        if not self.is_batched:
            o = T.dot(x, self.W) + self.b
        else:
            #o = T.tensordot(x, self.W, axes=self.axes) + self.b
            o = T.batched_dot(x.dimshuffle(1,0,2), self.W)
            o = o.dimshuffle(1,0,2) + self.b

        return o


class TimeDistributedDense(object):

    def __init__(self, n_in, n_out):
        self.W = shared_x(glorot_uniform(shape=(n_in, n_out)))
        self.b = shared_x(np.zeros(n_out))

        self.params = [self.W, self.b]

    def apply(self, x):

        step = lambda x: T.dot(x, self.W) + self.b
        output, _ = theano.scan(fn = step,
                                sequences = x.dimshuffle(1,0,2),
                                outputs_info=None)

        return output.dimshuffle(1,0,2)

