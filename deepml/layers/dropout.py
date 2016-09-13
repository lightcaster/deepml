import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as T
import theano

floatX = theano.config.floatX

rng = np.random.RandomState(42)
srng = RandomStreams(rng.randint(2**30))

def dropout(w, rate, mode): 
    ''' 
    Mode is a switch (symbolic constant) that indicates
    training (0) or testing (1) mode.

    '''
    w_do = T.switch(
        mode,
        w * (1 - rate),
        w * srng.binomial(size=w.shape, n=1, p=1-rate, dtype=floatX),
    )

    return w_do

def dropconnect(w, rate, mode):
    raise ValueError('Not implemented')

