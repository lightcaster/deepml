from deepml.initializations import uniform
import theano
from theano import tensor as T

class Embedding(object):

    def __init__(self, n_in, n_enc):
        # TODO: write me
        self.n_in = n_in
        self.n_enc = n_enc
        self.W = theano.shared(uniform((n_in, n_enc)))
        self.params = [self.W]

    def apply(self, x):
        # batch_size, seq_len

        x_shp = x.shape
        x_enc = self.W[x.flatten()] 
        # check me
        #x_enc = T.reshape(x_enc, x_shp + (self.n_enc,))
        x_enc = T.reshape(x_enc, (x_shp[0], self.n_enc))
        #x_enc = T.reshape(x_enc, (x_shp[0], x_shp[1], shp[2], self.n_enc))
        
        return x_enc
        
'''
def embedding(x, n_in, n_out, weights=None):
    if weights:
        [W] = weights
    else:
        W = theano.shared(uniform((n_in, n_out)))

    out = W[x]

    return out, [W]
'''

