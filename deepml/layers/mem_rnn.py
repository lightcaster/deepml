"""
Memory RNN implementation that follows the paper
Mikolov et al - "Learning Longer Memory in Recurrent Neural Networks", 2014.

It doesn't contain any training or optimization routines.
This code just symbolicly specifies network architecture.

(c) Konstantin Selivanov

"""
import numpy as np
import theano
from theano import tensor as T
from deepml.initializations import orthogonal, glorot_uniform
from theano.tensor.nnet import hard_sigmoid

floatX = theano.config.floatX

def mem_rnn(x, n_in, n_hid, n_out, alpha=0.5, iv=0.01):

    x = x.dimshuffle((1,0,2))

    # hidden cells
    #W_hh = np.random.normal(iv, size=(n_hid,n_hid))
    #W_hh *= 1.25 / max(abs(np.linalg.eigvals(W_hh)))
    W_ih = glorot_uniform(shape=(n_in, n_hid))
    W_hh = orthogonal(shape=(n_hid, n_hid))
    W_ho = glorot_uniform(shape=(n_hid, n_out))
    b_hh = np.zeros(n_hid)
    b_ho = np.zeros(n_out)

    #W_ih = np.random.normal(iv, size=(n_in, n_hid))
    #W_ho = np.random.normal(iv, size=(n_hid, n_out))
    #b_hh = np.zeros(n_hid)
    #b_ho = np.zeros(n_out)

    # long-memory cells

    W_is = glorot_uniform(shape=(n_in, n_hid))
    W_so = glorot_uniform(shape=(n_hid, n_out))
    W_sh = orthogonal(shape=(n_hid, n_hid))

    #W_is = np.random.normal(iv, size=(n_in, n_hid))
    #W_so = np.random.normal(iv, size=(n_hid, n_out))
    #W_sh = np.random.normal(iv, size=(n_hid, n_hid))

    h0 = T.alloc(np.cast[floatX](0.), x.shape[1], n_hid)
    s0 = T.alloc(np.cast[floatX](0.), x.shape[1], n_hid)

    params = [W_ih, W_hh, W_ho, W_is, W_so, W_sh, b_hh, b_ho]
    shared_params = [ theano.shared(p.astype(floatX)) for p in params ]

    # refactor
    [W_ih, W_hh, W_ho, W_is, W_so, W_sh, b_hh, b_ho] = shared_params

    #x = T.bmatrix('x')
    #y = T.bscalar('y')

    def forward(x, h_tm1, s_tm1):
        s = (1 - alpha)*T.dot(x, W_is) + alpha*s_tm1
        h = T.nnet.hard_sigmoid (
            T.dot(h_tm1, W_hh) + T.dot(x, W_ih) + T.dot(s, W_sh) + b_hh )
        return h, s

    [hv, sv], _ = theano.scan(
                    fn=forward,
                    sequences=x,
                    outputs_info=[h0,s0]
                )

    out = T.nnet.sigmoid( T.dot(hv[-1], W_ho) + T.dot(sv[-1], W_so) + b_ho )

    # no need this just for last item
    #out = out.dimshuffle((1,0,2))

    return out, shared_params

if __name__ == '__main__':
    pass

