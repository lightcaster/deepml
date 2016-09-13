import numpy as np
import theano
from theano import tensor as T
from deepml.initializations import orthogonal, glorot_uniform
from deepml.utils import floatX, shared_x, shared_zeros
from deepml.activations import hard_sigmoid

floatX = theano.config.floatX

class GridGRU(object):

    def __init__(self, n_in, n_hid, activation=hard_sigmoid,
                truncate_gradient=-1):

        # scaling spectral radius of hidden-to-hidden matrix
        # (optional, but hightly improves the search for a good local minima)

        self.n_in = n_in
        self.n_hid = n_hid
        self.activation = activation

        # update units
        self.W_iz = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hz = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.W_vz = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hz = shared_x(np.zeros(n_hid))

        # reset units

        self.W_ir = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hr = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.W_vr = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hr = shared_x(np.zeros(n_hid))

        # hidden units

        self.W_ih = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hh = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hh = shared_x(np.zeros(n_hid))

        self.params = [
            self.W_iz, self.W_hz, self.W_vz, self.b_hz,
            self.W_ir, self.W_hr, self.W_vr, self.b_hr,
            self.W_ih, self.W_hh, self.b_hh ]


    def apply(self, x, truncate_gradient=-1):
        ''' expect: batch_size, n_in, dim0, dim1 
            output: batch_size, n_hid, dim0, dim1 
        
        '''
        x = x.dimshuffle((2,3,0,1))

        n_in = self.n_in
        n_hid = self.n_hid

        def inner_loop(x, h_tmn, h_tm1):

            # update gate
            z = T.nnet.hard_sigmoid( T.dot(x, self.W_iz) + 
                    T.dot(h_tm1, self.W_hz) + T.dot(h_tmn, self.W_vz) + self.b_hz )

            # reset gate
            r = T.nnet.hard_sigmoid( T.dot(x, self.W_ir) + 
                T.dot(h_tm1, self.W_hr) + T.dot(h_tm1, self.W_vr) + self.b_hr )

            # candidate activation
            c = T.nnet.sigmoid( T.dot(x, self.W_ih) + 
                T.dot(r * 0.5 * (h_tm1 + h_tmn), self.W_hh) + self.b_hh )

            return z * 0.5 * (h_tm1 + h_tmn) + (1 - z)*c

        def outer_loop(row, row_prev):

            H, _ = theano.scan(
                inner_loop,
                outputs_info=T.zeros_like(row_prev[0]),
                sequences=[row, row_prev],
                truncate_gradient=truncate_gradient,
            )

            return H

        out, _ = theano.scan(
                        outer_loop,
                        outputs_info=T.zeros((x.shape[1], x.shape[2], n_hid), dtype='float32'),
                        sequences=[x],
                        truncate_gradient=truncate_gradient,
                )

        # got, <c0, c1, bs, n_hid>
        out = out.dimshuffle((2,3,0,1))

        return out

if __name__ == '__main__':

    # example
    
    bs, n_in, c0, c1 = 16, 3, 50, 50
    n_hid = 32
    ggru = GridGRU(n_in, n_hid)

    x= T.ftensor4('x')
    out = ggru.apply(x)

    o = out.eval({ x: np.ones((bs, n_in, c0, c1), dtype=np.float32) })

