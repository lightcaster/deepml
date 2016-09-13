"""
Implementation of the Gated Unit recurrent neural network (GatedRNN).
This imlementations follows the paper Chung at al - "Empirical Evaluation
of Gated Recurrent Neural Networks on Sequence Modeling", 2014

It doesn't contain any training or optimization routines.
This code just symbolicly specifies network architecture.

(c) Konstantin Selivanov

"""
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import hard_sigmoid
from deepml.initializations import orthogonal, glorot_uniform, uniform
from deepml.utils import floatX, shared_x, shared_zeros


class GRU(object):

    def __init__(self, n_in, n_hid, activation=hard_sigmoid):

        self.n_in = n_in
        self.n_hid = n_hid
        self.activation = activation

        # update units
        self.W_iz = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hz = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hz = shared_x(np.zeros(n_hid))

        # reset units
        self.W_ir = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hr = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hr = shared_x(np.zeros(n_hid))

        # hidden units
        self.W_ih = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hh = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hh = shared_x(np.zeros(n_hid))

        self.params = [
            self.W_ih, self.W_hh, 
            self.W_iz, self.W_hz, 
            self.W_ir, self.W_hr, 
            self.b_hh, self.b_hz, self.b_hr]

    def apply(self, x, truncate_gradient=-1):
        ''' x: (batch_size, seq_len, n_hidden) '''

        x = x.dimshuffle((1,0,2))

        # init hidden value
        h0 = T.alloc(np.cast[floatX](0.), x.shape[1], self.n_hid)

        def forward(xz_t, xr_t, xh_t, h_tm1, u_hz, u_hr, u_hh):

            # update gate
            z = self.activation(xz_t + T.dot(h_tm1, u_hz) )
            # reset gate
            r = self.activation(xr_t + T.dot(h_tm1, u_hr) )
            # candidate activation
            c = T.nnet.sigmoid(xh_t + T.dot(r * h_tm1, u_hh))

            return z * h_tm1 + (1 - z) * c

        # optimization: multiply by input before
        # the recurrent steps
        xz = T.dot(x, self.W_iz) + self.b_hz
        xr = T.dot(x, self.W_ir) + self.b_hr
        xh = T.dot(x, self.W_ih) + self.b_hh

        out, _ = theano.scan(
                        fn=forward,
                        sequences=[xz, xr, xh],
                        non_sequences=[self.W_hz, self.W_hr, self.W_hh],
                        outputs_info=h0,
                        truncate_gradient=truncate_gradient,
                    )

        # if you want to get only the last output item
        # take out[-1] in the client code to optimize the computation

        out = out.dimshuffle((1,0,2))

        return out

    def apply_one_step(self, x, h_tm1):
        ''' x: (batch_size, n_hidden) '''

        # update gate
        z = self.activation(
            T.dot(x, self.W_iz) + self.b_hz + T.dot(h_tm1, self.W_hz) )
        # reset gate
        r = self.activation(
            T.dot(x, self.W_ir) + self.b_hr + T.dot(h_tm1, self.W_hr) )
        # candidate activation
        c = T.nnet.sigmoid(
            T.dot(x, self.W_ih) + self.b_hh + T.dot(r * h_tm1, self.W_hh))

        return z * h_tm1 + (1 - z) * c


class StopGRU(object):

    def __init__(self, n_in, n_hid, activation=hard_sigmoid):

        self.n_in = n_in
        self.n_hid = n_hid
        self.activation = activation

        # update units
        self.W_iz = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hz = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hz = shared_x(np.zeros(n_hid))

        # reset units
        self.W_ir = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hr = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hr = shared_x(np.zeros(n_hid))

        # hidden units
        self.W_ih = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hh = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hh = shared_x(np.zeros(n_hid))

        # stop units
        self.W_hs = shared_x(glorot_uniform(shape=(n_hid, 1)))
        self.b_hs = shared_x(np.zeros(1))

        self.params = [
            self.W_ih, self.W_hh, 
            self.W_iz, self.W_hz, 
            self.W_ir, self.W_hr, 
            self.b_hh, self.b_hz, self.b_hr,
            self.W_hs, self.b_hs 
        ]

    def apply(self, x, stop_threshold=0.9, truncate_gradient=-1):
        '''
        For now we can work with batch_size=1 only.
        x: (batch_size, seq_len, n_in)

        '''
        x = x.dimshuffle((1,0,2))

        # init hidden value
        h0 = T.alloc(np.cast[floatX](0.), x.shape[1], self.n_hid)

        def forward(xz_t, xr_t, xh_t, h_tm1, u_hz, u_hr, u_hh):

            # update gate
            z = self.activation(xz_t + T.dot(h_tm1, u_hz) )
            # reset gate
            r = self.activation(xr_t + T.dot(h_tm1, u_hr) )
            # candidate activation
            c = T.nnet.sigmoid(xh_t + T.dot(r * h_tm1, u_hh))

            h = z * h_tm1 + (1 - z) * c
            s = T.nnet.sigmoid(T.dot(self.W_hs, h) + self.b_hs).mean()

            return h, theano.scan_module.until( 
                s > stop_threshold )

        # optimization: multiply by input before
        # the recurrent steps
        xz = T.dot(x, self.W_iz) + self.b_hz
        xr = T.dot(x, self.W_ir) + self.b_hr
        xh = T.dot(x, self.W_ih) + self.b_hh

        out, _ = theano.scan(
                        fn=forward,
                        sequences=[xz, xr, xh],
                        non_sequences=[self.W_hz, self.W_hr, self.W_hh],
                        outputs_info=h0,
                        truncate_gradient=truncate_gradient,
                    )

        # if you want to get only the last output item
        # take out[-1] in the client code to optimize the computation

        out = out.dimshuffle((1,0,2))

        return out

    def apply_one_step(self, x, h_tm1):
        ''' x: (batch_size, n_hidden) '''

        # update gate
        z = self.activation(
            T.dot(x, self.W_iz) + self.b_hz + T.dot(h_tm1, self.W_hz) )
        # reset gate
        r = self.activation(
            T.dot(x, self.W_ir) + self.b_hr + T.dot(h_tm1, self.W_hr) )
        # candidate activation
        c = T.nnet.sigmoid(
            T.dot(x, self.W_ih) + self.b_hh + T.dot(r * h_tm1, self.W_hh))

        return z * h_tm1 + (1 - z) * c

class BatchNormalization(object):
    ''' The most simple recurrent batch normalization implementation ''' 
    def __init__(self, shape, gamma=1e-1, beta=0, epsilon=1e-6):
        self.shape = shape
        self.gamma = gamma
        self.beta = beta
        #self.use_bias = use_bias
        self.epsilon = epsilon

    def apply(self, x, mean=None, var=None):

        mean = x.mean(axis=0)
        var = x.var(axis=0)

        y = theano.tensor.nnet.bn.batch_normalization(
            inputs=x,
            gamma=self.gamma, 
            beta=self.beta,
            mean=T.shape_padleft(mean),
            std=T.shape_padleft(T.sqrt(var + self.epsilon)))

        return y

class GRUBN(object):
    ''' Batch normalized GRU '''

    def __init__(self, n_in, n_hid, activation=hard_sigmoid,
                    gamma=0.1, beta=0., epsilon=1e-6):

        self.n_in = n_in
        self.n_hid = n_hid
        self.activation = activation

        # update units
        self.W_iz = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hz = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hz = shared_x(np.zeros(n_hid))

        # reset units
        self.W_ir = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hr = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hr = shared_x(np.zeros(n_hid))

        # hidden units
        self.W_ih = shared_x(glorot_uniform(shape=(n_in, n_hid)))
        self.W_hh = shared_x(orthogonal(shape=(n_hid, n_hid)))
        self.b_hh = shared_x(np.zeros(n_hid))

        # bn layers
        self.bn_xz = BatchNormalization(gamma, beta, epsilon)
        self.bn_xr = BatchNormalization(gamma, beta, epsilon)

        self.params = [
            self.W_ih, self.W_hh, 
            self.W_iz, self.W_hz, 
            self.W_ir, self.W_hr, 
            self.b_hh, self.b_hz, self.b_hr]

    def apply(self, x, truncate_gradient=-1):
        ''' x: (batch_size, seq_len, n_hidden) '''

        x = x.dimshuffle((1,0,2))

        # init hidden value
        h0 = T.alloc(np.cast[floatX](0.), x.shape[1], self.n_hid)

        def forward(xz_t, xr_t, xh_t, h_tm1, u_hz, u_hr, u_hh):


            # update gate
            #z = self.activation(xz_t + T.dot(h_tm1, u_hz) )
            z = self.activation(self.bn_xz.apply(xz_t) + self.bn_xz.apply(T.dot(h_tm1, u_hz)) )
            # reset gate
            #r = self.activation(xr_t + T.dot(h_tm1, u_hr) )
            r = self.activation(self.bn_xr.apply(xr_t) + self.bn_xr.apply(T.dot(h_tm1, u_hr)) )
            # candidate activation
            c = T.nnet.sigmoid(self.bn_xr.apply(xh_t) + T.dot(r * h_tm1, u_hh))

            return z * h_tm1 + (1 - z) * c

        # optimization: multiply by input before
        # the recurrent steps
        xz = T.dot(x, self.W_iz) + self.b_hz
        xr = T.dot(x, self.W_ir) + self.b_hr
        xh = T.dot(x, self.W_ih) + self.b_hh

        out, _ = theano.scan(
                        fn=forward,
                        sequences=[xz, xr, xh],
                        non_sequences=[self.W_hz, self.W_hr, self.W_hh],
                        outputs_info=h0,
                        truncate_gradient=truncate_gradient,
                    )

        # if you want to get only the last output item
        # take out[-1] in the client code to optimize the computation

        out = out.dimshuffle((1,0,2))

        return out

    def apply_one_step(self, x, h_tm1):
        ''' x: (batch_size, n_hidden) '''

        # update gate
        z = self.activation(
            T.dot(x, self.W_iz) + self.b_hz + T.dot(h_tm1, self.W_hz) )
        # reset gate
        r = self.activation(
            T.dot(x, self.W_ir) + self.b_hr + T.dot(h_tm1, self.W_hr) )
        # candidate activation
        c = T.nnet.sigmoid(
            T.dot(x, self.W_ih) + self.b_hh + T.dot(r * h_tm1, self.W_hh))

        return z * h_tm1 + (1 - z) * c


