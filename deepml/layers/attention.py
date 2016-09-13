import numpy as np
import theano
from theano import tensor as T
from deepml.initializations import orthogonal, glorot_uniform
from deepml.utils import floatX, shared_x, shared_zeros
from deepml.activations import hard_sigmoid, tanh, softmax, sigmoid, relu
from deepml.layers import GRU, StopGRU, Dense, Conv2D, Embedding

from theano.sandbox.rng_mrg import MRG_RandomStreams

srng = MRG_RandomStreams(seed=42)

floatX = theano.config.floatX

class AttentionGRU(object):

    def __init__(self, n_in, n_inner, n_outer):

        self.n_in = n_in
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_inner = GRU(n_in + n_outer, n_inner)
        self.gru_outer = GRU(n_in, n_outer)
        self.d_alpha = Dense(n_inner, 1)

        self.params = self.gru_inner.params + \
            self.gru_outer.params +  self.d_alpha.params
    
    def apply(self, x, n_cycles):
        
        def inner_loop(h, x):

            # x: batch_size, seq_len, n_in
            # h: batch_size, n_outer

            # for rnn attention inner loop  
            # we need to concatenate x with h

            hr = T.extra_ops.repeat(h[:,None,:], x.shape[1], axis=1)
            xh = T.concatenate([x, hr], axis=2)
            
            a = self.gru_inner.apply(xh)

            # alpha: batch_size, seq_len
            alpha = T.nnet.softmax( self.d_alpha.apply(a)[:,:,0] )
            
            # g: batch_size, n_in 
            g = T.sum( x*alpha.dimshuffle(0,1,'x') , axis=1)

            h = self.gru_outer.apply_one_step(g, h)

            return h, alpha


        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)

        (H,A), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,None],
            n_steps=n_cycles,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2)

class AttentionGRUw(object):

    def __init__(self, n_in, n_inner, n_outer, window_size, batch_size):

        self.n_in = n_in
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_inner = GRU(n_in + n_outer, n_inner)
        self.gru_outer = GRU(n_in, n_outer)
        self.d_alpha = Dense(n_inner, 1)

        # sorry, we need it now
        self.batch_size = batch_size

        self.params = self.gru_inner.params + \
            self.gru_outer.params +  self.d_alpha.params

        self.window_size = window_size
    
    def apply(self, x, n_cycles):
        
        def inner_loop(h, p, x):

            # x: batch_size, seq_len, n_in
            # h: batch_size, n_outer

            # for rnn attention inner loop  
            # we need to concatenate x with h

            XW = []

            for i in range(self.batch_size): 
                XW.append( x[i,p[i]:p[i]+self.window_size] )

            xw = T.stack(XW, axis=0)

            hr = T.extra_ops.repeat(h[:,None,:], xw.shape[1], axis=1)
            xh = T.concatenate([xw, hr], axis=2)
            
            a = self.gru_inner.apply(xh)

            # alpha: batch_size, seq_len
            alpha = T.nnet.softmax( self.d_alpha.apply(a)[:,:,0] )

            alpha_full = T.zeros((x.shape[0], x.shape[1]))
            for i in range(self.batch_size):
                alpha_full = T.inc_subtensor(alpha_full[i, p[i]:p[i]+self.window_size], alpha[i])

            # sensitive step - compute next window
            # p - is a scalar1
            p_new = T.cast( 
                T.sum(alpha * T.arange(xw.shape[1]), axis=1), 'int64')    

            #p = T.min( [p + p_new, x.shape[1] - self.window_size] )
            p = p + p_new - self.window_size/2
            p = T.clip(p, 0, x.shape[1] - self.window_size)
            
            # g: batch_size, n_in 
            g = T.sum(xw*alpha.dimshuffle(0,1,'x'), axis=1)
            h = self.gru_outer.apply_one_step(g, h)

            return (h, p, alpha_full) #, theano.scan_module.until( T.any( p >= x.shape[1] ))

        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        p0 = T.alloc(np.cast[np.int64](0.), x.shape[0])

        (H,P,A), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,p0,None],
            n_steps=n_cycles,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2), P

class AttentionARSG(object):

    def __init__(self, n_in, n_inner, n_outer):

        self.n_in = n_in
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_outer = GRU(n_in, n_outer)
        
        self.d_x = Dense(n_in, n_inner)
        self.d_h = Dense(n_outer, n_inner)
        self.d_alpha = Dense(1, n_inner)
        self.d_a = Dense(n_inner, 1)

        self.params = self.gru_outer.params +  self.d_alpha.params + \
                self.d_x.params + self.d_h.params + self.d_a.params
    
    def apply(self, x, n_cycles):

        def inner_loop(h, alpha, x):
            # TODO: convolve alpha

            alpha = alpha[:, :, None]

            # a: bs, seq_len, n_hid
            a = tanh ( self.d_x.apply(x) + self.d_h.apply(h).dimshuffle(0,'x',1) + self.d_alpha.apply(alpha) )

            a =  self.d_a.apply( a ).flatten(2)   # squeeze
            alpha = softmax(a) 

            glimpse = T.sum(x * alpha.dimshuffle(0,1,'x'), axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse, h)

            return h, alpha

        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], x.shape[1])

        (H,A), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,alpha0],
            n_steps=n_cycles,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2)

class AttentionARSGconv(object):

    def __init__(self, n_in, n_inner, n_outer):

        self.n_in = n_in
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_outer = GRU(n_in, n_outer)
        
        self.d_x = Dense(n_in, n_inner)
        self.d_h = Dense(n_outer, n_inner)
        #self.d_alpha = Dense(1, n_inner)
        self.d_a = Dense(n_inner, 1)

        #self.conv = Conv2D(n_inner, 1, 15, 1)
        self.Wconv = theano.shared(np.random.uniform(-0.01, 0.01, size=(10, 1, 15, 2)).astype(np.float32))


        #self.params = self.gru_outer.params + self.d_alpha.params + \
        self.params = self.gru_outer.params + \
                self.d_x.params + self.d_h.params + self.d_a.params + \
                [self.Wconv]
                #self.conv.params
    
    def apply(self, x, n_cycles):

        def inner_loop(h, alpha, x):
            # TODO: convolve alpha

            #alpha = alpha[:, None, :, None]
            alpha = T.reshape(alpha, (
                alpha.shape[0], 1, alpha.shape[1], 1))
            alpha = T.repeat(alpha, 30, axis=3)

            #al = self.conv.apply(alpha, border_mode='valid')
            al = T.nnet.conv2d(self.Wconv, alpha, border_mode='full')
            #alpha = alpha[:,:,7:-7,0]
            # bs, n_hid, seq_len, 1

            # a: bs, seq_len, n_hid
            a = tanh ( self.d_x.apply(x) + self.d_h.apply(h).dimshuffle(0,'x',1) + T.mean(al) ) #  self.d_alpha.apply(alpha) )

            a =  self.d_a.apply( a ).flatten(2)   # squeeze
            alpha = softmax(a) 

            glimpse = T.sum(x * alpha.dimshuffle(0,1,'x'), axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse, h)

            return h, alpha

        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        #alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], x.shape[1])
        alpha0 = T.zeros((x.shape[0], x.shape[1]))

        (H,A), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,alpha0],
            n_steps=n_cycles,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2)


class AttentionARSGy(object):

    def __init__(self, n_in, n_out, n_inner, n_outer):

        self.n_in = n_in
        self.n_out = n_out
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_outer = GRU(n_in, n_outer)
        
        self.d_x = Dense(n_in, n_inner)
        self.d_h = Dense(n_outer, n_inner)
        self.d_alpha = Dense(1, n_inner)
        self.d_a = Dense(n_inner, 1)

        self.d_ya = Dense(n_out, n_inner)
        self.d_yy = Dense(n_out, n_out)
        self.d_hy = Dense(n_outer, n_out)

        self.conv = Conv2D(n_inner, 1, 15, 1)

        self.params = self.gru_outer.params +  self.d_alpha.params + \
                self.d_x.params + self.d_h.params + self.d_a.params + \
                self.d_ya.params + self.d_yy.params + self.d_hy.params # + \
                #self.conv.params
    
    def apply(self, x, n_cycles):

        def inner_loop(h, alpha, y, x):
            # TODO: convolve alpha

            alpha = alpha[:, :, None]
            #alpha = alpha[:, None, :, None]
            #al = self.conv.apply(alpha, border_mode='full')[:,:,7:-7,0]

            # a: bs, seq_len, n_hid
            a = tanh ( self.d_x.apply(x) + self.d_h.apply(h).dimshuffle(0,'x',1) + self.d_alpha.apply(alpha) + self.d_ya.apply(y).dimshuffle(0,'x',1) )

            a =  self.d_a.apply( a ).flatten(2)   # squeeze
            alpha = softmax(a) 

            glimpse = T.sum(x * alpha.dimshuffle(0,1,'x'), axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse, h)

            y = softmax( self.d_hy.apply(h) + self.d_yy.apply(y) )

            return h, alpha, y

        #h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        #alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], x.shape[1])
        #y0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_out)
        alpha0 = T.zeros((x.shape[0], x.shape[1]))
        y0 = T.zeros((x.shape[0], self.n_out))
        h0 = T.zeros((x.shape[0], self.n_outer))

        (H,A,Y), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,alpha0,y0],
            n_steps=n_cycles,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2) , Y.dimshuffle(1,0,2)

class AttentionARSGyy(object):

    def __init__(self, n_in, n_out, n_inner, n_outer):

        self.n_in = n_in
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_outer = GRU(n_in, n_outer)
        
        self.d_x = Dense(n_in, n_inner)
        self.d_h = Dense(n_outer, n_inner)
        self.d_alpha = Dense(1, n_inner)
        self.d_a = Dense(n_inner, 1)
        self.d_ya = Embedding(n_out, n_inner)

        self.params = self.gru_outer.params +  self.d_alpha.params + \
                self.d_x.params + self.d_h.params + self.d_a.params + \
                self.d_ya.params
    
    def apply(self, x, y, n_cycles):

        def inner_loop(yt, h, alpha, x):
            # TODO: convolve alpha

            alpha = alpha[:, :, None]

            # a: bs, seq_len, n_hid
            a = tanh ( self.d_x.apply(x) + self.d_h.apply(h).dimshuffle(0,'x',1) + self.d_alpha.apply(alpha) + self.d_ya.apply(yt).dimshuffle(0,'x',1))

            a =  self.d_a.apply( a ).flatten(2)   # squeeze
            alpha = softmax(a) 

            glimpse = T.sum(x * alpha.dimshuffle(0,1,'x'), axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse, h)

            return h, alpha

        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], x.shape[1])
        y_prev = T.roll(y, shift=1, axis=1).dimshuffle(1,0)
        #y_prev = T.cast(y, 'int32')

        (H,A), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,alpha0],
            n_steps=n_cycles,
            sequences=[y_prev],
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2)

class AttentionARSGw(object):

    def __init__(self, n_in, n_inner, n_outer, window_size, batch_size):

        self.n_in = n_in
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_outer = GRU(n_in, n_outer)
        
        self.d_x = Dense(n_in, n_inner)
        self.d_h = Dense(n_outer, n_inner)
        self.d_alpha = Dense(1, n_inner)
        self.d_a = Dense(n_inner, 1)

        self.window_size = window_size
        self.batch_size = batch_size

        self.params = self.gru_outer.params +  self.d_alpha.params + \
                self.d_x.params + self.d_h.params + self.d_a.params
    
    def apply(self, x, n_cycles):

        def inner_loop(h, p, alpha, x):
            # TODO: convolve alpha

            alpha = alpha[:, :, None]

            XW = []
            for i in range(self.batch_size): 
                XW.append( x[i,p[i]:p[i]+self.window_size] )

            xw = T.stack(XW, axis=0)

            # a: bs, seq_len, n_hid
            a = tanh ( self.d_x.apply(xw) + self.d_h.apply(h).dimshuffle(0,'x',1) + self.d_alpha.apply(alpha) )

            a = self.d_a.apply( a )[:,:,0] #.flatten(2)   # squeeze
            alpha = softmax(a) 

            alpha_full = T.zeros((x.shape[0], x.shape[1]))
            for i in range(self.batch_size):
                alpha_full = T.inc_subtensor(alpha_full[i, p[i]:p[i]+self.window_size], alpha[i])

            p_new = T.cast( T.sum(alpha * T.arange(xw.shape[1]), axis=1), 'int64')    

            pc = p + p_new - self.window_size/2
            p = T.clip(pc, p, x.shape[1] - self.window_size)

            glimpse = T.sum(xw * alpha.dimshuffle(0,1,'x'), axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse, h)

            return h, p, alpha, alpha_full

        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.window_size)
        p0 = T.alloc(np.cast[np.int64](0.), x.shape[0])

        (H,P,A_,A), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,p0,alpha0,None],
            n_steps=n_cycles,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2)

class AttentionARSGwf(object):

    def __init__(self, n_in, n_inner, n_outer, window_size, batch_size):

        self.n_in = n_in
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_outer = GRU(n_in, n_outer)
        
        self.d_x = Dense(n_in*window_size, window_size)
        self.d_h = Dense(n_outer, window_size)
        self.d_alpha = Dense(1, n_inner)
        self.d_a = Dense(n_inner, 1)

        self.window_size = window_size
        self.batch_size = batch_size

        self.params = self.gru_outer.params + \
                self.d_x.params + self.d_h.params # + self.d_a.params
    
    def apply(self, x, n_cycles):

        def inner_loop(h, p, alpha, x):
            # TODO: convolve alpha

            alpha = alpha[:, :, None]

            XW = []
            for i in range(self.batch_size): 
                XW.append( x[i,p[i]:p[i]+self.window_size] )

            xw = T.stack(XW, axis=0) #.flatten(2)

            # a: bs, seq_len, n_hid
            #a = tanh ( self.d_x.apply(xw) + self.d_h.apply(h).dimshuffle(0,'x',1) + self.d_alpha.apply(alpha) )
            a = relu( self.d_x.apply(xw.flatten(2)) + self.d_h.apply(h) )

            #a = self.d_a.apply( a )[:,:,0] #.flatten(2)   # squeeze
            alpha = softmax(a) 
            alpha_full = T.zeros((x.shape[0], x.shape[1]))

            for i in range(self.batch_size):
                alpha_full = T.inc_subtensor(alpha_full[i, p[i]:p[i]+self.window_size], alpha[i])

            p_new = T.cast( T.sum(alpha * T.arange(xw.shape[1]), axis=1), 'int64')    

            pc = p + p_new - self.window_size/2
            p = T.clip(pc, p, x.shape[1] - self.window_size)

            glimpse = T.sum(xw * alpha.dimshuffle(0,1,'x'), axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse, h)

            return h, p, alpha, alpha_full

        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.window_size)
        p0 = T.alloc(np.cast[np.int64](0.), x.shape[0])

        (H,P,A_,A), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,p0,alpha0,None],
            n_steps=n_cycles,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2)



class AttentionARSGwRNN(object):

    def __init__(self, n_in, n_inner, n_outer, window_size, batch_size):

        self.n_in = n_in
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_outer = GRU(n_in, n_outer)
        self.gru_att = GRU(n_inner, 1)
        
        self.d_x = Dense(n_in, n_inner)
        self.d_h = Dense(n_outer, n_inner)
        self.d_alpha = Dense(1, n_inner)
        self.d_a = Dense(n_inner, 1)

        self.window_size = window_size
        self.batch_size = batch_size

        self.params = self.gru_outer.params +  self.d_alpha.params + \
                self.d_x.params + self.d_h.params + self.d_a.params + \
                self.gru_att.params
    
    def apply(self, x, n_cycles):

        def inner_loop(h, p, alpha, x):
            # TODO: convolve alpha

            alpha = alpha[:, :, None]

            XW = []
            for i in range(self.batch_size): 
                XW.append( x[i,p[i]:p[i]+self.window_size] )

            xw = T.stack(XW, axis=0)

            # a: bs, seq_len, n_hid
            #a = self.gru_att.apply ( self.d_x.apply(xw) + self.d_h.apply(h).dimshuffle(0,'x',1) + self.d_alpha.apply(alpha) )

            hr = T.extra_ops.repeat(h[:,None,:], xw.shape[1], axis=1)
            xh = T.concatenate([xw, hr], axis=2)
            
            a = self.gru_att.apply(xh)

            # alpha: batch_size, seq_len
            alpha = T.nnet.softmax( self.d_alpha.apply(a)[:,:,0] )
            
            glimpse = T.sum( x*alpha.dimshuffle(0,1,'x') , axis=1)

            alpha_full = T.zeros((x.shape[0], x.shape[1]))
            for i in range(self.batch_size):
                alpha_full = T.inc_subtensor(alpha_full[i, p[i]:p[i]+self.window_size], alpha[i])

            p_new = T.cast( T.sum(alpha * T.arange(xw.shape[1]), axis=1), 'int64')    

            pc = p + p_new - self.window_size/2
            p = T.clip(pc, p, x.shape[1] - self.window_size)

            glimpse = T.sum(xw * alpha.dimshuffle(0,1,'x'), axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse, h)

            return h, p, alpha, alpha_full

        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.window_size)
        p0 = T.alloc(np.cast[np.int64](0.), x.shape[0])

        (H,P,A_,A), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,p0,alpha0,None],
            n_steps=n_cycles,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2)


class AttentionARSGwfy(object):

    def __init__(self, n_in, n_out, n_inner, n_outer, window_size, batch_size):

        self.n_in = n_in
        self.n_out = n_out
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop
        self.batch_size = batch_size    # REMOVE THIS!

        self.gru_outer = GRU(n_in+n_out, n_outer)
        
        self.d_x = Dense(n_in*window_size, window_size)
        self.d_h = Dense(n_outer, window_size)
        #self.d_alpha = Dense(1, n_inner)
        #self.d_a = Dense(n_inner, 1)
        self.d_ya = Embedding(n_out, n_in)
        self.d_ho = Dense(n_outer, n_out)

        self.window_size = window_size
        self.batch_size = batch_size

        self.params = self.gru_outer.params + \
                self.d_x.params + self.d_h.params + \
                self.d_ho.params

    
    def apply(self, x, y, gamma, n_cycles):

        def inner_loop(y_true, y_tm1, h, p, alpha, x):
            # TODO: convolve alpha

            alpha = alpha[:, :, None]

            XW = []
            for i in range(self.batch_size): 
                XW.append( x[i,p[i]:p[i]+self.window_size] )

            xw = T.stack(XW, axis=0)

            # a: bs, seq_len, n_hid
            # TODO: add previous alpha as a single transform
            a = relu( self.d_x.apply(xw.flatten(2)) + self.d_h.apply(h) )

            alpha = softmax(a) 
            alpha_full = T.zeros((x.shape[0], x.shape[1]))

            for i in range(self.batch_size):
                alpha_full = T.inc_subtensor(alpha_full[i, p[i]:p[i]+self.window_size], alpha[i])

            p_new = T.cast( T.sum(alpha * T.arange(xw.shape[1]), axis=1), 'int64')    

            pc = p + p_new - self.window_size/2
            p = T.clip(pc, p, x.shape[1] - self.window_size)

            glimpse = T.sum(xw * alpha.dimshuffle(0,1,'x'), axis=1)

            # enocode the Y distribution
            y_true_dist = T.extra_ops.to_one_hot(
                y_true, self.n_out, dtype=floatX)

            rmask = srng.binomial(size=(self.batch_size,), p=gamma,
                dtype=floatX).dimshuffle(0,'x')
            y_enc = rmask*y_true_dist + (1-rmask)*y_tm1

            glimpse_y = T.concatenate([glimpse, y_enc], axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse_y, h)

            y_hat = softmax(self.d_ho.apply(h))

            return y_hat, h, p, alpha, alpha_full

        y0 = T.zeros((x.shape[0], self.n_out))
        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.window_size)
        p0 = T.alloc(np.cast[np.int64](0.), x.shape[0])

        y_prev = T.roll(y.dimshuffle(1,0), 1, axis=0)

        (Y,H,P,A_,A), _ = theano.scan(
            inner_loop,
            outputs_info=[y0,h0,p0,alpha0,None],
            n_steps=n_cycles,
            non_sequences=[x],
            sequences=[y_prev],
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), Y.dimshuffle(1,0,2), A.dimshuffle(1,0,2)


class AttentionARSGwfp(object):

    def __init__(self, n_in, n_out, n_inner, n_outer, window_size, batch_size):

        self.n_in = n_in
        self.n_out = n_out
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop
        self.batch_size = batch_size    # REMOVE THIS!

        self.gru_outer = GRU(n_in+n_out, n_outer)
        
        self.d_x = Dense(n_in*window_size, window_size)
        self.d_h = Dense(n_outer, window_size)
        #self.d_alpha = Dense(1, n_inner)
        #self.d_a = Dense(n_inner, 1)
        self.d_ya = Embedding(n_out, n_in)
        self.d_ho = Dense(n_outer, n_out)

        self.window_size = window_size
        self.batch_size = batch_size

        self.params = self.gru_outer.params + \
                self.d_x.params + self.d_h.params + \
                self.d_ho.params

    
    def apply(self, x, n_cycles):

        def inner_loop(y_tm1, h, p, alpha, x):
            # TODO: convolve alpha

            alpha = alpha[:, :, None]

            XW = []
            for i in range(self.batch_size): 
                XW.append( x[i,p[i]:p[i]+self.window_size] )

            xw = T.stack(XW, axis=0)

            # a: bs, seq_len, n_hid
            # TODO: add previous alpha as a single transform
            a = relu( self.d_x.apply(xw.flatten(2)) + self.d_h.apply(h) )

            alpha = softmax(a) 
            alpha_full = T.zeros((x.shape[0], x.shape[1]))

            for i in range(self.batch_size):
                alpha_full = T.inc_subtensor(alpha_full[i, p[i]:p[i]+self.window_size], alpha[i])

            p_new = T.cast( T.sum(alpha * T.arange(xw.shape[1]), axis=1), 'int64')    

            pc = p + p_new# - self.window_size/2
            p = T.clip(pc, p, x.shape[1] - self.window_size)

            glimpse = T.sum(xw * alpha.dimshuffle(0,1,'x'), axis=1)

            glimpse_y = T.concatenate([glimpse, y_tm1], axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse_y, h)

            y_hat = softmax(self.d_ho.apply(h))

            return y_hat, h, p, alpha, alpha_full

        y0 = T.zeros((x.shape[0], self.n_out))
        h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.window_size)
        p0 = T.alloc(np.cast[np.int64](0.), x.shape[0])

        (Y,H,P,A_,A), _ = theano.scan(
            inner_loop,
            outputs_info=[y0,h0,p0,alpha0,None],
            n_steps=n_cycles,
            non_sequences=[x],
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), Y.dimshuffle(1,0,2), A.dimshuffle(1,0,2)



class AttentionARSGNTM(object):

    def __init__(self, n_in, n_out, n_inner, n_outer):

        self.n_in = n_in
        self.n_out = n_out
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_outer = GRU(n_in + n_out, n_outer)
        
        self.d_x = Dense(n_in, n_inner)
        self.d_y = Dense(n_out, n_inner)

        self.d_h_x = Dense(n_outer, n_inner)
        self.d_h_y = Dense(n_outer, n_inner)

        self.d_alpha_x = Dense(1, n_inner)
        self.d_alpha_y = Dense(1, n_inner)

        self.d_a_x = Dense(n_inner, 1)
        self.d_a_y = Dense(n_inner, 1)


        #self.d_c = Dense(n_inner, 1)
        #self.h_c = Dense(n_outer, 1)
        self.h_u = Dense(n_outer, n_out)

        self.params = self.gru_outer.params + \
                self.d_alpha_x.params + \
                self.d_alpha_y.params + \
                self.d_x.params + \
                self.d_y.params + \
                self.d_h_x.params + \
                self.d_h_y.params + \
                self.d_a_x.params + \
                self.d_a_y.params + \
                self.h_u.params 
                #self.d_c.params
                #self.h_c.params
    
    def apply(self, x, y):

        def inner_loop(h, alpha_x, alpha_y, y_tm1, x):
            # TODO: convolve alpha

            # read
            alpha_x = alpha_x[:, :, None]
            # a: bs, x_len, n_hid
            a = tanh ( 
                        self.d_x.apply(x) + \
                        self.d_h_x.apply(h).dimshuffle(0,'x',1) +  \
                        self.d_alpha_x.apply(alpha_x) 
                )

            a =  self.d_a_x.apply( a ).flatten(2)   # squeeze
            alpha_x = softmax(a) 

            glimpse_x = T.sum(x * alpha_x.dimshuffle(0,1,'x'), axis=1)


            # TODO: read-write scheme

            # write

            alpha_y = alpha_y[:, :, None]

            # a: bs, y_len, n_hid
            a =  ( 
                        self.d_y.apply(y_tm1) + \
                        self.d_h_y.apply(h).dimshuffle(0,'x',1) +  \
                        0*self.d_alpha_y.apply(alpha_y) 
                )

            a_att = self.d_a_y.apply( a ).flatten(2)   # squeeze
            #alpha_y = softmax(a_att)
            alpha_y = a_att

            #c = sigmoid(self.d_c.apply(a).flatten(2).mean(axis=1))
            #c = sigmoid(self.h_c.apply(h).flatten()) 
            #c = c.dimshuffle(0,'x','x')

            u = softmax ( self.h_u.apply(h) )

            #y = (1 - c)*y_tm1 + c*(u * alpha_y.dimshuffle(0,1,'x'))

            #y =  c*(u * alpha_y.dimshuffle(0,1,'x'))

            # y: bs, y_len, n_out

            ur = T.repeat(u[:,None,:], y_tm1.shape[1], axis=1)
            y = y_tm1*0.5 + 0.5*(ur * alpha_y.dimshuffle(0,1,'x'))
            #y =  ur * alpha_y.dimshuffle(0,1,'x')

            #y_sh0, y_sh1, y_sh2 = y.shape
            #y = softmax(y.flatten(2))
            #y = y.reshape((y_sh0, y_sh1, y_sh2))

            glimpse_y=u
            glimpse = T.concatenate([glimpse_x, glimpse_y], axis=1) 
            h = self.gru_outer.apply_one_step(glimpse, h)

            return h, alpha_x, alpha_y, y

        h0 = T.alloc(np.cast[floatX](1.), x.shape[0], self.n_outer)
        y0 = T.alloc(np.cast[floatX](1.), y.shape[0], y.shape[1], self.n_out)

        alpha_x0 = T.alloc(np.cast[floatX](1.), x.shape[0], x.shape[1])
        alpha_y0 = T.alloc(np.cast[floatX](1.), y.shape[0], y.shape[1])

        (H,AX,AY,Y), _ = theano.scan(
            inner_loop,
            outputs_info=[h0,alpha_x0,alpha_y0,y0],
            #n_steps=y.shape[1],   # TODO: fix this
            n_steps=30,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), Y[-1], AX.dimshuffle(1,0,2), AY.dimshuffle(1,0,2)

if __name__ == '__main__':
    pass

