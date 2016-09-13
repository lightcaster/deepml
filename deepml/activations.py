from theano import tensor as T

def relu(x):
    return (x + T.abs_(x)) / 2.0

def softmax_lastaxis(x):
    # for sequence of distributions
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)

def crossentropy_lastaxes(yhat, y):
    # for sequence of distributions/targets
    return -(y * T.log(yhat)).sum(axis=yhat.ndim - 1)

# aliases

tanh =          T.tanh
sigmoid =       T.nnet.sigmoid
hard_sigmoid =  T.nnet.hard_sigmoid
softmax =       T.nnet.softmax

