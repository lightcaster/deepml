import numpy as np
import theano
from theano import tensor as T
from utils import shared_zeros

def sgd(cost, params, lr=1e-2, momentum=0.9, nesterov=False):
    ''' SGD solver with optional Nesterov update 

    Parameters
    ----------
    inputs :    list of theano shared variables
    cost :      theano shared variable (scalar)
    params :    list of theano shared variables
    lr :        float, optional
    momentum :  float, optional
    nesterov :  bool, optional

    Returns
    -------
    solver_func : function
                  `solver_func` is a one step of sgd

    '''
    grads = T.grad(cost, params)
    updates = []

    for p, g in zip(params, grads):
        m = shared_zeros(p.get_value(borrow=True).shape)
        m_new = m * momentum - lr * g

        if nesterov:
            p_new = p + momentum * m_new - lr * g
        else:
            p_new = p + m_new
        
        updates.append((m, m_new))
        updates.append((p, p_new))

    return updates

def rmsprop(cost, params, lr=1e-3, rho=0.9, eps=1e-6):
    ''' RMSprop solver '''

    grads = T.grad(cost, params)
    accums = []
    updates = []

    for p in params:
        accums.append(shared_zeros(p.get_value(borrow=True).shape))

    for p, g, a in zip(params, grads, accums):
        a_new = rho * a + (1 - rho) * g ** 2 
        p_new = p - lr * g / T.sqrt(a_new + eps)

        updates.append((a, a_new))
        updates.append((p, p_new))

    return updates

def adagrad(cost, params, lr=1e-2, eps=1e-6, grad_norm=None):
    ''' Adagrad solver 
    
    TODO: check for consistency
    
    '''
    grads = T.grad(cost, params)
    accums = []
    updates = []
    
    for p in params:
        accums.append(shared_zeros(p.get_value(borrow=True).shape))

    for p, g, a in zip(params, grads, accums):
        a_new = a + g ** 2 
        p_new = p - lr * g / T.sqrt(a_new + eps)

        updates.append((a, a_new))
        updates.append((p, p_new))

    return updates

def adadelta(cost, params, grads=None, lr=1.0, rho=0.95, eps=1e-6, grad_norm=None):
    '''
    Adadelta solver.
    Reference: Matthew D. Zeiler, http://arxiv.org/abs/1212.5701

    '''
    if grads is None:
        grads = T.grad(cost, params)

    if grad_norm:
        for i in range(len(grads)):
            g = grads[i]
            norm = T.sqrt((g*g).sum())
            limit = T.cast(grad_norm, 'float32')
            grads[i] = g * T.minimum(1., limit/norm)

    # create a list of variables to store the momentum
    exp_g = []
    exp_x = []

    updates = []

    for p in params:
        exp_g.append(shared_zeros(p.get_value(borrow=True).shape))
        exp_x.append(shared_zeros(p.get_value(borrow=True).shape))

    for p, g, eg, ex in zip(params, grads, exp_g, exp_x):

        eg_new = rho * eg + (1 - rho) * g ** 2

        p_upd = g * T.sqrt(ex + eps) / T.sqrt(eg_new + eps)

        # do we really need this 'lr' parameter?
        p_new = p - lr * p_upd

        ex_new = rho*ex + (1 - rho) * p_upd ** 2

        updates.append((eg, eg_new))
        updates.append((p, p_new))
        updates.append((ex, ex_new))

    return updates

def adadelta_g(grads, params, lr=1.0, rho=0.95, eps=1e-6, grad_norm=None):
    '''
    Adadelta solver.
    Reference: Matthew D. Zeiler, http://arxiv.org/abs/1212.5701

    '''

    if grad_norm:
        for i in range(len(grads)):
            g = grads[i]
            norm = T.sqrt((g*g).sum())
            limit = T.cast(grad_norm, 'float32')
            grads[i] = g * T.minimum(1., limit/norm)

    # create a list of variables to store the momentum
    exp_g = []
    exp_x = []

    updates = []

    for p in params:
        exp_g.append(shared_zeros(p.get_value(borrow=True).shape))
        exp_x.append(shared_zeros(p.get_value(borrow=True).shape))

    for p, g, eg, ex in zip(params, grads, exp_g, exp_x):

        eg_new = rho * eg + (1 - rho) * g ** 2

        p_upd = g * T.sqrt(ex + eps) / T.sqrt(eg_new + eps)

        # do we really need this 'lr' parameter?
        p_new = p - lr * p_upd

        ex_new = rho*ex + (1 - rho) * p_upd ** 2

        updates.append((eg, eg_new))
        updates.append((p, p_new))
        updates.append((ex, ex_new))

    return updates

