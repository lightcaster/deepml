'''
Compute whitening matrix on a dataset.

'''
import numpy as np
import pylab as pl
import scipy.linalg
import cPickle as pickle
import os

from sklearn.feature_extraction import image
from scipy.misc import imread

def whiten(X, bias=0.1, n_comp=0, norm_mean=True):
    _mean = np.mean(X, axis=0)

    # extract mean patch
    if norm_mean:
        X -= _mean

    sigma = np.dot(X.T, X)/X.shape[0]
    eigs, eigv = scipy.linalg.eigh(sigma)

    if n_comp:
        eigs = eigs[:n_comp]
        eigv = eigv[:, :n_comp]

    T = eigv * 1./np.sqrt(eigs + bias)
    T_full = np.dot(T, eigv.T)

    '''
    x = np.dot(X, T)
    pl.imshow(np.dot(x.T, x)/x.shape[0])
    pl.show()
    '''

    return _mean, T_full


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Learn the whitening transform')

    parser.add_argument('-s', '--size', type=int, required=True,
        help='Patch size')
    parser.add_argument('-n', '--num_of_patches', type=int, default=10,
        help='Number of patches per image')
    parser.add_argument('-d', '--dataset', type=str, required=True,
        help='Training dataset as numpy array')
    parser.add_argument('-o', '--output_file', type=str, required=True,
        help='Output file to store the parameters of transformation')
    parser.add_argument('-b', '--bias', type=float, required=True,
        help='Bias.')

    args = parser.parse_args()

    # get some data (mnist digits)
    ps = args.size
    max_patches = args.num_of_patches

    npz = np.load(args.dataset)

    train_x = npz['train_x']

    # assuming 'bc01' or 'b01' order
    if len(train_x.shape) == 4:
        n_ch = train_x.shape[1]
    elif len(train_x.shape) == 3:
        n_ch = 1

    if len(train_x.shape) < 4:
        train_x = tran_x[:,None,:,:]

    train_x = train_x.transpose(0,2,3,1)
    print train_x.shape

    patches = image.PatchExtractor((ps,ps), max_patches=max_patches,
        random_state=42).transform(train_x)

    # fix for PatchExtractor flattening
    if len(patches.shape) == 3:
        # add channel axis
        patches = patches[:,None,:,:]
    else:
        patches = patches.transpose(0,3,1,2)  # to bc01

    patches = patches.reshape((-1, n_ch*ps*ps))

    mean,T = whiten(patches, bias=args.bias)
    T.shape = (n_ch,ps*ps,-1)
    T = T[:,T.shape[1]/2]

    kernel = T.reshape((n_ch,n_ch,ps,ps)).astype(np.float32)
    mean = mean.reshape((n_ch,ps,ps)).astype(np.float32)

    print 'Saving to:', args.output_file
    np.savez(args.output_file, kernel=kernel, mean=mean)

