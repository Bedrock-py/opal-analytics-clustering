#****************************************************************
# Copyright (c) 2015, Georgia Tech Research Institute
# All rights reserved.
#
# This unpublished material is the property of the Georgia Tech
# Research Institute and is protected under copyright law.
# The methods and techniques described herein are considered
# trade secrets and/or confidential. Reproduction or distribution,
# in whole or in part, is forbidden except by the express written
# permission of the Georgia Tech Research Institute.
#****************************************************************/

from analytics.utils import * 
import numpy as np
import time, os, json
import scipy.sparse as sps
import numpy.linalg as nla
import math
import scipy.optimize as opt
from numpy import random
import scipy.linalg as sla


   
#k means tempalte
class JinguNMF(Algorithm):
    def __init__(self):
        super(JinguNMF, self).__init__()
        self.parameters = ['numClusters']
        self.inputs = ['matrix.csv']
        self.outputs = ['assignments.csv']
        self.name ='NMF'
        self.type = 'Clustering'
        self.description = 'Performs Nonnegative Matrix Factorization (in Python) on the input dataset.'
        self.parameters_spec = [ { "name" : "Clusters", "attrname" : "numClusters", "value" : 3, "type" : "input", "step": 1, "max": 15, "min": 1 }]
    
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')      
        print 'matrixpath', filepath['matrix.csv']['rootdir'] + 'matrix.csv'
        W, H, info = NMF().run(self.inputData, int(self.numClusters))
                
        self.clusters = np.argmax(W, axis = 1).astype(int)
        self.results = {'assignments.csv': self.clusters}


class NMF_Base(object):

    """ Base class for NMF algorithms

    Specific algorithms need to be implemented by deriving from this class.
    """
    default_max_iter = 100
    default_max_time = np.inf

    def __init__(self):
        raise NotImplementedError(
            'NMF_Base is a base class that cannot be instantiated')

    def set_default(self, default_max_iter, default_max_time):
        self.default_max_iter = default_max_iter
        self.default_max_time = default_max_time

    def run(self, A, k, init=None, max_iter=None, max_time=None, verbose=0):
        """ Run a NMF algorithm

        Parameters
        ----------
        A : numpy.array or scipy.sparse matrix, shape (m,n)
        k : int - target lower rank

        Optional Parameters
        -------------------
        init : (W_init, H_init) where
                    W_init is numpy.array of shape (m,k) and
                    H_init is numpy.array of shape (n,k).
                    If provided, these values are used as initial values for NMF iterations.
        max_iter : int - maximum number of iterations.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds.
                    If not provided, default maximum for each algorithm is used.
        verbose : int - 0 (default) - No debugging information is collected, but
                                    input and output information is printed on screen.
                        -1 - No debugging information is collected, and
                                    nothing is printed on screen.
                        1 (debugging/experimental purpose) - History of computation is
                                        returned. See 'rec' variable.
                        2 (debugging/experimental purpose) - History of computation is
                                        additionally printed on screen.
        Returns
        -------
        (W, H, rec)
        W : Obtained factor matrix, shape (m,k)
        H : Obtained coefficient matrix, shape (n,k)
        rec : dict - (debugging/experimental purpose) Auxiliary information about the execution
        """
        info = {'k': k,
                'alg': str(self.__class__),
                'A_dim_1': A.shape[0],
                'A_dim_2': A.shape[1],
                'A_type': str(A.__class__),
                'max_iter': max_iter if max_iter is not None else self.default_max_iter,
                'verbose': verbose,
                'max_time': max_time if max_time is not None else self.default_max_time}
        if init != None:
            W = init[0].copy()
            H = init[1].copy()
            info['init'] = 'user_provided'
        else:
            W = random.rand(A.shape[0], k)
            H = random.rand(A.shape[1], k)
            info['init'] = 'uniform_random'

        if verbose >= 0:
            print '[NMF] Running: '
            print json.dumps(info, indent=4, sort_keys=True)

        norm_A = norm_fro(A)
        total_time = 0

        if verbose >= 1:
            his = {'iter': [], 'elapsed': [], 'rel_error': []}

        start = time.time()
        # algorithm-specific initilization
        (W, H) = self.initializer(W, H)

        for i in range(1, info['max_iter'] + 1):
            start_iter = time.time()
            # algorithm-specific iteration solver
            (W, H) = self.iter_solver(A, W, H, k, i)
            elapsed = time.time() - start_iter

            if verbose >= 1:
                rel_error = norm_fro_err(A, W, H, norm_A) / norm_A
                his['iter'].append(i)
                his['elapsed'].append(elapsed)
                his['rel_error'].append(rel_error)
                if verbose >= 2:
                    print 'iter:' + str(i) + ', elapsed:' + str(elapsed) + ', rel_error:' + str(rel_error)

            total_time += elapsed
            if total_time > info['max_time']:
                break

        W, H, weights = normalize_column_pair(W, H)

        final = {}
        final['norm_A'] = norm_A
        final['rel_error'] = norm_fro_err(A, W, H, norm_A) / norm_A
        final['iterations'] = i
        final['elapsed'] = time.time() - start

        rec = {'info': info, 'final': final}
        if verbose >= 1:
            rec['his'] = his

        if verbose >= 0:
            print '[NMF] Completed: '
            print json.dumps(final, indent=4, sort_keys=True)
        return (W, H, rec)

    def run_repeat(self, A, k, num_trial, max_iter=None, max_time=None, verbose=0):
        """ Run an NMF algorithm several times with random initial values 
            and return the best result in terms of the Frobenius norm of
            the approximation error matrix

        Parameters
        ----------
        A : numpy.array or scipy.sparse matrix, shape (m,n)
        k : int - target lower rank
        num_trial : int number of trials

        Optional Parameters
        -------------------
        max_iter : int - maximum number of iterations for each trial.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds for each trial.
                    If not provided, default maximum for each algorithm is used.
        verbose : int - 0 (default) - No debugging information is collected, but
                                    input and output information is printed on screen.
                        -1 - No debugging information is collected, and
                                    nothing is printed on screen.
                        1 (debugging/experimental purpose) - History of computation is
                                        returned. See 'rec' variable.
                        2 (debugging/experimental purpose) - History of computation is
                                        additionally printed on screen.
        Returns
        -------
        (W, H, rec)
        W : Obtained factor matrix, shape (m,k)
        H : Obtained coefficient matrix, shape (n,k)
        rec : dict - (debugging/experimental purpose) Auxiliary information about the execution
        """
        for t in xrange(num_trial):
            if verbose >= 0:
                print '[NMF] Running the {0}/{1}-th trial ...'.format(t + 1, num_trial)
            this = self.run(A, k, verbose=(-1 if verbose is 0 else verbose))
            if t == 0:
                best = this
            else:
                if this[2]['final']['rel_error'] < best[2]['final']['rel_error']:
                    best = this
        if verbose >= 0:
            print '[NMF] Best result is as follows.'
            print json.dumps(best[2]['final'], indent=4, sort_keys=True)
        return best

    def iter_solver(self, A, W, H, k, it):
        raise NotImplementedError

    def initializer(self, W, H):
        return (W, H)


class NMF_ANLS_BLOCKPIVOT(NMF_Base):

    """ NMF algorithm: ANLS with block principal pivoting

    J. Kim and H. Park, Fast nonnegative matrix factorization: An active-set-like method and comparisons,
    SIAM Journal on Scientific Computing, 
    vol. 33, no. 6, pp. 3261-3281, 2011.
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        Sol, info = nnlsm_blockpivot(W, A, init=H.T)
        H = Sol.T
        Sol, info = nnlsm_blockpivot(H, A.T, init=W.T)
        W = Sol.T
        return (W, H)


class NMF_ANLS_AS_NUMPY(NMF_Base):

    """ NMF algorithm: ANLS with scipy.optimize.nnls solver
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        if not sps.issparse(A):
            for j in xrange(0, H.shape[0]):
                res = opt.nnls(W, A[:, j])
                H[j, :] = res[0]
        else:
            for j in xrange(0, H.shape[0]):
                res = opt.nnls(W, A[:, j].toarray()[:, 0])
                H[j, :] = res[0]

        if not sps.issparse(A):
            for j in xrange(0, W.shape[0]):
                res = opt.nnls(H, A[j, :])
                W[j, :] = res[0]
        else:
            for j in xrange(0, W.shape[0]):
                res = opt.nnls(H, A[j, :].toarray()[0,:])
                W[j, :] = res[0]
        return (W, H)


class NMF_ANLS_AS_GROUP(NMF_Base):

    """ NMF algorithm: ANLS with active-set method and column grouping

    H. Kim and H. Park, Nonnegative matrix factorization based on alternating nonnegativity 
    constrained least squares and active set method, SIAM Journal on Matrix Analysis and Applications, 
    vol. 30, no. 2, pp. 713-730, 2008.
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        if it == 1:
            Sol, info = nnlsm_activeset(W, A)
            H = Sol.T
            Sol, info = nnlsm_activeset(H, A.T)
            W = Sol.T
        else:
            Sol, info = nnlsm_activeset(W, A, init=H.T)
            H = Sol.T
            Sol, info = nnlsm_activeset(H, A.T, init=W.T)
            W = Sol.T
        return (W, H)


class NMF_HALS(NMF_Base):

    """ NMF algorithm: Hierarchical alternating least squares

    A. Cichocki and A.-H. Phan, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
    IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences,
    vol. E92-A, no. 3, pp. 708-721, 2009.
    """

    def __init__(self, default_max_iter=100, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)

    def initializer(self, W, H):
        W, H, weights = normalize_column_pair(W, H)
        return W, H

    def iter_solver(self, A, W, H, k, it):
        AtW = A.T.dot(W)
        WtW = W.T.dot(W)
        for kk in xrange(0, k):
            temp_vec = H[:, kk] + AtW[:, kk] - H.dot(WtW[:, kk])
            H[:, kk] = np.maximum(temp_vec, self.eps)

        AH = A.dot(H)
        HtH = H.T.dot(H)
        for kk in xrange(0, k):
            temp_vec = W[:, kk] * HtH[kk, kk] + AH[:, kk] - W.dot(HtH[:, kk])
            W[:, kk] = np.maximum(temp_vec, self.eps)
            ss = nla.norm(W[:, kk])
            if ss > 0:
                W[:, kk] = W[:, kk] / ss

        return (W, H)


class NMF_MU(NMF_Base):

    """ NMF algorithm: Multiplicative updating 

    Lee and Seung, Algorithms for non-negative matrix factorization, 
    Advances in Neural Information Processing Systems, 2001, pp. 556-562.
    """

    def __init__(self, default_max_iter=500, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        AtW = A.T.dot(W)
        HWtW = H.dot(W.T.dot(W)) + self.eps
        H = H * AtW
        H = H / HWtW

        AH = A.dot(H)
        WHtH = W.dot(H.T.dot(H)) + self.eps
        W = W * AH
        W = W / WHtH

        return (W, H)


class NMF(NMF_ANLS_BLOCKPIVOT):

    """ Default NMF algorithm: NMF_ANLS_BLOCKPIVOT
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)


def _mmio_example(m=100, n=100, k=10):
    print '\nTesting mmio read and write ...\n'
    import scipy.io.mmio as mmio

    W_org = random.rand(m, k)
    H_org = random.rand(n, k)
    X = W_org.dot(H_org.T)
    X[random.rand(n, k) < 0.5] = 0
    X_sparse = sps.csr_matrix(X)

    filename = '_temp_mmio.mtx'
    mmio.mmwrite(filename, X_sparse)
    A = mmio.mmread(filename)

    alg = NMF_ANLS_BLOCKPIVOT()
    rslt = alg.run(X_sparse, k, max_iter=50)


def _compare_nmf(m=300, n=300, k=10):
    from pylab import plot, show, legend, xlabel, ylabel

    W_org = random.rand(m, k)
    H_org = random.rand(n, k)
    A = W_org.dot(H_org.T)

    print '\nComparing NMF algorithms ...\n'

    names = [NMF_MU, NMF_HALS, NMF_ANLS_BLOCKPIVOT,
             NMF_ANLS_AS_NUMPY, NMF_ANLS_AS_GROUP]
    iters = [2000, 1000, 100, 100, 100]
    labels = ['mu', 'hals', 'anls_bp', 'anls_as_numpy', 'anls_as_group']
    styles = ['-x', '-o', '-+', '-s', '-D']

    results = []
    init_val = (random.rand(m, k), random.rand(n, k))

    for i in range(len(names)):
        alg = names[i]()
        results.append(
            alg.run(A, k, init=init_val, max_iter=iters[i], verbose=1))

    for i in range(len(names)):
        his = results[i][2]['his']
        plot(np.cumsum(his['elapsed']), his['rel_error'],
             styles[i], label=labels[i])

    xlabel('time (sec)')
    ylabel('relative error')
    legend()
    show()


def _test_nmf(m=300, n=300, k=10):
    W_org = random.rand(m, k)
    H_org = random.rand(n, k)
    A = W_org.dot(H_org.T)

    alg_names = [NMF_ANLS_BLOCKPIVOT, NMF_ANLS_AS_GROUP,
                 NMF_ANLS_AS_NUMPY, NMF_HALS, NMF_MU]
    iters = [50, 50, 50, 500, 1000]

    print '\nTesting with a dense matrix...\n'
    for alg_name, i in zip(alg_names, iters):
        alg = alg_name()
        rslt = alg.run(A, k, max_iter=i)

    print '\nTesting with a sparse matrix...\n'
    A_sparse = sps.csr_matrix(A)
    for alg_name, i in zip(alg_names, iters):
        alg = alg_name()
        rslt = alg.run(A_sparse, k, max_iter=i)


if __name__ == '__main__':
    _test_nmf()
    _mmio_example()

    # To see an example of comparisons of NMF algorithms, execute
    # _compare_nmf() with X-window enabled.
    # _compare_nmf()




def norm_fro(X):
    """ Compute the Frobenius norm of a matrix

    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix

    Returns
    -------
    float
    """
    if sps.issparse(X):     # scipy.sparse array
        return math.sqrt(X.multiply(X).sum())
    else:                   # numpy array
        return nla.norm(X)


def norm_fro_err(X, W, H, norm_X):
    """ Compute the approximation error in Frobeinus norm

    norm(X - W.dot(H.T)) is efficiently computed based on trace() expansion 
    when W and H are thin.

    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix, shape (m,n)
    W : numpy.array, shape (m,k)
    H : numpy.array, shape (n,k)
    norm_X : precomputed norm of X

    Returns
    -------
    float
    """
    sum_squared = norm_X * norm_X - 2 * np.trace(H.T.dot(X.T.dot(W))) \
        + np.trace((W.T.dot(W)).dot(H.T.dot(H)))
    return math.sqrt(np.maximum(sum_squared, 0))


def column_norm(X, by_norm='2'):
    """ Compute the norms of each column of a given matrix

    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix

    Optional Parameters
    -------------------
    by_norm : '2' for l2-norm, '1' for l1-norm.
              Default is '2'.

    Returns
    -------
    numpy.array
    """
    if sps.issparse(X):
        if by_norm == '2':
            norm_vec = np.sqrt(X.multiply(X).sum(axis=0))
        elif by_norm == '1':
            norm_vec = X.sum(axis=0)
        return np.asarray(norm_vec)[0]
    else:
        if by_norm == '2':
            norm_vec = np.sqrt(np.sum(X * X, axis=0))
        elif by_norm == '1':
            norm_vec = np.sum(X, axis=0)
        return norm_vec


def normalize_column_pair(W, H, by_norm='2'):
    """ Column normalization for a matrix pair 

    Scale the columns of W and H so that the columns of W have unit norms and 
    the product W.dot(H.T) remains the same.  The normalizing coefficients are 
    also returned.

    Side Effect
    -----------
    W and H given as input are changed and returned.

    Parameters
    ----------
    W : numpy.array, shape (m,k)
    H : numpy.array, shape (n,k)

    Optional Parameters
    -------------------
    by_norm : '1' for normalizing by l1-norm, '2' for normalizing by l2-norm.
              Default is '2'.

    Returns
    -------
    ( W, H, weights )
    W, H : normalized matrix pair
    weights : numpy.array, shape k 
    """
    norms = column_norm(W, by_norm=by_norm)

    toNormalize = norms > 0
    W[:, toNormalize] = W[:, toNormalize] / norms[toNormalize]
    H[:, toNormalize] = H[:, toNormalize] * norms[toNormalize]

    weights = np.ones(norms.shape)
    weights[toNormalize] = norms[toNormalize]
    return (W, H, weights)


def normalize_column(X, by_norm='2'):
    """ Column normalization

    Scale the columns of X so that they have unit l2-norms.
    The normalizing coefficients are also returned.

    Side Effect
    -----------
    X given as input are changed and returned

    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix

    Returns
    -------
    ( X, weights )
    X : normalized matrix
    weights : numpy.array, shape k 
    """
    if sps.issparse(X):
        weights = column_norm(X, by_norm)
        # construct a diagonal matrix
        dia = [1.0 / w if w > 0 else 1.0 for w in weights]
        N = X.shape[1]
        r = np.arange(N)
        c = np.arange(N)
        mat = sps.coo_matrix((dia, (r, c)), shape=(N, N))
        Y = X.dot(mat)
        return (Y, weights)
    else:
        norms = column_norm(X, by_norm)
        toNormalize = norms > 0
        X[:, toNormalize] = X[:, toNormalize] / norms[toNormalize]
        weights = np.ones(norms.shape)
        weights[toNormalize] = norms[toNormalize]
        return (X, weights)


def sparse_remove_row(X, to_remove):
    """ Delete rows from a sparse matrix

    Parameters
    ----------
    X : scipy.sparse matrix
    to_remove : a list of row indices to be removed.

    Returns
    -------
    Y : scipy.sparse matrix
    """
    if not sps.isspmatrix_lil(X):
        X = X.tolil()

    to_keep = [i for i in xrange(0, X.shape[0]) if i not in to_remove]
    Y = sps.vstack([X.getrowview(i) for i in to_keep])
    return Y


def sparse_remove_column(X, to_remove):
    """ Delete columns from a sparse matrix

    Parameters
    ----------
    X : scipy.sparse matrix
    to_remove : a list of column indices to be removed.

    Returns
    -------
    Y : scipy.sparse matrix
    """
    B = sparse_remove_row(X.transpose().tolil(), to_remove).tocoo().transpose()
    return B

if __name__ == '__main__':
    print '\nTesting norm_fro_err() ...'
    X = np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5]])
    W = np.array([[1.0], [2.0]])
    H = np.array([[1.0], [1.0], [1.0]])
    norm_X_fro = norm_fro(X)

    val1 = norm_fro(X - W.dot(H.T))
    val2 = norm_fro_err(X, W, H, norm_X_fro)
    print 'OK' if val1 == val2 else 'Fail'

    print '\nTesting column_norm() ...'
    X = np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5]])
    val1 = column_norm(X, by_norm='2')
    val2 = np.sqrt(np.array([4 + 9, 25, 1.5 * 1.5]))
    print 'OK' if np.allclose(val1, val2) else 'Fail'

    print '\nTesting normalize_column_pair() ...'
    W = np.array([[1.0, -2.0], [2.0, 3.0]])
    H = np.array([[-0.5, 1.0], [1.0, 2.0], [1.0, 0.0]])
    val1 = column_norm(W, by_norm='2')
    val3 = W.dot(H.T)
    W1, H1, weights = normalize_column_pair(W, H, by_norm='2')
    val2 = column_norm(W1, by_norm='2')
    val4 = W1.dot(H1.T)
    print 'OK' if np.allclose(val1, weights) else 'Fail'
    print 'OK' if np.allclose(val2, np.array([1.0, 1.0])) else 'Fail'
    print 'OK' if np.allclose(val3, val4) else 'Fail'

    print '\nTesting normalize_column() ...'
    X = np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5]])
    val1 = column_norm(X, by_norm='2')
    (X1, weights) = normalize_column(X, by_norm='2')
    val2 = column_norm(X1, by_norm='2')
    print 'OK' if np.allclose(val2, np.array([1.0, 1.0, 1.0])) else 'Fail'
    print 'OK' if np.allclose(val1, weights) else 'Fail'
    print 'OK' if np.allclose(X.shape, X1.shape) else 'Fail'

    X = sps.csr_matrix(np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5]]))
    val1 = column_norm(X, by_norm='2')
    (X1, weights) = normalize_column(X, by_norm='2')
    val2 = column_norm(X1, by_norm='2')
    print 'OK' if np.allclose(val2, np.array([1.0, 1.0, 1.0])) else 'Fail'
    print 'OK' if np.allclose(val1, weights) else 'Fail'
    print 'OK' if np.allclose(X.shape, X1.shape) else 'Fail'

    print '\nTesting sparse_remove_row() ...'
    X = sps.csr_matrix(
        np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5], [0.5, -2.0, 2.5]]))
    X1 = sparse_remove_row(X, [1]).todense()
    val1 = np.array([[2.0, 5.0, 0.0], [0.5, -2.0, 2.5]])
    print 'OK' if np.allclose(X1, val1) else 'Fail'

    print '\nTesting sparse_remove_column() ...'
    X = sps.csr_matrix(
        np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5], [0.5, -2.0, 2.5]]))
    X1 = sparse_remove_column(X, [1]).todense()
    val1 = np.array([[2.0, 0.0], [-3.0, 1.5], [0.5, 2.5]])
    print 'OK' if np.allclose(X1, val1) else 'Fail'

def nnlsm_blockpivot(A, B, is_input_prod=False, init=None):
    """ Nonnegativity-constrained least squares with block principal pivoting method and column grouping

    Solves min ||AX-B||_2^2 s.t. X >= 0 element-wise.

    J. Kim and H. Park, Fast nonnegative matrix factorization: An active-set-like method and comparisons,
    SIAM Journal on Scientific Computing, 
    vol. 33, no. 6, pp. 3261-3281, 2011.

    Parameters
    ----------
    A : numpy.array, shape (m,n)
    B : numpy.array or scipy.sparse matrix, shape (m,k)

    Optional Parameters
    -------------------
    is_input_prod : True/False. -  If True, the A and B arguments are interpreted as
            AtA and AtB, respectively. Default is False.
    init: numpy.array, shape (n,k). - If provided, init is used as an initial value for the algorithm.
            Default is None.

    Returns
    -------
    X, (success, Y, num_cholesky, num_eq, num_backup)
    X : numpy.array, shape (n,k) - solution
    success : True/False - True if the solution is found. False if the algorithm did not terminate
            due to numerical errors.
    Y : numpy.array, shape (n,k) - Y = A.T * A * X - A.T * B
    num_cholesky : int - the number of Cholesky factorizations needed
    num_eq : int - the number of linear systems of equations needed to be solved
    num_backup: int - the number of appearances of the back-up rule. See SISC paper for details.
    """
    if is_input_prod:
        AtA = A
        AtB = B
    else:
        AtA = A.T.dot(A)
        if sps.issparse(B):
            AtB = B.T.dot(A)
            AtB = AtB.T
        else:
            AtB = A.T.dot(B)

    (n, k) = AtB.shape
    MAX_ITER = n * 5

    if init != None:
        PassSet = init > 0
        X, num_cholesky, num_eq = normal_eq_comb(AtA, AtB, PassSet)
        Y = AtA.dot(X) - AtB
    else:
        X = np.zeros([n, k])
        Y = -AtB
        PassSet = np.zeros([n, k], dtype=bool)
        num_cholesky = 0
        num_eq = 0

    p_bar = 3
    p_vec = np.zeros([k])
    p_vec[:] = p_bar
    ninf_vec = np.zeros([k])
    ninf_vec[:] = n + 1
    not_opt_set = np.logical_and(Y < 0, -PassSet)
    infea_set = np.logical_and(X < 0, PassSet)

    not_good = np.sum(not_opt_set, axis=0) + np.sum(infea_set, axis=0)
    not_opt_colset = not_good > 0
    not_opt_cols = not_opt_colset.nonzero()[0]

    big_iter = 0
    num_backup = 0
    success = True
    while not_opt_cols.size > 0:
        big_iter += 1
        if MAX_ITER > 0 and big_iter > MAX_ITER:
            success = False
            break

        cols_set1 = np.logical_and(not_opt_colset, not_good < ninf_vec)
        temp1 = np.logical_and(not_opt_colset, not_good >= ninf_vec)
        temp2 = p_vec >= 1
        cols_set2 = np.logical_and(temp1, temp2)
        cols_set3 = np.logical_and(temp1, -temp2)

        cols1 = cols_set1.nonzero()[0]
        cols2 = cols_set2.nonzero()[0]
        cols3 = cols_set3.nonzero()[0]

        if cols1.size > 0:
            p_vec[cols1] = p_bar
            ninf_vec[cols1] = not_good[cols1]
            true_set = np.logical_and(not_opt_set, np.tile(cols_set1, (n, 1)))
            false_set = np.logical_and(infea_set, np.tile(cols_set1, (n, 1)))
            PassSet[true_set] = True
            PassSet[false_set] = False
        if cols2.size > 0:
            p_vec[cols2] = p_vec[cols2] - 1
            temp_tile = np.tile(cols_set2, (n, 1))
            true_set = np.logical_and(not_opt_set, temp_tile)
            false_set = np.logical_and(infea_set, temp_tile)
            PassSet[true_set] = True
            PassSet[false_set] = False
        if cols3.size > 0:
            for col in cols3:
                candi_set = np.logical_or(
                    not_opt_set[:, col], infea_set[:, col])
                to_change = np.max(candi_set.nonzero()[0])
                PassSet[to_change, col] = -PassSet[to_change, col]
                num_backup += 1

        (X[:, not_opt_cols], temp_cholesky, temp_eq) = normal_eq_comb(
            AtA, AtB[:, not_opt_cols], PassSet[:, not_opt_cols])
        num_cholesky += temp_cholesky
        num_eq += temp_eq
        X[abs(X) < 1e-12] = 0
        Y[:, not_opt_cols] = AtA.dot(X[:, not_opt_cols]) - AtB[:, not_opt_cols]
        Y[abs(Y) < 1e-12] = 0

        not_opt_mask = np.tile(not_opt_colset, (n, 1))
        not_opt_set = np.logical_and(
            np.logical_and(not_opt_mask, Y < 0), -PassSet)
        infea_set = np.logical_and(
            np.logical_and(not_opt_mask, X < 0), PassSet)
        not_good = np.sum(not_opt_set, axis=0) + np.sum(infea_set, axis=0)
        not_opt_colset = not_good > 0
        not_opt_cols = not_opt_colset.nonzero()[0]

    return X, (success, Y, num_cholesky, num_eq, num_backup)


def nnlsm_activeset(A, B, overwrite=False, is_input_prod=False, init=None):
    """ Nonnegativity-constrained least squares with active-set method and column grouping

    Solves min ||AX-B||_2^2 s.t. X >= 0 element-wise.

    Algorithm of this routine is close to the one presented in the following paper but
    is different in organising inner- and outer-loops:
    M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450

    Parameters
    ----------
    A : numpy.array, shape (m,n)
    B : numpy.array or scipy.sparse matrix, shape (m,k)

    Optional Parameters
    -------------------
    is_input_prod : True/False. -  If True, the A and B arguments are interpreted as
            AtA and AtB, respectively. Default is False.
    init: numpy.array, shape (n,k). - If provided, init is used as an initial value for the algorithm.
            Default is None.

    Returns
    -------
    X, (success, Y, num_cholesky, num_eq, num_backup)
    X : numpy.array, shape (n,k) - solution
    success : True/False - True if the solution is found. False if the algorithm did not terminate
            due to numerical errors.
    Y : numpy.array, shape (n,k) - Y = A.T * A * X - A.T * B
    num_cholesky : int - the number of Cholesky factorizations needed
    num_eq : int - the number of linear systems of equations needed to be solved
    """
    if is_input_prod:
        AtA = A
        AtB = B
    else:
        AtA = A.T.dot(A)
        if sps.issparse(B):
            AtB = B.T.dot(A)
            AtB = AtB.T
        else:
            AtB = A.T.dot(B)

    (n, k) = AtB.shape
    MAX_ITER = n * 5
    num_cholesky = 0
    num_eq = 0
    not_opt_set = np.ones([k], dtype=bool)

    if overwrite:
        X, num_cholesky, num_eq = normal_eq_comb(AtA, AtB)
        PassSet = X > 0
        not_opt_set = np.any(X < 0, axis=0)
    elif init != None:
        X = init
        X[X < 0] = 0
        PassSet = X > 0
    else:
        X = np.zeros([n, k])
        PassSet = np.zeros([n, k], dtype=bool)

    Y = np.zeros([n, k])
    opt_cols = (-not_opt_set).nonzero()[0]
    not_opt_cols = not_opt_set.nonzero()[0]

    Y[:, opt_cols] = AtA.dot(X[:, opt_cols]) - AtB[:, opt_cols]

    big_iter = 0
    success = True
    while not_opt_cols.size > 0:
        big_iter += 1
        if MAX_ITER > 0 and big_iter > MAX_ITER:
            success = False
            break

        (Z, temp_cholesky, temp_eq) = normal_eq_comb(
            AtA, AtB[:, not_opt_cols], PassSet[:, not_opt_cols])
        num_cholesky += temp_cholesky
        num_eq += temp_eq

        Z[abs(Z) < 1e-12] = 0

        infea_subset = Z < 0
        temp = np.any(infea_subset, axis=0)
        infea_subcols = temp.nonzero()[0]
        fea_subcols = (-temp).nonzero()[0]

        if infea_subcols.size > 0:
            infea_cols = not_opt_cols[infea_subcols]

            (ix0, ix1_subsub) = infea_subset[:, infea_subcols].nonzero()
            ix1_sub = infea_subcols[ix1_subsub]
            ix1 = not_opt_cols[ix1_sub]

            X_infea = X[(ix0, ix1)]

            alpha = np.zeros([n, len(infea_subcols)])
            alpha[:] = np.inf
            alpha[(ix0, ix1_subsub)] = X_infea / (X_infea - Z[(ix0, ix1_sub)])
            min_ix = np.argmin(alpha, axis=0)
            min_vals = alpha[(min_ix, range(0, alpha.shape[1]))]

            X[:, infea_cols] = X[:, infea_cols] + \
                (Z[:, infea_subcols] - X[:, infea_cols]) * min_vals
            X[(min_ix, infea_cols)] = 0
            PassSet[(min_ix, infea_cols)] = False

        elif fea_subcols.size > 0:
            fea_cols = not_opt_cols[fea_subcols]

            X[:, fea_cols] = Z[:, fea_subcols]
            Y[:, fea_cols] = AtA.dot(X[:, fea_cols]) - AtB[:, fea_cols]

            Y[abs(Y) < 1e-12] = 0

            not_opt_subset = np.logical_and(
                Y[:, fea_cols] < 0, -PassSet[:, fea_cols])
            new_opt_cols = fea_cols[np.all(-not_opt_subset, axis=0)]
            update_cols = fea_cols[np.any(not_opt_subset, axis=0)]

            if update_cols.size > 0:
                val = Y[:, update_cols] * -PassSet[:, update_cols]
                min_ix = np.argmin(val, axis=0)
                PassSet[(min_ix, update_cols)] = True

            not_opt_set[new_opt_cols] = False
            not_opt_cols = not_opt_set.nonzero()[0]

    return X, (success, Y, num_cholesky, num_eq)



def normal_eq_comb(AtA, AtB, PassSet=None):
    """ Solve many systems of linear equations using combinatorial grouping.

    M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450

    Parameters
    ----------
    AtA : numpy.array, shape (n,n)
    AtB : numpy.array, shape (n,k)

    Returns
    -------
    (Z,num_cholesky,num_eq)
    Z : numpy.array, shape (n,k) - solution
    num_cholesky : int - the number of unique cholesky decompositions done
    num_eq: int - the number of systems of linear equations solved
    """
    num_cholesky = 0
    num_eq = 0
    if AtB.size == 0:
        Z = np.zeros([])
    elif (PassSet == None) or np.all(PassSet):
        Z = nla.solve(AtA, AtB)
        num_cholesky = 1
        num_eq = AtB.shape[1]
    else:
        Z = np.zeros(AtB.shape)
        if PassSet.shape[1] == 1:
            if np.any(PassSet):
                cols = PassSet.nonzero()[0]
                Z[cols] = nla.solve(AtA[np.ix_(cols, cols)], AtB[cols])
                num_cholesky = 1
                num_eq = 1
        else:
            #
            # Both _column_group_loop() and _column_group_recursive() work well.
            # Based on preliminary testing,
            # _column_group_loop() is slightly faster for tiny k(<10), but
            # _column_group_recursive() is faster for large k's.
            #
            grps = _column_group_recursive(PassSet)
            for gr in grps:
                cols = PassSet[:, gr[0]].nonzero()[0]
                if cols.size > 0:
                    ix1 = np.ix_(cols, gr)
                    ix2 = np.ix_(cols, cols)
                    #
                    # scipy.linalg.cho_solve can be used instead of numpy.linalg.solve.
                    # For small n(<200), numpy.linalg.solve appears faster, whereas
                    # for large n(>500), scipy.linalg.cho_solve appears faster.
                    # Usage example of scipy.linalg.cho_solve:
                    # Z[ix1] = sla.cho_solve(sla.cho_factor(AtA[ix2]),AtB[ix1])
                    #
                    Z[ix1] = nla.solve(AtA[ix2], AtB[ix1])
                    num_cholesky += 1
                    num_eq += len(gr)
                    num_eq += len(gr)
    return Z, num_cholesky, num_eq


def _column_group_loop(B):
    """ Given a binary matrix, find groups of the same columns
        with a looping strategy

    Parameters
    ----------
    B : numpy.array, True/False in each element

    Returns
    -------
    A list of arrays - each array contain indices of columns that are the same.
    """
    initial = [np.arange(0, B.shape[1])]
    before = initial
    after = []
    for i in range(0, B.shape[0]):
        all_ones = True
        vec = B[i]
        for cols in before:
            if len(cols) == 1:
                after.append(cols)
            else:
                all_ones = False
                subvec = vec[cols]
                trues = subvec.nonzero()[0]
                falses = (-subvec).nonzero()[0]
                if trues.size > 0:
                    after.append(cols[trues])
                if falses.size > 0:
                    after.append(cols[falses])
        before = after
        after = []
        if all_ones:
            break
    return before


def _column_group_recursive(B):
    """ Given a binary matrix, find groups of the same columns
        with a recursive strategy

    Parameters
    ----------
    B : numpy.array, True/False in each element

    Returns
    -------
    A list of arrays - each array contain indices of columns that are the same.
    """
    initial = np.arange(0, B.shape[1])
    return [a for a in column_group_sub(B, 0, initial) if len(a) > 0]


def column_group_sub(B, i, cols):
    vec = B[i][cols]
    if len(cols) <= 1:
        return [cols]
    if i == (B.shape[0] - 1):
        col_trues = cols[vec.nonzero()[0]]
        col_falses = cols[(-vec).nonzero()[0]]
        return [col_trues, col_falses]
    else:
        col_trues = cols[vec.nonzero()[0]]
        col_falses = cols[(-vec).nonzero()[0]]
        after = column_group_sub(B, i + 1, col_trues)
        after.extend(column_group_sub(B, i + 1, col_falses))
    return after


def _test_column_grouping(m=10, n=5000, num_repeat=5, verbose=False):
    print '\nTesting column_grouping ...'
    A = np.array([[True, False, False, False, False],
                  [True, True, False, True, True]])
    grps1 = _column_group_loop(A)
    grps2 = _column_group_recursive(A)
    grps3 = [np.array([0]),
             np.array([1, 3, 4]),
             np.array([2])]
    print 'OK' if all([np.array_equal(a, b) for (a, b) in zip(grps1, grps2)]) else 'Fail'
    print 'OK' if all([np.array_equal(a, b) for (a, b) in zip(grps1, grps3)]) else 'Fail'

    for i in xrange(0, num_repeat):
        A = np.random.rand(m, n)
        B = A > 0.5
        start = time.time()
        grps1 = _column_group_loop(B)
        elapsed_loop = time.time() - start
        start = time.time()
        grps2 = _column_group_recursive(B)
        elapsed_recursive = time.time() - start
        if verbose:
            print 'Loop     :', elapsed_loop
            print 'Recursive:', elapsed_recursive
        print 'OK' if all([np.array_equal(a, b) for (a, b) in zip(grps1, grps2)]) else 'Fail'
    # sorted_idx = np.concatenate(grps)
    # print B
    # print sorted_idx
    # print B[:,sorted_idx]
    return


def _test_normal_eq_comb(m=10, k=3, num_repeat=5):
    print '\nTesting normal_eq_comb() ...'
    for i in xrange(0, num_repeat):
        A = np.random.rand(2 * m, m)
        X = np.random.rand(m, k)
        C = (np.random.rand(m, k) > 0.5)
        X[-C] = 0
        B = A.dot(X)
        B = A.T.dot(B)
        A = A.T.dot(A)
        Sol, a, b = normal_eq_comb(A, B, C)
        print 'OK' if np.allclose(X, Sol) else 'Fail'
    return


def _test_nnlsm():
    print '\nTesting nnls routines ...'
    m = 100
    n = 10
    k = 200
    rep = 5

    for r in xrange(0, rep):
        A = np.random.rand(m, n)
        X_org = np.random.rand(n, k)
        X_org[np.random.rand(n, k) < 0.5] = 0
        B = A.dot(X_org)
        # B = np.random.rand(m,k)
        # A = np.random.rand(m,n/2)
        # A = np.concatenate((A,A),axis=1)
        # A = A + np.random.rand(m,n)*0.01
        # B = np.random.rand(m,k)

        import time
        start = time.time()
        C1, info = nnlsm_blockpivot(A, B)
        elapsed2 = time.time() - start
        rel_norm2 = nla.norm(C1 - X_org) / nla.norm(X_org)
        print 'nnlsm_blockpivot:    ', 'OK  ' if info[0] else 'Fail',\
            'elapsed:{0:.4f} error:{1:.4e}'.format(elapsed2, rel_norm2)

        start = time.time()
        C2, info = nnlsm_activeset(A, B)
        num_backup = 0
        elapsed1 = time.time() - start
        rel_norm1 = nla.norm(C2 - X_org) / nla.norm(X_org)
        print 'nnlsm_activeset:     ', 'OK  ' if info[0] else 'Fail',\
            'elapsed:{0:.4f} error:{1:.4e}'.format(elapsed1, rel_norm1)

        import scipy.optimize as opt
        start = time.time()
        C3 = np.zeros([n, k])
        for i in xrange(0, k):
            res = opt.nnls(A, B[:, i])
            C3[:, i] = res[0]
        elapsed3 = time.time() - start
        rel_norm3 = nla.norm(C3 - X_org) / nla.norm(X_org)
        print 'scipy.optimize.nnls: ', 'OK  ',\
            'elapsed:{0:.4f} error:{1:.4e}'.format(elapsed3, rel_norm3)

        if num_backup > 0:
            break
        if rel_norm1 > 10e-5 or rel_norm2 > 10e-5 or rel_norm3 > 10e-5:
            break
        print ''

