# OLD version of presumably clean script pushed to repo 'CNTF'
# coupled network tensor factorization of brain networks and microbiome
import numpy as np
from tensorly.tenalg import multi_mode_dot as mmd
import tensorflow as tf
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Product, Euclidean
from pymanopt.solvers import ConjugateGradient

class CNTF:
    """
    Description:
        Coupled network tensor factorization of brain and individual microbiome networks. The first two modes of each tensor are nodes, the 3rd mode is subject (coupled) and last mode is time (uncoupled).

    Parameters
    ----------
    brain : numpy array
        Brain network data of shape (N,N,S,T) where N is number of nodes, S is number of subjects and T is the number of time points.
    taxa : numpy array
        Microbiome network data of shape (N,N,S,T) where N is number of nodes, S is number of subjects and T is the number of time points.'
    normalize_data : boolean
        Indicates whether to normalize data by scaling each network adjacency matrix to unit mass 1 and setting the diagonals to 0 (default=True).

    References
    ----------
    [1] Acar, E., Kolda, T. G., & Dunlavy, D. M. (2011). All-at-once optimization for coupled matrix and tensor factorizations. arXiv preprint arXiv:1105.3422.
    [2] Acar, E., Gürdeniz, G., Rasmussen, M. A., Rago, D., Dragsted, L. O., & Bro, R. (2012, December). Coupled matrix factorization with sparse factors to identify potential biomarkers in metabolomics. In 2012 IEEE 12th International Conference on Data Mining Workshops (pp. 1-8). IEEE.
    [3] Allen, G. (2012, March). Sparse higher-order principal components analysis. In Artificial Intelligence and Statistics (pp. 27-36). PMLR.

    Returns
    -------
    None
    """
    def __init__(self, brain, taxa, normalize_data=True):
        self.X = brain.copy()
        self.Y = taxa.copy()
        if self.normalize_data:
            self._norm_data()

    def _norm_data(self):
        """
        Description:
            Method for normalizing adjacency matrices to unit mass=1.

        Parameters
        ----------
        self : instance of self

        Returns
        -------
        None
        """
        n, m = self.X.shape[0], self.Y.shape[0]
        for i in range(self.X.shape[2]):
            for j in range(self.X.shape[3]):
                # zero diagonal
                self.X[:,:,i,j] *= (np.ones((n,n)) - np.eye(n))
                self.Y[:,:,i,j] *= (np.ones((m,m)) - np.eye(m))
                # scale each adjacency matrix to unit norm
                self.X[:,:,i,j] /= np.linalg.norm(self.X[:,:,i,j])
                self.Y[:,:,i,j] /= np.linalg.norm(self.Y[:,:,i,j])
        return

    def fit(self, rank=3, maxiter=100, alpha=1, manifolds=['E','E','E','E'], verbosity=2):
        """
        Description:
            Method for calling coupled tensor-tensor factorization via pymanopt adapted from [1,2].

        Parameters
        ----------
        rank : int
            Rank of decomposition (default=3).
        maxiter : int
            Maximum number of iterations for pymanopt optimization.
        alpha : float
            Weight for imposing unit-length to factor matrix columns (default=1).
        manifolds : list of length 4
            Each element is in {'E','S'} where each element is the manifold type for each factor matrix corresponding to tensor modes. 'E':Euclidean, 'S':Stiefel. (default=['E','E','E','E']: euclidean for all modes).
        verbosity : int
            Verbosity parameter passed to pymanopt (default=2).

        Returns
        -------
        None
        """
        self.rank, self.maxiter, self.alpha, self.manifolds, self.verbosity = rank, maxiter, alpha, manifolds, verbosity
        self._cntf_opt()
        return

    def _init_factors(self):
        """
        Description:
            Method for initializing TensorFlow Variables for pymanopt.
        """
        n,_,s,t = self.X.shape; m = self.Y.shape[0]; r = self.rank
        A = tf.Variable(tf.zeros((n,r), dtype=tf.float32), name="A")
        C = tf.Variable(tf.zeros((s,r), dtype=tf.float32), name="C")
        V = tf.Variable(tf.zeros((m,r), dtype=tf.float32), name="V")
        Tx = tf.Variable(tf.zeros((t,r), dtype=tf.float32), name="Tx")
        Ty = tf.Variable(tf.zeros((t,r), dtype=tf.float32), name="Ty")
        return A, C, V, Tx, Ty

    def _cntf_opt(self):
        """
        Description:
            Method for coupled tensor-tensor factorization via pymanopt adapted from [1,2].
        """
        manifold = self._get_manifolds()
        cost, egrad = self._get_solver_funs(manifold)
        solver = ConjugateGradient(**{'maxiter':self.maxiter})
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=self.verbosity)
        self.opt = solver.solve(problem)
        self._reconstruct()
        return

    def _diag_mask(self):
        """
        Description:
            Method for creating 'mask' arrays to mask diagonal values of adjacency matrices in network tensors for use with optimization.
        """
        Wb = tf.constant(np.ones(self.X.shape), dtype=tf.double)
        Wt = tf.constant(np.ones(self.Y.shape), dtype=tf.double)
        return Wb, Wt

    def _get_solver_funs(self, manifold):
        """
        Description:
            Method for locally defining cost functions for pymanopt.
        TODO:
            fix weird version issues with pymanopt to define cost as a function of manifold
        """
        egrad = None
        X, Y, a = self._get_data()
        X = tf.constant(X, dtype=tf.double)
        Y = tf.constant(Y, dtype=tf.double)
        A, C, V, Tx, Ty = self._init_factors()
        Wb, Wt = self._diag_mask()
        @pymanopt.function.TensorFlow
        def cost(A, C, V, Tx, Ty):
            AACT = tf.einsum('im,jm,km,lm->ijkl', A, A, C, Tx)
            f = 0.5 * tf.reduce_sum(Wb * ((X - AACT)**2))
            VVCT = tf.einsum('im,jm,km,lm->ijkl', V, V, C, Ty)
            f += 0.5 * tf.reduce_sum(Wt * ((Y - VVCT)**2))
            # 'l2-norm' reg
            f += a * tf.reduce_sum((tf.norm(A, axis=0) - 1)**2)
            f += a * tf.reduce_sum((tf.norm(C, axis=0) - 1)**2)
            f += a * tf.reduce_sum((tf.norm(V, axis=0) - 1)**2)
            f += a * tf.reduce_sum((tf.norm(Tx, axis=0) - 1)**2)
            f += a * tf.reduce_sum((tf.norm(Ty, axis=0) - 1)**2)
            return f
        return cost, egrad

    def _get_manifolds(self):
        """
        Description:
            Method for getting product manifold for all factor matrices.
        TODO:
            rewrite for better dimension checking, etc...
        """
        n,_,s,t = self.X.shape
        m = self.Y.shape[0]
        r = self.rank
        mans = tuple()
        mans += self._get_man(self.manifolds[0], n,r)
        mans += self._get_man(self.manifolds[1], s,r)
        mans += self._get_man(self.manifolds[2], m,r)
        mans += self._get_man(self.manifolds[3], t,r) # brain time
        mans += self._get_man(self.manifolds[3], t,r) # taxa time
        return Product(mans)

    def _get_man(self, man, n, m):
        """
        Description:
            Method for instantiating a pymanopt manifold (Euclidean or Stiefel).
        """
        if man == 'E':
            return (Euclidean(n,m),)
        elif man == 'S':
            if m > n:
                print('rank ({m}) must be <= dimension of mode ({n}) for Stiefel manifold. Setting this mode to Euclidean'.format(m,n))
                return (Euclidean(n,m),)
            return (Stiefel(n,m),)

    def get_factors(self, deflate=True):
        """
        Description:
            Method for getting tensor factor matrices from optimization.

        Parameters
        ----------
        self : instance of self
        deflate : boolean
            Indicates whether to scale each column to unit length (default=True).

        Returns
        -------
        A, C, V, Tx, Ty, L, S: numpy arrays
            Factor matrices: first five correspond to brain node, subject (coupled), taxa node, brain time and taxa time modes, respectively. Last two arrays are vectors of length = rank that indicate the scale for unit length factors for each component of the decomposition. If deflate=False, then L and S are all ones.
        """
        F = [a.copy() for a in self.opt]
        if deflate:
            return self._deflate(*F)
        return F

    def _deflate(self, A, C, V, Tx, Ty):
        """
        Description:
            Method for deflating factor matrix columns to unit length.
        TODO:
            rewrite in more elegant way...

        Parameters:
        -----------
        A, C, V, Tx, Ty: numpy arrays
            Factor matrices: correspond to brain node, subject (coupled), taxa node, brain time and taxa time modes, respectively.

        Returns
        -------
        A, C, V, Tx, Ty, L, S: numpy arrays
            Factor matrices with unit length columns. Last two arrays are vectors of length = rank that indicate the scale for unit length factors for each component of the decomposition.
        """
        L, S = np.ones(self.rank), np.ones(self.rank)
        for r in range(self.rank):
            L[r] *= np.linalg.norm(A[:,r])
            A[:,r] /= np.linalg.norm(A[:,r])
            L[r] *= np.linalg.norm(Tx[:,r])
            Tx[:,r] /= np.linalg.norm(Tx[:,r])
            L[r] *= np.linalg.norm(C[:,r])
            S[r] *= np.linalg.norm(C[:,r])
            C[:,r] /= np.linalg.norm(C[:,r])
            S[r] *= np.linalg.norm(V[:,r])
            V[:,r] /= np.linalg.norm(V[:,r])
            S[r] *= np.linalg.norm(Ty[:,r])
            Ty[:,r] /= np.linalg.norm(Ty[:,r])
        return A, C, V, Tx, Ty, L, S

    def _get_data(self):
        """
        Description:
            Method for getting tensor data/parameters for optimization.
        """
        return self.X, self.Y, self.alpha

    def _reconstruct(self):
        """
        Description:
            Method/wrapper for reconstructing estimated tensors and error/cumulative proportion of variance as in [3].
        """
        A, C, V, Tx, Ty, L, S = self.get_factors()
        X, Y, a = self._get_data()
        Wb, Wt = self._diag_mask()
        f1, f2, f = self._costT(A, C, V, Tx, Ty, L, S, X, Y, a)
        self.cost, self.rel_cost, self.cpve = {}, {}, {}
        self.cost['X'] = f1
        self.cost['Y'] = f2
        self.cost['total'] = f
        self.rel_cost['X'] = 2 * f1 / _ssq(self.X)
        self.rel_cost['Y'] = 2 * f2 / _ssq(self.Y)
        self.cpve['X'] = _cpvar(X, [A,A,C,Tx])
        self.cpve['Y'] = _cpvar(Y, [V,V,C,Ty])
        return

    def _costT(self, A, C, V, Tx, Ty, L, S, X, Y, a):
        """
        Description:
            Method/wrapper for computing cost function value.
        """
        AL = A @ np.diag(L)
        lAACT = np.einsum('im,jm,km,lm->ijkl', AL, A, C, Tx)
        f1 = 0.5 * np.sum((X - lAACT)**2)
        VS = V @ np.diag(S)
        sVVCT = np.einsum('im,jm,km,lm->ijkl', VS, V, C, Ty)
        f2 = 0.5 * np.sum((Y - sVVCT)**2)
        f = f1 + f2
        f += a * np.sum((np.linalg.norm(A, axis=0) - 1)**2)
        f += a * np.sum((np.linalg.norm(C, axis=0) - 1)**2)
        f += a * np.sum((np.linalg.norm(V, axis=0) - 1)**2)
        f += a * np.sum((np.linalg.norm(Tx, axis=0) - 1)**2)
        return f1, f2, f

# cntf helper fns
def _ssq(X):
    """
    Description:
        Method for computing sum of squares
    """
    return np.sum(X**2)

def _cpvar(X, F):
    """
    Description:
        Method for computing cumulative proportion of variance accounted for by factors as in [3].
    """
    nF = [norm_cols(f) for f in F]
    P = [u @ np.linalg.pinv(u.T @ u) @ u.T for u in nF]
    return ssq(mmd(X, P, [0, 1, 2, 3])) / ssq(X)

# misc
def rclr(X):
    """
    Description:
        Robust centered log ratio transform as in [4].

    Parameters
    ----------
    X : numpy array
        Matrix of shape S x M where S is number of subjects and M is number of taxa.

    Returns
    -------
    numpy array : numpy array
        Matrix rclr transformed data

    References
    ----------
    [4] Martino, C., Morton, J. T., Marotz, C. A., Thompson, L. R., Tripathi, A., Knight, R., & Zengler, K. (2019). A novel sparse compositional technique reveals microbial perturbations. MSystems, 4(1), e00016-19.
    """
    # compositional closure
    Xl = X / X.sum(axis=1, keepdims=True)
    Xi = np.log(Xl, where=Xl>0).copy()
    gmeans = Xi.mean(axis=1, where=Xi!=0, keepdims=True)
    return np.subtract(Xi, gmeans, where=Xi!=0)

def z2r(networks):
    """
    Description:
        Method to transform z score correlation matrices back to pearson r statistics.

    Parameters
    ----------
    networks : numpy array
        Stack of adjacency (correlation) matrices of shape N,N,S where N is the number of network nodes.

    Returns
    -------
    networks : numpy array
        Matrix rclr transformed data
    """
    for i in range(networks.shape[2]):
        networks[:,:,i] *= (np.ones(networks.shape[0]) - np.eye(networks.shape[0]))
        networks[:,:,i] += np.eye(networks.shape[0])
    return networks

def lioness(X):
    """
    Description:
        Method compute single sample network inferences (LIONESS) as in [5].

    Parameters
    ----------
    X : numpy array
        Matrix of shape N,M where N is subjects and M is taxa (must be rclr or isolog transformed).

    Returns
    -------
    Y : numpy array
        Stack of estimated adjacency matrices of shape M,M,N.

    Refernces
    ---------
    [5] Kuijjer, M. L., Tung, M. G., Yuan, G., Quackenbush, J., & Glass, K. (2019). Estimating sample-specific regulatory networks. Iscience, 14, 226-240.
    """
    n,m = X.shape
    Y = np.zeros((m,m,n))
    Yall = np.corrcoef(X.T)
    for i in range(n):
        Yi = np.corrcoef(np.delete(X.copy(),i,0).T)
        Y[:,:,i] = n * (Yall - Yi) + Yi
    return Y

