#01/14/22: coupled network tensor factorization of brain networks and microbiome

import numpy as np
from scipy.stats import spearmanr
from scipy.linalg import eigh, svd
from scipy.optimize import minimize_scalar, minimize
from tensorly.tenalg import multi_mode_dot as mmd
import tensorflow as tf
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Product, Euclidean
from pymanopt.optimizers import ConjugateGradient

class TNF:
    def __init__(self, networks, copy_data=True):
        if copy_data:
            self.X = networks.copy()
        else:
            self.X = networks

    def fit(self, rank=3, maxiter=100, alpha=1, verbosity=2, normalize_data=True):
        self.rank, self.maxiter, self.alpha, self.verbosity, self.normalize_data = rank, maxiter, alpha, verbosity, normalize_data
        self.norm_data()
        self._opt()
        return

    def norm_data(self):
        n = self.X.shape[0]
        if self.normalize_data:
            for i in range(self.X.shape[2]):
                for j in range(self.X.shape[3]):
                    self.X[:,:,i,j] *= (np.ones((n,n)) - np.eye(n))
                    self.X[:,:,i,j] /= np.linalg.norm(self.X[:,:,i,j])
        return

    def init_factors(self):
        n,_,s,t = self.X.shape; r = self.rank
        A = tf.Variable(tf.zeros((n,r), dtype=tf.float32), name="A")
        C = tf.Variable(tf.zeros((s,r), dtype=tf.float32), name="C")
        T = tf.Variable(tf.zeros((t,r), dtype=tf.float32), name="T")
        return A, C, T

    def _opt(self):
        n,_,s,t = self.X.shape
        r = self.rank
        manifold = Product((Euclidean(n,r), Euclidean(s,r), Euclidean(t,r)))
        cost, egrad = self._get_solver_funs()
        solver = ConjugateGradient(**{'maxiter':self.maxiter})
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=self.verbosity)
        self.opt = solver.solve(problem)
        self._reconstruct()
        return

    def _diag_mask(self, tfw=True):
        n,_,s,t = self.X.shape
        W = np.ones(self.X.shape)
        for i in range(s):
            for j in range(t):
                W[:,:,i,j] -= np.eye(n)
        if tfw:
            W = tf.constant(W.copy(), dtype=tf.double)
        return W

    def _get_solver_funs(self):
        egrad = None
        X, a = self.X.copy(), self.alpha
        X = tf.constant(X, dtype=tf.double)
        A, C, T = self.init_factors()
        W = self._diag_mask()
        @pymanopt.function.TensorFlow
        def cost(A, C, T):
            AACT = tf.einsum('im,jm,km,lm->ijkl', A, A, C, T)
            f = 0.5 * tf.reduce_sum(W * ((X - AACT)**2))
            # 'l2-norm' reg
            f += a * tf.reduce_sum((tf.norm(A, axis=0) - 1)**2)
            f += a * tf.reduce_sum((tf.norm(C, axis=0) - 1)**2)
            f += a * tf.reduce_sum((tf.norm(T, axis=0) - 1)**2)
            return f
        return cost, egrad

    def get_factors(self):
        return self._deflate(*[a.copy() for a in self.opt])

    def _deflate(self, A, C, T):
        L = np.ones(self.rank)
        for r in range(self.rank):
            L[r] *= np.linalg.norm(A[:,r])
            A[:,r] /= np.linalg.norm(A[:,r])
            L[r] *= np.linalg.norm(C[:,r])
            C[:,r] /= np.linalg.norm(C[:,r])
            L[r] *= np.linalg.norm(T[:,r])
            T[:,r] /= np.linalg.norm(T[:,r])
        return A, C, T, L

    def _reconstruct(self):
        A, C, T, L = self.get_factors()
        X, a = self.X.copy(), self.alpha
        W = self._diag_mask()
        f, fnet = _costNet(X, A, C, T, L, a)
        self.cost = f
        self.rel_sse = 2 * fnet / ssq(self.X)
        self.cpve = cpvar(X, [A,A,C,T])
        return

def _costNet(X, A, C, T, L, a):
    AL = A @ np.diag(L)
    lAACT = np.einsum('im,jm,km,lm->ijkl', AL, A, C, T)
    fnet = 0.5 * np.sum((X - lAACT)**2)
    f = a * np.sum((np.linalg.norm(A, axis=0) - 1)**2)
    f += fnet
    f += a * np.sum((np.linalg.norm(C, axis=0) - 1)**2)
    f += a * np.sum((np.linalg.norm(T, axis=0) - 1)**2)
    return f, fnet

class CTNTF:
    """
    Description:
        Coupled network tensor factorization of brain networks and (CSS normalized) microbiome taxa counts. The last network mode and first taxa mode (subject) are coupled

    Parameters
    ----------
    networks : numpy array
        brain network data of shape (N,N,S) where N is number of nodes and S is number of subjects
    taxa : numpy array
        normalized taxa data of shape (S,M) where S is number of subjects and M is taxa.

    References
    ----------
    [1] Acar, E., Kolda, T. G., & Dunlavy, D. M. (2011). All-at-once optimization for coupled matrix and tensor factorizations. arXiv preprint arXiv:1105.3422.
    [2] Acar, E., Gürdeniz, G., Rasmussen, M. A., Rago, D., Dragsted, L. O., & Bro, R. (2012, December). Coupled matrix factorization with sparse factors to identify potential biomarkers in metabolomics. In 2012 IEEE 12th International Conference on Data Mining Workshops (pp. 1-8). IEEE.

    Returns
    -------
    self: an instance of self
    """
    def __init__(self, brain, taxa, copy_data=True):
        if copy_data:
            self.X = brain.copy()
            self.Y = taxa.copy()
        else:
            self.X = brain
            self.Y = taxa

    def fit(self, rank=3, maxiter=100, alpha=1, beta=1e-3, gamma=0, zeta=0, epsl1=1e-8, manifolds=['E','E','E'], cost_fun='ACNTF', couple_time=False, verbosity=2, normalize_data=True, initial_factors=None):
        self.rank, self.maxiter, self.alpha, self.beta, self.gamma, self.zeta, self.epsl1, self.manifolds, self.cost_fun, self.couple_time, self.verbosity, self.normalize_data, self.initial_factors = rank, maxiter, alpha, beta, gamma, zeta, epsl1, manifolds, cost_fun, couple_time, verbosity, normalize_data, initial_factors
        self.norm_data()
        if self.cost_fun=='CNTF':
            self.beta,self.zeta = 0,0
        self._cntf_opt()
        return

    def norm_data(self):
        n, m = self.X.shape[0], self.Y.shape[0]
        if self.normalize_data:
            for i in range(self.X.shape[2]):
                for j in range(self.X.shape[3]):
                    self.X[:,:,i,j] *= (np.ones((n,n)) - np.eye(n))
                    self.Y[:,:,i,j] *= (np.ones((m,m)) - np.eye(m))
                    self.X[:,:,i,j] /= np.linalg.norm(self.X[:,:,i,j])
                    self.Y[:,:,i,j] /= np.linalg.norm(self.Y[:,:,i,j])
        return

    def init_factors(self):
        n,_,s,t = self.X.shape; m = self.Y.shape[0]; r = self.rank
        A = tf.Variable(tf.zeros((n,r), dtype=tf.float32), name="A")
        C = tf.Variable(tf.zeros((s,r), dtype=tf.float32), name="C")
        V = tf.Variable(tf.zeros((m,r), dtype=tf.float32), name="V")
        T = tf.Variable(tf.zeros((t,r), dtype=tf.float32), name="T")
        Tx = tf.Variable(tf.zeros((t,r), dtype=tf.float32), name="Tx")
        Ty = tf.Variable(tf.zeros((t,r), dtype=tf.float32), name="Ty")
        L = tf.Variable(tf.zeros(r, dtype=tf.float32), name="L")
        S = tf.Variable(tf.zeros(r, dtype=tf.float32), name="S")
        return A, C, V, T, Tx, Ty, L, S

    def _cntf_opt(self):
        manifold = self._get_manifolds()
        cost, egrad = self._get_solver_funs(manifold)
        solver = ConjugateGradient(**{'max_iterations':self.maxiter})
        problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=egrad)#, verbosity=self.verbosity)
        
        #problem = Problem(manifold=manifold, cost=cost)# egrad=egrad, verbosity=self.verbosity)
        #self.opt = solver.run(problem)
        #self._reconstruct()
        
        
        if self.initial_factors is not None:
            self.opt = solver.run(problem, x=self.initial_factors)
        else:
            self.opt = solver.run(problem)
        self._reconstruct()
        return

    def _diag_mask(self, tfw=True):
        n,_,s,t = self.X.shape
        m = self.Y.shape[0]
        Wb, Wt = np.ones(self.X.shape), np.ones(self.Y.shape)
        for i in range(s):
            for j in range(t):
                Wb[:,:,i,j] -= np.eye(n)
                Wt[:,:,i,j] -= np.eye(m)
        if tfw:
            Wb = tf.constant(Wb.copy(), dtype=tf.double)
            Wt = tf.constant(Wt.copy(), dtype=tf.double)
        return Wb, Wt

    def _get_solver_funs(self, manifold):
        egrad = None
        X, Y, a, b, g, z, el1 = self._get_data()
        X = tf.constant(X, dtype=tf.double)
        Y = tf.constant(Y, dtype=tf.double)
        A, C, V, T, Tx, Ty, L, S = self.init_factors()
        Wb, Wt = self._diag_mask(tfw=True)
        if not self.couple_time:
            if self.cost_fun=='ACNTF':
                @pymanopt.function.tensorflow(manifold)
                def cost(A, C, V, Tx, Ty, L, S):
                    AL = tf.matmul(A, tf.linalg.diag(L))
                    lAACT = tf.einsum('im,jm,km,lm->ijkl', AL, A, C, Tx)
                    f = 0.5 * tf.reduce_sum(Wb * ((X - lAACT)**2))
                    VS = tf.matmul(V, tf.linalg.diag(S))
                    sVVCT = tf.einsum('im,jm,km,lm->ijkl', VS, V, C, Ty)
                    f += 0.5 * tf.reduce_sum(Wt * ((Y - sVVCT)**2))
                    # 'l2-norm' reg
                    f += a * tf.reduce_sum((tf.norm(A, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(C, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(V, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(Tx, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(Ty, axis=0) - 1)**2)
                    # l1-norm reg
                    f += b * tf.reduce_sum(tf.sqrt(L**2 + el1))
                    f += b * tf.reduce_sum(tf.sqrt(S**2 + el1))
                    f += g * tf.reduce_sum(tf.sqrt(A**2 + el1))
                    f += g * tf.reduce_sum(tf.sqrt(V**2 + el1))
                    f += z * tf.reduce_sum((L-S)**2)
                    return f
            elif self.cost_fun=='CNTF':
                @pymanopt.function.tensorflow(manifold)
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
                    # l1-norm reg
                    f += g * tf.reduce_sum(tf.reduce_sum(tf.sqrt(A**2 + el1), axis=0))
                    f += g * tf.reduce_sum(tf.reduce_sum(tf.sqrt(V**2 + el1), axis=0))
                    return f
        else:
            if self.cost_fun=='ACNTF':
                @pymanopt.function.tensorflow(manifold)
                def cost(A, C, V, T, L, S):
                    AL = tf.matmul(A, tf.linalg.diag(L))
                    lAACT = tf.einsum('im,jm,km,lm->ijkl', AL, A, C, T)
                    f = 0.5 * tf.reduce_sum((X - lAACT)**2)
                    VS = tf.matmul(V, tf.linalg.diag(S))
                    sVVCT  = tf.einsum('im,jm,km,lm->ijkl', VS, V, C, T)
                    f += 0.5 * tf.reduce_sum((Y - sVVCT)**2)
                    # 'l2-norm' reg
                    f += a * tf.reduce_sum((tf.norm(A, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(C, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(V, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(T, axis=0) - 1)**2)
                    # l1-norm reg
                    f += b * tf.reduce_sum(tf.sqrt(L**2 + el1))
                    f += b * tf.reduce_sum(tf.sqrt(S**2 + el1))
                    f += g * tf.reduce_sum(tf.sqrt(A**2 + el1))
                    f += g * tf.reduce_sum(tf.sqrt(V**2 + el1))
                    f += z * tf.reduce_sum((L-S)**2)
                    return f
            elif self.cost_fun=='CNTF':
                @pymanopt.function.tensorflow(manifold)
                def cost(A, C, V, T):
                    AACT = tf.einsum('im,jm,km,lm->ijkl', A, A, C, T)
                    f = 0.5 * tf.reduce_sum((X - AACT)**2)
                    VVCT  = tf.einsum('im,jm,km,lm->ijkl', V, V, C, T)
                    f += 0.5 * tf.reduce_sum((Y - VVCT)**2)
                    # 'l2-norm' reg
                    f += a * tf.reduce_sum((tf.norm(A, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(C, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(V, axis=0) - 1)**2)
                    f += a * tf.reduce_sum((tf.norm(T, axis=0) - 1)**2)
                    # l1-norm reg
                    f += g * tf.reduce_sum(tf.sqrt(A**2 + el1))
                    f += g * tf.reduce_sum(tf.sqrt(V**2 + el1))
                    return f
        return cost, egrad

    def _get_manifolds(self):
        n,_,s,t = self.X.shape
        m = self.Y.shape[0]
        r = self.rank
        mans = tuple()
        mans += self._get_man(self.manifolds[0], n,r)
        mans += self._get_man(self.manifolds[1], s,r)
        mans += self._get_man(self.manifolds[2], m,r)
        mans += self._get_man('E',t,r)
        if not self.couple_time:
            mans += self._get_man('E',t,r)
        if self.cost_fun=='ACNTF':
            mans += (Euclidean(r),)
            mans += (Euclidean(r),)
        return Product(mans)

    def _get_man(self, man, n, m):
        if man == 'E':
            return (Euclidean(n,m),)
        elif man == 'S':
            return (Stiefel(n,m),)

    def get_factors(self, copy_data=True):
        # replaced self.opt with [a.copy() for a in self.opt.point]
        n = self.rank
        if self.cost_fun=='ACNTF':
            if not self.couple_time:
                return [a.copy() for a in self.opt.point]
            A, C, V, T, L, S = [a.copy() for a in self.opt.point]
            return A, C, V, T, T.copy(), L, S
        elif self.cost_fun=='CNTF':
            if not self.couple_time:
                A, C, V, Tx, Ty = [a.copy() for a in self.opt.point]
                if copy_data:
                    return self._deflate(*[a.copy() for a in [A, C, V, Tx, Ty]])
                else:
                    return self._deflate(A, C, V, Tx, Ty)
            else:
                A, C, V, T = [a.copy() for a in self.opt.point]
                return self._deflate(A, C, V, T, T.copy())

    def _deflate(self, A, C, V, Tx, Ty):
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

    def _get_data(self, copy_data=True):
        if copy_data:
            return [i.copy() if isinstance(i, np.ndarray) else i for i in [self.X, self.Y, self.alpha, self.beta, self.gamma, self.zeta, self.epsl1]]
        return [self.X, self.Y, self.alpha, self.beta, self.gamma, self.zeta, self.epsl1]

    def _reconstruct(self):
        A, C, V, Tx, Ty, L, S = self.get_factors()
        X, Y, a, b, g, z, el1 = self._get_data()
        Wb, Wt = self._diag_mask(tfw=False)
        #AL = A @ np.diag(L)
        #self.X_est = Wb * np.einsum('im,jm,km,lm->ijkl', AL, A, C, Tx)
        #VS = V @ np.diag(S)
        #self.Y_est = Wt * np.einsum('im,jm,km,lm->ijkl', VS, V, C, Ty)
        f1, f2, f = _costT(A, C, V, Tx, Ty, L, S, X, Y, a, b, g, z, el1, self.couple_time)
        self.cost, self.rel_cost, self.cpve = {}, {}, {}
        self.cost['X'] = f1
        self.cost['Y'] = f2
        self.cost['total'] = f
        self.rel_cost['X'] = 100*(1-(2 * f1 / ssq(self.X)))
        self.rel_cost['Y'] = 100*(1-(2 * f2 / ssq(self.Y)))#2 * f2 / ssq(self.Y)
        self.cpve['X'] = cpvar(X, [A,A,C,Tx])
        self.cpve['Y'] = cpvar(Y, [V,V,C,Ty])
        return

def _costT(A, C, V, Tx, Ty, L, S, X, Y, a, b, g, z, el1, ct=False):
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
    if ct:
        f += a * np.sum((np.linalg.norm(Ty, axis=0) - 1)**2)
    f += b * np.sum(np.sqrt(L**2 + el1))
    f += b * np.sum(np.sqrt(S**2 + el1))
    f += g * np.sum(np.sqrt(A**2 + el1))
    f += g * np.sum(np.sqrt(V**2 + el1))
    f += z * np.sum((L-S)**2)
    return f1, f2, f

class CNTF:
    """
    Description:
        Coupled network tensor factorization of brain networks and (CSS normalized) microbiome taxa counts. The last network mode and first taxa mode (subject) are coupled

    Parameters
    ----------
    networks : numpy array
        brain network data of shape (N,N,S) where N is number of nodes and S is number of subjects
    taxa : numpy array
        normalized taxa data of shape (S,M) where S is number of subjects and M is taxa.

    References
    ----------
    [1] Acar, E., Kolda, T. G., & Dunlavy, D. M. (2011). All-at-once optimization for coupled matrix and tensor factorizations. arXiv preprint arXiv:1105.3422.
    [2] Acar, E., Gürdeniz, G., Rasmussen, M. A., Rago, D., Dragsted, L. O., & Bro, R. (2012, December). Coupled matrix factorization with sparse factors to identify potential biomarkers in metabolomics. In 2012 IEEE 12th International Conference on Data Mining Workshops (pp. 1-8). IEEE.

    Returns
    -------
    self: an instance of self
    """
    def __init__(self, brain, taxa):
        self.X = brain.copy()
        self.Y = taxa.copy()

    def fit(self, rank=3, maxiter=100, alpha=1, beta=1e-3, gamma=0, zeta=0, epsl1=1e-8, manifolds=['E','E','E'], cost_fun='ACNTF', verbosity=2, normalize_data=True):
        self.rank, self.maxiter, self.alpha, self.beta, self.gamma, self.zeta, self.epsl1, self.manifolds, self.cost_fun, self.verbosity, self.normalize_data = rank, maxiter, init_base, alpha, beta, gamma, zeta, epsl1, manifolds, cost_fun, verbosity, normalize_data
        self.norm_data()
        if self.cost_fun=='CNTF':
            self.beta,self.zeta = 0,0
        self._cntf_opt()
        return

    def norm_data(self):
        if self.normalize_data:
            for i in range(self.X.shape[2]):
                self.X[:,:,i] /= np.linalg.norm(self.X[:,:,i])
                self.Y[:,:,i] /= np.linalg.norm(self.Y[:,:,i])
            #self.T /= np.linalg.norm(self.T)
            #self.M /= np.linalg.norm(self.M)
        return

    def init_factors(self):
        n,_,s = self.X.shape; m = self.Y.shape[0]; r = self.rank
        A = tf.Variable(tf.zeros((n,r), dtype=tf.float32), name="A")
        C = tf.Variable(tf.zeros((s,r), dtype=tf.float32), name="C")
        V = tf.Variable(tf.zeros((m,r), dtype=tf.float32), name="V")
        if self.cost_fun=='ACNTF':
            L = tf.Variable(tf.zeros(r, dtype=tf.float32), name="L")
            S = tf.Variable(tf.zeros(r, dtype=tf.float32), name="S")
        return A, C, V, L, S

    def _cmtf_opt(self):
        cost, egrad = self._get_solver_funs()
        manifold = self._get_manifolds()
        solver = ConjugateGradient(**{'maxiter':self.maxiter})
        problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=egrad)
        self.opt = solver.solve(problem)
        self._reconstruct()
        return

    def _get_solver_funs(self):
        egrad = None
        X, Y, a, b, g, z = self._get_data()
        A, C, V, L, S = self.init_factors()
        if self.cost_fun=='ACNTF':
            @pymanopt.function.tensorflow(manifold)
            def cost(A, C, V, L, S):
                AL = tf.matmul(A, tf.linalg.diag(L))
                lAAC = tf.einsum('il,jl,kl->ijk', AL, A, C)
                f = 0.5 * tf.reduce_sum((X - lAAC)**2)
                VS = tf.matmul(V, tf.linalg.diag(S))
                sVVC = tf.einsum('il,jl,kl->ijk', VS, V, C)
                f += 0.5 * tf.reduce_sum((Y - sVVC)**2)
                # 'l2-norm' reg
                f += a * tf.reduce_sum((tf.norm(A, axis=0) - 1)**2)
                f += a * tf.reduce_sum((tf.norm(C, axis=0) - 1)**2)
                f += a * tf.reduce_sum((tf.norm(V, axis=0) - 1)**2)
                # l1-norm reg
                f += b * tf.reduce_sum(tf.sqrt(L**2 + el1))
                f += b * tf.reduce_sum(tf.sqrt(S**2 + el1))
                f += g * tf.reduce_sum(tf.sqrt(A**2 + el1))
                f += g * tf.reduce_sum(tf.sqrt(V**2 + el1))
                f += z * tf.reduce_sum((L-S)**2)
                return f
        elif self.cost_fun=='CNTF':
            @pymanopt.function.tensorflow(manifold)
            def cost(A, C, V):
                lAAC = tf.einsum('il,jl,kl->ijk', A, A, C)
                f = 0.5 * tf.reduce_sum((X - lAAC)**2)
                sVVC = tf.einsum('il,jl,kl->ijk', V, V, C)
                f += 0.5 * tf.reduce_sum((Y - sVVC)**2)
                # 'l2-norm' reg
                f += a * tf.reduce_sum((tf.norm(A, axis=0) - 1)**2)
                f += a * tf.reduce_sum((tf.norm(C, axis=0) - 1)**2)
                f += a * tf.reduce_sum((tf.norm(V, axis=0) - 1)**2)
                # l1-norm reg
                f += g * tf.reduce_sum(tf.sqrt(A**2 + el1))
                f += g * tf.reduce_sum(tf.sqrt(V**2 + el1))
                return f
        return cost, egrad

    def _get_manifolds(self):
        n,_,s = self.X.shape
        m = self.Y.shape[0]
        r = self.rank
        mans = tuple()
        mans += self._get_man(self.manifolds[0], n,r)
        mans += self._get_man(self.manifolds[1], s,r)
        mans += self._get_man(self.manifolds[1], m,r)
        if self.cost_fun=='ACNTF':
            mans += (Euclidean(r),)
            mans += (Euclidean(r),)
        return Product(mans)

    def _get_man(self, man, n, m):
        if man == 'E':
            return (Euclidean(n,m),)
        elif man == 'S':
            return (Stiefel(n,m),)

    def get_factors(self):
        if self.cost_fun=='ACNTF':
            return self.opt
        elif self.cost_fun=='CNTF':
            A, C, V = self.opt
            n = self.rank
            return A, C, V, np.ones(n), np.ones(n)

    def _get_data(self):
        return [i.copy() if isinstance(i, np.ndarray) else i for i in [self.X, self.Y, self.alpha, self.beta, self.gamma, self.zeta, self.epsl1]]

    def _reconstruct(self):
        A, C, V, L, S = self.get_factors()
        X, Y, a, b, g, z, el1 = self._get_data()
        self.X_est = np.einsum('il,jl,kl->ijk',(A@np.diag(L)),A,C)
        self.Y_est = np.einsum('il,jl,kl->ijk',(V@np.diag(S)),V,C)
        f1, f2, f = _costA(A, C, V, L, S, X, Y, a, b, el1)
        self.cost, self.rel_cost, self.cpve = {}, {}, {}
        self.cost['X'] = f1
        self.cost['Y'] = f2
        self.cost['total'] = f
        self.rel_cost['X'] = 2 * f1 / ssq(self.T)
        self.rel_cost['Y'] = 2 * f2 / ssq(self.M)
        self.cpve['X'] = cpvar(X, [A,A,C])
        self.cpve['Y'] = cpvar(Y, [V,V,C])
        return

def _costA(A, C, V, L, S, X, Y, a, b, g, z, el1):
    lAAC = np.einsum('il,jl,kl->ijk',(A @ np.diag(L)), A, C)
    f1 = 0.5 * np.sum((X - lAAC)**2)
    sVVC = np.einsum('il,jl,kl->ijk',(V @ np.diag(S)), V, C)
    f1 = 0.5 * np.sum((Y - sVVC)**2)
    f = f1 + f2
    f += a * np.sum((np.linalg.norm(A, axis=0) - 1)**2)
    f += a * np.sum((np.linalg.norm(C, axis=0) - 1)**2)
    f += a * np.sum((np.linalg.norm(V, axis=0) - 1)**2)
    f += b * np.sum(np.sqrt(L**2 + el1))
    f += b * np.sum(np.sqrt(S**2 + el1))
    f += g * np.sum(np.sqrt(A**2 + el1))
    f += g * np.sum(np.sqrt(V**2 + el1))
    f += z * np.sum((L-S)**2)
    return f1, f2, f

class CMTF:
    """
    Description:
        Coupled network tensor factorization of brain networks and (CSS normalized) microbiome taxa counts. The last network mode and first taxa mode (subject) are coupled

    Parameters
    ----------
    networks : numpy array
        brain network data of shape (N,N,S) where N is number of nodes and S is number of subjects
    taxa : numpy array
        normalized taxa data of shape (S,M) where S is number of subjects and M is taxa.

    References
    ----------
    [1] Acar, E., Kolda, T. G., & Dunlavy, D. M. (2011). All-at-once optimization for coupled matrix and tensor factorizations. arXiv preprint arXiv:1105.3422.
    [2] Acar, E., Gürdeniz, G., Rasmussen, M. A., Rago, D., Dragsted, L. O., & Bro, R. (2012, December). Coupled matrix factorization with sparse factors to identify potential biomarkers in metabolomics. In 2012 IEEE 12th International Conference on Data Mining Workshops (pp. 1-8). IEEE.

    Returns
    -------
    self: an instance of self
    """
    def __init__(self, brain, taxa):
        self.T = brain.copy()
        self.M = taxa.copy()

    def fit(self, rank=3, maxiter=100, alpha=1, beta=1e-3, gamma=0, zeta=0, epsl1=1e-8, manifolds=['E','E','E'], verbosity=2, normalize_data=True):
        self.rank, self.maxiter, self.alpha, self.beta, self.gamma, self.zeta, self.epsl1, self.manifolds, self.verbosity, self.normalize_data = rank, maxiter, init_base, alpha, beta, gamma, zeta, epsl1, manifolds, verbosity, normalize_data
        self.norm_data()
        self._cntf_opt()
        return

    def norm_data(self):
        if self.normalize_data:
            for i in range(self.T.shape[2]):
                self.T[:,:,i] /= np.linalg.norm(self.T[:,:,i])
            #self.T /= np.linalg.norm(self.T)
            self.M /= np.linalg.norm(self.M)
        return

    def init_factors(self):
        n,_,s = self.T.shape; m = self.M.shape[1]; r = self.rank
        A = tf.Variable(tf.zeros((n,r), dtype=tf.float32), name="A")
        C = tf.Variable(tf.zeros((s,r), dtype=tf.float32), name="C")
        V = tf.Variable(tf.zeros((m,r), dtype=tf.float32), name="V")
        L = tf.Variable(tf.zeros(r, dtype=tf.float32), name="L")
        S = tf.Variable(tf.zeros(r, dtype=tf.float32), name="S")
        return A, C, V, L, S

    def _cmtf_opt(self):
        cost, egrad = self._get_solver_funs()
        manifold = self._get_manifolds()
        solver = ConjugateGradient(**{'maxiter':self.maxiter})
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=self.verbosity)
        self.opt = solver.solve(problem)
        self._reconstruct()
        return

    def _get_solver_funs(self):
        egrad = None
        T, M, a, b, g, z = self._get_data()
        A, C, V, L, S = self.init_factors()
        @pymanopt.function.TensorFlow
        def cost(A, C, V, L, S):
            #lAAC = np.einsum('l,il,jl,kl->ijk',L,A,A,C)
            AL = tf.matmul(A, tf.linalg.diag(L))
            lAAC = tf.einsum('il,jl,kl->ijk', AL, A, C)
            f = 0.5 * tf.reduce_sum((T - lAAC)**2)
            f += 0.5 * tf.reduce_sum((M - tf.matmul(tf.matmul(C,tf.linalg.diag(S)), tf.transpose(V)))**2)
            f += a * tf.reduce_sum((tf.norm(A, axis=0) - 1)**2)
            f += a * tf.reduce_sum((tf.norm(C, axis=0) - 1)**2)
            f += a * tf.reduce_sum((tf.norm(V, axis=0) - 1)**2)
            f += b * tf.reduce_sum(tf.sqrt(L**2 + el1))
            f += b * tf.reduce_sum(tf.sqrt(S**2 + el1))
            f += g * tf.reduce_sum(tf.sqrt(A**2 + el1))
            f += g * tf.reduce_sum(tf.sqrt(V**2 + el1))
            f += z * tf.reduce_sum((L-S)**2)
            return f
        return cost, egrad

    def _get_manifolds(self):
        n,_,s = self.T.shape
        m = self.M.shape[1]
        r = self.rank
        mans = tuple()
        mans += self._get_man(self.manifolds[0], n,r)
        mans += self._get_man(self.manifolds[1], s,r)
        mans += self._get_man(self.manifolds[1], m,r)
        mans += (Euclidean(r),)
        mans += (Euclidean(r),)
        return Product(mans)

    def _get_man(self, man, n, m):
        if man == 'E':
            return (Euclidean(n,m),)
        elif man == 'S':
            return (Stiefel(n,m),)

    def get_factors(self):
        return self.opt

    def _get_data(self):
        return [i.copy() if isinstance(i, np.ndarray) else i for i in [self.T, self.M, self.alpha, self.beta, self.gamma, self.zeta, self.epsl1]]

    def _reconstruct(self):
        A, C, V, L, S = self.get_factors()
        T, M, a, b, g, z, el1 = self._get_data()
        self.T_est = np.einsum('il,jl,kl->ijk',(A@np.diag(L)),A,C)
        self.M_est = C @ (V@np.diag(S)).T
        f1, f2, f = _cost(A, C, V, L, S, T, M, a, b, el1)
        self.cost, self.rel_cost, self.cpve = {}, {}, {}
        self.cost['T'] = f1
        self.cost['M'] = f2
        self.cost['total'] = f
        self.rel_cost['T'] = 2 * f1 / ssq(self.T)
        self.rel_cost['M'] = 2 * f2 / ssq(self.M)
        self.cpve['T'] = cpvar(T, [A,A,C])
        self.cpve['M'] = cpvar(M, [C,V])
        return

def _cost(A, C, V, L, S, T, M, a, b, g, z, el1):
    lAAC = np.einsum('il,jl,kl->ijk',(A @ np.diag(L)), A, C)
    f1 = 0.5 * np.sum((T - lAAC)**2)
    f2 = 0.5 * np.sum((M - (C @ np.diag(S) @ V.T))**2)
    f = f1 + f2
    f += a * np.sum((np.linalg.norm(A, axis=0) - 1)**2)
    f += a * np.sum((np.linalg.norm(C, axis=0) - 1)**2)
    f += a * np.sum((np.linalg.norm(V, axis=0) - 1)**2)
    f += b * np.sum(np.sqrt(L**2 + el1))
    f += b * np.sum(np.sqrt(S**2 + el1))
    f += g * np.sum(np.sqrt(A**2 + el1))
    f += g * np.sum(np.sqrt(V**2 + el1))
    f += z * np.sum((L-S)**2)
    return f1, f2, f

# cntf helper fns
def reconstruct(A,C,T,L=None):
    n,r = A.shape
    s,t = C.shape[0], T.shape[0]
    X = np.zeros((n,n,s,t))
    L = L if L is not None else np.ones(r)
    return np.einsum('m,im,jm,km,lm->ijkl', L,A,A,C,T)

def best_scalar(X,Y):
    return (X.reshape(-1).T @ Y.reshape(-1)) / (X.reshape(-1).T @ X.reshape(-1))

def sdiag(X):
    return np.array([X[tuple([i]*X.ndim)] for i in range(X.shape[0])])

def cpvar(X, F):
    nF = [norm_cols(f) for f in F]
    P = [u @ np.linalg.pinv(u.T @ u) @ u.T for u in nF]
    return ssq(mmd(X, P, [0, 1, 2, 3])) / ssq(X)

def norm_cols(X):
    return X / np.linalg.norm(X, axis=0)[None,:]

def diag3D(d):
    D = np.zeros(tuple([len(d)] * 3))
    np.einsum('iii->i', D)[...] = d.copy()
    return D

def unvec(v):
    i = int(np.sqrt(len(v)))
    return v.reshape(i,i).T

def ssq(X):
    return np.sum(X**2)

def _unvec(v):
    I = int(np.sqrt(len(v)))
    m = np.zeros((I,I))
    for j in range(I):
        for i in range(I):
            m[i,j] = v[j*I+i]
    return m

def _get_mask(self, X):
    mask = np.where(X > 0)
    W = np.zeros(X.shape)
    W[mask] = 1
    return W

# misc
def rclr(X):
    """
    Description
    -----------
        Robust centered log ratio transform as in [3].

    Parameters
    ----------
    X : numpy array
        Matrix of shape S x M where S is number of subjects and M is number of taxa.

    References
    ----------
    [3] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6372836/
    """
    # compositional closure
    Xl = X / X.sum(axis=1, keepdims=True)
    Xi = np.log(Xl, where=Xl>0).copy()
    gmeans = Xi.mean(axis=1, where=Xi!=0, keepdims=True)
    return np.subtract(Xi, gmeans, where=Xi!=0)

def z2r(networks):
    for i in range(networks.shape[2]):
        networks[:,:,i] *= (np.ones(networks.shape[0]) - np.eye(networks.shape[0]))
        networks[:,:,i] += np.eye(networks.shape[0])
    return networks

def _norm_data(Y):
    X = Y.copy()
    n = X.shape[0]
    for i in range(X.shape[2]):
        for j in range(X.shape[3]):
            X[:,:,i,j] *= (np.ones((n,n)) - np.eye(n))
            X[:,:,i,j] /= np.linalg.norm(X[:,:,i,j])
    return X

# lioness:
def lioness(X, corr_type='pearson'):
    # X: numpy array shape N x M, N is subjects, M is taxa
    n,m = X.shape
    Y = np.zeros((m,m,n))
    if corr_type == 'pearson':
        Yall = np.corrcoef(X.T)
    elif corr_type == 'spearman':
        Yall = spearmanr(X)[0]
    for i in range(n):
        if corr_type == 'pearson':
            Yi = np.corrcoef(np.delete(X.copy(),i,0).T)
        elif corr_type == 'spearman':
            Yi = spearmanr(np.delete(X.copy(),i,0))[0]
        Y[:,:,i] = n * (Yall - Yi) + Yi
    return Y


class ScaleFactors:
    def __init__(self, A, C, T, Y):
        self.A = A
        self.C = C
        self.T = T
        self.Y = Y
        self.rank = A.shape[1]

    def fit(self, maxiter=1000, verbosity=2):
        self.maxiter, self.verbosity = maxiter, verbosity
        self._opt()
        return

    def init_factors(self):
        L = tf.Variable(tf.zeros((self.rank), dtype=tf.float32), name="L")
        return L

    def _opt(self):
        manifold = Euclidean(self.rank)
        cost, egrad = self._get_solver_funs()
        solver = ConjugateGradient(**{'maxiter':self.maxiter})
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=self.verbosity)
        self.opt = solver.solve(problem)
        return

    def _diag_mask(self, tfw=True):
        n,_,s,t = self.Y.shape
        W = np.ones((n,n,s,t))
        for i in range(s):
            for j in range(t):
                W[:,:,i,j] -= np.eye(n)
        if tfw:
            W = tf.constant(W.copy(), dtype=tf.double)
        return W

    def _get_solver_funs(self):
        egrad = None
        Y = self.Y.copy()
        Y = tf.constant(Y, dtype=tf.double)
        L = self.init_factors()
        W = self._diag_mask()
        A, C, T = self.A.copy(), self.C.copy(), self.T.copy()
        @pymanopt.function.TensorFlow
        def cost(L):
            LAACT = tf.einsum('m,im,jm,km,lm->ijkl', L, A, A, C, T)
            f = 0.5 * tf.reduce_sum(W * ((Y - LAACT)**2))
            return f
        return cost, egrad

#
