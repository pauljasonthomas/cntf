import numpy as np
import pandas as pd
import pingouin as pg
import myutils as mu

def tnf_iter(iter, cn, fp, df, vars, cvars, maxiter=1000, alpha=1):
    r,m,g = iter
    RDF = []
    cn.fit(rank=r, maxiter=maxiter, alpha=1, verbosity=1, normalize_data=True)
    cost, rel_sse, cpve = cn.cost, cn.rel_sse, cn.cpve
    A, C, T, L = cn.get_factors()
    np.savez(fp+'_'.join([str(r), m])+'.npz', A, C, T, L)
    for v,c in zip(vars,cvars):
        DF = pcorsp(df, C, var=v, covar=[c]+['AGE','FEMALE'])
        tdf = add_cols(DF, params=[cost, rel_sse, cpve, r], pnames=['cost','rel_sse','cpve','rank'])
        RDF.append(tdf)
    pd.concat(RDF, axis=0, ignore_index=True).to_csv(fp+'_'.join([str(r), m])+'.csv')
    return

def cntf_iter(iter, cn, fp, df, vars, cvars, maxiter=1000, alpha=1, ifactors=None, verbosity=1):
    r,m,g = iter
    RDF = []
    cn.fit(rank=r, maxiter=maxiter, alpha=alpha, beta=0, gamma=g, zeta=0, epsl1=1e-8, manifolds=m, cost_fun='CNTF', couple_time=False, verbosity=verbosity, normalize_data=True, initial_factors=ifactors)
    print('done')
    cpve_brain, cpve_taxa = cn.cpve['X'], cn.cpve['Y']
    cost_brain, cost_taxa = cn.cost['X'], cn.cost['Y']
    A, C, V, Tx, Ty, L, S = cn.get_factors()
    np.savez(fp+'_'.join([str(r),''.join(m),str(g)])+'.npz', A, C, V, Tx, Ty, L, S)
    for v,c in zip(vars,cvars):
        DF = pcorsp(df, C, var=v, covar=[c]+['AGE','FEMALE'])
        tdf = add_cols(DF, params=[g, ''.join(m), cpve_brain, cpve_taxa, cost_brain, cost_taxa, r], pnames=['gamma','man','cpve_brain','cpve_taxa','cost_brain','cost_taxa','rank'])
        RDF.append(tdf)
    pd.concat(RDF, axis=0, ignore_index=True).to_csv(fp+'_'.join([str(r),''.join(m),str(g)])+'.csv')
    return

def costcntf_iter(iter, cn, fp, df, vars, cvars, maxiter=1000, alpha=1):
    r,m,g = iter
    RDF = []
    cn.fit(rank=r, maxiter=maxiter, alpha=alpha, beta=0, gamma=g, zeta=0, epsl1=1e-8, manifolds=m, cost_fun='CNTF', couple_time=False, verbosity=1, normalize_data=True)
    cpve_brain, cpve_taxa = cn.cpve['X'], cn.cpve['Y']
    cost_brain, cost_taxa = cn.cost['X'], cn.cost['Y']
    A, C, V, Tx, Ty, L, S = cn.get_factors()
    #np.savez(fp+'_'.join([str(r),''.join(m),str(g)])+'.npz', A, C, V, Tx, Ty, L, S)
    for v,c in zip(vars,cvars):
        DF = pcorsp(df, C, var=v, covar=[c]+['AGE','FEMALE'])
        tdf = add_cols(DF, params=[g, ''.join(m), cpve_brain, cpve_taxa, cost_brain, cost_taxa, r], pnames=['gamma','man','cpve_brain','cpve_taxa','cost_brain','cost_taxa','rank'])
        RDF.append(tdf)
    pd.concat(RDF, axis=0, ignore_index=True).to_csv(fp+'_'.join([str(r),''.join(m),str(g)])+'.csv')
    return

def acntf_iter(iter, cn, fp, df, vars, cvars, maxiter=1000):
    r,m,g = iter
    RDF = []
    cn.fit(rank=r, maxiter=maxiter, alpha=1, beta=g, gamma=0, zeta=0, epsl1=1e-8, manifolds=m, cost_fun='ACNTF', couple_time=False, verbosity=1, normalize_data=True)
    cpve_brain, cpve_taxa = cn.cpve['X'], cn.cpve['Y']
    A, C, V, Tx, Ty, L, S = cn.get_factors()
    np.savez(fp+'_'.join([str(r),''.join(m),str(g)])+'.npz', A, C, V, Tx, Ty, L, S)
    for v,c in zip(vars,cvars):
        DF = pcorsp(df, C, var=v, covar=[c]+['AGE','FEMALE'])
        tdf = add_cols(DF, params=[g, ''.join(m), cpve_brain, cpve_taxa, r], pnames=['gamma','man','cpve_brain','cpve_taxa','rank'])
        RDF.append(tdf)
    pd.concat(RDF, axis=0, ignore_index=True).to_csv(fp+'_'.join([str(r),''.join(m),str(g)])+'.csv')
    return

def acntf_iter2(iter, cn, fp, df, vars, cvars, maxiter=1000):
    r,b,z,m = iter
    RDF = []
    cn.fit(rank=r, maxiter=maxiter, alpha=1, beta=b, gamma=0, zeta=z, epsl1=1e-8, manifolds=m, cost_fun='ACNTF', couple_time=False, verbosity=1, normalize_data=True)
    #cpve_brain, cpve_taxa = cn.cpve['X'], cn.cpve['Y']
    ctot, crelx, crely = cn.cost['total'], cn.rel_cost['X'], cn.rel_cost['Y']
    A, C, V, Tx, Ty, L, S = cn.get_factors()
    np.savez(fp+'_'.join([str(r),''.join(m),str(b),str(z)])+'.npz', A, C, V, Tx, Ty, L, S)
    for v,c in zip(vars,cvars):
        DF = pcorsp(df, C, var=v, covar=[c]+['AGE','FEMALE'])
        tdf = add_cols(DF, params=[b,z, ''.join(m), ctot, crelx, crely, r], pnames=['beta','zeta','man','cost','fit_x', 'fit_y','rank'])
        RDF.append(tdf)
    pd.concat(RDF, axis=0, ignore_index=True).to_csv(fp+'_'.join([str(r),''.join(m),str(b),str(z)])+'.csv')
    return

def pcorsp(Df, factors, var='bmi_chg6', covar=['bv_bmi_calc','AGE','FEMALE'], return_df=False, cca=True):
    df = Df.copy()
    fns = ['Component_'+str(i) for i in range(factors.shape[1])]
    f = pd.DataFrame(factors.copy(), columns=fns)
    df = pd.concat([df.reset_index().copy(), f], axis=1)
    cdf = df.pairwise_corr(columns=[[var]]+[fns], covar=covar, padjust='fdr_bh')
    if cca:
        df['pCCA'] = _cca(factors.copy(), df[var].values.copy(), cov=df[covar].values.copy(), cca_type='pcca')
        ccdf = df.pairwise_corr(columns=[[var]]+[['pCCA']])
        rho, pval = _pcca(factors.copy(), df[var].values.copy(), cov=df[covar].values.copy())
        ccdf['cca_r'] = rho; ccdf['cca_p'] = pval
        cdf = pd.concat([cdf, ccdf], ignore_index=True)
    if return_df:
        return cdf, df
    return cdf

def add_cols(df, params=[], pnames=['beta','gamma','man','cpve_brain','cpve_taxa','rank']):
    for n,c in zip(pnames, params):
        df[n] = [c]*df.shape[0]
    return df

def _cca(U, y, cov, return_coef=False, cca_type=None):
    U, y = U.copy(), y.copy()
    U = mu.get_resid(U, cov)
    y = mu.get_resid(y, cov)
    y -= y.mean()
    for i in range(U.shape[0]):
        U[i,:] -= U[i,:].mean()
    w = U.T @ y
    w /= np.linalg.norm(w)
    if return_coef:
        return w
    return w.T @ U.T

def _pcca(u, Y, cov, nperm=1000):
    U, y = u.copy(), Y.copy()
    U = mu.get_resid(U, cov)
    y = mu.get_resid(y, cov)
    for i in range(U.shape[0]):
        U[i,:] -= U[i,:].mean()
    y -= y.mean()
    w = U.T @ y
    w /= np.linalg.norm(w)
    wU = w.T @ U.T
    r = _corvxv(wU, y)
    rp = _perm_cca(U, y, nperm=nperm)
    p = np.size(np.where(np.abs(rp) > np.abs(r))) / nperm
    return r, p

def _perm_cca(U, y, nperm=1000):
    n = y.shape[0]
    Y = np.zeros((n, nperm))
    for i in range(nperm):
        Y[:,i] = y[np.random.permutation(n)].copy()
    W = U.T @ Y
    for i in range(nperm):
        W[:,i] /= np.linalg.norm(W[:,i])
    WU = W.T @ U.T
    return _cormc(WU.T, Y)

def _corvxmc(X, y):
    cov = np.dot(y.T-y.mean(), X-X.mean(axis=0, keepdims=True)) / (y.shape[0]-1)
    return cov / np.sqrt(np.var(y, ddof=1) * np.var(X, axis=0, ddof=1))

def _cormc(X, Y):
    cov = np.einsum('ij,ij->j',Y-Y.mean(axis=0, keepdims=True), X-X.mean(axis=0, keepdims=True)) / (Y.shape[0]-1)
    return cov / np.sqrt(np.var(Y, axis=0, ddof=1) * np.var(X, axis=0, ddof=1))

def _corvxv(x, y):
    cov = np.dot(y.T-y.mean(), x-x.mean()) / (y.shape[0]-1)
    return cov / np.sqrt(np.var(y, ddof=1) * np.var(x, ddof=1))
