'''
10/05/2023: redo of factorization w/ spearman generated mbm nets
NOTES:
    
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
os.chdir('/Users/paul/Desktop/projects/tfac')
import cntf_utils as cu
from cntf import CTNTF, TNF
import myutils as mu
import pingouin as pg
from itertools import product, chain
from tqdm import tqdm
from tensorly.metrics.factors import congruence_coefficient

# load in data
fpath = '/Users/paul/Desktop/projects/engage/other_data/yichao/'
conn_atlas = pd.read_csv(fpath+'conn_atlas.csv')
eng2_data = pd.read_csv('/Users/paul/Desktop/projects/engage/other_data/lab_meeting/eng2_data_filt.csv')
taxa = np.load('/Users/paul/Desktop/projects/engage/other_data/lab_meeting/taxa_spmn.npy')
brain = np.load('/Users/paul/Desktop/projects/engage/other_data/lab_meeting/brain.npy')
taxa_names = mu.txt2list('/Users/paul/Desktop/projects/engage/other_data/taxa_names.txt')
mpath = '/Users/paul/Desktop/projects/engage/mbm/'
taxa_ids = pd.read_csv(mpath+'taxa_IDs.csv')
vars = ['scl20_chg6', 'bmi_chg6', 'GAD7_Score_chg6']
cvars = ['scl20_score_calc','bv_bmi_calc','GAD7_score']

df = eng2_data.copy()
fp_base = '/Users/paul/Desktop/projects/engage/other_data/r10r/'
maxiter = 1000
iter = (10, ['E','E','E'], 0)
fpvs = ['spv{}_'.format(i) for i in range(5)]

#f = 'spv0_'
#cn = CTNTF(brain, taxa)
#cu.cntf_iter(iter, cn, fp_base+f, df.copy(), vars, cvars, maxiter=maxiter, alpha=0, verbosity=1)

for f in tqdm(fpvs):
    cn = CTNTF(brain, taxa)
    cu.cntf_iter(iter, cn, fp_base+f, df.copy(), vars, cvars, maxiter=maxiter, alpha=0)
    
fpvs = ['aspv{}_'.format(i) for i in range(5)]
for f in tqdm(fpvs):
    cn = CTNTF(brain, taxa)
    cu.acntf_iter(iter, cn, fp_base+f, df.copy(), vars, cvars, maxiter=maxiter)


''' 10/29/23
iterate over rank, beta and zeta
- interesting that ACNTF better captures variance than CNTF in both mbm and brain even when
  parameters that make models differ are set to zero so should have no influence?
  - perhaps because L and S are still variables present to be optimized
  
*** need to center across subject modes (subtract mean)
> https://www.frontiersin.org/articles/10.3389/fnins.2019.00416/full
'''
import itertools as it
df = eng2_data.copy()
fp_base = '/Users/paul/Desktop/projects/engage/other_data/v2/'
maxiter = 1000
ranks = [10]
betas = [0,0.001]
zetas = [0,0.001]
iters = list(it.product(ranks, betas, zetas))
iters = [tuple(list(i) + [['E','E','E']]) for i in iters]

for iter in tqdm(iters):
    fpvs = ['v2x{}_'.format(i) for i in range(3)]
    for f in fpvs:
        cn = CTNTF(brain, taxa)
        cu.acntf_iter2(iter, cn, fp_base+f, df.copy(), vars, cvars, maxiter=maxiter)
        
 
    










#