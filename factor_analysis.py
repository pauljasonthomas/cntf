#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10/06/2023: analysis of factors for spearman redo
NOTES:
    
"""

import numpy as np
import pandas as pd

fpath = '/Users/paul/Desktop/projects/engage/other_data/yichao/'
conn_atlas = pd.read_csv('/Users/paul/Desktop/projects/atlas_merge/conn_atlas_yeo7nets.csv')
lname = conn_atlas['roiLongName'].to_list()
short = conn_atlas['roiName'].to_list()

ROI = 'SPL_l'
print(lname[short.index(ROI)])


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
from glob import glob
import os
os.chdir('/Users/paul/Desktop/projects/tfac')
import myutils as mu
#spath = '/opt/anaconda3/lib/python3.7/site-packages/'
import sys
#if spath in sys.path: sys.path.remove(spath)
from cntf_plot import factor_barplot, edge_select, visbrain_plot, get_node_colors, plot_legend, get_tabcols, diverg2hex, format_taxa_circos, circos_cols, plot_circos, colorbar_plot, get_taxa_colors, signed_strength, get_node_mask, node_str, corplot, plot_time, SUBCOLS
from visbrain.objects import ConnectObj, SceneObj, SourceObj, BrainObj
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import sklearn

fpath = '/Users/paul/Desktop/projects/engage/other_data/yichao/'
conn_atlas = pd.read_csv('/Users/paul/Desktop/projects/atlas_merge/conn_atlas_yeo7nets.csv')
xyz = np.load(fpath+'conn_hoa_xyz.npy')
eng2_data = pd.read_csv('/Users/paul/Desktop/projects/engage/other_data/lab_meeting/eng2_data_filt.csv')
subdf = eng2_data[SUBCOLS].copy()
taxa_names = mu.txt2list('/Users/paul/Desktop/projects/engage/other_data/taxa_names.txt')
mpath = '/Users/paul/Desktop/projects/engage/mbm/'
taxa_ids = pd.read_csv(mpath+'taxa_IDs.csv')
vars = ['scl20_chg6', 'bmi_chg6', 'GAD7_Score_chg6']
cvars = ['scl20_score_calc','bv_bmi_calc','GAD7_score']
fp_res = '/Users/paul/Desktop/projects/tfac/eng2_results/final/'
# brain
net_labs = conn_atlas['yeo_nets'].to_list()
roi_labs = conn_atlas['roiLongName'].to_list()
blabs = conn_atlas['roiName'].to_list()
node_colors, bcoldic = get_node_colors(net_labs, colors=None, return_dict=True)
bcdic = {l:c for l,c in zip(blabs, node_colors)}
# taxa
taxap = format_taxa_circos(taxa_ids, taxa_names)
taxap = taxap[~taxap['genus'].str.contains('Unnamed')].copy()
taxap.sort_values('Family', inplace=True)
# custom colors
tcols = circos_cols()
tcoldic = get_taxa_colors(taxap, tcols)
tlabs = taxap['taxa_name'].to_list()
fam_labs = taxap['Family'].to_list()
tcoldic2 = get_taxa_colors(taxap, tcols, return_dic='cdic')
tclist = [tcoldic2[l] for l in taxap['genus']]
tcdic = {l:c for l,c in zip(tlabs, tclist)}
def lda_net(L,A,T,W):
    n, r = A.shape
    t = T.shape[0]
    Nw = np.zeros((n,n,t))
    for l,a,t,w in zip(L,A.T,T.T,W):
        Nw += np.abs(l) * w * np.einsum('i,j,k->ijk',a,a,t)
    return Nw


#taxap[taxap['Phylum']=='Firmicutes']

''' NOTES:
spv3 - GAD7+

'''

# brain abbreviation translator:
lname = conn_atlas['roiLongName'].to_list()
short = conn_atlas['roiName'].to_list()
ROI = 'SPL_l'
print(lname[short.index(ROI)])

fp = '/Users/paul/Desktop/projects/engage/other_data/r10r/'
# load in data
fversion = 'aspv3'
basename = fversion + '_10_EEE_0'
vdf = pd.read_csv(fp+basename+'.csv')
vdf.query('`p-unc`<0.05').sort_values('p-unc')

fname = fp + basename+'.npz'
F = np.load(fname)
A, C, V, Tx, Ty, L, S = [F[f] for f in F.files]




#ACNTF redo
from glob import glob
import pandas as pd
def read_data(fp, flist=['v2x0','v2x1','v2x2'], frx='_*csv'):
    RDF = []
    for v in flist:
        files = glob(fp+v+frx)
        rdf = [pd.read_csv(f, index_col=[0]) for f in files]
        rdf = pd.concat(rdf, axis=0, ignore_index=True)
        rdf['version'] = [v]*rdf.shape[0]
        if len(flist) == 1:
            return rdf
        RDF.append(rdf)
    return pd.concat(RDF, axis=0, ignore_index=True)

df = read_data('/Users/paul/Desktop/projects/engage/other_data/v2/')
df = df.query('`p-corr`<0.05').sort_values('p-corr')


fp = '/Users/paul/Desktop/projects/engage/other_data/v2/'
flist = ['v2x{}_10'.format(i) for i in range(3)]
df = read_data(fp, flist=flist)
df = df.query('`p-unc`<0.05').sort_values('p-unc')
df.dropna(subset=['covar'], inplace=True)

#fp = '/Users/paul/Desktop/projects/engage/other_data/r10r/'
fversion = 'v2x0'
basename = fversion + '_10_EEE_0_0.001'
fname = fp + basename+'.npz'
F = np.load(fname)
A, C, V, Tx, Ty, L, S = [F[f] for f in F.files]
print(L[6], S[6])
plt.plot(L,c='r'); plt.plot(S,c='b'); plt.show()

fversion = 'v2x2'
basename = fversion + '_10_EEE_0.001_0'
fname = fp + basename+'.npz'
F = np.load(fname)
A, C, V, Tx, Ty, L, S = [F[f] for f in F.files]
print(L[8], S[8])
plt.plot(L,c='r'); plt.plot(S,c='b'); plt.show()

fversion = 'v2x1'
basename = fversion + '_10_EEE_0.001_0.001'
fname = fp + basename+'.npz'
F = np.load(fname)
A, C, V, Tx, Ty, L, S = [F[f] for f in F.files]
print(L[9], S[9])
plt.plot(L,c='r'); plt.plot(S,c='b'); plt.show()

fversion = 'aspv1'
basename = fversion + '_10_EEE_0'
fname = fp + basename+'.npz'
F = np.load(fname)
A, C, V, Tx, Ty, L1, S1 = [F[f] for f in F.files]

fversion = 'aspv2'
basename = fversion + '_10_EEE_0'
fname = fp + basename+'.npz'
F = np.load(fname)
A, C, V, Tx, Ty, L2, S2 = [F[f] for f in F.files]

#1 2 0 - 2 8 1
import matplotlib.pyplot as plt
#plt.plot(L0,c='r'); plt.plot(S0,c='b'); plt.show()
#print(L0[1], S0[1])
print(L1[2], S1[2])
print(L2[8], S2[8])



fname = fp + basename+'.npz'
F = np.load(fname)
A, C, V, Tx, Ty, L, S = [F[f] for f in F.files]

############
# factor 1 #
############
fp_res = '/Users/paul/Desktop/projects/tfac/new_1023/res/'
factor_index = 3
df_idx = 23
stats = [vdf.iloc[df_idx,:][s] for s in ['r','p-unc','p-corr']]; stats[0] *= -1
# subject
c = C[:,factor_index]
tx = Tx[:,factor_index]; ty = Ty[:,factor_index]
timepct_brain = (tx[1]-tx[0])/tx[0]
timepct_mbm = (ty[1]-ty[0])/ty[0]
filename = fp_res+'subjectfac1.png'
var = vdf.iloc[df_idx,:]['X'] #'GAD7_Score_chg6'
title = 'Component 1\nPearson: rho={:.3f} p={:.3f} q={:.3f}'.format(*stats)
ylabel = 'GAD7 reduction\n(baseline - 6mo)'
corplot(c, subdf, var=var, flip_dv=True, figsize=(4,3), title=title, 
        ylabel=ylabel, filename=filename, add_spearman=True)














#


