import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
from glob import glob
import os
os.chdir('/Users/paul/Desktop/projects/tfac')
import myutils as mu
from itertools import product, chain
from cntf_plot import factor_barplot, edge_select, visbrain_plot, get_node_colors, plot_legend, get_tabcols, diverg2hex, format_taxa_circos, circos_cols, plot_circos, colorbar_plot, get_taxa_colors, get_node_mask
from visbrain.objects import ConnectObj, SceneObj, SourceObj, BrainObj

# load in data
fpath = '/Users/paul/Desktop/projects/engage/other_data/yichao/'
conn_atlas = pd.read_csv('/Users/paul/Desktop/projects/atlas_merge/conn_atlas_yeo7nets.csv')
xyz = np.load(fpath+'conn_hoa_xyz.npy')
fp_res = '/Users/paul/Desktop/projects/tfac/eng2_results/'

''' NOTES:
2/22/22:

'''

fp = '/Users/paul/Desktop/projects/engage/other_data/r10r/'
fname = fp + 'v1_10_EEE_0.npz'
F = np.load(fname)
A, C, V, Tx, Ty, L, S = [F[f] for f in F.files]
factor_index = 1

# brain plot
a = A[:,factor_index]
edges = np.einsum('i,j->ij',a,a)
select = edge_select(edges, n=400)
net_labs = conn_atlas['yeo_nets'].to_list()
roi_labs = conn_atlas['roiLongName'].to_list()
fn_base = fp_res + 'brainnet1'

node_colors, coldic = get_node_colors(net_labs, colors=None, return_dict=True)
visbrain_plot(edges, select, xyz, node_colors, fn_base=fn_base,  braintype='B2', cmap_name='RdBu_r')
node_mask = get_node_mask(edges, select, inverted=True)
nlabs = [net_labs[i] for i,b in enumerate(node_mask) if b]
ul = pd.factorize(nlabs)[1]
ncoldic = {l:coldic[l] for l in ul}
plot_legend(ncoldic, filename=fn_base + '_legend.png', ncol=4)


#
