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
import nilearn
from visbrain.objects import ConnectObj, SceneObj, SourceObj, BrainObj
from visbrain.io.write_image import write_fig_canvas
import pycircos
Garc = pycircos.Garc 
Gcircle = pycircos.Gcircle
from itertools import cycle
from scipy.stats import spearmanr, pearsonr

SUBCOLS = ['studyid', 'AGE', 'ADM_cat', 'educ_rand', 'FEMALE', 'bv_bmi_calc', 'scl20_score_calc', 'GAD7_score', 'mon2_bmi_calc', 'mon2_scl20_calc', 'gad7_score_fu2', 'mon6_bmi_calc', 'mon6_scl20_calc', 'gad7_score_fu6', 'scl20_chg2', 'GAD7_Score_chg2', 'scl20_chg6', 'bmi_chg6', 'GAD7_Score_chg6', 'treat','bmi_chg2']

def factor_barplot(x, labs, ptype='v', topn=None, topn_abs=False, color_dic=None, color_sorter=None, negpos=False, subset=None, pparams=False, figsize=(10,2), fontsize=7, rotation=90, height=1, axmarg=0.01, ha='right', ylabel='',filename=None,ylims=None):
    color = None
    if subset is not None:
        x = np.array([i for i,b in zip(x, subset) if b])
        labs = [i for i,b in zip(labs, subset) if b]
    if topn is not None:
        if topn_abs:
            rix = np.argsort(np.abs(x))[-topn:] # get top n abs(x)
            rlabs = [labs[i] for i in rix]
            rx = x[rix]
            rix = np.argsort(rx) # now sort
            rlabs = [rlabs[i] for i in rix]
            rx = rx[rix]
        else:
            rix = np.argsort(x)
            rlabs = [labs[i] for i in rix]
            rx = x[rix]
            ix = np.arange(len(x))
            ix = np.concatenate((ix[:topn], ix[-topn:]))
            rx = rx[ix]
            rlabs = [rlabs[i] for i in ix]
        if color_dic is not None:
            color = [color_dic[l] for l in rlabs]
    else:
        rix = np.argsort(x)
        rlabs = [labs[i] for i in rix]
        rx = x[rix]
        if color_dic is not None:
            color = [color_dic[l] for l in rlabs]
    if color_sorter is not None:
        sign = ['neg' if i < 0 else 'pos' for i in rx]
        cdf = pd.DataFrame(list(zip(rlabs, color, sign, rx)), columns=['lab','col','sign','rx'])
        if negpos:
            cdfn = cdf[cdf['sign']=='neg'].copy()
            cdfp = cdf[cdf['sign']=='pos'].copy()
            gbn = cdfn.groupby(['col']).agg({'rx':['sum','mean','max','min']})
            gbn.columns = ['sum','mean','max','min']
            gbn.reset_index(inplace=True)
            cdfn = cdfn.merge(gbn, on='col', how='left')
            cdfn.sort_values([color_sorter, 'rx'], inplace=True)
            gbp = cdfp.groupby(['col']).agg({'rx':['sum','mean','max','min']})
            gbp.columns = ['sum','mean','max','min']
            gbp.reset_index(inplace=True)
            cdfp = cdfp.merge(gbp, on='col', how='left')
            cdfp.sort_values([color_sorter, 'rx'], inplace=True)
            cdf = pd.concat([cdfn, cdfp], ignore_index=True)
        else:
            gb = cdf.groupby('col').agg({'rx':['sum','mean','max','min']})
            gb.columns = ['sum','mean','max','min']
            gb.reset_index(inplace=True)
            cdf = cdf.merge(gb, on='col', how='left')
            cdf.sort_values([color_sorter, 'rx'], inplace=True)
        rx, rlabs, color = [cdf[v].to_list() for v in ['rx','lab','col']]
    if ptype == 'v':
        if pparams:
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(np.arange(len(rx)), rx, tick_label=rlabs, color=color, width=height, edgecolor='black')
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize, rotation=rotation, ha=ha)
            ax.margins(x=axmarg)
        elif topn is not None:
            fig, ax = plt.subplots(figsize=(8,2))
            ax.bar(np.arange(len(rx)), rx, tick_label=rlabs, color=color)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=90)
        else:
            fig, ax = plt.subplots(figsize=(15,3))
            ax.bar(np.arange(len(rx)), rx, tick_label=rlabs, color=color)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=90)
            ax.margins(x=axmarg)
    elif ptype == 'h':
        if pparams:
            fig, ax = plt.subplots(figsize=figsize)
            ax.barh(2*np.arange(len(rx)), rx, tick_label=rlabs, height=height, color=color)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize, rotation=rotation)
            ax.margins(y=axmarg, ha=ha)
        elif topn is not None:
            fig, ax = plt.subplots(figsize=(3,6))
            ax.barh(2*np.arange(len(rx)), rx, tick_label=rlabs, height=1.5, color=color)
            plt.setp(ax.get_yticklabels(), fontsize=10)
        else:
            fig, ax = plt.subplots(figsize=(3,20))
            ax.barh(2*np.arange(len(rx)), rx, tick_label=rlabs, height=0.8, color=color)
            plt.setp(ax.get_yticklabels(), fontsize=7)
            ax.margins(y=axmarg)
    plt.ylabel(ylabel, fontsize=12)
    if ylims is not None:
        plt.ylim(*ylims)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return

def node_str(a, asp, asn, blabs, bcdic, net_labs, subset=None, axmarg=0.01, figsize=(9,2), rotation=90, fontsize=9, rm_cblm_bs=False, sortby=['net_order','value'], ascending=[True, False], filename=None):
    bcols = [v for v in bcdic.values()]
    fdf = pd.DataFrame(list(zip(blabs, net_labs, bcols, a, asp, asn)), columns=['node','network','color','factor','pos','neg'])
    if subset is not None:
        fdf = fdf[subset].copy()
    mdf = pd.melt(fdf, id_vars=['node','color','network'], value_vars=['factor','pos','neg'])

    if rm_cblm_bs:
        mdf = mdf[mdf['network']!='cerebellum'].copy()
        mdf = mdf[mdf['network']!='brain stem'].copy()
    gb = mdf.query('`variable`=="factor"').groupby('network').value.apply(lambda c: c.abs().sum()).sort_values()
    corder = gb.reset_index()['network'].to_list()
    nord = {c:i for c,i in zip(corder, np.arange(len(corder)))}
    mdf['net_order'] = [nord[n] for n in mdf['network']]
    subdf = mdf.copy()
    sidx = subdf[subdf['variable']=='factor'].sort_values(sortby, ascending=ascending)['node'].to_list()

    fig, ax = plt.subplots(figsize=figsize)
    ssdf = subdf[subdf['variable']=='pos']
    ssdf = ssdf.set_index('node').loc[sidx].reset_index()
    ax.bar(ssdf['node'], ssdf['value'], color=ssdf['color'], edgecolor='black', width=1)
    ax.margins(x=axmarg)
    ssdf = subdf[subdf['variable']=='neg']
    ssdf = ssdf.set_index('node').loc[sidx].reset_index()
    ax.bar(ssdf['node'], ssdf['value'], color=ssdf['color'], edgecolor='black', width=1)
    ax.margins(x=axmarg)
    plt.xticks(rotation=rotation, fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return

def plot_time(tx, labs=['baseline','2 months'], title='brain', figsize=(1,2), axmarg=0.05, rotation=60, fontsize=12, filename=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labs, tx, color='darkred', edgecolor='black', width=1)
    ax.margins(x=axmarg)
    plt.title(title)
    plt.xticks(rotation=rotation, fontsize=fontsize, ha='right', va='center', rotation_mode='anchor')
    plt.xlabel('time factor')
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return

def corplot(c, subdf, var='GAD7_Score_chg6', auto_ctrl_baseline_sx=True, covar=['AGE','FEMALE'], flip_dv=False, figsize=(4,3), title='', xlabel='subject factor', ylabel = 'GAD7 reduction\n(baseline - 6mo)', filename=None, add_spearman=False):
    df = subdf.copy()
    if flip_dv:
        df[var] = -df[var].values
    df['factor'] = c
    vars = ['factor'] + [var]
    if covar is not None:
        if auto_ctrl_baseline_sx:
            if 'GAD7' in var:
                covar += ['GAD7_score']
            if 'scl20' in var:
                covar += ['scl20_score_calc']
            if 'bmi' in var:
                covar += ['bv_bmi_calc']
        df = mu.getResiduals(df, vars, covar)
    x, y = df['factor'].values, df[var].values
    if add_spearman:
        print(spearmanr(x,y))
        print(pearsonr(x,y))
        spearr,spearp = spearmanr(x,y)
        title+='\nSpearman: rho={:.3f} p={:.3f}'.format(spearr, spearp)
    else:
        print(spearmanr(x,y))
        print(pearsonr(x,y))
    ixc = np.where(df['treat']==0)[0]
    ixt = np.where(df['treat']==1)[0]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x[ixc], y[ixc], c='darkblue', label='control')
    ax.scatter(x[ixt], y[ixt], c='darkred', label='treatment')
    ax.legend()
    # add regression line
    m, b = np.polyfit(x, y, 1)
    xlims = ax.get_xlim()
    X_plot = np.linspace(xlims[0], xlims[1], 100)
    ax.plot(X_plot, m*X_plot + b, '-', c='black')
    plt.xlim(xlims)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return

def signed_strength(W):
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Spos = np.sum(W * (W > 0), axis=0)  # positive strengths
    Sneg = np.sum(W * (W < 0), axis=0) # negative strengths
    return Spos, Sneg


def get_node_colors(net_labs, return_dict=False, colors=None):
    if colors is not None:
        pass
    else:
        colors = [plt.cm.tab10(i) for i in range(10)]
        t10ix = [0,1,2,3,4,6,8,9,5,7]
        colors = [colors[c] for c in t10ix]
        colors = [mpl.colors.rgb2hex(c[:-1]) for c in colors]
    unq = pd.factorize(net_labs)[1]
    coldic = {u:colors[i] for i,u in enumerate(unq)}
    node_colors = [coldic[n] for n in net_labs]
    if return_dict:
        return node_colors, coldic
    return node_colors

def get_node_mask(edges, select, inverted=False):
    tmp = np.zeros(edges.shape)
    tmp[select] = 1
    rois = np.where(tmp.sum(0) > 0)[0]
    if inverted: # True on nodes to keep
        node_mask = [False] * edges.shape[0]
        for r in rois:
            node_mask[r] = True
    else: # True on nodes to mask
        node_mask = [True] * edges.shape[0]
        for r in rois:
            node_mask[r] = False
    return node_mask

def visbrain_plot(edges, select, nodes, node_colors, fn_base=None, braintype='B2', cmap_name='seismic'):
    # node mask
    node_mask = get_node_mask(edges, select, inverted=False)
    # node sizes
    X = edges.copy()
    xm = edge_select(X, n=400, invert_mask=True)
    X[xm] = 0
    node_size = np.abs(X).sum(0)
    # shared kwargs
    if fn_base is not None:
        ckw = {
        'select':select,
        'line_width':60.,
        'cmap':cmap_name,
        'color_by':'strength',
        'dynamic':(.1, 1.),
        'dynamic_orientation':'center'
        }
        skw = {
        'data':node_size,
        'color':node_colors,
        'radius_min':15.,
        'radius_max':35.,
        'mask':node_mask,
        'mask_radius':0.
        }
    else:
        ckw = {
        'select':select,
        'line_width':5.,
        'cmap':cmap_name,
        'color_by':'strength',
        'dynamic':(.1, 1.),
        'dynamic_orientation':'center'
        }
        skw = {
        'data':node_size,
        'color':node_colors,
        'radius_min':5.,
        'radius_max':15.,
        'mask':node_mask,
        'mask_radius':0.
        }
    sc = SceneObj(bgcolor='white')
    # left
    c_obj = ConnectObj('con', nodes, edges, **ckw)
    s_obj = SourceObj('sources', nodes, **skw)
    sc.add_to_subplot(c_obj)
    sc.add_to_subplot(s_obj)
    sc.add_to_subplot(BrainObj(braintype), use_this_cam=True, rotate='left')
    # top
    c_obj = ConnectObj('con', nodes, edges, **ckw)
    s_obj = SourceObj('sources', nodes, **skw)
    sc.add_to_subplot(c_obj, row=0, col=1)
    sc.add_to_subplot(s_obj, row=0, col=1)
    sc.add_to_subplot(BrainObj(braintype), use_this_cam=True, rotate='top', row=0, col=1)
    # right
    c_obj = ConnectObj('con', nodes, edges, **ckw)
    s_obj = SourceObj('sources', nodes, **skw)
    sc.add_to_subplot(c_obj, row=0, col=2)
    sc.add_to_subplot(s_obj, row=0, col=2)
    sc.add_to_subplot(BrainObj(braintype), use_this_cam=True, rotate='right', row=0, col=2)
    sc.preview()
    if fn_base is not None:
        mpl_plotsc(sc, filename=fn_base + '.png')
        colorbar_plot(cmap_name, X.min(), X.max(), filename=fn_base + '_cbar.png')
    return

def edge_select(X, n=200, invert_mask=False):
    Xc = X.copy()
    np.fill_diagonal(Xc, 0)
    Xut = np.triu(Xc)
    xf = np.abs(Xut.flatten())
    val = np.sort(xf)[-(n+1)]
    if invert_mask:
        return np.abs(Xc) <= val
    return np.abs(Xc) > val

def colorbar_plot(cmap_name, vmin, vmax, filename=None, label='edge value', figsize=(0.3, 3), labelsize=12, ticksize=12):
    fig, ax = plt.subplots(figsize=figsize)
    if vmin >= 0:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    elif vmax <= 0:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='vertical')
    cbar.set_label(label, size=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return

def plot_legend(cdic, filename=None, **kwargs):
    handles = [mpl.patches.Patch(color=v, label=k) for k,v in cdic.items()]
    fig, ax = plt.subplots()
    ax.legend(handles=handles, loc='center', **kwargs)
    plt.gca().set_axis_off()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return

def mpl_plotsc(sceneobj, filename=None):
    img = write_fig_canvas(None, sceneobj.canvas, widget=sceneobj.canvas.central_widget, print_size=(6, 4), dpi=300, unit='inch')[..., 0:-1]
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)
    ax.imshow(img, interpolation='bicubic')
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    plt.axis('off')
    fig.tight_layout(pad=0., h_pad=0., w_pad=0.)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()
    return

# circos
def plot_circos(edges, imask, taxap, tcols, fn_base=None):
    edges[imask] = 0
    edges[np.tril_indices(edges.shape[0])] = 0
    ind = np.where(edges!=0)
    cdc, rec_width, tn_fac = _get_taxa_colors(taxap, tcols, rec_tn=True)
    circle = Gcircle()
    for id, w in zip(tn_fac[1], rec_width):
        arc = Garc(arc_id=id, size=w, interspace=3, raxis_range=(500,550), labelposition=60, label_visible=True, labels_perpendicular=True, facecolor=cdc[id])
        circle.add_garc(arc)
    circle.set_garcs()#; plt.show()

    gen = taxap['genus'].to_list()
    tn = taxap['taxa_name'].to_list()
    dlist = []
    for i,j in zip(*ind):
        d = {}
        d['node1'] = gen[i]
        d['node1g'] = tn[i]
        d['node2'] = gen[j]
        d['node2g'] = tn[j]
        d['edge_val'] = edges[i,j]
        dlist.append(d)
    cdf = pd.DataFrame(dlist)
    cdf['mag'] = ['pos' if i > 0 else 'neg' for i in cdf['edge_val']]
    cdf['edge'] = ['---'.join([i,j]) for i,j in zip(cdf['node1'], cdf['node2'])]
    cdf['idx1'] = [int(n.split('.')[-1]) if '.' in n else 0 for n in cdf['node1g']]
    cdf['idx2'] = [int(n.split('.')[-1]) if '.' in n else 0 for n in cdf['node2g']]
    cdf['start1'] = [0 + i for i in cdf['idx1']]
    cdf['end1'] = [1 + i for i in cdf['idx1']]
    cdf['start2'] = [0 + i for i in cdf['idx2']]
    cdf['end2'] = [1 + i for i in cdf['idx2']]
    cdf.sort_values(by='edge_val', key=abs, inplace=True)

    vmin, vmax = cdf['edge_val'].min(), cdf['edge_val'].max()
    if vmin >= 0:
        cdf['color'] = [monoseq2hex(v, cmap_name='Reds', vmin=0, vmax=vmax) for v in cdf['edge_val']]
    elif vmax <= 0:
        cdf['color'] = [monoseq2hex(v, cmap_name='Blues_r', vmin=vmin, vmax=0) for v in cdf['edge_val']]
    else:
        cdf['color'] = [diverg2hex(v, vmin=vmin, vmax=vmax) for v in cdf['edge_val']]

    for i in range(cdf.shape[0]):
        row = cdf.iloc[i,:]
        source = (row['node1'], row['start1'], row['end1'], 500)
        destination = (row['node2'], row['start2'], row['end2'], 500)
        circle.chord_plot(source, destination, facecolor=row['color'])

    if fn_base is not None:
        circle.figure.savefig(fn_base + '.png', dpi=300, bbox_inches='tight')
        if vmin >= 0:
            colorbar_plot('Reds', 0, vmax, filename=fn_base + '_cbar.png')
        elif vmax <= 0:
            colorbar_plot('Blues_r', vmin, 0, filename=fn_base + '_cbar.png')
        else:
            colorbar_plot('RdBu_r', vmin, vmax, filename=fn_base + '_cbar.png')
    circle.figure
    return

def _get_taxa_colors(taxap, tcols, rec_tn=False):
    # rectangles
    tn_fac = pd.factorize(taxap['genus'])
    rec_width = np.unique(tn_fac[0], return_counts=True)[1] + 1
    ufam = pd.factorize(taxap['Family'])[1]
    f2c = {f:tcols[i] for i,f in enumerate(ufam)}
    # idx for one of each taxa
    uix = [np.where(tn_fac[0]==i)[0][0] for i in np.unique(tn_fac[0])]
    gen = taxap['genus'].to_list()
    fam = taxap['Family'].to_list()
    g2f = {gen[i]:fam[i] for i in uix}
    coldic = {i:f2c[g2f[i]] for i in tn_fac[1]}
    if rec_tn:
        return coldic, rec_width, tn_fac
    return coldic

def get_taxa_colors(taxap, tcols, return_dic='f2c'):
    # rectangles
    tn_fac = pd.factorize(taxap['genus'])
    rec_width = np.unique(tn_fac[0], return_counts=True)[1] + 1
    ufam = pd.factorize(taxap['Family'])[1]
    f2c = {f:tcols[i] for i,f in enumerate(ufam)}
    if return_dic=='f2c':
        return f2c
    # idx for one of each taxa
    uix = [np.where(tn_fac[0]==i)[0][0] for i in np.unique(tn_fac[0])]
    gen = taxap['genus'].to_list()
    fam = taxap['Family'].to_list()
    g2f = {gen[i]:fam[i] for i in uix}
    coldic = {i:f2c[g2f[i]] for i in tn_fac[1]}
    return coldic

def format_taxa_circos(taxa_ids, taxa_names):
    taxa_ids['Genus'] = taxa_ids['Genus'].fillna('NA')
    ix = pd.factorize(taxa_ids['Genus'])[0]
    ix = [np.where(ix==i)[0][0] for i in np.unique(ix)]
    tids = taxa_ids.iloc[ix,:]
    tnu = [t.split('.')[0] for t in taxa_names]
    taxap = pd.DataFrame(list(zip(np.arange(len(tnu)), taxa_names, tnu)), columns=['idx', 'taxa_name', 'genus'])
    return taxap.merge(tids, left_on='genus', right_on='Genus', how='left')

# misc
def get_tabcols(shuffle=False, indx=None):
    if indx is not None:
        pass
    else:
        indx = np.linspace(0,16,5).astype(int)
    cm1 = plt.get_cmap('tab20b').colors
    cm2 = plt.get_cmap('tab20c').colors
    colors = []
    for c in [cm1, cm2]:
        for i in indx:
            colors.append(c[i])
    indx += 2
    for c in [cm1, cm2]:
        for i in indx:
            colors.append(c[i])
    hcolors = [mpl.colors.to_hex(c, keep_alpha=False) for c in colors]
    if shuffle:
        rng = np.random.default_rng()
        ix = rng.permutation(len(hcolors))
        return [hcolors[i] for i in ix]
    return hcolors

def diverg2hex(value, cmap_name='RdBu_r', vmin=0, vmax=1):
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm(value))[:3]
    color = mpl.colors.rgb2hex(rgb)
    return color

def monoseq2hex(value, cmap_name='Reds', vmin=0, vmax=1):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm(value))[:3]
    color = mpl.colors.rgb2hex(rgb)
    return color

def circos_cols():
    tcols = plt.get_cmap('Paired').colors
    odds = list(range(1, 12, 2))
    cidx = []
    for i in odds:
        cidx.append(i); cidx.append(i-1)
    tcols = [tcols[i] for i in cidx]
    tcols = [mpl.colors.to_hex(c, keep_alpha=False) for c in tcols]
    return tcols

def netheatmap(Xa, net_labs):
    nfac = pd.factorize(net_labs)
    unets = np.unique(nfac[0])
    #rv = np.concatenate([np.where(nfac[0]==i)[0] for i in unets])
    rv = [np.where(nfac[0]==i)[0] for i in unets]
    #rXa = Xa[np.ix_(rv,rv)].copy()
    pxnets = np.zeros((len(unets), len(unets)))
    nxnets = np.zeros((len(unets), len(unets)))
    xp = Xa.copy(); xp[xp < 0] = 0
    xn = Xa.copy(); xn[xn > 0] = 0
    for i,ii in enumerate(rv):
        for j,jj in enumerate(rv):
            idx = np.ix_(ii,jj)
            if i >= j:
                pxnets[i,j] = xp[idx].sum()
                nxnets[i,j] = xn[idx].sum()
    vmax = np.max((np.abs(nxnets.min()), np.abs(pxnets.max())))
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sns.heatmap(nxnets, cmap='RdBu_r', norm=norm, xticklabels=nfac[1], yticklabels=nfac[1])
    sns.heatmap(pxnets, cmap='RdBu_r', norm=norm, xticklabels=nfac[1], yticklabels=nfac[1])
    return
#
